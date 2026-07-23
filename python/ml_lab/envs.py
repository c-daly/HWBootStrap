"""Vector-worker construction, deterministic episodes, and opponent routing."""

from __future__ import annotations

import csv
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env

from hexwars_gym import HexWarsEnv
from selfplay_env import SelfPlayEnv

from .contracts import ADAPTIVE_MONITOR_HEADER, EnvironmentContract, MONITOR_HEADER, RunConfig
from .controllers import ControllerBinding, ControllerResolver, snapshot_opponents
from .scenarios import resolve_scenario, validate_handshake


@dataclass(frozen=True)
class EpisodeAssignment:
    """The deterministic identity and learner seat of one worker episode."""

    worker_id: int
    episode_index: int
    seed: int
    learner_seat: int


class WorkerSchedule:
    """Deterministic, disjoint episode seeds plus a per-worker seat schedule."""

    def __init__(
        self,
        *,
        base_seed: int,
        worker_index: int,
        worker_count: int,
        learner_seat: str = "alternating",
    ) -> None:
        if worker_count <= 0 or not 0 <= worker_index < worker_count:
            raise ValueError("worker index must be inside a positive worker count")
        if learner_seat not in {"alternating", "0", "1"}:
            raise ValueError("learner seat must be 'alternating', '0', or '1'")
        self.base_seed = base_seed
        self.worker_index = worker_index
        self.worker_count = worker_count
        self.learner_seat = learner_seat
        self.episode_index = 0

    def next_episode(self) -> EpisodeAssignment:
        episode = self.episode_index
        self.episode_index += 1
        seed = self.base_seed + self.worker_index + episode * self.worker_count
        if self.learner_seat == "alternating":
            seat = (self.worker_index + episode) % 2
        else:
            seat = int(self.learner_seat)
        return EpisodeAssignment(
            worker_id=self.worker_index,
            episode_index=episode,
            seed=seed,
            learner_seat=seat,
        )


class ScheduledEnvironment(gym.Env):
    """Own one server-backed env at a time and rebuild it only when its seat changes."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        schedule: WorkerSchedule,
        builder: Callable[[int, int], gym.Env],
    ) -> None:
        super().__init__()
        self._schedule = schedule
        self._builder = builder
        self._pending = schedule.next_episode()
        self.current_assignment = self._pending
        self._env = builder(self._pending.learner_seat, self._pending.seed)
        self._seat = self._pending.learner_seat
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.contract = _environment_contract(self._env)
        self.spaces_info = dict(getattr(self._env, "spaces_info", {}))

    def _ensure_seat(self, seat: int, seed: int) -> None:
        if seat == self._seat:
            return
        previous = self._env
        replacement = self._builder(seat, seed)
        replacement_contract = _environment_contract(replacement)
        if replacement_contract != self.contract:
            replacement.close()
            raise ValueError("worker environment contract changed with learner seat")
        self._env = replacement
        self._seat = seat
        previous.close()

    def reset(self, *, seed=None, options=None):
        del seed
        if self._pending is None:
            assignment = self._schedule.next_episode()
        else:
            assignment = self._pending
            self._pending = None
        self._ensure_seat(assignment.learner_seat, assignment.seed)
        self.current_assignment = assignment
        return self._env.reset(seed=assignment.seed, options=options)

    def step(self, action):
        return self._env.step(action)

    def action_masks(self) -> np.ndarray:
        return np.asarray(self._env.action_masks(), dtype=bool)

    def close(self) -> None:
        self._env.close()


class EpisodeMonitor(gym.Wrapper):
    """Append one authoritative per-episode row without depending on SB3 Monitor files."""

    def __init__(
        self,
        env: gym.Env,
        path: Path,
        lock: threading.Lock,
        *,
        worker_id: int = 0,
        adaptive_path: Path | None = None,
    ) -> None:
        super().__init__(env)
        self.contract = _environment_contract(env)
        self.spaces_info = dict(getattr(env, "spaces_info", {}))
        self._path = Path(path)
        self._lock = lock
        self._worker_id = worker_id
        self._adaptive_path = (
            Path(adaptive_path) if adaptive_path is not None
            else self._path.parent / "adaptive_episodes.csv"
        )
        self._episode_number = 0
        self._started = time.monotonic()
        self._reward = 0.0
        self._length = 0
        self._diagnostics: dict[str, Any] = {}
        self._assignment: EpisodeAssignment | None = None

    def reset(self, **kwargs):
        self._reward = 0.0
        self._length = 0
        self._diagnostics = {}
        observation, info = self.env.reset(**kwargs)
        assignment = getattr(self.env, "current_assignment", None)
        self._assignment = assignment if isinstance(assignment, EpisodeAssignment) else None
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._reward += float(reward)
        self._length += 1
        diagnostics = info.get("diagnostics") if isinstance(info, Mapping) else None
        if isinstance(diagnostics, Mapping):
            self._diagnostics = dict(diagnostics)
        if terminated or truncated:
            self._episode_number += 1
            elapsed = time.monotonic() - self._started
            self._append_episode(elapsed)
            if self.contract.version == "adaptive-v1":
                self._append_adaptive_episode()
            info = dict(info)
            episode_info = {
                "r": self._reward,
                "l": self._length,
                "t": elapsed,
                "worker_id": self._worker_id,
                "episode_number": self._episode_number,
            }
            if self._assignment is not None:
                episode_info.update(
                    {
                        "worker_id": self._assignment.worker_id,
                        "episode_index": self._assignment.episode_index,
                        "episode_seed": self._assignment.seed,
                        "learner_seat": self._assignment.learner_seat,
                    }
                )
            info["episode"] = episode_info
        return observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env.action_masks(), dtype=bool)

    def _append_episode(self, elapsed: float) -> None:
        with self._lock:
            if not self._path.exists():
                self._path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with self._path.open("x", newline="", encoding="utf-8") as stream:
                        csv.writer(stream).writerow(MONITOR_HEADER)
                except FileExistsError:
                    pass
            with self._path.open("a", newline="", encoding="utf-8") as stream:
                assignment = self._assignment
                csv.writer(stream).writerow(
                    [
                        assignment.worker_id if assignment is not None else self._worker_id,
                        assignment.episode_index if assignment is not None else self._episode_number - 1,
                        assignment.seed if assignment is not None else "",
                        assignment.learner_seat if assignment is not None else "",
                        self._reward,
                        self._length,
                        elapsed,
                    ]
                )

    def _append_adaptive_episode(self) -> None:
        path = self._adaptive_path
        values = self._diagnostics
        row = [
            f"{self._worker_id}:{self._episode_number}",
            int(values.get("design_count", 0)),
            int(values.get("distinct_custom_templates_deployed", 0)),
            bool(values.get("deployment_completed", False)),
            int(values.get("invalid_sequences", 0)),
            int(values.get("pregame_decisions", 0)),
        ]
        with self._lock:
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with path.open("x", newline="", encoding="utf-8") as stream:
                        csv.writer(stream).writerow(ADAPTIVE_MONITOR_HEADER)
                except FileExistsError:
                    pass
            with path.open("a", newline="", encoding="utf-8") as stream:
                csv.writer(stream).writerow(row)


class MaskingDummyVecEnv(DummyVecEnv):
    """SB3 vector env whose masks remain directly available to callers and Contrib."""

    def __init__(self, env_fns) -> None:
        super().__init__(env_fns)
        contracts = self.get_attr("contract")
        if any(contract != contracts[0] for contract in contracts[1:]):
            self.close()
            raise ValueError("all vector workers must use the same environment contract")
        self.contract = contracts[0]
        self.spaces_info = dict(self.get_attr("spaces_info")[0])

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env_method("action_masks"), dtype=bool)


class MaskingSubprocVecEnv(SubprocVecEnv):
    """Spawned vector workers with the same direct mask and contract surface."""

    def __init__(self, env_fns, *, start_method: str = "spawn") -> None:
        self.waiting = False
        self.closed = False
        context = mp.get_context(start_method)
        self.remotes, self.work_remotes = zip(
            *[context.Pipe() for _ in range(len(env_fns))], strict=True
        )
        self.processes = []
        try:
            for work_remote, remote, env_fn in zip(
                self.work_remotes, self.remotes, env_fns, strict=True
            ):
                process = context.Process(
                    target=_exception_safe_worker,
                    args=(work_remote, remote, CloudpickleWrapper(env_fn)),
                    daemon=True,
                )
                process.start()
                self.processes.append(process)
                work_remote.close()

            self.remotes[0].send(("get_spaces", None))
            observation_space, action_space = self.remotes[0].recv()
            VecEnv.__init__(self, len(env_fns), observation_space, action_space)

            contracts = self.get_attr("contract")
            if any(contract != contracts[0] for contract in contracts[1:]):
                raise ValueError("all vector workers must use the same environment contract")
            self.contract = contracts[0]
            self.spaces_info = dict(self.get_attr("spaces_info")[0])
        except BaseException:
            self._shutdown_workers()
            raise

    def _pipe_guard(self, operation):
        try:
            return operation()
        except (EOFError, BrokenPipeError, ConnectionError, OSError) as error:
            self._shutdown_workers()
            if isinstance(error, EOFError):
                raise
            raise EOFError("spawned worker connection closed") from error

    def _shutdown_workers(self) -> None:
        if self.closed:
            return
        remotes = tuple(getattr(self, "remotes", ()))
        work_remotes = tuple(getattr(self, "work_remotes", ()))
        processes = list(getattr(self, "processes", ()))
        for process in processes:
            process.join(timeout=0.05)
        for remote, process in zip(remotes, processes):
            if process.is_alive():
                try:
                    remote.send(("close", None))
                except (EOFError, BrokenPipeError, ConnectionError, OSError):
                    pass
        for process in processes:
            process.join(timeout=2.0)
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=2.0)
        for remote in remotes:
            try:
                remote.close()
            except OSError:
                pass
        for remote in work_remotes:
            try:
                remote.close()
            except OSError:
                pass
        self.remotes = ()
        self.work_remotes = ()
        self.processes = []
        self.waiting = False
        self.closed = True

    def step_async(self, actions: np.ndarray) -> None:
        return self._pipe_guard(lambda: super(MaskingSubprocVecEnv, self).step_async(actions))

    def step_wait(self):
        return self._pipe_guard(lambda: super(MaskingSubprocVecEnv, self).step_wait())

    def reset(self):
        return self._pipe_guard(lambda: super(MaskingSubprocVecEnv, self).reset())

    def get_images(self):
        return self._pipe_guard(lambda: super(MaskingSubprocVecEnv, self).get_images())

    def has_attr(self, attr_name: str) -> bool:
        return self._pipe_guard(
            lambda: super(MaskingSubprocVecEnv, self).has_attr(attr_name)
        )

    def get_attr(self, attr_name: str, indices=None) -> list[Any]:
        return self._pipe_guard(
            lambda: super(MaskingSubprocVecEnv, self).get_attr(attr_name, indices)
        )

    def set_attr(self, attr_name: str, value: Any, indices=None) -> None:
        return self._pipe_guard(
            lambda: super(MaskingSubprocVecEnv, self).set_attr(attr_name, value, indices)
        )

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        return self._pipe_guard(
            lambda: super(MaskingSubprocVecEnv, self).env_method(
                method_name, *method_args, indices=indices, **method_kwargs
            )
        )

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return self._pipe_guard(
            lambda: super(MaskingSubprocVecEnv, self).env_is_wrapped(
                wrapper_class, indices
            )
        )

    def close(self) -> None:
        self._shutdown_workers()

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env_method("action_masks"), dtype=bool)


def _exception_safe_worker(remote, parent_remote, env_fn_wrapper) -> None:
    """Run SB3's protocol while guaranteeing child-owned environment cleanup."""
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    env = None
    try:
        env = _patch_env(env_fn_wrapper.var())
        reset_info: dict[str, Any] | None = {}
        while True:
            try:
                cmd, data = remote.recv()
                if cmd == "step":
                    observation, reward, terminated, truncated, info = env.step(data)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:
                        info["terminal_observation"] = observation
                        observation, reset_info = env.reset()
                    remote.send((observation, reward, done, info, reset_info))
                elif cmd == "reset":
                    maybe_options = {"options": data[1]} if data[1] else {}
                    observation, reset_info = env.reset(seed=data[0], **maybe_options)
                    remote.send((observation, reset_info))
                elif cmd == "render":
                    remote.send(env.render())
                elif cmd == "close":
                    break
                elif cmd == "get_spaces":
                    remote.send((env.observation_space, env.action_space))
                elif cmd == "env_method":
                    method = env.get_wrapper_attr(data[0])
                    remote.send(method(*data[1], **data[2]))
                elif cmd == "get_attr":
                    remote.send(env.get_wrapper_attr(data))
                elif cmd == "has_attr":
                    try:
                        env.get_wrapper_attr(data)
                        remote.send(True)
                    except AttributeError:
                        remote.send(False)
                elif cmd == "set_attr":
                    remote.send(setattr(env, data[0], data[1]))
                elif cmd == "is_wrapped":
                    remote.send(is_wrapped(env, data))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
            except (EOFError, KeyboardInterrupt):
                break
    finally:
        if env is not None:
            try:
                env.close()
            except BaseException:
                pass
        try:
            remote.close()
        except OSError:
            pass


def build_vector_env(
    worker_count: int,
    worker_factory: Callable[[int], gym.Env],
    *,
    subprocess_workers: bool = False,
) -> MaskingDummyVecEnv | MaskingSubprocVecEnv:
    if worker_count <= 0:
        raise ValueError("worker count must be positive")
    if subprocess_workers:
        return MaskingSubprocVecEnv(
            [
                lambda worker_index=index: worker_factory(worker_index)
                for index in range(worker_count)
            ],
            start_method="spawn",
        )
    workers: list[gym.Env] = []
    try:
        for index in range(worker_count):
            workers.append(worker_factory(index))
        return MaskingDummyVecEnv(
            [lambda environment=environment: environment for environment in workers]
        )
    except BaseException:
        for environment in reversed(workers):
            environment.close()
        raise


def _environment_contract(env: Any) -> EnvironmentContract:
    contract = getattr(env, "contract", None)
    if isinstance(contract, EnvironmentContract):
        return contract
    spaces_info = getattr(env, "spaces_info", None)
    if not isinstance(spaces_info, Mapping):
        raise ValueError("training environment does not expose an EnvironmentContract")
    required = (
        "contract_version",
        "contract_hash",
        "encoding_hash",
        "obs_len",
        "n_actions",
        "board",
        "contract_roster",
        "reward",
    )
    missing = [field for field in required if field not in spaces_info]
    if missing:
        raise ValueError(f"training handshake is missing {', '.join(missing)}")
    return EnvironmentContract(
        version=str(spaces_info["contract_version"]),
        contract_hash=str(spaces_info["contract_hash"]),
        encoding_hash=str(spaces_info["encoding_hash"]),
        observation_size=int(spaces_info["obs_len"]),
        action_size=int(spaces_info["n_actions"]),
        board=dict(spaces_info["board"]),
        roster=list(spaces_info["contract_roster"]),
        reward=dict(spaces_info["reward"]),
        semantics=dict(spaces_info.get("adaptive", {})),
    )


def _pool_specs(opponent: Mapping[str, Any]) -> list[Any] | None:
    if opponent.get("kind") != "pool":
        return None
    controllers = opponent.get("controllers")
    if not isinstance(controllers, list) or not controllers:
        raise ValueError("opponent pool requires at least one controller")
    return controllers


class TrainingEnvironmentFactory:
    """Build the real server-backed vector environment for a run."""

    def __init__(self, server_cmd: list[str]) -> None:
        self.server_cmd = list(server_cmd)

    def probe(
        self,
        config: RunConfig,
        scenario_path: Path,
    ) -> tuple[EnvironmentContract, Mapping[str, Any], Mapping[str, Any]]:
        scenario_path = Path(scenario_path)
        scenario = resolve_scenario(
            environment=config.environment,
            scenario_file=scenario_path,
            template_id=None,
        )
        opponent_snapshot = snapshot_opponents(config.opponent)
        env = self._build_worker(
            config,
            scenario_path.parent,
            0,
            scenario_path,
            opponent_snapshot,
            monitor=False,
        )
        try:
            spaces_info = dict(env.spaces_info)
            validate_handshake(scenario, spaces_info)
            return env.contract, spaces_info, opponent_snapshot
        finally:
            env.close()

    def __call__(
        self,
        config: RunConfig,
        run_dir: Path,
        scenario_path: Path,
        opponent_snapshot: Mapping[str, Any],
    ) -> MaskingDummyVecEnv | MaskingSubprocVecEnv:
        return build_vector_env(
            config.workers,
            lambda worker_index: self._build_worker(
                config,
                run_dir,
                worker_index,
                scenario_path,
                opponent_snapshot,
                monitor=True,
            ),
            subprocess_workers=config.workers > 1,
        )

    def _build_worker(
        self,
        config: RunConfig,
        run_dir: Path,
        worker_index: int,
        scenario_path: Path,
        opponent_snapshot: Mapping[str, Any],
        *,
        monitor: bool,
    ) -> gym.Env:
        resolver = ControllerResolver()
        pool = _pool_specs(opponent_snapshot)
        raw_specs = pool if pool is not None else [opponent_snapshot]
        bindings = [resolver.bind(spec) for spec in raw_specs]
        single_scripted = (
            len(bindings) == 1 and bindings[0].resolved.model is None and pool is None
        )

        def build(seat: int, seed: int) -> gym.Env:
            if single_scripted:
                return HexWarsEnv(
                    self.server_cmd,
                    opponent=bindings[0].resolved.server_controller,
                    seat=seat,
                    base_seed=seed,
                    environment=config.environment,
                    scenario_path=scenario_path,
                )
            return SelfPlayEnv(
                self.server_cmd,
                bindings,
                learner_seat=seat,
                base_seed=seed,
                environment=config.environment,
                scenario_path=scenario_path,
            )

        scheduled = ScheduledEnvironment(
            WorkerSchedule(
                base_seed=config.seed,
                worker_index=worker_index,
                worker_count=config.workers,
                learner_seat=config.learner_seat,
            ),
            build,
        )
        if not monitor:
            return scheduled
        monitor_name = "monitor.csv" if config.workers == 1 else f"monitor.worker_{worker_index}.csv"
        adaptive_name = (
            "adaptive_episodes.csv"
            if config.workers == 1
            else f"adaptive_episodes.worker_{worker_index}.csv"
        )
        return EpisodeMonitor(
            scheduled,
            Path(run_dir) / monitor_name,
            threading.Lock(),
            worker_id=worker_index,
            adaptive_path=Path(run_dir) / adaptive_name,
        )
