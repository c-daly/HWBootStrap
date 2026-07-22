"""Vector-worker construction, deterministic episodes, and opponent routing."""

from __future__ import annotations

import csv
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from hexwars_gym import HexWarsEnv
from selfplay_env import SelfPlayEnv

from .contracts import EnvironmentContract, MONITOR_HEADER, RunConfig
from .controllers import ControllerBinding, ControllerResolver


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

    def next_episode(self) -> tuple[int, int]:
        episode = self.episode_index
        self.episode_index += 1
        seed = self.base_seed + self.worker_index + episode * self.worker_count
        if self.learner_seat == "alternating":
            seat = (self.worker_index + episode) % 2
        else:
            seat = int(self.learner_seat)
        return seed, seat


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
        seed, seat = self._pending
        self._env = builder(seat, seed)
        self._seat = seat
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
        episode_seed, seat = assignment
        self._ensure_seat(seat, episode_seed)
        return self._env.reset(seed=episode_seed, options=options)

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
    ) -> None:
        super().__init__(env)
        self.contract = _environment_contract(env)
        self.spaces_info = dict(getattr(env, "spaces_info", {}))
        self._path = Path(path)
        self._lock = lock
        self._worker_id = worker_id
        self._episode_number = 0
        self._started = time.monotonic()
        self._reward = 0.0
        self._length = 0

    def reset(self, **kwargs):
        self._reward = 0.0
        self._length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._reward += float(reward)
        self._length += 1
        if terminated or truncated:
            self._episode_number += 1
            elapsed = time.monotonic() - self._started
            self._append_episode(elapsed)
            info = dict(info)
            info["episode"] = {
                "r": self._reward,
                "l": self._length,
                "t": elapsed,
                "worker_id": self._worker_id,
                "episode_number": self._episode_number,
            }
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
                csv.writer(stream).writerow([self._reward, self._length, elapsed])


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
        super().__init__(env_fns, start_method=start_method)
        contracts = self.get_attr("contract")
        if any(contract != contracts[0] for contract in contracts[1:]):
            self.close()
            raise ValueError("all vector workers must use the same environment contract")
        self.contract = contracts[0]
        self.spaces_info = dict(self.get_attr("spaces_info")[0])

    def action_masks(self) -> np.ndarray:
        return np.asarray(self.env_method("action_masks"), dtype=bool)


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
        observation_size=int(spaces_info["obs_len"]),
        action_size=int(spaces_info["n_actions"]),
        board=dict(spaces_info["board"]),
        roster=list(spaces_info["contract_roster"]),
        reward=dict(spaces_info["reward"]),
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

    def __call__(
        self, config: RunConfig, run_dir: Path
    ) -> MaskingDummyVecEnv | MaskingSubprocVecEnv:
        return build_vector_env(
            config.workers,
            lambda worker_index: self._build_worker(config, run_dir, worker_index),
            subprocess_workers=config.workers > 1,
        )

    def _build_worker(
        self, config: RunConfig, run_dir: Path, worker_index: int
    ) -> gym.Env:
        resolver = ControllerResolver()
        pool = _pool_specs(config.opponent)
        raw_specs = pool if pool is not None else [config.opponent]
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
                )
            return SelfPlayEnv(
                self.server_cmd,
                bindings,
                learner_seat=seat,
                base_seed=seed,
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
        monitor_name = "monitor.csv" if config.workers == 1 else f"monitor.worker_{worker_index}.csv"
        return EpisodeMonitor(
            scheduled,
            Path(run_dir) / monitor_name,
            threading.Lock(),
            worker_id=worker_index,
        )
