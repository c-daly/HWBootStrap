from __future__ import annotations

import csv
import multiprocessing as mp
import os
import threading
import time
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from ml_lab.callbacks import TrainingLifecycle
from ml_lab.cli import main as cli_main
from ml_lab.contracts import EnvironmentContract, RunConfig, create_run, request_stop
import ml_lab.envs as env_module
from ml_lab.envs import EpisodeMonitor, ScheduledEnvironment, WorkerSchedule, build_vector_env
from ml_lab.io import atomic_write_json, read_json
from ml_lab.tracking import TrackerHub
from ml_lab.training import run_training


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="b" * 64,
        observation_size=3,
        action_size=2,
        board={"width": 1, "height": 1},
        roster=["1,2,3,4,5,6,7,8,9"],
        reward={"terminal_win": 1.0},
    )


def config(run_name: str = "training") -> RunConfig:
    return RunConfig(
        backend="sb3",
        algorithm="maskable_ppo",
        policy="HexCNN",
        run_name=run_name,
        seed=17,
        total_timesteps=64,
        checkpoint_interval=32,
        workers=2,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "greedy"},
        trackers=[{"kind": "local"}],
        resume_source=None,
    )


def test_worker_seed_streams_are_deterministic_and_disjoint() -> None:
    worker0 = WorkerSchedule(base_seed=17, worker_index=0, worker_count=2)
    worker1 = WorkerSchedule(base_seed=17, worker_index=1, worker_count=2)

    assert [worker0.next_episode(), worker0.next_episode()] == [(17, 0), (19, 1)]
    assert [worker1.next_episode(), worker1.next_episode()] == [(18, 1), (20, 0)]


def test_fixed_learner_seat_is_available_for_diagnosis() -> None:
    schedule = WorkerSchedule(
        base_seed=5, worker_index=0, worker_count=1, learner_seat="1"
    )

    assert [schedule.next_episode(), schedule.next_episode()] == [(5, 1), (6, 1)]


class FakeGymEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, worker: int, seat: int = 0, seed: int = 0) -> None:
        self.worker = worker
        self.seat = seat
        self.initial_seed = seed
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.contract = EnvironmentContract(
            version="tactical-v1",
            contract_hash="b" * 64,
            observation_size=3,
            action_size=2,
            board={"width": 1, "height": 1},
            roster=["1,2,3,4,5,6,7,8,9"],
            reward={"terminal_win": 1.0},
        )
        self.spaces_info = {"channels": 1, "board_h": 1, "board_w": 1, "globals": 2}
        self.reset_seeds: list[int] = []
        self.closed = False
        self._mask = np.asarray([worker == 0, True], dtype=bool)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_seeds.append(int(seed))
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self):
        return self._mask

    def close(self):
        self.closed = True


class SpawnCleanupProbeEnv(gym.Env):
    """Pickle-safe spawned worker probe that records lifecycle events to disk."""

    metadata = {"render_modes": []}

    def __init__(self, marker_path: str, *, fail_step: bool = False) -> None:
        self._marker_path = marker_path
        self._fail_step = fail_step
        self.observation_space = spaces.Box(0.0, 1.0, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.contract = EnvironmentContract(
            version="tactical-v1",
            contract_hash="b" * 64,
            observation_size=3,
            action_size=2,
            board={"width": 1, "height": 1},
            roster=["1,2,3,4,5,6,7,8,9"],
            reward={"terminal_win": 1.0},
        )
        self.spaces_info = {"worker_pid": os.getpid()}
        self._record("opened")

    def _record(self, event: str) -> None:
        with Path(self._marker_path).open("a", encoding="utf-8") as stream:
            stream.write(f"{event}:{os.getpid()}\n")

    def reset(self, *, seed=None, options=None):
        del seed, options
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        del action
        if self._fail_step:
            raise RuntimeError("spawned step failed")
        return np.zeros(3, dtype=np.float32), 0.0, False, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(2, dtype=bool)

    def close(self) -> None:
        self._record("closed")


def _spawn_cleanup_probe(marker_path: str, *, fail_step: bool = False) -> SpawnCleanupProbeEnv:
    return SpawnCleanupProbeEnv(marker_path, fail_step=fail_step)


def _spawn_construction_failure(marker_path: str) -> SpawnCleanupProbeEnv:
    with Path(marker_path).open("a", encoding="utf-8") as stream:
        stream.write(f"construction_failed:{os.getpid()}\n")
    raise RuntimeError("spawned construction failed")


def _wait_for_no_new_children(previous_pids: set[int], timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        new_children = [
            process
            for process in mp.active_children()
            if process.pid not in previous_pids and process.is_alive()
        ]
        if not new_children:
            return True
        time.sleep(0.05)
    return False


def test_scheduled_environment_alternates_seat_and_uses_worker_seed_stream() -> None:
    created: list[FakeGymEnv] = []

    def build(seat: int, seed: int) -> FakeGymEnv:
        env = FakeGymEnv(0, seat, seed)
        created.append(env)
        return env

    env = ScheduledEnvironment(
        WorkerSchedule(base_seed=23, worker_index=0, worker_count=1), build
    )

    env.reset()
    env.reset()

    assert [(item.seat, item.initial_seed) for item in created] == [(0, 23), (1, 24)]
    assert created[0].reset_seeds == [23]
    assert created[0].closed is True
    assert created[1].reset_seeds == [24]


def test_vector_workers_expose_direct_action_masks_and_own_distinct_environments() -> None:
    created: list[FakeGymEnv] = []

    def factory(worker_index: int) -> FakeGymEnv:
        env = FakeGymEnv(worker_index)
        created.append(env)
        return env

    vector = build_vector_env(2, factory)
    try:
        assert vector.action_masks().tolist() == [[True, True], [False, True]]
        assert len({id(env) for env in created}) == 2
    finally:
        vector.close()


def test_vector_construction_closes_workers_started_before_a_later_failure() -> None:
    started = FakeGymEnv(0)

    def factory(worker_index: int) -> FakeGymEnv:
        if worker_index == 0:
            return started
        raise RuntimeError("second server failed")

    with pytest.raises(RuntimeError, match="second server failed"):
        build_vector_env(2, factory)

    assert started.closed is True


def test_multiworker_vectorization_selects_spawned_subprocesses_without_parent_envs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}
    parent_factory_calls: list[int] = []

    class FakeSubprocessVector:
        def __init__(self, env_fns, *, start_method: str) -> None:
            captured["env_fns"] = env_fns
            captured["start_method"] = start_method

    monkeypatch.setattr(env_module, "MaskingSubprocVecEnv", FakeSubprocessVector)

    vector = build_vector_env(
        2,
        lambda worker_index: parent_factory_calls.append(worker_index),
        subprocess_workers=True,
    )

    assert isinstance(vector, FakeSubprocessVector)
    assert captured["start_method"] == "spawn"
    assert len(captured["env_fns"]) == 2
    assert parent_factory_calls == []


def test_spawned_constructor_failure_closes_and_reaps_sibling_workers(tmp_path: Path) -> None:
    marker = tmp_path / "spawn-construction.txt"
    previous_pids = {process.pid for process in mp.active_children()}

    with pytest.raises(EOFError):
        env_module.MaskingSubprocVecEnv(
            [
                lambda: _spawn_cleanup_probe(str(marker)),
                lambda: _spawn_construction_failure(str(marker)),
            ],
            start_method="spawn",
        )

    assert _wait_for_no_new_children(previous_pids)
    events = marker.read_text(encoding="utf-8")
    assert "opened:" in events
    assert "construction_failed:" in events
    assert "closed:" in events


def test_spawned_runtime_failure_closes_worker_and_parent_reaps_it(tmp_path: Path) -> None:
    marker = tmp_path / "spawn-runtime.txt"
    vector = env_module.MaskingSubprocVecEnv(
        [lambda: _spawn_cleanup_probe(str(marker), fail_step=True)],
        start_method="spawn",
    )
    processes = list(vector.processes)
    vector.reset()

    with pytest.raises(EOFError):
        vector.step(np.asarray([0]))

    assert vector.closed is True
    assert all(not process.is_alive() for process in processes)
    assert "closed:" in marker.read_text(encoding="utf-8")
    vector.close()


def test_episode_monitor_emits_sb3_episode_info(tmp_path: Path) -> None:
    monitor_path = tmp_path / "monitor.csv"
    monitor_path.write_text(
        "episode_reward,episode_length,elapsed_seconds\n", encoding="utf-8"
    )
    monitored = EpisodeMonitor(FakeGymEnv(0), monitor_path, threading.Lock(), worker_id=3)

    monitored.reset(seed=7)
    _, _, terminated, _, info = monitored.step(0)

    assert terminated is True
    assert info["episode"]["r"] == 0.0
    assert info["episode"]["l"] == 1
    assert info["episode"]["worker_id"] == 3
    assert info["episode"]["episode_number"] == 1


class FakeCheckpointAdapter:
    name = "maskable_ppo"
    policy_name = "HexCNN"
    experimental = False

    def __init__(self, contract: EnvironmentContract) -> None:
        self.contract = contract
        self.inspected: list[Path] = []

    def save(self, model, path: Path) -> Path:
        generation = getattr(model, "update_generation", 0)
        path.write_bytes(
            f"model-{model.num_timesteps}-generation-{generation}".encode()
        )
        return path

    def inspect(self, path: Path, expected_contract: EnvironmentContract):
        self.inspected.append(path)
        assert path.read_bytes().startswith(b"model-")
        return {
            "contract_hash": expected_contract.contract_hash,
            "observation_size": expected_contract.observation_size,
            "action_size": expected_contract.action_size,
        }


def fake_model(step: int):
    logger = SimpleNamespace(
        name_to_value={
            "rollout/ep_rew_mean": 1.25,
            "time/episodes": 3,
            "time/fps": 40,
        },
        output_formats=[],
    )
    return SimpleNamespace(num_timesteps=step, logger=logger)


def test_rollout_updates_status_and_authoritative_progress(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("rollout"), contract)
    trackers = TrackerHub(run_dir, [{"kind": "local"}])
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        trackers,
        checkpoint_interval=32,
    )

    lifecycle.on_rollout_end(fake_model(24))

    manifest = read_json(run_dir / "run.json")
    assert manifest["state"] == "running"
    assert manifest["timesteps"] == 24
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert row["timesteps"] == "24"
    assert row["episodes"] == "3"
    assert row["mean_reward"] == "1.25"
    assert row["steps_per_second"] == "40"


def test_rollout_uses_sb3_episode_buffer_and_timing_when_logger_is_not_populated(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("fallback-progress"), contract)
    times = iter([10.0, 12.0])
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
        clock=lambda: next(times),
    )
    model = SimpleNamespace(
        num_timesteps=24,
        logger=SimpleNamespace(name_to_value={}, output_formats=[]),
        _episode_num=4,
        ep_info_buffer=[{"r": 1.0}, {"r": 3.0}],
    )

    lifecycle.on_rollout_end(model)

    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert row["episodes"] == "4"
    assert row["mean_reward"] == "2.0"
    assert row["steps_per_second"] == "12.0"


def test_rollout_counts_completed_episodes_from_per_worker_buffer_maxima(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("episode-count"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
    )
    model = SimpleNamespace(
        num_timesteps=64,
        logger=SimpleNamespace(name_to_value={}, output_formats=[]),
        _episode_num=0,
        ep_info_buffer=[
            {"r": -2.0, "worker_id": 0, "episode_number": 1},
            {"r": 1.0, "worker_id": 1, "episode_number": 1},
            {"r": 3.0, "worker_id": 0, "episode_number": 2},
        ],
    )

    lifecycle.on_rollout_end(model)

    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        row = list(csv.DictReader(stream))[-1]
    assert row["episodes"] == "3"
    assert float(row["mean_reward"]) == pytest.approx(2.0 / 3.0)


def test_periodic_checkpoint_waits_until_next_rollout_start_after_policy_update(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("checkpoint"), contract)
    adapter = FakeCheckpointAdapter(contract)
    lifecycle = TrainingLifecycle(
        run_dir, contract, adapter, TrackerHub(run_dir, []), checkpoint_interval=32
    )

    model = fake_model(32)
    model.update_generation = 0

    assert lifecycle.on_step(model) is True
    assert read_json(run_dir / "run.json")["latest_checkpoint"] is None

    lifecycle.on_rollout_end(model)
    model.update_generation = 1
    lifecycle.on_rollout_start(model)

    manifest = read_json(run_dir / "run.json")
    published = run_dir / manifest["latest_checkpoint"]
    assert published.name == "step_000000032.zip"
    assert published.read_bytes() == b"model-32-generation-1"
    assert manifest["latest_checkpoint_step"] == 32
    assert len(adapter.inspected) == 1
    assert not list((run_dir / "checkpoints").glob(".pending*"))
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[-1]["timesteps"] == "32"


def test_stop_after_checkpoint_publishes_post_update_then_stops_cleanly(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("controlled"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
    )
    request_stop(run_dir, after_checkpoint=True)

    model = fake_model(16)
    model.update_generation = 0
    assert lifecycle.on_step(model) is True
    assert read_json(run_dir / "run.json")["state"] == "stopping"
    model.num_timesteps = 32
    assert lifecycle.on_step(model) is True
    assert read_json(run_dir / "run.json")["latest_checkpoint"] is None

    model.update_generation = 1
    with pytest.raises(RuntimeError, match="stop after checkpoint"):
        lifecycle.on_rollout_start(model)

    manifest = read_json(run_dir / "run.json")
    assert manifest["latest_checkpoint_step"] == 32
    checkpoint = run_dir / manifest["latest_checkpoint"]
    assert checkpoint.read_bytes() == b"model-32-generation-1"
    assert lifecycle.stop_requested is True


def test_resume_checkpoint_cadence_starts_from_loaded_model_timesteps(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("resume-cadence"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
        initial_step=64,
    )
    model = fake_model(80)
    model.update_generation = 1

    assert lifecycle.on_step(model) is True
    lifecycle.on_rollout_start(model)
    assert read_json(run_dir / "run.json")["latest_checkpoint"] is None

    model.num_timesteps = 96
    lifecycle.on_step(model)
    model.update_generation = 2
    lifecycle.on_rollout_start(model)
    assert read_json(run_dir / "run.json")["latest_checkpoint_step"] == 96


def test_stop_now_does_not_publish_an_unrequested_checkpoint(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("stop-now"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
    )
    request_stop(run_dir, after_checkpoint=False)

    assert lifecycle.on_step(fake_model(16)) is False
    assert read_json(run_dir / "run.json")["latest_checkpoint"] is None


def test_episode_count_never_decreases_when_bounded_buffer_drops_a_worker(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("bounded-episodes"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
    )
    model = SimpleNamespace(
        num_timesteps=32,
        logger=SimpleNamespace(name_to_value={}, output_formats=[]),
        _episode_num=0,
        ep_info_buffer=[
            {"r": 1.0, "worker_id": 0, "episode_number": 100},
            {"r": 1.0, "worker_id": 1, "episode_number": 50},
        ],
    )
    lifecycle.on_rollout_end(model)

    model.num_timesteps = 64
    model.ep_info_buffer = [
        {"r": 2.0, "worker_id": 0, "episode_number": 101}
    ]
    lifecycle.on_rollout_end(model)

    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["episodes"] for row in rows[-2:]] == ["150", "151"]


def test_resumed_episode_numbers_add_to_source_run_offset(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("resumed-episodes"), contract)
    lifecycle = TrainingLifecycle(
        run_dir,
        contract,
        FakeCheckpointAdapter(contract),
        TrackerHub(run_dir, []),
        checkpoint_interval=32,
        episode_offset=200,
    )
    model = SimpleNamespace(
        num_timesteps=224,
        logger=SimpleNamespace(name_to_value={}, output_formats=[]),
        _episode_num=0,
        ep_info_buffer=[
            {"r": 1.0, "worker_id": 0, "episode_number": 1},
            {"r": 1.0, "worker_id": 1, "episode_number": 1},
        ],
    )

    lifecycle.on_rollout_end(model)

    manifest = read_json(run_dir / "run.json")
    assert manifest["episodes"] == 202
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[-1]["episodes"] == "202"


class FakeVectorEnv:
    def __init__(self, contract: EnvironmentContract) -> None:
        self.contract = contract
        self.spaces_info = {"channels": 1, "board_h": 1, "board_w": 1, "globals": 2}
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeTrainingModel:
    def __init__(self, fail: bool = False, initial_step: int = 0) -> None:
        self.num_timesteps = initial_step
        self.logger = SimpleNamespace(name_to_value={}, output_formats=[])
        self.fail = fail
        self.learn_calls: list[dict[str, object]] = []
        self.configured_logger = None

    def set_logger(self, logger) -> None:
        self.configured_logger = logger

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        if self.fail:
            raise RuntimeError("training exploded")
        self.num_timesteps += int(kwargs["total_timesteps"])
        return self


class StopAfterFinalRolloutModel(FakeTrainingModel):
    def __init__(self) -> None:
        super().__init__()
        self.update_generation = 0

    def learn(self, **kwargs):
        self.learn_calls.append(kwargs)
        lifecycle = kwargs["callback"].lifecycle
        lifecycle.on_rollout_start(self)
        self.num_timesteps += int(kwargs["total_timesteps"])
        request_stop(lifecycle.run_dir, after_checkpoint=True)
        assert lifecycle.on_step(self) is True
        lifecycle.on_rollout_end(self)
        self.update_generation += 1
        return self


class FakeTrainingAdapter(FakeCheckpointAdapter):
    def __init__(self, contract: EnvironmentContract, model: FakeTrainingModel) -> None:
        super().__init__(contract)
        self.model = model

    def create(self, env, **kwargs):
        return self.model

    def load(self, path: Path, *, env, device: str):
        return self.model

    def validate_model(self, model, expected_contract: EnvironmentContract) -> None:
        assert expected_contract == self.contract


def _resume_source(
    tmp_path: Path, contract: EnvironmentContract, *, algorithm: str = "maskable_ppo"
) -> Path:
    source_config = replace(
        config(f"source-{algorithm}"),
        algorithm=algorithm,
        policy="HexCNN" if algorithm == "maskable_ppo" else "MlpPolicy",
    )
    source_run = create_run(tmp_path, source_config, contract)
    checkpoint = source_run / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"model")
    manifest = read_json(source_run / "run.json")
    manifest["latest_checkpoint"] = "checkpoints/step_000000064.zip"
    manifest["latest_checkpoint_step"] = 64
    atomic_write_json(source_run / "run.json", manifest)
    return source_run


def test_training_runner_completes_publishes_final_checkpoint_and_closes_env(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    env = FakeVectorEnv(contract)
    model = FakeTrainingModel()
    adapter = FakeTrainingAdapter(contract, model)

    run_dir = run_training(
        config("complete"),
        runs_root=tmp_path,
        environment_factory=lambda _config, _run_dir: env,
        algorithm_adapter=adapter,
    )

    manifest = read_json(run_dir / "run.json")
    assert manifest["state"] == "completed"
    assert manifest["timesteps"] == 64
    assert manifest["latest_checkpoint_step"] == 64
    assert env.closed is True
    assert model.learn_calls[0]["reset_num_timesteps"] is True


def test_training_runner_marks_failure_and_closes_env(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    env = FakeVectorEnv(contract)
    adapter = FakeTrainingAdapter(contract, FakeTrainingModel(fail=True))

    with pytest.raises(RuntimeError, match="training exploded"):
        run_training(
            config("failed"),
            runs_root=tmp_path,
            environment_factory=lambda _config, _run_dir: env,
            algorithm_adapter=adapter,
        )

    assert read_json(tmp_path / "failed" / "run.json")["state"] == "failed"
    assert env.closed is True


def test_absolute_resume_treats_timesteps_as_final_target(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source = _resume_source(tmp_path, contract)
    env = FakeVectorEnv(contract)
    model = FakeTrainingModel(initial_step=64)
    adapter = FakeTrainingAdapter(contract, model)
    run_config = replace(
        config("absolute-resume"),
        resume_source=str(source),
        total_timesteps=96,
        timestep_mode="absolute",
    )

    run_training(
        run_config,
        runs_root=tmp_path,
        environment_factory=lambda _config, _run_dir: env,
        algorithm_adapter=adapter,
    )

    assert model.learn_calls[0]["total_timesteps"] == 32
    assert model.num_timesteps == 96


def test_legacy_additional_resume_trains_requested_extra_steps(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source = _resume_source(tmp_path, contract)
    env = FakeVectorEnv(contract)
    model = FakeTrainingModel(initial_step=64)
    adapter = FakeTrainingAdapter(contract, model)
    run_config = replace(
        config("additional-resume"),
        resume_source=str(source),
        total_timesteps=32,
        timestep_mode="additional",
    )

    run_dir = run_training(
        run_config,
        runs_root=tmp_path,
        environment_factory=lambda _config, _run_dir: env,
        algorithm_adapter=adapter,
    )

    assert model.learn_calls[0]["total_timesteps"] == 32
    assert model.num_timesteps == 96
    assert read_json(run_dir / "run.json")["config"]["timestep_mode"] == "additional"


def test_legacy_ppo_wrapper_persists_unsafe_additional_resume_mode() -> None:
    import train_maskable_ppo

    args = SimpleNamespace(
        opponent="greedy",
        seat=0,
        seed=2,
        timesteps=50,
        checkpoint_freq=10,
        resume="legacy.zip",
    )

    run_config = train_maskable_ppo.build_run_config(args, Path("runs/legacy"))

    assert run_config.timestep_mode == "additional"
    assert run_config.allow_unsafe_legacy_resume is True


def test_training_wires_functional_tensorboard_writer_when_requested(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    env = FakeVectorEnv(contract)
    model = FakeTrainingModel()
    adapter = FakeTrainingAdapter(contract, model)
    writer = SimpleNamespace(flush=lambda: None, close=lambda: None)
    captured: dict[str, object] = {}

    def tracker_factory(run_dir, specs, *, tensorboard_writer=None):
        captured["writer"] = tensorboard_writer
        return TrackerHub(run_dir, [], tensorboard_writer=tensorboard_writer)

    run_training(
        replace(config("tensorboard-run"), trackers=[{"kind": "tensorboard"}]),
        runs_root=tmp_path,
        environment_factory=lambda _config, _run_dir: env,
        algorithm_adapter=adapter,
        tracker_factory=tracker_factory,
        summary_writer_factory=lambda _log_dir: writer,
    )

    assert captured["writer"] is writer


def test_sb3_human_logger_appends_to_stdout_and_train_log(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    import ml_lab.training as training_module

    run_dir = create_run(tmp_path, config("human-log"), contract)
    stdout = StringIO()
    logger = training_module.build_sb3_logger(run_dir, stdout=stdout)

    logger.record("rollout/ep_rew_mean", 1.5)
    logger.dump(step=32)
    logger.close()

    assert "ep_rew_mean" in stdout.getvalue()
    assert "ep_rew_mean" in (run_dir / "train.log").read_text(encoding="utf-8")


def test_multiworker_run_manifest_exposes_monitor_shards_as_authoritative(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    run_dir = create_run(tmp_path, config("monitor-shards"), contract)

    manifest = read_json(run_dir / "run.json")
    assert manifest["monitor_files"] == [
        "monitor.worker_0.csv",
        "monitor.worker_1.csv",
    ]
    with (run_dir / "monitor.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [
            ["episode_reward", "episode_length", "elapsed_seconds"]
        ]


def test_stop_after_checkpoint_on_final_rollout_publishes_update_and_marks_stopped(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    env = FakeVectorEnv(contract)
    model = StopAfterFinalRolloutModel()
    adapter = FakeTrainingAdapter(contract, model)

    run_dir = run_training(
        config("final-stop"),
        runs_root=tmp_path,
        environment_factory=lambda _config, _run_dir: env,
        algorithm_adapter=adapter,
    )

    manifest = read_json(run_dir / "run.json")
    assert manifest["state"] == "stopped"
    assert manifest["latest_checkpoint_step"] == 64
    checkpoint = run_dir / manifest["latest_checkpoint"]
    assert checkpoint.read_bytes() == b"model-64-generation-1"


def test_train_cli_builds_run_config_and_invokes_unified_runner(tmp_path: Path) -> None:
    received: list[tuple[RunConfig, Path, list[str]]] = []

    def runner(run_config: RunConfig, *, runs_root: Path, server_cmd: list[str]):
        received.append((run_config, runs_root, server_cmd))
        return runs_root / run_config.run_name

    exit_code = cli_main(
        [
            "train",
            "--run",
            "cli-smoke",
            "--algorithm",
            "maskable_ppo",
            "--opponent",
            "greedy",
            "--timesteps",
            "64",
            "--checkpoint-every",
            "32",
            "--workers",
            "1",
            "--device",
            "cpu",
            "--runs-root",
            str(tmp_path),
            "--server",
            "fake-server.dll",
        ],
        runner=runner,
    )

    assert exit_code == 0
    run_config, runs_root, server_cmd = received[0]
    assert run_config.run_name == "cli-smoke"
    assert run_config.learner_seat == "alternating"
    assert run_config.opponent == {"kind": "scripted", "name": "greedy"}
    assert runs_root == tmp_path
    assert server_cmd == ["dotnet", "fake-server.dll"]
    assert run_config.timestep_mode == "absolute"
    assert run_config.allow_unsafe_legacy_resume is False
