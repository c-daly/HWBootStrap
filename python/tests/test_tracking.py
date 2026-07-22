import csv
import sys
import types
from pathlib import Path

import pytest

from ml_lab.contracts import EnvironmentContract, RunConfig, create_run
from ml_lab.io import read_json
from ml_lab.tracking import SB3TrackingFacade, TrackerHub


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    config = RunConfig(
        backend="stable_baselines3",
        algorithm="maskable_ppo",
        policy="hex_cnn",
        run_name="tracking_run",
        seed=17,
        total_timesteps=100,
        checkpoint_interval=10,
        workers=1,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "greedy"},
        trackers=[],
        resume_source=None,
    )
    contract = EnvironmentContract(
        version="tactical-v1",
        contract_hash="abc123",
        observation_size=761,
        action_size=379,
        board={"width": 13, "height": 9},
        roster=["scout"],
        reward={"win": 1.0},
    )
    return create_run(tmp_path, config, contract)


def test_local_progress_csv_is_authoritative_even_without_tracker_specs(run_dir: Path) -> None:
    trackers = TrackerHub(run_dir, [])

    trackers.start_run()
    trackers.log_metrics(
        {"episodes": 3, "mean_reward": 1.25, "steps_per_second": 8.5}, step=64
    )
    trackers.finish("completed")

    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[-1]["timesteps"] == "64"
    assert rows[-1]["episodes"] == "3"
    assert rows[-1]["mean_reward"] == "1.25"
    assert rows[-1]["steps_per_second"] == "8.5"
    assert read_json(run_dir / "run.json")["tracker_status"] == []


def test_optional_tracker_failure_degrades_without_stopping_local_tracking(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("failing_tracker")

    def fail(event: dict[str, object]) -> None:
        if event["type"] == "metrics":
            raise RuntimeError("remote service unavailable")

    module.record = fail
    monkeypatch.setitem(sys.modules, module.__name__, module)
    trackers = TrackerHub(run_dir, [{"kind": "custom", "adapter": "failing_tracker:record"}])

    trackers.start_run()
    trackers.log_metrics({"mean_reward": 2.0}, step=10)

    assert trackers.degraded["custom:failing_tracker:record"] == (
        "log_metrics failed: remote service unavailable"
    )
    assert read_json(run_dir / "run.json")["tracker_status"] == [
        {
            "message": "log_metrics failed: remote service unavailable",
            "name": "custom:failing_tracker:record",
            "status": "degraded",
        }
    ]
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[-1]["timesteps"] == "10"


def test_wandb_is_imported_only_when_selected_and_never_serializes_credentials(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_wandb = types.ModuleType("wandb")
    starts: list[dict[str, object]] = []

    class FakeRun:
        def log(self, _metrics: dict[str, object], step: int) -> None:
            pass

        def finish(self, exit_code: int) -> None:
            pass

    def init(**kwargs: object) -> FakeRun:
        starts.append(kwargs)
        return FakeRun()

    fake_wandb.init = init
    monkeypatch.delitem(sys.modules, "wandb", raising=False)
    no_wandb = TrackerHub(run_dir, [])
    no_wandb.start_run()
    assert "wandb" not in sys.modules

    monkeypatch.setitem(sys.modules, "wandb", fake_wandb)
    secret = "not-a-serialized-api-key"
    trackers = TrackerHub(
        run_dir,
        [
            {
                "kind": "wandb",
                "project": "hexwars",
                "entity": "training-team",
                "mode": "offline",
                "api_key": secret,
            }
        ],
    )

    trackers.start_run()

    assert starts == [
        {
            "config": {"run_name": "tracking_run"},
            "dir": str(run_dir),
            "entity": "training-team",
            "mode": "offline",
            "name": "tracking_run",
            "project": "hexwars",
        }
    ]
    assert secret not in (run_dir / "run.json").read_text(encoding="utf-8")


def test_custom_adapter_receives_normalized_metric_event(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = types.ModuleType("event_recorder")
    received: list[dict[str, object]] = []
    module.record = received.append
    monkeypatch.setitem(sys.modules, module.__name__, module)
    trackers = TrackerHub(run_dir, [{"kind": "custom", "adapter": "event_recorder:record"}])

    trackers.start_run()
    trackers.log_metrics({"loss": 0.125, "reward": 4}, step=20)

    event = received[-1]
    assert event["type"] == "metrics"
    assert event["run_name"] == "tracking_run"
    assert event["step"] == 20
    assert event["metrics"] == {"loss": 0.125, "reward": 4}
    assert isinstance(event["timestamp"], str)


def test_tensorboard_writer_and_sb3_facade_do_not_require_live_sb3(run_dir: Path) -> None:
    scalars: list[tuple[str, float, int]] = []

    class Writer:
        def add_scalar(self, key: str, value: float, step: int) -> None:
            scalars.append((key, value, step))

    class Logger:
        name_to_value = {"rollout/ep_rew_mean": 1.5, "time/fps": 30}

    trackers = TrackerHub(run_dir, [{"kind": "tensorboard"}], tensorboard_writer=Writer())

    SB3TrackingFacade(trackers).log_from_logger(Logger(), step=128)

    assert scalars == [("rollout/ep_rew_mean", 1.5, 128), ("time/fps", 30.0, 128)]
    with (run_dir / "progress.csv").open(newline="", encoding="utf-8") as stream:
        assert list(csv.DictReader(stream))[-1]["timesteps"] == "128"
