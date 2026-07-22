import csv
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from ml_lab.contracts import (
    ContractMismatch,
    EnvironmentContract,
    RunConfig,
    create_run,
    publish_checkpoint,
    request_stop,
    update_run_state,
    validate_run_name,
)
from ml_lab.io import atomic_write_json, read_json


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="abc123",
        encoding_hash="a" * 64,
        observation_size=761,
        action_size=379,
        board={"width": 13, "height": 9},
        roster=["scout", "soldier", "tank"],
        reward={"win": 1.0, "loss": -1.0},
    )


@pytest.fixture
def config() -> RunConfig:
    return RunConfig(
        backend="stable_baselines3",
        algorithm="maskable_ppo",
        policy="hex_cnn",
        run_name="ppo_counter_run1",
        seed=17,
        total_timesteps=1_000,
        checkpoint_interval=100,
        workers=2,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "greedy"},
        trackers=[{"kind": "local"}],
        resume_source=None,
    )


@pytest.mark.parametrize(
    "name",
    ["", " has-space", "has space", "../escape", "a/b", "a\\b", ".hidden", "name!"],
)
def test_validate_run_name_rejects_unsafe_names(name: str) -> None:
    with pytest.raises(ValueError, match="run name"):
        validate_run_name(name)


def test_create_run_writes_complete_manifest_and_tree(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    run = create_run(tmp_path, config, contract)

    assert run == tmp_path / config.run_name
    manifest = read_json(run / "run.json")
    assert manifest["state"] == "created"
    assert manifest["pid"] is None
    assert manifest["config"]["algorithm"] == "maskable_ppo"
    assert manifest["contract"]["observation_size"] == contract.observation_size
    assert manifest["contract"]["contract_hash"] == contract.contract_hash
    assert read_json(run / "params.json")["config"] == manifest["config"]
    assert read_json(run / "control.json") == {"request": None}
    assert read_json(run / "evaluation.json") == {}
    assert (run / "checkpoints").is_dir()
    assert (run / "replays").is_dir()
    assert (run / "train.log").read_text(encoding="utf-8") == ""
    with (run / "progress.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.reader(stream)) == [
            "timestamp",
            "timesteps",
            "episodes",
            "mean_reward",
            "steps_per_second",
        ]
    with (run / "monitor.csv").open(newline="", encoding="utf-8") as stream:
        assert next(csv.reader(stream)) == ["episode_reward", "episode_length", "elapsed_seconds"]


def test_create_run_refuses_existing_directory(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    create_run(tmp_path, config, contract)
    with pytest.raises(FileExistsError):
        create_run(tmp_path, config, contract)


@pytest.mark.parametrize(
    "tracker",
    [
        {"kind": "wandb", "api_key": "do-not-leak-value"},
        {"kind": "custom", "settings": {"token": "do-not-leak-value"}},
        {"kind": "custom", "settings": [{"password": "do-not-leak-value"}]},
        {"kind": "wandb", "authentication": {"secret": "do-not-leak-value"}},
    ],
)
def test_create_run_rejects_recursive_tracker_credentials_before_writing_files(
    tmp_path: Path,
    config: RunConfig,
    contract: EnvironmentContract,
    tracker: dict[str, object],
) -> None:
    unsafe = replace(config, trackers=[tracker])

    with pytest.raises(
        ValueError, match="tracker configuration contains forbidden credential key"
    ) as error:
        create_run(tmp_path, unsafe, contract)

    assert "do-not-leak-value" not in str(error.value)
    assert not (tmp_path / unsafe.run_name).exists()


def test_atomic_write_json_replaces_document_without_leaving_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"generation": 1})
    atomic_write_json(path, {"generation": 2, "ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"generation": 2, "ok": True}
    assert list(tmp_path.glob("*.tmp")) == []


def test_atomic_write_json_failure_preserves_document_and_removes_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    atomic_write_json(path, {"generation": 1})

    with pytest.raises(TypeError):
        atomic_write_json(path, {"not_json": {1, 2, 3}})

    assert read_json(path) == {"generation": 1}
    assert list(tmp_path.glob("*.tmp")) == []


def test_update_state_preserves_config_and_contract(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    run = create_run(tmp_path, config, contract)
    original = read_json(run / "run.json")

    update_run_state(run, "running", pid=42, timesteps=128, latest_message="rollout complete")
    updated = read_json(run / "run.json")

    assert updated["config"] == original["config"]
    assert updated["contract"] == original["contract"]
    assert updated["state"] == "running"
    assert updated["pid"] == 42
    assert updated["timesteps"] == 128
    assert updated["latest_message"] == "rollout complete"


def test_control_requests_distinguish_graceful_and_immediate_stop(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    run = create_run(tmp_path, config, contract)

    request_stop(run, after_checkpoint=True)
    assert read_json(run / "control.json")["request"] == "stop_after_checkpoint"

    request_stop(run, after_checkpoint=False)
    assert read_json(run / "control.json")["request"] == "stop_now"


def test_publish_checkpoint_validates_then_atomically_updates_latest(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    run = create_run(tmp_path, config, contract)
    pending = tmp_path / "pending.zip"
    pending.write_bytes(b"model")

    published = publish_checkpoint(
        source=pending,
        run_dir=run,
        step=100,
        expected_contract=contract,
        inspector=lambda _: {
            "environment": contract.environment,
            "contract_version": contract.version,
            "contract_hash": contract.contract_hash,
            "encoding_hash": contract.encoding_hash,
            "observation_size": contract.observation_size,
            "action_size": contract.action_size,
        },
    )

    assert published == run / "checkpoints" / "step_000000100.zip"
    assert published.read_bytes() == b"model"
    assert not pending.exists()
    manifest = read_json(run / "run.json")
    assert manifest["latest_checkpoint"] == "checkpoints/step_000000100.zip"
    assert manifest["latest_checkpoint_step"] == 100


def test_publish_checkpoint_stages_final_replace_inside_checkpoint_directory(
    tmp_path: Path,
    config: RunConfig,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = create_run(tmp_path / "runs", config, contract)
    pending = tmp_path / "external-volume" / "pending.zip"
    pending.parent.mkdir()
    pending.write_bytes(b"model")
    real_replace = os.replace
    replacements: list[tuple[Path, Path]] = []

    def require_same_directory(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        destination_path = Path(destination)
        assert source_path.parent == destination_path.parent
        replacements.append((source_path, destination_path))
        real_replace(source_path, destination_path)

    monkeypatch.setattr("ml_lab.contracts.os.replace", require_same_directory)
    published = publish_checkpoint(
        source=pending,
        run_dir=run,
        step=200,
        expected_contract=contract,
        inspector=lambda _: {
            "environment": contract.environment,
            "contract_version": contract.version,
            "contract_hash": contract.contract_hash,
            "encoding_hash": contract.encoding_hash,
            "observation_size": contract.observation_size,
            "action_size": contract.action_size,
        },
    )

    assert published.read_bytes() == b"model"
    assert not pending.exists()
    assert any(destination == published for _, destination in replacements)


def test_publish_checkpoint_rejects_incompatible_model_without_mutating_run(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract
) -> None:
    run = create_run(tmp_path, config, contract)
    pending = tmp_path / "pending.zip"
    pending.write_bytes(b"model")
    incompatible = replace(contract, encoding_hash="b" * 64)

    with pytest.raises(ContractMismatch, match="encoding hash"):
        publish_checkpoint(
            source=pending,
            run_dir=run,
            step=100,
            expected_contract=contract,
            inspector=lambda _: {
                "environment": incompatible.environment,
                "contract_version": incompatible.version,
                "contract_hash": incompatible.contract_hash,
                "encoding_hash": incompatible.encoding_hash,
                "observation_size": incompatible.observation_size,
                "action_size": incompatible.action_size,
            },
        )

    assert pending.exists()
    assert list((run / "checkpoints").iterdir()) == []
    assert read_json(run / "run.json")["latest_checkpoint"] is None
