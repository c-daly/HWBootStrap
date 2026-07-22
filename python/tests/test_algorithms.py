from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from ml_lab.algorithms import (
    create_or_resume_model,
    get_algorithm_adapter,
    resolve_resume_checkpoint,
)
from ml_lab.contracts import ContractMismatch, EnvironmentContract, RunConfig, create_run


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v1",
        contract_hash="a" * 64,
        encoding_hash="b" * 64,
        observation_size=12,
        action_size=7,
        board={"width": 3, "height": 2},
        roster=["1,2,3,4,5,6,7,8,9"],
        reward={"terminal_win": 1.0},
    )


def run_config(run_name: str, algorithm: str) -> RunConfig:
    return RunConfig(
        backend="sb3",
        algorithm=algorithm,
        policy="HexCNN" if algorithm == "maskable_ppo" else "MlpPolicy",
        run_name=run_name,
        seed=11,
        total_timesteps=128,
        checkpoint_interval=32,
        workers=1,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "greedy"},
        trackers=[{"kind": "local"}],
        resume_source=None,
    )


def test_algorithm_registry_defaults_to_verified_maskable_ppo() -> None:
    adapter = get_algorithm_adapter(None)

    assert adapter.name == "maskable_ppo"
    assert adapter.policy_name == "HexCNN"
    assert adapter.experimental is False


def test_algorithm_registry_marks_masked_dqn_experimental() -> None:
    adapter = get_algorithm_adapter("masked_dqn")

    assert adapter.name == "masked_dqn"
    assert adapter.policy_name == "MlpPolicy"
    assert adapter.experimental is True


def test_algorithm_registry_rejects_unknown_algorithm() -> None:
    with pytest.raises(ValueError, match="unsupported algorithm"):
        get_algorithm_adapter("rainbow")


def test_model_geometry_accepts_gymnasium_numpy_integer_action_count(
    contract: EnvironmentContract,
) -> None:
    model = type(
        "Model",
        (),
        {
            "observation_space": type("ObservationSpace", (), {"shape": (12,)})(),
            "action_space": type("ActionSpace", (), {"n": np.int64(7)})(),
        },
    )()

    get_algorithm_adapter("maskable_ppo").validate_model(model, contract)


def test_model_geometry_rejects_boolean_action_count(
    contract: EnvironmentContract,
) -> None:
    model = type(
        "Model",
        (),
        {
            "observation_space": type("ObservationSpace", (), {"shape": (12,)})(),
            "action_space": type("ActionSpace", (), {"n": True})(),
        },
    )()

    with pytest.raises(ContractMismatch, match="discrete action space"):
        get_algorithm_adapter("maskable_ppo").validate_model(model, contract)


def test_maskable_ppo_adapter_predicts_with_the_legal_action_mask() -> None:
    received: list[dict[str, object]] = []

    class Model:
        def predict(self, observation, **kwargs):
            received.append({"observation": observation, **kwargs})
            return np.asarray(1), None

    observation = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)
    mask = np.asarray([False, True], dtype=bool)

    action = get_algorithm_adapter("maskable_ppo").predict(Model(), observation, mask)

    assert action == 1
    assert received[0]["observation"] is observation
    assert received[0]["action_masks"] is mask
    assert received[0]["deterministic"] is True


def test_masked_dqn_adapter_predicts_only_among_legal_values() -> None:
    import torch

    class Model:
        device = "cpu"

        @staticmethod
        def q_net(_observation):
            return torch.asarray([[1.0, 9.0]])

    action = get_algorithm_adapter("masked_dqn").predict(
        Model(),
        np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
        np.asarray([True, False], dtype=bool),
    )

    assert action == 0


class FakeAdapter:
    name = "maskable_ppo"
    policy_name = "HexCNN"
    experimental = False

    def __init__(self) -> None:
        self.created: list[dict[str, object]] = []
        self.loaded: list[tuple[Path, object, str]] = []

    def create(self, env, **kwargs):
        self.created.append({"env": env, **kwargs})
        return "fresh-model"

    def load(self, path: Path, *, env, device: str):
        self.loaded.append((path, env, device))
        return "resumed-model"

    def validate_model(self, model, expected_contract: EnvironmentContract) -> None:
        assert model in {"fresh-model", "resumed-model"}
        assert expected_contract.observation_size == 12


def test_create_or_resume_model_builds_fresh_policy_with_handshake(
    contract: EnvironmentContract,
) -> None:
    adapter = FakeAdapter()
    env = object()
    spaces_info = {"channels": 3, "board_h": 2, "board_w": 3, "globals": 4}

    model, resumed = create_or_resume_model(
        adapter,
        env=env,
        expected_contract=contract,
        spaces_info=spaces_info,
        seed=17,
        device="cpu",
        checkpoint_interval=32,
        resume_source=None,
    )

    assert model == "fresh-model"
    assert resumed is False
    assert adapter.created == [
        {
            "env": env,
            "spaces_info": spaces_info,
            "seed": 17,
            "device": "cpu",
            "checkpoint_interval": 32,
        }
    ]
    assert adapter.loaded == []


def test_create_or_resume_model_loads_validated_run_checkpoint(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source_run = create_run(tmp_path, run_config("source", "maskable_ppo"), contract)
    checkpoint = source_run / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"model")
    manifest_path = source_run / "run.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latest_checkpoint"] = "checkpoints/step_000000064.zip"
    manifest["latest_checkpoint_step"] = 64
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    adapter = FakeAdapter()
    env = object()

    model, resumed = create_or_resume_model(
        adapter,
        env=env,
        expected_contract=contract,
        spaces_info={},
        seed=17,
        device="cpu",
        checkpoint_interval=32,
        resume_source=source_run,
    )

    assert model == "resumed-model"
    assert resumed is True
    assert adapter.loaded == [(checkpoint.resolve(), env, "cpu")]
    assert adapter.created == []


def test_resume_rejects_algorithm_mismatch(tmp_path: Path, contract: EnvironmentContract) -> None:
    source_run = create_run(tmp_path, run_config("source", "masked_dqn"), contract)

    with pytest.raises(ValueError, match="algorithm"):
        resolve_resume_checkpoint(source_run, "maskable_ppo", contract)


def test_resume_rejects_full_contract_mismatch_even_when_hash_matches(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    different_contract = replace(contract, reward={"terminal_win": 2.0})
    source_run = create_run(tmp_path, run_config("source", "maskable_ppo"), different_contract)

    with pytest.raises(ContractMismatch, match="training contract"):
        resolve_resume_checkpoint(source_run, "maskable_ppo", contract)


def test_unified_resume_rejects_raw_checkpoint_without_authoritative_metadata(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    checkpoint = tmp_path / "standalone.zip"
    checkpoint.write_bytes(b"model")
    adapter = FakeAdapter()

    with pytest.raises(ValueError, match="metadata-backed run directory"):
        create_or_resume_model(
            adapter,
            env=object(),
            expected_contract=contract,
            spaces_info={},
            seed=1,
            device="cpu",
            checkpoint_interval=32,
            resume_source=checkpoint,
        )


def test_explicit_unsafe_legacy_resume_is_no_longer_supported(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    checkpoint = tmp_path / "standalone.zip"
    checkpoint.write_bytes(b"model")
    adapter = FakeAdapter()

    with pytest.raises(ValueError, match="standalone checkpoint resume is unsupported"):
        create_or_resume_model(
            adapter,
            env=object(),
            expected_contract=contract,
            spaces_info={},
            seed=1,
            device="cpu",
            checkpoint_interval=32,
            resume_source=checkpoint,
            allow_unsafe_legacy_resume=True,
        )


def test_masked_dqn_resume_is_rejected_until_replay_buffer_sidecars_exist(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    source_run = create_run(tmp_path, run_config("dqn-source", "masked_dqn"), contract)
    checkpoint = source_run / "checkpoints" / "step_000000064.zip"
    checkpoint.write_bytes(b"model")
    manifest_path = source_run / "run.json"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latest_checkpoint"] = "checkpoints/step_000000064.zip"
    manifest["latest_checkpoint_step"] = 64
    manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
    adapter = FakeAdapter()
    adapter.name = "masked_dqn"

    with pytest.raises(ValueError, match="replay buffer"):
        create_or_resume_model(
            adapter,
            env=object(),
            expected_contract=contract,
            spaces_info={},
            seed=1,
            device="cpu",
            checkpoint_interval=32,
            resume_source=source_run,
        )
