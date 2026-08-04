from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium import spaces

import ml_lab.algorithms as algorithms_module
from ml_lab.algorithms import (
    MaskablePPOAdapter,
    create_or_resume_model,
    get_algorithm_adapter,
    resolve_resume_checkpoint,
)
from ml_lab.contracts import (
    ContractMismatch,
    EnvironmentContract,
    RunConfig,
    create_run as create_durable_run,
)
from ml_lab.scenarios import resolve_scenario
from ml_lab.controllers import ControllerSpec, ResolvedController


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


class _TinyActorEnv(gym.Env):
    observation_space = spaces.Box(
        low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32,
    )
    action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.asarray([True, False, True, False, False], dtype=bool)


def _actor_contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v2",
        contract_hash="c" * 64,
        encoding_hash="d" * 64,
        observation_size=3,
        action_size=5,
        board={"width": 1, "height": 1},
        roster=[],
        reward={},
    )


def _actor_model(seed: int):
    return MaskablePPOAdapter().create(
        _TinyActorEnv(),
        spaces_info={
            "channels": 1, "board_h": 1, "board_w": 1, "globals": 2,
        },
        seed=seed,
        device="cpu",
        checkpoint_interval=2,
    )


def _actor_modules(model):
    return (
        model.policy.features_extractor,
        model.policy.mlp_extractor.policy_net,
        model.policy.action_net,
    )


def _value_parameters(model):
    return tuple(
        parameter
        for module in (
            model.policy.mlp_extractor.value_net,
            model.policy.value_net,
        )
        for parameter in module.parameters()
    )


def _resolved_actor_source(
    tmp_path: Path,
    model,
    *,
    source_kind: str,
    step: int = 7,
) -> tuple[object, ResolvedController]:
    run = tmp_path / f"{source_kind}-source"
    checkpoint = run / "checkpoints" / f"step_{step:09d}.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"{source_kind}-{step}".encode("ascii"))
    manifest = {
        "schema_version": 1,
        "state": "completed" if source_kind == "dagger_actor" else "stopped",
        "latest_checkpoint": checkpoint.relative_to(run).as_posix(),
        "latest_checkpoint_step": step,
        "config": {"algorithm": "maskable_ppo", "policy": "HexCNN"},
        "contract": _actor_contract().to_dict(),
    }
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    if source_kind == "dagger_actor":
        (run / "bc.json").write_text(
            json.dumps({
                "schema_version": 1,
                "training_kind": "selective-dagger-distillation-v1",
                "distillation_iteration": 1,
                "algorithm": "maskable_ppo",
                "policy": "HexCNN",
                "checkpoint_sha256": hashlib.sha256(
                    checkpoint.read_bytes()
                ).hexdigest(),
            }),
            encoding="utf-8",
        )
    controller = {
        "kind": "snapshot",
        "path": str(checkpoint.resolve()),
        "source_run": str(run.resolve()),
        "algorithm": "maskable_ppo",
        "step": step,
        "inference_mode": "deterministic",
    }
    source = algorithms_module.ActorTransferSource(
        source_kind=source_kind,
        controller=controller,
        checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    )
    resolved = ResolvedController(
        spec=ControllerSpec(
            kind="snapshot",
            path=checkpoint.resolve(),
            source_run=run.resolve(),
            algorithm="maskable_ppo",
            step=step,
            inference_mode="deterministic",
        ),
        server_controller="external",
        model=model,
        path=checkpoint.resolve(),
        algorithm="maskable_ppo",
        step=step,
        contract=_actor_contract(),
        observation_size=3,
        action_size=5,
        legacy=False,
        promotable=True,
    )
    return source, resolved


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
        environment="tactical-v1",
    )


def create_run(
    runs_root: Path,
    config: RunConfig,
    contract: EnvironmentContract,
) -> Path:
    scenario = resolve_scenario(
        environment=config.environment,
        scenario_file=None,
        template_id="tactical-standard",
    )
    return create_durable_run(
        runs_root,
        config,
        contract,
        scenario,
        opponent_snapshot=config.opponent,
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


@pytest.mark.parametrize("source_kind", ["snapshot", "dagger_actor"])
def test_actor_transfer_copies_only_actor_and_shared_features_from_both_source_kinds(
    tmp_path: Path, source_kind: str,
) -> None:
    """Loading full policy state would overwrite the target's fresh value head."""

    source_model = _actor_model(101)
    target_model = _actor_model(227)
    with torch.no_grad():
        for parameter in (
            parameter for module in _actor_modules(source_model)
            for parameter in module.parameters()
        ):
            parameter.fill_(0.25)
        for parameter in _value_parameters(source_model):
            parameter.fill_(-0.75)
        for parameter in _value_parameters(target_model):
            parameter.fill_(0.875)
    source, resolved = _resolved_actor_source(
        tmp_path, source_model, source_kind=source_kind,
    )
    value_before = tuple(
        parameter.detach().clone() for parameter in _value_parameters(target_model)
    )
    target_before = algorithms_module.actor_state_sha256(target_model)

    provenance = MaskablePPOAdapter().initialize_actor_from_resolved(
        target_model,
        resolved,
        source=source,
        expected_contract=_actor_contract(),
        device="cpu",
    )

    assert algorithms_module.actor_state_sha256(target_model) == (
        algorithms_module.actor_state_sha256(source_model)
    )
    assert provenance["source_kind"] == source_kind
    assert provenance["target_actor_sha256_before"] == target_before
    assert provenance["source_actor_sha256"] == provenance[
        "target_actor_sha256_after"
    ]
    assert all(
        torch.equal(parameter.detach(), original)
        for parameter, original in zip(
            _value_parameters(target_model), value_before, strict=True,
        )
    )


def test_actor_transfer_preflights_all_tensor_shapes_before_mutating_target(
    tmp_path: Path,
) -> None:
    """Copying modules as they are checked would leave a half-copied actor."""

    source_model = _actor_model(101)
    target_model = _actor_model(227)
    source_model.policy.action_net = torch.nn.Linear(64, 6)
    source, resolved = _resolved_actor_source(
        tmp_path, source_model, source_kind="snapshot",
    )
    before = algorithms_module.actor_state_sha256(target_model)

    with pytest.raises(ContractMismatch, match="shape"):
        MaskablePPOAdapter().initialize_actor_from_resolved(
            target_model,
            resolved,
            source=source,
            expected_contract=_actor_contract(),
            device="cpu",
        )

    assert algorithms_module.actor_state_sha256(target_model) == before


def test_actor_transfer_restores_the_complete_target_after_a_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure after earlier modules copy must roll the target back transactionally."""

    source_model = _actor_model(101)
    target_model = _actor_model(227)
    source, resolved = _resolved_actor_source(
        tmp_path, source_model, source_kind="snapshot",
    )
    before = algorithms_module.actor_state_sha256(target_model)
    original_load = target_model.policy.action_net.load_state_dict
    attempts = 0

    def fail_first_load(state_dict, *, strict=True, assign=False):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected actor-copy failure")
        return original_load(state_dict, strict=strict, assign=assign)

    monkeypatch.setattr(
        target_model.policy.action_net, "load_state_dict", fail_first_load,
    )
    with pytest.raises(RuntimeError, match="injected actor-copy failure"):
        MaskablePPOAdapter().initialize_actor_from_resolved(
            target_model,
            resolved,
            source=source,
            expected_contract=_actor_contract(),
            device="cpu",
        )

    assert attempts == 2
    assert algorithms_module.actor_state_sha256(target_model) == before


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("hash", "SHA-256"),
        ("contract", "contract"),
        ("policy", "policy class"),
        ("algorithm", "algorithm"),
        ("mode", "deterministic"),
        ("step", "step"),
        ("containment", "source run"),
    ],
)
def test_malformed_actor_transfer_sources_leave_the_target_untouched(
    tmp_path: Path, mutation: str, message: str,
) -> None:
    """Relaxing source identity checks must never permit a partial or wrong warm-start."""

    source_model = _actor_model(101)
    target_model = _actor_model(227)
    source, resolved = _resolved_actor_source(
        tmp_path, source_model, source_kind="snapshot",
    )
    if mutation == "hash":
        source = replace(source, checkpoint_sha256="f" * 64)
    elif mutation == "contract":
        resolved = replace(
            resolved,
            contract=replace(_actor_contract(), contract_hash="e" * 64),
        )
    elif mutation == "policy":
        source_model.policy = torch.nn.Identity()
    elif mutation == "algorithm":
        resolved = replace(resolved, algorithm="masked_dqn")
    elif mutation == "mode":
        resolved = replace(
            resolved,
            spec=replace(resolved.spec, inference_mode="stochastic"),
        )
    elif mutation == "step":
        resolved = replace(resolved, step=8)
    elif mutation == "containment":
        outside = tmp_path / "outside" / resolved.path.name
        outside.parent.mkdir()
        outside.write_bytes(resolved.path.read_bytes())
        source = replace(
            source,
            controller={**source.controller, "path": str(outside.resolve())},
            checkpoint_sha256=hashlib.sha256(outside.read_bytes()).hexdigest(),
        )
        resolved = replace(
            resolved,
            path=outside.resolve(),
            spec=replace(resolved.spec, path=outside.resolve()),
        )
    else:
        raise AssertionError(mutation)
    before = algorithms_module.actor_state_sha256(target_model)

    with pytest.raises((ValueError, ContractMismatch), match=message):
        MaskablePPOAdapter().initialize_actor_from_resolved(
            target_model,
            resolved,
            source=source,
            expected_contract=_actor_contract(),
            device="cpu",
        )

    assert algorithms_module.actor_state_sha256(target_model) == before
