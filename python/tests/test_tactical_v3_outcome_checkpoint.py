from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from ml_lab.io import atomic_write_json
from ml_lab.tactical_v3_checkpoint import _state_sha256, semantic_identity_wire
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_outcome_checkpoint import (
    OUTCOME_ALGORITHM,
    OutcomeCheckpointMetadata,
    load_outcome_checkpoint,
    outcome_model_state_sha256,
    replace_outcome_checkpoint,
    save_outcome_checkpoint,
    validate_outcome_run,
)
from ml_lab.tactical_v3_schema import (
    TacticalV3Decision,
    TacticalV3Reward,
    TacticalV3SemanticIdentity,
)
from ml_lab.tactical_v3_trajectory import (
    ControllerProvenance,
    TacticalV3TrajectoryGame,
    TrajectoryDecisionRecord,
    publish_trajectory_game,
    write_trajectory_manifest,
)
from tests.tactical_v3_fixture_support import load_duel_identity_fixture
from tests.test_tactical_v3_model import canonical_model_example


def _case(tmp_path: Path):
    identity = load_duel_identity_fixture()
    config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(4, 8, 16),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(19)
        model = TacticalV3Policy(config).cpu().eval()
    trajectory = (
        json.dumps({
            "schema_version": 1,
            "kind": "tactical-v3-learner-trajectory-archive",
            "identity": {
                "scenario_id": identity.scenario_id,
                "scenario_schema_version": identity.scenario_schema_version,
                "contract_version": identity.contract_version,
                "environment_kind": identity.environment_kind,
                "contract_hash": identity.contract_hash,
                "encoding_hash": identity.encoding_hash,
                "capacity_hash": identity.capacity_hash,
            },
            "partitions": {
                "train": {"dataset_use": "optimization", "games": []},
                "validation": {
                    "dataset_use": "early_stop_only", "games": [],
                },
            },
        }, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    metadata = OutcomeCheckpointMetadata(
        format_version=1,
        algorithm=OUTCOME_ALGORITHM,
        identity=identity,
        model_config=config,
        trajectory_manifest_sha256=hashlib.sha256(trajectory).hexdigest(),
        model_state_sha256=outcome_model_state_sha256(model),
        update=3,
        validation_game_start=6,
        validation_games=2,
        validation_opponent_artifact_sha256="b" * 64,
        validation_win_rate=0.5,
        validation_mean_return=0.0,
        validation_mean_decisions=1.0,
        initialization={
            "kind": "scratch",
            "seed": 19,
            "model_state_sha256": outcome_model_state_sha256(model),
        },
        published_device="cpu",
    )
    decision = canonical_model_example().decision
    return identity, model, metadata, decision, trajectory


def _validation_game(
    model: TacticalV3Policy,
    *,
    actor_source: Path,
    game_index: int,
    learner_seat: int,
    learner_won: bool,
) -> TacticalV3TrajectoryGame:
    decision = replace(canonical_model_example().decision, seat=learner_seat)
    winner = learner_seat if learner_won else 1 - learner_seat
    terminal = 1.0 if learner_won else -1.0
    reward = TacticalV3Reward(
        terminal_outcome=terminal,
        known_health_adjusted_material_progress=0.0,
        public_resource_progress=0.0,
        time_pressure=0.0,
        total=terminal,
        finalized=True,
    )
    return TacticalV3TrajectoryGame(
        identity=load_duel_identity_fixture(),
        partition="validation",
        game_index=game_index,
        episode_seed=500_000_000 + game_index // 2,
        profile_id="conversion-1v1-near",
        learner_seat=learner_seat,
        reference_seat=learner_seat,
        actor=ControllerProvenance(
            "model", OUTCOME_ALGORITHM, str(actor_source.resolve()),
            outcome_model_state_sha256(model),
        ),
        opponent=ControllerProvenance(
            "scripted", "passive", "GymServer:passive", "b" * 64,
        ),
        records=(TrajectoryDecisionRecord(
            trajectory_index=0,
            decision=decision,
            selected_candidate_id=decision.candidates[0].candidate_id,
            behavior_mode="greedy",
            log_probability=0.0,
            entropy=0.0,
            successor_reward=reward,
            terminated_after_selection=True,
            truncated_after_selection=False,
        ),),
        replay=(
            "HEXWARS-REPLAY 1\n"
            "META 3 0 1 0 0 0\n"
            "CMDS 1\n"
            "E 0\n"
        ).encode("utf-8"),
        winner=winner,
        terminated=True,
        truncated=False,
        terminal_reward=reward,
        internal_fallback_count=0,
    )


@dataclass(frozen=True, slots=True)
class _ValidatedRunCase:
    run: Path
    identity: TacticalV3SemanticIdentity
    model: TacticalV3Policy
    metadata: OutcomeCheckpointMetadata
    checkpoint_path: Path
    snapshot_path: Path
    validation_fixture: tuple[TacticalV3Decision, ...]


def _validated_run_case(
    tmp_path: Path,
    *,
    initialization: dict[str, object] | None = None,
) -> _ValidatedRunCase:
    identity, model, metadata, decision, _ = _case(tmp_path)
    if initialization is not None:
        metadata = replace(metadata, initialization=initialization)
    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "trajectories" / "train").mkdir(parents=True)
    (run / "trajectories" / "validation").mkdir()
    checked_in_scenario = (
        Path(__file__).parents[1]
        / "config"
        / "annihilation-structured-imitation-v1.json"
    )
    (run / "scenario.json").write_bytes(checked_in_scenario.read_bytes())
    atomic_write_json(run / "policy-identity.json", semantic_identity_wire(identity))
    for game_index in range(8):
        publish_trajectory_game(
            run / "trajectories",
            _validation_game(
                model,
                actor_source=run,
                game_index=game_index,
                learner_seat=game_index % 2,
                learner_won=game_index % 2 == 0,
            ),
        )
    live_manifest = write_trajectory_manifest(run / "trajectories", identity)
    trajectory = live_manifest.read_bytes()
    metadata = replace(
        metadata,
        trajectory_manifest_sha256=hashlib.sha256(trajectory).hexdigest(),
    )
    snapshot_path = (
        run / "checkpoints" / "trajectory-manifest-update-000003.json"
    )
    checkpoint_path = run / "checkpoints" / "policy-update-000003.pt"
    snapshot_path.write_bytes(trajectory)
    validation_fixture = (decision, replace(decision, seat=1))
    replace_outcome_checkpoint(
        checkpoint_path, model, metadata, validation_fixture,
    )
    initialization_source = (
        metadata.initialization.get("run")
        if metadata.initialization.get("kind") == "structured-policy-run"
        else None
    )
    atomic_write_json(run / "run.json", {
        "schema_version": 1,
        "state": "running",
        "evidence_status": "unsealed-experimental",
        "config": {
            "backend": OUTCOME_ALGORITHM,
            "algorithm": OUTCOME_ALGORITHM,
            "validation_games": 2,
            "initialization_source": initialization_source,
        },
        "initialization": dict(metadata.initialization),
        "contract": {
            "environment": "tactical-v3",
            "version": identity.contract_version,
            "environment_kind": identity.environment_kind,
            "contract_hash": identity.contract_hash,
            "encoding_hash": identity.encoding_hash,
            "capacity_hash": identity.capacity_hash,
        },
        "policy_identity": "policy-identity.json",
        "latest_checkpoint": "checkpoints/policy-update-000003.pt",
        "latest_checkpoint_step": 3,
        "best_trajectory_manifest": (
            "checkpoints/trajectory-manifest-update-000003.json"
        ),
        "best_update": 3,
        "best_validation_win_rate": 0.5,
        "best_validation_mean_return": 0.0,
        "best_validation_mean_decisions": 1.0,
        "scenario": {"path": "scenario.json", "schema_version": 1},
        "validation_opponent_snapshot": {
            "kind": "scripted",
            "name": "passive",
            "source": "GymServer:passive",
            "artifact_sha256": "b" * 64,
        },
    })
    return _ValidatedRunCase(
        run,
        identity,
        model,
        metadata,
        checkpoint_path,
        snapshot_path,
        validation_fixture,
    )


def test_outcome_checkpoint_roundtrips_with_its_own_algorithm_identity(
    tmp_path: Path,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )

    loaded = load_outcome_checkpoint(
        path, identity.encoding_hash, identity.capacity_hash,
    )

    assert loaded.metadata == metadata
    assert loaded.metadata.algorithm == "structured_policy_gradient"
    assert loaded.fixture.decisions == (decision,)
    assert loaded.fixture.selected_identities[0].candidate_id in {
        candidate.candidate_id for candidate in decision.candidates
    }
    assert outcome_model_state_sha256(loaded.model) == metadata.model_state_sha256


def test_outcome_checkpoint_allows_validation_less_often_than_optimizer_updates(
    tmp_path: Path,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    metadata = replace(metadata, update=8)

    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )

    loaded = load_outcome_checkpoint(
        path, identity.encoding_hash, identity.capacity_hash,
    )
    assert loaded.metadata.update == 8
    assert loaded.metadata.validation_game_start == 6


def test_outcome_checkpoint_rejects_validation_start_inside_a_sweep(
    tmp_path: Path,
) -> None:
    _identity, model, metadata, decision, _ = _case(tmp_path)

    with pytest.raises(ValueError, match="sweep schedule"):
        save_outcome_checkpoint(
            tmp_path / "best.pt",
            model,
            replace(metadata, validation_game_start=5),
            (decision,),
        )


def test_live_outcome_checkpoint_remains_loadable_while_stopping(
    tmp_path: Path,
) -> None:
    case = _validated_run_case(tmp_path)
    manifest = json.loads(case.run.joinpath("run.json").read_text(encoding="utf-8"))
    manifest["state"] = "stopping"
    atomic_write_json(case.run / "run.json", manifest)

    loaded = validate_outcome_run(case.run)

    assert loaded.metadata == case.metadata


def test_outcome_checkpoint_rejects_state_tampering(tmp_path: Path) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    first = next(iter(payload["state_dict"]))
    payload["state_dict"][first].view(-1)[0] += 1.0
    torch.save(payload, path)

    with pytest.raises(ValueError, match="model state hash"):
        load_outcome_checkpoint(path, identity.encoding_hash, identity.capacity_hash)


def test_outcome_checkpoint_rejects_malformed_initialization_provenance(
    tmp_path: Path,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )
    valid_bytes = path.read_bytes()
    source = {
        "kind": "structured-policy-run",
        "algorithm": "structured_imitation",
        "run": "/tmp/source-run",
        "checkpoint": "/tmp/source-run/checkpoints/best.pt",
        "checkpoint_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
        "source_identity": semantic_identity_wire(identity),
    }
    cases = (
        {"kind": "scratch", "seed": 19},
        {
            "kind": "scratch",
            "seed": True,
            "model_state_sha256": "a" * 64,
        },
        {
            "kind": "scratch",
            "seed": 20_001,
            "model_state_sha256": "a" * 64,
        },
        {
            "kind": "scratch",
            "seed": 19,
            "model_state_sha256": "not-a-hash",
        },
        source | {"unexpected": True},
        source | {"algorithm": "maskable_ppo"},
        source | {"run": ""},
        source | {"checkpoint_sha256": "not-a-hash"},
        source | {"source_identity": {}},
    )
    for initialization in cases:
        path.write_bytes(valid_bytes)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["metadata"]["initialization"] = initialization
        torch.save(payload, path)
        with pytest.raises((TypeError, ValueError), match="initialization"):
            load_outcome_checkpoint(
                path, identity.encoding_hash, identity.capacity_hash,
            )


@pytest.mark.parametrize("nonfinite", (float("nan"), float("inf")))
def test_outcome_checkpoint_rejects_hashed_nonfinite_state(
    tmp_path: Path,
    nonfinite: float,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = next(
        key for key, value in payload["state_dict"].items()
        if value.is_floating_point()
    )
    payload["state_dict"][name].view(-1)[0] = nonfinite
    payload["metadata"]["model_state_sha256"] = _state_sha256(
        payload["state_dict"]
    )
    torch.save(payload, path)

    with pytest.raises(ValueError, match="state tensor.*nonfinite"):
        load_outcome_checkpoint(
            path, identity.encoding_hash, identity.capacity_hash,
        )


def test_outcome_checkpoint_rejects_state_silently_cast_by_load_state_dict(
    tmp_path: Path,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = save_outcome_checkpoint(
        tmp_path / "best.pt", model, metadata, (decision,),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    name = next(
        key for key, value in payload["state_dict"].items()
        if value.is_floating_point()
    )
    payload["state_dict"][name] = payload["state_dict"][name].double()
    payload["metadata"]["model_state_sha256"] = _state_sha256(
        payload["state_dict"]
    )
    torch.save(payload, path)

    with pytest.raises(ValueError, match="loaded outcome model state hash"):
        load_outcome_checkpoint(
            path, identity.encoding_hash, identity.capacity_hash,
        )


def test_live_outcome_checkpoint_is_replaced_only_after_validation(
    tmp_path: Path,
) -> None:
    identity, model, metadata, decision, _ = _case(tmp_path)
    path = tmp_path / "checkpoints" / "best.pt"
    replace_outcome_checkpoint(path, model, metadata, (decision,))
    before = path.read_bytes()

    invalid = replace(metadata, model_state_sha256="0" * 64)
    with pytest.raises(ValueError, match="state hash"):
        replace_outcome_checkpoint(path, model, invalid, (decision,))

    assert path.read_bytes() == before
    assert load_outcome_checkpoint(
        path, identity.encoding_hash, identity.capacity_hash,
    ).metadata == metadata


def test_live_outcome_checkpoint_survives_staged_reopen_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_outcome_checkpoint as checkpoint_module

    identity, model, metadata, decision, _ = _case(tmp_path)
    path = tmp_path / "checkpoints" / "best.pt"
    replace_outcome_checkpoint(path, model, metadata, (decision,))
    before = path.read_bytes()

    def reject_staged_checkpoint(*_args, **_kwargs):
        raise ValueError("simulated corrupt staged checkpoint")

    monkeypatch.setattr(
        checkpoint_module, "load_outcome_checkpoint", reject_staged_checkpoint,
    )
    with pytest.raises(ValueError, match="corrupt staged checkpoint"):
        replace_outcome_checkpoint(path, model, metadata, (decision,))

    assert path.read_bytes() == before


def test_outcome_run_rejects_top_level_initialization_tampering(
    tmp_path: Path,
) -> None:
    case = _validated_run_case(tmp_path)
    manifest = json.loads(case.run.joinpath("run.json").read_text(encoding="utf-8"))
    manifest["initialization"]["seed"] += 1
    atomic_write_json(case.run / "run.json", manifest)

    with pytest.raises(ValueError, match="initialization does not match checkpoint"):
        validate_outcome_run(case.run)


def test_outcome_run_rejects_source_initialization_path_tampering(
    tmp_path: Path,
) -> None:
    source = (tmp_path / "source-run").resolve()
    initialization = {
        "kind": "structured-policy-run",
        "algorithm": "structured_imitation",
        "run": str(source),
        "checkpoint": str(source / "checkpoints" / "best.pt"),
        "checkpoint_sha256": "c" * 64,
        "model_state_sha256": "d" * 64,
        "source_identity": semantic_identity_wire(load_duel_identity_fixture()),
    }
    case = _validated_run_case(tmp_path, initialization=initialization)
    assert validate_outcome_run(case.run).metadata == case.metadata
    manifest = json.loads(case.run.joinpath("run.json").read_text(encoding="utf-8"))
    manifest["config"]["initialization_source"] = str(tmp_path / "other-run")
    atomic_write_json(case.run / "run.json", manifest)

    with pytest.raises(ValueError, match="initialization source.*checkpoint"):
        validate_outcome_run(case.run)


def test_outcome_run_binds_checkpoint_identity_and_trajectory_snapshot(
    tmp_path: Path,
) -> None:
    case = _validated_run_case(tmp_path)
    run = case.run
    identity = case.identity
    model = case.model
    metadata = case.metadata
    checkpoint_path = case.checkpoint_path
    snapshot_path = case.snapshot_path
    validation_fixture = case.validation_fixture

    assert validate_outcome_run(run).metadata == metadata
    from ml_lab.controllers import ControllerResolutionError, ControllerResolver
    from ml_lab.tactical_v3_controller import load_structured_controller

    controller = load_structured_controller(
        run, identity.encoding_hash, identity.capacity_hash,
    )
    assert controller.algorithm == OUTCOME_ALGORITHM
    resolved = ControllerResolver(
        expected_structured_hashes=(
            identity.encoding_hash, identity.capacity_hash,
        )
    ).resolve(f"run:{run}")
    assert resolved.algorithm == OUTCOME_ALGORITHM
    assert resolved.path == checkpoint_path.resolve()
    with pytest.raises(ControllerResolutionError, match="deterministic inference"):
        ControllerResolver(
            expected_structured_hashes=(
                identity.encoding_hash, identity.capacity_hash,
            )
        ).resolve({
            "kind": "run",
            "path": str(run),
            "inference_mode": "stochastic",
        })

    dishonest = replace(metadata, validation_win_rate=0.75)
    replace_outcome_checkpoint(
        checkpoint_path, model, dishonest, validation_fixture,
    )
    run_manifest = json.loads((run / "run.json").read_text(encoding="utf-8"))
    run_manifest["best_validation_win_rate"] = 0.75
    atomic_write_json(run / "run.json", run_manifest)
    with pytest.raises(ValueError, match="win_rate.*validation games"):
        validate_outcome_run(run)

    replace_outcome_checkpoint(
        checkpoint_path, model, metadata, validation_fixture,
    )
    run_manifest["best_validation_win_rate"] = 0.5
    atomic_write_json(run / "run.json", run_manifest)

    with snapshot_path.open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="trajectory manifest hash"):
        validate_outcome_run(run)
