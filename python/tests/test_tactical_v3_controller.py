from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import pytest

from ml_lab.tactical_v3_checkpoint import publish_structured_run
from ml_lab.tactical_v3_corpus import load_corpus
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_schema import TacticalV3SemanticIdentity, TacticalV3View, parse_view
from ml_lab.tactical_v3_training import EpochMetrics, TrainerConfig, TrainingResult
from tests.tactical_v3_fixture_support import (
    DUEL_IDENTITY_FIXTURE,
    TINY_CORPUS_ROOT,
    load_duel_identity_fixture,
)


@dataclass(frozen=True, slots=True)
class StructuredRunCase:
    run_dir: Path
    identity: TacticalV3SemanticIdentity
    view: TacticalV3View


def make_structured_run_case(tmp_path: Path) -> StructuredRunCase:
    identity = load_duel_identity_fixture()
    corpus = load_corpus(TINY_CORPUS_ROOT, identity)
    model_config = TacticalV3ModelConfig(
        hidden_dim=16,
        categorical_dim=4,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=32,
        candidate_hidden_dim=32,
        horizon_turns=(4, 8, 16),
    )
    objective_config = ObjectiveConfig()
    trainer_config = TrainerConfig(seed=0, device="cpu")
    metrics = MappingProxyType({
        "total": 0.0,
        "policy": 0.0,
        "outcome": 0.0,
        "horizon": 0.0,
        "remaining_turns": 0.0,
    })
    result = TrainingResult(
        model=TacticalV3Policy(model_config).eval(),
        model_config=model_config,
        objective_config=objective_config,
        trainer_config=trainer_config,
        best_epoch=0,
        best_validation_policy_nll=0.0,
        stopped_early=False,
        history=(EpochMetrics(0, metrics, metrics, 0.0, True),),
    )
    run_dir = publish_structured_run(tmp_path / "run", result, corpus, DUEL_IDENTITY_FIXTURE)
    payload = json.loads(
        (Path(__file__).parent / "fixtures" / "tactical_v3" / "seed-41-decision.json").read_text(
            encoding="utf-8"
        )
    )
    return StructuredRunCase(run_dir, identity, parse_view(payload, identity))


def test_load_structured_controller_is_cpu_eval_and_selects_exact_legal_identity(
    tmp_path: Path,
) -> None:
    from ml_lab.tactical_v3_controller import load_structured_controller, select_candidate

    case = make_structured_run_case(tmp_path)
    first = load_structured_controller(
        case.run_dir, case.identity.encoding_hash, case.identity.capacity_hash
    )
    second = load_structured_controller(
        case.run_dir, case.identity.encoding_hash, case.identity.capacity_hash
    )

    assert first.run_dir == case.run_dir.resolve()
    assert first.checkpoint_path == case.run_dir / "checkpoints" / "best.pt"
    assert first.identity == case.identity
    assert first.policy.training is False
    assert next(first.policy.parameters()).device.type == "cpu"
    assert select_candidate(first, case.view) == select_candidate(second, case.view)
    selected = select_candidate(first, case.view)
    assert selected.decision_id == case.view.decision.decision_id
    assert sum(
        candidate.candidate_id == selected.candidate_id
        for candidate in case.view.decision.candidates
    ) == 1


def test_load_structured_controller_matches_hashes_before_checkpoint_tensor_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_controller as controller_module

    case = make_structured_run_case(tmp_path)
    called = False

    def should_not_load(_run_dir: Path) -> object:
        nonlocal called
        called = True
        raise AssertionError("checkpoint tensor loader must not run")

    monkeypatch.setattr(controller_module, "validate_structured_run", should_not_load)
    with pytest.raises(ValueError, match="encoding hash"):
        controller_module.load_structured_controller(
            case.run_dir, "0" * 64, case.identity.capacity_hash
        )
    assert called is False


def test_load_structured_controller_rejects_manifest_checkpoint_outside_checkpoints(
    tmp_path: Path,
) -> None:
    from ml_lab.tactical_v3_controller import load_structured_controller

    case = make_structured_run_case(tmp_path)
    manifest_path = case.run_dir / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latest_checkpoint"] = "scenario.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoints.*pt"):
        load_structured_controller(
            case.run_dir, case.identity.encoding_hash, case.identity.capacity_hash
        )
