from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from typing import Literal

import pytest
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ml_lab.tactical_v3_batching import (
    CandidateBatch,
    RaggedBatch,
    collate_decisions,
    collate_examples,
)
from ml_lab.tactical_v3_corpus import StructuredExample, load_corpus
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import PolicyOutput, TacticalV3Policy
from ml_lab.tactical_v3_objectives import (
    ObjectiveConfig,
    structured_imitation_loss,
)
from ml_lab.tactical_v3_schema import TacticalV3SemanticIdentity


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tactical_v3"
CORPUS_ROOT = FIXTURE_ROOT / "tiny-corpus"
IDENTITY_FIXTURE = FIXTURE_ROOT / "seed-41-spaces.json"


def objective_examples() -> tuple[StructuredExample, StructuredExample]:
    payload = json.loads(IDENTITY_FIXTURE.read_text(encoding="utf-8"))
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    identity = TacticalV3SemanticIdentity(
        scenario_id=payload["scenario_id"],
        scenario_schema_version=payload["scenario_schema_version"],
        contract_version=payload["contract_version"],
        contract_hash=manifest["contract_hash"],
        encoding_hash=payload["encoding_hash"],
        capacity_hash=payload["capacity_hash"],
        environment_kind=manifest["environment_kind"],
        match=payload["match"],
        encoding=payload["encoding"],
        capacity=payload["capacity"],
    )
    corpus = load_corpus(CORPUS_ROOT, identity)
    return corpus.train[0], corpus.validation[0]


def finite_output_for_batch(batch: RaggedBatch) -> PolicyOutput:
    batch_size, candidate_count = batch.candidates.mask.shape
    horizon_count = batch.horizon_targets.shape[1]
    candidate_logits = torch.zeros(batch_size, candidate_count).masked_fill(
        ~batch.candidates.mask, float("-inf")
    )
    return PolicyOutput(
        candidate_logits=candidate_logits.requires_grad_(True),
        outcome_logits=torch.zeros(batch_size, 3, requires_grad=True),
        horizon_logits=torch.zeros(batch_size, horizon_count, requires_grad=True),
        remaining_turns=torch.zeros(batch_size, requires_grad=True),
    )


def _objective_batch() -> RaggedBatch:
    examples = objective_examples()
    base = collate_examples(examples, horizons=(4, 8, 16))
    mask = torch.tensor([[True, True, False], [True, True, True]])
    candidates = base.candidates
    candidates = CandidateBatch(
        candidate_id=candidates.candidate_id[:, :3],
        decision_id=candidates.decision_id[:, :3],
        kind=candidates.kind[:, :3],
        reference_index=candidates.reference_index[:, :3],
        reference_mask=candidates.reference_mask[:, :3],
        projection_integer=candidates.projection_integer[:, :3],
        projection_boolean=candidates.projection_boolean[:, :3],
        mask=mask,
    )
    return replace(
        base,
        candidates=candidates,
        teacher_candidate_index=torch.tensor([0, 2], dtype=torch.int64),
        terminal_outcome=torch.tensor([0, 2], dtype=torch.int64),
        horizon_targets=torch.tensor([[1, 0, 0], [0, 1, 1]], dtype=torch.float32),
        horizon_target_mask=torch.tensor([[True, False, False], [False, True, False]]),
        remaining_turns=torch.tensor([5, 9], dtype=torch.float32),
        remaining_turns_mask=torch.tensor([True, False]),
    )


def make_objective_case() -> tuple[PolicyOutput, RaggedBatch]:
    batch = _objective_batch()
    output = PolicyOutput(
        candidate_logits=torch.tensor(
            [[2.0, 0.0, float("-inf")], [0.0, 1.0, 2.0]], requires_grad=True
        ),
        outcome_logits=torch.tensor(
            [[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]], requires_grad=True
        ),
        horizon_logits=torch.tensor(
            [[0.0, 2.0, -2.0], [1.0, -1.0, 3.0]], requires_grad=True
        ),
        remaining_turns=torch.tensor([4.0, 7.0], requires_grad=True),
    )
    return output, batch


def malformed_objective_case(failure: str) -> tuple[PolicyOutput, RaggedBatch]:
    output, batch = make_objective_case()
    if failure == "candidate_logits_shape":
        output = dataclasses.replace(output, candidate_logits=output.candidate_logits[:, :-1])
    elif failure == "candidate_mask_shape":
        candidates = dataclasses.replace(batch.candidates, mask=batch.candidates.mask[:, :-1])
        batch = dataclasses.replace(batch, candidates=candidates)
    elif failure == "outcome_logits_shape":
        output = dataclasses.replace(output, outcome_logits=output.outcome_logits[:, :2])
    elif failure == "horizon_count":
        output = dataclasses.replace(output, horizon_logits=output.horizon_logits[:, :-1])
    elif failure == "horizon_mask_shape":
        batch = dataclasses.replace(batch, horizon_target_mask=batch.horizon_target_mask[:, :-1])
    elif failure == "remaining_shape":
        output = dataclasses.replace(output, remaining_turns=output.remaining_turns.unsqueeze(1))
    elif failure == "remaining_mask_shape":
        batch = dataclasses.replace(batch, remaining_turns_mask=batch.remaining_turns_mask.unsqueeze(1))
    elif failure == "teacher_shape":
        batch = dataclasses.replace(batch, teacher_candidate_index=batch.teacher_candidate_index.unsqueeze(1))
    elif failure == "outcome_target_shape":
        batch = dataclasses.replace(batch, terminal_outcome=batch.terminal_outcome.unsqueeze(1))
    elif failure == "horizon_target_shape":
        batch = dataclasses.replace(batch, horizon_targets=batch.horizon_targets[:, :-1])
    elif failure == "remaining_target_shape":
        batch = dataclasses.replace(batch, remaining_turns=batch.remaining_turns.unsqueeze(1))
    elif failure in {"teacher_out_of_range", "teacher_padded"}:
        target = batch.teacher_candidate_index.clone()
        target[0] = batch.candidates.mask.shape[1] if failure.endswith("range") else 2
        batch = dataclasses.replace(batch, teacher_candidate_index=target)
    elif failure == "outcome_target":
        target = batch.terminal_outcome.clone(); target[0] = 3
        batch = dataclasses.replace(batch, terminal_outcome=target)
    elif failure == "horizon_target":
        target = batch.horizon_targets.clone(); target[0, 0] = 2.0
        batch = dataclasses.replace(batch, horizon_targets=target)
    elif failure == "remaining_target":
        target = batch.remaining_turns.clone(); target[0] = -1.0
        batch = dataclasses.replace(batch, remaining_turns=target)
    else:
        raise AssertionError(f"unknown malformed objective case {failure}")
    return output, batch


def replace_output_component(
    output: PolicyOutput,
    field: Literal["candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns"],
    value: float,
) -> PolicyOutput:
    changed = getattr(output, field).detach().clone()
    changed.reshape(-1)[0] = value
    changed.requires_grad_(True)
    return dataclasses.replace(output, **{field: changed})


def replace_padded_candidate_logits(output: PolicyOutput, value: float) -> PolicyOutput:
    changed = output.candidate_logits.detach().clone()
    changed[0, 2] = value
    changed.requires_grad_(True)
    return dataclasses.replace(output, candidate_logits=changed)


def valid_candidate_matrix(output: PolicyOutput, batch: RaggedBatch) -> Tensor:
    return output.candidate_logits.detach().clone().masked_fill(
        ~batch.candidates.mask, float("-inf")
    )


def clear_auxiliary_masks(batch: RaggedBatch) -> RaggedBatch:
    return replace(
        batch,
        horizon_target_mask=torch.zeros_like(batch.horizon_target_mask),
        remaining_turns_mask=torch.zeros_like(batch.remaining_turns_mask),
    )


def make_gradient_case(seed: int) -> tuple[TacticalV3Policy, RaggedBatch]:
    torch.manual_seed(seed)
    model = TacticalV3Policy(TacticalV3ModelConfig()).cpu()
    examples = objective_examples()
    batch = collate_examples(examples, horizons=model.config.horizon_turns)
    return model, batch


def named_required_gradients(
    model: nn.Module, prefixes: tuple[str, ...]
) -> dict[str, Tensor | None]:
    return {name: parameter.grad for name, parameter in model.named_parameters()
            if parameter.requires_grad and name.startswith(prefixes)}


def test_default_auxiliary_coefficient_sum_is_within_policy_coefficient() -> None:
    default = ObjectiveConfig()
    assert (default.outcome_coefficient + default.horizon_coefficient
            + default.remaining_turns_coefficient) == pytest.approx(0.5)
    assert 0.5 <= default.policy_coefficient


@pytest.mark.parametrize("field", ("policy_coefficient", "outcome_coefficient", "horizon_coefficient", "remaining_turns_coefficient"))
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), -0.1))
def test_every_coefficient_rejects_nonfinite_and_negative_values(field: str, value: float) -> None:
    with pytest.raises(ValueError, match=field):
        ObjectiveConfig(**{field: value})


def test_policy_coefficient_must_be_strictly_positive() -> None:
    with pytest.raises(ValueError, match="policy_coefficient"):
        ObjectiveConfig(policy_coefficient=0.0)


def test_auxiliary_coefficient_sum_cannot_exceed_policy_coefficient() -> None:
    with pytest.raises(ValueError, match="auxiliary coefficient sum"):
        ObjectiveConfig(outcome_coefficient=0.6, horizon_coefficient=0.3, remaining_turns_coefficient=0.2)


def test_target_free_collate_decisions_batch_is_rejected() -> None:
    examples = objective_examples()
    batch = collate_decisions(tuple(example.decision for example in examples), horizons=(4, 8, 16))
    assert batch.teacher_candidate_index.tolist() == [-1, -1]
    assert batch.terminal_outcome.tolist() == [-1, -1]
    with pytest.raises(ValueError, match="target-free.*teacher_candidate_index=-1.*terminal_outcome=-1"):
        structured_imitation_loss(finite_output_for_batch(batch), batch, ObjectiveConfig())


MALFORMED_OBJECTIVE_ERRORS = {
    "candidate_logits_shape": "candidate_logits shape", "candidate_mask_shape": "candidate mask shape",
    "outcome_logits_shape": "outcome_logits shape", "horizon_count": "horizon count",
    "horizon_mask_shape": "horizon_target_mask shape", "remaining_shape": "remaining_turns shape",
    "remaining_mask_shape": "remaining_turns_mask shape", "teacher_shape": "teacher_candidate_index shape",
    "outcome_target_shape": "terminal_outcome shape", "horizon_target_shape": "horizon_targets shape",
    "remaining_target_shape": "batch.remaining_turns shape", "teacher_out_of_range": "teacher_candidate_index.*out of range",
    "teacher_padded": "teacher_candidate_index.*padded", "outcome_target": "terminal_outcome.*0..2",
    "horizon_target": "horizon_targets.*binary", "remaining_target": "remaining_turns.*positive",
}


@pytest.mark.parametrize("failure", tuple(MALFORMED_OBJECTIVE_ERRORS))
def test_shapes_and_target_ranges_fail_closed_before_loss_math(failure: str) -> None:
    output, batch = malformed_objective_case(failure)
    with pytest.raises(ValueError, match=MALFORMED_OBJECTIVE_ERRORS[failure]):
        structured_imitation_loss(output, batch, ObjectiveConfig())


def test_policy_loss_ignores_padding_and_matches_manual_cross_entropy() -> None:
    output, batch = make_objective_case()
    changed = replace_padded_candidate_logits(output, value=1_000_000.0)
    expected = F.cross_entropy(valid_candidate_matrix(output, batch), batch.teacher_candidate_index)
    actual = structured_imitation_loss(changed, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.policy, expected, rtol=0.0, atol=1e-7)


def test_outcome_loss_uses_loss_draw_win_target_order() -> None:
    output, batch = make_objective_case()
    expected = F.cross_entropy(output.outcome_logits, torch.tensor([0, 2]))
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.outcome, expected, rtol=0.0, atol=1e-7)


def test_horizon_loss_uses_only_uncensored_target_mask() -> None:
    output, batch = make_objective_case()
    expected = F.binary_cross_entropy_with_logits(output.horizon_logits[batch.horizon_target_mask], batch.horizon_targets[batch.horizon_target_mask])
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.horizon, expected, rtol=0.0, atol=1e-7)


def test_remaining_turns_loss_uses_only_nontruncated_wins() -> None:
    output, batch = make_objective_case()
    expected = F.smooth_l1_loss(output.remaining_turns[batch.remaining_turns_mask], batch.remaining_turns[batch.remaining_turns_mask])
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.remaining_turns, expected, rtol=0.0, atol=1e-7)


def test_weighted_total_is_exact_coefficient_combination() -> None:
    output, batch = make_objective_case()
    config = ObjectiveConfig(policy_coefficient=1.0, outcome_coefficient=0.1, horizon_coefficient=0.2, remaining_turns_coefficient=0.15)
    actual = structured_imitation_loss(output, batch, config)
    expected = actual.policy + 0.1 * actual.outcome + 0.2 * actual.horizon + 0.15 * actual.remaining_turns
    torch.testing.assert_close(actual.total, expected, rtol=0.0, atol=0.0)


def test_empty_auxiliary_masks_produce_differentiable_finite_zeroes() -> None:
    output, batch = make_objective_case()
    actual = structured_imitation_loss(output, clear_auxiliary_masks(batch), ObjectiveConfig())
    assert actual.horizon.item() == 0.0 and actual.remaining_turns.item() == 0.0
    assert actual.horizon.requires_grad and actual.remaining_turns.requires_grad
    (actual.horizon + actual.remaining_turns).backward()
    assert torch.isfinite(output.horizon_logits.grad).all()
    assert torch.isfinite(output.remaining_turns.grad).all()


def test_padded_negative_infinity_is_allowed_and_total_remains_finite() -> None:
    output, batch = make_objective_case()
    assert torch.isneginf(output.candidate_logits[~batch.candidates.mask]).all()
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    assert torch.isfinite(actual.total)


@pytest.mark.parametrize("field, value", (("candidate_logits", float("nan")), ("candidate_logits", float("inf")), ("candidate_logits", float("-inf")), ("outcome_logits", float("inf")), ("horizon_logits", float("-inf")), ("remaining_turns", float("nan"))))
def test_each_nonfinite_output_component_fails_with_named_error(field: str, value: float) -> None:
    output, batch = make_objective_case()
    bad = replace_output_component(output, field, value)  # type: ignore[arg-type]
    with pytest.raises(FloatingPointError, match=field):
        structured_imitation_loss(bad, batch, ObjectiveConfig())


def test_default_loss_backpropagates_finite_scorer_and_encoder_gradients() -> None:
    model, batch = make_gradient_case(seed=103)
    loss = structured_imitation_loss(model(batch), batch, ObjectiveConfig()).total
    loss.backward()
    gradients = named_required_gradients(model, prefixes=("encoders.", "candidate_scorer."))
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients.values())
