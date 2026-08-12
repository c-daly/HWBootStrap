from __future__ import annotations

import dataclasses
import json
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType
from typing import Literal

import pytest
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ml_lab.tactical_v3_batching import (
    CandidateBatch,
    RaggedBatch,
    TABLE_ORDER,
    collate_decisions,
    collate_examples,
    validate_ragged_batch,
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
    assert default.policy_coefficient == 1.0


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

@pytest.mark.parametrize("field", ("candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns"))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float16, torch.bfloat16))
def test_policy_heads_require_exact_float32(field: str, dtype: torch.dtype) -> None:
    output, batch = make_objective_case()
    changed = getattr(output, field).to(dtype=dtype).detach().requires_grad_(True)
    bad = dataclasses.replace(output, **{field: changed})
    with pytest.raises(ValueError, match=rf"{field} dtype must be torch\.float32"):
        structured_imitation_loss(bad, batch, ObjectiveConfig())


@pytest.mark.parametrize("field", ("horizon_targets", "remaining_turns"))
@pytest.mark.parametrize("dtype", (torch.float64, torch.float16, torch.bfloat16))
def test_continuous_targets_require_exact_float32(field: str, dtype: torch.dtype) -> None:
    output, batch = make_objective_case()
    bad = replace(batch, **{field: getattr(batch, field).to(dtype=dtype)})
    with pytest.raises(ValueError, match=rf"{field} dtype must be torch\.float32"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    "field",
    ("candidate_id", "decision_id", "kind", "reference_index", "reference_mask",
     "projection_integer", "projection_boolean"),
)
def test_every_candidate_field_shape_is_validated(field: str) -> None:
    output, batch = make_objective_case()
    value = getattr(batch.candidates, field)
    changed = value[..., :-1]
    candidates = dataclasses.replace(batch.candidates, **{field: changed})
    bad = replace(batch, candidates=candidates)
    with pytest.raises(ValueError, match=rf"{field} shape"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    ("field", "dtype"),
    (("candidate_id", torch.int32), ("decision_id", torch.int32), ("kind", torch.int32),
     ("reference_index", torch.int32), ("reference_mask", torch.int64),
     ("projection_integer", torch.int32), ("projection_boolean", torch.int64)),
)
def test_every_candidate_field_dtype_is_validated(field: str, dtype: torch.dtype) -> None:
    output, batch = make_objective_case()
    value = getattr(batch.candidates, field).to(dtype=dtype)
    candidates = dataclasses.replace(batch.candidates, **{field: value})
    bad = replace(batch, candidates=candidates)
    with pytest.raises(ValueError, match=rf"{field} dtype"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    "field",
    ("candidate_id", "decision_id", "kind", "reference_index", "reference_mask",
     "projection_integer", "projection_boolean"),
)
def test_every_candidate_field_device_is_validated(field: str) -> None:
    output, batch = make_objective_case()
    value = getattr(batch.candidates, field).to(device="meta")
    candidates = dataclasses.replace(batch.candidates, **{field: value})
    bad = replace(batch, candidates=candidates)
    with pytest.raises(ValueError, match=rf"{field} must be on the candidate mask device"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


def test_candidate_id_active_values_must_fit_int32() -> None:
    output, batch = make_objective_case()
    value = batch.candidates.candidate_id.clone(); value[0, 0] = 2**31
    bad = replace(batch, candidates=dataclasses.replace(batch.candidates, candidate_id=value))
    with pytest.raises(ValueError, match="candidate_id.*int32"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


def test_candidate_kind_active_values_must_be_in_range() -> None:
    output, batch = make_objective_case()
    value = batch.candidates.kind.clone(); value[0, 0] = 4
    bad = replace(batch, candidates=dataclasses.replace(batch.candidates, kind=value))
    with pytest.raises(ValueError, match="kind.*out of range"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


def test_candidate_reference_active_values_must_select_nodes() -> None:
    output, batch = make_objective_case()
    value = batch.candidates.reference_index.clone()
    value[0, 0, 0] = batch.node_mask.shape[1]
    refs = dataclasses.replace(batch.candidates, reference_index=value)
    bad = replace(batch, candidates=refs)
    with pytest.raises(ValueError, match="reference_index.*out of range"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


def test_projection_integer_active_values_must_fit_int32() -> None:
    output, batch = make_objective_case()
    value = batch.candidates.projection_integer.clone(); value[0, 0, 0] = 2**31
    bad = replace(batch, candidates=dataclasses.replace(batch.candidates, projection_integer=value))
    with pytest.raises(ValueError, match="projection_integer.*int32"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    ("field", "dtype"),
    (("horizon_target_mask", torch.int64), ("remaining_turns_mask", torch.int64),
     ("teacher_candidate_index", torch.int32), ("terminal_outcome", torch.int32)),
)
def test_direct_target_masks_and_indices_require_exact_dtype(field: str, dtype: torch.dtype) -> None:
    output, batch = make_objective_case()
    bad = replace(batch, **{field: getattr(batch, field).to(dtype=dtype)})
    with pytest.raises(ValueError, match=rf"{field} dtype"):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    "field",
    ("horizon_target_mask", "remaining_turns_mask", "teacher_candidate_index", "terminal_outcome"),
)
def test_direct_target_masks_and_indices_require_batch_device(field: str) -> None:
    output, batch = make_objective_case()
    bad = replace(batch, **{field: getattr(batch, field).to(device="meta")})
    with pytest.raises(ValueError, match=rf"{field} must be on the candidate mask device"):
        structured_imitation_loss(output, bad, ObjectiveConfig())

@pytest.mark.parametrize("value", (0.5, 1.1, 1, True, torch.tensor(1.0)))
def test_policy_coefficient_must_be_exact_builtin_float_one(value: object) -> None:
    with pytest.raises(ValueError, match="policy_coefficient"):
        ObjectiveConfig(policy_coefficient=value)  # type: ignore[arg-type]


def test_policy_coefficient_rejects_numpy_scalar() -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match="policy_coefficient"):
        ObjectiveConfig(policy_coefficient=np.float64(1.0))  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("outcome_coefficient", "horizon_coefficient", "remaining_turns_coefficient"))
@pytest.mark.parametrize("value", (True, 1, torch.tensor(0.1)))
def test_auxiliary_coefficients_require_exact_builtin_float(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        ObjectiveConfig(**{field: value})  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ("outcome_coefficient", "horizon_coefficient", "remaining_turns_coefficient"))
def test_auxiliary_coefficients_reject_numpy_scalars(field: str) -> None:
    np = pytest.importorskip("numpy")
    with pytest.raises(ValueError, match=field):
        ObjectiveConfig(**{field: np.float64(0.1)})  # type: ignore[arg-type]


def test_auxiliary_coefficient_sum_is_capped_at_exactly_one_half() -> None:
    ObjectiveConfig(outcome_coefficient=0.2, horizon_coefficient=0.2, remaining_turns_coefficient=0.1)
    with pytest.raises(ValueError, match="auxiliary coefficient sum"):
        ObjectiveConfig(outcome_coefficient=0.2, horizon_coefficient=0.2, remaining_turns_coefficient=0.1000000000000001)


def test_valid_canonical_ragged_batch_control() -> None:
    output, batch = make_objective_case()
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    assert torch.isfinite(actual.total)


def test_valid_target_free_ragged_batch_control() -> None:
    examples = objective_examples()
    batch = collate_decisions(
        tuple(example.decision for example in examples), horizons=(4, 8, 16)
    )
    validate_ragged_batch(batch)


@pytest.mark.parametrize(
    "mutation",
    (
        "tables",
        "table_slices",
        "neighbor_index",
        "neighbor_mask",
        "neighborhood_source",
        "neighborhood_kind",
        "neighborhood_mask",
    ),
)
def test_complete_ragged_batch_contract_rejects_topology_mutations(
    mutation: str,
) -> None:
    output, batch = make_objective_case()
    if mutation == "tables":
        bad = replace(batch, tables={})
    elif mutation == "table_slices":
        slices = dict(batch.table_slices)
        slices["cells"] = slice(slices["cells"].start + 1, slices["cells"].stop)
        bad = replace(batch, table_slices=slices)
    elif mutation == "neighbor_index":
        value = batch.cell_neighbor_index.clone()
        value[0, 0, 0] = batch.node_mask.shape[1]
        bad = replace(batch, cell_neighbor_index=value)
    elif mutation == "neighbor_mask":
        bad = replace(
            batch, cell_neighbor_mask=batch.cell_neighbor_mask.to(dtype=torch.int64)
        )
    elif mutation == "neighborhood_source":
        value = batch.neighborhoods.source_index.clone()
        value[0, 0, 0] = batch.node_mask.shape[1]
        bad = replace(
            batch, neighborhoods=replace(batch.neighborhoods, source_index=value)
        )
    elif mutation == "neighborhood_kind":
        value = batch.neighborhoods.kind.clone()
        value[0, 0, 0] = 14
        bad = replace(batch, neighborhoods=replace(batch.neighborhoods, kind=value))
    else:
        bad = replace(
            batch,
            neighborhoods=replace(
                batch.neighborhoods,
                mask=batch.neighborhoods.mask.to(dtype=torch.int64),
            ),
        )
    with pytest.raises(
        ValueError, match="ragged batch|table|slice|neighbor|neighborhood"
    ):
        structured_imitation_loss(output, bad, ObjectiveConfig())


@pytest.mark.parametrize(
    "mutation",
    (
        "table_key_order",
        "table_tensor_dtype",
        "table_mapping",
        "table_category_range",
        "table_mask_gap",
        "node_mask_device",
        "neighbor_self",
        "neighbor_mask_gap",
        "neighborhood_integer_range",
        "neighborhood_mask_gap",
        "candidate_mask_gap",
        "candidate_reference_family",
        "candidate_reference_mask",
    ),
)
def test_complete_ragged_batch_contract_rejects_adversarial_canonical_mutations(
    mutation: str,
) -> None:
    output, batch = make_objective_case()
    if mutation == "table_key_order":
        bad = replace(
            batch,
            tables=MappingProxyType(
                {name: batch.tables[name] for name in reversed(TABLE_ORDER)}
            ),
        )
    elif mutation == "table_tensor_dtype":
        tables = dict(batch.tables)
        tables["cells"] = replace(
            tables["cells"], numeric=tables["cells"].numeric.to(torch.float64)
        )
        bad = replace(batch, tables=MappingProxyType(tables))
    elif mutation == "table_mapping":
        tables = dict(batch.tables)
        tables["cells"] = replace(
            tables["cells"], categorical=dict(tables["cells"].categorical)
        )
        bad = replace(batch, tables=MappingProxyType(tables))
    elif mutation == "table_category_range":
        tables = dict(batch.tables)
        terrain = tables["cells"].categorical["terrain"].clone()
        terrain[0, 0] = 4
        categorical = dict(tables["cells"].categorical)
        categorical["terrain"] = terrain
        tables["cells"] = replace(
            tables["cells"], categorical=MappingProxyType(categorical)
        )
        bad = replace(batch, tables=MappingProxyType(tables))
    elif mutation == "table_mask_gap":
        tables = dict(batch.tables)
        mask = tables["rules"].mask.clone()
        mask[0, 0] = False
        tables["rules"] = replace(tables["rules"], mask=mask)
        node_mask = batch.node_mask.clone()
        node_mask[0, batch.table_slices["rules"].start] = False
        bad = replace(batch, tables=MappingProxyType(tables), node_mask=node_mask)
    elif mutation == "node_mask_device":
        bad = replace(batch, node_mask=batch.node_mask.to("meta"))
    elif mutation == "neighbor_self":
        value = batch.cell_neighbor_index.clone()
        value[0, 0, 0] = batch.table_slices["cells"].start
        bad = replace(batch, cell_neighbor_index=value)
    elif mutation == "neighbor_mask_gap":
        mask = batch.cell_neighbor_mask.clone()
        mask[0, 0, 0] = False
        bad = replace(batch, cell_neighbor_mask=mask)
    elif mutation == "neighborhood_integer_range":
        value = batch.neighborhoods.int_feature.clone()
        sample, destination, slot = batch.neighborhoods.mask.nonzero()[0].tolist()
        value[sample, destination, slot] = 2**31
        bad = replace(
            batch, neighborhoods=replace(batch.neighborhoods, int_feature=value)
        )
    elif mutation == "neighborhood_mask_gap":
        mask = batch.neighborhoods.mask.clone()
        sample, destination = (mask.sum(dim=2) >= 2).nonzero()[0].tolist()
        mask[sample, destination, 0] = False
        bad = replace(batch, neighborhoods=replace(batch.neighborhoods, mask=mask))
    elif mutation == "candidate_mask_gap":
        mask = batch.candidates.mask.clone()
        mask[0] = torch.tensor([True, False, True])
        candidates = replace(batch.candidates, mask=mask)
        logits = output.candidate_logits.detach().clone()
        logits[0, 2] = 0.0
        output = replace(output, candidate_logits=logits.requires_grad_(True))
        bad = replace(batch, candidates=candidates)
    elif mutation == "candidate_reference_family":
        value = batch.candidates.reference_index.clone()
        value[0, 0, 0] = batch.table_slices["cells"].start
        bad = replace(
            batch, candidates=replace(batch.candidates, reference_index=value)
        )
    else:
        value = batch.candidates.reference_mask.clone()
        value[0, 0, 0] = False
        bad = replace(
            batch, candidates=replace(batch.candidates, reference_mask=value)
        )
    with pytest.raises(
        ValueError,
        match="table|node_mask|neighbor|neighborhood|candidate|reference|cells|categorical|numeric|mask",
    ):
        structured_imitation_loss(output, bad, ObjectiveConfig())


def test_complete_ragged_batch_contract_rejects_mixed_target_free_sentinels() -> None:
    output, batch = make_objective_case()
    teacher = batch.teacher_candidate_index.clone()
    teacher[0] = -1
    with pytest.raises(ValueError, match="teacher_candidate_index|target-free"):
        structured_imitation_loss(output, replace(batch, teacher_candidate_index=teacher), ObjectiveConfig())
