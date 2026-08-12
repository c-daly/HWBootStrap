from __future__ import annotations

import dataclasses
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Literal

import pytest
import torch
from torch.utils.hooks import RemovableHandle

from ml_lab.tactical_v3_batching import (
    TABLE_ORDER,
    CandidateBatch,
    RaggedBatch,
    TokenTableBatch,
    collate_examples,
)
from ml_lab.tactical_v3_corpus import StructuredExample
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, PolicyOutput, TacticalV3Policy

from .test_tactical_v3_layers import (
    canonical_example as _load_canonical_example,
    permute_table_and_remap,
)


@dataclass(frozen=True, slots=True)
class PolicyTestCase:
    policy: TacticalV3Policy
    batch: RaggedBatch


@lru_cache(maxsize=1)
def canonical_model_example() -> StructuredExample:
    return _load_canonical_example()


def make_cardinality_stress_example(
    example: StructuredExample, candidate_count: int
) -> StructuredExample:
    if candidate_count <= 0 or not example.decision.candidates:
        raise ValueError("cardinality stress data requires positive count and a source candidate")
    decision_id = example.decision.decision_id
    source = example.decision.candidates
    candidates = tuple(
        dataclasses.replace(
            source[index % len(source)], candidate_id=index, decision_id=decision_id
        )
        for index in range(candidate_count)
    )
    return dataclasses.replace(
        example,
        decision=dataclasses.replace(example.decision, candidates=candidates),
        target=dataclasses.replace(example.target, teacher_candidate_id=0),
    )


def seeded_policy(seed: int) -> TacticalV3Policy:
    torch.manual_seed(seed)
    return TacticalV3Policy(TacticalV3ModelConfig()).cpu().eval()


def make_policy_case(candidate_counts: tuple[int, ...], seed: int) -> PolicyTestCase:
    canonical = canonical_model_example()
    examples = tuple(
        make_cardinality_stress_example(canonical, count) for count in candidate_counts
    )
    policy = seeded_policy(seed)
    batch = collate_examples(examples, horizons=policy.config.horizon_turns)
    return PolicyTestCase(policy=policy, batch=batch)


def assert_auxiliary_heads_equal(
    actual: PolicyOutput, expected: PolicyOutput, atol: float
) -> None:
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0.0, atol=atol
        )


def expand_synthetic_cells_for_batch_shape(
    example: StructuredExample, total_cells: int
) -> StructuredExample:
    cells = example.decision.observation.cells
    if total_cells < len(cells) or not cells:
        raise ValueError("total_cells must include every source cell")
    extra = tuple(
        replace(
            cells[0],
            q=1000 + index,
            r=-1000 - index,
            terrain="plains",
            elevation=0,
            self_deployment_zone=False,
            opponent_deployment_zone=False,
            controller=None,
            is_boundary=False,
            currently_visible=False,
            previously_observed=False,
        )
        for index in range(total_cells - len(cells))
    )
    observation = replace(example.decision.observation, cells=cells + extra)
    return replace(example, decision=replace(example.decision, observation=observation))


def _candidate_row_order(batch: RaggedBatch, seed: int) -> tuple[torch.Tensor, ...]:
    orders = []
    for sample, mask in enumerate(batch.candidates.mask):
        valid_count = int(mask.sum())
        generator = torch.Generator(device=mask.device).manual_seed(seed + sample)
        valid = torch.randperm(valid_count, generator=generator, device=mask.device)
        padding = torch.arange(valid_count, mask.numel(), device=mask.device)
        orders.append(torch.cat((valid, padding)))
    return tuple(orders)


def _select_candidate_rows(value: torch.Tensor, orders: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.stack(
        tuple(value[sample].index_select(0, order) for sample, order in enumerate(orders))
    )


def permute_candidate_rows(
    batch: RaggedBatch, seed: int
) -> tuple[RaggedBatch, tuple[torch.Tensor, ...]]:
    orders = _candidate_row_order(batch, seed)
    inverses = []
    for order in orders:
        inverse = torch.empty_like(order)
        inverse[order] = torch.arange(order.numel(), device=order.device)
        inverses.append(inverse)
    candidates = replace(
        batch.candidates,
        candidate_id=_select_candidate_rows(batch.candidates.candidate_id, orders),
        decision_id=_select_candidate_rows(batch.candidates.decision_id, orders),
        kind=_select_candidate_rows(batch.candidates.kind, orders),
        reference_index=_select_candidate_rows(batch.candidates.reference_index, orders),
        reference_mask=_select_candidate_rows(batch.candidates.reference_mask, orders),
        projection_integer=_select_candidate_rows(batch.candidates.projection_integer, orders),
        projection_boolean=_select_candidate_rows(batch.candidates.projection_boolean, orders),
        mask=_select_candidate_rows(batch.candidates.mask, orders),
    )
    return replace(batch, candidates=candidates), tuple(inverses)


def restore_candidate_rows(
    logits: torch.Tensor, inverse: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    return torch.stack(
        tuple(logits[sample].index_select(0, order) for sample, order in enumerate(inverse))
    )


def _append(value: torch.Tensor, rows: int, fill: int | float | bool) -> torch.Tensor:
    shape = (value.shape[0], rows, *value.shape[2:])
    suffix = torch.full(shape, fill, dtype=value.dtype, device=value.device)
    return torch.cat((value, suffix), dim=1)


def append_candidate_padding(batch: RaggedBatch, rows: int, fill: float) -> RaggedBatch:
    if rows <= 0:
        raise ValueError("rows must be positive")
    candidates = batch.candidates
    padded = CandidateBatch(
        candidate_id=_append(candidates.candidate_id, rows, int(fill)),
        decision_id=_append(candidates.decision_id, rows, int(fill)),
        kind=_append(candidates.kind, rows, int(fill)),
        reference_index=_append(candidates.reference_index, rows, int(fill)),
        reference_mask=_append(candidates.reference_mask, rows, True),
        projection_integer=_append(candidates.projection_integer, rows, int(fill)),
        projection_boolean=_append(candidates.projection_boolean, rows, True),
        mask=_append(candidates.mask, rows, False),
    )
    return replace(batch, candidates=padded)


def permute_model_table_and_remap(
    batch: RaggedBatch, table: str, seed: int
) -> tuple[RaggedBatch, torch.Tensor]:
    return permute_table_and_remap(
        batch, "memory_records" if table == "memory" else table, seed
    )


def movable_projection_case(batch: RaggedBatch) -> tuple[int, int]:
    cells = batch.table_slices["cells"]
    for row in range(batch.candidates.mask.shape[1]):
        if not bool(batch.candidates.mask[0, row]):
            continue
        if not bool(batch.candidates.reference_mask[0, row, 5]):
            continue
        current = int(batch.candidates.reference_index[0, row, 5])
        current_q = float(batch.tables["cells"].numeric[0, current - cells.start, 0])
        for candidate_cell in range(cells.start, cells.stop):
            local = candidate_cell - cells.start
            if (
                bool(batch.tables["cells"].mask[0, local])
                and candidate_cell != current
                and float(batch.tables["cells"].numeric[0, local, 0]) != current_q
            ):
                return row, candidate_cell
    raise AssertionError("fixture must contain a projected destination and alternate q coordinate")


def retarget_projection(
    batch: RaggedBatch, candidate_row: int, cell_row: int
) -> RaggedBatch:
    indices = batch.candidates.reference_index.clone()
    indices[0, candidate_row, 5] = cell_row
    return replace(batch, candidates=replace(batch.candidates, reference_index=indices))


def make_reference_sensitive_policy_case(seed: int) -> PolicyTestCase:
    case = make_policy_case((19,), seed)
    policy = case.policy
    hidden = policy.config.hidden_dim
    categorical = policy.config.categorical_dim
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
        policy.token_encoders.numeric_projections["cells"].weight[0, 0] = 1.0
        policy.token_encoders.table_projections["cells"].weight[0, 0] = 1.0

        destination_offset = categorical + hidden + 2 * categorical + 5 * hidden
        policy.candidate_encoder[0].weight[0, destination_offset] = 1.0
        policy.candidate_encoder[0].weight[1, destination_offset] = -1.0
        policy.candidate_encoder[2].weight[0, 0] = 1.0
        policy.candidate_encoder[2].weight[0, 1] = -1.0

        candidate_offset = hidden
        policy.candidate_scorer[0].weight[0, candidate_offset] = 1.0
        policy.candidate_scorer[0].weight[1, candidate_offset] = -1.0
        policy.candidate_scorer[2].weight[0, 0] = 1.0
        policy.candidate_scorer[2].weight[0, 1] = -1.0
    return case


def mask_all_candidates(batch: RaggedBatch) -> RaggedBatch:
    return replace(
        batch,
        candidates=replace(batch.candidates, mask=torch.zeros_like(batch.candidates.mask)),
    )


def inject_nan_into_policy_output(
    policy: TacticalV3Policy,
    field: Literal[
        "candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns"
    ],
) -> RemovableHandle:
    module = {
        "candidate_logits": policy.candidate_scorer,
        "outcome_logits": policy.outcome_head,
        "horizon_logits": policy.horizon_head,
        "remaining_turns": policy.remaining_turns_head,
    }[field]

    def inject(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor):
        changed = output.clone()
        changed.reshape(-1)[0] = float("nan")
        return changed

    return module.register_forward_hook(inject)


def force_equal_candidate_logits(policy: TacticalV3Policy) -> None:
    with torch.no_grad():
        policy.candidate_scorer[-1].weight.zero_()
        policy.candidate_scorer[-1].bias.zero_()


def candidate_identity_set(batch: RaggedBatch, sample: int) -> set[tuple[int, int]]:
    mask = batch.candidates.mask[sample]
    return set(
        zip(
            batch.candidates.decision_id[sample, mask].tolist(),
            batch.candidates.candidate_id[sample, mask].tolist(),
            strict=True,
        )
    )


def _to_device(batch: RaggedBatch, device: torch.device) -> RaggedBatch:
    tables = {
        name: TokenTableBatch(
            table.numeric.to(device),
            {field: value.to(device) for field, value in table.categorical.items()},
            {field: value.to(device) for field, value in table.boolean.items()},
            table.mask.to(device),
        )
        for name, table in batch.tables.items()
    }
    neighborhoods = dataclasses.replace(
        batch.neighborhoods,
        **{
            field.name: getattr(batch.neighborhoods, field.name).to(device)
            for field in dataclasses.fields(batch.neighborhoods)
        },
    )
    candidates = dataclasses.replace(
        batch.candidates,
        **{
            field.name: getattr(batch.candidates, field.name).to(device)
            for field in dataclasses.fields(batch.candidates)
        },
    )
    tensors = {
        field.name: getattr(batch, field.name).to(device)
        for field in dataclasses.fields(batch)
        if isinstance(getattr(batch, field.name), torch.Tensor)
    }
    return replace(batch, tables=tables, neighborhoods=neighborhoods, candidates=candidates, **tensors)


def test_policy_contracts_are_frozen_and_do_not_accept_maximum_action_count() -> None:
    output_fields = tuple(field.name for field in dataclasses.fields(PolicyOutput))
    identity_fields = tuple(field.name for field in dataclasses.fields(CandidateIdentity))
    assert output_fields == (
        "candidate_logits",
        "outcome_logits",
        "horizon_logits",
        "remaining_turns",
    )
    assert identity_fields == ("decision_id", "candidate_id")
    with pytest.raises((AttributeError, TypeError)):
        CandidateIdentity(1, 2).candidate_id = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        TacticalV3Policy(TacticalV3ModelConfig(), max_candidates=19)  # type: ignore[call-arg]


def test_policy_output_shapes_and_finiteness() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=47)
    output = case.policy(case.batch)
    assert output.candidate_logits.shape == (3, 19)
    assert output.outcome_logits.shape == (3, 3)
    assert output.horizon_logits.shape == (3, 3)
    assert output.remaining_turns.shape == (3,)
    assert torch.isfinite(output.candidate_logits[case.batch.candidates.mask]).all()
    assert torch.isneginf(output.candidate_logits[~case.batch.candidates.mask]).all()
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        assert torch.isfinite(getattr(output, name)).all(), name


def test_candidate_permutation_permutes_logits_and_preserves_identity_selection() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=53)
    permuted, inverse = permute_candidate_rows(case.batch, seed=59)
    expected = case.policy(case.batch).candidate_logits
    actual = restore_candidate_rows(case.policy(permuted).candidate_logits, inverse)
    torch.testing.assert_close(
        actual[case.batch.candidates.mask],
        expected[case.batch.candidates.mask],
        rtol=0.0,
        atol=1e-6,
    )
    assert case.policy.select(permuted) == case.policy.select(case.batch)


def test_candidate_padding_cannot_change_valid_logits_or_argmax() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=61)
    padded = append_candidate_padding(case.batch, rows=11, fill=1_000_000.0)
    expected = case.policy(case.batch)
    actual = case.policy(padded)
    torch.testing.assert_close(
        actual.candidate_logits[:, : expected.candidate_logits.shape[1]][
            case.batch.candidates.mask
        ],
        expected.candidate_logits[case.batch.candidates.mask],
        rtol=0.0,
        atol=0.0,
    )
    assert_auxiliary_heads_equal(actual, expected, atol=0.0)
    assert case.policy.select(padded) == case.policy.select(case.batch)


def test_batch_shape_padding_beside_synthetic_384_cell_state_is_invariant() -> None:
    example = canonical_model_example()
    synthetic_large = expand_synthetic_cells_for_batch_shape(example, total_cells=384)
    policy = seeded_policy(seed=67)
    single = collate_examples((example,), horizons=policy.config.horizon_turns)
    mixed = collate_examples((example, synthetic_large), horizons=policy.config.horizon_turns)
    expected = policy(single)
    actual = policy(mixed)
    torch.testing.assert_close(
        actual.candidate_logits[0, : single.candidates.mask.shape[1]],
        expected.candidate_logits[0],
        rtol=0.0,
        atol=1e-6,
    )
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        torch.testing.assert_close(
            getattr(actual, name)[0], getattr(expected, name)[0], rtol=0.0, atol=1e-6
        )
    assert policy.select(mixed)[0] == policy.select(single)[0]


def test_softmax_probability_is_zero_on_padding_and_sums_to_one_on_valid_rows() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=71)
    probabilities = torch.softmax(case.policy(case.batch).candidate_logits, dim=-1)
    assert torch.equal(
        probabilities[~case.batch.candidates.mask],
        torch.zeros_like(probabilities[~case.batch.candidates.mask]),
    )
    torch.testing.assert_close(
        probabilities.sum(dim=1), torch.ones(3), rtol=0.0, atol=1e-7
    )


def test_state_table_permutations_leave_policy_output_unchanged() -> None:
    case = make_policy_case(candidate_counts=(19,), seed=73)
    expected = case.policy(case.batch)
    for table in (
        "cells",
        "units",
        "templates",
        "capability_definitions",
        "capability_allocations",
        "rules",
        "memory",
        "relations",
    ):
        permuted, _inverse = permute_model_table_and_remap(case.batch, table, seed=79)
        actual = case.policy(permuted)
        torch.testing.assert_close(
            actual.candidate_logits, expected.candidate_logits, rtol=0.0, atol=1e-6
        )
        assert_auxiliary_heads_equal(actual, expected, atol=1e-6)


def test_projection_reference_changes_affect_only_the_referenced_candidate_path() -> None:
    case = make_reference_sensitive_policy_case(seed=83)
    candidate_row, alternate_cell = movable_projection_case(case.batch)
    changed = retarget_projection(case.batch, candidate_row, alternate_cell)
    before = case.policy(case.batch).candidate_logits[0]
    after = case.policy(changed).candidate_logits[0]
    other = case.batch.candidates.mask[0].clone()
    other[candidate_row] = False
    torch.testing.assert_close(after[other], before[other], rtol=0.0, atol=0.0)
    assert not torch.equal(after[candidate_row], before[candidate_row])


def test_all_masked_candidate_rows_raise_before_argmax() -> None:
    case = make_policy_case(candidate_counts=(3,), seed=89)
    with pytest.raises(ValueError, match="sample 0 has no valid candidates"):
        case.policy.select(mask_all_candidates(case.batch))


@pytest.mark.parametrize(
    "field", ("candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns")
)
def test_nonfinite_policy_output_raises_before_selection(field: str) -> None:
    case = make_policy_case(candidate_counts=(3,), seed=97)
    handle = inject_nan_into_policy_output(case.policy, field)  # type: ignore[arg-type]
    try:
        with pytest.raises(FloatingPointError, match=field):
            case.policy(case.batch)
    finally:
        handle.remove()


def test_selected_candidate_is_an_exact_member_with_smallest_id_tie_break() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=101)
    force_equal_candidate_logits(case.policy)
    selections = case.policy.select(case.batch)
    for sample, selection in enumerate(selections):
        constructed = candidate_identity_set(case.batch, sample)
        assert (selection.decision_id, selection.candidate_id) in constructed
        assert selection.candidate_id == min(
            candidate_id for _, candidate_id in constructed
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("kind", lambda value: value[:, :-1], "candidates.kind shape"),
        (
            "reference_index",
            lambda value: value[:, :, :-1],
            "candidates.reference_index shape",
        ),
        ("mask", lambda value: value.to(torch.int64), "candidates.mask dtype"),
    ),
)
def test_invalid_candidate_tensor_contracts_fail_closed(
    field: str, replacement, message: str
) -> None:
    case = make_policy_case(candidate_counts=(3,), seed=103)
    candidates = replace(
        case.batch.candidates,
        **{field: replacement(getattr(case.batch.candidates, field))},
    )
    with pytest.raises(ValueError, match=message):
        case.policy(replace(case.batch, candidates=candidates))


def test_policy_gradients_dtype_and_cpu_determinism_are_finite() -> None:
    case = make_policy_case(candidate_counts=(3,), seed=107)
    policy = case.policy.train()
    first = policy(case.batch)
    loss = (
        first.candidate_logits[case.batch.candidates.mask].sum()
        + first.outcome_logits.sum()
        + first.horizon_logits.sum()
        + first.remaining_turns.sum()
    )
    loss.backward()
    gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert {parameter.dtype for parameter in policy.parameters()} == {torch.float32}
    assert first.candidate_logits.dtype == torch.float32

    policy.eval()
    with torch.inference_mode():
        repeat_one = policy(case.batch)
        repeat_two = policy(case.batch)
    for field in dataclasses.fields(PolicyOutput):
        torch.testing.assert_close(
            getattr(repeat_one, field.name),
            getattr(repeat_two, field.name),
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_forward_smoke_preserves_device_dtype_and_identity() -> None:
    case = make_policy_case(candidate_counts=(3,), seed=109)
    expected = case.policy.select(case.batch)
    device = torch.device("cuda")
    policy = case.policy.to(device)
    batch = _to_device(case.batch, device)
    output = policy(batch)
    assert output.candidate_logits.device == device
    assert output.candidate_logits.dtype == torch.float32
    assert policy.select(batch) == expected


def test_every_state_table_is_pooled_and_candidate_axes_are_not_parameters() -> None:
    policy = seeded_policy(seed=113)
    assert tuple(policy.state_table_names) == TABLE_ORDER
    parameter_shapes = tuple(parameter.shape for parameter in policy.parameters())
    assert all(19 not in shape and 384 not in shape for shape in parameter_shapes)
