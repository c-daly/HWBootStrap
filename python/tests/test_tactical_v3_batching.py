from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

import pytest
import torch

from ml_lab.tactical_v3_batching import (
    CANDIDATE_REFERENCE_FIELDS,
    RELATION_KIND_IDS,
    TABLE_BOOLEAN_FIELDS,
    TABLE_CATEGORICAL_FIELDS,
    TABLE_NUMERIC_FIELDS,
    TABLE_ORDER,
    CandidateBatch,
    RaggedBatch,
    RelationNeighborhoodBatch,
    TokenTableBatch,
    collate_decisions,
    collate_examples,
)
from ml_lab.tactical_v3_client import TacticalV3GymClient
from ml_lab.tactical_v3_corpus import StructuredExample, StructuredTarget, load_corpus
from ml_lab.tactical_v3_schema import (
    Candidate,
    CapabilityAllocationToken,
    CapabilityDefinitionToken,
    CellToken,
    MemoryToken,
    ProjectedDelta,
    RelationToken,
    RuleToken,
    TacticalV3Decision,
    TacticalV3Observation,
    TemplateToken,
    TokenRef,
    UnitToken,
)


ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tactical_v3"
CORPUS_ROOT = FIXTURE_ROOT / "tiny-corpus"
SCENARIO_24X16 = FIXTURE_ROOT / "scenario-24x16.json"
CHECKED_IN_SCENARIO = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
SERVER_DLL = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"


def _canonical_example() -> StructuredExample:
    with TacticalV3GymClient(
        ["dotnet", str(SERVER_DLL), "--scenario-file", str(CHECKED_IN_SCENARIO)],
        environment_kind="duel",
    ) as client:
        identity = client.identity
    return load_corpus(CORPUS_ROOT, identity).train[0]


@pytest.fixture(scope="module")
def example_13x9() -> StructuredExample:
    return _canonical_example()


@pytest.fixture(scope="module")
def example_24x16(example_13x9: StructuredExample) -> StructuredExample:
    """Test-only label wrapper around a real Task 2 24x16 client decision."""
    with TacticalV3GymClient(
        ["dotnet", str(SERVER_DLL), "--scenario-file", str(SCENARIO_24X16)],
        environment_kind="tactical",
    ) as client:
        decision = client.reset(41).decision
    target = replace(
        example_13x9.target,
        teacher_candidate_id=decision.candidates[0].candidate_id,
    )
    return replace(example_13x9, decision=decision, target=target)


def _rows(decision: TacticalV3Decision, table: str) -> tuple[object, ...]:
    observation = decision.observation
    return {
        "cells": observation.cells,
        "units": observation.units,
        "templates": observation.templates,
        "capability_definitions": observation.capability_definitions,
        "capability_allocations": observation.capability_allocations,
        "rules": observation.rules,
        "memory_records": observation.memory,
        "relations": observation.relations,
    }[table]


def _assert_tensor_fields_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    for field in fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, Mapping):
            assert tuple(left_value) == tuple(right_value)
            for key in left_value:
                torch.testing.assert_close(
                    left_value[key], right_value[key], rtol=0.0, atol=0.0
                )
        else:
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)


def _minimal_decision() -> TacticalV3Decision:
    cells = (
        CellToken(0, 0, "plains", 1, True, False, "self", True, True, False),
        CellToken(1, 0, "forest", 2, False, True, None, True, True, True),
    )
    units = (
        UnitToken("self", 8, 10, TokenRef("cells", 0), 1, False, True, 2, 1, 5, 7, True),
    )
    templates = (TemplateToken("self", 5, 7, True, True),)
    definitions = (CapabilityDefinitionToken("health"),)
    allocations = (
        CapabilityAllocationToken(
            TokenRef("units", 0), TokenRef("capability_definitions", 0), "health", 10, 10
        ),
    )
    rules = (RuleToken("round", 3, 0.25, False),)
    memory = (MemoryToken(TokenRef("cells", 1), 2, 1, 6, False),)
    relations = (
        RelationToken(
            "neighbor", TokenRef("cells", 0), TokenRef("cells", 1), 0, 0.0, False
        ),
        RelationToken(
            "occupies", TokenRef("units", 0), TokenRef("cells", 0), 0, 0.0, False
        ),
        RelationToken(
            "has_capability",
            TokenRef("units", 0),
            TokenRef("capability_definitions", 0),
            7,
            1.5,
            True,
        ),
    )
    projection = ProjectedDelta(None, None, None, None, 1, 2, -3, 4, True, 5, 6, 7, False)
    candidate = Candidate(9, 17, "end_turn", None, None, None, None, projection)
    return TacticalV3Decision(
        17,
        0,
        TacticalV3Observation(
            cells, units, templates, definitions, allocations, rules, memory, relations
        ),
        (candidate,),
    )


def test_batch_contract_is_frozen_and_feature_schemas_are_explicit() -> None:
    for dto in (TokenTableBatch, RelationNeighborhoodBatch, CandidateBatch, RaggedBatch):
        assert is_dataclass(dto)
        assert dto.__dataclass_params__.frozen

    assert TABLE_ORDER == (
        "cells",
        "units",
        "templates",
        "capability_definitions",
        "capability_allocations",
        "rules",
        "memory_records",
        "relations",
    )
    assert TABLE_NUMERIC_FIELDS == {
        "cells": ("q_centered", "r_centered", "elevation"),
        "units": (
            "current_hp",
            "max_hp",
            "elevation",
            "horizontal_movement_spent",
            "vertical_movement_spent",
            "point_cost",
            "deploy_cost",
        ),
        "templates": ("point_cost", "deploy_cost"),
        "capability_definitions": (),
        "capability_allocations": ("purchased_level", "effective_value"),
        "rules": ("int_value", "float_value"),
        "memory_records": (
            "last_seen_round",
            "observation_age",
            "last_known_current_hp",
        ),
        "relations": ("int_feature", "float_feature"),
    }
    assert TABLE_CATEGORICAL_FIELDS == {
        "cells": ("terrain", "controller"),
        "units": ("owner",),
        "templates": ("owner",),
        "capability_definitions": ("kind",),
        "capability_allocations": ("capability",),
        "rules": ("kind",),
        "memory_records": (),
        "relations": ("kind",),
    }
    assert TABLE_BOOLEAN_FIELDS == {
        "cells": (
            "self_deployment_zone",
            "opponent_deployment_zone",
            "is_boundary",
            "currently_visible",
            "previously_observed",
        ),
        "units": ("moved", "attacked", "currently_visible"),
        "templates": ("is_fixed", "is_deployable"),
        "capability_definitions": (),
        "capability_allocations": (),
        "rules": ("bool_value",),
        "memory_records": ("currently_visible",),
        "relations": ("bool_feature",),
    }
    assert CANDIDATE_REFERENCE_FIELDS == (
        "actor",
        "target",
        "template",
        "cell",
        "projection.source_cell",
        "projection.destination_cell",
        "projection.template",
        "projection.target",
    )
    assert not any("row" in name for names in TABLE_NUMERIC_FIELDS.values() for name in names)


def test_collate_uses_only_batch_maxima_and_preserves_feature_types(
    example_13x9: StructuredExample,
    example_24x16: StructuredExample,
) -> None:
    examples = (example_13x9, example_24x16)
    batch = collate_examples(examples, horizons=(4, 8, 16))

    assert tuple(batch.tables) == TABLE_ORDER
    start = 0
    masks = []
    for table_name in TABLE_ORDER:
        maximum = max(len(_rows(example.decision, table_name)) for example in examples)
        table = batch.tables[table_name]
        assert table.numeric.shape == (
            2,
            maximum,
            len(TABLE_NUMERIC_FIELDS[table_name]),
        )
        assert table.numeric.dtype == torch.float32
        assert table.mask.shape == (2, maximum) and table.mask.dtype == torch.bool
        assert tuple(table.categorical) == TABLE_CATEGORICAL_FIELDS[table_name]
        assert tuple(table.boolean) == TABLE_BOOLEAN_FIELDS[table_name]
        assert all(value.shape == (2, maximum) and value.dtype == torch.int64
                   for value in table.categorical.values())
        assert all(value.shape == (2, maximum) and value.dtype == torch.bool
                   for value in table.boolean.values())
        assert batch.table_slices[table_name] == slice(start, start + maximum)
        masks.append(table.mask)
        start += maximum
    assert batch.node_mask.shape == (2, start)
    assert torch.equal(batch.node_mask, torch.cat(masks, dim=1))
    assert batch.candidates.candidate_id.dtype == torch.int64
    assert batch.candidates.decision_id.dtype == torch.int64
    assert batch.candidates.kind.dtype == torch.int64
    assert batch.candidates.reference_index.dtype == torch.int64
    assert batch.candidates.reference_mask.dtype == torch.bool
    assert batch.candidates.projection_integer.dtype == torch.int64
    assert batch.candidates.projection_boolean.dtype == torch.bool
    assert batch.candidates.mask.dtype == torch.bool
    assert batch.teacher_candidate_index.dtype == torch.int64
    assert batch.terminal_outcome.dtype == torch.int64
    assert batch.horizon_targets.dtype == torch.float32
    assert batch.horizon_target_mask.dtype == torch.bool
    assert batch.remaining_turns.dtype == torch.float32
    assert batch.remaining_turns_mask.dtype == torch.bool


def test_collate_remaps_every_reference_and_hex_neighbor_into_masked_global_nodes(
    example_13x9: StructuredExample,
    example_24x16: StructuredExample,
) -> None:
    examples = (example_13x9, example_24x16)
    batch = collate_examples(examples, horizons=(4, 8, 16))
    assert batch.tables["cells"].mask.sum(dim=1).tolist() == [117, 384]
    cells_slice = batch.table_slices["cells"]
    assert batch.cell_neighbor_index.shape == (2, 384, 6)
    assert batch.cell_neighbor_mask.shape == (2, 384, 6)
    for sample_index, (example, cell_count) in enumerate(
        zip(examples, (117, 384), strict=True)
    ):
        valid = batch.candidates.reference_index[sample_index][
            batch.candidates.reference_mask[sample_index]
        ]
        assert torch.all(valid >= 0)
        assert torch.all(valid < batch.node_mask.shape[1])
        assert batch.node_mask[sample_index, valid].all()

        incoming = batch.neighborhoods.source_index[sample_index][
            batch.neighborhoods.mask[sample_index]
        ]
        assert torch.all(incoming >= 0)
        assert torch.all(incoming < batch.node_mask.shape[1])
        assert batch.node_mask[sample_index, incoming].all()

        neighbor_index = batch.cell_neighbor_index[sample_index]
        neighbor_mask = batch.cell_neighbor_mask[sample_index]
        valid_neighbors = neighbor_index[neighbor_mask]
        assert torch.all(valid_neighbors >= cells_slice.start)
        assert torch.all(valid_neighbors < cells_slice.stop)
        assert batch.node_mask[sample_index, valid_neighbors].all()
        assert not neighbor_mask[cell_count:].any()
        assert torch.all(neighbor_index[~neighbor_mask] == 0)
        for destination_row in range(cell_count):
            actual_sources = neighbor_index[destination_row][
                neighbor_mask[destination_row]
            ].tolist()
            expected_sources = sorted(
                cells_slice.start + relation.source.row
                for relation in example.decision.observation.relations
                if relation.kind == "neighbor" and relation.target.row == destination_row
            )
            assert actual_sources == expected_sources
            assert len(actual_sources) <= 6


def test_coordinates_are_centered_scaled_and_translation_invariant(
    example_13x9: StructuredExample,
) -> None:
    batch = collate_examples((example_13x9,), horizons=(4,))
    coordinates = [(cell.q, cell.r) for cell in example_13x9.decision.observation.cells]
    center_q = sum(q for q, _ in coordinates) / len(coordinates)
    center_r = sum(r for _, r in coordinates) / len(coordinates)
    scale = max(
        1.0,
        max(abs(q - center_q) for q, _ in coordinates),
        max(abs(r - center_r) for _, r in coordinates),
    )
    expected = torch.tensor(
        [[(q - center_q) / scale, (r - center_r) / scale] for q, r in coordinates],
        dtype=torch.float32,
    )
    torch.testing.assert_close(batch.tables["cells"].numeric[0, :, :2], expected)

    shifted_cells = tuple(replace(cell, q=cell.q + 19, r=cell.r - 11)
                          for cell in example_13x9.decision.observation.cells)
    shifted_observation = replace(example_13x9.decision.observation, cells=shifted_cells)
    shifted = replace(example_13x9, decision=replace(
        example_13x9.decision, observation=shifted_observation
    ))
    shifted_batch = collate_examples((shifted,), horizons=(4,))
    torch.testing.assert_close(
        shifted_batch.tables["cells"].numeric[0, :, :2], expected, rtol=0.0, atol=0.0
    )


def test_generic_neighborhoods_have_explicit_reverse_and_allocation_edges() -> None:
    decision = _minimal_decision()
    batch = collate_decisions((decision,), horizons=(4,))
    assert RELATION_KIND_IDS == {
        "neighbor": 0,
        "occupies": 1,
        "has_capability": 2,
        "neighbor_reverse": 3,
        "occupies_reverse": 4,
        "has_capability_reverse": 5,
        "allocation_owner": 6,
        "owner_allocation": 7,
        "allocation_definition": 8,
        "definition_allocation": 9,
    }

    def global_index(table: str, row: int) -> int:
        return batch.table_slices[table].start + row

    actual = []
    for destination in range(batch.node_mask.shape[1]):
        for slot in range(batch.neighborhoods.mask.shape[2]):
            if batch.neighborhoods.mask[0, destination, slot]:
                actual.append((
                    batch.neighborhoods.source_index[0, destination, slot].item(),
                    destination,
                    batch.neighborhoods.kind[0, destination, slot].item(),
                    batch.neighborhoods.int_feature[0, destination, slot].item(),
                    batch.neighborhoods.float_feature[0, destination, slot].item(),
                    batch.neighborhoods.bool_feature[0, destination, slot].item(),
                ))
    c0, c1 = global_index("cells", 0), global_index("cells", 1)
    unit = global_index("units", 0)
    definition = global_index("capability_definitions", 0)
    allocation = global_index("capability_allocations", 0)
    assert actual == sorted([
        (unit, c0, RELATION_KIND_IDS["occupies"], 0, 0.0, False),
        (c1, c0, RELATION_KIND_IDS["neighbor_reverse"], 0, 0.0, False),
        (c0, c1, RELATION_KIND_IDS["neighbor"], 0, 0.0, False),
        (definition, unit, RELATION_KIND_IDS["has_capability_reverse"], 7, 1.5, True),
        (c0, unit, RELATION_KIND_IDS["occupies_reverse"], 0, 0.0, False),
        (allocation, unit, RELATION_KIND_IDS["allocation_owner"], 0, 0.0, False),
        (unit, definition, RELATION_KIND_IDS["has_capability"], 7, 1.5, True),
        (allocation, definition, RELATION_KIND_IDS["allocation_definition"], 0, 0.0, False),
        (unit, allocation, RELATION_KIND_IDS["owner_allocation"], 0, 0.0, False),
        (definition, allocation, RELATION_KIND_IDS["definition_allocation"], 0, 0.0, False),
    ], key=lambda edge: (edge[1], edge[2], edge[0], edge[3], edge[4], edge[5]))
    relation_node = global_index("relations", 0)
    assert not batch.neighborhoods.mask[0, relation_node].any()
    assert torch.all(batch.neighborhoods.source_index[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.kind[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.int_feature[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.float_feature[~batch.neighborhoods.mask] == 0)
    assert not batch.neighborhoods.bool_feature[~batch.neighborhoods.mask].any()


def test_candidate_reference_projection_and_supervised_targets_are_exact(
    example_13x9: StructuredExample,
) -> None:
    target_cases = (
        StructuredTarget(example_13x9.target.teacher_candidate_id, "win", 0, 5, False),
        StructuredTarget(example_13x9.target.teacher_candidate_id, "loss", 0, None, False),
        StructuredTarget(example_13x9.target.teacher_candidate_id, "draw", 0, None, False),
        StructuredTarget(example_13x9.target.teacher_candidate_id, "draw", 0, None, True),
    )
    examples = tuple(replace(example_13x9, target=target) for target in target_cases)
    batch = collate_examples(examples, horizons=(4, 8, 16))
    assert batch.teacher_candidate_index.tolist() == [
        next(index for index, candidate in enumerate(example.decision.candidates)
             if candidate.candidate_id == example.target.teacher_candidate_id)
        for example in examples
    ]
    assert batch.terminal_outcome.tolist() == [2, 0, 1, 1]
    assert batch.horizon_targets.tolist() == [
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ]
    assert batch.horizon_target_mask.tolist() == [
        [True, True, True],
        [True, True, True],
        [True, True, True],
        [False, False, False],
    ]
    assert batch.remaining_turns.tolist() == [5.0, 0.0, 0.0, 0.0]
    assert batch.remaining_turns_mask.tolist() == [True, False, False, False]

    candidate = example_13x9.decision.candidates[0]
    refs = (
        candidate.actor,
        candidate.target,
        candidate.template,
        candidate.cell,
        candidate.projection.source_cell,
        candidate.projection.destination_cell,
        candidate.projection.template,
        candidate.projection.target,
    )
    expected_indices = [
        0 if ref is None else batch.table_slices[ref.table].start + ref.row for ref in refs
    ]
    assert batch.candidates.reference_index[0, 0].tolist() == expected_indices
    assert batch.candidates.reference_mask[0, 0].tolist() == [ref is not None for ref in refs]
    projection = candidate.projection
    assert batch.candidates.projection_integer[0, 0].tolist() == [
        projection.horizontal_movement_spent,
        projection.vertical_movement_spent,
        projection.target_hp_delta,
        projection.damage,
        projection.bounty_delta,
        projection.points_delta,
        projection.round_delta,
    ]
    assert batch.candidates.projection_boolean[0, 0].tolist() == [
        projection.is_lethal,
        projection.is_terminal,
    ]


def test_collate_decisions_matches_features_without_fabricating_targets(
    example_13x9: StructuredExample,
) -> None:
    supervised = collate_examples([example_13x9], horizons=(4, 8, 16))
    inference = collate_decisions([example_13x9.decision], horizons=(4, 8, 16))
    assert tuple(supervised.tables) == tuple(inference.tables)
    for table_name in supervised.tables:
        _assert_tensor_fields_equal(supervised.tables[table_name], inference.tables[table_name])
    assert supervised.table_slices == inference.table_slices
    torch.testing.assert_close(supervised.node_mask, inference.node_mask, rtol=0.0, atol=0.0)
    assert torch.equal(supervised.cell_neighbor_index, inference.cell_neighbor_index)
    assert torch.equal(supervised.cell_neighbor_mask, inference.cell_neighbor_mask)
    _assert_tensor_fields_equal(supervised.neighborhoods, inference.neighborhoods)
    _assert_tensor_fields_equal(supervised.candidates, inference.candidates)
    assert inference.teacher_candidate_index.tolist() == [-1]
    assert inference.terminal_outcome.tolist() == [-1]
    assert not inference.horizon_target_mask.any()
    assert not inference.remaining_turns_mask.any()
    assert torch.count_nonzero(inference.horizon_targets) == 0
    assert torch.count_nonzero(inference.remaining_turns) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("invalid_ref", "reference"),
        ("teacher_missing", "teacher candidate"),
        ("teacher_duplicate", "unique"),
        ("nan", "finite"),
        ("all_masked", "candidate"),
        ("zero_remaining_win", "remaining turns"),
    ),
)
def test_collate_examples_fails_closed_on_malformed_inputs(
    example_13x9: StructuredExample,
    mutation: str,
    message: str,
) -> None:
    example = example_13x9
    if mutation == "invalid_ref":
        unit = replace(example.decision.observation.units[0], cell=TokenRef("cells", 999_999))
        observation = replace(
            example.decision.observation,
            units=(unit, *example.decision.observation.units[1:]),
        )
        example = replace(example, decision=replace(example.decision, observation=observation))
    elif mutation == "teacher_missing":
        example = replace(example, target=replace(example.target, teacher_candidate_id=999_999))
    elif mutation == "teacher_duplicate":
        candidates = list(example.decision.candidates)
        candidates[1] = replace(candidates[1], candidate_id=candidates[0].candidate_id)
        example = replace(example, decision=replace(example.decision, candidates=tuple(candidates)),
                          target=replace(example.target, teacher_candidate_id=candidates[0].candidate_id))
    elif mutation == "nan":
        relation = replace(example.decision.observation.relations[0], float_feature=float("nan"))
        observation = replace(
            example.decision.observation,
            relations=(relation, *example.decision.observation.relations[1:]),
        )
        example = replace(example, decision=replace(example.decision, observation=observation))
    elif mutation == "all_masked":
        example = replace(example, decision=replace(example.decision, candidates=()))
    elif mutation == "zero_remaining_win":
        example = replace(example, target=replace(
            example.target, terminal_outcome="win", remaining_turns_to_victory=0, truncated=False
        ))
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        collate_examples([example], horizons=(4, 8, 16))


def test_collate_rejects_invalid_horizons_and_duplicate_topology() -> None:
    decision = _minimal_decision()
    for horizons in ((), (0,), (4, 4), (8, 4)):
        with pytest.raises(ValueError, match="horizons"):
            collate_decisions((decision,), horizons=horizons)

    relation = decision.observation.relations[0]
    observation = replace(
        decision.observation,
        relations=(relation, relation, *decision.observation.relations[1:]),
    )
    with pytest.raises(ValueError, match="duplicate neighbor"):
        collate_decisions((replace(decision, observation=observation),), horizons=(4,))


def test_collate_rejects_empty_batch_and_nonfinite_inference() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        collate_decisions((), horizons=(4,))
    decision = _minimal_decision()
    bad_rule = replace(decision.observation.rules[0], float_value=float("inf"))
    bad_observation = replace(decision.observation, rules=(bad_rule,))
    with pytest.raises(ValueError, match="finite"):
        collate_decisions((replace(decision, observation=bad_observation),), horizons=(4,))
