from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from types import MappingProxyType

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
    validate_ragged_batch,
)
from ml_lab.tactical_v3_client import TacticalV3GymClient
from ml_lab.tactical_v3_corpus import StructuredExample, StructuredTarget, TeacherEvidence, load_corpus
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
    TacticalV3SemanticIdentity,
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


@dataclass(frozen=True, slots=True)
class _LiveExample:
    example: StructuredExample
    identity: TacticalV3SemanticIdentity


@pytest.fixture(scope="module")
def example_13x9() -> StructuredExample:
    return _canonical_example()


@pytest.fixture(scope="module")
def example_24x16() -> _LiveExample:
    """Test-only example built entirely from one real Task 2 client reset."""
    with TacticalV3GymClient(
        ["dotnet", str(SERVER_DLL), "--scenario-file", str(SCENARIO_24X16)],
        environment_kind="tactical",
    ) as client:
        identity = client.identity
        decision = client.reset(41).decision
    example = StructuredExample(
        1,
        decision,
        StructuredTarget(decision.candidates[0].candidate_id, "draw", 0, None, False),
        TeacherEvidence("test-live-client", 0, 0, 0, "none", None),
        identity.scenario_id,
        identity.contract_hash,
        identity.encoding_hash,
        identity.capacity_hash,
        "test-seed-41-reset",
        41,
        decision.seat,
    )
    return _LiveExample(example, identity)


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

def _example_for_decision(decision: TacticalV3Decision) -> StructuredExample:
    return StructuredExample(
        1,
        decision,
        StructuredTarget(decision.candidates[0].candidate_id, "draw", 0, None, False),
        TeacherEvidence("test-local", 0, 0, 0, "none", None),
        "test-local",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "test-local",
        1,
        decision.seat,
    )


def _batching_identity(*, reach_cell: bool) -> TacticalV3SemanticIdentity:
    match: dict[str, object] = {}
    if reach_cell:
        match["objective"] = MappingProxyType({
            "kind": "reach_cell",
            "target_policy": "seeded_farthest_reachable_unoccupied_v1",
            "radius": 0,
        })
    return TacticalV3SemanticIdentity(
        "test-local",
        1,
        "tactical-v3",
        "a" * 64,
        "b" * 64,
        "c" * 64,
        "duel",
        MappingProxyType(match),
        MappingProxyType({}),
        MappingProxyType({}),
    )


def _move_decision(
    *,
    target: TokenRef | None,
    destination: TokenRef = TokenRef("cells", 1),
    terminal: bool = False,
) -> TacticalV3Decision:
    decision = _minimal_decision()
    projection = ProjectedDelta(
        TokenRef("cells", 0), destination, None, None,
        1, 0, 0, 0, False, 0, 0, 0, terminal,
    )
    move = Candidate(
        0, decision.decision_id, "move", TokenRef("units", 0),
        target, None, destination, projection,
    )
    return replace(decision, candidates=(move,))


def test_collate_decisions_encodes_optional_reach_target_in_existing_slot() -> None:
    decision = _move_decision(
        target=TokenRef("cells", 1),
        terminal=True,
    )
    batch = collate_decisions(
        (decision,),
        horizons=(4,),
        identity=_batching_identity(reach_cell=True),
    )

    assert batch.objective_kind == "reach_cell"
    assert batch.candidates.reference_mask[0, 0, 1].item() is True
    assert batch.candidates.reference_index[0, 0, 1].item() == 1


def test_collate_move_target_semantics_are_bound_to_objective_identity() -> None:
    reach = _move_decision(target=TokenRef("cells", 1), terminal=True)
    legacy = _move_decision(target=None)

    for identity in (None, _batching_identity(reach_cell=False)):
        with pytest.raises(ValueError, match="annihilation move.target"):
            collate_decisions((reach,), horizons=(4,), identity=identity)
    with pytest.raises(ValueError, match="reach_cell move.target is required"):
        collate_decisions(
            (legacy,),
            horizons=(4,),
            identity=_batching_identity(reach_cell=True),
        )


def test_reach_collate_rejects_inconsistent_target_and_nonterminal_completion() -> None:
    identity = _batching_identity(reach_cell=True)
    first = _move_decision(
        target=TokenRef("cells", 1),
        destination=TokenRef("cells", 0),
    )
    second = replace(
        first.candidates[0],
        candidate_id=1,
        target=TokenRef("cells", 0),
    )
    with pytest.raises(ValueError, match="move.target must be consistent"):
        collate_decisions(
            (replace(first, candidates=(*first.candidates, second)),),
            horizons=(4,),
            identity=identity,
        )
    with pytest.raises(ValueError, match="completing move projection must be terminal"):
        collate_decisions(
            (_move_decision(target=TokenRef("cells", 1)),),
            horizons=(4,),
            identity=identity,
        )


def test_collate_examples_cross_checks_explicit_identity() -> None:
    identity = _batching_identity(reach_cell=True)
    example = _example_for_decision(
        _move_decision(target=TokenRef("cells", 1), terminal=True)
    )
    batch = collate_examples(
        (example,), horizons=(4,), identity=identity
    )
    assert batch.objective_kind == "reach_cell"

    with pytest.raises(ValueError, match="identity does not match batching identity"):
        collate_examples(
            (replace(example, contract_hash="d" * 64),),
            horizons=(4,),
            identity=identity,
        )


def test_ragged_validation_does_not_reinterpret_cross_objective_target_masks() -> None:
    reach = collate_decisions(
        (_move_decision(target=TokenRef("cells", 1), terminal=True),),
        horizons=(4,),
        identity=_batching_identity(reach_cell=True),
    )
    legacy = collate_decisions(
        (_move_decision(target=None),),
        horizons=(4,),
    )
    validate_ragged_batch(reach)
    validate_ragged_batch(legacy)

    with pytest.raises(ValueError, match="reference mask"):
        validate_ragged_batch(replace(reach, objective_kind="annihilation"))
    with pytest.raises(ValueError, match="reference mask"):
        validate_ragged_batch(replace(legacy, objective_kind="reach_cell"))


def _neighborhood_edge_set(batch: RaggedBatch) -> set[tuple[int, int, int]]:
    return {
        (batch.neighborhoods.source_index[0, destination, slot].item(), destination,
         batch.neighborhoods.kind[0, destination, slot].item())
        for destination in range(batch.node_mask.shape[1])
        for slot in range(batch.neighborhoods.mask.shape[2])
        if batch.neighborhoods.mask[0, destination, slot]
    }


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
    example_24x16: _LiveExample,
) -> None:
    large = example_24x16.example
    identity = example_24x16.identity
    assert large.scenario_id == identity.scenario_id
    assert large.contract_hash == identity.contract_hash
    assert large.encoding_hash == identity.encoding_hash
    assert large.capacity_hash == identity.capacity_hash
    assert large.contract_hash != example_13x9.contract_hash
    assert large.encoding_hash == example_13x9.encoding_hash
    assert large.capacity_hash == example_13x9.capacity_hash
    examples = (example_13x9, large)
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
    example_24x16: _LiveExample,
) -> None:
    examples = (example_13x9, example_24x16.example)
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
        "unit_cell": 10,
        "cell_unit": 11,
        "memory_cell": 12,
        "cell_memory": 13,
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
    memory = global_index("memory_records", 0)
    assert actual == sorted([
        (unit, c0, RELATION_KIND_IDS["occupies"], 0, 0.0, False),
        (c1, c0, RELATION_KIND_IDS["neighbor_reverse"], 0, 0.0, False),
        (unit, c0, RELATION_KIND_IDS["unit_cell"], 0, 0.0, False),
        (c0, c1, RELATION_KIND_IDS["neighbor"], 0, 0.0, False),
        (definition, unit, RELATION_KIND_IDS["has_capability_reverse"], 7, 1.5, True),
        (c0, unit, RELATION_KIND_IDS["occupies_reverse"], 0, 0.0, False),
        (allocation, unit, RELATION_KIND_IDS["allocation_owner"], 0, 0.0, False),
        (c0, unit, RELATION_KIND_IDS["cell_unit"], 0, 0.0, False),
        (unit, definition, RELATION_KIND_IDS["has_capability"], 7, 1.5, True),
        (allocation, definition, RELATION_KIND_IDS["allocation_definition"], 0, 0.0, False),
        (unit, allocation, RELATION_KIND_IDS["owner_allocation"], 0, 0.0, False),
        (definition, allocation, RELATION_KIND_IDS["definition_allocation"], 0, 0.0, False),
        (memory, c1, RELATION_KIND_IDS["memory_cell"], 0, 0.0, False),
        (c1, memory, RELATION_KIND_IDS["cell_memory"], 0, 0.0, False),
    ], key=lambda edge: (edge[1], edge[2], edge[0], edge[3], edge[4], edge[5]))
    relation_node = global_index("relations", 0)
    assert not batch.neighborhoods.mask[0, relation_node].any()
    assert torch.all(batch.neighborhoods.source_index[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.kind[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.int_feature[~batch.neighborhoods.mask] == 0)
    assert torch.all(batch.neighborhoods.float_feature[~batch.neighborhoods.mask] == 0)
    assert not batch.neighborhoods.bool_feature[~batch.neighborhoods.mask].any()


def test_unit_cell_reference_changes_only_typed_neighborhood_edges() -> None:
    decision = _minimal_decision()
    changed_unit = replace(decision.observation.units[0], cell=TokenRef("cells", 1))
    changed_observation = replace(decision.observation, units=(changed_unit,))
    changed = replace(decision, observation=changed_observation)
    baseline_batch = collate_decisions((decision,), horizons=(4,))
    changed_batch = collate_decisions((changed,), horizons=(4,))

    for table in TABLE_ORDER:
        _assert_tensor_fields_equal(baseline_batch.tables[table], changed_batch.tables[table])
    assert torch.equal(baseline_batch.cell_neighbor_index, changed_batch.cell_neighbor_index)
    assert torch.equal(baseline_batch.cell_neighbor_mask, changed_batch.cell_neighbor_mask)

    cells = baseline_batch.table_slices["cells"]
    unit = baseline_batch.table_slices["units"].start
    baseline_edges = _neighborhood_edge_set(baseline_batch)
    changed_edges = _neighborhood_edge_set(changed_batch)
    assert (unit, cells.start, RELATION_KIND_IDS["occupies"]) in baseline_edges
    assert (unit, cells.start, RELATION_KIND_IDS["occupies"]) in changed_edges
    assert (unit, cells.start, RELATION_KIND_IDS["unit_cell"]) in baseline_edges
    assert (cells.start, unit, RELATION_KIND_IDS["cell_unit"]) in baseline_edges
    assert (unit, cells.start + 1, RELATION_KIND_IDS["unit_cell"]) in changed_edges
    assert (cells.start + 1, unit, RELATION_KIND_IDS["cell_unit"]) in changed_edges
    assert baseline_edges != changed_edges


def test_memory_cell_reference_changes_only_typed_neighborhood_edges() -> None:
    decision = _minimal_decision()
    changed_memory = replace(decision.observation.memory[0], cell=TokenRef("cells", 0))
    changed_observation = replace(decision.observation, memory=(changed_memory,))
    changed = replace(decision, observation=changed_observation)
    baseline_batch = collate_decisions((decision,), horizons=(4,))
    changed_batch = collate_decisions((changed,), horizons=(4,))

    for table in TABLE_ORDER:
        _assert_tensor_fields_equal(baseline_batch.tables[table], changed_batch.tables[table])
    assert torch.equal(baseline_batch.cell_neighbor_index, changed_batch.cell_neighbor_index)
    assert torch.equal(baseline_batch.cell_neighbor_mask, changed_batch.cell_neighbor_mask)

    cells = baseline_batch.table_slices["cells"]
    memory = baseline_batch.table_slices["memory_records"].start
    baseline_edges = _neighborhood_edge_set(baseline_batch)
    changed_edges = _neighborhood_edge_set(changed_batch)
    assert (memory, cells.start + 1, RELATION_KIND_IDS["memory_cell"]) in baseline_edges
    assert (cells.start + 1, memory, RELATION_KIND_IDS["cell_memory"]) in baseline_edges
    assert (memory, cells.start, RELATION_KIND_IDS["memory_cell"]) in changed_edges
    assert (cells.start, memory, RELATION_KIND_IDS["cell_memory"]) in changed_edges
    assert baseline_edges != changed_edges


@pytest.mark.parametrize("supervised", (False, True))
def test_self_neighbor_is_rejected_by_the_shared_validation_path(supervised: bool) -> None:
    decision = _minimal_decision()
    self_neighbor = replace(
        decision.observation.relations[0],
        source=TokenRef("cells", 0),
        target=TokenRef("cells", 0),
    )
    observation = replace(
        decision.observation,
        relations=(self_neighbor, *decision.observation.relations[1:]),
    )
    broken = replace(decision, observation=observation)
    with pytest.raises(ValueError, match="self-neighbor"):
        if supervised:
            collate_examples((_example_for_decision(broken),), horizons=(4,))
        else:
            collate_decisions((broken,), horizons=(4,))


def test_no_true_masked_local_or_generic_edge_is_a_self_edge() -> None:
    batch = collate_decisions((_minimal_decision(),), horizons=(4,))
    cells = batch.table_slices["cells"]
    for destination in range(batch.cell_neighbor_mask.shape[1]):
        mask = batch.cell_neighbor_mask[0, destination]
        assert torch.all(
            batch.cell_neighbor_index[0, destination][mask] != cells.start + destination
        )
    destinations = torch.arange(batch.node_mask.shape[1], dtype=torch.int64).unsqueeze(1)
    destinations = destinations.expand_as(batch.neighborhoods.source_index[0])
    mask = batch.neighborhoods.mask[0]
    assert torch.all(batch.neighborhoods.source_index[0][mask] != destinations[mask])


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
