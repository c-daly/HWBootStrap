"""Typed, ragged tactical-v3 batching with explicit reference remapping."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import math
from types import MappingProxyType

import torch

from .tactical_v3_corpus import StructuredExample, StructuredTarget
from .tactical_v3_schema import (
    CANDIDATE_REFERENCE_FAMILIES,
    PROJECTED_REFERENCE_FAMILIES,
    Candidate,
    TacticalV3Decision,
    TokenRef,
)


TABLE_ORDER = (
    "cells",
    "units",
    "templates",
    "capability_definitions",
    "capability_allocations",
    "rules",
    "memory_records",
    "relations",
)

TABLE_NUMERIC_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cells": ("q_centered", "r_centered", "elevation"),
    "units": (
        "current_hp", "max_hp", "elevation", "horizontal_movement_spent",
        "vertical_movement_spent", "point_cost", "deploy_cost",
    ),
    "templates": ("point_cost", "deploy_cost"),
    "capability_definitions": (),
    "capability_allocations": ("purchased_level", "effective_value"),
    "rules": ("int_value", "float_value"),
    "memory_records": ("last_seen_round", "observation_age", "last_known_current_hp"),
    "relations": ("int_feature", "float_feature"),
})

TABLE_CATEGORICAL_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cells": ("terrain", "controller"),
    "units": ("owner",),
    "templates": ("owner",),
    "capability_definitions": ("kind",),
    "capability_allocations": ("capability",),
    "rules": ("kind",),
    "memory_records": (),
    "relations": ("kind",),
})

TABLE_BOOLEAN_FIELDS: Mapping[str, tuple[str, ...]] = MappingProxyType({
    "cells": (
        "self_deployment_zone", "opponent_deployment_zone", "is_boundary",
        "currently_visible", "previously_observed",
    ),
    "units": ("moved", "attacked", "currently_visible"),
    "templates": ("is_fixed", "is_deployable"),
    "capability_definitions": (),
    "capability_allocations": (),
    "rules": ("bool_value",),
    "memory_records": ("currently_visible",),
    "relations": ("bool_feature",),
})

CANDIDATE_REFERENCE_FIELDS = (
    "actor",
    "target",
    "template",
    "cell",
    "projection.source_cell",
    "projection.destination_cell",
    "projection.template",
    "projection.target",
)

_OWNERS = ("self", "opponent")
_TERRAINS = ("plains", "forest", "rough", "water")
_CAPABILITIES = (
    "health", "damage", "defense", "movement", "vertical_movement", "range",
    "range_arc", "vision", "vision_arc",
)
_RULES = (
    "win_conditions", "round", "round_cap", "actions_per_turn", "starting_points",
    "self_points", "opponent_points", "damage_floor", "damage_high_ground_bonus",
    "range_high_ground_bonus", "bounty_rate", "deploy_cost_multiplier", "fog_of_war",
    "max_design_point_cost", "design_fee",
)
_WIRE_RELATIONS = ("neighbor", "occupies", "has_capability")
_CANDIDATE_KINDS = ("attack", "move", "deploy", "end_turn")

TABLE_CATEGORICAL_CARDINALITIES: Mapping[str, Mapping[str, int]] = MappingProxyType({
    "cells": MappingProxyType({"terrain": len(_TERRAINS), "controller": len(_OWNERS) + 1}),
    "units": MappingProxyType({"owner": len(_OWNERS)}),
    "templates": MappingProxyType({"owner": len(_OWNERS)}),
    "capability_definitions": MappingProxyType({"kind": len(_CAPABILITIES)}),
    "capability_allocations": MappingProxyType({"capability": len(_CAPABILITIES)}),
    "rules": MappingProxyType({"kind": len(_RULES)}),
    "memory_records": MappingProxyType({}),
    "relations": MappingProxyType({"kind": len(_WIRE_RELATIONS)}),
})

RELATION_KIND_IDS: Mapping[str, int] = MappingProxyType({
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
})
RELATION_KIND_COUNT = len(RELATION_KIND_IDS)

_CATEGORY_IDS = {
    "terrain": {value: index for index, value in enumerate(_TERRAINS)},
    "controller": {**{value: index for index, value in enumerate(_OWNERS)}, None: len(_OWNERS)},
    "owner": {value: index for index, value in enumerate(_OWNERS)},
    "capability": {value: index for index, value in enumerate(_CAPABILITIES)},
    "capability_kind": {value: index for index, value in enumerate(_CAPABILITIES)},
    "rule_kind": {value: index for index, value in enumerate(_RULES)},
    "relation_kind": {value: index for index, value in enumerate(_WIRE_RELATIONS)},
    "candidate_kind": {value: index for index, value in enumerate(_CANDIDATE_KINDS)},
}


@dataclass(frozen=True, slots=True)
class TokenTableBatch:
    numeric: torch.Tensor
    categorical: Mapping[str, torch.Tensor]
    boolean: Mapping[str, torch.Tensor]
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RelationNeighborhoodBatch:
    source_index: torch.Tensor
    kind: torch.Tensor
    int_feature: torch.Tensor
    float_feature: torch.Tensor
    bool_feature: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class CandidateBatch:
    candidate_id: torch.Tensor
    decision_id: torch.Tensor
    kind: torch.Tensor
    reference_index: torch.Tensor
    reference_mask: torch.Tensor
    projection_integer: torch.Tensor
    projection_boolean: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class RaggedBatch:
    tables: Mapping[str, TokenTableBatch]
    table_slices: Mapping[str, slice]
    node_mask: torch.Tensor
    cell_neighbor_index: torch.Tensor
    cell_neighbor_mask: torch.Tensor
    neighborhoods: RelationNeighborhoodBatch
    candidates: CandidateBatch
    teacher_candidate_index: torch.Tensor
    terminal_outcome: torch.Tensor
    horizon_targets: torch.Tensor
    horizon_target_mask: torch.Tensor
    remaining_turns: torch.Tensor
    remaining_turns_mask: torch.Tensor


def _validate_horizons(horizons: tuple[int, ...]) -> None:
    if (
        type(horizons) is not tuple
        or not horizons
        or any(type(value) is not int or value <= 0 for value in horizons)
        or any(left >= right for left, right in zip(horizons, horizons[1:]))
    ):
        raise ValueError("horizons must be a non-empty strictly increasing tuple of positive integers")


def _int32(value: object, field: str) -> int:
    if type(value) is not int or not -(2**31) <= value < 2**31:
        raise ValueError(f"{field} must be an int32")
    return value


def _int64(value: object, field: str) -> int:
    if type(value) is not int or not -(2**63) <= value < 2**63:
        raise ValueError(f"{field} must be an int64")
    return value


def _boolean(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a bool")
    return value


def _finite(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _category(value: object, values: tuple[str, ...], field: str) -> str:
    if type(value) is not str or value not in values:
        raise ValueError(f"{field} has an unknown categorical value")
    return value


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


def _counts(decision: TacticalV3Decision) -> dict[str, int]:
    result = {table: len(_rows(decision, table)) for table in TABLE_ORDER}
    result["candidates"] = len(decision.candidates)
    return result


def _validate_ref(
    ref: TokenRef | None,
    counts: Mapping[str, int],
    field: str,
    expected: str | tuple[str, ...] | None,
) -> None:
    if expected is None:
        if ref is not None:
            raise ValueError(f"{field} reference must be absent")
        return
    if ref is None:
        raise ValueError(f"{field} reference is required")
    if type(ref) is not TokenRef:
        raise ValueError(f"{field} reference must be TokenRef")
    families = (expected,) if type(expected) is str else expected
    if ref.table not in families:
        raise ValueError(f"{field} reference has the wrong table")
    if type(ref.row) is not int or ref.row < 0 or ref.row >= counts.get(ref.table, 0):
        raise ValueError(f"{field} reference row is out of range")


def _validate_decision(decision: TacticalV3Decision, sample: int) -> None:
    if type(decision) is not TacticalV3Decision:
        raise ValueError(f"decisions[{sample}] must be TacticalV3Decision")
    _int64(decision.decision_id, f"decisions[{sample}].decision_id")
    _int32(decision.seat, f"decisions[{sample}].seat")
    if decision.seat not in {0, 1}:
        raise ValueError(f"decisions[{sample}].seat must be 0 or 1")
    if not decision.candidates:
        raise ValueError(f"decisions[{sample}] must contain at least one candidate")
    counts = _counts(decision)
    observation = decision.observation

    for row, cell in enumerate(observation.cells):
        _int32(cell.q, f"cells[{row}].q")
        _int32(cell.r, f"cells[{row}].r")
        _category(cell.terrain, _TERRAINS, f"cells[{row}].terrain")
        _int32(cell.elevation, f"cells[{row}].elevation")
        if cell.controller is not None:
            _category(cell.controller, _OWNERS, f"cells[{row}].controller")
        for name in TABLE_BOOLEAN_FIELDS["cells"]:
            _boolean(getattr(cell, name), f"cells[{row}].{name}")

    for row, unit in enumerate(observation.units):
        _category(unit.owner, _OWNERS, f"units[{row}].owner")
        for name in TABLE_NUMERIC_FIELDS["units"]:
            _int32(getattr(unit, name), f"units[{row}].{name}")
        for name in TABLE_BOOLEAN_FIELDS["units"]:
            _boolean(getattr(unit, name), f"units[{row}].{name}")
        _validate_ref(unit.cell, counts, f"units[{row}].cell", "cells")

    for row, template in enumerate(observation.templates):
        _category(template.owner, _OWNERS, f"templates[{row}].owner")
        for name in TABLE_NUMERIC_FIELDS["templates"]:
            _int32(getattr(template, name), f"templates[{row}].{name}")
        for name in TABLE_BOOLEAN_FIELDS["templates"]:
            _boolean(getattr(template, name), f"templates[{row}].{name}")

    for row, definition in enumerate(observation.capability_definitions):
        _category(definition.kind, _CAPABILITIES, f"capability_definitions[{row}].kind")

    for row, allocation in enumerate(observation.capability_allocations):
        _validate_ref(
            allocation.owner, counts, f"capability_allocations[{row}].owner", ("units", "templates")
        )
        _validate_ref(
            allocation.definition, counts, f"capability_allocations[{row}].definition",
            "capability_definitions",
        )
        _category(allocation.capability, _CAPABILITIES, f"capability_allocations[{row}].capability")
        _int32(allocation.purchased_level, f"capability_allocations[{row}].purchased_level")
        _int32(allocation.effective_value, f"capability_allocations[{row}].effective_value")

    for row, rule in enumerate(observation.rules):
        _category(rule.kind, _RULES, f"rules[{row}].kind")
        _int32(rule.int_value, f"rules[{row}].int_value")
        _finite(rule.float_value, f"rules[{row}].float_value")
        _boolean(rule.bool_value, f"rules[{row}].bool_value")

    for row, memory in enumerate(observation.memory):
        _validate_ref(memory.cell, counts, f"memory_records[{row}].cell", "cells")
        for name in TABLE_NUMERIC_FIELDS["memory_records"]:
            _int32(getattr(memory, name), f"memory_records[{row}].{name}")
        _boolean(memory.currently_visible, f"memory_records[{row}].currently_visible")

    relation_families = {
        "neighbor": ("cells", "cells"),
        "occupies": ("units", "cells"),
        "has_capability": (("units", "templates"), "capability_definitions"),
    }
    for row, relation in enumerate(observation.relations):
        _category(relation.kind, _WIRE_RELATIONS, f"relations[{row}].kind")
        source_family, target_family = relation_families[relation.kind]
        _validate_ref(relation.source, counts, f"relations[{row}].source", source_family)
        _validate_ref(relation.target, counts, f"relations[{row}].target", target_family)
        if relation.kind == "neighbor" and relation.source == relation.target:
            raise ValueError("self-neighbor relation is not allowed")
        _int32(relation.int_feature, f"relations[{row}].int_feature")
        _finite(relation.float_feature, f"relations[{row}].float_feature")
        _boolean(relation.bool_feature, f"relations[{row}].bool_feature")

    candidate_ids: set[int] = set()
    for row, candidate in enumerate(decision.candidates):
        if type(candidate) is not Candidate:
            raise ValueError(f"candidates[{row}] must be Candidate")
        candidate_id = _int32(candidate.candidate_id, f"candidates[{row}].candidate_id")
        if candidate_id in candidate_ids:
            raise ValueError("candidate identity must be unique within a decision")
        candidate_ids.add(candidate_id)
        if _int64(candidate.decision_id, f"candidates[{row}].decision_id") != decision.decision_id:
            raise ValueError("candidate decision_id does not match decision_id")
        _category(candidate.kind, _CANDIDATE_KINDS, f"candidates[{row}].kind")
        for field, family in CANDIDATE_REFERENCE_FAMILIES[candidate.kind].items():
            _validate_ref(getattr(candidate, field), counts, f"candidates[{row}].{field}", family)
        for field, family in PROJECTED_REFERENCE_FAMILIES[candidate.kind].items():
            _validate_ref(
                getattr(candidate.projection, field), counts,
                f"candidates[{row}].projection.{field}", family,
            )
        for name in (
            "horizontal_movement_spent", "vertical_movement_spent", "target_hp_delta",
            "damage", "bounty_delta", "points_delta", "round_delta",
        ):
            _int32(getattr(candidate.projection, name), f"candidates[{row}].projection.{name}")
        _boolean(candidate.projection.is_lethal, f"candidates[{row}].projection.is_lethal")
        _boolean(candidate.projection.is_terminal, f"candidates[{row}].projection.is_terminal")


def _global_ref(
    sample: int,
    ref: TokenRef,
    reference_map: Mapping[tuple[int, str, int], int],
    node_mask: torch.Tensor,
) -> int:
    key = (sample, ref.table, ref.row)
    if key not in reference_map:
        raise ValueError("reference does not identify a batched node")
    result = reference_map[key]
    if not bool(node_mask[sample, result]):
        raise ValueError("reference does not land on a valid node mask")
    return result


def _encoded_row(table: str, row: object, coordinate: tuple[float, float] | None):
    if table == "cells":
        assert coordinate is not None
        numeric = (*coordinate, float(row.elevation))
        categorical = {
            "terrain": _CATEGORY_IDS["terrain"][row.terrain],
            "controller": _CATEGORY_IDS["controller"][row.controller],
        }
    elif table == "units":
        numeric = tuple(float(getattr(row, name)) for name in TABLE_NUMERIC_FIELDS[table])
        categorical = {"owner": _CATEGORY_IDS["owner"][row.owner]}
    elif table == "templates":
        numeric = tuple(float(getattr(row, name)) for name in TABLE_NUMERIC_FIELDS[table])
        categorical = {"owner": _CATEGORY_IDS["owner"][row.owner]}
    elif table == "capability_definitions":
        numeric = ()
        categorical = {"kind": _CATEGORY_IDS["capability_kind"][row.kind]}
    elif table == "capability_allocations":
        numeric = tuple(float(getattr(row, name)) for name in TABLE_NUMERIC_FIELDS[table])
        categorical = {"capability": _CATEGORY_IDS["capability"][row.capability]}
    elif table == "rules":
        numeric = (float(row.int_value), float(row.float_value))
        categorical = {"kind": _CATEGORY_IDS["rule_kind"][row.kind]}
    elif table == "memory_records":
        numeric = tuple(float(getattr(row, name)) for name in TABLE_NUMERIC_FIELDS[table])
        categorical = {}
    elif table == "relations":
        numeric = (float(row.int_feature), float(row.float_feature))
        categorical = {"kind": _CATEGORY_IDS["relation_kind"][row.kind]}
    else:
        raise AssertionError(table)
    boolean = {name: bool(getattr(row, name)) for name in TABLE_BOOLEAN_FIELDS[table]}
    return numeric, categorical, boolean


def _coordinate_features(decision: TacticalV3Decision) -> tuple[tuple[float, float], ...]:
    cells = decision.observation.cells
    if not cells:
        return ()
    center_q = sum(cell.q for cell in cells) / len(cells)
    center_r = sum(cell.r for cell in cells) / len(cells)
    scale = max(
        1.0,
        max(abs(cell.q - center_q) for cell in cells),
        max(abs(cell.r - center_r) for cell in cells),
    )
    result = tuple(((cell.q - center_q) / scale, (cell.r - center_r) / scale) for cell in cells)
    if not all(math.isfinite(value) for pair in result for value in pair):
        raise ValueError("centered cell coordinates must be finite")
    return result


def _build_tables(decisions: tuple[TacticalV3Decision, ...]):
    batch_size = len(decisions)
    maxima = {table: max(len(_rows(decision, table)) for decision in decisions) for table in TABLE_ORDER}
    table_slices: dict[str, slice] = {}
    start = 0
    for table in TABLE_ORDER:
        table_slices[table] = slice(start, start + maxima[table])
        start += maxima[table]

    node_mask = torch.zeros((batch_size, start), dtype=torch.bool)
    reference_map: dict[tuple[int, str, int], int] = {}
    tables: dict[str, TokenTableBatch] = {}
    coordinates = tuple(_coordinate_features(decision) for decision in decisions)
    for table in TABLE_ORDER:
        maximum = maxima[table]
        numeric = torch.zeros(
            (batch_size, maximum, len(TABLE_NUMERIC_FIELDS[table])), dtype=torch.float32
        )
        categorical = {
            name: torch.zeros((batch_size, maximum), dtype=torch.int64)
            for name in TABLE_CATEGORICAL_FIELDS[table]
        }
        boolean = {
            name: torch.zeros((batch_size, maximum), dtype=torch.bool)
            for name in TABLE_BOOLEAN_FIELDS[table]
        }
        mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
        for sample, decision in enumerate(decisions):
            for row_index, row in enumerate(_rows(decision, table)):
                values, categories, flags = _encoded_row(
                    table, row, coordinates[sample][row_index] if table == "cells" else None
                )
                if values:
                    numeric[sample, row_index] = torch.tensor(values, dtype=torch.float32)
                for name, value in categories.items():
                    categorical[name][sample, row_index] = value
                for name, value in flags.items():
                    boolean[name][sample, row_index] = value
                mask[sample, row_index] = True
                global_index = table_slices[table].start + row_index
                node_mask[sample, global_index] = True
                reference_map[(sample, table, row_index)] = global_index
        if not torch.isfinite(numeric).all():
            raise ValueError(f"{table} numeric features must be finite")
        tables[table] = TokenTableBatch(
            numeric,
            MappingProxyType(categorical),
            MappingProxyType(boolean),
            mask,
        )
    return (
        MappingProxyType(tables),
        MappingProxyType(table_slices),
        node_mask,
        reference_map,
    )


def _build_candidates(
    decisions: tuple[TacticalV3Decision, ...],
    reference_map: Mapping[tuple[int, str, int], int],
    node_mask: torch.Tensor,
) -> CandidateBatch:
    batch_size = len(decisions)
    maximum = max(len(decision.candidates) for decision in decisions)
    candidate_id = torch.zeros((batch_size, maximum), dtype=torch.int64)
    decision_id = torch.zeros((batch_size, maximum), dtype=torch.int64)
    kind = torch.zeros((batch_size, maximum), dtype=torch.int64)
    reference_index = torch.zeros((batch_size, maximum, 8), dtype=torch.int64)
    reference_mask = torch.zeros((batch_size, maximum, 8), dtype=torch.bool)
    projection_integer = torch.zeros((batch_size, maximum, 7), dtype=torch.int64)
    projection_boolean = torch.zeros((batch_size, maximum, 2), dtype=torch.bool)
    mask = torch.zeros((batch_size, maximum), dtype=torch.bool)
    for sample, decision in enumerate(decisions):
        for row, candidate in enumerate(decision.candidates):
            candidate_id[sample, row] = candidate.candidate_id
            decision_id[sample, row] = candidate.decision_id
            kind[sample, row] = _CATEGORY_IDS["candidate_kind"][candidate.kind]
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
            for slot, ref in enumerate(refs):
                if ref is not None:
                    reference_index[sample, row, slot] = _global_ref(
                        sample, ref, reference_map, node_mask
                    )
                    reference_mask[sample, row, slot] = True
            projection = candidate.projection
            projection_integer[sample, row] = torch.tensor((
                projection.horizontal_movement_spent,
                projection.vertical_movement_spent,
                projection.target_hp_delta,
                projection.damage,
                projection.bounty_delta,
                projection.points_delta,
                projection.round_delta,
            ), dtype=torch.int64)
            projection_boolean[sample, row] = torch.tensor(
                (projection.is_lethal, projection.is_terminal), dtype=torch.bool
            )
            mask[sample, row] = True
    return CandidateBatch(
        candidate_id, decision_id, kind, reference_index, reference_mask,
        projection_integer, projection_boolean, mask,
    )


def _build_cell_neighbors(
    decisions: tuple[TacticalV3Decision, ...],
    table_slices: Mapping[str, slice],
    reference_map: Mapping[tuple[int, str, int], int],
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(decisions)
    maximum_cells = table_slices["cells"].stop - table_slices["cells"].start
    index = torch.zeros((batch_size, maximum_cells, 6), dtype=torch.int64)
    mask = torch.zeros((batch_size, maximum_cells, 6), dtype=torch.bool)
    for sample, decision in enumerate(decisions):
        incoming: dict[int, list[int]] = {
            row: [] for row in range(len(decision.observation.cells))
        }
        for relation in decision.observation.relations:
            if relation.kind != "neighbor":
                continue
            source = _global_ref(sample, relation.source, reference_map, node_mask)
            sources = incoming[relation.target.row]
            if source in sources:
                raise ValueError("duplicate neighbor source for one destination")
            sources.append(source)
        for destination, sources in incoming.items():
            sources.sort()
            if len(sources) > 6:
                raise ValueError("cell has more than six neighbor sources")
            if sources:
                index[sample, destination, :len(sources)] = torch.tensor(sources, dtype=torch.int64)
                mask[sample, destination, :len(sources)] = True
    valid = index[mask]
    cells = table_slices["cells"]
    if valid.numel() and (
        not torch.all(valid >= cells.start)
        or not torch.all(valid < cells.stop)
    ):
        raise ValueError("cell neighbor reference is outside the canonical cells slice")
    for sample in range(batch_size):
        sample_valid = index[sample][mask[sample]]
        if sample_valid.numel() and not node_mask[sample, sample_valid].all():
            raise ValueError("cell neighbor reference does not land on a valid node mask")
    return index, mask


def _build_neighborhoods(
    decisions: tuple[TacticalV3Decision, ...],
    reference_map: Mapping[tuple[int, str, int], int],
    node_mask: torch.Tensor,
) -> RelationNeighborhoodBatch:
    all_edges: list[list[tuple[int, int, int, int, float, bool]]] = []
    maximum_incoming = 0
    for sample, decision in enumerate(decisions):
        edges: list[tuple[int, int, int, int, float, bool]] = []
        for relation in decision.observation.relations:
            source = _global_ref(sample, relation.source, reference_map, node_mask)
            destination = _global_ref(sample, relation.target, reference_map, node_mask)
            feature = (
                relation.int_feature,
                float(relation.float_feature),
                relation.bool_feature,
            )
            edges.append((
                destination, RELATION_KIND_IDS[relation.kind], source, *feature
            ))
            edges.append((
                source, RELATION_KIND_IDS[f"{relation.kind}_reverse"], destination, *feature
            ))
        for row, unit in enumerate(decision.observation.units):
            unit_index = reference_map[(sample, "units", row)]
            cell = _global_ref(sample, unit.cell, reference_map, node_mask)
            edges.extend((
                (cell, RELATION_KIND_IDS["unit_cell"], unit_index, 0, 0.0, False),
                (unit_index, RELATION_KIND_IDS["cell_unit"], cell, 0, 0.0, False),
            ))
        for row, memory in enumerate(decision.observation.memory):
            memory_index = reference_map[(sample, "memory_records", row)]
            cell = _global_ref(sample, memory.cell, reference_map, node_mask)
            edges.extend((
                (cell, RELATION_KIND_IDS["memory_cell"], memory_index, 0, 0.0, False),
                (memory_index, RELATION_KIND_IDS["cell_memory"], cell, 0, 0.0, False),
            ))
        for row, allocation in enumerate(decision.observation.capability_allocations):
            allocation_index = reference_map[(sample, "capability_allocations", row)]
            owner = _global_ref(sample, allocation.owner, reference_map, node_mask)
            definition = _global_ref(sample, allocation.definition, reference_map, node_mask)
            edges.extend((
                (owner, RELATION_KIND_IDS["allocation_owner"], allocation_index, 0, 0.0, False),
                (allocation_index, RELATION_KIND_IDS["owner_allocation"], owner, 0, 0.0, False),
                (definition, RELATION_KIND_IDS["allocation_definition"], allocation_index, 0, 0.0, False),
                (allocation_index, RELATION_KIND_IDS["definition_allocation"], definition, 0, 0.0, False),
            ))
        edges.sort(key=lambda edge: edge)
        counts: dict[int, int] = {}
        for destination, *_ in edges:
            counts[destination] = counts.get(destination, 0) + 1
        maximum_incoming = max(maximum_incoming, max(counts.values(), default=0))
        all_edges.append(edges)

    batch_size, nodes = node_mask.shape
    slots = max(1, maximum_incoming)
    source_index = torch.zeros((batch_size, nodes, slots), dtype=torch.int64)
    kind = torch.zeros((batch_size, nodes, slots), dtype=torch.int64)
    int_feature = torch.zeros((batch_size, nodes, slots), dtype=torch.int64)
    float_feature = torch.zeros((batch_size, nodes, slots), dtype=torch.float32)
    bool_feature = torch.zeros((batch_size, nodes, slots), dtype=torch.bool)
    mask = torch.zeros((batch_size, nodes, slots), dtype=torch.bool)
    for sample, edges in enumerate(all_edges):
        positions: dict[int, int] = {}
        for destination, relation_kind, source, integer, floating, boolean in edges:
            slot = positions.get(destination, 0)
            positions[destination] = slot + 1
            source_index[sample, destination, slot] = source
            kind[sample, destination, slot] = relation_kind
            int_feature[sample, destination, slot] = integer
            float_feature[sample, destination, slot] = floating
            bool_feature[sample, destination, slot] = boolean
            mask[sample, destination, slot] = True
    if not torch.isfinite(float_feature).all():
        raise ValueError("relation float features must be finite")
    for sample in range(batch_size):
        valid_sources = source_index[sample][mask[sample]]
        if valid_sources.numel() and not node_mask[sample, valid_sources].all():
            raise ValueError("relation source reference does not land on a valid node mask")
        destinations = mask[sample].any(dim=1)
        if destinations.any() and not node_mask[sample, destinations].all():
            raise ValueError("relation destination reference does not land on a valid node mask")
    return RelationNeighborhoodBatch(
        source_index, kind, int_feature, float_feature, bool_feature, mask
    )


def _collate_features(
    decisions: Sequence[TacticalV3Decision], horizons: tuple[int, ...]
) -> RaggedBatch:
    _validate_horizons(horizons)
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence) or not decisions:
        raise ValueError("decisions must be a non-empty sequence")
    frozen = tuple(decisions)
    for sample, decision in enumerate(frozen):
        _validate_decision(decision, sample)
    tables, table_slices, node_mask, reference_map = _build_tables(frozen)
    candidates = _build_candidates(frozen, reference_map, node_mask)
    neighbor_index, neighbor_mask = _build_cell_neighbors(
        frozen, table_slices, reference_map, node_mask
    )
    neighborhoods = _build_neighborhoods(frozen, reference_map, node_mask)
    batch_size = len(frozen)
    horizon_count = len(horizons)
    return RaggedBatch(
        tables=tables,
        table_slices=table_slices,
        node_mask=node_mask,
        cell_neighbor_index=neighbor_index,
        cell_neighbor_mask=neighbor_mask,
        neighborhoods=neighborhoods,
        candidates=candidates,
        teacher_candidate_index=torch.full((batch_size,), -1, dtype=torch.int64),
        terminal_outcome=torch.full((batch_size,), -1, dtype=torch.int64),
        horizon_targets=torch.zeros((batch_size, horizon_count), dtype=torch.float32),
        horizon_target_mask=torch.zeros((batch_size, horizon_count), dtype=torch.bool),
        remaining_turns=torch.zeros((batch_size,), dtype=torch.float32),
        remaining_turns_mask=torch.zeros((batch_size,), dtype=torch.bool),
    )


def collate_decisions(
    decisions: Sequence[TacticalV3Decision], horizons: tuple[int, ...]
) -> RaggedBatch:
    """Collate inference decisions with sentinel, fully masked supervision targets."""
    return _collate_features(decisions, horizons)


def _validate_target(
    target: StructuredTarget,
    decision: TacticalV3Decision,
    sample: int,
) -> int:
    if type(target) is not StructuredTarget:
        raise ValueError(f"examples[{sample}].target must be StructuredTarget")
    teacher = _int32(target.teacher_candidate_id, "target.teacher_candidate_id")
    matches = [
        row for row, candidate in enumerate(decision.candidates)
        if candidate.candidate_id == teacher
    ]
    if not matches:
        raise ValueError("teacher candidate does not identify a candidate in its decision")
    if len(matches) != 1:
        raise ValueError("teacher candidate must map to one unique candidate row")
    if target.terminal_outcome not in {"loss", "draw", "win"}:
        raise ValueError("target terminal_outcome must be loss, draw, or win")
    if _int32(target.trajectory_index, "target.trajectory_index") < 0:
        raise ValueError("target trajectory_index must be nonnegative")
    _boolean(target.truncated, "target.truncated")
    remaining = target.remaining_turns_to_victory
    if target.terminal_outcome == "win":
        if remaining is None or _int32(remaining, "target.remaining_turns_to_victory") <= 0:
            raise ValueError("remaining turns must be a positive integer for a win")
    elif remaining is not None:
        raise ValueError("remaining turns must be absent for a loss or draw")
    return matches[0]


def collate_examples(
    examples: Sequence[StructuredExample], horizons: tuple[int, ...]
) -> RaggedBatch:
    """Collate supervised examples through the inference-identical feature path."""
    if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence) or not examples:
        raise ValueError("examples must be a non-empty sequence")
    frozen = tuple(examples)
    if any(type(example) is not StructuredExample for example in frozen):
        raise ValueError("examples must contain only StructuredExample values")
    batch = _collate_features(tuple(example.decision for example in frozen), horizons)
    teacher = torch.empty((len(frozen),), dtype=torch.int64)
    outcome = torch.empty((len(frozen),), dtype=torch.int64)
    horizon_targets = torch.zeros((len(frozen), len(horizons)), dtype=torch.float32)
    horizon_mask = torch.zeros((len(frozen), len(horizons)), dtype=torch.bool)
    remaining_turns = torch.zeros((len(frozen),), dtype=torch.float32)
    remaining_mask = torch.zeros((len(frozen),), dtype=torch.bool)
    outcomes = {"loss": 0, "draw": 1, "win": 2}
    for sample, example in enumerate(frozen):
        target = example.target
        teacher[sample] = _validate_target(target, example.decision, sample)
        outcome[sample] = outcomes[target.terminal_outcome]
        if not target.truncated:
            horizon_mask[sample] = True
            if target.terminal_outcome == "win":
                assert target.remaining_turns_to_victory is not None
                horizon_targets[sample] = torch.tensor(
                    [float(target.remaining_turns_to_victory <= horizon) for horizon in horizons],
                    dtype=torch.float32,
                )
                remaining_turns[sample] = float(target.remaining_turns_to_victory)
                remaining_mask[sample] = True
    if not torch.isfinite(horizon_targets).all() or not torch.isfinite(remaining_turns).all():
        raise ValueError("supervision targets must be finite")
    return replace(
        batch,
        teacher_candidate_index=teacher,
        terminal_outcome=outcome,
        horizon_targets=horizon_targets,
        horizon_target_mask=horizon_mask,
        remaining_turns=remaining_turns,
        remaining_turns_mask=remaining_mask,
    )


def validate_ragged_batch(batch: RaggedBatch) -> None:
    """Validate the complete immutable ragged tensor contract."""
    if type(batch) is not RaggedBatch:
        raise ValueError("batch must be RaggedBatch")

    def require_tensor(value: object, name: str) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        return value

    def require_exact(
        value: torch.Tensor,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        name: str,
        device: torch.device,
    ) -> None:
        if value.shape != shape:
            raise ValueError(f"{name} shape must be {shape}")
        if value.dtype != dtype:
            raise ValueError(f"{name} dtype must be {dtype}")
        if value.device != device:
            raise ValueError(f"{name} must be on the candidate mask device")

    if type(batch.tables) is not MappingProxyType or tuple(batch.tables) != TABLE_ORDER:
        raise ValueError("ragged batch tables keys/order are invalid")
    if (
        type(batch.table_slices) is not MappingProxyType
        or tuple(batch.table_slices) != TABLE_ORDER
    ):
        raise ValueError("ragged batch table_slices keys/order are invalid")

    node_mask = require_tensor(batch.node_mask, "node_mask")
    if node_mask.ndim != 2 or node_mask.shape[0] <= 0:
        raise ValueError("node_mask shape must be [B, N]")
    if node_mask.dtype != torch.bool:
        raise ValueError("node_mask dtype must be torch.bool")
    device = node_mask.device
    batch_size, node_count = node_mask.shape

    expected_start = 0
    table_masks: list[torch.Tensor] = []
    for table_name in TABLE_ORDER:
        table_slice = batch.table_slices[table_name]
        if (
            type(table_slice) is not slice
            or type(table_slice.start) is not int
            or type(table_slice.stop) is not int
            or table_slice.start != expected_start
            or table_slice.stop < expected_start
            or table_slice.step not in (None, 1)
        ):
            raise ValueError(f"{table_name} table slice is not contiguous")
        width = table_slice.stop - table_slice.start
        table = batch.tables[table_name]
        if type(table) is not TokenTableBatch:
            raise ValueError(f"tables[{table_name}] must be TokenTableBatch")

        numeric = require_tensor(table.numeric, f"{table_name}.numeric")
        require_exact(
            numeric,
            torch.float32,
            (batch_size, width, len(TABLE_NUMERIC_FIELDS[table_name])),
            f"{table_name}.numeric",
            device,
        )
        if not bool(torch.isfinite(numeric).all()):
            raise ValueError(f"{table_name}.numeric must be finite")

        mask = require_tensor(table.mask, f"{table_name}.mask")
        require_exact(
            mask, torch.bool, (batch_size, width), f"{table_name}.mask", device
        )
        expected_mask = (
            torch.arange(width, device=device).unsqueeze(0)
            < mask.sum(dim=1, keepdim=True)
        )
        if not torch.equal(mask, expected_mask):
            raise ValueError(
                f"{table_name}.mask must select one contiguous row prefix"
            )

        if (
            type(table.categorical) is not MappingProxyType
            or tuple(table.categorical)
            != TABLE_CATEGORICAL_FIELDS[table_name]
        ):
            raise ValueError(f"{table_name}.categorical keys/order are invalid")
        for field in TABLE_CATEGORICAL_FIELDS[table_name]:
            name = f"{table_name}.categorical.{field}"
            value = require_tensor(table.categorical[field], name)
            require_exact(value, torch.int64, (batch_size, width), name, device)
            active = value[mask]
            cardinality = TABLE_CATEGORICAL_CARDINALITIES[table_name][field]
            if active.numel() and not bool(
                ((active >= 0) & (active < cardinality)).all()
            ):
                raise ValueError(f"{name} active values are out of range")

        if (
            type(table.boolean) is not MappingProxyType
            or tuple(table.boolean) != TABLE_BOOLEAN_FIELDS[table_name]
        ):
            raise ValueError(f"{table_name}.boolean keys/order are invalid")
        for field in TABLE_BOOLEAN_FIELDS[table_name]:
            name = f"{table_name}.boolean.{field}"
            value = require_tensor(table.boolean[field], name)
            require_exact(value, torch.bool, (batch_size, width), name, device)

        table_masks.append(mask)
        expected_start = table_slice.stop

    if expected_start != node_count:
        raise ValueError("table slices do not cover the global node count")
    expected_nodes = torch.cat(table_masks, dim=1)
    if not torch.equal(expected_nodes, node_mask):
        raise ValueError("node_mask disagrees with table masks and slices")

    cells = batch.table_slices["cells"]
    cell_count = cells.stop - cells.start
    neighbor_index = require_tensor(
        batch.cell_neighbor_index, "cell_neighbor_index"
    )
    neighbor_mask = require_tensor(batch.cell_neighbor_mask, "cell_neighbor_mask")
    require_exact(
        neighbor_index,
        torch.int64,
        (batch_size, cell_count, 6),
        "cell_neighbor_index",
        device,
    )
    require_exact(
        neighbor_mask,
        torch.bool,
        (batch_size, cell_count, 6),
        "cell_neighbor_mask",
        device,
    )
    expected_neighbor_mask = (
        torch.arange(6, device=device).view(1, 1, 6)
        < neighbor_mask.sum(dim=2, keepdim=True)
    )
    if not torch.equal(neighbor_mask, expected_neighbor_mask):
        raise ValueError(
            "cell_neighbor_mask must select one contiguous source prefix"
        )
    cell_mask = batch.tables["cells"].mask
    if bool((neighbor_mask & ~cell_mask.unsqueeze(-1)).any()):
        raise ValueError("padded cell destinations cannot have neighbors")
    destinations = (
        torch.arange(cell_count, device=device).view(1, cell_count, 1)
        + cells.start
    )
    if bool((neighbor_mask & (neighbor_index == destinations)).any()):
        raise ValueError("cell neighbor references cannot be self-neighbors")
    active_neighbors = neighbor_index[neighbor_mask]
    if active_neighbors.numel():
        if not bool(
            ((active_neighbors >= cells.start) & (active_neighbors < cells.stop)).all()
        ):
            raise ValueError(
                "cell neighbor references are outside the cells table slice"
            )
        samples = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1, 1)
            .expand_as(neighbor_index)[neighbor_mask]
        )
        if not bool(node_mask[samples, active_neighbors].all()):
            raise ValueError(
                "cell neighbor references do not select valid nodes"
            )
    for sample in range(batch_size):
        for destination in range(cell_count):
            values = neighbor_index[
                sample, destination, neighbor_mask[sample, destination]
            ]
            if values.numel() > 1 and not bool((values[1:] > values[:-1]).all()):
                raise ValueError(
                    "cell neighbor references must be unique and sorted"
                )

    neighborhoods = batch.neighborhoods
    if type(neighborhoods) is not RelationNeighborhoodBatch:
        raise ValueError(
            "batch.neighborhoods must be RelationNeighborhoodBatch"
        )
    source_index = require_tensor(
        neighborhoods.source_index, "neighborhoods.source_index"
    )
    if (
        source_index.ndim != 3
        or source_index.shape[:2] != (batch_size, node_count)
        or source_index.shape[2] <= 0
    ):
        raise ValueError("neighborhoods tensor shapes are invalid")
    slots = source_index.shape[2]
    neighborhood_contracts = (
        (
            neighborhoods.source_index,
            torch.int64,
            "neighborhoods.source_index",
        ),
        (neighborhoods.kind, torch.int64, "neighborhoods.kind"),
        (
            neighborhoods.int_feature,
            torch.int64,
            "neighborhoods.int_feature",
        ),
        (
            neighborhoods.float_feature,
            torch.float32,
            "neighborhoods.float_feature",
        ),
        (
            neighborhoods.bool_feature,
            torch.bool,
            "neighborhoods.bool_feature",
        ),
        (neighborhoods.mask, torch.bool, "neighborhoods.mask"),
    )
    shape = (batch_size, node_count, slots)
    for value, dtype, name in neighborhood_contracts:
        require_exact(require_tensor(value, name), dtype, shape, name, device)
    if not bool(torch.isfinite(neighborhoods.float_feature).all()):
        raise ValueError("neighborhoods.float_feature must be finite")

    edge_mask = neighborhoods.mask
    expected_edge_mask = (
        torch.arange(slots, device=device).view(1, 1, slots)
        < edge_mask.sum(dim=2, keepdim=True)
    )
    if not torch.equal(edge_mask, expected_edge_mask):
        raise ValueError(
            "neighborhoods.mask must select one contiguous incoming prefix"
        )
    active_sources = neighborhoods.source_index[edge_mask]
    if active_sources.numel():
        if not bool(
            ((active_sources >= 0) & (active_sources < node_count)).all()
        ):
            raise ValueError("neighborhood source references are out of range")
        samples = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1, 1)
            .expand_as(neighborhoods.source_index)[edge_mask]
        )
        if not bool(node_mask[samples, active_sources].all()):
            raise ValueError(
                "neighborhood source references do not select valid nodes"
            )
    active_kinds = neighborhoods.kind[edge_mask]
    if active_kinds.numel() and not bool(
        ((active_kinds >= 0) & (active_kinds < RELATION_KIND_COUNT)).all()
    ):
        raise ValueError("neighborhood relation kinds are out of range")
    active_integers = neighborhoods.int_feature[edge_mask]
    if active_integers.numel() and not bool(
        ((active_integers >= -(2**31)) & (active_integers < 2**31)).all()
    ):
        raise ValueError(
            "neighborhood int_feature active values are out of int32 range"
        )
    destination_mask = edge_mask.any(dim=2)
    if bool((destination_mask & ~node_mask).any()):
        raise ValueError(
            "neighborhood destinations do not select valid nodes"
        )

    candidates = batch.candidates
    if type(candidates) is not CandidateBatch:
        raise ValueError("batch.candidates must be CandidateBatch")
    candidate_mask = require_tensor(candidates.mask, "candidates.mask")
    if (
        candidate_mask.ndim != 2
        or candidate_mask.shape[0] != batch_size
        or candidate_mask.shape[1] <= 0
    ):
        raise ValueError("candidates.mask shape must be [B, C]")
    require_exact(
        candidate_mask,
        torch.bool,
        tuple(candidate_mask.shape),
        "candidates.mask",
        device,
    )
    candidate_count = candidate_mask.shape[1]
    candidate_id = require_tensor(candidates.candidate_id, "candidate_id")
    if (
        candidate_id.ndim == 2
        and candidate_id.shape[0] == batch_size
        and candidate_id.shape[1] != candidate_count
    ):
        peer_shapes = (
            getattr(candidates.decision_id, "shape", None),
            getattr(candidates.kind, "shape", None),
            getattr(candidates.reference_index, "shape", None),
            getattr(candidates.reference_mask, "shape", None),
            getattr(candidates.projection_integer, "shape", None),
            getattr(candidates.projection_boolean, "shape", None),
        )
        width = candidate_id.shape[1]
        expected_peers = (
            (batch_size, width),
            (batch_size, width),
            (batch_size, width, 8),
            (batch_size, width, 8),
            (batch_size, width, 7),
            (batch_size, width, 2),
        )
        if peer_shapes == expected_peers:
            raise ValueError(
                "candidate mask shape must agree with candidate fields"
            )
    candidate_contracts = (
        (candidate_id, torch.int64, (batch_size, candidate_count), "candidate_id"),
        (
            candidates.decision_id,
            torch.int64,
            (batch_size, candidate_count),
            "decision_id",
        ),
        (
            candidates.kind,
            torch.int64,
            (batch_size, candidate_count),
            "kind",
        ),
        (
            candidates.reference_index,
            torch.int64,
            (batch_size, candidate_count, 8),
            "reference_index",
        ),
        (
            candidates.reference_mask,
            torch.bool,
            (batch_size, candidate_count, 8),
            "reference_mask",
        ),
        (
            candidates.projection_integer,
            torch.int64,
            (batch_size, candidate_count, 7),
            "projection_integer",
        ),
        (
            candidates.projection_boolean,
            torch.bool,
            (batch_size, candidate_count, 2),
            "projection_boolean",
        ),
    )
    for value, dtype, shape, name in candidate_contracts:
        require_exact(require_tensor(value, name), dtype, shape, name, device)

    if not bool(candidate_mask.any(dim=1).all()):
        raise ValueError("each candidate sample must contain a valid candidate")
    expected_candidate_mask = (
        torch.arange(candidate_count, device=device).unsqueeze(0)
        < candidate_mask.sum(dim=1, keepdim=True)
    )
    if not torch.equal(candidate_mask, expected_candidate_mask):
        raise ValueError(
            "candidate mask must select one contiguous candidate prefix"
        )

    active_ids = candidates.candidate_id[candidate_mask]
    if active_ids.numel() and not bool(
        ((active_ids >= -(2**31)) & (active_ids < 2**31)).all()
    ):
        raise ValueError("candidate_id active values are out of int32 range")
    for sample in range(batch_size):
        valid = candidate_mask[sample]
        ids = candidates.candidate_id[sample, valid]
        expected_ids = torch.arange(
            ids.numel(), dtype=torch.int64, device=device
        )
        if not torch.equal(ids, expected_ids):
            raise ValueError(
                "candidate_id active values must be canonical dense row indices"
            )
        decisions = candidates.decision_id[sample, valid]
        if not bool((decisions == decisions[0]).all()):
            raise ValueError(
                "candidate decision_id values must agree within a sample"
            )

    active_kinds = candidates.kind[candidate_mask]
    if active_kinds.numel() and not bool(
        ((active_kinds >= 0) & (active_kinds < len(_CANDIDATE_KINDS))).all()
    ):
        raise ValueError("candidate kinds are out of range")
    active_integers = candidates.projection_integer[
        candidate_mask.unsqueeze(-1).expand_as(candidates.projection_integer)
    ]
    if active_integers.numel() and not bool(
        ((active_integers >= -(2**31)) & (active_integers < 2**31)).all()
    ):
        raise ValueError(
            "projection_integer active values are out of int32 range"
        )

    active_reference_mask = (
        candidates.reference_mask & candidate_mask.unsqueeze(-1)
    )
    active_references = candidates.reference_index[active_reference_mask]
    if active_references.numel():
        if not bool(
            ((active_references >= 0) & (active_references < node_count)).all()
        ):
            raise ValueError(
                "reference_index active values are out of range"
            )
        samples = (
            torch.arange(batch_size, device=device)
            .view(batch_size, 1, 1)
            .expand_as(candidates.reference_index)[active_reference_mask]
        )
        if not bool(node_mask[samples, active_references].all()):
            raise ValueError(
                "candidate references do not select valid nodes"
            )

    for sample, row in candidate_mask.nonzero(as_tuple=False).tolist():
        kind_name = _CANDIDATE_KINDS[
            int(candidates.kind[sample, row].item())
        ]
        families = (
            *CANDIDATE_REFERENCE_FAMILIES[kind_name].values(),
            *PROJECTED_REFERENCE_FAMILIES[kind_name].values(),
        )
        expected_reference_mask = tuple(
            family is not None for family in families
        )
        actual_reference_mask = tuple(
            bool(value)
            for value in candidates.reference_mask[sample, row].tolist()
        )
        if actual_reference_mask != expected_reference_mask:
            raise ValueError(
                "candidate reference mask does not match candidate kind"
            )
        for slot, family in enumerate(families):
            if family is None:
                continue
            reference = int(
                candidates.reference_index[sample, row, slot].item()
            )
            family_slice = batch.table_slices[family]
            if not family_slice.start <= reference < family_slice.stop:
                raise ValueError(
                    "candidate reference does not select its required table family"
                )

    horizon_targets = require_tensor(batch.horizon_targets, "horizon_targets")
    horizon_mask = require_tensor(
        batch.horizon_target_mask, "horizon_target_mask"
    )
    if (
        horizon_targets.ndim != 2
        or horizon_targets.shape[0] != batch_size
        or horizon_targets.shape[1] <= 0
    ):
        raise ValueError("horizon_targets shape must be [B, H]")
    if horizon_mask.ndim != 2 or horizon_mask.shape[0] != batch_size:
        raise ValueError("horizon_target_mask shape must be [B, H]")
    if horizon_targets.shape[1] != horizon_mask.shape[1]:
        if horizon_mask.shape[1] < horizon_targets.shape[1]:
            raise ValueError(
                "horizon_target_mask shape must agree with horizon_targets"
            )
        raise ValueError(
            "horizon_targets shape must agree with horizon_target_mask"
        )
    horizon_count = horizon_targets.shape[1]
    target_contracts = (
        (
            batch.teacher_candidate_index,
            torch.int64,
            (batch_size,),
            "teacher_candidate_index",
        ),
        (
            batch.terminal_outcome,
            torch.int64,
            (batch_size,),
            "terminal_outcome",
        ),
        (
            horizon_targets,
            torch.float32,
            (batch_size, horizon_count),
            "horizon_targets",
        ),
        (
            horizon_mask,
            torch.bool,
            (batch_size, horizon_count),
            "horizon_target_mask",
        ),
        (
            batch.remaining_turns,
            torch.float32,
            (batch_size,),
            "batch.remaining_turns",
        ),
        (
            batch.remaining_turns_mask,
            torch.bool,
            (batch_size,),
            "remaining_turns_mask",
        ),
    )
    for value, dtype, shape, name in target_contracts:
        require_exact(require_tensor(value, name), dtype, shape, name, device)

    if not bool(torch.isfinite(horizon_targets).all()):
        raise ValueError("horizon_targets must be finite")
    if not bool(torch.isfinite(batch.remaining_turns).all()):
        raise ValueError("batch.remaining_turns must be finite")

    teacher = batch.teacher_candidate_index
    outcome = batch.terminal_outcome
    teacher_free = teacher == -1
    outcome_free = outcome == -1
    if bool(teacher_free.any()) or bool(outcome_free.any()):
        if not bool(teacher_free.all()) or not bool(outcome_free.all()):
            raise ValueError(
                "target-free sentinels require teacher_candidate_index=-1 "
                "and terminal_outcome=-1 for every sample"
            )
        if (
            bool(horizon_targets.any())
            or bool(horizon_mask.any())
            or bool(batch.remaining_turns.any())
            or bool(batch.remaining_turns_mask.any())
        ):
            raise ValueError(
                "target-free auxiliary targets must be zero with all masks false"
            )
    else:
        if bool(((horizon_targets != 0) & (horizon_targets != 1)).any()):
            raise ValueError("horizon_targets must be binary")
        if bool(
            (batch.remaining_turns_mask & (batch.remaining_turns <= 0)).any()
        ):
            raise ValueError("remaining_turns targets must be positive")
        if bool(
            ((teacher < 0) | (teacher >= candidate_count)).any()
        ):
            raise ValueError(
                "teacher_candidate_index values are out of range"
            )
        samples = torch.arange(batch_size, device=device)
        if not bool(candidate_mask[samples, teacher].all()):
            raise ValueError(
                "teacher_candidate_index selects a padded candidate"
            )
        if bool(((outcome < 0) | (outcome > 2)).any()):
            raise ValueError(
                "terminal_outcome targets must be in 0..2"
            )
