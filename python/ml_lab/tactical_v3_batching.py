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
