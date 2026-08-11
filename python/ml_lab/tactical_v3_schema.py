"""Strict, immutable parser for the tactical-v3 structured wire protocol."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Literal, Mapping

import numpy as np


TableName = Literal[
    "cells", "units", "templates", "capability_definitions", "capability_allocations",
    "rules", "memory_records", "relations", "candidates",
]
RelativeOwner = Literal["self", "opponent"]
TerrainTypeName = Literal["plains", "forest", "rough", "water"]
RelationKind = Literal["neighbor", "occupies", "has_capability"]
CandidateKind = Literal["attack", "move", "deploy", "end_turn"]
CapabilityKind = Literal[
    "health", "damage", "defense", "movement", "vertical_movement", "range", "range_arc",
    "vision", "vision_arc",
]
RuleKind = Literal[
    "win_conditions", "round", "round_cap", "actions_per_turn", "starting_points",
    "self_points", "opponent_points", "damage_floor", "damage_high_ground_bonus",
    "range_high_ground_bonus", "bounty_rate", "deploy_cost_multiplier", "fog_of_war",
    "max_design_point_cost", "design_fee",
]

_TABLES = frozenset(TableName.__args__)
_OWNERS = frozenset(RelativeOwner.__args__)
_TERRAINS = frozenset(TerrainTypeName.__args__)
_RELATIONS = frozenset(RelationKind.__args__)
_CANDIDATES = frozenset(CandidateKind.__args__)
_CAPABILITIES = frozenset(CapabilityKind.__args__)
_RULES = frozenset(RuleKind.__args__)
_CAPACITY_FIELDS = frozenset({
    "max_cells", "max_units", "max_templates", "max_capability_definitions",
    "max_capability_allocations", "max_rules", "max_memory_records", "max_relations",
    "max_candidates",
})
_SPACE_FIELDS = frozenset({
    "scenario_id", "scenario_schema_version", "contract_version", "contract_hash",
    "encoding_hash", "capacity_hash", "environment_kind", "match", "encoding", "capacity",
})
_VIEW_FIELDS = frozenset({
    "decision_id", "seat", "observation", "candidates", "reward", "winner", "terminated",
    "truncated", "start_profile", "reference_seat",
})

CANDIDATE_REFERENCE_FAMILIES: Mapping[str, Mapping[str, str | None]] = MappingProxyType({
    "attack": MappingProxyType({"actor": "units", "target": "units", "template": None, "cell": None}),
    "move": MappingProxyType({"actor": "units", "target": None, "template": None, "cell": "cells"}),
    "deploy": MappingProxyType({"actor": None, "target": None, "template": "templates", "cell": "cells"}),
    "end_turn": MappingProxyType({"actor": None, "target": None, "template": None, "cell": None}),
})
PROJECTED_REFERENCE_FAMILIES: Mapping[str, Mapping[str, str | None]] = MappingProxyType({
    "attack": MappingProxyType({"source_cell": "cells", "destination_cell": None, "template": None, "target": "units"}),
    "move": MappingProxyType({"source_cell": "cells", "destination_cell": "cells", "template": None, "target": None}),
    "deploy": MappingProxyType({"source_cell": None, "destination_cell": "cells", "template": "templates", "target": None}),
    "end_turn": MappingProxyType({"source_cell": None, "destination_cell": None, "template": None, "target": None}),
})


@dataclass(frozen=True, slots=True)
class TokenRef:
    table: TableName
    row: int


@dataclass(frozen=True, slots=True)
class CellToken:
    q: int; r: int; terrain: TerrainTypeName; elevation: int
    self_deployment_zone: bool; opponent_deployment_zone: bool; controller: RelativeOwner | None
    is_boundary: bool; currently_visible: bool; previously_observed: bool


@dataclass(frozen=True, slots=True)
class UnitToken:
    owner: RelativeOwner; current_hp: int; max_hp: int; cell: TokenRef; elevation: int
    moved: bool; attacked: bool; horizontal_movement_spent: int; vertical_movement_spent: int
    point_cost: int; deploy_cost: int; currently_visible: bool


@dataclass(frozen=True, slots=True)
class TemplateToken:
    owner: RelativeOwner; point_cost: int; deploy_cost: int; is_fixed: bool; is_deployable: bool


@dataclass(frozen=True, slots=True)
class CapabilityDefinitionToken:
    kind: CapabilityKind


@dataclass(frozen=True, slots=True)
class CapabilityAllocationToken:
    owner: TokenRef; definition: TokenRef; capability: CapabilityKind; purchased_level: int; effective_value: int


@dataclass(frozen=True, slots=True)
class RuleToken:
    kind: RuleKind; int_value: int; float_value: float; bool_value: bool


@dataclass(frozen=True, slots=True)
class MemoryToken:
    cell: TokenRef; last_seen_round: int; observation_age: int; last_known_current_hp: int; currently_visible: bool


@dataclass(frozen=True, slots=True)
class RelationToken:
    kind: RelationKind; source: TokenRef; target: TokenRef; int_feature: int; float_feature: float; bool_feature: bool


@dataclass(frozen=True, slots=True)
class ProjectedDelta:
    source_cell: TokenRef | None; destination_cell: TokenRef | None; template: TokenRef | None; target: TokenRef | None
    horizontal_movement_spent: int; vertical_movement_spent: int; target_hp_delta: int; damage: int
    is_lethal: bool; bounty_delta: int; points_delta: int; round_delta: int; is_terminal: bool


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: int; decision_id: int; kind: CandidateKind; actor: TokenRef | None
    target: TokenRef | None; template: TokenRef | None; cell: TokenRef | None; projection: ProjectedDelta


@dataclass(frozen=True, slots=True)
class TacticalV3Observation:
    cells: tuple[CellToken, ...]; units: tuple[UnitToken, ...]; templates: tuple[TemplateToken, ...]
    capability_definitions: tuple[CapabilityDefinitionToken, ...]
    capability_allocations: tuple[CapabilityAllocationToken, ...]; rules: tuple[RuleToken, ...]
    memory: tuple[MemoryToken, ...]; relations: tuple[RelationToken, ...]


@dataclass(frozen=True, slots=True)
class TacticalV3Decision:
    decision_id: int; seat: int; observation: TacticalV3Observation; candidates: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class TacticalV3Reward:
    terminal_outcome: float; known_health_adjusted_material_progress: float
    public_resource_progress: float; time_pressure: float; total: float; finalized: bool


@dataclass(frozen=True, slots=True)
class TacticalV3View:
    decision: TacticalV3Decision; reward: TacticalV3Reward; winner: int; terminated: bool
    truncated: bool; start_profile: str; reference_seat: int

    @property
    def seat(self) -> int:
        return self.decision.seat


@dataclass(frozen=True, slots=True)
class TacticalV3SemanticIdentity:
    scenario_id: str; scenario_schema_version: int; contract_version: Literal["tactical-v3"]
    contract_hash: str; encoding_hash: str; capacity_hash: str; environment_kind: Literal["tactical", "duel"]
    match: Mapping[str, object]; encoding: Mapping[str, object]; capacity: Mapping[str, int]


def _exact_mapping(value: object, fields: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != fields or not all(type(key) is str for key in value):
        raise ValueError(f"{field} fields must be exactly {sorted(fields)}")
    return value


def _list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{field} must be a list")
    return value


def _int32(value: object, field: str) -> int:
    if type(value) is not int or not -(2**31) <= value < 2**31:
        raise TypeError(f"{field} must be an int32")
    return value


def _float32(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a float32 number")
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _bool(value: object, field: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field} must be a bool")
    return value


def _string(value: object, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a string")
    return value


def _literal(value: object, values: frozenset[str], field: str) -> str:
    result = _string(value, field)
    if result not in values:
        raise ValueError(f"{field} has an unknown value {result!r}")
    return result


def _hash(value: object, field: str) -> str:
    result = _string(value, field)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return result


def _freeze(value: object, field: str) -> object:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"{field} must be finite")
        return value
    if type(value) in (list, tuple):
        return tuple(_freeze(item, f"{field}[]") for item in value)
    if isinstance(value, Mapping) and all(type(key) is str for key in value):
        return MappingProxyType({key: _freeze(item, f"{field}.{key}") for key, item in value.items()})
    raise TypeError(f"{field} contains an unsupported canonical value")


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def canonical_sha256(value: object) -> str:
    frozen = _freeze(value, "canonical value")
    encoded = json.dumps(_thaw(frozen), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_spaces(payload: object) -> TacticalV3SemanticIdentity:
    data = _exact_mapping(payload, _SPACE_FIELDS, "spaces")
    match = _freeze(data["match"], "match")
    encoding = _freeze(data["encoding"], "encoding")
    raw_capacity = _exact_mapping(data["capacity"], _CAPACITY_FIELDS, "capacity")
    capacity = MappingProxyType({key: _int32(raw_capacity[key], f"capacity.{key}") for key in _CAPACITY_FIELDS})
    for key, value in capacity.items():
        if value < 0:
            raise ValueError(f"capacity.{key} must be nonnegative")
    encoding_hash = _hash(data["encoding_hash"], "encoding_hash")
    capacity_hash = _hash(data["capacity_hash"], "capacity_hash")
    if canonical_sha256(encoding) != encoding_hash:
        raise ValueError("encoding_hash does not match encoding")
    if canonical_sha256(capacity) != capacity_hash:
        raise ValueError("capacity_hash does not match capacity")
    return TacticalV3SemanticIdentity(
        scenario_id=_string(data["scenario_id"], "scenario_id"),
        scenario_schema_version=_int32(data["scenario_schema_version"], "scenario_schema_version"),
        contract_version=_literal(data["contract_version"], frozenset({"tactical-v3"}), "contract_version"),
        contract_hash=_hash(data["contract_hash"], "contract_hash"),
        encoding_hash=encoding_hash, capacity_hash=capacity_hash,
        environment_kind=_literal(data["environment_kind"], frozenset({"tactical", "duel"}), "environment_kind"),
        match=match, encoding=encoding, capacity=capacity,
    )


def _ref(value: object, field: str) -> TokenRef:
    data = _exact_mapping(value, frozenset({"table", "row"}), field)
    return TokenRef(_literal(data["table"], _TABLES, f"{field}.table"), _int32(data["row"], f"{field}.row"))


def _nullable_ref(value: object, field: str) -> TokenRef | None:
    return None if value is None else _ref(value, field)


def _row(value: object, fields: frozenset[str], field: str) -> Mapping[str, object]:
    return _exact_mapping(value, fields, field)


def _observation(value: object) -> TacticalV3Observation:
    data = _row(value, frozenset({"cells", "units", "templates", "capability_definitions", "capability_allocations", "rules", "memory", "relations"}), "observation")
    cells = tuple(CellToken(
        _int32(row["q"], f"cells[{i}].q"), _int32(row["r"], f"cells[{i}].r"),
        _literal(row["terrain"], _TERRAINS, f"cells[{i}].terrain"), _int32(row["elevation"], f"cells[{i}].elevation"),
        _bool(row["self_deployment_zone"], f"cells[{i}].self_deployment_zone"), _bool(row["opponent_deployment_zone"], f"cells[{i}].opponent_deployment_zone"),
        None if row["controller"] is None else _literal(row["controller"], _OWNERS, f"cells[{i}].controller"),
        _bool(row["is_boundary"], f"cells[{i}].is_boundary"), _bool(row["currently_visible"], f"cells[{i}].currently_visible"), _bool(row["previously_observed"], f"cells[{i}].previously_observed"),
    ) for i, row in enumerate(_row(item, frozenset({"q", "r", "terrain", "elevation", "self_deployment_zone", "opponent_deployment_zone", "controller", "is_boundary", "currently_visible", "previously_observed"}), f"cells[{i}]") for i, item in enumerate(_list(data["cells"], "cells"))))
    units = tuple(UnitToken(
        _literal(row["owner"], _OWNERS, f"units[{i}].owner"), _int32(row["current_hp"], f"units[{i}].current_hp"), _int32(row["max_hp"], f"units[{i}].max_hp"), _ref(row["cell"], f"units[{i}].cell"), _int32(row["elevation"], f"units[{i}].elevation"),
        _bool(row["moved"], f"units[{i}].moved"), _bool(row["attacked"], f"units[{i}].attacked"), _int32(row["horizontal_movement_spent"], f"units[{i}].horizontal_movement_spent"), _int32(row["vertical_movement_spent"], f"units[{i}].vertical_movement_spent"), _int32(row["point_cost"], f"units[{i}].point_cost"), _int32(row["deploy_cost"], f"units[{i}].deploy_cost"), _bool(row["currently_visible"], f"units[{i}].currently_visible"),
    ) for i, row in enumerate(_row(item, frozenset({"owner", "current_hp", "max_hp", "cell", "elevation", "moved", "attacked", "horizontal_movement_spent", "vertical_movement_spent", "point_cost", "deploy_cost", "currently_visible"}), f"units[{i}]") for i, item in enumerate(_list(data["units"], "units"))))
    templates = tuple(TemplateToken(_literal(row["owner"], _OWNERS, f"templates[{i}].owner"), _int32(row["point_cost"], f"templates[{i}].point_cost"), _int32(row["deploy_cost"], f"templates[{i}].deploy_cost"), _bool(row["is_fixed"], f"templates[{i}].is_fixed"), _bool(row["is_deployable"], f"templates[{i}].is_deployable")) for i, row in enumerate(_row(item, frozenset({"owner", "point_cost", "deploy_cost", "is_fixed", "is_deployable"}), f"templates[{i}]") for i, item in enumerate(_list(data["templates"], "templates"))))
    definitions = tuple(CapabilityDefinitionToken(_literal(row["kind"], _CAPABILITIES, f"capability_definitions[{i}].kind")) for i, row in enumerate(_row(item, frozenset({"kind"}), f"capability_definitions[{i}]") for i, item in enumerate(_list(data["capability_definitions"], "capability_definitions"))))
    allocations = tuple(CapabilityAllocationToken(_ref(row["owner"], f"capability_allocations[{i}].owner"), _ref(row["definition"], f"capability_allocations[{i}].definition"), _literal(row["capability"], _CAPABILITIES, f"capability_allocations[{i}].capability"), _int32(row["purchased_level"], f"capability_allocations[{i}].purchased_level"), _int32(row["effective_value"], f"capability_allocations[{i}].effective_value")) for i, row in enumerate(_row(item, frozenset({"owner", "definition", "capability", "purchased_level", "effective_value"}), f"capability_allocations[{i}]") for i, item in enumerate(_list(data["capability_allocations"], "capability_allocations"))))
    rules = tuple(RuleToken(_literal(row["kind"], _RULES, f"rules[{i}].kind"), _int32(row["int_value"], f"rules[{i}].int_value"), _float32(row["float_value"], f"rules[{i}].float_value"), _bool(row["bool_value"], f"rules[{i}].bool_value")) for i, row in enumerate(_row(item, frozenset({"kind", "int_value", "float_value", "bool_value"}), f"rules[{i}]") for i, item in enumerate(_list(data["rules"], "rules"))))
    memory = tuple(MemoryToken(_ref(row["cell"], f"memory[{i}].cell"), _int32(row["last_seen_round"], f"memory[{i}].last_seen_round"), _int32(row["observation_age"], f"memory[{i}].observation_age"), _int32(row["last_known_current_hp"], f"memory[{i}].last_known_current_hp"), _bool(row["currently_visible"], f"memory[{i}].currently_visible")) for i, row in enumerate(_row(item, frozenset({"cell", "last_seen_round", "observation_age", "last_known_current_hp", "currently_visible"}), f"memory[{i}]") for i, item in enumerate(_list(data["memory"], "memory"))))
    relations = tuple(RelationToken(_literal(row["kind"], _RELATIONS, f"relations[{i}].kind"), _ref(row["source"], f"relations[{i}].source"), _ref(row["target"], f"relations[{i}].target"), _int32(row["int_feature"], f"relations[{i}].int_feature"), _float32(row["float_feature"], f"relations[{i}].float_feature"), _bool(row["bool_feature"], f"relations[{i}].bool_feature")) for i, row in enumerate(_row(item, frozenset({"kind", "source", "target", "int_feature", "float_feature", "bool_feature"}), f"relations[{i}]") for i, item in enumerate(_list(data["relations"], "relations"))))
    return TacticalV3Observation(cells, units, templates, definitions, allocations, rules, memory, relations)


def _projection(value: object, field: str) -> ProjectedDelta:
    data = _row(value, frozenset({"source_cell", "destination_cell", "template", "target", "horizontal_movement_spent", "vertical_movement_spent", "target_hp_delta", "damage", "is_lethal", "bounty_delta", "points_delta", "round_delta", "is_terminal"}), field)
    return ProjectedDelta(_nullable_ref(data["source_cell"], f"{field}.source_cell"), _nullable_ref(data["destination_cell"], f"{field}.destination_cell"), _nullable_ref(data["template"], f"{field}.template"), _nullable_ref(data["target"], f"{field}.target"), _int32(data["horizontal_movement_spent"], f"{field}.horizontal_movement_spent"), _int32(data["vertical_movement_spent"], f"{field}.vertical_movement_spent"), _int32(data["target_hp_delta"], f"{field}.target_hp_delta"), _int32(data["damage"], f"{field}.damage"), _bool(data["is_lethal"], f"{field}.is_lethal"), _int32(data["bounty_delta"], f"{field}.bounty_delta"), _int32(data["points_delta"], f"{field}.points_delta"), _int32(data["round_delta"], f"{field}.round_delta"), _bool(data["is_terminal"], f"{field}.is_terminal"))


def _reference_length(ref: TokenRef, observation: TacticalV3Observation, candidates: int) -> int:
    return {"cells": len(observation.cells), "units": len(observation.units), "templates": len(observation.templates), "capability_definitions": len(observation.capability_definitions), "capability_allocations": len(observation.capability_allocations), "rules": len(observation.rules), "memory_records": len(observation.memory), "relations": len(observation.relations), "candidates": candidates}[ref.table]


def _require_reference(ref: TokenRef | None, expected: str | tuple[str, ...] | None, observation: TacticalV3Observation, candidates: int, field: str) -> None:
    if expected is None:
        if ref is not None:
            raise ValueError(f"{field} must be null")
        return
    if ref is None:
        raise ValueError(f"{field} is required")
    accepted = (expected,) if type(expected) is str else expected
    if ref.table not in accepted:
        raise ValueError(f"{field} references incompatible table {ref.table}")
    length = _reference_length(ref, observation, candidates)
    if ref.row < 0 or ref.row >= length:
        raise ValueError(f"{ref.table}[{ref.row}] of {length}")


def _validate_semantics(decision: TacticalV3Decision, terminated: bool, truncated: bool, identity: TacticalV3SemanticIdentity) -> None:
    observation, candidates = decision.observation, decision.candidates
    counts = {"cells": len(observation.cells), "units": len(observation.units), "templates": len(observation.templates), "capability_definitions": len(observation.capability_definitions), "capability_allocations": len(observation.capability_allocations), "rules": len(observation.rules), "memory_records": len(observation.memory), "relations": len(observation.relations), "candidates": len(candidates)}
    for table, count in counts.items():
        limit = identity.capacity[f"max_{table}"]
        if count > limit:
            raise ValueError(f"capacity exceeded for {table}: {count} > {limit}")
    if terminated or truncated:
        if candidates:
            raise ValueError("terminal decision must have no candidates")
    elif not candidates:
        raise ValueError("nonterminal decision must have candidates")
    if [candidate.candidate_id for candidate in candidates] != list(range(len(candidates))):
        raise ValueError("candidate ids must be exactly 0..len(candidates)-1")
    for unit in observation.units:
        _require_reference(unit.cell, "cells", observation, len(candidates), "unit.cell")
    for allocation in observation.capability_allocations:
        _require_reference(allocation.owner, ("units", "templates"), observation, len(candidates), "capability_allocation.owner")
        _require_reference(allocation.definition, "capability_definitions", observation, len(candidates), "capability_allocation.definition")
        if observation.capability_definitions[allocation.definition.row].kind != allocation.capability:
            raise ValueError("capability_allocation.capability does not match its definition")
    for memory in observation.memory:
        _require_reference(memory.cell, "cells", observation, len(candidates), "memory.cell")
    relation_families = {"neighbor": ("cells", "cells"), "occupies": ("units", "cells"), "has_capability": (("units", "templates"), "capability_definitions")}
    for relation in observation.relations:
        source, target = relation_families[relation.kind]
        _require_reference(relation.source, source, observation, len(candidates), f"{relation.kind}.source")
        _require_reference(relation.target, target, observation, len(candidates), f"{relation.kind}.target")
    for candidate in candidates:
        if candidate.decision_id != decision.decision_id:
            raise ValueError("candidate decision_id does not match decision_id")
        for field, family in CANDIDATE_REFERENCE_FAMILIES[candidate.kind].items():
            _require_reference(getattr(candidate, field), family, observation, len(candidates), f"{candidate.kind}.{field}")
        for field, family in PROJECTED_REFERENCE_FAMILIES[candidate.kind].items():
            _require_reference(getattr(candidate.projection, field), family, observation, len(candidates), f"{candidate.kind}.projection.{field}")


def parse_decision(payload: object, identity: TacticalV3SemanticIdentity) -> TacticalV3Decision:
    data = _exact_mapping(payload, _VIEW_FIELDS, "view")
    observation = _observation(data["observation"])
    candidates = tuple(Candidate(
        _int32(row["candidate_id"], f"candidates[{i}].candidate_id"), _int32(row["decision_id"], f"candidates[{i}].decision_id"), _literal(row["kind"], _CANDIDATES, f"candidates[{i}].kind"), _nullable_ref(row["actor"], f"candidates[{i}].actor"), _nullable_ref(row["target"], f"candidates[{i}].target"), _nullable_ref(row["template"], f"candidates[{i}].template"), _nullable_ref(row["cell"], f"candidates[{i}].cell"), _projection(row["projection"], f"candidates[{i}].projection"),
    ) for i, row in enumerate(_row(item, frozenset({"candidate_id", "decision_id", "kind", "actor", "target", "template", "cell", "projection"}), f"candidates[{i}]") for i, item in enumerate(_list(data["candidates"], "candidates"))))
    decision = TacticalV3Decision(_int32(data["decision_id"], "decision_id"), _int32(data["seat"], "seat"), observation, candidates)
    _validate_semantics(decision, _bool(data["terminated"], "terminated"), _bool(data["truncated"], "truncated"), identity)
    return decision


def parse_view(payload: object, identity: TacticalV3SemanticIdentity) -> TacticalV3View:
    data = _exact_mapping(payload, _VIEW_FIELDS, "view")
    decision = parse_decision(data, identity)
    reward = _row(data["reward"], frozenset({"terminal_outcome", "known_health_adjusted_material_progress", "public_resource_progress", "time_pressure", "total", "finalized"}), "reward")
    return TacticalV3View(
        decision, TacticalV3Reward(_float32(reward["terminal_outcome"], "reward.terminal_outcome"), _float32(reward["known_health_adjusted_material_progress"], "reward.known_health_adjusted_material_progress"), _float32(reward["public_resource_progress"], "reward.public_resource_progress"), _float32(reward["time_pressure"], "reward.time_pressure"), _float32(reward["total"], "reward.total"), _bool(reward["finalized"], "reward.finalized")),
        _int32(data["winner"], "winner"), _bool(data["terminated"], "terminated"), _bool(data["truncated"], "truncated"), _string(data["start_profile"], "start_profile"), _int32(data["reference_seat"], "reference_seat"),
    )
