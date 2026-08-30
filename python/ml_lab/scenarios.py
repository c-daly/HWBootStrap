"""Strict schema-v1 scenario resolution and canonical snapshots."""

from __future__ import annotations

import copy
import json
import math
import struct
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .io import atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_LIBRARY = (
    PROJECT_ROOT / "python" / "config" / "training-game-templates.json"
)
SUPPORTED_ENVIRONMENTS = frozenset(
    {"tactical-v1", "tactical-v2", "tactical-v3", "adaptive-v1"}
)
_LIBRARY_KEYS = frozenset({"schema_version", "templates"})
_COMMON_SCENARIO_KEYS = frozenset(
    {"schema_version", "id", "name", "environment", "board", "rules", "episode", "reward"}
)
_BOARD_KEYS = frozenset(
    {
        "width",
        "height",
        "max_elevation",
        "zone_depth",
        "flat_chance",
        "plains_weight",
        "forest_weight",
        "rough_weight",
        "water_weight",
    }
)
_RULE_KEYS = frozenset(
    {
        "actions_per_turn",
        "round_cap",
        "starting_points",
        "fog_of_war",
        "biomes_enabled",
        "bounty_rate",
        "deploy_cost_multiplier",
        "generator_cost",
        "generator_output",
        "generator_health",
    }
)
_TACTICAL_REWARD_KEYS = frozenset(
    {
        "shape_scale",
        "step_penalty",
        "closing_weight",
        "draw_credit_weight",
        "points_weight",
    }
)
_ADAPTIVE_REWARD_KEYS = frozenset(
    {"intermediate_decision_penalty", "deployment_completion_bonus"}
)
_TACTICAL_V3_REWARD_KEYS = frozenset(
    {
        "terminal_win",
        "terminal_non_win",
        "material_adjustment_bound",
        "time_pressure_bound",
        "points_weight",
    }
)
_ADAPTIVE_KEYS = frozenset(
    {"starting_unit_count", "starting_army_budget", "max_design_point_cost"}
)
_CHEAPEST_ADAPTIVE_TEMPLATE_COST = 20
_TACTICAL_V2_KEYS = frozenset(
    {"starting_unit_count", "max_controllable_units", "placement_policy", "templates"}
)
_TACTICAL_V2_PROFILED_KEYS = _TACTICAL_V2_KEYS | frozenset(
    {"start_profiles", "start_distribution"}
)
_TACTICAL_V2_TEMPLATE_KEYS = frozenset({"id", "name", "stats"})
_TACTICAL_V2_START_PROFILE_KEYS = frozenset(
    {"id", "learner_units", "opponent_units", "separation"}
)
_TACTICAL_V2_START_WEIGHT_KEYS = frozenset({"profile_id", "basis_points"})
_TACTICAL_V2_START_PROFILES = (
    ("standard-3v3", 3, 3, "legacy-mirrored"),
    ("conversion-3v1-near", 3, 1, "near"),
    ("conversion-3v1-medium", 3, 1, "medium"),
    ("conversion-3v1-far", 3, 1, "far"),
    ("conversion-2v1-near", 2, 1, "near"),
    ("conversion-2v1-medium", 2, 1, "medium"),
    ("conversion-2v1-far", 2, 1, "far"),
    ("conversion-1v1-near", 1, 1, "near"),
    ("conversion-1v1-medium", 1, 1, "medium"),
    ("conversion-1v1-far", 1, 1, "far"),
)
_TACTICAL_V2_STAT_KEYS = frozenset(
    {
        "health",
        "damage",
        "defense",
        "movement",
        "vertical_movement",
        "range",
        "range_arc",
        "vision",
        "vision_arc",
    }
)
_TACTICAL_V3_KEYS = frozenset(
    {
        "starting_unit_count",
        "max_controllable_units",
        "placement_policy",
        "capacity",
        "templates",
        "start_profiles",
        "start_distribution",
    }
)
_TACTICAL_V3_CAPACITY_KEYS = frozenset(
    {
        "max_cells",
        "max_units",
        "max_templates",
        "max_capability_definitions",
        "max_capability_allocations",
        "max_rules",
        "max_memory_records",
        "max_relations",
        "max_candidates",
    }
)
_TACTICAL_V3_REWARD = {
    "terminal_win": 1.0,
    "terminal_non_win": -1.0,
    "material_adjustment_bound": 0.2,
    "time_pressure_bound": 0.05,
    "points_weight": 0.5,
}


@dataclass(frozen=True)
class ResolvedScenario:
    schema_version: int
    template_id: str
    name: str
    environment: str
    document: Mapping[str, Any]
    canonical_json: str
    # Non-fatal advisories (see tactical_v2_round_cap_warning) that don't invalidate the scenario —
    # contrast a Validate-style hard error. Empty for every environment except tactical-v2, and empty
    # there too once episode.max_steps covers the configured round cap.
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.canonical_json + "\n")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"invalid JSON number {value}; numbers must be finite")


def _read_document(path: Path) -> Any:
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error.msg}") from error


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], path: str) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    errors = []
    if missing:
        errors.append("missing " + ", ".join(f"{path}.{key}" for key in missing))
    if extra:
        errors.append("unexpected " + ", ".join(f"{path}.{key}" for key in extra))
    if errors:
        raise ValueError("; ".join(errors))


def _text(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{path} must be an integer")
    return value


def _number(value: Any, path: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a number")
    if not math.isfinite(value):
        raise ValueError(f"{path} must be finite")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _positive(value: int, path: str) -> None:
    if value <= 0:
        raise ValueError(f"{path} must be positive")


def validate_scenario_document(raw: Any) -> Mapping[str, Any]:
    document = _mapping(raw, "scenario")
    environment = _text(document.get("environment"), "environment")
    expected_keys = _COMMON_SCENARIO_KEYS | (
        frozenset({"adaptive"})
        if environment == "adaptive-v1"
        else frozenset({"tactical_v2"}) if environment == "tactical-v2"
        else frozenset({"tactical_v3"}) if environment == "tactical-v3"
        else frozenset()
    )
    _exact_keys(document, expected_keys, "scenario")

    schema_version = _integer(document["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    _text(document["id"], "id")
    _text(document["name"], "name")
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            "environment must be tactical-v1, tactical-v2, tactical-v3, or adaptive-v1"
        )

    board = _mapping(document["board"], "board")
    _exact_keys(board, _BOARD_KEYS, "board")
    width = _integer(board["width"], "board.width")
    height = _integer(board["height"], "board.height")
    max_elevation = _integer(board["max_elevation"], "board.max_elevation")
    zone_depth = _integer(board["zone_depth"], "board.zone_depth")
    _positive(width, "board.width")
    _positive(height, "board.height")
    _positive(max_elevation, "board.max_elevation")
    _positive(zone_depth, "board.zone_depth")
    if zone_depth > width - zone_depth:
        raise ValueError("board deployment zones overlap")
    flat_chance = _number(board["flat_chance"], "board.flat_chance")
    if not 0 <= flat_chance <= 1:
        raise ValueError("board.flat_chance must be within [0,1]")
    terrain_weights = []
    for key in ("plains_weight", "forest_weight", "rough_weight", "water_weight"):
        weight = _integer(board[key], f"board.{key}")
        if weight < 0:
            raise ValueError(f"board.{key} must be non-negative")
        terrain_weights.append(weight)
    if sum(terrain_weights) <= 0:
        raise ValueError("board terrain weight sum must be positive")

    rules = _mapping(document["rules"], "rules")
    _exact_keys(rules, _RULE_KEYS, "rules")
    actions_per_turn = _integer(rules["actions_per_turn"], "rules.actions_per_turn")
    if actions_per_turn < 0:
        raise ValueError("rules.actions_per_turn must be non-negative")
    round_cap = _integer(rules["round_cap"], "rules.round_cap")
    _positive(round_cap, "rules.round_cap")
    for key in (
        "starting_points",
        "generator_cost",
        "generator_output",
        "generator_health",
    ):
        _integer(rules[key], f"rules.{key}")
    for key in ("fog_of_war", "biomes_enabled"):
        _boolean(rules[key], f"rules.{key}")
    for key in ("bounty_rate", "deploy_cost_multiplier"):
        _number(rules[key], f"rules.{key}")

    episode = _mapping(document["episode"], "episode")
    _exact_keys(episode, frozenset({"max_steps"}), "episode")
    max_steps = _integer(episode["max_steps"], "episode.max_steps")
    _positive(max_steps, "episode.max_steps")

    reward = _mapping(document["reward"], "reward")
    reward_keys = (
        _TACTICAL_V3_REWARD_KEYS
        if environment == "tactical-v3"
        else _TACTICAL_REWARD_KEYS
        if environment in {"tactical-v1", "tactical-v2"}
        else _ADAPTIVE_REWARD_KEYS
    )
    _exact_keys(reward, reward_keys, "reward")
    for key in reward_keys:
        _number(reward[key], f"reward.{key}")

    if environment == "tactical-v3":
        _validate_tactical_v3(
            document,
            width=width,
            height=height,
            zone_depth=zone_depth,
            round_cap=round_cap,
            max_steps=max_steps,
        )

    if environment == "tactical-v2":
        tactical_v2 = _mapping(document["tactical_v2"], "tactical_v2")
        placement_policy = _text(
            tactical_v2.get("placement_policy"), "tactical_v2.placement_policy"
        )
        expected_tactical_v2_keys = (
            _TACTICAL_V2_PROFILED_KEYS
            if placement_policy == "profiled-seeded-v1"
            or "start_profiles" in tactical_v2
            or "start_distribution" in tactical_v2
            else _TACTICAL_V2_KEYS
        )
        _exact_keys(tactical_v2, expected_tactical_v2_keys, "tactical_v2")
        starting_units = _integer(
            tactical_v2["starting_unit_count"], "tactical_v2.starting_unit_count"
        )
        max_controllable = _integer(
            tactical_v2["max_controllable_units"], "tactical_v2.max_controllable_units"
        )
        if placement_policy == "symmetric-random-v1":
            if not 1 <= starting_units <= 12:
                raise ValueError(
                    "tactical_v2.starting_unit_count must be between 1 and 12"
                )
            if max_controllable != starting_units:
                raise ValueError(
                    "tactical_v2.max_controllable_units must equal starting_unit_count"
                )
            if tactical_v2.get("start_profiles"):
                raise ValueError(
                    "tactical_v2.start_profiles is not valid for symmetric-random-v1"
                )
            if tactical_v2.get("start_distribution"):
                raise ValueError(
                    "tactical_v2.start_distribution is not valid for symmetric-random-v1"
                )
        elif placement_policy == "profiled-seeded-v1":
            if starting_units != 3 or max_controllable != 3:
                raise ValueError(
                    "profiled-seeded-v1 requires tactical_v2.starting_unit_count and "
                    "max_controllable_units to equal 3"
                )
            _validate_tactical_v2_start_profiles(tactical_v2, max_controllable)
        else:
            raise ValueError(
                "tactical_v2.placement_policy must be 'symmetric-random-v1' or "
                "'profiled-seeded-v1'"
            )

        raw_templates = tactical_v2["templates"]
        if not isinstance(raw_templates, list) or not raw_templates:
            raise ValueError("tactical_v2.templates must be a non-empty array")
        seen_template_ids: set[str] = set()
        for index, raw_template in enumerate(raw_templates):
            path = f"tactical_v2.templates[{index}]"
            template = _mapping(raw_template, path)
            _exact_keys(template, _TACTICAL_V2_TEMPLATE_KEYS, path)
            template_id = _text(template["id"], f"{path}.id")
            if template_id in seen_template_ids:
                raise ValueError(f"duplicate tactical_v2 template id {template_id!r}")
            seen_template_ids.add(template_id)
            _text(template["name"], f"{path}.name")
            stats = _mapping(template["stats"], f"{path}.stats")
            _exact_keys(stats, _TACTICAL_V2_STAT_KEYS, f"{path}.stats")
            for key in _TACTICAL_V2_STAT_KEYS:
                value = _integer(stats[key], f"{path}.stats.{key}")
                if value < 0:
                    raise ValueError(f"{path}.stats.{key} must be non-negative")

        if height * zone_depth < starting_units:
            raise ValueError(
                "tactical-v2 deployment cells must cover starting_unit_count"
            )

    if environment == "adaptive-v1":
        adaptive = _mapping(document["adaptive"], "adaptive")
        _exact_keys(adaptive, _ADAPTIVE_KEYS, "adaptive")
        starting_units = _integer(
            adaptive["starting_unit_count"], "adaptive.starting_unit_count"
        )
        if not 1 <= starting_units <= 24:
            raise ValueError(
                "adaptive.starting_unit_count must be between 1 and 24"
            )
        if height * zone_depth < starting_units:
            raise ValueError(
                "adaptive deployment cells must cover starting_unit_count"
            )
        starting_budget = _integer(
            adaptive["starting_army_budget"], "adaptive.starting_army_budget"
        )
        if starting_budget < _CHEAPEST_ADAPTIVE_TEMPLATE_COST * starting_units:
            raise ValueError(
                "adaptive.starting_army_budget is insufficient for starting_unit_count"
            )
        max_design = _integer(
            adaptive["max_design_point_cost"], "adaptive.max_design_point_cost"
        )
        _positive(max_design, "adaptive.max_design_point_cost")

    return document


def _validate_tactical_v3(
    document: Mapping[str, Any],
    *,
    width: int,
    height: int,
    zone_depth: int,
    round_cap: int,
    max_steps: int,
) -> None:
    section = _mapping(document["tactical_v3"], "tactical_v3")
    _exact_keys(section, _TACTICAL_V3_KEYS, "tactical_v3")
    starting_units = _integer(
        section["starting_unit_count"], "tactical_v3.starting_unit_count"
    )
    max_controllable = _integer(
        section["max_controllable_units"],
        "tactical_v3.max_controllable_units",
    )
    placement = _text(
        section["placement_policy"], "tactical_v3.placement_policy"
    )
    if placement == "profiled-seeded-v1":
        if starting_units != 3 or max_controllable != 3:
            raise ValueError(
                "profiled-seeded-v1 requires tactical_v3.starting_unit_count and "
                "max_controllable_units to equal 3"
            )
        _validate_tactical_v3_start_profiles(section, max_controllable)
    elif placement == "symmetric-random-v1":
        if not 1 <= starting_units <= 12:
            raise ValueError(
                "tactical_v3.starting_unit_count must be between 1 and 12"
            )
        if max_controllable != starting_units:
            raise ValueError(
                "tactical_v3.max_controllable_units must equal starting_unit_count"
            )
        if section["start_profiles"] or section["start_distribution"]:
            raise ValueError(
                "tactical_v3 symmetric-random-v1 must not declare start profiles "
                "or a start distribution"
            )
    else:
        raise ValueError(
            "tactical_v3.placement_policy must be 'symmetric-random-v1' or "
            "'profiled-seeded-v1'"
        )

    if document["rules"]["fog_of_war"] or document["rules"]["biomes_enabled"]:
        raise ValueError("tactical-v3 requires fog and biomes disabled")
    for key, expected in _TACTICAL_V3_REWARD.items():
        if float(document["reward"][key]) != expected:
            raise ValueError(
                f"reward.{key} must equal the tactical-v3 stage-one value {expected}"
            )
    minimum_steps = 2 * (starting_units + 1) * round_cap
    if max_steps < minimum_steps:
        raise ValueError(
            "tactical-v3 episode.max_steps is insufficient to reach the round cap; "
            f"minimum required is {minimum_steps}"
        )

    raw_templates = section["templates"]
    if not isinstance(raw_templates, list) or not raw_templates:
        raise ValueError("tactical_v3.templates must be a non-empty array")
    seen_template_ids: set[str] = set()
    for index, raw_template in enumerate(raw_templates):
        path = f"tactical_v3.templates[{index}]"
        template = _mapping(raw_template, path)
        _exact_keys(template, _TACTICAL_V2_TEMPLATE_KEYS, path)
        template_id = _text(template["id"], f"{path}.id")
        if template_id in seen_template_ids:
            raise ValueError(f"duplicate tactical-v3 template id {template_id!r}")
        seen_template_ids.add(template_id)
        _text(template["name"], f"{path}.name")
        stats = _mapping(template["stats"], f"{path}.stats")
        _exact_keys(stats, _TACTICAL_V2_STAT_KEYS, f"{path}.stats")
        for key in _TACTICAL_V2_STAT_KEYS:
            value = _integer(stats[key], f"{path}.stats.{key}")
            if value < 0 or (key == "health" and value == 0):
                qualifier = "positive" if key == "health" else "non-negative"
                raise ValueError(f"{path}.stats.{key} must be {qualifier}")

    capacity = _mapping(section["capacity"], "tactical_v3.capacity")
    _exact_keys(capacity, _TACTICAL_V3_CAPACITY_KEYS, "tactical_v3.capacity")
    values = {
        key: _integer(capacity[key], f"tactical_v3.capacity.{key}")
        for key in _TACTICAL_V3_CAPACITY_KEYS
    }
    for key, value in values.items():
        _positive(value, f"tactical_v3.capacity.{key}")

    profiles = section["start_profiles"] if placement == "profiled-seeded-v1" else []
    maximum_total_units = max(
        [2 * starting_units]
        + [
            profile["learner_units"] + profile["opponent_units"]
            for profile in profiles
        ]
    )
    cell_count = width * height
    template_rows = 2 * len(raw_templates)
    allocation_count = (values["max_units"] + template_rows) * 9
    directed_adjacency = 2 * (
        width * (height - 1) + (width - 1) * (2 * height - 1)
    )
    minimum_relations = (
        directed_adjacency + values["max_units"] + allocation_count
    )
    deployment_cells = height * zone_depth
    if deployment_cells < starting_units:
        raise ValueError(
            "tactical-v3 deployment cells must cover starting_unit_count"
        )
    candidate_requirement = (
        len(raw_templates) * deployment_cells
        + values["max_units"] * cell_count
        + values["max_units"] * (values["max_units"] - 1)
        + 1
    )
    requirements = {
        "max_cells": cell_count,
        "max_units": maximum_total_units,
        "max_templates": template_rows,
        "max_capability_definitions": 9,
        "max_capability_allocations": allocation_count,
        "max_rules": 15,
        "max_memory_records": maximum_total_units,
        "max_relations": minimum_relations,
        "max_candidates": candidate_requirement,
    }
    for key, minimum in requirements.items():
        if values[key] < minimum:
            raise ValueError(
                f"tactical_v3.capacity.{key} must be at least {minimum} for this scenario"
            )


def _validate_tactical_v3_start_profiles(
    section: Mapping[str, Any], max_controllable: int
) -> None:
    raw_profiles = section["start_profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("tactical_v3.start_profiles must be a non-empty array")
    actual_profiles: list[tuple[str, int, int, str]] = []
    seen_profile_ids: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        path = f"tactical_v3.start_profiles[{index}]"
        profile = _mapping(raw_profile, path)
        _exact_keys(profile, _TACTICAL_V2_START_PROFILE_KEYS, path)
        profile_id = _text(profile["id"], f"{path}.id")
        if profile_id in seen_profile_ids:
            raise ValueError(f"duplicate tactical-v3 start profile id {profile_id!r}")
        seen_profile_ids.add(profile_id)
        learner_units = _integer(profile["learner_units"], f"{path}.learner_units")
        opponent_units = _integer(
            profile["opponent_units"], f"{path}.opponent_units"
        )
        if not 1 <= learner_units <= max_controllable:
            raise ValueError(
                f"{path}.learner_units must be between 1 and max_controllable_units"
            )
        if not 1 <= opponent_units <= max_controllable:
            raise ValueError(
                f"{path}.opponent_units must be between 1 and max_controllable_units"
            )
        separation = _text(profile["separation"], f"{path}.separation")
        if separation not in {"legacy-mirrored", "near", "medium", "far"}:
            raise ValueError(f"{path}.separation is unknown: {separation!r}")
        actual_profiles.append(
            (profile_id, learner_units, opponent_units, separation)
        )
    if tuple(actual_profiles) != _TACTICAL_V2_START_PROFILES:
        raise ValueError(
            "profiled-seeded-v1 requires the exact versioned start profile catalog"
        )

    raw_distribution = section["start_distribution"]
    if not isinstance(raw_distribution, list) or not raw_distribution:
        raise ValueError("tactical_v3.start_distribution must be a non-empty array")
    declared = {profile[0] for profile in _TACTICAL_V2_START_PROFILES}
    seen_weights: set[str] = set()
    total_basis_points = 0
    for index, raw_weight in enumerate(raw_distribution):
        path = f"tactical_v3.start_distribution[{index}]"
        weight = _mapping(raw_weight, path)
        _exact_keys(weight, _TACTICAL_V2_START_WEIGHT_KEYS, path)
        profile_id = _text(weight["profile_id"], f"{path}.profile_id")
        if profile_id in seen_weights:
            raise ValueError(
                f"duplicate tactical-v3 start distribution weight for {profile_id!r}"
            )
        seen_weights.add(profile_id)
        if profile_id not in declared:
            raise ValueError(
                f"weight references undeclared start profile {profile_id!r}"
            )
        basis_points = _integer(weight["basis_points"], f"{path}.basis_points")
        if not 0 <= basis_points <= 10000:
            raise ValueError(f"{path}.basis_points must be within [0,10000]")
        total_basis_points += basis_points
    missing = sorted(declared - seen_weights)
    if missing:
        raise ValueError(
            "start distribution is missing declared profile " + repr(missing[0])
        )
    if total_basis_points != 10000:
        raise ValueError("start distribution weights must sum to 10000 basis points")


def _validate_tactical_v2_start_profiles(
    tactical_v2: Mapping[str, Any], max_controllable: int
) -> None:
    raw_profiles = tactical_v2["start_profiles"]
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise ValueError("tactical_v2.start_profiles must be a non-empty array")

    actual_profiles: list[tuple[str, int, int, str]] = []
    seen_profile_ids: set[str] = set()
    for index, raw_profile in enumerate(raw_profiles):
        path = f"tactical_v2.start_profiles[{index}]"
        profile = _mapping(raw_profile, path)
        _exact_keys(profile, _TACTICAL_V2_START_PROFILE_KEYS, path)
        profile_id = _text(profile["id"], f"{path}.id")
        if profile_id in seen_profile_ids:
            raise ValueError(f"duplicate tactical-v2 start profile id {profile_id!r}")
        seen_profile_ids.add(profile_id)
        learner_units = _integer(profile["learner_units"], f"{path}.learner_units")
        opponent_units = _integer(profile["opponent_units"], f"{path}.opponent_units")
        if not 1 <= learner_units <= max_controllable:
            raise ValueError(
                f"{path}.learner_units must be between 1 and max_controllable_units"
            )
        if not 1 <= opponent_units <= max_controllable:
            raise ValueError(
                f"{path}.opponent_units must be between 1 and max_controllable_units"
            )
        separation = _text(profile["separation"], f"{path}.separation")
        if separation not in {"legacy-mirrored", "near", "medium", "far"}:
            raise ValueError(f"{path}.separation is unknown: {separation!r}")
        actual_profiles.append(
            (profile_id, learner_units, opponent_units, separation)
        )

    if tuple(actual_profiles) != _TACTICAL_V2_START_PROFILES:
        raise ValueError(
            "profiled-seeded-v1 requires the exact versioned start profile catalog"
        )

    raw_distribution = tactical_v2["start_distribution"]
    if not isinstance(raw_distribution, list) or not raw_distribution:
        raise ValueError("tactical_v2.start_distribution must be a non-empty array")
    declared = {profile[0] for profile in _TACTICAL_V2_START_PROFILES}
    seen_weights: set[str] = set()
    total_basis_points = 0
    for index, raw_weight in enumerate(raw_distribution):
        path = f"tactical_v2.start_distribution[{index}]"
        weight = _mapping(raw_weight, path)
        _exact_keys(weight, _TACTICAL_V2_START_WEIGHT_KEYS, path)
        profile_id = _text(weight["profile_id"], f"{path}.profile_id")
        if profile_id in seen_weights:
            raise ValueError(
                f"duplicate tactical-v2 start distribution weight for {profile_id!r}"
            )
        seen_weights.add(profile_id)
        if profile_id not in declared:
            raise ValueError(
                f"weight references undeclared start profile {profile_id!r}"
            )
        basis_points = _integer(weight["basis_points"], f"{path}.basis_points")
        if not 0 <= basis_points <= 10000:
            raise ValueError(f"{path}.basis_points must be within [0,10000]")
        total_basis_points += basis_points

    missing = sorted(declared - seen_weights)
    if missing:
        raise ValueError(
            "start distribution is missing declared profile " + repr(missing[0])
        )
    if total_basis_points != 10000:
        raise ValueError("start distribution weights must sum to 10000 basis points")


def tactical_v2_round_cap_minimum(document: Mapping[str, Any]) -> int | None:
    """The fewest ``episode.max_steps`` that lets ``document``'s tactical-v2 army play out every
    round up to its configured round cap, or ``None`` for a non-tactical-v2 document.

    Mirrors the engine's ``TacticalV2Config.MinimumMaxSteps`` (see
    ``engine/HexWars.Engine/Rl/TacticalV2Config.cs``): MaxSteps counts RL actions (each
    move/attack/deploy/end-turn call is one), and both seats together can spend up to
    ``2 * (starting_unit_count + 1)`` actions per round (one action per starting-unit slot plus an
    end-turn, per seat). Reaching ``round_cap`` rounds therefore needs
    ``round_cap * 2 * (starting_unit_count + 1)`` actions at minimum. Below this, the RL step budget
    truncates the episode before the engine's own round-cap backstop ever fires, faking a draw.
    """
    if document.get("environment") != "tactical-v2":
        return None
    starting_units = document["tactical_v2"]["starting_unit_count"]
    round_cap = document["rules"]["round_cap"]
    actions_per_round = 2 * (starting_units + 1)
    return actions_per_round * round_cap


def tactical_v2_round_cap_warning(document: Mapping[str, Any]) -> str | None:
    """A human-readable warning if ``document``'s tactical-v2 ``episode.max_steps`` is too small to
    reach its own configured round cap, else ``None`` (including for non-tactical-v2 documents).
    Never raises — see ``resolve_scenario``'s ``enforce_round_cap_minimum`` for the hard-error form.
    """
    minimum = tactical_v2_round_cap_minimum(document)
    if minimum is None:
        return None
    max_steps = document["episode"]["max_steps"]
    if max_steps >= minimum:
        return None
    return (
        f"tactical-v2 episode.max_steps ({max_steps}) is insufficient to reach the round cap "
        f"({document['rules']['round_cap']}) for {document['tactical_v2']['starting_unit_count']} "
        f"starting units; minimum required is {minimum}"
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _resolve_document(raw: Any) -> ResolvedScenario:
    validated = validate_scenario_document(raw)
    owned = copy.deepcopy(dict(validated))
    canonical_json = json.dumps(
        owned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    scenario_warnings: tuple[str, ...] = ()
    round_cap_warning = tactical_v2_round_cap_warning(owned)
    if round_cap_warning is not None:
        # Warn, don't break: an old scenario.json written before this check existed (including an
        # already-checkpointed, multi-million-step training run) must keep loading for resume/Arena.
        # A hard rejection belongs only at new-run creation time — see resolve_scenario's
        # enforce_round_cap_minimum.
        warnings.warn(round_cap_warning, stacklevel=3)
        scenario_warnings = (round_cap_warning,)
    return ResolvedScenario(
        schema_version=owned["schema_version"],
        template_id=owned["id"],
        name=owned["name"],
        environment=owned["environment"],
        document=_freeze(owned),
        canonical_json=canonical_json,
        warnings=scenario_warnings,
    )


def load_template_library(path: Path) -> list[ResolvedScenario]:
    library = _mapping(_read_document(path), "library")
    _exact_keys(library, _LIBRARY_KEYS, "library")
    schema_version = _integer(library["schema_version"], "library.schema_version")
    if schema_version != 1:
        raise ValueError("library.schema_version must be 1")
    raw_templates = library["templates"]
    if not isinstance(raw_templates, list):
        raise ValueError("library.templates must be an array")

    templates: list[ResolvedScenario] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_templates):
        scenario = _resolve_document(raw)
        if scenario.template_id in seen:
            raise ValueError(f"duplicate template id {scenario.template_id!r}")
        seen.add(scenario.template_id)
        expected_prefix = (
            "tactical-v3-"
            if scenario.environment == "tactical-v3"
            else scenario.environment.split("-", 1)[0] + "-"
        )
        if not scenario.template_id.startswith(expected_prefix):
            raise ValueError(
                f"template {scenario.template_id!r} does not match its environment "
                f"{scenario.environment!r}"
            )
        templates.append(scenario)
    return templates


def resolve_scenario(
    *,
    environment: str,
    scenario_file: Path | None,
    template_id: str | None,
    library_path: Path = DEFAULT_TEMPLATE_LIBRARY,
    enforce_round_cap_minimum: bool = False,
) -> ResolvedScenario:
    """Resolve a scenario document (explicit file or a named/default library template).

    ``enforce_round_cap_minimum`` escalates a tactical-v2 round-cap shortfall (see
    ``tactical_v2_round_cap_warning``) from a warning into a hard ``ValueError``. Defaults to
    ``False`` so every existing caller — including resuming a run from its frozen scenario.json —
    keeps loading unchanged; callers creating a brand-new run should opt in so an undersized
    max_steps is caught before training starts rather than silently faking a draw thousands of
    steps into an already-truncated episode.
    """
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError(
            "environment must be tactical-v1, tactical-v2, tactical-v3, or adaptive-v1"
        )
    if scenario_file is not None and template_id is not None:
        raise ValueError("scenario_file and template_id are mutually exclusive")
    if scenario_file is not None:
        scenario = _resolve_document(_read_document(scenario_file))
    else:
        # tactical-v1/adaptive-v1 default ids drop the "-v1" suffix ("tactical-standard",
        # "adaptive-standard"); later tactical contracts keep their full version tag so ids do
        # not collide with tactical-v1's.
        selected_id = template_id or f"{environment.removesuffix('-v1')}-standard"
        templates = load_template_library(library_path)
        scenario = next(
            (item for item in templates if item.template_id == selected_id),
            None,
        )
        if scenario is None:
            raise ValueError(f"unknown training template {selected_id!r}")
    if scenario.environment != environment:
        raise ValueError(
            f"scenario environment {scenario.environment!r} does not match "
            f"selected environment {environment!r}"
        )
    if enforce_round_cap_minimum and scenario.warnings:
        # Temporary shim until the tactical-v2 step budget is removed outright: a stale
        # session-authored max_steps must never block a launch. Auto-raise it to the
        # round-cap minimum (re-resolving through the one canonical path so the frozen
        # document, canonical JSON, and warnings stay consistent), then fail only on
        # warnings that cannot be repaired here.
        minimum = tactical_v2_round_cap_minimum(scenario.document)
        current = scenario.document.get("episode", {}).get("max_steps")
        if minimum is not None and isinstance(current, int) and current < minimum:
            patched = json.loads(scenario.canonical_json)
            patched["episode"]["max_steps"] = minimum
            scenario = _resolve_document(patched)
        if scenario.warnings:
            raise ValueError("; ".join(scenario.warnings))
    return scenario


def legacy_default_scenario(
    environment: str,
    *,
    library_path: Path = DEFAULT_TEMPLATE_LIBRARY,
) -> ResolvedScenario:
    standard = resolve_scenario(
        environment=environment,
        scenario_file=None,
        template_id=None,
        library_path=library_path,
    )
    document = copy.deepcopy(json.loads(standard.canonical_json))
    document["id"] = "legacy-default"
    document["name"] = "Standard"
    return _resolve_document(document)


def _space_value(spaces_info: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = spaces_info
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(
                "GymServer handshake is missing authoritative field "
                + ".".join(path)
            )
        value = value[part]
    return value


def _float32_equivalent(left: Any, right: Any) -> bool:
    if (
        isinstance(left, bool)
        or isinstance(right, bool)
        or not isinstance(left, (int, float))
        or not isinstance(right, (int, float))
    ):
        return False
    try:
        return struct.pack("!f", float(left)) == struct.pack("!f", float(right))
    except (OverflowError, struct.error, TypeError, ValueError):
        return False


def validate_handshake(
    scenario: ResolvedScenario, spaces_info: Mapping[str, Any]
) -> None:
    document = scenario.document
    comparisons: list[tuple[str, Any, tuple[str, ...]]] = [
        ("id", scenario.template_id, ("scenario_id",)),
        ("schema_version", scenario.schema_version, ("scenario_schema_version",)),
        ("environment", scenario.environment, ("contract_version",)),
        ("board.width", document["board"]["width"], ("board_w",)),
        ("board.height", document["board"]["height"], ("board_h",)),
        ("rules.round_cap", document["rules"]["round_cap"], ("round_cap",)),
        (
            "rules.biomes_enabled",
            document["rules"]["biomes_enabled"],
            ("biomes",),
        ),
    ]
    for key, requested in document["board"].items():
        comparisons.append((f"board.{key}", requested, ("board", key)))
    for key, requested in document["rules"].items():
        authoritative = -1 if key == "actions_per_turn" and requested == 0 else requested
        comparisons.append((f"rules.{key}", authoritative, ("board", key)))
    comparisons.append(
        ("episode.max_steps", document["episode"]["max_steps"], ("max_steps",))
    )
    for key, requested in document["reward"].items():
        comparisons.append((f"reward.{key}", requested, ("reward", key)))
    if scenario.environment == "adaptive-v1":
        for key, requested in document["adaptive"].items():
            comparisons.append((f"adaptive.{key}", requested, ("adaptive", key)))
    elif scenario.environment == "tactical-v2":
        # Templates are echoed back in a different (flat, cost-annotated) shape than the
        # scenario document's authored form, so only the scalar roster knobs are compared
        # here; template-catalog fidelity is cross-checked against contract_roster instead.
        for key in ("starting_unit_count", "max_controllable_units", "placement_policy"):
            comparisons.append(
                (f"tactical_v2.{key}", document["tactical_v2"][key], ("tactical_v2", key))
            )

    for field_path, requested, handshake_path in comparisons:
        authoritative = _space_value(spaces_info, handshake_path)
        matches = authoritative == requested
        if not matches and field_path.startswith("reward."):
            matches = _float32_equivalent(authoritative, requested)
        if not matches:
            raise ValueError(
                f"scenario {scenario.template_id!r} field {field_path} requested "
                f"{requested!r} but GymServer reported {authoritative!r}"
            )
