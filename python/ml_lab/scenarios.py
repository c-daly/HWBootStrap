"""Strict schema-v1 scenario resolution and canonical snapshots."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .io import atomic_write_text


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TEMPLATE_LIBRARY = (
    PROJECT_ROOT / "python" / "config" / "training-game-templates.json"
)
SUPPORTED_ENVIRONMENTS = frozenset({"tactical-v1", "adaptive-v1"})
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
_ADAPTIVE_KEYS = frozenset(
    {"starting_unit_count", "starting_army_budget", "max_design_point_cost"}
)
_CHEAPEST_ADAPTIVE_TEMPLATE_COST = 20


@dataclass(frozen=True)
class ResolvedScenario:
    schema_version: int
    template_id: str
    name: str
    environment: str
    document: Mapping[str, Any]
    canonical_json: str

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


def _validate_document(raw: Any) -> Mapping[str, Any]:
    document = _mapping(raw, "scenario")
    environment = _text(document.get("environment"), "environment")
    expected_keys = _COMMON_SCENARIO_KEYS | (
        frozenset({"adaptive"}) if environment == "adaptive-v1" else frozenset()
    )
    _exact_keys(document, expected_keys, "scenario")

    schema_version = _integer(document["schema_version"], "schema_version")
    if schema_version != 1:
        raise ValueError("schema_version must be 1")
    _text(document["id"], "id")
    _text(document["name"], "name")
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError("environment must be tactical-v1 or adaptive-v1")

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
        _TACTICAL_REWARD_KEYS
        if environment == "tactical-v1"
        else _ADAPTIVE_REWARD_KEYS
    )
    _exact_keys(reward, reward_keys, "reward")
    for key in reward_keys:
        _number(reward[key], f"reward.{key}")

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


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _resolve_document(raw: Any) -> ResolvedScenario:
    validated = _validate_document(raw)
    owned = copy.deepcopy(dict(validated))
    canonical_json = json.dumps(
        owned,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return ResolvedScenario(
        schema_version=owned["schema_version"],
        template_id=owned["id"],
        name=owned["name"],
        environment=owned["environment"],
        document=_freeze(owned),
        canonical_json=canonical_json,
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
        expected_prefix = scenario.environment.split("-", 1)[0] + "-"
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
) -> ResolvedScenario:
    if environment not in SUPPORTED_ENVIRONMENTS:
        raise ValueError("environment must be tactical-v1 or adaptive-v1")
    if scenario_file is not None and template_id is not None:
        raise ValueError("scenario_file and template_id are mutually exclusive")
    if scenario_file is not None:
        scenario = _resolve_document(_read_document(scenario_file))
    else:
        selected_id = template_id or f"{environment.split('-', 1)[0]}-standard"
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

    for field_path, requested, handshake_path in comparisons:
        authoritative = _space_value(spaces_info, handshake_path)
        if authoritative != requested:
            raise ValueError(
                f"scenario {scenario.template_id!r} field {field_path} requested "
                f"{requested!r} but GymServer reported {authoritative!r}"
            )
