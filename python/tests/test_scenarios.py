from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from ml_lab.scenarios import (
    DEFAULT_TEMPLATE_LIBRARY,
    load_template_library,
    resolve_scenario,
    validate_handshake,
)


def scenario_document(environment: str = "adaptive-v1") -> dict:
    document = {
        "schema_version": 1,
        "id": f"{environment.split('-', 1)[0]}-test",
        "name": "Test",
        "environment": environment,
        "board": {
            "width": 13,
            "height": 9,
            "max_elevation": 4,
            "zone_depth": 3,
            "flat_chance": 0.6,
            "plains_weight": 70,
            "forest_weight": 15,
            "rough_weight": 10,
            "water_weight": 5,
        },
        "rules": {
            "actions_per_turn": 0,
            "round_cap": 100,
            "starting_points": 12,
            "fog_of_war": environment == "adaptive-v1",
            "biomes_enabled": False,
            "bounty_rate": 0.5,
            "deploy_cost_multiplier": 1.0,
            "generator_cost": 2,
            "generator_output": 1,
            "generator_health": 3,
        },
        "episode": {"max_steps": 900 if environment == "adaptive-v1" else 600},
    }
    if environment == "adaptive-v1":
        document["reward"] = {
            "intermediate_decision_penalty": 0.001,
            "deployment_completion_bonus": 0.0,
        }
        document["adaptive"] = {
            "starting_unit_count": 6,
            "starting_army_budget": 132,
            "max_design_point_cost": 24,
        }
    else:
        document["reward"] = {
            "shape_scale": 0.01,
            "step_penalty": 0.005,
            "closing_weight": 0.02,
            "draw_credit_weight": 0.25,
            "points_weight": 0.5,
        }
    return document


def write_scenario(
    tmp_path: Path, *, environment: str = "adaptive-v1", document: dict | None = None
) -> Path:
    path = tmp_path / "scenario.json"
    path.write_text(
        json.dumps(document or scenario_document(environment), indent=2),
        encoding="utf-8",
    )
    return path


def write_library(tmp_path: Path, templates: list[dict], schema_version=1) -> Path:
    path = tmp_path / "library.json"
    path.write_text(
        json.dumps({"schema_version": schema_version, "templates": templates}),
        encoding="utf-8",
    )
    return path


def set_path(document: dict, path: str, value) -> None:
    target = document
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = value


def test_builtin_library_has_three_templates_per_environment() -> None:
    templates = load_template_library(DEFAULT_TEMPLATE_LIBRARY)
    assert [item.template_id for item in templates] == [
        "tactical-standard",
        "tactical-long-battle",
        "tactical-large-battle",
        "adaptive-standard",
        "adaptive-long-battle",
        "adaptive-large-battle",
    ]


def test_builtin_library_presets_have_exact_horizons_and_geometry() -> None:
    templates = {
        item.template_id: item.document
        for item in load_template_library(DEFAULT_TEMPLATE_LIBRARY)
    }
    assert [
        (
            template_id,
            document["board"]["width"],
            document["board"]["height"],
            document["board"]["zone_depth"],
            document["rules"]["round_cap"],
            document["episode"]["max_steps"],
        )
        for template_id, document in templates.items()
    ] == [
        ("tactical-standard", 13, 9, 3, 100, 600),
        ("tactical-long-battle", 13, 9, 3, 200, 1200),
        ("tactical-large-battle", 24, 16, 4, 150, 1200),
        ("adaptive-standard", 13, 9, 3, 100, 900),
        ("adaptive-long-battle", 13, 9, 3, 200, 1800),
        ("adaptive-large-battle", 24, 16, 4, 150, 1800),
    ]


def test_explicit_scenario_is_canonical_and_environment_checked(
    tmp_path: Path,
) -> None:
    path = write_scenario(tmp_path, environment="adaptive-v1")
    first = resolve_scenario(
        environment="adaptive-v1", scenario_file=path, template_id=None
    )
    second = resolve_scenario(
        environment="adaptive-v1", scenario_file=path, template_id=None
    )
    assert first.canonical_json == second.canonical_json
    assert first.canonical_json == json.dumps(
        scenario_document(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    with pytest.raises(ValueError, match="environment"):
        resolve_scenario(
            environment="tactical-v1", scenario_file=path, template_id=None
        )


def test_resolved_document_is_deeply_immutable_and_write_is_canonical(
    tmp_path: Path,
) -> None:
    source = scenario_document()
    path = write_scenario(tmp_path, document=source)
    scenario = resolve_scenario(
        environment="adaptive-v1", scenario_file=path, template_id=None
    )

    source["board"]["width"] = 99
    with pytest.raises(TypeError):
        scenario.document["board"]["width"] = 99
    assert scenario.document["board"]["width"] == 13

    output = tmp_path / "nested" / "resolved.json"
    scenario.write(output)
    assert output.read_text(encoding="utf-8") == scenario.canonical_json + "\n"


def test_default_and_named_template_resolution_are_environment_scoped() -> None:
    default = resolve_scenario(
        environment="adaptive-v1", scenario_file=None, template_id=None
    )
    assert default.template_id == "adaptive-standard"

    selected = resolve_scenario(
        environment="adaptive-v1",
        scenario_file=None,
        template_id="adaptive-large-battle",
    )
    assert selected.document["board"]["width"] == 24

    with pytest.raises(ValueError, match="environment"):
        resolve_scenario(
            environment="tactical-v1",
            scenario_file=None,
            template_id="adaptive-standard",
        )


@pytest.mark.parametrize(
    "path",
    [
        "schema_version",
        "board.width",
        "board.height",
        "board.max_elevation",
        "board.zone_depth",
        "board.plains_weight",
        "rules.actions_per_turn",
        "rules.round_cap",
        "rules.starting_points",
        "rules.generator_cost",
        "episode.max_steps",
        "adaptive.starting_unit_count",
    ],
)
def test_boolean_is_not_accepted_as_an_integer(
    tmp_path: Path, path: str
) -> None:
    document = scenario_document()
    set_path(document, path, True)
    with pytest.raises(ValueError, match=path):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(tmp_path, document=document),
            template_id=None,
        )


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
@pytest.mark.parametrize(
    "path",
    [
        "board.flat_chance",
        "rules.bounty_rate",
        "rules.deploy_cost_multiplier",
        "reward.intermediate_decision_penalty",
    ],
)
def test_non_finite_numbers_are_rejected(
    tmp_path: Path, path: str, value: float
) -> None:
    document = scenario_document()
    set_path(document, path, value)
    scenario_path = tmp_path / "scenario.json"
    scenario_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="finite|JSON"):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=scenario_path,
            template_id=None,
        )


@pytest.mark.parametrize("section", ["board", "rules", "episode", "reward"])
def test_missing_required_sections_are_rejected(
    tmp_path: Path, section: str
) -> None:
    document = scenario_document()
    document.pop(section)
    with pytest.raises(ValueError, match=section):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(tmp_path, document=document),
            template_id=None,
        )


def test_wrong_environment_reward_kind_is_rejected(tmp_path: Path) -> None:
    document = scenario_document()
    document["reward"] = scenario_document("tactical-v1")["reward"]
    with pytest.raises(ValueError, match="reward"):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(tmp_path, document=document),
            template_id=None,
        )


@pytest.mark.parametrize(
    ("path", "value", "expected"),
    [
        ("adaptive.starting_unit_count", 0, "starting_unit_count"),
        ("adaptive.starting_unit_count", 25, "starting_unit_count"),
        ("adaptive.starting_army_budget", 0, "starting_army_budget"),
        ("adaptive.max_design_point_cost", 0, "max_design_point_cost"),
    ],
)
def test_invalid_adaptive_values_are_rejected(
    tmp_path: Path, path: str, value: int, expected: str
) -> None:
    document = scenario_document()
    set_path(document, path, value)
    with pytest.raises(ValueError, match=expected):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(tmp_path, document=document),
            template_id=None,
        )


def test_duplicate_template_ids_are_rejected(tmp_path: Path) -> None:
    template = scenario_document("tactical-v1")
    with pytest.raises(ValueError, match="duplicate"):
        load_template_library(write_library(tmp_path, [template, copy.deepcopy(template)]))


def test_library_template_id_must_match_its_environment(tmp_path: Path) -> None:
    template = scenario_document("adaptive-v1")
    template["id"] = "tactical-wrong"
    with pytest.raises(ValueError, match="environment"):
        load_template_library(write_library(tmp_path, [template]))


def test_template_library_schema_version_is_strict(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="schema_version"):
        load_template_library(
            write_library(tmp_path, [scenario_document("tactical-v1")], schema_version=2)
        )


def test_validate_handshake_checks_authoritative_scenario_values() -> None:
    scenario = resolve_scenario(
        environment="adaptive-v1", scenario_file=None, template_id=None
    )
    spaces = {
        "scenario_id": "adaptive-standard",
        "scenario_schema_version": 1,
        "contract_version": "adaptive-v1",
        "board_w": 13,
        "board_h": 9,
        "round_cap": 100,
        "max_steps": 900,
        "biomes": False,
        "board": {
            **dict(scenario.document["board"]),
            **dict(scenario.document["rules"]),
            "actions_per_turn": -1,
        },
        "reward": {
            **dict(scenario.document["reward"]),
            "terminal_win": 1.0,
            "terminal_loss": -1.0,
        },
        "adaptive": dict(scenario.document["adaptive"]),
    }
    validate_handshake(scenario, spaces)

    spaces["board_w"] = 24
    with pytest.raises(
        ValueError, match=r"adaptive-standard.*board\.width.*13.*24"
    ):
        validate_handshake(scenario, spaces)
