from __future__ import annotations

import copy
import json
import math
import tempfile
from pathlib import Path

import pytest

from ml_lab.scenarios import (
    DEFAULT_TEMPLATE_LIBRARY,
    load_template_library,
    resolve_scenario,
    tactical_v2_round_cap_minimum,
    tactical_v2_round_cap_warning,
    validate_handshake,
    validate_scenario_document,
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


TACTICAL_V2_TEMPLATES = [
    {
        "id": "brute-85597320",
        "name": "Brute",
        "stats": {
            "health": 7, "damage": 2, "defense": 2, "movement": 3, "vertical_movement": 2,
            "range": 1, "range_arc": 1, "vision": 2, "vision_arc": 1,
        },
    },
    {
        "id": "striker-0d7b6999",
        "name": "Striker",
        "stats": {
            "health": 2, "damage": 6, "defense": 0, "movement": 3, "vertical_movement": 2,
            "range": 2, "range_arc": 1, "vision": 3, "vision_arc": 1,
        },
    },
    {
        "id": "sniper-d065c02a",
        "name": "Sniper",
        "stats": {
            "health": 2, "damage": 2, "defense": 0, "movement": 2, "vertical_movement": 2,
            "range": 6, "range_arc": 1, "vision": 4, "vision_arc": 1,
        },
    },
    {
        "id": "artillery-27c01722",
        "name": "Artillery",
        "stats": {
            "health": 3, "damage": 6, "defense": 0, "movement": 0, "vertical_movement": 0,
            "range": 5, "range_arc": 2, "vision": 2, "vision_arc": 1,
        },
    },
    {
        "id": "scout-d3503dfa",
        "name": "Scout",
        "stats": {
            "health": 2, "damage": 0, "defense": 0, "movement": 4, "vertical_movement": 3,
            "range": 0, "range_arc": 0, "vision": 7, "vision_arc": 2,
        },
    },
]

TACTICAL_V2_START_PROFILES = [
    {"id": "standard-3v3", "learner_units": 3, "opponent_units": 3,
     "separation": "legacy-mirrored"},
    {"id": "conversion-3v1-near", "learner_units": 3, "opponent_units": 1,
     "separation": "near"},
    {"id": "conversion-3v1-medium", "learner_units": 3, "opponent_units": 1,
     "separation": "medium"},
    {"id": "conversion-3v1-far", "learner_units": 3, "opponent_units": 1,
     "separation": "far"},
    {"id": "conversion-2v1-near", "learner_units": 2, "opponent_units": 1,
     "separation": "near"},
    {"id": "conversion-2v1-medium", "learner_units": 2, "opponent_units": 1,
     "separation": "medium"},
    {"id": "conversion-2v1-far", "learner_units": 2, "opponent_units": 1,
     "separation": "far"},
    {"id": "conversion-1v1-near", "learner_units": 1, "opponent_units": 1,
     "separation": "near"},
    {"id": "conversion-1v1-medium", "learner_units": 1, "opponent_units": 1,
     "separation": "medium"},
    {"id": "conversion-1v1-far", "learner_units": 1, "opponent_units": 1,
     "separation": "far"},
]


def tactical_v2_document(*, starting_units: int = 3) -> dict:
    document = scenario_document("tactical-v1")
    document["id"] = "tactical-v2-test"
    document["environment"] = "tactical-v2"
    document["tactical_v2"] = {
        "starting_unit_count": starting_units,
        "max_controllable_units": starting_units,
        "placement_policy": "symmetric-random-v1",
        "templates": copy.deepcopy(TACTICAL_V2_TEMPLATES),
    }
    # Big enough to reach round_cap for any starting_units this helper is called with, so tests that
    # aren't specifically exercising the round-cap warning don't spuriously trigger one.
    document["episode"]["max_steps"] = tactical_v2_round_cap_minimum(document)
    return document


def profiled_tactical_v2_document(*, mixed: bool = False) -> dict:
    document = tactical_v2_document()
    tactical_v2 = document["tactical_v2"]
    tactical_v2["placement_policy"] = "profiled-seeded-v1"
    tactical_v2["start_profiles"] = copy.deepcopy(TACTICAL_V2_START_PROFILES)
    tactical_v2["start_distribution"] = [
        {
            "profile_id": profile["id"],
            "basis_points": (
                4000 if mixed and profile["id"] == "standard-3v3"
                else 1000 if mixed and not profile["id"].endswith("-medium")
                else 0 if mixed
                else 10000 if profile["id"] == "standard-3v3"
                else 0
            ),
        }
        for profile in TACTICAL_V2_START_PROFILES
    ]
    return document


def write_scenario(document: dict | None = None, *, environment: str = "adaptive-v1") -> Path:
    """Write a scenario to its own throwaway directory (no pytest tmp_path needed)."""
    directory = Path(tempfile.mkdtemp(prefix="hexwars-scenario-"))
    path = directory / "scenario.json"
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


def tactical_v3_document(template_id: str = "tactical-v3-standard") -> dict:
    library = json.loads(DEFAULT_TEMPLATE_LIBRARY.read_text(encoding="utf-8"))
    return copy.deepcopy(
        next(item for item in library["templates"] if item["id"] == template_id)
    )


def test_builtin_library_has_expected_ordered_templates() -> None:
    templates = load_template_library(DEFAULT_TEMPLATE_LIBRARY)
    assert [item.template_id for item in templates] == [
        "tactical-standard",
        "tactical-long-battle",
        "tactical-large-battle",
        "tactical-v2-standard",
        "tactical-v2-long-battle",
        "tactical-v2-large-battle",
        "adaptive-standard",
        "adaptive-long-battle",
        "adaptive-large-battle",
        "tactical-v3-standard",
        "tactical-v3-full-roster",
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
        # tactical-v2's max_steps is derived (100-round-rule): round_cap * 2 * (starting_unit_count +
        # 1) + one extra round's headroom. All three presets have starting_unit_count == 3, so
        # actions_per_round == 8: 100 -> 808, 200 -> 1608, 150 -> 1208.
        ("tactical-v2-standard", 13, 9, 3, 100, 808),
        ("tactical-v2-long-battle", 13, 9, 3, 200, 1608),
        ("tactical-v2-large-battle", 24, 16, 4, 150, 1208),
        ("adaptive-standard", 13, 9, 3, 100, 900),
        ("adaptive-long-battle", 13, 9, 3, 200, 1800),
        ("adaptive-large-battle", 24, 16, 4, 150, 1800),
        ("tactical-v3-standard", 13, 9, 3, 100, 808),
        ("tactical-v3-full-roster", 13, 9, 3, 100, 808),
    ]


def test_builtin_tactical_v2_presets_share_the_canonical_five_template_catalog() -> None:
    templates = {
        item.template_id: item.document
        for item in load_template_library(DEFAULT_TEMPLATE_LIBRARY)
    }
    for template_id in (
        "tactical-v2-standard", "tactical-v2-long-battle", "tactical-v2-large-battle",
    ):
        tactical_v2 = templates[template_id]["tactical_v2"]
        assert tactical_v2["starting_unit_count"] == 3
        assert tactical_v2["max_controllable_units"] == 3
        assert tactical_v2["placement_policy"] == "symmetric-random-v1"
        assert [item["name"] for item in tactical_v2["templates"]] == [
            "Brute", "Striker", "Sniper", "Artillery", "Scout",
        ]


def test_explicit_scenario_is_canonical_and_environment_checked() -> None:
    path = write_scenario(environment="adaptive-v1")
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
    path = write_scenario(document=source)
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


def test_tactical_v3_default_and_full_roster_templates_resolve_strictly() -> None:
    standard = resolve_scenario(
        environment="tactical-v3", scenario_file=None, template_id=None
    )
    full_roster = resolve_scenario(
        environment="tactical-v3",
        scenario_file=None,
        template_id="tactical-v3-full-roster",
    )

    assert standard.template_id == "tactical-v3-standard"
    assert len(standard.document["tactical_v3"]["templates"]) == 3
    assert full_roster.template_id == "tactical-v3-full-roster"
    assert len(full_roster.document["tactical_v3"]["templates"]) == 5
    assert (
        standard.document["tactical_v3"]["capacity"]
        == full_roster.document["tactical_v3"]["capacity"]
    )


def test_tactical_v3_accepts_unity_float32_stage_one_reward_values() -> None:
    document = tactical_v3_document()
    document["reward"].update(
        {
            "material_adjustment_bound": 0.20000000298023225,
            "time_pressure_bound": 0.05000000074505806,
        }
    )

    validate_scenario_document(document)


def test_tactical_v3_rejects_a_different_float32_stage_one_reward_value() -> None:
    document = tactical_v3_document()
    document["reward"]["material_adjustment_bound"] = 0.20000002

    with pytest.raises(ValueError, match="stage-one value"):
        validate_scenario_document(document)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["reward"].update({"terminal_win": 0.5}),
         "stage-one value"),
        (lambda value: value["rules"].update({"fog_of_war": True}),
         "fog and biomes disabled"),
        (lambda value: value["episode"].update({"max_steps": 799}),
         "insufficient to reach the round cap"),
        (lambda value: value["board"].update({"height": 1, "zone_depth": 1}),
         "deployment cells must cover starting_unit_count"),
        (lambda value: value["tactical_v3"]["capacity"].update({"max_cells": 1}),
         r"capacity\.max_cells must be at least"),
        (lambda value: value["tactical_v3"]["start_distribution"][0].update(
            {"basis_points": 6999}), "sum to 10000"),
        (lambda value: value["tactical_v3"]["start_profiles"].pop(),
         "exact versioned start profile catalog"),
        (lambda value: value["tactical_v3"]["templates"][0]["stats"].update(
            {"health": 0}), "health must be positive"),
        (lambda value: value["tactical_v3"].update({"unexpected": True}),
         "unexpected tactical_v3.unexpected"),
    ],
)
def test_tactical_v3_rejects_invalid_contract_documents(mutation, match: str) -> None:
    document = tactical_v3_document()
    mutation(document)

    with pytest.raises(ValueError, match=match):
        validate_scenario_document(document)


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
def test_boolean_is_not_accepted_as_an_integer(path: str) -> None:
    document = scenario_document()
    set_path(document, path, True)
    with pytest.raises(ValueError, match=path):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(document=document),
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
def test_missing_required_sections_are_rejected(section: str) -> None:
    document = scenario_document()
    document.pop(section)
    with pytest.raises(ValueError, match=section):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(document=document),
            template_id=None,
        )


def test_wrong_environment_reward_kind_is_rejected() -> None:
    document = scenario_document()
    document["reward"] = scenario_document("tactical-v1")["reward"]
    with pytest.raises(ValueError, match="reward"):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(document=document),
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
    path: str, value: int, expected: str
) -> None:
    document = scenario_document()
    set_path(document, path, value)
    with pytest.raises(ValueError, match=expected):
        resolve_scenario(
            environment="adaptive-v1",
            scenario_file=write_scenario(document=document),
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

    tactical_v3 = tactical_v3_document()
    tactical_v3["id"] = "tactical-unversioned"
    with pytest.raises(ValueError, match="environment"):
        load_template_library(write_library(tmp_path, [tactical_v3]))


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


def test_validate_handshake_accepts_float32_equivalent_reward_values() -> None:
    document = scenario_document("tactical-v1")
    document["reward"].update(
        {
            "shape_scale": 0.009999999776482582,
            "step_penalty": 0.004999999888241291,
            "closing_weight": 0.019999999552965165,
        }
    )
    scenario = resolve_scenario(
        environment="tactical-v1",
        scenario_file=write_scenario(document=document),
        template_id=None,
    )
    spaces = {
        "scenario_id": "tactical-test",
        "scenario_schema_version": 1,
        "contract_version": "tactical-v1",
        "board_w": 13,
        "board_h": 9,
        "round_cap": 100,
        "max_steps": 600,
        "biomes": False,
        "board": {
            **dict(scenario.document["board"]),
            **dict(scenario.document["rules"]),
            "actions_per_turn": -1,
        },
        "reward": {
            **dict(scenario.document["reward"]),
            "shape_scale": 0.01,
            "step_penalty": 0.005,
            "closing_weight": 0.02,
        },
    }

    validate_handshake(scenario, spaces)


def test_tactical_v2_scenario_preserves_roster_and_count():
    scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=write_scenario(tactical_v2_document(starting_units=7)),
        template_id=None,
    )

    assert scenario.document["tactical_v2"]["starting_unit_count"] == 7
    assert [item["name"] for item in scenario.document["tactical_v2"]["templates"]][:2] == [
        "Brute", "Striker"
    ]


@pytest.mark.parametrize("count", [0, 13])
def test_tactical_v2_rejects_invalid_starting_count(count):
    document = tactical_v2_document(starting_units=count)
    with pytest.raises(ValueError, match="between 1 and 12"):
        validate_scenario_document(document)


@pytest.mark.parametrize(
    ("starting_units", "round_cap", "expected_minimum"),
    [(1, 100, 400), (3, 100, 800), (12, 100, 2600), (3, 200, 1600), (3, 150, 1200)],
)
def test_tactical_v2_round_cap_minimum_matches_engine_derivation(
    starting_units: int, round_cap: int, expected_minimum: int
) -> None:
    document = tactical_v2_document(starting_units=starting_units)
    document["rules"]["round_cap"] = round_cap
    assert tactical_v2_round_cap_minimum(document) == expected_minimum


def test_tactical_v2_round_cap_minimum_is_none_for_non_tactical_v2() -> None:
    assert tactical_v2_round_cap_minimum(scenario_document("tactical-v1")) is None
    assert tactical_v2_round_cap_minimum(scenario_document("adaptive-v1")) is None


def test_tactical_v2_round_cap_warning_fires_only_when_max_steps_is_undersized() -> None:
    document = tactical_v2_document(starting_units=12)
    document["episode"]["max_steps"] = 600  # the old, pre-fix default: too small for 12 units

    warning = tactical_v2_round_cap_warning(document)

    assert warning is not None
    assert "600" in warning
    assert "100" in warning  # round cap
    assert "2600" in warning  # required minimum

    document["episode"]["max_steps"] = 2600
    assert tactical_v2_round_cap_warning(document) is None


def test_run_local_scenario_with_old_small_max_steps_still_resolves_with_a_warning(
    recwarn: pytest.WarningsChecker,
) -> None:
    """An existing run's frozen scenario.json (e.g. a checkpointed multi-million-step training run)
    predating this check must keep loading for resume/Arena — the shortfall is surfaced as a warning,
    never a hard rejection, unless the caller opts into enforce_round_cap_minimum (new-run creation)."""
    document = tactical_v2_document(starting_units=12)
    document["episode"]["max_steps"] = 600
    path = write_scenario(document=document)

    scenario = resolve_scenario(environment="tactical-v2", scenario_file=path, template_id=None)

    assert scenario.document["episode"]["max_steps"] == 600
    assert scenario.warnings
    assert "insufficient to reach the round cap" in scenario.warnings[0]
    assert any("insufficient to reach the round cap" in str(w.message) for w in recwarn.list)


def test_new_run_with_insufficient_max_steps_is_auto_raised_when_enforced() -> None:
    """The repair form: a new-run creation path (e.g. the CLI's `train` command, not `resume`)
    opts into enforce_round_cap_minimum, and an undersized max_steps — typically a stale
    session-authored value — is auto-raised to the round-cap minimum instead of blocking the
    launch. A launch must never fail over a value the resolver itself knows how to correct;
    the whole knob is scheduled for deletion (game length is denominated in rounds)."""
    document = tactical_v2_document(starting_units=12)
    document["episode"]["max_steps"] = 600
    path = write_scenario(document=document)

    scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=path,
        template_id=None,
        enforce_round_cap_minimum=True,
    )

    assert scenario.document["episode"]["max_steps"] == 2600
    assert scenario.warnings == ()


def test_new_run_with_sufficient_max_steps_is_not_rejected_when_enforced() -> None:
    document = tactical_v2_document(starting_units=12)
    document["episode"]["max_steps"] = 2600
    path = write_scenario(document=document)

    scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=path,
        template_id=None,
        enforce_round_cap_minimum=True,
    )

    assert scenario.warnings == ()


def test_tactical_v2_default_template_selection_does_not_collide_with_tactical_v1() -> None:
    default = resolve_scenario(
        environment="tactical-v2", scenario_file=None, template_id=None
    )
    assert default.template_id == "tactical-v2-standard"
    assert default.environment == "tactical-v2"


def test_tactical_v2_rejects_mismatched_controllable_units() -> None:
    document = tactical_v2_document(starting_units=3)
    document["tactical_v2"]["max_controllable_units"] = 4
    with pytest.raises(ValueError, match="max_controllable_units"):
        validate_scenario_document(document)


def test_tactical_v2_rejects_non_canonical_placement_policy() -> None:
    document = tactical_v2_document()
    document["tactical_v2"]["placement_policy"] = "random"
    with pytest.raises(ValueError, match="placement_policy"):
        validate_scenario_document(document)


def test_tactical_v2_profiled_scenarios_accept_zero_weights_and_freeze_different_json() -> None:
    control = validate_scenario_document(profiled_tactical_v2_document(mixed=False))
    mixed = validate_scenario_document(profiled_tactical_v2_document(mixed=True))

    assert sum(
        item["basis_points"]
        for item in control["tactical_v2"]["start_distribution"]
    ) == 10000
    assert control["tactical_v2"]["start_profiles"] == mixed["tactical_v2"]["start_profiles"]
    assert control["tactical_v2"]["start_distribution"] != mixed["tactical_v2"]["start_distribution"]


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["start_profiles"].pop(), "exact versioned start profile catalog"),
        (lambda value: value["start_profiles"][1].update({"separation": "adjacent"}),
         "separation"),
        (lambda value: value["start_distribution"].pop(), "missing declared profile"),
        (lambda value: value["start_distribution"].append(
            {"profile_id": "standard-3v3", "basis_points": 0}), "duplicate"),
        (lambda value: value["start_distribution"][1].update(
            {"profile_id": "not-declared"}), "undeclared"),
        (lambda value: value["start_distribution"][0].update(
            {"basis_points": 9999}), "sum to 10000"),
    ],
)
def test_tactical_v2_profiled_scenario_rejects_invalid_catalog_and_distribution(
    mutation, match: str
) -> None:
    document = profiled_tactical_v2_document()
    mutation(document["tactical_v2"])

    with pytest.raises(ValueError, match=match):
        validate_scenario_document(document)


def test_tactical_v2_rejects_duplicate_template_ids() -> None:
    document = tactical_v2_document()
    document["tactical_v2"]["templates"][1] = copy.deepcopy(
        document["tactical_v2"]["templates"][0]
    )
    with pytest.raises(ValueError, match="duplicate"):
        validate_scenario_document(document)


def test_tactical_v2_rejects_extra_and_missing_fields() -> None:
    extra_section_field = tactical_v2_document()
    extra_section_field["tactical_v2"]["extra_field"] = 1
    with pytest.raises(ValueError, match="tactical_v2"):
        validate_scenario_document(extra_section_field)

    missing_stat = tactical_v2_document()
    del missing_stat["tactical_v2"]["templates"][0]["stats"]["range_arc"]
    with pytest.raises(ValueError, match="range_arc"):
        validate_scenario_document(missing_stat)

    missing_section = tactical_v2_document()
    del missing_section["tactical_v2"]
    with pytest.raises(ValueError, match="tactical_v2"):
        validate_scenario_document(missing_section)


def test_tactical_v2_rejects_deployment_zone_too_small_for_starting_units() -> None:
    document = tactical_v2_document(starting_units=12)
    document["board"]["height"] = 4
    document["board"]["zone_depth"] = 1
    with pytest.raises(ValueError, match="deployment cells"):
        validate_scenario_document(document)


def test_tactical_v2_library_template_ids_match_their_environment_prefix(
    tmp_path: Path,
) -> None:
    template = tactical_v2_document()
    template["id"] = "adaptive-wrong"
    with pytest.raises(ValueError, match="environment"):
        load_template_library(write_library(tmp_path, [template]))
