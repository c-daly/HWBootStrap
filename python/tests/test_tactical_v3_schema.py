import copy
import hashlib
import json

import pytest

from ml_lab.tactical_v3_schema import (
    canonical_sha256,
    parse_decision,
    parse_spaces,
    parse_view,
)


def raw_canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def minimal_spaces_payload() -> dict[str, object]:
    encoding = {
        "schema_version": 1,
        "version": "tactical-v3",
        "hex_offset_layout": "odd-q",
        "token_reference_schema": ["table:table_kind", "row:int32"],
    }
    capacity = {
        "max_cells": 4,
        "max_units": 4,
        "max_templates": 4,
        "max_capability_definitions": 9,
        "max_capability_allocations": 16,
        "max_rules": 16,
        "max_memory_records": 4,
        "max_relations": 16,
        "max_candidates": 16,
    }
    return {
        "scenario_id": "in-memory-schema-test",
        "scenario_schema_version": 1,
        "contract_version": "tactical-v3",
        "contract_hash": "a" * 64,
        "encoding_hash": raw_canonical_sha256(encoding),
        "capacity_hash": raw_canonical_sha256(capacity),
        "environment_kind": "tactical",
        "match": {"board": {"width": 1, "height": 1}, "max_steps": 8},
        "encoding": encoding,
        "capacity": capacity,
    }


def minimal_view_payload() -> dict[str, object]:
    cell_ref = {"table": "cells", "row": 0}
    unit_ref = {"table": "units", "row": 0}
    definition_ref = {"table": "capability_definitions", "row": 0}
    projection = {
        "source_cell": cell_ref.copy(),
        "destination_cell": cell_ref.copy(),
        "template": None,
        "target": None,
        "horizontal_movement_spent": 1,
        "vertical_movement_spent": 0,
        "target_hp_delta": 0,
        "damage": 0,
        "is_lethal": False,
        "bounty_delta": 0,
        "points_delta": 0,
        "round_delta": 0,
        "is_terminal": False,
    }
    return {
        "decision_id": 7,
        "seat": 0,
        "observation": {
            "cells": [{
                "q": 0, "r": 0, "terrain": "plains", "elevation": 0,
                "self_deployment_zone": True, "opponent_deployment_zone": False,
                "controller": "self", "is_boundary": True,
                "currently_visible": True, "previously_observed": True,
            }],
            "units": [{
                "owner": "self", "current_hp": 2, "max_hp": 2,
                "cell": cell_ref.copy(), "elevation": 0,
                "moved": False, "attacked": False,
                "horizontal_movement_spent": 0, "vertical_movement_spent": 0,
                "point_cost": 3, "deploy_cost": 3, "currently_visible": True,
            }],
            "templates": [{
                "owner": "self", "point_cost": 3, "deploy_cost": 3,
                "is_fixed": True, "is_deployable": True,
            }],
            "capability_definitions": [{"kind": "health"}],
            "capability_allocations": [{
                "owner": unit_ref.copy(), "definition": definition_ref,
                "capability": "health", "purchased_level": 2, "effective_value": 2,
            }],
            "rules": [{
                "kind": "round", "int_value": 1,
                "float_value": 0.0, "bool_value": False,
            }],
            "memory": [],
            "relations": [{
                "kind": "occupies", "source": unit_ref.copy(), "target": cell_ref.copy(),
                "int_feature": 0, "float_feature": 0.0, "bool_feature": False,
            }],
        },
        "candidates": [{
            "candidate_id": 0, "decision_id": 7, "kind": "move",
            "actor": unit_ref.copy(), "target": None, "template": None,
            "cell": cell_ref.copy(), "projection": projection,
        }],
        "reward": {
            "terminal_outcome": 0.0,
            "known_health_adjusted_material_progress": 0.0,
            "public_resource_progress": 0.0,
            "time_pressure": 0.0,
            "total": 0.0,
            "finalized": False,
        },
        "winner": -1,
        "terminated": False,
        "truncated": False,
        "start_profile": "in-memory-1v1",
        "reference_seat": 0,
    }


EXPECTED_ERRORS = {
    "unknown_field": "view fields",
    "bool_candidate_id": "candidate_id must be an int32",
    "nan_rule_float": "rules\\[0\\].float_value must be finite",
    "wrong_ref_table": "move.actor references incompatible table",
    "ref_row_equal_to_length": "cells\\[1\\] of 1",
    "duplicate_candidate_id": "candidate ids must be exactly",
    "stale_candidate_decision": "candidate decision_id does not match",
}


def mutated_payload(mutation: str) -> tuple[dict[str, object], dict[str, object]]:
    spaces = minimal_spaces_payload()
    view = minimal_view_payload()
    if mutation == "unknown_field":
        view["extra"] = 1
    elif mutation == "bool_candidate_id":
        view["candidates"][0]["candidate_id"] = True
    elif mutation == "nan_rule_float":
        view["observation"]["rules"][0]["float_value"] = float("nan")
    elif mutation == "wrong_ref_table":
        view["candidates"][0]["actor"] = {"table": "cells", "row": 0}
    elif mutation == "ref_row_equal_to_length":
        view["observation"]["units"][0]["cell"] = {"table": "cells", "row": 1}
    elif mutation == "duplicate_candidate_id":
        view["candidates"].append(copy.deepcopy(view["candidates"][0]))
    elif mutation == "stale_candidate_decision":
        view["candidates"][0]["decision_id"] = 8
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    return spaces, view


def test_spaces_and_view_parse_to_deeply_immutable_semantic_values() -> None:
    spaces_payload = minimal_spaces_payload()
    view_payload = minimal_view_payload()
    identity = parse_spaces(spaces_payload)
    view = parse_view(view_payload, identity)
    assert identity.encoding_hash == raw_canonical_sha256(spaces_payload["encoding"])
    assert identity.capacity_hash == raw_canonical_sha256(spaces_payload["capacity"])
    assert canonical_sha256(identity.capacity) == identity.capacity_hash
    assert view.seat == view.decision.seat == 0
    assert view.decision == parse_decision(view_payload, identity)
    assert view.decision.candidates[0].decision_id == view.decision.decision_id == 7
    assert view.reward.finalized is False
    with pytest.raises(TypeError):
        identity.capacity["max_cells"] = 1
    with pytest.raises(AttributeError):
        view.decision.candidates.append(view.decision.candidates[0])



def test_spaces_preserves_finite_match_floats_immutably() -> None:
    spaces = minimal_spaces_payload()
    spaces["match"]["board"]["flat_chance"] = 0.25
    identity = parse_spaces(spaces)
    assert identity.match["board"]["flat_chance"] == 0.25
    with pytest.raises(TypeError):
        identity.match["board"]["flat_chance"] = 0.5

@pytest.mark.parametrize("mutation", tuple(EXPECTED_ERRORS))
def test_parser_rejects_malformed_wire_before_numeric_coercion(mutation: str) -> None:
    spaces, view = mutated_payload(mutation)
    with pytest.raises((TypeError, ValueError), match=EXPECTED_ERRORS[mutation]):
        parse_view(view, parse_spaces(spaces))


@pytest.mark.parametrize("kind", ("attack", "deploy", "end_turn"))
def test_parser_accepts_each_candidate_reference_family(kind: str) -> None:
    spaces = minimal_spaces_payload()
    view = minimal_view_payload()
    candidate = view["candidates"][0]
    projection = candidate["projection"]
    if kind == "attack":
        candidate.update({"kind": kind, "target": {"table": "units", "row": 0}, "cell": None})
        projection.update({"destination_cell": None, "target": {"table": "units", "row": 0}})
    elif kind == "deploy":
        candidate.update({"kind": kind, "actor": None, "template": {"table": "templates", "row": 0}})
        projection.update({"source_cell": None, "destination_cell": {"table": "cells", "row": 0}, "template": {"table": "templates", "row": 0}})
    else:
        candidate.update({"kind": kind, "actor": None, "cell": None})
        projection.update({"source_cell": None, "destination_cell": None})
    assert parse_view(view, parse_spaces(spaces)).decision.candidates[0].kind == kind


def test_parser_enforces_capacity_and_terminal_candidate_rules() -> None:
    spaces = minimal_spaces_payload()
    view = minimal_view_payload()
    spaces["capacity"]["max_candidates"] = 0
    spaces["capacity_hash"] = raw_canonical_sha256(spaces["capacity"])
    with pytest.raises(ValueError, match="capacity exceeded for candidates"):
        parse_view(view, parse_spaces(spaces))

    terminal = minimal_view_payload()
    terminal["terminated"] = True
    terminal["candidates"] = []
    assert parse_view(terminal, parse_spaces(minimal_spaces_payload())).decision.candidates == ()

    nonterminal = minimal_view_payload()
    nonterminal["candidates"] = []
    with pytest.raises(ValueError, match="nonterminal decision must have candidates"):
        parse_view(nonterminal, parse_spaces(minimal_spaces_payload()))
