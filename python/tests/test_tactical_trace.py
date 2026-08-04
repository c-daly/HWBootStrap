from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
from math import inf, nan

import pytest

from ml_lab.tactical_trace import EpisodeTrace


def _unit(*, unit_id: int = 10, current_hp: int = 4) -> dict[str, object]:
    return {
        "Id": unit_id,
        "Q": 2,
        "R": 3,
        "CurrentHp": current_hp,
        "MaximumHp": 5,
        "PointCost": 7,
        "Damage": 2,
        "Defense": 1,
        "Movement": 3,
        "VerticalMovement": 1,
        "Range": 4,
        "Moved": False,
        "Attacked": False,
        "MovementSpentH": 2,
        "MovementSpentV": 1,
    }


def _seat(*, seat: int, unit_id: int, material: float = 5.6) -> dict[str, object]:
    return {
        "Seat": seat,
        "Points": 9,
        "DestroyedValue": 0,
        "AliveUnits": 1,
        "CurrentHitPoints": 4,
        "MaximumHitPoints": 5,
        "HealthAdjustedMaterial": material,
        "CanDamageEnemy": True,
        "CanCurrentlyAttackEnemy": False,
        "CanMove": True,
        "Units": [_unit(unit_id=unit_id)],
    }


def _state(*, round_number: int, active_seat: int) -> dict[str, object]:
    return {
        "Round": round_number,
        "ActiveSeat": active_seat,
        "IsGameOver": False,
        "Winner": None,
        "ProductiveLegalActions": 3,
        "ControlledHexes": [
            {"Q": 1, "R": 0, "Controller": 1},
            {"Q": 0, "R": 0, "Controller": 0},
        ],
        "Seats": [_seat(seat=0, unit_id=10), _seat(seat=1, unit_id=20)],
    }


def transport_payload() -> dict[str, object]:
    first_before = _state(round_number=1, active_seat=0)
    first_after = _state(round_number=1, active_seat=1)
    second_after = _state(round_number=2, active_seat=0)
    return {
        "schema_version": 1,
        "transitions": [
            {
                "Before": first_before,
                "Command": {
                    "Kind": "move", "Issuer": 0, "ActorId": 10, "TargetId": None,
                    "Q": 3, "R": 3,
                },
                "After": first_after,
            },
            {
                "Before": copy.deepcopy(first_after),
                "Command": {
                    "Kind": "end_turn", "Issuer": 1, "ActorId": None, "TargetId": None,
                    "Q": None, "R": None,
                },
                "After": second_after,
            },
        ],
    }


def _camel_case(value: object) -> object:
    if isinstance(value, dict):
        return {key[:1].lower() + key[1:]: _camel_case(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_camel_case(item) for item in value]
    return value


def canonical_payload() -> dict[str, object]:
    state_1_seat_0 = {
        "seat": 0, "points": 9, "destroyed_value": 0, "alive_units": 1,
        "current_hit_points": 4, "maximum_hit_points": 5, "health_adjusted_material": 5.6,
        "can_damage_enemy": True, "can_currently_attack_enemy": False, "can_move": True,
        "units": [{
            "id": 10, "q": 2, "r": 3, "current_hp": 4, "maximum_hp": 5,
            "point_cost": 7, "damage": 2, "defense": 1, "movement": 3,
            "vertical_movement": 1, "range": 4, "moved": False, "attacked": False,
            "movement_spent_h": 2, "movement_spent_v": 1,
        }],
    }
    state_1_seat_1 = {
        **state_1_seat_0,
        "seat": 1,
        "units": [{**state_1_seat_0["units"][0], "id": 20}],
    }
    state_1_active_0 = {
        "round": 1, "active_seat": 0, "is_game_over": False, "winner": None,
        "productive_legal_actions": 3, "seats": [state_1_seat_0, state_1_seat_1],
        "controlled_hexes": [
            {"q": 0, "r": 0, "controller": 0},
            {"q": 1, "r": 0, "controller": 1},
        ],
    }
    state_1_active_1 = {**state_1_active_0, "active_seat": 1}
    state_2_active_0 = {**state_1_active_0, "round": 2}
    return {
        "schema_version": 1,
        "transitions": [
            {
                "before": state_1_active_0,
                "command": {
                    "kind": "move", "issuer": 0, "actor_id": 10, "target_id": None,
                    "q": 3, "r": 3,
                },
                "after": state_1_active_1,
            },
            {
                "before": state_1_active_1,
                "command": {
                    "kind": "end_turn", "issuer": 1, "actor_id": None, "target_id": None,
                    "q": None, "r": None,
                },
                "after": state_2_active_0,
            },
        ],
    }


def test_episode_trace_accepts_pascal_case_transport_and_emits_canonical_payload() -> None:
    parsed = EpisodeTrace.from_payload(transport_payload())

    assert parsed.schema_version == 1
    assert parsed.transitions[0].command.kind == "move"
    assert parsed.to_dict() == canonical_payload()


def test_episode_trace_accepts_camel_case_transport() -> None:
    payload = _camel_case(transport_payload())

    parsed = EpisodeTrace.from_payload(payload)

    assert parsed.to_dict() == canonical_payload()


def test_episode_trace_rejects_snake_case_transport_fields() -> None:
    payload = transport_payload()
    transition = payload["transitions"][0]
    assert isinstance(transition, dict)
    before = transition["Before"]
    assert isinstance(before, dict)
    before["active_seat"] = before.pop("ActiveSeat")

    with pytest.raises(ValueError, match=r"transitions\[0\]\.before\.active_seat"):
        EpisodeTrace.from_payload(payload)


def test_episode_trace_canonicalizes_seat_order() -> None:
    payload = transport_payload()
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    for transition in transitions:
        assert isinstance(transition, dict)
        for state_name in ("Before", "After"):
            state = transition[state_name]
            assert isinstance(state, dict)
            seats = state["Seats"]
            assert isinstance(seats, list)
            seats.reverse()

    assert EpisodeTrace.from_payload(payload).to_dict() == canonical_payload()



def test_episode_trace_rejects_discontinuous_transitions() -> None:
    payload = transport_payload()
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    second = transitions[1]
    assert isinstance(second, dict)
    before = second["Before"]
    assert isinstance(before, dict)
    before["Round"] = 2

    with pytest.raises(ValueError, match="transition 1 does not chain"):
        EpisodeTrace.from_payload(payload)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["transitions"][0]["Before"]["Seats"].__setitem__(
                1, _seat(seat=2, unit_id=20)
            ),
            "seat must be 0 or 1",
        ),
        (
            lambda payload: payload["transitions"][0]["Before"].__setitem__(
                "Seats", [_seat(seat=0, unit_id=10), _seat(seat=0, unit_id=20)]
            ),
            "seats must contain exactly 0 and 1",
        ),
        (
            lambda payload: payload["transitions"][0]["Before"]["Seats"][0]["Units"].append(
                _unit(unit_id=10)
            ),
            "duplicate unit id 10",
        ),
        (
            lambda payload: payload["transitions"][0]["Before"]["Seats"][0]["Units"][0].__setitem__(
                "CurrentHp", -1
            ),
            "current_hp must be non-negative",
        ),
        (
            lambda payload: payload["transitions"][0]["Before"]["Seats"][0].__setitem__(
                "AliveUnits", -1
            ),
            "alive_units must be non-negative",
        ),
        (
            lambda payload: payload["transitions"][0]["Before"]["Seats"][0].__setitem__(
                "HealthAdjustedMaterial", nan
            ),
            "health_adjusted_material must be finite",
        ),
        (
            lambda payload: payload.__setitem__("schema_version", True),
            "schema_version must be integer 1",
        ),
    ],
)
def test_episode_trace_rejects_corrupt_transport(
    mutate, message: str
) -> None:
    payload = transport_payload()
    mutate(payload)

    with pytest.raises(ValueError, match=message):
        EpisodeTrace.from_payload(payload)


@pytest.mark.parametrize("material", [inf, -inf])
def test_episode_trace_rejects_non_finite_material(material: float) -> None:
    payload = transport_payload()
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    before = transitions[0]["Before"]
    assert isinstance(before, dict)
    seats = before["Seats"]
    assert isinstance(seats, list)
    seats[0]["HealthAdjustedMaterial"] = material

    with pytest.raises(ValueError, match="health_adjusted_material must be finite"):
        EpisodeTrace.from_payload(payload)


def test_episode_trace_rejects_boolean_as_integer() -> None:
    payload = transport_payload()
    transitions = payload["transitions"]
    assert isinstance(transitions, list)
    transitions[0]["Command"]["Issuer"] = True

    with pytest.raises(ValueError, match="issuer must be an integer"):
        EpisodeTrace.from_payload(payload)


def test_episode_trace_rejects_wrong_schema_and_missing_required_values() -> None:
    wrong_schema = transport_payload()
    wrong_schema["schema_version"] = 2
    missing_kind = transport_payload()
    transitions = missing_kind["transitions"]
    assert isinstance(transitions, list)
    del transitions[0]["Command"]["Kind"]

    with pytest.raises(ValueError, match="schema_version must be integer 1"):
        EpisodeTrace.from_payload(wrong_schema)
    with pytest.raises(ValueError, match=r"command\.kind is required"):
        EpisodeTrace.from_payload(missing_kind)


def test_episode_trace_accepts_empty_transport_trace() -> None:
    assert EpisodeTrace.from_payload({"schema_version": 1, "transitions": []}).to_dict() == {
        "schema_version": 1,
        "transitions": [],
    }


def test_episode_trace_records_are_immutable_tuples() -> None:
    parsed = EpisodeTrace.from_payload(transport_payload())

    assert isinstance(parsed.transitions, tuple)
    assert isinstance(parsed.transitions[0].before.seats, tuple)
    assert isinstance(parsed.transitions[0].before.seats[0].units, tuple)
    with pytest.raises(FrozenInstanceError):
        parsed.transitions[0].command.kind = "attack"  # type: ignore[misc]
    assert isinstance(parsed.transitions[0].before.controlled_hexes, tuple)


def test_episode_trace_reopens_its_canonical_retained_payload() -> None:
    trace = EpisodeTrace.from_payload(transport_payload())

    assert EpisodeTrace.from_payload(trace.to_dict()) == trace
