"""Immutable, validated diagnostics for TacticalV2 duel evaluation traces."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class UnitFrame:
    id: int
    q: int
    r: int
    current_hp: int
    maximum_hp: int
    point_cost: int
    damage: int
    defense: int
    movement: int
    vertical_movement: int
    range: int
    moved: bool
    attacked: bool


@dataclass(frozen=True)
class SeatFrame:
    seat: int
    points: int
    destroyed_value: int
    alive_units: int
    current_hit_points: int
    maximum_hit_points: int
    health_adjusted_material: float
    can_damage_enemy: bool
    can_currently_attack_enemy: bool
    can_move: bool
    units: tuple[UnitFrame, ...]


@dataclass(frozen=True)
class StateFrame:
    round: int
    active_seat: int
    is_game_over: bool
    winner: int | None
    productive_legal_actions: int
    seats: tuple[SeatFrame, SeatFrame]


@dataclass(frozen=True)
class CommandFrame:
    kind: str
    issuer: int
    actor_id: int | None
    target_id: int | None
    q: int | None
    r: int | None


@dataclass(frozen=True)
class TransitionFrame:
    before: StateFrame
    command: CommandFrame
    after: StateFrame


@dataclass(frozen=True)
class EpisodeTrace:
    schema_version: int
    transitions: tuple[TransitionFrame, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeTrace":
        _require_trace_envelope(payload)
        schema_version = _required_schema_version(payload)
        transitions = _parse_and_validate_transitions(payload)
        return cls(schema_version=schema_version, transitions=transitions)

    def to_dict(self) -> dict[str, Any]:
        return _canonical_trace_dict(self)


_ENVELOPE_KEYS = frozenset({"schema_version", "transitions"})


def _require_trace_envelope(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("trace payload must be a JSON object")
    if set(payload) != _ENVELOPE_KEYS:
        raise ValueError("trace payload must contain only schema_version and transitions")


def _required_schema_version(payload: Mapping[str, Any]) -> int:
    value = payload["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value != 1:
        raise ValueError("schema_version must be integer 1")
    return value


def _parse_and_validate_transitions(
    payload: Mapping[str, Any],
) -> tuple[TransitionFrame, ...]:
    raw_transitions = payload["transitions"]
    if not isinstance(raw_transitions, list):
        raise ValueError("transitions must be an array")
    transitions = tuple(
        _parse_transition(raw_transition, f"transitions[{index}]")
        for index, raw_transition in enumerate(raw_transitions)
    )
    for index in range(1, len(transitions)):
        if transitions[index - 1].after != transitions[index].before:
            raise ValueError(f"transition {index} does not chain from its predecessor")
    return transitions


def _parse_transition(value: Any, path: str) -> TransitionFrame:
    fields = _dto_fields(value, path, ("Before", "Command", "After"))
    return TransitionFrame(
        before=_parse_state(fields["Before"], f"{path}.before"),
        command=_parse_command(fields["Command"], f"{path}.command"),
        after=_parse_state(fields["After"], f"{path}.after"),
    )


def _parse_state(value: Any, path: str) -> StateFrame:
    fields = _dto_fields(
        value,
        path,
        ("Round", "ActiveSeat", "IsGameOver", "Winner", "ProductiveLegalActions", "Seats"),
    )
    raw_seats = fields["Seats"]
    if not isinstance(raw_seats, list):
        raise ValueError(f"{path}.seats must be an array")
    if len(raw_seats) != 2:
        raise ValueError(f"{path}.seats must contain exactly 0 and 1")
    seats = tuple(_parse_seat(item, f"{path}.seats[{index}]") for index, item in enumerate(raw_seats))
    by_seat = {seat.seat: seat for seat in seats}
    if set(by_seat) != {0, 1} or len(by_seat) != 2:
        raise ValueError(f"{path}.seats must contain exactly 0 and 1")
    winner = _optional_seat(fields["Winner"], f"{path}.winner")
    return StateFrame(
        round=_non_negative_integer(fields["Round"], f"{path}.round"),
        active_seat=_seat_number(fields["ActiveSeat"], f"{path}.active_seat"),
        is_game_over=_boolean(fields["IsGameOver"], f"{path}.is_game_over"),
        winner=winner,
        productive_legal_actions=_non_negative_integer(
            fields["ProductiveLegalActions"], f"{path}.productive_legal_actions"
        ),
        seats=(by_seat[0], by_seat[1]),
    )


def _parse_seat(value: Any, path: str) -> SeatFrame:
    fields = _dto_fields(
        value,
        path,
        (
            "Seat", "Points", "DestroyedValue", "AliveUnits", "CurrentHitPoints",
            "MaximumHitPoints", "HealthAdjustedMaterial", "CanDamageEnemy",
            "CanCurrentlyAttackEnemy", "CanMove", "Units",
        ),
    )
    raw_units = fields["Units"]
    if not isinstance(raw_units, list):
        raise ValueError(f"{path}.units must be an array")
    units = tuple(_parse_unit(item, f"{path}.units[{index}]") for index, item in enumerate(raw_units))
    unit_ids = [unit.id for unit in units]
    duplicate = next((unit_id for unit_id in unit_ids if unit_ids.count(unit_id) > 1), None)
    if duplicate is not None:
        raise ValueError(f"{path} has duplicate unit id {duplicate}")
    return SeatFrame(
        seat=_seat_number(fields["Seat"], f"{path}.seat"),
        points=_non_negative_integer(fields["Points"], f"{path}.points"),
        destroyed_value=_non_negative_integer(fields["DestroyedValue"], f"{path}.destroyed_value"),
        alive_units=_non_negative_integer(fields["AliveUnits"], f"{path}.alive_units"),
        current_hit_points=_non_negative_integer(
            fields["CurrentHitPoints"], f"{path}.current_hit_points"
        ),
        maximum_hit_points=_non_negative_integer(
            fields["MaximumHitPoints"], f"{path}.maximum_hit_points"
        ),
        health_adjusted_material=_non_negative_finite_number(
            fields["HealthAdjustedMaterial"], f"{path}.health_adjusted_material"
        ),
        can_damage_enemy=_boolean(fields["CanDamageEnemy"], f"{path}.can_damage_enemy"),
        can_currently_attack_enemy=_boolean(
            fields["CanCurrentlyAttackEnemy"], f"{path}.can_currently_attack_enemy"
        ),
        can_move=_boolean(fields["CanMove"], f"{path}.can_move"),
        units=tuple(sorted(units, key=lambda unit: unit.id)),
    )


def _parse_unit(value: Any, path: str) -> UnitFrame:
    fields = _dto_fields(
        value,
        path,
        (
            "Id", "Q", "R", "CurrentHp", "MaximumHp", "PointCost", "Damage",
            "Defense", "Movement", "VerticalMovement", "Range", "Moved", "Attacked",
        ),
    )
    return UnitFrame(
        id=_non_negative_integer(fields["Id"], f"{path}.id"),
        q=_integer(fields["Q"], f"{path}.q"),
        r=_integer(fields["R"], f"{path}.r"),
        current_hp=_non_negative_integer(fields["CurrentHp"], f"{path}.current_hp"),
        maximum_hp=_non_negative_integer(fields["MaximumHp"], f"{path}.maximum_hp"),
        point_cost=_non_negative_integer(fields["PointCost"], f"{path}.point_cost"),
        damage=_non_negative_integer(fields["Damage"], f"{path}.damage"),
        defense=_non_negative_integer(fields["Defense"], f"{path}.defense"),
        movement=_non_negative_integer(fields["Movement"], f"{path}.movement"),
        vertical_movement=_non_negative_integer(
            fields["VerticalMovement"], f"{path}.vertical_movement"
        ),
        range=_non_negative_integer(fields["Range"], f"{path}.range"),
        moved=_boolean(fields["Moved"], f"{path}.moved"),
        attacked=_boolean(fields["Attacked"], f"{path}.attacked"),
    )


def _parse_command(value: Any, path: str) -> CommandFrame:
    fields = _dto_fields(value, path, ("Kind", "Issuer", "ActorId", "TargetId", "Q", "R"))
    kind = fields["Kind"]
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{path}.kind must be a non-empty string")
    return CommandFrame(
        kind=kind,
        issuer=_seat_number(fields["Issuer"], f"{path}.issuer"),
        actor_id=_optional_non_negative_integer(fields["ActorId"], f"{path}.actor_id"),
        target_id=_optional_non_negative_integer(fields["TargetId"], f"{path}.target_id"),
        q=_optional_integer(fields["Q"], f"{path}.q"),
        r=_optional_integer(fields["R"], f"{path}.r"),
    )


def _dto_fields(value: Any, path: str, pascal_names: tuple[str, ...]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must be a JSON object")
    camel_names = tuple(_camel_case(name) for name in pascal_names)
    keys = set(value)
    if keys == set(pascal_names):
        return {name: value[name] for name in pascal_names}
    if keys == set(camel_names):
        return {name: value[_camel_case(name)] for name in pascal_names}
    for pascal, camel in zip(pascal_names, camel_names):
        snake = "".join(
            ("_" if index else "") + character.lower()
            if character.isupper() else character
            for index, character in enumerate(pascal)
        )
        if snake != camel and snake in value:
            raise ValueError(f"{path}.{snake} must use a transport casing alias")
        if pascal not in value and camel not in value:
            raise ValueError(f"{path}.{camel} is required")
    raise ValueError(f"{path} must use only PascalCase or camelCase DTO fields")


def _camel_case(name: str) -> str:
    return name[:1].lower() + name[1:]


def _integer(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _non_negative_integer(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _optional_integer(value: Any, path: str) -> int | None:
    return None if value is None else _integer(value, path)


def _optional_non_negative_integer(value: Any, path: str) -> int | None:
    return None if value is None else _non_negative_integer(value, path)


def _seat_number(value: Any, path: str) -> int:
    result = _integer(value, path)
    if result not in {0, 1}:
        raise ValueError(f"{path} must be 0 or 1")
    return result


def _optional_seat(value: Any, path: str) -> int | None:
    return None if value is None else _seat_number(value, path)


def _boolean(value: Any, path: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{path} must be a boolean")
    return value


def _non_negative_finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{path} must be finite")
    if result < 0:
        raise ValueError(f"{path} must be non-negative")
    return result


def _canonical_trace_dict(trace: EpisodeTrace) -> dict[str, Any]:
    return {
        "schema_version": trace.schema_version,
        "transitions": [_transition_dict(transition) for transition in trace.transitions],
    }


def _transition_dict(transition: TransitionFrame) -> dict[str, Any]:
    return {
        "before": _state_dict(transition.before),
        "command": {
            "kind": transition.command.kind,
            "issuer": transition.command.issuer,
            "actor_id": transition.command.actor_id,
            "target_id": transition.command.target_id,
            "q": transition.command.q,
            "r": transition.command.r,
        },
        "after": _state_dict(transition.after),
    }


def _state_dict(state: StateFrame) -> dict[str, Any]:
    return {
        "round": state.round,
        "active_seat": state.active_seat,
        "is_game_over": state.is_game_over,
        "winner": state.winner,
        "productive_legal_actions": state.productive_legal_actions,
        "seats": [_seat_dict(seat) for seat in state.seats],
    }


def _seat_dict(seat: SeatFrame) -> dict[str, Any]:
    return {
        "seat": seat.seat,
        "points": seat.points,
        "destroyed_value": seat.destroyed_value,
        "alive_units": seat.alive_units,
        "current_hit_points": seat.current_hit_points,
        "maximum_hit_points": seat.maximum_hit_points,
        "health_adjusted_material": seat.health_adjusted_material,
        "can_damage_enemy": seat.can_damage_enemy,
        "can_currently_attack_enemy": seat.can_currently_attack_enemy,
        "can_move": seat.can_move,
        "units": [_unit_dict(unit) for unit in seat.units],
    }


def _unit_dict(unit: UnitFrame) -> dict[str, Any]:
    return {
        "id": unit.id,
        "q": unit.q,
        "r": unit.r,
        "current_hp": unit.current_hp,
        "maximum_hp": unit.maximum_hp,
        "point_cost": unit.point_cost,
        "damage": unit.damage,
        "defense": unit.defense,
        "movement": unit.movement,
        "vertical_movement": unit.vertical_movement,
        "range": unit.range,
        "moved": unit.moved,
        "attacked": unit.attacked,
    }
