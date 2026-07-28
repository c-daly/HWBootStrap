"""Deterministic, descriptive evidence classification for TacticalV2 draws."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import TypeAlias

from ml_lab.tactical_trace import EpisodeTrace, StateFrame


Numeric: TypeAlias = int | float


class DrawCategory(str, Enum):
    INVALID_SCENARIO = "invalid_scenario"
    TRUNCATION = "truncation"
    FAILED_CONVERSION = "failed_conversion"
    DAMAGE_STALEMATE = "damage_stalemate"
    MOBILITY_STALEMATE = "mobility_stalemate"
    CYCLING = "cycling"
    ACTION_WASTE = "action_waste"
    AVOIDANCE = "avoidance"
    BALANCED_ATTRITION = "balanced_attrition"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class DrawThresholds:
    decisive_advantage: float = 0.35
    balanced_advantage: float = 0.10
    cycle_repetitions: int = 3
    wasted_end_turns: int = 3
    wasted_end_turn_ratio: float = 0.25


@dataclass(frozen=True)
class EpisodeSummary:
    command_count: int
    round_count: int
    damage_by_seat: tuple[int, int]
    kills_by_seat: tuple[int, int]
    end_turns_by_seat: tuple[int, int]
    wasted_end_turns_by_seat: tuple[int, int]
    peak_normalized_advantage: float
    final_normalized_advantage: float
    maximum_state_repetition: int


@dataclass(frozen=True)
class DrawClassification:
    primary: DrawCategory
    flags: tuple[DrawCategory, ...]
    evidence: Mapping[str, Numeric]


_PRECEDENCE = (
    DrawCategory.INVALID_SCENARIO,
    DrawCategory.TRUNCATION,
    DrawCategory.FAILED_CONVERSION,
    DrawCategory.DAMAGE_STALEMATE,
    DrawCategory.MOBILITY_STALEMATE,
    DrawCategory.CYCLING,
    DrawCategory.ACTION_WASTE,
    DrawCategory.AVOIDANCE,
    DrawCategory.BALANCED_ATTRITION,
)


def summarize_episode(trace: EpisodeTrace, candidate_seat: int) -> EpisodeSummary:
    """Summarize factual episode evidence from a non-empty immutable trace."""
    _require_candidate_seat(candidate_seat)
    states = _episode_states(trace)
    denominator = max(
        sum(seat.health_adjusted_material for seat in states[0].seats),
        1.0,
    )
    advantages = tuple(
        (
            state.seats[candidate_seat].health_adjusted_material
            - state.seats[1 - candidate_seat].health_adjusted_material
        )
        / denominator
        for state in states
    )

    damage = [0, 0]
    kills = [0, 0]
    end_turns = [0, 0]
    wasted_end_turns = [0, 0]
    for transition in trace.transitions:
        issuer = transition.command.issuer
        opponent = 1 - issuer
        damage[issuer] += max(
            transition.before.seats[opponent].current_hit_points
            - transition.after.seats[opponent].current_hit_points,
            0,
        )
        kills[issuer] += max(
            transition.before.seats[opponent].alive_units
            - transition.after.seats[opponent].alive_units,
            0,
        )
        if transition.command.kind == "end_turn":
            end_turns[issuer] += 1
            if transition.before.productive_legal_actions > 0:
                wasted_end_turns[issuer] += 1

    repetitions = Counter(_cycle_key(state) for state in states)
    return EpisodeSummary(
        command_count=len(trace.transitions),
        round_count=len({state.round for state in states}),
        damage_by_seat=(damage[0], damage[1]),
        kills_by_seat=(kills[0], kills[1]),
        end_turns_by_seat=(end_turns[0], end_turns[1]),
        wasted_end_turns_by_seat=(wasted_end_turns[0], wasted_end_turns[1]),
        peak_normalized_advantage=max(advantages),
        final_normalized_advantage=advantages[-1],
        maximum_state_repetition=max(repetitions.values()),
    )


def classify_draw(
    trace: EpisodeTrace,
    *,
    candidate_seat: int,
    terminated: bool,
    truncated: bool,
    winner: int | None,
    thresholds: DrawThresholds = DrawThresholds(),
) -> DrawClassification:
    """Classify independently supported evidence and select its ordered primary label."""
    summary = summarize_episode(trace, candidate_seat)
    initial = trace.transitions[0].before
    final = trace.transitions[-1].after
    candidate_end_turns = summary.end_turns_by_seat[candidate_seat]
    candidate_wasted_end_turns = summary.wasted_end_turns_by_seat[candidate_seat]
    candidate_wasted_ratio = (
        candidate_wasted_end_turns / candidate_end_turns
        if candidate_end_turns
        else 0.0
    )

    supported: set[DrawCategory] = set()
    if any(seat.alive_units == 0 for seat in initial.seats):
        supported.add(DrawCategory.INVALID_SCENARIO)
    terminal_draw = terminated and winner is None
    if truncated and not terminal_draw:
        supported.add(DrawCategory.TRUNCATION)
    if summary.peak_normalized_advantage >= thresholds.decisive_advantage:
        supported.add(DrawCategory.FAILED_CONVERSION)
    if all(seat.alive_units > 0 for seat in final.seats) and any(
        not seat.can_damage_enemy for seat in final.seats
    ):
        supported.add(DrawCategory.DAMAGE_STALEMATE)
    if (
        all(seat.alive_units > 0 and seat.can_damage_enemy for seat in final.seats)
        and all(not seat.can_move for seat in final.seats)
        and all(not seat.can_currently_attack_enemy for seat in final.seats)
    ):
        supported.add(DrawCategory.MOBILITY_STALEMATE)
    if summary.maximum_state_repetition >= thresholds.cycle_repetitions:
        supported.add(DrawCategory.CYCLING)
    if (
        candidate_wasted_end_turns >= thresholds.wasted_end_turns
        and candidate_wasted_ratio >= thresholds.wasted_end_turn_ratio
    ):
        supported.add(DrawCategory.ACTION_WASTE)
    if sum(summary.damage_by_seat) == 0:
        supported.add(DrawCategory.AVOIDANCE)
    if (
        all(damage > 0 for damage in summary.damage_by_seat)
        and abs(summary.final_normalized_advantage) <= thresholds.balanced_advantage
    ):
        supported.add(DrawCategory.BALANCED_ATTRITION)

    flags = tuple(category for category in _PRECEDENCE if category in supported)
    primary = flags[0] if flags else DrawCategory.UNCLASSIFIED
    return DrawClassification(
        primary=primary,
        flags=flags,
        evidence=MappingProxyType(
            _numeric_evidence(
                summary=summary,
                initial=initial,
                final=final,
                candidate_seat=candidate_seat,
                candidate_wasted_ratio=candidate_wasted_ratio,
                terminated=terminated,
                truncated=truncated,
                terminal_draw=terminal_draw,
            )
        ),
    )


def _episode_states(trace: EpisodeTrace) -> tuple[StateFrame, ...]:
    if not trace.transitions:
        raise ValueError("cannot summarize an empty trace")
    return (trace.transitions[0].before,) + tuple(
        transition.after for transition in trace.transitions
    )


def _require_candidate_seat(candidate_seat: int) -> None:
    if isinstance(candidate_seat, bool) or candidate_seat not in (0, 1):
        raise ValueError("candidate_seat must be 0 or 1")


def _cycle_key(state: StateFrame) -> tuple[object, ...]:
    living_units = tuple(
        sorted(
            (
                seat.seat,
                unit.id,
                unit.q,
                unit.r,
                unit.current_hp,
                unit.moved,
                unit.attacked,
            )
            for seat in state.seats
            for unit in seat.units
            if unit.current_hp > 0
        )
    )
    return (
        state.active_seat,
        tuple(seat.points for seat in state.seats),
        living_units,
    )


def _numeric_evidence(
    *,
    summary: EpisodeSummary,
    initial: StateFrame,
    final: StateFrame,
    candidate_seat: int,
    candidate_wasted_ratio: float,
    terminated: bool,
    truncated: bool,
    terminal_draw: bool,
) -> dict[str, Numeric]:
    return {
        "command_count": summary.command_count,
        "round_count": summary.round_count,
        "damage_seat_0": summary.damage_by_seat[0],
        "damage_seat_1": summary.damage_by_seat[1],
        "total_damage": sum(summary.damage_by_seat),
        "kills_seat_0": summary.kills_by_seat[0],
        "kills_seat_1": summary.kills_by_seat[1],
        "end_turns_seat_0": summary.end_turns_by_seat[0],
        "end_turns_seat_1": summary.end_turns_by_seat[1],
        "wasted_end_turns_seat_0": summary.wasted_end_turns_by_seat[0],
        "wasted_end_turns_seat_1": summary.wasted_end_turns_by_seat[1],
        "candidate_end_turns": summary.end_turns_by_seat[candidate_seat],
        "candidate_wasted_end_turns": summary.wasted_end_turns_by_seat[candidate_seat],
        "candidate_wasted_end_turn_ratio": candidate_wasted_ratio,
        "peak_normalized_advantage": summary.peak_normalized_advantage,
        "final_normalized_advantage": summary.final_normalized_advantage,
        "maximum_state_repetition": summary.maximum_state_repetition,
        "initial_alive_units_seat_0": initial.seats[0].alive_units,
        "initial_alive_units_seat_1": initial.seats[1].alive_units,
        "final_alive_units_seat_0": final.seats[0].alive_units,
        "final_alive_units_seat_1": final.seats[1].alive_units,
        "final_can_damage_enemy_seat_0": int(final.seats[0].can_damage_enemy),
        "final_can_damage_enemy_seat_1": int(final.seats[1].can_damage_enemy),
        "final_can_currently_attack_enemy_seat_0": int(
            final.seats[0].can_currently_attack_enemy
        ),
        "final_can_currently_attack_enemy_seat_1": int(
            final.seats[1].can_currently_attack_enemy
        ),
        "final_can_move_seat_0": int(final.seats[0].can_move),
        "final_can_move_seat_1": int(final.seats[1].can_move),
        "terminated": int(terminated),
        "truncated": int(truncated),
        "terminal_draw": int(terminal_draw),
    }
