from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import pytest

from ml_lab.draw_classification import (
    DrawCategory,
    classify_draw,
    summarize_episode,
)
from ml_lab.tactical_trace import (
    CommandFrame,
    EpisodeTrace,
    SeatFrame,
    StateFrame,
    TransitionFrame,
    UnitFrame,
)


def _units(*, seat: int, alive: int, hit_points: int, marker: int) -> tuple[UnitFrame, ...]:
    if alive == 0:
        return ()
    quotient, remainder = divmod(hit_points, alive)
    return tuple(
        UnitFrame(
            id=seat * 100 + 10 + index,
            q=seat * 6 + marker + index,
            r=seat,
            current_hp=quotient + (1 if index < remainder else 0),
            maximum_hp=10,
            point_cost=10,
            damage=3,
            defense=1,
            movement=3,
            vertical_movement=1,
            range=2,
            moved=False,
            attacked=False,
        )
        for index in range(alive)
    )


def _state(
    *,
    round_number: int,
    material: tuple[float, float] = (10.0, 10.0),
    hit_points: tuple[int, int] = (10, 10),
    alive: tuple[int, int] = (1, 1),
    can_damage: tuple[bool, bool] = (True, True),
    can_attack: tuple[bool, bool] = (True, True),
    can_move: tuple[bool, bool] = (True, True),
    active_seat: int = 0,
    productive_actions: int = 0,
    marker: int = 0,
) -> StateFrame:
    seats = tuple(
        SeatFrame(
            seat=seat,
            points=0,
            destroyed_value=0,
            alive_units=alive[seat],
            current_hit_points=hit_points[seat],
            maximum_hit_points=max(hit_points[seat], alive[seat] * 10),
            health_adjusted_material=material[seat],
            can_damage_enemy=can_damage[seat],
            can_currently_attack_enemy=can_attack[seat],
            can_move=can_move[seat],
            units=_units(
                seat=seat,
                alive=alive[seat],
                hit_points=hit_points[seat],
                marker=marker,
            ),
        )
        for seat in (0, 1)
    )
    return StateFrame(
        round=round_number,
        active_seat=active_seat,
        is_game_over=False,
        winner=None,
        productive_legal_actions=productive_actions,
        seats=seats,  # type: ignore[arg-type]
    )


def _trace(
    states: tuple[StateFrame, ...],
    commands: tuple[tuple[str, int], ...],
) -> EpisodeTrace:
    assert len(states) == len(commands) + 1
    return EpisodeTrace(
        schema_version=1,
        transitions=tuple(
            TransitionFrame(
                before=states[index],
                command=CommandFrame(
                    kind=kind,
                    issuer=issuer,
                    actor_id=None,
                    target_id=None,
                    q=None,
                    r=None,
                ),
                after=states[index + 1],
            )
            for index, (kind, issuer) in enumerate(commands)
        ),
    )


def _classify(trace: EpisodeTrace, **overrides: object):
    arguments: dict[str, object] = {
        "candidate_seat": 0,
        "terminated": True,
        "truncated": False,
        "winner": None,
    }
    arguments.update(overrides)
    return classify_draw(trace, **arguments)  # type: ignore[arg-type]


def test_episode_summary_attributes_metrics_to_command_issuer() -> None:
    trace = _trace(
        (
            _state(round_number=1, material=(10.0, 10.0), hit_points=(10, 10), alive=(2, 2)),
            _state(
                round_number=1,
                material=(10.0, 7.0),
                hit_points=(10, 7),
                alive=(2, 1),
                productive_actions=2,
            ),
            _state(
                round_number=2,
                material=(8.0, 7.0),
                hit_points=(8, 7),
                alive=(2, 1),
                productive_actions=1,
            ),
            _state(
                round_number=2,
                material=(8.0, 7.0),
                hit_points=(8, 7),
                alive=(2, 1),
                active_seat=1,
            ),
        ),
        (("attack", 0), ("end_turn", 1), ("end_turn", 0)),
    )

    summary = summarize_episode(trace, candidate_seat=0)

    assert summary.command_count == 3
    assert summary.round_count == 2
    assert summary.damage_by_seat == (3, 2)
    assert summary.kills_by_seat == (1, 0)
    assert summary.end_turns_by_seat == (1, 1)
    assert summary.wasted_end_turns_by_seat == (1, 1)
    assert summary.peak_normalized_advantage == pytest.approx(0.15)
    assert summary.final_normalized_advantage == pytest.approx(0.05)
    assert summary.maximum_state_repetition == 1


def test_zero_initial_material_uses_one_as_normalization_denominator() -> None:
    trace = _trace(
        (_state(round_number=1, material=(0.0, 0.0)), _state(round_number=1, material=(0.4, 0.0))),
        (("move", 0),),
    )

    summary = summarize_episode(trace, candidate_seat=0)

    assert summary.peak_normalized_advantage == pytest.approx(0.4)


def test_invalid_initial_roster_is_invalid_scenario() -> None:
    trace = _trace(
        (
            _state(round_number=1, material=(0.0, 10.0), hit_points=(0, 10), alive=(0, 1)),
            _state(round_number=1, material=(0.0, 10.0), hit_points=(0, 10), alive=(0, 1), marker=1),
        ),
        (("move", 1),),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.INVALID_SCENARIO
    assert DrawCategory.INVALID_SCENARIO in result.flags
    assert DrawCategory.AVOIDANCE in result.flags


def test_time_limit_without_terminal_draw_is_truncation() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(round_number=1, material=(10.0, 8.0), hit_points=(10, 8), marker=1),
        ),
        (("attack", 0),),
    )

    result = _classify(trace, terminated=False, truncated=True)

    assert result.primary == DrawCategory.TRUNCATION
    assert DrawCategory.TRUNCATION in result.flags


def test_terminal_draw_suppresses_truncation_flag() -> None:
    trace = _trace(
        (_state(round_number=1), _state(round_number=1, material=(10.0, 8.0), hit_points=(10, 8), marker=1)),
        (("attack", 0),),
    )

    result = _classify(trace, terminated=True, truncated=True, winner=None)

    assert DrawCategory.TRUNCATION not in result.flags


@pytest.mark.parametrize(
    ("candidate_seat", "peak_material"),
    [(0, (17.0, 10.0)), (1, (10.0, 17.0))],
)
def test_decisive_peak_is_failed_conversion_for_either_seat(
    candidate_seat: int, peak_material: tuple[float, float]
) -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(round_number=1, material=peak_material, hit_points=(10, 8), marker=1),
        ),
        (("attack", candidate_seat),),
    )

    result = _classify(trace, candidate_seat=candidate_seat)

    assert result.primary == DrawCategory.FAILED_CONVERSION
    assert result.evidence["peak_normalized_advantage"] == pytest.approx(0.35)


def test_survivors_without_damage_capability_are_damage_stalemate() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(
                round_number=1,
                material=(10.0, 8.0),
                hit_points=(10, 8),
                can_damage=(False, True),
                marker=1,
            ),
        ),
        (("attack", 0),),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.DAMAGE_STALEMATE


def test_damage_capable_immobile_non_attackers_are_mobility_stalemate() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(
                round_number=1,
                material=(10.0, 8.0),
                hit_points=(10, 8),
                can_attack=(False, False),
                can_move=(False, False),
                marker=1,
            ),
        ),
        (("attack", 0),),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.MOBILITY_STALEMATE


def test_round_independent_state_repetition_is_cycling() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(round_number=2),
            _state(round_number=3),
        ),
        (("move", 0), ("move", 0)),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.CYCLING
    assert result.evidence["maximum_state_repetition"] == 3


def test_candidate_wasted_end_turn_count_and_ratio_are_action_waste() -> None:
    states = tuple(
        _state(round_number=index + 1, productive_actions=2, marker=index)
        for index in range(5)
    )
    trace = _trace(states, tuple(("end_turn", 0) for _ in range(4)))

    result = _classify(trace)

    assert result.primary == DrawCategory.ACTION_WASTE
    assert result.evidence["candidate_wasted_end_turns"] == 4
    assert result.evidence["candidate_wasted_end_turn_ratio"] == pytest.approx(1.0)


def test_zero_damage_episode_is_avoidance() -> None:
    trace = _trace(
        (_state(round_number=1), _state(round_number=1, marker=1)),
        (("move", 0),),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.AVOIDANCE


def test_mutual_damage_with_small_final_gap_is_balanced_attrition() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(round_number=1, material=(10.0, 8.0), hit_points=(10, 8), marker=1),
            _state(round_number=2, material=(8.0, 8.0), hit_points=(8, 8), marker=2),
        ),
        (("attack", 0), ("attack", 1)),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.BALANCED_ATTRITION
    assert result.evidence["final_normalized_advantage"] == pytest.approx(0.0)


def test_failed_conversion_precedes_cycling_but_preserves_cycle_flag() -> None:
    repeated_peak = tuple(
        _state(round_number=round_number, material=(18.0, 10.0))
        for round_number in (2, 3, 4)
    )
    trace = _trace(
        (_state(round_number=1), *repeated_peak),
        (("attack", 0), ("move", 0), ("move", 0)),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.FAILED_CONVERSION
    assert DrawCategory.FAILED_CONVERSION in result.flags
    assert DrawCategory.CYCLING in result.flags
    assert result.flags.index(DrawCategory.FAILED_CONVERSION) < result.flags.index(DrawCategory.CYCLING)


def test_distance_does_not_substitute_for_recorded_mobility_evidence() -> None:
    trace = _trace(
        (
            _state(round_number=1),
            _state(
                round_number=1,
                material=(10.0, 8.0),
                hit_points=(10, 8),
                can_attack=(False, False),
                can_move=(True, True),
                marker=20,
            ),
        ),
        (("attack", 0),),
    )

    result = _classify(trace)

    assert result.primary == DrawCategory.UNCLASSIFIED
    assert result.flags == ()


def test_empty_trace_is_rejected_by_summary_and_classifier() -> None:
    trace = EpisodeTrace(schema_version=1, transitions=())

    with pytest.raises(ValueError, match="empty trace"):
        summarize_episode(trace, candidate_seat=0)
    with pytest.raises(ValueError, match="empty trace"):
        _classify(trace)


def test_classification_is_immutable_deterministic_and_numeric() -> None:
    trace = _trace(
        (_state(round_number=1), _state(round_number=1, marker=1)),
        (("move", 0),),
    )

    first = _classify(trace)
    second = _classify(trace)

    assert first == second
    assert json.loads(json.dumps(dict(first.evidence)))["command_count"] == 1
    assert all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in first.evidence.values())
    with pytest.raises(FrozenInstanceError):
        first.primary = DrawCategory.UNCLASSIFIED  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.evidence["command_count"] = 99  # type: ignore[index]
