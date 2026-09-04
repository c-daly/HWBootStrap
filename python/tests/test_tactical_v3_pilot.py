from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from ml_lab.tactical_v3_client import OracleStepResult, TeacherSelection
from ml_lab.tactical_v3_schema import parse_spaces, parse_view
from tests.test_tactical_v3_schema import minimal_view_payload


ROOT = Path(__file__).resolve().parents[2]
DUEL_SPACES = ROOT / "python" / "tests" / "fixtures" / "tactical_v3" / "seed-41-duel-spaces.json"


def _identity():
    return parse_spaces(json.loads(DUEL_SPACES.read_text(encoding="utf-8")))


def _transfer_identity():
    identity = _identity()
    match = dict(identity.match)
    match["max_steps"] = int(match["max_steps"]) + 8
    return replace(
        identity,
        scenario_id="compatible-transfer-target",
        contract_hash="f" * 64,
        match=MappingProxyType(match),
    )


def _view(decision_id: int, *, seat: int = 0, profile: str = "standard-3v3",
          reference_seat: int = 0, terminal: bool = False, truncated: bool = False):
    payload = minimal_view_payload()
    payload["decision_id"] = decision_id
    payload["seat"] = seat
    payload["start_profile"] = profile
    payload["reference_seat"] = reference_seat
    payload["candidates"][0]["decision_id"] = decision_id
    if terminal or truncated:
        payload["candidates"] = []
        payload["terminated"] = terminal
        payload["truncated"] = truncated
        payload["winner"] = seat if terminal else -1
        payload["reward"]["finalized"] = True
    return parse_view(payload, _identity())


def _view_with_two_candidates(
    decision_id: int,
    *,
    seat: int = 0,
    reference_seat: int = 0,
    profile: str = "standard-3v3",
    terminal: bool = False,
):
    payload = minimal_view_payload()
    payload["decision_id"] = decision_id
    payload["seat"] = seat
    payload["reference_seat"] = reference_seat
    payload["start_profile"] = profile
    payload["candidates"][0]["decision_id"] = decision_id
    if terminal:
        payload["candidates"] = []
        payload["terminated"] = True
        payload["winner"] = 0
        payload["reward"]["finalized"] = True
    else:
        second = dict(payload["candidates"][0])
        second["candidate_id"] = 1
        payload["candidates"].append(second)
    return parse_view(payload, _identity())


class _FakeClient:
    def __init__(self, views, *, fallback: int = 0, identity_drift: bool = False) -> None:
        self._views = tuple(views)
        self._index = 0
        self._identity = _identity()
        self._identity_reads = 0
        self._identity_drift = identity_drift
        self._fallback = fallback
        self.reset_args = None

    @property
    def identity(self):
        self._identity_reads += 1
        if self._identity_drift and self._identity_reads > 1:
            return replace(self._identity, scenario_id="drift")
        return self._identity

    def duel_reset(self, *args):
        self.reset_args = args
        return self._views[0]

    def duel_oracle_step(self, decision_id: int) -> OracleStepResult:
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        self._index += 1
        return OracleStepResult(
            TeacherSelection(decision_id, current.decision.candidates[0].candidate_id,
                             4, 512, 11 + self._index, "material-plus-pursuit-v1"),
            self._views[self._index],
        )

    def duel_greedy_step(self, decision_id: int) -> OracleStepResult:
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        self._index += 1
        return OracleStepResult(
            TeacherSelection(decision_id, current.decision.candidates[0].candidate_id,
                             0, 0, 0, "greedy-one-ply-v1"),
            self._views[self._index],
        )

    def duel_status(self) -> int:
        return self._fallback


class _DaggerClient:
    def __init__(self, seat: int = 0, profile: str = "standard-3v3") -> None:
        self._identity = _identity()
        self._views = (
            _view_with_two_candidates(
                7, seat=seat, reference_seat=seat, profile=profile,
            ),
            _view_with_two_candidates(
                8, seat=seat, reference_seat=seat, profile=profile,
            ),
            _view_with_two_candidates(
                9, seat=seat, reference_seat=seat, profile=profile, terminal=True,
            ),
        )
        self._index = 0
        self.events = []
        self.oracle_budgets = []

    @property
    def identity(self):
        return self._identity

    def duel_reset(self, *args):
        self.events.append(("reset", args))
        return self._views[0]

    def duel_oracle_query(self, decision_id, *, expansion_budget=512):
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        teacher_candidate = 1 if self._index == 0 else 0
        self.events.append(("query", decision_id, teacher_candidate))
        self.oracle_budgets.append(expansion_budget)
        return TeacherSelection(
            decision_id, teacher_candidate, 4, expansion_budget, 21 + self._index,
            "material-plus-pursuit-v1",
        )

    def duel_dagger_inspect(self, decision_id, learner_candidate_id):
        from ml_lab.tactical_v3_client import SelectiveDaggerInspection

        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        self.events.append(("inspect", decision_id, learner_candidate_id))
        return SelectiveDaggerInspection(
            decision_id, learner_candidate_id, ("favorable",),
            ("e" if self._index == 0 else "f") * 64,
            1, 0.25, 1, 2,
        )

    def duel_step(self, selection):
        current = self._views[self._index]
        assert selection.decision_id == current.decision.decision_id
        self.events.append(("step", selection.decision_id, selection.candidate_id))
        self._index += 1
        return self._views[self._index]

    def duel_status(self):
        return 0


class _SelectiveDaggerClient(_DaggerClient):
    def duel_dagger_inspect(self, decision_id, learner_candidate_id):
        from ml_lab.tactical_v3_client import SelectiveDaggerInspection

        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        reasons = () if self._index == 0 else ("favorable",)
        self.events.append(("inspect", decision_id, learner_candidate_id, reasons))
        return SelectiveDaggerInspection(
            decision_id, learner_candidate_id, reasons,
            ("c" if self._index == 0 else "d") * 64,
            1, -0.25 if self._index == 0 else 0.25,
            3 if self._index == 0 else 1,
            2,
        )


class _PreflightClient:
    def __init__(self, item) -> None:
        self._views = (
            _view(7, seat=item.learner_seat, profile=item.profile_id,
                  reference_seat=item.reference_seat),
            _view(8, seat=item.learner_seat, profile=item.profile_id,
                  reference_seat=item.reference_seat),
            _view(9, seat=item.learner_seat, profile=item.profile_id,
                  reference_seat=item.reference_seat, truncated=True),
        )
        self._index = 0
        self.events = []

    def duel_reset(self, *args):
        self.events.append(("reset", args))
        return self._views[0]

    def duel_oracle_query(self, decision_id, *, expansion_budget):
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        self.events.append(("query", decision_id, expansion_budget))
        return TeacherSelection(
            decision_id, current.decision.candidates[0].candidate_id,
            4, expansion_budget, 17, "material-plus-pursuit-v1",
        )

    def duel_dagger_inspect(self, decision_id, candidate_id):
        from ml_lab.tactical_v3_client import SelectiveDaggerInspection

        occurrence = 1 if self._index == 0 else 3
        self.events.append(("inspect", decision_id, candidate_id))
        return SelectiveDaggerInspection(
            decision_id, candidate_id, (),
            ("a" if self._index == 0 else "b") * 64,
            occurrence, 0.0, 1, 1,
        )

    def duel_step(self, selection):
        self.events.append(("step", selection.decision_id, selection.candidate_id))
        self._index += 1
        return self._views[self._index]

    def duel_status(self):
        self.events.append(("status",))
        return 0


def _canonical_collection():
    from ml_lab.tactical_v3_pilot import PilotCollection, collect_game, collection_schedule

    partitions = []
    games = []
    for partition in ("train", "validation"):
        examples = []
        for item in collection_schedule(partition):
            rows, game = collect_game(_FakeClient([
                _view(7, seat=item.learner_seat, profile=item.profile_id,
                      reference_seat=item.reference_seat),
                _view(8, seat=item.learner_seat, profile=item.profile_id,
                      reference_seat=item.reference_seat,
                      terminal=partition == "train", truncated=partition == "validation"),
            ]), item)
            examples.extend(rows)
            games.append(game)
        partitions.append(tuple(examples))
    return PilotCollection(_identity(), partitions[0], partitions[1], tuple(games))


def test_pilot_schedules_are_frozen_balanced_disjoint_and_exact() -> None:
    from ml_lab.tactical_v3_pilot import (
        PILOT_PROFILES,
        TACTICAL_V3_START_PROFILES,
        ContinuationScheduleItem,
        PilotScheduleItem,
        collection_schedule,
        evaluation_schedule,
    )

    assert PILOT_PROFILES == (
        "standard-3v3", "conversion-3v1-near", "conversion-3v1-far",
        "conversion-2v1-near", "conversion-2v1-far",
        "conversion-1v1-near", "conversion-1v1-far",
    )
    assert TACTICAL_V3_START_PROFILES == (
        "standard-3v3",
        "conversion-3v1-near", "conversion-3v1-medium", "conversion-3v1-far",
        "conversion-2v1-near", "conversion-2v1-medium", "conversion-2v1-far",
        "conversion-1v1-near", "conversion-1v1-medium", "conversion-1v1-far",
    )
    with pytest.raises(ValueError, match="pilot profile"):
        PilotScheduleItem("train", "conversion-2v1-medium", 1, 0, 0)
    assert ContinuationScheduleItem(
        "train", "conversion-2v1-medium", 1, 0, 0,
    ).profile_id == "conversion-2v1-medium"
    train = collection_schedule("train")
    validation = collection_schedule("validation")
    evaluation = evaluation_schedule()
    assert (len(train), len(validation), len(evaluation)) == (28, 14, 28)
    assert [(row.profile_id, row.episode_seed, row.learner_seat) for row in train[:4]] == [
        (PILOT_PROFILES[0], 61_000_000, 0),
        (PILOT_PROFILES[0], 61_000_000, 1),
        (PILOT_PROFILES[0], 61_000_001, 0),
        (PILOT_PROFILES[0], 61_000_001, 1),
    ]
    assert {row.episode_seed for row in train}.isdisjoint(
        {row.episode_seed for row in validation} | {row.episode_seed for row in evaluation}
    )
    for schedule in (train, validation, evaluation):
        assert all(row.learner_seat == row.reference_seat for row in schedule)
        grouped = {(row.profile_id, row.episode_seed): [] for row in schedule}
        for row in schedule: grouped[(row.profile_id, row.episode_seed)].append(row.learner_seat)
        assert all(seats == [0, 1] for seats in grouped.values())


def test_diagnostic_schedule_uses_five_new_reciprocal_seeds_per_profile() -> None:
    from ml_lab.tactical_v3_pilot import (
        PILOT_PROFILES, collection_schedule, diagnostic_evaluation_schedule,
        evaluation_schedule,
    )

    schedule = diagnostic_evaluation_schedule()
    prior_seeds = {
        item.episode_seed
        for item in (
            *collection_schedule("train"), *collection_schedule("validation"),
            *evaluation_schedule(),
        )
    }

    assert len(schedule) == 70
    assert {item.episode_seed for item in schedule}.isdisjoint(prior_seeds)
    for profile_index, profile in enumerate(PILOT_PROFILES):
        rows = tuple(item for item in schedule if item.profile_id == profile)
        assert len(rows) == 10
        assert {item.episode_seed for item in rows} == {
            64_000_000 + profile_index * 5 + offset for offset in range(5)
        }
        assert [item.learner_seat for item in rows] == [0, 1] * 5


def test_selective_dagger_preflight_and_evaluation_schedules_are_exact() -> None:
    from ml_lab.tactical_v3_pilot import (
        SELECTIVE_DAGGER_CONVERSION_PROFILES,
        oracle_preflight_schedule,
        selective_dagger_evaluation_schedule,
    )

    preflight = oracle_preflight_schedule()
    assert len(preflight) == 240
    assert tuple(dict.fromkeys(item.profile_id for item in preflight)) == (
        SELECTIVE_DAGGER_CONVERSION_PROFILES
    )
    assert [(item.episode_seed, item.learner_seat) for item in preflight[:4]] == [
        (18_900_000, 0), (18_900_000, 1),
        (18_900_001, 0), (18_900_001, 1),
    ]
    assert [(item.episode_seed, item.learner_seat) for item in preflight[-2:]] == [
        (18_900_119, 0), (18_900_119, 1),
    ]
    for profile_index, profile in enumerate(SELECTIVE_DAGGER_CONVERSION_PROFILES):
        rows = tuple(item for item in preflight if item.profile_id == profile)
        assert len(rows) == 40
        assert {item.episode_seed for item in rows} == set(range(
            18_900_000 + profile_index * 20,
            18_900_020 + profile_index * 20,
        ))

    evaluation = selective_dagger_evaluation_schedule()
    assert len(evaluation) == 200
    assert all(item.profile_id == "standard-3v3" for item in evaluation)
    assert [(item.episode_seed, item.learner_seat) for item in evaluation[:4]] == [
        (20_000_000, 0), (20_000_000, 1),
        (20_000_001, 0), (20_000_001, 1),
    ]
    assert [(item.episode_seed, item.learner_seat) for item in evaluation[-2:]] == [
        (20_000_099, 0), (20_000_099, 1),
    ]


def test_oracle_preflight_runs_both_candidates_and_applies_frozen_selection() -> None:
    import ml_lab.tactical_v3_pilot as module

    calls = []

    def run_game(item, budget):
        calls.append((item, budget))
        return module.OraclePreflightGame(
            won=True,
            cycling_draw=False,
            labels=8 if budget == 512 else 12,
            duration_seconds=0.5,
            deterministic_queries=True,
            roundtrip_failures=0,
        )

    result = module.run_oracle_preflight(run_game)

    assert len(calls) == 480
    assert tuple(budget for _, budget in calls[:240]) == (512,) * 240
    assert tuple(budget for _, budget in calls[240:]) == (2048,) * 240
    assert tuple(item for item, _ in calls[:240]) == module.oracle_preflight_schedule()
    assert tuple(item for item, _ in calls[240:]) == module.oracle_preflight_schedule()
    assert result.selected_expansion_budget == 2048
    assert all(summary.games == 240 and summary.win_rate == 1.0
               and summary.labels_per_second >= 10.0 and summary.passed
               for summary in result.candidates)


def test_oracle_preflight_fails_closed_when_no_candidate_meets_thresholds() -> None:
    import ml_lab.tactical_v3_pilot as module

    def run_game(item, budget):
        del item, budget
        return module.OraclePreflightGame(
            won=False, cycling_draw=False, labels=4, duration_seconds=1.0,
            deterministic_queries=True, roundtrip_failures=0,
        )

    with pytest.raises(RuntimeError, match="no oracle candidate"):
        module.run_oracle_preflight(run_game)


def test_physical_oracle_preflight_game_double_queries_and_roundtrips_each_label() -> None:
    import ml_lab.tactical_v3_pilot as module

    item = module.oracle_preflight_schedule()[0]
    client = _PreflightClient(item)
    result = module.run_physical_oracle_preflight_game(client, item, 2048)

    assert result.labels == 2
    assert result.deterministic_queries is True
    assert result.roundtrip_failures == 0
    assert result.won is False
    assert result.cycling_draw is True
    assert client.events == [
        ("reset", (
            item.episode_seed, "external", "random", item.learner_seat,
            item.profile_id, item.reference_seat,
        )),
        ("query", 7, 2048), ("query", 7, 2048),
        ("inspect", 7, 0), ("step", 7, 0),
        ("query", 8, 2048), ("query", 8, 2048),
        ("inspect", 8, 0), ("step", 8, 0),
        ("status",),
    ]


@pytest.mark.parametrize(
    ("partition", "iteration", "expected_start", "expected_games"),
    [
        ("train", 1, 18_000_000, 2_000),
        ("train", 2, 18_100_000, 2_000),
        ("train", 3, 18_200_000, 2_000),
        ("validation", 1, 19_000_000, 200),
        ("validation", 2, 19_010_000, 200),
        ("validation", 3, 19_020_000, 200),
    ],
)
def test_selective_dagger_schedule_uses_frozen_seed_banks_reciprocal_70_30_mix(
    partition: str, iteration: int, expected_start: int, expected_games: int,
) -> None:
    from ml_lab.tactical_v3_pilot import selective_dagger_schedule

    schedule = selective_dagger_schedule(partition, iteration)

    assert len(schedule) == expected_games
    assert schedule[0].episode_seed == expected_start
    assert schedule[-1].episode_seed == expected_start + expected_games // 2 - 1
    assert [item.learner_seat for item in schedule] == [0, 1] * (expected_games // 2)
    pairs = [schedule[index] for index in range(0, len(schedule), 2)]
    assert sum(item.profile_id == "standard-3v3" for item in pairs) == (
        len(pairs) * 7 // 10
    )
    assert [item.profile_id for item in pairs[:10]] == [
        "conversion-3v1-near", "standard-3v3", "standard-3v3",
        "conversion-3v1-far", "standard-3v3", "standard-3v3",
        "conversion-2v1-near", "standard-3v3", "standard-3v3", "standard-3v3",
    ]
    for left, right in zip(schedule[::2], schedule[1::2], strict=True):
        assert (left.episode_seed, left.profile_id) == (
            right.episode_seed, right.profile_id,
        )
        assert (left.reference_seat, right.reference_seat) == (0, 1)


@pytest.mark.parametrize(
    ("partition", "labels_per_game", "expected_target"),
    [("train", 10_000, 20_000), ("validation", 1_000, 2_000)],
)
def test_selective_collection_stops_only_after_complete_pair_at_fixed_target(
    partition: str, labels_per_game: int, expected_target: int,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    calls = []

    def collect(item):
        calls.append(item)
        summary = module.PilotDaggerGameSummary(
            item, -1, False, True, labels_per_game, 0, 0, 0,
        )
        return module.PilotDaggerEpisode(
            _identity(), (None,) * labels_per_game, summary,
            "a" * 64, "b" * 64, 3, 0.125,
        )

    result = module.collect_selective_dagger_partition(
        partition, 1, collect,
    )

    assert result.label_target == expected_target
    assert result.label_count == expected_target
    assert result.game_count == 2
    assert len(calls) == 2
    assert calls[0].episode_seed == calls[1].episode_seed
    assert (calls[0].learner_seat, calls[1].learner_seat) == (0, 1)


def test_selective_collection_fails_when_frozen_ceiling_cannot_reach_target() -> None:
    import ml_lab.tactical_v3_pilot as module

    def collect(item):
        summary = module.PilotDaggerGameSummary(item, -1, False, True, 0, 0, 0, 0)
        return module.PilotDaggerEpisode(
            _identity(), (), summary, "a" * 64, "b" * 64, 3, 0.125,
        )

    with pytest.raises(RuntimeError, match="20,000.*2,000"):
        module.collect_selective_dagger_partition("train", 1, collect)


def test_validation_metric_breakdown_reports_each_profile_and_seat() -> None:
    import torch
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_model import TacticalV3Policy

    collection = _canonical_collection()
    model_config, _, _ = module._pilot_configs(227, "cpu")
    model = TacticalV3Policy(model_config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    breakdown = module.validation_metric_breakdown(model, collection.validation)

    assert set(breakdown) == set(module.PILOT_PROFILES)
    for profile in module.PILOT_PROFILES:
        entry = breakdown[profile]
        assert entry["all"]["examples"] == 2
        assert entry["all"]["policy_accuracy"] == pytest.approx(1.0)
        assert set(entry["seats"]) == {"0", "1"}
        for seat in ("0", "1"):
            assert entry["seats"][seat]["examples"] == 1
            assert entry["seats"][seat]["policy_accuracy"] == pytest.approx(1.0)
            assert entry["seats"][seat]["finite_logit_count"] == (
                entry["seats"][seat]["valid_logit_count"]
            )


def test_collect_game_labels_every_teacher_decision_and_backfills_win() -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("train", "standard-3v3", 61_000_000, 0, 0)
    client = _FakeClient([_view(7), _view(8), _view(9, terminal=True)])
    examples, summary = collect_game(client, item)

    assert client.reset_args == (61_000_000, "external", "random", 0, "standard-3v3", 0)
    assert [row.decision.decision_id for row in examples] == [7, 8]
    assert [row.target.trajectory_index for row in examples] == [0, 1]
    assert [row.target.remaining_turns_to_victory for row in examples] == [2, 1]
    assert all(row.target.terminal_outcome == "win" and not row.target.truncated for row in examples)
    assert all(row.teacher.identity == "greedy-one-ply-v1" for row in examples)
    assert all(row.teacher.actual_expansions == 0 for row in examples)
    assert all(row.teacher.search_depth == 0 and row.teacher.expansion_budget == 0
               and row.teacher.confidence is None for row in examples)
    assert summary.decisions == 2 and summary.winner == 0 and summary.internal_fallback_count == 0


def test_collect_game_uses_bounded_search_only_for_conversion_profiles() -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("train", "conversion-1v1-near", 61_000_012, 0, 0)
    client = _FakeClient([
        _view(7, profile=item.profile_id),
        _view(8, profile=item.profile_id, terminal=True),
    ])
    examples, _ = collect_game(client, item)

    assert [row.teacher.identity for row in examples] == ["bounded-search-v1"]
    assert examples[0].teacher.search_depth == 4
    assert examples[0].teacher.expansion_budget == 512


def test_dagger_episode_queries_teacher_then_steps_learner_and_persists_records(
    tmp_path: Path,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collect = getattr(module, "collect_dagger_game", None)
    write = getattr(module, "write_dagger_episode", None)
    assert callable(collect), "tactical-v3 DAgger collector is missing"
    assert callable(write), "tactical-v3 DAgger episode writer is missing"
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 65_000_000, 0, 0,
    )
    client = _DaggerClient()
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(),
            model_state_sha256="a" * 64,
            corpus_sha256="b" * 64,
            best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )

    episode = collect(client, loaded, item)

    assert client.events[1:] == [
        ("inspect", 7, 0), ("query", 7, 1), ("step", 7, 0),
        ("inspect", 8, 0), ("query", 8, 0), ("step", 8, 0),
    ]
    assert episode.summary.decisions == 2
    assert episode.summary.disagreements == 1
    assert episode.summary.teacher_interventions == 0
    assert episode.actor_model_state_sha256 == "a" * 64
    assert episode.actor_corpus_sha256 == "b" * 64
    assert [record.learner_candidate_id for record in episode.records] == [0, 0]
    assert [record.teacher_candidate_id for record in episode.records] == [1, 0]
    assert all(not record.teacher_intervened for record in episode.records)
    assert [record.example.target.teacher_candidate_id
            for record in episode.records] == [1, 0]

    output = tmp_path / "dagger-episode"
    assert write(output, episode) == output
    manifest = json.loads((output / "episode.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (output / "decisions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert manifest["schema_version"] == 1
    assert manifest["kind"] == "tactical-v3-dagger-episode"
    assert manifest["records"]["count"] == 2
    assert manifest["summary"]["disagreements"] == 1
    assert manifest["actor"] == {
        "algorithm": "structured_imitation",
        "best_epoch": 3,
        "best_validation_policy_nll": 0.125,
        "corpus_sha256": "b" * 64,
        "model_state_sha256": "a" * 64,
    }
    assert set(rows[0]) == {
        "disagreement", "example", "learner_candidate_id",
        "teacher_candidate_id", "teacher_intervened", "eligibility_reasons",
        "state_hash", "state_occurrence", "normalized_advantage",
        "opponent_living_unit_count", "productive_legal_action_count",
    }
    assert rows[0]["learner_candidate_id"] == 0
    assert rows[0]["teacher_candidate_id"] == 1
    assert rows[0]["disagreement"] is True
    assert rows[0]["teacher_intervened"] is False


def test_dagger_episode_reader_roundtrips_and_rejects_record_tamper(
    tmp_path: Path,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    load = getattr(module, "load_dagger_episode", None)
    assert callable(load), "the strict DAgger episode reader is missing"
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 65_000_000, 0, 0,
    )
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    expected = module.collect_dagger_game(
        _DaggerClient(), loaded, item, oracle_expansion_budget=2048,
    )
    output = module.write_dagger_episode(tmp_path / "episode", expected)

    assert load(output, _identity(), oracle_expansion_budget=2048) == expected

    records = output / "decisions.jsonl"
    records.write_bytes(records.read_bytes() + b" ")
    with pytest.raises(ValueError, match="records|canonical|hash"):
        load(output, _identity(), oracle_expansion_budget=2048)


def test_dagger_episode_reader_preserves_continuation_medium_profile(
    tmp_path: Path,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    item = module.ContinuationScheduleItem(
        "train", "conversion-1v1-medium", 65_000_001, 0, 0,
    )
    actor = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    episode = module.collect_dagger_game(
        _DaggerClient(profile=item.profile_id), actor, item,
    )
    output = module.write_dagger_episode(tmp_path / "medium", episode)

    loaded = module.load_dagger_episode(
        output,
        _identity(),
        oracle_expansion_budget=512,
        expected_schedule=item,
    )

    assert loaded == episode
    assert type(loaded.summary.schedule) is module.ContinuationScheduleItem


def test_selective_partition_reader_reopens_exact_overlay_and_rejects_manifest_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    load = getattr(module, "load_selective_dagger_partition", None)
    assert callable(load), "the strict selective-DAgger partition reader is missing"
    monkeypatch.setitem(module._SELECTIVE_DAGGER_LABEL_TARGETS, "train", 4)
    schedule = module.selective_dagger_schedule("train", 1)[:2]
    actor = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    episodes = tuple(
        module.collect_dagger_game(
            _DaggerClient(item.learner_seat, item.profile_id), actor, item,
            oracle_expansion_budget=2048,
        )
        for item in schedule
    )
    root = tmp_path / "overlay"
    root.mkdir()
    games = []
    for index, episode in enumerate(episodes):
        game = module.write_dagger_episode(root / f"game-{index:04d}", episode)
        games.append({
            "index": index,
            "schedule": {
                "partition": episode.summary.schedule.partition,
                "profile_id": episode.summary.schedule.profile_id,
                "episode_seed": episode.summary.schedule.episode_seed,
                "learner_seat": episode.summary.schedule.learner_seat,
                "reference_seat": episode.summary.schedule.reference_seat,
            },
            "labels": len(episode.records),
            "episode_sha256": hashlib.sha256(
                (game / "episode.json").read_bytes(),
            ).hexdigest(),
            "records_sha256": hashlib.sha256(
                (game / "decisions.jsonl").read_bytes(),
            ).hexdigest(),
        })
    manifest = {
        "schema_version": 1,
        "kind": "tactical-v3-selective-dagger-overlay",
        "status": "completed",
        "partition": "train",
        "iteration": 1,
        "label_target": 4,
        "label_count": 4,
        "game_ceiling": 2000,
        "game_count": 2,
        "oracle_expansion_budget": 2048,
        "actor_checkpoint_sha256": "c" * 64,
        "actor_model_state_sha256": "a" * 64,
        "actor_corpus_sha256": "b" * 64,
        "repository_commit": "d" * 40,
        "wall_seconds": 1.25,
        "games": games,
    }
    overlay = root / "overlay.json"
    overlay.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )

    actual = load(root, _identity())
    assert actual == module.SelectiveDaggerPartitionCollection(
        "train", 1, episodes, 4, 2, 4, 2000,
    )

    manifest["games"][0]["episode_sha256"] = "e" * 64
    overlay.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )
    with pytest.raises(ValueError, match="episode|manifest|hash"):
        load(root, _identity())


def test_dagger_episode_queries_teacher_only_for_selectively_eligible_states() -> None:
    import ml_lab.tactical_v3_pilot as module

    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    client = _SelectiveDaggerClient()
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 18_990_000, 0, 0,
    )

    episode = module.collect_dagger_game(client, loaded, item)

    assert [event for event in client.events if event[0] == "query"] == [
        ("query", 8, 0),
    ]
    assert [event[:3] for event in client.events if event[0] == "step"] == [
        ("step", 7, 0), ("step", 8, 0),
    ]
    assert episode.summary.decisions == 1
    assert len(episode.records) == 1
    assert episode.records[0].eligibility_reasons == ("favorable",)
    assert episode.records[0].state_hash == "d" * 64
    assert episode.records[0].example.target.trajectory_index == 1


def test_dagger_collection_and_evidence_use_selected_2048_oracle(
    tmp_path: Path,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    client = _DaggerClient()
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 18_990_000, 0, 0,
    )

    episode = module.collect_dagger_game(
        client, loaded, item, oracle_expansion_budget=2048,
    )
    output = module.write_dagger_episode(tmp_path / "episode", episode)
    manifest = json.loads((output / "episode.json").read_text(encoding="utf-8"))

    assert client.oracle_budgets == [2048, 2048]
    assert all(record.example.teacher.expansion_budget == 2048
               for record in episode.records)
    assert manifest["teacher"]["expansion_budget"] == 2048


def test_dagger_collection_uses_selected_greedy_opponent() -> None:
    import ml_lab.tactical_v3_pilot as module

    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    client = _DaggerClient(seat=1)
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 34_540_000, 1, 1,
    )

    episode = module.collect_dagger_game(
        client, loaded, item, opponent="greedy",
    )

    assert client.events[0] == (
        "reset", (34_540_000, "greedy", "external", 1, "standard-3v3", 1),
    )
    assert episode.summary.schedule == item


def test_dagger_collection_uses_selected_passive_opponent() -> None:
    import ml_lab.tactical_v3_pilot as module

    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256="b" * 64, best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    client = _DaggerClient(seat=1)
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 34_540_001, 1, 1,
    )

    episode = module.collect_dagger_game(
        client, loaded, item, opponent="passive",
    )

    assert client.events[0] == (
        "reset", (34_540_001, "passive", "external", 1, "standard-3v3", 1),
    )
    assert episode.summary.schedule == item


def test_dagger_compatible_transfer_is_explicit_and_model_facing_only() -> None:
    import ml_lab.tactical_v3_pilot as module

    source_identity = _identity()
    target_identity = _transfer_identity()
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=source_identity,
            model_state_sha256="a" * 64,
            corpus_sha256="b" * 64,
            best_epoch=3,
            best_validation_policy_nll=0.125,
        ),
    )
    item = module.PilotScheduleItem(
        "train", "standard-3v3", 34_540_000, 0, 0,
    )
    strict_client = _DaggerClient()
    strict_client._identity = target_identity

    with pytest.raises(ValueError, match="policy identity"):
        module.collect_dagger_game(strict_client, loaded, item)

    transfer_client = _DaggerClient()
    transfer_client._identity = target_identity
    episode = module.collect_dagger_game(
        transfer_client,
        loaded,
        item,
        allow_compatible_identity_transfer=True,
    )

    assert episode.identity == target_identity
    assert episode.actor_model_state_sha256 == "a" * 64


@pytest.mark.parametrize(
    ("changed", "message"),
    [
        ({"encoding_hash": "0" * 64}, "encoding hash"),
        ({"capacity_hash": "0" * 64}, "capacity hash"),
        ({"environment_kind": "tactical"}, "duel policy"),
    ],
)
def test_compatible_transfer_rejects_model_interface_drift(
    changed: dict[str, object],
    message: str,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    with pytest.raises(ValueError, match=message):
        module._validate_compatible_transfer_identity(
            replace(_identity(), **changed),
            _transfer_identity(),
            subject="source policy",
        )


def test_pilot_training_transfer_still_requires_the_exact_model_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_checkpoint import LoadedStructuredPolicy
    from ml_lab.tactical_v3_model import TacticalV3Policy

    source_identity = _identity()
    target_identity = _transfer_identity()
    collection = _canonical_collection()
    train = tuple(
        replace(
            row,
            scenario_id=target_identity.scenario_id,
            contract_hash=target_identity.contract_hash,
        )
        for row in collection.train[:1]
    )
    validation = tuple(
        replace(
            row,
            scenario_id=target_identity.scenario_id,
            contract_hash=target_identity.contract_hash,
        )
        for row in collection.validation[:1]
    )
    model_config, _, _ = module._pilot_configs(227, "cpu")

    def loaded(model):
        return LoadedStructuredPolicy(
            model,
            SimpleNamespace(identity=source_identity),
            SimpleNamespace(),
        )

    strict_output = tmp_path / "strict"
    strict_output.mkdir()
    with pytest.raises(ValueError, match="initial policy identity"):
        module._train_pilot_dataset(
            target_identity,
            train,
            validation,
            "a" * 64,
            strict_output,
            227,
            "cpu",
            initial_policy=loaded(TacticalV3Policy(model_config)),
            tensorboard_enabled=False,
        )

    wrong_output = tmp_path / "wrong-config"
    wrong_output.mkdir()
    wrong_config = replace(model_config, hidden_dim=model_config.hidden_dim * 2)
    with pytest.raises(ValueError, match="architecture"):
        module._train_pilot_dataset(
            target_identity,
            train,
            validation,
            "a" * 64,
            wrong_output,
            227,
            "cpu",
            initial_policy=loaded(TacticalV3Policy(wrong_config)),
            tensorboard_enabled=False,
            allow_compatible_identity_transfer=True,
        )

    accepted_output = tmp_path / "accepted"
    accepted_output.mkdir()

    def accepted(*args, **kwargs):
        raise RuntimeError("compatible transfer reached target metrics")

    monkeypatch.setattr(module, "_policy_target_metrics", accepted)
    with pytest.raises(RuntimeError, match="reached target metrics"):
        module._train_pilot_dataset(
            target_identity,
            train,
            validation,
            "a" * 64,
            accepted_output,
            227,
            "cpu",
            initial_policy=loaded(TacticalV3Policy(model_config)),
            tensorboard_enabled=False,
            allow_compatible_identity_transfer=True,
        )


def test_first_dagger_iteration_is_reciprocal_and_only_augments_training() -> None:
    import ml_lab.tactical_v3_pilot as module

    schedule = getattr(module, "dagger_iteration_schedule", None)
    build = getattr(module, "build_dagger_training_set", None)
    assert callable(schedule), "the bounded reciprocal DAgger schedule is missing"
    assert callable(build), "the DAgger training-set builder is missing"
    base = _canonical_collection()
    base_train = base.train
    base_validation = base.validation
    evidence = module.PilotCollectionEvidence("1" * 64, "2" * 64, "3" * 64)
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(),
            model_state_sha256="a" * 64,
            corpus_sha256=evidence.collection_sha256,
            best_epoch=0,
            best_validation_policy_nll=1.25,
        ),
    )
    items = schedule()
    episodes = tuple(
        module.collect_dagger_game(_DaggerClient(item.learner_seat), loaded, item)
        for item in items
    )

    augmented = build(base, evidence, episodes)

    assert [(item.profile_id, item.episode_seed, item.learner_seat)
            for item in items] == [
        ("standard-3v3", 65_100_000, 0),
        ("standard-3v3", 65_100_000, 1),
    ]
    assert base.train is base_train
    assert base.validation is base_validation
    assert augmented.train[:len(base.train)] == base.train
    assert augmented.train[len(base.train):] == tuple(
        record.example for episode in episodes for record in episode.records
    )
    assert augmented.validation is base.validation
    assert augmented.base_collection_sha256 == evidence.collection_sha256
    assert len(augmented.dagger_records_sha256) == 64
    assert len(augmented.corpus_sha256) == 64


def test_selective_dagger_training_sets_are_cumulative_with_heldout_only_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    build = getattr(module, "build_selective_dagger_training_set", None)
    assert callable(build), "cumulative selective-DAgger composition is missing"
    monkeypatch.setattr(
        module, "_SELECTIVE_DAGGER_LABEL_TARGETS",
        {"train": 4, "validation": 4},
    )
    base = _canonical_collection()
    evidence = module.PilotCollectionEvidence("1" * 64, "2" * 64, "3" * 64)

    def partition(name, iteration, actor_corpus):
        loaded = SimpleNamespace(
            model=_EvaluationPolicy(),
            metadata=SimpleNamespace(
                identity=_identity(), model_state_sha256=str(iteration) * 64,
                corpus_sha256=actor_corpus, best_epoch=iteration,
                best_validation_policy_nll=0.125,
            ),
        )
        items = module.selective_dagger_schedule(name, iteration)[:2]
        episodes = tuple(
                module.collect_dagger_game(
                    _DaggerClient(item.learner_seat, item.profile_id), loaded, item,
                )
            for item in items
        )
        return module.SelectiveDaggerPartitionCollection(
            name, iteration, episodes, 4, 2, 4,
            len(module.selective_dagger_schedule(name, iteration)),
        )

    train1 = partition("train", 1, evidence.collection_sha256)
    heldout1 = partition("validation", 1, evidence.collection_sha256)
    iteration1 = build(base, evidence, train1, heldout1)
    train1_rows = tuple(record.example for episode in train1.episodes
                        for record in episode.records)
    heldout1_rows = tuple(record.example for episode in heldout1.episodes
                          for record in episode.records)

    assert iteration1.iteration == 1
    assert iteration1.train == base.train + train1_rows
    assert iteration1.validation == heldout1_rows
    assert all(row not in base.validation for row in iteration1.validation)

    train2 = partition("train", 2, iteration1.corpus_sha256)
    heldout2 = partition("validation", 2, iteration1.corpus_sha256)
    iteration2 = build(base, evidence, train2, heldout2, prior=iteration1)
    train2_rows = tuple(record.example for episode in train2.episodes
                        for record in episode.records)
    heldout2_rows = tuple(record.example for episode in heldout2.episodes
                          for record in episode.records)

    assert iteration2.iteration == 2
    assert iteration2.train == iteration1.train + train2_rows
    assert iteration2.validation == iteration1.validation + heldout2_rows
    assert iteration2.episodes[:len(iteration1.episodes)] == iteration1.episodes
    assert (iteration2.validation_episodes[:len(iteration1.validation_episodes)]
            == iteration1.validation_episodes)
    train_keys = {(row.episode_seed, row.learner_seat, row.decision.decision_id)
                  for row in iteration2.train}
    heldout_keys = {(row.episode_seed, row.learner_seat, row.decision.decision_id)
                    for row in iteration2.validation}
    assert train_keys.isdisjoint(heldout_keys)


def test_dagger_retrain_persists_reciprocal_evidence_and_uses_same_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    train = getattr(module, "train_dagger_pilot", None)
    assert callable(train), "the bounded DAgger retraining entry point is missing"
    base = _canonical_collection()
    evidence = module.PilotCollectionEvidence("1" * 64, "2" * 64, "3" * 64)
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(),
            model_state_sha256="a" * 64,
            corpus_sha256=evidence.collection_sha256,
            best_epoch=0,
            best_validation_policy_nll=1.25,
        ),
    )
    episodes = tuple(
        module.collect_dagger_game(_DaggerClient(item.learner_seat), loaded, item)
        for item in module.dagger_iteration_schedule()
    )
    augmented = module.build_dagger_training_set(base, evidence, episodes)
    captured = {}
    sentinel = object()

    def fake_pipeline(identity, train_rows, validation_rows, corpus_sha, output,
                      seed, device, **kwargs):
        captured.update(
            identity=identity,
            train=train_rows,
            validation=validation_rows,
            corpus_sha=corpus_sha,
            output=output,
            seed=seed,
            device=device,
            **kwargs,
        )
        return sentinel

    monkeypatch.setattr(module, "_train_pilot_dataset", fake_pipeline, raising=False)
    output = tmp_path / "dagger-iteration-1"

    result = train(augmented, loaded, output, 227, "cpu")

    assert result is sentinel
    assert {
        key: captured[key]
        for key in ("identity", "train", "validation", "corpus_sha",
                    "output", "seed", "device")
    } == {
        "identity": augmented.identity,
        "train": augmented.train,
        "validation": base.validation,
        "corpus_sha": augmented.corpus_sha256,
        "output": output / "training",
        "seed": 227,
        "device": "cpu",
    }
    assert captured["initial_policy"] is loaded
    assert isinstance(captured["batch_provider"], module.StructuredDaggerMixtureSampler)
    manifest_path = output / "training-set.json"
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == (
        augmented.corpus_sha256
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["base_train_examples"] == len(base.train)
    assert manifest["dagger_examples"] == 4
    assert manifest["validation_examples"] == len(base.validation)
    assert (output / "seat-0" / "episode.json").is_file()
    assert (output / "seat-1" / "episode.json").is_file()


def test_structured_dagger_sampler_exposes_exact_49_21_30_and_cycles_uniformly() -> None:
    import ml_lab.tactical_v3_pilot as module

    base = _canonical_collection()
    evidence = module.PilotCollectionEvidence("1" * 64, "2" * 64, "3" * 64)
    loaded = SimpleNamespace(
        model=_EvaluationPolicy(),
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256="a" * 64,
            corpus_sha256=evidence.collection_sha256, best_epoch=0,
            best_validation_policy_nll=1.25,
        ),
    )
    episodes = tuple(
        module.collect_dagger_game(_DaggerClient(item.learner_seat), loaded, item)
        for item in module.dagger_iteration_schedule()
    )
    training_set = module.build_dagger_training_set(base, evidence, episodes)
    sampler = module.StructuredDaggerMixtureSampler(
        training_set, batch_size=256, seed=227,
    )

    counts = {"greedy_standard": 0, "search_conversion": 0, "dagger_targeted": 0}
    first_batch_targeted = set()
    for _ in range(100):
        batch = sampler.next_batch()
        assert len(batch.examples) == 256
        for source, example in zip(batch.sources, batch.examples, strict=True):
            counts[source] += 1
            if source == "dagger_targeted" and _ == 0:
                first_batch_targeted.add((
                    example.episode_seed, example.learner_seat,
                    example.decision.decision_id,
                ))

    assert counts == {
        "greedy_standard": 12_544,
        "search_conversion": 5_376,
        "dagger_targeted": 7_680,
    }
    assert len(first_batch_targeted) == 4


def test_dagger_retrain_warm_starts_incoming_actor_with_frozen_training_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
    from ml_lab.tactical_v3_model import TacticalV3Policy

    base = _canonical_collection()
    evidence = module.PilotCollectionEvidence("1" * 64, "2" * 64, "3" * 64)
    model = TacticalV3Policy(TacticalV3ModelConfig(
        hidden_dim=32, categorical_dim=8, cell_message_rounds=1,
        relation_rounds=1, attention_heads=4, feed_forward_dim=64,
        candidate_hidden_dim=64, horizon_turns=(4, 8, 16),
    ))
    incoming = SimpleNamespace(
        model=model,
        metadata=SimpleNamespace(
            identity=_identity(), model_state_sha256=module.structured_model_state_sha256(model),
            corpus_sha256=evidence.collection_sha256, best_epoch=0,
            best_validation_policy_nll=1.25,
        ),
    )
    episodes = tuple(
        module.collect_dagger_game(_DaggerClient(item.learner_seat), incoming, item)
        for item in module.dagger_iteration_schedule()
    )
    training_set = module.build_dagger_training_set(base, evidence, episodes)
    captured = {}

    def fake_pipeline(*args, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(module, "_train_pilot_dataset", fake_pipeline)
    module.train_dagger_pilot(training_set, incoming, tmp_path / "iteration", 227, "cuda")

    assert captured["initial_policy"] is incoming
    assert captured["trainer_config"] == module.TrainerConfig(
        seed=227, batch_size=256, learning_rate=3e-4, max_epochs=50,
        patience_epochs=5, gradient_clip_norm=1.0, device="cuda",
    )
    assert captured["micro_batch_size"] == 32
    assert captured["training_deadline_seconds"] == 12 * 60 * 60
    assert isinstance(captured["batch_provider"], module.StructuredDaggerMixtureSampler)


def test_collect_game_backfills_truncation_without_remaining_turns() -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("validation", "standard-3v3", 62_000_000, 1, 1)
    client = _FakeClient([
        _view(10, seat=1, reference_seat=1),
        _view(11, seat=1, reference_seat=1),
        _view(12, seat=1, reference_seat=1),
        _view(13, seat=1, reference_seat=1, truncated=True),
    ])
    examples, summary = collect_game(client, item)
    assert len(examples) == 3
    assert all(row.target.terminal_outcome == "draw" for row in examples)
    assert all(row.target.remaining_turns_to_victory is None for row in examples)
    assert all(row.target.truncated for row in examples)
    assert summary.truncated and not summary.terminated


@pytest.mark.parametrize("drift", ["profile", "seat", "reference", "identity", "fallback"])
def test_collect_game_rejects_semantic_or_fallback_drift(drift: str) -> None:
    from ml_lab.tactical_v3_pilot import PilotScheduleItem, collect_game

    item = PilotScheduleItem("train", "standard-3v3", 61_000_000, 0, 0)
    kwargs = {}
    if drift == "profile": kwargs["profile"] = "conversion-1v1-near"
    if drift == "seat": kwargs["seat"] = 1
    if drift == "reference": kwargs["reference_seat"] = 1
    client = _FakeClient(
        [_view(7, **kwargs), _view(8, terminal=True, **kwargs)],
        fallback=1 if drift == "fallback" else 0,
        identity_drift=drift == "identity",
    )
    with pytest.raises(ValueError, match="profile|seat|identity|fallback"):
        collect_game(client, item)


def test_collection_evidence_is_canonical_content_addressed_and_write_once(tmp_path: Path) -> None:
    from ml_lab.tactical_v3_pilot import write_collection_evidence

    collection = _canonical_collection()
    output = tmp_path / "pilot"

    evidence = write_collection_evidence(output, collection)

    assert set(path.name for path in output.iterdir()) == {
        "train.jsonl", "validation.jsonl", "collection.json",
    }
    assert evidence.train_sha256 == hashlib.sha256((output / "train.jsonl").read_bytes()).hexdigest()
    assert evidence.validation_sha256 == hashlib.sha256(
        (output / "validation.jsonl").read_bytes()
    ).hexdigest()
    manifest = json.loads((output / "collection.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["label_sources"] == {
        "standard-3v3": "greedy-one-ply-v1",
        "conversion-profiles": "bounded-search-v1",
    }
    assert manifest["teachers"] == {
        "greedy": {"identity": "greedy-one-ply-v1"},
        "bounded_search": {
            "identity": "bounded-search-v1",
            "search_depth": 4, "expansion_budget": 512,
            "heuristic_identity": "material-plus-pursuit-v1",
        },
    }
    assert (output / "collection.json").read_bytes().endswith(b"\n")
    with pytest.raises(FileExistsError):
        write_collection_evidence(output, collection)


def test_collection_evidence_round_trips_without_collecting_again(tmp_path: Path) -> None:
    from ml_lab.tactical_v3_pilot import (
        load_collection_evidence,
        write_collection_evidence,
    )

    expected = _canonical_collection()
    output = tmp_path / "pilot"
    expected_evidence = write_collection_evidence(output, expected)

    actual, actual_evidence = load_collection_evidence(output, expected.identity)

    assert actual == expected
    assert actual_evidence == expected_evidence


def test_collection_evidence_reopens_frozen_schema1_bounded_search_corpus(
    tmp_path: Path,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_pilot import (
        load_collection_evidence,
        write_collection_evidence,
    )

    expected = _canonical_collection()
    output = tmp_path / "pilot"
    write_collection_evidence(output, expected)
    manifest_path = output / "collection.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = 1
    manifest["label_source"] = "bounded-search-v1"
    manifest["teacher"] = {
        "search_depth": 4,
        "expansion_budget": 512,
        "heuristic_identity": "material-plus-pursuit-v1",
    }
    del manifest["label_sources"]
    del manifest["teachers"]
    for partition in ("train", "validation"):
        path = output / f"{partition}.jsonl"
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            raw = json.loads(line)
            raw["teacher"] = {
                "identity": "bounded-search-v1",
                "search_depth": 4,
                "expansion_budget": 512,
                "actual_expansions": 17,
                "heuristic_identity": "material-plus-pursuit-v1",
                "confidence": None,
            }
            rows.append(
                json.dumps(raw, sort_keys=True, separators=(",", ":")) + "\n",
            )
        data = "".join(rows).encode("utf-8")
        path.write_bytes(data)
        manifest[partition]["sha256"] = hashlib.sha256(data).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8", newline="\n",
    )

    actual, evidence = load_collection_evidence(output, expected.identity)

    assert len(actual.train) == len(expected.train)
    assert len(actual.validation) == len(expected.validation)
    assert all(row.teacher.identity == "bounded-search-v1" for row in actual.train)
    assert all(row.teacher.expansion_budget == 512 for row in actual.validation)
    module._require_collection(actual)
    assert evidence.collection_sha256 == hashlib.sha256(
        manifest_path.read_bytes(),
    ).hexdigest()


def test_train_pilot_uses_exact_configs_reloads_cpu_and_writes_exact_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_training import (
        EpochMetrics,
        StepMetrics,
        TrainingResult,
        _checkpoint_state,
        _snapshot_state,
    )

    collection = _canonical_collection()
    output = tmp_path / "pilot"
    evidence = module.write_collection_evidence(output, collection)
    captured = {}

    def fake_train(
        train, validation, model_config, objective_config, trainer_config,
        *, epoch_callback, step_callback, deadline_monotonic,
        initial_state_dict, training_batch_provider, checkpoint_callback,
        resume_state,
    ):
        assert initial_state_dict is None
        assert training_batch_provider is None
        assert resume_state is None
        captured.update(train=train, validation=validation, model=model_config,
                        objective=objective_config, trainer=trainer_config,
                        deadline=deadline_monotonic)
        model = TacticalV3Policy(model_config).eval()
        metrics = MappingProxyType({
            "total": 1.0, "policy": 1.0, "outcome": 0.0,
            "horizon": 0.0, "remaining_turns": 0.0,
        })
        history = (EpochMetrics(0, metrics, metrics, 1.0, True),)
        step_callback(StepMetrics("train", 0, 0, 1, 2, metrics))
        step_callback(StepMetrics("train", 0, 1, 2, 1, metrics))
        step_callback(StepMetrics("validation", 0, 0, 1, 2, metrics))
        step_callback(StepMetrics("validation", 0, 1, 2, 1, metrics))
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=trainer_config.learning_rate,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(trainer_config.seed)
        checkpoint_callback(_checkpoint_state(
            model=model,
            optimizer=optimizer,
            generator=generator,
            model_config=model_config,
            objective_config=objective_config,
            trainer_config=trainer_config,
            micro_batch_size=None,
            next_epoch=1,
            best_state=_snapshot_state(model),
            history=list(history),
            best_epoch=0,
            best_nll=1.0,
            epochs_without_improvement=0,
            train_global_step=1,
            validation_global_step=1,
            uses_external_batch_provider=False,
        ))
        epoch_callback(history[0])
        return TrainingResult(model, model_config, objective_config, trainer_config,
                              0, 1.0, False, history)

    monkeypatch.setattr(module, "train_offline", fake_train)
    attempt = output / "retry-1"
    artifacts = module.train_pilot(
        collection,
        evidence,
        output,
        227,
        "cpu",
        artifacts_output=attempt,
    )

    assert captured["train"] is collection.train
    assert captured["validation"] is collection.validation
    assert captured["model"] == TacticalV3ModelConfig(
        hidden_dim=32, categorical_dim=8, cell_message_rounds=1,
        relation_rounds=1, attention_heads=4, feed_forward_dim=64,
        candidate_hidden_dim=64, horizon_turns=(4, 8, 16),
    )
    assert captured["objective"].policy_coefficient == 1.0
    assert captured["objective"].outcome_coefficient == 0.0
    assert captured["trainer"].batch_size == 32
    assert captured["trainer"].learning_rate == 0.001
    assert captured["trainer"].max_epochs == 2
    assert captured["trainer"].patience_epochs == 1
    assert captured["deadline"] > 0.0
    assert artifacts.loaded.metadata.corpus_sha256 == evidence.collection_sha256
    assert next(artifacts.loaded.model.parameters()).device.type == "cpu"
    assert artifacts.restored_train.valid_logit_count == len(collection.train)
    assert artifacts.restored_validation.valid_logit_count == len(collection.validation)
    expected_history = (
        '{"epoch":0,"improved":true,"train":{"horizon":0.0,"outcome":0.0,'
        '"policy":1.0,"remaining_turns":0.0,"total":1.0},'
        '"validation":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"validation_policy_nll":1.0}\n'
    )
    assert (attempt / "metrics.jsonl").read_text(encoding="utf-8") == expected_history
    assert (attempt / "telemetry.jsonl").read_text(encoding="utf-8") == expected_history
    expected_steps = (
        '{"batch_index":0,"epoch":0,"example_count":2,"global_step":1,'
        '"metrics":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"phase":"train"}\n'
        '{"batch_index":1,"epoch":0,"example_count":1,"global_step":2,'
        '"metrics":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"phase":"train"}\n'
        '{"batch_index":0,"epoch":0,"example_count":2,"global_step":1,'
        '"metrics":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"phase":"validation"}\n'
        '{"batch_index":1,"epoch":0,"example_count":1,"global_step":2,'
        '"metrics":{"horizon":0.0,"outcome":0.0,"policy":1.0,'
        '"remaining_turns":0.0,"total":1.0},"phase":"validation"}\n'
    )
    assert (attempt / "steps.jsonl").read_text(encoding="utf-8") == expected_steps
    console = capsys.readouterr().out
    assert "phase=initial_metrics" in console
    assert "epoch=1/2" in console
    assert "validation_policy_nll=1.000000" in console
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(attempt / "tensorboard")).Reload()
    assert events.Scalars("baseline/train_policy_nll")[0].value >= 0.0
    assert events.Scalars("epoch/validation_policy_nll")[0].value == pytest.approx(1.0)
    assert events.Scalars("progress/started")[0].value == pytest.approx(1.0)
    assert [item.step for item in events.Scalars("step/train_policy")] == [1, 2]
    assert [item.value for item in events.Scalars("step/train_policy")] == pytest.approx(
        [1.0, 1.0]
    )
    assert [item.step for item in events.Scalars("step/validation_policy")] == [1, 2]


def test_policy_target_metrics_honors_training_deadline_before_a_batch() -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_model import TacticalV3Policy

    collection = _canonical_collection()
    model_config, _, _ = module._pilot_configs(227, "cpu")
    model = TacticalV3Policy(model_config).eval()

    with pytest.raises(TimeoutError, match="training deadline"):
        module._policy_target_metrics(
            model,
            collection.train,
            deadline_monotonic=0.0,
        )


def test_pilot_training_accepts_an_explicitly_disabled_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collection = _canonical_collection()
    output = tmp_path / "training"
    output.mkdir()
    observed = []

    def inspect_deadline(model, examples, *, deadline_monotonic, **kwargs):
        observed.append(deadline_monotonic)
        raise RuntimeError("deadline inspected")

    monkeypatch.setattr(module, "_policy_target_metrics", inspect_deadline)

    with pytest.raises(RuntimeError, match="deadline inspected"):
        module._train_pilot_dataset(
            _identity(),
            collection.train,
            collection.validation,
            "a" * 64,
            output,
            227,
            "cpu",
            training_deadline_seconds=None,
            tensorboard_enabled=False,
        )

    assert observed == [None]


def test_pilot_training_honors_cooperative_stop_without_tensorboard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_training import StepMetrics

    collection = _canonical_collection()
    stopped = {"value": False}
    metrics = MappingProxyType({
        "total": 1.0,
        "policy": 1.0,
        "outcome": 0.0,
        "horizon": 0.0,
        "remaining_turns": 0.0,
    })

    def fake_policy_metrics(model, examples, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(len(examples), len(examples))
        return module.PolicyTargetMetrics(1.0, 0.5, len(examples), len(examples))

    def fake_train(*args, step_callback, **kwargs):
        stopped["value"] = True
        step_callback(StepMetrics("train", 0, 0, 1, 1, metrics))
        raise AssertionError("cooperative stop did not interrupt training")

    monkeypatch.setattr(module, "_policy_target_metrics", fake_policy_metrics)
    monkeypatch.setattr(module, "train_offline", fake_train)
    output = tmp_path / "training"
    output.mkdir()

    with pytest.raises(
        module.PilotTrainingStopRequested,
        match="training stop requested",
    ):
        module._train_pilot_dataset(
            _identity(),
            collection.train,
            collection.validation,
            "a" * 64,
            output,
            227,
            "cpu",
            tensorboard_enabled=False,
            stop_requested=lambda: stopped["value"],
        )

    assert not (output / "tensorboard").exists()
    assert not (output / "checkpoints").exists()
    assert (output / "steps.jsonl").is_file()


def test_pilot_stop_after_completed_epoch_retains_best_and_resume_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_training import (
        EpochMetrics,
        TrainingResult,
        _checkpoint_state,
        _snapshot_state,
    )

    collection = _canonical_collection()
    stopped = {"value": False}
    metrics = MappingProxyType({
        "total": 1.0,
        "policy": 1.0,
        "outcome": 0.0,
        "horizon": 0.0,
        "remaining_turns": 0.0,
    })

    def fake_policy_metrics(model, examples, **kwargs):
        callback = kwargs.get("progress_callback")
        if callback is not None:
            callback(len(examples), len(examples))
        return module.PolicyTargetMetrics(1.0, 0.5, len(examples), len(examples))

    def fake_train(
        train, validation, model_config, objective_config, trainer_config,
        *, checkpoint_callback, epoch_callback, **kwargs,
    ):
        del train, validation, kwargs
        model = TacticalV3Policy(model_config).eval()
        metric = EpochMetrics(0, metrics, metrics, 1.0, True)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=trainer_config.learning_rate,
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(trainer_config.seed)
        checkpoint_callback(_checkpoint_state(
            model=model,
            optimizer=optimizer,
            generator=generator,
            model_config=model_config,
            objective_config=objective_config,
            trainer_config=trainer_config,
            micro_batch_size=None,
            next_epoch=1,
            best_state=_snapshot_state(model),
            history=[metric],
            best_epoch=0,
            best_nll=1.0,
            epochs_without_improvement=0,
            train_global_step=1,
            validation_global_step=1,
            uses_external_batch_provider=False,
        ))
        stopped["value"] = True
        epoch_callback(metric)
        return TrainingResult(
            model, model_config, objective_config, trainer_config,
            0, 1.0, False, (metric,),
        )

    monkeypatch.setattr(module, "_policy_target_metrics", fake_policy_metrics)
    monkeypatch.setattr(module, "train_offline", fake_train)
    output = tmp_path / "training"
    output.mkdir()

    with pytest.raises(
        module.PilotTrainingStopRequested,
        match="training stop requested",
    ):
        module._train_pilot_dataset(
            _identity(),
            collection.train,
            collection.validation,
            "a" * 64,
            output,
            227,
            "cpu",
            tensorboard_enabled=False,
            stop_requested=lambda: stopped["value"],
        )

    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "last.pt").is_file()
    assert (output / "metrics.jsonl").read_text(encoding="utf-8") == (
        output / "telemetry.jsonl"
    ).read_text(encoding="utf-8")


def test_pilot_honors_deferred_stop_during_restored_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collection = _canonical_collection()
    original_metrics = module._policy_target_metrics
    metric_pass = {"count": 0}
    deferred_stop = {"requested": False}

    def request_during_restored_metrics(model, examples, **kwargs):
        metric_pass["count"] += 1
        progress_callback = kwargs.get("progress_callback")
        if metric_pass["count"] == 3:
            def request_then_report(completed, total):
                deferred_stop["requested"] = True
                progress_callback(completed, total)

            kwargs["progress_callback"] = request_then_report
        return original_metrics(model, examples, **kwargs)

    monkeypatch.setattr(
        module, "_policy_target_metrics", request_during_restored_metrics,
    )
    _, _, default_trainer = module._pilot_configs(227, "cpu")
    trainer = replace(
        default_trainer, max_epochs=1, patience_epochs=1,
    )
    output = tmp_path / "training"
    output.mkdir()

    with pytest.raises(
        module.PilotTrainingStopRequested,
        match="stop-after-checkpoint requested",
    ):
        module._train_pilot_dataset(
            _identity(),
            collection.train,
            collection.validation,
            "a" * 64,
            output,
            227,
            "cpu",
            trainer_config=trainer,
            tensorboard_enabled=False,
            stop_after_checkpoint_requested=(
                lambda: deferred_stop["requested"]
            ),
        )

    assert metric_pass["count"] == 3
    assert (output / "checkpoints" / "best.pt").is_file()
    assert (output / "checkpoints" / "last.pt").is_file()


class _EvaluationClient:
    def __init__(self) -> None:
        self.resets = []
        self.steps = []
        self.status_calls = 0
        self.current = None

    def duel_reset(self, seed, p0, p1, learner, profile, reference):
        self.resets.append((seed, p0, p1, learner, profile, reference))
        self.current = (learner, profile, reference)
        return _view(8 if p0 == p1 == "random" else 7, seat=learner, profile=profile,
                     reference_seat=reference, terminal=p0 == p1 == "random")

    def duel_step(self, selection):
        self.steps.append(selection)
        learner, profile, reference = self.current
        return _view(8, seat=learner, profile=profile, reference_seat=reference,
                     terminal=True)

    def duel_status(self):
        self.status_calls += 1
        return 0


class _EvaluationPolicy:
    config = SimpleNamespace(horizon_turns=(4, 8, 16))

    def select(self, batch):
        from ml_lab.tactical_v3_model import CandidateIdentity
        return tuple(CandidateIdentity(
            int(batch.candidates.decision_id[index, 0]),
            int(batch.candidates.candidate_id[index, 0]),
        ) for index in range(batch.candidates.decision_id.shape[0]))


def test_matched_evaluation_uses_one_schedule_random_baseline_and_legal_model_actions() -> None:
    from ml_lab.tactical_v3_pilot import evaluate_pilot, evaluation_schedule

    schedule = evaluation_schedule()
    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    model_client = _EvaluationClient()
    random_client = _EvaluationClient()
    model = evaluate_pilot(model_client, loaded, "model", schedule)
    baseline = evaluate_pilot(random_client, loaded, "random", schedule)

    assert tuple(game.schedule for game in model.games) == schedule
    assert tuple(game.schedule for game in baseline.games) == schedule
    assert model.aggregate.games == baseline.aggregate.games == 28
    assert model.aggregate.wins == baseline.aggregate.wins == 28
    assert model.aggregate.candidate_errors == 0
    assert model_client.status_calls == random_client.status_calls == 28
    assert len(model_client.steps) == 28
    assert not random_client.steps
    assert all(p0 == p1 == "random" for _, p0, p1, *_ in random_client.resets)
    assert all(selection.candidate_id == 0 for selection in model_client.steps)


def test_evaluation_observer_sees_selected_decisions_and_terminal_views_without_changing_results() -> None:
    from ml_lab.tactical_v3_pilot import evaluate_pilot, evaluation_schedule

    schedule = evaluation_schedule()
    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    client = _EvaluationClient()
    events = []

    evaluation = evaluate_pilot(
        client,
        loaded,
        'model',
        schedule,
        observation_callback=lambda item, view, selected: events.append(
            (item, view, selected)
        ),
    )

    assert evaluation.aggregate.games == 28
    assert len(client.steps) == 28
    assert len(events) == 56
    for index, item in enumerate(schedule):
        decision_event, terminal_event = events[index * 2:index * 2 + 2]
        assert decision_event[0] == terminal_event[0] == item
        assert decision_event[1].terminated is False
        assert decision_event[2].candidate_id == client.steps[index].candidate_id
        assert terminal_event[1].terminated is True
        assert terminal_event[2] is None


def test_diagnostic_evaluation_uses_only_the_new_frozen_schedule() -> None:
    from ml_lab.tactical_v3_pilot import (
        diagnostic_evaluation_schedule, evaluate_pilot,
    )

    schedule = diagnostic_evaluation_schedule()
    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    model_client = _EvaluationClient()

    model = evaluate_pilot(model_client, loaded, "model", schedule)

    assert tuple(game.schedule for game in model.games) == schedule
    assert model.aggregate.games == 70
    assert len(model_client.steps) == 70


def test_selective_dagger_evaluation_is_standard_only_and_never_uses_oracle() -> None:
    from ml_lab.tactical_v3_pilot import (
        evaluate_pilot, selective_dagger_evaluation_schedule,
    )

    schedule = selective_dagger_evaluation_schedule()
    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    client = _EvaluationClient()

    evaluation = evaluate_pilot(client, loaded, "model", schedule)

    assert evaluation.aggregate.games == 200
    assert tuple(game.schedule for game in evaluation.games) == schedule
    assert len(client.steps) == 200
    assert all(reset[4] == "standard-3v3" for reset in client.resets)
    assert not hasattr(client, "duel_oracle_query")


def test_point_mobility_diagnostic_schedule_is_exact_standard_reciprocal_subset() -> None:
    from ml_lab.tactical_v3_pilot import (
        evaluate_pilot,
        point_mobility_diagnostic_schedule,
    )

    schedule = point_mobility_diagnostic_schedule()
    assert len(schedule) == 20
    assert [(item.episode_seed, item.learner_seat) for item in schedule] == [
        (seed, seat)
        for seed in range(20_000_000, 20_000_010)
        for seat in (0, 1)
    ]
    assert all(item.profile_id == 'standard-3v3' for item in schedule)

    loaded = SimpleNamespace(model=_EvaluationPolicy(),
                             metadata=SimpleNamespace(identity=_identity()))
    evaluation = evaluate_pilot(_EvaluationClient(), loaded, 'model', schedule)
    assert evaluation.aggregate.games == 20


def test_evaluation_moves_model_and_decision_batches_to_requested_cuda_device() -> None:
    import torch
    from ml_lab.tactical_v3_model import CandidateIdentity
    from ml_lab.tactical_v3_pilot import evaluate_pilot, evaluation_schedule

    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    class CudaEvaluationPolicy(torch.nn.Module):
        config = SimpleNamespace(horizon_turns=(4, 8, 16))

        def __init__(self) -> None:
            super().__init__()
            self.marker = torch.nn.Parameter(torch.zeros(()))

        def select(self, batch):
            assert self.marker.device.type == "cuda"
            assert batch.candidates.decision_id.device.type == "cuda"
            return tuple(CandidateIdentity(
                int(batch.candidates.decision_id[index, 0].item()),
                int(batch.candidates.candidate_id[index, 0].item()),
            ) for index in range(batch.candidates.decision_id.shape[0]))

    loaded = SimpleNamespace(
        model=CudaEvaluationPolicy(),
        metadata=SimpleNamespace(identity=_identity()),
    )
    client = _EvaluationClient()

    evaluation = evaluate_pilot(
        client, loaded, "model", evaluation_schedule(), device="cuda",
    )

    assert evaluation.aggregate.games == 28
    assert len(client.steps) == 28


def test_pilot_decision_requires_metric_improvement_zero_errors_and_ten_point_margin() -> None:
    from ml_lab.tactical_v3_pilot import (
        PilotEvaluation, PilotEvaluationSummary, PilotTrainingArtifacts,
        PolicyTargetMetrics, pilot_decision,
    )

    initial = PolicyTargetMetrics(1.0, 0.25, 4, 4)
    restored = PolicyTargetMetrics(0.5, 0.75, 4, 4)
    training = PilotTrainingArtifacts(
        Path("best.pt"), initial, initial, restored, restored, object(), 1.0,
    )
    def evaluation(controller, win_rate, errors=0):
        summary = PilotEvaluationSummary(28, 0, 0, 0, 0, 1.0, errors, 0, win_rate)
        return PilotEvaluation(controller, (), summary, ())

    promising, reasons = pilot_decision(
        training, evaluation("model", 0.60), evaluation("random", 0.50),
    )
    assert promising and reasons == ()
    assert pilot_decision(
        training, evaluation("model", 0.59), evaluation("random", 0.50),
    )[0] is False
    assert pilot_decision(
        training, evaluation("model", 0.60, errors=1), evaluation("random", 0.50),
    )[0] is False


def test_run_pilot_writes_machine_authority_and_report_from_one_mapping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collection = _canonical_collection()
    initial = module.PolicyTargetMetrics(1.0, 0.25, 4, 4)
    restored = module.PolicyTargetMetrics(0.5, 0.75, 4, 4)
    training = module.PilotTrainingArtifacts(
        Path("checkpoints/best.pt"), initial, initial, restored, restored,
        SimpleNamespace(metadata=SimpleNamespace(identity=collection.identity)), 2.5,
    )
    summary = module.PilotEvaluationSummary(28, 17, 9, 2, 2, 11.5, 0, 0, 17 / 28)
    baseline = module.PilotEvaluationSummary(28, 12, 12, 4, 4, 13.0, 0, 0, 12 / 28)
    evaluations = {
        "model": module.PilotEvaluation("model", (), summary, ()),
        "random": module.PilotEvaluation("random", (), baseline, ()),
    }

    monkeypatch.setattr(module, "collect_pilot", lambda command: collection)
    monkeypatch.setattr(module, "train_pilot",
                        lambda *args: training)
    monkeypatch.setattr(module, "evaluate_pilot",
                        lambda client, loaded, controller, schedule: evaluations[controller])

    class Client:
        def __init__(self, command, *, environment_kind):
            assert tuple(command) == ("dotnet", "server.dll")
            assert environment_kind == "duel"
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(module, "TacticalV3GymClient", Client)
    output = tmp_path / "pilot"
    command = ("python", "-m", "ml_lab", "pilot")
    assert module.run_pilot(
        ("dotnet", "server.dll"), output, 227, "cpu", command,
    ) == output

    evaluation = json.loads((output / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["execution"]["command"] == list(command)
    assert evaluation["execution"]["device"] == "cpu"
    assert evaluation["identity"]["contract_hash"] == collection.identity.contract_hash
    assert evaluation["collection"] == {
        "train_examples": 28, "validation_examples": 14,
        "train_games": 28, "validation_games": 14,
    }
    assert evaluation["decision"]["promising"] is True
    assert evaluation["claim_limit"].startswith("directional fixed-schedule")
    report = (output / "report.md").read_text(encoding="utf-8")
    assert "python -m ml_lab pilot" in report
    assert "PROMISING" in report
    assert collection.identity.contract_hash in report


def test_pilot_cli_routes_only_fixed_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    server.write_bytes(b"server")
    scenario.write_text("{}", encoding="utf-8")
    output = tmp_path / "pilot"
    captured = {}
    monkeypatch.setattr(cli, "run_pilot", lambda *args: captured.setdefault("args", args))
    argv = (
        "pilot", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--seed", "227", "--device", "cuda",
    )

    assert cli.main(argv) == 0
    server_cmd, actual_output, seed, device, command = captured["args"]
    assert server_cmd == ("dotnet", str(server), "--scenario-file", str(scenario))
    assert actual_output == output
    assert seed == 227 and device == "cuda"
    assert tuple(command[-len(argv):]) == argv


def test_pilot_retry_cli_routes_existing_collection_without_collection_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    output = tmp_path / "pilot"
    server.write_bytes(b"server")
    scenario.write_text("{}", encoding="utf-8")
    output.mkdir()
    (output / "collection.json").write_text("{}\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        cli, "run_pilot_retry",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )
    argv = (
        "pilot-retry", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--seed", "227", "--device", "cuda",
        "--attempt", "2",
    )

    assert cli.main(argv) == 0
    server_cmd, actual_output, seed, device, command = captured["args"]
    assert server_cmd == ("dotnet", str(server), "--scenario-file", str(scenario))
    assert actual_output == output
    assert seed == 227 and device == "cuda"
    assert captured["kwargs"] == {"attempt_number": 2}
    assert tuple(command[-len(argv):]) == argv


def test_pilot_diagnose_cli_routes_frozen_checkpoint_without_training_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    output = tmp_path / "pilot"
    checkpoint = output / "retry-2" / "checkpoints" / "best.pt"
    server.write_bytes(b"server")
    scenario.write_text("{}\n", encoding="utf-8")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    (output / "collection.json").write_text("{}\n", encoding="utf-8")
    captured = {}
    monkeypatch.setattr(
        cli, "run_pilot_diagnostics",
        lambda *args: captured.setdefault("args", args),
    )
    argv = (
        "pilot-diagnose", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--attempt", "2", "--device", "cuda",
    )

    assert cli.main(argv) == 0

    server_cmd, actual_output, attempt, device, command = captured["args"]
    assert server_cmd == ("dotnet", str(server), "--scenario-file", str(scenario))
    assert actual_output == output
    assert attempt == 2 and device == "cuda"
    assert tuple(command[-len(argv):]) == argv


def test_run_pilot_diagnostics_reuses_frozen_evidence_and_writes_game_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    collection = _canonical_collection()
    root = tmp_path / "pilot"
    module.write_collection_evidence(root, collection)
    checkpoint = root / "retry-2" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    device_calls = []

    class Model:
        def to(self, device):
            device_calls.append(str(device))
            return self
        def cpu(self):
            device_calls.append("cpu")
            return self

    loaded = SimpleNamespace(
        model=Model(), metadata=SimpleNamespace(identity=collection.identity),
    )

    class Client:
        def __init__(self, command, *, environment_kind):
            assert tuple(command) == ("dotnet", "server.dll")
            assert environment_kind == "duel"
            self.identity = collection.identity
        def __enter__(self): return self
        def __exit__(self, *args): return False

    def fake_load(path, encoding_hash, capacity_hash):
        assert path == checkpoint
        assert encoding_hash == collection.identity.encoding_hash
        assert capacity_hash == collection.identity.capacity_hash
        return loaded

    breakdown = {"standard-3v3": {"all": {"examples": 2}, "seats": {}}}
    monkeypatch.setattr(module, "TacticalV3GymClient", Client)
    monkeypatch.setattr(module, "load_structured_checkpoint", fake_load)
    monkeypatch.setattr(
        module, "validation_metric_breakdown",
        lambda model, examples: breakdown,
    )
    controllers = []

    def fake_evaluate(client, actual_loaded, controller, schedule):
        assert actual_loaded is loaded
        assert schedule == module.diagnostic_evaluation_schedule()
        controllers.append(controller)
        games = tuple(module.PilotEvaluationGame(
            controller, item, item.learner_seat, "win", True, False, 1, 0, 0,
        ) for item in schedule)
        profiles = tuple((
            profile,
            module._evaluation_summary(tuple(
                game for game in games if game.schedule.profile_id == profile
            )),
        ) for profile in module.PILOT_PROFILES)
        return module.PilotEvaluation(
            controller, games, module._evaluation_summary(games), profiles,
        )

    monkeypatch.setattr(module, "evaluate_pilot", fake_evaluate)
    command = ("python", "run_tactical_v3_imitation.py", "pilot-diagnose")

    path = module.run_pilot_diagnostics(
        ("dotnet", "server.dll"), root, 2, "cuda", command,
    )

    assert path == root / "retry-2" / "diagnostics.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["schema_version"] == 1
    assert data["execution"]["command"] == list(command)
    assert data["checkpoint"]["sha256"] == hashlib.sha256(b"checkpoint").hexdigest()
    assert data["validation"]["examples"] == len(collection.validation)
    assert data["validation"]["profiles"] == breakdown
    assert len(data["schedule"]) == 70
    assert len(data["model"]["games"]) == 70
    assert data["model"]["aggregate"]["games"] == 70
    assert data["random"]["aggregate"]["games"] == 70
    assert controllers == ["model", "random"]
    assert device_calls == ["cuda", "cpu"]
    with pytest.raises(FileExistsError, match="diagnostics"):
        module.run_pilot_diagnostics(
            ("dotnet", "server.dll"), root, 2, "cuda", command,
        )


def test_run_pilot_retry_reuses_collection_and_writes_only_retry_subdirectory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_pilot as module

    expected = _canonical_collection()
    root = tmp_path / "pilot"
    module.write_collection_evidence(root, expected)
    initial = module.PolicyTargetMetrics(1.0, 0.25, 4, 4)
    restored = module.PolicyTargetMetrics(0.5, 0.75, 4, 4)
    training = module.PilotTrainingArtifacts(
        Path("checkpoints/best.pt"), initial, initial, restored, restored,
        SimpleNamespace(metadata=SimpleNamespace(identity=expected.identity)), 2.5,
    )
    summary = module.PilotEvaluationSummary(28, 17, 9, 2, 2, 11.5, 0, 0, 17 / 28)
    baseline = module.PilotEvaluationSummary(28, 12, 12, 4, 4, 13.0, 0, 0, 12 / 28)
    evaluations = {
        "model": module.PilotEvaluation("model", (), summary, ()),
        "random": module.PilotEvaluation("random", (), baseline, ()),
    }
    captured = {}

    def fail_collection(*args):
        pytest.fail("retry must not collect games")

    def fake_train(collection, evidence, collection_root, seed, device, *, artifacts_output):
        captured.update(
            collection=collection,
            evidence=evidence,
            collection_root=collection_root,
            seed=seed,
            device=device,
            artifacts_output=artifacts_output,
        )
        artifacts_output.mkdir()
        return training

    class Client:
        def __init__(self, command, *, environment_kind):
            assert tuple(command) == ("dotnet", "server.dll")
            assert environment_kind == "duel"
            self.identity = expected.identity
        def __enter__(self): return self
        def __exit__(self, *args): return False

    monkeypatch.setattr(module, "collect_pilot", fail_collection)
    monkeypatch.setattr(module, "train_pilot", fake_train)
    monkeypatch.setattr(module, "TacticalV3GymClient", Client)
    monkeypatch.setattr(
        module, "evaluate_pilot",
        lambda client, loaded, controller, schedule: evaluations[controller],
    )
    command = ("python", "-m", "ml_lab", "pilot-retry")

    attempt = module.run_pilot_retry(
        ("dotnet", "server.dll"), root, 227, "cpu", command,
        attempt_number=2,
    )

    assert attempt == root / "retry-2"
    assert captured["collection"] == expected
    assert captured["collection_root"] == root
    assert captured["artifacts_output"] == root / "retry-2"
    assert set(path.name for path in root.iterdir()) == {
        "train.jsonl", "validation.jsonl", "collection.json", "retry-2",
    }
    evaluation = json.loads((attempt / "evaluation.json").read_text(encoding="utf-8"))
    assert evaluation["execution"]["command"] == list(command)
    assert evaluation["collection"]["train_games"] == 28


@pytest.mark.parametrize("failure", ["server", "scenario", "output", "seed", "extra"])
def test_pilot_cli_rejects_invalid_inputs_before_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str,
) -> None:
    import run_tactical_v3_imitation as cli

    server = tmp_path / "GymServer.dll"
    scenario = tmp_path / "scenario.json"
    server.write_bytes(b"server")
    scenario.write_text("{}", encoding="utf-8")
    output = tmp_path / "pilot"
    if failure == "server": server.unlink()
    if failure == "scenario": scenario.unlink()
    if failure == "output": output.mkdir()
    called = False
    def fail_if_called(*args):
        nonlocal called
        called = True
    monkeypatch.setattr(cli, "run_pilot", fail_if_called)
    argv = [
        "pilot", "--server-dll", str(server), "--scenario", str(scenario),
        "--output", str(output), "--seed", "226" if failure == "seed" else "227",
        "--device", "cpu",
    ]
    if failure == "extra": argv.extend(("--profiles", "all"))

    with pytest.raises((ValueError, FileExistsError, SystemExit)):
        cli.main(argv)
    assert not called


def test_real_server_collects_reciprocal_standard_games() -> None:
    from ml_lab.tactical_v3_client import TacticalV3GymClient
    from ml_lab.tactical_v3_pilot import collect_game, collection_schedule

    server = ROOT / "engine" / "HexWars.GymServer" / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
    scenario = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"
    assert server.is_file() and scenario.is_file()
    items = collection_schedule("train")[:2]
    with TacticalV3GymClient(
        ("dotnet", str(server), "--scenario-file", str(scenario)),
        environment_kind="duel",
    ) as client:
        results = tuple(collect_game(client, item) for item in items)

    assert [item.learner_seat for item in items] == [0, 1]
    assert all(examples for examples, _ in results)
    assert all(summary.terminated or summary.truncated for _, summary in results)
    assert all(summary.internal_fallback_count == 0 for _, summary in results)
    assert all(
        example.teacher.identity == "greedy-one-ply-v1" and
        example.teacher.search_depth == 0 and
        example.teacher.expansion_budget == 0 and
        example.teacher.actual_expansions == 0 and
        example.teacher.heuristic_identity == "greedy-one-ply-v1"
        for examples, _ in results for example in examples
    )
