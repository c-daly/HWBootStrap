from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ml_lab.tactical_v3_client import OracleStepResult, TeacherSelection
from ml_lab.tactical_v3_schema import parse_spaces, parse_view
from tests.test_tactical_v3_schema import minimal_view_payload


ROOT = Path(__file__).resolve().parents[2]
DUEL_SPACES = ROOT / "python" / "tests" / "fixtures" / "tactical_v3" / "seed-41-duel-spaces.json"


def _identity():
    return parse_spaces(json.loads(DUEL_SPACES.read_text(encoding="utf-8")))


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
    terminal: bool = False,
):
    payload = minimal_view_payload()
    payload["decision_id"] = decision_id
    payload["seat"] = seat
    payload["reference_seat"] = reference_seat
    payload["start_profile"] = "standard-3v3"
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

    def duel_status(self) -> int:
        return self._fallback


class _DaggerClient:
    def __init__(self, seat: int = 0) -> None:
        self._identity = _identity()
        self._views = (
            _view_with_two_candidates(7, seat=seat, reference_seat=seat),
            _view_with_two_candidates(8, seat=seat, reference_seat=seat),
            _view_with_two_candidates(
                9, seat=seat, reference_seat=seat, terminal=True,
            ),
        )
        self._index = 0
        self.events = []

    @property
    def identity(self):
        return self._identity

    def duel_reset(self, *args):
        self.events.append(("reset", args))
        return self._views[0]

    def duel_oracle_query(self, decision_id):
        current = self._views[self._index]
        assert decision_id == current.decision.decision_id
        teacher_candidate = 1 if self._index == 0 else 0
        self.events.append(("query", decision_id, teacher_candidate))
        return TeacherSelection(
            decision_id, teacher_candidate, 4, 512, 21 + self._index,
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
        PILOT_PROFILES, collection_schedule, evaluation_schedule,
    )

    assert PILOT_PROFILES == (
        "standard-3v3", "conversion-3v1-near", "conversion-3v1-far",
        "conversion-2v1-near", "conversion-2v1-far",
        "conversion-1v1-near", "conversion-1v1-far",
    )
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
    assert all(row.teacher.identity == "bounded-search-v1" for row in examples)
    assert [row.teacher.actual_expansions for row in examples] == [12, 13]
    assert all(row.teacher.search_depth == 4 and row.teacher.expansion_budget == 512
               and row.teacher.confidence is None for row in examples)
    assert summary.decisions == 2 and summary.winner == 0 and summary.internal_fallback_count == 0


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
                      seed, device):
        captured.update(
            identity=identity,
            train=train_rows,
            validation=validation_rows,
            corpus_sha=corpus_sha,
            output=output,
            seed=seed,
            device=device,
        )
        return sentinel

    monkeypatch.setattr(module, "_train_pilot_dataset", fake_pipeline, raising=False)
    output = tmp_path / "dagger-iteration-1"

    result = train(augmented, output, 227, "cpu")

    assert result is sentinel
    assert captured == {
        "identity": augmented.identity,
        "train": augmented.train,
        "validation": base.validation,
        "corpus_sha": augmented.corpus_sha256,
        "output": output / "training",
        "seed": 227,
        "device": "cpu",
    }
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
    assert manifest["label_source"] == "bounded-search-v1"
    assert manifest["teacher"] == {
        "search_depth": 4, "expansion_budget": 512,
        "heuristic_identity": "material-plus-pursuit-v1",
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


def test_train_pilot_uses_exact_configs_reloads_cpu_and_writes_exact_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    import ml_lab.tactical_v3_pilot as module
    from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_training import EpochMetrics, StepMetrics, TrainingResult

    collection = _canonical_collection()
    output = tmp_path / "pilot"
    evidence = module.write_collection_evidence(output, collection)
    captured = {}

    def fake_train(
        train, validation, model_config, objective_config, trainer_config,
        *, epoch_callback, step_callback, deadline_monotonic,
    ):
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
        example.teacher.search_depth == 4 and
        example.teacher.expansion_budget == 512 and
        example.teacher.heuristic_identity == "material-plus-pursuit-v1"
        for examples, _ in results for example in examples
    )
