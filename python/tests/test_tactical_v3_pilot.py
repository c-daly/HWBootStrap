from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

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
    from ml_lab.tactical_v3_pilot import (
        PilotCollection, collect_game, collection_schedule, write_collection_evidence,
    )

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
    collection = PilotCollection(_identity(), partitions[0], partitions[1], tuple(games))
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
