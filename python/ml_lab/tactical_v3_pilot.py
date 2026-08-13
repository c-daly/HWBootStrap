"""Frozen schedules and complete-game evidence for the tactical-v3 training pilot."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Literal

from .tactical_v3_client import TacticalV3GymClient, TeacherSelection
from .tactical_v3_corpus import StructuredExample, StructuredTarget, TeacherEvidence
from .tactical_v3_schema import TacticalV3Decision, TacticalV3SemanticIdentity, TacticalV3View


PILOT_PROFILES = (
    "standard-3v3",
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)

_COLLECTION_DEADLINE_SECONDS = 4 * 60 * 60


@dataclass(frozen=True, slots=True)
class PilotScheduleItem:
    partition: Literal["train", "validation", "evaluation"]
    profile_id: str
    episode_seed: int
    learner_seat: Literal[0, 1]
    reference_seat: Literal[0, 1]

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation", "evaluation"}:
            raise ValueError("pilot partition is unsupported")
        if self.profile_id not in PILOT_PROFILES:
            raise ValueError("pilot profile is unsupported")
        if type(self.episode_seed) is not int or not 0 <= self.episode_seed < 2**31:
            raise TypeError("pilot episode_seed must be a nonnegative int32")
        if type(self.learner_seat) is not int or self.learner_seat not in {0, 1}:
            raise ValueError("pilot learner_seat must be 0 or 1")
        if type(self.reference_seat) is not int or self.reference_seat not in {0, 1}:
            raise ValueError("pilot reference_seat must be 0 or 1")
        if self.learner_seat != self.reference_seat:
            raise ValueError("pilot learner and reference seats must match")


@dataclass(frozen=True, slots=True)
class PilotGameSummary:
    schedule: PilotScheduleItem
    winner: int
    terminated: bool
    truncated: bool
    decisions: int
    internal_fallback_count: int


@dataclass(frozen=True, slots=True)
class PilotCollection:
    identity: TacticalV3SemanticIdentity
    train: tuple[StructuredExample, ...]
    validation: tuple[StructuredExample, ...]
    games: tuple[PilotGameSummary, ...]


@dataclass(frozen=True, slots=True)
class PilotCollectionEvidence:
    train_sha256: str
    validation_sha256: str
    collection_sha256: str


def collection_schedule(
    partition: Literal["train", "validation"],
) -> tuple[PilotScheduleItem, ...]:
    if partition == "train":
        base = 61_000_000
        seeds_per_profile = 2
    elif partition == "validation":
        base = 62_000_000
        seeds_per_profile = 1
    else:
        raise ValueError("collection partition must be train or validation")
    return tuple(
        PilotScheduleItem(partition, profile, base + profile_index * seeds_per_profile + offset,
                          seat, seat)
        for profile_index, profile in enumerate(PILOT_PROFILES)
        for offset in range(seeds_per_profile)
        for seat in (0, 1)
    )


def evaluation_schedule() -> tuple[PilotScheduleItem, ...]:
    return tuple(
        PilotScheduleItem("evaluation", profile, 63_000_000 + profile_index * 2 + offset,
                          seat, seat)
        for profile_index, profile in enumerate(PILOT_PROFILES)
        for offset in range(2)
        for seat in (0, 1)
    )


def _validate_identity(identity: object) -> TacticalV3SemanticIdentity:
    if type(identity) is not TacticalV3SemanticIdentity:
        raise TypeError("pilot identity must be TacticalV3SemanticIdentity")
    if identity.environment_kind != "duel":
        raise ValueError("pilot identity environment_kind must be duel")
    return identity


def _validate_view(view: TacticalV3View, item: PilotScheduleItem,
                   identity: TacticalV3SemanticIdentity, current_identity: object,
                   seen: set[int]) -> None:
    if current_identity != identity:
        raise ValueError("pilot semantic identity drifted during collection")
    if view.start_profile != item.profile_id:
        raise ValueError("pilot view profile does not match schedule")
    if view.seat != item.learner_seat:
        raise ValueError("pilot view learner seat does not match schedule")
    if view.reference_seat != item.reference_seat:
        raise ValueError("pilot view reference seat does not match schedule")
    if view.terminated or view.truncated:
        raise ValueError("pilot decision view must be nonterminal")
    if not view.decision.candidates:
        raise ValueError("pilot decision must contain candidates")
    if view.decision.decision_id in seen:
        raise ValueError("pilot decision identity repeated within a game")
    for value in _numbers(asdict(view)):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("pilot view contains a non-finite number")


def _numbers(value: object):
    if isinstance(value, dict):
        for child in value.values(): yield from _numbers(child)
    elif isinstance(value, (list, tuple)):
        for child in value: yield from _numbers(child)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield value


def _validate_selection(selection: TeacherSelection, decision: TacticalV3Decision) -> None:
    if type(selection) is not TeacherSelection:
        raise TypeError("pilot oracle selection must be TeacherSelection")
    if selection.decision_id != decision.decision_id:
        raise ValueError("pilot teacher decision identity drifted")
    if sum(candidate.candidate_id == selection.candidate_id
           for candidate in decision.candidates) != 1:
        raise ValueError("pilot teacher candidate must occur exactly once")
    if (selection.search_depth != 4 or selection.expansion_budget != 512 or
            selection.heuristic_identity != "material-plus-pursuit-v1" or
            not 1 <= selection.actual_expansions <= 512):
        raise ValueError("pilot teacher metadata drifted")


def collect_game(
    client: TacticalV3GymClient,
    item: PilotScheduleItem,
) -> tuple[tuple[StructuredExample, ...], PilotGameSummary]:
    if type(item) is not PilotScheduleItem:
        raise TypeError("item must be PilotScheduleItem")
    if item.partition == "evaluation":
        raise ValueError("evaluation games are not teacher collection games")
    identity = _validate_identity(client.identity)
    p0, p1 = ("external", "random") if item.learner_seat == 0 else ("random", "external")
    view = client.duel_reset(
        item.episode_seed, p0, p1, item.learner_seat,
        item.profile_id, item.reference_seat,
    )
    retained: list[tuple[TacticalV3Decision, TeacherSelection]] = []
    seen: set[int] = set()
    while not view.terminated and not view.truncated:
        _validate_view(view, item, identity, client.identity, seen)
        decision = view.decision
        seen.add(decision.decision_id)
        result = client.duel_oracle_step(decision.decision_id)
        _validate_selection(result.selection, decision)
        retained.append((decision, result.selection))
        view = result.view
    if not retained:
        raise ValueError("pilot game must contain at least one teacher decision")
    if client.identity != identity:
        raise ValueError("pilot semantic identity drifted during collection")
    if view.start_profile != item.profile_id or view.reference_seat != item.reference_seat:
        raise ValueError("pilot terminal profile or reference seat drifted")
    fallback = client.duel_status()
    if fallback != 0:
        raise ValueError("pilot internal fallback count must remain zero")

    outcome: Literal["win", "loss", "draw"]
    if view.winner == item.learner_seat:
        outcome = "win"
    elif view.winner in {0, 1}:
        outcome = "loss"
    else:
        outcome = "draw"
    examples = tuple(
        StructuredExample(
            1,
            decision,
            StructuredTarget(
                selection.candidate_id,
                outcome,
                index,
                len(retained) - index if outcome == "win" else None,
                view.truncated,
            ),
            TeacherEvidence(
                "bounded-search-v1", selection.search_depth,
                selection.expansion_budget, selection.actual_expansions,
                selection.heuristic_identity, None,
            ),
            identity.scenario_id,
            identity.contract_hash,
            identity.encoding_hash,
            identity.capacity_hash,
            item.profile_id,
            item.episode_seed,
            item.learner_seat,
        )
        for index, (decision, selection) in enumerate(retained)
    )
    return examples, PilotGameSummary(
        item, view.winner, view.terminated, view.truncated, len(examples), fallback,
    )


def collect_pilot(server_cmd: Sequence[str]) -> PilotCollection:
    started = time.monotonic()
    schedule = collection_schedule("train") + collection_schedule("validation")
    train: list[StructuredExample] = []
    validation: list[StructuredExample] = []
    games: list[PilotGameSummary] = []
    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        identity = _validate_identity(client.identity)
        for index, item in enumerate(schedule):
            examples, summary = collect_game(client, item)
            if client.identity != identity:
                raise ValueError("pilot semantic identity changed between games")
            (train if item.partition == "train" else validation).extend(examples)
            games.append(summary)
            elapsed = time.monotonic() - started
            if elapsed > _COLLECTION_DEADLINE_SECONDS:
                raise TimeoutError(
                    f"pilot collection exceeded deadline after {index + 1}/{len(schedule)} "
                    f"games and {elapsed:.1f} seconds"
                )
    return PilotCollection(identity, tuple(train), tuple(validation), tuple(games))


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def _is_reparse(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _require_collection(collection: PilotCollection) -> None:
    if type(collection) is not PilotCollection:
        raise TypeError("collection must be PilotCollection")
    identity = _validate_identity(collection.identity)
    if not collection.train or not collection.validation:
        raise ValueError("pilot collection partitions must not be empty")
    expected_games = collection_schedule("train") + collection_schedule("validation")
    if tuple(game.schedule for game in collection.games) != expected_games:
        raise ValueError("pilot collection game schedule is noncanonical")
    for partition, examples in (("train", collection.train),
                                ("validation", collection.validation)):
        allowed = {(
            item.profile_id, item.episode_seed, item.learner_seat,
        ) for item in collection_schedule(partition)}
        for example in examples:
            if type(example) is not StructuredExample:
                raise TypeError("pilot collection rows must be StructuredExample")
            if (example.scenario_id, example.contract_hash, example.encoding_hash,
                    example.capacity_hash) != (
                identity.scenario_id, identity.contract_hash,
                identity.encoding_hash, identity.capacity_hash,
            ):
                raise ValueError("pilot collection contains mixed identities")
            if (example.profile_id, example.episode_seed, example.learner_seat) not in allowed:
                raise ValueError("pilot collection row is outside the frozen schedule")


def write_collection_evidence(
    output: Path,
    collection: PilotCollection,
) -> PilotCollectionEvidence:
    _require_collection(collection)
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"pilot evidence output already exists: {output}")
    if not output.parent.is_dir() or _is_reparse(output.parent):
        raise ValueError("pilot evidence parent must be a plain directory")

    train_bytes = b"".join(_canonical_bytes(asdict(example)) for example in collection.train)
    validation_bytes = b"".join(
        _canonical_bytes(asdict(example)) for example in collection.validation
    )
    train_hash = hashlib.sha256(train_bytes).hexdigest()
    validation_hash = hashlib.sha256(validation_bytes).hexdigest()
    manifest = {
        "schema_version": 1,
        "label_source": "bounded-search-v1",
        "identity": {
            "scenario_id": collection.identity.scenario_id,
            "contract_hash": collection.identity.contract_hash,
            "encoding_hash": collection.identity.encoding_hash,
            "capacity_hash": collection.identity.capacity_hash,
            "environment_kind": "duel",
        },
        "profiles": list(PILOT_PROFILES),
        "teacher": {
            "search_depth": 4,
            "expansion_budget": 512,
            "heuristic_identity": "material-plus-pursuit-v1",
        },
        "train": {"games": 28, "examples": len(collection.train), "sha256": train_hash},
        "validation": {
            "games": 14, "examples": len(collection.validation),
            "sha256": validation_hash,
        },
        "games": [asdict(game) for game in collection.games],
    }
    manifest_bytes = _canonical_bytes(manifest)
    output.mkdir()
    if _is_reparse(output):
        raise ValueError("pilot evidence output must be a plain directory")
    for name, data in (
        ("train.jsonl", train_bytes),
        ("validation.jsonl", validation_bytes),
        ("collection.json", manifest_bytes),
    ):
        with (output / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return PilotCollectionEvidence(
        train_hash,
        validation_hash,
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
