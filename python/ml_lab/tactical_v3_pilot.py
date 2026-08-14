"""Frozen schedules and complete-game evidence for the tactical-v3 training pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import time
from typing import Literal

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from .tactical_v3_batching import collate_decisions, collate_examples
from .tactical_v3_checkpoint import (
    LoadedStructuredPolicy,
    StructuredCheckpointMetadata,
    load_structured_checkpoint,
    save_structured_checkpoint,
    structured_model_state_sha256,
)
from .tactical_v3_client import (
    CandidateSelection,
    SelectiveDaggerInspection,
    TacticalV3GymClient,
    TeacherSelection,
)
from .tactical_v3_corpus import (
    StructuredExample,
    StructuredTarget,
    TeacherEvidence,
    _ROW_FIELDS,
    _TEACHER_FIELDS,
    _exact_mapping,
    _int,
    _parse_target,
    _text,
)
from .tactical_v3_layers import TacticalV3ModelConfig
from .tactical_v3_model import TacticalV3Policy
from .tactical_v3_objectives import ObjectiveConfig
from .tactical_v3_schema import (
    TacticalV3Decision,
    TacticalV3SemanticIdentity,
    TacticalV3View,
    parse_decision,
)
from .tactical_v3_training import (
    EpochMetrics,
    StepMetrics,
    TrainerConfig,
    _batch_to_device,
    train_offline,
)


PILOT_PROFILES = (
    "standard-3v3",
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)

SELECTIVE_DAGGER_CONVERSION_PROFILES = (
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)

_SELECTIVE_DAGGER_SEED_BANKS = {
    ("train", 1): (18_000_000, 18_099_999, 2_000),
    ("train", 2): (18_100_000, 18_199_999, 2_000),
    ("train", 3): (18_200_000, 18_299_999, 2_000),
    ("validation", 1): (19_000_000, 19_009_999, 200),
    ("validation", 2): (19_010_000, 19_019_999, 200),
    ("validation", 3): (19_020_000, 19_029_999, 200),
}

_COLLECTION_DEADLINE_SECONDS = 4 * 60 * 60
_TRAINING_DEADLINE_SECONDS = 2 * 60 * 60
_EVALUATION_DEADLINE_SECONDS = 4 * 60 * 60


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
class PilotDaggerDecision:
    example: StructuredExample
    learner_candidate_id: int
    teacher_candidate_id: int
    disagreement: bool
    teacher_intervened: bool
    eligibility_reasons: tuple[str, ...]
    state_hash: str
    state_occurrence: int
    normalized_advantage: float
    opponent_living_unit_count: int
    productive_legal_action_count: int


@dataclass(frozen=True, slots=True)
class PilotDaggerGameSummary:
    schedule: PilotScheduleItem
    winner: int
    terminated: bool
    truncated: bool
    decisions: int
    disagreements: int
    teacher_interventions: int
    internal_fallback_count: int


@dataclass(frozen=True, slots=True)
class PilotDaggerEpisode:
    identity: TacticalV3SemanticIdentity
    records: tuple[PilotDaggerDecision, ...]
    summary: PilotDaggerGameSummary
    actor_model_state_sha256: str
    actor_corpus_sha256: str
    actor_best_epoch: int
    actor_best_validation_policy_nll: float


@dataclass(frozen=True, slots=True)
class PilotDaggerTrainingSet:
    identity: TacticalV3SemanticIdentity
    train: tuple[StructuredExample, ...]
    validation: tuple[StructuredExample, ...]
    episodes: tuple[PilotDaggerEpisode, ...]
    base_collection_sha256: str
    dagger_records_sha256: str
    validation_sha256: str
    corpus_sha256: str


@dataclass(frozen=True, slots=True)
class SelectiveDaggerPartitionCollection:
    partition: Literal["train", "validation"]
    iteration: int
    episodes: tuple[PilotDaggerEpisode, ...]
    label_count: int
    game_count: int
    label_target: int
    game_ceiling: int


@dataclass(frozen=True, slots=True)
class StructuredDaggerMixtureBatch:
    examples: tuple[StructuredExample, ...]
    sources: tuple[
        Literal["greedy_standard", "search_conversion", "dagger_targeted"], ...
    ]


class _StructuredRowCycler:
    def __init__(
        self, rows: tuple[StructuredExample, ...], generator: torch.Generator,
    ) -> None:
        if not rows:
            raise ValueError("structured source pool must not be empty")
        self._rows = rows
        self._generator = generator
        self._order: list[int] = []
        self._offset = 0

    def take(self, count: int) -> tuple[StructuredExample, ...]:
        selected: list[StructuredExample] = []
        while len(selected) < count:
            if self._offset == len(self._order):
                self._order = torch.randperm(
                    len(self._rows), generator=self._generator,
                ).tolist()
                self._offset = 0
            available = min(count - len(selected), len(self._order) - self._offset)
            selected.extend(
                self._rows[index]
                for index in self._order[self._offset:self._offset + available]
            )
            self._offset += available
        return tuple(selected)


class StructuredDaggerMixtureSampler:
    _SOURCES = ("greedy_standard", "search_conversion", "dagger_targeted")
    _FRACTIONS = (0.49, 0.21, 0.30)

    def __init__(
        self, training_set: PilotDaggerTrainingSet, *, batch_size: int, seed: int,
    ) -> None:
        if type(training_set) is not PilotDaggerTrainingSet:
            raise TypeError("training_set must be PilotDaggerTrainingSet")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive int")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative int")
        targeted_count = sum(len(episode.records) for episode in training_set.episodes)
        base_count = len(training_set.train) - targeted_count
        if base_count < 1 or targeted_count < 1:
            raise ValueError("structured DAgger mixture requires base and targeted rows")
        base = training_set.train[:base_count]
        targeted = training_set.train[base_count:]
        pools = (
            tuple(row for row in base if row.profile_id == "standard-3v3"),
            tuple(row for row in base if row.profile_id != "standard-3v3"),
            targeted,
        )
        generators = []
        for offset in range(4):
            generator = torch.Generator(device="cpu")
            generator.manual_seed((seed + offset) % (2**63 - 1))
            generators.append(generator)
        self._cyclers = tuple(
            _StructuredRowCycler(rows, generators[index])
            for index, rows in enumerate(pools)
        )
        self._shuffle = generators[-1]
        self._batch_size = batch_size
        self._carry = [0.0, 0.0, 0.0]

    def next_batch(self) -> StructuredDaggerMixtureBatch:
        targets = [
            self._batch_size * fraction + carry
            for fraction, carry in zip(self._FRACTIONS, self._carry, strict=True)
        ]
        counts = [math.floor(target) for target in targets]
        while sum(counts) < self._batch_size:
            source = max(
                range(len(counts)),
                key=lambda index: (targets[index] - counts[index], -index),
            )
            counts[source] += 1
        self._carry = [
            target - count for target, count in zip(targets, counts, strict=True)
        ]
        examples = []
        sources = []
        for source, cycler, count in zip(
            self._SOURCES, self._cyclers, counts, strict=True,
        ):
            examples.extend(cycler.take(count))
            sources.extend((source,) * count)
        order = torch.randperm(
            self._batch_size, generator=self._shuffle,
        ).tolist()
        return StructuredDaggerMixtureBatch(
            tuple(examples[index] for index in order),
            tuple(sources[index] for index in order),
        )


@dataclass(frozen=True, slots=True)
class PilotCollectionEvidence:
    train_sha256: str
    validation_sha256: str
    collection_sha256: str


@dataclass(frozen=True, slots=True)
class PolicyTargetMetrics:
    policy_nll: float
    policy_accuracy: float
    finite_logit_count: int
    valid_logit_count: int


@dataclass(frozen=True, slots=True)
class PilotTrainingArtifacts:
    checkpoint: Path
    initial_train: PolicyTargetMetrics
    initial_validation: PolicyTargetMetrics
    restored_train: PolicyTargetMetrics
    restored_validation: PolicyTargetMetrics
    loaded: LoadedStructuredPolicy
    duration_seconds: float


@dataclass(frozen=True, slots=True)
class PilotEvaluationGame:
    controller: Literal["model", "random"]
    schedule: PilotScheduleItem
    winner: int
    outcome: Literal["win", "loss", "draw"]
    terminated: bool
    truncated: bool
    decisions: int
    candidate_errors: int
    internal_fallback_count: int


@dataclass(frozen=True, slots=True)
class PilotEvaluationSummary:
    games: int
    wins: int
    losses: int
    draws: int
    truncations: int
    mean_decisions: float
    candidate_errors: int
    internal_fallback_count: int
    win_rate: float


@dataclass(frozen=True, slots=True)
class PilotEvaluation:
    controller: Literal["model", "random"]
    games: tuple[PilotEvaluationGame, ...]
    aggregate: PilotEvaluationSummary
    profiles: tuple[tuple[str, PilotEvaluationSummary], ...]


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


def diagnostic_evaluation_schedule() -> tuple[PilotScheduleItem, ...]:
    return tuple(
        PilotScheduleItem("evaluation", profile, 64_000_000 + profile_index * 5 + offset,
                          seat, seat)
        for profile_index, profile in enumerate(PILOT_PROFILES)
        for offset in range(5)
        for seat in (0, 1)
    )


def dagger_iteration_schedule() -> tuple[PilotScheduleItem, ...]:
    return tuple(
        PilotScheduleItem("train", "standard-3v3", 65_100_000, seat, seat)
        for seat in (0, 1)
    )


def selective_dagger_schedule(
    partition: Literal["train", "validation"], iteration: int,
) -> tuple[PilotScheduleItem, ...]:
    key = (partition, iteration)
    if key not in _SELECTIVE_DAGGER_SEED_BANKS:
        raise ValueError("selective DAgger partition or iteration is not frozen")
    start, stop, game_ceiling = _SELECTIVE_DAGGER_SEED_BANKS[key]
    if game_ceiling % 2:
        raise ValueError("selective DAgger game ceiling must contain reciprocal pairs")
    pair_count = game_ceiling // 2
    if start + pair_count - 1 > stop:
        raise ValueError("selective DAgger seed bank is too small")

    scheduled: list[PilotScheduleItem] = []
    standard_residual = 0
    conversion_index = 0
    for pair_index in range(pair_count):
        standard_residual += 7
        standard_count = standard_residual // 10
        standard_residual -= standard_count * 10
        if standard_count:
            profile = "standard-3v3"
        else:
            profile = SELECTIVE_DAGGER_CONVERSION_PROFILES[
                conversion_index % len(SELECTIVE_DAGGER_CONVERSION_PROFILES)
            ]
            conversion_index += 1
        seed = start + pair_index
        for seat in (0, 1):
            scheduled.append(PilotScheduleItem(
                partition, profile, seed, seat, seat,
            ))
    return tuple(scheduled)


def collect_selective_dagger_partition(
    partition: Literal["train", "validation"],
    iteration: int,
    collect_game: Callable[[PilotScheduleItem], PilotDaggerEpisode],
) -> SelectiveDaggerPartitionCollection:
    if not callable(collect_game):
        raise TypeError("collect_game must be callable")
    schedule = selective_dagger_schedule(partition, iteration)
    label_target = 20_000 if partition == "train" else 2_000
    game_ceiling = len(schedule)
    episodes: list[PilotDaggerEpisode] = []
    label_count = 0
    for pair_start in range(0, game_ceiling, 2):
        pair = schedule[pair_start:pair_start + 2]
        for item in pair:
            episode = collect_game(item)
            if type(episode) is not PilotDaggerEpisode:
                raise TypeError("selective DAgger collector returned an invalid episode")
            if episode.summary.schedule != item:
                raise ValueError("selective DAgger episode schedule drifted")
            if episode.summary.decisions != len(episode.records):
                raise ValueError("selective DAgger episode label count drifted")
            episodes.append(episode)
            label_count += len(episode.records)
        if label_count >= label_target:
            break
    if label_count < label_target:
        raise RuntimeError(
            f"selective DAgger did not reach {label_target:,} labels before "
            f"the {game_ceiling:,}-game ceiling"
        )
    return SelectiveDaggerPartitionCollection(
        partition, iteration, tuple(episodes), label_count, len(episodes),
        label_target, game_ceiling,
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
        standard = item.profile_id == "standard-3v3"
        result = (
            client.duel_greedy_step(decision.decision_id)
            if standard else client.duel_oracle_step(decision.decision_id)
        )
        if standard:
            if (
                result.selection.search_depth != 0
                or result.selection.expansion_budget != 0
                or result.selection.actual_expansions != 0
                or result.selection.heuristic_identity != "greedy-one-ply-v1"
            ):
                raise ValueError("pilot Greedy teacher metadata drifted")
        else:
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
                ("greedy-one-ply-v1" if item.profile_id == "standard-3v3"
                 else "bounded-search-v1"), selection.search_depth,
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


def collect_dagger_game(
    client: TacticalV3GymClient,
    loaded: LoadedStructuredPolicy,
    item: PilotScheduleItem,
) -> PilotDaggerEpisode:
    if type(item) is not PilotScheduleItem:
        raise TypeError("item must be PilotScheduleItem")
    if item.partition == "evaluation":
        raise ValueError("evaluation schedule is not DAgger collection evidence")
    identity = _validate_identity(client.identity)
    if loaded.metadata.identity != identity:
        raise ValueError("DAgger policy identity does not match GymServer identity")
    if (
        re.fullmatch(r"[0-9a-f]{64}", loaded.metadata.model_state_sha256) is None
        or re.fullmatch(r"[0-9a-f]{64}", loaded.metadata.corpus_sha256) is None
        or type(loaded.metadata.best_epoch) is not int
        or loaded.metadata.best_epoch < 0
        or not math.isfinite(loaded.metadata.best_validation_policy_nll)
    ):
        raise ValueError("DAgger actor provenance is invalid")
    p0, p1 = ("external", "random") if item.learner_seat == 0 else ("random", "external")
    view = client.duel_reset(
        item.episode_seed, p0, p1, item.learner_seat,
        item.profile_id, item.reference_seat,
    )
    retained: list[tuple[
        TacticalV3Decision, TeacherSelection, int,
        SelectiveDaggerInspection, int,
    ]] = []
    emitted_state_hashes: set[str] = set()
    seen: set[int] = set()
    learner_decision_index = 0
    while not view.terminated and not view.truncated:
        _validate_view(view, item, identity, client.identity, seen)
        decision = view.decision
        seen.add(decision.decision_id)
        batch = collate_decisions(
            (decision,),
            loaded.model.config.horizon_turns,
        )
        learner = loaded.model.select(batch)[0]
        if learner.decision_id != decision.decision_id:
            raise ValueError("DAgger learner selected a stale decision")
        if sum(candidate.candidate_id == learner.candidate_id
               for candidate in decision.candidates) != 1:
            raise ValueError("DAgger learner candidate must occur exactly once")
        inspection = client.duel_dagger_inspect(
            decision.decision_id, learner.candidate_id,
        )
        if (
            inspection.decision_id != decision.decision_id
            or inspection.learner_candidate_id != learner.candidate_id
        ):
            raise ValueError("selective DAgger inspection identity drifted")
        if inspection.reasons and inspection.state_hash not in emitted_state_hashes:
            teacher = client.duel_oracle_query(decision.decision_id)
            _validate_selection(teacher, decision)
            retained.append((
                decision, teacher, learner.candidate_id,
                inspection, learner_decision_index,
            ))
            emitted_state_hashes.add(inspection.state_hash)
        view = client.duel_step(CandidateSelection(
            learner.decision_id,
            learner.candidate_id,
        ))
        learner_decision_index += 1
    if client.identity != identity or loaded.metadata.identity != identity:
        raise ValueError("DAgger semantic identity drifted during collection")
    if view.start_profile != item.profile_id or view.reference_seat != item.reference_seat:
        raise ValueError("DAgger terminal profile or reference seat drifted")
    fallback = client.duel_status()
    if fallback != 0:
        raise ValueError("DAgger internal fallback count must remain zero")

    outcome: Literal["win", "loss", "draw"]
    if view.winner == item.learner_seat:
        outcome = "win"
    elif view.winner in {0, 1}:
        outcome = "loss"
    else:
        outcome = "draw"
    records = tuple(
        PilotDaggerDecision(
            StructuredExample(
                1,
                decision,
                StructuredTarget(
                    teacher.candidate_id,
                    outcome,
                    decision_index,
                    len(retained) - index if outcome == "win" else None,
                    view.truncated,
                ),
                TeacherEvidence(
                    "bounded-search-v1", teacher.search_depth,
                    teacher.expansion_budget, teacher.actual_expansions,
                    teacher.heuristic_identity, None,
                ),
                identity.scenario_id,
                identity.contract_hash,
                identity.encoding_hash,
                identity.capacity_hash,
                item.profile_id,
                item.episode_seed,
                item.learner_seat,
            ),
            learner_candidate_id,
            teacher.candidate_id,
            learner_candidate_id != teacher.candidate_id,
            False,
            inspection.reasons,
            inspection.state_hash,
            inspection.state_occurrence,
            inspection.normalized_advantage,
            inspection.opponent_living_unit_count,
            inspection.productive_legal_action_count,
        )
        for index, (
            decision, teacher, learner_candidate_id, inspection, decision_index,
        ) in enumerate(retained)
    )
    disagreements = sum(record.disagreement for record in records)
    summary = PilotDaggerGameSummary(
        item,
        view.winner,
        view.terminated,
        view.truncated,
        len(records),
        disagreements,
        0,
        fallback,
    )
    return PilotDaggerEpisode(
        identity,
        records,
        summary,
        loaded.metadata.model_state_sha256,
        loaded.metadata.corpus_sha256,
        loaded.metadata.best_epoch,
        loaded.metadata.best_validation_policy_nll,
    )


def write_dagger_episode(output: Path, episode: PilotDaggerEpisode) -> Path:
    if type(episode) is not PilotDaggerEpisode or not episode.records:
        raise ValueError("DAgger episode must contain records")
    if episode.summary.decisions != len(episode.records):
        raise ValueError("DAgger episode summary decision count changed")
    if episode.summary.disagreements != sum(
        record.disagreement for record in episode.records
    ):
        raise ValueError("DAgger episode disagreement count changed")
    if episode.summary.teacher_interventions != 0 or any(
        record.teacher_intervened for record in episode.records
    ):
        raise ValueError("DAgger evidence must be collected under learner control")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"DAgger episode output already exists: {output}")
    if not output.parent.is_dir() or _is_reparse(output.parent):
        raise ValueError("DAgger episode parent must be a plain directory")

    rows = b"".join(_canonical_bytes(asdict(record)) for record in episode.records)
    rows_hash = hashlib.sha256(rows).hexdigest()
    manifest = {
        "schema_version": 1,
        "kind": "tactical-v3-dagger-episode",
        "identity": {
            "scenario_id": episode.identity.scenario_id,
            "contract_hash": episode.identity.contract_hash,
            "encoding_hash": episode.identity.encoding_hash,
            "capacity_hash": episode.identity.capacity_hash,
            "environment_kind": episode.identity.environment_kind,
        },
        "actor": {
            "algorithm": "structured_imitation",
            "model_state_sha256": episode.actor_model_state_sha256,
            "corpus_sha256": episode.actor_corpus_sha256,
            "best_epoch": episode.actor_best_epoch,
            "best_validation_policy_nll":
                episode.actor_best_validation_policy_nll,
        },
        "teacher": {
            "identity": "bounded-search-v1",
            "search_depth": 4,
            "expansion_budget": 512,
            "heuristic_identity": "material-plus-pursuit-v1",
        },
        "records": {
            "path": "decisions.jsonl",
            "count": len(episode.records),
            "sha256": rows_hash,
        },
        "summary": asdict(episode.summary),
    }
    output.mkdir()
    if _is_reparse(output):
        raise ValueError("DAgger episode output must be a plain directory")
    for name, data in (
        ("decisions.jsonl", rows),
        ("episode.json", _canonical_bytes(manifest)),
    ):
        with (output / name).open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    return output


def build_dagger_training_set(
    collection: PilotCollection,
    evidence: PilotCollectionEvidence,
    episodes: tuple[PilotDaggerEpisode, ...],
) -> PilotDaggerTrainingSet:
    _require_collection(collection)
    if type(evidence) is not PilotCollectionEvidence:
        raise TypeError("evidence must be PilotCollectionEvidence")
    for name in (
        "train_sha256", "validation_sha256", "collection_sha256",
    ):
        if re.fullmatch(r"[0-9a-f]{64}", getattr(evidence, name)) is None:
            raise ValueError("pilot collection evidence hashes must be lowercase SHA-256")
    if type(episodes) is not tuple or not episodes:
        raise ValueError("DAgger iteration episodes must be a nonempty tuple")
    expected_schedule = dagger_iteration_schedule()
    if tuple(episode.summary.schedule for episode in episodes) != expected_schedule:
        raise ValueError("DAgger iteration must use the frozen reciprocal schedule")

    actor = episodes[0]
    appended: list[StructuredExample] = []
    rows = bytearray()
    for episode in episodes:
        if type(episode) is not PilotDaggerEpisode:
            raise TypeError("DAgger iteration rows must be PilotDaggerEpisode")
        if episode.identity != collection.identity:
            raise ValueError("DAgger iteration contains mixed semantic identities")
        if (
            episode.actor_model_state_sha256 != actor.actor_model_state_sha256
            or episode.actor_corpus_sha256 != actor.actor_corpus_sha256
            or episode.actor_best_epoch != actor.actor_best_epoch
            or episode.actor_best_validation_policy_nll
            != actor.actor_best_validation_policy_nll
        ):
            raise ValueError("DAgger iteration actor provenance changed between seats")
        if episode.actor_corpus_sha256 != evidence.collection_sha256:
            raise ValueError("DAgger actor corpus does not match the base collection")
        if (
            not episode.records
            or episode.summary.decisions != len(episode.records)
            or episode.summary.disagreements
            != sum(record.disagreement for record in episode.records)
            or episode.summary.teacher_interventions != 0
            or episode.summary.internal_fallback_count != 0
            or any(record.teacher_intervened for record in episode.records)
        ):
            raise ValueError("DAgger iteration episode evidence is inconsistent")
        for record in episode.records:
            if type(record) is not PilotDaggerDecision:
                raise TypeError("DAgger iteration records must be PilotDaggerDecision")
            appended.append(record.example)
            rows.extend(_canonical_bytes(asdict(record)))

    combined = collection.train + tuple(appended)
    keys = tuple((
        example.scenario_id,
        example.episode_seed,
        example.learner_seat,
        example.profile_id,
        example.decision.decision_id,
    ) for example in combined)
    if len(set(keys)) != len(keys):
        raise ValueError("DAgger iteration introduces duplicate training decisions")
    records_hash = hashlib.sha256(rows).hexdigest()
    corpus_manifest = _dagger_training_manifest(
        evidence.collection_sha256,
        len(collection.train),
        records_hash,
        len(appended),
        evidence.validation_sha256,
        len(collection.validation),
    )
    return PilotDaggerTrainingSet(
        collection.identity,
        combined,
        collection.validation,
        episodes,
        evidence.collection_sha256,
        records_hash,
        evidence.validation_sha256,
        hashlib.sha256(_canonical_bytes(corpus_manifest)).hexdigest(),
    )


def _dagger_training_manifest(
    base_collection_sha256: str,
    base_train_examples: int,
    dagger_records_sha256: str,
    dagger_examples: int,
    validation_sha256: str,
    validation_examples: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "tactical-v3-dagger-training-set",
        "base_collection_sha256": base_collection_sha256,
        "base_train_examples": base_train_examples,
        "dagger_records_sha256": dagger_records_sha256,
        "dagger_examples": dagger_examples,
        "validation_sha256": validation_sha256,
        "validation_examples": validation_examples,
    }


def train_dagger_pilot(
    training_set: PilotDaggerTrainingSet,
    incoming: LoadedStructuredPolicy,
    output: Path,
    seed: int,
    device: str,
) -> PilotTrainingArtifacts:
    if type(training_set) is not PilotDaggerTrainingSet:
        raise TypeError("training_set must be PilotDaggerTrainingSet")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"DAgger iteration output already exists: {output}")
    if not output.parent.is_dir() or _is_reparse(output.parent):
        raise ValueError("DAgger iteration parent must be a plain directory")
    output.mkdir()
    if _is_reparse(output):
        raise ValueError("DAgger iteration output must be a plain directory")
    for episode in training_set.episodes:
        write_dagger_episode(
            output / f"seat-{episode.summary.schedule.learner_seat}",
            episode,
        )
    manifest = _dagger_training_manifest(
        training_set.base_collection_sha256,
        len(training_set.train) - sum(
            len(episode.records) for episode in training_set.episodes
        ),
        training_set.dagger_records_sha256,
        sum(len(episode.records) for episode in training_set.episodes),
        training_set.validation_sha256,
        len(training_set.validation),
    )
    manifest_bytes = _canonical_bytes(manifest)
    if hashlib.sha256(manifest_bytes).hexdigest() != training_set.corpus_sha256:
        raise ValueError("DAgger training-set evidence hash changed")
    with (output / "training-set.json").open("xb") as handle:
        handle.write(manifest_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    training_output = output / "training"
    training_output.mkdir()
    if incoming.metadata.identity != training_set.identity:
        raise ValueError("incoming DAgger actor identity does not match training set")
    trainer_config = TrainerConfig(
        seed=seed,
        batch_size=256,
        learning_rate=3e-4,
        max_epochs=50,
        patience_epochs=5,
        gradient_clip_norm=1.0,
        device=device,
    )
    batch_provider = StructuredDaggerMixtureSampler(
        training_set, batch_size=trainer_config.batch_size, seed=seed,
    )
    return _train_pilot_dataset(
        training_set.identity,
        training_set.train,
        training_set.validation,
        training_set.corpus_sha256,
        training_output,
        seed,
        device,
        initial_policy=incoming,
        trainer_config=trainer_config,
        batch_provider=batch_provider,
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
            expected_source = (
                "greedy-one-ply-v1"
                if example.profile_id == "standard-3v3"
                else "bounded-search-v1"
            )
            if example.teacher.identity != expected_source:
                raise ValueError("pilot collection row has the wrong profile label source")


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
        "schema_version": 2,
        "label_sources": {
            "standard-3v3": "greedy-one-ply-v1",
            "conversion-profiles": "bounded-search-v1",
        },
        "identity": {
            "scenario_id": collection.identity.scenario_id,
            "contract_hash": collection.identity.contract_hash,
            "encoding_hash": collection.identity.encoding_hash,
            "capacity_hash": collection.identity.capacity_hash,
            "environment_kind": "duel",
        },
        "profiles": list(PILOT_PROFILES),
        "teachers": {
            "greedy": {
                "identity": "greedy-one-ply-v1",
            },
            "bounded_search": {
                "identity": "bounded-search-v1",
                "search_depth": 4,
                "expansion_budget": 512,
                "heuristic_identity": "material-plus-pursuit-v1",
            },
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


def load_collection_evidence(
    output: Path,
    identity: TacticalV3SemanticIdentity,
) -> tuple[PilotCollection, PilotCollectionEvidence]:
    identity = _validate_identity(identity)
    output = Path(output)
    if not output.is_dir() or _is_reparse(output):
        raise ValueError("pilot evidence output must be a plain directory")
    if {path.name for path in output.iterdir() if path.is_file()} < {
        "train.jsonl", "validation.jsonl", "collection.json",
    }:
        raise ValueError("pilot collection evidence is incomplete")

    manifest_bytes = (output / "collection.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if type(manifest) is not dict or manifest_bytes != _canonical_bytes(manifest):
        raise ValueError("pilot collection manifest must be canonical JSON")
    if set(manifest) != {
        "schema_version", "label_sources", "identity", "profiles", "teachers",
        "train", "validation", "games",
    }:
        raise ValueError("pilot collection manifest fields are not exact")
    expected_identity = {
        "scenario_id": identity.scenario_id,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
        "environment_kind": "duel",
    }
    if (
        manifest["schema_version"] != 2
        or manifest["label_sources"] != {
            "standard-3v3": "greedy-one-ply-v1",
            "conversion-profiles": "bounded-search-v1",
        }
        or manifest["identity"] != expected_identity
        or manifest["profiles"] != list(PILOT_PROFILES)
        or manifest["teachers"] != {
            "greedy": {
                "identity": "greedy-one-ply-v1",
            },
            "bounded_search": {
                "identity": "bounded-search-v1",
                "search_depth": 4,
                "expansion_budget": 512,
                "heuristic_identity": "material-plus-pursuit-v1",
            },
        }
    ):
        raise ValueError("pilot collection manifest does not match the frozen contract")

    def load_partition(name: str) -> tuple[StructuredExample, ...]:
        metadata = manifest[name]
        if type(metadata) is not dict or set(metadata) != {"games", "examples", "sha256"}:
            raise ValueError(f"pilot {name} metadata fields are not exact")
        path = output / f"{name}.jsonl"
        digest = hashlib.sha256()
        rows = []
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                digest.update(line)
                if not line.endswith(b"\n") or line == b"\n":
                    raise ValueError(f"pilot {name} row {index} is not canonical JSONL")
                try:
                    raw = json.loads(line[:-1].decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise ValueError(f"pilot {name} row {index} is invalid JSON") from error
                if line != _canonical_bytes(raw):
                    raise ValueError(f"pilot {name} row {index} is not canonical JSON")
                rows.append(_parse_pilot_row(raw, identity))
        if digest.hexdigest() != metadata["sha256"] or len(rows) != metadata["examples"]:
            raise ValueError(f"pilot {name} evidence does not match its manifest")
        return tuple(rows)

    games = []
    for raw in manifest["games"]:
        if type(raw) is not dict or set(raw) != {
            "schedule", "winner", "terminated", "truncated", "decisions",
            "internal_fallback_count",
        }:
            raise ValueError("pilot game summary fields are not exact")
        schedule = raw["schedule"]
        if type(schedule) is not dict or set(schedule) != {
            "partition", "profile_id", "episode_seed", "learner_seat",
            "reference_seat",
        }:
            raise ValueError("pilot game schedule fields are not exact")
        games.append(PilotGameSummary(
            PilotScheduleItem(**schedule),
            raw["winner"],
            raw["terminated"],
            raw["truncated"],
            raw["decisions"],
            raw["internal_fallback_count"],
        ))
    collection = PilotCollection(
        identity,
        load_partition("train"),
        load_partition("validation"),
        tuple(games),
    )
    _require_collection(collection)
    if len(collection.games) != len(manifest["games"]):
        raise ValueError("pilot game count does not match manifest")
    evidence = PilotCollectionEvidence(
        manifest["train"]["sha256"],
        manifest["validation"]["sha256"],
        hashlib.sha256(manifest_bytes).hexdigest(),
    )
    return collection, evidence


def _parse_pilot_row(
    value: object,
    identity: TacticalV3SemanticIdentity,
) -> StructuredExample:
    data = _exact_mapping(value, _ROW_FIELDS, "pilot example")
    if _int(data["example_schema_version"], "example.example_schema_version") != 1:
        raise ValueError("pilot example schema version is unsupported")
    for field, expected in (
        ("scenario_id", identity.scenario_id),
        ("contract_hash", identity.contract_hash),
        ("encoding_hash", identity.encoding_hash),
        ("capacity_hash", identity.capacity_hash),
    ):
        if data[field] != expected:
            raise ValueError(f"pilot example {field} does not match collection identity")
    decision = parse_decision(data["decision"], identity)
    target = _parse_target(data["target"], decision)
    teacher_data = _exact_mapping(data["teacher"], _TEACHER_FIELDS, "teacher")
    teacher = TeacherEvidence(
        _text(teacher_data["identity"], "teacher.identity"),
        _int(teacher_data["search_depth"], "teacher.search_depth", nonnegative=True),
        _int(teacher_data["expansion_budget"], "teacher.expansion_budget", nonnegative=True),
        _int(teacher_data["actual_expansions"], "teacher.actual_expansions", nonnegative=True),
        _text(teacher_data["heuristic_identity"], "teacher.heuristic_identity"),
        teacher_data["confidence"],
    )
    profile_id = _text(data["profile_id"], "example.profile_id")
    if profile_id == "standard-3v3":
        valid_teacher = (
            teacher.identity == "greedy-one-ply-v1"
            and teacher.search_depth == 0
            and teacher.expansion_budget == 0
            and teacher.actual_expansions == 0
            and teacher.heuristic_identity == "greedy-one-ply-v1"
            and teacher.confidence is None
        )
    else:
        valid_teacher = (
            profile_id in SELECTIVE_DAGGER_CONVERSION_PROFILES
            and teacher.identity == "bounded-search-v1"
            and teacher.search_depth == 4
            and teacher.expansion_budget == 512
            and 1 <= teacher.actual_expansions <= 512
            and teacher.heuristic_identity == "material-plus-pursuit-v1"
            and teacher.confidence is None
        )
    if not valid_teacher:
        raise ValueError("pilot teacher evidence does not match its profile source")
    learner_seat = _int(data["learner_seat"], "example.learner_seat")
    if learner_seat not in {0, 1} or decision.seat != learner_seat:
        raise ValueError("pilot learner seat must match decision seat")
    return StructuredExample(
        1,
        decision,
        target,
        teacher,
        identity.scenario_id,
        identity.contract_hash,
        identity.encoding_hash,
        identity.capacity_hash,
        profile_id,
        _int(data["episode_seed"], "example.episode_seed"),
        learner_seat,
    )


def _pilot_configs(
    seed: int,
    device: str,
) -> tuple[TacticalV3ModelConfig, ObjectiveConfig, TrainerConfig]:
    model = TacticalV3ModelConfig(
        hidden_dim=32,
        categorical_dim=8,
        cell_message_rounds=1,
        relation_rounds=1,
        attention_heads=4,
        feed_forward_dim=64,
        candidate_hidden_dim=64,
        horizon_turns=(4, 8, 16),
    )
    objective = ObjectiveConfig(
        policy_coefficient=1.0,
        outcome_coefficient=0.0,
        horizon_coefficient=0.0,
        remaining_turns_coefficient=0.0,
    )
    trainer = TrainerConfig(
        seed=seed,
        batch_size=32,
        learning_rate=0.001,
        max_epochs=2,
        patience_epochs=1,
        gradient_clip_norm=1.0,
        device=device,
    )
    return model, objective, trainer


def _policy_target_metrics(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
    *,
    batch_size: int = 32,
    deadline_monotonic: float | None = None,
    progress_callback=None,
) -> PolicyTargetMetrics:
    if type(model) is not TacticalV3Policy:
        raise TypeError("policy metric model must be TacticalV3Policy")
    if not examples:
        raise ValueError("policy metric examples must not be empty")
    device = next(model.parameters()).device
    model.eval()
    total_nll = 0.0
    correct = 0
    finite = 0
    valid = 0
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
                raise TimeoutError("training deadline reached during policy metrics")
            rows = examples[offset:offset + batch_size]
            batch = _batch_to_device(
                collate_examples(rows, model.config.horizon_turns), device,
            )
            logits = model(batch).candidate_logits
            valid_logits = logits[batch.candidates.mask]
            finite += int(torch.isfinite(valid_logits).sum().item())
            valid += int(valid_logits.numel())
            total_nll += float(F.cross_entropy(
                logits,
                batch.teacher_candidate_index,
                reduction="sum",
            ).item())
            correct += int((
                logits.argmax(dim=1) == batch.teacher_candidate_index
            ).sum().item())
            if progress_callback is not None:
                progress_callback(min(offset + len(rows), len(examples)), len(examples))
    count = len(examples)
    return PolicyTargetMetrics(total_nll / count, correct / count, finite, valid)


def validation_metric_breakdown(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
) -> dict[str, dict[str, object]]:
    if not examples:
        raise ValueError("validation metric examples must not be empty")

    def metric_wire(rows: tuple[StructuredExample, ...]) -> dict[str, object]:
        return {"examples": len(rows), **asdict(_policy_target_metrics(model, rows))}

    result: dict[str, dict[str, object]] = {}
    for profile in PILOT_PROFILES:
        profile_rows = tuple(row for row in examples if row.profile_id == profile)
        if not profile_rows:
            raise ValueError(f"validation metrics are missing profile {profile}")
        seats: dict[str, dict[str, object]] = {}
        for seat in (0, 1):
            seat_rows = tuple(row for row in profile_rows if row.learner_seat == seat)
            if not seat_rows:
                raise ValueError(
                    f"validation metrics are missing profile {profile} seat {seat}"
                )
            seats[str(seat)] = metric_wire(seat_rows)
        result[profile] = {"all": metric_wire(profile_rows), "seats": seats}
    return result


def _history_bytes(history: tuple[EpochMetrics, ...]) -> bytes:
    if type(history) is not tuple or not history:
        raise ValueError("pilot training history must be a non-empty tuple")
    rows = []
    for metric in history:
        if type(metric) is not EpochMetrics:
            raise TypeError("pilot training history must contain EpochMetrics")
        rows.append(_canonical_bytes({
            "epoch": metric.epoch,
            "improved": metric.improved,
            "train": dict(metric.train),
            "validation": dict(metric.validation),
            "validation_policy_nll": metric.validation_policy_nll,
        }))
    return b"".join(rows)


class _PilotTelemetry:
    def __init__(self, output: Path, max_epochs: int, started: float) -> None:
        self.output = output
        self.max_epochs = max_epochs
        self.started = started
        self._handle = (output / "telemetry.jsonl").open("xb")
        self._step_handle = (output / "steps.jsonl").open("xb")
        self._steps_since_sync = 0
        self._writer = SummaryWriter(log_dir=str(output / "tensorboard"), flush_secs=1)
        self._writer.add_scalar("progress/started", 1.0, 0)
        self._writer.flush()
        print("pilot telemetry phase=initial_metrics", flush=True)

    def baseline(
        self,
        train: PolicyTargetMetrics,
        validation: PolicyTargetMetrics,
    ) -> None:
        self._writer.add_scalar("baseline/train_policy_nll", train.policy_nll, 0)
        self._writer.add_scalar(
            "baseline/train_policy_accuracy", train.policy_accuracy, 0,
        )
        self._writer.add_scalar(
            "baseline/validation_policy_nll", validation.policy_nll, 0,
        )
        self._writer.add_scalar(
            "baseline/validation_policy_accuracy", validation.policy_accuracy, 0,
        )
        self._writer.flush()
        print(
            "pilot telemetry baseline "
            f"train_policy_nll={train.policy_nll:.6f} "
            f"validation_policy_nll={validation.policy_nll:.6f}",
            flush=True,
        )

    def progress(self, phase: str, completed: int, total: int) -> None:
        self._writer.add_scalar(f"progress/{phase}_examples", completed, completed)
        self._writer.flush()
        if completed == total or completed % 320 == 0:
            print(
                f"pilot telemetry phase={phase} examples={completed}/{total}",
                flush=True,
            )

    def epoch(self, metric: EpochMetrics) -> None:
        row = _history_bytes((metric,))
        self._handle.write(row)
        self._handle.flush()
        os.fsync(self._handle.fileno())
        step = metric.epoch + 1
        for name, value in metric.train.items():
            self._writer.add_scalar(f"epoch/train_{name}", value, step)
        for name, value in metric.validation.items():
            self._writer.add_scalar(f"epoch/validation_{name}", value, step)
        self._writer.add_scalar(
            "epoch/validation_policy_nll", metric.validation_policy_nll, step,
        )
        self._writer.add_scalar("epoch/improved", float(metric.improved), step)
        self._writer.flush()
        print(
            "pilot telemetry "
            f"epoch={step}/{self.max_epochs} "
            f"elapsed_seconds={time.monotonic() - self.started:.1f} "
            f"train_policy_nll={metric.train['policy']:.6f} "
            f"validation_policy_nll={metric.validation_policy_nll:.6f} "
            f"improved={str(metric.improved).lower()}",
            flush=True,
        )

    def step(self, metric: StepMetrics) -> None:
        row = _canonical_bytes({
            "phase": metric.phase,
            "epoch": metric.epoch,
            "batch_index": metric.batch_index,
            "global_step": metric.global_step,
            "example_count": metric.example_count,
            "metrics": dict(metric.metrics),
        })
        self._step_handle.write(row)
        self._step_handle.flush()
        self._steps_since_sync += 1
        if self._steps_since_sync >= 10:
            os.fsync(self._step_handle.fileno())
            self._steps_since_sync = 0
        for name, value in metric.metrics.items():
            self._writer.add_scalar(
                f"step/{metric.phase}_{name}", value, metric.global_step,
            )
        if metric.global_step == 1 or metric.global_step % 5 == 0:
            self._writer.flush()
        if metric.global_step == 1 or metric.global_step % 10 == 0:
            print(
                "pilot telemetry "
                f"phase={metric.phase} "
                f"epoch={metric.epoch + 1}/{self.max_epochs} "
                f"batch={metric.batch_index + 1} "
                f"step={metric.global_step} "
                f"policy_nll={metric.metrics['policy']:.6f}",
                flush=True,
            )

    def close(self) -> None:
        self._step_handle.flush()
        os.fsync(self._step_handle.fileno())
        self._step_handle.close()
        self._handle.close()
        self._writer.flush()
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _train_pilot_dataset(
    identity: TacticalV3SemanticIdentity,
    train: tuple[StructuredExample, ...],
    validation: tuple[StructuredExample, ...],
    corpus_sha256: str,
    output: Path,
    seed: int,
    device: str,
    *,
    initial_policy: LoadedStructuredPolicy | None = None,
    trainer_config: TrainerConfig | None = None,
    batch_provider: StructuredDaggerMixtureSampler | None = None,
) -> PilotTrainingArtifacts:
    started = time.monotonic()
    identity = _validate_identity(identity)
    if type(train) is not tuple or not train or type(validation) is not tuple or not validation:
        raise ValueError("pilot training partitions must be nonempty tuples")
    if re.fullmatch(r"[0-9a-f]{64}", corpus_sha256) is None:
        raise ValueError("pilot training corpus hash must be lowercase SHA-256")
    if type(seed) is not int or seed != 227:
        raise ValueError("pilot training seed must be exactly 227")
    output = Path(output)
    if not output.is_dir() or _is_reparse(output):
        raise ValueError("pilot training artifacts output must be a plain directory")

    model_config, objective_config, default_trainer_config = _pilot_configs(seed, device)
    if trainer_config is None:
        trainer_config = default_trainer_config
    elif (
        type(trainer_config) is not TrainerConfig
        or trainer_config.seed != seed
        or trainer_config.device != device
    ):
        raise ValueError("pilot trainer override does not match seed/device")
    if batch_provider is not None and type(batch_provider) is not StructuredDaggerMixtureSampler:
        raise TypeError("pilot batch_provider must be StructuredDaggerMixtureSampler")
    initial_state = None
    if initial_policy is not None:
        if type(initial_policy) is not LoadedStructuredPolicy:
            raise TypeError("initial_policy must be LoadedStructuredPolicy")
        if initial_policy.metadata.identity != identity:
            raise ValueError("initial policy identity does not match pilot training identity")
        if initial_policy.model.config != model_config:
            raise ValueError("initial policy architecture does not match pilot model")
        initial_state = {
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in initial_policy.model.state_dict().items()
        }
    deadline = started + _TRAINING_DEADLINE_SECONDS
    metric_device = torch.device(trainer_config.device)
    with _PilotTelemetry(output, trainer_config.max_epochs, started) as telemetry:
        if initial_state is None:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(seed % (2**63 - 1))
                initial_model = TacticalV3Policy(model_config).to(metric_device).eval()
        else:
            initial_model = TacticalV3Policy(model_config).to(metric_device)
            initial_model.load_state_dict(initial_state, strict=True)
            initial_model.eval()
        initial_train = _policy_target_metrics(
            initial_model,
            train,
            deadline_monotonic=deadline,
            progress_callback=lambda completed, total: telemetry.progress(
                "initial_train", completed, total,
            ),
        )
        initial_validation = _policy_target_metrics(
            initial_model,
            validation,
            deadline_monotonic=deadline,
            progress_callback=lambda completed, total: telemetry.progress(
                "initial_validation", completed, total,
            ),
        )
        telemetry.baseline(initial_train, initial_validation)
        del initial_model

        result = train_offline(
            train,
            validation,
            model_config,
            objective_config,
            trainer_config,
            epoch_callback=telemetry.epoch,
            step_callback=telemetry.step,
            deadline_monotonic=deadline,
            initial_state_dict=initial_state,
            training_batch_provider=(
                None if batch_provider is None
                else lambda epoch, batch_index: batch_provider.next_batch().examples
            ),
        )
    elapsed = time.monotonic() - started
    if elapsed > _TRAINING_DEADLINE_SECONDS:
        raise TimeoutError(f"pilot training exceeded deadline after {elapsed:.1f} seconds")

    checkpoint_dir = output / "checkpoints"
    checkpoint = checkpoint_dir / "best.pt"
    metrics_path = output / "metrics.jsonl"
    if checkpoint_dir.exists() or checkpoint_dir.is_symlink() or metrics_path.exists():
        raise FileExistsError("pilot training artifacts already exist")
    checkpoint_dir.mkdir()
    metadata = StructuredCheckpointMetadata(
        format_version=1,
        algorithm="structured_imitation",
        identity=identity,
        model_config=result.model_config,
        objective_config=result.objective_config,
        trainer_config=result.trainer_config,
        corpus_sha256=corpus_sha256,
        model_state_sha256=structured_model_state_sha256(result.model),
        best_epoch=result.best_epoch,
        best_validation_policy_nll=result.best_validation_policy_nll,
        published_device="cpu",
    )
    save_structured_checkpoint(
        checkpoint,
        result.model,
        metadata,
        validation[:2],
    )
    with metrics_path.open("xb") as handle:
        history_bytes = _history_bytes(result.history)
        if (output / "telemetry.jsonl").read_bytes() != history_bytes:
            raise RuntimeError("pilot live telemetry does not match completed history")
        handle.write(history_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_structured_checkpoint(
        checkpoint,
        identity.encoding_hash,
        identity.capacity_hash,
    )
    loaded.model.to(metric_device)
    try:
        restored_train = _policy_target_metrics(
            loaded.model,
            train,
            deadline_monotonic=deadline,
        )
        restored_validation = _policy_target_metrics(
            loaded.model,
            validation,
            deadline_monotonic=deadline,
        )
    finally:
        loaded.model.cpu()
    duration = time.monotonic() - started
    if duration > _TRAINING_DEADLINE_SECONDS:
        raise TimeoutError(f"pilot training exceeded deadline after {duration:.1f} seconds")
    return PilotTrainingArtifacts(
        checkpoint,
        initial_train,
        initial_validation,
        restored_train,
        restored_validation,
        loaded,
        duration,
    )


def train_pilot(
    collection: PilotCollection,
    evidence: PilotCollectionEvidence,
    output: Path,
    seed: int,
    device: str,
    *,
    artifacts_output: Path | None = None,
) -> PilotTrainingArtifacts:
    _require_collection(collection)
    if type(seed) is not int or seed != 227:
        raise ValueError("pilot training seed must be exactly 227")
    if type(evidence) is not PilotCollectionEvidence:
        raise TypeError("evidence must be PilotCollectionEvidence")
    collection_root = Path(output)
    if not collection_root.is_dir() or _is_reparse(collection_root):
        raise ValueError("pilot training output must be the plain collection directory")
    if hashlib.sha256((collection_root / "collection.json").read_bytes()).hexdigest() != (
        evidence.collection_sha256
    ):
        raise ValueError("pilot collection evidence hash does not match output")
    if artifacts_output is None:
        output = collection_root
    else:
        output = Path(artifacts_output)
        if (
            output.parent != collection_root
            or re.fullmatch(r"retry-[1-9][0-9]*", output.name) is None
        ):
            raise ValueError("pilot retry artifacts must use a numbered retry subdirectory")
        if output.exists() or output.is_symlink():
            raise FileExistsError("pilot retry artifacts already exist")
        output.mkdir()

    return _train_pilot_dataset(
        collection.identity,
        collection.train,
        collection.validation,
        evidence.collection_sha256,
        output,
        seed,
        device,
    )


def _evaluation_summary(games: tuple[PilotEvaluationGame, ...]) -> PilotEvaluationSummary:
    count = len(games)
    wins = sum(game.winner == game.schedule.learner_seat for game in games)
    losses = sum(game.winner in {0, 1} and game.winner != game.schedule.learner_seat
                 for game in games)
    draws = count - wins - losses
    truncated = sum(game.truncated for game in games)
    return PilotEvaluationSummary(
        count,
        wins,
        losses,
        draws,
        truncated,
        sum(game.decisions for game in games) / count if count else 0.0,
        sum(game.candidate_errors for game in games),
        sum(game.internal_fallback_count for game in games),
        wins / count if count else 0.0,
    )


def evaluate_pilot(
    client: TacticalV3GymClient,
    loaded: LoadedStructuredPolicy,
    controller: Literal["model", "random"],
    schedule: tuple[PilotScheduleItem, ...],
) -> PilotEvaluation:
    if controller not in {"model", "random"}:
        raise ValueError("pilot evaluation controller must be model or random")
    if tuple(schedule) not in {
        evaluation_schedule(), diagnostic_evaluation_schedule(),
    }:
        raise ValueError("pilot evaluation schedule must be the frozen evaluation schedule")
    started = time.monotonic()
    games: list[PilotEvaluationGame] = []
    for item in schedule:
        if controller == "random":
            p0 = p1 = "random"
        else:
            p0, p1 = (
                ("external", "random")
                if item.learner_seat == 0
                else ("random", "external")
            )
        view = client.duel_reset(
            item.episode_seed,
            p0,
            p1,
            item.learner_seat,
            item.profile_id,
            item.reference_seat,
        )
        candidate_errors = 0
        while not view.terminated and not view.truncated:
            if controller != "model":
                raise ValueError("random-vs-random evaluation must complete internally")
            if (view.start_profile != item.profile_id or
                    view.seat != item.learner_seat or
                    view.reference_seat != item.reference_seat or
                    not view.decision.candidates):
                candidate_errors += 1
                raise ValueError("pilot evaluation view drifted or has no candidates")
            batch = collate_decisions(
                (view.decision,),
                loaded.model.config.horizon_turns,
            )
            selected = loaded.model.select(batch)[0]
            matches = tuple(candidate for candidate in view.decision.candidates
                            if candidate.candidate_id == selected.candidate_id)
            if (selected.decision_id != view.decision.decision_id or len(matches) != 1):
                candidate_errors += 1
                raise ValueError("pilot model selected a stale or illegal candidate")
            view = client.duel_step(CandidateSelection(
                selected.decision_id,
                selected.candidate_id,
            ))
        fallback = client.duel_status()
        if view.winner == item.learner_seat:
            outcome: Literal["win", "loss", "draw"] = "win"
        elif view.winner in {0, 1}:
            outcome = "loss"
        else:
            outcome = "draw"
        games.append(PilotEvaluationGame(
            controller,
            item,
            view.winner,
            outcome,
            view.terminated,
            view.truncated,
            view.decision.decision_id,
            candidate_errors,
            fallback,
        ))
        elapsed = time.monotonic() - started
        if elapsed > _EVALUATION_DEADLINE_SECONDS:
            raise TimeoutError(
                f"pilot {controller} evaluation exceeded deadline after "
                f"{len(games)}/{len(schedule)} games and {elapsed:.1f} seconds"
            )
    frozen = tuple(games)
    aggregate = _evaluation_summary(frozen)
    profiles = tuple((
        profile,
        _evaluation_summary(tuple(
            game for game in frozen if game.schedule.profile_id == profile
        )),
    ) for profile in PILOT_PROFILES)
    return PilotEvaluation(controller, frozen, aggregate, profiles)


def _diagnostic_evaluation_wire(
    evaluation: PilotEvaluation,
) -> dict[str, object]:
    return {
        **_evaluation_wire(evaluation),
        "games": [asdict(game) for game in evaluation.games],
    }


def run_pilot_diagnostics(
    server_cmd: Sequence[str],
    output: Path,
    attempt_number: int,
    device: str,
    command: Sequence[str],
) -> Path:
    started = time.monotonic()
    output = Path(output)
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("pilot diagnostic attempt number must be a positive built-in int")
    for value, name in ((server_cmd, "server command"), (command, "command")):
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or not value
            or any(type(part) is not str or not part for part in value)
        ):
            raise ValueError(
                f"pilot diagnostic {name} must be a non-empty sequence of strings"
            )
    if not output.is_dir() or _is_reparse(output):
        raise ValueError("pilot diagnostic output must be the plain collection directory")
    attempt = output / f"retry-{attempt_number}"
    if not attempt.is_dir() or _is_reparse(attempt):
        raise ValueError("pilot diagnostic attempt must be a plain directory")
    diagnostics_path = attempt / "diagnostics.json"
    if diagnostics_path.exists() or diagnostics_path.is_symlink():
        raise FileExistsError("pilot diagnostics already exist")
    checkpoint = attempt / "checkpoints" / "best.pt"
    if not checkpoint.is_file() or _is_reparse(checkpoint):
        raise ValueError("pilot diagnostic checkpoint must be a plain file")

    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        identity = _validate_identity(client.identity)
    collection, evidence = load_collection_evidence(output, identity)
    loaded = load_structured_checkpoint(
        checkpoint, identity.encoding_hash, identity.capacity_hash,
    )
    if loaded.metadata.identity != collection.identity:
        raise ValueError("pilot diagnostic checkpoint identity does not match collection")

    metric_started = time.monotonic()
    loaded.model.to(device)
    try:
        validation = validation_metric_breakdown(
            loaded.model, collection.validation,
        )
    finally:
        loaded.model.cpu()
    metric_duration = time.monotonic() - metric_started

    schedule = diagnostic_evaluation_schedule()
    model_started = time.monotonic()
    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        model = evaluate_pilot(client, loaded, "model", schedule)
    model_duration = time.monotonic() - model_started
    random_started = time.monotonic()
    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        random_baseline = evaluate_pilot(client, loaded, "random", schedule)
    random_duration = time.monotonic() - random_started

    diagnostics: dict[str, object] = {
        "schema_version": 1,
        "execution": {
            "command": list(command),
            "device": device,
            "duration_seconds": time.monotonic() - started,
            "validation_metric_duration_seconds": metric_duration,
            "model_evaluation_duration_seconds": model_duration,
            "random_evaluation_duration_seconds": random_duration,
        },
        "identity": {
            "scenario_id": identity.scenario_id,
            "contract_hash": identity.contract_hash,
            "encoding_hash": identity.encoding_hash,
            "capacity_hash": identity.capacity_hash,
            "environment_kind": identity.environment_kind,
        },
        "checkpoint": {
            "path": str(checkpoint.relative_to(attempt)).replace("\\", "/"),
            "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        },
        "collection_sha256": evidence.collection_sha256,
        "validation": {
            "examples": len(collection.validation),
            "profiles": validation,
        },
        "schedule": [asdict(item) for item in schedule],
        "model": _diagnostic_evaluation_wire(model),
        "random": _diagnostic_evaluation_wire(random_baseline),
        "win_rate_margin": (
            model.aggregate.win_rate - random_baseline.aggregate.win_rate
        ),
        "claim_limit": (
            "diagnostic evidence from one frozen checkpoint and a fixed fresh-seed "
            "schedule; not a retraining result or statistical significance claim"
        ),
    }
    with diagnostics_path.open("xb") as handle:
        handle.write(_canonical_bytes(diagnostics))
        handle.flush()
        os.fsync(handle.fileno())
    return diagnostics_path


def pilot_decision(
    training: PilotTrainingArtifacts,
    model: PilotEvaluation,
    random: PilotEvaluation,
) -> tuple[bool, tuple[str, ...]]:
    reasons: list[str] = []
    if training.restored_validation.policy_nll >= training.initial_validation.policy_nll:
        reasons.append("validation policy NLL did not improve")
    if training.restored_validation.policy_accuracy <= training.initial_validation.policy_accuracy:
        reasons.append("validation policy accuracy did not improve")
    if (model.aggregate.candidate_errors or random.aggregate.candidate_errors or
            model.aggregate.internal_fallback_count or
            random.aggregate.internal_fallback_count):
        reasons.append("evaluation contained candidate errors or internal fallbacks")
    if model.aggregate.win_rate < random.aggregate.win_rate + 0.10:
        reasons.append("model win rate did not beat Random by at least 0.10")
    return not reasons, tuple(reasons)


_CLAIM_LIMIT = (
    "directional fixed-schedule evidence only; not statistical significance, "
    "generalization, production quality, or player-facing strength"
)


def _evaluation_wire(evaluation: PilotEvaluation) -> dict[str, object]:
    return {
        "aggregate": asdict(evaluation.aggregate),
        "profiles": {
            profile: asdict(summary)
            for profile, summary in evaluation.profiles
        },
    }


def render_pilot_report(evaluation: Mapping[str, object]) -> str:
    if not isinstance(evaluation, Mapping):
        raise TypeError("evaluation report input must be a mapping")
    execution = evaluation["execution"]
    identity = evaluation["identity"]
    collection = evaluation["collection"]
    initial = evaluation["initial"]
    restored = evaluation["restored_best"]
    model = evaluation["model"]
    random = evaluation["random"]
    decision = evaluation["decision"]
    if not all(isinstance(value, Mapping) for value in (
        execution, identity, collection, initial, restored, model, random, decision,
    )):
        raise TypeError("evaluation report sections must be mappings")
    model_aggregate = model["aggregate"]
    random_aggregate = random["aggregate"]
    if not isinstance(model_aggregate, Mapping) or not isinstance(random_aggregate, Mapping):
        raise TypeError("evaluation aggregate sections must be mappings")
    initial_validation = initial["validation"]
    restored_validation = restored["validation"]
    if not isinstance(initial_validation, Mapping) or not isinstance(
        restored_validation, Mapping
    ):
        raise TypeError("evaluation metric sections must be mappings")
    command = " ".join(str(value) for value in execution["command"])
    margin = float(model_aggregate["win_rate"]) - float(random_aggregate["win_rate"])
    nll_gate = float(restored_validation["policy_nll"]) < float(
        initial_validation["policy_nll"]
    )
    accuracy_gate = float(restored_validation["policy_accuracy"]) > float(
        initial_validation["policy_accuracy"]
    )
    error_gate = all(int(value) == 0 for value in (
        model_aggregate["candidate_errors"],
        model_aggregate["internal_fallback_count"],
        random_aggregate["candidate_errors"],
        random_aggregate["internal_fallback_count"],
    ))
    margin_gate = margin >= 0.10
    result = "PROMISING" if decision["promising"] else "NOT PROMISING"
    reason_lines = decision["reasons"] or ["All gates passed."]
    lines = [
        "# Tactical-v3 training-first pilot",
        "",
        f"**Decision: {result}**",
        "",
        f"- Command: `{command}`",
        f"- Device: `{execution['device']}`",
        f"- Total duration: {float(execution['duration_seconds']):.1f}s",
        f"- Scenario: `{identity['scenario_id']}`",
        f"- Contract: `{identity['contract_hash']}`",
        f"- Encoding: `{identity['encoding_hash']}`",
        f"- Capacity: `{identity['capacity_hash']}`",
        f"- Collection: {collection['train_games']} train games / "
        f"{collection['validation_games']} validation games; "
        f"{collection['train_examples']} / {collection['validation_examples']} examples",
        "",
        "## Validation metrics",
        "",
        "| Model | Policy NLL | Exact target accuracy | Valid logits |",
        "|---|---:|---:|---:|",
        f"| Initial | {float(initial_validation['policy_nll']):.6f} | "
        f"{float(initial_validation['policy_accuracy']):.3f} | "
        f"{initial_validation['valid_logit_count']} |",
        f"| Restored best | {float(restored_validation['policy_nll']):.6f} | "
        f"{float(restored_validation['policy_accuracy']):.3f} | "
        f"{restored_validation['valid_logit_count']} |",
        "",
        "## Matched evaluation",
        "",
        "| Controller | Games | Wins | Losses | Draws | Truncations | "
        "Mean decisions | Win rate | Candidate errors | Fallbacks |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, aggregate in (("Model", model_aggregate), ("Random", random_aggregate)):
        lines.append(
            f"| {label} | {aggregate['games']} | {aggregate['wins']} | "
            f"{aggregate['losses']} | {aggregate['draws']} | "
            f"{aggregate['truncations']} | {float(aggregate['mean_decisions']):.2f} | "
            f"{float(aggregate['win_rate']):.3f} | {aggregate['candidate_errors']} | "
            f"{aggregate['internal_fallback_count']} |"
        )
    lines.extend([
        "",
        "### Per-profile results",
        "",
        "| Profile | Controller | Games | Wins | Losses | Draws | Win rate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for profile in PILOT_PROFILES:
        for label, section in (("Model", model), ("Random", random)):
            profiles = section["profiles"]
            if not isinstance(profiles, Mapping) or profile not in profiles:
                continue
            summary = profiles[profile]
            if not isinstance(summary, Mapping):
                raise TypeError("evaluation profile summaries must be mappings")
            lines.append(
                f"| {profile} | {label} | {summary['games']} | {summary['wins']} | "
                f"{summary['losses']} | {summary['draws']} | "
                f"{float(summary['win_rate']):.3f} |"
            )
    lines.extend([
        "",
        f"Matched win-rate margin: **{margin:+.3f}**",
        "",
        "## Gates",
        "",
        f"- Validation NLL improved: `{str(nll_gate).lower()}`",
        f"- Validation accuracy improved: `{str(accuracy_gate).lower()}`",
        f"- Candidate/fallback errors are zero: `{str(error_gate).lower()}`",
        f"- Model margin is at least 0.10: `{str(margin_gate).lower()}`",
        "",
        "## Decision reasons",
        "",
        *(f"- {reason}" for reason in reason_lines),
        "",
        f"> Claim limit: {evaluation['claim_limit']}.",
        "",
    ])
    return "\n".join(lines)


def run_pilot(
    server_cmd: Sequence[str],
    output: Path,
    seed: int,
    device: str,
    command: Sequence[str],
) -> Path:
    started = time.monotonic()
    output = Path(output)
    if type(seed) is not int or seed != 227:
        raise ValueError("pilot seed must be exactly 227")
    _pilot_configs(seed, device)
    for value, name in ((server_cmd, "server command"), (command, "command")):
        if (isinstance(value, (str, bytes)) or not isinstance(value, Sequence) or
                not value or any(type(part) is not str or not part for part in value)):
            raise ValueError(f"pilot {name} must be a non-empty sequence of strings")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"pilot output already exists: {output}")
    if not output.parent.is_dir() or _is_reparse(output.parent):
        raise ValueError("pilot output parent must be a plain directory")

    phase = "collection"
    try:
        phase_started = time.monotonic()
        collection = collect_pilot(server_cmd)
        collection_duration = time.monotonic() - phase_started
        evidence = write_collection_evidence(output, collection)

        phase = "training"
        training = train_pilot(collection, evidence, output, seed, device)

        schedule = evaluation_schedule()
        phase = "model evaluation"
        phase_started = time.monotonic()
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            model = evaluate_pilot(client, training.loaded, "model", schedule)
        model_duration = time.monotonic() - phase_started

        phase = "random evaluation"
        phase_started = time.monotonic()
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            random_baseline = evaluate_pilot(
                client, training.loaded, "random", schedule,
            )
        random_duration = time.monotonic() - phase_started
        promising, reasons = pilot_decision(training, model, random_baseline)
        evaluation: dict[str, object] = {
            "schema_version": 1,
            "execution": {
                "command": list(command),
                "device": device,
                "duration_seconds": time.monotonic() - started,
                "collection_duration_seconds": collection_duration,
                "training_duration_seconds": training.duration_seconds,
                "model_evaluation_duration_seconds": model_duration,
                "random_evaluation_duration_seconds": random_duration,
            },
            "identity": {
                "scenario_id": collection.identity.scenario_id,
                "contract_hash": collection.identity.contract_hash,
                "encoding_hash": collection.identity.encoding_hash,
                "capacity_hash": collection.identity.capacity_hash,
                "environment_kind": collection.identity.environment_kind,
            },
            "collection": {
                "train_games": 28,
                "validation_games": 14,
                "train_examples": len(collection.train),
                "validation_examples": len(collection.validation),
            },
            "schedule": [asdict(item) for item in schedule],
            "initial": {
                "train": asdict(training.initial_train),
                "validation": asdict(training.initial_validation),
            },
            "restored_best": {
                "train": asdict(training.restored_train),
                "validation": asdict(training.restored_validation),
            },
            "model": _evaluation_wire(model),
            "random": _evaluation_wire(random_baseline),
            "decision": {"promising": promising, "reasons": list(reasons)},
            "claim_limit": _CLAIM_LIMIT,
        }
        with (output / "evaluation.json").open("xb") as handle:
            handle.write(_canonical_bytes(evaluation))
            handle.flush()
            os.fsync(handle.fileno())
        report = render_pilot_report(evaluation).encode("utf-8")
        with (output / "report.md").open("xb") as handle:
            handle.write(report)
            handle.flush()
            os.fsync(handle.fileno())
        return output
    except Exception as error:
        if not output.exists():
            output.mkdir()
        report_path = output / "report.md"
        if not report_path.exists():
            failure = (
                "# Tactical-v3 training-first pilot\n\n"
                f"Failed during **{phase}** after {time.monotonic() - started:.1f}s.\n\n"
                f"`{type(error).__name__}: {error}`\n"
            ).encode("utf-8")
            with report_path.open("xb") as handle:
                handle.write(failure)
                handle.flush()
                os.fsync(handle.fileno())
        raise


def run_pilot_retry(
    server_cmd: Sequence[str],
    output: Path,
    seed: int,
    device: str,
    command: Sequence[str],
    *,
    attempt_number: int = 1,
) -> Path:
    started = time.monotonic()
    output = Path(output)
    if type(seed) is not int or seed != 227:
        raise ValueError("pilot retry seed must be exactly 227")
    if type(attempt_number) is not int or attempt_number < 1:
        raise ValueError("pilot retry attempt number must be a positive built-in int")
    _pilot_configs(seed, device)
    for value, name in ((server_cmd, "server command"), (command, "command")):
        if (
            isinstance(value, (str, bytes))
            or not isinstance(value, Sequence)
            or not value
            or any(type(part) is not str or not part for part in value)
        ):
            raise ValueError(f"pilot retry {name} must be a non-empty sequence of strings")
    if not output.is_dir() or _is_reparse(output):
        raise ValueError("pilot retry output must be the plain collection directory")
    attempt = output / f"retry-{attempt_number}"
    if attempt.exists() or attempt.is_symlink():
        raise FileExistsError(f"pilot retry-{attempt_number} artifacts already exist")

    phase = "collection authentication"
    try:
        print("pilot telemetry phase=load_collection", flush=True)
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            identity = client.identity
        collection, evidence = load_collection_evidence(output, identity)
        print(
            "pilot telemetry collection_loaded "
            f"train_examples={len(collection.train)} "
            f"validation_examples={len(collection.validation)}",
            flush=True,
        )

        phase = "training"
        training = train_pilot(
            collection,
            evidence,
            output,
            seed,
            device,
            artifacts_output=attempt,
        )
        schedule = evaluation_schedule()

        phase = "model evaluation"
        phase_started = time.monotonic()
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            model = evaluate_pilot(client, training.loaded, "model", schedule)
        model_duration = time.monotonic() - phase_started

        phase = "random evaluation"
        phase_started = time.monotonic()
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            random_baseline = evaluate_pilot(
                client, training.loaded, "random", schedule,
            )
        random_duration = time.monotonic() - phase_started
        promising, reasons = pilot_decision(training, model, random_baseline)
        evaluation: dict[str, object] = {
            "schema_version": 1,
            "execution": {
                "command": list(command),
                "device": device,
                "duration_seconds": time.monotonic() - started,
                "collection_duration_seconds": 0.0,
                "training_duration_seconds": training.duration_seconds,
                "model_evaluation_duration_seconds": model_duration,
                "random_evaluation_duration_seconds": random_duration,
            },
            "identity": {
                "scenario_id": collection.identity.scenario_id,
                "contract_hash": collection.identity.contract_hash,
                "encoding_hash": collection.identity.encoding_hash,
                "capacity_hash": collection.identity.capacity_hash,
                "environment_kind": collection.identity.environment_kind,
            },
            "collection": {
                "train_games": 28,
                "validation_games": 14,
                "train_examples": len(collection.train),
                "validation_examples": len(collection.validation),
            },
            "schedule": [asdict(item) for item in schedule],
            "initial": {
                "train": asdict(training.initial_train),
                "validation": asdict(training.initial_validation),
            },
            "restored_best": {
                "train": asdict(training.restored_train),
                "validation": asdict(training.restored_validation),
            },
            "model": _evaluation_wire(model),
            "random": _evaluation_wire(random_baseline),
            "decision": {"promising": promising, "reasons": list(reasons)},
            "claim_limit": _CLAIM_LIMIT,
        }
        with (attempt / "evaluation.json").open("xb") as handle:
            handle.write(_canonical_bytes(evaluation))
            handle.flush()
            os.fsync(handle.fileno())
        with (attempt / "report.md").open("xb") as handle:
            handle.write(render_pilot_report(evaluation).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        return attempt
    except Exception as error:
        if attempt.is_dir() and not _is_reparse(attempt):
            report_path = attempt / "report.md"
            if not report_path.exists():
                failure = (
                    "# Tactical-v3 training-first pilot retry\n\n"
                    f"Failed during **{phase}** after "
                    f"{time.monotonic() - started:.1f}s.\n\n"
                    f"`{type(error).__name__}: {error}`\n"
                ).encode("utf-8")
                with report_path.open("xb") as handle:
                    handle.write(failure)
                    handle.flush()
                    os.fsync(handle.fileno())
        raise
