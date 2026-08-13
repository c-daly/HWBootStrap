"""Frozen schedules and complete-game evidence for the tactical-v3 training pilot."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time
from typing import Literal

import torch
import torch.nn.functional as F

from .tactical_v3_batching import collate_decisions, collate_examples
from .tactical_v3_checkpoint import (
    LoadedStructuredPolicy,
    StructuredCheckpointMetadata,
    load_structured_checkpoint,
    save_structured_checkpoint,
    structured_model_state_sha256,
)
from .tactical_v3_client import CandidateSelection, TacticalV3GymClient, TeacherSelection
from .tactical_v3_corpus import StructuredExample, StructuredTarget, TeacherEvidence
from .tactical_v3_layers import TacticalV3ModelConfig
from .tactical_v3_model import TacticalV3Policy
from .tactical_v3_objectives import ObjectiveConfig
from .tactical_v3_schema import TacticalV3Decision, TacticalV3SemanticIdentity, TacticalV3View
from .tactical_v3_training import EpochMetrics, TrainerConfig, train_offline


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
        max_epochs=100,
        patience_epochs=12,
        gradient_clip_norm=1.0,
        device=device,
    )
    return model, objective, trainer


def _policy_target_metrics(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
    *,
    batch_size: int = 32,
) -> PolicyTargetMetrics:
    if type(model) is not TacticalV3Policy:
        raise TypeError("policy metric model must be TacticalV3Policy")
    if not examples:
        raise ValueError("policy metric examples must not be empty")
    if next(model.parameters()).device.type != "cpu":
        raise ValueError("policy metric model must be on CPU")
    model.eval()
    total_nll = 0.0
    correct = 0
    finite = 0
    valid = 0
    with torch.inference_mode():
        for offset in range(0, len(examples), batch_size):
            rows = examples[offset:offset + batch_size]
            batch = collate_examples(rows, model.config.horizon_turns)
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
    count = len(examples)
    return PolicyTargetMetrics(total_nll / count, correct / count, finite, valid)


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


def train_pilot(
    collection: PilotCollection,
    evidence: PilotCollectionEvidence,
    output: Path,
    seed: int,
    device: str,
) -> PilotTrainingArtifacts:
    started = time.monotonic()
    _require_collection(collection)
    if type(seed) is not int or seed != 227:
        raise ValueError("pilot training seed must be exactly 227")
    if type(evidence) is not PilotCollectionEvidence:
        raise TypeError("evidence must be PilotCollectionEvidence")
    output = Path(output)
    if not output.is_dir() or _is_reparse(output):
        raise ValueError("pilot training output must be the plain collection directory")
    if hashlib.sha256((output / "collection.json").read_bytes()).hexdigest() != (
        evidence.collection_sha256
    ):
        raise ValueError("pilot collection evidence hash does not match output")

    model_config, objective_config, trainer_config = _pilot_configs(seed, device)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed % (2**63 - 1))
        initial_model = TacticalV3Policy(model_config).cpu().eval()
    initial_train = _policy_target_metrics(initial_model, collection.train)
    initial_validation = _policy_target_metrics(initial_model, collection.validation)

    result = train_offline(
        collection.train,
        collection.validation,
        model_config,
        objective_config,
        trainer_config,
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
        identity=collection.identity,
        model_config=result.model_config,
        objective_config=result.objective_config,
        trainer_config=result.trainer_config,
        corpus_sha256=evidence.collection_sha256,
        model_state_sha256=structured_model_state_sha256(result.model),
        best_epoch=result.best_epoch,
        best_validation_policy_nll=result.best_validation_policy_nll,
        published_device="cpu",
    )
    save_structured_checkpoint(
        checkpoint,
        result.model,
        metadata,
        collection.validation[:2],
    )
    with metrics_path.open("xb") as handle:
        handle.write(_history_bytes(result.history))
        handle.flush()
        os.fsync(handle.fileno())
    loaded = load_structured_checkpoint(
        checkpoint,
        collection.identity.encoding_hash,
        collection.identity.capacity_hash,
    )
    restored_train = _policy_target_metrics(loaded.model, collection.train)
    restored_validation = _policy_target_metrics(loaded.model, collection.validation)
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
    if tuple(schedule) != evaluation_schedule():
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
