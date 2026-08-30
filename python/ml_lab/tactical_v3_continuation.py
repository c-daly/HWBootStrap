"""ML Lab lifecycle for a weight-initialized tactical-v3 DAgger continuation.

The mutable run owns collection evidence, local status, logs, and TensorBoard data.
When training completes, a separate strict schema-2 policy package is published next
to it.  This keeps operational files out of the immutable seven-file policy format.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from collections import Counter
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import torch
from torch.utils.tensorboard import SummaryWriter

from .contracts import (
    MONITOR_HEADER,
    PROGRESS_HEADER,
    request_stop,
    update_run_state,
    utc_now,
    validate_run_name,
    validate_tracker_specs,
)
from .controllers import (
    ControllerBinding,
    ControllerResolver,
    ResolvedController,
    normalize_controller_spec,
)
from .io import atomic_write_json, read_json
from .tactical_v3_checkpoint import (
    LoadedStructuredPolicy,
    adopt_structured_run,
    semantic_identity_wire,
    validate_structured_run,
)
from .tactical_v3_controller import StructuredController
from .tactical_v3_pilot import (
    PilotDaggerEpisode,
    PilotScheduleItem,
    PilotTrainingStopRequested,
    _canonical_bytes,
    _train_pilot_dataset,
    collect_dagger_game,
    write_dagger_episode,
)
from .tactical_v3_schema import TacticalV3SemanticIdentity
from .tactical_v3_training import TrainerConfig
from .tactical_v3_client import TacticalV3GymClient


OpponentKind = Literal["random", "greedy", "fixed_run", "live_run"]


@dataclass(frozen=True, slots=True)
class StructuredContinuationConfig:
    run_name: str
    source_run: Path
    scenario_file: Path
    opponent: str
    train_label_target: int
    validation_label_target: int
    seed: int
    device: str
    learner_seat: Literal["alternating", "0", "1"]
    trackers: tuple[Mapping[str, Any], ...]
    oracle_expansion_budget: Literal[512, 2048] = 512

    def validate(self) -> None:
        validate_run_name(self.run_name)
        if not Path(self.source_run).is_dir():
            raise FileNotFoundError(self.source_run)
        if not Path(self.scenario_file).is_file():
            raise FileNotFoundError(self.scenario_file)
        expected_scenario = Path(self.source_run).resolve() / "scenario.json"
        if Path(self.scenario_file).resolve() != expected_scenario:
            raise ValueError(
                "tactical-v3 continuation must use source_run/scenario.json exactly"
            )
        if type(self.train_label_target) is not int or self.train_label_target < 2:
            raise ValueError("tactical-v3 train label target must be at least 2")
        if (
            type(self.validation_label_target) is not int
            or self.validation_label_target < 2
        ):
            raise ValueError("tactical-v3 validation label target must be at least 2")
        if type(self.seed) is not int or not 0 <= self.seed <= 20_000:
            raise ValueError("tactical-v3 seed must be an integer from 0 through 20000")
        if self.learner_seat not in {"alternating", "0", "1"}:
            raise ValueError("tactical-v3 learner seat is invalid")
        if not self.device.strip():
            raise ValueError("tactical-v3 device is required")
        if self.oracle_expansion_budget not in {512, 2048}:
            raise ValueError("tactical-v3 oracle expansion budget is unsupported")
        normalize_controller_spec(self.opponent)
        validate_tracker_specs(list(self.trackers))
        unsupported_trackers = sorted({
            str(tracker.get("kind", ""))
            for tracker in self.trackers
            if tracker.get("kind") not in {"local", "tensorboard"}
        })
        if unsupported_trackers:
            raise ValueError(
                "tactical-v3 continuation supports only local and TensorBoard "
                f"trackers, not {', '.join(unsupported_trackers)}"
            )


@dataclass(frozen=True, slots=True)
class _Opponent:
    kind: OpponentKind
    binding: ControllerBinding | None
    metadata: Mapping[str, Any]

    def controller_for_game(
        self,
        identity: TacticalV3SemanticIdentity,
    ) -> str | StructuredController:
        if self.binding is None:
            return self.kind
        if self.kind == "live_run":
            self.binding.reload(lambda value: _validate_model_opponent(value, identity))
        resolved = self.binding.resolved
        _validate_model_opponent(resolved, identity)
        assert type(resolved.model) is StructuredController
        return resolved.model

    def game_metadata(self) -> Mapping[str, Any]:
        if self.binding is None:
            return self.metadata
        return _resolved_opponent_metadata(self.binding.resolved, self.kind)


class _LiveTelemetry:
    def __init__(
        self,
        run_dir: Path,
        *,
        enabled: bool,
        train_target: int,
        validation_target: int,
    ) -> None:
        self.run_dir = Path(run_dir)
        self.train_target = train_target
        self.validation_target = validation_target
        self.writer = (
            SummaryWriter(log_dir=str(self.run_dir / "tensorboard"), flush_secs=1)
            if enabled
            else None
        )
        if self.writer is not None:
            self._record({
                "lifecycle/phase": 0.0,
                "collection/train_label_target": train_target,
                "collection/validation_label_target": validation_target,
            }, 0)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def _record(self, metrics: Mapping[str, float | int], step: int) -> None:
        writer = self.writer
        if writer is None:
            return
        try:
            for name, value in metrics.items():
                writer.add_scalar(name, value, step)
            writer.flush()
        except Exception as error:
            self.writer = None
            try:
                writer.close()
            except Exception:
                pass
            _mark_tracker_degraded(self.run_dir, "tensorboard", error)

    def collection(
        self,
        *,
        partition: Literal["train", "validation"],
        global_game: int,
        partition_games: int,
        partition_labels: int,
        total_labels: int,
        started: float,
        outcomes: Counter[str],
        seat_outcomes: Mapping[int, Counter[str]],
        disagreements: int,
        fallbacks: int,
        reasons: Counter[str],
    ) -> None:
        elapsed = max(time.monotonic() - started, 1e-9)
        target = (
            self.train_target if partition == "train" else self.validation_target
        )
        games = max(sum(outcomes.values()), 1)
        metrics = {
            f"collection/{partition}_games": partition_games,
            f"collection/{partition}_labels": partition_labels,
            f"collection/{partition}_target": target,
            "collection/total_labels": total_labels,
            "collection/labels_per_second": total_labels / elapsed,
            "collection/win_rate": outcomes["win"] / games,
            "collection/loss_rate": outcomes["loss"] / games,
            "collection/draw_rate": outcomes["draw"] / games,
            "collection/disagreement_rate": disagreements / max(total_labels, 1),
            "collection/internal_fallback_count": fallbacks,
            "collection/reason_conversion": reasons["conversion"],
            "collection/reason_favorable": reasons["favorable"],
            "collection/reason_cycle_warning": reasons["cycle_warning"],
            "collection/reason_wasted_end_turn": reasons["wasted_end_turn"],
        }
        for seat in (0, 1):
            seat_games = max(sum(seat_outcomes[seat].values()), 1)
            metrics[f"collection/seat_{seat}_win_rate"] = (
                seat_outcomes[seat]["win"] / seat_games
            )
            metrics[f"collection/seat_{seat}_loss_rate"] = (
                seat_outcomes[seat]["loss"] / seat_games
            )
            metrics[f"collection/seat_{seat}_draw_rate"] = (
                seat_outcomes[seat]["draw"] / seat_games
            )
        self._record(metrics, global_game)

    def phase(self, value: float, step: int) -> None:
        self._record({"lifecycle/phase": value}, step)

    def final(self, artifacts: Any, step: int) -> None:
        metrics: dict[str, float] = {}
        for split in ("train", "validation"):
            baseline = getattr(artifacts, f"initial_{split}")
            restored = getattr(artifacts, f"restored_{split}")
            metrics[f"final/{split}_policy_nll"] = restored.policy_nll
            metrics[f"final/{split}_policy_accuracy"] = restored.policy_accuracy
            metrics[f"final/{split}_nll_change"] = (
                restored.policy_nll - baseline.policy_nll
            )
        improved = (
            artifacts.restored_validation.policy_nll
            < artifacts.initial_validation.policy_nll - 1e-12
        )
        metrics["final/candidate_improved_over_source"] = float(improved)
        metrics["final/source_retained"] = float(not improved)
        self._record(metrics, step)

    def close(self) -> None:
        writer = self.writer
        self.writer = None
        if writer is None:
            return
        try:
            writer.flush()
            writer.close()
        except Exception as error:
            _mark_tracker_degraded(self.run_dir, "tensorboard", error)

    def __enter__(self) -> "_LiveTelemetry":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv_header(path: Path, fields: Sequence[str]) -> None:
    with Path(path).open("x", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerow(fields)


def _append_csv(path: Path, fields: Sequence[str], row: Mapping[str, Any]) -> None:
    with Path(path).open("a", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writerow(row)


def _log(run_dir: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    with (Path(run_dir) / "train.log").open(
        "a", encoding="utf-8", buffering=1,
    ) as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _contract(identity: TacticalV3SemanticIdentity) -> dict[str, str]:
    return {
        "environment": "tactical-v3",
        "version": identity.contract_version,
        "environment_kind": identity.environment_kind,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }


def _resolved_opponent_metadata(
    resolved: ResolvedController,
    kind: OpponentKind,
) -> dict[str, Any]:
    path = resolved.path
    return {
        "kind": kind,
        "mode": resolved.spec.mode,
        "source_run": (
            str(resolved.spec.path.resolve())
            if resolved.spec.path is not None else None
        ),
        "checkpoint": str(path) if path is not None else None,
        "checkpoint_sha256": (
            _sha256_file(path) if path is not None and path.is_file() else None
        ),
        "step": resolved.step,
        "algorithm": resolved.algorithm,
    }


def _validate_model_opponent(
    resolved: ResolvedController,
    identity: TacticalV3SemanticIdentity,
) -> None:
    if (
        resolved.algorithm != "structured_imitation"
        or type(resolved.model) is not StructuredController
        or resolved.contract != identity
    ):
        raise ValueError(
            "tactical-v3 model opponent must match the source scenario and policy identity"
        )


def _resolve_opponent(raw: str) -> _Opponent:
    spec = normalize_controller_spec(raw)
    if spec.kind == "scripted":
        if spec.name not in {"random", "greedy"}:
            raise ValueError("tactical-v3 continuation opponent must be random or greedy")
        return _Opponent(spec.name, None, {"kind": "scripted", "name": spec.name})
    if spec.kind != "run":
        raise ValueError("tactical-v3 model opponent must be a metadata-backed run")
    binding = ControllerResolver().bind(spec)
    kind: OpponentKind = "live_run" if spec.mode == "live" else "fixed_run"
    return _Opponent(kind, binding, _resolved_opponent_metadata(binding.resolved, kind))


def _create_live_run(
    runs_root: Path,
    config: StructuredContinuationConfig,
    identity: TacticalV3SemanticIdentity,
    source: LoadedStructuredPolicy,
    opponent: _Opponent,
) -> Path:
    runs_root = Path(runs_root)
    runs_root.mkdir(parents=True, exist_ok=True)
    run_dir = runs_root / config.run_name
    run_dir.mkdir(exist_ok=True)
    if (run_dir / "run.json").exists():
        raise FileExistsError(run_dir / "run.json")
    unexpected = {
        path.name for path in run_dir.iterdir() if path.name != "train-err.log"
    }
    if unexpected:
        raise FileExistsError(f"run directory is not empty: {run_dir}")
    (run_dir / "collection" / "train").mkdir(parents=True)
    (run_dir / "collection" / "validation").mkdir()
    (run_dir / "training").mkdir()
    scenario_value = json.loads(Path(config.scenario_file).read_text(encoding="utf-8"))
    atomic_write_json(run_dir / "scenario.json", scenario_value)
    atomic_write_json(run_dir / "control.json", {"request": None})
    atomic_write_json(run_dir / "evaluation.json", {})
    _write_csv_header(run_dir / "progress.csv", PROGRESS_HEADER)
    _write_csv_header(run_dir / "monitor.csv", MONITOR_HEADER)
    (run_dir / "train.log").touch(exist_ok=False)
    created = utc_now()
    manifest = {
        "schema_version": 1,
        "created_at": created,
        "updated_at": created,
        "state": "created",
        "pid": os.getpid(),
        "timesteps": 0,
        "target_step": config.train_label_target + config.validation_label_target,
        "latest_message": "validating tactical-v3 continuation",
        "latest_checkpoint": None,
        "latest_checkpoint_step": None,
        "monitor_files": ["monitor.csv"],
        "config": {
            "backend": "structured_dagger",
            "algorithm": "structured_dagger",
            "policy": "TacticalV3Policy",
            "run_name": config.run_name,
            "environment": "tactical-v3",
            "seed": config.seed,
            "total_timesteps": config.train_label_target,
            "validation_label_target": config.validation_label_target,
            "workers": 1,
            "device": config.device,
            "collection_device": "cpu",
            "training_device": config.device,
            "learner_seat": config.learner_seat,
            "opponent": dict(opponent.metadata),
            "trackers": [dict(value) for value in config.trackers],
            "resume_source": None,
            "initialization_source": str(Path(config.source_run).resolve()),
            "continuation_semantics": "weights-only-initialization",
        },
        "contract": _contract(identity),
        "scenario": {"path": "scenario.json", "schema_version": identity.scenario_schema_version},
        "opponent_snapshot": dict(opponent.metadata),
        "source_policy": {
            "run": str(Path(config.source_run).resolve()),
            "model_state_sha256": source.metadata.model_state_sha256,
            "corpus_sha256": source.metadata.corpus_sha256,
            "best_epoch": source.metadata.best_epoch,
            "best_validation_policy_nll": source.metadata.best_validation_policy_nll,
        },
        "evidence_status": "unsealed-experimental",
        "tracker_status": [],
    }
    atomic_write_json(run_dir / "run.json", manifest)
    atomic_write_json(
        run_dir / "params.json",
        {"config": manifest["config"], "contract": manifest["contract"]},
    )
    return run_dir


def _stop_requested(run_dir: Path) -> bool:
    try:
        return read_json(run_dir / "control.json").get("request") in {
            "stop_now", "stop_after_checkpoint",
        }
    except (OSError, json.JSONDecodeError, AttributeError):
        return False


def _seat_sequence(value: str) -> tuple[int, ...]:
    if value == "0":
        return (0,)
    if value == "1":
        return (1,)
    return (0, 1)


def _partition_target_complete(
    labels: int,
    target: int,
    seat: int,
    seats: tuple[int, ...],
) -> bool:
    """Reach a label target only after the current reciprocal seat cycle ends."""

    return labels >= target and seat == seats[-1]


def _outcome(episode: PilotDaggerEpisode) -> Literal["win", "loss", "draw"]:
    winner = episode.summary.winner
    learner = episode.summary.schedule.learner_seat
    if winner == learner:
        return "win"
    if winner in {0, 1}:
        return "loss"
    return "draw"


def _collect_partition(
    client: TacticalV3GymClient,
    source: LoadedStructuredPolicy,
    opponent: _Opponent,
    config: StructuredContinuationConfig,
    run_dir: Path,
    telemetry: _LiveTelemetry,
    *,
    partition: Literal["train", "validation"],
    target: int,
    seed_start: int,
    global_game_start: int,
    global_label_start: int,
    started: float,
    outcomes: Counter[str],
    seat_outcomes: Mapping[int, Counter[str]],
    reasons: Counter[str],
    global_disagreements: list[int],
    global_fallbacks: list[int],
) -> tuple[tuple[PilotDaggerEpisode, ...], list[dict[str, Any]], int, int]:
    episodes: list[PilotDaggerEpisode] = []
    evidence: list[dict[str, Any]] = []
    labels = 0
    games = 0
    max_games = max(100, target * 2)
    seats = _seat_sequence(config.learner_seat)
    while labels < target:
        if games >= max_games:
            raise RuntimeError(
                f"{partition} collection did not reach {target} labels in {max_games} games"
            )
        for seat in seats:
            if _stop_requested(run_dir):
                return tuple(episodes), evidence, labels, games
            seed = seed_start + games // len(seats)
            item = PilotScheduleItem(
                partition, "standard-3v3", seed, seat, seat,
            )
            game_opponent = opponent.controller_for_game(client.identity)
            game_metadata = opponent.game_metadata()
            game_started = time.monotonic()
            episode = collect_dagger_game(
                client,
                source,
                item,
                oracle_expansion_budget=config.oracle_expansion_budget,
                opponent=game_opponent,
            )
            duration = time.monotonic() - game_started
            outcome = _outcome(episode)
            outcomes[outcome] += 1
            seat_outcomes[seat][outcome] += 1
            games += 1
            global_game = global_game_start + games
            labels += len(episode.records)
            global_labels = global_label_start + labels
            global_disagreements[0] += episode.summary.disagreements
            global_fallbacks[0] += episode.summary.internal_fallback_count
            for record in episode.records:
                reasons.update(record.eligibility_reasons)
            episode_entry: dict[str, Any] = {
                "index": games - 1,
                "schedule": asdict(item),
                "outcome": outcome,
                "winner": episode.summary.winner,
                "labels": len(episode.records),
                "disagreements": episode.summary.disagreements,
                "duration_seconds": duration,
                "opponent": dict(game_metadata),
            }
            if episode.records:
                episode_dir = (
                    run_dir / "collection" / partition / f"game-{games - 1:04d}"
                )
                write_dagger_episode(episode_dir, episode)
                episode_entry.update({
                    "episode_sha256": _sha256_file(episode_dir / "episode.json"),
                    "records_sha256": _sha256_file(episode_dir / "decisions.jsonl"),
                })
                episodes.append(episode)
            else:
                episode_entry.update({
                    "episode_sha256": None,
                    "records_sha256": None,
                })
            evidence.append(episode_entry)
            reward = 1.0 if outcome == "win" else -1.0 if outcome == "loss" else 0.0
            _append_csv(run_dir / "monitor.csv", MONITOR_HEADER, {
                "worker_id": 0,
                "episode_index": global_game - 1,
                "episode_seed": seed,
                "learner_seat": seat,
                "episode_reward": reward,
                "episode_length": episode.summary.decisions,
                "elapsed_seconds": duration,
            })
            elapsed = max(time.monotonic() - started, 1e-9)
            _append_csv(run_dir / "progress.csv", PROGRESS_HEADER, {
                "timestamp": utc_now(),
                "timesteps": global_labels,
                "episodes": global_game,
                "mean_reward": reward,
                "steps_per_second": global_labels / elapsed,
            })
            update_run_state(
                run_dir,
                "running",
                pid=os.getpid(),
                timesteps=global_labels,
                episodes=global_game,
                throughput=global_labels / elapsed,
                latest_message=(
                    f"collecting {partition}: {labels}/{target} labels "
                    f"after {games} games"
                ),
            )
            telemetry.collection(
                partition=partition,
                global_game=global_game,
                partition_games=games,
                partition_labels=labels,
                total_labels=global_labels,
                started=started,
                outcomes=outcomes,
                seat_outcomes=seat_outcomes,
                disagreements=global_disagreements[0],
                fallbacks=global_fallbacks[0],
                reasons=reasons,
            )
            _log(
                run_dir,
                f"collection partition={partition} game={games} seed={seed} "
                f"seat={seat} opponent={opponent.kind} outcome={outcome} "
                f"labels={len(episode.records)} cumulative={labels}/{target} "
                f"disagreements={episode.summary.disagreements} "
                f"elapsed_seconds={duration:.1f}",
            )
            if _partition_target_complete(labels, target, seat, seats):
                break
    return tuple(episodes), evidence, labels, games


def _records_bytes(episodes: Sequence[PilotDaggerEpisode]) -> bytes:
    return b"".join(
        _canonical_bytes(asdict(record))
        for episode in episodes
        for record in episode.records
    )


def _write_collection_manifest(
    run_dir: Path,
    config: StructuredContinuationConfig,
    identity: TacticalV3SemanticIdentity,
    source: LoadedStructuredPolicy,
    opponent: _Opponent,
    train_evidence: list[dict[str, Any]],
    validation_evidence: list[dict[str, Any]],
    train_records: bytes,
    validation_records: bytes,
) -> tuple[Path, str]:
    value = {
        "schema_version": 1,
        "kind": "tactical-v3-ml-lab-dagger-continuation-collection",
        "evidence_status": "unsealed-experimental",
        "identity": semantic_identity_wire(identity),
        "source": {
            "run": str(Path(config.source_run).resolve()),
            "model_state_sha256": source.metadata.model_state_sha256,
            "corpus_sha256": source.metadata.corpus_sha256,
            "best_epoch": source.metadata.best_epoch,
            "best_validation_policy_nll": source.metadata.best_validation_policy_nll,
        },
        "opponent": dict(opponent.metadata),
        "oracle": {
            "identity": "bounded-search-v1",
            "search_depth": 4,
            "expansion_budget": config.oracle_expansion_budget,
            "heuristic_identity": "material-plus-pursuit-v1",
        },
        "schedule": {
            "seed": config.seed,
            "learner_seat": config.learner_seat,
            "profile": "standard-3v3",
            "train_label_target": config.train_label_target,
            "validation_label_target": config.validation_label_target,
        },
        "train": {
            "games": train_evidence,
            "labels": sum(value["labels"] for value in train_evidence),
            "records_sha256": hashlib.sha256(train_records).hexdigest(),
        },
        "validation": {
            "games": validation_evidence,
            "labels": sum(value["labels"] for value in validation_evidence),
            "records_sha256": hashlib.sha256(validation_records).hexdigest(),
        },
    }
    path = run_dir / "collection.json"
    data = _canonical_bytes(value)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    return path, hashlib.sha256(data).hexdigest()


def _training_manifest(
    config: StructuredContinuationConfig,
    source: LoadedStructuredPolicy,
    collection_sha256: str,
    corpus_sha256: str,
    artifacts: Any,
    published_run: Path | None,
) -> dict[str, Any]:
    source_baseline_nll = artifacts.initial_validation.policy_nll
    candidate_nll = artifacts.restored_validation.policy_nll
    improved_over_source = candidate_nll < source_baseline_nll - 1e-12
    return {
        "schema_version": 1,
        "kind": "tactical-v3-ml-lab-dagger-continuation-training",
        "evidence_status": "unsealed-experimental",
        "semantics": "fresh-optimizer-weight-initialized-continuation",
        "source_run": str(Path(config.source_run).resolve()),
        "source_model_state_sha256": source.metadata.model_state_sha256,
        "source_corpus_sha256": source.metadata.corpus_sha256,
        "collection_sha256": collection_sha256,
        "corpus_sha256": corpus_sha256,
        "trainer": {
            "seed": config.seed,
            "batch_size": 256,
            "micro_batch_size": 32,
            "learning_rate": 3e-4,
            "max_epochs": 50,
            "patience_epochs": 5,
            "gradient_clip_norm": 1.0,
            "device": config.device,
        },
        "result": {
            "best_epoch": artifacts.loaded.metadata.best_epoch,
            "best_validation_policy_nll": (
                artifacts.loaded.metadata.best_validation_policy_nll
            ),
            "model_state_sha256": artifacts.loaded.metadata.model_state_sha256,
            "duration_seconds": artifacts.duration_seconds,
            "source_baseline_validation_policy_nll": source_baseline_nll,
            "candidate_validation_policy_nll": candidate_nll,
            "improved_over_source": improved_over_source,
            "publication_decision": (
                "published" if improved_over_source else "source-retained"
            ),
            "published_run": (
                str(published_run.resolve()) if published_run is not None else None
            ),
        },
    }


def _candidate_improves_source(artifacts: Any) -> bool:
    baseline = artifacts.initial_validation.policy_nll
    candidate = artifacts.restored_validation.policy_nll
    if not math.isfinite(baseline) or not math.isfinite(candidate):
        raise ValueError("tactical-v3 publication metrics must be finite")
    return candidate < baseline - 1e-12


def _mark_tracker_degraded(run_dir: Path, name: str, error: Exception) -> None:
    manifest = read_json(run_dir / "run.json")
    manifest["tracker_status"] = [{
        "name": name,
        "status": "degraded",
        "message": f"configure failed: {error}",
    }]
    atomic_write_json(run_dir / "run.json", manifest)


def run_structured_continuation(
    config: StructuredContinuationConfig,
    *,
    runs_root: Path,
    server_cmd: Sequence[str],
) -> Path:
    """Collect Greedy/model-distribution labels, fine-tune, and publish a strict run."""

    if type(config) is not StructuredContinuationConfig:
        raise TypeError("config must be StructuredContinuationConfig")
    config.validate()
    if config.device == "auto" or config.device.startswith("cuda"):
        os.environ.setdefault(
            "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True",
        )
    if config.device == "auto":
        config = replace(
            config,
            device="cuda:0" if torch.cuda.is_available() else "cpu",
        )
    source = validate_structured_run(Path(config.source_run))
    opponent = _resolve_opponent(config.opponent)
    run_dir: Path | None = None
    telemetry: _LiveTelemetry | None = None
    try:
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            identity = client.identity
            if source.metadata.identity != identity:
                raise ValueError(
                    "source policy identity does not match the selected tactical-v3 scenario"
                )
            if opponent.binding is not None:
                _validate_model_opponent(opponent.binding.resolved, identity)
            requested_device = torch.device(config.device)
            if requested_device.type == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("tactical-v3 continuation requested CUDA but it is unavailable")
            run_dir = _create_live_run(
                Path(runs_root), config, identity, source, opponent,
            )
            tensorboard_enabled = any(
                tracker.get("kind") == "tensorboard" for tracker in config.trackers
            )
            try:
                telemetry = _LiveTelemetry(
                    run_dir,
                    enabled=tensorboard_enabled,
                    train_target=config.train_label_target,
                    validation_target=config.validation_label_target,
                )
            except Exception as error:
                _mark_tracker_degraded(run_dir, "tensorboard", error)
                telemetry = _LiveTelemetry(
                    run_dir,
                    enabled=False,
                    train_target=config.train_label_target,
                    validation_target=config.validation_label_target,
                )
            update_run_state(
                run_dir,
                "running",
                pid=os.getpid(),
                latest_message="collecting tactical-v3 train labels",
            )
            _log(
                run_dir,
                "started structured DAgger continuation "
                f"source_state={source.metadata.model_state_sha256} "
                f"opponent={opponent.kind} collection_device=cpu "
                f"training_device={config.device}",
            )
            started = time.monotonic()
            outcomes: Counter[str] = Counter()
            seat_outcomes: dict[int, Counter[str]] = {0: Counter(), 1: Counter()}
            reasons: Counter[str] = Counter()
            disagreements = [0]
            fallbacks = [0]
            train_seed_start = 30_000_000 + config.seed * 20_000
            validation_seed_start = train_seed_start + 100_000
            train, train_evidence, train_labels, train_games = _collect_partition(
                client,
                source,
                opponent,
                config,
                run_dir,
                telemetry,
                partition="train",
                target=config.train_label_target,
                seed_start=train_seed_start,
                global_game_start=0,
                global_label_start=0,
                started=started,
                outcomes=outcomes,
                seat_outcomes=seat_outcomes,
                reasons=reasons,
                global_disagreements=disagreements,
                global_fallbacks=fallbacks,
            )
            if _stop_requested(run_dir):
                update_run_state(
                    run_dir, "stopped", pid=None,
                    latest_message="stopped during tactical-v3 train collection",
                )
                telemetry.phase(4.0, train_games)
                return run_dir
            telemetry.phase(1.0, train_games)
            validation, validation_evidence, validation_labels, validation_games = (
                _collect_partition(
                    client,
                    source,
                    opponent,
                    config,
                    run_dir,
                    telemetry,
                    partition="validation",
                    target=config.validation_label_target,
                    seed_start=validation_seed_start,
                    global_game_start=train_games,
                    global_label_start=train_labels,
                    started=started,
                    outcomes=outcomes,
                    seat_outcomes=seat_outcomes,
                    reasons=reasons,
                    global_disagreements=disagreements,
                    global_fallbacks=fallbacks,
                )
            )
        if run_dir is None or telemetry is None:
            raise RuntimeError("tactical-v3 continuation did not create a live run")
        if _stop_requested(run_dir):
            update_run_state(
                run_dir, "stopped", pid=None,
                latest_message="stopped during tactical-v3 validation collection",
            )
            telemetry.phase(4.0, train_games + validation_games)
            return run_dir
        train_records = _records_bytes(train)
        validation_records = _records_bytes(validation)
        collection_path, collection_sha256 = _write_collection_manifest(
            run_dir,
            config,
            identity,
            source,
            opponent,
            train_evidence,
            validation_evidence,
            train_records,
            validation_records,
        )
        corpus_value = {
            "schema_version": 1,
            "kind": "tactical-v3-ml-lab-dagger-continuation-corpus",
            "source_corpus_sha256": source.metadata.corpus_sha256,
            "collection_sha256": collection_sha256,
            "train_records_sha256": hashlib.sha256(train_records).hexdigest(),
            "validation_records_sha256": hashlib.sha256(validation_records).hexdigest(),
            "train_examples": sum(len(episode.records) for episode in train),
            "validation_examples": sum(
                len(episode.records) for episode in validation
            ),
        }
        corpus_sha256 = hashlib.sha256(_canonical_bytes(corpus_value)).hexdigest()
        train_examples = tuple(
            record.example for episode in train for record in episode.records
        )
        validation_examples = tuple(
            record.example for episode in validation for record in episode.records
        )
        update_run_state(
            run_dir,
            "running",
            pid=os.getpid(),
            latest_message=(
                f"training on {len(train_examples)} train and "
                f"{len(validation_examples)} validation labels"
            ),
        )
        telemetry.phase(2.0, train_games + validation_games)
        _log(
            run_dir,
            f"training started train_examples={len(train_examples)} "
            f"validation_examples={len(validation_examples)} device={config.device}",
        )
        trainer = TrainerConfig(
            seed=config.seed,
            batch_size=256,
            learning_rate=3e-4,
            max_epochs=50,
            patience_epochs=5,
            gradient_clip_norm=1.0,
            device=config.device,
        )
        artifacts = _train_pilot_dataset(
            identity,
            train_examples,
            validation_examples,
            corpus_sha256,
            run_dir / "training",
            config.seed,
            config.device,
            initial_policy=source,
            trainer_config=trainer,
            micro_batch_size=32,
            training_deadline_seconds=12 * 60 * 60,
            tensorboard_dir=run_dir / "tensorboard",
            log_path=run_dir / "train.log",
            tensorboard_enabled=telemetry.enabled,
            stop_requested=lambda: _stop_requested(run_dir),
        )
        candidate_improved = _candidate_improves_source(artifacts)
        published_run = (
            run_dir.parent / f"{config.run_name}-model"
            if candidate_improved else None
        )
        training_value = _training_manifest(
            config,
            source,
            collection_sha256,
            corpus_sha256,
            artifacts,
            published_run,
        )
        training_path = run_dir / "training" / "dagger-training.json"
        training_bytes = _canonical_bytes(training_value)
        with training_path.open("xb") as handle:
            handle.write(training_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        telemetry.final(artifacts, train_games + validation_games)
        if not candidate_improved:
            telemetry.phase(3.0, train_games + validation_games)
            update_run_state(
                run_dir,
                "completed",
                pid=None,
                timesteps=train_labels + validation_labels,
                latest_checkpoint=None,
                latest_checkpoint_step=None,
                source_baseline_validation_policy_nll=(
                    artifacts.initial_validation.policy_nll
                ),
                candidate_validation_policy_nll=(
                    artifacts.restored_validation.policy_nll
                ),
                rejected_candidate_checkpoint="training/checkpoints/best.pt",
                published_run=None,
                latest_message=(
                    "source retained: trained candidate did not improve "
                    "held-out validation policy NLL"
                ),
            )
            _log(
                run_dir,
                "training completed without publication "
                f"source_validation_policy_nll="
                f"{artifacts.initial_validation.policy_nll:.6f} "
                f"candidate_validation_policy_nll="
                f"{artifacts.restored_validation.policy_nll:.6f}",
            )
            return run_dir
        assert published_run is not None
        scenario_path = run_dir / "scenario.json"
        adopt_structured_run(
            published_run,
            source_checkpoint_path=artifacts.checkpoint,
            source_collection_path=collection_path,
            source_training_path=training_path,
            source_metrics_path=run_dir / "training" / "metrics.jsonl",
            training_scenario_path=scenario_path,
            expected_identity=identity,
            expected_checkpoint_sha256=_sha256_file(artifacts.checkpoint),
            expected_collection_sha256=collection_sha256,
            expected_training_sha256=hashlib.sha256(training_bytes).hexdigest(),
            expected_metrics_sha256=_sha256_file(
                run_dir / "training" / "metrics.jsonl"
            ),
            expected_scenario_sha256=_sha256_file(scenario_path),
        )
        final_step = artifacts.loaded.metadata.best_epoch
        telemetry.phase(3.0, train_games + validation_games)
        update_run_state(
            run_dir,
            "completed",
            pid=None,
            timesteps=train_labels + validation_labels,
            latest_checkpoint="training/checkpoints/best.pt",
            latest_checkpoint_step=final_step,
            best_epoch=final_step,
            best_validation_policy_nll=(
                artifacts.loaded.metadata.best_validation_policy_nll
            ),
            published_run=str(published_run.resolve()),
            latest_message=f"published strict tactical-v3 model: {published_run.name}",
        )
        _log(
            run_dir,
            f"training completed best_epoch={final_step} "
            f"validation_policy_nll={artifacts.loaded.metadata.best_validation_policy_nll:.6f} "
            f"published_run={published_run}",
        )
        return run_dir
    except PilotTrainingStopRequested:
        if run_dir is None:
            raise
        update_run_state(
            run_dir,
            "stopped",
            pid=None,
            latest_message="stopped during tactical-v3 training",
        )
        _log(run_dir, "training stopped at a safe metric or optimizer boundary")
        if telemetry is not None:
            telemetry.phase(4.0, 0)
        return run_dir
    except BaseException as error:
        if run_dir is not None and (run_dir / "run.json").is_file():
            try:
                update_run_state(
                    run_dir,
                    "failed",
                    pid=None,
                    latest_message=f"{type(error).__name__}: {error}",
                )
                _log(run_dir, f"training failed error={type(error).__name__}: {error}")
            except Exception:
                pass
        raise
    finally:
        if telemetry is not None:
            telemetry.close()


__all__ = [
    "StructuredContinuationConfig",
    "run_structured_continuation",
    "request_stop",
]
