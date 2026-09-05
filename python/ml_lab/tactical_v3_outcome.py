"""Complete-game outcome training for tactical-v3 research candidates.

This is a deliberately bounded on-policy vertical slice.  It freezes one
``TacticalV3Policy`` for a reciprocal rollout batch, samples only authoritative
legal candidates, applies one Monte-Carlo policy-gradient update, and then
evaluates the deterministic policy on a disjoint development seed bank.  Each
training trajectory is consumed exactly once; historical replay requires a
future importance-corrected/PPO trainer.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Literal

import torch

from .contracts import (
    MONITOR_HEADER,
    PROGRESS_HEADER,
    update_run_state,
    utc_now,
    validate_run_name,
    validate_tracker_specs,
)
from .controllers import ControllerResolver, normalize_controller_spec
from .io import atomic_write_bytes, atomic_write_json, read_json
from .scenarios import ResolvedScenario, resolve_scenario
from .tactical_v3_batching import collate_decisions
from .tactical_v3_checkpoint import semantic_identity_wire
from .tactical_v3_client import CandidateSelection, TacticalV3GymClient
from .tactical_v3_continuation import (
    _Opponent,
    _StartProfileScheduler,
    _contract,
    _resolve_opponent,
    _start_distribution,
    _validate_target_scenario_identity,
)
from .tactical_v3_controller import StructuredController, select_candidate
from .tactical_v3_layers import TacticalV3ModelConfig
from .tactical_v3_model import TacticalV3Policy
from .tactical_v3_outcome_checkpoint import (
    OUTCOME_ALGORITHM,
    OutcomeCheckpointMetadata,
    outcome_model_state_sha256,
    replace_outcome_checkpoint,
)
from .tactical_v3_outcome_objectives import (
    ENTROPY_COEFFICIENT,
    OUTCOME_COEFFICIENT,
    clip_outcome_gradients,
    outcome_policy_gradient_loss,
    sample_legal_candidates,
)
from .tactical_v3_pilot import (
    _pilot_configs,
    _validate_compatible_transfer_identity,
)
from .tactical_v3_schema import (
    TacticalV3Decision,
    TacticalV3Reward,
    TacticalV3SemanticIdentity,
)
from .tactical_v3_training import _batch_to_device
from .tactical_v3_trajectory import (
    ControllerProvenance,
    TacticalV3TrajectoryGame,
    TrajectoryDecisionRecord,
    publish_trajectory_game,
    write_trajectory_manifest,
)


_TRAIN_SEED_BASE = 40_000_000
_VALIDATION_SEED_BASE = 500_000_000
_SEED_RUN_STRIDE = 20_000
_DEFAULT_ROLLOUT_DECISIONS = 64
_DEFAULT_VALIDATION_GAMES = 32
_DEFAULT_VALIDATION_EVERY_UPDATES = 8
_DEFAULT_MICRO_BATCH_SIZE = 32
_DEFAULT_LEARNING_RATE = 3e-4
_MAX_GAMES_PER_ROLLOUT = 10_000


@dataclass(frozen=True, slots=True)
class OutcomeTrainingConfig:
    run_name: str
    scenario_file: Path
    opponent: str
    total_decisions: int
    seed: int
    device: str
    learner_seat: Literal["alternating", "0", "1"]
    trackers: tuple[Mapping[str, Any], ...]
    source_run: Path | None = None
    rollout_decisions: int = _DEFAULT_ROLLOUT_DECISIONS
    validation_games: int = _DEFAULT_VALIDATION_GAMES
    validation_every_updates: int = _DEFAULT_VALIDATION_EVERY_UPDATES
    micro_batch_size: int = _DEFAULT_MICRO_BATCH_SIZE
    learning_rate: float = _DEFAULT_LEARNING_RATE

    def validate(self) -> None:
        validate_run_name(self.run_name)
        if not Path(self.scenario_file).is_file():
            raise FileNotFoundError(self.scenario_file)
        if self.source_run is not None and not Path(self.source_run).is_dir():
            raise FileNotFoundError(self.source_run)
        for name in (
            "total_decisions", "rollout_decisions", "validation_games",
            "validation_every_updates", "micro_batch_size",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"outcome {name} must be a positive built-in int")
        if self.validation_games % 2 != 0:
            raise ValueError("outcome validation_games must be reciprocal and even")
        if type(self.seed) is not int or not 0 <= self.seed <= 20_000:
            raise ValueError("tactical-v3 seed must be an integer from 0 through 20000")
        if self.learner_seat not in {"alternating", "0", "1"}:
            raise ValueError("tactical-v3 learner seat is invalid")
        if type(self.device) is not str or not self.device.strip():
            raise ValueError("tactical-v3 device is required")
        if (
            type(self.learning_rate) is not float
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("outcome learning_rate must be finite and positive")
        normalize_controller_spec(self.opponent)
        validate_tracker_specs(list(self.trackers))
        unsupported = sorted({
            str(tracker.get("kind", ""))
            for tracker in self.trackers
            if tracker.get("kind") not in {"local", "tensorboard"}
        })
        if unsupported:
            raise ValueError(
                "tactical-v3 outcome training supports only local and TensorBoard "
                f"trackers, not {', '.join(unsupported)}"
            )


@dataclass(frozen=True, slots=True)
class OutcomeGameResult:
    game: TacticalV3TrajectoryGame
    terminal_reward: TacticalV3Reward


@dataclass(frozen=True, slots=True)
class OutcomeEvaluation:
    games: int
    wins: int
    losses: int
    draws: int
    truncations: int
    mean_return: float
    mean_learner_decisions: float
    choice_decisions: int
    forced_decisions: int
    seat_0_win_rate: float
    seat_1_win_rate: float
    opponent_artifact_sha256: str
    fixture_decisions: tuple[TacticalV3Decision, ...]

    @property
    def win_rate(self) -> float:
        return self.wins / self.games


@dataclass(frozen=True, slots=True)
class OutcomeUpdateMetrics:
    total_loss: float
    policy_loss: float
    outcome_loss: float
    entropy: float
    mean_return: float
    mean_choice_return: float
    mean_game_return: float
    mean_baseline: float
    mean_advantage: float
    mean_choice_baseline: float
    mean_choice_advantage: float
    approximate_kl: float
    gradient_norm: float
    decisions: int
    choice_decisions: int
    forced_decisions: int
    games: int
    wins: int
    losses: int
    draws: int


class _Telemetry:
    def __init__(self, run_dir: Path, *, enabled: bool, target: int) -> None:
        self.run_dir = Path(run_dir)
        self.writer = None
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(
                log_dir=str(self.run_dir / "tensorboard"), flush_secs=1,
            )
            self.record({
                "lifecycle/phase": 0.0,
                "training/decision_target": target,
                "training/algorithm_version": 2.0,
            }, 0)

    @property
    def enabled(self) -> bool:
        return self.writer is not None

    def _record_failure(self, error: BaseException) -> None:
        try:
            manifest = read_json(self.run_dir / "run.json")
            if not isinstance(manifest, dict):
                return
            statuses = [
                value for value in manifest.get("tracker_status", [])
                if not (
                    isinstance(value, Mapping)
                    and value.get("name") == "tensorboard:0"
                )
            ]
            statuses.append({
                "name": "tensorboard:0",
                "status": "degraded",
                "message": f"{type(error).__name__}: {error}",
            })
            manifest["tracker_status"] = statuses
            atomic_write_json(self.run_dir / "run.json", manifest)
        except Exception:
            # Training and its checkpoints remain authoritative even if both the
            # optional tracker and its diagnostic update are unavailable.
            pass

    def record(self, values: Mapping[str, float | int], step: int) -> None:
        if self.writer is None:
            return
        try:
            for name, value in values.items():
                self.writer.add_scalar(name, value, step)
            self.writer.flush()
        except Exception as error:
            try:
                self.writer.close()
            except Exception:
                pass
            finally:
                self.writer = None
            self._record_failure(error)

    def close(self) -> None:
        if self.writer is not None:
            writer = self.writer
            self.writer = None
            try:
                writer.flush()
                writer.close()
            except Exception as error:
                self._record_failure(error)


def _resolve_device(requested: str) -> str:
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid tactical-v3 outcome device {requested!r}") from error
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("tactical-v3 outcome device must be cpu or cuda")
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("tactical-v3 outcome training requested unavailable CUDA")
        index = torch.cuda.current_device() if device.index is None else device.index
        if index < 0 or index >= torch.cuda.device_count():
            raise RuntimeError(f"tactical-v3 outcome CUDA device {index} is unavailable")
        return f"cuda:{index}"
    return "cpu"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _scripted_hash(name: str) -> str:
    return hashlib.sha256(
        f"hexwars-tactical-v3-scripted-controller-v1:{name}".encode("utf-8")
    ).hexdigest()


def _actor_provenance(run_dir: Path, actor_hash: str) -> ControllerProvenance:
    return ControllerProvenance(
        kind="model",
        name=OUTCOME_ALGORITHM,
        source=str(Path(run_dir).resolve()),
        artifact_sha256=actor_hash,
    )


def _opponent_provenance(
    opponent: str | StructuredController,
) -> ControllerProvenance:
    if type(opponent) is str:
        return ControllerProvenance(
            kind="scripted",
            name=opponent,
            source=f"GymServer:{opponent}",
            artifact_sha256=_scripted_hash(opponent),
        )
    return ControllerProvenance(
        kind="model",
        name=opponent.algorithm,
        source=str(opponent.run_dir),
        artifact_sha256=_sha256_file(opponent.checkpoint_path),
    )


def _freeze_opponent(
    opponent: _Opponent,
    identity: TacticalV3SemanticIdentity,
) -> tuple[str | StructuredController, ControllerProvenance]:
    """Resolve one opponent revision and bind the provenance used for its games."""

    controller = opponent.controller_for_game(identity)
    return controller, _opponent_provenance(controller)


def _seat_sequence(value: str) -> tuple[int, ...]:
    if value == "0":
        return (0,)
    if value == "1":
        return (1,)
    return (0, 1)


def _write_csv_header(path: Path, fields: Sequence[str]) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        csv.writer(stream).writerow(fields)


def _append_csv(path: Path, fields: Sequence[str], row: Mapping[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())


def _append_jsonl(path: Path, value: Mapping[str, object]) -> None:
    data = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def _log(run_dir: Path, message: str) -> None:
    line = f"{utc_now()} {message}"
    with (Path(run_dir) / "train.log").open(
        "a", encoding="utf-8", buffering=1,
    ) as stream:
        stream.write(line + "\n")
    print(line, flush=True)


def _create_run(
    runs_root: Path,
    config: OutcomeTrainingConfig,
    scenario: ResolvedScenario,
    identity: TacticalV3SemanticIdentity,
    opponent: _Opponent,
    initialization: Mapping[str, object],
    validation_opponent: ControllerProvenance,
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
        raise FileExistsError(f"outcome run directory is not empty: {run_dir}")
    (run_dir / "checkpoints").mkdir()
    (run_dir / "trajectories" / "train").mkdir(parents=True)
    (run_dir / "trajectories" / "validation").mkdir()
    (run_dir / "training").mkdir()
    scenario.write(run_dir / "scenario.json")
    atomic_write_json(run_dir / "policy-identity.json", semantic_identity_wire(identity))
    atomic_write_json(run_dir / "control.json", {"request": None})
    atomic_write_json(run_dir / "evaluation.json", {})
    _write_csv_header(run_dir / "monitor.csv", MONITOR_HEADER)
    _write_csv_header(run_dir / "progress.csv", PROGRESS_HEADER)
    (run_dir / "metrics.jsonl").touch(exist_ok=False)
    (run_dir / "train.log").touch(exist_ok=False)
    created = utc_now()
    config_value = {
        "backend": OUTCOME_ALGORITHM,
        "algorithm": OUTCOME_ALGORITHM,
        "policy": "TacticalV3Policy",
        "run_name": config.run_name,
        "environment": "tactical-v3",
        "seed": config.seed,
        "total_timesteps": config.total_decisions,
        "rollout_decisions": config.rollout_decisions,
        "validation_games": config.validation_games,
        "validation_every_updates": config.validation_every_updates,
        "micro_batch_size": config.micro_batch_size,
        "learning_rate": config.learning_rate,
        "workers": 1,
        "device": config.device,
        "learner_seat": config.learner_seat,
        "opponent": dict(opponent.metadata),
        "trackers": [dict(value) for value in config.trackers],
        "initialization_source": (
            None if config.source_run is None
            else str(Path(config.source_run).resolve())
        ),
        "training_semantics": "on-policy-choice-normalized-complete-game-reinforce-v2",
    }
    manifest = {
        "schema_version": 1,
        "created_at": created,
        "updated_at": created,
        "state": "created",
        "pid": os.getpid(),
        "timesteps": 0,
        "target_step": config.total_decisions,
        "episodes": 0,
        "latest_message": "starting outcome-training baseline evaluation",
        "latest_checkpoint": None,
        "latest_checkpoint_step": None,
        "latest_training_checkpoint": None,
        "latest_training_checkpoint_step": None,
        "monitor_files": ["monitor.csv"],
        "config": config_value,
        "contract": _contract(identity),
        "scenario": {
            "path": "scenario.json",
            "schema_version": identity.scenario_schema_version,
        },
        "opponent_snapshot": dict(opponent.metadata),
        "validation_opponent_snapshot": asdict(validation_opponent),
        "initialization": dict(initialization),
        "policy_identity": "policy-identity.json",
        "trajectory_archive": {
            "path": "trajectories",
            "manifest": None,
            "train_games": 0,
            "validation_games": 0,
            "train_decisions": 0,
            "collected_train_decisions": 0,
        },
        "best_trajectory_manifest": None,
        "best_update": None,
        "best_validation_win_rate": None,
        "best_validation_mean_return": None,
        "best_validation_mean_decisions": None,
        "evidence_status": "unsealed-experimental",
        "tracker_status": [],
    }
    atomic_write_json(run_dir / "run.json", manifest)
    atomic_write_json(
        run_dir / "params.json",
        {"config": config_value, "contract": manifest["contract"]},
    )
    return run_dir


def _resolve_initial_source(
    config: OutcomeTrainingConfig,
    identity: TacticalV3SemanticIdentity,
    model_config: TacticalV3ModelConfig,
) -> tuple[StructuredController, dict[str, object]]:
    if config.source_run is None:
        raise ValueError("outcome initialization source is absent")
    resolved = ControllerResolver(expected_structured_hashes=(
        identity.encoding_hash, identity.capacity_hash,
    )).resolve(f"run:{Path(config.source_run)}")
    if (
        not isinstance(resolved.model, StructuredController)
        or resolved.algorithm not in {
            "structured_imitation", "structured_policy_gradient",
        }
        or resolved.path is None
        or not isinstance(resolved.contract, TacticalV3SemanticIdentity)
    ):
        raise ValueError("outcome initialization source must be a tactical-v3 run")
    _validate_compatible_transfer_identity(
        resolved.contract, identity, subject="outcome initialization source",
    )
    if resolved.model.policy.config != model_config:
        raise ValueError("outcome initialization source model config does not match")
    provenance = {
        "kind": "structured-policy-run",
        "algorithm": resolved.algorithm,
        "run": str(Path(config.source_run).resolve()),
        "checkpoint": str(resolved.path),
        "checkpoint_sha256": _sha256_file(resolved.path),
        "model_state_sha256": outcome_model_state_sha256(resolved.model.policy),
        "source_identity": semantic_identity_wire(resolved.contract),
    }
    return resolved.model, provenance


def _load_initial_policy(
    config: OutcomeTrainingConfig,
    identity: TacticalV3SemanticIdentity,
    effective_device: str,
) -> tuple[TacticalV3Policy, dict[str, object]]:
    model_config, _, _ = _pilot_configs(config.seed, effective_device)
    if config.source_run is None:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed % (2**63 - 1))
            model = TacticalV3Policy(model_config).to(effective_device)
        state_hash = outcome_model_state_sha256(model)
        return model, {
            "kind": "scratch",
            "seed": config.seed,
            "model_state_sha256": state_hash,
        }
    source, provenance = _resolve_initial_source(config, identity, model_config)
    model = TacticalV3Policy(model_config).to(effective_device)
    model.load_state_dict(source.policy.state_dict(), strict=True)
    if outcome_model_state_sha256(model) != provenance["model_state_sha256"]:
        raise ValueError("outcome initialization transfer changed model weights")
    return model, provenance


def _cpu_policy_snapshot(model: TacticalV3Policy) -> TacticalV3Policy:
    """Freeze the exact CPU policy that validation and publication will share."""

    state = {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }
    snapshot = TacticalV3Policy(model.config).to(device="cpu")
    snapshot.load_state_dict(state, strict=True)
    snapshot.eval()
    if outcome_model_state_sha256(snapshot) != outcome_model_state_sha256(model):
        raise ValueError("outcome CPU validation snapshot changed model weights")
    return snapshot


def _sample_seed(episode_seed: int, learner_seat: int, index: int) -> int:
    return (
        episode_seed * 1_000_003 + learner_seat * 65_537 + index * 97 + 17
    ) % (2**63 - 1)


def _greedy_sample(
    model: TacticalV3Policy, decision: TacticalV3Decision,
) -> tuple[int, float, float]:
    batch = _batch_to_device(
        collate_decisions((decision,), model.config.horizon_turns),
        next(model.parameters()).device,
    )
    with torch.inference_mode():
        selected = model.select(batch)[0]
    rows = [
        index for index, candidate in enumerate(decision.candidates)
        if candidate.candidate_id == selected.candidate_id
    ]
    if len(rows) != 1:
        raise ValueError("deterministic outcome selection is not a unique candidate")
    # The behavior policy is deterministic during validation: its selected action
    # has probability one and its entropy is zero.  The model's underlying
    # categorical distribution is not the behavior policy recorded in the archive.
    return selected.candidate_id, 0.0, 0.0


def _categorical_sample(
    model: TacticalV3Policy,
    decision: TacticalV3Decision,
    seed: int,
) -> tuple[int, float, float]:
    batch = _batch_to_device(
        collate_decisions((decision,), model.config.horizon_turns),
        next(model.parameters()).device,
    )
    with torch.inference_mode():
        output = model(batch)
        sampled = sample_legal_candidates(output, batch, seed=seed)[0]
    return sampled.candidate_id, sampled.log_probability, sampled.entropy


def _replay_bytes(client: TacticalV3GymClient, run_dir: Path) -> bytes:
    with tempfile.TemporaryDirectory(
        prefix=".outcome-replay-", dir=run_dir / "training",
    ) as directory:
        path = Path(directory) / "game.replay"
        client.save_replay(path)
        return path.read_bytes()


def collect_outcome_game(
    client: TacticalV3GymClient,
    model: TacticalV3Policy,
    opponent: str | StructuredController,
    *,
    run_dir: Path,
    partition: Literal["train", "validation"],
    game_index: int,
    episode_seed: int,
    profile_id: str,
    learner_seat: int,
    stochastic: bool,
    frozen_opponent_provenance: ControllerProvenance | None = None,
) -> OutcomeGameResult:
    """Play and materialize one complete game without updating ``model``."""

    identity = client.identity
    actor_hash = outcome_model_state_sha256(model)
    model.eval()
    structured_opponent = (
        opponent if isinstance(opponent, StructuredController) else None
    )
    opponent_provenance = (
        _opponent_provenance(opponent)
        if frozen_opponent_provenance is None
        else frozen_opponent_provenance
    )
    if type(opponent_provenance) is not ControllerProvenance:
        raise TypeError("frozen opponent provenance must be ControllerProvenance")
    if opponent_provenance != _opponent_provenance(opponent):
        raise ValueError("frozen opponent provenance does not match controller")
    if structured_opponent is None:
        if opponent not in {"random", "greedy", "passive"}:
            raise ValueError("outcome scripted opponent is unsupported")
        p0, p1 = (
            ("external", opponent)
            if learner_seat == 0 else (opponent, "external")
        )
    else:
        _validate_compatible_transfer_identity(
            structured_opponent.identity,
            identity,
            subject="outcome model opponent",
        )
        p0, p1 = "external", "external"
    view = client.duel_reset(
        episode_seed, p0, p1, learner_seat, profile_id, learner_seat,
    )
    if view.start_profile != profile_id or view.reference_seat != learner_seat:
        raise ValueError("outcome game profile or reference seat drifted")
    records: list[TrajectoryDecisionRecord] = []
    seen: set[int] = set()
    while not view.terminated and not view.truncated:
        if client.identity != identity:
            raise ValueError("outcome semantic identity drifted during game")
        if view.start_profile != profile_id or view.reference_seat != learner_seat:
            raise ValueError("outcome game profile or reference seat drifted")
        decision = view.decision
        if decision.decision_id in seen:
            raise ValueError("outcome game repeated a decision identity")
        seen.add(decision.decision_id)
        if view.seat != learner_seat:
            if structured_opponent is None:
                raise ValueError("scripted opponent exposed an external decision")
            selection = select_candidate(structured_opponent, view)
            view = client.duel_step(CandidateSelection(
                selection.decision_id, selection.candidate_id,
            ))
            continue
        if stochastic:
            candidate_id, log_probability, entropy = _categorical_sample(
                model,
                decision,
                _sample_seed(episode_seed, learner_seat, len(records)),
            )
            mode: Literal["categorical", "greedy"] = "categorical"
        else:
            candidate_id, log_probability, entropy = _greedy_sample(model, decision)
            mode = "greedy"
        successor = client.duel_step(CandidateSelection(
            decision.decision_id, candidate_id,
        ))
        records.append(TrajectoryDecisionRecord(
            trajectory_index=len(records),
            decision=decision,
            selected_candidate_id=candidate_id,
            behavior_mode=mode,
            log_probability=log_probability,
            entropy=entropy,
            successor_reward=successor.reward,
            terminated_after_selection=successor.terminated,
            truncated_after_selection=successor.truncated,
        ))
        view = successor
    if not view.reward.finalized:
        raise ValueError("outcome terminal reward must be finalized")
    if view.start_profile != profile_id or view.reference_seat != learner_seat:
        raise ValueError("outcome terminal profile or reference seat drifted")
    if view.terminated == view.truncated:
        raise ValueError("outcome game must terminate or truncate exactly once")
    fallback = client.duel_status()
    if fallback != 0:
        raise ValueError("outcome game internal fallback count must remain zero")
    if outcome_model_state_sha256(model) != actor_hash:
        raise ValueError("outcome actor weights changed during collection")
    game = TacticalV3TrajectoryGame(
        identity=identity,
        partition=partition,
        game_index=game_index,
        episode_seed=episode_seed,
        profile_id=profile_id,
        learner_seat=learner_seat,
        reference_seat=learner_seat,
        actor=_actor_provenance(run_dir, actor_hash),
        opponent=opponent_provenance,
        records=tuple(records),
        replay=_replay_bytes(client, Path(run_dir)),
        winner=view.winner,
        terminated=view.terminated,
        truncated=view.truncated,
        terminal_reward=view.reward,
        internal_fallback_count=fallback,
    )
    publish_trajectory_game(Path(run_dir) / "trajectories", game)
    return OutcomeGameResult(game, view.reward)


def _outcome_class(game: TacticalV3TrajectoryGame) -> int:
    return {"loss": 0, "draw": 1, "win": 2}[game.outcome]


def _optimization_rows(
    games: Sequence[OutcomeGameResult],
) -> tuple[
    tuple[TacticalV3Decision, ...], tuple[int, ...], tuple[float, ...],
    tuple[int, ...], tuple[float, ...],
]:
    decisions: list[TacticalV3Decision] = []
    rows: list[int] = []
    returns: list[float] = []
    outcomes: list[int] = []
    behavior_log_probabilities: list[float] = []
    actor_hashes: set[str] = set()
    for result in games:
        if result.game.partition != "train" or result.game.dataset_use != "optimization":
            raise ValueError("outcome optimization accepts only training trajectories")
        if result.game.terminal_reward != result.terminal_reward:
            raise ValueError("outcome optimization terminal reward does not match game")
        if not result.terminal_reward.finalized:
            raise ValueError("outcome optimization requires finalized rewards")
        actor_hashes.add(result.game.actor.artifact_sha256)
        for record in result.game.records:
            if record.behavior_mode != "categorical":
                raise ValueError(
                    "outcome optimization requires categorical behavior trajectories"
                )
            matching = [
                index for index, candidate in enumerate(record.decision.candidates)
                if candidate.candidate_id == record.selected_candidate_id
            ]
            if len(matching) != 1:
                raise ValueError("outcome optimization selected candidate is not unique")
            decisions.append(record.decision)
            rows.append(matching[0])
            returns.append(result.terminal_reward.total)
            outcomes.append(_outcome_class(result.game))
            behavior_log_probabilities.append(record.log_probability)
    if len(actor_hashes) != 1:
        raise ValueError("outcome rollout contains more than one actor version")
    if not decisions:
        raise ValueError("outcome rollout contains no learner decisions")
    return (
        tuple(decisions), tuple(rows), tuple(returns), tuple(outcomes),
        tuple(behavior_log_probabilities),
    )


def optimize_outcome_rollout(
    model: TacticalV3Policy,
    optimizer: torch.optim.Optimizer,
    games: Sequence[OutcomeGameResult],
    *,
    micro_batch_size: int,
) -> OutcomeUpdateMetrics:
    """Consume every decision in one frozen-policy rollout exactly once."""

    decisions, rows, returns, outcomes, behavior_logs = _optimization_rows(games)
    expected_actor = games[0].game.actor.artifact_sha256
    if outcome_model_state_sha256(model) != expected_actor:
        raise ValueError("outcome rollout actor hash does not match pre-update model")
    if type(micro_batch_size) is not int or micro_batch_size < 1:
        raise ValueError("outcome micro_batch_size must be positive")
    device = next(model.parameters()).device
    count = len(decisions)
    choice_rows = tuple(len(decision.candidates) > 1 for decision in decisions)
    choice_count = sum(choice_rows)
    totals = Counter()
    optimizer.zero_grad(set_to_none=True)
    model.train()
    for offset in range(0, count, micro_batch_size):
        stop = min(offset + micro_batch_size, count)
        batch = _batch_to_device(
            collate_decisions(
                decisions[offset:stop], model.config.horizon_turns,
            ),
            device,
        )
        output = model(batch)
        loss = outcome_policy_gradient_loss(
            output,
            batch,
            torch.tensor(rows[offset:stop], dtype=torch.int64, device=device),
            torch.tensor(returns[offset:stop], dtype=torch.float32, device=device),
            torch.tensor(outcomes[offset:stop], dtype=torch.int64, device=device),
        )
        row_count = stop - offset
        micro_choice_count = sum(choice_rows[offset:stop])
        row_weight = row_count / count
        choice_weight = (
            micro_choice_count / choice_count if choice_count else 0.0
        )
        scaled_loss = (
            loss.policy * choice_weight
            + OUTCOME_COEFFICIENT * loss.outcome * row_weight
            - ENTROPY_COEFFICIENT * loss.entropy * choice_weight
        )
        scaled_loss.backward()
        totals.update({
            "policy": float(loss.policy.detach()) * micro_choice_count,
            "outcome": float(loss.outcome.detach()) * row_count,
            "entropy": float(loss.entropy.detach()) * micro_choice_count,
            "baseline": float(loss.baselines.detach().mean()) * row_count,
            "advantage": float(loss.detached_advantages.detach().mean()) * row_count,
        })
        if micro_choice_count:
            totals.update({
                "choice_baseline": float(
                    loss.baselines.detach()[loss.choice_mask].sum()
                ),
                "choice_advantage": float(
                    loss.detached_advantages.detach()[loss.choice_mask].sum()
                ),
            })
    gradient_norm = float(clip_outcome_gradients(model.parameters(), max_norm=1.0))
    optimizer.step()
    model.eval()
    old_choice_logs: list[float] = []
    new_choice_logs: list[float] = []
    with torch.inference_mode():
        for offset in range(0, count, micro_batch_size):
            stop = min(offset + micro_batch_size, count)
            batch = _batch_to_device(
                collate_decisions(
                    decisions[offset:stop], model.config.horizon_turns,
                ),
                device,
            )
            output = model(batch)
            log_probabilities = torch.log_softmax(output.candidate_logits, dim=1)
            selected = torch.tensor(
                rows[offset:stop], dtype=torch.int64, device=device,
            )
            sample = torch.arange(stop - offset, device=device)
            selected_logs = (
                log_probabilities[sample, selected].detach().cpu().tolist()
            )
            for local_index, new_log in enumerate(selected_logs):
                decision_index = offset + local_index
                if choice_rows[decision_index]:
                    old_choice_logs.append(behavior_logs[decision_index])
                    new_choice_logs.append(float(new_log))
    approximate_kl = (
        sum(
            old - new
            for old, new in zip(old_choice_logs, new_choice_logs, strict=True)
        ) / choice_count
        if choice_count
        else 0.0
    )
    policy_loss = totals["policy"] / choice_count if choice_count else 0.0
    entropy = totals["entropy"] / choice_count if choice_count else 0.0
    outcome_loss = totals["outcome"] / count
    outcome_counts = Counter(result.game.outcome for result in games)
    return OutcomeUpdateMetrics(
        total_loss=(
            policy_loss
            + OUTCOME_COEFFICIENT * outcome_loss
            - ENTROPY_COEFFICIENT * entropy
        ),
        policy_loss=policy_loss,
        outcome_loss=outcome_loss,
        entropy=entropy,
        mean_return=sum(returns) / count,
        mean_choice_return=(
            sum(
                value for value, is_choice in zip(returns, choice_rows, strict=True)
                if is_choice
            ) / choice_count
            if choice_count
            else 0.0
        ),
        mean_game_return=(
            sum(result.terminal_reward.total for result in games) / len(games)
        ),
        mean_baseline=totals["baseline"] / count,
        mean_advantage=totals["advantage"] / count,
        mean_choice_baseline=(
            totals["choice_baseline"] / choice_count if choice_count else 0.0
        ),
        mean_choice_advantage=(
            totals["choice_advantage"] / choice_count if choice_count else 0.0
        ),
        approximate_kl=approximate_kl,
        gradient_norm=gradient_norm,
        decisions=count,
        choice_decisions=choice_count,
        forced_decisions=count - choice_count,
        games=len(games),
        wins=outcome_counts["win"],
        losses=outcome_counts["loss"],
        draws=outcome_counts["draw"],
    )


def _record_game(
    run_dir: Path,
    telemetry: _Telemetry,
    result: OutcomeGameResult,
    *,
    global_episode: int,
    scalar_step: int,
    elapsed: float,
) -> None:
    game = result.game
    _append_csv(run_dir / "monitor.csv", MONITOR_HEADER, {
        "worker_id": 0,
        "episode_index": global_episode,
        "episode_seed": game.episode_seed,
        "learner_seat": game.learner_seat,
        "episode_reward": result.terminal_reward.total,
        "episode_length": len(game.records),
        "elapsed_seconds": elapsed,
    })
    prefix = "rollout/train" if game.partition == "train" else "evaluation/game"
    telemetry.record({
        f"{prefix}_return": result.terminal_reward.total,
        f"{prefix}_win": float(game.outcome == "win"),
        f"{prefix}_loss": float(game.outcome == "loss"),
        f"{prefix}_draw": float(game.outcome == "draw"),
        f"{prefix}_learner_decisions": len(game.records),
        f"{prefix}_seat": game.learner_seat,
    }, scalar_step)


def _evaluate(
    client: TacticalV3GymClient,
    model: TacticalV3Policy,
    opponent: str | StructuredController,
    opponent_provenance: ControllerProvenance,
    config: OutcomeTrainingConfig,
    run_dir: Path,
    telemetry: _Telemetry,
    start_distribution: tuple[tuple[str, int], ...],
    *,
    validation_game_start: int,
    global_episode_start: int,
    update: int,
    started: float,
) -> tuple[OutcomeEvaluation, int]:
    results: list[OutcomeGameResult] = []
    scheduler = _StartProfileScheduler(start_distribution)
    for local_index in range(config.validation_games):
        seat = local_index % 2
        pair = local_index // 2
        # Advance once per reciprocal pair, not once per seat.
        if seat == 0:
            profile = scheduler.next_profile()
        result = collect_outcome_game(
            client,
            model,
            opponent,
            run_dir=run_dir,
            partition="validation",
            game_index=validation_game_start + local_index,
            episode_seed=(
                _VALIDATION_SEED_BASE + config.seed * _SEED_RUN_STRIDE + pair
            ),
            profile_id=profile,
            learner_seat=seat,
            stochastic=False,
            frozen_opponent_provenance=opponent_provenance,
        )
        if result.game.opponent != opponent_provenance:
            raise ValueError("outcome validation opponent provenance drifted")
        results.append(result)
        _record_game(
            run_dir, telemetry, result,
            global_episode=global_episode_start + local_index,
            scalar_step=update,
            elapsed=time.monotonic() - started,
        )
    counts = Counter(result.game.outcome for result in results)
    seat_counts = {
        seat: Counter(
            result.game.outcome
            for result in results if result.game.learner_seat == seat
        )
        for seat in (0, 1)
    }
    fixtures = tuple(
        record.decision
        for result in results
        for record in result.game.records
    )[:2]
    if not fixtures:
        raise RuntimeError("outcome validation produced no inference fixture decisions")
    validation_decisions = tuple(
        record.decision for result in results for record in result.game.records
    )
    choice_decisions = sum(
        len(decision.candidates) > 1 for decision in validation_decisions
    )
    evaluation = OutcomeEvaluation(
        games=len(results),
        wins=counts["win"],
        losses=counts["loss"],
        draws=counts["draw"],
        truncations=sum(result.game.truncated for result in results),
        mean_return=sum(result.terminal_reward.total for result in results) / len(results),
        mean_learner_decisions=sum(len(result.game.records) for result in results) / len(results),
        choice_decisions=choice_decisions,
        forced_decisions=len(validation_decisions) - choice_decisions,
        seat_0_win_rate=seat_counts[0]["win"] / max(sum(seat_counts[0].values()), 1),
        seat_1_win_rate=seat_counts[1]["win"] / max(sum(seat_counts[1].values()), 1),
        opponent_artifact_sha256=opponent_provenance.artifact_sha256,
        fixture_decisions=fixtures,
    )
    telemetry.record({
        "evaluation/win_rate": evaluation.win_rate,
        "evaluation/loss_rate": evaluation.losses / evaluation.games,
        "evaluation/draw_rate": evaluation.draws / evaluation.games,
        "evaluation/truncation_rate": evaluation.truncations / evaluation.games,
        "evaluation/mean_return": evaluation.mean_return,
        "evaluation/mean_learner_decisions": evaluation.mean_learner_decisions,
        "evaluation/choice_decisions": evaluation.choice_decisions,
        "evaluation/forced_decisions": evaluation.forced_decisions,
        "evaluation/seat_0_win_rate": evaluation.seat_0_win_rate,
        "evaluation/seat_1_win_rate": evaluation.seat_1_win_rate,
    }, update)
    return evaluation, len(results)


def _better(candidate: OutcomeEvaluation, best: OutcomeEvaluation) -> bool:
    if candidate.opponent_artifact_sha256 != best.opponent_artifact_sha256:
        raise ValueError(
            "outcome validation opponent changed across best-checkpoint comparison"
        )
    return (candidate.win_rate, candidate.mean_return) > (
        best.win_rate, best.mean_return,
    )


def _validation_due(
    *,
    update: int,
    optimized_decisions: int,
    total_decisions: int,
    validation_every_updates: int,
    checkpoint_stop: bool,
) -> bool:
    """Choose validation points without delaying a final or requested checkpoint."""

    return (
        checkpoint_stop
        or optimized_decisions >= total_decisions
        or update % validation_every_updates == 0
    )


def _checkpoint_candidate(
    run_dir: Path,
    model: TacticalV3Policy,
    identity: TacticalV3SemanticIdentity,
    initialization: Mapping[str, object],
    evaluation: OutcomeEvaluation,
    update: int,
    validation_game_start: int,
) -> tuple[Path, Path, OutcomeCheckpointMetadata]:
    archive_manifest = write_trajectory_manifest(
        run_dir / "trajectories", identity,
    )
    manifest_bytes = archive_manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    checkpoint_dir = run_dir / "checkpoints"
    snapshot = checkpoint_dir / f"trajectory-manifest-update-{update:06d}.json"
    checkpoint = checkpoint_dir / f"policy-update-{update:06d}.pt"
    atomic_write_bytes(snapshot, manifest_bytes)
    metadata = OutcomeCheckpointMetadata(
        format_version=1,
        algorithm=OUTCOME_ALGORITHM,
        identity=identity,
        model_config=model.config,
        trajectory_manifest_sha256=manifest_sha,
        model_state_sha256=outcome_model_state_sha256(model),
        update=update,
        validation_game_start=validation_game_start,
        validation_games=evaluation.games,
        validation_opponent_artifact_sha256=(
            evaluation.opponent_artifact_sha256
        ),
        validation_win_rate=evaluation.win_rate,
        validation_mean_return=evaluation.mean_return,
        validation_mean_decisions=evaluation.mean_learner_decisions,
        initialization=dict(initialization),
        published_device="cpu",
    )
    replace_outcome_checkpoint(
        checkpoint, model, metadata, evaluation.fixture_decisions,
    )
    return checkpoint, snapshot, metadata


def _stop_mode(run_dir: Path) -> str | None:
    try:
        value = read_json(run_dir / "control.json")
    except json.JSONDecodeError as error:
        raise ValueError("outcome control file contains malformed JSON") from error
    except UnicodeError as error:
        raise RuntimeError("outcome control file is unreadable") from error
    except OSError as error:
        raise RuntimeError("outcome control file is unreadable") from error
    if not isinstance(value, Mapping) or "request" not in value:
        raise ValueError("outcome control file must contain a request field")
    request = value["request"]
    if request is None:
        return None
    if (
        type(request) is not str
        or request not in {"stop_now", "stop_after_checkpoint"}
    ):
        raise ValueError(f"outcome control request is invalid: {request!r}")
    return request


def preflight_outcome_training(
    *,
    source_run: Path | None,
    scenario_file: Path,
    opponent: str,
    seed: int,
    device: str,
    learner_seat: Literal["alternating", "0", "1"],
    server_cmd: Sequence[str],
    rollout_decisions: int = _DEFAULT_ROLLOUT_DECISIONS,
    validation_games: int = _DEFAULT_VALIDATION_GAMES,
    validation_every_updates: int = _DEFAULT_VALIDATION_EVERY_UPDATES,
    micro_batch_size: int = _DEFAULT_MICRO_BATCH_SIZE,
    learning_rate: float = _DEFAULT_LEARNING_RATE,
) -> dict[str, object]:
    """Validate an outcome launch without creating a run or allocating a GPU model."""

    config = OutcomeTrainingConfig(
        run_name="preflight",
        scenario_file=Path(scenario_file),
        opponent=opponent,
        total_decisions=1,
        seed=seed,
        device=device,
        learner_seat=learner_seat,
        trackers=(),
        source_run=source_run,
        rollout_decisions=rollout_decisions,
        validation_games=validation_games,
        validation_every_updates=validation_every_updates,
        micro_batch_size=micro_batch_size,
        learning_rate=learning_rate,
    )
    config.validate()
    scenario = resolve_scenario(
        environment="tactical-v3",
        scenario_file=Path(scenario_file),
        template_id=None,
    )
    distribution = _start_distribution(scenario.document)
    effective_device = _resolve_device(device)
    opponent_value = _resolve_opponent(opponent)
    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        identity = client.identity
        _validate_target_scenario_identity(scenario, identity, distribution)
        model_config, _, _ = _pilot_configs(seed, effective_device)
        source_value: Mapping[str, object] | None = None
        if source_run is not None:
            _source, source_value = _resolve_initial_source(
                config, identity, model_config,
            )
        if opponent_value.binding is not None:
            opponent_value.controller_for_game(identity)
    return {
        "environment": "tactical-v3",
        "algorithm": OUTCOME_ALGORITHM,
        "source": source_value,
        "target": {
            "scenario_file": str(Path(scenario_file).resolve()),
            "scenario_id": scenario.template_id,
            "scenario_schema_version": scenario.schema_version,
            "contract_hash": identity.contract_hash,
            "encoding_hash": identity.encoding_hash,
            "capacity_hash": identity.capacity_hash,
            "start_distribution": [
                {"profile_id": profile, "basis_points": weight}
                for profile, weight in distribution
            ],
        },
        "opponent": dict(opponent_value.metadata),
        "learner_seat": learner_seat,
        "device": {"requested": device, "effective": effective_device},
        "model_config": asdict(model_config),
        "optimizer": {
            "kind": "reinforce",
            "gamma": 1.0,
            "learning_rate": config.learning_rate,
            "rollout_decisions": config.rollout_decisions,
            "validation_games": config.validation_games,
            "validation_every_updates": config.validation_every_updates,
            "micro_batch_size": config.micro_batch_size,
            "policy_normalization": "choice_decisions",
            "historical_trajectory_reuse": False,
        },
    }


def run_outcome_training(
    config: OutcomeTrainingConfig,
    *,
    runs_root: Path,
    server_cmd: Sequence[str],
) -> Path:
    """Run the complete-game outcome-learning lifecycle."""

    if type(config) is not OutcomeTrainingConfig:
        raise TypeError("config must be OutcomeTrainingConfig")
    config.validate()
    if config.device == "auto" or config.device.startswith("cuda"):
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    effective_device = _resolve_device(config.device)
    config = replace(config, device=effective_device)
    scenario = resolve_scenario(
        environment="tactical-v3",
        scenario_file=Path(config.scenario_file),
        template_id=None,
    )
    distribution = _start_distribution(scenario.document)
    opponent = _resolve_opponent(config.opponent)
    run_dir: Path | None = None
    telemetry: _Telemetry | None = None
    started = time.monotonic()
    try:
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            identity = client.identity
            _validate_target_scenario_identity(scenario, identity, distribution)
            model, initialization = _load_initial_policy(
                config, identity, effective_device,
            )
            validation_opponent, validation_opponent_provenance = _freeze_opponent(
                opponent, identity,
            )
            run_dir = _create_run(
                Path(runs_root), config, scenario, identity, opponent, initialization,
                validation_opponent_provenance,
            )
            tensorboard = any(
                tracker.get("kind") == "tensorboard" for tracker in config.trackers
            )
            try:
                telemetry = _Telemetry(
                    run_dir, enabled=tensorboard, target=config.total_decisions,
                )
            except Exception as error:
                manifest = read_json(run_dir / "run.json")
                manifest["tracker_status"] = [{
                    "name": "tensorboard:0", "status": "degraded",
                    "message": f"{type(error).__name__}: {error}",
                }]
                atomic_write_json(run_dir / "run.json", manifest)
                telemetry = _Telemetry(
                    run_dir, enabled=False, target=config.total_decisions,
                )
            telemetry.record({
                "configuration/rollout_decisions": config.rollout_decisions,
                "configuration/validation_games": config.validation_games,
                "configuration/validation_every_updates": (
                    config.validation_every_updates
                ),
                "configuration/micro_batch_size": config.micro_batch_size,
                "configuration/learning_rate": config.learning_rate,
            }, 0)
            update_run_state(
                run_dir, "running", pid=os.getpid(),
                latest_message="evaluating initial deterministic policy",
            )
            _log(
                run_dir,
                f"started {OUTCOME_ALGORITHM} opponent={opponent.kind} "
                f"device={effective_device} teacher_queries=0",
            )
            validation_game_index = 0
            global_episode = 0
            validation_model = _cpu_policy_snapshot(model)
            baseline, games_added = _evaluate(
                client, validation_model, validation_opponent,
                validation_opponent_provenance,
                config, run_dir, telemetry, distribution,
                validation_game_start=validation_game_index,
                global_episode_start=global_episode,
                update=0,
                started=started,
            )
            validation_game_index += games_added
            global_episode += games_added
            checkpoint, snapshot, metadata = _checkpoint_candidate(
                run_dir, validation_model, identity, initialization, baseline, 0, 0,
            )
            update_run_state(
                run_dir,
                "running",
                latest_checkpoint=checkpoint.relative_to(run_dir).as_posix(),
                latest_checkpoint_step=0,
                latest_training_checkpoint=checkpoint.relative_to(run_dir).as_posix(),
                latest_training_checkpoint_step=0,
                best_trajectory_manifest=snapshot.relative_to(run_dir).as_posix(),
                best_update=0,
                best_validation_win_rate=baseline.win_rate,
                best_validation_mean_return=baseline.mean_return,
                best_validation_mean_decisions=baseline.mean_learner_decisions,
                episodes=global_episode,
                trajectory_archive={
                    "path": "trajectories",
                    "manifest": "trajectories/manifest.json",
                    "train_games": 0,
                    "validation_games": validation_game_index,
                    "train_decisions": 0,
                    "collected_train_decisions": 0,
                },
                latest_message="baseline checkpoint is ready",
            )
            _append_jsonl(run_dir / "metrics.jsonl", {
                "schema_version": 1,
                "update": 0,
                "training": None,
                "validation": asdict(baseline) | {"fixture_decisions": len(baseline.fixture_decisions)},
                "improved": True,
                "checkpointed": True,
                "model_state_sha256": metadata.model_state_sha256,
            })
            if _stop_mode(run_dir) is not None:
                update_run_state(
                    run_dir, "stopped", pid=None,
                    latest_message="stopped at the durable baseline checkpoint",
                )
                telemetry.record({"lifecycle/phase": 4.0}, 0)
                return run_dir
            best = baseline
            best_update = 0
            optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
            optimized_decisions = 0
            train_game_index = 0
            train_pair_index = 0
            train_scheduler = _StartProfileScheduler(distribution)
            seats = _seat_sequence(config.learner_seat)
            update = 0
            telemetry.record({"lifecycle/phase": 1.0}, 0)
            while optimized_decisions < config.total_decisions:
                update += 1
                actor_hash = outcome_model_state_sha256(model)
                rollout: list[OutcomeGameResult] = []
                rollout_decisions = 0
                target = min(
                    config.rollout_decisions,
                    config.total_decisions - optimized_decisions,
                )
                games_this_rollout = 0
                deferred_stop = False
                immediate_stop = _stop_mode(run_dir) == "stop_now"
                while rollout_decisions < target and not immediate_stop:
                    if games_this_rollout >= _MAX_GAMES_PER_ROLLOUT:
                        raise RuntimeError(
                            "outcome rollout produced no usable learner decisions"
                        )
                    profile = train_scheduler.next_profile()
                    seed = (
                        _TRAIN_SEED_BASE
                        + config.seed * _SEED_RUN_STRIDE
                        + train_pair_index
                    )
                    pair_opponent, pair_opponent_provenance = _freeze_opponent(
                        opponent, identity,
                    )
                    for seat in seats:
                        result = collect_outcome_game(
                            client,
                            model,
                            pair_opponent,
                            run_dir=run_dir,
                            partition="train",
                            game_index=train_game_index,
                            episode_seed=seed,
                            profile_id=profile,
                            learner_seat=seat,
                            stochastic=True,
                            frozen_opponent_provenance=pair_opponent_provenance,
                        )
                        if result.game.opponent != pair_opponent_provenance:
                            raise ValueError(
                                "outcome reciprocal-pair opponent provenance drifted"
                            )
                        rollout.append(result)
                        train_game_index += 1
                        games_this_rollout += 1
                        global_episode += 1
                        rollout_decisions += len(result.game.records)
                        _record_game(
                            run_dir, telemetry, result,
                            global_episode=global_episode - 1,
                            scalar_step=optimized_decisions + rollout_decisions,
                            elapsed=time.monotonic() - started,
                        )
                        if outcome_model_state_sha256(model) != actor_hash:
                            raise ValueError("outcome actor changed within rollout batch")
                    train_pair_index += 1
                    stop_mode = _stop_mode(run_dir)
                    immediate_stop = stop_mode == "stop_now"
                    deferred_stop = stop_mode == "stop_after_checkpoint"
                    update_run_state(
                        run_dir,
                        "stopping" if immediate_stop or deferred_stop else "running",
                        episodes=global_episode,
                        latest_message=(
                            f"collected {rollout_decisions} decisions for update {update}"
                            + (
                                "; stopping before optimization"
                                if immediate_stop
                                else "; completing checkpoint stop"
                                if deferred_stop
                                else ""
                            )
                        ),
                    )
                    if immediate_stop or deferred_stop:
                        break
                if immediate_stop or rollout_decisions == 0:
                    write_trajectory_manifest(run_dir / "trajectories", identity)
                    update_run_state(
                        run_dir, "stopped", pid=None,
                        episodes=global_episode,
                        trajectory_archive={
                            "path": "trajectories",
                            "manifest": "trajectories/manifest.json",
                            "train_games": train_game_index,
                            "validation_games": validation_game_index,
                            "train_decisions": optimized_decisions,
                            "collected_train_decisions": (
                                optimized_decisions + rollout_decisions
                            ),
                        },
                        latest_message=(
                            "stopped before optimization; complete collected games "
                            "remain archived and the last checkpoint is unchanged"
                        ),
                    )
                    telemetry.record({"lifecycle/phase": 4.0}, optimized_decisions)
                    return run_dir
                telemetry.record({"lifecycle/phase": 2.0}, optimized_decisions)
                training = optimize_outcome_rollout(
                    model,
                    optimizer,
                    rollout,
                    micro_batch_size=config.micro_batch_size,
                )
                optimized_decisions += training.decisions
                telemetry.record({
                    "train/total_loss": training.total_loss,
                    "train/policy_loss": training.policy_loss,
                    "train/outcome_loss": training.outcome_loss,
                    "train/entropy": training.entropy,
                    "train/mean_return": training.mean_return,
                    "train/mean_choice_return": training.mean_choice_return,
                    "train/mean_game_return": training.mean_game_return,
                    "train/mean_baseline": training.mean_baseline,
                    "train/mean_advantage": training.mean_advantage,
                    "train/mean_choice_baseline": training.mean_choice_baseline,
                    "train/mean_choice_advantage": training.mean_choice_advantage,
                    "train/approximate_kl": training.approximate_kl,
                    "train/gradient_norm": training.gradient_norm,
                    "train/rollout_decisions": training.decisions,
                    "train/choice_decisions": training.choice_decisions,
                    "train/forced_decisions": training.forced_decisions,
                    "train/choice_rate": (
                        training.choice_decisions / training.decisions
                    ),
                    "train/rollout_games": training.games,
                    "train/game_win_rate": training.wins / training.games,
                    "train/game_loss_rate": training.losses / training.games,
                    "train/game_draw_rate": training.draws / training.games,
                }, optimized_decisions)
                stop_request = _stop_mode(run_dir)
                stop_now_after_update = stop_request == "stop_now"
                checkpoint_stop = not stop_now_after_update and (
                    deferred_stop or stop_request == "stop_after_checkpoint"
                )
                validation_due = not stop_now_after_update and _validation_due(
                    update=update,
                    optimized_decisions=optimized_decisions,
                    total_decisions=config.total_decisions,
                    validation_every_updates=config.validation_every_updates,
                    checkpoint_stop=checkpoint_stop,
                )
                fields: dict[str, object] = {
                    "timesteps": optimized_decisions,
                    "episodes": global_episode,
                }
                validation_value: Mapping[str, object] | None = None
                improved: bool | None = None
                if validation_due:
                    update_run_state(
                        run_dir,
                        "stopping" if checkpoint_stop else "running",
                        timesteps=optimized_decisions,
                        latest_message=f"evaluating completed update {update}",
                    )
                    telemetry.record({"lifecycle/phase": 3.0}, optimized_decisions)
                    validation_start = validation_game_index
                    validation_model = _cpu_policy_snapshot(model)
                    validation, games_added = _evaluate(
                        client, validation_model, validation_opponent,
                        validation_opponent_provenance,
                        config, run_dir, telemetry, distribution,
                        validation_game_start=validation_game_index,
                        global_episode_start=global_episode,
                        update=optimized_decisions,
                        started=started,
                    )
                    validation_game_index += games_added
                    global_episode += games_added
                    candidate_checkpoint, candidate_snapshot, candidate_metadata = (
                        _checkpoint_candidate(
                            run_dir, validation_model, identity, initialization,
                            validation, update, validation_start,
                        )
                    )
                    improved = _better(validation, best)
                    fields.update({
                        "episodes": global_episode,
                        "latest_training_checkpoint": (
                            candidate_checkpoint.relative_to(run_dir).as_posix()
                        ),
                        "latest_training_checkpoint_step": update,
                        "latest_message": f"completed outcome update {update}",
                    })
                    if improved:
                        best = validation
                        best_update = update
                        fields.update({
                            "latest_checkpoint": (
                                candidate_checkpoint.relative_to(run_dir).as_posix()
                            ),
                            "latest_checkpoint_step": update,
                            "best_trajectory_manifest": (
                                candidate_snapshot.relative_to(run_dir).as_posix()
                            ),
                            "best_update": update,
                            "best_validation_win_rate": validation.win_rate,
                            "best_validation_mean_return": validation.mean_return,
                            "best_validation_mean_decisions": (
                                validation.mean_learner_decisions
                            ),
                        })
                    validation_value = asdict(validation) | {
                        "fixture_decisions": len(validation.fixture_decisions),
                    }
                    model_hash = candidate_metadata.model_state_sha256
                    log_message = (
                        f"update={update} decisions={optimized_decisions} "
                        f"validation_wins={validation.wins}/{validation.games} "
                        f"mean_return={validation.mean_return:.6f} "
                        f"improved={improved}"
                    )
                else:
                    fields["latest_message"] = (
                        f"completed optimizer update {update}; validation pending"
                    )
                    model_hash = outcome_model_state_sha256(model)
                    log_message = (
                        f"update={update} decisions={optimized_decisions} "
                        "validation=deferred"
                    )
                trajectory_archive = {
                    "path": "trajectories",
                    "manifest": (
                        "trajectories/manifest.json" if validation_due else None
                    ),
                    "train_games": train_game_index,
                    "validation_games": validation_game_index,
                    "train_decisions": optimized_decisions,
                    "collected_train_decisions": optimized_decisions,
                }
                fields["trajectory_archive"] = trajectory_archive
                update_run_state(
                    run_dir,
                    "stopping"
                    if checkpoint_stop or stop_now_after_update
                    else "running",
                    **fields,
                )
                _append_jsonl(run_dir / "metrics.jsonl", {
                    "schema_version": 1,
                    "update": update,
                    "training": asdict(training),
                    "validation": validation_value,
                    "improved": improved,
                    "checkpointed": validation_due,
                    "model_state_sha256": model_hash,
                })
                _append_csv(run_dir / "progress.csv", PROGRESS_HEADER, {
                    "timestamp": utc_now(),
                    "timesteps": optimized_decisions,
                    "episodes": global_episode,
                    "mean_reward": training.mean_game_return,
                    "steps_per_second": optimized_decisions / max(
                        time.monotonic() - started, 1e-9,
                    ),
                })
                _log(run_dir, log_message)
                telemetry.record({"lifecycle/phase": 1.0}, optimized_decisions)
                latest_stop_request = _stop_mode(run_dir)
                if stop_now_after_update or latest_stop_request == "stop_now":
                    if not validation_due:
                        write_trajectory_manifest(
                            run_dir / "trajectories", identity,
                        )
                        trajectory_archive["manifest"] = (
                            "trajectories/manifest.json"
                        )
                    update_run_state(
                        run_dir, "stopped", pid=None,
                        trajectory_archive=trajectory_archive,
                        latest_message=(
                            f"stopped after durable outcome update {update}"
                            if validation_due
                            else (
                                "stopped after the current optimizer update; "
                                "the last validated checkpoint is unchanged"
                            )
                        ),
                    )
                    telemetry.record({"lifecycle/phase": 4.0}, optimized_decisions)
                    return run_dir
                if checkpoint_stop or (
                    validation_due
                    and latest_stop_request == "stop_after_checkpoint"
                ):
                    update_run_state(
                        run_dir, "stopped", pid=None,
                        latest_message=(
                            f"stopped after durable outcome update {update}; "
                            f"best update {best_update}"
                        ),
                    )
                    telemetry.record({"lifecycle/phase": 4.0}, optimized_decisions)
                    return run_dir
            update_run_state(
                run_dir,
                "completed",
                pid=None,
                timesteps=optimized_decisions,
                latest_message=(
                    f"outcome training completed; best update {best_update}"
                ),
            )
            telemetry.record({"lifecycle/phase": 4.0}, optimized_decisions)
            return run_dir
    except BaseException as error:
        if run_dir is not None and (run_dir / "run.json").is_file():
            try:
                update_run_state(
                    run_dir, "failed", pid=None,
                    latest_message=f"{type(error).__name__}: {error}",
                )
            except Exception:
                pass
        raise
    finally:
        if telemetry is not None:
            telemetry.close()
