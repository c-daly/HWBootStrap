"""Strict inference checkpoints for outcome-trained tactical-v3 policies.

Outcome-trained policies share the :class:`TacticalV3Policy` inference contract
with structured-imitation policies, but they deliberately do not share its
teacher-labelled checkpoint metadata.  This module keeps that provenance
boundary explicit while retaining the same strict CPU, hash, and inference
fixture checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal, Mapping

import torch

from .io import read_json
from .scenarios import resolve_scenario
from .tactical_v3_batching import collate_decisions
from .tactical_v3_checkpoint import (
    _state_sha256,
    _state_wire,
    _sync_directory,
    _validate_state,
    _write_checkpoint,
    semantic_identity_wire,
)
from .tactical_v3_layers import TacticalV3ModelConfig
from .tactical_v3_model import CandidateIdentity, TacticalV3Policy
from .tactical_v3_schema import (
    TacticalV3Decision,
    TacticalV3SemanticIdentity,
    parse_decision,
    parse_spaces,
)
from .tactical_v3_trajectory import (
    TacticalV3TrajectoryGame,
    load_trajectory_game,
    validate_trajectory_manifest_snapshot,
)


OUTCOME_ALGORITHM = "structured_policy_gradient"
_FORMAT_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset({
    "format_version", "metadata", "state_dict", "inference_fixture",
})
_METADATA_FIELDS = frozenset({
    "format_version", "algorithm", "identity", "model_config",
    "trajectory_manifest_sha256", "model_state_sha256", "update",
    "validation_game_start", "validation_games",
    "validation_opponent_artifact_sha256",
    "validation_win_rate", "validation_mean_return",
    "validation_mean_decisions", "initialization", "published_device",
})
_FIXTURE_FIELDS = frozenset({
    "decisions", "valid_candidate_logits", "selected_identities",
})
_SCRATCH_INITIALIZATION_FIELDS = frozenset({
    "kind", "seed", "model_state_sha256",
})
_SOURCE_INITIALIZATION_FIELDS = frozenset({
    "kind", "algorithm", "run", "checkpoint", "checkpoint_sha256",
    "model_state_sha256", "source_identity",
})
_RUN_STATES = frozenset({
    "created", "running", "stopping", "stopped", "completed",
})


@dataclass(frozen=True, slots=True)
class OutcomeCheckpointMetadata:
    format_version: int
    algorithm: Literal["structured_policy_gradient"]
    identity: TacticalV3SemanticIdentity
    model_config: TacticalV3ModelConfig
    trajectory_manifest_sha256: str
    model_state_sha256: str
    update: int
    validation_game_start: int
    validation_games: int
    validation_opponent_artifact_sha256: str
    validation_win_rate: float
    validation_mean_return: float
    validation_mean_decisions: float
    initialization: Mapping[str, object]
    published_device: Literal["cpu"]


@dataclass(frozen=True, slots=True)
class OutcomeInferenceFixture:
    decisions: tuple[TacticalV3Decision, ...]
    valid_candidate_logits: tuple[tuple[float, ...], ...]
    selected_identities: tuple[CandidateIdentity, ...]


@dataclass(frozen=True, slots=True)
class LoadedOutcomePolicy:
    model: TacticalV3Policy
    metadata: OutcomeCheckpointMetadata
    fixture: OutcomeInferenceFixture


def _exact_mapping(
    value: object, expected: frozenset[str], label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != expected
        or not all(type(key) is str for key in value)
    ):
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}")
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be a built-in int at least {minimum}")
    return value


def _finite(value: object, label: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite built-in float")
    return value


def _sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _plain(value: object, label: str = "value") -> object:
    if value is None or type(value) in {str, int, bool}:
        return value
    if type(value) is float:
        return _finite(value, label)
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError(f"{label} mapping keys must be built-in str")
        return {key: _plain(item, f"{label}.{key}") for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_plain(item, f"{label}[]") for item in value]
    raise TypeError(f"{label} contains unsupported {type(value).__name__}")


def _model_config_wire(config: TacticalV3ModelConfig) -> dict[str, object]:
    if type(config) is not TacticalV3ModelConfig:
        raise TypeError("model_config must be TacticalV3ModelConfig")
    value = _plain(asdict(config), "model_config")
    assert isinstance(value, dict)
    return value


def _model_config_from_wire(value: object) -> TacticalV3ModelConfig:
    expected = frozenset(field.name for field in fields(TacticalV3ModelConfig))
    data = _exact_mapping(value, expected, "metadata.model_config")
    parsed: dict[str, object] = {}
    for name, item in data.items():
        if name == "horizon_turns":
            if type(item) is not list or not item:
                raise ValueError("metadata.model_config.horizon_turns must be non-empty")
            parsed[name] = tuple(
                _integer(row, "metadata.model_config.horizon_turns[]", minimum=1)
                for row in item
            )
        else:
            parsed[name] = _integer(item, f"metadata.model_config.{name}", minimum=1)
    return TacticalV3ModelConfig(**parsed)  # type: ignore[arg-type]


def _nonempty_string(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a non-empty built-in str")
    return value


def _initialization_wire(value: object) -> dict[str, object]:
    plain = _plain(value, "metadata.initialization")
    if not isinstance(plain, dict):
        raise ValueError("metadata.initialization must be a mapping")
    kind = plain.get("kind")
    if kind == "scratch":
        data = _exact_mapping(
            plain,
            _SCRATCH_INITIALIZATION_FIELDS,
            "metadata.initialization",
        )
        seed = _integer(data["seed"], "metadata.initialization.seed")
        if seed > 20_000:
            raise ValueError(
                "metadata.initialization.seed must be at most 20000"
            )
        return {
            "kind": "scratch",
            "seed": seed,
            "model_state_sha256": _sha256(
                data["model_state_sha256"],
                "metadata.initialization.model_state_sha256",
            ),
        }
    if kind == "structured-policy-run":
        data = _exact_mapping(
            plain,
            _SOURCE_INITIALIZATION_FIELDS,
            "metadata.initialization",
        )
        algorithm = data["algorithm"]
        if (
            type(algorithm) is not str
            or algorithm not in {"structured_imitation", OUTCOME_ALGORITHM}
        ):
            raise ValueError(
                "metadata.initialization.algorithm is unsupported"
            )
        try:
            source_identity = parse_spaces(data["source_identity"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "metadata.initialization.source_identity is invalid"
            ) from error
        return {
            "kind": "structured-policy-run",
            "algorithm": algorithm,
            "run": _nonempty_string(
                data["run"], "metadata.initialization.run",
            ),
            "checkpoint": _nonempty_string(
                data["checkpoint"], "metadata.initialization.checkpoint",
            ),
            "checkpoint_sha256": _sha256(
                data["checkpoint_sha256"],
                "metadata.initialization.checkpoint_sha256",
            ),
            "model_state_sha256": _sha256(
                data["model_state_sha256"],
                "metadata.initialization.model_state_sha256",
            ),
            "source_identity": semantic_identity_wire(source_identity),
        }
    raise ValueError("metadata.initialization.kind is invalid")


def _metadata_wire(metadata: OutcomeCheckpointMetadata) -> dict[str, object]:
    if type(metadata) is not OutcomeCheckpointMetadata:
        raise TypeError("metadata must be OutcomeCheckpointMetadata")
    if (
        metadata.format_version != _FORMAT_VERSION
        or metadata.algorithm != OUTCOME_ALGORITHM
        or metadata.published_device != "cpu"
    ):
        raise ValueError("unsupported outcome checkpoint metadata format")
    win_rate = _finite(
        metadata.validation_win_rate,
        "metadata.validation_win_rate",
    )
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("metadata.validation_win_rate must be within [0, 1]")
    mean_decisions = _finite(
        metadata.validation_mean_decisions,
        "metadata.validation_mean_decisions",
    )
    if mean_decisions <= 0.0:
        raise ValueError("metadata.validation_mean_decisions must be positive")
    initialization = _initialization_wire(metadata.initialization)
    validation_games = _integer(
        metadata.validation_games, "metadata.validation_games", minimum=2,
    )
    if validation_games % 2 != 0:
        raise ValueError("metadata.validation_games must be reciprocal and even")
    validation_start = _integer(
        metadata.validation_game_start,
        "metadata.validation_game_start",
    )
    if validation_start % validation_games != 0:
        raise ValueError(
            "metadata.validation_game_start must align with the sweep schedule"
        )
    return {
        "format_version": _FORMAT_VERSION,
        "algorithm": OUTCOME_ALGORITHM,
        "identity": semantic_identity_wire(metadata.identity),
        "model_config": _model_config_wire(metadata.model_config),
        "trajectory_manifest_sha256": _sha256(
            metadata.trajectory_manifest_sha256,
            "metadata.trajectory_manifest_sha256",
        ),
        "model_state_sha256": _sha256(
            metadata.model_state_sha256, "metadata.model_state_sha256",
        ),
        "update": _integer(metadata.update, "metadata.update"),
        "validation_game_start": validation_start,
        "validation_games": validation_games,
        "validation_opponent_artifact_sha256": _sha256(
            metadata.validation_opponent_artifact_sha256,
            "metadata.validation_opponent_artifact_sha256",
        ),
        "validation_win_rate": win_rate,
        "validation_mean_return": _finite(
            metadata.validation_mean_return,
            "metadata.validation_mean_return",
        ),
        "validation_mean_decisions": mean_decisions,
        "initialization": initialization,
        "published_device": "cpu",
    }


def _metadata_from_wire(value: object) -> OutcomeCheckpointMetadata:
    data = _exact_mapping(value, _METADATA_FIELDS, "metadata")
    if _integer(data["format_version"], "metadata.format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported outcome checkpoint metadata version")
    if data["algorithm"] != OUTCOME_ALGORITHM:
        raise ValueError("unsupported outcome checkpoint algorithm")
    if data["published_device"] != "cpu":
        raise ValueError("outcome checkpoint published_device must be cpu")
    initialization = _initialization_wire(data["initialization"])
    win_rate = _finite(
        data["validation_win_rate"], "metadata.validation_win_rate",
    )
    if not 0.0 <= win_rate <= 1.0:
        raise ValueError("metadata.validation_win_rate must be within [0, 1]")
    mean_decisions = _finite(
        data["validation_mean_decisions"],
        "metadata.validation_mean_decisions",
    )
    if mean_decisions <= 0.0:
        raise ValueError("metadata.validation_mean_decisions must be positive")
    update = _integer(data["update"], "metadata.update")
    validation_games = _integer(
        data["validation_games"], "metadata.validation_games", minimum=2,
    )
    if validation_games % 2 != 0:
        raise ValueError("metadata.validation_games must be reciprocal and even")
    validation_start = _integer(
        data["validation_game_start"], "metadata.validation_game_start",
    )
    if validation_start % validation_games != 0:
        raise ValueError(
            "metadata.validation_game_start must align with the sweep schedule"
        )
    return OutcomeCheckpointMetadata(
        _FORMAT_VERSION,
        OUTCOME_ALGORITHM,
        parse_spaces(data["identity"]),
        _model_config_from_wire(data["model_config"]),
        _sha256(
            data["trajectory_manifest_sha256"],
            "metadata.trajectory_manifest_sha256",
        ),
        _sha256(data["model_state_sha256"], "metadata.model_state_sha256"),
        update,
        validation_start,
        validation_games,
        _sha256(
            data["validation_opponent_artifact_sha256"],
            "metadata.validation_opponent_artifact_sha256",
        ),
        win_rate,
        _finite(
            data["validation_mean_return"],
            "metadata.validation_mean_return",
        ),
        mean_decisions,
        initialization,
        "cpu",
    )


def _fixture_outputs(
    model: TacticalV3Policy,
    decisions: tuple[TacticalV3Decision, ...],
    identity: TacticalV3SemanticIdentity,
) -> tuple[tuple[tuple[float, ...], ...], tuple[CandidateIdentity, ...]]:
    if not decisions:
        raise ValueError("outcome inference fixture must contain decisions")
    model.eval()
    batch = collate_decisions(decisions, model.config.horizon_turns, identity=identity)
    with torch.inference_mode():
        output = model(batch)
        selected = model.select(batch)
    logits: list[tuple[float, ...]] = []
    for sample in range(len(decisions)):
        valid = batch.candidates.mask[sample]
        row = output.candidate_logits[sample, valid].detach().cpu()
        if not bool(torch.isfinite(row).all()):
            raise FloatingPointError("outcome inference fixture logits are nonfinite")
        logits.append(tuple(float(item) for item in row.tolist()))
    return tuple(logits), selected


def _fixture_wire(
    model: TacticalV3Policy,
    decisions: tuple[TacticalV3Decision, ...],
    identity: TacticalV3SemanticIdentity,
) -> dict[str, object]:
    logits, selected = _fixture_outputs(model, decisions, identity)
    return {
        "decisions": [_plain(asdict(item), "fixture.decision") for item in decisions],
        "valid_candidate_logits": [list(row) for row in logits],
        "selected_identities": [asdict(item) for item in selected],
    }


def _fixture_from_wire(
    value: object,
    identity: TacticalV3SemanticIdentity,
) -> OutcomeInferenceFixture:
    data = _exact_mapping(value, _FIXTURE_FIELDS, "inference_fixture")
    raw_decisions = data["decisions"]
    raw_logits = data["valid_candidate_logits"]
    raw_selected = data["selected_identities"]
    if type(raw_decisions) is not list or not raw_decisions:
        raise ValueError("inference_fixture.decisions must be non-empty")
    if type(raw_logits) is not list or type(raw_selected) is not list:
        raise TypeError("inference fixture outputs must be lists")
    decisions = tuple(parse_decision(item, identity) for item in raw_decisions)
    if len(raw_logits) != len(decisions) or len(raw_selected) != len(decisions):
        raise ValueError("inference fixture row counts must agree")
    logits: list[tuple[float, ...]] = []
    selected: list[CandidateIdentity] = []
    for index, (row, chosen) in enumerate(zip(raw_logits, raw_selected, strict=True)):
        if type(row) is not list or not row:
            raise ValueError(f"inference_fixture.valid_candidate_logits[{index}] is empty")
        logits.append(tuple(
            _finite(item, f"inference_fixture.valid_candidate_logits[{index}][]")
            for item in row
        ))
        chosen_data = _exact_mapping(
            chosen,
            frozenset({"decision_id", "candidate_id"}),
            f"inference_fixture.selected_identities[{index}]",
        )
        selected.append(CandidateIdentity(
            _integer(chosen_data["decision_id"], "fixture decision_id"),
            _integer(chosen_data["candidate_id"], "fixture candidate_id"),
        ))
    return OutcomeInferenceFixture(decisions, tuple(logits), tuple(selected))


def save_outcome_checkpoint(
    path: Path,
    model: TacticalV3Policy,
    metadata: OutcomeCheckpointMetadata,
    fixture_decisions: tuple[TacticalV3Decision, ...],
) -> Path:
    """Create one immutable, CPU-only, self-validating checkpoint."""

    state = _state_wire(model)
    if metadata.model_config != model.config:
        raise ValueError("outcome metadata model config does not match model")
    if metadata.model_state_sha256 != _state_sha256(state):
        raise ValueError("outcome metadata model state hash does not match model")
    cpu_model = TacticalV3Policy(model.config).to(device="cpu")
    cpu_model.load_state_dict(state, strict=True)
    cpu_model.eval()
    return _write_checkpoint(Path(path), {
        "format_version": _FORMAT_VERSION,
        "metadata": _metadata_wire(metadata),
        "state_dict": state,
        "inference_fixture": _fixture_wire(
            cpu_model, tuple(fixture_decisions), metadata.identity,
        ),
    })


def load_outcome_checkpoint(
    path: Path,
    expected_encoding_hash: str,
    expected_capacity_hash: str,
) -> LoadedOutcomePolicy:
    """Load and replay-validate an outcome checkpoint on CPU."""

    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _TOP_LEVEL_FIELDS
        or not all(type(key) is str for key in raw)
    ):
        raise ValueError("outcome checkpoint top-level inventory is invalid")
    if _integer(raw["format_version"], "format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported outcome checkpoint format_version")
    metadata = _metadata_from_wire(raw["metadata"])
    if metadata.identity.encoding_hash != _sha256(
        expected_encoding_hash, "expected encoding hash",
    ):
        raise ValueError("outcome checkpoint encoding hash does not match")
    if metadata.identity.capacity_hash != _sha256(
        expected_capacity_hash, "expected capacity hash",
    ):
        raise ValueError("outcome checkpoint capacity hash does not match")
    state = _validate_state(raw["state_dict"])
    for name, tensor in state.items():
        if (
            (tensor.is_floating_point() or tensor.is_complex())
            and not bool(torch.isfinite(tensor).all())
        ):
            raise ValueError(
                f"outcome checkpoint state tensor {name!r} is nonfinite"
            )
    if _state_sha256(state) != metadata.model_state_sha256:
        raise ValueError("outcome checkpoint model state hash does not match metadata")
    model = TacticalV3Policy(metadata.model_config).to(device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    if outcome_model_state_sha256(model) != metadata.model_state_sha256:
        raise ValueError(
            "loaded outcome model state hash does not match metadata"
        )
    fixture = _fixture_from_wire(raw["inference_fixture"], metadata.identity)
    actual_logits, actual_selected = _fixture_outputs(
        model, fixture.decisions, metadata.identity,
    )
    if (
        actual_logits != fixture.valid_candidate_logits
        or actual_selected != fixture.selected_identities
    ):
        raise ValueError("outcome checkpoint inference fixture does not replay exactly")
    return LoadedOutcomePolicy(model, metadata, fixture)


def replace_outcome_checkpoint(
    path: Path,
    model: TacticalV3Policy,
    metadata: OutcomeCheckpointMetadata,
    fixture_decisions: tuple[TacticalV3Decision, ...],
) -> Path:
    """Validate a staged checkpoint, then atomically replace the live best file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("outcome live checkpoint path must not be a symlink")
    stage = Path(tempfile.mkdtemp(prefix=f".{path.name}.stage-", dir=path.parent))
    try:
        staged = save_outcome_checkpoint(
            stage / path.name, model, metadata, tuple(fixture_decisions),
        )
        loaded = load_outcome_checkpoint(
            staged,
            metadata.identity.encoding_hash,
            metadata.identity.capacity_hash,
        )
        if loaded.metadata != metadata:
            raise ValueError("staged outcome checkpoint metadata changed")
        os.replace(staged, path)
        _sync_directory(path.parent)
        return path
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def outcome_model_state_sha256(model: TacticalV3Policy) -> str:
    return _state_sha256(_state_wire(model))


def _checkpoint_validation_games(
    snapshot: Mapping[str, object],
    archive_root: Path,
    metadata: OutcomeCheckpointMetadata,
) -> tuple[TacticalV3TrajectoryGame, ...]:
    """Recover and bind the exact reciprocal evaluation behind a checkpoint."""

    partitions = snapshot.get("partitions")
    if not isinstance(partitions, Mapping):  # Already rejected by snapshot validation.
        raise ValueError("outcome trajectory snapshot partitions are invalid")
    validation = partitions.get("validation")
    if not isinstance(validation, Mapping):
        raise ValueError("outcome trajectory snapshot validation partition is invalid")
    entries = validation.get("games")
    if type(entries) is not list:
        raise ValueError("outcome trajectory snapshot validation games are invalid")
    start = metadata.validation_game_start
    stop = start + metadata.validation_games
    if len(entries) != stop:
        raise ValueError(
            "outcome checkpoint snapshot does not end at its validation sweep"
        )
    games = tuple(
        load_trajectory_game(
            archive_root / "validation" / f"game-{index:06d}",
            metadata.identity,
        )
        for index in range(start, stop)
    )
    for game in games:
        if (
            game.actor.kind != "model"
            or game.actor.name != OUTCOME_ALGORITHM
            or Path(game.actor.source).resolve() != archive_root.parent.resolve()
        ):
            raise ValueError(
                "outcome checkpoint validation actor provenance is invalid"
            )
        if game.actor.artifact_sha256 != metadata.model_state_sha256:
            raise ValueError(
                "outcome checkpoint actor does not match its validation games"
            )
        if (
            game.opponent.artifact_sha256
            != metadata.validation_opponent_artifact_sha256
        ):
            raise ValueError(
                "outcome checkpoint opponent does not match its validation games"
            )
        if any(record.behavior_mode != "greedy" for record in game.records):
            raise ValueError(
                "outcome checkpoint validation games must use deterministic behavior"
            )
    for offset in range(0, len(games), 2):
        seat_zero, seat_one = games[offset:offset + 2]
        if (seat_zero.learner_seat, seat_one.learner_seat) != (0, 1):
            raise ValueError(
                "outcome checkpoint validation games are not reciprocal by seat"
            )
        if (
            seat_zero.episode_seed != seat_one.episode_seed
            or seat_zero.profile_id != seat_one.profile_id
        ):
            raise ValueError(
                "outcome checkpoint validation pair schedule does not match"
            )
    return games


def _validate_checkpoint_evidence(
    snapshot: Mapping[str, object],
    archive_root: Path,
    loaded: LoadedOutcomePolicy,
) -> tuple[TacticalV3TrajectoryGame, ...]:
    games = _checkpoint_validation_games(snapshot, archive_root, loaded.metadata)
    count = len(games)
    win_rate = sum(game.outcome == "win" for game in games) / count
    mean_return = sum(game.terminal_reward.total for game in games) / count
    mean_decisions = sum(len(game.records) for game in games) / count
    for label, actual, expected in (
        (
            "validation_win_rate",
            loaded.metadata.validation_win_rate,
            win_rate,
        ),
        (
            "validation_mean_return",
            loaded.metadata.validation_mean_return,
            mean_return,
        ),
        (
            "validation_mean_decisions",
            loaded.metadata.validation_mean_decisions,
            mean_decisions,
        ),
    ):
        if actual != expected:
            raise ValueError(
                f"outcome checkpoint {label} does not match its validation games"
            )
    fixture_decisions = tuple(
        record.decision for game in games for record in game.records
    )[:2]
    if not fixture_decisions or loaded.fixture.decisions != fixture_decisions:
        raise ValueError(
            "outcome checkpoint inference fixture does not match its validation games"
        )
    return games


def _validate_run_initialization(
    manifest: Mapping[str, object],
    config: Mapping[str, object],
    metadata: OutcomeCheckpointMetadata,
) -> None:
    """Bind the run's declared initialization to its published checkpoint."""

    run_initialization = manifest.get("initialization")
    if (
        not isinstance(run_initialization, Mapping)
        or dict(run_initialization) != dict(metadata.initialization)
    ):
        raise ValueError(
            "outcome run initialization does not match checkpoint"
        )
    kind = metadata.initialization.get("kind")
    if kind == "scratch":
        expected_source = None
    elif kind == "structured-policy-run":
        expected_source = metadata.initialization.get("run")
        if type(expected_source) is not str or not expected_source:
            raise ValueError(
                "outcome checkpoint source initialization is invalid"
            )
    else:
        raise ValueError("outcome checkpoint initialization kind is invalid")
    if (
        "initialization_source" not in config
        or config["initialization_source"] != expected_source
    ):
        raise ValueError(
            "outcome run initialization source does not match checkpoint"
        )


def validate_outcome_run(run_dir: Path) -> LoadedOutcomePolicy:
    """Validate the inference-bearing portion of a live outcome-training run."""

    root = Path(run_dir).resolve()
    manifest = read_json(root / "run.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("outcome run manifest must be an object")
    if manifest.get("schema_version") != 1:
        raise ValueError("outcome run schema_version must be 1")
    if manifest.get("state") not in _RUN_STATES:
        raise ValueError("outcome run state is invalid")
    if manifest.get("evidence_status") != "unsealed-experimental":
        raise ValueError("outcome run evidence_status is invalid")
    config = manifest.get("config")
    if not isinstance(config, Mapping) or (
        config.get("algorithm") != OUTCOME_ALGORITHM
        or config.get("backend") != OUTCOME_ALGORITHM
    ):
        raise ValueError("outcome run must declare structured_policy_gradient")
    if manifest.get("policy_identity") != "policy-identity.json":
        raise ValueError("outcome run must declare policy-identity.json")
    identity = parse_spaces(read_json(root / "policy-identity.json"))
    expected_contract = {
        "environment": "tactical-v3",
        "version": identity.contract_version,
        "environment_kind": identity.environment_kind,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }
    if manifest.get("contract") != expected_contract:
        raise ValueError("outcome run contract does not match policy identity")
    checkpoint_relative = manifest.get("latest_checkpoint")
    if type(checkpoint_relative) is not str or not checkpoint_relative:
        raise ValueError("outcome run has no validated checkpoint yet")
    checkpoint = (root / checkpoint_relative).resolve()
    if checkpoint.parent != (root / "checkpoints").resolve() or checkpoint.suffix != ".pt":
        raise ValueError("outcome run checkpoint must be checkpoints/*.pt")
    snapshot_relative = manifest.get("best_trajectory_manifest")
    if type(snapshot_relative) is not str or not snapshot_relative:
        raise ValueError("outcome run is missing best trajectory manifest")
    snapshot = (root / snapshot_relative).resolve()
    if snapshot.parent != (root / "checkpoints").resolve() or snapshot.suffix != ".json":
        raise ValueError("outcome best trajectory manifest must be checkpoints/*.json")
    snapshot_bytes = snapshot.read_bytes()
    loaded = load_outcome_checkpoint(
        checkpoint, identity.encoding_hash, identity.capacity_hash,
    )
    _validate_run_initialization(manifest, config, loaded.metadata)
    expected_checkpoint_name = f"policy-update-{loaded.metadata.update:06d}.pt"
    expected_snapshot_name = (
        f"trajectory-manifest-update-{loaded.metadata.update:06d}.json"
    )
    if checkpoint.name != expected_checkpoint_name:
        raise ValueError("outcome checkpoint filename does not match its update")
    if snapshot.name != expected_snapshot_name:
        raise ValueError(
            "outcome trajectory snapshot filename does not match its update"
        )
    if hashlib.sha256(snapshot_bytes).hexdigest() != (
        loaded.metadata.trajectory_manifest_sha256
    ):
        raise ValueError("outcome best trajectory manifest hash does not match checkpoint")
    snapshot_value = validate_trajectory_manifest_snapshot(
        snapshot, root / "trajectories", identity,
    )
    validation_games = _validate_checkpoint_evidence(
        snapshot_value, root / "trajectories", loaded,
    )
    if loaded.metadata.identity != identity:
        raise ValueError("outcome checkpoint identity does not match policy identity")
    if manifest.get("latest_checkpoint_step") != loaded.metadata.update:
        raise ValueError("outcome run checkpoint step does not match checkpoint")
    if manifest.get("best_update") != loaded.metadata.update:
        raise ValueError("outcome run best update does not match checkpoint")
    for field, expected in (
        ("best_validation_win_rate", loaded.metadata.validation_win_rate),
        ("best_validation_mean_return", loaded.metadata.validation_mean_return),
        (
            "best_validation_mean_decisions",
            loaded.metadata.validation_mean_decisions,
        ),
    ):
        if manifest.get(field) != expected:
            raise ValueError(f"outcome run {field} does not match checkpoint")
    scenario = manifest.get("scenario")
    if (
        not isinstance(scenario, Mapping)
        or scenario.get("path") != "scenario.json"
        or scenario.get("schema_version") != identity.scenario_schema_version
    ):
        raise ValueError("outcome run scenario reference is invalid")
    resolved_scenario = resolve_scenario(
        environment="tactical-v3",
        scenario_file=root / "scenario.json",
        template_id=None,
    )
    # Imported lazily to keep the inference checkpoint module out of the
    # controller/continuation import cycle.
    from .tactical_v3_continuation import (
        _start_distribution,
        _validate_target_scenario_identity,
    )

    _validate_target_scenario_identity(
        resolved_scenario,
        identity,
        _start_distribution(resolved_scenario.document),
    )
    if config.get("validation_games") != loaded.metadata.validation_games:
        raise ValueError(
            "outcome run validation game count does not match checkpoint"
        )
    validation_opponent = manifest.get("validation_opponent_snapshot")
    if (
        not isinstance(validation_opponent, Mapping)
        or dict(validation_opponent) != asdict(validation_games[0].opponent)
    ):
        raise ValueError(
            "outcome run validation opponent does not match checkpoint"
        )
    return loaded
