"""Strict CPU checkpoint and unsealed run publication for tactical-v3 policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tempfile
from typing import Literal, Mapping

import torch
from torch import Tensor

from .tactical_v3_batching import collate_examples
from .tactical_v3_corpus import (
    StructuredCorpus,
    StructuredExample,
    StructuredTarget,
    TeacherEvidence,
    _file_leases,
    _publish_no_replace,
    _read_partition,
    _root_lease,
    _validate_manifest,
    _validate_partitions,
)
from .tactical_v3_layers import TacticalV3ModelConfig
from .tactical_v3_model import CandidateIdentity, TacticalV3Policy
from .tactical_v3_objectives import ObjectiveConfig
from .tactical_v3_schema import TacticalV3SemanticIdentity, canonical_sha256, parse_decision, parse_spaces
from .tactical_v3_training import (
    METRIC_KEYS,
    EpochMetrics,
    TrainerConfig,
    TrainingResult,
)


_FORMAT_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset({"format_version", "metadata", "state_dict", "inference_fixture"})
_METADATA_FIELDS = frozenset({
    "format_version", "algorithm", "identity", "model_config", "objective_config",
    "trainer_config", "corpus_sha256", "model_state_sha256", "best_epoch",
    "best_validation_policy_nll", "published_device",
})
_FIXTURE_FIELDS = frozenset({"examples", "valid_candidate_logits", "selected_identities"})
_RUN_INVENTORY = frozenset({
    "run.json", "scenario.json", "corpus-manifest.json", "metrics.jsonl",
    "inference-fixture.json", "policy-identity.json", "checkpoints",
})
_RUN_FIELDS = frozenset({
    "schema_version", "state", "evidence_status", "config", "contract",
    "latest_checkpoint", "latest_checkpoint_step", "dataset_manifest_sha256",
    "best_epoch", "best_validation_policy_nll", "policy_identity",
})
_METRICS_FIELDS = frozenset({
    "epoch", "train", "validation", "validation_policy_nll", "improved",
})
_ADOPTED_EVIDENCE_KIND = "tactical-v3-adopted-dagger-evidence"
_ADOPTED_EVIDENCE_FIELDS = frozenset({
    "schema_version", "kind", "dataset_manifest_sha256", "source",
    "source_scenario_json", "published_scenario_sha256", "provenance_scope",
    "collection", "training",
})
_ADOPTED_SOURCE_FIELDS = frozenset({
    "checkpoint_sha256", "collection_sha256", "training_sha256",
    "metrics_sha256", "scenario_sha256",
})


@dataclass(frozen=True, slots=True)
class StructuredCheckpointMetadata:
    format_version: int
    algorithm: Literal["structured_imitation"]
    identity: TacticalV3SemanticIdentity
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig
    corpus_sha256: str
    model_state_sha256: str
    best_epoch: int
    best_validation_policy_nll: float
    published_device: Literal["cpu"]


@dataclass(frozen=True, slots=True)
class StructuredInferenceFixture:
    examples: tuple[StructuredExample, ...]
    valid_candidate_logits: tuple[tuple[float, ...], ...]
    selected_identities: tuple[CandidateIdentity, ...]


@dataclass(frozen=True, slots=True)
class LoadedStructuredPolicy:
    model: TacticalV3Policy
    metadata: StructuredCheckpointMetadata
    fixture: StructuredInferenceFixture


def _plain(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("checkpoint plain values must be finite")
        return value
    if type(value) is dict:
        if not all(type(key) is str for key in value):
            raise TypeError("checkpoint mapping keys must be built-in str")
        return {key: _plain(item) for key, item in value.items()}
    if type(value) is list:
        return [_plain(item) for item in value]
    raise TypeError("checkpoint contains a non-whitelisted plain value")


def _wire_plain(value: object) -> object:
    if value is None or type(value) in (str, int, float, bool):
        if type(value) is float and not math.isfinite(value):
            raise ValueError("checkpoint plain values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("checkpoint mapping keys must be built-in str")
        return {key: _wire_plain(item) for key, item in value.items()}
    if type(value) in (tuple, list):
        return [_wire_plain(item) for item in value]
    raise TypeError("checkpoint contains a non-whitelisted plain value")


def _plain_mapping(value: object, expected: frozenset[str], field: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != expected or not all(type(key) is str for key in value):
        raise ValueError(f"{field} fields must be exactly {sorted(expected)}")
    result = {key: _plain(item) for key, item in value.items()}
    if not isinstance(result, dict):  # pragma: no cover - narrows static type only.
        raise AssertionError
    return result


def _int(value: object, field: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a built-in int")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _float(value: object, field: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{field} must be a finite built-in float")
    return value


def _sha256(value: object, field: str) -> str:
    if type(value) is not str or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a lowercase SHA-256 hash")
    return value


def _canonical_json(value: object) -> bytes:
    return (json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _identity_wire(identity: TacticalV3SemanticIdentity) -> dict[str, object]:
    if type(identity) is not TacticalV3SemanticIdentity:
        raise TypeError("metadata.identity must be TacticalV3SemanticIdentity")
    return _wire_plain({field.name: getattr(identity, field.name) for field in fields(identity)})  # type: ignore[return-value]


def _identity_manifest(identity: TacticalV3SemanticIdentity) -> dict[str, object]:
    return {
        "environment": "tactical-v3", "version": "tactical-v3",
        "environment_kind": identity.environment_kind,
        "contract_hash": identity.contract_hash, "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
    }


def _dataclass_wire(value: object) -> dict[str, object]:
    return _wire_plain(asdict(value))  # type: ignore[return-value]


def _state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name]
        if type(name) is not str or not isinstance(value, Tensor):
            raise TypeError("model state must map names to tensors")
        cpu = value.detach().to(device="cpu").contiguous()
        header = json.dumps(
            {"name": name, "dtype": str(cpu.dtype), "shape": list(cpu.shape)},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        digest.update(len(header).to_bytes(8, "big")); digest.update(header)
        digest.update(cpu.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _state_wire(model: TacticalV3Policy) -> dict[str, Tensor]:
    if type(model) is not TacticalV3Policy:
        raise TypeError("model must be TacticalV3Policy")
    return {
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    }


def structured_model_state_sha256(model: TacticalV3Policy) -> str:
    if type(model) is not TacticalV3Policy:
        raise TypeError("model must be TacticalV3Policy")
    return _state_sha256(_state_wire(model))


def _example_wire(example: StructuredExample) -> dict[str, object]:
    if type(example) is not StructuredExample:
        raise TypeError("fixture examples must be StructuredExample")
    return _wire_plain(asdict(example))  # type: ignore[return-value]


def _example_from_wire(value: object, identity: TacticalV3SemanticIdentity) -> StructuredExample:
    fields_expected = frozenset({
        "example_schema_version", "decision", "target", "teacher", "scenario_id",
        "contract_hash", "encoding_hash", "capacity_hash", "profile_id", "episode_seed",
        "learner_seat",
    })
    data = _plain_mapping(value, fields_expected, "fixture example")
    target_data = _plain_mapping(data["target"], frozenset({
        "teacher_candidate_id", "terminal_outcome", "trajectory_index",
        "remaining_turns_to_victory", "truncated",
    }), "fixture example.target")
    teacher_data = _plain_mapping(data["teacher"], frozenset({
        "identity", "search_depth", "expansion_budget", "actual_expansions",
        "heuristic_identity", "confidence",
    }), "fixture example.teacher")
    if data["example_schema_version"] != 1:
        raise ValueError("fixture example schema version")
    if target_data["terminal_outcome"] not in {"win", "loss", "draw"}:
        raise ValueError("fixture example target outcome")
    if type(target_data["truncated"]) is not bool:
        raise TypeError("fixture example target truncated")
    remaining = target_data["remaining_turns_to_victory"]
    if remaining is not None:
        remaining = _int(remaining, "fixture example.target.remaining_turns_to_victory", minimum=1)
    confidence = teacher_data["confidence"]
    if confidence is not None:
        confidence = _float(confidence, "fixture example.teacher.confidence")
    for name in ("scenario_id", "contract_hash", "encoding_hash", "capacity_hash", "profile_id"):
        if type(data[name]) is not str:
            raise TypeError(f"fixture example.{name}")
    example = StructuredExample(
        1, parse_decision(data["decision"], identity),
        StructuredTarget(
            _int(target_data["teacher_candidate_id"], "fixture example.target.teacher_candidate_id"),
            target_data["terminal_outcome"],
            _int(target_data["trajectory_index"], "fixture example.target.trajectory_index", minimum=0),
            remaining, target_data["truncated"],
        ),
        TeacherEvidence(
            _string(teacher_data["identity"], "fixture example.teacher.identity"),
            _int(teacher_data["search_depth"], "fixture example.teacher.search_depth", minimum=0),
            _int(teacher_data["expansion_budget"], "fixture example.teacher.expansion_budget", minimum=0),
            _int(teacher_data["actual_expansions"], "fixture example.teacher.actual_expansions", minimum=0),
            _string(teacher_data["heuristic_identity"], "fixture example.teacher.heuristic_identity"), confidence,
        ),
        data["scenario_id"], data["contract_hash"], data["encoding_hash"], data["capacity_hash"],
        data["profile_id"], _int(data["episode_seed"], "fixture example.episode_seed"),
        _int(data["learner_seat"], "fixture example.learner_seat"),
    )
    if (example.scenario_id, example.contract_hash, example.encoding_hash, example.capacity_hash) != (
        identity.scenario_id, identity.contract_hash, identity.encoding_hash, identity.capacity_hash,
    ):
        raise ValueError("fixture example identity does not match checkpoint identity")
    return example


def _string(value: object, field: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{field} must be a built-in str")
    return value


def _fixture_logits_and_actions(model: TacticalV3Policy, examples: tuple[StructuredExample, ...]) -> tuple[tuple[tuple[float, ...], ...], tuple[CandidateIdentity, ...]]:
    if not examples:
        raise ValueError("inference fixture must contain examples")
    model.eval()
    with torch.no_grad():
        batch = collate_examples(examples, model.config.horizon_turns)
        output = model(batch)
    logits: list[tuple[float, ...]] = []
    actions: list[CandidateIdentity] = []
    for index, example in enumerate(examples):
        valid = batch.candidates.mask[index]
        row = output.candidate_logits[index, valid].detach().cpu()
        if not bool(torch.isfinite(row).all()):
            raise ValueError("fixture has nonfinite valid candidate logits")
        values = tuple(float(value) for value in row.tolist())
        logits.append(values)
        offset = int(torch.argmax(row).item())
        candidate = example.decision.candidates[offset]
        actions.append(CandidateIdentity(candidate.decision_id, candidate.candidate_id))
    return tuple(logits), tuple(actions)


def _fixture_wire(model: TacticalV3Policy, examples: tuple[StructuredExample, ...]) -> dict[str, object]:
    logits, actions = _fixture_logits_and_actions(model, examples)
    return {
        "examples": [_example_wire(example) for example in examples],
        "valid_candidate_logits": [list(row) for row in logits],
        "selected_identities": [asdict(item) for item in actions],
    }


def _fixture_from_wire(value: object, identity: TacticalV3SemanticIdentity) -> StructuredInferenceFixture:
    data = _plain_mapping(value, _FIXTURE_FIELDS, "inference_fixture")
    if type(data["examples"]) is not list or not data["examples"]:
        raise ValueError("inference_fixture.examples must be a non-empty list")
    if type(data["valid_candidate_logits"]) is not list or type(data["selected_identities"]) is not list:
        raise TypeError("inference_fixture rows must be lists")
    examples = tuple(_example_from_wire(item, identity) for item in data["examples"])
    if len(data["valid_candidate_logits"]) != len(examples) or len(data["selected_identities"]) != len(examples):
        raise ValueError("inference_fixture row counts must agree")
    logits: list[tuple[float, ...]] = []
    identities: list[CandidateIdentity] = []
    for index, (row, selected) in enumerate(zip(data["valid_candidate_logits"], data["selected_identities"], strict=True)):
        if type(row) is not list or not row:
            raise ValueError(f"inference_fixture.valid_candidate_logits[{index}] must be non-empty")
        logits.append(tuple(_float(item, f"inference_fixture.valid_candidate_logits[{index}][]") for item in row))
        selected_data = _plain_mapping(selected, frozenset({"decision_id", "candidate_id"}), f"inference_fixture.selected_identities[{index}]")
        identities.append(CandidateIdentity(
            _int(selected_data["decision_id"], "inference fixture decision_id"),
            _int(selected_data["candidate_id"], "inference fixture candidate_id"),
        ))
    return StructuredInferenceFixture(examples, tuple(logits), tuple(identities))


def _metadata_wire(metadata: StructuredCheckpointMetadata) -> dict[str, object]:
    if type(metadata) is not StructuredCheckpointMetadata:
        raise TypeError("metadata must be StructuredCheckpointMetadata")
    if metadata.format_version != _FORMAT_VERSION or metadata.algorithm != "structured_imitation":
        raise ValueError("unsupported checkpoint metadata format")
    if metadata.published_device != "cpu":
        raise ValueError("published_device must be cpu")
    _sha256(metadata.corpus_sha256, "metadata.corpus_sha256")
    _sha256(metadata.model_state_sha256, "metadata.model_state_sha256")
    _int(metadata.best_epoch, "metadata.best_epoch", minimum=0)
    _float(metadata.best_validation_policy_nll, "metadata.best_validation_policy_nll")
    return {
        "format_version": _FORMAT_VERSION, "algorithm": "structured_imitation",
        "identity": _identity_wire(metadata.identity),
        "model_config": _dataclass_wire(metadata.model_config),
        "objective_config": _dataclass_wire(metadata.objective_config),
        "trainer_config": _dataclass_wire(metadata.trainer_config),
        "corpus_sha256": metadata.corpus_sha256,
        "model_state_sha256": metadata.model_state_sha256,
        "best_epoch": metadata.best_epoch,
        "best_validation_policy_nll": metadata.best_validation_policy_nll,
        "published_device": "cpu",
    }


def _metadata_from_wire(value: object) -> StructuredCheckpointMetadata:
    data = _plain_mapping(value, _METADATA_FIELDS, "metadata")
    if _int(data["format_version"], "metadata.format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported metadata format_version")
    if data["algorithm"] != "structured_imitation":
        raise ValueError("unsupported metadata.algorithm")
    if data["published_device"] != "cpu":
        raise ValueError("metadata.published_device must be cpu")
    identity = parse_spaces(data["identity"])
    model_data = _plain_mapping(data["model_config"], frozenset(field.name for field in fields(TacticalV3ModelConfig)), "metadata.model_config")
    model_config = TacticalV3ModelConfig(**{  # type: ignore[arg-type]
        **{name: _int(model_data[name], f"metadata.model_config.{name}") for name in model_data if name != "horizon_turns"},
        "horizon_turns": tuple(_int(item, "metadata.model_config.horizon_turns") for item in _list(model_data["horizon_turns"], "metadata.model_config.horizon_turns")),
    })
    objective_data = _plain_mapping(data["objective_config"], frozenset(field.name for field in fields(ObjectiveConfig)), "metadata.objective_config")
    objective = ObjectiveConfig(**{name: _float(item, f"metadata.objective_config.{name}") for name, item in objective_data.items()})
    trainer_data = _plain_mapping(data["trainer_config"], frozenset(field.name for field in fields(TrainerConfig)), "metadata.trainer_config")
    trainer = TrainerConfig(
        seed=_int(trainer_data["seed"], "metadata.trainer_config.seed", minimum=0),
        batch_size=_int(trainer_data["batch_size"], "metadata.trainer_config.batch_size", minimum=1),
        learning_rate=_float(trainer_data["learning_rate"], "metadata.trainer_config.learning_rate"),
        max_epochs=_int(trainer_data["max_epochs"], "metadata.trainer_config.max_epochs", minimum=1),
        patience_epochs=_int(trainer_data["patience_epochs"], "metadata.trainer_config.patience_epochs", minimum=1),
        gradient_clip_norm=_float(trainer_data["gradient_clip_norm"], "metadata.trainer_config.gradient_clip_norm"),
        device=_string(trainer_data["device"], "metadata.trainer_config.device"),
    )
    return StructuredCheckpointMetadata(
        _FORMAT_VERSION, "structured_imitation", identity, model_config, objective, trainer,
        _sha256(data["corpus_sha256"], "metadata.corpus_sha256"),
        _sha256(data["model_state_sha256"], "metadata.model_state_sha256"),
        _int(data["best_epoch"], "metadata.best_epoch", minimum=0),
        _float(data["best_validation_policy_nll"], "metadata.best_validation_policy_nll"), "cpu",
    )


def _list(value: object, field: str) -> list[object]:
    if type(value) is not list:
        raise TypeError(f"{field} must be a list")
    return value


def _write_checkpoint(path: Path, payload: Mapping[str, object]) -> Path:
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite checkpoint {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(dict(payload), handle)
            handle.flush(); os.fsync(handle.fileno())
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite checkpoint {path}")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return path


def _cpu_model(config: TacticalV3ModelConfig, state: Mapping[str, Tensor]) -> TacticalV3Policy:
    copy = TacticalV3Policy(config).to(device="cpu")
    copy.load_state_dict(state, strict=True)
    copy.eval()
    return copy


def save_structured_checkpoint(path: Path, model: TacticalV3Policy, metadata: StructuredCheckpointMetadata, fixture_examples: tuple[StructuredExample, ...]) -> Path:
    state = _state_wire(model)
    state_hash = _state_sha256(state)
    if metadata.model_config != model.config:
        raise ValueError("metadata model config does not match model config")
    if metadata.model_state_sha256 != state_hash:
        raise ValueError("metadata model state SHA-256 does not match model state")
    cpu_model = _cpu_model(model.config, state)
    payload = {
        "format_version": _FORMAT_VERSION,
        "metadata": _metadata_wire(metadata),
        "state_dict": state,
        "inference_fixture": _fixture_wire(cpu_model, tuple(fixture_examples)),
    }
    return _write_checkpoint(Path(path), payload)


def _validate_state(value: object) -> dict[str, Tensor]:
    if not isinstance(value, Mapping) or not value or not all(type(name) is str for name in value):
        raise TypeError("state_dict must be a non-empty string-to-tensor mapping")
    result: dict[str, Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu" or not tensor.is_contiguous():
            raise TypeError("state_dict tensors must be contiguous CPU tensors")
        result[name] = tensor
    return result


def load_structured_checkpoint(path: Path, expected_encoding_hash: str, expected_capacity_hash: str) -> LoadedStructuredPolicy:
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(raw, Mapping) or set(raw) != _TOP_LEVEL_FIELDS or not all(type(key) is str for key in raw):
        raise ValueError("checkpoint fields must be exactly format_version, metadata, state_dict, inference_fixture")
    if _int(raw["format_version"], "format_version") != _FORMAT_VERSION:
        raise ValueError("unsupported checkpoint format_version")
    metadata = _metadata_from_wire(raw["metadata"])
    if metadata.identity.encoding_hash != _sha256(expected_encoding_hash, "expected encoding hash"):
        raise ValueError("checkpoint encoding hash does not match expected encoding hash")
    if metadata.identity.capacity_hash != _sha256(expected_capacity_hash, "expected capacity hash"):
        raise ValueError("checkpoint capacity hash does not match expected capacity hash")
    state = _validate_state(raw["state_dict"])
    if _state_sha256(state) != metadata.model_state_sha256:
        raise ValueError("model state SHA-256 does not match metadata")
    model = TacticalV3Policy(metadata.model_config).to(device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    fixture = _fixture_from_wire(raw["inference_fixture"], metadata.identity)
    actual_logits, actual_actions = _fixture_logits_and_actions(model, fixture.examples)
    if actual_logits != fixture.valid_candidate_logits or actual_actions != fixture.selected_identities:
        raise ValueError("checkpoint inference fixture does not replay exactly")
    return LoadedStructuredPolicy(model, metadata, fixture)


def _write_bytes(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data); handle.flush(); os.fsync(handle.fileno())


def _after_checkpoint_written() -> None:
    """Test seam for confirming publication cleanup after checkpoint generation."""


def _metrics_jsonl(history: tuple) -> bytes:
    if type(history) is not tuple or not history:
        raise ValueError("training history must be a non-empty immutable tuple")
    rows: list[bytes] = []
    for item in history:
        if type(item) is not EpochMetrics:
            raise TypeError("training history must contain EpochMetrics")
        rows.append(_canonical_json({
            "epoch": item.epoch, "train": dict(item.train), "validation": dict(item.validation),
            "validation_policy_nll": item.validation_policy_nll, "improved": item.improved,
        }))
    return b"".join(rows)


def _reject_duplicate_json_keys(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"JSON contains duplicate key {key!r}")
        result[key] = value
    return result


def _decode_json_bytes(data: bytes, label: str) -> object:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error


def _read_json_file(path: Path, label: str) -> object:
    try:
        return _decode_json_bytes(path.read_bytes(), label)
    except OSError as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error


def _read_canonical_json_file(path: Path, label: str) -> object:
    try:
        data = path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} must be valid UTF-8 JSON") from error
    value = _decode_json_bytes(data, label)
    if data != _canonical_json(value):
        raise ValueError(f"{label} must be canonical compact JSON")
    return value


def _bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
            byte_count += len(chunk)
    return digest.hexdigest(), byte_count


def _stat_token(path: Path) -> tuple[int, int, int, int]:
    value = path.stat(follow_symlinks=False)
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _require_destination_outside_source_directories(
    destination: Path,
    sources: tuple[Path, ...],
) -> None:
    resolved_destination = destination.resolve(strict=False)
    for source in sources:
        source_directory = source.parent.resolve(strict=True)
        if (
            resolved_destination == source_directory
            or source_directory in resolved_destination.parents
        ):
            raise ValueError(
                "run destination must not be inside a source evidence directory"
            )


def _require_plain_source_file(path: Path, label: str) -> Path:
    path = _public_run_path(Path(path), label)
    _require_plain_lexical_chain(path, label)
    if _is_reparse(path) or not path.is_file():
        raise ValueError(f"{label} must be a plain file")
    return path


def _validate_adopted_scenario(
    value: object,
    identity: TacticalV3SemanticIdentity,
) -> None:
    scenario = _plain_mapping(
        value,
        frozenset({
            "schema_version", "id", "name", "environment", "board", "rules",
            "episode", "reward", "tactical_v3",
        }),
        "adopted tactical-v3 scenario",
    )
    if (
        _int(scenario["schema_version"], "scenario schema_version")
        != identity.scenario_schema_version
        or _string(scenario["id"], "scenario id") != identity.scenario_id
        or _string(scenario["environment"], "scenario environment") != "tactical-v3"
    ):
        raise ValueError("adopted scenario identity does not match checkpoint")
    if not _string(scenario["name"], "scenario name"):
        raise ValueError("scenario name must not be empty")
    for name in ("board", "rules", "episode", "reward", "tactical_v3"):
        if type(scenario[name]) is not dict or not scenario[name]:
            raise TypeError(f"scenario {name} must be a non-empty object")


def _load_self_describing_checkpoint(path: Path) -> LoadedStructuredPolicy:
    raw = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _TOP_LEVEL_FIELDS
        or not all(type(key) is str for key in raw)
    ):
        raise ValueError(
            "checkpoint fields must be exactly format_version, metadata, state_dict, inference_fixture"
        )
    metadata = _metadata_from_wire(raw["metadata"])
    return load_structured_checkpoint(
        path,
        metadata.identity.encoding_hash,
        metadata.identity.capacity_hash,
    )


def _metrics_from_jsonl(
    data: bytes,
    best_epoch: int,
    best_validation_policy_nll: float,
) -> tuple[EpochMetrics, ...]:
    if not data or not data.endswith(b"\n"):
        raise ValueError("metrics.jsonl must be non-empty newline-delimited JSON")
    history: list[EpochMetrics] = []
    running_best = math.inf
    actual_best_epoch = -1
    actual_best_nll = math.inf
    for index, line in enumerate(data.splitlines(keepends=True)):
        if line == b"\n" or not line.endswith(b"\n"):
            raise ValueError(f"metrics.jsonl row {index} is not canonical JSONL")
        value = _decode_json_bytes(line, f"metrics.jsonl row {index}")
        row = _plain_mapping(
            value,
            _METRICS_FIELDS,
            f"metrics.jsonl row {index}",
        )
        epoch = _int(row["epoch"], f"metrics.jsonl row {index}.epoch", minimum=0)
        if epoch != index:
            raise ValueError("metrics.jsonl epochs must be contiguous from zero")
        train_data = _plain_mapping(
            row["train"],
            frozenset(METRIC_KEYS),
            f"metrics.jsonl row {index}.train",
        )
        validation_data = _plain_mapping(
            row["validation"],
            frozenset(METRIC_KEYS),
            f"metrics.jsonl row {index}.validation",
        )
        train = {
            name: _float(train_data[name], f"metrics.jsonl row {index}.train.{name}")
            for name in METRIC_KEYS
        }
        validation = {
            name: _float(
                validation_data[name],
                f"metrics.jsonl row {index}.validation.{name}",
            )
            for name in METRIC_KEYS
        }
        validation_nll = _float(
            row["validation_policy_nll"],
            f"metrics.jsonl row {index}.validation_policy_nll",
        )
        if validation_nll != validation["policy"]:
            raise ValueError(
                f"metrics.jsonl row {index} validation policy NLL is inconsistent"
            )
        if type(row["improved"]) is not bool:
            raise TypeError(
                f"metrics.jsonl row {index}.improved must be a built-in bool"
            )
        expected_improved = validation_nll < running_best - 1e-12
        if row["improved"] is not expected_improved:
            raise ValueError(
                f"metrics.jsonl row {index} improved flag is inconsistent"
            )
        if expected_improved:
            running_best = validation_nll
            actual_best_epoch = epoch
            actual_best_nll = validation_nll
        if line != _canonical_json(value):
            raise ValueError(f"metrics.jsonl row {index} must be canonical compact JSON")
        history.append(EpochMetrics(
            epoch=epoch,
            train=train,
            validation=validation,
            validation_policy_nll=validation_nll,
            improved=row["improved"],
        ))
    if actual_best_epoch != best_epoch or actual_best_nll != best_validation_policy_nll:
        raise ValueError("metrics.jsonl best epoch/NLL is inconsistent")
    return tuple(history)


def _validate_relative(value: object, field: str) -> str:
    path = _string(value, field)
    parsed = PurePosixPath(path)
    if parsed.is_absolute() or "\\" in path or any(part in {"", ".", ".."} for part in parsed.parts):
        raise ValueError(f"{field} must be a safe relative POSIX path")
    return path


def _is_reparse(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _public_run_path(value: Path, label: str) -> Path:
    raw = os.fspath(value)
    components = (
        raw.replace(os.altsep, os.sep).split(os.sep)
        if os.altsep
        else raw.split(os.sep)
    )
    if any(component in {".", ".."} for component in components):
        raise ValueError(f"{label} must not contain a dot path component or traversal")
    return Path(raw)


def _require_plain_lexical_chain(path: Path, label: str) -> None:
    lexical = path if path.is_absolute() else Path.cwd() / path
    for component in reversed((lexical, *lexical.parents)):
        try:
            os.lstat(component)
        except FileNotFoundError:
            break
        if _is_reparse(component):
            raise ValueError(
                f"{label} ancestor must not be a symlink or reparse point: {component}"
            )


def _require_plain_run_inventory(root: Path) -> dict[str, Path]:
    if _is_reparse(root) or not root.is_dir():
        raise ValueError(
            "run directory must be a plain directory, not a symlink or reparse point"
        )
    entries = {item.name: item for item in root.iterdir()}
    if set(entries) != _RUN_INVENTORY:
        raise ValueError("run inventory is invalid")
    for name in _RUN_INVENTORY - {"checkpoints"}:
        entry = entries[name]
        if _is_reparse(entry) or not entry.is_file():
            raise ValueError(f"run inventory entry {name!r} must be a plain file")
    checkpoint_dir = entries["checkpoints"]
    if _is_reparse(checkpoint_dir) or not checkpoint_dir.is_dir():
        raise ValueError("run checkpoints inventory is invalid")
    checkpoint_entries = {item.name: item for item in checkpoint_dir.iterdir()}
    if set(checkpoint_entries) != {"best.pt"}:
        raise ValueError("run checkpoints inventory is invalid")
    checkpoint = checkpoint_entries["best.pt"]
    if _is_reparse(checkpoint) or not checkpoint.is_file():
        raise ValueError("checkpoint must be a plain file")
    return entries


def _authenticated_corpus_manifest(
    corpus: StructuredCorpus,
    expected: TacticalV3SemanticIdentity,
) -> bytes:
    root = Path(corpus.root)
    with _root_lease(root):
        with _file_leases(root) as evidence:
            identity, files = _validate_manifest(
                root,
                expected,
                evidence["manifest.json"],
            )
            train = _read_partition(
                root,
                files[0],
                expected,
                evidence["train.jsonl"],
            )
            validation = _read_partition(
                root,
                files[1],
                expected,
                evidence["validation.jsonl"],
            )
            _validate_partitions(train, validation)
            if identity != corpus.identity:
                raise ValueError("authenticated manifest identity does not match corpus")
            if train != corpus.train or validation != corpus.validation:
                raise ValueError("authenticated corpus bytes do not match loaded corpus")
            return evidence["manifest.json"]


def _validate_adopted_evidence(
    value: object,
    *,
    root: Path,
    loaded: LoadedStructuredPolicy,
    scenario_value: object,
    metrics_bytes: bytes,
) -> str:
    evidence = _plain_mapping(
        value,
        _ADOPTED_EVIDENCE_FIELDS,
        "adopted DAgger evidence",
    )
    if _int(evidence["schema_version"], "adopted evidence schema_version") != 1:
        raise ValueError("adopted evidence schema_version is invalid")
    if _string(evidence["kind"], "adopted evidence kind") != _ADOPTED_EVIDENCE_KIND:
        raise ValueError("adopted evidence kind is invalid")
    if (
        _string(evidence["provenance_scope"], "adopted evidence provenance_scope")
        != "adopted-local-artifact"
    ):
        raise ValueError("adopted evidence provenance scope is invalid")
    dataset_sha256 = _sha256(
        evidence["dataset_manifest_sha256"],
        "adopted evidence dataset_manifest_sha256",
    )
    if dataset_sha256 != loaded.metadata.corpus_sha256:
        raise ValueError("adopted dataset identity does not match checkpoint")
    source = _plain_mapping(
        evidence["source"],
        _ADOPTED_SOURCE_FIELDS,
        "adopted evidence source",
    )
    for name in source:
        _sha256(source[name], f"adopted evidence source.{name}")
    checkpoint_sha256, _ = _file_sha256(root / "checkpoints" / "best.pt")
    if source["checkpoint_sha256"] != checkpoint_sha256:
        raise ValueError("adopted source checkpoint hash is inconsistent")
    collection_bytes = _canonical_json(evidence["collection"])
    collection_sha256 = _bytes_sha256(collection_bytes)
    if source["collection_sha256"] != collection_sha256:
        raise ValueError("adopted source collection hash is inconsistent")
    training_bytes = _canonical_json(evidence["training"])
    if source["training_sha256"] != _bytes_sha256(training_bytes):
        raise ValueError("adopted source training hash is inconsistent")
    if source["metrics_sha256"] != _bytes_sha256(metrics_bytes):
        raise ValueError("adopted source metrics hash is inconsistent")
    source_scenario_text = _string(
        evidence["source_scenario_json"],
        "adopted evidence source_scenario_json",
    )
    source_scenario_bytes = source_scenario_text.encode("utf-8")
    if source["scenario_sha256"] != _bytes_sha256(source_scenario_bytes):
        raise ValueError("adopted source scenario hash is inconsistent")
    source_scenario_value = _decode_json_bytes(
        source_scenario_bytes,
        "adopted source scenario",
    )
    if _canonical_json(source_scenario_value) != _canonical_json(scenario_value):
        raise ValueError("adopted source and published scenarios differ semantically")
    published_scenario_sha256 = _sha256(
        evidence["published_scenario_sha256"],
        "adopted evidence published_scenario_sha256",
    )
    if published_scenario_sha256 != _bytes_sha256(_canonical_json(scenario_value)):
        raise ValueError("adopted published scenario hash is inconsistent")
    _validate_adopted_scenario(scenario_value, loaded.metadata.identity)
    history = _metrics_from_jsonl(
        metrics_bytes,
        loaded.metadata.best_epoch,
        loaded.metadata.best_validation_policy_nll,
    )
    trainer = loaded.metadata.trainer_config
    if len(history) > trainer.max_epochs:
        raise ValueError("adopted metrics exceed trainer max_epochs")
    epochs_after_best = history[-1].epoch - loaded.metadata.best_epoch
    if epochs_after_best > trainer.patience_epochs:
        raise ValueError("adopted metrics exceed trainer early-stopping patience")
    if len(history) < trainer.max_epochs:
        if epochs_after_best != trainer.patience_epochs:
            raise ValueError("adopted metrics do not match trainer early stopping")
    return dataset_sha256


def _copy_checkpoint_bytes(
    source: Path,
    destination: Path,
) -> tuple[str, int, tuple[int, int, int, int]]:
    source = _require_plain_source_file(source, "source checkpoint")
    before = _stat_token(source)
    digest = hashlib.sha256()
    size = 0
    with source.open("rb") as reader, destination.open("xb") as writer:
        opened = os.fstat(reader.fileno())
        opened_token = (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        )
        if opened_token != before or not stat.S_ISREG(opened.st_mode):
            raise ValueError("source checkpoint changed before it could be copied")
        while chunk := reader.read(1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
            writer.write(chunk)
        writer.flush()
        os.fsync(writer.fileno())
    after = _stat_token(source)
    if before != after or size != before[2]:
        raise ValueError("source checkpoint changed while it was copied")
    copied_sha256, copied_size = _file_sha256(destination)
    if copied_size != size or copied_sha256 != digest.hexdigest():
        raise ValueError("staged checkpoint bytes do not match source")
    return copied_sha256, copied_size, after


def adopt_structured_run(
    run_dir: Path,
    *,
    source_checkpoint_path: Path,
    source_collection_path: Path,
    source_training_path: Path,
    source_metrics_path: Path,
    training_scenario_path: Path,
    expected_identity: TacticalV3SemanticIdentity,
    expected_checkpoint_sha256: str,
    expected_collection_sha256: str,
    expected_training_sha256: str,
    expected_metrics_sha256: str,
    expected_scenario_sha256: str,
) -> Path:
    """Adopt a directly observed DAgger artifact without rewriting its checkpoint.

    The resulting unsealed run proves that its seven published files remain mutually
    consistent. Collection and training JSON are retained as opaque, hash-pinned source
    observations; this deliberately does not claim a tracked, semantically revalidated,
    or independently reproducible historical training run.
    """

    run_dir = _public_run_path(run_dir, "run destination")
    _require_plain_lexical_chain(run_dir.parent, "run destination")
    if run_dir.exists() or _is_reparse(run_dir):
        raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
    parent = run_dir.parent
    if _is_reparse(parent) or not parent.is_dir():
        raise ValueError(
            "run parent must be a plain directory, not a symlink or reparse point"
        )

    source_collection = _require_plain_source_file(
        source_collection_path, "source collection manifest"
    )
    source_root = source_collection.parent
    source_training = _require_plain_source_file(
        source_training_path, "source training provenance"
    )
    source_metrics = _require_plain_source_file(
        source_metrics_path, "source training metrics"
    )
    source_scenario = _require_plain_source_file(
        training_scenario_path, "source training scenario"
    )
    source_checkpoint = _require_plain_source_file(
        source_checkpoint_path, "source checkpoint"
    )
    _require_destination_outside_source_directories(
        run_dir,
        (
            source_checkpoint,
            source_collection,
            source_training,
            source_metrics,
            source_scenario,
        ),
    )
    expected_hashes = {
        "checkpoint": _sha256(
            expected_checkpoint_sha256, "expected source checkpoint SHA-256"
        ),
        "collection": _sha256(
            expected_collection_sha256, "expected source collection SHA-256"
        ),
        "training": _sha256(
            expected_training_sha256, "expected source training SHA-256"
        ),
        "metrics": _sha256(
            expected_metrics_sha256, "expected source metrics SHA-256"
        ),
        "scenario": _sha256(
            expected_scenario_sha256, "expected source scenario SHA-256"
        ),
    }
    training_root = source_root / "training"
    if (
        source_collection.name != "collection.json"
        or source_training != training_root / "dagger-training.json"
        or source_metrics != training_root / "metrics.jsonl"
        or source_checkpoint != training_root / "checkpoints" / "best.pt"
    ):
        raise ValueError("source artifact does not use the expected DAgger layout")

    collection_bytes = source_collection.read_bytes()
    collection_value = _decode_json_bytes(collection_bytes, "source collection manifest")
    if collection_bytes != _canonical_json(collection_value):
        raise ValueError("source collection manifest must be canonical compact JSON")
    training_bytes = source_training.read_bytes()
    training_value = _decode_json_bytes(training_bytes, "source training provenance")
    if training_bytes != _canonical_json(training_value):
        raise ValueError("source training provenance must be canonical compact JSON")
    metrics_bytes = source_metrics.read_bytes()
    scenario_bytes = source_scenario.read_bytes()
    scenario_value = _decode_json_bytes(scenario_bytes, "source training scenario")
    scenario_published_bytes = _canonical_json(scenario_value)
    actual_hashes = {
        "collection": _bytes_sha256(collection_bytes),
        "training": _bytes_sha256(training_bytes),
        "metrics": _bytes_sha256(metrics_bytes),
        "scenario": _bytes_sha256(scenario_bytes),
    }
    for name, actual in actual_hashes.items():
        if actual != expected_hashes[name]:
            raise ValueError(f"source {name} SHA-256 does not match expected artifact")
    source_tokens = {
        path: _stat_token(path)
        for path in (
            source_collection,
            source_training,
            source_metrics,
            source_scenario,
        )
    }

    temporary = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.tmp-", dir=parent))
    try:
        checkpoints = temporary / "checkpoints"
        checkpoints.mkdir()
        checkpoint = checkpoints / "best.pt"
        checkpoint_sha256, _, checkpoint_token = _copy_checkpoint_bytes(
            source_checkpoint, checkpoint
        )
        if checkpoint_sha256 != expected_hashes["checkpoint"]:
            raise ValueError(
                "source checkpoint SHA-256 does not match expected artifact"
            )
        loaded = _load_self_describing_checkpoint(checkpoint)
        identity = loaded.metadata.identity
        if type(expected_identity) is not TacticalV3SemanticIdentity:
            raise TypeError("expected_identity must be TacticalV3SemanticIdentity")
        if identity != expected_identity:
            raise ValueError(
                "checkpoint identity does not match the authoritative scenario probe"
            )
        _validate_adopted_scenario(scenario_value, identity)
        collection_sha256 = _bytes_sha256(collection_bytes)
        evidence = {
            "schema_version": 1,
            "kind": _ADOPTED_EVIDENCE_KIND,
            "provenance_scope": "adopted-local-artifact",
            "dataset_manifest_sha256": loaded.metadata.corpus_sha256,
            "source": {
                "checkpoint_sha256": checkpoint_sha256,
                "collection_sha256": collection_sha256,
                "training_sha256": _bytes_sha256(training_bytes),
                "metrics_sha256": _bytes_sha256(metrics_bytes),
                "scenario_sha256": _bytes_sha256(scenario_bytes),
            },
            "source_scenario_json": scenario_bytes.decode("utf-8"),
            "published_scenario_sha256": _bytes_sha256(scenario_published_bytes),
            "collection": collection_value,
            "training": training_value,
        }
        run_manifest = {
            "schema_version": 2,
            "state": "completed",
            "evidence_status": "unsealed-experimental",
            "config": {"algorithm": "structured_imitation"},
            "contract": _identity_manifest(identity),
            "policy_identity": "policy-identity.json",
            "latest_checkpoint": "checkpoints/best.pt",
            "latest_checkpoint_step": loaded.metadata.best_epoch,
            "dataset_manifest_sha256": loaded.metadata.corpus_sha256,
            "best_epoch": loaded.metadata.best_epoch,
            "best_validation_policy_nll": loaded.metadata.best_validation_policy_nll,
        }
        _write_bytes(temporary / "run.json", _canonical_json(run_manifest))
        _write_bytes(temporary / "scenario.json", scenario_published_bytes)
        _write_bytes(
            temporary / "policy-identity.json",
            _canonical_json(_identity_wire(identity)),
        )
        _write_bytes(temporary / "corpus-manifest.json", _canonical_json(evidence))
        _write_bytes(temporary / "metrics.jsonl", metrics_bytes)
        _write_bytes(temporary / "inference-fixture.json", _canonical_json({
            "examples": [_example_wire(item) for item in loaded.fixture.examples],
            "valid_candidate_logits": [
                list(item) for item in loaded.fixture.valid_candidate_logits
            ],
            "selected_identities": [
                asdict(item) for item in loaded.fixture.selected_identities
            ],
        }))
        validate_structured_run(temporary)
        if _stat_token(source_checkpoint) != checkpoint_token:
            raise ValueError("source checkpoint changed during adoption")
        final_source_sha256, _ = _file_sha256(source_checkpoint)
        if final_source_sha256 != checkpoint_sha256:
            raise ValueError("source checkpoint changed during adoption")
        for path, token in source_tokens.items():
            if _stat_token(path) != token:
                raise ValueError(f"source evidence changed during adoption: {path}")
        _require_plain_lexical_chain(run_dir.parent, "run destination")
        if run_dir.exists() or _is_reparse(run_dir):
            raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
        _publish_no_replace(temporary, run_dir)
        return run_dir
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def publish_structured_run(
    run_dir: Path,
    result: TrainingResult,
    corpus: StructuredCorpus,
    *,
    training_scenario_path: Path,
    policy_identity: TacticalV3SemanticIdentity,
) -> Path:
    run_dir = _public_run_path(run_dir, "run destination")
    _require_plain_lexical_chain(run_dir.parent, "run destination")
    if run_dir.exists() or _is_reparse(run_dir):
        raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
    if type(result) is not TrainingResult or type(corpus) is not StructuredCorpus:
        raise TypeError("result and corpus must use their immutable tactical-v3 DTOs")
    if type(result.model_config) is not TacticalV3ModelConfig:
        raise TypeError("result model config must be TacticalV3ModelConfig")
    if type(result.objective_config) is not ObjectiveConfig:
        raise TypeError("result objective config must be ObjectiveConfig")
    if type(result.trainer_config) is not TrainerConfig:
        raise TypeError("result trainer config must be TrainerConfig")
    if result.model.config != result.model_config:
        raise ValueError("result model config does not match model config")
    if next(result.model.parameters()).device != torch.device(result.trainer_config.device):
        raise ValueError("result trainer device does not match model device")
    scenario_value = _read_json_file(
        Path(training_scenario_path), "training scenario"
    )
    identity = policy_identity
    if identity != parse_spaces(_identity_wire(identity)):
        raise ValueError("policy identity cannot be canonicalized")
    if any((example.encoding_hash, example.capacity_hash) != (identity.encoding_hash, identity.capacity_hash)
           for example in corpus.train + corpus.validation):
        raise ValueError("corpus examples do not match scenario identity")
    corpus_manifest_bytes = _authenticated_corpus_manifest(corpus, identity)
    metadata = StructuredCheckpointMetadata(
        _FORMAT_VERSION,
        "structured_imitation",
        identity,
        result.model_config,
        result.objective_config,
        result.trainer_config,
        corpus.identity,
        _state_sha256(_state_wire(result.model)),
        result.best_epoch,
        result.best_validation_policy_nll,
        "cpu",
    )
    parent = run_dir.parent
    if _is_reparse(parent) or not parent.is_dir():
        raise ValueError(
            "run parent must be a plain directory, not a symlink or reparse point"
        )
    temporary = Path(tempfile.mkdtemp(prefix=f".{run_dir.name}.tmp-", dir=parent))
    try:
        checkpoints = temporary / "checkpoints"; checkpoints.mkdir()
        checkpoint = save_structured_checkpoint(checkpoints / "best.pt", result.model, metadata, corpus.validation[:2])
        _after_checkpoint_written()
        fixture = load_structured_checkpoint(checkpoint, identity.encoding_hash, identity.capacity_hash).fixture
        run_manifest = {
            "schema_version": 2, "state": "completed", "evidence_status": "unsealed-experimental",
            "config": {"algorithm": "structured_imitation"}, "contract": _identity_manifest(identity),
            "policy_identity": "policy-identity.json",
            "latest_checkpoint": "checkpoints/best.pt", "latest_checkpoint_step": metadata.best_epoch,
            "dataset_manifest_sha256": metadata.corpus_sha256, "best_epoch": metadata.best_epoch,
            "best_validation_policy_nll": metadata.best_validation_policy_nll,
        }
        _write_bytes(temporary / "run.json", _canonical_json(run_manifest))
        _write_bytes(temporary / "scenario.json", _canonical_json(scenario_value))
        _write_bytes(
            temporary / "policy-identity.json",
            _canonical_json(_identity_wire(identity)),
        )
        _write_bytes(temporary / "corpus-manifest.json", corpus_manifest_bytes)
        metrics_bytes = _metrics_jsonl(result.history)
        _metrics_from_jsonl(
            metrics_bytes,
            metadata.best_epoch,
            metadata.best_validation_policy_nll,
        )
        _write_bytes(temporary / "metrics.jsonl", metrics_bytes)
        _write_bytes(temporary / "inference-fixture.json", _canonical_json({
            "examples": [_example_wire(item) for item in fixture.examples],
            "valid_candidate_logits": [list(item) for item in fixture.valid_candidate_logits],
            "selected_identities": [asdict(item) for item in fixture.selected_identities],
        }))
        validate_structured_run(temporary)
        _require_plain_lexical_chain(run_dir.parent, "run destination")
        if run_dir.exists() or _is_reparse(run_dir):
            raise FileExistsError(f"refusing to overwrite existing run {run_dir}")
        _publish_no_replace(temporary, run_dir)
        return run_dir
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def validate_structured_run(run_dir: Path) -> LoadedStructuredPolicy:
    root = _public_run_path(run_dir, "run directory")
    _require_plain_lexical_chain(root, "run directory")
    entries = _require_plain_run_inventory(root)
    checkpoint_dir = entries["checkpoints"]
    checkpoint = checkpoint_dir / "best.pt"
    manifest = _plain_mapping(
        _read_canonical_json_file(root / "run.json", "run manifest"),
        _RUN_FIELDS,
        "run manifest",
    )
    if _int(manifest["schema_version"], "run manifest schema_version") != 2:
        raise ValueError("run manifest schema_version is invalid")
    if _string(manifest["state"], "run manifest state") != "completed":
        raise ValueError("run manifest state is invalid")
    if (
        _string(manifest["evidence_status"], "run manifest evidence_status")
        != "unsealed-experimental"
    ):
        raise ValueError("run manifest evidence_status is invalid")
    config = _plain_mapping(manifest["config"], frozenset({"algorithm"}), "run manifest config")
    if _string(config["algorithm"], "run manifest config.algorithm") != "structured_imitation":
        raise ValueError("run manifest algorithm is invalid")
    contract = _plain_mapping(manifest["contract"], frozenset({
        "environment", "version", "environment_kind", "contract_hash", "encoding_hash", "capacity_hash",
    }), "run manifest contract")
    for name in contract:
        _string(contract[name], f"run manifest contract.{name}")
    scenario_value = _read_canonical_json_file(
        root / "scenario.json", "training scenario"
    )
    if (
        _validate_relative(
            manifest["policy_identity"], "run manifest policy_identity"
        )
        != "policy-identity.json"
    ):
        raise ValueError("run manifest policy_identity is invalid")
    identity = parse_spaces(
        _read_canonical_json_file(
            root / "policy-identity.json", "policy identity"
        )
    )
    if contract != _identity_manifest(identity):
        raise ValueError("run manifest contract does not match policy identity")
    if _validate_relative(manifest["latest_checkpoint"], "run manifest latest_checkpoint") != "checkpoints/best.pt":
        raise ValueError("run manifest latest_checkpoint is invalid")
    dataset_sha256 = _sha256(
        manifest["dataset_manifest_sha256"],
        "run manifest dataset_manifest_sha256",
    )
    best_epoch = _int(manifest["best_epoch"], "run manifest best_epoch", minimum=0)
    latest_step = _int(
        manifest["latest_checkpoint_step"],
        "run manifest latest_checkpoint_step",
        minimum=0,
    )
    best_nll = _float(
        manifest["best_validation_policy_nll"],
        "run manifest best_validation_policy_nll",
    )
    metrics_bytes = (root / "metrics.jsonl").read_bytes()
    _metrics_from_jsonl(
        metrics_bytes,
        best_epoch,
        best_nll,
    )
    corpus_manifest_path = root / "corpus-manifest.json"
    corpus_manifest_bytes = corpus_manifest_path.read_bytes()
    corpus_value = _decode_json_bytes(
        corpus_manifest_bytes, "corpus manifest"
    )
    if corpus_manifest_bytes != _canonical_json(corpus_value):
        raise ValueError("corpus manifest must be canonical compact JSON")
    if (
        type(corpus_value) is dict
        and corpus_value.get("kind") == _ADOPTED_EVIDENCE_KIND
    ):
        loaded = load_structured_checkpoint(
            checkpoint,
            identity.encoding_hash,
            identity.capacity_hash,
        )
        corpus_identity = _validate_adopted_evidence(
            corpus_value,
            root=root,
            loaded=loaded,
            scenario_value=scenario_value,
            metrics_bytes=metrics_bytes,
        )
        if corpus_identity != dataset_sha256:
            raise ValueError("corpus SHA-256 does not match run manifest")
    else:
        corpus_identity, _ = _validate_manifest(
            root,
            identity,
            corpus_manifest_bytes,
        )
        if corpus_identity != dataset_sha256:
            raise ValueError("corpus SHA-256 does not match run manifest")
        loaded = load_structured_checkpoint(
            checkpoint,
            identity.encoding_hash,
            identity.capacity_hash,
        )
    if loaded.metadata.corpus_sha256 != dataset_sha256:
        raise ValueError("corpus SHA-256 does not match checkpoint")
    if loaded.metadata.identity != identity:
        raise ValueError("checkpoint identity does not match policy identity")
    if latest_step != loaded.metadata.best_epoch or best_epoch != loaded.metadata.best_epoch:
        raise ValueError("run manifest best epoch does not match checkpoint")
    if best_nll != loaded.metadata.best_validation_policy_nll:
        raise ValueError("run manifest best validation policy NLL does not match checkpoint")
    fixture_value = _read_canonical_json_file(
        root / "inference-fixture.json",
        "inference fixture",
    )
    if _fixture_from_wire(fixture_value, identity) != loaded.fixture:
        raise ValueError("run inference fixture does not match checkpoint")
    return loaded
