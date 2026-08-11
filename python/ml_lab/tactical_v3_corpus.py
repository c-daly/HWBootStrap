"""Immutable, content-addressed tactical-v3 smoke-corpus artifacts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Literal

from .tactical_v3_client import CandidateSelection, TacticalV3GymClient
from .tactical_v3_schema import (
    Candidate,
    TacticalV3Decision,
    TacticalV3SemanticIdentity,
    canonical_sha256,
    parse_decision,
)


_CORPUS_SCHEMA_VERSION = 1
_LABEL_SOURCE = "tiny-fixture-policy-v1"
_TINY_TEACHER = ("tiny-fixture-policy-v1", 0, 0, 0, "none", None)
_TINY_PROFILE = "standard-3v3"
_TRAIN_SEEDS = (4101, 4102)
_VALIDATION_SEEDS = (5101,)
_INVENTORY = frozenset({"manifest.json", "train.jsonl", "validation.jsonl"})
_MANIFEST_FIELDS = frozenset({
    "corpus_schema_version", "identity", "label_source", "scenario_id", "contract_hash",
    "encoding_hash", "capacity_hash", "environment_kind", "files",
})
_FILE_FIELDS = frozenset({"path", "partition", "byte_length", "sha256", "row_count", "seeds"})
_ROW_FIELDS = frozenset({
    "example_schema_version", "decision", "target", "teacher", "scenario_id", "contract_hash",
    "encoding_hash", "capacity_hash", "profile_id", "episode_seed", "learner_seat",
})
_TARGET_FIELDS = frozenset({
    "teacher_candidate_id", "terminal_outcome", "trajectory_index",
    "remaining_turns_to_victory", "truncated",
})
_TEACHER_FIELDS = frozenset({
    "identity", "search_depth", "expansion_budget", "actual_expansions", "heuristic_identity",
    "confidence",
})


@dataclass(frozen=True, slots=True)
class TeacherEvidence:
    identity: str
    search_depth: int
    expansion_budget: int
    actual_expansions: int
    heuristic_identity: str
    confidence: float | None


@dataclass(frozen=True, slots=True)
class StructuredTarget:
    teacher_candidate_id: int
    terminal_outcome: Literal["win", "loss", "draw"]
    trajectory_index: int
    remaining_turns_to_victory: int | None
    truncated: bool


@dataclass(frozen=True, slots=True)
class StructuredExample:
    example_schema_version: Literal[1]
    decision: TacticalV3Decision
    target: StructuredTarget
    teacher: TeacherEvidence
    scenario_id: str
    contract_hash: str
    encoding_hash: str
    capacity_hash: str
    profile_id: str
    episode_seed: int
    learner_seat: int


@dataclass(frozen=True, slots=True)
class StructuredCorpus:
    root: Path
    identity: str
    train: tuple[StructuredExample, ...]
    validation: tuple[StructuredExample, ...]


def _exact_mapping(value: object, fields: frozenset[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != fields or not all(type(key) is str for key in value):
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a non-empty string")
    return value


def _int(value: object, label: str, *, nonnegative: bool = False) -> int:
    if type(value) is not int or not -(2**31) <= value < 2**31:
        raise TypeError(f"{label} must be an int32")
    if nonnegative and value < 0:
        raise ValueError(f"{label} must be nonnegative")
    return value


def _hash(value: object, label: str) -> str:
    result = _text(value, label)
    if len(result) != 64 or any(char not in "0123456789abcdef" for char in result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
    return result


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a bool")
    return value


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))


def _require_plain_directory(root: Path) -> None:
    if _is_reparse(root) or not root.is_dir():
        raise ValueError("corpus root must be a plain directory, not a symlink or reparse point")


def _require_inventory(root: Path) -> None:
    _require_plain_directory(root)
    entries = {entry.name: entry for entry in root.iterdir()}
    if set(entries) != _INVENTORY:
        raise ValueError("corpus inventory must contain exactly manifest.json, train.jsonl, validation.jsonl")
    for name, entry in entries.items():
        if _is_reparse(entry) or not entry.is_file():
            raise ValueError(f"corpus inventory entry {name!r} must be a plain file")


def _read_json(path: Path, label: str) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid UTF-8 JSON") from error


def _manifest_without_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in manifest.items() if key != "identity"}


def _validate_manifest(
    root: Path, expected: TacticalV3SemanticIdentity,
) -> tuple[str, tuple[Mapping[str, object], Mapping[str, object]]]:
    manifest = _exact_mapping(_read_json(root / "manifest.json", "manifest"), _MANIFEST_FIELDS, "manifest")
    if _int(manifest["corpus_schema_version"], "manifest.corpus_schema_version") != _CORPUS_SCHEMA_VERSION:
        raise ValueError("manifest corpus_schema_version is unsupported")
    identity = _hash(manifest["identity"], "manifest.identity")
    if identity != canonical_sha256(_manifest_without_identity(manifest)):
        raise ValueError("manifest identity does not content-address the manifest")
    if manifest["label_source"] != _LABEL_SOURCE:
        raise ValueError("manifest label_source is not the tiny fixture policy")
    for field, actual, wanted in (
        ("scenario_id", manifest["scenario_id"], expected.scenario_id),
        ("contract_hash", manifest["contract_hash"], expected.contract_hash),
        ("encoding_hash", manifest["encoding_hash"], expected.encoding_hash),
        ("capacity_hash", manifest["capacity_hash"], expected.capacity_hash),
        ("environment_kind", manifest["environment_kind"], expected.environment_kind),
    ):
        if actual != wanted:
            raise ValueError(f"manifest {field} does not match the expected tactical-v3 identity")
    files = manifest["files"]
    if type(files) is not list or len(files) != 2:
        raise ValueError("manifest files must describe exactly the train and validation partitions")
    parsed = tuple(_exact_mapping(item, _FILE_FIELDS, f"manifest.files[{index}]")
                   for index, item in enumerate(files))
    required = (("train.jsonl", "train"), ("validation.jsonl", "validation"))
    for item, (path, partition) in zip(parsed, required, strict=True):
        if item["path"] != path or item["partition"] != partition:
            raise ValueError("manifest files must be ordered train.jsonl then validation.jsonl")
        _int(item["byte_length"], f"manifest.{path}.byte_length", nonnegative=True)
        _hash(item["sha256"], f"manifest.{path}.sha256")
        _int(item["row_count"], f"manifest.{path}.row_count", nonnegative=True)
        seeds = item["seeds"]
        if type(seeds) is not list or not seeds:
            raise ValueError(f"manifest.{path}.seeds must be a non-empty list")
        parsed_seeds = tuple(_int(seed, f"manifest.{path}.seeds[]") for seed in seeds)
        if parsed_seeds != tuple(sorted(set(parsed_seeds))):
            raise ValueError(f"manifest.{path}.seeds must be sorted and unique")
    if set(parsed[0]["seeds"]) & set(parsed[1]["seeds"]):
        raise ValueError("manifest partitions must have disjoint episode seeds")
    return identity, parsed  # type: ignore[return-value]


def _parse_teacher(value: object) -> TeacherEvidence:
    data = _exact_mapping(value, _TEACHER_FIELDS, "teacher")
    confidence = data["confidence"]
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise TypeError("teacher.confidence must be a finite number or null")
        confidence = float(confidence)
        if not math.isfinite(confidence):
            raise ValueError("teacher.confidence must be finite")
    result = TeacherEvidence(
        _text(data["identity"], "teacher.identity"),
        _int(data["search_depth"], "teacher.search_depth", nonnegative=True),
        _int(data["expansion_budget"], "teacher.expansion_budget", nonnegative=True),
        _int(data["actual_expansions"], "teacher.actual_expansions", nonnegative=True),
        _text(data["heuristic_identity"], "teacher.heuristic_identity"), confidence,
    )
    if result != TeacherEvidence(*_TINY_TEACHER):
        raise ValueError("tiny corpus teacher evidence must be exactly tiny-fixture-policy-v1")
    return result


def _parse_target(value: object, decision: TacticalV3Decision) -> StructuredTarget:
    data = _exact_mapping(value, _TARGET_FIELDS, "target")
    candidate_id = _int(data["teacher_candidate_id"], "target.teacher_candidate_id", nonnegative=True)
    if candidate_id not in {candidate.candidate_id for candidate in decision.candidates}:
        raise ValueError("target teacher_candidate_id does not identify a candidate in its decision")
    outcome = data["terminal_outcome"]
    if outcome not in {"win", "loss", "draw"}:
        raise ValueError("target.terminal_outcome must be win, loss, or draw")
    remaining = data["remaining_turns_to_victory"]
    if remaining is not None:
        remaining = _int(remaining, "target.remaining_turns_to_victory", nonnegative=True)
    if (outcome == "win") != (remaining is not None):
        raise ValueError("remaining_turns_to_victory must be present only for wins")
    return StructuredTarget(
        candidate_id, outcome, _int(data["trajectory_index"], "target.trajectory_index", nonnegative=True),
        remaining, _bool(data["truncated"], "target.truncated"),
    )


def _parse_row(value: object, expected: TacticalV3SemanticIdentity) -> StructuredExample:
    data = _exact_mapping(value, _ROW_FIELDS, "example")
    if _int(data["example_schema_version"], "example.example_schema_version") != 1:
        raise ValueError("example schema version is unsupported")
    for field, wanted in (
        ("scenario_id", expected.scenario_id), ("contract_hash", expected.contract_hash),
        ("encoding_hash", expected.encoding_hash), ("capacity_hash", expected.capacity_hash),
    ):
        if data[field] != wanted:
            raise ValueError(f"example {field} does not match the expected tactical-v3 identity")
    decision = parse_decision(data["decision"], expected)
    target = _parse_target(data["target"], decision)
    learner_seat = _int(data["learner_seat"], "example.learner_seat")
    if learner_seat not in {0, 1} or decision.seat != learner_seat:
        raise ValueError("example learner_seat must match the decision seat")
    return StructuredExample(
        1, decision, target, _parse_teacher(data["teacher"]), expected.scenario_id,
        expected.contract_hash, expected.encoding_hash, expected.capacity_hash,
        _text(data["profile_id"], "example.profile_id"),
        _int(data["episode_seed"], "example.episode_seed"), learner_seat,
    )


def _read_partition(
    root: Path, metadata: Mapping[str, object], expected: TacticalV3SemanticIdentity,
) -> tuple[StructuredExample, ...]:
    path = root / str(metadata["path"])
    data = path.read_bytes()
    if _sha256_bytes(data) != metadata["sha256"]:
        raise ValueError(f"{path.name} SHA-256 does not match manifest")
    if len(data) != metadata["byte_length"]:
        raise ValueError(f"{path.name} byte length does not match manifest")
    if not data or not data.endswith(b"\n"):
        raise ValueError(f"{path.name} must be non-empty newline-delimited JSON")
    rows: list[StructuredExample] = []
    raw_rows: set[str] = set()
    for index, line in enumerate(data.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError(f"{path.name} row {index} is not canonical JSONL")
        try:
            text = line[:-1].decode("utf-8")
            raw = json.loads(text)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"{path.name} row {index} is not valid UTF-8 JSON") from error
        canonical = _canonical_bytes(raw)
        if canonical != line:
            raise ValueError(f"{path.name} row {index} is not canonical compact JSON")
        row_hash = _sha256_bytes(canonical)
        if row_hash in raw_rows:
            raise ValueError(f"{path.name} contains a duplicate example")
        raw_rows.add(row_hash)
        rows.append(_parse_row(raw, expected))
    if len(rows) != metadata["row_count"]:
        raise ValueError(f"{path.name} row count does not match manifest")
    actual_seeds = tuple(sorted({row.episode_seed for row in rows}))
    if actual_seeds != tuple(metadata["seeds"]):
        raise ValueError(f"{path.name} seed list does not match manifest")
    return tuple(rows)


def _validate_partitions(train: tuple[StructuredExample, ...], validation: tuple[StructuredExample, ...]) -> None:
    train_seeds = {row.episode_seed for row in train}
    validation_seeds = {row.episode_seed for row in validation}
    if train_seeds & validation_seeds:
        raise ValueError("cross-partition episode seed reuse is forbidden")
    for partition, rows in (("train", train), ("validation", validation)):
        by_seed: dict[int, list[StructuredExample]] = defaultdict(list)
        for row in rows:
            by_seed[row.episode_seed].append(row)
        for seed, examples in by_seed.items():
            indices = tuple(sorted(row.target.trajectory_index for row in examples))
            if indices != tuple(range(len(examples))):
                raise ValueError(f"{partition} seed {seed} trajectory indices must be contiguous")
            outcomes = {row.target.terminal_outcome for row in examples}
            truncations = {row.target.truncated for row in examples}
            if len(outcomes) != 1 or len(truncations) != 1:
                raise ValueError(f"{partition} seed {seed} does not have a single terminal outcome")
            if len({(row.target.trajectory_index, row.decision.decision_id) for row in examples}) != len(examples):
                raise ValueError(f"{partition} seed {seed} contains duplicate decision examples")


def load_corpus(root: Path, expected: TacticalV3SemanticIdentity) -> StructuredCorpus:
    """Load and physically authenticate the exact tiny corpus without mutating it."""
    if type(expected) is not TacticalV3SemanticIdentity:
        raise TypeError("expected must be a TacticalV3SemanticIdentity")
    root = Path(root)
    _require_inventory(root)
    identity, files = _validate_manifest(root, expected)
    train = _read_partition(root, files[0], expected)
    validation = _read_partition(root, files[1], expected)
    _validate_partitions(train, validation)
    return StructuredCorpus(root, identity, train, validation)


def _row_wire(example: StructuredExample) -> dict[str, object]:
    return asdict(example)


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _outcome(view, learner_seat: int) -> Literal["win", "loss", "draw"]:
    if view.winner == learner_seat:
        return "win"
    if view.winner in {0, 1}:
        return "loss"
    return "draw"


def _select_tiny_candidate(candidates: Sequence[Candidate]) -> Candidate:
    """Use the deliberately fixed smoke-corpus policy, with no teacher configuration."""
    for candidate in candidates:
        if candidate.kind != "end_turn":
            return candidate
    if len(candidates) == 1 and candidates[0].kind == "end_turn":
        return candidates[0]
    raise ValueError("tiny corpus policy requires a first non-end_turn candidate or the sole end_turn")


def _collect_seed(
    client: TacticalV3GymClient, seed: int,
) -> tuple[StructuredExample, ...]:
    learner_seat = 0
    view = client.duel_reset(
        seed, "external", "random", learner_seat, _TINY_PROFILE, learner_seat,
    )
    records: list[tuple[TacticalV3Decision, int]] = []
    while not view.terminated and not view.truncated:
        if view.decision.seat != learner_seat or view.start_profile != _TINY_PROFILE:
            raise ValueError("tiny corpus duel did not expose the fixed external learner profile")
        candidate = _select_tiny_candidate(view.decision.candidates)
        records.append((view.decision, candidate.candidate_id))
        view = client.duel_step(CandidateSelection(view.decision.decision_id, candidate.candidate_id))
    if not records:
        raise ValueError(f"tiny corpus seed {seed} ended before the learner made a decision")
    outcome = _outcome(view, learner_seat)
    retained = records[:4]
    examples: list[StructuredExample] = []
    for index, (decision, candidate_id) in enumerate(retained):
        remaining = len(records) - index - 1 if outcome == "win" else None
        examples.append(StructuredExample(
            1, decision,
            StructuredTarget(candidate_id, outcome, index, remaining, view.truncated),
            TeacherEvidence(*_TINY_TEACHER), client.identity.scenario_id,
            client.identity.contract_hash, client.identity.encoding_hash, client.identity.capacity_hash,
            _TINY_PROFILE, seed, learner_seat,
        ))
    return tuple(examples)


def _file_metadata(path: str, partition: Literal["train", "validation"], data: bytes,
                   rows: tuple[StructuredExample, ...]) -> dict[str, object]:
    return {
        "path": path,
        "partition": partition,
        "byte_length": len(data),
        "sha256": _sha256_bytes(data),
        "row_count": len(rows),
        "seeds": sorted({row.episode_seed for row in rows}),
    }


def _create_manifest(
    identity: TacticalV3SemanticIdentity, train_data: bytes, validation_data: bytes,
    train: tuple[StructuredExample, ...], validation: tuple[StructuredExample, ...],
) -> dict[str, object]:
    manifest: dict[str, object] = {
        "corpus_schema_version": _CORPUS_SCHEMA_VERSION,
        "label_source": _LABEL_SOURCE,
        "scenario_id": identity.scenario_id,
        "contract_hash": identity.contract_hash,
        "encoding_hash": identity.encoding_hash,
        "capacity_hash": identity.capacity_hash,
        "environment_kind": identity.environment_kind,
        "files": [
            _file_metadata("train.jsonl", "train", train_data, train),
            _file_metadata("validation.jsonl", "validation", validation_data, validation),
        ],
    }
    manifest["identity"] = canonical_sha256(manifest)
    return manifest


def create_tiny_corpus(output: Path, server_cmd: Sequence[str]) -> StructuredCorpus:
    """Publish the fixed three-seed smoke corpus exactly once at ``output``."""
    output = Path(output)
    if output.exists() or _is_reparse(output):
        raise FileExistsError(f"refusing to overwrite existing corpus output {output}")
    parent = output.parent
    _require_plain_directory(parent)
    lock = parent / f".{output.name}.publish-lock"
    try:
        lock.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"corpus output is already being published: {output}") from error
    temporary: Path | None = None
    try:
        if output.exists() or _is_reparse(output):
            raise FileExistsError(f"refusing to overwrite existing corpus output {output}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=parent))
        with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
            train = tuple(item for seed in _TRAIN_SEEDS for item in _collect_seed(client, seed))
            validation = tuple(item for seed in _VALIDATION_SEEDS for item in _collect_seed(client, seed))
            identity = client.identity
        _validate_partitions(train, validation)
        train_data = b"".join(_canonical_bytes(_row_wire(row)) for row in train)
        validation_data = b"".join(_canonical_bytes(_row_wire(row)) for row in validation)
        manifest = _create_manifest(identity, train_data, validation_data, train, validation)
        _write_fsynced(temporary / "train.jsonl", train_data)
        _write_fsynced(temporary / "validation.jsonl", validation_data)
        _write_fsynced(temporary / "manifest.json", _canonical_bytes(manifest))
        _fsync_directory(temporary)
        load_corpus(temporary, identity)
        if output.exists() or _is_reparse(output):
            raise FileExistsError(f"refusing to overwrite existing corpus output {output}")
        os.rename(temporary, output)
        temporary = None
        _fsync_directory(parent)
        return load_corpus(output, identity)
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)
        try:
            lock.rmdir()
        except FileNotFoundError:
            pass
