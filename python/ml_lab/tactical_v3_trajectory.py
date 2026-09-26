"""Strict, append-safe tactical-v3 learner trajectory artifacts.

Each completed game is an immutable directory published with an atomic,
no-replace rename.  Hidden staging directories are never part of a reconstructed
archive, so an interrupted write cannot invalidate earlier games.
"""

from __future__ import annotations

import ctypes
from dataclasses import asdict, dataclass
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Literal, Mapping

from .io import atomic_write_bytes
from .tactical_v3_schema import (
    TacticalV3Decision,
    TacticalV3Reward,
    TacticalV3SemanticIdentity,
    parse_decision,
)


TrajectoryPartition = Literal["train", "validation"]
DatasetUse = Literal["optimization", "early_stop_only"]
BehaviorMode = Literal["greedy", "categorical"]
ControllerKind = Literal["model", "scripted"]

_PARTITION_USE: Mapping[str, str] = {
    "train": "optimization",
    "validation": "early_stop_only",
}
_GAME_DIRECTORY = re.compile(r"game-(\d{6})\Z")
_TEMP_DIRECTORY = re.compile(r"\.game-\d{6}\.tmp-.+\Z")
_MANIFEST_TEMP_FILE = re.compile(r"\.manifest\.json\..+\.tmp\Z")
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_GAME_FILES = frozenset({"trajectory.jsonl", "game.replay", "game.json"})
_GAME_FIELDS = frozenset({
    "schema_version",
    "kind",
    "partition",
    "dataset_use",
    "game_index",
    "schedule",
    "identity",
    "actor",
    "opponent",
    "result",
    "files",
})
_SCHEDULE_FIELDS = frozenset({
    "episode_seed", "profile_id", "learner_seat", "reference_seat",
})
_IDENTITY_FIELDS = frozenset({
    "scenario_id",
    "scenario_schema_version",
    "contract_version",
    "environment_kind",
    "contract_hash",
    "encoding_hash",
    "capacity_hash",
})
_PROVENANCE_FIELDS = frozenset({"kind", "name", "source", "artifact_sha256"})
_RESULT_FIELDS = frozenset({
    "winner",
    "outcome",
    "terminated",
    "truncated",
    "terminal_reward",
    "learner_decisions",
    "engine_commands",
    "internal_fallback_count",
})
_FILES_FIELDS = frozenset({"trajectory", "replay"})
_TRAJECTORY_FILE_FIELDS = frozenset({"path", "byte_length", "sha256", "row_count"})
_REPLAY_FILE_FIELDS = frozenset({"path", "byte_length", "sha256"})
_ROW_FIELDS = frozenset({
    "schema_version",
    "trajectory_index",
    "decision",
    "selected_candidate_id",
    "behavior",
    "successor_reward",
    "terminated_after_selection",
    "truncated_after_selection",
})
_BEHAVIOR_FIELDS = frozenset({"mode", "log_probability", "entropy"})
_REWARD_FIELDS = frozenset({
    "terminal_outcome",
    "known_health_adjusted_material_progress",
    "public_resource_progress",
    "time_pressure",
    "total",
    "finalized",
})
_ARCHIVE_FIELDS = frozenset({"schema_version", "kind", "identity", "partitions"})
_ARCHIVE_PARTITIONS = frozenset({"train", "validation"})
_ARCHIVE_PARTITION_FIELDS = frozenset({"dataset_use", "games"})
_ARCHIVE_GAME_FIELDS = frozenset({"game_index", "path", "game_manifest_sha256"})


@dataclass(frozen=True, slots=True)
class ControllerProvenance:
    """The exact model or scripted implementation that controlled one seat."""

    kind: ControllerKind
    name: str
    source: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class TrajectoryDecisionRecord:
    """One learner-visible decision and the result returned after its selection."""

    trajectory_index: int
    decision: TacticalV3Decision
    selected_candidate_id: int
    behavior_mode: BehaviorMode
    log_probability: float
    entropy: float
    successor_reward: TacticalV3Reward
    terminated_after_selection: bool
    truncated_after_selection: bool


@dataclass(frozen=True, slots=True)
class TacticalV3TrajectoryGame:
    """A complete game ready for immutable publication."""

    identity: TacticalV3SemanticIdentity
    partition: TrajectoryPartition
    game_index: int
    episode_seed: int
    profile_id: str
    learner_seat: int
    reference_seat: int
    actor: ControllerProvenance
    opponent: ControllerProvenance
    records: tuple[TrajectoryDecisionRecord, ...]
    replay: bytes
    winner: int
    terminated: bool
    truncated: bool
    terminal_reward: TacticalV3Reward
    internal_fallback_count: int

    @property
    def dataset_use(self) -> DatasetUse:
        return _PARTITION_USE[self.partition]  # type: ignore[return-value]

    @property
    def outcome(self) -> Literal["win", "loss", "draw"]:
        if self.winner == self.learner_seat:
            return "win"
        if self.winner in {0, 1}:
            return "loss"
        return "draw"


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _wire_plain(value: object) -> object:
    """Convert immutable schema containers to their strict JSON wire shapes."""

    if value is None or type(value) in {str, int, float, bool}:
        if type(value) is float and not math.isfinite(value):
            raise ValueError("trajectory wire values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(type(key) is str for key in value):
            raise TypeError("trajectory wire mappings must have built-in string keys")
        return {key: _wire_plain(item) for key, item in value.items()}
    if type(value) in {tuple, list}:
        return [_wire_plain(item) for item in value]
    raise TypeError("trajectory contains a non-JSON wire value")


def _decision_wire(value: TacticalV3Decision) -> dict[str, object]:
    wire = _wire_plain(asdict(value))
    if type(wire) is not dict:  # pragma: no cover - narrows the static type.
        raise AssertionError
    return wire


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _exact_mapping(
    value: object,
    fields: frozenset[str],
    label: str,
) -> Mapping[str, object]:
    if (
        not isinstance(value, Mapping)
        or frozenset(value) != fields
        or not all(type(key) is str for key in value)
    ):
        raise ValueError(f"{label} fields must be exactly {sorted(fields)}")
    return value


def _int(value: object, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be at least {minimum}")
    return value


def _float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _text(value: object, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _hash(value: object, label: str) -> str:
    if type(value) is not str or _HASH.fullmatch(value) is None:
        raise ValueError(f"{label} must be lowercase SHA-256")
    return value


def _bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a boolean")
    return value


def _is_reparse(path: Path) -> bool:
    try:
        attributes = os.lstat(path).st_file_attributes
    except (AttributeError, OSError):
        attributes = 0
    return path.is_symlink() or bool(
        attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    )


def _require_plain_directory(path: Path, label: str) -> None:
    if not path.is_dir() or _is_reparse(path):
        raise ValueError(f"{label} must be a plain directory")


def _write_fsynced(path: Path, data: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a finished directory without replacing an old game."""

    if os.name == "nt":
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileW
        move.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p)
        move.restype = ctypes.c_int
        if not move(str(source), str(destination)):
            code = ctypes.get_last_error()
            if code in {80, 183}:
                raise FileExistsError(
                    f"refusing to overwrite trajectory game {destination}"
                )
            raise OSError(
                code,
                "MoveFileW no-replace trajectory publication failed",
                str(destination),
            )
        return

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as error:
        raise OSError(
            "atomic no-replace directory publication is unsupported"
        ) from error
    renameat2.argtypes = (
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    )
    renameat2.restype = ctypes.c_int
    if renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(destination),
        1,
    ) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            raise FileExistsError(
                f"refusing to overwrite trajectory game {destination}"
            )
        raise OSError(
            code,
            "renameat2 no-replace trajectory publication failed",
            str(destination),
        )


def _identity_wire(identity: TacticalV3SemanticIdentity) -> dict[str, object]:
    if type(identity) is not TacticalV3SemanticIdentity:
        raise TypeError("trajectory identity must be TacticalV3SemanticIdentity")
    if identity.contract_version != "tactical-v3":
        raise ValueError("trajectory identity must use tactical-v3")
    if identity.environment_kind != "duel":
        raise ValueError("trajectory identity must describe a duel environment")
    return {
        "scenario_id": _text(identity.scenario_id, "identity scenario_id"),
        "scenario_schema_version": _int(
            identity.scenario_schema_version,
            "identity scenario_schema_version",
            minimum=1,
        ),
        "contract_version": identity.contract_version,
        "environment_kind": identity.environment_kind,
        "contract_hash": _hash(identity.contract_hash, "identity contract_hash"),
        "encoding_hash": _hash(identity.encoding_hash, "identity encoding_hash"),
        "capacity_hash": _hash(identity.capacity_hash, "identity capacity_hash"),
    }


def _parse_identity(
    value: object,
    expected: TacticalV3SemanticIdentity,
) -> None:
    data = _exact_mapping(value, _IDENTITY_FIELDS, "trajectory identity")
    if dict(data) != _identity_wire(expected):
        raise ValueError("trajectory identity does not match expected tactical-v3 identity")


def _provenance_wire(value: ControllerProvenance) -> dict[str, object]:
    if type(value) is not ControllerProvenance:
        raise TypeError("controller provenance must be ControllerProvenance")
    if value.kind not in {"model", "scripted"}:
        raise ValueError("controller provenance kind is unsupported")
    return {
        "kind": value.kind,
        "name": _text(value.name, "controller provenance name"),
        "source": _text(value.source, "controller provenance source"),
        "artifact_sha256": _hash(
            value.artifact_sha256,
            "controller provenance artifact_sha256",
        ),
    }


def _parse_provenance(value: object, label: str) -> ControllerProvenance:
    data = _exact_mapping(value, _PROVENANCE_FIELDS, label)
    kind = data["kind"]
    if kind not in {"model", "scripted"}:
        raise ValueError(f"{label} kind is unsupported")
    return ControllerProvenance(
        kind=kind,  # type: ignore[arg-type]
        name=_text(data["name"], f"{label} name"),
        source=_text(data["source"], f"{label} source"),
        artifact_sha256=_hash(
            data["artifact_sha256"], f"{label} artifact_sha256"
        ),
    )


def _reward_wire(value: TacticalV3Reward) -> dict[str, object]:
    if type(value) is not TacticalV3Reward:
        raise TypeError("successor reward must be TacticalV3Reward")
    data = asdict(value)
    for field in _REWARD_FIELDS - {"finalized"}:
        _float(data[field], f"successor reward {field}")
    _bool(data["finalized"], "successor reward finalized")
    return data


def _parse_reward(value: object) -> TacticalV3Reward:
    data = _exact_mapping(value, _REWARD_FIELDS, "successor reward")
    return TacticalV3Reward(
        terminal_outcome=_float(
            data["terminal_outcome"], "successor reward terminal_outcome"
        ),
        known_health_adjusted_material_progress=_float(
            data["known_health_adjusted_material_progress"],
            "successor reward known_health_adjusted_material_progress",
        ),
        public_resource_progress=_float(
            data["public_resource_progress"],
            "successor reward public_resource_progress",
        ),
        time_pressure=_float(
            data["time_pressure"], "successor reward time_pressure"
        ),
        total=_float(data["total"], "successor reward total"),
        finalized=_bool(data["finalized"], "successor reward finalized"),
    )


def _record_wire(
    record: TrajectoryDecisionRecord,
    identity: TacticalV3SemanticIdentity,
    learner_seat: int,
) -> dict[str, object]:
    _validate_record(record, identity, learner_seat)
    return {
        "schema_version": 1,
        "trajectory_index": record.trajectory_index,
        "decision": _decision_wire(record.decision),
        "selected_candidate_id": record.selected_candidate_id,
        "behavior": {
            "mode": record.behavior_mode,
            "log_probability": record.log_probability,
            "entropy": record.entropy,
        },
        "successor_reward": _reward_wire(record.successor_reward),
        "terminated_after_selection": record.terminated_after_selection,
        "truncated_after_selection": record.truncated_after_selection,
    }


def _validate_record(
    record: TrajectoryDecisionRecord,
    identity: TacticalV3SemanticIdentity,
    learner_seat: int,
) -> None:
    if type(record) is not TrajectoryDecisionRecord:
        raise TypeError("trajectory records must be TrajectoryDecisionRecord")
    _int(record.trajectory_index, "trajectory index", minimum=0)
    if type(record.decision) is not TacticalV3Decision:
        raise TypeError("trajectory decision must be TacticalV3Decision")
    parsed = parse_decision(_decision_wire(record.decision), identity)
    if parsed != record.decision:
        raise ValueError("trajectory decision changed through canonical parsing")
    if record.decision.seat != learner_seat:
        raise ValueError("trajectory decision seat must equal learner seat")
    selected = _int(
        record.selected_candidate_id,
        "selected candidate id",
        minimum=0,
    )
    if sum(
        candidate.candidate_id == selected
        for candidate in record.decision.candidates
    ) != 1:
        raise ValueError(
            "selected candidate id must occur exactly once in the decision"
        )
    if record.behavior_mode not in {"greedy", "categorical"}:
        raise ValueError("trajectory behavior mode is unsupported")
    log_probability = _float(
        record.log_probability, "trajectory behavior log_probability"
    )
    entropy = _float(record.entropy, "trajectory behavior entropy")
    if log_probability > 0.0:
        raise ValueError("trajectory behavior log_probability must be nonpositive")
    if entropy < 0.0:
        raise ValueError("trajectory behavior entropy must be nonnegative")
    if record.behavior_mode == "greedy" and (
        log_probability != 0.0 or entropy != 0.0
    ):
        raise ValueError(
            "greedy trajectory behavior must have log_probability and entropy zero"
        )
    terminated = _bool(
        record.terminated_after_selection,
        "trajectory terminated_after_selection",
    )
    truncated = _bool(
        record.truncated_after_selection,
        "trajectory truncated_after_selection",
    )
    if terminated and truncated:
        raise ValueError("trajectory selection cannot terminate and truncate together")
    reward = _reward_wire(record.successor_reward)
    if reward["finalized"] != (terminated or truncated):
        raise ValueError(
            "successor reward must be finalized exactly for a terminal or truncated selection"
        )


def _parse_record(
    value: object,
    identity: TacticalV3SemanticIdentity,
    learner_seat: int,
) -> TrajectoryDecisionRecord:
    data = _exact_mapping(value, _ROW_FIELDS, "trajectory row")
    if data["schema_version"] != 1:
        raise ValueError("trajectory row schema version is unsupported")
    behavior = _exact_mapping(
        data["behavior"], _BEHAVIOR_FIELDS, "trajectory behavior"
    )
    record = TrajectoryDecisionRecord(
        trajectory_index=_int(
            data["trajectory_index"], "trajectory row index", minimum=0
        ),
        decision=parse_decision(data["decision"], identity),
        selected_candidate_id=_int(
            data["selected_candidate_id"],
            "trajectory selected candidate id",
            minimum=0,
        ),
        behavior_mode=behavior["mode"],  # type: ignore[arg-type]
        log_probability=_float(
            behavior["log_probability"], "trajectory behavior log_probability"
        ),
        entropy=_float(behavior["entropy"], "trajectory behavior entropy"),
        successor_reward=_parse_reward(data["successor_reward"]),
        terminated_after_selection=_bool(
            data["terminated_after_selection"],
            "trajectory terminated_after_selection",
        ),
        truncated_after_selection=_bool(
            data["truncated_after_selection"],
            "trajectory truncated_after_selection",
        ),
    )
    _validate_record(record, identity, learner_seat)
    return record


def _validate_records(
    records: tuple[TrajectoryDecisionRecord, ...],
    identity: TacticalV3SemanticIdentity,
    learner_seat: int,
    terminated: bool,
    truncated: bool,
    terminal_reward: TacticalV3Reward,
) -> None:
    if type(records) is not tuple:
        raise TypeError("trajectory records must be a tuple")
    for index, record in enumerate(records):
        _validate_record(record, identity, learner_seat)
        if record.trajectory_index != index:
            raise ValueError("trajectory indices must be contiguous from zero")
        if index and record.decision.decision_id <= records[index - 1].decision.decision_id:
            raise ValueError("trajectory decision ids must increase strictly")
        if index < len(records) - 1 and (
            record.terminated_after_selection or record.truncated_after_selection
        ):
            raise ValueError("only the final trajectory row may finish the game")
    if records:
        final = records[-1]
        selection_finished = (
            final.terminated_after_selection or final.truncated_after_selection
        )
        if selection_finished and (
            final.terminated_after_selection != terminated
            or final.truncated_after_selection != truncated
            or final.successor_reward != terminal_reward
        ):
            raise ValueError(
                "a learner-finished final row must match the finalized game result"
            )


def _replay_command_count(replay: bytes) -> int:
    if type(replay) is not bytes or not replay:
        raise ValueError("trajectory replay must be nonempty bytes")
    try:
        text = replay.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("trajectory replay must be UTF-8") from error
    if not text.startswith("HEXWARS-REPLAY 1\n") or "\r" in text:
        raise ValueError("trajectory replay must use canonical HexWars replay v1 text")
    lines = text.split("\n")
    markers = [index for index, line in enumerate(lines) if line.startswith("CMDS ")]
    if len(markers) != 1:
        raise ValueError("trajectory replay must contain exactly one CMDS marker")
    marker = markers[0]
    pieces = lines[marker].split(" ")
    if len(pieces) != 2:
        raise ValueError("trajectory replay CMDS marker is malformed")
    try:
        count = int(pieces[1])
    except ValueError as error:
        raise ValueError("trajectory replay command count is invalid") from error
    command_lines = [line for line in lines[marker + 1:] if line]
    if count < 0 or len(command_lines) != count:
        raise ValueError("trajectory replay command count does not match its commands")
    return count


def _validate_game(game: TacticalV3TrajectoryGame) -> int:
    if type(game) is not TacticalV3TrajectoryGame:
        raise TypeError("trajectory game must be TacticalV3TrajectoryGame")
    _identity_wire(game.identity)
    if game.partition not in _PARTITION_USE:
        raise ValueError("trajectory partition is unsupported")
    game_index = _int(game.game_index, "trajectory game index", minimum=0)
    if game_index > 999_999:
        raise ValueError("trajectory game index must fit six decimal digits")
    seed = _int(game.episode_seed, "trajectory episode seed", minimum=0)
    if seed >= 2**31:
        raise ValueError("trajectory episode seed must be an int32")
    _text(game.profile_id, "trajectory profile id")
    if type(game.learner_seat) is not int or game.learner_seat not in {0, 1}:
        raise ValueError("trajectory learner seat must be 0 or 1")
    if (
        type(game.reference_seat) is not int
        or game.reference_seat not in {0, 1}
        or game.reference_seat != game.learner_seat
    ):
        raise ValueError("trajectory reference seat must equal learner seat")
    actor = _provenance_wire(game.actor)
    _provenance_wire(game.opponent)
    if actor["kind"] != "model":
        raise ValueError("trajectory actor provenance must identify a model")
    if type(game.terminated) is not bool or type(game.truncated) is not bool:
        raise TypeError("trajectory game terminal flags must be booleans")
    if game.terminated == game.truncated:
        raise ValueError("trajectory game must terminate or truncate exactly once")
    if type(game.winner) is not int or game.winner not in {-1, 0, 1}:
        raise ValueError("trajectory winner must be -1, 0, or 1")
    if game.truncated and game.winner != -1:
        raise ValueError("a truncated trajectory cannot declare a winner")
    terminal_reward = _reward_wire(game.terminal_reward)
    if terminal_reward["finalized"] is not True:
        raise ValueError("trajectory game terminal reward must be finalized")
    expected_terminal_outcome = 1.0 if game.outcome == "win" else -1.0
    if terminal_reward["terminal_outcome"] != expected_terminal_outcome:
        raise ValueError(
            "trajectory terminal reward does not match winner and learner seat"
        )
    _int(
        game.internal_fallback_count,
        "trajectory internal fallback count",
        minimum=0,
    )
    _validate_records(
        game.records,
        game.identity,
        game.learner_seat,
        game.terminated,
        game.truncated,
        game.terminal_reward,
    )
    return _replay_command_count(game.replay)


def _game_wire(
    game: TacticalV3TrajectoryGame,
    trajectory_bytes: bytes,
    replay_bytes: bytes,
    command_count: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "tactical-v3-learner-trajectory-game",
        "partition": game.partition,
        "dataset_use": game.dataset_use,
        "game_index": game.game_index,
        "schedule": {
            "episode_seed": game.episode_seed,
            "profile_id": game.profile_id,
            "learner_seat": game.learner_seat,
            "reference_seat": game.reference_seat,
        },
        "identity": _identity_wire(game.identity),
        "actor": _provenance_wire(game.actor),
        "opponent": _provenance_wire(game.opponent),
        "result": {
            "winner": game.winner,
            "outcome": game.outcome,
            "terminated": game.terminated,
            "truncated": game.truncated,
            "terminal_reward": _reward_wire(game.terminal_reward),
            "learner_decisions": len(game.records),
            "engine_commands": command_count,
            "internal_fallback_count": game.internal_fallback_count,
        },
        "files": {
            "trajectory": {
                "path": "trajectory.jsonl",
                "byte_length": len(trajectory_bytes),
                "sha256": _sha256(trajectory_bytes),
                "row_count": len(game.records),
            },
            "replay": {
                "path": "game.replay",
                "byte_length": len(replay_bytes),
                "sha256": _sha256(replay_bytes),
            },
        },
    }


def _ensure_archive(root: Path) -> None:
    if os.path.lexists(root):
        _require_plain_directory(root, "trajectory archive root")
    else:
        root.mkdir(parents=True)
    for partition in _ARCHIVE_PARTITIONS:
        path = root / partition
        if os.path.lexists(path):
            _require_plain_directory(path, f"trajectory {partition} root")
        else:
            path.mkdir()


def publish_trajectory_game(
    archive_root: Path,
    game: TacticalV3TrajectoryGame,
) -> Path:
    """Publish one complete game atomically, refusing every overwrite."""

    command_count = _validate_game(game)
    archive_root = Path(archive_root)
    _ensure_archive(archive_root)
    partition_root = archive_root / game.partition
    destination = partition_root / f"game-{game.game_index:06d}"
    if os.path.lexists(destination) or _is_reparse(destination):
        raise FileExistsError(f"trajectory game already exists: {destination}")

    trajectory_bytes = b"".join(
        _canonical_bytes(_record_wire(record, game.identity, game.learner_seat))
        for record in game.records
    )
    manifest = _game_wire(
        game,
        trajectory_bytes,
        game.replay,
        command_count,
    )
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".game-{game.game_index:06d}.tmp-",
            dir=partition_root,
        )
    )
    try:
        _write_fsynced(temporary / "trajectory.jsonl", trajectory_bytes)
        _write_fsynced(temporary / "game.replay", game.replay)
        # This is deliberately last: it is the completed-game marker.
        _write_fsynced(temporary / "game.json", _canonical_bytes(manifest))
        _fsync_directory(temporary)
        load_trajectory_game(temporary, game.identity)
        _publish_no_replace(temporary, destination)
        temporary = None  # type: ignore[assignment]
        _fsync_directory(partition_root)
        return destination
    finally:
        if temporary is not None and temporary.exists():
            shutil.rmtree(temporary)


def _read_canonical_json(data: bytes, label: str) -> Mapping[str, object]:
    try:
        value = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is invalid JSON") from error
    if not isinstance(value, Mapping) or data != _canonical_bytes(value):
        raise ValueError(f"{label} must be canonical JSON")
    return value


def _read_trajectory_rows(
    data: bytes,
    identity: TacticalV3SemanticIdentity,
    learner_seat: int,
) -> tuple[TrajectoryDecisionRecord, ...]:
    rows: list[TrajectoryDecisionRecord] = []
    for index, line in enumerate(data.splitlines(keepends=True)):
        if not line.endswith(b"\n") or line == b"\n":
            raise ValueError(f"trajectory row {index} is not canonical JSONL")
        try:
            value = json.loads(line)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"trajectory row {index} is invalid JSON") from error
        if line != _canonical_bytes(value):
            raise ValueError(f"trajectory row {index} is not canonical JSON")
        rows.append(_parse_record(value, identity, learner_seat))
    return tuple(rows)


def load_trajectory_game(
    game_directory: Path,
    expected_identity: TacticalV3SemanticIdentity,
) -> TacticalV3TrajectoryGame:
    """Authenticate and parse one immutable completed-game directory."""

    _identity_wire(expected_identity)
    game_directory = Path(game_directory)
    _require_plain_directory(game_directory, "trajectory game")
    entries = {path.name: path for path in game_directory.iterdir()}
    if set(entries) != _GAME_FILES or any(
        not path.is_file() or _is_reparse(path) for path in entries.values()
    ):
        raise ValueError("trajectory game inventory must contain exactly three plain files")

    manifest_bytes = entries["game.json"].read_bytes()
    trajectory_bytes = entries["trajectory.jsonl"].read_bytes()
    replay_bytes = entries["game.replay"].read_bytes()
    manifest = _exact_mapping(
        _read_canonical_json(manifest_bytes, "trajectory game manifest"),
        _GAME_FIELDS,
        "trajectory game manifest",
    )
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != "tactical-v3-learner-trajectory-game"
    ):
        raise ValueError("trajectory game manifest contract is unsupported")

    partition = manifest["partition"]
    if partition not in _PARTITION_USE:
        raise ValueError("trajectory game partition is unsupported")
    dataset_use = manifest["dataset_use"]
    if dataset_use != _PARTITION_USE[partition]:
        raise ValueError("trajectory partition and dataset use do not match")
    if game_directory.parent.name != partition:
        raise ValueError("trajectory directory and manifest partition do not match")
    game_index = _int(manifest["game_index"], "trajectory game index", minimum=0)
    if game_index > 999_999:
        raise ValueError("trajectory game index must fit six decimal digits")
    _parse_identity(manifest["identity"], expected_identity)

    schedule = _exact_mapping(
        manifest["schedule"], _SCHEDULE_FIELDS, "trajectory schedule"
    )
    episode_seed = _int(
        schedule["episode_seed"], "trajectory episode seed", minimum=0
    )
    if episode_seed >= 2**31:
        raise ValueError("trajectory episode seed must be an int32")
    learner_seat = _int(schedule["learner_seat"], "trajectory learner seat")
    reference_seat = _int(
        schedule["reference_seat"], "trajectory reference seat"
    )
    if learner_seat not in {0, 1} or reference_seat != learner_seat:
        raise ValueError("trajectory reference seat must equal learner seat")
    profile_id = _text(schedule["profile_id"], "trajectory profile id")
    actor = _parse_provenance(manifest["actor"], "trajectory actor")
    opponent = _parse_provenance(manifest["opponent"], "trajectory opponent")
    if actor.kind != "model":
        raise ValueError("trajectory actor provenance must identify a model")

    files = _exact_mapping(manifest["files"], _FILES_FIELDS, "trajectory files")
    trajectory_meta = _exact_mapping(
        files["trajectory"],
        _TRAJECTORY_FILE_FIELDS,
        "trajectory data file",
    )
    replay_meta = _exact_mapping(
        files["replay"], _REPLAY_FILE_FIELDS, "trajectory replay file"
    )
    for metadata, name, data in (
        (trajectory_meta, "trajectory.jsonl", trajectory_bytes),
        (replay_meta, "game.replay", replay_bytes),
    ):
        if metadata["path"] != name:
            raise ValueError(f"trajectory file path must be {name}")
        if _int(metadata["byte_length"], f"{name} byte length", minimum=0) != len(data):
            raise ValueError(f"{name} byte length changed")
        if _hash(metadata["sha256"], f"{name} SHA-256") != _sha256(data):
            raise ValueError(f"{name} SHA-256 changed")

    records = _read_trajectory_rows(
        trajectory_bytes, expected_identity, learner_seat
    )
    if _int(
        trajectory_meta["row_count"], "trajectory row count", minimum=0
    ) != len(records):
        raise ValueError("trajectory row count changed")
    command_count = _replay_command_count(replay_bytes)

    result = _exact_mapping(
        manifest["result"], _RESULT_FIELDS, "trajectory result"
    )
    winner = _int(result["winner"], "trajectory winner")
    if winner not in {-1, 0, 1}:
        raise ValueError("trajectory winner must be -1, 0, or 1")
    terminated = _bool(result["terminated"], "trajectory terminated")
    truncated = _bool(result["truncated"], "trajectory truncated")
    if terminated == truncated:
        raise ValueError("trajectory game must terminate or truncate exactly once")
    if truncated and winner != -1:
        raise ValueError("a truncated trajectory cannot declare a winner")
    terminal_reward = _parse_reward(result["terminal_reward"])
    if not terminal_reward.finalized:
        raise ValueError("trajectory game terminal reward must be finalized")
    if _int(
        result["learner_decisions"],
        "trajectory learner decision count",
        minimum=0,
    ) != len(records):
        raise ValueError("trajectory learner decision count changed")
    if _int(
        result["engine_commands"], "trajectory engine command count", minimum=0
    ) != command_count:
        raise ValueError("trajectory engine command count changed")
    fallback_count = _int(
        result["internal_fallback_count"],
        "trajectory internal fallback count",
        minimum=0,
    )
    game = TacticalV3TrajectoryGame(
        identity=expected_identity,
        partition=partition,  # type: ignore[arg-type]
        game_index=game_index,
        episode_seed=episode_seed,
        profile_id=profile_id,
        learner_seat=learner_seat,
        reference_seat=reference_seat,
        actor=actor,
        opponent=opponent,
        records=records,
        replay=replay_bytes,
        winner=winner,
        terminated=terminated,
        truncated=truncated,
        terminal_reward=terminal_reward,
        internal_fallback_count=fallback_count,
    )
    _validate_game(game)
    if result["outcome"] != game.outcome:
        raise ValueError("trajectory outcome does not match winner and learner seat")

    match = _GAME_DIRECTORY.fullmatch(game_directory.name)
    if match is not None and int(match.group(1)) != game.game_index:
        raise ValueError("trajectory directory and game index do not match")
    return game


def _archive_entries(root: Path, partition: str) -> list[Path]:
    partition_root = root / partition
    _require_plain_directory(partition_root, f"trajectory {partition} root")
    games: list[tuple[int, Path]] = []
    for path in partition_root.iterdir():
        match = _GAME_DIRECTORY.fullmatch(path.name)
        if match is not None:
            _require_plain_directory(path, "trajectory game")
            games.append((int(match.group(1)), path))
            continue
        if _TEMP_DIRECTORY.fullmatch(path.name) is not None:
            # Only a real plain directory can be an interrupted staging artifact.
            _require_plain_directory(path, "trajectory staging directory")
            continue
        raise ValueError(
            f"trajectory {partition} root contains unexpected entry {path.name!r}"
        )
    games.sort(key=lambda item: item[0])
    if [index for index, _ in games] != list(range(len(games))):
        raise ValueError(f"trajectory {partition} game indices are not contiguous")
    return [path for _, path in games]


def reconstruct_trajectory_manifest(
    archive_root: Path,
    expected_identity: TacticalV3SemanticIdentity,
) -> dict[str, object]:
    """Derive the complete archive manifest from atomically published games."""

    _identity_wire(expected_identity)
    archive_root = Path(archive_root)
    _require_plain_directory(archive_root, "trajectory archive root")
    allowed = {"train", "validation", "manifest.json"}
    unexpected: set[str] = set()
    for path in archive_root.iterdir():
        if path.name in allowed:
            continue
        if _MANIFEST_TEMP_FILE.fullmatch(path.name) is not None:
            if not path.is_file() or _is_reparse(path):
                unexpected.add(path.name)
            continue
        unexpected.add(path.name)
    if unexpected:
        raise ValueError(
            "trajectory archive contains unexpected entries: "
            + ", ".join(sorted(unexpected))
        )
    stored_manifest = archive_root / "manifest.json"
    if os.path.lexists(stored_manifest) and (
        not stored_manifest.is_file() or _is_reparse(stored_manifest)
    ):
        raise ValueError("trajectory archive manifest must be a plain file")

    partitions: dict[str, object] = {}
    for partition in ("train", "validation"):
        games = []
        for path in _archive_entries(archive_root, partition):
            loaded = load_trajectory_game(path, expected_identity)
            if loaded.partition != partition:
                raise ValueError("trajectory directory partition does not match manifest")
            game_manifest = (path / "game.json").read_bytes()
            games.append({
                "game_index": loaded.game_index,
                "path": f"{partition}/{path.name}/game.json",
                "game_manifest_sha256": _sha256(game_manifest),
            })
        partitions[partition] = {
            "dataset_use": _PARTITION_USE[partition],
            "games": games,
        }
    return {
        "schema_version": 1,
        "kind": "tactical-v3-learner-trajectory-archive",
        "identity": _identity_wire(expected_identity),
        "partitions": partitions,
    }


def write_trajectory_manifest(
    archive_root: Path,
    expected_identity: TacticalV3SemanticIdentity,
) -> Path:
    """Atomically replace the derived archive index; game data remains immutable."""

    archive_root = Path(archive_root)
    manifest = reconstruct_trajectory_manifest(archive_root, expected_identity)
    path = archive_root / "manifest.json"
    atomic_write_bytes(path, _canonical_bytes(manifest))
    _fsync_directory(archive_root)
    return path


def validate_trajectory_manifest_snapshot(
    path: Path,
    archive_root: Path,
    expected_identity: TacticalV3SemanticIdentity,
) -> dict[str, object]:
    """Authenticate one immutable manifest snapshot and every game it names.

    Additional games may have been appended to the live archive after a best
    checkpoint was selected, so snapshot validation deliberately requires the
    referenced prefix rather than equality with the current physical inventory.
    """

    archive_root = Path(archive_root)
    _require_plain_directory(archive_root, "trajectory archive root")
    path = Path(path)
    if not path.is_file() or _is_reparse(path):
        raise ValueError("trajectory manifest snapshot must be a plain file")
    stored = _exact_mapping(
        _read_canonical_json(path.read_bytes(), "trajectory manifest snapshot"),
        _ARCHIVE_FIELDS,
        "trajectory manifest snapshot",
    )
    if (
        stored["schema_version"] != 1
        or stored["kind"] != "tactical-v3-learner-trajectory-archive"
    ):
        raise ValueError("trajectory archive manifest contract is unsupported")
    _parse_identity(stored["identity"], expected_identity)
    partitions = _exact_mapping(
        stored["partitions"], _ARCHIVE_PARTITIONS, "trajectory archive partitions"
    )
    for partition in ("train", "validation"):
        value = _exact_mapping(
            partitions[partition],
            _ARCHIVE_PARTITION_FIELDS,
            f"trajectory archive {partition} partition",
        )
        if value["dataset_use"] != _PARTITION_USE[partition]:
            raise ValueError("trajectory archive partition and dataset use do not match")
        games = value["games"]
        if type(games) is not list:
            raise TypeError("trajectory archive games must be a list")
        for index, raw in enumerate(games):
            entry = _exact_mapping(
                raw,
                _ARCHIVE_GAME_FIELDS,
                f"trajectory archive {partition} game {index}",
            )
            if entry["game_index"] != index:
                raise ValueError("trajectory archive game indices are not contiguous")
            expected_path = f"{partition}/game-{index:06d}/game.json"
            if entry["path"] != expected_path:
                raise ValueError("trajectory archive game path changed")
            expected_hash = _hash(
                entry["game_manifest_sha256"],
                "trajectory archive game manifest SHA-256",
            )
            game_manifest = archive_root / expected_path
            if not game_manifest.is_file() or _is_reparse(game_manifest):
                raise ValueError("trajectory manifest snapshot references a missing game")
            if _sha256(game_manifest.read_bytes()) != expected_hash:
                raise ValueError(
                    "trajectory manifest snapshot game hash does not match physical game"
                )
            loaded = load_trajectory_game(game_manifest.parent, expected_identity)
            if loaded.partition != partition or loaded.game_index != index:
                raise ValueError(
                    "trajectory manifest snapshot game identity does not match its path"
                )
    return dict(stored)


def load_trajectory_manifest(
    archive_root: Path,
    expected_identity: TacticalV3SemanticIdentity,
) -> dict[str, object]:
    """Authenticate a stored archive index against the physical game inventory."""

    archive_root = Path(archive_root)
    stored = validate_trajectory_manifest_snapshot(
        archive_root / "manifest.json", archive_root, expected_identity,
    )
    reconstructed = reconstruct_trajectory_manifest(
        archive_root, expected_identity
    )
    if stored != reconstructed:
        raise ValueError("trajectory archive manifest does not match physical games")
    return reconstructed


__all__ = [
    "ControllerProvenance",
    "TacticalV3TrajectoryGame",
    "TrajectoryDecisionRecord",
    "load_trajectory_game",
    "load_trajectory_manifest",
    "publish_trajectory_game",
    "reconstruct_trajectory_manifest",
    "validate_trajectory_manifest_snapshot",
    "write_trajectory_manifest",
]
