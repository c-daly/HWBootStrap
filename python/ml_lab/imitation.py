"""Restart-safe normalized storage for scripted tactical-v2 demonstrations."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import atomic_write_json

DATASET_SCHEMA_VERSION = 1
MAX_SHARD_ROWS = 4096
FORBIDDEN_RANGES = (range(10_000_000, 10_100_000), range(16_000_000, 16_000_100), range(17_000_000, 17_000_250))
STANDARD_PROFILE = "standard-3v3"
CONVERSION_PROFILES = frozenset({"conversion-3v1-near", "conversion-3v1-far", "conversion-2v1-near", "conversion-2v1-far", "conversion-1v1-near", "conversion-1v1-far"})
ACTION_KINDS = {"end_turn": 0, "move": 1, "attack": 2, "deploy": 3}


@dataclass(frozen=True)
class DecisionBatch:
    observations: np.ndarray
    packed_masks: np.ndarray
    actions: np.ndarray
    game_ids: np.ndarray
    decision_indices: np.ndarray
    seats: np.ndarray
    action_kinds: np.ndarray


@dataclass(frozen=True)
class DemonstrationGame:
    partition: str
    teacher: str
    teacher_parameters: Mapping[str, Any]
    opponent: str
    profile: str
    seed: int
    teacher_seat: int
    replay_path: str
    replay_hash: str
    outcome: str
    scenario_hash: str
    contract_hash: str
    encoding_hash: str

    @property
    def key(self) -> tuple[str, str, str, int, int]:
        return (self.partition, self.teacher, self.profile, self.seed, self.teacher_seat)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _safe_relative(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("dataset artifact path must be relative and contained")
    return candidate


def _value(row: Mapping[str, Any], lower: str, upper: str) -> Any:
    return row[lower] if lower in row else row[upper]


def _validate_game(game: DemonstrationGame, contract: EnvironmentContract) -> None:
    if game.partition not in {"train", "validation"}:
        raise ValueError("demonstration partition is unknown")
    if game.teacher not in {"greedy", "bounded-search"}:
        raise ValueError("demonstration teacher is unknown")
    if game.opponent != "random":
        raise ValueError("demonstration opponent must be random")
    if type(game.seed) is not int or any(game.seed in blocked for blocked in FORBIDDEN_RANGES):
        raise ValueError("demonstration seed is invalid or forbidden")
    if game.teacher == "greedy" and game.profile != STANDARD_PROFILE:
        raise ValueError("greedy demonstrations require the standard profile")
    if game.teacher == "bounded-search" and game.profile not in CONVERSION_PROFILES:
        raise ValueError("bounded-search demonstrations require a near/far conversion profile")
    expected_parameters: Mapping[str, Any] = {} if game.teacher == "greedy" else {"depth": 4, "expansion_budget": 512, "use_heuristic": True}
    if dict(game.teacher_parameters) != expected_parameters:
        raise ValueError("demonstration teacher parameters are not locked")
    if type(game.teacher_seat) is not int or game.teacher_seat not in {0, 1}:
        raise ValueError("demonstration teacher seat is invalid")
    if game.outcome not in {"win", "loss", "draw"} or not isinstance(game.teacher_parameters, Mapping):
        raise ValueError("demonstration provenance is invalid")
    if game.contract_hash != contract.contract_hash or game.encoding_hash != contract.encoding_hash:
        raise ValueError("demonstration contract or encoding hash does not match writer")
    if not all(_is_hash(value) for value in (game.replay_hash, game.scenario_hash, game.contract_hash, game.encoding_hash)):
        raise ValueError("demonstration provenance hash is invalid")
    _safe_relative(game.replay_path)
    allowed = range(11_000_000, 11_500_000) if game.teacher == "greedy" else range(11_500_000, 12_000_000)
    if game.partition == "train" and game.seed not in allowed:
        raise ValueError("training demonstration seed is outside its teacher namespace")
    if game.partition == "validation" and game.seed not in range(12_000_000, 12_100_000):
        raise ValueError("validation demonstration seed is outside its namespace")


def _source_ranges(games: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = {}
    for game in games:
        grouped.setdefault((game["partition"], game["teacher"], game["profile"]), []).append(game)
    return [{"partition": partition, "teacher": teacher, "profile": profile, "seed_start": min(item["seed"] for item in items), "seed_stop": max(item["seed"] for item in items), "game_count": len(items), "decision_count": sum(item["row_count"] for item in items)} for (partition, teacher, profile), items in sorted(grouped.items())]


def validate_decision(row: Mapping[str, Any], contract: EnvironmentContract) -> None:
    try:
        observation = np.asarray(_value(row, "observation", "Observation"), dtype=np.float32)
        mask = np.asarray(_value(row, "legal_mask", "LegalMask"), dtype=bool)
        action = int(_value(row, "action", "Action"))
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError("invalid demonstration decision") from exc
    if observation.shape != (contract.observation_size,) or not np.isfinite(observation).all():
        raise ValueError("invalid demonstration observation")
    if mask.shape != (contract.action_size,) or not 0 <= action < contract.action_size or not mask[action]:
        raise ValueError("demonstration action is not legal")
    if _value(row, "seat", "Seat") not in {0, 1}:
        raise ValueError("demonstration seat is invalid")


def _action_kind(row: Mapping[str, Any]) -> int:
    if "action_kind" in row:
        value = row["action_kind"]
        if type(value) is not int or value not in ACTION_KINDS.values():
            raise ValueError("demonstration action kind is invalid")
        return value
    command = row.get("command", row.get("Command"))
    if not isinstance(command, Mapping) or command.get("Kind") not in ACTION_KINDS:
        raise ValueError("demonstration command kind is invalid")
    return ACTION_KINDS[command["Kind"]]


def _atomic_npz(path: Path, batch: DecisionBatch) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **asdict(batch))
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            for record in records:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _git_metadata(dataset_root: Path) -> tuple[str, bool]:
    try:
        repository = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], cwd=dataset_root, text=True, stderr=subprocess.DEVNULL).strip()).resolve()
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repository, text=True, stderr=subprocess.DEVNULL).strip()
        excluded = dataset_root.resolve().relative_to(repository).as_posix() + "/"
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repository, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True).stdout.splitlines()
        changed = []
        for line in status:
            path = line[3:].replace("\\", "/")
            if " -> " in path: path = path.rsplit(" -> ", 1)[-1]
            if not path.startswith(excluded): changed.append(path)
        return revision, bool(changed)
    except (OSError, ValueError, subprocess.SubprocessError):
        return "unknown", True


class DemonstrationWriter:
    """Commits fully-hashed game shards before atomically publishing their manifest."""

    def __init__(self, root: Path, contract: EnvironmentContract, shard_rows: int) -> None:
        self.root, self.contract, self.shard_rows = Path(root), contract, shard_rows
        self.fail_before_manifest_replace = False
        self._manifest: dict[str, Any] = {}
        self._games: list[dict[str, Any]] = []

    @classmethod
    def create(cls, root: Path, *, contract: EnvironmentContract, shard_rows: int = MAX_SHARD_ROWS) -> "DemonstrationWriter":
        if not 1 <= shard_rows <= MAX_SHARD_ROWS:
            raise ValueError("shard_rows must be within 1..4096")
        writer = cls(root, contract, shard_rows)
        writer.root.mkdir(parents=True, exist_ok=True); (writer.root / "shards").mkdir(exist_ok=True)
        writer._recover()
        return writer

    def _new_manifest(self) -> dict[str, Any]:
        revision, dirty = _git_metadata(self.root)
        return {"schema_version": DATASET_SCHEMA_VERSION, "code_revision": revision, "dirty": dirty, "contract_hash": self.contract.contract_hash, "encoding_hash": self.contract.encoding_hash, "source_ranges": [], "decision_count": 0, "game_count": 0, "shards": [], "replays": []}

    def _recover(self) -> None:
        for temporary in self.root.rglob(".*.tmp"):
            temporary.unlink(missing_ok=True)
        manifest_path, games_path = self.root / "manifest.json", self.root / "games.jsonl"
        if not manifest_path.exists():
            for orphan in (self.root / "shards").glob("*.npz"):
                orphan.unlink(missing_ok=True)
            games_path.unlink(missing_ok=True); self._manifest, self._games = self._new_manifest(), []
            return
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        required = {"schema_version", "code_revision", "dirty", "contract_hash", "encoding_hash", "source_ranges", "decision_count", "game_count", "shards", "replays"}
        if set(manifest) != required or manifest["schema_version"] != DATASET_SCHEMA_VERSION or manifest["contract_hash"] != self.contract.contract_hash or manifest["encoding_hash"] != self.contract.encoding_hash:
            raise ValueError("dataset manifest is incompatible")
        games = [json.loads(line) for line in games_path.read_text(encoding="utf-8").splitlines()] if games_path.exists() else []
        if len(games) < manifest["game_count"]:
            raise ValueError("manifest owns games missing from games.jsonl")
        self._games = games[:manifest["game_count"]]
        if len(games) != len(self._games): _atomic_jsonl(games_path, self._games)
        owned_shards = set(); rows = 0
        for item in manifest["shards"]:
            if set(item) != {"path", "sha256", "rows", "game_id"} or item["path"] in owned_shards or not _is_hash(item["sha256"]) or not 1 <= item["rows"] <= MAX_SHARD_ROWS:
                raise ValueError("dataset shard manifest integrity is invalid")
            path = self.root / _safe_relative(item["path"])
            if not path.is_file() or sha256_file(path) != item["sha256"]: raise ValueError("dataset shard hash mismatch")
            owned_shards.add(item["path"]); rows += item["rows"]
        if rows != manifest["decision_count"]: raise ValueError("dataset manifest decision count does not match shards")
        for item in manifest["replays"]:
            path = self.root / _safe_relative(item["path"])
            if set(item) != {"path", "sha256"} or not _is_hash(item["sha256"]) or not path.is_file() or sha256_file(path) != item["sha256"]: raise ValueError("dataset replay hash mismatch")
        for orphan in (self.root / "shards").glob("*.npz"):
            if orphan.relative_to(self.root).as_posix() not in owned_shards: orphan.unlink(missing_ok=True)
        self._manifest = manifest
        self._validate_existing_games()
        self._validate_physical_rows()

    def _validate_existing_games(self) -> None:
        keys: set[tuple[str, str, str, int, int]] = set(); seeds: dict[int, str] = {}
        for index, record in enumerate(self._games):
            game = DemonstrationGame(**{field: record[field] for field in DemonstrationGame.__dataclass_fields__})
            _validate_game(game, self.contract)
            if game.key in keys: raise ValueError("duplicate completed game key")
            if game.seed in seeds and seeds[game.seed] != game.partition: raise ValueError("seed reuse across partitions")
            if record.get("game_id") != index or type(record.get("row_count")) is not int or record["row_count"] < 1: raise ValueError("dataset game record is invalid")
            keys.add(game.key); seeds[game.seed] = game.partition

    def _validate_physical_rows(self) -> None:
        if self._manifest["source_ranges"] != _source_ranges(self._games): raise ValueError("dataset source ranges do not match games")
        replays = {item["path"]: item["sha256"] for item in self._manifest["replays"]}
        if len(replays) != len(self._manifest["replays"]): raise ValueError("dataset replay ownership is duplicated")
        by_game: dict[int, list[Mapping[str, Any]]] = {}
        for shard in self._manifest["shards"]: by_game.setdefault(shard["game_id"], []).append(shard)
        cursor = 0; required = {"observations", "packed_masks", "actions", "game_ids", "decision_indices", "seats", "action_kinds"}
        for game_id, game in enumerate(self._games):
            if game["row_start"] != cursor or game["row_stop"] != cursor + game["row_count"]: raise ValueError("dataset game row spans are inconsistent")
            cursor = game["row_stop"]
            if replays.get(game["replay_path"]) != game["replay_hash"]: raise ValueError("dataset replay ownership does not match game")
            indices: list[int] = []; rows = 0
            for shard in by_game.get(game_id, []):
                try:
                    with np.load(self.root / shard["path"], allow_pickle=False) as data:
                        if set(data.files) != required: raise ValueError("dataset shard fields are invalid")
                        obs, masks, actions = data["observations"], data["packed_masks"], data["actions"]
                        ids, decision_indices, seats, kinds = data["game_ids"], data["decision_indices"], data["seats"], data["action_kinds"]; count = len(actions)
                        if count != shard["rows"] or obs.dtype != np.float32 or obs.shape != (count, self.contract.observation_size) or not np.isfinite(obs).all() or masks.dtype != np.uint8 or masks.shape != (count, (self.contract.action_size + 7) // 8) or actions.dtype != np.int64 or ids.dtype != np.int64 or decision_indices.dtype != np.int32 or seats.dtype != np.uint8 or kinds.dtype != np.uint8 or any(value.shape != (count,) for value in (actions, ids, decision_indices, seats, kinds)): raise ValueError("dataset shard physical shape or dtype is invalid")
                        legal = np.unpackbits(masks, axis=1, bitorder="little")[:, :self.contract.action_size]
                        if np.any(actions < 0) or np.any(actions >= self.contract.action_size) or not np.all(legal[np.arange(count), actions]) or not np.all(ids == game_id) or not np.all(seats == game["teacher_seat"]) or np.any(kinds > max(ACTION_KINDS.values())): raise ValueError("dataset shard physical values are invalid")
                        indices.extend(int(value) for value in decision_indices); rows += count
                except (OSError, ValueError) as exc:
                    raise ValueError("dataset shard physical validation failed") from exc
            if rows != game["row_count"] or indices != list(range(rows)): raise ValueError("dataset shard rows do not exactly own game")
        if cursor != self._manifest["decision_count"] or set(by_game) != set(range(len(self._games))) or len(replays) != len(self._games): raise ValueError("dataset artifact ownership counts are inconsistent")

    def retained_decision_count(self, teacher: str) -> int:
        return sum(item["row_count"] for item in self._games if item["teacher"] == teacher)

    def completed_keys(self) -> set[tuple[str, str, str, int, int]]:
        return {(item["partition"], item["teacher"], item["profile"], item["seed"], item["teacher_seat"]) for item in self._games}

    def append_game(self, game: DemonstrationGame, decisions: Sequence[Mapping[str, Any]]) -> None:
        if any(item["seed"] == game.seed and item["partition"] != game.partition for item in self._games): raise ValueError("seed reuse across partitions")
        _validate_game(game, self.contract)
        if game.key in self.completed_keys(): raise ValueError("duplicate completed game key")
        replay = self.root / _safe_relative(game.replay_path)
        if not replay.is_file() or sha256_file(replay) != game.replay_hash: raise ValueError("demonstration replay hash mismatch")
        if not decisions: raise ValueError("complete demonstration game has no teacher decisions")
        for expected, row in enumerate(decisions):
            validate_decision(row, self.contract)
            if _value(row, "seat", "Seat") != game.teacher_seat: raise ValueError("demonstration row is not from the teacher seat")
            if row.get("decision_index", expected) != expected: raise ValueError("demonstration decision indices contain a gap")
        game_id, shards = len(self._games), []
        for chunk_number, start in enumerate(range(0, len(decisions), self.shard_rows)):
            chunk = decisions[start:start + self.shard_rows]
            batch = DecisionBatch(np.asarray([_value(row, "observation", "Observation") for row in chunk], dtype=np.float32), np.packbits(np.asarray([_value(row, "legal_mask", "LegalMask") for row in chunk], dtype=np.uint8), axis=1, bitorder="little"), np.asarray([_value(row, "action", "Action") for row in chunk], dtype=np.int64), np.full(len(chunk), game_id, dtype=np.int64), np.asarray([row.get("decision_index", start + offset) for offset, row in enumerate(chunk)], dtype=np.int32), np.asarray([_value(row, "seat", "Seat") for row in chunk], dtype=np.uint8), np.asarray([_action_kind(row) for row in chunk], dtype=np.uint8))
            relative = f"shards/game-{game_id:08d}-{chunk_number:03d}.npz"; _atomic_npz(self.root / relative, batch)
            shards.append({"path": relative, "sha256": sha256_file(self.root / relative), "rows": len(chunk), "game_id": game_id})
        record = {**asdict(game), "teacher_parameters": dict(game.teacher_parameters), "game_id": game_id, "row_count": len(decisions), "row_start": self._manifest["decision_count"], "row_stop": self._manifest["decision_count"] + len(decisions)}
        next_games = [*self._games, record]
        next_manifest = {**self._manifest, "decision_count": self._manifest["decision_count"] + len(decisions), "game_count": game_id + 1, "shards": [*self._manifest["shards"], *shards], "replays": [*self._manifest["replays"], {"path": game.replay_path, "sha256": game.replay_hash}], "source_ranges": _source_ranges(next_games)}
        # An interruption here leaves only unowned data. Recovery truncates games to the old manifest and removes orphan shards.
        _atomic_jsonl(self.root / "games.jsonl", next_games)
        if self.fail_before_manifest_replace: raise RuntimeError("simulated interruption before manifest replacement")
        atomic_write_json(self.root / "manifest.json", next_manifest)
        self._games, self._manifest = next_games, next_manifest

    def close(self) -> None:
        return None

# Reader and training primitives deliberately repeat writer validation: a loader must
# not trust a manifest merely because it was once produced by DemonstrationWriter.
from collections import OrderedDict
from enum import Enum
from types import MappingProxyType


class Source(Enum):
    GREEDY_STANDARD = "greedy_standard"
    SEARCH_CONVERSION = "search_conversion"


@dataclass(frozen=True)
class _ShardDescriptor:
    path: str
    sha256: str
    rows: int
    game_id: int


@dataclass(frozen=True)
class ImitationBatch:
    observations: np.ndarray
    legal_masks: np.ndarray
    actions: np.ndarray
    game_ids: np.ndarray
    decision_indices: np.ndarray
    sources: np.ndarray
    profiles: np.ndarray
    seats: np.ndarray
    action_kinds: np.ndarray
    partitions: np.ndarray


class _DecodedShardCache:
    """A tiny LRU cache; rows returned to callers are always copied."""

    def __init__(self) -> None:
        self._entries: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def get(
        self,
        root: Path,
        descriptor: _ShardDescriptor,
        game: Mapping[str, Any],
        contract: EnvironmentContract,
    ) -> dict[str, np.ndarray]:
        key = id(descriptor)
        cached = self._entries.pop(key, None)
        if cached is not None:
            self._entries[key] = cached
            return cached
        path = root / descriptor.path
        if sha256_file(path) != descriptor.sha256:
            raise ValueError("dataset shard SHA-256 does not match manifest")
        loaded = _read_shard(path, descriptor, game, contract)
        self._entries[key] = loaded
        while len(self._entries) > 2:
            self._entries.popitem(last=False)
        return loaded


def _immutable_index(index: dict[str, dict[Source, dict[str, dict[int, dict[int, list[tuple[int, int]]]]]]]) -> Mapping[str, Any]:
    return MappingProxyType({
        partition: MappingProxyType({
            source: MappingProxyType({
                profile: MappingProxyType({
                    seat: MappingProxyType({kind: tuple(rows) for kind, rows in kinds.items()})
                    for seat, kinds in seats.items()
                })
                for profile, seats in profiles.items()
            })
            for source, profiles in sources.items()
        })
        for partition, sources in index.items()
    })


@dataclass(frozen=True)
class ImitationDataset:
    root: Path
    contract: EnvironmentContract
    games: tuple[Mapping[str, Any], ...]
    shards: tuple[_ShardDescriptor, ...]
    index: Mapping[str, Any]
    _cache: _DecodedShardCache

    def _row_data(self, refs: Sequence[tuple[int, int]]) -> dict[str, np.ndarray]:
        selected: dict[str, list[np.ndarray]] = {name: [] for name in ("observations", "packed_masks", "actions", "game_ids", "decision_indices", "seats", "action_kinds")}
        for shard_index, local_row in refs:
            descriptor = self.shards[shard_index]
            arrays = self._cache.get(self.root, descriptor, self.games[descriptor.game_id], self.contract)
            for name in selected:
                selected[name].append(arrays[name][local_row].copy())
        return {name: np.asarray(values) for name, values in selected.items()}


def _source_for_game(game: Mapping[str, Any]) -> Source:
    if game["teacher"] == "greedy" and game["profile"] == STANDARD_PROFILE:
        return Source.GREEDY_STANDARD
    if game["teacher"] == "bounded-search" and game["profile"] in CONVERSION_PROFILES:
        return Source.SEARCH_CONVERSION
    raise ValueError("demonstration source/profile is invalid")


def _scan_shard_index(path: Path, descriptor: _ShardDescriptor, game: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Read only metadata required for immutable row references during construction."""
    required = {"observations", "packed_masks", "actions", "game_ids", "decision_indices", "seats", "action_kinds"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                raise ValueError("dataset shard fields are invalid")
            game_ids = data["game_ids"]
            decision_indices = data["decision_indices"]
            seats = data["seats"]
            action_kinds = data["action_kinds"]
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("dataset shard index metadata is invalid") from exc
    count = descriptor.rows
    if game_ids.dtype != np.int64 or decision_indices.dtype != np.int32 or seats.dtype != np.uint8 or action_kinds.dtype != np.uint8 or any(value.shape != (count,) for value in (game_ids, decision_indices, seats, action_kinds)):
        raise ValueError("dataset shard index metadata shape or dtype is invalid")
    if not np.all(game_ids == descriptor.game_id) or not np.all(seats == game["teacher_seat"]) or np.any(action_kinds > max(ACTION_KINDS.values())):
        raise ValueError("dataset shard index metadata values are invalid")
    return decision_indices, action_kinds

def _read_shard(path: Path, descriptor: _ShardDescriptor, game: Mapping[str, Any], contract: EnvironmentContract) -> dict[str, np.ndarray]:
    required = {"observations", "packed_masks", "actions", "game_ids", "decision_indices", "seats", "action_kinds"}
    try:
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != required:
                raise ValueError("dataset shard fields are invalid")
            arrays = {name: data[name].copy() for name in required}
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("dataset shard physical structure is invalid") from exc
    observations, masks, actions = arrays["observations"], arrays["packed_masks"], arrays["actions"]
    game_ids, indices, seats, kinds = arrays["game_ids"], arrays["decision_indices"], arrays["seats"], arrays["action_kinds"]
    count = len(actions)
    if count != descriptor.rows or observations.dtype != np.float32 or observations.shape != (count, contract.observation_size) or not np.isfinite(observations).all():
        raise ValueError("dataset shard physical shape, dtype, or values are invalid")
    if masks.dtype != np.uint8 or masks.shape != (count, (contract.action_size + 7) // 8):
        raise ValueError("dataset shard physical shape or dtype is invalid")
    if actions.dtype != np.int64 or game_ids.dtype != np.int64 or indices.dtype != np.int32 or seats.dtype != np.uint8 or kinds.dtype != np.uint8 or any(value.shape != (count,) for value in (actions, game_ids, indices, seats, kinds)):
        raise ValueError("dataset shard physical shape or dtype is invalid")
    legal = np.unpackbits(masks, axis=1, count=contract.action_size, bitorder="little").astype(bool, copy=False)
    if np.any(actions < 0) or np.any(actions >= contract.action_size) or not np.all(legal[np.arange(count), actions]):
        raise ValueError("dataset shard contains an illegal action")
    if not np.all(game_ids == descriptor.game_id) or not np.all(seats == game["teacher_seat"]) or np.any(kinds > max(ACTION_KINDS.values())):
        raise ValueError("dataset shard physical values are invalid")
    return arrays


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("dataset metadata is unreadable") from exc


def _validate_manifest(manifest: Mapping[str, Any], contract: EnvironmentContract) -> None:
    required = {"schema_version", "code_revision", "dirty", "contract_hash", "encoding_hash", "source_ranges", "decision_count", "game_count", "shards", "replays"}
    if set(manifest) != required or manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("dataset manifest schema is invalid")
    if manifest.get("contract_hash") != contract.contract_hash or manifest.get("encoding_hash") != contract.encoding_hash:
        raise ContractMismatch("dataset contract or encoding hash does not match expected contract")
    if not isinstance(manifest["code_revision"], str) or type(manifest["dirty"]) is not bool or type(manifest["decision_count"]) is not int or type(manifest["game_count"]) is not int or manifest["decision_count"] < 0 or manifest["game_count"] < 0 or not isinstance(manifest["source_ranges"], list) or not isinstance(manifest["shards"], list) or not isinstance(manifest["replays"], list):
        raise ValueError("dataset manifest values are invalid")


def load_imitation_dataset(root: Path, expected_contract: EnvironmentContract) -> ImitationDataset:
    """Validate metadata eagerly; validate payload arrays when their shard enters the cache."""
    root = Path(root)
    manifest = _load_json(root / "manifest.json")
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be an object")
    _validate_manifest(manifest, expected_contract)
    try:
        game_records = [json.loads(line) for line in (root / "games.jsonl").read_text(encoding="utf-8").splitlines()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("dataset games metadata is unreadable") from exc
    if len(game_records) != manifest["game_count"]:
        raise ValueError("dataset game count does not match manifest")
    fields = set(DemonstrationGame.__dataclass_fields__) | {"game_id", "row_count", "row_start", "row_stop"}
    games: list[Mapping[str, Any]] = []
    seen_keys: set[tuple[str, str, str, int, int]] = set()
    seeds: dict[int, str] = {}
    scenarios: set[str] = set()
    cursor = 0
    for game_id, record in enumerate(game_records):
        if not isinstance(record, Mapping) or set(record) != fields:
            raise ValueError("dataset game record is invalid")
        if record.get("game_id") != game_id or type(record.get("row_count")) is not int or record["row_count"] < 1 or record.get("row_start") != cursor or record.get("row_stop") != cursor + record["row_count"]:
            raise ValueError("dataset game spans are inconsistent")
        if record.get("seed") in seeds and seeds[record["seed"]] != record.get("partition"):
            raise ValueError("dataset seed is shared across a partition")
        if record.get("contract_hash") != expected_contract.contract_hash or record.get("encoding_hash") != expected_contract.encoding_hash:
            raise ContractMismatch("game contract or encoding hash does not match expected contract")
        try:
            game = DemonstrationGame(**{name: record[name] for name in DemonstrationGame.__dataclass_fields__})
            _validate_game(game, expected_contract)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("dataset game provenance is invalid") from exc
        if game.key in seen_keys:
            raise ValueError("dataset game key is duplicated")
        seen_keys.add(game.key); seeds[game.seed] = game.partition; scenarios.add(game.scenario_hash)
        games.append(MappingProxyType(dict(record))); cursor = record["row_stop"]
    if cursor != manifest["decision_count"] or len(scenarios) != 1:
        raise ValueError("dataset scenario or decision count is invalid")
    replay_hashes: dict[str, str] = {}
    for replay in manifest["replays"]:
        if not isinstance(replay, Mapping) or set(replay) != {"path", "sha256"} or not _is_hash(replay.get("sha256")):
            raise ValueError("dataset replay manifest is invalid")
        relative = _safe_relative(replay["path"])
        if replay["path"] in replay_hashes or not (root / relative).is_file() or sha256_file(root / relative) != replay["sha256"]:
            raise ValueError("dataset replay SHA-256 does not match manifest")
        replay_hashes[replay["path"]] = replay["sha256"]
    if len(replay_hashes) != len(games) or any(replay_hashes.get(game["replay_path"]) != game["replay_hash"] for game in games):
        raise ValueError("dataset replay provenance is invalid")
    descriptors: list[_ShardDescriptor] = []
    seen_paths: set[str] = set()
    by_game: dict[int, list[tuple[int, _ShardDescriptor]]] = {}
    for item in manifest["shards"]:
        if not isinstance(item, Mapping) or set(item) != {"path", "sha256", "rows", "game_id"} or item.get("path") in seen_paths or not _is_hash(item.get("sha256")) or type(item.get("rows")) is not int or not 1 <= item["rows"] <= MAX_SHARD_ROWS or type(item.get("game_id")) is not int or item["game_id"] not in range(len(games)):
            raise ValueError("dataset shard manifest is invalid")
        relative = _safe_relative(item["path"])
        if not (root / relative).is_file() or sha256_file(root / relative) != item["sha256"]:
            raise ValueError("dataset shard SHA-256 does not match manifest")
        descriptor = _ShardDescriptor(item["path"], item["sha256"], item["rows"], item["game_id"])
        by_game.setdefault(item["game_id"], []).append((len(descriptors), descriptor)); descriptors.append(descriptor); seen_paths.add(item["path"])
    actual_shards = {path.relative_to(root).as_posix() for path in (root / "shards").rglob("*.npz")} if (root / "shards").is_dir() else set()
    if actual_shards != seen_paths or set(by_game) != set(range(len(games))):
        raise ValueError("dataset shard ownership is invalid")
    index: dict[str, dict[Source, dict[str, dict[int, dict[int, list[tuple[int, int]]]]]]] = {}
    for game_id, game in enumerate(games):
        sequence: list[int] = []
        rows = 0
        source = _source_for_game(game)
        for shard_index, descriptor in by_game[game_id]:
            decision_indices, action_kinds = _scan_shard_index(root / descriptor.path, descriptor, game)
            sequence.extend(int(value) for value in decision_indices)
            for local_row, kind in enumerate(action_kinds):
                index.setdefault(game["partition"], {}).setdefault(source, {}).setdefault(game["profile"], {}).setdefault(int(game["teacher_seat"]), {}).setdefault(int(kind), []).append((shard_index, local_row))
            rows += descriptor.rows
        if rows != game["row_count"] or sequence != list(range(rows)):
            raise ValueError("dataset decision indices are not contiguous")
    if manifest["source_ranges"] != _source_ranges(games):
        raise ValueError("dataset source ranges do not match games")
    return ImitationDataset(root.resolve(), expected_contract, tuple(games), tuple(descriptors), _immutable_index(index), _DecodedShardCache())


class _StratumCycler:
    def __init__(self, strata: Sequence[Sequence[tuple[int, int]]], rng: np.random.Generator) -> None:
        if not strata or any(not rows for rows in strata):
            raise ValueError("sampler source contains an empty stratum")
        self._rows = [tuple(rows[index] for index in rng.permutation(len(rows))) for rows in strata]
        self._strata = list(rng.permutation(len(self._rows)))
        self._row_positions = [0] * len(self._rows)
        self._stratum_position = 0

    def take(self, count: int) -> list[tuple[int, int]]:
        result: list[tuple[int, int]] = []
        for _ in range(count):
            group = self._strata[self._stratum_position % len(self._strata)]
            self._stratum_position += 1
            row = self._rows[group][self._row_positions[group] % len(self._rows[group])]
            self._row_positions[group] += 1
            result.append(row)
        return result


def _source_strata(dataset: ImitationDataset, partition: str, source: Source) -> list[tuple[tuple[int, int], ...]]:
    try:
        profiles = dataset.index[partition][source]
    except KeyError as exc:
        raise ValueError(f"sampler has no {source.value} rows in {partition} partition") from exc
    return [rows for profile in sorted(profiles) for seat in sorted(profiles[profile]) for kind in sorted(profiles[profile][seat]) for rows in (profiles[profile][seat][kind],)]


class StratifiedDecisionSampler:
    """Partition-scoped, seeded strata cycling; undersized strata cycle deterministically."""

    def __init__(self, dataset: ImitationDataset, batch_size: int = 1, standard_fraction: float = 0.70, seed: int = 0, partition: str = "train") -> None:
        if type(batch_size) is not int or batch_size < 1 or partition not in {"train", "validation"} or not isinstance(standard_fraction, (int, float)) or not 0.0 < standard_fraction < 1.0:
            raise ValueError("sampler configuration is invalid")
        self.dataset, self.batch_size, self.standard_fraction, self.partition = dataset, batch_size, float(standard_fraction), partition
        self._rng = np.random.default_rng(seed)
        self._cyclers = {Source.GREEDY_STANDARD: _StratumCycler(_source_strata(dataset, partition, Source.GREEDY_STANDARD), self._rng), Source.SEARCH_CONVERSION: _StratumCycler(_source_strata(dataset, partition, Source.SEARCH_CONVERSION), self._rng)}
        self._residual = 0.0

    def next_batch(self) -> ImitationBatch:
        target = self.batch_size * self.standard_fraction + self._residual
        standard_count = int(np.floor(target + 1e-12))
        self._residual = target - standard_count
        refs_and_sources = [(ref, Source.GREEDY_STANDARD) for ref in self._cyclers[Source.GREEDY_STANDARD].take(standard_count)]
        refs_and_sources += [(ref, Source.SEARCH_CONVERSION) for ref in self._cyclers[Source.SEARCH_CONVERSION].take(self.batch_size - standard_count)]
        order = self._rng.permutation(len(refs_and_sources))
        refs_and_sources = [refs_and_sources[int(index)] for index in order]
        rows = self.dataset._row_data([ref for ref, _source in refs_and_sources])
        legal_masks = np.unpackbits(rows["packed_masks"], axis=1, count=self.dataset.contract.action_size, bitorder="little").astype(bool, copy=False)
        if not np.all(legal_masks[np.arange(len(rows["actions"])), rows["actions"]]):
            raise ValueError("selected teacher action is masked")
        metadata = [self.dataset.games[int(game_id)] for game_id in rows["game_ids"]]
        return ImitationBatch(rows["observations"].copy(), legal_masks.copy(), rows["actions"].copy(), rows["game_ids"].copy(), rows["decision_indices"].copy(), np.asarray([source for _ref, source in refs_and_sources], dtype=object), np.asarray([game["profile"] for game in metadata], dtype=object), rows["seats"].copy(), rows["action_kinds"].copy(), np.asarray([game["partition"] for game in metadata], dtype=object))


def masked_cross_entropy(logits: Any, legal_masks: Any, actions: Any) -> Any:
    """Cross-entropy over legal actions only, rejecting malformed teacher data."""
    import torch
    import torch.nn.functional as functional

    if not all(isinstance(value, torch.Tensor) for value in (logits, legal_masks, actions)):
        raise TypeError("logits, legal_masks, and actions must be tensors")
    if logits.ndim != 2 or legal_masks.shape != logits.shape:
        raise ValueError("logits and masks must have identical shape")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("logits must be finite floating-point values")
    if legal_masks.dtype is not torch.bool or actions.dtype != torch.int64 or actions.ndim != 1 or actions.shape[0] != logits.shape[0] or logits.shape[0] == 0:
        raise ValueError("mask or actions shape/dtype is invalid")
    if legal_masks.device != logits.device or actions.device != logits.device or not legal_masks.any(dim=1).all():
        raise ValueError("mask or actions device/values are invalid")
    if torch.any(actions < 0) or torch.any(actions >= logits.shape[1]):
        raise ValueError("teacher action is out of bounds")
    if not legal_masks.gather(1, actions[:, None]).all():
        raise ValueError("teacher action is masked")
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
    return functional.cross_entropy(masked_logits, actions)
