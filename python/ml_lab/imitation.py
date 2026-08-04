"""Restart-safe normalized storage for scripted tactical-v2 demonstrations."""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import chain
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Callable
from typing import Any

import numpy as np

from .contracts import ContractMismatch, EnvironmentContract
from .io import atomic_write_json
from .scenarios import ResolvedScenario

DATASET_SCHEMA_VERSION = 1
MAX_SHARD_ROWS = 4096
FORBIDDEN_RANGES = (range(10_000_000, 10_100_000), range(16_000_000, 16_000_100), range(17_000_000, 17_000_250))
STANDARD_PROFILE = "standard-3v3"
CONVERSION_PROFILES = frozenset({"conversion-3v1-near", "conversion-3v1-far", "conversion-2v1-near", "conversion-2v1-far", "conversion-1v1-near", "conversion-1v1-far"})
ACTION_KINDS = {"end_turn": 0, "move": 1, "attack": 2, "deploy": 3}
END_TURN_ACTION = 0


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
    DAGGER_TARGETED = "dagger_targeted"


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
        if not refs:
            raise ValueError("row gathering requires at least one reference")
        grouped: dict[int, list[tuple[int, int]]] = {}
        for destination, (shard_index, local_row) in enumerate(refs):
            if shard_index not in range(len(self.shards)):
                raise ValueError("row reference shard is invalid")
            descriptor = self.shards[shard_index]
            if local_row not in range(descriptor.rows):
                raise ValueError("row reference offset is invalid")
            grouped.setdefault(shard_index, []).append((destination, local_row))

        selected: dict[str, np.ndarray] | None = None
        for shard_index, placements in grouped.items():
            descriptor = self.shards[shard_index]
            arrays = self._cache.get(
                self.root, descriptor, self.games[descriptor.game_id], self.contract,
            )
            destinations = np.asarray([item[0] for item in placements], dtype=np.int64)
            local_rows = np.asarray([item[1] for item in placements], dtype=np.int64)
            if selected is None:
                selected = {
                    name: np.empty(
                        (len(refs), *values.shape[1:]), dtype=values.dtype,
                    )
                    for name, values in arrays.items()
                }
            for name in selected:
                selected[name][destinations] = arrays[name][local_rows]

        assert selected is not None
        return selected

@dataclass(frozen=True)
class MaterializedImitationPartition:
    partition: str
    batch: ImitationBatch
    offsets: Mapping[tuple[Any, ...], int]

    def __post_init__(self) -> None:
        if self.partition not in {"train", "validation"}:
            raise ValueError("materialized partition name is invalid")
        count = len(self.batch.actions)
        if count < 1 or len(self.offsets) != count:
            raise ValueError("materialized partition reference map is incomplete")
        if set(self.batch.partitions) != {self.partition}:
            raise ValueError("materialized partition metadata differs")
        if set(self.offsets.values()) != set(range(count)):
            raise ValueError("materialized partition offsets are invalid")
        if any(
            not isinstance(values, np.ndarray) or len(values) != count
            for values in (
                getattr(self.batch, name)
                for name in ImitationBatch.__dataclass_fields__
            )
        ):
            raise ValueError("materialized partition batch fields are inconsistent")
        frozen_offsets = MappingProxyType(dict(self.offsets))
        object.__setattr__(self, "offsets", frozen_offsets)
        for name in ImitationBatch.__dataclass_fields__:
            getattr(self.batch, name).setflags(write=False)


def _freeze_source_fractions(
    value: Mapping[Source, float],
) -> Mapping[Source, float]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("source fractions must be a non-empty ordered mapping")
    frozen: OrderedDict[Source, float] = OrderedDict()
    for source, fraction in value.items():
        if (
            not isinstance(source, Source)
            or isinstance(fraction, bool)
            or not isinstance(fraction, (int, float))
            or not math.isfinite(float(fraction))
            or float(fraction) <= 0.0
        ):
            raise ValueError("source fractions must contain finite positive values")
        frozen[source] = float(fraction)
    if not math.isclose(sum(frozen.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("source fractions must sum to 1.0")
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class ActorSupervisionCorpus:
    training: MaterializedImitationPartition
    validation: MaterializedImitationPartition
    source_fractions: Mapping[Source, float]
    identity: Mapping[str, Any]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.training, MaterializedImitationPartition)
            or self.training.partition != "train"
            or not isinstance(self.validation, MaterializedImitationPartition)
            or self.validation.partition != "validation"
        ):
            raise ValueError("actor supervision corpus partitions are invalid")
        fractions = _freeze_source_fractions(self.source_fractions)
        if set(self.training.batch.sources) != set(fractions):
            raise ValueError("actor supervision training sources differ from mixture")
        if set(self.training.offsets) & set(self.validation.offsets):
            raise ValueError("actor supervision train and validation identities overlap")
        if (
            not isinstance(self.identity, Mapping)
            or not self.identity
            or any(not isinstance(key, str) for key in self.identity)
        ):
            raise ValueError("actor supervision corpus identity is invalid")
        source_mixture = tuple(
            (source.value, fraction) for source, fraction in fractions.items()
        )
        identity = dict(self.identity)
        if (
            "source_mixture" in identity
            and tuple(tuple(item) for item in identity["source_mixture"])
            != source_mixture
        ):
            raise ValueError(
                "actor supervision corpus identity source mixture differs"
            )
        identity["source_mixture"] = source_mixture
        object.__setattr__(self, "source_fractions", fractions)
        object.__setattr__(self, "identity", MappingProxyType(identity))


def training_rows_as_validation(dataset: ImitationDataset) -> ImitationDataset:
    """Return a smoke-only view whose validation index aliases physical training rows."""

    if not isinstance(dataset, ImitationDataset):
        raise TypeError("dataset must be a loaded ImitationDataset")
    try:
        training_index = dataset.index["train"]
    except KeyError as exc:
        raise ValueError("dataset has no training rows to reuse as validation") from exc
    index = dict(dataset.index)
    index["validation"] = training_index
    return ImitationDataset(
        dataset.root, dataset.contract, dataset.games, dataset.shards,
        MappingProxyType(index), dataset._cache,
    )


def audit_imitation_dataset(dataset: ImitationDataset) -> dict[str, int]:
    """Reopen every physical row and replay and return smoke-gate counts."""

    if not isinstance(dataset, ImitationDataset):
        raise TypeError("dataset must be a loaded ImitationDataset")
    refs = [
        (shard_index, row)
        for shard_index, descriptor in enumerate(dataset.shards)
        for row in range(descriptor.rows)
    ]
    if not refs:
        raise ValueError("dataset audit requires teacher labels")
    rows = dataset._row_data(refs)
    legal_masks = np.unpackbits(
        rows["packed_masks"],
        axis=1,
        count=dataset.contract.action_size,
        bitorder="little",
    ).astype(bool, copy=False)
    repacked = np.packbits(legal_masks, axis=1, bitorder="little")
    masked = int(np.count_nonzero(
        ~legal_masks[np.arange(len(rows["actions"])), rows["actions"]]
    ))
    round_trip = int(np.count_nonzero(
        np.any(repacked != rows["packed_masks"], axis=1)
    ))
    replay_mismatches = sum(
        sha256_file(dataset.root / game["replay_path"]) != game["replay_hash"]
        for game in dataset.games
    )
    return {
        "games": len(dataset.games),
        "teacher_labels": len(rows["actions"]),
        "masked_labels": masked,
        "round_trip_mismatches": round_trip,
        "replay_mismatches": int(replay_mismatches),
    }


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
    def __init__(self, strata: Sequence[Sequence[Any]], rng: np.random.Generator) -> None:
        if not strata or any(not rows for rows in strata):
            raise ValueError("sampler source contains an empty stratum")
        self._rows = [tuple(rows[index] for index in rng.permutation(len(rows))) for rows in strata]
        self._strata = list(rng.permutation(len(self._rows)))
        self._row_positions = [0] * len(self._rows)
        self._stratum_position = 0

    def take(self, count: int) -> list[Any]:
        result: list[Any] = []
        for _ in range(count):
            group = self._strata[self._stratum_position % len(self._strata)]
            self._stratum_position += 1
            row = self._rows[group][self._row_positions[group] % len(self._rows[group])]
            self._row_positions[group] += 1
            result.append(row)
        return result


def _materialized_source_strata(
    materialized: MaterializedImitationPartition, source: Source,
) -> list[tuple[int, ...]]:
    batch = materialized.batch
    selected = np.flatnonzero(batch.sources == source)
    if not len(selected):
        raise ValueError(
            f"sampler has no {source.value} rows in {materialized.partition} partition"
        )
    if source is Source.DAGGER_TARGETED:
        return [tuple(int(index) for index in selected)]
    grouped: dict[tuple[str, int, int], list[int]] = {}
    for raw_index in selected:
        index = int(raw_index)
        key = (
            str(batch.profiles[index]),
            int(batch.seats[index]),
            int(batch.action_kinds[index]),
        )
        grouped.setdefault(key, []).append(index)
    return [tuple(grouped[key]) for key in sorted(grouped)]


class SourceMixtureSampler:
    """Seeded residual-accounted source exposure over one materialized partition."""

    def __init__(
        self,
        materialized: MaterializedImitationPartition,
        *,
        source_fractions: Mapping[Source, float],
        batch_size: int = 1,
        seed: int = 0,
        partition: str | None = None,
    ) -> None:
        if (
            isinstance(seed, (bool, np.bool_))
            or not isinstance(seed, Integral)
            or int(seed) < 0
        ):
            raise ValueError("sampler configuration is invalid")
        self._initialize_with_rng(
            materialized,
            source_fractions=source_fractions,
            batch_size=batch_size,
            rng=np.random.default_rng(int(seed)),
            partition=partition,
            legacy_standard_fraction=None,
        )

    def _initialize_with_rng(
        self,
        materialized: MaterializedImitationPartition,
        *,
        source_fractions: Mapping[Source, float],
        batch_size: int,
        rng: np.random.Generator,
        partition: str | None,
        legacy_standard_fraction: float | None,
    ) -> None:
        if (
            not isinstance(materialized, MaterializedImitationPartition)
            or type(batch_size) is not int
            or batch_size < 1
            or not isinstance(rng, np.random.Generator)
        ):
            raise ValueError("sampler configuration is invalid")
        expected_partition = materialized.partition if partition is None else partition
        if (
            expected_partition not in {"train", "validation"}
            or materialized.partition != expected_partition
        ):
            raise ValueError("sampler materialized partition differs")
        fractions = _freeze_source_fractions(source_fractions)
        if set(materialized.batch.sources) != set(fractions):
            raise ValueError("sampler materialized sources differ from mixture")
        if legacy_standard_fraction is not None and (
            tuple(fractions) != (
                Source.GREEDY_STANDARD, Source.SEARCH_CONVERSION,
            )
            or not math.isclose(
                float(legacy_standard_fraction),
                fractions[Source.GREEDY_STANDARD],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("legacy sampler mixture is invalid")
        self.materialized = materialized
        self.batch_size = batch_size
        self.partition = expected_partition
        self.source_fractions = fractions
        self._rng = rng
        self._cyclers = {
            source: _StratumCycler(
                _materialized_source_strata(materialized, source), self._rng,
            )
            for source in fractions
        }
        self._carry = {source: 0.0 for source in fractions}
        self._legacy_standard_fraction = legacy_standard_fraction
        self._legacy_residual = 0.0

    def _source_counts(self) -> Mapping[Source, int]:
        if self._legacy_standard_fraction is not None:
            target = (
                self.batch_size * self._legacy_standard_fraction
                + self._legacy_residual
            )
            standard_count = int(np.floor(target + 1e-12))
            self._legacy_residual = target - standard_count
            return {
                Source.GREEDY_STANDARD: standard_count,
                Source.SEARCH_CONVERSION: self.batch_size - standard_count,
            }

        targets = {
            source: self.batch_size * fraction + self._carry[source]
            for source, fraction in self.source_fractions.items()
        }
        source_order = tuple(self.source_fractions)
        # Reconciliation preserves the per-batch invariant: every count is
        # nonnegative, counts sum to batch_size, and source order breaks ties.
        counts = {
            source: max(0, int(np.floor(target + 1e-12)))
            for source, target in targets.items()
        }
        while sum(counts.values()) > self.batch_size:
            loser = min(
                (
                    index for index, source in enumerate(source_order)
                    if counts[source] > 0
                ),
                key=lambda index: (
                    targets[source_order[index]] - counts[source_order[index]],
                    index,
                ),
            )
            counts[source_order[loser]] -= 1
        while sum(counts.values()) < self.batch_size:
            winner = max(
                range(len(source_order)),
                key=lambda index: (
                    targets[source_order[index]] - counts[source_order[index]],
                    -index,
                ),
            )
            counts[source_order[winner]] += 1
        if sum(counts.values()) != self.batch_size or any(
            count < 0 for count in counts.values()
        ):
            raise RuntimeError("source residual allocation is inconsistent")
        self._carry = {
            source: targets[source] - counts[source] for source in source_order
        }
        return counts

    def _next_indices_and_sources(self) -> list[tuple[int, Source]]:
        counts = self._source_counts()
        indices_and_sources = [
            (int(index), source)
            for source in self.source_fractions
            for index in self._cyclers[source].take(counts[source])
        ]
        order = self._rng.permutation(len(indices_and_sources))
        return [indices_and_sources[int(index)] for index in order]

    def next_batch(self) -> ImitationBatch:
        indices_and_sources = self._next_indices_and_sources()
        indices = np.fromiter(
            (index for index, _source in indices_and_sources),
            dtype=np.int64,
            count=len(indices_and_sources),
        )
        batch = _take_batch(self.materialized.batch, indices)
        scheduled_sources = np.asarray(
            [source for _index, source in indices_and_sources], dtype=object,
        )
        if not np.array_equal(batch.sources, scheduled_sources):
            raise RuntimeError("materialized source metadata differs from scheduler")
        if set(batch.partitions) != {self.partition}:
            raise RuntimeError("materialized sampler crossed a partition")
        if not np.all(
            batch.legal_masks[np.arange(len(batch.actions)), batch.actions]
        ):
            raise ValueError("selected teacher action is masked")
        return batch


class StratifiedDecisionSampler(SourceMixtureSampler):
    """Compatibility wrapper preserving the original two-source sequence."""

    def __init__(
        self,
        dataset: ImitationDataset,
        materialized: MaterializedImitationPartition,
        batch_size: int = 1,
        standard_fraction: float = 0.70,
        seed: int = 0,
        partition: str = "train",
    ) -> None:
        if (
            type(batch_size) is not int
            or batch_size < 1
            or partition not in {"train", "validation"}
            or not isinstance(standard_fraction, (int, float))
            or isinstance(standard_fraction, bool)
            or not 0.0 < float(standard_fraction) < 1.0
        ):
            raise ValueError("sampler configuration is invalid")
        if (
            not isinstance(materialized, MaterializedImitationPartition)
            or materialized.partition != partition
        ):
            raise ValueError("sampler materialized partition differs")
        if set(materialized.offsets) != set(_partition_refs(dataset, partition)):
            raise ValueError("sampler materialized reference map differs")
        self.dataset = dataset
        self.standard_fraction = float(standard_fraction)
        self._refs_by_index = {
            index: ref for ref, index in materialized.offsets.items()
        }
        fractions = MappingProxyType(OrderedDict((
            (Source.GREEDY_STANDARD, self.standard_fraction),
            (Source.SEARCH_CONVERSION, 1.0 - self.standard_fraction),
        )))
        legacy_rng = np.random.default_rng(seed)
        self._initialize_with_rng(
            materialized,
            source_fractions=fractions,
            batch_size=batch_size,
            rng=legacy_rng,
            partition=partition,
            legacy_standard_fraction=self.standard_fraction,
        )

    def _next_refs_and_sources(
        self,
    ) -> list[tuple[tuple[int, int], Source]]:
        return [
            (self._refs_by_index[index], source)
            for index, source in self._next_indices_and_sources()
        ]


def benchmark_imitation_sampler(
    dataset: ImitationDataset, *, batch_size: int, seed: int, batches: int = 200,
) -> Mapping[str, Any]:
    """Measure deterministic materialized sampler throughput without touching shards."""
    if (
        type(batch_size) is not int
        or batch_size < 1
        or type(seed) is not int
        or seed < 0
        or type(batches) is not int
        or batches < 1
    ):
        raise ValueError("sampler benchmark configuration is invalid")

    materialization_started = time.perf_counter()
    training = materialize_imitation_partition(dataset, "train")
    sampler = StratifiedDecisionSampler(
        dataset, training, batch_size=batch_size, seed=seed, partition="train",
    )
    materialization_seconds = time.perf_counter() - materialization_started
    sampling_started = time.perf_counter()
    digest = hashlib.sha256()
    source_codes = {
        Source.GREEDY_STANDARD: 0,
        Source.SEARCH_CONVERSION: 1,
    }
    for _ in range(batches):
        batch = sampler.next_batch()
        for values in (
            batch.game_ids,
            batch.decision_indices,
            batch.actions,
            np.fromiter(
                (source_codes[source] for source in batch.sources),
                dtype=np.uint8,
                count=len(batch.sources),
            ),
        ):
            digest.update(np.ascontiguousarray(values).tobytes())
    sampling_seconds = time.perf_counter() - sampling_started
    if not math.isfinite(sampling_seconds) or sampling_seconds <= 0:
        raise ValueError("sampler benchmark elapsed duration is invalid")

    examples = batches * batch_size
    return MappingProxyType({
        "schema_version": 1,
        "batches": batches,
        "examples": examples,
        "examples_per_second": examples / sampling_seconds,
        "materialization_seconds": materialization_seconds,
        "sampling_seconds": sampling_seconds,
        "sequence_sha256": digest.hexdigest(),
    })


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


@dataclass(frozen=True)
class BehavioralCloningConfig:
    model_seed: int = 0
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_epochs: int = 50
    patience: int = 5
    device: str = ""

    def __post_init__(self) -> None:
        if type(self.model_seed) is not int or self.model_seed < 0:
            raise ValueError("behavioral-cloning model_seed must be a non-negative integer")
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError("behavioral-cloning batch_size must be a positive integer")
        if not isinstance(self.learning_rate, (int, float)) or not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("behavioral-cloning learning_rate must be finite and positive")
        if type(self.max_epochs) is not int or self.max_epochs < 1:
            raise ValueError("behavioral-cloning max_epochs must be a positive integer")
        if type(self.patience) is not int or self.patience < 1:
            raise ValueError("behavioral-cloning patience must be a positive integer")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "behavioral-cloning device must be exactly 'cpu' or 'cuda'"
            )



def resolve_behavioral_cloning_device(requested: str) -> dict[str, Any]:
    import torch

    if requested not in {"cpu", "cuda"}:
        raise ValueError("unsupported behavioral-cloning device")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("behavioral-cloning CUDA device is unavailable")
        index = int(torch.cuda.current_device())
        return {
            "requested": "cuda",
            "resolved": f"cuda:{index}",
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "device_index": index,
            "device_name": str(torch.cuda.get_device_name(index)),
        }
    return {
        "requested": "cpu",
        "resolved": "cpu",
        "torch_version": str(torch.__version__),
        "cuda_runtime": None,
        "device_index": None,
        "device_name": None,
    }


def _software_provenance() -> dict[str, str]:
    import importlib.metadata
    import platform
    import sys

    import torch

    return {
        "python_version": platform.python_version(),
        "numpy_version": str(np.__version__),
        "torch_version": str(torch.__version__),
        "stable_baselines3_version": importlib.metadata.version(
            "stable-baselines3"
        ),
        "sb3_contrib_version": importlib.metadata.version("sb3-contrib"),
        "platform": platform.platform() or sys.platform,
    }

@dataclass(frozen=True)
class CloneMetrics:
    nll: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    expected_calibration_error: float
    mean_end_turn_probability: float
    illegal_probability: float
    strata: Mapping[str, Mapping[str, float | int]]


@dataclass(frozen=True)
class BehavioralCloningResult:
    run_dir: Path
    validation: CloneMetrics
    best_epoch: int
    epochs_trained: int


def _partition_refs(dataset: ImitationDataset, partition: str) -> list[tuple[int, int]]:
    try:
        sources = dataset.index[partition]
    except KeyError as exc:
        raise ValueError(f"imitation dataset has no {partition} partition") from exc
    refs: list[tuple[int, int]] = []
    for source in sorted(sources, key=lambda value: value.value):
        for profile in sorted(sources[source]):
            for seat in sorted(sources[source][profile]):
                for kind in sorted(sources[source][profile][seat]):
                    refs.extend(sources[source][profile][seat][kind])
    if not refs or len(set(refs)) != len(refs):
        raise ValueError("partition references are empty or duplicated")
    return refs


def materialize_imitation_partition(
    dataset: ImitationDataset, partition: str,
) -> MaterializedImitationPartition:
    refs = _partition_refs(dataset, partition)
    rows = dataset._row_data(refs)
    order = np.lexsort((rows["decision_indices"], rows["game_ids"]))
    rows = {name: values[order] for name, values in rows.items()}
    ordered_refs = [refs[int(index)] for index in order]
    legal_masks = np.unpackbits(
        rows["packed_masks"], axis=1, count=dataset.contract.action_size, bitorder="little"
    ).astype(bool, copy=False)
    metadata = [dataset.games[int(game_id)] for game_id in rows["game_ids"]]
    batch = ImitationBatch(
        observations=rows["observations"],
        legal_masks=legal_masks,
        actions=rows["actions"],
        game_ids=rows["game_ids"],
        decision_indices=rows["decision_indices"],
        sources=np.asarray([_source_for_game(game) for game in metadata], dtype=object),
        profiles=np.asarray([game["profile"] for game in metadata], dtype=object),
        seats=rows["seats"],
        action_kinds=rows["action_kinds"],
        partitions=np.full(len(rows["actions"]), partition, dtype=object),
    )
    return MaterializedImitationPartition(
        partition=partition,
        batch=batch,
        offsets=MappingProxyType({ref: index for index, ref in enumerate(ordered_refs)}),
    )


def _take_batch(batch: ImitationBatch, indices: np.ndarray) -> ImitationBatch:
    return ImitationBatch(**{name: getattr(batch, name)[indices].copy() for name in ImitationBatch.__dataclass_fields__})


def _fixture_batch(validation: ImitationBatch, limit: int = 32) -> ImitationBatch:
    if type(limit) is not int or limit < 1:
        raise ValueError("actor fixture limit must be positive")
    count = len(validation.actions)
    non_end = np.flatnonzero(validation.actions != END_TURN_ACTION)
    if not len(non_end):
        raise ValueError("actor fixtures require a non-EndTurn validation row")
    required = [int(non_end[0])]
    for seat in sorted(int(value) for value in np.unique(validation.seats)):
        required.append(int(np.flatnonzero(validation.seats == seat)[0]))
    selected: list[int] = []
    for index in [*required, *range(count)]:
        if index not in selected:
            selected.append(index)
        if len(selected) == min(limit, count):
            break
    selected.sort()
    return _take_batch(validation, np.asarray(selected, dtype=np.int64))


def _actor_named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    groups = (
        ("features_extractor", model.policy.features_extractor),
        ("mlp_extractor.policy_net", model.policy.mlp_extractor.policy_net),
        ("action_net", model.policy.action_net),
    )
    return tuple((f"{prefix}.{name}", parameter) for prefix, module in groups for name, parameter in module.named_parameters())


def _value_named_parameters(model: Any) -> tuple[tuple[str, Any], ...]:
    groups = (
        ("mlp_extractor.value_net", model.policy.mlp_extractor.value_net),
        ("value_net", model.policy.value_net),
    )
    return tuple((f"{prefix}.{name}", parameter) for prefix, module in groups for name, parameter in module.named_parameters())


def _parameter_hash(named_parameters: Sequence[tuple[str, Any]]) -> str:
    digest = hashlib.sha256()
    for name, parameter in named_parameters:
        array = parameter.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _distribution_tensors(model: Any, batch: ImitationBatch) -> tuple[Any, Any, Any]:
    import torch

    def tensor_source(values: np.ndarray) -> np.ndarray:
        return values if values.flags.writeable else values.copy()

    observations = torch.as_tensor(
        tensor_source(batch.observations), dtype=torch.float32, device=model.device,
    )
    legal_masks = torch.as_tensor(
        tensor_source(batch.legal_masks), dtype=torch.bool, device=model.device,
    )
    actions = torch.as_tensor(
        tensor_source(batch.actions), dtype=torch.int64, device=model.device,
    )
    distribution = model.policy.get_distribution(observations, action_masks=legal_masks)
    return distribution, actions, legal_masks


def _masked_logits(model: Any, batch: ImitationBatch) -> Any:
    import torch

    model.policy.set_training_mode(False)
    with torch.no_grad():
        distribution, _actions, _masks = _distribution_tensors(model, batch)
        return distribution.distribution.logits.detach().cpu()


def _strata_metrics(predictions: np.ndarray, actions: np.ndarray, batch: ImitationBatch) -> Mapping[str, Mapping[str, float | int]]:
    teacher_names = {
        Source.GREEDY_STANDARD: "greedy",
        Source.SEARCH_CONVERSION: "bounded-search",
        Source.DAGGER_TARGETED: "dagger-targeted",
    }
    teacher = np.asarray(
        [teacher_names[source] for source in batch.sources], dtype=object,
    )
    kind_names = {value: name for name, value in ACTION_KINDS.items()}
    dimensions = {
        "teacher": teacher,
        "profile": batch.profiles,
        "action_kind": np.asarray([kind_names[int(value)] for value in batch.action_kinds], dtype=object),
        "seat": np.asarray([str(int(value)) for value in batch.seats], dtype=object),
    }
    strata: dict[str, Mapping[str, float | int]] = {}
    for dimension, values in dimensions.items():
        for value in sorted(str(item) for item in np.unique(values)):
            selected = np.asarray([str(item) == value for item in values], dtype=bool)
            strata[f"{dimension}/{value}"] = {
                "count": int(selected.sum()),
                "accuracy": float(np.mean(predictions[selected] == actions[selected])),
            }
    return strata


def _clone_metrics(model: Any, batch: ImitationBatch) -> CloneMetrics:
    import torch

    if len(batch.actions) == 0:
        raise ValueError("clone metrics require at least one row")
    model.policy.set_training_mode(False)
    with torch.no_grad():
        distribution, actions, legal_masks = _distribution_tensors(model, batch)
        log_prob = distribution.log_prob(actions)
        probabilities = distribution.distribution.probs
        predictions = probabilities.argmax(dim=1)
        accuracies: dict[int, float] = {}
        for requested in (1, 3, 5):
            width = min(requested, probabilities.shape[1])
            top = probabilities.topk(width, dim=1).indices
            accuracies[requested] = float((top == actions[:, None]).any(dim=1).float().mean().cpu())
        confidence = probabilities.max(dim=1).values
        correct = predictions == actions
        bins = torch.clamp((confidence * 10).floor().to(torch.int64), max=9)
        ece = torch.zeros((), dtype=probabilities.dtype, device=probabilities.device)
        for index in range(10):
            selected = bins == index
            if selected.any():
                ece = ece + selected.float().mean() * torch.abs(
                    correct[selected].float().mean() - confidence[selected].mean()
                )
        metrics = CloneMetrics(
            nll=float((-log_prob.mean()).cpu()),
            top1_accuracy=accuracies[1],
            top3_accuracy=accuracies[3],
            top5_accuracy=accuracies[5],
            expected_calibration_error=float(ece.cpu()),
            mean_end_turn_probability=float(probabilities[:, END_TURN_ACTION].mean().cpu()),
            illegal_probability=float((probabilities * (~legal_masks)).sum(dim=1).mean().cpu()),
            strata=_strata_metrics(
                predictions.cpu().numpy(), actions.cpu().numpy(), batch
            ),
        )
    scalars = (
        metrics.nll,
        metrics.top1_accuracy,
        metrics.top3_accuracy,
        metrics.top5_accuracy,
        metrics.expected_calibration_error,
        metrics.mean_end_turn_probability,
        metrics.illegal_probability,
    )
    if not all(np.isfinite(value) for value in scalars):
        raise ValueError("clone validation metrics must be finite")
    return metrics


def _atomic_actor_fixtures(path: Path, fixtures: ImitationBatch) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(
                stream,
                observations=fixtures.observations.astype(np.float32, copy=False),
                legal_masks=fixtures.legal_masks.astype(bool, copy=False),
            )
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _assert_no_bc_optimizer_state(checkpoint: Path) -> None:
    import torch

    with zipfile.ZipFile(checkpoint) as archive:
        names = archive.namelist()
        if any("bc_optimizer" in name.lower() for name in names):
            raise RuntimeError("behavioral-cloning optimizer state entered the saved archive")
        if "policy.optimizer.pth" in names:
            state = torch.load(
                io.BytesIO(archive.read("policy.optimizer.pth")),
                map_location="cpu",
                weights_only=True,
            )
            if state.get("state"):
                raise RuntimeError("saved PPO optimizer unexpectedly contains training state")


def _verify_reload_identity(
    checkpoint: Path,
    adapter: Any,
    contract: EnvironmentContract,
    fixtures: ImitationBatch,
    expected_logits: Any,
) -> Mapping[str, Any]:
    import torch

    from .algorithms import actor_state_sha256

    checkpoint = Path(checkpoint)
    reloaded = adapter.load(checkpoint, env=None, device="cpu")
    adapter.validate_model(reloaded, contract)
    if {parameter.device.type for parameter in reloaded.policy.parameters()} != {
        "cpu"
    }:
        raise RuntimeError("reloaded actor publication is not canonical CPU")
    actual_logits = _masked_logits(reloaded, fixtures)
    torch.testing.assert_close(actual_logits, expected_logits, rtol=0, atol=0)
    return {
        "checkpoint_sha256": sha256_file(checkpoint),
        "actor_sha256": actor_state_sha256(reloaded),
        "contract_hash": contract.contract_hash,
        "encoding_hash": contract.encoding_hash,
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
        "comparison_rtol": 0.0,
        "comparison_atol": 0.0,
        "maximum_absolute_logit_difference": float(
            torch.max(torch.abs(actual_logits - expected_logits)).item()
        ),
    }


def canonicalize_behavioral_clone_for_publication(model: Any) -> None:
    import torch

    model.policy.to(torch.device("cpu"))
    model.device = torch.device("cpu")
    devices = {parameter.device.type for parameter in model.policy.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError("behavioral-cloning publication model is not on CPU")


_BC_PHASE_FIELDS = (
    "sampling_seconds",
    "transfer_forward_seconds",
    "optimization_seconds",
    "validation_seconds",
    "unclassified_seconds",
)


def _validate_behavioral_cloning_progress_event(event: Mapping[str, Any]) -> None:
    count_fields = (
        "schema_version", "model_seed", "epoch", "max_epochs", "batches", "examples",
        "best_epoch", "epochs_without_improvement", "patience",
    )
    scalar_fields = (
        "mean_training_loss", "validation_nll", "top1_accuracy", "top3_accuracy",
        "top5_accuracy", "best_validation_nll", "epoch_seconds", "elapsed_seconds",
        "examples_per_second", *_BC_PHASE_FIELDS,
    )
    if any(field not in event for field in _BC_PHASE_FIELDS):
        raise ValueError("behavioral-cloning progress phase timing fields are required")
    for field in count_fields:
        value = event[field]
        if type(value) is not int or value < 0:
            raise ValueError(f"behavioral-cloning progress {field} must be a non-negative integer")
    for field in scalar_fields:
        value = event[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
            raise ValueError(f"behavioral-cloning progress {field} must be finite")
    if any(event[field] < 0 for field in ("epoch_seconds", "elapsed_seconds", "examples_per_second")):
        raise ValueError("behavioral-cloning progress timings and rate must be non-negative")
    if any(event[field] < 0 for field in _BC_PHASE_FIELDS):
        raise ValueError("behavioral-cloning progress phase timing must be non-negative")
    if not math.isclose(
        sum(float(event[key]) for key in _BC_PHASE_FIELDS), float(event["epoch_seconds"]),
        rel_tol=1e-9, abs_tol=1e-6,
    ):
        raise ValueError("behavioral-cloning phase timing differs from epoch duration")


def _prepare_behavioral_cloning_request(
    config: BehavioralCloningConfig, run_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(config, BehavioralCloningConfig):
        raise TypeError("config must be BehavioralCloningConfig")
    prepared_run_dir = Path(run_dir)
    training_device = resolve_behavioral_cloning_device(config.device)
    prepared_run_dir.parent.mkdir(parents=True, exist_ok=True)
    if prepared_run_dir.exists():
        raise FileExistsError(prepared_run_dir)
    return prepared_run_dir, training_device


def train_actor_supervision(
    *,
    corpus: ActorSupervisionCorpus,
    scenario: ResolvedScenario,
    env: Any,
    contract: EnvironmentContract,
    spaces_info: Mapping[str, Any],
    run_dir: Path,
    config: BehavioralCloningConfig = BehavioralCloningConfig(device="cpu"),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    _sampler_factory: Callable[
        [MaterializedImitationPartition, int, int], Any
    ] | None = None,
    _prepared_request: tuple[Path, dict[str, Any]] | None = None,
    warm_start: Any | None = None,
    distillation_mode: str | None = None,
) -> BehavioralCloningResult:
    """Train the production actor from generic immutable supervision partitions."""
    import torch
    from .algorithms import (
        ActorTransferSource,
        MaskablePPOAdapter,
        actor_state_sha256,
    )
    from .controllers import ControllerResolver

    if not isinstance(corpus, ActorSupervisionCorpus):
        raise TypeError("corpus must be an ActorSupervisionCorpus")
    if not isinstance(scenario, ResolvedScenario):
        raise TypeError("scenario must be a ResolvedScenario")
    if scenario.environment != contract.environment:
        raise ContractMismatch("behavioral-cloning scenario environment does not match the contract")
    scenario_hash = hashlib.sha256(scenario.canonical_json.encode("utf-8")).hexdigest()
    if (
        corpus.identity.get("contract_hash") != contract.contract_hash
        or corpus.identity.get("encoding_hash") != contract.encoding_hash
    ):
        raise ContractMismatch("actor-supervision corpus contract does not match")
    if corpus.identity.get("scenario_hash") != scenario_hash:
        raise ContractMismatch("behavioral-cloning scenario hash does not match the corpus")
    dataset_manifest_sha256 = corpus.identity.get("base_manifest_sha256")
    if not _is_hash(dataset_manifest_sha256):
        raise ValueError("actor-supervision base manifest identity is invalid")
    if (warm_start is None) != (distillation_mode is None):
        raise ValueError(
            "actor-supervision warm start and distillation mode are inseparable"
        )
    if warm_start is not None:
        if not isinstance(warm_start, ActorTransferSource):
            raise TypeError("warm_start must be an ActorTransferSource")
        if distillation_mode not in {"production", "nonproduction_cpu"}:
            raise ValueError("actor-supervision distillation mode is invalid")
        if set(corpus.validation.batch.sources) != {Source.DAGGER_TARGETED}:
            raise ValueError(
                "DAgger distillation validation must contain targeted labels only"
            )

    if _prepared_request is None:
        run_dir, training_device = _prepare_behavioral_cloning_request(
            config, run_dir,
        )
    else:
        run_dir, training_device = _prepared_request

    training = corpus.training
    validation = corpus.validation
    fixtures = _fixture_batch(validation.batch)
    adapter = MaskablePPOAdapter()
    model = adapter.create(
        env,
        spaces_info=dict(spaces_info),
        seed=config.model_seed,
        device=config.device,
        checkpoint_interval=2,
    )
    adapter.validate_model(model, contract)
    parameter_devices = {parameter.device.type for parameter in model.policy.parameters()}
    if parameter_devices != {config.device}:
        raise RuntimeError("behavioral-cloning model parameters are not on the requested device")

    value_named = _value_named_parameters(model)
    value_parameters = tuple(parameter for _name, parameter in value_named)
    if not value_parameters:
        raise RuntimeError("production policy did not expose value parameters")
    value_before = tuple(
        parameter.detach().cpu().clone() for parameter in value_parameters
    )
    value_hash_before = _parameter_hash(value_named)
    actor_initialization = None
    if warm_start is not None:
        resolved_source = ControllerResolver(contract).resolve(
            dict(warm_start.controller)
        )
        actor_initialization = adapter.initialize_actor_from_resolved(
            model,
            resolved_source,
            source=warm_start,
            expected_contract=contract,
            device=config.device,
        )
        if _parameter_hash(value_named) != value_hash_before:
            raise RuntimeError("actor warm-start modified value-side parameters")
    target_actor_sha256_initial = actor_state_sha256(model)
    actor_named = _actor_named_parameters(model)
    actor_parameters = tuple(chain(
        model.policy.features_extractor.parameters(),
        model.policy.mlp_extractor.policy_net.parameters(),
        model.policy.action_net.parameters(),
    ))
    assert {id(parameter) for parameter in actor_parameters} == {id(parameter) for _name, parameter in actor_named}
    if not actor_parameters or not value_parameters:
        raise RuntimeError("production policy did not expose actor/value parameter groups")
    if {id(parameter) for parameter in actor_parameters} & {id(parameter) for parameter in value_parameters}:
        raise RuntimeError("behavioral-cloning actor and value parameter groups overlap")
    optimizer = torch.optim.Adam(actor_parameters, lr=float(config.learning_rate))

    sampler = (
        SourceMixtureSampler(
            training,
            source_fractions=corpus.source_fractions,
            batch_size=config.batch_size,
            seed=config.model_seed,
            partition="train",
        )
        if _sampler_factory is None
        else _sampler_factory(training, config.batch_size, config.model_seed)
    )
    steps_per_epoch = max(1, int(np.ceil(len(training.batch.actions) / config.batch_size)))
    best_nll = float("inf")
    best_epoch = 0
    best_actor_state: tuple[Any, ...] | None = None
    epochs_without_improvement = 0
    epochs_trained = 0
    source_example_counts: Counter[str] = Counter()

    history: list[dict[str, Any]] = []
    training_started = time.perf_counter()
    for epoch in range(1, config.max_epochs + 1):
        model.policy.set_training_mode(True)
        epoch_started = time.perf_counter()
        losses: list[float] = []
        sampling_seconds = 0.0
        transfer_forward_seconds = 0.0
        optimization_seconds = 0.0
        validation_seconds = 0.0
        epoch_source_counts: Counter[str] = Counter()
        for _step in range(steps_per_epoch):
            phase_started = time.perf_counter()
            batch = sampler.next_batch()
            sampling_seconds += time.perf_counter() - phase_started
            if set(batch.partitions) != {"train"}:
                raise RuntimeError("validation rows entered behavioral-cloning optimization")
            selected_sources = Counter(
                source.value for source in batch.sources
            )
            source_example_counts.update(selected_sources)
            epoch_source_counts.update(selected_sources)

            phase_started = time.perf_counter()
            distribution, actions, _legal_masks = _distribution_tensors(model, batch)
            loss = -distribution.log_prob(actions).mean()
            losses.append(float(loss.detach().cpu()))
            transfer_forward_seconds += time.perf_counter() - phase_started

            phase_started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if any(parameter.grad is not None for parameter in value_parameters):
                raise RuntimeError(
                    "actor supervision produced a value-side gradient"
                )
            torch.nn.utils.clip_grad_norm_(actor_parameters, max_norm=1.0)
            optimizer.step()
            optimization_seconds += time.perf_counter() - phase_started
        epochs_trained = epoch
        validation_started = time.perf_counter()
        validation_metrics = _clone_metrics(model, validation.batch)
        validation_seconds = time.perf_counter() - validation_started
        if validation_metrics.nll < best_nll:
            best_nll = validation_metrics.nll
            best_epoch = epoch
            best_actor_state = tuple(parameter.detach().cpu().clone() for parameter in actor_parameters)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        epoch_elapsed = time.perf_counter() - epoch_started
        total_elapsed = time.perf_counter() - training_started
        classified = (
            sampling_seconds
            + transfer_forward_seconds
            + optimization_seconds
            + validation_seconds
        )
        raw_unclassified = epoch_elapsed - classified
        if raw_unclassified < -1e-6:
            raise RuntimeError("behavioral-cloning phase timing exceeds epoch duration")
        unclassified_seconds = max(0.0, raw_unclassified)
        examples = steps_per_epoch * config.batch_size
        event = {
            "schema_version": 1,
            "event": "bc_epoch",
            "model_seed": config.model_seed,
            "device": training_device["resolved"],
            "epoch": epoch,
            "max_epochs": config.max_epochs,
            "batches": steps_per_epoch,
            "examples": examples,
            "mean_training_loss": float(sum(losses) / len(losses)),
            "validation_nll": float(validation_metrics.nll),
            "top1_accuracy": float(validation_metrics.top1_accuracy),
            "top3_accuracy": float(validation_metrics.top3_accuracy),
            "top5_accuracy": float(validation_metrics.top5_accuracy),
            "best_epoch": int(best_epoch),
            "best_validation_nll": float(best_nll),
            "epochs_without_improvement": int(epochs_without_improvement),
            "patience": int(config.patience),
            "epoch_seconds": float(epoch_elapsed),
            "elapsed_seconds": float(total_elapsed),
            "examples_per_second": float(examples / epoch_elapsed) if epoch_elapsed else 0.0,
            "sampling_seconds": float(sampling_seconds),
            "transfer_forward_seconds": float(transfer_forward_seconds),
            "optimization_seconds": float(optimization_seconds),
            "validation_seconds": float(validation_seconds),
            "unclassified_seconds": float(unclassified_seconds),
        }
        if warm_start is not None:
            event["source_example_counts"] = {
                source.value: int(epoch_source_counts[source.value])
                for source in corpus.source_fractions
            }
        _validate_behavioral_cloning_progress_event(event)
        history.append(event)
        if progress is not None:
            progress(dict(event))
        if epochs_without_improvement >= config.patience:
            break

    if best_actor_state is None:
        raise RuntimeError("behavioral cloning did not produce a finite validation epoch")
    with torch.no_grad():
        for parameter, best_value in zip(actor_parameters, best_actor_state, strict=True):
            parameter.copy_(best_value.to(parameter.device))
    canonicalize_behavioral_clone_for_publication(model)
    for parameter, original in zip(value_parameters, value_before, strict=True):
        if not torch.equal(parameter.detach().cpu(), original):
            raise RuntimeError("behavioral cloning modified a value-side parameter")
    value_hash_after = _parameter_hash(value_named)
    if value_hash_after != value_hash_before:
        raise RuntimeError("behavioral cloning modified value-side parameters")
    target_actor_sha256_final = actor_state_sha256(model)
    validation_metrics = _clone_metrics(model, validation.batch)
    expected_logits = _masked_logits(model, fixtures)

    config_data = asdict(config)
    metrics_data = asdict(validation_metrics)
    corpus_identity = dict(corpus.identity)
    source_mixture_data = [
        {"source": source.value, "fraction": fraction}
        for source, fraction in corpus.source_fractions.items()
    ]
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{run_dir.name}.publishing-", dir=run_dir.parent)
    )
    try:
        checkpoint = temporary / "checkpoints" / "step_000000000.zip"
        checkpoint.parent.mkdir(parents=True)
        checkpoint = adapter.save(model, checkpoint)
        _assert_no_bc_optimizer_state(checkpoint)
        _atomic_actor_fixtures(temporary / "actor-fixtures.npz", fixtures)
        publication_verification = dict(_verify_reload_identity(
            checkpoint,
            adapter,
            contract,
            fixtures,
            expected_logits,
        ))
        if publication_verification["actor_sha256"] != target_actor_sha256_final:
            raise RuntimeError(
                "reloaded actor hash differs from the canonical training actor"
            )
        checkpoint_sha256 = publication_verification["checkpoint_sha256"]
        software_provenance = _software_provenance()
        total_source_counts = {
            source.value: int(source_example_counts[source.value])
            for source in corpus.source_fractions
        }
        bc_data = {
            "schema_version": 1,
            "algorithm": adapter.name,
            "policy": adapter.policy_name,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "supervision_corpus": corpus_identity,
            "source_mixture": source_mixture_data,
            "config": config_data,
            "model_seed": config.model_seed,
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "training_device": training_device,
            "publication_device": "cpu",
            "best_validation_nll": best_nll,
            "actor_parameter_count": int(sum(parameter.numel() for parameter in actor_parameters)),
            "value_parameter_count": int(sum(parameter.numel() for parameter in value_parameters)),
            "value_parameters_sha256_before": value_hash_before,
            "value_parameters_sha256_after": value_hash_after,
            "target_actor_sha256_initial": target_actor_sha256_initial,
            "target_actor_sha256_final": target_actor_sha256_final,
            "checkpoint_sha256": checkpoint_sha256,
            "publication_verification": publication_verification,
            "software_provenance": software_provenance,
            "source_example_counts": total_source_counts,
            "validation_partition": (
                "held_out_targeted"
                if warm_start is not None
                else "held_out_demonstration"
            ),
            "training_decision_count": int(len(training.batch.actions)),
            "validation_decision_count": int(len(validation.batch.actions)),
            "validation_game_count": int(len(np.unique(validation.batch.game_ids))),
        }
        if warm_start is not None:
            bc_data.update({
                "training_kind": "selective-dagger-distillation-v1",
                "distillation_iteration": len(
                    corpus.identity["train_overlays"]
                ),
                "production": distillation_mode == "production",
                "actor_initialization": actor_initialization,
            })
        manifest = {
            "schema_version": 1,
            "state": "completed",
            "timesteps": 0,
            "latest_checkpoint": "checkpoints/step_000000000.zip",
            "latest_checkpoint_step": 0,
            "checkpoint_sha256": checkpoint_sha256,
            "config": {
                "backend": "stable_baselines3",
                "algorithm": adapter.name,
                "policy": adapter.policy_name,
                "seed": config.model_seed,
                "model_seed": config.model_seed,
                "device": "cpu",
                "behavioral_cloning": config_data,
            },
            "contract": contract.to_dict(),
            "scenario": {"path": "scenario.json", "template_id": scenario.template_id, "schema_version": scenario.schema_version},
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "supervision_corpus": corpus_identity,
            "source_mixture": source_mixture_data,
            "bc_config": config_data,
            "model_seed": config.model_seed,
            "best_epoch": best_epoch,
            "target_actor_sha256_initial": target_actor_sha256_initial,
            "target_actor_sha256_final": target_actor_sha256_final,
            "value_parameters_sha256_before": value_hash_before,
            "value_parameters_sha256_after": value_hash_after,
            "publication_verification": publication_verification,
            "software_provenance": software_provenance,
            "source_example_counts": total_source_counts,
        }
        if warm_start is not None:
            manifest.update({
                "training_kind": "selective-dagger-distillation-v1",
                "distillation_iteration": len(
                    corpus.identity["train_overlays"]
                ),
                "production": distillation_mode == "production",
                "actor_initialization": actor_initialization,
            })
        atomic_write_json(
            temporary / "training-history.json",
            {
                "schema_version": 1,
                "model_seed": config.model_seed,
                "training_device": training_device,
                "publication_device": "cpu",
                "epochs": history,
            },
        )
        scenario.write(temporary / "scenario.json")
        atomic_write_json(temporary / "bc.json", bc_data)
        atomic_write_json(temporary / "metrics.json", metrics_data)
        atomic_write_json(temporary / "run.json", manifest)
        os.replace(temporary, run_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    if progress is not None:
        progress({
            "schema_version": 1,
            "event": "bc_complete",
            "model_seed": config.model_seed,
            "device": training_device["resolved"],
            "best_epoch": best_epoch,
            "epochs_trained": epochs_trained,
            "elapsed_seconds": total_elapsed,
            "run_dir": str(run_dir.resolve()),
        })

    return BehavioralCloningResult(
        run_dir=run_dir,
        validation=validation_metrics,
        best_epoch=best_epoch,
        epochs_trained=epochs_trained,
    )


def train_behavioral_clone(
    *,
    dataset: ImitationDataset,
    scenario: ResolvedScenario,
    env: Any,
    contract: EnvironmentContract,
    spaces_info: Mapping[str, Any],
    run_dir: Path,
    config: BehavioralCloningConfig = BehavioralCloningConfig(device="cpu"),
    progress: Callable[[Mapping[str, Any]], None] | None = None,
) -> BehavioralCloningResult:
    """Behavior-preserving adapter for the original two-source imitation dataset."""

    if not isinstance(dataset, ImitationDataset):
        raise TypeError("dataset must be a loaded ImitationDataset")
    if dataset.contract != contract:
        raise ContractMismatch("behavioral-cloning dataset contract does not match")
    if not isinstance(scenario, ResolvedScenario):
        raise TypeError("scenario must be a ResolvedScenario")
    if scenario.environment != contract.environment:
        raise ContractMismatch(
            "behavioral-cloning scenario environment does not match the contract"
        )
    scenario_hash = hashlib.sha256(
        scenario.canonical_json.encode("utf-8")
    ).hexdigest()
    if any(game["scenario_hash"] != scenario_hash for game in dataset.games):
        raise ContractMismatch(
            "behavioral-cloning scenario hash does not match the dataset"
        )

    prepared_request = _prepare_behavioral_cloning_request(config, run_dir)
    training = materialize_imitation_partition(dataset, "train")
    validation = materialize_imitation_partition(dataset, "validation")
    if set(training.offsets) & set(validation.offsets):
        validation = MaterializedImitationPartition(
            partition="validation",
            batch=validation.batch,
            offsets=MappingProxyType({
                ("validation-alias", *identity): index
                for identity, index in validation.offsets.items()
            }),
        )
    base_manifest_sha256 = sha256_file(dataset.root / "manifest.json")
    fractions = MappingProxyType(OrderedDict((
        (Source.GREEDY_STANDARD, 0.70),
        (Source.SEARCH_CONVERSION, 0.30),
    )))
    corpus = ActorSupervisionCorpus(
        training=training,
        validation=validation,
        source_fractions=fractions,
        identity=MappingProxyType({
            "schema_version": 1,
            "kind": "behavioral-cloning-v1",
            "base_manifest_sha256": base_manifest_sha256,
            "contract_hash": contract.contract_hash,
            "encoding_hash": contract.encoding_hash,
            "scenario_hash": scenario_hash,
        }),
    )

    def legacy_sampler(
        materialized: MaterializedImitationPartition,
        batch_size: int,
        seed: int,
    ) -> StratifiedDecisionSampler:
        return StratifiedDecisionSampler(
            dataset,
            materialized,
            batch_size=batch_size,
            standard_fraction=0.70,
            seed=seed,
            partition="train",
        )

    return train_actor_supervision(
        corpus=corpus,
        scenario=scenario,
        env=env,
        contract=contract,
        spaces_info=spaces_info,
        run_dir=run_dir,
        config=config,
        progress=progress,
        _sampler_factory=legacy_sampler,
        _prepared_request=prepared_request,
    )
