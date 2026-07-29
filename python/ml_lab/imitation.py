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

from .contracts import EnvironmentContract
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
