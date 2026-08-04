"""Immutable selective-DAgger overlay schemas and physical storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import EnvironmentContract
from .controllers import ResolvedController, predict, validate_inference_input
from .evaluation import DuelClient, MAX_DECISIONS_PER_GAME, validate_dagger_payload
from .io import atomic_write_json
from .protocol import validate_step_payload
from .tactical_trace import EpisodeTrace


OVERLAY_SCHEMA_VERSION = 1
_INT32_MAX = (1 << 31) - 1
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_STATE_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_COMMAND_FIELDS = frozenset({"Kind", "Issuer", "ActorId", "TargetId", "Q", "R"})
_ROW_FIELDS = frozenset({
    "observation", "legal_mask", "learner_action", "learner_command",
    "teacher_action", "teacher_command", "reason_bits", "state_hash",
    "normalized_advantage", "opponent_living_unit_count",
    "productive_legal_action_count", "seat", "round", "decision_index",
    "disagreement", "oracle_actual_expansion_count",
})
_GAME_FIELDS = frozenset({
    "game_id", "partition", "iteration", "map_seed", "episode_seed",
    "schedule_index", "profile", "reference_seat", "learner_seat", "opponent",
    "outcome", "transition_count", "trace_path", "replay_path",
})
_PROFILES = frozenset({
    "standard-3v3",
    "conversion-3v1-near", "conversion-3v1-far",
    "conversion-2v1-near", "conversion-2v1-far",
    "conversion-1v1-near", "conversion-1v1-far",
})


# Ranges are inclusive. This is the single source of truth; consumers never
# classify a seed by decimal prefix.
SEED_DEFINITIONS: tuple[tuple[str, int | None, int, int], ...] = (
    ("train", 1, 18_000_000, 18_099_999),
    ("train", 2, 18_100_000, 18_199_999),
    ("train", 3, 18_200_000, 18_299_999),
    ("oracle_preflight", None, 18_900_000, 18_900_119),
    ("smoke", None, 18_990_000, 18_990_009),
    ("validation", 1, 19_000_000, 19_009_999),
    ("validation", 2, 19_010_000, 19_019_999),
    ("validation", 3, 19_020_000, 19_029_999),
    ("reserved", None, 19_030_000, 19_099_999),
    ("development_evaluation", None, 20_000_000, 20_000_099),
)


def _strict_fields(value: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ValueError(f"{label} fields are invalid")
    return value


def _strict_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{label} must be an integer" + (
            f" >= {minimum}" if minimum is not None else ""
        ))
    return value


def _strict_int32(value: Any, label: str, *, minimum: int | None = None) -> int:
    parsed = _strict_int(value, label, minimum=minimum)
    if parsed > _INT32_MAX:
        raise ValueError(f"{label} must fit int32")
    return parsed


def _strict_json_float_for_float32(value: Any, label: str) -> float:
    """Validate a JSON float can be deterministically narrowed to finite float32.

    The binary64 JSON value need not equal the widened float32 storage value.
    """

    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"{label} must be an exact finite float")
    with np.errstate(over="ignore"):
        narrowed = np.float32(value)
    if not np.isfinite(narrowed):
        raise ValueError(f"{label} must be representable as finite float32")
    return value


def _strict_string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HASH_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _safe_relative(value: Any, label: str) -> str:
    text = _strict_string(value, label)
    windows = PureWindowsPath(text)
    posix = PurePosixPath(text.replace(chr(92), "/"))
    if (
        windows.drive
        or windows.root
        or windows.anchor
        or posix.root
        or ".." in windows.parts
        or ".." in posix.parts
        or posix == PurePosixPath(".")
    ):
        raise ValueError(f"{label} must be a contained relative path")
    return posix.as_posix()


def _contained_file(root: Path, relative: Any, label: str) -> Path:
    canonical = _safe_relative(relative, label)
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved = (resolved_root / canonical).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
        raise ValueError(f"{label} must resolve to a contained file")
    return resolved


def validate_seed_definitions(
    definitions: Sequence[tuple[str, int | None, int, int]] = SEED_DEFINITIONS,
) -> None:
    seen_keys: set[tuple[str, int | None]] = set()
    checked: list[tuple[str, int | None, int, int]] = []
    for item in definitions:
        if not isinstance(item, tuple) or len(item) != 4:
            raise ValueError("seed definition fields are invalid")
        partition, iteration, start, stop = item
        _strict_string(partition, "seed partition")
        if iteration is not None:
            _strict_int(iteration, "seed iteration", minimum=1)
        _strict_int(start, "seed range start", minimum=0)
        _strict_int(stop, "seed range stop", minimum=0)
        if start > stop:
            raise ValueError("seed range start exceeds stop")
        if (partition, iteration) in seen_keys:
            raise ValueError("seed definition logical key is duplicated")
        seen_keys.add((partition, iteration))
        checked.append(item)
    for index, left in enumerate(checked):
        for right in checked[index + 1:]:
            if max(left[2], right[2]) <= min(left[3], right[3]):
                raise ValueError("seed definition ranges overlap")


validate_seed_definitions()


def require_seed_in_partition(seed: int, partition: str, iteration: int | None) -> None:
    _strict_int(seed, "seed", minimum=0)
    _strict_string(partition, "partition")
    if iteration is not None:
        _strict_int(iteration, "iteration", minimum=1)
    exact = [item for item in SEED_DEFINITIONS if item[:2] == (partition, iteration)]
    if not exact:
        known_partition = any(item[0] == partition for item in SEED_DEFINITIONS)
        word = "iteration" if known_partition else "partition"
        raise ValueError(f"seed {word} definition is unknown")
    _, _, start, stop = exact[0]
    if not start <= seed <= stop:
        raise ValueError("seed is outside the requested partition and iteration")


@dataclass(frozen=True)
class OracleSpec:
    oracle_type: str
    depth: int
    expansion_budget: int
    heuristic_identity: str
    code_hash: str

    @classmethod
    def from_dict(cls, value: Any) -> "OracleSpec":
        fields = _strict_fields(value, frozenset({
            "oracle_type", "depth", "expansion_budget", "heuristic_identity",
            "code_hash",
        }), "oracle")
        oracle_type = _strict_string(fields["oracle_type"], "oracle_type")
        if oracle_type != "bounded-search":
            raise ValueError("oracle_type is unsupported")
        depth = _strict_int(fields["depth"], "depth", minimum=1)
        budget = _strict_int(fields["expansion_budget"], "expansion_budget", minimum=1)
        heuristic = _strict_string(fields["heuristic_identity"], "heuristic_identity")
        if heuristic != "material-plus-pursuit-v1":
            raise ValueError("heuristic_identity is unsupported")
        return cls(oracle_type, depth, budget, heuristic, _hash(fields["code_hash"], "code_hash"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_type": self.oracle_type,
            "depth": self.depth,
            "expansion_budget": self.expansion_budget,
            "heuristic_identity": self.heuristic_identity,
            "code_hash": self.code_hash,
        }


@dataclass(frozen=True)
class LearnerIdentity:
    checkpoint_path: str
    checkpoint_sha256: str
    source_run: str
    source_manifest_sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "LearnerIdentity":
        fields = _strict_fields(value, frozenset({
            "checkpoint_path", "checkpoint_sha256", "source_run",
            "source_manifest_sha256",
        }), "learner")
        return cls(
            _strict_string(fields["checkpoint_path"], "checkpoint_path"),
            _hash(fields["checkpoint_sha256"], "checkpoint_sha256"),
            _strict_string(fields["source_run"], "source_run"),
            _hash(fields["source_manifest_sha256"], "source_manifest_sha256"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_run": self.source_run,
            "source_manifest_sha256": self.source_manifest_sha256,
        }


@dataclass(frozen=True)
class DatasetFileIdentity:
    path: str
    sha256: str

    @classmethod
    def from_dict(cls, value: Any) -> "DatasetFileIdentity":
        fields = _strict_fields(
            value, frozenset({"path", "sha256"}), "original dataset file",
        )
        return cls(
            _safe_relative(fields["path"], "original dataset file path"),
            _hash(fields["sha256"], "original dataset file sha256"),
        )

    def to_dict(self) -> dict[str, str]:
        return {"path": self.path, "sha256": self.sha256}


@dataclass(frozen=True)
class OriginalDatasetIdentity:
    manifest_sha256: str
    files: tuple[DatasetFileIdentity, ...]

    @classmethod
    def from_dict(cls, value: Any) -> "OriginalDatasetIdentity":
        fields = _strict_fields(
            value, frozenset({"manifest_sha256", "files"}), "original dataset",
        )
        if not isinstance(fields["files"], list) or not fields["files"]:
            raise ValueError("original dataset files must be a non-empty list")
        files = tuple(DatasetFileIdentity.from_dict(item) for item in fields["files"])
        if len({item.path for item in files}) != len(files):
            raise ValueError("original dataset file paths are duplicated")
        return cls(
            _hash(fields["manifest_sha256"], "original dataset manifest sha256"),
            files,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "files": [item.to_dict() for item in self.files],
        }


def _action_regions(contract: EnvironmentContract) -> Mapping[str, tuple[int, int]]:
    semantics = contract.semantics
    raw = semantics.get("action_regions") if isinstance(semantics, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != {"move", "attack", "deploy"}:
        raise ValueError("contract action regions are invalid")
    parsed: dict[str, tuple[int, int]] = {}
    covered = {0}
    for name in ("move", "attack", "deploy"):
        item = _strict_fields(raw[name], frozenset({"offset", "count"}), f"{name} action region")
        offset = _strict_int(item["offset"], f"{name} action offset", minimum=1)
        count = _strict_int(item["count"], f"{name} action count", minimum=1)
        indices = set(range(offset, offset + count))
        if covered & indices or max(indices) >= contract.action_size:
            raise ValueError("contract action regions overlap or exceed action size")
        covered.update(indices)
        parsed[name] = (offset, count)
    if covered != set(range(contract.action_size)):
        raise ValueError("contract action regions do not cover action size")
    return MappingProxyType(parsed)


def _command(value: Any, *, seat: int, action: int, contract: EnvironmentContract, label: str) -> Mapping[str, Any]:
    fields = _strict_fields(value, _COMMAND_FIELDS, f"{label} command")
    kind = fields["Kind"]
    if not isinstance(kind, str) or kind not in {
        "end_turn", "move", "attack", "deploy"
    }:
        raise ValueError(f"{label} command kind is invalid")
    if type(fields["Issuer"]) is not int or fields["Issuer"] != seat:
        raise ValueError(f"{label} command issuer does not match seat")
    for name in ("ActorId", "TargetId", "Q", "R"):
        if fields[name] is not None and type(fields[name]) is not int:
            raise ValueError(f"{label} command {name} is invalid")
    for name in ("ActorId", "TargetId"):
        if fields[name] is not None and fields[name] < 0:
            raise ValueError(f"{label} command {name} must be nonnegative")
    shape = (
        (kind == "end_turn" and all(fields[name] is None for name in ("ActorId", "TargetId", "Q", "R")))
        or (kind == "move" and fields["ActorId"] is not None and fields["TargetId"] is None and fields["Q"] is not None and fields["R"] is not None)
        or (kind == "attack" and fields["ActorId"] is not None and fields["TargetId"] is not None and fields["Q"] is None and fields["R"] is None)
        or (kind == "deploy" and fields["ActorId"] is None and fields["TargetId"] is None and fields["Q"] is not None and fields["R"] is not None)
    )
    if not shape:
        raise ValueError(f"{label} command shape is invalid")
    if action == 0:
        expected = "end_turn"
    else:
        matches = [
            name for name, (offset, count) in _action_regions(contract).items()
            if offset <= action < offset + count
        ]
        expected = matches[0] if len(matches) == 1 else ""
    if kind != expected:
        raise ValueError(f"{label} command kind does not match action region")
    return MappingProxyType(dict(fields))


@dataclass(frozen=True)
class DaggerRow:
    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    learner_action: int
    learner_command: Mapping[str, Any]
    teacher_action: int
    teacher_command: Mapping[str, Any]
    reason_bits: int
    state_hash: str
    normalized_advantage: float
    opponent_living_unit_count: int
    productive_legal_action_count: int
    seat: int
    round: int
    decision_index: int
    disagreement: bool
    oracle_actual_expansion_count: int

    @classmethod
    def from_dict(
        cls, value: Any, *, contract: EnvironmentContract, oracle: OracleSpec
    ) -> "DaggerRow":
        fields = _strict_fields(value, _ROW_FIELDS, "DAgger row")
        raw_observation = fields["observation"]
        if not isinstance(raw_observation, list) or len(raw_observation) != contract.observation_size:
            raise ValueError("DAgger row observation shape is invalid")
        if any(type(item) is not float for item in raw_observation):
            raise ValueError("DAgger row observation values are invalid")
        observation = tuple(
            _strict_json_float_for_float32(
                item, "DAgger row observation float32 value",
            )
            for item in raw_observation
        )
        raw_mask = fields["legal_mask"]
        if (
            not isinstance(raw_mask, list)
            or len(raw_mask) != contract.action_size
            or any(type(item) is not bool for item in raw_mask)
        ):
            raise ValueError("DAgger row legal mask shape or values are invalid")
        mask = tuple(raw_mask)
        seat = _strict_int32(fields["seat"], "DAgger row seat")
        if seat not in {0, 1}:
            raise ValueError("DAgger row seat is invalid")
        actions: dict[str, int] = {}
        for name in ("learner_action", "teacher_action"):
            action = _strict_int32(fields[name], f"DAgger row {name}", minimum=0)
            if action >= contract.action_size or not mask[action]:
                raise ValueError(f"DAgger row {name} is not legal")
            actions[name] = action
        learner_command = _command(
            fields["learner_command"], seat=seat, action=actions["learner_action"],
            contract=contract, label="learner",
        )
        teacher_command = _command(
            fields["teacher_command"], seat=seat, action=actions["teacher_action"],
            contract=contract, label="teacher",
        )
        reason_bits = _strict_int(fields["reason_bits"], "DAgger row reason bits", minimum=1)
        if reason_bits & ~0b1111:
            raise ValueError("DAgger row reason bits are invalid")
        state_hash = fields["state_hash"]
        if not isinstance(state_hash, str) or _STATE_HASH_PATTERN.fullmatch(state_hash) is None:
            raise ValueError("DAgger row state hash is invalid")
        advantage = _strict_json_float_for_float32(
            fields["normalized_advantage"], "DAgger row normalized advantage",
        )
        disagreement = fields["disagreement"]
        expected_disagreement = actions["learner_action"] != actions["teacher_action"]
        if type(disagreement) is not bool or disagreement is not expected_disagreement:
            raise ValueError("DAgger row disagreement is inconsistent")
        actual = _strict_int(
            fields["oracle_actual_expansion_count"],
            "DAgger row actual expansion count", minimum=0,
        )
        if actual > oracle.expansion_budget:
            raise ValueError("DAgger row actual expansion count exceeds oracle expansion budget")
        return cls(
            observation, mask, actions["learner_action"], learner_command,
            actions["teacher_action"], teacher_command, reason_bits, state_hash,
            float(advantage),
            _strict_int(fields["opponent_living_unit_count"], "opponent living unit count", minimum=0),
            _strict_int(fields["productive_legal_action_count"], "productive legal action count", minimum=0),
            seat,
            _strict_int32(fields["round"], "DAgger row round", minimum=0),
            _strict_int32(
                fields["decision_index"], "DAgger row decision index", minimum=0,
            ),
            disagreement,
            actual,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation": list(self.observation),
            "legal_mask": list(self.legal_mask),
            "learner_action": self.learner_action,
            "learner_command": dict(self.learner_command),
            "teacher_action": self.teacher_action,
            "teacher_command": dict(self.teacher_command),
            "reason_bits": self.reason_bits,
            "state_hash": self.state_hash,
            "normalized_advantage": self.normalized_advantage,
            "opponent_living_unit_count": self.opponent_living_unit_count,
            "productive_legal_action_count": self.productive_legal_action_count,
            "seat": self.seat,
            "round": self.round,
            "decision_index": self.decision_index,
            "disagreement": self.disagreement,
            "oracle_actual_expansion_count": self.oracle_actual_expansion_count,
        }


@dataclass(frozen=True)
class DaggerGame:
    game_id: int
    partition: str
    iteration: int
    map_seed: int
    episode_seed: int
    schedule_index: int
    profile: str
    reference_seat: int
    learner_seat: int
    opponent: str
    outcome: str
    transition_count: int
    trace_path: str
    replay_path: str

    @classmethod
    def from_dict(cls, value: Any) -> "DaggerGame":
        fields = _strict_fields(value, _GAME_FIELDS, "DAgger game")
        partition = _strict_string(fields["partition"], "DAgger game partition")
        if partition not in {"train", "validation"}:
            raise ValueError("DAgger game partition is invalid")
        iteration = _strict_int(fields["iteration"], "DAgger game iteration", minimum=1)
        if iteration not in {1, 2, 3}:
            raise ValueError("DAgger game iteration is invalid")
        map_seed = _strict_int(fields["map_seed"], "DAgger game map seed", minimum=0)
        require_seed_in_partition(map_seed, partition, iteration)
        profile = _strict_string(fields["profile"], "DAgger game profile")
        if profile not in _PROFILES:
            raise ValueError("DAgger game profile is invalid")
        reference_seat = _strict_int(fields["reference_seat"], "reference seat")
        learner_seat = _strict_int(fields["learner_seat"], "learner seat")
        if reference_seat not in {0, 1} or learner_seat not in {0, 1}:
            raise ValueError("DAgger game seats are invalid")
        if fields["opponent"] != "random":
            raise ValueError("DAgger game opponent must be random")
        outcome = _strict_string(fields["outcome"], "DAgger game outcome")
        if outcome not in {"win", "loss", "draw"}:
            raise ValueError("DAgger game outcome is invalid")
        return cls(
            _strict_int(fields["game_id"], "DAgger game id", minimum=0),
            partition,
            iteration,
            map_seed,
            _strict_int(fields["episode_seed"], "DAgger game episode seed", minimum=0),
            _strict_int(fields["schedule_index"], "DAgger game schedule index", minimum=0),
            profile,
            reference_seat,
            learner_seat,
            fields["opponent"],
            outcome,
            _strict_int(fields["transition_count"], "DAgger game transition count", minimum=0),
            _safe_relative(fields["trace_path"], "trace_path"),
            _safe_relative(fields["replay_path"], "replay_path"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "partition": self.partition,
            "iteration": self.iteration,
            "map_seed": self.map_seed,
            "episode_seed": self.episode_seed,
            "schedule_index": self.schedule_index,
            "profile": self.profile,
            "reference_seat": self.reference_seat,
            "learner_seat": self.learner_seat,
            "opponent": self.opponent,
            "outcome": self.outcome,
            "transition_count": self.transition_count,
            "trace_path": self.trace_path,
            "replay_path": self.replay_path,
        }


@dataclass(frozen=True)
class OverlayDefinition:
    partition: str
    iteration: int
    observation_size: int
    action_size: int
    action_regions: tuple[tuple[str, int, int], ...]
    oracle: OracleSpec
    learner: LearnerIdentity
    original_dataset: OriginalDatasetIdentity
    scenario_hash: str
    contract_hash: str
    encoding_hash: str
    repository_hash: str
    panel_hash: str
    schedule_hash: str
    label_target: int
    game_ceiling: int

    @classmethod
    def from_dict(cls, value: Any) -> "OverlayDefinition":
        fields = _strict_fields(value, _DEFINITION_FIELDS, "overlay definition")
        partition = _strict_string(fields["partition"], "overlay partition")
        if partition not in {"train", "validation"}:
            raise ValueError("overlay partition is invalid")
        iteration = _strict_int(fields["iteration"], "overlay iteration", minimum=1)
        if iteration not in {1, 2, 3}:
            raise ValueError("overlay iteration is invalid")
        observation_size = _strict_int(
            fields["observation_size"], "overlay observation_size", minimum=1,
        )
        action_size = _strict_int(
            fields["action_size"], "overlay action_size", minimum=1,
        )
        contract = EnvironmentContract(
            version="tactical-v2",
            contract_hash=_hash(fields["contract_hash"], "contract_hash"),
            encoding_hash=_hash(fields["encoding_hash"], "encoding_hash"),
            observation_size=observation_size,
            action_size=action_size,
            board={},
            roster=[],
            reward={},
            semantics={"action_regions": fields["action_regions"]},
        )
        regions = tuple(
            (name, offset, count)
            for name, (offset, count) in _action_regions(contract).items()
        )
        return cls(
            partition, iteration, observation_size, action_size, regions,
            OracleSpec.from_dict(fields["oracle"]),
            LearnerIdentity.from_dict(fields["learner"]),
            OriginalDatasetIdentity.from_dict(fields["original_dataset"]),
            _hash(fields["scenario_hash"], "scenario_hash"),
            contract.contract_hash,
            contract.encoding_hash,
            _hash(fields["repository_hash"], "repository_hash"),
            _hash(fields["panel_hash"], "panel_hash"),
            _hash(fields["schedule_hash"], "schedule_hash"),
            _strict_int(fields["label_target"], "label_target", minimum=0),
            _strict_int(fields["game_ceiling"], "game_ceiling", minimum=1),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "partition": self.partition,
            "iteration": self.iteration,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
            "action_regions": {
                name: {"offset": offset, "count": count}
                for name, offset, count in self.action_regions
            },
            "oracle": self.oracle.to_dict(),
            "learner": self.learner.to_dict(),
            "original_dataset": self.original_dataset.to_dict(),
            "scenario_hash": self.scenario_hash,
            "contract_hash": self.contract_hash,
            "encoding_hash": self.encoding_hash,
            "repository_hash": self.repository_hash,
            "panel_hash": self.panel_hash,
            "schedule_hash": self.schedule_hash,
            "label_target": self.label_target,
            "game_ceiling": self.game_ceiling,
        }


@dataclass(frozen=True)
class ScheduledDuel:
    schedule_index: int
    map_seed: int
    episode_seed: int
    profile: str
    reference_seat: int
    learner_seat: int

    def __post_init__(self) -> None:
        _strict_int(self.schedule_index, "scheduled duel index", minimum=0)
        _strict_int(self.map_seed, "scheduled duel map seed", minimum=0)
        _strict_int(self.episode_seed, "scheduled duel episode seed", minimum=0)
        if self.profile not in _PROFILES:
            raise ValueError("scheduled duel profile is invalid")
        if (
            type(self.reference_seat) is not int
            or type(self.learner_seat) is not int
            or self.reference_seat not in {0, 1}
            or self.learner_seat not in {0, 1}
        ):
            raise ValueError("scheduled duel seats are invalid")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schedule_index": self.schedule_index,
            "map_seed": self.map_seed,
            "episode_seed": self.episode_seed,
            "profile": self.profile,
            "reference_seat": self.reference_seat,
            "learner_seat": self.learner_seat,
        }


def _freeze_contract_value(value: Any, label: str) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{label} keys must be strings")
            frozen[key] = _freeze_contract_value(item, label)
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_contract_value(item, label) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ValueError(f"{label} contains an unsupported value")


def _frozen_collection_contract(contract: EnvironmentContract) -> EnvironmentContract:
    if not isinstance(contract, EnvironmentContract) or contract.version != "tactical-v2":
        raise ValueError("collection requires a tactical-v2 contract")
    _hash(contract.contract_hash, "collection contract hash")
    _hash(contract.encoding_hash, "collection encoding hash")
    _strict_int(contract.observation_size, "collection observation size", minimum=1)
    _strict_int(contract.action_size, "collection action size", minimum=1)
    frozen = EnvironmentContract(
        version=contract.version,
        contract_hash=contract.contract_hash,
        encoding_hash=contract.encoding_hash,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
        board=_freeze_contract_value(contract.board, "collection board"),
        roster=tuple(contract.roster),  # type: ignore[arg-type]
        reward=_freeze_contract_value(contract.reward, "collection reward"),
        semantics=_freeze_contract_value(contract.semantics, "collection semantics"),
    )
    _action_regions(frozen)
    return frozen


def _declared_collection_profiles(contract: EnvironmentContract) -> tuple[str, ...]:
    raw = contract.semantics.get("start_profiles")
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("collection contract has no declared start profiles")
    profiles: list[str] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise ValueError(f"collection start profile {index} is invalid")
        profile = item.get("id")
        if not isinstance(profile, str) or profile not in _PROFILES:
            raise ValueError(f"collection start profile {index} is invalid")
        profiles.append(profile)
    if (
        profiles[0] != "standard-3v3"
        or len(profiles) != len(set(profiles))
        or set(profiles) != set(_PROFILES)
    ):
        raise ValueError("collection start profile catalog is incomplete or duplicated")
    return tuple(profiles)


def _seed_range(partition: str, iteration: int) -> tuple[int, int]:
    matches = [
        (start, stop)
        for candidate, candidate_iteration, start, stop in SEED_DEFINITIONS
        if (candidate, candidate_iteration) == (partition, iteration)
    ]
    if len(matches) != 1:
        raise ValueError("collection seed partition or iteration is unknown")
    return matches[0]


def _collection_schedule(
    *,
    partition: str,
    iteration: int,
    game_ceiling: int,
    conversion_profiles: tuple[str, ...],
) -> tuple[ScheduledDuel, ...]:
    if game_ceiling % 2:
        raise ValueError("collection game ceiling must contain complete reciprocal pairs")
    start, stop = _seed_range(partition, iteration)
    pair_count = game_ceiling // 2
    if start + pair_count - 1 > stop:
        raise ValueError("collection seed range is too small for the complete schedule")
    scheduled: list[ScheduledDuel] = []
    standard_residual = 0
    conversion_index = 0
    for pair_index in range(pair_count):
        standard_residual += 7
        standard_count = standard_residual // 10
        standard_residual -= standard_count * 10
        if standard_count:
            profile = "standard-3v3"
        else:
            profile = conversion_profiles[conversion_index % len(conversion_profiles)]
            conversion_index += 1
        seed = start + pair_index
        for learner_seat in (0, 1):
            scheduled.append(ScheduledDuel(
                schedule_index=pair_index,
                map_seed=seed,
                episode_seed=seed,
                profile=profile,
                reference_seat=learner_seat,
                learner_seat=learner_seat,
            ))
    return tuple(scheduled)


def _schedule_identity(schedule: Sequence[ScheduledDuel]) -> str:
    payload = json.dumps(
        [duel.to_dict() for duel in schedule],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CollectionDefinition:
    contract: EnvironmentContract
    partition: str
    iteration: int
    oracle: OracleSpec
    learner: LearnerIdentity
    original_dataset: OriginalDatasetIdentity
    scenario_hash: str
    repository_hash: str
    panel_hash: str
    conversion_profiles: tuple[str, ...]
    schedule: tuple[ScheduledDuel, ...]
    schedule_hash: str
    label_target: int
    game_ceiling: int
    overlay_definition: OverlayDefinition

    @classmethod
    def create(
        cls,
        *,
        contract: EnvironmentContract,
        partition: str,
        iteration: int,
        oracle: OracleSpec,
        learner: LearnerIdentity,
        original_dataset: OriginalDatasetIdentity,
        scenario_hash: str,
        repository_hash: str,
        panel_hash: str,
    ) -> "CollectionDefinition":
        if partition not in {"train", "validation"}:
            raise ValueError("collection partition is invalid")
        if type(iteration) is not int or iteration not in {1, 2, 3}:
            raise ValueError("collection iteration is invalid")
        frozen_contract = _frozen_collection_contract(contract)
        canonical_oracle = OracleSpec.from_dict(oracle.to_dict())
        canonical_learner = LearnerIdentity.from_dict(learner.to_dict())
        canonical_dataset = OriginalDatasetIdentity.from_dict(
            original_dataset.to_dict()
        )
        declared = _declared_collection_profiles(frozen_contract)
        conversions = tuple(
            profile for profile in declared if profile != "standard-3v3"
        )
        label_target, game_ceiling = (
            (20_000, 2_000) if partition == "train" else (2_000, 200)
        )
        schedule = _collection_schedule(
            partition=partition,
            iteration=iteration,
            game_ceiling=game_ceiling,
            conversion_profiles=conversions,
        )
        schedule_hash = _schedule_identity(schedule)
        overlay = OverlayDefinition.from_dict({
            "partition": partition,
            "iteration": iteration,
            "observation_size": frozen_contract.observation_size,
            "action_size": frozen_contract.action_size,
            "action_regions": _regions_to_dict(_action_regions(frozen_contract)),
            "oracle": canonical_oracle.to_dict(),
            "learner": canonical_learner.to_dict(),
            "original_dataset": canonical_dataset.to_dict(),
            "scenario_hash": _hash(scenario_hash, "collection scenario hash"),
            "contract_hash": frozen_contract.contract_hash,
            "encoding_hash": frozen_contract.encoding_hash,
            "repository_hash": _hash(
                repository_hash, "collection repository hash"
            ),
            "panel_hash": _hash(panel_hash, "collection panel hash"),
            "schedule_hash": schedule_hash,
            "label_target": label_target,
            "game_ceiling": game_ceiling,
        })
        return cls(
            frozen_contract,
            partition,
            iteration,
            canonical_oracle,
            canonical_learner,
            canonical_dataset,
            overlay.scenario_hash,
            overlay.repository_hash,
            overlay.panel_hash,
            conversions,
            schedule,
            schedule_hash,
            label_target,
            game_ceiling,
            overlay,
        )


@dataclass(frozen=True)
class GameArtifactIdentity:
    path: str
    sha256: str
    byte_size: int
    game_id: int
    row_count: int
    content_identity: str

    @classmethod
    def from_dict(cls, value: Any) -> "GameArtifactIdentity":
        fields = _strict_fields(
            value, _GAME_DESCRIPTOR_FIELDS, "overlay game descriptor",
        )
        return cls(
            _safe_relative(fields["path"], "game descriptor path"),
            _hash(fields["sha256"], "game descriptor sha256"),
            _strict_int(fields["byte_size"], "game descriptor byte_size", minimum=1),
            _strict_int(fields["game_id"], "game descriptor game_id", minimum=0),
            _strict_int(fields["row_count"], "game descriptor row_count", minimum=0),
            _hash(fields["content_identity"], "game descriptor content_identity"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path, "sha256": self.sha256, "byte_size": self.byte_size,
            "game_id": self.game_id, "row_count": self.row_count,
            "content_identity": self.content_identity,
        }


@dataclass(frozen=True)
class DaggerOverlayManifest:
    definition: OverlayDefinition
    games: tuple[GameArtifactIdentity, ...]
    row_count: int
    content_identity: str

    @classmethod
    def from_dict(cls, value: Any) -> "DaggerOverlayManifest":
        fields = _strict_fields(value, _OVERLAY_FIELDS, "DAgger overlay")
        if _strict_int(fields["schema_version"], "overlay schema_version") != OVERLAY_SCHEMA_VERSION:
            raise ValueError("DAgger overlay schema version is invalid")
        if _strict_string(fields["status"], "overlay status") != "completed":
            raise ValueError("DAgger overlay is not completed")
        definition = OverlayDefinition.from_dict({
            key: fields[key] for key in _DEFINITION_FIELDS
        })
        game_count = _strict_int(fields["game_count"], "overlay game_count", minimum=1)
        row_count = _strict_int(fields["row_count"], "overlay row_count", minimum=0)
        if not isinstance(fields["games"], list) or len(fields["games"]) != game_count:
            raise ValueError("overlay game descriptors are invalid")
        games = tuple(GameArtifactIdentity.from_dict(item) for item in fields["games"])
        if tuple(item.game_id for item in games) != tuple(range(game_count)):
            raise ValueError("overlay game descriptor IDs are not canonical")
        if sum(item.row_count for item in games) != row_count:
            raise ValueError("overlay descriptor row_count is inconsistent")
        identity = _hash(fields["content_identity"], "overlay content_identity")
        if _content_identity(fields) != identity:
            raise ValueError("DAgger overlay content identity is invalid")
        return cls(definition, games, row_count, identity)


@dataclass(frozen=True, init=False)
class DaggerOverlay:
    root: Path
    manifest: DaggerOverlayManifest
    games: tuple[DaggerGame, ...]

    def __new__(cls) -> "DaggerOverlay":
        raise TypeError("DaggerOverlay instances require physical reopen validation")

    @classmethod
    def _create(
        cls,
        *,
        root: Path,
        manifest: DaggerOverlayManifest,
        games: Sequence[DaggerGame],
    ) -> "DaggerOverlay":
        canonical_games = tuple(games)
        if len(canonical_games) != len(manifest.games):
            raise ValueError("physical overlay game count is inconsistent")
        for descriptor, game in zip(manifest.games, canonical_games, strict=True):
            if (
                descriptor.game_id != game.game_id
                or game.partition != manifest.definition.partition
                or game.iteration != manifest.definition.iteration
            ):
                raise ValueError("physical overlay game metadata is inconsistent")
        instance = object.__new__(cls)
        object.__setattr__(instance, "root", Path(root))
        object.__setattr__(instance, "manifest", manifest)
        object.__setattr__(instance, "games", canonical_games)
        return instance

    @property
    def definition(self) -> OverlayDefinition:
        return self.manifest.definition

    @property
    def partition(self) -> str:
        return self.definition.partition

    @property
    def iteration(self) -> int:
        return self.definition.iteration

    @property
    def row_count(self) -> int:
        return self.manifest.row_count

    @property
    def content_identity(self) -> str:
        return self.manifest.content_identity


class DaggerOverlayWriter:
    def __init__(
        self,
        root: Path,
        *,
        contract: EnvironmentContract,
        partition: str,
        iteration: int,
        oracle: OracleSpec,
        learner: LearnerIdentity,
        original_dataset: OriginalDatasetIdentity,
        scenario_hash: str,
        repository_hash: str,
        panel_hash: str,
        schedule_hash: str,
        label_target: int,
        game_ceiling: int,
    ) -> None:
        self.root = root
        self.contract = contract
        self.partition = partition
        self.iteration = iteration
        self.oracle = oracle
        self.learner = learner
        self.original_dataset = original_dataset
        self.scenario_hash = scenario_hash
        self.repository_hash = repository_hash
        self.panel_hash = panel_hash
        self.schedule_hash = schedule_hash
        self.label_target = label_target
        self.game_ceiling = game_ceiling
        self._games: list[DaggerGame] = []
        self._descriptors: list[dict[str, Any]] = []
        self._row_count = 0
        self._row_keys: set[tuple[int, int]] = set()

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        contract: EnvironmentContract,
        partition: str,
        iteration: int,
        oracle: OracleSpec,
        learner: LearnerIdentity,
        original_dataset: OriginalDatasetIdentity,
        scenario_hash: str,
        repository_hash: str,
        panel_hash: str,
        schedule_hash: str,
        label_target: int,
        game_ceiling: int,
    ) -> "DaggerOverlayWriter":
        root = Path(root)
        if not isinstance(original_dataset, OriginalDatasetIdentity):
            raise ValueError("original dataset identity is invalid")
        original_dataset = OriginalDatasetIdentity.from_dict(
            original_dataset.to_dict()
        )
        if contract.version != "tactical-v2":
            raise ValueError("DAgger overlays require a tactical-v2 contract")
        if partition not in {"train", "validation"}:
            raise ValueError("overlay partition is invalid")
        if type(iteration) is not int or iteration not in {1, 2, 3}:
            raise ValueError("overlay iteration is invalid")
        for value, label in (
            (scenario_hash, "scenario_hash"),
            (contract.contract_hash, "contract_hash"),
            (contract.encoding_hash, "encoding_hash"),
            (repository_hash, "repository_hash"),
            (panel_hash, "panel_hash"),
            (schedule_hash, "schedule_hash"),
        ):
            _hash(value, label)
        _action_regions(contract)
        _strict_int(label_target, "overlay label_target", minimum=0)
        _strict_int(game_ceiling, "overlay game_ceiling", minimum=1)
        checkpoint = Path(learner.checkpoint_path)
        if not checkpoint.is_file() or _sha256_file(checkpoint) != learner.checkpoint_sha256:
            raise ValueError("learner checkpoint SHA-256 does not match identity")
        if root.exists():
            if any(root.iterdir()):
                raise ValueError("overlay staging root is not empty")
        else:
            root.mkdir(parents=True)
        for child in ("shards", "games", "evidence"):
            (root / child).mkdir(exist_ok=True)
        return cls(
            root, contract=contract, partition=partition, iteration=iteration,
            oracle=oracle, learner=learner, original_dataset=original_dataset,
            scenario_hash=scenario_hash,
            repository_hash=repository_hash, panel_hash=panel_hash,
            schedule_hash=schedule_hash, label_target=label_target,
            game_ceiling=game_ceiling,
        )

    def append_game(
        self,
        game: DaggerGame | Mapping[str, Any],
        rows: Sequence[DaggerRow | Mapping[str, Any]],
    ) -> None:
        if not isinstance(game, DaggerGame):
            game = DaggerGame.from_dict(game)
        if game.partition != self.partition or game.iteration != self.iteration:
            raise ValueError("game partition or iteration does not match overlay")
        if game.game_id != len(self._games):
            raise ValueError("game_id must be contiguous and unique")
        parsed: list[DaggerRow] = []
        decision_indices: set[int] = set()
        state_hashes: set[str] = set()
        for raw in rows:
            row = raw if isinstance(raw, DaggerRow) else DaggerRow.from_dict(
                raw, contract=self.contract, oracle=self.oracle
            )
            key = (game.game_id, row.decision_index)
            if key in self._row_keys or row.decision_index in decision_indices:
                raise ValueError("duplicate DAgger row identity")
            if row.state_hash in state_hashes:
                raise ValueError("duplicate canonical state hash in episode")
            if row.seat != game.learner_seat:
                raise ValueError("DAgger row seat does not match game learner seat")
            decision_indices.add(row.decision_index)
            state_hashes.add(row.state_hash)
            parsed.append(row)

        trace = _contained_file(self.root, game.trace_path, "trace_path")
        replay = _contained_file(self.root, game.replay_path, "replay_path")
        _validate_trace_file(trace, game)
        if not replay.is_file() or replay.stat().st_size <= 0:
            raise ValueError("DAgger replay evidence is missing or empty")

        shard_relative = f"shards/game-{game.game_id:08d}.npz"
        shard_path = self.root / shard_relative
        _atomic_npz(
            shard_path,
            observations=np.asarray(
                [row.observation for row in parsed], dtype=np.float32,
            ).reshape(len(parsed), self.contract.observation_size),
            packed_masks=np.packbits(
                np.asarray(
                    [row.legal_mask for row in parsed], dtype=np.uint8,
                ).reshape(len(parsed), self.contract.action_size),
                axis=1, bitorder="little",
            ),
            actions=np.asarray([row.teacher_action for row in parsed], dtype=np.int32),
            learner_actions=np.asarray([row.learner_action for row in parsed], dtype=np.int32),
            seats=np.asarray([row.seat for row in parsed], dtype=np.int32),
            rounds=np.asarray([row.round for row in parsed], dtype=np.int32),
            decision_indices=np.asarray([row.decision_index for row in parsed], dtype=np.int32),
            reason_bits=np.asarray([row.reason_bits for row in parsed], dtype=np.uint8),
            state_hashes=np.asarray(
                [row.state_hash.encode("ascii") for row in parsed], dtype="S64"
            ),
        )
        shard = _artifact_descriptor(
            shard_relative, shard_path, row_count=len(parsed)
        )
        common_evidence = {
            "seed": game.map_seed,
            "reference_seat": game.reference_seat,
            "learner_seat": game.learner_seat,
            "profile": game.profile,
            "outcome": game.outcome,
            "transition_count": game.transition_count,
        }
        trace_descriptor = {
            "path": game.trace_path,
            "sha256": _sha256_file(trace),
            "byte_size": trace.stat().st_size,
            **common_evidence,
        }
        replay_descriptor = {
            "path": game.replay_path,
            "sha256": _sha256_file(replay),
            "byte_size": replay.stat().st_size,
            **common_evidence,
        }
        row_metadata = [
            {
                "learner_command": dict(row.learner_command),
                "teacher_command": dict(row.teacher_command),
                "normalized_advantage": row.normalized_advantage,
                "opponent_living_unit_count": row.opponent_living_unit_count,
                "productive_legal_action_count": row.productive_legal_action_count,
                "disagreement": row.disagreement,
                "oracle_actual_expansion_count": row.oracle_actual_expansion_count,
            }
            for row in parsed
        ]
        game_manifest: dict[str, Any] = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            **{
                key: value for key, value in game.to_dict().items()
                if key not in {"trace_path", "replay_path"}
            },
            "oracle": self.oracle.to_dict(),
            "learner": self.learner.to_dict(),
            "original_dataset": self.original_dataset.to_dict(),
            "scenario_hash": self.scenario_hash,
            "contract_hash": self.contract.contract_hash,
            "encoding_hash": self.contract.encoding_hash,
            "repository_hash": self.repository_hash,
            "panel_hash": self.panel_hash,
            "schedule_hash": self.schedule_hash,
            "label_target": self.label_target,
            "game_ceiling": self.game_ceiling,
            "row_count": len(parsed),
            "shard": shard,
            "trace": trace_descriptor,
            "replay": replay_descriptor,
            "row_metadata": row_metadata,
        }
        game_manifest["content_identity"] = _content_identity(game_manifest)
        relative = f"games/game-{game.game_id:08d}.json"
        path = self.root / relative
        atomic_write_json(path, game_manifest)
        self._games.append(game)
        self._descriptors.append(_artifact_descriptor(
            relative, path, game_id=game.game_id, row_count=len(parsed),
            content_identity=game_manifest["content_identity"],
        ))
        self._row_count += len(parsed)
        self._row_keys.update((game.game_id, row.decision_index) for row in parsed)

    def seal(self) -> DaggerOverlay:
        _validate_reciprocal_games(self._games)
        manifest: dict[str, Any] = {
            "schema_version": OVERLAY_SCHEMA_VERSION,
            "status": "completed",
            "partition": self.partition,
            "iteration": self.iteration,
            "observation_size": self.contract.observation_size,
            "action_size": self.contract.action_size,
            "action_regions": _regions_to_dict(_action_regions(self.contract)),
            "oracle": self.oracle.to_dict(),
            "learner": self.learner.to_dict(),
            "original_dataset": self.original_dataset.to_dict(),
            "scenario_hash": self.scenario_hash,
            "contract_hash": self.contract.contract_hash,
            "encoding_hash": self.contract.encoding_hash,
            "repository_hash": self.repository_hash,
            "panel_hash": self.panel_hash,
            "schedule_hash": self.schedule_hash,
            "label_target": self.label_target,
            "game_ceiling": self.game_ceiling,
            "game_count": len(self._games),
            "row_count": self._row_count,
            "games": self._descriptors,
        }
        manifest["content_identity"] = _content_identity(manifest)
        atomic_write_json(self.root / "manifest.json", manifest)
        return open_dagger_overlay(self.root)


def open_dagger_overlay(root: Path) -> DaggerOverlay:
    root = Path(root)
    manifest_path = _contained_file(root, "manifest.json", "outer manifest")
    manifest = _read_json(manifest_path)
    logical = DaggerOverlayManifest.from_dict(manifest)
    _strict_int(manifest["observation_size"], "observation_size", minimum=1)
    _strict_int(manifest["action_size"], "action_size", minimum=1)
    for name in (
        "scenario_hash", "contract_hash", "encoding_hash", "repository_hash",
        "panel_hash", "schedule_hash",
    ):
        _hash(manifest[name], name)
    oracle = OracleSpec.from_dict(manifest["oracle"])
    learner = LearnerIdentity.from_dict(manifest["learner"])
    original_dataset = OriginalDatasetIdentity.from_dict(manifest["original_dataset"])
    checkpoint = Path(learner.checkpoint_path)
    if not checkpoint.is_file() or _sha256_file(checkpoint) != learner.checkpoint_sha256:
        raise ValueError("learner checkpoint SHA-256 changed")
    contract = EnvironmentContract(
        version="tactical-v2",
        contract_hash=manifest["contract_hash"],
        encoding_hash=manifest["encoding_hash"],
        observation_size=manifest["observation_size"],
        action_size=manifest["action_size"],
        board={},
        roster=[],
        reward={},
        semantics={"action_regions": manifest["action_regions"]},
    )
    if _regions_to_dict(_action_regions(contract)) != manifest["action_regions"]:
        raise ValueError("overlay action regions are not canonical")
    if not isinstance(manifest["games"], list):
        raise ValueError("overlay games must be a list")

    games: list[DaggerGame] = []
    row_keys: set[tuple[int, int]] = set()
    episode_hashes: dict[int, set[str]] = {}
    seen_game_paths: set[str] = set()
    seen_shards: set[str] = set()
    evidence_paths: set[str] = set()
    total_rows = 0
    for expected_game_id, descriptor_raw in enumerate(manifest["games"]):
        descriptor = _strict_fields(
            descriptor_raw, _GAME_DESCRIPTOR_FIELDS, "overlay game descriptor"
        )
        if _strict_int(descriptor["game_id"], "game descriptor id", minimum=0) != expected_game_id:
            raise ValueError("overlay game descriptor id is invalid")
        game_path = _verified_artifact(root, descriptor, seen_game_paths)
        game_manifest = _read_json(game_path)
        _strict_fields(game_manifest, _GAME_MANIFEST_FIELDS, "game manifest")
        if _strict_int(game_manifest["schema_version"], "game schema_version") != OVERLAY_SCHEMA_VERSION:
            raise ValueError("game manifest schema version is invalid")
        if _content_identity(game_manifest) != game_manifest["content_identity"]:
            raise ValueError("game manifest content identity is invalid")
        if descriptor["content_identity"] != game_manifest["content_identity"]:
            raise ValueError("game descriptor content identity does not match")
        game = DaggerGame.from_dict({
            **{
                key: game_manifest[key] for key in _GAME_FIELDS
                if key not in {"trace_path", "replay_path"}
            },
            "trace_path": game_manifest["trace"]["path"],
            "replay_path": game_manifest["replay"]["path"],
        })
        if game.game_id != expected_game_id:
            raise ValueError("game manifest id is invalid")
        if (
            game.partition != logical.definition.partition
            or game.iteration != logical.definition.iteration
        ):
            raise ValueError("game manifest partition or iteration is invalid")
        for key, expected in (
            ("oracle", oracle.to_dict()),
            ("learner", learner.to_dict()),
            ("original_dataset", original_dataset.to_dict()),
            ("scenario_hash", manifest["scenario_hash"]),
            ("contract_hash", manifest["contract_hash"]),
            ("encoding_hash", manifest["encoding_hash"]),
            ("repository_hash", manifest["repository_hash"]),
            ("panel_hash", manifest["panel_hash"]),
            ("schedule_hash", manifest["schedule_hash"]),
            ("label_target", manifest["label_target"]),
            ("game_ceiling", manifest["game_ceiling"]),
        ):
            if game_manifest[key] != expected:
                raise ValueError(f"game manifest {key} does not match overlay")
        row_count = _strict_int(game_manifest["row_count"], "game row_count", minimum=0)
        if _strict_int(descriptor["row_count"], "game descriptor row_count", minimum=0) != row_count:
            raise ValueError("game descriptor row count does not match")
        _validate_evidence(root, game_manifest["trace"], game, trace=True)
        _validate_evidence(root, game_manifest["replay"], game, trace=False)
        for item in (game_manifest["trace"], game_manifest["replay"]):
            evidence_relative = _safe_relative(item["path"], "evidence path")
            if evidence_relative in evidence_paths:
                raise ValueError("evidence path is duplicated")
            evidence_paths.add(evidence_relative)
        shard_raw = _strict_fields(
            game_manifest["shard"], _SHARD_DESCRIPTOR_FIELDS, "shard descriptor"
        )
        if _strict_int(shard_raw["row_count"], "shard row_count", minimum=0) != row_count:
            raise ValueError("shard row count does not match game")
        if _content_identity(shard_raw) != shard_raw["content_identity"]:
            raise ValueError("shard descriptor content identity is invalid")
        shard_path = _verified_artifact(root, shard_raw, seen_shards)
        arrays = _read_and_validate_shard(
            shard_path, row_count=row_count, contract=contract, game=game,
            oracle=oracle, row_metadata=game_manifest["row_metadata"],
        )
        for decision_index, state_hash in zip(
            arrays["decision_indices"], arrays["state_hashes"], strict=True
        ):
            key = (game.game_id, int(decision_index))
            decoded = bytes(state_hash).decode("ascii")
            if key in row_keys:
                raise ValueError("duplicate physical DAgger row identity")
            hashes = episode_hashes.setdefault(game.game_id, set())
            if decoded in hashes:
                raise ValueError("duplicate physical canonical state hash in episode")
            row_keys.add(key)
            hashes.add(decoded)
        games.append(game)
        total_rows += row_count

    if len(games) != manifest["game_count"] or total_rows != manifest["row_count"]:
        raise ValueError("overlay physical counts do not match manifest")
    _validate_reciprocal_games(games)
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    expected_files = {
        "manifest.json", *seen_game_paths, *seen_shards, *evidence_paths,
    }
    if actual_files != expected_files:
        raise ValueError("overlay contains missing or unowned physical files")
    return DaggerOverlay._create(root=root, manifest=logical, games=games)


def _require_expected_definition(
    overlay: DaggerOverlay, expected: OverlayDefinition,
) -> None:
    if not isinstance(expected, OverlayDefinition) or overlay.definition != expected:
        raise ValueError("overlay does not match expected definition")


def _preflight_overlay_publication(
    staging: Path,
    destination: Path,
    expected: OverlayDefinition,
) -> DaggerOverlay:
    if destination.exists():
        existing = open_dagger_overlay(destination)
        _require_expected_definition(existing, expected)
        if staging.exists():
            candidate = open_dagger_overlay(staging)
            _require_expected_definition(candidate, expected)
            if candidate.content_identity != existing.content_identity:
                raise ValueError("published overlay conflicts with staging")
        return existing
    if not staging.exists():
        raise ValueError("overlay staging and destination are both missing")
    candidate = open_dagger_overlay(staging)
    _require_expected_definition(candidate, expected)
    return candidate


def publish_dagger_overlay(
    staging: Path,
    destination: Path,
    *,
    expected: OverlayDefinition,
) -> DaggerOverlay:
    staging, destination = Path(staging), Path(destination)
    if destination.exists():
        return _preflight_overlay_publication(staging, destination, expected)
    candidate = _preflight_overlay_publication(staging, destination, expected)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(staging, destination)
    published = open_dagger_overlay(destination)
    if published.content_identity != candidate.content_identity:
        raise ValueError("published overlay identity changed during atomic rename")
    return published


def publish_dagger_overlays(
    train_staging: Path,
    validation_staging: Path,
    train_destination: Path,
    validation_destination: Path,
    *,
    train_expected: OverlayDefinition,
    validation_expected: OverlayDefinition,
) -> tuple[DaggerOverlay, DaggerOverlay]:
    paths = tuple(map(Path, (
        train_staging, validation_staging, train_destination, validation_destination
    )))
    if len(set(paths)) != 4:
        raise ValueError("train and validation overlay paths must be distinct")
    train_candidate = _preflight_overlay_publication(
        paths[0], paths[2], train_expected,
    )
    validation_candidate = _preflight_overlay_publication(
        paths[1], paths[3], validation_expected,
    )
    if train_candidate.partition != "train" or validation_candidate.partition != "validation":
        raise ValueError("paired overlay partitions are invalid")
    train_shared = train_candidate.definition.to_dict()
    validation_shared = validation_candidate.definition.to_dict()
    for fields in (train_shared, validation_shared):
        for name in ("partition", "label_target", "game_ceiling"):
            fields.pop(name)
    if train_shared != validation_shared:
        raise ValueError("paired overlays have inconsistent shared identities")
    train = publish_dagger_overlay(paths[0], paths[2], expected=train_expected)
    validation = publish_dagger_overlay(
        paths[1], paths[3], expected=validation_expected,
    )
    return train, validation


_OVERLAY_FIELDS = frozenset({
    "schema_version", "status", "partition", "iteration", "observation_size",
    "action_size", "action_regions", "oracle", "learner", "scenario_hash",
    "contract_hash", "encoding_hash", "repository_hash", "panel_hash",
    "schedule_hash", "original_dataset", "label_target", "game_ceiling",
    "game_count", "row_count", "games", "content_identity",
})
_DEFINITION_FIELDS = frozenset({
    "partition", "iteration", "observation_size", "action_size", "action_regions",
    "oracle", "learner", "original_dataset", "scenario_hash", "contract_hash",
    "encoding_hash", "repository_hash", "panel_hash", "schedule_hash",
    "label_target", "game_ceiling",
})
_GAME_DESCRIPTOR_FIELDS = frozenset({
    "path", "sha256", "byte_size", "game_id", "row_count", "content_identity",
})
_SHARD_DESCRIPTOR_FIELDS = frozenset({
    "path", "sha256", "byte_size", "row_count", "content_identity",
})
_EVIDENCE_FIELDS = frozenset({
    "path", "sha256", "byte_size", "seed", "reference_seat", "learner_seat",
    "profile", "outcome", "transition_count",
})
_ROW_METADATA_FIELDS = frozenset({
    "learner_command", "teacher_command", "normalized_advantage",
    "opponent_living_unit_count", "productive_legal_action_count",
    "disagreement", "oracle_actual_expansion_count",
})
_GAME_MANIFEST_FIELDS = frozenset({
    "schema_version", "game_id", "partition", "iteration", "map_seed",
    "episode_seed", "schedule_index", "profile", "reference_seat",
    "learner_seat", "opponent", "outcome", "transition_count", "oracle",
    "learner", "scenario_hash", "contract_hash", "encoding_hash",
    "repository_hash", "panel_hash", "schedule_hash", "original_dataset",
    "label_target", "game_ceiling", "row_count", "shard",
    "trace", "replay", "row_metadata", "content_identity",
})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_identity(value: Mapping[str, Any]) -> str:
    canonical = {key: item for key, item in value.items() if key != "content_identity"}
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _artifact_descriptor(
    relative: str,
    path: Path,
    *,
    row_count: int,
    game_id: int | None = None,
    content_identity: str | None = None,
) -> dict[str, Any]:
    descriptor: dict[str, Any] = {
        "path": relative,
        "sha256": _sha256_file(path),
        "byte_size": path.stat().st_size,
        "row_count": row_count,
    }
    if game_id is not None:
        descriptor["game_id"] = game_id
    descriptor["content_identity"] = (
        content_identity if content_identity is not None
        else _content_identity(descriptor)
    )
    return descriptor


def _regions_to_dict(
    regions: Mapping[str, tuple[int, int]],
) -> dict[str, dict[str, int]]:
    return {
        name: {"offset": offset, "count": count}
        for name, (offset, count) in regions.items()
    }


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        with open(temporary, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(
                stream,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON number {token}")
                ),
            )
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"invalid JSON artifact {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact {path} must be an object")
    return value


def _validate_trace_file(path: Path, game: DaggerGame) -> EpisodeTrace:
    if not path.is_file():
        raise ValueError("DAgger trace evidence is missing")
    trace = EpisodeTrace.from_payload(_read_json(path))
    if len(trace.transitions) != game.transition_count:
        raise ValueError("trace transition count does not match game")
    if not trace.transitions or not trace.transitions[-1].after.is_game_over:
        raise ValueError("completed DAgger trace must end in a terminal state")
    terminal = trace.transitions[-1].after
    derived = (
        "draw" if terminal.winner is None
        else "win" if terminal.winner == game.learner_seat
        else "loss"
    )
    if derived != game.outcome:
        raise ValueError("trace terminal outcome does not match game")
    return trace


def _verified_artifact(
    root: Path, descriptor: Mapping[str, Any], seen: set[str]
) -> Path:
    relative = _safe_relative(descriptor["path"], "artifact path")
    if relative in seen:
        raise ValueError("artifact path is duplicated")
    path = _contained_file(root, relative, "artifact path")
    if (
        not path.is_file()
        or type(descriptor["byte_size"]) is not int
        or descriptor["byte_size"] < 1
        or path.stat().st_size != descriptor["byte_size"]
        or _sha256_file(path) != _hash(descriptor["sha256"], "artifact sha256")
    ):
        raise ValueError("physical artifact hash or size does not match descriptor")
    seen.add(relative)
    return path


def _validate_evidence(
    root: Path, raw: Any, game: DaggerGame, *, trace: bool
) -> None:
    descriptor = _strict_fields(raw, _EVIDENCE_FIELDS, "evidence descriptor")
    for key in ("seed", "reference_seat", "learner_seat", "transition_count"):
        _strict_int(descriptor[key], f"evidence {key}", minimum=0)
    _strict_string(descriptor["profile"], "evidence profile")
    _strict_string(descriptor["outcome"], "evidence outcome")
    expected = {
        "seed": game.map_seed,
        "reference_seat": game.reference_seat,
        "learner_seat": game.learner_seat,
        "profile": game.profile,
        "outcome": game.outcome,
        "transition_count": game.transition_count,
    }
    if any(descriptor[key] != value for key, value in expected.items()):
        raise ValueError("evidence metadata does not match game")
    path = _contained_file(root, descriptor["path"], "evidence path")
    if (
        not path.is_file()
        or type(descriptor["byte_size"]) is not int
        or descriptor["byte_size"] < 1
        or path.stat().st_size != descriptor["byte_size"]
        or _sha256_file(path) != _hash(descriptor["sha256"], "evidence sha256")
    ):
        raise ValueError("physical evidence hash or size does not match descriptor")
    if trace:
        _validate_trace_file(path, game)


def _validate_reciprocal_games(games: Sequence[DaggerGame]) -> None:
    if not games:
        raise ValueError("overlay contains no reciprocal games")
    pairs: dict[tuple[int, str], list[DaggerGame]] = {}
    for game in games:
        pairs.setdefault((game.map_seed, game.profile), []).append(game)
    for pair in pairs.values():
        if len(pair) != 2 or {game.learner_seat for game in pair} != {0, 1}:
            raise ValueError("overlay contains an incomplete reciprocal pair")
        if len({game.schedule_index for game in pair}) != 1:
            raise ValueError("reciprocal games have inconsistent schedule indices")
        if len({game.episode_seed for game in pair}) != 1:
            raise ValueError("reciprocal games have inconsistent episode seeds")


def _read_and_validate_shard(
    path: Path,
    *,
    row_count: int,
    contract: EnvironmentContract,
    game: DaggerGame,
    oracle: OracleSpec,
    row_metadata: Any,
) -> dict[str, np.ndarray]:
    required = {
        "observations", "packed_masks", "actions", "learner_actions", "seats",
        "rounds", "decision_indices", "reason_bits", "state_hashes",
    }
    try:
        with np.load(path, allow_pickle=False) as loaded:
            if set(loaded.files) != required:
                raise ValueError("shard fields are invalid")
            arrays = {name: loaded[name] for name in required}
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError("DAgger shard cannot be physically reopened") from exc
    mask_bytes = (contract.action_size + 7) // 8
    shapes = {
        "observations": (row_count, contract.observation_size),
        "packed_masks": (row_count, mask_bytes),
        **{name: (row_count,) for name in required - {"observations", "packed_masks"}},
    }
    dtypes = {
        "observations": np.dtype(np.float32),
        "packed_masks": np.dtype(np.uint8),
        "actions": np.dtype(np.int32),
        "learner_actions": np.dtype(np.int32),
        "seats": np.dtype(np.int32),
        "rounds": np.dtype(np.int32),
        "decision_indices": np.dtype(np.int32),
        "reason_bits": np.dtype(np.uint8),
        "state_hashes": np.dtype("S64"),
    }
    if any(
        arrays[name].shape != shapes[name] or arrays[name].dtype != dtypes[name]
        for name in required
    ):
        raise ValueError("DAgger shard shape or dtype is invalid")
    if not np.isfinite(arrays["observations"]).all():
        raise ValueError("DAgger shard observations are non-finite")
    unused = mask_bytes * 8 - contract.action_size
    if unused and np.any(
        arrays["packed_masks"][:, -1] & (((1 << unused) - 1) << (8 - unused))
    ):
        raise ValueError("DAgger shard packed mask has nonzero unused bits")
    masks = np.unpackbits(
        arrays["packed_masks"], axis=1, count=contract.action_size,
        bitorder="little",
    ).astype(bool)
    if not isinstance(row_metadata, list) or len(row_metadata) != row_count:
        raise ValueError("game row metadata count is invalid")
    for index in range(row_count):
        metadata = _strict_fields(
            row_metadata[index], _ROW_METADATA_FIELDS, "row metadata"
        )
        try:
            state_hash = bytes(arrays["state_hashes"][index]).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ValueError("state hash is not fixed-width ASCII") from exc
        payload = {
            "observation": arrays["observations"][index].tolist(),
            "legal_mask": masks[index].tolist(),
            "learner_action": int(arrays["learner_actions"][index]),
            "learner_command": metadata["learner_command"],
            "teacher_action": int(arrays["actions"][index]),
            "teacher_command": metadata["teacher_command"],
            "reason_bits": int(arrays["reason_bits"][index]),
            "state_hash": state_hash,
            "normalized_advantage": metadata["normalized_advantage"],
            "opponent_living_unit_count": metadata["opponent_living_unit_count"],
            "productive_legal_action_count": metadata["productive_legal_action_count"],
            "seat": int(arrays["seats"][index]),
            "round": int(arrays["rounds"][index]),
            "decision_index": int(arrays["decision_indices"][index]),
            "disagreement": metadata["disagreement"],
            "oracle_actual_expansion_count": metadata["oracle_actual_expansion_count"],
        }
        row = DaggerRow.from_dict(payload, contract=contract, oracle=oracle)
        if row.seat != game.learner_seat:
            raise ValueError("physical row seat does not match game")
    return arrays


_REASON_NAMES: tuple[tuple[int, str], ...] = (
    (1, "conversion"),
    (2, "favorable"),
    (4, "cycle_warning"),
    (8, "action_waste"),
)


def _validate_collection_definition(definition: CollectionDefinition) -> None:
    if not isinstance(definition, CollectionDefinition):
        raise ValueError("collection definition is invalid")
    try:
        expected = CollectionDefinition.create(
            contract=definition.contract,
            partition=definition.partition,
            iteration=definition.iteration,
            oracle=definition.oracle,
            learner=definition.learner,
            original_dataset=definition.original_dataset,
            scenario_hash=definition.scenario_hash,
            repository_hash=definition.repository_hash,
            panel_hash=definition.panel_hash,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("collection schedule or immutable definition is invalid") from exc
    if definition != expected:
        raise ValueError("collection schedule or immutable definition is invalid")


def _validate_collection_learner(
    definition: CollectionDefinition, learner: ResolvedController,
) -> None:
    if not isinstance(learner, ResolvedController):
        raise ValueError("collection learner is not a resolved controller")
    if (
        learner.spec.inference_mode != "deterministic"
        or learner.model is None
        or learner.algorithm is None
        or learner.path is None
        or learner.server_controller != "external"
    ):
        raise ValueError("collection learner must be deterministic external inference")
    try:
        checkpoint = Path(learner.path).resolve(strict=True)
    except OSError as exc:
        raise ValueError("collection learner checkpoint is missing") from exc
    identity = definition.learner
    if (
        checkpoint != Path(identity.checkpoint_path).resolve()
        or _sha256_file(checkpoint) != identity.checkpoint_sha256
    ):
        raise ValueError("collection learner checkpoint identity changed")
    source_run = (
        learner.spec.path
        if learner.spec.kind == "run"
        else learner.spec.source_run
    )
    if source_run is None:
        raise ValueError("collection learner source run is missing")
    try:
        resolved_source = Path(source_run).resolve(strict=True)
    except OSError as exc:
        raise ValueError("collection learner source run is missing") from exc
    source_manifest = resolved_source / "run.json"
    if (
        str(resolved_source) != identity.source_run
        or not source_manifest.is_file()
        or _sha256_file(source_manifest) != identity.source_manifest_sha256
    ):
        raise ValueError("collection learner source manifest identity changed")
    contract = learner.contract
    expected = definition.contract
    if (
        not isinstance(contract, EnvironmentContract)
        or contract.version != expected.version
        or contract.contract_hash != expected.contract_hash
        or contract.encoding_hash != expected.encoding_hash
        or learner.observation_size != expected.observation_size
        or learner.action_size != expected.action_size
    ):
        raise ValueError("collection learner contract identity changed")


def _validate_collection_client(
    definition: CollectionDefinition, client: DuelClient,
) -> None:
    contract = getattr(client, "contract", None)
    expected = definition.contract
    if (
        not isinstance(contract, EnvironmentContract)
        or contract.version != expected.version
        or contract.contract_hash != expected.contract_hash
        or contract.encoding_hash != expected.encoding_hash
        or contract.observation_size != expected.observation_size
        or contract.action_size != expected.action_size
        or _action_regions(contract) != _action_regions(expected)
        or _declared_collection_profiles(contract)
        != ("standard-3v3", *definition.conversion_profiles)
    ):
        raise ValueError("collection runtime contract identity changed")


def _require_collection_overlay(
    overlay: DaggerOverlay, definition: CollectionDefinition,
) -> None:
    _require_expected_definition(overlay, definition.overlay_definition)
    labels_before_final_pair = sum(
        descriptor.row_count for descriptor in overlay.manifest.games[:-2]
    )
    if (
        overlay.row_count < definition.label_target
        or not overlay.games
        or len(overlay.games) % 2
        or len(overlay.games) > definition.game_ceiling
    ):
        raise ValueError("completed collection overlay did not reach its target")
    if labels_before_final_pair >= definition.label_target:
        raise ValueError(
            "completed collection overlay did not stop at the first pair boundary"
        )
    for game_id, (game, scheduled) in enumerate(zip(
        overlay.games, definition.schedule[:len(overlay.games)], strict=True,
    )):
        if (
            game.game_id != game_id
            or game.map_seed != scheduled.map_seed
            or game.episode_seed != scheduled.episode_seed
            or game.schedule_index != scheduled.schedule_index
            or game.profile != scheduled.profile
            or game.reference_seat != scheduled.reference_seat
            or game.learner_seat != scheduled.learner_seat
            or game.opponent != "random"
        ):
            raise ValueError("completed collection overlay schedule changed")


def _empty_collection_stats() -> dict[str, Any]:
    return {
        "games": 0,
        "labels": 0,
        "reason_counts": Counter(),
        "disagreements": 0,
        "expansion_total": 0,
        "max_expansions": 0,
        "completed_pairs": 0,
    }


def _update_collection_stats(
    stats: dict[str, Any], rows: Sequence[DaggerRow],
) -> None:
    stats["games"] += 1
    stats["labels"] += len(rows)
    for row in rows:
        for bit, name in _REASON_NAMES:
            if row.reason_bits & bit:
                stats["reason_counts"][name] += 1
        stats["disagreements"] += int(row.disagreement)
        stats["expansion_total"] += row.oracle_actual_expansion_count
        stats["max_expansions"] = max(
            stats["max_expansions"], row.oracle_actual_expansion_count,
        )


def _overlay_collection_stats(overlay: DaggerOverlay) -> dict[str, Any]:
    stats = _empty_collection_stats()
    for descriptor in overlay.manifest.games:
        game_manifest = _read_json(
            _contained_file(overlay.root, descriptor.path, "collection game manifest")
        )
        shard_descriptor = _strict_fields(
            game_manifest["shard"], _SHARD_DESCRIPTOR_FIELDS,
            "collection shard descriptor",
        )
        shard_path = _contained_file(
            overlay.root, shard_descriptor["path"], "collection shard",
        )
        with np.load(shard_path, allow_pickle=False) as shard:
            reason_bits = shard["reason_bits"].tolist()
        metadata = game_manifest["row_metadata"]
        if not isinstance(metadata, list) or len(metadata) != len(reason_bits):
            raise ValueError("collection progress metadata is invalid")
        stats["games"] += 1
        stats["labels"] += len(reason_bits)
        for bits, row_metadata in zip(reason_bits, metadata, strict=True):
            for bit, name in _REASON_NAMES:
                if int(bits) & bit:
                    stats["reason_counts"][name] += 1
            stats["disagreements"] += int(row_metadata["disagreement"])
            expansions = row_metadata["oracle_actual_expansion_count"]
            stats["expansion_total"] += expansions
            stats["max_expansions"] = max(stats["max_expansions"], expansions)
    stats["completed_pairs"] = stats["games"] // 2
    return stats


def _emit_collection_progress(
    progress: Callable[[str], None],
    *,
    definition: CollectionDefinition,
    stats: Mapping[str, Any],
    event: str,
    pair_index: int | None,
    pair_complete: bool,
    new_games: int,
    started: float,
) -> None:
    elapsed = max(0.0, time.monotonic() - started)
    labels = int(stats["labels"])
    rate = labels / elapsed if elapsed > 0.0 else 0.0
    remaining = max(0, definition.label_target - labels)
    eta = remaining / rate if rate > 0.0 else (0.0 if remaining == 0 else None)
    payload = {
        "event": event,
        "partition": definition.partition,
        "iteration": definition.iteration,
        "games": int(stats["games"]),
        "labels": labels,
        "reason_counts": {
            name: int(stats["reason_counts"].get(name, 0))
            for _, name in _REASON_NAMES
        },
        "disagreements": int(stats["disagreements"]),
        "mean_expansions": (
            float(stats["expansion_total"]) / labels if labels else 0.0
        ),
        "max_expansions": int(stats["max_expansions"]),
        "labels_per_second": rate,
        "elapsed_seconds": elapsed,
        "eta_seconds": eta,
        "pair_index": pair_index,
        "pair_complete": pair_complete,
        "new_games": new_games,
    }
    progress(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")


def _normalized_dagger_rows(
    raw_rows: Any,
    *,
    contract: EnvironmentContract,
    oracle: OracleSpec,
) -> list[DaggerRow]:
    validated = validate_dagger_payload(
        {"schema_version": 1, "decisions": raw_rows}, contract,
    )
    rows: list[DaggerRow] = []
    seen_hashes: set[str] = set()
    for raw in validated:
        if (
            raw["OracleDepth"] != oracle.depth
            or raw["OracleExpansionBudget"] != oracle.expansion_budget
            or raw["OracleHeuristicIdentity"] != oracle.heuristic_identity
        ):
            raise ValueError("DAgger decision oracle identity changed")
        state_hash = raw["StateHash"]
        if state_hash in seen_hashes:
            raise ValueError("duplicate canonical state hash in episode")
        seen_hashes.add(state_hash)
        rows.append(DaggerRow.from_dict({
            "observation": [float(value) for value in raw["Observation"]],
            "legal_mask": list(raw["LegalMask"]),
            "learner_action": raw["LearnerAction"],
            "learner_command": raw["LearnerCommand"],
            "teacher_action": raw["TeacherAction"],
            "teacher_command": raw["TeacherCommand"],
            "reason_bits": raw["Reasons"],
            "state_hash": state_hash,
            "normalized_advantage": float(raw["NormalizedAdvantage"]),
            "opponent_living_unit_count": raw["OpponentLivingUnitCount"],
            "productive_legal_action_count": raw["ProductiveLegalActionCount"],
            "seat": raw["Seat"],
            "round": raw["Round"],
            "decision_index": raw["DecisionIndex"],
            "disagreement": raw["Disagreement"],
            "oracle_actual_expansion_count": raw["OracleActualExpansionCount"],
        }, contract=contract, oracle=oracle))
    return rows


def _reset_collection_duel(
    client: DuelClient,
    *,
    seed: int,
    p0: str,
    p1: str,
    learner: int,
    start_profile: str,
    reference_seat: int,
) -> dict[str, Any]:
    """Reset with an explicit observer/reward learner seat.

    The general evaluation client predates reciprocal DAgger and fixes its wire
    learner to seat 0. Collection must set this engine field independently of
    which controller slot is external, so the production path uses the same
    validated RPC boundary with the authoritative scheduled seat.
    """

    if type(learner) is not int or learner not in {0, 1}:
        raise ValueError("collection learner seat must be 0 or 1")
    if type(reference_seat) is not int or reference_seat not in {0, 1}:
        raise ValueError("collection reference seat must be 0 or 1")
    if start_profile not in _declared_collection_profiles(client.contract):
        raise ValueError("collection start profile is not declared by the runtime")
    if isinstance(client, DuelClient):
        request = {
            "cmd": "duel_reset",
            "seed": seed,
            "p0": p0,
            "p1": p1,
            "learner": learner,
            "start_profile": start_profile,
            "reference_seat": reference_seat,
        }
        response = client._rpc(request)
        validate_step_payload(
            response,
            observation_size=client.contract.observation_size,
            action_size=client.contract.action_size,
        )
        if (
            response.get("start_profile") != start_profile
            or response.get("reference_seat") != reference_seat
        ):
            raise ValueError("collection reset did not acknowledge its schedule")
        return response
    return client.reset(
        seed=seed,
        p0=p0,
        p1=p1,
        learner=learner,
        start_profile=start_profile,
        reference_seat=reference_seat,
    )


def _collect_scheduled_game(
    *,
    client: DuelClient,
    learner: ResolvedController,
    oracle: OracleSpec,
    writer: DaggerOverlayWriter,
    definition: CollectionDefinition,
    scheduled: ScheduledDuel,
    game_id: int,
) -> list[DaggerRow]:
    client.configure_dagger(
        enabled=True,
        depth=oracle.depth,
        expansion_budget=oracle.expansion_budget,
        use_heuristic=True,
    )
    client.enable_trace(True)
    seats = (
        (learner.server_controller, "random")
        if scheduled.learner_seat == 0 else ("random", learner.server_controller)
    )
    state = _reset_collection_duel(
        client,
        seed=scheduled.map_seed,
        p0=seats[0],
        p1=seats[1],
        learner=scheduled.learner_seat,
        start_profile=scheduled.profile,
        reference_seat=scheduled.reference_seat,
    )
    if (
        not isinstance(state, Mapping)
        or state.get("start_profile") != scheduled.profile
        or state.get("reference_seat") != scheduled.reference_seat
    ):
        raise ValueError("collection reset schedule acknowledgement changed")
    decisions = 0
    while not bool(state.get("terminated")) and not bool(state.get("truncated")):
        if decisions >= MAX_DECISIONS_PER_GAME:
            raise RuntimeError("collection game exceeded the decision ceiling")
        if state.get("seat") != scheduled.learner_seat:
            raise RuntimeError("collection runtime surfaced a non-learner external seat")
        observation = np.asarray(state.get("obs"), dtype=np.float32)
        mask = np.asarray(state.get("mask"), dtype=bool)
        validate_inference_input(learner, observation, mask)
        action = predict(
            learner.model,
            learner.algorithm,
            observation,
            mask,
            deterministic=True,
        )
        if action < 0 or action >= mask.size or not bool(mask[action]):
            raise RuntimeError("collection learner selected a masked action")
        state = client.step(action)
        decisions += 1
    if not bool(state.get("terminated")) or bool(state.get("truncated")):
        raise RuntimeError("collection game did not terminate authoritatively")
    trace = client.drain_trace()
    if (
        not isinstance(trace, EpisodeTrace)
        or not trace.transitions
        or not trace.transitions[-1].after.is_game_over
    ):
        raise ValueError("collection trace is missing a terminal transition")
    terminal_winner = trace.transitions[-1].after.winner
    state_winner = state.get("winner")
    normalized_state_winner = (
        state_winner
        if type(state_winner) is int and state_winner in {0, 1}
        else None
    )
    if terminal_winner != normalized_state_winner:
        raise ValueError("collection trace winner does not match runtime state")
    trace_relative = f"evidence/game-{game_id:08d}.trace.json"
    replay_relative = f"evidence/game-{game_id:08d}.replay"
    atomic_write_json(writer.root / trace_relative, trace.to_dict())
    client.save_replay(writer.root / replay_relative)
    rows = _normalized_dagger_rows(
        client.drain_dagger(), contract=definition.contract, oracle=oracle,
    )
    outcome = (
        "draw" if terminal_winner is None
        else "win" if terminal_winner == scheduled.learner_seat
        else "loss"
    )
    game = DaggerGame.from_dict({
        "game_id": game_id,
        "partition": definition.partition,
        "iteration": definition.iteration,
        "map_seed": scheduled.map_seed,
        "episode_seed": scheduled.episode_seed,
        "schedule_index": scheduled.schedule_index,
        "profile": scheduled.profile,
        "reference_seat": scheduled.reference_seat,
        "learner_seat": scheduled.learner_seat,
        "opponent": "random",
        "outcome": outcome,
        "transition_count": len(trace.transitions),
        "trace_path": trace_relative,
        "replay_path": replay_relative,
    })
    writer.append_game(game, rows)
    return rows


def _write_collection_failure(
    staging: Path,
    *,
    error: BaseException,
    definition: CollectionDefinition,
    stats: Mapping[str, Any],
) -> None:
    if not staging.exists():
        return
    (staging / "manifest.json").unlink(missing_ok=True)
    files = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file() and path.name != "diagnostic.json"
    )
    files.append("diagnostic.json")
    atomic_write_json(staging / "diagnostic.json", {
        "schema_version": 1,
        "status": "failed",
        "partition": definition.partition,
        "iteration": definition.iteration,
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "games": int(stats["games"]),
        "labels": int(stats["labels"]),
        "last_complete_pair": (
            int(stats["completed_pairs"]) - 1
            if stats["completed_pairs"] else None
        ),
        "physical_files": sorted(files),
    })


def collect_selective_dagger(
    *,
    definition: CollectionDefinition,
    learner: ResolvedController,
    oracle: OracleSpec,
    output_root: Path,
    client_factory: Callable[[], DuelClient],
    progress: Callable[[str], None],
) -> DaggerOverlay:
    """Collect one deterministic, reciprocal, immutable selective-DAgger overlay."""

    started = time.monotonic()
    _validate_collection_definition(definition)
    if oracle != definition.oracle:
        raise ValueError("collection oracle identity changed")
    _validate_collection_learner(definition, learner)
    if not callable(client_factory) or not callable(progress):
        raise ValueError("collection runtime callbacks are invalid")
    destination = Path(output_root).resolve()
    staging = destination.with_name(destination.name + ".staging")
    if destination.exists() and staging.exists():
        raise ValueError(
            "collection destination and staging coexist ambiguously"
        )
    if destination.exists():
        try:
            existing = open_dagger_overlay(destination)
            _require_collection_overlay(existing, definition)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "existing collection destination is not exactly reusable"
            ) from exc
        stats = _overlay_collection_stats(existing)
        _emit_collection_progress(
            progress,
            definition=definition,
            stats=stats,
            event="reuse",
            pair_index=len(existing.games) // 2 - 1,
            pair_complete=True,
            new_games=0,
            started=started,
        )
        return existing

    if staging.exists():
        try:
            candidate = open_dagger_overlay(staging)
            _require_collection_overlay(candidate, definition)
            resumed = publish_dagger_overlay(
                staging, destination, expected=definition.overlay_definition,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "collection staging exists but is not exactly reusable"
            ) from exc
        stats = _overlay_collection_stats(resumed)
        _emit_collection_progress(
            progress,
            definition=definition,
            stats=stats,
            event="reuse",
            pair_index=len(resumed.games) // 2 - 1,
            pair_complete=True,
            new_games=0,
            started=started,
        )
        return resumed
    stats = _empty_collection_stats()
    writer: DaggerOverlayWriter | None = None
    client: DuelClient | None = None
    close_attempted = False
    try:
        writer = DaggerOverlayWriter.create(
            staging,
            contract=definition.contract,
            partition=definition.partition,
            iteration=definition.iteration,
            oracle=definition.oracle,
            learner=definition.learner,
            original_dataset=definition.original_dataset,
            scenario_hash=definition.scenario_hash,
            repository_hash=definition.repository_hash,
            panel_hash=definition.panel_hash,
            schedule_hash=definition.schedule_hash,
            label_target=definition.label_target,
            game_ceiling=definition.game_ceiling,
        )
        client = client_factory()
        _validate_collection_client(definition, client)
        for game_id, scheduled in enumerate(definition.schedule):
            rows = _collect_scheduled_game(
                client=client,
                learner=learner,
                oracle=oracle,
                writer=writer,
                definition=definition,
                scheduled=scheduled,
                game_id=game_id,
            )
            _update_collection_stats(stats, rows)
            if scheduled.learner_seat == 1:
                stats["completed_pairs"] += 1
            _emit_collection_progress(
                progress,
                definition=definition,
                stats=stats,
                event="game",
                pair_index=scheduled.schedule_index,
                pair_complete=scheduled.learner_seat == 1,
                new_games=stats["games"],
                started=started,
            )
            if scheduled.learner_seat == 1:
                _emit_collection_progress(
                    progress,
                    definition=definition,
                    stats=stats,
                    event="pair",
                    pair_index=scheduled.schedule_index,
                    pair_complete=True,
                    new_games=stats["games"],
                    started=started,
                )
                if stats["labels"] >= definition.label_target:
                    break
        if stats["labels"] < definition.label_target:
            raise RuntimeError(
                "collection label target was not reached at the game ceiling"
            )
        close_attempted = True
        client.close()
        client = None
        candidate = writer.seal()
        _require_collection_overlay(candidate, definition)
        return publish_dagger_overlay(
            staging, destination, expected=definition.overlay_definition,
        )
    except BaseException as exc:
        if client is not None and not close_attempted:
            close_attempted = True
            try:
                client.close()
            except BaseException as close_error:
                exc.add_note(
                    "collection client close also failed: "
                    f"{type(close_error).__name__}: {close_error}"
                )
        _write_collection_failure(
            staging, error=exc, definition=definition, stats=stats,
        )
        raise
