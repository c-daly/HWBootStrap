"""Immutable selective-DAgger overlay schemas and physical storage."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
import time
import uuid
from collections import Counter, OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import EnvironmentContract
from .algorithms import ActorTransferSource
from .controllers import ResolvedController, predict, validate_inference_input
from .draw_classification import DrawCategory, classify_draw, summarize_episode
from .evaluation import (
    DuelClient,
    MAX_DECISIONS_PER_GAME,
    validate_dagger_payload,
    wilson_interval,
)
from .io import atomic_write_json
from .imitation import (
    ACTION_KINDS,
    ActorSupervisionCorpus,
    BehavioralCloningConfig,
    ImitationBatch,
    ImitationDataset,
    MaterializedImitationPartition,
    PRODUCTION_DAGGER_DISTILLATION_CONFIG,
    Source,
    audit_imitation_dataset,
    load_imitation_dataset,
    materialize_imitation_partition,
    train_actor_supervision,
)
from .protocol import validate_step_payload
from .scenarios import resolve_scenario
from .tactical_trace import EpisodeTrace


OVERLAY_SCHEMA_VERSION = 1
DAGGER_DISTILLATION_CONFIG = PRODUCTION_DAGGER_DISTILLATION_CONFIG
_ITERATION_ONE_CONTROLLER = {
    "kind": "snapshot",
    "path": (
        "C:/Users/cddal/HexWars/python/runs/"
        "bc227-ppo-random-s227-20260802-v2/checkpoints/"
        "step_000038912.zip"
    ),
    "source_run": (
        "C:/Users/cddal/HexWars/python/runs/"
        "bc227-ppo-random-s227-20260802-v2"
    ),
    "algorithm": "maskable_ppo",
    "step": 38_912,
    "inference_mode": "deterministic",
}
_ITERATION_ONE_CHECKPOINT_SHA256 = (
    "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
)
_ITERATION_ONE_SOURCE_MANIFEST_SHA256 = (
    "7f02152c2ea39a08e5e203c0b0ba13928b2ad1847e276cc1b19f53331151ba46"
)
_INT32_MAX = (1 << 31) - 1
_MAX_PREFLIGHT_SAMPLES_PER_GAME = 1024
_MAX_ORACLE_STATE_DEPTH = 32
_MAX_ORACLE_STATE_NODES = 50_000
_MAX_ORACLE_STATE_BYTES = 1 << 20
_MAX_PREFLIGHT_MANIFEST_BYTES = 8 << 20
_MAX_PREFLIGHT_TRACE_BYTES = 32 << 20
_MAX_PREFLIGHT_REPLAY_BYTES = 8 << 20
_MAX_PREFLIGHT_BENCHMARK_BYTES = 64 << 20
_MAX_PREFLIGHT_DIAGNOSTIC_BYTES = 8 << 20
_MAX_PREFLIGHT_OWNER_BYTES = 4096
_MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILES = 1500
_MAX_PREFLIGHT_DIAGNOSTIC_TREE_BYTES = 2 << 30
_MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILE_BYTES = 64 << 20
_PRIVATE_TEST_EXECUTION_TRUST = MappingProxyType({
    "schema_version": 1,
    "mode": "private-test-transcript",
    "evidence_class": "untrusted-test-transcript",
    "engine_authenticated": False,
    "engine_evidence_root": None,
    "task_9_production_seal_required": True,
})
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
    use_heuristic: bool
    heuristic_identity: str
    code_hash: str

    @classmethod
    def from_dict(cls, value: Any) -> "OracleSpec":
        fields = _strict_fields(value, frozenset({
            "oracle_type", "depth", "expansion_budget", "use_heuristic",
            "heuristic_identity", "code_hash",
        }), "oracle")
        oracle_type = _strict_string(fields["oracle_type"], "oracle_type")
        if oracle_type != "bounded-search":
            raise ValueError("oracle_type is unsupported")
        depth = _strict_int(fields["depth"], "depth", minimum=1)
        budget = _strict_int(fields["expansion_budget"], "expansion_budget", minimum=1)
        use_heuristic = fields["use_heuristic"]
        if type(use_heuristic) is not bool:
            raise ValueError("use_heuristic must be a boolean")
        heuristic = _strict_string(fields["heuristic_identity"], "heuristic_identity")
        if heuristic != "material-plus-pursuit-v1":
            raise ValueError("heuristic_identity is unsupported")
        return cls(
            oracle_type, depth, budget, use_heuristic, heuristic,
            _hash(fields["code_hash"], "code_hash"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "oracle_type": self.oracle_type,
            "depth": self.depth,
            "expansion_budget": self.expansion_budget,
            "use_heuristic": self.use_heuristic,
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


@dataclass(frozen=True)
class SeedBankDefinition:
    partition: str
    iteration: int | None
    start: int
    stop: int
    assigned: bool


@dataclass(frozen=True)
class PanelDefinition:
    panel_path: Path
    repository_root: Path
    panel_id: str
    environment: str
    panel_sha256: str
    seed_banks_sha256: str
    scenario_path: Path
    scenario_sha256: str
    runtime_scenario_sha256: str
    contract_hash: str
    encoding_hash: str
    observation_size: int
    action_size: int
    action_regions: Mapping[str, Any]
    repository_policy: Mapping[str, Any]
    starting_learner: ActorTransferSource
    learner_source_manifest_sha256: str
    learner_source_scenario_sha256: str
    dataset_root: Path
    dataset_manifest_sha256: str
    dataset_content_sha256: str
    dataset_file_count: int
    dataset_byte_size: int
    dataset_contract_hash: str
    dataset_encoding_hash: str
    dataset_scenario_hash: str
    profiles: tuple[str, ...]
    oracle_candidates: tuple[OracleSpec, ...]
    preflight: Mapping[str, Any]
    collection: Mapping[str, Any]
    training: Mapping[str, Any]
    evaluation: Mapping[str, Any]
    smoke: Mapping[str, Any]
    success: Mapping[str, Any]
    seed_banks: tuple[SeedBankDefinition, ...]
    preflight_schedule: tuple[ScheduledDuel, ...]


_PANEL_FIELDS = frozenset({
    "schema_version", "id", "environment", "scenario", "contract",
    "repository", "starting_learner", "original_dataset", "profiles", "oracle",
    "collection", "training", "evaluation", "smoke", "success", "seed_banks",
})
_SEED_BANK_FIELDS = frozenset({
    "schema_version", "banks", "oracle_preflight_profiles", "reciprocal",
})
_PANEL_PROFILES = (
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)
_ORACLE_CODE_SHA256 = (
    "5f03a7c8d0fda16497a9e6a2f1ad1ba4fcb920957b7a4b5fbc2545e0ae893061"
)
_BASE_DATASET_PATH = (
    "C:/Users/cddal/HexWars/.worktrees/tactical-baseline-evidence/"
    "python/datasets/annihilation-imitation-v1"
)
_BASE_DATASET_MANIFEST_SHA256 = (
    "6c9f1fd43cded0691080dd12c390aee086d49b144ebc0207d2f80e6b5a9422c4"
)
_BASE_DATASET_CONTRACT_HASH = (
    "2d6984089aa151cee59e10bb37b0d2239e7a0668f34d90e1af64216aaf713edf"
)
_BASE_DATASET_ENCODING_HASH = (
    "2f334bc2163fd931d84c004e9dc8f44bae68934e46fbf2ec2c819fa3e297054a"
)
_BASE_DATASET_SCENARIO_HASH = (
    "3236e528f9f3c41a1c696834d48f24db4ffae48cf96c174b5aaafa862c3a589f"
)
_BASE_DATASET_CONTENT_SHA256 = (
    "da077fb8291f00bf2359e3ce9c834a331032dc4f43397e3fa434d69c0d28989c"
)
_BASE_DATASET_FILE_COUNT = 3966
_BASE_DATASET_BYTE_SIZE = 17_852_257
_BASE_DATASET_AUDIT_VERSION = "full-imitation-audit-v1"
_BASE_DATASET_SEMANTIC_AUDIT_CACHE: dict[
    tuple[str, int, int, str, str, str, str], dict[str, int]
] = {}
_PANEL_SCENARIO_PATH = "python/config/annihilation-imitation-v1.json"
_ORACLE_SOURCE_PATH = "engine/HexWars.Engine/BoundedSearchAgent.cs"
_PANEL_SCENARIO_SHA256 = (
    "4f085b8a80f7ba8e450a85dbcceb73e05723ce7b37045f1ddd1ef91d67a95632"
)
_RUNTIME_SCENARIO_SHA256 = (
    "00684a8623f3f1deadd8d31cb71a0492441508c34a42d6f5ac6a1f8e662aaaa4"
)
_PANEL_CONTRACT_HASH = (
    "7347819c2e68fa2d216dc712afc4785e185ca50d3832487d66589a68eee5a9d6"
)
_PANEL_ENCODING_HASH = _BASE_DATASET_ENCODING_HASH
_PANEL_ACTION_REGIONS = {
    "move": {"offset": 1, "count": 351},
    "attack": {"offset": 352, "count": 351},
    "deploy": {"offset": 703, "count": 585},
}


def _exact_json(value: Any, expected: Mapping[str, Any], label: str) -> Mapping[str, Any]:
    fields = _strict_fields(value, frozenset(expected), label)
    if not _same_exact_json(fields, expected):
        raise ValueError(f"{label} values are invalid")
    return fields


def _same_exact_json(actual: Any, expected: Any) -> bool:
    if isinstance(expected, Mapping):
        return (
            isinstance(actual, Mapping)
            and set(actual) == set(expected)
            and all(_same_exact_json(actual[key], value)
                    for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            type(actual) is list
            and len(actual) == len(expected)
            and all(_same_exact_json(left, right)
                    for left, right in zip(actual, expected, strict=True))
        )
    return type(actual) is type(expected) and actual == expected


def _is_reparse_point(path: Path) -> bool:
    try:
        stat = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"authoritative path is missing: {path}") from exc
    return path.is_symlink() or bool(
        getattr(stat, "st_file_attributes", 0) & 0x400
    )


def _reject_reparse_chain(path: Path, label: str) -> None:
    absolute = Path(path).absolute()
    parts = absolute.parts
    current = Path(parts[0])
    for part in parts[1:]:
        current /= part
        if current.exists() and _is_reparse_point(current):
            raise ValueError(f"{label} must not traverse a symlink, junction, or reparse point")


def _lexical_canonical_path(
    path: Path, label: str, *, strict: bool,
) -> Path:
    supplied = Path(path).absolute()
    _reject_reparse_chain(supplied, label)
    try:
        resolved = supplied.resolve(strict=strict)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if supplied != resolved:
        raise ValueError(f"{label} must be a lexical canonical path")
    return resolved


def _canonical_external_path(value: Any, label: str) -> Path:
    text = _strict_string(value, label)
    path = Path(text)
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute")
    _reject_reparse_chain(path, label)
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if path.absolute() != resolved:
        raise ValueError(f"{label} must not traverse a symlink or junction")
    return resolved


def _definition_file(
    root: Path, relative: Any, label: str,
) -> Path:
    canonical = _safe_relative(relative, label)
    _reject_reparse_chain(Path(root), label)
    _reject_reparse_chain(Path(root) / canonical, label)
    try:
        resolved_root = Path(root).resolve(strict=True)
        resolved = (resolved_root / canonical).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if (
        not resolved.is_relative_to(resolved_root)
        or resolved.parent != resolved_root
        or not resolved.is_file()
    ):
        raise ValueError(f"{label} must be a contained direct file")
    return resolved


def _repository_file(
    repository_root: Path, relative: Any, label: str,
) -> Path:
    canonical = _safe_relative(relative, label)
    _reject_reparse_chain(Path(repository_root), label)
    _reject_reparse_chain(Path(repository_root) / canonical, label)
    try:
        root = Path(repository_root).resolve(strict=True)
        resolved = (root / canonical).resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"{label} is missing") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError(f"{label} must be contained by the repository")
    return resolved


def _parse_seed_banks(
    raw: Any,
) -> tuple[tuple[SeedBankDefinition, ...], tuple[ScheduledDuel, ...]]:
    fields = _strict_fields(raw, _SEED_BANK_FIELDS, "seed bank definition")
    if _strict_int(fields["schema_version"], "seed bank schema_version") != 1:
        raise ValueError("seed bank schema_version must be integer 1")
    raw_banks = fields["banks"]
    if not isinstance(raw_banks, list):
        raise ValueError("seed banks must be an array")
    banks: list[SeedBankDefinition] = []
    definitions: list[tuple[str, int | None, int, int]] = []
    for value in raw_banks:
        bank = _strict_fields(
            value,
            frozenset({"partition", "iteration", "start", "stop", "assigned"}),
            "seed bank",
        )
        partition = _strict_string(bank["partition"], "seed bank partition")
        iteration = bank["iteration"]
        if iteration is not None:
            iteration = _strict_int(iteration, "seed bank iteration", minimum=1)
        start = _strict_int(bank["start"], "seed bank start", minimum=0)
        stop = _strict_int(bank["stop"], "seed bank stop", minimum=0)
        assigned = bank["assigned"]
        if type(assigned) is not bool:
            raise ValueError("seed bank assigned must be boolean")
        banks.append(SeedBankDefinition(
            partition, iteration, start, stop, assigned,
        ))
        definitions.append((partition, iteration, start, stop))
    validate_seed_definitions(tuple(definitions))
    if tuple(definitions) != SEED_DEFINITIONS:
        if any(start <= 17_000_249 and stop >= 17_000_000 for _, _, start, stop in definitions):
            raise ValueError("final evaluation seeds are forbidden")
        raise ValueError("seed banks do not match the locked definition")
    expected_assignments = tuple(
        partition != "reserved" for partition, _iteration, _start, _stop in definitions
    )
    if tuple(bank.assigned for bank in banks) != expected_assignments:
        raise ValueError("seed bank assignment state is invalid")

    reciprocal = _exact_json(fields["reciprocal"], {
        "seat_order": [0, 1],
        "episode_seed": "map_seed",
        "reference_seat": "learner_seat",
    }, "reciprocal expansion")
    profiles = fields["oracle_preflight_profiles"]
    if not isinstance(profiles, list) or len(profiles) != len(_PANEL_PROFILES):
        raise ValueError("oracle preflight profiles must contain six entries")
    schedule: list[ScheduledDuel] = []
    expected_seed = 18_900_000
    for profile_index, (raw_profile, expected_profile) in enumerate(
        zip(profiles, _PANEL_PROFILES, strict=True)
    ):
        item = _strict_fields(raw_profile, frozenset({
            "profile", "start", "stop", "maps", "both_seats",
        }), "oracle preflight profile")
        start = _strict_int(item["start"], "oracle preflight start", minimum=0)
        stop = _strict_int(item["stop"], "oracle preflight stop", minimum=0)
        maps = _strict_int(item["maps"], "oracle preflight maps", minimum=1)
        if (
            item["profile"] != expected_profile
            or start != expected_seed
            or stop != start + 19
            or maps != 20
            or item["both_seats"] is not True
        ):
            raise ValueError("oracle preflight profile must own exactly 20 canonical maps")
        for offset, seed in enumerate(range(start, stop + 1)):
            pair_index = profile_index * 20 + offset
            for seat in reciprocal["seat_order"]:
                schedule.append(ScheduledDuel(
                    schedule_index=pair_index,
                    map_seed=seed,
                    episode_seed=seed,
                    profile=expected_profile,
                    reference_seat=seat,
                    learner_seat=seat,
                ))
        expected_seed = stop + 1
    if expected_seed != 18_900_120 or len(schedule) != 240:
        raise ValueError("oracle preflight schedule count is invalid")
    return tuple(banks), tuple(schedule)


def load_panel_definition(
    path: Path, *, repository_root: Path,
) -> PanelDefinition:
    """Physically reopen and strictly validate the selective-DAgger panel."""

    supplied_root = Path(repository_root)
    panel_path = _lexical_canonical_path(
        Path(path), "panel definition path", strict=True,
    )
    root = _lexical_canonical_path(
        supplied_root, "repository root", strict=True,
    )
    if not panel_path.is_file():
        raise ValueError("panel definition must be a file")
    panel = _strict_fields(_read_json(panel_path), _PANEL_FIELDS, "panel definition")
    if _strict_int(panel["schema_version"], "panel schema_version") != 1:
        raise ValueError("panel schema_version must be integer 1")
    if panel["id"] != "annihilation-selective-dagger-v1":
        raise ValueError("panel id is invalid")
    if panel["environment"] != "tactical-v2":
        raise ValueError("panel environment is invalid")

    seed_identity = _strict_fields(
        panel["seed_banks"], frozenset({"path", "sha256"}), "seed bank identity",
    )
    seeds_path = _definition_file(
        panel_path.parent, seed_identity["path"], "seed bank path",
    )
    seeds_sha256 = _hash(seed_identity["sha256"], "seed bank sha256")
    if _sha256_file(seeds_path) != seeds_sha256:
        raise ValueError("seed bank physical hash changed")
    seed_banks, preflight_schedule = _parse_seed_banks(_read_json(seeds_path))

    scenario = _strict_fields(panel["scenario"], frozenset({
        "path", "sha256", "runtime_snapshot_sha256",
    }), "panel scenario")
    if scenario["path"] != _PANEL_SCENARIO_PATH:
        raise ValueError("panel scenario path is not the locked definition")
    scenario_path = _repository_file(root, scenario["path"], "panel scenario")
    scenario_sha256 = _hash(scenario["sha256"], "scenario sha256")
    if (
        scenario_sha256 != _PANEL_SCENARIO_SHA256
        or scenario["runtime_snapshot_sha256"] != _RUNTIME_SCENARIO_SHA256
    ):
        raise ValueError("scenario identity is not the locked panel scenario")
    if _sha256_file(scenario_path) != scenario_sha256:
        raise ValueError("scenario physical hash changed")
    resolved_scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=scenario_path,
        template_id=None,
        enforce_round_cap_minimum=True,
    )
    rules = resolved_scenario.document.get("rules")
    if (
        resolved_scenario.template_id != "annihilation-imitation-v1"
        or not isinstance(rules, Mapping)
        or rules.get("fog_of_war") is not False
    ):
        raise ValueError("locked scenario must disable fog of war")
    canonical_scenario_hash = hashlib.sha256(
        resolved_scenario.canonical_json.encode("utf-8")
    ).hexdigest()
    runtime_scenario_sha256 = _hash(
        scenario["runtime_snapshot_sha256"], "runtime scenario sha256",
    )

    contract_fields = _strict_fields(panel["contract"], frozenset({
        "version", "contract_hash", "encoding_hash", "observation_size",
        "action_size", "action_regions",
    }), "panel contract")
    contract = EnvironmentContract(
        version=_strict_string(contract_fields["version"], "contract version"),
        contract_hash=_hash(contract_fields["contract_hash"], "contract hash"),
        encoding_hash=_hash(contract_fields["encoding_hash"], "encoding hash"),
        observation_size=_strict_int(
            contract_fields["observation_size"], "observation size", minimum=1,
        ),
        action_size=_strict_int(
            contract_fields["action_size"], "action size", minimum=1,
        ),
        board={},
        roster=[],
        reward={},
        semantics={"action_regions": contract_fields["action_regions"]},
    )
    if (
        contract.version != "tactical-v2"
        or contract.contract_hash != _PANEL_CONTRACT_HASH
        or contract.encoding_hash != _PANEL_ENCODING_HASH
        or contract.observation_size != 1292
        or contract.action_size != 1288
        or not _same_exact_json(contract_fields["action_regions"], _PANEL_ACTION_REGIONS)
    ):
        raise ValueError("panel tactical-v2 contract geometry is invalid")
    action_regions = _regions_to_dict(_action_regions(contract))

    repository = _exact_json(panel["repository"], {
        "required_clean": True,
        "identity_fields": ["commit", "source_tree", "dirty"],
        "output_policy": "outside_repository",
    }, "repository policy")
    repository_policy = MappingProxyType({
        "required_clean": True,
        "identity_fields": tuple(repository["identity_fields"]),
        "output_policy": repository["output_policy"],
    })

    learner = _strict_fields(panel["starting_learner"], frozenset({
        "source_kind", "controller", "checkpoint_sha256",
        "source_manifest_sha256", "source_scenario_sha256", "contract_hash",
        "encoding_hash",
    }), "starting learner")
    starting_learner = ActorTransferSource(
        source_kind=learner["source_kind"],
        controller=learner["controller"],
        checkpoint_sha256=_hash(
            learner["checkpoint_sha256"], "learner checkpoint sha256",
        ),
    )
    if learner["source_manifest_sha256"] != _ITERATION_ONE_SOURCE_MANIFEST_SHA256:
        raise ValueError("learner source manifest is not the locked snapshot manifest")
    if (
        dict(starting_learner.controller) != _ITERATION_ONE_CONTROLLER
        or starting_learner.checkpoint_sha256 != _ITERATION_ONE_CHECKPOINT_SHA256
        or learner["contract_hash"] != contract.contract_hash
        or learner["encoding_hash"] != contract.encoding_hash
        or learner["source_scenario_sha256"] != runtime_scenario_sha256
    ):
        raise ValueError(
            "starting learner checkpoint or identity is not the locked seed-227 snapshot"
        )
    checkpoint = _canonical_external_path(
        starting_learner.controller["path"], "learner checkpoint",
    )
    source_run = _canonical_external_path(
        starting_learner.controller["source_run"], "learner source run",
    )
    if (
        not source_run.is_dir()
        or checkpoint.parent != (source_run / "checkpoints").resolve()
        or not checkpoint.is_relative_to(source_run)
        or _sha256_file(checkpoint) != starting_learner.checkpoint_sha256
    ):
        raise ValueError("learner checkpoint physical hash or containment changed")
    learner_manifest_sha256 = _hash(
        learner["source_manifest_sha256"], "learner source manifest sha256",
    )
    source_manifest = source_run / "run.json"
    if (
        not source_manifest.is_file()
        or _sha256_file(source_manifest) != learner_manifest_sha256
    ):
        raise ValueError("learner source manifest physical hash changed")
    source_manifest_payload = _read_json(source_manifest)
    source_config = source_manifest_payload.get("config")
    source_contract = source_manifest_payload.get("contract")
    source_semantics = (
        source_contract.get("semantics")
        if isinstance(source_contract, Mapping)
        else None
    )
    if (
        not isinstance(source_config, Mapping)
        or source_config.get("environment") != "tactical-v2"
        or source_config.get("algorithm") != "maskable_ppo"
        or source_config.get("seed") != 227
        or source_manifest_payload.get("latest_checkpoint_step") != 38_912
        or not isinstance(source_contract, Mapping)
        or source_contract.get("version") != contract.version
        or source_contract.get("contract_hash") != contract.contract_hash
        or source_contract.get("encoding_hash") != contract.encoding_hash
        or source_contract.get("observation_size") != contract.observation_size
        or source_contract.get("action_size") != contract.action_size
        or not isinstance(source_semantics, Mapping)
        or source_semantics.get("action_regions")
        != contract_fields["action_regions"]
    ):
        raise ValueError("panel contract is not the locked learner contract")
    source_scenario = _canonical_external_path(
        str(source_run / "scenario.json"), "learner runtime scenario",
    )
    if (
        not source_scenario.is_file()
        or _sha256_file(source_scenario) != runtime_scenario_sha256
    ):
        raise ValueError("learner runtime scenario physical hash changed")

    dataset = _strict_fields(panel["original_dataset"], frozenset({
        "path", "manifest_sha256", "content_sha256", "file_count", "byte_size",
        "contract_hash", "encoding_hash", "scenario_hash",
    }), "original dataset")
    if dataset["path"] != _BASE_DATASET_PATH:
        raise ValueError("original dataset path is not the locked corpus")
    dataset_root = _canonical_external_path(dataset["path"], "original dataset")
    if not dataset_root.is_dir():
        raise ValueError("original dataset path must be a directory")
    dataset_manifest_sha256 = _hash(
        dataset["manifest_sha256"], "original dataset manifest sha256",
    )
    if dataset_manifest_sha256 != _BASE_DATASET_MANIFEST_SHA256:
        raise ValueError("original dataset identity is not the locked corpus")
    dataset_manifest_path = dataset_root / "manifest.json"
    if (
        not dataset_manifest_path.is_file()
        or _sha256_file(dataset_manifest_path) != dataset_manifest_sha256
    ):
        raise ValueError("original dataset physical hash changed")
    dataset_manifest = _read_json(dataset_manifest_path)
    dataset_contract_hash = _hash(
        dataset["contract_hash"], "original dataset contract hash",
    )
    dataset_encoding_hash = _hash(
        dataset["encoding_hash"], "original dataset encoding hash",
    )
    dataset_scenario_hash = _hash(
        dataset["scenario_hash"], "original dataset scenario hash",
    )
    dataset_content_sha256 = _hash(
        dataset["content_sha256"], "original dataset content sha256",
    )
    dataset_file_count = _strict_int(
        dataset["file_count"], "original dataset file count", minimum=1,
    )
    dataset_byte_size = _strict_int(
        dataset["byte_size"], "original dataset byte size", minimum=1,
    )
    if (
        dataset_manifest.get("contract_hash") != dataset_contract_hash
        or dataset_manifest.get("encoding_hash") != dataset_encoding_hash
        or dataset_encoding_hash != contract.encoding_hash
        or dataset_scenario_hash != canonical_scenario_hash
        or dataset_contract_hash != _BASE_DATASET_CONTRACT_HASH
        or dataset_encoding_hash != _BASE_DATASET_ENCODING_HASH
        or dataset_scenario_hash != _BASE_DATASET_SCENARIO_HASH
        or dataset_content_sha256 != _BASE_DATASET_CONTENT_SHA256
        or dataset_file_count != _BASE_DATASET_FILE_COUNT
        or dataset_byte_size != _BASE_DATASET_BYTE_SIZE
    ):
        raise ValueError("original dataset manifest or scenario identity changed")

    profiles = panel["profiles"]
    if not isinstance(profiles, list) or tuple(profiles) != _PANEL_PROFILES:
        raise ValueError("panel conversion profile order is invalid")

    oracle = _strict_fields(panel["oracle"], frozenset({
        "oracle_type", "heuristic_identity", "code_sha256", "candidates",
        "preflight",
    }), "panel oracle")
    if (
        oracle["oracle_type"] != "bounded-search"
        or oracle["heuristic_identity"] != "material-plus-pursuit-v1"
        or oracle["code_sha256"] != _ORACLE_CODE_SHA256
    ):
        raise ValueError("panel oracle identity is invalid")
    oracle_source = _repository_file(
        root, _ORACLE_SOURCE_PATH, "bounded-search oracle source",
    )
    if _sha256_file(oracle_source) != _ORACLE_CODE_SHA256:
        raise ValueError("bounded-search oracle source physical hash changed")
    candidates_raw = oracle["candidates"]
    if not isinstance(candidates_raw, list) or not _same_exact_json(
        candidates_raw,
        [
            {"depth": 4, "expansion_budget": 512, "use_heuristic": True},
            {"depth": 4, "expansion_budget": 2048, "use_heuristic": True},
        ],
    ):
        raise ValueError("panel oracle candidates are invalid")
    candidates = tuple(OracleSpec.from_dict({
        "oracle_type": oracle["oracle_type"],
        "depth": item["depth"],
        "expansion_budget": item["expansion_budget"],
        "use_heuristic": item["use_heuristic"],
        "heuristic_identity": oracle["heuristic_identity"],
        "code_hash": oracle["code_sha256"],
    }) for item in candidates_raw)
    preflight_raw = _exact_json(oracle["preflight"], {
        "maps_per_profile": 20,
        "games_per_candidate": 240,
        "queries_per_sample": 2,
        "pooled_win_rate_minimum_basis_points": 8500,
        "labels_per_second_minimum": 10.0,
        "tie_break": [
            "higher_win_rate", "fewer_cycling_draws", "higher_throughput",
            "smaller_expansion_budget",
        ],
    }, "oracle preflight")
    preflight = MappingProxyType({
        **dict(preflight_raw),
        "tie_break": tuple(preflight_raw["tie_break"]),
    })

    collection = _exact_json(panel["collection"], {
        "iterations": 3,
        "train_label_target": 20_000,
        "train_game_ceiling": 2_000,
        "validation_label_target": 2_000,
        "validation_game_ceiling": 200,
        "standard_basis_points": 7_000,
        "conversion_basis_points": 3_000,
        "opponent": "random",
        "both_seats": True,
    }, "collection")
    training = _exact_json(panel["training"], {
        "source_mixture_basis_points": {
            "greedy_standard": 4_900,
            "search_conversion": 2_100,
            "dagger_targeted": 3_000,
        },
        "batch_size": 256,
        "learning_rate": 3e-4,
        "max_epochs": 50,
        "patience": 5,
        "model_seed": 227,
        "sampler_seed": 227,
        "device": "cuda",
        "publication_device": "cpu",
        "objective": "actor_only_masked_cross_entropy",
        "validation_metric": "targeted_negative_log_likelihood",
    }, "training")
    evaluation = _exact_json(panel["evaluation"], {
        "maps": 100,
        "games_per_candidate": 200,
        "profile": "standard-3v3",
        "opponent": "random",
        "both_seats": True,
        "draws_are_non_wins": True,
        "tie_break": [
            "higher_win_rate", "lower_cycling_incidence",
            "lower_action_waste_incidence", "earlier_iteration",
        ],
    }, "evaluation")
    smoke = _exact_json(panel["smoke"], {
        "collection": [
            {"seed": 18_990_000, "profile": "standard-3v3", "seats": [0, 1]},
            {
                "seed": 18_990_001, "profile": "conversion-3v1-near",
                "seats": [0, 1],
            },
        ],
        "training_epochs": 1,
        "training_device": "cpu",
        "evaluation": [
            {"seed": 18_990_002, "profile": "standard-3v3", "seats": [0, 1]},
            {"seed": 18_990_003, "profile": "standard-3v3", "seats": [0, 1]},
        ],
        "required_collection_games": 4,
        "required_evaluation_games": 4,
        "require_reuse_new_games": 0,
    }, "smoke")
    success = _exact_json(panel["success"], {
        "win_rate_gain_minimum_basis_points": 2_000,
        "absolute_win_rate_minimum_basis_points": 6_500,
        "cycling_relative_reduction_minimum_basis_points": 5_000,
        "replicate_win_rate_minimum_basis_points": 6_500,
        "pooled_replication_win_rate_minimum_basis_points": 7_000,
    }, "success")

    return PanelDefinition(
        panel_path=panel_path,
        repository_root=root,
        panel_id=panel["id"],
        environment=panel["environment"],
        panel_sha256=_sha256_file(panel_path),
        seed_banks_sha256=seeds_sha256,
        scenario_path=scenario_path,
        scenario_sha256=scenario_sha256,
        runtime_scenario_sha256=runtime_scenario_sha256,
        contract_hash=contract.contract_hash,
        encoding_hash=contract.encoding_hash,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
        action_regions=_freeze_contract_value(action_regions, "panel action regions"),
        repository_policy=repository_policy,
        starting_learner=starting_learner,
        learner_source_manifest_sha256=learner_manifest_sha256,
        learner_source_scenario_sha256=runtime_scenario_sha256,
        dataset_root=dataset_root,
        dataset_manifest_sha256=dataset_manifest_sha256,
        dataset_content_sha256=dataset_content_sha256,
        dataset_file_count=dataset_file_count,
        dataset_byte_size=dataset_byte_size,
        dataset_contract_hash=dataset_contract_hash,
        dataset_encoding_hash=dataset_encoding_hash,
        dataset_scenario_hash=dataset_scenario_hash,
        profiles=tuple(profiles),
        oracle_candidates=candidates,
        preflight=preflight,
        collection=_freeze_contract_value(collection, "panel collection"),
        training=_freeze_contract_value(training, "panel training"),
        evaluation=_freeze_contract_value(evaluation, "panel evaluation"),
        smoke=_freeze_contract_value(smoke, "panel smoke"),
        success=_freeze_contract_value(success, "panel success"),
        seed_banks=seed_banks,
        preflight_schedule=preflight_schedule,
    )


def validate_panel_definition(definition: PanelDefinition) -> None:
    """Reopen a frozen definition and reject in-memory or physical drift."""

    if not isinstance(definition, PanelDefinition):
        raise TypeError("definition must be a PanelDefinition")
    reopened = load_panel_definition(
        definition.panel_path, repository_root=definition.repository_root,
    )
    if reopened != definition:
        raise ValueError("panel definition identity changed")


@dataclass(frozen=True)
class OraclePreflightGameResult:
    outcome: str
    cycling: bool
    action_waste: bool
    wasted_end_turns: int
    trace: EpisodeTrace
    replay: str
    samples: tuple["OracleBenchmarkSample", ...]

    def __post_init__(self) -> None:
        if self.outcome not in {"win", "loss", "draw"}:
            raise ValueError("preflight outcome is invalid")
        if type(self.cycling) is not bool or type(self.action_waste) is not bool:
            raise ValueError("preflight diagnostics must be boolean")
        _strict_int(
            self.wasted_end_turns, "preflight wasted EndTurn count", minimum=0,
        )
        if not isinstance(self.trace, EpisodeTrace) or not self.trace.transitions:
            raise ValueError("preflight trace must contain transitions")
        if not isinstance(self.replay, str) or not self.replay:
            raise ValueError("preflight replay must be non-empty text")
        if (
            not isinstance(self.samples, tuple)
            or not self.samples
            or len(self.samples) > _MAX_PREFLIGHT_SAMPLES_PER_GAME
            or any(not isinstance(item, OracleBenchmarkSample) for item in self.samples)
        ):
            raise ValueError(
                "preflight game benchmark sample limit is 1024"
            )


def _validate_oracle_state_payload_v2(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("oracle benchmark state must be an object")
    nodes = 0

    def visit(item: Any, depth: int) -> None:
        nonlocal nodes
        if depth > _MAX_ORACLE_STATE_DEPTH:
            raise ValueError("oracle benchmark state nesting depth is too large")
        nodes += 1
        if nodes > _MAX_ORACLE_STATE_NODES:
            raise ValueError("oracle benchmark state node count is too large")
        if isinstance(item, Mapping):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError("oracle benchmark state keys must be strings")
                visit(child, depth + 1)
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child, depth + 1)
            return
        if item is None or type(item) in {bool, str}:
            return
        if type(item) is int:
            if abs(item) > _INT32_MAX:
                raise ValueError(
                    "oracle benchmark state integer magnitude must fit int32"
                )
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("oracle benchmark state floats must be finite")
            return
        raise ValueError("oracle benchmark state contains an unsupported value")

    visit(value, 0)
    try:
        encoded = json.dumps(
            _mutable_json_value(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("oracle benchmark state is not finite JSON") from exc
    if len(encoded) > _MAX_ORACLE_STATE_BYTES:
        raise ValueError("oracle benchmark state byte size is too large")
    return _freeze_contract_value(value, "oracle benchmark state")


@dataclass(frozen=True)
class OracleBenchmarkSample:
    state_hash: str
    decision_index: int
    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    state: Mapping[str, Any]

    def __post_init__(self) -> None:
        frozen_state = _validate_oracle_state_payload_v2(self.state)
        if self.state_hash != _json_sha256(frozen_state):
            raise ValueError(
                'oracle benchmark canonical state hash does not match its state'
            )
        _hash(self.state_hash, "oracle benchmark state hash")
        _strict_int(
            self.decision_index, "oracle benchmark decision index", minimum=0,
        )
        if (
            not isinstance(self.observation, (tuple, list))
            or not self.observation
            or any(type(item) is not float or not math.isfinite(item)
                   for item in self.observation)
        ):
            raise ValueError("oracle benchmark observation must contain finite floats")
        if (
            not isinstance(self.legal_mask, (tuple, list))
            or not self.legal_mask
            or any(type(item) is not bool for item in self.legal_mask)
        ):
            raise ValueError("oracle benchmark legal mask is invalid")
        object.__setattr__(self, "observation", tuple(self.observation))
        object.__setattr__(self, "legal_mask", tuple(self.legal_mask))
        object.__setattr__(self, "state", frozen_state)

    @classmethod
    def from_dict(cls, value: Any) -> "OracleBenchmarkSample":
        fields = _strict_fields(value, frozenset({
            "state_hash", "decision_index", "observation", "legal_mask", "state",
        }), "oracle benchmark sample")
        return cls(
            state_hash=fields["state_hash"],
            decision_index=fields["decision_index"],
            observation=fields["observation"],
            legal_mask=fields["legal_mask"],
            state=fields["state"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_hash": self.state_hash,
            "decision_index": self.decision_index,
            "observation": list(self.observation),
            "legal_mask": list(self.legal_mask),
            "state": _mutable_json_value(self.state),
        }

    @property
    def content_sha256(self) -> str:
        return _json_sha256(self.to_dict())


_PRIVATE_TEST_TRUST_FIELDS = frozenset({
    "schema_version", "mode", "evidence_class", "engine_authenticated",
    "engine_evidence_root", "task_9_production_seal_required",
})


def _private_test_execution_trust_v3() -> dict[str, Any]:
    return dict(_PRIVATE_TEST_EXECUTION_TRUST)


def _validate_private_test_execution_trust_v3(
    value: Any, label: str,
) -> dict[str, Any]:
    fields = _strict_fields(value, _PRIVATE_TEST_TRUST_FIELDS, label)
    if (
        _strict_int(fields["schema_version"], f"{label} schema_version") != 1
        or fields["mode"] != "private-test-transcript"
        or fields["evidence_class"] != "untrusted-test-transcript"
        or fields["engine_authenticated"] is not False
        or fields["engine_evidence_root"] is not None
        or fields["task_9_production_seal_required"] is not True
    ):
        raise ValueError(f"{label} is not the exact private test trust declaration")
    return _private_test_execution_trust_v3()


@dataclass(frozen=True)
class OracleBenchmarkDecision:
    encoded_action: int
    command: Mapping[str, Any]
    actual_expansion_count: int

    def __post_init__(self) -> None:
        _strict_int(self.encoded_action, "oracle benchmark action", minimum=0)
        if not isinstance(self.command, Mapping):
            raise ValueError("oracle benchmark command is invalid")
        _strict_int(
            self.actual_expansion_count,
            "oracle benchmark expansion count",
            minimum=0,
        )
        object.__setattr__(
            self, "command",
            _freeze_contract_value(self.command, "oracle benchmark command"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "OracleBenchmarkDecision":
        fields = _strict_fields(value, frozenset({
            "encoded_action", "command", "actual_expansion_count",
        }), "oracle benchmark decision")
        return cls(**fields)

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoded_action": self.encoded_action,
            "command": _mutable_json_value(self.command),
            "actual_expansion_count": self.actual_expansion_count,
        }


@dataclass(frozen=True)
class OracleCodecEvidence:
    provenance: str
    state_hash: str
    contract_hash: str
    encoding_hash: str
    requested_action: int
    encoded_action: int
    encoded_command: Mapping[str, Any]
    decoded_command: Mapping[str, Any]
    mask_legal: bool
    apply_success: bool

    def __post_init__(self) -> None:
        if self.provenance != "private-test-callback":
            raise ValueError("oracle codec provenance is not private test callback evidence")
        _hash(self.state_hash, "oracle codec state hash")
        _hash(self.contract_hash, "oracle codec contract hash")
        _hash(self.encoding_hash, "oracle codec encoding hash")
        _strict_int(self.requested_action, "oracle codec requested action", minimum=0)
        _strict_int(self.encoded_action, "oracle codec encoded action", minimum=0)
        if not isinstance(self.encoded_command, Mapping):
            raise ValueError("oracle codec encoded command is invalid")
        if not isinstance(self.decoded_command, Mapping):
            raise ValueError("oracle codec decoded command is invalid")
        if type(self.mask_legal) is not bool or type(self.apply_success) is not bool:
            raise ValueError("oracle codec legality evidence must be boolean")
        object.__setattr__(
            self, "encoded_command",
            _freeze_contract_value(self.encoded_command, "oracle codec encoded command"),
        )
        object.__setattr__(
            self, "decoded_command",
            _freeze_contract_value(self.decoded_command, "oracle codec decoded command"),
        )

    @classmethod
    def from_dict(cls, value: Any) -> "OracleCodecEvidence":
        fields = _strict_fields(value, frozenset({
            "provenance", "execution_trust", "state_hash", "contract_hash",
            "encoding_hash",
            "requested_action", "encoded_action", "encoded_command",
            "decoded_command", "mask_legal", "apply_success",
        }), "private oracle codec callback record")
        _validate_private_test_execution_trust_v3(
            fields["execution_trust"], "oracle codec execution trust",
        )
        payload = dict(fields)
        del payload["execution_trust"]
        return cls(**payload)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance,
            "execution_trust": _private_test_execution_trust_v3(),
            "state_hash": self.state_hash,
            "contract_hash": self.contract_hash,
            "encoding_hash": self.encoding_hash,
            "requested_action": self.requested_action,
            "encoded_action": self.encoded_action,
            "encoded_command": _mutable_json_value(self.encoded_command),
            "decoded_command": _mutable_json_value(self.decoded_command),
            "mask_legal": self.mask_legal,
            "apply_success": self.apply_success,
        }


def _preflight_file_descriptor(root: Path, path: Path) -> dict[str, Any]:
    relative = path.relative_to(root).as_posix()
    sha256, byte_size = _opened_file_identity(path)
    return {
        "path": relative,
        "sha256": sha256,
        "byte_size": byte_size,
    }


def _opened_file_snapshot_v2(
    path: Path, *, max_bytes: int | None, label: str,
) -> bytes:
    _reject_reparse_chain(Path(path), label)
    with Path(path).open("rb") as stream:
        before = os.fstat(stream.fileno())
        payload = stream.read() if max_bytes is None else stream.read(max_bytes + 1)
        after = os.fstat(stream.fileno())
    before_identity = (
        before.st_size, before.st_mtime_ns, getattr(before, "st_ino", None),
    )
    after_identity = (
        after.st_size, after.st_mtime_ns, getattr(after, "st_ino", None),
    )
    if before_identity != after_identity:
        raise ValueError("authoritative file changed while its snapshot was read")
    if max_bytes is not None and len(payload) > max_bytes:
        raise ValueError(f"{label} byte size is too large")
    return payload


def _opened_file_identity(path: Path) -> tuple[str, int]:
    payload = _opened_file_snapshot_v2(
        path, max_bytes=None, label="authoritative file",
    )
    return hashlib.sha256(payload).hexdigest(), len(payload)


def _read_bounded_json_v2(
    path: Path, *, max_bytes: int, label: str,
) -> Mapping[str, Any]:
    payload = _opened_file_snapshot_v2(
        path, max_bytes=max_bytes, label=label,
    )
    try:
        value = json.loads(
            payload,
            parse_constant=lambda token: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {token}")
            ),
        )
    except (json.JSONDecodeError, UnicodeError) as exc:
        raise ValueError(f"invalid {label}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _bounded_atomic_write_json_v2(
    path: Path,
    value: Any,
    *,
    max_bytes: int,
    label: str,
) -> None:
    try:
        encoded = (
            json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite JSON") from exc
    if len(encoded) > max_bytes:
        raise ValueError(f"{label} byte size is too large")
    atomic_write_json(path, value)


def _dataset_tree_identity(root: Path) -> dict[str, Any]:
    root = Path(root)
    _reject_reparse_chain(root, "original dataset")
    files: list[Path] = []
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in [*directories, *names]:
            child = current_path / name
            if _is_reparse_point(child):
                raise ValueError(
                    "original dataset must not contain a symlink, junction, or reparse point"
                )
        files.extend(current_path / name for name in names)
    files.sort(key=lambda path: path.relative_to(root).as_posix())
    digest = hashlib.sha256()
    byte_size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        file_hash, file_size = _opened_file_identity(path)
        digest.update(relative.encode("utf-8"))
        digest.update(bytes((0,)))
        digest.update(file_hash.encode("ascii"))
        digest.update(b"\n")
        byte_size += file_size
    return {
        "content_sha256": digest.hexdigest(),
        "file_count": len(files),
        "byte_size": byte_size,
    }


def _audit_base_dataset(definition: PanelDefinition) -> dict[str, Any]:
    physical_before = _dataset_tree_identity(definition.dataset_root)
    expected_physical = {
        "content_sha256": definition.dataset_content_sha256,
        "file_count": definition.dataset_file_count,
        "byte_size": definition.dataset_byte_size,
    }
    if physical_before != expected_physical:
        raise ValueError("original dataset full content identity changed")
    cache_key = (
        physical_before["content_sha256"],
        physical_before["file_count"],
        physical_before["byte_size"],
        definition.dataset_scenario_hash,
        definition.dataset_contract_hash,
        definition.dataset_encoding_hash,
        _BASE_DATASET_AUDIT_VERSION,
    )
    cached_audit = _BASE_DATASET_SEMANTIC_AUDIT_CACHE.get(cache_key)
    if cached_audit is None:
        contract = EnvironmentContract(
            version="tactical-v2",
            contract_hash=definition.dataset_contract_hash,
            encoding_hash=definition.dataset_encoding_hash,
            observation_size=definition.observation_size,
            action_size=definition.action_size,
            board={},
            roster=[],
            reward={},
            semantics={"action_regions": definition.action_regions},
        )
        dataset = load_imitation_dataset(
            definition.dataset_root, expected_contract=contract,
        )
        audit = dict(audit_imitation_dataset(dataset))
    else:
        audit = dict(cached_audit)
    physical_after = _dataset_tree_identity(definition.dataset_root)
    if physical_after != physical_before or physical_after != expected_physical:
        raise ValueError("original dataset changed during its stable semantic audit")
    if cached_audit is None:
        _BASE_DATASET_SEMANTIC_AUDIT_CACHE[cache_key] = dict(audit)
    if (
        audit["games"] <= 0
        or audit["teacher_labels"] <= 0
        or audit["masked_labels"] != 0
        or audit["round_trip_mismatches"] != 0
        or audit["replay_mismatches"] != 0
    ):
        raise ValueError("original dataset full semantic audit failed")
    return {**physical_after, "audit": dict(audit)}


def _git_object_id(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None
    ):
        raise ValueError(f"{label} must be a lowercase Git object id")
    return value


def _repository_execution_identity(repository_root: Path) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise ValueError(
                f"repository identity command failed: {' '.join(arguments)}"
            )
        return completed.stdout.strip()

    discovered = Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)
    if discovered != root:
        raise ValueError("panel repository root is not the Git toplevel")
    return {
        "root": str(root),
        "commit": git("rev-parse", "HEAD").lower(),
        "source_tree": git("rev-parse", "HEAD^{tree}").lower(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def _validated_repository_identity(
    definition: PanelDefinition,
    provider: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    raw = _strict_fields(
        provider(definition.repository_root),
        frozenset({"root", "commit", "source_tree", "dirty"}),
        "repository execution identity",
    )
    try:
        root = Path(_strict_string(raw["root"], "repository identity root")).resolve(
            strict=True,
        )
    except OSError as exc:
        raise ValueError("repository identity root is missing") from exc
    if root != definition.repository_root:
        raise ValueError("repository identity root changed")
    if type(raw["dirty"]) is not bool or raw["dirty"]:
        raise ValueError("oracle preflight requires a clean repository")
    return {
        "root": str(root),
        "commit": _git_object_id(raw["commit"], "repository commit"),
        "source_tree": _git_object_id(raw["source_tree"], "repository source tree"),
        "dirty": False,
    }


def _oracle_preflight_identity(
    definition: PanelDefinition,
    repository: Mapping[str, Any],
    dataset: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "execution_trust": _private_test_execution_trust_v3(),
        "schema_version": 1,
        "panel_id": definition.panel_id,
        "panel_sha256": definition.panel_sha256,
        "seed_banks_sha256": definition.seed_banks_sha256,
        "scenario_sha256": definition.scenario_sha256,
        "runtime_scenario_sha256": definition.runtime_scenario_sha256,
        "contract_hash": definition.contract_hash,
        "encoding_hash": definition.encoding_hash,
        "repository": dict(repository),
        "starting_learner": definition.starting_learner.to_dict(),
        "learner_source_manifest_sha256": (
            definition.learner_source_manifest_sha256
        ),
        "original_dataset": {
            "manifest_sha256": definition.dataset_manifest_sha256,
            "contract_hash": definition.dataset_contract_hash,
            "encoding_hash": definition.dataset_encoding_hash,
            "scenario_hash": definition.dataset_scenario_hash,
            "content_sha256": dataset["content_sha256"],
            "file_count": dataset["file_count"],
            "byte_size": dataset["byte_size"],
            "audit": dict(dataset["audit"]),
        },
        "profiles": list(definition.profiles),
        "oracle_candidates": [
            candidate.to_dict() for candidate in definition.oracle_candidates
        ],
        "preflight": {
            "maps_per_profile": definition.preflight["maps_per_profile"],
            "games_per_candidate": definition.preflight["games_per_candidate"],
            "queries_per_sample": definition.preflight["queries_per_sample"],
            "pooled_win_rate_minimum_basis_points": (
                definition.preflight["pooled_win_rate_minimum_basis_points"]
            ),
            "labels_per_second_minimum": (
                definition.preflight["labels_per_second_minimum"]
            ),
            "tie_break": list(definition.preflight["tie_break"]),
        },
        "teacher_schedule": [
            game.to_dict() for game in definition.preflight_schedule
        ],
        "teacher_schedule_sha256": _schedule_identity(
            definition.preflight_schedule,
        ),
    }


def _validate_preflight_game_result(
    result: OraclePreflightGameResult, game: ScheduledDuel,
) -> None:
    if not isinstance(result, OraclePreflightGameResult):
        raise ValueError("preflight evaluator returned an invalid game result")
    trace = result.trace
    if (
        not trace.transitions
        or not trace.transitions[-1].after.is_game_over
    ):
        raise ValueError("preflight evaluator trace is not terminal")
    winner = trace.transitions[-1].after.winner
    expected = (
        "draw"
        if winner is None
        else "win"
        if winner == game.learner_seat
        else "loss"
    )
    if result.outcome != expected:
        raise ValueError("preflight outcome does not match its trace")
    diagnostics = _preflight_trace_diagnostics(
        trace, learner_seat=game.learner_seat, outcome=result.outcome,
    )
    if (
        result.cycling != diagnostics["cycling"]
        or result.action_waste != diagnostics["action_waste"]
        or result.wasted_end_turns != diagnostics["wasted_end_turns"]
    ):
        raise ValueError(
            "preflight cycling or action-waste diagnostics do not match the trace"
        )


def _preflight_trace_diagnostics(
    trace: EpisodeTrace, *, learner_seat: int, outcome: str,
) -> dict[str, Any]:
    summary = summarize_episode(trace, learner_seat)
    winner = trace.transitions[-1].after.winner
    classification = classify_draw(
        trace,
        candidate_seat=learner_seat,
        terminated=True,
        truncated=False,
        winner=winner,
    )
    flags = set(classification.flags)
    return {
        "cycling": (
            outcome == "draw" and DrawCategory.CYCLING in flags
        ),
        "action_waste": DrawCategory.ACTION_WASTE in flags,
        "wasted_end_turns": summary.wasted_end_turns_by_seat[learner_seat],
    }


def _select_preflight_candidate(
    summaries: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    eligible = [summary for summary in summaries if summary["eligible"] is True]
    if not eligible:
        return None
    return min(eligible, key=lambda summary: (
        -float(summary["rates"]["win"]),
        int(summary["cycling_draws"]),
        -float(summary["labels_per_second"]),
        int(summary["oracle"]["expansion_budget"]),
    ))


def _validate_preflight_file(
    root: Path, value: Any, label: str,
) -> Path:
    descriptor = _strict_fields(value, _PREFLIGHT_FILE_FIELDS, label)
    path = _contained_file(root, descriptor["path"], label)
    sha256 = _hash(descriptor["sha256"], f"{label} sha256")
    byte_size = _strict_int(
        descriptor["byte_size"], f"{label} byte size", minimum=1,
    )
    actual_sha256, actual_byte_size = _opened_file_identity(path)
    if actual_byte_size != byte_size or actual_sha256 != sha256:
        raise ValueError(f"{label} physical hash or size changed")
    return path


_PREFLIGHT_V2_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "identity", "candidates", "selected_oracle",
    "games", "content_identity",
})
_PREFLIGHT_FILE_FIELDS = frozenset({"path", "sha256", "byte_size"})
_PREFLIGHT_V2_GAME_FIELDS = frozenset({
    "candidate_index", "game_index", "schedule_index", "map_seed",
    "episode_seed", "profile", "reference_seat", "learner_seat", "outcome",
    "cycling", "action_waste", "wasted_end_turns", "transition_count",
    "sample_count", "trace", "replay", "benchmark",
})
_PREFLIGHT_V2_CANDIDATE_FIELDS = frozenset({
    "oracle", "games", "wins", "losses", "draws", "rates",
    "confidence_intervals", "cycling_draws", "action_waste_games",
    "wasted_end_turns", "paired_maps", "seats", "profiles", "samples",
    "query_count", "labels", "determinism_failures", "round_trip_failures",
    "expansion_total", "max_expansions", "mean_expansions",
    "benchmark_seconds", "labels_per_second", "eligible",
})
_PREFLIGHT_TRACE_ENVELOPE_FIELDS = frozenset({
    "schema_version", "candidate_index", "game_index", "schedule", "outcome",
    "execution_trust", "trace",
})
_PREFLIGHT_REPLAY_ENVELOPE_FIELDS = frozenset({
    "schema_version", "candidate_index", "game_index", "schedule", "outcome",
    "execution_trust", "payload", "payload_sha256",
})
_PREFLIGHT_BENCHMARK_ENVELOPE_FIELDS = frozenset({
    "schema_version", "candidate_index", "game_index", "schedule",
    "execution_trust", "records",
})
_PREFLIGHT_BENCHMARK_RECORD_FIELDS = frozenset({
    "sample_index", "sample_sha256", "sample", "first", "second",
    "pair_seconds", "execution_trust",
})
_PREFLIGHT_QUERY_FIELDS = frozenset({
    "decision", "codec", "elapsed_seconds",
})


def _preflight_decision_valid_v2(
    decision: OracleBenchmarkDecision,
    *,
    sample: OracleBenchmarkSample,
    oracle: OracleSpec,
    game: ScheduledDuel,
    definition: PanelDefinition,
) -> None:
    if not isinstance(decision, OracleBenchmarkDecision):
        raise ValueError("oracle benchmark returned an invalid decision")
    if (
        len(sample.observation) != definition.observation_size
        or len(sample.legal_mask) != definition.action_size
    ):
        raise ValueError("oracle benchmark sample shape changed")
    action = decision.encoded_action
    if action >= definition.action_size or not sample.legal_mask[action]:
        raise ValueError("oracle benchmark action is masked")
    contract = EnvironmentContract(
        version="tactical-v2",
        contract_hash=definition.contract_hash,
        encoding_hash=definition.encoding_hash,
        observation_size=definition.observation_size,
        action_size=definition.action_size,
        board={},
        roster=[],
        reward={},
        semantics={"action_regions": definition.action_regions},
    )
    _command(
        decision.command,
        seat=game.learner_seat,
        action=action,
        contract=contract,
        label="oracle benchmark",
    )
    if decision.actual_expansion_count > oracle.expansion_budget:
        raise ValueError("oracle benchmark exceeded its expansion budget")


def _preflight_codec_valid_v2(
    evidence: OracleCodecEvidence,
    *,
    decision: OracleBenchmarkDecision,
    sample: OracleBenchmarkSample,
    definition: PanelDefinition,
) -> bool:
    if not isinstance(evidence, OracleCodecEvidence):
        raise ValueError("oracle codec callback returned invalid evidence")
    action = decision.encoded_action
    return (
        evidence.provenance == "private-test-callback"
        and evidence.state_hash == sample.state_hash
        and evidence.contract_hash == definition.contract_hash
        and evidence.encoding_hash == definition.encoding_hash
        and evidence.requested_action == action
        and evidence.encoded_action == action
        and evidence.encoded_command == decision.command
        and evidence.decoded_command == decision.command
        and evidence.mask_legal is True
        and evidence.apply_success is True
        and action < len(sample.legal_mask)
        and sample.legal_mask[action]
    )


def _preflight_query_from_payload_v2(
    value: Any,
    *,
    sample: OracleBenchmarkSample,
    oracle: OracleSpec,
    game: ScheduledDuel,
    definition: PanelDefinition,
) -> tuple[OracleBenchmarkDecision, bool, float]:
    fields = _strict_fields(value, _PREFLIGHT_QUERY_FIELDS, "preflight query")
    decision = OracleBenchmarkDecision.from_dict(fields["decision"])
    evidence = OracleCodecEvidence.from_dict(fields["codec"])
    elapsed = fields["elapsed_seconds"]
    if type(elapsed) is not float or not math.isfinite(elapsed) or elapsed < 0.0:
        raise ValueError("preflight query elapsed_seconds must be a finite float")
    _preflight_decision_valid_v2(
        decision, sample=sample, oracle=oracle, game=game, definition=definition,
    )
    valid = _preflight_codec_valid_v2(
        evidence, decision=decision, sample=sample, definition=definition,
    )
    return decision, valid, elapsed


def _preflight_profile_counts_v2(
    games: Sequence[Mapping[str, Any]], profiles: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for profile in profiles:
        selected = [game for game in games if game["profile"] == profile]
        counts = Counter(game["outcome"] for game in selected)
        seats = {}
        for seat in (0, 1):
            by_seat = [game for game in selected if game["learner_seat"] == seat]
            by_outcome = Counter(game["outcome"] for game in by_seat)
            seats[str(seat)] = {
                "games": len(by_seat),
                "wins": by_outcome["win"],
                "losses": by_outcome["loss"],
                "draws": by_outcome["draw"],
            }
        result[profile] = {
            "games": len(selected),
            "wins": counts["win"],
            "losses": counts["loss"],
            "draws": counts["draw"],
            "seats": seats,
        }
    return result


def _preflight_candidate_summary_v2(
    *,
    oracle: OracleSpec,
    games: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    definition: PanelDefinition,
) -> dict[str, Any]:
    counts = Counter(game["outcome"] for game in games)
    total = len(games)
    seats = {}
    for seat in (0, 1):
        selected = [game for game in games if game["learner_seat"] == seat]
        by_outcome = Counter(game["outcome"] for game in selected)
        seats[str(seat)] = {
            "games": len(selected),
            "wins": by_outcome["win"],
            "losses": by_outcome["loss"],
            "draws": by_outcome["draw"],
        }
    determinism_failures = sum(
        item["first_decision"] != item["second_decision"] for item in records
    )
    round_trip_failures = sum(
        int(not item["first_codec_valid"]) + int(not item["second_codec_valid"])
        for item in records
    )
    labels = sum(
        item["first_decision"] == item["second_decision"]
        and item["first_codec_valid"]
        and item["second_codec_valid"]
        for item in records
    )
    decisions = [
        item[key]
        for item in records
        for key in ("first_decision", "second_decision")
    ]
    query_count = len(decisions)
    expansion_total = sum(item.actual_expansion_count for item in decisions)
    max_expansions = max(
        (item.actual_expansion_count for item in decisions), default=0,
    )
    benchmark_seconds = sum(
        item["first_seconds"] + item["second_seconds"] for item in records
    )
    throughput = labels / benchmark_seconds if benchmark_seconds > 0.0 else 0.0
    win_rate = counts["win"] / total if total else 0.0
    eligible = (
        total == definition.preflight["games_per_candidate"]
        and win_rate
        >= definition.preflight["pooled_win_rate_minimum_basis_points"] / 10_000
        and determinism_failures == 0
        and round_trip_failures == 0
        and labels > 0
        and throughput >= definition.preflight["labels_per_second_minimum"]
    )
    return {
        "oracle": oracle.to_dict(),
        "games": total,
        "wins": counts["win"],
        "losses": counts["loss"],
        "draws": counts["draw"],
        "rates": {
            "win": win_rate,
            "loss": counts["loss"] / total if total else 0.0,
            "draw": counts["draw"] / total if total else 0.0,
        },
        "confidence_intervals": {
            name: wilson_interval(counts[counter], total, 0.95)
            for name, counter in (
                ("win", "win"), ("loss", "loss"), ("draw", "draw")
            )
        },
        "cycling_draws": sum(
            game["cycling"] and game["outcome"] == "draw" for game in games
        ),
        "action_waste_games": sum(game["action_waste"] for game in games),
        "wasted_end_turns": sum(game["wasted_end_turns"] for game in games),
        "paired_maps": total // 2,
        "seats": seats,
        "profiles": _preflight_profile_counts_v2(games, definition.profiles),
        "samples": len(records),
        "query_count": query_count,
        "labels": labels,
        "determinism_failures": determinism_failures,
        "round_trip_failures": round_trip_failures,
        "expansion_total": expansion_total,
        "max_expansions": max_expansions,
        "mean_expansions": expansion_total / query_count if query_count else 0.0,
        "benchmark_seconds": benchmark_seconds,
        "labels_per_second": throughput,
        "eligible": eligible,
    }


def _preflight_trace_envelope_v2(
    *, candidate_index: int, game_index: int, game: ScheduledDuel,
    result: OraclePreflightGameResult,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_trust": _private_test_execution_trust_v3(),
        "candidate_index": candidate_index,
        "game_index": game_index,
        "schedule": game.to_dict(),
        "outcome": result.outcome,
        "trace": result.trace.to_dict(),
    }


def _preflight_replay_envelope_v2(
    *, candidate_index: int, game_index: int, game: ScheduledDuel,
    result: OraclePreflightGameResult,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "execution_trust": _private_test_execution_trust_v3(),
        "candidate_index": candidate_index,
        "game_index": game_index,
        "schedule": game.to_dict(),
        "outcome": result.outcome,
        "payload": result.replay,
        "payload_sha256": hashlib.sha256(result.replay.encode("utf-8")).hexdigest(),
    }


def _preflight_benchmark_record_v2(
    *, sample_index: int, sample: OracleBenchmarkSample,
    first: OracleBenchmarkDecision, first_codec: OracleCodecEvidence,
    first_seconds: float, second: OracleBenchmarkDecision,
    second_codec: OracleCodecEvidence, second_seconds: float,
) -> dict[str, Any]:
    return {
        "execution_trust": _private_test_execution_trust_v3(),
        "sample_index": sample_index,
        "sample_sha256": sample.content_sha256,
        "sample": sample.to_dict(),
        "first": {
            "decision": first.to_dict(),
            "codec": first_codec.to_dict(),
            "elapsed_seconds": first_seconds,
        },
        "second": {
            "decision": second.to_dict(),
            "codec": second_codec.to_dict(),
            "elapsed_seconds": second_seconds,
        },
        "pair_seconds": first_seconds + second_seconds,
    }


def _validate_preflight_envelope_context_v2(
    fields: Mapping[str, Any],
    *,
    candidate_index: int,
    game_index: int,
    game: ScheduledDuel,
    label: str,
) -> None:
    _validate_private_test_execution_trust_v3(
        fields["execution_trust"], f"{label} execution trust",
    )
    schema_version = _strict_int(
        fields["schema_version"], f"{label} schema_version",
    )
    supplied_candidate = _strict_int(
        fields["candidate_index"], f"{label} candidate index", minimum=0,
    )
    supplied_game = _strict_int(
        fields["game_index"], f"{label} game index", minimum=0,
    )
    if (
        schema_version != 1
        or supplied_candidate != candidate_index
        or supplied_game != game_index
        or not _same_exact_json(fields["schedule"], game.to_dict())
    ):
        raise ValueError(f"{label} context does not match its manifest game")


def _open_oracle_preflight_v2(
    root: Path, *, expected_identity: Mapping[str, Any],
    definition: PanelDefinition,
) -> OracleSpec:
    canonical_root = _lexical_canonical_path(
        Path(root), "oracle preflight root", strict=True,
    )
    if not canonical_root.is_dir():
        raise ValueError("oracle preflight root must be a directory")
    for current, directories, files in os.walk(canonical_root, followlinks=False):
        for name in [*directories, *files]:
            if _is_reparse_point(Path(current) / name):
                raise ValueError("oracle preflight evidence contains a reparse point")
    manifest = _strict_fields(
        _read_bounded_json_v2(
            canonical_root / "oracle-preflight.json",
            max_bytes=_MAX_PREFLIGHT_MANIFEST_BYTES,
            label="oracle preflight manifest",
        ),
        _PREFLIGHT_V2_MANIFEST_FIELDS,
        "oracle preflight manifest",
    )
    if (
        _strict_int(
            manifest["schema_version"], "oracle preflight manifest schema_version",
        ) != 2
        or manifest["status"] != "completed"
    ):
        raise ValueError("oracle preflight is not completed")
    if not _same_exact_json(manifest["identity"], expected_identity):
        raise ValueError("oracle preflight identity changed")
    identity = manifest["identity"]
    if not isinstance(identity, Mapping):
        raise ValueError("oracle preflight identity must be an object")
    _validate_private_test_execution_trust_v3(
        identity.get("execution_trust"), "oracle preflight identity execution trust",
    )
    if _content_identity(manifest) != _hash(
        manifest["content_identity"], "oracle preflight content identity",
    ):
        raise ValueError("oracle preflight manifest content identity changed")
    candidates = manifest["candidates"]
    games = manifest["games"]
    expected_candidates = expected_identity["oracle_candidates"]
    expected_schedule = expected_identity["teacher_schedule"]
    if (
        type(candidates) is not list
        or len(candidates) != len(expected_candidates)
        or type(games) is not list
        or len(games) != len(expected_candidates) * len(expected_schedule)
    ):
        raise ValueError("oracle preflight candidate or game count changed")

    summaries: list[Mapping[str, Any]] = []
    by_candidate_games: dict[int, list[Mapping[str, Any]]] = {
        index: [] for index in range(len(candidates))
    }
    by_candidate_records: dict[int, list[Mapping[str, Any]]] = {
        index: [] for index in range(len(candidates))
    }
    state_hashes: dict[int, set[str]] = {
        index: set() for index in range(len(candidates))
    }
    owned = {"oracle-preflight.json"}
    descriptor_paths: set[str] = set()

    for position, raw_game in enumerate(games):
        record = _strict_fields(
            raw_game, _PREFLIGHT_V2_GAME_FIELDS, "oracle preflight game",
        )
        candidate_index = _strict_int(
            record["candidate_index"], "preflight candidate index", minimum=0,
        )
        game_index = _strict_int(
            record["game_index"], "preflight game index", minimum=0,
        )
        if (
            candidate_index not in by_candidate_games
            or game_index >= len(expected_schedule)
            or position != candidate_index * len(expected_schedule) + game_index
        ):
            raise ValueError("preflight game ordering changed")
        game = ScheduledDuel(**expected_schedule[game_index])
        for field, expected in game.to_dict().items():
            if type(record[field]) is not type(expected) or record[field] != expected:
                raise ValueError("preflight game schedule changed")
        if record["outcome"] not in {"win", "loss", "draw"}:
            raise ValueError("preflight game outcome is invalid")
        if type(record["cycling"]) is not bool or type(record["action_waste"]) is not bool:
            raise ValueError("preflight game diagnostics are invalid")
        wasted = _strict_int(
            record["wasted_end_turns"], "preflight wasted EndTurns", minimum=0,
        )
        transitions = _strict_int(
            record["transition_count"], "preflight transition count", minimum=1,
        )
        sample_count = _strict_int(
            record["sample_count"], "preflight sample count", minimum=1,
        )
        paths: dict[str, Path] = {}
        for label in ("trace", "replay", "benchmark"):
            descriptor = _strict_fields(
                record[label], _PREFLIGHT_FILE_FIELDS, f"preflight {label}",
            )
            relative = _safe_relative(
                descriptor["path"], f"preflight {label} path",
            )
            if relative in descriptor_paths:
                raise ValueError("preflight evidence descriptor path is duplicated")
            descriptor_paths.add(relative)
            _reject_reparse_chain(
                canonical_root / relative, f"preflight {label}",
            )
            paths[label] = _validate_preflight_file(
                canonical_root, descriptor, f"preflight {label}",
            )
            owned.add(relative)

        trace_fields = _strict_fields(
            _read_bounded_json_v2(
                paths["trace"],
                max_bytes=_MAX_PREFLIGHT_TRACE_BYTES,
                label="preflight trace envelope",
            ),
            _PREFLIGHT_TRACE_ENVELOPE_FIELDS,
            "preflight trace envelope",
        )
        _validate_preflight_envelope_context_v2(
            trace_fields, candidate_index=candidate_index, game_index=game_index,
            game=game, label="preflight trace",
        )
        if trace_fields["outcome"] != record["outcome"]:
            raise ValueError("preflight trace outcome context changed")
        trace = EpisodeTrace.from_payload(trace_fields["trace"])
        if (
            len(trace.transitions) != transitions
            or not trace.transitions[-1].after.is_game_over
        ):
            raise ValueError("preflight trace terminal evidence changed")
        winner = trace.transitions[-1].after.winner
        outcome = (
            "draw" if winner is None else
            "win" if winner == game.learner_seat else "loss"
        )
        diagnostics = _preflight_trace_diagnostics(
            trace, learner_seat=game.learner_seat, outcome=outcome,
        )
        if (
            outcome != record["outcome"]
            or record["cycling"] != diagnostics["cycling"]
            or record["action_waste"] != diagnostics["action_waste"]
            or wasted != diagnostics["wasted_end_turns"]
        ):
            raise ValueError("preflight physical trace diagnostics changed")

        replay_fields = _strict_fields(
            _read_bounded_json_v2(
                paths["replay"],
                max_bytes=_MAX_PREFLIGHT_REPLAY_BYTES,
                label="preflight replay envelope",
            ),
            _PREFLIGHT_REPLAY_ENVELOPE_FIELDS,
            "preflight replay envelope",
        )
        _validate_preflight_envelope_context_v2(
            replay_fields, candidate_index=candidate_index, game_index=game_index,
            game=game, label="preflight replay",
        )
        payload = replay_fields["payload"]
        if (
            replay_fields["outcome"] != outcome
            or not isinstance(payload, str)
            or not payload
            or _hash(replay_fields["payload_sha256"], "preflight replay payload hash")
            != hashlib.sha256(payload.encode("utf-8")).hexdigest()
        ):
            raise ValueError("preflight replay semantic binding changed")

        benchmark_fields = _strict_fields(
            _read_bounded_json_v2(
                paths["benchmark"],
                max_bytes=_MAX_PREFLIGHT_BENCHMARK_BYTES,
                label="preflight benchmark envelope",
            ),
            _PREFLIGHT_BENCHMARK_ENVELOPE_FIELDS,
            "preflight benchmark envelope",
        )
        _validate_preflight_envelope_context_v2(
            benchmark_fields, candidate_index=candidate_index,
            game_index=game_index, game=game, label="preflight benchmark",
        )
        benchmark_records = benchmark_fields["records"]
        if type(benchmark_records) is not list or len(benchmark_records) != sample_count:
            raise ValueError("preflight benchmark sample count changed")
        oracle = OracleSpec.from_dict(expected_candidates[candidate_index])
        for sample_index, raw_benchmark in enumerate(benchmark_records):
            benchmark_record = _strict_fields(
                raw_benchmark, _PREFLIGHT_BENCHMARK_RECORD_FIELDS,
                "preflight benchmark record",
            )
            _validate_private_test_execution_trust_v3(
                benchmark_record["execution_trust"],
                "preflight benchmark record execution trust",
            )
            if (
                _strict_int(
                    benchmark_record["sample_index"],
                    "preflight benchmark sample index",
                    minimum=0,
                )
                != sample_index
            ):
                raise ValueError("preflight benchmark sample ordering changed")
            sample = OracleBenchmarkSample.from_dict(benchmark_record["sample"])
            if (
                sample.content_sha256
                != _hash(benchmark_record["sample_sha256"], "preflight sample sha256")
                or len(sample.observation) != definition.observation_size
                or len(sample.legal_mask) != definition.action_size
            ):
                raise ValueError("preflight benchmark sample identity or shape changed")
            if sample.state_hash in state_hashes[candidate_index]:
                raise ValueError("preflight benchmark state identity is duplicated")
            state_hashes[candidate_index].add(sample.state_hash)
            first, first_valid, first_seconds = _preflight_query_from_payload_v2(
                benchmark_record["first"], sample=sample, oracle=oracle, game=game,
                definition=definition,
            )
            second, second_valid, second_seconds = _preflight_query_from_payload_v2(
                benchmark_record["second"], sample=sample, oracle=oracle, game=game,
                definition=definition,
            )
            pair_seconds = benchmark_record["pair_seconds"]
            if (
                type(pair_seconds) is not float
                or not math.isfinite(pair_seconds)
                or pair_seconds < 0.0
                or pair_seconds != first_seconds + second_seconds
            ):
                raise ValueError("preflight benchmark pair timing changed")
            by_candidate_records[candidate_index].append({
                "first_decision": first,
                "second_decision": second,
                "first_codec_valid": first_valid,
                "second_codec_valid": second_valid,
                "first_seconds": first_seconds,
                "second_seconds": second_seconds,
            })
        by_candidate_games[candidate_index].append({
            **game.to_dict(),
            "outcome": outcome,
            "cycling": record["cycling"],
            "action_waste": record["action_waste"],
            "wasted_end_turns": wasted,
        })

    return _finish_open_oracle_preflight_v2(
        canonical_root=canonical_root,
        manifest=manifest,
        candidates=candidates,
        expected_candidates=expected_candidates,
        by_candidate_games=by_candidate_games,
        by_candidate_records=by_candidate_records,
        owned=owned,
        definition=definition,
    )


def _finish_open_oracle_preflight_v2(
    *,
    canonical_root: Path,
    manifest: Mapping[str, Any],
    candidates: Sequence[Any],
    expected_candidates: Sequence[Any],
    by_candidate_games: Mapping[int, Sequence[Mapping[str, Any]]],
    by_candidate_records: Mapping[int, Sequence[Mapping[str, Any]]],
    owned: set[str],
    definition: PanelDefinition,
) -> OracleSpec:
    actual = {
        path.relative_to(canonical_root).as_posix()
        for path in canonical_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != owned:
        raise ValueError("oracle preflight contains missing or unowned evidence")
    summaries: list[Mapping[str, Any]] = []
    for index, (raw, expected_oracle) in enumerate(zip(
        candidates, expected_candidates, strict=True,
    )):
        supplied = _strict_fields(
            raw, _PREFLIGHT_V2_CANDIDATE_FIELDS, "oracle preflight candidate",
        )
        oracle = OracleSpec.from_dict(expected_oracle)
        recomputed = _preflight_candidate_summary_v2(
            oracle=oracle,
            games=by_candidate_games[index],
            records=by_candidate_records[index],
            definition=definition,
        )
        if not _same_exact_json(supplied, recomputed):
            raise ValueError(
                "oracle preflight summary metrics do not match physical evidence"
            )
        for profile in definition.profiles:
            profile_summary = supplied["profiles"][profile]
            if (
                profile_summary["games"] != 40
                or profile_summary["seats"]["0"]["games"] != 20
                or profile_summary["seats"]["1"]["games"] != 20
            ):
                raise ValueError("oracle preflight per-profile coverage changed")
        summaries.append(supplied)
    selected_summary = _select_preflight_candidate(summaries)
    if selected_summary is None:
        raise ValueError("completed oracle preflight has no eligible candidate")
    selected = OracleSpec.from_dict(manifest["selected_oracle"])
    if not _same_exact_json(selected.to_dict(), selected_summary["oracle"]):
        raise ValueError("oracle preflight selected candidate changed")
    return selected


def _reserve_preflight_diagnostic_root_v2(
    destination: Path,
) -> tuple[Path, Path]:
    parent = destination.parent / f"{destination.name}.diagnostics"
    _lexical_canonical_path(
        parent, "oracle preflight diagnostics", strict=False,
    )
    parent.mkdir(parents=True, exist_ok=True)
    for index in range(1_000_000):
        target = parent / f"attempt-{index:06d}"
        marker = parent / f".attempt-{index:06d}.reserve"
        try:
            descriptor = os.open(
                marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600,
            )
        except FileExistsError:
            continue
        os.close(descriptor)
        if target.exists():
            marker.unlink()
            continue
        return target, marker
    raise RuntimeError("oracle preflight diagnostic namespace is exhausted")


def _move_to_preflight_diagnostic_v2(
    source: Path, *, destination: Path,
) -> Path:
    _validate_preflight_diagnostic_tree_v3(source)
    while True:
        target, marker = _reserve_preflight_diagnostic_root_v2(destination)
        try:
            os.replace(source, target)
        except OSError:
            if not target.exists():
                raise
        else:
            return target
        finally:
            marker.unlink(missing_ok=True)


def _validate_preflight_diagnostic_tree_v3(source: Path) -> dict[str, int]:
    canonical = _lexical_canonical_path(
        Path(source), "oracle preflight diagnostic source", strict=True,
    )
    if not canonical.is_dir():
        raise ValueError("oracle preflight diagnostic source must be a directory")
    file_count = 0
    byte_size = 0
    for current, directories, files in os.walk(canonical, followlinks=False):
        current_path = Path(current)
        for name in directories:
            _reject_reparse_chain(
                current_path / name, "oracle preflight diagnostic directory",
            )
        for name in files:
            path = current_path / name
            _reject_reparse_chain(path, "oracle preflight diagnostic file")
            file_count += 1
            if file_count > _MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILES:
                raise ValueError(
                    "oracle preflight diagnostic file-count limit exceeded"
                )
            payload = _opened_file_snapshot_v2(
                path,
                max_bytes=_MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILE_BYTES,
                label="oracle preflight diagnostic file",
            )
            byte_size += len(payload)
            if byte_size > _MAX_PREFLIGHT_DIAGNOSTIC_TREE_BYTES:
                raise ValueError(
                    "oracle preflight diagnostic total-byte limit exceeded"
                )
    return {"file_count": file_count, "byte_size": byte_size}


def _seal_preflight_diagnostic_v2(
    staging: Path,
    *,
    destination: Path,
    error: BaseException,
    identity: Mapping[str, Any],
    summaries: Sequence[Mapping[str, Any]],
) -> None:
    if not staging.exists():
        return
    _validate_preflight_diagnostic_tree_v3(staging)
    (staging / "oracle-preflight.json").unlink(missing_ok=True)
    _bounded_atomic_write_json_v2(
        staging / "diagnostic.json",
        {
        "schema_version": 2,
        "status": "failed",
        "identity": _mutable_json_value(identity),
        "exception": {
            "type": type(error).__name__,
            "message": str(error),
        },
        "candidates": [_mutable_json_value(summary) for summary in summaries],
        "physical_files": sorted(
            path.relative_to(staging).as_posix()
            for path in staging.rglob("*")
            if path.is_file() and path.name != "diagnostic.json"
        ),
        },
        max_bytes=_MAX_PREFLIGHT_DIAGNOSTIC_BYTES,
        label="oracle preflight diagnostic",
    )
    _move_to_preflight_diagnostic_v2(staging, destination=destination)


_PREFLIGHT_LEASE_FIELDS = frozenset({
    "schema_version", "destination", "destination_identity", "owner_id", "pid",
    "process_start_marker", "created_ns",
})


@dataclass(frozen=True)
class _PreflightLease:
    root: Path
    payload: Mapping[str, Any]
    stale_owners: tuple[Mapping[str, Any], ...]


def _preflight_destination_identity_v3(destination: Path) -> str:
    return hashlib.sha256(str(destination).encode("utf-8")).hexdigest()


def _preflight_lease_payload(
    value: Any, *, destination: Path,
) -> dict[str, Any]:
    fields = _strict_fields(value, _PREFLIGHT_LEASE_FIELDS, "oracle preflight lease")
    if _strict_int(fields["schema_version"], "preflight lease schema_version") != 2:
        raise ValueError("oracle preflight lease schema_version is invalid")
    if fields["destination"] != str(destination):
        raise ValueError("oracle preflight lease destination changed")
    if fields["destination_identity"] != _preflight_destination_identity_v3(
        destination,
    ):
        raise ValueError("oracle preflight lease destination identity changed")
    owner_id = _strict_string(fields["owner_id"], "preflight lease owner id")
    if re.fullmatch(r"[0-9a-f]{32}", owner_id) is None:
        raise ValueError("oracle preflight lease owner id must be a UUID hex token")
    marker = _strict_string(
        fields["process_start_marker"], "preflight process start marker",
    )
    if (
        len(marker) > 256
        or re.fullmatch(r"[A-Za-z0-9._:-]+", marker) is None
    ):
        raise ValueError("oracle preflight process start marker is invalid")
    pid = _strict_int(fields["pid"], "preflight lease pid", minimum=1)
    if pid > _INT32_MAX:
        raise ValueError("oracle preflight lease pid exceeds int32")
    return {
        "schema_version": 2,
        "destination": str(destination),
        "destination_identity": _preflight_destination_identity_v3(destination),
        "owner_id": owner_id,
        "pid": pid,
        "process_start_marker": marker,
        "created_ns": _strict_int(
            fields["created_ns"], "preflight lease creation time", minimum=0,
        ),
    }


def _read_preflight_owner_v3(
    path: Path, *, destination: Path, label: str,
) -> dict[str, Any]:
    canonical = _lexical_canonical_path(Path(path), label, strict=True)
    if not canonical.is_file():
        raise ValueError(f"{label} must be a regular file")
    return _preflight_lease_payload(
        _read_bounded_json_v2(
            canonical,
            max_bytes=_MAX_PREFLIGHT_OWNER_BYTES,
            label=label,
        ),
        destination=destination,
    )


def _preflight_process_start_marker_v3(pid: int) -> str | None:
    pid = _strict_int(pid, "preflight owner pid", minimum=1)
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = (
                ("low", wintypes.DWORD),
                ("high", wintypes.DWORD),
            )

        process_query_limited_information = 0x1000
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.OpenProcess.argtypes = (
            wintypes.DWORD, wintypes.BOOL, wintypes.DWORD,
        )
        kernel32.OpenProcess.restype = wintypes.HANDLE
        kernel32.GetProcessTimes.argtypes = (
            wintypes.HANDLE,
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
            ctypes.POINTER(FileTime),
        )
        kernel32.GetProcessTimes.restype = wintypes.BOOL
        kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
        kernel32.CloseHandle.restype = wintypes.BOOL
        handle = kernel32.OpenProcess(
            process_query_limited_information, False, pid,
        )
        if not handle:
            error = ctypes.get_last_error()
            if error == 87:
                return None
            raise RuntimeError(
                f"cannot verify preflight owner process identity: WinError {error}"
            )
        try:
            created = FileTime()
            exited = FileTime()
            kernel = FileTime()
            user = FileTime()
            if not kernel32.GetProcessTimes(
                handle,
                ctypes.byref(created),
                ctypes.byref(exited),
                ctypes.byref(kernel),
                ctypes.byref(user),
            ):
                raise RuntimeError(
                    "cannot read preflight owner process start time"
                )
            ticks = (created.high << 32) | created.low
            return f"windows-filetime:{ticks}"
        finally:
            kernel32.CloseHandle(handle)

    proc_stat = Path(f"/proc/{pid}/stat")
    try:
        stat = proc_stat.read_text(encoding="ascii")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RuntimeError(
            "cannot verify preflight owner process identity"
        ) from exc
    closing = stat.rfind(")")
    fields_after_name = stat[closing + 2:].split()
    if closing < 0 or len(fields_after_name) <= 19:
        raise RuntimeError("preflight owner process identity is malformed")
    return f"proc-start:{fields_after_name[19]}"


def _acquire_preflight_lease_v2(destination: Path) -> _PreflightLease:
    root = destination.with_name(destination.name + ".lock")
    stale_owners: list[Mapping[str, Any]] = []
    while True:
        try:
            root.mkdir()
        except FileExistsError:
            _lexical_canonical_path(root, "oracle preflight lease", strict=True)
            payload = _read_preflight_owner_v3(
                root / "lease.json",
                destination=destination,
                label="oracle preflight lease metadata",
            )
            current_marker = _preflight_process_start_marker_v3(payload["pid"])
            if current_marker == payload["process_start_marker"]:
                raise RuntimeError("oracle preflight lease has a live owner")
            stale_owners.append(MappingProxyType(dict(payload)))
            atomic_write_json(root / "diagnostic.json", {
                "schema_version": 2,
                "status": "stale-lease",
                "lease": payload,
            })
            _move_to_preflight_diagnostic_v2(root, destination=destination)
            continue
        process_start_marker = _preflight_process_start_marker_v3(os.getpid())
        if process_start_marker is None:
            raise RuntimeError("current preflight process identity disappeared")
        payload = {
            "schema_version": 2,
            "destination": str(destination),
            "destination_identity": _preflight_destination_identity_v3(destination),
            "owner_id": uuid.uuid4().hex,
            "pid": os.getpid(),
            "process_start_marker": process_start_marker,
            "created_ns": time.time_ns(),
        }
        try:
            _bounded_atomic_write_json_v2(
                root / "lease.json",
                payload,
                max_bytes=_MAX_PREFLIGHT_OWNER_BYTES,
                label="oracle preflight lease metadata",
            )
        except BaseException:
            try:
                root.rmdir()
            except OSError:
                pass
            raise
        return _PreflightLease(
            root=root,
            payload=MappingProxyType(payload),
            stale_owners=tuple(stale_owners),
        )


def _release_preflight_lease_v2(lease: _PreflightLease) -> None:
    if not lease.root.exists():
        raise RuntimeError("oracle preflight lease disappeared while owned")
    _lexical_canonical_path(
        lease.root, "oracle preflight lease", strict=True,
    )
    actual = _read_preflight_owner_v3(
        lease.root / "lease.json",
        destination=Path(lease.payload["destination"]),
        label="oracle preflight lease metadata",
    )
    if not _same_exact_json(actual, dict(lease.payload)):
        raise RuntimeError("oracle preflight complete lease owner record changed")
    (lease.root / "lease.json").unlink()
    lease.root.rmdir()


def _require_stale_staging_owner_v2(
    staging: Path, *, lease: _PreflightLease,
) -> None:
    _lexical_canonical_path(
        staging, "unsealed oracle staging", strict=True,
    )
    owner_path = staging / ".owner.json"
    if not owner_path.is_file():
        raise RuntimeError("unsealed oracle staging has no proven stale owner")
    owner = _read_preflight_owner_v3(
        owner_path,
        destination=Path(lease.payload["destination"]),
        label="unsealed oracle staging owner metadata",
    )
    if not any(
        _same_exact_json(owner, dict(stale_owner))
        for stale_owner in lease.stale_owners
    ):
        raise RuntimeError(
            "unsealed oracle staging complete owner record is not proven stale"
        )


def _validate_preflight_output_root_v2(
    definition: PanelDefinition, destination: Path,
) -> None:
    _reject_reparse_chain(destination.parent, "oracle preflight output")
    if destination.exists():
        _reject_reparse_chain(destination, "oracle preflight output")
    repository = definition.repository_root
    if destination.is_relative_to(repository):
        raise ValueError("oracle preflight output must be outside the repository")


def _validate_preflight_path_set_v2(
    definition: PanelDefinition, destination: Path,
) -> tuple[Path, Path, Path, Path]:
    canonical = _lexical_canonical_path(
        Path(destination), "oracle preflight output", strict=False,
    )
    _validate_preflight_output_root_v2(definition, canonical)
    canonical.parent.mkdir(parents=True, exist_ok=True)
    _lexical_canonical_path(
        canonical.parent, "oracle preflight output parent", strict=True,
    )
    canonical = _lexical_canonical_path(
        canonical, "oracle preflight output", strict=False,
    )
    staging = canonical.with_name(canonical.name + ".staging")
    diagnostics = canonical.with_name(canonical.name + ".diagnostics")
    lease = canonical.with_name(canonical.name + ".lock")
    for path, label in (
        (staging, "oracle preflight staging"),
        (diagnostics, "oracle preflight diagnostics"),
        (lease, "oracle preflight lease"),
    ):
        _lexical_canonical_path(path, label, strict=False)
    _validate_preflight_output_root_v2(definition, canonical)
    return canonical, staging, diagnostics, lease


def run_oracle_preflight(
    definition: PanelDefinition,
    *,
    output_root: Path,
    execution_session: object | None = None,
) -> OracleSpec:
    '''Fail closed until Task 9 supplies the production engine-session seal.'''

    del output_root, execution_session
    if not isinstance(definition, PanelDefinition):
        raise TypeError('definition must be a PanelDefinition')
    raise RuntimeError(
        'oracle preflight requires a sealed engine execution session; '
        'the production factory is required from Task 9'
    )


def _run_oracle_preflight_for_test(
    definition: PanelDefinition,
    *,
    output_root: Path,
    evaluator: Callable[
        [OracleSpec, ScheduledDuel], OraclePreflightGameResult
    ],
    benchmark: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample],
        OracleBenchmarkDecision,
    ],
    codec: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample, OracleBenchmarkDecision],
        OracleCodecEvidence,
    ],
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _repository_execution_identity,
    clock: Callable[[], float] = time.perf_counter,
    on_selected: Callable[[OracleSpec], None] | None = None,
) -> OracleSpec:
    """Build and reopen an explicitly untrusted private callback transcript."""

    validate_panel_definition(definition)
    for boundary, label in (
        (evaluator, "evaluator"), (benchmark, "benchmark"), (codec, "codec"),
        (repository_identity_provider, "repository identity provider"),
        (clock, "clock"),
    ):
        if not callable(boundary):
            raise TypeError(f"oracle preflight {label} must be callable")
    if on_selected is not None and not callable(on_selected):
        raise TypeError("oracle preflight success callback must be callable")
    destination, staging, _diagnostics, _lease = _validate_preflight_path_set_v2(
        definition, Path(output_root),
    )
    lease = _acquire_preflight_lease_v2(destination)
    try:
        return _run_oracle_preflight_for_test_owned_v2(
            definition=definition,
            destination=destination,
            staging=staging,
            lease=lease,
            repository_identity_provider=repository_identity_provider,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
            on_selected=on_selected,
        )
    finally:
        _release_preflight_lease_v2(lease)


def _run_oracle_preflight_for_test_owned_v2(
    *,
    definition: PanelDefinition,
    destination: Path,
    staging: Path,
    lease: _PreflightLease,
    repository_identity_provider: Callable[[Path], Mapping[str, Any]],
    evaluator: Callable[[OracleSpec, ScheduledDuel], OraclePreflightGameResult],
    benchmark: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample],
        OracleBenchmarkDecision,
    ],
    codec: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample, OracleBenchmarkDecision],
        OracleCodecEvidence,
    ],
    clock: Callable[[], float],
    on_selected: Callable[[OracleSpec], None] | None,
) -> OracleSpec:
    if destination.exists() and staging.exists():
        raise ValueError("oracle preflight destination and staging coexist ambiguously")
    repository = _validated_repository_identity(
        definition, repository_identity_provider,
    )
    dataset = _audit_base_dataset(definition)
    identity = _oracle_preflight_identity(definition, repository, dataset)
    if destination.exists():
        try:
            return _open_oracle_preflight_v2(
                destination, expected_identity=identity, definition=definition,
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "existing oracle preflight is not exactly reusable"
            ) from exc
    if staging.exists():
        try:
            selected = _open_oracle_preflight_v2(
                staging, expected_identity=identity, definition=definition,
            )
        except (OSError, TypeError, ValueError) as exc:
            _require_stale_staging_owner_v2(staging, lease=lease)
            _seal_preflight_diagnostic_v2(
                staging, destination=destination, error=exc, identity=identity,
                summaries=(),
            )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.replace(staging, destination)
            return _open_oracle_preflight_v2(
                destination, expected_identity=identity, definition=definition,
            )
    return _execute_oracle_preflight_v2(
        definition=definition,
        destination=destination,
        staging=staging,
        identity=identity,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
        lease=lease,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
        on_selected=on_selected,
    )


def _execute_oracle_preflight_v2(
    *,
    definition: PanelDefinition,
    destination: Path,
    staging: Path,
    identity: Mapping[str, Any],
    repository: Mapping[str, Any],
    repository_identity_provider: Callable[[Path], Mapping[str, Any]],
    lease: _PreflightLease,
    evaluator: Callable[[OracleSpec, ScheduledDuel], OraclePreflightGameResult],
    benchmark: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample],
        OracleBenchmarkDecision,
    ],
    codec: Callable[
        [OracleSpec, ScheduledDuel, OracleBenchmarkSample, OracleBenchmarkDecision],
        OracleCodecEvidence,
    ],
    clock: Callable[[], float],
    on_selected: Callable[[OracleSpec], None] | None,
) -> OracleSpec:
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    _bounded_atomic_write_json_v2(
        staging / ".owner.json",
        dict(lease.payload),
        max_bytes=_MAX_PREFLIGHT_OWNER_BYTES,
        label="oracle preflight staging owner metadata",
    )
    summaries: list[Mapping[str, Any]] = []
    game_evidence: list[Mapping[str, Any]] = []
    try:
        for candidate_index, oracle in enumerate(definition.oracle_candidates):
            candidate_games: list[Mapping[str, Any]] = []
            candidate_records: list[Mapping[str, Any]] = []
            seen_state_hashes: set[str] = set()
            for game_index, game in enumerate(definition.preflight_schedule):
                result = evaluator(oracle, game)
                _validate_preflight_game_result(result, game)
                benchmark_records: list[Mapping[str, Any]] = []
                for sample_index, sample in enumerate(result.samples):
                    if (
                        len(sample.observation) != definition.observation_size
                        or len(sample.legal_mask) != definition.action_size
                    ):
                        raise ValueError("oracle benchmark sample shape changed")
                    if sample.state_hash in seen_state_hashes:
                        raise ValueError("oracle benchmark state identity is duplicated")
                    seen_state_hashes.add(sample.state_hash)
                    before_hash = sample.content_sha256
                    first_started = float(clock())
                    first = benchmark(oracle, game, sample)
                    first_codec = codec(oracle, game, sample, first)
                    first_seconds = float(clock()) - first_started
                    between_hash = sample.content_sha256
                    second_started = float(clock())
                    second = benchmark(oracle, game, sample)
                    second_codec = codec(oracle, game, sample, second)
                    second_seconds = float(clock()) - second_started
                    after_hash = sample.content_sha256
                    if before_hash != between_hash or before_hash != after_hash:
                        raise ValueError("oracle benchmark sample mutated across queries")
                    for elapsed in (first_seconds, second_seconds):
                        if not math.isfinite(elapsed) or elapsed < 0.0:
                            raise ValueError("oracle benchmark clock moved backwards")
                    _preflight_decision_valid_v2(
                        first, sample=sample, oracle=oracle, game=game,
                        definition=definition,
                    )
                    _preflight_decision_valid_v2(
                        second, sample=sample, oracle=oracle, game=game,
                        definition=definition,
                    )
                    first_valid = _preflight_codec_valid_v2(
                        first_codec, decision=first, sample=sample,
                        definition=definition,
                    )
                    second_valid = _preflight_codec_valid_v2(
                        second_codec, decision=second, sample=sample,
                        definition=definition,
                    )
                    benchmark_records.append(_preflight_benchmark_record_v2(
                        sample_index=sample_index,
                        sample=sample,
                        first=first,
                        first_codec=first_codec,
                        first_seconds=first_seconds,
                        second=second,
                        second_codec=second_codec,
                        second_seconds=second_seconds,
                    ))
                    candidate_records.append({
                        "first_decision": first,
                        "second_decision": second,
                        "first_codec_valid": first_valid,
                        "second_codec_valid": second_valid,
                        "first_seconds": first_seconds,
                        "second_seconds": second_seconds,
                    })
                _write_preflight_game_v2(
                    staging=staging,
                    candidate_index=candidate_index,
                    game_index=game_index,
                    oracle=oracle,
                    game=game,
                    result=result,
                    benchmark_records=benchmark_records,
                    game_evidence=game_evidence,
                    candidate_games=candidate_games,
                )
            summaries.append(_preflight_candidate_summary_v2(
                oracle=oracle,
                games=candidate_games,
                records=candidate_records,
                definition=definition,
            ))
        return _publish_preflight_v2(
            definition=definition,
            destination=destination,
            staging=staging,
            identity=identity,
            repository=repository,
            repository_identity_provider=repository_identity_provider,
            summaries=summaries,
            game_evidence=game_evidence,
            on_selected=on_selected,
        )
    except BaseException as exc:
        _seal_preflight_diagnostic_v2(
            staging, destination=destination, error=exc, identity=identity,
            summaries=summaries,
        )
        raise


def _write_preflight_game_v2(
    *,
    staging: Path,
    candidate_index: int,
    game_index: int,
    oracle: OracleSpec,
    game: ScheduledDuel,
    result: OraclePreflightGameResult,
    benchmark_records: Sequence[Mapping[str, Any]],
    game_evidence: list[Mapping[str, Any]],
    candidate_games: list[Mapping[str, Any]],
) -> None:
    candidate_root = (
        staging / "games" / f"candidate-{oracle.expansion_budget:08d}"
    )
    trace_path = candidate_root / f"game-{game_index:08d}.trace.json"
    replay_path = candidate_root / f"game-{game_index:08d}.replay.json"
    benchmark_path = candidate_root / f"game-{game_index:08d}.benchmark.json"
    _bounded_atomic_write_json_v2(
        trace_path,
        _preflight_trace_envelope_v2(
            candidate_index=candidate_index, game_index=game_index,
            game=game, result=result,
        ),
        max_bytes=_MAX_PREFLIGHT_TRACE_BYTES,
        label="oracle preflight trace",
    )
    _bounded_atomic_write_json_v2(
        replay_path,
        _preflight_replay_envelope_v2(
            candidate_index=candidate_index, game_index=game_index,
            game=game, result=result,
        ),
        max_bytes=_MAX_PREFLIGHT_REPLAY_BYTES,
        label="oracle preflight replay",
    )
    _bounded_atomic_write_json_v2(
        benchmark_path,
        {
            "schema_version": 1,
            "execution_trust": _private_test_execution_trust_v3(),
            "candidate_index": candidate_index,
            "game_index": game_index,
            "schedule": game.to_dict(),
            "records": [_mutable_json_value(item) for item in benchmark_records],
        },
        max_bytes=_MAX_PREFLIGHT_BENCHMARK_BYTES,
        label="oracle preflight benchmark",
    )
    record = {
        "candidate_index": candidate_index,
        "game_index": game_index,
        **game.to_dict(),
        "outcome": result.outcome,
        "cycling": result.cycling,
        "action_waste": result.action_waste,
        "wasted_end_turns": result.wasted_end_turns,
        "transition_count": len(result.trace.transitions),
        "sample_count": len(benchmark_records),
        "trace": _preflight_file_descriptor(staging, trace_path),
        "replay": _preflight_file_descriptor(staging, replay_path),
        "benchmark": _preflight_file_descriptor(staging, benchmark_path),
    }
    game_evidence.append(record)
    candidate_games.append(record)


def _publish_preflight_v2(
    *,
    definition: PanelDefinition,
    destination: Path,
    staging: Path,
    identity: Mapping[str, Any],
    repository: Mapping[str, Any],
    repository_identity_provider: Callable[[Path], Mapping[str, Any]],
    summaries: Sequence[Mapping[str, Any]],
    game_evidence: Sequence[Mapping[str, Any]],
    on_selected: Callable[[OracleSpec], None] | None,
) -> OracleSpec:
    selected_summary = _select_preflight_candidate(summaries)
    if selected_summary is None:
        raise RuntimeError(
            "oracle preflight has no candidate that passes every gate"
        )
    final_repository = _validated_repository_identity(
        definition, repository_identity_provider,
    )
    if not _same_exact_json(final_repository, repository):
        raise ValueError("repository identity changed during oracle preflight")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "completed",
        "identity": _mutable_json_value(identity),
        "candidates": [_mutable_json_value(item) for item in summaries],
        "selected_oracle": _mutable_json_value(selected_summary["oracle"]),
        "games": [_mutable_json_value(item) for item in game_evidence],
    }
    manifest["content_identity"] = _content_identity(manifest)
    _bounded_atomic_write_json_v2(
        staging / "oracle-preflight.json",
        manifest,
        max_bytes=_MAX_PREFLIGHT_MANIFEST_BYTES,
        label="oracle preflight manifest",
    )
    (staging / ".owner.json").unlink()
    _open_oracle_preflight_v2(
        staging, expected_identity=identity, definition=definition,
    )
    publication_dataset = _audit_base_dataset(definition)
    expected_dataset = identity["original_dataset"]
    if not _same_exact_json(
        {
            "content_sha256": publication_dataset["content_sha256"],
            "file_count": publication_dataset["file_count"],
            "byte_size": publication_dataset["byte_size"],
            "audit": publication_dataset["audit"],
        },
        {
            "content_sha256": expected_dataset["content_sha256"],
            "file_count": expected_dataset["file_count"],
            "byte_size": expected_dataset["byte_size"],
            "audit": expected_dataset["audit"],
        },
    ):
        raise ValueError("original dataset identity changed before oracle publication")
    publication_repository = _validated_repository_identity(
        definition, repository_identity_provider,
    )
    if not _same_exact_json(publication_repository, repository):
        raise ValueError("repository identity changed before oracle publication")
    os.replace(staging, destination)
    selected = _open_oracle_preflight_v2(
        destination, expected_identity=identity, definition=definition,
    )
    if on_selected is not None:
        on_selected(selected)
    return selected


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


def _loaded_base_dataset_identity(base: ImitationDataset) -> OriginalDatasetIdentity:
    manifest = base.root / "manifest.json"
    if not manifest.is_file():
        raise ValueError("base imitation manifest is missing")
    files = tuple(
        DatasetFileIdentity(
            path.relative_to(base.root).as_posix(), _sha256_file(path),
        )
        for path in sorted(base.root.rglob("*"))
        if path.is_file() and path != manifest
    )
    return OriginalDatasetIdentity(
        manifest_sha256=_sha256_file(manifest),
        files=files,
    )


def _action_kind_for_teacher_action(
    action: int, contract: EnvironmentContract,
) -> int:
    if action == 0:
        return ACTION_KINDS["end_turn"]
    for name, (offset, count) in _action_regions(contract).items():
        if offset <= action < offset + count:
            return ACTION_KINDS[name]
    raise ValueError("DAgger teacher action has no action-kind region")


def _overlay_imitation_batch(
    overlay: DaggerOverlay, contract: EnvironmentContract,
) -> ImitationBatch:
    observations: list[np.ndarray] = []
    legal_masks: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    game_ids: list[np.ndarray] = []
    decision_indices: list[np.ndarray] = []
    profiles: list[np.ndarray] = []
    seats: list[np.ndarray] = []
    action_kinds: list[np.ndarray] = []
    for descriptor, game in zip(
        overlay.manifest.games, overlay.games, strict=True,
    ):
        game_manifest = _read_json(
            _contained_file(overlay.root, descriptor.path, "overlay game manifest")
        )
        shard = _strict_fields(
            game_manifest["shard"], _SHARD_DESCRIPTOR_FIELDS, "shard descriptor",
        )
        shard_path = _contained_file(
            overlay.root, shard["path"], "overlay shard",
        )
        with np.load(shard_path, allow_pickle=False) as stored:
            row_observations = stored["observations"].copy()
            packed_masks = stored["packed_masks"].copy()
            teacher_actions = stored["actions"].copy()
            row_seats = stored["seats"].copy()
            row_decisions = stored["decision_indices"].copy()
        count = len(teacher_actions)
        observations.append(row_observations)
        legal_masks.append(np.unpackbits(
            packed_masks,
            axis=1,
            count=contract.action_size,
            bitorder="little",
        ).astype(bool, copy=False))
        actions.append(teacher_actions)
        game_ids.append(np.full(count, game.game_id, dtype=np.int64))
        decision_indices.append(row_decisions)
        profiles.append(np.full(count, game.profile, dtype=object))
        seats.append(row_seats)
        action_kinds.append(np.asarray([
            _action_kind_for_teacher_action(int(action), contract)
            for action in teacher_actions
        ], dtype=np.uint8))

    count = sum(len(values) for values in actions)
    if count != overlay.row_count:
        raise ValueError("DAgger overlay row count changed during materialization")
    return ImitationBatch(
        observations=np.concatenate(observations, axis=0),
        legal_masks=np.concatenate(legal_masks, axis=0),
        actions=np.concatenate(actions, axis=0),
        game_ids=np.concatenate(game_ids, axis=0),
        decision_indices=np.concatenate(decision_indices, axis=0),
        sources=np.full(count, Source.DAGGER_TARGETED, dtype=object),
        profiles=np.concatenate(profiles, axis=0),
        seats=np.concatenate(seats, axis=0),
        action_kinds=np.concatenate(action_kinds, axis=0),
        partitions=np.full(count, overlay.partition, dtype=object),
    )


def _combine_supervision_components(
    partition: str,
    components: Sequence[tuple[str, ImitationBatch]],
) -> MaterializedImitationPartition:
    fields: dict[str, list[np.ndarray]] = {
        name: [] for name in ImitationBatch.__dataclass_fields__
    }
    offsets: dict[tuple[str, int, int], int] = {}
    next_row = 0
    next_game = 0
    for component_hash, batch in components:
        local_games = tuple(dict.fromkeys(int(value) for value in batch.game_ids))
        remapped = {
            game_id: next_game + index
            for index, game_id in enumerate(local_games)
        }
        next_game += len(local_games)
        fields["game_ids"].append(np.asarray(
            [remapped[int(value)] for value in batch.game_ids], dtype=np.int64,
        ))
        for name in ImitationBatch.__dataclass_fields__:
            if name != "game_ids":
                fields[name].append(getattr(batch, name))
        for game_id, decision_index in zip(
            batch.game_ids, batch.decision_indices, strict=True,
        ):
            identity = (
                component_hash, int(game_id), int(decision_index),
            )
            if identity in offsets:
                raise ValueError("actor-supervision row identity is duplicated")
            offsets[identity] = next_row
            next_row += 1
    if next_row < 1:
        raise ValueError(f"actor-supervision {partition} partition is empty")
    combined = ImitationBatch(**{
        name: np.concatenate(values, axis=0) for name, values in fields.items()
    })
    return MaterializedImitationPartition(
        partition=partition,
        batch=combined,
        offsets=MappingProxyType(offsets),
    )


def _validated_cumulative_overlays(
    overlays: Sequence[DaggerOverlay],
    *,
    partition: str,
    base: ImitationDataset,
    original_dataset: OriginalDatasetIdentity,
    scenario_hash: str,
) -> tuple[DaggerOverlay, ...]:
    if not isinstance(overlays, Sequence) or not overlays:
        raise ValueError(f"cumulative {partition} overlays are required")
    reopened: list[DaggerOverlay] = []
    expected_regions = tuple(
        (name, offset, count)
        for name, (offset, count) in _action_regions(base.contract).items()
    )
    for overlay in overlays:
        if not isinstance(overlay, DaggerOverlay):
            raise TypeError("cumulative overlays must be physically reopened")
        physical = open_dagger_overlay(overlay.root)
        definition = physical.definition
        if physical.content_identity != overlay.content_identity:
            raise ValueError("DAgger overlay identity changed before materialization")
        if (
            definition.partition != partition
            or definition.original_dataset != original_dataset
            or definition.contract_hash != base.contract.contract_hash
            or definition.encoding_hash != base.contract.encoding_hash
            or definition.observation_size != base.contract.observation_size
            or definition.action_size != base.contract.action_size
            or definition.action_regions != expected_regions
            or definition.scenario_hash != scenario_hash
        ):
            raise ValueError("DAgger overlay is incompatible with the base corpus")
        reopened.append(physical)
    if tuple(item.iteration for item in reopened) != tuple(
        range(1, len(reopened) + 1)
    ):
        raise ValueError(f"cumulative {partition} overlay iterations are not canonical")
    if len({item.content_identity for item in reopened}) != len(reopened):
        raise ValueError(f"cumulative {partition} overlay identities are duplicated")
    return tuple(reopened)


def build_dagger_corpus(
    base: ImitationDataset,
    train_overlays: Sequence[DaggerOverlay],
    validation_overlays: Sequence[DaggerOverlay],
) -> ActorSupervisionCorpus:
    """Materialize base plus cumulative train overlays and held-out overlays only."""

    if not isinstance(base, ImitationDataset):
        raise TypeError("base must be a loaded ImitationDataset")
    original_dataset = _loaded_base_dataset_identity(base)
    scenario_hashes = {str(game["scenario_hash"]) for game in base.games}
    if len(scenario_hashes) != 1:
        raise ValueError("base imitation dataset scenario identity is ambiguous")
    scenario_hash = next(iter(scenario_hashes))
    training_overlays = _validated_cumulative_overlays(
        train_overlays,
        partition="train",
        base=base,
        original_dataset=original_dataset,
        scenario_hash=scenario_hash,
    )
    held_out_overlays = _validated_cumulative_overlays(
        validation_overlays,
        partition="validation",
        base=base,
        original_dataset=original_dataset,
        scenario_hash=scenario_hash,
    )
    if tuple(item.iteration for item in training_overlays) != tuple(
        item.iteration for item in held_out_overlays
    ):
        raise ValueError("training and validation overlay iterations differ")
    all_overlay_hashes = [
        *(item.content_identity for item in training_overlays),
        *(item.content_identity for item in held_out_overlays),
    ]
    if len(set(all_overlay_hashes)) != len(all_overlay_hashes):
        raise ValueError("training and validation overlay identities overlap")

    base_training = materialize_imitation_partition(base, "train")
    training = _combine_supervision_components(
        "train",
        [
            (original_dataset.manifest_sha256, base_training.batch),
            *(
                (overlay.content_identity, _overlay_imitation_batch(
                    overlay, base.contract,
                ))
                for overlay in training_overlays
            ),
        ],
    )
    validation = _combine_supervision_components(
        "validation",
        [
            (overlay.content_identity, _overlay_imitation_batch(
                overlay, base.contract,
            ))
            for overlay in held_out_overlays
        ],
    )
    fractions = MappingProxyType(OrderedDict((
        (Source.GREEDY_STANDARD, 0.49),
        (Source.SEARCH_CONVERSION, 0.21),
        (Source.DAGGER_TARGETED, 0.30),
    )))
    identity = MappingProxyType({
        "schema_version": 1,
        "kind": "selective-dagger-v1",
        "base_manifest_sha256": original_dataset.manifest_sha256,
        "base_files": tuple(
            (item.path, item.sha256) for item in original_dataset.files
        ),
        "train_overlays": tuple(
            item.content_identity for item in training_overlays
        ),
        "validation_overlays": tuple(
            item.content_identity for item in held_out_overlays
        ),
        "contract_hash": base.contract.contract_hash,
        "encoding_hash": base.contract.encoding_hash,
        "scenario_hash": scenario_hash,
    })
    return ActorSupervisionCorpus(
        training=training,
        validation=validation,
        source_fractions=fractions,
        identity=identity,
    )


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


def dagger_actor_source(
    iteration: int, *, preceding_run: Path | None = None,
) -> ActorTransferSource:
    """Resolve the immutable incoming actor for one DAgger iteration."""

    if type(iteration) is not int or iteration not in {1, 2, 3}:
        raise ValueError("DAgger actor source iteration is invalid")
    if iteration == 1:
        if preceding_run is not None:
            raise ValueError("iteration one has no preceding DAgger actor")
        source = ActorTransferSource(
            source_kind="snapshot",
            controller=_ITERATION_ONE_CONTROLLER,
            checkpoint_sha256=_ITERATION_ONE_CHECKPOINT_SHA256,
        )
        checkpoint = Path(source.controller["path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        if _sha256_file(checkpoint) != source.checkpoint_sha256:
            raise ValueError("iteration-one checkpoint SHA-256 does not match")
        return source

    if preceding_run is None:
        raise ValueError("later DAgger iterations require the preceding actor run")
    run = Path(preceding_run).resolve()
    run_manifest_path = run / "run.json"
    bc_path = run / "bc.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(run_manifest_path)
    if not bc_path.is_file():
        raise FileNotFoundError(bc_path)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    latest_checkpoint = (
        run_manifest.get("latest_checkpoint")
        if isinstance(run_manifest, Mapping)
        else None
    )
    if not isinstance(latest_checkpoint, str):
        raise ValueError("preceding DAgger actor publication is invalid")
    checkpoint = (run / latest_checkpoint).resolve()
    if (
        not checkpoint.is_relative_to(run)
        or checkpoint.parent != (run / "checkpoints").resolve()
    ):
        raise ValueError(
            "preceding DAgger actor checkpoint is not contained by its resolved run"
        )
    if (
        latest_checkpoint != "checkpoints/step_000000000.zip"
        or run_manifest.get("latest_checkpoint_step") != 0
        or checkpoint.name != "step_000000000.zip"
    ):
        raise ValueError(
            "preceding DAgger actor checkpoint is not the canonical step-zero publication"
        )
    distillation_iteration = run_manifest.get("distillation_iteration")
    actor_initialization = run_manifest.get("actor_initialization")
    published_actor_sha256 = run_manifest.get("target_actor_sha256_final")
    publication_verification = run_manifest.get("publication_verification")
    if (
        not isinstance(run_manifest, Mapping)
        or run_manifest.get("schema_version") != 1
        or run_manifest.get("state") != "completed"
        or run_manifest.get("production") is not True
        or run_manifest.get("training_kind")
        != "selective-dagger-distillation-v1"
        or type(distillation_iteration) is not int
        or distillation_iteration not in {1, 2}
        or not isinstance(actor_initialization, Mapping)
        or not isinstance(publication_verification, Mapping)
        or not _HASH_PATTERN.fullmatch(str(published_actor_sha256))
        or publication_verification.get("actor_sha256")
        != published_actor_sha256
        or publication_verification.get("checkpoint_sha256")
        != run_manifest.get("checkpoint_sha256")
        or not isinstance(run_manifest.get("config"), Mapping)
        or run_manifest["config"].get("algorithm") != "maskable_ppo"
        or run_manifest["config"].get("policy") != "HexCNN"
        or not isinstance(bc, Mapping)
        or bc.get("schema_version") != 1
        or bc.get("training_kind") != "selective-dagger-distillation-v1"
        or bc.get("algorithm") != "maskable_ppo"
        or bc.get("policy") != "HexCNN"
        or bc.get("production") is not True
        or bc.get("distillation_iteration") != distillation_iteration
        or bc.get("checkpoint_sha256")
        != run_manifest.get("checkpoint_sha256")
        or bc.get("target_actor_sha256_final") != published_actor_sha256
        or bc.get("actor_initialization") != actor_initialization
        or bc.get("publication_verification") != publication_verification
    ):
        raise ValueError("preceding DAgger actor provenance does not agree")
    if distillation_iteration != iteration - 1:
        raise ValueError("preceding DAgger actor iteration does not match")
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    digest = _sha256_file(checkpoint)
    if (
        run_manifest.get("checkpoint_sha256") != digest
        or bc.get("checkpoint_sha256") != digest
    ):
        raise ValueError("preceding DAgger actor checkpoint SHA-256 does not match")
    return ActorTransferSource(
        source_kind="dagger_actor",
        controller={
            "kind": "snapshot",
            "path": str(checkpoint),
            "source_run": str(run),
            "algorithm": "maskable_ppo",
            "step": 0,
            "inference_mode": "deterministic",
        },
        checkpoint_sha256=digest,
        published_actor_sha256=published_actor_sha256,
    )


def train_dagger_actor(
    *,
    corpus: ActorSupervisionCorpus,
    scenario: Any,
    env: Any,
    contract: EnvironmentContract,
    spaces_info: Mapping[str, Any],
    run_dir: Path,
    source: ActorTransferSource,
    progress: Callable[[Mapping[str, Any]], None] | None = None,
    nonproduction_cpu_config: BehavioralCloningConfig | None = None,
) -> Any:
    """Train one actor with locked production settings or explicit CPU smoke settings."""

    if not isinstance(corpus, ActorSupervisionCorpus):
        raise TypeError("corpus must be an ActorSupervisionCorpus")
    if not isinstance(source, ActorTransferSource):
        raise TypeError("source must be an ActorTransferSource")
    train_overlays = corpus.identity.get("train_overlays")
    validation_overlays = corpus.identity.get("validation_overlays")
    if (
        corpus.identity.get("kind") != "selective-dagger-v1"
        or not isinstance(train_overlays, (tuple, list))
        or not isinstance(validation_overlays, (tuple, list))
        or len(train_overlays) != len(validation_overlays)
        or len(train_overlays) not in {1, 2, 3}
    ):
        raise ValueError("selective-DAgger distillation corpus identity is invalid")
    iteration = len(train_overlays)
    if nonproduction_cpu_config is None:
        config = DAGGER_DISTILLATION_CONFIG
        preceding_run = (
            None
            if iteration == 1
            else Path(source.controller["source_run"])
        )
        if source.to_dict() != dagger_actor_source(
            iteration, preceding_run=preceding_run,
        ).to_dict():
            raise ValueError("production DAgger actor source is not canonical")
        mode = "production"
    else:
        if (
            not isinstance(nonproduction_cpu_config, BehavioralCloningConfig)
            or nonproduction_cpu_config.device != "cpu"
        ):
            raise ValueError(
                "nonproduction DAgger distillation requires an explicit CPU config"
            )
        config = nonproduction_cpu_config
        mode = "nonproduction_cpu"
    return train_actor_supervision(
        corpus=corpus,
        scenario=scenario,
        env=env,
        contract=contract,
        spaces_info=spaces_info,
        run_dir=run_dir,
        config=config,
        progress=progress,
        warm_start=source,
        distillation_mode=mode,
    )


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


def _mutable_json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mutable_json_value(item) for item in value]
    return value


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(
        _mutable_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
