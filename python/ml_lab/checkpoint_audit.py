"""Fail-closed discovery of physical checkpoint-audit candidates."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping


_CHECKPOINT_NAME = re.compile(r"step_(?P<step>\d{9})\.zip\Z")
_EXPECTED_SEED = 227
_COMPATIBILITY_FIELDS = (
    ("environment", "environment"),
    ("version", "version"),
    ("encoding_hash", "encoding"),
    ("observation_size", "observation"),
    ("action_size", "action"),
)


@dataclass(frozen=True)
class AuditSchedule:
    seed_start: int = 16_000_000
    maps: int = 100
    both_seats: bool = True
    profile: str = "standard-3v3"
    opponent: str = "random"

    def to_dict(self) -> dict[str, object]:
        return {
            "seed_start": self.seed_start,
            "maps": self.maps,
            "both_seats": self.both_seats,
            "profile": self.profile,
            "opponent": self.opponent,
        }


@dataclass(frozen=True)
class AuditCandidate:
    candidate_id: str
    family: Literal["pure_bc", "bc_ppo", "scratch_ppo", "control"]
    trajectory_order: int | None
    controller: str
    model_seed: int | None
    actual_step: int | None
    checkpoint_path: str | None
    checkpoint_sha256: str | None
    source_run: str | None
    source_run_manifest_sha256: str | None
    source_scenario_sha256: str | None
    source_contract_hash: str | None
    source_encoding_hash: str | None
    observation_size: int | None
    action_size: int | None

    def to_dict(self) -> dict[str, object | None]:
        return {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "trajectory_order": self.trajectory_order,
            "controller": self.controller,
            "model_seed": self.model_seed,
            "actual_step": self.actual_step,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "source_run": self.source_run,
            "source_run_manifest_sha256": self.source_run_manifest_sha256,
            "source_scenario_sha256": self.source_scenario_sha256,
            "source_contract_hash": self.source_contract_hash,
            "source_encoding_hash": self.source_encoding_hash,
            "observation_size": self.observation_size,
            "action_size": self.action_size,
        }


@dataclass(frozen=True)
class AuditDefinition:
    schema_version: int
    audit_id: str
    exploratory: bool
    locked_panel_replacement: bool
    schedule: AuditSchedule
    candidates: tuple[AuditCandidate, ...]
    omitted_optional_candidates: tuple[Mapping[str, str], ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "exploratory": self.exploratory,
            "locked_panel_replacement": self.locked_panel_replacement,
            "schedule": self.schedule.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "omitted_optional_candidates": [dict(item) for item in self.omitted_optional_candidates],
        }


@dataclass(frozen=True)
class _SourceRun:
    path: Path
    manifest: Mapping[str, Any]
    manifest_sha256: str
    scenario_sha256: str
    contract: Mapping[str, Any]
    config: Mapping[str, Any]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _read_json_bytes(path: Path, *, label: str) -> tuple[Mapping[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be readable JSON: {path}") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value, raw


def _require_mapping(value: object, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_string(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _require_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _canonical_run(path: Path, *, label: str) -> Path:
    try:
        resolved = Path(path).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} does not exist: {path}") from error
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory: {resolved}")
    return resolved


def _load_source(path: Path, *, label: str, require_seed: bool = True) -> _SourceRun:
    root = _canonical_run(path, label=label)
    manifest, manifest_bytes = _read_json_bytes(root / "run.json", label=f"{label} run.json")
    if manifest.get("schema_version") != 1:
        raise ValueError(f"{label} run.json schema_version must be 1")
    config = _require_mapping(manifest.get("config"), label=f"{label} config")
    contract = _require_mapping(manifest.get("contract"), label=f"{label} contract")
    scenario = _require_mapping(manifest.get("scenario"), label=f"{label} scenario")
    if scenario.get("path") != "scenario.json":
        raise ValueError(f"{label} scenario path must be scenario.json")
    scenario_path = root / "scenario.json"
    try:
        scenario_bytes = scenario_path.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} scenario snapshot is missing: {scenario_path}") from error
    if not scenario_bytes:
        raise ValueError(f"{label} scenario snapshot is empty: {scenario_path}")
    _require_string(contract.get("environment"), label=f"{label} contract environment")
    _require_string(contract.get("version"), label=f"{label} contract version")
    _require_string(contract.get("contract_hash"), label=f"{label} contract hash")
    _require_string(contract.get("encoding_hash"), label=f"{label} contract encoding hash")
    if _require_int(contract.get("observation_size"), label=f"{label} observation size") < 1:
        raise ValueError(f"{label} observation size must be positive")
    if _require_int(contract.get("action_size"), label=f"{label} action size") < 1:
        raise ValueError(f"{label} action size must be positive")
    if config.get("environment") != contract.get("environment"):
        raise ValueError(f"{label} environment does not match its contract")
    if config.get("algorithm") != "maskable_ppo":
        raise ValueError(f"{label} algorithm must be maskable_ppo")
    if require_seed:
        seeds = (manifest.get("model_seed"), config.get("model_seed"), config.get("seed"))
        if any(seed != _EXPECTED_SEED or isinstance(seed, bool) for seed in seeds):
            raise ValueError(f"{label} model seed must be {_EXPECTED_SEED}")
    return _SourceRun(
        path=root,
        manifest=manifest,
        manifest_sha256=_sha256(manifest_bytes),
        scenario_sha256=_sha256(scenario_bytes),
        contract=contract,
        config=config,
    )


def _checkpoint_bytes(path: Path, *, label: str) -> tuple[Path, str]:
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise ValueError(f"{label} checkpoint is missing: {path}") from error
    if not resolved.is_file():
        raise ValueError(f"{label} checkpoint must be a regular file: {resolved}")
    try:
        content = resolved.read_bytes()
    except OSError as error:
        raise ValueError(f"{label} checkpoint cannot be read: {resolved}") from error
    if not content:
        raise ValueError(f"{label} checkpoint bytes are empty: {resolved}")
    return resolved, _sha256(content)


def _checkpoint_step(path: Path, *, label: str) -> int:
    match = _CHECKPOINT_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"{label} checkpoint filename is malformed: {path.name}")
    return int(match.group("step"))


def _clone_checkpoint(source: _SourceRun) -> tuple[Path, str]:
    raw_path = _require_string(source.manifest.get("latest_checkpoint"), label="clone latest_checkpoint")
    candidate = (source.path / raw_path).resolve()
    try:
        candidate.relative_to(source.path)
    except ValueError as error:
        raise ValueError("clone latest_checkpoint must stay within the source run") from error
    step = _checkpoint_step(candidate, label="clone")
    if step != 0 or source.manifest.get("latest_checkpoint_step") != 0:
        raise ValueError("clone latest checkpoint step must agree with physical step zero")
    return _checkpoint_bytes(candidate, label="clone")


def _physical_checkpoints(source: _SourceRun, *, label: str) -> list[tuple[int, Path, str]]:
    directory = source.path / "checkpoints"
    if not directory.is_dir():
        raise ValueError(f"{label} checkpoints directory is missing: {directory}")
    checkpoints: list[tuple[int, Path, str]] = []
    for path in directory.glob("step_*.zip"):
        match = _CHECKPOINT_NAME.fullmatch(path.name)
        if match is None:
            continue
        checkpoint, digest = _checkpoint_bytes(path, label=label)
        checkpoints.append((int(match.group("step")), checkpoint, digest))
    checkpoints.sort(key=lambda item: item[0])
    steps = [item[0] for item in checkpoints]
    if not checkpoints:
        raise ValueError(f"{label} has no physical checkpoint files")
    if len(steps) != len(set(steps)) or steps != sorted(steps):
        raise ValueError(f"{label} checkpoint steps must be unique and strictly increasing")
    recorded = source.manifest.get("checkpoint_steps")
    if recorded is not None:
        if not isinstance(recorded, list) or any(
            isinstance(step, bool) or not isinstance(step, int) for step in recorded
        ):
            raise ValueError(f"{label} checkpoint steps must be integer history")
        if recorded != steps:
            raise ValueError(f"{label} checkpoint steps must be unique and strictly increasing physical steps")
    return checkpoints


def _require_compatible(left: _SourceRun, right: _SourceRun, *, label: str) -> None:
    for field, name in _COMPATIBILITY_FIELDS:
        if left.contract.get(field) != right.contract.get(field):
            raise ValueError(f"{label} {name} is incompatible")


def _controller(spec: Mapping[str, object]) -> str:
    return json.dumps(spec, sort_keys=True, separators=(",", ":"))


def _learned_candidate(
    *,
    candidate_id: str,
    family: Literal["pure_bc", "bc_ppo", "scratch_ppo"],
    trajectory_order: int | None,
    source: _SourceRun,
    checkpoint: Path,
    checkpoint_sha256: str,
    step: int,
    controller: str,
) -> AuditCandidate:
    return AuditCandidate(
        candidate_id=candidate_id,
        family=family,
        trajectory_order=trajectory_order,
        controller=controller,
        model_seed=_EXPECTED_SEED,
        actual_step=step,
        checkpoint_path=str(checkpoint),
        checkpoint_sha256=checkpoint_sha256,
        source_run=str(source.path),
        source_run_manifest_sha256=source.manifest_sha256,
        source_scenario_sha256=source.scenario_sha256,
        source_contract_hash=_require_string(source.contract.get("contract_hash"), label="contract hash"),
        source_encoding_hash=_require_string(source.contract.get("encoding_hash"), label="encoding hash"),
        observation_size=_require_int(source.contract.get("observation_size"), label="observation size"),
        action_size=_require_int(source.contract.get("action_size"), label="action size"),
    )


def _controls() -> tuple[AuditCandidate, AuditCandidate]:
    shared = {
        "trajectory_order": None,
        "model_seed": None,
        "actual_step": None,
        "checkpoint_path": None,
        "checkpoint_sha256": None,
        "source_run": None,
        "source_run_manifest_sha256": None,
        "source_scenario_sha256": None,
        "source_contract_hash": None,
        "source_encoding_hash": None,
        "observation_size": None,
        "action_size": None,
    }
    return (
        AuditCandidate(candidate_id="random-anchor", family="control", controller="random", **shared),
        AuditCandidate(
            candidate_id="bounded-search-anchor",
            family="control",
            controller="bounded-search",
            **shared,
        ),
    )


def discover_audit_candidates(
    *,
    clone_run: Path,
    ppo_run: Path,
    scratch_run: Path | None,
) -> tuple[AuditCandidate, ...]:
    """Discover only supplied, physically present checkpoints for the audit."""
    clone = _load_source(clone_run, label="clone")
    clone_bc = _require_mapping(clone.config.get("behavioral_cloning"), label="clone behavioral_cloning")
    if clone_bc.get("model_seed") != _EXPECTED_SEED:
        raise ValueError(f"clone behavioral_cloning model seed must be {_EXPECTED_SEED}")
    clone_checkpoint, clone_digest = _clone_checkpoint(clone)

    ppo = _load_source(ppo_run, label="PPO")
    actor_source = _require_string(ppo.config.get("actor_init_source"), label="PPO actor_init_source")
    actor_path = Path(actor_source)
    if not actor_path.is_absolute():
        actor_path = ppo.path / actor_path
    if actor_path.resolve() != clone.path:
        raise ValueError("PPO actor_init_source must resolve to the supplied clone")
    if ppo.scenario_sha256 != clone.scenario_sha256:
        raise ValueError("PPO scenario snapshot is incompatible with clone")
    _require_compatible(clone, ppo, label="PPO")

    candidates: list[AuditCandidate] = [
        _learned_candidate(
            candidate_id="pure-bc-seed-227",
            family="pure_bc",
            trajectory_order=0,
            source=clone,
            checkpoint=clone_checkpoint,
            checkpoint_sha256=clone_digest,
            step=0,
            controller=_controller({"kind": "run", "mode": "fixed", "path": str(clone.path)}),
        )
    ]
    for index, (step, checkpoint, digest) in enumerate(_physical_checkpoints(ppo, label="PPO"), start=1):
        candidates.append(
            _learned_candidate(
                candidate_id=f"bc-ppo-seed-227-step-{step:09d}",
                family="bc_ppo",
                trajectory_order=index,
                source=ppo,
                checkpoint=checkpoint,
                checkpoint_sha256=digest,
                step=step,
                controller=_controller(
                    {
                        "algorithm": "maskable_ppo",
                        "kind": "snapshot",
                        "path": str(checkpoint),
                        "source_run": str(ppo.path),
                        "step": step,
                    }
                ),
            )
        )

    if scratch_run is not None:
        scratch = _load_source(scratch_run, label="scratch")
        if scratch.config.get("actor_init_source") is not None or (scratch.path / "initialization.json").exists():
            raise ValueError("scratch run must not be BC-initialized")
        if scratch.scenario_sha256 != clone.scenario_sha256:
            raise ValueError("scratch scenario snapshot is incompatible with clone")
        _require_compatible(clone, scratch, label="scratch")
        for step, checkpoint, digest in _physical_checkpoints(scratch, label="scratch"):
            candidates.append(
                _learned_candidate(
                    candidate_id=f"scratch-ppo-seed-227-step-{step:09d}",
                    family="scratch_ppo",
                    trajectory_order=None,
                    source=scratch,
                    checkpoint=checkpoint,
                    checkpoint_sha256=digest,
                    step=step,
                    controller=_controller(
                        {
                            "algorithm": "maskable_ppo",
                            "kind": "snapshot",
                            "path": str(checkpoint),
                            "source_run": str(scratch.path),
                            "step": step,
                        }
                    ),
                )
            )

    candidates.extend(_controls())
    return tuple(candidates)


def build_audit_definition(
    *,
    clone_run: Path,
    ppo_run: Path,
    scratch_run: Path | None,
) -> AuditDefinition:
    """Build the immutable definition for an exploratory checkpoint audit."""
    omitted: tuple[Mapping[str, str], ...] = ()
    if scratch_run is None:
        omitted = ({"family": "scratch_ppo", "reason": "no physical compatible run supplied"},)
    return AuditDefinition(
        schema_version=1,
        audit_id="annihilation-checkpoint-audit-v1",
        exploratory=True,
        locked_panel_replacement=False,
        schedule=AuditSchedule(),
        candidates=discover_audit_candidates(
            clone_run=clone_run,
            ppo_run=ppo_run,
            scratch_run=scratch_run,
        ),
        omitted_optional_candidates=omitted,
    )
