"""Fail-closed discovery of physical checkpoint-audit candidates."""

from __future__ import annotations

import base64
import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .contracts import utc_now
from .evaluation import DuelClient, evaluate_controllers, wilson_interval
from .io import atomic_write_json


_CHECKPOINT_NAME = re.compile(r"step_(?P<step>\d{9})\.zip\Z")
_EXPECTED_SEED = 227
_COMPATIBILITY_FIELDS = (
    ("environment", "environment"),
    ("version", "version"),
    ("encoding_hash", "encoding"),
    ("observation_size", "observation"),
    ("action_size", "action"),
)
TRACE_FIELDS = {
    "rounds": "round_count",
    "decisions": "command_count",
    "peak_health_adjusted_advantage": "peak_normalized_advantage",
    "final_health_adjusted_advantage": "final_normalized_advantage",
}


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
    environment = _require_string(contract.get("environment"), label=f"{label} contract environment")
    version = _require_string(contract.get("version"), label=f"{label} contract version")
    if environment != "tactical-v2":
        raise ValueError(f"{label} contract environment must be tactical-v2")
    if version != "tactical-v2":
        raise ValueError(f"{label} contract version must be tactical-v2")
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


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _runtime_contract(server_cmd: Sequence[str]) -> Mapping[str, Any]:
    """Read one evaluation-time tactical-v2 contract without retaining a process."""
    client = DuelClient(server_cmd, environment="tactical-v2")
    try:
        return client.contract.to_dict()
    finally:
        client.close()


def _repository_identity() -> Mapping[str, Any]:
    """Return the current Git identity; tests replace this private physical boundary."""
    cwd = Path(__file__).resolve().parent
    try:
        repository = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=cwd,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        ).resolve()
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("checkpoint audit requires a readable Git repository identity") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("checkpoint audit Git commit must be a full lowercase SHA-1")
    return {"repository": str(repository), "commit": commit, "dirty": bool(status.strip())}


def _source_material(
    definition: AuditDefinition,
) -> tuple[bytes, str, tuple[Mapping[str, Any], ...]]:
    scenario_bytes: bytes | None = None
    source_rows: dict[str, Mapping[str, Any]] = {}
    for candidate in definition.candidates:
        if candidate.source_run is None:
            continue
        source = _canonical_run(Path(candidate.source_run), label=candidate.candidate_id)
        manifest, manifest_bytes = _read_json_bytes(
            source / "run.json", label=f"{candidate.candidate_id} run.json"
        )
        if _sha256(manifest_bytes) != candidate.source_run_manifest_sha256:
            raise ValueError(f"{candidate.candidate_id} source run manifest bytes changed")
        try:
            current_scenario = (source / "scenario.json").read_bytes()
        except OSError as error:
            raise ValueError(f"{candidate.candidate_id} scenario snapshot is missing") from error
        current_scenario_hash = _sha256(current_scenario)
        if current_scenario_hash != candidate.source_scenario_sha256:
            raise ValueError(f"{candidate.candidate_id} scenario bytes changed")
        if scenario_bytes is None:
            scenario_bytes = current_scenario
        elif current_scenario != scenario_bytes:
            raise ValueError("learned audit sources must have byte-identical scenarios")
        contract = _require_mapping(
            manifest.get("contract"), label=f"{candidate.candidate_id} source contract"
        )
        expected_fields = {
            "contract_hash": candidate.source_contract_hash,
            "encoding_hash": candidate.source_encoding_hash,
            "observation_size": candidate.observation_size,
            "action_size": candidate.action_size,
        }
        if any(contract.get(field) != expected for field, expected in expected_fields.items()):
            raise ValueError(f"{candidate.candidate_id} source contract changed")
        source_rows[str(source)] = {
            "source_run": str(source),
            "run_manifest_sha256": _sha256(manifest_bytes),
            "scenario_sha256": current_scenario_hash,
            "contract": dict(contract),
        }
    if scenario_bytes is None:
        raise ValueError("checkpoint audit requires at least one learned physical source")
    return scenario_bytes, _sha256(scenario_bytes), tuple(
        source_rows[path] for path in sorted(source_rows)
    )


def _validate_runtime_contract(
    runtime: Mapping[str, Any], source_contracts: Sequence[Mapping[str, Any]]
) -> None:
    if runtime.get("environment") != "tactical-v2" or runtime.get("version") != "tactical-v2":
        raise ValueError("evaluation runtime contract must be tactical-v2")
    if not source_contracts:
        raise ValueError("audit manifest is missing source contracts")
    source = _require_mapping(source_contracts[0].get("contract"), label="source contract")
    for field in ("encoding_hash", "observation_size", "action_size", "board"):
        if runtime.get(field) != source.get(field):
            raise ValueError(f"evaluation runtime contract {field} is incompatible")


def _definition_identity(definition: AuditDefinition) -> tuple[Mapping[str, Any], str]:
    payload = definition.to_dict()
    return payload, _sha256(_canonical_json_bytes(payload))


def _initial_manifest(
    definition: AuditDefinition,
    *,
    scenario_bytes: bytes,
    scenario_sha256: str,
    source_contracts: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    definition_payload, definition_sha256 = _definition_identity(definition)
    return {
        "schema_version": 1,
        "generated_at": utc_now(),
        "state": "in_progress",
        "definition": definition_payload,
        "definition_sha256": definition_sha256,
        "repository_identity": dict(_repository_identity()),
        "scenario": {
            "encoding": "base64",
            "bytes_base64": base64.b64encode(scenario_bytes).decode("ascii"),
            "sha256": scenario_sha256,
        },
        "source_contracts": [dict(row) for row in source_contracts],
        "runtime_contract": dict(runtime_contract),
    }


def _load_audit_manifest(root: Path) -> Mapping[str, Any]:
    manifest, _raw = _read_json_bytes(root / "manifest.json", label="audit manifest")
    if manifest.get("schema_version") != 1:
        raise ValueError("audit manifest schema_version must be 1")
    _require_string(manifest.get("generated_at"), label="audit manifest generated_at")
    state = manifest.get("state")
    if state not in {"in_progress", "completed"}:
        raise ValueError("audit manifest state must be in_progress or completed")
    repository = _require_mapping(
        manifest.get("repository_identity"), label="audit manifest repository identity"
    )
    if set(repository) != {"repository", "commit", "dirty"}:
        raise ValueError("audit manifest repository identity fields are invalid")
    _require_string(repository.get("repository"), label="audit manifest repository path")
    commit = _require_string(
        repository.get("commit"), label="audit manifest repository commit"
    )
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("audit manifest repository commit must be a full lowercase SHA-1")
    if not isinstance(repository.get("dirty"), bool):
        raise ValueError("audit manifest repository dirty flag must be boolean")
    if state == "completed":
        _require_string(
            manifest.get("completed_at"), label="audit manifest completed_at"
        )
        aggregate_sha256 = _require_string(
            manifest.get("aggregate_sha256"), label="audit manifest aggregate SHA-256"
        )
        if not re.fullmatch(r"[0-9a-f]{64}", aggregate_sha256):
            raise ValueError("audit manifest aggregate SHA-256 is invalid")
    definition = _require_mapping(manifest.get("definition"), label="audit definition")
    definition_sha256 = _require_string(
        manifest.get("definition_sha256"), label="audit definition SHA-256"
    )
    if _sha256(_canonical_json_bytes(definition)) != definition_sha256:
        raise ValueError("audit manifest definition identity is invalid")
    scenario = _require_mapping(manifest.get("scenario"), label="audit scenario")
    if scenario.get("encoding") != "base64":
        raise ValueError("audit scenario encoding must be base64")
    encoded = _require_string(scenario.get("bytes_base64"), label="audit scenario bytes")
    try:
        raw_scenario = base64.b64decode(encoded, validate=True)
    except ValueError as error:
        raise ValueError("audit scenario bytes are invalid base64") from error
    scenario_sha256 = _require_string(
        scenario.get("sha256"), label="audit scenario SHA-256"
    )
    if _sha256(raw_scenario) != scenario_sha256:
        raise ValueError("audit scenario bytes do not match their SHA-256")
    source_contracts = manifest.get("source_contracts")
    if not isinstance(source_contracts, list) or not source_contracts:
        raise ValueError("audit manifest source_contracts must be a non-empty list")
    runtime = _require_mapping(manifest.get("runtime_contract"), label="runtime contract")
    _validate_runtime_contract(runtime, source_contracts)
    return manifest


def _require_existing_manifest(
    root: Path,
    definition: AuditDefinition,
    *,
    scenario_bytes: bytes,
    scenario_sha256: str,
    source_contracts: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
) -> Mapping[str, Any]:
    manifest = _load_audit_manifest(root)
    definition_payload, definition_sha256 = _definition_identity(definition)
    if (
        manifest.get("definition") != definition_payload
        or manifest.get("definition_sha256") != definition_sha256
    ):
        raise ValueError("existing audit manifest has a different frozen definition")
    scenario = _require_mapping(manifest.get("scenario"), label="audit scenario")
    if (
        scenario.get("bytes_base64")
        != base64.b64encode(scenario_bytes).decode("ascii")
        or scenario.get("sha256") != scenario_sha256
    ):
        raise ValueError("existing audit manifest has different physical scenario bytes")
    if manifest.get("source_contracts") != [dict(row) for row in source_contracts]:
        raise ValueError("existing audit manifest has different physical source contracts")
    if manifest.get("runtime_contract") != dict(runtime_contract):
        raise ValueError("existing audit manifest has a different evaluation runtime contract")
    return manifest


def audit_map_path(root: Path, candidate_id: str, map_seed: int) -> Path:
    return Path(root) / "candidates" / candidate_id / f"map-{map_seed}" / "evaluation.json"


def _source_contract_for(
    manifest: Mapping[str, Any], candidate: AuditCandidate
) -> Mapping[str, Any] | None:
    if candidate.source_run is None:
        return None
    source = _canonical_run(Path(candidate.source_run), label=candidate.candidate_id)
    physical_manifest, physical_manifest_bytes = _read_json_bytes(
        source / "run.json", label=f"{candidate.candidate_id} physical run.json"
    )
    if _sha256(physical_manifest_bytes) != candidate.source_run_manifest_sha256:
        raise ValueError(f"{candidate.candidate_id} physical source manifest changed")
    try:
        physical_scenario = (source / "scenario.json").read_bytes()
    except OSError as error:
        raise ValueError(f"{candidate.candidate_id} physical scenario is missing") from error
    if _sha256(physical_scenario) != candidate.source_scenario_sha256:
        raise ValueError(f"{candidate.candidate_id} physical scenario changed")
    physical_contract = _require_mapping(
        physical_manifest.get("contract"),
        label=f"{candidate.candidate_id} physical source contract",
    )
    for row in manifest["source_contracts"]:
        mapping = _require_mapping(row, label="source contract row")
        if mapping.get("source_run") == candidate.source_run:
            if (
                mapping.get("run_manifest_sha256")
                != candidate.source_run_manifest_sha256
                or mapping.get("scenario_sha256")
                != candidate.source_scenario_sha256
            ):
                raise ValueError(
                    f"audit source identity changed for {candidate.candidate_id}"
                )
            stored_contract = _require_mapping(
                mapping.get("contract"), label="candidate source contract"
            )
            if stored_contract != physical_contract:
                raise ValueError(
                    f"audit source contract changed for {candidate.candidate_id}"
                )
            return physical_contract
    raise ValueError(f"audit manifest has no source contract for {candidate.candidate_id}")


def _scripted_controller_identity(name: str) -> Mapping[str, Any]:
    return {
        "kind": "scripted",
        "inference_mode": "deterministic",
        "path": None,
        "algorithm": None,
        "step": None,
        "contract_hash": None,
        "contract_version": None,
        "environment": None,
        "encoding_hash": None,
        "contract": None,
        "observation_size": None,
        "action_size": None,
        "legacy": False,
        "promotable": False,
        "name": name,
    }


def _expected_candidate_identity(
    candidate: AuditCandidate, source_contract: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    if candidate.family == "control":
        return _scripted_controller_identity(candidate.controller)
    if source_contract is None:
        raise ValueError(f"{candidate.candidate_id} is missing its source contract")
    try:
        spec = json.loads(candidate.controller)
    except json.JSONDecodeError as error:
        raise ValueError(f"{candidate.candidate_id} controller spec is invalid") from error
    if not isinstance(spec, Mapping) or spec.get("kind") not in {"run", "snapshot"}:
        raise ValueError(f"{candidate.candidate_id} controller spec is unsupported")
    identity: dict[str, Any] = {
        "kind": spec["kind"],
        "inference_mode": spec.get("inference_mode", "deterministic"),
        "path": candidate.checkpoint_path,
        "algorithm": "maskable_ppo",
        "step": candidate.actual_step,
        "contract_hash": candidate.source_contract_hash,
        "contract_version": source_contract.get("version"),
        "environment": source_contract.get("environment"),
        "encoding_hash": candidate.source_encoding_hash,
        "contract": dict(source_contract),
        "observation_size": candidate.observation_size,
        "action_size": candidate.action_size,
        "legacy": False,
        "promotable": True,
    }
    if spec["kind"] == "run":
        identity["source_run"] = candidate.source_run
    return identity


def _artifact_path(root: Path, map_root: Path, raw_path: object, *, label: str) -> Path:
    value = _require_string(raw_path, label=label)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a relative path below the audit root")
    try:
        resolved = (root / path).resolve(strict=True)
        resolved.relative_to(root)
        resolved.relative_to(map_root / "evidence")
    except (OSError, ValueError) as error:
        raise ValueError(f"{label} must resolve inside the map evidence directory") from error
    if not resolved.is_file():
        raise ValueError(f"{label} must name a retained file")
    return resolved


def _checkpoint_digest(candidate: AuditCandidate) -> str | None:
    if candidate.checkpoint_path is None:
        if candidate.checkpoint_sha256 is not None:
            raise ValueError(f"{candidate.candidate_id} has a digest without a checkpoint")
        return None
    _path, digest = _checkpoint_bytes(
        Path(candidate.checkpoint_path), label=candidate.candidate_id
    )
    if digest != candidate.checkpoint_sha256:
        raise ValueError(f"{candidate.candidate_id} physical checkpoint bytes changed")
    return digest


def _require_exact_counts(value: object, expected: Mapping[str, int], *, label: str) -> None:
    mapping = _require_mapping(value, label=label)
    if set(mapping) != set(expected):
        raise ValueError(f"{label} fields are invalid")
    for field, count in expected.items():
        if isinstance(mapping.get(field), bool) or mapping.get(field) != count:
            raise ValueError(f"{label} does not reconcile")


def validate_physical_map(
    root: Path,
    candidate: AuditCandidate,
    schedule: AuditSchedule,
    map_seed: int,
) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Reopen and validate one reciprocal map and every retained physical byte."""
    try:
        audit_root = Path(root).resolve(strict=True)
    except OSError as error:
        raise ValueError(f"audit root does not exist: {root}") from error
    manifest = _load_audit_manifest(audit_root)
    frozen_definition = _require_mapping(
        manifest.get("definition"), label="frozen audit definition"
    )
    if frozen_definition.get("schedule") != schedule.to_dict():
        raise ValueError("frozen audit schedule does not match the requested schedule")
    frozen_candidates = frozen_definition.get("candidates")
    if (
        not isinstance(frozen_candidates, list)
        or candidate.to_dict() not in frozen_candidates
    ):
        raise ValueError("candidate is not present in the frozen audit definition")
    evaluation_path = audit_map_path(
        audit_root, candidate.candidate_id, map_seed
    )
    evaluation, _raw = _read_json_bytes(evaluation_path, label="map evaluation")
    map_root = evaluation_path.parent.resolve(strict=True)
    if evaluation.get("schema_version") != 1:
        raise ValueError("map evaluation schema_version must be 1")
    _require_string(evaluation.get("generated_at"), label="map evaluation timestamp")

    source_contract = _source_contract_for(manifest, candidate)
    expected_candidate = _expected_candidate_identity(candidate, source_contract)
    if evaluation.get("candidate") != expected_candidate:
        raise ValueError("map evaluation candidate controller identity changed")
    if evaluation.get("opponent") != _scripted_controller_identity(schedule.opponent):
        raise ValueError("map evaluation opponent controller identity changed")
    if schedule.opponent != "random":
        raise ValueError("checkpoint audit opponent must be random")
    if schedule.profile != "standard-3v3" or not schedule.both_seats:
        raise ValueError("checkpoint audit requires reciprocal standard-3v3 maps")
    if evaluation.get("schedule") != {
        "start_profile": schedule.profile,
        "reference_seat_policy": "candidate-seat",
    }:
        raise ValueError("map evaluation profile schedule changed")
    if (
        evaluation.get("seed_start") != map_seed
        or evaluation.get("seeds") != [map_seed]
        or evaluation.get("reciprocal") is not True
        or evaluation.get("games") != 2
    ):
        raise ValueError("map evaluation reciprocal seed schedule changed")

    audit_identity = _require_mapping(
        evaluation.get("audit_identity"), label="map audit identity"
    )
    definition_sha256 = _require_string(
        manifest.get("definition_sha256"), label="audit definition SHA-256"
    )
    scenario = _require_mapping(manifest.get("scenario"), label="audit scenario")
    runtime = _require_mapping(manifest.get("runtime_contract"), label="runtime contract")
    frozen_scenario_hashes = {
        row.get("source_scenario_sha256")
        for row in frozen_candidates
        if isinstance(row, Mapping) and row.get("source_scenario_sha256") is not None
    }
    if len(frozen_scenario_hashes) != 1 or scenario.get("sha256") not in frozen_scenario_hashes:
        raise ValueError(
            "audit scenario identity does not match frozen physical candidate provenance"
        )
    if audit_identity.get("definition_sha256") != definition_sha256:
        raise ValueError("map audit definition identity changed")
    if audit_identity.get("scenario_sha256") != scenario.get("sha256"):
        raise ValueError("map audit scenario identity changed")
    if audit_identity.get("runtime_contract") != runtime:
        raise ValueError("map evaluation runtime contract changed")
    if audit_identity.get("checkpoint_sha256") != _checkpoint_digest(candidate):
        raise ValueError("map audit checkpoint identity changed")

    matches_raw = evaluation.get("matches")
    if not isinstance(matches_raw, list) or len(matches_raw) != 2:
        raise ValueError("map evaluation must contain exactly two reciprocal matches")
    matches = tuple(
        _require_mapping(match, label=f"map match {index}")
        for index, match in enumerate(matches_raw)
    )
    if [match.get("candidate_seat") for match in matches] != [0, 1]:
        raise ValueError("map evaluation reciprocal seats changed")

    totals = {"wins": 0, "losses": 0, "draws": 0}
    seats = {
        "candidate_as_p0": {"wins": 0, "losses": 0, "draws": 0},
        "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 0},
    }
    draw_categories: dict[str, int] = {}
    artifacts = audit_identity.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 2:
        raise ValueError("map audit identity must hash both retained artifact pairs")
    artifact_pairs: set[tuple[Path, Path]] = set()
    for index, match in enumerate(matches):
        seat = index
        if match.get("seed") != map_seed:
            raise ValueError("map match seed changed")
        if (
            match.get("start_profile") != schedule.profile
            or match.get("reference_seat") != seat
        ):
            raise ValueError("map match profile reference changed")
        winner = match.get("winner")
        if isinstance(winner, bool) or winner not in {-1, 0, 1}:
            raise ValueError("map match winner is invalid")
        outcome = "win" if winner == seat else "loss" if winner in {0, 1} else "draw"
        if match.get("outcome") != outcome:
            raise ValueError("map match outcome does not reconcile with its winner")
        if not isinstance(match.get("terminated"), bool) or not isinstance(
            match.get("truncated"), bool
        ):
            raise ValueError("map match termination fields are invalid")
        summary = match.get("summary")
        if not isinstance(summary, Mapping) or not summary:
            raise ValueError("map match trace summary is missing")
        classification = match.get("classification")
        if outcome == "draw":
            if not isinstance(classification, Mapping):
                raise ValueError("draw match classification is missing")
            primary = _require_string(
                classification.get("primary"), label="draw classification primary"
            )
            draw_categories[primary] = draw_categories.get(primary, 0) + 1
        elif classification is not None and not isinstance(classification, Mapping):
            raise ValueError("non-draw match classification is invalid")

        trace_path = _artifact_path(
            audit_root, map_root, match.get("trace_path"), label="trace path"
        )
        replay_path = _artifact_path(
            audit_root, map_root, match.get("replay_path"), label="replay path"
        )
        artifact_pair = (trace_path, replay_path)
        if artifact_pair in artifact_pairs:
            raise ValueError("reciprocal matches must retain distinct trace/replay artifacts")
        artifact_pairs.add(artifact_pair)

        artifact = _require_mapping(artifacts[index], label=f"match {index} artifact identity")
        if artifact.get("trace_sha256") != _sha256(trace_path.read_bytes()):
            raise ValueError("retained trace bytes changed")
        if artifact.get("replay_sha256") != _sha256(replay_path.read_bytes()):
            raise ValueError("retained replay bytes changed")
        if (
            match.get("trace_sha256") != artifact.get("trace_sha256")
            or match.get("replay_sha256") != artifact.get("replay_sha256")
        ):
            raise ValueError("map match artifact SHA-256 fields changed")

        counter = f"{outcome}s" if outcome != "loss" else "losses"
        totals[counter] += 1
        seat_key = "candidate_as_p0" if seat == 0 else "candidate_as_p1"
        seats[seat_key][counter] += 1

    _require_exact_counts(
        {key: evaluation.get(key) for key in totals}, totals, label="map outcome totals"
    )
    expected_rates = {
        "win": totals["wins"] / 2,
        "loss": totals["losses"] / 2,
        "draw": totals["draws"] / 2,
    }
    if evaluation.get("rates") != expected_rates:
        raise ValueError("map outcome rates do not reconcile")
    expected_intervals = {
        "win": wilson_interval(totals["wins"], 2, 0.95),
        "loss": wilson_interval(totals["losses"], 2, 0.95),
        "draw": wilson_interval(totals["draws"], 2, 0.95),
    }
    if evaluation.get("confidence_intervals") != expected_intervals:
        raise ValueError("map Wilson confidence intervals do not reconcile")
    seat_results = _require_mapping(evaluation.get("seat_results"), label="seat results")
    if set(seat_results) != set(seats):
        raise ValueError("map seat result fields are invalid")
    for seat_key, counts in seats.items():
        _require_exact_counts(seat_results.get(seat_key), counts, label=seat_key)

    expected_evidence = {
        "retention": "all",
        "retained": 2,
        "draw_traces": totals["draws"],
        "control_traces": 2 - totals["draws"],
        "draw_categories": dict(sorted(draw_categories.items())),
    }
    if evaluation.get("evidence") != expected_evidence:
        raise ValueError("map evidence summary does not reconcile")
    return evaluation, (matches[0], matches[1])


def _normalize_published_artifact(
    audit_root: Path, map_root: Path, raw_path: object, *, label: str
) -> tuple[str, str]:
    value = _require_string(raw_path, label=label)
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"published {label} contains a parent traversal")
    candidate = path if path.is_absolute() else audit_root / path
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(audit_root)
        resolved.relative_to(map_root / "evidence")
    except (OSError, ValueError) as error:
        raise ValueError(f"published {label} escaped the map evidence directory") from error
    if not resolved.is_file():
        raise ValueError(f"published {label} must name a retained file")
    return resolved.relative_to(audit_root).as_posix(), _sha256(resolved.read_bytes())


def _enrich_evaluation(
    root: Path,
    evaluation_path: Path,
    candidate: AuditCandidate,
    manifest: Mapping[str, Any],
) -> None:
    evaluation, _raw = _read_json_bytes(evaluation_path, label="published map evaluation")
    matches = evaluation.get("matches")
    if not isinstance(matches, list) or len(matches) != 2:
        raise ValueError("published map evaluation must contain two matches")
    normalized_matches: list[dict[str, Any]] = []
    artifacts: list[Mapping[str, str]] = []
    map_root = evaluation_path.parent.resolve(strict=True)
    for index, raw_match in enumerate(matches):
        match = dict(_require_mapping(raw_match, label=f"published match {index}"))
        trace_path, trace_sha256 = _normalize_published_artifact(
            root, map_root, match.get("trace_path"), label="trace path"
        )
        replay_path, replay_sha256 = _normalize_published_artifact(
            root, map_root, match.get("replay_path"), label="replay path"
        )
        match.update(
            trace_path=trace_path,
            replay_path=replay_path,
            trace_sha256=trace_sha256,
            replay_sha256=replay_sha256,
        )
        normalized_matches.append(match)
        artifacts.append(
            {"trace_sha256": trace_sha256, "replay_sha256": replay_sha256}
        )
    enriched = dict(evaluation)
    enriched["matches"] = normalized_matches
    enriched["audit_identity"] = {
        "definition_sha256": manifest["definition_sha256"],
        "scenario_sha256": manifest["scenario"]["sha256"],
        "runtime_contract": manifest["runtime_contract"],
        "checkpoint_sha256": _checkpoint_digest(candidate),
        "artifacts": artifacts,
    }
    atomic_write_json(evaluation_path, enriched)


def _duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _progress_line(
    candidate_index: int,
    candidate_total: int,
    candidate: AuditCandidate,
    maps_done: int,
    maps_total: int,
    reused: int,
    elapsed: float,
    total_done: int,
    total_maps: int,
) -> str:
    rate = total_done / elapsed if elapsed > 0 and total_done > 0 else 0.0
    eta = (total_maps - total_done) / rate if rate > 0 else 0.0
    return (
        f"[{candidate_index}/{candidate_total} {candidate.candidate_id}] "
        f"maps {maps_done}/{maps_total}, games {maps_done * 2}/{maps_total * 2}, "
        f"reused {reused}, elapsed {_duration(elapsed)}, eta {_duration(eta)}"
    )


def evaluate_audit(
    definition: AuditDefinition,
    *,
    output_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_controllers,
    progress: Callable[[str], None] = print,
) -> Mapping[str, Any]:
    """Evaluate every immutable candidate/map and reuse only validated physical evidence."""
    if workers < 1:
        raise ValueError("checkpoint audit workers must be positive")
    if definition.schema_version != 1:
        raise ValueError("checkpoint audit definition schema_version must be 1")
    schedule = definition.schedule
    if (
        schedule.maps != 100
        or schedule.seed_start != 16_000_000
        or not schedule.both_seats
        or schedule.profile != "standard-3v3"
        or schedule.opponent != "random"
    ):
        raise ValueError("checkpoint audit schedule must be reciprocal standard-3v3 versus random")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    scenario_bytes, scenario_sha256, source_contracts = _source_material(definition)
    runtime_contract = dict(_runtime_contract(server_cmd))
    _validate_runtime_contract(runtime_contract, source_contracts)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = _require_existing_manifest(
            root,
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
        )
    else:
        manifest = _initial_manifest(
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
        )
        atomic_write_json(manifest_path, manifest)

    started = time.monotonic()
    completed = 0
    reused_total = 0
    total_maps = len(definition.candidates) * schedule.maps
    for candidate_index, candidate in enumerate(definition.candidates, start=1):
        candidate_reused = 0
        progress(
            _progress_line(
                candidate_index,
                len(definition.candidates),
                candidate,
                0,
                schedule.maps,
                0,
                time.monotonic() - started,
                completed,
                total_maps,
            )
        )
        for map_offset in range(schedule.maps):
            map_seed = schedule.seed_start + map_offset
            evaluation_path = audit_map_path(root, candidate.candidate_id, map_seed)
            if evaluation_path.exists():
                validate_physical_map(root, candidate, schedule, map_seed)
                candidate_reused += 1
                reused_total += 1
                completed += 1
                progress(
                    _progress_line(
                        candidate_index,
                        len(definition.candidates),
                        candidate,
                        map_offset + 1,
                        schedule.maps,
                        candidate_reused,
                        time.monotonic() - started,
                        completed,
                        total_maps,
                    )
                )
                continue
            if evaluation_path.parent.exists() and any(evaluation_path.parent.iterdir()):
                raise ValueError(
                    f"map directory exists without valid evaluation evidence: {evaluation_path.parent}"
                )
            evidence_dir = evaluation_path.parent / "evidence"
            result = evaluator(
                candidate.controller,
                schedule.opponent,
                games=1,
                seed_start=map_seed,
                both_seats=True,
                workers=workers,
                server_cmd=server_cmd,
                output_path=evaluation_path,
                environment="tactical-v2",
                evidence_dir=evidence_dir,
                start_profile=schedule.profile,
                capture_trace=True,
                evidence_retention="all",
            )
            if not isinstance(result, Mapping):
                raise ValueError("checkpoint audit evaluator must return an evaluation mapping")
            _enrich_evaluation(root, evaluation_path, candidate, manifest)
            validate_physical_map(root, candidate, schedule, map_seed)
            completed += 1
            if (map_offset + 1) % 10 == 0:
                progress(
                    _progress_line(
                        candidate_index,
                        len(definition.candidates),
                        candidate,
                        map_offset + 1,
                        schedule.maps,
                        candidate_reused,
                        time.monotonic() - started,
                        completed,
                        total_maps,
                    )
                )
        progress(
            _progress_line(
                candidate_index,
                len(definition.candidates),
                candidate,
                schedule.maps,
                schedule.maps,
                candidate_reused,
                time.monotonic() - started,
                completed,
                total_maps,
            )
        )
    return {
        "state": "in_progress",
        "manifest": str(manifest_path),
        "maps": completed,
        "games": completed * 2,
        "reused": reused_total,
    }


def _trace_metric(summary: Mapping[str, Any], aggregate_field: str) -> float:
    raw_field = next(
        (raw for raw, mapped in TRACE_FIELDS.items() if mapped == aggregate_field),
        aggregate_field,
    )
    value = summary.get(raw_field, summary.get(aggregate_field))
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"match summary {raw_field} must be numeric")
    return float(value)


def _distribution(values: Sequence[float]) -> Mapping[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "p90": None}
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2
    )
    p90_index = (9 * len(ordered) + 9) // 10 - 1
    return {
        "mean": sum(ordered) / len(ordered),
        "median": median,
        "p90": ordered[p90_index],
    }


def summarize_candidate(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    """Summarize one candidate's frozen 200-game reciprocal panel."""
    if len(rows) != 200:
        raise ValueError("candidate summary requires exactly 200 match rows")
    counts = {"games": 200, "wins": 0, "losses": 0, "draws": 0}
    seats: dict[str, dict[str, Any]] = {
        "candidate_as_p0": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
        "candidate_as_p1": {"games": 0, "wins": 0, "losses": 0, "draws": 0},
    }
    primary_categories: dict[str, int] = {}
    cycling_draws = 0
    action_waste_draws = 0
    win_rounds: list[float] = []
    win_commands: list[float] = []
    all_final_advantage: list[float] = []
    all_peak_advantage: list[float] = []
    draw_final_advantage: list[float] = []
    draw_peak_advantage: list[float] = []
    candidate_end_turns = 0
    candidate_wasted_end_turns = 0

    for row in rows:
        outcome = row.get("outcome")
        if outcome not in {"win", "loss", "draw"}:
            raise ValueError("candidate match outcome must be win, loss, or draw")
        seat = row.get("candidate_seat")
        if isinstance(seat, bool) or seat not in {0, 1}:
            raise ValueError("candidate match seat must be 0 or 1")
        summary = _require_mapping(row.get("summary"), label="candidate match summary")
        round_count = _trace_metric(summary, "round_count")
        command_count = _trace_metric(summary, "command_count")
        peak_advantage = _trace_metric(summary, "peak_normalized_advantage")
        final_advantage = _trace_metric(summary, "final_normalized_advantage")
        end_turns = summary.get("end_turns_by_seat")
        wasted_end_turns = summary.get("wasted_end_turns_by_seat")
        for values, label in (
            (end_turns, "end_turns_by_seat"),
            (wasted_end_turns, "wasted_end_turns_by_seat"),
        ):
            if (
                not isinstance(values, (list, tuple))
                or len(values) != 2
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in values
                )
            ):
                raise ValueError(f"match summary {label} must contain two nonnegative integers")

        counter = "losses" if outcome == "loss" else f"{outcome}s"
        counts[counter] += 1
        seat_summary = seats["candidate_as_p0" if seat == 0 else "candidate_as_p1"]
        seat_summary["games"] += 1
        seat_summary[counter] += 1
        all_peak_advantage.append(peak_advantage)
        all_final_advantage.append(final_advantage)
        candidate_end_turns += end_turns[seat]
        candidate_wasted_end_turns += wasted_end_turns[seat]

        if outcome == "win":
            win_rounds.append(round_count)
            win_commands.append(command_count)
        if outcome == "draw":
            classification = _require_mapping(
                row.get("classification"), label="draw classification"
            )
            primary = _require_string(
                classification.get("primary"), label="draw classification primary"
            )
            flags = classification.get("flags")
            if not isinstance(flags, list) or any(
                not isinstance(flag, str) or not flag for flag in flags
            ):
                raise ValueError("draw classification flags must be strings")
            primary_categories[primary] = primary_categories.get(primary, 0) + 1
            cycling_draws += int("cycling" in flags)
            action_waste_draws += int("action_waste" in flags)
            draw_peak_advantage.append(peak_advantage)
            draw_final_advantage.append(final_advantage)

    rates = {
        "win": counts["wins"] / 200,
        "loss": counts["losses"] / 200,
        "draw": counts["draws"] / 200,
    }
    for seat_summary in seats.values():
        games = seat_summary["games"]
        seat_summary["rates"] = {
            "win": seat_summary["wins"] / games if games else 0.0,
            "loss": seat_summary["losses"] / games if games else 0.0,
            "draw": seat_summary["draws"] / games if games else 0.0,
        }
    wasted_ratio = (
        candidate_wasted_end_turns / candidate_end_turns
        if candidate_end_turns
        else 0.0
    )
    return {
        "counts": counts,
        "rates": rates,
        "confidence_intervals": {
            outcome: wilson_interval(counts[counter], 200, 0.95)
            for outcome, counter in (
                ("win", "wins"),
                ("loss", "losses"),
                ("draw", "draws"),
            )
        },
        "seats": seats,
        "win_rate_p0_minus_p1": (
            seats["candidate_as_p0"]["rates"]["win"]
            - seats["candidate_as_p1"]["rates"]["win"]
        ),
        "draw_diagnostics": {
            "cycling": {"count": cycling_draws, "incidence": cycling_draws / 200},
            "action_waste": {
                "count": action_waste_draws,
                "incidence": action_waste_draws / 200,
            },
            "primary_categories": dict(sorted(primary_categories.items())),
        },
        "winning_games": {
            "round_count": _distribution(win_rounds),
            "command_count": _distribution(win_commands),
        },
        "normalized_advantage": {
            "all_games": {
                "final": sum(all_final_advantage) / 200,
                "peak": sum(all_peak_advantage) / 200,
            },
            "draws": {
                "final": (
                    sum(draw_final_advantage) / len(draw_final_advantage)
                    if draw_final_advantage else None
                ),
                "peak": (
                    sum(draw_peak_advantage) / len(draw_peak_advantage)
                    if draw_peak_advantage else None
                ),
            },
        },
        "candidate_end_turns": {
            "total": candidate_end_turns,
            "wasted": candidate_wasted_end_turns,
            "wasted_ratio": wasted_ratio,
        },
        "end_turn_policy_diagnostics": {
            "available": False,
            "reason": (
                "integer-action inference boundary does not expose action probabilities or ranks"
            ),
        },
    }


def _paired_index(
    rows: Sequence[Mapping[str, Any]], *, label: str
) -> Mapping[tuple[int, int], str]:
    if len(rows) != 200:
        raise ValueError(f"{label} paired table has missing schedule keys")
    indexed: dict[tuple[int, int], str] = {}
    for row in rows:
        map_seed = row.get("map_seed")
        candidate_seat = row.get("candidate_seat")
        if (
            isinstance(map_seed, bool)
            or not isinstance(map_seed, int)
            or isinstance(candidate_seat, bool)
            or candidate_seat not in {0, 1}
        ):
            raise ValueError(f"{label} paired table has an invalid schedule key")
        key = (map_seed, candidate_seat)
        if key in indexed:
            raise ValueError(f"{label} paired table has a duplicate schedule key")
        outcome = row.get("outcome")
        if outcome not in {"win", "draw", "loss"}:
            raise ValueError(f"{label} paired table has an invalid outcome")
        indexed[key] = outcome
    return indexed


def _binomial(n: int, k: int) -> int:
    k = min(k, n - k)
    result = 1
    for index in range(1, k + 1):
        result = result * (n - k + index) // index
    return result


def _exact_sign_test(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if discordant == 0:
        return 1.0
    tail = sum(
        _binomial(discordant, index)
        for index in range(min(left_only, right_only) + 1)
    )
    return min(1.0, 2.0 * tail / (2**discordant))


def paired_change(
    earlier: Sequence[Mapping[str, Any]],
    later: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Compare two candidate panels on identical reciprocal schedule keys."""
    left = _paired_index(earlier, label="earlier")
    right = _paired_index(later, label="later")
    if set(left) != set(right):
        raise ValueError("paired tables have missing schedule keys")
    outcomes = ("win", "draw", "loss")
    transitions = {
        earlier_outcome: {later_outcome: 0 for later_outcome in outcomes}
        for earlier_outcome in outcomes
    }
    for key in sorted(left):
        transitions[left[key]][right[key]] += 1
    left_only = sum(transitions["win"][outcome] for outcome in ("draw", "loss"))
    right_only = sum(transitions[outcome]["win"] for outcome in ("draw", "loss"))
    net_change = right_only - left_only
    return {
        "transition_table": transitions,
        "left_only_wins": left_only,
        "right_only_wins": right_only,
        "net_win_change": net_change,
        "absolute_win_rate_change": net_change / 200,
        "exact_sign_test_p_value": _exact_sign_test(left_only, right_only),
    }

CandidateAggregate = Mapping[str, Any]


def _aggregate_summary(candidate: CandidateAggregate) -> Mapping[str, Any]:
    return _require_mapping(candidate.get("summary"), label="candidate aggregate summary")


def _aggregate_win_rate(candidate: CandidateAggregate) -> float:
    rates = _require_mapping(_aggregate_summary(candidate).get("rates"), label="candidate rates")
    value = rates.get("win")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("candidate aggregate win rate must be numeric")
    return float(value)


def choose_next_experiment(
    trajectory: Sequence[CandidateAggregate],
) -> Mapping[str, Any]:
    """Return every approved evidence clause and one precedence-selected next step."""
    if not trajectory:
        raise ValueError("checkpoint trajectory cannot be empty")
    ordered = sorted(trajectory, key=lambda row: row.get("trajectory_order", -1))
    if ordered[0].get("family") != "pure_bc" or any(
        row.get("family") != "bc_ppo" for row in ordered[1:]
    ):
        raise ValueError("decision trajectory must contain only clone then BC-initialized PPO")
    expected_orders = list(range(len(ordered)))
    if [row.get("trajectory_order") for row in ordered] != expected_orders:
        raise ValueError("decision trajectory orders must be unique and contiguous")

    ppo = ordered[1:]
    ppo_rates = [_aggregate_win_rate(row) for row in ppo]
    qualifying_indexes = [index for index, rate in enumerate(ppo_rates) if rate >= 0.65]
    qualifying_ids = [
        _require_string(ppo[index].get("candidate_id"), label="candidate ID")
        for index in qualifying_indexes
    ]
    consistent_improvement = False
    if qualifying_indexes:
        earliest = qualifying_indexes[0] + 1
        rates_to_qualifier = [_aggregate_win_rate(row) for row in ordered[: earliest + 1]]
        changes = [
            later - earlier
            for earlier, later in zip(rates_to_qualifier, rates_to_qualifier[1:])
        ]
        consistent_improvement = all(change >= 0 for change in changes) and any(
            change > 0 for change in changes
        )

    large_late_regression = any(
        earlier - later >= 0.10
        for earlier_index, earlier in enumerate(ppo_rates)
        for later in ppo_rates[earlier_index + 1 :]
    )
    cycling_dominant = False
    if ppo:
        latest_summary = _aggregate_summary(ppo[-1])
        latest_counts = _require_mapping(
            latest_summary.get("counts"), label="latest PPO outcome counts"
        )
        draw_diagnostics = _require_mapping(
            latest_summary.get("draw_diagnostics"), label="latest PPO draw diagnostics"
        )
        cycling = _require_mapping(
            draw_diagnostics.get("cycling"), label="latest PPO cycling draws"
        ).get("count")
        wins = latest_counts.get("wins")
        losses = latest_counts.get("losses")
        draws = latest_counts.get("draws")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (cycling, wins, losses, draws)
        ):
            raise ValueError("latest PPO counts must be nonnegative integers")
        if cycling > draws:
            raise ValueError("latest PPO cycling draws exceed all draws")
        cycling_dominant = cycling > max(wins, losses, draws - cycling)

    all_ppo_below_half = bool(ppo_rates) and all(rate < 0.50 for rate in ppo_rates)
    clauses = {
        "qualifying_ppo": qualifying_ids,
        "consistent_improvement": consistent_improvement,
        "large_late_regression": large_late_regression,
        "cycling_dominant": cycling_dominant,
        "all_ppo_below_half": all_ppo_below_half,
    }
    if large_late_regression:
        recommendation = "test_retained_imitation_constraint"
    elif qualifying_ids and consistent_improvement:
        recommendation = "replicate_seeds_211_223"
    elif all_ppo_below_half or cycling_dominant:
        recommendation = "proceed_to_dagger"
    else:
        recommendation = "inconclusive_review_trajectory"
    return {"clauses": clauses, "recommended_next_step": recommendation}

def aggregate_audit(
    definition: AuditDefinition,
    *,
    output_root: Path,
) -> Mapping[str, Any]:
    """Reopen every physical map, aggregate deterministically, then seal the audit."""
    root = Path(output_root).resolve(strict=True)
    manifest = _load_audit_manifest(root)
    definition_payload, definition_sha256 = _definition_identity(definition)
    if (
        manifest.get("definition") != definition_payload
        or manifest.get("definition_sha256") != definition_sha256
    ):
        raise ValueError("audit manifest does not match the requested frozen definition")

    candidates: dict[str, Mapping[str, Any]] = {}
    physical_rows: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in definition.candidates:
        rows: list[Mapping[str, Any]] = []
        for offset in range(definition.schedule.maps):
            map_seed = definition.schedule.seed_start + offset
            _evaluation, matches = validate_physical_map(
                root, candidate, definition.schedule, map_seed
            )
            for candidate_seat, match in enumerate(matches):
                if match.get("candidate_seat") != candidate_seat:
                    raise ValueError("validated reciprocal matches are not ordered by candidate seat")
                rows.append({**dict(match), "map_seed": map_seed})
        if len(rows) != 200:
            raise ValueError(
                f"{candidate.candidate_id} must contain exactly 200 ordered match rows"
            )
        summary = dict(summarize_candidate(rows))
        aggregate = {**candidate.to_dict(), "summary": summary}
        candidates[candidate.candidate_id] = aggregate
        physical_rows[candidate.candidate_id] = rows

    trajectory = sorted(
        (
            aggregate
            for aggregate in candidates.values()
            if aggregate.get("trajectory_order") is not None
        ),
        key=lambda aggregate: aggregate["trajectory_order"],
    )
    paired_changes: list[Mapping[str, Any]] = []
    for earlier, later in zip(trajectory, trajectory[1:]):
        earlier_id = _require_string(earlier.get("candidate_id"), label="candidate ID")
        later_id = _require_string(later.get("candidate_id"), label="candidate ID")
        paired_changes.append(
            {
                "earlier_candidate_id": earlier_id,
                "later_candidate_id": later_id,
                **paired_change(physical_rows[earlier_id], physical_rows[later_id]),
            }
        )
    anchors = [
        candidate.candidate_id
        for candidate in definition.candidates
        if candidate.family == "control"
    ]
    total_maps = len(definition.candidates) * definition.schedule.maps
    aggregate_payload: Mapping[str, Any] = {
        "schema_version": 1,
        "audit_id": definition.audit_id,
        "exploratory": definition.exploratory,
        "locked_panel_replacement": definition.locked_panel_replacement,
        "schedule": definition.schedule.to_dict(),
        "definition_sha256": definition_sha256,
        "repository_identity": manifest["repository_identity"],
        "scenario": manifest["scenario"],
        "source_contracts": manifest["source_contracts"],
        "runtime_contract": manifest["runtime_contract"],
        "omitted_optional_candidates": [
            dict(item) for item in definition.omitted_optional_candidates
        ],
        "candidates": candidates,
        "trajectory": trajectory,
        "paired_successive_changes": paired_changes,
        "anchors": anchors,
        "decision": choose_next_experiment(trajectory),
        "physical_evidence": {
            "maps": total_maps,
            "games": total_maps * 2,
            "traces": total_maps * 2,
            "replays": total_maps * 2,
        },
    }

    audit_path = root / "audit.json"
    state = manifest.get("state")
    if state == "completed":
        expected_digest = _require_string(
            manifest.get("aggregate_sha256"), label="completed aggregate SHA-256"
        )
        existing, existing_bytes = _read_json_bytes(audit_path, label="completed aggregate")
        if _sha256(existing_bytes) != expected_digest:
            raise ValueError("completed aggregate bytes do not match the root manifest")
        if existing != aggregate_payload:
            raise ValueError("completed aggregate does not match revalidated physical evidence")
        return existing
    if state != "in_progress":
        raise ValueError("audit manifest state must be in_progress or completed")

    atomic_write_json(audit_path, aggregate_payload)
    aggregate_digest = _sha256(audit_path.read_bytes())
    completed_manifest = {
        **dict(manifest),
        "state": "completed",
        "completed_at": utc_now(),
        "aggregate_sha256": aggregate_digest,
    }
    atomic_write_json(root / "manifest.json", completed_manifest)
    return aggregate_payload
