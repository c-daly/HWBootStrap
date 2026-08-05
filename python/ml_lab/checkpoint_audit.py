"""Fail-closed discovery of physical checkpoint-audit candidates."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import re
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

from .contracts import utc_now
from .draw_classification import classify_draw, summarize_episode
from .evaluation import DuelClient, evaluate_controllers, wilson_interval
from .tactical_trace import EpisodeTrace
from .io import atomic_write_json


_CHECKPOINT_NAME = re.compile(r"step_(?P<step>\d{9})\.zip\Z")
_CANDIDATE_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,127}\Z")
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

_PROGRAMMATIC_SMOKE_SCHEDULE = AuditSchedule(seed_start=16_000_000, maps=2)


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
class AuditSourceRoot:
    role: Literal["clone", "ppo", "scratch"]
    source_run: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "source_run": self.source_run}


@dataclass(frozen=True)
class AuditDefinition:
    schema_version: int
    audit_id: str
    exploratory: bool
    locked_panel_replacement: bool
    schedule: AuditSchedule
    candidates: tuple[AuditCandidate, ...]
    omitted_optional_candidates: tuple[Mapping[str, str], ...]
    source_roots: tuple[AuditSourceRoot, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "exploratory": self.exploratory,
            "locked_panel_replacement": self.locked_panel_replacement,
            "schedule": self.schedule.to_dict(),
            "candidates": [candidate.to_dict() for candidate in self.candidates],
            "omitted_optional_candidates": [dict(item) for item in self.omitted_optional_candidates],
        }
        if self.schema_version >= 2:
            payload["source_roots"] = [source.to_dict() for source in self.source_roots]
        return payload


@dataclass(frozen=True)
class PreparedAuditInputs:
    scenario_bytes: bytes
    scenario_sha256: str
    source_contracts: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class RetainedArtifactIdentity:
    """Authenticated identity for one retained trace/replay pair."""

    trace_path: str
    trace_sha256: str
    trace_byte_size: int
    replay_path: str
    replay_sha256: str
    replay_byte_size: int


@dataclass(frozen=True)
class RetainedEvaluation:
    """Canonical rows and physical identities reconstructed from retained evidence."""

    evaluation: Mapping[str, Any]
    matches: tuple[Mapping[str, Any], ...]
    artifacts: tuple[RetainedArtifactIdentity, ...]


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
    config_environment = config.get("environment")
    if config_environment is not None and config_environment != environment:
        raise ValueError(f"{label} environment does not match its contract")
    if config.get("algorithm") != "maskable_ppo":
        raise ValueError(f"{label} algorithm must be maskable_ppo")
    if require_seed:
        seeds = tuple(
            seed
            for seed in (
                manifest.get("model_seed"),
                config.get("model_seed"),
                config.get("seed"),
            )
            if seed is not None
        )
        if not seeds or any(
            seed != _EXPECTED_SEED or isinstance(seed, bool) for seed in seeds
        ):
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
    candidates = discover_audit_candidates(
        clone_run=clone_run,
        ppo_run=ppo_run,
        scratch_run=scratch_run,
    )
    source_roots = [
        AuditSourceRoot("clone", str(_canonical_run(clone_run, label="clone"))),
        AuditSourceRoot("ppo", str(_canonical_run(ppo_run, label="PPO"))),
    ]
    if scratch_run is not None:
        source_roots.append(
            AuditSourceRoot("scratch", str(_canonical_run(scratch_run, label="scratch")))
        )
    return AuditDefinition(
        schema_version=2,
        audit_id="annihilation-checkpoint-audit-v1",
        exploratory=True,
        locked_panel_replacement=False,
        schedule=AuditSchedule(),
        candidates=candidates,
        omitted_optional_candidates=omitted,
        source_roots=tuple(source_roots),
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _validate_exact_candidate_set(definition: AuditDefinition) -> None:
    roles = tuple(source.role for source in definition.source_roots)
    if roles not in {("clone", "ppo"), ("clone", "ppo", "scratch")}:
        raise ValueError(
            "audit source roots must contain canonical clone/PPO roles and optional scratch"
        )
    canonical_paths: list[str] = []
    for source in definition.source_roots:
        canonical = str(
            _canonical_run(Path(source.source_run), label=f"{source.role} source")
        )
        if canonical != source.source_run:
            raise ValueError(f"{source.role} source root is not canonical")
        canonical_paths.append(canonical)
    if len(canonical_paths) != len(set(canonical_paths)):
        raise ValueError("audit source roots must be unique")

    expected_omission: tuple[Mapping[str, str], ...] = ()
    if "scratch" not in roles:
        expected_omission = (
            {"family": "scratch_ppo", "reason": "no physical compatible run supplied"},
        )
    if definition.omitted_optional_candidates != expected_omission:
        raise ValueError("audit candidate set has inconsistent optional scratch omission")

    candidate_ids = tuple(candidate.candidate_id for candidate in definition.candidates)
    if (
        len(candidate_ids) != len(set(candidate_ids))
        or any(_CANDIDATE_ID.fullmatch(candidate_id) is None for candidate_id in candidate_ids)
    ):
        raise ValueError("audit candidate set contains duplicate or unsafe candidate IDs")
    physical_membership = tuple(
        candidate.checkpoint_path
        for candidate in definition.candidates
        if candidate.family != "control"
    )
    if len(physical_membership) != len(set(physical_membership)):
        raise ValueError("audit candidate set contains duplicate physical checkpoints")

    by_role = {source.role: Path(source.source_run) for source in definition.source_roots}
    expected_candidates = discover_audit_candidates(
        clone_run=by_role["clone"],
        ppo_run=by_role["ppo"],
        scratch_run=by_role.get("scratch"),
    )
    if definition.candidates != expected_candidates and tuple(
        replace(candidate, checkpoint_sha256=None)
        if candidate.source_run is not None
        else candidate
        for candidate in definition.candidates
    ) != tuple(
        replace(candidate, checkpoint_sha256=None)
        if candidate.source_run is not None
        else candidate
        for candidate in expected_candidates
    ):
        raise ValueError(
            "frozen audit candidate set does not exactly match rediscovered physical candidates"
        )


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


def _inspect_replays(paths: Sequence[Path]) -> Mapping[Path, int]:
    """Reconstruct retained replays through the authoritative engine reader in batches."""
    resolved = tuple(Path(path).resolve(strict=True) for path in paths)
    if not resolved:
        return {}
    repository = Path(__file__).resolve().parents[2]
    dll = repository / "engine" / "HexWars.Sim" / "bin" / "Debug" / "net8.0" / "HexWars.Sim.dll"
    project = repository / "engine" / "HexWars.Sim" / "HexWars.Sim.csproj"
    base = (
        ["dotnet", str(dll), "inspect"]
        if dll.is_file()
        else ["dotnet", "run", "--project", str(project), "--configuration", "Debug", "--", "inspect"]
    )
    chunks: list[list[Path]] = []
    current: list[Path] = []
    length = sum(len(part) + 3 for part in base)
    for path in resolved:
        addition = len(str(path)) + 3
        if current and length + addition > 24_000:
            chunks.append(current)
            current = []
            length = sum(len(part) + 3 for part in base)
        current.append(path)
        length += addition
    if current:
        chunks.append(current)

    winners: dict[Path, int] = {}
    winner_values = {"DRAW": -1, "P0": 0, "P1": 1}
    for chunk in chunks:
        try:
            result = subprocess.run(
                [*base, *(str(path) for path in chunk)],
                cwd=repository,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("retained replay failed authoritative engine reconstruction") from error
        summaries = [
            match
            for line in result.stdout.splitlines()
            if (match := re.search(r": round=\d+ winner=(DRAW|P0|P1)$", line))
        ]
        if len(summaries) != len(chunk):
            raise ValueError("authoritative replay inspector returned incomplete results")
        for path, match in zip(chunk, summaries):
            winners[path] = winner_values[match.group(1)]
    return winners


def _evaluation_source_identity() -> Mapping[str, Any]:
    """Bind evaluation to one commit and its relevant tracked source diff."""
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
        tracked_diff = subprocess.run(
            [
                "git",
                "diff",
                "--binary",
                "--no-ext-diff",
                "HEAD",
                "--",
                ".",
                ":(exclude)**/__pycache__/**",
                ":(exclude)**/*.pyc",
            ],
            cwd=repository,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("checkpoint audit requires a readable evaluation source identity") from error
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("evaluation source Git commit must be a full lowercase SHA-1")
    return {
        "repository": str(repository),
        "commit": commit,
        "tracked_diff_sha256": _sha256(tracked_diff),
    }


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


def validate_prepared_definition(
    definition: AuditDefinition,
    *,
    _smoke: bool = False,
    _allow_completed_legacy: bool = False,
) -> PreparedAuditInputs:
    """Reopen and validate every learned physical input frozen by prepare."""
    _validate_global_audit_definition(
        definition,
        smoke=_smoke,
        allow_completed_legacy=_allow_completed_legacy,
    )
    scenario_bytes, scenario_sha256, source_contracts = _source_material(definition)
    for candidate in definition.candidates:
        if candidate.source_run is None:
            continue
        if candidate.checkpoint_path is None or candidate.checkpoint_sha256 is None:
            raise ValueError(
                f"{candidate.candidate_id} is missing frozen checkpoint provenance"
            )
        checkpoint, digest = _checkpoint_bytes(
            Path(candidate.checkpoint_path), label=candidate.candidate_id
        )
        if str(checkpoint) != candidate.checkpoint_path:
            raise ValueError(
                f"{candidate.candidate_id} checkpoint path changed after prepare"
            )
        if digest != candidate.checkpoint_sha256:
            raise ValueError(
                f"{candidate.candidate_id} checkpoint bytes changed after prepare"
            )
    return PreparedAuditInputs(
        scenario_bytes=scenario_bytes,
        scenario_sha256=scenario_sha256,
        source_contracts=source_contracts,
    )


def _validate_runtime_contract(
    runtime: Mapping[str, Any], source_contracts: Sequence[Mapping[str, Any]]
) -> None:
    if runtime.get("environment") != "tactical-v2" or runtime.get("version") != "tactical-v2":
        raise ValueError("evaluation runtime contract must be tactical-v2")
    if not source_contracts:
        raise ValueError("audit manifest is missing source contracts")
    runtime_board = _require_mapping(
        runtime.get("board"), label="evaluation runtime board"
    )
    runtime_geometry = {
        dimension: _require_int(
            runtime_board.get(dimension),
            label=f"evaluation runtime board {dimension}",
        )
        for dimension in ("width", "height")
    }
    for index, source_row in enumerate(source_contracts, start=1):
        row = _require_mapping(source_row, label=f"source contract row {index}")
        source = _require_mapping(
            row.get("contract"), label=f"source contract {index}"
        )
        for field in (
            "environment",
            "version",
            "encoding_hash",
            "observation_size",
            "action_size",
        ):
            if runtime.get(field) != source.get(field):
                raise ValueError(
                    f"evaluation runtime contract {field} is incompatible "
                    f"with source contract {index}"
                )
        source_board = _require_mapping(
            source.get("board"), label=f"source contract {index} board"
        )
        for dimension, runtime_size in runtime_geometry.items():
            source_size = _require_int(
                source_board.get(dimension),
                label=f"source contract {index} board {dimension}",
            )
            if runtime_size != source_size:
                raise ValueError(
                    "evaluation runtime contract board geometry is incompatible "
                    f"with source contract {index}"
                )


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
    smoke: bool = False,
    evaluation_source_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    definition_payload, definition_sha256 = _definition_identity(definition)
    return {
        "schema_version": 1,
        "smoke": smoke,
        "exploratory": definition.exploratory,
        "locked_panel_replacement": definition.locked_panel_replacement,
        "generated_at": utc_now(),
        "state": "in_progress",
        "definition": definition_payload,
        "definition_sha256": definition_sha256,
        "repository_identity": dict(_repository_identity()),
        "evaluation_source_identity": dict(
            _evaluation_source_identity()
            if evaluation_source_identity is None
            else evaluation_source_identity
        ),
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
    if not isinstance(manifest.get("smoke"), bool):
        raise ValueError("audit manifest smoke flag must be boolean")
    if manifest.get("exploratory") is not True:
        raise ValueError("audit manifest must remain exploratory")
    if manifest.get("locked_panel_replacement") is not False:
        raise ValueError("audit manifest cannot replace the locked panel")

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
    evaluation_source = manifest.get("evaluation_source_identity")
    if evaluation_source is None:
        if state != "completed":
            raise ValueError("in-progress audit manifest is missing evaluation source identity")
    else:
        evaluation_source = _require_mapping(
            evaluation_source, label="audit manifest evaluation source identity"
        )
        if set(evaluation_source) != {
            "repository",
            "commit",
            "tracked_diff_sha256",
        }:
            raise ValueError("audit manifest evaluation source identity fields are invalid")
        _require_string(
            evaluation_source.get("repository"),
            label="audit manifest evaluation source repository",
        )
        source_commit = _require_string(
            evaluation_source.get("commit"),
            label="audit manifest evaluation source commit",
        )
        source_diff = _require_string(
            evaluation_source.get("tracked_diff_sha256"),
            label="audit manifest tracked source diff SHA-256",
        )
        if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not re.fullmatch(
            r"[0-9a-f]{64}", source_diff
        ):
            raise ValueError("audit manifest evaluation source identity is invalid")
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


def validate_completed_legacy_definition(
    root: Path, definition: AuditDefinition
) -> Mapping[str, Any]:
    """Accept schema-v1 definitions only as sealed, byte-identified evidence."""
    if definition.schema_version != 1 or definition.source_roots:
        raise ValueError("legacy audit validation requires a schema-v1 definition")
    resolved_root = Path(root).resolve(strict=True)
    manifest = _load_audit_manifest(resolved_root)
    if manifest.get("state") != "completed":
        raise ValueError("legacy audit evidence must already be completed")
    definition_payload, definition_sha256 = _definition_identity(definition)
    if (
        manifest.get("definition") != definition_payload
        or manifest.get("definition_sha256") != definition_sha256
    ):
        raise ValueError("legacy audit manifest does not match the frozen definition")
    _aggregate, aggregate_bytes = _read_json_bytes(
        resolved_root / "audit.json", label="legacy audit aggregate"
    )
    if _sha256(aggregate_bytes) != manifest.get("aggregate_sha256"):
        raise ValueError("legacy audit aggregate bytes do not match the manifest")
    return manifest

def _require_existing_manifest(
    root: Path,
    definition: AuditDefinition,
    *,
    scenario_bytes: bytes,
    scenario_sha256: str,
    source_contracts: Sequence[Mapping[str, Any]],
    runtime_contract: Mapping[str, Any],
    smoke: bool = False,
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
        manifest.get("smoke") is not smoke
        or manifest.get("exploratory") is not definition.exploratory
        or manifest.get("locked_panel_replacement")
        is not definition.locked_panel_replacement
    ):
        raise ValueError("existing audit manifest has different isolation flags")

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


def _require_exact_map_inventory(
    map_root: Path,
    evaluation_path: Path,
    artifact_pairs: set[tuple[Path, Path]],
) -> None:
    expected = {
        evaluation_path.relative_to(map_root).as_posix(),
        "evidence",
        "evidence/traces",
        "evidence/replays",
    }
    expected.update(
        path.relative_to(map_root).as_posix()
        for pair in artifact_pairs
        for path in pair
    )
    actual = {
        path.relative_to(map_root).as_posix()
        for path in map_root.rglob("*")
    }
    if actual != expected:
        raise ValueError("completed map directory contains an unexpected evidence inventory")


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
        if type(mapping.get(field)) is not int or mapping.get(field) != count:
            raise ValueError(f"{label} does not reconcile")


def _replay_paths_for_definition(
    root: Path,
    definition: AuditDefinition,
    *,
    existing_only: bool,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for candidate in definition.candidates:
        for offset in range(definition.schedule.maps):
            map_seed = definition.schedule.seed_start + offset
            evaluation_path = audit_map_path(root, candidate.candidate_id, map_seed)
            if existing_only and not evaluation_path.exists():
                continue
            evaluation, _raw = _read_json_bytes(
                evaluation_path, label="map evaluation for replay inspection"
            )
            matches = evaluation.get("matches")
            if not isinstance(matches, list) or len(matches) != 2:
                raise ValueError("map evaluation must contain two replays for inspection")
            map_root = evaluation_path.parent.resolve(strict=True)
            for match in matches:
                mapping = _require_mapping(match, label="map match for replay inspection")
                paths.append(
                    _artifact_path(
                        root,
                        map_root,
                        mapping.get("replay_path"),
                        label="replay path",
                    )
                )
    return tuple(paths)


def _trace_summary_payload(trace: EpisodeTrace, candidate_seat: int) -> Mapping[str, Any]:
    summary = summarize_episode(trace, candidate_seat)
    return {
        "command_count": summary.command_count,
        "round_count": summary.round_count,
        "damage_by_seat": list(summary.damage_by_seat),
        "kills_by_seat": list(summary.kills_by_seat),
        "end_turns_by_seat": list(summary.end_turns_by_seat),
        "wasted_end_turns_by_seat": list(summary.wasted_end_turns_by_seat),
        "peak_normalized_advantage": summary.peak_normalized_advantage,
        "final_normalized_advantage": summary.final_normalized_advantage,
        "maximum_state_repetition": summary.maximum_state_repetition,
    }


def _trace_classification_payload(
    trace: EpisodeTrace,
    *,
    candidate_seat: int,
    terminated: bool,
    truncated: bool,
) -> Mapping[str, Any]:
    classification = classify_draw(
        trace,
        candidate_seat=candidate_seat,
        terminated=terminated,
        truncated=truncated,
        winner=None,
    )
    return {
        "primary": classification.primary.value,
        "flags": [flag.value for flag in classification.flags],
        "evidence": dict(classification.evidence),
    }


def _validated_trace(path: Path, *, candidate_seat: int) -> tuple[int, bool, bool, Mapping[str, Any], Mapping[str, Any] | None]:
    payload, _raw = _read_json_bytes(path, label="retained trace")
    try:
        trace = EpisodeTrace.from_payload(payload)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("retained trace payload is invalid") from error
    if not trace.transitions:
        raise ValueError("retained trace is empty")
    final = trace.transitions[-1].after
    terminated = final.is_game_over
    truncated = not terminated
    winner = final.winner if terminated and final.winner is not None else -1
    classification = (
        _trace_classification_payload(
            trace,
            candidate_seat=candidate_seat,
            terminated=terminated,
            truncated=truncated,
        )
        if winner == -1
        else None
    )
    return (
        winner,
        terminated,
        truncated,
        _trace_summary_payload(trace, candidate_seat),
        classification,
    )


def _validate_retained_schedule(schedule: AuditSchedule) -> None:
    if type(schedule) is not AuditSchedule:
        raise ValueError("retained evaluation schedule must be an AuditSchedule")
    if (
        type(schedule.seed_start) is not int
        or schedule.seed_start < 0
        or type(schedule.maps) is not int
        or schedule.maps < 1
        or schedule.both_seats is not True
        or schedule.profile != "standard-3v3"
        or schedule.opponent != "random"
    ):
        raise ValueError(
            "retained evaluation requires reciprocal standard-3v3 maps versus random"
        )


_RETAINED_EVALUATION_FIELDS = frozenset({
    "schema_version", "generated_at", "schedule", "candidate", "opponent",
    "seed_start", "seeds", "reciprocal", "games", "wins", "losses",
    "draws", "rates", "confidence_intervals", "seat_results", "matches",
    "evidence",
})
_RETAINED_MATCH_FIELDS = frozenset({
    "seed", "candidate_seat", "winner", "outcome", "start_profile",
    "reference_seat", "terminated", "truncated", "summary", "classification",
    "trace_path", "trace_sha256", "trace_byte_size", "replay_path",
    "replay_sha256", "replay_byte_size",
})
_LEGACY_RETAINED_MATCH_FIELDS = _RETAINED_MATCH_FIELDS - {
    "trace_byte_size", "replay_byte_size",
}


def _require_exact_fields(
    value: object, fields: frozenset[str], *, label: str,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label=label)
    if set(mapping) != fields:
        raise ValueError(f"{label} fields are invalid")
    return mapping


def _require_exact_json(value: object, expected: object, *, label: str) -> None:
    """Compare JSON values without Python's bool/int/float equality aliases."""

    if isinstance(expected, Mapping):
        actual = _require_mapping(value, label=label)
        if set(actual) != set(expected):
            raise ValueError(f"{label} fields are invalid")
        for key, expected_item in expected.items():
            _require_exact_json(
                actual[key], expected_item, label=f"{label}.{key}",
            )
        return
    if isinstance(expected, (list, tuple)):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f"{label} list identity changed")
        for index, (actual_item, expected_item) in enumerate(
            zip(value, expected, strict=True)
        ):
            _require_exact_json(
                actual_item, expected_item, label=f"{label}[{index}]",
            )
        return
    if expected is None:
        if value is not None:
            raise ValueError(f"{label} must be null")
        return
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{label} identity changed")
    if type(value) is float and not math.isfinite(value):
        raise ValueError(f"{label} must be finite")


def _retained_artifact_path(
    publication_root: Path,
    evidence_root: Path,
    raw_path: object,
    *,
    label: str,
) -> Path:
    value = _require_string(raw_path, label=label)
    supplied = Path(value)
    candidate = supplied if supplied.is_absolute() else publication_root / supplied
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(publication_root)
        resolved.relative_to(evidence_root)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"{label} must resolve inside the retained evidence directory"
        ) from error
    if not resolved.is_file():
        raise ValueError(f"{label} must name a retained file")
    return resolved


def _require_exact_retained_inventory(
    evaluation_path: Path,
    evidence_root: Path,
    artifact_paths: Sequence[Path],
) -> None:
    root = evaluation_path.parent
    try:
        evidence_root.relative_to(root)
    except ValueError as error:
        raise ValueError("retained evidence root must be below the evaluation directory") from error
    expected = {evaluation_path.relative_to(root).as_posix()}
    expected.add(evidence_root.relative_to(root).as_posix())
    for artifact in artifact_paths:
        relative = artifact.relative_to(root)
        expected.add(relative.as_posix())
        parent = relative.parent
        while parent != Path("."):
            expected.add(parent.as_posix())
            parent = parent.parent
    actual: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("retained evaluation inventory contains a symbolic link")
        actual.add(path.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError(
            "retained evaluation inventory contains missing or unowned evidence"
        )


def _validate_retained_evaluation_core(
    evaluation_path: Path,
    *,
    publication_root: Path,
    evidence_root: Path,
    schedule: AuditSchedule,
    expected_candidate_identity: Mapping[str, Any],
    replay_winners: Mapping[Path, int] | None,
) -> RetainedEvaluation:
    _validate_retained_schedule(schedule)
    if not isinstance(expected_candidate_identity, Mapping) or not expected_candidate_identity:
        raise ValueError("expected retained candidate identity must be a non-empty object")
    try:
        publication = Path(publication_root).resolve(strict=True)
        evidence = Path(evidence_root).resolve(strict=True)
        path = Path(evaluation_path).resolve(strict=True)
        path.relative_to(publication)
        evidence.relative_to(publication)
    except (OSError, ValueError) as error:
        raise ValueError(
            "retained evaluation and evidence must exist below the publication root"
        ) from error
    if not publication.is_dir() or not evidence.is_dir() or not path.is_file():
        raise ValueError("retained evaluation publication paths are invalid")
    if evidence != path.parent / "evidence":
        raise ValueError("retained evidence root must be the evaluation evidence directory")

    evaluation, _raw = _read_json_bytes(path, label="retained evaluation")
    legacy_identity = evaluation.get("audit_identity")
    expected_top_fields = (
        _RETAINED_EVALUATION_FIELDS | {"audit_identity"}
        if legacy_identity is not None else _RETAINED_EVALUATION_FIELDS
    )
    _require_exact_fields(
        evaluation, frozenset(expected_top_fields), label="retained evaluation",
    )
    if type(evaluation.get("schema_version")) is not int or evaluation.get("schema_version") != 1:
        raise ValueError("retained evaluation schema_version must be 1")
    _require_string(
        evaluation.get("generated_at"), label="retained evaluation timestamp"
    )
    _require_exact_json(
        evaluation.get("candidate"), expected_candidate_identity,
        label="retained evaluation candidate controller",
    )
    _require_exact_json(
        evaluation.get("opponent"), _scripted_controller_identity("random"),
        label="retained evaluation opponent controller",
    )
    _require_exact_json(evaluation.get("schedule"), {
        "start_profile": "standard-3v3",
        "reference_seat_policy": "candidate-seat",
    }, label="retained evaluation profile schedule")

    expected_seeds = list(range(schedule.seed_start, schedule.seed_start + schedule.maps))
    expected_games = schedule.maps * 2
    if (
        type(evaluation.get("seed_start")) is not int
        or evaluation.get("seed_start") != schedule.seed_start
        or type(evaluation.get("seeds")) is not list
        or evaluation.get("reciprocal") is not True
        or type(evaluation.get("games")) is not int
        or evaluation.get("games") != expected_games
    ):
        raise ValueError("retained evaluation reciprocal seed schedule changed")
    _require_exact_json(
        evaluation.get("seeds"), expected_seeds,
        label="retained evaluation seeds",
    )
    matches_raw = evaluation.get("matches")
    if not isinstance(matches_raw, list) or len(matches_raw) != expected_games:
        raise ValueError("retained evaluation reciprocal match count changed")
    matches = tuple(
        _require_mapping(match, label=f"retained match {index}")
        for index, match in enumerate(matches_raw)
    )

    totals = {"wins": 0, "losses": 0, "draws": 0}
    seats = {
        "candidate_as_p0": {"wins": 0, "losses": 0, "draws": 0},
        "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 0},
    }
    draw_categories: dict[str, int] = {}
    artifact_paths: list[Path] = []
    artifact_identities: list[RetainedArtifactIdentity] = []
    replay_rows: list[tuple[int, Path]] = []
    seen_artifacts: set[Path] = set()
    legacy_artifacts: object = None
    if legacy_identity is not None:
        identity = _require_mapping(legacy_identity, label="retained audit identity")
        legacy_artifacts = identity.get("artifacts")
        if not isinstance(legacy_artifacts, list) or len(legacy_artifacts) != expected_games:
            raise ValueError("retained audit identity must hash every artifact pair")

    for index, match in enumerate(matches):
        _require_exact_fields(
            match,
            _LEGACY_RETAINED_MATCH_FIELDS
            if legacy_identity is not None else _RETAINED_MATCH_FIELDS,
            label=f"retained match {index}",
        )
        seed = schedule.seed_start + index // 2
        seat = index % 2
        if (
            type(match.get("seed")) is not int
            or match.get("seed") != seed
            or type(match.get("candidate_seat")) is not int
            or match.get("candidate_seat") != seat
        ):
            raise ValueError("retained evaluation match ordering or seed changed")
        if (
            match.get("start_profile") != "standard-3v3"
            or type(match.get("reference_seat")) is not int
            or match.get("reference_seat") != seat
        ):
            raise ValueError("retained evaluation match profile reference changed")
        winner = match.get("winner")
        if type(winner) is not int or winner not in {-1, 0, 1}:
            raise ValueError("retained evaluation match winner is invalid")
        outcome = "win" if winner == seat else "loss" if winner in {0, 1} else "draw"
        if match.get("outcome") != outcome:
            raise ValueError("retained evaluation outcome does not reconcile")
        if type(match.get("terminated")) is not bool or type(match.get("truncated")) is not bool:
            raise ValueError("retained evaluation termination fields are invalid")

        trace_path = _retained_artifact_path(
            publication, evidence, match.get("trace_path"), label="retained trace path"
        )
        replay_path = _retained_artifact_path(
            publication, evidence, match.get("replay_path"), label="retained replay path"
        )
        if trace_path == replay_path or trace_path in seen_artifacts or replay_path in seen_artifacts:
            raise ValueError("retained matches must own distinct trace and replay artifacts")
        seen_artifacts.update((trace_path, replay_path))
        artifact_paths.extend((trace_path, replay_path))
        trace_bytes = trace_path.read_bytes()
        replay_bytes = replay_path.read_bytes()
        trace_sha256 = _sha256(trace_bytes)
        replay_sha256 = _sha256(replay_bytes)
        for field, expected in (
            ("trace_sha256", trace_sha256),
            ("replay_sha256", replay_sha256),
        ):
            if type(match.get(field)) is not str or match.get(field) != expected:
                raise ValueError(f"retained match {field} changed")
        if legacy_identity is None:
            if (
                type(match.get("trace_byte_size")) is not int
                or match.get("trace_byte_size") != len(trace_bytes)
                or type(match.get("replay_byte_size")) is not int
                or match.get("replay_byte_size") != len(replay_bytes)
            ):
                raise ValueError("retained match artifact byte sizes changed")
        if isinstance(legacy_artifacts, list):
            legacy = _require_exact_fields(
                legacy_artifacts[index],
                frozenset({"trace_sha256", "replay_sha256"}),
                label=f"retained artifact identity {index}",
            )
            if (
                legacy.get("trace_sha256") != trace_sha256
                or legacy.get("replay_sha256") != replay_sha256
            ):
                raise ValueError("retained audit artifact bytes changed")

        (
            trace_winner,
            trace_terminated,
            trace_truncated,
            trace_summary,
            trace_classification,
        ) = _validated_trace(trace_path, candidate_seat=seat)
        if winner != trace_winner:
            raise ValueError("retained winner does not reconcile with its trace")
        if (
            match.get("terminated") is not trace_terminated
            or match.get("truncated") is not trace_truncated
        ):
            raise ValueError("retained termination does not reconcile with its trace")
        _require_exact_json(
            match.get("summary"), trace_summary,
            label="retained summary",
        )
        _require_exact_json(
            match.get("classification"), trace_classification,
            label="retained classification",
        )

        replay_rows.append((winner, replay_path))
        artifact_identities.append(RetainedArtifactIdentity(
            trace_path=trace_path.relative_to(publication).as_posix(),
            trace_sha256=trace_sha256,
            trace_byte_size=len(trace_bytes),
            replay_path=replay_path.relative_to(publication).as_posix(),
            replay_sha256=replay_sha256,
            replay_byte_size=len(replay_bytes),
        ))
        counter = "losses" if outcome == "loss" else f"{outcome}s"
        totals[counter] += 1
        seat_key = "candidate_as_p0" if seat == 0 else "candidate_as_p1"
        seats[seat_key][counter] += 1
        if outcome == "draw":
            classification = _require_mapping(
                trace_classification, label="retained draw classification"
            )
            primary = _require_string(
                classification.get("primary"), label="retained draw classification primary"
            )
            draw_categories[primary] = draw_categories.get(primary, 0) + 1

    authoritative_winners = (
        _inspect_replays(tuple(path for _winner, path in replay_rows))
        if replay_winners is None
        else replay_winners
    )
    for winner, replay_path in replay_rows:
        if authoritative_winners.get(replay_path) != winner:
            raise ValueError("retained winner does not reconcile with its replay")

    _require_exact_retained_inventory(path, evidence, artifact_paths)
    _require_exact_counts(
        {key: evaluation.get(key) for key in totals},
        totals,
        label="retained outcome totals",
    )
    expected_rates = {
        "win": totals["wins"] / expected_games,
        "loss": totals["losses"] / expected_games,
        "draw": totals["draws"] / expected_games,
    }
    _require_exact_json(
        evaluation.get("rates"), expected_rates,
        label="retained outcome rates",
    )
    expected_intervals = {
        outcome: wilson_interval(totals[counter], expected_games, 0.95)
        for outcome, counter in (("win", "wins"), ("loss", "losses"), ("draw", "draws"))
    }
    _require_exact_json(
        evaluation.get("confidence_intervals"), expected_intervals,
        label="retained Wilson confidence intervals",
    )
    seat_results = _require_mapping(evaluation.get("seat_results"), label="retained seat results")
    if set(seat_results) != set(seats):
        raise ValueError("retained seat result fields are invalid")
    for seat_key, counts in seats.items():
        _require_exact_counts(seat_results.get(seat_key), counts, label=seat_key)
    expected_evidence = {
        "retention": "all",
        "retained": expected_games,
        "draw_traces": totals["draws"],
        "control_traces": expected_games - totals["draws"],
        "draw_categories": dict(sorted(draw_categories.items())),
    }
    _require_exact_json(
        evaluation.get("evidence"), expected_evidence,
        label="retained evidence summary",
    )
    return RetainedEvaluation(
        evaluation=evaluation,
        matches=matches,
        artifacts=tuple(artifact_identities),
    )


def validate_retained_evaluation(
    evaluation_path: Path,
    *,
    publication_root: Path,
    evidence_root: Path,
    schedule: AuditSchedule,
    expected_candidate_identity: Mapping[str, Any],
) -> RetainedEvaluation:
    """Authoritatively reopen one generic retained reciprocal evaluation."""

    return _validate_retained_evaluation_core(
        evaluation_path,
        publication_root=publication_root,
        evidence_root=evidence_root,
        schedule=schedule,
        expected_candidate_identity=expected_candidate_identity,
        replay_winners=None,
    )


def _seal_retained_evaluation_artifacts(
    evaluation_path: Path, *, publication_root: Path, evidence_root: Path,
) -> None:
    """Add mandatory physical hashes/sizes before public validation and reuse."""

    evaluation, _raw = _read_json_bytes(
        evaluation_path, label="retained evaluator output",
    )
    matches = evaluation.get("matches")
    if not isinstance(matches, list):
        raise ValueError("retained evaluator output matches must be a list")
    sealed_matches: list[dict[str, Any]] = []
    for index, raw_match in enumerate(matches):
        match = dict(_require_mapping(raw_match, label=f"retained match {index}"))
        trace = _retained_artifact_path(
            publication_root, evidence_root, match.get("trace_path"),
            label="retained trace path",
        )
        replay = _retained_artifact_path(
            publication_root, evidence_root, match.get("replay_path"),
            label="retained replay path",
        )
        trace_bytes = trace.read_bytes()
        replay_bytes = replay.read_bytes()
        match.update({
            "trace_sha256": _sha256(trace_bytes),
            "trace_byte_size": len(trace_bytes),
            "replay_sha256": _sha256(replay_bytes),
            "replay_byte_size": len(replay_bytes),
        })
        sealed_matches.append(match)
    sealed = dict(evaluation)
    sealed["matches"] = sealed_matches
    atomic_write_json(evaluation_path, sealed)


def evaluate_retained_candidate(
    controller: str,
    *,
    expected_candidate_identity: Mapping[str, Any],
    schedule: AuditSchedule,
    publication_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_controllers,
) -> RetainedEvaluation:
    """Run the generic audit-grade evaluator and reopen all published evidence."""

    _validate_retained_schedule(schedule)
    _require_string(controller, label="retained candidate controller")
    if type(workers) is not int or workers < 1:
        raise ValueError("retained evaluation workers must be positive")
    if not isinstance(server_cmd, Sequence) or isinstance(server_cmd, (str, bytes)):
        raise ValueError("retained evaluation server command must be a sequence")
    root = Path(publication_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    evaluation_path = root / "evaluation.json"
    if evaluation_path.exists():
        raise ValueError("retained evaluation output already exists")
    result = evaluator(
        controller,
        "random",
        games=schedule.maps,
        seed_start=schedule.seed_start,
        both_seats=True,
        workers=workers,
        server_cmd=server_cmd,
        output_path=evaluation_path,
        environment="tactical-v2",
        evidence_dir=root / "evidence",
        start_profile="standard-3v3",
        capture_trace=True,
        evidence_retention="all",
    )
    if not isinstance(result, Mapping):
        raise ValueError("retained evaluator must return an evaluation mapping")
    _seal_retained_evaluation_artifacts(
        evaluation_path,
        publication_root=root,
        evidence_root=root / "evidence",
    )
    return validate_retained_evaluation(
        evaluation_path,
        publication_root=root,
        evidence_root=root / "evidence",
        schedule=schedule,
        expected_candidate_identity=expected_candidate_identity,
    )


def validate_physical_map(
    root: Path,
    candidate: AuditCandidate,
    schedule: AuditSchedule,
    map_seed: int,
    *,
    _replay_winners: Mapping[Path, int] | None = None,
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
    frozen_evaluation_source = manifest.get("evaluation_source_identity")
    if (
        frozen_evaluation_source is not None
        and audit_identity.get("evaluation_source_identity") != frozen_evaluation_source
    ):
        raise ValueError("map evaluation source identity changed")
    if audit_identity.get("checkpoint_sha256") != _checkpoint_digest(candidate):
        raise ValueError("map audit checkpoint identity changed")

    # Keep the legacy audit-specific envelope checks above, then delegate all
    # retained game/evidence semantics to the public generic core.  The legacy
    # checks below remain as defense in depth and preserve its frozen contract.
    _validate_retained_evaluation_core(
        evaluation_path,
        publication_root=audit_root,
        evidence_root=map_root / "evidence",
        schedule=replace(schedule, seed_start=map_seed, maps=1),
        expected_candidate_identity=expected_candidate,
        replay_winners=_replay_winners,
    )

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
    replay_rows: list[tuple[int, Path]] = []
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
        (
            trace_winner,
            trace_terminated,
            trace_truncated,
            trace_summary,
            trace_classification,
        ) = _validated_trace(trace_path, candidate_seat=seat)
        if winner != trace_winner:
            raise ValueError("map match winner does not reconcile with retained trace")
        if (
            match.get("terminated") is not trace_terminated
            or match.get("truncated") is not trace_truncated
        ):
            raise ValueError("map match termination does not reconcile with retained trace")
        if match.get("summary") != trace_summary:
            raise ValueError("map match summary does not reconcile with retained trace")
        if match.get("classification") != trace_classification:
            raise ValueError("map match classification does not reconcile with retained trace")
        replay_rows.append((winner, replay_path))

        counter = f"{outcome}s" if outcome != "loss" else "losses"
        totals[counter] += 1
        seat_key = "candidate_as_p0" if seat == 0 else "candidate_as_p1"
        seats[seat_key][counter] += 1

    replay_winners = (
        _inspect_replays(tuple(path for _winner, path in replay_rows))
        if _replay_winners is None
        else _replay_winners
    )
    for winner, replay_path in replay_rows:
        if replay_winners.get(replay_path) != winner:
            raise ValueError("map match winner does not reconcile with retained replay")
    _require_exact_map_inventory(map_root, evaluation_path, artifact_pairs)
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
        "evaluation_source_identity": manifest["evaluation_source_identity"],
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


def _validate_global_audit_definition(
    definition: AuditDefinition,
    *,
    smoke: bool = False,
    allow_completed_legacy: bool = False,
) -> None:
    expected_schedule = _PROGRAMMATIC_SMOKE_SCHEDULE if smoke else AuditSchedule()
    valid_schema = definition.schema_version == 2 or (
        allow_completed_legacy and definition.schema_version == 1 and not definition.source_roots
    )
    if (
        not valid_schema
        or definition.audit_id != "annihilation-checkpoint-audit-v1"
        or definition.exploratory is not True
        or definition.locked_panel_replacement is not False
        or definition.schedule != expected_schedule
    ):
        raise ValueError(
            "checkpoint audit definition or schedule violates the frozen "
            "global identity/isolation contract"
        )
    if definition.schema_version == 2:
        _validate_exact_candidate_set(definition)


def evaluate_audit(
    definition: AuditDefinition,
    *,
    output_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_controllers,
    progress: Callable[[str], None] = print,
    _smoke: bool = False,
) -> Mapping[str, Any]:
    """Evaluate every immutable candidate/map and reuse only validated physical evidence."""
    if workers < 1:
        raise ValueError("checkpoint audit workers must be positive")
    _validate_global_audit_definition(definition, smoke=_smoke)
    schedule = definition.schedule
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_path = root / "manifest.json"
    evaluation_source_identity = dict(_evaluation_source_identity())
    if manifest_path.exists():
        frozen_manifest = _load_audit_manifest(root)
        if frozen_manifest.get("evaluation_source_identity") != evaluation_source_identity:
            raise ValueError("current evaluation source identity differs from the frozen audit")
    prepared = validate_prepared_definition(definition, _smoke=_smoke)
    scenario_bytes = prepared.scenario_bytes
    scenario_sha256 = prepared.scenario_sha256
    source_contracts = prepared.source_contracts
    runtime_contract = dict(_runtime_contract(server_cmd))
    _validate_runtime_contract(runtime_contract, source_contracts)
    if manifest_path.exists():
        manifest = _require_existing_manifest(
            root,
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
            smoke=_smoke,
        )
    else:
        manifest = _initial_manifest(
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
            smoke=_smoke,
            evaluation_source_identity=evaluation_source_identity,
        )
        atomic_write_json(manifest_path, manifest)

    existing_replay_winners = _inspect_replays(
        _replay_paths_for_definition(root, definition, existing_only=True)
    )
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
                validate_physical_map(
                    root,
                    candidate,
                    schedule,
                    map_seed,
                    _replay_winners=existing_replay_winners,
                )
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
    if not math.isfinite(value):
        raise ValueError(f"match summary {raw_field} metric must be finite")
    if aggregate_field in {"round_count", "command_count"}:
        if type(value) is not int or value < 0:
            raise ValueError(
                f"match summary {raw_field} metric must be a nonnegative integer"
            )
    elif not -1.0 <= value <= 1.0:
        raise ValueError(
            f"match summary {raw_field} advantage must be between -1 and 1"
        )
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


def summarize_candidate(
    rows: Sequence[Mapping[str, Any]], _expected_games: int = 200
) -> Mapping[str, Any]:
    """Summarize one candidate's frozen reciprocal panel."""
    if len(rows) != _expected_games:
        raise ValueError(
            f"candidate summary requires exactly {_expected_games} match rows"
        )
    counts = {"games": _expected_games, "wins": 0, "losses": 0, "draws": 0}
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
        if peak_advantage < final_advantage:
            raise ValueError(
                "match summary peak advantage cannot be below final advantage"
            )
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
        if any(
            wasted_end_turns[index] > end_turns[index] for index in (0, 1)
        ):
            raise ValueError("match summary wasted EndTurns exceed total EndTurns")

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
        "win": counts["wins"] / _expected_games,
        "loss": counts["losses"] / _expected_games,
        "draw": counts["draws"] / _expected_games,
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
            outcome: wilson_interval(counts[counter], _expected_games, 0.95)
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
            "cycling": {
                "count": cycling_draws,
                "incidence": cycling_draws / _expected_games,
            },
            "action_waste": {
                "count": action_waste_draws,
                "incidence": action_waste_draws / _expected_games,
            },
            "primary_categories": dict(sorted(primary_categories.items())),
        },
        "winning_games": {
            "round_count": _distribution(win_rounds),
            "command_count": _distribution(win_commands),
        },
        "normalized_advantage": {
            "all_games": {
                "final": sum(all_final_advantage) / _expected_games,
                "peak": sum(all_peak_advantage) / _expected_games,
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
    rows: Sequence[Mapping[str, Any]],
    *,
    label: str,
    expected_games: int = 200,
) -> Mapping[tuple[int, int], str]:
    if len(rows) != expected_games:
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
    _expected_games: int = 200,
) -> Mapping[str, Any]:
    """Compare two candidate panels on identical reciprocal schedule keys."""
    left = _paired_index(
        earlier, label="earlier", expected_games=_expected_games
    )
    right = _paired_index(
        later, label="later", expected_games=_expected_games
    )
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
        "absolute_win_rate_change": net_change / _expected_games,
        "exact_sign_test_p_value": _exact_sign_test(left_only, right_only),
    }

CandidateAggregate = Mapping[str, Any]


def _aggregate_summary(candidate: CandidateAggregate) -> Mapping[str, Any]:
    return _require_mapping(candidate.get("summary"), label="candidate aggregate summary")


def _aggregate_win_count(candidate: CandidateAggregate) -> int:
    counts = _require_mapping(
        _aggregate_summary(candidate).get("counts"), label="candidate outcome counts"
    )
    games = counts.get("games")
    wins = counts.get("wins")
    if (
        isinstance(games, bool)
        or games != 200
        or isinstance(wins, bool)
        or not isinstance(wins, int)
        or not 0 <= wins <= 200
    ):
        raise ValueError("candidate aggregate must contain a valid 200-game win count")
    return wins


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
    ppo_wins = [_aggregate_win_count(row) for row in ppo]
    qualifying_indexes = [
        index for index, wins in enumerate(ppo_wins) if wins >= 130
    ]
    qualifying_ids = [
        _require_string(ppo[index].get("candidate_id"), label="candidate ID")
        for index in qualifying_indexes
    ]
    consistent_improvement = False
    if qualifying_indexes:
        earliest = qualifying_indexes[0] + 1
        wins_to_qualifier = [
            _aggregate_win_count(row) for row in ordered[: earliest + 1]
        ]
        changes = [
            later - earlier
            for earlier, later in zip(wins_to_qualifier, wins_to_qualifier[1:])
        ]
        consistent_improvement = all(change >= 0 for change in changes) and any(
            change > 0 for change in changes
        )

    large_late_regression = any(
        earlier - later >= 20
        for earlier_index, earlier in enumerate(ppo_wins)
        for later in ppo_wins[earlier_index + 1 :]
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

    all_ppo_below_half = bool(ppo_wins) and all(wins < 100 for wins in ppo_wins)
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
    _smoke: bool = False,
) -> Mapping[str, Any]:
    """Reopen every physical map, aggregate deterministically, then seal the audit."""
    root = Path(output_root).resolve(strict=True)
    if definition.schema_version == 1:
        manifest = validate_completed_legacy_definition(root, definition)
        _validate_global_audit_definition(
            definition, smoke=_smoke, allow_completed_legacy=True
        )
    else:
        _validate_global_audit_definition(definition, smoke=_smoke)
        manifest = _load_audit_manifest(root)
    definition_payload, definition_sha256 = _definition_identity(definition)
    if (
        manifest.get("definition") != definition_payload
        or manifest.get("definition_sha256") != definition_sha256
    ):
        raise ValueError("audit manifest does not match the requested frozen definition")
    if manifest.get("smoke") is not _smoke:
        raise ValueError("audit manifest smoke mode does not match aggregation mode")

    replay_winners = _inspect_replays(
        _replay_paths_for_definition(root, definition, existing_only=False)
    )
    candidates: dict[str, Mapping[str, Any]] = {}
    physical_rows: dict[str, list[Mapping[str, Any]]] = {}
    for candidate in definition.candidates:
        rows: list[Mapping[str, Any]] = []
        for offset in range(definition.schedule.maps):
            map_seed = definition.schedule.seed_start + offset
            _evaluation, matches = validate_physical_map(
                root,
                candidate,
                definition.schedule,
                map_seed,
                _replay_winners=replay_winners,
            )
            for candidate_seat, match in enumerate(matches):
                if match.get("candidate_seat") != candidate_seat:
                    raise ValueError("validated reciprocal matches are not ordered by candidate seat")
                rows.append({**dict(match), "map_seed": map_seed})
        expected_rows = definition.schedule.maps * 2
        if len(rows) != expected_rows:
            raise ValueError(
                f"{candidate.candidate_id} must contain exactly {expected_rows} ordered match rows"
            )
        summary = dict(
            summarize_candidate(rows, _expected_games=expected_rows)
            if _smoke
            else summarize_candidate(rows)
        )
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
                **paired_change(
                    physical_rows[earlier_id],
                    physical_rows[later_id],
                    _expected_games=definition.schedule.maps * 2,
                ),
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
        "smoke": _smoke,
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
        "decision": (
            {
                "available": False,
                "reason": "two-map programmatic smoke is not a decision panel",
            }
            if _smoke
            else choose_next_experiment(trajectory)
        ),
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


def run_programmatic_smoke(
    definition: AuditDefinition,
    *,
    output_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    evaluator: Callable[..., Mapping[str, Any]] = evaluate_controllers,
    progress: Callable[[str], None] = print,
) -> Mapping[str, Any]:
    """Run the exact two-map preflight without exposing a CLI schedule override."""
    _validate_global_audit_definition(definition)
    smoke_definition = replace(
        definition,
        schedule=_PROGRAMMATIC_SMOKE_SCHEDULE,
    )
    evaluation = evaluate_audit(
        smoke_definition,
        output_root=output_root,
        server_cmd=server_cmd,
        workers=workers,
        evaluator=evaluator,
        progress=progress,
        _smoke=True,
    )
    aggregate = aggregate_audit(
        smoke_definition,
        output_root=output_root,
        _smoke=True,
    )
    return {
        "definition": smoke_definition.to_dict(),
        "evaluation": dict(evaluation),
        "aggregate": dict(aggregate),
    }
