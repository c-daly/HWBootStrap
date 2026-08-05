"""Orchestrate the selective-DAgger training experiment."""

from __future__ import annotations

import json
import hashlib
import math
import os
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ml_lab.dagger as dagger_domain
import ml_lab.checkpoint_audit as checkpoint_audit_domain
import ml_lab.imitation as imitation_domain
from ml_lab.contracts import EnvironmentContract
from ml_lab.dagger import IterationManifest
from ml_lab.io import atomic_write_json


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PANEL_PATH = (
    _REPOSITORY_ROOT
    / "python"
    / "panels"
    / "annihilation-selective-dagger-v1"
    / "panel.json"
)
_PREPARE_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "identity", "artifacts", "content_identity",
})
_PREPARE_IDENTITY_FIELDS = frozenset({"definition", "repository"})
_DEFINITION_IDENTITY_FIELDS = frozenset({
    "panel_sha256", "panel_byte_size", "seed_banks_sha256",
    "seed_banks_byte_size",
})
_REPOSITORY_IDENTITY_FIELDS = frozenset({
    "root", "commit", "source_tree", "dirty",
})


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _content_identity(value: Mapping[str, Any]) -> str:
    canonical = {
        key: item for key, item in value.items() if key != "content_identity"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("stage mapping keys must be strings")
        return MappingProxyType({
            key: _freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    if value is None or type(value) in {bool, int, float, str}:
        return value
    raise ValueError("stage value is not JSON-compatible")


def _same_json(left: Any, right: Any) -> bool:
    def thaw(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {key: thaw(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [thaw(item) for item in value]
        return value

    return json.dumps(
        thaw(left), sort_keys=True, separators=(",", ":"), allow_nan=False,
    ) == json.dumps(
        thaw(right), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


def _git_repository_identity(repository_root: Path) -> Mapping[str, Any]:
    root = Path(repository_root).resolve(strict=True)

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode:
            raise ValueError(
                f"repository identity command failed: {' '.join(arguments)}"
            )
        return completed.stdout.strip()

    return {
        "root": str(Path(git("rev-parse", "--show-toplevel")).resolve(strict=True)),
        "commit": git("rev-parse", "HEAD").lower(),
        "source_tree": git("rev-parse", "HEAD^{tree}").lower(),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=all")),
    }


def _recorded_repository_identity(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != _REPOSITORY_IDENTITY_FIELDS:
        raise ValueError("prepare repository identity fields are invalid")
    try:
        claimed_root = Path(raw["root"]).resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError("prepare repository root is invalid") from exc
    for name in ("commit", "source_tree"):
        value = raw[name]
        if (
            not isinstance(value, str)
            or len(value) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"prepare repository {name} is invalid")
    if type(raw["dirty"]) is not bool or raw["dirty"]:
        raise ValueError("prepare requires a clean repository")
    return {
        "root": str(claimed_root),
        "commit": raw["commit"],
        "source_tree": raw["source_tree"],
        "dirty": False,
    }


def _validated_repository_identity(
    repository_root: Path,
    provider: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    root = Path(repository_root).resolve(strict=True)
    recorded = _recorded_repository_identity(provider(root))
    if Path(recorded["root"]) != root:
        raise ValueError("prepare repository root identity changed")
    return recorded


def _write_exact_file(path: Path, value: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class PreparedStage:
    root: Path
    identity: Mapping[str, Any]
    content_identity: str

    @property
    def panel_path(self) -> Path:
        return self.root / "panel.json"

    @property
    def seed_banks_path(self) -> Path:
        return self.root / "seed-banks.json"


def _open_prepared_stage(
    root: Path,
    *,
    expected_identity: Mapping[str, Any] | None = None,
) -> PreparedStage:
    stage_root = Path(root)
    manifest_path = stage_root / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared definition manifest is unreadable") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _PREPARE_MANIFEST_FIELDS
        or raw["schema_version"] != 1
        or raw["status"] != "completed"
    ):
        raise ValueError("prepared definition manifest fields are invalid")
    identity = raw["identity"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _PREPARE_IDENTITY_FIELDS
        or not isinstance(identity["definition"], Mapping)
        or set(identity["definition"]) != _DEFINITION_IDENTITY_FIELDS
        or not isinstance(identity["repository"], Mapping)
        or set(identity["repository"]) != _REPOSITORY_IDENTITY_FIELDS
    ):
        raise ValueError("prepared definition identity fields are invalid")
    definition = identity["definition"]
    repository = _recorded_repository_identity(identity["repository"])
    for name in ("panel_sha256", "seed_banks_sha256"):
        value = definition[name]
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError("prepared definition hash is invalid")
    for name in ("panel_byte_size", "seed_banks_byte_size"):
        if type(definition[name]) is not int or definition[name] < 1:
            raise ValueError("prepared definition byte size is invalid")
    artifacts = raw["artifacts"]
    if (
        not isinstance(artifacts, Mapping)
        or set(artifacts) != {"panel", "seed_banks"}
    ):
        raise ValueError("prepared definition artifact fields are invalid")
    expected_descriptors = {
        "panel": {
            "path": "panel.json",
            "sha256": definition["panel_sha256"],
            "byte_size": definition["panel_byte_size"],
        },
        "seed_banks": {
            "path": "seed-banks.json",
            "sha256": definition["seed_banks_sha256"],
            "byte_size": definition["seed_banks_byte_size"],
        },
    }
    if not _same_json(artifacts, expected_descriptors):
        raise ValueError("prepared definition descriptors are inconsistent")
    for artifact, label in (("panel", "panel"), ("seed_banks", "seed banks")):
        descriptor = expected_descriptors[artifact]
        path = stage_root / descriptor["path"]
        try:
            value = path.read_bytes()
        except OSError as exc:
            raise ValueError(
                f"prepared {label} physical bytes are missing"
            ) from exc
        if (
            len(value) != descriptor["byte_size"]
            or _sha256_bytes(value) != descriptor["sha256"]
        ):
            raise ValueError(f"prepared {label} physical bytes changed")
    content_identity = raw["content_identity"]
    if (
        not isinstance(content_identity, str)
        or _content_identity(raw) != content_identity
    ):
        raise ValueError("prepared definition content identity is invalid")
    if expected_identity is not None and not _same_json(identity, expected_identity):
        raise ValueError(
            "completed prepared definition identity differs; use a new output root"
        )
    return PreparedStage(
        root=stage_root.resolve(strict=True),
        identity=_freeze_json({
            "definition": definition,
            "repository": repository,
        }),
        content_identity=content_identity,
    )


def run_prepare(
    *,
    output_root: Path,
    panel_path: Path = _PANEL_PATH,
    repository_root: Path = _REPOSITORY_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> PreparedStage:
    """Publish the exact panel and seed-bank bytes once, then physically reopen."""

    source_panel = Path(panel_path).resolve(strict=True)
    source_seeds = (source_panel.parent / "seed-banks.json").resolve(strict=True)
    panel_bytes = source_panel.read_bytes()
    seed_bytes = source_seeds.read_bytes()
    repository = _validated_repository_identity(
        repository_root, repository_identity_provider,
    )
    identity = {
        "definition": {
            "panel_sha256": _sha256_bytes(panel_bytes),
            "panel_byte_size": len(panel_bytes),
            "seed_banks_sha256": _sha256_bytes(seed_bytes),
            "seed_banks_byte_size": len(seed_bytes),
        },
        "repository": repository,
    }
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / "definition"
    staging = root / "definition.staging"
    if destination.exists():
        if staging.exists():
            raise ValueError("prepared definition destination and staging are ambiguous")
        return _open_prepared_stage(
            destination, expected_identity=identity,
        )
    if staging.exists():
        raise ValueError(
            "prepared definition staging is partial; use a new output root"
        )

    staging.mkdir()
    _write_exact_file(staging / "panel.json", panel_bytes)
    _write_exact_file(staging / "seed-banks.json", seed_bytes)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "identity": identity,
        "artifacts": {
            "panel": {
                "path": "panel.json",
                "sha256": identity["definition"]["panel_sha256"],
                "byte_size": identity["definition"]["panel_byte_size"],
            },
            "seed_banks": {
                "path": "seed-banks.json",
                "sha256": identity["definition"]["seed_banks_sha256"],
                "byte_size": identity["definition"]["seed_banks_byte_size"],
            },
        },
    }
    manifest["content_identity"] = _content_identity(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    candidate = _open_prepared_stage(staging, expected_identity=identity)
    publication_repository = _validated_repository_identity(
        repository_root, repository_identity_provider,
    )
    if not _same_json(publication_repository, repository):
        raise ValueError("prepare repository identity changed before publication")
    os.replace(staging, destination)
    published = _open_prepared_stage(destination, expected_identity=identity)
    if published.content_identity != candidate.content_identity:
        raise ValueError("prepared definition changed during publication")
    return published


_VALIDATION_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "identity", "content_identity",
})
_VALIDATION_IDENTITY_FIELDS = frozenset({
    "prepared_content_identity", "physical", "runtime",
})
_VALIDATION_PHYSICAL_FIELDS = frozenset({
    "starting_learner", "base_dataset", "scenario", "contract",
    "seed_isolation",
})


def _require_sha256(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256")
    return value


def _validate_physical_identity(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _VALIDATION_PHYSICAL_FIELDS:
        raise ValueError("validation physical identity fields are invalid")
    learner = value["starting_learner"]
    if not isinstance(learner, Mapping) or set(learner) != {
        "checkpoint_path", "checkpoint_sha256", "source_run",
        "source_manifest_sha256",
    }:
        raise ValueError("validation starting learner identity is invalid")
    for name in ("checkpoint_path", "source_run"):
        if not isinstance(learner[name], str) or not learner[name]:
            raise ValueError("validation starting learner path is invalid")
    for name in ("checkpoint_sha256", "source_manifest_sha256"):
        _require_sha256(learner[name], f"validation learner {name}")

    dataset = value["base_dataset"]
    if not isinstance(dataset, Mapping) or set(dataset) != {
        "root", "manifest_sha256", "content_sha256", "file_count",
        "byte_size", "contract_hash", "encoding_hash", "scenario_hash",
    }:
        raise ValueError("validation base dataset identity is invalid")
    if not isinstance(dataset["root"], str) or not dataset["root"]:
        raise ValueError("validation base dataset root is invalid")
    for name in (
        "manifest_sha256", "content_sha256", "contract_hash", "encoding_hash",
        "scenario_hash",
    ):
        _require_sha256(dataset[name], f"validation dataset {name}")
    for name in ("file_count", "byte_size"):
        if type(dataset[name]) is not int or dataset[name] < 1:
            raise ValueError(f"validation dataset {name} is invalid")

    scenario = value["scenario"]
    if not isinstance(scenario, Mapping) or set(scenario) != {
        "source_sha256", "runtime_sha256",
    }:
        raise ValueError("validation scenario identity is invalid")
    _require_sha256(scenario["source_sha256"], "validation scenario source")
    _require_sha256(scenario["runtime_sha256"], "validation scenario runtime")

    contract = value["contract"]
    if not isinstance(contract, Mapping) or set(contract) != {
        "version", "contract_hash", "encoding_hash", "observation_size",
        "action_size", "action_regions",
    }:
        raise ValueError("validation contract identity is invalid")
    if contract["version"] != "tactical-v2":
        raise ValueError("validation contract must be tactical-v2")
    _require_sha256(contract["contract_hash"], "validation contract hash")
    _require_sha256(contract["encoding_hash"], "validation encoding hash")
    if (
        type(contract["observation_size"]) is not int
        or contract["observation_size"] < 1
        or type(contract["action_size"]) is not int
        or contract["action_size"] < 1
        or not isinstance(contract["action_regions"], Mapping)
    ):
        raise ValueError("validation contract geometry is invalid")
    if dataset["encoding_hash"] != contract["encoding_hash"]:
        raise ValueError("validation dataset and contract identities differ")

    isolation = value["seed_isolation"]
    if not isinstance(isolation, Mapping) or set(isolation) != {
        "definition_count", "overlap_count", "final_bank_touched",
    }:
        raise ValueError("validation seed isolation identity is invalid")
    if (
        type(isolation["definition_count"]) is not int
        or isolation["definition_count"] < 1
        or isolation["overlap_count"] != 0
        or isolation["final_bank_touched"] is not False
    ):
        raise ValueError("validation seed isolation failed")
    return value


def _validate_runtime_identity(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"hardware", "software"}:
        raise ValueError("validation runtime identity is invalid")
    hardware = value["hardware"]
    software = value["software"]
    hardware_fields = {
        "training_device", "publication_device", "cuda_available",
        "device_index", "device_name", "cuda_runtime",
    }
    software_fields = {
        "python", "implementation", "platform", "executable", "numpy", "torch",
        "stable_baselines3", "sb3_contrib",
    }
    if (
        not isinstance(hardware, Mapping)
        or set(hardware) != hardware_fields
        or not isinstance(software, Mapping)
        or set(software) != software_fields
        or not isinstance(hardware["training_device"], str)
        or not hardware["training_device"].startswith("cuda")
        or hardware["publication_device"] != "cpu"
        or hardware["cuda_available"] is not True
        or type(hardware["device_index"]) is not int
        or hardware["device_index"] < 0
        or any(
            not isinstance(hardware[name], str) or not hardware[name]
            for name in ("device_name", "cuda_runtime")
        )
        or any(not isinstance(software[name], str) or not software[name] for name in software_fields)
    ):
        raise ValueError("validation runtime identity is invalid")
    _freeze_json(value)
    return value


@dataclass(frozen=True)
class ValidatedStage:
    root: Path
    prepared: PreparedStage
    physical: Mapping[str, Any]
    runtime: Mapping[str, Any]
    identity: Mapping[str, Any]
    content_identity: str


def _open_validated_stage(
    root: Path,
    *,
    prepared: PreparedStage,
    expected_identity: Mapping[str, Any] | None = None,
) -> ValidatedStage:
    manifest_path = Path(root) / "manifest.json"
    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("validation manifest is unreadable") from exc
    if (
        not isinstance(raw, Mapping)
        or set(raw) != _VALIDATION_MANIFEST_FIELDS
        or raw["schema_version"] != 1
        or raw["status"] != "completed"
    ):
        raise ValueError("validation manifest fields are invalid")
    identity = raw["identity"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _VALIDATION_IDENTITY_FIELDS
        or identity["prepared_content_identity"] != prepared.content_identity
    ):
        raise ValueError("validation prepared identity differs")
    physical = _validate_physical_identity(identity["physical"])
    runtime = _validate_runtime_identity(identity["runtime"])
    content_identity = raw["content_identity"]
    if (
        not isinstance(content_identity, str)
        or _content_identity(raw) != content_identity
    ):
        raise ValueError("validation content identity is invalid")
    if expected_identity is not None and not _same_json(identity, expected_identity):
        raise ValueError(
            "completed validation identity differs; use a new output root"
        )
    return ValidatedStage(
        root=Path(root).resolve(strict=True),
        prepared=prepared,
        physical=_freeze_json(physical),
        runtime=_freeze_json(runtime),
        identity=_freeze_json(identity),
        content_identity=content_identity,
    )


def production_physical_validator(
    prepared: PreparedStage,
) -> Mapping[str, Any]:
    """Reopen the accepted panel, learner, and full base-corpus audit."""

    repository_root = Path(
        prepared.identity["repository"]["root"],
    ).resolve(strict=True)
    definition = dagger_domain.load_panel_definition(
        prepared.panel_path,
        repository_root=repository_root,
    )
    dagger_domain.validate_panel_definition(definition)
    audit = dagger_domain.audit_base_dataset(definition)
    if (
        audit.get("content_sha256") != definition.dataset_content_sha256
        or audit.get("file_count") != definition.dataset_file_count
        or audit.get("byte_size") != definition.dataset_byte_size
        or not isinstance(audit.get("audit"), Mapping)
        or audit["audit"].get("games", 0) <= 0
        or audit["audit"].get("teacher_labels", 0) <= 0
        or audit["audit"].get("masked_labels") != 0
        or audit["audit"].get("round_trip_mismatches") != 0
        or audit["audit"].get("replay_mismatches") != 0
    ):
        raise ValueError("production base dataset physical audit failed")
    dagger_domain.validate_seed_definitions()
    ranges = tuple(dagger_domain.SEED_DEFINITIONS)
    overlap_count = sum(
        int(max(left[2], right[2]) <= min(left[3], right[3]))
        for offset, left in enumerate(ranges)
        for right in ranges[offset + 1:]
    )
    final_bank_touched = any(
        max(start, 17_000_000) <= min(stop, 17_000_249)
        for _partition, _iteration, start, stop in ranges
    )
    source = definition.starting_learner
    return {
        "starting_learner": {
            "checkpoint_path": source.controller["path"],
            "checkpoint_sha256": source.checkpoint_sha256,
            "source_run": source.controller["source_run"],
            "source_manifest_sha256": (
                definition.learner_source_manifest_sha256
            ),
        },
        "base_dataset": {
            "root": str(definition.dataset_root),
            "manifest_sha256": definition.dataset_manifest_sha256,
            "content_sha256": definition.dataset_content_sha256,
            "file_count": definition.dataset_file_count,
            "byte_size": definition.dataset_byte_size,
            "contract_hash": definition.dataset_contract_hash,
            "encoding_hash": definition.dataset_encoding_hash,
            "scenario_hash": definition.dataset_scenario_hash,
        },
        "scenario": {
            "source_sha256": definition.scenario_sha256,
            "runtime_sha256": definition.runtime_scenario_sha256,
        },
        "contract": {
            "version": "tactical-v2",
            "contract_hash": definition.contract_hash,
            "encoding_hash": definition.encoding_hash,
            "observation_size": definition.observation_size,
            "action_size": definition.action_size,
            "action_regions": {
                name: dict(region)
                for name, region in definition.action_regions.items()
            },
        },
        "seed_isolation": {
            "definition_count": len(definition.seed_banks),
            "overlap_count": overlap_count,
            "final_bank_touched": final_bank_touched,
        },
    }


def production_runtime_probe() -> Mapping[str, Any]:
    """Resolve the locked CUDA-training and CPU-publication runtime."""

    import importlib.metadata
    import platform
    import sys

    import numpy as np

    from ml_lab.imitation import resolve_behavioral_cloning_device

    hardware = resolve_behavioral_cloning_device("cuda")

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "hardware": {
            "training_device": hardware["resolved"],
            "publication_device": "cpu",
            "cuda_available": True,
            "device_index": hardware["device_index"],
            "device_name": hardware["device_name"],
            "cuda_runtime": hardware["cuda_runtime"],
        },
        "software": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
            "executable": sys.executable,
            "numpy": np.__version__,
            "torch": hardware["torch_version"],
            "stable_baselines3": version("stable-baselines3"),
            "sb3_contrib": version("sb3-contrib"),
        },
    }


def run_validate(
    *,
    output_root: Path,
    physical_validator: Callable[
        [PreparedStage], Mapping[str, Any]
    ] = production_physical_validator,
    runtime_probe: Callable[
        [], Mapping[str, Any]
    ] = production_runtime_probe,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> ValidatedStage:
    """Physically validate frozen learner/data/runtime inputs without games."""

    root = Path(output_root)
    prepared = _open_prepared_stage(root / "definition")
    recorded_repository = prepared.identity["repository"]
    repository_root = Path(recorded_repository["root"])

    def require_stable_repository() -> None:
        current = _validated_repository_identity(
            repository_root, repository_identity_provider,
        )
        if not _same_json(current, recorded_repository):
            raise ValueError("validation repository identity changed")

    require_stable_repository()
    physical = _validate_physical_identity(physical_validator(prepared))
    runtime = _validate_runtime_identity(runtime_probe())
    identity = {
        "prepared_content_identity": prepared.content_identity,
        "physical": physical,
        "runtime": runtime,
    }
    destination = root / "validation"
    staging = root / "validation.staging"
    if destination.exists():
        if staging.exists():
            raise ValueError("validation destination and staging are ambiguous")
        require_stable_repository()
        return _open_validated_stage(
            destination,
            prepared=prepared,
            expected_identity=identity,
        )
    if staging.exists():
        raise ValueError("validation staging is partial; use a new output root")

    staging.mkdir()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "identity": identity,
    }
    manifest["content_identity"] = _content_identity(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    candidate = _open_validated_stage(
        staging, prepared=prepared, expected_identity=identity,
    )
    require_stable_repository()
    os.replace(staging, destination)
    published = _open_validated_stage(
        destination, prepared=prepared, expected_identity=identity,
    )
    if published.content_identity != candidate.content_identity:
        raise ValueError("validation changed during publication")
    return published


def run_sealed_oracle_preflight(
    *,
    definition: object,
    output_root: Path,
    execution_session_factory: Callable[..., object] | None,
    repository_root: Path = _REPOSITORY_ROOT,
) -> object:
    """Invoke only the public engine-sealed preflight boundary, or fail closed."""

    evidence_root = Path(output_root).resolve()
    if evidence_root.is_relative_to(Path(repository_root).resolve(strict=True)):
        raise ValueError("oracle preflight evidence root must be outside the repository")
    if execution_session_factory is None:
        raise RuntimeError(
            "production preflight requires a sealed engine execution-session factory"
        )
    session = execution_session_factory(
        definition=definition,
        output_root=evidence_root,
    )
    if session is None:
        raise RuntimeError(
            "sealed engine execution-session factory returned no session"
        )
    return dagger_domain.run_oracle_preflight(
        definition,
        output_root=evidence_root,
        execution_session=session,
    )


@dataclass(frozen=True)
class DaggerDependencies:
    """Injected stage boundaries used by the orchestration layer."""

    prepare: Callable[..., object]
    validate: Callable[..., object]
    preflight: Callable[..., object]
    baseline: Callable[..., object]
    resolve_incoming: Callable[..., object]
    collect: Callable[..., object]
    build_corpus: Callable[..., object]
    train: Callable[..., object]
    reopen_actor: Callable[..., object]
    load_iteration_context: Callable[..., Mapping[str, Any]]
    reopen_overlay: Callable[..., object]
    build_iteration_identity: Callable[..., Mapping[str, Any]]
    build_iteration_manifest: Callable[..., Mapping[str, Any]]
    repository_identity_provider: Callable[[Path], Mapping[str, Any]]
    clock: Callable[[], float] = time.monotonic


_ITERATION_CONTEXT_FIELDS = frozenset({
    "validated", "preflight", "preceding_actor", "train_overlays",
    "validation_overlays", "repository",
})


def _require_iteration_boundary(
    value: Callable[..., object] | None, label: str,
) -> Callable[..., object]:
    if value is None or not callable(value):
        raise RuntimeError(f"selective-DAgger {label} boundary is unavailable")
    return value


def _iteration_context(
    index: int,
    *,
    output_root: Path,
    dependencies: DaggerDependencies,
) -> Mapping[str, Any]:
    loader = _require_iteration_boundary(
        dependencies.load_iteration_context, "iteration context",
    )
    context = loader(index=index, output_root=output_root)
    if not isinstance(context, Mapping) or set(context) != _ITERATION_CONTEXT_FIELDS:
        raise ValueError("selective-DAgger iteration context fields are invalid")
    for field in ("train_overlays", "validation_overlays"):
        overlays = context[field]
        if (
            not isinstance(overlays, (tuple, list))
            or len(overlays) != index - 1
        ):
            raise ValueError(
                f"selective-DAgger {field} must contain iterations before {index}"
            )
    return context


def _reopen_iteration_overlay(
    *,
    partition: str,
    index: int,
    output_root: Path,
    expected: object | None,
    dependencies: DaggerDependencies,
) -> object:
    opener = _require_iteration_boundary(
        dependencies.reopen_overlay, "overlay reopen",
    )
    return opener(
        partition=partition,
        index=index,
        output_root=output_root,
        expected=expected,
    )


def _build_iteration_identity(
    *,
    index: int,
    context: Mapping[str, Any],
    incoming: object,
    train_overlays: tuple[object, ...],
    validation_overlays: tuple[object, ...],
    dependencies: DaggerDependencies,
) -> Mapping[str, Any]:
    builder = _require_iteration_boundary(
        dependencies.build_iteration_identity, "manifest identity",
    )
    identity = builder(
        index=index,
        context=context,
        incoming=incoming,
        train_overlays=train_overlays,
        validation_overlays=validation_overlays,
    )
    if not isinstance(identity, Mapping):
        raise ValueError("selective-DAgger iteration identity must be an object")
    return identity


def _build_iteration_manifest(
    *,
    index: int,
    identity: Mapping[str, Any],
    train_overlay: object,
    validation_overlay: object,
    actor: object,
    timings: Mapping[str, Any],
    dependencies: DaggerDependencies,
) -> IterationManifest:
    builder = _require_iteration_boundary(
        dependencies.build_iteration_manifest, "manifest construction",
    )
    payload = builder(
        index=index,
        identity=identity,
        train_overlay=train_overlay,
        validation_overlay=validation_overlay,
        actor=actor,
        timings=timings,
    )
    manifest = IterationManifest.from_dict(payload)
    expected_collections = {
        "train_collection": _iteration_overlay_collection_metrics(train_overlay),
        "validation_collection": _iteration_overlay_collection_metrics(
            validation_overlay
        ),
    }
    if any(
        not _same_json(manifest.metrics[name], expected)
        for name, expected in expected_collections.items()
    ):
        raise ValueError(
            "selective-DAgger iteration collection metrics do not match "
            "reopened overlay evidence"
        )
    return manifest


def _iteration_overlay_collection_metrics(overlay: object) -> Mapping[str, Any]:
    if isinstance(overlay, dagger_domain.DaggerOverlay):
        metrics = dagger_domain.dagger_overlay_collection_metrics(overlay)
    elif isinstance(overlay, Mapping):
        metrics = overlay.get("collection_metrics")
    else:
        metrics = getattr(overlay, "collection_metrics", None)
    if not isinstance(metrics, Mapping):
        raise ValueError(
            "selective-DAgger reopened overlay collection metrics are unavailable"
        )
    return metrics


def _iteration_overlay_row_count(overlay: object) -> int:
    value = (
        overlay.get("row_count")
        if isinstance(overlay, Mapping)
        else getattr(overlay, "row_count", None)
    )
    if type(value) is not int or value < 1:
        raise ValueError("selective-DAgger overlay row count is invalid")
    return value


def _read_iteration_manifest(path: Path) -> IterationManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"selective-DAgger iteration manifest is unreadable: {path}") from exc
    return IterationManifest.from_dict(payload)


def _require_iteration_repository(
    expected: Mapping[str, Any], dependencies: DaggerDependencies,
) -> None:
    try:
        recorded = _recorded_repository_identity(expected)
        current = _validated_repository_identity(
            Path(recorded["root"]), dependencies.repository_identity_provider,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "selective-DAgger iteration repository identity changed"
        ) from exc
    if not _same_json(current, recorded):
        raise ValueError("selective-DAgger iteration repository identity changed")


def _validated_iteration_physical(validated: object) -> Mapping[str, Any]:
    physical = getattr(validated, "physical", None)
    if physical is None and isinstance(validated, Mapping):
        physical = validated.get("physical", validated)
    if not isinstance(physical, Mapping):
        raise ValueError("selective-DAgger validated physical identity is unavailable")
    return physical


def _iteration_contract_identity(validated: object) -> Mapping[str, Any]:
    physical = _validated_iteration_physical(validated)
    contract = physical.get("contract")
    required = {
        "version", "contract_hash", "encoding_hash", "observation_size",
        "action_size", "action_regions",
    }
    if not isinstance(contract, Mapping) or set(contract) != required:
        raise ValueError("selective-DAgger validated contract identity is invalid")
    return contract


def _actor_publication_contract(actor_root: Path) -> EnvironmentContract:
    try:
        run = json.loads((actor_root / "run.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("selective-DAgger actor run manifest is unreadable") from exc
    raw = run.get("contract") if isinstance(run, Mapping) else None
    fields = {
        "environment", "version", "contract_hash", "encoding_hash",
        "observation_size", "action_size", "board", "roster", "reward",
        "semantics",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != fields
        or raw["environment"] != raw["version"]
        or not isinstance(raw["board"], Mapping)
        or type(raw["roster"]) is not list
        or not isinstance(raw["reward"], Mapping)
        or not isinstance(raw["semantics"], Mapping)
    ):
        raise ValueError("selective-DAgger actor contract is invalid")
    try:
        return EnvironmentContract(
            version=raw["version"],
            contract_hash=raw["contract_hash"],
            encoding_hash=raw["encoding_hash"],
            observation_size=raw["observation_size"],
            action_size=raw["action_size"],
            board=dict(raw["board"]),
            roster=list(raw["roster"]),
            reward=dict(raw["reward"]),
            semantics=dict(raw["semantics"]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("selective-DAgger actor contract is invalid") from exc


def _critical_contract(contract: EnvironmentContract) -> dict[str, Any]:
    action_regions = contract.semantics.get("action_regions")
    return {
        "version": contract.version,
        "contract_hash": contract.contract_hash,
        "encoding_hash": contract.encoding_hash,
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
        "action_regions": action_regions,
    }


def _actor_root(value: object) -> Path:
    root = value.get("root") if isinstance(value, Mapping) else getattr(value, "root", None)
    if not isinstance(root, (str, os.PathLike)):
        raise ValueError("selective-DAgger preceding actor root is invalid")
    try:
        return Path(root).resolve(strict=True)
    except OSError as exc:
        raise ValueError("selective-DAgger preceding actor root is invalid") from exc


def _authenticate_iteration_incoming(
    index: int,
    *,
    output_root: Path,
    context: Mapping[str, Any],
    dependencies: DaggerDependencies,
) -> object:
    validated = context["validated"]
    physical = _validated_iteration_physical(validated)
    contract_identity = _iteration_contract_identity(validated)
    preceding = context["preceding_actor"]

    if index == 1:
        if preceding is not None:
            raise ValueError("selective-DAgger iteration one has a preceding actor")
        source = dagger_domain.dagger_actor_source(1)
        source_run = Path(source.controller["source_run"]).resolve(strict=True)
        run_manifest = source_run / "run.json"
        starting = physical.get("starting_learner")
        expected_identity = {
            "checkpoint_path": source.controller["path"],
            "checkpoint_sha256": source.checkpoint_sha256,
            "source_run": source.controller["source_run"],
            "source_manifest_sha256": _sha256_bytes(run_manifest.read_bytes()),
        }
        if not isinstance(starting, Mapping) or not _same_json(starting, expected_identity):
            raise ValueError(
                "selective-DAgger iteration one physical learner identity changed"
            )
        incoming = dependencies.resolve_incoming(
            index=index, validated=validated, preceding_actor=None,
        )
        expected_incoming = {
            "source": source.to_dict(),
            "identity": expected_identity,
            "published_actor_sha256": source.checkpoint_sha256,
        }
        if not _same_json(incoming, expected_incoming):
            raise ValueError(
                "selective-DAgger iteration one incoming learner is unauthenticated"
            )
        source_contract = _actor_publication_contract(source_run)
        if not _same_json(_critical_contract(source_contract), contract_identity):
            raise ValueError(
                "selective-DAgger iteration one contract identity changed"
            )
        return incoming

    root = Path(output_root).resolve(strict=True)
    canonical_actor = (
        root / "iterations" / f"iteration-{index - 1}" / "actor"
    ).resolve(strict=True)
    if not canonical_actor.is_relative_to(root):
        raise ValueError("selective-DAgger canonical preceding actor escaped output root")
    if preceding is None or _actor_root(preceding) != canonical_actor:
        raise ValueError("selective-DAgger canonical preceding actor is required")

    prior_manifest = _read_iteration_manifest(canonical_actor.parent / "manifest.json")
    if prior_manifest.iteration != index - 1:
        raise ValueError(
            "selective-DAgger canonical predecessor iteration does not match"
        )
    if not _same_json(prior_manifest.identity["repository"], context["repository"]):
        raise ValueError("selective-DAgger preceding actor repository identity changed")
    if not _same_json(prior_manifest.identity["contract"], contract_identity):
        raise ValueError("selective-DAgger preceding actor contract identity changed")

    run_path = canonical_actor / "run.json"
    bc_path = canonical_actor / "bc.json"
    run_sha256_before = _sha256_bytes(run_path.read_bytes())
    bc_sha256_before = _sha256_bytes(bc_path.read_bytes())
    source = dagger_domain.dagger_actor_source(
        index, preceding_run=canonical_actor,
    )
    actor_contract = _actor_publication_contract(canonical_actor)
    if not _same_json(_critical_contract(actor_contract), contract_identity):
        raise ValueError("selective-DAgger preceding actor contract identity changed")
    verification = imitation_domain.validate_actor_supervision_publication(
        canonical_actor, actor_contract,
    )
    run_sha256_after = _sha256_bytes(run_path.read_bytes())
    bc_sha256_after = _sha256_bytes(bc_path.read_bytes())
    if (
        run_sha256_after != run_sha256_before
        or bc_sha256_after != bc_sha256_before
    ):
        raise ValueError("selective-DAgger preceding actor changed during validation")

    expected_verification = {
        "checkpoint_sha256": source.checkpoint_sha256,
        "actor_sha256": source.published_actor_sha256,
        "publication_metadata_sha256": prior_manifest.artifacts["actor"][
            "publication_metadata_sha256"
        ],
        "contract_hash": actor_contract.contract_hash,
        "encoding_hash": actor_contract.encoding_hash,
        "observation_size": actor_contract.observation_size,
        "action_size": actor_contract.action_size,
    }
    if any(verification.get(key) != value for key, value in expected_verification.items()):
        raise ValueError("selective-DAgger preceding actor publication is unauthenticated")

    expected_actor = {
        "checkpoint_sha256": source.checkpoint_sha256,
        "actor_sha256": source.published_actor_sha256,
        "publication_metadata_sha256": verification["publication_metadata_sha256"],
        "run_manifest_sha256": run_sha256_after,
        "bc_manifest_sha256": bc_sha256_after,
    }
    artifact = prior_manifest.artifacts["actor"]
    if any(artifact.get(key) != value for key, value in expected_actor.items()):
        raise ValueError("selective-DAgger preceding actor does not match its manifest")

    reopened_actor = dependencies.reopen_actor(
        index=index - 1, trained=canonical_actor,
    )
    if _actor_root(reopened_actor) != canonical_actor or any(
        reopened_actor.get(key) != value for key, value in expected_actor.items()
    ):
        raise ValueError("selective-DAgger preceding actor reopen changed")
    expected_incoming = {
        "source": source.to_dict(),
        "identity": {
            "checkpoint_path": source.controller["path"],
            "checkpoint_sha256": source.checkpoint_sha256,
            "source_run": source.controller["source_run"],
            "source_manifest_sha256": run_sha256_after,
        },
        "published_actor_sha256": source.published_actor_sha256,
    }
    incoming = dependencies.resolve_incoming(
        index=index, validated=validated, preceding_actor=reopened_actor,
    )
    if not _same_json(incoming, expected_incoming):
        raise ValueError("selective-DAgger preceding actor ownership changed")
    return incoming


def _require_iteration_identity_repository(
    identity: Mapping[str, Any], expected: Mapping[str, Any],
) -> None:
    repository = identity.get("repository")
    if not isinstance(repository, Mapping) or not _same_json(repository, expected):
        raise ValueError("selective-DAgger iteration manifest repository identity changed")


_PREDECESSOR_STABLE_IDENTITY_FIELDS = (
    "definition", "repository", "scenario", "contract", "base_dataset",
    "selected_oracle", "optimizer", "runtime",
)


def _require_iteration_predecessor_chain(
    identity: Mapping[str, Any],
    *,
    index: int,
    predecessor: IterationManifest | None,
) -> None:
    if "predecessor" not in identity:
        raise ValueError("selective-DAgger iteration predecessor identity is missing")
    reference = identity["predecessor"]
    if index == 1:
        if predecessor is not None or reference is not None:
            raise ValueError("selective-DAgger iteration one predecessor identity is invalid")
        return
    if predecessor is None:
        raise ValueError("selective-DAgger physical predecessor is unavailable")
    expected_reference = {
        "iteration": index - 1,
        "content_identity": predecessor.content_identity,
    }
    if not _same_json(reference, expected_reference):
        raise ValueError("selective-DAgger predecessor content identity changed")
    for field in _PREDECESSOR_STABLE_IDENTITY_FIELDS:
        if not _same_json(identity.get(field), predecessor.identity[field]):
            raise ValueError(
                f"selective-DAgger predecessor causal identity changed: {field}"
            )
    for field in (
        "cumulative_train_overlays", "cumulative_validation_overlays",
    ):
        current = identity.get(field)
        if (
            not isinstance(current, (list, tuple))
            or len(current) != index
            or not _same_json(current[:-1], predecessor.identity[field])
        ):
            raise ValueError(
                f"selective-DAgger predecessor overlay chain changed: {field}"
            )


def run_iteration(
    index: int,
    *,
    output_root: Path,
    dependencies: DaggerDependencies,
) -> IterationManifest:
    """Run or physically reopen one immutable selective-DAgger iteration."""

    if type(index) is not int or index not in {1, 2, 3}:
        raise ValueError("selective-DAgger iteration index must be 1, 2, or 3")
    root = Path(output_root)
    iterations_root = root / "iterations"
    iteration_root = iterations_root / f"iteration-{index}"
    staging = iterations_root / f"iteration-{index}.staging"
    manifest_path = iteration_root / "manifest.json"

    if iteration_root.exists() and not manifest_path.is_file():
        raise ValueError(
            "selective-DAgger iteration is partial; use a new output root"
        )
    completed = _read_iteration_manifest(manifest_path) if manifest_path.is_file() else None
    if completed is not None and completed.iteration != index:
        raise ValueError(
            "selective-DAgger canonical predecessor iteration does not match"
        )
    if completed is not None and staging.exists():
        raise ValueError(
            "completed selective-DAgger iteration has ambiguous staging"
        )
    if completed is None and staging.exists():
        raise ValueError(
            "selective-DAgger iteration staging is partial; use a new output root"
        )

    predecessor: IterationManifest | None = None
    if index > 1:
        predecessor_path = (
            iterations_root / f"iteration-{index - 1}" / "manifest.json"
        )
        if not predecessor_path.is_file():
            raise ValueError(
                "selective-DAgger preceding iteration is incomplete"
            )
        predecessor = run_iteration(
            index - 1,
            output_root=root,
            dependencies=dependencies,
        )

    context = _iteration_context(
        index, output_root=root, dependencies=dependencies,
    )
    expected_repository = _recorded_repository_identity(context["repository"])
    _require_iteration_repository(expected_repository, dependencies)
    incoming = _authenticate_iteration_incoming(
        index,
        output_root=root,
        context=context,
        dependencies=dependencies,
    )
    _require_iteration_repository(expected_repository, dependencies)
    prior_train = tuple(context["train_overlays"])
    prior_validation = tuple(context["validation_overlays"])

    if completed is not None:
        validation_overlay = _reopen_iteration_overlay(
            partition="validation",
            index=index,
            output_root=iteration_root / "validation-overlay",
            expected=None,
            dependencies=dependencies,
        )
        train_overlay = _reopen_iteration_overlay(
            partition="train",
            index=index,
            output_root=iteration_root / "train-overlay",
            expected=None,
            dependencies=dependencies,
        )
        actor = dependencies.reopen_actor(
            index=index, trained=iteration_root / "actor",
        )
        identity = _build_iteration_identity(
            index=index,
            context=context,
            incoming=incoming,
            train_overlays=(*prior_train, train_overlay),
            validation_overlays=(*prior_validation, validation_overlay),
            dependencies=dependencies,
        )
        _require_iteration_identity_repository(identity, expected_repository)
        _require_iteration_predecessor_chain(
            identity, index=index, predecessor=predecessor,
        )
        completed.require_identity(identity)
        reconstructed = _build_iteration_manifest(
            index=index,
            identity=identity,
            train_overlay=train_overlay,
            validation_overlay=validation_overlay,
            actor=actor,
            timings=completed.timings,
            dependencies=dependencies,
        )
        if reconstructed.content_identity != completed.content_identity:
            raise ValueError(
                "selective-DAgger iteration physical children do not match manifest"
            )
        _require_iteration_repository(expected_repository, dependencies)
        return completed

    iterations_root.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    started = dependencies.clock()
    validation_started = dependencies.clock()
    validation_collected = dependencies.collect(
        partition="validation",
        index=index,
        learner=incoming,
        oracle=context["preflight"]["selected_oracle"],
        output_root=staging / "validation-overlay",
    )
    _require_iteration_repository(expected_repository, dependencies)
    validation_seconds = max(0.0, dependencies.clock() - validation_started)
    train_started = dependencies.clock()
    train_collected = dependencies.collect(
        partition="train",
        index=index,
        learner=incoming,
        oracle=context["preflight"]["selected_oracle"],
        output_root=staging / "train-overlay",
    )
    _require_iteration_repository(expected_repository, dependencies)
    train_seconds = max(0.0, dependencies.clock() - train_started)
    validation_overlay = _reopen_iteration_overlay(
        partition="validation",
        index=index,
        output_root=staging / "validation-overlay",
        expected=validation_collected,
        dependencies=dependencies,
    )
    train_overlay = _reopen_iteration_overlay(
        partition="train",
        index=index,
        output_root=staging / "train-overlay",
        expected=train_collected,
        dependencies=dependencies,
    )
    cumulative_train = (*prior_train, train_overlay)
    cumulative_validation = (*prior_validation, validation_overlay)
    corpus_started = dependencies.clock()
    corpus = dependencies.build_corpus(
        index=index,
        train_overlays=cumulative_train,
        validation_overlays=cumulative_validation,
    )
    _require_iteration_repository(expected_repository, dependencies)
    corpus_seconds = max(0.0, dependencies.clock() - corpus_started)
    training_started = dependencies.clock()
    trained = dependencies.train(
        index=index,
        learner=incoming,
        corpus=corpus,
        output_root=staging / "actor",
    )
    _require_iteration_repository(expected_repository, dependencies)
    training_seconds = max(0.0, dependencies.clock() - training_started)
    dependencies.reopen_actor(index=index, trained=trained)
    _require_iteration_repository(expected_repository, dependencies)

    publication_started = dependencies.clock()
    validation_overlay = _reopen_iteration_overlay(
        partition="validation",
        index=index,
        output_root=staging / "validation-overlay",
        expected=validation_collected,
        dependencies=dependencies,
    )
    train_overlay = _reopen_iteration_overlay(
        partition="train",
        index=index,
        output_root=staging / "train-overlay",
        expected=train_collected,
        dependencies=dependencies,
    )
    actor = dependencies.reopen_actor(
        index=index, trained=staging / "actor",
    )
    _require_iteration_repository(expected_repository, dependencies)
    publication_seconds = max(0.0, dependencies.clock() - publication_started)
    identity = _build_iteration_identity(
        index=index,
        context=context,
        incoming=incoming,
        train_overlays=(*prior_train, train_overlay),
        validation_overlays=(*prior_validation, validation_overlay),
        dependencies=dependencies,
    )
    _require_iteration_identity_repository(identity, expected_repository)
    _require_iteration_predecessor_chain(
        identity, index=index, predecessor=predecessor,
    )
    elapsed = max(0.0, dependencies.clock() - started)
    timings = {
        "elapsed_seconds": elapsed,
        "validation_collection_seconds": validation_seconds,
        "train_collection_seconds": train_seconds,
        "corpus_seconds": corpus_seconds,
        "training_seconds": training_seconds,
        "publication_seconds": publication_seconds,
        "train_labels_per_second": (
            _iteration_overlay_row_count(train_overlay) / train_seconds
            if train_seconds > 0.0
            else 0.0
        ),
        "validation_labels_per_second": (
            _iteration_overlay_row_count(validation_overlay)
            / validation_seconds
            if validation_seconds > 0.0
            else 0.0
        ),
    }
    manifest = _build_iteration_manifest(
        index=index,
        identity=identity,
        train_overlay=train_overlay,
        validation_overlay=validation_overlay,
        actor=actor,
        timings=timings,
        dependencies=dependencies,
    )
    _require_iteration_repository(expected_repository, dependencies)
    manifest.require_identity(identity)
    staging_manifest = staging / "manifest.json"
    atomic_write_json(staging_manifest, manifest.to_dict())
    staged = _read_iteration_manifest(staging_manifest)
    if staged.content_identity != manifest.content_identity:
        raise ValueError(
            "selective-DAgger staged iteration manifest changed"
        )
    _require_iteration_repository(expected_repository, dependencies)
    os.replace(staging, iteration_root)

    reopened = _read_iteration_manifest(manifest_path)
    validation_overlay = _reopen_iteration_overlay(
        partition="validation",
        index=index,
        output_root=iteration_root / "validation-overlay",
        expected=validation_collected,
        dependencies=dependencies,
    )
    train_overlay = _reopen_iteration_overlay(
        partition="train",
        index=index,
        output_root=iteration_root / "train-overlay",
        expected=train_collected,
        dependencies=dependencies,
    )
    actor = dependencies.reopen_actor(
        index=index, trained=iteration_root / "actor",
    )
    published_identity = _build_iteration_identity(
        index=index,
        context=context,
        incoming=incoming,
        train_overlays=(*prior_train, train_overlay),
        validation_overlays=(*prior_validation, validation_overlay),
        dependencies=dependencies,
    )
    _require_iteration_identity_repository(published_identity, expected_repository)
    _require_iteration_predecessor_chain(
        published_identity, index=index, predecessor=predecessor,
    )
    reopened.require_identity(published_identity)
    reconstructed = _build_iteration_manifest(
        index=index,
        identity=published_identity,
        train_overlay=train_overlay,
        validation_overlay=validation_overlay,
        actor=actor,
        timings=reopened.timings,
        dependencies=dependencies,
    )
    if reconstructed.content_identity != reopened.content_identity:
        raise ValueError(
            "selective-DAgger published iteration changed during publication"
        )
    _require_iteration_repository(expected_repository, dependencies)
    return reopened


def run_training_pipeline(
    *,
    output_root: Path,
    dependencies: DaggerDependencies,
) -> tuple[IterationManifest, ...]:
    """Execute the fixed three-iteration stage order through injected boundaries."""

    root = Path(output_root)
    prepared = dependencies.prepare(output_root=root)
    validated = dependencies.validate(output_root=root, prepared=prepared)
    preflight = dependencies.preflight(output_root=root, validated=validated)
    dependencies.baseline(
        output_root=root,
        validated=validated,
        preflight=preflight,
    )
    return tuple(
        run_iteration(
            index,
            output_root=root,
            dependencies=dependencies,
        )
        for index in (1, 2, 3)
    )


# Task 10: immutable reciprocal development evaluation and publication.


def _mutable_stage_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _mutable_stage_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_mutable_stage_json(item) for item in value]
    return value


def _canonical_stage_bytes(value: Any) -> bytes:
    return json.dumps(
        _mutable_stage_json(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _development_schedule() -> checkpoint_audit_domain.AuditSchedule:
    return checkpoint_audit_domain.AuditSchedule(
        seed_start=20_000_000,
        maps=100,
        both_seats=True,
        profile="standard-3v3",
        opponent="random",
    )


def _development_definition_identity(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
) -> Mapping[str, Any]:
    if not isinstance(definition, dagger_domain.DevelopmentEvaluationDefinition):
        raise ValueError("development evaluation definition type is invalid")
    return definition.to_dict()


def _require_development_repository(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    provider: Callable[[Path], Mapping[str, Any]],
) -> Mapping[str, Any]:
    actual = _validated_repository_identity(
        Path(definition.repository["root"]), provider,
    )
    if not _same_json(actual, definition.repository):
        raise ValueError("development evaluation repository identity changed")
    return actual


def _require_development_candidate_inputs(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    candidate: dagger_domain.DevelopmentCandidate,
    provider: Callable[[Path], Mapping[str, Any]],
) -> None:
    try:
        checkpoint_sha256 = _sha256_bytes(Path(candidate.checkpoint_path).read_bytes())
    except OSError as exc:
        raise ValueError("development candidate checkpoint is missing") from exc
    if checkpoint_sha256 != candidate.checkpoint_sha256:
        raise ValueError("development candidate checkpoint identity changed")
    _require_development_repository(definition, provider)


def _retained_artifact_payload(
    artifact: checkpoint_audit_domain.RetainedArtifactIdentity,
) -> Mapping[str, Any]:
    if not isinstance(
        artifact, checkpoint_audit_domain.RetainedArtifactIdentity,
    ):
        raise ValueError("development retained artifact identity type is invalid")
    return {
        "trace_path": artifact.trace_path,
        "trace_sha256": artifact.trace_sha256,
        "trace_byte_size": artifact.trace_byte_size,
        "replay_path": artifact.replay_path,
        "replay_sha256": artifact.replay_sha256,
        "replay_byte_size": artifact.replay_byte_size,
    }


@dataclass(frozen=True)
class DevelopmentCandidateEvidence:
    root: Path
    candidate_id: str
    controller: str
    checkpoint_sha256: str
    controller_identity: Mapping[str, Any]
    content_identity: str
    matches: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True)
class DevelopmentCandidateRun:
    new_games: int
    reused: bool
    result: DevelopmentCandidateEvidence


_DEVELOPMENT_EVALUATION_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "identity", "physical", "content_identity",
})
_DEVELOPMENT_EVALUATION_IDENTITY_FIELDS = frozenset({
    "definition", "candidate",
})
_DEVELOPMENT_EVALUATION_PHYSICAL_FIELDS = frozenset({
    "evaluation", "artifacts", "rows_sha256",
})


def _development_candidate_identity(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    candidate: dagger_domain.DevelopmentCandidate,
) -> Mapping[str, Any]:
    if (
        not isinstance(candidate, dagger_domain.DevelopmentCandidate)
        or candidate not in definition.candidates
    ):
        raise ValueError("development candidate is not in the frozen definition")
    return {
        "definition": _development_definition_identity(definition),
        "candidate": candidate.to_dict(),
    }


def _open_development_candidate_evaluation(
    root: Path,
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    candidate: dagger_domain.DevelopmentCandidate,
    validate_candidate: Callable[..., object] | None = None,
) -> DevelopmentCandidateEvidence:
    # Retained evidence is always authenticated by the first-party opener.  The
    # compatibility argument is intentionally non-authoritative.
    del validate_candidate
    candidate_root = Path(root).resolve(strict=True)
    manifest_path = candidate_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("development evaluation manifest is unreadable") from exc
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _DEVELOPMENT_EVALUATION_MANIFEST_FIELDS
        or manifest["schema_version"] != 1
        or manifest["status"] != "completed"
    ):
        raise ValueError("development evaluation manifest fields are invalid")
    identity = manifest["identity"]
    if (
        not isinstance(identity, Mapping)
        or set(identity) != _DEVELOPMENT_EVALUATION_IDENTITY_FIELDS
        or not _same_json(
            identity, _development_candidate_identity(definition, candidate),
        )
    ):
        raise ValueError("development evaluation identity changed")
    if (
        not isinstance(manifest["content_identity"], str)
        or _content_identity(manifest) != manifest["content_identity"]
    ):
        raise ValueError("development evaluation content identity changed")
    physical = manifest["physical"]
    if (
        not isinstance(physical, Mapping)
        or set(physical) != _DEVELOPMENT_EVALUATION_PHYSICAL_FIELDS
    ):
        raise ValueError("development evaluation physical fields are invalid")
    descriptor = physical["evaluation"]
    if (
        not isinstance(descriptor, Mapping)
        or set(descriptor) != {"path", "sha256", "byte_size"}
        or descriptor["path"] != "physical/evaluation.json"
    ):
        raise ValueError("development evaluation descriptor is invalid")
    evaluation_path = candidate_root / descriptor["path"]
    try:
        evaluation_bytes = evaluation_path.read_bytes()
    except OSError as exc:
        raise ValueError("development physical evaluation is missing") from exc
    if (
        type(descriptor["byte_size"]) is not int
        or descriptor["byte_size"] < 1
        or len(evaluation_bytes) != descriptor["byte_size"]
        or _sha256_bytes(evaluation_bytes)
        != _require_sha256(descriptor["sha256"], "development evaluation sha256")
    ):
        raise ValueError("development physical evaluation bytes changed")
    retained = checkpoint_audit_domain.validate_retained_evaluation(
        evaluation_path,
        publication_root=candidate_root / "physical",
        evidence_root=candidate_root / "physical" / "evidence",
        schedule=_development_schedule(),
        expected_candidate_identity=candidate.controller_identity,
    )
    if not isinstance(retained, checkpoint_audit_domain.RetainedEvaluation):
        raise ValueError("development validator returned an invalid retained evaluation")
    artifacts = physical["artifacts"]
    expected_artifacts = [
        _retained_artifact_payload(artifact) for artifact in retained.artifacts
    ]
    if not _same_json(artifacts, expected_artifacts):
        raise ValueError("development retained artifact identities changed")
    expected_rows_sha256 = _sha256_bytes(
        _canonical_stage_bytes(retained.matches)
    )
    if physical["rows_sha256"] != expected_rows_sha256:
        raise ValueError("development canonical match rows changed")
    physical_root = candidate_root / "physical"
    artifact_snapshots: list[tuple[Path, bytes]] = []
    for artifact in retained.artifacts:
        for path_text, sha256, byte_size in (
            (artifact.trace_path, artifact.trace_sha256, artifact.trace_byte_size),
            (artifact.replay_path, artifact.replay_sha256, artifact.replay_byte_size),
        ):
            artifact_path = (physical_root / path_text).resolve(strict=True)
            supplied_artifact = physical_root / path_text
            junction = getattr(supplied_artifact, "is_junction", None)
            if (
                supplied_artifact.is_symlink()
                or bool(junction is not None and junction())
                or not artifact_path.is_relative_to(physical_root)
            ):
                raise ValueError("development retained artifact escaped its root")
            artifact_bytes = artifact_path.read_bytes()
            artifact_snapshots.append((artifact_path, artifact_bytes))
            if (
                len(artifact_bytes) != byte_size
                or _sha256_bytes(artifact_bytes) != sha256
            ):
                raise ValueError("development retained artifact bytes changed")
    actual = {
        path.relative_to(candidate_root).as_posix()
        for path in candidate_root.iterdir()
    }
    if actual != {"manifest.json", "physical"}:
        raise ValueError("development candidate directory contains unowned evidence")
    if (
        manifest_path.read_bytes() != manifest_bytes
        or evaluation_path.read_bytes() != evaluation_bytes
        or any(path.read_bytes() != raw for path, raw in artifact_snapshots)
    ):
        raise ValueError("development candidate evidence changed while reopening")
    return DevelopmentCandidateEvidence(
        root=candidate_root,
        candidate_id=candidate.candidate_id,
        controller=candidate.controller,
        checkpoint_sha256=candidate.checkpoint_sha256,
        controller_identity=_freeze_json(candidate.controller_identity),
        content_identity=manifest["content_identity"],
        matches=tuple(_freeze_json(row) for row in retained.matches),
    )


def reopen_development_candidate_evaluation(
    root: Path,
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    candidate: dagger_domain.DevelopmentCandidate,
    validate_candidate: Callable[..., object] = (
        checkpoint_audit_domain.validate_retained_evaluation
    ),
) -> DevelopmentCandidateEvidence:
    """Public Task 10 adapter that reopens all candidate trace/replay evidence."""

    return _open_development_candidate_evaluation(
        root,
        definition=definition,
        candidate=candidate,
        validate_candidate=validate_candidate,
    )


def run_development_candidate_evaluation(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    candidate: dagger_domain.DevelopmentCandidate,
    output_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    evaluate_candidate: Callable[..., object] = (
        checkpoint_audit_domain.evaluate_retained_candidate
    ),
    validate_candidate: Callable[..., object] = (
        checkpoint_audit_domain.validate_retained_evaluation
    ),
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> DevelopmentCandidateRun:
    """Run or physically reopen one immutable 200-game candidate evaluation."""

    identity = _development_candidate_identity(definition, candidate)
    _require_development_candidate_inputs(
        definition, candidate, repository_identity_provider,
    )
    if type(workers) is not int or workers < 1:
        raise ValueError("development evaluation workers must be positive")
    if not callable(evaluate_candidate):
        raise RuntimeError("development physical evaluation boundary is unavailable")
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / candidate.candidate_id
    staging = root / f"{candidate.candidate_id}.staging"
    if destination.exists():
        if staging.exists():
            raise ValueError("development evaluation destination and staging are ambiguous")
        reopened = _open_development_candidate_evaluation(
            destination,
            definition=definition,
            candidate=candidate,
            validate_candidate=validate_candidate,
        )
        _require_development_candidate_inputs(
            definition, candidate, repository_identity_provider,
        )
        return DevelopmentCandidateRun(new_games=0, reused=True, result=reopened)
    if staging.exists():
        raise ValueError("development evaluation staging is partial; use a new output root")
    staging.mkdir()
    physical_root = staging / "physical"
    evaluate_candidate(
        candidate.controller,
        expected_candidate_identity=candidate.controller_identity,
        schedule=_development_schedule(),
        publication_root=physical_root,
        server_cmd=server_cmd,
        workers=workers,
    )
    _require_development_candidate_inputs(
        definition, candidate, repository_identity_provider,
    )
    retained = checkpoint_audit_domain.validate_retained_evaluation(
        physical_root / "evaluation.json",
        publication_root=physical_root,
        evidence_root=physical_root / "evidence",
        schedule=_development_schedule(),
        expected_candidate_identity=candidate.controller_identity,
    )
    if not isinstance(retained, checkpoint_audit_domain.RetainedEvaluation):
        raise ValueError("development validator returned an invalid retained evaluation")
    _require_development_candidate_inputs(
        definition, candidate, repository_identity_provider,
    )
    evaluation_bytes = (physical_root / "evaluation.json").read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "identity": identity,
        "physical": {
            "evaluation": {
                "path": "physical/evaluation.json",
                "sha256": _sha256_bytes(evaluation_bytes),
                "byte_size": len(evaluation_bytes),
            },
            "artifacts": [
                _retained_artifact_payload(artifact)
                for artifact in retained.artifacts
            ],
            "rows_sha256": _sha256_bytes(
                _canonical_stage_bytes(retained.matches)
            ),
        },
    }
    manifest["content_identity"] = _content_identity(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    staged = _open_development_candidate_evaluation(
        staging,
        definition=definition,
        candidate=candidate,
        validate_candidate=validate_candidate,
    )
    _require_development_candidate_inputs(
        definition, candidate, repository_identity_provider,
    )
    os.replace(staging, destination)
    try:
        published = _open_development_candidate_evaluation(
            destination,
            definition=definition,
            candidate=candidate,
            validate_candidate=validate_candidate,
        )
        _require_development_candidate_inputs(
            definition, candidate, repository_identity_provider,
        )
        if published.content_identity != staged.content_identity:
            raise ValueError("development evaluation changed during publication")
    except Exception:
        if destination.exists() and not staging.exists():
            os.replace(destination, staging)
        raise
    return DevelopmentCandidateRun(new_games=200, reused=False, result=published)


def run_development_evaluation(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    output_root: Path,
    server_cmd: Sequence[str],
    workers: int,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> tuple[DevelopmentCandidateRun, ...]:
    """Evaluate baseline, then iterations one through three in frozen order."""

    return tuple(
        run_development_candidate_evaluation(
            definition=definition,
            candidate=candidate,
            output_root=output_root,
            server_cmd=server_cmd,
            workers=workers,
            repository_identity_provider=repository_identity_provider,
        )
        for candidate in definition.candidates
    )


_DEVELOPMENT_SUPERVISED_REASONS = (
    "conversion", "favorable", "cycle_warning", "wasted_end_turn",
)


@dataclass(frozen=True)
class DevelopmentHeldoutOverlayEvidence:
    root: Path
    content_identity: str
    examples: tuple[Mapping[str, Any], ...]
    tree_directories: tuple[str, ...] = ()
    tree_files: tuple[tuple[str, bytes], ...] = ()


@dataclass(frozen=True)
class DevelopmentSupervisedEvidence:
    root: Path
    iteration: int
    content_identity: str
    heldout_overlay_roots: tuple[Path, ...]
    heldout_overlay_prefix: tuple[str, ...]
    incoming_candidate_id: str
    trained_candidate_id: str
    metrics: Mapping[str, Any]


@dataclass(frozen=True)
class DevelopmentSupervisedRun:
    new_inferences: int
    reused: bool
    result: DevelopmentSupervisedEvidence


def _canonical_unresolved_overlay_root(raw_root: Path) -> Path:
    supplied = Path(raw_root)
    if not supplied.is_absolute() or ".." in supplied.parts:
        raise ValueError("development heldout overlay root is not canonical")
    for path in (*reversed(supplied.parents), supplied):
        junction = getattr(path, "is_junction", None)
        try:
            if path.is_symlink() or bool(junction is not None and junction()):
                raise ValueError(
                    "development heldout overlay root chain contains a reparse point"
                )
        except OSError as exc:
            raise ValueError(
                "development heldout overlay root chain is unreadable"
            ) from exc
    try:
        canonical = supplied.resolve(strict=True)
    except OSError as exc:
        raise ValueError("development heldout overlay root is unreadable") from exc
    if supplied != canonical or not canonical.is_dir():
        raise ValueError("development heldout overlay root is a canonical alias")
    return canonical


def _supervised_overlay_tree_snapshot(
    root: Path,
) -> tuple[tuple[str, ...], tuple[tuple[str, bytes], ...]]:
    directories: set[str] = set()
    files: dict[str, bytes] = {}
    try:
        paths = tuple(root.rglob("*"))
        for path in paths:
            junction = getattr(path, "is_junction", None)
            if path.is_symlink() or bool(junction is not None and junction()):
                raise ValueError(
                    "development heldout overlay tree contains a reparse point"
                )
            canonical = path.resolve(strict=True)
            if not canonical.is_relative_to(root):
                raise ValueError("development heldout overlay tree escaped its root")
            relative = path.relative_to(root).as_posix()
            if path.is_dir():
                directories.add(relative)
            elif path.is_file():
                files[relative] = path.read_bytes()
            else:
                raise ValueError("development heldout overlay tree is invalid")
    except OSError as exc:
        raise ValueError("development heldout overlay tree is unreadable") from exc
    return (
        tuple(sorted(directories)),
        tuple((relative, files[relative]) for relative in sorted(files)),
    )


def _require_supervised_overlay_stability(
    overlays: Sequence[DevelopmentHeldoutOverlayEvidence],
) -> None:
    for item in overlays:
        if _supervised_overlay_tree_snapshot(item.root) != (
            item.tree_directories,
            item.tree_files,
        ):
            raise ValueError(
                "development heldout overlay physical bytes changed during evaluation"
            )
        try:
            reopened = dagger_domain.open_dagger_overlay(item.root)
            reopened_examples = dagger_domain.dagger_overlay_supervised_examples(reopened)
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "development heldout overlay final physical reopen failed"
            ) from exc
        if (
            reopened.partition != "validation"
            or reopened.content_identity != item.content_identity
            or not _same_json(reopened_examples, item.examples)
        ):
            raise ValueError(
                "development heldout overlay physical rows changed during evaluation"
            )
        if _supervised_overlay_tree_snapshot(item.root) != (
            item.tree_directories,
            item.tree_files,
        ):
            raise ValueError(
                "development heldout overlay physical bytes changed during final reopen"
            )


def _validated_supervised_overlays(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
    roots: Sequence[Path],
    reopen: Callable[[Path], object] | None = None,
) -> tuple[tuple[DevelopmentHeldoutOverlayEvidence, ...], tuple[Mapping[str, Any], ...]]:
    if type(iteration) is not int or iteration not in {1, 2, 3}:
        raise ValueError("development supervised iteration must be 1, 2, or 3")
    expected_prefix = tuple(
        definition.candidates[iteration].source_publication[
            "validation_overlay_prefix"
        ]
    )
    if type(roots) not in {list, tuple} or len(roots) != iteration:
        raise ValueError("development supervised cumulative overlay roots are incomplete")
    opened: list[DevelopmentHeldoutOverlayEvidence] = []
    examples: list[Mapping[str, Any]] = []
    sample_ids: set[str] = set()
    action_size = definition.candidates[iteration].controller_identity["action_size"]
    for index, raw_root in enumerate(roots):
        root = _canonical_unresolved_overlay_root(Path(raw_root))
        frozen_source = definition.candidates[index + 1].source_publication
        source_run = Path(frozen_source["source_run"])
        if frozen_source["kind"] == "dagger-iteration" and source_run.name == "actor":
            expected_root = _canonical_unresolved_overlay_root(
                source_run.parent / "validation-overlay"
            )
            if root != expected_root:
                raise ValueError(
                    "development heldout overlay root differs from its frozen source"
                )
        tree_directories, tree_files = _supervised_overlay_tree_snapshot(root)
        try:
            physical = dagger_domain.open_dagger_overlay(root)
            physical_examples = dagger_domain.dagger_overlay_supervised_examples(
                physical
            )
        except (OSError, TypeError, ValueError) as exc:
            raise ValueError(
                "development heldout overlay physical rows are invalid"
            ) from exc
        if (
            physical.partition != "validation"
            or physical.iteration != index + 1
            or physical.content_identity != expected_prefix[index]
            or physical.definition.action_size != action_size
        ):
            raise ValueError("development heldout overlay prefix identity changed")
        value = DevelopmentHeldoutOverlayEvidence(
            root=root,
            content_identity=physical.content_identity,
            examples=physical_examples,
            tree_directories=tree_directories,
            tree_files=tree_files,
        )
        _require_sha256(
            value.content_identity, "development heldout overlay identity"
        )
        for raw in physical_examples:
            if not isinstance(raw, Mapping) or set(raw) != {
                "sample_id", "oracle_action", "reasons",
            }:
                raise ValueError("development heldout sample fields are invalid")
            sample_id = raw["sample_id"]
            action = raw["oracle_action"]
            reasons = raw["reasons"]
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in sample_ids
                or type(action) is not int
                or not 0 <= action < action_size
                or type(reasons) not in {list, tuple}
                or not reasons
                or len(set(reasons)) != len(reasons)
                or any(reason not in _DEVELOPMENT_SUPERVISED_REASONS for reason in reasons)
            ):
                raise ValueError("development heldout sample identity is invalid")
            sample_ids.add(sample_id)
            examples.append(_freeze_json({
                "sample_id": sample_id,
                "oracle_action": action,
                "reasons": list(reasons),
            }))
        opened.append(value)
    return tuple(opened), tuple(examples)


def _validated_supervised_predictions(
    value: object,
    *,
    examples: Sequence[Mapping[str, Any]],
    action_size: int,
    label: str,
) -> tuple[Mapping[str, Any], ...]:
    if type(value) not in {list, tuple} or len(value) != len(examples):
        raise ValueError(f"development {label} predictions are incomplete")
    parsed: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"sample_id", "action"}:
            raise ValueError(f"development {label} prediction fields are invalid")
        if (
            raw["sample_id"] != examples[index]["sample_id"]
            or type(raw["action"]) is not int
            or not 0 <= raw["action"] < action_size
        ):
            raise ValueError(
                f"development {label} ordered sample prediction identity changed"
            )
        parsed.append(_freeze_json(dict(raw)))
    return tuple(parsed)


def _supervised_accuracy(
    examples: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    agreements = sum(
        prediction["action"] == example["oracle_action"]
        for example, prediction in zip(examples, predictions, strict=True)
    )
    labels = len(examples)
    by_reason: dict[str, Any] = {}
    for reason in _DEVELOPMENT_SUPERVISED_REASONS:
        indices = [
            index for index, example in enumerate(examples)
            if reason in example["reasons"]
        ]
        reason_agreements = sum(
            predictions[index]["action"] == examples[index]["oracle_action"]
            for index in indices
        )
        reason_labels = len(indices)
        by_reason[reason] = {
            "labels": reason_labels,
            "agreements": reason_agreements,
            "disagreements": reason_labels - reason_agreements,
            "accuracy": (
                reason_agreements / reason_labels if reason_labels else None
            ),
        }
    return {
        "agreements": agreements,
        "disagreements": labels - agreements,
        "accuracy": agreements / labels if labels else None,
        "by_reason": by_reason,
    }


def _supervised_metrics(
    examples: Sequence[Mapping[str, Any]],
    pre: Sequence[Mapping[str, Any]],
    post: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    pre_metrics = _supervised_accuracy(examples, pre)
    post_metrics = _supervised_accuracy(examples, post)
    return {
        "labels": len(examples),
        "pre": pre_metrics,
        "post": post_metrics,
        "accuracy_change": post_metrics["accuracy"] - pre_metrics["accuracy"],
    }


def _supervised_identity(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
    overlays: Sequence[DevelopmentHeldoutOverlayEvidence],
) -> dict[str, Any]:
    return {
        "definition": _development_definition_identity(definition),
        "iteration": iteration,
        "heldout_overlay_roots": [str(Path(item.root)) for item in overlays],
        "heldout_overlay_prefix": [item.content_identity for item in overlays],
        "incoming_candidate": definition.candidates[iteration - 1].to_dict(),
        "trained_candidate": definition.candidates[iteration].to_dict(),
    }


def _supervised_artifact_descriptor(path: Path, relative: str) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": _sha256_bytes(payload),
        "byte_size": len(payload),
    }


def _require_supervised_checkpoints(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
) -> None:
    for candidate in definition.candidates[iteration - 1:iteration + 1]:
        try:
            actual = _sha256_bytes(Path(candidate.checkpoint_path).read_bytes())
        except OSError as exc:
            raise ValueError("development supervised checkpoint is missing") from exc
        if actual != candidate.checkpoint_sha256:
            raise ValueError("development supervised checkpoint bytes changed")


def _open_development_supervised_evaluation(
    root: Path,
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
    overlays: Sequence[DevelopmentHeldoutOverlayEvidence],
    examples: Sequence[Mapping[str, Any]],
) -> DevelopmentSupervisedEvidence:
    supplied_root = Path(root)
    root_junction = getattr(supplied_root, "is_junction", None)
    if supplied_root.is_symlink() or bool(
        root_junction is not None and root_junction()
    ):
        raise ValueError("development supervised root is a reparse point")
    publication_root = supplied_root.resolve(strict=True)
    manifest_path = publication_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("development supervised manifest is unreadable") from exc
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != {
            "schema_version", "status", "identity", "artifacts", "content_identity",
        }
        or manifest["schema_version"] != 1
        or manifest["status"] != "completed"
        or not _same_json(
            manifest["identity"],
            _supervised_identity(
                definition=definition, iteration=iteration, overlays=overlays,
            ),
        )
        or _content_identity(manifest) != manifest["content_identity"]
    ):
        raise ValueError("development supervised publication identity changed")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, Mapping) or set(artifacts) != {
        "evidence", "predictions", "metrics",
    }:
        raise ValueError("development supervised artifact descriptors are invalid")
    payloads: dict[str, Any] = {}
    artifact_snapshots: dict[Path, bytes] = {}
    for name in ("evidence", "predictions", "metrics"):
        descriptor = artifacts[name]
        expected_path = f"{name}.json"
        if (
            not isinstance(descriptor, Mapping)
            or set(descriptor) != {"path", "sha256", "byte_size"}
            or descriptor["path"] != expected_path
            or type(descriptor["byte_size"]) is not int
            or descriptor["byte_size"] < 1
        ):
            raise ValueError("development supervised artifact descriptor changed")
        supplied_path = publication_root / expected_path
        junction = getattr(supplied_path, "is_junction", None)
        try:
            artifact_path = supplied_path.resolve(strict=True)
            if (
                supplied_path.is_symlink()
                or bool(junction is not None and junction())
                or artifact_path.parent != publication_root
            ):
                raise ValueError(
                    "development supervised artifact is not a contained regular file"
                )
            raw = artifact_path.read_bytes()
            payload = json.loads(raw.decode("utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"development supervised {name} artifact is unreadable") from exc
        if (
            len(raw) != descriptor["byte_size"]
            or _sha256_bytes(raw)
            != _require_sha256(
                descriptor["sha256"], f"development supervised {name} sha256",
            )
        ):
            raise ValueError(f"development supervised {name} artifact hash changed")
        payloads[name] = payload
        artifact_snapshots[artifact_path] = raw
    expected_evidence = {
        "schema_version": 1,
        "iteration": iteration,
        "overlays": [
            {
                "root": str(Path(item.root)),
                "content_identity": item.content_identity,
            }
            for item in overlays
        ],
        "examples": _mutable_stage_json(examples),
    }
    if not _same_json(payloads["evidence"], expected_evidence):
        raise ValueError("development supervised evidence bytes changed")
    predictions = payloads["predictions"]
    if not isinstance(predictions, Mapping) or set(predictions) != {
        "schema_version", "iteration", "pre", "post",
    } or predictions["schema_version"] != 1 or predictions["iteration"] != iteration:
        raise ValueError("development supervised prediction fields are invalid")
    action_size = definition.candidates[iteration].controller_identity["action_size"]
    pre = _validated_supervised_predictions(
        predictions["pre"], examples=examples, action_size=action_size, label="pre",
    )
    post = _validated_supervised_predictions(
        predictions["post"], examples=examples, action_size=action_size, label="post",
    )
    metrics = _supervised_metrics(examples, pre, post)
    if not _same_json(payloads["metrics"], metrics):
        raise ValueError("development supervised metrics changed")
    actual = set()
    for path in publication_root.iterdir():
        junction = getattr(path, "is_junction", None)
        if path.is_symlink() or bool(junction is not None and junction()):
            raise ValueError(
                "development supervised publication contains a reparse point"
            )
        actual.add(path.name)
    if actual != {"evidence.json", "predictions.json", "metrics.json", "manifest.json"}:
        raise ValueError("development supervised publication contains unowned files")
    if (
        manifest_path.read_bytes() != manifest_bytes
        or any(path.read_bytes() != raw for path, raw in artifact_snapshots.items())
    ):
        raise ValueError("development supervised bytes changed while reopening")
    _require_supervised_overlay_stability(overlays)
    return DevelopmentSupervisedEvidence(
        root=publication_root,
        iteration=iteration,
        content_identity=manifest["content_identity"],
        heldout_overlay_roots=tuple(Path(item.root) for item in overlays),
        heldout_overlay_prefix=tuple(item.content_identity for item in overlays),
        incoming_candidate_id=definition.candidates[iteration - 1].candidate_id,
        trained_candidate_id=definition.candidates[iteration].candidate_id,
        metrics=_freeze_json(metrics),
    )


def _open_development_supervised_evaluation_from_physical_bytes(
    root: Path,
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
) -> DevelopmentSupervisedEvidence:
    """Discover overlay roots from the owned evidence artifact, then reopen all."""

    supplied_root = Path(root)
    root_junction = getattr(supplied_root, "is_junction", None)
    if supplied_root.is_symlink() or bool(
        root_junction is not None and root_junction()
    ):
        raise ValueError("development supervised root is a reparse point")
    publication_root = supplied_root.resolve(strict=True)
    manifest_path = publication_root / "manifest.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes.decode("utf-8"))
        descriptor = manifest["artifacts"]["evidence"]
        if (
            descriptor["path"] != "evidence.json"
            or set(descriptor) != {"path", "sha256", "byte_size"}
        ):
            raise ValueError("development supervised evidence descriptor changed")
        supplied_evidence_path = publication_root / descriptor["path"]
        evidence_junction = getattr(supplied_evidence_path, "is_junction", None)
        evidence_path = supplied_evidence_path.resolve(strict=True)
        if (
            supplied_evidence_path.is_symlink()
            or bool(evidence_junction is not None and evidence_junction())
            or not evidence_path.is_relative_to(publication_root)
            or evidence_path.parent != publication_root
        ):
            raise ValueError("development supervised evidence escaped publication")
        raw = evidence_path.read_bytes()
        evidence = json.loads(raw.decode("utf-8"))
    except (OSError, KeyError, TypeError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            "development supervised physical evidence is unreadable"
        ) from exc
    if (
        len(raw) != descriptor["byte_size"]
        or _sha256_bytes(raw) != descriptor["sha256"]
        or not isinstance(evidence, Mapping)
        or set(evidence) != {
            "schema_version", "iteration", "overlays", "examples",
        }
        or evidence["schema_version"] != 1
        or evidence["iteration"] != iteration
        or type(evidence["overlays"]) is not list
        or len(evidence["overlays"]) != iteration
    ):
        raise ValueError("development supervised physical evidence identity changed")
    roots: list[Path] = []
    for item in evidence["overlays"]:
        if (
            not isinstance(item, Mapping)
            or set(item) != {"root", "content_identity"}
            or not isinstance(item["root"], str)
        ):
            raise ValueError("development supervised overlay descriptor changed")
        roots.append(Path(item["root"]))
    overlays, examples = _validated_supervised_overlays(
        definition=definition,
        iteration=iteration,
        roots=roots,
        reopen=lambda _root: None,
    )
    result = _open_development_supervised_evaluation(
        publication_root,
        definition=definition,
        iteration=iteration,
        overlays=overlays,
        examples=examples,
    )
    if (
        manifest_path.read_bytes() != manifest_bytes
        or evidence_path.read_bytes() != raw
    ):
        raise ValueError("development supervised bytes changed while reopening")
    return result


def reopen_development_supervised_evaluation(
    root: Path,
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
    heldout_overlay_roots: Sequence[Path],
    reopen_heldout_overlay: Callable[[Path], object] | None,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> DevelopmentSupervisedEvidence:
    """Reopen every source and byte of a completed Task 10 supervised stage."""

    _require_development_repository(definition, repository_identity_provider)
    overlays, examples = _validated_supervised_overlays(
        definition=definition,
        iteration=iteration,
        roots=heldout_overlay_roots,
        reopen=reopen_heldout_overlay,
    )
    _require_supervised_checkpoints(definition, iteration)
    result = _open_development_supervised_evaluation(
        root,
        definition=definition,
        iteration=iteration,
        overlays=overlays,
        examples=examples,
    )
    _require_supervised_checkpoints(definition, iteration)
    _require_development_repository(definition, repository_identity_provider)
    return result


def run_development_supervised_evaluation(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    iteration: int,
    heldout_overlay_roots: Sequence[Path],
    output_root: Path,
    reopen_heldout_overlay: Callable[[Path], object] | None,
    predict_actions: Callable[..., object] | None = None,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> DevelopmentSupervisedRun:
    """Run ordered pre/post held-out predictions or exactly reuse their publication."""

    _require_development_repository(definition, repository_identity_provider)
    overlays, examples = _validated_supervised_overlays(
        definition=definition,
        iteration=iteration,
        roots=heldout_overlay_roots,
        reopen=reopen_heldout_overlay,
    )
    _require_supervised_checkpoints(definition, iteration)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"iteration-{iteration}"
    staging = root / f"iteration-{iteration}.staging"
    if destination.exists():
        if staging.exists():
            raise ValueError("development supervised destination and staging are ambiguous")
        reopened = _open_development_supervised_evaluation(
            destination,
            definition=definition,
            iteration=iteration,
            overlays=overlays,
            examples=examples,
        )
        _require_supervised_checkpoints(definition, iteration)
        _require_development_repository(definition, repository_identity_provider)
        return DevelopmentSupervisedRun(
            new_inferences=0, reused=True, result=reopened,
        )
    if staging.exists():
        raise ValueError("development supervised staging is partial; use a new output root")
    if not callable(predict_actions):
        raise RuntimeError(
            "development supervised predictor is unavailable until the Task 11 adapter is injected"
        )
    action_size = definition.candidates[iteration].controller_identity["action_size"]
    incoming = definition.candidates[iteration - 1]
    trained = definition.candidates[iteration]
    pre_raw = predict_actions(
        controller=incoming.controller,
        controller_identity=incoming.controller_identity,
        checkpoint_path=incoming.checkpoint_path,
        examples=examples,
    )
    _require_supervised_checkpoints(definition, iteration)
    _require_development_repository(definition, repository_identity_provider)
    pre = _validated_supervised_predictions(
        pre_raw, examples=examples, action_size=action_size, label="pre",
    )
    post_raw = predict_actions(
        controller=trained.controller,
        controller_identity=trained.controller_identity,
        checkpoint_path=trained.checkpoint_path,
        examples=examples,
    )
    _require_supervised_checkpoints(definition, iteration)
    _require_development_repository(definition, repository_identity_provider)
    post = _validated_supervised_predictions(
        post_raw, examples=examples, action_size=action_size, label="post",
    )
    staging.mkdir()
    evidence = {
        "schema_version": 1,
        "iteration": iteration,
        "overlays": [
            {"root": str(Path(item.root)), "content_identity": item.content_identity}
            for item in overlays
        ],
        "examples": _mutable_stage_json(examples),
    }
    predictions = {
        "schema_version": 1,
        "iteration": iteration,
        "pre": _mutable_stage_json(pre),
        "post": _mutable_stage_json(post),
    }
    metrics = _supervised_metrics(examples, pre, post)
    atomic_write_json(staging / "evidence.json", evidence)
    atomic_write_json(staging / "predictions.json", predictions)
    atomic_write_json(staging / "metrics.json", metrics)
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "identity": _supervised_identity(
            definition=definition, iteration=iteration, overlays=overlays,
        ),
        "artifacts": {
            name: _supervised_artifact_descriptor(staging / f"{name}.json", f"{name}.json")
            for name in ("evidence", "predictions", "metrics")
        },
    }
    manifest["content_identity"] = _content_identity(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    staged = _open_development_supervised_evaluation(
        staging,
        definition=definition,
        iteration=iteration,
        overlays=overlays,
        examples=examples,
    )
    _require_supervised_checkpoints(definition, iteration)
    _require_development_repository(definition, repository_identity_provider)
    os.replace(staging, destination)
    try:
        published = _open_development_supervised_evaluation(
            destination,
            definition=definition,
            iteration=iteration,
            overlays=overlays,
            examples=examples,
        )
        if published.content_identity != staged.content_identity:
            raise ValueError("development supervised publication changed")
        _require_supervised_checkpoints(definition, iteration)
        _require_development_repository(
            definition, repository_identity_provider
        )
    except BaseException:
        if destination.exists() and not staging.exists():
            os.replace(destination, staging)
        raise
    return DevelopmentSupervisedRun(
        new_inferences=len(examples) * 2,
        reused=False,
        result=published,
    )


@dataclass(frozen=True)
class DevelopmentPreflightEvidence:
    evidence_root: Path
    content_identity: str
    selected_oracle: dagger_domain.OracleSpec
    evidence_class: str
    starting_learner_checkpoint_path: str
    starting_learner_checkpoint_sha256: str
    starting_learner_controller: str
    starting_learner_controller_identity: Mapping[str, Any]
    starting_learner_model_seed: int
    starting_learner_step: int
    starting_learner_source_content_identity: str


def _open_development_preflight_evidence(
    root: Path,
) -> DevelopmentPreflightEvidence:
    """Authenticate Task 8 physically and fail closed until Task 11 seals it."""

    try:
        definition = dagger_domain.load_panel_definition(
            _PANEL_PATH, repository_root=_REPOSITORY_ROOT,
        )
        publication = dagger_domain.open_oracle_preflight_publication(
            Path(root),
            definition=definition,
            repository_identity_provider=_git_repository_identity,
        )
    except (OSError, TypeError, ValueError) as exc:
        raise ValueError(
            "development Task 8 schema-2 preflight publication is invalid"
        ) from exc
    if publication.evidence_class == "untrusted-test-transcript":
        raise ValueError(
            "Task 8 evidence is an untrusted-test-transcript; Task 11 sealed-engine "
            "provenance is required for production Task 10 evaluation"
        )
    raise ValueError(
        "Task 11 sealed-engine provenance adapter is required before production "
        "Task 10 evaluation can accept Task 8 evidence"
    )


@dataclass(frozen=True)
class DevelopmentSourcePublicationEvidence:
    root: Path
    kind: str
    iteration: int
    content_identity: str
    preflight_root: Path
    preflight_content_identity: str
    incoming_source_content_identity: str | None
    source_run: str
    model_seed: int
    step: int
    controller: str
    controller_identity: Mapping[str, Any]
    checkpoint_path: str
    checkpoint_sha256: str
    actor_sha256: str
    publication_metadata_sha256: str
    run_manifest_sha256: str
    bc_manifest_sha256: str
    train_overlay_prefix: tuple[str, ...]
    validation_overlay_prefix: tuple[str, ...]
    publication_identity: Mapping[str, Any]


@dataclass(frozen=True)
class DevelopmentIterationEvidence:
    root: Path
    iteration: int
    content_identity: str
    selected_oracle: dagger_domain.OracleSpec
    preflight_root: Path
    preflight_content_identity: str
    preflight_evidence_class: str
    actor_checkpoint_sha256: str
    actor_controller: str
    actor_controller_identity: Mapping[str, Any]
    validation_collection: Mapping[str, Any]
    collection_metrics: Mapping[str, Any]
    training_metrics: Mapping[str, Any]
    timings: Mapping[str, Any]
    training_history_root: Path
    training_history: Mapping[str, Any]
    training_history_identity: Mapping[str, Any]


@dataclass(frozen=True)
class DevelopmentAggregatePublication:
    root: Path
    aggregate: Mapping[str, Any]
    report: str
    content_identity: str


_DEVELOPMENT_SOURCE_PUBLICATION_FIELDS = frozenset({
    "kind", "iteration", "content_identity", "preflight_root",
    "preflight_content_identity", "incoming_source_content_identity",
    "source_run", "model_seed", "step", "controller", "controller_identity",
    "checkpoint_path", "checkpoint_sha256", "actor_sha256",
    "publication_metadata_sha256", "run_manifest_sha256", "bc_manifest_sha256",
    "train_overlay_prefix", "validation_overlay_prefix",
})

_LOCKED_DEVELOPMENT_BASELINE_CHECKPOINT_SHA256 = (
    "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
)


def _development_source_publication_identity(
    value: DevelopmentSourcePublicationEvidence,
) -> dict[str, Any]:
    return {
        "kind": value.kind,
        "iteration": value.iteration,
        "content_identity": value.content_identity,
        "preflight_root": str(Path(value.preflight_root)),
        "preflight_content_identity": value.preflight_content_identity,
        "incoming_source_content_identity": value.incoming_source_content_identity,
        "source_run": value.source_run,
        "model_seed": value.model_seed,
        "step": value.step,
        "controller": value.controller,
        "controller_identity": value.controller_identity,
        "checkpoint_path": value.checkpoint_path,
        "checkpoint_sha256": value.checkpoint_sha256,
        "actor_sha256": value.actor_sha256,
        "publication_metadata_sha256": value.publication_metadata_sha256,
        "run_manifest_sha256": value.run_manifest_sha256,
        "bc_manifest_sha256": value.bc_manifest_sha256,
        "train_overlay_prefix": list(value.train_overlay_prefix),
        "validation_overlay_prefix": list(value.validation_overlay_prefix),
    }


def _open_development_source_publication_claim(
    root: Path,
    *,
    preflight: DevelopmentPreflightEvidence,
) -> DevelopmentSourcePublicationEvidence:
    """Open the real audited baseline publication, never a Task 10 claim."""

    physical = checkpoint_audit_domain.validate_audited_baseline_publication(
        Path(root),
        expected_checkpoint_sha256=(
            _LOCKED_DEVELOPMENT_BASELINE_CHECKPOINT_SHA256
        ),
    )
    canonical = physical.root
    contract = _actor_publication_contract(canonical)
    if physical.contract != contract.to_dict():
        raise ValueError("development baseline contract bytes changed")
    controller_payload = {
        "kind": "snapshot",
        "path": str(physical.checkpoint_path),
        "source_run": str(canonical),
        "algorithm": "maskable_ppo",
        "step": physical.step,
        "inference_mode": "deterministic",
    }
    controller = json.dumps(controller_payload, sort_keys=True)
    controller_identity = _freeze_json({
        "kind": "snapshot",
        "inference_mode": "deterministic",
        "path": str(physical.checkpoint_path),
        "algorithm": "maskable_ppo",
        "step": physical.step,
        "contract_hash": contract.contract_hash,
        "contract_version": contract.version,
        "environment": contract.environment,
        "encoding_hash": contract.encoding_hash,
        "contract": contract.to_dict(),
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
        "legacy": False,
        "promotable": True,
    })
    values = {
        "kind": "audited-baseline",
        "iteration": 0,
        "content_identity": physical.content_identity,
        "preflight_root": str(Path(preflight.evidence_root)),
        "preflight_content_identity": preflight.content_identity,
        "incoming_source_content_identity": None,
        "source_run": str(canonical),
        "model_seed": physical.model_seed,
        "step": physical.step,
        "controller": controller,
        "controller_identity": controller_identity,
        "checkpoint_path": str(physical.checkpoint_path),
        "checkpoint_sha256": physical.checkpoint_sha256,
        "actor_sha256": physical.checkpoint_sha256,
        "publication_metadata_sha256": physical.initialization_sha256,
        "run_manifest_sha256": physical.run_manifest_sha256,
        "bc_manifest_sha256": physical.source_bc_sha256,
        "train_overlay_prefix": [],
        "validation_overlay_prefix": [],
    }
    return DevelopmentSourcePublicationEvidence(
        root=canonical,
        kind=values["kind"],
        iteration=0,
        content_identity=values["content_identity"],
        preflight_root=Path(preflight.evidence_root),
        preflight_content_identity=preflight.content_identity,
        incoming_source_content_identity=None,
        source_run=str(canonical),
        model_seed=physical.model_seed,
        step=physical.step,
        controller=controller,
        controller_identity=controller_identity,
        checkpoint_path=str(physical.checkpoint_path),
        checkpoint_sha256=physical.checkpoint_sha256,
        actor_sha256=physical.checkpoint_sha256,
        publication_metadata_sha256=physical.initialization_sha256,
        run_manifest_sha256=physical.run_manifest_sha256,
        bc_manifest_sha256=physical.source_bc_sha256,
        train_overlay_prefix=(),
        validation_overlay_prefix=(),
        publication_identity=_freeze_json(values),
    )


def _development_actor_controller_identity(
    *,
    controller: Mapping[str, Any],
    contract: EnvironmentContract,
) -> Mapping[str, Any]:
    return _freeze_json({
        "kind": "snapshot",
        "inference_mode": "deterministic",
        "path": controller["path"],
        "algorithm": "maskable_ppo",
        "step": 0,
        "contract_hash": contract.contract_hash,
        "contract_version": contract.version,
        "environment": contract.environment,
        "encoding_hash": contract.encoding_hash,
        "contract": contract.to_dict(),
        "observation_size": contract.observation_size,
        "action_size": contract.action_size,
        "legacy": False,
        "promotable": True,
    })


_TASK7_ACTOR_FILES = frozenset({
    "actor-fixtures.npz", "bc.json", "metrics.json", "publication.json",
    "run.json", "scenario.json", "training-history.json",
    "checkpoints/step_000000000.zip",
})


def _task7_actor_snapshot(root: Path) -> Mapping[str, bytes]:
    files: set[str] = set()
    directories: set[str] = set()
    for path in root.rglob("*"):
        junction = getattr(path, "is_junction", None)
        if path.is_symlink() or bool(junction is not None and junction()):
            raise ValueError("development Task 7 actor inventory contains a reparse point")
        try:
            path.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError("development Task 7 actor inventory escaped its root") from exc
        relative = path.relative_to(root).as_posix()
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise ValueError("development Task 7 actor inventory is invalid")
    if files != _TASK7_ACTOR_FILES or directories != {"checkpoints"}:
        raise ValueError("development Task 7 actor exact inventory changed")
    return {
        relative: (root / relative).read_bytes() for relative in sorted(files)
    }


def _task9_repository_hash(repository: Mapping[str, Any]) -> str:
    return _sha256_bytes(json.dumps(
        _mutable_stage_json(repository),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8"))


def _require_task10_iteration_causal_identity(
    manifest: IterationManifest,
    *,
    iteration: int,
    previous: DevelopmentSourcePublicationEvidence,
    frozen_identity: Mapping[str, Any] | None,
) -> tuple[IterationManifest | None, bytes | None]:
    """Bind Task 9's complete incoming claim to its physical predecessor."""

    try:
        source_root = Path(previous.source_run).resolve(strict=True)
        checkpoint = Path(previous.checkpoint_path).resolve(strict=True)
        controller = json.loads(previous.controller)
    except (OSError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("development Task 9 physical predecessor is invalid") from exc
    if (
        str(source_root) != previous.source_run
        or str(checkpoint) != previous.checkpoint_path
        or not checkpoint.is_file()
        or checkpoint.parent != (source_root / "checkpoints").resolve(strict=True)
        or _sha256_bytes(checkpoint.read_bytes()) != previous.checkpoint_sha256
        or _sha256_bytes((source_root / "run.json").read_bytes())
        != previous.run_manifest_sha256
        or not isinstance(controller, Mapping)
        or set(controller) != {
            "kind", "path", "source_run", "algorithm", "step", "inference_mode",
        }
        or controller["kind"] != "snapshot"
        or controller["path"] != previous.checkpoint_path
        or controller["source_run"] != previous.source_run
        or controller["algorithm"] != "maskable_ppo"
        or controller["step"] != previous.step
        or controller["inference_mode"] != "deterministic"
        or previous.model_seed != 227
    ):
        raise ValueError("development Task 9 physical predecessor identity changed")

    learner = {
        "checkpoint_path": previous.checkpoint_path,
        "checkpoint_sha256": previous.checkpoint_sha256,
        "source_run": previous.source_run,
        "source_manifest_sha256": previous.run_manifest_sha256,
    }
    actor_source = {
        "schema_version": 1,
        "source_kind": "snapshot" if previous.iteration == 0 else "dagger_actor",
        "controller": dict(controller),
        "checkpoint_sha256": previous.checkpoint_sha256,
    }
    if previous.iteration > 0:
        actor_source["published_actor_sha256"] = previous.actor_sha256
    expected_incoming = {
        "source": actor_source,
        "identity": learner,
        "published_actor_sha256": previous.actor_sha256,
    }
    controller_identity = previous.controller_identity
    contract = (
        controller_identity.get("contract")
        if isinstance(controller_identity, Mapping)
        else None
    )
    semantics = contract.get("semantics") if isinstance(contract, Mapping) else None
    expected_contract = {
        "version": controller_identity.get("contract_version"),
        "contract_hash": controller_identity.get("contract_hash"),
        "encoding_hash": controller_identity.get("encoding_hash"),
        "observation_size": controller_identity.get("observation_size"),
        "action_size": controller_identity.get("action_size"),
        "action_regions": (
            semantics.get("action_regions") if isinstance(semantics, Mapping) else None
        ),
    }
    if previous.iteration == 0:
        baseline_contract_header = {
            name: expected_contract[name]
            for name in ("version", "contract_hash", "encoding_hash")
        }
        if not _same_json(
            baseline_contract_header,
            {
                name: manifest.identity["contract"][name]
                for name in ("version", "contract_hash", "encoding_hash")
            },
        ):
            raise ValueError(
                "development Task 9 baseline controller contract identity changed"
            )
        # The compact unit boundary trains a tiny Task 7 actor under the same
        # authenticated baseline contract/encoding. Production geometry is
        # independently authenticated below from the physical Task 7 actor.
        expected_contract = manifest.identity["contract"]
    actual_incoming = _mutable_stage_json(manifest.identity["incoming_learner"])
    try:
        actual_incoming["source"]["controller"]["path"] = str(
            Path(actual_incoming["source"]["controller"]["path"]).resolve(strict=True)
        )
        actual_incoming["source"]["controller"]["source_run"] = str(
            Path(
                actual_incoming["source"]["controller"]["source_run"]
            ).resolve(strict=True)
        )
        actual_incoming["identity"]["checkpoint_path"] = str(
            Path(actual_incoming["identity"]["checkpoint_path"]).resolve(strict=True)
        )
        actual_incoming["identity"]["source_run"] = str(
            Path(actual_incoming["identity"]["source_run"]).resolve(strict=True)
        )
    except (KeyError, OSError, TypeError) as exc:
        raise ValueError(
            "development Task 9 incoming learner paths are invalid"
        ) from exc
    actual_causal_identity = {
        "incoming_learner": actual_incoming,
        "contract": manifest.identity["contract"],
    }
    if not _same_json(actual_causal_identity, {
        "incoming_learner": expected_incoming,
        "contract": expected_contract,
    }):
        raise ValueError(
            "development Task 9 learner does not match its physical predecessor"
        )

    predecessor: IterationManifest | None = None
    predecessor_bytes: bytes | None = None
    if iteration > 1:
        predecessor_path = Path(previous.root).resolve(strict=True) / "manifest.json"
        predecessor_bytes = predecessor_path.read_bytes()
        predecessor = _read_iteration_manifest(predecessor_path)
        actor = predecessor.artifacts["actor"]
        expected_actor = {
            "checkpoint_sha256": previous.checkpoint_sha256,
            "actor_sha256": previous.actor_sha256,
            "publication_metadata_sha256": previous.publication_metadata_sha256,
            "run_manifest_sha256": previous.run_manifest_sha256,
            "bc_manifest_sha256": previous.bc_manifest_sha256,
        }
        if (
            predecessor.content_identity != previous.content_identity
            or any(actor.get(name) != value for name, value in expected_actor.items())
        ):
            raise ValueError(
                "development Task 9 predecessor publication identity changed"
            )
    _require_iteration_predecessor_chain(
        manifest.identity,
        index=iteration,
        predecessor=predecessor,
    )

    if frozen_identity is not None:
        expected_frozen = {
            "panel_hash": frozen_identity["panel_hash"],
            "scenario_hash": frozen_identity["scenario_hash"],
            "repository": frozen_identity["repository"],
            "contract_hash": frozen_identity["contract_hash"],
            "encoding_hash": frozen_identity["encoding_hash"],
        }
        actual_frozen = {
            "panel_hash": manifest.identity["definition"]["panel_sha256"],
            "scenario_hash": manifest.identity["scenario"]["runtime_sha256"],
            "repository": manifest.identity["repository"],
            "contract_hash": manifest.identity["contract"]["contract_hash"],
            "encoding_hash": manifest.identity["contract"]["encoding_hash"],
        }
        if not _same_json(actual_frozen, expected_frozen):
            raise ValueError("development Task 9 frozen causal identity changed")
    return predecessor, predecessor_bytes


def _open_development_iteration_source(
    root: Path,
    *,
    iteration: int,
    preflight: DevelopmentPreflightEvidence,
    previous: DevelopmentSourcePublicationEvidence,
    frozen_identity: Mapping[str, Any] | None = None,
) -> DevelopmentSourcePublicationEvidence:
    """Authenticate one Task 9 manifest, both overlays, and its Task 7 actor."""

    canonical = Path(root).resolve(strict=True)
    manifest_path = canonical / "manifest.json"
    manifest_bytes_before = manifest_path.read_bytes()
    manifest = _read_iteration_manifest(manifest_path)
    if manifest.iteration != iteration:
        raise ValueError("development Task 9 iteration number changed")
    if {path.name for path in canonical.iterdir()} != {
        "manifest.json", "train-overlay", "validation-overlay", "actor",
    }:
        raise ValueError("development Task 9 publication contains unowned entries")
    predecessor, predecessor_bytes = _require_task10_iteration_causal_identity(
        manifest,
        iteration=iteration,
        previous=previous,
        frozen_identity=frozen_identity,
    )
    expected_predecessor_learner = {
        "checkpoint_path": previous.checkpoint_path,
        "checkpoint_sha256": previous.checkpoint_sha256,
        "source_run": previous.source_run,
        "source_manifest_sha256": previous.run_manifest_sha256,
    }

    opened_overlays: dict[str, dagger_domain.DaggerOverlay] = {}
    recomputed_metrics: dict[str, Mapping[str, Any]] = {}
    for partition in ("train", "validation"):
        descriptor = manifest.artifacts[f"{partition}_overlay"]
        relative = descriptor["path"]
        overlay_root = (canonical / relative).resolve(strict=True)
        if (
            relative != f"{partition}-overlay"
            or not overlay_root.is_relative_to(canonical)
            or overlay_root.parent != canonical
        ):
            raise ValueError("development Task 9 overlay escaped its publication")
        overlay = dagger_domain.open_dagger_overlay(overlay_root)
        if (
            overlay.partition != partition
            or overlay.iteration != iteration
            or overlay.content_identity != descriptor["content_identity"]
            or overlay.row_count != descriptor["row_count"]
            or overlay.definition.learner.checkpoint_path
            != previous.checkpoint_path
            or overlay.definition.learner.checkpoint_sha256
            != previous.checkpoint_sha256
            or overlay.definition.learner.source_run != previous.source_run
            or not _same_json(
                overlay.definition.learner.to_dict(),
                expected_predecessor_learner,
            )
        ):
            raise ValueError("development Task 9 physical overlay identity changed")
        expected_overlay_definition = {
            "partition": partition,
            "iteration": iteration,
            "observation_size": manifest.identity["contract"]["observation_size"],
            "action_size": manifest.identity["contract"]["action_size"],
            "action_regions": manifest.identity["contract"]["action_regions"],
            "oracle": manifest.identity["selected_oracle"]["spec"],
            "learner": overlay.definition.learner.to_dict(),
            "original_dataset": overlay.definition.original_dataset.to_dict(),
            "scenario_hash": manifest.identity["scenario"]["runtime_sha256"],
            "contract_hash": manifest.identity["contract"]["contract_hash"],
            "encoding_hash": manifest.identity["contract"]["encoding_hash"],
            "repository_hash": _task9_repository_hash(
                manifest.identity["repository"]
            ),
            "panel_hash": manifest.identity["definition"]["panel_sha256"],
            "schedule_hash": manifest.identity["schedules"][partition]["sha256"],
            "label_target": manifest.identity["schedules"][partition]["label_target"],
            "game_ceiling": manifest.identity["schedules"][partition]["game_ceiling"],
        }
        actual_overlay_definition = overlay.definition.to_dict()
        incoming_learner = manifest.identity["incoming_learner"]["identity"]
        learner_mismatch = (
            Path(incoming_learner["checkpoint_path"]).resolve(strict=True)
            != Path(overlay.definition.learner.checkpoint_path).resolve(strict=True)
            or Path(incoming_learner["source_run"]).resolve(strict=True)
            != Path(overlay.definition.learner.source_run).resolve(strict=True)
            or incoming_learner["checkpoint_sha256"]
            != overlay.definition.learner.checkpoint_sha256
            or incoming_learner["source_manifest_sha256"]
            != overlay.definition.learner.source_manifest_sha256
        )
        definition_mismatches = tuple(
            key for key in expected_overlay_definition
            if not _same_json(
                actual_overlay_definition[key], expected_overlay_definition[key]
            )
        )
        if definition_mismatches or learner_mismatch or (
            overlay.definition.original_dataset.manifest_sha256
            != manifest.identity["base_dataset"]["manifest_sha256"]
        ):
            raise ValueError(
                "development Task 9 overlay definition does not match its manifest: "
                + ", ".join(
                    definition_mismatches
                    or (("learner",) if learner_mismatch else ("original_dataset",))
                )
            )
        physical_metrics = dagger_domain.dagger_overlay_collection_metrics(overlay)
        if not _same_json(
            physical_metrics, manifest.metrics[f"{partition}_collection"],
        ):
            raise ValueError(
                "development Task 9 collection metrics differ from physical rows"
            )
        recomputed_metrics[partition] = physical_metrics
        opened_overlays[partition] = overlay
    train_shared = opened_overlays["train"].definition.to_dict()
    validation_shared = opened_overlays["validation"].definition.to_dict()
    for shared in (train_shared, validation_shared):
        for name in ("partition", "label_target", "game_ceiling"):
            shared.pop(name)
    if not _same_json(train_shared, validation_shared):
        raise ValueError("development Task 9 paired overlay identity changed")

    expected_train = (
        *previous.train_overlay_prefix,
        opened_overlays["train"].content_identity,
    )
    expected_validation = (
        *previous.validation_overlay_prefix,
        opened_overlays["validation"].content_identity,
    )
    cumulative_train = manifest.identity["cumulative_train_overlays"]
    cumulative_validation = manifest.identity["cumulative_validation_overlays"]
    if (
        tuple(item["iteration"] for item in cumulative_train)
        != tuple(range(1, iteration + 1))
        or tuple(item["iteration"] for item in cumulative_validation)
        != tuple(range(1, iteration + 1))
        or tuple(item["content_identity"] for item in cumulative_train)
        != expected_train
        or tuple(item["content_identity"] for item in cumulative_validation)
        != expected_validation
    ):
        raise ValueError("development Task 9 cumulative overlay chain changed")

    selected = manifest.identity["selected_oracle"]
    expected_selected = {
        "spec": preflight.selected_oracle.to_dict(),
        "evidence_root": str(preflight.evidence_root),
        "evidence_content_identity": preflight.content_identity,
        "evidence_class": preflight.evidence_class,
    }
    if not _same_json(selected, expected_selected):
        raise ValueError("development Task 9 preflight/oracle identity changed")
    predecessor = manifest.identity["predecessor"]
    if (
        (iteration == 1 and predecessor is not None)
        or (
            iteration > 1
            and (
                not isinstance(predecessor, Mapping)
                or predecessor.get("iteration") != iteration - 1
                or predecessor.get("content_identity") != previous.content_identity
            )
        )
    ):
        raise ValueError("development Task 9 predecessor identity changed")

    actor_descriptor = manifest.artifacts["actor"]
    actor_root = (canonical / actor_descriptor["path"]).resolve(strict=True)
    if (
        actor_descriptor["path"] != "actor"
        or not actor_root.is_relative_to(canonical)
        or actor_root.parent != canonical
    ):
        raise ValueError("development Task 7 actor escaped its Task 9 publication")
    actor_snapshot = _task7_actor_snapshot(actor_root)
    contract = _actor_publication_contract(actor_root)
    if not _same_json(
        _critical_contract(contract), manifest.identity["contract"],
    ):
        raise ValueError("development Task 7 actor contract changed")
    run_path = actor_root / "run.json"
    bc_path = actor_root / "bc.json"
    run_sha_before = _sha256_bytes(actor_snapshot["run.json"])
    bc_sha_before = _sha256_bytes(actor_snapshot["bc.json"])
    verification = imitation_domain.validate_actor_supervision_publication(
        actor_root, contract,
    )
    run_sha_after = _sha256_bytes(run_path.read_bytes())
    bc_sha_after = _sha256_bytes(bc_path.read_bytes())
    if run_sha_before != run_sha_after or bc_sha_before != bc_sha_after:
        raise ValueError("development Task 7 actor changed during authentication")
    try:
        run_manifest = json.loads(actor_snapshot["run.json"].decode("utf-8"))
        latest = run_manifest["latest_checkpoint"]
        checkpoint = (actor_root / latest).resolve(strict=True)
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("development Task 7 checkpoint descriptor is invalid") from exc
    if (
        latest != "checkpoints/step_000000000.zip"
        or not checkpoint.is_relative_to(actor_root)
        or checkpoint.parent != (actor_root / "checkpoints").resolve()
    ):
        raise ValueError("development Task 7 checkpoint containment changed")
    checkpoint_sha = _sha256_bytes(
        actor_snapshot["checkpoints/step_000000000.zip"]
    )
    physical_actor = {
        "checkpoint_sha256": checkpoint_sha,
        "actor_sha256": verification.get("actor_sha256"),
        "publication_metadata_sha256": verification.get(
            "publication_metadata_sha256"
        ),
        "run_manifest_sha256": run_sha_after,
        "bc_manifest_sha256": bc_sha_after,
    }
    if any(
        actor_descriptor.get(name) != value
        for name, value in physical_actor.items()
    ) or verification.get("checkpoint_sha256") != checkpoint_sha:
        raise ValueError("development Task 7 actor bytes do not match Task 9")

    # Close the read window: every independently opened source must still reopen
    # to the same immutable content identity after actor validation.
    if manifest_path.read_bytes() != manifest_bytes_before:
        raise ValueError("development Task 9 manifest changed during authentication")
    if predecessor is not None and (
        predecessor_bytes is None
        or (
            Path(previous.root).resolve(strict=True) / "manifest.json"
        ).read_bytes() != predecessor_bytes
    ):
        raise ValueError(
            "development Task 9 predecessor changed during authentication"
        )
    if _task7_actor_snapshot(actor_root) != actor_snapshot:
        raise ValueError("development Task 7 actor changed during authentication")
    for partition, original in opened_overlays.items():
        reopened = dagger_domain.open_dagger_overlay(
            canonical / f"{partition}-overlay"
        )
        if (
            reopened.content_identity != original.content_identity
            or not _same_json(
                dagger_domain.dagger_overlay_collection_metrics(reopened),
                recomputed_metrics[partition],
            )
        ):
            raise ValueError("development Task 9 overlay changed during authentication")

    controller_payload = {
        "kind": "snapshot",
        "path": str(checkpoint),
        "source_run": str(actor_root),
        "algorithm": "maskable_ppo",
        "step": 0,
        "inference_mode": "deterministic",
    }
    controller = json.dumps(controller_payload, sort_keys=True)
    identity = _development_actor_controller_identity(
        controller=controller_payload, contract=contract,
    )
    values = {
        "kind": "dagger-iteration",
        "iteration": iteration,
        "content_identity": manifest.content_identity,
        "preflight_root": str(preflight.evidence_root),
        "preflight_content_identity": preflight.content_identity,
        "incoming_source_content_identity": previous.content_identity,
        "source_run": str(actor_root),
        "model_seed": 227,
        "step": 0,
        "controller": controller,
        "controller_identity": identity,
        "checkpoint_path": str(checkpoint),
        **physical_actor,
        "train_overlay_prefix": list(expected_train),
        "validation_overlay_prefix": list(expected_validation),
    }
    return DevelopmentSourcePublicationEvidence(
        root=canonical,
        kind=values["kind"],
        iteration=iteration,
        content_identity=values["content_identity"],
        preflight_root=Path(values["preflight_root"]),
        preflight_content_identity=values["preflight_content_identity"],
        incoming_source_content_identity=values[
            "incoming_source_content_identity"
        ],
        source_run=values["source_run"],
        model_seed=227,
        step=0,
        controller=controller,
        controller_identity=identity,
        checkpoint_path=values["checkpoint_path"],
        checkpoint_sha256=checkpoint_sha,
        actor_sha256=physical_actor["actor_sha256"],
        publication_metadata_sha256=physical_actor[
            "publication_metadata_sha256"
        ],
        run_manifest_sha256=run_sha_after,
        bc_manifest_sha256=bc_sha_after,
        train_overlay_prefix=expected_train,
        validation_overlay_prefix=expected_validation,
        publication_identity=_freeze_json(values),
    )


def _validated_development_source_publication(
    value: object,
    *,
    root: Path,
    iteration: int,
    preflight: DevelopmentPreflightEvidence,
    previous: DevelopmentSourcePublicationEvidence | None,
    frozen_identity: Mapping[str, Any] | None = None,
) -> DevelopmentSourcePublicationEvidence:
    if not isinstance(value, DevelopmentSourcePublicationEvidence):
        raise ValueError("development source publication reopener returned an invalid type")
    canonical = Path(root).resolve(strict=True)
    if iteration == 0:
        physical = _open_development_source_publication_claim(
            canonical, preflight=preflight,
        )
    else:
        if previous is None:
            raise ValueError("development Task 9 source predecessor is missing")
        physical = _open_development_iteration_source(
            canonical,
            iteration=iteration,
            preflight=preflight,
            previous=previous,
            frozen_identity=frozen_identity,
        )
    if not _same_json(
        _development_source_publication_identity(value),
        _development_source_publication_identity(physical),
    ):
        raise ValueError(
            "development source callback does not match physical manifest"
        )
    value = physical
    if Path(value.root) != canonical or type(value.iteration) is not int:
        raise ValueError("development source publication physical identity changed")
    expected_identity = _development_source_publication_identity(value)
    if (
        set(value.publication_identity) != _DEVELOPMENT_SOURCE_PUBLICATION_FIELDS
        or not _same_json(value.publication_identity, expected_identity)
    ):
        raise ValueError(
            "development source publication identity or cumulative overlay prefix changed"
        )
    expected_kind = "audited-baseline" if iteration == 0 else "dagger-iteration"
    expected_step = 38_912 if iteration == 0 else 0
    if (
        value.iteration != iteration
        or value.kind != expected_kind
        or Path(value.preflight_root) != Path(preflight.evidence_root)
        or value.preflight_content_identity != preflight.content_identity
        or type(value.model_seed) is not int
        or value.model_seed != 227
        or type(value.step) is not int
        or value.step != expected_step
    ):
        raise ValueError("development source preflight, seed, or step identity changed")
    for field in (
        "content_identity", "preflight_content_identity", "checkpoint_sha256",
        "actor_sha256", "publication_metadata_sha256", "run_manifest_sha256",
        "bc_manifest_sha256",
    ):
        _require_sha256(getattr(value, field), f"development source {field}")
    checkpoint = Path(value.checkpoint_path).resolve(strict=True)
    if (
        not checkpoint.is_file()
        or str(checkpoint) != value.checkpoint_path
        or _sha256_bytes(checkpoint.read_bytes()) != value.checkpoint_sha256
    ):
        raise ValueError("development source checkpoint identity changed")
    try:
        controller = json.loads(value.controller)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("development source controller identity is invalid") from exc
    if (
        not isinstance(controller, Mapping)
        or controller.get("path") != value.checkpoint_path
        or controller.get("source_run") != value.source_run
        or controller.get("step") != expected_step
        or not isinstance(value.controller_identity, Mapping)
        or value.controller_identity.get("path") != value.checkpoint_path
        or value.controller_identity.get("step") != expected_step
    ):
        raise ValueError("development source actor controller identity changed")
    for field in ("train_overlay_prefix", "validation_overlay_prefix"):
        prefix = getattr(value, field)
        if type(prefix) is not tuple or len(prefix) != iteration:
            raise ValueError("development source cumulative overlay prefix changed")
        for content_identity in prefix:
            _require_sha256(content_identity, "development source overlay identity")
    if iteration == 0:
        if (
            value.root != Path(value.source_run).resolve(strict=True)
            or value.actor_sha256 != value.checkpoint_sha256
        ):
            raise ValueError("development baseline physical publication bytes changed")
        if (
            previous is not None
            or value.incoming_source_content_identity is not None
            or value.content_identity
            != preflight.starting_learner_source_content_identity
            or value.checkpoint_sha256
            != _LOCKED_DEVELOPMENT_BASELINE_CHECKPOINT_SHA256
        ):
            raise ValueError("development baseline source identity changed")
    else:
        if previous is None or (
            value.incoming_source_content_identity != previous.content_identity
            or value.train_overlay_prefix[:-1] != previous.train_overlay_prefix
            or value.validation_overlay_prefix[:-1]
            != previous.validation_overlay_prefix
        ):
            raise ValueError("development source cumulative overlay chain changed")
    return value


def build_development_evaluation_definition(
    *,
    preflight_root: Path,
    baseline_root: Path,
    iteration_roots: Sequence[Path],
    panel_hash: str,
    scenario_hash: str,
    contract_hash: str,
    encoding_hash: str,
    repository_root: Path,
    reopen_preflight: Callable[[Path], object] | None,
    reopen_baseline: Callable[[Path], object] | None,
    reopen_iteration: Callable[[Path], object] | None,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> dagger_domain.DevelopmentEvaluationDefinition:
    """Build the four-candidate definition only from reopened physical evidence."""

    if type(iteration_roots) not in {list, tuple} or len(iteration_roots) != 3:
        raise ValueError("development definition requires exactly three iteration roots")
    repository = _validated_repository_identity(
        Path(repository_root), repository_identity_provider,
    )
    canonical_preflight_root = Path(preflight_root).resolve(strict=True)
    preflight = _open_development_preflight_evidence(
        canonical_preflight_root
    )
    if (
        Path(preflight.evidence_root) != canonical_preflight_root
        or canonical_preflight_root.is_relative_to(Path(repository["root"]))
        or preflight.evidence_class != "sealed-engine"
        or not isinstance(preflight.selected_oracle, dagger_domain.OracleSpec)
        or type(preflight.starting_learner_model_seed) is not int
        or preflight.starting_learner_model_seed != 227
        or type(preflight.starting_learner_step) is not int
        or preflight.starting_learner_step != 38_912
    ):
        raise ValueError("development physical preflight identity is invalid")
    _require_sha256(preflight.content_identity, "development preflight content identity")
    _require_sha256(
        preflight.starting_learner_source_content_identity,
        "development preflight starting source identity",
    )
    dagger_domain.OracleSpec.from_dict(preflight.selected_oracle.to_dict())
    roots = (
        Path(baseline_root).resolve(strict=True),
        *(Path(root).resolve(strict=True) for root in iteration_roots),
    )
    candidates: list[dagger_domain.DevelopmentCandidate] = []
    sources: list[DevelopmentSourcePublicationEvidence] = []
    frozen_identity = {
        "panel_hash": panel_hash,
        "scenario_hash": scenario_hash,
        "contract_hash": contract_hash,
        "encoding_hash": encoding_hash,
        "repository": repository,
    }
    for iteration, root in enumerate(roots):
        reopened = (
            _open_development_source_publication_claim(root, preflight=preflight)
            if iteration == 0
            else _open_development_iteration_source(
                root,
                iteration=iteration,
                preflight=preflight,
                previous=sources[-1],
                frozen_identity=frozen_identity,
            )
        )
        source = _validated_development_source_publication(
            reopened,
            root=root,
            iteration=iteration,
            preflight=preflight,
            previous=None if iteration == 0 else sources[-1],
            frozen_identity=frozen_identity,
        )
        source_identity = _development_source_publication_identity(source)
        candidate = dagger_domain.DevelopmentCandidate.from_dict({
            "candidate_id": "baseline" if iteration == 0 else f"iteration-{iteration}",
            "iteration": iteration,
            "controller": source.controller,
            "checkpoint_path": source.checkpoint_path,
            "checkpoint_sha256": source.checkpoint_sha256,
            "controller_identity": source.controller_identity,
            "source_publication": source_identity,
        })
        if iteration == 0 and (
            preflight.starting_learner_checkpoint_path != candidate.checkpoint_path
            or preflight.starting_learner_checkpoint_sha256
            != candidate.checkpoint_sha256
            or preflight.starting_learner_controller != candidate.controller
            or not _same_json(
                preflight.starting_learner_controller_identity,
                candidate.controller_identity,
            )
        ):
            raise ValueError("development baseline does not match preflight starting learner")
        sources.append(source)
        candidates.append(candidate)
    return dagger_domain.DevelopmentEvaluationDefinition.create(
        candidates=candidates,
        panel_hash=panel_hash,
        scenario_hash=scenario_hash,
        contract_hash=contract_hash,
        encoding_hash=encoding_hash,
        repository=repository,
    )


def _validated_development_preflight(
    value: object,
    *,
    root: Path,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
) -> DevelopmentPreflightEvidence:
    if not isinstance(value, DevelopmentPreflightEvidence):
        raise ValueError("development preflight reopener returned an invalid type")
    canonical = Path(root).resolve(strict=True)
    repository_root = Path(definition.repository["root"]).resolve(strict=True)
    if canonical.is_relative_to(repository_root):
        raise ValueError("development preflight evidence root must be external")
    if Path(value.evidence_root) != canonical:
        raise ValueError("development preflight evidence root identity changed")
    _require_sha256(value.content_identity, "development preflight content identity")
    if (
        not isinstance(value.selected_oracle, dagger_domain.OracleSpec)
        or value.evidence_class != "sealed-engine"
    ):
        raise ValueError("development preflight must be sealed engine evidence")
    # Reparse the value through the public strict type contract.
    dagger_domain.OracleSpec.from_dict(value.selected_oracle.to_dict())
    baseline = definition.candidates[0]
    if (
        value.starting_learner_checkpoint_path != baseline.checkpoint_path
        or value.starting_learner_checkpoint_sha256 != baseline.checkpoint_sha256
        or value.starting_learner_controller != baseline.controller
        or not _same_json(
            value.starting_learner_controller_identity,
            baseline.controller_identity,
        )
        or _sha256_bytes(Path(baseline.checkpoint_path).read_bytes())
        != baseline.checkpoint_sha256
    ):
        raise ValueError(
            "development baseline checkpoint does not match the validated starting learner"
        )
    return value


def _validated_development_iteration(
    value: object,
    *,
    root: Path,
    iteration: int,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    preflight: DevelopmentPreflightEvidence,
) -> DevelopmentIterationEvidence:
    if not isinstance(value, DevelopmentIterationEvidence):
        raise ValueError("development iteration reopener returned an invalid type")
    canonical = Path(root).resolve(strict=True)
    if Path(value.root) != canonical or value.iteration != iteration:
        raise ValueError("development iteration physical identity changed")
    _require_sha256(value.content_identity, "development iteration content identity")
    if (
        not isinstance(value.selected_oracle, dagger_domain.OracleSpec)
        or not _same_json(
            value.selected_oracle.to_dict(), preflight.selected_oracle.to_dict(),
        )
        or Path(value.preflight_root) != Path(preflight.evidence_root)
        or value.preflight_content_identity != preflight.content_identity
        or value.preflight_evidence_class != preflight.evidence_class
    ):
        raise ValueError(
            "development iteration oracle/preflight identity does not reconcile"
        )
    expected = definition.candidates[iteration]
    if (
        value.actor_checkpoint_sha256 != expected.checkpoint_sha256
        or value.actor_controller != expected.controller
        or not _same_json(
            value.actor_controller_identity, expected.controller_identity,
        )
    ):
        raise ValueError("development iteration actor candidate identity changed")
    for payload, label in (
        (value.validation_collection, "validation collection"),
        (value.collection_metrics, "collection metrics"),
        (value.training_metrics, "training metrics"),
        (value.timings, "timings"),
    ):
        if not isinstance(payload, Mapping):
            raise ValueError(f"development iteration {label} is invalid")
    expected_training_fields = {
        "schema_version", "model_seed", "device", "epoch", "max_epochs",
        "batches", "examples", "mean_training_loss", "validation_nll",
        "top1_accuracy", "top3_accuracy", "top5_accuracy", "best_epoch",
        "best_validation_nll", "epochs_without_improvement", "patience",
        "epoch_seconds", "elapsed_seconds", "examples_per_second",
        "sampling_seconds", "transfer_forward_seconds", "optimization_seconds",
        "validation_seconds", "unclassified_seconds",
    }
    try:
        actor_source_root = Path(json.loads(expected.controller)["source_run"]).resolve(
            strict=True
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError("development iteration actor source run is invalid") from exc
    if Path(value.training_history_root) != actor_source_root:
        raise ValueError("development iteration training history root identity changed")
    physical_history, physical_identity = imitation_domain._read_training_history_identity(
        actor_source_root,
    )
    if (
        not _same_json(value.training_history, physical_history)
        or not _same_json(value.training_history_identity, physical_identity)
        or set(physical_history) != {
            "schema_version", "model_seed", "training_device",
            "publication_device", "epochs",
        }
        or physical_history["schema_version"] != 1
        or type(physical_history["model_seed"]) is not int
        or physical_history["model_seed"] != 227
        or not isinstance(physical_history["training_device"], Mapping)
        or physical_history["publication_device"] != "cpu"
    ):
        raise ValueError("development iteration physical training history identity changed")
    epochs = physical_history["epochs"]
    previous_elapsed = -1.0
    for epoch_index, event in enumerate(epochs, start=1):
        if not isinstance(event, Mapping) or set(event) != expected_training_fields:
            raise ValueError("development iteration training history epoch fields changed")
        imitation_domain._validate_behavioral_cloning_progress_event(event)
        if (
            event["model_seed"] != 227
            or event["epoch"] != epoch_index
            or event["max_epochs"] != 50
            or not isinstance(event["device"], str)
            or not event["device"]
            or event["validation_nll"] < 0
            or event["best_validation_nll"] < 0
            or any(
                not 0.0 <= event[field] <= 1.0
                for field in ("top1_accuracy", "top3_accuracy", "top5_accuracy")
            )
            or not (
                event["top1_accuracy"]
                <= event["top3_accuracy"]
                <= event["top5_accuracy"]
            )
            or not 1 <= event["best_epoch"] <= event["epoch"]
            or event["elapsed_seconds"] < previous_elapsed
        ):
            raise ValueError("development iteration training history epoch is invalid")
        previous_elapsed = float(event["elapsed_seconds"])
    training_metrics = value.training_metrics
    if set(training_metrics) != {
        "best_epoch", "best_validation_nll", "epochs_trained",
    }:
        raise ValueError("development iteration training summary fields changed")
    final = epochs[-1]
    if (
        type(training_metrics["best_epoch"]) is not int
        or type(training_metrics["epochs_trained"]) is not int
        or isinstance(training_metrics["best_validation_nll"], bool)
        or not isinstance(training_metrics["best_validation_nll"], (int, float))
        or not math.isfinite(training_metrics["best_validation_nll"])
        or training_metrics["best_validation_nll"] < 0
        or training_metrics["best_epoch"] != final["best_epoch"]
        or training_metrics["best_validation_nll"]
        != final["best_validation_nll"]
        or training_metrics["epochs_trained"] != len(epochs)
        or physical_identity.get("epoch_count") != len(epochs)
    ):
        raise ValueError("development iteration training history summary changed")
    return value


def _development_source_from_candidate(
    candidate: dagger_domain.DevelopmentCandidate,
) -> DevelopmentSourcePublicationEvidence:
    source = candidate.source_publication
    return DevelopmentSourcePublicationEvidence(
        root=Path(source["source_run"]),
        kind=source["kind"],
        iteration=source["iteration"],
        content_identity=source["content_identity"],
        preflight_root=Path(source["preflight_root"]),
        preflight_content_identity=source["preflight_content_identity"],
        incoming_source_content_identity=source[
            "incoming_source_content_identity"
        ],
        source_run=source["source_run"],
        model_seed=source["model_seed"],
        step=source["step"],
        controller=source["controller"],
        controller_identity=source["controller_identity"],
        checkpoint_path=source["checkpoint_path"],
        checkpoint_sha256=source["checkpoint_sha256"],
        actor_sha256=source["actor_sha256"],
        publication_metadata_sha256=source[
            "publication_metadata_sha256"
        ],
        run_manifest_sha256=source["run_manifest_sha256"],
        bc_manifest_sha256=source["bc_manifest_sha256"],
        train_overlay_prefix=tuple(source["train_overlay_prefix"]),
        validation_overlay_prefix=tuple(
            source["validation_overlay_prefix"]
        ),
        publication_identity=source,
    )


def _open_development_iteration_evidence_from_physical_bytes(
    root: Path,
    *,
    iteration: int,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    preflight: DevelopmentPreflightEvidence,
    previous: DevelopmentSourcePublicationEvidence,
) -> tuple[DevelopmentIterationEvidence, DevelopmentSourcePublicationEvidence]:
    canonical = Path(root).resolve(strict=True)
    source = _open_development_iteration_source(
        canonical,
        iteration=iteration,
        preflight=preflight,
        previous=previous,
        frozen_identity={
            "panel_hash": definition.panel_hash,
            "scenario_hash": definition.scenario_hash,
            "contract_hash": definition.contract_hash,
            "encoding_hash": definition.encoding_hash,
            "repository": definition.repository,
        },
    )
    expected = definition.candidates[iteration]
    if not _same_json(
        _development_source_publication_identity(source),
        expected.source_publication,
    ):
        raise ValueError("development aggregate Task 9 source differs from definition")
    manifest_path = canonical / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest = _read_iteration_manifest(manifest_path)
    if manifest.content_identity != source.content_identity:
        raise ValueError("development aggregate Task 9 manifest identity changed")
    physical_collection_metrics = {
        partition: dagger_domain.dagger_overlay_collection_metrics(
            dagger_domain.open_dagger_overlay(
                canonical / f"{partition}-overlay"
            )
        )
        for partition in ("train", "validation")
    }
    if any(
        not _same_json(
            physical_collection_metrics[partition],
            manifest.metrics[f"{partition}_collection"],
        )
        for partition in ("train", "validation")
    ):
        raise ValueError("development aggregate Task 9 physical metrics changed")
    actor_root = Path(source.source_run)
    history, history_identity = imitation_domain._read_training_history_identity(
        actor_root
    )
    training = manifest.metrics["training"]
    evidence = DevelopmentIterationEvidence(
        root=canonical,
        iteration=iteration,
        content_identity=manifest.content_identity,
        selected_oracle=preflight.selected_oracle,
        preflight_root=preflight.evidence_root,
        preflight_content_identity=preflight.content_identity,
        preflight_evidence_class=preflight.evidence_class,
        actor_checkpoint_sha256=source.checkpoint_sha256,
        actor_controller=source.controller,
        actor_controller_identity=source.controller_identity,
        validation_collection=_freeze_json(
            physical_collection_metrics["validation"]
        ),
        collection_metrics=_freeze_json(physical_collection_metrics),
        training_metrics=_freeze_json({
            "best_epoch": training["best_epoch"],
            "best_validation_nll": training["best_validation_nll"],
            "epochs_trained": training["epochs_trained"],
        }),
        timings=manifest.timings,
        training_history_root=actor_root,
        training_history=history,
        training_history_identity=history_identity,
    )
    evidence = _validated_development_iteration(
        evidence,
        root=canonical,
        iteration=iteration,
        definition=definition,
        preflight=preflight,
    )
    if manifest_path.read_bytes() != manifest_bytes:
        raise ValueError("development aggregate Task 9 manifest changed while reopening")
    return evidence, source


def _validated_development_evaluation_evidence(
    value: object,
    *,
    root: Path,
    candidate: dagger_domain.DevelopmentCandidate,
) -> DevelopmentCandidateEvidence:
    if not isinstance(value, DevelopmentCandidateEvidence):
        raise ValueError("development evaluation reopener returned an invalid type")
    canonical = Path(root).resolve(strict=True)
    if (
        Path(value.root) != canonical
        or value.candidate_id != candidate.candidate_id
        or value.controller != candidate.controller
        or value.checkpoint_sha256 != candidate.checkpoint_sha256
        or not _same_json(value.controller_identity, candidate.controller_identity)
    ):
        raise ValueError("development physical evaluation candidate identity changed")
    _require_sha256(value.content_identity, "development evaluation content identity")
    if type(value.matches) is not tuple:
        raise ValueError("development physical evaluation rows are invalid")
    return value


def _validated_development_supervised_evidence(
    value: object,
    *,
    root: Path,
    iteration: int,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
) -> DevelopmentSupervisedEvidence:
    if not isinstance(value, DevelopmentSupervisedEvidence):
        raise ValueError("development supervised reopener returned an invalid type")
    canonical = Path(root).resolve(strict=True)
    expected_prefix = tuple(
        definition.candidates[iteration].source_publication[
            "validation_overlay_prefix"
        ]
    )
    if (
        Path(value.root) != canonical
        or type(value.iteration) is not int
        or value.iteration != iteration
        or value.incoming_candidate_id
        != definition.candidates[iteration - 1].candidate_id
        or value.trained_candidate_id != definition.candidates[iteration].candidate_id
        or value.heldout_overlay_prefix != expected_prefix
        or type(value.heldout_overlay_roots) is not tuple
        or len(value.heldout_overlay_roots) != iteration
        or any(not Path(item).resolve(strict=True).is_dir()
               for item in value.heldout_overlay_roots)
        or not isinstance(value.metrics, Mapping)
    ):
        raise ValueError("development physical supervised evidence identity changed")
    _require_sha256(value.content_identity, "development supervised content identity")
    return value


def _development_preflight_snapshot(value: DevelopmentPreflightEvidence) -> Mapping[str, Any]:
    return {
        "evidence_root": str(value.evidence_root),
        "content_identity": value.content_identity,
        "selected_oracle": value.selected_oracle.to_dict(),
        "evidence_class": value.evidence_class,
        "starting_learner_checkpoint_path": value.starting_learner_checkpoint_path,
        "starting_learner_checkpoint_sha256": value.starting_learner_checkpoint_sha256,
        "starting_learner_controller": value.starting_learner_controller,
        "starting_learner_controller_identity": value.starting_learner_controller_identity,
        "starting_learner_model_seed": value.starting_learner_model_seed,
        "starting_learner_step": value.starting_learner_step,
        "starting_learner_source_content_identity": (
            value.starting_learner_source_content_identity
        ),
    }


def _development_iteration_snapshot(value: DevelopmentIterationEvidence) -> Mapping[str, Any]:
    return {
        "root": str(value.root), "iteration": value.iteration,
        "content_identity": value.content_identity,
        "selected_oracle": value.selected_oracle.to_dict(),
        "preflight_root": str(value.preflight_root),
        "preflight_content_identity": value.preflight_content_identity,
        "preflight_evidence_class": value.preflight_evidence_class,
        "actor_checkpoint_sha256": value.actor_checkpoint_sha256,
        "actor_controller": value.actor_controller,
        "actor_controller_identity": value.actor_controller_identity,
        "validation_collection": value.validation_collection,
        "collection_metrics": value.collection_metrics,
        "training_metrics": value.training_metrics,
        "timings": value.timings,
        "training_history_root": str(value.training_history_root),
        "training_history": value.training_history,
        "training_history_identity": value.training_history_identity,
    }


def _development_evaluation_snapshot(value: DevelopmentCandidateEvidence) -> Mapping[str, Any]:
    return {
        "root": str(value.root), "candidate_id": value.candidate_id,
        "controller": value.controller,
        "checkpoint_sha256": value.checkpoint_sha256,
        "controller_identity": value.controller_identity,
        "content_identity": value.content_identity,
        "matches": value.matches,
    }


def _development_supervised_snapshot(value: DevelopmentSupervisedEvidence) -> Mapping[str, Any]:
    return {
        "root": str(value.root), "iteration": value.iteration,
        "content_identity": value.content_identity,
        "heldout_overlay_roots": [str(root) for root in value.heldout_overlay_roots],
        "heldout_overlay_prefix": value.heldout_overlay_prefix,
        "incoming_candidate_id": value.incoming_candidate_id,
        "trained_candidate_id": value.trained_candidate_id,
        "metrics": value.metrics,
    }


def _require_development_definition_checkpoints(
    definition: dagger_domain.DevelopmentEvaluationDefinition,
) -> None:
    for candidate in definition.candidates:
        try:
            actual = _sha256_bytes(Path(candidate.checkpoint_path).read_bytes())
        except OSError as exc:
            raise ValueError("development aggregate source checkpoint is missing") from exc
        if actual != candidate.checkpoint_sha256:
            raise ValueError("development aggregate source checkpoint identity changed")


_DEVELOPMENT_AGGREGATE_MANIFEST_FIELDS = frozenset({
    "schema_version", "status", "identity", "artifacts", "content_identity",
})


def _open_development_aggregate_publication(
    root: Path,
    *,
    expected_aggregate: Mapping[str, Any],
    expected_report: str,
) -> DevelopmentAggregatePublication:
    publication_root = Path(root).resolve(strict=True)
    manifest_path = publication_root / "manifest.json"
    aggregate_path = publication_root / "aggregate.json"
    report_path = publication_root / "REPORT.md"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        aggregate_bytes = aggregate_path.read_bytes()
        aggregate = json.loads(aggregate_bytes.decode("utf-8"))
        report_bytes = report_path.read_bytes()
        report = report_bytes.decode("utf-8")
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("development aggregate publication is unreadable") from exc
    if (
        not isinstance(manifest, Mapping)
        or set(manifest) != _DEVELOPMENT_AGGREGATE_MANIFEST_FIELDS
        or manifest["schema_version"] != 1
        or manifest["status"] != "completed"
        or not isinstance(manifest["identity"], Mapping)
        or not _same_json(manifest["identity"], expected_aggregate["evidence_identity"])
        or not _same_json(aggregate, expected_aggregate)
        or report != expected_report
    ):
        raise ValueError("development aggregate publication identity changed")
    artifacts = manifest["artifacts"]
    expected_descriptors = {
        "aggregate": {
            "path": "aggregate.json",
            "sha256": _sha256_bytes(aggregate_bytes),
            "byte_size": len(aggregate_bytes),
        },
        "report": {
            "path": "REPORT.md",
            "sha256": _sha256_bytes(report_bytes),
            "byte_size": len(report_bytes),
        },
    }
    if not _same_json(artifacts, expected_descriptors):
        raise ValueError("development aggregate artifact descriptors changed")
    if (
        not isinstance(manifest["content_identity"], str)
        or _content_identity(manifest) != manifest["content_identity"]
    ):
        raise ValueError("development aggregate content identity changed")
    actual = {
        path.relative_to(publication_root).as_posix()
        for path in publication_root.iterdir()
    }
    if actual != {"manifest.json", "aggregate.json", "REPORT.md"}:
        raise ValueError("development aggregate contains unowned files")
    return DevelopmentAggregatePublication(
        root=publication_root,
        aggregate=_freeze_json(aggregate),
        report=report,
        content_identity=manifest["content_identity"],
    )


def publish_development_aggregate(
    *,
    definition: dagger_domain.DevelopmentEvaluationDefinition,
    preflight_root: Path,
    iteration_roots: Sequence[Path],
    evaluations_root: Path,
    supervised_roots: Sequence[Path],
    output_root: Path,
    reopen_preflight: Callable[[Path], object] | None,
    reopen_iteration: Callable[[Path], object] | None,
    reopen_evaluation: Callable[
        [Path, dagger_domain.DevelopmentCandidate], object
    ] | None,
    reopen_supervised: Callable[[Path, int], object] | None,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] = _git_repository_identity,
) -> DevelopmentAggregatePublication:
    """Reconstruct and transactionally publish Task 10 from physical evidence."""

    _development_definition_identity(definition)
    _require_development_definition_checkpoints(definition)
    _require_development_repository(definition, repository_identity_provider)
    canonical_preflight_root = Path(preflight_root).resolve(strict=True)
    preflight = _validated_development_preflight(
        _open_development_preflight_evidence(canonical_preflight_root),
        root=canonical_preflight_root,
        definition=definition,
    )
    _require_development_repository(definition, repository_identity_provider)
    if type(iteration_roots) not in {list, tuple} or len(iteration_roots) != 3:
        raise ValueError("development aggregate requires three physical iterations")
    canonical_iteration_roots = tuple(
        Path(root).resolve(strict=True) for root in iteration_roots
    )
    iteration_values: list[DevelopmentIterationEvidence] = []
    canonical_baseline_root = Path(
        definition.candidates[0].source_publication["source_run"]
    ).resolve(strict=True)
    previous_source = _validated_development_source_publication(
        _open_development_source_publication_claim(
            canonical_baseline_root, preflight=preflight,
        ),
        root=canonical_baseline_root,
        iteration=0,
        preflight=preflight,
        previous=None,
    )
    if not _same_json(
        _development_source_publication_identity(previous_source),
        definition.candidates[0].source_publication,
    ):
        raise ValueError("development aggregate baseline source identity changed")
    for iteration, root in enumerate(canonical_iteration_roots, start=1):
        evidence, previous_source = (
            _open_development_iteration_evidence_from_physical_bytes(
                root,
                iteration=iteration,
                definition=definition,
                preflight=preflight,
                previous=previous_source,
            )
        )
        iteration_values.append(evidence)
        _require_development_repository(definition, repository_identity_provider)
    iterations = tuple(iteration_values)
    canonical_evaluations_root = Path(evaluations_root).resolve(strict=True)
    evaluation_values: list[DevelopmentCandidateEvidence] = []
    for candidate in definition.candidates:
        root = (canonical_evaluations_root / candidate.candidate_id).resolve(strict=True)
        evaluation_values.append(_open_development_candidate_evaluation(
            root,
            definition=definition,
            candidate=candidate,
            validate_candidate=(
                checkpoint_audit_domain.validate_retained_evaluation
            ),
        ))
        _require_development_repository(definition, repository_identity_provider)
    evaluations = tuple(evaluation_values)
    if type(supervised_roots) not in {list, tuple} or len(supervised_roots) != 3:
        raise ValueError("development aggregate requires three physical supervised evidences")
    canonical_supervised_roots = tuple(
        Path(root).resolve(strict=True) for root in supervised_roots
    )
    supervised_values: list[DevelopmentSupervisedEvidence] = []
    for iteration, root in enumerate(canonical_supervised_roots, start=1):
        supervised_values.append(
            _open_development_supervised_evaluation_from_physical_bytes(
                root,
                definition=definition,
                iteration=iteration,
            )
        )
        _require_development_repository(definition, repository_identity_provider)
    supervised = tuple(supervised_values)
    baseline = definition.candidates[0]
    if (
        evaluations[0].controller != preflight.starting_learner_controller
        or not _same_json(
            evaluations[0].controller_identity,
            preflight.starting_learner_controller_identity,
        )
        or evaluations[0].checkpoint_sha256
        != preflight.starting_learner_checkpoint_sha256
        or not _same_json(
            evaluations[0].controller_identity, baseline.controller_identity,
        )
    ):
        raise ValueError(
            "development baseline evaluation does not match the validated learner identity"
        )
    aggregate = dict(dagger_domain.build_development_aggregate(
        rows_by_candidate={
            candidate.candidate_id: evidence.matches
            for candidate, evidence in zip(
                definition.candidates, evaluations, strict=True,
            )
        },
        supervised_metrics_by_iteration={
            item.iteration: item.metrics for item in supervised
        },
        evidence_identity={
            "preflight": preflight.content_identity,
            "baseline": baseline.checkpoint_sha256,
            "iterations": [item.content_identity for item in iterations],
            "evaluations": [item.content_identity for item in evaluations],
            "supervised": [item.content_identity for item in supervised],
        },
    ))
    aggregate["frozen_inputs"] = definition.to_dict()
    aggregate["oracle_selection"] = {
        "spec": preflight.selected_oracle.to_dict(),
        "evidence_root": str(preflight.evidence_root),
        "evidence_content_identity": preflight.content_identity,
        "evidence_class": preflight.evidence_class,
        "starting_learner": {
            "controller": preflight.starting_learner_controller,
            "controller_identity": _mutable_stage_json(
                preflight.starting_learner_controller_identity
            ),
            "checkpoint_path": preflight.starting_learner_checkpoint_path,
            "checkpoint_sha256": preflight.starting_learner_checkpoint_sha256,
        },
    }
    aggregate["iteration_evidence"] = [
        {
            "iteration": item.iteration,
            "content_identity": item.content_identity,
            "actor_checkpoint_sha256": item.actor_checkpoint_sha256,
            "actor_controller": item.actor_controller,
            "actor_controller_identity": _mutable_stage_json(
                item.actor_controller_identity
            ),
            "validation_collection": _mutable_stage_json(
                item.validation_collection
            ),
            "collection_metrics": _mutable_stage_json(item.collection_metrics),
            "training_metrics": _mutable_stage_json(item.training_metrics),
            "timings": _mutable_stage_json(item.timings),
            "training_history_root": str(item.training_history_root),
            "training_history_identity": _mutable_stage_json(
                item.training_history_identity
            ),
            "training_history": _mutable_stage_json(item.training_history),
        }
        for item in iterations
    ]
    aggregate["supervised_evidence"] = [
        {
            "iteration": item.iteration,
            "content_identity": item.content_identity,
            "heldout_overlay_roots": [
                str(root) for root in item.heldout_overlay_roots
            ],
            "heldout_overlay_prefix": list(item.heldout_overlay_prefix),
            "incoming_candidate_id": item.incoming_candidate_id,
            "trained_candidate_id": item.trained_candidate_id,
            "metrics": _mutable_stage_json(item.metrics),
        }
        for item in supervised
    ]
    report = dagger_domain.render_development_report(aggregate)

    reopened_preflight = _validated_development_preflight(
        _open_development_preflight_evidence(canonical_preflight_root),
        root=canonical_preflight_root,
        definition=definition,
    )
    if not _same_json(
        _development_preflight_snapshot(reopened_preflight),
        _development_preflight_snapshot(preflight),
    ):
        raise ValueError("development aggregate preflight source identity changed")
    reopened_baseline = _validated_development_source_publication(
        _open_development_source_publication_claim(
            canonical_baseline_root, preflight=reopened_preflight,
        ),
        root=canonical_baseline_root,
        iteration=0,
        preflight=reopened_preflight,
        previous=None,
    )
    if not _same_json(
        _development_source_publication_identity(reopened_baseline),
        definition.candidates[0].source_publication,
    ):
        raise ValueError("development aggregate baseline source identity changed")
    previous_source = reopened_baseline
    for iteration, root in enumerate(canonical_iteration_roots, start=1):
        reopened, previous_source = (
            _open_development_iteration_evidence_from_physical_bytes(
                root,
                iteration=iteration,
                definition=definition,
                preflight=preflight,
                previous=previous_source,
            )
        )
        if not _same_json(
            _development_iteration_snapshot(reopened),
            _development_iteration_snapshot(iterations[iteration - 1]),
        ):
            raise ValueError("development aggregate iteration source identity changed")
    for candidate, original in zip(
        definition.candidates, evaluations, strict=True,
    ):
        root = (canonical_evaluations_root / candidate.candidate_id).resolve(strict=True)
        reopened = _open_development_candidate_evaluation(
            root,
            definition=definition,
            candidate=candidate,
            validate_candidate=(
                checkpoint_audit_domain.validate_retained_evaluation
            ),
        )
        if not _same_json(
            _development_evaluation_snapshot(reopened),
            _development_evaluation_snapshot(original),
        ):
            raise ValueError("development aggregate evaluation source identity changed")
    for iteration, root in enumerate(canonical_supervised_roots, start=1):
        reopened = _open_development_supervised_evaluation_from_physical_bytes(
            root,
            definition=definition,
            iteration=iteration,
        )
        if not _same_json(
            _development_supervised_snapshot(reopened),
            _development_supervised_snapshot(supervised[iteration - 1]),
        ):
            raise ValueError("development aggregate supervised source identity changed")
    _require_development_definition_checkpoints(definition)
    _require_development_repository(definition, repository_identity_provider)

    destination = Path(output_root).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f"{destination.name}.staging"
    if destination.exists():
        if staging.exists():
            raise ValueError("development aggregate destination and staging are ambiguous")
        reopened = _open_development_aggregate_publication(
            destination,
            expected_aggregate=aggregate,
            expected_report=report,
        )
        _require_development_definition_checkpoints(definition)
        _require_development_repository(definition, repository_identity_provider)
        return reopened
    if staging.exists():
        raise ValueError("development aggregate staging is partial; use a new output root")
    staging.mkdir()
    atomic_write_json(staging / "aggregate.json", aggregate)
    _write_exact_file(staging / "REPORT.md", report.encode("utf-8"))
    aggregate_bytes = (staging / "aggregate.json").read_bytes()
    report_bytes = (staging / "REPORT.md").read_bytes()
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "identity": aggregate["evidence_identity"],
        "artifacts": {
            "aggregate": {
                "path": "aggregate.json",
                "sha256": _sha256_bytes(aggregate_bytes),
                "byte_size": len(aggregate_bytes),
            },
            "report": {
                "path": "REPORT.md",
                "sha256": _sha256_bytes(report_bytes),
                "byte_size": len(report_bytes),
            },
        },
    }
    manifest["content_identity"] = _content_identity(manifest)
    atomic_write_json(staging / "manifest.json", manifest)
    staged = _open_development_aggregate_publication(
        staging,
        expected_aggregate=aggregate,
        expected_report=report,
    )
    _require_development_definition_checkpoints(definition)
    _require_development_repository(definition, repository_identity_provider)
    os.replace(staging, destination)
    try:
        published = _open_development_aggregate_publication(
            destination,
            expected_aggregate=aggregate,
            expected_report=report,
        )
        _require_development_definition_checkpoints(definition)
        _require_development_repository(definition, repository_identity_provider)
        if published.content_identity != staged.content_identity:
            raise ValueError("development aggregate changed during publication")
    except Exception:
        if destination.exists() and not staging.exists():
            os.replace(destination, staging)
        raise
    return published
