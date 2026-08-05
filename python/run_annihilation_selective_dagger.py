"""Orchestrate the selective-DAgger training experiment."""

from __future__ import annotations

import json
import hashlib
import os
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import ml_lab.dagger as dagger_domain
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
    return IterationManifest.from_dict(payload)


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
    if completed is not None and staging.exists():
        raise ValueError(
            "completed selective-DAgger iteration has ambiguous staging"
        )
    if completed is None and staging.exists():
        raise ValueError(
            "selective-DAgger iteration staging is partial; use a new output root"
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
