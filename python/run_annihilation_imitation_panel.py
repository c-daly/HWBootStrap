"""Locked Task 8 orchestration for imitation collection, pure clones, and their gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ml_lab.scenarios import ResolvedScenario, resolve_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCENARIO_PATH = PROJECT_ROOT / "python" / "config" / "annihilation-imitation-v1.json"
PANEL_ROOT = PROJECT_ROOT / "python" / "panels" / "annihilation-imitation-v1"
PANEL_PATH = PANEL_ROOT / "panel.json"
SEED_BANKS_PATH = PANEL_ROOT / "seed-banks.json"
DATASET_PATH = PROJECT_ROOT / "python" / "datasets" / "annihilation-imitation-v1"
CLONE_RUNS_PATH = PANEL_ROOT / "bc-clones"
CLONE_EVALUATION_PATH = PANEL_ROOT / "bc-development-gate"
PPO_RUNS_PATH = PANEL_ROOT / "ppo-runs"
DEVELOPMENT_PATH = PANEL_ROOT / "development"
SELECTION_PATH = PANEL_ROOT / "selection.json"
SMOKE_ROOT = PANEL_ROOT / "evidence" / "smoke"
EXECUTION_IDENTITY_PATH = PANEL_ROOT / "execution-identity.json"
_SMOKE_GENERATED_ROOT = "python/panels/annihilation-imitation-v1/evidence/"
_FULL_GENERATED_PATHS = (
    "python/datasets/annihilation-imitation-v1/manifest.json",
    "python/datasets/annihilation-imitation-v1/games.jsonl",
    "python/datasets/annihilation-imitation-v1/shards/",
    "python/datasets/annihilation-imitation-v1/replays/",
    "python/datasets/annihilation-imitation-v1/runtime-scenario.json",
    "python/datasets/annihilation-imitation-v1/runtime-scenario-provenance.json",
    "python/datasets/annihilation-imitation-v1/.stage-definitions.json",
    "python/datasets/annihilation-imitation-v1/stage.json",
    "python/datasets/.annihilation-imitation-v1.staging/",
    "python/panels/annihilation-imitation-v1/execution-identity.json",
    "python/panels/annihilation-imitation-v1/bc-clones/",
    "python/panels/annihilation-imitation-v1/.bc-clones.staging/",
    "python/panels/annihilation-imitation-v1/bc-development-gate/",
    "python/panels/annihilation-imitation-v1/.bc-development-gate.staging/",
    "python/panels/annihilation-imitation-v1/ppo-runs/",
    "python/panels/annihilation-imitation-v1/.ppo-runs.staging/",
    "python/panels/annihilation-imitation-v1/development/",
    "python/panels/annihilation-imitation-v1/.development.staging/",
    "python/panels/annihilation-imitation-v1/selection.json",
    "python/panels/annihilation-imitation-v1/final-seal.json",
    "python/panels/annihilation-imitation-v1/final-evaluation.json",
    "python/panels/annihilation-imitation-v1/.final-evaluation.pending/",
    "python/panels/annihilation-imitation-v1/final-publication.json",
    "python/panels/annihilation-imitation-v1/.final-generations/",
    _SMOKE_GENERATED_ROOT,
)
_PUBLISHABLE_RESULT_PATHS = (
    "python/panels/annihilation-imitation-v1/aggregate.json",
    "python/panels/annihilation-imitation-v1/REPORT.md",
)

_MODEL_SEEDS = [211, 223, 227]
_SAMPLER_SEEDS = {"211": 211, "223": 223, "227": 227}
_OUTCOME_REWARDS = {"win": 1, "loss": -1, "draw": 0}
_COLLECTION = {
    "standard_decisions_minimum": 100_000,
    "conversion_decisions_minimum": 50_000,
    "validation_standard_maps": 100,
    "validation_conversion_maps_per_profile": 20,
}
_CLONE_GATE = {
    "maps": 100,
    "games_per_clone": 200,
    "seed_start": 16_000_000,
    "profile": "standard-3v3",
    "opponent": "random",
    "per_seed_win_rate_minimum_basis_points": 3000,
    "pooled_win_rate_minimum_basis_points": 4000,
}
_BANKS = {
    "greedy_demonstrations": {"start": 11_000_000, "stop": 11_499_999, "assigned": True},
    "search_demonstrations": {"start": 11_500_000, "stop": 11_999_999, "assigned": True},
    "bc_validation": {"start": 12_000_000, "stop": 12_099_999, "assigned": True},
    "ppo_replicates": {"start": 13_000_000, "stop": 15_999_999, "assigned": True},
    "development": {"start": 16_000_000, "stop": 16_000_099, "assigned": True},
    "final": {"start": 17_000_000, "stop": 17_000_249, "assigned": False},
}
_WEIGHTS = [
    {"profile_id": "standard-3v3", "basis_points": 7000},
    {"profile_id": "conversion-3v1-near", "basis_points": 500},
    {"profile_id": "conversion-3v1-medium", "basis_points": 0},
    {"profile_id": "conversion-3v1-far", "basis_points": 500},
    {"profile_id": "conversion-2v1-near", "basis_points": 500},
    {"profile_id": "conversion-2v1-medium", "basis_points": 0},
    {"profile_id": "conversion-2v1-far", "basis_points": 500},
    {"profile_id": "conversion-1v1-near", "basis_points": 500},
    {"profile_id": "conversion-1v1-medium", "basis_points": 0},
    {"profile_id": "conversion-1v1-far", "basis_points": 500},
]
_REWARD = {
    "shape_scale": 0.01,
    "step_penalty": 0.005,
    "closing_weight": 0,
    "draw_credit_weight": 0,
    "points_weight": 0.5,
}
_CONVERSION_PROFILES = (
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path}: definition is unreadable") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path}: definition must be an object")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_text(path: Path, value: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=f".{path.name}.", suffix=".tmp",
            dir=path.parent, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def current_definition_hashes(
    *,
    panel_path: Path = PANEL_PATH,
    seed_banks_path: Path = SEED_BANKS_PATH,
    scenario_path: Path = SCENARIO_PATH,
) -> dict[str, str]:
    return {
        "panel_sha256": _sha256(panel_path),
        "scenario_sha256": _sha256(scenario_path),
        "seed_banks_sha256": _sha256(seed_banks_path),
    }


def _execution_policy() -> dict[str, Any]:
    return {
        "required_clean": True,
        "ignored_generated_paths": list(_FULL_GENERATED_PATHS),
        "publishable_result_paths": list(_PUBLISHABLE_RESULT_PATHS),
    }


def _validate_hex_digest(value: Any, label: str, lengths: set[int]) -> str:
    if (
        not isinstance(value, str)
        or len(value) not in lengths
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"execution identity {label} is invalid")
    return value


def _validate_execution_identity(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("execution identity is missing")
    identity = dict(raw)
    if set(identity) != {
        "schema_version", "commit", "source_tree", "dirty", "policy",
        "definition_hashes",
    }:
        raise ValueError("execution identity schema is invalid")
    if identity.get("schema_version") != 1:
        raise ValueError("execution identity schema is invalid")
    _validate_hex_digest(identity.get("commit"), "commit", {40, 64})
    _validate_hex_digest(identity.get("source_tree"), "source tree", {40, 64})
    if identity.get("dirty") is not False:
        raise ValueError("execution identity requires a clean repository")
    if identity.get("policy") != _execution_policy():
        raise ValueError("execution identity policy is invalid")
    hashes = identity.get("definition_hashes")
    if (
        not isinstance(hashes, Mapping)
        or set(hashes)
        != {"panel_sha256", "scenario_sha256", "seed_banks_sha256"}
    ):
        raise ValueError("execution identity definition hashes are invalid")
    normalized_hashes = {
        name: _validate_hex_digest(value, name, {64})
        for name, value in hashes.items()
    }
    return {
        "schema_version": 1,
        "commit": identity["commit"],
        "source_tree": identity["source_tree"],
        "dirty": False,
        "policy": _execution_policy(),
        "definition_hashes": normalized_hashes,
    }


def _repository_execution_source(repository: Path) -> dict[str, Any]:
    repository = Path(repository)
    revision, dirty = _repository_identity(repository)
    source_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": revision, "source_tree": source_tree, "dirty": dirty}


def _build_execution_identity(
    *,
    definition_hashes: Mapping[str, str],
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    provider = repository_identity_provider or _repository_execution_source
    source = provider(Path(repository))
    if not isinstance(source, Mapping) or set(source) != {
        "commit", "source_tree", "dirty",
    }:
        raise ValueError("execution identity repository source is invalid")
    return _validate_execution_identity({
        "schema_version": 1,
        "commit": source["commit"],
        "source_tree": source["source_tree"],
        "dirty": source["dirty"],
        "policy": _execution_policy(),
        "definition_hashes": dict(definition_hashes),
    })


def _validate_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    validate_definitions()
    identity = _build_execution_identity(
        definition_hashes=current_definition_hashes(),
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    _atomic_json(Path(execution_identity_path), identity)
    return {"state": "validated", "execution_identity": identity}


def _require_execution_identity(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    definition_hashes: Mapping[str, str],
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> dict[str, Any]:
    try:
        stored = _validate_execution_identity(
            _read_json(Path(execution_identity_path))
        )
    except ValueError as exc:
        raise ValueError("execution identity is missing or invalid") from exc
    current = _build_execution_identity(
        definition_hashes=definition_hashes,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    if stored != current:
        raise ValueError("execution identity changed after validation")
    return stored


def _full_execution_context(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> tuple[dict, dict, ResolvedScenario, dict[str, str], dict[str, Any]]:
    panel, banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
    identity = _require_execution_identity(
        execution_identity_path=execution_identity_path,
        definition_hashes=hashes,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return panel, banks, scenario, hashes, identity


def _full_stage_validator(
    physical_validator: Callable[[Path], Mapping[str, Any]],
    *,
    expected_identity: Mapping[str, Any],
    execution_identity_path: Path,
    repository: Path,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None,
) -> Callable[[Path], Mapping[str, Any]]:
    expected = _validate_execution_identity(expected_identity)

    def validate(root: Path) -> Mapping[str, Any]:
        summary = dict(physical_validator(Path(root)))
        current = _require_execution_identity(
            execution_identity_path=execution_identity_path,
            definition_hashes=current_definition_hashes(),
            repository=repository,
            repository_identity_provider=repository_identity_provider,
        )
        if current != expected:
            raise ValueError("execution identity changed during stage")
        return summary

    return validate


def _validate_dataset_execution_identity(
    dataset_root: Path,
    execution_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = _validate_execution_identity(execution_identity)
    manifest = _read_json(Path(dataset_root) / "manifest.json")
    if (
        manifest.get("code_revision") != identity["commit"]
        or manifest.get("dirty") is not False
    ):
        raise ValueError("dataset execution identity does not match")
    return manifest


def validate_definitions(
    *,
    panel_path: Path = PANEL_PATH,
    seed_banks_path: Path = SEED_BANKS_PATH,
    scenario_path: Path = SCENARIO_PATH,
) -> tuple[dict, dict, ResolvedScenario]:
    panel = _read_json(panel_path)
    banks = _read_json(seed_banks_path)

    registered = panel.get("definition_hashes")
    if not isinstance(registered, dict):
        raise ValueError("panel definition hashes are missing")
    if registered.get("scenario_sha256") != _sha256(scenario_path):
        raise ValueError("scenario definition hash does not match panel")
    if registered.get("seed_banks_sha256") != _sha256(seed_banks_path):
        raise ValueError("seed-banks definition hash does not match panel")

    expected_panel_keys = {
        "schema_version", "id", "environment", "scenario", "seed_banks",
        "definition_hashes", "model_seeds", "sampler_seeds", "outcome_rewards",
        "collection", "behavioral_cloning", "clone_gate", "ppo", "development",
    }
    if set(panel) != expected_panel_keys or panel.get("schema_version") != 1:
        raise ValueError("panel definition schema is invalid")
    if panel["id"] != "annihilation-imitation-v1" or panel["environment"] != "tactical-v2":
        raise ValueError("panel identity is invalid")
    if panel["scenario"] != "python/config/annihilation-imitation-v1.json":
        raise ValueError("panel scenario path is invalid")
    if panel["seed_banks"] != "python/panels/annihilation-imitation-v1/seed-banks.json":
        raise ValueError("panel seed-bank path is invalid")
    if panel["model_seeds"] != _MODEL_SEEDS or panel["sampler_seeds"] != _SAMPLER_SEEDS:
        raise ValueError("panel model or sampler seeds changed")
    if panel["outcome_rewards"] != _OUTCOME_REWARDS:
        raise ValueError("panel outcome rewards changed")
    if (
        panel["collection"] != _COLLECTION
        or panel["clone_gate"] != _CLONE_GATE
        or panel["ppo"] != _PPO
        or panel["development"] != _DEVELOPMENT
    ):
        raise ValueError(
            "panel collection, clone gate, PPO, or development definition changed"
        )
    expected_bc = {
        "batch_size": 256,
        "learning_rate": 0.0003,
        "max_epochs": 50,
        "patience": 5,
        "standard_fraction_basis_points": 7000,
    }
    if panel["behavioral_cloning"] != expected_bc:
        raise ValueError("behavioral-cloning definition changed")

    if set(banks) != {"schema_version", *_BANKS} or banks.get("schema_version") != 1:
        raise ValueError("seed-bank namespaces changed")
    ranges: list[tuple[int, int, str]] = []
    for name, expected in _BANKS.items():
        if banks.get(name) != expected:
            raise ValueError(f"seed-bank namespace {name!r} changed")
        ranges.append((expected["start"], expected["stop"], name))
    for previous, following in zip(sorted(ranges), sorted(ranges)[1:]):
        if previous[1] >= following[0]:
            raise ValueError(f"seed-bank namespaces {previous[2]!r} and {following[2]!r} overlap")

    scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=Path(scenario_path),
        template_id=None,
        enforce_round_cap_minimum=True,
    )
    if scenario.template_id != "annihilation-imitation-v1":
        raise ValueError("locked scenario identity changed")
    distribution = [dict(item) for item in scenario.document["tactical_v2"]["start_distribution"]]
    if distribution != _WEIGHTS:
        raise ValueError("locked scenario start distribution changed")
    if dict(scenario.document["reward"]) != _REWARD:
        raise ValueError("locked scenario reward changed")
    return panel, banks, scenario


def _materialize_runtime_scenario(
    scenario: ResolvedScenario,
    stage_root: Path,
    definition_hashes: Mapping[str, str],
) -> Path:
    """Freeze validated scenario semantics inside the artifact being published."""
    root = Path(stage_root)
    path = root / "runtime-scenario.json"
    canonical = scenario.canonical_json
    _atomic_text(path, canonical + "\n")
    frozen = resolve_scenario(
        environment="tactical-v2", scenario_file=path, template_id=None,
        enforce_round_cap_minimum=True,
    )
    if frozen.canonical_json != canonical:
        raise ValueError("runtime scenario snapshot does not match the validated definition")
    _atomic_json(
        root / "runtime-scenario-provenance.json",
        {
            "schema_version": 1,
            "definition_hashes": dict(definition_hashes),
            "canonical_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        },
    )
    return path


def _validate_runtime_scenario(
    root: Path, scenario: ResolvedScenario, hashes: Mapping[str, str]
) -> Path:
    path = Path(root) / "runtime-scenario.json"
    provenance = _read_json(Path(root) / "runtime-scenario-provenance.json")
    if provenance.get("definition_hashes") != dict(hashes):
        raise ValueError("runtime scenario definition hashes do not match")
    frozen = resolve_scenario(
        environment="tactical-v2", scenario_file=path, template_id=None,
        enforce_round_cap_minimum=True,
    )
    canonical_hash = hashlib.sha256(frozen.canonical_json.encode("utf-8")).hexdigest()
    if (
        frozen.canonical_json != scenario.canonical_json
        or provenance.get("canonical_sha256") != canonical_hash
    ):
        raise ValueError("runtime scenario snapshot does not match the locked scenario")
    return path


def _stage_definitions(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    hashes = value.get("definition_hashes")
    if not isinstance(hashes, dict):
        raise ValueError("stage definition hashes are missing")
    return dict(hashes)


def _stage_identity(path: Path) -> dict[str, Any] | None:
    value = _read_json(path)
    if "stage_identity" not in value:
        return None
    identity = value["stage_identity"]
    if not isinstance(identity, dict):
        raise ValueError("stage identity is invalid")
    return dict(identity)


def run_atomic_stage(
    destination: Path,
    definition_hashes: Mapping[str, str],
    *,
    stage_identity: Mapping[str, Any] | None = None,
    build: Callable[[Path], None],
    validate: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    destination = Path(destination)
    hashes = dict(definition_hashes)
    identity = dict(stage_identity) if stage_identity is not None else None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        stage_path = destination / "stage.json"
        if not stage_path.is_file() or _stage_definitions(stage_path) != hashes:
            raise ValueError("completed stage definition hashes do not match")
        if _stage_identity(stage_path) != identity:
            raise ValueError("completed stage identity does not match")
        summary = dict(validate(destination))
        result = _read_json(stage_path)
        result.update(summary=summary, reused=True)
        return result

    staging = destination.with_name(f".{destination.name}.staging")
    provenance_path = staging / ".stage-definitions.json"
    if staging.exists():
        if not provenance_path.is_file() or _stage_definitions(provenance_path) != hashes:
            raise ValueError("staged definition hashes do not match")
        if _stage_identity(provenance_path) != identity:
            raise ValueError("staged stage identity does not match")
    else:
        staging.mkdir(parents=True)
        provenance = {"definition_hashes": hashes}
        if identity is not None:
            provenance["stage_identity"] = identity
        _atomic_json(provenance_path, provenance)
    build(staging)
    summary = dict(validate(staging))
    result = {
        "schema_version": 1,
        "state": "completed",
        "definition_hashes": hashes,
        "summary": summary,
        "reused": False,
    }
    if identity is not None:
        result["stage_identity"] = identity
    _atomic_json(staging / "stage.json", result)
    os.replace(staging, destination)
    return result


def _safe_relative_file(root: Path, raw: Any, label: str) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise ValueError(f"{label} must be a stage-root-relative path")
    root = Path(root).resolve()
    path = (root / raw).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its artifact root") from exc
    if not path.is_file():
        raise ValueError(f"{label} is missing: {raw}")
    return path


def _validate_clone_run(
    run: Path,
    seed: int,
    hashes: Mapping[str, str],
    *,
    expected_scenario: ResolvedScenario | None = None,
    require_provenance: bool = True,
    expected_dataset_manifest: Path | None = None,
) -> dict[str, str]:
    import numpy as np

    run = Path(run)
    manifest = _read_json(run / "run.json")
    if manifest.get("schema_version") != 1 or manifest.get("state") != "completed":
        raise ValueError(f"clone seed {seed} run manifest is incomplete")
    actual_seed = manifest.get("model_seed", manifest.get("config", {}).get("model_seed"))
    if actual_seed != seed:
        raise ValueError(f"clone seed {seed} run identity does not match")

    checkpoint = _safe_relative_file(
        run, manifest.get("latest_checkpoint"), f"clone seed {seed} checkpoint"
    )
    try:
        with zipfile.ZipFile(checkpoint) as archive:
            if not archive.namelist() or archive.testzip() is not None:
                raise ValueError(f"clone seed {seed} checkpoint archive is corrupt")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"clone seed {seed} checkpoint archive is unreadable") from exc

    scenario_record = manifest.get("scenario")
    if not isinstance(scenario_record, Mapping):
        raise ValueError(f"clone seed {seed} scenario metadata is missing")
    scenario_path = _safe_relative_file(
        run, scenario_record.get("path"), f"clone seed {seed} scenario"
    )
    frozen_scenario = resolve_scenario(
        environment="tactical-v2", scenario_file=scenario_path, template_id=None,
        enforce_round_cap_minimum=True,
    )
    locked = expected_scenario or validate_definitions()[2]
    if (
        frozen_scenario.canonical_json != locked.canonical_json
        or scenario_record.get("template_id") != locked.template_id
    ):
        raise ValueError(f"clone seed {seed} scenario does not match the locked scenario")

    bc = _read_json(run / "bc.json")
    metrics = _read_json(run / "metrics.json")
    scalar_metrics = {
        "nll", "top1_accuracy", "top3_accuracy", "top5_accuracy",
        "expected_calibration_error", "mean_end_turn_probability",
        "illegal_probability",
    }
    if set(metrics) != scalar_metrics | {"strata"}:
        raise ValueError(f"clone seed {seed} metrics are invalid")
    for name in scalar_metrics:
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"clone seed {seed} metrics are invalid")
    strata = metrics.get("strata")
    if not isinstance(strata, Mapping) or not strata:
        raise ValueError(f"clone seed {seed} metrics are invalid")
    for key, value in strata.items():
        if (
            not isinstance(key, str)
            or "/" not in key
            or not isinstance(value, Mapping)
            or set(value) != {"count", "accuracy"}
            or isinstance(value["count"], bool)
            or not isinstance(value["count"], int)
            or value["count"] <= 0
            or isinstance(value["accuracy"], bool)
            or not isinstance(value["accuracy"], (int, float))
            or not math.isfinite(float(value["accuracy"]))
            or not 0.0 <= float(value["accuracy"]) <= 1.0
        ):
            raise ValueError(f"clone seed {seed} metrics are invalid")
    if (
        bc.get("schema_version") != 1
        or bc.get("model_seed") != seed
        or bc.get("config") != manifest.get("bc_config")
    ):
        raise ValueError(f"clone seed {seed} behavioral-cloning metadata does not match")

    contract = manifest.get("contract")
    required_contract = {
        "version", "contract_hash", "encoding_hash", "observation_size", "action_size",
        "board", "roster", "reward",
    }
    if not isinstance(contract, Mapping) or not required_contract.issubset(contract):
        raise ValueError(f"clone seed {seed} contract is incomplete")
    observation_size = contract.get("observation_size")
    action_size = contract.get("action_size")
    if (
        isinstance(observation_size, bool) or not isinstance(observation_size, int)
        or isinstance(action_size, bool) or not isinstance(action_size, int)
        or observation_size <= 0 or action_size <= 0
    ):
        raise ValueError(f"clone seed {seed} contract geometry is invalid")
    fixtures_path = _safe_relative_file(
        run, "actor-fixtures.npz", f"clone seed {seed} actor fixtures"
    )
    try:
        with np.load(fixtures_path, allow_pickle=False) as fixtures:
            observations = fixtures["observations"]
            legal_masks = fixtures["legal_masks"]
            if (
                observations.dtype != np.float32
                or legal_masks.dtype != np.bool_
                or observations.ndim != 2
                or legal_masks.ndim != 2
                or observations.shape[0] == 0
                or observations.shape != (legal_masks.shape[0], observation_size)
                or legal_masks.shape[1] != action_size
            ):
                raise ValueError(f"clone seed {seed} actor fixtures do not match its contract")
    except (OSError, KeyError, ValueError) as exc:
        if isinstance(exc, ValueError) and "clone seed" in str(exc):
            raise
        raise ValueError(f"clone seed {seed} actor fixtures are unreadable") from exc

    dataset_hash = manifest.get("dataset_manifest_sha256")
    if (
        not isinstance(dataset_hash, str)
        or bc.get("dataset_manifest_sha256") != dataset_hash
    ):
        raise ValueError(f"clone seed {seed} dataset identity does not match")
    if expected_dataset_manifest is not None:
        expected_dataset_manifest = Path(expected_dataset_manifest)
        if (
            not expected_dataset_manifest.is_file()
            or _sha256(expected_dataset_manifest) != dataset_hash
        ):
            raise ValueError(f"clone seed {seed} dataset manifest does not match")

    if require_provenance:
        provenance = _read_json(run / "panel-provenance.json")
        dataset_path_raw = provenance.get("dataset_manifest")
        if not isinstance(dataset_path_raw, str):
            raise ValueError(f"clone seed {seed} dataset provenance is missing")
        dataset_path = Path(dataset_path_raw)
        if not dataset_path.is_file():
            raise ValueError(f"clone seed {seed} dataset manifest is missing")
        if (
            provenance.get("schema_version") != 1
            or provenance.get("model_seed") != seed
            or provenance.get("sampler_seed") != _SAMPLER_SEEDS[str(seed)]
            or provenance.get("definition_hashes") != dict(hashes)
            or provenance.get("dataset_manifest_sha256") != dataset_hash
            or _sha256(dataset_path) != dataset_hash
        ):
            raise ValueError(f"clone seed {seed} definition provenance does not match physical artifacts")

    identity = {
        "contract_hash": contract.get("contract_hash"),
        "encoding_hash": contract.get("encoding_hash"),
    }
    if not all(isinstance(value, str) and len(value) == 64 for value in identity.values()):
        raise ValueError(f"clone seed {seed} contract identity is invalid")
    return identity


def train_clone_runs(
    *,
    dataset: Any,
    scenario: ResolvedScenario,
    env: Any,
    contract: Any,
    spaces_info: Mapping[str, Any],
    output_root: Path,
    panel: Mapping[str, Any],
    definition_hashes: Mapping[str, str],
    trainer: Callable[..., Any] | None = None,
) -> list[Path]:
    from ml_lab.imitation import BehavioralCloningConfig, train_behavioral_clone

    selected_trainer = trainer or train_behavioral_clone
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    bc = panel["behavioral_cloning"]
    dataset_manifest = Path(getattr(dataset, "root", "")) / "manifest.json"
    if not dataset_manifest.is_file():
        raise ValueError("behavioral-cloning dataset manifest is missing")
    runs: list[Path] = []
    for seed in panel["model_seeds"]:
        run = root / f"seed-{seed}"
        pending = root / f".seed-{seed}.staging"
        if run.exists():
            _validate_clone_run(
                run, seed, definition_hashes, expected_scenario=scenario,
                expected_dataset_manifest=dataset_manifest,
            )
            runs.append(run)
            continue
        if not pending.exists():
            config = BehavioralCloningConfig(
                model_seed=seed,
                batch_size=bc["batch_size"],
                learning_rate=bc["learning_rate"],
                max_epochs=bc["max_epochs"],
                patience=bc["patience"],
            )
            result = selected_trainer(
                dataset=dataset,
                scenario=scenario,
                env=env,
                contract=contract,
                spaces_info=dict(spaces_info),
                run_dir=pending,
                config=config,
            )
            if Path(result.run_dir) != pending or not (pending / "run.json").is_file():
                raise ValueError(f"clone seed {seed} did not publish the expected pending run")
        _validate_clone_run(
            pending, seed, definition_hashes, expected_scenario=scenario,
            require_provenance=False, expected_dataset_manifest=dataset_manifest,
        )
        _atomic_json(
            pending / "panel-provenance.json",
            {
                "schema_version": 1,
                "model_seed": seed,
                "sampler_seed": panel["sampler_seeds"][str(seed)],
                "definition_hashes": dict(definition_hashes),
                "dataset_manifest": str(dataset_manifest.resolve()),
                "dataset_manifest_sha256": _sha256(dataset_manifest),
            },
        )
        _validate_clone_run(
            pending, seed, definition_hashes, expected_scenario=scenario,
            expected_dataset_manifest=dataset_manifest,
        )
        os.replace(pending, run)
        runs.append(run)
    return runs


def clone_gate(per_seed_wins: Mapping[int, int]) -> bool:
    rates = {seed: wins / 200 for seed, wins in per_seed_wins.items()}
    return (
        all(rate >= 0.30 for rate in rates.values())
        and sum(per_seed_wins.values()) / 600 >= 0.40
    )


def _failed_gate(errors: Sequence[str], hashes: Mapping[str, str]) -> dict[str, Any]:
    return {
        "schema_version": 1, "state": "failed", "passed": False,
        "definition_hashes": dict(hashes), "per_seed_wins": {},
        "pooled_win_rate": 0.0, "integrity_errors": list(errors), "clones": [],
    }


def _clone_metadata(
    clone_runs: Sequence[Path], hashes: Mapping[str, str]
) -> tuple[dict[int, Path], dict[str, str], list[str]]:
    by_seed: dict[int, Path] = {}
    expected_contract: dict[str, str] | None = None
    errors: list[str] = []
    locked = validate_definitions()[2]
    for raw_run in clone_runs:
        run = Path(raw_run)
        try:
            manifest = _read_json(run / "run.json")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        seed = manifest.get("model_seed", manifest.get("config", {}).get("model_seed"))
        if seed not in _MODEL_SEEDS or seed in by_seed:
            errors.append(f"clone run has duplicate or unexpected model seed {seed!r}")
            continue
        by_seed[seed] = run
        try:
            identity = _validate_clone_run(
                run, seed, hashes, expected_scenario=locked
            )
        except ValueError as exc:
            errors.append(str(exc))
            continue
        if expected_contract is None:
            expected_contract = identity
        elif identity != expected_contract:
            errors.append(f"clone seed {seed} contract does not match the panel")
    if set(by_seed) != set(_MODEL_SEEDS):
        errors.append("clone runs do not contain exactly model seeds 211, 223, and 227")
    return by_seed, expected_contract or {}, errors


def _evaluate_standard_controllers(
    p0: str,
    p1: str,
    *,
    games: int,
    seed_start: int,
    both_seats: bool,
    workers: int,
    server_cmd: Sequence[str],
    output_path: Path | None,
    environment: str | None,
    evidence_dir: Path | None,
    capture_trace: bool,
    start_profile: str,
    client_factory: Callable[[int], Any] | None = None,
) -> dict[str, Any]:
    """Resolve controllers normally and force the locked profile at reset."""
    from ml_lab.controllers import ControllerResolver, _validate_contract_compatibility
    from ml_lab.evaluation import DuelClient, evaluate_matchup

    resolver = ControllerResolver()
    candidate = resolver.resolve(p0)
    opponent = resolver.resolve(p1)
    _validate_contract_compatibility(candidate.contract, opponent.contract)
    for controller in (candidate, opponent):
        if (
            environment is not None
            and controller.model is not None
            and controller.contract is not None
            and controller.contract.version != environment
        ):
            raise ValueError("controller contract does not match the explicit environment")
    selected_environment = environment or next(
        (
            controller.contract.version
            for controller in (candidate, opponent)
            if controller.contract is not None
        ),
        "tactical-v1",
    )
    factory = client_factory or (
        lambda _index: DuelClient(server_cmd, environment=selected_environment)
    )
    return evaluate_matchup(
        candidate, opponent, games=games, seed_start=seed_start,
        both_seats=both_seats, workers=workers, client_factory=factory,
        output_path=output_path, start_profile=start_profile,
        evidence_dir=evidence_dir, capture_trace=capture_trace,
    )


def _relativize_evaluation(
    result: Mapping[str, Any], output_root: Path, output_path: Path
) -> dict[str, Any]:
    root = Path(output_root).resolve()
    normalized = dict(result)
    raw_matches = normalized.get("matches")
    if not isinstance(raw_matches, list):
        return normalized
    matches: list[Any] = []
    for raw_match in raw_matches:
        if not isinstance(raw_match, Mapping):
            matches.append(raw_match)
            continue
        match = dict(raw_match)
        for field in ("trace_path", "replay_path"):
            raw = match.get(field)
            if raw is None:
                continue
            if not isinstance(raw, str):
                raise ValueError(f"evaluation {field} is invalid")
            supplied = Path(raw)
            path = (supplied if supplied.is_absolute() else root / supplied).resolve()
            try:
                relative = path.relative_to(root)
            except ValueError as exc:
                raise ValueError(f"evaluation {field} escapes the gate artifact") from exc
            if not path.is_file():
                raise ValueError(f"evaluation {field} is missing")
            match[field] = relative.as_posix()
        matches.append(match)
    normalized["matches"] = matches
    _atomic_json(output_path, normalized)
    return normalized


def _artifact_exists(root: Path, match: Mapping[str, Any], field: str) -> bool:
    try:
        _safe_relative_file(root, match.get(field), f"evaluation {field}")
    except ValueError:
        return False
    return True


def evaluate_clone_gate(
    clone_runs: Sequence[Path],
    *,
    output_dir: Path = CLONE_EVALUATION_PATH,
    server_cmd: Sequence[str] | None = None,
    workers: int = 1,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    from ml_lab.cli import DEFAULT_SERVER

    panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
    runs, expected_contract, errors = _clone_metadata(clone_runs, hashes)
    if errors:
        return _failed_gate(errors, hashes)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gate_scenario = _materialize_runtime_scenario(scenario, output_root, hashes)
    command = list(server_cmd) if server_cmd is not None else ["dotnet", str(DEFAULT_SERVER)]
    command.extend(["--scenario-file", str(gate_scenario)])
    selected_evaluator = evaluator or _evaluate_standard_controllers
    start = panel["clone_gate"]["seed_start"]
    stop = start + panel["clone_gate"]["maps"]
    expected_keys = {(map_seed, seat) for map_seed in range(start, stop) for seat in (0, 1)}

    clones: list[dict[str, Any]] = []
    per_seed_wins: dict[int, int] = {}
    for model_seed in panel["model_seeds"]:
        run = runs[model_seed]
        matches: list[dict[str, Any]] = []
        seen: set[tuple[int, int]] = set()
        for map_seed in range(start, stop):
            map_root = output_root / f"seed-{model_seed}" / f"map-{map_seed}"
            output_path = map_root / "evaluation.json"
            result = selected_evaluator(
                f"run:{run}", panel["clone_gate"]["opponent"], games=1,
                seed_start=map_seed, both_seats=True, workers=workers,
                server_cmd=command, output_path=output_path,
                environment="tactical-v2", evidence_dir=map_root / "evidence",
                capture_trace=True, start_profile=panel["clone_gate"]["profile"],
            )
            result = _relativize_evaluation(result, output_root, output_path)
            candidate = result.get("candidate")
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("contract_hash") != expected_contract.get("contract_hash")
                or candidate.get("encoding_hash") != expected_contract.get("encoding_hash")
            ):
                errors.append(f"clone seed {model_seed} evaluation contract does not match")
            if (
                result.get("reciprocal") is not True
                or result.get("seeds") != [map_seed]
                or result.get("games") != 2
                or result.get("schedule") != {
                    "start_profile": panel["clone_gate"]["profile"],
                    "reference_seat_policy": "candidate-seat",
                }
            ):
                errors.append(f"clone seed {model_seed} map {map_seed} schedule is not reciprocal")
            raw_matches = result.get("matches")
            if not isinstance(raw_matches, list):
                errors.append(f"clone seed {model_seed} map {map_seed} matches are missing")
                continue
            for raw_match in raw_matches:
                if not isinstance(raw_match, Mapping):
                    errors.append(f"clone seed {model_seed} has a malformed match")
                    continue
                match = dict(raw_match)
                key = (match.get("seed"), match.get("candidate_seat"))
                if key not in expected_keys or key in seen:
                    errors.append(
                        f"clone seed {model_seed} has duplicate or unexpected seed/seat {key!r}"
                    )
                else:
                    seen.add(key)
                outcome = match.get("outcome")
                if outcome not in {"win", "loss", "draw"}:
                    errors.append(f"clone seed {model_seed} has an invalid outcome")
                if outcome in {"loss", "draw"} and (
                    not _artifact_exists(output_root, match, "trace_path")
                    or not _artifact_exists(output_root, match, "replay_path")
                ):
                    errors.append(f"clone seed {model_seed} loss/draw evidence is missing")
                matches.append(match)
        if seen != expected_keys:
            errors.append(
                f"clone seed {model_seed} does not contain the exact 100-map reciprocal schedule"
            )
        wins = sum(match.get("outcome") == "win" for match in matches)
        per_seed_wins[model_seed] = wins
        clones.append(
            {
                "model_seed": model_seed, "run_path": str(run.resolve()),
                "contract": expected_contract, "wins": wins,
                "games": len(matches), "matches": matches,
            }
        )

    pooled = sum(per_seed_wins.values()) / 600
    passed = not errors and clone_gate(per_seed_wins)
    aggregate = {
        "schema_version": 1, "state": "completed" if passed else "failed",
        "passed": passed, "definition_hashes": hashes,
        "schedule": {
            "seed_start": 16_000_000, "maps": 100, "both_seats": True,
            "profile": "standard-3v3", "opponent": "random",
        },
        "per_seed_wins": per_seed_wins, "pooled_win_rate": pooled,
        "integrity_errors": errors, "clones": clones,
    }
    _atomic_json(output_root / "gate.json", aggregate)
    return aggregate


def _validate_collection_dataset(
    root: Path,
    contract: Any,
    scenario: ResolvedScenario,
    execution_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    from ml_lab.imitation import load_imitation_dataset

    _validate_dataset_execution_identity(root, execution_identity)
    dataset = load_imitation_dataset(Path(root), expected_contract=contract)
    scenario_hash = hashlib.sha256(scenario.canonical_json.encode("utf-8")).hexdigest()
    pairs: dict[tuple[Any, ...], set[int]] = {}
    standard_decisions = 0
    conversion_decisions = 0
    validation_maps: dict[str, set[int]] = {}
    for game in dataset.games:
        if game["scenario_hash"] != scenario_hash:
            raise ValueError("collection scenario hash does not match the locked scenario")
        partition = game["partition"]
        teacher = game["teacher"]
        profile = game["profile"]
        seed = game["seed"]
        if partition == "train" and teacher == "greedy" and profile == "standard-3v3":
            bank = _BANKS["greedy_demonstrations"]
            standard_decisions += game["row_count"]
        elif partition == "train" and teacher == "bounded-search" and profile in _CONVERSION_PROFILES:
            bank = _BANKS["search_demonstrations"]
            conversion_decisions += game["row_count"]
        elif partition == "validation" and (
            (teacher == "greedy" and profile == "standard-3v3")
            or (teacher == "bounded-search" and profile in _CONVERSION_PROFILES)
        ):
            bank = _BANKS["bc_validation"]
            validation_maps.setdefault(profile, set()).add(seed)
        else:
            raise ValueError("collection game source/profile is outside the locked protocol")
        if not bank["start"] <= seed <= bank["stop"]:
            raise ValueError("collection game seed is outside its locked namespace")
        key = (partition, teacher, profile, seed)
        pairs.setdefault(key, set()).add(game["teacher_seat"])
    if any(seats != {0, 1} for seats in pairs.values()):
        raise ValueError("collection contains a missing reciprocal seat")
    if standard_decisions < _COLLECTION["standard_decisions_minimum"]:
        raise ValueError("collection has too few standard decisions")
    if conversion_decisions < _COLLECTION["conversion_decisions_minimum"]:
        raise ValueError("collection has too few conversion decisions")
    expected_validation = {"standard-3v3": 100, **{profile: 20 for profile in _CONVERSION_PROFILES}}
    actual_validation = {profile: len(seeds) for profile, seeds in validation_maps.items()}
    if actual_validation != expected_validation:
        raise ValueError("collection validation map counts do not match the locked protocol")
    return {
        "games": len(dataset.games),
        "standard_decisions": standard_decisions,
        "conversion_decisions": conversion_decisions,
        "validation_maps": sum(actual_validation.values()),
    }


def _validate_clone_stage(root: Path, hashes: Mapping[str, str]) -> Mapping[str, Any]:
    runs, _contract, errors = _clone_metadata(
        [Path(root) / f"seed-{seed}" for seed in _MODEL_SEEDS], hashes
    )
    if errors:
        raise ValueError("; ".join(errors))
    if set(runs) != set(_MODEL_SEEDS):
        raise ValueError("clone stage outputs are incomplete")
    return {"clone_runs": len(runs), "model_seeds": list(_MODEL_SEEDS)}


def _validate_gate_stage(root: Path, hashes: Mapping[str, str]) -> Mapping[str, Any]:
    root = Path(root)
    gate = _read_json(root / "gate.json")
    _panel, _banks, scenario = validate_definitions()
    _validate_runtime_scenario(root, scenario, hashes)
    if gate.get("definition_hashes") != dict(hashes):
        raise ValueError("clone gate definition hashes do not match")
    clones = gate.get("clones")
    if not isinstance(clones, list) or len(clones) != 3:
        raise ValueError("clone gate expected outputs are incomplete")
    expected = {
        (root / f"seed-{seed}" / f"map-{map_seed}" / "evaluation.json").resolve()
        for seed in _MODEL_SEEDS
        for map_seed in range(_CLONE_GATE["seed_start"], _CLONE_GATE["seed_start"] + _CLONE_GATE["maps"])
    }
    actual = {path.resolve() for path in root.glob("seed-*/map-*/evaluation.json")}
    if actual != expected:
        raise ValueError("clone gate evaluation manifests are missing or unexpected")
    recomputed: dict[int, int] = {}
    for clone in clones:
        if not isinstance(clone, Mapping):
            raise ValueError("clone gate clone result is malformed")
        seed = clone.get("model_seed")
        run_path = clone.get("run_path")
        if seed not in _MODEL_SEEDS or seed in recomputed or not isinstance(run_path, str):
            raise ValueError("clone gate model-seed results are incomplete")
        identity = _validate_clone_run(Path(run_path), seed, hashes, expected_scenario=scenario)
        if clone.get("contract") != identity:
            raise ValueError(f"clone seed {seed} gate contract identity does not match")
        physical: list[dict[str, Any]] = []
        for map_seed in range(_CLONE_GATE["seed_start"], _CLONE_GATE["seed_start"] + _CLONE_GATE["maps"]):
            path = root / f"seed-{seed}" / f"map-{map_seed}" / "evaluation.json"
            try:
                evaluation = _read_json(path)
            except ValueError as exc:
                raise ValueError(f"clone gate evaluation is unreadable: {path}") from exc
            candidate = evaluation.get("candidate")
            opponent = evaluation.get("opponent")
            if (
                not isinstance(candidate, Mapping)
                or candidate.get("contract_hash") != identity["contract_hash"]
                or candidate.get("encoding_hash") != identity["encoding_hash"]
                or not isinstance(opponent, Mapping)
                or opponent.get("name") != _CLONE_GATE["opponent"]
            ):
                raise ValueError(f"clone seed {seed} map {map_seed} evaluation identity is invalid")
            if (
                evaluation.get("seed_start") != map_seed
                or evaluation.get("seeds") != [map_seed]
                or evaluation.get("reciprocal") is not True
                or evaluation.get("games") != 2
                or evaluation.get("schedule") != {
                    "start_profile": _CLONE_GATE["profile"],
                    "reference_seat_policy": "candidate-seat",
                }
            ):
                raise ValueError(f"clone seed {seed} map {map_seed} evaluation schedule is invalid")
            matches = evaluation.get("matches")
            if not isinstance(matches, list) or len(matches) != 2:
                raise ValueError(f"clone seed {seed} map {map_seed} evaluation matches are invalid")
            keys: set[tuple[Any, Any]] = set()
            for raw in matches:
                if not isinstance(raw, Mapping):
                    raise ValueError(f"clone seed {seed} evaluation match is malformed")
                match = dict(raw)
                keys.add((match.get("seed"), match.get("candidate_seat")))
                if match.get("outcome") not in {"win", "loss", "draw"}:
                    raise ValueError(f"clone seed {seed} evaluation outcome is invalid")
                if match["outcome"] in {"loss", "draw"} and (
                    not _artifact_exists(root, match, "trace_path")
                    or not _artifact_exists(root, match, "replay_path")
                ):
                    raise ValueError(f"clone seed {seed} evaluation evidence is missing")
                physical.append(match)
            if keys != {(map_seed, 0), (map_seed, 1)}:
                raise ValueError(f"clone seed {seed} map {map_seed} evaluation seats are invalid")
        wins = sum(match["outcome"] == "win" for match in physical)
        recomputed[seed] = wins
        if clone.get("games") != len(physical) or clone.get("wins") != wins or clone.get("matches") != physical:
            raise ValueError(f"clone seed {seed} gate summary differs from recomputed evidence")
    pooled = sum(recomputed.values()) / 600
    passed = clone_gate(recomputed)
    if (
        gate.get("per_seed_wins") != {str(seed): wins for seed, wins in recomputed.items()}
        or gate.get("pooled_win_rate") != pooled
        or gate.get("passed") is not passed
        or gate.get("state") != ("completed" if passed else "failed")
        or gate.get("integrity_errors") != []
    ):
        raise ValueError("clone gate aggregate differs from recomputed physical evidence")
    if not passed:
        raise ValueError("clone gate did not pass recomputed thresholds")
    return {"clone_runs": 3, "development_games": 600}


def _server_command(scenario_path: Path) -> list[str]:
    server = (
        PROJECT_ROOT / "engine" / "HexWars.GymServer"
        / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
    )
    return ["dotnet", str(server), "--scenario-file", str(scenario_path)]


def _collect_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    from collect_annihilation_demonstrations import CollectionSpec, collect_partition
    from ml_lab.evaluation import DuelClient

    _panel, _banks, scenario, hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    probe_root = Path(tempfile.mkdtemp(prefix=".collect-runtime-"))
    try:
        probe_scenario = _materialize_runtime_scenario(scenario, probe_root, hashes)
        probe_command = _server_command(probe_scenario)
        probe = DuelClient(probe_command, environment="tactical-v2")
        try:
            contract = probe.contract
        finally:
            probe.close()
    finally:
        shutil.rmtree(probe_root, ignore_errors=True)

    def build(staging: Path) -> None:
        stage_scenario = _materialize_runtime_scenario(scenario, staging, hashes)
        stage_command = _server_command(stage_scenario)
        factory = lambda _worker: DuelClient(stage_command, environment="tactical-v2")
        collect_partition(
            CollectionSpec(
                staging, "train",
                hashlib.sha256(scenario.canonical_json.encode("utf-8")).hexdigest(),
                contract, factory,
            )
        )
        collect_partition(
            CollectionSpec(
                staging, "validation",
                hashlib.sha256(scenario.canonical_json.encode("utf-8")).hexdigest(),
                contract, factory,
            )
        )

    validate = _full_stage_validator(
        lambda root: _validate_collection_dataset(
            root, contract, scenario, identity,
        ),
        expected_identity=identity,
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return run_atomic_stage(
        DATASET_PATH, hashes, stage_identity=identity, build=build,
        validate=validate,
    )


def _validate_bc_source_target_compatibility(
    source_contract: Any,
    target_contract: Any,
) -> None:
    from ml_lab.controllers import (
        ControllerResolutionError,
        _validate_contract_compatibility,
    )

    if source_contract is None or target_contract is None:
        raise ValueError("behavioral-cloning source or target contract is missing")
    try:
        _validate_contract_compatibility(source_contract, target_contract)
    except ControllerResolutionError as exc:
        raise ValueError(
            "behavioral-cloning source contract is incompatible with target policy"
        ) from exc


def _train_bc_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    from hexwars_gym import HexWarsEnv
    from ml_lab.evaluation import DuelClient
    from ml_lab.imitation import load_imitation_dataset

    panel, _banks, scenario, hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    _validate_dataset_execution_identity(DATASET_PATH, identity)

    def build(staging: Path) -> None:
        stage_scenario = _materialize_runtime_scenario(scenario, staging, hashes)
        stage_command = _server_command(stage_scenario)
        source_probe = DuelClient(stage_command, environment="tactical-v2")
        try:
            source_contract = source_probe.contract
            env = HexWarsEnv(
                stage_command[:-2],
                opponent="random", seat=0, base_seed=12_000_000,
                environment="tactical-v2", scenario_path=stage_scenario,
            )
            try:
                dataset = load_imitation_dataset(
                    DATASET_PATH, expected_contract=source_contract,
                )
                _validate_bc_source_target_compatibility(
                    source_contract, env.contract,
                )
                train_clone_runs(
                    dataset=dataset, scenario=scenario, env=env,
                    contract=source_contract, spaces_info=env.spaces_info,
                    output_root=staging, panel=panel,
                    definition_hashes=hashes,
                )
            finally:
                env.close()
        finally:
            source_probe.close()

    validate = _full_stage_validator(
        lambda root: _validate_clone_stage(root, hashes),
        expected_identity=identity,
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return run_atomic_stage(
        CLONE_RUNS_PATH, hashes, stage_identity=identity, build=build,
        validate=validate,
    )


def _evaluate_bc_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    _panel, _banks, _scenario, hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    clone_runs = [CLONE_RUNS_PATH / f"seed-{seed}" for seed in _MODEL_SEEDS]
    _validate_clone_stage(CLONE_RUNS_PATH, hashes)

    def build(staging: Path) -> None:
        evaluate_clone_gate(
            clone_runs, output_dir=staging,
            server_cmd=_server_command(SCENARIO_PATH)[:-2],
        )

    validate = _full_stage_validator(
        lambda root: _validate_gate_stage(root, hashes),
        expected_identity=identity,
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return run_atomic_stage(
        CLONE_EVALUATION_PATH, hashes, stage_identity=identity, build=build,
        validate=validate,
    )


def build_smoke_schedule() -> dict[str, Any]:
    return {
        "collection": {
            "partition": "train",
            "standard_pairs": 1,
            "conversion_pairs": 6,
            "reciprocal_pairs": 7,
            "games": 14,
            "profiles": [
                "standard-3v3",
                "conversion-3v1-near",
                "conversion-3v1-far",
                "conversion-2v1-near",
                "conversion-2v1-far",
                "conversion-1v1-near",
                "conversion-1v1-far",
            ],
        },
        "behavioral_cloning": {
            "model_seed": 211,
            "batch_size": 32,
            "max_epochs": 1,
            "patience": 1,
            "validation_source": "training-rows-reused-not-held-out",
        },
        "ppo": {
            "run_name": "initialized-ppo",
            "model_seed": 211,
            "episode_seed_base": 18_100_000,
            "workers": 1,
            "total_timesteps": 2,
            "checkpoint_interval": 2,
            "completed_rollout_size": 2,
        },
        "evaluation": {
            "seed_start": 18_200_000,
            "maps": 2,
            "both_seats": True,
            "games": 4,
            "profile": "standard-3v3",
        },
    }


def _validate_smoke_repository_identity(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("smoke repository identity is missing")
    identity = dict(raw)
    if set(identity) != {
        "schema_version", "commit", "source_tree", "dirty", "policy",
    }:
        raise ValueError("smoke repository identity schema is invalid")
    policy = identity.get("policy")
    expected_policy = {
        "required_clean": True,
        "ignored_generated_root": _SMOKE_GENERATED_ROOT,
    }
    if policy != expected_policy:
        raise ValueError("smoke repository identity policy is invalid")
    for field in ("commit", "source_tree"):
        digest = identity.get(field)
        if (
            not isinstance(digest, str)
            or len(digest) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("smoke repository source digest is invalid")
    if identity.get("schema_version") != 1 or identity.get("dirty") is not False:
        raise ValueError("authoritative smoke requires a clean repository identity")
    identity["policy"] = dict(expected_policy)
    return identity


def _smoke_repository_identity(repository: Path = PROJECT_ROOT) -> dict[str, Any]:
    repository = Path(repository)
    revision, dirty = _repository_identity(repository)
    source_tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return _validate_smoke_repository_identity({
        "schema_version": 1,
        "commit": revision,
        "source_tree": source_tree,
        "dirty": dirty,
        "policy": {
            "required_clean": True,
            "ignored_generated_root": _SMOKE_GENERATED_ROOT,
        },
    })


def build_smoke_manifest(
    *, checks: Mapping[str, Any], artifacts: Mapping[str, str],
    repository_identity: Mapping[str, Any],
) -> dict[str, Any]:
    schedule = build_smoke_schedule()
    values = dict(checks)
    identity = _validate_smoke_repository_identity(repository_identity)
    integer_fields = {
        "reciprocal_pairs", "games", "teacher_labels", "masked_labels",
        "round_trip_mismatches", "replay_mismatches", "ppo_timesteps",
        "ppo_completed_rollouts", "evaluation_games",
    }
    actor_error = values.get("actor_fixture_max_error")
    required = {
        "reciprocal_pairs",
        "games",
        "teacher_labels",
        "masked_labels",
        "round_trip_mismatches",
        "replay_mismatches",
        "actor_fixture_max_error",
        "actor_fixture_device",
        "ppo_timesteps",
        "ppo_completed_rollouts",
        "evaluation_games",
        "checkpoint_reloaded",
    }
    ppo = schedule["ppo"]
    if (
        any(type(values.get(field)) is not int for field in integer_fields)
        or isinstance(actor_error, bool)
        or not isinstance(actor_error, (int, float))
        or not math.isfinite(float(actor_error))
        or set(values) != required
        or values["reciprocal_pairs"] != schedule["collection"]["reciprocal_pairs"]
        or values["games"] != schedule["collection"]["games"]
        or type(values["teacher_labels"]) is not int
        or values["teacher_labels"] <= 0
        or values["masked_labels"] != 0
        or values["round_trip_mismatches"] != 0
        or values["replay_mismatches"] != 0
        or actor_error != 0
        or values["actor_fixture_device"] != "cpu"
        or type(values["ppo_timesteps"]) is not int
        or values["ppo_timesteps"] < ppo["completed_rollout_size"]
        or type(values["ppo_completed_rollouts"]) is not int
        or values["ppo_completed_rollouts"] < 1
        or values["evaluation_games"] != schedule["evaluation"]["games"]
        or values["checkpoint_reloaded"] is not True
    ):
        raise ValueError("smoke checks do not satisfy the end-to-end gate")
    physical = dict(artifacts)
    if not physical:
        raise ValueError("smoke artifacts are incomplete")
    for raw, digest in physical.items():
        path = Path(raw)
        if (
            not isinstance(raw, str)
            or not raw
            or path.is_absolute()
            or ".." in path.parts
            or not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("smoke artifacts are invalid")
    return {
        "schema_version": 1,
        "state": "completed",
        "schedule": schedule,
        "repository_identity": identity,
        "checks": values,
        "artifacts": physical,
    }


def run_smoke_stage(
    destination: Path,
    definition_hashes: Mapping[str, str],
    *,
    stage_identity: Mapping[str, Any] | None = None,
    build: Callable[[Path], None],
    validate: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    destination = Path(destination).resolve()
    protected = (
        DATASET_PATH, CLONE_RUNS_PATH, CLONE_EVALUATION_PATH,
        PPO_RUNS_PATH, DEVELOPMENT_PATH, SELECTION_PATH,
        PANEL_ROOT / "final-seal.json",
        PANEL_ROOT / "final-evaluation.json",
        PANEL_ROOT / ".final-evaluation.pending",
        PANEL_ROOT / "final-publication.json",
        PANEL_ROOT / ".final-generations",
    )
    for raw in protected:
        full = Path(raw).resolve()
        if destination == full or destination in full.parents or full in destination.parents:
            raise ValueError("smoke root overlaps a full experiment artifact")
    try:
        return run_atomic_stage(
            destination, definition_hashes, stage_identity=stage_identity,
            build=build, validate=validate,
        )
    except BaseException as error:
        completion = (
            destination.with_name(f".{destination.name}.staging") / "smoke.json"
        )
        try:
            completion.unlink(missing_ok=True)
        except OSError as cleanup_error:
            error.add_note(
                f"could not remove staged smoke completion manifest: {cleanup_error}"
            )
        raise


def _validate_smoke_evaluation_matches(
    matches: Any, *, seed_start: int, maps: int,
) -> None:
    expected = {
        (seed, seat)
        for seed in range(seed_start, seed_start + maps)
        for seat in (0, 1)
    }
    if not isinstance(matches, list) or len(matches) != maps * 2:
        raise ValueError("smoke evaluation matches are incomplete")
    keys = []
    for row in matches:
        if (
            not isinstance(row, Mapping)
            or type(row.get("seed")) is not int
            or type(row.get("candidate_seat")) is not int
        ):
            raise ValueError("smoke evaluation matches have invalid identities")
        keys.append((row["seed"], row["candidate_seat"]))
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("smoke evaluation matches are incomplete")


def _validate_smoke_stage(
    root: Path,
    expected_repository_identity: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    root = Path(root)
    manifest = _read_json(root / "smoke.json")
    checks = manifest.get("checks")
    artifacts = manifest.get("artifacts")
    identity = _validate_smoke_repository_identity(
        manifest.get("repository_identity")
    )
    if (
        expected_repository_identity is not None
        and identity != dict(expected_repository_identity)
    ):
        raise ValueError("smoke repository identity changed")
    if not isinstance(checks, Mapping) or not isinstance(artifacts, Mapping):
        raise ValueError("smoke manifest evidence is incomplete")
    expected = build_smoke_manifest(
        checks=checks, artifacts=artifacts,
        repository_identity=identity,
    )
    if manifest != expected:
        raise ValueError("smoke manifest schema or schedule is invalid")
    for relative, digest in artifacts.items():
        path = _safe_relative_file(root, relative, "smoke artifact")
        if _sha256(path) != digest:
            raise ValueError(f"smoke artifact hash changed: {relative}")
    return dict(checks)


def _published_smoke_root(root: Path) -> Path:
    root = Path(root).resolve()
    name = root.name
    if name.startswith(".") and name.endswith(".staging"):
        return root.with_name(name[1:-len(".staging")])
    return root


def _smoke_recorded_file(root: Path, raw: Any, label: str) -> Path:
    root = Path(root).resolve()
    published = _published_smoke_root(root)
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} path is invalid")
    supplied = Path(raw)
    if not supplied.is_absolute():
        return _safe_relative_file(root, raw, label)
    path = supplied.resolve()
    try:
        relative = path.relative_to(published)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes the smoke root") from exc
    return _safe_relative_file(root, relative.as_posix(), label)


def _smoke_contract(raw: Any) -> Any:
    from ml_lab.contracts import EnvironmentContract

    if not isinstance(raw, Mapping):
        raise ValueError("smoke PPO contract is missing")
    value = dict(raw)
    environment = value.pop("environment", None)
    try:
        contract = EnvironmentContract(**value)
    except (TypeError, ValueError) as exc:
        raise ValueError("smoke PPO contract is invalid") from exc
    if environment != contract.environment:
        raise ValueError("smoke PPO contract environment is invalid")
    return contract



def _load_smoke_source_dataset(
    dataset_root: Path,
    bc_manifest: Mapping[str, Any],
) -> tuple[Any, Any]:
    from ml_lab.imitation import load_imitation_dataset

    dataset_root = Path(dataset_root)
    source_contract = _smoke_contract(bc_manifest.get("contract"))
    dataset_manifest = _read_json(dataset_root / "manifest.json")
    if (
        dataset_manifest.get("contract_hash") != source_contract.contract_hash
        or dataset_manifest.get("encoding_hash") != source_contract.encoding_hash
    ):
        raise ValueError("smoke dataset manifest does not match the BC source contract")
    dataset = load_imitation_dataset(
        dataset_root, expected_contract=source_contract,
    )
    if dataset.contract != source_contract:
        raise ValueError("smoke dataset did not reopen with the exact BC source contract")
    return dataset, source_contract


def _validate_smoke_actor_source(
    *,
    source_contract: Any,
    target_contract: Any,
    resolved_contract: Any,
    initialization: Mapping[str, Any],
) -> None:
    _validate_bc_source_target_compatibility(source_contract, target_contract)
    if resolved_contract != source_contract:
        raise ValueError("smoke actor resolver did not return the exact source contract")
    if (
        initialization.get("source_contract_hash") != source_contract.contract_hash
        or initialization.get("source_encoding_hash") != source_contract.encoding_hash
    ):
        raise ValueError("smoke actor provenance does not match the source contract")


def _reuse_completed_smoke_bc(
    run: Path,
    dataset_manifest: Path,
    scenario: ResolvedScenario,
    definition_hashes: Mapping[str, str],
    contract: Any,
    config: Any,
) -> bool:
    run = Path(run)
    if not run.exists():
        return False
    manifest = _read_json(run / "run.json")
    expected_config = {
        "model_seed": config.model_seed,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("state") != "completed"
        or manifest.get("bc_config") != expected_config
        or manifest.get("contract") != contract.to_dict()
    ):
        raise ValueError("completed smoke BC config or contract does not match")
    identity = _validate_clone_run(
        run,
        config.model_seed,
        definition_hashes,
        expected_scenario=scenario,
        require_provenance=False,
        expected_dataset_manifest=dataset_manifest,
    )
    if identity != {
        "contract_hash": contract.contract_hash,
        "encoding_hash": contract.encoding_hash,
    }:
        raise ValueError("completed smoke BC contract identity does not match")
    return True


def _run_restart_safe_smoke_training(
    config: Any,
    *,
    runs_root: Path,
    recovery_root: Path,
    trainer: Callable[..., Path],
    allowed_actor_init_sources: Sequence[str] = (),
    **trainer_kwargs: Any,
) -> Path:
    runs_root = Path(runs_root).resolve()
    recovery_root = Path(recovery_root).resolve()
    run = runs_root / config.run_name
    try:
        recovery_root.relative_to(runs_root.parent)
    except ValueError:
        pass
    else:
        raise ValueError("smoke recovery root must be outside the publishable stage")

    expected_config = config.to_dict()
    if run.exists():
        manifest = _read_json(run / "run.json")
        actual_config = manifest.get("config")
        matches = actual_config == expected_config
        if not matches and isinstance(actual_config, Mapping):
            actual_normalized = dict(actual_config)
            expected_normalized = dict(expected_config)
            actual_source = actual_normalized.pop("actor_init_source", None)
            expected_source = expected_normalized.pop("actor_init_source", None)
            matches = (
                actual_normalized == expected_normalized
                and actual_source in {expected_source, *allowed_actor_init_sources}
            )
        if (
            manifest.get("schema_version") != 1
            or not matches
            or manifest.get("state") not in {"completed", "failed"}
        ):
            raise ValueError("existing smoke PPO attempt is not safely reusable")
        if manifest["state"] == "completed":
            return run

        recovery_root.mkdir(parents=True, exist_ok=True)
        attempt = 0
        while True:
            archived = recovery_root / f"{config.run_name}-attempt-{attempt:04d}"
            if not archived.exists():
                break
            attempt += 1
        os.replace(run, archived)

    result = Path(trainer(
        config, runs_root=runs_root, **trainer_kwargs,
    )).resolve()
    if result != run or _read_json(run / "run.json").get("state") != "completed":
        raise ValueError("smoke PPO trainer did not publish the completed canonical run")
    return run


def _smoke_artifact_hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    ignored = {"smoke.json", "stage.json", ".stage-definitions.json"}
    artifacts = {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and path.relative_to(root).as_posix() not in ignored
        and ".publishing-" not in path.as_posix()
    }
    if not artifacts:
        raise ValueError("smoke produced no physical artifacts")
    return artifacts


def _collect_smoke_evidence(
    root: Path,
    *, expected_repository_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    from ml_lab.controllers import ControllerResolver
    from ml_lab.imitation import audit_imitation_dataset

    root = Path(root).resolve()
    identity = _validate_smoke_repository_identity(expected_repository_identity)

    schedule = build_smoke_schedule()
    ppo_run = root / "ppo" / schedule["ppo"]["run_name"]
    ppo_manifest = _read_json(ppo_run / "run.json")
    contract = _smoke_contract(ppo_manifest.get("contract"))

    dataset_root = root / "dataset"
    bc_root = root / "bc"
    dataset_manifest = _read_json(dataset_root / "manifest.json")
    if (
        dataset_manifest.get("code_revision") != identity["commit"]
        or dataset_manifest.get("dirty") is not False
    ):
        raise ValueError("smoke dataset repository identity does not match")

    bc_manifest = _read_json(bc_root / "run.json")
    dataset, source_contract = _load_smoke_source_dataset(
        dataset_root, bc_manifest,
    )
    audit = audit_imitation_dataset(dataset)
    pairs: dict[tuple[str, str, str, int], set[int]] = {}
    for game in dataset.games:
        key = (
            game["partition"], game["teacher"], game["profile"], game["seed"],
        )
        pairs.setdefault(key, set()).add(game["teacher_seat"])
    expected_pairs = {
        ("train", "greedy", "standard-3v3", 11_000_000),
        *{
            ("train", "bounded-search", profile, 11_500_000 + index)
            for index, profile in enumerate(schedule["collection"]["profiles"][1:])
        },
    }
    if (
        set(pairs) != expected_pairs
        or any(seats != {0, 1} for seats in pairs.values())
        or audit["games"] != schedule["collection"]["games"]
    ):
        raise ValueError("smoke collection schedule is incomplete")

    dataset_digest = _sha256(dataset_root / "manifest.json")
    bc = _read_json(bc_root / "bc.json")
    _read_json(bc_root / "metrics.json")
    if (
        bc_manifest.get("state") != "completed"
        or bc_manifest.get("latest_checkpoint_step") != 0
        or bc_manifest.get("dataset_manifest_sha256") != dataset_digest
        or bc.get("dataset_manifest_sha256") != dataset_digest
        or bc.get("training_decision_count") != audit["teacher_labels"]
        or bc.get("validation_decision_count") != audit["teacher_labels"]
        or bc.get("validation_game_count") != audit["games"]
    ):
        raise ValueError("smoke behavioral clone evidence is invalid")

    ppo_config = ppo_manifest.get("config")
    ppo_schedule = schedule["ppo"]
    if (
        ppo_manifest.get("state") != "completed"
        or not isinstance(ppo_config, Mapping)
        or ppo_config.get("run_name") != ppo_schedule["run_name"]
        or ppo_config.get("seed") != ppo_schedule["model_seed"]
        or ppo_config.get("device") != "cpu"
        or ppo_config.get("workers") != ppo_schedule["workers"]
        or ppo_config.get("total_timesteps") != ppo_schedule["total_timesteps"]
        or ppo_config.get("checkpoint_interval") != ppo_schedule["checkpoint_interval"]
        or ppo_config.get("resume_source") is not None
        or Path(str(ppo_config.get("actor_init_source", ""))).resolve()
        != (_published_smoke_root(root) / "bc").resolve()
    ):
        raise ValueError("smoke PPO run evidence is invalid")
    timesteps = ppo_manifest.get("timesteps")
    rollout_size = ppo_schedule["completed_rollout_size"]
    if (
        type(timesteps) is not int
        or timesteps < rollout_size
        or timesteps % rollout_size
        or ppo_manifest.get("latest_checkpoint_step") != timesteps
    ):
        raise ValueError("smoke PPO did not complete a physical rollout")
    checkpoint = _safe_relative_file(
        ppo_run, ppo_manifest.get("latest_checkpoint"), "smoke PPO checkpoint"
    )
    initialization = _read_json(ppo_run / "initialization.json")
    maximum_error = initialization.get("maximum_absolute_logit_difference")
    if (
        initialization.get("kind") != "actor_only"
        or initialization.get("device") != "cpu"
        or maximum_error != 0
    ):
        raise ValueError("smoke actor transfer evidence is invalid")

    resolved_source = ControllerResolver(contract).resolve(f"run:{bc_root}")
    _validate_smoke_actor_source(
        source_contract=source_contract,
        target_contract=contract,
        resolved_contract=resolved_source.contract,
        initialization=initialization,
    )
    source_checkpoint = _safe_relative_file(
        bc_root,
        bc_manifest.get("latest_checkpoint"),
        "smoke BC source checkpoint",
    )
    source_fixtures = _safe_relative_file(
        bc_root, "actor-fixtures.npz", "smoke BC actor fixtures",
    )
    source_bc = _safe_relative_file(
        bc_root, "bc.json", "smoke BC metadata",
    )
    if (
        Path(str(initialization.get("source_run", ""))).resolve()
        != (_published_smoke_root(root) / "bc").resolve()
        or initialization.get("source_checkpoint")
        != source_checkpoint.relative_to(bc_root).as_posix()
        or initialization.get("source_checkpoint_sha256") != _sha256(source_checkpoint)
        or initialization.get("source_actor_fixtures_sha256") != _sha256(source_fixtures)
        or initialization.get("source_run_manifest_sha256")
        != _sha256(bc_root / "run.json")
        or initialization.get("source_bc_sha256") != _sha256(source_bc)
        or initialization.get("source_dataset_manifest_sha256") != dataset_digest
    ):
        raise ValueError("smoke actor source artifact provenance is invalid")

    resolved = ControllerResolver(contract).resolve(f"run:{ppo_run}")
    if (
        resolved.path is None
        or resolved.path.resolve() != checkpoint.resolve()
        or resolved.step != timesteps
        or resolved.algorithm != "maskable_ppo"
        or resolved.contract != contract
    ):
        raise ValueError("smoke PPO checkpoint reload is invalid")

    evaluation_root = root / "evaluation"
    evaluation = _read_json(evaluation_root / "evaluation.json")
    candidate = evaluation.get("candidate")
    opponent = evaluation.get("opponent")
    evaluation_schedule = schedule["evaluation"]
    if (
        evaluation.get("games") != evaluation_schedule["games"]
        or evaluation.get("seeds") != list(range(
            evaluation_schedule["seed_start"],
            evaluation_schedule["seed_start"] + evaluation_schedule["maps"],
        ))
        or evaluation.get("reciprocal") is not True
        or evaluation.get("schedule") != {
            "start_profile": evaluation_schedule["profile"],
            "reference_seat_policy": "candidate-seat",
        }
        or not isinstance(candidate, Mapping)
        or _smoke_recorded_file(root, candidate.get("path"), "evaluation checkpoint").resolve()
        != checkpoint.resolve()
        or candidate.get("source_run") != str(
            (_published_smoke_root(root) / "ppo" / ppo_schedule["run_name"]).resolve()
        )
        or candidate.get("step") != timesteps
        or candidate.get("algorithm") != "maskable_ppo"
        or not isinstance(opponent, Mapping)
        or opponent.get("name") != "random"
    ):
        raise ValueError("smoke reciprocal evaluation evidence is invalid")
    matches = evaluation.get("matches")
    _validate_smoke_evaluation_matches(
        matches,
        seed_start=evaluation_schedule["seed_start"],
        maps=evaluation_schedule["maps"],
    )
    assert isinstance(matches, list)
    declared: set[Path] = set()
    for match in matches:
        trace, replay = match.get("trace_path"), match.get("replay_path")
        if (trace is None) != (replay is None):
            raise ValueError("smoke evaluation trace/replay ownership is incomplete")
        if trace is not None:
            declared.add(_smoke_recorded_file(root, trace, "evaluation trace").resolve())
            declared.add(_smoke_recorded_file(root, replay, "evaluation replay").resolve())
    actual = {
        path.resolve()
        for pattern in ("traces/*.json", "replays/*.replay")
        for path in (evaluation_root / "evidence").glob(pattern)
    }
    if not declared or actual != declared:
        raise ValueError("smoke evaluation trace/replay artifacts are incomplete")

    checks = {
        "reciprocal_pairs": len(pairs),
        "games": audit["games"],
        "teacher_labels": audit["teacher_labels"],
        "masked_labels": audit["masked_labels"],
        "round_trip_mismatches": audit["round_trip_mismatches"],
        "replay_mismatches": audit["replay_mismatches"],
        "actor_fixture_max_error": maximum_error,
        "actor_fixture_device": initialization["device"],
        "ppo_timesteps": timesteps,
        "ppo_completed_rollouts": timesteps // rollout_size,
        "evaluation_games": evaluation["games"],
        "checkpoint_reloaded": True,
    }
    return checks, _smoke_artifact_hashes(root)


def _build_smoke_pipeline(
    root: Path,
    scenario: ResolvedScenario,
    definition_hashes: Mapping[str, str],
    repository_identity: Mapping[str, Any],
) -> None:
    from collect_annihilation_demonstrations import CollectionSpec, collect_partition
    from hexwars_gym import HexWarsEnv
    from ml_lab.contracts import RunConfig
    from ml_lab.evaluation import DuelClient
    from ml_lab.imitation import (
        BehavioralCloningConfig,
        load_imitation_dataset,
        train_behavioral_clone,
        training_rows_as_validation,
    )
    from ml_lab.training import run_training

    root = Path(root).resolve()
    published = _published_smoke_root(root)
    schedule = build_smoke_schedule()
    stage_scenario = _materialize_runtime_scenario(
        scenario, root, definition_hashes
    )
    duel_command = _server_command(stage_scenario)
    probe = DuelClient(duel_command, environment="tactical-v2")
    try:
        contract = probe.contract
    finally:
        probe.close()

    dataset_root = root / "dataset"
    scenario_hash = hashlib.sha256(
        scenario.canonical_json.encode("utf-8")
    ).hexdigest()
    collection = schedule["collection"]
    collect_partition(
        CollectionSpec(
            dataset=dataset_root,
            partition=collection["partition"],
            scenario_hash=scenario_hash,
            contract=contract,
            client_factory=lambda _worker: DuelClient(
                duel_command, environment="tactical-v2"
            ),
            workers=1,
            standard_pairs=collection["standard_pairs"],
            conversion_pairs=collection["conversion_pairs"],
        )
    )
    dataset = load_imitation_dataset(dataset_root, expected_contract=contract)

    bc_schedule = schedule["behavioral_cloning"]
    bc_config = BehavioralCloningConfig(
        model_seed=bc_schedule["model_seed"],
        batch_size=bc_schedule["batch_size"],
        learning_rate=3e-4,
        max_epochs=bc_schedule["max_epochs"],
        patience=bc_schedule["patience"],
    )
    bc_root = root / "bc"
    if not _reuse_completed_smoke_bc(
        bc_root,
        dataset_root / "manifest.json",
        scenario,
        definition_hashes,
        contract,
        bc_config,
    ):
        env = HexWarsEnv(
            duel_command[:-2],
            opponent="random",
            seat=0,
            base_seed=18_000_000,
            environment="tactical-v2",
            scenario_path=stage_scenario,
        )
        try:
            train_behavioral_clone(
                dataset=training_rows_as_validation(dataset),
                scenario=scenario,
                env=env,
                contract=contract,
                spaces_info=env.spaces_info,
                run_dir=bc_root,
                config=bc_config,
            )
        finally:
            env.close()

    ppo = schedule["ppo"]
    config = RunConfig(
        backend="sb3",
        algorithm="maskable_ppo",
        policy="HexCNN",
        run_name=ppo["run_name"],
        seed=ppo["model_seed"],
        episode_seed_base=ppo["episode_seed_base"],
        total_timesteps=ppo["total_timesteps"],
        checkpoint_interval=ppo["checkpoint_interval"],
        workers=ppo["workers"],
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "random"},
        trackers=[],
        resume_source=None,
        environment="tactical-v2",
        algorithm_options={
            "learning_rate": 3e-4,
            "n_epochs": 1,
            "target_kl": 0.02,
        },
        actor_init_source=str((root / "bc").resolve()),
    )
    ppo_run = _run_restart_safe_smoke_training(
        config,
        runs_root=root / "ppo",
        recovery_root=root.with_name(".smoke.recovery") / "ppo",
        trainer=run_training,
        allowed_actor_init_sources=(str((published / "bc").resolve()),),
        scenario=scenario,
        server_cmd=duel_command[:-2],
        console_output=False,
    )
    ppo_manifest = _read_json(ppo_run / "run.json")
    checkpoint = _safe_relative_file(
        ppo_run, ppo_manifest.get("latest_checkpoint"), "smoke PPO checkpoint"
    )

    evaluation = schedule["evaluation"]
    evaluation_root = root / "evaluation"
    evaluation_path = evaluation_root / "evaluation.json"
    result = _evaluate_standard_controllers(
        f"run:{ppo_run}",
        "random",
        games=evaluation["maps"],
        seed_start=evaluation["seed_start"],
        both_seats=evaluation["both_seats"],
        workers=1,
        server_cmd=duel_command,
        output_path=evaluation_path,
        environment="tactical-v2",
        evidence_dir=evaluation_root / "evidence",
        capture_trace=True,
        start_profile=evaluation["profile"],
    )
    _relativize_evaluation(result, root, evaluation_path)

    published_ppo = published / "ppo" / ppo["run_name"]
    ppo_manifest = _read_json(ppo_run / "run.json")
    recorded_config = dict(ppo_manifest["config"])
    recorded_config["actor_init_source"] = str((published / "bc").resolve())
    ppo_manifest["config"] = recorded_config
    _atomic_json(ppo_run / "run.json", ppo_manifest)
    initialization = _read_json(ppo_run / "initialization.json")
    initialization["source_run"] = str((published / "bc").resolve())
    _atomic_json(ppo_run / "initialization.json", initialization)
    evaluation_manifest = _read_json(evaluation_path)
    candidate = dict(evaluation_manifest["candidate"])
    candidate["path"] = str(
        (published_ppo / checkpoint.relative_to(ppo_run)).resolve()
    )
    candidate["source_run"] = str(published_ppo.resolve())
    evaluation_manifest["candidate"] = candidate
    _atomic_json(evaluation_path, evaluation_manifest)

    checks, artifacts = _collect_smoke_evidence(
        root,
        expected_repository_identity=repository_identity,
    )
    _atomic_json(
        root / "smoke.json",
        build_smoke_manifest(
            checks=checks,
            artifacts=artifacts,
            repository_identity=repository_identity,
        ),
    )


def _validate_real_smoke_stage(
    root: Path,
    repository_identity: Mapping[str, Any],
) -> Mapping[str, Any]:
    root = Path(root)
    checks = dict(_validate_smoke_stage(root, repository_identity))
    recomputed_checks, recomputed_artifacts = _collect_smoke_evidence(
        root,
        expected_repository_identity=repository_identity,
    )
    manifest = _read_json(root / "smoke.json")
    if (
        checks != recomputed_checks
        or manifest.get("artifacts") != recomputed_artifacts
    ):
        raise ValueError("smoke manifest differs from reopened physical evidence")
    return checks


def _smoke_command(
    *,
    smoke_root: Path = SMOKE_ROOT,
    pipeline: Callable[
        [Path, ResolvedScenario, Mapping[str, str], Mapping[str, Any]], None
    ]
    | None = None,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    _panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
    selected = pipeline
    if selected is None:
        selected = _build_smoke_pipeline
    identity_provider = repository_identity_provider or _smoke_repository_identity
    repository_identity = _validate_smoke_repository_identity(
        identity_provider(PROJECT_ROOT)
    )

    def build(staging: Path) -> None:
        selected(staging, scenario, hashes, repository_identity)

    def validate(root: Path) -> Mapping[str, Any]:
        summary = (
            _validate_smoke_stage(root, repository_identity)
            if pipeline is not None
            else _validate_real_smoke_stage(root, repository_identity)
        )
        current = _validate_smoke_repository_identity(
            identity_provider(PROJECT_ROOT)
        )
        if current != repository_identity:
            raise ValueError("smoke repository identity changed during execution")
        return summary

    return run_smoke_stage(
        smoke_root,
        hashes,
        stage_identity=repository_identity,
        build=build,
        validate=validate,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_annihilation_imitation_panel.py")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("collect")
    commands.add_parser("train-bc")
    commands.add_parser("evaluate-bc")
    commands.add_parser("train-ppo")
    commands.add_parser("smoke")
    commands.add_parser("evaluate-dev")
    commands.add_parser("select-budget")
    freeze = commands.add_parser("freeze-final")
    freeze.add_argument("--incumbent-panel", type=Path, required=True)
    commands.add_parser("evaluate-final")
    commands.add_parser("report")
    return parser


def _freeze_final_command(
    *,
    incumbent_panel: Path,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    _panel, _banks, _scenario, _hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )

    def final_identity_validator() -> None:
        current = _require_execution_identity(
            execution_identity_path=execution_identity_path,
            definition_hashes=current_definition_hashes(),
            repository=repository,
            repository_identity_provider=repository_identity_provider,
        )
        if current != identity:
            raise ValueError("execution identity changed during final freeze")

    return freeze_final(
        PANEL_ROOT,
        incumbent_panel=incumbent_panel,
        revision=identity["commit"],
        dirty=identity["dirty"],
        repository=repository,
        final_identity_validator=final_identity_validator,
    )


def _evaluate_final_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    _panel, _banks, _scenario, _hashes, _identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return evaluate_final(
        PANEL_ROOT,
        server_cmd=_server_command(SCENARIO_PATH)[:-2],
    )


def _report_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    _panel, _banks, _scenario, _hashes, _identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    result, _report = publish_final_report(PANEL_ROOT)
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        result: Mapping[str, Any] = _validate_command()
    elif args.command == "collect":
        result = _collect_command()
    elif args.command == "train-bc":
        result = _train_bc_command()
    elif args.command == "evaluate-bc":
        result = _evaluate_bc_command()
    elif args.command == "smoke":
        result = _smoke_command()
    elif args.command == "train-ppo":
        result = _train_ppo_command()
    elif args.command == "evaluate-dev":
        result = _evaluate_dev_command()
    elif args.command == "select-budget":
        result = _select_budget_command()
    elif args.command == "freeze-final":
        result = _freeze_final_command(incumbent_panel=args.incumbent_panel)
    elif args.command == "evaluate-final":
        result = _evaluate_final_command()
    elif args.command == "report":
        result = _report_command()
    else:
        raise AssertionError(f"unreachable command {args.command!r}")
    print(json.dumps(result, sort_keys=True))


@dataclass(frozen=True)
class TrainingRun:
    model_seed: int
    episode_seed_base: int
    condition: str
    scenario_sha256: str
    config: Any


def build_training_matrix(
    panel: Mapping[str, Any],
    *,
    clone_runs_root: Path = CLONE_RUNS_PATH,
    workers: int = 4,
    device: str = "auto",
) -> list[TrainingRun]:
    from ml_lab.contracts import RunConfig

    ppo = panel["ppo"]
    episode_bases = {int(seed): value for seed, value in ppo["episode_seed_bases"].items()}
    runs: list[TrainingRun] = []
    for model_seed in panel["model_seeds"]:
        for condition in ppo["conditions"]:
            initialized = condition == "bc_ppo"
            run_name = (
                f"bc-ppo-seed-{model_seed}"
                if initialized
                else f"scratch-ppo-seed-{model_seed}"
            )
            config = RunConfig(
                backend="sb3",
                algorithm="maskable_ppo",
                policy="HexCNN",
                run_name=run_name,
                seed=model_seed,
                episode_seed_base=episode_bases[model_seed],
                total_timesteps=51_200,
                checkpoint_interval=12_800,
                workers=workers,
                device=device,
                learner_seat="alternating",
                opponent={"kind": "scripted", "name": "random"},
                trackers=[{"kind": "local"}],
                resume_source=None,
                environment="tactical-v2",
                algorithm_options={
                    "learning_rate": 3e-4,
                    "n_epochs": 10,
                    "target_kl": 0.02,
                },
                actor_init_source=(
                    str((Path(clone_runs_root) / f"seed-{model_seed}").resolve())
                    if initialized
                    else None
                ),
            )
            runs.append(
                TrainingRun(
                    model_seed=model_seed,
                    episode_seed_base=episode_bases[model_seed],
                    condition=condition,
                    scenario_sha256=_sha256(SCENARIO_PATH),
                    config=config,
                )
            )
    return runs


@dataclass(frozen=True)
class CheckpointIdentity:
    actual_step: int
    path: Path


@dataclass(frozen=True)
class BudgetCheckpoint:
    nominal_step: int
    actual_step: int
    path: Path


@dataclass(frozen=True)
class BudgetSelection:
    nominal_step: int
    actual_steps: dict[int, int]


def validate_rollout_checkpoints(
    checkpoints: Sequence[CheckpointIdentity],
    *,
    rollout_size: int,
) -> None:
    if rollout_size <= 0:
        raise ValueError("rollout size must be positive")
    previous: int | None = None
    for checkpoint in checkpoints:
        step = checkpoint.actual_step
        if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
            raise ValueError("checkpoint actual step must be positive")
        if previous is not None:
            if step == previous:
                raise ValueError("checkpoint actual steps are duplicated")
            if step < previous:
                raise ValueError("checkpoint actual steps are decreasing")
        if step % rollout_size:
            raise ValueError("checkpoint actual step is not a rollout boundary")
        previous = step


def first_checkpoint_at_or_after(
    checkpoints: Sequence[CheckpointIdentity],
    nominal: int,
) -> CheckpointIdentity:
    eligible = sorted(
        (checkpoint for checkpoint in checkpoints if checkpoint.actual_step >= nominal),
        key=lambda checkpoint: checkpoint.actual_step,
    )
    if not eligible:
        raise RuntimeError(f"no completed rollout reaches {nominal}")
    return eligible[0]


def resolve_checkpoint_budgets(
    checkpoints: Sequence[CheckpointIdentity],
    *,
    nominal_steps: Sequence[int],
    rollout_size: int,
) -> list[BudgetCheckpoint]:
    validate_rollout_checkpoints(checkpoints, rollout_size=rollout_size)
    return [
        BudgetCheckpoint(
            nominal_step=nominal,
            actual_step=(selected := first_checkpoint_at_or_after(checkpoints, nominal)).actual_step,
            path=selected.path,
        )
        for nominal in nominal_steps
    ]


def _outcomes(
    row: Mapping[str, Any], field: str, *, allow_empty: bool = False
) -> list[str]:
    raw = row.get(field)
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise ValueError(f"selection row {field} outcomes are incomplete")
    if any(outcome not in {"win", "loss", "draw"} for outcome in raw):
        raise ValueError(f"selection row {field} outcome is invalid")
    return list(raw)


def select_global_budget(table: Sequence[Mapping[str, Any]]) -> BudgetSelection:
    rows = [dict(row) for row in table if row.get("condition") == "bc_ppo"]
    keys = [
        (row.get("model_seed"), row.get("nominal_step"))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("development candidate rows are duplicate")
    expected = {
        (seed, nominal)
        for seed in _MODEL_SEEDS
        for nominal in (12_800, 25_600, 51_200)
    }
    if set(keys) != expected:
        raise ValueError("development candidate rows are missing")
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        seed = row["model_seed"]
        actual = row.get("actual_step")
        if isinstance(actual, bool) or not isinstance(actual, int) or actual <= 0:
            raise ValueError("development actual checkpoint is invalid")
        _outcomes(row, "standard")
        _outcomes(row, "conversion", allow_empty=True)
        by_seed.setdefault(seed, []).append(row)
    for seed_rows in by_seed.values():
        ordered = sorted(seed_rows, key=lambda row: row["nominal_step"])
        actual = [row["actual_step"] for row in ordered]
        if any(following < previous for previous, following in zip(actual, actual[1:])):
            raise ValueError("development actual checkpoints are decreasing")
        if len(actual) != len(set(actual)):
            raise ValueError("development actual checkpoints are duplicate")

    scores: dict[int, tuple[float, float, float, float, int]] = {}
    for nominal in (12_800, 25_600, 51_200):
        candidates = [row for row in rows if row["nominal_step"] == nominal]
        standard_by_seed = {
            row["model_seed"]: _outcomes(row, "standard")
            for row in candidates
        }
        standard = [
            outcome for outcomes in standard_by_seed.values() for outcome in outcomes
        ]
        conversion = [
            outcome
            for row in candidates
            for outcome in _outcomes(row, "conversion", allow_empty=True)
        ]
        all_outcomes = standard + conversion
        scores[nominal] = (
            standard.count("win") / len(standard),
            min(
                outcomes.count("win") / len(outcomes)
                for outcomes in standard_by_seed.values()
            ),
            conversion.count("win") / len(conversion) if conversion else 0.0,
            -all_outcomes.count("draw") / len(all_outcomes),
            -nominal,
        )
    selected_nominal = max(scores, key=scores.__getitem__)
    return BudgetSelection(
        nominal_step=selected_nominal,
        actual_steps={
            row["model_seed"]: row["actual_step"]
            for row in rows
            if row["nominal_step"] == selected_nominal
        },
    )




@dataclass(frozen=True)
class DevelopmentCandidate:
    condition: str
    model_seed: int
    nominal_step: int
    actual_step: int
    controller: Any
    checkpoint_sha256: str | None = None
    checkpoint_path: str | None = None
    source_run: str | None = None
    algorithm: str | None = None


@dataclass(frozen=True)
class DevelopmentGame:
    condition: str
    model_seed: int
    nominal_step: int
    actual_step: int
    controller: str
    map_seed: int
    candidate_seat: int
    profile: str
    opponent: str


def build_development_schedule(
    candidates: Sequence[DevelopmentCandidate],
) -> list[DevelopmentGame]:
    keys = [
        (
            candidate.condition,
            candidate.model_seed,
            candidate.nominal_step,
            candidate.actual_step,
        )
        for candidate in candidates
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("development candidates are duplicate")
    return [
        DevelopmentGame(
            condition=candidate.condition,
            model_seed=candidate.model_seed,
            nominal_step=candidate.nominal_step,
            actual_step=candidate.actual_step,
            controller=candidate.controller,
            map_seed=map_seed,
            candidate_seat=seat,
            profile="standard-3v3",
            opponent="random",
        )
        for candidate in candidates
        for map_seed in range(16_000_000, 16_000_100)
        for seat in (0, 1)
    ]


def publish_selection(
    *,
    development_path: Path,
    output_path: Path,
    definition_hashes: Mapping[str, str],
) -> dict[str, Any]:
    development_path = Path(development_path)
    output_path = Path(output_path)
    hashes = {
        **dict(definition_hashes),
        "development_sha256": _sha256(development_path),
    }
    development = _read_json(development_path)
    candidates = development.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("development candidates are incomplete")
    selectable = [
        candidate
        for candidate in candidates
        if isinstance(candidate, Mapping) and candidate.get("condition") == "bc_ppo"
    ]
    if len(selectable) != 9:
        raise ValueError("development schedule is incomplete")
    for candidate in selectable:
        if (
            not isinstance(candidate.get("standard"), list)
            or len(candidate["standard"]) != 200
        ):
            raise ValueError("development schedule is incomplete")
    checkpoint_hashes: dict[str, str] = {}
    for candidate in selectable:
        key = (
            f"{candidate['condition']}/seed-{candidate['model_seed']}/"
            f"nominal-{candidate['nominal_step']}"
        )
        digest = candidate.get("checkpoint_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("development checkpoint hash is incomplete")
        checkpoint_path = candidate.get("checkpoint_path")
        if not isinstance(checkpoint_path, str) or not checkpoint_path:
            raise ValueError("development checkpoint identity is incomplete")
        physical_path = Path(checkpoint_path)
        if (
            not physical_path.is_absolute()
            or str(physical_path.resolve()) != checkpoint_path
            or not physical_path.is_file()
        ):
            raise ValueError("development checkpoint identity is not canonical")
        physical_digest = _sha256(physical_path)
        if physical_digest != digest:
            raise ValueError("development checkpoint digest does not match physical bytes")
        source_run = candidate.get("source_run")
        algorithm = candidate.get("algorithm")
        controller = candidate.get("controller")
        source_path = Path(source_run) if isinstance(source_run, str) else None
        if (
            source_path is None
            or not source_path.is_absolute()
            or str(source_path.resolve()) != source_run
            or physical_path.parent != source_path / "checkpoints"
            or algorithm != "maskable_ppo"
            or not isinstance(controller, Mapping)
            or controller.get("kind") != "snapshot"
            or controller.get("path") != checkpoint_path
            or controller.get("algorithm") != algorithm
            or controller.get("step") != candidate.get("actual_step")
        ):
            raise ValueError("development checkpoint controller identity is incomplete")
        checkpoint_hashes[key] = physical_digest
    hashes["candidate_checkpoints_sha256"] = checkpoint_hashes
    selected = select_global_budget(candidates)
    selection = {
        "nominal_step": selected.nominal_step,
        "actual_steps": {
            str(seed): step for seed, step in sorted(selected.actual_steps.items())
        },
    }
    payload = {
        "schema_version": 1,
        "state": "completed",
        "input_hashes": hashes,
        "selection": selection,
        "reused": False,
    }
    if output_path.exists():
        existing = _read_json(output_path)
        if existing != payload:
            raise ValueError("existing selection does not match current inputs")
        return {**existing, "reused": True}
    _atomic_json(output_path, payload)
    return payload


_PPO = {
    "conditions": ["bc_ppo", "scratch_ppo"],
    "episode_seed_bases": {"211": 13_000_000, "223": 14_000_000, "227": 15_000_000},
    "total_timesteps": 51_200,
    "checkpoint_interval": 12_800,
    "rollout_steps_per_worker": 512,
    "nominal_budgets": [12_800, 25_600, 51_200],
    "learner_seat": "alternating",
    "opponent": {"kind": "scripted", "name": "random"},
    "trackers": [{"kind": "local"}],
    "algorithm_options": {
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "target_kl": 0.02,
    },
}
_DEVELOPMENT = {
    "maps": 100,
    "seed_start": 16_000_000,
    "both_seats": True,
    "profile": "standard-3v3",
    "opponent": "random",
}
_DEVELOPMENT_CONVERSION = {
    "profiles": list(_CONVERSION_PROFILES),
    "maps_per_profile": _DEVELOPMENT["maps"],
    "both_seats": True,
    "seed_start": _DEVELOPMENT["seed_start"],
}


def evaluate_development_candidates(
    candidates: Sequence[DevelopmentCandidate],
    *,
    output_root: Path,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    server_cmd: Sequence[str],
    workers: int = 1,
    conversion_profiles: Sequence[str] = (),
) -> dict[str, Any]:
    conversion_profiles = tuple(conversion_profiles)
    if (
        len(set(conversion_profiles)) != len(conversion_profiles)
        or any(profile not in _CONVERSION_PROFILES for profile in conversion_profiles)
    ):
        raise ValueError("development conversion profiles are invalid")
    selected_evaluator = evaluator or _evaluate_standard_controllers
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    candidate_results: list[dict[str, Any]] = []
    for candidate in candidates:
        matches: list[dict[str, Any]] = []
        for map_seed in range(_DEVELOPMENT["seed_start"], _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"]):
            map_root = (
                root
                / candidate.condition
                / f"seed-{candidate.model_seed}"
                / f"nominal-{candidate.nominal_step}"
                / f"map-{map_seed}"
            )
            output_path = map_root / "evaluation.json"
            selected_evaluator(
                candidate.controller,
                _DEVELOPMENT["opponent"],
                games=1,
                seed_start=map_seed,
                both_seats=True,
                workers=workers,
                server_cmd=list(server_cmd),
                output_path=output_path,
                environment="tactical-v2",
                evidence_dir=map_root / "evidence",
                capture_trace=True,
                start_profile=_DEVELOPMENT["profile"],
            )
            _normalized, physical_matches = _validated_development_map(root, candidate, map_seed, output_path)
            matches.extend(physical_matches)
        if [(row["map_seed"], row["candidate_seat"]) for row in matches] != [
            (map_seed, seat)
            for map_seed in range(_DEVELOPMENT["seed_start"], _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"])
            for seat in (0, 1)
        ]:
            raise ValueError("development candidate schedule is incomplete")
        conversion_matches: list[dict[str, Any]] = []
        if candidate.condition == "bc_ppo":
            for profile in conversion_profiles:
                for map_seed in range(
                    _DEVELOPMENT["seed_start"],
                    _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
                ):
                    map_root = (
                        root
                        / candidate.condition
                        / f"seed-{candidate.model_seed}"
                        / f"nominal-{candidate.nominal_step}"
                        / "conversion"
                        / profile
                        / f"map-{map_seed}"
                    )
                    output_path = map_root / "evaluation.json"
                    selected_evaluator(
                        candidate.controller,
                        _DEVELOPMENT["opponent"],
                        games=1,
                        seed_start=map_seed,
                        both_seats=True,
                        workers=workers,
                        server_cmd=list(server_cmd),
                        output_path=output_path,
                        environment="tactical-v2",
                        evidence_dir=map_root / "evidence",
                        capture_trace=True,
                        start_profile=profile,
                    )
                    _normalized, physical_matches = _validated_development_map(
                        root,
                        candidate,
                        map_seed,
                        output_path,
                        expected_profile=profile,
                    )
                    conversion_matches.extend(physical_matches)
        expected_conversion_schedule = [
            (profile, map_seed, seat)
            for profile in conversion_profiles
            for map_seed in range(
                _DEVELOPMENT["seed_start"],
                _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
            )
            for seat in (0, 1)
        ]
        if candidate.condition == "bc_ppo" and [
            (row["profile"], row["map_seed"], row["candidate_seat"])
            for row in conversion_matches
        ] != expected_conversion_schedule:
            raise ValueError("development conversion schedule is incomplete")
        candidate_results.append(
            {
                "condition": candidate.condition,
                "model_seed": candidate.model_seed,
                "nominal_step": candidate.nominal_step,
                "actual_step": candidate.actual_step,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "checkpoint_path": candidate.checkpoint_path or matches[0]["controller"].get("path"),
                "source_run": candidate.source_run or matches[0]["controller"].get("source_run"),
                "algorithm": candidate.algorithm or matches[0]["controller"].get("algorithm"),
                "controller": matches[0]["controller"],
                "standard": [row["outcome"] for row in matches],
                "conversion": [row["outcome"] for row in conversion_matches],
                "conversion_matches": conversion_matches,
                "conversion_schedule": (
                    {
                        "profiles": list(conversion_profiles),
                        "maps_per_profile": _DEVELOPMENT["maps"],
                        "both_seats": True,
                        "seed_start": _DEVELOPMENT["seed_start"],
                    }
                    if candidate.condition == "bc_ppo" and conversion_profiles
                    else None
                ),
                "matches": matches,
            }
        )
    result = {
        "schema_version": 1,
        "state": "completed",
        "schedule": dict(_DEVELOPMENT),
        "candidates": candidate_results,
    }
    _atomic_json(root / "development.json", result)
    return result


_ACTOR_INITIALIZER_MODULES = [
    "features_extractor",
    "mlp_extractor.policy_net",
    "action_net",
]


def _validate_actor_initialization(run: TrainingRun, provenance: Mapping[str, Any]) -> None:
    source = Path(run.config.actor_init_source).resolve()
    source_manifest = _read_json(source / "run.json")
    source_checkpoint_relative = source_manifest.get("latest_checkpoint")
    source_checkpoint = _safe_relative_file(
        source, source_checkpoint_relative, "actor initializer checkpoint"
    )
    source_panel = _read_json(source / "panel-provenance.json")
    dataset_path_raw = source_panel.get("dataset_manifest")
    if not isinstance(dataset_path_raw, str):
        raise ValueError("PPO initialization provenance dataset identity is invalid")
    dataset_path = Path(dataset_path_raw)
    dataset_digest = _sha256(dataset_path)
    if dataset_digest != source_manifest.get("dataset_manifest_sha256"):
        raise ValueError("PPO initialization provenance dataset identity is invalid")
    contract = source_manifest.get("contract")
    if not isinstance(contract, Mapping):
        raise ValueError("PPO initialization provenance contract is invalid")
    expected = {
        "schema_version": 1,
        "kind": "actor_only",
        "source_run": str(source),
        "source_checkpoint": source_checkpoint_relative,
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_actor_fixtures_sha256": _sha256(source / "actor-fixtures.npz"),
        "source_run_manifest_sha256": _sha256(source / "run.json"),
        "source_bc_sha256": _sha256(source / "bc.json"),
        "source_dataset_manifest_sha256": dataset_digest,
        "source_contract_hash": contract.get("contract_hash"),
        "source_encoding_hash": contract.get("encoding_hash"),
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ValueError("PPO initialization provenance does not match physical clone inputs")
    if (
        provenance.get("actor_modules") != _ACTOR_INITIALIZER_MODULES
        or not isinstance(provenance.get("device"), str)
        or not provenance["device"]
    ):
        raise ValueError("PPO initialization provenance metadata is incomplete")
    for field in (
        "comparison_rtol",
        "comparison_atol",
        "maximum_absolute_logit_difference",
    ):
        value = provenance.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError("PPO initialization provenance comparison is invalid")


def _ppo_budget_map(
    root: Path,
    matrix: Sequence[TrainingRun],
    scenario: ResolvedScenario,
) -> dict[str, list[BudgetCheckpoint]]:
    root = Path(root)
    expected_names = {run.config.run_name for run in matrix}
    actual_names = {
        path.name for path in root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    } if root.is_dir() else set()
    if actual_names != expected_names:
        raise ValueError("PPO stage run destinations are incomplete or unexpected")
    budgets: dict[str, list[BudgetCheckpoint]] = {}
    for run in matrix:
        run_dir = root / run.config.run_name
        manifest = _read_json(run_dir / "run.json")
        if (
            manifest.get("state") != "completed"
            or manifest.get("config") != run.config.to_dict()
        ):
            raise ValueError(f"PPO run {run.config.run_name} is incomplete or changed")
        frozen = resolve_scenario(
            environment="tactical-v2",
            scenario_file=run_dir / "scenario.json",
            template_id=None,
            enforce_round_cap_minimum=True,
        )
        if frozen.canonical_json != scenario.canonical_json:
            raise ValueError(f"PPO run {run.config.run_name} scenario changed")
        initialization = run_dir / "initialization.json"
        if (run.condition == "bc_ppo") != initialization.is_file():
            raise ValueError(f"PPO run {run.config.run_name} initialization provenance is invalid")
        if run.condition == "bc_ppo":
            provenance = _read_json(initialization)
            _validate_actor_initialization(run, provenance)
        checkpoints: list[CheckpointIdentity] = []
        for path in sorted((run_dir / "checkpoints").glob("step_*.zip")):
            try:
                step = int(path.stem.removeprefix("step_"))
            except ValueError as exc:
                raise ValueError(f"PPO run {run.config.run_name} checkpoint identity is invalid") from exc
            checkpoints.append(CheckpointIdentity(step, path))
        budgets[run.config.run_name] = resolve_checkpoint_budgets(
            checkpoints,
            nominal_steps=_PPO["nominal_budgets"],
            rollout_size=_PPO["rollout_steps_per_worker"] * run.config.workers,
        )
    return budgets


def _validate_ppo_stage(
    root: Path,
    hashes: Mapping[str, str],
    matrix: Sequence[TrainingRun],
    scenario: ResolvedScenario,
) -> Mapping[str, Any]:
    budget_map = _ppo_budget_map(root, matrix, scenario)
    if any(
        run.scenario_sha256 != hashes.get("scenario_sha256")
        for run in matrix
    ):
        raise ValueError("PPO matrix scenario hash does not match definitions")
    return {
        "run_count": len(budget_map),
        "conditions": list(_PPO["conditions"]),
        "model_seeds": list(_MODEL_SEEDS),
        "nominal_budgets": list(_PPO["nominal_budgets"]),
    }


def train_ppo_runs(
    runs: Sequence[TrainingRun],
    *,
    runs_root: Path,
    scenario: ResolvedScenario,
    server_cmd: list[str],
    trainer: Callable[..., Path] | None = None,
) -> list[Path]:
    from ml_lab.training import run_training

    selected_trainer = trainer or run_training
    root = Path(runs_root)
    root.mkdir(parents=True, exist_ok=True)
    pending_root = root / ".pending"
    pending_root.mkdir(exist_ok=True)

    def reusable(path: Path, run: TrainingRun) -> bool:
        manifest_path = path / "run.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = _read_json(manifest_path)
        except (OSError, json.JSONDecodeError, ValueError):
            return False
        return (
            manifest.get("state") == "completed"
            and manifest.get("config") == run.config.to_dict()
        )

    outputs: list[Path] = []
    for run in runs:
        expected = root / run.config.run_name
        pending = pending_root / run.config.run_name
        if reusable(expected, run):
            outputs.append(expected)
            continue
        if expected.exists():
            shutil.rmtree(expected)
        if pending.exists():
            if reusable(pending, run):
                os.replace(pending, expected)
                outputs.append(expected)
                continue
            shutil.rmtree(pending)
        output = Path(
            selected_trainer(
                run.config,
                runs_root=pending_root,
                scenario=scenario,
                server_cmd=list(server_cmd),
            )
        )
        if output != pending:
            raise ValueError(
                f"PPO run {run.config.run_name} trained outside its pending destination"
            )
        if not reusable(pending, run):
            raise ValueError(f"PPO run {run.config.run_name} did not complete in pending")
        os.replace(pending, expected)
        outputs.append(expected)
    if len({path.resolve() for path in outputs}) != len(outputs):
        raise ValueError("PPO conditions share a run destination")
    return outputs


def _train_ppo_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    panel, _banks, scenario, hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    _validate_clone_stage(CLONE_RUNS_PATH, hashes)
    _validate_gate_stage(CLONE_EVALUATION_PATH, hashes)
    matrix = build_training_matrix(panel)

    def build(staging: Path) -> None:
        train_ppo_runs(
            matrix,
            runs_root=staging,
            scenario=scenario,
            server_cmd=_server_command(SCENARIO_PATH)[:-2],
        )

    validate = _full_stage_validator(
        lambda root: _validate_ppo_stage(root, hashes, matrix, scenario),
        expected_identity=identity,
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return run_atomic_stage(
        PPO_RUNS_PATH,
        hashes,
        stage_identity=identity,
        build=build,
        validate=validate,
    )


def build_panel_development_candidates(
    panel: Mapping[str, Any],
    *,
    clone_runs_root: Path,
    ppo_runs_root: Path,
    scenario: ResolvedScenario,
) -> list[DevelopmentCandidate]:
    matrix = build_training_matrix(panel, clone_runs_root=clone_runs_root)
    budget_map = _ppo_budget_map(ppo_runs_root, matrix, scenario)
    candidates: list[DevelopmentCandidate] = []
    for seed in _MODEL_SEEDS:
        run = Path(clone_runs_root) / f"seed-{seed}"
        manifest = _read_json(run / "run.json")
        checkpoint = _safe_relative_file(
            run, manifest.get("latest_checkpoint"), f"clone seed {seed} checkpoint"
        )
        candidates.append(
            DevelopmentCandidate(
                "pure_bc", seed, 0, 0, f"run:{run}",
                checkpoint_sha256=_sha256(checkpoint),
                checkpoint_path=str(checkpoint.resolve()),
                source_run=str(run.resolve()),
            )
        )
    for run in matrix:
        run_dir = Path(ppo_runs_root) / run.config.run_name
        for budget in budget_map[run.config.run_name]:
            controller = json.dumps(
                {
                    "kind": "snapshot",
                    "path": str(budget.path.resolve()),
                    "source_run": str(run_dir.resolve()),
                    "algorithm": "maskable_ppo",
                    "step": budget.actual_step,
                },
                sort_keys=True,
            )
            candidates.append(
                DevelopmentCandidate(
                    run.condition,
                    run.model_seed,
                    budget.nominal_step,
                    budget.actual_step,
                    controller,
                    checkpoint_sha256=_sha256(budget.path),
                    checkpoint_path=str(budget.path.resolve()),
                    source_run=str(run_dir.resolve()),
                    algorithm="maskable_ppo",
                )
            )
    if len(candidates) != 21:
        raise ValueError("development candidate matrix is incomplete")
    return candidates


def _validate_development_candidate_evidence(
    root: Path,
    candidate: Mapping[str, Any],
) -> list[dict[str, Any]]:
    matches = candidate.get("matches")
    if (
        not isinstance(matches, list)
        or len(matches) != 200
        or candidate.get("standard") != [match.get("outcome") for match in matches]
        or not isinstance(candidate.get("checkpoint_sha256"), str)
    ):
        raise ValueError("development candidate schedule is incomplete")
    expected_schedule = [
        (seed, seat)
        for seed in range(
            _DEVELOPMENT["seed_start"],
            _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
        )
        for seat in (0, 1)
    ]
    if [
        (match.get("map_seed"), match.get("candidate_seat"))
        for match in matches
        if isinstance(match, Mapping)
    ] != expected_schedule:
        raise ValueError("development candidate seed/seat schedule is incomplete")

    checkpoint_path_raw = candidate.get("checkpoint_path")
    checkpoint_digest = candidate.get("checkpoint_sha256")
    if (
        not isinstance(checkpoint_path_raw, str)
        or not isinstance(checkpoint_digest, str)
    ):
        raise ValueError("development checkpoint identity is incomplete")
    checkpoint_path = Path(checkpoint_path_raw)
    if (
        not checkpoint_path.is_absolute()
        or str(checkpoint_path.resolve()) != checkpoint_path_raw
        or not checkpoint_path.is_file()
        or _sha256(checkpoint_path) != checkpoint_digest
    ):
        raise ValueError("development checkpoint digest does not match physical bytes")
    identity = DevelopmentCandidate(
        str(candidate.get("condition")),
        int(candidate.get("model_seed")),
        int(candidate.get("nominal_step")),
        int(candidate.get("actual_step")),
        candidate.get("controller"),
        checkpoint_sha256=checkpoint_digest,
        checkpoint_path=checkpoint_path_raw,
        source_run=candidate.get("source_run"),
        algorithm=candidate.get("algorithm"),
    )
    physical_matches: list[dict[str, Any]] = []
    for map_seed in range(
        _DEVELOPMENT["seed_start"],
        _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
    ):
        output_path = (
            Path(root)
            / identity.condition
            / f"seed-{identity.model_seed}"
            / f"nominal-{identity.nominal_step}"
            / f"map-{map_seed}"
            / "evaluation.json"
        )
        _normalized, map_matches = _validated_development_map(
            Path(root), identity, map_seed, output_path
        )
        physical_matches.extend(map_matches)
    if (
        matches != physical_matches
        or candidate.get("standard")
        != [match["outcome"] for match in physical_matches]
    ):
        raise ValueError("development aggregate does not match physical evaluations")

    conversion_schedule = candidate.get("conversion_schedule")
    if conversion_schedule is not None:
        if conversion_schedule != _DEVELOPMENT_CONVERSION:
            raise ValueError("development conversion schedule is incomplete")
        conversion_matches = candidate.get("conversion_matches")
        expected_conversion_schedule = [
            (profile, seed, seat)
            for profile in _CONVERSION_PROFILES
            for seed in range(
                _DEVELOPMENT["seed_start"],
                _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
            )
            for seat in (0, 1)
        ]
        if (
            not isinstance(conversion_matches, list)
            or [
                (match.get("profile"), match.get("map_seed"), match.get("candidate_seat"))
                for match in conversion_matches
                if isinstance(match, Mapping)
            ]
            != expected_conversion_schedule
            or candidate.get("conversion")
            != [match.get("outcome") for match in conversion_matches]
        ):
            raise ValueError("development conversion schedule is incomplete")

        physical_conversion_matches: list[dict[str, Any]] = []
        for profile in _CONVERSION_PROFILES:
            for map_seed in range(
                _DEVELOPMENT["seed_start"],
                _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
            ):
                output_path = (
                    Path(root)
                    / identity.condition
                    / f"seed-{identity.model_seed}"
                    / f"nominal-{identity.nominal_step}"
                    / "conversion"
                    / profile
                    / f"map-{map_seed}"
                    / "evaluation.json"
                )
                _normalized, map_matches = _validated_development_map(
                    Path(root),
                    identity,
                    map_seed,
                    output_path,
                    expected_profile=profile,
                )
                physical_conversion_matches.extend(map_matches)
        if conversion_matches != physical_conversion_matches:
            raise ValueError(
                "development conversion aggregate does not match physical evaluations"
            )
    return physical_matches


def _validate_development_stage(
    root: Path,
    hashes: Mapping[str, str],
    scenario: ResolvedScenario,
) -> Mapping[str, Any]:
    root = Path(root)
    _validate_runtime_scenario(root, scenario, hashes)
    development = _read_json(root / "development.json")
    if development.get("definition_hashes") != dict(hashes):
        raise ValueError("development definition hashes do not match")
    candidates = development.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 21:
        raise ValueError("development candidates are incomplete")
    expected_keys = {
        ("pure_bc", seed, 0) for seed in _MODEL_SEEDS
    } | {
        (condition, seed, nominal)
        for condition in _PPO["conditions"]
        for seed in _MODEL_SEEDS
        for nominal in _PPO["nominal_budgets"]
    }
    keys = {
        (candidate.get("condition"), candidate.get("model_seed"), candidate.get("nominal_step"))
        for candidate in candidates
        if isinstance(candidate, Mapping)
    }
    if keys != expected_keys:
        raise ValueError("development candidate identities are incomplete")
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("development candidate is malformed")
        if candidate.get("condition") == "bc_ppo":
            if candidate.get("conversion_schedule") != _DEVELOPMENT_CONVERSION:
                raise ValueError("initialized-PPO conversion evidence is incomplete")
        elif (
            candidate.get("conversion_schedule") is not None
            or candidate.get("conversion") != []
            or candidate.get("conversion_matches") != []
        ):
            raise ValueError("unexpected development conversion evidence")
        _validate_development_candidate_evidence(root, candidate)

    return {
        "candidate_count": 21,
        "standard_games": 4_200,
        "conversion_games": (
            len(_MODEL_SEEDS)
            * len(_PPO["nominal_budgets"])
            * len(_CONVERSION_PROFILES)
            * _DEVELOPMENT["maps"]
            * 2
        ),
    }


def _evaluate_dev_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    panel, _banks, scenario, hashes, identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    _validate_clone_stage(CLONE_RUNS_PATH, hashes)
    _validate_gate_stage(CLONE_EVALUATION_PATH, hashes)
    matrix = build_training_matrix(panel)
    _validate_ppo_stage(PPO_RUNS_PATH, hashes, matrix, scenario)
    candidates = build_panel_development_candidates(
        panel,
        clone_runs_root=CLONE_RUNS_PATH,
        ppo_runs_root=PPO_RUNS_PATH,
        scenario=scenario,
    )

    def build(staging: Path) -> None:
        runtime_scenario = _materialize_runtime_scenario(scenario, staging, hashes)
        result = evaluate_development_candidates(
            candidates,
            output_root=staging,
            server_cmd=_server_command(runtime_scenario),
            conversion_profiles=_CONVERSION_PROFILES,
        )
        for row, candidate in zip(result["candidates"], candidates, strict=True):
            row["checkpoint_sha256"] = candidate.checkpoint_sha256
        result["definition_hashes"] = dict(hashes)
        _atomic_json(staging / "development.json", result)

    validate = _full_stage_validator(
        lambda root: _validate_development_stage(root, hashes, scenario),
        expected_identity=identity,
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    return run_atomic_stage(
        DEVELOPMENT_PATH,
        hashes,
        stage_identity=identity,
        build=build,
        validate=validate,
    )


def _select_budget_command(
    *,
    execution_identity_path: Path = EXECUTION_IDENTITY_PATH,
    repository: Path = PROJECT_ROOT,
    repository_identity_provider: Callable[
        [Path], Mapping[str, Any]
    ] | None = None,
) -> Mapping[str, Any]:
    panel, _banks, scenario, hashes, _identity = _full_execution_context(
        execution_identity_path=execution_identity_path,
        repository=repository,
        repository_identity_provider=repository_identity_provider,
    )
    del panel
    _validate_development_stage(DEVELOPMENT_PATH, hashes, scenario)
    return publish_selection(
        development_path=DEVELOPMENT_PATH / "development.json",
        output_path=SELECTION_PATH,
        definition_hashes=hashes,
    )


def _validated_development_map(
    root: Path,
    candidate: DevelopmentCandidate,
    map_seed: int,
    output_path: Path,
    *,
    expected_profile: str = _DEVELOPMENT["profile"],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        physical = _read_json(output_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("physical development evaluation is missing or malformed") from exc
    controller = physical.get("candidate")
    opponent = physical.get("opponent")
    schedule = physical.get("schedule")
    raw_matches = physical.get("matches")
    if (
        physical.get("schema_version") != 1
        or not isinstance(physical.get("generated_at"), str)
        or not isinstance(controller, Mapping)
        or not isinstance(opponent, Mapping)
        or opponent.get("kind") != "scripted"
        or opponent.get("name") != "random"
        or physical.get("seed_start") != map_seed
        or physical.get("seeds") != [map_seed]
        or physical.get("reciprocal") is not True
        or physical.get("games") != 2
        or schedule
        != {
            "start_profile": expected_profile,
            "reference_seat_policy": "candidate-seat",
        }
        or not isinstance(raw_matches, list)
        or len(raw_matches) != 2
    ):
        raise ValueError("physical development evaluation identity is invalid")

    expected_kind = "run" if candidate.condition == "pure_bc" else "snapshot"
    if controller.get("kind") != expected_kind:
        raise ValueError("physical development evaluation controller kind is invalid")
    checkpoint_path = (
        Path(candidate.checkpoint_path).resolve()
        if candidate.checkpoint_path is not None
        else None
    )
    source_run = (
        Path(candidate.source_run).resolve()
        if candidate.source_run is not None
        else None
    )
    if checkpoint_path is not None and (
        not isinstance(controller.get("path"), str)
        or Path(controller["path"]).resolve() != checkpoint_path
    ):
        raise ValueError("physical development evaluation checkpoint is invalid")
    if source_run is not None:
        if checkpoint_path is None or checkpoint_path.parent != source_run / "checkpoints":
            raise ValueError("physical development evaluation source run is invalid")
        recorded_source = controller.get("source_run")
        if recorded_source is not None and (
            not isinstance(recorded_source, str)
            or Path(recorded_source).resolve() != source_run
        ):
            raise ValueError("physical development evaluation source run is invalid")
        if expected_kind == "snapshot":
            raw_spec = candidate.controller
            if isinstance(raw_spec, str):
                try:
                    raw_spec = json.loads(raw_spec)
                except json.JSONDecodeError:
                    raw_spec = None
            if isinstance(raw_spec, Mapping) and "source_run" in raw_spec and (
                not isinstance(raw_spec.get("source_run"), str)
                or Path(raw_spec["source_run"]).resolve() != source_run
            ):
                raise ValueError("physical development evaluation source run is invalid")
    if (
        controller.get("step") != candidate.actual_step
        or (
            candidate.algorithm is not None
            and controller.get("algorithm") != candidate.algorithm
        )
    ):
        raise ValueError("physical development evaluation algorithm or step is invalid")

    normalized = _relativize_evaluation(physical, root, output_path)
    matches: list[dict[str, Any]] = []
    totals = {"wins": 0, "losses": 0, "draws": 0}
    seat_results = {
        "candidate_as_p0": {"wins": 0, "losses": 0, "draws": 0},
        "candidate_as_p1": {"wins": 0, "losses": 0, "draws": 0},
    }
    for seat, raw in enumerate(normalized["matches"]):
        if (
            not isinstance(raw, Mapping)
            or raw.get("seed") != map_seed
            or raw.get("candidate_seat") != seat
            or raw.get("outcome") not in {"win", "loss", "draw"}
            or not isinstance(raw.get("trace_path"), str)
            or not isinstance(raw.get("replay_path"), str)
        ):
            raise ValueError("physical development evaluation match is invalid")
        outcome = raw["outcome"]
        counter = f"{outcome}s" if outcome != "loss" else "losses"
        totals[counter] += 1
        seat_key = "candidate_as_p0" if seat == 0 else "candidate_as_p1"
        seat_results[seat_key][counter] += 1
        row = dict(raw)
        row["map_seed"] = row.pop("seed")
        row.update(
            {
                "condition": candidate.condition,
                "model_seed": candidate.model_seed,
                "nominal_step": candidate.nominal_step,
                "actual_step": candidate.actual_step,
                "checkpoint_sha256": candidate.checkpoint_sha256,
                "controller": dict(controller),
                "opponent": dict(opponent),
                "profile": expected_profile,
            }
        )
        matches.append(row)

    rates = {
        "win": totals["wins"] / 2,
        "loss": totals["losses"] / 2,
        "draw": totals["draws"] / 2,
    }
    confidence = physical.get("confidence_intervals")
    evidence = physical.get("evidence")
    if (
        any(physical.get(key) != value for key, value in totals.items())
        or physical.get("rates") != rates
        or physical.get("seat_results") != seat_results
        or not isinstance(confidence, Mapping)
        or set(confidence) != {"win", "loss", "draw"}
        or not isinstance(evidence, Mapping)
        or evidence.get("draw_traces") != totals["draws"]
        or evidence.get("control_traces") != 2 - totals["draws"]
        or not isinstance(evidence.get("draw_categories"), Mapping)
    ):
        raise ValueError("physical development evaluation aggregate is invalid")
    return normalized, matches


@dataclass(frozen=True)
class GateResult:
    passed: bool
    per_seed_wins: dict[int, int]
    pooled_wins: int
    games_per_seed: int
    per_seed_minimum: int = 325
    pooled_minimum: int = 1050


_FINAL = {
    "seed_start": 17_000_000,
    "maps": 250,
    "games_per_model": 500,
    "profile": "standard-3v3",
    "opponent": "random",
}


def _tree_sha256(root: Path) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"{root}: hash source is empty")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _repository_identity(repository: Path) -> tuple[str, bool]:
    repository = Path(repository)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip()
    dirty = bool(subprocess.run(
        ["git", "status", "--porcelain"], cwd=repository, check=True,
        capture_output=True, text=True,
    ).stdout.strip())
    return revision, dirty


def _selected_incumbents(incumbent_panel: Path) -> list[dict[str, Any]]:
    incumbent_panel = Path(incumbent_panel)
    metadata = _read_json(incumbent_panel / "aggregate.json")
    models = metadata.get("models")
    raw = models.get("profiled_standard") if isinstance(models, Mapping) else None
    if not isinstance(raw, Mapping) or set(raw) != {"101", "113", "127"}:
        raise ValueError("incumbent panel must resolve exactly three profiled-standard runs")

    runs_root = incumbent_panel.parent.parent / "runs"
    selected = []
    for pairing_seed, incumbent_seed in zip(_MODEL_SEEDS, (101, 113, 127), strict=True):
        item = raw[str(incumbent_seed)]
        training = item.get("training") if isinstance(item, Mapping) else None
        if not isinstance(training, Mapping):
            raise ValueError("incumbent comparator metadata is malformed")
        run_name = training.get("run")
        checkpoint_name = training.get("checkpoint")
        if not isinstance(run_name, str) or not isinstance(checkpoint_name, str):
            raise ValueError("incumbent comparator identity is incomplete")
        run_path = (runs_root / run_name).resolve()
        checkpoint = (run_path / checkpoint_name).resolve()
        if (
            not run_path.is_dir()
            or not checkpoint.is_file()
            or checkpoint.parent != run_path / "checkpoints"
            or training.get("checkpoint_sha256") != _sha256(checkpoint)
        ):
            raise ValueError("incumbent comparator physical identity changed")

        manifest = _read_json(run_path / "run.json")
        config = manifest.get("config")
        contract = manifest.get("contract")
        scenario_record = manifest.get("scenario")
        scenario_path = run_path / str(
            scenario_record.get("path", "") if isinstance(scenario_record, Mapping) else ""
        )
        scenario = _read_json(scenario_path)
        tactical = scenario.get("tactical_v2")
        distribution = tactical.get("start_distribution") if isinstance(tactical, Mapping) else None
        profile_weights = {
            row.get("profile_id"): row.get("basis_points")
            for row in distribution
            if isinstance(row, Mapping)
        } if isinstance(distribution, list) else {}
        algorithm = config.get("algorithm") if isinstance(config, Mapping) else None
        step = manifest.get("latest_checkpoint_step")
        contract_hash = contract.get("contract_hash") if isinstance(contract, Mapping) else None
        if (
            manifest.get("state") != "completed"
            or not isinstance(config, Mapping)
            or config.get("seed") != incumbent_seed
            or config.get("environment") != "tactical-v2"
            or algorithm != "maskable_ppo"
            or manifest.get("latest_checkpoint") != checkpoint_name
            or step != 51_200
            or not isinstance(contract_hash, str)
            or len(contract_hash) != 64
            or scenario.get("environment") != "tactical-v2"
            or profile_weights.get("standard-3v3") != 10_000
            or sum(value for value in profile_weights.values() if isinstance(value, int)) != 10_000
        ):
            raise ValueError("incumbent comparator is not a profiled-standard completed run")
        selected.append({
            "model_seed": pairing_seed,
            "pairing_seed": pairing_seed,
            "incumbent_seed": incumbent_seed,
            "run_path": str(run_path),
            "run_sha256": _tree_sha256(run_path),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "algorithm": algorithm,
            "step": step,
            "contract_hash": contract_hash,
        })
    return selected


def freeze_final(
    panel_dir: Path,
    *,
    incumbent_panel: Path | None = None,
    dataset_dir: Path | None = None,
    revision: str | None = None,
    dirty: bool | None = None,
    repository: Path = PROJECT_ROOT,
    final_identity_validator: Callable[[], None] | None = None,
) -> dict[str, Any]:
    panel_dir = Path(panel_dir)
    verify_repository = (
        final_identity_validator is not None
        or revision is None
        or dirty is None
    )
    seal_path = panel_dir / "final-seal.json"
    if seal_path.exists():
        raise RuntimeError("final bank is already assigned")
    if dirty is not None and dirty is not False:
        raise ValueError("final seal requires a clean execution identity")
    selection_path = panel_dir / "selection.json"
    if not selection_path.is_file():
        raise RuntimeError("global checkpoint selection is not frozen")
    selection = _read_json(selection_path)
    choice = selection.get("selection")
    if selection.get("state") != "completed" or not isinstance(choice, Mapping):
        raise RuntimeError("global checkpoint selection is not frozen")
    nominal = choice.get("nominal_step")
    actual_steps = choice.get("actual_steps")
    if (
        nominal not in _PPO["nominal_budgets"]
        or not isinstance(actual_steps, Mapping)
        or set(actual_steps) != {str(seed) for seed in _MODEL_SEEDS}
    ):
        raise RuntimeError("global checkpoint selection is not frozen")

    seed_path = panel_dir / "seed-banks.json"
    banks_document = _read_json(seed_path)
    banks = banks_document.get("banks", banks_document)
    final_bank = banks.get("final") if isinstance(banks, Mapping) else None
    if (
        not isinstance(final_bank, Mapping)
        or final_bank.get("start") != _FINAL["seed_start"]
        or final_bank.get("stop") != _FINAL["seed_start"] + _FINAL["maps"] - 1
        or final_bank.get("assigned") is not False
    ):
        raise RuntimeError("final bank is already assigned or does not match the locked bank")

    scenario_path = panel_dir / "scenario.json"
    if not scenario_path.is_file():
        scenario_path = SCENARIO_PATH
    definitions = {
        "panel.json": _sha256(panel_dir / "panel.json"),
        "scenario.json": _sha256(scenario_path),
        "seed-banks.json": _sha256(seed_path),
    }
    dataset_root = Path(dataset_dir) if dataset_dir is not None else DATASET_PATH
    if not (dataset_root / "manifest.json").is_file():
        raise ValueError("completed dataset manifest is missing")
    dataset_hashes = {
        path.relative_to(dataset_root).as_posix(): _sha256(path)
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }

    runs_root = panel_dir / "ppo-runs"
    expected_run_names = {
        f"{prefix}-seed-{seed}"
        for seed in _MODEL_SEEDS
        for prefix in ("bc-ppo", "scratch-ppo")
    }
    actual_run_names = {
        path.name for path in runs_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    } if runs_root.is_dir() else set()
    if actual_run_names != expected_run_names:
        raise ValueError("six PPO training runs are not complete")
    for name in sorted(expected_run_names):
        manifest = _read_json(runs_root / name / "run.json")
        if manifest.get("state") != "completed":
            raise ValueError("six PPO training runs are not complete")
    training_hashes = {name: _tree_sha256(runs_root / name) for name in sorted(expected_run_names)}

    development_path = panel_dir / "development" / "development.json"
    development = _read_json(development_path)
    candidates = development.get("candidates")
    if development.get("state") != "completed" or not isinstance(candidates, list):
        raise ValueError("development evidence is incomplete")
    selected_checkpoints = {"initialized": [], "control": []}
    expected_identities = {
        (condition, seed)
        for condition in ("bc_ppo", "scratch_ppo")
        for seed in _MODEL_SEEDS
    }
    chosen = [
        row for row in candidates
        if isinstance(row, Mapping) and row.get("nominal_step") == nominal
        and row.get("condition") in {"bc_ppo", "scratch_ppo"}
    ]
    identities = {(row.get("condition"), row.get("model_seed")) for row in chosen}
    if identities != expected_identities or len(chosen) != 6:
        raise ValueError("selected initialized/control checkpoints are incomplete")
    for row in sorted(chosen, key=lambda value: (value["condition"], value["model_seed"])):
        seed = row["model_seed"]
        checkpoint = Path(str(row.get("checkpoint_path", "")))
        run_path = Path(str(row.get("source_run", "")))
        if (
            not checkpoint.is_absolute() or not checkpoint.is_file()
            or not run_path.is_absolute() or run_path not in checkpoint.parents
            or row.get("actual_step") != actual_steps[str(seed)]
            or row.get("checkpoint_sha256") != _sha256(checkpoint)
        ):
            raise ValueError("selected checkpoint physical identity changed")
        item = {
            "model_seed": seed,
            "nominal_step": nominal,
            "actual_step": row["actual_step"],
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "source_run": str(run_path.resolve()),
        }
        key = "initialized" if row["condition"] == "bc_ppo" else "control"
        selected_checkpoints[key].append(item)

    if incumbent_panel is None:
        raise ValueError("--incumbent-panel is required")
    incumbents = _selected_incumbents(incumbent_panel)
    if revision is None or dirty is None:
        actual_revision, actual_dirty = _repository_identity(repository)
        revision = actual_revision if revision is None else revision
        dirty = actual_dirty if dirty is None else dirty
    if dirty is not False:
        raise ValueError("final seal requires a clean execution identity")

    report_source_paths = {}
    for source_root in (
        panel_dir / "bc-development-gate",
        panel_dir / "bc-clones",
    ):
        if source_root.is_dir():
            for path in sorted(source_root.rglob("*")):
                if path.is_file():
                    key = path.relative_to(panel_dir).as_posix()
                    report_source_paths[key] = str(path.resolve())
    report_source_hashes = {
        key: _sha256(Path(path)) for key, path in report_source_paths.items()
    }

    assigned_banks = json.loads(json.dumps(banks))
    assigned_banks["final"]["assigned"] = True
    payload = {
        "schema_version": 1,
        "state": "assigned",
        "revision": {"commit": revision, "dirty": bool(dirty)},
        "repository_root": str(Path(repository).resolve()) if verify_repository else None,
        "definition_hashes": definitions,
        "definition_paths": {
            "panel.json": str((panel_dir / "panel.json").resolve()),
            "scenario.json": str(scenario_path.resolve()),
            "seed-banks.json": str(seed_path.resolve()),
        },
        "selection_hashes": {
            "selection.json": _sha256(selection_path),
            "development.json": _sha256(development_path),
        },
        "selection_paths": {
            "selection.json": str(selection_path.resolve()),
            "development.json": str(development_path.resolve()),
        },
        "dataset_root": str(dataset_root.resolve()),
        "dataset_hashes": dataset_hashes,
        "training_run_paths": {
            name: str((runs_root / name).resolve()) for name in sorted(expected_run_names)
        },
        "training_run_hashes": training_hashes,
        "report_source_paths": report_source_paths,
        "report_source_hashes": report_source_hashes,
        "selected_checkpoints": selected_checkpoints,
        "incumbent_comparators": incumbents,
        "seed_banks": assigned_banks,
        "final": dict(assigned_banks["final"]),
    }
    if final_identity_validator is not None:
        final_identity_validator()
    _atomic_json(seal_path, payload)
    return payload


def apply_final_gate(
    matches: Sequence[Mapping[str, Any]] | None = None,
    *,
    wins: Mapping[int, int] | None = None,
    games: int = 500,
) -> GateResult:
    if games != 500:
        raise ValueError("final gate requires exactly 500 games per model seed")
    if wins is None:
        if matches is None:
            raise ValueError("final gate requires matches or per-seed wins")
        primary = [
            row for row in matches
            if row.get("condition") in (None, "initialized_ppo")
        ]
        wins = {
            seed: sum(
                row.get("outcome") == "win"
                for row in primary
                if row.get("model_seed") == seed
            )
            for seed in _MODEL_SEEDS
        }
        counts = {
            seed: sum(row.get("model_seed") == seed for row in primary)
            for seed in _MODEL_SEEDS
        }
        if set(counts.values()) != {games}:
            raise ValueError("final gate requires exactly 500 games per model seed")
    normalized = {int(seed): int(value) for seed, value in wins.items()}
    if any(value < 0 or value > games for value in normalized.values()):
        raise ValueError("final gate wins must be between zero and 500")
    if set(normalized) != set(_MODEL_SEEDS):
        raise ValueError("final gate requires all three model seeds")
    pooled = sum(normalized.values())
    passed = all(normalized[seed] >= 325 for seed in _MODEL_SEEDS) and pooled >= 1050
    return GateResult(passed, normalized, pooled, games)


def _validate_final_schedule(matches: Sequence[Mapping[str, Any]]) -> None:
    expected = {
        (seed, map_seed, seat)
        for seed in _MODEL_SEEDS
        for map_seed in range(_FINAL["seed_start"], _FINAL["seed_start"] + _FINAL["maps"])
        for seat in (0, 1)
    }
    actual = []
    for row in matches:
        if (
            not isinstance(row, Mapping)
            or row.get("model_seed") not in _MODEL_SEEDS
            or row.get("outcome") not in {"win", "loss", "draw"}
        ):
            raise ValueError("final schedule contains a malformed match")
        actual.append((row.get("model_seed"), row.get("seed"), row.get("candidate_seat")))
    if len(actual) != 1500 or len(set(actual)) != 1500 or set(actual) != expected:
        raise ValueError("final schedule is incomplete, duplicated, or outside the bank")


def evaluate_final(
    panel_dir: Path,
    *,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    server_cmd: Sequence[str],
    workers: int = 1,
) -> dict[str, Any]:
    panel_dir = Path(panel_dir)
    output_path = panel_dir / "final-evaluation.json"
    if output_path.exists():
        raise RuntimeError("final evaluation already completed")
    seal_path = panel_dir / "final-seal.json"
    if not seal_path.is_file():
        raise RuntimeError("final bank is not assigned")
    seal = _read_json(seal_path)
    _validate_final_seal(seal)
    captured_seal_sha256 = _sha256(seal_path)
    selected_evaluator = evaluator or _evaluate_standard_controllers
    work_root = panel_dir / ".final-evaluation.pending"
    if work_root.exists():
        shutil.rmtree(work_root)
    work_root.mkdir(parents=True)

    evaluation_specs: list[tuple[str, Mapping[str, Any], str]] = []
    for condition, sealed_key in (
        ("initialized_ppo", "initialized"),
        ("scratch_ppo", "control"),
    ):
        for item in seal["selected_checkpoints"][sealed_key]:
            checkpoint = Path(item["checkpoint_path"])
            controller = json.dumps({
                "kind": "snapshot",
                "path": str(checkpoint.resolve()),
                "source_run": item["source_run"],
                "algorithm": "maskable_ppo",
                "step": item["actual_step"],
            }, sort_keys=True)
            evaluation_specs.append((condition, item, controller))
    for item in seal["incumbent_comparators"]:
        controller = json.dumps({
            "kind": "snapshot",
            "path": str(Path(item["checkpoint_path"]).resolve()),
            "source_run": item["run_path"],
            "algorithm": item["algorithm"],
            "step": item["step"],
            "contract_hash": item["contract_hash"],
        }, sort_keys=True)
        evaluation_specs.append(("incumbent_ppo", item, controller))

    by_condition: dict[str, list[dict[str, Any]]] = {
        "initialized_ppo": [],
        "scratch_ppo": [],
        "incumbent_ppo": [],
    }
    try:
        for condition, item, controller in evaluation_specs:
            seed = item["model_seed"]
            result_path = work_root / condition / f"seed-{seed}" / "evaluation.json"
            selected_evaluator(
                controller,
                _FINAL["opponent"],
                games=_FINAL["maps"],
                seed_start=_FINAL["seed_start"],
                both_seats=True,
                workers=workers,
                server_cmd=list(server_cmd),
                output_path=result_path,
                environment="tactical-v2",
                evidence_dir=result_path.parent / "evidence",
                capture_trace=True,
                start_profile=_FINAL["profile"],
            )
            physical = _read_json(result_path)
            raw = physical.get("matches")
            if not isinstance(raw, list):
                raise ValueError("final schedule contains no raw matches")
            for row in raw:
                normalized = dict(row)
                normalized["model_seed"] = seed
                normalized["condition"] = condition
                by_condition[condition].append(normalized)
        for rows in by_condition.values():
            _validate_final_schedule(rows)
        if _sha256(seal_path) != captured_seal_sha256:
            raise ValueError("sealed final assignment changed during evaluation")
        _validate_final_seal(_read_json(seal_path))
        payload = {
            "schema_version": 1,
            "state": "completed",
            "seal_sha256": captured_seal_sha256,
            "schedule": dict(_FINAL),
            "matches": by_condition["initialized_ppo"],
            "comparison_matches": [
                *by_condition["scratch_ppo"],
                *by_condition["incumbent_ppo"],
            ],
        }
        _atomic_json(output_path, payload)
        return payload
    except Exception:
        output_path.unlink(missing_ok=True)
        raise

def exact_sign_test(left_only: int, right_only: int) -> float:
    if left_only < 0 or right_only < 0:
        raise ValueError("discordant counts cannot be negative")
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))


def _outcome_statistics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    from ml_lab.evaluation import wilson_interval

    total = len(rows)
    if total <= 0:
        raise ValueError("cannot summarize an empty match table")
    counts = {
        "wins": sum(row.get("outcome") == "win" for row in rows),
        "losses": sum(row.get("outcome") == "loss" for row in rows),
        "draws": sum(row.get("outcome") == "draw" for row in rows),
        "games": total,
    }
    rates = {
        name: counts[f"{name}s" if name != "loss" else "losses"] / total
        for name in ("win", "loss", "draw")
    }
    intervals = {
        name: wilson_interval(
            counts[f"{name}s" if name != "loss" else "losses"], total
        )
        for name in ("win", "loss", "draw")
    }
    diagnostics = {}
    metric_fields = {
        "rounds": "round_count",
        "decisions": "command_count",
        "action_waste": "wasted_end_turns_by_seat",
        "peak_material_advantage": "peak_normalized_advantage",
    }
    for field, source_field in metric_fields.items():
        values = []
        for row in rows:
            summary = row.get("summary")
            value = summary.get(source_field) if isinstance(summary, Mapping) else None
            if source_field == "wasted_end_turns_by_seat":
                seat = row.get("candidate_seat")
                if (
                    not isinstance(value, list)
                    or seat not in {0, 1}
                    or len(value) != 2
                ):
                    value = None
                else:
                    value = value[seat]
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append(value)
        diagnostics[field] = sum(values) / len(values) if values else None
    draw_categories: dict[str, int] = {}
    for row in rows:
        if row.get("outcome") != "draw":
            continue
        classification = row.get("classification")
        category = classification.get("primary") if isinstance(classification, Mapping) else classification
        category = category if isinstance(category, str) and category else "unclassified"
        draw_categories[category] = draw_categories.get(category, 0) + 1
    seats = {}
    for seat, name in ((0, "candidate_as_p0"), (1, "candidate_as_p1")):
        seat_rows = [row for row in rows if row.get("candidate_seat") == seat]
        seat_total = len(seat_rows)
        seats[name] = {
            "wins": sum(row.get("outcome") == "win" for row in seat_rows),
            "losses": sum(row.get("outcome") == "loss" for row in seat_rows),
            "draws": sum(row.get("outcome") == "draw" for row in seat_rows),
            "games": seat_total,
        }
        seats[name]["rates"] = {
            outcome: seats[name][f"{outcome}s" if outcome != "loss" else "losses"] / seat_total
            for outcome in ("win", "loss", "draw")
        }
    return {
        "counts": counts,
        "rates": rates,
        "confidence_intervals": intervals,
        "seats": seats,
        "diagnostics": diagnostics,
        "draw_categories": dict(sorted(draw_categories.items())),
    }


def build_final_aggregate(
    matches: Sequence[Mapping[str, Any]],
    *,
    supporting_evidence: Mapping[str, Any],
) -> dict[str, Any]:
    required_evidence = {"clone", "conversion", "bc_metrics", "learning_curves", "compute"}
    if not isinstance(supporting_evidence, Mapping) or set(supporting_evidence) != required_evidence:
        raise ValueError("sealed report evidence is incomplete")
    evidence = json.loads(json.dumps(supporting_evidence))
    rows = [dict(row) for row in matches]
    primary = [row for row in rows if row.get("condition") == "initialized_ppo"]
    _validate_final_schedule(primary)
    conditions = {}
    for condition in ("initialized_ppo", "scratch_ppo", "incumbent_ppo"):
        condition_rows = [row for row in rows if row.get("condition") == condition]
        if not condition_rows:
            continue
        conditions[condition] = {
            "per_seed": {
                str(seed): _outcome_statistics([
                    row for row in condition_rows if row.get("model_seed") == seed
                ])
                for seed in _MODEL_SEEDS
            },
            "pooled": _outcome_statistics(condition_rows),
        }
    comparisons = {}
    left_index = {
        (row["model_seed"], row["seed"], row["candidate_seat"]): row["outcome"]
        for row in primary
    }
    for condition in ("scratch_ppo", "incumbent_ppo"):
        right_rows = [row for row in rows if row.get("condition") == condition]
        if not right_rows:
            continue
        right_index = {
            (row["model_seed"], row["seed"], row["candidate_seat"]): row["outcome"]
            for row in right_rows
        }
        if len(right_index) != len(right_rows) or set(right_index) != set(left_index):
            raise ValueError(f"{condition} comparison schedule is incomplete or duplicated")
        left_only = sum(
            left_index[key] == "win" and right_index[key] != "win" for key in left_index
        )
        right_only = sum(
            left_index[key] != "win" and right_index[key] == "win" for key in left_index
        )
        comparisons[condition] = {
            "pairs": len(left_index),
            "initialized_only_wins": left_only,
            "comparator_only_wins": right_only,
            "exact_two_sided_sign_p": exact_sign_test(left_only, right_only),
        }
    gate = apply_final_gate(primary)
    return {
        "schema_version": 1,
        "primary_metric": "annihilation win rate against Random",
        "gate": {
            "passed": gate.passed,
            "per_seed_minimum": gate.per_seed_minimum,
            "pooled_minimum": gate.pooled_minimum,
            "per_seed_wins": {str(key): value for key, value in gate.per_seed_wins.items()},
            "pooled_wins": gate.pooled_wins,
        },
        "conditions": conditions,
        "comparisons": comparisons,
        "supporting_evidence": evidence,
        "matches": rows,
    }


def _render_final_report(aggregate: Mapping[str, Any]) -> str:
    conditions = aggregate["conditions"]
    initialized = conditions["initialized_ppo"]
    scratch = conditions["scratch_ppo"]
    incumbent = conditions["incumbent_ppo"]
    gate = aggregate["gate"]
    evidence = aggregate["supporting_evidence"]
    lines = [
        "# Annihilation imitation v1 results",
        "",
        "## Primary milestone gate",
        "",
        "| Model seed | Wins | Losses | Draws | Games | Win rate | Gate |",
        "|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for seed in _MODEL_SEEDS:
        summary = initialized["per_seed"][str(seed)]
        counts = summary["counts"]
        passed = counts["wins"] >= gate["per_seed_minimum"]
        lines.append(
            f"| {seed} | {counts['wins']} | {counts['losses']} | {counts['draws']} | "
            f"{counts['games']} | {summary['rates']['win']:.3%} | {'PASS' if passed else 'FAIL'} |"
        )
    pooled = initialized["pooled"]
    counts = pooled["counts"]
    lines.append(
        f"| Pooled | {counts['wins']} | {counts['losses']} | {counts['draws']} | "
        f"{counts['games']} | {pooled['rates']['win']:.3%} | "
        f"{'PASS' if gate['passed'] else 'FAIL'} |"
    )
    clone = evidence["clone"]["pooled"]
    conversion = evidence["conversion"]
    bc = evidence["bc_metrics"]
    curves = evidence["learning_curves"]
    compute = evidence["compute"]

    def curve_text(points: Sequence[Mapping[str, Any]]) -> str:
        return ", ".join(
            f"{point['nominal_step']}: {point['pooled_standard_win_rate']:.3%}"
            for point in points
        )

    def timing_text(label: str, value: Mapping[str, Any]) -> str:
        if value.get("status") == "available":
            return f"{label} {value['seconds']} seconds"
        return f"{label} unavailable ({value.get('reason', 'not recorded')})"

    lines.extend([
        "",
        "The primary result treats every draw and loss as a non-win; material diagnostics never alter the gate.",
        "",
        "## Clone",
        "",
        f"Clone pooled W/L/D: {clone['wins']}/{clone['losses']}/{clone['draws']} over {clone['games']} games.",
        "",
        "## Initialized PPO",
        "",
        f"Initialized pooled W/L/D: {counts['wins']}/{counts['losses']}/{counts['draws']}.",
        f"Win Wilson 95% interval: {pooled['confidence_intervals']['win']}.",
        f"Loss/draw Wilson 95% intervals: {pooled['confidence_intervals']['loss']} / {pooled['confidence_intervals']['draw']}.",
        f"Seat summaries: {pooled['seats']}. Diagnostics: {pooled['diagnostics']}. Draw categories: {pooled['draw_categories']}.",
    ])
    for title, condition, summary in (
        ("Scratch PPO", "scratch_ppo", scratch),
        ("Incumbent PPO", "incumbent_ppo", incumbent),
    ):
        stats = summary["pooled"]
        condition_counts = stats["counts"]
        comparison = aggregate["comparisons"][condition]
        label = "Scratch" if condition == "scratch_ppo" else "Incumbent"
        lines.extend([
            "",
            f"## {title}",
            "",
            f"{label} pooled W/L/D: {condition_counts['wins']}/{condition_counts['losses']}/{condition_counts['draws']}.",
            f"Win Wilson 95% interval: {stats['confidence_intervals']['win']}.",
            f"Loss/draw Wilson 95% intervals: {stats['confidence_intervals']['loss']} / {stats['confidence_intervals']['draw']}.",
            f"Comparator seat summaries: {stats['seats']}.",
            f"Comparator diagnostics: {stats['diagnostics']}.",
            f"Comparator draw categories: {stats['draw_categories']}.",
            f"Paired initialized-only/comparator-only wins: {comparison['initialized_only_wins']}/"
            f"{comparison['comparator_only_wins']}; exact two-sided sign p="
            f"{comparison['exact_two_sided_sign_p']}.",
        ])
    lines.extend([
        "",
        "## Conversion performance",
        "",
        f"Initialized conversion wins/games: {conversion['initialized_wins']}/{conversion['initialized_games']}.",
        "",
        "## BC metrics",
        "",
        f"BC validation loss {bc['validation_loss']}; validation accuracy {bc['validation_accuracy']}.",
        "",
        "## Learning curves",
        "",
        f"Initialized pooled standard win rate by nominal step: {curve_text(curves['initialized'])}.",
        f"Scratch pooled standard win rate by nominal step: {curve_text(curves['scratch'])}.",
        "",
        "## Compute",
        "",
        f"Compute: teacher games {compute['teacher_games']}; "
        f"{timing_text('BC wall clock', compute['bc_wall_clock'])}; "
        f"PPO environment steps {compute['ppo_environment_steps']}; "
        f"{timing_text('PPO wall clock', compute['ppo_wall_clock'])}.",
        "",
        "## Failure traces",
        "",
        f"Draw categories: {pooled['draw_categories']}. Loss and draw traces remain authoritative evidence.",
        "",
        "## Limitations",
        "",
        "This milestone covers only fixed tactical-v2 standard-3v3 games against Random.",
        "",
    ])
    return "\n".join(lines)


def _sealed_report_evidence(panel_dir: Path) -> dict[str, Any]:
    panel_dir = Path(panel_dir)
    seal = _read_json(panel_dir / "final-seal.json")
    _validate_final_seal(seal)
    source_paths = {
        name: Path(path) for name, path in seal["report_source_paths"].items()
    }
    gate_candidates = [
        path for name, path in source_paths.items()
        if name.endswith("bc-development-gate/gate.json")
    ]
    metric_paths = [
        path for name, path in source_paths.items()
        if name.startswith("bc-clones/") and name.endswith("/metrics.json")
    ]
    if len(gate_candidates) != 1 or len(metric_paths) != 3:
        raise ValueError("sealed clone and BC report evidence is incomplete")

    gate = _read_json(gate_candidates[0])
    clones = gate.get("clones")
    if not isinstance(clones, list) or len(clones) != 3:
        raise ValueError("sealed clone report evidence is incomplete")
    clone_counts = {"wins": 0, "losses": 0, "draws": 0, "games": 0}
    for clone in clones:
        if not isinstance(clone, Mapping):
            raise ValueError("sealed clone report evidence is malformed")
        raw_matches = clone.get("matches")
        if isinstance(raw_matches, list):
            outcomes = [row.get("outcome") for row in raw_matches if isinstance(row, Mapping)]
            wins = outcomes.count("win")
            losses = outcomes.count("loss")
            draws = outcomes.count("draw")
        else:
            wins = int(clone.get("wins", 0))
            losses = int(clone.get("losses", 0))
            draws = int(clone.get("draws", 0))
        clone_counts["wins"] += wins
        clone_counts["losses"] += losses
        clone_counts["draws"] += draws
        clone_counts["games"] += wins + losses + draws

    bc_rows = [_read_json(path) for path in metric_paths]
    validation_loss = sum(float(row["nll"]) for row in bc_rows) / len(bc_rows)
    validation_accuracy = sum(float(row["top1_accuracy"]) for row in bc_rows) / len(bc_rows)

    development = _read_json(Path(seal["selection_paths"]["development.json"]))
    selection = _read_json(Path(seal["selection_paths"]["selection.json"]))
    nominal = selection["selection"]["nominal_step"]
    candidates = development.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("sealed development report evidence is incomplete")
    ppo_rows = [
        row for row in candidates
        if isinstance(row, Mapping) and row.get("condition") in {"bc_ppo", "scratch_ppo"}
    ]

    def learning_curve(condition: str) -> list[dict[str, Any]]:
        points = []
        for budget in _PPO["nominal_budgets"]:
            rows = [
                row for row in ppo_rows
                if row.get("condition") == condition and row.get("nominal_step") == budget
            ]
            if len(rows) != len(_MODEL_SEEDS):
                raise ValueError("sealed learning-curve evidence is incomplete")
            outcomes = [outcome for row in rows for outcome in row.get("standard", [])]
            if not outcomes:
                raise ValueError("sealed learning-curve evidence is incomplete")
            points.append({
                "nominal_step": budget,
                "pooled_standard_win_rate": outcomes.count("win") / len(outcomes),
            })
        return points

    initialized_rows = [
        row for row in ppo_rows
        if row.get("condition") == "bc_ppo" and row.get("nominal_step") == nominal
    ]
    conversion_outcomes = [
        outcome for row in initialized_rows for outcome in row.get("conversion", [])
    ]
    if not conversion_outcomes:
        raise ValueError("conversion report evidence is unavailable")

    games_path = Path(seal["dataset_root"]) / "games.jsonl"
    teacher_games = sum(
        1 for line in games_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    ppo_steps = 0
    ppo_wall_values = []
    for run_path in seal["training_run_paths"].values():
        manifest = _read_json(Path(run_path) / "run.json")
        timesteps = manifest.get("timesteps")
        if isinstance(timesteps, bool) or not isinstance(timesteps, int):
            raise ValueError("sealed PPO compute evidence is incomplete")
        ppo_steps += timesteps
        value = manifest.get("wall_clock_seconds")
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("sealed PPO compute evidence has invalid timing")
            ppo_wall_values.append(float(value))

    bc_run_paths = [
        path for name, path in source_paths.items()
        if name.startswith("bc-clones/") and name.endswith("/run.json")
    ]
    if len(bc_run_paths) != len(_MODEL_SEEDS):
        raise ValueError("BC compute evidence requires exactly three run manifests")
    bc_wall_values = []
    for path in bc_run_paths:
        value = _read_json(path).get("wall_clock_seconds")
        if value is not None:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError("BC compute evidence has invalid timing")
            bc_wall_values.append(float(value))

    def timing(values: Sequence[float], expected: int) -> dict[str, Any]:
        if len(values) == expected:
            return {"status": "available", "seconds": sum(values)}
        return {"status": "unavailable"}

    return {
        "clone": {"pooled": clone_counts},
        "conversion": {
            "initialized_wins": conversion_outcomes.count("win"),
            "initialized_games": len(conversion_outcomes),
        },
        "bc_metrics": {
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        },
        "learning_curves": {
            "initialized": learning_curve("bc_ppo"),
            "scratch": learning_curve("scratch_ppo"),
        },
        "compute": {
            "teacher_games": teacher_games,
            "bc_wall_clock": timing(bc_wall_values, len(_MODEL_SEEDS)),
            "ppo_environment_steps": ppo_steps,
            "ppo_wall_clock": timing(
                ppo_wall_values, len(seal["training_run_paths"])
            ),
        },
    }


def load_final_publication(
    panel_dir: Path,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    panel_dir = Path(panel_dir)
    manifest = _read_json(panel_dir / "final-publication.json")
    generation = manifest.get("generation")
    if (
        manifest.get("schema_version") != 1
        or not isinstance(generation, str)
        or len(generation) != 64
        or any(character not in "0123456789abcdef" for character in generation)
    ):
        raise ValueError("final publication pointer is invalid")
    root = panel_dir / ".final-generations" / generation
    aggregate_path = root / "aggregate.json"
    report_path = root / "REPORT.md"
    if (
        not aggregate_path.is_file()
        or not report_path.is_file()
        or _sha256(aggregate_path) != manifest.get("aggregate_sha256")
        or _sha256(report_path) != manifest.get("report_sha256")
    ):
        raise ValueError("final publication generation is incomplete or changed")
    aggregate = _read_json(aggregate_path)
    report = report_path.read_text(encoding="utf-8")
    return aggregate, report, manifest


def _validated_final_evaluation(
    panel_dir: Path,
) -> tuple[list[dict[str, Any]], str]:
    panel_dir = Path(panel_dir)
    seal_path = panel_dir / "final-seal.json"
    if not seal_path.is_file():
        raise ValueError("final evaluation envelope has no current seal")
    seal = _read_json(seal_path)
    _validate_final_seal(seal)
    captured_seal_sha256 = _sha256(seal_path)
    evaluation = _read_json(panel_dir / "final-evaluation.json")
    primary = evaluation.get("matches")
    comparisons = evaluation.get("comparison_matches")
    if (
        evaluation.get("schema_version") != 1
        or evaluation.get("state") != "completed"
        or evaluation.get("schedule") != _FINAL
        or evaluation.get("seal_sha256") != captured_seal_sha256
        or not isinstance(primary, list)
        or not isinstance(comparisons, list)
    ):
        raise ValueError("final evaluation envelope does not match the current sealed bank")

    primary_rows = [dict(row) for row in primary if isinstance(row, Mapping)]
    comparison_rows = [dict(row) for row in comparisons if isinstance(row, Mapping)]
    if (
        len(primary_rows) != len(primary)
        or len(comparison_rows) != len(comparisons)
        or any(row.get("condition") != "initialized_ppo" for row in primary_rows)
    ):
        raise ValueError("final evaluation envelope contains malformed match tables")
    _validate_final_schedule(primary_rows)
    for condition in ("scratch_ppo", "incumbent_ppo"):
        rows = [row for row in comparison_rows if row.get("condition") == condition]
        _validate_final_schedule(rows)
    if any(
        row.get("condition") not in {"scratch_ppo", "incumbent_ppo"}
        for row in comparison_rows
    ):
        raise ValueError("final evaluation envelope contains malformed comparison tables")
    return [*primary_rows, *comparison_rows], captured_seal_sha256


def publish_final_report(
    panel_dir: Path,
    *,
    matches: Sequence[Mapping[str, Any]] | None = None,
    supporting_evidence: Mapping[str, Any] | None = None,
    failure_injector: Callable[[str], None] | None = None,
) -> tuple[dict[str, Any], str]:
    panel_dir = Path(panel_dir)
    captured_seal_sha256 = None
    if matches is None:
        matches, captured_seal_sha256 = _validated_final_evaluation(panel_dir)
    if supporting_evidence is None:
        supporting_evidence = _sealed_report_evidence(panel_dir)
    aggregate = build_final_aggregate(matches, supporting_evidence=supporting_evidence)
    report = _render_final_report(aggregate)

    generations = panel_dir / ".final-generations"
    generations.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".pending-", dir=generations))
    aggregate_stage = staging / "aggregate.json"
    report_stage = staging / "REPORT.md"
    moved = False
    try:
        _atomic_json(aggregate_stage, aggregate)
        if failure_injector is not None:
            failure_injector("after_staged_aggregate")
        _atomic_text(report_stage, report)
        aggregate_hash = _sha256(aggregate_stage)
        report_hash = _sha256(report_stage)
        generation = hashlib.sha256(
            f"{aggregate_hash}\n{report_hash}\n".encode("ascii")
        ).hexdigest()
        generation_root = generations / generation
        if generation_root.exists():
            if (
                _sha256(generation_root / "aggregate.json") != aggregate_hash
                or _sha256(generation_root / "REPORT.md") != report_hash
            ):
                raise ValueError("immutable final publication generation changed")
        else:
            os.replace(staging, generation_root)
            moved = True
        manifest = {
            "schema_version": 1,
            "generation": generation,
            "aggregate_sha256": aggregate_hash,
            "report_sha256": report_hash,
        }
        if failure_injector is not None:
            failure_injector("before_pointer")
        if captured_seal_sha256 is not None:
            seal_path = panel_dir / "final-seal.json"
            if _sha256(seal_path) != captured_seal_sha256:
                raise ValueError("final evaluation envelope seal changed during reporting")
            _validate_final_seal(_read_json(seal_path))
        _atomic_json(panel_dir / "final-publication.json", manifest)
    finally:
        if not moved:
            shutil.rmtree(staging, ignore_errors=True)
    return aggregate, report


def _validate_final_seal(seal: Mapping[str, Any]) -> None:
    if seal.get("state") != "assigned" or seal.get("final", {}).get("assigned") is not True:
        raise ValueError("sealed final assignment is invalid")

    for hashes_key, paths_key in (
        ("definition_hashes", "definition_paths"),
        ("selection_hashes", "selection_paths"),
    ):
        hashes = seal.get(hashes_key)
        paths = seal.get(paths_key)
        if not isinstance(hashes, Mapping) or not isinstance(paths, Mapping):
            raise ValueError("sealed provenance paths are incomplete")
        if set(hashes) != set(paths):
            raise ValueError("sealed provenance paths are incomplete")
        for name, expected in hashes.items():
            path = Path(str(paths[name]))
            if not path.is_file() or _sha256(path) != expected:
                raise ValueError(f"sealed {name} changed")

    dataset_root = Path(str(seal.get("dataset_root", "")))
    dataset_hashes = seal.get("dataset_hashes")
    if not dataset_root.is_absolute() or not isinstance(dataset_hashes, Mapping):
        raise ValueError("sealed dataset provenance is incomplete")
    current_dataset = {
        path.relative_to(dataset_root).as_posix(): _sha256(path)
        for path in sorted(dataset_root.rglob("*"))
        if path.is_file()
    }
    if current_dataset != dict(dataset_hashes):
        raise ValueError("sealed dataset changed")

    training_hashes = seal.get("training_run_hashes")
    training_paths = seal.get("training_run_paths")
    if (
        not isinstance(training_hashes, Mapping)
        or not isinstance(training_paths, Mapping)
        or set(training_hashes) != set(training_paths)
    ):
        raise ValueError("sealed training provenance is incomplete")
    for name, expected in training_hashes.items():
        path = Path(str(training_paths[name]))
        if not path.is_dir() or _tree_sha256(path) != expected:
            raise ValueError(f"sealed training run {name} changed")

    report_hashes = seal.get("report_source_hashes")
    report_paths = seal.get("report_source_paths")
    if (
        not isinstance(report_hashes, Mapping)
        or not isinstance(report_paths, Mapping)
        or set(report_hashes) != set(report_paths)
    ):
        raise ValueError("sealed report provenance is incomplete")
    for name, expected in report_hashes.items():
        path = Path(str(report_paths[name]))
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"sealed report source {name} changed")

    selected = seal.get("selected_checkpoints")
    if not isinstance(selected, Mapping):
        raise ValueError("sealed checkpoint provenance is incomplete")
    for condition in ("initialized", "control"):
        rows = selected.get(condition)
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError("sealed checkpoint provenance is incomplete")
        for row in rows:
            checkpoint = Path(str(row.get("checkpoint_path", "")))
            if not checkpoint.is_file() or _sha256(checkpoint) != row.get("checkpoint_sha256"):
                raise ValueError(f"sealed {condition} checkpoint changed")

    incumbents = seal.get("incumbent_comparators")
    if not isinstance(incumbents, list) or len(incumbents) != 3:
        raise ValueError("sealed incumbent provenance is incomplete")
    for row in incumbents:
        run_path = Path(str(row.get("run_path", "")))
        checkpoint = Path(str(row.get("checkpoint_path", "")))
        if (
            not run_path.is_dir() or _tree_sha256(run_path) != row.get("run_sha256")
            or not checkpoint.is_file() or _sha256(checkpoint) != row.get("checkpoint_sha256")
        ):
            raise ValueError("sealed incumbent comparator changed")

    repository_root = seal.get("repository_root")
    if isinstance(repository_root, str):
        revision, dirty = _repository_identity(Path(repository_root))
        if {"commit": revision, "dirty": dirty} != seal.get("revision"):
            raise ValueError("sealed code revision or dirty state changed")

if __name__ == "__main__":
    main()
