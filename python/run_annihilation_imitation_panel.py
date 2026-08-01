"""Locked Task 8 orchestration for imitation collection, pure clones, and their gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
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


def run_atomic_stage(
    destination: Path,
    definition_hashes: Mapping[str, str],
    *,
    build: Callable[[Path], None],
    validate: Callable[[Path], Mapping[str, Any]],
) -> dict[str, Any]:
    destination = Path(destination)
    hashes = dict(definition_hashes)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        stage_path = destination / "stage.json"
        if not stage_path.is_file() or _stage_definitions(stage_path) != hashes:
            raise ValueError("completed stage definition hashes do not match")
        summary = dict(validate(destination))
        result = _read_json(stage_path)
        result.update(summary=summary, reused=True)
        return result

    staging = destination.with_name(f".{destination.name}.staging")
    provenance_path = staging / ".stage-definitions.json"
    if staging.exists():
        if not provenance_path.is_file() or _stage_definitions(provenance_path) != hashes:
            raise ValueError("staged definition hashes do not match")
    else:
        staging.mkdir(parents=True)
        _atomic_json(provenance_path, {"definition_hashes": hashes})
    build(staging)
    summary = dict(validate(staging))
    result = {
        "schema_version": 1,
        "state": "completed",
        "definition_hashes": hashes,
        "summary": summary,
        "reused": False,
    }
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
    root: Path, contract: Any, scenario: ResolvedScenario
) -> Mapping[str, Any]:
    from ml_lab.imitation import load_imitation_dataset

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
    from ml_lab.cli import DEFAULT_SERVER

    return ["dotnet", str(DEFAULT_SERVER), "--scenario-file", str(scenario_path)]


def _collect_command() -> Mapping[str, Any]:
    from collect_annihilation_demonstrations import CollectionSpec, collect_partition
    from ml_lab.evaluation import DuelClient

    _panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
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

    return run_atomic_stage(
        DATASET_PATH, hashes, build=build,
        validate=lambda root: _validate_collection_dataset(root, contract, scenario),
    )


def _train_bc_command() -> Mapping[str, Any]:
    from hexwars_gym import HexWarsEnv
    from ml_lab.imitation import load_imitation_dataset

    panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()

    def build(staging: Path) -> None:
        stage_scenario = _materialize_runtime_scenario(scenario, staging, hashes)
        env = HexWarsEnv(
            _server_command(stage_scenario)[:-2],
            opponent="random", seat=0, base_seed=12_000_000,
            environment="tactical-v2", scenario_path=stage_scenario,
        )
        try:
            dataset = load_imitation_dataset(DATASET_PATH, expected_contract=env.contract)
            train_clone_runs(
                dataset=dataset, scenario=scenario, env=env, contract=env.contract,
                spaces_info=env.spaces_info, output_root=staging, panel=panel,
                definition_hashes=hashes,
            )
        finally:
            env.close()

    return run_atomic_stage(
        CLONE_RUNS_PATH, hashes, build=build,
        validate=lambda root: _validate_clone_stage(root, hashes),
    )


def _evaluate_bc_command() -> Mapping[str, Any]:
    validate_definitions()
    hashes = current_definition_hashes()
    clone_runs = [CLONE_RUNS_PATH / f"seed-{seed}" for seed in _MODEL_SEEDS]
    _validate_clone_stage(CLONE_RUNS_PATH, hashes)

    def build(staging: Path) -> None:
        evaluate_clone_gate(
            clone_runs, output_dir=staging,
            server_cmd=_server_command(SCENARIO_PATH)[:-2],
        )

    return run_atomic_stage(
        CLONE_EVALUATION_PATH, hashes, build=build,
        validate=lambda root: _validate_gate_stage(root, hashes),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_annihilation_imitation_panel.py")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("validate")
    commands.add_parser("collect")
    commands.add_parser("train-bc")
    commands.add_parser("evaluate-bc")
    commands.add_parser("train-ppo")
    commands.add_parser("evaluate-dev")
    commands.add_parser("select-budget")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "validate":
        validate_definitions()
        result: Mapping[str, Any] = {
            "state": "validated", "definition_hashes": current_definition_hashes()
        }
    elif args.command == "collect":
        result = _collect_command()
    elif args.command == "train-bc":
        result = _train_bc_command()
    elif args.command == "evaluate-bc":
        result = _evaluate_bc_command()
    elif args.command == "train-ppo":
        result = _train_ppo_command()
    elif args.command == "evaluate-dev":
        result = _evaluate_dev_command()
    elif args.command == "select-budget":
        result = _select_budget_command()
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
        if (
            not isinstance(source_run, str)
            or str(Path(source_run).resolve()) != source_run
            or algorithm != "maskable_ppo"
            or not isinstance(controller, Mapping)
            or controller.get("path") != checkpoint_path
            or controller.get("source_run") != source_run
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


def evaluate_development_candidates(
    candidates: Sequence[DevelopmentCandidate],
    *,
    output_root: Path,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
    server_cmd: Sequence[str],
    workers: int = 1,
) -> dict[str, Any]:
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
                "conversion": [],
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


def _train_ppo_command() -> Mapping[str, Any]:
    panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
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

    return run_atomic_stage(
        PPO_RUNS_PATH,
        hashes,
        build=build,
        validate=lambda root: _validate_ppo_stage(root, hashes, matrix, scenario),
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
        matches = candidate.get("matches")
        if (
            not isinstance(matches, list)
            or len(matches) != 200
            or candidate.get("standard") != [match.get("outcome") for match in matches]
            or not isinstance(candidate.get("checkpoint_sha256"), str)
        ):
            raise ValueError("development candidate schedule is incomplete")
        actual_keys = [
            (match.get("map_seed"), match.get("candidate_seat"))
            for match in matches
        ]
        expected_schedule = [
            (seed, seat)
            for seed in range(16_000_000, 16_000_100)
            for seat in (0, 1)
        ]
        if actual_keys != expected_schedule:
            raise ValueError("development candidate seed/seat schedule is incomplete")
        for match in matches:
            if (
                match.get("actual_step") != candidate.get("actual_step")
                or not _artifact_exists(root, match, "trace_path")
                or not _artifact_exists(root, match, "replay_path")
            ):
                raise ValueError("development match checkpoint or evidence is incomplete")
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
            checkpoint_sha256=candidate.get("checkpoint_sha256"),
            checkpoint_path=candidate.get("checkpoint_path"),
            source_run=candidate.get("source_run"),
            algorithm=candidate.get("algorithm"),
        )
        physical_matches: list[dict[str, Any]] = []
        for map_seed in range(
            _DEVELOPMENT["seed_start"],
            _DEVELOPMENT["seed_start"] + _DEVELOPMENT["maps"],
        ):
            output_path = (
                root
                / identity.condition
                / f"seed-{identity.model_seed}"
                / f"nominal-{identity.nominal_step}"
                / f"map-{map_seed}"
                / "evaluation.json"
            )
            _normalized, map_matches = _validated_development_map(
                root, identity, map_seed, output_path
            )
            physical_matches.extend(map_matches)
        if (
            matches != physical_matches
            or candidate.get("standard")
            != [match["outcome"] for match in physical_matches]
        ):
            raise ValueError("development aggregate does not match physical evaluations")

    return {"candidate_count": 21, "games": 4_200}


def _evaluate_dev_command() -> Mapping[str, Any]:
    panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
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
        )
        for row, candidate in zip(result["candidates"], candidates, strict=True):
            row["checkpoint_sha256"] = candidate.checkpoint_sha256
        result["definition_hashes"] = dict(hashes)
        _atomic_json(staging / "development.json", result)

    return run_atomic_stage(
        DEVELOPMENT_PATH,
        hashes,
        build=build,
        validate=lambda root: _validate_development_stage(root, hashes, scenario),
    )


def _select_budget_command() -> Mapping[str, Any]:
    panel, _banks, scenario = validate_definitions()
    del panel
    hashes = current_definition_hashes()
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
        not isinstance(controller, Mapping)
        or controller.get("step") != candidate.actual_step
        or opponent != {"kind": "scripted", "name": "random"}
        or physical.get("seed_start") != map_seed
        or physical.get("seeds") != [map_seed]
        or physical.get("reciprocal") is not True
        or physical.get("games") != 2
        or schedule
        != {
            "start_profile": "standard-3v3",
            "reference_seat_policy": "candidate-seat",
        }
        or not isinstance(raw_matches, list)
        or len(raw_matches) != 2
    ):
        raise ValueError("physical development evaluation identity is invalid")
    if candidate.checkpoint_path is not None and (
        not isinstance(controller.get("path"), str)
        or Path(controller["path"]).resolve() != Path(candidate.checkpoint_path).resolve()
    ):
        raise ValueError("physical development evaluation checkpoint is invalid")
    if candidate.source_run is not None and (
        not isinstance(controller.get("source_run"), str)
        or Path(controller["source_run"]).resolve() != Path(candidate.source_run).resolve()
    ):
        raise ValueError("physical development evaluation source run is invalid")
    if candidate.algorithm is not None and controller.get("algorithm") != candidate.algorithm:
        raise ValueError("physical development evaluation algorithm is invalid")

    normalized = _relativize_evaluation(physical, root, output_path)
    matches: list[dict[str, Any]] = []
    for seat, raw in enumerate(normalized["matches"]):
        if (
            not isinstance(raw, Mapping)
            or raw.get("seed") != map_seed
            or raw.get("candidate_seat") != seat
            or raw.get("outcome") not in {"win", "loss", "draw"}
            or "trace_path" not in raw
            or "replay_path" not in raw
        ):
            raise ValueError("physical development evaluation match is invalid")
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
                "profile": "standard-3v3",
            }
        )
        matches.append(row)
    return normalized, matches


if __name__ == "__main__":
    main()
