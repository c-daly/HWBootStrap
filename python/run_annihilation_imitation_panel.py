"""Locked Task 8 orchestration for imitation collection, pure clones, and their gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import zipfile
from collections.abc import Callable, Mapping, Sequence
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
        "collection", "behavioral_cloning", "clone_gate",
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
    if panel["collection"] != _COLLECTION or panel["clone_gate"] != _CLONE_GATE:
        raise ValueError("panel collection or clone gate changed")
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
    if not metrics or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in metrics.values()
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
            path = Path(raw).resolve()
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
    command = _server_command(SCENARIO_PATH)
    probe = DuelClient(command, environment="tactical-v2")
    try:
        contract = probe.contract
    finally:
        probe.close()

    def build(staging: Path) -> None:
        factory = lambda _worker: DuelClient(command, environment="tactical-v2")
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
        env = HexWarsEnv(
            _server_command(SCENARIO_PATH)[:-2],
            opponent="random", seat=0, base_seed=12_000_000,
            environment="tactical-v2", scenario_path=SCENARIO_PATH,
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
    else:
        raise AssertionError(f"unreachable command {args.command!r}")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
