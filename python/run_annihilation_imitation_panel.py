"""Locked Task 8 orchestration for imitation collection, pure clones, and their gate."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
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
    runs: list[Path] = []
    for seed in panel["model_seeds"]:
        run = root / f"seed-{seed}"
        provenance_path = run / "panel-provenance.json"
        if run.exists():
            manifest = _read_json(run / "run.json")
            provenance = _read_json(provenance_path)
            if (
                manifest.get("state") != "completed"
                or manifest.get("model_seed", manifest.get("config", {}).get("model_seed")) != seed
                or provenance.get("model_seed") != seed
                or provenance.get("sampler_seed") != panel["sampler_seeds"][str(seed)]
                or provenance.get("definition_hashes") != dict(definition_hashes)
            ):
                raise ValueError(f"clone seed {seed} is incomplete or incompatible")
            runs.append(run)
            continue
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
            run_dir=run,
            config=config,
        )
        if Path(result.run_dir) != run or not (run / "run.json").is_file():
            raise ValueError(f"clone seed {seed} did not publish the expected run")
        _atomic_json(
            provenance_path,
            {
                "schema_version": 1,
                "model_seed": seed,
                "sampler_seed": panel["sampler_seeds"][str(seed)],
                "definition_hashes": dict(definition_hashes),
            },
        )
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
    for raw_run in clone_runs:
        run = Path(raw_run)
        try:
            manifest = _read_json(run / "run.json")
            provenance = _read_json(run / "panel-provenance.json")
        except ValueError as exc:
            errors.append(str(exc))
            continue
        seed = manifest.get("model_seed", manifest.get("config", {}).get("model_seed"))
        if seed not in _MODEL_SEEDS or seed in by_seed:
            errors.append(f"clone run has duplicate or unexpected model seed {seed!r}")
            continue
        by_seed[seed] = run
        if manifest.get("state") != "completed":
            errors.append(f"clone seed {seed} is not completed")
        if (
            provenance.get("model_seed") != seed
            or provenance.get("sampler_seed") != _SAMPLER_SEEDS[str(seed)]
            or provenance.get("definition_hashes") != dict(hashes)
        ):
            errors.append(f"clone seed {seed} definition provenance does not match")
        contract = manifest.get("contract")
        if not isinstance(contract, dict):
            errors.append(f"clone seed {seed} contract is missing")
            continue
        identity = {
            "contract_hash": contract.get("contract_hash"),
            "encoding_hash": contract.get("encoding_hash"),
        }
        if not all(isinstance(value, str) and value for value in identity.values()):
            errors.append(f"clone seed {seed} contract identity is invalid")
        elif expected_contract is None:
            expected_contract = identity
        elif identity != expected_contract:
            errors.append(f"clone seed {seed} contract does not match the panel")
    if set(by_seed) != set(_MODEL_SEEDS):
        errors.append("clone runs do not contain exactly model seeds 211, 223, and 227")
    return by_seed, expected_contract or {}, errors


def _standard_gate_scenario(scenario: ResolvedScenario, path: Path) -> None:
    document = copy.deepcopy(json.loads(scenario.canonical_json))
    for item in document["tactical_v2"]["start_distribution"]:
        item["basis_points"] = 10_000 if item["profile_id"] == "standard-3v3" else 0
    _atomic_json(path, document)
    resolve_scenario(
        environment="tactical-v2", scenario_file=path, template_id=None,
        enforce_round_cap_minimum=True,
    )


def _artifact_exists(match: Mapping[str, Any], field: str) -> bool:
    raw = match.get(field)
    return isinstance(raw, str) and Path(raw).is_file()


def evaluate_clone_gate(
    clone_runs: Sequence[Path],
    *,
    output_dir: Path = CLONE_EVALUATION_PATH,
    server_cmd: Sequence[str] | None = None,
    workers: int = 1,
    evaluator: Callable[..., Mapping[str, Any]] | None = None,
) -> Mapping[str, Any]:
    from ml_lab.cli import DEFAULT_SERVER
    from ml_lab.evaluation import evaluate_controllers

    panel, _banks, scenario = validate_definitions()
    hashes = current_definition_hashes()
    runs, expected_contract, errors = _clone_metadata(clone_runs, hashes)
    if errors:
        return _failed_gate(errors, hashes)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    gate_scenario = output_root / "standard-gate-scenario.json"
    _standard_gate_scenario(scenario, gate_scenario)
    command = list(server_cmd) if server_cmd is not None else ["dotnet", str(DEFAULT_SERVER)]
    command.extend(["--scenario-file", str(gate_scenario)])
    selected_evaluator = evaluator or evaluate_controllers
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
            result = selected_evaluator(
                f"run:{run}", panel["clone_gate"]["opponent"], games=1,
                seed_start=map_seed, both_seats=True, workers=workers,
                server_cmd=command, output_path=map_root / "evaluation.json",
                environment="tactical-v2", evidence_dir=map_root / "evidence",
                capture_trace=True,
            )
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
                    not _artifact_exists(match, "trace_path")
                    or not _artifact_exists(match, "replay_path")
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
            {"model_seed": model_seed, "wins": wins, "games": len(matches), "matches": matches}
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
    gate = _read_json(Path(root) / "gate.json")
    if gate.get("definition_hashes") != dict(hashes):
        raise ValueError("clone gate definition hashes do not match")
    if gate.get("state") != "completed" or gate.get("passed") is not True:
        raise ValueError("clone gate did not pass")
    if gate.get("per_seed_wins", {}).keys() != {str(seed) for seed in _MODEL_SEEDS}:
        raise ValueError("clone gate model-seed results are incomplete")
    clones = gate.get("clones")
    if (
        not isinstance(clones, list)
        or len(clones) != 3
        or any(item.get("games") != 200 for item in clones if isinstance(item, dict))
    ):
        raise ValueError("clone gate expected outputs are incomplete")
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
