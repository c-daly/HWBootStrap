from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from zipfile import ZipFile

import numpy as np

import pytest


MODULE_NAME = "run_annihilation_imitation_panel"
EXPECTED_BANKS = {
    "greedy_demonstrations": (11_000_000, 11_499_999, True),
    "search_demonstrations": (11_500_000, 11_999_999, True),
    "bc_validation": (12_000_000, 12_099_999, True),
    "ppo_replicates": (13_000_000, 15_999_999, True),
    "development": (16_000_000, 16_000_099, True),
    "final": (17_000_000, 17_000_249, False),
}
LOCKED_WEIGHTS = [
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


FULL_GENERATED_PATHS = [
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
    "python/panels/annihilation-imitation-v1/evidence/",
]
PUBLISHABLE_RESULT_PATHS = [
    "python/panels/annihilation-imitation-v1/aggregate.json",
    "python/panels/annihilation-imitation-v1/REPORT.md",
]


def _execution_identity(
    module,
    *,
    commit: str = "a" * 40,
    source_tree: str = "b" * 40,
    dirty: bool = False,
    definition_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "commit": commit,
        "source_tree": source_tree,
        "dirty": dirty,
        "policy": {
            "required_clean": True,
            "ignored_generated_paths": list(FULL_GENERATED_PATHS),
            "publishable_result_paths": list(PUBLISHABLE_RESULT_PATHS),
        },
        "definition_hashes": (
            dict(definition_hashes)
            if definition_hashes is not None
            else module.current_definition_hashes()
        ),
    }


def _command_identity_kwargs(
    tmp_path: Path,
    module,
    *,
    commit: str = "a" * 40,
    source_tree: str = "b" * 40,
) -> tuple[dict[str, Any], dict[str, Any]]:
    identity = _execution_identity(
        module, commit=commit, source_tree=source_tree,
    )
    path = tmp_path / "execution-identity.json"
    path.write_text(json.dumps(identity), encoding="utf-8")
    kwargs = {
        "execution_identity_path": path,
        "repository": tmp_path,
        "repository_identity_provider": lambda _repository: {
            "commit": commit,
            "source_tree": source_tree,
            "dirty": False,
        },
    }
    return identity, kwargs


def _subject():
    specification = importlib.util.find_spec(MODULE_NAME)
    assert specification is not None, "Task 8 panel orchestrator is missing"
    return importlib.import_module(MODULE_NAME)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _production_controller_identity(
    *,
    kind: str,
    path: Path | None = None,
    algorithm: str | None = None,
    step: int | None = None,
    name: str | None = None,
) -> dict[str, Any]:
    identity = {
        "kind": kind,
        "inference_mode": "deterministic",
        "path": str(path.resolve()) if path is not None else None,
        "algorithm": algorithm,
        "step": step,
        "contract_hash": None,
        "contract_version": None,
        "environment": None,
        "encoding_hash": None,
        "contract": None,
        "observation_size": None,
        "action_size": None,
        "legacy": False,
        "promotable": False,
    }
    if name is not None:
        identity["name"] = name
    return identity



def test_panel_orchestrator_is_importable() -> None:
    """Removing the Task 8 command module must make the protocol unusable."""

    assert importlib.util.find_spec(MODULE_NAME) is not None


def test_locked_definitions_have_disjoint_exact_namespaces_and_values() -> None:
    """A changed namespace, model seed, weight, or reward must invalidate the preregistration."""

    module = _subject()
    panel, banks, scenario = module.validate_definitions()

    assert panel["model_seeds"] == [211, 223, 227]
    assert panel["sampler_seeds"] == {"211": 211, "223": 223, "227": 227}
    assert panel["outcome_rewards"] == {"win": 1, "loss": -1, "draw": 0}
    assert panel["collection"] == {
        "standard_decisions_minimum": 100_000,
        "conversion_decisions_minimum": 50_000,
        "validation_standard_maps": 100,
        "validation_conversion_maps_per_profile": 20,
    }
    assert panel["clone_gate"] == {
        "maps": 100,
        "games_per_clone": 200,
        "seed_start": 16_000_000,
        "profile": "standard-3v3",
        "opponent": "random",
        "per_seed_win_rate_minimum_basis_points": 3000,
        "pooled_win_rate_minimum_basis_points": 4000,
    }
    assert set(banks) == {"schema_version", *EXPECTED_BANKS}
    actual_ranges = []
    for name, (start, stop, assigned) in EXPECTED_BANKS.items():
        assert banks[name] == {"start": start, "stop": stop, "assigned": assigned}
        actual_ranges.append((start, stop, name))
    for previous, following in zip(sorted(actual_ranges), sorted(actual_ranges)[1:]):
        assert previous[1] < following[0]
    assert [dict(item) for item in scenario.document["tactical_v2"]["start_distribution"]] == LOCKED_WEIGHTS
    assert dict(scenario.document["reward"]) == {
        "shape_scale": 0.01,
        "step_penalty": 0.005,
        "closing_weight": 0,
        "draw_credit_weight": 0,
        "points_weight": 0.5,
    }


def test_locked_behavioral_cloning_uses_cuda() -> None:
    panel, _banks, _scenario = _subject().validate_definitions()
    assert panel["behavioral_cloning"] == {
        "batch_size": 256,
        "learning_rate": 0.0003,
        "max_epochs": 50,
        "patience": 5,
        "standard_fraction_basis_points": 7000,
        "device": "cuda",
    }


def test_emit_bc_progress_prints_one_sorted_flushed_json_object(monkeypatch):
    calls = []
    monkeypatch.setattr(
        builtins, "print",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    event = {"event": "bc_epoch", "schema_version": 1, "epoch": 1}
    _subject().emit_bc_progress(event)
    assert calls == [((json.dumps(event, sort_keys=True),), {"flush": True})]


def test_definition_hashes_bind_the_exact_scenario_and_seed_bank_bytes() -> None:
    """Editing either locked subordinate definition without updating its registration must fail."""

    module = _subject()
    panel, _banks, _scenario = module.validate_definitions()

    assert panel["definition_hashes"] == {
        "scenario_sha256": _sha256(module.SCENARIO_PATH),
        "seed_banks_sha256": _sha256(module.SEED_BANKS_PATH),
    }
    hashes = module.current_definition_hashes()
    assert hashes == {
        "panel_sha256": _sha256(module.PANEL_PATH),
        "scenario_sha256": _sha256(module.SCENARIO_PATH),
        "seed_banks_sha256": _sha256(module.SEED_BANKS_PATH),
    }


def test_validate_definitions_rejects_changed_registered_definition(tmp_path: Path) -> None:
    """A changed definition hash must stop every command before work begins."""

    module = _subject()
    scenario = tmp_path / "scenario.json"
    panel = tmp_path / "panel.json"
    banks = tmp_path / "seed-banks.json"
    scenario.write_bytes(module.SCENARIO_PATH.read_bytes())
    panel.write_bytes(module.PANEL_PATH.read_bytes())
    banks.write_bytes(module.SEED_BANKS_PATH.read_bytes())
    changed = _json(banks)
    changed["final"]["assigned"] = True
    banks.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(ValueError, match="seed-banks definition hash"):
        module.validate_definitions(
            panel_path=panel,
            seed_banks_path=banks,
            scenario_path=scenario,
        )


def test_atomic_stage_publishes_only_after_validation_and_is_restart_safe(tmp_path: Path) -> None:
    """A completed stage must publish once and a restart must not rewrite or rebuild it."""

    module = _subject()
    destination = tmp_path / "published"
    definitions = {"panel_sha256": "a" * 64}
    calls: list[Path] = []

    def build(staging: Path) -> None:
        calls.append(staging)
        (staging / "payload.txt").write_text("complete", encoding="utf-8")

    def validate(root: Path) -> dict[str, int]:
        assert (root / "payload.txt").read_text(encoding="utf-8") == "complete"
        return {"expected_outputs": 1, "manifest_count": 1}

    first = module.run_atomic_stage(destination, definitions, build=build, validate=validate)
    before = {path.relative_to(destination): path.stat().st_mtime_ns for path in destination.rglob("*")}
    second = module.run_atomic_stage(destination, definitions, build=build, validate=validate)
    after = {path.relative_to(destination): path.stat().st_mtime_ns for path in destination.rglob("*")}

    assert first["state"] == "completed"
    assert first["reused"] is False
    assert second["reused"] is True
    assert calls == [tmp_path / ".published.staging"]
    assert before == after
    assert _json(destination / "stage.json")["definition_hashes"] == definitions
    assert not (tmp_path / ".published.staging").exists()


def test_atomic_stage_keeps_invalid_work_staged_and_rejects_changed_hash(tmp_path: Path) -> None:
    """Validation failure must not publish, and changed definitions must not reuse old staging."""

    module = _subject()
    destination = tmp_path / "published"
    first_hashes = {"panel_sha256": "a" * 64}

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("bad", encoding="utf-8")

    def reject(_root: Path) -> dict[str, int]:
        raise ValueError("expected output is missing")

    with pytest.raises(ValueError, match="expected output"):
        module.run_atomic_stage(destination, first_hashes, build=build, validate=reject)
    assert not destination.exists()
    assert (tmp_path / ".published.staging").is_dir()

    with pytest.raises(ValueError, match="definition hashes"):
        module.run_atomic_stage(
            destination,
            {"panel_sha256": "b" * 64},
            build=build,
            validate=reject,
        )


def test_clone_run_construction_uses_fresh_configs_and_distinct_destinations(tmp_path: Path) -> None:
    """Reusing a trainer configuration or run directory could share initialization state."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    dataset_manifest = tmp_path / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir()
    dataset_manifest.write_text('{"state":"complete"}', encoding="utf-8")
    dataset = SimpleNamespace(root=dataset_manifest.parent)
    contract = SimpleNamespace(contract_hash="c" * 64, encoding_hash="e" * 64)
    calls: list[dict[str, Any]] = []

    def trainer(**kwargs):
        calls.append(kwargs)
        run_dir = kwargs["run_dir"]
        _write_clone_artifact(
            run_dir,
            module,
            kwargs["config"].model_seed,
            dataset_manifest=dataset_manifest,
        )
        return SimpleNamespace(run_dir=run_dir)

    runs = module.train_clone_runs(
        dataset=dataset,
        scenario=scenario,
        env=object(),
        contract=contract,
        spaces_info={"channels": 1},
        output_root=tmp_path / "clones",
        panel=panel,
        definition_hashes={"panel_sha256": "a" * 64},
        trainer=trainer,
    )

    assert [path.name for path in runs] == ["seed-211", "seed-223", "seed-227"]
    assert [call["config"].model_seed for call in calls] == [211, 223, 227]
    assert len({id(call["config"]) for call in calls}) == 3
    assert len({call["run_dir"] for call in calls}) == 3
    assert all(call["dataset"] is dataset for call in calls)
    assert [call["config"].device for call in calls] == ["cuda", "cuda", "cuda"]
    assert all(call["progress"] is module.emit_bc_progress for call in calls)
    for seed, run in zip((211, 223, 227), runs, strict=True):
        provenance = _json(run / "panel-provenance.json")
        assert provenance["model_seed"] == seed
        assert provenance["sampler_seed"] == seed
        assert provenance["definition_hashes"] == {"panel_sha256": "a" * 64}


def _write_clone_artifact(
    run: Path,
    module,
    seed: int,
    *,
    dataset_manifest: Path,
) -> None:
    run.mkdir(parents=True)
    checkpoint = run / "checkpoints" / "step_000000000.zip"
    checkpoint.parent.mkdir()
    with ZipFile(checkpoint, "w") as archive:
        archive.writestr("data", "{}")
    np.savez(
        run / "actor-fixtures.npz",
        observations=np.zeros((1, 2), dtype=np.float32),
        legal_masks=np.ones((1, 3), dtype=np.bool_),
    )
    (run / "scenario.json").write_bytes(module.SCENARIO_PATH.read_bytes())
    dataset_hash = _sha256(dataset_manifest)
    contract = {
        "environment": "tactical-v2",
        "version": "tactical-v2",
        "contract_hash": "c" * 64,
        "encoding_hash": "e" * 64,
        "observation_size": 2,
        "action_size": 3,
        "board": {},
        "roster": [],
        "reward": {},
        "semantics": {"start_profiles": [{"id": "standard-3v3"}]},
    }
    config = {
        "model_seed": seed,
        "batch_size": 256,
        "learning_rate": 0.0003,
        "max_epochs": 50,
        "patience": 5,
        "device": "cuda",
    }
    training_device = {
        "requested": "cuda",
        "resolved": "cuda:0",
        "torch_version": "2.12.1+cu130",
        "cuda_runtime": "13.0",
        "device_index": 0,
        "device_name": "NVIDIA GeForce RTX 5070",
    }
    (run / "bc.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_seed": seed,
                "dataset_manifest_sha256": dataset_hash,
                "config": config,
                "best_epoch": 1,
                "epochs_trained": 1,
                "training_device": training_device,
                "publication_device": "cpu",
                "best_validation_nll": 1.0,
            }
        ),
        encoding="utf-8",
    )
    (run / "training-history.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_seed": seed,
                "training_device": training_device,
                "publication_device": "cpu",
                "epochs": [
                    {
                        "schema_version": 1,
                        "event": "bc_epoch",
                        "model_seed": seed,
                        "device": "cuda:0",
                        "epoch": 1,
                        "max_epochs": 50,
                        "batches": 1,
                        "examples": 256,
                        "mean_training_loss": 1.0,
                        "validation_nll": 1.0,
                        "top1_accuracy": 0.5,
                        "top3_accuracy": 0.75,
                        "top5_accuracy": 0.9,
                        "best_epoch": 1,
                        "best_validation_nll": 1.0,
                        "epochs_without_improvement": 0,
                        "patience": 5,
                        "epoch_seconds": 0.1,
                        "elapsed_seconds": 0.1,
                        "examples_per_second": 2560.0,
                        "sampling_seconds": 0.01,
                        "transfer_forward_seconds": 0.02,
                        "optimization_seconds": 0.03,
                        "validation_seconds": 0.04,
                        "unclassified_seconds": 0.0,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    (run / "metrics.json").write_text(
        json.dumps(
            {
                "nll": 1.0,
                "top1_accuracy": 0.5,
                "top3_accuracy": 0.75,
                "top5_accuracy": 0.9,
                "expected_calibration_error": 0.1,
                "mean_end_turn_probability": 0.2,
                "illegal_probability": 0.0,
                "strata": {"teacher/greedy": {"count": 1, "accuracy": 0.5}},
            }
        ),
        encoding="utf-8",
    )
    (run / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "state": "completed",
                "timesteps": 0,
                "latest_checkpoint": "checkpoints/step_000000000.zip",
                "latest_checkpoint_step": 0,
                "model_seed": seed,
                "config": {"model_seed": seed, "behavioral_cloning": config},
                "contract": contract,
                "scenario": {
                    "path": "scenario.json",
                    "template_id": "annihilation-imitation-v1",
                    "schema_version": 1,
                },
                "dataset_manifest_sha256": dataset_hash,
                "bc_config": config,
                "best_epoch": 1,
            }
        ),
        encoding="utf-8",
    )


def _write_clone_runs(root: Path, module, *, changed_hash: bool = False) -> list[Path]:
    hashes = module.current_definition_hashes()
    dataset_manifest = root.parent / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir(parents=True, exist_ok=True)
    dataset_manifest.write_text('{"state":"complete"}', encoding="utf-8")
    runs = []
    for seed in (211, 223, 227):
        run = root / f"seed-{seed}"
        _write_clone_artifact(run, module, seed, dataset_manifest=dataset_manifest)
        provenance_hashes = dict(hashes)
        if changed_hash and seed == 211:
            provenance_hashes["panel_sha256"] = "f" * 64
        (run / "panel-provenance.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "model_seed": seed,
                    "sampler_seed": seed,
                    "definition_hashes": provenance_hashes,
                    "dataset_manifest": str(dataset_manifest.resolve()),
                    "dataset_manifest_sha256": _sha256(dataset_manifest),
                }
            ),
            encoding="utf-8",
        )
        runs.append(run)
    return runs


def _validate_cuda_clone(run: Path, module) -> dict[str, str]:
    return module._validate_clone_run(
        run,
        211,
        module.current_definition_hashes(),
        expected_scenario=module.validate_definitions()[2],
        expected_device="cuda",
    )


def test_clone_validation_rejects_missing_training_history(tmp_path: Path) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    (run / "training-history.json").unlink()

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


def test_clone_validation_rejects_non_contiguous_training_history(tmp_path: Path) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    history["epochs"][0]["epoch"] = 2
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


@pytest.mark.parametrize(
    ("field", "value"),
    [("mean_training_loss", float("inf")), ("epoch_seconds", float("nan"))],
)
def test_clone_validation_rejects_non_finite_training_history(
    tmp_path: Path, field: str, value: float,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    history["epochs"][0][field] = value
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda row: row.pop("sampling_seconds"),
        lambda row: row.__setitem__("optimization_seconds", -1.0),
        lambda row: row.__setitem__("unclassified_seconds", 9.0),
    ],
)
def test_clone_validation_rejects_invalid_phase_timing(
    tmp_path: Path, mutation,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    mutation(history["epochs"][0])
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="timing"):
        _validate_cuda_clone(run, module)


def test_clone_validation_accepts_cuda_to_cpu_best_nll_rounding_difference(
    tmp_path: Path,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    bc_path = run / "bc.json"
    bc = _json(bc_path)
    bc["best_validation_nll"] = 1.0000005
    bc_path.write_text(json.dumps(bc), encoding="utf-8")

    _validate_cuda_clone(run, module)


def test_clone_validation_rejects_material_best_nll_difference(
    tmp_path: Path,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    bc_path = run / "bc.json"
    bc = _json(bc_path)
    bc["best_validation_nll"] = 1.0001
    bc_path.write_text(json.dumps(bc), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("examples", 255),
        ("examples_per_second", 2559.0),
        ("epoch_seconds", 0.0),
    ],
)
def test_clone_validation_rejects_inconsistent_training_history_counts_or_rate(
    tmp_path: Path, field: str, value: int | float,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    history["epochs"][0][field] = value
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


@pytest.mark.parametrize(
    ("location", "field", "value"),
    [("history", "model_seed", 223), ("epoch", "patience", 4)],
)
def test_clone_validation_rejects_training_history_seed_or_config_mismatch(
    tmp_path: Path, location: str, field: str, value: Any,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    target = history if location == "history" else history["epochs"][0]
    target[field] = value
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


def test_clone_validation_rejects_invalid_training_history_patience_counter(
    tmp_path: Path,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    history_path = run / "training-history.json"
    history = _json(history_path)
    history["epochs"][0]["epochs_without_improvement"] = 1
    history_path.write_text(json.dumps(history), encoding="utf-8")

    with pytest.raises(ValueError, match="training history"):
        _validate_cuda_clone(run, module)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("training_device", None),
        ("requested", "cpu"),
        ("resolved", "cuda:not-an-index"),
        ("torch_version", ""),
        ("cuda_runtime", None),
        ("device_index", True),
        ("device_name", ""),
    ],
)
def test_clone_validation_rejects_missing_or_malformed_cuda_device_provenance(
    tmp_path: Path, field: str, value: Any,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    bc_path = run / "bc.json"
    bc = _json(bc_path)
    if field == "training_device":
        bc.pop(field)
    else:
        bc["training_device"][field] = value
    bc_path.write_text(json.dumps(bc), encoding="utf-8")

    with pytest.raises(ValueError, match="device provenance"):
        _validate_cuda_clone(run, module)


def test_clone_validation_rejects_device_provenance_different_from_locked_panel(
    tmp_path: Path,
) -> None:
    module = _subject()
    run = _write_clone_runs(tmp_path / "runs", module)[0]
    manifest_path = run / "run.json"
    manifest = _json(manifest_path)
    manifest["bc_config"]["device"] = "cpu"
    manifest["config"]["behavioral_cloning"]["device"] = "cpu"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    bc_path = run / "bc.json"
    bc = _json(bc_path)
    bc["config"]["device"] = "cpu"
    bc_path.write_text(json.dumps(bc), encoding="utf-8")

    with pytest.raises(ValueError, match="device"):
        _validate_cuda_clone(run, module)


def _write_actor_initialization(run_dir: Path, training_run) -> None:
    source = Path(training_run.config.actor_init_source)
    source_manifest = _json(source / "run.json")
    checkpoint_relative = source_manifest["latest_checkpoint"]
    provenance = {
        "schema_version": 1,
        "kind": "actor_only",
        "actor_modules": [
            "features_extractor",
            "mlp_extractor.policy_net",
            "action_net",
        ],
        "device": "cpu",
        "comparison_rtol": 0.0,
        "comparison_atol": 0.0,
        "maximum_absolute_logit_difference": 0.0,
        "source_run": str(source.resolve()),
        "source_checkpoint": checkpoint_relative,
        "source_checkpoint_sha256": _sha256(source / checkpoint_relative),
        "source_actor_fixtures_sha256": _sha256(source / "actor-fixtures.npz"),
        "source_run_manifest_sha256": _sha256(source / "run.json"),
        "source_bc_sha256": _sha256(source / "bc.json"),
        "source_dataset_manifest_sha256": source_manifest["dataset_manifest_sha256"],
        "source_contract_hash": source_manifest["contract"]["contract_hash"],
        "source_encoding_hash": source_manifest["contract"]["encoding_hash"],
    }
    (run_dir / "initialization.json").write_text(
        json.dumps(provenance), encoding="utf-8"
    )


class ControlledEvaluator:
    """Local substitute for the external GymServer evaluation boundary."""

    def __init__(self, wins: dict[int, int], mutation: str | None = None) -> None:
        self.wins = wins
        self.mutation = mutation
        self.calls: list[dict[str, Any]] = []
        self.games_seen = {seed: 0 for seed in wins}

    def __call__(self, p0: str, p1: str, **kwargs):
        run = Path(p0.removeprefix("run:"))
        seed = _json(run / "run.json")["model_seed"]
        map_seed = kwargs["seed_start"]
        matches = []
        for seat in (0, 1):
            index = self.games_seen[seed]
            self.games_seen[seed] += 1
            outcome = "win" if index < self.wins[seed] else "loss"
            match = {"seed": map_seed, "candidate_seat": seat, "outcome": outcome}
            if outcome != "win":
                evidence = Path(kwargs["evidence_dir"])
                evidence.mkdir(parents=True, exist_ok=True)
                trace = evidence / f"seed-{map_seed}-seat-{seat}.json"
                replay = evidence / f"seed-{map_seed}-seat-{seat}.replay"
                trace.write_text("{}", encoding="utf-8")
                replay.write_bytes(b"replay")
                match.update(trace_path=str(trace), replay_path=str(replay))
            matches.append(match)
        if self.mutation == "missing-seat" and seed == 211 and map_seed == 16_000_000:
            matches.pop()
        if self.mutation == "duplicate-seat" and seed == 211 and map_seed == 16_000_000:
            matches.append(dict(matches[0]))
        contract_hash = "x" * 64 if self.mutation == "contract" and seed == 211 else "c" * 64
        result = {
            "schema_version": 1,
            "schedule": {
                "start_profile": kwargs.get("start_profile"),
                "reference_seat_policy": "candidate-seat",
            },
            "candidate": {"contract_hash": contract_hash, "encoding_hash": "e" * 64},
            "opponent": {"kind": "builtin", "name": "random"},
            "seed_start": map_seed,
            "seeds": [map_seed],
            "reciprocal": True,
            "games": len(matches),
            "matches": matches,
        }
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(json.dumps(result), encoding="utf-8")
        self.calls.append({"p0": p0, "p1": p1, **kwargs})
        return result



def test_evaluate_clone_gate_runs_exact_development_schedule_and_passes(tmp_path: Path) -> None:
    """Changing the map count, seat reciprocity, opponent, or profile must fail the gate contract."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    evaluator = ControlledEvaluator({211: 60, 223: 90, 227: 90})

    result = module.evaluate_clone_gate(
        runs,
        output_dir=tmp_path / "evaluation",
        evaluator=evaluator,
        server_cmd=["fake-server"],
    )

    assert result["state"] == "completed"
    assert result["passed"] is True
    assert result["per_seed_wins"] == {211: 60, 223: 90, 227: 90}
    assert result["pooled_win_rate"] == 0.4
    assert result["integrity_errors"] == []
    assert len(evaluator.calls) == 300
    for clone_index in range(3):
        calls = evaluator.calls[clone_index * 100:(clone_index + 1) * 100]
        assert [call["seed_start"] for call in calls] == list(range(16_000_000, 16_000_100))
        assert all(call["games"] == 1 and call["both_seats"] is True for call in calls)
        assert all(call["p1"] == "random" and call["environment"] == "tactical-v2" for call in calls)
        assert all(call["capture_trace"] is True for call in calls)
        assert all(call.get("start_profile") == "standard-3v3" for call in calls)
    assert not (tmp_path / "evaluation" / "standard-gate-scenario.json").exists()
    launched_scenario = Path(evaluator.calls[0]["server_cmd"][-1])
    assert json.loads(launched_scenario.read_text(encoding="utf-8")) == json.loads(
        module.validate_definitions()[2].canonical_json
    )
    assert _json(launched_scenario)["tactical_v2"]["start_distribution"] == LOCKED_WEIGHTS
    assert all(
        not Path(match[field]).is_absolute()
        and (tmp_path / "evaluation" / match[field]).is_file()
        for clone in result["clones"]
        for match in clone["matches"]
        if match["outcome"] in {"draw", "loss"}
        for field in ("trace_path", "replay_path")
    )



def test_atomic_gate_publication_keeps_relative_evidence_references_valid(
    tmp_path: Path,
) -> None:
    """Publishing a staged gate must not strand JSON references under the old staging path."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    evaluator = ControlledEvaluator({211: 60, 223: 90, 227: 90})
    destination = tmp_path / "published-gate"
    hashes = module.current_definition_hashes()

    def build(staging: Path) -> None:
        result = module.evaluate_clone_gate(
            runs,
            output_dir=staging,
            evaluator=evaluator,
            server_cmd=["fake-server"],
        )
        assert result["passed"] is True

    module.run_atomic_stage(
        destination,
        hashes,
        build=build,
        validate=lambda _root: {"development_games": 600},
    )
    published = _json(destination / "gate.json")
    references = [
        match[field]
        for clone in published["clones"]
        for match in clone["matches"]
        if match["outcome"] in {"loss", "draw"}
        for field in ("trace_path", "replay_path")
    ]
    assert references
    assert all(not Path(reference).is_absolute() for reference in references)
    assert all((destination / reference).is_file() for reference in references)
    assert all(".staging" not in reference for reference in references)


@pytest.mark.parametrize(
    ("wins", "expected_pooled"),
    [
        ({211: 79, 223: 80, 227: 80}, 239 / 600),
        ({211: 59, 223: 90, 227: 91}, 240 / 600),
    ],
)
def test_clone_gate_rejects_pooled_39_point_9_or_one_29_point_5_seed(
    tmp_path: Path, wins: dict[int, int], expected_pooled: float
) -> None:
    """Weakening either literal threshold would incorrectly advance a failed clone panel."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    result = module.evaluate_clone_gate(
        runs,
        output_dir=tmp_path / "evaluation",
        evaluator=ControlledEvaluator(wins),
        server_cmd=["fake-server"],
    )

    assert result["state"] == "failed"
    assert result["passed"] is False
    assert result["pooled_win_rate"] == expected_pooled


@pytest.mark.parametrize("mutation", ["missing-seat", "duplicate-seat", "contract"])
def test_clone_gate_integrity_failures_override_passing_rates(tmp_path: Path, mutation: str) -> None:
    """Malformed schedules and mismatched contracts must fail even with perfect wins."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    result = module.evaluate_clone_gate(
        runs,
        output_dir=tmp_path / "evaluation",
        evaluator=ControlledEvaluator({211: 200, 223: 200, 227: 200}, mutation),
        server_cmd=["fake-server"],
    )

    assert result["state"] == "failed"
    assert result["passed"] is False
    assert result["integrity_errors"]


def test_clone_gate_rejects_changed_definition_provenance_before_evaluation(tmp_path: Path) -> None:
    """A clone built under different definitions must never enter the development gate."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module, changed_hash=True)
    evaluator = ControlledEvaluator({211: 200, 223: 200, 227: 200})

    result = module.evaluate_clone_gate(
        runs,
        output_dir=tmp_path / "evaluation",
        evaluator=evaluator,
        server_cmd=["fake-server"],
    )

    assert result["state"] == "failed"
    assert result["passed"] is False
    assert any("definition" in error for error in result["integrity_errors"])
    assert evaluator.calls == []


def test_parser_exposes_restart_safe_task_9_commands() -> None:
    """Omitting a staged Task 9 command would make the preregistered workflow manual."""

    module = _subject()
    parser = module.build_parser()
    for command in (
        "validate", "collect", "train-bc", "evaluate-bc",
        "train-ppo", "evaluate-dev", "select-budget",
    ):
        assert parser.parse_args([command]).command == command



def test_real_evaluation_boundary_forces_standard_profile_without_changing_contract(
    tmp_path: Path,
) -> None:
    """Task 8 must force the profile through evaluate_matchup, not rewrite scenario semantics."""

    module = _subject()
    assert callable(getattr(module, "_evaluate_standard_controllers", None))
    from ml_lab.contracts import EnvironmentContract

    class ProfileClient:
        def __init__(self) -> None:
            self.contract = EnvironmentContract(
                version="tactical-v2",
                contract_hash="c" * 64,
                encoding_hash="e" * 64,
                observation_size=2,
                action_size=3,
                board={},
                roster=[],
                reward={},
                semantics={"start_profiles": [{"id": "standard-3v3"}]},
            )
            self.resets: list[dict[str, Any]] = []

        def reset(self, **kwargs):
            self.resets.append(kwargs)
            return {
                "terminated": True,
                "truncated": False,
                "winner": kwargs["reference_seat"],
            }

        def close(self) -> None:
            return None

    clients: list[ProfileClient] = []

    def factory(_worker: int) -> ProfileClient:
        client = ProfileClient()
        clients.append(client)
        return client

    result = module._evaluate_standard_controllers(
        "random",
        "random",
        games=1,
        seed_start=16_000_000,
        both_seats=True,
        workers=1,
        server_cmd=["unused"],
        output_path=tmp_path / "evaluation.json",
        environment="tactical-v2",
        evidence_dir=tmp_path / "evidence",
        capture_trace=False,
        start_profile="standard-3v3",
        client_factory=factory,
    )

    assert result["schedule"]["start_profile"] == "standard-3v3"
    assert [(call["start_profile"], call["reference_seat"]) for call in clients[0].resets] == [
        ("standard-3v3", 0),
        ("standard-3v3", 1),
    ]


def test_clone_stage_reopens_physical_checkpoint_and_sidecars(tmp_path: Path) -> None:
    """Deleting a checkpoint after publication must make restart validation fail."""

    module = _subject()
    root = tmp_path / "runs"
    _write_clone_runs(root, module)
    hashes = module.current_definition_hashes()
    assert module._validate_clone_stage(root, hashes)["clone_runs"] == 3

    (root / "seed-211" / "checkpoints" / "step_000000000.zip").unlink()

    with pytest.raises(ValueError, match="checkpoint"):
        module._validate_clone_stage(root, hashes)


def test_gate_stage_rejects_deleted_map_evaluation(tmp_path: Path) -> None:
    """A gate summary cannot substitute for any of its 300 physical evaluation manifests."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    root = tmp_path / "gate"
    result = module.evaluate_clone_gate(
        runs,
        output_dir=root,
        evaluator=ControlledEvaluator({211: 60, 223: 90, 227: 90}),
        server_cmd=["fake-server"],
    )
    assert result["passed"] is True
    hashes = module.current_definition_hashes()

    (root / "seed-211" / "map-16000000" / "evaluation.json").unlink()

    with pytest.raises(ValueError, match="evaluation"):
        module._validate_gate_stage(root, hashes)


def test_gate_stage_recomputes_thresholds_instead_of_trusting_summary(tmp_path: Path) -> None:
    """Tampering a passing aggregate must be detected from immutable map evaluations."""

    module = _subject()
    runs = _write_clone_runs(tmp_path / "runs", module)
    root = tmp_path / "gate"
    result = module.evaluate_clone_gate(
        runs,
        output_dir=root,
        evaluator=ControlledEvaluator({211: 60, 223: 90, 227: 90}),
        server_cmd=["fake-server"],
    )
    assert result["passed"] is True
    gate_path = root / "gate.json"
    gate = _json(gate_path)
    gate["per_seed_wins"] = {"211": 200, "223": 200, "227": 200}
    gate["pooled_win_rate"] = 1.0
    for clone in gate["clones"]:
        clone["wins"] = 200
    gate_path.write_text(json.dumps(gate), encoding="utf-8")

    with pytest.raises(ValueError, match="recomputed"):
        module._validate_gate_stage(root, module.current_definition_hashes())


def test_clone_training_recovers_interruption_before_provenance_publication(
    tmp_path: Path,
) -> None:
    """A fully trained pending clone must be completed on retry without retraining it."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    dataset_manifest = tmp_path / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir()
    dataset_manifest.write_text('{"state":"complete"}', encoding="utf-8")
    dataset = SimpleNamespace(root=dataset_manifest.parent)
    interrupted = False
    calls: list[int] = []

    def trainer(**kwargs):
        nonlocal interrupted
        seed = kwargs["config"].model_seed
        calls.append(seed)
        _write_clone_artifact(
            kwargs["run_dir"],
            module,
            seed,
            dataset_manifest=dataset_manifest,
        )
        if seed == 211 and not interrupted:
            interrupted = True
            raise RuntimeError("interrupt after trainer publication")
        return SimpleNamespace(run_dir=kwargs["run_dir"])

    common = {
        "dataset": dataset,
        "scenario": scenario,
        "env": object(),
        "contract": SimpleNamespace(contract_hash="c" * 64, encoding_hash="e" * 64),
        "spaces_info": {"channels": 1},
        "output_root": tmp_path / "clones",
        "panel": panel,
        "definition_hashes": module.current_definition_hashes(),
        "trainer": trainer,
    }
    with pytest.raises(RuntimeError, match="interrupt"):
        module.train_clone_runs(**common)

    runs = module.train_clone_runs(**common)

    assert [run.name for run in runs] == ["seed-211", "seed-223", "seed-227"]
    assert calls.count(211) == 1
    assert not list((tmp_path / "clones").glob(".seed-*.staging"))
    assert all((run / "panel-provenance.json").is_file() for run in runs)


def test_runtime_scenario_snapshot_is_immune_to_definition_path_mutation(
    tmp_path: Path,
) -> None:
    """Runtime launch bytes and provenance must come from the validated immutable snapshot."""

    module = _subject()
    panel_path = tmp_path / "panel.json"
    banks_path = tmp_path / "seed-banks.json"
    scenario_path = tmp_path / "scenario.json"
    panel_path.write_bytes(module.PANEL_PATH.read_bytes())
    banks_path.write_bytes(module.SEED_BANKS_PATH.read_bytes())
    scenario_path.write_bytes(module.SCENARIO_PATH.read_bytes())
    panel, _banks, scenario = module.validate_definitions(
        panel_path=panel_path,
        seed_banks_path=banks_path,
        scenario_path=scenario_path,
    )
    hashes = module.current_definition_hashes(
        panel_path=panel_path,
        seed_banks_path=banks_path,
        scenario_path=scenario_path,
    )
    changed = _json(scenario_path)
    changed["reward"]["points_weight"] = 0.25
    scenario_path.write_text(json.dumps(changed), encoding="utf-8")

    assert callable(getattr(module, "_materialize_runtime_scenario", None))
    snapshot = module._materialize_runtime_scenario(
        scenario,
        tmp_path / "stage",
        hashes,
    )

    assert json.loads(snapshot.read_text(encoding="utf-8")) == json.loads(scenario.canonical_json)
    provenance = _json(snapshot.parent / "runtime-scenario-provenance.json")
    assert provenance["definition_hashes"] == hashes
    assert provenance["canonical_sha256"] == hashlib.sha256(
        scenario.canonical_json.encode("utf-8")
    ).hexdigest()
    assert _json(snapshot)["reward"]["points_weight"] == 0.5

def test_clone_stage_accepts_real_nested_clone_metrics(tmp_path: Path) -> None:
    module = _subject()
    root = tmp_path / "runs"
    _write_clone_runs(root, module)
    for seed in (211, 223, 227):
        metrics_path = root / f"seed-{seed}" / "metrics.json"
        metrics_path.write_text(
            json.dumps(
                {
                    "nll": 1.0,
                    "top1_accuracy": 0.5,
                    "top3_accuracy": 0.75,
                    "top5_accuracy": 0.9,
                    "expected_calibration_error": 0.1,
                    "mean_end_turn_probability": 0.2,
                    "illegal_probability": 0.0,
                    "strata": {
                        "partition/train": {
                            "count": 4,
                            "accuracy": 0.5,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
    assert module._validate_clone_stage(root, module.current_definition_hashes())["clone_runs"] == 3


def test_clone_stage_rejects_malformed_nested_clone_metrics(tmp_path: Path) -> None:
    module = _subject()
    root = tmp_path / "runs"
    _write_clone_runs(root, module)
    metrics_path = root / "seed-211" / "metrics.json"
    metrics_path.write_text(
        json.dumps({"nll": 1.0, "strata": {"partition/train": {"count": "bad"}}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="metrics"):
        module._validate_clone_stage(root, module.current_definition_hashes())


def test_collect_command_uses_immutable_stage_scenario_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    _identity, identity_kwargs = _command_identity_kwargs(tmp_path, module)
    original = module.SCENARIO_PATH.read_bytes()
    captured: list[bytes] = []

    class FakeClient:
        def __init__(self, command, *, environment):
            captured.append(Path(command[command.index("--scenario-file") + 1]).read_bytes())
            self.contract = SimpleNamespace(contract_hash="c" * 64, encoding_hash="e" * 64)
        def close(self) -> None:
            return None

    def fake_collect(spec) -> None:
        client = spec.client_factory(0)
        client.close()

    def fake_atomic(destination, hashes, *, stage_identity, build, validate):
        changed = _json(module.SCENARIO_PATH)
        changed["reward"]["points_weight"] = 0.25
        module.SCENARIO_PATH.write_text(json.dumps(changed), encoding="utf-8")
        staging = tmp_path / "dataset-stage"
        staging.mkdir()
        build(staging)
        return {"state": "completed"}

    import ml_lab.evaluation
    import collect_annihilation_demonstrations
    monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeClient)
    monkeypatch.setattr(collect_annihilation_demonstrations, "collect_partition", fake_collect)
    monkeypatch.setattr(module, "run_atomic_stage", fake_atomic)
    try:
        module._collect_command(**identity_kwargs)
    finally:
        module.SCENARIO_PATH.write_bytes(original)
    assert captured
    assert all(
        json.loads(data.decode("utf-8")) == json.loads(module.validate_definitions()[2].canonical_json)
        for data in captured
    )


def test_train_command_uses_immutable_stage_scenario_after_source_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _subject()
    identity, identity_kwargs = _command_identity_kwargs(tmp_path, module)
    dataset_root = tmp_path / "source-dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text(
        json.dumps({
            "code_revision": identity["commit"],
            "dirty": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATASET_PATH", dataset_root)
    original = module.SCENARIO_PATH.read_bytes()
    captured: list[bytes] = []
    source = _bc_boundary_contract(
        contract_hash="a" * 64, environment_kind="duel",
    )
    target = _bc_boundary_contract(
        contract_hash="b" * 64, environment_kind="tactical",
    )

    class FakeDuelClient:
        def __init__(self, command, *, environment):
            assert environment == "tactical-v2"
            scenario_path = Path(command[command.index("--scenario-file") + 1])
            captured.append(scenario_path.read_bytes())
            self.contract = source
        def close(self) -> None:
            return None

    class FakeEnv:
        def __init__(self, command, **kwargs):
            del command
            captured.append(Path(kwargs["scenario_path"]).read_bytes())
            self.contract = target
            self.spaces_info = {}
        def close(self) -> None:
            return None

    def fake_atomic(destination, hashes, *, stage_identity, build, validate):
        changed = _json(module.SCENARIO_PATH)
        changed["reward"]["points_weight"] = 0.25
        module.SCENARIO_PATH.write_text(json.dumps(changed), encoding="utf-8")
        staging = tmp_path / "clones-stage"
        staging.mkdir()
        build(staging)
        return {"state": "completed"}

    import hexwars_gym
    import ml_lab.evaluation
    import ml_lab.imitation
    monkeypatch.setattr(hexwars_gym, "HexWarsEnv", FakeEnv)
    monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeDuelClient)
    monkeypatch.setattr(ml_lab.imitation, "load_imitation_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "train_clone_runs", lambda **kwargs: [])
    monkeypatch.setattr(module, "run_atomic_stage", fake_atomic)
    try:
        module._train_bc_command(**identity_kwargs)
    finally:
        module.SCENARIO_PATH.write_bytes(original)
    assert captured
    assert all(
        json.loads(data.decode("utf-8")) == json.loads(module.validate_definitions()[2].canonical_json)
        for data in captured
    )


def test_training_matrix_pairs_initialized_and_control_runs_with_isolated_configs(
    tmp_path: Path,
) -> None:
    """Changing online settings or sharing a config inside a seed pair breaks the control."""

    module = _subject()
    panel, _banks, _scenario = module.validate_definitions()
    runs = module.build_training_matrix(
        panel,
        clone_runs_root=tmp_path / "clones",
        workers=4,
        device="cpu",
    )

    assert [(run.model_seed, run.episode_seed_base, run.condition) for run in runs] == [
        (211, 13_000_000, "bc_ppo"),
        (211, 13_000_000, "scratch_ppo"),
        (223, 14_000_000, "bc_ppo"),
        (223, 14_000_000, "scratch_ppo"),
        (227, 15_000_000, "bc_ppo"),
        (227, 15_000_000, "scratch_ppo"),
    ]
    assert len({id(run.config) for run in runs}) == 6
    assert all(run.scenario_sha256 == runs[0].scenario_sha256 for run in runs)

    for initialized, scratch in zip(runs[::2], runs[1::2], strict=True):
        initialized_config = initialized.config.to_dict()
        scratch_config = scratch.config.to_dict()
        assert initialized_config.pop("actor_init_source") == str(
            (tmp_path / "clones" / f"seed-{initialized.model_seed}").resolve()
        )
        assert scratch_config.pop("actor_init_source") is None
        assert initialized_config.pop("run_name") == f"bc-ppo-seed-{initialized.model_seed}"
        assert scratch_config.pop("run_name") == f"scratch-ppo-seed-{scratch.model_seed}"
        assert initialized_config == scratch_config
        assert initialized_config == {
            "backend": "sb3",
            "algorithm": "maskable_ppo",
            "policy": "HexCNN",
            "seed": initialized.model_seed,
            "total_timesteps": 51_200,
            "checkpoint_interval": 12_800,
            "workers": 4,
            "device": "cpu",
            "learner_seat": "alternating",
            "opponent": {"kind": "scripted", "name": "random"},
            "trackers": [{"kind": "local"}],
            "resume_source": None,
            "algorithm_options": {
                "learning_rate": 0.0003,
                "n_epochs": 10,
                "target_kl": 0.02,
            },
            "episode_seed_base": initialized.episode_seed_base,
            "timestep_mode": "absolute",
            "allow_unsafe_legacy_resume": False,
            "environment": "tactical-v2",
        }


@pytest.mark.parametrize(
    ("steps", "message"),
    [
        ([100, 100, 300], "duplicated"),
        ([100, 300, 200], "decreasing"),
        ([100, 250, 300], "rollout boundary"),
    ],
)
def test_checkpoint_validation_rejects_duplicate_decreasing_and_unaligned_steps(
    tmp_path: Path, steps: list[int], message: str,
) -> None:
    """Accepting a malformed publication sequence could relabel a non-comparable budget."""

    module = _subject()
    checkpoints = [
        module.CheckpointIdentity(
            actual_step=step,
            path=tmp_path / f"step_{index:09d}.zip",
        )
        for index, step in enumerate(steps)
    ]

    with pytest.raises(ValueError, match=message):
        module.validate_rollout_checkpoints(checkpoints, rollout_size=100)


def test_checkpoint_budgets_use_first_completed_rollout_at_or_after_nominal(
    tmp_path: Path,
) -> None:
    """Selecting the preceding rollout or inventing the nominal step changes model identity."""

    module = _subject()
    checkpoints = [
        module.CheckpointIdentity(8_192, tmp_path / "step_000008192.zip"),
        module.CheckpointIdentity(16_384, tmp_path / "step_000016384.zip"),
        module.CheckpointIdentity(32_768, tmp_path / "step_000032768.zip"),
        module.CheckpointIdentity(57_344, tmp_path / "step_000057344.zip"),
    ]

    selected = module.resolve_checkpoint_budgets(
        checkpoints,
        nominal_steps=(12_800, 25_600, 51_200),
        rollout_size=8_192,
    )

    assert [(item.nominal_step, item.actual_step) for item in selected] == [
        (12_800, 16_384),
        (25_600, 32_768),
        (51_200, 57_344),
    ]
    with pytest.raises(RuntimeError, match="no completed rollout reaches 60"):
        module.first_checkpoint_at_or_after(checkpoints, 60_000)


def _selection_table() -> list[dict[str, Any]]:
    return [
        {
            "condition": "bc_ppo",
            "model_seed": seed,
            "nominal_step": nominal,
            "actual_step": nominal + seed,
            "standard": standard,
            "conversion": conversion,
        }
        for nominal, rows in [
            (
                12_800,
                [
                    (211, ["win", "loss"], ["loss"]),
                    (223, ["win", "loss"], ["loss"]),
                    (227, ["win", "loss"], ["loss"]),
                ],
            ),
            (
                25_600,
                [
                    (211, ["win", "win"], ["loss"]),
                    (223, ["win", "loss"], ["win"]),
                    (227, ["win", "loss"], ["loss"]),
                ],
            ),
            (
                51_200,
                [
                    (211, ["win", "win"], ["loss"]),
                    (223, ["win", "loss"], ["win"]),
                    (227, ["win", "loss"], ["loss"]),
                ],
            ),
        ]
        for seed, standard, conversion in rows
    ]


def test_selection_uses_one_global_budget_and_seed_specific_actual_steps() -> None:
    """Per-seed best checkpoints would invalidate the preregistered pooled comparison."""

    module = _subject()
    selected = module.select_global_budget(_selection_table())

    assert selected.nominal_step == 25_600
    assert selected.actual_steps == {
        211: 25_811,
        223: 25_823,
        227: 25_827,
    }


def test_selection_tiebreak_order_is_pooled_worst_conversion_draw_then_earlier() -> None:
    """Reordering any tie-break can choose a different budget after development data."""

    module = _subject()

    def tied() -> list[dict[str, Any]]:
        rows = _selection_table()
        for row in rows:
            row["standard"] = ["win", "loss"]
            row["conversion"] = ["loss"]
        return rows

    table = tied()
    table[6]["standard"] = ["win", "win"]
    assert module.select_global_budget(table).nominal_step == 51_200

    table = tied()
    table[0]["standard"] = ["win", "win"]
    table[2]["standard"] = ["loss", "loss"]
    for row in table[:3]:
        row["conversion"] = ["win"]

    assert module.select_global_budget(table).nominal_step == 25_600

    table = tied()
    table[3]["conversion"] = ["win"]
    assert module.select_global_budget(table).nominal_step == 25_600

    table = tied()
    table[0]["standard"] = ["win", "draw"]
    assert module.select_global_budget(table).nominal_step == 25_600

    assert module.select_global_budget(tied()).nominal_step == 12_800


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "decreasing"])
def test_selection_rejects_incomplete_duplicate_or_decreasing_candidate_rows(
    mutation: str,
) -> None:
    """A partial or inconsistent development table must never publish a selection."""

    module = _subject()
    table = _selection_table()
    if mutation == "missing":
        table.pop()
    elif mutation == "duplicate":
        table.append(dict(table[-1]))
    else:
        table[-1]["actual_step"] = 1

    with pytest.raises(ValueError, match=mutation):
        module.select_global_budget(table)


def test_development_schedule_reuses_all_100_maps_and_both_seats_for_every_candidate() -> None:
    """Dropping a condition, seed, budget, map, or seat destroys paired comparability."""

    module = _subject()
    candidates = [
        module.DevelopmentCandidate("pure_bc", seed, 0, 0, f"bc-{seed}")
        for seed in (211, 223, 227)
    ] + [
        module.DevelopmentCandidate(
            condition,
            seed,
            nominal,
            nominal + seed,
            f"{condition}-{seed}-{nominal}",
        )
        for condition in ("bc_ppo", "scratch_ppo")
        for seed in (211, 223, 227)
        for nominal in (12_800, 25_600, 51_200)
    ]

    schedule = module.build_development_schedule(candidates)

    assert len(candidates) == 21
    assert len(schedule) == 4_200
    for candidate in candidates:
        games = [
            game
            for game in schedule
            if (
                game.condition,
                game.model_seed,
                game.nominal_step,
                game.actual_step,
            ) == (
                candidate.condition,
                candidate.model_seed,
                candidate.nominal_step,
                candidate.actual_step,
            )
        ]
        assert [(game.map_seed, game.candidate_seat) for game in games] == [
            (map_seed, seat)
            for map_seed in range(16_000_000, 16_000_100)
            for seat in (0, 1)
        ]
        assert all(
            game.profile == "standard-3v3" and game.opponent == "random"
            for game in games
        )


def test_train_ppo_command_refuses_to_start_before_clone_gate_passes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Training before the clone gate passes would spend compute outside the protocol."""

    module = _subject()
    _identity, identity_kwargs = _command_identity_kwargs(tmp_path, module)
    monkeypatch.setattr(module, "_validate_clone_stage", lambda *_args: {})
    monkeypatch.setattr(
        module,
        "_validate_gate_stage",
        lambda *_args: (_ for _ in ()).throw(ValueError("clone gate did not pass")),
    )
    monkeypatch.setattr(
        module,
        "build_training_matrix",
        lambda *_args, **_kwargs: pytest.fail("training matrix must not be built"),
    )

    with pytest.raises(ValueError, match="clone gate did not pass"):
        module._train_ppo_command(**identity_kwargs)


def test_selection_publication_is_atomic_restart_safe_and_hashes_every_input(
    tmp_path: Path,
) -> None:
    """A rewritten or unhashed development table could silently change the chosen budget."""

    module = _subject()
    table = _selection_table()
    checkpoint_hashes: dict[str, str] = {}
    for row in table:
        row["standard"] = (row["standard"] * 100)[:200]
        key = f"{row['condition']}/seed-{row['model_seed']}/nominal-{row['nominal_step']}"
        source_run = (
            tmp_path
            / "runs"
            / row["condition"]
            / f"seed-{row['model_seed']}"
        )
        checkpoint = (
            source_run
            / "checkpoints"
            / f"step_{row['actual_step']:09d}.zip"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(key.encode("utf-8"))
        digest = _sha256(checkpoint)
        row["checkpoint_sha256"] = digest
        row["checkpoint_path"] = str(checkpoint.resolve())
        row["source_run"] = str(source_run.resolve())
        row["algorithm"] = "maskable_ppo"
        row["controller"] = _production_controller_identity(
            kind="snapshot",
            path=checkpoint,
            algorithm="maskable_ppo",
            step=row["actual_step"],
        )
        checkpoint_hashes[key] = digest
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps({"schema_version": 1, "candidates": table}),
        encoding="utf-8",
    )
    output = tmp_path / "selection.json"
    hashes = {
        "panel_sha256": "a" * 64,
        "scenario_sha256": "b" * 64,
        "seed_banks_sha256": "c" * 64,
    }

    first = module.publish_selection(
        development_path=development,
        output_path=output,
        definition_hashes=hashes,
    )
    before = output.stat().st_mtime_ns
    second = module.publish_selection(
        development_path=development,
        output_path=output,
        definition_hashes=hashes,
    )

    assert first["selection"]["nominal_step"] == 25_600
    assert first["input_hashes"] == {
        **hashes,
        "development_sha256": _sha256(development),
        "candidate_checkpoints_sha256": checkpoint_hashes,
    }
    assert second == {**first, "reused": True}
    assert output.stat().st_mtime_ns == before
    assert not list(tmp_path.glob(".selection.json.*.tmp"))


def test_train_ppo_runs_uses_real_training_boundary_with_fresh_run_configs(
    tmp_path: Path,
) -> None:
    """Bypassing run_training or reusing one config can share mutable optimizer state."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    matrix = module.build_training_matrix(
        panel,
        clone_runs_root=tmp_path / "clones",
        workers=4,
        device="cpu",
    )
    calls: list[dict[str, Any]] = []

    def trainer(config, **kwargs):
        calls.append({"config": config, **kwargs})
        destination = Path(kwargs["runs_root"]) / config.run_name
        destination.mkdir(parents=True)
        scenario.write(destination / "scenario.json")
        (destination / "run.json").write_text(
            json.dumps({"state": "completed", "config": config.to_dict()}),
            encoding="utf-8",
        )
        return destination

    paths = module.train_ppo_runs(
        matrix,
        runs_root=tmp_path / "ppo",
        scenario=scenario,
        server_cmd=["fake-server"],
        trainer=trainer,
    )

    assert len(paths) == len(calls) == 6
    assert len({id(call["config"]) for call in calls}) == 6
    assert len(set(paths)) == 6
    assert all(call["scenario"] is scenario for call in calls)
    assert all(call["server_cmd"] == ["fake-server"] for call in calls)
    assert all(call["config"].resume_source is None for call in calls)


def test_development_evaluation_uses_real_boundary_and_records_every_game_identity(
    tmp_path: Path,
) -> None:
    """Aggregating before recording seat, map, checkpoint, trace, and replay loses evidence."""

    module = _subject()
    source_run = tmp_path / "run"
    checkpoint = source_run / "checkpoints" / "step_000016384.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    controller_spec = json.dumps(
        {
            "kind": "snapshot",
            "path": str(checkpoint.resolve()),
            "source_run": str(source_run.resolve()),
            "algorithm": "maskable_ppo",
            "step": 16_384,
        },
        sort_keys=True,
    )
    controller_identity = _production_controller_identity(
        kind="snapshot",
        path=checkpoint,
        algorithm="maskable_ppo",
        step=16_384,
    )
    opponent_identity = _production_controller_identity(
        kind="scripted", name="random"
    )
    candidate = module.DevelopmentCandidate(
        "bc_ppo",
        211,
        12_800,
        16_384,
        controller_spec,
        _sha256(checkpoint),
        str(checkpoint.resolve()),
        str(source_run.resolve()),
        "maskable_ppo",
    )
    calls: list[dict[str, Any]] = []

    def evaluator(p0, p1, **kwargs):
        calls.append({"p0": p0, "p1": p1, **kwargs})
        map_seed = kwargs["seed_start"]
        evidence = Path(kwargs["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        matches = []
        for seat in (0, 1):
            trace = evidence / f"{map_seed}-{seat}.json"
            replay = evidence / f"{map_seed}-{seat}.replay"
            trace.write_text("{}", encoding="utf-8")
            replay.write_bytes(b"replay")
            matches.append(
                {
                    "seed": map_seed,
                    "candidate_seat": seat,
                    "outcome": "win" if seat == 0 else "loss",
                    "trace_path": str(trace),
                    "replay_path": str(replay),
                }
            )
        result = {
            "schema_version": 1,
            "generated_at": "2026-08-01T00:00:00+00:00",
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "rates": {"win": 0.5, "loss": 0.5, "draw": 0.0},
            "confidence_intervals": {
                "win": {"low": 0.0, "high": 1.0, "confidence": 0.95},
                "loss": {"low": 0.0, "high": 1.0, "confidence": 0.95},
                "draw": {"low": 0.0, "high": 1.0, "confidence": 0.95},
            },
            "seat_results": {
                "candidate_as_p0": {"wins": 1, "losses": 0, "draws": 0},
                "candidate_as_p1": {"wins": 0, "losses": 1, "draws": 0},
            },
            "evidence": {
                "draw_traces": 0,
                "control_traces": 2,
                "draw_categories": {},
            },

            "candidate": dict(controller_identity),
            "opponent": dict(opponent_identity),
            "seed_start": map_seed,
            "seeds": [map_seed],
            "reciprocal": True,
            "games": 2,
            "schedule": {
                "start_profile": kwargs["start_profile"],
                "reference_seat_policy": "candidate-seat",
            },
            "matches": matches,
        }
        Path(kwargs["output_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kwargs["output_path"]).write_text(json.dumps(result), encoding="utf-8")
        return result

    result = module.evaluate_development_candidates(
        [candidate],
        output_root=tmp_path / "development",
        evaluator=evaluator,
        server_cmd=["fake-server"],
        conversion_profiles=module._CONVERSION_PROFILES,
    )

    assert len(calls) == 700
    assert all(
        call["p0"] == controller_spec
        and call["p1"] == "random"
        and call["games"] == 1
        and call["both_seats"] is True
        and call["capture_trace"] is True
        for call in calls
    )
    assert sum(call["start_profile"] == "standard-3v3" for call in calls) == 100
    assert {
        call["start_profile"] for call in calls if call["start_profile"] != "standard-3v3"
    } == set(module._CONVERSION_PROFILES)
    rows = result["candidates"][0]["matches"]
    assert len(rows) == 200
    assert [(row["map_seed"], row["candidate_seat"]) for row in rows] == [
        (seed, seat)
        for seed in range(16_000_000, 16_000_100)
        for seat in (0, 1)
    ]
    assert all(row["actual_step"] == 16_384 for row in rows)
    assert len(result["candidates"][0]["conversion"]) == 1_200
    assert result["candidates"][0]["conversion_schedule"] == {
        "profiles": list(module._CONVERSION_PROFILES),
        "maps_per_profile": 100,
        "both_seats": True,
        "seed_start": 16_000_000,
    }
    assert all(
        row["condition"] == "bc_ppo"
        and row["model_seed"] == 211
        and row["nominal_step"] == 12_800
        and row["checkpoint_sha256"] == candidate.checkpoint_sha256
        and row["controller"] == controller_identity
        and row["opponent"] == opponent_identity
        and row["profile"] == "standard-3v3"
        for row in rows
    )
    assert all(
        (tmp_path / "development" / row[field]).is_file()
        for row in rows
        for field in ("trace_path", "replay_path")
    )


def test_ppo_stage_resolves_actual_budgets_and_rejects_wrong_actor_source(
    tmp_path: Path,
) -> None:
    """A shared or wrong clone source invalidates initialized-vs-scratch attribution."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    _write_clone_runs(tmp_path / "clones", module)
    matrix = module.build_training_matrix(
        panel,
        clone_runs_root=tmp_path / "clones",
        workers=4,
        device="cpu",
    )
    root = tmp_path / "ppo"
    for run in matrix:
        run_dir = root / run.config.run_name
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        scenario.write(run_dir / "scenario.json")
        (run_dir / "run.json").write_text(
            json.dumps({"state": "completed", "config": run.config.to_dict()}),
            encoding="utf-8",
        )
        for step in (2_048, 14_336, 26_624, 38_912, 51_200):
            (checkpoints / f"step_{step:09d}.zip").write_bytes(b"checkpoint")
        if run.condition == "bc_ppo":
            _write_actor_initialization(run_dir, run)

    budgets = module._ppo_budget_map(root, matrix, scenario)
    assert {
        name: [(item.nominal_step, item.actual_step) for item in items]
        for name, items in budgets.items()
    } == {
        run.config.run_name: [
            (12_800, 14_336),
            (25_600, 26_624),
            (51_200, 51_200),
        ]
        for run in matrix
    }

    initialized = root / "bc-ppo-seed-211" / "initialization.json"
    changed = _json(initialized)
    changed["source_actor_fixtures_sha256"] = "f" * 64
    initialized.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="initialization provenance"):
        module._ppo_budget_map(root, matrix, scenario)


def test_ppo_budget_mapping_uses_the_real_2048_step_rollout_boundary(
    tmp_path: Path,
) -> None:
    """Using the preregistration's stale 8192-step rollout selects the wrong snapshots."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    run = next(
        item
        for item in module.build_training_matrix(
            panel,
            clone_runs_root=tmp_path / "clones",
            workers=4,
            device="cpu",
        )
        if item.condition == "scratch_ppo" and item.model_seed == 211
    )
    root = tmp_path / "ppo"
    run_dir = root / run.config.run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    scenario.write(run_dir / "scenario.json")
    (run_dir / "run.json").write_text(
        json.dumps({"state": "completed", "config": run.config.to_dict()}),
        encoding="utf-8",
    )
    for step in (2_048, 14_336, 26_624, 38_912, 51_200):
        (checkpoints / f"step_{step:09d}.zip").write_bytes(b"checkpoint")

    budgets = module._ppo_budget_map(root, [run], scenario)

    assert [
        (item.nominal_step, item.actual_step)
        for item in budgets[run.config.run_name]
    ] == [(12_800, 14_336), (25_600, 26_624), (51_200, 51_200)]


def test_development_evaluation_reopens_physical_map_evidence(
    tmp_path: Path,
) -> None:
    """Trusting the evaluator return lets a different on-disk controller enter selection."""

    module = _subject()
    candidate = module.DevelopmentCandidate(
        "bc_ppo", 211, 12_800, 14_336, "snapshot-controller", "a" * 64
    )

    def evaluator(_p0, _p1, **kwargs):
        map_seed = kwargs["seed_start"]
        evidence = Path(kwargs["evidence_dir"])
        evidence.mkdir(parents=True, exist_ok=True)
        matches = []
        for seat in (0, 1):
            trace = evidence / f"{map_seed}-{seat}.json"
            replay = evidence / f"{map_seed}-{seat}.replay"
            trace.write_text("{}", encoding="utf-8")
            replay.write_bytes(b"replay")
            matches.append(
                {
                    "seed": map_seed,
                    "candidate_seat": seat,
                    "outcome": "win",
                    "trace_path": str(trace),
                    "replay_path": str(replay),
                }
            )
        returned = {
            "candidate": {
                "kind": "snapshot",
                "path": "expected.zip",
                "source_run": "expected-run",
                "algorithm": "maskable_ppo",
                "step": 14_336,
                "inference_mode": "deterministic",
            },
            "opponent": {"kind": "scripted", "name": "random"},
            "seed_start": map_seed,
            "seeds": [map_seed],
            "reciprocal": True,
            "games": 2,
            "schedule": {
                "start_profile": kwargs["start_profile"],
                "reference_seat_policy": "candidate-seat",
            },
            "matches": matches,
        }
        persisted = dict(returned)
        persisted["candidate"] = {**returned["candidate"], "step": 26_624}
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(persisted), encoding="utf-8")
        return returned

    with pytest.raises(ValueError, match="physical development evaluation"):
        module.evaluate_development_candidates(
            [candidate],
            output_root=tmp_path / "development",
            evaluator=evaluator,
            server_cmd=["fake-server"],
        )


def test_selection_recomputes_physical_checkpoint_digests(tmp_path: Path) -> None:
    """A copied 64-character digest must not bless checkpoint bytes changed after evaluation."""

    module = _subject()
    table = _selection_table()
    for row in table:
        row["standard"] = (row["standard"] * 100)[:200]
        checkpoint = (
            tmp_path
            / "checkpoints"
            / f"{row['model_seed']}-{row['nominal_step']}.zip"
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"evaluated")
        row["checkpoint_path"] = str(checkpoint.resolve())
        row["source_run"] = str(tmp_path.resolve())
        row["algorithm"] = "maskable_ppo"
        row["checkpoint_sha256"] = _sha256(checkpoint)
    Path(table[0]["checkpoint_path"]).write_bytes(b"tampered")
    development = tmp_path / "development.json"
    development.write_text(
        json.dumps({"schema_version": 1, "candidates": table}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="checkpoint digest"):
        module.publish_selection(
            development_path=development,
            output_path=tmp_path / "selection.json",
            definition_hashes={"panel_sha256": "a" * 64},
        )


def test_train_ppo_runs_retries_incomplete_deterministic_pending_run(
    tmp_path: Path,
) -> None:
    """An interrupted first attempt must not require resume flags or manual cleanup."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    run = module.build_training_matrix(
        panel,
        clone_runs_root=tmp_path / "clones",
        workers=4,
        device="cpu",
    )[0]
    root = tmp_path / "ppo"
    trainer_roots: list[Path] = []
    attempts = 0

    def trainer(config, **kwargs):
        nonlocal attempts
        attempts += 1
        trainer_root = Path(kwargs["runs_root"])
        trainer_roots.append(trainer_root)
        destination = trainer_root / config.run_name
        if attempts == 1:
            destination.mkdir(parents=True)
            (destination / "partial.marker").write_text("partial", encoding="utf-8")
            raise RuntimeError("interrupted")
        assert not (destination / "partial.marker").exists()
        destination.mkdir(parents=True, exist_ok=True)
        scenario.write(destination / "scenario.json")
        (destination / "run.json").write_text(
            json.dumps({"state": "completed", "config": config.to_dict()}),
            encoding="utf-8",
        )
        return destination

    with pytest.raises(RuntimeError, match="interrupted"):
        module.train_ppo_runs(
            [run],
            runs_root=root,
            scenario=scenario,
            server_cmd=["fake-server"],
            trainer=trainer,
        )

    outputs = module.train_ppo_runs(
        [run],
        runs_root=root,
        scenario=scenario,
        server_cmd=["fake-server"],
        trainer=trainer,
    )

    assert outputs == [root / run.config.run_name]
    assert trainer_roots == [root / ".pending", root / ".pending"]
    assert not (root / ".pending" / run.config.run_name).exists()


def test_ppo_stage_recomputes_complete_actor_initializer_provenance(
    tmp_path: Path,
) -> None:
    """A seed-matched source_run string alone cannot prove which actor bytes initialized PPO."""

    module = _subject()
    panel, _banks, scenario = module.validate_definitions()
    _write_clone_runs(tmp_path / "clones", module)
    run = next(
        item
        for item in module.build_training_matrix(
            panel,
            clone_runs_root=tmp_path / "clones",
            workers=4,
            device="cpu",
        )
        if item.condition == "bc_ppo" and item.model_seed == 211
    )
    root = tmp_path / "ppo"
    run_dir = root / run.config.run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True)
    scenario.write(run_dir / "scenario.json")
    (run_dir / "run.json").write_text(
        json.dumps({"state": "completed", "config": run.config.to_dict()}),
        encoding="utf-8",
    )
    for step in (14_336, 26_624, 51_200):
        (checkpoints / f"step_{step:09d}.zip").write_bytes(b"checkpoint")
    _write_actor_initialization(run_dir, run)
    provenance_path = run_dir / "initialization.json"
    provenance = _json(provenance_path)
    provenance["source_bc_sha256"] = "f" * 64
    provenance_path.write_text(json.dumps(provenance), encoding="utf-8")

    with pytest.raises(ValueError, match="initialization provenance"):
        module._ppo_budget_map(root, [run], scenario)


def _actual_development_candidate_fixture(
    tmp_path: Path, module
) -> tuple[Path, dict[str, Any]]:
    root = tmp_path / "development"
    source_run = tmp_path / "source-run"
    checkpoint = source_run / "checkpoints" / "step_000014336.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    controller_spec = json.dumps(
        {
            "kind": "snapshot",
            "path": str(checkpoint.resolve()),
            "source_run": str(source_run.resolve()),
            "algorithm": "maskable_ppo",
            "step": 14_336,
        },
        sort_keys=True,
    )
    controller_identity = _production_controller_identity(
        kind="snapshot",
        path=checkpoint,
        algorithm="maskable_ppo",
        step=14_336,
    )
    opponent_identity = _production_controller_identity(
        kind="scripted", name="random"
    )
    candidate = module.DevelopmentCandidate(
        "bc_ppo",
        211,
        12_800,
        14_336,
        controller_spec,
        _sha256(checkpoint),
        str(checkpoint.resolve()),
        str(source_run.resolve()),
        "maskable_ppo",
    )

    def evaluator(_p0, _p1, **kwargs):
        map_seed = kwargs["seed_start"]
        evidence_root = Path(kwargs["evidence_dir"])
        matches = []
        for seat, outcome in ((0, "win"), (1, "loss")):
            trace = evidence_root / "traces" / f"{map_seed}-{seat}.json"
            replay = evidence_root / "replays" / f"{map_seed}-{seat}.replay"
            trace.parent.mkdir(parents=True, exist_ok=True)
            replay.parent.mkdir(parents=True, exist_ok=True)
            trace.write_text("{}", encoding="utf-8")
            replay.write_bytes(b"replay")
            matches.append(
                {
                    "seed": map_seed,
                    "candidate_seat": seat,
                    "winner": seat,
                    "outcome": outcome,
                    "terminated": True,
                    "truncated": False,
                    "summary": {},
                    "classification": None,
                    "trace_path": str(trace),
                    "replay_path": str(replay),
                }
            )
        manifest = {
            "schema_version": 1,
            "generated_at": "2026-08-01T00:00:00+00:00",
            "schedule": {
                "start_profile": kwargs["start_profile"],
                "reference_seat_policy": "candidate-seat",
            },
            "candidate": dict(controller_identity),
            "opponent": dict(opponent_identity),
            "seed_start": map_seed,
            "seeds": [map_seed],
            "reciprocal": True,
            "games": 2,
            "wins": 1,
            "losses": 1,
            "draws": 0,
            "rates": {"win": 0.5, "loss": 0.5, "draw": 0.0},
            "confidence_intervals": {
                "win": {"low": 0.0, "high": 1.0, "confidence": 0.95},
                "loss": {"low": 0.0, "high": 1.0, "confidence": 0.95},
                "draw": {"low": 0.0, "high": 1.0, "confidence": 0.95},
            },
            "seat_results": {
                "candidate_as_p0": {"wins": 1, "losses": 0, "draws": 0},
                "candidate_as_p1": {"wins": 0, "losses": 1, "draws": 0},
            },
            "matches": matches,
            "evidence": {
                "draw_traces": 0,
                "control_traces": 2,
                "draw_categories": {},
            },
        }
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest

    result = module.evaluate_development_candidates(
        [candidate],
        output_root=root,
        evaluator=evaluator,
        server_cmd=["fake-server"],
    )
    return root, result["candidates"][0]


def test_development_candidate_audit_reopens_all_100_actual_manifests(
    tmp_path: Path,
) -> None:
    """Auditing only copied aggregate rows cannot prove all physical maps were evaluated."""

    module = _subject()
    root, aggregate = _actual_development_candidate_fixture(tmp_path, module)

    matches = module._validate_development_candidate_evidence(root, aggregate)

    assert len(matches) == 200
    assert [(row["map_seed"], row["candidate_seat"]) for row in matches] == [
        (map_seed, seat)
        for map_seed in range(16_000_000, 16_000_100)
        for seat in (0, 1)
    ]


@pytest.mark.parametrize(
    "mutation",
    ["manifest", "controller", "opponent", "schedule", "aggregate"],
)
def test_development_candidate_audit_rejects_physical_or_aggregate_tampering(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Any changed manifest identity, summary, or copied aggregate must block selection."""

    module = _subject()
    root, aggregate = _actual_development_candidate_fixture(tmp_path, module)
    first_manifest = (
        root
        / "bc_ppo"
        / "seed-211"
        / "nominal-12800"
        / "map-16000000"
        / "evaluation.json"
    )
    physical = _json(first_manifest)
    if mutation == "manifest":
        physical["wins"] = 2
    elif mutation == "controller":
        physical["candidate"]["path"] = str((tmp_path / "other.zip").resolve())
    elif mutation == "opponent":
        physical["opponent"]["name"] = "greedy"
    elif mutation == "schedule":
        physical["schedule"]["start_profile"] = "conversion-3v1-near"
    else:
        aggregate["standard"][0] = "draw"
    if mutation != "aggregate":
        first_manifest.write_text(json.dumps(physical), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="physical development|development aggregate|development candidate schedule",
    ):
        module._validate_development_candidate_evidence(root, aggregate)


def _final_panel_fixture(tmp_path: Path, *, selection: bool = True):
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    (panel_dir / "panel.json").write_text(json.dumps({"schema_version": 1, "model_seeds": [211, 223, 227]}), encoding="utf-8")
    (panel_dir / "scenario.json").write_text(json.dumps({"environment": "tactical-v2", "contract_hash": "c" * 64, "encoding_hash": "e" * 64}), encoding="utf-8")
    (panel_dir / "seed-banks.json").write_text(json.dumps({"schema_version": 1, "banks": {name: {"start": start, "stop": stop, "assigned": assigned} for name, (start, stop, assigned) in EXPECTED_BANKS.items()}}), encoding="utf-8")
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "games.jsonl").write_text('{"seed":11000000}\\n', encoding="utf-8")
    (dataset_dir / "manifest.json").write_text(json.dumps({"schema_version": 1, "state": "completed"}), encoding="utf-8")
    candidates = []
    for seed in (211, 223, 227):
        for condition, prefix in (("bc_ppo", "bc-ppo"), ("scratch_ppo", "scratch-ppo")):
            run = panel_dir / "ppo-runs" / f"{prefix}-seed-{seed}"
            checkpoint = run / "checkpoints" / "step_000025600.zip"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(f"{condition}-{seed}".encode())
            (run / "run.json").write_text(json.dumps({
                "schema_version": 1, "state": "completed", "model_seed": seed,
                "condition": condition, "timesteps": 25_600,
            }), encoding="utf-8")
            candidates.append({
                "condition": condition, "model_seed": seed, "nominal_step": 25_600,
                "actual_step": 25_600, "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_sha256": _sha256(checkpoint), "source_run": str(run.resolve()),
                "standard": ["win", "loss"],
                "conversion": ["win", "draw"] if condition == "bc_ppo" else [],
            })
            for nominal, standard in (
                (12_800, ["loss", "draw"]),
                (51_200, ["win", "win"]),
            ):
                candidates.append({
                    "condition": condition, "model_seed": seed,
                    "nominal_step": nominal, "actual_step": nominal,
                    "standard": standard,
                    "conversion": [],
                })
    development = panel_dir / "development"
    development.mkdir()
    (development / "development.json").write_text(json.dumps({"schema_version": 1, "state": "completed", "candidates": candidates}), encoding="utf-8")
    if selection:
        (panel_dir / "selection.json").write_text(json.dumps({"schema_version": 1, "state": "completed", "selection": {"nominal_step": 25_600, "actual_steps": {"211": 25_600, "223": 25_600, "227": 25_600}}}), encoding="utf-8")
    incumbent = tmp_path / "python" / "panels" / "incumbent"
    incumbent.mkdir(parents=True)
    runs_root = tmp_path / "python" / "runs"
    profiled_standard = {}
    for seed in (101, 113, 127):
        run_name = f"annihilation-conversion-profiled_standard-seed{seed}-tb-v1"
        run = runs_root / run_name
        checkpoint = run / "checkpoints" / "step_000051200.zip"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"incumbent-{seed}".encode())
        (run / "scenario.json").write_text(json.dumps({
            "schema_version": 1, "environment": "tactical-v2",
            "id": "tactical-v2-annihilation-profiled-standard",
            "tactical_v2": {"start_distribution": [
                {"profile_id": "standard-3v3", "basis_points": 10_000},
            ]},
        }), encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "schema_version": 1, "state": "completed",
            "config": {"seed": seed, "algorithm": "maskable_ppo", "environment": "tactical-v2"},
            "contract": {"contract_hash": "c" * 64, "encoding_hash": "e" * 64, "environment": "tactical-v2"},
            "scenario": {"path": "scenario.json", "template_id": "tactical-v2-annihilation-profiled-standard"},
            "latest_checkpoint": "checkpoints/step_000051200.zip",
            "latest_checkpoint_step": 51_200,
        }), encoding="utf-8")
        profiled_standard[str(seed)] = {"standard": {}, "conversion": {}, "training": {
            "run": run_name,
            "checkpoint": "checkpoints/step_000051200.zip",
            "checkpoint_sha256": _sha256(checkpoint),
            "environment_steps": 51_200,
            "wall_clock_seconds": 12.5,
        }}
    (incumbent / "aggregate.json").write_text(json.dumps({
        "schema_version": 1, "models": {"profiled_standard": profiled_standard},
    }), encoding="utf-8")
    clone_gate = panel_dir / "bc-development-gate"
    clone_gate.mkdir()
    (clone_gate / "gate.json").write_text(json.dumps({
        "schema_version": 1,
        "clones": [
            {"model_seed": seed, "wins": 10, "losses": 5, "draws": 5}
            for seed in (211, 223, 227)
        ],
    }), encoding="utf-8")
    for seed in (211, 223, 227):
        clone_run = panel_dir / "bc-clones" / f"seed-{seed}"
        clone_run.mkdir(parents=True)
        (clone_run / "metrics.json").write_text(json.dumps({
            "nll": 0.4, "top1_accuracy": 0.8,
        }), encoding="utf-8")
        (clone_run / "run.json").write_text(json.dumps({
            "state": "completed", "model_seed": seed,
        }), encoding="utf-8")
    return panel_dir, dataset_dir, incumbent


def test_final_bank_cannot_be_assigned_before_selection_is_frozen(tmp_path: Path) -> None:
    """Removing the completed global selection must keep the final bank sealed."""
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path, selection=False)
    with pytest.raises(RuntimeError, match="global checkpoint"):
        module.freeze_final(panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir, revision="37cc8f9", dirty=False)


def test_final_bank_seal_is_complete_immutable_and_single_use(tmp_path: Path) -> None:
    """Omitting any frozen input or allowing resealing makes the held-out bank mutable."""
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    seal = module.freeze_final(panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir, revision="37cc8f9", dirty=False)
    assert seal["revision"] == {"commit": "37cc8f9", "dirty": False}
    assert set(seal["definition_hashes"]) == {"panel.json", "scenario.json", "seed-banks.json"}
    assert set(seal["dataset_hashes"]) == {"games.jsonl", "manifest.json"}
    assert len(seal["training_run_hashes"]) == 6
    assert len(seal["selected_checkpoints"]["initialized"]) == 3
    assert len(seal["selected_checkpoints"]["control"]) == 3
    assert len(seal["incumbent_comparators"]) == 3
    assert seal["seed_banks"]["final"]["assigned"] is True
    before = (panel_dir / "final-seal.json").read_bytes()
    with pytest.raises(RuntimeError, match="already assigned"):
        module.freeze_final(panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir, revision="different", dirty=True)
    assert (panel_dir / "final-seal.json").read_bytes() == before


def test_final_gate_requires_each_seed_and_pooled_thresholds() -> None:
    """Weakening either threshold would turn a preregistered milestone failure into a pass."""
    module = _subject()
    assert module.apply_final_gate(wins={211: 325, 223: 325, 227: 400}, games=500).passed
    assert not module.apply_final_gate(wins={211: 324, 223: 400, 227: 400}, games=500).passed
    assert not module.apply_final_gate(wins={211: 325, 223: 325, 227: 399}, games=500).passed



def _final_evaluator(calls: list[dict[str, Any]], *, mutation: str | None = None):
    def evaluator(candidate, opponent, **kwargs):
        calls.append({"candidate": candidate, "opponent": opponent, **kwargs})
        seed_start = kwargs["seed_start"]
        matches = []
        for map_seed in range(seed_start, seed_start + kwargs["games"]):
            for seat in (0, 1):
                matches.append({
                    "seed": map_seed,
                    "candidate_seat": seat,
                    "outcome": "win" if (map_seed + seat) % 3 else "draw",
                    "summary": {
                        "rounds": 10,
                        "decisions": 20,
                        "action_waste": 2,
                        "peak_material_advantage": 4,
                    },
                    "classification": {"category": "material_lead"} if (map_seed + seat) % 3 == 0 else None,
                })
        if mutation == "partial":
            matches.pop()
        elif mutation == "duplicate":
            matches[-1] = dict(matches[0])
        elif mutation == "outside":
            matches[-1]["seed"] = 17_000_250
        result = {"schema_version": 1, "matches": matches}
        output = Path(kwargs["output_path"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result), encoding="utf-8")
        return result
    return evaluator


def test_final_evaluation_runs_exact_reciprocal_bank_once(tmp_path: Path) -> None:
    """Changing map count, seat policy, opponent, profile, or reuse invalidates the final bank."""
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
                        revision="37cc8f9", dirty=False)
    calls: list[dict[str, Any]] = []

    result = module.evaluate_final(panel_dir, evaluator=_final_evaluator(calls),
                                   server_cmd=["fake-server"])

    assert len(calls) == 9
    assert all(call["games"] == 250 and call["seed_start"] == 17_000_000
               and call["both_seats"] is True and call["opponent"] == "random"
               and call["start_profile"] == "standard-3v3" for call in calls)
    incumbent_specs = [json.loads(call["candidate"]) for call in calls[-3:]]
    assert all(spec["kind"] == "snapshot" for spec in incumbent_specs)
    assert all(Path(spec["path"]).name == "step_000051200.zip" for spec in incumbent_specs)
    assert all(spec["algorithm"] == "maskable_ppo" and spec["step"] == 51_200
               and spec["contract_hash"] == "c" * 64 for spec in incumbent_specs)
    assert len(result["matches"]) == 1_500
    assert len(result["comparison_matches"]) == 3_000
    assert len({(row["model_seed"], row["seed"], row["candidate_seat"])
                for row in result["matches"]}) == 1_500
    with pytest.raises(RuntimeError, match="already completed"):
        module.evaluate_final(panel_dir, evaluator=_final_evaluator([]),
                              server_cmd=["fake-server"])


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "outside"])
def test_final_evaluation_refuses_incomplete_duplicate_or_out_of_bank_results(
    tmp_path: Path, mutation: str
) -> None:
    """Publishing any result other than the exact sealed schedule spends the bank incorrectly."""
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
                        revision="37cc8f9", dirty=False)

    with pytest.raises(ValueError, match="final schedule"):
        module.evaluate_final(panel_dir, evaluator=_final_evaluator([], mutation=mutation),
                              server_cmd=["fake-server"])

    assert not (panel_dir / "final-evaluation.json").exists()


def test_final_evaluation_revalidates_seal_after_games_before_publication(
    tmp_path: Path,
) -> None:
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(
        panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
        revision="37cc8f9", dirty=False,
    )
    calls: list[dict[str, Any]] = []
    base = _final_evaluator(calls)

    def mutate_during_evaluation(candidate, opponent, **kwargs):
        result = base(candidate, opponent, **kwargs)
        if len(calls) == 1:
            (dataset_dir / "games.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
        return result

    with pytest.raises(ValueError, match="sealed dataset changed"):
        module.evaluate_final(
            panel_dir, evaluator=mutate_during_evaluation, server_cmd=["fake-server"]
        )

    assert not (panel_dir / "final-evaluation.json").exists()


def _report_matches() -> list[dict[str, Any]]:
    rows = []
    outcome_counts = {
        "initialized_ppo": (325, 100, 75),
        "scratch_ppo": (250, 150, 100),
        "incumbent_ppo": (200, 200, 100),
    }
    for condition, (wins, losses, draws) in outcome_counts.items():
        for model_seed in (211, 223, 227):
            outcomes = ["win"] * wins + ["loss"] * losses + ["draw"] * draws
            for index, outcome in enumerate(outcomes):
                rows.append({
                    "condition": condition,
                    "model_seed": model_seed,
                    "seed": 17_000_000 + index // 2,
                    "candidate_seat": index % 2,
                    "outcome": outcome,
                    "summary": {
                        "round_count": 10 + index % 3,
                        "command_count": 20 + index % 5,
                        "wasted_end_turns_by_seat": [index % 4, (index + 1) % 4],
                        "peak_normalized_advantage": 99 if outcome == "draw" else index % 7,
                    },
                    "classification": {"primary": "lopsided"} if outcome == "draw" else None,
                })
    return rows


def _report_evidence() -> dict[str, Any]:
    return {
        "clone": {"pooled": {"wins": 900, "losses": 450, "draws": 150, "games": 1500}},
        "conversion": {"initialized_wins": 71, "initialized_games": 90},
        "bc_metrics": {"validation_loss": 0.4321, "validation_accuracy": 0.8765},
        "learning_curves": {
            "initialized": [
                {"nominal_step": 12_800, "pooled_standard_win_rate": 0.25},
                {"nominal_step": 25_600, "pooled_standard_win_rate": 0.5},
                {"nominal_step": 51_200, "pooled_standard_win_rate": 0.75},
            ],
            "scratch": [
                {"nominal_step": 12_800, "pooled_standard_win_rate": 0.2},
                {"nominal_step": 25_600, "pooled_standard_win_rate": 0.4},
                {"nominal_step": 51_200, "pooled_standard_win_rate": 0.6},
            ],
        },
        "compute": {
            "teacher_games": 12_000,
            "bc_wall_clock": {"status": "unavailable"},
            "ppo_environment_steps": 153_600,
            "ppo_wall_clock": {"status": "available", "seconds": 301.25},
        },
    }


def test_exact_sign_test_and_statistics_use_wilson_and_preserve_draws() -> None:
    """Approximate tests or reclassifying lopsided draws can overstate the learned policy."""
    module = _subject()
    from ml_lab.evaluation import wilson_interval

    assert module.exact_sign_test(0, 0) == 1.0
    assert module.exact_sign_test(3, 0) == 0.25
    assert module.exact_sign_test(4, 1) == 0.375
    aggregate = module.build_final_aggregate(
        _report_matches(), supporting_evidence=_report_evidence()
    )
    primary = aggregate["conditions"]["initialized_ppo"]["pooled"]
    assert primary["counts"] == {"wins": 975, "losses": 300, "draws": 225, "games": 1500}
    assert primary["confidence_intervals"]["win"] == wilson_interval(975, 1500)
    assert primary["confidence_intervals"]["loss"] == wilson_interval(300, 1500)
    assert primary["confidence_intervals"]["draw"] == wilson_interval(225, 1500)
    assert primary["draw_categories"] == {"lopsided": 225}
    assert set(primary["seats"]) == {"candidate_as_p0", "candidate_as_p1"}
    assert primary["diagnostics"] == {
        "rounds": 10.998,
        "decisions": 22.0,
        "action_waste": 1.0,
        "peak_material_advantage": 17.39,
    }
    assert aggregate["comparisons"]["scratch_ppo"]["pairs"] == 1_500
    assert aggregate["comparisons"]["incumbent_ppo"]["pairs"] == 1_500
    assert aggregate["supporting_evidence"] == _report_evidence()


def test_report_recomputes_raw_matches_orders_primary_first_and_is_consistent(
    tmp_path: Path,
) -> None:
    """Copied summaries or secondary-first prose can silently substitute diagnostics for the gate."""
    module = _subject()
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    matches = _report_matches()

    aggregate, report = module.publish_final_report(
        panel_dir, matches=matches, supporting_evidence=_report_evidence()
    )

    raw_primary = [row for row in matches if row["condition"] == "initialized_ppo"]
    recomputed = {
        "wins": sum(row["outcome"] == "win" for row in raw_primary),
        "losses": sum(row["outcome"] == "loss" for row in raw_primary),
        "draws": sum(row["outcome"] == "draw" for row in raw_primary),
        "games": len(raw_primary),
    }
    assert aggregate["conditions"]["initialized_ppo"]["pooled"]["counts"] == recomputed
    assert report.index("## Primary milestone gate") < report.index("## Clone") < report.index("## Initialized PPO")
    assert report.index("## Initialized PPO") < report.index("## Scratch PPO") < report.index("## Incumbent PPO")
    assert "| Pooled | 975 | 300 | 225 | 1500 | 65.000% | FAIL |" in report
    assert "Scratch pooled W/L/D: 750/450/300" in report
    assert "exact two-sided sign p=3.7092061506874214e-68" in report
    assert "Incumbent pooled W/L/D: 600/600/300" in report
    assert "exact two-sided sign p=2.598852441411225e-113" in report
    assert "Loss/draw Wilson 95% intervals:" in report
    assert "Comparator seat summaries:" in report
    assert "Comparator diagnostics:" in report
    assert "Comparator draw categories:" in report
    assert "900/450/150 over 1500 games" in report
    assert "71/90" in report
    assert "validation loss 0.4321" in report
    assert "12800: 25.000%" in report and "51200: 75.000%" in report
    assert "BC wall clock unavailable" in report
    assert "teacher games 12000" in report and "PPO environment steps 153600" in report
    published_aggregate, published_report, manifest = module.load_final_publication(panel_dir)
    assert published_aggregate == aggregate
    assert published_report == report
    assert manifest["generation"]


def test_report_publication_failure_preserves_prior_atomic_generation(tmp_path: Path) -> None:
    """A crash between generation files must preserve the prior reader-visible pair."""
    module = _subject()
    panel_dir = tmp_path / "panel"
    panel_dir.mkdir()
    first_aggregate, first_report = module.publish_final_report(
        panel_dir, matches=_report_matches(), supporting_evidence=_report_evidence()
    )
    first_manifest_bytes = (panel_dir / "final-publication.json").read_bytes()
    first_manifest = _json(panel_dir / "final-publication.json")

    def fail_between_files(stage: str) -> None:
        if stage == "after_staged_aggregate":
            raise OSError("injected")

    with pytest.raises(OSError, match="injected"):
        module.publish_final_report(
            panel_dir,
            matches=[dict(row, outcome="loss") for row in _report_matches()],
            supporting_evidence=_report_evidence(),
            failure_injector=fail_between_files,
        )

    assert (panel_dir / "final-publication.json").read_bytes() == first_manifest_bytes
    aggregate, report, manifest = module.load_final_publication(panel_dir)
    assert aggregate == first_aggregate
    assert report == first_report
    assert manifest == first_manifest
    generation = panel_dir / ".final-generations" / first_manifest["generation"]
    assert (generation / "aggregate.json").is_file()
    assert (generation / "REPORT.md").is_file()


def test_selected_incumbents_accepts_production_aggregate_schema(tmp_path: Path) -> None:
    module = _subject()
    panel = tmp_path / "python" / "panels" / "incumbent"
    runs = tmp_path / "python" / "runs"
    panel.mkdir(parents=True)
    models = {}
    for seed in (101, 113, 127):
        run_name = f"profiled-standard-seed-{seed}"
        run = runs / run_name
        checkpoint = run / "checkpoints" / "step_000051200.zip"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(str(seed).encode())
        (run / "scenario.json").write_text(json.dumps({
            "environment": "tactical-v2",
            "id": "tactical-v2-annihilation-profiled-standard",
            "tactical_v2": {"start_distribution": [
                {"profile_id": "standard-3v3", "basis_points": 10_000},
            ]},
        }), encoding="utf-8")
        (run / "run.json").write_text(json.dumps({
            "state": "completed",
            "config": {"seed": seed, "algorithm": "maskable_ppo", "environment": "tactical-v2"},
            "contract": {"contract_hash": "c" * 64, "encoding_hash": "e" * 64},
            "scenario": {"path": "scenario.json"},
            "latest_checkpoint": "checkpoints/step_000051200.zip",
            "latest_checkpoint_step": 51_200,
        }), encoding="utf-8")
        models[str(seed)] = {"training": {
            "run": run_name,
            "checkpoint": "checkpoints/step_000051200.zip",
            "checkpoint_sha256": _sha256(checkpoint),
        }}
    (panel / "aggregate.json").write_text(json.dumps({
        "schema_version": 1,
        "models": {"profiled_standard": models},
    }), encoding="utf-8")

    selected = module._selected_incumbents(panel)

    assert [item["pairing_seed"] for item in selected] == [211, 223, 227]
    assert [item["incumbent_seed"] for item in selected] == [101, 113, 127]
    assert all(item["algorithm"] == "maskable_ppo" for item in selected)
    assert all(item["step"] == 51_200 and item["contract_hash"] == "c" * 64 for item in selected)


def test_report_evidence_is_derived_from_hash_sealed_artifacts(tmp_path: Path) -> None:
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(
        panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
        revision="37cc8f9", dirty=False,
    )

    evidence = module._sealed_report_evidence(panel_dir)

    assert evidence == {
        "clone": {"pooled": {"wins": 30, "losses": 15, "draws": 15, "games": 60}},
        "conversion": {"initialized_wins": 3, "initialized_games": 6},
        "bc_metrics": {"validation_loss": 0.4000000000000001,
                       "validation_accuracy": 0.8000000000000002},
        "learning_curves": {
            "initialized": [
                {"nominal_step": 12_800, "pooled_standard_win_rate": 0.0},
                {"nominal_step": 25_600, "pooled_standard_win_rate": 0.5},
                {"nominal_step": 51_200, "pooled_standard_win_rate": 1.0},
            ],
            "scratch": [
                {"nominal_step": 12_800, "pooled_standard_win_rate": 0.0},
                {"nominal_step": 25_600, "pooled_standard_win_rate": 0.5},
                {"nominal_step": 51_200, "pooled_standard_win_rate": 1.0},
            ],
        },
        "compute": {
            "teacher_games": 1,
            "bc_wall_clock": {"status": "unavailable"},
            "ppo_environment_steps": 153_600,
            "ppo_wall_clock": {"status": "unavailable"},
        },
    }


@pytest.mark.parametrize("mutation", ["missing", "negative", "nonfinite"])
def test_report_evidence_rejects_missing_or_invalid_bc_timing_manifests(
    tmp_path: Path, mutation: str,
) -> None:
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    runs = [panel_dir / "bc-clones" / f"seed-{seed}" / "run.json"
            for seed in (211, 223, 227)]
    if mutation == "missing":
        for path in runs:
            path.unlink()
    else:
        manifest = _json(runs[0])
        manifest["wall_clock_seconds"] = -1.0 if mutation == "negative" else float("nan")
        runs[0].write_text(json.dumps(manifest), encoding="utf-8")
    module.freeze_final(
        panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
        revision="37cc8f9", dirty=False,
    )

    with pytest.raises(ValueError, match="BC compute evidence"):
        module._sealed_report_evidence(panel_dir)


def test_report_evidence_rejects_missing_conversion_instead_of_zero_over_zero(
    tmp_path: Path,
) -> None:
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    development_path = panel_dir / "development" / "development.json"
    development = _json(development_path)
    for row in development["candidates"]:
        row["conversion"] = []
    development_path.write_text(json.dumps(development), encoding="utf-8")
    module.freeze_final(
        panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
        revision="37cc8f9", dirty=False,
    )

    with pytest.raises(ValueError, match="conversion report evidence is unavailable"):
        module._sealed_report_evidence(panel_dir)


@pytest.mark.parametrize("mutation", ["schema", "state", "schedule", "seal"])
def test_report_rejects_invalid_or_transplanted_final_evaluation_envelope(
    tmp_path: Path, mutation: str,
) -> None:
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(
        panel_dir, incumbent_panel=incumbent, dataset_dir=dataset_dir,
        revision="37cc8f9", dirty=False,
    )
    seal_hash = _sha256(panel_dir / "final-seal.json")
    payload = {
        "schema_version": 1,
        "state": "completed",
        "seal_sha256": seal_hash,
        "schedule": dict(module._FINAL),
        "matches": [
            row for row in _report_matches() if row["condition"] == "initialized_ppo"
        ],
        "comparison_matches": [
            row for row in _report_matches() if row["condition"] != "initialized_ppo"
        ],
    }
    if mutation == "schema":
        payload["schema_version"] = 2
    elif mutation == "state":
        payload["state"] = "running"
    elif mutation == "schedule":
        payload["schedule"]["maps"] = 249
    else:
        payload["seal_sha256"] = "0" * 64
    (panel_dir / "final-evaluation.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="final evaluation envelope"):
        module.publish_final_report(
            panel_dir, supporting_evidence=_report_evidence()
        )

    assert not (panel_dir / "final-publication.json").exists()


def test_parser_exposes_final_seal_commands() -> None:
    """Dropping a final command makes the sealed workflow impossible to execute."""
    module = _subject()
    parser = module.build_parser()
    actions = next(action for action in parser._actions if action.dest == "command")
    assert set(actions.choices) == {
        "validate", "collect", "train-bc", "evaluate-bc", "train-ppo", "smoke",
        "evaluate-dev", "select-budget", "freeze-final", "evaluate-final", "report",
    }


def test_final_commands_are_defined_before_the_script_entry_point(tmp_path: Path) -> None:
    """Moving final functions below main's invocation makes the advertised CLI crash."""
    module = _subject()

    completed = subprocess.run(
        [sys.executable, str(Path(module.__file__).resolve()), "freeze-final",
         "--incumbent-panel", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "ValueError:" in completed.stderr
    assert "NameError" not in completed.stderr


@pytest.mark.parametrize(
    "mutation",
    ["definition", "dataset", "training", "control", "incumbent"],
)
def test_final_evaluation_rejects_any_changed_sealed_input(
    tmp_path: Path, mutation: str
) -> None:
    """A final run after any frozen input changes is not a sealed evaluation."""
    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    module.freeze_final(
        panel_dir,
        incumbent_panel=incumbent,
        dataset_dir=dataset_dir,
        revision="37cc8f9",
        dirty=False,
    )
    if mutation == "definition":
        (panel_dir / "panel.json").write_text('{"changed":true}', encoding="utf-8")
    elif mutation == "dataset":
        (dataset_dir / "games.jsonl").write_text('{"changed":true}\n', encoding="utf-8")
    elif mutation == "training":
        run = panel_dir / "ppo-runs" / "bc-ppo-seed-211" / "run.json"
        run.write_text('{"changed":true}', encoding="utf-8")
    elif mutation == "control":
        checkpoint = (
            panel_dir / "ppo-runs" / "scratch-ppo-seed-211"
            / "checkpoints" / "step_000025600.zip"
        )
        checkpoint.write_bytes(b"changed")
    else:
        seal = _json(panel_dir / "final-seal.json")
        run = Path(seal["incumbent_comparators"][0]["run_path"]) / "run.json"
        run.write_text('{"changed":true}', encoding="utf-8")

    with pytest.raises(ValueError, match="sealed"):
        module.evaluate_final(
            panel_dir,
            evaluator=_final_evaluator([]),
            server_cmd=["fake-server"],
        )


@pytest.mark.parametrize(
    ("wins", "games"),
    [
        ({211: 325, 223: 325, 227: 400}, 499),
        ({211: -1, 223: 400, 227: 400}, 500),
        ({211: 501, 223: 325, 227: 325}, 500),
    ],
)
def test_final_gate_rejects_nonprotocol_counts(wins: dict[int, int], games: int) -> None:
    """Non-500 denominators or impossible win counts cannot enter the final gate."""
    module = _subject()
    with pytest.raises(ValueError, match="500|wins"):
        module.apply_final_gate(wins=wins, games=games)


def test_parser_exposes_end_to_end_smoke_command() -> None:
    """Removing the isolated smoke entry point must break the documented gate."""

    assert _subject().build_parser().parse_args(["smoke"]).command == "smoke"


def test_smoke_schedule_is_exact_and_disjoint_from_full_experiment_artifacts() -> None:
    """A changed teacher/profile/seed schedule could stop exercising a required boundary."""

    module = _subject()

    assert module.build_smoke_schedule() == {
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
            "device": "cpu",
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
    assert module.SMOKE_ROOT == module.PANEL_ROOT / "evidence" / "smoke"
    assert module.SMOKE_ROOT not in {
        module.DATASET_PATH,
        module.CLONE_RUNS_PATH,
        module.CLONE_EVALUATION_PATH,
        module.PPO_RUNS_PATH,
        module.DEVELOPMENT_PATH,
    }


def _smoke_checks() -> dict[str, Any]:
    return {
        "reciprocal_pairs": 7,
        "games": 14,
        "teacher_labels": 19,
        "masked_labels": 0,
        "round_trip_mismatches": 0,
        "replay_mismatches": 0,
        "actor_fixture_max_error": 0.0,
        "actor_fixture_device": "cpu",
        "ppo_timesteps": 2,
        "ppo_completed_rollouts": 1,
        "evaluation_games": 4,
        "checkpoint_reloaded": True,
    }


def _smoke_artifacts() -> dict[str, str]:
    return {
        "dataset/manifest.json": "a" * 64,
        "bc/run.json": "b" * 64,
        "bc/metrics.json": "c" * 64,
        "ppo/run.json": "d" * 64,
        "ppo/initialization.json": "e" * 64,
        "ppo/checkpoints/step_000000002.zip": "f" * 64,
        "evaluation/evaluation.json": "1" * 64,
        "evaluation/evidence/traces/match-000000.json": "2" * 64,
        "evaluation/evidence/replays/match-000000.replay": "3" * 64,
    }


def test_smoke_manifest_has_exact_versioned_physical_evidence_contract() -> None:
    """Missing checks or unbound artifact bytes could synthesize a passing smoke."""

    module = _subject()
    expected = {
        "schema_version": 1,
        "state": "completed",
        "schedule": module.build_smoke_schedule(),
        "checks": _smoke_checks(),
        "artifacts": _smoke_artifacts(),
        "repository_identity": _smoke_repository_identity(),
    }

    assert module.build_smoke_manifest(
        checks=_smoke_checks(), artifacts=_smoke_artifacts(),
        repository_identity=_smoke_repository_identity(),
    ) == expected

    invalid = _smoke_checks()
    invalid["masked_labels"] = 1
    with pytest.raises(ValueError, match="smoke checks"):
        module.build_smoke_manifest(
            checks=invalid, artifacts=_smoke_artifacts(),
            repository_identity=_smoke_repository_identity(),
        )


def test_smoke_stage_is_isolated_and_failure_cannot_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed smoke must not publish or mutate any full experiment artifact."""

    module = _subject()
    full_dataset = tmp_path / "full-dataset"
    full_clones = tmp_path / "full-clones"
    full_ppo = tmp_path / "full-ppo"
    for root in (full_dataset, full_clones, full_ppo):
        root.mkdir()
        (root / "sentinel.txt").write_text(root.name, encoding="utf-8")
    monkeypatch.setattr(module, "DATASET_PATH", full_dataset)
    monkeypatch.setattr(module, "CLONE_RUNS_PATH", full_clones)
    monkeypatch.setattr(module, "PPO_RUNS_PATH", full_ppo)
    smoke_root = tmp_path / "evidence" / "smoke"

    def fail(staging: Path) -> None:
        assert staging.parent == smoke_root.parent
        raise RuntimeError("injected smoke failure")

    with pytest.raises(RuntimeError, match="injected smoke failure"):
        module.run_smoke_stage(
            smoke_root,
            {"panel_sha256": "a" * 64},
            build=fail,
            validate=lambda _root: pytest.fail("failed build reached validation"),
        )

    assert not smoke_root.exists()
    assert not (smoke_root / "smoke.json").exists()
    assert (full_dataset / "sentinel.txt").read_text(encoding="utf-8") == "full-dataset"
    assert (full_clones / "sentinel.txt").read_text(encoding="utf-8") == "full-clones"
    assert (full_ppo / "sentinel.txt").read_text(encoding="utf-8") == "full-ppo"


def test_smoke_command_reopens_hashes_and_atomically_publishes_only_smoke_root(
    tmp_path: Path,
) -> None:
    """Publishing without reopening named bytes could accept stale or missing evidence."""

    module = _subject()
    smoke_root = tmp_path / "evidence" / "smoke"
    identity = _smoke_repository_identity()

    def pipeline(staging: Path, _scenario, _hashes, repository_identity) -> None:
        artifacts: dict[str, str] = {}
        for relative in _smoke_artifacts():
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode("utf-8"))
            artifacts[relative] = _sha256(path)
        module._atomic_json(
            staging / "smoke.json",
            module.build_smoke_manifest(
                checks=_smoke_checks(),
                artifacts=artifacts,
                repository_identity=repository_identity,
            ),
        )

    result = module._smoke_command(
        smoke_root=smoke_root,
        pipeline=pipeline,
        repository_identity_provider=lambda _repository: identity,
    )

    manifest = _json(smoke_root / "smoke.json")
    assert result["summary"] == _smoke_checks()
    assert result["reused"] is False
    assert manifest == module.build_smoke_manifest(
        checks=_smoke_checks(),
        artifacts={
            relative: _sha256(smoke_root / relative)
            for relative in _smoke_artifacts()
        },
        repository_identity=identity,
    )

    assert not smoke_root.with_name(".smoke.staging").exists()

    changed = smoke_root / next(iter(manifest["artifacts"]))
    changed.write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash"):
        module._validate_smoke_stage(smoke_root)


def test_main_dispatches_smoke_and_prints_completed_manifest(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A registered parser choice without main dispatch would remain unusable."""

    module = _subject()
    expected = module.build_smoke_manifest(
        checks=_smoke_checks(), artifacts=_smoke_artifacts(),
        repository_identity=_smoke_repository_identity(),
    )
    monkeypatch.setattr(module, "_smoke_command", lambda: expected)

    module.main(["smoke"])

    assert json.loads(capsys.readouterr().out) == expected


def test_panel_server_command_uses_the_required_default_build_output(
    tmp_path: Path,
) -> None:
    """Launching stale Release bytes would ignore the GymServer built by the smoke gate."""

    module = _subject()
    scenario = tmp_path / "scenario.json"
    expected_server = (
        module.PROJECT_ROOT / "engine" / "HexWars.GymServer"
        / "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
    )
    assert module._server_command(scenario) == [
        "dotnet", str(expected_server), "--scenario-file", str(scenario),
    ]


def test_smoke_restart_rotates_failed_ppo_outside_publish_root_and_reuses_completed(
    tmp_path: Path,
) -> None:
    """A retry must preserve diagnostics without publishing failed-attempt artifacts."""

    from ml_lab.contracts import RunConfig

    module = _subject()
    smoke_staging = tmp_path / "evidence" / ".smoke.staging"
    runs_root = smoke_staging / "ppo"
    recovery_root = smoke_staging.with_name(".smoke.recovery") / "ppo"
    config = RunConfig(
        backend="sb3",
        algorithm="maskable_ppo",
        policy="HexCNN",
        run_name="initialized-ppo",
        seed=211,
        total_timesteps=2,
        checkpoint_interval=2,
        workers=1,
        device="cpu",
        learner_seat="alternating",
        opponent={"kind": "scripted", "name": "random"},
        trackers=[],
        resume_source=None,
        environment="tactical-v2",
        actor_init_source=str((smoke_staging / "bc").resolve()),
    )
    failed = runs_root / config.run_name
    failed.mkdir(parents=True)
    (failed / "run.json").write_text(
        json.dumps({
            "schema_version": 1,
            "state": "failed",
            "config": config.to_dict(),
        }),
        encoding="utf-8",
    )
    (failed / "failure.txt").write_text("actor init failed", encoding="utf-8")
    calls: list[Path] = []

    def trainer(_config, *, runs_root: Path, **_kwargs) -> Path:
        calls.append(runs_root)
        run = runs_root / _config.run_name
        run.mkdir(parents=True)
        (run / "run.json").write_text(
            json.dumps({
                "schema_version": 1,
                "state": "completed",
                "config": _config.to_dict(),
            }),
            encoding="utf-8",
        )
        return run

    completed = module._run_restart_safe_smoke_training(
        config,
        runs_root=runs_root,
        recovery_root=recovery_root,
        trainer=trainer,
    )
    reused = module._run_restart_safe_smoke_training(
        config,
        runs_root=runs_root,
        recovery_root=recovery_root,
        trainer=lambda *_args, **_kwargs: pytest.fail("completed PPO was retrained"),
    )

    assert completed == reused == failed
    assert calls == [runs_root]
    assert _json(failed / "run.json")["state"] == "completed"
    assert (recovery_root / "initialized-ppo-attempt-0000" / "failure.txt").is_file()
    assert recovery_root.parent.parent == smoke_staging.parent
    assert recovery_root.parent not in smoke_staging.parents
    assert not any("recovery" in path.parts for path in smoke_staging.rglob("*"))


def test_smoke_evidence_reopens_dataset_with_exact_bc_source_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The duel dataset is authoritative even when PPO uses a compatible tactical horizon."""

    from ml_lab.contracts import EnvironmentContract
    import ml_lab.imitation

    module = _subject()
    source = EnvironmentContract(
        version="tactical-v2",
        contract_hash="a" * 64,
        encoding_hash="c" * 64,
        observation_size=12,
        action_size=8,
        board={"environment_kind": "duel", "max_steps": 20},
        roster=["unit"],
        reward={"terminal_win": 1},
    )
    target = EnvironmentContract(
        version="tactical-v2",
        contract_hash="b" * 64,
        encoding_hash=source.encoding_hash,
        observation_size=source.observation_size,
        action_size=source.action_size,
        board={"environment_kind": "tactical", "max_steps": 10},
        roster=list(source.roster),
        reward=dict(source.reward),
    )
    dataset_root = tmp_path / "dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text(
        json.dumps({
            "contract_hash": source.contract_hash,
            "encoding_hash": source.encoding_hash,
        }),
        encoding="utf-8",
    )
    seen: list[EnvironmentContract] = []

    def loader(root: Path, expected_contract: EnvironmentContract):
        assert root == dataset_root
        seen.append(expected_contract)
        return SimpleNamespace(contract=expected_contract)

    monkeypatch.setattr(ml_lab.imitation, "load_imitation_dataset", loader)
    dataset, loaded_source = module._load_smoke_source_dataset(
        dataset_root,
        {"contract": source.to_dict()},
    )
    module._validate_smoke_actor_source(
        source_contract=loaded_source,
        target_contract=target,
        resolved_contract=dataset.contract,
        initialization={
            "source_contract_hash": source.contract_hash,
            "source_encoding_hash": source.encoding_hash,
        },
    )

    assert seen == [source]
    assert loaded_source.contract_hash != target.contract_hash
    with pytest.raises(ValueError, match="source contract"):
        module._validate_smoke_actor_source(
            source_contract=loaded_source,
            target_contract=target,
            resolved_contract=dataset.contract,
            initialization={
                "source_contract_hash": target.contract_hash,
                "source_encoding_hash": source.encoding_hash,
            },
        )


def test_smoke_restart_validates_and_reuses_completed_bc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed clone may be reused only after exact physical and contract validation."""

    from ml_lab.contracts import EnvironmentContract
    from ml_lab.imitation import BehavioralCloningConfig

    module = _subject()
    source = EnvironmentContract(
        version="tactical-v2",
        contract_hash="a" * 64,
        encoding_hash="c" * 64,
        observation_size=12,
        action_size=8,
        board={},
        roster=["unit"],
        reward={},
    )
    config = BehavioralCloningConfig(
        model_seed=211,
        batch_size=32,
        learning_rate=3e-4,
        max_epochs=1,
        patience=1,
        device="cpu",
    )
    run = tmp_path / "bc"
    run.mkdir()
    (run / "run.json").write_text(
        json.dumps({
            "schema_version": 1,
            "state": "completed",
            "contract": source.to_dict(),
            "bc_config": {
                "model_seed": 211,
                "batch_size": 32,
                "learning_rate": 3e-4,
                "max_epochs": 1,
                "patience": 1,
                "device": "cpu",
            },
        }),
        encoding="utf-8",
    )
    dataset_manifest = tmp_path / "dataset" / "manifest.json"
    dataset_manifest.parent.mkdir()
    dataset_manifest.write_text("{}", encoding="utf-8")
    scenario = SimpleNamespace(canonical_json="{}")
    calls: list[Path] = []

    def validate(path, seed, hashes, **kwargs):
        calls.append(path)
        assert seed == 211
        assert hashes == {"panel": "hash"}
        assert kwargs == {
            "expected_scenario": scenario,
            "require_provenance": False,
            "expected_dataset_manifest": dataset_manifest,
            "expected_device": "cpu",
        }
        return {
            "contract_hash": source.contract_hash,
            "encoding_hash": source.encoding_hash,
        }

    monkeypatch.setattr(module, "_validate_clone_run", validate)
    assert module._reuse_completed_smoke_bc(
        run, dataset_manifest, scenario, {"panel": "hash"}, source, config,
    ) is True
    assert calls == [run]

    manifest = _json(run / "run.json")
    manifest["contract"]["contract_hash"] = "b" * 64
    (run / "run.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="contract"):
        module._reuse_completed_smoke_bc(
            run, dataset_manifest, scenario, {"panel": "hash"}, source, config,
        )


def test_smoke_evidence_collector_is_a_callable_module_boundary() -> None:
    """An indented evidence body without its function header cannot seal a physical smoke."""

    collector = getattr(_subject(), "_collect_smoke_evidence", None)
    assert callable(collector)


def _smoke_repository_identity(
    *, commit: str = "a" * 40, source_tree: str = "b" * 40, dirty: bool = False,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "commit": commit,
        "source_tree": source_tree,
        "dirty": dirty,
        "policy": {
            "required_clean": True,
            "ignored_generated_root": "python/panels/annihilation-imitation-v1/evidence/",
        },
    }


@pytest.mark.parametrize(
    ("code_revision", "dirty"),
    [
        ("c" * 40, False),
        ("a" * 40, True),
    ],
    ids=("mismatched-revision", "dirty-dataset"),
)
def test_smoke_evidence_rejects_dataset_from_different_or_dirty_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    code_revision: str,
    dirty: bool,
) -> None:
    """Physical evidence must come from the same clean code identity as its smoke."""

    module = _subject()
    root = tmp_path / "evidence" / "smoke"
    identity = _smoke_repository_identity()
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True)
    (dataset_root / "manifest.json").write_text(
        json.dumps({
            "code_revision": code_revision,
            "dirty": dirty,
        }),
        encoding="utf-8",
    )
    ppo_manifest = root / "ppo" / "initialized-ppo" / "run.json"
    bc_manifest = root / "bc" / "run.json"
    original_read = module._read_json

    def read_manifest(path: Path) -> dict[str, Any]:
        path = Path(path)
        if path == ppo_manifest:
            return {"contract": {}}
        if path == bc_manifest:
            return {}
        return original_read(path)

    monkeypatch.setattr(module, "_read_json", read_manifest)
    monkeypatch.setattr(module, "_smoke_contract", lambda _raw: object())
    monkeypatch.setattr(
        module,
        "_load_smoke_source_dataset",
        lambda *_args: pytest.fail("invalid dataset identity reached dataset loading"),
    )

    with pytest.raises(ValueError, match="smoke dataset repository identity"):
        module._collect_smoke_evidence(
            root,
            expected_repository_identity=identity,
        )


def test_smoke_validation_failure_removes_only_staged_completion_manifest(
    tmp_path: Path,
) -> None:
    """A post-build validation error must not leave a completed-looking staged smoke."""

    module = _subject()
    destination = tmp_path / "evidence" / "smoke"
    staging = destination.with_name(".smoke.staging")

    def build(root: Path) -> None:
        (root / "diagnostic.txt").write_text("preserve me", encoding="utf-8")
        (root / "smoke.json").write_text(
            '{"schema_version":1,"state":"completed"}', encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="post-manifest failure"):
        module.run_smoke_stage(
            destination,
            {"panel_sha256": "a" * 64},
            build=build,
            validate=lambda _root: (_ for _ in ()).throw(
                RuntimeError("post-manifest failure")
            ),
        )

    assert not destination.exists()
    assert staging.is_dir()
    assert (staging / "diagnostic.txt").read_text(encoding="utf-8") == "preserve me"
    assert not (staging / "smoke.json").exists()


def test_smoke_completion_cleanup_failure_preserves_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A filesystem cleanup error must annotate, not mask, the publication failure."""

    module = _subject()
    destination = tmp_path / "evidence" / "smoke"
    staging = destination.with_name(".smoke.staging")
    completion = staging / "smoke.json"
    original_unlink = Path.unlink

    def fail_completion_unlink(path: Path, *args, **kwargs) -> None:
        if path == completion:
            raise OSError("injected cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_completion_unlink)

    def build(root: Path) -> None:
        (root / "smoke.json").write_text(
            '{"schema_version":1,"state":"completed"}', encoding="utf-8",
        )

    with pytest.raises(RuntimeError, match="original validation failure") as captured:
        module.run_smoke_stage(
            destination,
            {"panel_sha256": "a" * 64},
            build=build,
            validate=lambda _root: (_ for _ in ()).throw(
                RuntimeError("original validation failure")
            ),
        )

    assert any(
        "injected cleanup failure" in note
        for note in getattr(captured.value, "__notes__", ())
    )


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("reciprocal_pairs", 7.0),
        ("games", 14.0),
        ("teacher_labels", 19.0),
        ("masked_labels", False),
        ("round_trip_mismatches", False),
        ("replay_mismatches", False),
        ("ppo_timesteps", 2.0),
        ("ppo_completed_rollouts", True),
        ("evaluation_games", 4.0),
    ],
)
def test_smoke_manifest_rejects_bool_or_float_count_fields(
    field: str, invalid: Any,
) -> None:
    """Python equality must not let bools/floats impersonate integer evidence counts."""

    module = _subject()
    checks = _smoke_checks()
    checks[field] = invalid
    with pytest.raises(ValueError, match="smoke checks"):
        module.build_smoke_manifest(
            checks=checks, artifacts=_smoke_artifacts(),
            repository_identity=_smoke_repository_identity(),
        )


def test_smoke_manifest_rejects_boolean_actor_error() -> None:
    """False compares equal to zero but is not a numeric logit-error measurement."""

    module = _subject()
    checks = _smoke_checks()
    checks["actor_fixture_max_error"] = False
    with pytest.raises(ValueError, match="smoke checks"):
        module.build_smoke_manifest(
            checks=checks, artifacts=_smoke_artifacts(),
            repository_identity=_smoke_repository_identity(),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "float-seed",
        "bool-seat",
    ],
)
def test_smoke_evaluation_matches_require_four_exact_unique_integer_keys(
    mutation: str,
) -> None:
    """Set equality alone must not accept duplicate or type-aliased match identities."""

    module = _subject()
    matches = [
        {"seed": 18_200_000, "candidate_seat": 0},
        {"seed": 18_200_000, "candidate_seat": 1},
        {"seed": 18_200_001, "candidate_seat": 0},
        {"seed": 18_200_001, "candidate_seat": 1},
    ]
    if mutation == "duplicate":
        matches.append(dict(matches[0]))
    elif mutation == "float-seed":
        matches[0]["seed"] = 18_200_000.0
    else:
        matches[0]["candidate_seat"] = False

    with pytest.raises(ValueError, match="matches"):
        module._validate_smoke_evaluation_matches(
            matches, seed_start=18_200_000, maps=2,
        )


def test_atomic_stage_identity_prevents_cross_revision_reuse(tmp_path: Path) -> None:
    """Matching definitions cannot authorize reuse of artifacts from different code."""

    module = _subject()
    destination = tmp_path / "published"
    definitions = {"panel_sha256": "a" * 64}
    first_identity = _smoke_repository_identity()
    changed_identity = _smoke_repository_identity(commit="c" * 40, source_tree="d" * 40)

    def build(staging: Path) -> None:
        (staging / "payload.txt").write_text("complete", encoding="utf-8")

    module.run_atomic_stage(
        destination,
        definitions,
        stage_identity=first_identity,
        build=build,
        validate=lambda _root: {"outputs": 1},
    )
    with pytest.raises(ValueError, match="stage identity"):
        module.run_atomic_stage(
            destination,
            definitions,
            stage_identity=changed_identity,
            build=lambda _root: pytest.fail("cross-revision stage was rebuilt"),
            validate=lambda _root: pytest.fail("cross-revision stage was reused"),
        )


def test_smoke_repository_identity_requires_clean_source_and_exact_policy() -> None:
    """A dirty source tree or relaxed evidence exclusion cannot be authoritative."""

    module = _subject()
    module._validate_smoke_repository_identity(_smoke_repository_identity())
    dirty = _smoke_repository_identity(dirty=True)
    with pytest.raises(ValueError, match="clean"):
        module._validate_smoke_repository_identity(dirty)
    relaxed = _smoke_repository_identity()
    relaxed["policy"]["ignored_generated_root"] = "python/panels/"
    with pytest.raises(ValueError, match="policy"):
        module._validate_smoke_repository_identity(relaxed)


def test_smoke_command_rechecks_repository_identity_after_physical_validation(
    tmp_path: Path,
) -> None:
    """A tracked change during the long smoke must fail before atomic publication."""

    module = _subject()
    destination = tmp_path / "evidence" / "smoke"
    initial = _smoke_repository_identity()
    changed = _smoke_repository_identity(commit="c" * 40, source_tree="d" * 40)
    identities = iter((initial, changed))

    def identity_provider(_repository: Path) -> dict[str, Any]:
        return next(identities)

    def pipeline(staging: Path, _scenario, _hashes, identity) -> None:
        artifact = staging / "artifact.txt"
        artifact.write_text("physical", encoding="utf-8")
        module._atomic_json(
            staging / "smoke.json",
            module.build_smoke_manifest(
                checks=_smoke_checks(),
                artifacts={"artifact.txt": _sha256(artifact)},
                repository_identity=identity,
            ),
        )

    with pytest.raises(ValueError, match="repository identity changed"):
        module._smoke_command(
            smoke_root=destination,
            pipeline=pipeline,
            repository_identity_provider=identity_provider,
        )
    assert not destination.exists()
    assert not (destination.with_name(".smoke.staging") / "smoke.json").exists()


@pytest.mark.parametrize(
    "destination",
    [
        "selection.json",
        "final-seal.json",
        "final-evaluation.json",
        ".final-evaluation.pending",
        "final-publication.json",
        ".final-generations",
    ],
)
def test_smoke_root_refuses_selection_and_final_artifact_destinations(
    destination: str,
) -> None:
    """An isolated smoke must never occupy a selection or final publication path."""

    module = _subject()
    with pytest.raises(ValueError, match="overlaps"):
        module.run_smoke_stage(
            module.PANEL_ROOT / destination,
            {"panel_sha256": "a" * 64},
            build=lambda _root: pytest.fail("protected destination reached build"),
            validate=lambda _root: pytest.fail("protected destination reached validation"),
        )


def _bc_boundary_contract(
    *,
    contract_hash: str,
    environment_kind: str,
    encoding_hash: str = "e" * 64,
    observation_size: int = 12,
    action_size: int = 8,
):
    from ml_lab.contracts import EnvironmentContract

    return EnvironmentContract(
        version="tactical-v2",
        contract_hash=contract_hash,
        encoding_hash=encoding_hash,
        observation_size=observation_size,
        action_size=action_size,
        board={"environment_kind": environment_kind},
        roster=["unit"],
        reward={"terminal_win": 1},
    )


def test_train_bc_loads_and_records_exact_duel_source_contract_for_tactical_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using the tactical hash for strict dataset loading would erase capture provenance."""

    module = _subject()
    identity, identity_kwargs = _command_identity_kwargs(tmp_path, module)
    dataset_root = tmp_path / "source-dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text(
        json.dumps({
            "code_revision": identity["commit"],
            "dirty": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATASET_PATH", dataset_root)
    source = _bc_boundary_contract(
        contract_hash="a" * 64, environment_kind="duel",
    )
    target = _bc_boundary_contract(
        contract_hash="b" * 64, environment_kind="tactical",
    )
    events: list[tuple[str, Any]] = []
    stage_scenarios: list[Path] = []
    dataset = SimpleNamespace(contract=source)

    class FakeDuelClient:
        def __init__(self, command, *, environment):
            assert environment == "tactical-v2"
            stage_scenarios.append(Path(command[command.index("--scenario-file") + 1]))
            self.contract = source

        def close(self) -> None:
            events.append(("closed", "duel"))

    class FakeEnv:
        def __init__(self, command, **kwargs):
            del command
            stage_scenarios.append(Path(kwargs["scenario_path"]))
            self.contract = target
            self.spaces_info = {"channels": 11}

        def close(self) -> None:
            events.append(("closed", "tactical"))

    def loader(root: Path, *, expected_contract):
        events.append(("loaded", (Path(root), expected_contract)))
        return dataset

    def train_clones(**kwargs):
        events.append(("trained", kwargs))
        return []

    def fake_atomic(_destination, _hashes, *, stage_identity, build, validate):
        del validate
        staging = tmp_path / "clones-stage"
        staging.mkdir()
        build(staging)
        return {"state": "completed"}

    import hexwars_gym
    import ml_lab.evaluation
    import ml_lab.imitation

    monkeypatch.setattr(hexwars_gym, "HexWarsEnv", FakeEnv)
    monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeDuelClient)
    monkeypatch.setattr(ml_lab.imitation, "load_imitation_dataset", loader)
    monkeypatch.setattr(module, "train_clone_runs", train_clones)
    monkeypatch.setattr(module, "run_atomic_stage", fake_atomic)

    module._train_bc_command(**identity_kwargs)

    loaded = next(value for name, value in events if name == "loaded")
    trained = next(value for name, value in events if name == "trained")
    assert loaded == (module.DATASET_PATH, source)
    assert trained["dataset"] is dataset
    assert trained["contract"] is source
    assert trained["env"].contract is target
    assert trained["spaces_info"] == {"channels": 11}
    assert source.contract_hash != target.contract_hash
    assert stage_scenarios[0] == stage_scenarios[1]
    assert events[-2:] == [("closed", "tactical"), ("closed", "duel")]


@pytest.mark.parametrize(
    "target_overrides",
    [
        {"encoding_hash": "f" * 64},
        {"observation_size": 13},
        {"action_size": 9},
    ],
    ids=("encoding", "observation-geometry", "action-geometry"),
)
def test_train_bc_rejects_incompatible_source_before_trainer_and_closes_resources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_overrides: dict[str, Any],
) -> None:
    """A source/target tensor mismatch must never enter behavioral-clone optimization."""

    module = _subject()
    identity, identity_kwargs = _command_identity_kwargs(tmp_path, module)
    dataset_root = tmp_path / "source-dataset"
    dataset_root.mkdir()
    (dataset_root / "manifest.json").write_text(
        json.dumps({
            "code_revision": identity["commit"],
            "dirty": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATASET_PATH", dataset_root)
    source = _bc_boundary_contract(
        contract_hash="a" * 64, environment_kind="duel",
    )
    target = _bc_boundary_contract(
        contract_hash="b" * 64,
        environment_kind="tactical",
        **target_overrides,
    )
    closed: list[str] = []
    trainer_calls: list[dict[str, Any]] = []

    class FakeDuelClient:
        def __init__(self, _command, *, environment):
            assert environment == "tactical-v2"
            self.contract = source

        def close(self) -> None:
            closed.append("duel")

    class FakeEnv:
        def __init__(self, _command, **_kwargs):
            self.contract = target
            self.spaces_info = {}

        def close(self) -> None:
            closed.append("tactical")

    def fake_atomic(_destination, _hashes, *, stage_identity, build, validate):
        del validate
        staging = tmp_path / "clones-stage"
        staging.mkdir()
        build(staging)
        return {"state": "completed"}

    import hexwars_gym
    import ml_lab.evaluation
    import ml_lab.imitation

    monkeypatch.setattr(hexwars_gym, "HexWarsEnv", FakeEnv)
    monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeDuelClient)
    monkeypatch.setattr(
        ml_lab.imitation,
        "load_imitation_dataset",
        lambda _root, *, expected_contract: SimpleNamespace(contract=expected_contract),
    )
    monkeypatch.setattr(
        module,
        "train_clone_runs",
        lambda **kwargs: trainer_calls.append(kwargs),
    )
    monkeypatch.setattr(module, "run_atomic_stage", fake_atomic)

    with pytest.raises(ValueError, match="incompatible"):
        module._train_bc_command(**identity_kwargs)

    assert trainer_calls == []
    assert closed == ["tactical", "duel"]

def test_validate_command_requires_clean_source_and_atomically_persists_identity(
    tmp_path: Path,
) -> None:
    """A dirty or unrecorded source must never authorize expensive experiment work."""

    module = _subject()
    path = tmp_path / "execution-identity.json"
    dirty_provider = lambda _repository: {
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": True,
    }
    with pytest.raises(ValueError, match="clean"):
        module._validate_command(
            execution_identity_path=path,
            repository=tmp_path,
            repository_identity_provider=dirty_provider,
        )
    assert not path.exists()

    clean_provider = lambda _repository: {
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": False,
    }
    result = module._validate_command(
        execution_identity_path=path,
        repository=tmp_path,
        repository_identity_provider=clean_provider,
    )
    expected = _execution_identity(module)

    assert result == {"state": "validated", "execution_identity": expected}
    assert _json(path) == expected
    assert not list(tmp_path.glob(".execution-identity.json.*.tmp"))


@pytest.mark.parametrize(
    "mutation",
    ["commit", "source-tree", "policy", "definitions"],
)
def test_execution_identity_reopen_rejects_every_changed_boundary(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A later command must fail closed if code, policy, or definitions drift."""

    module = _subject()
    stored = _execution_identity(module)
    path = tmp_path / "execution-identity.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    current = {
        "commit": stored["commit"],
        "source_tree": stored["source_tree"],
        "dirty": False,
    }
    hashes = dict(stored["definition_hashes"])
    if mutation == "commit":
        current["commit"] = "c" * 40
    elif mutation == "source-tree":
        current["source_tree"] = "d" * 40
    elif mutation == "policy":
        stored["policy"]["ignored_generated_paths"] = ["python/"]
        path.write_text(json.dumps(stored), encoding="utf-8")
    else:
        hashes["panel_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="execution identity"):
        module._require_execution_identity(
            execution_identity_path=path,
            definition_hashes=hashes,
            repository=tmp_path,
            repository_identity_provider=lambda _repository: current,
        )


def test_full_generated_output_policy_is_narrow_and_keeps_results_visible() -> None:
    """Ignoring too little dirties valid runs; ignoring result artifacts hides publication."""

    module = _subject()
    assert list(module._FULL_GENERATED_PATHS) == FULL_GENERATED_PATHS
    assert list(module._PUBLISHABLE_RESULT_PATHS) == PUBLISHABLE_RESULT_PATHS

    for relative in FULL_GENERATED_PATHS:
        probe = relative + "__identity_probe__" if relative.endswith("/") else relative
        ignored = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", probe],
            cwd=module.PROJECT_ROOT,
            check=False,
        )
        assert ignored.returncode == 0, relative
    for relative in PUBLISHABLE_RESULT_PATHS:
        visible = subprocess.run(
            ["git", "check-ignore", "--no-index", "--quiet", relative],
            cwd=module.PROJECT_ROOT,
            check=False,
        )
        assert visible.returncode == 1, relative
    source_fixture = subprocess.run(
        [
            "git", "check-ignore", "--no-index", "--quiet",
            "python/datasets/annihilation-imitation-v1/source-fixture.json",
        ],
        cwd=module.PROJECT_ROOT,
        check=False,
    )
    assert source_fixture.returncode == 1

@pytest.mark.parametrize(
    "command",
    [
        "collect",
        "train-bc",
        "evaluate-bc",
        "train-ppo",
        "evaluate-dev",
        "select-budget",
        "freeze-final",
        "evaluate-final",
        "report",
    ],
)
def test_every_full_command_rejects_changed_execution_identity_before_work(
    tmp_path: Path,
    command: str,
) -> None:
    """No downstream command may build, validate, reuse, freeze, evaluate, or report stale work."""

    module = _subject()
    stored = _execution_identity(module)
    path = tmp_path / "execution-identity.json"
    path.write_text(json.dumps(stored), encoding="utf-8")
    kwargs = {
        "execution_identity_path": path,
        "repository": tmp_path,
        "repository_identity_provider": lambda _repository: {
            "commit": "c" * 40,
            "source_tree": stored["source_tree"],
            "dirty": False,
        },
    }

    with pytest.raises(ValueError, match="execution identity"):
        if command == "collect":
            module._collect_command(**kwargs)
        elif command == "train-bc":
            module._train_bc_command(**kwargs)
        elif command == "evaluate-bc":
            module._evaluate_bc_command(**kwargs)
        elif command == "train-ppo":
            module._train_ppo_command(**kwargs)
        elif command == "evaluate-dev":
            module._evaluate_dev_command(**kwargs)
        elif command == "select-budget":
            module._select_budget_command(**kwargs)
        elif command == "freeze-final":
            module._freeze_final_command(
                incumbent_panel=tmp_path / "incumbent",
                **kwargs,
            )
        elif command == "evaluate-final":
            module._evaluate_final_command(**kwargs)
        else:
            module._report_command(**kwargs)

@pytest.mark.parametrize(
    "command",
    ["collect", "train-bc", "evaluate-bc", "train-ppo", "evaluate-dev"],
)
def test_reusable_full_stages_receive_the_exact_execution_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
) -> None:
    """Definitions alone must not authorize reuse of any long-running atomic stage."""

    module = _subject()
    identity, kwargs = _command_identity_kwargs(tmp_path, module)
    captured: dict[str, Any] = {}

    def atomic(destination, definitions, *, stage_identity, build, validate):
        del build, validate
        captured.update(
            destination=Path(destination),
            definitions=dict(definitions),
            stage_identity=dict(stage_identity),
        )
        return {"state": "completed"}

    monkeypatch.setattr(module, "run_atomic_stage", atomic)
    if command == "collect":
        import ml_lab.evaluation

        class FakeClient:
            def __init__(self, _command, *, environment):
                assert environment == "tactical-v2"
                self.contract = SimpleNamespace(
                    contract_hash="c" * 64,
                    encoding_hash="e" * 64,
                )

            def close(self) -> None:
                return None

        monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeClient)
        expected = module.DATASET_PATH
        module._collect_command(**kwargs)
    elif command == "train-bc":
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "manifest.json").write_text(
            json.dumps({
                "code_revision": identity["commit"],
                "dirty": False,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(module, "DATASET_PATH", dataset)
        expected = module.CLONE_RUNS_PATH
        module._train_bc_command(**kwargs)
    elif command == "evaluate-bc":
        monkeypatch.setattr(module, "_validate_clone_stage", lambda *_args: {})
        expected = module.CLONE_EVALUATION_PATH
        module._evaluate_bc_command(**kwargs)
    elif command == "train-ppo":
        monkeypatch.setattr(module, "_validate_clone_stage", lambda *_args: {})
        monkeypatch.setattr(module, "_validate_gate_stage", lambda *_args: {})
        expected = module.PPO_RUNS_PATH
        module._train_ppo_command(**kwargs)
    else:
        monkeypatch.setattr(module, "_validate_clone_stage", lambda *_args: {})
        monkeypatch.setattr(module, "_validate_gate_stage", lambda *_args: {})
        monkeypatch.setattr(module, "_validate_ppo_stage", lambda *_args: {})
        monkeypatch.setattr(
            module, "build_panel_development_candidates",
            lambda *_args, **_kwargs: [],
        )
        expected = module.DEVELOPMENT_PATH
        module._evaluate_dev_command(**kwargs)

    assert captured == {
        "destination": expected,
        "definitions": identity["definition_hashes"],
        "stage_identity": identity,
    }


@pytest.mark.parametrize(
    ("code_revision", "dirty"),
    [
        ("c" * 40, False),
        ("a" * 40, True),
    ],
)
def test_dataset_execution_identity_requires_exact_clean_revision(
    tmp_path: Path,
    code_revision: str,
    dirty: bool,
) -> None:
    """A valid imitation manifest from another or dirty source cannot enter this execution."""

    module = _subject()
    identity = _execution_identity(module)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps({
            "code_revision": code_revision,
            "dirty": dirty,
        }),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="dataset execution identity"):
        module._validate_dataset_execution_identity(dataset, identity)


def test_train_bc_rejects_wrong_dataset_identity_before_atomic_or_optimizer_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """BC must reject stale demonstrations before entering a reusable stage or trainer."""

    module = _subject()
    _identity, kwargs = _command_identity_kwargs(tmp_path, module)
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "manifest.json").write_text(
        json.dumps({
            "code_revision": "c" * 40,
            "dirty": False,
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "DATASET_PATH", dataset)
    monkeypatch.setattr(
        module,
        "run_atomic_stage",
        lambda *_args, **_kwargs: pytest.fail(
            "wrong-revision dataset reached atomic BC work"
        ),
    )

    with pytest.raises(ValueError, match="dataset execution identity"):
        module._train_bc_command(**kwargs)


@pytest.mark.parametrize(
    "command",
    ["collect", "train-bc", "evaluate-bc", "train-ppo", "evaluate-dev"],
)
@pytest.mark.parametrize("lifecycle", ["publication", "reuse"])
@pytest.mark.parametrize("mutation", ["repository", "definitions"])
def test_full_stage_rechecks_identity_after_physical_validation_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    command: str,
    lifecycle: str,
    mutation: str,
) -> None:
    """A long build/reuse may not publish after code or definitions drift."""

    module = _subject()
    identity = _execution_identity(module)
    identity_path = tmp_path / "execution-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    physical_root = tmp_path / "physical-stage"
    state = {"changed": False}
    events: list[str] = []

    def repository_identity(_repository: Path) -> dict[str, Any]:
        changed = state["changed"] and mutation == "repository"
        return {
            "commit": ("c" if changed else "a") * 40,
            "source_tree": ("d" if changed else "b") * 40,
            "dirty": False,
        }

    original_hashes = dict(identity["definition_hashes"])

    def definition_hashes(*_args, **_kwargs) -> dict[str, str]:
        hashes = dict(original_hashes)
        if state["changed"] and mutation == "definitions":
            hashes["panel_sha256"] = "f" * 64
        return hashes

    monkeypatch.setattr(module, "current_definition_hashes", definition_hashes)

    def physical_validator(root: Path, *_args, **_kwargs) -> dict[str, int]:
        if Path(root) == physical_root:
            events.append("physical")
            state["changed"] = True
        return {"outputs": 1}

    monkeypatch.setattr(
        module, "_validate_collection_dataset", physical_validator,
    )
    monkeypatch.setattr(module, "_validate_clone_stage", physical_validator)
    monkeypatch.setattr(module, "_validate_gate_stage", physical_validator)
    monkeypatch.setattr(module, "_validate_ppo_stage", physical_validator)
    monkeypatch.setattr(
        module, "_validate_development_stage", physical_validator,
    )

    def atomic(
        _destination,
        _definitions,
        *,
        stage_identity,
        build,
        validate,
    ) -> dict[str, Any]:
        del stage_identity, build
        summary = validate(physical_root)
        events.append(lifecycle)
        return {
            "state": "completed",
            "reused": lifecycle == "reuse",
            "summary": summary,
        }

    monkeypatch.setattr(module, "run_atomic_stage", atomic)
    kwargs = {
        "execution_identity_path": identity_path,
        "repository": tmp_path,
        "repository_identity_provider": repository_identity,
    }

    if command == "collect":
        import ml_lab.evaluation

        class FakeClient:
            def __init__(self, _command, *, environment):
                assert environment == "tactical-v2"
                self.contract = SimpleNamespace(
                    contract_hash="c" * 64,
                    encoding_hash="e" * 64,
                )

            def close(self) -> None:
                return None

        monkeypatch.setattr(ml_lab.evaluation, "DuelClient", FakeClient)
        invoke = lambda: module._collect_command(**kwargs)
    elif command == "train-bc":
        dataset = tmp_path / "dataset"
        dataset.mkdir()
        (dataset / "manifest.json").write_text(
            json.dumps({
                "code_revision": identity["commit"],
                "dirty": False,
            }),
            encoding="utf-8",
        )
        monkeypatch.setattr(module, "DATASET_PATH", dataset)
        invoke = lambda: module._train_bc_command(**kwargs)
    elif command == "evaluate-bc":
        invoke = lambda: module._evaluate_bc_command(**kwargs)
    elif command == "train-ppo":
        invoke = lambda: module._train_ppo_command(**kwargs)
    else:
        monkeypatch.setattr(
            module,
            "build_panel_development_candidates",
            lambda *_args, **_kwargs: [],
        )
        invoke = lambda: module._evaluate_dev_command(**kwargs)

    with pytest.raises(ValueError, match="execution identity"):
        invoke()
    assert events == ["physical"]



def test_freeze_command_passes_recorded_clean_revision_and_late_validator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Final freeze must seal the validated revision, not query an ad hoc one."""

    module = _subject()
    identity, kwargs = _command_identity_kwargs(tmp_path, module)
    captured: dict[str, Any] = {}

    def freeze(panel_dir: Path, **freeze_kwargs) -> dict[str, Any]:
        captured["panel_dir"] = Path(panel_dir)
        captured.update(freeze_kwargs)
        return {"state": "assigned"}

    monkeypatch.setattr(module, "freeze_final", freeze)
    incumbent = tmp_path / "incumbent"

    assert module._freeze_final_command(
        incumbent_panel=incumbent,
        **kwargs,
    ) == {"state": "assigned"}
    assert captured["panel_dir"] == module.PANEL_ROOT
    assert captured["incumbent_panel"] == incumbent
    assert captured["repository"] == tmp_path
    assert captured["revision"] == identity["commit"]
    assert captured["dirty"] is False
    assert callable(captured["final_identity_validator"])


def test_freeze_final_rejects_dirty_revision_before_seal(
    tmp_path: Path,
) -> None:
    """No direct caller may create a final seal whose revision is dirty."""

    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)

    with pytest.raises(ValueError, match="clean"):
        module.freeze_final(
            panel_dir,
            incumbent_panel=incumbent,
            dataset_dir=dataset_dir,
            revision="a" * 40,
            dirty=True,
        )

    assert not (panel_dir / "final-seal.json").exists()


def test_freeze_command_rechecks_identity_at_last_point_before_seal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repository drift during physical sealing must leave the bank unassigned."""

    module = _subject()
    panel_dir, dataset_dir, incumbent = _final_panel_fixture(tmp_path)
    identity = _execution_identity(module)
    identity_path = tmp_path / "execution-identity.json"
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    calls = 0

    def repository_identity(_repository: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        changed = calls > 1
        return {
            "commit": ("c" if changed else "a") * 40,
            "source_tree": ("d" if changed else "b") * 40,
            "dirty": False,
        }

    monkeypatch.setattr(module, "PANEL_ROOT", panel_dir)
    monkeypatch.setattr(module, "DATASET_PATH", dataset_dir)
    monkeypatch.setattr(
        module,
        "_repository_identity",
        lambda _repository: (identity["commit"], False),
    )

    with pytest.raises(ValueError, match="execution identity"):
        module._freeze_final_command(
            incumbent_panel=incumbent,
            execution_identity_path=identity_path,
            repository=tmp_path,
            repository_identity_provider=repository_identity,
        )

    assert calls == 2
    assert not (panel_dir / "final-seal.json").exists()
