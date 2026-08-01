from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
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


def _subject():
    specification = importlib.util.find_spec(MODULE_NAME)
    assert specification is not None, "Task 8 panel orchestrator is missing"
    return importlib.import_module(MODULE_NAME)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    }
    (run / "bc.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_seed": seed,
                "dataset_manifest_sha256": dataset_hash,
                "config": config,
                "best_epoch": 1,
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

    def fake_atomic(destination, hashes, *, build, validate):
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
        module._collect_command()
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
    original = module.SCENARIO_PATH.read_bytes()
    captured: list[bytes] = []

    class FakeEnv:
        def __init__(self, command, **kwargs):
            captured.append(Path(kwargs["scenario_path"]).read_bytes())
            self.contract = SimpleNamespace(contract_hash="c" * 64, encoding_hash="e" * 64)
            self.spaces_info = {}
        def close(self) -> None:
            return None

    def fake_atomic(destination, hashes, *, build, validate):
        changed = _json(module.SCENARIO_PATH)
        changed["reward"]["points_weight"] = 0.25
        module.SCENARIO_PATH.write_text(json.dumps(changed), encoding="utf-8")
        staging = tmp_path / "clones-stage"
        staging.mkdir()
        build(staging)
        return {"state": "completed"}

    import hexwars_gym
    import ml_lab.imitation
    monkeypatch.setattr(hexwars_gym, "HexWarsEnv", FakeEnv)
    monkeypatch.setattr(ml_lab.imitation, "load_imitation_dataset", lambda *args, **kwargs: object())
    monkeypatch.setattr(module, "train_clone_runs", lambda **kwargs: [])
    monkeypatch.setattr(module, "run_atomic_stage", fake_atomic)
    try:
        module._train_bc_command()
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
) -> None:
    """Training before the clone gate passes would spend compute outside the protocol."""

    module = _subject()
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
        module._train_ppo_command()


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
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        row["checkpoint_sha256"] = digest
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
        return Path(kwargs["runs_root"]) / config.run_name

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
    candidate = module.DevelopmentCandidate(
        "bc_ppo", 211, 12_800, 16_384, "snapshot-controller"
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
            "candidate": {"step": 16_384},
            "opponent": {"name": "random"},
            "seed_start": map_seed,
            "seeds": [map_seed],
            "reciprocal": True,
            "games": 2,
            "schedule": {
                "start_profile": "standard-3v3",
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
    )

    assert len(calls) == 100
    assert all(
        call["p0"] == "snapshot-controller"
        and call["p1"] == "random"
        and call["games"] == 1
        and call["both_seats"] is True
        and call["capture_trace"] is True
        and call["start_profile"] == "standard-3v3"
        for call in calls
    )
    rows = result["candidates"][0]["matches"]
    assert len(rows) == 200
    assert [(row["map_seed"], row["candidate_seat"]) for row in rows] == [
        (seed, seat)
        for seed in range(16_000_000, 16_000_100)
        for seat in (0, 1)
    ]
    assert all(row["actual_step"] == 16_384 for row in rows)
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
        for step in (8_192, 16_384, 32_768, 57_344):
            (checkpoints / f"step_{step:09d}.zip").write_bytes(b"checkpoint")
        if run.condition == "bc_ppo":
            (run_dir / "initialization.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "kind": "actor_only",
                        "source_run": run.config.actor_init_source,
                    }
                ),
                encoding="utf-8",
            )

    budgets = module._ppo_budget_map(root, matrix, scenario)
    assert {
        name: [(item.nominal_step, item.actual_step) for item in items]
        for name, items in budgets.items()
    } == {
        run.config.run_name: [
            (12_800, 16_384),
            (25_600, 32_768),
            (51_200, 57_344),
        ]
        for run in matrix
    }

    initialized = root / "bc-ppo-seed-211" / "initialization.json"
    initialized.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "kind": "actor_only",
                "source_run": str(tmp_path / "clones" / "seed-223"),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="initialization source"):
        module._ppo_budget_map(root, matrix, scenario)
