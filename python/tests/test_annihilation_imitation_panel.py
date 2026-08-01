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


def test_parser_exposes_only_restart_safe_task_8_commands() -> None:
    """Renaming or adding later-phase commands would expand Task 8's protocol surface."""

    module = _subject()
    parser = module.build_parser()
    for command in ("validate", "collect", "train-bc", "evaluate-bc"):
        assert parser.parse_args([command]).command == command
    with pytest.raises(SystemExit):
        parser.parse_args(["train-ppo"])



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
