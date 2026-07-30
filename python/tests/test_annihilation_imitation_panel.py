from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

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
    dataset = object()
    contract = SimpleNamespace(contract_hash="c" * 64, encoding_hash="e" * 64)
    calls: list[dict[str, Any]] = []

    def trainer(**kwargs):
        calls.append(kwargs)
        run_dir = kwargs["run_dir"]
        run_dir.mkdir(parents=True)
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "state": "completed",
                    "model_seed": kwargs["config"].model_seed,
                    "config": {"model_seed": kwargs["config"].model_seed},
                    "contract": {
                        "contract_hash": contract.contract_hash,
                        "encoding_hash": contract.encoding_hash,
                    },
                }
            ),
            encoding="utf-8",
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


def _write_clone_runs(root: Path, module, *, changed_hash: bool = False) -> list[Path]:
    hashes = module.current_definition_hashes()
    runs = []
    for seed in (211, 223, 227):
        run = root / f"seed-{seed}"
        run.mkdir(parents=True)
        contract = {"contract_hash": "c" * 64, "encoding_hash": "e" * 64}
        (run / "run.json").write_text(
            json.dumps({"state": "completed", "model_seed": seed, "config": {"model_seed": seed}, "contract": contract}),
            encoding="utf-8",
        )
        provenance_hashes = dict(hashes)
        if changed_hash and seed == 211:
            provenance_hashes["panel_sha256"] = "f" * 64
        (run / "panel-provenance.json").write_text(
            json.dumps({"model_seed": seed, "sampler_seed": seed, "definition_hashes": provenance_hashes}),
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
    standard = _json(tmp_path / "evaluation" / "standard-gate-scenario.json")
    assert standard["tactical_v2"]["start_distribution"][0]["basis_points"] == 10_000
    assert all(item["basis_points"] == 0 for item in standard["tactical_v2"]["start_distribution"][1:])
    assert all(
        Path(match[field]).is_file()
        for clone in result["clones"]
        for match in clone["matches"]
        if match["outcome"] in {"draw", "loss"}
        for field in ("trace_path", "replay_path")
    )


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
