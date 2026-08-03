import json
import logging
from pathlib import Path

import pytest

import run_annihilation_checkpoint_audit as runner


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _source_runs(tmp_path: Path) -> tuple[Path, Path]:
    scenario = b'{"environment":"tactical-v2","template_id":"standard-3v3"}\n'
    common_contract = {
        "environment": "tactical-v2",
        "version": "tactical-v2",
        "encoding_hash": "e" * 64,
        "observation_size": 761,
        "action_size": 379,
        "board": {"width": 13, "height": 9},
    }
    clone = tmp_path / "clone"
    ppo = tmp_path / "ppo"
    for root in (clone, ppo):
        root.mkdir()
        (root / "checkpoints").mkdir()
        (root / "scenario.json").write_bytes(scenario)
    (clone / "checkpoints" / "step_000000000.zip").write_bytes(b"clone")
    _write_json(clone / "run.json", {
        "schema_version": 1, "latest_checkpoint": "checkpoints/step_000000000.zip",
        "latest_checkpoint_step": 0, "model_seed": 227,
        "config": {"algorithm": "maskable_ppo", "environment": "tactical-v2", "seed": 227, "model_seed": 227, "behavioral_cloning": {"model_seed": 227}},
        "contract": {**common_contract, "contract_hash": "c" * 64},
        "scenario": {"path": "scenario.json"},
    })
    (ppo / "checkpoints" / "step_000014336.zip").write_bytes(b"ppo")
    _write_json(ppo / "run.json", {
        "schema_version": 1, "state": "stopped", "timesteps": 51_036,
        "latest_checkpoint": "checkpoints/step_000051200.zip", "latest_checkpoint_step": 51_036,
        "checkpoint_steps": [14_336], "model_seed": 227,
        "config": {"algorithm": "maskable_ppo", "environment": "tactical-v2", "seed": 227, "model_seed": 227, "actor_init_source": str(clone)},
        "contract": {**common_contract, "contract_hash": "d" * 64},
        "scenario": {"path": "scenario.json"},
    })
    return clone, ppo


@pytest.mark.parametrize(
    ("command", "required"),
    [
        ("prepare", ("--clone-run", "--ppo-run", "--output-root")),
        ("validate", ("--output-root",)),
        ("evaluate", ("--output-root",)),
        ("aggregate", ("--output-root",)),
        ("report", ("--output-root",)),
        ("all", ("--clone-run", "--ppo-run", "--output-root")),
    ],
)
def test_parser_requires_exact_command_inputs(
    command: str, required: tuple[str, ...]
) -> None:
    parser = runner.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args([command])

    values = {
        "--clone-run": Path("clone"),
        "--ppo-run": Path("ppo"),
        "--output-root": Path("output"),
    }
    arguments = [command]
    for option in required:
        arguments.extend((option, str(values[option])))
    parsed = parser.parse_args(arguments)

    assert parsed.command == command
    assert all(getattr(parsed, option.removeprefix("--").replace("-", "_")) for option in required)


def test_prepare_freezes_physical_definition_without_opening_server(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone, ppo = _source_runs(tmp_path)
    output_root = tmp_path / "audit"
    monkeypatch.setattr(
        runner.audit,
        "_runtime_contract",
        lambda _command: pytest.fail("prepare must not open the duel server"),
    )

    definition = runner.run_prepare(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
        output_root=output_root,
        logger=logging.getLogger("prepare-test"),
    )

    assert json.loads((output_root / "definition.json").read_text(encoding="utf-8")) == definition.to_dict()
    assert [candidate.actual_step for candidate in definition.candidates] == [0, 14_336, None, None]

    runner.run_prepare(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
        output_root=output_root,
        logger=logging.getLogger("prepare-test"),
    )
    (ppo / "checkpoints" / "step_000014336.zip").write_bytes(b"changed")
    with pytest.raises(ValueError, match="different physical definition"):
        runner.run_prepare(
            clone_run=clone,
            ppo_run=ppo,
            scratch_run=None,
            output_root=output_root,
            logger=logging.getLogger("prepare-test"),
        )


def test_validate_opens_one_runtime_and_reports_physical_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clone, ppo = _source_runs(tmp_path)
    (ppo / "checkpoints" / "step_000026624.zip").write_bytes(b"ppo-later")
    ppo_manifest = json.loads((ppo / "run.json").read_text(encoding="utf-8"))
    ppo_manifest["checkpoint_steps"] = [14_336, 26_624]
    _write_json(ppo / "run.json", ppo_manifest)
    output_root = tmp_path / "audit"
    logger = logging.getLogger("validate-test")
    runner.run_prepare(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
        output_root=output_root,
        logger=logger,
    )
    runtime = {
        "environment": "tactical-v2",
        "version": "tactical-v2",
        "contract_hash": "r" * 64,
        "encoding_hash": "e" * 64,
        "observation_size": 761,
        "action_size": 379,
        "board": {"width": 13, "height": 9},
    }
    runtime_calls: list[list[str]] = []
    validation_rows: list[int] = []
    real_validate = runner.audit._validate_runtime_contract

    def runtime_contract(command: list[str]) -> dict[str, object]:
        runtime_calls.append(command)
        return runtime

    def record_validation(value: object, rows: object) -> None:
        validation_rows.append(len(rows))
        real_validate(value, rows)

    monkeypatch.setattr(runner.audit, "_runtime_contract", runtime_contract)
    monkeypatch.setattr(runner.audit, "_validate_runtime_contract", record_validation)
    monkeypatch.setattr(
        runner.audit,
        "_repository_identity",
        lambda: {"repository": str(tmp_path), "commit": "a" * 40, "dirty": False},
    )

    with caplog.at_level(logging.INFO, logger="validate-test"):
        manifest = runner.run_validate(output_root=output_root, logger=logger)

    assert len(runtime_calls) == 1
    assert validation_rows == [1, 1, 1]
    assert manifest["runtime_contract"]["contract_hash"] == "r" * 64
    assert "pure-bc-seed-227: physical step 0" in caplog.text
    assert "bc-ppo-seed-227-step-000014336: physical step 14,336" in caplog.text
    assert "no physical 51,200 checkpoint exists" in caplog.text.lower()


def test_all_dispatches_in_order_and_stops_on_first_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = runner.build_parser().parse_args([
        "all", "--clone-run", "clone", "--ppo-run", "ppo",
        "--output-root", str(tmp_path), "--workers", "3",
    ])
    calls: list[str] = []

    def execute(command: str, _args: object, _logger: logging.Logger) -> object:
        calls.append(command)
        if command == "aggregate":
            raise RuntimeError("stop here")
        return None

    monkeypatch.setattr(runner, "_execute", execute)

    with pytest.raises(RuntimeError, match="stop here"):
        runner.dispatch(args, logging.getLogger("dispatch-test"))

    assert calls == ["prepare", "validate", "evaluate", "aggregate"]


def test_evaluate_aggregate_and_report_delegate_to_physical_domain_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    definition = object()
    monkeypatch.setattr(runner, "load_definition", lambda _root: definition)
    evaluate_calls: list[dict[str, object]] = []
    aggregate_calls: list[tuple[object, Path]] = []

    def evaluate(value: object, **kwargs: object) -> dict[str, object]:
        assert value is definition
        evaluate_calls.append(kwargs)
        kwargs["progress"]("candidate maps 10/100, reused 1, elapsed 00:01, eta 00:09")
        return {"maps": 10, "reused": 1}

    aggregate = {
        "audit_id": "annihilation-checkpoint-audit-v1",
        "definition_sha256": "d" * 64,
        "candidates": {},
        "trajectory": [],
        "anchors": [],
        "paired_successive_changes": [],
        "decision": {"clauses": {}, "recommended_next_step": "inconclusive"},
        "physical_evidence": {"maps": 0, "games": 0, "traces": 0, "replays": 0},
        "schedule": {"seed_start": 16_000_000, "maps": 100, "both_seats": True, "profile": "standard-3v3", "opponent": "random"},
        "repository_identity": {"repository": "repo", "commit": "a" * 40, "dirty": False},
        "runtime_contract": {"contract_hash": "r" * 64, "encoding_hash": "e" * 64},
        "source_contracts": [],
        "scenario": {"sha256": "s" * 64},
        "omitted_optional_candidates": [],
        "exploratory": True,
        "locked_panel_replacement": False,
    }

    def aggregate_audit(value: object, *, output_root: Path) -> dict[str, object]:
        aggregate_calls.append((value, output_root))
        return aggregate

    monkeypatch.setattr(runner.audit, "evaluate_audit", evaluate)
    monkeypatch.setattr(runner.audit, "aggregate_audit", aggregate_audit)
    logger = logging.getLogger("delegate-test")

    runner.run_evaluate(output_root=tmp_path, workers=3, logger=logger)
    assert evaluate_calls[0]["workers"] == 3
    assert evaluate_calls[0]["server_cmd"] == runner.server_command()
    assert callable(evaluate_calls[0]["progress"])

    assert runner.run_aggregate(output_root=tmp_path, logger=logger) is aggregate
    (tmp_path / "audit.json").write_text('{"untrusted":true}', encoding="utf-8")
    assert runner.run_report(output_root=tmp_path, logger=logger) == tmp_path.resolve() / "report.md"
    assert len(aggregate_calls) == 2
    assert "# Annihilation Physical Checkpoint Audit" in (tmp_path / "report.md").read_text(encoding="utf-8")


def _report_summary(*, wins: int, losses: int, draws: int) -> dict[str, object]:
    return {
        "counts": {"games": 200, "wins": wins, "losses": losses, "draws": draws},
        "rates": {"win": wins / 200, "loss": losses / 200, "draw": draws / 200},
        "confidence_intervals": {"win": [0.42, 0.56]},
        "seats": {},
        "win_rate_p0_minus_p1": -0.05,
        "draw_diagnostics": {
            "cycling": {"count": 30, "incidence": 0.15},
            "action_waste": {"count": 20, "incidence": 0.10},
            "primary_categories": {"cycling": 30},
        },
        "winning_games": {
            "round_count": {"mean": None, "median": None, "p90": None},
            "command_count": {"mean": None, "median": None, "p90": None},
        },
        "normalized_advantage": {
            "all_games": {"final": 0.3, "peak": 0.6},
            "draws": {"final": None, "peak": None},
        },
        "candidate_end_turns": {"total": 50, "wasted": 10, "wasted_ratio": 0.2},
        "end_turn_policy_diagnostics": {
            "available": False,
            "reason": "integer-action inference boundary does not expose action probabilities or ranks",
        },
    }


def test_render_report_covers_all_evidence_sections_and_nulls(
    tmp_path: Path,
) -> None:
    ppo = tmp_path / "ppo"
    ppo.mkdir()
    _write_json(ppo / "run.json", {"state": "stopped", "timesteps": 51_036, "latest_checkpoint_step": 51_036})
    candidate = {
        "candidate_id": "bc-ppo-seed-227-step-000014336",
        "family": "bc_ppo",
        "trajectory_order": 1,
        "actual_step": 14_336,
        "checkpoint_sha256": "1" * 64,
        "source_run": str(ppo),
        "source_run_manifest_sha256": "2" * 64,
        "source_scenario_sha256": "3" * 64,
        "source_contract_hash": "4" * 64,
        "source_encoding_hash": "5" * 64,
        "summary": _report_summary(wins=98, losses=22, draws=80),
    }
    anchor = {
        **candidate,
        "candidate_id": "random-anchor",
        "family": "control",
        "trajectory_order": None,
        "actual_step": None,
        "checkpoint_sha256": None,
        "source_run": None,
    }
    aggregate = {
        "audit_id": "annihilation-checkpoint-audit-v1",
        "exploratory": True,
        "locked_panel_replacement": False,
        "schedule": {"seed_start": 16_000_000, "maps": 100, "both_seats": True, "profile": "standard-3v3", "opponent": "random"},
        "definition_sha256": "6" * 64,
        "repository_identity": {"repository": "repo", "commit": "a" * 40, "dirty": False},
        "scenario": {"sha256": "7" * 64},
        "source_contracts": [{"source_run": str(ppo), "run_manifest_sha256": "2" * 64, "scenario_sha256": "3" * 64, "contract": {"contract_hash": "4" * 64, "encoding_hash": "5" * 64}}],
        "runtime_contract": {"contract_hash": "8" * 64, "encoding_hash": "5" * 64, "observation_size": 761, "action_size": 379},
        "omitted_optional_candidates": [{"family": "scratch_ppo", "reason": "no physical compatible run supplied"}],
        "candidates": {candidate["candidate_id"]: candidate, "random-anchor": anchor},
        "trajectory": [candidate],
        "paired_successive_changes": [{"earlier_candidate_id": "pure-bc-seed-227", "later_candidate_id": candidate["candidate_id"], "transition_table": {"win": {"win": 80, "draw": 10, "loss": 10}, "draw": {"win": 18, "draw": 50, "loss": 2}, "loss": {"win": 0, "draw": 10, "loss": 20}}, "net_win_change": 8, "absolute_win_rate_change": 0.04, "exact_sign_test_p_value": 0.03}],
        "anchors": ["random-anchor", "bounded-search-anchor"],
        "decision": {"clauses": {"all_ppo_below_half": True}, "recommended_next_step": "proceed_to_dagger"},
        "physical_evidence": {"maps": 600, "games": 1200, "traces": 1200, "replays": 1200},
    }

    report = runner.render_report(aggregate)

    for heading in (
        "## 1. Status", "## 2. Schedule and identities", "## 3. Candidate metrics",
        "## 4. Successive paired transitions", "## 5. Random and bounded-search anchors",
        "## 6. Decision", "## 7. EndTurn policy diagnostics",
        "## 8. Optional scratch reference", "## 9. Physical artifacts",
    ):
        assert heading in report
    assert "W/L/D 98/22/80" in report
    assert "Wilson win interval [0.420, 0.560]" in report
    assert "cycling 30 (15.0%)" in report
    assert "seat win-rate delta -0.050" in report
    assert "win rounds mean unavailable" in report
    assert "draw final advantage unavailable" in report
    assert "EndTurn rank/probability is unavailable" in report
    assert "51,036 was a stopped in-memory training count, not an evaluated checkpoint" in report
    assert "does not replace the locked panel" in report
    assert "None" not in report

    scratch_checkpoint = tmp_path / "scratch-step-000051200.zip"
    scratch_checkpoint.write_bytes(b"physical-scratch")
    scratch = {
        **candidate,
        "candidate_id": "scratch-ppo-seed-227-step-000051200",
        "family": "scratch_ppo",
        "trajectory_order": None,
        "actual_step": 51_200,
        "checkpoint_path": str(scratch_checkpoint),
        "checkpoint_sha256": "9" * 64,
        "source_run": str(tmp_path / "scratch"),
    }
    aggregate["candidates"][scratch["candidate_id"]] = scratch
    assert (
        "51,036 was a stopped in-memory training count, not an evaluated checkpoint"
        in runner.render_report(aggregate)
    )


def test_main_logs_to_stdout_and_audit_file_and_reraises_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(_args: object, logger: logging.Logger) -> object:
        logger.info("candidate=clone maps=10/100 reused=2 elapsed=00:01 ETA=00:09")
        raise RuntimeError("physical evidence failed")

    monkeypatch.setattr(runner, "dispatch", fail)

    with pytest.raises(RuntimeError, match="physical evidence failed"):
        runner.main(["validate", "--output-root", str(tmp_path)])

    stdout = capsys.readouterr().out
    log = (tmp_path / "audit.log").read_text(encoding="utf-8")
    for text in (
        "command=validate", "candidate=clone", "reused=2", "elapsed=00:01",
        "ETA=00:09", "checkpoint audit failed", "physical evidence failed",
    ):
        assert text in stdout
        assert text in log
    assert log[:4].isdigit()


def test_protocol_documents_research_contract_and_reserved_seed_banks() -> None:
    protocol = (
        runner.ROOT
        / "python"
        / "panels"
        / "annihilation-checkpoint-audit-v1"
        / "PROTOCOL.md"
    ).read_text(encoding="utf-8")

    for topic in (
        "Research question", "Candidates", "Schedule", "Metrics",
        "Decision precedence", "Seed isolation", "Commands", "Output tree",
        "Recovery", "Full contract and encoding compatibility",
    ):
        assert topic in protocol
    assert "17m" in protocol and "untouched" in protocol
    assert "18m" in protocol and "19m" in protocol and "20m" in protocol
    assert "reserved" in protocol and "not consumed" in protocol
    for command in ("prepare", "validate", "evaluate", "aggregate", "report", "all"):
        assert f" {command} " in protocol


def test_definition_reuse_requires_canonical_bytes_and_loader_rejects_extra_fields(
    tmp_path: Path,
) -> None:
    clone, ppo = _source_runs(tmp_path)
    output_root = tmp_path / "audit"
    logger = logging.getLogger("definition-shape-test")
    runner.run_prepare(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
        output_root=output_root,
        logger=logger,
    )
    definition_path = output_root / "definition.json"
    payload = json.loads(definition_path.read_text(encoding="utf-8"))
    definition_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="serialized bytes"):
        runner.run_prepare(
            clone_run=clone,
            ppo_run=ppo,
            scratch_run=None,
            output_root=output_root,
            logger=logger,
        )

    payload["unexpected"] = True
    _write_json(definition_path, payload)
    with pytest.raises(ValueError, match="shape"):
        runner.load_definition(output_root)


def test_validate_rejects_checkpoint_mutated_after_prepare_before_runtime_or_games(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clone, ppo = _source_runs(tmp_path)
    output_root = tmp_path / "audit"
    logger = logging.getLogger("mutated-checkpoint-test")
    definition = runner.run_prepare(
        clone_run=clone,
        ppo_run=ppo,
        scratch_run=None,
        output_root=output_root,
        logger=logger,
    )
    Path(definition.candidates[1].checkpoint_path).write_bytes(b"mutated-after-prepare")
    monkeypatch.setattr(
        runner.audit,
        "_runtime_contract",
        lambda _command: pytest.fail("runtime/game path must not be reached"),
    )
    monkeypatch.setattr(
        runner.audit,
        "evaluate_audit",
        lambda *_args, **_kwargs: pytest.fail("evaluator must not be reached"),
    )

    with pytest.raises(ValueError, match="checkpoint bytes changed"):
        runner.run_validate(output_root=output_root, logger=logger)

    assert not (output_root / "manifest.json").exists()


def test_prepare_definition_publication_is_atomic_no_clobber_under_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clone, ppo = _source_runs(tmp_path)
    logger = logging.getLogger("definition-race-test")

    def publish_identical(temp: object, destination: object) -> None:
        Path(destination).write_bytes(Path(temp).read_bytes())
        raise FileExistsError

    monkeypatch.setattr(runner.os, "link", publish_identical)
    identical_root = tmp_path / "identical"
    with caplog.at_level(logging.INFO, logger="definition-race-test"):
        runner.run_prepare(
            clone_run=clone,
            ppo_run=ppo,
            scratch_run=None,
            output_root=identical_root,
            logger=logger,
        )
    assert "reused byte/hash-equivalent physical definition" in caplog.text
    assert not list(identical_root.glob(".definition.json.*.tmp"))

    caplog.clear()

    def publish_different(_temp: object, destination: object) -> None:
        Path(destination).write_bytes(b"{}\n")
        raise FileExistsError

    monkeypatch.setattr(runner.os, "link", publish_different)
    different_root = tmp_path / "different"
    with caplog.at_level(logging.INFO, logger="definition-race-test"):
        with pytest.raises(ValueError, match="different serialized bytes"):
            runner.run_prepare(
                clone_run=clone,
                ppo_run=ppo,
                scratch_run=None,
                output_root=different_root,
                logger=logger,
            )
    assert (different_root / "definition.json").read_bytes() == b"{}\n"
    assert "reused byte/hash-equivalent physical definition" not in caplog.text
    assert not list(different_root.glob(".definition.json.*.tmp"))
