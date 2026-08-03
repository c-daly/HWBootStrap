"""Run the exploratory annihilation physical-checkpoint audit."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from ml_lab import checkpoint_audit as audit
from ml_lab.io import atomic_write_json, atomic_write_text

ROOT = Path(__file__).resolve().parents[1]
SCENARIO = ROOT / "python" / "config" / "annihilation-imitation-v1.json"
DEFAULT_OUTPUT = ROOT / "python" / "evidence" / "annihilation-checkpoint-audit-v1"


def server_command() -> list[str]:
    return [
        "dotnet",
        str(
            ROOT
            / "engine"
            / "HexWars.GymServer"
            / "bin"
            / "Debug"
            / "net8.0"
            / "HexWars.GymServer.dll"
        ),
        "--scenario-file",
        str(SCENARIO),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare")
    prepare.add_argument("--clone-run", type=Path, required=True)
    prepare.add_argument("--ppo-run", type=Path, required=True)
    prepare.add_argument("--scratch-run", type=Path)
    prepare.add_argument("--output-root", type=Path, required=True)

    for name in ("validate", "evaluate", "aggregate", "report"):
        command = commands.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        if name == "evaluate":
            command.add_argument("--workers", type=int, default=1)

    all_command = commands.add_parser("all")
    all_command.add_argument("--clone-run", type=Path, required=True)
    all_command.add_argument("--ppo-run", type=Path, required=True)
    all_command.add_argument("--scratch-run", type=Path)
    all_command.add_argument("--output-root", type=Path, required=True)
    all_command.add_argument("--workers", type=int, default=1)
    return parser




def _serialized_definition(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def run_prepare(
    *,
    clone_run: Path,
    ppo_run: Path,
    scratch_run: Path | None,
    output_root: Path,
    logger: logging.Logger,
) -> audit.AuditDefinition:
    """Discover and freeze only physically present audit candidates."""
    definition = audit.build_audit_definition(
        clone_run=clone_run,
        ppo_run=ppo_run,
        scratch_run=scratch_run,
    )
    payload = definition.to_dict()
    expected_bytes = _serialized_definition(payload)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = root / "definition.json"
    if path.exists():
        try:
            existing_bytes = path.read_bytes()
            existing = json.loads(existing_bytes.decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"existing physical definition is unreadable: {path}") from error
        if existing != payload:
            raise ValueError("existing audit has a different physical definition")
        logger.info("reused byte/hash-equivalent physical definition: %s", path)
        if existing_bytes != expected_bytes:
            raise ValueError("existing physical definition has different serialized bytes")
    else:
        atomic_write_json(path, payload)
        logger.info("wrote physical definition: %s", path)
    return definition


def load_definition(output_root: Path) -> audit.AuditDefinition:
    """Load the frozen definition using the domain dataclasses."""
    path = Path(output_root).resolve() / "definition.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("definition must be an object")
        schedule_payload = payload["schedule"]
        candidate_payloads = payload["candidates"]
        omitted_payloads = payload["omitted_optional_candidates"]
        if not isinstance(schedule_payload, dict) or not isinstance(candidate_payloads, list):
            raise TypeError("definition members have invalid shapes")
        if not isinstance(omitted_payloads, list):
            raise TypeError("omitted candidates must be a list")
        definition = audit.AuditDefinition(
            schema_version=payload["schema_version"],
            audit_id=payload["audit_id"],
            exploratory=payload["exploratory"],
            locked_panel_replacement=payload["locked_panel_replacement"],
            schedule=audit.AuditSchedule(**schedule_payload),
            candidates=tuple(audit.AuditCandidate(**row) for row in candidate_payloads),
            omitted_optional_candidates=tuple(dict(row) for row in omitted_payloads),
        )
        if definition.to_dict() != payload:
            raise ValueError("frozen audit definition shape does not round-trip")
        return definition
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"frozen audit definition is unreadable: {path}") from error


def run_validate(
    *, output_root: Path, logger: logging.Logger
) -> dict[str, object]:
    """Validate frozen physical sources against one evaluation runtime."""
    definition = load_definition(output_root)
    root = Path(output_root).resolve(strict=True)
    scenario_bytes, scenario_sha256, source_contracts = audit._source_material(definition)
    runtime_contract = dict(audit._runtime_contract(server_command()))
    contracts_by_run = {row["source_run"]: row for row in source_contracts}
    for candidate in definition.candidates:
        if candidate.source_run is None:
            continue
        source_contract = contracts_by_run.get(candidate.source_run)
        if source_contract is None:
            raise ValueError(
                f"{candidate.candidate_id} has no validated physical source contract"
            )
        audit._validate_runtime_contract(runtime_contract, [source_contract])
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        manifest = audit._require_existing_manifest(
            root,
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
        )
        logger.info("reused validated audit manifest: %s", manifest_path)
    else:
        manifest = audit._initial_manifest(
            definition,
            scenario_bytes=scenario_bytes,
            scenario_sha256=scenario_sha256,
            source_contracts=source_contracts,
            runtime_contract=runtime_contract,
        )
        atomic_write_json(manifest_path, manifest)
        logger.info("wrote validated audit manifest: %s", manifest_path)
    for candidate in definition.candidates:
        if candidate.actual_step is None:
            logger.info("%s: scripted anchor", candidate.candidate_id)
        else:
            logger.info("%s: physical step %s", candidate.candidate_id, f"{candidate.actual_step:,}")
    if all(candidate.actual_step != 51_200 for candidate in definition.candidates):
        logger.info("No physical 51,200 checkpoint exists; it is not an audit candidate.")
    return dict(manifest)


def run_evaluate(
    *, output_root: Path, workers: int, logger: logging.Logger
) -> dict[str, object]:
    definition = load_definition(output_root)
    result = audit.evaluate_audit(
        definition,
        output_root=Path(output_root).resolve(),
        server_cmd=server_command(),
        workers=workers,
        progress=logger.info,
    )
    logger.info("evaluation artifacts: %s", Path(output_root).resolve())
    return dict(result)


def run_aggregate(*, output_root: Path, logger: logging.Logger) -> dict[str, object]:
    aggregate = audit.aggregate_audit(
        load_definition(output_root), output_root=Path(output_root).resolve()
    )
    logger.info("aggregate artifact: %s", Path(output_root).resolve() / "audit.json")
    return aggregate


def run_report(*, output_root: Path, logger: logging.Logger) -> Path:
    aggregate = run_aggregate(output_root=output_root, logger=logger)
    path = Path(output_root).resolve() / "report.md"
    atomic_write_text(path, render_report(aggregate))
    logger.info("report artifact: %s", path)
    return path


def _execute(command: str, args: argparse.Namespace, logger: logging.Logger) -> object:
    if command == "prepare":
        return run_prepare(
            clone_run=args.clone_run,
            ppo_run=args.ppo_run,
            scratch_run=args.scratch_run,
            output_root=args.output_root,
            logger=logger,
        )
    if command == "validate":
        return run_validate(output_root=args.output_root, logger=logger)
    if command == "evaluate":
        return run_evaluate(output_root=args.output_root, workers=args.workers, logger=logger)
    if command == "aggregate":
        return run_aggregate(output_root=args.output_root, logger=logger)
    if command == "report":
        return run_report(output_root=args.output_root, logger=logger)
    raise ValueError(f"unsupported checkpoint audit command: {command}")


def dispatch(args: argparse.Namespace, logger: logging.Logger) -> object:
    if args.command == "all":
        result: object = None
        for command in ("prepare", "validate", "evaluate", "aggregate", "report"):
            result = _execute(command, args, logger)
        return result
    return _execute(args.command, args, logger)


def _format_metric(value: object, *, percent: bool = False) -> str:
    if value is None:
        return "unavailable"
    number = float(value)
    return f"{number * 100:.1f}%" if percent else f"{number:.3f}"


def _stopped_count_explanation(candidates: dict[str, object]) -> str | None:
    if any(
        isinstance(row, dict) and row.get("actual_step") == 51_200
        for row in candidates.values()
    ):
        return None
    sources = {
        row.get("source_run")
        for row in candidates.values()
        if isinstance(row, dict) and row.get("family") == "bc_ppo"
    }
    for raw_source in sorted(source for source in sources if isinstance(source, str)):
        path = Path(raw_source) / "run.json"
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"PPO source manifest is unreadable while reporting: {path}") from error
        stopped_count = manifest.get("timesteps", manifest.get("latest_checkpoint_step"))
        if manifest.get("state") == "stopped" and stopped_count == 51_036:
            return "51,036 was a stopped in-memory training count, not an evaluated checkpoint"
    return None


def render_report(aggregate: dict[str, object]) -> str:
    candidates = aggregate["candidates"]
    if not isinstance(candidates, dict):
        raise ValueError("aggregate candidates must be an object")
    schedule = aggregate["schedule"]
    repository = aggregate["repository_identity"]
    scenario = aggregate["scenario"]
    runtime = aggregate["runtime_contract"]
    lines = [
        "# Annihilation Physical Checkpoint Audit",
        "",
        "## 1. Status",
        "",
        "This is exploratory checkpoint-trajectory evidence. It does not replace the locked panel or authorize model promotion.",
        "",
        "## 2. Schedule and identities",
        "",
        f"- Schedule: {schedule['maps']} maps from seed {schedule['seed_start']:,}, both seats={str(schedule['both_seats']).lower()}, profile `{schedule['profile']}`, opponent `{schedule['opponent']}`.",
        f"- Definition SHA-256: `{aggregate['definition_sha256']}`.",
        f"- Repository: `{repository['repository']}` at `{repository['commit']}`; dirty={str(repository['dirty']).lower()}.",
        f"- Scenario SHA-256: `{scenario['sha256']}`.",
        f"- Runtime contract/encoding: `{runtime.get('contract_hash', 'unavailable')}` / `{runtime['encoding_hash']}`; observation/action {runtime.get('observation_size', 'unavailable')}/{runtime.get('action_size', 'unavailable')}.",
    ]
    for source in aggregate.get("source_contracts", []):
        contract = source["contract"]
        lines.append(
            f"- Source `{source['source_run']}`: manifest `{source['run_manifest_sha256']}`, scenario `{source['scenario_sha256']}`, full contract `{contract['contract_hash']}`, encoding `{contract['encoding_hash']}`."
        )

    lines.extend(["", "## 3. Candidate metrics", ""])
    for candidate_id, candidate in candidates.items():
        summary = candidate["summary"]
        counts = summary["counts"]
        interval = summary["confidence_intervals"]["win"]
        draws = summary["draw_diagnostics"]
        winning = summary["winning_games"]
        advantage = summary["normalized_advantage"]
        end_turns = summary["candidate_end_turns"]
        lines.extend(
            [
                f"### {candidate_id}",
                "",
                f"- W/L/D {counts['wins']}/{counts['losses']}/{counts['draws']}; Wilson win interval [{float(interval[0]):.3f}, {float(interval[1]):.3f}].",
                f"- Draw pathology: cycling {draws['cycling']['count']} ({_format_metric(draws['cycling']['incidence'], percent=True)}); action waste {draws['action_waste']['count']} ({_format_metric(draws['action_waste']['incidence'], percent=True)}).",
                f"- Winning speed: win rounds mean {_format_metric(winning['round_count']['mean'])}, median {_format_metric(winning['round_count']['median'])}, p90 {_format_metric(winning['round_count']['p90'])}; decisions mean {_format_metric(winning['command_count']['mean'])}, median {_format_metric(winning['command_count']['median'])}, p90 {_format_metric(winning['command_count']['p90'])}.",
                f"- Advantage: all-game final {_format_metric(advantage['all_games']['final'])}, peak {_format_metric(advantage['all_games']['peak'])}; draw final advantage {_format_metric(advantage['draws']['final'])}, draw peak advantage {_format_metric(advantage['draws']['peak'])}.",
                f"- seat win-rate delta {_format_metric(summary['win_rate_p0_minus_p1'])}; EndTurn waste {end_turns['wasted']}/{end_turns['total']} ({_format_metric(end_turns['wasted_ratio'], percent=True)}).",
            ]
        )

    lines.extend(["", "## 4. Successive paired transitions", ""])
    transitions = aggregate.get("paired_successive_changes", [])
    if not transitions:
        lines.append("No successive learned-candidate pair is available.")
    for change in transitions:
        table = change["transition_table"]
        lines.append(
            f"- `{change['earlier_candidate_id']}` -> `{change['later_candidate_id']}`: earlier win row {table['win']['win']}/{table['win']['draw']}/{table['win']['loss']}, draw row {table['draw']['win']}/{table['draw']['draw']}/{table['draw']['loss']}, loss row {table['loss']['win']}/{table['loss']['draw']}/{table['loss']['loss']}; net wins {change['net_win_change']:+d}, absolute win-rate change {_format_metric(change['absolute_win_rate_change'], percent=True)}, exact sign-test p={_format_metric(change['exact_sign_test_p_value'])}."
        )

    lines.extend(["", "## 5. Random and bounded-search anchors", ""])
    for anchor_id in aggregate.get("anchors", []):
        anchor = candidates.get(anchor_id)
        if anchor is None:
            lines.append(f"- `{anchor_id}`: listed by the frozen definition; no aggregate row supplied.")
        else:
            counts = anchor["summary"]["counts"]
            lines.append(f"- `{anchor_id}`: W/L/D {counts['wins']}/{counts['losses']}/{counts['draws']}.")

    decision = aggregate["decision"]
    lines.extend(["", "## 6. Decision", ""])
    for clause, value in sorted(decision["clauses"].items()):
        lines.append(f"- `{clause}`: `{str(value).lower()}`.")
    lines.append(f"- Recommended next experiment: `{decision['recommended_next_step']}` (deterministic precedence).")

    lines.extend(["", "## 7. EndTurn policy diagnostics", ""])
    reasons = {
        candidate["summary"]["end_turn_policy_diagnostics"]["reason"]
        for candidate in candidates.values()
    }
    lines.append("EndTurn rank/probability is unavailable: " + "; ".join(sorted(reasons)) + ".")

    lines.extend(["", "## 8. Optional scratch reference", ""])
    omitted = aggregate.get("omitted_optional_candidates", [])
    if omitted:
        for row in omitted:
            lines.append(f"- `{row['family']}` omitted: {row['reason']}.")
    else:
        lines.append("No optional scratch reference was omitted.")

    evidence = aggregate["physical_evidence"]
    lines.extend(
        [
            "",
            "## 9. Physical artifacts",
            "",
            f"- Counts: maps {evidence['maps']}, games {evidence['games']}, traces {evidence['traces']}, replays {evidence['replays']}.",
        ]
    )
    for candidate_id, candidate in candidates.items():
        checkpoint = candidate.get("checkpoint_sha256") or "scripted-controller"
        source_manifest = candidate.get("source_run_manifest_sha256") or "scripted-controller"
        source_scenario = candidate.get("source_scenario_sha256") or "scripted-controller"
        lines.append(f"- `{candidate_id}`: checkpoint `{checkpoint}`, source manifest `{source_manifest}`, source scenario `{source_scenario}`.")
    stopped = _stopped_count_explanation(candidates)
    if stopped is not None:
        lines.extend(["", stopped + "."])
    return "\n".join(lines) + "\n"


def configure_logging(output_root: Path) -> logging.Logger:
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hexwars.checkpoint_audit")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%dT%H:%M:%S"
    )
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(root / "audit.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logger = configure_logging(args.output_root)
    started = time.monotonic()
    logger.info("command=%s output_root=%s", args.command, Path(args.output_root).resolve())
    try:
        dispatch(args, logger)
    except Exception:
        logger.exception(
            "checkpoint audit failed command=%s elapsed=%.1fs",
            args.command,
            time.monotonic() - started,
        )
        raise
    logger.info(
        "command=%s completed elapsed=%.1fs",
        args.command,
        time.monotonic() - started,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
