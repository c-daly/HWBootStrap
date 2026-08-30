"""Command-line surface for the HexWars ML Lab."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import traceback
from contextlib import ExitStack, contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence, TextIO

from .benchmark import benchmark_gymserver
from .contracts import RunConfig, request_stop
from .controllers import ControllerResolver, ControllerSpec, normalize_controller_spec
from .doctor import doctor_environment
from .evaluation import DEFAULT_HELD_OUT_SEED, evaluate_controllers, publish_candidate
from .io import read_json
from .scenarios import (
    ResolvedScenario,
    legacy_default_scenario,
    resolve_scenario,
)
from .training import run_training
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = PROJECT_ROOT / "engine" / "HexWars.GymServer" / "bin" / "Release" / "net8.0" / "HexWars.GymServer.dll"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "python" / "runs"
TERMINAL_STATES = frozenset({"stopped", "completed", "failed"})
JSON_SCHEMA_VERSION = 1


def controller_config(raw: str | dict[str, Any] | ControllerSpec) -> dict[str, Any]:
    spec = normalize_controller_spec(raw)
    value: dict[str, Any] = {"kind": spec.kind}
    if spec.name is not None:
        value["name"] = spec.name
    if spec.path is not None:
        value["path"] = str(spec.path)
    if spec.algorithm is not None:
        value["algorithm"] = spec.algorithm
    if spec.mode != "fixed":
        value["mode"] = spec.mode
    return value


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--server", default=str(DEFAULT_SERVER))


def _add_json_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--json", action="store_true", help="print one stable JSON object"
    )


def _add_no_console_output_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--no-console-output",
        action="store_true",
        help="run without writing to stdout or stderr",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hexwars_ml.py")
    subcommands = parser.add_subparsers(dest="command", required=True)

    doctor = subcommands.add_parser("doctor", help="check headless ML dependencies")
    doctor.add_argument("--tracker", action="append", default=[])
    doctor.add_argument(
        "--environment",
        choices=["tactical-v1", "tactical-v2", "adaptive-v1"],
        default="tactical-v1",
    )
    _add_runtime_arguments(doctor)
    _add_json_argument(doctor)

    train = subcommands.add_parser("train", help="run headless SB3 training")
    train.add_argument("--run", required=True)
    train.add_argument(
        "--environment",
        choices=["tactical-v1", "tactical-v2", "adaptive-v1"],
        default=None,
    )
    train.add_argument(
        "--algorithm", choices=["maskable_ppo", "masked_dqn"], default="maskable_ppo"
    )
    train.add_argument("--opponent", default="greedy")
    train.add_argument("--timesteps", type=int, default=200_000)
    train.add_argument("--checkpoint-every", type=int, default=25_000)
    train.add_argument("--workers", type=int, default=1)
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--device", default="auto")
    train.add_argument(
        "--learner-seat", choices=["alternating", "0", "1"], default="alternating"
    )
    train.add_argument("--resume")
    train.add_argument("--actor-init")
    train.add_argument("--learning-rate", type=float)
    train.add_argument("--ppo-epochs", type=int)
    train.add_argument("--target-kl", type=float)
    train.add_argument("--episode-seed-base", type=int)
    scenario = train.add_mutually_exclusive_group()
    scenario.add_argument("--scenario-file", type=Path)
    scenario.add_argument("--template")
    train.add_argument(
        "--tracker",
        action="append",
        help="local, tensorboard, wandb, or custom=module:function",
    )
    train.add_argument("--wandb-project")
    train.add_argument("--wandb-entity")
    train.add_argument("--wandb-mode")
    train.add_argument("--wandb-group")
    train.add_argument("--wandb-tag", action="append", default=[])
    train.add_argument("--wandb-upload-artifacts", action="store_true")
    _add_runtime_arguments(train)
    _add_no_console_output_argument(train)
    _add_json_argument(train)

    structured = subcommands.add_parser(
        "train-structured",
        help="start a weight-initialized tactical-v3 DAgger continuation",
    )
    structured.add_argument("--run", required=True)
    structured.add_argument("--source-run", type=Path, required=True)
    structured.add_argument("--scenario-file", type=Path, required=True)
    structured.add_argument("--opponent", default="greedy")
    structured.add_argument("--train-labels", type=int, required=True)
    structured.add_argument("--validation-labels", type=int, required=True)
    structured.add_argument("--seed", type=int, default=227)
    structured.add_argument("--device", default="auto")
    structured.add_argument(
        "--learner-seat", choices=["alternating", "0", "1"], default="alternating"
    )
    structured.add_argument(
        "--tracker",
        action="append",
        help="local, tensorboard, wandb, or custom=module:function",
    )
    structured.add_argument("--wandb-project")
    structured.add_argument("--wandb-entity")
    structured.add_argument("--wandb-mode")
    structured.add_argument("--wandb-group")
    structured.add_argument("--wandb-tag", action="append", default=[])
    structured.add_argument("--wandb-upload-artifacts", action="store_true")
    _add_runtime_arguments(structured)
    _add_no_console_output_argument(structured)
    _add_json_argument(structured)

    structured_preflight = subcommands.add_parser(
        "preflight-structured",
        help="validate a tactical-v3 continuation without creating a run",
    )
    structured_preflight.add_argument("--source-run", type=Path, required=True)
    structured_preflight.add_argument("--scenario-file", type=Path, required=True)
    structured_preflight.add_argument("--opponent", default="greedy")
    structured_preflight.add_argument("--seed", type=int, default=227)
    structured_preflight.add_argument("--device", default="auto")
    structured_preflight.add_argument("--server", default=str(DEFAULT_SERVER))
    _add_json_argument(structured_preflight)

    resume = subcommands.add_parser(
        "resume", help="continue a metadata-backed run as a new run"
    )
    resume.add_argument("source", type=Path)
    resume.add_argument("--run", required=True)
    resume.add_argument("--timesteps", required=True, type=int)
    _add_runtime_arguments(resume)
    _add_no_console_output_argument(resume)
    _add_json_argument(resume)

    status = subcommands.add_parser("status", help="read durable local run status")
    status.add_argument("run", type=Path)
    status.add_argument("--follow", action="store_true")
    status.add_argument("--interval", type=float, default=1.0)
    _add_json_argument(status)

    stop = subcommands.add_parser("stop", help="request a controlled training stop")
    stop.add_argument("run", type=Path)
    mode = stop.add_mutually_exclusive_group()
    mode.add_argument("--after-checkpoint", action="store_true")
    mode.add_argument("--now", action="store_true")
    _add_json_argument(stop)

    inspect = subcommands.add_parser(
        "inspect-model", help="resolve model identity and compatibility metadata"
    )
    inspect.add_argument("model")
    _add_json_argument(inspect)

    publish = subcommands.add_parser(
        "publish-checkpoint", help="create an editor-lab-only candidate artifact"
    )
    publish.add_argument("run", type=Path)
    publish.add_argument("--name", required=True)
    _add_json_argument(publish)

    evaluate = subcommands.add_parser(
        "evaluate", help="run a deterministic controller matchup"
    )
    evaluate.add_argument("--p0", required=True)
    evaluate.add_argument("--p1", required=True)
    evaluate.add_argument("--games", type=int, default=30)
    evaluate.add_argument("--seed-start", type=int, default=DEFAULT_HELD_OUT_SEED)
    evaluate.add_argument("--both-seats", action="store_true")
    evaluate.add_argument("--workers", type=int, default=1)
    evaluate.add_argument("--server", default=str(DEFAULT_SERVER))
    evaluate.add_argument("--output", type=Path)
    evaluate.add_argument(
        "--capture-trace",
        action="store_true",
        help="capture evaluation-only tactical transition evidence",
    )
    evaluate.add_argument(
        "--start-profile",
        help="force a declared tactical start profile with the candidate as reference",
    )
    evaluate.add_argument(
        "--evidence-retention",
        choices=("diagnostic", "all"),
        default="diagnostic",
        help="retain diagnostic traces or every trace/replay",
    )
    evaluate.add_argument(
        "--evidence-dir",
        type=Path,
        help="write per-match traces and replays; implies --capture-trace",
    )
    evaluate.add_argument(
        "--environment",
        choices=["tactical-v1", "tactical-v2", "adaptive-v1"],
        help="explicit environment; required to select tactical-v2 for scripted-only matchups",
    )
    _add_json_argument(evaluate)

    benchmark = subcommands.add_parser(
        "benchmark", help="measure headless GymServer throughput"
    )
    benchmark.add_argument("--games", type=int, default=10)
    benchmark.add_argument("--seed-start", type=int, default=DEFAULT_HELD_OUT_SEED)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument(
        "--environment",
        choices=["tactical-v1", "tactical-v2", "adaptive-v1"],
        default="tactical-v1",
    )
    benchmark.add_argument("--server", default=str(DEFAULT_SERVER))
    _add_json_argument(benchmark)
    return parser


def _tracker_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    raw_trackers = args.tracker or ["local"]
    wandb_options = {
        "project": args.wandb_project,
        "entity": args.wandb_entity,
        "mode": args.wandb_mode,
        "group": args.wandb_group,
        "tags": args.wandb_tag or None,
        "upload_artifacts": True if args.wandb_upload_artifacts else None,
    }
    if any(value is not None for value in wandb_options.values()) and "wandb" not in raw_trackers:
        raise ValueError("W&B options require --tracker wandb")
    trackers: list[dict[str, Any]] = []
    for raw in raw_trackers:
        if raw in {"local", "tensorboard"}:
            trackers.append({"kind": raw})
        elif raw == "wandb":
            trackers.append(
                {
                    "kind": "wandb",
                    **{
                        key: value
                        for key, value in wandb_options.items()
                        if value is not None
                    },
                }
            )
        elif raw.startswith("custom="):
            adapter = raw.removeprefix("custom=")
            if adapter.count(":") != 1 or not all(adapter.split(":", 1)):
                raise ValueError("custom tracker must use custom=module:function")
            trackers.append({"kind": "custom", "adapter": adapter})
        else:
            raise ValueError(
                "tracker must be local, tensorboard, wandb, or custom=module:function"
            )
    return trackers


def _training_config(args: argparse.Namespace) -> RunConfig:
    policy = "HexCNN" if args.algorithm == "maskable_ppo" else "MlpPolicy"
    environment = args.environment or "tactical-v2"
    requested_algorithm_options = {
        key: value
        for key, value in {
            "learning_rate": args.learning_rate,
            "n_epochs": args.ppo_epochs,
            "target_kl": args.target_kl,
        }.items()
        if value is not None
    }
    algorithm_options = requested_algorithm_options
    episode_seed_base = args.episode_seed_base
    if args.resume:
        source_manifest = read_json(_source_run_dir(Path(args.resume)) / "run.json")
        source_config = source_manifest.get("config")
        if not isinstance(source_config, dict):
            raise ValueError("resume source run is missing configuration metadata")
        source_environment = source_config.get("environment")
        if source_environment not in {"tactical-v1", "tactical-v2", "adaptive-v1"}:
            raise ValueError("resume source run is missing valid environment metadata")
        if args.environment is not None and args.environment != source_environment:
            raise ValueError("resume environment does not match the source run")
        environment = source_environment
        if requested_algorithm_options:
            raise ValueError("PPO options cannot be overridden during resume")
        source_options = source_config.get("algorithm_options", {})
        if not isinstance(source_options, dict):
            raise ValueError("resume source run has invalid algorithm options")
        algorithm_options = dict(source_options)
        if episode_seed_base is None:
            episode_seed_base = source_config.get("episode_seed_base")
    return RunConfig(
        backend="sb3",
        algorithm=args.algorithm,
        policy=policy,
        run_name=args.run,
        seed=args.seed,
        total_timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_every,
        workers=args.workers,
        device=args.device,
        learner_seat=args.learner_seat,
        opponent=controller_config(args.opponent),
        trackers=_tracker_configs(args),
        resume_source=args.resume,
        algorithm_options=algorithm_options,
        actor_init_source=args.actor_init,
        episode_seed_base=episode_seed_base,
        environment=environment,
    )


def _source_run_dir(source: Path) -> Path:
    source = Path(source).resolve()
    candidates = [source] if source.is_dir() else [source.parent, source.parent.parent]
    for candidate in candidates:
        if (candidate / "run.json").is_file():
            return candidate
    raise ValueError("resume source must belong to a metadata-backed run")


def _source_environment(source_run: Path) -> str:
    manifest = read_json(Path(source_run) / "run.json")
    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("resume source run is missing configuration metadata")
    environment = raw_config.get("environment")
    if environment not in {"tactical-v1", "tactical-v2", "adaptive-v1"}:
        raise ValueError("resume source run is missing valid environment metadata")
    return environment


def _source_scenario(source_run: Path, environment: str) -> ResolvedScenario:
    scenario_path = Path(source_run) / "scenario.json"
    if scenario_path.is_file():
        return resolve_scenario(
            environment=environment,
            scenario_file=scenario_path,
            template_id=None,
        )
    return legacy_default_scenario(environment)


def _training_scenario(args: argparse.Namespace) -> ResolvedScenario:
    if args.resume:
        source_run = _source_run_dir(Path(args.resume))
        environment = _source_environment(source_run)
        return _source_scenario(source_run, environment)
    environment = args.environment or "tactical-v2"
    return resolve_scenario(
        environment=environment,
        scenario_file=args.scenario_file,
        template_id=args.template,
        enforce_round_cap_minimum=True,
    )


def _resume_scenario(args: argparse.Namespace) -> ResolvedScenario:
    source_run = _source_run_dir(args.source)
    return _source_scenario(source_run, _source_environment(source_run))


def _resume_config(args: argparse.Namespace) -> RunConfig:
    source_run = _source_run_dir(args.source)
    source = read_json(source_run / "run.json")
    raw_config = source.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("resume source run is missing configuration metadata")
    if raw_config.get("environment") not in {"tactical-v1", "tactical-v2", "adaptive-v1"}:
        raise ValueError("resume source run is missing valid environment metadata")
    config = RunConfig(**raw_config)
    return replace(
        config,
        run_name=args.run,
        total_timesteps=args.timesteps,
        resume_source=str(source_run),
        actor_init_source=None,
    )


def read_seat_audit(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Aggregate learner-seat episode counts from the manifest's monitor shards."""
    run_dir = Path(run_dir).resolve()
    unreadable = {
        "seat_0_episodes": 0,
        "seat_1_episodes": 0,
        "readable": False,
        "balanced": False,
        "warning": "",
    }
    monitor_files = manifest.get("monitor_files")
    if (
        not isinstance(monitor_files, list)
        or not monitor_files
        or any(not isinstance(relative, str) or not relative for relative in monitor_files)
    ):
        unreadable["warning"] = (
            f"{run_dir / 'run.json'}: monitor_files must be a non-empty list of paths"
        )
        return unreadable

    seat_counts = {0: 0, 1: 0}
    for relative in monitor_files:
        monitor_path = run_dir / relative
        try:
            with monitor_path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream, strict=True)
                if reader.fieldnames is None or "learner_seat" not in reader.fieldnames:
                    unreadable["warning"] = (
                        f"{monitor_path}: missing learner_seat header"
                    )
                    return unreadable
                for line_number, row in enumerate(reader, start=2):
                    raw_seat = row.get("learner_seat")
                    if raw_seat not in {"0", "1"}:
                        unreadable["warning"] = (
                            f"{monitor_path}:{line_number}: invalid learner_seat "
                            f"{raw_seat!r}"
                        )
                        return unreadable
                    seat_counts[int(raw_seat)] += 1
        except (OSError, UnicodeError, csv.Error) as error:
            unreadable["warning"] = f"{monitor_path}: {error}"
            return unreadable

    config = manifest.get("config")
    if not isinstance(config, dict):
        config = {}
    try:
        tolerance = max(1, int(config.get("workers", 1)))
    except (TypeError, ValueError) as error:
        unreadable["warning"] = f"{run_dir / 'run.json'}: invalid config.workers: {error}"
        return unreadable
    balanced = abs(seat_counts[0] - seat_counts[1]) <= tolerance
    warning = ""
    if (
        config.get("learner_seat") == "alternating"
        and manifest.get("state") in TERMINAL_STATES
        and not balanced
    ):
        warning = (
            "Learner seat audit is materially imbalanced: "
            f"Seat 0 has {seat_counts[0]} episodes and "
            f"Seat 1 has {seat_counts[1]} episodes "
            f"(worker tolerance {tolerance})."
        )
    return {
        "seat_0_episodes": seat_counts[0],
        "seat_1_episodes": seat_counts[1],
        "readable": True,
        "balanced": balanced,
        "warning": warning,
    }


def _run_result(run_dir: Path, *, require_manifest: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run.json"
    result: dict[str, Any] = {"run_dir": str(run_dir)}
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        result["run"] = manifest
        result["seat_audit"] = read_seat_audit(run_dir, manifest)
    elif require_manifest:
        raise FileNotFoundError(manifest_path)
    return result


def inspect_model(raw: str) -> dict[str, Any]:
    resolved = ControllerResolver().resolve(raw)
    metadata = resolved.metadata()
    spec = normalize_controller_spec(raw)
    if spec.kind == "run" and spec.path is not None:
        metadata["source_run"] = str(spec.path.resolve())
    elif resolved.path is not None:
        for parent in resolved.path.parents:
            if (parent / "run.json").is_file():
                metadata["source_run"] = str(parent.resolve())
                break
    return metadata


def _structured_preflight_device(requested: str) -> str:
    """Resolve the training device without allocating model or run state."""

    import torch

    if type(requested) is not str or not requested.strip():
        raise ValueError("tactical-v3 device is required")
    if requested == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (RuntimeError, TypeError) as error:
        raise ValueError(f"invalid tactical-v3 device {requested!r}") from error
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "tactical-v3 continuation requested CUDA but it is unavailable"
            )
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError(
                "tactical-v3 continuation requested CUDA but no CUDA devices are visible"
            )
        if device.index is not None and device.index >= device_count:
            raise RuntimeError(
                "tactical-v3 continuation requested CUDA device "
                f"{device.index}, but only {device_count} CUDA device(s) are visible"
            )
    return str(device)


def preflight_structured_continuation(
    *,
    source_run: Path,
    scenario_file: Path,
    opponent: str,
    seed: int,
    device: str,
    server_cmd: Sequence[str],
) -> dict[str, Any]:
    """Authenticate and cross-check a continuation without creating run artifacts."""

    from .tactical_v3_checkpoint import validate_structured_run
    from .tactical_v3_client import TacticalV3GymClient
    from .tactical_v3_continuation import (
        _resolve_opponent,
        _start_distribution,
        _validate_model_opponent,
        _validate_target_scenario_identity,
    )
    from .tactical_v3_pilot import (
        _pilot_configs,
        _validate_compatible_transfer_identity,
    )

    if type(seed) is not int or not 0 <= seed <= 20_000:
        raise ValueError("tactical-v3 seed must be an integer from 0 through 20000")
    target_scenario = resolve_scenario(
        environment="tactical-v3",
        scenario_file=Path(scenario_file),
        template_id=None,
    )
    start_distribution = _start_distribution(target_scenario.document)
    effective_device = _structured_preflight_device(device)

    # This is intentionally the full schema-2 validator, rather than the cheaper
    # manifest inspection used by the Editor. It authenticates the checkpoint and
    # all package evidence while loading the exact source architecture on CPU.
    source = validate_structured_run(Path(source_run))
    resolved_opponent = _resolve_opponent(opponent)
    target_model_config, _, _ = _pilot_configs(seed, effective_device)
    if source.model.config != target_model_config:
        raise ValueError(
            "source policy model config does not match the continuation model"
        )

    with TacticalV3GymClient(server_cmd, environment_kind="duel") as client:
        target_identity = client.identity
        _validate_target_scenario_identity(
            target_scenario,
            target_identity,
            start_distribution,
        )
        _validate_compatible_transfer_identity(
            source.metadata.identity,
            target_identity,
            subject="source policy",
        )
        if resolved_opponent.binding is not None:
            _validate_model_opponent(
                resolved_opponent.binding.resolved,
                target_identity,
            )

    source_identity = source.metadata.identity
    return {
        "environment": "tactical-v3",
        "source": {
            "run_dir": str(Path(source_run).resolve()),
            "checkpoint": str((Path(source_run).resolve() / "checkpoints" / "best.pt")),
            "contract_hash": source_identity.contract_hash,
            "encoding_hash": source_identity.encoding_hash,
            "capacity_hash": source_identity.capacity_hash,
        },
        "target": {
            "scenario_file": str(Path(scenario_file).resolve()),
            "scenario_id": target_scenario.template_id,
            "scenario_schema_version": target_scenario.schema_version,
            "contract_hash": target_identity.contract_hash,
            "encoding_hash": target_identity.encoding_hash,
            "capacity_hash": target_identity.capacity_hash,
        },
        "opponent": dict(resolved_opponent.metadata),
        "device": {
            "requested": device,
            "effective": effective_device,
        },
        "model_config": asdict(target_model_config),
    }


def _emit_json(
    stdout: TextIO, command: str, result: dict[str, Any], *, ok: bool = True
) -> None:
    print(
        json.dumps(
            {
                "schema_version": JSON_SCHEMA_VERSION,
                "command": command,
                "ok": ok,
                "result": result,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        file=stdout,
        flush=True,
    )


def _emit_human(stdout: TextIO, command: str, result: dict[str, Any]) -> None:
    if command == "doctor":
        print("doctor: ok" if result.get("ok") else "doctor: problems found", file=stdout)
        for check in result.get("checks", []):
            marker = "ok" if check.get("ok") else "unavailable"
            print(f"  {check.get('name')}: {marker} ({check.get('detail', '')})", file=stdout)
        return
    if command in {"train", "train-structured", "resume", "status"}:
        run = result.get("run")
        if run is None:
            print(f"run completed: {result['run_dir']}", file=stdout)
            return
        print(
            f"{result['run_dir']}: {run.get('state')} at {run.get('timesteps', 0)} timesteps",
            file=stdout,
        )
        return
    if command == "stop":
        print(
            f"{result['run_dir']}: requested {result['control'].get('request')}",
            file=stdout,
        )
        return
    if command == "inspect-model":
        print(
            f"{result.get('algorithm') or result.get('kind')}: {result.get('path') or result.get('name')}",
            file=stdout,
        )
        return
    if command == "publish-checkpoint":
        print(
            f"candidate {result.get('name')} published to {result.get('candidate_dir')}",
            file=stdout,
        )
        return
    if command == "evaluate":
        print(
            f"evaluation: {result.get('wins', 0)} W / {result.get('losses', 0)} L / "
            f"{result.get('draws', 0)} D over {result.get('games', 0)} games",
            file=stdout,
        )
        return
    if command == "benchmark":
        print(
            f"benchmark: {result.get('resets_per_second', 0):.2f} resets/s, "
            f"{result.get('decisions_per_second', 0):.2f} decisions/s",
            file=stdout,
        )
        return
    print(json.dumps(result, indent=2, sort_keys=True), file=stdout)


def _dispatch(
    args: argparse.Namespace,
    *,
    runner: Callable[..., Path],
    sleeper: Callable[[float], None],
    structured_runner: Callable[..., Path] | None = None,
    status_update: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    if args.command == "doctor":
        return doctor_environment(
            server_cmd=["dotnet", args.server],
            environment=args.environment,
            runs_root=args.runs_root,
            trackers=args.tracker,
        )
    if args.command == "train":
        scenario = _training_scenario(args)
        config = _training_config(args)
        runner_options = {
            "runs_root": Path(args.runs_root),
            "server_cmd": ["dotnet", args.server],
        }
        if args.no_console_output:
            runner_options["console_output"] = False
        return _run_result(
            runner(
                config,
                scenario=scenario,
                **runner_options,
            )
        )
    if args.command == "train-structured":
        from .tactical_v3_continuation import (
            StructuredContinuationConfig,
            run_structured_continuation,
        )

        config = StructuredContinuationConfig(
            run_name=args.run,
            source_run=args.source_run,
            scenario_file=args.scenario_file,
            opponent=args.opponent,
            train_label_target=args.train_labels,
            validation_label_target=args.validation_labels,
            seed=args.seed,
            device=args.device,
            learner_seat=args.learner_seat,
            trackers=tuple(_tracker_configs(args)),
        )
        return _run_result(
            (structured_runner or run_structured_continuation)(
                config,
                runs_root=Path(args.runs_root),
                server_cmd=["dotnet", args.server, "--scenario-file", str(args.scenario_file)],
            )
        )
    if args.command == "preflight-structured":
        return preflight_structured_continuation(
            source_run=args.source_run,
            scenario_file=args.scenario_file,
            opponent=args.opponent,
            seed=args.seed,
            device=args.device,
            server_cmd=[
                "dotnet",
                args.server,
                "--scenario-file",
                str(args.scenario_file),
            ],
        )
    if args.command == "resume":
        scenario = _resume_scenario(args)
        config = _resume_config(args)
        runner_options = {
            "runs_root": Path(args.runs_root),
            "server_cmd": ["dotnet", args.server],
        }
        if args.no_console_output:
            runner_options["console_output"] = False
        return _run_result(
            runner(
                config,
                scenario=scenario,
                **runner_options,
            )
        )
    if args.command == "status":
        if args.interval <= 0:
            raise ValueError("status interval must be positive")
        result = _run_result(args.run, require_manifest=True)
        if args.follow and status_update is not None:
            status_update(result)
        while args.follow and result["run"].get("state") not in TERMINAL_STATES:
            sleeper(args.interval)
            result = _run_result(args.run, require_manifest=True)
            if status_update is not None:
                status_update(result)
        return result
    if args.command == "stop":
        run_dir = Path(args.run).resolve()
        control = request_stop(run_dir, after_checkpoint=args.after_checkpoint)
        return {"run_dir": str(run_dir), "control": control}
    if args.command == "inspect-model":
        return inspect_model(args.model)
    if args.command == "publish-checkpoint":
        candidate_dir = publish_candidate(args.run, args.name)
        return read_json(candidate_dir / "candidate.json")
    if args.command == "evaluate":
        return evaluate_controllers(
            args.p0,
            args.p1,
            games=args.games,
            seed_start=args.seed_start,
            both_seats=args.both_seats,
            workers=args.workers,
            server_cmd=["dotnet", args.server],
            output_path=args.output,
            environment=args.environment,
            start_profile=args.start_profile,
            capture_trace=args.capture_trace or args.evidence_dir is not None,
            evidence_dir=args.evidence_dir,
            evidence_retention=args.evidence_retention,
        )
    if args.command == "benchmark":
        return benchmark_gymserver(
            games=args.games,
            seed_start=args.seed_start,
            workers=args.workers,
            server_cmd=["dotnet", args.server],
            environment=args.environment,
        )
    raise AssertionError(f"unhandled command {args.command!r}")


def _training_run_dir(args: argparse.Namespace) -> Path:
    return Path(args.runs_root) / args.run


@contextmanager
def _capture_stderr_to_file(path: Path) -> Iterator[TextIO]:
    """Duplicate the process's stderr onto ``path`` for the lifetime of the context.

    Trainers launched detached by the Unity ML Lab have their console discarded, so an
    uncaught traceback, a CUDA/native fault, or worker/SB3 noise written to stderr was
    previously unrecoverable. The file is opened line-buffered and created unconditionally
    (it may legitimately stay empty on a clean run). Both ``sys.stderr`` and the raw OS
    file descriptor are redirected so writes land in the file whether they come from
    Python, a native extension, or a child process inheriting this process's stderr handle.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(path, "w", buffering=1, encoding="utf-8", errors="replace")
    original_stderr = sys.stderr
    target_fd: int | None = None
    saved_fd: int | None = None
    try:
        target_fd = original_stderr.fileno()
    except (OSError, ValueError, AttributeError):
        target_fd = None
    if target_fd is not None:
        try:
            saved_fd = os.dup(target_fd)
            os.dup2(log_file.fileno(), target_fd)
        except OSError:
            saved_fd = None
    sys.stderr = log_file
    try:
        yield log_file
    finally:
        sys.stderr = original_stderr
        if target_fd is not None and saved_fd is not None:
            try:
                os.dup2(saved_fd, target_fd)
            finally:
                os.close(saved_fd)
        log_file.close()


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Path] = run_training,
    structured_runner: Callable[..., Path] | None = None,
    sleeper: Callable[[float], None] = time.sleep,
    stdout: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    human_follow = args.command == "status" and args.follow and not args.json
    no_console_output = getattr(args, "no_console_output", False)
    with ExitStack() as console_stack:
        stderr_log: TextIO | None = None
        if args.command in {"train", "train-structured", "resume"}:
            stderr_log = console_stack.enter_context(
                _capture_stderr_to_file(_training_run_dir(args) / "train-err.log")
            )
        if no_console_output:
            sink = console_stack.enter_context(
                open(os.devnull, "w", encoding="utf-8")
            )
            console_stack.enter_context(redirect_stdout(sink))
            console_stack.enter_context(redirect_stderr(sink))
        try:
            result = _dispatch(
                args,
                runner=runner,
                sleeper=sleeper,
                structured_runner=structured_runner,
                status_update=(
                    (lambda update: _emit_human(output, "status", update))
                    if human_follow
                    else None
                ),
            )
        except Exception as error:
            if stderr_log is not None:
                traceback.print_exc(file=stderr_log)
                stderr_log.flush()
            if no_console_output:
                return 1
            if args.json:
                _emit_json(
                    output,
                    args.command,
                    {"error": type(error).__name__, "message": str(error)},
                    ok=False,
                )
                return 1
            raise
    if no_console_output:
        return 0
    if args.json:
        _emit_json(output, args.command, result)
    elif not human_follow:
        _emit_human(output, args.command, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
