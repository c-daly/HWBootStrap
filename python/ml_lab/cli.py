"""Command-line surface for the HexWars ML Lab."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, TextIO

from .benchmark import benchmark_gymserver
from .contracts import RunConfig, request_stop
from .controllers import ControllerResolver, ControllerSpec, normalize_controller_spec
from .doctor import doctor_environment
from .evaluation import DEFAULT_HELD_OUT_SEED, evaluate_controllers, publish_candidate
from .io import read_json
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
        "--environment", choices=["tactical-v1", "adaptive-v1"], default="tactical-v1"
    )
    _add_runtime_arguments(doctor)
    _add_json_argument(doctor)

    train = subcommands.add_parser("train", help="run headless SB3 training")
    train.add_argument("--run", required=True)
    train.add_argument(
        "--environment",
        choices=["tactical-v1", "adaptive-v1"],
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
    _add_json_argument(evaluate)

    benchmark = subcommands.add_parser(
        "benchmark", help="measure headless GymServer throughput"
    )
    benchmark.add_argument("--games", type=int, default=10)
    benchmark.add_argument("--seed-start", type=int, default=DEFAULT_HELD_OUT_SEED)
    benchmark.add_argument("--workers", type=int, default=1)
    benchmark.add_argument(
        "--environment", choices=["tactical-v1", "adaptive-v1"], default="tactical-v1"
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
    environment = args.environment or "tactical-v1"
    if args.resume:
        source_manifest = read_json(_source_run_dir(Path(args.resume)) / "run.json")
        source_config = source_manifest.get("config")
        if not isinstance(source_config, dict):
            raise ValueError("resume source run is missing configuration metadata")
        source_environment = source_config.get("environment")
        if source_environment not in {"tactical-v1", "adaptive-v1"}:
            raise ValueError("resume source run is missing valid environment metadata")
        if args.environment is not None and args.environment != source_environment:
            raise ValueError("resume environment does not match the source run")
        environment = source_environment
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
        environment=environment,
    )


def _source_run_dir(source: Path) -> Path:
    source = Path(source).resolve()
    candidates = [source] if source.is_dir() else [source.parent, source.parent.parent]
    for candidate in candidates:
        if (candidate / "run.json").is_file():
            return candidate
    raise ValueError("resume source must belong to a metadata-backed run")


def _resume_config(args: argparse.Namespace) -> RunConfig:
    source_run = _source_run_dir(args.source)
    source = read_json(source_run / "run.json")
    raw_config = source.get("config")
    if not isinstance(raw_config, dict):
        raise ValueError("resume source run is missing configuration metadata")
    if raw_config.get("environment") not in {"tactical-v1", "adaptive-v1"}:
        raise ValueError("resume source run is missing valid environment metadata")
    config = RunConfig(**raw_config)
    return replace(
        config,
        run_name=args.run,
        total_timesteps=args.timesteps,
        resume_source=str(source_run),
    )


def _run_result(run_dir: Path, *, require_manifest: bool = False) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    manifest_path = run_dir / "run.json"
    result: dict[str, Any] = {"run_dir": str(run_dir)}
    if manifest_path.is_file():
        result["run"] = read_json(manifest_path)
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
    if command in {"train", "resume", "status"}:
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
                **runner_options,
            )
        )
    if args.command == "resume":
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


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Path] = run_training,
    sleeper: Callable[[float], None] = time.sleep,
    stdout: TextIO | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    output = stdout if stdout is not None else sys.stdout
    human_follow = args.command == "status" and args.follow and not args.json
    no_console_output = getattr(args, "no_console_output", False)
    with ExitStack() as console_stack:
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
                status_update=(
                    (lambda update: _emit_human(output, "status", update))
                    if human_follow
                    else None
                ),
            )
        except Exception as error:
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
