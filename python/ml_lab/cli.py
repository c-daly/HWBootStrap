"""Command-line surface for the HexWars ML Lab."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from .contracts import RunConfig
from .controllers import ControllerSpec, normalize_controller_spec
from .training import run_training


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER = PROJECT_ROOT / "engine" / "HexWars.GymServer" / "bin" / "Release" / "net8.0" / "HexWars.GymServer.dll"
DEFAULT_RUNS_ROOT = PROJECT_ROOT / "python" / "runs"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hexwars_ml.py")
    subcommands = parser.add_subparsers(dest="command", required=True)
    train = subcommands.add_parser("train", help="run headless SB3 training")
    train.add_argument("--run", required=True)
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
    train.add_argument("--tracker", action="append", choices=["local", "tensorboard", "wandb"])
    train.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT)
    train.add_argument("--server", default=str(DEFAULT_SERVER))
    return parser


def main(
    argv: list[str] | None = None,
    *,
    runner: Callable[..., Path] = run_training,
) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "train":
        raise AssertionError(f"unhandled command {args.command!r}")
    policy = "HexCNN" if args.algorithm == "maskable_ppo" else "MlpPolicy"
    tracker_names = args.tracker or ["local"]
    config = RunConfig(
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
        trackers=[{"kind": name} for name in tracker_names],
        resume_source=args.resume,
    )
    run_dir = runner(
        config,
        runs_root=Path(args.runs_root),
        server_cmd=["dotnet", args.server],
    )
    print(f"run completed: {run_dir}")
    return 0
