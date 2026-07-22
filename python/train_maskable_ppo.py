"""Deprecated compatibility wrapper for the unified HexWars ML trainer."""

import argparse
import os
import shutil
from pathlib import Path

from ml_lab.cli import controller_config
from ml_lab.contracts import RunConfig
from ml_lab.io import read_json
from ml_lab.training import run_training


DEFAULT_DLL = "../engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll"


def _export_latest(run_dir: Path, output: str) -> Path:
    manifest = read_json(run_dir / "run.json")
    source = run_dir / manifest["latest_checkpoint"]
    destination = Path(output)
    if destination.suffix.lower() != ".zip":
        destination = Path(f"{destination}.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", choices=["greedy", "random"], default="greedy")
    parser.add_argument("--seat", type=int, choices=[0, 1], default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--out", default="hexwars_ppo")
    parser.add_argument("--logdir", default=None)
    parser.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    parser.add_argument("--resume", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=25_000)
    args = parser.parse_args()

    print("DEPRECATED: use hexwars_ml.py train --algorithm maskable_ppo")
    run_path = Path(args.logdir or Path("runs") / args.out)
    config = RunConfig(
        backend="sb3",
        algorithm="maskable_ppo",
        policy="HexCNN",
        run_name=run_path.name,
        seed=args.seed,
        total_timesteps=args.timesteps,
        checkpoint_interval=args.checkpoint_freq,
        workers=1,
        device="auto",
        learner_seat=str(args.seat),
        opponent=controller_config(args.opponent),
        trackers=[{"kind": "local"}],
        resume_source=args.resume,
    )
    run_dir = run_training(
        config,
        runs_root=run_path.parent,
        server_cmd=["dotnet", args.server],
    )
    exported = _export_latest(run_dir, args.out)
    print(f"done -> {exported}  logs: {run_dir}")


if __name__ == "__main__":
    main()
