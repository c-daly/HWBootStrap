"""Deprecated compatibility wrapper for successive frozen-opponent training rounds."""

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
    source = run_dir / read_json(run_dir / "run.json")["latest_checkpoint"]
    destination = Path(output)
    if destination.suffix.lower() != ".zip":
        destination = Path(f"{destination}.zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--opponent",
        required=True,
        help="opponent controller: greedy|random|ppo:PATH|dqn:PATH|run:PATH|JSON|@controller.json",
    )
    parser.add_argument("--out", default="sp")
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--server", default=os.environ.get("HEXWARS_SERVER", DEFAULT_DLL))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("DEPRECATED: use hexwars_ml.py train with a fixed or live opponent")
    pool: list[dict[str, object]] = [
        controller_config("greedy"),
        controller_config(args.opponent),
    ]
    final: Path | None = None
    for round_index in range(args.rounds):
        run_name = f"{args.out}_r{round_index}"
        config = RunConfig(
            backend="sb3",
            algorithm="maskable_ppo",
            policy="HexCNN",
            run_name=run_name,
            seed=args.seed + round_index,
            total_timesteps=args.timesteps,
            checkpoint_interval=25_000,
            workers=1,
            device="auto",
            learner_seat="0",
            opponent={"kind": "pool", "controllers": pool},
            trackers=[{"kind": "local"}],
            resume_source=None,
        )
        run_dir = run_training(
            config,
            runs_root=Path("runs"),
            server_cmd=["dotnet", args.server],
        )
        final = _export_latest(run_dir, run_name)
        pool = [*pool, {"kind": "run", "path": str(run_dir), "mode": "fixed"}]
    print(f"done -> {final}  ({args.rounds} round(s))")


if __name__ == "__main__":
    main()
