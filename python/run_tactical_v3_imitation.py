from __future__ import annotations

import argparse
import json
from typing import Sequence

from ml_lab.tactical_v3_checkpoint import publish_structured_run, validate_structured_run
from ml_lab.tactical_v3_corpus import create_tiny_corpus, load_corpus
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_schema import parse_spaces
from ml_lab.tactical_v3_training import TrainerConfig, train_offline
from pathlib import Path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the fixed tactical-v3 imitation smoke corpus.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build-tiny-corpus")
    build.add_argument("--server-dll", required=True, type=Path)
    build.add_argument("--scenario", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    train = subcommands.add_parser("train")
    train.add_argument("--corpus", required=True, type=Path)
    train.add_argument("--scenario", required=True, type=Path)
    train.add_argument("--run-dir", required=True, type=Path)
    train.add_argument("--seed", required=True, type=int)
    train.add_argument("--device", required=True, type=str)
    validate = subcommands.add_parser("validate-run")
    validate.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.command == "build-tiny-corpus":
        create_tiny_corpus(args.output, ["dotnet", str(args.server_dll), "--scenario-file", str(args.scenario)])
        return 0
    if args.command == "train":
        identity = parse_spaces(json.loads(args.scenario.read_text(encoding="utf-8")))
        corpus = load_corpus(args.corpus, identity)
        result = train_offline(
            corpus.train, corpus.validation, TacticalV3ModelConfig(), ObjectiveConfig(),
            TrainerConfig(seed=args.seed, device=args.device),
        )
        publish_structured_run(args.run_dir, result, corpus, args.scenario)
        return 0
    if args.command == "validate-run":
        validate_structured_run(args.run_dir)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
