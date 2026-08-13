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


def _smoke_training_configs(
    seed: int,
    device: str,
) -> tuple[TacticalV3ModelConfig, ObjectiveConfig, TrainerConfig]:
    return (
        TacticalV3ModelConfig(
            hidden_dim=16,
            categorical_dim=4,
            cell_message_rounds=1,
            relation_rounds=1,
            attention_heads=4,
            feed_forward_dim=32,
            candidate_hidden_dim=32,
            horizon_turns=(4, 8, 16),
        ),
        ObjectiveConfig(
            policy_coefficient=1.0,
            outcome_coefficient=0.0,
            horizon_coefficient=0.0,
            remaining_turns_coefficient=0.0,
        ),
        TrainerConfig(
            seed=seed,
            batch_size=8,
            learning_rate=0.005,
            max_epochs=80,
            patience_epochs=80,
            gradient_clip_norm=1.0,
            device=device,
        ),
    )


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
    train.add_argument("--policy-spaces", required=True, type=Path)
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
        identity = parse_spaces(json.loads(
            args.policy_spaces.read_text(encoding="utf-8")
        ))
        corpus = load_corpus(args.corpus, identity)
        model_config, objective_config, trainer_config = _smoke_training_configs(
            args.seed, args.device
        )
        result = train_offline(
            corpus.train,
            corpus.validation,
            model_config,
            objective_config,
            trainer_config,
        )
        publish_structured_run(
            args.run_dir,
            result,
            corpus,
            training_scenario_path=args.scenario,
            policy_identity=identity,
        )
        return 0
    if args.command == "validate-run":
        validate_structured_run(args.run_dir)
        return 0
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
