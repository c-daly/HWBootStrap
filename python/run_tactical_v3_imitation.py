from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the fixed tactical-v3 imitation smoke corpus.")
    subcommands = parser.add_subparsers(dest="command", required=True)
    build = subcommands.add_parser("build-tiny-corpus")
    build.add_argument("--server-dll", required=True, type=Path)
    build.add_argument("--scenario", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command != "build-tiny-corpus":
        raise AssertionError(args.command)
    from ml_lab.tactical_v3_corpus import create_tiny_corpus
    create_tiny_corpus(args.output, ["dotnet", str(args.server_dll), "--scenario-file", str(args.scenario)])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
