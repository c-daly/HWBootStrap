"""Compatibility entry point for the unified HexWars evaluator.

Existing ``winrate.py`` arguments are forwarded unchanged to
``hexwars_ml.py evaluate`` so there is only one evaluation implementation.
"""

from __future__ import annotations

import sys

from ml_lab.cli import main as ml_main


def main(argv: list[str] | None = None) -> int:
    return ml_main(["evaluate", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    raise SystemExit(main())
