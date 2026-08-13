"""Shared authenticated tactical-v3 test fixture access."""

from __future__ import annotations

import json
from pathlib import Path

from ml_lab.tactical_v3_corpus import StructuredCorpus, load_corpus
from ml_lab.tactical_v3_schema import TacticalV3SemanticIdentity, parse_spaces


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "tactical_v3"
DUEL_IDENTITY_FIXTURE = FIXTURE_ROOT / "seed-41-duel-spaces.json"
TINY_CORPUS_ROOT = FIXTURE_ROOT / "tiny-corpus"


def load_duel_identity_fixture() -> TacticalV3SemanticIdentity:
    payload = json.loads(DUEL_IDENTITY_FIXTURE.read_text(encoding="utf-8"))
    return parse_spaces(payload)


def load_tiny_corpus_fixture() -> StructuredCorpus:
    return load_corpus(TINY_CORPUS_ROOT, load_duel_identity_fixture())
