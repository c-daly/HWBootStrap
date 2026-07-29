from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from ml_lab.contracts import EnvironmentContract
from ml_lab.imitation import DemonstrationGame, DemonstrationWriter, validate_decision


def contract() -> EnvironmentContract:
    return EnvironmentContract("tactical-v2", "a" * 64, "b" * 64, 3, 5, {}, [], {})


def decision(index: int, seat: int = 0) -> dict[str, object]:
    return {"observation": [float(index), 1.0, 2.0], "legal_mask": [True, False, True, False, False], "action": 2, "seat": seat, "decision_index": index, "action_kind": 1}


def game(seed: int = 11_000_000) -> DemonstrationGame:
    return DemonstrationGame("train", "greedy", {}, "random", "standard-3v3", seed, 0, "replays/game.replay", hashlib.sha256(b"replay").hexdigest(), "win", "c" * 64, "a" * 64, "b" * 64)


def stage_replay(root: Path) -> None:
    replay = root / "replays" / "game.replay"
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_bytes(b"replay")


def test_writer_commits_one_complete_game_atomically(tmp_path: Path) -> None:
    stage_replay(tmp_path)
    writer = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=4)
    writer.append_game(game(), [decision(index) for index in range(3)])
    writer.close()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    games = [json.loads(line) for line in (tmp_path / "games.jsonl").read_text().splitlines()]
    assert manifest["decision_count"] == 3 and games[0]["row_count"] == 3
    assert all((tmp_path / item["path"]).is_file() for item in manifest["shards"])
    assert np.load(tmp_path / manifest["shards"][0]["path"])["packed_masks"].shape == (3, 1)


def test_writer_reconciles_orphan_before_manifest_replacement(tmp_path: Path) -> None:
    stage_replay(tmp_path)
    writer = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=4)
    writer.fail_before_manifest_replace = True
    with pytest.raises(RuntimeError, match="simulated interruption"):
        writer.append_game(game(), [decision(index) for index in range(2)])
    reopened = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=4)
    assert reopened.completed_keys() == set() and not list((tmp_path / "shards").glob("*.npz"))
    reopened.append_game(game(), [decision(index) for index in range(2)])
    assert len((tmp_path / "games.jsonl").read_text().splitlines()) == 1


def test_writer_rejects_invalid_rows_and_cross_partition_seed_reuse(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not legal"):
        validate_decision({**decision(0), "legal_mask": [True, False, False, False, False]}, contract())
    with pytest.raises(ValueError, match="invalid demonstration observation"):
        validate_decision({**decision(0), "observation": [0.0, float("nan"), 1.0]}, contract())
    stage_replay(tmp_path)
    writer = DemonstrationWriter.create(tmp_path, contract=contract())
    writer.append_game(game(), [decision(0)])
    with pytest.raises(ValueError, match="seed reuse"):
        writer.append_game(DemonstrationGame(**{**game().__dict__, "partition": "validation"}), [decision(0)])
