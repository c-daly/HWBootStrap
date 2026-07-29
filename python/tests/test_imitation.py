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

def test_reopen_rejects_corrupt_physical_shard_and_replay_ownership(tmp_path: Path) -> None:
    stage_replay(tmp_path)
    writer = DemonstrationWriter.create(tmp_path, contract=contract())
    writer.append_game(game(), [decision(0)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    shard = tmp_path / manifest["shards"][0]["path"]
    with np.load(shard) as loaded:
        corrupted = {name: loaded[name] for name in loaded.files}
    corrupted["actions"] = np.asarray([4], dtype=np.int64)
    np.savez_compressed(shard, **corrupted)
    with pytest.raises(ValueError, match="shard hash mismatch"):
        DemonstrationWriter.create(tmp_path, contract=contract())


def test_writer_enforces_locked_search_provenance_and_records_source_ranges(tmp_path: Path) -> None:
    stage_replay(tmp_path)
    replay = tmp_path / "replays" / "search.replay"; replay.write_bytes(b"search")
    search = DemonstrationGame("train", "bounded-search", {"depth": 4, "expansion_budget": 512, "use_heuristic": True}, "random", "conversion-3v1-near", 11_500_000, 0, "replays/search.replay", hashlib.sha256(b"search").hexdigest(), "win", "c" * 64, "a" * 64, "b" * 64)
    writer = DemonstrationWriter.create(tmp_path, contract=contract())
    with pytest.raises(ValueError, match="teacher parameters"):
        writer.append_game(DemonstrationGame(**{**search.__dict__, "teacher_parameters": {"depth": 3}}), [decision(0)])
    writer.append_game(search, [decision(0)])
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["source_ranges"] == [{"partition": "train", "teacher": "bounded-search", "profile": "conversion-3v1-near", "seed_start": 11_500_000, "seed_stop": 11_500_000, "game_count": 1, "decision_count": 1}]

def test_dirty_provenance_includes_untracked_source_but_excludes_dataset_output(monkeypatch, tmp_path: Path) -> None:
    import ml_lab.imitation as module
    source = tmp_path / "repo"; dataset = source / "python" / "datasets" / "annihilation-imitation-v1"
    def output(command, **_kwargs):
        return str(source) if command[-1] == "--show-toplevel" else "d" * 40
    monkeypatch.setattr(module.subprocess, "check_output", output)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **_kwargs: type("Result", (), {"stdout": "?? python/datasets/annihilation-imitation-v1/shards/new.npz\n?? python/new_source.py\n"})())
    assert module._git_metadata(dataset) == ("d" * 40, True)
    monkeypatch.setattr(module.subprocess, "run", lambda command, **_kwargs: type("Result", (), {"stdout": "?? python/datasets/annihilation-imitation-v1/shards/new.npz\n"})())
    assert module._git_metadata(dataset) == ("d" * 40, False)
