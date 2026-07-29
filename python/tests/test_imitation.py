from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from ml_lab.contracts import ContractMismatch, EnvironmentContract
from ml_lab.imitation import DemonstrationGame, DemonstrationWriter, Source, StratifiedDecisionSampler, load_imitation_dataset, masked_cross_entropy, validate_decision


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


def _staged_game(root: Path, partition: str, teacher: str, profile: str, seed: int, seat: int) -> DemonstrationGame:
    relative = f"replays/{partition}-{teacher}-{seed}-{seat}.replay"
    payload = relative.encode("utf-8")
    replay = root / relative
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_bytes(payload)
    return DemonstrationGame(partition, teacher, {} if teacher == "greedy" else {"depth": 4, "expansion_budget": 512, "use_heuristic": True}, "random", profile, seed, seat, relative, hashlib.sha256(payload).hexdigest(), "win", "c" * 64, "a" * 64, "b" * 64)


@pytest.fixture
def sampled_dataset(tmp_path: Path) -> Path:
    writer = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=2)
    writer.append_game(_staged_game(tmp_path, "train", "greedy", "standard-3v3", 11_000_000, 0), [decision(value, 0) for value in range(3)])
    writer.append_game(_staged_game(tmp_path, "train", "greedy", "standard-3v3", 11_000_000, 1), [{**decision(value, 1), "decision_index": index} for index, value in enumerate(range(3, 6))])
    writer.append_game(_staged_game(tmp_path, "train", "bounded-search", "conversion-3v1-near", 11_500_000, 0), [{**decision(value, 0), "action_kind": 2, "decision_index": index} for index, value in enumerate(range(6, 9))])
    writer.append_game(_staged_game(tmp_path, "train", "bounded-search", "conversion-3v1-near", 11_500_000, 1), [{**decision(value, 1), "action_kind": 3, "decision_index": index} for index, value in enumerate(range(9, 12))])
    writer.append_game(_staged_game(tmp_path, "validation", "greedy", "standard-3v3", 12_000_000, 0), [{**decision(12, 0), "decision_index": 0}])
    return tmp_path


def test_loader_rejects_contract_content_and_partition_seed_mismatches(sampled_dataset: Path) -> None:
    other = EnvironmentContract("tactical-v2", "d" * 64, "b" * 64, 3, 5, {}, [], {})
    with pytest.raises(ContractMismatch):
        load_imitation_dataset(sampled_dataset, expected_contract=other)
    records = [json.loads(line) for line in (sampled_dataset / "games.jsonl").read_text().splitlines()]
    records[-1]["seed"] = records[0]["seed"]
    (sampled_dataset / "games.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
    with pytest.raises(ValueError, match="partition"):
        load_imitation_dataset(sampled_dataset, expected_contract=contract())
    records[-1]["seed"] = 12_000_000
    (sampled_dataset / "games.jsonl").write_text("".join(json.dumps(item) + "\n" for item in records))
    manifest = json.loads((sampled_dataset / "manifest.json").read_text())
    shard = sampled_dataset / manifest["shards"][0]["path"]
    payload = bytearray(shard.read_bytes()); payload[-1] ^= 1; shard.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256"):
        load_imitation_dataset(sampled_dataset, expected_contract=contract())


def test_sampler_keeps_70_30_ratio_is_seeded_and_excludes_validation_rows(sampled_dataset: Path) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    first = StratifiedDecisionSampler(dataset, batch_size=100, standard_fraction=0.70, seed=211).next_batch()
    second = StratifiedDecisionSampler(dataset, batch_size=100, standard_fraction=0.70, seed=211).next_batch()
    different = StratifiedDecisionSampler(dataset, batch_size=100, standard_fraction=0.70, seed=212).next_batch()
    assert (first.sources == Source.GREEDY_STANDARD).sum() == 70
    assert (first.sources == Source.SEARCH_CONVERSION).sum() == 30
    assert list(zip(first.game_ids, first.decision_indices)) == list(zip(second.game_ids, second.decision_indices))
    assert list(zip(first.game_ids, first.decision_indices)) != list(zip(different.game_ids, different.decision_indices))
    assert set(first.partitions) == {"train"}
    assert first.observations.flags.owndata
    assert np.all(first.legal_masks[np.arange(len(first.actions)), first.actions])


def test_masked_cross_entropy_masks_illegal_logits_and_has_finite_gradient() -> None:
    logits = torch.tensor([[0.0, 900.0, 1.0], [2.0, -700.0, -1.0], [3.0, 1.0, 2.0], [0.0, 0.0, 0.0], [1.0, -1.0, 5.0]], requires_grad=True)
    masks = torch.tensor([[True, False, True], [True, False, True], [True, True, False], [False, True, True], [True, True, True]])
    actions = torch.tensor([2, 0, 1, 2, 0], dtype=torch.int64)
    loss = masked_cross_entropy(logits, masks, actions)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError, match="masked"):
        masked_cross_entropy(logits.detach(), masks, torch.tensor([1, 0, 1, 2, 0], dtype=torch.int64))
