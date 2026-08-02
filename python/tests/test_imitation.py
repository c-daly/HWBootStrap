from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile
from dataclasses import asdict

import numpy as np
import pytest
import torch
import gymnasium as gym
from gymnasium import spaces
import ml_lab.imitation as imitation_module

from ml_lab.contracts import ContractMismatch, EnvironmentContract
from hex_cnn import HexCNN
from ml_lab.algorithms import MaskablePPOAdapter
from ml_lab.controllers import ControllerResolver
from ml_lab.scenarios import resolve_scenario
from ml_lab.imitation import BehavioralCloningConfig, DemonstrationGame, DemonstrationWriter, ImitationBatch, Source, StratifiedDecisionSampler, load_imitation_dataset, masked_cross_entropy, train_behavioral_clone, validate_decision


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


def _staged_game(root: Path, partition: str, teacher: str, profile: str, seed: int, seat: int, scenario_hash: str = "c" * 64) -> DemonstrationGame:
    relative = f"replays/{partition}-{teacher}-{seed}-{seat}.replay"
    payload = relative.encode("utf-8")
    replay = root / relative
    replay.parent.mkdir(parents=True, exist_ok=True)
    replay.write_bytes(payload)
    return DemonstrationGame(partition, teacher, {} if teacher == "greedy" else {"depth": 4, "expansion_budget": 512, "use_heuristic": True}, "random", profile, seed, seat, relative, hashlib.sha256(payload).hexdigest(), "win", scenario_hash, "a" * 64, "b" * 64)


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


def test_loader_index_scan_does_not_decode_full_shard_payloads(sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    requested: set[str] = set()
    original_load = imitation_module.np.load

    class TrackingNpz:
        def __init__(self, wrapped): self._wrapped = wrapped
        @property
        def files(self): return self._wrapped.files
        def __getitem__(self, key: str): requested.add(key); return self._wrapped[key]
        def __enter__(self): self._wrapped.__enter__(); return self
        def __exit__(self, *args): return self._wrapped.__exit__(*args)

    monkeypatch.setattr(imitation_module.np, "load", lambda *args, **kwargs: TrackingNpz(original_load(*args, **kwargs)))
    load_imitation_dataset(sampled_dataset, expected_contract=contract())
    assert requested == {"game_ids", "decision_indices", "seats", "action_kinds"}


def test_loader_defers_full_decode_and_keeps_only_two_cached_shards(sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original_decode = imitation_module._read_shard

    def tracked_decode(path: Path, *args, **kwargs):
        calls.append(path.name)
        return original_decode(path, *args, **kwargs)

    monkeypatch.setattr(imitation_module, "_read_shard", tracked_decode)
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    assert calls == []
    first = dataset._row_data([(0, 0)])
    assert calls == [dataset.shards[0].path.split("/")[-1]]
    first["observations"][0, 0] = -999.0
    second = dataset._row_data([(0, 0)])
    assert calls == [dataset.shards[0].path.split("/")[-1]]
    assert second["observations"][0, 0] != -999.0
    dataset._row_data([(1, 0)])
    dataset._row_data([(2, 0)])
    assert len(dataset._cache._entries) == 2
    dataset._row_data([(0, 0)])
    assert calls.count(dataset.shards[0].path.split("/")[-1]) == 2

def test_first_sampler_batch_decodes_only_its_selected_shard(sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original_decode = imitation_module._read_shard
    monkeypatch.setattr(imitation_module, "_read_shard", lambda path, *args, **kwargs: calls.append(path.name) or original_decode(path, *args, **kwargs))
    batch = StratifiedDecisionSampler(load_imitation_dataset(sampled_dataset, expected_contract=contract()), batch_size=1, seed=37).next_batch()
    assert len(calls) == 1
    assert batch.game_ids.shape == (1,)


def test_matching_hash_invalid_payload_is_rejected_on_first_row_use(sampled_dataset: Path) -> None:
    manifest_path = sampled_dataset / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    shard = sampled_dataset / manifest["shards"][0]["path"]
    with np.load(shard, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["actions"] = np.full(len(arrays["actions"]), 1, dtype=np.int64)
    np.savez_compressed(shard, **arrays)
    manifest["shards"][0]["sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    with pytest.raises(ValueError, match="illegal action"):
        dataset._row_data([(0, 0)])




@pytest.fixture
def clone_scenario():
    return resolve_scenario(environment="tactical-v2", scenario_file=None, template_id="tactical-v2-standard")


@pytest.fixture
def clone_dataset(tmp_path: Path, clone_scenario) -> Path:
    writer = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=2)
    scenario_hash = hashlib.sha256(clone_scenario.canonical_json.encode("utf-8")).hexdigest()
    writer.append_game(
        _staged_game(tmp_path, "train", "greedy", "standard-3v3", 11_000_100, 0, scenario_hash),
        [decision(index, 0) for index in range(3)],
    )
    writer.append_game(
        _staged_game(tmp_path, "train", "bounded-search", "conversion-3v1-near", 11_500_100, 1, scenario_hash),
        [{**decision(index + 3, 1), "decision_index": index, "action_kind": 2} for index in range(2)],
    )
    writer.append_game(
        _staged_game(tmp_path, "validation", "greedy", "standard-3v3", 12_000_100, 0, scenario_hash),
        [{**decision(5, 0), "decision_index": 0}],
    )
    writer.append_game(
        _staged_game(tmp_path, "validation", "bounded-search", "conversion-3v1-near", 12_000_101, 1, scenario_hash),
        [{**decision(6, 1), "decision_index": 0, "action_kind": 3}],
    )
    writer.close()
    return tmp_path

class _TinyCloneEnv(gym.Env):
    observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
    action_space = spaces.Discrete(5)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(3, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(3, dtype=np.float32), 0.0, True, False, {}

    def action_masks(self):
        return np.asarray([True, False, True, False, False], dtype=bool)


def _test_masked_logits(model, observations: np.ndarray, masks: np.ndarray) -> torch.Tensor:
    with torch.no_grad():
        observation_tensor = torch.as_tensor(observations, dtype=torch.float32, device=model.device)
        mask_tensor = torch.as_tensor(masks, dtype=torch.bool, device=model.device)
        distribution = model.policy.get_distribution(observation_tensor, action_masks=mask_tensor)
        return distribution.distribution.logits.detach().cpu()


def test_behavioral_clone_overfits_a_five_example_masked_dataset_and_publishes_a_resolvable_run(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    run_dir = tmp_path / "bc"
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=run_dir,
        config=BehavioralCloningConfig(model_seed=211, batch_size=5, learning_rate=3e-4, max_epochs=200, patience=200),
    )

    assert result.validation.top1_accuracy == pytest.approx(1.0)
    assert result.validation.illegal_probability == pytest.approx(0.0)
    assert result.best_epoch <= result.epochs_trained <= 200
    expected_files = {
        "run.json", "scenario.json", "bc.json", "metrics.json", "actor-fixtures.npz",
        "checkpoints/step_000000000.zip",
    }
    assert {path.relative_to(run_dir).as_posix() for path in run_dir.rglob("*") if path.is_file()} == expected_files

    manifest = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    bc = json.loads((run_dir / "bc.json").read_text(encoding="utf-8"))
    metrics = json.loads(
        (run_dir / "metrics.json").read_text(encoding="utf-8"),
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
    )
    dataset_hash = hashlib.sha256((clone_dataset / "manifest.json").read_bytes()).hexdigest()
    assert manifest["config"]["algorithm"] == "maskable_ppo"
    assert manifest["config"]["policy"] == "HexCNN"
    assert manifest["latest_checkpoint_step"] == 0
    assert manifest["contract"] == contract().to_dict()
    assert manifest["dataset_manifest_sha256"] == dataset_hash
    assert manifest["bc_config"]["model_seed"] == manifest["model_seed"] == 211
    assert manifest["best_epoch"] == result.best_epoch
    assert manifest["scenario"] == {
        "path": "scenario.json",
        "template_id": clone_scenario.template_id,
        "schema_version": clone_scenario.schema_version,
    }
    assert (run_dir / "scenario.json").read_text(encoding="utf-8") == clone_scenario.canonical_json + "\n"
    published_scenario = resolve_scenario(environment="tactical-v2", scenario_file=run_dir / "scenario.json", template_id=None)
    assert published_scenario.canonical_json == clone_scenario.canonical_json
    assert bc["dataset_manifest_sha256"] == dataset_hash
    assert bc["training_decision_count"] == 5
    assert bc["validation_decision_count"] == 2
    assert bc["validation_game_count"] == 2
    assert bc["value_parameters_sha256_before"] == bc["value_parameters_sha256_after"]
    assert bc["actor_parameter_count"] > 0 and bc["value_parameter_count"] > 0
    assert metrics == asdict(result.validation)
    assert {"teacher/greedy", "teacher/bounded-search", "profile/standard-3v3",
            "profile/conversion-3v1-near", "action_kind/move", "action_kind/deploy",
            "seat/0", "seat/1"} <= set(metrics["strata"])

    with np.load(run_dir / "actor-fixtures.npz", allow_pickle=False) as fixtures:
        assert set(fixtures.files) == {"observations", "legal_masks"}
        observations = fixtures["observations"]
        masks = fixtures["legal_masks"]
        assert observations.dtype == np.float32 and masks.dtype == np.bool_
        assert observations.shape == (2, 3) and masks.shape == (2, 5)
        assert set(observations[:, 0]) == {5.0, 6.0}

    first = ControllerResolver(contract()).resolve(f"run:{run_dir}")
    second = ControllerResolver(contract()).resolve(f"run:{run_dir}")
    assert isinstance(first.model.policy.features_extractor, HexCNN)
    torch.testing.assert_close(
        _test_masked_logits(first.model, observations, masks),
        _test_masked_logits(second.model, observations, masks),
        rtol=0,
        atol=0,
    )
    checkpoint = run_dir / "checkpoints" / "step_000000000.zip"
    with zipfile.ZipFile(checkpoint) as archive:
        assert not any("bc_optimizer" in name.lower() for name in archive.namelist())
        optimizer_state = torch.load(io.BytesIO(archive.read("policy.optimizer.pth")), map_location="cpu", weights_only=True)
        assert optimizer_state["state"] == {}


def test_smoke_validation_view_reuses_training_rows_without_mutating_dataset(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    """Using held-out rows would hide that smoke validates plumbing, not generalization."""

    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    validation_view = imitation_module.training_rows_as_validation(dataset)

    result = train_behavioral_clone(
        dataset=validation_view,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "smoke-bc",
        config=BehavioralCloningConfig(
            model_seed=211,
            batch_size=5,
            learning_rate=3e-4,
            max_epochs=1,
            patience=1,
        ),
    )

    bc = json.loads((result.run_dir / "bc.json").read_text(encoding="utf-8"))
    assert bc["training_decision_count"] == 5
    assert bc["validation_decision_count"] == 5
    assert bc["validation_game_count"] == 2
    assert dataset.index["validation"] is not dataset.index["train"]
    assert validation_view.index["validation"] is validation_view.index["train"]


def test_dataset_audit_reopens_all_labels_masks_round_trips_and_replays(
    clone_dataset: Path,
) -> None:
    """Manifest counts alone must not satisfy the smoke collection checks."""

    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())

    assert imitation_module.audit_imitation_dataset(dataset) == {
        "games": 4,
        "teacher_labels": 7,
        "masked_labels": 0,
        "round_trip_mismatches": 0,
        "replay_mismatches": 0,
    }


def _batch_for_metrics(action_size: int = 2) -> ImitationBatch:
    return ImitationBatch(
        observations=np.asarray([[0.0, 1.0, 2.0], [1.0, 1.0, 2.0]], dtype=np.float32),
        legal_masks=np.asarray([[True] * action_size, [False] + [True] * (action_size - 1)], dtype=bool),
        actions=np.asarray([0, action_size - 1], dtype=np.int64),
        game_ids=np.asarray([0, 1], dtype=np.int64),
        decision_indices=np.asarray([0, 0], dtype=np.int32),
        sources=np.asarray([Source.GREEDY_STANDARD, Source.SEARCH_CONVERSION], dtype=object),
        profiles=np.asarray(["standard-3v3", "conversion-3v1-near"], dtype=object),
        seats=np.asarray([0, 1], dtype=np.uint8),
        action_kinds=np.asarray([0, 1], dtype=np.uint8),
        partitions=np.asarray(["validation", "validation"], dtype=object),
    )


def test_fixture_selection_is_sorted_and_keeps_non_end_turn_and_both_available_seats() -> None:
    count = 40
    batch = ImitationBatch(
        observations=np.arange(count * 3, dtype=np.float32).reshape(count, 3),
        legal_masks=np.ones((count, 5), dtype=bool),
        actions=np.asarray([0] * 38 + [2, 0], dtype=np.int64),
        game_ids=np.zeros(count, dtype=np.int64),
        decision_indices=np.arange(count, dtype=np.int32),
        sources=np.full(count, Source.GREEDY_STANDARD, dtype=object),
        profiles=np.full(count, "standard-3v3", dtype=object),
        seats=np.asarray([0] * 39 + [1], dtype=np.uint8),
        action_kinds=np.asarray([0] * 38 + [1, 0], dtype=np.uint8),
        partitions=np.full(count, "validation", dtype=object),
    )

    fixtures = imitation_module._fixture_batch(batch)

    assert len(fixtures.actions) == 32
    assert np.all(fixtures.decision_indices[:-1] <= fixtures.decision_indices[1:])
    assert 38 in fixtures.decision_indices and 39 in fixtures.decision_indices
    assert set(fixtures.seats) == {0, 1}
    assert np.any(fixtures.actions != 0)
    with pytest.raises(ValueError, match="non-EndTurn"):
        imitation_module._fixture_batch(ImitationBatch(**{
            name: (np.zeros_like(value) if name == "actions" else value.copy())
            for name, value in batch.__dict__.items()
        }))


class _TwoActionEnv(_TinyCloneEnv):
    action_space = spaces.Discrete(2)

    def action_masks(self):
        return np.asarray([True, True], dtype=bool)


def test_clone_metrics_clamp_top_k_for_small_action_spaces_and_remain_finite() -> None:
    model = MaskablePPOAdapter().create(
        _TwoActionEnv(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        seed=13,
        device="cpu",
        checkpoint_interval=2,
    )

    metrics = imitation_module._clone_metrics(model, _batch_for_metrics())

    assert metrics.top3_accuracy == pytest.approx(1.0)
    assert metrics.top5_accuracy == pytest.approx(1.0)
    assert metrics.illegal_probability == pytest.approx(0.0)
    assert all(np.isfinite(value) for name, value in asdict(metrics).items() if name != "strata")


def test_clone_publication_failure_never_exposes_a_partial_run(clone_dataset: Path, clone_scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    run_dir = tmp_path / "failed"
    monkeypatch.setattr(
        imitation_module,
        "_verify_reload_identity",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("injected reload failure")),
    )

    with pytest.raises(RuntimeError, match="injected reload failure"):
        train_behavioral_clone(
            dataset=dataset,
            scenario=clone_scenario,
            env=_TinyCloneEnv(),
            contract=contract(),
            spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
            run_dir=run_dir,
            config=BehavioralCloningConfig(model_seed=17, batch_size=5, max_epochs=1, patience=1),
        )

    assert not run_dir.exists()
    assert list(tmp_path.glob(".failed.publishing-*")) == []


def test_clone_training_is_seed_deterministic_and_validation_is_not_optimized(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    config = BehavioralCloningConfig(model_seed=29, batch_size=5, max_epochs=3, patience=3)
    results = [
        train_behavioral_clone(
            dataset=dataset,
            scenario=clone_scenario,
            env=_TinyCloneEnv(),
            contract=contract(),
            spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
            run_dir=tmp_path / name,
            config=config,
        )
        for name in ("first", "second")
    ]

    assert asdict(results[0].validation) == asdict(results[1].validation)
    assert results[0].best_epoch == results[1].best_epoch
    first_bc = json.loads((results[0].run_dir / "bc.json").read_text(encoding="utf-8"))
    second_bc = json.loads((results[1].run_dir / "bc.json").read_text(encoding="utf-8"))
    assert first_bc["training_decision_count"] == second_bc["training_decision_count"] == 5
    assert first_bc["validation_decision_count"] == second_bc["validation_decision_count"] == 2
    with np.load(results[0].run_dir / "actor-fixtures.npz", allow_pickle=False) as first_fixtures:
        observations, masks = first_fixtures["observations"], first_fixtures["legal_masks"]
    first_model = ControllerResolver(contract()).resolve(f"run:{results[0].run_dir}").model
    second_model = ControllerResolver(contract()).resolve(f"run:{results[1].run_dir}").model
    torch.testing.assert_close(
        _test_masked_logits(first_model, observations, masks),
        _test_masked_logits(second_model, observations, masks),
        rtol=0,
        atol=0,
    )


def test_behavioral_cloning_defaults_are_the_reviewed_production_limits() -> None:
    config = BehavioralCloningConfig()
    assert (config.batch_size, config.learning_rate, config.max_epochs, config.patience) == (256, 3e-4, 50, 5)
    with pytest.raises(ValueError, match="finite"):
        BehavioralCloningConfig(learning_rate=float("nan"))


def test_clone_rejects_any_validation_row_before_an_optimizer_step(clone_dataset: Path, clone_scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    original = StratifiedDecisionSampler.next_batch

    def leak_validation(sampler: StratifiedDecisionSampler) -> ImitationBatch:
        batch = original(sampler)
        values = {name: getattr(batch, name).copy() for name in ImitationBatch.__dataclass_fields__}
        values["partitions"][:] = "validation"
        return ImitationBatch(**values)

    monkeypatch.setattr(StratifiedDecisionSampler, "next_batch", leak_validation)
    run_dir = tmp_path / "leaked"
    with pytest.raises(RuntimeError, match="validation rows"):
        train_behavioral_clone(
            dataset=dataset,
            scenario=clone_scenario,
            env=_TinyCloneEnv(),
            contract=contract(),
            spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
            run_dir=run_dir,
            config=BehavioralCloningConfig(model_seed=31, batch_size=5, max_epochs=1, patience=1),
        )
    assert not run_dir.exists()


def test_clone_rejects_scenario_hash_and_environment_mismatches_before_writing(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    common = {
        "dataset": dataset,
        "env": _TinyCloneEnv(),
        "contract": contract(),
        "spaces_info": {"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        "config": BehavioralCloningConfig(model_seed=37, batch_size=5, max_epochs=1, patience=1),
    }
    other_tactical = resolve_scenario(
        environment="tactical-v2",
        scenario_file=None,
        template_id="tactical-v2-long-battle",
    )
    with pytest.raises(ContractMismatch, match="scenario hash"):
        train_behavioral_clone(
            **common,
            scenario=other_tactical,
            run_dir=tmp_path / "hash-mismatch",
        )
    adaptive = resolve_scenario(
        environment="adaptive-v1",
        scenario_file=None,
        template_id="adaptive-standard",
    )
    with pytest.raises(ContractMismatch, match="scenario environment"):
        train_behavioral_clone(
            **common,
            scenario=adaptive,
            run_dir=tmp_path / "environment-mismatch",
        )
    assert not (tmp_path / "hash-mismatch").exists()
    assert not (tmp_path / "environment-mismatch").exists()
