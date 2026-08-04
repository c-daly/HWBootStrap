from __future__ import annotations

import hashlib
import io
import json
import math
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any
from types import MappingProxyType
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
from ml_lab.imitation import BehavioralCloningConfig, DemonstrationGame, DemonstrationWriter, ImitationBatch, MaterializedImitationPartition, Source, StratifiedDecisionSampler, benchmark_imitation_sampler, load_imitation_dataset, masked_cross_entropy, materialize_imitation_partition, resolve_behavioral_cloning_device, train_behavioral_clone, validate_decision


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


def _physical_batch_for_scheduled_refs(
    dataset,
    refs_and_sources: list[tuple[tuple[int, int], Source]],
) -> ImitationBatch:
    refs = [ref for ref, _source in refs_and_sources]
    rows = dataset._row_data(refs)
    legal_masks = np.unpackbits(
        rows["packed_masks"], axis=1,
        count=dataset.contract.action_size, bitorder="little",
    ).astype(bool, copy=False)
    metadata = [dataset.games[int(game_id)] for game_id in rows["game_ids"]]
    return ImitationBatch(
        observations=rows["observations"].copy(),
        legal_masks=legal_masks.copy(),
        actions=rows["actions"].copy(),
        game_ids=rows["game_ids"].copy(),
        decision_indices=rows["decision_indices"].copy(),
        sources=np.asarray([source for _ref, source in refs_and_sources], dtype=object),
        profiles=np.asarray([game["profile"] for game in metadata], dtype=object),
        seats=rows["seats"].copy(),
        action_kinds=rows["action_kinds"].copy(),
        partitions=np.asarray([game["partition"] for game in metadata], dtype=object),
    )


def test_sampler_keeps_70_30_ratio_is_seeded_and_excludes_validation_rows(sampled_dataset: Path) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    training = materialize_imitation_partition(dataset, "train")
    first = StratifiedDecisionSampler(dataset, training, batch_size=100, standard_fraction=0.70, seed=211).next_batch()
    second = StratifiedDecisionSampler(dataset, training, batch_size=100, standard_fraction=0.70, seed=211).next_batch()
    different = StratifiedDecisionSampler(dataset, training, batch_size=100, standard_fraction=0.70, seed=212).next_batch()
    assert (first.sources == Source.GREEDY_STANDARD).sum() == 70
    assert (first.sources == Source.SEARCH_CONVERSION).sum() == 30
    assert list(zip(first.game_ids, first.decision_indices)) == list(zip(second.game_ids, second.decision_indices))
    assert list(zip(first.game_ids, first.decision_indices)) != list(zip(different.game_ids, different.decision_indices))
    assert set(first.partitions) == {"train"}
    assert first.observations.flags.owndata
    assert np.all(first.legal_masks[np.arange(len(first.actions)), first.actions])


def _three_source_partition(
    *, targeted_rows: int = 4, targeted_metadata_variant: bool = False,
) -> MaterializedImitationPartition:
    sources = [
        *([Source.GREEDY_STANDARD] * 6),
        *([Source.SEARCH_CONVERSION] * 5),
        *([imitation_module.Source.DAGGER_TARGETED] * targeted_rows),
    ]
    count = len(sources)
    profiles = [
        *("standard-3v3" for _ in range(6)),
        *("conversion-3v1-near" for _ in range(5)),
        *(
            ("conversion-1v1-far" if targeted_metadata_variant and index % 2 else "standard-3v3")
            for index in range(targeted_rows)
        ),
    ]
    seats = np.asarray(
        [index % 2 for index in range(11)]
        + [((index + 1) % 2 if targeted_metadata_variant else 0) for index in range(targeted_rows)],
        dtype=np.int32,
    )
    action_kinds = np.asarray(
        [index % 4 for index in range(11)]
        + [((index + 2) % 4 if targeted_metadata_variant else 1) for index in range(targeted_rows)],
        dtype=np.int32,
    )
    game_ids = np.arange(count, dtype=np.int64)
    decision_indices = np.zeros(count, dtype=np.int32)
    batch = ImitationBatch(
        observations=np.asarray(
            [[float(index), 1.0, 2.0] for index in range(count)], dtype=np.float32,
        ),
        legal_masks=np.ones((count, 5), dtype=bool),
        actions=np.full(count, 2, dtype=np.int64),
        game_ids=game_ids,
        decision_indices=decision_indices,
        sources=np.asarray(sources, dtype=object),
        profiles=np.asarray(profiles, dtype=object),
        seats=seats,
        action_kinds=action_kinds,
        partitions=np.full(count, "train", dtype=object),
    )
    identities = MappingProxyType({
        (f"component-{index:02d}", int(game_ids[index]), 0): index
        for index in range(count)
    })
    return MaterializedImitationPartition("train", batch, identities)


def _locked_dagger_mixture() -> MappingProxyType:
    return MappingProxyType(OrderedDict((
        (Source.GREEDY_STANDARD, 0.49),
        (Source.SEARCH_CONVERSION, 0.21),
        (imitation_module.Source.DAGGER_TARGETED, 0.30),
    )))


@pytest.mark.parametrize(
    "fractions",
    [
        OrderedDict((
            (Source.GREEDY_STANDARD, 0.49),
            (Source.SEARCH_CONVERSION, 0.21),
            (imitation_module.Source.DAGGER_TARGETED, 0.0),
        )),
        OrderedDict((
            (Source.GREEDY_STANDARD, 0.49),
            (Source.SEARCH_CONVERSION, float("nan")),
            (imitation_module.Source.DAGGER_TARGETED, 0.30),
        )),
        OrderedDict((
            (Source.GREEDY_STANDARD, 0.40),
            (Source.SEARCH_CONVERSION, 0.20),
            (imitation_module.Source.DAGGER_TARGETED, 0.30),
        )),
    ],
)
def test_source_mixture_sampler_rejects_nonpositive_nonfinite_or_unbalanced_fractions(
    fractions: OrderedDict,
) -> None:
    """Relaxing fraction validation must permit undefined long-run source exposure."""

    with pytest.raises(ValueError, match="source fractions"):
        imitation_module.SourceMixtureSampler(
            _three_source_partition(),
            source_fractions=fractions,
            batch_size=256,
            seed=227,
        )


def test_source_mixture_sampler_freezes_the_ordered_fraction_mapping() -> None:
    """Retaining the caller's mutable mapping must let later mutation alter provenance."""

    fractions = OrderedDict(_locked_dagger_mixture())
    sampler = imitation_module.SourceMixtureSampler(
        _three_source_partition(),
        source_fractions=fractions,
        batch_size=256,
        seed=227,
    )
    fractions[Source.GREEDY_STANDARD] = 0.10
    assert list(sampler.source_fractions.items()) == list(
        _locked_dagger_mixture().items()
    )
    with pytest.raises(TypeError):
        sampler.source_fractions[Source.GREEDY_STANDARD] = 0.10


def test_source_mixture_sampler_residual_accounts_nonintegral_256_batches() -> None:
    """Allocating rounding remainder to a fixed source must drift from 49/21/30."""

    materialized = _three_source_partition()
    sampler = imitation_module.SourceMixtureSampler(
        materialized,
        source_fractions=_locked_dagger_mixture(),
        batch_size=256,
        seed=227,
    )
    batches = [sampler.next_batch() for _ in range(40)]

    first_counts = [
        int(np.count_nonzero(batches[0].sources == source))
        for source in _locked_dagger_mixture()
    ]
    second_counts = [
        int(np.count_nonzero(batches[1].sources == source))
        for source in _locked_dagger_mixture()
    ]
    totals = [
        sum(int(np.count_nonzero(batch.sources == source)) for batch in batches)
        for source in _locked_dagger_mixture()
    ]
    assert first_counts == [125, 54, 77]
    assert second_counts == [126, 53, 77]
    assert totals == [5_018, 2_150, 3_072]
    assert sum(totals) == 10_240
    assert list(batches[0].sources) != [
        *([Source.GREEDY_STANDARD] * 125),
        *([Source.SEARCH_CONVERSION] * 54),
        *([imitation_module.Source.DAGGER_TARGETED] * 77),
    ]

    repeated = imitation_module.SourceMixtureSampler(
        materialized,
        source_fractions=_locked_dagger_mixture(),
        batch_size=256,
        seed=227,
    ).next_batch()
    different = imitation_module.SourceMixtureSampler(
        materialized,
        source_fractions=_locked_dagger_mixture(),
        batch_size=256,
        seed=228,
    ).next_batch()
    np.testing.assert_array_equal(repeated.observations, batches[0].observations)
    assert not np.array_equal(different.observations, batches[0].observations)


def test_targeted_sampler_is_uniform_before_repeat_and_ignores_row_metadata() -> None:
    """Stratifying targeted labels by profile, seat, or action kind must bias exposure."""

    def targeted_sequence(materialized: MaterializedImitationPartition) -> list[int]:
        sampler = imitation_module.SourceMixtureSampler(
            materialized,
            source_fractions=_locked_dagger_mixture(),
            batch_size=10,
            seed=227,
        )
        selected: list[int] = []
        while len(selected) < 404:
            batch = sampler.next_batch()
            selected.extend(
                int(value)
                for value in batch.observations[
                    batch.sources == imitation_module.Source.DAGGER_TARGETED, 0
                ]
            )
        return selected[:404]

    ordinary = targeted_sequence(_three_source_partition(targeted_rows=3))
    changed_metadata = targeted_sequence(
        _three_source_partition(
            targeted_rows=3, targeted_metadata_variant=True,
        )
    )
    assert ordinary == changed_metadata
    assert len(set(ordinary[:3])) == 3
    assert ordinary[3] in set(ordinary[:3])
    counts = Counter(ordinary[:300])
    assert set(counts.values()) == {100}


def test_legacy_sampler_default_sequence_remains_byte_for_byte_stable(
    sampled_dataset: Path,
) -> None:
    """Routing the compatibility wrapper through new allocation must not alter old runs."""

    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    materialized = materialize_imitation_partition(dataset, "train")
    sampler = StratifiedDecisionSampler(dataset, materialized, batch_size=7, seed=211)

    actual = [
        [
            (int(game_id), int(decision_index), source.value)
            for game_id, decision_index, source in zip(
                batch.game_ids, batch.decision_indices, batch.sources, strict=True,
            )
        ]
        for batch in (sampler.next_batch() for _ in range(4))
    ]
    assert actual == [
        [(0, 0, "greedy_standard"), (1, 2, "greedy_standard"), (3, 0, "search_conversion"), (0, 2, "greedy_standard"), (2, 1, "search_conversion"), (3, 2, "search_conversion"), (1, 1, "greedy_standard")],
        [(1, 0, "greedy_standard"), (3, 1, "search_conversion"), (0, 0, "greedy_standard"), (1, 1, "greedy_standard"), (0, 2, "greedy_standard"), (0, 1, "greedy_standard"), (2, 0, "search_conversion")],
        [(1, 2, "greedy_standard"), (1, 0, "greedy_standard"), (1, 1, "greedy_standard"), (2, 2, "search_conversion"), (3, 2, "search_conversion"), (0, 1, "greedy_standard"), (0, 2, "greedy_standard")],
        [(3, 0, "search_conversion"), (2, 1, "search_conversion"), (0, 1, "greedy_standard"), (0, 0, "greedy_standard"), (0, 2, "greedy_standard"), (1, 2, "greedy_standard"), (1, 0, "greedy_standard")],
    ]


@pytest.mark.filterwarnings("error:The given NumPy array is not writable")
def test_actor_supervision_corpus_never_optimizes_validation_sentinel(
    clone_scenario, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Using one combined partition for training and metrics must leak sentinel action 4."""

    validation_batch = ImitationBatch(
        observations=np.asarray(
            [[40.0, 1.0, 2.0], [41.0, 1.0, 2.0]], dtype=np.float32,
        ),
        legal_masks=np.ones((2, 5), dtype=bool),
        actions=np.asarray([4, 4], dtype=np.int64),
        game_ids=np.asarray([0, 1], dtype=np.int64),
        decision_indices=np.zeros(2, dtype=np.int32),
        sources=np.asarray(
            [imitation_module.Source.DAGGER_TARGETED] * 2, dtype=object,
        ),
        profiles=np.asarray(["standard-3v3", "conversion-1v1-far"], dtype=object),
        seats=np.asarray([0, 1], dtype=np.int32),
        action_kinds=np.asarray([3, 3], dtype=np.int32),
        partitions=np.full(2, "validation", dtype=object),
    )
    validation = MaterializedImitationPartition(
        "validation",
        validation_batch,
        MappingProxyType({
            ("e" * 64, 0, 0): 0,
            ("e" * 64, 1, 0): 1,
        }),
    )
    scenario_hash = hashlib.sha256(
        clone_scenario.canonical_json.encode("utf-8")
    ).hexdigest()
    corpus = imitation_module.ActorSupervisionCorpus(
        training=_three_source_partition(),
        validation=validation,
        source_fractions=_locked_dagger_mixture(),
        identity=MappingProxyType({
            "schema_version": 1,
            "kind": "test-corpus",
            "base_manifest_sha256": "d" * 64,
            "contract_hash": contract().contract_hash,
            "encoding_hash": contract().encoding_hash,
            "scenario_hash": scenario_hash,
        }),
    )
    observed: dict[str, list[int]] = {"train": [], "validation": []}
    original_distribution = imitation_module._distribution_tensors
    original_metrics = imitation_module._clone_metrics

    def record_distribution(model, batch: ImitationBatch):
        if set(batch.partitions) == {"train"}:
            observed["train"].extend(int(action) for action in batch.actions)
        return original_distribution(model, batch)

    def record_metrics(model, batch: ImitationBatch):
        observed["validation"].extend(int(action) for action in batch.actions)
        return original_metrics(model, batch)

    monkeypatch.setattr(
        imitation_module, "_distribution_tensors", record_distribution,
    )
    monkeypatch.setattr(imitation_module, "_clone_metrics", record_metrics)
    result = imitation_module.train_actor_supervision(
        corpus=corpus,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "mixed-supervision",
        config=BehavioralCloningConfig(
            model_seed=227,
            batch_size=10,
            learning_rate=3e-4,
            max_epochs=1,
            patience=1,
            device="cpu",
        ),
    )

    assert observed["train"] and 4 not in observed["train"]
    assert observed["validation"] and set(observed["validation"]) == {4}
    assert result.validation.strata["teacher/dagger-targeted"]["count"] == 2
    bc = json.loads((result.run_dir / "bc.json").read_text(encoding="utf-8"))
    assert bc["source_fractions"] == {
        "greedy_standard": 0.49,
        "search_conversion": 0.21,
        "dagger_targeted": 0.30,
    }


def test_sampler_benchmark_runs_exact_batches_and_reports_checksum(
    sampled_dataset: Path,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    result = benchmark_imitation_sampler(
        dataset, batch_size=100, seed=211, batches=4,
    )
    assert result["schema_version"] == 1
    assert result["batches"] == 4
    assert result["examples"] == 400
    assert result["examples_per_second"] > 0
    assert result["materialization_seconds"] >= 0
    assert result["sampling_seconds"] > 0
    assert len(result["sequence_sha256"]) == 64


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

def test_materialization_decodes_shards_but_sampler_batches_do_not(sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    original_decode = imitation_module._read_shard
    monkeypatch.setattr(imitation_module, "_read_shard", lambda path, *args, **kwargs: calls.append(path.name) or original_decode(path, *args, **kwargs))
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    training = materialize_imitation_partition(dataset, "train")
    assert calls
    calls.clear()
    batch = StratifiedDecisionSampler(dataset, training, batch_size=1, seed=37).next_batch()
    assert calls == []
    assert batch.game_ids.shape == (1,)


def test_materialized_sampler_matches_physical_batches_and_never_rereads(
    sampled_dataset: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    materialized = materialize_imitation_partition(dataset, "train")
    oracle_scheduler = StratifiedDecisionSampler(
        dataset, materialized, batch_size=100,
        standard_fraction=0.70, seed=211,
    )
    optimized = StratifiedDecisionSampler(
        dataset, materialized, batch_size=100,
        standard_fraction=0.70, seed=211,
    )
    expected_batches = [
        _physical_batch_for_scheduled_refs(
            dataset, oracle_scheduler._next_refs_and_sources()
        )
        for _ in range(4)
    ]
    monkeypatch.setattr(
        dataset._cache, "get",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("physical shard access entered sampler hot path")
        ),
    )
    actual_batches = [optimized.next_batch() for _ in range(4)]
    for expected, actual in zip(expected_batches, actual_batches, strict=True):
        for field in ImitationBatch.__dataclass_fields__:
            np.testing.assert_array_equal(
                getattr(actual, field), getattr(expected, field)
            )


def test_materialized_partition_rejects_wrong_partition_and_missing_reference(
    sampled_dataset: Path,
) -> None:
    dataset = load_imitation_dataset(sampled_dataset, expected_contract=contract())
    validation = materialize_imitation_partition(dataset, "validation")
    with pytest.raises(ValueError, match="partition"):
        StratifiedDecisionSampler(
            dataset, validation, batch_size=1, partition="train", seed=211,
        )
    train = materialize_imitation_partition(dataset, "train")
    with pytest.raises(ValueError, match="reference map"):
        MaterializedImitationPartition(
            partition=train.partition,
            batch=train.batch,
            offsets=MappingProxyType(dict(list(train.offsets.items())[1:])),
        )


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


def test_canonical_cpu_artifact_overfits_a_five_example_masked_dataset_and_publishes_a_resolvable_run(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    run_dir = tmp_path / "bc"
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=run_dir,
        config=BehavioralCloningConfig(model_seed=211, batch_size=5, learning_rate=3e-4, max_epochs=200, patience=200, device="cpu"),
    )

    assert result.validation.top1_accuracy == pytest.approx(1.0)
    assert result.validation.illegal_probability == pytest.approx(0.0)
    assert result.best_epoch <= result.epochs_trained <= 200
    expected_files = {
        "run.json", "scenario.json", "bc.json", "metrics.json", "actor-fixtures.npz",
        "training-history.json",
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
    assert bc["publication_device"] == "cpu"
    assert bc["training_device"]["requested"] == "cpu"
    assert bc["training_device"]["resolved"] == "cpu"
    assert {parameter.device.type for parameter in first.model.policy.parameters()} == {"cpu"}
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
            device="cpu",
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
            config=BehavioralCloningConfig(model_seed=17, batch_size=5, max_epochs=1, patience=1, device="cpu"),
        )

    assert not run_dir.exists()
    assert list(tmp_path.glob(".failed.publishing-*")) == []


def test_clone_training_is_seed_deterministic_and_validation_is_not_optimized(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    config = BehavioralCloningConfig(model_seed=29, batch_size=5, max_epochs=3, patience=3, device="cpu")
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
    config = BehavioralCloningConfig(device="cpu")
    assert (config.batch_size, config.learning_rate, config.max_epochs, config.patience) == (256, 3e-4, 50, 5)
    with pytest.raises(ValueError, match="finite"):
        BehavioralCloningConfig(learning_rate=float("nan"), device="cpu")

@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_behavioral_cloning_config_accepts_explicit_supported_device(device: str) -> None:
    assert BehavioralCloningConfig(device=device).device == device


@pytest.mark.parametrize("device", ["", "auto", "cuda:0", "mps", "CPU"])
def test_behavioral_cloning_config_rejects_unlocked_device(device: str) -> None:
    with pytest.raises(ValueError, match="device"):
        BehavioralCloningConfig(device=device)


def test_cuda_preflight_fails_closed_when_cuda_is_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_behavioral_cloning_device("cuda")


class _CapturedDevice(RuntimeError):
    pass


def test_clone_trainer_passes_requested_device_to_production_adapter(
    clone_dataset: Path,
    clone_scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    captured: dict[str, str] = {}

    def capture_create(
        self, env, *, spaces_info, seed, device, checkpoint_interval, options=None
    ):
        captured["device"] = device
        raise _CapturedDevice(device)

    monkeypatch.setattr(MaskablePPOAdapter, "create", capture_create)
    monkeypatch.setattr(
        imitation_module,
        "resolve_behavioral_cloning_device",
        lambda requested: {
            "requested": requested,
            "resolved": "cuda:0",
            "torch_version": "test",
            "cuda_runtime": "test",
            "device_index": 0,
            "device_name": "test-gpu",
        },
    )

    with pytest.raises(_CapturedDevice, match="cuda"):
        train_behavioral_clone(
            dataset=dataset,
            scenario=clone_scenario,
            env=_TinyCloneEnv(),
            contract=contract(),
            spaces_info={
                "channels": 1, "board_h": 1, "board_w": 1, "globals": 2,
            },
            run_dir=tmp_path / "bc",
            config=BehavioralCloningConfig(
                model_seed=211, batch_size=5, max_epochs=1, patience=1,
                device="cuda",
            ),
        )

    assert captured == {"device": "cuda"}


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
            config=BehavioralCloningConfig(model_seed=31, batch_size=5, max_epochs=1, patience=1, device="cpu"),
        )
    assert not run_dir.exists()


def test_clone_rejects_scenario_hash_and_environment_mismatches_before_writing(clone_dataset: Path, clone_scenario, tmp_path: Path) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    common = {
        "dataset": dataset,
        "env": _TinyCloneEnv(),
        "contract": contract(),
        "spaces_info": {"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        "config": BehavioralCloningConfig(model_seed=37, batch_size=5, max_epochs=1, patience=1, device="cpu"),
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


def test_clone_trainer_emits_finite_epoch_and_completion_progress(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    events: list[dict[str, Any]] = []
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "progress-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=1, patience=1,
            device="cpu",
        ),
        progress=events.append,
    )

    assert [event["event"] for event in events] == ["bc_epoch", "bc_complete"]
    epoch = events[0]
    assert epoch["schema_version"] == 1
    assert epoch["model_seed"] == 211
    assert epoch["device"] == "cpu"
    assert epoch["epoch"] == epoch["max_epochs"] == 1
    assert epoch["batches"] > 0
    assert epoch["examples"] > 0
    for key in (
        "mean_training_loss", "validation_nll", "top1_accuracy",
        "top3_accuracy", "top5_accuracy", "epoch_seconds",
        "elapsed_seconds", "examples_per_second",
    ):
        assert math.isfinite(epoch[key])
    assert epoch["epoch_seconds"] >= 0
    assert epoch["examples_per_second"] >= 0
    phase_fields = (
        "sampling_seconds",
        "transfer_forward_seconds",
        "optimization_seconds",
        "validation_seconds",
        "unclassified_seconds",
    )
    for key in phase_fields:
        assert math.isfinite(epoch[key])
        assert epoch[key] >= 0
    assert math.isclose(
        sum(epoch[key] for key in phase_fields),
        epoch["epoch_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    assert events[-1]["run_dir"] == str(result.run_dir.resolve())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda event: event.pop("sampling_seconds"),
        lambda event: event.__setitem__("optimization_seconds", -1.0),
        lambda event: event.__setitem__("unclassified_seconds", 9.0),
    ],
)
def test_behavioral_cloning_progress_event_rejects_invalid_phase_timing(mutation) -> None:
    event = {
        "schema_version": 1,
        "model_seed": 211,
        "epoch": 1,
        "max_epochs": 1,
        "batches": 1,
        "examples": 5,
        "mean_training_loss": 1.0,
        "validation_nll": 1.0,
        "top1_accuracy": 0.5,
        "top3_accuracy": 0.75,
        "top5_accuracy": 0.9,
        "best_epoch": 1,
        "best_validation_nll": 1.0,
        "epochs_without_improvement": 0,
        "patience": 1,
        "epoch_seconds": 0.1,
        "elapsed_seconds": 0.1,
        "examples_per_second": 50.0,
        "sampling_seconds": 0.01,
        "transfer_forward_seconds": 0.02,
        "optimization_seconds": 0.03,
        "validation_seconds": 0.04,
        "unclassified_seconds": 0.0,
    }
    mutation(event)

    with pytest.raises(ValueError, match="timing"):
        imitation_module._validate_behavioral_cloning_progress_event(event)


def test_clone_publishes_complete_epoch_history(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "history-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=2, patience=2,
            device="cpu",
        ),
    )
    payload = json.loads(
        (result.run_dir / "training-history.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["model_seed"] == 211
    assert payload["training_device"]["requested"] == "cpu"
    assert len(payload["epochs"]) == result.epochs_trained
    assert [row["epoch"] for row in payload["epochs"]] == list(
        range(1, result.epochs_trained + 1)
    )
    assert payload["epochs"][-1]["best_epoch"] == result.best_epoch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_behavioral_clone_real_cuda_training_publishes_cpu_artifact(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    import run_annihilation_imitation_panel as panel_module

    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    events: list[dict[str, Any]] = []
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "cuda-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=1, patience=1,
            device="cuda",
        ),
        progress=events.append,
    )
    assert events[0]["device"].startswith("cuda:")
    epoch = events[0]
    phase_fields = (
        "sampling_seconds",
        "transfer_forward_seconds",
        "optimization_seconds",
        "validation_seconds",
        "unclassified_seconds",
    )
    for key in phase_fields:
        assert math.isfinite(epoch[key])
        assert epoch[key] >= 0
    assert math.isclose(
        sum(epoch[key] for key in phase_fields),
        epoch["epoch_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    )
    bc = json.loads((result.run_dir / "bc.json").read_text(encoding="utf-8"))
    assert bc["training_device"]["device_name"] == torch.cuda.get_device_name(
        torch.cuda.current_device()
    )
    assert bc["publication_device"] == "cpu"
    resolved = ControllerResolver(contract()).resolve(f"run:{result.run_dir}")
    assert {
        parameter.device.type for parameter in resolved.model.policy.parameters()
    } == {"cpu"}
    assert panel_module._validate_clone_run(
        result.run_dir,
        211,
        {},
        expected_scenario=clone_scenario,
        require_provenance=False,
        expected_dataset_manifest=clone_dataset / "manifest.json",
        expected_device="cuda",
    ) == {
        "contract_hash": contract().contract_hash,
        "encoding_hash": contract().encoding_hash,
    }
    print(json.dumps({
        key: epoch[key]
        for key in ("device", "epoch_seconds", *phase_fields)
    }, sort_keys=True))
