"""Frozen-definition and oracle-preflight tests for selective DAgger."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import ml_lab.dagger as dagger_module
import ml_lab.imitation as imitation_module
from ml_lab.contracts import EnvironmentContract
from ml_lab.dagger import (
    OracleBenchmarkDecision,
    OracleBenchmarkSample,
    OracleCodecEvidence,
    OraclePreflightGameResult,
    load_panel_definition,
    validate_panel_definition,
)
from ml_lab.tactical_trace import (
    CommandFrame,
    EpisodeTrace,
    SeatFrame,
    StateFrame,
    TransitionFrame,
)


REAL_BASE_DATASET_AUDIT = dagger_module._audit_base_dataset
REAL_DATASET_TREE_IDENTITY = dagger_module._dataset_tree_identity
# Callback-driven preflight construction is intentionally private test infrastructure.
run_oracle_preflight = dagger_module._run_oracle_preflight_for_test


ROOT = Path(__file__).resolve().parents[2]
PANEL_ROOT = ROOT / "python" / "panels" / "annihilation-selective-dagger-v1"
PANEL_PATH = PANEL_ROOT / "panel.json"
SEED_BANKS_PATH = PANEL_ROOT / "seed-banks.json"

PROFILES = (
    "conversion-3v1-near",
    "conversion-3v1-far",
    "conversion-2v1-near",
    "conversion-2v1-far",
    "conversion-1v1-near",
    "conversion-1v1-far",
)
PRIVATE_TEST_TRUST = {
    'schema_version': 1,
    'mode': 'private-test-transcript',
    'evidence_class': 'untrusted-test-transcript',
    'engine_authenticated': False,
    'engine_evidence_root': None,
    'task_9_production_seal_required': True,
}
SEED_RANGES = (
    ("train", 1, 18_000_000, 18_099_999),
    ("train", 2, 18_100_000, 18_199_999),
    ("train", 3, 18_200_000, 18_299_999),
    ("oracle_preflight", None, 18_900_000, 18_900_119),
    ("smoke", None, 18_990_000, 18_990_009),
    ("validation", 1, 19_000_000, 19_009_999),
    ("validation", 2, 19_010_000, 19_019_999),
    ("validation", 3, 19_020_000, 19_029_999),
    ("reserved", None, 19_030_000, 19_099_999),
    ("development_evaluation", None, 20_000_000, 20_000_099),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_definitions(tmp_path: Path) -> tuple[Path, Path, dict, dict]:
    target = tmp_path / "annihilation-selective-dagger-v1"
    target.mkdir()
    panel_path = target / "panel.json"
    seeds_path = target / "seed-banks.json"
    shutil.copyfile(PANEL_PATH, panel_path)
    shutil.copyfile(SEED_BANKS_PATH, seeds_path)
    return (
        panel_path,
        seeds_path,
        json.loads(panel_path.read_text(encoding="utf-8")),
        json.loads(seeds_path.read_text(encoding="utf-8")),
    )


def _rewrite(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )


def _content_identity(payload: dict) -> str:
    canonical = {
        key: value for key, value in payload.items() if key != "content_identity"
    }
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _repository_provider(root: Path) -> dict[str, Any]:
    return {
        "root": str(root.resolve()),
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": False,
    }


@pytest.fixture(autouse=True)
def _stub_full_base_dataset_audit(monkeypatch: pytest.MonkeyPatch) -> None:
    """The production boundary performs the expensive full corpus audit once."""

    monkeypatch.setattr(dagger_module, "_audit_base_dataset", lambda definition: {
        "content_sha256": definition.dataset_content_sha256,
        "file_count": definition.dataset_file_count,
        "byte_size": definition.dataset_byte_size,
        "audit": {
            "games": 1980,
            "teacher_labels": 199973,
            "masked_labels": 0,
            "round_trip_mismatches": 0,
            "replay_mismatches": 0,
        },
    })


def _symlink_or_skip_windows_privilege(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as exc:
        if os.name == 'nt' and getattr(exc, 'winerror', None) == 1314:
            pytest.skip(f'Windows symbolic-link privilege is unavailable: {exc}')
        raise


def _windows_directory_junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction coverage runs only on Windows")
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        pytest.fail(f"failed to create test junction: {result.stderr or result.stdout}")


def test_panel_definition_freezes_every_causal_input_and_threshold() -> None:
    """Changing a locked model, corpus, oracle, optimizer, or gate must be visible."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)

    assert definition.panel_id == "annihilation-selective-dagger-v1"
    assert definition.environment == "tactical-v2"
    assert definition.panel_sha256 == _sha256(PANEL_PATH)
    assert definition.seed_banks_sha256 == _sha256(SEED_BANKS_PATH)
    assert definition.scenario_path == (
        ROOT / "python" / "config" / "annihilation-imitation-v1.json"
    ).resolve()
    assert definition.scenario_sha256 == (
        "4f085b8a80f7ba8e450a85dbcceb73e05723ce7b37045f1ddd1ef91d67a95632"
    )
    assert definition.runtime_scenario_sha256 == (
        "00684a8623f3f1deadd8d31cb71a0492441508c34a42d6f5ac6a1f8e662aaaa4"
    )
    assert definition.contract_hash == (
        "7347819c2e68fa2d216dc712afc4785e185ca50d3832487d66589a68eee5a9d6"
    )
    assert definition.encoding_hash == (
        "2f334bc2163fd931d84c004e9dc8f44bae68934e46fbf2ec2c819fa3e297054a"
    )
    assert (definition.observation_size, definition.action_size) == (1292, 1288)
    assert definition.repository_policy == {
        "required_clean": True,
        "identity_fields": ("commit", "source_tree", "dirty"),
        "output_policy": "outside_repository",
    }
    assert definition.starting_learner.to_dict() == {
        "schema_version": 1,
        "source_kind": "snapshot",
        "controller": {
            "kind": "snapshot",
            "path": (
                "C:/Users/cddal/HexWars/python/runs/"
                "bc227-ppo-random-s227-20260802-v2/checkpoints/"
                "step_000038912.zip"
            ),
            "source_run": (
                "C:/Users/cddal/HexWars/python/runs/"
                "bc227-ppo-random-s227-20260802-v2"
            ),
            "algorithm": "maskable_ppo",
            "step": 38_912,
            "inference_mode": "deterministic",
        },
        "checkpoint_sha256": (
            "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
        ),
    }
    assert definition.learner_source_manifest_sha256 == (
        "7f02152c2ea39a08e5e203c0b0ba13928b2ad1847e276cc1b19f53331151ba46"
    )
    assert definition.dataset_manifest_sha256 == (
        "6c9f1fd43cded0691080dd12c390aee086d49b144ebc0207d2f80e6b5a9422c4"
    )
    assert definition.dataset_contract_hash == (
        "2d6984089aa151cee59e10bb37b0d2239e7a0668f34d90e1af64216aaf713edf"
    )
    assert definition.profiles == PROFILES
    assert [
        (
            item.depth, item.expansion_budget, item.use_heuristic,
            item.heuristic_identity,
        )
        for item in definition.oracle_candidates
    ] == [
        (4, 512, True, "material-plus-pursuit-v1"),
        (4, 2_048, True, "material-plus-pursuit-v1"),
    ]
    assert definition.preflight == {
        "maps_per_profile": 20,
        "games_per_candidate": 240,
        "queries_per_sample": 2,
        "pooled_win_rate_minimum_basis_points": 8500,
        "labels_per_second_minimum": 10.0,
        "tie_break": (
            "higher_win_rate",
            "fewer_cycling_draws",
            "higher_throughput",
            "smaller_expansion_budget",
        ),
    }
    assert definition.collection == {
        "iterations": 3,
        "train_label_target": 20_000,
        "train_game_ceiling": 2_000,
        "validation_label_target": 2_000,
        "validation_game_ceiling": 200,
        "standard_basis_points": 7_000,
        "conversion_basis_points": 3_000,
        "opponent": "random",
        "both_seats": True,
    }
    assert definition.training == {
        "source_mixture_basis_points": {
            "greedy_standard": 4_900,
            "search_conversion": 2_100,
            "dagger_targeted": 3_000,
        },
        "batch_size": 256,
        "learning_rate": 3e-4,
        "max_epochs": 50,
        "patience": 5,
        "model_seed": 227,
        "sampler_seed": 227,
        "device": "cuda",
        "publication_device": "cpu",
        "objective": "actor_only_masked_cross_entropy",
        "validation_metric": "targeted_negative_log_likelihood",
    }
    assert definition.success == {
        "win_rate_gain_minimum_basis_points": 2_000,
        "absolute_win_rate_minimum_basis_points": 6_500,
        "cycling_relative_reduction_minimum_basis_points": 5_000,
        "replicate_win_rate_minimum_basis_points": 6_500,
        "pooled_replication_win_rate_minimum_basis_points": 7_000,
    }


def test_seed_definition_freezes_all_disjoint_banks_and_reciprocal_preflight() -> None:
    """Range drift or non-reciprocal conversion coverage invalidates the experiment."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)

    assert tuple(
        (bank.partition, bank.iteration, bank.start, bank.stop)
        for bank in definition.seed_banks
    ) == SEED_RANGES
    assert len(definition.preflight_schedule) == 240
    assert [
        (game.profile, game.map_seed, game.learner_seat)
        for game in definition.preflight_schedule[:4]
    ] == [
        ("conversion-3v1-near", 18_900_000, 0),
        ("conversion-3v1-near", 18_900_000, 1),
        ("conversion-3v1-near", 18_900_001, 0),
        ("conversion-3v1-near", 18_900_001, 1),
    ]
    for profile_index, profile in enumerate(PROFILES):
        games = definition.preflight_schedule[
            profile_index * 40:(profile_index + 1) * 40
        ]
        assert {game.profile for game in games} == {profile}
        assert [game.map_seed for game in games[::2]] == list(range(
            18_900_000 + profile_index * 20,
            18_900_020 + profile_index * 20,
        ))
        assert [game.learner_seat for game in games] == [0, 1] * 20
        assert all(game.reference_seat == game.learner_seat for game in games)
        assert all(game.episode_seed == game.map_seed for game in games)


@pytest.mark.parametrize("location", ["panel", "seed_banks"])
def test_definition_rejects_unknown_fields(tmp_path: Path, location: str) -> None:
    """A misspelled or newly introduced field must never be silently ignored."""

    panel_path, seeds_path, panel, seeds = _copy_definitions(tmp_path)
    target = panel if location == "panel" else seeds
    target["unexpected"] = True
    if location == "panel":
        _rewrite(panel_path, panel)
    else:
        _rewrite(seeds_path, seeds)
        panel["seed_banks"]["sha256"] = _sha256(seeds_path)
        _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match="fields|schema"):
        load_panel_definition(panel_path, repository_root=ROOT)


@pytest.mark.parametrize(
    'mutation',
    ['panel_bool', 'panel_float', 'seed_bool', 'seed_float', 'maps_float'],
)
def test_panel_and_seed_versions_and_counts_reject_scalar_aliases(
    tmp_path: Path, mutation: str,
) -> None:
    panel_path, seeds_path, panel, seeds = _copy_definitions(tmp_path)
    if mutation == 'panel_bool':
        panel['schema_version'] = True
    elif mutation == 'panel_float':
        panel['schema_version'] = 1.0
    elif mutation == 'seed_bool':
        seeds['schema_version'] = True
    elif mutation == 'seed_float':
        seeds['schema_version'] = 1.0
    else:
        seeds['oracle_preflight_profiles'][0]['maps'] = 20.0
    if mutation.startswith('seed') or mutation == 'maps_float':
        _rewrite(seeds_path, seeds)
        panel['seed_banks']['sha256'] = _sha256(seeds_path)
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match='integer|schema|maps'):
        load_panel_definition(panel_path, repository_root=ROOT)


@pytest.mark.parametrize("mutation", ["overlap", "wrong_count", "final_seed"])
def test_seed_definition_rejects_overlap_count_drift_and_final_bank_use(
    tmp_path: Path, mutation: str,
) -> None:
    """No Task 8 schedule may borrow another partition or the locked final bank."""

    panel_path, seeds_path, panel, seeds = _copy_definitions(tmp_path)
    if mutation == "overlap":
        seeds["banks"][1]["start"] = seeds["banks"][0]["stop"]
    elif mutation == "wrong_count":
        seeds["oracle_preflight_profiles"][0]["stop"] += 1
    else:
        seeds["banks"].append({
            "partition": "final_evaluation",
            "iteration": None,
            "start": 17_000_000,
            "stop": 17_000_249,
            "assigned": True,
        })
    _rewrite(seeds_path, seeds)
    panel["seed_banks"]["sha256"] = _sha256(seeds_path)
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match="overlap|20|final|bank|preflight"):
        load_panel_definition(panel_path, repository_root=ROOT)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("checkpoint_sha256", "checkpoint"),
        ("source_manifest_sha256", "source manifest"),
        ("dataset_manifest_sha256", "dataset"),
        ("scenario_sha256", "scenario"),
    ],
)
def test_definition_reopens_and_rehashes_every_physical_input(
    tmp_path: Path, field: str, message: str,
) -> None:
    """Declared identity alone cannot hide replaced model, data, or scenario bytes."""

    panel_path, _seeds_path, panel, _seeds = _copy_definitions(tmp_path)
    if field == "checkpoint_sha256":
        panel["starting_learner"]["checkpoint_sha256"] = "0" * 64
    elif field == "source_manifest_sha256":
        panel["starting_learner"]["source_manifest_sha256"] = "0" * 64
    elif field == "dataset_manifest_sha256":
        panel["original_dataset"]["manifest_sha256"] = "0" * 64
    else:
        panel["scenario"]["sha256"] = "0" * 64
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match=message):
        load_panel_definition(panel_path, repository_root=ROOT)


@pytest.mark.parametrize(
    "mutation",
    ["runtime_scenario", "contract", "action_regions", "dataset_scenario"],
)
def test_definition_rejects_coherent_drift_from_locked_causal_identities(
    tmp_path: Path, mutation: str,
) -> None:
    """Cross-consistent declarations still may not redefine the frozen panel."""

    panel_path, _seeds_path, panel, _seeds = _copy_definitions(tmp_path)
    if mutation == "runtime_scenario":
        panel["scenario"]["runtime_snapshot_sha256"] = "f" * 64
        panel["starting_learner"]["source_scenario_sha256"] = "f" * 64
    elif mutation == "contract":
        panel["contract"]["contract_hash"] = "f" * 64
        panel["starting_learner"]["contract_hash"] = "f" * 64
    elif mutation == "action_regions":
        panel["contract"]["action_regions"] = {
            "move": {"offset": 1, "count": 350},
            "attack": {"offset": 351, "count": 352},
            "deploy": {"offset": 703, "count": 585},
        }
    else:
        panel["original_dataset"]["scenario_hash"] = "f" * 64
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match="locked|scenario|contract|region|identity"):
        load_panel_definition(panel_path, repository_root=ROOT)


def test_definition_derives_scenario_identity_and_rejects_fog(
    tmp_path: Path,
) -> None:
    """A coherently rehashed fog scenario cannot silently enable omniscient search."""

    panel_path, _seeds_path, panel, _seeds = _copy_definitions(tmp_path)
    repository = tmp_path / "repository"
    scenario_path = (
        repository / "python" / "config" / "annihilation-imitation-v1.json"
    )
    scenario_path.parent.mkdir(parents=True)
    scenario = json.loads((
        ROOT / "python" / "config" / "annihilation-imitation-v1.json"
    ).read_text(encoding="utf-8"))
    scenario["rules"]["fog_of_war"] = True
    _rewrite(scenario_path, scenario)
    search_path = (
        repository / "engine" / "HexWars.Engine" / "BoundedSearchAgent.cs"
    )
    search_path.parent.mkdir(parents=True)
    shutil.copyfile(
        ROOT / "engine" / "HexWars.Engine" / "BoundedSearchAgent.cs",
        search_path,
    )
    panel["scenario"]["sha256"] = _sha256(scenario_path)
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match="scenario|fog|locked"):
        load_panel_definition(panel_path, repository_root=repository)


def test_definition_rejects_repository_root_junction_alias(tmp_path: Path) -> None:
    """A junction must not relocate the supposedly authoritative repository root."""

    alias = tmp_path / "repository-junction"
    _windows_directory_junction(alias, ROOT)

    with pytest.raises(ValueError, match="junction|canonical|repository"):
        load_panel_definition(PANEL_PATH, repository_root=alias)


def test_definition_rejects_panel_file_symlink_even_when_target_is_in_root(
    tmp_path: Path,
) -> None:
    panel_path, _seeds_path, _panel, _seeds = _copy_definitions(tmp_path)
    alias = panel_path.with_name('panel-alias.json')
    _symlink_or_skip_windows_privilege(alias, panel_path)

    with pytest.raises(ValueError, match='canonical|symlink|reparse'):
        load_panel_definition(alias, repository_root=ROOT)


def test_definition_rejects_panel_directory_junction_even_when_target_is_in_root(
    tmp_path: Path,
) -> None:
    panel_path, _seeds_path, _panel, _seeds = _copy_definitions(tmp_path)
    alias = tmp_path / 'panel-junction'
    _windows_directory_junction(alias, panel_path.parent)

    with pytest.raises(ValueError, match='canonical|junction|reparse'):
        load_panel_definition(alias / panel_path.name, repository_root=ROOT)


def test_preflight_path_set_rejects_lexical_alias_and_staging_junction(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    parent = tmp_path / 'canonical'
    child = parent / 'child'
    child.mkdir(parents=True)
    lexical_alias = child / '..' / 'preflight'
    with pytest.raises(ValueError, match='canonical|lexical'):
        dagger_module._validate_preflight_path_set_v2(definition, lexical_alias)

    destination = tmp_path / 'destination'
    staging = destination.with_name(destination.name + '.staging')
    target = tmp_path / 'staging-target'
    target.mkdir()
    sentinel = target / 'sentinel.bin'
    sentinel.write_bytes(b'unchanged')
    _windows_directory_junction(staging, target)
    try:
        with pytest.raises(ValueError, match='junction|reparse|symlink'):
            dagger_module._validate_preflight_path_set_v2(definition, destination)
        assert sentinel.read_bytes() == b'unchanged'
    finally:
        staging.rmdir()


def test_dataset_tree_rejects_an_inner_windows_junction(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    target = tmp_path / "payload"
    root.mkdir()
    target.mkdir()
    (target / "shard.bin").write_bytes(b"payload")
    _windows_directory_junction(root / "shards", target)

    with pytest.raises(ValueError, match="junction|reparse|symlink"):
        dagger_module._dataset_tree_identity(root)


def test_preflight_rejects_an_output_parent_windows_junction(tmp_path: Path) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    target = tmp_path / "real-output"
    target.mkdir()
    alias = tmp_path / "output-junction"
    _windows_directory_junction(alias, target)

    with pytest.raises(ValueError, match="junction|reparse|symlink"):
        run_oracle_preflight(
            definition,
            output_root=alias / "preflight",
            repository_identity_provider=_repository_provider,
            evaluator=lambda *_args: pytest.fail("evaluator must not run"),
            benchmark=lambda *_args: pytest.fail("benchmark must not run"),
            codec=lambda *_args: pytest.fail("codec must not run"),
        )


def test_base_dataset_boundary_binds_tree_and_invokes_loader_and_full_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    (root / "manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "shard.bin").write_bytes(b"rows")
    physical = dagger_module._dataset_tree_identity(root)
    definition = replace(
        load_panel_definition(PANEL_PATH, repository_root=ROOT),
        dataset_root=root,
        dataset_content_sha256=physical["content_sha256"],
        dataset_file_count=physical["file_count"],
        dataset_byte_size=physical["byte_size"],
    )
    calls: list[str] = []
    sentinel = object()
    monkeypatch.setattr(
        dagger_module, "load_imitation_dataset",
        lambda dataset_root, expected_contract: (
            calls.append(f"load:{dataset_root}") or sentinel
        ),
    )
    monkeypatch.setattr(
        dagger_module, "audit_imitation_dataset",
        lambda dataset: calls.append("audit") or {
            "games": 1,
            "teacher_labels": 1,
            "masked_labels": 0,
            "round_trip_mismatches": 0,
            "replay_mismatches": 0,
        },
    )

    evidence = REAL_BASE_DATASET_AUDIT(definition)
    assert evidence["content_sha256"] == physical["content_sha256"]
    assert calls == [f"load:{root}", "audit"]

    (root / "shard.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="full content identity"):
        REAL_BASE_DATASET_AUDIT(definition)


def test_base_dataset_audit_rejects_mutation_during_semantic_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'manifest.json').write_text('{}\n', encoding='utf-8')
    shard = root / 'shard.bin'
    shard.write_bytes(b'rows-a')
    physical = REAL_DATASET_TREE_IDENTITY(root)
    definition = replace(
        load_panel_definition(PANEL_PATH, repository_root=ROOT),
        dataset_root=root,
        dataset_content_sha256=physical['content_sha256'],
        dataset_file_count=physical['file_count'],
        dataset_byte_size=physical['byte_size'],
    )
    dagger_module._BASE_DATASET_SEMANTIC_AUDIT_CACHE.clear()
    monkeypatch.setattr(
        dagger_module, 'load_imitation_dataset',
        lambda *_args, **_kwargs: object(),
    )

    def mutate_during_audit(_dataset):
        shard.write_bytes(b'rows-b')
        return {
            'games': 1,
            'teacher_labels': 1,
            'masked_labels': 0,
            'round_trip_mismatches': 0,
            'replay_mismatches': 0,
        }

    monkeypatch.setattr(
        dagger_module, 'audit_imitation_dataset', mutate_during_audit,
    )
    with pytest.raises(ValueError, match='changed during|stable|content identity'):
        REAL_BASE_DATASET_AUDIT(definition)


def test_base_dataset_cache_hit_rehashes_before_return(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / 'dataset'
    root.mkdir()
    (root / 'manifest.json').write_text('{}\n', encoding='utf-8')
    shard = root / 'shard.bin'
    shard.write_bytes(b'cache-a')
    physical = REAL_DATASET_TREE_IDENTITY(root)
    definition = replace(
        load_panel_definition(PANEL_PATH, repository_root=ROOT),
        dataset_root=root,
        dataset_content_sha256=physical['content_sha256'],
        dataset_file_count=physical['file_count'],
        dataset_byte_size=physical['byte_size'],
    )
    dagger_module._BASE_DATASET_SEMANTIC_AUDIT_CACHE.clear()
    monkeypatch.setattr(
        dagger_module, 'load_imitation_dataset',
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        dagger_module, 'audit_imitation_dataset',
        lambda _dataset: {
            'games': 1,
            'teacher_labels': 1,
            'masked_labels': 0,
            'round_trip_mismatches': 0,
            'replay_mismatches': 0,
        },
    )
    REAL_BASE_DATASET_AUDIT(definition)
    calls = 0

    def mutate_after_snapshot(dataset_root: Path):
        nonlocal calls
        identity = REAL_DATASET_TREE_IDENTITY(dataset_root)
        calls += 1
        if calls == 1:
            shard.write_bytes(b'cache-b')
        return identity

    monkeypatch.setattr(
        dagger_module, '_dataset_tree_identity', mutate_after_snapshot,
    )
    with pytest.raises(ValueError, match='changed during|stable|content identity'):
        REAL_BASE_DATASET_AUDIT(definition)
    assert calls == 2


def test_definition_is_deeply_immutable_and_revalidates_from_disk() -> None:
    """Post-load mutation must not change a preflight or downstream stage identity."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)

    with pytest.raises(TypeError):
        definition.training["batch_size"] = 1
    with pytest.raises(TypeError):
        definition.repository_policy["required_clean"] = False
    with pytest.raises((AttributeError, TypeError)):
        definition.oracle_candidates[0].depth = 1
    with pytest.raises(ValueError, match="identity"):
        validate_panel_definition(replace(definition, contract_hash="0" * 64))


@pytest.mark.parametrize(
    ("location", "field"),
    [
        ("repository", "required_clean"),
        ("candidate", "use_heuristic"),
        ("preflight", "labels_per_second_minimum"),
    ],
)
def test_definition_rejects_bool_integer_and_float_integer_aliases(
    tmp_path: Path, location: str, field: str,
) -> None:
    panel_path, _seeds_path, panel, _seeds = _copy_definitions(tmp_path)
    if location == "candidate":
        panel["oracle"]["candidates"][0][field] = 1
    elif location == "preflight":
        panel["oracle"]["preflight"][field] = 10
    else:
        panel[location][field] = 1
    _rewrite(panel_path, panel)

    with pytest.raises(ValueError, match="candidate|preflight|repository|values"):
        load_panel_definition(panel_path, repository_root=ROOT)


def test_benchmark_sample_is_typed_deeply_frozen_and_exact() -> None:
    state = {"round": 1, "units": [{"id": 7}]}
    sample = OracleBenchmarkSample(
        state_hash=hashlib.sha256(json.dumps(
            state, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")).hexdigest(),
        decision_index=0,
        observation=(0.0, 1.0),
        legal_mask=(True, False),
        state=state,
    )
    state["units"][0]["id"] = 99

    assert sample.state["units"][0]["id"] == 7
    assert isinstance(sample.observation, tuple)
    with pytest.raises(TypeError):
        sample.state["round"] = 2
    with pytest.raises(FrozenInstanceError):
        sample.decision_index = 1
    malformed = sample.to_dict()
    malformed["observation"][0] = 0
    with pytest.raises(ValueError, match="finite floats"):
        OracleBenchmarkSample.from_dict(malformed)


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _trace(
    outcome: str,
    learner_seat: int,
    *,
    cycling: bool = False,
    action_waste: bool = False,
) -> EpisodeTrace:
    winner = (
        learner_seat
        if outcome == "win"
        else 1 - learner_seat
        if outcome == "loss"
        else None
    )

    transition_count = max(
        3 if action_waste else 1,
        2 if cycling else 1,
    )

    def state(index: int) -> StateFrame:
        terminal = index == transition_count
        points = 0 if cycling else index
        seats = tuple(
            SeatFrame(
                seat=seat,
                points=points if seat == learner_seat else 0,
                destroyed_value=0,
                alive_units=1,
                current_hit_points=1,
                maximum_hit_points=1,
                health_adjusted_material=1.0,
                can_damage_enemy=True,
                can_currently_attack_enemy=False,
                can_move=True,
                units=(),
            )
            for seat in (0, 1)
        )
        return StateFrame(
            round=index + 1,
            active_seat=learner_seat,
            is_game_over=terminal,
            winner=winner if terminal else None,
            productive_legal_actions=int(action_waste and not terminal),
            seats=seats,
        )

    return EpisodeTrace(
        schema_version=1,
        transitions=tuple(
            TransitionFrame(
                before=state(index),
                command=CommandFrame(
                    kind="end_turn",
                    issuer=learner_seat,
                    actor_id=None,
                    target_id=None,
                    q=None,
                    r=None,
                ),
                after=state(index + 1),
            )
            for index in range(transition_count)
        ),
    )


def _benchmark_sample(game) -> OracleBenchmarkSample:
    state = {
        "round": 1,
        "active_seat": game.learner_seat,
        "map_seed": game.map_seed,
        "profile": game.profile,
        "units": [],
    }
    return OracleBenchmarkSample(
        state_hash=hashlib.sha256(json.dumps(
            state, sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")).hexdigest(),
        decision_index=0,
        observation=(0.0,) * 1292,
        legal_mask=(True, True, *([False] * 1286)),
        state=state,
    )


def _preflight_boundaries(
    *,
    wins: dict[int, int],
    cycling_draws: dict[int, int] | None = None,
    seconds_per_query: dict[int, float] | None = None,
    nondeterministic: bool = False,
    round_trip_failure: bool = False,
) -> tuple[Any, Any, Any, _FakeClock, dict[int, list], Counter]:
    clock = _FakeClock()
    schedules: dict[int, list] = defaultdict(list)
    queries: Counter = Counter()
    cycling_draws = cycling_draws or {}
    seconds_per_query = seconds_per_query or {512: 0.001, 2048: 0.001}
    legal_mask = (True, True, *([False] * 1286))

    def evaluator(oracle, game):
        schedules[oracle.expansion_budget].append(game)
        game_index = len(schedules[oracle.expansion_budget]) - 1
        if game_index < wins[oracle.expansion_budget]:
            outcome = "win"
        else:
            outcome = "draw"
        draw_index = game_index - wins[oracle.expansion_budget]
        cycling = outcome == "draw" and draw_index < cycling_draws.get(
            oracle.expansion_budget, 0,
        )
        action_waste = outcome == "draw"
        return OraclePreflightGameResult(
            outcome=outcome,
            cycling=cycling,
            action_waste=action_waste,
            wasted_end_turns=3 if action_waste else 0,
            trace=_trace(
                outcome,
                game.learner_seat,
                cycling=cycling,
                action_waste=action_waste,
            ),
            replay=f"preflight {oracle.expansion_budget} {game.map_seed} "
            f"{game.learner_seat}\n",
            samples=(_benchmark_sample(game),),
        )

    def benchmark(oracle, game, sample):
        key = (oracle.expansion_budget, game.map_seed, game.learner_seat)
        queries[key] += 1
        clock.advance(seconds_per_query[oracle.expansion_budget])
        action = (
            1
            if nondeterministic and queries[key] == 2
            else 0
        )
        command = {
            "Kind": "end_turn" if action == 0 else "move",
            "Issuer": game.learner_seat,
            "ActorId": None if action == 0 else 0,
            "TargetId": None,
            "Q": None if action == 0 else 0,
            "R": None if action == 0 else 0,
        }
        return OracleBenchmarkDecision(
            encoded_action=action,
            command=command,
            actual_expansion_count=min(127, oracle.expansion_budget),
        )

    def codec(oracle, game, sample, decision):
        decoded = dict(decision.command)
        if round_trip_failure:
            decoded["Kind"] = "move" if decoded["Kind"] == "end_turn" else "end_turn"
        return OracleCodecEvidence(
            provenance="private-test-callback",
            state_hash=sample.state_hash,
            contract_hash="7347819c2e68fa2d216dc712afc4785e185ca50d3832487d66589a68eee5a9d6",
            encoding_hash="2f334bc2163fd931d84c004e9dc8f44bae68934e46fbf2ec2c819fa3e297054a",
            requested_action=decision.encoded_action,
            encoded_action=decision.encoded_action,
            encoded_command=decision.command,
            decoded_command=decoded,
            mask_legal=sample.legal_mask[decision.encoded_action],
            apply_success=True,
        )

    return evaluator, benchmark, codec, clock, schedules, queries


def test_public_oracle_preflight_rejects_unsealed_execution_and_exposes_no_fake_seams(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    parameters = inspect.signature(dagger_module.run_oracle_preflight).parameters

    assert set(parameters) == {'definition', 'output_root', 'execution_session'}
    assert not {
        'evaluator', 'benchmark', 'codec', 'repository_identity_provider', 'clock',
        'on_selected',
    } & set(parameters)
    with pytest.raises(RuntimeError, match='sealed engine execution session|Task 9'):
        dagger_module.run_oracle_preflight(
            definition,
            output_root=tmp_path / 'production-preflight',
            execution_session=object(),
        )


@pytest.mark.parametrize(
    ('state', 'message'),
    [
        ({'value': float('inf')}, 'finite'),
        ({'value': 1 << 40}, 'int32|magnitude'),
        ({'value': 'x' * 1_048_577}, 'byte|size'),
    ],
)
def test_oracle_benchmark_state_rejects_nonfinite_huge_or_oversized_payload(
    state: dict[str, Any], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        dagger_module._validate_oracle_state_payload_v2(state)


def test_oracle_benchmark_state_rejects_excessive_nesting() -> None:
    state: dict[str, Any] = {'leaf': 0}
    for _ in range(40):
        state = {'nested': state}

    with pytest.raises(ValueError, match='depth|nest'):
        dagger_module._validate_oracle_state_payload_v2(state)


def test_oracle_preflight_bounds_samples_per_game() -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    game = definition.preflight_schedule[0]
    sample = _benchmark_sample(game)

    assert dagger_module._MAX_PREFLIGHT_SAMPLES_PER_GAME == 1024
    with pytest.raises(ValueError, match='sample.*limit|too many'):
        OraclePreflightGameResult(
            outcome='win',
            cycling=False,
            action_waste=False,
            wasted_end_turns=0,
            trace=_trace('win', game.learner_seat),
            replay='bounded samples\n',
            samples=(sample,) * 1025,
        )


def test_preflight_json_read_and_write_reject_oversized_bytes_before_io(
    tmp_path: Path,
) -> None:
    source = tmp_path / 'oversized.json'
    source.write_text('{"payload":"' + ('x' * 64) + '"}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='byte|size|large'):
        dagger_module._read_bounded_json_v2(
            source, max_bytes=32, label='test artifact',
        )

    destination = tmp_path / 'not-written.json'
    with pytest.raises(ValueError, match='byte|size|large'):
        dagger_module._bounded_atomic_write_json_v2(
            destination,
            {'payload': 'x' * 64},
            max_bytes=32,
            label='test artifact',
        )
    assert not destination.exists()


def test_preflight_writer_rejects_oversized_replay_before_writing_file(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    game = definition.preflight_schedule[0]
    result = OraclePreflightGameResult(
        outcome='win',
        cycling=False,
        action_waste=False,
        wasted_end_turns=0,
        trace=_trace('win', game.learner_seat),
        replay='x' * (dagger_module._MAX_PREFLIGHT_REPLAY_BYTES + 1),
        samples=(_benchmark_sample(game),),
    )

    with pytest.raises(ValueError, match='replay.*byte|byte.*large'):
        dagger_module._write_preflight_game_v2(
            staging=tmp_path,
            candidate_index=0,
            game_index=0,
            oracle=definition.oracle_candidates[0],
            game=game,
            result=result,
            benchmark_records=[],
            game_evidence=[],
            candidate_games=[],
        )
    assert not next(tmp_path.rglob('*.replay.json'), None)


def test_preflight_reopen_rejects_oversized_manifest_before_json_parse(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    manifest = tmp_path / 'oracle-preflight.json'
    manifest.write_bytes(
        (b' ' * dagger_module._MAX_PREFLIGHT_MANIFEST_BYTES) + b'{}',
    )

    with pytest.raises(ValueError, match='manifest.*byte|byte.*large'):
        dagger_module._open_oracle_preflight_v2(
            tmp_path,
            expected_identity={},
            definition=definition,
        )


def test_oracle_benchmark_sample_hash_is_derived_from_state() -> None:
    state = {'round': 1, 'active_seat': 0, 'units': []}

    with pytest.raises(ValueError, match='canonical state hash'):
        OracleBenchmarkSample(
            state_hash='f' * 64,
            decision_index=0,
            observation=(0.0,),
            legal_mask=(True,),
            state=state,
        )


def test_private_preflight_identity_declares_unauthenticated_test_transcript() -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    identity = dagger_module._oracle_preflight_identity(
        definition,
        _repository_provider(ROOT),
        {
            'content_sha256': definition.dataset_content_sha256,
            'file_count': definition.dataset_file_count,
            'byte_size': definition.dataset_byte_size,
            'audit': {
                'games': 1,
                'teacher_labels': 1,
                'masked_labels': 0,
                'round_trip_mismatches': 0,
                'replay_mismatches': 0,
            },
        },
    )

    assert identity['execution_trust'] == PRIVATE_TEST_TRUST


@pytest.mark.parametrize(
    ('field', 'alias'),
    [
        ('schema_version', True),
        ('schema_version', 1.0),
        ('candidate_index', False),
        ('candidate_index', 0.0),
        ('game_index', False),
        ('game_index', 0.0),
    ],
)
def test_preflight_envelope_context_rejects_bool_and_float_integer_aliases(
    field: str, alias: Any,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    game = definition.preflight_schedule[0]
    envelope = {
        'schema_version': 1,
        'execution_trust': PRIVATE_TEST_TRUST,
        'candidate_index': 0,
        'game_index': 0,
        'schedule': game.to_dict(),
    }
    envelope[field] = alias

    with pytest.raises(ValueError, match='integer|context|schema'):
        dagger_module._validate_preflight_envelope_context_v2(
            envelope,
            candidate_index=0,
            game_index=0,
            game=game,
            label='test envelope',
        )


def test_oracle_expansion_budget_accepts_exact_limit_and_rejects_one_over() -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    oracle = definition.oracle_candidates[0]
    game = definition.preflight_schedule[0]
    sample = _benchmark_sample(game)
    decision = OracleBenchmarkDecision(
        encoded_action=0,
        command={
            'Kind': 'end_turn',
            'Issuer': game.learner_seat,
            'ActorId': None,
            'TargetId': None,
            'Q': None,
            'R': None,
        },
        actual_expansion_count=oracle.expansion_budget,
    )

    dagger_module._preflight_decision_valid_v2(
        decision, sample=sample, oracle=oracle, game=game, definition=definition,
    )
    with pytest.raises(ValueError, match='exceeded.*expansion budget'):
        dagger_module._preflight_decision_valid_v2(
            replace(
                decision,
                actual_expansion_count=oracle.expansion_budget + 1,
            ),
            sample=sample,
            oracle=oracle,
            game=game,
            definition=definition,
        )


def test_oracle_preflight_runs_identical_240_game_schedules_and_double_queries(
    tmp_path: Path,
) -> None:
    """Dropping a seat/map or one determinism query would invalidate oracle selection."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, schedules, queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    callbacks: list[int] = []

    selected = run_oracle_preflight(
        definition,
        output_root=tmp_path / "oracle-preflight",
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
        on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
    )

    assert selected.expansion_budget == 2048
    assert callbacks == [2048]
    assert schedules[512] == schedules[2048] == list(
        definition.preflight_schedule,
    )
    assert len(schedules[512]) == len(schedules[2048]) == 240
    assert set(queries.values()) == {2}
    assert len(queries) == 480
    manifest = json.loads((
        tmp_path / "oracle-preflight" / "oracle-preflight.json"
    ).read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["identity"]["execution_trust"] == PRIVATE_TEST_TRUST
    assert manifest["selected_oracle"]["expansion_budget"] == 2048
    assert [item["games"] for item in manifest["candidates"]] == [240, 240]
    assert [item["labels"] for item in manifest["candidates"]] == [240, 240]
    assert all(item["determinism_failures"] == 0 for item in manifest["candidates"])
    assert all(item["round_trip_failures"] == 0 for item in manifest["candidates"])
    assert all(item["labels_per_second"] >= 10.0 for item in manifest["candidates"])
    for candidate in manifest["candidates"]:
        assert set(candidate["profiles"]) == set(PROFILES)
        assert all(
            profile["games"] == 40
            and profile["seats"]["0"]["games"] == 20
            and profile["seats"]["1"]["games"] == 20
            and (
                profile["wins"] + profile["losses"] + profile["draws"] == 40
            )
            for profile in candidate["profiles"].values()
        )
        assert candidate["query_count"] == 480
        assert candidate["expansion_total"] == 480 * 127
        assert candidate["mean_expansions"] == 127.0

    physical_payloads = {
        artifact: json.loads((
            tmp_path / 'oracle-preflight' / manifest['games'][0][artifact]['path']
        ).read_text(encoding='utf-8'))
        for artifact in ('trace', 'replay', 'benchmark')
    }
    assert all(
        payload['execution_trust'] == PRIVATE_TEST_TRUST
        for payload in physical_payloads.values()
    )
    physical_record = physical_payloads['benchmark']['records'][0]
    assert physical_record['execution_trust'] == PRIVATE_TEST_TRUST
    for query in ('first', 'second'):
        codec_payload = physical_record[query]['codec']
        assert codec_payload['execution_trust'] == PRIVATE_TEST_TRUST
        assert codec_payload['provenance'] == 'private-test-callback'
        assert 'authority' not in codec_payload

    forged_root = tmp_path / 'coherently-forged-test-transcript'
    shutil.copytree(tmp_path / 'oracle-preflight', forged_root)
    forged_manifest_path = forged_root / 'oracle-preflight.json'
    forged_manifest = json.loads(forged_manifest_path.read_text(encoding='utf-8'))
    descriptor = forged_manifest['games'][0]['benchmark']
    benchmark_path = forged_root / descriptor['path']
    payload = json.loads(benchmark_path.read_text(encoding='utf-8'))
    record = payload['records'][0]
    record['sample']['state']['round'] += 1
    state_hash = _content_identity({
        **record['sample']['state'],
        'content_identity': 'discarded',
    })
    record['sample']['state_hash'] = state_hash
    record['sample_sha256'] = _content_identity({
        **record['sample'],
        'content_identity': 'discarded',
    })
    for query in ('first', 'second'):
        record[query]['codec']['state_hash'] = state_hash
        record[query]['decision']['actual_expansion_count'] += 1
        record[query]['elapsed_seconds'] += 0.125
    record['pair_seconds'] = (
        record['first']['elapsed_seconds'] + record['second']['elapsed_seconds']
    )
    _rewrite(benchmark_path, payload)
    descriptor['sha256'] = _sha256(benchmark_path)
    descriptor['byte_size'] = benchmark_path.stat().st_size
    summary = forged_manifest['candidates'][0]
    summary['expansion_total'] += 2
    summary['max_expansions'] = 128
    summary['mean_expansions'] = (
        summary['expansion_total'] / summary['query_count']
    )
    summary['benchmark_seconds'] += 0.25
    summary['labels_per_second'] = (
        summary['labels'] / summary['benchmark_seconds']
    )
    forged_manifest['content_identity'] = _content_identity(forged_manifest)
    _rewrite(forged_manifest_path, forged_manifest)

    with pytest.raises(RuntimeError, match='sealed engine execution session|Task 9'):
        dagger_module.run_oracle_preflight(
            definition,
            output_root=forged_root,
            execution_session=object(),
        )
    valid_root = tmp_path / 'oracle-preflight'
    valid_manifest_path = valid_root / 'oracle-preflight.json'
    valid_manifest_bytes = valid_manifest_path.read_bytes()
    valid_manifest = json.loads(valid_manifest_bytes)

    manifest = json.loads(valid_manifest_bytes)
    manifest['identity']['execution_trust']['mode'] = 'engine-authenticated'
    manifest['content_identity'] = _content_identity(manifest)
    try:
        _rewrite(valid_manifest_path, manifest)
        with pytest.raises(ValueError, match='trust|private|authenticated'):
            dagger_module._open_oracle_preflight_v2(
                valid_root,
                expected_identity=manifest['identity'],
                definition=definition,
            )
    finally:
        valid_manifest_path.write_bytes(valid_manifest_bytes)

    def remove_envelope_trust(payload):
        payload.pop('execution_trust')

    def change_replay_trust(payload):
        payload['execution_trust']['engine_authenticated'] = True

    def change_benchmark_trust(payload):
        payload['execution_trust']['evidence_class'] = 'engine-authoritative'

    def remove_record_trust(payload):
        payload['records'][0].pop('execution_trust')

    def remove_codec_trust(payload):
        payload['records'][0]['first']['codec'].pop('execution_trust')

    def overclaim_codec_provenance(payload):
        payload['records'][0]['second']['codec']['provenance'] = (
            'HexWars.Engine.TacticalV2Coding-v1'
        )

    trust_mutations = (
        ('trace', remove_envelope_trust),
        ('replay', change_replay_trust),
        ('benchmark', change_benchmark_trust),
        ('benchmark', remove_record_trust),
        ('benchmark', remove_codec_trust),
        ('benchmark', overclaim_codec_provenance),
    )
    for artifact, mutate in trust_mutations:
        manifest = json.loads(valid_manifest_bytes)
        descriptor = manifest['games'][0][artifact]
        artifact_path = valid_root / descriptor['path']
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(artifact_bytes)
        mutate(payload)
        try:
            _rewrite(artifact_path, payload)
            descriptor['sha256'] = _sha256(artifact_path)
            descriptor['byte_size'] = artifact_path.stat().st_size
            manifest['content_identity'] = _content_identity(manifest)
            _rewrite(valid_manifest_path, manifest)

            with pytest.raises(
                ValueError, match='trust|private|provenance|fields',
            ):
                dagger_module._open_oracle_preflight_v2(
                    valid_root,
                    expected_identity=valid_manifest['identity'],
                    definition=definition,
                )
        finally:
            artifact_path.write_bytes(artifact_bytes)
            valid_manifest_path.write_bytes(valid_manifest_bytes)

    artifact_aliases = (
        ('trace', 'schema_version', True, None),
        ('replay', 'candidate_index', 0.0, None),
        ('benchmark', 'game_index', False, None),
        ('benchmark', 'sample_index', 0.0, 0),
    )
    for artifact, field, alias, record_index in artifact_aliases:
        manifest = json.loads(valid_manifest_bytes)
        descriptor = manifest['games'][0][artifact]
        artifact_path = valid_root / descriptor['path']
        artifact_bytes = artifact_path.read_bytes()
        payload = json.loads(artifact_bytes)
        if record_index is None:
            payload[field] = alias
        else:
            payload['records'][record_index][field] = alias
        try:
            _rewrite(artifact_path, payload)
            descriptor['sha256'] = _sha256(artifact_path)
            descriptor['byte_size'] = artifact_path.stat().st_size
            manifest['content_identity'] = _content_identity(manifest)
            _rewrite(valid_manifest_path, manifest)

            with pytest.raises(ValueError, match='integer|schema|context|sample'):
                dagger_module._open_oracle_preflight_v2(
                    valid_root,
                    expected_identity=valid_manifest['identity'],
                    definition=definition,
                )
        finally:
            artifact_path.write_bytes(artifact_bytes)
            valid_manifest_path.write_bytes(valid_manifest_bytes)

    manifest = json.loads(valid_manifest_bytes)
    manifest['schema_version'] = 2.0
    manifest['content_identity'] = _content_identity(manifest)
    try:
        _rewrite(valid_manifest_path, manifest)
        with pytest.raises(ValueError, match='integer|completed'):
            dagger_module._open_oracle_preflight_v2(
                valid_root,
                expected_identity=valid_manifest['identity'],
                definition=definition,
            )
    finally:
        valid_manifest_path.write_bytes(valid_manifest_bytes)


def test_oracle_preflight_accepts_exact_win_and_throughput_boundaries(
    tmp_path: Path,
) -> None:
    """The locked gates are inclusive: 204/240 wins and 10 labels/s both pass."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, base_benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 204, 2048: 204},
        seconds_per_query={512: 0.0, 2048: 0.0},
    )
    calls = Counter()

    def benchmark(oracle, game, sample):
        decision = base_benchmark(oracle, game, sample)
        calls[oracle.expansion_budget] += 1
        if calls[oracle.expansion_budget] <= 24:
            clock.advance(1.0)
        return decision

    selected = run_oracle_preflight(
        definition,
        output_root=tmp_path / "exact-boundaries",
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )

    assert selected.expansion_budget == 512
    manifest = json.loads((
        tmp_path / "exact-boundaries" / "oracle-preflight.json"
    ).read_text(encoding="utf-8"))
    assert [candidate["wins"] for candidate in manifest["candidates"]] == [204, 204]
    assert [candidate["labels_per_second"] for candidate in manifest["candidates"]] == [10.0, 10.0]
    assert [candidate["eligible"] for candidate in manifest["candidates"]] == [True, True]


@pytest.mark.parametrize(
    ("wins", "cycling", "seconds", "expected"),
    [
        ({512: 210, 2048: 211}, {512: 0, 2048: 9}, None, 2048),
        ({512: 210, 2048: 210}, {512: 5, 2048: 4}, None, 2048),
        (
            {512: 210, 2048: 210},
            {512: 4, 2048: 4},
            {512: 0.002, 2048: 0.001},
            2048,
        ),
        ({512: 210, 2048: 210}, {512: 4, 2048: 4}, None, 512),
    ],
)
def test_oracle_preflight_uses_the_locked_lexicographic_tie_break(
    tmp_path: Path,
    wins: dict[int, int],
    cycling: dict[int, int],
    seconds: dict[int, float] | None,
    expected: int,
) -> None:
    """Oracle choice is win rate, cycling, throughput, then the smaller budget."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins=wins,
        cycling_draws=cycling,
        seconds_per_query=seconds,
    )

    selected = run_oracle_preflight(
        definition,
        output_root=tmp_path / f"preflight-{expected}",
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )

    assert selected.expansion_budget == expected


@pytest.mark.parametrize("failure", ["wins", "throughput", "determinism", "round_trip"])
def test_oracle_preflight_gate_failure_blocks_downstream_callback(
    tmp_path: Path, failure: str,
) -> None:
    """No collection or training callback may follow an oracle that misses any gate."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 203, 2048: 203} if failure == "wins" else {512: 210, 2048: 210},
        seconds_per_query=(
            {512: 0.1, 2048: 0.1}
            if failure == "throughput"
            else None
        ),
        nondeterministic=failure == "determinism",
        round_trip_failure=failure == "round_trip",
    )
    callbacks: list[int] = []

    with pytest.raises(RuntimeError, match="oracle preflight"):
        run_oracle_preflight(
            definition,
            output_root=tmp_path / f"failed-{failure}",
            repository_identity_provider=_repository_provider,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
            on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
        )

    assert callbacks == []
    assert not (tmp_path / f"failed-{failure}").exists()
    diagnostics = tmp_path / f"failed-{failure}.diagnostics"
    attempts = list(diagnostics.glob("attempt-*/diagnostic.json"))
    assert len(attempts) == 1
    assert not (tmp_path / f"failed-{failure}.staging").exists()
    diagnostic_payload = json.loads(attempts[0].read_text(encoding="utf-8"))
    if failure == "determinism":
        assert all(
            item["determinism_failures"] == 240
            and item["round_trip_failures"] == 0
            for item in diagnostic_payload["candidates"]
        )
    if failure == "round_trip":
        assert all(
            item["determinism_failures"] == 0
            and item["round_trip_failures"] == 480
            for item in diagnostic_payload["candidates"]
        )


@pytest.mark.parametrize("failure", ["mutation", "duplicate"])
def test_oracle_preflight_rejects_mutated_or_duplicated_typed_samples(
    tmp_path: Path, failure: str,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    if failure == "duplicate":
        base_evaluator = evaluator

        def evaluator(oracle, game):
            result = base_evaluator(oracle, game)
            return replace(result, samples=(result.samples[0], result.samples[0]))
    else:
        base_benchmark = benchmark
        mutated = False

        def benchmark(oracle, game, sample):
            nonlocal mutated
            decision = base_benchmark(oracle, game, sample)
            if not mutated:
                object.__setattr__(sample, "decision_index", 1)
                mutated = True
            return decision

    root = tmp_path / failure
    with pytest.raises(ValueError, match="mutated|duplicated"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
        )
    assert not root.exists()
    assert not root.with_name(root.name + ".staging").exists()
    assert len(list(
        root.with_name(root.name + ".diagnostics").glob(
            "attempt-*/diagnostic.json"
        )
    )) == 1


@pytest.mark.parametrize("diagnostic", ["cycling", "action_waste"])
def test_oracle_preflight_rejects_diagnostics_not_supported_by_the_trace(
    tmp_path: Path, diagnostic: str,
) -> None:
    """Caller metadata cannot manufacture cycling or wasted-EndTurn evidence."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)

    def evaluator(_oracle, game):
        return OraclePreflightGameResult(
            outcome="draw",
            cycling=diagnostic == "cycling",
            action_waste=diagnostic == "action_waste",
            wasted_end_turns=3 if diagnostic == "action_waste" else 0,
            trace=_trace("draw", game.learner_seat),
            replay="unsupported diagnostic\n",
            samples=(_benchmark_sample(game),),
        )

    with pytest.raises(ValueError, match="cycling|waste|diagnostic|trace"):
        run_oracle_preflight(
            definition,
            output_root=tmp_path / f"unsupported-{diagnostic}",
            repository_identity_provider=_repository_provider,
            evaluator=evaluator,
            benchmark=lambda *_args: pytest.fail(
                "diagnostics must be checked before an oracle query"
            ),
            codec=lambda *_args: pytest.fail(
                "diagnostics must be checked before codec validation"
            ),
        )


def test_oracle_preflight_exact_reuse_launches_zero_games_and_rehashes_evidence(
    tmp_path: Path,
) -> None:
    """Reuse must physically reopen all traces/replays and reject any identity drift."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    selected = run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )

    calls = 0

    def forbidden(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("completed preflight must not launch work")

    reused = run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=forbidden,
        benchmark=forbidden,
        codec=forbidden,
        on_selected=forbidden,
    )
    assert reused == selected
    assert calls == 0

    before = (root / "oracle-preflight.json").read_bytes()
    with pytest.raises(ValueError, match="identity|reusable"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=lambda root: {
                **_repository_provider(root), "commit": "e" * 40,
            },
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert (root / "oracle-preflight.json").read_bytes() == before
    assert calls == 0

    games = root / "games"
    games_target = tmp_path / "games-target"
    os.replace(games, games_target)
    _windows_directory_junction(games, games_target)
    try:
        with pytest.raises(ValueError, match="reparse|junction|reusable"):
            run_oracle_preflight(
                definition,
                output_root=root,
                repository_identity_provider=_repository_provider,
                evaluator=forbidden,
                benchmark=forbidden,
                codec=forbidden,
                on_selected=forbidden,
            )
    finally:
        games.rmdir()
        os.replace(games_target, games)
    assert calls == 0

    replay = next((root / "games").rglob("*.replay.json"))
    replay.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|reusable|evidence"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert calls == 0


def test_repository_execution_identity_uses_real_clean_git_and_rejects_unrelated_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / 'repository'
    repository.mkdir()
    subprocess.run(['git', 'init', '-q', str(repository)], check=True)
    subprocess.run(
        ['git', '-C', str(repository), 'config', 'user.email', 'test@example.invalid'],
        check=True,
    )
    subprocess.run(
        ['git', '-C', str(repository), 'config', 'user.name', 'Task 8 Test'],
        check=True,
    )
    tracked = repository / 'tracked.txt'
    tracked.write_text('tracked\n', encoding='utf-8')
    subprocess.run(['git', '-C', str(repository), 'add', 'tracked.txt'], check=True)
    subprocess.run(
        ['git', '-C', str(repository), 'commit', '-q', '-m', 'initial'],
        check=True,
    )

    identity = dagger_module._repository_execution_identity(repository)
    assert identity['root'] == str(repository.resolve())
    assert identity['dirty'] is False
    assert len(identity['commit']) == len(identity['source_tree']) == 40

    nested = repository / 'nested'
    nested.mkdir()
    with pytest.raises(ValueError, match='toplevel|root'):
        dagger_module._repository_execution_identity(nested)

    (repository / 'untracked.txt').write_text('dirty\n', encoding='utf-8')
    assert dagger_module._repository_execution_identity(repository)['dirty'] is True


def test_oracle_preflight_rechecks_repository_identity_before_publication(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    calls = 0

    def drifting_provider(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls > 1:
            identity["source_tree"] = "e" * 40
        return identity

    root = tmp_path / "repository-drift"
    with pytest.raises(ValueError, match="repository identity changed"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=drifting_provider,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
        )
    assert calls == 2
    assert not root.exists()
    assert not root.with_name(root.name + ".staging").exists()
    assert len(list(
        root.with_name(root.name + ".diagnostics").glob(
            "attempt-*/diagnostic.json"
        )
    )) == 1


def test_oracle_preflight_rechecks_repository_identity_immediately_before_publish(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    calls = 0

    def third_check_drift(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls == 3:
            identity['commit'] = 'e' * 40
        return identity

    root = tmp_path / 'third-repository-drift'
    with pytest.raises(ValueError, match='repository identity changed'):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=third_check_drift,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
        )
    assert calls == 3
    assert not root.exists()


def test_oracle_preflight_rehashes_dataset_immediately_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    initial = {
        'content_sha256': definition.dataset_content_sha256,
        'file_count': definition.dataset_file_count,
        'byte_size': definition.dataset_byte_size,
        'audit': {
            'games': 1980,
            'teacher_labels': 199973,
            'masked_labels': 0,
            'round_trip_mismatches': 0,
            'replay_mismatches': 0,
        },
    }
    calls = 0

    def drifting_dataset(_definition):
        nonlocal calls
        calls += 1
        if calls == 1:
            return initial
        return {**initial, 'content_sha256': 'f' * 64}

    monkeypatch.setattr(dagger_module, '_audit_base_dataset', drifting_dataset)
    root = tmp_path / 'dataset-drift'
    with pytest.raises(ValueError, match='dataset.*changed|content identity'):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=evaluator,
            benchmark=benchmark,
            codec=codec,
            clock=clock,
        )
    assert calls == 2
    assert not root.exists()


def test_oracle_preflight_reuse_recomputes_reported_metrics_from_physical_games(
    tmp_path: Path,
) -> None:
    """A refreshed self-hash cannot make a false rate or selection authoritative."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )
    manifest_path = root / "oracle-preflight.json"
    original_manifest = manifest_path.read_bytes()
    manifest = json.loads(original_manifest)
    manifest["candidates"][0]["rates"]["win"] = 1.0
    manifest["selected_oracle"] = manifest["candidates"][0]["oracle"]
    manifest["content_identity"] = _content_identity(manifest)
    _rewrite(manifest_path, manifest)
    calls = 0

    def forbidden(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("invalid completed evidence must not launch work")

    with pytest.raises(ValueError, match="metric|rate|summary|reusable"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert calls == 0

    manifest_path.write_bytes(original_manifest)
    manifest = json.loads(original_manifest)
    benchmark_descriptor = manifest["games"][0]["benchmark"]
    benchmark_path = root / benchmark_descriptor["path"]
    original_benchmark = benchmark_path.read_bytes()
    benchmark_payload = json.loads(original_benchmark)
    benchmark_payload["records"][0]["first"]["decision"][
        "actual_expansion_count"
    ] += 1
    _rewrite(benchmark_path, benchmark_payload)
    benchmark_descriptor["sha256"] = _sha256(benchmark_path)
    benchmark_descriptor["byte_size"] = benchmark_path.stat().st_size
    manifest["content_identity"] = _content_identity(manifest)
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="metric|summary|reusable|determin"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
            on_selected=forbidden,
        )
    assert calls == 0

    benchmark_path.write_bytes(original_benchmark)
    manifest_path.write_bytes(original_manifest)
    manifest = json.loads(original_manifest)
    manifest["games"][1]["trace"] = manifest["games"][0]["trace"]
    manifest["content_identity"] = _content_identity(manifest)
    _rewrite(manifest_path, manifest)
    with pytest.raises(ValueError, match="duplicated|reusable|owned"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert calls == 0

    manifest_path.write_bytes(original_manifest)
    orphan = root / "games" / "orphan.json"
    orphan.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unowned|reusable|evidence"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert calls == 0


def test_oracle_preflight_recovers_completed_staging_and_rejects_coexistence(
    tmp_path: Path,
) -> None:
    """A crash after sealing is reusable, while two authoritative roots are not."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    selected = run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )
    staging = root.with_name(root.name + ".staging")
    os.replace(root, staging)
    calls = 0

    def forbidden(*_args):
        nonlocal calls
        calls += 1
        raise AssertionError("sealed staging reuse must not launch work")

    recovered = run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=forbidden,
        benchmark=forbidden,
        codec=forbidden,
    )
    assert recovered == selected
    assert root.is_dir()
    assert not staging.exists()
    assert calls == 0

    staging.mkdir()
    (staging / "sentinel.txt").write_text("do not overwrite\n", encoding="utf-8")
    before = (root / "oracle-preflight.json").read_bytes()
    with pytest.raises(ValueError, match="coexist|ambiguous"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=forbidden,
            benchmark=forbidden,
            codec=forbidden,
        )
    assert (root / "oracle-preflight.json").read_bytes() == before
    assert (staging / "sentinel.txt").read_text(encoding="utf-8") == "do not overwrite\n"
    assert calls == 0


def _owner_record(
    root: Path,
    *,
    pid: int,
    process_start_marker: str,
    owner_id: str = '11111111111111111111111111111111',
    created_ns: int = 1,
) -> dict[str, Any]:
    destination = str(root.resolve())
    return {
        'schema_version': 2,
        'destination': destination,
        'destination_identity': hashlib.sha256(
            destination.encode('utf-8'),
        ).hexdigest(),
        'owner_id': owner_id,
        'pid': pid,
        'process_start_marker': process_start_marker,
        'created_ns': created_ns,
    }


def test_oracle_preflight_live_owner_is_never_stolen_or_rotated(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / 'live-owner'
    staging = root.with_name(root.name + '.staging')
    lease_root = root.with_name(root.name + '.lock')
    owner = _owner_record(
        root,
        pid=os.getpid(),
        process_start_marker=dagger_module._preflight_process_start_marker_v3(
            os.getpid(),
        ),
    )
    lease_root.mkdir()
    _rewrite(lease_root / 'lease.json', owner)
    staging.mkdir()
    _rewrite(staging / '.owner.json', owner)
    sentinel = staging / 'sentinel.bin'
    sentinel.write_bytes(b'owned-by-live-process')
    before = sentinel.read_bytes()

    try:
        for _ in range(2):
            with pytest.raises(RuntimeError, match='live.*owner|lease'):
                run_oracle_preflight(
                    definition,
                    output_root=root,
                    repository_identity_provider=_repository_provider,
                    evaluator=lambda *_args: pytest.fail('evaluator must not run'),
                    benchmark=lambda *_args: pytest.fail('benchmark must not run'),
                    codec=lambda *_args: pytest.fail('codec must not run'),
                )
            assert sentinel.read_bytes() == before
            assert _read_owner(staging / '.owner.json') == owner
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if lease_root.exists():
            shutil.rmtree(lease_root)


def test_preflight_process_start_marker_distinguishes_live_and_absent_pid() -> None:
    marker = dagger_module._preflight_process_start_marker_v3(os.getpid())

    assert isinstance(marker, str)
    assert marker
    assert dagger_module._preflight_process_start_marker_v3((1 << 31) - 1) is None


def test_preflight_path_set_creates_and_revalidates_missing_parent_before_lease(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / 'missing' / 'nested' / 'preflight'

    destination, staging, diagnostics, lease = (
        dagger_module._validate_preflight_path_set_v2(definition, root)
    )

    assert destination == root.resolve()
    assert destination.parent.is_dir()
    assert destination.parent.resolve(strict=True) == destination.parent
    assert staging.parent == diagnostics.parent == lease.parent == destination.parent


def test_acquired_lease_records_complete_canonical_owner_identity(
    tmp_path: Path,
) -> None:
    root = (tmp_path / 'complete-owner').resolve()
    lease = dagger_module._acquire_preflight_lease_v2(root)
    try:
        payload = dict(lease.payload)
        assert set(payload) == {
            'schema_version', 'destination', 'destination_identity', 'owner_id',
            'pid', 'process_start_marker', 'created_ns',
        }
        assert payload['schema_version'] == 2
        assert payload['destination'] == str(root)
        assert payload['destination_identity'] == hashlib.sha256(
            str(root).encode('utf-8'),
        ).hexdigest()
        assert len(payload['owner_id']) == 32
        int(payload['owner_id'], 16)
        assert payload['pid'] == os.getpid()
        assert payload['process_start_marker'] == (
            dagger_module._preflight_process_start_marker_v3(os.getpid())
        )
        assert type(payload['created_ns']) is int and payload['created_ns'] > 0
    finally:
        dagger_module._release_preflight_lease_v2(lease)


def test_oracle_preflight_dead_owner_authorizes_exact_staging_recovery(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / 'dead-owner'
    staging = root.with_name(root.name + '.staging')
    lease_root = root.with_name(root.name + '.lock')
    stale_owner = _owner_record(
        root,
        pid=(1 << 31) - 1,
        process_start_marker='absent-process:start-1',
    )
    lease_root.mkdir()
    _rewrite(lease_root / 'lease.json', stale_owner)
    staging.mkdir()
    _rewrite(staging / '.owner.json', stale_owner)
    (staging / 'sentinel.bin').write_bytes(b'owned-by-dead-process')

    def fail(*_args):
        raise ConnectionError('new owner evaluation failed')

    with pytest.raises(ConnectionError, match='new owner evaluation failed'):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=fail,
            benchmark=lambda *_args: pytest.fail('benchmark must not run'),
            codec=lambda *_args: pytest.fail('codec must not run'),
        )

    assert not root.exists()
    assert not staging.exists()
    assert not lease_root.exists()
    diagnostics = root.with_name(root.name + '.diagnostics')
    attempts = sorted(diagnostics.glob('attempt-*/diagnostic.json'))
    assert len(attempts) == 3
    statuses = [json.loads(path.read_text(encoding='utf-8'))['status'] for path in attempts]
    assert statuses == ['stale-lease', 'failed', 'failed']
    assert any(
        (path.parent / 'sentinel.bin').read_bytes() == b'owned-by-dead-process'
        for path in attempts
        if (path.parent / 'sentinel.bin').is_file()
    )


def _mutate_owner_field(
    owner: dict[str, Any], field: str, root: Path,
) -> dict[str, Any]:
    changed = dict(owner)
    replacements = {
        'destination': str((root.parent / 'different-destination').resolve()),
        'destination_identity': 'f' * 64,
        'owner_id': '22222222222222222222222222222222',
        'pid': (1 << 31) - 2,
        'process_start_marker': 'absent-process:start-2',
        'created_ns': owner['created_ns'] + 1,
    }
    changed[field] = replacements[field]
    return changed


@pytest.mark.parametrize(
    'field',
    [
        'destination', 'destination_identity', 'owner_id', 'pid',
        'process_start_marker', 'created_ns',
    ],
)
def test_stale_staging_requires_the_exact_complete_owner_record(
    tmp_path: Path, field: str,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / f'stale-owner-mismatch-{field}'
    staging = root.with_name(root.name + '.staging')
    lease_root = root.with_name(root.name + '.lock')
    stale_owner = _owner_record(
        root,
        pid=(1 << 31) - 1,
        process_start_marker='absent-process:start-1',
    )
    lease_root.mkdir()
    _rewrite(lease_root / 'lease.json', stale_owner)
    staging.mkdir()
    _rewrite(
        staging / '.owner.json',
        _mutate_owner_field(stale_owner, field, root),
    )
    sentinel = staging / 'sentinel.bin'
    sentinel.write_bytes(b'mismatched-owner-must-remain')

    with pytest.raises(
        (RuntimeError, ValueError), match='owner|lease|destination|process',
    ):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=lambda *_args: pytest.fail('evaluator must not run'),
            benchmark=lambda *_args: pytest.fail('benchmark must not run'),
            codec=lambda *_args: pytest.fail('codec must not run'),
        )

    assert sentinel.read_bytes() == b'mismatched-owner-must-remain'
    assert staging.is_dir()
    assert not lease_root.exists()


@pytest.mark.parametrize(
    'field',
    [
        'destination', 'destination_identity', 'owner_id', 'pid',
        'process_start_marker', 'created_ns',
    ],
)
def test_release_mismatch_leaves_the_foreign_lease_untouched(
    tmp_path: Path, field: str,
) -> None:
    root = (tmp_path / f'release-mismatch-{field}').resolve()
    lease = dagger_module._acquire_preflight_lease_v2(root)
    foreign = _mutate_owner_field(dict(lease.payload), field, root)
    _rewrite(lease.root / 'lease.json', foreign)

    try:
        with pytest.raises(
            (RuntimeError, ValueError), match='owner|lease|destination|process',
        ):
            dagger_module._release_preflight_lease_v2(lease)
        assert _read_owner(lease.root / 'lease.json') == foreign
    finally:
        if lease.root.exists():
            shutil.rmtree(lease.root)


def test_pid_reuse_with_a_different_start_marker_is_not_the_live_owner(
    tmp_path: Path,
) -> None:
    root = (tmp_path / 'pid-reuse').resolve()
    old_owner = _owner_record(
        root,
        pid=os.getpid(),
        process_start_marker='windows-filetime:1',
    )
    lease_root = root.with_name(root.name + '.lock')
    lease_root.mkdir()
    _rewrite(lease_root / 'lease.json', old_owner)

    lease = dagger_module._acquire_preflight_lease_v2(root)
    try:
        assert [dict(owner) for owner in lease.stale_owners] == [old_owner]
        stale_diagnostics = list(
            root.with_name(root.name + '.diagnostics').glob(
                'attempt-*/diagnostic.json',
            )
        )
        assert len(stale_diagnostics) == 1
        assert json.loads(stale_diagnostics[0].read_text(encoding='utf-8'))[
            'lease'
        ] == old_owner
    finally:
        dagger_module._release_preflight_lease_v2(lease)


@pytest.mark.parametrize('case', ['missing', 'malformed', 'oversized'])
def test_owner_metadata_is_bounded_strict_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    root = (tmp_path / f'owner-metadata-{case}').resolve()
    metadata = tmp_path / f'{case}.owner.json'
    if case == 'malformed':
        metadata.write_bytes(b'{not-json')
    elif case == 'oversized':
        monkeypatch.setattr(dagger_module, '_MAX_PREFLIGHT_OWNER_BYTES', 32)
        _rewrite(metadata, _owner_record(
            root,
            pid=(1 << 31) - 1,
            process_start_marker='absent-process:start-1',
        ))

    with pytest.raises(ValueError, match='missing|invalid|byte|size|large'):
        dagger_module._read_preflight_owner_v3(
            metadata,
            destination=root,
            label='test owner metadata',
        )


@pytest.mark.parametrize(
    ('field', 'alias'),
    [
        ('schema_version', True),
        ('pid', 1.0),
        ('created_ns', False),
    ],
)
def test_owner_metadata_rejects_scalar_aliases_and_missing_fields(
    tmp_path: Path, field: str, alias: Any,
) -> None:
    root = (tmp_path / f'owner-alias-{field}').resolve()
    metadata = tmp_path / f'{field}.owner.json'
    owner = _owner_record(
        root,
        pid=(1 << 31) - 1,
        process_start_marker='absent-process:start-1',
    )
    owner[field] = alias
    _rewrite(metadata, owner)

    with pytest.raises(ValueError, match='integer|schema|boolean'):
        dagger_module._read_preflight_owner_v3(
            metadata,
            destination=root,
            label='test owner metadata',
        )

    owner = _owner_record(
        root,
        pid=(1 << 31) - 1,
        process_start_marker='absent-process:start-1',
    )
    owner.pop(field)
    _rewrite(metadata, owner)
    with pytest.raises(ValueError, match='fields'):
        dagger_module._read_preflight_owner_v3(
            metadata,
            destination=root,
            label='test owner metadata',
        )


@pytest.mark.parametrize('location', ['lease', 'staging-owner'])
def test_lease_and_staging_owner_metadata_reject_symlinked_inner_files(
    tmp_path: Path, location: str,
) -> None:
    root = (tmp_path / f'symlinked-{location}').resolve()
    target = tmp_path / f'{location}-target.json'
    owner = _owner_record(
        root,
        pid=(1 << 31) - 1,
        process_start_marker='absent-process:start-1',
    )
    _rewrite(target, owner)

    if location == 'lease':
        lease_root = root.with_name(root.name + '.lock')
        lease_root.mkdir()
        _symlink_or_skip_windows_privilege(lease_root / 'lease.json', target)
        try:
            with pytest.raises(ValueError, match='canonical|symlink|reparse'):
                dagger_module._acquire_preflight_lease_v2(root)
        finally:
            if lease_root.exists():
                shutil.rmtree(lease_root)
        return

    lease = dagger_module._acquire_preflight_lease_v2(root)
    staging = root.with_name(root.name + '.staging')
    staging.mkdir()
    _symlink_or_skip_windows_privilege(staging / '.owner.json', target)
    try:
        with pytest.raises(ValueError, match='canonical|symlink|reparse'):
            dagger_module._require_stale_staging_owner_v2(staging, lease=lease)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        dagger_module._release_preflight_lease_v2(lease)


def _read_owner(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8'))


def test_oracle_preflight_diagnostic_attempt_reservation_skips_collisions(
    tmp_path: Path,
) -> None:
    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / 'diagnostic-collision'
    diagnostics = root.with_name(root.name + '.diagnostics')
    diagnostics.mkdir()
    reservation = diagnostics / '.attempt-000000.reserve'
    reservation.write_bytes(b'live-reservation')
    collision = diagnostics / 'attempt-000001'
    collision.mkdir()
    sentinel = collision / 'sentinel.bin'
    sentinel.write_bytes(b'unrelated-attempt')

    def fail(*_args):
        raise ConnectionError('collision test failure')

    with pytest.raises(ConnectionError, match='collision test failure'):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=fail,
            benchmark=lambda *_args: pytest.fail('benchmark must not run'),
            codec=lambda *_args: pytest.fail('codec must not run'),
        )

    assert reservation.read_bytes() == b'live-reservation'
    assert sentinel.read_bytes() == b'unrelated-attempt'
    assert (diagnostics / 'attempt-000002' / 'diagnostic.json').is_file()
    assert not root.with_name(root.name + '.lock').exists()


@pytest.mark.parametrize('limit_kind', ['file-count', 'total-bytes', 'file-bytes'])
def test_diagnostic_rotation_rejects_over_limit_tree_without_moving_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, limit_kind: str,
) -> None:
    root = tmp_path / f'diagnostic-tree-{limit_kind}'
    staging = root.with_name(root.name + '.staging')
    staging.mkdir()
    (staging / 'first.bin').write_bytes(b'abc')
    (staging / 'second.bin').write_bytes(b'def')
    if limit_kind == 'file-count':
        monkeypatch.setattr(
            dagger_module, '_MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILES', 1,
        )
    elif limit_kind == 'total-bytes':
        monkeypatch.setattr(
            dagger_module, '_MAX_PREFLIGHT_DIAGNOSTIC_TREE_BYTES', 5,
        )
    else:
        monkeypatch.setattr(
            dagger_module, '_MAX_PREFLIGHT_DIAGNOSTIC_TREE_FILE_BYTES', 2,
        )

    with pytest.raises(ValueError, match='diagnostic.*(file|byte|size|limit)'):
        dagger_module._move_to_preflight_diagnostic_v2(
            staging,
            destination=root,
        )

    assert staging.is_dir()
    assert (staging / 'first.bin').read_bytes() == b'abc'
    diagnostics = root.with_name(root.name + '.diagnostics')
    assert not list(diagnostics.glob('attempt-*')) if diagnostics.exists() else True


def test_oracle_preflight_runtime_failure_remains_diagnostic_not_complete(
    tmp_path: Path,
) -> None:
    """An evaluator failure may leave evidence, but never a completed artifact."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    root = tmp_path / "oracle-preflight"
    callbacks: list[int] = []

    def failed_evaluator(*_args):
        raise ConnectionError("test evaluator disconnected")

    with pytest.raises(ConnectionError, match="disconnected"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=failed_evaluator,
            benchmark=lambda *_args: pytest.fail("benchmark must not run"),
            codec=lambda *_args: pytest.fail("codec must not run"),
            on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
        )

    assert not root.exists()
    assert not root.with_name(root.name + ".staging").exists()
    attempts = list(
        root.with_name(root.name + ".diagnostics").glob("attempt-*/diagnostic.json")
    )
    assert len(attempts) == 1
    diagnostic = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["exception"] == {
        "type": "ConnectionError",
        "message": "test evaluator disconnected",
    }
    assert callbacks == []

    with pytest.raises(ConnectionError, match="disconnected"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_identity_provider=_repository_provider,
            evaluator=failed_evaluator,
            benchmark=lambda *_args: pytest.fail("benchmark must not run"),
            codec=lambda *_args: pytest.fail("codec must not run"),
        )
    assert len(list(
        root.with_name(root.name + ".diagnostics").glob(
            "attempt-*/diagnostic.json"
        )
    )) == 2

    evaluator, benchmark, codec, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    selected = run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
        on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
    )
    assert selected.expansion_budget == 2048
    assert callbacks == [2048]
    assert len(list(
        root.with_name(root.name + ".diagnostics").glob(
            "attempt-*/diagnostic.json"
        )
    )) == 2


class _PhysicalIterationHarness:
    """Small physical Task 9 pipeline used to authenticate the k1-k3 chain."""

    def __init__(self, root: Path, runner: object) -> None:
        self.root = root
        self.runner = runner
        self.events: list[str] = []
        self.counts: Counter[str] = Counter()
        self._published_events: set[int] = set()
        self._clock_value = 0.0
        self.drift_phase: str | None = None
        self.repository = {
            "root": str(root.parent.resolve()),
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "dirty": False,
        }
        self.oracle = {
            "spec": {
                "oracle_type": "bounded-search",
                "depth": 4,
                "expansion_budget": 512,
                "use_heuristic": True,
                "heuristic_identity": "material-plus-pursuit-v1",
                "code_hash": "1" * 64,
            },
            "evidence_root": str((root.parent / "sealed-preflight").resolve()),
            "evidence_content_identity": "2" * 64,
            "evidence_class": "sealed-engine",
        }

        source = runner.dagger_domain.dagger_actor_source(1)
        source_run = Path(source.controller["source_run"])
        checkpoint = Path(source.controller["path"])
        source_manifest = json.loads(
            (source_run / "run.json").read_text(encoding="utf-8")
        )
        self.contract = dict(source_manifest["contract"])
        checkpoint_sha256 = source.checkpoint_sha256
        self.starting = {
            "source": source.to_dict(),
            "identity": {
                "checkpoint_path": source.controller["path"],
                "checkpoint_sha256": checkpoint_sha256,
                "source_run": source.controller["source_run"],
                "source_manifest_sha256": _sha256(source_run / "run.json"),
            },
            "published_actor_sha256": checkpoint_sha256,
        }

    def _drift_repository(self, phase: str) -> None:
        if self.drift_phase == phase:
            self.repository = {
                **self.repository,
                "commit": "c" * 40,
                "source_tree": "d" * 40,
            }

    def clock(self) -> float:
        self._clock_value += 1.0
        return self._clock_value

    def prepare(self, *, output_root: Path) -> dict[str, Any]:
        assert output_root == self.root
        self.events.append("prepare")
        output_root.mkdir(parents=True, exist_ok=True)
        _rewrite(output_root / "definition.json", {"status": "prepared"})
        return {"root": str(output_root), "status": "prepared"}

    def validate(
        self, *, output_root: Path, prepared: object,
    ) -> dict[str, Any]:
        assert prepared == {"root": str(output_root), "status": "prepared"}
        self.events.append("validate")
        validated = {
            "starting_learner": self.starting,
            "physical": {
                "starting_learner": self.starting["identity"],
                "contract": {
                    "version": self.contract["version"],
                    "contract_hash": self.contract["contract_hash"],
                    "encoding_hash": self.contract["encoding_hash"],
                    "observation_size": self.contract["observation_size"],
                    "action_size": self.contract["action_size"],
                    "action_regions": self.contract["semantics"]["action_regions"],
                },
            },
        }
        _rewrite(output_root / "validated.json", validated)
        return validated

    def preflight(
        self, *, output_root: Path, validated: object,
    ) -> dict[str, Any]:
        assert validated["starting_learner"] == self.starting
        self.events.append("oracle-preflight")
        preflight = {"selected_oracle": self.oracle}
        _rewrite(output_root / "preflight.json", preflight)
        return preflight

    def baseline(
        self, *, output_root: Path, validated: object, preflight: object,
    ) -> dict[str, str]:
        assert validated["starting_learner"] == self.starting
        assert preflight["selected_oracle"] == self.oracle
        self.events.append("starting-baseline")
        _rewrite(output_root / "baseline.json", {"status": "completed"})
        return {"status": "completed"}

    def load_iteration_context(
        self, *, index: int, output_root: Path,
    ) -> dict[str, Any]:
        assert output_root == self.root
        train_overlays = tuple(
            self.reopen_overlay(
                partition="train",
                index=prior,
                output_root=(
                    output_root / "iterations" / f"iteration-{prior}"
                    / "train-overlay"
                ),
                expected=None,
            )
            for prior in range(1, index)
        )
        validation_overlays = tuple(
            self.reopen_overlay(
                partition="validation",
                index=prior,
                output_root=(
                    output_root / "iterations" / f"iteration-{prior}"
                    / "validation-overlay"
                ),
                expected=None,
            )
            for prior in range(1, index)
        )
        preceding_actor = (
            None
            if index == 1
            else self.reopen_actor(
                index=index - 1,
                trained=(
                    output_root / "iterations" / f"iteration-{index - 1}"
                    / "actor"
                ),
            )
        )
        return {
            "validated": json.loads(
                (output_root / "validated.json").read_text(encoding="utf-8")
            ),
            "preflight": json.loads(
                (output_root / "preflight.json").read_text(encoding="utf-8")
            ),
            "preceding_actor": preceding_actor,
            "train_overlays": train_overlays,
            "validation_overlays": validation_overlays,
            "repository": dict(self.repository),
        }

    def repository_identity_provider(self, root: Path) -> dict[str, Any]:
        assert root == Path(self.repository["root"])
        return dict(self.repository)

    def resolve_incoming(
        self, *, index: int, validated: object, preceding_actor: object | None,
    ) -> dict[str, Any]:
        if index == 1:
            assert preceding_actor is None
            return validated["starting_learner"]
        assert preceding_actor is not None
        return preceding_actor["incoming"]

    def collection_metrics(self, *, partition: str, index: int) -> dict[str, Any]:
        row_count = 20_000 + index if partition == "train" else 2_000 + index
        return {
            "games": 2,
            "labels": row_count,
            "reason_counts": {
                "conversion": row_count,
                "favorable": index,
                "cycle_warning": index,
                "wasted_end_turn": 0,
            },
            "disagreements": index,
            "disagreement_reason_counts": {
                "conversion": index,
                "favorable": index,
                "cycle_warning": 0,
                "wasted_end_turn": 0,
            },
            "mean_expansions": 100.0,
            "max_expansions": 512,
        }

    def collect(
        self,
        *,
        partition: str,
        index: int,
        learner: object,
        oracle: object,
        output_root: Path,
    ) -> dict[str, Any]:
        assert oracle == self.oracle
        source_kind = learner["source"]["source_kind"]
        expected_kind = "snapshot" if index == 1 else "dagger_actor"
        assert source_kind == expected_kind
        self.counts[f"collect-{partition}"] += 1
        self.events.append(f"{partition}-collection-{index}:{source_kind}")
        output_root.mkdir(parents=True)
        artifact = output_root / "artifact.bin"
        artifact.write_bytes(f"{partition}-{index}\n".encode("ascii"))
        self._drift_repository(f"{partition}_collection")
        return {
            "partition": partition,
            "iteration": index,
            "root": output_root,
            "content_identity": _sha256(artifact),
            "row_count": (
                20_000 + index if partition == "train" else 2_000 + index
            ),
            "collection_metrics": self.collection_metrics(
                partition=partition, index=index,
            ),
        }

    def reopen_overlay(
        self,
        *,
        partition: str,
        index: int,
        output_root: Path,
        expected: object | None,
    ) -> dict[str, Any]:
        artifact = output_root / "artifact.bin"
        try:
            payload = artifact.read_bytes()
        except OSError as exc:
            raise ValueError(f"{partition} physical overlay is missing") from exc
        if payload != f"{partition}-{index}\n".encode("ascii"):
            raise ValueError(f"{partition} physical overlay is corrupt")
        reopened = {
            "partition": partition,
            "iteration": index,
            "root": output_root,
            "content_identity": _sha256(artifact),
            "row_count": (
                20_000 + index if partition == "train" else 2_000 + index
            ),
            "collection_metrics": self.collection_metrics(
                partition=partition, index=index,
            ),
        }
        if (
            expected is not None
            and expected["content_identity"] != reopened["content_identity"]
        ):
            raise ValueError(f"{partition} overlay identity changed")
        return reopened

    def build_corpus(
        self,
        *,
        index: int,
        train_overlays: tuple[object, ...],
        validation_overlays: tuple[object, ...],
    ) -> dict[str, Any]:
        assert len(train_overlays) == len(validation_overlays) == index
        assert all(item["partition"] == "train" for item in train_overlays)
        assert all(
            item["partition"] == "validation" for item in validation_overlays
        )
        self.counts["corpus"] += 1
        self.events.append(f"cumulative-corpus-{index}")
        self._drift_repository("corpus")
        return {
            "training": train_overlays,
            "held_out": validation_overlays,
        }

    def train(
        self,
        *,
        index: int,
        learner: object,
        corpus: object,
        output_root: Path,
    ) -> dict[str, Path]:
        source_kind = learner["source"]["source_kind"]
        assert len(corpus["training"]) == len(corpus["held_out"]) == index
        assert all(row["partition"] == "train" for row in corpus["training"])
        self.counts["train"] += 1
        self.events.append(f"actor-distillation-{index}:{source_kind}")
        checkpoint = output_root / "checkpoints" / "step_000000000.zip"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(f"checkpoint-{index}\n".encode("ascii"))
        _rewrite(output_root / "publication.json", {"iteration": index})
        (output_root / "actor-fixtures.npz").write_bytes(b"test-fixtures")
        _rewrite(output_root / "metrics.json", {"iteration": index})
        _rewrite(output_root / "scenario.json", {"environment": "tactical-v2"})
        checkpoint_sha256 = _sha256(checkpoint)
        actor_sha256 = hashlib.sha256(
            f"actor-state-{index}".encode("ascii")
        ).hexdigest()
        publication_sha256 = _sha256(output_root / "publication.json")
        verification = {
            "checkpoint_sha256": checkpoint_sha256,
            "actor_sha256": actor_sha256,
            "value_parameters_sha256": str(index + 3) * 64,
            "actor_fixtures_sha256": str(index + 4) * 64,
            "publication_metadata_sha256": publication_sha256,
            "contract_hash": self.contract["contract_hash"],
            "encoding_hash": self.contract["encoding_hash"],
            "observation_size": self.contract["observation_size"],
            "action_size": self.contract["action_size"],
            "comparison_rtol": 0.0,
            "comparison_atol": 0.0,
            "maximum_absolute_logit_difference": 0.0,
        }
        actor_initialization = learner["source"]
        run_manifest = {
            "schema_version": 1,
            "state": "completed",
            "production": True,
            "training_kind": "selective-dagger-distillation-v1",
            "distillation_iteration": index,
            "timesteps": 0,
            "latest_checkpoint": "checkpoints/step_000000000.zip",
            "latest_checkpoint_step": 0,
            "checkpoint_sha256": checkpoint_sha256,
            "target_actor_sha256_final": actor_sha256,
            "publication_metadata_sha256": publication_sha256,
            "actor_initialization": actor_initialization,
            "publication_verification": verification,
            "config": {
                "algorithm": "maskable_ppo",
                "policy": "HexCNN",
            },
            "contract": self.contract,
        }
        bc_manifest = {
            "schema_version": 1,
            "production": True,
            "training_kind": "selective-dagger-distillation-v1",
            "distillation_iteration": index,
            "algorithm": "maskable_ppo",
            "policy": "HexCNN",
            "checkpoint_sha256": checkpoint_sha256,
            "target_actor_sha256_final": actor_sha256,
            "publication_metadata_sha256": publication_sha256,
            "actor_initialization": actor_initialization,
            "publication_verification": verification,
        }
        _rewrite(output_root / "run.json", run_manifest)
        _rewrite(output_root / "bc.json", bc_manifest)
        self._drift_repository("train")
        return {"root": output_root}

    def reopen_actor(
        self, *, index: int, trained: object,
    ) -> dict[str, Any]:
        actor_root = (
            Path(trained["root"]) if isinstance(trained, dict) else Path(trained)
        )
        checkpoint = actor_root / "checkpoints" / "step_000000000.zip"
        publication_path = actor_root / "publication.json"
        run_path = actor_root / "run.json"
        bc_path = actor_root / "bc.json"
        try:
            checkpoint_payload = checkpoint.read_bytes()
            publication = json.loads(
                publication_path.read_text(encoding="utf-8")
            )
            run_manifest = json.loads(run_path.read_text(encoding="utf-8"))
            bc_manifest = json.loads(bc_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("actor physical publication is missing") from exc
        if (
            checkpoint_payload != f"checkpoint-{index}\n".encode("ascii")
            or publication != {"iteration": index}
            or run_manifest["distillation_iteration"] != index
            or bc_manifest["distillation_iteration"] != index
            or run_manifest["contract"] != self.contract
        ):
            raise ValueError("actor physical publication is corrupt")

        canonical_root = actor_root.resolve()
        canonical_checkpoint = checkpoint.resolve()
        checkpoint_sha256 = _sha256(checkpoint)
        actor_sha256 = hashlib.sha256(
            f"actor-state-{index}".encode("ascii")
        ).hexdigest()
        publication_sha256 = _sha256(publication_path)
        run_sha256 = _sha256(run_path)
        bc_sha256 = _sha256(bc_path)
        if (
            run_manifest["checkpoint_sha256"] != checkpoint_sha256
            or bc_manifest["checkpoint_sha256"] != checkpoint_sha256
            or run_manifest["target_actor_sha256_final"] != actor_sha256
            or bc_manifest["target_actor_sha256_final"] != actor_sha256
            or run_manifest["publication_metadata_sha256"] != publication_sha256
            or bc_manifest["publication_metadata_sha256"] != publication_sha256
        ):
            raise ValueError("actor physical publication hashes changed")
        if (
            actor_root.parent.name == f"iteration-{index}"
            and index not in self._published_events
        ):
            self._published_events.add(index)
            self.events.append(f"physical-publication-{index}")
        incoming = {
            "source": {
                "schema_version": 1,
                "source_kind": "dagger_actor",
                "controller": {
                    "kind": "snapshot",
                    "path": str(canonical_checkpoint),
                    "source_run": str(canonical_root),
                    "algorithm": "maskable_ppo",
                    "step": 0,
                    "inference_mode": "deterministic",
                },
                "checkpoint_sha256": checkpoint_sha256,
                "published_actor_sha256": actor_sha256,
            },
            "identity": {
                "checkpoint_path": str(canonical_checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
                "source_run": str(canonical_root),
                "source_manifest_sha256": run_sha256,
            },
            "published_actor_sha256": actor_sha256,
        }
        return {
            "root": canonical_root,
            "checkpoint_sha256": checkpoint_sha256,
            "actor_sha256": actor_sha256,
            "publication_metadata_sha256": publication_sha256,
            "run_manifest_sha256": run_sha256,
            "bc_manifest_sha256": bc_sha256,
            "incoming": incoming,
        }

    def validate_actor_supervision_publication(
        self, run_dir: Path, expected_contract: object,
    ) -> dict[str, Any]:
        if expected_contract.to_dict() != self.contract:
            raise ValueError("actor publication contract changed")
        run = json.loads((Path(run_dir) / "run.json").read_text(encoding="utf-8"))
        index = run["distillation_iteration"]
        actor = self.reopen_actor(index=index, trained=run_dir)
        verification = dict(run["publication_verification"])
        if (
            verification["checkpoint_sha256"] != actor["checkpoint_sha256"]
            or verification["actor_sha256"] != actor["actor_sha256"]
            or verification["publication_metadata_sha256"]
            != actor["publication_metadata_sha256"]
        ):
            raise ValueError("actor publication verification changed")
        self._drift_repository("authentication")
        return verification

    def build_iteration_identity(
        self,
        *,
        index: int,
        context: object,
        incoming: object,
        train_overlays: tuple[object, ...],
        validation_overlays: tuple[object, ...],
    ) -> dict[str, Any]:
        assert len(train_overlays) == len(validation_overlays) == index
        encoding_hash = self.contract["encoding_hash"]
        train_seed_start = 18_000_000 + (index - 1) * 100_000
        validation_seed_start = 19_000_000 + (index - 1) * 10_000

        def descriptors(overlays: tuple[object, ...]) -> list[dict[str, Any]]:
            return [
                {
                    "iteration": overlay["iteration"],
                    "content_identity": overlay["content_identity"],
                    "row_count": overlay["row_count"],
                }
                for overlay in overlays
            ]

        return {
            "predecessor": (
                None
                if index == 1
                else {
                    "iteration": index - 1,
                    "content_identity": json.loads(
                        (
                            self.root / "iterations"
                            / f"iteration-{index - 1}" / "manifest.json"
                        ).read_text(encoding="utf-8")
                    )["content_identity"],
                }
            ),
            "definition": {
                "panel_sha256": "3" * 64,
                "panel_byte_size": 101,
                "seed_banks_sha256": "4" * 64,
                "seed_banks_byte_size": 202,
            },
            "repository": dict(self.repository),
            "scenario": {
                "source_sha256": "5" * 64,
                "runtime_sha256": "6" * 64,
            },
            "contract": {
                "version": self.contract["version"],
                "contract_hash": self.contract["contract_hash"],
                "encoding_hash": encoding_hash,
                "observation_size": self.contract["observation_size"],
                "action_size": self.contract["action_size"],
                "action_regions": self.contract["semantics"]["action_regions"],
            },
            "base_dataset": {
                "root": str((self.root.parent / "base-dataset").resolve()),
                "manifest_sha256": "7" * 64,
                "content_sha256": "8" * 64,
                "file_count": 3966,
                "byte_size": 17_852_257,
                "contract_hash": "d" * 64,
                "encoding_hash": encoding_hash,
                "scenario_hash": "9" * 64,
            },
            "selected_oracle": context["preflight"]["selected_oracle"],
            "incoming_learner": incoming,
            "cumulative_train_overlays": descriptors(train_overlays),
            "cumulative_validation_overlays": descriptors(validation_overlays),
            "schedules": {
                "train": {
                    "sha256": str(index + 1) * 64,
                    "seed_start": train_seed_start,
                    "seed_stop": train_seed_start + 99_999,
                    "label_target": 20_000,
                    "game_ceiling": 2_000,
                },
                "validation": {
                    "sha256": str(index + 4) * 64,
                    "seed_start": validation_seed_start,
                    "seed_stop": validation_seed_start + 9_999,
                    "label_target": 2_000,
                    "game_ceiling": 200,
                },
            },
            "optimizer": {
                "source_mixture_basis_points": {
                    "greedy_standard": 4_900,
                    "search_conversion": 2_100,
                    "dagger_targeted": 3_000,
                },
                "batch_size": 256,
                "learning_rate": 3e-4,
                "max_epochs": 50,
                "patience": 5,
                "model_seed": 227,
                "sampler_seed": 227,
                "device": "cuda",
                "publication_device": "cpu",
                "objective": "actor_only_masked_cross_entropy",
                "validation_metric": "targeted_negative_log_likelihood",
            },
            "runtime": {
                "hardware": {
                    "training_device": "cuda:0",
                    "publication_device": "cpu",
                    "cuda_available": True,
                    "device_index": 0,
                    "cuda_runtime": "12.8",
                    "device_name": "test-gpu",
                },
                "software": {
                    "python": "3.11.test",
                    "implementation": "CPython",
                    "platform": "test-platform",
                    "executable": "C:/python/python.exe",
                    "numpy": "test",
                    "torch": "test",
                    "stable_baselines3": "test",
                    "sb3_contrib": "test",
                },
            },
        }

    def build_iteration_manifest(
        self,
        *,
        index: int,
        identity: object,
        train_overlay: object,
        validation_overlay: object,
        actor: object,
        timings: object,
    ) -> dict[str, Any]:
        source_counts = {
            1: {
                "greedy_standard": 9_910,
                "search_conversion": 4_247,
                "dagger_targeted": 6_067,
            },
            2: {
                "greedy_standard": 19_694,
                "search_conversion": 8_440,
                "dagger_targeted": 12_058,
            },
            3: {
                "greedy_standard": 29_478,
                "search_conversion": 12_634,
                "dagger_targeted": 18_048,
            },
        }
        training_rows = sum(
            item["row_count"] for item in identity["cumulative_train_overlays"]
        )
        validation_rows = sum(
            item["row_count"]
            for item in identity["cumulative_validation_overlays"]
        )

        def collection(overlay: object) -> dict[str, Any]:
            return json.loads(json.dumps(overlay["collection_metrics"]))

        payload = {
            "schema_version": 1,
            "status": "completed",
            "iteration": index,
            "identity": identity,
            "artifacts": {
                "train_overlay": {
                    "path": "train-overlay",
                    "content_identity": train_overlay["content_identity"],
                    "row_count": train_overlay["row_count"],
                },
                "validation_overlay": {
                    "path": "validation-overlay",
                    "content_identity": validation_overlay["content_identity"],
                    "row_count": validation_overlay["row_count"],
                },
                "actor": {
                    "path": "actor",
                    "checkpoint_sha256": actor["checkpoint_sha256"],
                    "actor_sha256": actor["actor_sha256"],
                    "publication_metadata_sha256": (
                        actor["publication_metadata_sha256"]
                    ),
                    "run_manifest_sha256": actor["run_manifest_sha256"],
                    "bc_manifest_sha256": actor["bc_manifest_sha256"],
                },
            },
            "metrics": {
                "train_collection": collection(train_overlay),
                "validation_collection": collection(validation_overlay),
                "training": {
                    "training_rows": training_rows,
                    "validation_rows": validation_rows,
                    "source_example_counts": source_counts[index],
                    "best_epoch": 1,
                    "best_validation_nll": 0.5,
                    "epochs_trained": 1,
                },
            },
            "timings": dict(timings),
        }
        payload["content_identity"] = _content_identity(payload)
        self._drift_repository("prepublish")
        return payload

    def dependencies(self) -> object:
        return self.runner.DaggerDependencies(
            prepare=self.prepare,
            validate=self.validate,
            preflight=self.preflight,
            baseline=self.baseline,
            resolve_incoming=self.resolve_incoming,
            collect=self.collect,
            build_corpus=self.build_corpus,
            train=self.train,
            reopen_actor=self.reopen_actor,
            load_iteration_context=self.load_iteration_context,
            reopen_overlay=self.reopen_overlay,
            build_iteration_identity=self.build_iteration_identity,
            build_iteration_manifest=self.build_iteration_manifest,
            repository_identity_provider=self.repository_identity_provider,
            clock=self.clock,
        )


def test_training_iteration_stage_order_preserves_learner_ownership_and_validation_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The physical chain owns k2/k3 inputs and exact reuse does no compute."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "selective-dagger"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    dependencies = harness.dependencies()
    manifests = runner.run_training_pipeline(
        output_root=root, dependencies=dependencies,
    )

    assert harness.events == [
        "prepare",
        "validate",
        "oracle-preflight",
        "starting-baseline",
        "validation-collection-1:snapshot",
        "train-collection-1:snapshot",
        "cumulative-corpus-1",
        "actor-distillation-1:snapshot",
        "physical-publication-1",
        "validation-collection-2:dagger_actor",
        "train-collection-2:dagger_actor",
        "cumulative-corpus-2",
        "actor-distillation-2:dagger_actor",
        "physical-publication-2",
        "validation-collection-3:dagger_actor",
        "train-collection-3:dagger_actor",
        "cumulative-corpus-3",
        "actor-distillation-3:dagger_actor",
        "physical-publication-3",
    ]
    assert [
        manifest.identity["incoming_learner"]["source"]["source_kind"]
        for manifest in manifests
    ] == ["snapshot", "dagger_actor", "dagger_actor"]
    assert [
        len(manifest.identity["cumulative_train_overlays"])
        for manifest in manifests
    ] == [1, 2, 3]
    assert [
        len(manifest.identity["cumulative_validation_overlays"])
        for manifest in manifests
    ] == [1, 2, 3]
    assert [
        manifest.metrics["train_collection"]["disagreement_reason_counts"]
        for manifest in manifests
    ] == [
        {
            "conversion": index,
            "favorable": index,
            "cycle_warning": 0,
            "wasted_end_turn": 0,
        }
        for index in (1, 2, 3)
    ]
    assert [
        manifest.metrics["validation_collection"][
            "disagreement_reason_counts"
        ]
        for manifest in manifests
    ] == [
        {
            "conversion": index,
            "favorable": index,
            "cycle_warning": 0,
            "wasted_end_turn": 0,
        }
        for index in (1, 2, 3)
    ]
    assert [
        Path(manifests[index].identity["incoming_learner"]["identity"]["source_run"])
        for index in (1, 2)
    ] == [
        (root / "iterations" / "iteration-1" / "actor").resolve(),
        (root / "iterations" / "iteration-2" / "actor").resolve(),
    ]

    compute_keys = (
        "collect-validation", "collect-train", "corpus", "train",
    )
    compute_counts = {key: harness.counts[key] for key in compute_keys}
    reused = runner.run_training_pipeline(
        output_root=root, dependencies=dependencies,
    )
    assert [item.content_identity for item in reused] == [
        item.content_identity for item in manifests
    ]
    assert {key: harness.counts[key] for key in compute_keys} == compute_counts


def test_run_iteration_rejects_manifest_collection_metrics_not_in_overlay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A builder cannot invent disagreement reasons absent from reopened data."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "collection-metric-tamper"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )

    def tampered_manifest(**kwargs: object) -> dict[str, Any]:
        payload = harness.build_iteration_manifest(**kwargs)
        payload["metrics"]["train_collection"][
            "disagreement_reason_counts"
        ]["conversion"] = 0
        payload["content_identity"] = _content_identity(payload)
        return payload

    dependencies = replace(
        harness.dependencies(),
        build_iteration_manifest=tampered_manifest,
    )

    with pytest.raises(
        ValueError,
        match="collection metrics do not match reopened overlay evidence",
    ):
        runner.run_training_pipeline(
            output_root=root,
            dependencies=dependencies,
        )

    assert not (root / "iterations" / "iteration-1").exists()
    assert (root / "iterations" / "iteration-1.staging").is_dir()


@pytest.mark.parametrize(
    "drift_phase",
    ["validation_collection", "train", "prepublish"],
)
def test_run_iteration_repository_drift_fails_before_publication(
    tmp_path: Path,
    drift_phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository change during any long phase cannot enter a manifest."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / f"repository-drift-{drift_phase}"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    harness.drift_phase = drift_phase

    with pytest.raises(ValueError, match="repository identity changed"):
        runner.run_training_pipeline(
            output_root=root,
            dependencies=harness.dependencies(),
        )

    assert not (root / "iterations" / "iteration-1").exists()
    assert (root / "iterations" / "iteration-1.staging").is_dir()
    if drift_phase == "validation_collection":
        assert harness.counts["collect-validation"] == 1
        assert harness.counts["collect-train"] == 0
        assert harness.counts["corpus"] == 0
        assert harness.counts["train"] == 0


def test_run_iteration_repository_drift_during_authentication_stops_before_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A long physical actor audit cannot hide repository drift before games."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "repository-drift-authentication"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    dependencies = harness.dependencies()
    prepared = harness.prepare(output_root=root)
    validated = harness.validate(output_root=root, prepared=prepared)
    preflight = harness.preflight(output_root=root, validated=validated)
    harness.baseline(
        output_root=root,
        validated=validated,
        preflight=preflight,
    )
    runner.run_iteration(1, output_root=root, dependencies=dependencies)

    compute_keys = (
        "collect-validation", "collect-train", "corpus", "train",
    )
    before = {key: harness.counts[key] for key in compute_keys}
    harness.drift_phase = "authentication"

    with pytest.raises(ValueError, match="repository identity changed"):
        runner.run_iteration(2, output_root=root, dependencies=dependencies)

    assert {key: harness.counts[key] for key in compute_keys} == before
    assert not (root / "iterations" / "iteration-2").exists()
    assert not (root / "iterations" / "iteration-2.staging").exists()


def test_run_iteration_rejects_self_consistent_wrong_predecessor_iteration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid iteration manifest cannot be rebound to another predecessor."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "wrong-predecessor-iteration"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    dependencies = harness.dependencies()
    prepared = harness.prepare(output_root=root)
    validated = harness.validate(output_root=root, prepared=prepared)
    preflight = harness.preflight(output_root=root, validated=validated)
    harness.baseline(
        output_root=root,
        validated=validated,
        preflight=preflight,
    )
    runner.run_iteration(1, output_root=root, dependencies=dependencies)
    runner.run_iteration(2, output_root=root, dependencies=dependencies)

    iteration_one_manifest = (
        root / "iterations" / "iteration-1" / "manifest.json"
    ).read_bytes()
    (root / "iterations" / "iteration-2" / "manifest.json").write_bytes(
        iteration_one_manifest
    )
    compute_keys = (
        "collect-validation", "collect-train", "corpus", "train",
    )
    before = {key: harness.counts[key] for key in compute_keys}

    with pytest.raises(
        ValueError, match="canonical predecessor iteration does not match",
    ):
        runner.run_iteration(3, output_root=root, dependencies=dependencies)

    assert {key: harness.counts[key] for key in compute_keys} == before
    assert not (root / "iterations" / "iteration-3").exists()
    assert not (root / "iterations" / "iteration-3.staging").exists()


@pytest.mark.parametrize(
    "tamper_mode",
    ["self-consistent-manifest", "cross-chain-manifest-and-actor"],
)
@pytest.mark.parametrize(
    ("predecessor_index", "next_index"),
    [(1, 2), (2, 3)],
)
def test_run_iteration_rejects_spliced_predecessor_chain_before_compute(
    tmp_path: Path,
    tamper_mode: str,
    predecessor_index: int,
    next_index: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """k2/k3 authenticate the recursively reconstructed physical causal chain."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "selective-dagger"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    dependencies = harness.dependencies()
    prepared = harness.prepare(output_root=root)
    validated = harness.validate(output_root=root, prepared=prepared)
    preflight = harness.preflight(output_root=root, validated=validated)
    harness.baseline(
        output_root=root,
        validated=validated,
        preflight=preflight,
    )
    for index in range(1, predecessor_index + 1):
        runner.run_iteration(index, output_root=root, dependencies=dependencies)

    predecessor_iteration = (
        root / "iterations" / f"iteration-{predecessor_index}"
    )
    manifest_path = predecessor_iteration / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["definition"]["panel_sha256"] = "f" * 64
    manifest["content_identity"] = _content_identity(manifest)

    if tamper_mode == "cross-chain-manifest-and-actor":
        foreign_iteration = (
            tmp_path / "foreign-experiment" / "iterations"
            / f"iteration-{predecessor_index}"
        )
        foreign_iteration.mkdir(parents=True)
        _rewrite(foreign_iteration / "manifest.json", manifest)
        shutil.copytree(
            predecessor_iteration / "actor",
            foreign_iteration / "actor",
        )
        shutil.copytree(
            foreign_iteration / "actor",
            predecessor_iteration / "actor",
            dirs_exist_ok=True,
        )
        manifest_path.write_bytes(
            (foreign_iteration / "manifest.json").read_bytes()
        )
    else:
        _rewrite(manifest_path, manifest)

    compute_keys = (
        "collect-validation", "collect-train", "corpus", "train",
    )
    before = {key: harness.counts[key] for key in compute_keys}

    with pytest.raises(ValueError, match="predecessor.*identity|iteration.*identity"):
        runner.run_iteration(next_index, output_root=root, dependencies=dependencies)

    assert {key: harness.counts[key] for key in compute_keys} == before
    assert not (root / "iterations" / f"iteration-{next_index}").exists()
    assert not (
        root / "iterations" / f"iteration-{next_index}.staging"
    ).exists()


def test_run_iteration_rejects_forged_external_prior_actor_before_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A suffix-compatible actor outside this experiment cannot own k2."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "selective-dagger"
    harness = _PhysicalIterationHarness(root, runner)
    monkeypatch.setattr(
        runner.imitation_domain,
        "validate_actor_supervision_publication",
        harness.validate_actor_supervision_publication,
    )
    dependencies = harness.dependencies()
    prepared = harness.prepare(output_root=root)
    validated = harness.validate(output_root=root, prepared=prepared)
    preflight = harness.preflight(output_root=root, validated=validated)
    harness.baseline(
        output_root=root,
        validated=validated,
        preflight=preflight,
    )
    runner.run_iteration(1, output_root=root, dependencies=dependencies)

    forged_actor_root = (
        tmp_path / "forged" / "iterations" / "iteration-1" / "actor"
    )
    forged_actor_root.parent.mkdir(parents=True)
    shutil.copytree(
        root / "iterations" / "iteration-1" / "actor",
        forged_actor_root,
    )
    forged_actor = harness.reopen_actor(
        index=1, trained=forged_actor_root,
    )

    def forged_context(
        *, index: int, output_root: Path,
    ) -> dict[str, Any]:
        context = harness.load_iteration_context(
            index=index, output_root=output_root,
        )
        return (
            {**context, "preceding_actor": forged_actor}
            if index == 2
            else context
        )

    forged_dependencies = replace(
        dependencies,
        load_iteration_context=forged_context,
    )
    compute_keys = (
        "collect-validation", "collect-train", "corpus", "train",
    )
    before = {key: harness.counts[key] for key in compute_keys}

    with pytest.raises(ValueError, match="canonical preceding actor"):
        runner.run_iteration(
            2,
            output_root=root,
            dependencies=forged_dependencies,
        )

    assert not (root / "iterations" / "iteration-2").exists()
    assert {key: harness.counts[key] for key in compute_keys} == before


def test_dagger_dependencies_cannot_omit_transactional_iteration_boundaries() -> None:
    """Production orchestration has no in-memory path around physical manifests."""

    import run_annihilation_selective_dagger as runner

    def boundary(*args: object, **kwargs: object) -> object:
        return None

    with pytest.raises(TypeError):
        runner.DaggerDependencies(
            prepare=boundary,
            validate=boundary,
            preflight=boundary,
            baseline=boundary,
            resolve_incoming=boundary,
            collect=boundary,
            build_corpus=boundary,
            train=boundary,
            reopen_actor=boundary,
        )


def test_run_iteration_collection_failure_stops_and_retains_unpublished_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed child remains diagnostic evidence and cannot look completed."""

    import run_annihilation_selective_dagger as runner

    events: list[str] = []
    root = tmp_path / "selective-dagger"
    repository = {
        "root": str(root.parent.resolve()),
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": False,
    }

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("a downstream callback ran after collection failed")

    def load_iteration_context(
        *, index: int, output_root: Path,
    ) -> dict[str, Any]:
        assert index == 1
        assert output_root == root
        events.append("context")
        return {
            "validated": {"starting_learner": {"learner_id": "step-38912"}},
            "preflight": {"selected_oracle": "depth-4-budget-512"},
            "preceding_actor": None,
            "train_overlays": (),
            "validation_overlays": (),
            "repository": repository,
        }

    def resolve_incoming(
        *, index: int, validated: object, preceding_actor: object | None,
    ) -> dict[str, str]:
        assert index == 1 and preceding_actor is None
        events.append("incoming")
        return validated["starting_learner"]

    def collect(
        *,
        partition: str,
        index: int,
        learner: object,
        oracle: object,
        output_root: Path,
    ) -> dict[str, Any]:
        events.append(f"collect-{partition}")
        assert index == 1
        assert learner["learner_id"] == "step-38912"
        assert oracle == "depth-4-budget-512"
        if partition == "train":
            raise RuntimeError("injected train collection failure")
        output_root.mkdir(parents=True)
        (output_root / "diagnostic.txt").write_text(
            "held-out collection completed\n", encoding="utf-8",
        )
        return {
            "partition": partition,
            "iteration": index,
            "root": output_root,
            "content_identity": "validation-1",
        }

    monkeypatch.setattr(
        runner,
        "_authenticate_iteration_incoming",
        lambda index, output_root, context, dependencies: (
            dependencies.resolve_incoming(
                index=index,
                validated=context["validated"],
                preceding_actor=context["preceding_actor"],
            )
        ),
    )
    dependencies = runner.DaggerDependencies(
        prepare=unexpected,
        validate=unexpected,
        preflight=unexpected,
        baseline=unexpected,
        resolve_incoming=resolve_incoming,
        collect=collect,
        build_corpus=unexpected,
        train=unexpected,
        reopen_actor=unexpected,
        load_iteration_context=load_iteration_context,
        reopen_overlay=unexpected,
        build_iteration_identity=unexpected,
        build_iteration_manifest=unexpected,
        repository_identity_provider=lambda _root: repository,
    )

    with pytest.raises(RuntimeError, match="injected train collection failure"):
        runner.run_iteration(1, output_root=root, dependencies=dependencies)

    iteration_root = root / "iterations" / "iteration-1"
    assert events == [
        "context",
        "incoming",
        "collect-validation",
        "collect-train",
    ]
    assert (
        iteration_root.with_name("iteration-1.staging")
        / "validation-overlay"
        / "diagnostic.txt"
    ).read_text(encoding="utf-8") == "held-out collection completed\n"
    assert not (iteration_root / "manifest.json").exists()


def test_run_iteration_exact_reuse_reopens_children_without_games_or_epochs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Completed reuse is physical; corrupt or missing children fail before work."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "selective-dagger"
    counts: Counter[str] = Counter()
    contract_hash = "5" * 64
    encoding_hash = "6" * 64
    train_identity = "e" * 64
    validation_identity = "f" * 64
    source_run = (
        "C:/Users/cddal/HexWars/python/runs/"
        "bc227-ppo-random-s227-20260802-v2"
    )
    checkpoint_path = f"{source_run}/checkpoints/step_000038912.zip"
    checkpoint_sha256 = (
        "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
    )
    incoming = {
        "source": {
            "schema_version": 1,
            "source_kind": "snapshot",
            "controller": {
                "kind": "snapshot",
                "path": checkpoint_path,
                "source_run": source_run,
                "algorithm": "maskable_ppo",
                "step": 38_912,
                "inference_mode": "deterministic",
            },
            "checkpoint_sha256": checkpoint_sha256,
        },
        "identity": {
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": checkpoint_sha256,
            "source_run": source_run,
            "source_manifest_sha256": (
                "7f02152c2ea39a08e5e203c0b0ba13928b2ad1847e276cc1b19f53331151ba46"
            ),
        },
        "published_actor_sha256": checkpoint_sha256,
    }
    selected_oracle = {
        "spec": {
            "oracle_type": "bounded-search",
            "depth": 4,
            "expansion_budget": 512,
            "use_heuristic": True,
            "heuristic_identity": "material-plus-pursuit-v1",
            "code_hash": "d" * 64,
        },
        "evidence_root": "C:/evidence/oracle-preflight",
        "evidence_content_identity": "1" * 64,
        "evidence_class": "sealed-engine",
    }
    repository = {
        "root": str(root.parent.resolve()),
        "commit": "a" * 40,
        "source_tree": "b" * 40,
        "dirty": False,
    }

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("unused pipeline boundary was invoked")

    def load_iteration_context(
        *, index: int, output_root: Path,
    ) -> dict[str, Any]:
        counts["context"] += 1
        assert index == 1 and output_root == root
        return {
            "validated": {"starting_learner": incoming},
            "preflight": {"selected_oracle": selected_oracle},
            "preceding_actor": None,
            "train_overlays": (),
            "validation_overlays": (),
            "repository": repository,
        }

    def resolve_incoming(
        *, index: int, validated: object, preceding_actor: object | None,
    ) -> dict[str, Any]:
        counts["incoming"] += 1
        assert index == 1 and preceding_actor is None
        return validated["starting_learner"]

    def collection_metrics(partition: str) -> dict[str, Any]:
        row_count = 20_001 if partition == "train" else 2_001
        return {
            "games": 2,
            "labels": row_count,
            "reason_counts": {
                "conversion": row_count,
                "favorable": 1,
                "cycle_warning": 1,
                "wasted_end_turn": 0,
            },
            "disagreements": 1,
            "disagreement_reason_counts": {
                "conversion": 1,
                "favorable": 1,
                "cycle_warning": 0,
                "wasted_end_turn": 0,
            },
            "mean_expansions": 100.0,
            "max_expansions": 512,
        }

    def collect(
        *,
        partition: str,
        index: int,
        learner: object,
        oracle: object,
        output_root: Path,
    ) -> dict[str, Any]:
        counts[f"collect-{partition}"] += 1
        assert index == 1 and learner is incoming and oracle is selected_oracle
        output_root.mkdir(parents=True)
        payload = f"{partition}-{index}\n".encode()
        (output_root / "artifact.bin").write_bytes(payload)
        return {
            "partition": partition,
            "iteration": index,
            "root": output_root,
            "content_identity": (
                train_identity if partition == "train" else validation_identity
            ),
            "row_count": 20_001 if partition == "train" else 2_001,
            "collection_metrics": collection_metrics(partition),
        }

    def reopen_overlay(
        *,
        partition: str,
        index: int,
        output_root: Path,
        expected: object | None,
    ) -> dict[str, Any]:
        counts[f"reopen-{partition}"] += 1
        artifact = output_root / "artifact.bin"
        try:
            payload = artifact.read_bytes()
        except OSError as exc:
            raise ValueError(f"{partition} physical overlay is missing") from exc
        if payload != f"{partition}-{index}\n".encode():
            raise ValueError(f"{partition} physical overlay is corrupt")
        result = {
            "partition": partition,
            "iteration": index,
            "root": output_root,
            "content_identity": (
                train_identity if partition == "train" else validation_identity
            ),
            "row_count": 20_001 if partition == "train" else 2_001,
            "collection_metrics": collection_metrics(partition),
        }
        if (
            expected is not None
            and expected["content_identity"] != result["content_identity"]
        ):
            raise ValueError(f"{partition} overlay identity changed")
        return result

    def build_corpus(
        *,
        index: int,
        train_overlays: tuple[object, ...],
        validation_overlays: tuple[object, ...],
    ) -> dict[str, Any]:
        counts["corpus"] += 1
        assert index == 1
        assert [item["partition"] for item in train_overlays] == ["train"]
        assert [item["partition"] for item in validation_overlays] == [
            "validation"
        ]
        return {"training": train_overlays, "held_out": validation_overlays}

    def train(
        *,
        index: int,
        learner: object,
        corpus: object,
        output_root: Path,
    ) -> dict[str, Any]:
        counts["train"] += 1
        assert index == 1 and learner is incoming
        assert corpus["training"][0]["partition"] == "train"
        output_root.mkdir(parents=True)
        (output_root / "actor.bin").write_bytes(b"actor-1\n")
        return {"root": output_root}

    def reopen_actor(*, index: int, trained: object) -> dict[str, Any]:
        counts["reopen-actor"] += 1
        actor_root = (
            Path(trained["root"]) if isinstance(trained, dict) else Path(trained)
        )
        try:
            actor_bytes = (actor_root / "actor.bin").read_bytes()
        except OSError as exc:
            raise ValueError("actor physical publication is missing") from exc
        if actor_bytes != b"actor-1\n":
            raise ValueError("actor physical publication is corrupt")
        return {
            "root": actor_root,
            "checkpoint_sha256": "2" * 64,
            "actor_sha256": "3" * 64,
            "publication_metadata_sha256": "4" * 64,
            "run_manifest_sha256": "5" * 64,
            "bc_manifest_sha256": "6" * 64,
        }

    def build_iteration_identity(
        *,
        index: int,
        context: Mapping[str, Any],
        incoming: object,
        train_overlays: tuple[object, ...],
        validation_overlays: tuple[object, ...],
    ) -> dict[str, Any]:
        counts["identity"] += 1
        assert index == 1 and incoming is context["validated"]["starting_learner"]
        return {
            "predecessor": None,
            "definition": {
                "panel_sha256": "7" * 64,
                "panel_byte_size": 101,
                "seed_banks_sha256": "8" * 64,
                "seed_banks_byte_size": 202,
            },
            "repository": repository,
            "scenario": {
                "source_sha256": "9" * 64,
                "runtime_sha256": "0" * 64,
            },
            "contract": {
                "version": "tactical-v2",
                "contract_hash": "d" * 64,
                "encoding_hash": encoding_hash,
                "observation_size": 1292,
                "action_size": 1288,
                "action_regions": {
                    "move": {"offset": 1, "count": 351},
                    "attack": {"offset": 352, "count": 351},
                    "deploy": {"offset": 703, "count": 585},
                },
            },
            "base_dataset": {
                "root": "C:/dataset",
                "manifest_sha256": "1" * 64,
                "content_sha256": "2" * 64,
                "file_count": 3966,
                "byte_size": 17_852_257,
                "contract_hash": contract_hash,
                "encoding_hash": encoding_hash,
                "scenario_hash": "3" * 64,
            },
            "selected_oracle": selected_oracle,
            "incoming_learner": incoming,
            "cumulative_train_overlays": [
                {
                    "iteration": item["iteration"],
                    "content_identity": item["content_identity"],
                    "row_count": item["row_count"],
                }
                for item in train_overlays
            ],
            "cumulative_validation_overlays": [
                {
                    "iteration": item["iteration"],
                    "content_identity": item["content_identity"],
                    "row_count": item["row_count"],
                }
                for item in validation_overlays
            ],
            "schedules": {
                "train": {
                    "sha256": "4" * 64,
                    "seed_start": 18_000_000,
                    "seed_stop": 18_099_999,
                    "label_target": 20_000,
                    "game_ceiling": 2_000,
                },
                "validation": {
                    "sha256": "5" * 64,
                    "seed_start": 19_000_000,
                    "seed_stop": 19_009_999,
                    "label_target": 2_000,
                    "game_ceiling": 200,
                },
            },
            "optimizer": {
                "source_mixture_basis_points": {
                    "greedy_standard": 4_900,
                    "search_conversion": 2_100,
                    "dagger_targeted": 3_000,
                },
                "batch_size": 256,
                "learning_rate": 3e-4,
                "max_epochs": 50,
                "patience": 5,
                "model_seed": 227,
                "sampler_seed": 227,
                "device": "cuda",
                "publication_device": "cpu",
                "objective": "actor_only_masked_cross_entropy",
                "validation_metric": "targeted_negative_log_likelihood",
            },
            "runtime": {
                "hardware": {
                    "training_device": "cuda:0",
                    "publication_device": "cpu",
                    "cuda_available": True,
                    "device_index": 0,
                    "cuda_runtime": "12.8",
                    "device_name": "test-gpu",
                },
                "software": {
                    "python": "3.11.test",
                    "implementation": "CPython",
                    "platform": "test-platform",
                    "executable": "C:/python/python.exe",
                    "numpy": "test",
                    "torch": "test",
                    "stable_baselines3": "test",
                    "sb3_contrib": "test",
                },
            },
        }

    def build_iteration_manifest(
        *,
        index: int,
        identity: Mapping[str, Any],
        train_overlay: Mapping[str, Any],
        validation_overlay: Mapping[str, Any],
        actor: Mapping[str, Any],
        timings: Mapping[str, Any],
    ) -> dict[str, Any]:
        counts["manifest"] += 1
        payload = {
            "schema_version": 1,
            "status": "completed",
            "iteration": index,
            "identity": dict(identity),
            "artifacts": {
                "train_overlay": {
                    "path": "train-overlay",
                    "content_identity": train_overlay["content_identity"],
                    "row_count": train_overlay["row_count"],
                },
                "validation_overlay": {
                    "path": "validation-overlay",
                    "content_identity": validation_overlay["content_identity"],
                    "row_count": validation_overlay["row_count"],
                },
                "actor": {
                    "path": "actor",
                    "checkpoint_sha256": actor["checkpoint_sha256"],
                    "actor_sha256": actor["actor_sha256"],
                    "publication_metadata_sha256": (
                        actor["publication_metadata_sha256"]
                    ),
                    "run_manifest_sha256": actor["run_manifest_sha256"],
                    "bc_manifest_sha256": actor["bc_manifest_sha256"],
                },
            },
            "metrics": {
                "train_collection": dict(train_overlay["collection_metrics"]),
                "validation_collection": dict(
                    validation_overlay["collection_metrics"]
                ),
                "training": {
                    "training_rows": train_overlay["row_count"],
                    "validation_rows": validation_overlay["row_count"],
                    "source_example_counts": {
                        "greedy_standard": 9_910,
                        "search_conversion": 4_247,
                        "dagger_targeted": 6_067,
                    },
                    "best_epoch": 1,
                    "best_validation_nll": 0.5,
                    "epochs_trained": 1,
                },
            },
            "timings": dict(timings),
        }
        payload["content_identity"] = _content_identity(payload)
        return payload

    monkeypatch.setattr(
        runner,
        "_authenticate_iteration_incoming",
        lambda index, output_root, context, dependencies: (
            dependencies.resolve_incoming(
                index=index,
                validated=context["validated"],
                preceding_actor=context["preceding_actor"],
            )
        ),
    )
    dependencies = runner.DaggerDependencies(
        prepare=unexpected,
        validate=unexpected,
        preflight=unexpected,
        baseline=unexpected,
        resolve_incoming=resolve_incoming,
        collect=collect,
        build_corpus=build_corpus,
        train=train,
        reopen_actor=reopen_actor,
        load_iteration_context=load_iteration_context,
        reopen_overlay=reopen_overlay,
        build_iteration_identity=build_iteration_identity,
        build_iteration_manifest=build_iteration_manifest,
        repository_identity_provider=lambda _root: repository,
    )

    first = runner.run_iteration(1, output_root=root, dependencies=dependencies)
    compute_counts = {
        name: counts[name]
        for name in ("collect-validation", "collect-train", "corpus", "train")
    }
    second = runner.run_iteration(1, output_root=root, dependencies=dependencies)

    assert second.content_identity == first.content_identity
    assert {
        name: counts[name] for name in compute_counts
    } == compute_counts

    completed_root = root
    root = tmp_path / "atomic-publication"
    real_replace = runner.os.replace
    staging = root / "iterations" / "iteration-1.staging"

    def fail_final_publication(source: object, destination: object) -> None:
        if Path(source) == staging:
            raise RuntimeError("injected final publication failure")
        real_replace(source, destination)

    monkeypatch.setattr(runner.os, "replace", fail_final_publication)
    with pytest.raises(RuntimeError, match="final publication failure"):
        runner.run_iteration(1, output_root=root, dependencies=dependencies)

    assert not (root / "iterations" / "iteration-1").exists()
    assert (staging / "manifest.json").is_file()
    assert (staging / "validation-overlay" / "artifact.bin").is_file()
    assert (staging / "train-overlay" / "artifact.bin").is_file()
    assert (staging / "actor" / "actor.bin").is_file()
    compute_counts = {
        name: counts[name]
        for name in ("collect-validation", "collect-train", "corpus", "train")
    }
    root = completed_root
    iteration_root = root / "iterations" / "iteration-1"
    assert (iteration_root / "manifest.json").is_file()
    assert not (iteration_root / ".staging").exists()

    actor_path = iteration_root / "actor" / "actor.bin"
    actor_path.write_bytes(b"corrupt\n")
    with pytest.raises(ValueError, match="actor physical publication is corrupt"):
        runner.run_iteration(1, output_root=root, dependencies=dependencies)
    assert {
        name: counts[name] for name in compute_counts
    } == compute_counts

    actor_path.write_bytes(b"actor-1\n")
    (iteration_root / "train-overlay" / "artifact.bin").unlink()
    with pytest.raises(ValueError, match="train physical overlay is missing"):
        runner.run_iteration(1, output_root=root, dependencies=dependencies)
    assert {
        name: counts[name] for name in compute_counts
    } == compute_counts


def test_prepare_copies_exact_definition_bytes_and_never_repairs_completed_output(
    tmp_path: Path,
) -> None:
    """Prepare is a publish-once byte snapshot, not a mutable configuration copy."""

    import run_annihilation_selective_dagger as runner

    output_root = tmp_path / "selective-dagger"
    first = runner.run_prepare(
        output_root=output_root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    copied_panel = output_root / "definition" / "panel.json"
    copied_seeds = output_root / "definition" / "seed-banks.json"

    assert copied_panel.read_bytes() == PANEL_PATH.read_bytes()
    assert copied_seeds.read_bytes() == SEED_BANKS_PATH.read_bytes()
    assert first.identity["definition"] == {
        "panel_sha256": _sha256(PANEL_PATH),
        "panel_byte_size": PANEL_PATH.stat().st_size,
        "seed_banks_sha256": _sha256(SEED_BANKS_PATH),
        "seed_banks_byte_size": SEED_BANKS_PATH.stat().st_size,
    }
    assert first.identity["repository"] == _repository_provider(ROOT)

    first_manifest_bytes = (
        output_root / "definition" / "manifest.json"
    ).read_bytes()
    second = runner.run_prepare(
        output_root=output_root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    assert second.content_identity == first.content_identity
    assert (
        output_root / "definition" / "manifest.json"
    ).read_bytes() == first_manifest_bytes
    copied_panel.write_bytes(copied_panel.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="panel physical bytes changed"):
        runner.run_prepare(
            output_root=output_root,
            panel_path=PANEL_PATH,
            repository_root=ROOT,
            repository_identity_provider=_repository_provider,
        )
    assert copied_panel.read_bytes().endswith(b"\n\n")
    assert (
        output_root / "definition" / "manifest.json"
    ).read_bytes() == first_manifest_bytes


def test_prepare_rechecks_repository_identity_immediately_before_publish(
    tmp_path: Path,
) -> None:
    """A prepared definition cannot straddle a repository identity change."""

    import run_annihilation_selective_dagger as runner

    calls = 0

    def drifting_provider(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls == 2:
            identity["commit"] = "e" * 40
        return identity

    output_root = tmp_path / "repository-drift"
    with pytest.raises(ValueError, match="repository identity changed"):
        runner.run_prepare(
            output_root=output_root,
            panel_path=PANEL_PATH,
            repository_root=ROOT,
            repository_identity_provider=drifting_provider,
        )

    assert calls == 2
    assert not (output_root / "definition").exists()
    assert (output_root / "definition.staging").is_dir()


def test_prepare_allows_the_documented_ignored_output_tree_inside_repository(
    tmp_path: Path,
) -> None:
    """Production artifacts may live in the documented python/runs location."""

    import run_annihilation_selective_dagger as runner

    repository_root = tmp_path / "repository"
    panel_path = (
        repository_root
        / "python"
        / "panels"
        / "annihilation-selective-dagger-v1"
        / "panel.json"
    )
    panel_path.parent.mkdir(parents=True)
    panel_path.write_bytes(b'{"panel":1}\n')
    (panel_path.parent / "seed-banks.json").write_bytes(b'{"seeds":1}\n')

    def clean_repository(root: Path) -> dict[str, Any]:
        return {
            "root": str(root.resolve(strict=True)),
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "dirty": False,
        }

    output_root = repository_root / "python" / "runs" / "selective-dagger"
    prepared = runner.run_prepare(
        output_root=output_root,
        panel_path=panel_path,
        repository_root=repository_root,
        repository_identity_provider=clean_repository,
    )

    assert prepared.root == (output_root / "definition").resolve(strict=True)
    assert prepared.panel_path.read_bytes() == b'{"panel":1}\n'


def test_validate_reopens_physical_inputs_without_games_and_reuses_only_exact_identity(
    tmp_path: Path,
) -> None:
    """Validation authenticates learner/data/runtime inputs but launches no duel."""

    import run_annihilation_selective_dagger as runner

    output_root = tmp_path / "selective-dagger"
    runner.run_prepare(
        output_root=output_root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    counts: Counter[str] = Counter()
    drift = {"dataset": False}

    def physical_validator(
        prepared: runner.PreparedStage,
    ) -> dict[str, Any]:
        counts["physical-reopen"] += 1
        assert prepared.panel_path.read_bytes() == PANEL_PATH.read_bytes()
        assert prepared.seed_banks_path.read_bytes() == SEED_BANKS_PATH.read_bytes()
        return {
            "starting_learner": {
                "checkpoint_path": "C:/evidence/checkpoints/step_000038912.zip",
                "checkpoint_sha256": "a" * 64,
                "source_run": "C:/evidence/source-run",
                "source_manifest_sha256": "b" * 64,
            },
            "base_dataset": {
                "root": "C:/evidence/base-dataset",
                "manifest_sha256": "c" * 64,
                "content_sha256": ("d" if not drift["dataset"] else "e") * 64,
                "file_count": 3966,
                "byte_size": 17_852_257,
                "contract_hash": "9" * 64,
                "encoding_hash": "1" * 64,
                "scenario_hash": "2" * 64,
            },
            "scenario": {
                "source_sha256": "3" * 64,
                "runtime_sha256": "4" * 64,
            },
            "contract": {
                "version": "tactical-v2",
                "contract_hash": "f" * 64,
                "encoding_hash": "1" * 64,
                "observation_size": 1292,
                "action_size": 1288,
                "action_regions": {
                    "move": {"offset": 1, "count": 351},
                    "attack": {"offset": 352, "count": 351},
                    "deploy": {"offset": 703, "count": 585},
                },
            },
            "seed_isolation": {
                "definition_count": len(SEED_RANGES),
                "overlap_count": 0,
                "final_bank_touched": False,
            },
        }

    def runtime_probe() -> dict[str, Any]:
        counts["runtime-probe"] += 1
        return {
            "hardware": {
                "training_device": "cuda:0",
                "publication_device": "cpu",
                "cuda_available": True,
                "device_index": 0,
                "device_name": "test-gpu",
                "cuda_runtime": "12.8",
            },
            "software": {
                "python": "3.11.test",
                "implementation": "CPython",
                "platform": "test-windows",
                "executable": "C:/python/python.exe",
                "numpy": "test",
                "torch": "test",
                "stable_baselines3": "test",
                "sb3_contrib": "test",
            },
        }

    first = runner.run_validate(
        output_root=output_root,
        physical_validator=physical_validator,
        runtime_probe=runtime_probe,
        repository_identity_provider=_repository_provider,
    )
    manifest_path = output_root / "validation" / "manifest.json"
    first_bytes = manifest_path.read_bytes()
    second = runner.run_validate(
        output_root=output_root,
        physical_validator=physical_validator,
        runtime_probe=runtime_probe,
        repository_identity_provider=_repository_provider,
    )

    assert second.content_identity == first.content_identity
    assert (
        first.physical["base_dataset"]["contract_hash"]
        != first.physical["contract"]["contract_hash"]
    )
    assert manifest_path.read_bytes() == first_bytes
    assert counts == {"physical-reopen": 2, "runtime-probe": 2}
    assert not (output_root / "iterations").exists()

    drift["dataset"] = True
    with pytest.raises(ValueError, match="validation identity differs"):
        runner.run_validate(
            output_root=output_root,
            physical_validator=physical_validator,
            runtime_probe=runtime_probe,
            repository_identity_provider=_repository_provider,
        )
    assert manifest_path.read_bytes() == first_bytes
    assert counts == {"physical-reopen": 3, "runtime-probe": 3}


@pytest.mark.parametrize("completed", [False, True])
def test_validate_rechecks_repository_identity_before_publish_or_reuse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed: bool,
) -> None:
    """Validation cannot publish or reuse evidence across a repository change."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / ("completed" if completed else "fresh")
    runner.run_prepare(
        output_root=root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    monkeypatch.setattr(runner, "_validate_physical_identity", lambda value: value)
    monkeypatch.setattr(runner, "_validate_runtime_identity", lambda value: value)
    boundaries = {
        "physical_validator": lambda _prepared: {"physical": "authenticated"},
        "runtime_probe": lambda: {"runtime": "authenticated"},
    }
    if completed:
        runner.run_validate(
            output_root=root,
            repository_identity_provider=_repository_provider,
            **boundaries,
        )
        completed_bytes = (root / "validation" / "manifest.json").read_bytes()

    calls = 0

    def drifting_provider(repository_root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(repository_root)
        if calls == 2:
            identity["source_tree"] = "e" * 40
        return identity

    with pytest.raises(ValueError, match="repository identity changed"):
        runner.run_validate(
            output_root=root,
            repository_identity_provider=drifting_provider,
            **boundaries,
        )

    assert calls == 2
    if completed:
        assert (root / "validation" / "manifest.json").read_bytes() == completed_bytes
        assert not (root / "validation.staging").exists()
    else:
        assert not (root / "validation").exists()
        assert (root / "validation.staging").is_dir()


def test_prepared_stage_rejects_rehashed_dirty_repository_before_validation(
    tmp_path: Path,
) -> None:
    """A forged content hash cannot turn dirty prepared provenance into evidence."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "dirty-prepared"
    runner.run_prepare(
        output_root=root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    manifest_path = root / "definition" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["repository"]["dirty"] = True
    manifest["content_identity"] = _content_identity(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="clean repository"):
        runner.run_validate(
            output_root=root,
            repository_identity_provider=lambda _root: (
                (_ for _ in ()).throw(
                    AssertionError("dirty prepared stage queried the repository")
                )
            ),
            physical_validator=lambda _prepared: (
                (_ for _ in ()).throw(
                    AssertionError("dirty prepared stage opened physical inputs")
                )
            ),
            runtime_probe=lambda: (
                (_ for _ in ()).throw(
                    AssertionError("dirty prepared stage probed runtime")
                )
            ),
        )


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("hardware", "training_device", "cpu"),
        ("hardware", "publication_device", "cuda:0"),
        ("hardware", "cuda_available", 1),
        ("hardware", "device_index", True),
        ("hardware", "device_name", ""),
        ("hardware", "cuda_runtime", None),
        ("software", "python", ""),
        ("software", "unexpected", "extra"),
    ],
)
def test_validation_runtime_identity_is_exact_cuda_to_cpu_provenance(
    section: str,
    field: str,
    value: object,
) -> None:
    """Loose runtime dictionaries cannot satisfy a production validation stage."""

    import run_annihilation_selective_dagger as runner

    runtime = {
        "hardware": {
            "training_device": "cuda:0",
            "publication_device": "cpu",
            "cuda_available": True,
            "device_index": 0,
            "device_name": "test-gpu",
            "cuda_runtime": "12.8",
        },
        "software": {
            "python": "3.11.test",
            "implementation": "CPython",
            "platform": "test-windows",
            "executable": "C:/python/python.exe",
            "numpy": "test",
            "torch": "test",
            "stable_baselines3": "test",
            "sb3_contrib": "test",
        },
    }
    runtime[section][field] = value

    with pytest.raises(ValueError, match="runtime identity"):
        runner._validate_runtime_identity(runtime)


def test_production_preflight_uses_only_the_public_sealed_engine_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Task 9 must never promote Task 8's private transcript seam."""

    import run_annihilation_selective_dagger as runner

    calls: list[tuple[object, Path, object]] = []
    session = object()
    definition = object()

    def public_preflight(
        supplied_definition: object,
        *,
        output_root: Path,
        execution_session: object | None = None,
    ) -> dict[str, Any]:
        calls.append((supplied_definition, output_root, execution_session))
        return {
            "selected_oracle": "depth-4-budget-512",
            "evidence_class": "sealed-engine",
        }

    def forbidden_private(*args: object, **kwargs: object) -> object:
        raise AssertionError("private callback transcript boundary was invoked")

    monkeypatch.setattr(
        dagger_module, "run_oracle_preflight", public_preflight,
    )
    monkeypatch.setattr(
        dagger_module, "_run_oracle_preflight_for_test", forbidden_private,
    )

    selected = runner.run_sealed_oracle_preflight(
        definition=definition,
        output_root=tmp_path / "preflight",
        execution_session_factory=lambda **kwargs: session,
        repository_root=ROOT,
    )
    assert selected["evidence_class"] == "sealed-engine"
    assert calls == [(definition, tmp_path / "preflight", session)]

    with pytest.raises(
        RuntimeError, match="sealed engine execution-session factory",
    ):
        runner.run_sealed_oracle_preflight(
            definition=definition,
            output_root=tmp_path / "blocked",
            execution_session_factory=None,
            repository_root=ROOT,
        )
    assert len(calls) == 1

    with pytest.raises(ValueError, match="outside the repository"):
        runner.run_sealed_oracle_preflight(
            definition=definition,
            output_root=ROOT / "python" / "runs" / "in-repository-preflight",
            execution_session_factory=lambda **kwargs: (
                (_ for _ in ()).throw(
                    AssertionError("in-repository preflight opened execution")
                )
            ),
            repository_root=ROOT,
        )
    assert len(calls) == 1


def test_production_validation_boundary_reopens_panel_and_full_base_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The default physical validator delegates to accepted strict audits."""

    import run_annihilation_selective_dagger as runner

    output_root = tmp_path / "selective-dagger"
    prepared = runner.run_prepare(
        output_root=output_root,
        panel_path=PANEL_PATH,
        repository_root=ROOT,
        repository_identity_provider=_repository_provider,
    )
    source = SimpleNamespace(
        controller={
            "path": "C:/evidence/checkpoints/step_000038912.zip",
            "source_run": "C:/evidence/source-run",
        },
        checkpoint_sha256="a" * 64,
    )
    definition = SimpleNamespace(
        repository_root=ROOT.resolve(),
        starting_learner=source,
        learner_source_manifest_sha256="b" * 64,
        dataset_root=Path("C:/evidence/base-dataset"),
        dataset_manifest_sha256="c" * 64,
        dataset_content_sha256="d" * 64,
        dataset_file_count=3966,
        dataset_byte_size=17_852_257,
        dataset_contract_hash="a" * 64,
        dataset_encoding_hash="f" * 64,
        dataset_scenario_hash="1" * 64,
        scenario_sha256="2" * 64,
        runtime_scenario_sha256="3" * 64,
        contract_hash="e" * 64,
        encoding_hash="f" * 64,
        observation_size=1292,
        action_size=1288,
        action_regions={
            "move": {"offset": 1, "count": 351},
            "attack": {"offset": 352, "count": 351},
            "deploy": {"offset": 703, "count": 585},
        },
        seed_banks=tuple(range(len(SEED_RANGES))),
    )
    calls: list[object] = []

    def load(
        path: Path, *, repository_root: Path,
    ) -> object:
        calls.append(("load", path, repository_root))
        return definition

    def validate(supplied: object) -> None:
        calls.append(("validate", supplied))

    def audit(supplied: object) -> dict[str, Any]:
        calls.append(("audit", supplied))
        return {
            "content_sha256": definition.dataset_content_sha256,
            "file_count": definition.dataset_file_count,
            "byte_size": definition.dataset_byte_size,
            "audit": {
                "games": 1980,
                "teacher_labels": 199_973,
                "masked_labels": 0,
                "round_trip_mismatches": 0,
                "replay_mismatches": 0,
            },
        }

    monkeypatch.setattr(dagger_module, "load_panel_definition", load)
    monkeypatch.setattr(dagger_module, "validate_panel_definition", validate)
    monkeypatch.setattr(dagger_module, "audit_base_dataset", audit)

    identity = runner.production_physical_validator(prepared)

    assert calls == [
        ("load", prepared.panel_path, ROOT.resolve()),
        ("validate", definition),
        ("audit", definition),
    ]
    assert identity["starting_learner"]["checkpoint_sha256"] == "a" * 64
    assert identity["base_dataset"]["content_sha256"] == "d" * 64
    assert identity["base_dataset"]["contract_hash"] != identity["contract"]["contract_hash"]
    assert identity["base_dataset"]["encoding_hash"] == identity["contract"]["encoding_hash"]
    assert identity["seed_isolation"] == {
        "definition_count": len(SEED_RANGES),
        "overlap_count": 0,
        "final_bank_touched": False,
    }


def test_training_pipeline_dispatches_physical_iterations_transactionally(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The production-shaped top level must route each k through run_iteration."""

    import run_annihilation_selective_dagger as runner

    root = tmp_path / "selective-dagger"
    events: list[str] = []

    def prepare(*, output_root: Path) -> str:
        assert output_root == root
        events.append("prepare")
        return "prepared"

    def validate(*, output_root: Path, prepared: object) -> str:
        assert output_root == root and prepared == "prepared"
        events.append("validate")
        return "validated"

    def preflight(*, output_root: Path, validated: object) -> str:
        assert output_root == root and validated == "validated"
        events.append("preflight")
        return "preflight"

    def baseline(
        *, output_root: Path, validated: object, preflight: object,
    ) -> str:
        assert validated == "validated" and preflight == "preflight"
        events.append("baseline")
        return "baseline"

    def physical_iteration(
        index: int,
        *,
        output_root: Path,
        dependencies: runner.DaggerDependencies,
    ) -> str:
        assert output_root == root
        events.append(f"iteration-{index}")
        return f"manifest-{index}"

    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("in-memory orchestration path was invoked")

    monkeypatch.setattr(runner, "run_iteration", physical_iteration)
    dependencies = runner.DaggerDependencies(
        prepare=prepare,
        validate=validate,
        preflight=preflight,
        baseline=baseline,
        resolve_incoming=unexpected,
        collect=unexpected,
        build_corpus=unexpected,
        train=unexpected,
        reopen_actor=unexpected,
        load_iteration_context=lambda **kwargs: {},
        reopen_overlay=unexpected,
        build_iteration_identity=unexpected,
        build_iteration_manifest=unexpected,
        repository_identity_provider=_repository_provider,
    )

    manifests = runner.run_training_pipeline(
        output_root=root,
        dependencies=dependencies,
    )

    assert manifests == ("manifest-1", "manifest-2", "manifest-3")
    assert events == [
        "prepare",
        "validate",
        "preflight",
        "baseline",
        "iteration-1",
        "iteration-2",
        "iteration-3",
    ]


def _task10_definition(
    tmp_path: Path, *, physical_overlays: bool = False,
    locked_baseline: bool = False,
) -> object:
    train_overlay_ids: list[str] = []
    validation_overlay_ids: list[str] = []
    if physical_overlays:
        import importlib.util

        fixture_path = Path(__file__).with_name("test_dagger.py")
        spec = importlib.util.spec_from_file_location(
            "_task10_physical_overlay_fixture", fixture_path,
        )
        if spec is None or spec.loader is None:
            raise AssertionError("physical overlay fixture module is unavailable")
        fixture_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixture_module)
        _seal_pair = fixture_module._seal_pair

        contract = EnvironmentContract(
            version="tactical-v2",
            contract_hash="c" * 64,
            encoding_hash="e" * 64,
            observation_size=2,
            action_size=7,
            board={"width": 2, "height": 1},
            roster=["one"],
            reward={},
            semantics={
                "action_regions": {
                    "move": {"offset": 1, "count": 2},
                    "attack": {"offset": 3, "count": 2},
                    "deploy": {"offset": 5, "count": 2},
                },
                "start_profiles": [
                    {"id": "standard-3v3"},
                    {"id": "conversion-3v1-near"},
                    {"id": "conversion-3v1-far"},
                    {"id": "conversion-2v1-near"},
                    {"id": "conversion-2v1-far"},
                    {"id": "conversion-1v1-near"},
                    {"id": "conversion-1v1-far"},
                ],
            },
        )
        for overlay_iteration in (1, 2, 3):
            train, _ = _seal_pair(
                tmp_path / f"train-overlay-{overlay_iteration}",
                contract,
                partition="train",
                iteration=overlay_iteration,
            )
            validation, _ = _seal_pair(
                tmp_path / f"heldout-overlay-{overlay_iteration}",
                contract,
                partition="validation",
                iteration=overlay_iteration,
            )
            train_overlay_ids.append(train.content_identity)
            validation_overlay_ids.append(validation.content_identity)
    candidates = []
    for iteration in range(4):
        actor = tmp_path / f"task10-actor-{iteration}"
        checkpoint_dir = actor / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        step = 38_912 if iteration == 0 else 0
        checkpoint = checkpoint_dir / f"step_{step:09d}.zip"
        checkpoint.write_bytes(f"task10-actor-{iteration}".encode("ascii"))
        if locked_baseline and iteration == 0:
            locked_checkpoint = Path(
                "C:/Users/cddal/HexWars/python/runs/"
                "bc227-ppo-random-s227-20260802-v2/"
                "checkpoints/step_000038912.zip"
            ).resolve(strict=True)
            checkpoint.write_bytes(locked_checkpoint.read_bytes())
        controller = {
            "kind": "snapshot",
            "path": str(checkpoint.resolve()),
            "source_run": str(actor.resolve()),
            "algorithm": "maskable_ppo",
            "step": step,
            "inference_mode": "deterministic",
        }
        identity = {
            "kind": "snapshot",
            "inference_mode": "deterministic",
            "path": str(checkpoint.resolve()),
            "algorithm": "maskable_ppo",
            "step": step,
            "contract_hash": "c" * 64,
            "contract_version": "tactical-v2",
            "environment": "tactical-v2",
            "encoding_hash": "e" * 64,
            "contract": {"version": "tactical-v2"},
            "observation_size": 2 if physical_overlays else 1292,
            "action_size": 7 if physical_overlays else 1288,
            "legacy": False,
            "promotable": True,
        }
        source_publication = {
            "kind": "audited-baseline" if iteration == 0 else "dagger-iteration",
            "iteration": iteration,
            "content_identity": hashlib.sha256(
                f"source-{iteration}".encode("ascii")
            ).hexdigest(),
            "preflight_root": str((tmp_path / "task10-preflight").resolve()),
            "preflight_content_identity": hashlib.sha256(b"preflight").hexdigest(),
            "incoming_source_content_identity": (
                None if iteration == 0 else hashlib.sha256(
                    f"source-{iteration - 1}".encode("ascii")
                ).hexdigest()
            ),
            "source_run": controller["source_run"],
            "model_seed": 227,
            "step": step,
            "controller": json.dumps(controller, sort_keys=True),
            "controller_identity": identity,
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "actor_sha256": (
                _sha256(checkpoint)
                if locked_baseline and iteration == 0
                else hashlib.sha256(f"actor-{iteration}".encode()).hexdigest()
            ),
            "publication_metadata_sha256": hashlib.sha256(
                f"metadata-{iteration}".encode()
            ).hexdigest(),
            "run_manifest_sha256": hashlib.sha256(
                f"run-{iteration}".encode()
            ).hexdigest(),
            "bc_manifest_sha256": hashlib.sha256(
                f"bc-{iteration}".encode()
            ).hexdigest(),
            "train_overlay_prefix": (
                train_overlay_ids[:iteration]
                if physical_overlays
                else [
                    hashlib.sha256(f"train-{index}".encode()).hexdigest()
                    for index in range(1, iteration + 1)
                ]
            ),
            "validation_overlay_prefix": (
                validation_overlay_ids[:iteration]
                if physical_overlays
                else [
                    hashlib.sha256(f"validation-{index}".encode()).hexdigest()
                    for index in range(1, iteration + 1)
                ]
            ),
        }
        candidates.append(dagger_module.DevelopmentCandidate.from_dict({
            "candidate_id": (
                "baseline" if iteration == 0 else f"iteration-{iteration}"
            ),
            "iteration": iteration,
            "controller": json.dumps(controller, sort_keys=True),
            "checkpoint_path": str(checkpoint.resolve()),
            "checkpoint_sha256": _sha256(checkpoint),
            "controller_identity": identity,
            "source_publication": source_publication,
        }))
    return dagger_module.DevelopmentEvaluationDefinition.create(
        candidates=candidates,
        panel_hash="1" * 64,
        scenario_hash="2" * 64,
        contract_hash="c" * 64,
        encoding_hash="e" * 64,
        repository={
            "root": str(ROOT.resolve()),
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "dirty": False,
        },
    )


def _task10_rows() -> tuple[dict[str, Any], ...]:
    return tuple({
        "seed": 20_000_000 + index // 2,
        "candidate_seat": index % 2,
        "winner": index % 2,
        "outcome": "win",
        "start_profile": "standard-3v3",
        "reference_seat": index % 2,
        "terminated": True,
        "truncated": False,
        "summary": {
            "command_count": 1,
            "round_count": 1,
            "damage_by_seat": [0, 0],
            "kills_by_seat": [0, 0],
            "end_turns_by_seat": [1, 1],
            "wasted_end_turns_by_seat": [0, 0],
            "peak_normalized_advantage": 1.0,
            "final_normalized_advantage": 1.0,
            "maximum_state_repetition": 1,
        },
        "classification": None,
        "trace_path": f"evidence/traces/match-{index:06d}.json",
        "replay_path": f"evidence/replays/match-{index:06d}.replay",
    } for index in range(200))


def _task10_source_evidence(
    runner: object,
    *,
    definition: object,
    root: Path,
    preflight_root: Path,
    preflight_identity: str,
    iteration: int,
) -> object:
    candidate = definition.candidates[iteration]
    root.mkdir()
    controller_spec = json.loads(candidate.controller)
    values = {
        "root": root.resolve(),
        "kind": "audited-baseline" if iteration == 0 else "dagger-iteration",
        "iteration": iteration,
        "content_identity": hashlib.sha256(
            f"source-{iteration}".encode("ascii")
        ).hexdigest(),
        "preflight_root": preflight_root.resolve(),
        "preflight_content_identity": preflight_identity,
        "incoming_source_content_identity": (
            None
            if iteration == 0
            else hashlib.sha256(
                f"source-{iteration - 1}".encode("ascii")
            ).hexdigest()
        ),
        "source_run": controller_spec["source_run"],
        "model_seed": 227,
        "step": 38_912 if iteration == 0 else 0,
        "controller": candidate.controller,
        "controller_identity": candidate.controller_identity,
        "checkpoint_path": candidate.checkpoint_path,
        "checkpoint_sha256": candidate.checkpoint_sha256,
        "actor_sha256": hashlib.sha256(
            f"actor-{iteration}".encode("ascii")
        ).hexdigest(),
        "publication_metadata_sha256": hashlib.sha256(
            f"metadata-{iteration}".encode("ascii")
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(
            f"run-{iteration}".encode("ascii")
        ).hexdigest(),
        "bc_manifest_sha256": hashlib.sha256(
            f"bc-{iteration}".encode("ascii")
        ).hexdigest(),
        "train_overlay_prefix": tuple(
            hashlib.sha256(f"train-{index}".encode("ascii")).hexdigest()
            for index in range(1, iteration + 1)
        ),
        "validation_overlay_prefix": tuple(
            hashlib.sha256(f"validation-{index}".encode("ascii")).hexdigest()
            for index in range(1, iteration + 1)
        ),
    }
    publication_identity = {
        key: str(value) if key == "preflight_root" else value
        for key, value in values.items()
        if key != "root"
    }
    return runner.DevelopmentSourcePublicationEvidence(
        **values,
        publication_identity=publication_identity,
    )


def _task10_physical_source_chain(
    tmp_path: Path,
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Create a minimal real baseline + Task 9/Task 7 k1-k3 byte chain."""

    import importlib.util

    fixture_path = Path(__file__).with_name("test_dagger.py")
    spec = importlib.util.spec_from_file_location(
        "_task10_source_overlay_fixture", fixture_path,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("physical overlay fixture module is unavailable")
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    seal_pair = fixture_module._seal_pair

    baseline_run = Path(
        "C:/Users/cddal/HexWars/python/runs/"
        "bc227-ppo-random-s227-20260802-v2"
    ).resolve(strict=True)
    baseline_checkpoint = (
        baseline_run / "checkpoints" / "step_000038912.zip"
    ).resolve(strict=True)
    baseline_contract = runner._actor_publication_contract(baseline_run)
    baseline_controller_payload = {
        "kind": "snapshot",
        "path": str(baseline_checkpoint),
        "source_run": str(baseline_run),
        "algorithm": "maskable_ppo",
        "step": 38_912,
        "inference_mode": "deterministic",
    }
    baseline_identity = {
        "kind": "snapshot",
        "inference_mode": "deterministic",
        "path": str(baseline_checkpoint),
        "algorithm": "maskable_ppo",
        "step": 38_912,
        "contract_hash": baseline_contract.contract_hash,
        "contract_version": baseline_contract.version,
        "environment": baseline_contract.environment,
        "encoding_hash": baseline_contract.encoding_hash,
        "contract": baseline_contract.to_dict(),
        "observation_size": baseline_contract.observation_size,
        "action_size": baseline_contract.action_size,
        "legacy": False,
        "promotable": True,
    }
    oracle = dagger_module.OracleSpec(
        oracle_type="bounded-search",
        depth=4,
        expansion_budget=512,
        use_heuristic=True,
        heuristic_identity="material-plus-pursuit-v1",
        code_hash="a" * 64,
    )
    preflight_root = tmp_path / "sealed-source-preflight"
    preflight_root.mkdir()
    preflight_artifact = preflight_root / "oracle-preflight.json"
    preflight_artifact.write_bytes(b"sealed-source-preflight")
    preflight_identity = _sha256(preflight_artifact)

    baseline_root = baseline_run
    baseline_physical = (
        runner.checkpoint_audit_domain.validate_audited_baseline_publication(
            baseline_run,
            expected_checkpoint_sha256=_sha256(baseline_checkpoint),
        )
    )
    preflight = runner.DevelopmentPreflightEvidence(
        evidence_root=preflight_root.resolve(),
        content_identity=preflight_identity,
        selected_oracle=oracle,
        evidence_class="sealed-engine",
        starting_learner_checkpoint_path=str(baseline_checkpoint),
        starting_learner_checkpoint_sha256=_sha256(baseline_checkpoint),
        starting_learner_controller=json.dumps(
            baseline_controller_payload, sort_keys=True,
        ),
        starting_learner_controller_identity=runner._freeze_json(baseline_identity),
        starting_learner_model_seed=227,
        starting_learner_step=38_912,
        starting_learner_source_content_identity=baseline_physical.content_identity,
    )

    def open_test_preflight(root: Path) -> object:
        canonical = Path(root).resolve(strict=True)
        if (
            canonical != preflight_root.resolve()
            or _sha256(preflight_artifact) != preflight_identity
        ):
            raise ValueError("test preflight physical identity changed")
        return preflight

    monkeypatch.setattr(
        runner, "_open_development_preflight_evidence", open_test_preflight,
    )
    baseline = runner._open_development_source_publication_claim(
        baseline_root, preflight=preflight,
    )

    compact_contract = EnvironmentContract(
        version="tactical-v2",
        contract_hash=baseline_contract.contract_hash,
        encoding_hash=baseline_contract.encoding_hash,
        observation_size=2,
        action_size=7,
        board={"width": 2, "height": 1},
        roster=["one"],
        reward={},
        semantics={
            "action_regions": {
                "move": {"offset": 1, "count": 2},
                "attack": {"offset": 3, "count": 2},
                "deploy": {"offset": 5, "count": 2},
            },
            "start_profiles": [
                {"id": "standard-3v3"},
                {"id": "conversion-3v1-near"},
                {"id": "conversion-3v1-far"},
                {"id": "conversion-2v1-near"},
                {"id": "conversion-2v1-far"},
                {"id": "conversion-1v1-near"},
                {"id": "conversion-1v1-far"},
            ],
        },
    )
    pipeline_root = tmp_path / "physical-task9"
    iterations_root = pipeline_root / "iterations"
    iterations_root.mkdir(parents=True)
    harness = _PhysicalIterationHarness(pipeline_root, runner)
    harness.contract = compact_contract.to_dict()
    harness.repository = _repository_provider(ROOT)
    harness.oracle = {
        "spec": oracle.to_dict(),
        "evidence_root": str(preflight_root.resolve()),
        "evidence_content_identity": preflight_identity,
        "evidence_class": "sealed-engine",
    }
    repository_hash = hashlib.sha256(json.dumps(
        harness.repository,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()
    original_dataset = dagger_module.OriginalDatasetIdentity.from_dict(
        fixture_module._dataset_payload()
    )

    def validate_fake_actor(
        run_dir: Path, expected_contract: EnvironmentContract,
    ) -> dict[str, Any]:
        actor_root = Path(run_dir).resolve(strict=True)
        run = json.loads(
            (actor_root / "run.json").read_text(encoding="utf-8")
        )
        verification = dict(run["publication_verification"])
        checkpoint = actor_root / "checkpoints" / "step_000000000.zip"
        physical = {
            "checkpoint_sha256": _sha256(checkpoint),
            "actor_sha256": verification["actor_sha256"],
            "publication_metadata_sha256": _sha256(
                actor_root / "publication.json"
            ),
            "contract_hash": expected_contract.contract_hash,
            "encoding_hash": expected_contract.encoding_hash,
            "observation_size": expected_contract.observation_size,
            "action_size": expected_contract.action_size,
        }
        if any(verification.get(key) != value for key, value in physical.items()):
            raise ValueError("fake Task 7 actor physical bytes changed")
        return verification

    monkeypatch.setattr(
        imitation_module,
        "validate_actor_supervision_publication",
        validate_fake_actor,
    )

    sources = [baseline]
    iteration_roots = []
    cumulative_train: list[dict[str, Any]] = []
    cumulative_validation: list[dict[str, Any]] = []
    for index in (1, 2, 3):
        root = iterations_root / f"iteration-{index}"
        root.mkdir()
        previous = sources[-1]
        previous_manifest_sha = _sha256(
            Path(previous.source_run) / "run.json"
        )
        train, _ = seal_pair(
            root / "train-overlay",
            compact_contract,
            partition="train",
            iteration=index,
            row_count=20_000 + index,
            learner_checkpoint=Path(previous.checkpoint_path),
            learner_source_run=previous.source_run,
            learner_source_manifest_sha256=previous_manifest_sha,
            oracle=oracle,
            original_dataset=original_dataset,
            scenario_hash="2" * 64,
            repository_hash=repository_hash,
            panel_hash="1" * 64,
            schedule_hash="4" * 64,
        )
        validation, _ = seal_pair(
            root / "validation-overlay",
            compact_contract,
            partition="validation",
            iteration=index,
            row_count=2_000 + index,
            learner_checkpoint=Path(previous.checkpoint_path),
            learner_source_run=previous.source_run,
            learner_source_manifest_sha256=previous_manifest_sha,
            oracle=oracle,
            original_dataset=original_dataset,
            scenario_hash="2" * 64,
            repository_hash=repository_hash,
            panel_hash="1" * 64,
            schedule_hash="4" * 64,
        )

        def overlay_payload(value: object) -> dict[str, Any]:
            return {
                "partition": value.partition,
                "iteration": index,
                "root": value.root,
                "content_identity": value.content_identity,
                "row_count": value.row_count,
                "collection_metrics": dict(
                    dagger_module.dagger_overlay_collection_metrics(value)
                ),
            }

        train_payload = overlay_payload(train)
        validation_payload = overlay_payload(validation)
        cumulative_train.append(train_payload)
        cumulative_validation.append(validation_payload)
        actor_root = root / "actor"
        harness.train(
            index=index,
            learner={
                "source": {
                    "source_kind": "snapshot" if index == 1 else "dagger_actor",
                },
            },
            corpus={
                "training": tuple(cumulative_train),
                "held_out": tuple(cumulative_validation),
            },
            output_root=actor_root,
        )
        training_event = {
            "schema_version": 1,
            "model_seed": 227,
            "device": "cuda",
            "epoch": 1,
            "max_epochs": 50,
            "batches": 1,
            "examples": 256,
            "mean_training_loss": 1.0,
            "validation_nll": 0.5,
            "top1_accuracy": 0.5,
            "top3_accuracy": 0.75,
            "top5_accuracy": 0.9,
            "best_epoch": 1,
            "best_validation_nll": 0.5,
            "epochs_without_improvement": 0,
            "patience": 5,
            "epoch_seconds": 1.0,
            "elapsed_seconds": 1.0,
            "examples_per_second": 256.0,
            "sampling_seconds": 0.1,
            "transfer_forward_seconds": 0.2,
            "optimization_seconds": 0.4,
            "validation_seconds": 0.2,
            "unclassified_seconds": 0.1,
        }
        _rewrite(actor_root / "training-history.json", {
            "schema_version": 1,
            "model_seed": 227,
            "training_device": {"requested": "cuda", "resolved": "cuda"},
            "publication_device": "cpu",
            "epochs": [training_event],
        })
        actor = harness.reopen_actor(index=index, trained=actor_root)
        incoming = (
            harness.starting
            if index == 1
            else harness.reopen_actor(
                index=index - 1,
                trained=iteration_roots[-1] / "actor",
            )["incoming"]
        )
        identity = harness.build_iteration_identity(
            index=index,
            context={"preflight": {"selected_oracle": harness.oracle}},
            incoming=incoming,
            train_overlays=tuple(cumulative_train),
            validation_overlays=tuple(cumulative_validation),
        )
        identity["definition"]["panel_sha256"] = train.definition.panel_hash
        identity["scenario"]["runtime_sha256"] = train.definition.scenario_hash
        identity["base_dataset"]["manifest_sha256"] = (
            train.definition.original_dataset.manifest_sha256
        )
        identity["schedules"]["train"]["sha256"] = (
            train.definition.schedule_hash
        )
        identity["schedules"]["validation"]["sha256"] = (
            validation.definition.schedule_hash
        )
        timings = {
            "elapsed_seconds": 6.0,
            "validation_collection_seconds": 1.0,
            "train_collection_seconds": 1.0,
            "corpus_seconds": 1.0,
            "training_seconds": 1.0,
            "publication_seconds": 1.0,
            "train_labels_per_second": float(train.row_count),
            "validation_labels_per_second": float(validation.row_count),
        }
        manifest = harness.build_iteration_manifest(
            index=index,
            identity=identity,
            train_overlay=train_payload,
            validation_overlay=validation_payload,
            actor=actor,
            timings=timings,
        )
        dagger_module.IterationManifest.from_dict(manifest)
        _rewrite(root / "manifest.json", manifest)
        iteration_roots.append(root)
        sources.append(runner._open_development_iteration_source(
            root,
            iteration=index,
            preflight=preflight,
            previous=previous,
        ))
    return SimpleNamespace(
        preflight=preflight,
        preflight_root=preflight_root,
        baseline_root=baseline_root,
        iteration_roots=tuple(iteration_roots),
        sources=tuple(sources),
        contract=baseline_contract,
    )


def test_task10_definition_builder_requires_existing_physical_roots(
    tmp_path: Path,
) -> None:
    """Compatibility callbacks cannot replace missing physical evidence roots."""

    import run_annihilation_selective_dagger as runner

    with pytest.raises(FileNotFoundError):
        runner.build_development_evaluation_definition(
            preflight_root=tmp_path / "preflight",
            baseline_root=tmp_path / "baseline",
            iteration_roots=tuple(tmp_path / f"iteration-{i}" for i in (1, 2, 3)),
            panel_hash="1" * 64,
            scenario_hash="2" * 64,
            contract_hash="c" * 64,
            encoding_hash="e" * 64,
            repository_root=ROOT,
            reopen_preflight=None,
            reopen_baseline=None,
            reopen_iteration=None,
            repository_identity_provider=_repository_provider,
        )


def test_task10_preflight_opener_rejects_self_authored_envelope(
    tmp_path: Path,
) -> None:
    """A Task 10 checksum wrapper is not authentic Task 8 evidence."""

    import run_annihilation_selective_dagger as runner

    checkpoint = Path(
        "C:/Users/cddal/HexWars/python/runs/"
        "bc227-ppo-random-s227-20260802-v2/"
        "checkpoints/step_000038912.zip"
    ).resolve(strict=True)
    root = tmp_path / "self-authored-task8-envelope"
    root.mkdir()
    artifact = root / "oracle-preflight.json"
    artifact.write_bytes(b"arbitrary bytes, not a Task 8 publication")
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "evidence_class": "sealed-engine",
        "selected_oracle": dagger_module.OracleSpec(
            oracle_type="bounded-search",
            depth=4,
            expansion_budget=512,
            use_heuristic=True,
            heuristic_identity="material-plus-pursuit-v1",
            code_hash="a" * 64,
        ).to_dict(),
        "starting_learner": {
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": _sha256(checkpoint),
            "controller": "{}",
            "controller_identity": {},
            "model_seed": 227,
            "step": 38_912,
            "source_content_identity": "0" * 64,
        },
        "artifact": {
            "path": artifact.name,
            "sha256": _sha256(artifact),
            "byte_size": artifact.stat().st_size,
        },
    }
    unsigned = json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    manifest["envelope_identity"] = hashlib.sha256(unsigned).hexdigest()
    _rewrite(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="Task 8|schema|preflight"):
        runner._open_development_preflight_evidence(root)


def test_task10_preflight_opener_rejects_real_untrusted_task8_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-2 test transcripts stay explicitly non-production until Task 11."""

    import run_annihilation_selective_dagger as runner

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, codec, clock, _schedules, _queries = (
        _preflight_boundaries(wins={512: 210, 2048: 220})
    )
    root = tmp_path / "real-task8-publication"
    run_oracle_preflight(
        definition,
        output_root=root,
        repository_identity_provider=_repository_provider,
        evaluator=evaluator,
        benchmark=benchmark,
        codec=codec,
        clock=clock,
    )
    publication = dagger_module.open_oracle_preflight_publication(
        root,
        definition=definition,
        repository_identity_provider=_repository_provider,
    )
    assert publication.evidence_class == "untrusted-test-transcript"

    monkeypatch.setattr(runner, "_git_repository_identity", _repository_provider)
    with pytest.raises(
        ValueError, match="untrusted-test-transcript.*Task 11|Task 11.*sealed-engine",
    ):
        runner._open_development_preflight_evidence(root)


def test_task10_public_definition_cannot_select_private_preflight_seam(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Public callback arguments cannot bypass the authenticated Task 8 opener."""

    import run_annihilation_selective_dagger as runner

    preflight_root = tmp_path / "task8"
    preflight_root.mkdir()
    locked_definition = object()
    public_calls: list[tuple[Path, object, object]] = []
    callback_calls = 0

    monkeypatch.setattr(
        runner.dagger_domain,
        "load_panel_definition",
        lambda panel_path, *, repository_root: locked_definition,
    )

    def open_public_task8(
        root: Path, *, definition: object, repository_identity_provider: object,
    ) -> object:
        public_calls.append((Path(root), definition, repository_identity_provider))
        return SimpleNamespace(evidence_class="untrusted-test-transcript")

    monkeypatch.setattr(
        runner.dagger_domain,
        "open_oracle_preflight_publication",
        open_public_task8,
    )

    def forbidden_public_callback(_root: Path) -> object:
        nonlocal callback_calls
        callback_calls += 1
        return SimpleNamespace(evidence_class="sealed-engine")

    with pytest.raises(ValueError, match="untrusted-test-transcript.*Task 11"):
        runner.build_development_evaluation_definition(
            preflight_root=preflight_root,
            baseline_root=tmp_path / "baseline",
            iteration_roots=tuple(
                tmp_path / f"iteration-{iteration}" for iteration in (1, 2, 3)
            ),
            panel_hash="1" * 64,
            scenario_hash="2" * 64,
            contract_hash="c" * 64,
            encoding_hash="e" * 64,
            repository_root=ROOT,
            reopen_preflight=forbidden_public_callback,
            reopen_baseline=None,
            reopen_iteration=None,
            repository_identity_provider=_repository_provider,
        )

    assert callback_calls == 0
    assert public_calls == [(
        preflight_root.resolve(),
        locked_definition,
        runner._git_repository_identity,
    )]


def test_task10_definition_builder_authenticates_real_task7_task9_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The definition is derived from strict Task 9 overlays and Task 7 actor bytes."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    definition = runner.build_development_evaluation_definition(
        preflight_root=chain.preflight_root,
        baseline_root=chain.baseline_root,
        iteration_roots=chain.iteration_roots,
        panel_hash="1" * 64,
        scenario_hash="2" * 64,
        contract_hash=chain.contract.contract_hash,
        encoding_hash=chain.contract.encoding_hash,
        repository_root=ROOT,
        reopen_preflight=lambda _root: chain.preflight,
        reopen_baseline=lambda _root: chain.sources[0],
        reopen_iteration=lambda root: chain.sources[
            chain.iteration_roots.index(root) + 1
        ],
        repository_identity_provider=_repository_provider,
    )

    assert definition.candidates[0].checkpoint_sha256 == (
        "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
    )
    assert tuple(
        definition.candidates[3].source_publication[
            "validation_overlay_prefix"
        ]
    ) == tuple(
        chain.sources[index].validation_overlay_prefix[-1]
        for index in (1, 2, 3)
    )


@pytest.mark.parametrize(
    "mutation", ("copied-metrics", "overlay-definition", "actor-extra-file"),
)
def test_task10_iteration_source_rejects_unphysical_task7_task9_claims(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Task 9 summaries and Task 7 inventory must come from reopened bytes."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    root = chain.iteration_roots[0]
    manifest_path = root / "manifest.json"
    if mutation == "actor-extra-file":
        (root / "actor" / "unowned.bin").write_bytes(b"unowned")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "copied-metrics":
            physical = dagger_module.dagger_overlay_collection_metrics(
                dagger_module.open_dagger_overlay(root / "train-overlay")
            )
            manifest["metrics"]["train_collection"]["mean_expansions"] = (
                physical["mean_expansions"] / 2.0
            )
        else:
            manifest["identity"]["scenario"]["runtime_sha256"] = "f" * 64
        manifest["content_identity"] = _content_identity(manifest)
        dagger_module.IterationManifest.from_dict(manifest)
        _rewrite(manifest_path, manifest)

    with pytest.raises(
        ValueError,
        match="Task 7|Task 9|actor|inventory|metric|overlay|scenario|physical",
    ):
        runner._open_development_iteration_source(
            root,
            iteration=1,
            preflight=chain.preflight,
            previous=chain.sources[0],
        )


def test_task10_iteration_source_rejects_resealed_predecessor_manifest_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlay and Task 9 reseals cannot replace the physical predecessor hash."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    previous = chain.sources[1]
    previous_run = Path(previous.source_run) / "run.json"
    physical_predecessor_manifest_sha256 = hashlib.sha256(
        previous_run.read_bytes()
    ).hexdigest()
    forged_manifest_sha256 = "f" * 64
    assert physical_predecessor_manifest_sha256 != forged_manifest_sha256

    root = chain.iteration_roots[1]
    overlay_identities: dict[str, str] = {}
    for partition in ("train", "validation"):
        overlay_root = root / f"{partition}-overlay"
        overlay_path = overlay_root / "manifest.json"
        overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
        overlay["learner"]["source_manifest_sha256"] = forged_manifest_sha256
        for descriptor in overlay["games"]:
            game_path = overlay_root / descriptor["path"]
            game = json.loads(game_path.read_text(encoding="utf-8"))
            game["learner"]["source_manifest_sha256"] = forged_manifest_sha256
            game["content_identity"] = _content_identity(game)
            _rewrite(game_path, game)
            descriptor.update({
                "sha256": _sha256(game_path),
                "byte_size": game_path.stat().st_size,
                "content_identity": game["content_identity"],
            })
        overlay["content_identity"] = _content_identity(overlay)
        _rewrite(overlay_path, overlay)
        overlay_identities[partition] = overlay["content_identity"]

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["incoming_learner"]["identity"][
        "source_manifest_sha256"
    ] = forged_manifest_sha256
    for partition in ("train", "validation"):
        identity_key = f"cumulative_{partition}_overlays"
        artifact_key = f"{partition}_overlay"
        manifest["identity"][identity_key][-1]["content_identity"] = (
            overlay_identities[partition]
        )
        manifest["artifacts"][artifact_key]["content_identity"] = (
            overlay_identities[partition]
        )
    manifest["content_identity"] = _content_identity(manifest)
    dagger_module.IterationManifest.from_dict(manifest)
    _rewrite(manifest_path, manifest)

    with pytest.raises(ValueError, match="predecessor|source|manifest|learner"):
        runner._open_development_iteration_source(
            root,
            iteration=2,
            preflight=chain.preflight,
            previous=previous,
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "actor-bytes", "overlay-shard-bytes", "task9-manifest-bytes",
        "preflight-artifact-bytes",
    ),
)
def test_task10_definition_builder_rejects_deep_physical_source_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Unchanged callback DTOs cannot conceal any Task 7/8/9 byte mutation."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    iteration_root = chain.iteration_roots[1]
    if mutation == "actor-bytes":
        (iteration_root / "actor" / "actor.bin").write_bytes(b"forged-actor")
    elif mutation == "overlay-shard-bytes":
        overlay_root = iteration_root / "validation-overlay"
        manifest = json.loads(
            (overlay_root / "manifest.json").read_text(encoding="utf-8")
        )
        game = json.loads(
            (overlay_root / manifest["games"][0]["path"]).read_text(
                encoding="utf-8"
            )
        )
        (overlay_root / game["shard"]["path"]).write_bytes(b"forged-shard")
    elif mutation == "task9-manifest-bytes":
        (iteration_root / "manifest.json").write_bytes(b"forged-task9-manifest")
    else:
        (chain.preflight_root / "oracle-preflight.json").write_bytes(
            b"forged-task8-preflight"
        )

    with pytest.raises(
        ValueError,
        match="Task 7|Task 8|Task 9|actor|overlay|manifest|preflight|physical|bytes",
    ):
        runner.build_development_evaluation_definition(
            preflight_root=chain.preflight_root,
            baseline_root=chain.baseline_root,
            iteration_roots=chain.iteration_roots,
            panel_hash="1" * 64,
            scenario_hash="2" * 64,
            contract_hash=chain.contract.contract_hash,
            encoding_hash=chain.contract.encoding_hash,
            repository_root=ROOT,
            reopen_preflight=lambda _root: chain.preflight,
            reopen_baseline=lambda _root: chain.sources[0],
            reopen_iteration=lambda root: chain.sources[
                chain.iteration_roots.index(root) + 1
            ],
            repository_identity_provider=_repository_provider,
        )


def test_task10_iteration_evidence_rejects_manifest_change_during_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate iteration evidence cannot mix two Task 9 manifest generations."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    definition = runner.build_development_evaluation_definition(
        preflight_root=chain.preflight_root,
        baseline_root=chain.baseline_root,
        iteration_roots=chain.iteration_roots,
        panel_hash="1" * 64,
        scenario_hash="2" * 64,
        contract_hash=chain.contract.contract_hash,
        encoding_hash=chain.contract.encoding_hash,
        repository_root=ROOT,
        reopen_preflight=None,
        reopen_baseline=None,
        reopen_iteration=None,
        repository_identity_provider=_repository_provider,
    )
    original = runner.imitation_domain._read_training_history_identity
    manifest_path = chain.iteration_roots[0] / "manifest.json"

    def mutate_manifest(actor_root: Path) -> object:
        value = original(actor_root)
        manifest_path.write_bytes(b"changed-during-iteration-reopen")
        return value

    monkeypatch.setattr(
        runner.imitation_domain,
        "_read_training_history_identity",
        mutate_manifest,
    )

    with pytest.raises(ValueError, match="Task 9|manifest|changed|physical"):
        runner._open_development_iteration_evidence_from_physical_bytes(
            chain.iteration_roots[0],
            iteration=1,
            definition=definition,
            preflight=chain.preflight,
            previous=chain.sources[0],
        )


class _Task10PhysicalBoundary:
    def __init__(self) -> None:
        self.evaluate_calls: list[dict[str, Any]] = []
        self.validate_calls: list[dict[str, Any]] = []

    def _open(self, publication_root: Path) -> object:
        import ml_lab.checkpoint_audit as audit_domain

        root = Path(publication_root)
        rows = _task10_rows()
        artifacts = []
        for row in rows:
            trace_path = root / row["trace_path"]
            replay_path = root / row["replay_path"]
            if not trace_path.is_file() or not replay_path.is_file():
                raise ValueError("retained Task 10 artifact is missing")
            artifacts.append(audit_domain.RetainedArtifactIdentity(
                trace_path=row["trace_path"],
                trace_sha256=_sha256(trace_path),
                trace_byte_size=trace_path.stat().st_size,
                replay_path=row["replay_path"],
                replay_sha256=_sha256(replay_path),
                replay_byte_size=replay_path.stat().st_size,
            ))
        evaluation_path = root / "evaluation.json"
        if not evaluation_path.is_file():
            raise ValueError("retained Task 10 evaluation is missing")
        return audit_domain.RetainedEvaluation(
            evaluation=json.loads(evaluation_path.read_text(encoding="utf-8")),
            matches=rows,
            artifacts=tuple(artifacts),
        )

    def evaluate(self, controller: str, **kwargs: Any) -> object:
        self.evaluate_calls.append({"controller": controller, **kwargs})
        root = Path(kwargs["publication_root"])
        (root / "evidence" / "traces").mkdir(parents=True)
        (root / "evidence" / "replays").mkdir(parents=True)
        for index, row in enumerate(_task10_rows()):
            (root / row["trace_path"]).write_bytes(
                f"trace-{index}".encode("ascii")
            )
            (root / row["replay_path"]).write_bytes(
                f"replay-{index}".encode("ascii")
            )
        (root / "evaluation.json").write_text(
            json.dumps({"schema_version": 1}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self._open(root)

    def validate(self, evaluation_path: Path, **kwargs: Any) -> object:
        self.validate_calls.append({
            "evaluation_path": Path(evaluation_path),
            **kwargs,
        })
        return self._open(Path(kwargs["publication_root"]))


def test_task10_candidate_evaluation_is_transactional_and_exact_reuse_launches_zero_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A completed candidate reopens every byte and never calls the game adapter."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    candidate = definition.candidates[0]
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    evaluations_root = tmp_path / "development-evaluations"

    first = runner.run_development_candidate_evaluation(
        definition=definition,
        candidate=candidate,
        output_root=evaluations_root,
        server_cmd=("fake-gym-server",),
        workers=3,
        evaluate_candidate=boundary.evaluate,
        validate_candidate=boundary.validate,
        repository_identity_provider=_repository_provider,
    )
    assert first.new_games == 200
    assert first.reused is False
    assert first.result.root == (evaluations_root / "baseline").resolve()
    assert len(boundary.evaluate_calls) == 1
    assert boundary.evaluate_calls[0]["schedule"].seed_start == 20_000_000
    assert boundary.evaluate_calls[0]["schedule"].maps == 100

    second = runner.run_development_candidate_evaluation(
        definition=definition,
        candidate=candidate,
        output_root=evaluations_root,
        server_cmd=("fake-gym-server",),
        workers=3,
        evaluate_candidate=boundary.evaluate,
        validate_candidate=boundary.validate,
        repository_identity_provider=_repository_provider,
    )
    assert second.new_games == 0
    assert second.reused is True
    assert second.result.content_identity == first.result.content_identity
    assert len(boundary.evaluate_calls) == 1

    trace = evaluations_root / "baseline" / "physical" / (
        "evidence/traces/match-000000.json"
    )
    trace.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact|hash|physical|reusable"):
        runner.run_development_candidate_evaluation(
            definition=definition,
            candidate=candidate,
            output_root=evaluations_root,
            server_cmd=("fake-gym-server",),
            workers=3,
            evaluate_candidate=boundary.evaluate,
            validate_candidate=boundary.validate,
            repository_identity_provider=_repository_provider,
        )
    assert len(boundary.evaluate_calls) == 1


def test_task10_candidate_manifest_ignores_callback_authored_validator_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate identity comes from the first-party physical validator only."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    candidate = definition.candidates[0]
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )

    result = runner.run_development_candidate_evaluation(
        definition=definition,
        candidate=candidate,
        output_root=tmp_path / "first-party-candidate-validation",
        server_cmd=("fake-gym-server",),
        workers=1,
        evaluate_candidate=boundary.evaluate,
        validate_candidate=lambda *_args, **_kwargs: pytest.fail(
            "callback-authored retained evaluation was trusted"
        ),
        repository_identity_provider=_repository_provider,
    )

    assert result.new_games == 200
    assert len(boundary.validate_calls) == 3


def test_task10_candidate_final_guard_rereads_trace_and_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A trace changed during the final evaluation reread cannot escape detection."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    candidate = definition.candidates[0]
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    result = runner.run_development_candidate_evaluation(
        definition=definition,
        candidate=candidate,
        output_root=tmp_path / "candidate-final-reread",
        server_cmd=("fake",),
        workers=1,
        evaluate_candidate=boundary.evaluate,
        validate_candidate=boundary.validate,
        repository_identity_provider=_repository_provider,
    )
    evaluation_path = result.result.root / "physical" / "evaluation.json"
    trace_path = result.result.root / "physical" / _task10_rows()[0]["trace_path"]
    original_read_bytes = Path.read_bytes
    evaluation_reads = 0

    def mutate_on_final_evaluation_read(path: Path) -> bytes:
        nonlocal evaluation_reads
        raw = original_read_bytes(path)
        if path == evaluation_path:
            evaluation_reads += 1
            if evaluation_reads == 2:
                trace_path.write_bytes(b"changed-at-final-guard")
        return raw

    monkeypatch.setattr(Path, "read_bytes", mutate_on_final_evaluation_read)
    with pytest.raises(ValueError, match="trace|replay|artifact|changed"):
        runner.reopen_development_candidate_evaluation(
            result.result.root,
            definition=definition,
            candidate=candidate,
        )


def test_task10_candidate_publication_rejects_repository_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A repository change after games leaves diagnostic staging and no result."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    calls = 0

    def drifting_repository(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls >= 2:
            identity["source_tree"] = "f" * 40
        return identity

    output_root = tmp_path / "repository-drift-evaluations"
    with pytest.raises(ValueError, match="repository|identity|changed"):
        runner.run_development_candidate_evaluation(
            definition=definition,
            candidate=definition.candidates[0],
            output_root=output_root,
            server_cmd=("fake-gym-server",),
            workers=2,
            evaluate_candidate=boundary.evaluate,
            validate_candidate=boundary.validate,
            repository_identity_provider=drifting_repository,
        )

    assert len(boundary.evaluate_calls) == 1
    assert not (output_root / "baseline").exists()
    assert (output_root / "baseline.staging").is_dir()


@pytest.mark.parametrize(
    ("phase", "validate_call"),
    (
        ("after-evaluator", 0),
        ("after-validator", 1),
        ("staged-reopen", 2),
        ("published-reopen", 3),
    ),
)
def test_task10_candidate_checkpoint_drift_at_every_late_boundary_never_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    validate_call: int,
) -> None:
    """Checkpoint identity is probed through staged and published reopen boundaries."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    candidate = definition.candidates[0]
    checkpoint = Path(candidate.checkpoint_path)
    boundary = _Task10PhysicalBoundary()
    validation_calls = 0

    def mutate_checkpoint() -> None:
        checkpoint.write_bytes(f"drift-{phase}".encode("ascii"))

    def evaluate(*args: Any, **kwargs: Any) -> object:
        result = boundary.evaluate(*args, **kwargs)
        if phase == "after-evaluator":
            mutate_checkpoint()
        return result

    def validate(*args: Any, **kwargs: Any) -> object:
        nonlocal validation_calls
        validation_calls += 1
        result = boundary.validate(*args, **kwargs)
        if validation_calls == validate_call:
            mutate_checkpoint()
        return result

    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        validate,
    )

    output_root = tmp_path / f"checkpoint-drift-{phase}"
    with pytest.raises(ValueError, match="checkpoint|identity|changed|drift"):
        runner.run_development_candidate_evaluation(
            definition=definition,
            candidate=candidate,
            output_root=output_root,
            server_cmd=("fake-gym-server",),
            workers=2,
            evaluate_candidate=evaluate,
            validate_candidate=validate,
            repository_identity_provider=_repository_provider,
        )
    assert not (output_root / "baseline").exists()
    assert len(boundary.evaluate_calls) == 1


def test_task10_candidate_reuse_rejects_checkpoint_drift_after_reopen_with_zero_games(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact reuse probes the checkpoint after reopening every retained artifact."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    candidate = definition.candidates[0]
    checkpoint = Path(candidate.checkpoint_path)
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    output_root = tmp_path / "reuse-checkpoint-drift"
    runner.run_development_candidate_evaluation(
        definition=definition,
        candidate=candidate,
        output_root=output_root,
        server_cmd=("fake-gym-server",),
        workers=2,
        evaluate_candidate=boundary.evaluate,
        validate_candidate=boundary.validate,
        repository_identity_provider=_repository_provider,
    )
    original_validate = boundary.validate

    def mutate_after_reopen(*args: Any, **kwargs: Any) -> object:
        result = original_validate(*args, **kwargs)
        checkpoint.write_bytes(b"reuse-checkpoint-drift")
        return result

    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        mutate_after_reopen,
    )

    with pytest.raises(ValueError, match="checkpoint|identity|changed|drift"):
        runner.run_development_candidate_evaluation(
            definition=definition,
            candidate=candidate,
            output_root=output_root,
            server_cmd=("fake-gym-server",),
            workers=2,
            evaluate_candidate=boundary.evaluate,
            validate_candidate=mutate_after_reopen,
            repository_identity_provider=_repository_provider,
        )
    assert len(boundary.evaluate_calls) == 1


def test_task10_panel_evaluation_dispatches_baseline_then_three_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Panel orchestration owns exactly four candidate directories in locked order."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path)
    calls: list[tuple[str, Path]] = []

    def candidate_stage(*, candidate: object, output_root: Path, **kwargs: Any) -> object:
        calls.append((candidate.candidate_id, Path(output_root)))
        return candidate.candidate_id

    monkeypatch.setattr(
        runner, "run_development_candidate_evaluation", candidate_stage,
    )
    results = runner.run_development_evaluation(
        definition=definition,
        output_root=tmp_path / "panel",
        server_cmd=("fake-gym-server",),
        workers=2,
        repository_identity_provider=_repository_provider,
    )

    assert results == (
        "baseline", "iteration-1", "iteration-2", "iteration-3",
    )
    assert calls == [
        (candidate_id, tmp_path / "panel")
        for candidate_id in (
            "baseline", "iteration-1", "iteration-2", "iteration-3",
        )
    ]


def _task10_heldout_overlays(
    runner: object,
    *,
    definition: object,
    tmp_path: Path,
    iteration: int,
) -> tuple[object, ...]:
    prefix = definition.candidates[iteration].source_publication[
        "validation_overlay_prefix"
    ]
    overlays = []
    for overlay_index, content_identity in enumerate(prefix, start=1):
        root = tmp_path / f"heldout-overlay-{overlay_index}"
        physical = dagger_module.open_dagger_overlay(root)
        examples = dagger_module.dagger_overlay_supervised_examples(
            physical
        )
        assert physical.content_identity == content_identity
        overlays.append(runner.DevelopmentHeldoutOverlayEvidence(
            root=root.resolve(),
            content_identity=content_identity,
            examples=examples,
        ))
    return tuple(overlays)


def test_task10_supervised_evaluation_is_transactional_and_reuse_runs_zero_inference(
    tmp_path: Path,
) -> None:
    """Task 10 owns ordered pre/post inference and exact physical reuse."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=2,
    )
    by_root = {item.root: item for item in overlays}
    prediction_calls: list[str] = []

    def predict_actions(*, controller: str, examples: tuple[object, ...], **_kwargs: Any) -> object:
        prediction_calls.append(controller)
        trained = controller == definition.candidates[2].controller
        records = []
        for index, example in enumerate(examples):
            action = example["oracle_action"]
            if (trained and index == len(examples) - 1) or (not trained and index % 2):
                action += 1
            records.append({"sample_id": example["sample_id"], "action": action})
        return tuple(records)

    output_root = tmp_path / "supervised-evaluations"
    first = runner.run_development_supervised_evaluation(
        definition=definition,
        iteration=2,
        heldout_overlay_roots=tuple(item.root for item in overlays),
        output_root=output_root,
        reopen_heldout_overlay=lambda root: by_root[Path(root)],
        predict_actions=predict_actions,
        repository_identity_provider=_repository_provider,
    )
    assert first.new_inferences == 8
    assert first.reused is False
    assert len(prediction_calls) == 2
    assert first.result.metrics["labels"] == 4
    assert first.result.metrics["pre"]["accuracy"] == 0.5
    assert first.result.metrics["post"]["accuracy"] == 0.75
    assert set(first.result.metrics["post"]["by_reason"]) == {
        "conversion", "favorable", "cycle_warning", "wasted_end_turn",
    }
    assert {path.name for path in first.result.root.iterdir()} == {
        "evidence.json", "predictions.json", "metrics.json", "manifest.json",
    }

    second = runner.run_development_supervised_evaluation(
        definition=definition,
        iteration=2,
        heldout_overlay_roots=tuple(item.root for item in overlays),
        output_root=output_root,
        reopen_heldout_overlay=lambda root: by_root[Path(root)],
        predict_actions=predict_actions,
        repository_identity_provider=_repository_provider,
    )
    assert second.reused is True
    assert second.new_inferences == 0
    assert len(prediction_calls) == 2
    assert second.result.content_identity == first.result.content_identity

    (first.result.root / "predictions.json").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="prediction|hash|reusable|artifact"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=2,
            heldout_overlay_roots=tuple(item.root for item in overlays),
            output_root=output_root,
            reopen_heldout_overlay=lambda root: by_root[Path(root)],
            predict_actions=predict_actions,
            repository_identity_provider=_repository_provider,
        )
    assert len(prediction_calls) == 2


def test_task10_supervised_ignores_callback_label_forgery_and_uses_physical_rows(
    tmp_path: Path,
) -> None:
    """A callback cannot replace oracle labels or reason flags with in-memory claims."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )
    forged = replace(
        overlays[0],
        examples=(
            {
                **dict(overlays[0].examples[0]),
                "oracle_action": overlays[0].examples[0]["oracle_action"] + 1,
                "reasons": ("wasted_end_turn",),
            },
            *overlays[0].examples[1:],
        ),
    )
    callback_calls = 0
    prediction_calls = 0

    def reopen_forgery(_root: Path) -> object:
        nonlocal callback_calls
        callback_calls += 1
        return forged

    def predictor(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        nonlocal prediction_calls
        prediction_calls += 1
        assert all(item["oracle_action"] == 3 for item in examples)
        assert all(
            item["reasons"]
            == ("conversion", "favorable", "cycle_warning", "wasted_end_turn")
            for item in examples
        )
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in examples
        )

    result = runner.run_development_supervised_evaluation(
        definition=definition,
        iteration=1,
        heldout_overlay_roots=(overlays[0].root,),
        output_root=tmp_path / "forged-supervised",
        reopen_heldout_overlay=reopen_forgery,
        predict_actions=predictor,
        repository_identity_provider=_repository_provider,
    )
    assert result.result.metrics["labels"] == 2
    assert callback_calls == 0
    assert prediction_calls == 2


@pytest.mark.parametrize("mutation", ("end-of-open", "reparse-file"))
def test_task10_supervised_rejects_local_toctou_and_reparse_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Every local supervised artifact remains contained and stable to return."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )

    def predict(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in examples
        )

    result = runner.run_development_supervised_evaluation(
        definition=definition,
        iteration=1,
        heldout_overlay_roots=tuple(item.root for item in overlays),
        output_root=tmp_path / "supervised-local-guard",
        reopen_heldout_overlay=None,
        predict_actions=predict,
        repository_identity_provider=_repository_provider,
    )
    if mutation == "reparse-file":
        predictions = result.result.root / "predictions.json"
        target = tmp_path / "external-predictions.json"
        target.write_bytes(predictions.read_bytes())
        predictions.unlink()
        _symlink_or_skip_windows_privilege(predictions, target)
    else:
        original_metrics = runner._supervised_metrics

        def mutate_evidence(*args: Any, **kwargs: Any) -> object:
            metrics = original_metrics(*args, **kwargs)
            (result.result.root / "evidence.json").write_bytes(
                b"changed-at-end-of-open"
            )
            return metrics

        monkeypatch.setattr(runner, "_supervised_metrics", mutate_evidence)

    with pytest.raises(ValueError, match="supervised|reparse|contained|changed"):
        runner._open_development_supervised_evaluation_from_physical_bytes(
            result.result.root,
            definition=definition,
            iteration=1,
        )


def test_task10_supervised_rejects_overlay_shard_mutated_during_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-row-extraction shard mutation must fail the final physical reopen."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )

    def predict(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in examples
        )

    result = runner.run_development_supervised_evaluation(
        definition=definition,
        iteration=1,
        heldout_overlay_roots=(overlays[0].root,),
        output_root=tmp_path / "supervised-overlay-mid-metrics",
        reopen_heldout_overlay=None,
        predict_actions=predict,
        repository_identity_provider=_repository_provider,
    )
    overlay_manifest = json.loads(
        (overlays[0].root / "manifest.json").read_text(encoding="utf-8")
    )
    first_game = json.loads(
        (
            overlays[0].root / overlay_manifest["games"][0]["path"]
        ).read_text(encoding="utf-8")
    )
    shard = overlays[0].root / first_game["shard"]["path"]
    original_metrics = runner._supervised_metrics

    def mutate_shard(*args: Any, **kwargs: Any) -> object:
        metrics = original_metrics(*args, **kwargs)
        shard.write_bytes(b"corrupted-after-row-extraction")
        return metrics

    monkeypatch.setattr(runner, "_supervised_metrics", mutate_shard)
    with pytest.raises(ValueError, match="overlay|shard|physical|changed"):
        runner._open_development_supervised_evaluation_from_physical_bytes(
            result.result.root,
            definition=definition,
            iteration=1,
        )


def test_task10_supervised_rejects_reparse_overlay_root(
    tmp_path: Path,
) -> None:
    """The supplied overlay root's unresolved path chain must contain no reparse."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )
    alias = tmp_path / "heldout-overlay-reparse-alias"
    try:
        alias.symlink_to(overlays[0].root, target_is_directory=True)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"Windows symbolic-link privilege is unavailable: {exc}")
        raise

    with pytest.raises(ValueError, match="overlay|reparse|canonical|alias"):
        runner._validated_supervised_overlays(
            definition=definition,
            iteration=1,
            roots=(alias,),
        )


def test_task10_supervised_post_rename_repository_drift_rolls_back_to_staging(
    tmp_path: Path,
) -> None:
    """A late repository failure must leave no newly published destination."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )
    calls = 0

    def provider(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls >= 5:
            identity["source_tree"] = "f" * 40
        return identity

    def predictor(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in examples
        )

    output_root = tmp_path / "late-repository-supervised"
    with pytest.raises(ValueError, match="repository|identity|changed"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=1,
            heldout_overlay_roots=(overlays[0].root,),
            output_root=output_root,
            reopen_heldout_overlay=lambda _root: overlays[0],
            predict_actions=predictor,
            repository_identity_provider=provider,
        )
    assert calls == 5
    assert not (output_root / "iteration-1").exists()
    assert (output_root / "iteration-1.staging").is_dir()


def test_task10_supervised_post_rename_checkpoint_mutation_rolls_back_to_staging(
    tmp_path: Path,
) -> None:
    """A late checkpoint failure must leave recoverable staging, not publication."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=1,
    )
    calls = 0

    def provider(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 4:
            Path(definition.candidates[1].checkpoint_path).write_bytes(
                b"late-checkpoint-mutation"
            )
        return _repository_provider(root)

    def predictor(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in examples
        )

    output_root = tmp_path / "late-checkpoint-supervised"
    with pytest.raises(ValueError, match="checkpoint|bytes|changed"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=1,
            heldout_overlay_roots=(overlays[0].root,),
            output_root=output_root,
            reopen_heldout_overlay=lambda _root: overlays[0],
            predict_actions=predictor,
            repository_identity_provider=provider,
        )
    assert calls == 4
    assert not (output_root / "iteration-1").exists()
    assert (output_root / "iteration-1.staging").is_dir()
def test_task10_supervised_evaluation_fails_closed_on_shift_or_unordered_predictions(
    tmp_path: Path,
) -> None:
    """Cumulative corpus identity and prediction order reconcile before publication."""

    import run_annihilation_selective_dagger as runner

    definition = _task10_definition(tmp_path, physical_overlays=True)
    overlays = _task10_heldout_overlays(
        runner, definition=definition, tmp_path=tmp_path, iteration=2,
    )
    calls = 0

    def predictor(*, examples: tuple[object, ...], **_kwargs: Any) -> object:
        nonlocal calls
        calls += 1
        return tuple(
            {"sample_id": item["sample_id"], "action": item["oracle_action"]}
            for item in reversed(examples)
        )

    shifted_manifest_path = overlays[1].root / "manifest.json"
    original_manifest = shifted_manifest_path.read_bytes()
    shifted_manifest = json.loads(original_manifest.decode("utf-8"))
    shifted_manifest["content_identity"] = "f" * 64
    _rewrite(shifted_manifest_path, shifted_manifest)
    with pytest.raises(ValueError, match="overlay|prefix|identity"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=2,
            heldout_overlay_roots=tuple(item.root for item in overlays),
            output_root=tmp_path / "shifted-supervised",
            reopen_heldout_overlay=lambda _root: pytest.fail(
                "overlay DTO callback must not run"
            ),
            predict_actions=predictor,
            repository_identity_provider=_repository_provider,
        )
    assert calls == 0
    shifted_manifest_path.write_bytes(original_manifest)

    with pytest.raises(ValueError, match="ordered|sample|prediction"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=2,
            heldout_overlay_roots=tuple(item.root for item in overlays),
            output_root=tmp_path / "unordered-supervised",
            reopen_heldout_overlay=lambda root: dict(
                (item.root, item) for item in overlays
            )[Path(root)],
            predict_actions=predictor,
            repository_identity_provider=_repository_provider,
        )
    assert not (tmp_path / "unordered-supervised" / "iteration-2").exists()

    with pytest.raises(RuntimeError, match="predictor|Task 11|inference"):
        runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=2,
            heldout_overlay_roots=tuple(item.root for item in overlays),
            output_root=tmp_path / "missing-predictor",
            reopen_heldout_overlay=lambda root: dict(
                (item.root, item) for item in overlays
            )[Path(root)],
            predict_actions=None,
            repository_identity_provider=_repository_provider,
        )


def _task10_fast_aggregate_fixture(
    tmp_path: Path,
    runner: object,
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    definition = _task10_definition(
        tmp_path, physical_overlays=True, locked_baseline=True,
    )
    preflight_root = tmp_path / "task10-preflight"
    preflight_root.mkdir()
    seal = preflight_root / "oracle-preflight.json"
    seal.write_bytes(b"preflight")
    seal_sha256 = _sha256(seal)
    oracle = dagger_module.OracleSpec(
        oracle_type="bounded-search", depth=4, expansion_budget=512,
        use_heuristic=True, heuristic_identity="material-plus-pursuit-v1",
        code_hash="a" * 64,
    )
    preflight = runner.DevelopmentPreflightEvidence(
        evidence_root=preflight_root.resolve(),
        content_identity=seal_sha256,
        selected_oracle=oracle,
        evidence_class="sealed-engine",
        starting_learner_checkpoint_path=definition.candidates[0].checkpoint_path,
        starting_learner_checkpoint_sha256=definition.candidates[0].checkpoint_sha256,
        starting_learner_controller=definition.candidates[0].controller,
        starting_learner_controller_identity=definition.candidates[0].controller_identity,
        starting_learner_model_seed=227,
        starting_learner_step=38_912,
        starting_learner_source_content_identity=(
            definition.candidates[0].source_publication["content_identity"]
        ),
    )
    iteration_roots = tuple(tmp_path / f"aggregate-matrix-iteration-{i}" for i in (1, 2, 3))
    iterations = []
    for iteration, root in enumerate(iteration_roots, start=1):
        root.mkdir()
        actor_root = Path(json.loads(definition.candidates[iteration].controller)["source_run"])
        event = {
            "schema_version": 1, "model_seed": 227, "device": "cuda",
            "epoch": 1, "max_epochs": 50, "batches": 1, "examples": 256,
            "mean_training_loss": 1.0, "validation_nll": 0.5,
            "top1_accuracy": 0.5, "top3_accuracy": 0.75,
            "top5_accuracy": 0.9, "best_epoch": 1,
            "best_validation_nll": 0.5, "epochs_without_improvement": 0,
            "patience": 5, "epoch_seconds": 1.0, "elapsed_seconds": 1.0,
            "examples_per_second": 256.0, "sampling_seconds": 0.1,
            "transfer_forward_seconds": 0.2, "optimization_seconds": 0.4,
            "validation_seconds": 0.2, "unclassified_seconds": 0.1,
        }
        history = {
            "schema_version": 1, "model_seed": 227,
            "training_device": {"requested": "cuda", "resolved": "cuda"},
            "publication_device": "cpu", "epochs": [event],
        }
        history_path = actor_root / "training-history.json"
        history_path.write_text(json.dumps(history, sort_keys=True) + "\n", encoding="utf-8")
        physical_history, history_identity = imitation_module._read_training_history_identity(
            actor_root,
        )
        iterations.append(runner.DevelopmentIterationEvidence(
            root=root.resolve(), iteration=iteration,
            content_identity=hashlib.sha256(f"matrix-iteration-{iteration}".encode()).hexdigest(),
            selected_oracle=oracle, preflight_root=preflight_root.resolve(),
            preflight_content_identity=seal_sha256,
            preflight_evidence_class="sealed-engine",
            actor_checkpoint_sha256=definition.candidates[iteration].checkpoint_sha256,
            actor_controller=definition.candidates[iteration].controller,
            actor_controller_identity=definition.candidates[iteration].controller_identity,
            validation_collection={"iteration": iteration},
            collection_metrics={"iteration": iteration},
            training_metrics={
                "best_epoch": 1, "best_validation_nll": 0.5, "epochs_trained": 1,
            },
            timings={"elapsed_seconds": float(iteration)},
            training_history_root=actor_root,
            training_history=physical_history,
            training_history_identity=history_identity,
        ))

    monkeypatch.setattr(
        runner,
        "_open_development_preflight_evidence",
        lambda _root: preflight,
    )

    def open_iteration_evidence(
        root: Path,
        *,
        iteration: int,
        definition: object,
        **_kwargs: Any,
    ) -> tuple[object, object]:
        assert Path(root) == iteration_roots[iteration - 1].resolve()
        return (
            iterations[iteration - 1],
            runner._development_source_from_candidate(
                definition.candidates[iteration]
            ),
        )

    monkeypatch.setattr(
        runner,
        "_open_development_iteration_evidence_from_physical_bytes",
        open_iteration_evidence,
    )
    evaluations_root = tmp_path / "aggregate-matrix-evaluations"
    evaluations = {}
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    for candidate in definition.candidates:
        evaluations[candidate.candidate_id] = (
            runner.run_development_candidate_evaluation(
                definition=definition,
                candidate=candidate,
                output_root=evaluations_root,
                server_cmd=("dotnet", "run"),
                workers=1,
                evaluate_candidate=boundary.evaluate,
                validate_candidate=boundary.validate,
                repository_identity_provider=_repository_provider,
            ).result
        )
    supervised_root = tmp_path / "aggregate-matrix-supervised"
    supervised = {}
    for iteration in (1, 2, 3):
        overlays = _task10_heldout_overlays(
            runner, definition=definition, tmp_path=tmp_path, iteration=iteration,
        )

        def predict_actions(
            *, examples: tuple[object, ...], **_kwargs: Any,
        ) -> object:
            return tuple(
                {
                    "sample_id": item["sample_id"],
                    "action": item["oracle_action"],
                }
                for item in examples
            )

        supervised[iteration] = runner.run_development_supervised_evaluation(
            definition=definition,
            iteration=iteration,
            heldout_overlay_roots=tuple(item.root for item in overlays),
            output_root=supervised_root,
            reopen_heldout_overlay=None,
            predict_actions=predict_actions,
            repository_identity_provider=_repository_provider,
        ).result
    callback_counts: defaultdict[str, int] = defaultdict(int)

    def reopen_preflight(root: Path) -> object:
        callback_counts["preflight"] += 1
        if _sha256(seal) != seal_sha256:
            raise ValueError("preflight physical bytes changed")
        return preflight

    def reopen_iteration(root: Path) -> object:
        index = iteration_roots.index(Path(root)) + 1
        callback_counts[f"iteration-{index}"] += 1
        return iterations[index - 1]

    def reopen_evaluation(root: Path, candidate: object) -> object:
        callback_counts[f"evaluation-{candidate.candidate_id}"] += 1
        return evaluations[candidate.candidate_id]

    def reopen_supervised(root: Path, iteration: int) -> object:
        callback_counts[f"supervised-{iteration}"] += 1
        return supervised[iteration]

    baseline_source = runner._development_source_from_candidate(
        definition.candidates[0]
    )

    def open_baseline(
        root: Path, *, preflight: object,
    ) -> object:
        assert Path(root) == Path(baseline_source.source_run).resolve()
        assert preflight is not None
        return baseline_source

    monkeypatch.setattr(
        runner, "_open_development_source_publication_claim", open_baseline,
    )

    kwargs = {
        "definition": definition,
        "preflight_root": preflight_root,
        "iteration_roots": iteration_roots,
        "evaluations_root": evaluations_root,
        "supervised_roots": tuple(supervised[i].root for i in (1, 2, 3)),
        "reopen_preflight": reopen_preflight,
        "reopen_iteration": reopen_iteration,
        "reopen_evaluation": reopen_evaluation,
        "reopen_supervised": reopen_supervised,
    }
    return SimpleNamespace(
        definition=definition, kwargs=kwargs, callback_counts=callback_counts,
        seal=seal, iterations=iterations, iteration_roots=iteration_roots,
        evaluations=evaluations, evaluations_root=evaluations_root,
        supervised=supervised, boundary=boundary, baseline_source=baseline_source,
    )


_TASK10_AGGREGATE_REPOSITORY_PHASES = (
    ("after-preflight", 2, False),
    ("after-iteration-1", 3, False),
    ("after-iteration-2", 4, False),
    ("after-iteration-3", 5, False),
    ("after-evaluation-baseline", 6, False),
    ("after-evaluation-1", 7, False),
    ("after-evaluation-2", 8, False),
    ("after-evaluation-3", 9, False),
    ("after-supervised-1", 10, False),
    ("after-supervised-2", 11, False),
    ("after-supervised-3", 12, False),
    ("immediate-prepublish", 13, False),
    ("postpublished-reopen", 15, False),
    ("reuse", 14, True),
)


@pytest.mark.parametrize(
    ("phase", "drift_call", "reuse"),
    _TASK10_AGGREGATE_REPOSITORY_PHASES,
    ids=[item[0] for item in _TASK10_AGGREGATE_REPOSITORY_PHASES],
)
def test_task10_aggregate_repository_probe_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    drift_call: int,
    reuse: bool,
) -> None:
    """Repository identity is probed after every source and publication boundary."""

    import run_annihilation_selective_dagger as runner

    fixture = _task10_fast_aggregate_fixture(tmp_path, runner, monkeypatch)
    destination = tmp_path / f"aggregate-repository-{phase}"
    if reuse:
        runner.publish_development_aggregate(
            **fixture.kwargs,
            output_root=destination,
            repository_identity_provider=_repository_provider,
        )
        original_manifest = (destination / "manifest.json").read_bytes()
    calls = 0

    def provider(root: Path) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        identity = _repository_provider(root)
        if calls >= drift_call:
            identity["source_tree"] = "f" * 40
        return identity

    with pytest.raises(ValueError, match="repository|identity|changed"):
        runner.publish_development_aggregate(
            **fixture.kwargs,
            output_root=destination,
            repository_identity_provider=provider,
        )
    assert calls == drift_call
    if reuse:
        assert (destination / "manifest.json").read_bytes() == original_manifest
    else:
        assert not destination.exists()


def test_task10_aggregate_success_reopens_every_source_twice_and_probes_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_annihilation_selective_dagger as runner

    fixture = _task10_fast_aggregate_fixture(tmp_path, runner, monkeypatch)
    repository_calls = 0

    def provider(root: Path) -> dict[str, Any]:
        nonlocal repository_calls
        repository_calls += 1
        return _repository_provider(root)

    runner.publish_development_aggregate(
        **fixture.kwargs,
        output_root=tmp_path / "aggregate-matrix-success",
        repository_identity_provider=provider,
    )
    assert fixture.callback_counts == {}
    assert repository_calls == 15


def test_task10_aggregate_ignores_forged_candidate_and_supervised_dtos(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate authority is physical bytes, never callback-returned evidence."""

    import run_annihilation_selective_dagger as runner

    fixture = _task10_fast_aggregate_fixture(tmp_path, runner, monkeypatch)
    kwargs = {
        **fixture.kwargs,
        "reopen_evaluation": lambda *_args: pytest.fail(
            "candidate DTO callback must not run"
        ),
        "reopen_supervised": lambda *_args: pytest.fail(
            "supervised DTO callback must not run"
        ),
    }
    publication = runner.publish_development_aggregate(
        **kwargs,
        output_root=tmp_path / "aggregate-no-dto-authority",
        repository_identity_provider=_repository_provider,
    )
    assert publication.aggregate["evidence_identity"]["evaluations"] == tuple(
        fixture.evaluations[candidate.candidate_id].content_identity
        for candidate in fixture.definition.candidates
    )


def test_task10_aggregate_internally_authenticates_task8_task9_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Neither aggregate pass invokes preflight/iteration DTO reopen callbacks."""

    import run_annihilation_selective_dagger as runner

    chain = _task10_physical_source_chain(tmp_path, runner, monkeypatch)
    definition = runner.build_development_evaluation_definition(
        preflight_root=chain.preflight_root,
        baseline_root=chain.baseline_root,
        iteration_roots=chain.iteration_roots,
        panel_hash="1" * 64,
        scenario_hash="2" * 64,
        contract_hash=chain.contract.contract_hash,
        encoding_hash=chain.contract.encoding_hash,
        repository_root=ROOT,
        reopen_preflight=lambda *_args: pytest.fail("preflight DTO callback"),
        reopen_baseline=lambda *_args: pytest.fail("baseline DTO callback"),
        reopen_iteration=lambda *_args: pytest.fail("iteration DTO callback"),
        repository_identity_provider=_repository_provider,
    )
    evaluations_root = tmp_path / "source-auth-evaluations"
    boundary = _Task10PhysicalBoundary()
    monkeypatch.setattr(
        runner.checkpoint_audit_domain,
        "validate_retained_evaluation",
        boundary.validate,
    )
    for candidate in definition.candidates:
        runner.run_development_candidate_evaluation(
            definition=definition,
            candidate=candidate,
            output_root=evaluations_root,
            server_cmd=("dotnet", "run"),
            workers=1,
            evaluate_candidate=boundary.evaluate,
            validate_candidate=boundary.validate,
            repository_identity_provider=_repository_provider,
        )
    supervised_roots = []
    for iteration in (1, 2, 3):
        roots = tuple(
            chain.iteration_roots[index - 1] / "validation-overlay"
            for index in range(1, iteration + 1)
        )

        def predict(
            *, examples: tuple[object, ...], **_kwargs: Any,
        ) -> object:
            return tuple(
                {
                    "sample_id": item["sample_id"],
                    "action": item["oracle_action"],
                }
                for item in examples
            )

        supervised_roots.append(
            runner.run_development_supervised_evaluation(
                definition=definition,
                iteration=iteration,
                heldout_overlay_roots=roots,
                output_root=tmp_path / "source-auth-supervised",
                reopen_heldout_overlay=None,
                predict_actions=predict,
                repository_identity_provider=_repository_provider,
            ).result.root
        )
    publication = runner.publish_development_aggregate(
        definition=definition,
        preflight_root=chain.preflight_root,
        iteration_roots=chain.iteration_roots,
        evaluations_root=evaluations_root,
        supervised_roots=tuple(supervised_roots),
        output_root=tmp_path / "source-auth-aggregate",
        reopen_preflight=lambda *_args: pytest.fail("preflight DTO callback"),
        reopen_iteration=lambda *_args: pytest.fail("iteration DTO callback"),
        reopen_evaluation=lambda *_args: pytest.fail("evaluation DTO callback"),
        reopen_supervised=lambda *_args: pytest.fail("supervised DTO callback"),
        repository_identity_provider=_repository_provider,
    )
    assert publication.aggregate["evidence_identity"]["preflight"] == (
        chain.preflight.content_identity
    )
    assert publication.aggregate["evidence_identity"]["iterations"] == tuple(
        source.content_identity for source in chain.sources[1:]
    )


@pytest.mark.parametrize(
    "mutation", (
        "baseline-checkpoint-second-pass",
        "baseline-checkpoint-delete-second-pass",
        "candidate-trace-second-pass",
        "supervised-metrics-second-pass",
    ),
)
def test_task10_aggregate_internal_second_pass_rejects_physical_byte_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Both aggregate passes independently reopen their owned artifact bytes."""

    import run_annihilation_selective_dagger as runner

    fixture = _task10_fast_aggregate_fixture(tmp_path, runner, monkeypatch)
    if mutation.startswith("baseline-checkpoint"):
        original = runner._open_development_source_publication_claim
        baseline_calls = 0

        def open_baseline(root: Path, *, preflight: object) -> object:
            nonlocal baseline_calls
            baseline_calls += 1
            if baseline_calls == 3:
                checkpoint = Path(fixture.baseline_source.checkpoint_path)
                if "delete" in mutation:
                    checkpoint.unlink()
                else:
                    checkpoint.write_bytes(b"changed-between-aggregate-passes")
            return original(root, preflight=preflight)

        monkeypatch.setattr(
            runner, "_open_development_source_publication_claim", open_baseline,
        )
    elif mutation == "candidate-trace-second-pass":
        calls: defaultdict[str, int] = defaultdict(int)
        target = fixture.definition.candidates[2]

        def validate(evaluation_path: Path, **kwargs: Any) -> object:
            key = str(kwargs["expected_candidate_identity"]["path"])
            calls[key] += 1
            if key == target.checkpoint_path and calls[key] == 2:
                trace = (
                    Path(kwargs["publication_root"])
                    / _task10_rows()[0]["trace_path"]
                )
                trace.write_bytes(b"changed-between-aggregate-passes")
            return fixture.boundary.validate(evaluation_path, **kwargs)

        monkeypatch.setattr(
            runner.checkpoint_audit_domain,
            "validate_retained_evaluation",
            validate,
        )
    else:
        original = (
            runner._open_development_supervised_evaluation_from_physical_bytes
        )
        calls = defaultdict(int)

        def open_supervised(
            root: Path, *, definition: object, iteration: int,
        ) -> object:
            calls[str(iteration)] += 1
            if iteration == 2 and calls[str(iteration)] == 2:
                (Path(root) / "metrics.json").write_bytes(
                    b"changed-between-aggregate-passes"
                )
            return original(
                root, definition=definition, iteration=iteration,
            )

        monkeypatch.setattr(
            runner,
            "_open_development_supervised_evaluation_from_physical_bytes",
            open_supervised,
        )

    destination = tmp_path / f"aggregate-internal-{mutation}"
    with pytest.raises(
        (ValueError, FileNotFoundError),
        match="artifact|hash|bytes|changed|physical|checkpoint|missing",
    ):
        runner.publish_development_aggregate(
            **fixture.kwargs,
            output_root=destination,
            repository_identity_provider=_repository_provider,
        )
    assert not destination.exists()
