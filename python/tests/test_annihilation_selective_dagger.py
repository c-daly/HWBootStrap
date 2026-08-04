"""Frozen-definition and oracle-preflight tests for selective DAgger."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ml_lab.dagger import (
    OracleBenchmarkDecision,
    OraclePreflightGameResult,
    load_panel_definition,
    run_oracle_preflight,
    validate_panel_definition,
)
from ml_lab.tactical_trace import (
    CommandFrame,
    EpisodeTrace,
    SeatFrame,
    StateFrame,
    TransitionFrame,
)


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
        "ignored_generated_root": (
            "python/panels/annihilation-selective-dagger-v1/evidence/"
        ),
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
        (item.depth, item.expansion_budget, item.heuristic_identity)
        for item in definition.oracle_candidates
    ] == [
        (4, 512, "material-plus-pursuit-v1"),
        (4, 2_048, "material-plus-pursuit-v1"),
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


def _preflight_boundaries(
    *,
    wins: dict[int, int],
    cycling_draws: dict[int, int] | None = None,
    seconds_per_query: dict[int, float] | None = None,
    nondeterministic: bool = False,
    round_trip_failure: bool = False,
) -> tuple[Any, Any, _FakeClock, dict[int, list], Counter]:
    clock = _FakeClock()
    schedules: dict[int, list] = defaultdict(list)
    queries: Counter = Counter()
    cycling_draws = cycling_draws or {}
    seconds_per_query = seconds_per_query or {512: 0.001, 2048: 0.001}
    legal_mask = (True, *([False] * 1287))

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
            samples=(f"{game.map_seed}:{game.learner_seat}",),
        )

    def benchmark(oracle, game, sample):
        key = (oracle.expansion_budget, sample)
        queries[key] += 1
        clock.advance(seconds_per_query[oracle.expansion_budget])
        action = (
            1
            if nondeterministic and queries[key] == 2
            else 0
        )
        round_trip_action = 1 if round_trip_failure else action
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
            round_trip_action=round_trip_action,
            legal_mask=legal_mask,
            command=command,
            actual_expansion_count=min(127, oracle.expansion_budget),
        )

    return evaluator, benchmark, clock, schedules, queries


def test_oracle_preflight_runs_identical_240_game_schedules_and_double_queries(
    tmp_path: Path,
) -> None:
    """Dropping a seat/map or one determinism query would invalidate oracle selection."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, clock, schedules, queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    callbacks: list[int] = []

    selected = run_oracle_preflight(
        definition,
        output_root=tmp_path / "oracle-preflight",
        repository_hash="a" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
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
    assert manifest["selected_oracle"]["expansion_budget"] == 2048
    assert [item["games"] for item in manifest["candidates"]] == [240, 240]
    assert [item["labels"] for item in manifest["candidates"]] == [240, 240]
    assert all(item["determinism_failures"] == 0 for item in manifest["candidates"])
    assert all(item["round_trip_failures"] == 0 for item in manifest["candidates"])
    assert all(item["labels_per_second"] >= 10.0 for item in manifest["candidates"])


def test_oracle_preflight_accepts_exact_win_and_throughput_boundaries(
    tmp_path: Path,
) -> None:
    """The locked gates are inclusive: 204/240 wins and 10 labels/s both pass."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, base_benchmark, clock, _schedules, _queries = _preflight_boundaries(
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
        repository_hash="9" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
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
    evaluator, benchmark, clock, _schedules, _queries = _preflight_boundaries(
        wins=wins,
        cycling_draws=cycling,
        seconds_per_query=seconds,
    )

    selected = run_oracle_preflight(
        definition,
        output_root=tmp_path / f"preflight-{expected}",
        repository_hash="b" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
        clock=clock,
    )

    assert selected.expansion_budget == expected


@pytest.mark.parametrize("failure", ["wins", "throughput", "determinism", "round_trip"])
def test_oracle_preflight_gate_failure_blocks_downstream_callback(
    tmp_path: Path, failure: str,
) -> None:
    """No collection or training callback may follow an oracle that misses any gate."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, clock, _schedules, _queries = _preflight_boundaries(
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
            repository_hash="c" * 64,
            evaluator=evaluator,
            benchmark=benchmark,
            clock=clock,
            on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
        )

    assert callbacks == []
    assert not (tmp_path / f"failed-{failure}").exists()
    assert (tmp_path / f"failed-{failure}.staging" / "diagnostic.json").is_file()


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
            samples=("sample",),
        )

    with pytest.raises(ValueError, match="cycling|waste|diagnostic|trace"):
        run_oracle_preflight(
            definition,
            output_root=tmp_path / f"unsupported-{diagnostic}",
            repository_hash="5" * 64,
            evaluator=evaluator,
            benchmark=lambda *_args: pytest.fail(
                "diagnostics must be checked before an oracle query"
            ),
        )


def test_oracle_preflight_exact_reuse_launches_zero_games_and_rehashes_evidence(
    tmp_path: Path,
) -> None:
    """Reuse must physically reopen all traces/replays and reject any identity drift."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    selected = run_oracle_preflight(
        definition,
        output_root=root,
        repository_hash="d" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
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
        repository_hash="d" * 64,
        evaluator=forbidden,
        benchmark=forbidden,
    )
    assert reused == selected
    assert calls == 0

    before = (root / "oracle-preflight.json").read_bytes()
    with pytest.raises(ValueError, match="identity|reusable"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_hash="e" * 64,
            evaluator=forbidden,
            benchmark=forbidden,
        )
    assert (root / "oracle-preflight.json").read_bytes() == before
    assert calls == 0

    replay = next((root / "games").rglob("*.replay"))
    replay.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash|reusable|evidence"):
        run_oracle_preflight(
            definition,
            output_root=root,
            repository_hash="d" * 64,
            evaluator=forbidden,
            benchmark=forbidden,
        )
    assert calls == 0


def test_oracle_preflight_reuse_recomputes_reported_metrics_from_physical_games(
    tmp_path: Path,
) -> None:
    """A refreshed self-hash cannot make a false rate or selection authoritative."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    run_oracle_preflight(
        definition,
        output_root=root,
        repository_hash="8" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
        clock=clock,
    )
    manifest_path = root / "oracle-preflight.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            repository_hash="8" * 64,
            evaluator=forbidden,
            benchmark=forbidden,
        )
    assert calls == 0


def test_oracle_preflight_recovers_completed_staging_and_rejects_coexistence(
    tmp_path: Path,
) -> None:
    """A crash after sealing is reusable, while two authoritative roots are not."""

    definition = load_panel_definition(PANEL_PATH, repository_root=ROOT)
    evaluator, benchmark, clock, _schedules, _queries = _preflight_boundaries(
        wins={512: 210, 2048: 220},
    )
    root = tmp_path / "oracle-preflight"
    selected = run_oracle_preflight(
        definition,
        output_root=root,
        repository_hash="7" * 64,
        evaluator=evaluator,
        benchmark=benchmark,
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
        repository_hash="7" * 64,
        evaluator=forbidden,
        benchmark=forbidden,
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
            repository_hash="7" * 64,
            evaluator=forbidden,
            benchmark=forbidden,
        )
    assert (root / "oracle-preflight.json").read_bytes() == before
    assert (staging / "sentinel.txt").read_text(encoding="utf-8") == "do not overwrite\n"
    assert calls == 0


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
            repository_hash="6" * 64,
            evaluator=failed_evaluator,
            benchmark=lambda *_args: pytest.fail("benchmark must not run"),
            on_selected=lambda oracle: callbacks.append(oracle.expansion_budget),
        )

    staging = root.with_name(root.name + ".staging")
    assert not root.exists()
    assert not (staging / "oracle-preflight.json").exists()
    diagnostic = json.loads(
        (staging / "diagnostic.json").read_text(encoding="utf-8")
    )
    assert diagnostic["status"] == "failed"
    assert diagnostic["exception"] == {
        "type": "ConnectionError",
        "message": "test evaluator disconnected",
    }
    assert callbacks == []
