"""Physical-contract tests for immutable selective-DAgger overlays."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from dataclasses import FrozenInstanceError, asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import ml_lab.dagger as dagger_module
from ml_lab.contracts import EnvironmentContract
from ml_lab.controllers import ControllerSpec, ResolvedController
from ml_lab.imitation import (
    DemonstrationGame,
    DemonstrationWriter,
    Source,
    SourceMixtureSampler,
    load_imitation_dataset,
)
from ml_lab.dagger import (
    CollectionDefinition,
    DaggerGame,
    DaggerOverlay,
    DaggerOverlayManifest,
    DaggerOverlayWriter,
    DaggerRow,
    IterationManifest,
    LearnerIdentity,
    OriginalDatasetIdentity,
    OracleBenchmarkDecision,
    OraclePreflightGameResult,
    OverlayDefinition,
    OracleSpec,
    SEED_DEFINITIONS,
    ScheduledDuel,
    collect_selective_dagger,
    load_panel_definition,
    open_dagger_overlay,
    publish_dagger_overlay,
    publish_dagger_overlays,
    require_seed_in_partition,
    run_oracle_preflight,
    validate_seed_definitions,
    validate_panel_definition,
)


HASHES = {letter: letter * 64 for letter in "abcdef1234567890"}


def _symlink_or_skip_windows_privilege(
    link: Path, target: Path, *, target_is_directory: bool = False,
) -> None:
    """Skip only when Windows explicitly denies symbolic-link privilege."""

    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:
        if os.name == "nt" and getattr(exc, "winerror", None) == 1314:
            pytest.skip(f"Windows symbolic-link privilege is unavailable: {exc}")
        raise


def _windows_directory_junction(link: Path, target: Path) -> None:
    if os.name != "nt":
        pytest.skip("Windows junction coverage runs only on Windows")
    completed = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise OSError(
            f"could not create Windows junction: {completed.stderr}"
        )


@pytest.fixture
def contract() -> EnvironmentContract:
    return EnvironmentContract(
        version="tactical-v2",
        contract_hash=HASHES["a"],
        encoding_hash=HASHES["b"],
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


def _oracle_payload() -> dict[str, Any]:
    return {
        "oracle_type": "bounded-search",
        "depth": 4,
        "expansion_budget": 512,
        "use_heuristic": True,
        "heuristic_identity": "material-plus-pursuit-v1",
        "code_hash": HASHES["c"],
    }


def _learner_payload(checkpoint: Path) -> dict[str, Any]:
    return {
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_run": "seed-227-step-38912",
        "source_manifest_sha256": HASHES["d"],
    }


def _dataset_payload() -> dict[str, Any]:
    return {
        "manifest_sha256": HASHES["5"],
        "files": [
            {"path": "shards/base-000.npz", "sha256": HASHES["6"]},
            {"path": "games.jsonl", "sha256": HASHES["7"]},
        ],
    }


def _command(kind: str, seat: int = 0) -> dict[str, Any]:
    values: dict[str, Any] = {
        "Kind": kind,
        "Issuer": seat,
        "ActorId": None,
        "TargetId": None,
        "Q": None,
        "R": None,
    }
    if kind == "move":
        values.update(ActorId=10, Q=1, R=0)
    elif kind == "attack":
        values.update(ActorId=10, TargetId=20)
    elif kind == "deploy":
        values.update(Q=1, R=0)
    return values


def _row_payload(*, decision_index: int = 3, state_hash: str | None = None) -> dict[str, Any]:
    return {
        "observation": [0.25, -0.5],
        "legal_mask": [True, True, False, True, False, True, False],
        "learner_action": 1,
        "learner_command": _command("move"),
        "teacher_action": 3,
        "teacher_command": _command("attack"),
        "reason_bits": 15,
        "state_hash": state_hash or HASHES["e"],
        "normalized_advantage": 0.25,
        "opponent_living_unit_count": 1,
        "productive_legal_action_count": 2,
        "seat": 0,
        "round": 4,
        "decision_index": decision_index,
        "disagreement": True,
        "oracle_actual_expansion_count": 127,
    }


def _game_payload(
    *, game_id: int = 0, partition: str = "train", seat: int = 0,
    iteration: int = 1, profile: str = "standard-3v3",
) -> dict[str, Any]:
    seed = (
        18_000_000 + (iteration - 1) * 100_000
        if partition == "train"
        else 19_000_000 + (iteration - 1) * 10_000
    )
    return {
        "game_id": game_id,
        "partition": partition,
        "iteration": iteration,
        "map_seed": seed,
        "episode_seed": seed,
        "schedule_index": 0,
        "profile": profile,
        "reference_seat": 0,
        "learner_seat": seat,
        "opponent": "random",
        "outcome": "win" if seat == 0 else "loss",
        "transition_count": 1,
        "trace_path": f"evidence/game-{game_id}.trace.json",
        "replay_path": f"evidence/game-{game_id}.replay",
    }


def _definition_payload(checkpoint: Path, *, partition: str = "train") -> dict[str, Any]:
    return {
        "partition": partition,
        "iteration": 1,
        "observation_size": 2,
        "action_size": 7,
        "action_regions": {
            "move": {"offset": 1, "count": 2},
            "attack": {"offset": 3, "count": 2},
            "deploy": {"offset": 5, "count": 2},
        },
        "oracle": _oracle_payload(),
        "learner": _learner_payload(checkpoint),
        "original_dataset": _dataset_payload(),
        "scenario_hash": HASHES["1"],
        "contract_hash": HASHES["a"],
        "encoding_hash": HASHES["b"],
        "repository_hash": HASHES["2"],
        "panel_hash": HASHES["3"],
        "schedule_hash": HASHES["4"],
        "label_target": 20_000 if partition == "train" else 2_000,
        "game_ceiling": 2_000 if partition == "train" else 200,
    }


def test_dagger_schema_api_is_importable() -> None:
    """The overlay contract must be exposed from its dedicated module."""

    assert all(
        item is not None
        for item in (
            OracleSpec,
            LearnerIdentity,
            DaggerRow,
            DaggerGame,
            DaggerOverlay,
            DaggerOverlayWriter,
            open_dagger_overlay,
            publish_dagger_overlay,
            publish_dagger_overlays,
            require_seed_in_partition,
            validate_seed_definitions,
            load_panel_definition,
            validate_panel_definition,
            OracleBenchmarkDecision,
            OraclePreflightGameResult,
            run_oracle_preflight,
        )
    )


@pytest.mark.parametrize(
    ("factory", "payload"),
    [
        (OracleSpec.from_dict, _oracle_payload()),
        (DaggerGame.from_dict, _game_payload()),
    ],
)
@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_schema_parsers_reject_missing_and_extra_fields(factory, payload, mutation: str) -> None:
    """Deleting strict-key equality must make malformed provenance parse successfully."""

    malformed = copy.deepcopy(payload)
    if mutation == "missing":
        malformed.pop(next(iter(malformed)))
    else:
        malformed["surprise"] = 1
    with pytest.raises(ValueError, match="fields"):
        factory(malformed)


def test_schema_parsers_reject_coercible_values_and_are_frozen(tmp_path: Path) -> None:
    """Replacing exact type checks with int/str coercion must make this test fail."""

    checkpoint = tmp_path / "actor.zip"
    checkpoint.write_bytes(b"actor")
    learner_payload = _learner_payload(checkpoint)
    learner_payload["checkpoint_sha256"] = 123
    with pytest.raises(ValueError, match="checkpoint_sha256"):
        LearnerIdentity.from_dict(learner_payload)

    oracle_payload = _oracle_payload()
    oracle_payload["depth"] = "4"
    with pytest.raises(ValueError, match="depth"):
        OracleSpec.from_dict(oracle_payload)

    oracle = OracleSpec.from_dict(_oracle_payload())
    assert oracle.use_heuristic is True
    assert oracle.to_dict()["use_heuristic"] is True
    with pytest.raises(FrozenInstanceError):
        oracle.depth = 5  # type: ignore[misc]

    oracle_payload = _oracle_payload()
    oracle_payload["use_heuristic"] = 1
    with pytest.raises(ValueError, match="use_heuristic"):
        OracleSpec.from_dict(oracle_payload)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_learner_identity_parser_is_exact(
    tmp_path: Path, mutation: str
) -> None:
    """Relaxing learner field equality must admit unbound checkpoint provenance."""

    checkpoint = tmp_path / "actor.zip"
    checkpoint.write_bytes(b"actor")
    payload = _learner_payload(checkpoint)
    if mutation == "missing":
        payload.pop("source_run")
    else:
        payload["surprise"] = True
    with pytest.raises(ValueError, match="fields"):
        LearnerIdentity.from_dict(payload)


def test_original_dataset_identity_is_strict_frozen_and_deeply_immutable() -> None:
    """Mutable mappings or malformed file identities must not enter provenance."""

    payload = _dataset_payload()
    identity = OriginalDatasetIdentity.from_dict(payload)
    payload["files"][0]["sha256"] = HASHES["8"]
    assert identity.to_dict() == _dataset_payload()
    assert isinstance(identity.files, tuple)
    with pytest.raises(FrozenInstanceError):
        identity.manifest_sha256 = HASHES["8"]  # type: ignore[misc]

    malformed = _dataset_payload()
    malformed["files"][0]["extra"] = True
    with pytest.raises(ValueError, match="fields"):
        OriginalDatasetIdentity.from_dict(malformed)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(observation=[0.0]), "observation"),
        (lambda row: row.update(observation=[0.0, float("nan")]), "finite"),
        (lambda row: row.update(legal_mask=[True]), "legal mask"),
        (lambda row: row.update(teacher_action=4), "legal"),
        (lambda row: row.update(teacher_command=_command("move")), "action region"),
        (lambda row: row.update(disagreement=False), "disagreement"),
        (lambda row: row.update(reason_bits=16), "reason"),
        (lambda row: row.update(oracle_actual_expansion_count=513), "expansion"),
        (lambda row: row.update(seat=True), "seat"),
        (lambda row: row.update(extra=True), "fields"),
    ],
)
def test_dagger_row_schema_rejects_malformed_policy_and_diagnostic_values(
    contract: EnvironmentContract, mutate, message: str
) -> None:
    """Removing any row-boundary check must accept one malformed live-protocol row."""

    payload = _row_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        DaggerRow.from_dict(
            payload, contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
        )


def test_overlay_definition_is_frozen_exact_and_deeply_immutable(
    tmp_path: Path,
) -> None:
    """Reuse expectations must not retain mutable nested provenance inputs."""

    checkpoint = tmp_path / "actor.zip"
    checkpoint.write_bytes(b"actor")
    payload = _definition_payload(checkpoint)
    definition = OverlayDefinition.from_dict(payload)
    payload["oracle"]["depth"] = 9
    payload["action_regions"]["move"]["offset"] = 99
    assert definition.oracle.depth == 4
    assert definition.action_regions[0] == ("move", 1, 2)
    assert definition.to_dict() == _definition_payload(checkpoint)
    with pytest.raises(FrozenInstanceError):
        definition.iteration = 2  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda row: row.update(observation=[1, -0.5]), "observation"),
        (lambda row: row.update(observation=[True, -0.5]), "observation"),
        (lambda row: row.update(observation=[3.5e38, -0.5]), "float32"),
        (lambda row: row.update(normalized_advantage=1), "advantage"),
        (lambda row: row.update(normalized_advantage="0.25"), "advantage"),
        (lambda row: row.update(round=2**31), "round"),
        (lambda row: row.update(decision_index=2**31), "decision index"),
        (
            lambda row: row["learner_command"].update(ActorId=-1),
            "ActorId",
        ),
        (
            lambda row: row["teacher_command"].update(TargetId=-1),
            "TargetId",
        ),
    ],
)
def test_dagger_row_parser_rejects_invalid_float_int32_and_id_values(
    contract: EnvironmentContract, mutate, message: str,
) -> None:
    """Storage DTOs reject wrong JSON types, nonfinite narrowing, and integer wrap."""

    payload = _row_payload()
    mutate(payload)
    with pytest.raises(ValueError, match=message):
        DaggerRow.from_dict(
            payload, contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
        )


def test_dagger_row_preserves_every_eligibility_reason(
    contract: EnvironmentContract,
) -> None:
    """Collapsing eligibility to one reason must lose the literal four-bit mask."""

    row = DaggerRow.from_dict(
        _row_payload(), contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
    )
    assert row.reason_bits == 0b1111
    assert row.to_dict()["reason_bits"] == 0b1111


def test_dagger_row_parser_rejects_a_missing_field(
    contract: EnvironmentContract,
) -> None:
    """Using get/default for a row field would silently fabricate diagnostics."""

    payload = _row_payload()
    payload.pop("round")
    with pytest.raises(ValueError, match="fields"):
        DaggerRow.from_dict(
            payload, contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
        )


def test_schema_parsers_convert_unhashable_coercions_to_value_errors(
    contract: EnvironmentContract,
) -> None:
    """Direct set-membership must not leak TypeError for malformed JSON values."""

    game = _game_payload()
    game["profile"] = []
    with pytest.raises(ValueError, match="profile"):
        DaggerGame.from_dict(game)
    row = _row_payload()
    row["teacher_command"]["Kind"] = []
    with pytest.raises(ValueError, match="command kind"):
        DaggerRow.from_dict(
            row, contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
        )


@pytest.mark.parametrize(
    "unsafe",
    [
        r"\root-relative.trace.json",
        r"C:drive-relative.trace.json",
        r"C:\absolute.trace.json",
        "../escape.trace.json",
    ],
)
def test_game_evidence_paths_reject_every_windows_escape_form(unsafe: str) -> None:
    """Drive/root/anchor and traversal syntax must never name overlay evidence."""

    payload = _game_payload()
    payload["trace_path"] = unsafe
    with pytest.raises(ValueError, match="contained relative"):
        DaggerGame.from_dict(payload)


@pytest.mark.parametrize(
    ("seed", "partition", "iteration"),
    [
        (18_900_000, "oracle_preflight", None),
        (18_000_000, "train", 1),
        (18_100_000, "train", 2),
        (18_200_000, "train", 3),
        (19_000_000, "validation", 1),
        (19_010_000, "validation", 2),
        (19_020_000, "validation", 3),
        (18_990_000, "smoke", None),
        (19_030_000, "reserved", None),
        (20_000_000, "development_evaluation", None),
    ],
)
def test_seed_definitions_accept_each_explicit_namespace(
    seed: int, partition: str, iteration: int | None
) -> None:
    """Dropping a named range from the authoritative table must reject its lower bound."""

    require_seed_in_partition(seed, partition, iteration)


def test_seed_definition_table_is_exact_and_every_range_is_inclusive() -> None:
    """Range drift or an off-by-one partition check must fail this locked contract."""

    expected = (
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
    assert SEED_DEFINITIONS == expected
    for partition, iteration, start, stop in expected:
        require_seed_in_partition(start, partition, iteration)
        require_seed_in_partition(stop, partition, iteration)
        with pytest.raises(ValueError, match="outside"):
            require_seed_in_partition(start - 1, partition, iteration)
        with pytest.raises(ValueError, match="outside"):
            require_seed_in_partition(stop + 1, partition, iteration)


def test_seed_partition_is_never_inferred_from_a_numeric_prefix() -> None:
    """Prefix inference would incorrectly accept a train seed as validation."""

    with pytest.raises(ValueError, match="partition"):
        require_seed_in_partition(18_000_000, "validation", 1)
    with pytest.raises(ValueError, match="iteration"):
        require_seed_in_partition(18_000_000, "train", 2)


def test_seed_definition_validator_rejects_any_overlap() -> None:
    """Removing whole-table overlap detection must accept a duplicated endpoint."""

    with pytest.raises(ValueError, match="overlap"):
        validate_seed_definitions(
            (
                ("one", None, 1, 3),
                ("two", None, 3, 5),
            )
        )


def _row(
    contract: EnvironmentContract,
    *,
    seat: int,
    decision_index: int = 3,
    state_hash: str | None = None,
) -> DaggerRow:
    payload = _row_payload(decision_index=decision_index, state_hash=state_hash)
    payload["seat"] = seat
    payload["learner_command"] = _command("move", seat)
    payload["teacher_command"] = _command("attack", seat)
    return DaggerRow.from_dict(
        payload, contract=contract, oracle=OracleSpec.from_dict(_oracle_payload())
    )


def _game(
    *, game_id: int, partition: str, seat: int, reference_seat: int = 0,
    iteration: int = 1, profile: str = "standard-3v3",
) -> DaggerGame:
    payload = _game_payload(
        game_id=game_id, partition=partition, seat=seat,
        iteration=iteration, profile=profile,
    )
    payload["reference_seat"] = reference_seat
    return DaggerGame.from_dict(payload)


def _write_evidence(root: Path, game: DaggerGame) -> None:
    trace = root / game.trace_path
    replay = root / game.replay_path
    trace.parent.mkdir(parents=True, exist_ok=True)
    winner = (
        game.learner_seat if game.outcome == "win"
        else 1 - game.learner_seat if game.outcome == "loss"
        else None
    )
    trace.write_text(json.dumps(_terminal_trace(winner=winner)) + "\n", encoding="utf-8")
    replay.write_text("HEXWARS-REPLAY 1\n", encoding="utf-8")


def _terminal_trace(*, winner: int | None) -> dict[str, Any]:
    def seat(number: int) -> dict[str, Any]:
        return {
            "Seat": number,
            "Points": 0,
            "DestroyedValue": 0,
            "AliveUnits": 0,
            "CurrentHitPoints": 0,
            "MaximumHitPoints": 0,
            "HealthAdjustedMaterial": 0.0,
            "CanDamageEnemy": False,
            "CanCurrentlyAttackEnemy": False,
            "CanMove": False,
            "Units": [],
        }

    def state(game_over: bool, state_winner: int | None) -> dict[str, Any]:
        return {
            "Round": 1,
            "ActiveSeat": 0,
            "IsGameOver": game_over,
            "Winner": state_winner,
            "ProductiveLegalActions": 0,
            "Seats": [seat(0), seat(1)],
            "ControlledHexes": [],
        }

    return {
        "schema_version": 1,
        "transitions": [{
            "Before": state(False, None),
            "Command": {
                "Kind": "end_turn", "Issuer": 0, "ActorId": None,
                "TargetId": None, "Q": None, "R": None,
            },
            "After": state(True, winner),
        }],
    }


def _new_writer(
    root: Path,
    contract: EnvironmentContract,
    *,
    partition: str = "train",
    repository_hash: str = HASHES["2"],
    original_dataset: OriginalDatasetIdentity | None = None,
    iteration: int = 1,
    scenario_hash: str = HASHES["1"],
    learner_checkpoint: Path | None = None,
    learner_source_run: str = "seed-227-step-38912",
    learner_source_manifest_sha256: str = HASHES["d"],
    oracle: OracleSpec | None = None,
    panel_hash: str = HASHES["3"],
    schedule_hash: str = HASHES["4"],
) -> tuple[DaggerOverlayWriter, Path]:
    checkpoint = (
        root.parent / "actor.zip"
        if learner_checkpoint is None
        else Path(learner_checkpoint)
    )
    if learner_checkpoint is None:
        checkpoint.write_bytes(b"actor")
    learner_payload = _learner_payload(checkpoint)
    learner_payload["source_run"] = learner_source_run
    learner_payload["source_manifest_sha256"] = (
        learner_source_manifest_sha256
    )
    writer = DaggerOverlayWriter.create(
        root,
        contract=contract,
        partition=partition,
        iteration=iteration,
        oracle=(OracleSpec.from_dict(_oracle_payload()) if oracle is None else oracle),
        learner=LearnerIdentity.from_dict(learner_payload),
        original_dataset=(
            original_dataset
            if original_dataset is not None
            else OriginalDatasetIdentity.from_dict(_dataset_payload())
        ),
        scenario_hash=scenario_hash,
        repository_hash=repository_hash,
        panel_hash=panel_hash,
        schedule_hash=schedule_hash,
        label_target=20_000 if partition == "train" else 2_000,
        game_ceiling=2_000 if partition == "train" else 200,
    )
    return writer, checkpoint


def _seal_pair(
    root: Path,
    contract: EnvironmentContract,
    *,
    partition: str = "train",
    repository_hash: str = HASHES["2"],
    iteration: int = 1,
    row_count: int = 2,
    learner_checkpoint: Path | None = None,
    learner_source_run: str = "seed-227-step-38912",
    learner_source_manifest_sha256: str = HASHES["d"],
    oracle: OracleSpec | None = None,
    original_dataset: OriginalDatasetIdentity | None = None,
    scenario_hash: str = HASHES["1"],
    panel_hash: str = HASHES["3"],
    schedule_hash: str = HASHES["4"],
) -> tuple[DaggerOverlay, Path]:
    writer, checkpoint = _new_writer(
        root, contract, partition=partition, repository_hash=repository_hash,
        iteration=iteration,
        learner_checkpoint=learner_checkpoint,
        learner_source_run=learner_source_run,
        learner_source_manifest_sha256=learner_source_manifest_sha256,
        oracle=oracle,
        original_dataset=original_dataset,
        scenario_hash=scenario_hash,
        panel_hash=panel_hash,
        schedule_hash=schedule_hash,
    )
    for game_id, seat in enumerate((0, 1)):
        game = _game(
            game_id=game_id, partition=partition, seat=seat,
            iteration=iteration,
        )
        _write_evidence(root, game)
        game_row_count = row_count // 2 + (
            1 if game_id == 0 and row_count % 2 else 0
        )
        rows = (
            [_row(contract, seat=seat)]
            if row_count == 2
            else [
                _row(
                    contract,
                    seat=seat,
                    decision_index=index,
                    state_hash=hashlib.sha256(
                        f"{iteration}:{game_id}:{index}".encode("ascii")
                    ).hexdigest(),
                )
                for index in range(game_row_count)
            ]
        )
        writer.append_game(game, rows)
    return writer.seal(), checkpoint


def _write_base_corpus_dataset(
    root: Path, contract: EnvironmentContract,
) -> tuple[Any, OriginalDatasetIdentity]:
    writer = DemonstrationWriter.create(root, contract=contract, shard_rows=2)

    def append(
        *, partition: str, teacher: str, profile: str, seed: int,
        seat: int, action: int, action_kind: int,
    ) -> None:
        relative = f"replays/{partition}-{teacher}-{seed}-{seat}.replay"
        payload = relative.encode("utf-8")
        replay = root / relative
        replay.parent.mkdir(parents=True, exist_ok=True)
        replay.write_bytes(payload)
        writer.append_game(
            DemonstrationGame(
                partition=partition,
                teacher=teacher,
                teacher_parameters=(
                    {} if teacher == "greedy"
                    else {"depth": 4, "expansion_budget": 512, "use_heuristic": True}
                ),
                opponent="random",
                profile=profile,
                seed=seed,
                teacher_seat=seat,
                replay_path=relative,
                replay_hash=hashlib.sha256(payload).hexdigest(),
                outcome="win",
                scenario_hash=HASHES["1"],
                contract_hash=contract.contract_hash,
                encoding_hash=contract.encoding_hash,
            ),
            [{
                "observation": [float(action), float(seat)],
                "legal_mask": [True] * contract.action_size,
                "action": action,
                "seat": seat,
                "decision_index": 0,
                "action_kind": action_kind,
            }],
        )

    append(
        partition="train", teacher="greedy", profile="standard-3v3",
        seed=11_000_200, seat=0, action=1, action_kind=1,
    )
    append(
        partition="train", teacher="bounded-search", profile="conversion-3v1-near",
        seed=11_500_200, seat=1, action=3, action_kind=2,
    )
    append(
        partition="validation", teacher="greedy", profile="standard-3v3",
        seed=12_000_200, seat=0, action=0, action_kind=0,
    )
    writer.close()
    dataset = load_imitation_dataset(root, expected_contract=contract)
    manifest = root / "manifest.json"
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != manifest
    ]
    identity = OriginalDatasetIdentity.from_dict({
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "files": files,
    })
    return dataset, identity


def _seal_corpus_overlay(
    root: Path,
    contract: EnvironmentContract,
    *,
    original_dataset: OriginalDatasetIdentity,
    partition: str,
    iteration: int,
    teacher_action: int,
    profile: str,
) -> DaggerOverlay:
    writer, _ = _new_writer(
        root,
        contract,
        partition=partition,
        original_dataset=original_dataset,
        iteration=iteration,
    )
    command_kind = (
        "end_turn" if teacher_action == 0
        else "move" if teacher_action < 3
        else "attack" if teacher_action < 5
        else "deploy"
    )
    for game_id, seat in enumerate((0, 1)):
        game = _game(
            game_id=game_id,
            partition=partition,
            seat=seat,
            iteration=iteration,
            profile=profile,
        )
        _write_evidence(root, game)
        payload = _row_payload(
            decision_index=3,
            state_hash=hashlib.sha256(
                f"{partition}-{iteration}-{seat}".encode("ascii")
            ).hexdigest(),
        )
        learner_action = 1 if seat == 0 else teacher_action
        learner_kind = "move" if seat == 0 else command_kind
        payload.update({
            "observation": [float(iteration * 10 + teacher_action), float(seat)],
            "legal_mask": [True] * contract.action_size,
            "seat": seat,
            "learner_action": learner_action,
            "learner_command": _command(learner_kind, seat),
            "teacher_action": teacher_action,
            "teacher_command": _command(command_kind, seat),
            "reason_bits": 1 << ((iteration + seat) % 4),
            "disagreement": learner_action != teacher_action,
        })
        writer.append_game(
            game,
            [DaggerRow.from_dict(
                payload,
                contract=contract,
                oracle=OracleSpec.from_dict(_oracle_payload()),
            )],
        )
    return writer.seal()


def test_build_dagger_corpus_is_cumulative_teacher_labeled_and_validation_isolated(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Replacing cumulative overlays, supervising learner actions, or adding base validation must fail."""

    base_root = tmp_path / "base"
    base, base_identity = _write_base_corpus_dataset(base_root, contract)
    original_shards = tuple((item.path, item.sha256) for item in base.shards)
    original_files = {
        path.relative_to(base_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in base_root.rglob("*") if path.is_file()
    }
    train_one = _seal_corpus_overlay(
        tmp_path / "train-1", contract,
        original_dataset=base_identity, partition="train", iteration=1,
        teacher_action=3, profile="standard-3v3",
    )
    train_two = _seal_corpus_overlay(
        tmp_path / "train-2", contract,
        original_dataset=base_identity, partition="train", iteration=2,
        teacher_action=4, profile="conversion-1v1-far",
    )
    validation_one = _seal_corpus_overlay(
        tmp_path / "validation-1", contract,
        original_dataset=base_identity, partition="validation", iteration=1,
        teacher_action=5, profile="standard-3v3",
    )
    validation_two = _seal_corpus_overlay(
        tmp_path / "validation-2", contract,
        original_dataset=base_identity, partition="validation", iteration=2,
        teacher_action=6, profile="conversion-1v1-far",
    )

    first = dagger_module.build_dagger_corpus(
        base, [train_one], [validation_one],
    )
    cumulative = dagger_module.build_dagger_corpus(
        base, [train_one, train_two], [validation_one, validation_two],
    )

    assert list(cumulative.source_fractions.items()) == [
        (Source.GREEDY_STANDARD, 0.49),
        (Source.SEARCH_CONVERSION, 0.21),
        (Source.DAGGER_TARGETED, 0.30),
    ]
    assert len(first.training.batch.actions) == 4
    assert len(cumulative.training.batch.actions) == 6
    assert len(first.validation.batch.actions) == 2
    assert len(cumulative.validation.batch.actions) == 4
    assert set(cumulative.training.batch.actions) == {1, 3, 4}
    assert set(cumulative.validation.batch.actions) == {5, 6}
    targeted = cumulative.training.batch.sources == Source.DAGGER_TARGETED
    assert cumulative.training.batch.actions[targeted].tolist() == [3, 3, 4, 4]
    assert cumulative.training.batch.observations[targeted, 0].tolist() == [
        13.0, 13.0, 24.0, 24.0,
    ]
    assert set(cumulative.validation.batch.sources) == {Source.DAGGER_TARGETED}
    assert set(cumulative.training.offsets).isdisjoint(cumulative.validation.offsets)
    assert all(
        isinstance(identity, tuple)
        and len(identity) == 3
        and len(identity[0]) == 64
        for identity in (*cumulative.training.offsets, *cumulative.validation.offsets)
    )
    assert tuple(
        identity[0] for identity in cumulative.training.offsets
    ) == (
        base_identity.manifest_sha256,
        base_identity.manifest_sha256,
        train_one.content_identity,
        train_one.content_identity,
        train_two.content_identity,
        train_two.content_identity,
    )
    assert tuple(
        identity[0] for identity in cumulative.validation.offsets
    ) == (
        validation_one.content_identity,
        validation_one.content_identity,
        validation_two.content_identity,
        validation_two.content_identity,
    )
    for partition in (cumulative.training, cumulative.validation):
        assert all(
            not getattr(partition.batch, field).flags.writeable
            for field in partition.batch.__dataclass_fields__
        )
        with pytest.raises(TypeError):
            partition.offsets[next(iter(partition.offsets))] = 0
        with pytest.raises(ValueError, match="read-only"):
            partition.batch.actions[0] = partition.batch.actions[0]
    suffixes = [(identity[1], identity[2]) for identity in cumulative.training.offsets]
    assert len(suffixes) > len(set(suffixes))
    assert tuple((item.path, item.sha256) for item in base.shards) == original_shards
    assert {
        path.relative_to(base_root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in base_root.rglob("*") if path.is_file()
    } == original_files

    sampler = SourceMixtureSampler(
        cumulative.training,
        source_fractions=cumulative.source_fractions,
        batch_size=10,
        seed=227,
    )
    targeted_rows: list[tuple[float, float]] = []
    for _ in range(400):
        batch = sampler.next_batch()
        targeted_rows.extend(
            tuple(float(item) for item in observation)
            for observation in batch.observations[
                batch.sources == Source.DAGGER_TARGETED
            ]
        )
    assert set(Counter(targeted_rows).values()) == {300}


def _identity(value: dict[str, Any]) -> str:
    canonical = {key: item for key, item in value.items() if key != "content_identity"}
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _expected_iteration_source_counts(
    training_rows: int, epochs: int,
) -> dict[str, int]:
    source_order = (
        "greedy_standard", "search_conversion", "dagger_targeted",
    )
    fractions = {
        "greedy_standard": 0.49,
        "search_conversion": 0.21,
        "dagger_targeted": 0.30,
    }
    carry = {source: 0.0 for source in source_order}
    totals = {source: 0 for source in source_order}
    batches = ((training_rows + 255) // 256) * epochs
    for _batch in range(batches):
        targets = {
            source: 256 * fractions[source] + carry[source]
            for source in source_order
        }
        counts = {
            source: max(0, int(np.floor(targets[source] + 1e-12)))
            for source in source_order
        }
        while sum(counts.values()) > 256:
            loser = min(
                (source for source in source_order if counts[source] > 0),
                key=lambda source: (
                    targets[source] - counts[source],
                    source_order.index(source),
                ),
            )
            counts[loser] -= 1
        while sum(counts.values()) < 256:
            winner = max(
                source_order,
                key=lambda source: (
                    targets[source] - counts[source],
                    -source_order.index(source),
                ),
            )
            counts[winner] += 1
        for source in source_order:
            carry[source] = targets[source] - counts[source]
            totals[source] += counts[source]
    return totals


def _iteration_manifest_payload() -> dict[str, Any]:
    source = {
        "schema_version": 1,
        "source_kind": "dagger_actor",
        "controller": {
            "kind": "snapshot",
            "path": "C:/evidence/iterations/iteration-1/actor/checkpoints/step_000000000.zip",
            "source_run": "C:/evidence/iterations/iteration-1/actor",
            "algorithm": "maskable_ppo",
            "step": 0,
            "inference_mode": "deterministic",
        },
        "checkpoint_sha256": HASHES["a"],
        "published_actor_sha256": HASHES["b"],
    }
    learner = {
        "checkpoint_path": source["controller"]["path"],
        "checkpoint_sha256": HASHES["a"],
        "source_run": source["controller"]["source_run"],
        "source_manifest_sha256": HASHES["c"],
    }
    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "completed",
        "iteration": 2,
        "identity": {
            "predecessor": {
                "iteration": 1,
                "content_identity": HASHES["0"],
            },
            "definition": {
                "panel_sha256": HASHES["1"],
                "panel_byte_size": 101,
                "seed_banks_sha256": HASHES["2"],
                "seed_banks_byte_size": 202,
            },
            "repository": {
                "root": "C:/repo",
                "commit": "a" * 40,
                "source_tree": "b" * 40,
                "dirty": False,
            },
            "scenario": {
                "source_sha256": HASHES["3"],
                "runtime_sha256": HASHES["4"],
            },
            "contract": {
                "version": "tactical-v2",
                "contract_hash": HASHES["5"],
                "encoding_hash": HASHES["6"],
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
                "manifest_sha256": HASHES["7"],
                "content_sha256": HASHES["8"],
                "file_count": 3966,
                "byte_size": 17_852_257,
                "contract_hash": HASHES["5"],
                "encoding_hash": HASHES["6"],
                "scenario_hash": HASHES["0"],
            },
            "selected_oracle": {
                "spec": _oracle_payload(),
                "evidence_root": "C:/evidence/oracle-preflight",
                "evidence_content_identity": HASHES["d"],
                "evidence_class": "sealed-engine",
            },
            "incoming_learner": {
                "source": source,
                "identity": learner,
                "published_actor_sha256": HASHES["b"],
            },
            "cumulative_train_overlays": [
                {"iteration": 1, "content_identity": HASHES["e"], "row_count": 20_001},
                {"iteration": 2, "content_identity": HASHES["f"], "row_count": 20_002},
            ],
            "cumulative_validation_overlays": [
                {"iteration": 1, "content_identity": HASHES["1"], "row_count": 2_001},
                {"iteration": 2, "content_identity": HASHES["2"], "row_count": 2_002},
            ],
            "schedules": {
                "train": {
                    "sha256": HASHES["3"],
                    "seed_start": 18_100_000,
                    "seed_stop": 18_199_999,
                    "label_target": 20_000,
                    "game_ceiling": 2_000,
                },
                "validation": {
                    "sha256": HASHES["4"],
                    "seed_start": 19_010_000,
                    "seed_stop": 19_019_999,
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
        },
        "artifacts": {
            "train_overlay": {
                "path": "train-overlay",
                "content_identity": HASHES["f"],
                "row_count": 20_002,
            },
            "validation_overlay": {
                "path": "validation-overlay",
                "content_identity": HASHES["2"],
                "row_count": 2_002,
            },
            "actor": {
                "path": "actor",
                "checkpoint_sha256": HASHES["a"],
                "actor_sha256": HASHES["b"],
                "publication_metadata_sha256": HASHES["c"],
                "run_manifest_sha256": HASHES["d"],
                "bc_manifest_sha256": HASHES["e"],
            },
        },
        "metrics": {
            "train_collection": {
                "games": 20,
                "labels": 20_002,
                "reason_counts": {
                    "conversion": 10,
                    "favorable": 20_002,
                    "cycle_warning": 2,
                    "wasted_end_turn": 3,
                },
                "disagreement_reason_counts": {
                    "conversion": 4,
                    "favorable": 3,
                    "cycle_warning": 1,
                    "wasted_end_turn": 0,
                },
                "disagreements": 4,
                "mean_expansions": 100.0,
                "max_expansions": 512,
            },
            "validation_collection": {
                "games": 4,
                "labels": 2_002,
                "reason_counts": {
                    "conversion": 2,
                    "favorable": 2_002,
                    "cycle_warning": 1,
                    "wasted_end_turn": 0,
                },
                "disagreement_reason_counts": {
                    "conversion": 1,
                    "favorable": 1,
                    "cycle_warning": 0,
                    "wasted_end_turn": 0,
                },
                "disagreements": 1,
                "mean_expansions": 90.0,
                "max_expansions": 512,
            },
            "training": {
                "training_rows": 240_004,
                "validation_rows": 4_003,
                "source_example_counts": {
                    "greedy_standard": 1_411_953,
                    "search_conversion": 605_122,
                    "dagger_targeted": 864_461,
                },
                "best_epoch": 7,
                "best_validation_nll": 0.25,
                "epochs_trained": 12,
            },
        },
        "timings": {
            "elapsed_seconds": 20.0,
            "validation_collection_seconds": 2.0,
            "train_collection_seconds": 8.0,
            "corpus_seconds": 1.0,
            "training_seconds": 8.0,
            "publication_seconds": 1.0,
            "train_labels_per_second": 2_500.25,
            "validation_labels_per_second": 1_001.0,
        },
    }
    payload["content_identity"] = _identity(payload)
    return payload


def _set_iteration_incoming_source(
    payload: dict[str, Any], iteration: int,
) -> None:
    if iteration == 1:
        source_run = (
            "C:/Users/cddal/HexWars/python/runs/"
            "bc227-ppo-random-s227-20260802-v2"
        )
        step = 38_912
        checkpoint_sha256 = (
            "ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561"
        )
        source_manifest_sha256 = (
            "7f02152c2ea39a08e5e203c0b0ba13928b2ad1847e276cc1b19f53331151ba46"
        )
        published_actor_sha256 = checkpoint_sha256
        source: dict[str, Any] = {
            "schema_version": 1,
            "source_kind": "snapshot",
            "controller": {
                "kind": "snapshot",
                "path": (
                    f"{source_run}/checkpoints/step_{step:09d}.zip"
                ),
                "source_run": source_run,
                "algorithm": "maskable_ppo",
                "step": step,
                "inference_mode": "deterministic",
            },
            "checkpoint_sha256": checkpoint_sha256,
        }
    else:
        source_run = (
            f"C:/evidence/iterations/iteration-{iteration - 1}/actor"
        )
        source = {
            "schema_version": 1,
            "source_kind": "dagger_actor",
            "controller": {
                "kind": "snapshot",
                "path": f"{source_run}/checkpoints/step_000000000.zip",
                "source_run": source_run,
                "algorithm": "maskable_ppo",
                "step": 0,
                "inference_mode": "deterministic",
            },
            "checkpoint_sha256": HASHES["a"],
            "published_actor_sha256": HASHES["b"],
        }
        checkpoint_sha256 = HASHES["a"]
        source_manifest_sha256 = HASHES["c"]
        published_actor_sha256 = HASHES["b"]
    payload["identity"]["incoming_learner"] = {
        "source": source,
        "identity": {
            "checkpoint_path": source["controller"]["path"],
            "checkpoint_sha256": checkpoint_sha256,
            "source_run": source["controller"]["source_run"],
            "source_manifest_sha256": source_manifest_sha256,
        },
        "published_actor_sha256": published_actor_sha256,
    }


def _configure_iteration_payload(
    payload: dict[str, Any], iteration: int,
) -> None:
    payload["iteration"] = iteration
    payload["identity"]["predecessor"] = (
        None
        if iteration == 1
        else {
            "iteration": iteration - 1,
            "content_identity": f"{iteration + 9:x}" * 64,
        }
    )
    train = [
        {
            "iteration": item,
            "content_identity": f"{item:x}" * 64,
            "row_count": 20_000 + item,
        }
        for item in range(1, iteration + 1)
    ]
    validation = [
        {
            "iteration": item,
            "content_identity": f"{item + 8:x}" * 64,
            "row_count": 2_000 + item,
        }
        for item in range(1, iteration + 1)
    ]
    payload["identity"]["cumulative_train_overlays"] = train
    payload["identity"]["cumulative_validation_overlays"] = validation
    train_start = 18_000_000 + (iteration - 1) * 100_000
    validation_start = 19_000_000 + (iteration - 1) * 10_000
    payload["identity"]["schedules"]["train"].update({
        "seed_start": train_start,
        "seed_stop": train_start + 99_999,
    })
    payload["identity"]["schedules"]["validation"].update({
        "seed_start": validation_start,
        "seed_stop": validation_start + 9_999,
    })
    payload["artifacts"]["train_overlay"].update({
        "content_identity": train[-1]["content_identity"],
        "row_count": train[-1]["row_count"],
    })
    payload["artifacts"]["validation_overlay"].update({
        "content_identity": validation[-1]["content_identity"],
        "row_count": validation[-1]["row_count"],
    })
    payload["metrics"]["train_collection"]["labels"] = train[-1]["row_count"]
    payload["metrics"]["train_collection"]["reason_counts"]["favorable"] = (
        train[-1]["row_count"]
    )
    payload["metrics"]["validation_collection"]["labels"] = (
        validation[-1]["row_count"]
    )
    payload["metrics"]["validation_collection"]["reason_counts"]["favorable"] = (
        validation[-1]["row_count"]
    )
    training_rows = 200_000 + sum(item["row_count"] for item in train)
    payload["metrics"]["training"]["training_rows"] = training_rows
    payload["metrics"]["training"]["validation_rows"] = sum(
        item["row_count"] for item in validation
    )
    payload["metrics"]["training"]["source_example_counts"] = (
        _expected_iteration_source_counts(
            training_rows,
            payload["metrics"]["training"]["epochs_trained"],
        )
    )
    payload["timings"]["train_labels_per_second"] = (
        train[-1]["row_count"]
        / payload["timings"]["train_collection_seconds"]
    )
    payload["timings"]["validation_labels_per_second"] = (
        validation[-1]["row_count"]
        / payload["timings"]["validation_collection_seconds"]
    )
    _set_iteration_incoming_source(payload, iteration)
    payload["content_identity"] = _identity(payload)


@pytest.mark.parametrize(
    "field",
    [
        "predecessor",
        "definition",
        "repository",
        "scenario",
        "contract",
        "base_dataset",
        "selected_oracle",
        "incoming_learner",
        "cumulative_train_overlays",
        "cumulative_validation_overlays",
        "schedules",
        "optimizer",
        "runtime",
    ],
)
def test_iteration_stage_identity_rejects_every_causal_input_mismatch(
    field: str,
) -> None:
    """A self-consistent rehash must not make a changed stage reusable."""

    expected = IterationManifest.from_dict(_iteration_manifest_payload())
    changed = _iteration_manifest_payload()
    value = changed["identity"][field]
    if field == "predecessor":
        value["content_identity"] = HASHES["f"]
    elif isinstance(value, list):
        value[-1]["content_identity"] = HASHES["0"]
    elif field == "repository":
        value["commit"] = "c" * 40
    elif field == "definition":
        value["panel_byte_size"] += 1
    elif field == "scenario":
        value["runtime_sha256"] = HASHES["0"]
    elif field == "contract":
        value["contract_hash"] = HASHES["0"]
    elif field == "base_dataset":
        value["content_sha256"] = HASHES["0"]
    elif field == "selected_oracle":
        value["evidence_content_identity"] = HASHES["0"]
    elif field == "incoming_learner":
        value["published_actor_sha256"] = HASHES["0"]
    elif field == "schedules":
        value["train"]["sha256"] = HASHES["0"]
    elif field == "runtime":
        value["hardware"]["device_name"] = "different-gpu"
    else:
        value["max_epochs"] = 49
    changed["content_identity"] = _identity(changed)

    with pytest.raises(ValueError, match="identity|invalid|inconsistent"):
        reopened = IterationManifest.from_dict(changed)
        reopened.require_identity(expected.identity)


@pytest.mark.parametrize("section", ["artifacts", "metrics", "timings"])
def test_iteration_manifest_rejects_incomplete_physical_summary(
    section: str,
) -> None:
    """A completion marker needs reconstructable child, training, and timing facts."""

    payload = _iteration_manifest_payload()
    if section == "artifacts":
        payload["artifacts"]["actor"].pop("actor_sha256")
    elif section == "metrics":
        payload["metrics"]["train_collection"].pop("reason_counts")
    else:
        payload["timings"].pop("train_labels_per_second")
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match=section[:-1] if section.endswith("s") else section):
        IterationManifest.from_dict(payload)


def test_iteration_manifest_rejects_boolean_schema_version_alias() -> None:
    payload = _iteration_manifest_payload()
    payload["schema_version"] = True
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="schema_version.*integer"):
        IterationManifest.from_dict(payload)


_ITERATION_CONTINUOUS_PATHS = (
    ("metrics", "train_collection", "mean_expansions"),
    ("metrics", "validation_collection", "mean_expansions"),
    ("metrics", "training", "best_validation_nll"),
    ("timings", "elapsed_seconds"),
    ("timings", "validation_collection_seconds"),
    ("timings", "train_collection_seconds"),
    ("timings", "corpus_seconds"),
    ("timings", "training_seconds"),
    ("timings", "publication_seconds"),
    ("timings", "train_labels_per_second"),
    ("timings", "validation_labels_per_second"),
)


@pytest.mark.parametrize("path", _ITERATION_CONTINUOUS_PATHS)
def test_iteration_manifest_rejects_integer_alias_for_continuous_summary(
    path: tuple[str, ...],
) -> None:
    payload = _iteration_manifest_payload()
    if path[-1] == "train_labels_per_second":
        payload["timings"]["train_collection_seconds"] = 2.0
        payload["timings"]["train_labels_per_second"] = 10_001
    else:
        target: Any = payload
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = int(target[path[-1]])
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="float|continuous"):
        IterationManifest.from_dict(payload)


@pytest.mark.parametrize(
    "path",
    [
        ("metrics", "train_collection", "mean_expansions"),
        ("metrics", "training", "best_validation_nll"),
        ("timings", "elapsed_seconds"),
        ("timings", "validation_labels_per_second"),
    ],
)
def test_iteration_manifest_rejects_boolean_alias_for_continuous_summary(
    path: tuple[str, ...],
) -> None:
    payload = _iteration_manifest_payload()
    target: Any = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = True
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="float|continuous"):
        IterationManifest.from_dict(payload)


@pytest.mark.parametrize(
    "mutation",
    ["overlay_iteration_float", "schedule_seed_float", "runtime_index_float"],
)
def test_iteration_manifest_rejects_float_alias_for_integer_identity(
    mutation: str,
) -> None:
    payload = _iteration_manifest_payload()
    if mutation == "overlay_iteration_float":
        payload["identity"]["cumulative_train_overlays"][1]["iteration"] = 2.0
    elif mutation == "schedule_seed_float":
        payload["identity"]["schedules"]["train"]["seed_start"] = 18_100_000.0
    else:
        payload["identity"]["runtime"]["hardware"]["device_index"] = 0.0
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="integer|runtime"):
        IterationManifest.from_dict(payload)


def test_iteration_manifest_preserves_overlapping_disagreement_reason_counts() -> None:
    manifest = IterationManifest.from_dict(_iteration_manifest_payload())
    train = manifest.metrics["train_collection"]

    assert train["disagreement_reason_counts"] == {
        "conversion": 4,
        "favorable": 3,
        "cycle_warning": 1,
        "wasted_end_turn": 0,
    }
    assert sum(train["disagreement_reason_counts"].values()) > train["disagreements"]
    with pytest.raises(TypeError):
        train["disagreement_reason_counts"]["conversion"] = 0


@pytest.mark.parametrize(
    "mutation",
    ["unknown", "float", "bool", "exceeds_reason", "exceeds_disagreements"],
)
def test_iteration_manifest_rejects_invalid_disagreement_reason_counts(
    mutation: str,
) -> None:
    payload = _iteration_manifest_payload()
    collection = payload["metrics"]["train_collection"]
    counts = collection["disagreement_reason_counts"]
    if mutation == "unknown":
        counts["other"] = 0
    elif mutation == "float":
        counts["conversion"] = 4.0
    elif mutation == "bool":
        counts["conversion"] = True
    elif mutation == "exceeds_reason":
        counts["conversion"] = collection["reason_counts"]["conversion"] + 1
    else:
        counts["conversion"] = collection["disagreements"] + 1
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="disagreement|fields|integer"):
        IterationManifest.from_dict(payload)


def test_iteration_manifest_preserves_distinct_compatible_dataset_contract() -> None:
    """Base demonstrations retain their own contract identity when encoding matches."""

    payload = _iteration_manifest_payload()
    payload["identity"]["base_dataset"]["contract_hash"] = HASHES["9"]
    payload["content_identity"] = _identity(payload)

    manifest = IterationManifest.from_dict(payload)

    assert (
        manifest.identity["base_dataset"]["contract_hash"]
        != manifest.identity["contract"]["contract_hash"]
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "train_under_target",
        "validation_under_target",
        "train_over_ceiling",
        "validation_over_ceiling",
        "wrong_source_mixture",
    ],
)
def test_iteration_manifest_rejects_incomplete_collection_or_mixture(
    mutation: str,
) -> None:
    """Self-consistent hashes cannot certify incomplete collection or fake exposure."""

    payload = _iteration_manifest_payload()
    if mutation.endswith("under_target"):
        partition = mutation.removesuffix("_under_target")
        labels = 19_999 if partition == "train" else 1_999
        payload["identity"][f"cumulative_{partition}_overlays"][-1][
            "row_count"
        ] = labels
        payload["artifacts"][f"{partition}_overlay"]["row_count"] = labels
        payload["metrics"][f"{partition}_collection"]["labels"] = labels
        payload["metrics"][f"{partition}_collection"]["reason_counts"][
            "favorable"
        ] = labels
        if partition == "validation":
            payload["metrics"]["training"]["validation_rows"] = 2_001 + labels
        payload["timings"][f"{partition}_labels_per_second"] = (
            labels
            / payload["timings"][f"{partition}_collection_seconds"]
        )
    elif mutation.endswith("over_ceiling"):
        partition = mutation.removesuffix("_over_ceiling")
        payload["metrics"][f"{partition}_collection"]["games"] = (
            2_002 if partition == "train" else 202
        )
    else:
        payload["metrics"]["training"]["source_example_counts"] = {
            "greedy_standard": 1,
            "search_conversion": 1,
            "dagger_targeted": 1,
        }
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="target|ceiling|source example"):
        IterationManifest.from_dict(payload)


@pytest.mark.parametrize("mutation", ["iteration_one_dagger", "wrong_prior_actor"])
def test_iteration_manifest_rejects_wrong_incoming_actor_ownership(
    mutation: str,
) -> None:
    """Persisted provenance enforces snapshot -> actor 1 -> actor 2 ownership."""

    payload = _iteration_manifest_payload()
    if mutation == "iteration_one_dagger":
        _configure_iteration_payload(payload, 1)
        source_run = "C:/evidence/iterations/iteration-0/actor"
        source = payload["identity"]["incoming_learner"]["source"]
        source.update({
            "source_kind": "dagger_actor",
            "published_actor_sha256": HASHES["b"],
        })
        source["controller"].update({
            "path": f"{source_run}/checkpoints/step_000000000.zip",
            "source_run": source_run,
            "step": 0,
        })
    else:
        source_run = "C:/evidence/iterations/iteration-2/actor"
        source = payload["identity"]["incoming_learner"]["source"]
        source["controller"].update({
            "path": f"{source_run}/checkpoints/step_000000000.zip",
            "source_run": source_run,
        })
    learner = payload["identity"]["incoming_learner"]["identity"]
    learner["checkpoint_path"] = source["controller"]["path"]
    learner["source_run"] = source["controller"]["source_run"]
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="incoming learner"):
        IterationManifest.from_dict(payload)


def test_iteration_one_rejects_self_consistent_alternate_snapshot() -> None:
    """An exact-looking step number cannot replace the audited seed-227 bytes."""

    payload = _iteration_manifest_payload()
    _configure_iteration_payload(payload, 1)
    source_run = "C:/forged/starting-run"
    checkpoint = f"{source_run}/checkpoints/step_000038912.zip"
    source = payload["identity"]["incoming_learner"]["source"]
    source["controller"]["source_run"] = source_run
    source["controller"]["path"] = checkpoint
    source["checkpoint_sha256"] = HASHES["e"]
    incoming = payload["identity"]["incoming_learner"]
    incoming["identity"] = {
        "checkpoint_path": checkpoint,
        "checkpoint_sha256": HASHES["e"],
        "source_run": source_run,
        "source_manifest_sha256": HASHES["f"],
    }
    incoming["published_actor_sha256"] = HASHES["d"]
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match="audited iteration-one learner"):
        IterationManifest.from_dict(payload)


@pytest.mark.parametrize("iteration", [1, 2, 3])
def test_iteration_manifest_records_exact_cumulative_overlay_prefix(
    iteration: int,
) -> None:
    """Iteration k owns the canonical train and held-out prefixes 1 through k."""

    payload = _iteration_manifest_payload()
    _configure_iteration_payload(payload, iteration)

    manifest = IterationManifest.from_dict(payload)

    assert [
        item["iteration"]
        for item in manifest.identity["cumulative_train_overlays"]
    ] == list(range(1, iteration + 1))
    assert [
        item["iteration"]
        for item in manifest.identity["cumulative_validation_overlays"]
    ] == list(range(1, iteration + 1))


def _rewrite_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rebind_modified_game(root: Path, game_id: int) -> None:
    """Rebind physical hashes so reopen must inspect shard values, not stale hashes."""

    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    game_descriptor = manifest["games"][game_id]
    game_path = root / game_descriptor["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    shard = game_manifest["shard"]
    shard_path = root / shard["path"]
    shard["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    shard["byte_size"] = shard_path.stat().st_size
    shard["content_identity"] = _identity(shard)
    game_manifest["content_identity"] = _identity(game_manifest)
    _rewrite_json(game_path, game_manifest)
    game_descriptor["sha256"] = hashlib.sha256(game_path.read_bytes()).hexdigest()
    game_descriptor["byte_size"] = game_path.stat().st_size
    game_descriptor["content_identity"] = game_manifest["content_identity"]
    manifest["content_identity"] = _identity(manifest)
    _rewrite_json(manifest_path, manifest)


def _single_iteration_corpus_inputs(
    tmp_path: Path, contract: EnvironmentContract,
) -> tuple[Any, OriginalDatasetIdentity, DaggerOverlay, DaggerOverlay]:
    base, base_identity = _write_base_corpus_dataset(tmp_path / "base", contract)
    train = _seal_corpus_overlay(
        tmp_path / "train-1", contract,
        original_dataset=base_identity, partition="train", iteration=1,
        teacher_action=3, profile="standard-3v3",
    )
    validation = _seal_corpus_overlay(
        tmp_path / "validation-1", contract,
        original_dataset=base_identity, partition="validation", iteration=1,
        teacher_action=5, profile="standard-3v3",
    )
    return base, base_identity, train, validation


def test_build_dagger_corpus_physically_reopens_each_overlay(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A still-valid in-memory overlay must not hide changed physical evidence."""

    base, _base_identity, train, validation = _single_iteration_corpus_inputs(
        tmp_path, contract,
    )
    replay = next((train.root / "evidence").glob("*.replay"))
    replay.write_bytes(replay.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="physical evidence"):
        dagger_module.build_dagger_corpus(base, [train], [validation])


def test_build_dagger_corpus_rejects_changed_physical_base_identity(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Overlay provenance must bind the physical base corpus at build time."""

    base, _base_identity, train, validation = _single_iteration_corpus_inputs(
        tmp_path, contract,
    )
    replay = next((base.root / "replays").glob("*.replay"))
    replay.write_bytes(replay.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="incompatible with the base corpus"):
        dagger_module.build_dagger_corpus(base, [train], [validation])


@pytest.mark.parametrize("ordering", ["duplicate", "reversed"])
def test_build_dagger_corpus_rejects_duplicate_or_noncanonical_iterations(
    tmp_path: Path, contract: EnvironmentContract, ordering: str,
) -> None:
    """Cumulative input is an ordered prefix, never a set or multiset."""

    base, base_identity = _write_base_corpus_dataset(tmp_path / "base", contract)
    train_one = _seal_corpus_overlay(
        tmp_path / "train-1", contract,
        original_dataset=base_identity, partition="train", iteration=1,
        teacher_action=3, profile="standard-3v3",
    )
    train_two = _seal_corpus_overlay(
        tmp_path / "train-2", contract,
        original_dataset=base_identity, partition="train", iteration=2,
        teacher_action=4, profile="conversion-1v1-far",
    )
    validation = _seal_corpus_overlay(
        tmp_path / "validation-1", contract,
        original_dataset=base_identity, partition="validation", iteration=1,
        teacher_action=5, profile="standard-3v3",
    )
    training = (
        [train_one, train_one]
        if ordering == "duplicate"
        else [train_two, train_one]
    )

    with pytest.raises(ValueError, match="iterations are not canonical"):
        dagger_module.build_dagger_corpus(base, training, [validation])


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("shape", "shard shape or dtype"),
        ("teacher_mask", "teacher_action is not legal"),
    ],
)
def test_build_dagger_corpus_reopens_and_rejects_corrupted_shards(
    tmp_path: Path, contract: EnvironmentContract,
    corruption: str, message: str,
) -> None:
    """Rebound hashes must not let malformed physical training rows enter a corpus."""

    base, _base_identity, train, validation = _single_iteration_corpus_inputs(
        tmp_path, contract,
    )
    manifest = json.loads(
        (train.root / "manifest.json").read_text(encoding="utf-8")
    )
    game_path = train.root / manifest["games"][0]["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    shard_path = train.root / game_manifest["shard"]["path"]
    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if corruption == "shape":
        arrays["observations"] = arrays["observations"][:, :1]
    elif corruption == "teacher_mask":
        arrays["packed_masks"][0, 0] &= np.uint8(~(1 << 3) & 0xFF)
    else:
        raise AssertionError(corruption)
    np.savez_compressed(shard_path, **arrays)
    _rebind_modified_game(train.root, 0)

    with pytest.raises(ValueError, match=message):
        dagger_module.build_dagger_corpus(base, [train], [validation])


def test_overlay_storage_writes_exact_compact_shards_and_strict_manifests(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """Changing any physical dtype, descriptor, or identity must break this contract."""

    root = tmp_path / "train.staging"
    overlay, _ = _seal_pair(root, contract)

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version", "status", "partition", "iteration", "observation_size",
        "action_size", "action_regions", "oracle", "learner", "scenario_hash",
        "contract_hash", "encoding_hash", "repository_hash", "panel_hash",
        "schedule_hash", "original_dataset", "label_target", "game_ceiling",
        "game_count", "row_count", "games", "content_identity",
    }
    assert manifest["status"] == "completed"
    assert manifest["game_count"] == 2
    assert manifest["row_count"] == 2
    assert manifest["original_dataset"] == _dataset_payload()
    assert manifest["label_target"] == 20_000
    assert manifest["game_ceiling"] == 2_000
    assert overlay.content_identity == manifest["content_identity"]

    for descriptor in manifest["games"]:
        assert set(descriptor) == {
            "path", "sha256", "byte_size", "game_id", "row_count",
            "content_identity",
        }
        game_manifest = json.loads(
            (root / descriptor["path"]).read_text(encoding="utf-8")
        )
        assert set(game_manifest["trace"]) == {
            "path", "sha256", "byte_size", "seed", "reference_seat",
            "learner_seat", "profile", "outcome", "transition_count",
        }
        assert set(game_manifest["replay"]) == set(game_manifest["trace"])
        assert game_manifest["original_dataset"] == _dataset_payload()
        shard = root / game_manifest["shard"]["path"]
        with np.load(shard, allow_pickle=False) as arrays:
            assert set(arrays.files) == {
                "observations", "packed_masks", "actions", "learner_actions",
                "seats", "rounds", "decision_indices", "reason_bits",
                "state_hashes",
            }
            assert arrays["observations"].dtype == np.float32
            assert arrays["observations"].shape == (1, 2)
            assert arrays["packed_masks"].dtype == np.uint8
            assert arrays["packed_masks"].shape == (1, 1)
            assert arrays["actions"].dtype == np.int32
            assert arrays["learner_actions"].dtype == np.int32
            assert arrays["seats"].dtype == np.int32
            assert arrays["rounds"].dtype == np.int32
            assert arrays["decision_indices"].dtype == np.int32
            assert arrays["reason_bits"].dtype == np.uint8
            assert arrays["state_hashes"].dtype == np.dtype("S64")
            assert arrays["reason_bits"].tolist() == [15]

    reopened = open_dagger_overlay(root)
    assert reopened.content_identity == overlay.content_identity
    assert reopened.games == overlay.games


def test_supervised_examples_are_projected_from_authenticated_overlay_rows(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Task 10 labels and reason flags must originate in physically reopened shards."""

    overlay, _ = _seal_pair(
        tmp_path / "validation", contract, partition="validation",
    )
    examples = dagger_module.dagger_overlay_supervised_examples(
        open_dagger_overlay(overlay.root)
    )

    assert len(examples) == 2
    assert [item["oracle_action"] for item in examples] == [3, 3]
    assert [item["reasons"] for item in examples] == [
        ("conversion", "favorable", "cycle_warning", "wasted_end_turn"),
        ("conversion", "favorable", "cycle_warning", "wasted_end_turn"),
    ]
    assert examples[0]["sample_id"] == (
        f"{overlay.content_identity}:0:3:{HASHES['e']}"
    )


def test_normal_json_float_narrows_to_float32_bits_and_physically_reopens(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A normal JSON double need not equal its deterministic finite float32 storage."""

    json_float = json.loads("0.1")
    assert type(json_float) is float
    assert float(np.float32(json_float)) != json_float
    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    first = _game(game_id=0, partition="train", seat=0)
    second = _game(game_id=1, partition="train", seat=1)
    for game in (first, second):
        _write_evidence(root, game)
    payload = _row_payload()
    payload["observation"] = [json_float, -0.5]
    payload["normalized_advantage"] = json_float
    writer.append_game(first, [payload])
    writer.append_game(second, [_row(contract, seat=1)])

    sealed = writer.seal()
    game_manifest = json.loads(
        (root / "games/game-00000000.json").read_text(encoding="utf-8")
    )
    with np.load(root / game_manifest["shard"]["path"], allow_pickle=False) as shard:
        stored = shard["observations"][0, 0]
        assert stored.view(np.uint32).item() == 0x3DCCCCCD
    assert open_dagger_overlay(root).content_identity == sealed.content_identity


def test_overlay_writer_revalidates_direct_dataset_identity_instances(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Direct dataclass construction must not bypass strict dataset provenance."""

    malformed = OriginalDatasetIdentity(manifest_sha256="bad", files=())
    with pytest.raises(ValueError, match="original dataset"):
        _new_writer(
            tmp_path / "overlay", contract, original_dataset=malformed,
        )


@pytest.mark.parametrize("row_counts", [(0, 1), (0, 0)])
def test_completed_reciprocal_games_allow_zero_label_shards(
    tmp_path: Path, contract: EnvironmentContract, row_counts: tuple[int, int],
) -> None:
    """Selective observation may legitimately retain no labels for either seat."""

    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    for game_id, (seat, row_count) in enumerate(zip((0, 1), row_counts, strict=True)):
        game = _game(game_id=game_id, partition="train", seat=seat)
        _write_evidence(root, game)
        writer.append_game(game, [_row(contract, seat=seat)] if row_count else [])

    overlay = writer.seal()
    assert overlay.row_count == sum(row_counts)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for descriptor, expected_count in zip(manifest["games"], row_counts, strict=True):
        game_manifest = json.loads(
            (root / descriptor["path"]).read_text(encoding="utf-8")
        )
        with np.load(root / game_manifest["shard"]["path"], allow_pickle=False) as shard:
            assert game_manifest["row_count"] == expected_count
            assert shard["observations"].shape == (expected_count, 2)
            assert shard["packed_masks"].shape == (expected_count, 1)
            assert shard["actions"].shape == (expected_count,)


def test_overlay_reopen_rejects_rehashed_original_dataset_tamper(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Per-game base-dataset drift must fail after child hashes are rebound."""

    root = tmp_path / "overlay"
    _seal_pair(root, contract)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    game_path = root / manifest["games"][0]["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    game_manifest["original_dataset"]["manifest_sha256"] = HASHES["8"]
    game_manifest["content_identity"] = _identity(game_manifest)
    _rewrite_json(game_path, game_manifest)
    manifest["games"][0]["sha256"] = hashlib.sha256(game_path.read_bytes()).hexdigest()
    manifest["games"][0]["byte_size"] = game_path.stat().st_size
    manifest["games"][0]["content_identity"] = game_manifest["content_identity"]
    manifest["content_identity"] = _identity(manifest)
    _rewrite_json(root / "manifest.json", manifest)

    with pytest.raises(ValueError, match="original_dataset"):
        open_dagger_overlay(root)


def test_overlay_writer_rejects_evidence_symlink_escape_when_supported(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Resolved evidence targets must remain below the resolved overlay root."""

    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    game_payload = _game_payload()
    game_payload["replay_path"] = "evidence/escape.replay"
    game = DaggerGame.from_dict(game_payload)
    _write_evidence(root, game)
    outside = tmp_path / "outside.replay"
    outside.write_text("outside\n", encoding="utf-8")
    link = root / game.replay_path
    link.unlink()
    _symlink_or_skip_windows_privilege(link, outside)

    with pytest.raises(ValueError, match="contained"):
        writer.append_game(game, [_row(contract, seat=0)])


def test_overlay_reopen_rejects_outer_manifest_symlink_escape_when_supported(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """The expected manifest name must not bless an outside resolved target."""

    root = tmp_path / "overlay"
    _seal_pair(root, contract)
    manifest = root / "manifest.json"
    outside = tmp_path / "outside-manifest.json"
    outside.write_bytes(manifest.read_bytes())
    manifest.unlink()
    _symlink_or_skip_windows_privilege(manifest, outside)

    with pytest.raises(ValueError, match="contained"):
        open_dagger_overlay(root)


def test_overlay_reopen_rejects_outer_manifest_directory(
    tmp_path: Path,
) -> None:
    """The outer manifest must cross the contained-file boundary before parsing."""

    root = tmp_path / "overlay"
    root.mkdir()
    (root / "manifest.json").mkdir()

    with pytest.raises(ValueError, match="contained file"):
        open_dagger_overlay(root)


@pytest.mark.parametrize(
    ("relative", "contents"),
    [
        ("games/nested/extra.json", "{}"),
        ("games/wrong-extension.txt", "extra"),
        ("evidence/extra.replay", "extra"),
        ("shards/wrong-extension.bin", "extra"),
    ],
)
def test_overlay_reopen_rejects_every_unowned_nested_file(
    tmp_path: Path, contract: EnvironmentContract, relative: str, contents: str,
) -> None:
    """Recursive ownership validation must reject every undeclared physical file."""

    root = tmp_path / "overlay"
    _seal_pair(root, contract)
    extra = root / relative
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="unowned"):
        open_dagger_overlay(root)
    assert extra.exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("action_size", "action_size"),
        ("oracle", "depth"),
        ("provenance", "repository_hash"),
        ("game_descriptor", "row_count"),
    ],
)
def test_overlay_manifest_parser_validates_every_nested_declared_field(
    tmp_path: Path, contract: EnvironmentContract, mutation: str, message: str,
) -> None:
    """A strict public DTO parser must validate the complete declared schema."""

    root = tmp_path / "overlay"
    _seal_pair(root, contract)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if mutation == "action_size":
        payload["action_size"] = "7"
    elif mutation == "oracle":
        payload["oracle"]["depth"] = "4"
    elif mutation == "provenance":
        payload["repository_hash"] = "not-a-hash"
    elif mutation == "game_descriptor":
        payload["games"][0]["row_count"] = True
    else:
        raise AssertionError(mutation)
    payload["content_identity"] = _identity(payload)

    with pytest.raises(ValueError, match=message):
        DaggerOverlayManifest.from_dict(payload)


def test_overlay_manifest_and_physical_overlay_keep_canonical_immutable_games(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Logical descriptors stay frozen and complete overlays expose matching games."""

    root = tmp_path / "overlay"
    opened, _ = _seal_pair(root, contract)
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    logical = DaggerOverlayManifest.from_dict(payload)
    payload["games"][0]["game_id"] = 99
    assert logical.games[0].game_id == 0
    assert isinstance(logical.games, tuple)
    assert tuple(game.game_id for game in opened.games) == tuple(
        descriptor.game_id for descriptor in opened.manifest.games
    )
    with pytest.raises(TypeError):
        DaggerOverlay(root=root, manifest=logical, games=())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        DaggerOverlay()  # type: ignore[call-arg]


@pytest.mark.parametrize(
    "rows",
    [
        lambda contract: [
            _row(contract, seat=0, decision_index=3, state_hash=HASHES["e"]),
            _row(contract, seat=0, decision_index=3, state_hash=HASHES["f"]),
        ],
        lambda contract: [
            _row(contract, seat=0, decision_index=3, state_hash=HASHES["e"]),
            _row(contract, seat=0, decision_index=4, state_hash=HASHES["e"]),
        ],
    ],
)
def test_overlay_writer_rejects_duplicate_row_identity_or_episode_state(
    tmp_path: Path, contract: EnvironmentContract, rows
) -> None:
    """Removing either deduplication set must admit ambiguous training labels."""

    root = tmp_path / "train.staging"
    writer, _ = _new_writer(root, contract)
    game = _game(game_id=0, partition="train", seat=0)
    _write_evidence(root, game)
    with pytest.raises(ValueError, match="duplicate"):
        writer.append_game(game, rows(contract))


def test_overlay_writer_requires_a_complete_reciprocal_pair(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """Publishing after one seat would expose a schedule-biased partial artifact."""

    root = tmp_path / "train.staging"
    writer, _ = _new_writer(root, contract)
    game = _game(game_id=0, partition="train", seat=0)
    _write_evidence(root, game)
    writer.append_game(game, [_row(contract, seat=0)])
    with pytest.raises(ValueError, match="reciprocal"):
        writer.seal()


def test_overlay_accepts_reciprocal_pair_when_reference_tracks_learner_seat(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Task 5 may swap both learner and reference seats across a map pair."""

    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    for game_id, seat in enumerate((0, 1)):
        game = _game(
            game_id=game_id, partition="train", seat=seat, reference_seat=seat,
        )
        _write_evidence(root, game)
        writer.append_game(game, [_row(contract, seat=seat)])

    assert writer.seal().games[1].reference_seat == 1


def test_overlay_rejects_reciprocal_games_from_different_schedule_entries(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Two seat games cannot be paired across distinct schedule assignments."""

    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    for game_id, seat in enumerate((0, 1)):
        payload = _game_payload(game_id=game_id, partition="train", seat=seat)
        payload["schedule_index"] = seat
        game = DaggerGame.from_dict(payload)
        _write_evidence(root, game)
        writer.append_game(game, [_row(contract, seat=seat)])

    with pytest.raises(ValueError, match="schedule"):
        writer.seal()


def test_overlay_writer_rejects_nonterminal_completed_trace(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A completed game's outcome must be provable from its final trace state."""

    root = tmp_path / "overlay"
    writer, _ = _new_writer(root, contract)
    game = _game(game_id=0, partition="train", seat=0)
    _write_evidence(root, game)
    trace_path = root / game.trace_path
    trace = _terminal_trace(winner=None)
    trace["transitions"][-1]["After"]["IsGameOver"] = False
    trace_path.write_text(json.dumps(trace) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="terminal"):
        writer.append_game(game, [_row(contract, seat=0)])


def test_overlay_writer_rejects_cross_partition_games(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """Dropping the overlay/game partition check could leak validation labels into train."""

    root = tmp_path / "train.staging"
    writer, _ = _new_writer(root, contract)
    game = _game(game_id=0, partition="validation", seat=0)
    _write_evidence(root, game)
    with pytest.raises(ValueError, match="partition"):
        writer.append_game(game, [_row(contract, seat=0)])


@pytest.mark.parametrize(
    "mutation",
    [
        "corrupt_shard",
        "missing_shard",
        "missing_trace",
        "extra_manifest_field",
        "game_metadata_mismatch",
        "checkpoint_hash_change",
    ],
)
def test_overlay_physical_reopen_fails_closed_on_corruption(
    tmp_path: Path, contract: EnvironmentContract, mutation: str
) -> None:
    """Trusting completed status or stale hashes would accept one of these corruptions."""

    root = tmp_path / "train.staging"
    _, checkpoint = _seal_pair(root, contract)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    game_path = root / manifest["games"][0]["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    shard = root / game_manifest["shard"]["path"]
    trace = root / game_manifest["trace"]["path"]

    if mutation == "corrupt_shard":
        data = bytearray(shard.read_bytes())
        data[len(data) // 2] ^= 1
        shard.write_bytes(data)
    elif mutation == "missing_shard":
        shard.unlink()
    elif mutation == "missing_trace":
        trace.unlink()
    elif mutation == "extra_manifest_field":
        manifest["surprise"] = True
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    elif mutation == "game_metadata_mismatch":
        game_manifest["learner_seat"] = 1
        game_path.write_text(json.dumps(game_manifest), encoding="utf-8")
    elif mutation == "checkpoint_hash_change":
        checkpoint.write_bytes(b"different actor")
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        open_dagger_overlay(root)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_row", "duplicate physical DAgger row identity"),
        ("duplicate_hash", "duplicate physical canonical state hash"),
        ("unused_mask_bit", "nonzero unused bits"),
    ],
)
def test_overlay_reopen_rejects_logical_corruption_with_fresh_outer_hashes(
    tmp_path: Path,
    contract: EnvironmentContract,
    mutation: str,
    message: str,
) -> None:
    """Removing physical row validation must accept a completely rehashed corrupt shard."""

    root = tmp_path / "train.staging"
    writer, _ = _new_writer(root, contract)
    first = _game(game_id=0, partition="train", seat=0)
    second = _game(game_id=1, partition="train", seat=1)
    for game in (first, second):
        _write_evidence(root, game)
    writer.append_game(first, [
        _row(contract, seat=0, decision_index=3, state_hash=HASHES["e"]),
        _row(contract, seat=0, decision_index=4, state_hash=HASHES["f"]),
    ])
    writer.append_game(second, [_row(contract, seat=1)])
    writer.seal()

    game_manifest = json.loads(
        (root / "games/game-00000000.json").read_text(encoding="utf-8")
    )
    shard_path = root / game_manifest["shard"]["path"]
    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name].copy() for name in loaded.files}
    if mutation == "duplicate_row":
        arrays["decision_indices"][1] = arrays["decision_indices"][0]
    elif mutation == "duplicate_hash":
        arrays["state_hashes"][1] = arrays["state_hashes"][0]
    elif mutation == "unused_mask_bit":
        arrays["packed_masks"][0, -1] |= np.uint8(0b1000_0000)
    else:
        raise AssertionError(mutation)
    np.savez_compressed(shard_path, **arrays)
    _rebind_modified_game(root, 0)

    with pytest.raises(ValueError, match=message):
        open_dagger_overlay(root)


@pytest.mark.parametrize(
    "mutation", ["boolean_schema", "boolean_game_id", "boolean_transition_count"]
)
def test_overlay_reopen_rejects_coercible_manifest_values_after_rehash(
    tmp_path: Path, contract: EnvironmentContract, mutation: str
) -> None:
    """Loose equality must not accept booleans as exact manifest integers."""

    root = tmp_path / "train.staging"
    _seal_pair(root, contract)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "boolean_schema":
        manifest["schema_version"] = True
        manifest["content_identity"] = _identity(manifest)
        _rewrite_json(manifest_path, manifest)
    elif mutation == "boolean_game_id":
        manifest["games"][0]["game_id"] = False
        manifest["content_identity"] = _identity(manifest)
        _rewrite_json(manifest_path, manifest)
    elif mutation == "boolean_transition_count":
        game_path = root / manifest["games"][0]["path"]
        game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
        game_manifest["trace"]["transition_count"] = False
        game_manifest["replay"]["transition_count"] = False
        game_manifest["content_identity"] = _identity(game_manifest)
        _rewrite_json(game_path, game_manifest)
        manifest["games"][0]["sha256"] = hashlib.sha256(
            game_path.read_bytes()
        ).hexdigest()
        manifest["games"][0]["byte_size"] = game_path.stat().st_size
        manifest["games"][0]["content_identity"] = game_manifest["content_identity"]
        manifest["content_identity"] = _identity(manifest)
        _rewrite_json(manifest_path, manifest)
    else:
        raise AssertionError(mutation)

    with pytest.raises(ValueError):
        open_dagger_overlay(root)


def test_overlay_reopen_derives_terminal_outcome_from_trace(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """Hash-consistent evidence must still fail when terminal winner contradicts outcome."""

    root = tmp_path / "train.staging"
    writer, _ = _new_writer(root, contract)
    first_payload = _game_payload(game_id=0, partition="train", seat=0)
    first_payload["transition_count"] = 1
    first = DaggerGame.from_dict(first_payload)
    second = _game(game_id=1, partition="train", seat=1)
    for game in (first, second):
        _write_evidence(root, game)
    (root / first.trace_path).write_text(
        json.dumps(_terminal_trace(winner=0)) + "\n", encoding="utf-8"
    )
    writer.append_game(first, [_row(contract, seat=0)])
    writer.append_game(second, [_row(contract, seat=1)])
    writer.seal()

    trace_path = root / first.trace_path
    trace_path.write_text(
        json.dumps(_terminal_trace(winner=1)) + "\n", encoding="utf-8"
    )
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    game_path = root / manifest["games"][0]["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    game_manifest["trace"]["sha256"] = hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    game_manifest["trace"]["byte_size"] = trace_path.stat().st_size
    game_manifest["content_identity"] = _identity(game_manifest)
    _rewrite_json(game_path, game_manifest)
    manifest["games"][0]["sha256"] = hashlib.sha256(
        game_path.read_bytes()
    ).hexdigest()
    manifest["games"][0]["byte_size"] = game_path.stat().st_size
    manifest["games"][0]["content_identity"] = game_manifest["content_identity"]
    manifest["content_identity"] = _identity(manifest)
    _rewrite_json(manifest_path, manifest)

    with pytest.raises(ValueError, match="terminal outcome"):
        open_dagger_overlay(root)


def test_overlay_publication_is_atomic_and_existing_reuse_physically_reopens(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """Renaming before reopen or trusting an existing completed marker must fail this test."""

    staging = tmp_path / "train.staging"
    destination = tmp_path / "train-overlay"
    expected, checkpoint = _seal_pair(staging, contract)
    definition = expected.definition

    published = publish_dagger_overlay(staging, destination, expected=definition)

    assert not staging.exists()
    assert destination.is_dir()
    assert published.content_identity == expected.content_identity
    assert open_dagger_overlay(destination).content_identity == expected.content_identity

    checkpoint.write_bytes(b"tampered after publication")
    with pytest.raises(ValueError):
        publish_dagger_overlay(staging, destination, expected=definition)


@pytest.mark.parametrize(
    ("name", "mutate"),
    [
        ("partition", lambda d: replace(d, partition="validation")),
        ("iteration", lambda d: replace(d, iteration=2)),
        ("repository", lambda d: replace(d, repository_hash=HASHES["8"])),
        ("panel", lambda d: replace(d, panel_hash=HASHES["8"])),
        ("scenario", lambda d: replace(d, scenario_hash=HASHES["8"])),
        ("contract", lambda d: replace(d, contract_hash=HASHES["8"])),
        ("encoding", lambda d: replace(d, encoding_hash=HASHES["8"])),
        ("schedule", lambda d: replace(d, schedule_hash=HASHES["8"])),
        (
            "dataset",
            lambda d: replace(
                d,
                original_dataset=replace(
                    d.original_dataset, manifest_sha256=HASHES["8"],
                ),
            ),
        ),
        (
            "learner_path",
            lambda d: replace(
                d, learner=replace(d.learner, checkpoint_path="missing.zip"),
            ),
        ),
        (
            "learner_hash",
            lambda d: replace(
                d, learner=replace(d.learner, checkpoint_sha256=HASHES["8"]),
            ),
        ),
        (
            "learner_source",
            lambda d: replace(
                d, learner=replace(d.learner, source_run="different-run"),
            ),
        ),
        (
            "learner_manifest",
            lambda d: replace(
                d, learner=replace(d.learner, source_manifest_sha256=HASHES["8"]),
            ),
        ),
        ("oracle", lambda d: replace(d, oracle=replace(d.oracle, depth=5))),
        ("label_target", lambda d: replace(d, label_target=d.label_target + 1)),
        ("game_ceiling", lambda d: replace(d, game_ceiling=d.game_ceiling + 1)),
        (
            "observation_size",
            lambda d: replace(d, observation_size=d.observation_size + 1),
        ),
        ("action_size", lambda d: replace(d, action_size=d.action_size + 1)),
        (
            "action_regions",
            lambda d: replace(
                d,
                action_regions=(
                    ("move", 2, 2), ("attack", 3, 2), ("deploy", 5, 2),
                ),
            ),
        ),
    ],
)
def test_overlay_reuse_rejects_every_stale_expected_identity_without_staging(
    tmp_path: Path, contract: EnvironmentContract, name: str, mutate,
) -> None:
    """Destination-only reuse must compare every immutable expected identity."""

    staging = tmp_path / "train.staging"
    destination = tmp_path / "train-overlay"
    candidate, _ = _seal_pair(staging, contract)
    publish_dagger_overlay(staging, destination, expected=candidate.definition)
    assert not staging.exists()

    with pytest.raises(ValueError, match="expected"):
        publish_dagger_overlay(
            staging, destination, expected=mutate(candidate.definition),
        )


def test_train_and_validation_overlays_publish_to_distinct_destinations(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """A shared destination or swapped partition must never publish a mixed artifact."""

    train_staging = tmp_path / "train.staging"
    validation_staging = tmp_path / "validation.staging"
    train_candidate, _ = _seal_pair(train_staging, contract, partition="train")
    validation_candidate, _ = _seal_pair(
        validation_staging, contract, partition="validation"
    )
    train_destination = tmp_path / "train-overlay"
    validation_destination = tmp_path / "validation-overlay"

    train, validation = publish_dagger_overlays(
        train_staging,
        validation_staging,
        train_destination,
        validation_destination,
        train_expected=train_candidate.definition,
        validation_expected=validation_candidate.definition,
    )

    assert train.partition == "train"
    assert validation.partition == "validation"
    assert train.root != validation.root
    reused_train, reused_validation = publish_dagger_overlays(
        train_staging,
        validation_staging,
        train_destination,
        validation_destination,
        train_expected=train_candidate.definition,
        validation_expected=validation_candidate.definition,
    )
    assert reused_train.content_identity == train.content_identity
    assert reused_validation.content_identity == validation.content_identity


def test_paired_publication_preflights_shared_identities_before_any_rename(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A globally inconsistent pair must leave both complete staging trees intact."""

    train_staging = tmp_path / "train.staging"
    validation_staging = tmp_path / "validation.staging"
    train, _ = _seal_pair(train_staging, contract, partition="train")
    validation, _ = _seal_pair(
        validation_staging,
        contract,
        partition="validation",
        repository_hash=HASHES["8"],
    )
    train_destination = tmp_path / "train-overlay"
    validation_destination = tmp_path / "validation-overlay"

    with pytest.raises(ValueError, match="shared identities"):
        publish_dagger_overlays(
            train_staging,
            validation_staging,
            train_destination,
            validation_destination,
            train_expected=train.definition,
            validation_expected=validation.definition,
        )
    assert train_staging.is_dir() and validation_staging.is_dir()
    assert not train_destination.exists() and not validation_destination.exists()


def test_paired_publication_preflights_existing_destination_conflicts(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A conflicting validation staging tree must be found before train rename."""

    published_validation_staging = tmp_path / "published-validation.staging"
    published_validation, _ = _seal_pair(
        published_validation_staging, contract, partition="validation",
    )
    validation_destination = tmp_path / "validation-overlay"
    publish_dagger_overlay(
        published_validation_staging,
        validation_destination,
        expected=published_validation.definition,
    )

    train_staging = tmp_path / "train.staging"
    train, _ = _seal_pair(train_staging, contract, partition="train")
    conflicting_validation_staging = tmp_path / "conflicting-validation.staging"
    conflicting_validation, _ = _seal_pair(
        conflicting_validation_staging, contract, partition="validation",
    )
    manifest = json.loads(
        (conflicting_validation_staging / "manifest.json").read_text(encoding="utf-8")
    )
    game_path = conflicting_validation_staging / manifest["games"][0]["path"]
    game_manifest = json.loads(game_path.read_text(encoding="utf-8"))
    shard_path = conflicting_validation_staging / game_manifest["shard"]["path"]
    with np.load(shard_path, allow_pickle=False) as loaded:
        arrays = {name: loaded[name] for name in loaded.files}
    arrays["state_hashes"][0] = HASHES["f"]
    np.savez_compressed(shard_path, **arrays)
    _rebind_modified_game(conflicting_validation_staging, 0)
    conflicting_validation = open_dagger_overlay(conflicting_validation_staging)
    train_destination = tmp_path / "train-overlay"

    with pytest.raises(ValueError, match="conflicts"):
        publish_dagger_overlays(
            train_staging,
            conflicting_validation_staging,
            train_destination,
            validation_destination,
            train_expected=train.definition,
            validation_expected=conflicting_validation.definition,
        )
    assert train_staging.is_dir()
    assert not train_destination.exists()


class _DeterministicModel:
    def __init__(self, events: list[tuple[str, Any]]) -> None:
        self.events = events

    def predict(
        self, observation: np.ndarray, *, action_masks: np.ndarray,
        deterministic: bool,
    ) -> tuple[np.ndarray, None]:
        self.events.append(("predict", {
            "observation": observation.tolist(),
            "mask": action_masks.tolist(),
            "deterministic": deterministic,
        }))
        return np.asarray(1), None


def _collection_inputs(
    tmp_path: Path,
    contract: EnvironmentContract,
    *,
    partition: str = "train",
) -> tuple[CollectionDefinition, ResolvedController, OracleSpec]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    checkpoint = tmp_path / f"{partition}-actor.zip"
    checkpoint.write_bytes(b"frozen actor")
    source_run = tmp_path / f"{partition}-source-run"
    source_run.mkdir()
    source_manifest = source_run / "run.json"
    source_manifest.write_text('{"status":"completed"}\n', encoding="utf-8")
    identity = LearnerIdentity.from_dict({
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "source_run": str(source_run.resolve()),
        "source_manifest_sha256": hashlib.sha256(
            source_manifest.read_bytes()
        ).hexdigest(),
    })
    oracle = OracleSpec.from_dict(_oracle_payload())
    definition = CollectionDefinition.create(
        contract=contract,
        partition=partition,
        iteration=1,
        oracle=oracle,
        learner=identity,
        original_dataset=OriginalDatasetIdentity.from_dict(_dataset_payload()),
        scenario_hash=HASHES["1"],
        repository_hash=HASHES["2"],
        panel_hash=HASHES["3"],
    )
    resolved = ResolvedController(
        spec=ControllerSpec(
            kind="run", path=source_run,
            algorithm="maskable_ppo", inference_mode="deterministic",
        ),
        server_controller="external",
        model=_DeterministicModel([]),
        path=checkpoint,
        algorithm="maskable_ppo",
        step=38_912,
        contract=contract,
        observation_size=contract.observation_size,
        action_size=contract.action_size,
        legacy=False,
        promotable=True,
    )
    return definition, resolved, oracle


def _engine_dagger_row(
    *,
    game_index: int,
    decision_index: int,
    seat: int,
    teacher_action: int,
    expansion_count: int,
    state_hash: str | None = None,
) -> dict[str, Any]:
    teacher_kind = "move" if teacher_action == 1 else "attack"
    return {
        "Observation": [0.25, -0.5],
        "LegalMask": [True, True, False, True, False, True, False],
        "LearnerAction": 1,
        "LearnerCommand": _command("move", seat),
        "TeacherAction": teacher_action,
        "TeacherCommand": _command(teacher_kind, seat),
        "Reasons": 15,
        "StateHash": state_hash or f"{game_index * 100_000 + decision_index + 1:064x}",
        "NormalizedAdvantage": 0.25,
        "OpponentLivingUnitCount": 1,
        "ProductiveLegalActionCount": 2,
        "Seat": seat,
        "Round": 4,
        "DecisionIndex": decision_index,
        "Disagreement": teacher_action != 1,
        "OracleDepth": 4,
        "OracleExpansionBudget": 512,
        "OracleHeuristicIdentity": "material-plus-pursuit-v1",
        "OracleActualExpansionCount": expansion_count,
    }


class _CollectionClient:
    def __init__(
        self,
        contract: EnvironmentContract,
        *,
        rows_per_game: int = 0,
        rows_by_game: tuple[int, ...] | None = None,
        duplicate_states: bool = False,
        terminal_on_reset: bool = False,
        write_replay: bool = True,
    ) -> None:
        self.contract = contract
        self.rows_per_game = rows_per_game
        self.rows_by_game = rows_by_game
        self.duplicate_states = duplicate_states
        self.terminal_on_reset = terminal_on_reset
        self.write_replay = write_replay
        self.events: list[tuple[str, Any]] = []
        self.game_index = -1
        self.learner_seat = 0
        self.close_attempts = 0

    def configure_dagger(self, **kwargs: Any) -> None:
        self.events.append(("configure_dagger", kwargs))

    def enable_trace(self, enabled: bool) -> None:
        self.events.append(("enable_trace", enabled))

    def reset(self, **kwargs: Any) -> dict[str, Any]:
        self.game_index += 1
        self.learner_seat = kwargs["learner"]
        assert self.learner_seat in {0, 1}
        assert (kwargs["p0"], kwargs["p1"])[self.learner_seat] == "external"
        self.events.append(("reset", kwargs))
        return {
            "obs": [0.25, -0.5],
            "mask": [True, True, False, True, False, True, False],
            "seat": self.learner_seat,
            "winner": self.learner_seat if self.terminal_on_reset else -1,
            "terminated": self.terminal_on_reset,
            "truncated": False,
            "start_profile": kwargs["start_profile"],
            "reference_seat": kwargs["reference_seat"],
        }

    def step(self, action: int) -> dict[str, Any]:
        self.events.append(("step", action))
        return {
            "obs": [0.25, -0.5],
            "mask": [True, True, False, True, False, True, False],
            "seat": self.learner_seat,
            "winner": self.learner_seat,
            "terminated": True,
            "truncated": False,
        }

    def drain_trace(self):
        from ml_lab.tactical_trace import EpisodeTrace

        self.events.append(("drain_trace", None))
        return EpisodeTrace.from_payload(
            _terminal_trace(winner=self.learner_seat)
        )

    def save_replay(self, path: Path) -> Path:
        self.events.append(("save_replay", path))
        if self.write_replay:
            path.write_text("HEXWARS-REPLAY 1\n", encoding="utf-8")
        return path

    def drain_dagger(self) -> list[dict[str, Any]]:
        self.events.append(("drain_dagger", None))
        if self.duplicate_states:
            duplicate = HASHES["e"]
            return [
                _engine_dagger_row(
                    game_index=self.game_index, decision_index=index,
                    seat=self.learner_seat, teacher_action=3,
                    expansion_count=17, state_hash=duplicate,
                )
                for index in range(2)
            ]
        teacher_action = 1 if self.game_index % 2 == 0 else 3
        expansion_count = 5 if teacher_action == 1 else 10
        row_count = (
            self.rows_by_game[self.game_index]
            if self.rows_by_game is not None
            else self.rows_per_game
        )
        return [
            _engine_dagger_row(
                game_index=self.game_index, decision_index=index,
                seat=self.learner_seat, teacher_action=teacher_action,
                expansion_count=expansion_count,
            )
            for index in range(row_count)
        ]

    def close(self) -> None:
        self.close_attempts += 1
        self.events.append(("close", None))


class _RuntimeFailureClient(_CollectionClient):
    def __init__(
        self,
        contract: EnvironmentContract,
        *,
        fail_configure_at: int | None = None,
        fail_drain_at: int | None = None,
        close_error: bool = False,
        rows_per_game: int = 0,
    ) -> None:
        super().__init__(contract, rows_per_game=rows_per_game)
        self.fail_configure_at = fail_configure_at
        self.fail_drain_at = fail_drain_at
        self.close_error = close_error
        self.configure_calls = 0
        self.drain_calls = 0

    def configure_dagger(self, **kwargs: Any) -> None:
        call = self.configure_calls
        self.configure_calls += 1
        if call == self.fail_configure_at:
            raise RuntimeError(f"configure failed at call {call}")
        super().configure_dagger(**kwargs)

    def drain_dagger(self) -> list[dict[str, Any]]:
        call = self.drain_calls
        self.drain_calls += 1
        if call == self.fail_drain_at:
            raise RuntimeError(f"drain failed at call {call}")
        return super().drain_dagger()

    def close(self) -> None:
        self.close_attempts += 1
        self.events.append(("close", None))
        if self.close_error:
            raise RuntimeError("close failed")


def _progress_payloads(lines: list[str]) -> list[dict[str, Any]]:
    assert all(line.endswith("\n") for line in lines)
    return [json.loads(line) for line in lines]


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_failed_collection_stage(
    *,
    destination: Path,
    definition: CollectionDefinition,
    learner: ResolvedController,
    oracle: OracleSpec,
    expected_type: str,
    expected_message: str,
    expected_last_pair: int | None,
    expected_file_count: int,
) -> dict[str, Any]:
    staging = destination.with_name(destination.name + ".staging")
    assert not destination.exists()
    assert not (staging / "manifest.json").exists()
    diagnostic = json.loads((staging / "diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["exception"] == {
        "type": expected_type,
        "message": expected_message,
    }
    assert diagnostic["last_complete_pair"] == expected_last_pair
    actual_files = sorted(
        path.relative_to(staging).as_posix()
        for path in staging.rglob("*")
        if path.is_file()
    )
    assert diagnostic["physical_files"] == actual_files
    assert len(actual_files) == expected_file_count
    before = _tree_bytes(staging)
    with pytest.raises(ValueError, match="staging"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("failed staging opened a runtime")
            ),
            progress=lambda _line: None,
        )
    assert _tree_bytes(staging) == before
    return diagnostic


def test_collection_schedule_is_frozen_reciprocal_residual_and_seed_bound(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A yield-driven or independently rounded schedule would change this literal prefix."""

    definition, _, _ = _collection_inputs(tmp_path, contract)
    assert isinstance(definition.schedule[0], ScheduledDuel)
    assert len(definition.schedule) == 2_000
    assert [duel.profile for duel in definition.schedule[::2][:10]] == [
        "conversion-3v1-near",
        "standard-3v3",
        "standard-3v3",
        "conversion-3v1-far",
        "standard-3v3",
        "standard-3v3",
        "conversion-2v1-near",
        "standard-3v3",
        "standard-3v3",
        "standard-3v3",
    ]
    conversion_cycle = [
        "conversion-3v1-near",
        "conversion-3v1-far",
        "conversion-2v1-near",
        "conversion-2v1-far",
        "conversion-1v1-near",
        "conversion-1v1-far",
    ]
    scheduled_conversions = [
        duel.profile
        for duel in definition.schedule[::2]
        if duel.profile != "standard-3v3"
    ]
    assert scheduled_conversions[:7] == [*conversion_cycle, conversion_cycle[0]]
    for offset in range(0, len(definition.schedule), 20):
        profiles = [duel.profile for duel in definition.schedule[offset:offset + 20:2]]
        assert profiles.count("standard-3v3") == 7
        assert sum(profile != "standard-3v3" for profile in profiles) == 3
    for pair_index in range(1_000):
        first, second = definition.schedule[2 * pair_index:2 * pair_index + 2]
        assert (first.learner_seat, second.learner_seat) == (0, 1)
        assert (first.reference_seat, second.reference_seat) == (0, 1)
        assert first.map_seed == second.map_seed == 18_000_000 + pair_index
        assert first.episode_seed == second.episode_seed == first.map_seed
        assert first.schedule_index == second.schedule_index == pair_index
        assert first.profile == second.profile
    with pytest.raises(FrozenInstanceError):
        definition.schedule[0].profile = "standard-3v3"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        definition.partition = "validation"  # type: ignore[misc]
    with pytest.raises(ValueError, match="seats"):
        ScheduledDuel(
            schedule_index=0,
            map_seed=18_000_000,
            episode_seed=18_000_000,
            profile="standard-3v3",
            reference_seat=True,  # type: ignore[arg-type]
            learner_seat=1,
        )


def test_train_collection_runs_both_seats_executes_only_learner_and_reports_progress(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Executing the teacher action or dropping agreements changes physical rows and steps."""

    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    model_events: list[tuple[str, Any]] = []
    object.__setattr__(learner, "model", _DeterministicModel(model_events))
    client = _CollectionClient(contract, rows_per_game=10_000)
    lines: list[str] = []
    destination = tmp_path / "train-overlay"

    overlay = collect_selective_dagger(
        definition=definition,
        learner=learner,
        oracle=oracle,
        output_root=destination,
        client_factory=lambda: client,
        progress=lines.append,
    )

    assert overlay.row_count == 20_000
    assert [game.learner_seat for game in overlay.games] == [0, 1]
    assert [event for event in client.events if event[0] == "step"] == [
        ("step", 1), ("step", 1),
    ]
    assert [event[1]["deterministic"] for event in model_events] == [True, True]
    assert sum(
        descriptor.row_count for descriptor in overlay.manifest.games
    ) == 20_000
    with np.load(
        destination / json.loads((destination / "games/game-00000000.json").read_text(
            encoding="utf-8"
        ))["shard"]["path"],
        allow_pickle=False,
    ) as first_shard:
        assert np.count_nonzero(
            first_shard["actions"] == first_shard["learner_actions"]
        ) == 10_000
    names = [name for name, _ in client.events]
    assert names == [
        "configure_dagger", "enable_trace", "reset", "step", "drain_trace",
        "save_replay", "drain_dagger",
        "configure_dagger", "enable_trace", "reset", "step", "drain_trace",
        "save_replay", "drain_dagger", "close",
    ]
    resets = [payload for name, payload in client.events if name == "reset"]
    assert [(call["p0"], call["p1"], call["reference_seat"]) for call in resets] == [
        ("external", "random", 0),
        ("random", "external", 1),
    ]
    assert [call["learner"] for call in resets] == [0, 1]
    configured = [payload for name, payload in client.events if name == "configure_dagger"]
    assert configured == [{
        "enabled": True, "depth": 4, "expansion_budget": 512,
        "use_heuristic": True,
    }] * 2

    progress = _progress_payloads(lines)
    assert [item["event"] for item in progress] == ["game", "game", "pair"]
    assert [item["games"] for item in progress] == [1, 2, 2]
    assert [item["labels"] for item in progress] == [10_000, 20_000, 20_000]
    assert [item["new_games"] for item in progress] == [1, 2, 2]
    assert progress[-1]["pair_complete"] is True
    assert progress[-1]["disagreements"] == 10_000
    assert progress[-1]["disagreement_reason_counts"] == {
        "conversion": 10_000,
        "favorable": 10_000,
        "cycle_warning": 10_000,
        "action_waste": 10_000,
    }
    assert progress[-1]["reason_counts"] == {
        "conversion": 20_000,
        "favorable": 20_000,
        "cycle_warning": 20_000,
        "action_waste": 20_000,
    }
    assert progress[-1]["mean_expansions"] == 7.5
    assert progress[-1]["max_expansions"] == 10
    assert set(progress[-1]) == {
        "event", "partition", "iteration", "games", "labels",
        "reason_counts", "disagreements", "disagreement_reason_counts",
        "mean_expansions",
        "max_expansions", "labels_per_second", "elapsed_seconds",
        "eta_seconds", "pair_index", "pair_complete", "new_games",
    }
    assert all(
        earlier["elapsed_seconds"] <= later["elapsed_seconds"]
        for earlier, later in zip(progress, progress[1:])
    )
    for field in ("games", "labels", "disagreements", "new_games"):
        assert [item[field] for item in progress] == sorted(
            item[field] for item in progress
        )
    for reason in (
        "conversion", "favorable", "cycle_warning", "action_waste",
    ):
        assert [item["reason_counts"][reason] for item in progress] == sorted(
            item["reason_counts"][reason] for item in progress
        )
        assert [
            item["disagreement_reason_counts"][reason] for item in progress
        ] == sorted(
            item["disagreement_reason_counts"][reason] for item in progress
        )
    assert progress[-1]["labels_per_second"] >= 0.0
    assert progress[-1]["eta_seconds"] == 0.0

    reuse_lines: list[str] = []
    reused = collect_selective_dagger(
        definition=definition,
        learner=learner,
        oracle=oracle,
        output_root=destination,
        client_factory=lambda: (_ for _ in ()).throw(
            AssertionError("runtime was opened during exact reuse")
        ),
        progress=reuse_lines.append,
    )
    assert reused.content_identity == overlay.content_identity
    assert _progress_payloads(reuse_lines)[0]["new_games"] == 0

    mismatched = CollectionDefinition.create(
        contract=contract,
        partition="train",
        iteration=1,
        oracle=oracle,
        learner=definition.learner,
        original_dataset=definition.original_dataset,
        scenario_hash=definition.scenario_hash,
        repository_hash=HASHES["9"],
        panel_hash=definition.panel_hash,
    )
    with pytest.raises(ValueError, match="existing collection destination"):
        collect_selective_dagger(
            definition=mismatched,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("identity mismatch opened a runtime")
            ),
            progress=lambda _line: None,
        )

    staging = destination.with_name(destination.name + ".staging")
    for staging_kind in ("partial", "stale", "sealed"):
        if staging_kind == "partial":
            staging.mkdir()
            (staging / "junk.txt").write_text("partial\n", encoding="utf-8")
        else:
            shutil.copytree(destination, staging)
            if staging_kind == "stale":
                manifest_path = staging / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["repository_hash"] = HASHES["8"]
                manifest["content_identity"] = _identity(manifest)
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
        destination_before = _tree_bytes(destination)
        staging_before = _tree_bytes(staging)
        with pytest.raises(ValueError, match="destination.*staging.*coexist"):
            collect_selective_dagger(
                definition=definition,
                learner=learner,
                oracle=oracle,
                output_root=destination,
                client_factory=lambda: (_ for _ in ()).throw(
                    AssertionError("coexisting stage opened a runtime")
                ),
                progress=lambda _line: None,
            )
        assert _tree_bytes(destination) == destination_before
        assert _tree_bytes(staging) == staging_before
        shutil.rmtree(staging)

    crashed_destination = tmp_path / "crashed-overlay"
    crashed_staging = tmp_path / "crashed-overlay.staging"
    destination.rename(crashed_staging)
    resumed_lines: list[str] = []
    resumed = collect_selective_dagger(
        definition=definition,
        learner=learner,
        oracle=oracle,
        output_root=crashed_destination,
        client_factory=lambda: (_ for _ in ()).throw(
            AssertionError("completed staging opened a runtime")
        ),
        progress=resumed_lines.append,
    )
    assert resumed.root == crashed_destination.resolve()
    assert not crashed_staging.exists()
    assert _progress_payloads(resumed_lines)[0]["new_games"] == 0


def test_validation_collection_stops_at_target_only_after_complete_pair(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Stopping at the first 1,000-label game would publish an incomplete reciprocal pair."""

    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition="validation",
    )
    client = _CollectionClient(contract, rows_per_game=1_000)
    overlay = collect_selective_dagger(
        definition=definition,
        learner=learner,
        oracle=oracle,
        output_root=tmp_path / "validation-overlay",
        client_factory=lambda: client,
        progress=lambda _line: None,
    )
    assert overlay.row_count == 2_000
    assert len(overlay.games) == 2
    assert {game.map_seed for game in overlay.games} == {19_000_000}


def test_validation_collection_finishes_seat_one_when_seat_zero_crosses_target(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """The target crossing in seat 0 cannot publish a half reciprocal pair."""

    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition="validation",
    )
    client = _CollectionClient(contract, rows_by_game=(2_001, 0))
    overlay = collect_selective_dagger(
        definition=definition,
        learner=learner,
        oracle=oracle,
        output_root=tmp_path / "seat-zero-crossing-overlay",
        client_factory=lambda: client,
        progress=lambda _line: None,
    )
    assert overlay.row_count == 2_001
    assert [game.learner_seat for game in overlay.games] == [0, 1]
    assert [game.row_count for game in overlay.manifest.games] == [2_001, 0]
    assert sum(name == "reset" for name, _ in client.events) == 2
    assert sum(name == "drain_dagger" for name, _ in client.events) == 2


def _delay_collection_target_until_second_pair(
    real_update: Any,
) -> Any:
    def update(stats: dict[str, Any], rows: list[DaggerRow]) -> None:
        real_update(stats, rows)
        if stats["games"] <= 2:
            stats["labels"] = 0

    return update


def test_collection_rejects_newly_sealed_output_that_passed_target_one_pair_early(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Publication must independently verify the deterministic first crossing pair."""

    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition="validation",
    )
    client = _CollectionClient(contract, rows_per_game=1_000)
    destination = tmp_path / "newly-sealed-overrun"
    monkeypatch.setattr(
        dagger_module,
        "_update_collection_stats",
        _delay_collection_target_until_second_pair(
            dagger_module._update_collection_stats,
        ),
    )
    with pytest.raises(ValueError, match="first.*pair|pair.*boundary"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    staging = destination.with_name(destination.name + ".staging")
    assert not destination.exists()
    assert not (staging / "manifest.json").exists()
    diagnostic = json.loads((staging / "diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["exception"]["type"] == "ValueError"
    assert diagnostic["last_complete_pair"] == 1


def test_collection_reuse_rejects_rehashed_extra_pair_destination_and_staging(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A physically valid extra pair is corruption even when all hashes are consistent."""

    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition="validation",
    )
    corrupt_destination = tmp_path / "extra-pair-destination"
    with monkeypatch.context() as patch:
        patch.setattr(
            dagger_module,
            "_update_collection_stats",
            _delay_collection_target_until_second_pair(
                dagger_module._update_collection_stats,
            ),
        )
        patch.setattr(
            dagger_module,
            "_require_collection_overlay",
            lambda _overlay, _definition: None,
        )
        corrupt = collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=corrupt_destination,
            client_factory=lambda: _CollectionClient(contract, rows_per_game=1_000),
            progress=lambda _line: None,
        )
    assert len(corrupt.games) == 4
    destination_before = _tree_bytes(corrupt_destination)
    with pytest.raises(ValueError, match="existing collection destination"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=corrupt_destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("extra-pair destination opened a runtime")
            ),
            progress=lambda _line: None,
        )
    assert _tree_bytes(corrupt_destination) == destination_before

    staging_destination = tmp_path / "extra-pair-staging"
    staging = staging_destination.with_name(staging_destination.name + ".staging")
    shutil.copytree(corrupt_destination, staging)
    staging_before = _tree_bytes(staging)
    with pytest.raises(ValueError, match="staging"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=staging_destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("extra-pair staging opened a runtime")
            ),
            progress=lambda _line: None,
        )
    assert not staging_destination.exists()
    assert _tree_bytes(staging) == staging_before


def test_collection_reset_rpc_sets_authoritative_observer_learner_seat(
    contract: EnvironmentContract,
) -> None:
    """Controller placement alone cannot replace the engine learner-seat request."""

    client = object.__new__(dagger_module.DuelClient)
    client.contract = contract
    requests: list[dict[str, Any]] = []

    def rpc(request: dict[str, Any]) -> dict[str, Any]:
        requests.append(request)
        return {
            "obs": [0.25, 0.5],
            "mask": [True, True, False, True, False, True, False],
            "reward": 0.0,
            "terminated": False,
            "truncated": False,
            "seat": 1,
            "winner": -1,
            "start_profile": "conversion-3v1-near",
            "reference_seat": 1,
        }

    client._rpc = rpc  # type: ignore[method-assign]
    state = dagger_module._reset_collection_duel(
        client,
        seed=18_000_000,
        p0="random",
        p1="external",
        learner=1,
        start_profile="conversion-3v1-near",
        reference_seat=1,
    )
    assert state["seat"] == 1
    assert requests == [{
        "cmd": "duel_reset",
        "seed": 18_000_000,
        "p0": "random",
        "p1": "external",
        "learner": 1,
        "start_profile": "conversion-3v1-near",
        "reference_seat": 1,
    }]


@pytest.mark.parametrize(
    ("partition", "expected_games"),
    [("train", 2_000), ("validation", 200)],
)
def test_collection_fails_at_exact_game_ceiling_when_target_is_unmet(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
    partition: str,
    expected_games: int,
) -> None:
    """An off-by-one ceiling or early zero-yield abort changes the frozen experiment."""

    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition=partition,
    )
    client = _CollectionClient(
        contract, terminal_on_reset=True, write_replay=False,
    )

    class NullWriter:
        def __init__(self, root: Path) -> None:
            self.root = root

        def append_game(self, _game: DaggerGame, _rows: list[DaggerRow]) -> None:
            return None

        def seal(self) -> None:
            raise AssertionError("an unmet target was sealed")

    def create_writer(root: Path, **_kwargs: Any) -> NullWriter:
        root.mkdir(parents=True)
        return NullWriter(root)

    real_atomic_write = dagger_module.atomic_write_json

    def diagnostic_only(path: Path, payload: Any) -> None:
        if path.name == "diagnostic.json":
            real_atomic_write(path, payload)

    monkeypatch.setattr(dagger_module.DaggerOverlayWriter, "create", create_writer)
    monkeypatch.setattr(dagger_module, "atomic_write_json", diagnostic_only)
    with pytest.raises(RuntimeError, match="label target.*game ceiling"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=tmp_path / f"{partition}-ceiling",
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert sum(name == "reset" for name, _ in client.events) == expected_games
    assert sum(name == "drain_dagger" for name, _ in client.events) == expected_games


def test_collection_rejects_duplicate_episode_hash_and_retains_failure_diagnostics(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Deduplicating silently would hide invalid engine evidence and make the stage reusable."""

    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    client = _CollectionClient(contract, duplicate_states=True)
    destination = tmp_path / "duplicate-overlay"
    staging = tmp_path / "duplicate-overlay.staging"
    with pytest.raises(ValueError, match="duplicate canonical state hash"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert not destination.exists()
    assert not (staging / "manifest.json").exists()
    diagnostic = json.loads((staging / "diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["status"] == "failed"
    assert diagnostic["exception"]["type"] == "ValueError"
    assert diagnostic["last_complete_pair"] is None
    assert sorted(diagnostic["physical_files"]) == [
        "diagnostic.json",
        "evidence/game-00000000.replay",
        "evidence/game-00000000.trace.json",
    ]
    assert sum(name == "drain_dagger" for name, _ in client.events) == 1
    before = (staging / "diagnostic.json").read_bytes()
    with pytest.raises(ValueError, match="staging"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("partial staging opened a runtime")
            ),
            progress=lambda _line: None,
        )
    assert (staging / "diagnostic.json").read_bytes() == before


def test_collection_failure_after_second_game_records_last_complete_pair(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """A game-progress failure after seat 1 must not hide the physical complete pair."""

    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    client = _CollectionClient(contract, rows_per_game=1)
    destination = tmp_path / "progress-failure-overlay"

    def fail_after_second_game(line: str) -> None:
        payload = json.loads(line)
        if payload["event"] == "game" and payload["games"] == 2:
            raise RuntimeError("progress sink failed")

    with pytest.raises(RuntimeError, match="progress sink failed"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=fail_after_second_game,
        )
    diagnostic = json.loads(
        (tmp_path / "progress-failure-overlay.staging" / "diagnostic.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostic["games"] == 2
    assert diagnostic["last_complete_pair"] == 0
    assert not (
        tmp_path / "progress-failure-overlay.staging" / "manifest.json"
    ).exists()


def test_collection_writer_creation_failure_retains_diagnostic_staging(
    tmp_path: Path,
    contract: EnvironmentContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A post-directory writer failure must not leave an unexplained partial stage."""

    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    original_create = DaggerOverlayWriter.create

    def fail_after_creation(
        cls: type[DaggerOverlayWriter], root: Path, **kwargs: Any,
    ) -> DaggerOverlayWriter:
        original_create(root, **kwargs)
        raise RuntimeError("writer creation failed")

    monkeypatch.setattr(
        DaggerOverlayWriter, "create", classmethod(fail_after_creation),
    )
    destination = tmp_path / "writer-failure-overlay"
    with pytest.raises(RuntimeError, match="writer creation failed"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: (_ for _ in ()).throw(
                AssertionError("writer failure opened a runtime")
            ),
            progress=lambda _line: None,
        )
    staging = tmp_path / "writer-failure-overlay.staging"
    diagnostic = json.loads((staging / "diagnostic.json").read_text(encoding="utf-8"))
    assert diagnostic["exception"] == {
        "type": "RuntimeError", "message": "writer creation failed",
    }
    assert diagnostic["physical_files"] == ["diagnostic.json"]
    assert not (staging / "manifest.json").exists()


def test_collection_client_factory_failure_is_diagnostic_and_nonreusable(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    destination = tmp_path / "client-factory-failure"

    def fail_factory() -> _CollectionClient:
        raise RuntimeError("client factory failed")

    with pytest.raises(RuntimeError, match="client factory failed"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=fail_factory,
            progress=lambda _line: None,
        )
    _assert_failed_collection_stage(
        destination=destination,
        definition=definition,
        learner=learner,
        oracle=oracle,
        expected_type="RuntimeError",
        expected_message="client factory failed",
        expected_last_pair=None,
        expected_file_count=1,
    )


def test_collection_client_contract_failure_closes_once_and_is_nonreusable(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    client = _CollectionClient(replace(contract, contract_hash=HASHES["f"]))
    destination = tmp_path / "client-contract-failure"
    with pytest.raises(ValueError, match="runtime contract identity"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert client.close_attempts == 1
    _assert_failed_collection_stage(
        destination=destination,
        definition=definition,
        learner=learner,
        oracle=oracle,
        expected_type="ValueError",
        expected_message="collection runtime contract identity changed",
        expected_last_pair=None,
        expected_file_count=1,
    )


def test_collection_runtime_failure_before_pair_preserves_error_when_close_fails(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    client = _RuntimeFailureClient(
        contract, fail_configure_at=0, close_error=True,
    )
    destination = tmp_path / "runtime-before-pair-failure"
    with pytest.raises(RuntimeError, match="configure failed at call 0"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert client.close_attempts == 1
    _assert_failed_collection_stage(
        destination=destination,
        definition=definition,
        learner=learner,
        oracle=oracle,
        expected_type="RuntimeError",
        expected_message="configure failed at call 0",
        expected_last_pair=None,
        expected_file_count=1,
    )


def test_collection_runtime_failure_after_pair_records_inventory_and_closes_once(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    client = _RuntimeFailureClient(
        contract, fail_drain_at=2, rows_per_game=1,
    )
    destination = tmp_path / "runtime-after-pair-failure"
    with pytest.raises(RuntimeError, match="drain failed at call 2"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert client.close_attempts == 1
    diagnostic = _assert_failed_collection_stage(
        destination=destination,
        definition=definition,
        learner=learner,
        oracle=oracle,
        expected_type="RuntimeError",
        expected_message="drain failed at call 2",
        expected_last_pair=0,
        expected_file_count=11,
    )
    assert diagnostic["games"] == 2
    assert diagnostic["labels"] == 2


def test_collection_close_failure_is_attempted_once_and_cannot_leave_sealed_stage(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    definition, learner, oracle = _collection_inputs(
        tmp_path, contract, partition="validation",
    )
    client = _RuntimeFailureClient(
        contract, close_error=True, rows_per_game=1_000,
    )
    destination = tmp_path / "client-close-failure"
    with pytest.raises(RuntimeError, match="close failed"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=destination,
            client_factory=lambda: client,
            progress=lambda _line: None,
        )
    assert client.close_attempts == 1
    _assert_failed_collection_stage(
        destination=destination,
        definition=definition,
        learner=learner,
        oracle=oracle,
        expected_type="RuntimeError",
        expected_message="close failed",
        expected_last_pair=0,
        expected_file_count=9,
    )


def test_collection_reuse_and_invalid_schedule_fail_before_runtime_creation(
    tmp_path: Path, contract: EnvironmentContract,
) -> None:
    """Trusting a partial destination or a mutated schedule would launch or append games."""

    definition, learner, oracle = _collection_inputs(tmp_path, contract)
    partial = tmp_path / "partial-destination"
    partial.mkdir()
    (partial / "junk.txt").write_text("partial\n", encoding="utf-8")
    calls = 0

    def factory() -> _CollectionClient:
        nonlocal calls
        calls += 1
        return _CollectionClient(contract)

    with pytest.raises(ValueError, match="existing collection destination"):
        collect_selective_dagger(
            definition=definition,
            learner=learner,
            oracle=oracle,
            output_root=partial,
            client_factory=factory,
            progress=lambda _line: None,
        )
    assert calls == 0

    corrupted, other_learner, other_oracle = _collection_inputs(
        tmp_path / "corrupted", contract,
    )
    object.__setattr__(corrupted, "schedule", corrupted.schedule[1:])
    with pytest.raises(ValueError, match="collection schedule"):
        collect_selective_dagger(
            definition=corrupted,
            learner=other_learner,
            oracle=other_oracle,
            output_root=tmp_path / "corrupt-schedule-output",
            client_factory=factory,
            progress=lambda _line: None,
        )
    assert calls == 0


def test_iteration_one_warm_start_is_the_exact_immutable_seed_227_snapshot() -> None:
    """Changing the first warm-start silently changes the entire causal experiment."""

    source = dagger_module.dagger_actor_source(1)
    expected = {
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
    assert source.to_dict() == expected
    checkpoint = Path(expected["controller"]["path"])
    assert checkpoint.is_file()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected[
        "checkpoint_sha256"
    ]
    with pytest.raises(TypeError):
        source.controller["step"] = 1


def _write_published_dagger_actor(root: Path, iteration: int) -> Path:
    checkpoint = root / "checkpoints" / "step_000000000.zip"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(f"dagger-actor-{iteration}".encode("ascii"))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    actor_initialization = {
        "schema_version": 1,
        "kind": "actor_only",
        "source_kind": "snapshot" if iteration == 1 else "dagger_actor",
        "source_checkpoint_sha256": HASHES["e"],
    }
    publication_verification = {
        "checkpoint_sha256": digest,
        "actor_sha256": HASHES["a"],
    }
    run = {
        "schema_version": 1,
        "state": "completed",
        "latest_checkpoint": "checkpoints/step_000000000.zip",
        "latest_checkpoint_step": 0,
        "checkpoint_sha256": digest,
        "training_kind": "selective-dagger-distillation-v1",
        "distillation_iteration": iteration,
        "target_actor_sha256_final": HASHES["a"],
        "actor_initialization": actor_initialization,
        "publication_verification": publication_verification,
        "production": True,
        "config": {
            "algorithm": "maskable_ppo",
            "policy": "HexCNN",
            "device": "cpu",
        },
    }
    bc = {
        "schema_version": 1,
        "training_kind": "selective-dagger-distillation-v1",
        "distillation_iteration": iteration,
        "algorithm": "maskable_ppo",
        "policy": "HexCNN",
        "checkpoint_sha256": digest,
        "target_actor_sha256_final": HASHES["a"],
        "actor_initialization": actor_initialization,
        "publication_verification": publication_verification,
        "production": True,
    }
    (root / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (root / "bc.json").write_text(json.dumps(bc), encoding="utf-8")
    return checkpoint


def test_later_actor_source_binds_the_preceding_completed_publication_and_sha(
    tmp_path: Path,
) -> None:
    """Resolving a live run or the wrong prior iteration would break the DAgger chain."""

    run = tmp_path / "iteration-1-actor"
    checkpoint = _write_published_dagger_actor(run, 1)

    source = dagger_module.dagger_actor_source(2, preceding_run=run)

    assert source.source_kind == "dagger_actor"
    assert source.controller == {
        "kind": "snapshot",
        "path": str(checkpoint.resolve()),
        "source_run": str(run.resolve()),
        "algorithm": "maskable_ppo",
        "step": 0,
        "inference_mode": "deterministic",
    }
    assert source.checkpoint_sha256 == hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    assert source.published_actor_sha256 == HASHES["a"]

    checkpoint.write_bytes(b"replaced")
    with pytest.raises(ValueError, match="SHA-256"):
        dagger_module.dagger_actor_source(2, preceding_run=run)
    with pytest.raises(ValueError, match="preceding"):
        dagger_module.dagger_actor_source(3, preceding_run=run)


def test_later_actor_source_rejects_checkpoint_directory_symlink_escape_when_supported(
    tmp_path: Path,
) -> None:
    """Resolving checkpoints must not move a publication outside its source run."""

    run = tmp_path / "iteration-1-actor"
    checkpoint = _write_published_dagger_actor(run, 1)
    outside = tmp_path / "outside-checkpoints"
    checkpoint.parent.rename(outside)
    _symlink_or_skip_windows_privilege(
        run / "checkpoints", outside, target_is_directory=True,
    )

    with pytest.raises(ValueError, match="contained"):
        dagger_module.dagger_actor_source(2, preceding_run=run)


def test_later_actor_source_rejects_checkpoint_file_symlink_escape_when_supported(
    tmp_path: Path,
) -> None:
    """A checkpoint file link cannot escape the preceding publication root."""

    run = tmp_path / "iteration-1-actor"
    checkpoint = _write_published_dagger_actor(run, 1)
    outside = tmp_path / "outside-checkpoint.zip"
    checkpoint.replace(outside)
    _symlink_or_skip_windows_privilege(checkpoint, outside)

    with pytest.raises(ValueError, match="contained"):
        dagger_module.dagger_actor_source(2, preceding_run=run)


def test_later_actor_source_rejects_checkpoint_directory_junction_escape(
    tmp_path: Path,
) -> None:
    """A Windows junction cannot move the canonical checkpoint outside its run."""

    run = tmp_path / "iteration-1-actor"
    checkpoint = _write_published_dagger_actor(run, 1)
    outside = tmp_path / "outside-junction-checkpoints"
    checkpoint.parent.rename(outside)
    _windows_directory_junction(run / "checkpoints", outside)

    with pytest.raises(ValueError, match="contained"):
        dagger_module.dagger_actor_source(2, preceding_run=run)


def test_later_actor_source_rejects_ordinary_checkpoint_escape(
    tmp_path: Path,
) -> None:
    """A lexical parent escape is rejected independently of symlink support."""

    run = tmp_path / "iteration-1-actor"
    checkpoint = _write_published_dagger_actor(run, 1)
    outside = tmp_path / "outside-checkpoints"
    checkpoint.parent.rename(outside)
    manifest_path = run / "run.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["latest_checkpoint"] = (
        "../outside-checkpoints/step_000000000.zip"
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="contained"):
        dagger_module.dagger_actor_source(2, preceding_run=run)


@pytest.mark.parametrize(
    "mutation",
    [
        "run_checkpoint_hash",
        "bc_checkpoint_hash",
        "run_actor_hash",
        "bc_actor_hash",
        "run_iteration",
        "bc_iteration",
        "run_training_kind",
        "bc_training_kind",
        "run_source_identity",
        "bc_source_identity",
        "run_publication_verification",
        "bc_publication_verification",
        "run_production",
        "bc_production",
    ],
)
def test_later_actor_source_authenticates_matching_completed_manifests(
    tmp_path: Path, mutation: str,
) -> None:
    """The actor source object must carry only provenance agreed by both manifests."""

    run = tmp_path / "iteration-1-actor"
    _write_published_dagger_actor(run, 1)
    run_path = run / "run.json"
    bc_path = run / "bc.json"
    manifest = json.loads(run_path.read_text(encoding="utf-8"))
    bc = json.loads(bc_path.read_text(encoding="utf-8"))
    if mutation == "run_checkpoint_hash":
        manifest["checkpoint_sha256"] = HASHES["f"]
    elif mutation == "bc_checkpoint_hash":
        bc["checkpoint_sha256"] = HASHES["f"]
    elif mutation == "run_actor_hash":
        manifest["target_actor_sha256_final"] = HASHES["f"]
    elif mutation == "bc_actor_hash":
        bc["target_actor_sha256_final"] = HASHES["f"]
    elif mutation == "run_iteration":
        manifest["distillation_iteration"] = 2
    elif mutation == "bc_iteration":
        bc["distillation_iteration"] = 2
    elif mutation == "run_training_kind":
        manifest["training_kind"] = "forged"
    elif mutation == "bc_training_kind":
        bc["training_kind"] = "forged"
    elif mutation == "run_source_identity":
        manifest["actor_initialization"]["source_kind"] = "dagger_actor"
    elif mutation == "bc_source_identity":
        bc["actor_initialization"]["source_kind"] = "dagger_actor"
    elif mutation == "run_publication_verification":
        manifest["publication_verification"]["actor_sha256"] = HASHES["f"]
    elif mutation == "bc_publication_verification":
        bc["publication_verification"]["actor_sha256"] = HASHES["f"]
    elif mutation == "run_production":
        manifest["production"] = False
    elif mutation == "bc_production":
        bc["production"] = False
    else:
        raise AssertionError(mutation)
    run_path.write_text(json.dumps(manifest), encoding="utf-8")
    bc_path.write_text(json.dumps(bc), encoding="utf-8")

    with pytest.raises(ValueError, match="provenance|SHA-256"):
        dagger_module.dagger_actor_source(2, preceding_run=run)


def test_production_dagger_distillation_configuration_is_locked() -> None:
    """A caller-supplied production optimizer would invalidate cross-iteration results."""

    assert asdict(dagger_module.DAGGER_DISTILLATION_CONFIG) == {
        "model_seed": 227,
        "batch_size": 256,
        "learning_rate": 3e-4,
        "max_epochs": 50,
        "patience": 5,
        "device": "cuda",
    }


def _development_candidate_payload(
    checkpoint: Path,
    *,
    iteration: int,
) -> dict[str, Any]:
    candidate_id = "baseline" if iteration == 0 else f"iteration-{iteration}"
    controller = {
        "kind": "snapshot",
        "path": str(checkpoint.resolve()),
        "source_run": str(checkpoint.parent.parent.resolve()),
        "algorithm": "maskable_ppo",
        "step": 38_912 if iteration == 0 else 0,
        "inference_mode": "deterministic",
    }
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    controller_identity = {
        "kind": "snapshot",
        "inference_mode": "deterministic",
        "path": str(checkpoint.resolve()),
        "algorithm": "maskable_ppo",
        "step": controller["step"],
        "contract_hash": HASHES["a"],
        "contract_version": "tactical-v2",
        "environment": "tactical-v2",
        "encoding_hash": HASHES["b"],
        "contract": {"version": "tactical-v2"},
        "observation_size": 2,
        "action_size": 7,
        "legacy": False,
        "promotable": True,
    }
    source_publication = {
        "kind": "audited-baseline" if iteration == 0 else "dagger-iteration",
        "iteration": iteration,
        "content_identity": hashlib.sha256(
            f"source-{iteration}".encode("ascii")
        ).hexdigest(),
        "preflight_root": str((checkpoint.parents[2] / "preflight").resolve()),
        "preflight_content_identity": hashlib.sha256(b"preflight").hexdigest(),
        "incoming_source_content_identity": (
            None if iteration == 0 else hashlib.sha256(
                f"source-{iteration - 1}".encode("ascii")
            ).hexdigest()
        ),
        "source_run": controller["source_run"],
        "model_seed": 227,
        "step": controller["step"],
        "controller": json.dumps(controller, sort_keys=True),
        "controller_identity": controller_identity,
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "actor_sha256": hashlib.sha256(f"actor-{iteration}".encode()).hexdigest(),
        "publication_metadata_sha256": hashlib.sha256(
            f"metadata-{iteration}".encode()
        ).hexdigest(),
        "run_manifest_sha256": hashlib.sha256(f"run-{iteration}".encode()).hexdigest(),
        "bc_manifest_sha256": hashlib.sha256(f"bc-{iteration}".encode()).hexdigest(),
        "train_overlay_prefix": [
            hashlib.sha256(f"train-{index}".encode()).hexdigest()
            for index in range(1, iteration + 1)
        ],
        "validation_overlay_prefix": [
            hashlib.sha256(f"validation-{index}".encode()).hexdigest()
            for index in range(1, iteration + 1)
        ],
    }
    return {
        "candidate_id": candidate_id,
        "iteration": iteration,
        "controller": json.dumps(controller, sort_keys=True),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "controller_identity": controller_identity,
        "source_publication": source_publication,
    }


def _development_stable_identity(tmp_path: Path) -> dict[str, Any]:
    return {
        "definition": {
            "panel_sha256": HASHES["1"],
            "panel_byte_size": 101,
            "seed_banks_sha256": HASHES["3"],
            "seed_banks_byte_size": 202,
        },
        "repository": {
            "root": str(tmp_path.resolve()),
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "dirty": False,
        },
        "scenario": {
            "source_sha256": HASHES["4"],
            "runtime_sha256": HASHES["2"],
        },
        "contract": {
            "version": "tactical-v2",
            "contract_hash": HASHES["a"],
            "encoding_hash": HASHES["b"],
            "observation_size": 2,
            "action_size": 7,
            "action_regions": {
                "move": {"offset": 1, "count": 2},
                "attack": {"offset": 3, "count": 2},
                "deploy": {"offset": 5, "count": 2},
            },
        },
        "base_dataset": {
            "root": str((tmp_path / "base-dataset").resolve()),
            "manifest_sha256": HASHES["5"],
            "content_sha256": HASHES["6"],
            "file_count": 1,
            "byte_size": 1,
            "contract_hash": HASHES["a"],
            "encoding_hash": HASHES["b"],
            "scenario_hash": HASHES["7"],
        },
        "selected_oracle": {
            "spec": _oracle_payload(),
            "evidence_root": str((tmp_path / "preflight").resolve()),
            "evidence_content_identity": HASHES["8"],
            "evidence_class": "sealed-engine",
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
                "training_device": "cuda:0", "publication_device": "cpu",
                "cuda_available": True, "device_index": 0,
                "device_name": "test-gpu", "cuda_runtime": "12.8",
            },
            "software": {
                "python": "test", "implementation": "CPython",
                "platform": "test", "executable": "C:/python.exe",
                "numpy": "test", "torch": "test",
                "stable_baselines3": "test", "sb3_contrib": "test",
            },
        },
    }


def _development_definition(tmp_path: Path) -> Any:
    candidates = []
    for iteration in range(4):
        run = tmp_path / f"actor-{iteration}"
        checkpoint_dir = run / "checkpoints"
        checkpoint_dir.mkdir(parents=True)
        step = 38_912 if iteration == 0 else 0
        checkpoint = checkpoint_dir / f"step_{step:09d}.zip"
        checkpoint.write_bytes(f"actor-{iteration}".encode("ascii"))
        candidates.append(
            dagger_module.DevelopmentCandidate.from_dict(
                _development_candidate_payload(
                    checkpoint, iteration=iteration,
                )
            )
        )
    return dagger_module.DevelopmentEvaluationDefinition.create(
        candidates=candidates,
        panel_hash=HASHES["1"],
        scenario_hash=HASHES["2"],
        contract_hash=HASHES["a"],
        encoding_hash=HASHES["b"],
        repository={
            "root": str(tmp_path.resolve()),
            "commit": "a" * 40,
            "source_tree": "b" * 40,
            "dirty": False,
        },
        stable_iteration_identity=_development_stable_identity(tmp_path),
    )


def test_development_evaluation_definition_freezes_four_reciprocal_candidates(
    tmp_path: Path,
) -> None:
    """Wrong seed/profile/seat/model schedules must fail before any game launches."""

    definition = _development_definition(tmp_path)

    assert [candidate.candidate_id for candidate in definition.candidates] == [
        "baseline",
        "iteration-1",
        "iteration-2",
        "iteration-3",
    ]
    assert [candidate.iteration for candidate in definition.candidates] == [
        0, 1, 2, 3,
    ]
    assert len(definition.schedule) == 200
    assert [
        (
            game.schedule_index,
            game.map_seed,
            game.episode_seed,
            game.profile,
            game.reference_seat,
            game.learner_seat,
        )
        for game in definition.schedule
    ] == [
        (
            seed - 20_000_000,
            seed,
            seed,
            "standard-3v3",
            seat,
            seat,
        )
        for seed in range(20_000_000, 20_000_100)
        for seat in (0, 1)
    ]
    assert definition.opponent == "random"
    assert definition.inference_mode == "deterministic"
    assert definition.observer_enabled is False
    assert definition.search_enabled is False

    changed = list(definition.candidates)
    changed[2] = replace(
        changed[2], checkpoint_sha256=HASHES["f"],
    )
    with pytest.raises(ValueError, match="candidate|checkpoint|definition"):
        dagger_module.DevelopmentEvaluationDefinition.create(
            candidates=changed,
            panel_hash=definition.panel_hash,
            scenario_hash=definition.scenario_hash,
            contract_hash=definition.contract_hash,
            encoding_hash=definition.encoding_hash,
            repository=definition.repository,
            stable_iteration_identity=definition.stable_iteration_identity,
        )


def _development_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seat_outcomes = {
        0: ["win"] * 60 + ["loss"] * 10 + ["draw"] * 30,
        1: ["win"] * 40 + ["loss"] * 30 + ["draw"] * 30,
    }
    draw_index = 0
    for map_offset in range(100):
        for seat in (0, 1):
            outcome = seat_outcomes[seat][map_offset]
            classification = None
            if outcome == "draw":
                flags: list[str] = []
                if draw_index < 40:
                    flags.append("cycling")
                if draw_index < 20:
                    flags.append("action_waste")
                classification = {
                    "primary": flags[0] if flags else "unclassified",
                    "flags": flags,
                    "evidence": {},
                }
                draw_index += 1
            rows.append({
                "map_seed": 20_000_000 + map_offset,
                "candidate_seat": seat,
                "outcome": outcome,
                "summary": {
                    "command_count": 10,
                    "round_count": 5,
                    "peak_normalized_advantage": 0.4,
                    "final_normalized_advantage": 0.2,
                    "end_turns_by_seat": [2, 2],
                    "wasted_end_turns_by_seat": (
                        [1, 0] if seat == 0 else [0, 1]
                    ),
                },
                "classification": classification,
            })
    return rows


def test_development_metrics_are_typed_complete_and_cumulative_by_reason() -> None:
    """All Task 10 metrics must be derived from rows and held-out label counts."""

    metrics = dagger_module.summarize_development_candidate(
        _development_rows(),
        heldout_collections=(
            {
                "labels": 100,
                "reason_counts": {
                    "conversion": 50,
                    "favorable": 40,
                    "cycle_warning": 30,
                    "wasted_end_turn": 20,
                },
                "disagreements": 20,
                "disagreement_reason_counts": {
                    "conversion": 10,
                    "favorable": 8,
                    "cycle_warning": 6,
                    "wasted_end_turn": 4,
                },
            },
            {
                "labels": 100,
                "reason_counts": {
                    "conversion": 50,
                    "favorable": 40,
                    "cycle_warning": 30,
                    "wasted_end_turn": 20,
                },
                "disagreements": 10,
                "disagreement_reason_counts": {
                    "conversion": 5,
                    "favorable": 4,
                    "cycle_warning": 3,
                    "wasted_end_turn": 2,
                },
            },
        ),
    )

    assert metrics["counts"] == {
        "games": 200, "wins": 100, "losses": 40, "draws": 60,
    }
    assert metrics["rates"] == {"win": 0.5, "loss": 0.2, "draw": 0.3}
    assert metrics["confidence_intervals"]["win"] == pytest.approx({
        "low": 0.4313608596,
        "high": 0.5686391404,
        "confidence": 0.95,
    })
    assert metrics["seats"]["candidate_as_p0"]["rates"]["win"] == 0.6
    assert metrics["seats"]["candidate_as_p1"]["rates"]["win"] == 0.4
    assert metrics["win_rate_p0_minus_p1"] == pytest.approx(0.2)
    assert metrics["draw_diagnostics"]["cycling"] == {
        "count": 40, "incidence": 0.2,
    }
    assert metrics["draw_diagnostics"]["action_waste"] == {
        "count": 20, "incidence": 0.1,
    }
    assert metrics["candidate_end_turns"] == {
        "total": 400, "wasted": 200, "wasted_ratio": 0.5,
    }
    assert metrics["normalized_advantage"]["all_games"] == {
        "final": pytest.approx(0.2),
        "peak": pytest.approx(0.4),
    }
    assert metrics["winning_games"]["round_count"]["mean"] == 5
    assert metrics["winning_games"]["command_count"]["mean"] == 10
    assert metrics["heldout_teacher"] == {
        "labels": 200,
        "agreements": 170,
        "disagreements": 30,
        "accuracy": 0.85,
        "by_reason": {
            "conversion": {
                "labels": 100, "disagreements": 15, "accuracy": 0.85,
            },
            "favorable": {
                "labels": 80, "disagreements": 12, "accuracy": 0.85,
            },
            "cycle_warning": {
                "labels": 60, "disagreements": 9, "accuracy": 0.85,
            },
            "wasted_end_turn": {
                "labels": 40, "disagreements": 6, "accuracy": 0.85,
            },
        },
    }


def _selection_candidate(
    candidate_id: str,
    iteration: int,
    *,
    win_rate: float,
    cycling: float,
    action_waste: float,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "iteration": iteration,
        "summary": {
            "counts": {
                "games": 200,
                "wins": int(win_rate * 200),
                "losses": 0,
                "draws": 200 - int(win_rate * 200),
            },
            "rates": {
                "win": win_rate,
                "loss": 0.0,
                "draw": 1.0 - win_rate,
            },
            "draw_diagnostics": {
                "cycling": {
                    "count": int(cycling * 200),
                    "incidence": cycling,
                },
                "action_waste": {
                    "count": int(action_waste * 200),
                    "incidence": action_waste,
                },
            },
        },
    }


def test_development_selection_success_and_confirmation_are_separate() -> None:
    """Teacher accuracy cannot select a model or satisfy a game milestone."""

    candidates = (
        _selection_candidate(
            "baseline", 0, win_rate=0.40, cycling=0.60, action_waste=0.30,
        ),
        _selection_candidate(
            "iteration-1", 1, win_rate=0.60, cycling=0.30, action_waste=0.20,
        ),
        _selection_candidate(
            "iteration-2", 2, win_rate=0.60, cycling=0.20, action_waste=0.20,
        ),
        _selection_candidate(
            "iteration-3", 3, win_rate=0.60, cycling=0.20, action_waste=0.10,
        ),
    )

    selected = dagger_module.select_development_candidate(candidates)
    assert selected["candidate_id"] == "iteration-3"

    gain_only = dagger_module.decide_development_success(
        baseline=candidates[0], selected=selected,
    )
    assert gain_only == {
        "baseline_win_rate": 0.4,
        "selected_win_rate": 0.6,
        "win_rate_gain": pytest.approx(0.2),
        "baseline_cycling_incidence": 0.6,
        "selected_cycling_incidence": 0.2,
        "cycling_relative_reduction": pytest.approx(2 / 3),
        "performance_gate": True,
        "cycling_gate": True,
        "cycling_gate_reason": "threshold-met",
        "treatment_success": True,
        "replication_authorized": True,
        "one_seed_winning_rate_gate": False,
        "confirmation_required": True,
    }
    assert dagger_module.confirm_replication_milestone(
        replicate_win_rates=(0.65, 0.70, 0.75),
        pooled_wins=420,
        pooled_games=600,
    ) == {
        "replicate_win_rates": [0.65, 0.7, 0.75],
        "every_replicate_at_least_0_65": True,
        "pooled_win_rate": 0.7,
        "pooled_at_least_0_70": True,
        "confirmed": True,
    }
    assert dagger_module.confirm_replication_milestone(
        replicate_win_rates=(0.64, 0.73, 0.73),
        pooled_wins=420,
        pooled_games=600,
    )["confirmed"] is False


def test_development_selection_preserves_dominant_baseline_and_names_best_treatment() -> None:
    """A worse treatment can be analyzed without silently replacing the baseline."""

    candidates = (
        _selection_candidate(
            "baseline", 0, win_rate=0.70, cycling=0.10, action_waste=0.05,
        ),
        _selection_candidate(
            "iteration-1", 1, win_rate=0.60, cycling=0.05, action_waste=0.02,
        ),
        _selection_candidate(
            "iteration-2", 2, win_rate=0.65, cycling=0.08, action_waste=0.03,
        ),
        _selection_candidate(
            "iteration-3", 3, win_rate=0.65, cycling=0.08, action_waste=0.04,
        ),
    )

    assert dagger_module.select_development_candidate(candidates)["candidate_id"] == (
        "baseline"
    )
    assert dagger_module.select_development_treatment(candidates)["candidate_id"] == (
        "iteration-2"
    )
    decision = dagger_module.decide_development_success(
        baseline=candidates[0], selected=candidates[2],
    )
    assert decision["win_rate_gain"] == pytest.approx(-0.05)
    assert decision["treatment_success"] is False


def test_development_thresholds_use_exact_counts_and_zero_cycling_is_undefined() -> None:
    """Binary thresholds have no epsilon halo and no invented zero-denominator reduction."""

    baseline = _selection_candidate(
        "baseline", 0, win_rate=0.40, cycling=0.20, action_waste=0.10,
    )
    at_gain = _selection_candidate(
        "iteration-1", 1, win_rate=0.60, cycling=0.10, action_waste=0.10,
    )
    below_gain = _selection_candidate(
        "iteration-1", 1, win_rate=0.595, cycling=0.10, action_waste=0.10,
    )
    below_rate = _selection_candidate(
        "iteration-1", 1, win_rate=0.645, cycling=0.10, action_waste=0.10,
    )
    assert dagger_module.decide_development_success(
        baseline=baseline, selected=at_gain,
    )["performance_gate"] is True
    assert dagger_module.decide_development_success(
        baseline=baseline, selected=below_gain,
    )["performance_gate"] is False
    high_baseline = _selection_candidate(
        "baseline", 0, win_rate=0.50, cycling=0.20, action_waste=0.10,
    )
    assert dagger_module.decide_development_success(
        baseline=high_baseline, selected=below_rate,
    )["performance_gate"] is False

    zero_cycling = _selection_candidate(
        "baseline", 0, win_rate=0.40, cycling=0.0, action_waste=0.10,
    )
    zero_selected = _selection_candidate(
        "iteration-1", 1, win_rate=0.65, cycling=0.0, action_waste=0.10,
    )
    zero_decision = dagger_module.decide_development_success(
        baseline=zero_cycling, selected=zero_selected,
    )
    assert zero_decision["cycling_relative_reduction"] is None
    assert zero_decision["cycling_gate"] is False
    assert zero_decision["cycling_gate_reason"] == "no-reducible-baseline"


def test_development_pairing_and_metric_inputs_fail_closed_on_exact_types() -> None:
    """Missing reciprocals and scalar aliases must never enter selection."""

    rows = _development_rows()
    later = copy.deepcopy(rows)
    later[0]["outcome"] = "loss"
    later[81]["outcome"] = "win"
    change = dagger_module.compare_development_candidates(rows, later)
    assert change["left_only_wins"] == 1
    assert change["right_only_wins"] == 1
    assert change["net_win_change"] == 0
    assert change["exact_sign_test_p_value"] == 1.0

    with pytest.raises(ValueError, match="schedule|missing|games"):
        dagger_module.compare_development_candidates(rows[:-1], later)
    malformed = {
        "labels": True,
        "reason_counts": {
            "conversion": 0,
            "favorable": 0,
            "cycle_warning": 0,
            "wasted_end_turn": 0,
        },
        "disagreements": 0,
        "disagreement_reason_counts": {
            "conversion": 0,
            "favorable": 0,
            "cycle_warning": 0,
            "wasted_end_turn": 0,
        },
    }
    with pytest.raises(ValueError, match="labels|integer"):
        dagger_module.summarize_development_candidate(
            rows, heldout_collections=(malformed,),
        )


@pytest.mark.parametrize(
    "mutation",
    ("nan-advantage", "negative-decisions", "impossible-peak", "waste-over-total"),
)
def test_development_summary_rejects_impossible_physical_metrics(
    mutation: str,
) -> None:
    """Non-finite, negative, incoherent, or impossible trace metrics cannot aggregate."""

    rows = _development_rows()
    summary = rows[0]["summary"]
    if mutation == "nan-advantage":
        summary["final_normalized_advantage"] = float("nan")
    elif mutation == "negative-decisions":
        summary["command_count"] = -1
    elif mutation == "impossible-peak":
        summary["peak_normalized_advantage"] = 99.0
    elif mutation == "waste-over-total":
        summary["end_turns_by_seat"] = [1, 2]
        summary["wasted_end_turns_by_seat"] = [2, 0]
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match="metric|advantage|waste|summary"):
        dagger_module.summarize_development_candidate(
            rows, heldout_collections=(),
        )


def _development_supervised_metrics(
    *,
    pre_agreements: int,
    post_agreements: int,
    labels: int = 100,
) -> dict[str, Any]:
    def side(agreements: int) -> dict[str, Any]:
        return {
            "agreements": agreements,
            "disagreements": labels - agreements,
            "accuracy": agreements / labels,
            "by_reason": {
                reason: {
                    "labels": labels if reason == "conversion" else 0,
                    "agreements": agreements if reason == "conversion" else 0,
                    "disagreements": (
                        labels - agreements if reason == "conversion" else 0
                    ),
                    "accuracy": agreements / labels if reason == "conversion" else None,
                }
                for reason in (
                    "conversion", "favorable", "cycle_warning", "wasted_end_turn",
                )
            },
        }

    return {
        "labels": labels,
        "pre": side(pre_agreements),
        "post": side(post_agreements),
        "accuracy_change": post_agreements / labels - pre_agreements / labels,
    }


def test_development_aggregate_and_report_use_games_not_supervised_accuracy() -> None:
    """The canonical report must preserve the fixed causal interpretation branch."""

    baseline_rows = _development_rows()
    rows_by_candidate = {
        "baseline": baseline_rows,
        "iteration-1": copy.deepcopy(baseline_rows),
        "iteration-2": copy.deepcopy(baseline_rows),
        "iteration-3": copy.deepcopy(baseline_rows),
    }
    # Make iteration 3 the unambiguous game winner while leaving its supervised
    # accuracy deliberately lower than iteration 2.
    for index in range(20):
        rows_by_candidate["iteration-3"][140 + index]["outcome"] = "win"
        rows_by_candidate["iteration-3"][140 + index]["classification"] = None
    supervised = {
        1: _development_supervised_metrics(pre_agreements=80, post_agreements=80),
        2: _development_supervised_metrics(pre_agreements=80, post_agreements=99),
        3: _development_supervised_metrics(pre_agreements=60, post_agreements=70),
    }

    aggregate = dagger_module.build_development_aggregate(
        rows_by_candidate=rows_by_candidate,
        supervised_metrics_by_iteration=supervised,
        evidence_identity={
            "preflight": HASHES["1"],
            "baseline": HASHES["2"],
            "iterations": [HASHES["3"], HASHES["4"], HASHES["5"]],
            "evaluations": [HASHES["6"], HASHES["7"], HASHES["8"], HASHES["9"]],
            "supervised": [HASHES["a"], HASHES["b"], HASHES["c"]],
        },
    )

    assert aggregate["selected_candidate_id"] == "iteration-3"
    assert (
        aggregate["candidates"]["iteration-2"]["summary"]["heldout_supervised"][
            "accuracy"
        ]
        >
        aggregate["candidates"]["iteration-3"]["summary"]["heldout_supervised"][
            "accuracy"
        ]
    )
    assert aggregate["best_treatment_candidate_id"] == "iteration-3"
    assert aggregate["treatment_supervised_comparison"] == {
        "iteration": 3,
        "labels": 100,
        "pre_accuracy": 0.6,
        "post_accuracy": 0.7,
        "accuracy_change": pytest.approx(0.1),
    }
    assert set(aggregate["paired_vs_baseline"]) == {
        "iteration-1", "iteration-2", "iteration-3",
    }
    canonical = json.dumps(
        aggregate, sort_keys=True, separators=(",", ":"), allow_nan=False,
    )
    assert json.dumps(
        dagger_module.build_development_aggregate(
            rows_by_candidate=rows_by_candidate,
            supervised_metrics_by_iteration=supervised,
            evidence_identity=aggregate["evidence_identity"],
        ),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) == canonical

    report = dagger_module.render_development_report(aggregate)
    for required in (
        "Can high-confidence search supervision",
        "Frozen inputs",
        "Oracle selection",
        "Collection yields",
        "Training curves and timings",
        "Supervised metrics",
        "Game outcomes",
        "Paired tests",
        "Chosen candidate",
        "Threshold decisions",
        "supervised accuracy alone is not success",
        "final 17m seeds were not used",
        "three-replicate confirmation is a later experiment",
    ):
        assert required in report
    assert "| baseline | 100 | 80 | 20 | 0.800000 |" in report
    assert "| iteration-2 | 100 | 99 | 1 | 0.990000 |" in report
    assert "| 0 | 20 | 20 | 0.100000 |" in report
    assert (
        "| baseline | 60/10/30 (0.600000/0.100000/0.300000) | "
        "40/30/30 (0.400000/0.300000/0.300000) | 0.200000 | "
        "40 / 0.200000 | 20 / 0.100000 | 200 / 400 / 0.500000 |"
    ) in report
    assert (
        "| baseline | 0.200000 / 0.400000 | 0.2 / 0.4 | "
        "5 / 5 / 5 | 10 / 10 / 10 |"
    ) in report
    assert (
        'transitions={"draw":{"draw":60,"loss":0,"win":0},'
        '"loss":{"draw":0,"loss":40,"win":0},'
        '"win":{"draw":0,"loss":0,"win":100}}'
    ) in report
    assert (
        "| 0.500000 | 0.600000 | 0.100000 | False | 0.200000 | "
        "0.100000 | 0.500000 | True | False | False | False | False |"
    ) in report
    assert "Chosen candidate: iteration-3" in report
    assert "Best treatment iteration: iteration-3" in report
    assert "Win/loss/draw Wilson 95%" in report
    assert "Interpretation branch: `better-imitation-and-games`." in report


def test_development_aggregate_preserves_baseline_and_regression_precedes_imitation() -> None:
    """A treatment with better imitation but worse games is a regression, not deployable."""

    baseline_rows = _development_rows()
    rows_by_candidate = {"baseline": baseline_rows}
    for iteration in (1, 2, 3):
        rows = copy.deepcopy(baseline_rows)
        for index in range(iteration * 5):
            rows[index]["outcome"] = "loss"
        rows_by_candidate[f"iteration-{iteration}"] = rows
    supervised = {
        1: _development_supervised_metrics(pre_agreements=50, post_agreements=90),
        2: _development_supervised_metrics(pre_agreements=50, post_agreements=90),
        3: _development_supervised_metrics(pre_agreements=50, post_agreements=90),
    }
    aggregate = dagger_module.build_development_aggregate(
        rows_by_candidate=rows_by_candidate,
        supervised_metrics_by_iteration=supervised,
        evidence_identity={
            "preflight": HASHES["1"], "baseline": HASHES["2"],
            "iterations": [HASHES["3"], HASHES["4"], HASHES["5"]],
            "evaluations": [HASHES["6"], HASHES["7"], HASHES["8"], HASHES["9"]],
            "supervised": [HASHES["a"], HASHES["b"], HASHES["c"]],
        },
    )

    assert aggregate["selected_candidate_id"] == "baseline"
    assert aggregate["best_treatment_candidate_id"] == "iteration-1"
    assert aggregate["interpretation"] == "regression"
