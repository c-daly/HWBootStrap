"""Physical-contract tests for immutable selective-DAgger overlays."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_lab.contracts import EnvironmentContract
from ml_lab.dagger import (
    DaggerGame,
    DaggerOverlay,
    DaggerOverlayWriter,
    DaggerRow,
    LearnerIdentity,
    OracleSpec,
    open_dagger_overlay,
    publish_dagger_overlay,
    publish_dagger_overlays,
    require_seed_in_partition,
    validate_seed_definitions,
)


HASHES = {letter: letter * 64 for letter in "abcdef1234567890"}


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
            }
        },
    )


def _oracle_payload() -> dict[str, Any]:
    return {
        "oracle_type": "bounded-search",
        "depth": 4,
        "expansion_budget": 512,
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


def _game_payload(*, game_id: int = 0, partition: str = "train", seat: int = 0) -> dict[str, Any]:
    seed = 18_000_000 if partition == "train" else 19_000_000
    return {
        "game_id": game_id,
        "partition": partition,
        "iteration": 1,
        "map_seed": seed,
        "episode_seed": seed,
        "schedule_index": 0,
        "profile": "standard-3v3",
        "reference_seat": 0,
        "learner_seat": seat,
        "opponent": "random",
        "outcome": "win" if seat == 0 else "loss",
        "transition_count": 1,
        "trace_path": f"evidence/game-{game_id}.trace.json",
        "replay_path": f"evidence/game-{game_id}.replay",
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
    with pytest.raises(FrozenInstanceError):
        oracle.depth = 5  # type: ignore[misc]


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
) -> DaggerGame:
    payload = _game_payload(game_id=game_id, partition=partition, seat=seat)
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
) -> tuple[DaggerOverlayWriter, Path]:
    checkpoint = root.parent / f"{partition}-actor.zip"
    checkpoint.write_bytes(b"actor")
    writer = DaggerOverlayWriter.create(
        root,
        contract=contract,
        partition=partition,
        iteration=1,
        oracle=OracleSpec.from_dict(_oracle_payload()),
        learner=LearnerIdentity.from_dict(_learner_payload(checkpoint)),
        scenario_hash=HASHES["1"],
        repository_hash=HASHES["2"],
        panel_hash=HASHES["3"],
        schedule_hash=HASHES["4"],
    )
    return writer, checkpoint


def _seal_pair(
    root: Path,
    contract: EnvironmentContract,
    *,
    partition: str = "train",
) -> tuple[DaggerOverlay, Path]:
    writer, checkpoint = _new_writer(root, contract, partition=partition)
    for game_id, seat in enumerate((0, 1)):
        game = _game(game_id=game_id, partition=partition, seat=seat)
        _write_evidence(root, game)
        writer.append_game(game, [_row(contract, seat=seat)])
    return writer.seal(), checkpoint


def _identity(value: dict[str, Any]) -> str:
    canonical = {key: item for key, item in value.items() if key != "content_identity"}
    payload = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        "schedule_hash", "game_count", "row_count", "games", "content_identity",
    }
    assert manifest["status"] == "completed"
    assert manifest["game_count"] == 2
    assert manifest["row_count"] == 2
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


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("action_size", "action_size"),
        ("oracle", "depth"),
        ("provenance", "repository_hash"),
        ("game_descriptor", "row_count"),
    ],
)
def test_overlay_dto_parser_validates_every_nested_declared_field(
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
        DaggerOverlay.from_dict(payload, root=root)


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

    published = publish_dagger_overlay(staging, destination)

    assert not staging.exists()
    assert destination.is_dir()
    assert published.content_identity == expected.content_identity
    assert open_dagger_overlay(destination).content_identity == expected.content_identity

    checkpoint.write_bytes(b"tampered after publication")
    with pytest.raises(ValueError):
        publish_dagger_overlay(staging, destination)


def test_train_and_validation_overlays_publish_to_distinct_destinations(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    """A shared destination or swapped partition must never publish a mixed artifact."""

    train_staging = tmp_path / "train.staging"
    validation_staging = tmp_path / "validation.staging"
    _seal_pair(train_staging, contract, partition="train")
    _seal_pair(validation_staging, contract, partition="validation")

    train, validation = publish_dagger_overlays(
        train_staging,
        validation_staging,
        tmp_path / "train-overlay",
        tmp_path / "validation-overlay",
    )

    assert train.partition == "train"
    assert validation.partition == "validation"
    assert train.root != validation.root
