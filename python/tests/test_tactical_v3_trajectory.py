from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from ml_lab.tactical_v3_schema import (
    TacticalV3Reward,
    parse_decision,
)
from ml_lab.tactical_v3_trajectory import (
    ControllerProvenance,
    TacticalV3TrajectoryGame,
    TrajectoryDecisionRecord,
    load_trajectory_game,
    load_trajectory_manifest,
    publish_trajectory_game,
    reconstruct_trajectory_manifest,
    validate_trajectory_manifest_snapshot,
    write_trajectory_manifest,
)
from tests.tactical_v3_fixture_support import load_duel_identity_fixture
from tests.test_tactical_v3_schema import minimal_view_payload


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _decision(decision_id: int):
    identity = load_duel_identity_fixture()
    payload = minimal_view_payload()
    payload["decision_id"] = decision_id
    payload["seat"] = 0
    payload["candidates"][0]["decision_id"] = decision_id
    return parse_decision(payload, identity)


def _reward(*, finalized: bool, terminal: float = 0.0) -> TacticalV3Reward:
    known_material = 0.125
    public_resources = 0.25
    time_pressure = -0.0625
    return TacticalV3Reward(
        terminal_outcome=terminal,
        known_health_adjusted_material_progress=known_material,
        public_resource_progress=public_resources,
        time_pressure=time_pressure,
        total=terminal + known_material + public_resources + time_pressure,
        finalized=finalized,
    )


def _record(
    index: int,
    *,
    finished: bool,
    selected_candidate_id: int = 0,
) -> TrajectoryDecisionRecord:
    return TrajectoryDecisionRecord(
        trajectory_index=index,
        decision=_decision(7 + index),
        selected_candidate_id=selected_candidate_id,
        behavior_mode="categorical",
        log_probability=-0.25,
        entropy=0.5,
        successor_reward=_reward(
            finalized=finished,
            terminal=1.0 if finished else 0.0,
        ),
        terminated_after_selection=finished,
        truncated_after_selection=False,
    )


def _provenance(kind: str, name: str, digest: str):
    return ControllerProvenance(
        kind=kind,
        name=name,
        source=f"fixture:{name}",
        artifact_sha256=digest,
    )


def _replay(command_count: int = 2) -> bytes:
    commands = "".join(f"E {index % 2}\n" for index in range(command_count))
    return (
        "HEXWARS-REPLAY 1\n"
        "META 3 0 1 0 0 0\n"
        f"CMDS {command_count}\n"
        f"{commands}"
    ).encode("utf-8")


def _game(
    *,
    partition: str = "train",
    game_index: int = 0,
    records: tuple[TrajectoryDecisionRecord, ...] | None = None,
) -> TacticalV3TrajectoryGame:
    return TacticalV3TrajectoryGame(
        identity=load_duel_identity_fixture(),
        partition=partition,
        game_index=game_index,
        episode_seed=30_000_000 + game_index,
        profile_id="conversion-1v1-near",
        learner_seat=0,
        reference_seat=0,
        actor=_provenance("model", "candidate", "a" * 64),
        opponent=_provenance("scripted", "passive", "b" * 64),
        records=(
            records
            if records is not None
            else (_record(0, finished=False), _record(1, finished=True))
        ),
        replay=_replay(),
        winner=0,
        terminated=True,
        truncated=False,
        terminal_reward=_reward(finalized=True, terminal=1.0),
        internal_fallback_count=0,
    )


def _rewrite_manifest(path: Path, mutate) -> None:
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    path.write_bytes(_canonical(value))


def test_publish_load_and_reconstruct_roundtrip_is_canonical_and_hashed(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trajectories"
    game = _game()

    published = publish_trajectory_game(archive, game)
    loaded = load_trajectory_game(published, game.identity)

    assert published == archive / "train" / "game-000000"
    assert loaded == game
    assert set(path.name for path in published.iterdir()) == {
        "trajectory.jsonl", "game.replay", "game.json",
    }
    manifest_bytes = (published / "game.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert manifest_bytes == _canonical(manifest)
    assert manifest["partition"] == "train"
    assert manifest["dataset_use"] == "optimization"
    assert manifest["identity"]["scenario_id"] == game.identity.scenario_id
    assert manifest["identity"]["contract_hash"] == game.identity.contract_hash
    assert manifest["identity"]["encoding_hash"] == game.identity.encoding_hash
    assert manifest["identity"]["capacity_hash"] == game.identity.capacity_hash
    for key, filename in (
        ("trajectory", "trajectory.jsonl"),
        ("replay", "game.replay"),
    ):
        data = (published / filename).read_bytes()
        assert manifest["files"][key]["byte_length"] == len(data)
        assert manifest["files"][key]["sha256"] == hashlib.sha256(data).hexdigest()

    reconstructed = reconstruct_trajectory_manifest(archive, game.identity)
    assert reconstructed["partitions"]["train"] == {
        "dataset_use": "optimization",
        "games": [{
            "game_index": 0,
            "path": "train/game-000000/game.json",
            "game_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }],
    }
    assert reconstructed["partitions"]["validation"] == {
        "dataset_use": "early_stop_only",
        "games": [],
    }
    manifest_path = write_trajectory_manifest(archive, game.identity)
    assert manifest_path.read_bytes() == _canonical(reconstructed)
    assert load_trajectory_manifest(archive, game.identity) == reconstructed


def test_validation_partition_is_early_stop_only(tmp_path: Path) -> None:
    game = _game(partition="validation")
    published = publish_trajectory_game(tmp_path / "trajectories", game)

    manifest = json.loads((published / "game.json").read_text(encoding="utf-8"))

    assert manifest["dataset_use"] == "early_stop_only"
    assert load_trajectory_game(published, game.identity).dataset_use == (
        "early_stop_only"
    )


def test_complete_game_may_have_no_learner_decisions(tmp_path: Path) -> None:
    game = replace(
        _game(records=()),
        winner=1,
        terminal_reward=_reward(finalized=True, terminal=-1.0),
    )

    published = publish_trajectory_game(tmp_path / "trajectories", game)
    loaded = load_trajectory_game(published, game.identity)

    assert loaded == game
    assert (published / "trajectory.jsonl").read_bytes() == b""
    manifest = json.loads((published / "game.json").read_text(encoding="utf-8"))
    assert manifest["result"]["learner_decisions"] == 0
    assert manifest["result"]["terminal_reward"] == {
        "finalized": True,
        "known_health_adjusted_material_progress": 0.125,
        "public_resource_progress": 0.25,
        "terminal_outcome": -1.0,
        "time_pressure": -0.0625,
        "total": -0.6875,
    }


def test_opponent_may_finish_after_a_nonfinal_learner_reward(tmp_path: Path) -> None:
    game = replace(
        _game(records=(_record(0, finished=False),)),
        winner=1,
        terminal_reward=_reward(finalized=True, terminal=-1.0),
    )

    published = publish_trajectory_game(tmp_path / "trajectories", game)

    assert load_trajectory_game(published, game.identity) == game


def test_loader_rejects_partition_use_mismatch(tmp_path: Path) -> None:
    game = _game()
    published = publish_trajectory_game(tmp_path / "trajectories", game)
    _rewrite_manifest(
        published / "game.json",
        lambda value: value.__setitem__("dataset_use", "early_stop_only"),
    )

    with pytest.raises(ValueError, match="partition and dataset use"):
        load_trajectory_game(published, game.identity)


def test_loader_rejects_manifest_partition_that_disagrees_with_directory(
    tmp_path: Path,
) -> None:
    game = _game()
    published = publish_trajectory_game(tmp_path / "trajectories", game)
    manifest_path = published / "game.json"
    _rewrite_manifest(
        manifest_path,
        lambda value: value.update({
            "partition": "validation",
            "dataset_use": "early_stop_only",
        }),
    )

    with pytest.raises(ValueError, match="directory and manifest partition"):
        load_trajectory_game(published, game.identity)


def test_loader_rejects_noncontiguous_trajectory_rows(tmp_path: Path) -> None:
    game = _game()
    published = publish_trajectory_game(tmp_path / "trajectories", game)
    trajectory_path = published / "trajectory.jsonl"
    rows = [
        json.loads(line)
        for line in trajectory_path.read_text(encoding="utf-8").splitlines()
    ]
    rows[1]["trajectory_index"] = 2
    trajectory_bytes = b"".join(_canonical(row) for row in rows)
    trajectory_path.write_bytes(trajectory_bytes)

    def update_hashes(value) -> None:
        metadata = value["files"]["trajectory"]
        metadata["byte_length"] = len(trajectory_bytes)
        metadata["sha256"] = hashlib.sha256(trajectory_bytes).hexdigest()

    _rewrite_manifest(published / "game.json", update_hashes)

    with pytest.raises(ValueError, match="indices must be contiguous"):
        load_trajectory_game(published, game.identity)


@pytest.mark.parametrize("mutation", ("hash", "extra", "missing", "noncanonical"))
def test_loader_rejects_tamper_and_nonexact_inventory(
    tmp_path: Path,
    mutation: str,
) -> None:
    game = _game()
    published = publish_trajectory_game(tmp_path / "trajectories", game)
    if mutation == "hash":
        with (published / "trajectory.jsonl").open("ab") as handle:
            handle.write(b" \n")
    elif mutation == "extra":
        (published / "unexpected.txt").write_text("x", encoding="utf-8")
    elif mutation == "missing":
        (published / "game.replay").unlink()
    else:
        value = json.loads((published / "game.json").read_text(encoding="utf-8"))
        (published / "game.json").write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    with pytest.raises(
        ValueError,
        match="inventory|byte length|SHA-256|canonical",
    ):
        load_trajectory_game(published, game.identity)


def test_loader_rejects_wrong_expected_identity(tmp_path: Path) -> None:
    game = _game()
    published = publish_trajectory_game(tmp_path / "trajectories", game)
    match = dict(game.identity.match)
    match["max_steps"] = int(match["max_steps"]) + 1
    wrong = replace(
        game.identity,
        scenario_id="different-scenario",
        contract_hash="c" * 64,
        match=MappingProxyType(match),
    )

    with pytest.raises(ValueError, match="identity does not match"):
        load_trajectory_game(published, wrong)


def test_publish_rejects_nonunique_selection_and_unfinalized_finish(
    tmp_path: Path,
) -> None:
    missing_selection = _game(
        records=(
            _record(0, finished=False),
            _record(1, finished=True, selected_candidate_id=9),
        )
    )
    with pytest.raises(ValueError, match="exactly once"):
        publish_trajectory_game(tmp_path / "first", missing_selection)

    final = replace(
        _record(1, finished=True),
        successor_reward=_reward(finalized=False, terminal=1.0),
    )
    unfinalized = _game(records=(_record(0, finished=False), final))
    with pytest.raises(ValueError, match="finalized"):
        publish_trajectory_game(tmp_path / "second", unfinalized)


def test_publish_requires_actor_and_opponent_artifact_hashes(tmp_path: Path) -> None:
    bad_actor = replace(
        _game(),
        actor=_provenance("model", "candidate", "not-a-hash"),
    )
    with pytest.raises(ValueError, match="artifact_sha256"):
        publish_trajectory_game(tmp_path / "actor", bad_actor)

    bad_opponent = replace(
        _game(),
        opponent=_provenance("scripted", "passive", "not-a-hash"),
    )
    with pytest.raises(ValueError, match="artifact_sha256"):
        publish_trajectory_game(tmp_path / "opponent", bad_opponent)


def test_publish_binds_terminal_reward_to_authoritative_winner(tmp_path: Path) -> None:
    inconsistent = replace(
        _game(),
        winner=1,
        terminal_reward=_reward(finalized=True, terminal=1.0),
    )

    with pytest.raises(ValueError, match="terminal reward does not match winner"):
        publish_trajectory_game(tmp_path / "trajectories", inconsistent)


def test_greedy_behavior_records_the_deterministic_distribution(tmp_path: Path) -> None:
    invalid = replace(
        _record(0, finished=False),
        behavior_mode="greedy",
        log_probability=-0.25,
        entropy=0.5,
    )

    with pytest.raises(ValueError, match="greedy trajectory behavior"):
        publish_trajectory_game(
            tmp_path / "trajectories",
            _game(records=(invalid, _record(1, finished=True))),
        )


def test_publication_never_overwrites_a_completed_game(tmp_path: Path) -> None:
    archive = tmp_path / "trajectories"
    first = _game()
    published = publish_trajectory_game(archive, first)
    before = {path.name: path.read_bytes() for path in published.iterdir()}
    different = replace(first, replay=_replay(3))

    with pytest.raises(FileExistsError, match="already exists|overwrite"):
        publish_trajectory_game(archive, different)

    assert {path.name: path.read_bytes() for path in published.iterdir()} == before


def test_failed_next_publication_preserves_prior_game_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_trajectory as module

    archive = tmp_path / "trajectories"
    first_path = publish_trajectory_game(archive, _game())
    before = {path.name: path.read_bytes() for path in first_path.iterdir()}
    actual_write = module._write_fsynced

    def fail_on_replay(path: Path, data: bytes) -> None:
        if path.name == "game.replay":
            raise OSError("simulated interrupted replay write")
        actual_write(path, data)

    monkeypatch.setattr(module, "_write_fsynced", fail_on_replay)
    with pytest.raises(OSError, match="interrupted"):
        publish_trajectory_game(archive, _game(game_index=1))

    assert {path.name: path.read_bytes() for path in first_path.iterdir()} == before
    assert not (archive / "train" / "game-000001").exists()
    assert not any(
        path.name.startswith(".game-000001.tmp-")
        for path in (archive / "train").iterdir()
    )


def test_manifest_reconstruction_ignores_plain_temp_directories(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trajectories"
    game = _game()
    publish_trajectory_game(archive, game)
    temporary = archive / "train" / ".game-000001.tmp-interrupted"
    temporary.mkdir()
    (temporary / "partial").write_text("not complete", encoding="utf-8")

    manifest = reconstruct_trajectory_manifest(archive, game.identity)

    assert [
        entry["game_index"]
        for entry in manifest["partitions"]["train"]["games"]
    ] == [0]


def test_manifest_reconstruction_ignores_interrupted_atomic_manifest_file(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trajectories"
    game = _game()
    publish_trajectory_game(archive, game)
    (archive / ".manifest.json.interrupted.tmp").write_bytes(b"partial")

    written = write_trajectory_manifest(archive, game.identity)

    assert written == archive / "manifest.json"
    assert load_trajectory_manifest(archive, game.identity)["partitions"][
        "train"
    ]["games"][0]["game_index"] == 0


def test_manifest_reconstruction_rejects_noncontiguous_games(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trajectories"
    game = _game()
    published = publish_trajectory_game(archive, game)
    published.rename(archive / "train" / "game-000001")

    with pytest.raises(ValueError, match="not contiguous"):
        reconstruct_trajectory_manifest(archive, game.identity)


def test_manifest_loader_rejects_stale_physical_inventory(tmp_path: Path) -> None:
    archive = tmp_path / "trajectories"
    first = _game()
    publish_trajectory_game(archive, first)
    write_trajectory_manifest(archive, first.identity)
    publish_trajectory_game(archive, _game(game_index=1))

    with pytest.raises(ValueError, match="does not match physical games"):
        load_trajectory_manifest(archive, first.identity)


def test_checkpoint_snapshot_authenticates_each_referenced_physical_game(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "trajectories"
    game = _game()
    published = publish_trajectory_game(archive, game)
    live = write_trajectory_manifest(archive, game.identity)
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_bytes(live.read_bytes())

    assert validate_trajectory_manifest_snapshot(
        snapshot, archive, game.identity,
    )["partitions"]["train"]["games"][0]["game_index"] == 0

    (published / "game.replay").unlink()
    with pytest.raises(ValueError, match="inventory"):
        validate_trajectory_manifest_snapshot(snapshot, archive, game.identity)
