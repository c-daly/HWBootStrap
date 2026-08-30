from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

from ml_lab.tactical_v3_continuation import (
    StructuredContinuationConfig,
    _LiveTelemetry,
    _StartProfileScheduler,
    _candidate_improves_source,
    _collect_partition,
    _create_live_run,
    _identity_start_distribution,
    _partition_target_complete,
    _publication_target,
    _resolve_opponent,
    _seat_sequence,
    _start_distribution,
    _training_manifest,
    _validate_model_opponent,
    _validate_target_scenario_identity,
    _write_collection_manifest,
    run_structured_continuation,
)
from ml_lab.scenarios import resolve_scenario
from ml_lab.tactical_v3_schema import parse_spaces


ROOT = Path(__file__).resolve().parents[2]
DUEL_SPACES = (
    ROOT / "python" / "tests" / "fixtures" / "tactical_v3"
    / "seed-41-duel-spaces.json"
)
SCENARIO = ROOT / "python" / "config" / "annihilation-structured-imitation-v1.json"


def _config(source: Path, **overrides) -> StructuredContinuationConfig:
    values = {
        "run_name": "latest-vs-greedy",
        "source_run": source,
        "scenario_file": source / "scenario.json",
        "opponent": "greedy",
        "train_label_target": 7500,
        "validation_label_target": 3000,
        "seed": 227,
        "device": "cuda:0",
        "learner_seat": "alternating",
        "trackers": ({"kind": "local"}, {"kind": "tensorboard"}),
    }
    values.update(overrides)
    return StructuredContinuationConfig(**values)


def test_config_accepts_an_independent_target_scenario(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario.json").write_text("{}\n", encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")

    _config(source).validate()
    _config(source, scenario_file=other).validate()


@pytest.mark.parametrize("entry_kind", ["directory", "file"])
def test_publication_target_collision_fails_before_lifecycle_creation(
    tmp_path: Path,
    entry_kind: str,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    scenario = tmp_path / "scenario.json"
    scenario.write_text("{}\n", encoding="utf-8")
    runs = tmp_path / "runs"
    runs.mkdir()
    publication = runs / "latest-vs-greedy-model"
    if entry_kind == "directory":
        publication.mkdir()
    else:
        publication.write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="publication target already exists"):
        run_structured_continuation(
            _config(source, scenario_file=scenario, device="cpu"),
            runs_root=runs,
            server_cmd=("server-must-not-start",),
        )

    assert not (runs / "latest-vs-greedy").exists()


def test_publication_target_returns_unoccupied_model_namespace(
    tmp_path: Path,
) -> None:
    assert _publication_target(tmp_path, "new-run") == (
        tmp_path / "new-run-model"
    )


def test_continuation_rejects_a_form_valid_symmetric_random_target(
    tmp_path: Path,
) -> None:
    document = json.loads(SCENARIO.read_text(encoding="utf-8"))
    document["tactical_v3"].update({
        "placement_policy": "symmetric-random-v1",
        "start_profiles": [],
        "start_distribution": [],
    })
    path = tmp_path / "symmetric.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    resolved = resolve_scenario(
        environment="tactical-v3",
        scenario_file=path,
        template_id=None,
    )

    with pytest.raises(ValueError, match="requires profiled-seeded-v1"):
        _start_distribution(resolved.document)


def test_target_start_distribution_is_weighted_and_reciprocal_ready() -> None:
    resolved = resolve_scenario(
        environment="tactical-v3",
        scenario_file=SCENARIO,
        template_id=None,
    )
    distribution = _start_distribution(resolved.document)
    scheduler = _StartProfileScheduler(distribution)

    profiles = [scheduler.next_profile() for _ in range(20)]

    assert Counter(profiles) == {
        "standard-3v3": 14,
        "conversion-3v1-near": 1,
        "conversion-3v1-far": 1,
        "conversion-2v1-near": 1,
        "conversion-2v1-far": 1,
        "conversion-1v1-near": 1,
        "conversion-1v1-far": 1,
    }
    assert all("medium" not in profile for profile in profiles)


def test_target_scenario_is_cross_checked_against_gymserver_identity() -> None:
    resolved = resolve_scenario(
        environment="tactical-v3",
        scenario_file=SCENARIO,
        template_id=None,
    )
    identity = parse_spaces(json.loads(DUEL_SPACES.read_text(encoding="utf-8")))
    distribution = _start_distribution(resolved.document)

    _validate_target_scenario_identity(resolved, identity, distribution)

    with pytest.raises(ValueError, match="identity does not match GymServer"):
        _validate_target_scenario_identity(
            replace(resolved, template_id="wrong-target"),
            identity,
            distribution,
        )

    wrong_document = json.loads(resolved.canonical_json)
    wrong_document["board"]["width"] += 1
    with pytest.raises(ValueError, match=r"board\.width does not match GymServer"):
        _validate_target_scenario_identity(
            replace(resolved, document=wrong_document),
            identity,
            distribution,
        )


def test_collection_assigns_a_weighted_start_to_the_complete_reciprocal_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_continuation as module

    resolved = resolve_scenario(
        environment="tactical-v3",
        scenario_file=SCENARIO,
        template_id=None,
    )
    identity = parse_spaces(
        json.loads(DUEL_SPACES.read_text(encoding="utf-8"))
    )
    observed = []

    def collect(client, source, item, **kwargs):
        observed.append((item, kwargs["allow_compatible_identity_transfer"]))
        return SimpleNamespace(
            summary=SimpleNamespace(
                winner=-1,
                schedule=item,
                disagreements=0,
                internal_fallback_count=0,
                decisions=1,
            ),
            records=(SimpleNamespace(eligibility_reasons=()),),
        )

    monkeypatch.setattr(module, "collect_dagger_game", collect)
    monkeypatch.setattr(module, "_stop_requested", lambda run_dir: False)
    monkeypatch.setattr(module, "write_dagger_episode", lambda path, episode: None)
    monkeypatch.setattr(module, "_sha256_file", lambda path: "a" * 64)
    monkeypatch.setattr(module, "_append_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "update_run_state", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_log", lambda *args, **kwargs: None)

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source = SimpleNamespace(model=SimpleNamespace(config=None))
    episodes, _, labels, games = _collect_partition(
        SimpleNamespace(identity=identity),
        source,
        _resolve_opponent("random"),
        _config(source_dir, device="cpu"),
        tmp_path,
        SimpleNamespace(collection=lambda **kwargs: None),
        partition="train",
        target=1,
        seed_start=123,
        global_game_start=0,
        global_label_start=0,
        started=0.0,
        outcomes=Counter(),
        seat_outcomes={0: Counter(), 1: Counter()},
        reasons=Counter(),
        global_disagreements=[0],
        global_fallbacks=[0],
        start_distribution=_start_distribution(resolved.document),
    )

    assert labels == games == len(episodes) == 2
    assert [item.learner_seat for item, _ in observed] == [0, 1]
    assert len({item.episode_seed for item, _ in observed}) == 1
    assert len({item.profile_id for item, _ in observed}) == 1
    assert all(transfer for _, transfer in observed)


def test_model_opponent_transfer_accepts_its_own_compatible_architecture(
    tmp_path: Path,
) -> None:
    from ml_lab.tactical_v3_controller import StructuredController
    from ml_lab.tactical_v3_model import TacticalV3Policy
    from ml_lab.tactical_v3_pilot import _pilot_configs

    identity = parse_spaces(
        json.loads(DUEL_SPACES.read_text(encoding="utf-8"))
    )
    source_match = dict(identity.match)
    source_match["max_steps"] += 8
    source_identity = replace(
        identity,
        scenario_id="compatible-source-scenario",
        contract_hash="e" * 64,
        match=MappingProxyType(source_match),
    )
    expected, _, _ = _pilot_configs(227, "cpu")
    wrong = replace(expected, hidden_dim=expected.hidden_dim * 2)
    controller = StructuredController(
        tmp_path,
        tmp_path / "best.pt",
        TacticalV3Policy(wrong),
        source_identity,
    )
    resolved = SimpleNamespace(
        algorithm="structured_imitation",
        model=controller,
        contract=source_identity,
    )

    _validate_model_opponent(resolved, identity)

    _validate_model_opponent(
        SimpleNamespace(
            algorithm="structured_imitation",
            model=replace(controller, policy=TacticalV3Policy(expected)),
            contract=source_identity,
        ),
        identity,
    )

    incompatible_identity = replace(
        source_identity, capacity_hash="f" * 64,
    )
    with pytest.raises(ValueError, match="capacity hash"):
        _validate_model_opponent(
            SimpleNamespace(
                algorithm="structured_imitation",
                model=replace(controller, identity=incompatible_identity),
                contract=incompatible_identity,
            ),
            identity,
        )


def test_config_rejects_unsafe_targets_seed_and_model_free_checkpoint(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="train label target"):
        _config(source, train_label_target=1).validate()
    with pytest.raises(ValueError, match="seed"):
        _config(source, seed=20_001).validate()
    with pytest.raises(ValueError, match="metadata-backed run"):
        _resolve_opponent("ppo:standalone.zip")
    with pytest.raises(ValueError, match="only local and TensorBoard"):
        _config(source, trackers=({"kind": "wandb"},)).validate()


def test_scripted_opponents_and_learner_seats_preserve_existing_choices() -> None:
    assert _resolve_opponent("greedy").kind == "greedy"
    assert _resolve_opponent("random").kind == "random"
    assert _seat_sequence("alternating") == (0, 1)
    assert _seat_sequence("0") == (0,)
    assert _seat_sequence("1") == (1,)


def test_alternating_collection_finishes_the_reciprocal_seat_pair() -> None:
    seats = _seat_sequence("alternating")

    assert not _partition_target_complete(10, 10, 0, seats)
    assert _partition_target_complete(11, 10, 1, seats)
    assert _partition_target_complete(10, 10, 0, (0,))


def test_tensorboard_writer_failure_degrades_without_stopping_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ml_lab.tactical_v3_continuation as module

    class FailingWriter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def add_scalar(self, *args, **kwargs) -> None:
            raise RuntimeError("event sink unavailable")

        def flush(self) -> None:
            pass

        def close(self) -> None:
            pass

    run = tmp_path / "run"
    run.mkdir()
    (run / "run.json").write_text(
        '{"tracker_status":[]}\n', encoding="utf-8",
    )
    monkeypatch.setattr(module, "SummaryWriter", FailingWriter)

    telemetry = _LiveTelemetry(
        run, enabled=True, train_target=10, validation_target=4,
    )

    assert not telemetry.enabled
    status = json.loads((run / "run.json").read_text(encoding="utf-8"))
    assert status["tracker_status"][0]["status"] == "degraded"
    assert "event sink unavailable" in status["tracker_status"][0]["message"]


def test_lifecycle_collection_and_training_embed_full_source_policy_provenance(
    tmp_path: Path,
) -> None:
    from ml_lab.tactical_v3_pilot import _pilot_configs

    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario.json").write_text("{}\n", encoding="utf-8")
    identity = parse_spaces(
        json.loads(DUEL_SPACES.read_text(encoding="utf-8"))
    )
    source_identity = replace(
        identity,
        scenario_id="source-scenario",
        contract_hash="e" * 64,
    )
    model_config, _, _ = _pilot_configs(227, "cpu")
    loaded = SimpleNamespace(
        model=SimpleNamespace(config=model_config),
        metadata=SimpleNamespace(
            identity=source_identity,
            model_state_sha256="a" * 64,
            corpus_sha256="b" * 64,
            best_epoch=3,
            best_validation_policy_nll=1.25,
        ),
    )
    config = _config(source, device="cpu")
    opponent = _resolve_opponent("greedy")

    path, digest = _write_collection_manifest(
        tmp_path,
        config,
        identity,
        loaded,
        opponent,
        [],
        [],
        b"",
        b"",
        _identity_start_distribution(identity),
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert type(manifest["identity"]["match"]) is dict
    assert manifest["identity"]["contract_hash"] == identity.contract_hash
    assert manifest["schedule"]["profile_scheduler"] == (
        "smooth-weighted-reciprocal-v1"
    )
    assert sum(
        row["basis_points"]
        for row in manifest["schedule"]["start_distribution"]
    ) == 10_000
    assert len(digest) == 64

    resolved = resolve_scenario(
        environment="tactical-v3",
        scenario_file=SCENARIO,
        template_id=None,
    )
    lifecycle_dir = _create_live_run(
        tmp_path / "runs",
        config,
        resolved,
        identity,
        loaded,
        opponent,
    )
    lifecycle = json.loads(
        (lifecycle_dir / "run.json").read_text(encoding="utf-8")
    )
    artifacts = SimpleNamespace(
        initial_validation=SimpleNamespace(policy_nll=1.25),
        restored_validation=SimpleNamespace(policy_nll=1.0),
        loaded=SimpleNamespace(metadata=SimpleNamespace(
            best_epoch=4,
            best_validation_policy_nll=1.0,
            model_state_sha256="c" * 64,
        )),
        duration_seconds=12.0,
    )
    training = _training_manifest(
        config,
        loaded,
        "d" * 64,
        "f" * 64,
        artifacts,
        None,
    )
    expected_identity = manifest["source"]["semantic_identity"]
    expected_config = json.loads(json.dumps(asdict(model_config)))
    assert expected_identity["scenario_id"] == "source-scenario"
    assert "match" in expected_identity and "capacity" in expected_identity
    for provenance in (
        lifecycle["source_policy"],
        manifest["source"],
        training["source"],
    ):
        assert provenance["semantic_identity"] == expected_identity
        assert json.loads(json.dumps(provenance["model_config"])) == expected_config


def test_publication_requires_improvement_over_source_validation() -> None:
    def artifacts(baseline: float, candidate: float) -> SimpleNamespace:
        return SimpleNamespace(
            initial_validation=SimpleNamespace(policy_nll=baseline),
            restored_validation=SimpleNamespace(policy_nll=candidate),
        )

    assert _candidate_improves_source(artifacts(2.0, 1.9))
    assert not _candidate_improves_source(artifacts(2.0, 2.0))
    assert not _candidate_improves_source(artifacts(2.0, 2.1))
