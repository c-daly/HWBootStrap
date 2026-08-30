from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ml_lab.tactical_v3_continuation import (
    StructuredContinuationConfig,
    _LiveTelemetry,
    _candidate_improves_source,
    _partition_target_complete,
    _resolve_opponent,
    _seat_sequence,
    _write_collection_manifest,
)
from ml_lab.tactical_v3_schema import parse_spaces


ROOT = Path(__file__).resolve().parents[2]
DUEL_SPACES = (
    ROOT / "python" / "tests" / "fixtures" / "tactical_v3"
    / "seed-41-duel-spaces.json"
)


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


def test_config_requires_the_source_runs_exact_scenario(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario.json").write_text("{}\n", encoding="utf-8")
    other = tmp_path / "other.json"
    other.write_text("{}\n", encoding="utf-8")

    _config(source).validate()
    with pytest.raises(ValueError, match="source_run/scenario.json exactly"):
        _config(source, scenario_file=other).validate()


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
    import json
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


def test_collection_manifest_serializes_immutable_semantic_identity(
    tmp_path: Path,
) -> None:
    import json

    source = tmp_path / "source"
    source.mkdir()
    (source / "scenario.json").write_text("{}\n", encoding="utf-8")
    identity = parse_spaces(
        json.loads(DUEL_SPACES.read_text(encoding="utf-8"))
    )
    loaded = SimpleNamespace(metadata=SimpleNamespace(
        model_state_sha256="a" * 64,
        corpus_sha256="b" * 64,
        best_epoch=3,
        best_validation_policy_nll=1.25,
    ))

    path, digest = _write_collection_manifest(
        tmp_path,
        _config(source),
        identity,
        loaded,
        _resolve_opponent("greedy"),
        [],
        [],
        b"",
        b"",
    )

    manifest = json.loads(path.read_text(encoding="utf-8"))
    assert type(manifest["identity"]["match"]) is dict
    assert manifest["identity"]["contract_hash"] == identity.contract_hash
    assert len(digest) == 64


def test_publication_requires_improvement_over_source_validation() -> None:
    def artifacts(baseline: float, candidate: float) -> SimpleNamespace:
        return SimpleNamespace(
            initial_validation=SimpleNamespace(policy_nll=baseline),
            restored_validation=SimpleNamespace(policy_nll=candidate),
        )

    assert _candidate_improves_source(artifacts(2.0, 1.9))
    assert not _candidate_improves_source(artifacts(2.0, 2.0))
    assert not _candidate_improves_source(artifacts(2.0, 2.1))
