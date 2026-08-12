from __future__ import annotations

import dataclasses
import hashlib
import math
from types import MappingProxyType

import numpy as np
import pytest
import torch

from ml_lab.tactical_v3_batching import collate_examples
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_training import TrainerConfig, train_offline
from tests.tactical_v3_fixture_support import load_tiny_corpus_fixture


def test_offline_training_returns_finite_immutable_metrics_and_policy() -> None:
    corpus = load_tiny_corpus_fixture()
    config = TacticalV3ModelConfig()
    result = train_offline(
        corpus.train,
        corpus.validation,
        config,
        ObjectiveConfig(),
        TrainerConfig(max_epochs=1, patience_epochs=1),
    )

    assert result.best_epoch == 0
    assert result.stopped_early is False
    assert len(result.history) == 1
    metric = result.history[0]
    assert type(metric.train) is MappingProxyType
    assert type(metric.validation) is MappingProxyType
    assert metric.validation_policy_nll == metric.validation["policy"]
    assert all(math.isfinite(value) for value in metric.train.values())
    assert all(math.isfinite(value) for value in metric.validation.values())
    batch = collate_examples(corpus.validation[:1], config.horizon_turns)
    assert result.model.select(batch)
    assert all(torch.isfinite(parameter).all() for parameter in result.model.parameters())


def _state_digest(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _train(*, max_epochs: int = 2, patience_epochs: int = 2):
    corpus = load_tiny_corpus_fixture()
    return train_offline(
        corpus.train,
        corpus.validation,
        TacticalV3ModelConfig(),
        ObjectiveConfig(),
        TrainerConfig(
            seed=227,
            batch_size=4,
            max_epochs=max_epochs,
            patience_epochs=patience_epochs,
        ),
    )


def test_repeated_training_is_bitwise_deterministic() -> None:
    left = _train()
    right = _train()

    assert left.history == right.history
    assert left.best_epoch == right.best_epoch
    assert left.best_validation_policy_nll == right.best_validation_policy_nll
    assert _state_digest(dict(left.model.state_dict())) == _state_digest(
        dict(right.model.state_dict())
    )


def test_reversed_split_inputs_produce_the_same_result() -> None:
    corpus = load_tiny_corpus_fixture()
    config = TacticalV3ModelConfig()
    trainer = TrainerConfig(max_epochs=2, patience_epochs=2)
    left = train_offline(corpus.train, corpus.validation, config, ObjectiveConfig(), trainer)
    right = train_offline(
        tuple(reversed(corpus.train)),
        tuple(reversed(corpus.validation)),
        config,
        ObjectiveConfig(),
        trainer,
    )

    assert left.history == right.history
    assert _state_digest(dict(left.model.state_dict())) == _state_digest(
        dict(right.model.state_dict())
    )


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", True), ("seed", -1), ("batch_size", 0),
        ("max_epochs", 0), ("patience_epochs", 0),
        ("learning_rate", 1), ("learning_rate", float("nan")),
        ("gradient_clip_norm", 0.0), ("device", "cpu:0"),
        ("device", np.str_("cpu")),
    ),
)
def test_trainer_config_rejects_noncanonical_field_values(
    field: str, value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        TrainerConfig(**{field: value})


def test_cuda_bare_device_normalizes_to_current_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    assert TrainerConfig(device="cuda").device == "cuda:1"
    assert TrainerConfig(device="cuda:0").device == "cuda:0"


def test_empty_duplicate_and_overlapping_splits_fail_before_training() -> None:
    corpus = load_tiny_corpus_fixture()
    config = TacticalV3ModelConfig()
    trainer = TrainerConfig(max_epochs=1, patience_epochs=1)
    invalid = (
        ((), corpus.validation, "training split must be non-empty"),
        (corpus.train, (), "validation split must be non-empty"),
        ((corpus.train[0], corpus.train[0]), corpus.validation, "duplicate training example"),
        (corpus.train, (corpus.validation[0], corpus.validation[0]), "duplicate validation example"),
        (corpus.train, (dataclasses.replace(corpus.train[0]),), "splits overlap"),
    )
    for train, validation, message in invalid:
        with pytest.raises(ValueError, match=message):
            train_offline(train, validation, config, ObjectiveConfig(), trainer)


def test_batch_transfer_reconstructs_immutable_nested_mappings() -> None:
    import ml_lab.tactical_v3_training as training

    corpus = load_tiny_corpus_fixture()
    source = collate_examples(corpus.validation[:2], TacticalV3ModelConfig().horizon_turns)
    moved = training._batch_to_device(source, torch.device("cpu"))

    assert moved is not source
    assert type(moved.tables) is MappingProxyType
    assert type(moved.table_slices) is MappingProxyType
    assert moved.table_slices == source.table_slices
    assert moved.table_slices is not source.table_slices
    assert all(type(table.categorical) is MappingProxyType for table in moved.tables.values())
    assert all(type(table.boolean) is MappingProxyType for table in moved.tables.values())
