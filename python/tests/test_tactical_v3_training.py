from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Literal, Mapping

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import ml_lab.tactical_v3_training as training
from ml_lab.tactical_v3_batching import CandidateBatch, RaggedBatch, RelationNeighborhoodBatch, TokenTableBatch, collate_examples
from ml_lab.tactical_v3_corpus import StructuredExample, _parse_row
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, PolicyOutput, TacticalV3Policy
from ml_lab.tactical_v3_objectives import LossBreakdown, ObjectiveConfig, structured_imitation_loss
from ml_lab.tactical_v3_training import (
    EpochMetrics,
    StepMetrics,
    TrainerConfig,
    TrainingResult,
    train_offline,
)
from tests.tactical_v3_fixture_support import (
    TINY_CORPUS_ROOT,
    load_duel_identity_fixture,
    load_tiny_corpus_fixture,
)

METRIC_KEYS = ("total", "policy", "outcome", "horizon", "remaining_turns")

@dataclass(frozen=True, slots=True)
class TrainerTestCase:
    train: tuple
    validation: tuple
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig

@dataclass(frozen=True, slots=True)
class FaultResult:
    error: BaseException
    optimizer_steps: int
    before_state_sha256: str
    after_state_sha256: str
    result: TrainingResult | None

@dataclass(slots=True)
class TrainingTrace:
    optimizer_orders: list[tuple] = field(default_factory=list)
    validation_orders: list[tuple] = field(default_factory=list)

def stable_example_identity(example: StructuredExample) -> str:
    key = training._canonical_example_key(example)
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

def make_trainer_case(*, device: str = "cpu", max_epochs: int = 6, batch_size: int = 4, patience_epochs: int = 6) -> TrainerTestCase:
    corpus = load_tiny_corpus_fixture()
    return TrainerTestCase(corpus.train, corpus.validation, TacticalV3ModelConfig(), ObjectiveConfig(), TrainerConfig(seed=227, batch_size=batch_size, max_epochs=max_epochs, patience_epochs=patience_epochs, device=device))


def make_unleased_trainer_case(
    *, max_epochs: int, patience_epochs: int,
) -> TrainerTestCase:
    identity = load_duel_identity_fixture()

    def first_row(name: str) -> StructuredExample:
        path = Path(TINY_CORPUS_ROOT) / name
        raw = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        return _parse_row(raw, identity)

    return TrainerTestCase(
        (first_row("train.jsonl"),),
        (first_row("validation.jsonl"),),
        TacticalV3ModelConfig(),
        ObjectiveConfig(),
        TrainerConfig(
            seed=227,
            batch_size=1,
            max_epochs=max_epochs,
            patience_epochs=patience_epochs,
            device="cpu",
        ),
    )

def run_training_case(case: TrainerTestCase) -> TrainingResult:
    return train_offline(case.train, case.validation, case.model_config, case.objective_config, case.trainer_config)

def assert_state_dict_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None:
    assert tuple(left) == tuple(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert left[name].device == right[name].device
        assert left[name].shape == right[name].shape
        assert torch.equal(left[name], right[name]), name

def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii")); digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def iter_batch_tensors(batch: RaggedBatch) -> Iterator[Tensor]:
    def walk(value: object) -> Iterator[Tensor]:
        if isinstance(value, Tensor):
            yield value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field_info in dataclasses.fields(value):
                yield from walk(getattr(value, field_info.name))
        elif isinstance(value, Mapping):
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                yield from walk(nested)
    yield from walk(batch)

def mapping_proxy_batch(batch: RaggedBatch) -> RaggedBatch:
    tables = MappingProxyType({
        name: TokenTableBatch(
            table.numeric,
            MappingProxyType(dict(table.categorical)),
            MappingProxyType(dict(table.boolean)),
            table.mask,
        )
        for name, table in batch.tables.items()
    })
    return dataclasses.replace(
        batch, tables=tables,
        table_slices=MappingProxyType(dict(batch.table_slices)),
    )

@contextmanager
def capture_training_trace(trace: TrainingTrace) -> Iterator[None]:
    original_collate = training._collate_training_batch
    original_evaluate = training._evaluate_validation

    def traced_collate(
        rows: tuple, horizons: tuple,
    ) -> RaggedBatch:
        trace.optimizer_orders.append(
            tuple(stable_example_identity(row) for row in rows)
        )
        return original_collate(rows, horizons)

    def traced_evaluate(
        model: TacticalV3Policy,
        rows: tuple,
        model_config: TacticalV3ModelConfig,
        objective_config: ObjectiveConfig,
        batch_size: int,
        device: torch.device,
        *,
        epoch: int,
    ) -> tuple[Mapping[str, float], float]:
        trace.validation_orders.append(
            tuple(stable_example_identity(row) for row in rows)
        )
        return original_evaluate(
            model, rows, model_config, objective_config, batch_size, device,
            epoch=epoch,
        )

    training._collate_training_batch = traced_collate
    training._evaluate_validation = traced_evaluate
    try:
        yield
    finally:
        training._collate_training_batch = original_collate
        training._evaluate_validation = original_evaluate


def scripted_validation_batch_losses(
    script: tuple,
    states: dict[int, str],
    observed: list[tuple[int, int, int]],
) -> Callable:
    def compute(
        model: TacticalV3Policy,
        batch: RaggedBatch,
        objective_config: ObjectiveConfig,
        *,
        epoch: int,
        batch_index: int,
    ) -> LossBreakdown:
        del objective_config
        states.setdefault(epoch, state_dict_sha256(model.state_dict()))
        observed.append((epoch, batch_index, int(batch.node_mask.shape[0])))
        value = torch.tensor(
            script[epoch][batch_index],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        return LossBreakdown(value, value, value, value, value)

    return compute


def assert_valid_padded_negative_infinity_case() -> None:
    case = make_trainer_case(
        max_epochs=1, batch_size=2, patience_epochs=1
    )
    batch = collate_examples(case.train[:2], case.model_config.horizon_turns)
    assert bool((~batch.candidates.mask).any())
    torch.manual_seed(case.trainer_config.seed)
    model = TacticalV3Policy(case.model_config).eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(
        output.candidate_logits[batch.candidates.mask]
    ).all()
    assert torch.isneginf(
        output.candidate_logits[~batch.candidates.mask]
    ).all()
    assert torch.isfinite(output.outcome_logits).all()
    assert torch.isfinite(output.horizon_logits).all()
    assert torch.isfinite(output.remaining_turns).all()
    result = run_training_case(case)
    assert tuple(metric.epoch for metric in result.history) == (0,)


def run_fault_case(
    stage: Literal[
        "valid_logit_nan", "valid_logit_neg_inf", "outcome", "horizon",
        "remaining", "policy", "outcome_loss", "horizon_loss",
        "remaining_loss", "total", "mask", "gradient", "clip", "parameter",
    ],
) -> FaultResult:
    case = make_trainer_case(max_epochs=1, patience_epochs=1)
    steps = 0
    captured_model: TacticalV3Policy | None = None
    before = ""
    original_init = TacticalV3Policy.__init__
    original_forward = TacticalV3Policy.forward
    original_loss = training.structured_imitation_loss
    original_transfer = training._batch_to_device
    original_after_backward = training._after_backward
    original_clip = training._clip_grad_norm
    original_after_step = training._after_optimizer_step

    def initialize(
        model: TacticalV3Policy, config: TacticalV3ModelConfig,
    ) -> None:
        nonlocal captured_model, before
        original_init(model, config)
        captured_model = model
        before = state_dict_sha256(model.state_dict())

    def forward(model: TacticalV3Policy, batch: RaggedBatch) -> PolicyOutput:
        output = original_forward(model, batch)
        if stage in {"valid_logit_nan", "valid_logit_neg_inf"}:
            logits = output.candidate_logits.clone()
            row = int(torch.nonzero(
                batch.candidates.mask[0], as_tuple=False
            )[0, 0])
            logits[0, row] = (
                float("nan") if stage == "valid_logit_nan" else float("-inf")
            )
            return dataclasses.replace(output, candidate_logits=logits)
        field_name = {
            "outcome": "outcome_logits",
            "horizon": "horizon_logits",
            "remaining": "remaining_turns",
        }.get(stage)
        if field_name is None:
            return output
        changed = getattr(output, field_name).clone()
        changed.reshape(-1)[0] = float("nan")
        return dataclasses.replace(output, **{field_name: changed})

    def loss(
        output: PolicyOutput, batch: RaggedBatch, config: ObjectiveConfig,
    ) -> LossBreakdown:
        value = original_loss(output, batch, config)
        field_name = {
            "policy": "policy",
            "outcome_loss": "outcome",
            "horizon_loss": "horizon",
            "remaining_loss": "remaining_turns",
            "total": "total",
        }.get(stage)
        if field_name is None:
            return value
        return dataclasses.replace(
            value,
            **{field_name: getattr(value, field_name) * float("nan")},
        )

    def transfer(batch: RaggedBatch, device: torch.device) -> RaggedBatch:
        moved = original_transfer(batch, device)
        if stage != "mask":
            return moved
        return dataclasses.replace(
            moved,
            candidates=dataclasses.replace(
                moved.candidates,
                mask=moved.candidates.mask.to(torch.int64),
            ),
        )

    def after_backward(
        model: TacticalV3Policy, *, epoch: int, batch_index: int,
    ) -> None:
        original_after_backward(model, epoch=epoch, batch_index=batch_index)
        if stage == "gradient":
            parameter = next(
                value for value in model.parameters() if value.grad is not None
            )
            with torch.no_grad():
                parameter.grad.reshape(-1)[0] = float("nan")

    def clip(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
        result = original_clip(parameters, max_norm)
        if stage == "clip":
            return torch.tensor(float("inf"), device=result.device)
        return result

    def after_step(
        model: TacticalV3Policy, *, epoch: int, batch_index: int,
    ) -> None:
        nonlocal steps
        original_after_step(model, epoch=epoch, batch_index=batch_index)
        steps += 1
        if stage == "parameter":
            parameter = next(model.parameters())
            with torch.no_grad():
                parameter.reshape(-1)[0] = float("inf")

    TacticalV3Policy.__init__ = initialize
    TacticalV3Policy.forward = forward
    training.structured_imitation_loss = loss
    training._batch_to_device = transfer
    training._after_backward = after_backward
    training._clip_grad_norm = clip
    training._after_optimizer_step = after_step
    try:
        try:
            run_training_case(case)
        except BaseException as caught:
            error = caught
        else:
            error = AssertionError("fault did not fail")
    finally:
        TacticalV3Policy.__init__ = original_init
        TacticalV3Policy.forward = original_forward
        training.structured_imitation_loss = original_loss
        training._batch_to_device = original_transfer
        training._after_backward = original_after_backward
        training._clip_grad_norm = original_clip
        training._after_optimizer_step = original_after_step
    assert captured_model is not None and before
    after = state_dict_sha256(captured_model.state_dict())
    return FaultResult(error, steps, before, after, None)

INTEGER_FIELDS = ("seed", "batch_size", "max_epochs", "patience_epochs")
POSITIVE_INTEGER_FIELDS = ("batch_size", "max_epochs", "patience_epochs")
FLOAT_FIELDS = ("learning_rate", "gradient_clip_norm")
CONFIG_INVALID_CASES: tuple = (
    *((field, True) for field in (*INTEGER_FIELDS, *FLOAT_FIELDS, "device")),
    *((field, torch.tensor(1)) for field in INTEGER_FIELDS),
    *((field, np.int64(1)) for field in INTEGER_FIELDS),
    *((field, 1.0) for field in INTEGER_FIELDS),
    ("seed", -1),
    *((field, value) for field in POSITIVE_INTEGER_FIELDS for value in (0, -1)),
    *((field, torch.tensor(1.0)) for field in FLOAT_FIELDS),
    *((field, np.float64(1.0)) for field in FLOAT_FIELDS),
    *((field, 1) for field in FLOAT_FIELDS),
    *((field, value) for field in FLOAT_FIELDS for value in (
        0.0, -1.0, float("nan"), float("inf"), float("-inf"),
    )),
    ("device", torch.tensor(0)),
    ("device", np.str_("cpu")),
    ("device", torch.device("cpu")),
    *(("device", value) for value in (
        "", " cpu", "cpu:0", "mps", "CUDA", "cuda:x", "cuda:-1", "cuda:0:1",
    )),
)


def test_seed_zero_is_the_only_nonpositive_integer_exception() -> None:
    assert TrainerConfig(seed=0).seed == 0


def test_public_train_offline_interface_binds_a_callable() -> None:
    assert train_offline is training.train_offline
    assert callable(train_offline)


def test_train_offline_reports_each_completed_epoch() -> None:
    case = make_unleased_trainer_case(max_epochs=2, patience_epochs=2)
    observed: list[EpochMetrics] = []

    result = train_offline(
        case.train,
        case.validation,
        case.model_config,
        case.objective_config,
        case.trainer_config,
        epoch_callback=observed.append,
    )

    assert tuple(observed) == result.history
    assert tuple(metric.epoch for metric in observed) == (0, 1)


def test_train_offline_reports_every_completed_train_and_validation_batch() -> None:
    case = make_unleased_trainer_case(max_epochs=2, patience_epochs=2)
    observed: list[StepMetrics] = []

    train_offline(
        case.train,
        case.validation,
        case.model_config,
        case.objective_config,
        case.trainer_config,
        step_callback=observed.append,
    )

    assert tuple(metric.phase for metric in observed) == (
        "train", "validation", "train", "validation",
    )
    assert tuple(metric.epoch for metric in observed) == (0, 0, 1, 1)
    assert tuple(metric.batch_index for metric in observed) == (0, 0, 0, 0)
    assert tuple(metric.global_step for metric in observed) == (1, 1, 2, 2)
    assert tuple(metric.example_count for metric in observed) == (1, 1, 1, 1)
    assert all(tuple(metric.metrics) == METRIC_KEYS for metric in observed)
    with pytest.raises(TypeError):
        observed[0].metrics["policy"] = 0.0


def test_train_offline_deadline_interrupts_before_an_optimizer_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_unleased_trainer_case(max_epochs=2, patience_epochs=2)
    optimizer_steps = 0
    original = training._after_optimizer_step

    def count_step(model, *, epoch: int, batch_index: int) -> None:
        nonlocal optimizer_steps
        original(model, epoch=epoch, batch_index=batch_index)
        optimizer_steps += 1

    monkeypatch.setattr(training, "_after_optimizer_step", count_step)
    with pytest.raises(TimeoutError, match="training deadline"):
        train_offline(
            case.train,
            case.validation,
            case.model_config,
            case.objective_config,
            case.trainer_config,
            deadline_monotonic=0.0,
        )

    assert optimizer_steps == 0


def test_completed_optimizer_step_is_reported_before_deadline_interrupts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_unleased_trainer_case(max_epochs=2, patience_epochs=2)
    observed: list[StepMetrics] = []
    deadline_checks = 0

    def expire_after_first_optimizer_step(deadline: float | None) -> None:
        nonlocal deadline_checks
        deadline_checks += 1
        if deadline_checks == 4:
            raise TimeoutError("training deadline reached")

    monkeypatch.setattr(
        training, "_check_training_deadline", expire_after_first_optimizer_step,
    )
    with pytest.raises(TimeoutError, match="training deadline"):
        train_offline(
            case.train,
            case.validation,
            case.model_config,
            case.objective_config,
            case.trainer_config,
            step_callback=observed.append,
            deadline_monotonic=1.0,
        )

    assert tuple((item.phase, item.global_step) for item in observed) == (("train", 1),)

@pytest.mark.parametrize(("field", "value"), CONFIG_INVALID_CASES)
def test_every_config_field_rejects_its_invalid_type_and_domain_matrix(
    field: str, value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        TrainerConfig(**{field: value})


def test_cuda_device_availability_index_and_bare_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="device.*CUDA.*unavailable"):
        TrainerConfig(device="cuda")
    with pytest.raises(ValueError, match="device.*CUDA.*unavailable"):
        TrainerConfig(device="cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    bare = TrainerConfig(device="cuda")
    assert bare.device == "cuda:1"
    assert torch.device(bare.device) == torch.device("cuda:1")
    explicit = TrainerConfig(device="cuda:0")
    assert explicit.device == "cuda:0"
    assert torch.device(explicit.device) == torch.device("cuda:0")
    with pytest.raises(ValueError, match="device.*index 2.*unavailable"):
        TrainerConfig(device="cuda:2")


def test_recursive_mapping_proxy_transfer_preserves_every_tensor_and_dtype() -> None:
    case = make_trainer_case()
    source = mapping_proxy_batch(collate_examples(
        case.validation[:2], case.model_config.horizon_turns
    ))
    source_tensors = tuple(iter_batch_tensors(source))
    expected_count = (
        sum(
            2 + len(table.categorical) + len(table.boolean)
            for table in source.tables.values()
        )
        + len(dataclasses.fields(RelationNeighborhoodBatch))
        + len(dataclasses.fields(CandidateBatch))
        + sum(
            isinstance(getattr(source, field_info.name), Tensor)
            for field_info in dataclasses.fields(RaggedBatch)
        )
    )
    assert len(source_tensors) == expected_count
    snapshots = tuple(value.clone() for value in source_tensors)
    moved = training._batch_to_device(source, torch.device("cpu"))
    moved_tensors = tuple(iter_batch_tensors(moved))
    assert type(moved.tables) is MappingProxyType
    assert type(moved.table_slices) is MappingProxyType
    assert moved.table_slices == source.table_slices
    assert moved.table_slices is not source.table_slices
    assert all(
        type(table.categorical) is MappingProxyType
        and type(table.boolean) is MappingProxyType
        for table in moved.tables.values()
    )
    with pytest.raises(TypeError):
        moved.tables["cells"] = moved.tables["cells"]  # type: ignore[index]
    assert len(source_tensors) == len(moved_tensors)
    for original, snapshot, transferred in zip(
        source_tensors, snapshots, moved_tensors, strict=True
    ):
        assert original.device.type == "cpu"
        assert transferred.device.type == "cpu"
        assert original.dtype == transferred.dtype
        assert torch.equal(original, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("requested", ("cuda", "cuda:0"))
def test_cuda_training_uses_canonical_index_for_all_tensors_and_selection(
    requested: str,
) -> None:
    expected_index = torch.cuda.current_device() if requested == "cuda" else 0
    canonical = f"cuda:{expected_index}"
    case = make_trainer_case(
        device=requested, max_epochs=1, patience_epochs=1
    )
    assert case.trainer_config.device == canonical
    device = torch.device(case.trainer_config.device)
    assert device == torch.device(canonical)
    assert device.type == "cuda" and device.index == expected_index
    result = run_training_case(case)
    source = mapping_proxy_batch(collate_examples(
        case.validation[:1], case.model_config.horizon_turns
    ))
    batch = training._batch_to_device(source, device)
    assert all(tensor.device == device for tensor in iter_batch_tensors(batch))
    assert all(tensor.device.type == "cpu" for tensor in iter_batch_tensors(source))
    assert all(
        parameter.device == device and parameter.dtype == torch.float32
        for parameter in result.model.parameters()
    )
    output = result.model(batch)
    assert all(
        getattr(output, field_info.name).device == device
        for field_info in dataclasses.fields(PolicyOutput)
    )
    selected = result.model.select(batch)[0]
    legal = {
        (
            int(batch.candidates.decision_id[0, row]),
            int(batch.candidates.candidate_id[0, row]),
        )
        for row in torch.nonzero(
            batch.candidates.mask[0], as_tuple=False
        ).flatten()
    }
    assert (selected.decision_id, selected.candidate_id) in legal
def test_reversed_inputs_preserve_history_state_logits_actions_and_trace() -> None:
    case = make_trainer_case(max_epochs=2, patience_epochs=2)
    left_trace = TrainingTrace()
    right_trace = TrainingTrace()
    with capture_training_trace(left_trace):
        left = run_training_case(case)
    reversed_case = dataclasses.replace(
        case,
        train=tuple(reversed(case.train)),
        validation=tuple(reversed(case.validation)),
    )
    with capture_training_trace(right_trace):
        right = run_training_case(reversed_case)
    batch = collate_examples(case.validation, case.model_config.horizon_turns)
    assert left.history == right.history
    assert_state_dict_equal(left.model.state_dict(), right.model.state_dict())
    torch.testing.assert_close(
        left.model(batch).candidate_logits,
        right.model(batch).candidate_logits,
        rtol=0.0,
        atol=0.0,
    )
    assert left.model.select(batch) == right.model.select(batch)
    assert left_trace.optimizer_orders == right_trace.optimizer_orders
    assert left_trace.validation_orders == right_trace.validation_orders


def test_empty_duplicate_and_cross_split_keys_fail_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_trainer_case()

    def forbid_model_construction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("model constructed before split validation")

    monkeypatch.setattr(
        training.TacticalV3Policy, "__init__", forbid_model_construction
    )
    invalid = (
        (dataclasses.replace(case, train=()), "training split must be non-empty"),
        (dataclasses.replace(case, validation=()), "validation split must be non-empty"),
        (
            dataclasses.replace(
                case, train=(case.train[0], dataclasses.replace(case.train[0]))
            ),
            "duplicate training example",
        ),
        (
            dataclasses.replace(
                case,
                validation=(case.validation[0], dataclasses.replace(case.validation[0])),
            ),
            "duplicate validation example",
        ),
        (
            dataclasses.replace(
                case, validation=(dataclasses.replace(case.train[0]),)
            ),
            "splits overlap",
        ),
    )
    for invalid_case, message in invalid:
        with pytest.raises(ValueError, match=message):
            run_training_case(invalid_case)


def test_real_weighted_validation_changes_ranking_and_restores_epoch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_trainer_case(max_epochs=2, batch_size=2, patience_epochs=2)
    case = dataclasses.replace(base, validation=base.validation[:3])
    script = ((0.0, 1.5), (1.0, 0.0))
    states: dict[int, str] = {}
    observed: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        training,
        "_validation_batch_losses",
        scripted_validation_batch_losses(script, states, observed),
    )
    result = run_training_case(case)
    weighted = (0.5, 2.0 / 3.0)
    batch_means = tuple(sum(epoch_values) / 2.0 for epoch_values in script)
    assert weighted[0] < weighted[1]
    assert batch_means[1] < batch_means[0]
    assert observed == [
        (0, 0, 2), (0, 1, 1),
        (1, 0, 2), (1, 1, 1),
    ]
    assert tuple(
        metric.validation_policy_nll for metric in result.history
    ) == pytest.approx(weighted)
    assert tuple(
        metric.validation["policy"] for metric in result.history
    ) == pytest.approx(weighted)
    assert tuple(metric.improved for metric in result.history) == (True, False)
    assert result.best_epoch == 0
    assert result.best_validation_policy_nll == pytest.approx(weighted[0])
    assert state_dict_sha256(result.model.state_dict()) == states[0]


def test_real_loop_exact_history_patience_and_best_state_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_trainer_case(max_epochs=8, batch_size=2, patience_epochs=3)
    case = dataclasses.replace(base, validation=base.validation[:3])
    nlls = tuple(
        float(torch.tensor(value, dtype=torch.float32))
        for value in (0.8, 0.2, 0.4, 0.4, 0.4)
    )
    script = tuple((value, value) for value in nlls)
    states: dict[int, str] = {}
    observed: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        training,
        "_validation_batch_losses",
        scripted_validation_batch_losses(script, states, observed),
    )
    result = run_training_case(case)
    assert tuple(
        metric.validation_policy_nll for metric in result.history
    ) == nlls
    assert tuple(metric.validation["policy"] for metric in result.history) == nlls
    assert tuple(metric.improved for metric in result.history) == (
        True, True, False, False, False,
    )
    assert tuple(metric.epoch for metric in result.history) == (0, 1, 2, 3, 4)
    assert result.best_epoch == 1
    assert result.best_validation_policy_nll == nlls[1]
    assert result.stopped_early
    assert state_dict_sha256(result.model.state_dict()) == states[1]


def test_padded_negative_inf_control_and_full_finite_fault_matrix() -> None:
    assert_valid_padded_negative_infinity_case()
    matrix = (
        ("valid_logit_nan", "candidate_logits", 0),
        ("valid_logit_neg_inf", "candidate_logits", 0),
        ("outcome", "outcome_logits", 0),
        ("horizon", "horizon_logits", 0),
        ("remaining", "remaining_turns", 0),
        ("policy", "loss.policy", 0),
        ("outcome_loss", "loss.outcome", 0),
        ("horizon_loss", "loss.horizon", 0),
        ("remaining_loss", "loss.remaining_turns", 0),
        ("total", "loss.total", 0),
        ("mask", "candidate_mask", 0),
        ("gradient", "gradient=", 0),
        ("clip", "gradient_norm", 0),
        ("parameter", "parameter=", 1),
    )
    for stage, field_name, expected_steps in matrix:
        fault = run_fault_case(stage)
        assert isinstance(fault.error, FloatingPointError)
        assert f"epoch=0 batch=0 {field_name}" in str(fault.error)
        assert fault.optimizer_steps == expected_steps
        if expected_steps == 0:
            assert fault.after_state_sha256 == fault.before_state_sha256
        else:
            assert fault.after_state_sha256 != fault.before_state_sha256


def test_epoch_metrics_are_exact_immutable_plain_float_maps() -> None:
    metric = run_training_case(make_trainer_case(
        max_epochs=1, patience_epochs=1
    )).history[0]
    assert type(metric.epoch) is int
    assert type(metric.improved) is bool
    assert type(metric.train) is MappingProxyType
    assert type(metric.validation) is MappingProxyType
    assert tuple(metric.train) == METRIC_KEYS
    assert tuple(metric.validation) == METRIC_KEYS
    assert all(
        type(value) is float and math.isfinite(value)
        for value in (*metric.train.values(), *metric.validation.values())
    )
    assert type(metric.validation_policy_nll) is float
    assert metric.validation_policy_nll == metric.validation["policy"]
    with pytest.raises(TypeError):
        metric.train["policy"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        metric.validation["policy"] = 0.0  # type: ignore[index]


def test_sub_tolerance_validation_improvement_does_not_replace_best_epoch() -> None:
    case = make_trainer_case(max_epochs=2, patience_epochs=1)
    script = ((1e-13,), (0.0,))
    states: dict[int, str] = {}
    observed: list[tuple[int, int, int]] = []
    original = training._validation_batch_losses
    training._validation_batch_losses = scripted_validation_batch_losses(
        script, states, observed,
    )
    try:
        result = run_training_case(case)
    finally:
        training._validation_batch_losses = original

    assert tuple(metric.improved for metric in result.history) == (True, False)
    assert tuple(metric.epoch for metric in result.history) == (0, 1)
    assert result.best_epoch == 0
    assert result.stopped_early is True
    assert state_dict_sha256(result.model.state_dict()) == states[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_snapshot_state_is_detached_cpu_clone_independent_of_cuda_model() -> None:
    model = TacticalV3Policy(TacticalV3ModelConfig()).to("cuda:0")
    snapshot = training._snapshot_state(model)
    name, source = next(iter(model.state_dict().items()))
    copied = snapshot[name]

    assert copied.device.type == "cpu"
    assert copied.requires_grad is False
    before = copied.clone()
    with torch.no_grad():
        source.reshape(-1)[0].add_(1.0)
    assert torch.equal(copied, before)


def test_restore_state_rejects_missing_and_unexpected_keys() -> None:
    model = TacticalV3Policy(TacticalV3ModelConfig())
    state = dict(training._snapshot_state(model))
    missing = dict(state)
    missing.pop(next(iter(missing)))
    with pytest.raises(RuntimeError, match="Missing key"):
        training._restore_state(model, missing, torch.device("cpu"))

    unexpected = dict(state)
    unexpected["unexpected"] = next(iter(state.values())).clone()
    with pytest.raises(RuntimeError, match="Unexpected key"):
        training._restore_state(model, unexpected, torch.device("cpu"))


def test_snapshot_state_cpu_source_has_independent_cpu_clone_storage() -> None:
    model = TacticalV3Policy(TacticalV3ModelConfig())
    source = model.state_dict()
    snapshot = training._snapshot_state(model)

    assert type(snapshot) is MappingProxyType
    assert tuple(snapshot) == tuple(source)
    before = {name: value.clone() for name, value in snapshot.items()}
    for name, copied in snapshot.items():
        original = source[name]
        assert copied.device.type == "cpu"
        assert copied.requires_grad is False
        assert copied.is_contiguous()
        assert copied.dtype == original.dtype
        assert copied.shape == original.shape
        assert copied.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()

    with torch.no_grad():
        next(iter(source.values())).reshape(-1)[0].add_(1.0)
    for name, copied in snapshot.items():
        assert torch.equal(copied, before[name])
