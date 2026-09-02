"""Deterministic offline training for the tactical-v3 imitation policy."""

from __future__ import annotations

import dataclasses
import math
import random
import re
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

import numpy as np
import torch
from torch import Tensor, nn

from ml_lab.tactical_v3_batching import (
    CandidateBatch,
    RaggedBatch,
    RelationNeighborhoodBatch,
    TokenTableBatch,
    collate_examples,
)
from ml_lab.tactical_v3_corpus import StructuredExample
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import PolicyOutput, TacticalV3Policy
from ml_lab.tactical_v3_objectives import (
    LossBreakdown,
    ObjectiveConfig,
    structured_imitation_loss,
)


def _canonical_device(value: object) -> str:
    if type(value) is not str:
        raise ValueError("device must be a built-in str")
    if value == "cpu":
        return "cpu"
    match = re.fullmatch(r"cuda(?::([0-9]+))?", value)
    if match is None:
        raise ValueError(
            "device must be exactly cpu, cuda, or cuda:<nonnegative decimal index>"
        )
    if not torch.cuda.is_available():
        raise ValueError("device requests CUDA but CUDA is unavailable")
    index = (
        torch.cuda.current_device()
        if match.group(1) is None
        else int(match.group(1))
    )
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"device CUDA index {index} is unavailable")
    return f"cuda:{index}"


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    seed: int = 227
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_epochs: int = 400
    patience_epochs: int = 100
    gradient_clip_norm: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("seed", "batch_size", "max_epochs", "patience_epochs"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                qualifier = "nonnegative" if name == "seed" else "positive"
                raise ValueError(f"{name} must be a {qualifier} built-in int")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be a finite positive built-in float"
                )
        object.__setattr__(self, "device", _canonical_device(self.device))


@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train: Mapping[str, float]
    validation: Mapping[str, float]
    validation_policy_nll: float
    improved: bool


@dataclass(frozen=True, slots=True)
class StepMetrics:
    phase: Literal["train", "validation"]
    epoch: int
    batch_index: int
    global_step: int
    example_count: int
    metrics: Mapping[str, float]


@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: TacticalV3Policy
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig
    best_epoch: int
    best_validation_policy_nll: float
    stopped_early: bool
    history: tuple


@dataclass(frozen=True, slots=True)
class TrainingCheckpointState:
    """Exact state at the end of one fully validated training epoch."""

    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig
    micro_batch_size: int | None
    next_epoch: int
    model_state: Mapping[str, Tensor]
    best_state: Mapping[str, Tensor]
    optimizer_state: Mapping[str, object]
    history: tuple[EpochMetrics, ...]
    best_epoch: int
    best_validation_policy_nll: float
    epochs_without_improvement: int
    train_global_step: int
    validation_global_step: int
    permutation_generator_state: Tensor
    python_random_state: tuple[int, tuple[int, ...], float | None]
    numpy_random_state: tuple[str, Tensor, int, int, float]
    torch_random_state: Tensor
    cuda_random_states: tuple[Tensor, ...]
    uses_external_batch_provider: bool


METRIC_KEYS = ("total", "policy", "outcome", "horizon", "remaining_turns")


def train_offline(
    train_examples: tuple,
    validation_examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
    *,
    epoch_callback: Callable[[EpochMetrics], None] | None = None,
    step_callback: Callable[[StepMetrics], None] | None = None,
    deadline_monotonic: float | None = None,
    initial_state_dict: Mapping[str, Tensor] | None = None,
    training_batch_provider: Callable[[int, int], tuple] | None = None,
    micro_batch_size: int | None = None,
    resume_state: TrainingCheckpointState | None = None,
    checkpoint_callback: Callable[[TrainingCheckpointState], None] | None = None,
) -> TrainingResult:
    return _train_offline_impl(
        train_examples,
        validation_examples,
        model_config,
        objective_config,
        trainer_config,
        epoch_callback=epoch_callback,
        step_callback=step_callback,
        deadline_monotonic=deadline_monotonic,
        initial_state_dict=initial_state_dict,
        training_batch_provider=training_batch_provider,
        micro_batch_size=micro_batch_size,
        resume_state=resume_state,
        checkpoint_callback=checkpoint_callback,
    )


def _canonical_example_key(example: StructuredExample) -> tuple[str, int, int, str, int]:
    if type(example) is not StructuredExample:
        raise TypeError("example must be StructuredExample")
    return (
        example.scenario_id,
        example.episode_seed,
        example.learner_seat,
        example.profile_id,
        example.decision.decision_id,
    )


def _batch_to_device(batch: RaggedBatch, device: torch.device) -> RaggedBatch:
    if type(batch) is not RaggedBatch or type(device) is not torch.device:
        raise TypeError("_batch_to_device requires RaggedBatch and torch.device")

    def move(value: Tensor) -> Tensor:
        return value.to(device=device)

    tables = MappingProxyType({
        name: TokenTableBatch(
            numeric=move(table.numeric),
            categorical=MappingProxyType({
                field_name: move(value)
                for field_name, value in table.categorical.items()
            }),
            boolean=MappingProxyType({
                field_name: move(value)
                for field_name, value in table.boolean.items()
            }),
            mask=move(table.mask),
        )
        for name, table in batch.tables.items()
    })
    neighborhoods = RelationNeighborhoodBatch(
        source_index=move(batch.neighborhoods.source_index),
        kind=move(batch.neighborhoods.kind),
        int_feature=move(batch.neighborhoods.int_feature),
        float_feature=move(batch.neighborhoods.float_feature),
        bool_feature=move(batch.neighborhoods.bool_feature),
        mask=move(batch.neighborhoods.mask),
    )
    candidates = CandidateBatch(
        candidate_id=move(batch.candidates.candidate_id),
        decision_id=move(batch.candidates.decision_id),
        kind=move(batch.candidates.kind),
        reference_index=move(batch.candidates.reference_index),
        reference_mask=move(batch.candidates.reference_mask),
        projection_integer=move(batch.candidates.projection_integer),
        projection_boolean=move(batch.candidates.projection_boolean),
        mask=move(batch.candidates.mask),
    )
    return RaggedBatch(
        tables=tables,
        table_slices=MappingProxyType(dict(batch.table_slices)),
        node_mask=move(batch.node_mask),
        cell_neighbor_index=move(batch.cell_neighbor_index),
        cell_neighbor_mask=move(batch.cell_neighbor_mask),
        neighborhoods=neighborhoods,
        candidates=candidates,
        teacher_candidate_index=move(batch.teacher_candidate_index),
        terminal_outcome=move(batch.terminal_outcome),
        horizon_targets=move(batch.horizon_targets),
        horizon_target_mask=move(batch.horizon_target_mask),
        remaining_turns=move(batch.remaining_turns),
        remaining_turns_mask=move(batch.remaining_turns_mask),
    )


def _after_backward(
    model: TacticalV3Policy, *, epoch: int, batch_index: int,
) -> None:
    del model, epoch, batch_index


def _clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _after_optimizer_step(
    model: TacticalV3Policy, *, epoch: int, batch_index: int,
) -> None:
    del model, epoch, batch_index


def _validation_batch_losses(
    model: TacticalV3Policy,
    batch: RaggedBatch,
    objective_config: ObjectiveConfig,
    *,
    epoch: int,
    batch_index: int,
) -> LossBreakdown:
    device = next(model.parameters()).device
    context = f"epoch={epoch} validation_batch={batch_index}"
    output = model(batch)
    _validate_policy_output(output, batch, device, context)
    return structured_imitation_loss(output, batch, objective_config)


def _collate_training_batch(examples: tuple, horizons: tuple) -> RaggedBatch:
    return collate_examples(examples, horizons)


def _evaluate_validation(
    model: TacticalV3Policy,
    examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    batch_size: int,
    device: torch.device,
    *,
    epoch: int,
    step_callback: Callable[[StepMetrics], None] | None = None,
    global_step_start: int = 0,
    deadline_monotonic: float | None = None,
) -> tuple[Mapping[str, float], float]:
    weighted = {name: 0.0 for name in METRIC_KEYS}
    example_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(examples), batch_size)):
            _check_training_deadline(deadline_monotonic)
            rows = examples[start:start + batch_size]
            batch = _batch_to_device(
                collate_examples(rows, model_config.horizon_turns), device
            )
            context = f"epoch={epoch} validation_batch={batch_index}"
            _validate_batch_contract(batch, device, context)
            losses = _validation_batch_losses(
                model,
                batch,
                objective_config,
                epoch=epoch,
                batch_index=batch_index,
            )
            _validate_losses(losses, device, context)
            batch_metric_values = {}
            for name in METRIC_KEYS:
                value = getattr(losses, name)
                if (
                    value.ndim != 0
                    or value.device != device
                    or not bool(torch.isfinite(value))
                ):
                    raise FloatingPointError(f"{context} loss.{name}")
                batch_value = float(value.detach().item())
                batch_metric_values[name] = batch_value
                contribution = batch_value * len(rows)
                if not math.isfinite(contribution):
                    raise FloatingPointError(f"{context} weighted loss.{name}")
                weighted[name] += contribution
            example_count += len(rows)
            if step_callback is not None:
                step_callback(StepMetrics(
                    phase="validation",
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=global_step_start + batch_index + 1,
                    example_count=len(rows),
                    metrics=MappingProxyType(batch_metric_values),
                ))
    metrics = MappingProxyType({
        name: float(weighted[name] / example_count) for name in METRIC_KEYS
    })
    return metrics, metrics["policy"]


def _check_training_deadline(deadline_monotonic: float | None) -> None:
    if deadline_monotonic is not None and time.monotonic() >= deadline_monotonic:
        raise TimeoutError("training deadline reached")


def _canonical_split(examples: tuple, label: str) -> tuple[StructuredExample, ...]:
    if type(examples) is not tuple:
        raise TypeError(f"{label} split must be an immutable tuple")
    if not examples:
        raise ValueError(f"{label} split must be non-empty")
    keyed = tuple((_canonical_example_key(example), example) for example in examples)
    keys = tuple(key for key, _ in keyed)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {label} example")
    return tuple(example for _, example in sorted(keyed, key=lambda item: item[0]))


def _named_batch_tensors(value: object, path: str = "batch"):
    if isinstance(value, Tensor):
        yield path, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field_info in dataclasses.fields(value):
            yield from _named_batch_tensors(
                getattr(value, field_info.name), f"{path}.{field_info.name}"
            )
    elif isinstance(value, Mapping):
        for name, nested in value.items():
            yield from _named_batch_tensors(nested, f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            yield from _named_batch_tensors(nested, f"{path}[{index}]")


def _validate_batch_contract(
    batch: RaggedBatch, device: torch.device, context: str,
) -> None:
    if type(batch) is not RaggedBatch:
        raise TypeError(f"{context} batch must be RaggedBatch")
    for path, value in _named_batch_tensors(batch):
        if value.device != device:
            raise ValueError(f"{context} {path} device")
        if value.dtype not in {torch.float32, torch.int64, torch.bool}:
            raise ValueError(f"{context} {path} dtype")
        if path.endswith("mask") and value.dtype != torch.bool:
            field = "candidate_mask" if path == "batch.candidates.mask" else path
            raise FloatingPointError(f"{context} {field}")


def _validate_policy_output(
    output: PolicyOutput, batch: RaggedBatch, device: torch.device, context: str,
) -> None:
    if type(output) is not PolicyOutput:
        raise TypeError(f"{context} output must be PolicyOutput")
    batch_size = int(batch.node_mask.shape[0])
    expected = {
        "candidate_logits": tuple(batch.candidates.mask.shape),
        "outcome_logits": (batch_size, 3),
        "horizon_logits": tuple(batch.horizon_targets.shape),
        "remaining_turns": (batch_size,),
    }
    for name, shape in expected.items():
        value = getattr(output, name)
        if (
            not isinstance(value, Tensor)
            or value.device != device
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
        ):
            raise ValueError(f"{context} {name} contract")
    logits = output.candidate_logits
    valid = batch.candidates.mask
    if not bool(torch.isfinite(logits[valid]).all()):
        raise FloatingPointError(f"{context} candidate_logits")
    if not bool(torch.isneginf(logits[~valid]).all()):
        raise FloatingPointError(f"{context} candidate_logits.padding")
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        if not bool(torch.isfinite(getattr(output, name)).all()):
            raise FloatingPointError(f"{context} {name}")


def _validate_losses(
    losses: LossBreakdown, device: torch.device, context: str,
) -> None:
    if type(losses) is not LossBreakdown:
        raise TypeError(f"{context} losses must be LossBreakdown")
    for name in METRIC_KEYS:
        value = getattr(losses, name)
        if (
            not isinstance(value, Tensor)
            or value.ndim != 0
            or value.device != device
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value))
        ):
            raise FloatingPointError(f"{context} loss.{name}")


def _frozen_metrics(weighted: Mapping[str, float], count: int) -> Mapping[str, float]:
    values = {name: float(weighted[name] / count) for name in METRIC_KEYS}
    if any(
        type(value) is not float or not math.isfinite(value)
        for value in values.values()
    ):
        raise FloatingPointError("epoch metrics are nonfinite")
    return MappingProxyType(values)


def _snapshot_state(model: TacticalV3Policy) -> Mapping[str, Tensor]:
    return MappingProxyType({
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    })


def _cpu_clone_tree(value: object) -> object:
    if isinstance(value, Tensor):
        return value.detach().to(device="cpu").contiguous().clone()
    if isinstance(value, Mapping):
        return {
            key: _cpu_clone_tree(item)
            for key, item in value.items()
        }
    if type(value) is tuple:
        return tuple(_cpu_clone_tree(item) for item in value)
    if type(value) is list:
        return [_cpu_clone_tree(item) for item in value]
    if value is None or type(value) in {str, int, float, bool}:
        return value
    raise TypeError(
        f"training checkpoint contains unsupported {type(value).__name__}"
    )


def _capture_python_random_state() -> tuple[int, tuple[int, ...], float | None]:
    version, values, gaussian = random.getstate()
    if type(version) is not int or type(values) is not tuple:
        raise TypeError("Python random state is not canonical")
    if not all(type(value) is int for value in values):
        raise TypeError("Python random state values are not built-in ints")
    if gaussian is not None and type(gaussian) is not float:
        raise TypeError("Python random Gaussian cache is not a built-in float")
    return version, values, gaussian


def _capture_numpy_random_state() -> tuple[str, Tensor, int, int, float]:
    bit_generator, keys, position, has_gauss, cached_gaussian = np.random.get_state()
    return (
        str(bit_generator),
        torch.from_numpy(keys.copy()).to(device="cpu").contiguous(),
        int(position),
        int(has_gauss),
        float(cached_gaussian),
    )


def _restore_random_states(state: TrainingCheckpointState) -> None:
    random.setstate(state.python_random_state)
    numpy_name, numpy_keys, numpy_position, numpy_has_gauss, numpy_cached = (
        state.numpy_random_state
    )
    np.random.set_state((
        numpy_name,
        numpy_keys.detach().to(device="cpu").contiguous().numpy().copy(),
        numpy_position,
        numpy_has_gauss,
        numpy_cached,
    ))
    torch.set_rng_state(
        state.torch_random_state.detach().to(device="cpu").contiguous()
    )
    if state.cuda_random_states:
        if not torch.cuda.is_available():
            raise ValueError(
                "training resume requires CUDA RNG state but CUDA is unavailable"
            )
        if len(state.cuda_random_states) != torch.cuda.device_count():
            raise ValueError("training resume CUDA RNG device count changed")
        torch.cuda.set_rng_state_all([
            value.detach().to(device="cpu").contiguous()
            for value in state.cuda_random_states
        ])


def _checkpoint_state(
    *,
    model: TacticalV3Policy,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
    micro_batch_size: int | None,
    next_epoch: int,
    best_state: Mapping[str, Tensor],
    history: list[EpochMetrics],
    best_epoch: int,
    best_nll: float,
    epochs_without_improvement: int,
    train_global_step: int,
    validation_global_step: int,
    uses_external_batch_provider: bool,
) -> TrainingCheckpointState:
    optimizer_state = _cpu_clone_tree(optimizer.state_dict())
    if not isinstance(optimizer_state, Mapping):
        raise TypeError("optimizer state must be a mapping")
    device = torch.device(trainer_config.device)
    cuda_states = (
        tuple(
            value.detach().to(device="cpu").contiguous().clone()
            for value in torch.cuda.get_rng_state_all()
        )
        if device.type == "cuda" else ()
    )
    return TrainingCheckpointState(
        model_config=model_config,
        objective_config=objective_config,
        trainer_config=trainer_config,
        micro_batch_size=micro_batch_size,
        next_epoch=next_epoch,
        model_state=_snapshot_state(model),
        best_state=MappingProxyType({
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in best_state.items()
        }),
        optimizer_state=MappingProxyType(dict(optimizer_state)),
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_policy_nll=float(best_nll),
        epochs_without_improvement=epochs_without_improvement,
        train_global_step=train_global_step,
        validation_global_step=validation_global_step,
        permutation_generator_state=(
            generator.get_state().detach().to(device="cpu").contiguous().clone()
        ),
        python_random_state=_capture_python_random_state(),
        numpy_random_state=_capture_numpy_random_state(),
        torch_random_state=(
            torch.get_rng_state().detach().to(device="cpu").contiguous().clone()
        ),
        cuda_random_states=cuda_states,
        uses_external_batch_provider=uses_external_batch_provider,
    )


def _validate_resume_state(
    state: TrainingCheckpointState,
    *,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
    micro_batch_size: int | None,
    train_count: int,
    validation_count: int,
    validation_batch_size: int,
) -> None:
    if type(state) is not TrainingCheckpointState:
        raise TypeError("resume_state must be TrainingCheckpointState")
    if (
        state.model_config != model_config
        or state.objective_config != objective_config
        or state.trainer_config != trainer_config
        or state.micro_batch_size != micro_batch_size
    ):
        raise ValueError("training resume configuration changed")
    if state.uses_external_batch_provider:
        raise ValueError("training resume cannot restore an external batch provider")
    if not 1 <= state.next_epoch <= trainer_config.max_epochs:
        raise ValueError("training resume next epoch is invalid")
    if (
        len(state.history) != state.next_epoch
        or tuple(metric.epoch for metric in state.history)
        != tuple(range(state.next_epoch))
        or any(type(metric) is not EpochMetrics for metric in state.history)
    ):
        raise ValueError("training resume history is not contiguous")
    running_best = math.inf
    best: EpochMetrics | None = None
    for metric in state.history:
        expected_improved = (
            metric.validation_policy_nll < running_best - 1e-12
        )
        if metric.improved is not expected_improved:
            raise ValueError("training resume improvement history is inconsistent")
        if expected_improved:
            running_best = metric.validation_policy_nll
            best = metric
    if (
        best is None
        or state.best_epoch != best.epoch
        or state.best_validation_policy_nll != best.validation_policy_nll
    ):
        raise ValueError("training resume best metric is inconsistent")
    trailing = 0
    for metric in reversed(state.history):
        if metric.improved:
            break
        trailing += 1
    if trailing != state.epochs_without_improvement:
        raise ValueError("training resume patience state is inconsistent")
    expected_train_steps = state.next_epoch * math.ceil(
        train_count / trainer_config.batch_size
    )
    expected_validation_steps = state.next_epoch * math.ceil(
        validation_count / validation_batch_size
    )
    if (
        state.train_global_step != expected_train_steps
        or state.validation_global_step != expected_validation_steps
    ):
        raise ValueError("training resume global steps are inconsistent")
    for label, tensors in (
        ("current", state.model_state),
        ("best", state.best_state),
    ):
        if not isinstance(tensors, Mapping) or not tensors:
            raise TypeError(f"training resume {label} model state is invalid")
        if any(
            type(name) is not str
            or not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or not value.is_contiguous()
            for name, value in tensors.items()
        ):
            raise TypeError(
                f"training resume {label} model state must use contiguous CPU tensors"
            )
    if not isinstance(state.optimizer_state, Mapping):
        raise TypeError("training resume optimizer state is invalid")
    for label, value in (
        ("permutation", state.permutation_generator_state),
        ("torch", state.torch_random_state),
        ("numpy", state.numpy_random_state[1]),
    ):
        if (
            not isinstance(value, Tensor)
            or value.device.type != "cpu"
            or not value.is_contiguous()
        ):
            raise TypeError(f"training resume {label} RNG state is invalid")


def _restore_state(
    model: TacticalV3Policy, state: Mapping[str, Tensor], device: torch.device,
) -> None:
    model.load_state_dict(
        {name: value.to(device=device) for name, value in state.items()},
        strict=True,
    )


def _train_offline_impl(
    train_examples: tuple,
    validation_examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
    *,
    epoch_callback: Callable[[EpochMetrics], None] | None,
    step_callback: Callable[[StepMetrics], None] | None,
    deadline_monotonic: float | None,
    initial_state_dict: Mapping[str, Tensor] | None,
    training_batch_provider: Callable[[int, int], tuple] | None,
    micro_batch_size: int | None,
    resume_state: TrainingCheckpointState | None,
    checkpoint_callback: Callable[[TrainingCheckpointState], None] | None,
) -> TrainingResult:
    if type(train_examples) is not tuple:
        raise TypeError("training split must be an immutable tuple")
    if type(validation_examples) is not tuple:
        raise TypeError("validation split must be an immutable tuple")
    if not train_examples:
        raise ValueError("training split must be non-empty")
    if not validation_examples:
        raise ValueError("validation split must be non-empty")
    if type(model_config) is not TacticalV3ModelConfig:
        raise TypeError("model_config must be TacticalV3ModelConfig")
    if type(objective_config) is not ObjectiveConfig:
        raise TypeError("objective_config must be ObjectiveConfig")
    if type(trainer_config) is not TrainerConfig:
        raise TypeError("trainer_config must be TrainerConfig")
    if epoch_callback is not None and not callable(epoch_callback):
        raise TypeError("epoch_callback must be callable")
    if step_callback is not None and not callable(step_callback):
        raise TypeError("step_callback must be callable")
    if initial_state_dict is not None and not isinstance(initial_state_dict, Mapping):
        raise TypeError("initial_state_dict must be a tensor mapping")
    if training_batch_provider is not None and not callable(training_batch_provider):
        raise TypeError("training_batch_provider must be callable")
    if checkpoint_callback is not None and not callable(checkpoint_callback):
        raise TypeError("checkpoint_callback must be callable")
    if initial_state_dict is not None and resume_state is not None:
        raise ValueError("initial state and training resume are mutually exclusive")
    if resume_state is not None and training_batch_provider is not None:
        raise ValueError("training resume cannot restore an external batch provider")
    if micro_batch_size is not None and (
        type(micro_batch_size) is not int
        or micro_batch_size < 1
        or micro_batch_size > trainer_config.batch_size
        or trainer_config.batch_size % micro_batch_size != 0
    ):
        raise ValueError(
            "micro_batch_size must be a positive built-in int that evenly divides "
            "the configured batch_size"
        )
    if deadline_monotonic is not None and (
        type(deadline_monotonic) is not float or not math.isfinite(deadline_monotonic)
    ):
        raise ValueError("deadline_monotonic must be a finite built-in float")
    _check_training_deadline(deadline_monotonic)
    train_rows = _canonical_split(train_examples, "training")
    validation_rows = _canonical_split(validation_examples, "validation")
    train_keys = {_canonical_example_key(example) for example in train_rows}
    validation_keys = {_canonical_example_key(example) for example in validation_rows}
    if train_keys & validation_keys:
        raise ValueError("splits overlap")

    random.seed(trainer_config.seed)
    np.random.seed(trainer_config.seed % (2**32))
    torch_seed = trainer_config.seed % (2**63 - 1)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(trainer_config.device)
    execution_batch_size = (
        trainer_config.batch_size
        if micro_batch_size is None else micro_batch_size
    )
    if resume_state is not None:
        _validate_resume_state(
            resume_state,
            model_config=model_config,
            objective_config=objective_config,
            trainer_config=trainer_config,
            micro_batch_size=micro_batch_size,
            train_count=len(train_rows),
            validation_count=len(validation_rows),
            validation_batch_size=execution_batch_size,
        )
    model = TacticalV3Policy(model_config).to(device=device, dtype=torch.float32)
    if resume_state is not None:
        model.load_state_dict(
            {
                name: value.detach().to(device=device).contiguous().clone()
                for name, value in resume_state.model_state.items()
            },
            strict=True,
        )
    elif initial_state_dict is not None:
        copied_state = {
            name: value.detach().to(device=device).contiguous().clone()
            if isinstance(value, Tensor) else value
            for name, value in initial_state_dict.items()
        }
        model.load_state_dict(copied_state, strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, weight_decay=0.0
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed)

    if resume_state is None:
        history: list[EpochMetrics] = []
        best_epoch = -1
        best_nll = math.inf
        best_state: Mapping[str, Tensor] | None = None
        epochs_without_improvement = 0
        stopped_early = False
        train_global_step = 0
        validation_global_step = 0
        start_epoch = 0
    else:
        optimizer.load_state_dict(dict(resume_state.optimizer_state))
        generator.set_state(resume_state.permutation_generator_state)
        history = list(resume_state.history)
        best_epoch = resume_state.best_epoch
        best_nll = resume_state.best_validation_policy_nll
        best_state = MappingProxyType({
            name: value.detach().to(device="cpu").contiguous().clone()
            for name, value in resume_state.best_state.items()
        })
        epochs_without_improvement = resume_state.epochs_without_improvement
        stopped_early = (
            epochs_without_improvement >= trainer_config.patience_epochs
        )
        train_global_step = resume_state.train_global_step
        validation_global_step = resume_state.validation_global_step
        start_epoch = resume_state.next_epoch
        _restore_random_states(resume_state)

    for epoch in range(start_epoch, trainer_config.max_epochs):
        if stopped_early:
            break
        _check_training_deadline(deadline_monotonic)
        permutation = (
            torch.randperm(len(train_rows), generator=generator).tolist()
            if training_batch_provider is None else None
        )
        train_weighted = {name: 0.0 for name in METRIC_KEYS}
        train_count = 0
        model.train()
        batch_count = math.ceil(len(train_rows) / trainer_config.batch_size)
        for batch_index in range(batch_count):
            _check_training_deadline(deadline_monotonic)
            if training_batch_provider is None:
                start = batch_index * trainer_config.batch_size
                indices = permutation[start:start + trainer_config.batch_size]
                rows = tuple(train_rows[index] for index in indices)
            else:
                rows = training_batch_provider(epoch, batch_index)
                if type(rows) is not tuple or not rows:
                    raise TypeError("training batch provider must return a nonempty tuple")
                if len(rows) > trainer_config.batch_size:
                    raise ValueError("training batch provider exceeded configured batch size")
                for row in rows:
                    if _canonical_example_key(row) not in train_keys:
                        raise ValueError(
                            "training batch provider returned a row outside training split"
                        )
            context = f"epoch={epoch} batch={batch_index}"
            optimizer.zero_grad(set_to_none=True)
            batch_weighted = {name: 0.0 for name in METRIC_KEYS}
            for micro_index, start in enumerate(
                range(0, len(rows), execution_batch_size)
            ):
                micro_rows = rows[start:start + execution_batch_size]
                micro_context = f"{context} micro_batch={micro_index}"
                batch = _batch_to_device(
                    _collate_training_batch(
                        micro_rows, model_config.horizon_turns,
                    ),
                    device,
                )
                _validate_batch_contract(batch, device, micro_context)
                output = model(batch)
                _validate_policy_output(output, batch, device, micro_context)
                losses = structured_imitation_loss(
                    output, batch, objective_config,
                )
                _validate_losses(losses, device, micro_context)
                (
                    losses.total * (len(micro_rows) / len(rows))
                ).backward()
                for name in METRIC_KEYS:
                    value = float(getattr(losses, name).detach().item())
                    contribution = value * len(micro_rows)
                    if not math.isfinite(contribution):
                        raise FloatingPointError(
                            f"{micro_context} weighted loss.{name}"
                        )
                    batch_weighted[name] += contribution
            _after_backward(model, epoch=epoch, batch_index=batch_index)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(f"{context} gradient={name}")
            gradient_norm = _clip_grad_norm(
                model.parameters(), trainer_config.gradient_clip_norm
            )
            if (
                not isinstance(gradient_norm, Tensor)
                or gradient_norm.ndim != 0
                or gradient_norm.device != device
                or not bool(torch.isfinite(gradient_norm))
            ):
                raise FloatingPointError(f"{context} gradient_norm")
            optimizer.step()
            _after_optimizer_step(model, epoch=epoch, batch_index=batch_index)
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(f"{context} parameter={name}")
            batch_metric_values = {
                name: batch_weighted[name] / len(rows)
                for name in METRIC_KEYS
            }
            for name in METRIC_KEYS:
                train_weighted[name] += batch_weighted[name]
            train_count += len(rows)
            train_global_step += 1
            if step_callback is not None:
                step_callback(StepMetrics(
                    phase="train",
                    epoch=epoch,
                    batch_index=batch_index,
                    global_step=train_global_step,
                    example_count=len(rows),
                    metrics=MappingProxyType(batch_metric_values),
                ))
            _check_training_deadline(deadline_monotonic)

        train_metrics = _frozen_metrics(train_weighted, train_count)
        validation_arguments = {"epoch": epoch}
        if step_callback is not None:
            validation_arguments.update(
                step_callback=step_callback,
                global_step_start=validation_global_step,
            )
        if deadline_monotonic is not None:
            validation_arguments["deadline_monotonic"] = deadline_monotonic
        validation_metrics, candidate_nll = _evaluate_validation(
            model,
            validation_rows,
            model_config,
            objective_config,
            execution_batch_size,
            device,
            **validation_arguments,
        )
        validation_global_step += math.ceil(
            len(validation_rows) / execution_batch_size
        )
        if type(candidate_nll) is not float or not math.isfinite(candidate_nll):
            raise FloatingPointError(f"epoch={epoch} validation policy")
        improved = candidate_nll < best_nll - 1e-12
        if improved:
            best_epoch = epoch
            best_nll = candidate_nll
            best_state = _snapshot_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        metric = EpochMetrics(
            epoch=epoch,
            train=MappingProxyType(dict(train_metrics)),
            validation=MappingProxyType(dict(validation_metrics)),
            validation_policy_nll=candidate_nll,
            improved=improved,
        )
        history.append(metric)
        if best_state is None:
            raise RuntimeError("completed epoch did not produce a best state")
        checkpoint_state = _checkpoint_state(
            model=model,
            optimizer=optimizer,
            generator=generator,
            model_config=model_config,
            objective_config=objective_config,
            trainer_config=trainer_config,
            micro_batch_size=micro_batch_size,
            next_epoch=epoch + 1,
            best_state=best_state,
            history=history,
            best_epoch=best_epoch,
            best_nll=best_nll,
            epochs_without_improvement=epochs_without_improvement,
            train_global_step=train_global_step,
            validation_global_step=validation_global_step,
            uses_external_batch_provider=training_batch_provider is not None,
        )
        try:
            if checkpoint_callback is not None:
                checkpoint_callback(checkpoint_state)
            if epoch_callback is not None:
                epoch_callback(metric)
        finally:
            # Checkpoint generation constructs validation models and may otherwise
            # consume global RNG. An observability callback must not change the
            # deterministic optimization trajectory.
            _restore_random_states(checkpoint_state)
        if epochs_without_improvement >= trainer_config.patience_epochs:
            stopped_early = True
            break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("training did not produce a best state")
    _restore_state(model, best_state, device)
    model.eval()
    return TrainingResult(
        model=model,
        model_config=model_config,
        objective_config=objective_config,
        trainer_config=trainer_config,
        best_epoch=best_epoch,
        best_validation_policy_nll=float(best_nll),
        stopped_early=stopped_early,
        history=tuple(history),
    )
