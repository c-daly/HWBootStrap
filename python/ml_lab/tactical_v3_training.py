"""Deterministic offline training for the tactical-v3 imitation policy."""

from __future__ import annotations

import dataclasses
import math
import random
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
class TrainingResult:
    model: TacticalV3Policy
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig
    best_epoch: int
    best_validation_policy_nll: float
    stopped_early: bool
    history: tuple


METRIC_KEYS = ("total", "policy", "outcome", "horizon", "remaining_turns")


def train_offline(
    train_examples: tuple,
    validation_examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
) -> TrainingResult:
    return _train_offline_impl(
        train_examples,
        validation_examples,
        model_config,
        objective_config,
        trainer_config,
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
) -> tuple[Mapping[str, float], float]:
    weighted = {name: 0.0 for name in METRIC_KEYS}
    example_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(examples), batch_size)):
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
            for name in METRIC_KEYS:
                value = getattr(losses, name)
                if (
                    value.ndim != 0
                    or value.device != device
                    or not bool(torch.isfinite(value))
                ):
                    raise FloatingPointError(f"{context} loss.{name}")
                contribution = float(value.detach().item()) * len(rows)
                if not math.isfinite(contribution):
                    raise FloatingPointError(f"{context} weighted loss.{name}")
                weighted[name] += contribution
            example_count += len(rows)
    metrics = MappingProxyType({
        name: float(weighted[name] / example_count) for name in METRIC_KEYS
    })
    return metrics, metrics["policy"]


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
    model = TacticalV3Policy(model_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, weight_decay=0.0
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed)

    history: list[EpochMetrics] = []
    best_epoch = -1
    best_nll = math.inf
    best_state: Mapping[str, Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(trainer_config.max_epochs):
        permutation = torch.randperm(len(train_rows), generator=generator).tolist()
        train_weighted = {name: 0.0 for name in METRIC_KEYS}
        train_count = 0
        model.train()
        for batch_index, start in enumerate(
            range(0, len(permutation), trainer_config.batch_size)
        ):
            indices = permutation[start:start + trainer_config.batch_size]
            rows = tuple(train_rows[index] for index in indices)
            batch = _batch_to_device(
                _collate_training_batch(rows, model_config.horizon_turns), device
            )
            context = f"epoch={epoch} batch={batch_index}"
            _validate_batch_contract(batch, device, context)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            _validate_policy_output(output, batch, device, context)
            losses = structured_imitation_loss(output, batch, objective_config)
            _validate_losses(losses, device, context)
            losses.total.backward()
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
            for name in METRIC_KEYS:
                contribution = float(getattr(losses, name).detach().item()) * len(rows)
                if not math.isfinite(contribution):
                    raise FloatingPointError(f"{context} weighted loss.{name}")
                train_weighted[name] += contribution
            train_count += len(rows)

        train_metrics = _frozen_metrics(train_weighted, train_count)
        validation_metrics, candidate_nll = _evaluate_validation(
            model,
            validation_rows,
            model_config,
            objective_config,
            trainer_config.batch_size,
            device,
            epoch=epoch,
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
        history.append(EpochMetrics(
            epoch=epoch,
            train=MappingProxyType(dict(train_metrics)),
            validation=MappingProxyType(dict(validation_metrics)),
            validation_policy_nll=candidate_nll,
            improved=improved,
        ))
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
