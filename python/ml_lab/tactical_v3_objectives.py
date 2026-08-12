"""Supervised imitation objectives for the tactical-v3 policy heads."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .tactical_v3_batching import RaggedBatch
from .tactical_v3_model import PolicyOutput


@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    policy_coefficient: float = 1.0
    outcome_coefficient: float = 0.2
    horizon_coefficient: float = 0.2
    remaining_turns_coefficient: float = 0.1

    def __post_init__(self) -> None:
        fields = (
            "policy_coefficient",
            "outcome_coefficient",
            "horizon_coefficient",
            "remaining_turns_coefficient",
        )
        for field in fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{field} must be a finite nonnegative number")
            value = float(value)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and nonnegative")
        if self.policy_coefficient <= 0:
            raise ValueError("policy_coefficient must be strictly positive")
        auxiliary = (
            self.outcome_coefficient
            + self.horizon_coefficient
            + self.remaining_turns_coefficient
        )
        if auxiliary > self.policy_coefficient:
            raise ValueError("auxiliary coefficient sum cannot exceed policy_coefficient")


@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    policy: Tensor
    outcome: Tensor
    horizon: Tensor
    remaining_turns: Tensor


def _tensor(value: object, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    return value


def _shape(value: object, expected: tuple[int, ...], name: str) -> Tensor:
    tensor = _tensor(value, name)
    if tensor.shape != expected:
        raise ValueError(f"{name} shape must be {expected}")
    return tensor


def _dtype(value: Tensor, expected: torch.dtype, name: str) -> None:
    if value.dtype != expected:
        raise ValueError(f"{name} dtype must be {expected}")


def _floating(value: Tensor, name: str) -> None:
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")


def _float32(value: Tensor, name: str) -> None:
    if value.dtype != torch.float32:
        raise ValueError(f"{name} dtype must be torch.float32")

def _finite(value: Tensor, name: str, *, allow_negative_infinity: bool = False) -> None:
    if allow_negative_infinity:
        bad = ~torch.isfinite(value) & ~torch.isneginf(value)
    else:
        bad = ~torch.isfinite(value)
    if bool(bad.any()):
        raise FloatingPointError(f"nonfinite values in {name}")


def _validate(
    output: PolicyOutput, batch: RaggedBatch, config: ObjectiveConfig,
) -> tuple[int, int, int]:
    if type(output) is not PolicyOutput:
        raise ValueError("output must be PolicyOutput")
    if type(batch) is not RaggedBatch:
        raise ValueError("batch must be RaggedBatch")
    if type(config) is not ObjectiveConfig:
        raise ValueError("config must be ObjectiveConfig")

    mask = _tensor(batch.candidates.mask, "candidate mask")
    if mask.ndim != 2 or mask.shape[0] <= 0 or mask.shape[1] <= 0:
        raise ValueError("candidate mask shape must be [B, C] with B,C > 0")
    _dtype(mask, torch.bool, "candidate mask")
    batch_size, candidate_count = mask.shape
    if not bool(mask.any(dim=1).all()):
        raise ValueError("candidate mask must contain a valid candidate per sample")
    device = mask.device

    node_mask = _tensor(batch.node_mask, "node_mask")
    if node_mask.ndim != 2 or node_mask.shape[0] != batch_size:
        raise ValueError("node_mask shape must be [B, N]")
    _dtype(node_mask, torch.bool, "node_mask")
    if node_mask.device != device:
        raise ValueError("node_mask must be on the candidate mask device")
    candidate_id_raw = _tensor(batch.candidates.candidate_id, "candidate_id")
    if candidate_id_raw.shape != (batch_size, candidate_count):
        peer_shapes = ((batch.candidates.decision_id, candidate_id_raw.shape), (batch.candidates.kind, candidate_id_raw.shape), (batch.candidates.reference_index, (*candidate_id_raw.shape, 8)), (batch.candidates.reference_mask, (*candidate_id_raw.shape, 8)), (batch.candidates.projection_integer, (*candidate_id_raw.shape, 7)), (batch.candidates.projection_boolean, (*candidate_id_raw.shape, 2)))
        if all(isinstance(value, Tensor) and value.shape == expected for value, expected in peer_shapes):
            raise ValueError("candidate mask shape must agree with candidate fields")
    candidate_fields = (
        ("candidate_id", batch.candidates.candidate_id, torch.int64, (batch_size, candidate_count)),
        ("decision_id", batch.candidates.decision_id, torch.int64, (batch_size, candidate_count)),
        ("kind", batch.candidates.kind, torch.int64, (batch_size, candidate_count)),
        ("reference_index", batch.candidates.reference_index, torch.int64, (batch_size, candidate_count, 8)),
        ("reference_mask", batch.candidates.reference_mask, torch.bool, (batch_size, candidate_count, 8)),
        ("projection_integer", batch.candidates.projection_integer, torch.int64, (batch_size, candidate_count, 7)),
        ("projection_boolean", batch.candidates.projection_boolean, torch.bool, (batch_size, candidate_count, 2)),
    )
    checked_candidates: dict[str, Tensor] = {}
    for name, value, dtype, expected in candidate_fields:
        tensor = _tensor(value, name)
        if tensor.shape != expected:
            raise ValueError(f"{name} shape must be {expected}")
        _dtype(tensor, dtype, name)
        if tensor.device != device:
            raise ValueError(f"{name} must be on the candidate mask device")
        checked_candidates[name] = tensor
    candidate_id = checked_candidates["candidate_id"]
    kind = checked_candidates["kind"]
    reference_index = checked_candidates["reference_index"]
    reference_mask = checked_candidates["reference_mask"]
    projection_integer = checked_candidates["projection_integer"]
    candidate_logits_raw = _tensor(output.candidate_logits, "candidate_logits")
    if candidate_logits_raw.ndim != 2 or candidate_logits_raw.shape[0] != batch_size:
        raise ValueError("candidate_logits shape must be [B, C]")
    if candidate_logits_raw.shape[1] != candidate_count:
        raise ValueError("candidate_logits shape must be [B, C]")
    candidate_logits = candidate_logits_raw
    _float32(candidate_logits, "candidate_logits")
    if candidate_logits.device != device:
        raise ValueError("candidate_logits must be on the candidate mask device")
    _finite(candidate_logits, "candidate_logits", allow_negative_infinity=True)
    if bool((~torch.isfinite(candidate_logits) & mask).any()):
        raise FloatingPointError("nonfinite values in candidate_logits")

    outcome_logits = _shape(output.outcome_logits, (batch_size, 3), "outcome_logits")
    _float32(outcome_logits, "outcome_logits")
    if outcome_logits.device != device:
        raise ValueError("outcome_logits must be on the candidate mask device")
    _finite(outcome_logits, "outcome_logits")

    horizon_targets = _tensor(batch.horizon_targets, "horizon_targets")
    if horizon_targets.ndim != 2 or horizon_targets.shape[0] != batch_size:
        raise ValueError("horizon_targets shape must be [B, H]")
    horizon_count = horizon_targets.shape[1]
    if horizon_count <= 0:
        raise ValueError("horizon count must be positive")
    horizon_mask = _tensor(batch.horizon_target_mask, "horizon_target_mask")
    if horizon_mask.ndim != 2 or horizon_mask.shape[0] != batch_size:
        raise ValueError("horizon_target_mask shape must be [B, H]")
    if horizon_mask.shape[1] != horizon_count:
        if horizon_mask.shape[1] > horizon_count:
            raise ValueError("horizon_targets shape must agree with horizon_target_mask")
        raise ValueError("horizon_target_mask shape must agree with horizon_targets")
    _float32(horizon_targets, "horizon_targets")
    if horizon_targets.device != device:
        raise ValueError("horizon_targets must be on the candidate mask device")
    horizon_logits_raw = _tensor(output.horizon_logits, "horizon_logits")
    if horizon_logits_raw.ndim != 2 or horizon_logits_raw.shape[0] != batch_size:
        raise ValueError("horizon_logits shape must be [B, H]")
    if horizon_logits_raw.shape[1] != horizon_count:
        raise ValueError("horizon count must agree with horizon_logits")
    horizon_logits = horizon_logits_raw
    _float32(horizon_logits, "horizon_logits")
    if horizon_logits.device != device:
        raise ValueError("horizon_logits must be on the candidate mask device")
    _finite(horizon_logits, "horizon_logits")
    horizon_mask = _shape(batch.horizon_target_mask, (batch_size, horizon_count), "horizon_target_mask")
    _dtype(horizon_mask, torch.bool, "horizon_target_mask")
    if horizon_mask.device != device:
        raise ValueError("horizon_target_mask must be on the candidate mask device")

    remaining_targets = _shape(batch.remaining_turns, (batch_size,), "batch.remaining_turns")
    _float32(remaining_targets, "batch.remaining_turns")
    if remaining_targets.device != device:
        raise ValueError("batch.remaining_turns must be on the candidate mask device")
    remaining_logits = _shape(output.remaining_turns, (batch_size,), "remaining_turns")
    _float32(remaining_logits, "remaining_turns")
    if remaining_logits.device != device:
        raise ValueError("remaining_turns must be on the candidate mask device")
    _finite(remaining_logits, "remaining_turns")
    remaining_mask = _shape(batch.remaining_turns_mask, (batch_size,), "remaining_turns_mask")
    _dtype(remaining_mask, torch.bool, "remaining_turns_mask")
    if remaining_mask.device != device:
        raise ValueError("remaining_turns_mask must be on the candidate mask device")

    teacher = _shape(batch.teacher_candidate_index, (batch_size,), "teacher_candidate_index")
    _dtype(teacher, torch.int64, "teacher_candidate_index")
    if teacher.device != device:
        raise ValueError("teacher_candidate_index must be on the candidate mask device")
    outcome = _shape(batch.terminal_outcome, (batch_size,), "terminal_outcome")
    _dtype(outcome, torch.int64, "terminal_outcome")
    if outcome.device != device:
        raise ValueError("terminal_outcome must be on the candidate mask device")

    if bool((teacher == -1).all()) and bool((outcome == -1).all()):
        raise ValueError(
            "target-free batch with teacher_candidate_index=-1 and terminal_outcome=-1"
        )
    if bool(((teacher < 0) | (teacher >= candidate_count)).any()):
        raise ValueError("teacher_candidate_index values are out of range")
    rows = torch.arange(batch_size, device=device)
    if not bool(mask[rows, teacher].all()):
        raise ValueError("teacher_candidate_index selects a padded candidate")
    if bool(((outcome < 0) | (outcome > 2)).any()):
        raise ValueError("terminal_outcome targets must be in 0..2")

    active = mask
    if not bool(((candidate_id[active] >= -(2**31)) & (candidate_id[active] < 2**31)).all()):
        raise ValueError("candidate_id active values are out of int32 range")
    active_kinds = kind[active]
    if not bool(((active_kinds >= 0) & (active_kinds < 4)).all()):
        raise ValueError("kind active values are out of range")
    active_integers = projection_integer[active.unsqueeze(-1).expand_as(projection_integer)]
    if not bool(((active_integers >= -(2**31)) & (active_integers < 2**31)).all()):
        raise ValueError("projection_integer active values are out of int32 range")
    active_references = reference_mask & active.unsqueeze(-1)
    references = reference_index[active_references]
    if references.numel():
        if not bool(((references >= 0) & (references < node_mask.shape[1])).all()):
            raise ValueError("reference_index active values are out of range")
        samples = torch.arange(batch_size, device=device).view(batch_size, 1, 1).expand_as(reference_index)[active_references]
        if not bool(node_mask[samples, references].all()):
            raise ValueError("reference_index active values must select valid nodes")

    _finite(horizon_targets, "horizon_targets")
    if bool(((horizon_targets != 0) & (horizon_targets != 1)).any()):
        raise ValueError("horizon_targets must be binary")
    _finite(remaining_targets, "batch.remaining_turns")
    if bool((remaining_mask & (remaining_targets <= 0)).any()):
        raise ValueError("remaining_turns targets must be positive")
    return batch_size, candidate_count, horizon_count


def _masked_zero(value: Tensor) -> Tensor:
    return value.sum() * 0.0


def structured_imitation_loss(
    output: PolicyOutput, batch: RaggedBatch, config: ObjectiveConfig,
) -> LossBreakdown:
    """Compute the policy and structured auxiliary imitation losses."""
    _validate(output, batch, config)
    mask = batch.candidates.mask

    logits = output.candidate_logits.masked_fill(~mask, float("-inf"))
    policy = F.cross_entropy(logits, batch.teacher_candidate_index)
    outcome = F.cross_entropy(output.outcome_logits, batch.terminal_outcome)

    horizon_mask = batch.horizon_target_mask
    if bool(horizon_mask.any()):
        horizon = F.binary_cross_entropy_with_logits(
            output.horizon_logits[horizon_mask],
            batch.horizon_targets[horizon_mask].to(output.horizon_logits.dtype),
        )
    else:
        horizon = _masked_zero(output.horizon_logits)

    remaining_mask = batch.remaining_turns_mask
    if bool(remaining_mask.any()):
        remaining_turns = F.smooth_l1_loss(
            output.remaining_turns[remaining_mask],
            batch.remaining_turns[remaining_mask].to(output.remaining_turns.dtype),
        )
    else:
        remaining_turns = _masked_zero(output.remaining_turns)

    total = (
        config.policy_coefficient * policy
        + config.outcome_coefficient * outcome
        + config.horizon_coefficient * horizon
        + config.remaining_turns_coefficient * remaining_turns
    )
    for value, name in (
        (policy, "policy"), (outcome, "outcome"), (horizon, "horizon"),
        (remaining_turns, "remaining_turns"), (total, "total"),
    ):
        _finite(value, name)
    return LossBreakdown(total, policy, outcome, horizon, remaining_turns)
