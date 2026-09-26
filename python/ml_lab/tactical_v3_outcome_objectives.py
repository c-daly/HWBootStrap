"""Bounded on-policy objectives for complete tactical-v3 mini-games.

This module deliberately does not depend on the structured-imitation objective.
Each collated decision receives its complete mini-game's final ``reward.total``
with no discounting.  The resulting REINFORCE update is therefore on-policy:
callers must not replay a rollout after the policy that sampled it has changed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math

import torch
from torch import Tensor
import torch.nn.functional as F

from .tactical_v3_batching import RaggedBatch, validate_ragged_batch
from .tactical_v3_model import PolicyOutput


DISCOUNT_FACTOR = 1.0
OUTCOME_COEFFICIENT = 0.2
ENTROPY_COEFFICIENT = 0.01


@dataclass(frozen=True, slots=True)
class SampledCandidate:
    """One auditable action sampled from a decision's real candidates."""

    decision_id: int
    candidate_id: int
    candidate_row: int
    log_probability: float
    entropy: float


@dataclass(frozen=True, slots=True)
class OutcomeLossBreakdown:
    """Scalar losses plus the per-decision policy-gradient quantities."""

    total: Tensor
    policy: Tensor
    outcome: Tensor
    entropy: Tensor
    selected_log_probabilities: Tensor
    baselines: Tensor
    detached_advantages: Tensor
    choice_mask: Tensor


def _candidate_distribution(
    output: PolicyOutput, batch: RaggedBatch,
) -> tuple[Tensor, Tensor, Tensor]:
    if type(output) is not PolicyOutput:
        raise ValueError("output must be PolicyOutput")
    if type(batch) is not RaggedBatch:
        raise ValueError("batch must be RaggedBatch")
    validate_ragged_batch(batch)

    mask = batch.candidates.mask
    logits = output.candidate_logits
    if not isinstance(logits, Tensor):
        raise ValueError("candidate_logits must be a tensor")
    if logits.shape != mask.shape:
        raise ValueError("candidate_logits shape must agree with candidates.mask")
    if logits.dtype != torch.float32:
        raise ValueError("candidate_logits dtype must be torch.float32")
    if logits.device != mask.device:
        raise ValueError("candidate_logits must be on the candidate mask device")
    if not bool(torch.isfinite(logits[mask]).all()):
        raise FloatingPointError("active candidate logits must be finite")
    padded = logits[~mask]
    if padded.numel() and bool(
        (torch.isnan(padded) | torch.isposinf(padded)).any()
    ):
        raise FloatingPointError(
            "padded candidate logits cannot be NaN or positive infinity"
        )

    # Re-mask even a finite, accidentally attractive padding value.  Padding has
    # exactly zero categorical mass and therefore cannot be sampled.
    masked_logits = logits.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(masked_logits, dim=1)
    log_probabilities = torch.log_softmax(masked_logits, dim=1)
    safe_log_probabilities = torch.where(
        mask, log_probabilities, torch.zeros_like(log_probabilities)
    )
    entropies = -(probabilities * safe_log_probabilities).sum(dim=1)
    if not bool(torch.isfinite(probabilities).all()):
        raise FloatingPointError("candidate probabilities must be finite")
    if not bool(torch.isfinite(entropies).all()):
        raise FloatingPointError("candidate entropies must be finite")
    return probabilities, safe_log_probabilities, entropies


@torch.inference_mode()
def sample_legal_candidates(
    output: PolicyOutput, batch: RaggedBatch, *, seed: int,
) -> tuple[SampledCandidate, ...]:
    """Sample once per decision from the masked categorical policy.

    Reusing a seed with identical inputs on the same device reproduces the
    selected rows.  The returned log probability and entropy are collection-time
    audit values; training recomputes differentiable values from the policy.
    """
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be a nonnegative signed int64")
    probabilities, log_probabilities, entropies = _candidate_distribution(
        output, batch
    )
    generator = torch.Generator(device=probabilities.device)
    generator.manual_seed(seed)
    selected_rows = torch.multinomial(
        probabilities, num_samples=1, replacement=True, generator=generator
    ).squeeze(1)
    samples = torch.arange(probabilities.shape[0], device=probabilities.device)
    if not bool(batch.candidates.mask[samples, selected_rows].all()):
        raise AssertionError("masked categorical selected a padded candidate")

    result: list[SampledCandidate] = []
    for sample, row in enumerate(selected_rows.tolist()):
        result.append(SampledCandidate(
            decision_id=int(batch.candidates.decision_id[sample, row].item()),
            candidate_id=int(batch.candidates.candidate_id[sample, row].item()),
            candidate_row=int(row),
            log_probability=float(log_probabilities[sample, row].item()),
            entropy=float(entropies[sample].item()),
        ))
    return tuple(result)


def _decision_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    batch_size: int,
    device: torch.device,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise ValueError(f"{name} must be a tensor")
    if value.shape != (batch_size,):
        raise ValueError(f"{name} shape must be ({batch_size},)")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype must be {dtype}")
    if value.device != device:
        raise ValueError(f"{name} must be on the candidate mask device")
    return value


def outcome_policy_gradient_loss(
    output: PolicyOutput,
    batch: RaggedBatch,
    selected_candidate_rows: Tensor,
    terminal_reward_totals: Tensor,
    terminal_outcomes: Tensor,
) -> OutcomeLossBreakdown:
    """Compute a gamma-one REINFORCE loss for completed mini-games.

    ``terminal_reward_totals`` must repeat the finalized ``reward.total`` for
    every decision from its mini-game.  Outcome classes use the existing order:
    loss=0, draw=1, win=2.  The outcome-derived baseline is detached only from
    the policy term; cross entropy continues to train the outcome head.
    """
    probabilities, log_probabilities, entropies = _candidate_distribution(
        output, batch
    )
    batch_size = probabilities.shape[0]
    device = probabilities.device
    rows = _decision_tensor(
        selected_candidate_rows,
        name="selected_candidate_rows",
        dtype=torch.int64,
        batch_size=batch_size,
        device=device,
    )
    returns = _decision_tensor(
        terminal_reward_totals,
        name="terminal_reward_totals",
        dtype=torch.float32,
        batch_size=batch_size,
        device=device,
    )
    outcomes = _decision_tensor(
        terminal_outcomes,
        name="terminal_outcomes",
        dtype=torch.int64,
        batch_size=batch_size,
        device=device,
    )
    if not bool(torch.isfinite(returns).all()):
        raise FloatingPointError("terminal_reward_totals must be finite")
    candidate_count = probabilities.shape[1]
    if bool(((rows < 0) | (rows >= candidate_count)).any()):
        raise ValueError("selected_candidate_rows values are out of range")
    samples = torch.arange(batch_size, device=device)
    if not bool(batch.candidates.mask[samples, rows].all()):
        raise ValueError("selected_candidate_rows selects a padded candidate")
    if bool(((outcomes < 0) | (outcomes > 2)).any()):
        raise ValueError("terminal_outcomes values must be in 0..2")

    outcome_logits = output.outcome_logits
    if not isinstance(outcome_logits, Tensor):
        raise ValueError("outcome_logits must be a tensor")
    if outcome_logits.shape != (batch_size, 3):
        raise ValueError("outcome_logits shape must be [B, 3]")
    if outcome_logits.dtype != torch.float32:
        raise ValueError("outcome_logits dtype must be torch.float32")
    if outcome_logits.device != device:
        raise ValueError("outcome_logits must be on the candidate mask device")
    if not bool(torch.isfinite(outcome_logits).all()):
        raise FloatingPointError("outcome_logits must be finite")

    selected_log_probabilities = log_probabilities[samples, rows]
    outcome_probabilities = torch.softmax(outcome_logits, dim=1)
    baselines = (
        -outcome_probabilities[:, 0]
        - outcome_probabilities[:, 1]
        + outcome_probabilities[:, 2]
    )
    detached_advantages = DISCOUNT_FACTOR * returns - baselines.detach()
    choice_mask = batch.candidates.mask.sum(dim=1) > 1
    if bool(choice_mask.any()):
        policy = -(
            detached_advantages[choice_mask]
            * selected_log_probabilities[choice_mask]
        ).mean()
        entropy = entropies[choice_mask].mean()
    else:
        # Preserve a differentiable zero so an all-forced micro-batch can be
        # accumulated normally while contributing no actor gradient.
        policy = selected_log_probabilities.sum() * 0.0
        entropy = entropies.sum() * 0.0
    outcome = F.cross_entropy(outcome_logits, outcomes)
    total = (
        policy
        + OUTCOME_COEFFICIENT * outcome
        - ENTROPY_COEFFICIENT * entropy
    )
    for value, name in (
        (policy, "policy"),
        (outcome, "outcome"),
        (entropy, "entropy"),
        (total, "total"),
    ):
        if (
            value.dtype != torch.float32
            or value.ndim != 0
            or not bool(torch.isfinite(value))
        ):
            raise FloatingPointError(f"{name} loss must be a finite float32 scalar")
    return OutcomeLossBreakdown(
        total=total,
        policy=policy,
        outcome=outcome,
        entropy=entropy,
        selected_log_probabilities=selected_log_probabilities,
        baselines=baselines,
        detached_advantages=detached_advantages,
        choice_mask=choice_mask,
    )


def clip_outcome_gradients(
    parameters: Iterable[Tensor], *, max_norm: float = 1.0,
) -> Tensor:
    """Clip outcome-training gradients and fail on nonfinite norms."""
    if (
        type(max_norm) is not float
        or not math.isfinite(max_norm)
        or max_norm <= 0.0
    ):
        raise ValueError("max_norm must be a finite positive built-in float")
    frozen = tuple(parameters)
    if not frozen or any(
        not isinstance(parameter, Tensor) for parameter in frozen
    ):
        raise ValueError("parameters must be a non-empty iterable of tensors")
    return torch.nn.utils.clip_grad_norm_(
        frozen, max_norm=max_norm, error_if_nonfinite=True
    )
