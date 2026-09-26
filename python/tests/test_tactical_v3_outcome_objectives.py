from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from ml_lab.tactical_v3_batching import RaggedBatch, collate_decisions
from ml_lab.tactical_v3_model import PolicyOutput
from ml_lab.tactical_v3_outcome_objectives import (
    ENTROPY_COEFFICIENT,
    OUTCOME_COEFFICIENT,
    outcome_policy_gradient_loss,
    sample_legal_candidates,
)
from tests.tactical_v3_fixture_support import load_tiny_corpus_fixture


def _target_free_batch() -> RaggedBatch:
    corpus = load_tiny_corpus_fixture()
    first = replace(
        corpus.train[0].decision,
        candidates=corpus.train[0].decision.candidates[:2],
    )
    second = replace(
        corpus.validation[0].decision,
        candidates=corpus.validation[0].decision.candidates[:3],
    )
    return collate_decisions((first, second), horizons=(4, 8, 16))


def _choice_and_forced_batches() -> tuple[RaggedBatch, RaggedBatch, RaggedBatch]:
    corpus = load_tiny_corpus_fixture()
    choice = replace(
        corpus.train[0].decision,
        candidates=corpus.train[0].decision.candidates[:2],
    )
    forced = replace(
        corpus.validation[0].decision,
        candidates=corpus.validation[0].decision.candidates[:1],
    )
    return (
        collate_decisions((choice,), horizons=(4, 8, 16)),
        collate_decisions((forced,), horizons=(4, 8, 16)),
        collate_decisions((choice, forced), horizons=(4, 8, 16)),
    )


def _output(batch: RaggedBatch, *, padded_value: float = float("-inf")) -> PolicyOutput:
    batch_size, candidate_count = batch.candidates.mask.shape
    candidate_logits = torch.zeros(batch_size, candidate_count)
    candidate_logits[~batch.candidates.mask] = padded_value
    return PolicyOutput(
        candidate_logits=candidate_logits.requires_grad_(True),
        outcome_logits=torch.zeros(batch_size, 3, requires_grad=True),
        horizon_logits=torch.zeros(batch_size, 3),
        remaining_turns=torch.zeros(batch_size),
    )


def _selected_probability(
    output: PolicyOutput, batch: RaggedBatch, sample: int,
) -> float:
    logits = output.candidate_logits[sample].masked_fill(
        ~batch.candidates.mask[sample], float("-inf")
    )
    return float(torch.softmax(logits.detach(), dim=0)[0].item())


def test_seeded_sampling_is_reproducible_and_never_selects_padding() -> None:
    batch = _target_free_batch()
    assert bool((~batch.candidates.mask).any())
    output = _output(batch, padded_value=1_000_000.0)

    expected = sample_legal_candidates(output, batch, seed=193)
    assert sample_legal_candidates(output, batch, seed=193) == expected
    for seed in range(64):
        sampled = sample_legal_candidates(output, batch, seed=seed)
        for sample, choice in enumerate(sampled):
            assert batch.candidates.mask[sample, choice.candidate_row]
            assert choice.candidate_id == int(
                batch.candidates.candidate_id[sample, choice.candidate_row]
            )
            assert choice.decision_id == int(
                batch.candidates.decision_id[sample, choice.candidate_row]
            )
            assert choice.log_probability <= 0.0
            assert choice.entropy >= 0.0


@pytest.mark.parametrize(
    ("reward", "outcome", "direction"),
    ((1.0, 2, 1), (-1.0, 0, -1)),
)
def test_terminal_result_moves_selected_action_probability_in_expected_direction(
    reward: float, outcome: int, direction: int,
) -> None:
    batch = _target_free_batch()
    output = _output(batch)
    selected_rows = torch.zeros(2, dtype=torch.int64)
    returns = torch.full((2,), reward, dtype=torch.float32)
    outcomes = torch.full((2,), outcome, dtype=torch.int64)
    before = _selected_probability(output, batch, sample=0)
    optimizer = torch.optim.SGD((output.candidate_logits,), lr=0.25)

    losses = outcome_policy_gradient_loss(
        output, batch, selected_rows, returns, outcomes
    )
    optimizer.zero_grad()
    losses.policy.backward()
    optimizer.step()

    after = _selected_probability(output, batch, sample=0)
    assert direction * (after - before) > 0.0


def test_baseline_is_detached_from_policy_but_auxiliary_trains_outcome_head() -> None:
    batch = _target_free_batch()
    output = _output(batch)
    losses = outcome_policy_gradient_loss(
        output,
        batch,
        torch.tensor([0, 1], dtype=torch.int64),
        torch.tensor([1.0, -1.0], dtype=torch.float32),
        torch.tensor([2, 0], dtype=torch.int64),
    )

    baseline_policy_gradient = torch.autograd.grad(
        losses.policy,
        output.outcome_logits,
        retain_graph=True,
        allow_unused=True,
    )[0]
    auxiliary_gradient = torch.autograd.grad(losses.total, output.outcome_logits)[0]

    assert baseline_policy_gradient is None
    assert not losses.detached_advantages.requires_grad
    assert losses.baselines.requires_grad
    assert bool((auxiliary_gradient != 0.0).any())
    expected = (
        losses.policy
        + OUTCOME_COEFFICIENT * losses.outcome
        - ENTROPY_COEFFICIENT * losses.entropy
    )
    torch.testing.assert_close(losses.total, expected)


def test_all_failure_batch_has_nonzero_policy_and_total_gradients() -> None:
    batch = _target_free_batch()
    output = _output(batch)
    losses = outcome_policy_gradient_loss(
        output,
        batch,
        torch.tensor([0, 0], dtype=torch.int64),
        torch.full((2,), -1.0, dtype=torch.float32),
        torch.zeros(2, dtype=torch.int64),
    )

    assert losses.policy.item() != 0.0
    assert losses.total.item() != 0.0
    losses.total.backward()
    assert output.candidate_logits.grad is not None
    assert bool((output.candidate_logits.grad[batch.candidates.mask] != 0.0).any())
    assert output.outcome_logits.grad is not None
    assert bool((output.outcome_logits.grad != 0.0).any())


def test_forced_rows_do_not_dilute_choice_policy_or_entropy_objectives() -> None:
    choice_batch, _forced_batch, mixed_batch = _choice_and_forced_batches()
    choice_losses = outcome_policy_gradient_loss(
        _output(choice_batch),
        choice_batch,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0], dtype=torch.float32),
        torch.tensor([2], dtype=torch.int64),
    )
    mixed_losses = outcome_policy_gradient_loss(
        _output(mixed_batch),
        mixed_batch,
        torch.tensor([0, 0], dtype=torch.int64),
        torch.tensor([1.0, -1.0], dtype=torch.float32),
        torch.tensor([2, 0], dtype=torch.int64),
    )

    assert mixed_losses.choice_mask.tolist() == [True, False]
    torch.testing.assert_close(mixed_losses.policy, choice_losses.policy)
    torch.testing.assert_close(mixed_losses.entropy, choice_losses.entropy)


def test_all_forced_batch_has_zero_actor_objective_but_trains_outcome_head() -> None:
    _choice_batch, forced_batch, _mixed_batch = _choice_and_forced_batches()
    output = _output(forced_batch)
    losses = outcome_policy_gradient_loss(
        output,
        forced_batch,
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([-1.0], dtype=torch.float32),
        torch.tensor([0], dtype=torch.int64),
    )

    assert losses.choice_mask.tolist() == [False]
    assert losses.policy.item() == 0.0
    assert losses.entropy.item() == 0.0
    losses.total.backward()
    assert output.candidate_logits.grad is not None
    assert not bool((output.candidate_logits.grad != 0.0).any())
    assert output.outcome_logits.grad is not None
    assert bool((output.outcome_logits.grad != 0.0).any())
