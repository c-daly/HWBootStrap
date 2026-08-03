# Materialized Imitation Sampler Design

**Date:** 2026-08-02

**Status:** Approved design; implementation not started

## Problem

Behavioral cloning is configured correctly for CUDA, but the production GPU is
mostly idle. Process-specific measurements during the interrupted seed-211 run
showed:

- Windows GPU-engine utilization averaging 0.61% and peaking at 1.28%;
- NVIDIA device utilization near 2%, memory-controller utilization near 1%,
  and power near 22 W;
- one CPU core saturated continuously; and
- 44,037 read operations and 100.59 MiB read over eight seconds.

The training worker nevertheless reserved roughly 11.2 GiB of VRAM. Reserved
memory therefore did not indicate productive GPU work.

The hot path is `StratifiedDecisionSampler.next_batch`. It selects physical row
references across many game shards. `ImitationDataset._row_data` then handles
each row separately, uses a two-entry decoded-shard cache, verifies and reopens
evicted shards, copies seven fields row by row, reconstructs NumPy arrays,
unpacks masks, and copies the arrays again. A stratified 256-row batch commonly
touches far more than two of the 1,980 game shards.

The trainer has already materialized the complete training partition before it
constructs this sampler, but the sampler ignores those arrays and returns to the
physical dataset for every optimizer batch. The GPU waits for serial shard I/O,
hashing, decompression, Python iteration, and copying.

The interrupted run was stopped after seed 211 epoch 8 and its unpublished
staging was preserved under the ignored evidence archive
`archive-task12-9e8bd19-low-gpu-interrupted`.

## Goals

1. Remove physical shard access from the optimizer-batch hot path.
2. Preserve the exact physical reference sequence, batch contents, batch order,
   source ratio, seeds, batch size, and optimizer-step sequence.
3. Improve production-data behavioral-cloning throughput by at least 5x over
   the observed approximately 565 examples per second.
4. Report the five defined phase durations to distinguish sampling,
   tensor/forward, optimization, validation, and unclassified costs.
5. Preserve all existing contract, identity, validation, and atomic-publication
   protections.

## Non-Goals

- Changing the model, optimizer, learning rate, batch size, patience, epoch
  limit, source ratio, seeds, or action-mask semantics.
- Reordering examples for device locality.
- Holding the complete training dataset permanently on the GPU.
- Treating high GPU-utilization percentage as the acceptance criterion.
- Weakening exact CPU fixture-logit reload equality.

CUDA floating-point execution may still exhibit its existing small numerical
variation. This design guarantees identical input references and batch arrays,
not bit-identical CUDA weights across machines.

## Chosen Architecture

### Materialized Partition

Introduce a focused immutable representation containing:

- the fully validated `ImitationBatch` arrays for one partition;
- the partition name; and
- an immutable mapping from every `(shard_index, local_row)` physical reference
  to its logical array offset.

Materialization first constructs the canonical logical reference order. It then
groups those references by shard, validates and decodes each physical shard
once, gathers its requested rows vectorially, and scatters them back to their
canonical logical offsets. Resulting arrays must match the existing logical
ordering exactly.

Training and validation receive separate materialized partitions. A validation
reference cannot resolve through the training view.

### Deterministic Reference Scheduling

`_StratumCycler`, its seeded permutations, residual source-ratio calculation,
and the final per-batch permutation remain unchanged. The scheduler continues
to produce the same ordered physical references and source labels.

`StratifiedDecisionSampler.next_batch` translates those references through the
train materialization's offset map. It then uses vectorized NumPy indexing to
gather every array and metadata field. It performs no shard hashing, opening,
decompression, or cache access.

An absent reference, duplicate materialization reference, partition mismatch,
or metadata disagreement is an immediate error. There is no physical-read
fallback from the optimizer loop.

### Trainer Integration

The trainer creates the materialized training and validation partitions once.
The training materialization supplies both `len(training.actions)` and the
sampler's batch source. Validation metrics and actor fixtures use the separate
validation materialization.

Tensor construction and policy calls retain their current dtypes, device,
action masks, and optimizer order. Pinned memory and nonblocking transfer are
deferred until post-fix profiling proves transfer remains material.

## Progress Evidence

Each `bc_epoch` event will retain its existing fields and add accumulated
finite, nonnegative phase durations with these exact names:

- `sampling_seconds`: deterministic scheduling and vectorized batch gathering;
- `transfer_forward_seconds`: tensor transfer and forward/loss construction;
- `optimization_seconds`: backward pass, clipping, and optimizer step;
- `validation_seconds`: validation metrics; and
- `unclassified_seconds`: remaining epoch bookkeeping.

Compute the raw unclassified remainder as `epoch_seconds` minus the other four
phase durations. Reject a remainder below `-1e-6` seconds; clamp only a remainder
within that clock tolerance to zero. The retained validator requires all five
fields to be finite and nonnegative and requires their sum to agree with
`epoch_seconds` using `rel_tol=1e-9` and `abs_tol=1e-6`.

Timing calls must not introduce CUDA synchronization beyond synchronization
already required by the existing loss/metric reads. These measurements diagnose
coarse pipeline phases; they are not kernel-profiler substitutes.

The production panel's retained-history validator and protocol documentation
will recognize and physically validate these fields. This change does not add a
live staging progress file; flushed process stdout remains authoritative during
training.

## Correctness Tests

1. Capture the existing scheduler's golden reference and batch behavior for
   multiple batches spanning both sources, profiles, seats, and action kinds.
2. Require exact equality between legacy physical gathering and materialized
   gathering for observations, masks, actions, identifiers, sources, profiles,
   seats, action kinds, and partitions.
3. After materialization, replace physical shard access with a function that
   raises. `next_batch` must continue to produce valid batches.
4. Reject duplicate or missing references, wrong-partition materializations,
   and inconsistent materialized metadata.
5. Preserve the existing deterministic cycling, undersized-stratum, source
   ratio, dataset-integrity, CPU publication, and exact reload tests.
6. Validate the new timing fields and reject non-finite, negative, or
   internally impossible timing evidence.

## Performance Gate

Run a production-dataset microbenchmark using the accepted interpreter and the
locked batch size. It must:

- exercise exactly 200 consecutive batches after materialization;
- verify that no physical shard access occurs after materialization;
- report examples per second and phase timings; and
- achieve at least 5x the interrupted run's approximately 565 examples per
  second, or at least 2,825 examples per second.

GPU utilization, power, memory, and Windows process-engine counters will be
recorded as diagnostics. They are not hard gates because a small network can
complete bursty kernels quickly even when the input pipeline is healthy.

A one-epoch real-CUDA gate will then report total epoch duration and phase
breakdown before full three-seed training is authorized.

## Verification and Review

Implementation uses TDD. Required gates are focused sampler and imitation
tests, panel tests, the full Python suite, engine tests, GymServer build, exact
batch equivalence, the production-data performance gate, and the one-epoch real
CUDA gate. Independent task review and a final whole-change review must accept
the implementation before evidence is regenerated.

## Evidence Restart

The code revision and source-tree identity will change. After accepted review:

1. move the current `9e8bd19` dataset into the preserved interrupted-run
   archive;
2. run `validate` on the new clean commit;
3. recollect the identity-bound dataset without inspecting live staging;
4. re-run the production-data performance and real-CUDA gates; and
5. start all three production clones only if correctness and the 5x throughput
   requirement pass.

No existing dataset, execution identity, or staging artifact may be relabeled
to the new commit.
