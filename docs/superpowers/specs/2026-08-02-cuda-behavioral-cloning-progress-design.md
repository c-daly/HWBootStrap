# CUDA Behavioral Cloning and Progress Logging Design

**Date:** 2026-08-02  
**Status:** Approved in conversation; written-spec review pending  
**Parent experiment:** Annihilation imitation v1

## Problem

The full behavioral-cloning stage constructed the production MaskablePPO policy
with `device="cpu"` even when CUDA was available. On the current machine,
PyTorch detects an NVIDIA GeForce RTX 5070, but the first clone consumed more
than forty minutes of CPU computation without completing an epoch-visible
artifact.

The CPU choice came from conflating two requirements:

1. behavioral-cloning optimization must run reproducibly with recorded
   provenance; and
2. the published actor must reload on CPU and reproduce its fixture logits
   exactly.

Only the second requirement mandates CPU. Training the actor on CPU is not
required by the approved experiment plan.

The same run exposed an observability defect. The trainer produced no stdout
progress and wrote no clone artifact until a complete seed finished training.
A healthy, compute-intensive run was therefore indistinguishable from a hung
run without inspecting process metadata.

## Goals

- Train the three behavioral clones on an explicitly locked CUDA device.
- Fail before optimizer work when the requested accelerator is unavailable.
- Preserve exact CPU save/reload verification for published clone artifacts.
- Emit useful, flushed progress after every epoch.
- Retain compact epoch history and training-device provenance in each clone.
- Keep dataset composition, model seeds, sampler seeds, architecture,
  optimizer, epoch/patience limits, clone gate, PPO protocol, and final gate
  unchanged.

## Non-goals

- Do not introduce mixed precision, distributed training, gradient
  accumulation, new hyperparameters, or performance-driven algorithm changes.
- Do not weaken execution-identity or dataset-revision checks.
- Do not relabel the dataset collected under an earlier commit.
- Do not make CUDA selection an unrecorded CLI or environment override.
- Do not require bitwise-identical optimization trajectories across different
  GPU models or PyTorch/CUDA versions.

## Considered approaches

### 1. Locked CUDA training with CPU publication verification ? selected

Add `device: "cuda"` to the immutable behavioral-cloning panel definition and
thread it through `BehavioralCloningConfig`. Resolve and validate CUDA before
model construction or optimizer work. Train on CUDA, then move the selected
actor to CPU before computing fixture logits, saving, reloading, and performing
the exact equality check.

This is fail-closed, fast, and fully represented in experiment provenance.

### 2. Automatic CUDA/CPU selection

Select CUDA when available and otherwise fall back to CPU. This is convenient,
but two invocations of the same definition could silently execute materially
different compute methods. It is unsuitable for the locked research panel.

### 3. Runtime CLI device override

Expose a `--device` argument without changing the panel definition. This is
flexible, but the device would not participate in definition hashes and could
change across resumptions. It violates the identity-bound experiment model.

## Configuration contract

`BehavioralCloningConfig` gains a required, validated `device` field whose
supported values are `"cpu"` and `"cuda"`. The production panel locks
`"cuda"`. Tests and the tiny smoke gate may explicitly use `"cpu"`.

The realized device record contains:

- requested device;
- resolved PyTorch device;
- PyTorch version;
- CUDA runtime version when applicable;
- device name and index when applicable.

A CUDA request fails before model construction when
`torch.cuda.is_available()` is false or no CUDA device exists. There is no
silent CPU fallback.

The requested and realized device records are stored in `bc.json` and are
validated when a completed clone is reused.

## Training and publication data flow

1. Validate panel definitions, execution identity, dataset identity, and the
   requested training device.
2. Construct the production MaskablePPO/HexCNN policy on the requested device.
3. Train only the actor-side parameter groups with the existing optimizer and
   sampler.
4. After each epoch, compute the existing full validation metrics and emit one
   structured progress event.
5. Restore the best actor state.
6. Move the policy to CPU and confirm its actor and value parameters are on
   CPU.
7. Compute fixture logits on CPU.
8. Save the CPU model, fixtures, metrics, epoch history, and provenance into
   the temporary publishing directory.
9. Reload through the production controller resolver on CPU and require exact
   fixture-logit equality.
10. Atomically publish the completed seed directory.

Moving to CPU occurs only after optimization is complete. It does not change
the learned parameter values; it establishes a canonical publication and
verification device.

## Progress interface

`train_behavioral_clone` gains an optional progress callback. The library
emits immutable mappings; the panel owns presentation and prints each mapping
as one compact JSON object with `flush=True`. This keeps the training library
independent of console formatting and makes progress testable.

Each epoch event contains:

- schema version and event kind `bc_epoch`;
- model seed;
- requested and realized device;
- epoch and maximum epochs;
- batches and examples processed;
- mean training loss;
- validation NLL, top-1, top-3, and top-5 accuracy;
- current best epoch and best validation NLL;
- epochs without improvement and patience limit;
- epoch seconds, total elapsed seconds, and examples per second.

A final `bc_complete` event includes the selected best epoch, total epochs,
elapsed time, and publication target. All numeric fields must be finite and
counts must be non-negative.

Progress is written to stdout immediately and is therefore safe to observe
without opening live staging files. The same epoch records are retained in a
small `training-history.json` inside the final clone artifact. No live
append-only file is introduced.

## Failure and restart behavior

A device-validation failure occurs before optimizer work and before any clone
run is published. An interruption during training leaves the existing
identity-bound stage recovery directory. The normal atomic-stage cleanup and
restart behavior remains authoritative; partial clone directories are never
accepted as completed evidence.

The currently interrupted CPU staging directory and the current dataset remain
preserved until the CUDA correction passes tests and independent review. After
the new commit is accepted, both are archived intact. A new clean execution
identity and fresh dataset collection are required because source and panel
definition hashes will change.

## Testing strategy

TDD will cover:

1. `BehavioralCloningConfig` accepts only supported explicit devices.
2. The production panel locks CUDA and passes it to clone training.
3. CUDA unavailability fails before adapter/model/optimizer work.
4. The adapter receives the requested CUDA device.
5. Epoch callbacks contain the required finite fields and are emitted once per
   completed epoch.
6. Panel logging writes one flushed JSON object per event.
7. The best actor is moved to CPU before fixture computation and saving.
8. Published device provenance and training history survive validation and
   completed-run reuse.
9. Exact CPU save/reload fixture equality remains unchanged.
10. The CPU smoke gate remains supported explicitly.
11. Full panel, Python, engine, and GymServer regression gates remain green.

## Research consequence

CUDA optimization is seeded but is not promised to be bitwise identical across
GPU models, drivers, or PyTorch/CUDA releases. Recording the realized hardware
and software stack makes this limitation explicit. The published model remains
canonicalized and verified on CPU, so downstream actor transfer consumes the
same stable artifact format as before.

