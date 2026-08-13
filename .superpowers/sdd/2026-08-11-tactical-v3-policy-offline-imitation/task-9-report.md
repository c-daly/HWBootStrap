# Task 9 report

- Added strict weights-only CPU checkpoints with state hashing, inference replay, and immutable metadata.
- Added atomic unsealed experimental run publication and read-only inventory/manifest validation.
- Added `train` and `validate-run` tactical-v3 CLI handoffs; artifact creation remains publisher-owned.
- Focused verification: `65 passed, 11 deselected` for Task 9 and fast trainer state/interface regressions.
- CUDA availability-dependent device preservation coverage is present and skips when CUDA is unavailable.
- The full `test_tactical_v3_training.py` suite exceeded the 120-second execution cap without emitting a failure; the focused regression gate above completed.
- Publication metadata uses the canonical default objective/trainer settings because `TrainingResult` has no fields for the original configs and the public Task 9 publisher signature supplies no alternate source.

## Fix Round 1 — 2026-08-12

### Review findings fixed

- Extended frozen, slotted `TrainingResult` with the exact
  `model_config`, `objective_config`, and `trainer_config` supplied to
  `train_offline`. The unchanged publisher signature now consumes those
  values and rejects model/config or model/device disagreement. A CLI
  `--seed 0` therefore publishes seed 0, while checkpoint tensors and the
  loaded policy remain CPU-only.
- Changed checkpoint save from metadata repair to strict validation:
  `metadata.model_config` must equal `model.config`, and the supplied
  `metadata.model_state_sha256` must equal the independently computed state
  hash.
- Reauthenticated the corpus DTO during publication under the Task 8
  platform-specific root and file leases. Manifest, train, and validation
  bytes are read once from leased handles, schema/canonical/duplicate-key and
  semantic identity checks are rerun, the byte-derived DTO must equal the
  supplied corpus, stable-handle rereads close the mutation window, and the
  exact authenticated manifest bytes are copied.
- Added strict canonical JSON and JSONL validation for every run artifact:
  duplicate keys, unknown/missing fields, noncanonical bytes, non-built-in
  scalar types, and nonfinite floats are rejected. Metrics require exact
  metric maps, contiguous epochs from zero, trainer-consistent `improved`
  flags, validation policy equality, and a best epoch/NLL consistent with
  `run.json` and the checkpoint.
- Added exact recursive inventory and entry-kind checks, safe relative
  checkpoint containment, POSIX symlink rejection, Windows reparse/junction
  rejection, reparse-parent refusal during publication, and atomic
  no-replace directory publication.
- Completed the normative Task 9 test matrix: recursive plain-value
  checkpoint whitelist, nested missing keys, semantic two-save identity,
  exact CPU logits/actions after real seed-227 training, actual CUDA
  train/publication tolerance, injected atomic cleanup, exact inventory,
  reparse/junction boundaries, containment, and read-only preservation.

### Representative RED evidence

- Clean baseline:
  `C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe -m pytest python/tests/test_tactical_v3_checkpoint.py -q`
  — `7 passed in 21.20s`.
- Scoped executable adversary harness:
  `seed0_result_published_seed=227`;
  `corrupt_metrics=ACCEPTED`;
  `manifest_extra_tamper=ACCEPTED`;
  `schema_version_true_epoch_float=ACCEPTED`;
  `metadata_mismatch_save=ACCEPTED recorded_hash_repaired=True`.
- Provenance/save RED:
  `5 failed, 7 deselected in 9.84s`.
- Corpus authentication RED:
  `5 failed, 1 passed, 12 deselected in 22.09s`.
- Metrics/run strictness RED:
  `20 failed, 4 passed, 18 deselected in 121.01s`.
- Inventory/reparse RED:
  `2 failed, 6 passed, 4 skipped, 42 deselected in 11.02s`; a real
  Windows junction was accepted.
- Recursive checkpoint RED:
  `1 failed, 7 passed, 54 deselected in 24.46s`; an injected tuple was
  accepted.
- Atomic/CUDA/reparse-parent RED:
  `1 failed, 2 passed, 62 deselected in 19.52s`; injected cleanup and real
  CUDA publication already passed, while a real junction parent was
  accepted.

### GREEN and dependency evidence

- Focused provenance/save:
  `5 passed, 7 deselected in 12.62s`.
- Focused corpus authentication:
  `6 passed, 12 deselected in 14.84s`.
- Focused metrics/run strictness:
  `24 passed, 18 deselected in 7.42s`.
- Focused inventory/reparse:
  `8 passed, 4 skipped, 42 deselected in 8.86s`. The real Windows junction
  case executed; ordinary symlink creation was unavailable on this host.
- Focused recursive checkpoint/CPU:
  `8 passed, 54 deselected in 22.43s`.
- Focused atomic/CUDA/reparse parent:
  `3 passed, 62 deselected in 15.93s`; CUDA was available with one device,
  so the real CUDA test executed.
- Full Task 9:
  `61 passed, 4 skipped in 79.83s`.
- Final fresh Task 9 verification after the complete diff and CLI provenance
  assertion:
  `61 passed, 4 skipped in 81.95s`.
- Full Task 8 trainer dependency:
  `69 passed in 252.60s`.
- Full Task 8 corpus dependency:
  `26 passed in 39.95s`.
- CLI smoke:
  `python/run_tactical_v3_imitation.py validate-run --help` exited 0 and
  exposed required `--run-dir`.

The earlier report note about publication using silent default
objective/trainer settings is superseded by this fix round.

## Fix Round 2 - lexical ancestor reparse rejection

The scoped re-review found that publication checked only the immediate run
parent and validation checked only the run root and descendants. A junction
above an ordinary immediate parent therefore redirected both operations.

### RED evidence

- Added nested-ancestor publish and validation regressions for Windows
  junctions plus ordinary directory symlinks where host privileges permit.
- On unchanged production, focused `-k nested` produced
  `2 failed, 5 passed, 2 skipped, 60 deselected in 18.03s`. Both real Windows
  junction cases executed and failed because publish and validation accepted
  the redirected run. Ordinary symlink creation was unavailable on this host.

### GREEN evidence

- Added a lexical ancestor-chain guard that walks existing path components
  from the filesystem anchor with `lstat`, without resolving away link or
  reparse evidence. Validation checks through the run root. Publication checks
  the destination parent before writing and again immediately before the
  no-replace publication step.
- Focused `-k nested`:
  `7 passed, 2 skipped, 60 deselected in 13.81s`. Both real Windows nested
  junction tests executed and passed; the publish test also proved the
  physical target run was not created.
- Full Task 9:
  `63 passed, 6 skipped in 67.91s`. The six skips are ordinary-symlink cases
  unavailable under this Windows host privileges; real junction coverage
  executed.

## Fix Round 3 - dot-component ancestor bypass rejection

The scoped re-review found that the lexical ancestor walker stopped at the
first nonexistent raw component. On Windows, a later `..` cancels that
component before filesystem traversal, so
`missing\\..\\junction\\ordinary-parent\\run` bypassed the initial guard.

### RED evidence

- Added direct public-boundary `.`/`..` rejection coverage and executed real
  Windows publish and validation regressions using the bypass path.
- On unchanged production, focused
  `-k "dot_path_components or missing_dotdot"` produced
  `4 failed, 69 deselected in 9.29s`. Validation accepted the junction-backed
  run. Publication crossed the boundary and created its temporary directory
  through the junction before later validation rejected it and cleaned up.

### GREEN evidence

- Public run paths now reject raw `.` or `..` components before conversion to
  `Path` and before any filesystem access. The existing non-resolving `lstat`
  ancestor walk remains responsible for symlink, junction, and reparse
  detection on accepted paths.
- Focused `-k "dot_path_components or missing_dotdot"`:
  `4 passed, 69 deselected in 6.19s`. Both real Windows junction regressions
  executed; publication rejected before creating the physical target run.
- Full Task 9:
  `67 passed, 6 skipped in 69.40s`. The six skips remain the ordinary Windows
  symlink cases unavailable under host privileges; real junction cases
  executed.
