# Task 10 Breaker Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two load-bearing Task 10 defects that remained after the original selective-DAgger plan's five-round breaker, without changing the accepted experiment design or beginning Task 11.

**Architecture:** Preserve the existing physical-evidence validators and add two narrow last-mile invariants. Audited contracts compare the duplicated board document with JSON-type-exact semantics. Supervised evaluation captures authenticated Task 9 inventories into one deterministic evaluator-owned, content-addressed bundle and reconstructs all rows and metrics from that retained artifact.

**Tech Stack:** Python 3.11, pytest, immutable JSON publications, `pathlib`, HexWars DAgger physical evidence validators.

## Governing design

This repair implements the already-approved design in `docs/superpowers/specs/2026-08-03-selective-dagger-search-distillation-design.md` plus the approved ownership amendment in `docs/superpowers/specs/2026-08-09-task10-owned-overlay-evidence-design.md`. The original SDD ledger remains historical evidence that Task 10 exhausted its five repair rounds. The user explicitly authorized this new two-defect plan and the evaluator-owned-copy amendment on 2026-08-09; this plan has its own ledger and review breaker.

## Global Constraints

- Do not modify the Task 8/Task 11 trust boundary: production Task 8 evidence remains fail-closed until Task 11 supplies the sealed-engine adapter.
- Do not begin Task 11 until both tasks below pass task review, combined verification, and an independent whole-plan review.
- Preserve arbitrary valid board dimensions, unit templates, rosters, and scenario configuration; never pin the retained baseline's concrete values as general rules.
- JSON equality at the audited boundary is type-sensitive: booleans are not integers, and integer/float aliases are distinct unless the schema explicitly normalizes them.
- The supervised evaluator owns one deterministic bundle containing every cumulative held-out overlay inventory. Its relative filename includes its SHA-256; manifest and evidence schema version 2 bind its path, hash, size, ordered source provenance, overlay content identities, and archive prefixes.
- Build the bundle only from authenticated in-memory tree snapshots. Reopen it through a strict archive decoder and the existing first-party DAgger overlay opener; predictions, metrics, reuse, and aggregate evidence use only owned rows.
- Reuse and aggregate physical reopening must not require original Task 9 roots to exist. Caller-supplied source roots remain ordered provenance claims, not physical authorities.
- The archive and publication enforce exact inventory, containment, no duplicate/absolute/parent-traversal entries, and no symlink/reparse encodings. Windows symbolic-link tests may skip only for WinError 1314.
- Atomic rename makes the supervised publication the sole authorized bundle writer. A direct external write is corruption detected on the next open; the reader returns a hash-bound byte snapshot and does not claim to prevent writes after its last read.
- Use strict TDD: each production change follows a focused failing regression whose failure is observed and recorded.
- Run Python verification only through the pinned environment:

  ```powershell
  $env:VIRTUAL_ENV='C:\Users\cddal\HexWars\python\winenv'
  $env:UV_CACHE_DIR=(Resolve-Path '.uv-cache').Path
  $env:PYTHONDONTWRITEBYTECODE='1'
  uv run --active --no-project python -m pytest ...
  ```

- Do not push or create a PR. Do not add attribution trailers.

---

### Task 1: Make duplicated board validation JSON-type-exact

**Files:**
- Modify: `python/ml_lab/checkpoint_audit.py:624-745`
- Test: `python/tests/test_checkpoint_audit.py:529`

**Interfaces:**
- Consumes: `_validate_audited_semantics(value, *, board, roster, action_size, observation_size, environment_kind, label)` and the already strictly validated top-level `board` mapping.
- Produces: the same validated semantics mapping; no public signature or publication schema changes.

- [ ] **Step 1: Write the failing regression**

  Add a test beside `test_audited_baseline_validator_rejects_mutated_real_contract_board`. Copy the retained locked baseline, change only `run["contract"]["semantics"]["board"]["score_kills"]` from integer `1` to boolean `True`, recompute the documented contract digest exactly as the fixture's production format requires, and leave the top-level board value as integer `1`. Call `validate_audited_baseline_publication` and require `ValueError` matching `board|contract|semantics`.

  The expected result is derived from JSON's scalar model: `true` and `1` are different values even though Python's ordinary mapping equality aliases them.

- [ ] **Step 2: Run the regression and verify RED**

  Run:

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_checkpoint_audit.py::test_audited_baseline_validator_rejects_nested_board_bool_integer_alias -q
  ```

  Expected: FAIL with `DID NOT RAISE ValueError`. A fixture/setup error is not an acceptable RED.

- [ ] **Step 3: Implement the minimal type-exact comparison**

  Add a private recursive JSON comparator near the audited schema helpers, or validate the nested board through the same strict board validator and compare canonical typed structures. It must distinguish at least `type(left) is type(right)` at scalar leaves, require identical mapping key sets and recursively compare values, and require equal list lengths/order. Use it for `semantics["board"]` versus the validated top-level `board`. Do not use `json.dumps` alone as a substitute for schema validation, and do not hardcode the retained board's dimensions.

- [ ] **Step 4: Verify GREEN and adjacent behavior**

  Run the focused regression, the retained-real-baseline acceptance test, the prior board-drift test, and the audited-baseline validator matrix:

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_checkpoint_audit.py -k "nested_board_bool_integer_alias or accepts_real_locked_step_38912_run or mutated_real_contract_board or audited_baseline_validator" -q
  ```

  Expected: all selected tests pass; the real locked configurable contract remains accepted.

- [ ] **Step 5: Run the full checkpoint-audit suite**

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_checkpoint_audit.py -q
  ```

  Expected: PASS with no new skips or warnings.

- [ ] **Step 6: Self-review and commit**

  Confirm the regression would fail if the new type-exact comparison were replaced by `==`, run `git diff --check`, then commit only the Task 1 production/test files:

  ```powershell
  git add python/ml_lab/checkpoint_audit.py python/tests/test_checkpoint_audit.py
  git commit -m "fix: enforce type-exact audited board identity"
  ```

---

### Task 2: Publish evaluator-owned content-addressed overlay evidence

**Files:**
- Modify: `python/run_annihilation_selective_dagger.py:1960-2700,4150-4243,4470-4507`
- Test: `python/tests/test_annihilation_selective_dagger.py:6700-7350`

**Interfaces:**
- Consumes: authenticated `DevelopmentHeldoutOverlayEvidence` objects containing exact source tree bytes, `dagger_domain.open_dagger_overlay`, and `dagger_domain.dagger_overlay_supervised_examples`.
- Produces: schema-2 `DevelopmentSupervisedEvidence` bound to a contained `owned-overlays/<sha256>.zip` artifact, with ordered source-root provenance and unchanged overlay content-identity prefix.

- [ ] **Step 1: Replace the impossible race assertion with ownership REDs**

  Retain the two source-race tests as historical evidence of why foreign-root stability is impossible, but change their desired contract to owned-copy isolation. Add:

  ```text
  test_task10_supervised_reopens_from_owned_bundle_after_source_overlays_are_deleted
  test_task10_supervised_owned_bundle_isolated_from_late_source_mutation
  ```

  Publish iteration 2 over real physical overlays, then delete or mutate the original roots. Reopen from the completed supervised publication and require identical prefix, rows-derived metrics, and content identity. Instrument original roots so any reuse/open attempt fails, proving the owned artifact is authoritative.

- [ ] **Step 2: Run the ownership tests and verify RED**

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py -k "owned_bundle_after_source_overlays_are_deleted or owned_bundle_isolated_from_late_source_mutation" -q
  ```

  Expected: fail because schema 1 records and physically reopens the deleted or mutated source roots. A failure before initial publication is not an acceptable RED.

- [ ] **Step 3: Implement the deterministic owned bundle**

  Add a focused internal bundle DTO and writer/opener. The writer uses sorted `tree_directories` and `tree_files`, fixed ZIP metadata, ordered prefixes `<iteration>-<content_identity>/`, and no source rereads. Write a staging bundle, fsync it, hash it, and rename it to `owned-overlays/<sha256>.zip`.

  The opener reads and hashes the contained regular bundle artifact; rejects duplicate, absolute, `..`, undeclared, and reparse entries; manually materializes it under a fresh private temporary directory; and invokes `open_dagger_overlay` plus `dagger_overlay_supervised_examples` for every prefix. Exact-compare content identities, partitions, iterations, action size, rows, and inventory against schema-2 evidence and the frozen definition.

  Advance supervised manifest and evidence schemas to 2. Store source roots only as provenance strings. Bind bundle path, hash, size, ordered prefixes, and source identities into the manifest identity and evidence. Update `DevelopmentSupervisedEvidence`, aggregate validation/snapshots/report payloads, and exact publication inventory. Remove the foreign-root final-stability barrier from successful reopen paths.

  Build and validate the bundle before either predictor call; derive predictor inputs and metrics from owned rows. On reuse and aggregate passes, physically open only the completed publication bundle and local artifacts. Preserve transactional staging, atomic rename, checkpoint/repository probes, and post-rename rollback.

- [ ] **Step 4: Add adversarial bundle verification and verify GREEN**

  Add focused tests for bundle-byte mutation, duplicate or `..` archive entries, unowned top-level entries, bundle symlink/reparse replacement, bundle mutation between aggregate passes, zero-inference reuse, and zero original-root reads. Construct malicious archives independently; do not use the production writer to derive expected structures.

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py -k "owned_bundle or supervised_evaluation_is_transactional or supervised_post_rename or aggregate and supervised" -q
  ```

  Expected: PASS; only an explicit WinError-1314 reparse test may skip.

- [ ] **Step 5: Run bounded Task 10 verification**

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py -k "task10 or development_candidate or development_supervised or development_aggregate" -q
  ```

  Expected: PASS within the existing bounded test budget; only explicitly privilege-gated Windows reparse tests may skip.

- [ ] **Step 6: Self-review and commit**

  Audit that source roots are read only during initial capture, owned rows feed predictions and metrics, reuse and aggregate depend only on the bundle, archive extraction is contained and exact, and the publication is never mutated after atomic rename. Run `git diff --check`, then commit only the Task 2 production/test files:

  ```powershell
  git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py
  git commit -m "fix: own supervised overlay evidence"
  ```

---

## Completion gate

After both task-scoped reviews pass:

1. Run the combined focused A/B regressions and the full `test_checkpoint_audit.py` suite.
2. Run the bounded Task 10 selector from Task 2.
3. Run `git diff --check` and verify the worktree contains no generated bytecode changes.
4. Dispatch an independent whole-plan reviewer over this plan's complete commit range. The reviewer must pressure-test JSON scalar aliases, deterministic bundle identity, archive containment and inventory, source deletion after capture, zero-source reuse, and aggregate pass-two bundle mutation.
5. Only a clean whole-plan verdict permits appending an acceptance entry to the original Task 10 ledger and beginning Task 11. Preserve the original five-round BLOCKED history; append the superseding repair-plan evidence rather than rewriting it.
