# Task 10 Breaker Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the two load-bearing Task 10 defects that remained after the original selective-DAgger plan's five-round breaker, without changing the accepted experiment design or beginning Task 11.

**Architecture:** Preserve the existing physical-evidence validators and add two narrow last-mile invariants. Audited contracts compare the duplicated board document with JSON-type-exact semantics, and supervised evaluation ends with a group-level stability barrier over every held-out overlay so no earlier overlay can change while a later overlay is being reopened.

**Tech Stack:** Python 3.11, pytest, immutable JSON publications, `pathlib`, HexWars DAgger physical evidence validators.

## Governing design

This repair implements the already-approved design in `docs/superpowers/specs/2026-08-03-selective-dagger-search-distillation-design.md`. The original SDD ledger remains historical evidence that Task 10 exhausted its five repair rounds. The user explicitly authorized this new two-defect plan on 2026-08-09; this plan has its own ledger and review breaker.

## Global Constraints

- Do not modify the Task 8/Task 11 trust boundary: production Task 8 evidence remains fail-closed until Task 11 supplies the sealed-engine adapter.
- Do not begin Task 11 until both tasks below pass task review, combined verification, and an independent whole-plan review.
- Preserve arbitrary valid board dimensions, unit templates, rosters, and scenario configuration; never pin the retained baseline's concrete values as general rules.
- JSON equality at the audited boundary is type-sensitive: booleans are not integers, and integer/float aliases are distinct unless the schema explicitly normalizes them.
- The supervised-evaluation physical overlay barrier must be the absolute-final filesystem operation before return. After it begins, no callback, provider, local-publication read, checkpoint read, or repository probe may run.
- The final barrier covers the full overlay set as one group: snapshot all roots, reopen/validate all roots, snapshot all roots again, compare every result, then return immediately.
- Keep pre-resolution reparse/junction rejection and exact physical overlay inventory checks. Windows symbolic-link tests may skip only for WinError 1314.
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

### Task 2: Make all held-out overlays stable as one final group

**Files:**
- Modify: `python/run_annihilation_selective_dagger.py:2017-2080`
- Test: `python/tests/test_annihilation_selective_dagger.py:6911-7047`

**Interfaces:**
- Consumes: immutable `DevelopmentHeldoutOverlayEvidence` objects, `_supervised_overlay_tree_snapshot`, `dagger_domain.open_dagger_overlay`, and `dagger_domain.dagger_overlay_supervised_examples`.
- Produces: the same `DevelopmentSupervisedEvidence`; no public signature or publication schema changes.

- [ ] **Step 1: Write the failing two-overlay race regression**

  Add `test_task10_supervised_rejects_first_overlay_mutated_while_second_overlay_is_finalized`. Build a physical Task 10 definition at iteration 2, publish supervised evaluation over both cumulative validation overlays, and identify a shard in overlay 1. Instrument a real finalization boundary for overlay 2 (for example, wrap the physical overlay opener and mutate overlay 1 only when overlay 2 is reopened during the final stability phase). Reopen the published supervised evaluation and require `ValueError` matching `overlay|shard|physical|changed`.

  The mutation trigger must be deterministic, must occur after overlay 1's old per-item final snapshot, and must exercise real physical overlay parsing rather than asserting mock calls.

- [ ] **Step 2: Run the regression and verify RED**

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py::test_task10_supervised_rejects_first_overlay_mutated_while_second_overlay_is_finalized -q
  ```

  Expected: FAIL with `DID NOT RAISE ValueError`. A malformed overlay or incomplete fixture is not an acceptable RED.

- [ ] **Step 3: Implement the group-level final barrier**

  Refactor `_require_supervised_overlay_stability` so its ordering is globally phased rather than per-item:

  ```text
  initial_snapshots = snapshot every overlay root
  compare every initial snapshot to its frozen evidence
  reopened = physically open and project every overlay
  compare every reopened identity/partition/examples to frozen evidence
  final_snapshots = snapshot every overlay root
  compare every final snapshot to both its initial snapshot and frozen evidence
  return immediately
  ```

  Do not interleave `snapshot -> reopen -> snapshot` per overlay. Preserve canonical unresolved-root and reparse checks. Ensure `_open_development_supervised_evaluation` constructs its result and completes all local publication, checkpoint, repository, and expected-content checks before invoking this barrier, then returns immediately afterward.

- [ ] **Step 4: Verify GREEN and prior race coverage**

  Run the new two-overlay regression together with the mid-metrics mutation, fourth-evidence-read mutation, reparse-root test, and supervised transactional/reuse tests:

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py -k "first_overlay_mutated_while_second_overlay_is_finalized or overlay_shard_mutated_during_metrics or overlay_shard_mutated_on_fourth_evidence_read or reparse_overlay_root or task10_supervised_evaluation_is_transactional" -q
  ```

  Expected: PASS; only the existing WinError-1314 symbolic-link case may skip.

- [ ] **Step 5: Run bounded Task 10 verification**

  ```powershell
  uv run --active --no-project python -m pytest python/tests/test_annihilation_selective_dagger.py -k "task10 or development_candidate or development_supervised or development_aggregate" -q
  ```

  Expected: PASS within the existing bounded test budget; only explicitly privilege-gated Windows reparse tests may skip.

- [ ] **Step 6: Self-review and commit**

  Audit the final return path to prove no read/provider/callback occurs after the group barrier. Confirm the regression fails if the loop is reverted to per-overlay snapshot/reopen/snapshot ordering. Run `git diff --check`, then commit only the Task 2 production/test files:

  ```powershell
  git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py
  git commit -m "fix: seal supervised overlays as a final group"
  ```

---

## Completion gate

After both task-scoped reviews pass:

1. Run the combined focused A/B regressions and the full `test_checkpoint_audit.py` suite.
2. Run the bounded Task 10 selector from Task 2.
3. Run `git diff --check` and verify the worktree contains no generated bytecode changes.
4. Dispatch an independent whole-plan reviewer over this plan's complete commit range. The reviewer must pressure-test JSON scalar aliases and multi-overlay final-read ordering, not merely inspect the happy paths.
5. Only a clean whole-plan verdict permits appending an acceptance entry to the original Task 10 ledger and beginning Task 11. Preserve the original five-round BLOCKED history; append the superseding repair-plan evidence rather than rewriting it.
