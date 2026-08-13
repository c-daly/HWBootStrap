# Tactical-v3 Split Run Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Publish structured policies with separate canonical Arena scenario and authenticated policy identity artifacts.

**Architecture:** Schema-v2 runs keep ML Lab scenario configuration in scenario.json and policy/corpus/checkpoint spaces identity in policy-identity.json. Python validates the policy chain strictly; Unity admits the run by semantic encoding/capacity compatibility while allowing match-specific contract hashes to differ.

**Tech Stack:** Python 3.14, pytest, PyTorch, C#/.NET, Unity 6000.5, Coplay.

## Global Constraints

- Preserve existing v1/v2 acceptance runs unchanged and publish only a fresh sibling.
- Retain pre-tensor-load identity rejection and strict contained regular-file checks.
- Do not weaken deterministic, cross-size, or unsealed-evidence requirements.
- Add no dependencies and no attribution trailers.

---

### Task 1: Python schema-v2 publication and loading

**Files:**
- Modify: python/ml_lab/tactical_v3_checkpoint.py
- Modify: python/ml_lab/tactical_v3_controller.py
- Modify: python/run_tactical_v3_imitation.py
- Modify: python/tests/test_tactical_v3_checkpoint.py
- Modify: python/tests/test_tactical_v3_controller.py
- Modify: python/tests/test_tactical_v3_end_to_end.py

**Interfaces:**
- Produce: publish_structured_run(..., *, training_scenario_path, policy_identity)
- Consume: canonical training scenario plus realized policy spaces identity.

- [ ] Add tests proving schema-v2 inventory, canonical scenario preservation, sidecar authentication/tamper rejection, pre-load controller checks, and exact CLI argument routing.
- [ ] Run the focused tests and confirm failures are caused only by the missing split-artifact behavior.
- [ ] Implement the minimal publisher, validator, controller, and CLI changes.
- [ ] Run checkpoint/controller/E2E tests and the 13-module focused gate.

### Task 2: Unity semantic admission

**Files:**
- Modify: Assets/HexWars/Editor/MlLab/MlLabWindow.cs
- Modify: Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs

**Interfaces:**
- Consume: schema-v2 run.json, canonical scenario.json, and contained policy-identity.json.
- Produce: MlArenaLaunchPlan with unchanged run:PATH controller spec.

- [ ] Add a Unity test where Arena and policy contract hashes differ but encoding/capacity match, plus sidecar/environment/kind/hash rejection cases.
- [ ] Run the focused EditMode test and confirm RED.
- [ ] Implement structural sidecar validation and semantic compatibility rules.
- [ ] Rebuild/sync, run Coplay compile checks and the six affected EditMode classes, and inspect logs.

### Task 3: Fresh acceptance evidence and final verification

**Files:**
- Create: docs/superpowers/reports/2026-08-11-generalizable-structured-imitation-project-b.md

- [ ] Publish fresh python/runs/tactical-v3-policy-project-b-acceptance-v3 without modifying v1/v2; validate it and record metrics/hashes.
- [ ] Re-run live Unity selected-run admission against v3 and confirm capacity identity.
- [ ] Run full engine, focused Python, and feasible full-suite gates; document the independently verified Project A identity-drift limitation precisely.
- [ ] Restore generated files and pycs, run diff/status checks, commit with test: complete tactical-v3 policy gate, and request independent review.

