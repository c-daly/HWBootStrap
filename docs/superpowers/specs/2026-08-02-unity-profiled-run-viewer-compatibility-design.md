# Unity Profiled-Run Viewer Compatibility Design

**Date:** 2026-08-02
**Status:** Approved for planning

## Problem

A metadata-backed tactical-v2 run using the annihilation imitation curriculum
cannot be watched from Unity even when the user selects the correct run. The
Editor log records two independent boundary failures:

1. Unity's strict scenario reader rejects the versioned tactical-v2
   `start_profiles` and `start_distribution` fields.
2. `ReplayViewerMenu` looks only for
   `<worktree>/python/winenv/Scripts/python.exe`, while ML Lab training already
   uses `MlLabPaths.ResolvePythonExecutable` to find the shared main-checkout
   environment from a linked worktree.

The selected run and its immutable metadata are correct. The viewer is behind
the training contract.

## Goal

Selecting a metadata-backed tactical-v2 run must reconstruct and launch its
recorded profiled scenario faithfully from a linked Unity worktree, using the
same Windows Python resolution policy as ML Lab training.

The fix must not alter, restart, or pause active trainers. It must not weaken
strict JSON validation or silently translate a profiled run into a standard
scenario.

## Considered approaches

### A. Extend both existing boundaries (selected)

Bring the Unity scenario model, strict wire schema, validation, serialization,
and engine conversion into parity with the existing engine/GymServer
`profiled-seeded-v1` contract. Make the replay viewer resolve Python through
the existing worktree-aware `MlLabPaths` boundary.

This preserves provenance and removes duplicated Python-resolution behavior.

### B. Strip unsupported scenario fields in the viewer

This would make the file parse but would reconstruct a different game from the
one trained. It violates the strict-presentation contract and is rejected.

### C. Copy the virtual environment and rewrite scenarios per worktree

This avoids code changes but duplicates large mutable artifacts and still loses
profiled-start semantics. It is a fragile operational workaround and is
rejected.

## Design

### Scenario parity

Extend `MlTrainingTacticalV2` with explicit lists for:

- start profiles: ID, learner unit count, opponent unit count, and separation;
- start-distribution weights: profile ID and basis points.

Extend `MlTrainingTacticalV2Wire` with matching snake-case wire arrays. Add
those keys to the strict tactical-v2 key set and validate the nested objects
and primitive types without accepting unknown fields.

Map the wire values in both directions so loading and reserializing a run-local
scenario preserves the fields exactly. Extend `MlTrainingScenario.Validate`
with the same semantic distinction as the engine:

- `symmetric-random-v1` requires no profiled catalog or distribution;
- `profiled-seeded-v1` requires the versioned profile catalog and a valid
  distribution;
- any other placement policy remains invalid.

`MlTrainingScenarioPreflight.ToEngine` must copy the profiles and weights into
the engine `TrainingTacticalV2Config`. The engine remains authoritative for
the final detailed profile/distribution validation; Unity's model validation
provides actionable preflight errors and prevents lossy conversion.

### Python resolution

Keep the selected run's Python scripts rooted in the current worktree, but
resolve the executable from the project root through
`MlLabPaths.ResolvePythonExecutable`. Refactor `ReplayViewerMenu` so
`PyReady`, `LaunchDuel`, and Start & Watch use that resolved executable
instead of rebuilding a local-only path.

This deliberately shares only the interpreter environment. It does not redirect
the worktree's `policy_server.py`, GymServer, scenarios, or run metadata to a
different checkout.

### Failure behavior

If neither a local nor shared Windows environment exists, keep a visible,
actionable error containing the attempted resolved path. Invalid or unknown
scenario fields remain hard failures. A selected profiled run must never fall
back to the standard scenario.

## Tests

Use strict TDD with two independent red-green cycles.

1. Add an EditMode scenario test using a literal profiled tactical-v2 JSON
   fixture. It must load, validate, convert to the engine scenario, preserve
   profile/distribution values, and round-trip through serialization. Before
   implementation it fails at the current strict-key rejection.
2. Add an EditMode replay-viewer path test using a temporary linked-worktree
   layout. It must show that the viewer resolves the same shared executable as
   `MlLabPaths`. Before implementation it fails because the viewer constructs
   only the local worktree path.

Retain existing tests that reject truly unknown JSON properties and invalid
placement policies. After each C# edit, verify Unity compilation. Run the
relevant EditMode tests and then the full EditMode suite if the focused tests
pass.

## Operational verification

With the Editor reopened on the linked worktree:

1. select a profiled metadata-backed run;
2. choose its scenario / Start & Watch;
3. confirm no unexpected-field or missing-Python error is logged;
4. confirm the Arena launches from the run's recorded scenario and checkpoint;
5. confirm an unrelated active trainer remains alive throughout.
