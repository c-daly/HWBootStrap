# Task 12 Report: ML Lab Tactical-v3 Run Loading

## Outcome

- Added strict tactical-v3 scenario JSON parsing, round-trip serialization, explicit engine conversion, semantic validation, and structured preflight identity with nullable fixed geometry.
- Added fail-closed Arena loading for contained structured-imitation runs, including evidence, environment/version, contract/encoding/capacity hashes, variable geometry, checkpoint, traversal, and reparse checks while preserving exact `run:PATH` specs.
- Kept tactical-v3 out of SB3 Train choices and reject imported tactical-v3 training with offline-imitation/Arena guidance.
- Preserved legacy tactical-v1, tactical-v2, adaptive-v1, and Task 11 policy bridge behavior. No old runs or checkpoints were changed.

## TDD Evidence

- Initial Unity RED failed only on missing `UsesStructuredCandidates` and `ContractIdentity` preflight APIs; production was untouched.
- First scenario GREEN passed 26/26 after implementing strict loading/conversion.
- Expanded containment RED passed 50/51; the sole failure proved lexical `..` was accepted. After raw dot-component rejection, the suite passed 51/51.
- Expanded strict scenario matrix passed 29/29; final focused window matrix passed 53/53.

## Final Verification

- Unity 6000.5.0f1 exact-worktree EditMode filter:
  - `MlTrainingScenarioTests;MlLabConfigTests;MlLabWindowStateTests;TacticalV3PolicyPayloadTests;PolicyBridgeProtocolTests;ModelDuelConfigurationTests`
  - Passed 178/178; failed 0; skipped 0.
- Final log contains no C# compile errors. Its exception lines are expected and asserted Task 11 render-fault test evidence.
- `git diff --check` passed; Unity-generated settings were restored.

## Unity State

- Coplay MCP tools were unavailable in this child session, so Unity 6000.5.0f1 exact-worktree batchmode and direct XML/log inspection were used as the required fallback.
- Every Unity PID was monitored to terminal state; no active batch was restarted.

## Scope and Concerns

- Final scope is the six Task 12 ML Lab implementation/test files plus this report.
- No known remaining concerns.

## Fix Round 1 (2026-08-13)

- Rejected duplicate JSON object properties after escape decoding at every scenario nesting level, preventing last-value overwrite at the root, reward, capacity, and array-row levels.
- Replaced Arena's raw fixed-geometry token scan with structural inspection of decoded direct `contract` members. Escaped observation/action member names now reject, while the same words inside unrelated string metadata remain valid.
- RED: the focused scenario/window gate ran 89 tests; 83 passed and exactly 6 failed: four duplicate-key cases and two escaped fixed-geometry members. The string-metadata control already passed.
- Focused GREEN: the same gate passed 89/89.
- Final Unity 6000.5 exact-worktree six-class Task 12 + Task 11 gate passed 185/185, with 0 failed and 0 skipped. The log contains no C# compile errors; its two exception lines are expected Task 11 render-fault assertions.
- Coplay remained unavailable, so exact-worktree batchmode/XML/log inspection was used. Unity-generated settings were restored; no runs or checkpoints were touched.
