# Task 11 Report: Engine Transition Drain and Unity Tactical-v3 Arena Bridge

## Outcome

- Added a non-destructive tactical-v3 transition drain cursor. Draining advances a detached presentation cursor without changing replay history, state, decision ids, or truncation accounting.
- Added the Unity tactical-v3 environment enum, semantic identity with capacity hash, structured adapter/view, explicit snake_case DTO projection, semantic/reference validation, strict decision/candidate response parsing, and capacity-only policy-server startup evidence.
- Routed the arena driver through exact tactical-v3 decision/candidate identities, proved candidate membership before stepping, preserved legacy fixed-geometry routes, and labeled structured imitation as a structured PyTorch policy.
- Preserved default-false transition capture semantics at the adapter boundary without removing the engine's complete replay history.

## TDD Evidence

- Engine RED: the focused suite failed with seven CS1061 references because `TacticalV3DuelEnv.DrainTransitions` was absent.
- Engine first GREEN: 25/25 `TacticalV3DuelEnvTests` passed.
- Unity RED: after syncing the ignored engine DLL required by the fresh worktree, compilation failed on the absent tactical-v3 DTO, enum, structured adapter, capacity argument, and structured result APIs.
- Unity first GREEN: 70/70 focused EditMode tests passed.
- Capture semantics RED: 1/1 focused test failed because the default-false adapter drain exposed one transition.
- Capture semantics GREEN: 1/1 focused test passed after advancing the presentation cursor while capture is disabled.

## Final Verification

- `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV3DuelEnvTests|FullyQualifiedName~TacticalV2DuelEnvTests|FullyQualifiedName~ReplayTests|FullyQualifiedName~ReplayFileTests" --no-restore`
  - Passed 65/65; failed 0; skipped 0.
- `powershell -ExecutionPolicy Bypass -File engine/build-to-unity.ps1`
  - Release build succeeded with 0 warnings and 0 errors; DLL copied to the Unity Plugins directory.
- Unity 6000.5.0f1 exact-worktree batchmode EditMode filter:
  - `TacticalV3PolicyPayloadTests;PolicyBridgeProtocolTests;ModelDuelConfigurationTests`
  - Passed 71/71; failed 0; skipped 0.
  - Final log contains no C# compile errors, tactical-v3 exceptions/errors, or unhandled exceptions.

## Unity State

- Coplay MCP tools were not exposed in this session, so live Editor state, `check_compile_errors`, and `get_unity_logs` were unavailable.
- Used the required fallback: Unity 6000.5.0f1 batchmode against the exact task worktree, then inspected test XML and log files directly.
- One compiler-error batch reached a terminal log but left PID 40208 holding the project lock. With explicit authorization, only that PID was terminated before continuing.

## Scope and Concerns

- Restored Unity/import-generated settings and unrelated Python bytecode files before staging.
- No old runs or checkpoints were modified.
- No known remaining concerns.

## Fix Round 1 (2026-08-12)

- Tightened structured policy action parsing to accept only JSON whitespace (space, tab, CR, and LF) and JSON integer syntax (-?(?:0|[1-9][0-9]*)) while retaining invariant Int64/Int32 bounds and either exact key order.
- RED: focused PolicyBridgeProtocolTests ran 13 tests; 10 passed and exactly 3 failed for decision 08, candidate 01, and an NBSP separator.
- GREEN: the same focused gate passed 13/13.
- Final exact-worktree Unity 6000.5 EditMode gate passed 75/75 with exit code 0 and no C# compile errors or unhandled exceptions in the log.
- Engine production and tests were untouched, so the engine gate was omitted as authorized for this fix round.
- Coplay accepted the exact project root, but no Editor was running there; check_compile_errors was attempted after the C# edit and the exact-worktree Unity batch compile/log gate provided the fallback verification.
