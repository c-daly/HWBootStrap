# Tactical-v3 final-review fix report

Date: 2026-08-11
Worktree: `C:\Users\cddal\HexWars\.worktrees\physical-checkpoint-audit`
Branch: `codex/selective-dagger-design`
Starting HEAD: `2a36dcce6387af79dfa61a1b4370ee6a159029c0`
Status: complete; the final-fix commit is the commit containing this report.

## Scope and root causes

1. Unsupported capture mechanics crossed the direct-config boundary because
   `TacticalV3Config.Validate` checked fog, annihilation, and generators but did not require
   `CaptureCost == int.MaxValue`, `TerritoryMode == false`, or zero territory income.
   `LegalMoves.For` can enumerate `CaptureHex` independently of territory mode when a unit is on
   an uncontrolled tile and can afford the configured capture cost, but tactical-v3 supports only
   attack, move, deploy, and end-turn candidates.
2. Observation cell rows came from the canonical `TacticalV2Layout.Cells`, while candidate
   references were assigned by `state.Board.Tiles` insertion order. The two orders are not an
   interchangeable coordinate authority.
3. Tactical-v3 JSON and runtime configuration allowed template health 0. Reward material value
   divided current HP by template health, producing NaN.
4. `RefreshFrame` cleared candidates only for engine-terminal state, not for step truncation, even
   though `Step` rejected a finished episode.
5. `TacticalV3Wire.View` validated token references but did not re-check actual table counts
   against the capacity attached to the scenario contract.
6. The rectangular topology comment said even-q although the authoritative encoding is odd-q.
7. The tracked Project-A report omitted ancestor commit `1895505` and described the topology
   wording correction as deferred.

## Fixes

- The public runtime config boundary now rejects all unsupported stage-one capture/income paths,
  non-positive max steps, negative starting points, non-finite or negative bounty/deploy rates,
  template health below 1, negative template stats, Int32 point/deploy/bounty overflow, and any
  reachable point total that could reach the disabled capture sentinel.
- Direct-config construction still fails before observation/candidate enumeration. JSON applies
  health >= 1 only to tactical-v3; the shared tactical-v2 validation keeps its prior health >= 0
  rule.
- `TacticalV3Observation` privately snapshots the actual-coordinate-to-canonical-row mapping.
  Candidates and projections resolve every cell through that mapping. Public DTO fields and wire
  token semantics are unchanged.
- Reward evaluation uses a defensive denominator of at least 1 and converts NaN shaping inputs to
  zero before bounding; terminal totals remain within the published range.
- Every finished frame uses an empty candidate/command set, whether terminated or truncated.
  Further steps fail before candidate resolution and do not append replay commands.
- `TacticalV3Wire.View(view, capacity)` rejects overflow in cells, units, templates,
  capability_definitions, capability_allocations, rules, memory, relations, and candidates before
  serialization. All live GymServer tactical-v3 callers pass `tacticalV3Config.Capacity`.
- The topology wording is corrected to odd-q. The tracked completion report now includes
  `1895505` and current final-review evidence.

## TDD evidence

The baseline tactical-v3 selector command was attempted before edits but the command wrapper timed
out after 124 seconds without assertion output. Coplay was already bound to the exact worktree and
reported a healthy Editor.

Independent RED results before production edits:

- Direct config: 4 failed, 0 passed. Both capture variants returned no capture error; the
  reachable-points case returned no points/capture error; zero health plus point-cost overflow
  returned neither required error.
- Canonical rows: 2 failed, 0 passed. With reversed board tile insertion, Player 0 mapped its first
  cell evidence to `101@10,2` instead of `1@1,0`; Player 1 failed analogously.
- Reward: 1 failed, 0 passed. The observed breakdown fields were
  `[-1, NaN, 0, -0, NaN]`.
- JSON: 1 failed, 0 passed. A tactical-v3 zero-health template did not throw
  `InvalidDataException`.
- Truncation: 2 failed, 0 passed. Both exact-step finished frames retained candidates.
- Wire: 1 failed, 0 passed. Reflection could not find the requested `View(view, capacity)`
  boundary (`Sequence contains no matching element`).
- No RED unexpectedly passed and no RED had a product compile/test error.

The first GREEN build exposed two mechanical patch-placement errors:
`TacticalV3Config.IsFinite` was outside its type and the observation helper interrupted the
`IObservationMemory` declaration. After moving those members into the intended type bodies,
Coplay reported no compile errors and the six focused groups passed 11, failed 0.

The first complete selector then passed 277 and failed one existing ordering test:
`CreateFrame_OrdersWithinKindsByDecisionLocalRows` still built its oracle from
`Board.Tiles` insertion order and hard-coded the old move rows. Its title and production
comparator both require decision-local observation rows. The test oracle was corrected to derive
coordinate rows from `frame.Observation`; the focused canonical order group then passed 4/4.
No production behavior was reverted.

## Final verification

- Exact selector:
  `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV3" --no-restore`
  - passed 278, failed 0, skipped 0; duration 2m07s.
- Full engine/GymServer suite:
  `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --no-restore`
  - passed 991, failed 0, skipped 0; duration 3m33s.
- GymServer build:
  `dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore`
  - succeeded with 0 warnings and 0 errors.
- Structured JSONL smoke:
  - `spaces` returned tactical-v3 structured contract evidence;
  - seed-41 `reset` returned decision 0 with 10 candidates;
  - candidate 0 (`move`) returned successor decision 1 with 9 candidates;
  - structured observation rows were present and flat `obs`/`mask` were absent;
  - `close` exited 0 with empty stderr.
- The legacy Windows PowerShell process API could not provide `ArgumentList` or
  `StandardInputEncoding`; two PowerShell smoke-driver attempts therefore failed before a valid
  JSON request. The repository Python runtime was used only as a BOM-less UTF-8 process driver for
  the successful identical JSONL sequence. No repository state changed during those harness
  failures.
- Final Coplay sequence, in required order:
  - Editor state: `playMode=false`, `hasCompilationErrors=false`;
  - compile: `No compile errors`;
  - 200-entry Editor log query: empty.
- `git diff --check`: clean apart from Git's informational LF-to-CRLF working-copy warnings.

## Residual concerns

- The existing GymServer process-test harness remains slow enough that the tactical-v3 selector
  takes about two minutes and the full suite takes about three and a half minutes. Both final exact
  runs completed normally.
- Project A remains an unsealed experimental contract as stated in the tracked completion report.
