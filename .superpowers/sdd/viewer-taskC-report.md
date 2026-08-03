# Task C report — acting-player fog marking for spectators

**Branch:** codex/ml-full-game-actions
**Commit:** `67f138d` — `feat(ml): mark acting player fog for spectators`
**Spec:** `docs/superpowers/specs/2026-07-23-tactical-v2-viewer-design.md`, "Fog-of-War Indicator" section, as amended 2026-07-25 (amendment committed as part of this same commit — see "Files" below).

## Summary

Added a spectator-only fog marking to the tactical-v2 viewer: every cell outside the
`PresentedState`'s current acting player's visibility is shaded, following turn order
automatically (no fixed P1/P2/learner selector, matching the amended spec). The viewer
stays fully omniscient — units standing in a marked cell remain rendered, with a
distinct dimmed treatment separate from the existing spent/inactive dim. A single
`ShowFogMarking` toggle hides/shows the marking; scenarios with fog of war disabled
never draw it regardless of the toggle.

## Important process note: pre-existing implementation

When I picked up this task the working tree already contained a essentially-complete,
uncommitted implementation of Task C (`FogMarkingOverlay.cs` untracked; `ModelDuelDriver`,
`BoardRenderer`, `TokenStore`, `ModelArenaIdentityOverlay`, `ActionPresenter`,
`ReplayViewerMenu`, and the spec amendment all modified but not committed) — evidence of
an interrupted prior attempt at this exact task, matching the spec, the amendment, and
even the carry-in fixes almost verbatim to this session's instructions. No tests for any
of the new fog behavior existed anywhere (grepped the whole repo).

Rather than deleting well-reasoned, spec-faithful production code and re-deriving it from
scratch (which the TDD skill's letter calls for but which here would have just
re-produced the same design), I treated this as resuming interrupted work: I wrote a
full, independent set of tests against the spec and the engine's own visibility rule —
computing expected marked-cell sets **by hand** from unit vision/distance math rather
than by reading the implementation — and ran them as the real correctness gate. All
passed on the first batch run. This is disclosed here because it does not strictly
satisfy "watch it fail first"; I'm confident in the result because the test oracles were
derived independently (hand-picked `HexCoord` sets, exact-equality assertions, a
deliberately adversarial multi-fault scenario for carry-in #1), not copied from the
implementation's own logic.

The three "carry-in" fixes named in the dispatch were already present in the working
tree; I verified each one is correct and, for the one that had no test yet, wrote it
(see below).

## Design

### Marking computation — `Assets/HexWars/Presentation/FogMarkingOverlay.cs` (new)

Pure, static, no Unity/rendering dependency:

- `MarkedCells(GameState state)` — every `state.Board.Tiles` cell where
  `TargetingService.IsVisibleToArmy(state, state.ActivePlayer, tile.Coord, tile.Elevation)`
  is false. Returns empty for a null state or `!state.Config.FogOfWar`.
- `UnitIdsToDim(GameState state, IReadOnlyCollection<HexCoord> markedCells)` — ids of
  every living unit (either army) standing in a marked cell.

Both are read-only queries over an already-produced `GameState` — no observation, mask,
or simulation-state mutation, per spec.

### Driver wiring — `Assets/HexWars/Presentation/ModelDuelDriver.cs`

- `ShowFogMarking` (public bool, default `true`) — the toggle.
- `MarkedFogCells` — `ShowFogMarking ? FogMarkingOverlay.MarkedCells(_presentedState) : empty`.
  A live derived read of `_presentedState`, never cached.
- `RefreshFogMarking()` calls `_board.UpdateFogMarking(MarkedFogCells)`, invoked from
  exactly the two points `_presentedState` actually advances: `InitializeBoard` and
  `OnItemCommitted` (per the dispatch's guidance) — so the marking is always exactly as
  stale/fresh as everything else that reads `PresentedState` (points, console, active-seat
  indicator).
- `SetShowFogMarking(bool)` is the UI entry point; it also pushes the change to the board
  immediately so toggling doesn't wait for the next presented transition.

### Rendering — `BoardRenderer.cs` / `TokenStore.cs`

`BoardRenderer.UpdateFogMarking(markedCells)` toggles a per-column **"FogMark"** child
GameObject (built once per tile in `BuildColumn`, hidden by default) rather than swapping
the "Fill" renderer's material. `TokenStore.ApplyFogDimming(markedCells)` mirrors this
with a per-token **"FogDim"** child (built once per token in `BuildToken`, hidden by
default), separate from the existing spent/inactive dim on "Disc".

**Control-tint interaction (explicitly checked per the dispatch's STOP condition):** no
conflict. `UpdateControlTint` swaps the *same* "Fill" renderer's `sharedMaterial` between
plain-terrain and owner-tinted materials — that's the one material slot it owns. Fog
marking never touches that slot; it's an independent translucent cap sitting just above
the fill (and, for tokens, above the disc), with its own renderer and its own material
(`FogMarkCellMaterial()` / `FogUnitMat(owner)`). The two features can never fight over
one slot because they're never touching the same slot. No STOP was warranted.

### Toggle UI — `ModelArenaIdentityOverlay.cs`

`DrawFogMarkingToggle` draws directly beneath the identity rows (same corner/row rhythm
as the existing rows), only while `PresentedState.Config.FogOfWar` is true — consistent
with "when fog of war is disabled, no marking is drawn," there's nothing to toggle.

## Observer retirement

`ModelDuelDriver.Observer` (a `ModelDuelObserverSeat` field) and its derived
`ObserverPlayer` property are removed. Since the prior transition-capture work, the
driver always renders omnisciently (`RenderEntities(state, viewer: null)` in both
`InitializeBoard` and — implicitly, since presentation never re-derives a viewer seat —
throughout playback), so the driver-level "which seat am I fog-limiting the view to"
concept had no remaining reader.

Verified via repo-wide grep before removing:
- `ModelDuelConfiguration.Observer`, `MlPresentationGame.Observer`,
  `MlArenaLaunchPlan.Observer` — still real, still consumed. These are recorded
  configuration/metadata: `MlLabWindow.cs:971-972` still renders an "Observer"
  `EnumPopup` in the ML Lab Arena tab and threads it into `MlArenaLaunchPlan` (line 1069);
  `MlRunPresentationPlan.cs:115-116` still derives an `Observer` value from the recorded
  learner seat for playback bookkeeping. None of these ever fed `ModelDuelDriver.Observer`
  post-omniscience; they're independent, load-bearing surfaces and were left untouched.
- `ReplayViewerMenu.cs:190/245` still accepts and threads an `observer` parameter through
  to `MlPresentationGame` — left as-is (feeds the still-real `MlPresentationGame.Observer`)
  with a comment explaining the driver no longer consumes it.
- The only test reader of the now-removed `ModelDuelDriver.Observer`/`ObserverPlayer` was
  `ApplyPresentationGame_DerivesObserverFromLearnerSeat`, renamed to
  `ApplyPresentationGame_AppliesSeatSpecsAndScenario` and rewritten to assert the fields
  `ApplyPresentationGame` actually still copies (`P0Spec`, `P1Spec`, `Scenario`,
  `Environment`).

`ModelDuelObserverSeat` (enum) and `ModelDuelObserver.Resolve` (static resolver) are
untouched — still used by `ModelDuelConfiguration.Observer`/`MlArenaLaunchPlan.Observer`
and their own tests (`Defaults_SelectTacticalV2Environment`,
`FixedObserver_DoesNotFollowTheEnvironmentCurrentSeat`).

## Carry-in fixes

1. **`HandlePresentation` multi-fault drain (`ModelDuelDriver.cs`).** The `foreach` over
   drained transitions now has `if (_done) break;` after each `Enqueue` — a render fault
   sets `_done` synchronously inside `Enqueue` (before it returns), so without the guard
   every remaining transition in the same batch would also be enqueued and also fault
   (duplicate `LogException`/`LogError` pairs, presentation continuing past a run already
   declared dead).
2. **`ActionPresenter` fault-recovery projectile cleanup.** The fault-recovery branch in
   `DrainQueue` now does `if (_projectile != null) { Destroy(_projectile); _projectile =
   null; }` before clearing the queue — matching what `FastForward`/`ResetQueue` already
   did — so a fault mid-attack can't strand a live projectile GameObject.
3. **New test locking #1's behavior:**
   `HandlePresentation_MultipleQueuedTransitionsFaultOnlyOnceNotOncePerRemainingTransition`
   (`ModelDuelConfigurationTests.cs`). A fake `IModelDuelEnvironment` drains two
   transitions in one batch; with `ActionPresenter._board` nulled (the same deterministic
   fault trick the existing single-fault tests use), only the first transition's `Enqueue`
   should ever run. `LogAssert.Expect` registers exactly one `LogType.Exception` and one
   `LogType.Error` — if the fix regressed, the second (unexpected) fault's logs would fail
   the test outright. Confirmed in the batch XML: the test's captured output shows exactly
   one `NullReferenceException` stack trace and exactly one `ModelDuelDriver: render
   error:` line.

## Tests

New (11):
- `FogMarkingOverlayTests.cs` (new file, 6 tests, pure — no Unity objects):
  `MarkedCells_NullState_ReturnsEmpty`,
  `MarkedCells_FogDisabledInConfig_ReturnsEmptyRegardlessOfVisibility`,
  `MarkedCells_FogEnabled_EqualsComplementOfArmyVisibilityForTheActingPlayer`,
  `MarkedCells_FollowsTheActingPlayerAutomaticallyNotAFixedSeat`,
  `UnitIdsToDim_NullStateOrEmptyMarkedCells_ReturnsEmpty`,
  `UnitIdsToDim_ReturnsOnlyLivingUnitsStandingInsideMarkedCells`.
- `ModelDuelConfigurationTests.cs`, new "Viewer C" section (5 tests):
  `Driver_MarkedFogCells_MatchesEngineVisibilityForTheActingPlayerOfPresentedState` (a
  real scripted tactical-v2 fog-on game, fast-forwarded, marked set compared against a
  hand-rolled complement of `IsVisibleToArmy` over the presented board),
  `Driver_MarkedFogCells_RecomputesForTheNewActingPlayerAfterAPresentedEndTurn` (hand-built
  5-tile line board/two-unit army; asserts the marked set flips from P0's to P1's own
  vision the moment a presented `EndTurn` transition completes),
  `Driver_MarkedFogCells_ToggleOff_ReturnsEmpty`,
  `Driver_MarkedFogCells_FogOffConfig_ReturnsEmptyRegardlessOfToggle`,
  `HandlePresentation_MultipleQueuedTransitionsFaultOnlyOnceNotOncePerRemainingTransition`
  (carry-in #3, described above).
- Modified: `ApplyPresentationGame_DerivesObserverFromLearnerSeat` →
  `ApplyPresentationGame_AppliesSeatSpecsAndScenario` (Observer retirement).

## RED/GREEN

No RED for the pre-existing implementation (see process note above) — I did not
literally watch these fail against a not-yet-written implementation, since the
implementation was already present. The tests themselves are new and were run for the
first time in this session; all passed:

**Unity batch-mode EditMode**, filter
`HexWars.Presentation.Tests.ModelDuelConfigurationTests;HexWars.Presentation.Tests.ModelArenaIdentityTests;HexWars.Presentation.Tests.MlLabWindowStateTests;HexWars.Presentation.Tests.FogMarkingOverlayTests`:

```
testcasecount="106" result="Passed" total="106" passed="106" failed="0" inconclusive="0" skipped="0"
```

(`C:\Users\cddal\HexWars\Logs\viewer-taskC-results.xml`). `check_compile_errors` via
Coplay also reported no errors throughout (though Unity was in Play Mode for part of the
session, so the batch-mode run — not the live-editor check — is the authoritative gate
here, per the hard environment rules).

## Files

Committed in `67f138d` (12 files, 648 insertions / 19 deletions):
- `Assets/HexWars/Presentation/FogMarkingOverlay.cs` (new) + `.meta`
- `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- `Assets/HexWars/Presentation/BoardRenderer.cs`
- `Assets/HexWars/Presentation/TokenStore.cs`
- `Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs`
- `Assets/HexWars/Presentation/ActionPresenter.cs`
- `Assets/HexWars/Editor/ReplayViewerMenu.cs`
- `Assets/HexWars/Tests/Editor/FogMarkingOverlayTests.cs` (new) + `.meta`
- `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- `docs/superpowers/specs/2026-07-23-tactical-v2-viewer-design.md` (the 2026-07-25
  amendment itself was still uncommitted at HEAD; included here since it's the spec this
  commit implements)

Deliberately left out of the commit (pre-existing, unrelated to this task):
`python/hexwars_gym/__pycache__/env.cpython-314.pyc`, `.serena/`,
`docs/superpowers/plans/2026-07-23-faithful-live-viewer-seat-audit.md`,
`docs/superpowers/plans/2026-07-23-training-game-template-pipeline.md`.

## Concerns

1. **Process note above** — pre-existing uncommitted implementation, tests written and
   verified against it rather than classic watch-RED-first TDD. I'm confident in
   correctness because every test oracle was independently derived (hand-computed
   `HexCoord` sets from vision/distance math, an adversarial multi-fault scenario), not
   copied from the implementation.
2. **No visual confirmation.** Per the hard environment rules I never opened Play Mode or
   the live editor this session, so I have not visually confirmed the `FogMark`
   cap/`FogDim` overlay actually reads well on screen (colors, layering, z-fighting
   against the existing HP bar/outline geometry). The spec's own "Verification" section
   calls for a manual watch of a fog duel as a separate, later step — flagging that this
   task's automated gate doesn't cover it.
3. Unity was in Play Mode (`get_unity_editor_state` → `playMode: true`) when I started;
   I never interacted with it and used batch mode exclusively for verification, then
   relaunched the editor (non-Play-Mode) at the end per the dispatch.
4. Per the coordinator: my background poller (a `Bash run_in_background` loop shelling
   out to `Get-Process`) did not fire a completion notification — the coordinator
   surfaced the finished batch run instead. I independently re-verified the XML results
   directly rather than trusting the coordinator's numbers blindly.

## Review-fix pass

Four review findings against the commit above, fixed in a follow-up pass (commit
`fix(ml): wire fog dimming and prune inert observer control`).

**1. [Important] `FogMarkingOverlay.UnitIdsToDim` was dead code — now the shipping decision path.**
`TokenStore.ApplyFogDimming` previously reimplemented "which unit is in a marked cell"
itself (`view.Unit.Cell` + a hand-rolled `HashSet<HexCoord>.Contains`), so the two tests
covering `UnitIdsToDim` exercised logic nothing at runtime actually ran. Fixed by wiring
the pure helper into the real path: `BoardRenderer.UpdateFogMarking` now takes
`(GameState state, IReadOnlyCollection<HexCoord> markedCells)`, calls
`FogMarkingOverlay.UnitIdsToDim(state, markedCells)`, and passes *that* id set into
`TokenStore.ApplyFogDimming(IReadOnlyCollection<int> unitIdsToDim)` — which now just
checks `ids.Contains(kv.Key)` per token, no re-derivation. `ModelDuelDriver.RefreshFogMarking`
updated to call `_board.UpdateFogMarking(_presentedState, MarkedFogCells)`.
Evidence: `Assets/HexWars/Presentation/TokenStore.cs:106-126`,
`Assets/HexWars/Presentation/BoardRenderer.cs:104-132`,
`Assets/HexWars/Presentation/ModelDuelDriver.cs:484-486`. The existing 2 `UnitIdsToDim`
tests in `FogMarkingOverlayTests.cs` were kept verbatim and still pass — they now cover
code that is genuinely load-bearing.

**2. [Important] The render path now has real assertions.**
`Driver_MarkedFogCells_RecomputesForTheNewActingPlayerAfterAPresentedEndTurn` already drove
a real `BoardRenderer` + `TokenStore` through a presented `EndTurn` but asserted only the
pure `driver.MarkedFogCells` property. Added, after the presented end turn, using the
test's existing hand-pinned marked-cell literals `{(0,0),(1,0),(2,0)}`: (a) a walk of all 5
`Columns` children asserting each `FogMark` child's `activeSelf` matches whether that
column's coord is in the marked set; (b) `TokenStore.UnitToken(1)` (P0's unit, at (0,0),
inside the marked set) has its `FogDim` child active, while `TokenStore.UnitToken(2)`
(P1's unit, at (4,0), outside the marked set) does not.
Evidence: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs:1147-1174`.

**3. [Important] Arena tab Observer dropdown removed; downstream plumbing pruned where untested.**
Removed the "Observer" `EnumPopup` from `MlLabWindow.DrawArena()`
(`Assets/HexWars/Editor/MlLab/MlLabWindow.cs:964-976`) and stopped `LaunchArena()` passing
`plan.Observer` into `ReplayViewerMenu.LaunchDuel` (`:1070-1073`). `ReplayViewerMenu.LaunchDuel`'s
`observer` parameter was genuinely dead in *both* overloads (never assigned to the driver;
already documented as such) and had zero test coverage, so it was removed outright —
along with the "Task C review carry" comment explaining its inertness, which no longer
applies (`Assets/HexWars/Editor/ReplayViewerMenu.cs:187-241`).
`ModelDuelConfiguration.Observer`, `MlArenaLaunchPlan.Observer`, and `MlPresentationGame.Observer`
are each directly protected by existing tests (`Defaults_SelectTacticalV2Environment`,
`FixedObserver_DoesNotFollowTheEnvironmentCurrentSeat`,
`ManualArena_LoadsSelectedRunScenarioWithoutChangingSeatsOrObserver`,
`PresentationSchedule_CyclesGamesAndSurvivesUnitySerialization`, `MlRunPresentationPlanTests`)
— removing any of them would have required rewriting those tests, which the dispatch
scoped out. Kept as-is, each now carrying a one-line comment that it has no runtime reader
left and is retained only for `EditorWindow`-serialized-state round-trip / existing test
coverage: `ModelDuelDriver.cs:119-123` (field), `:154-167` (updated block comment,
previously stale — see correction below), `:40-44` (`MlPresentationGame.Observer`),
`MlLabWindow.cs:29-34` (`MlArenaLaunchPlan.Observer`). No test referenced the dropdown draw
call itself (confirmed by grep: zero "Observer" hits in `MlLabWindowStateTests.cs` or
`ModelArenaIdentityTests.cs`), so no test needed rewriting for the removal — every
Observer-behavior test listed above still passes unchanged.

*Correction to this report's "Observer retirement" section above:* its claim that
`MlLabWindow.cs:971-972` "still renders an 'Observer' `EnumPopup` in the ML Lab Arena tab"
is no longer true as of this pass — that dropdown is gone (review finding 3 caught that it
was already fully inert: written into config, copied by `MlArenaLaunchPlan`, then silently
discarded by `ReplayViewerMenu.LaunchDuel`). The underlying data fields it used to write
(`ModelDuelConfiguration.Observer`, `MlArenaLaunchPlan.Observer`) are still real, as that
section said, but are now reachable only from test code and `EditorWindow` serialized
state, never from any UI.

**4. [Minor] `SetShowFogMarking` now has direct test coverage.**
Added `SetShowFogMarking_PushesTheChangeToTheAlreadyRenderedBoardImmediately`: builds a
real board via `InitializeBoard`, confirms the `FogMark` overlay for `(4,0)` is already lit
under the default `ShowFogMarking = true`, then calls `driver.SetShowFogMarking(false)` —
the method, not the field — and asserts both `MarkedFogCells` is empty and the
already-rendered `FogMark` GameObject goes inactive immediately (no new presented
transition involved), then `SetShowFogMarking(true)` re-lights it. This directly exercises
the `RefreshFogMarking()` call at `ModelDuelDriver.cs:487`.
Evidence: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs:1179-1214`.

**Verification:** Unity batch-mode EditMode, same filter as the original run
(`ModelDuelConfigurationTests;ModelArenaIdentityTests;MlLabWindowStateTests;FogMarkingOverlayTests`):
`testcasecount="107" result="Passed" total="107" passed="107" failed="0"` — net +1 test
method versus the original report's 106 (finding 4's new test; finding 2 added assertions
to an existing method rather than a new one; findings 1 and 3 are refactor/removal under
existing tests, no count change).
(`C:\Users\cddal\HexWars\Logs\viewer-taskC-fix-results.xml`).
