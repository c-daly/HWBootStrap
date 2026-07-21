# Intelligent Movement Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let units spend movement over multiple commands, end movement after attacking, and preview/animate the exact deterministic engine-owned route.

**Architecture:** `MovementService` exposes immutable `MovementRoute` values while keeping current reachability APIs as projections. Presentation owns only preview interaction and pooled rings; it never recalculates legality. Accepted moves recompute the same route from the queued pre-command state for animation, so `MoveUnit` wire and replay formats stay unchanged.

**Tech Stack:** C# 9, .NET Standard 2.1, NUnit, Unity URP, Unity Input System, Unity EditMode/PlayMode tests.

## Global Constraints

- Work in an isolated git worktree: the main checkout has extensive unrelated user changes, including line-ending edits and a changed binary.
- Never discard, normalize, stage, or commit those existing changes.
- For each behavior, add one failing test, run it, implement the minimum change, then rerun it.
- `MovementService.Routes` is authoritative. Presentation must not invent a legal route or use a straight fallback.
- Preserve `MoveUnit` serialization and all network/replay formats.
- Keep `MovementSpent` equal to movement actually paid. `AttackedUnitIds` closes movement.
- After every C# edit run Unity `check_compile_errors`; inspect Unity logs before runtime claims.
- If Coplay remains unavailable, do not claim completion without compile, EditMode, PlayMode, and log verification.
- Commit each task without attribution trailers.

---

## Task 1: Expose immutable deterministic movement routes

**Files:**

- Create: `engine/HexWars.Engine/MovementRoute.cs`
- Modify: `engine/HexWars.Engine/MovementService.cs`
- Test: `engine/HexWars.Engine.Tests/MovementServiceTests.cs`

- [ ] Add a failing test named `Routes_ReturnsOrderedLegalDetour_WithCostsAndRemainingBudgets`. Build four cells with an occupied direct cell and one climbing two-hop detour. Assert `Cells` contains origin, detour, destination in order; H/V costs are `2/1`; remaining budgets reflect the unit's original `3/2` budgets.

- [ ] Run:

```bash
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter FullyQualifiedName~MovementServiceTests.Routes_ReturnsOrderedLegalDetour
```

Expected: compile failure because `MovementService.Routes` and `MovementRoute` do not exist.

- [ ] Create this immutable value object:

```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    public sealed class MovementRoute
    {
        readonly HexCoord[] _cells;
        public IReadOnlyList<HexCoord> Cells => _cells;
        public int HorizontalCost { get; }
        public int VerticalCost { get; }
        public int HorizontalRemaining { get; }
        public int VerticalRemaining { get; }

        public MovementRoute(IEnumerable<HexCoord> cells, int horizontalCost, int verticalCost,
                             int horizontalRemaining, int verticalRemaining)
        {
            _cells = cells == null ? throw new ArgumentNullException(nameof(cells))
                                   : new List<HexCoord>(cells).ToArray();
            if (_cells.Length < 2)
                throw new ArgumentException("A route requires origin and destination.", nameof(cells));
            HorizontalCost = horizontalCost;
            VerticalCost = verticalCost;
            HorizontalRemaining = horizontalRemaining;
            VerticalRemaining = verticalRemaining;
        }
    }
}
```

- [ ] In `MovementService`, replace cost-only queue labels with private immutable labels containing `Coord`, `H`, `V`, and a copied `HexCoord[] Cells`. Add:

```csharp
public static Dictionary<HexCoord, MovementRoute> Routes(GameState state, Unit unit)
```

Retain occupied/passable/terrain/elevation rules and a Pareto H/V frontier. Return empty if the unit attacked or has no horizontal budget. Each entered cell adds terrain move cost and positive elevation gain.

- [ ] Choose the destination label deterministically by H, then V, then path length, then every path coordinate by Q and R. Sort neighbor expansion by Q/R too. For equal H/V labels at the same coordinate retain only the lexicographically preferred path; never drop distinct non-dominated H/V labels.

- [ ] Project the old APIs from the route map without a second search:

```csharp
public static IReadOnlyCollection<HexCoord> ReachableTiles(GameState state, Unit unit)
    => Routes(state, unit).Keys;

public static Dictionary<HexCoord, (int H, int V)> ReachableCosts(GameState state, Unit unit)
{
    var result = new Dictionary<HexCoord, (int H, int V)>();
    foreach (var pair in Routes(state, unit))
        result[pair.Key] = (pair.Value.HorizontalCost, pair.Value.VerticalCost);
    return result;
}
```

- [ ] Add a deterministic equal-cost-route test and a Pareto tradeoff test where lower-H/higher-V and higher-H/lower-V labels must both survive to reach later destinations.

- [ ] Run the full fixture:

```bash
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter FullyQualifiedName~MovementServiceTests
```

Expected: all tests pass.

- [ ] Commit:

```bash
git add engine/HexWars.Engine/MovementRoute.cs engine/HexWars.Engine/MovementService.cs engine/HexWars.Engine.Tests/MovementServiceTests.cs
git commit -m "feat: expose deterministic movement routes"
```

---

## Task 2: Make attack close movement in all engine entry points

**Files:**

- Modify: `engine/HexWars.Engine/RejectionReason.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/LegalMoves.cs`
- Test: `engine/HexWars.Engine.Tests/MultiHopMovementTests.cs`
- Test: `engine/HexWars.Engine.Tests/GameEngineApplyAttackTests.cs`
- Test: `engine/HexWars.Engine.Tests/LegalMovesTests.cs`

- [ ] Add failing tests proving move-then-attack succeeds, attack-then-move returns `MovementEndedByAttack`, routes are empty after attack, actual `MovementSpent` is unchanged by attack, `LegalMoves` omits post-attack moves, and a full turn cycle restores routes.

- [ ] Run:

```bash
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~MultiHopMovementTests|FullyQualifiedName~GameEngineApplyAttackTests|FullyQualifiedName~LegalMovesTests"
```

Expected: attack-then-move and post-attack legal-move assertions fail.

- [ ] Add `MovementEndedByAttack` to `RejectionReason`. In `GameEngine.ApplyMoveUnit`, after finding the living unit and before budget/route lookup, add:

```csharp
if (state.AttackedUnitIds.Contains(c.UnitId))
    return Result.Reject(state, RejectionReason.MovementEndedByAttack);
```

- [ ] Resolve the move through `MovementService.Routes` and charge `route.HorizontalCost`/`route.VerticalCost`. Leave the route service's attacked-unit empty result as defense in depth. Let `LegalMoves` inherit the rule via `ReachableTiles`; do not duplicate it there.

- [ ] Rerun the focused fixtures and expect all pass.

- [ ] Commit:

```bash
git add engine/HexWars.Engine/RejectionReason.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine.Tests/MultiHopMovementTests.cs engine/HexWars.Engine.Tests/GameEngineApplyAttackTests.cs engine/HexWars.Engine.Tests/LegalMovesTests.cs
git commit -m "feat: end unit movement after attack"
```

---

## Task 3: Extract testable desktop/touch preview state

**Files:**

- Create: `Assets/HexWars/Presentation/MovementPreviewState.cs`
- Create: `Assets/HexWars/Tests/Editor/HexWars.Presentation.Tests.asmdef`
- Create: `Assets/HexWars/Tests/Editor/MovementPreviewStateTests.cs`
- Create associated Unity `.meta` files.

- [ ] Create an Editor test asmdef referencing `HexWars.Presentation` and `Unity.InputSystem`, with `includePlatforms: ["Editor"]` and `optionalUnityReferences: ["TestAssemblies"]`.

- [ ] Add failing pure tests: desktop reachable hover previews; desktop leave clears; first reachable touch tap previews; second identical tap confirms; another reachable tap switches; unreachable/unrelated tap clears; `Clear` resets all state.

- [ ] Run the presentation EditMode assembly. Expected: compile failure because preview types do not exist.

- [ ] Implement:

```csharp
using HexWars.Engine;

namespace HexWars.Presentation
{
    public enum MovementPreviewDecision { None, Preview, Confirm }

    public sealed class MovementPreviewState
    {
        public HexCoord? Destination { get; private set; }
        public bool TouchLocked { get; private set; }

        public void Hover(HexCoord? destination)
        {
            if (!TouchLocked) Destination = destination;
        }

        public MovementPreviewDecision Tap(HexCoord destination, bool reachable)
        {
            if (!reachable) { Clear(); return MovementPreviewDecision.None; }
            if (TouchLocked && Destination == destination)
                return MovementPreviewDecision.Confirm;
            Destination = destination;
            TouchLocked = true;
            return MovementPreviewDecision.Preview;
        }

        public void Clear()
        {
            Destination = null;
            TouchLocked = false;
        }
    }
}
```

- [ ] Rerun EditMode tests and expect all pass. Commit:

```bash
git add Assets/HexWars/Presentation/MovementPreviewState.cs Assets/HexWars/Presentation/MovementPreviewState.cs.meta Assets/HexWars/Tests
git commit -m "test: define movement preview interactions"
```

---

## Task 4: Render intelligent movement highlights

**Files:**

- Create: `Assets/HexWars/Presentation/MovementHighlightController.cs`
- Test: `Assets/HexWars/Tests/Editor/MovementHighlightControllerTests.cs`
- Create associated Unity `.meta` files.

- [ ] Add failing pure classification tests for `Reachable`, `Route`, `Expensive`, and `Destination`. Expensive means entering terrain with `MoveCost > 1` or positive elevation; destination has highest priority.

- [ ] Expose:

```csharp
public enum MovementHighlightKind { Reachable, Route, Expensive, Destination }
public static MovementHighlightKind Classify(
    GameState state, MovementRoute preview, HexCoord cell, bool isReachable)
```

- [ ] Run the focused EditMode test. Expected: compile failure for missing highlight types.

- [ ] Implement `MovementHighlightController : MonoBehaviour` with:

```csharp
public void Show(GameState state,
    IReadOnlyDictionary<HexCoord, MovementRoute> routes,
    HexCoord? previewDestination)
public void Clear()
```

Create/find `MovementHighlights` under `BoardRenderer`. Pool collider-free GameObjects; each uses `HexMesh.Ring`, an unlit cached material, no shadows, and sits `0.035f` above tile top. Use thin green reachable rings, wider blue route rings, amber expensive/climb rings, and a bright destination ring. Render reachable first and preview states above them. Never change terrain/control materials.

- [ ] Run classification tests, compile check, and inspect one Play Mode frame for colliders, z-fighting, elevation, shadows, and hierarchy cleanup.

- [ ] Commit:

```bash
git add Assets/HexWars/Presentation/MovementHighlightController.cs Assets/HexWars/Presentation/MovementHighlightController.cs.meta Assets/HexWars/Tests/Editor/MovementHighlightControllerTests.cs Assets/HexWars/Tests/Editor/MovementHighlightControllerTests.cs.meta
git commit -m "feat: render intelligent movement highlights"
```

---

## Task 5: Integrate repeated moves, hover, touch, tooltip, and attack finish

**Files:**

- Modify: `Assets/HexWars/Presentation/UnitInputController.cs`
- Modify: `Assets/HexWars/Presentation/UnitTooltip.cs`
- Test: `Assets/HexWars/Tests/Editor/MovementPreviewStateTests.cs`

- [ ] Extract/test exact tooltip route formatting:

```csharp
public static string FormatRoute(MovementRoute route) =>
    $"Route: {route.HorizontalCost} move · {route.VerticalCost} climb · leaves " +
    $"{route.HorizontalRemaining} move / {route.VerticalRemaining} climb";
```

Also test that an attacked active unit says `Movement ended by attack` and does not advertise arithmetic remainder as available.

- [ ] Delete the `MovedUnitIds` guard that blocks second moves. Resolve clicked destinations against the selected unit's cached route map; preserve the existing friendly toast for absent/illegal destinations.

- [ ] Add `MovementPreviewState`, `MovementHighlightController`, route-map cache, cached `GameState` reference, and selected-ID key. Refresh on selection or state-reference change. Clear on invalid selection, attack, turn change, game over, build/deploy mode, read-only/demo mode, or opponent-turn waiting.

- [ ] Desktop: hovering a reachable tile previews; leaving clears only the emphasized path; clicking a reachable tile issues `MoveUnit` immediately; selection changes/unrelated clicks clear preview.

- [ ] Touch (`Pointer.current is Touchscreen`): first reachable tap previews, second same tap confirms, another reachable tap switches, unrelated tap clears. Continue using existing UI and drag thresholds.

- [ ] After a successful move, reacquire the same unit and rebuild routes from the new state. After a successful attack, clear preview and auto-advance immediately. Treat `AttackedUnitIds` alone as finished in both `AutoAdvance` and next-actionable selection; do not fill `MovementSpent` artificially.

- [ ] Pass the previewed route to `UnitTooltip.Show`. Show route costs while previewed and `Movement ended by attack` after attack. Map `MovementEndedByAttack` to `Attacking ends movement for this unit.` Replace the inaccurate tip with `Green rings show reachable hexes. Hover to preview the route.`

- [ ] Run compile and presentation EditMode checks. In Play Mode verify desktop repeated movement, touch two-tap behavior, preview switching/cancel, move-then-attack auto-advance, attack-then-move rejection, and no actionable overlays in build/deploy/read-only/opponent-turn states. Inspect Unity logs.

- [ ] Commit:

```bash
git add Assets/HexWars/Presentation/UnitInputController.cs Assets/HexWars/Presentation/UnitTooltip.cs Assets/HexWars/Tests/Editor/MovementPreviewStateTests.cs
git commit -m "feat: preview and continue unit movement"
```

---

## Task 6: Animate the accepted engine route

**Files:**

- Create: `Assets/HexWars/Presentation/MovementPresentationRoute.cs`
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs`
- Test: `Assets/HexWars/Tests/Editor/MovementPresentationRouteTests.cs`
- Create associated Unity `.meta` files.

- [ ] Add a failing pure test on a board where `HexPath.Line` crosses a blocked cell but the legal route detours. Assert:

```csharp
MovementPresentationRoute.Resolve(previous, command)
```

equals `MovementService.Routes(previous, unit)[command.Dest].Cells` and differs from the straight line.

- [ ] Run the focused EditMode test. Expected: missing adapter failure.

- [ ] Implement `Resolve(GameState previous, MoveUnit command)` by finding the issuer's living unit and returning the engine route cells. Return an empty array if unit/route is absent. Never fall back to `HexPath.Line`.

- [ ] In `ActionPresenter.PlayMove`, replace `HexPath.Line` with the adapter. If empty, `yield break` so the normal commit synchronizes to `item.Next`. Use the real route for fog `VisibleSpan`, elevations, pop-in/out, and every tween hop.

- [ ] Run compile/EditMode checks and Play Mode detour/climb scenarios. Confirm highlight, tooltip cost, animation, fog clipping, and final cell agree. Inspect Unity logs.

- [ ] Commit:

```bash
git add Assets/HexWars/Presentation/ActionPresenter.cs Assets/HexWars/Presentation/MovementPresentationRoute.cs Assets/HexWars/Presentation/MovementPresentationRoute.cs.meta Assets/HexWars/Tests/Editor/MovementPresentationRouteTests.cs Assets/HexWars/Tests/Editor/MovementPresentationRouteTests.cs.meta
git commit -m "fix: animate authoritative movement routes"
```

---

## Task 7: Rebuild the Unity engine plugin and verify end to end

**Files:**

- Update generated artifact: `Assets/HexWars/Plugins/HexWars.Engine.dll`

- [ ] Run the complete engine suite:

```bash
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```

Expected: zero failures.

- [ ] From Windows PowerShell run `./engine/build-to-unity.ps1`. Confirm the Release netstandard2.1 engine DLL is copied to `Assets/HexWars/Plugins/HexWars.Engine.dll`.

- [ ] Run Unity `check_compile_errors`, all presentation EditMode tests, relevant PlayMode tests, and determinism tests. Inspect logs for exceptions, input errors, missing shaders, and destroyed-object errors.

- [ ] Verify local, AI, and replay/network presentation: repeated hops shrink routes; attack removes routes; desktop/touch semantics match; blocked cells are never traversed; displayed costs equal `MovementSpent` deltas; accepted moves animate the engine route; missing historical route skips tween and still commits; next turn restores routes.

- [ ] Review scope:

```bash
git status --short
git diff --check
git diff --stat
```

Confirm no wire-format files, scenes, unrelated line endings, or main-worktree changes entered the feature branch.

- [ ] Commit the plugin:

```bash
git add Assets/HexWars/Plugins/HexWars.Engine.dll
git commit -m "build: update Unity engine plugin for movement routes"
```

- [ ] Rerun all final gates from the clean feature worktree and record commands/pass totals. Do not call the feature complete while Unity/Coplay verification is unavailable.

---

## Plan Self-Review Checklist

- [ ] Every approved requirement has an implementation and verification step.
- [ ] `MoveUnit` wire/replay formats remain untouched.
- [ ] Equal-cost route selection is deterministic and H/V Pareto labels survive.
- [ ] Attack closes movement without corrupting `MovementSpent`.
- [ ] Desktop hover and touch confirmation have independent tests.
- [ ] Highlight rings have no colliders and do not mutate tile materials.
- [ ] Presenter has no geometric-line fallback.
- [ ] New Unity assets have `.meta` files.
- [ ] No placeholders or unresolved production types remain.
- [ ] Commits contain only feature files and no attribution trailers.
