# Intelligent Movement Routes Design

Date: 2026-07-21
Status: Approved in conversation; awaiting written-spec review

## Goal

Make movement readable and incremental. A selected unit may move repeatedly while it has movement budget, the board previews the exact legal route and its cost, and the movement animation follows that route. Attacking closes movement for that unit for the rest of the turn.

## Current Behavior and Root Cause

The engine already treats movement as a per-turn budget shared across multiple hops. `MovementService` subtracts `GameState.MovementSpent` from the unit's horizontal and vertical budgets and returns destinations reachable with the remainder (`engine/HexWars.Engine/MovementService.cs:18-69`). `GameEngine` records the incremental cost after each accepted move (`engine/HexWars.Engine/GameEngine.cs:167-193`).

The presentation layer blocks that supported behavior. `UnitInputController` rejects every subsequent click as soon as the unit appears in `MovedUnitIds`, even if `MovementSpent` shows budget remaining (`Assets/HexWars/Presentation/UnitInputController.cs:132-139`).

Movement animation also does not represent the path validated by the engine. `ActionPresenter` constructs a straight geometric `HexPath.Line` from origin to destination (`Assets/HexWars/Presentation/ActionPresenter.cs:133-181`). That line can cross blocked, impassable, expensive, or off-board cells even though the engine found a different legal route.

There is no board movement-highlight implementation today. The selection tip says green and red highlights exist (`Assets/HexWars/Presentation/UnitInputController.cs:225-238`), but the board only renders terrain, control tint, and entity tokens (`Assets/HexWars/Presentation/BoardRenderer.cs:80-100`).

## User-Visible Requirements

1. A unit can receive another move command after a partial move while horizontal and vertical budgets permit it.
2. Moving recalculates the remaining destinations from the unit's new position and reduced budgets.
3. Attacking immediately closes any remaining movement for that unit.
4. Moving and then attacking remains legal. Attacking and then moving is illegal.
5. Selecting an active owned unit shows every reachable destination with a subtle green hex ring.
6. On desktop, hovering a reachable destination previews the exact route. Clicking commits that displayed route.
7. On touch, the first tap on a reachable destination locks its preview. A second tap on the same destination commits the move. Tapping another reachable destination changes the preview; selecting another unit or tapping elsewhere cancels it.
8. The preview emphasizes the chosen route, identifies climbing or expensive terrain steps, identifies the destination, and reports horizontal cost, climb cost, and movement remaining.
9. The movement animation follows the same hex sequence shown by the preview.
10. After a partial move, the unit remains selected. After an attack, preview state clears and selection advances to the next unit because the attacker is finished.

## Non-Goals

- User-authored waypoints or manual path editing.
- Threat-aware, cover-aware, or enemy-zone scoring.
- Changing `MoveUnit` network or replay wire format.
- Adding attack-target highlighting in this slice. The inaccurate selection tip will instead be changed to describe the movement route behavior that exists.
- Rebalancing terrain or movement statistics.

## Engine-Owned Route Model

`MovementService` remains the single authority for movement legality, cost, and route selection. Introduce an immutable route result with:

- ordered cells, including origin and destination;
- horizontal movement cost;
- vertical/climb cost;
- horizontal movement remaining after arrival;
- vertical movement remaining after arrival.

Add a route query that returns the chosen route for every currently reachable destination. Existing `ReachableTiles` and `ReachableCosts` become projections of this route query, keeping their public behavior and current AI callers intact.

The search continues to maintain Pareto-optimal horizontal/vertical labels because neither budget can be collapsed safely into a single scalar. Each label also retains an immutable predecessor so a legal path can be reconstructed. For a destination with multiple legal labels, choose deterministically by:

1. lower horizontal cost;
2. lower vertical cost;
3. fewer hops;
4. lexicographically smaller coordinate sequence (`Q`, then `R`) as the final tie-breaker.

This preserves the current cheapest-horizontal-then-cheapest-vertical policy while making ties stable across local play, server echoes, AI matches, and replay playback.

Occupied and impassable cells remain excluded, and each entered tile contributes its terrain movement cost plus positive elevation gain exactly as it does now (`engine/HexWars.Engine/MovementService.cs:48-64`).

## Attack Ends Movement

`AttackedUnitIds` is the authoritative record that a unit attacked this turn. The movement route query returns no routes for a unit in that collection. `GameEngine.ApplyMoveUnit` also checks this condition directly and rejects a manually constructed or remote move command with a specific movement-ended-by-attack reason before route lookup.

An attack does not falsify `MovementSpent` by marking unused points as consumed. The spent values continue to mean points actually paid for movement. Presentation code reports movement as closed after an attack even if those raw values leave arithmetic remainder.

`LegalMoves` automatically omits post-attack movement because it consumes the route-derived reachable destinations (`engine/HexWars.Engine/LegalMoves.cs:41-57`). Engine validation remains the final authority for local input, AI, network commands, and replay commands.

Current turn-policy accounting is unchanged: a prior movement action and a later attack retain their existing action-count effects. This change only closes unused movement after the attack.

## Presentation Components

### MovementHighlightController

Add a focused presentation component responsible only for transient movement overlays. It receives the current route map and optional preview destination; it does not calculate movement rules.

It creates a `MovementHighlights` child under the board and uses pooled, collider-free `HexMesh.Ring` objects positioned slightly above each tile top. This keeps highlights separate from terrain/control materials, which `BoardRenderer.UpdateControlTint` replaces during state synchronization (`Assets/HexWars/Presentation/BoardRenderer.cs:85-100`). Raised rings avoid top-face z-fighting and use unlit materials with shadow casting disabled.

Visual states:

- thin green ring: legal destination;
- strong blue ring/path: the chosen route;
- amber route ring: entering terrain with movement cost greater than one or making a positive climb;
- bright destination ring: hovered or touch-locked destination.

The controller clears all overlays when selection becomes invalid, the turn changes, an attack is committed, deployment/build mode takes precedence, or the game state is replaced.

### UnitInputController

Remove the `MovedUnitIds` guard that currently blocks every second move. Cache the route map for the selected unit and refresh it whenever selection or the `GameState` reference changes.

Desktop pointer behavior:

- hovering a reachable tile sets the preview destination;
- leaving reachable tiles clears only the emphasized route, not the reachable rings;
- clicking a reachable hovered destination issues `MoveUnit` immediately.

Touch behavior:

- first tap on a reachable tile records the pending destination without issuing a command;
- second tap on that same destination issues `MoveUnit`;
- another reachable tap replaces the pending destination;
- unit selection, an unrelated board tap, build/deploy mode, turn change, or attack clears the pending destination.

After a successful move, the existing selection reacquisition stays in place, but route data and overlays refresh from the new state. After a successful attack, auto-advance runs immediately because attack now closes both remaining actions for that unit.

### Route Cost Readout

Extend the existing selected-unit tooltip/status area rather than creating another floating canvas. While a route is previewed it shows:

`Route: <H> move · <V> climb · leaves <remaining H> move / <remaining V> climb`

When no route is previewed, the existing per-turn remaining budget display remains. After attack, it explicitly says movement ended by attack rather than showing unused arithmetic remainder as available.

### ActionPresenter

For an accepted `MoveUnit`, recompute the engine-owned route from the queued item's `Prev` state and moving unit. Animate each ordered route cell instead of using `HexPath.Line`. Fog visibility clipping and camera focus operate over that same real route.

Because `MoveUnit` still carries only issuer, unit ID, and destination, network and replay formats do not change. The route is deterministic from the authoritative pre-command state. If an accepted historical/inconsistent item cannot produce a route, presentation skips the tween and lets the existing commit synchronize directly to engine truth; it must not invent a misleading straight path.

## Data Flow

1. Selection or state change asks `MovementService` for route results.
2. `UnitInputController` supplies all destinations to `MovementHighlightController`.
3. Hover or first touch tap selects one result and displays its path and costs.
4. Confirmation issues the existing `MoveUnit` command.
5. `GameEngine` independently resolves that destination through the same route service and charges the returned costs.
6. `ActionPresenter` resolves the route again from its queued `Prev` state and animates that cell sequence.
7. The committed state refreshes selection, remaining routes, tooltip text, and overlays.

## Rejections and Edge Cases

- Occupied, impassable, off-board, or over-budget destinations never receive reachable rings and retain the existing explanatory toast when clicked.
- Movement after attack receives a specific friendly message explaining that attacking ended movement.
- A destination becoming illegal between a touch preview and server confirmation is rejected by the authoritative engine/server; the pending preview clears when the echoed/rejected state resolves.
- Zero remaining horizontal movement produces no route results.
- Zero remaining vertical movement still permits level movement and descent but not ascent, matching current rules.
- Other units remain obstacles; paths never pass through them.
- Build mode, deployment mode, read-only spectators, demo mode, and opponent-turn waiting do not expose actionable movement previews.

## Testing Strategy

Follow test-driven development: add each regression test and observe the expected failure before production changes.

Engine tests:

- returns an ordered legal path around an occupied or impassable cell;
- reports costs matching the cost charged by `GameEngine`;
- chooses deterministic routes for equal-cost alternatives;
- preserves Pareto behavior where horizontal and vertical trade off;
- permits repeated partial moves until the budget is exhausted;
- permits move then attack;
- rejects attack then move and exposes no post-attack reachable routes;
- omits post-attack moves from `LegalMoves`;
- resets route availability on the unit's next turn.

Presentation/EditMode tests:

- desktop hover selects the route and click confirms it;
- touch first tap previews, second identical tap confirms;
- touching another destination switches preview;
- unrelated taps and selection changes cancel preview;
- highlight classification identifies reachable, route, climb/expensive, and destination cells;
- presenter consumes the engine route rather than a straight geometric line.

Verification:

- run the focused engine test fixture, then the complete engine test suite;
- run Unity EditMode/PlayMode tests relevant to input, presentation, and determinism;
- run Unity `check_compile_errors` after every C# edit and fix all errors;
- exercise desktop and touch flows in Play Mode;
- inspect Unity runtime logs rather than inferring runtime behavior from code;
- confirm overlays persist correctly after a partial move and disappear after attack/turn change;
- confirm route animation, fog clipping, and destination state agree in local, AI, and network/replay playback.

The Unity MCP connection is unavailable in the current session. Implementation cannot be declared complete until those compile, runtime-log, and Play Mode checks are available and pass.

## Files Expected to Change

- `engine/HexWars.Engine/MovementService.cs`
- `engine/HexWars.Engine/GameEngine.cs`
- `engine/HexWars.Engine/RejectionReason.cs`
- `engine/HexWars.Engine/LegalMoves.cs`
- `engine/HexWars.Engine.Tests/MovementServiceTests.cs`
- `engine/HexWars.Engine.Tests/MultiHopMovementTests.cs`
- presentation tests under `Assets/HexWars/Editor/`
- `Assets/HexWars/Presentation/UnitInputController.cs`
- `Assets/HexWars/Presentation/UnitTooltip.cs`
- `Assets/HexWars/Presentation/ActionPresenter.cs`
- new `Assets/HexWars/Presentation/MovementHighlightController.cs`
- associated Unity `.meta` files

No scene value or runtime default change is required; the component can be attached at runtime alongside the existing board/input components.
