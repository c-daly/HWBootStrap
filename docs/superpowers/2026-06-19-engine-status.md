# HexWars — Engine Status (autonomous session, 2026-06-19)

## What's built

The **complete headless rules engine** (`engine/HexWars.Engine`, pure netstandard2.1, zero Unity
deps), TDD'd module-by-module — **101 passing tests**, every step committed.

**Data model (immutable):** `HexCoord` (q,r + neighbors/distance, value-equal), `UnitStats`
(9 stats, PointCost = flat sum), `TerrainType`/`TerrainDef`, `GameConfig` (terrain table + all
tunables + `TurnPolicy`), `Tile`, `Board` (column lookup + per-player deployment zones), `Unit`
(3D position: cell + own elevation), `Generator`, `PlayerState`, `GameState` (+ `Clone`, per-turn
acted-tracking).

**Rules (pure):** `MovementService` (two-budget reachability — horizontal `Movement` + vertical
`VerticalMovement` ascent, descent free, Pareto over both), `TargetingService` (army-wide shared
vision + per-unit `Range`/`RangeArc` + high-ground range bonus; the 0-damage spotter works),
`CombatResolver` (deterministic damage w/ high-ground + terrain defense; bounty), `Economy`
(generator income), `WinCheck` (`Evaluate` total value, elimination w/ opening guard, round-cap
tie-break, `IsTerminal`).

**Orchestration:** `Command` (`CreateUnit`/`DeployUnit`/`DeployGenerator`/`MoveUnit`/`AttackUnit`/
`EndTurn`) + `GameEngine.Apply` — the single non-mutating mutation path, `Result`/`RejectionReason`,
centralized post-command win check. `LegalMoves` enumeration. `ITurnPolicy` (`AllUnitsPolicy`
default + `OneActionPolicy` auto-end).

**Agent API:** `IAgent` + `RandomAgent` (seeded) + `Match` (headless loop). Self-play test proves
games **terminate** and are **reproducible** for a seed.

**Board generation:** `IBoardGenerator`, `RandomBoardGenerator` (seeded, mirror-symmetric terrain
+ elevation, per-player zones), `AuthoredBoardGenerator`.

## How to run the tests

```
"/c/Program Files/dotnet/dotnet.exe" test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```
(.NET 8 SDK was installed this session. Engine targets netstandard2.1 so the same source/DLL drops
into Unity later.)

## Intentionally deferred

- **JSON serialization of `GameState`/`Command`.** A stated M1 *foundation*, but only exercised by
  the later RL/networking milestone, and the clean approach needs an **observation-schema design**
  (you serialize an RL observation + actions, not the whole `GameConfig`, which holds interface-typed
  members). Deferred to that milestone so it isn't built speculatively. Not needed for hotseat.

## Next (with the user present)

The **Unity presentation layer** (`HexWars.Presentation`): wire the engine DLL/source into the
Unity project, `BoardRenderer` (stacked hex prefab columns, terrain materials, starfield skybox),
`UnitView`/`GeneratorView` (cyan vs red, black outlines), `CameraRig`, `InputController`, the HUD +
the self-documenting create-unit point-allocation panel, `GamePresenter`, hotseat flow. Best done
interactively (visual verification + actually playing both sides).
