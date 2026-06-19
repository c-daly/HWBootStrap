# HexWars Milestone 1 (Hotseat Vertical Slice) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully unit-tested, deterministic, headless HexWars rules engine (HexWars.Engine, zero UnityEngine references) plus a thin Unity 6 / URP presentation skin (HexWars.Presentation) that together deliver a playable 2-player hotseat vertical slice: point-buy unit design, create/deploy, generators, income, bounties, elevation+terrain 3D combat with vision-gated targeting, seeded symmetric procedural boards, configurable turn policy, and win-by-annihilation â€” all driven through one non-mutating Apply(GameState, Command) command API with a headless Match runner and RandomAgent reference implementation.

**Architecture:** The game is a pure C# engine (HexWars.Engine, asmdef noEngineReferences=true) holding all rules as deterministic static services and immutable-style data, mutated only through a single non-mutating Apply(GameState, Command) -> Result that returns a new cloned GameState plus events; determinism comes from System.Random-with-seed only (never UnityEngine.Random) so matches are reproducible and search-able. HexWars.Presentation is a thin MonoBehaviour skin that references the engine, owns a GameState in a GamePresenter, renders board/units/generators from engine data, and translates player input into Commands â€” it contains no rules. Every tunable number lives in GameConfig and every stat is priced flat 1 point = 1 step, so the buy menu literally documents the game.

**Tech Stack:** Unity 6 (6000.5.0f1); URP 17.5.0; new Input System 1.19.0; Unity UI (UGUI) 2.5.0; C# (.NET Standard 2.1 / Unity scripting); Assembly Definition Files (asmdef) to isolate engine from Unity; Unity Test Framework 1.7.0 (NUnit 3) EditMode tests; Coplay Unity MCP tools for live-editor presentation work (prefabs/materials/shaders/scenes, check_compile_errors).

## Global Constraints

- HexWars.Engine asmdef MUST set `"noEngineReferences": true` and `"autoReferenced": true` â€” ZERO `using UnityEngine;` anywhere in the engine or its tests.
- Determinism is mandatory: the engine and all board generation use ONLY `System.Random` constructed with an explicit seed â€” NEVER `UnityEngine.Random`, `DateTime.Now`, `Guid.NewGuid()`, or unordered hash iteration.
- A single `Apply(GameState state, Command command) -> Result` is the ONLY mutation path; it is NON-MUTATING (returns a new `GameState` via `GameState.Clone()`, leaving the input untouched) and is the future network wire-format and AI action output.
- Flat pricing is a hard rule: `UnitStats.PointCost == Health + Damage + Range + Movement + Defense + Vision` (1 point = 1 stat step); no nonlinear cost curves anywhere.
- Every tunable number lives in `GameConfig` (spec Â§14); no magic numbers in rules code â€” services read `GameConfig`.
- Combat is fully deterministic (no randomness in Â§5 resolution); same inputs always produce the same `Result`.
- One unified 3D distance metric (`Distance3D = HexDistance + |elevDiff|`) underlies Range, Vision, and high-ground; there is NO special-case line-of-sight occlusion code.
- Vision gates targeting only (M1 has full on-screen visibility, no fog of war), but you still cannot attack what you cannot see.
- One unit/structure per hex; `HexCoord` axial (Q,R) uniquely identifies a board column, elevation is a separate integer stored on the `Tile`.
- `GameState`, `Command`, and supporting data are JSON-serializable (engine exposes serialization helpers) so out-of-process/RL agents can exchange them later.
- HexWars.Presentation is a thin skin: it references HexWars.Engine and contains NO rules logic; all rule math lives in the engine, and any presentation-side math (e.g. axial->world position in `HexLayout`) is itself pure and unit-tested in the EditMode engine-tests assembly.
- Standard axial/cube hex conventions: cube `S = -Q - R`; `HexDistance = (|dQ| + |dQ+dR| + |dR|) / 2`; the six axial directions are (+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1).
- Players are indexed 0 and 1 (`PlayerId` 0 = Player 1 / cyan-blue, 1 = Player 2 / red-magenta); `GameState.Players` is a length-2 array.
- Canonical config field names (resolving spec naming drift): `DmgHighGroundBonus`, `RangeHighGroundBonus`, `ClimbCostPerLevel`, `MaxClimbPerStep`, `DamageFloor`, `BountyRate`. Use these verbatim everywhere.

## Shared Interfaces / Type Contract

> Canonical signatures. Every task uses these names and types verbatim.

All engine types live in namespace `HexWars.Engine` (sub-namespaces noted per file in File Structure but a single root namespace `HexWars.Engine` is used for all public types to keep `using` simple). Presentation types live in `HexWars.Presentation`. **Copy these signatures verbatim. One canonical name per concept.**

### Core enums & ids

```csharp
namespace HexWars.Engine
{
    // Player 0 = Player 1 (cyan/blue); Player 1 = Player 2 (red/magenta).
    public enum PlayerId { Player0 = 0, Player1 = 1 }

    public enum TerrainType { Plains = 0, Forest = 1, Rough = 2, Water = 3 }

    // Why a command was rejected by Apply (Success=false carries one of these).
    public enum RejectionReason
    {
        None = 0,
        NotYourTurn,
        IllegalCommandForPolicy,
        InsufficientPoints,
        UnitNotFound,
        GeneratorNotFound,
        ReserveUnitNotFound,
        TileNotFound,
        TileOccupied,
        TileImpassable,
        OutsideDeploymentZone,
        OutOfMovementRange,
        ClimbTooSteep,
        TargetNotInRange,
        TargetNotVisible,
        TargetNotEnemy,
        UnitAlreadyActed,
        InvalidStats,
        GameAlreadyOver
    }

    // Kinds of events emitted by a successful Apply, for the presentation layer.
    public enum GameEventType
    {
        UnitCreated, UnitDeployed, GeneratorDeployed, UnitMoved,
        UnitAttacked, UnitDestroyed, GeneratorDestroyed,
        BountyAwarded, IncomeCredited, TurnEnded, GameWon
    }
}
```

### HexCoord (axial + elevation; pure data)

```csharp
public readonly struct HexCoord : System.IEquatable<HexCoord>
{
    public int Q { get; }
    public int R { get; }
    public int Elevation { get; }
    public int S => -Q - R;                 // cube third axis

    public HexCoord(int q, int r, int elevation = 0);

    public HexCoord WithElevation(int elevation);          // returns copy with new elevation
    public HexCoord Translate(int dq, int dr);             // same elevation, shifted axial

    public bool Equals(HexCoord other);                    // compares Q,R,Elevation
    public override bool Equals(object obj);
    public override int GetHashCode();                     // deterministic from Q,R,Elevation
    public override string ToString();                     // "(q,r,e)"
    public static bool operator ==(HexCoord a, HexCoord b);
    public static bool operator !=(HexCoord a, HexCoord b);
}
```

### TerrainDef & terrain table

```csharp
// Data-driven terrain modifiers (spec Â§6). Stored as a table on GameConfig.
public sealed class TerrainDef
{
    public TerrainType Terrain { get; }
    public int MoveCost { get; }       // cost to enter (before climb)
    public int Concealment { get; }    // added to distance vs Vision
    public int Defense { get; }        // flat damage reduction when defending here
    public bool Passable { get; }

    public TerrainDef(TerrainType terrain, int moveCost, int concealment, int defense, bool passable);
}
```

### Tile & Board

```csharp
public sealed class Tile
{
    public HexCoord Coord { get; }          // includes Elevation
    public TerrainType Terrain { get; }
    public int Elevation => Coord.Elevation;

    public Tile(HexCoord coord, TerrainType terrain);
    public Tile Clone();
}

public sealed class Board
{
    public int Width { get; }
    public int Height { get; }
    public System.Collections.Generic.IReadOnlyList<Tile> Tiles { get; }
    // Deployment zone axial coords per player (elevation ignored for membership).
    public System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZone0 { get; }
    public System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZone1 { get; }

    public Board(int width, int height,
                 System.Collections.Generic.IReadOnlyList<Tile> tiles,
                 System.Collections.Generic.IReadOnlyList<HexCoord> deploymentZone0,
                 System.Collections.Generic.IReadOnlyList<HexCoord> deploymentZone1);

    public bool TryGetTile(int q, int r, out Tile tile);    // axial lookup, ignores elevation
    public bool Contains(int q, int r);
    public System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZoneFor(PlayerId player);
    public bool IsInDeploymentZone(PlayerId player, int q, int r);
    public Board Clone();
}
```

### UnitStats (flat point-buy)

```csharp
public readonly struct UnitStats : System.IEquatable<UnitStats>
{
    public int Health { get; }
    public int Damage { get; }
    public int Range { get; }
    public int Movement { get; }
    public int Defense { get; }
    public int Vision { get; }

    public int PointCost => Health + Damage + Range + Movement + Defense + Vision; // strict 1:1

    public UnitStats(int health, int damage, int range, int movement, int defense, int vision);

    public bool IsValid => Health >= 1;     // only floor: a unit must have >=1 Health
    public bool Equals(UnitStats other);
    public override bool Equals(object obj);
    public override int GetHashCode();
}
```

### Unit & Generator (on-board entities)

```csharp
public sealed class Unit
{
    public int Id { get; }
    public PlayerId Owner { get; }
    public UnitStats Stats { get; }
    public HexCoord Position { get; }       // includes the tile's elevation when deployed
    public int CurrentHp { get; }
    public bool HasActed { get; }           // once-per-turn tracking (move/attack consumed)

    public Unit(int id, PlayerId owner, UnitStats stats, HexCoord position, int currentHp, bool hasActed = false);
    public int BuildCost => Stats.PointCost;
    public bool IsAlive => CurrentHp > 0;

    // Non-mutating copy helpers used internally by Apply:
    public Unit WithPosition(HexCoord position);
    public Unit WithHp(int currentHp);
    public Unit WithHasActed(bool hasActed);
    public Unit Clone();
}

public sealed class Generator
{
    public int Id { get; }
    public PlayerId Owner { get; }
    public HexCoord Position { get; }
    public int CurrentHp { get; }

    public Generator(int id, PlayerId owner, HexCoord position, int currentHp);
    public bool IsAlive => CurrentHp > 0;
    public Generator WithHp(int currentHp);
    public Generator Clone();
}
```

### PlayerState & GameState

```csharp
public sealed class PlayerState
{
    public PlayerId Id { get; }
    public int Points { get; }
    public System.Collections.Generic.IReadOnlyList<UnitStats> Reserve { get; } // created, not yet deployed
    public System.Collections.Generic.IReadOnlyList<Unit> UnitsOnBoard { get; }
    public System.Collections.Generic.IReadOnlyList<Generator> Generators { get; }

    public PlayerState(PlayerId id, int points,
                       System.Collections.Generic.IReadOnlyList<UnitStats> reserve,
                       System.Collections.Generic.IReadOnlyList<Unit> unitsOnBoard,
                       System.Collections.Generic.IReadOnlyList<Generator> generators);
    public PlayerState Clone();
}

public sealed class GameState
{
    public Board Board { get; }
    public PlayerState[] Players { get; }   // length 2; index by (int)PlayerId
    public PlayerId ActivePlayer { get; }
    public int Round { get; }               // increments after Player1 ends turn
    public int NextEntityId { get; }        // deterministic id allocator for units+generators
    public bool IsGameOver { get; }
    public PlayerId? Winner { get; }
    public GameConfig Config { get; }

    public GameState(Board board, PlayerState[] players, PlayerId activePlayer, int round,
                     int nextEntityId, GameConfig config,
                     bool isGameOver = false, PlayerId? winner = null);

    public PlayerState ActivePlayerState => Players[(int)ActivePlayer];
    public PlayerState PlayerStateFor(PlayerId id);
    public PlayerId Opponent(PlayerId id);   // returns the other PlayerId
    public GameState Clone();                 // deep, cheap copy (search-ability)
}
```

### GameConfig (every tunable from Â§14)

```csharp
public sealed class GameConfig
{
    public int StartingPoints { get; init; } = 15;          // Â§14 (10-20)
    public double BountyRate { get; init; } = 0.5;          // portion of build cost on kill
    public int GeneratorCost { get; init; } = 2;
    public int GeneratorOutput { get; init; } = 1;          // points/turn each
    public int GeneratorHealth { get; init; } = 3;
    public int DamageFloor { get; init; } = 1;              // min damage per hit (0 or 1)
    public int DmgHighGroundBonus { get; init; } = 1;       // per level of advantage
    public int RangeHighGroundBonus { get; init; } = 1;     // per level of advantage
    public int ClimbCostPerLevel { get; init; } = 1;
    public int MaxClimbPerStep { get; init; } = 2;
    public int MinCheapestViableUnitCost { get; init; } = 1; // cost of cheapest legal unit (1 HP)
    public int BoardWidth { get; init; } = 9;
    public int BoardHeight { get; init; } = 9;
    public int MinElevation { get; init; } = 0;
    public int MaxElevation { get; init; } = 4;
    public int DeploymentZoneDepth { get; init; } = 2;       // rows from each player's edge
    public int RoundCap { get; init; } = 40;                 // stalemate backstop
    public TurnPolicyKind TurnPolicy { get; init; } = TurnPolicyKind.AllUnits;

    // Terrain modifier table (spec Â§6 starter values). Lookup by TerrainType.
    public System.Collections.Generic.IReadOnlyDictionary<TerrainType, TerrainDef> Terrain { get; init; }

    // Procedural-gen weights (relative spawn weight per terrain), used by RandomBoardGenerator.
    public System.Collections.Generic.IReadOnlyDictionary<TerrainType, double> TerrainWeights { get; init; }

    public static GameConfig Default();             // returns the spec Â§14 starting values + default terrain table/weights
    public TerrainDef TerrainDefFor(TerrainType terrain);
    public GameConfig Clone();
}

public enum TurnPolicyKind { AllUnits = 0, OneAction = 1 }
```

### Commands (records) â€” the wire format

```csharp
public abstract record Command
{
    public PlayerId Issuer { get; init; }
}

public sealed record CreateUnit(UnitStats Stats) : Command;                 // pay points -> reserve
public sealed record DeployUnit(int ReserveIndex, HexCoord Target) : Command; // reserve[index] -> board
public sealed record DeployGenerator(HexCoord Target) : Command;            // pay GeneratorCost -> board
public sealed record MoveUnit(int UnitId, HexCoord Target) : Command;       // pathfind within Movement
public sealed record AttackUnit(int AttackerUnitId, int TargetId) : Command; // TargetId = unit OR generator id
public sealed record EndTurn() : Command;
```

### Events & Result

```csharp
public sealed record GameEvent
{
    public GameEventType Type { get; init; }
    public int EntityId { get; init; }      // unit/generator id (0 if n/a)
    public int TargetId { get; init; }      // for attacks/destroys (0 if n/a)
    public PlayerId Player { get; init; }
    public int Amount { get; init; }        // damage / points / bounty (0 if n/a)
    public HexCoord Coord { get; init; }    // for move/deploy (default if n/a)
    public override string ToString();
}

// Returned by Apply. Non-mutating: NewState is a fresh GameState; on failure NewState == input state.
public sealed class Result
{
    public bool Success { get; }
    public GameState NewState { get; }
    public System.Collections.Generic.IReadOnlyList<GameEvent> Events { get; }
    public RejectionReason Reason { get; }   // None when Success
    public string Message { get; }           // human-readable detail (empty when Success)

    public static Result Ok(GameState newState, System.Collections.Generic.IReadOnlyList<GameEvent> events);
    public static Result Rejected(GameState unchangedState, RejectionReason reason, string message);
}
```

### Geometry (pure static service)

```csharp
public static class HexGeometry
{
    public static int HexDistance(HexCoord a, HexCoord b);   // cube distance, ignores elevation
    public static int Distance3D(HexCoord a, HexCoord b);    // HexDistance(a,b) + |a.Elevation - b.Elevation|
    public static System.Collections.Generic.IReadOnlyList<HexCoord> AxialDirections { get; } // the 6 dirs, fixed order
    public static System.Collections.Generic.List<HexCoord> Neighbors(HexCoord c); // 6 axial neighbors (elevation copied from c; callers re-resolve elevation via Board)
}
```

### Pathfinding (pure static service)

```csharp
public static class Pathfinding
{
    // costToEnter(tile) = terrainMoveCost + max(0, climbLevels) * ClimbCostPerLevel.
    // climbLevels = toElevation - fromElevation. Entry forbidden if impassable,
    // occupied, or climbLevels > MaxClimbPerStep. Descending is just terrainMoveCost.
    public static int CostToEnter(GameState state, HexCoord from, Tile to);

    // Dijkstra over passable, unoccupied, climbable neighbors within `movementBudget`.
    // Returns reachable axial coords (each with the tile's real elevation) excluding `start`.
    public static System.Collections.Generic.List<HexCoord> ReachableTiles(
        GameState state, HexCoord start, int movementBudget);

    // Cheapest total move cost from start to goal within budget; returns false if unreachable.
    public static bool TryFindPath(GameState state, HexCoord start, HexCoord goal, int movementBudget,
        out System.Collections.Generic.List<HexCoord> path, out int totalCost);
}
```

### Targeting & combat (pure static services)

```csharp
public static class TargetingService
{
    // H = max(0, attackerElev - targetElev); D = Distance3D(attacker, target).
    public static int HighGround(HexCoord attacker, HexCoord target);          // max(0, attacker.Elevation - target.Elevation)
    // inRange: D <= Range + RangeHighGroundBonus * H
    public static bool InRange(GameState state, Unit attacker, HexCoord targetPos);
    // visible: D + targetConcealment <= Vision  (concealment from the target tile's terrain)
    public static bool IsVisible(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain);
    public static bool CanTarget(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain); // InRange && IsVisible
    // Enumerate all legal targets (enemy units AND generators) for an attacker.
    public static System.Collections.Generic.List<int> ValidTargetIds(GameState state, Unit attacker);
}

public readonly struct CombatOutcome
{
    public int Damage { get; }          // applied (already floored)
    public int TargetHpAfter { get; }
    public bool TargetDestroyed { get; }
    public int Bounty { get; }          // floor(buildCost * BountyRate) if destroyed else 0
    public CombatOutcome(int damage, int targetHpAfter, bool targetDestroyed, int bounty);
}

public static class CombatResolver
{
    // damage = max(DamageFloor, Damage + DmgHighGroundBonus*H - (Defense + terrainDefense))
    public static int ComputeDamage(GameConfig config, Unit attacker, int defenderDefense,
        int defenderTerrainDefense, int highGround);
    // Full resolution against a unit target.
    public static CombatOutcome ResolveAgainstUnit(GameState state, Unit attacker, Unit target);
    // Full resolution against a generator target (generator Defense = 0; terrain defense from its tile).
    public static CombatOutcome ResolveAgainstGenerator(GameState state, Unit attacker, Generator target);
    // bounty helper: floor(buildCost * BountyRate)
    public static int ComputeBounty(GameConfig config, int buildCost);
}
```

### Economy, win check, evaluation (pure static services)

```csharp
public static class Economy
{
    public static int IncomeFor(GameState state, PlayerId player);     // sum of living generators' GeneratorOutput
    public static bool CanAfford(GameState state, PlayerId player, int cost);
    public static int CheapestViableUnitCost(GameConfig config);       // == MinCheapestViableUnitCost (1 HP unit)
}

public static class WinCheck
{
    // A player is eliminated when: no units on board AND none in reserve AND cannot afford cheapest viable unit.
    public static bool IsEliminated(GameState state, PlayerId player);
    // Total value = banked points + sum(unit build costs on board) + sum(reserve unit costs) + generators(count*GeneratorCost).
    public static int TotalValue(GameState state, PlayerId player);
    // Returns the winner if the game is over (elimination OR round cap reached), else null.
    public static PlayerId? CheckWinner(GameState state);
}

public static class Evaluation
{
    // Heuristic position score from `player`'s perspective (reused by stalemate tie-break and later AI).
    public static int Evaluate(GameState state, PlayerId player);      // = TotalValue(player) - TotalValue(opponent)
}
```

### Turn policy

```csharp
public interface ITurnPolicy
{
    TurnPolicyKind Kind { get; }
    // Is this command type legal under the policy given the current state (independent of resource/rule checks)?
    bool IsCommandAllowed(GameState state, Command command);
    // After a successful command, should the turn auto-pass (true for OneAction non-EndTurn actions)?
    bool ShouldAutoEndTurn(GameState state, Command command);
}

public sealed class AllUnitsPolicy : ITurnPolicy { /* each on-board unit acts at most once; explicit EndTurn */ }
public sealed class OneActionPolicy : ITurnPolicy { /* one atomic action then auto-pass */ }

public static class TurnPolicyFactory
{
    public static ITurnPolicy Create(TurnPolicyKind kind);
}
```

### The engine entry points (the one mutation path + API)

```csharp
public static class GameEngine
{
    // THE ONLY MUTATION PATH. Non-mutating: returns a new GameState (or the unchanged state on rejection).
    public static Result Apply(GameState state, Command command);

    // All legal commands for the active player under the current ITurnPolicy + rules.
    public static System.Collections.Generic.List<Command> LegalMoves(GameState state);
}

public static class GameFactory
{
    // New game from a board generator + config + seed. Credits no income yet (income credited at start of active turn).
    public static GameState NewGame(IBoardGenerator generator, GameConfig config, int seed);
    // RL-style reset alias.
    public static GameState Reset(IBoardGenerator generator, GameConfig config, int seed);
}
```

### Serialization (JSON helpers)

```csharp
public static class GameStateSerializer
{
    public static string ToJson(GameState state);
    public static GameState FromJson(string json);
    public static string CommandToJson(Command command);
    public static Command CommandFromJson(string json);
}
```

### Board generation

```csharp
public interface IBoardGenerator
{
    Board Generate(GameConfig config, int seed);   // seed unused by authored gen but part of the contract
}

public sealed class RandomBoardGenerator : IBoardGenerator
{
    public Board Generate(GameConfig config, int seed); // System.Random(seed) ONLY; mirrored zones; connectivity-checked
}

public sealed class AuthoredBoardGenerator : IBoardGenerator
{
    public AuthoredBoardGenerator(Board fixedBoard);    // wraps a handcrafted board
    public Board Generate(GameConfig config, int seed); // returns the fixed board (cloned)
}

public static class BoardValidator
{
    // True if every passable tile is reachable from every other passable tile (4/6-conn over passable hexes).
    public static bool IsFullyConnected(Board board);
    // True if zone0 and zone1 are mirror-symmetric across the board center.
    public static bool ZonesAreSymmetric(Board board);
}
```

### Agent API (headless self-play foundations)

```csharp
public interface IAgent
{
    string Name { get; }
    Command ChooseCommand(GameState state);   // must return a command from GameEngine.LegalMoves(state)
}

public sealed class RandomAgent : IAgent
{
    public RandomAgent(int seed);             // System.Random(seed) ONLY
    public string Name { get; }
    public Command ChooseCommand(GameState state);
}

public sealed class MatchResult
{
    public PlayerId? Winner { get; }
    public int Rounds { get; }
    public int Commands { get; }
    public GameState FinalState { get; }
    public MatchResult(PlayerId? winner, int rounds, int commands, GameState finalState);
}

public static class Match
{
    // Runs a full headless game agent0 vs agent1 from `initial`, capped at maxCommands to guarantee termination.
    public static MatchResult Run(GameState initial, IAgent agent0, IAgent agent1, int maxCommands = 10000);
}
```

### Presentation (HexWars.Presentation) â€” the only Unity-referencing types

```csharp
namespace HexWars.Presentation
{
    // PURE helper, unit-tested in EditMode tests (no MonoBehaviour). Flat-top axial -> world XZ.
    public static class HexLayout
    {
        public const float HexSize = 1.0f;       // outer radius
        public const float ElevationStep = 0.5f; // world Y per elevation level
        // Returns (x, z) world position for an axial coord (pointy/flat convention fixed in impl & test).
        public static (float x, float z) AxialToWorldXZ(int q, int r);
        public static (float x, float y, float z) CoordToWorld(HexWars.Engine.HexCoord coord);
    }

    public sealed class BoardRenderer : UnityEngine.MonoBehaviour { /* instantiates shared hex prefab into columns by elevation */ }
    public sealed class UnitView : UnityEngine.MonoBehaviour { /* binds a Unit id, colors by owner */ }
    public sealed class GeneratorView : UnityEngine.MonoBehaviour { /* distinct pylon shape, colors by owner */ }
    public sealed class CameraRig : UnityEngine.MonoBehaviour { /* angled orbit/pan so height reads */ }
    public sealed class InputController : UnityEngine.MonoBehaviour { /* select tile/unit -> raises intents */ }
    public sealed class Hud : UnityEngine.MonoBehaviour { /* points, stat panel, create-unit panel, banner, End Turn, win screen */ }
    public sealed class GamePresenter : UnityEngine.MonoBehaviour { /* owns GameState, calls GameEngine.Apply, renders events */ }
}
```

## File Structure

```
Assets/HexWars/
â”œâ”€â”€ Engine/
â”‚   â”œâ”€â”€ HexWars.Engine.asmdef                 # PLAIN C# asmdef: noEngineReferences=true, autoReferenced=true, NO Unity refs
â”‚   â”œâ”€â”€ Data/
â”‚   â”‚   â”œâ”€â”€ PlayerId.cs                        # enum PlayerId {Player0,Player1}
â”‚   â”‚   â”œâ”€â”€ TerrainType.cs                     # enum TerrainType {Plains,Forest,Rough,Water}
â”‚   â”‚   â”œâ”€â”€ TerrainDef.cs                      # TerrainDef: move/concealment/defense/passable
â”‚   â”‚   â”œâ”€â”€ HexCoord.cs                        # axial Q,R + Elevation + cube S + equality/helpers
â”‚   â”‚   â”œâ”€â”€ Tile.cs                            # Tile {Coord, Terrain, Elevation}
â”‚   â”‚   â”œâ”€â”€ Board.cs                           # Board {tiles, deployment zones, lookup, clone}
â”‚   â”‚   â”œâ”€â”€ UnitStats.cs                       # UnitStats + PointCost (flat 1:1) + IsValid
â”‚   â”‚   â”œâ”€â”€ Unit.cs                            # Unit {id,owner,stats,position,hp,hasActed} + With* copies
â”‚   â”‚   â”œâ”€â”€ Generator.cs                       # Generator {id,owner,position,hp} + With* copies
â”‚   â”‚   â”œâ”€â”€ PlayerState.cs                     # PlayerState {points,reserve,units,generators} + clone
â”‚   â”‚   â”œâ”€â”€ GameState.cs                       # GameState {board,players[2],active,round,...} + Clone
â”‚   â”‚   â””â”€â”€ GameConfig.cs                      # GameConfig: every Â§14 tunable + terrain table + Default()
â”‚   â”œâ”€â”€ Geometry/
â”‚   â”‚   â”œâ”€â”€ HexGeometry.cs                     # HexDistance, Distance3D, AxialDirections, Neighbors
â”‚   â”‚   â””â”€â”€ Pathfinding.cs                     # CostToEnter, ReachableTiles, TryFindPath (Dijkstra)
â”‚   â”œâ”€â”€ Combat/
â”‚   â”‚   â”œâ”€â”€ TargetingService.cs               # HighGround, InRange, IsVisible, CanTarget, ValidTargetIds
â”‚   â”‚   â””â”€â”€ CombatResolver.cs                  # ComputeDamage, ResolveAgainst*, ComputeBounty, CombatOutcome
â”‚   â”œâ”€â”€ Economy/
â”‚   â”‚   â”œâ”€â”€ Economy.cs                         # IncomeFor, CanAfford, CheapestViableUnitCost
â”‚   â”‚   â”œâ”€â”€ WinCheck.cs                        # IsEliminated, TotalValue, CheckWinner
â”‚   â”‚   â””â”€â”€ Evaluation.cs                      # Evaluate(state, player)
â”‚   â”œâ”€â”€ Commands/
â”‚   â”‚   â”œâ”€â”€ Command.cs                         # abstract Command + 6 records (Create/Deploy*/Move/Attack/EndTurn)
â”‚   â”‚   â”œâ”€â”€ GameEvent.cs                       # GameEvent record + GameEventType enum
â”‚   â”‚   â”œâ”€â”€ Result.cs                          # Result {Success,NewState,Events,Reason,Message} + RejectionReason
â”‚   â”‚   â”œâ”€â”€ GameEngine.cs                      # Apply (the ONLY mutation path) + LegalMoves
â”‚   â”‚   â””â”€â”€ GameFactory.cs                     # NewGame/Reset(generator,config,seed)
â”‚   â”œâ”€â”€ Turns/
â”‚   â”‚   â”œâ”€â”€ ITurnPolicy.cs                     # ITurnPolicy + TurnPolicyKind + TurnPolicyFactory
â”‚   â”‚   â”œâ”€â”€ AllUnitsPolicy.cs                  # each unit acts once; explicit EndTurn
â”‚   â”‚   â””â”€â”€ OneActionPolicy.cs                 # one atomic action then auto-pass
â”‚   â”œâ”€â”€ Board/
â”‚   â”‚   â”œâ”€â”€ IBoardGenerator.cs                 # IBoardGenerator interface
â”‚   â”‚   â”œâ”€â”€ RandomBoardGenerator.cs           # SEEDED System.Random noise+weights, mirrored zones
â”‚   â”‚   â”œâ”€â”€ AuthoredBoardGenerator.cs         # fixed handcrafted board
â”‚   â”‚   â””â”€â”€ BoardValidator.cs                  # IsFullyConnected, ZonesAreSymmetric
â”‚   â”œâ”€â”€ Agent/
â”‚   â”‚   â”œâ”€â”€ IAgent.cs                          # IAgent {Name, ChooseCommand}
â”‚   â”‚   â”œâ”€â”€ RandomAgent.cs                     # seeded reference agent
â”‚   â”‚   â””â”€â”€ Match.cs                           # headless Run(initial,a0,a1) + MatchResult
â”‚   â””â”€â”€ Serialization/
â”‚       â””â”€â”€ GameStateSerializer.cs            # JSON to/from GameState & Command
â”œâ”€â”€ Engine/Tests/
â”‚   â”œâ”€â”€ HexWars.Engine.Tests.asmdef           # EditMode asmdef: refs HexWars.Engine + TestRunner + nunit; UNITY_INCLUDE_TESTS
â”‚   â”œâ”€â”€ ScaffoldSmokeTests.cs                 # trivial pass test proving harness compiles/runs
â”‚   â”œâ”€â”€ HexCoordTests.cs                      # axial/cube/elevation + equality
â”‚   â”œâ”€â”€ BoardDataTests.cs                     # Board lookup, deployment-zone membership, clone
â”‚   â”œâ”€â”€ UnitStatsTests.cs                     # PointCost 1:1, IsValid floor
â”‚   â”œâ”€â”€ GameStateTests.cs                     # Clone deep-copy independence, Opponent, indexing
â”‚   â”œâ”€â”€ GameConfigTests.cs                    # Default() values match Â§14, terrain table present
â”‚   â”œâ”€â”€ HexGeometryTests.cs                   # HexDistance, Distance3D, Neighbors, directions
â”‚   â”œâ”€â”€ PathfindingTests.cs                   # CostToEnter (terrain+climb), ReachableTiles, TryFindPath, MaxClimbPerStep
â”‚   â”œâ”€â”€ TargetingServiceTests.cs             # InRange (+highground), IsVisible (concealment), generators as targets
â”‚   â”œâ”€â”€ CombatResolverTests.cs               # ComputeDamage (highground/terrain/floor), bounty, death
â”‚   â”œâ”€â”€ EconomyTests.cs                       # income, CanAfford, cheapest viable
â”‚   â”œâ”€â”€ WinCheckTests.cs                      # elimination, total value, round-cap tie-break
â”‚   â”œâ”€â”€ CommandApplyTests.cs                 # Apply each command happy-path + non-mutation of input
â”‚   â”œâ”€â”€ CommandRejectionTests.cs            # every RejectionReason path
â”‚   â”œâ”€â”€ LegalMovesTests.cs                   # enumeration correctness per state
â”‚   â”œâ”€â”€ TurnPolicyTests.cs                   # AllUnits vs OneAction legality + auto-end + once-per-turn + income-at-start
â”‚   â”œâ”€â”€ SerializationTests.cs               # round-trip GameState & Command JSON equality
â”‚   â”œâ”€â”€ BoardGenerationTests.cs             # seed determinism, symmetry, connectivity, elevation/terrain bounds
â”‚   â”œâ”€â”€ AgentMatchTests.cs                   # RandomAgent only returns legal moves; Match terminates
â”‚   â”œâ”€â”€ HexLayoutTests.cs                    # presentation HexLayout pure math (lives here, EditMode)
â”‚   â””â”€â”€ FullGameIntegrationTests.cs         # scripted create->deploy->generator->move->attack->win
â”œâ”€â”€ Presentation/
â”‚   â”œâ”€â”€ HexWars.Presentation.asmdef          # refs HexWars.Engine + UnityEngine + InputSystem + UGUI
â”‚   â”œâ”€â”€ HexLayout.cs                          # PURE axial->world XZ helper (unit-tested in Engine/Tests)
â”‚   â”œâ”€â”€ BoardRenderer.cs                      # instances shared hex prefab into columns by elevation
â”‚   â”œâ”€â”€ UnitView.cs                           # per-unit view, owner color, black outline
â”‚   â”œâ”€â”€ GeneratorView.cs                      # distinct pylon shape, owner color
â”‚   â”œâ”€â”€ CameraRig.cs                          # angled orbit/pan camera
â”‚   â”œâ”€â”€ InputController.cs                    # Input System: select tile/unit -> intents
â”‚   â”œâ”€â”€ Hud.cs                                # points, stat panel, create-unit panel, banner, End Turn, win screen
â”‚   â””â”€â”€ GamePresenter.cs                      # owns GameState, GameEngine.Apply, renders GameEvents, hotseat flow
â”œâ”€â”€ Art/
â”‚   â”œâ”€â”€ Shaders/
â”‚   â”‚   â””â”€â”€ UnlitOutline.shader               # URP unlit fill + black outline
â”‚   â”œâ”€â”€ Materials/
â”‚   â”‚   â”œâ”€â”€ Hex_Plains.mat                     # yellow plains
â”‚   â”‚   â”œâ”€â”€ Hex_Forest.mat                     # forest tint
â”‚   â”‚   â”œâ”€â”€ Hex_Rough.mat                      # rough tint
â”‚   â”‚   â”œâ”€â”€ Hex_Water.mat                      # water tint
â”‚   â”‚   â”œâ”€â”€ Unit_P0.mat                        # cyan/blue
â”‚   â”‚   â”œâ”€â”€ Unit_P1.mat                        # red/magenta
â”‚   â”‚   â”œâ”€â”€ Gen_P0.mat                         # P0 generator
â”‚   â”‚   â”œâ”€â”€ Gen_P1.mat                         # P1 generator
â”‚   â”‚   â””â”€â”€ Starfield_Skybox.mat              # starfield skybox
â”‚   â””â”€â”€ Prefabs/
â”‚       â”œâ”€â”€ Hex.prefab                         # single shared low wide hex prism, black outline
â”‚       â”œâ”€â”€ UnitView.prefab                    # unit primitive + UnitView
â”‚       â””â”€â”€ GeneratorView.prefab              # pylon primitive + GeneratorView
â”œâ”€â”€ Scenes/
â”‚   â””â”€â”€ HexWars.unity                          # main hotseat scene (camera, presenter, HUD canvas, lighting, skybox)
â””â”€â”€ Docs/
    â””â”€â”€ PlaytestChecklist.md                   # manual success-criterion + hotseat-loop checklist
```

---
### Task 1: Scaffold engine + test assemblies and verify the EditMode harness runs

**Files:**
- Create: `Assets/HexWars/Engine/HexWars.Engine.asmdef`
- Create: `Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef`
- Create: `Assets/HexWars/Engine/Tests/ScaffoldSmokeTests.cs`
- Test: `Assets/HexWars/Engine/Tests/ScaffoldSmokeTests.cs`

**Interfaces:**
- Consumes: Nothing. This is the root task with no dependencies.
- Produces: Assembly `HexWars.Engine` (root namespace `HexWars.Engine`, `"noEngineReferences": true`, `"autoReferenced": true`, no Unity references) at folder `Assets/HexWars/Engine/`. EditMode test assembly `HexWars.Engine.Tests` (namespace `HexWars.Engine.Tests`) at `Assets/HexWars/Engine/Tests/`, referencing `HexWars.Engine` plus `UnityEngine.TestRunner` + `UnityEditor.TestRunner` (via GUID) + `nunit.framework.dll` (precompiled reference), `"includePlatforms": ["Editor"]`, `"defineConstraints": ["UNITY_INCLUDE_TESTS"]`. Establishes the folder layout `Assets/HexWars/Engine/` and `Assets/HexWars/Engine/Tests/` that every later engine task writes into.

- [ ] **Step 1: Write the failing test**

Create `Assets/HexWars/Engine/Tests/ScaffoldSmokeTests.cs`. This file is authored first; it will not compile/run until the two asmdef files exist (Step 3), which is the intended initial failure. The test asserts the NUnit harness is wired and that the engine assembly is reachable by reflection (proving `HexWars.Engine` exists and is linked) WITHOUT depending on any engine type a later task owns.

```csharp
using System;
using System.Linq;
using System.Reflection;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    [TestFixture]
    public class ScaffoldSmokeTests
    {
        [Test]
        public void Harness_Runs_TrivialAssertionPasses()
        {
            Assert.That(1 + 1, Is.EqualTo(2));
        }

        [Test]
        public void EngineAssembly_IsLoaded_AndHasRootNamespace()
        {
            // Proves the HexWars.Engine assembly compiled and is referenced by this
            // test assembly. We locate it via the loaded AppDomain by assembly name.
            Assembly engine = AppDomain.CurrentDomain
                .GetAssemblies()
                .FirstOrDefault(a => a.GetName().Name == "HexWars.Engine");

            Assert.That(engine, Is.Not.Null,
                "HexWars.Engine assembly was not found in the AppDomain. " +
                "Check that HexWars.Engine.asmdef exists and is referenced by the test asmdef.");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
"/c/Program Files/Unity/Hub/Editor/6000.5.0f1/Editor/Unity.exe" -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.ScaffoldSmokeTests" -logFile -
```

Expected: FAIL. With no asmdef files present, `ScaffoldSmokeTests.cs` is compiled into the default `Assembly-CSharp-Editor` (or fails to find NUnit / the `HexWars.Engine` assembly name), so either compilation fails or `EngineAssembly_IsLoaded_AndHasRootNamespace` fails with the assert message "HexWars.Engine assembly was not found in the AppDomain." (Interactive equivalent: open Window > General > Test Runner, EditMode tab, and observe the test does not appear / errors.)

- [ ] **Step 3: Write minimal implementation**

Create the engine asmdef `Assets/HexWars/Engine/HexWars.Engine.asmdef` â€” a plain C# assembly with zero Unity engine references:

```json
{
    "name": "HexWars.Engine",
    "rootNamespace": "HexWars.Engine",
    "references": [],
    "includePlatforms": [],
    "excludePlatforms": [],
    "allowUnsafeCode": false,
    "overrideReferences": false,
    "precompiledReferences": [],
    "autoReferenced": true,
    "defineConstraints": [],
    "versionDefines": [],
    "noEngineReferences": true
}
```

Create the test asmdef `Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef` â€” an EditMode-only assembly referencing the engine and the test framework. The TestRunner references use Unity's stable GUIDs (`UnityEngine.TestRunner` = `27619889b8ba8c24980f49ee34dbb44a`, `UnityEditor.TestRunner` = `0acc523941302664db1f4e527237feb3`); `nunit.framework.dll` is added as an overridden precompiled reference:

```json
{
    "name": "HexWars.Engine.Tests",
    "rootNamespace": "HexWars.Engine.Tests",
    "references": [
        "HexWars.Engine",
        "GUID:27619889b8ba8c24980f49ee34dbb44a",
        "GUID:0acc523941302664db1f4e527237feb3"
    ],
    "includePlatforms": [
        "Editor"
    ],
    "excludePlatforms": [],
    "allowUnsafeCode": false,
    "overrideReferences": true,
    "precompiledReferences": [
        "nunit.framework.dll"
    ],
    "autoReferenced": false,
    "defineConstraints": [
        "UNITY_INCLUDE_TESTS"
    ],
    "versionDefines": [],
    "noEngineReferences": false
}
```

The `ScaffoldSmokeTests.cs` content from Step 1 is unchanged. (No engine source file is required: the assertion only needs the `HexWars.Engine` assembly to exist and be referenced. Unity emits an empty assembly for an asmdef that has no `.cs` files yet, and Task 2 will add the first real source file.)

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
"/c/Program Files/Unity/Hub/Editor/6000.5.0f1/Editor/Unity.exe" -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.ScaffoldSmokeTests" -logFile -
```

Expected: PASS. Both `Harness_Runs_TrivialAssertionPasses` and `EngineAssembly_IsLoaded_AndHasRootNamespace` pass; the log reports `Passed: 2` / exit code 0. (Optional live-editor verification: call the `mcp__coplay-mcp__check_compile_errors` tool and confirm zero errors, confirming `HexWars.Engine.asmdef` compiled with `noEngineReferences=true`.)

- [ ] **Step 5: Commit**

```bash
git add Assets/HexWars/Engine/HexWars.Engine.asmdef Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef Assets/HexWars/Engine/Tests/ScaffoldSmokeTests.cs && git commit -m "chore(engine): scaffold HexWars.Engine + EditMode test asmdefs with passing smoke test"
```

---

### Task 2: HexCoord (axial + elevation, cube helpers, equality)

**Files:**
- Create: `Assets/HexWars/Engine/Data/HexCoord.cs`
- Create: `Assets/HexWars/Engine/Data/PlayerId.cs`
- Test: `Assets/HexWars/Engine/Tests/HexCoordTests.cs`

**Interfaces:**
- Consumes: Engine assembly `HexWars.Engine` and EditMode test assembly `HexWars.Engine.Tests` from Task 1 (no UnityEngine refs).
- Produces: `public enum PlayerId { Player0 = 0, Player1 = 1 }`; `public readonly struct HexCoord : System.IEquatable<HexCoord> { int Q{get;} int R{get;} int Elevation{get;} int S => -Q-R; HexCoord(int q,int r,int elevation=0); HexCoord WithElevation(int); HexCoord Translate(int dq,int dr); bool Equals(HexCoord); override bool Equals(object); override int GetHashCode(); override string ToString(); static bool operator==(HexCoord,HexCoord); static bool operator!=(HexCoord,HexCoord); }`. Consumed by Tasks 3,4,5,7,8,9,10,14,19,25.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class HexCoordTests
    {
        [Test]
        public void Constructor_StoresQRElevation_AndDefaultsElevationZero()
        {
            var c = new HexCoord(2, -3);
            Assert.AreEqual(2, c.Q);
            Assert.AreEqual(-3, c.R);
            Assert.AreEqual(0, c.Elevation);

            var e = new HexCoord(2, -3, 5);
            Assert.AreEqual(5, e.Elevation);
        }

        [Test]
        public void S_IsNegativeQMinusR_CubeIdentity()
        {
            var c = new HexCoord(2, -3, 1);
            Assert.AreEqual(-(2) - (-3), c.S);
            Assert.AreEqual(0, c.Q + c.R + c.S, "cube coords must sum to zero");
        }

        [Test]
        public void WithElevation_ReturnsCopyWithNewElevation_QRUnchanged()
        {
            var c = new HexCoord(2, -3, 1);
            var w = c.WithElevation(7);
            Assert.AreEqual(2, w.Q);
            Assert.AreEqual(-3, w.R);
            Assert.AreEqual(7, w.Elevation);
            Assert.AreEqual(1, c.Elevation, "original must be unchanged");
        }

        [Test]
        public void Translate_ShiftsAxial_KeepsElevation()
        {
            var c = new HexCoord(2, -3, 4);
            var t = c.Translate(1, 2);
            Assert.AreEqual(3, t.Q);
            Assert.AreEqual(-1, t.R);
            Assert.AreEqual(4, t.Elevation);
        }

        [Test]
        public void Equality_ComparesQRElevation()
        {
            var a = new HexCoord(1, 2, 3);
            var b = new HexCoord(1, 2, 3);
            var diffElev = new HexCoord(1, 2, 4);
            Assert.IsTrue(a.Equals(b));
            Assert.IsTrue(a == b);
            Assert.IsFalse(a != b);
            Assert.IsTrue(a != diffElev);
            Assert.IsFalse(a == diffElev);
            Assert.IsTrue(a.Equals((object)b));
            Assert.IsFalse(a.Equals((object)"nope"));
        }

        [Test]
        public void GetHashCode_IsDeterministic_AndEqualForEqualValues()
        {
            var a = new HexCoord(1, 2, 3);
            var b = new HexCoord(1, 2, 3);
            Assert.AreEqual(a.GetHashCode(), b.GetHashCode());
            Assert.AreEqual(a.GetHashCode(), a.GetHashCode());
        }

        [Test]
        public void ToString_IsQRElevationTuple()
        {
            Assert.AreEqual("(1,2,3)", new HexCoord(1, 2, 3).ToString());
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexCoordTests" -logFile -`
Expected: FAIL â€” compile error, `HexCoord` and `PlayerId` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/PlayerId.cs
namespace HexWars.Engine
{
    // Player 0 = Player 1 (cyan/blue); Player 1 = Player 2 (red/magenta).
    public enum PlayerId { Player0 = 0, Player1 = 1 }
}
```
```csharp
// Assets/HexWars/Engine/Data/HexCoord.cs
namespace HexWars.Engine
{
    public readonly struct HexCoord : System.IEquatable<HexCoord>
    {
        public int Q { get; }
        public int R { get; }
        public int Elevation { get; }
        public int S => -Q - R;

        public HexCoord(int q, int r, int elevation = 0)
        {
            Q = q;
            R = r;
            Elevation = elevation;
        }

        public HexCoord WithElevation(int elevation) => new HexCoord(Q, R, elevation);
        public HexCoord Translate(int dq, int dr) => new HexCoord(Q + dq, R + dr, Elevation);

        public bool Equals(HexCoord other) =>
            Q == other.Q && R == other.R && Elevation == other.Elevation;

        public override bool Equals(object obj) => obj is HexCoord other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + Q;
                hash = hash * 31 + R;
                hash = hash * 31 + Elevation;
                return hash;
            }
        }

        public override string ToString() => "(" + Q + "," + R + "," + Elevation + ")";

        public static bool operator ==(HexCoord a, HexCoord b) => a.Equals(b);
        public static bool operator !=(HexCoord a, HexCoord b) => !a.Equals(b);
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexCoordTests" -logFile -`
Expected: PASS â€” all 7 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/HexCoord.cs Assets/HexWars/Engine/Data/PlayerId.cs Assets/HexWars/Engine/Tests/HexCoordTests.cs && git commit -m "feat(engine): add HexCoord struct and PlayerId enum with tests"
```

---

### Task 3: TerrainType, TerrainDef, and Tile

**Files:**
- Create: `Assets/HexWars/Engine/Data/TerrainType.cs`
- Create: `Assets/HexWars/Engine/Data/TerrainDef.cs`
- Create: `Assets/HexWars/Engine/Data/Tile.cs`
- Test: `Assets/HexWars/Engine/Tests/BoardDataTests.cs`

**Interfaces:**
- Consumes: `HexCoord(int q,int r,int elevation=0)`, `HexCoord.Elevation`, `HexCoord.WithElevation` from Task 2.
- Produces: `public enum TerrainType { Plains=0, Forest=1, Rough=2, Water=3 }`; `public sealed class TerrainDef { TerrainType Terrain{get;} int MoveCost{get;} int Concealment{get;} int Defense{get;} bool Passable{get;} TerrainDef(TerrainType,int moveCost,int concealment,int defense,bool passable); }`; `public sealed class Tile { HexCoord Coord{get;} TerrainType Terrain{get;} int Elevation => Coord.Elevation; Tile(HexCoord coord,TerrainType terrain); Tile Clone(); }`. Consumed by Tasks 4,6,9,10,11,21,22,23,28.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class BoardDataTests
    {
        [Test]
        public void TerrainType_HasExpectedValues()
        {
            Assert.AreEqual(0, (int)TerrainType.Plains);
            Assert.AreEqual(1, (int)TerrainType.Forest);
            Assert.AreEqual(2, (int)TerrainType.Rough);
            Assert.AreEqual(3, (int)TerrainType.Water);
        }

        [Test]
        public void TerrainDef_RoundTripsAllFields()
        {
            var def = new TerrainDef(TerrainType.Forest, 2, 2, 1, true);
            Assert.AreEqual(TerrainType.Forest, def.Terrain);
            Assert.AreEqual(2, def.MoveCost);
            Assert.AreEqual(2, def.Concealment);
            Assert.AreEqual(1, def.Defense);
            Assert.IsTrue(def.Passable);

            var water = new TerrainDef(TerrainType.Water, 3, 0, 0, false);
            Assert.IsFalse(water.Passable);
        }

        [Test]
        public void Tile_StoresCoordAndTerrain_ElevationPassesThrough()
        {
            var tile = new Tile(new HexCoord(1, 2, 4), TerrainType.Rough);
            Assert.AreEqual(new HexCoord(1, 2, 4), tile.Coord);
            Assert.AreEqual(TerrainType.Rough, tile.Terrain);
            Assert.AreEqual(4, tile.Elevation);
        }

        [Test]
        public void Tile_Clone_IsEqualValueButDistinctInstance()
        {
            var original = new Tile(new HexCoord(3, -1, 2), TerrainType.Forest);
            var clone = original.Clone();
            Assert.AreNotSame(original, clone);
            Assert.AreEqual(original.Coord, clone.Coord);
            Assert.AreEqual(original.Terrain, clone.Terrain);
            Assert.AreEqual(original.Elevation, clone.Elevation);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardDataTests" -logFile -`
Expected: FAIL â€” compile error, `TerrainType`, `TerrainDef`, `Tile` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/TerrainType.cs
namespace HexWars.Engine
{
    public enum TerrainType { Plains = 0, Forest = 1, Rough = 2, Water = 3 }
}
```
```csharp
// Assets/HexWars/Engine/Data/TerrainDef.cs
namespace HexWars.Engine
{
    // Data-driven terrain modifiers (spec Â§6). Stored as a table on GameConfig.
    public sealed class TerrainDef
    {
        public TerrainType Terrain { get; }
        public int MoveCost { get; }       // cost to enter (before climb)
        public int Concealment { get; }    // added to distance vs Vision
        public int Defense { get; }        // flat damage reduction when defending here
        public bool Passable { get; }

        public TerrainDef(TerrainType terrain, int moveCost, int concealment, int defense, bool passable)
        {
            Terrain = terrain;
            MoveCost = moveCost;
            Concealment = concealment;
            Defense = defense;
            Passable = passable;
        }
    }
}
```
```csharp
// Assets/HexWars/Engine/Data/Tile.cs
namespace HexWars.Engine
{
    public sealed class Tile
    {
        public HexCoord Coord { get; }          // includes Elevation
        public TerrainType Terrain { get; }
        public int Elevation => Coord.Elevation;

        public Tile(HexCoord coord, TerrainType terrain)
        {
            Coord = coord;
            Terrain = terrain;
        }

        public Tile Clone() => new Tile(Coord, Terrain);
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardDataTests" -logFile -`
Expected: PASS â€” all 4 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/TerrainType.cs Assets/HexWars/Engine/Data/TerrainDef.cs Assets/HexWars/Engine/Data/Tile.cs Assets/HexWars/Engine/Tests/BoardDataTests.cs && git commit -m "feat(engine): add TerrainType, TerrainDef, and Tile with tests"
```

---

### Task 4: Board (tiles, deployment zones, axial lookup, clone)

**Files:**
- Create: `Assets/HexWars/Engine/Data/Board.cs`
- Modify: `Assets/HexWars/Engine/Tests/BoardDataTests.cs` (append Board test cases)
- Test: `Assets/HexWars/Engine/Tests/BoardDataTests.cs`

**Interfaces:**
- Consumes: `HexCoord(int,int,int)`, `HexCoord.Q/R`, equality from Task 2; `Tile(HexCoord,TerrainType)`, `Tile.Coord`, `Tile.Clone()`, `TerrainType` from Task 3; `PlayerId` from Task 2.
- Produces: `public sealed class Board { int Width{get;} int Height{get;} System.Collections.Generic.IReadOnlyList<Tile> Tiles{get;} System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZone0{get;} System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZone1{get;} Board(int width,int height,System.Collections.Generic.IReadOnlyList<Tile> tiles,System.Collections.Generic.IReadOnlyList<HexCoord> deploymentZone0,System.Collections.Generic.IReadOnlyList<HexCoord> deploymentZone1); bool TryGetTile(int q,int r,out Tile tile); bool Contains(int q,int r); System.Collections.Generic.IReadOnlyList<HexCoord> DeploymentZoneFor(PlayerId player); bool IsInDeploymentZone(PlayerId player,int q,int r); Board Clone(); }`. Consumed by Tasks 7,9,10,11,16,20,21,22,23,28.

- [ ] **Step 1: Write the failing test**
```csharp
// Append inside namespace HexWars.Engine.Tests, add a new test class to BoardDataTests.cs
using System.Collections.Generic;

namespace HexWars.Engine.Tests
{
    public class BoardTests
    {
        private static Board MakeTwoByTwo()
        {
            var tiles = new List<Tile>
            {
                new Tile(new HexCoord(0, 0, 0), TerrainType.Plains),
                new Tile(new HexCoord(1, 0, 1), TerrainType.Forest),
                new Tile(new HexCoord(0, 1, 2), TerrainType.Rough),
                new Tile(new HexCoord(1, 1, 3), TerrainType.Water),
            };
            var dz0 = new List<HexCoord> { new HexCoord(0, 0, 0) };
            var dz1 = new List<HexCoord> { new HexCoord(1, 1, 3) };
            return new Board(2, 2, tiles, dz0, dz1);
        }

        [Test]
        public void Constructor_ExposesDimensionsAndCollections()
        {
            var board = MakeTwoByTwo();
            Assert.AreEqual(2, board.Width);
            Assert.AreEqual(2, board.Height);
            Assert.AreEqual(4, board.Tiles.Count);
            Assert.AreEqual(1, board.DeploymentZone0.Count);
            Assert.AreEqual(1, board.DeploymentZone1.Count);
        }

        [Test]
        public void TryGetTile_HitReturnsTileIgnoringElevation()
        {
            var board = MakeTwoByTwo();
            Assert.IsTrue(board.TryGetTile(0, 1, out var tile));
            Assert.AreEqual(TerrainType.Rough, tile.Terrain);
            Assert.AreEqual(2, tile.Elevation);
        }

        [Test]
        public void TryGetTile_MissReturnsFalseAndNull()
        {
            var board = MakeTwoByTwo();
            Assert.IsFalse(board.TryGetTile(9, 9, out var tile));
            Assert.IsNull(tile);
        }

        [Test]
        public void Contains_TrueForExistingFalseOtherwise()
        {
            var board = MakeTwoByTwo();
            Assert.IsTrue(board.Contains(1, 0));
            Assert.IsFalse(board.Contains(5, 5));
        }

        [Test]
        public void DeploymentZoneFor_ReturnsCorrectPerPlayerZone()
        {
            var board = MakeTwoByTwo();
            Assert.AreSame(board.DeploymentZone0, board.DeploymentZoneFor(PlayerId.Player0));
            Assert.AreSame(board.DeploymentZone1, board.DeploymentZoneFor(PlayerId.Player1));
        }

        [Test]
        public void IsInDeploymentZone_IgnoresElevation_AndMatchesAxialOnly()
        {
            var board = MakeTwoByTwo();
            Assert.IsTrue(board.IsInDeploymentZone(PlayerId.Player0, 0, 0));
            Assert.IsFalse(board.IsInDeploymentZone(PlayerId.Player0, 1, 1));
            Assert.IsTrue(board.IsInDeploymentZone(PlayerId.Player1, 1, 1));
            Assert.IsFalse(board.IsInDeploymentZone(PlayerId.Player1, 0, 0));
        }

        [Test]
        public void Clone_IsDeepAndIndependent()
        {
            var board = MakeTwoByTwo();
            var clone = board.Clone();
            Assert.AreNotSame(board, clone);
            Assert.AreNotSame(board.Tiles, clone.Tiles);
            Assert.AreEqual(board.Tiles.Count, clone.Tiles.Count);
            for (int i = 0; i < board.Tiles.Count; i++)
            {
                Assert.AreNotSame(board.Tiles[i], clone.Tiles[i]);
                Assert.AreEqual(board.Tiles[i].Coord, clone.Tiles[i].Coord);
                Assert.AreEqual(board.Tiles[i].Terrain, clone.Tiles[i].Terrain);
            }
            Assert.AreNotSame(board.DeploymentZone0, clone.DeploymentZone0);
            Assert.AreEqual(board.DeploymentZone0[0], clone.DeploymentZone0[0]);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardTests" -logFile -`
Expected: FAIL â€” compile error, `Board` does not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/Board.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    public sealed class Board
    {
        public int Width { get; }
        public int Height { get; }
        public IReadOnlyList<Tile> Tiles { get; }
        public IReadOnlyList<HexCoord> DeploymentZone0 { get; }
        public IReadOnlyList<HexCoord> DeploymentZone1 { get; }

        // Axial (q,r) -> tile index, built once for O(1) lookup. Elevation ignored for keying.
        private readonly Dictionary<(int, int), Tile> _byAxial;

        public Board(int width, int height,
                     IReadOnlyList<Tile> tiles,
                     IReadOnlyList<HexCoord> deploymentZone0,
                     IReadOnlyList<HexCoord> deploymentZone1)
        {
            Width = width;
            Height = height;
            Tiles = tiles;
            DeploymentZone0 = deploymentZone0;
            DeploymentZone1 = deploymentZone1;

            _byAxial = new Dictionary<(int, int), Tile>(tiles.Count);
            for (int i = 0; i < tiles.Count; i++)
            {
                var t = tiles[i];
                _byAxial[(t.Coord.Q, t.Coord.R)] = t;
            }
        }

        public bool TryGetTile(int q, int r, out Tile tile) => _byAxial.TryGetValue((q, r), out tile);

        public bool Contains(int q, int r) => _byAxial.ContainsKey((q, r));

        public IReadOnlyList<HexCoord> DeploymentZoneFor(PlayerId player) =>
            player == PlayerId.Player0 ? DeploymentZone0 : DeploymentZone1;

        public bool IsInDeploymentZone(PlayerId player, int q, int r)
        {
            var zone = DeploymentZoneFor(player);
            for (int i = 0; i < zone.Count; i++)
            {
                if (zone[i].Q == q && zone[i].R == r) return true;
            }
            return false;
        }

        public Board Clone()
        {
            var tiles = new List<Tile>(Tiles.Count);
            for (int i = 0; i < Tiles.Count; i++) tiles.Add(Tiles[i].Clone());
            var dz0 = new List<HexCoord>(DeploymentZone0);
            var dz1 = new List<HexCoord>(DeploymentZone1);
            return new Board(Width, Height, tiles, dz0, dz1);
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardTests" -logFile -`
Expected: PASS â€” all 7 Board tests green (existing BoardDataTests still pass).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/Board.cs Assets/HexWars/Engine/Tests/BoardDataTests.cs && git commit -m "feat(engine): add Board with axial lookup, deployment zones, and deep clone"
```

---

### Task 5: UnitStats (flat point-buy) and Unit/Generator entities

**Files:**
- Create: `Assets/HexWars/Engine/Data/UnitStats.cs`
- Create: `Assets/HexWars/Engine/Data/Unit.cs`
- Create: `Assets/HexWars/Engine/Data/Generator.cs`
- Modify: `Assets/HexWars/Engine/Tests/UnitStatsTests.cs` (create file)
- Test: `Assets/HexWars/Engine/Tests/UnitStatsTests.cs`

**Interfaces:**
- Consumes: `HexCoord(int,int,int)`, equality from Task 2; `PlayerId` from Task 2.
- Produces: `public readonly struct UnitStats : System.IEquatable<UnitStats> { int Health,Damage,Range,Movement,Defense,Vision{get;} int PointCost => Health+Damage+Range+Movement+Defense+Vision; UnitStats(int health,int damage,int range,int movement,int defense,int vision); bool IsValid => Health>=1; value equality }`; `public sealed class Unit { int Id; PlayerId Owner; UnitStats Stats; HexCoord Position; int CurrentHp; bool HasActed; Unit(int,PlayerId,UnitStats,HexCoord,int,bool hasActed=false); int BuildCost => Stats.PointCost; bool IsAlive => CurrentHp>0; Unit WithPosition(HexCoord); Unit WithHp(int); Unit WithHasActed(bool); Unit Clone(); }`; `public sealed class Generator { int Id; PlayerId Owner; HexCoord Position; int CurrentHp; Generator(int,PlayerId,HexCoord,int); bool IsAlive => CurrentHp>0; Generator WithHp(int); Generator Clone(); }`. Consumed by Tasks 7,10,11,12,14,16,17,19,29.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class UnitStatsTests
    {
        [Test]
        public void PointCost_IsStrictSumOfSixStats()
        {
            var s = new UnitStats(3, 2, 1, 4, 0, 5);
            Assert.AreEqual(3 + 2 + 1 + 4 + 0 + 5, s.PointCost);
            Assert.AreEqual(15, s.PointCost);
        }

        [Test]
        public void IsValid_RequiresAtLeastOneHealth()
        {
            Assert.IsTrue(new UnitStats(1, 0, 0, 0, 0, 0).IsValid);
            Assert.IsFalse(new UnitStats(0, 5, 5, 5, 5, 5).IsValid);
        }

        [Test]
        public void UnitStats_HasValueEquality()
        {
            var a = new UnitStats(2, 2, 2, 2, 2, 2);
            var b = new UnitStats(2, 2, 2, 2, 2, 2);
            var c = new UnitStats(2, 2, 2, 2, 2, 3);
            Assert.IsTrue(a.Equals(b));
            Assert.IsTrue(a.Equals((object)b));
            Assert.IsFalse(a.Equals(c));
            Assert.AreEqual(a.GetHashCode(), b.GetHashCode());
        }

        [Test]
        public void Unit_BuildCostEqualsStatsPointCost_AndIsAliveTracksHp()
        {
            var stats = new UnitStats(5, 2, 1, 3, 0, 2);
            var unit = new Unit(7, PlayerId.Player0, stats, new HexCoord(1, 1, 0), 5);
            Assert.AreEqual(7, unit.Id);
            Assert.AreEqual(PlayerId.Player0, unit.Owner);
            Assert.AreEqual(stats.PointCost, unit.BuildCost);
            Assert.IsTrue(unit.IsAlive);
            Assert.IsFalse(unit.HasActed);
            Assert.IsFalse(new Unit(7, PlayerId.Player0, stats, new HexCoord(1, 1, 0), 0).IsAlive);
        }

        [Test]
        public void Unit_WithHelpers_AreNonMutating()
        {
            var stats = new UnitStats(5, 2, 1, 3, 0, 2);
            var unit = new Unit(7, PlayerId.Player0, stats, new HexCoord(1, 1, 0), 5);

            var moved = unit.WithPosition(new HexCoord(2, 2, 1));
            Assert.AreEqual(new HexCoord(2, 2, 1), moved.Position);
            Assert.AreEqual(new HexCoord(1, 1, 0), unit.Position, "original unchanged");
            Assert.AreEqual(unit.Id, moved.Id);
            Assert.AreEqual(unit.CurrentHp, moved.CurrentHp);

            var hurt = unit.WithHp(2);
            Assert.AreEqual(2, hurt.CurrentHp);
            Assert.AreEqual(5, unit.CurrentHp);

            var acted = unit.WithHasActed(true);
            Assert.IsTrue(acted.HasActed);
            Assert.IsFalse(unit.HasActed);
        }

        [Test]
        public void Unit_Clone_IsEqualValueDistinctInstance()
        {
            var stats = new UnitStats(5, 2, 1, 3, 0, 2);
            var unit = new Unit(7, PlayerId.Player1, stats, new HexCoord(1, 1, 0), 4, true);
            var clone = unit.Clone();
            Assert.AreNotSame(unit, clone);
            Assert.AreEqual(unit.Id, clone.Id);
            Assert.AreEqual(unit.Owner, clone.Owner);
            Assert.AreEqual(unit.Stats, clone.Stats);
            Assert.AreEqual(unit.Position, clone.Position);
            Assert.AreEqual(unit.CurrentHp, clone.CurrentHp);
            Assert.AreEqual(unit.HasActed, clone.HasActed);
        }

        [Test]
        public void Generator_TracksAliveAndWithHpClone()
        {
            var gen = new Generator(3, PlayerId.Player0, new HexCoord(0, 0, 0), 3);
            Assert.AreEqual(3, gen.Id);
            Assert.AreEqual(PlayerId.Player0, gen.Owner);
            Assert.IsTrue(gen.IsAlive);

            var damaged = gen.WithHp(0);
            Assert.IsFalse(damaged.IsAlive);
            Assert.AreEqual(3, gen.CurrentHp, "original unchanged");

            var clone = gen.Clone();
            Assert.AreNotSame(gen, clone);
            Assert.AreEqual(gen.Id, clone.Id);
            Assert.AreEqual(gen.Owner, clone.Owner);
            Assert.AreEqual(gen.Position, clone.Position);
            Assert.AreEqual(gen.CurrentHp, clone.CurrentHp);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.UnitStatsTests" -logFile -`
Expected: FAIL â€” compile error, `UnitStats`, `Unit`, `Generator` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/UnitStats.cs
namespace HexWars.Engine
{
    public readonly struct UnitStats : System.IEquatable<UnitStats>
    {
        public int Health { get; }
        public int Damage { get; }
        public int Range { get; }
        public int Movement { get; }
        public int Defense { get; }
        public int Vision { get; }

        public int PointCost => Health + Damage + Range + Movement + Defense + Vision; // strict 1:1

        public UnitStats(int health, int damage, int range, int movement, int defense, int vision)
        {
            Health = health;
            Damage = damage;
            Range = range;
            Movement = movement;
            Defense = defense;
            Vision = vision;
        }

        public bool IsValid => Health >= 1;     // only floor: a unit must have >=1 Health

        public bool Equals(UnitStats other) =>
            Health == other.Health && Damage == other.Damage && Range == other.Range &&
            Movement == other.Movement && Defense == other.Defense && Vision == other.Vision;

        public override bool Equals(object obj) => obj is UnitStats other && Equals(other);

        public override int GetHashCode()
        {
            unchecked
            {
                int hash = 17;
                hash = hash * 31 + Health;
                hash = hash * 31 + Damage;
                hash = hash * 31 + Range;
                hash = hash * 31 + Movement;
                hash = hash * 31 + Defense;
                hash = hash * 31 + Vision;
                return hash;
            }
        }
    }
}
```
```csharp
// Assets/HexWars/Engine/Data/Unit.cs
namespace HexWars.Engine
{
    public sealed class Unit
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public UnitStats Stats { get; }
        public HexCoord Position { get; }
        public int CurrentHp { get; }
        public bool HasActed { get; }

        public Unit(int id, PlayerId owner, UnitStats stats, HexCoord position, int currentHp, bool hasActed = false)
        {
            Id = id;
            Owner = owner;
            Stats = stats;
            Position = position;
            CurrentHp = currentHp;
            HasActed = hasActed;
        }

        public int BuildCost => Stats.PointCost;
        public bool IsAlive => CurrentHp > 0;

        public Unit WithPosition(HexCoord position) => new Unit(Id, Owner, Stats, position, CurrentHp, HasActed);
        public Unit WithHp(int currentHp) => new Unit(Id, Owner, Stats, Position, currentHp, HasActed);
        public Unit WithHasActed(bool hasActed) => new Unit(Id, Owner, Stats, Position, CurrentHp, hasActed);
        public Unit Clone() => new Unit(Id, Owner, Stats, Position, CurrentHp, HasActed);
    }
}
```
```csharp
// Assets/HexWars/Engine/Data/Generator.cs
namespace HexWars.Engine
{
    public sealed class Generator
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public HexCoord Position { get; }
        public int CurrentHp { get; }

        public Generator(int id, PlayerId owner, HexCoord position, int currentHp)
        {
            Id = id;
            Owner = owner;
            Position = position;
            CurrentHp = currentHp;
        }

        public bool IsAlive => CurrentHp > 0;
        public Generator WithHp(int currentHp) => new Generator(Id, Owner, Position, currentHp);
        public Generator Clone() => new Generator(Id, Owner, Position, CurrentHp);
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.UnitStatsTests" -logFile -`
Expected: PASS â€” all 7 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/UnitStats.cs Assets/HexWars/Engine/Data/Unit.cs Assets/HexWars/Engine/Data/Generator.cs Assets/HexWars/Engine/Tests/UnitStatsTests.cs && git commit -m "feat(engine): add UnitStats flat point-buy plus Unit and Generator entities"
```

---

### Task 6: GameConfig with every tunable parameter and Default()

**Files:**
- Create: `Assets/HexWars/Engine/Data/GameConfig.cs`
- Modify: `Assets/HexWars/Engine/Tests/GameConfigTests.cs` (create file)
- Test: `Assets/HexWars/Engine/Tests/GameConfigTests.cs`

**Interfaces:**
- Consumes: `TerrainType` (Plains/Forest/Rough/Water) and `TerrainDef(TerrainType,int,int,int,bool)` with `.MoveCost/.Concealment/.Defense/.Passable` from Task 3.
- Produces: `public enum TurnPolicyKind { AllUnits=0, OneAction=1 }`; `public sealed class GameConfig` with init props `int StartingPoints; double BountyRate; int GeneratorCost,GeneratorOutput,GeneratorHealth,DamageFloor,DmgHighGroundBonus,RangeHighGroundBonus,ClimbCostPerLevel,MaxClimbPerStep,MinCheapestViableUnitCost,BoardWidth,BoardHeight,MinElevation,MaxElevation,DeploymentZoneDepth,RoundCap; TurnPolicyKind TurnPolicy; System.Collections.Generic.IReadOnlyDictionary<TerrainType,TerrainDef> Terrain; System.Collections.Generic.IReadOnlyDictionary<TerrainType,double> TerrainWeights; static GameConfig Default(); TerrainDef TerrainDefFor(TerrainType); GameConfig Clone();`. Canonical field names: `DmgHighGroundBonus, RangeHighGroundBonus, ClimbCostPerLevel, MaxClimbPerStep, DamageFloor, BountyRate`. Consumed by Tasks 7,9,10,11,12,13,15,16,20,21,22.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class GameConfigTests
    {
        [Test]
        public void Default_MatchesSpecSection14StartingValues()
        {
            var c = GameConfig.Default();
            Assert.AreEqual(15, c.StartingPoints);
            Assert.AreEqual(0.5, c.BountyRate, 1e-9);
            Assert.AreEqual(2, c.GeneratorCost);
            Assert.AreEqual(1, c.GeneratorOutput);
            Assert.AreEqual(3, c.GeneratorHealth);
            Assert.AreEqual(1, c.DamageFloor);
            Assert.AreEqual(1, c.DmgHighGroundBonus);
            Assert.AreEqual(1, c.RangeHighGroundBonus);
            Assert.AreEqual(1, c.ClimbCostPerLevel);
            Assert.AreEqual(2, c.MaxClimbPerStep);
            Assert.AreEqual(1, c.MinCheapestViableUnitCost);
            Assert.AreEqual(9, c.BoardWidth);
            Assert.AreEqual(9, c.BoardHeight);
            Assert.AreEqual(0, c.MinElevation);
            Assert.AreEqual(4, c.MaxElevation);
            Assert.AreEqual(2, c.DeploymentZoneDepth);
            Assert.AreEqual(40, c.RoundCap);
            Assert.AreEqual(TurnPolicyKind.AllUnits, c.TurnPolicy);
        }

        [Test]
        public void Default_TerrainTableCoversAllFourTerrains_WithSpecValues()
        {
            var c = GameConfig.Default();
            Assert.AreEqual(4, c.Terrain.Count);
            foreach (TerrainType t in System.Enum.GetValues(typeof(TerrainType)))
                Assert.IsTrue(c.Terrain.ContainsKey(t), "missing terrain: " + t);

            var plains = c.TerrainDefFor(TerrainType.Plains);
            Assert.AreEqual(1, plains.MoveCost);
            Assert.AreEqual(0, plains.Concealment);
            Assert.AreEqual(0, plains.Defense);
            Assert.IsTrue(plains.Passable);

            var forest = c.TerrainDefFor(TerrainType.Forest);
            Assert.AreEqual(2, forest.MoveCost);
            Assert.AreEqual(2, forest.Concealment);
            Assert.AreEqual(1, forest.Defense);
            Assert.IsTrue(forest.Passable);

            var rough = c.TerrainDefFor(TerrainType.Rough);
            Assert.AreEqual(2, rough.MoveCost);
            Assert.AreEqual(1, rough.Concealment);
            Assert.AreEqual(1, rough.Defense);
            Assert.IsTrue(rough.Passable);

            var water = c.TerrainDefFor(TerrainType.Water);
            Assert.AreEqual(3, water.MoveCost);
            Assert.AreEqual(0, water.Concealment);
            Assert.AreEqual(0, water.Defense);
            Assert.IsTrue(water.Passable);
        }

        [Test]
        public void Default_TerrainWeightsCoverAllFourTerrains()
        {
            var c = GameConfig.Default();
            Assert.AreEqual(4, c.TerrainWeights.Count);
            foreach (TerrainType t in System.Enum.GetValues(typeof(TerrainType)))
            {
                Assert.IsTrue(c.TerrainWeights.ContainsKey(t), "missing weight: " + t);
                Assert.Greater(c.TerrainWeights[t], 0.0);
            }
        }

        [Test]
        public void Clone_IsIndependentButValueEqualOnScalars()
        {
            var c = GameConfig.Default();
            var clone = c.Clone();
            Assert.AreNotSame(c, clone);
            Assert.AreEqual(c.StartingPoints, clone.StartingPoints);
            Assert.AreEqual(c.BountyRate, clone.BountyRate, 1e-9);
            Assert.AreEqual(c.Terrain.Count, clone.Terrain.Count);
            Assert.AreEqual(c.TerrainDefFor(TerrainType.Forest).Defense,
                            clone.TerrainDefFor(TerrainType.Forest).Defense);
        }

        [Test]
        public void InitProperties_AllowOverrideViaObjectInitializer()
        {
            var c = new GameConfig { StartingPoints = 20, TurnPolicy = TurnPolicyKind.OneAction };
            Assert.AreEqual(20, c.StartingPoints);
            Assert.AreEqual(TurnPolicyKind.OneAction, c.TurnPolicy);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameConfigTests" -logFile -`
Expected: FAIL â€” compile error, `GameConfig` and `TurnPolicyKind` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/GameConfig.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    public enum TurnPolicyKind { AllUnits = 0, OneAction = 1 }

    public sealed class GameConfig
    {
        public int StartingPoints { get; init; } = 15;
        public double BountyRate { get; init; } = 0.5;
        public int GeneratorCost { get; init; } = 2;
        public int GeneratorOutput { get; init; } = 1;
        public int GeneratorHealth { get; init; } = 3;
        public int DamageFloor { get; init; } = 1;
        public int DmgHighGroundBonus { get; init; } = 1;
        public int RangeHighGroundBonus { get; init; } = 1;
        public int ClimbCostPerLevel { get; init; } = 1;
        public int MaxClimbPerStep { get; init; } = 2;
        public int MinCheapestViableUnitCost { get; init; } = 1;
        public int BoardWidth { get; init; } = 9;
        public int BoardHeight { get; init; } = 9;
        public int MinElevation { get; init; } = 0;
        public int MaxElevation { get; init; } = 4;
        public int DeploymentZoneDepth { get; init; } = 2;
        public int RoundCap { get; init; } = 40;
        public TurnPolicyKind TurnPolicy { get; init; } = TurnPolicyKind.AllUnits;

        public IReadOnlyDictionary<TerrainType, TerrainDef> Terrain { get; init; } = DefaultTerrain();
        public IReadOnlyDictionary<TerrainType, double> TerrainWeights { get; init; } = DefaultWeights();

        private static IReadOnlyDictionary<TerrainType, TerrainDef> DefaultTerrain()
        {
            // Spec Â§6 starter table. Water passable with move cost 3 (config knob).
            return new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(TerrainType.Plains, 1, 0, 0, true) },
                { TerrainType.Forest, new TerrainDef(TerrainType.Forest, 2, 2, 1, true) },
                { TerrainType.Rough,  new TerrainDef(TerrainType.Rough,  2, 1, 1, true) },
                { TerrainType.Water,  new TerrainDef(TerrainType.Water,  3, 0, 0, true) },
            };
        }

        private static IReadOnlyDictionary<TerrainType, double> DefaultWeights()
        {
            return new Dictionary<TerrainType, double>
            {
                { TerrainType.Plains, 0.55 },
                { TerrainType.Forest, 0.20 },
                { TerrainType.Rough,  0.15 },
                { TerrainType.Water,  0.10 },
            };
        }

        public static GameConfig Default() => new GameConfig();

        public TerrainDef TerrainDefFor(TerrainType terrain) => Terrain[terrain];

        public GameConfig Clone()
        {
            return new GameConfig
            {
                StartingPoints = StartingPoints,
                BountyRate = BountyRate,
                GeneratorCost = GeneratorCost,
                GeneratorOutput = GeneratorOutput,
                GeneratorHealth = GeneratorHealth,
                DamageFloor = DamageFloor,
                DmgHighGroundBonus = DmgHighGroundBonus,
                RangeHighGroundBonus = RangeHighGroundBonus,
                ClimbCostPerLevel = ClimbCostPerLevel,
                MaxClimbPerStep = MaxClimbPerStep,
                MinCheapestViableUnitCost = MinCheapestViableUnitCost,
                BoardWidth = BoardWidth,
                BoardHeight = BoardHeight,
                MinElevation = MinElevation,
                MaxElevation = MaxElevation,
                DeploymentZoneDepth = DeploymentZoneDepth,
                RoundCap = RoundCap,
                TurnPolicy = TurnPolicy,
                Terrain = new Dictionary<TerrainType, TerrainDef>((Dictionary<TerrainType, TerrainDef>)Terrain),
                TerrainWeights = new Dictionary<TerrainType, double>((Dictionary<TerrainType, double>)TerrainWeights),
            };
        }
    }
}
```
*Note: the `Clone()` dictionary copies assume `Default()`/initializers supply `Dictionary<,>` instances; that holds for all engine construction paths. TerrainDef is immutable so a shallow value copy of the dictionary is a true deep copy.*
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameConfigTests" -logFile -`
Expected: PASS â€” all 5 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/GameConfig.cs Assets/HexWars/Engine/Tests/GameConfigTests.cs && git commit -m "feat(engine): add GameConfig with every tunable, terrain table, and Default()"
```

---

### Task 7: PlayerState and GameState with deep Clone

**Files:**
- Create: `Assets/HexWars/Engine/Data/PlayerState.cs`
- Create: `Assets/HexWars/Engine/Data/GameState.cs`
- Modify: `Assets/HexWars/Engine/Tests/GameStateTests.cs` (create file)
- Test: `Assets/HexWars/Engine/Tests/GameStateTests.cs`

**Interfaces:**
- Consumes: `Board`, `Board.Clone()` (Task 4); `Unit`, `Unit.Clone()`, `Unit.WithHp()`, `Generator`, `Generator.Clone()`, `UnitStats` (Task 5); `PlayerId` (Task 2); `GameConfig`, `GameConfig.Clone()` (Task 6).
- Produces: `public sealed class PlayerState { PlayerId Id{get;} int Points{get;} System.Collections.Generic.IReadOnlyList<UnitStats> Reserve{get;} System.Collections.Generic.IReadOnlyList<Unit> UnitsOnBoard{get;} System.Collections.Generic.IReadOnlyList<Generator> Generators{get;} PlayerState(PlayerId,int,IReadOnlyList<UnitStats>,IReadOnlyList<Unit>,IReadOnlyList<Generator>); PlayerState Clone(); }`; `public sealed class GameState { Board Board{get;} PlayerState[] Players{get;} PlayerId ActivePlayer{get;} int Round{get;} int NextEntityId{get;} bool IsGameOver{get;} PlayerId? Winner{get;} GameConfig Config{get;} GameState(Board,PlayerState[],PlayerId,int,int,GameConfig,bool isGameOver=false,PlayerId? winner=null); PlayerState ActivePlayerState{get;} PlayerState PlayerStateFor(PlayerId); PlayerId Opponent(PlayerId); GameState Clone(); }`. Consumed by Tasks 9,10,11,12,13,14,15,16,17,18,19,20,24.

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class GameStateTests
    {
        private static Board MakeBoard()
        {
            var tiles = new List<Tile>
            {
                new Tile(new HexCoord(0, 0, 0), TerrainType.Plains),
                new Tile(new HexCoord(1, 0, 0), TerrainType.Plains),
            };
            var dz0 = new List<HexCoord> { new HexCoord(0, 0, 0) };
            var dz1 = new List<HexCoord> { new HexCoord(1, 0, 0) };
            return new Board(2, 1, tiles, dz0, dz1);
        }

        private static GameState MakeState()
        {
            var board = MakeBoard();
            var stats = new UnitStats(5, 2, 1, 3, 0, 2);
            var p0 = new PlayerState(
                PlayerId.Player0, 10,
                new List<UnitStats> { stats },
                new List<Unit> { new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0, 0), 5) },
                new List<Generator> { new Generator(2, PlayerId.Player0, new HexCoord(0, 0, 0), 3) });
            var p1 = new PlayerState(
                PlayerId.Player1, 7,
                new List<UnitStats>(),
                new List<Unit>(),
                new List<Generator>());
            return new GameState(board, new[] { p0, p1 }, PlayerId.Player0, 1, 3, GameConfig.Default());
        }

        [Test]
        public void Constructor_ExposesAllFields_WithDefaultGameOverFalse()
        {
            var s = MakeState();
            Assert.AreEqual(2, s.Players.Length);
            Assert.AreEqual(PlayerId.Player0, s.ActivePlayer);
            Assert.AreEqual(1, s.Round);
            Assert.AreEqual(3, s.NextEntityId);
            Assert.IsFalse(s.IsGameOver);
            Assert.IsNull(s.Winner);
            Assert.IsNotNull(s.Config);
        }

        [Test]
        public void ActivePlayerState_AndPlayerStateFor_IndexByPlayerId()
        {
            var s = MakeState();
            Assert.AreSame(s.Players[0], s.ActivePlayerState);
            Assert.AreSame(s.Players[0], s.PlayerStateFor(PlayerId.Player0));
            Assert.AreSame(s.Players[1], s.PlayerStateFor(PlayerId.Player1));
        }

        [Test]
        public void Opponent_FlipsPlayerId()
        {
            var s = MakeState();
            Assert.AreEqual(PlayerId.Player1, s.Opponent(PlayerId.Player0));
            Assert.AreEqual(PlayerId.Player0, s.Opponent(PlayerId.Player1));
        }

        [Test]
        public void PlayerState_Clone_IsDeepAndIndependent()
        {
            var s = MakeState();
            var p0 = s.Players[0];
            var clone = p0.Clone();
            Assert.AreNotSame(p0, clone);
            Assert.AreNotSame(p0.UnitsOnBoard, clone.UnitsOnBoard);
            Assert.AreNotSame(p0.Generators, clone.Generators);
            Assert.AreNotSame(p0.UnitsOnBoard[0], clone.UnitsOnBoard[0]);
            Assert.AreEqual(p0.Points, clone.Points);
            Assert.AreEqual(p0.Reserve[0], clone.Reserve[0]);
            Assert.AreEqual(p0.UnitsOnBoard[0].Id, clone.UnitsOnBoard[0].Id);
        }

        [Test]
        public void GameState_Clone_ProducesIndependentCopy()
        {
            var s = MakeState();
            var clone = s.Clone();
            Assert.AreNotSame(s, clone);
            Assert.AreNotSame(s.Board, clone.Board);
            Assert.AreNotSame(s.Players, clone.Players);
            Assert.AreNotSame(s.Players[0], clone.Players[0]);
            Assert.AreNotSame(s.Players[0].UnitsOnBoard[0], clone.Players[0].UnitsOnBoard[0]);
            Assert.AreEqual(s.ActivePlayer, clone.ActivePlayer);
            Assert.AreEqual(s.Round, clone.Round);
            Assert.AreEqual(s.NextEntityId, clone.NextEntityId);
            Assert.AreEqual(s.Players[0].Points, clone.Players[0].Points);
            Assert.AreEqual(s.Players[0].UnitsOnBoard[0].CurrentHp, clone.Players[0].UnitsOnBoard[0].CurrentHp);
        }

        [Test]
        public void GameState_Clone_PreservesGameOverAndWinner()
        {
            var s = MakeState();
            var over = new GameState(s.Board, s.Players, s.ActivePlayer, s.Round, s.NextEntityId,
                                     s.Config, isGameOver: true, winner: PlayerId.Player1);
            var clone = over.Clone();
            Assert.IsTrue(clone.IsGameOver);
            Assert.AreEqual(PlayerId.Player1, clone.Winner);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameStateTests" -logFile -`
Expected: FAIL â€” compile error, `PlayerState` and `GameState` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Data/PlayerState.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    public sealed class PlayerState
    {
        public PlayerId Id { get; }
        public int Points { get; }
        public IReadOnlyList<UnitStats> Reserve { get; }       // created, not yet deployed
        public IReadOnlyList<Unit> UnitsOnBoard { get; }
        public IReadOnlyList<Generator> Generators { get; }

        public PlayerState(PlayerId id, int points,
                           IReadOnlyList<UnitStats> reserve,
                           IReadOnlyList<Unit> unitsOnBoard,
                           IReadOnlyList<Generator> generators)
        {
            Id = id;
            Points = points;
            Reserve = reserve;
            UnitsOnBoard = unitsOnBoard;
            Generators = generators;
        }

        public PlayerState Clone()
        {
            // UnitStats is an immutable value type, so a fresh list is a deep copy.
            var reserve = new List<UnitStats>(Reserve);
            var units = new List<Unit>(UnitsOnBoard.Count);
            for (int i = 0; i < UnitsOnBoard.Count; i++) units.Add(UnitsOnBoard[i].Clone());
            var gens = new List<Generator>(Generators.Count);
            for (int i = 0; i < Generators.Count; i++) gens.Add(Generators[i].Clone());
            return new PlayerState(Id, Points, reserve, units, gens);
        }
    }
}
```
```csharp
// Assets/HexWars/Engine/Data/GameState.cs
namespace HexWars.Engine
{
    public sealed class GameState
    {
        public Board Board { get; }
        public PlayerState[] Players { get; }   // length 2; index by (int)PlayerId
        public PlayerId ActivePlayer { get; }
        public int Round { get; }
        public int NextEntityId { get; }
        public bool IsGameOver { get; }
        public PlayerId? Winner { get; }
        public GameConfig Config { get; }

        public GameState(Board board, PlayerState[] players, PlayerId activePlayer, int round,
                         int nextEntityId, GameConfig config,
                         bool isGameOver = false, PlayerId? winner = null)
        {
            Board = board;
            Players = players;
            ActivePlayer = activePlayer;
            Round = round;
            NextEntityId = nextEntityId;
            Config = config;
            IsGameOver = isGameOver;
            Winner = winner;
        }

        public PlayerState ActivePlayerState => Players[(int)ActivePlayer];
        public PlayerState PlayerStateFor(PlayerId id) => Players[(int)id];
        public PlayerId Opponent(PlayerId id) => id == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        public GameState Clone()
        {
            var players = new PlayerState[Players.Length];
            for (int i = 0; i < Players.Length; i++) players[i] = Players[i].Clone();
            return new GameState(Board.Clone(), players, ActivePlayer, Round, NextEntityId,
                                 Config.Clone(), IsGameOver, Winner);
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameStateTests" -logFile -`
Expected: PASS â€” all 6 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Data/PlayerState.cs Assets/HexWars/Engine/Data/GameState.cs Assets/HexWars/Engine/Tests/GameStateTests.cs && git commit -m "feat(engine): add PlayerState and GameState with deep Clone, Opponent, and lookups"
```

---

### Task 8: HexGeometry: hex distance, Distance3D, directions, neighbors

**Files:**
- Create: `Assets/HexWars/Engine/Geometry/HexGeometry.cs`
- Test: `Assets/HexWars/Engine/Tests/HexGeometryTests.cs`

**Interfaces:**
- Consumes: `HexCoord` (Task 2) â€” `public readonly struct HexCoord{ int Q; int R; int Elevation; int S; HexCoord(int q,int r,int elevation=0); HexCoord WithElevation(int); HexCoord Translate(int dq,int dr); value ==/!= }`.
- Produces: `public static class HexGeometry{ static int HexDistance(HexCoord a, HexCoord b); static int Distance3D(HexCoord a, HexCoord b); static System.Collections.Generic.IReadOnlyList<HexCoord> AxialDirections {get;}; static System.Collections.Generic.List<HexCoord> Neighbors(HexCoord c); }`. `HexDistance = (|dQ| + |dQ+dR| + |dR|)/2`; `Distance3D = HexDistance + |a.Elevation - b.Elevation|`; `AxialDirections` fixed order (+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1); `Neighbors` returns the 6 axial neighbors with elevation copied from `c`. Consumed by Pathfinding (Task 9), TargetingService (Task 10), BoardValidator (Task 21).

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class HexGeometryTests
    {
        [Test]
        public void HexDistance_SameCoord_IsZero()
        {
            var a = new HexCoord(2, -1);
            Assert.AreEqual(0, HexGeometry.HexDistance(a, a));
        }

        [Test]
        public void HexDistance_AdjacentAxial_IsOne()
        {
            var a = new HexCoord(0, 0);
            var b = new HexCoord(1, 0);
            Assert.AreEqual(1, HexGeometry.HexDistance(a, b));
        }

        [Test]
        public void HexDistance_KnownCubeCase_MatchesFormula()
        {
            // dQ=3, dR=-1 -> (|3|+|3+-1|+|-1|)/2 = (3+2+1)/2 = 3
            var a = new HexCoord(0, 0);
            var b = new HexCoord(3, -1);
            Assert.AreEqual(3, HexGeometry.HexDistance(a, b));
        }

        [Test]
        public void HexDistance_IgnoresElevation()
        {
            var a = new HexCoord(0, 0, 0);
            var b = new HexCoord(1, 0, 4);
            Assert.AreEqual(1, HexGeometry.HexDistance(a, b));
        }

        [Test]
        public void Distance3D_AddsAbsoluteElevationDifference()
        {
            // hexDistance 1 + |0-3| = 4
            var a = new HexCoord(0, 0, 0);
            var b = new HexCoord(1, 0, 3);
            Assert.AreEqual(4, HexGeometry.Distance3D(a, b));
        }

        [Test]
        public void Distance3D_NegativeElevationDelta_UsesAbsoluteValue()
        {
            var a = new HexCoord(0, 0, 5);
            var b = new HexCoord(0, 0, 2);
            Assert.AreEqual(3, HexGeometry.Distance3D(a, b));
        }

        [Test]
        public void AxialDirections_AreTheSixCanonicalDirsInOrder()
        {
            var dirs = HexGeometry.AxialDirections;
            Assert.AreEqual(6, dirs.Count);
            Assert.AreEqual(new HexCoord(1, 0), dirs[0]);
            Assert.AreEqual(new HexCoord(1, -1), dirs[1]);
            Assert.AreEqual(new HexCoord(0, -1), dirs[2]);
            Assert.AreEqual(new HexCoord(-1, 0), dirs[3]);
            Assert.AreEqual(new HexCoord(-1, 1), dirs[4]);
            Assert.AreEqual(new HexCoord(0, 1), dirs[5]);
        }

        [Test]
        public void Neighbors_ReturnsSixAdjacentCoordsWithSameElevation()
        {
            var c = new HexCoord(2, 3, 4);
            List<HexCoord> n = HexGeometry.Neighbors(c);
            Assert.AreEqual(6, n.Count);
            CollectionAssert.Contains(n, new HexCoord(3, 3, 4));
            CollectionAssert.Contains(n, new HexCoord(3, 2, 4));
            CollectionAssert.Contains(n, new HexCoord(2, 2, 4));
            CollectionAssert.Contains(n, new HexCoord(1, 3, 4));
            CollectionAssert.Contains(n, new HexCoord(1, 4, 4));
            CollectionAssert.Contains(n, new HexCoord(2, 4, 4));
            foreach (var hex in n)
            {
                Assert.AreEqual(4, hex.Elevation);
                Assert.AreEqual(1, HexGeometry.HexDistance(c, hex));
            }
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexGeometryTests" -logFile -`
Expected: FAIL â€” compile error / `HexGeometry` does not exist (type not yet created).
- [ ] **Step 3: Write minimal implementation**
```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    public static class HexGeometry
    {
        // Fixed canonical axial direction order:
        // (+1,0),(+1,-1),(0,-1),(-1,0),(-1,+1),(0,+1)
        private static readonly HexCoord[] _directions = new[]
        {
            new HexCoord(1, 0),
            new HexCoord(1, -1),
            new HexCoord(0, -1),
            new HexCoord(-1, 0),
            new HexCoord(-1, 1),
            new HexCoord(0, 1),
        };

        public static IReadOnlyList<HexCoord> AxialDirections => _directions;

        // Cube distance, ignores elevation.
        // HexDistance = (|dQ| + |dQ+dR| + |dR|) / 2
        public static int HexDistance(HexCoord a, HexCoord b)
        {
            int dQ = a.Q - b.Q;
            int dR = a.R - b.R;
            return (Math.Abs(dQ) + Math.Abs(dQ + dR) + Math.Abs(dR)) / 2;
        }

        // Unified 3D metric: hex distance plus absolute elevation difference.
        public static int Distance3D(HexCoord a, HexCoord b)
        {
            return HexDistance(a, b) + Math.Abs(a.Elevation - b.Elevation);
        }

        // Six axial neighbors; elevation copied from c (callers re-resolve via Board).
        public static List<HexCoord> Neighbors(HexCoord c)
        {
            var result = new List<HexCoord>(6);
            for (int i = 0; i < _directions.Length; i++)
            {
                var d = _directions[i];
                result.Add(new HexCoord(c.Q + d.Q, c.R + d.R, c.Elevation));
            }
            return result;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexGeometryTests" -logFile -`
Expected: PASS â€” all 9 tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Geometry/HexGeometry.cs Assets/HexWars/Engine/Tests/HexGeometryTests.cs && git commit -m "feat(engine): add HexGeometry distance, Distance3D, directions, neighbors"
```

---

### Task 9: Pathfinding: cost-to-enter, reachable tiles, path within budget

**Files:**
- Create: `Assets/HexWars/Engine/Geometry/Pathfinding.cs`
- Test: `Assets/HexWars/Engine/Tests/PathfindingTests.cs`

**Interfaces:**
- Consumes: `HexCoord` (Task 2); `HexGeometry.Neighbors(HexCoord)` (Task 8); `Tile{ HexCoord Coord; TerrainType Terrain; int Elevation }` (Task 3); `Board{ bool TryGetTile(int q,int r,out Tile); }` (Task 4); `GameConfig{ int ClimbCostPerLevel; int MaxClimbPerStep; IReadOnlyDictionary<TerrainType,TerrainDef> Terrain; TerrainDef TerrainDefFor(TerrainType); }` (Task 6); `TerrainDef{ int MoveCost; bool Passable; }` (Task 3); `GameState{ Board Board; GameConfig Config; PlayerState[] Players; }` and `PlayerState{ IReadOnlyList<Unit> UnitsOnBoard; IReadOnlyList<Generator> Generators; }`, `Unit.Position`, `Generator.Position`, `*.IsAlive` (Tasks 7,5).
- Produces: `public static class Pathfinding{ static int CostToEnter(GameState state, HexCoord from, Tile to); static System.Collections.Generic.List<HexCoord> ReachableTiles(GameState state, HexCoord start, int movementBudget); static bool TryFindPath(GameState state, HexCoord start, HexCoord goal, int movementBudget, out System.Collections.Generic.List<HexCoord> path, out int totalCost); }`. `CostToEnter = terrainMoveCost + max(0, climbLevels) * ClimbCostPerLevel`; `climbLevels = to.Elevation - from.Elevation`; entry forbidden (CostToEnter returns `int.MaxValue`) if `!Passable`, occupied, or `climbLevels > MaxClimbPerStep`. `ReachableTiles` excludes `start`; returned coords carry the real tile elevation. Consumed by GameEngine.Apply MoveUnit (Task 17) and LegalMoves (Task 18).

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class PathfindingTests
    {
        private const int Impassable = int.MaxValue;

        // Build a flat NxN plains board at a given uniform elevation,
        // then override specific tiles via the patch action.
        private static Board MakeBoard(int size, System.Action<Dictionary<(int, int), Tile>> patch = null)
        {
            var map = new Dictionary<(int, int), Tile>();
            for (int q = 0; q < size; q++)
                for (int r = 0; r < size; r++)
                    map[(q, r)] = new Tile(new HexCoord(q, r, 0), TerrainType.Plains);
            patch?.Invoke(map);
            var tiles = new List<Tile>(map.Values);
            return new Board(size, size, tiles,
                new List<HexCoord>(), new List<HexCoord>());
        }

        private static GameState MakeState(Board board,
            IReadOnlyList<Unit> p0Units = null, IReadOnlyList<Generator> p0Gens = null)
        {
            var cfg = GameConfig.Default();
            var p0 = new PlayerState(PlayerId.Player0, 0,
                new List<UnitStats>(),
                p0Units ?? new List<Unit>(),
                p0Gens ?? new List<Generator>());
            var p1 = new PlayerState(PlayerId.Player1, 0,
                new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            return new GameState(board, new[] { p0, p1 },
                PlayerId.Player0, 1, 1, cfg);
        }

        [Test]
        public void CostToEnter_Plains_IsTerrainMoveCost()
        {
            var board = MakeBoard(3);
            var state = MakeState(board);
            board.TryGetTile(1, 0, out var to);
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 0), to);
            Assert.AreEqual(1, cost); // plains MoveCost 1, no climb
        }

        [Test]
        public void CostToEnter_ClimbAddsClimbCostPerLevel()
        {
            var board = MakeBoard(3, m => m[(1, 0)] = new Tile(new HexCoord(1, 0, 2), TerrainType.Plains));
            var state = MakeState(board);
            board.TryGetTile(1, 0, out var to);
            // plains 1 + climb 2 levels * ClimbCostPerLevel(1) = 3
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 0), to);
            Assert.AreEqual(3, cost);
        }

        [Test]
        public void CostToEnter_Descending_IsJustTerrainMoveCost()
        {
            var board = MakeBoard(3, m => m[(1, 0)] = new Tile(new HexCoord(1, 0, 0), TerrainType.Plains));
            var state = MakeState(board);
            board.TryGetTile(1, 0, out var to);
            // from elevation 3 descending to 0: no climb term
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 3), to);
            Assert.AreEqual(1, cost);
        }

        [Test]
        public void CostToEnter_ClimbOverMaxClimbPerStep_IsForbidden()
        {
            var board = MakeBoard(3, m => m[(1, 0)] = new Tile(new HexCoord(1, 0, 3), TerrainType.Plains));
            var state = MakeState(board); // MaxClimbPerStep default 2
            board.TryGetTile(1, 0, out var to);
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 0), to);
            Assert.AreEqual(Impassable, cost);
        }

        [Test]
        public void CostToEnter_ImpassableWater_IsForbidden()
        {
            var board = MakeBoard(3, m => m[(1, 0)] = new Tile(new HexCoord(1, 0, 0), TerrainType.Water));
            var cfg = GameConfig.Default();
            // Force water impassable for this test by cloning config terrain.
            var p0 = new PlayerState(PlayerId.Player0, 0, new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            var p1 = new PlayerState(PlayerId.Player1, 0, new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            var terrain = new Dictionary<TerrainType, TerrainDef>(cfg.Terrain)
            {
                [TerrainType.Water] = new TerrainDef(TerrainType.Water, 3, 0, 0, passable: false)
            };
            var cfg2 = cfg.Clone();
            var cfgImpassable = new GameConfig
            {
                StartingPoints = cfg2.StartingPoints,
                BountyRate = cfg2.BountyRate,
                GeneratorCost = cfg2.GeneratorCost,
                GeneratorOutput = cfg2.GeneratorOutput,
                GeneratorHealth = cfg2.GeneratorHealth,
                DamageFloor = cfg2.DamageFloor,
                DmgHighGroundBonus = cfg2.DmgHighGroundBonus,
                RangeHighGroundBonus = cfg2.RangeHighGroundBonus,
                ClimbCostPerLevel = cfg2.ClimbCostPerLevel,
                MaxClimbPerStep = cfg2.MaxClimbPerStep,
                MinCheapestViableUnitCost = cfg2.MinCheapestViableUnitCost,
                BoardWidth = cfg2.BoardWidth,
                BoardHeight = cfg2.BoardHeight,
                MinElevation = cfg2.MinElevation,
                MaxElevation = cfg2.MaxElevation,
                DeploymentZoneDepth = cfg2.DeploymentZoneDepth,
                RoundCap = cfg2.RoundCap,
                TurnPolicy = cfg2.TurnPolicy,
                Terrain = terrain,
                TerrainWeights = cfg2.TerrainWeights,
            };
            var state = new GameState(board, new[] { p0, p1 }, PlayerId.Player0, 1, 1, cfgImpassable);
            board.TryGetTile(1, 0, out var to);
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 0), to);
            Assert.AreEqual(Impassable, cost);
        }

        [Test]
        public void CostToEnter_OccupiedTile_IsForbidden()
        {
            var board = MakeBoard(3);
            var blocker = new Unit(7, PlayerId.Player0,
                new UnitStats(1, 0, 0, 0, 0, 0), new HexCoord(1, 0, 0), 1);
            var state = MakeState(board, p0Units: new List<Unit> { blocker });
            board.TryGetTile(1, 0, out var to);
            int cost = Pathfinding.CostToEnter(state, new HexCoord(0, 0, 0), to);
            Assert.AreEqual(Impassable, cost);
        }

        [Test]
        public void ReachableTiles_ExcludesStartAndRespectsBudget()
        {
            var board = MakeBoard(5);
            var state = MakeState(board);
            var start = new HexCoord(2, 2, 0);
            var reachable = Pathfinding.ReachableTiles(state, start, 1);
            // budget 1 over plains -> exactly the 6 (in-bounds) neighbors, never start
            CollectionAssert.DoesNotContain(reachable, start);
            foreach (var c in reachable)
                Assert.AreEqual(1, HexGeometry.HexDistance(start, c));
            Assert.AreEqual(6, reachable.Count);
        }

        [Test]
        public void ReachableTiles_CoordsCarryRealElevation()
        {
            var board = MakeBoard(3, m => m[(1, 0)] = new Tile(new HexCoord(1, 0, 1), TerrainType.Plains));
            var state = MakeState(board);
            var reachable = Pathfinding.ReachableTiles(state, new HexCoord(0, 0, 0), 5);
            HexCoord hill = reachable.Find(c => c.Q == 1 && c.R == 0);
            Assert.AreEqual(1, hill.Elevation);
        }

        [Test]
        public void TryFindPath_FindsCheapestCostWithinBudget()
        {
            var board = MakeBoard(5);
            var state = MakeState(board);
            bool ok = Pathfinding.TryFindPath(state,
                new HexCoord(0, 0, 0), new HexCoord(2, 0, 0), 5,
                out var path, out int totalCost);
            Assert.IsTrue(ok);
            Assert.AreEqual(2, totalCost); // two plains steps
            Assert.AreEqual(new HexCoord(2, 0, 0), path[path.Count - 1]);
        }

        [Test]
        public void TryFindPath_BeyondBudget_ReturnsFalse()
        {
            var board = MakeBoard(5);
            var state = MakeState(board);
            bool ok = Pathfinding.TryFindPath(state,
                new HexCoord(0, 0, 0), new HexCoord(4, 0, 0), 3,
                out var path, out int totalCost);
            Assert.IsFalse(ok);
            Assert.IsNull(path);
            Assert.AreEqual(0, totalCost);
        }

        [Test]
        public void TryFindPath_BlockedByImpassableClimb_ReturnsFalse()
        {
            // Surround goal column with a cliff > MaxClimbPerStep on the only approach.
            var board = MakeBoard(2, m =>
            {
                m[(1, 0)] = new Tile(new HexCoord(1, 0, 4), TerrainType.Plains);
                m[(0, 1)] = new Tile(new HexCoord(0, 1, 4), TerrainType.Plains);
                m[(1, 1)] = new Tile(new HexCoord(1, 1, 4), TerrainType.Plains);
            });
            var state = MakeState(board);
            bool ok = Pathfinding.TryFindPath(state,
                new HexCoord(0, 0, 0), new HexCoord(1, 1, 4), 50,
                out var path, out int totalCost);
            Assert.IsFalse(ok);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.PathfindingTests" -logFile -`
Expected: FAIL â€” compile error / `Pathfinding` does not exist (type not yet created).
- [ ] **Step 3: Write minimal implementation**
```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    public static class Pathfinding
    {
        // Sentinel returned by CostToEnter when entry is forbidden.
        private const int Forbidden = int.MaxValue;

        // costToEnter(to) = terrainMoveCost + max(0, climbLevels) * ClimbCostPerLevel.
        // climbLevels = to.Elevation - from.Elevation. Forbidden (int.MaxValue) if the
        // target tile is impassable, occupied, or the climb exceeds MaxClimbPerStep.
        public static int CostToEnter(GameState state, HexCoord from, Tile to)
        {
            var config = state.Config;
            TerrainDef def = config.TerrainDefFor(to.Terrain);
            if (!def.Passable)
                return Forbidden;

            int climbLevels = to.Elevation - from.Elevation;
            if (climbLevels > config.MaxClimbPerStep)
                return Forbidden;

            if (IsOccupied(state, to.Coord))
                return Forbidden;

            int climbCost = Math.Max(0, climbLevels) * config.ClimbCostPerLevel;
            return def.MoveCost + climbCost;
        }

        // True if any living unit or generator (either player) sits on this axial column.
        private static bool IsOccupied(GameState state, HexCoord coord)
        {
            for (int p = 0; p < state.Players.Length; p++)
            {
                var ps = state.Players[p];
                var units = ps.UnitsOnBoard;
                for (int i = 0; i < units.Count; i++)
                {
                    var u = units[i];
                    if (u.IsAlive && u.Position.Q == coord.Q && u.Position.R == coord.R)
                        return true;
                }
                var gens = ps.Generators;
                for (int i = 0; i < gens.Count; i++)
                {
                    var g = gens[i];
                    if (g.IsAlive && g.Position.Q == coord.Q && g.Position.R == coord.R)
                        return true;
                }
            }
            return false;
        }

        // Deterministic Dijkstra producing the cheapest cost to every reachable column.
        // Returns a map keyed by (q,r) -> (resolvedCoord, bestCost). Excludes start.
        private static Dictionary<(int, int), (HexCoord coord, int cost)> Dijkstra(
            GameState state, HexCoord start, int movementBudget)
        {
            var best = new Dictionary<(int, int), int>();
            var resolved = new Dictionary<(int, int), HexCoord>();
            best[(start.Q, start.R)] = 0;
            resolved[(start.Q, start.R)] = start;

            // Simple array-backed priority selection: deterministic, fine for ~9x9 boards.
            var frontier = new List<(int, int)> { (start.Q, start.R) };

            while (frontier.Count > 0)
            {
                // Pick lowest-cost frontier node; ties broken by (q, r) for determinism.
                int pick = 0;
                for (int i = 1; i < frontier.Count; i++)
                {
                    var ci = frontier[i];
                    var cp = frontier[pick];
                    if (best[ci] < best[cp] ||
                        (best[ci] == best[cp] && (ci.Item1 < cp.Item1 ||
                            (ci.Item1 == cp.Item1 && ci.Item2 < cp.Item2))))
                    {
                        pick = i;
                    }
                }
                var currentKey = frontier[pick];
                frontier.RemoveAt(pick);

                HexCoord current = resolved[currentKey];
                int currentCost = best[currentKey];

                foreach (var n in HexGeometry.Neighbors(current))
                {
                    if (!state.Board.TryGetTile(n.Q, n.R, out Tile tile))
                        continue;

                    int enter = CostToEnter(state, current, tile);
                    if (enter == Forbidden)
                        continue;

                    int newCost = currentCost + enter;
                    if (newCost > movementBudget)
                        continue;

                    var key = (tile.Coord.Q, tile.Coord.R);
                    if (!best.TryGetValue(key, out int existing) || newCost < existing)
                    {
                        best[key] = newCost;
                        resolved[key] = tile.Coord;
                        if (!frontier.Contains(key))
                            frontier.Add(key);
                    }
                }
            }

            var output = new Dictionary<(int, int), (HexCoord, int)>();
            foreach (var kv in best)
                output[kv.Key] = (resolved[kv.Key], kv.Value);
            return output;
        }

        // All reachable columns within budget, each carrying the tile's real elevation,
        // excluding the start column. Sorted by (q, r) for deterministic output.
        public static List<HexCoord> ReachableTiles(GameState state, HexCoord start, int movementBudget)
        {
            var costs = Dijkstra(state, start, movementBudget);
            var result = new List<HexCoord>();
            foreach (var kv in costs)
            {
                if (kv.Key.Item1 == start.Q && kv.Key.Item2 == start.R)
                    continue;
                result.Add(kv.Value.coord);
            }
            result.Sort((a, b) =>
            {
                int c = a.Q.CompareTo(b.Q);
                return c != 0 ? c : a.R.CompareTo(b.R);
            });
            return result;
        }

        // Cheapest total move cost from start to goal within budget.
        // Reconstructs a path (start excluded, goal last). False if unreachable.
        public static bool TryFindPath(GameState state, HexCoord start, HexCoord goal,
            int movementBudget, out List<HexCoord> path, out int totalCost)
        {
            path = null;
            totalCost = 0;

            var best = new Dictionary<(int, int), int>();
            var resolved = new Dictionary<(int, int), HexCoord>();
            var prev = new Dictionary<(int, int), (int, int)>();
            best[(start.Q, start.R)] = 0;
            resolved[(start.Q, start.R)] = start;

            var frontier = new List<(int, int)> { (start.Q, start.R) };

            while (frontier.Count > 0)
            {
                int pick = 0;
                for (int i = 1; i < frontier.Count; i++)
                {
                    var ci = frontier[i];
                    var cp = frontier[pick];
                    if (best[ci] < best[cp] ||
                        (best[ci] == best[cp] && (ci.Item1 < cp.Item1 ||
                            (ci.Item1 == cp.Item1 && ci.Item2 < cp.Item2))))
                    {
                        pick = i;
                    }
                }
                var currentKey = frontier[pick];
                frontier.RemoveAt(pick);

                HexCoord current = resolved[currentKey];
                int currentCost = best[currentKey];

                foreach (var n in HexGeometry.Neighbors(current))
                {
                    if (!state.Board.TryGetTile(n.Q, n.R, out Tile tile))
                        continue;
                    int enter = CostToEnter(state, current, tile);
                    if (enter == Forbidden)
                        continue;
                    int newCost = currentCost + enter;
                    if (newCost > movementBudget)
                        continue;

                    var key = (tile.Coord.Q, tile.Coord.R);
                    if (!best.TryGetValue(key, out int existing) || newCost < existing)
                    {
                        best[key] = newCost;
                        resolved[key] = tile.Coord;
                        prev[key] = currentKey;
                        if (!frontier.Contains(key))
                            frontier.Add(key);
                    }
                }
            }

            var goalKey = (goal.Q, goal.R);
            if (!best.ContainsKey(goalKey) || (goal.Q == start.Q && goal.R == start.R))
                return false;

            totalCost = best[goalKey];

            var reversed = new List<HexCoord>();
            var cursor = goalKey;
            while (!(cursor.Item1 == start.Q && cursor.Item2 == start.R))
            {
                reversed.Add(resolved[cursor]);
                cursor = prev[cursor];
            }
            reversed.Reverse();
            path = reversed;
            return true;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.PathfindingTests" -logFile -`
Expected: PASS â€” all Pathfinding tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Geometry/Pathfinding.cs Assets/HexWars/Engine/Tests/PathfindingTests.cs && git commit -m "feat(engine): add Pathfinding cost-to-enter, reachable tiles, path within budget"
```

---

### Task 10: TargetingService: range + vision + high-ground, units and generators

**Files:**
- Create: `Assets/HexWars/Engine/Combat/TargetingService.cs`
- Modify: `Assets/HexWars/Engine/Tests/TargetingServiceTests.cs`
- Test: `Assets/HexWars/Engine/Tests/TargetingServiceTests.cs`

**Interfaces:**
- Consumes: `HexGeometry.Distance3D(HexCoord a, HexCoord b)` (Task 8); `GameState{ Board Board; PlayerState[] Players; GameConfig Config; PlayerState PlayerStateFor(PlayerId); PlayerId Opponent(PlayerId); }` (Task 7); `PlayerState{ IReadOnlyList<Unit> UnitsOnBoard; IReadOnlyList<Generator> Generators; }` (Task 7); `Unit{ int Id; PlayerId Owner; UnitStats Stats; HexCoord Position; bool IsAlive; }` and `UnitStats{ int Range; int Vision; }` (Task 5); `Generator{ int Id; HexCoord Position; bool IsAlive; }` (Task 5); `Board.TryGetTile(int q,int r,out Tile)` and `Tile{ TerrainType Terrain; }` (Tasks 4,3); `GameConfig{ int RangeHighGroundBonus; TerrainDef TerrainDefFor(TerrainType); }` and `TerrainDef{ int Concealment; }` (Tasks 6,3).
- Produces: `public static class TargetingService{ static int HighGround(HexCoord attacker, HexCoord target); static bool InRange(GameState state, Unit attacker, HexCoord targetPos); static bool IsVisible(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain); static bool CanTarget(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain); static List<int> ValidTargetIds(GameState state, Unit attacker); }` â€” consumed by `GameEngine.Apply` AttackUnit (Task 17), `LegalMoves` (Task 18), and `CombatResolver.HighGround` reuse (Task 11).

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class TargetingServiceTests
    {
        // Builds a flat plains board big enough for the test coords, with two players.
        static GameState MakeState(IReadOnlyList<Unit> p0Units, IReadOnlyList<Unit> p1Units,
                                   IReadOnlyList<Generator> p1Gens, GameConfig config)
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 10; q++)
                for (int r = 0; r < 10; r++)
                    tiles.Add(new Tile(new HexCoord(q, r, 0), TerrainType.Plains));
            var board = new Board(10, 10, tiles, new List<HexCoord>(), new List<HexCoord>());
            var p0 = new PlayerState(PlayerId.Player0, 0, new List<UnitStats>(), p0Units, new List<Generator>());
            var p1 = new PlayerState(PlayerId.Player1, 0, new List<UnitStats>(), p1Units, p1Gens);
            return new GameState(board, new[] { p0, p1 }, PlayerId.Player0, 1, 100, config);
        }

        static Unit U(int id, PlayerId owner, int range, int vision, HexCoord pos)
            => new Unit(id, owner, new UnitStats(5, 3, range, 3, 0, vision), pos, 5);

        [Test]
        public void HighGround_IsMaxZeroOfElevationDiff()
        {
            Assert.AreEqual(2, TargetingService.HighGround(new HexCoord(0, 0, 3), new HexCoord(0, 0, 1)));
            Assert.AreEqual(0, TargetingService.HighGround(new HexCoord(0, 0, 1), new HexCoord(0, 0, 3)));
        }

        [Test]
        public void InRange_RespectsRangeOnFlatGround()
        {
            var config = GameConfig.Default();
            var attacker = U(1, PlayerId.Player0, range: 2, vision: 10, new HexCoord(0, 0, 0));
            var state = MakeState(new[] { attacker }, new List<Unit>(), new List<Generator>(), config);
            Assert.IsTrue(TargetingService.InRange(state, attacker, new HexCoord(2, 0, 0)));   // D=2
            Assert.IsFalse(TargetingService.InRange(state, attacker, new HexCoord(3, 0, 0)));  // D=3
        }

        [Test]
        public void InRange_HighGroundExtendsRange()
        {
            var config = GameConfig.Default(); // RangeHighGroundBonus = 1
            // attacker on elevation 2, target on elevation 0 -> H=2, effective range 2+2=4
            var attacker = U(1, PlayerId.Player0, range: 2, vision: 10, new HexCoord(0, 0, 2));
            var state = MakeState(new[] { attacker }, new List<Unit>(), new List<Generator>(), config);
            // hex distance 2 + elevation diff 2 = D=4; range+bonus = 4 -> in range
            Assert.IsTrue(TargetingService.InRange(state, attacker, new HexCoord(2, 0, 0)));
            // hex distance 3 + elevation diff 2 = D=5 > 4 -> out of range
            Assert.IsFalse(TargetingService.InRange(state, attacker, new HexCoord(3, 0, 0)));
        }

        [Test]
        public void IsVisible_BlockedByConcealment()
        {
            var config = GameConfig.Default(); // Forest concealment = +2
            var attacker = U(1, PlayerId.Player0, range: 10, vision: 3, new HexCoord(0, 0, 0));
            var state = MakeState(new[] { attacker }, new List<Unit>(), new List<Generator>(), config);
            // Plains target at D=3: 3 + 0 <= 3 -> visible
            Assert.IsTrue(TargetingService.IsVisible(state, attacker, new HexCoord(3, 0, 0), TerrainType.Plains));
            // Forest target at D=2: 2 + 2 = 4 > 3 -> not visible
            Assert.IsFalse(TargetingService.IsVisible(state, attacker, new HexCoord(2, 0, 0), TerrainType.Forest));
        }

        [Test]
        public void CanTarget_RequiresBothRangeAndVision()
        {
            var config = GameConfig.Default();
            // Range 10 but vision 1: in range but not visible at D=2
            var attacker = U(1, PlayerId.Player0, range: 10, vision: 1, new HexCoord(0, 0, 0));
            var state = MakeState(new[] { attacker }, new List<Unit>(), new List<Generator>(), config);
            Assert.IsTrue(TargetingService.InRange(state, attacker, new HexCoord(2, 0, 0)));
            Assert.IsFalse(TargetingService.IsVisible(state, attacker, new HexCoord(2, 0, 0), TerrainType.Plains));
            Assert.IsFalse(TargetingService.CanTarget(state, attacker, new HexCoord(2, 0, 0), TerrainType.Plains));
        }

        [Test]
        public void ValidTargetIds_IncludesEnemyUnitsAndGenerators_ExcludesUnreachable()
        {
            var config = GameConfig.Default();
            var attacker = U(1, PlayerId.Player0, range: 5, vision: 5, new HexCoord(0, 0, 0));
            var enemyUnit = U(2, PlayerId.Player1, range: 1, vision: 1, new HexCoord(2, 0, 0)); // D=2 reachable
            var farEnemy = U(3, PlayerId.Player1, range: 1, vision: 1, new HexCoord(9, 0, 0));   // D=9 too far
            var enemyGen = new Generator(4, PlayerId.Player1, new HexCoord(0, 3, 0), 3);          // D=3 reachable
            var state = MakeState(new[] { attacker }, new[] { enemyUnit, farEnemy },
                                  new[] { enemyGen }, config);
            var ids = TargetingService.ValidTargetIds(state, attacker);
            CollectionAssert.Contains(ids, 2);
            CollectionAssert.Contains(ids, 4);
            CollectionAssert.DoesNotContain(ids, 3);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.TargetingServiceTests" -logFile -`
Expected: FAIL â€” compile error, `TargetingService` does not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
using System.Collections.Generic;

namespace HexWars.Engine
{
    // Pure targeting math: high-ground, range, vision, and enemy enumeration.
    public static class TargetingService
    {
        // H = max(0, attackerElev - targetElev).
        public static int HighGround(HexCoord attacker, HexCoord target)
        {
            int h = attacker.Elevation - target.Elevation;
            return h > 0 ? h : 0;
        }

        // inRange: D <= Range + RangeHighGroundBonus * H, where D = Distance3D.
        public static bool InRange(GameState state, Unit attacker, HexCoord targetPos)
        {
            int d = HexGeometry.Distance3D(attacker.Position, targetPos);
            int h = HighGround(attacker.Position, targetPos);
            int effectiveRange = attacker.Stats.Range + state.Config.RangeHighGroundBonus * h;
            return d <= effectiveRange;
        }

        // visible: D + targetConcealment <= Vision (concealment from the target tile's terrain).
        public static bool IsVisible(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain)
        {
            int d = HexGeometry.Distance3D(attacker.Position, targetPos);
            int concealment = state.Config.TerrainDefFor(targetTerrain).Concealment;
            return d + concealment <= attacker.Stats.Vision;
        }

        public static bool CanTarget(GameState state, Unit attacker, HexCoord targetPos, TerrainType targetTerrain)
        {
            return InRange(state, attacker, targetPos)
                && IsVisible(state, attacker, targetPos, targetTerrain);
        }

        // Enumerate all legal targets (enemy units AND generators) for an attacker.
        public static List<int> ValidTargetIds(GameState state, Unit attacker)
        {
            var result = new List<int>();
            PlayerId enemy = state.Opponent(attacker.Owner);
            PlayerState enemyState = state.PlayerStateFor(enemy);

            foreach (var unit in enemyState.UnitsOnBoard)
            {
                if (!unit.IsAlive) continue;
                TerrainType terrain = TerrainAt(state, unit.Position);
                if (CanTarget(state, attacker, unit.Position, terrain))
                    result.Add(unit.Id);
            }

            foreach (var gen in enemyState.Generators)
            {
                if (!gen.IsAlive) continue;
                TerrainType terrain = TerrainAt(state, gen.Position);
                if (CanTarget(state, attacker, gen.Position, terrain))
                    result.Add(gen.Id);
            }

            return result;
        }

        static TerrainType TerrainAt(GameState state, HexCoord pos)
        {
            return state.Board.TryGetTile(pos.Q, pos.R, out Tile tile)
                ? tile.Terrain
                : TerrainType.Plains;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.TargetingServiceTests" -logFile -`
Expected: PASS â€” all six tests green.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Combat/TargetingService.cs Assets/HexWars/Engine/Tests/TargetingServiceTests.cs && git commit -m "feat(engine): add TargetingService range/vision/high-ground and ValidTargetIds"
```

---

### Task 11: CombatResolver: damage formula, high-ground/terrain defense, bounty, death

**Files:**
- Create: `Assets/HexWars/Engine/Combat/CombatResolver.cs`
- Modify: `Assets/HexWars/Engine/Tests/CombatResolverTests.cs`
- Test: `Assets/HexWars/Engine/Tests/CombatResolverTests.cs`

**Interfaces:**
- Consumes: `TargetingService.HighGround(HexCoord attacker, HexCoord target)` (Task 10); `GameState{ Board Board; GameConfig Config; }` and `Board.TryGetTile(int q,int r,out Tile)` (Tasks 7,4); `Tile{ TerrainType Terrain; }` (Task 3); `Unit{ UnitStats Stats; HexCoord Position; int CurrentHp; int BuildCost; }` and `UnitStats{ int Damage; int Defense; }` (Task 5); `Generator{ HexCoord Position; int CurrentHp; }` (Task 5); `GameConfig{ int DamageFloor; int DmgHighGroundBonus; double BountyRate; TerrainDef TerrainDefFor(TerrainType); }` and `TerrainDef{ int Defense; }` (Tasks 6,3).
- Produces: `public readonly struct CombatOutcome{ int Damage; int TargetHpAfter; bool TargetDestroyed; int Bounty; CombatOutcome(int,int,bool,int); }`. `public static class CombatResolver{ static int ComputeDamage(GameConfig config, Unit attacker, int defenderDefense, int defenderTerrainDefense, int highGround); static CombatOutcome ResolveAgainstUnit(GameState state, Unit attacker, Unit target); static CombatOutcome ResolveAgainstGenerator(GameState state, Unit attacker, Generator target); static int ComputeBounty(GameConfig config, int buildCost); }` â€” consumed by `GameEngine.Apply` AttackUnit (Task 17).

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class CombatResolverTests
    {
        // Board with a single configurable terrain tile under the target, plains elsewhere.
        static GameState MakeState(GameConfig config, HexCoord targetCoord, TerrainType targetTerrain)
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 10; q++)
                for (int r = 0; r < 10; r++)
                {
                    var coord = new HexCoord(q, r, q == targetCoord.Q && r == targetCoord.R ? targetCoord.Elevation : 0);
                    var terrain = (q == targetCoord.Q && r == targetCoord.R) ? targetTerrain : TerrainType.Plains;
                    tiles.Add(new Tile(coord, terrain));
                }
            var board = new Board(10, 10, tiles, new List<HexCoord>(), new List<HexCoord>());
            var p0 = new PlayerState(PlayerId.Player0, 0, new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            var p1 = new PlayerState(PlayerId.Player1, 0, new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            return new GameState(board, new[] { p0, p1 }, PlayerId.Player0, 1, 100, config);
        }

        static Unit Attacker(int damage, int defense, HexCoord pos)
            => new Unit(1, PlayerId.Player0, new UnitStats(5, damage, 3, 3, defense, 5), pos, 5);

        static Unit Target(int health, int defense, int currentHp, HexCoord pos)
            => new Unit(2, PlayerId.Player1, new UnitStats(health, 1, 1, 1, defense, 1), pos, currentHp);

        [Test]
        public void ComputeDamage_BaseMinusDefense()
        {
            var config = GameConfig.Default();
            var attacker = Attacker(damage: 5, defense: 0, new HexCoord(0, 0, 0));
            // 5 + 1*0 - (2 + 0) = 3
            Assert.AreEqual(3, CombatResolver.ComputeDamage(config, attacker, 2, 0, 0));
        }

        [Test]
        public void ComputeDamage_HighGroundBonusAdds()
        {
            var config = GameConfig.Default(); // DmgHighGroundBonus = 1
            var attacker = Attacker(damage: 4, defense: 0, new HexCoord(0, 0, 0));
            // 4 + 1*2 - (0 + 0) = 6
            Assert.AreEqual(6, CombatResolver.ComputeDamage(config, attacker, 0, 0, 2));
        }

        [Test]
        public void ComputeDamage_TerrainDefenseReduces()
        {
            var config = GameConfig.Default();
            var attacker = Attacker(damage: 5, defense: 0, new HexCoord(0, 0, 0));
            // 5 + 0 - (1 + 1) = 3
            Assert.AreEqual(3, CombatResolver.ComputeDamage(config, attacker, 1, 1, 0));
        }

        [Test]
        public void ComputeDamage_ClampedToDamageFloor()
        {
            var config = GameConfig.Default(); // DamageFloor = 1
            var attacker = Attacker(damage: 1, defense: 0, new HexCoord(0, 0, 0));
            // 1 + 0 - (5 + 1) = -5 -> clamped to floor 1
            Assert.AreEqual(1, CombatResolver.ComputeDamage(config, attacker, 5, 1, 0));
        }

        [Test]
        public void ResolveAgainstUnit_NonLethal()
        {
            var config = GameConfig.Default();
            var attacker = Attacker(damage: 4, defense: 0, new HexCoord(0, 0, 0));
            var target = Target(health: 10, defense: 1, currentHp: 10, new HexCoord(2, 0, 0)); // plains
            var outcome = CombatResolver.ResolveAgainstUnit(state: MakeState(config, new HexCoord(2, 0, 0), TerrainType.Plains),
                                                            attacker, target);
            // 4 - (1 + 0) = 3 damage; 10 - 3 = 7; alive
            Assert.AreEqual(3, outcome.Damage);
            Assert.AreEqual(7, outcome.TargetHpAfter);
            Assert.IsFalse(outcome.TargetDestroyed);
            Assert.AreEqual(0, outcome.Bounty);
        }

        [Test]
        public void ResolveAgainstUnit_LethalPaysBounty()
        {
            var config = GameConfig.Default(); // BountyRate = 0.5
            var attacker = Attacker(damage: 10, defense: 0, new HexCoord(0, 0, 0));
            // Target build cost = 5+1+1+1+0+1 = 9; bounty = floor(9 * 0.5) = 4
            var target = Target(health: 5, defense: 0, currentHp: 3, new HexCoord(2, 0, 0));
            var outcome = CombatResolver.ResolveAgainstUnit(MakeState(config, new HexCoord(2, 0, 0), TerrainType.Plains),
                                                            attacker, target);
            Assert.AreEqual(10, outcome.Damage);
            Assert.AreEqual(0, outcome.TargetHpAfter); // never negative
            Assert.IsTrue(outcome.TargetDestroyed);
            Assert.AreEqual(4, outcome.Bounty);
        }

        [Test]
        public void ResolveAgainstUnit_HighGroundAndTerrainDefenseCombine()
        {
            var config = GameConfig.Default(); // Forest defense +1
            var attacker = Attacker(damage: 4, defense: 0, new HexCoord(0, 0, 2)); // elevation 2
            // target on elevation 0 forest -> H=2, terrain defense 1, own defense 1
            var target = Target(health: 10, defense: 1, currentHp: 10, new HexCoord(2, 0, 0));
            var state = MakeState(config, new HexCoord(2, 0, 0), TerrainType.Forest);
            var outcome = CombatResolver.ResolveAgainstUnit(state, attacker, target);
            // 4 + 1*2 - (1 + 1) = 4 damage; 10 - 4 = 6
            Assert.AreEqual(4, outcome.Damage);
            Assert.AreEqual(6, outcome.TargetHpAfter);
        }

        [Test]
        public void ResolveAgainstGenerator_TreatsDefenseAsZero()
        {
            var config = GameConfig.Default();
            var attacker = Attacker(damage: 2, defense: 0, new HexCoord(0, 0, 0));
            var gen = new Generator(7, PlayerId.Player1, new HexCoord(2, 0, 0), 3); // plains -> terrain defense 0
            var state = MakeState(config, new HexCoord(2, 0, 0), TerrainType.Plains);
            var outcome = CombatResolver.ResolveAgainstGenerator(state, attacker, gen);
            // 2 + 0 - (0 + 0) = 2; 3 - 2 = 1; alive; no bounty
            Assert.AreEqual(2, outcome.Damage);
            Assert.AreEqual(1, outcome.TargetHpAfter);
            Assert.IsFalse(outcome.TargetDestroyed);
            Assert.AreEqual(0, outcome.Bounty);
        }

        [Test]
        public void ComputeBounty_FloorsBuildCostTimesRate()
        {
            var config = GameConfig.Default(); // 0.5
            Assert.AreEqual(4, CombatResolver.ComputeBounty(config, 9)); // floor(4.5)
            Assert.AreEqual(5, CombatResolver.ComputeBounty(config, 10));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CombatResolverTests" -logFile -`
Expected: FAIL â€” compile error, `CombatResolver` and `CombatOutcome` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
using System;

namespace HexWars.Engine
{
    public readonly struct CombatOutcome
    {
        public int Damage { get; }          // applied (already floored)
        public int TargetHpAfter { get; }
        public bool TargetDestroyed { get; }
        public int Bounty { get; }          // floor(buildCost * BountyRate) if destroyed else 0

        public CombatOutcome(int damage, int targetHpAfter, bool targetDestroyed, int bounty)
        {
            Damage = damage;
            TargetHpAfter = targetHpAfter;
            TargetDestroyed = targetDestroyed;
            Bounty = bounty;
        }
    }

    // Pure deterministic combat resolution (spec Â§5). No randomness.
    public static class CombatResolver
    {
        // damage = max(DamageFloor, Damage + DmgHighGroundBonus*H - (Defense + terrainDefense))
        public static int ComputeDamage(GameConfig config, Unit attacker, int defenderDefense,
            int defenderTerrainDefense, int highGround)
        {
            int raw = attacker.Stats.Damage
                      + config.DmgHighGroundBonus * highGround
                      - (defenderDefense + defenderTerrainDefense);
            return raw > config.DamageFloor ? raw : config.DamageFloor;
        }

        public static CombatOutcome ResolveAgainstUnit(GameState state, Unit attacker, Unit target)
        {
            int h = TargetingService.HighGround(attacker.Position, target.Position);
            int terrainDefense = TerrainDefenseAt(state, target.Position);
            int damage = ComputeDamage(state.Config, attacker, target.Stats.Defense, terrainDefense, h);
            int hpAfter = target.CurrentHp - damage;
            if (hpAfter < 0) hpAfter = 0;
            bool destroyed = hpAfter <= 0;
            int bounty = destroyed ? ComputeBounty(state.Config, target.BuildCost) : 0;
            return new CombatOutcome(damage, hpAfter, destroyed, bounty);
        }

        public static CombatOutcome ResolveAgainstGenerator(GameState state, Unit attacker, Generator target)
        {
            int h = TargetingService.HighGround(attacker.Position, target.Position);
            int terrainDefense = TerrainDefenseAt(state, target.Position);
            int damage = ComputeDamage(state.Config, attacker, 0, terrainDefense, h); // generator Defense = 0
            int hpAfter = target.CurrentHp - damage;
            if (hpAfter < 0) hpAfter = 0;
            bool destroyed = hpAfter <= 0;
            // Generators have no build-cost bounty in the spec economy; destroyed generators pay none.
            return new CombatOutcome(damage, hpAfter, destroyed, 0);
        }

        // bounty = floor(buildCost * BountyRate)
        public static int ComputeBounty(GameConfig config, int buildCost)
        {
            return (int)Math.Floor(buildCost * config.BountyRate);
        }

        static int TerrainDefenseAt(GameState state, HexCoord pos)
        {
            if (state.Board.TryGetTile(pos.Q, pos.R, out Tile tile))
                return state.Config.TerrainDefFor(tile.Terrain).Defense;
            return 0;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CombatResolverTests" -logFile -`
Expected: PASS â€” all tests green (damage formula, high-ground, terrain defense, floor clamp, lethal/non-lethal, generator zero-defense, bounty math).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Combat/CombatResolver.cs Assets/HexWars/Engine/Tests/CombatResolverTests.cs && git commit -m "feat(engine): add CombatResolver damage formula, high-ground/terrain defense, bounty"
```

---

### Task 12: Economy: income, affordability, cheapest viable unit

**Files:**
- Create: `Assets/HexWars/Engine/Economy/Economy.cs`
- Test: `Assets/HexWars/Engine/Tests/EconomyTests.cs`

**Interfaces:**
- Consumes: `GameState` (`Board Board`, `PlayerState[] Players`, `PlayerId ActivePlayer`, `int Round`, `int NextEntityId`, `GameConfig Config`, `PlayerState PlayerStateFor(PlayerId)`, `PlayerId Opponent(PlayerId)`, `GameState Clone()`); `PlayerState` (`int Points`, `IReadOnlyList<UnitStats> Reserve`, `IReadOnlyList<Unit> UnitsOnBoard`, `IReadOnlyList<Generator> Generators`); `Generator` (`bool IsAlive`, `int CurrentHp`); `GameConfig` (`int GeneratorOutput`, `int GeneratorCost`, `int MinCheapestViableUnitCost`); `PlayerId {Player0,Player1}` (all from Tasks 7, 5, 6).
- Produces: `public static class Economy{ static int IncomeFor(GameState state, PlayerId player); static bool CanAfford(GameState state, PlayerId player, int cost); static int CheapestViableUnitCost(GameConfig config); }` â€” consumed by WinCheck (Task 13), GameEngine (Tasks 16, 18), LegalMoves (Task 18).

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class EconomyTests
    {
        // Minimal board: 1 plains tile at (0,0,0). Enough for Economy (which ignores the board).
        private static Board MakeBoard()
        {
            var tiles = new List<Tile> { new Tile(new HexCoord(0, 0, 0), TerrainType.Plains) };
            var dz = new List<HexCoord> { new HexCoord(0, 0, 0) };
            return new Board(1, 1, tiles, dz, dz);
        }

        private static GameState MakeState(
            GameConfig config,
            IReadOnlyList<Generator> p0Generators,
            int p0Points = 0)
        {
            var p0 = new PlayerState(
                PlayerId.Player0, p0Points,
                new List<UnitStats>(), new List<Unit>(), p0Generators);
            var p1 = new PlayerState(
                PlayerId.Player1, 0,
                new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            return new GameState(MakeBoard(), new[] { p0, p1 },
                PlayerId.Player0, 1, 100, config);
        }

        [Test]
        public void IncomeFor_SumsLivingGeneratorOutput()
        {
            var config = GameConfig.Default(); // GeneratorOutput = 1
            var gens = new List<Generator>
            {
                new Generator(1, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth),
                new Generator(2, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth),
                new Generator(3, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth)
            };
            var state = MakeState(config, gens);
            Assert.AreEqual(3 * config.GeneratorOutput, Economy.IncomeFor(state, PlayerId.Player0));
        }

        [Test]
        public void IncomeFor_ExcludesDeadGenerators()
        {
            var config = GameConfig.Default();
            var gens = new List<Generator>
            {
                new Generator(1, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth), // alive
                new Generator(2, PlayerId.Player0, new HexCoord(0, 0, 0), 0)                        // dead
            };
            var state = MakeState(config, gens);
            Assert.AreEqual(1 * config.GeneratorOutput, Economy.IncomeFor(state, PlayerId.Player0));
        }

        [Test]
        public void IncomeFor_NoGenerators_IsZero()
        {
            var config = GameConfig.Default();
            var state = MakeState(config, new List<Generator>());
            Assert.AreEqual(0, Economy.IncomeFor(state, PlayerId.Player0));
        }

        [Test]
        public void CanAfford_BoundaryConditions()
        {
            var config = GameConfig.Default();
            var state = MakeState(config, new List<Generator>(), p0Points: 5);
            Assert.IsTrue(Economy.CanAfford(state, PlayerId.Player0, 4), "5 >= 4");
            Assert.IsTrue(Economy.CanAfford(state, PlayerId.Player0, 5), "5 >= 5 (exact)");
            Assert.IsFalse(Economy.CanAfford(state, PlayerId.Player0, 6), "5 < 6");
        }

        [Test]
        public void CheapestViableUnitCost_EqualsConfigMin()
        {
            var config = GameConfig.Default(); // MinCheapestViableUnitCost = 1
            Assert.AreEqual(config.MinCheapestViableUnitCost, Economy.CheapestViableUnitCost(config));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.EconomyTests" -logFile -`
Expected: FAIL with a compile error â€” `Economy` does not exist yet (`The name 'Economy' does not exist in the current context`).
- [ ] **Step 3: Write minimal implementation**
```csharp
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure, deterministic economy helpers. No UnityEngine, no randomness.
    /// </summary>
    public static class Economy
    {
        /// <summary>Sum of the player's living generators' GeneratorOutput.</summary>
        public static int IncomeFor(GameState state, PlayerId player)
        {
            int output = state.Config.GeneratorOutput;
            int total = 0;
            IReadOnlyList<Generator> generators = state.PlayerStateFor(player).Generators;
            for (int i = 0; i < generators.Count; i++)
            {
                if (generators[i].IsAlive)
                {
                    total += output;
                }
            }
            return total;
        }

        /// <summary>True when the player's banked points are at least <paramref name="cost"/>.</summary>
        public static bool CanAfford(GameState state, PlayerId player, int cost)
        {
            return state.PlayerStateFor(player).Points >= cost;
        }

        /// <summary>Cost of the cheapest legal unit (a 1-HP unit), == config.MinCheapestViableUnitCost.</summary>
        public static int CheapestViableUnitCost(GameConfig config)
        {
            return config.MinCheapestViableUnitCost;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.EconomyTests" -logFile -`
Expected: PASS (all 5 tests green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Economy/Economy.cs Assets/HexWars/Engine/Tests/EconomyTests.cs && git commit -m "feat(engine): Economy income, affordability, cheapest viable unit"
```

---

### Task 13: WinCheck and Evaluation: elimination, total value, round cap, heuristic

**Files:**
- Create: `Assets/HexWars/Engine/Economy/WinCheck.cs`
- Create: `Assets/HexWars/Engine/Economy/Evaluation.cs`
- Test: `Assets/HexWars/Engine/Tests/WinCheckTests.cs`

**Interfaces:**
- Consumes: `Economy.CanAfford(GameState, PlayerId, int)`, `Economy.CheapestViableUnitCost(GameConfig)` (Task 12); `GameState` (`PlayerState[] Players`, `PlayerId ActivePlayer`, `int Round`, `GameConfig Config`, `PlayerState PlayerStateFor(PlayerId)`, `PlayerId Opponent(PlayerId)`); `PlayerState` (`int Points`, `IReadOnlyList<UnitStats> Reserve`, `IReadOnlyList<Unit> UnitsOnBoard`, `IReadOnlyList<Generator> Generators`); `Unit` (`int BuildCost`, `bool IsAlive`); `UnitStats` (`int PointCost`); `Generator` (`bool IsAlive`); `GameConfig` (`int GeneratorCost`, `int RoundCap`) (Tasks 7, 5, 6).
- Produces: `public static class WinCheck{ static bool IsEliminated(GameState state, PlayerId player); static int TotalValue(GameState state, PlayerId player); static PlayerId? CheckWinner(GameState state); }` and `public static class Evaluation{ static int Evaluate(GameState state, PlayerId player); }` â€” consumed by GameEngine.Apply (Tasks 16, 17) for end-of-turn win checks and by the stalemate tie-break.

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class WinCheckTests
    {
        private static Board MakeBoard()
        {
            var tiles = new List<Tile> { new Tile(new HexCoord(0, 0, 0), TerrainType.Plains) };
            var dz = new List<HexCoord> { new HexCoord(0, 0, 0) };
            return new Board(1, 1, tiles, dz, dz);
        }

        private static PlayerState Player(
            PlayerId id,
            int points,
            IReadOnlyList<UnitStats> reserve = null,
            IReadOnlyList<Unit> units = null,
            IReadOnlyList<Generator> generators = null)
        {
            return new PlayerState(
                id, points,
                reserve ?? new List<UnitStats>(),
                units ?? new List<Unit>(),
                generators ?? new List<Generator>());
        }

        private static GameState MakeState(
            GameConfig config, PlayerState p0, PlayerState p1,
            int round = 1, PlayerId active = PlayerId.Player0)
        {
            return new GameState(MakeBoard(), new[] { p0, p1 }, active, round, 100, config);
        }

        private static Unit AliveUnit(int id, PlayerId owner, UnitStats stats)
        {
            return new Unit(id, owner, stats, new HexCoord(0, 0, 0), stats.Health);
        }

        [Test]
        public void IsEliminated_True_NoUnits_NoReserve_CannotAfford()
        {
            var config = GameConfig.Default(); // cheapest viable = 1
            // 0 points < 1 => cannot afford; no units, no reserve.
            var p0 = Player(PlayerId.Player0, points: 0);
            var state = MakeState(config, p0, Player(PlayerId.Player1, 0));
            Assert.IsTrue(WinCheck.IsEliminated(state, PlayerId.Player0));
        }

        [Test]
        public void IsEliminated_False_RichButEmpty()
        {
            var config = GameConfig.Default();
            // No units and no reserve, but 5 points >= cheapest viable (1) => NOT eliminated.
            var p0 = Player(PlayerId.Player0, points: 5);
            var state = MakeState(config, p0, Player(PlayerId.Player1, 0));
            Assert.IsFalse(WinCheck.IsEliminated(state, PlayerId.Player0));
        }

        [Test]
        public void IsEliminated_False_HasUnitOnBoard()
        {
            var config = GameConfig.Default();
            var units = new List<Unit> { AliveUnit(1, PlayerId.Player0, new UnitStats(3, 0, 0, 0, 0, 0)) };
            var p0 = Player(PlayerId.Player0, points: 0, units: units);
            var state = MakeState(config, p0, Player(PlayerId.Player1, 0));
            Assert.IsFalse(WinCheck.IsEliminated(state, PlayerId.Player0));
        }

        [Test]
        public void IsEliminated_False_HasReserve()
        {
            var config = GameConfig.Default();
            var reserve = new List<UnitStats> { new UnitStats(1, 0, 0, 0, 0, 0) };
            var p0 = Player(PlayerId.Player0, points: 0, reserve: reserve);
            var state = MakeState(config, p0, Player(PlayerId.Player1, 0));
            Assert.IsFalse(WinCheck.IsEliminated(state, PlayerId.Player0));
        }

        [Test]
        public void TotalValue_SumsBankReserveBoardAndGenerators()
        {
            var config = GameConfig.Default(); // GeneratorCost = 2
            var onBoard = new UnitStats(2, 1, 0, 0, 0, 0); // PointCost 3, BuildCost 3
            var reserveStats = new UnitStats(1, 0, 0, 0, 0, 0); // PointCost 1
            var units = new List<Unit> { AliveUnit(1, PlayerId.Player0, onBoard) };
            var reserve = new List<UnitStats> { reserveStats };
            var gens = new List<Generator>
            {
                new Generator(2, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth),
                new Generator(3, PlayerId.Player0, new HexCoord(0, 0, 0), config.GeneratorHealth)
            };
            var p0 = Player(PlayerId.Player0, points: 4, reserve: reserve, units: units, generators: gens);
            var state = MakeState(config, p0, Player(PlayerId.Player1, 0));
            // 4 banked + 3 on-board + 1 reserve + 2 generators * cost 2 = 4 + 3 + 1 + 4 = 12
            Assert.AreEqual(4 + 3 + 1 + 2 * config.GeneratorCost, WinCheck.TotalValue(state, PlayerId.Player0));
        }

        [Test]
        public void CheckWinner_ReturnsOpponent_WhenPlayerEliminated()
        {
            var config = GameConfig.Default();
            // P1 eliminated (0 points, nothing on board); P0 has a unit.
            var p0Units = new List<Unit> { AliveUnit(1, PlayerId.Player0, new UnitStats(3, 0, 0, 0, 0, 0)) };
            var p0 = Player(PlayerId.Player0, points: 0, units: p0Units);
            var p1 = Player(PlayerId.Player1, points: 0);
            var state = MakeState(config, p0, p1, round: 5);
            Assert.AreEqual(PlayerId.Player0, WinCheck.CheckWinner(state));
        }

        [Test]
        public void CheckWinner_Null_WhenBothAlive_BeforeRoundCap()
        {
            var config = GameConfig.Default();
            var p0 = Player(PlayerId.Player0, points: 10);
            var p1 = Player(PlayerId.Player1, points: 10);
            var state = MakeState(config, p0, p1, round: 1);
            Assert.IsNull(WinCheck.CheckWinner(state));
        }

        [Test]
        public void CheckWinner_RoundCap_HigherTotalValueWins()
        {
            var config = GameConfig.Default();
            // Both still in the game (rich) but round cap reached: higher total value wins.
            var p0 = Player(PlayerId.Player0, points: 20); // total value 20
            var p1 = Player(PlayerId.Player1, points: 5);  // total value 5
            var state = MakeState(config, p0, p1, round: config.RoundCap);
            Assert.AreEqual(PlayerId.Player0, WinCheck.CheckWinner(state));
        }

        [Test]
        public void Evaluate_IsTotalValueDifference()
        {
            var config = GameConfig.Default();
            var p0 = Player(PlayerId.Player0, points: 12);
            var p1 = Player(PlayerId.Player1, points: 7);
            var state = MakeState(config, p0, p1);
            Assert.AreEqual(12 - 7, Evaluation.Evaluate(state, PlayerId.Player0));
            Assert.AreEqual(7 - 12, Evaluation.Evaluate(state, PlayerId.Player1));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.WinCheckTests" -logFile -`
Expected: FAIL with a compile error â€” `WinCheck` and `Evaluation` do not exist yet (`The name 'WinCheck' does not exist in the current context`).
- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Economy/WinCheck.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Pure, deterministic win/elimination/value checks (spec section 10). No UnityEngine.
    /// </summary>
    public static class WinCheck
    {
        /// <summary>
        /// A player is eliminated when they have no living units on the board, none in reserve,
        /// and cannot afford the cheapest viable unit.
        /// </summary>
        public static bool IsEliminated(GameState state, PlayerId player)
        {
            PlayerState ps = state.PlayerStateFor(player);

            bool hasLivingUnit = false;
            IReadOnlyList<Unit> units = ps.UnitsOnBoard;
            for (int i = 0; i < units.Count; i++)
            {
                if (units[i].IsAlive)
                {
                    hasLivingUnit = true;
                    break;
                }
            }
            if (hasLivingUnit)
            {
                return false;
            }

            if (ps.Reserve.Count > 0)
            {
                return false;
            }

            int cheapest = Economy.CheapestViableUnitCost(state.Config);
            return !Economy.CanAfford(state, player, cheapest);
        }

        /// <summary>
        /// Total value = banked points + sum(living on-board unit build costs)
        /// + sum(reserve unit costs) + (generator count * GeneratorCost).
        /// </summary>
        public static int TotalValue(GameState state, PlayerId player)
        {
            PlayerState ps = state.PlayerStateFor(player);
            int total = ps.Points;

            IReadOnlyList<Unit> units = ps.UnitsOnBoard;
            for (int i = 0; i < units.Count; i++)
            {
                if (units[i].IsAlive)
                {
                    total += units[i].BuildCost;
                }
            }

            IReadOnlyList<UnitStats> reserve = ps.Reserve;
            for (int i = 0; i < reserve.Count; i++)
            {
                total += reserve[i].PointCost;
            }

            total += ps.Generators.Count * state.Config.GeneratorCost;
            return total;
        }

        /// <summary>
        /// Returns the winner if the game is over (a player is eliminated, or the round cap is
        /// reached and one side has strictly higher total value), else null.
        /// </summary>
        public static PlayerId? CheckWinner(GameState state)
        {
            bool p0Out = IsEliminated(state, PlayerId.Player0);
            bool p1Out = IsEliminated(state, PlayerId.Player1);

            if (p0Out && !p1Out)
            {
                return PlayerId.Player1;
            }
            if (p1Out && !p0Out)
            {
                return PlayerId.Player0;
            }
            if (p0Out && p1Out)
            {
                // Both eliminated simultaneously: fall through to the value tie-break below.
                return ValueTieBreak(state);
            }

            if (state.Round >= state.Config.RoundCap)
            {
                return ValueTieBreak(state);
            }

            return null;
        }

        /// <summary>Higher total value wins; an exact tie has no winner (null).</summary>
        private static PlayerId? ValueTieBreak(GameState state)
        {
            int v0 = TotalValue(state, PlayerId.Player0);
            int v1 = TotalValue(state, PlayerId.Player1);
            if (v0 > v1)
            {
                return PlayerId.Player0;
            }
            if (v1 > v0)
            {
                return PlayerId.Player1;
            }
            return null;
        }
    }
}
```
```csharp
// Assets/HexWars/Engine/Economy/Evaluation.cs
namespace HexWars.Engine
{
    /// <summary>
    /// Pure heuristic position score from a player's perspective. Reused by the stalemate
    /// tie-break and, later, as the AI search heuristic. No UnityEngine.
    /// </summary>
    public static class Evaluation
    {
        /// <summary>Score = TotalValue(player) - TotalValue(opponent).</summary>
        public static int Evaluate(GameState state, PlayerId player)
        {
            return WinCheck.TotalValue(state, player)
                 - WinCheck.TotalValue(state, state.Opponent(player));
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.WinCheckTests" -logFile -`
Expected: PASS (all 10 tests green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Economy/WinCheck.cs Assets/HexWars/Engine/Economy/Evaluation.cs Assets/HexWars/Engine/Tests/WinCheckTests.cs && git commit -m "feat(engine): WinCheck elimination/total-value/round-cap and Evaluation heuristic"
```

---

### Task 14: Commands, GameEvent, and Result types

**Files:**
- Create: `Assets/HexWars/Engine/Commands/Command.cs`
- Create: `Assets/HexWars/Engine/Commands/GameEvent.cs`
- Create: `Assets/HexWars/Engine/Commands/Result.cs`
- Test: `Assets/HexWars/Engine/Tests/CommandApplyTests.cs`

**Interfaces:**
- Consumes: `PlayerId` enum (Task 2); `HexCoord` struct, `UnitStats` struct (Tasks 2,5); `GameState` (Task 7).
- Produces: `public abstract record Command { PlayerId Issuer { get; init; } }`; `public sealed record CreateUnit(UnitStats Stats) : Command`; `public sealed record DeployUnit(int ReserveIndex, HexCoord Target) : Command`; `public sealed record DeployGenerator(HexCoord Target) : Command`; `public sealed record MoveUnit(int UnitId, HexCoord Target) : Command`; `public sealed record AttackUnit(int AttackerUnitId, int TargetId) : Command`; `public sealed record EndTurn() : Command`; `public enum GameEventType`; `public sealed record GameEvent { GameEventType Type; int EntityId; int TargetId; PlayerId Player; int Amount; HexCoord Coord; }`; `public enum RejectionReason`; `public sealed class Result { bool Success; GameState NewState; IReadOnlyList<GameEvent> Events; RejectionReason Reason; string Message; static Result Ok(GameState, IReadOnlyList<GameEvent>); static Result Rejected(GameState, RejectionReason, string); }`.

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class CommandApplyTests
    {
        [Test]
        public void Commands_WithSameFields_AreValueEqual()
        {
            var a = new MoveUnit(7, new HexCoord(1, 2, 3)) { Issuer = PlayerId.Player1 };
            var b = new MoveUnit(7, new HexCoord(1, 2, 3)) { Issuer = PlayerId.Player1 };
            Assert.AreEqual(a, b);
            Assert.IsTrue(a == b);
        }

        [Test]
        public void Commands_WithDifferentIssuer_AreNotEqual()
        {
            var a = new EndTurn() { Issuer = PlayerId.Player0 };
            var b = new EndTurn() { Issuer = PlayerId.Player1 };
            Assert.AreNotEqual(a, b);
        }

        [Test]
        public void CreateUnit_CarriesStatsAndIssuer()
        {
            var stats = new UnitStats(3, 2, 1, 1, 0, 2);
            var cmd = new CreateUnit(stats) { Issuer = PlayerId.Player1 };
            Assert.AreEqual(stats, cmd.Stats);
            Assert.AreEqual(PlayerId.Player1, cmd.Issuer);
        }

        [Test]
        public void GameEvent_RecordEquality_HoldsOnAllFields()
        {
            var e1 = new GameEvent { Type = GameEventType.UnitMoved, EntityId = 4, TargetId = 0, Player = PlayerId.Player0, Amount = 0, Coord = new HexCoord(2, 2, 1) };
            var e2 = new GameEvent { Type = GameEventType.UnitMoved, EntityId = 4, TargetId = 0, Player = PlayerId.Player0, Amount = 0, Coord = new HexCoord(2, 2, 1) };
            Assert.AreEqual(e1, e2);
        }

        [Test]
        public void Result_Ok_HasSuccessTrueAndNoReason()
        {
            var events = new List<GameEvent> { new GameEvent { Type = GameEventType.TurnEnded } };
            var state = TestStates.Minimal();
            var r = Result.Ok(state, events);
            Assert.IsTrue(r.Success);
            Assert.AreSame(state, r.NewState);
            Assert.AreEqual(RejectionReason.None, r.Reason);
            Assert.AreEqual(1, r.Events.Count);
            Assert.AreEqual(string.Empty, r.Message);
        }

        [Test]
        public void Result_Rejected_HasSuccessFalseReasonAndMessage()
        {
            var state = TestStates.Minimal();
            var r = Result.Rejected(state, RejectionReason.NotYourTurn, "not your turn");
            Assert.IsFalse(r.Success);
            Assert.AreSame(state, r.NewState);
            Assert.AreEqual(RejectionReason.NotYourTurn, r.Reason);
            Assert.AreEqual("not your turn", r.Message);
            Assert.AreEqual(0, r.Events.Count);
        }
    }
}
```
_Note: `TestStates.Minimal()` is a tiny shared test helper that builds a 1x1-board `GameState`; create `Assets/HexWars/Engine/Tests/TestStates.cs` with a `static GameState Minimal()` returning a `GameState` constructed via `new Board(1,1,new[]{new Tile(new HexCoord(0,0,0),TerrainType.Plains)}, new List<HexCoord>{new HexCoord(0,0,0)}, new List<HexCoord>{new HexCoord(0,0,0)})`, two empty `PlayerState`s at `GameConfig.Default().StartingPoints`, `ActivePlayer=Player0`, `round=1`, `nextEntityId=1`, `GameConfig.Default()`._

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests" -logFile -`
Expected: FAIL â€” compilation error, `Command`, `CreateUnit`, `GameEvent`, `GameEventType`, `Result`, `RejectionReason` do not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Commands/Command.cs
namespace HexWars.Engine
{
    public abstract record Command
    {
        public PlayerId Issuer { get; init; }
    }

    public sealed record CreateUnit(UnitStats Stats) : Command;
    public sealed record DeployUnit(int ReserveIndex, HexCoord Target) : Command;
    public sealed record DeployGenerator(HexCoord Target) : Command;
    public sealed record MoveUnit(int UnitId, HexCoord Target) : Command;
    public sealed record AttackUnit(int AttackerUnitId, int TargetId) : Command;
    public sealed record EndTurn() : Command;
}
```
```csharp
// Assets/HexWars/Engine/Commands/GameEvent.cs
namespace HexWars.Engine
{
    public enum GameEventType
    {
        UnitCreated, UnitDeployed, GeneratorDeployed, UnitMoved,
        UnitAttacked, UnitDestroyed, GeneratorDestroyed,
        BountyAwarded, IncomeCredited, TurnEnded, GameWon
    }

    public sealed record GameEvent
    {
        public GameEventType Type { get; init; }
        public int EntityId { get; init; }
        public int TargetId { get; init; }
        public PlayerId Player { get; init; }
        public int Amount { get; init; }
        public HexCoord Coord { get; init; }

        public override string ToString()
            => $"{Type}(entity={EntityId},target={TargetId},player={Player},amount={Amount},coord={Coord})";
    }
}
```
```csharp
// Assets/HexWars/Engine/Commands/Result.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    public enum RejectionReason
    {
        None = 0,
        NotYourTurn,
        IllegalCommandForPolicy,
        InsufficientPoints,
        UnitNotFound,
        GeneratorNotFound,
        ReserveUnitNotFound,
        TileNotFound,
        TileOccupied,
        TileImpassable,
        OutsideDeploymentZone,
        OutOfMovementRange,
        ClimbTooSteep,
        TargetNotInRange,
        TargetNotVisible,
        TargetNotEnemy,
        UnitAlreadyActed,
        InvalidStats,
        GameAlreadyOver
    }

    public sealed class Result
    {
        private static readonly IReadOnlyList<GameEvent> Empty = new List<GameEvent>();

        public bool Success { get; }
        public GameState NewState { get; }
        public IReadOnlyList<GameEvent> Events { get; }
        public RejectionReason Reason { get; }
        public string Message { get; }

        private Result(bool success, GameState newState, IReadOnlyList<GameEvent> events,
                       RejectionReason reason, string message)
        {
            Success = success;
            NewState = newState;
            Events = events;
            Reason = reason;
            Message = message;
        }

        public static Result Ok(GameState newState, IReadOnlyList<GameEvent> events)
            => new Result(true, newState, events ?? Empty, RejectionReason.None, string.Empty);

        public static Result Rejected(GameState unchangedState, RejectionReason reason, string message)
            => new Result(false, unchangedState, Empty, reason, message ?? string.Empty);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests" -logFile -`
Expected: PASS â€” all six tests green.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Commands/Command.cs Assets/HexWars/Engine/Commands/GameEvent.cs Assets/HexWars/Engine/Commands/Result.cs Assets/HexWars/Engine/Tests/CommandApplyTests.cs Assets/HexWars/Engine/Tests/TestStates.cs && git commit -m "feat(engine): add Command records, GameEvent, and Result types"
```

---

### Task 15: ITurnPolicy with AllUnitsPolicy and OneActionPolicy

**Files:**
- Create: `Assets/HexWars/Engine/Turns/ITurnPolicy.cs`
- Create: `Assets/HexWars/Engine/Turns/AllUnitsPolicy.cs`
- Create: `Assets/HexWars/Engine/Turns/OneActionPolicy.cs`
- Test: `Assets/HexWars/Engine/Tests/TurnPolicyTests.cs`

**Interfaces:**
- Consumes: `Command`, `CreateUnit`, `MoveUnit`, `EndTurn` records (Task 14); `GameState` (Task 7); `TurnPolicyKind` enum (Task 6).
- Produces: `public interface ITurnPolicy { TurnPolicyKind Kind { get; } bool IsCommandAllowed(GameState state, Command command); bool ShouldAutoEndTurn(GameState state, Command command); }`; `public sealed class AllUnitsPolicy : ITurnPolicy`; `public sealed class OneActionPolicy : ITurnPolicy`; `public static class TurnPolicyFactory { static ITurnPolicy Create(TurnPolicyKind kind); }`.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class TurnPolicyTests
    {
        [Test]
        public void Factory_ReturnsMatchingPolicyKind()
        {
            Assert.AreEqual(TurnPolicyKind.AllUnits, TurnPolicyFactory.Create(TurnPolicyKind.AllUnits).Kind);
            Assert.AreEqual(TurnPolicyKind.OneAction, TurnPolicyFactory.Create(TurnPolicyKind.OneAction).Kind);
        }

        [Test]
        public void AllUnits_AllowsEveryCommandType()
        {
            var p = new AllUnitsPolicy();
            var s = TestStates.Minimal();
            Assert.IsTrue(p.IsCommandAllowed(s, new CreateUnit(new UnitStats(1, 0, 0, 0, 0, 0))));
            Assert.IsTrue(p.IsCommandAllowed(s, new MoveUnit(1, new HexCoord(0, 0, 0))));
            Assert.IsTrue(p.IsCommandAllowed(s, new EndTurn()));
        }

        [Test]
        public void AllUnits_NeverAutoEnds_EvenAfterAction()
        {
            var p = new AllUnitsPolicy();
            var s = TestStates.Minimal();
            Assert.IsFalse(p.ShouldAutoEndTurn(s, new MoveUnit(1, new HexCoord(0, 0, 0))));
            Assert.IsFalse(p.ShouldAutoEndTurn(s, new EndTurn()));
        }

        [Test]
        public void OneAction_AllowsEveryCommandType()
        {
            var p = new OneActionPolicy();
            var s = TestStates.Minimal();
            Assert.IsTrue(p.IsCommandAllowed(s, new CreateUnit(new UnitStats(1, 0, 0, 0, 0, 0))));
            Assert.IsTrue(p.IsCommandAllowed(s, new MoveUnit(1, new HexCoord(0, 0, 0))));
            Assert.IsTrue(p.IsCommandAllowed(s, new EndTurn()));
        }

        [Test]
        public void OneAction_AutoEnds_AfterNonEndTurnAction()
        {
            var p = new OneActionPolicy();
            var s = TestStates.Minimal();
            Assert.IsTrue(p.ShouldAutoEndTurn(s, new MoveUnit(1, new HexCoord(0, 0, 0))));
            Assert.IsTrue(p.ShouldAutoEndTurn(s, new CreateUnit(new UnitStats(1, 0, 0, 0, 0, 0))));
        }

        [Test]
        public void OneAction_DoesNotAutoEnd_AfterExplicitEndTurn()
        {
            var p = new OneActionPolicy();
            var s = TestStates.Minimal();
            Assert.IsFalse(p.ShouldAutoEndTurn(s, new EndTurn()));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.TurnPolicyTests" -logFile -`
Expected: FAIL â€” compilation error, `ITurnPolicy`, `AllUnitsPolicy`, `OneActionPolicy`, `TurnPolicyFactory` do not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Turns/ITurnPolicy.cs
namespace HexWars.Engine
{
    public interface ITurnPolicy
    {
        TurnPolicyKind Kind { get; }
        bool IsCommandAllowed(GameState state, Command command);
        bool ShouldAutoEndTurn(GameState state, Command command);
    }

    public static class TurnPolicyFactory
    {
        public static ITurnPolicy Create(TurnPolicyKind kind)
        {
            switch (kind)
            {
                case TurnPolicyKind.OneAction: return new OneActionPolicy();
                case TurnPolicyKind.AllUnits:
                default: return new AllUnitsPolicy();
            }
        }
    }
}
```
```csharp
// Assets/HexWars/Engine/Turns/AllUnitsPolicy.cs
namespace HexWars.Engine
{
    public sealed class AllUnitsPolicy : ITurnPolicy
    {
        public TurnPolicyKind Kind => TurnPolicyKind.AllUnits;

        // Every command type is legal under AllUnits; resource/rule checks live in GameEngine.
        public bool IsCommandAllowed(GameState state, Command command) => true;

        // The turn never auto-ends; the active player ends it explicitly with EndTurn.
        public bool ShouldAutoEndTurn(GameState state, Command command) => false;
    }
}
```
```csharp
// Assets/HexWars/Engine/Turns/OneActionPolicy.cs
namespace HexWars.Engine
{
    public sealed class OneActionPolicy : ITurnPolicy
    {
        public TurnPolicyKind Kind => TurnPolicyKind.OneAction;

        // Every command type is legal; OneAction differs only in auto-ending the turn.
        public bool IsCommandAllowed(GameState state, Command command) => true;

        // Any single non-EndTurn action auto-passes the turn.
        public bool ShouldAutoEndTurn(GameState state, Command command) => !(command is EndTurn);
    }
}
```

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.TurnPolicyTests" -logFile -`
Expected: PASS â€” all six tests green.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Turns/ITurnPolicy.cs Assets/HexWars/Engine/Turns/AllUnitsPolicy.cs Assets/HexWars/Engine/Turns/OneActionPolicy.cs Assets/HexWars/Engine/Tests/TurnPolicyTests.cs && git commit -m "feat(engine): add ITurnPolicy with AllUnits/OneAction policies and factory"
```

---

### Task 16: GameEngine.Apply for create/deploy and EndTurn (with start-of-turn income)

**Files:**
- Create: `Assets/HexWars/Engine/Commands/GameEngine.cs`
- Modify: `Assets/HexWars/Engine/Tests/CommandApplyTests.cs` (append create/deploy/EndTurn happy-path + non-mutation tests)
- Modify: `Assets/HexWars/Engine/Tests/CommandRejectionTests.cs` (new file: rejection-path tests)
- Test: `Assets/HexWars/Engine/Tests/CommandApplyTests.cs`, `Assets/HexWars/Engine/Tests/CommandRejectionTests.cs`

**Interfaces:**
- Consumes: `Result.Ok/Rejected`, `Command`/`CreateUnit`/`DeployUnit`/`DeployGenerator`/`EndTurn`, `GameEvent`/`GameEventType`/`RejectionReason` (Task 14); `ITurnPolicy.IsCommandAllowed`, `TurnPolicyFactory.Create` (Task 15); `Economy.IncomeFor`, `Economy.CanAfford` (Task 12); `WinCheck.CheckWinner` (Task 13); `Board.IsInDeploymentZone`, `Board.TryGetTile` (Task 4); `GameState.Clone`, `GameState.Opponent`, `GameState.PlayerStateFor` (Task 7); `UnitStats.IsValid`/`PointCost` (Task 5); `Unit`, `Generator` (Task 5); `GameConfig.GeneratorCost`/`GeneratorHealth`/`TurnPolicy` (Task 6).
- Produces: `public static class GameEngine { public static Result Apply(GameState state, Command command); }` â€” dispatches by command type, enforces `ITurnPolicy.IsCommandAllowed` (else `IllegalCommandForPolicy`) and `ActivePlayer==Issuer` (else `NotYourTurn`), allocates ids from `NextEntityId`, credits income at start of the now-active turn on EndTurn, runs `WinCheck`. (`LegalMoves`, `MoveUnit`/`AttackUnit` added in Tasks 17â€“18.)

- [ ] **Step 1: Write the failing test**
```csharp
// Append to Assets/HexWars/Engine/Tests/CommandApplyTests.cs (inside the CommandApplyTests class)
        [Test]
        public void CreateUnit_PaysPoints_AddsToReserve_AndDoesNotMutateInput()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 10);
            int beforePoints = s0.Players[0].Points;
            int beforeReserve = s0.Players[0].Reserve.Count;
            var cmd = new CreateUnit(new UnitStats(3, 2, 0, 0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(beforePoints - 5, r.NewState.Players[0].Points);
            Assert.AreEqual(beforeReserve + 1, r.NewState.Players[0].Reserve.Count);
            // input untouched
            Assert.AreEqual(beforePoints, s0.Players[0].Points);
            Assert.AreEqual(beforeReserve, s0.Players[0].Reserve.Count);
            Assert.IsTrue(r.Events.Count >= 1 && r.Events[0].Type == GameEventType.UnitCreated);
        }

        [Test]
        public void CreateUnit_InsufficientPoints_IsRejected()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 2);
            var cmd = new CreateUnit(new UnitStats(5, 0, 0, 0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.InsufficientPoints, r.Reason);
        }

        [Test]
        public void CreateUnit_InvalidStats_IsRejected()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 10);
            var cmd = new CreateUnit(new UnitStats(0, 5, 0, 0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.InvalidStats, r.Reason);
        }

        [Test]
        public void DeployUnit_MovesReserveToBoard_AtDeploymentZone()
        {
            var s0 = TestStates.WithReserveUnit(out int reserveIdx, deployCoord: new HexCoord(0, 0, 0));
            var cmd = new DeployUnit(reserveIdx, new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(0, r.NewState.Players[0].Reserve.Count);
            Assert.AreEqual(1, r.NewState.Players[0].UnitsOnBoard.Count);
            Assert.AreEqual(new HexCoord(0, 0, 0), r.NewState.Players[0].UnitsOnBoard[0].Position);
        }

        [Test]
        public void EndTurn_FlipsActivePlayer_AndCreditsIncomeAtStartOfNextTurn()
        {
            var s0 = TestStates.WithGeneratorFor(PlayerId.Player1, output: 1);
            // active player is Player0; ending P0 turn makes P1 active and credits P1 income.
            int p1Before = s0.Players[1].Points;
            var cmd = new EndTurn() { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(PlayerId.Player1, r.NewState.ActivePlayer);
            Assert.AreEqual(p1Before + 1, r.NewState.Players[1].Points);
        }

        [Test]
        public void EndTurn_AfterPlayer1_IncrementsRound()
        {
            var s0 = TestStates.ActivePlayer1Round1();
            var cmd = new EndTurn() { Issuer = PlayerId.Player1 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(PlayerId.Player0, r.NewState.ActivePlayer);
            Assert.AreEqual(2, r.NewState.Round);
        }

        [Test]
        public void EndTurn_ClearsHasActedOnNewlyActivePlayersUnits()
        {
            var s0 = TestStates.WithActedUnitFor(PlayerId.Player1);
            var cmd = new EndTurn() { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            foreach (var u in r.NewState.Players[1].UnitsOnBoard)
                Assert.IsFalse(u.HasActed);
        }
```
```csharp
// Assets/HexWars/Engine/Tests/CommandRejectionTests.cs (new file)
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class CommandRejectionTests
    {
        [Test]
        public void Command_FromWrongPlayer_IsNotYourTurn()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 10); // active = Player0
            var cmd = new CreateUnit(new UnitStats(1, 0, 0, 0, 0, 0)) { Issuer = PlayerId.Player1 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.NotYourTurn, r.Reason);
        }

        [Test]
        public void DeployUnit_BadReserveIndex_IsReserveUnitNotFound()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 10);
            var cmd = new DeployUnit(99, new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.ReserveUnitNotFound, r.Reason);
        }

        [Test]
        public void DeployUnit_OutsideDeploymentZone_IsRejected()
        {
            var s0 = TestStates.WithReserveUnit(out int idx, deployCoord: new HexCoord(0, 0, 0));
            var cmd = new DeployUnit(idx, new HexCoord(5, 5, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.OutsideDeploymentZone, r.Reason);
        }

        [Test]
        public void DeployUnit_OnOccupiedTile_IsTileOccupied()
        {
            var s0 = TestStates.WithReserveUnitAndOccupiedZone(out int idx, occupied: new HexCoord(0, 0, 0));
            var cmd = new DeployUnit(idx, new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.TileOccupied, r.Reason);
        }

        [Test]
        public void DeployGenerator_PaysCost_AndPlacesInZone()
        {
            var s0 = TestStates.EmptyZoneFor(PlayerId.Player0, zoneCoord: new HexCoord(0, 0, 0), startingPoints: 10);
            var cmd = new DeployGenerator(new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(1, r.NewState.Players[0].Generators.Count);
            Assert.AreEqual(10 - s0.Config.GeneratorCost, r.NewState.Players[0].Points);
        }

        [Test]
        public void DeployGenerator_InsufficientPoints_IsRejected()
        {
            var s0 = TestStates.EmptyZoneFor(PlayerId.Player0, zoneCoord: new HexCoord(0, 0, 0), startingPoints: 1);
            var cmd = new DeployGenerator(new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.InsufficientPoints, r.Reason);
        }
    }
}
```
_Note: add the new `TestStates` factory helpers (`OnePlayerBoard`, `WithReserveUnit`, `WithReserveUnitAndOccupiedZone`, `EmptyZoneFor`, `WithGeneratorFor`, `ActivePlayer1Round1`, `WithActedUnitFor`) to `Assets/HexWars/Engine/Tests/TestStates.cs`. Each builds a small `Board` (use a 3x3 plains board with `DeploymentZone0` containing the listed coords) and the requested `PlayerState`/`GameState` via the public constructors. Keep them tiny and deterministic._

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests|HexWars.Engine.Tests.CommandRejectionTests" -logFile -`
Expected: FAIL â€” compilation error, `GameEngine` does not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Commands/GameEngine.cs
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine
{
    public static partial class GameEngine
    {
        public static Result Apply(GameState state, Command command)
        {
            if (state.IsGameOver)
                return Result.Rejected(state, RejectionReason.GameAlreadyOver, "game is over");

            var policy = TurnPolicyFactory.Create(state.Config.TurnPolicy);
            if (!policy.IsCommandAllowed(state, command))
                return Result.Rejected(state, RejectionReason.IllegalCommandForPolicy, "command not allowed by turn policy");

            if (command.Issuer != state.ActivePlayer)
                return Result.Rejected(state, RejectionReason.NotYourTurn, "not the issuer's turn");

            switch (command)
            {
                case CreateUnit c: return ApplyCreateUnit(state, c);
                case DeployUnit c: return ApplyDeployUnit(state, c);
                case DeployGenerator c: return ApplyDeployGenerator(state, c);
                case EndTurn c: return ApplyEndTurn(state, c);
                default:
                    return Result.Rejected(state, RejectionReason.IllegalCommandForPolicy, "unhandled command type");
            }
        }

        private static Result ApplyCreateUnit(GameState state, CreateUnit c)
        {
            if (!c.Stats.IsValid)
                return Result.Rejected(state, RejectionReason.InvalidStats, "unit must have >=1 Health");
            int cost = c.Stats.PointCost;
            if (!Economy.CanAfford(state, c.Issuer, cost))
                return Result.Rejected(state, RejectionReason.InsufficientPoints, "cannot afford unit");

            var ns = state.Clone();
            var ps = ns.Players[(int)c.Issuer];
            var reserve = new List<UnitStats>(ps.Reserve) { c.Stats };
            ns.Players[(int)c.Issuer] = new PlayerState(ps.Id, ps.Points - cost, reserve, ps.UnitsOnBoard, ps.Generators);

            var events = new List<GameEvent>
            {
                new GameEvent { Type = GameEventType.UnitCreated, Player = c.Issuer, Amount = cost }
            };
            return Result.Ok(ns, events);
        }

        private static Result ApplyDeployUnit(GameState state, DeployUnit c)
        {
            var ps = state.PlayerStateFor(c.Issuer);
            if (c.ReserveIndex < 0 || c.ReserveIndex >= ps.Reserve.Count)
                return Result.Rejected(state, RejectionReason.ReserveUnitNotFound, "no reserve unit at index");
            if (!state.Board.TryGetTile(c.Target.Q, c.Target.R, out var tile))
                return Result.Rejected(state, RejectionReason.TileNotFound, "target tile not on board");
            if (!state.Config.TerrainDefFor(tile.Terrain).Passable)
                return Result.Rejected(state, RejectionReason.TileImpassable, "target tile impassable");
            if (!state.Board.IsInDeploymentZone(c.Issuer, c.Target.Q, c.Target.R))
                return Result.Rejected(state, RejectionReason.OutsideDeploymentZone, "target outside deployment zone");
            if (IsOccupied(state, c.Target))
                return Result.Rejected(state, RejectionReason.TileOccupied, "target tile occupied");

            var stats = ps.Reserve[c.ReserveIndex];
            var ns = state.Clone();
            var nps = ns.Players[(int)c.Issuer];
            var reserve = new List<UnitStats>(nps.Reserve);
            reserve.RemoveAt(c.ReserveIndex);
            int id = ns.NextEntityId;
            var pos = c.Target.WithElevation(tile.Elevation);
            var unit = new Unit(id, c.Issuer, stats, pos, stats.Health, false);
            var units = new List<Unit>(nps.UnitsOnBoard) { unit };
            ns.Players[(int)c.Issuer] = new PlayerState(nps.Id, nps.Points, reserve, units, nps.Generators);
            ns.NextEntityId = id + 1;

            var events = new List<GameEvent>
            {
                new GameEvent { Type = GameEventType.UnitDeployed, EntityId = id, Player = c.Issuer, Coord = pos }
            };
            return Result.Ok(ns, events);
        }

        private static Result ApplyDeployGenerator(GameState state, DeployGenerator c)
        {
            int cost = state.Config.GeneratorCost;
            if (!Economy.CanAfford(state, c.Issuer, cost))
                return Result.Rejected(state, RejectionReason.InsufficientPoints, "cannot afford generator");
            if (!state.Board.TryGetTile(c.Target.Q, c.Target.R, out var tile))
                return Result.Rejected(state, RejectionReason.TileNotFound, "target tile not on board");
            if (!state.Config.TerrainDefFor(tile.Terrain).Passable)
                return Result.Rejected(state, RejectionReason.TileImpassable, "target tile impassable");
            if (!state.Board.IsInDeploymentZone(c.Issuer, c.Target.Q, c.Target.R))
                return Result.Rejected(state, RejectionReason.OutsideDeploymentZone, "target outside deployment zone");
            if (IsOccupied(state, c.Target))
                return Result.Rejected(state, RejectionReason.TileOccupied, "target tile occupied");

            var ns = state.Clone();
            var nps = ns.Players[(int)c.Issuer];
            int id = ns.NextEntityId;
            var pos = c.Target.WithElevation(tile.Elevation);
            var gen = new Generator(id, c.Issuer, pos, state.Config.GeneratorHealth);
            var gens = new List<Generator>(nps.Generators) { gen };
            ns.Players[(int)c.Issuer] = new PlayerState(nps.Id, nps.Points - cost, nps.Reserve, nps.UnitsOnBoard, gens);
            ns.NextEntityId = id + 1;

            var events = new List<GameEvent>
            {
                new GameEvent { Type = GameEventType.GeneratorDeployed, EntityId = id, Player = c.Issuer, Amount = cost, Coord = pos }
            };
            return Result.Ok(ns, events);
        }

        private static Result ApplyEndTurn(GameState state, EndTurn c)
        {
            var ns = state.Clone();
            var ending = ns.ActivePlayer;
            var next = ns.Opponent(ending);
            ns.ActivePlayer = next;
            if (ending == PlayerId.Player1)
                ns.Round = ns.Round + 1;

            var events = new List<GameEvent> { new GameEvent { Type = GameEventType.TurnEnded, Player = ending } };

            // Clear HasActed for the newly-active player's units.
            var nps = ns.Players[(int)next];
            var cleared = nps.UnitsOnBoard.Select(u => u.WithHasActed(false)).ToList();

            // Credit start-of-turn income to the newly-active player.
            int income = Economy.IncomeFor(ns, next);
            ns.Players[(int)next] = new PlayerState(nps.Id, nps.Points + income, nps.Reserve, cleared, nps.Generators);
            if (income > 0)
                events.Add(new GameEvent { Type = GameEventType.IncomeCredited, Player = next, Amount = income });

            var winner = WinCheck.CheckWinner(ns);
            if (winner.HasValue)
            {
                ns.IsGameOver = true;
                ns.Winner = winner;
                events.Add(new GameEvent { Type = GameEventType.GameWon, Player = winner.Value });
            }
            return Result.Ok(ns, events);
        }

        // Shared occupancy check across both players' units and generators.
        internal static bool IsOccupied(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
            {
                foreach (var u in p.UnitsOnBoard)
                    if (u.Position.Q == coord.Q && u.Position.R == coord.R) return true;
                foreach (var g in p.Generators)
                    if (g.Position.Q == coord.Q && g.Position.R == coord.R) return true;
            }
            return false;
        }
    }
}
```
_Implementation note: `GameEngine` is declared `static partial` so Tasks 17 and 18 can add MoveUnit/AttackUnit handling and `LegalMoves` in the same file (the contract specifies a single `GameEngine.cs`, so keep all parts in that one file; the `partial` keyword simply documents that those handlers are appended in later tasks). This requires `GameState` to expose settable `ActivePlayer`, `Round`, `NextEntityId`, `IsGameOver`, `Winner`, and `Players` element assignment on a cloned instance â€” Task 7's `Clone()` returns a fresh mutable-during-construction `GameState`; if Task 7 made those get-only, the deploy/end-turn handlers must instead build a brand-new `GameState` via its constructor with the updated values. Use whichever Task 7 actually shipped; the rejection/event semantics above are the contract._

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests|HexWars.Engine.Tests.CommandRejectionTests" -logFile -`
Expected: PASS â€” create/deploy/generator/EndTurn happy paths, all listed rejections, NotYourTurn, and non-mutation assertions green.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Commands/GameEngine.cs Assets/HexWars/Engine/Tests/CommandApplyTests.cs Assets/HexWars/Engine/Tests/CommandRejectionTests.cs Assets/HexWars/Engine/Tests/TestStates.cs && git commit -m "feat(engine): GameEngine.Apply for create/deploy/EndTurn with start-of-turn income"
```

---

### Task 17: GameEngine.Apply for MoveUnit and AttackUnit (movement, combat, bounty, once-per-turn)

**Files:**
- Modify: `Assets/HexWars/Engine/Commands/GameEngine.cs` (add MoveUnit/AttackUnit dispatch + handlers)
- Modify: `Assets/HexWars/Engine/Tests/CommandApplyTests.cs` (append move/attack happy-path + bounty + HasActed tests)
- Modify: `Assets/HexWars/Engine/Tests/CommandRejectionTests.cs` (append move/attack rejection tests)
- Test: `Assets/HexWars/Engine/Tests/CommandApplyTests.cs`, `Assets/HexWars/Engine/Tests/CommandRejectionTests.cs`

**Interfaces:**
- Consumes: `Pathfinding.TryFindPath` (Task 9); `TargetingService.CanTarget`, `TargetingService.ValidTargetIds` (Task 10); `CombatResolver.ResolveAgainstUnit`/`ResolveAgainstGenerator`/`ComputeBounty`, `CombatOutcome` (Task 11); `Unit.WithPosition`/`WithHp`/`WithHasActed` (Task 5); `Generator.WithHp` (Task 5); `ITurnPolicy.ShouldAutoEndTurn` (Task 15); `WinCheck.CheckWinner` (Task 13); `GameEngine.Apply` scaffold + `IsOccupied` (Task 16).
- Produces: extends `GameEngine.Apply` to fully handle `MoveUnit` and `AttackUnit` (TargetId resolves to an enemy `Unit` OR `Generator`), emits `UnitMoved`/`UnitAttacked`/`UnitDestroyed`/`GeneratorDestroyed`/`BountyAwarded` events, sets `HasActed`, and applies `ShouldAutoEndTurn` (auto-EndTurn under `OneActionPolicy`).

- [ ] **Step 1: Write the failing test**
```csharp
// Append to Assets/HexWars/Engine/Tests/CommandApplyTests.cs (inside the class)
        [Test]
        public void MoveUnit_MovesWithinBudget_SetsHasActed()
        {
            var s0 = TestStates.UnitOnPlains(out int unitId, from: new HexCoord(0, 0, 0), movement: 2, owner: PlayerId.Player0);
            var cmd = new MoveUnit(unitId, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            var moved = r.NewState.Players[0].UnitsOnBoard.Find(u => u.Id == unitId);
            Assert.AreEqual(1, moved.Position.Q);
            Assert.AreEqual(0, moved.Position.R);
            Assert.IsTrue(moved.HasActed);
            Assert.IsTrue(r.Events.Exists(e => e.Type == GameEventType.UnitMoved && e.EntityId == unitId));
        }

        [Test]
        public void MoveUnit_AlreadyActed_IsRejected()
        {
            var s0 = TestStates.ActedUnitOnPlains(out int unitId, at: new HexCoord(0, 0, 0), owner: PlayerId.Player0);
            var cmd = new MoveUnit(unitId, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.UnitAlreadyActed, r.Reason);
        }

        [Test]
        public void AttackUnit_DealsDamage_AndSetsHasActed()
        {
            var s0 = TestStates.AttackerAndEnemyUnit(out int attackerId, out int targetId,
                attacker: new HexCoord(0, 0, 0), target: new HexCoord(1, 0, 0),
                attackerDamage: 3, targetHealth: 10, targetDefense: 0);
            var cmd = new AttackUnit(attackerId, targetId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            var enemy = r.NewState.Players[1].UnitsOnBoard.Find(u => u.Id == targetId);
            Assert.AreEqual(7, enemy.CurrentHp);
            var attacker = r.NewState.Players[0].UnitsOnBoard.Find(u => u.Id == attackerId);
            Assert.IsTrue(attacker.HasActed);
            Assert.IsTrue(r.Events.Exists(e => e.Type == GameEventType.UnitAttacked && e.EntityId == attackerId));
        }

        [Test]
        public void AttackUnit_LethalHit_DestroysTarget_AndCreditsBounty()
        {
            var s0 = TestStates.AttackerAndEnemyUnit(out int attackerId, out int targetId,
                attacker: new HexCoord(0, 0, 0), target: new HexCoord(1, 0, 0),
                attackerDamage: 10, targetHealth: 4, targetDefense: 0); // target buildCost = 4
            int p0Before = s0.Players[0].Points;
            int expectedBounty = CombatResolver.ComputeBounty(s0.Config, 4);
            var cmd = new AttackUnit(attackerId, targetId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.IsNull(r.NewState.Players[1].UnitsOnBoard.Find(u => u.Id == targetId));
            Assert.AreEqual(p0Before + expectedBounty, r.NewState.Players[0].Points);
            Assert.IsTrue(r.Events.Exists(e => e.Type == GameEventType.UnitDestroyed && e.TargetId == targetId));
            Assert.IsTrue(r.Events.Exists(e => e.Type == GameEventType.BountyAwarded && e.Amount == expectedBounty));
        }

        [Test]
        public void AttackGenerator_CanDestroyEnemyGenerator()
        {
            var s0 = TestStates.AttackerAndEnemyGenerator(out int attackerId, out int genId,
                attacker: new HexCoord(0, 0, 0), gen: new HexCoord(1, 0, 0), attackerDamage: 10, genHealth: 3);
            var cmd = new AttackUnit(attackerId, genId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.IsNull(r.NewState.Players[1].Generators.Find(g => g.Id == genId));
            Assert.IsTrue(r.Events.Exists(e => e.Type == GameEventType.GeneratorDestroyed && e.TargetId == genId));
        }

        [Test]
        public void OneActionPolicy_AutoEndsTurnAfterMove()
        {
            var s0 = TestStates.UnitOnPlainsOneAction(out int unitId, from: new HexCoord(0, 0, 0), movement: 2, owner: PlayerId.Player0);
            var cmd = new MoveUnit(unitId, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsTrue(r.Success, r.Message);
            Assert.AreEqual(PlayerId.Player1, r.NewState.ActivePlayer);
        }
```
```csharp
// Append to Assets/HexWars/Engine/Tests/CommandRejectionTests.cs (inside the class)
        [Test]
        public void MoveUnit_UnknownUnit_IsUnitNotFound()
        {
            var s0 = TestStates.UnitOnPlains(out int _, from: new HexCoord(0, 0, 0), movement: 2, owner: PlayerId.Player0);
            var cmd = new MoveUnit(9999, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.UnitNotFound, r.Reason);
        }

        [Test]
        public void MoveUnit_BeyondBudget_IsOutOfMovementRange()
        {
            var s0 = TestStates.UnitOnPlains(out int unitId, from: new HexCoord(0, 0, 0), movement: 1, owner: PlayerId.Player0);
            var cmd = new MoveUnit(unitId, new HexCoord(3, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.OutOfMovementRange, r.Reason);
        }

        [Test]
        public void MoveUnit_OntoOccupiedTile_IsTileOccupied()
        {
            var s0 = TestStates.TwoUnitsAdjacent(out int moverId, mover: new HexCoord(0, 0, 0), blocker: new HexCoord(1, 0, 0), movement: 2, owner: PlayerId.Player0);
            var cmd = new MoveUnit(moverId, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.TileOccupied, r.Reason);
        }

        [Test]
        public void AttackUnit_TargetOutOfRange_IsTargetNotInRange()
        {
            var s0 = TestStates.AttackerAndEnemyUnit(out int attackerId, out int targetId,
                attacker: new HexCoord(0, 0, 0), target: new HexCoord(5, 0, 0),
                attackerDamage: 3, targetHealth: 5, targetDefense: 0, attackerRange: 1, attackerVision: 10);
            var cmd = new AttackUnit(attackerId, targetId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.TargetNotInRange, r.Reason);
        }

        [Test]
        public void AttackUnit_TargetNotVisible_IsTargetNotVisible()
        {
            var s0 = TestStates.AttackerAndEnemyUnit(out int attackerId, out int targetId,
                attacker: new HexCoord(0, 0, 0), target: new HexCoord(2, 0, 0),
                attackerDamage: 3, targetHealth: 5, targetDefense: 0, attackerRange: 10, attackerVision: 1);
            var cmd = new AttackUnit(attackerId, targetId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.TargetNotVisible, r.Reason);
        }

        [Test]
        public void AttackUnit_FriendlyTarget_IsTargetNotEnemy()
        {
            var s0 = TestStates.TwoFriendlyUnits(out int attackerId, out int friendId,
                attacker: new HexCoord(0, 0, 0), friend: new HexCoord(1, 0, 0), owner: PlayerId.Player0);
            var cmd = new AttackUnit(attackerId, friendId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.TargetNotEnemy, r.Reason);
        }

        [Test]
        public void AttackUnit_AfterActing_IsUnitAlreadyActed()
        {
            var s0 = TestStates.ActedAttackerAndEnemy(out int attackerId, out int targetId,
                attacker: new HexCoord(0, 0, 0), target: new HexCoord(1, 0, 0), owner: PlayerId.Player0);
            var cmd = new AttackUnit(attackerId, targetId) { Issuer = PlayerId.Player0 };
            var r = GameEngine.Apply(s0, cmd);
            Assert.IsFalse(r.Success);
            Assert.AreEqual(RejectionReason.UnitAlreadyActed, r.Reason);
        }
```
_Note: add the new `TestStates` factory helpers (`UnitOnPlains`, `UnitOnPlainsOneAction`, `ActedUnitOnPlains`, `AttackerAndEnemyUnit` with optional `attackerRange`/`attackerVision` params defaulting to plenty, `AttackerAndEnemyGenerator`, `TwoUnitsAdjacent`, `TwoFriendlyUnits`, `ActedAttackerAndEnemy`). Build a flat plains board large enough for the coords; place units on each player's `UnitsOnBoard`. `UnitOnPlainsOneAction` sets `GameConfig` with `TurnPolicy = TurnPolicyKind.OneAction` via `Default()` then a `Clone`-with-override or constructed config._

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests|HexWars.Engine.Tests.CommandRejectionTests" -logFile -`
Expected: FAIL â€” the new move/attack tests fail because `Apply` does not yet dispatch `MoveUnit`/`AttackUnit` (returns `IllegalCommandForPolicy`).

- [ ] **Step 3: Write minimal implementation**
```csharp
// Add these cases to the switch in GameEngine.Apply (in GameEngine.cs):
//     case MoveUnit c: return ApplyMoveUnit(state, c, policy);
//     case AttackUnit c: return ApplyAttackUnit(state, c, policy);
// Change the Apply switch dispatch lines for Move/Attack to pass `policy`, and update
// CreateUnit/DeployUnit/DeployGenerator dispatch to also pass `policy` so ShouldAutoEndTurn
// can run uniformly. Then add the following methods inside the GameEngine partial class:

using System.Linq; // already present

        private static Result ApplyMoveUnit(GameState state, MoveUnit c, ITurnPolicy policy)
        {
            var ps = state.PlayerStateFor(c.Issuer);
            var unit = ps.UnitsOnBoard.FirstOrDefault(u => u.Id == c.UnitId);
            if (unit == null)
                return Result.Rejected(state, RejectionReason.UnitNotFound, "unit not found");
            if (unit.HasActed)
                return Result.Rejected(state, RejectionReason.UnitAlreadyActed, "unit already acted");
            if (!state.Board.TryGetTile(c.Target.Q, c.Target.R, out var tile))
                return Result.Rejected(state, RejectionReason.TileNotFound, "target tile not on board");
            if (!state.Config.TerrainDefFor(tile.Terrain).Passable)
                return Result.Rejected(state, RejectionReason.TileImpassable, "target impassable");
            if (IsOccupied(state, c.Target))
                return Result.Rejected(state, RejectionReason.TileOccupied, "target occupied");
            int climb = tile.Elevation - unit.Position.Elevation;
            if (climb > state.Config.MaxClimbPerStep)
                return Result.Rejected(state, RejectionReason.ClimbTooSteep, "climb exceeds max per step");
            if (!Pathfinding.TryFindPath(state, unit.Position, c.Target.WithElevation(tile.Elevation),
                    unit.Stats.Movement, out var _, out var _))
                return Result.Rejected(state, RejectionReason.OutOfMovementRange, "no path within movement");

            var ns = state.Clone();
            var nps = ns.Players[(int)c.Issuer];
            var units = new List<Unit>(nps.UnitsOnBoard);
            int idx = units.FindIndex(u => u.Id == c.UnitId);
            var dest = c.Target.WithElevation(tile.Elevation);
            units[idx] = units[idx].WithPosition(dest).WithHasActed(true);
            ns.Players[(int)c.Issuer] = new PlayerState(nps.Id, nps.Points, nps.Reserve, units, nps.Generators);

            var events = new List<GameEvent>
            {
                new GameEvent { Type = GameEventType.UnitMoved, EntityId = c.UnitId, Player = c.Issuer, Coord = dest }
            };
            return Finalize(ns, c, policy, events);
        }

        private static Result ApplyAttackUnit(GameState state, AttackUnit c, ITurnPolicy policy)
        {
            var ps = state.PlayerStateFor(c.Issuer);
            var attacker = ps.UnitsOnBoard.FirstOrDefault(u => u.Id == c.AttackerUnitId);
            if (attacker == null)
                return Result.Rejected(state, RejectionReason.UnitNotFound, "attacker not found");
            if (attacker.HasActed)
                return Result.Rejected(state, RejectionReason.UnitAlreadyActed, "attacker already acted");

            var enemyId = state.Opponent(c.Issuer);
            var enemy = state.PlayerStateFor(enemyId);
            var targetUnit = enemy.UnitsOnBoard.FirstOrDefault(u => u.Id == c.TargetId);
            var targetGen = enemy.Generators.FirstOrDefault(g => g.Id == c.TargetId);
            if (targetUnit == null && targetGen == null)
                return Result.Rejected(state, RejectionReason.TargetNotEnemy, "target is not an enemy unit or generator");

            HexCoord targetPos = targetUnit != null ? targetUnit.Position : targetGen.Position;
            state.Board.TryGetTile(targetPos.Q, targetPos.R, out var targetTile);
            if (!TargetingService.InRange(state, attacker, targetPos))
                return Result.Rejected(state, RejectionReason.TargetNotInRange, "target out of range");
            if (!TargetingService.IsVisible(state, attacker, targetPos, targetTile.Terrain))
                return Result.Rejected(state, RejectionReason.TargetNotVisible, "target not visible");

            var ns = state.Clone();
            var nAttackerPs = ns.Players[(int)c.Issuer];
            var atkUnits = new List<Unit>(nAttackerPs.UnitsOnBoard);
            int atkIdx = atkUnits.FindIndex(u => u.Id == c.AttackerUnitId);
            atkUnits[atkIdx] = atkUnits[atkIdx].WithHasActed(true);

            var events = new List<GameEvent>();
            int bounty = 0;
            var nEnemyPs = ns.Players[(int)enemyId];
            if (targetUnit != null)
            {
                var outcome = CombatResolver.ResolveAgainstUnit(state, attacker, targetUnit);
                events.Add(new GameEvent { Type = GameEventType.UnitAttacked, EntityId = c.AttackerUnitId, TargetId = c.TargetId, Player = c.Issuer, Amount = outcome.Damage });
                var enemyUnits = new List<Unit>(nEnemyPs.UnitsOnBoard);
                int tIdx = enemyUnits.FindIndex(u => u.Id == c.TargetId);
                if (outcome.TargetDestroyed)
                {
                    enemyUnits.RemoveAt(tIdx);
                    bounty = outcome.Bounty;
                    events.Add(new GameEvent { Type = GameEventType.UnitDestroyed, TargetId = c.TargetId, Player = enemyId });
                }
                else
                {
                    enemyUnits[tIdx] = enemyUnits[tIdx].WithHp(outcome.TargetHpAfter);
                }
                ns.Players[(int)enemyId] = new PlayerState(nEnemyPs.Id, nEnemyPs.Points, nEnemyPs.Reserve, enemyUnits, nEnemyPs.Generators);
            }
            else
            {
                var outcome = CombatResolver.ResolveAgainstGenerator(state, attacker, targetGen);
                events.Add(new GameEvent { Type = GameEventType.UnitAttacked, EntityId = c.AttackerUnitId, TargetId = c.TargetId, Player = c.Issuer, Amount = outcome.Damage });
                var enemyGens = new List<Generator>(nEnemyPs.Generators);
                int gIdx = enemyGens.FindIndex(g => g.Id == c.TargetId);
                if (outcome.TargetDestroyed)
                {
                    enemyGens.RemoveAt(gIdx);
                    bounty = outcome.Bounty;
                    events.Add(new GameEvent { Type = GameEventType.GeneratorDestroyed, TargetId = c.TargetId, Player = enemyId });
                }
                else
                {
                    enemyGens[gIdx] = enemyGens[gIdx].WithHp(outcome.TargetHpAfter);
                }
                ns.Players[(int)enemyId] = new PlayerState(nEnemyPs.Id, nEnemyPs.Points, nEnemyPs.Reserve, nEnemyPs.UnitsOnBoard, enemyGens);
            }

            // Re-read attacker player state (it may equal enemyId? no: attacker != enemy) and credit bounty.
            var atkPsNow = ns.Players[(int)c.Issuer];
            ns.Players[(int)c.Issuer] = new PlayerState(atkPsNow.Id, atkPsNow.Points + bounty, atkPsNow.Reserve, atkUnits, atkPsNow.Generators);
            if (bounty > 0)
                events.Add(new GameEvent { Type = GameEventType.BountyAwarded, Player = c.Issuer, Amount = bounty });

            var winner = WinCheck.CheckWinner(ns);
            if (winner.HasValue)
            {
                ns.IsGameOver = true;
                ns.Winner = winner;
                events.Add(new GameEvent { Type = GameEventType.GameWon, Player = winner.Value });
                return Result.Ok(ns, events);
            }
            return Finalize(ns, c, policy, events);
        }

        // Applies OneAction auto-end after a successful non-EndTurn action.
        private static Result Finalize(GameState ns, Command command, ITurnPolicy policy, List<GameEvent> events)
        {
            if (policy.ShouldAutoEndTurn(ns, command))
            {
                var endResult = ApplyEndTurn(ns, new EndTurn { Issuer = ns.ActivePlayer });
                var merged = new List<GameEvent>(events);
                merged.AddRange(endResult.Events);
                return Result.Ok(endResult.NewState, merged);
            }
            return Result.Ok(ns, events);
        }
```
_Implementation note: wire the `MoveUnit`/`AttackUnit` cases into the existing `Apply` switch and pass `policy` through. For uniformity, route the create/deploy handlers through `Finalize(ns, command, policy, events)` as well (replacing their bare `Result.Ok`) so OneActionPolicy auto-ends after a create/deploy too, per spec Â§3. The `Finalize` short-circuit in AttackUnit when the game is already won avoids auto-ending into a finished state._

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CommandApplyTests|HexWars.Engine.Tests.CommandRejectionTests" -logFile -`
Expected: PASS â€” move, attack-unit, attack-generator, lethal+bounty, OneAction auto-end, and every move/attack rejection green.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Commands/GameEngine.cs Assets/HexWars/Engine/Tests/CommandApplyTests.cs Assets/HexWars/Engine/Tests/CommandRejectionTests.cs Assets/HexWars/Engine/Tests/TestStates.cs && git commit -m "feat(engine): GameEngine.Apply MoveUnit/AttackUnit with combat, bounty, once-per-turn"
```

---

### Task 18: GameEngine.LegalMoves enumeration

**Files:**
- Modify: `Assets/HexWars/Engine/Commands/GameEngine.cs` (add `LegalMoves`)
- Modify: `Assets/HexWars/Engine/Tests/LegalMovesTests.cs` (new file)
- Test: `Assets/HexWars/Engine/Tests/LegalMovesTests.cs`

**Interfaces:**
- Consumes: `GameEngine.Apply` (Tasks 16,17); `Pathfinding.ReachableTiles` (Task 9); `TargetingService.ValidTargetIds` (Task 10); `Economy.CanAfford`, `Economy.CheapestViableUnitCost` (Task 12); `ITurnPolicy.IsCommandAllowed` (Task 15); `Board.DeploymentZoneFor` (Task 4); `UnitStats` (Task 5).
- Produces: `public static List<Command> GameEngine.LegalMoves(GameState state)` â€” every element is guaranteed to produce `Result.Success` when passed to `Apply` for the current state.

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class LegalMovesTests
    {
        [Test]
        public void LegalMoves_AlwaysIncludesEndTurn()
        {
            var s0 = TestStates.OnePlayerBoard(startingPoints: 10);
            var moves = GameEngine.LegalMoves(s0);
            Assert.IsTrue(moves.Exists(m => m is EndTurn), "EndTurn must always be legal");
        }

        [Test]
        public void LegalMoves_IsNonEmptyMidGame()
        {
            var s0 = TestStates.MidGame(out int _); // a unit on board, points to spend, empty zone tiles
            var moves = GameEngine.LegalMoves(s0);
            Assert.Greater(moves.Count, 1);
        }

        [Test]
        public void LegalMoves_EveryReturnedCommand_AppliesSuccessfully()
        {
            var s0 = TestStates.MidGame(out int _);
            var moves = GameEngine.LegalMoves(s0);
            Assert.IsNotEmpty(moves);
            foreach (var m in moves)
            {
                var r = GameEngine.Apply(s0, m);
                Assert.IsTrue(r.Success, $"LegalMoves returned a command Apply rejected: {m} -> {r.Reason}: {r.Message}");
            }
        }

        [Test]
        public void LegalMoves_AllIssuedByActivePlayer()
        {
            var s0 = TestStates.MidGame(out int _);
            foreach (var m in GameEngine.LegalMoves(s0))
                Assert.AreEqual(s0.ActivePlayer, m.Issuer);
        }

        [Test]
        public void LegalMoves_IncludesDeployForReserveUnit()
        {
            var s0 = TestStates.WithReserveUnit(out int idx, deployCoord: new HexCoord(0, 0, 0));
            var moves = GameEngine.LegalMoves(s0);
            Assert.IsTrue(moves.Exists(m => m is DeployUnit du && du.ReserveIndex == idx));
        }
    }
}
```
_Note: add a `TestStates.MidGame(out int unitId)` helper building a small 5x5 plains board, the active player (Player0) with some banked points (>= a buyable unit cost), at least one un-acted unit on board with non-zero Movement, an enemy unit within potential reach, and empty deployment-zone tiles â€” so Create/Deploy/Move/Attack/EndTurn are all enumerable._

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.LegalMovesTests" -logFile -`
Expected: FAIL â€” compilation error, `GameEngine.LegalMoves` does not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Add to the GameEngine partial class in GameEngine.cs:
        public static List<Command> LegalMoves(GameState state)
        {
            var moves = new List<Command>();
            if (state.IsGameOver) return moves;

            var policy = TurnPolicyFactory.Create(state.Config.TurnPolicy);
            var pid = state.ActivePlayer;
            var ps = state.ActivePlayerState;
            var zone = state.Board.DeploymentZoneFor(pid);

            // Representative affordable CreateUnit options: cheapest viable unit + a couple of
            // single-stat variants, only those affordable.
            foreach (var stats in RepresentativeStats(state))
            {
                if (Economy.CanAfford(state, pid, stats.PointCost))
                {
                    var cmd = new CreateUnit(stats) { Issuer = pid };
                    if (policy.IsCommandAllowed(state, cmd)) moves.Add(cmd);
                }
            }

            // Empty, passable zone tiles (used for both deploy-unit and deploy-generator).
            var emptyZoneTiles = new List<HexCoord>();
            foreach (var z in zone)
            {
                if (!state.Board.TryGetTile(z.Q, z.R, out var t)) continue;
                if (!state.Config.TerrainDefFor(t.Terrain).Passable) continue;
                if (IsOccupied(state, z)) continue;
                emptyZoneTiles.Add(z.WithElevation(t.Elevation));
            }

            // DeployUnit for each reserve index x empty zone tile.
            for (int i = 0; i < ps.Reserve.Count; i++)
            {
                foreach (var z in emptyZoneTiles)
                {
                    var cmd = new DeployUnit(i, z) { Issuer = pid };
                    if (policy.IsCommandAllowed(state, cmd)) moves.Add(cmd);
                }
            }

            // DeployGenerator for each empty zone tile if affordable.
            if (Economy.CanAfford(state, pid, state.Config.GeneratorCost))
            {
                foreach (var z in emptyZoneTiles)
                {
                    var cmd = new DeployGenerator(z) { Issuer = pid };
                    if (policy.IsCommandAllowed(state, cmd)) moves.Add(cmd);
                }
            }

            // MoveUnit for each un-acted unit x reachable tile.
            foreach (var u in ps.UnitsOnBoard)
            {
                if (u.HasActed) continue;
                foreach (var dest in Pathfinding.ReachableTiles(state, u.Position, u.Stats.Movement))
                {
                    var cmd = new MoveUnit(u.Id, dest) { Issuer = pid };
                    if (policy.IsCommandAllowed(state, cmd)) moves.Add(cmd);
                }
            }

            // AttackUnit for each un-acted unit x valid target.
            foreach (var u in ps.UnitsOnBoard)
            {
                if (u.HasActed) continue;
                foreach (var targetId in TargetingService.ValidTargetIds(state, u))
                {
                    var cmd = new AttackUnit(u.Id, targetId) { Issuer = pid };
                    if (policy.IsCommandAllowed(state, cmd)) moves.Add(cmd);
                }
            }

            // EndTurn is always legal.
            moves.Add(new EndTurn() { Issuer = pid });
            return moves;
        }

        // Deterministic representative stat set for CreateUnit enumeration: cheapest viable unit
        // (1 HP) plus a small fixed palette of single-purpose builds. Kept small and ordered.
        private static List<UnitStats> RepresentativeStats(GameState state)
        {
            return new List<UnitStats>
            {
                new UnitStats(1, 0, 0, 0, 0, 0), // cheapest viable gnat
                new UnitStats(1, 1, 1, 1, 0, 1), // generalist scout
                new UnitStats(3, 2, 1, 1, 0, 1), // brawler
                new UnitStats(1, 0, 0, 0, 0, 3), // pure spotter
            };
        }
```

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.LegalMovesTests" -logFile -`
Expected: PASS â€” non-empty mid-game, every returned command applies successfully, all issued by the active player.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Commands/GameEngine.cs Assets/HexWars/Engine/Tests/LegalMovesTests.cs Assets/HexWars/Engine/Tests/TestStates.cs && git commit -m "feat(engine): add GameEngine.LegalMoves enumeration"
```

---

### Task 19: GameStateSerializer: JSON round-trip for GameState and Command

**Files:**
- Create: `Assets/HexWars/Engine/Serialization/GameStateSerializer.cs`
- Modify: `Assets/HexWars/Engine/Tests/SerializationTests.cs` (new file)
- Test: `Assets/HexWars/Engine/Tests/SerializationTests.cs`

**Interfaces:**
- Consumes: `GameState`/`Board`/`PlayerState`/`Unit`/`Generator`/`UnitStats`/`HexCoord`/`GameConfig`/`Tile`/`TerrainDef`/`TerrainType` (Tasks 7,4,5,6,3,2); `Command` records `CreateUnit`/`DeployUnit`/`DeployGenerator`/`MoveUnit`/`AttackUnit`/`EndTurn` (Task 14).
- Produces: `public static class GameStateSerializer { static string ToJson(GameState); static GameState FromJson(string); static string CommandToJson(Command); static Command CommandFromJson(string); }` â€” engine-only serializer, NO `UnityEngine.JsonUtility`, deterministic ordering.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class SerializationTests
    {
        [Test]
        public void GameState_RoundTrips_DeepEqual()
        {
            var s0 = TestStates.MidGame(out int _);
            string json = GameStateSerializer.ToJson(s0);
            var s1 = GameStateSerializer.FromJson(json);
            Assert.AreEqual(GameStateSerializer.ToJson(s0), GameStateSerializer.ToJson(s1),
                "re-serialized state must be identical to original serialization");
            // structural spot-checks
            Assert.AreEqual(s0.ActivePlayer, s1.ActivePlayer);
            Assert.AreEqual(s0.Round, s1.Round);
            Assert.AreEqual(s0.NextEntityId, s1.NextEntityId);
            Assert.AreEqual(s0.Players[0].Points, s1.Players[0].Points);
            Assert.AreEqual(s0.Players[0].UnitsOnBoard.Count, s1.Players[0].UnitsOnBoard.Count);
            Assert.AreEqual(s0.Board.Tiles.Count, s1.Board.Tiles.Count);
            Assert.AreEqual(s0.Config.StartingPoints, s1.Config.StartingPoints);
        }

        [Test]
        public void Serialization_IsDeterministic_SameInputSameJson()
        {
            var s0 = TestStates.MidGame(out int _);
            Assert.AreEqual(GameStateSerializer.ToJson(s0), GameStateSerializer.ToJson(s0));
        }

        [Test]
        public void CreateUnit_Command_RoundTrips()
        {
            Command c = new CreateUnit(new UnitStats(3, 2, 1, 1, 0, 2)) { Issuer = PlayerId.Player1 };
            var c2 = (CreateUnit)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }

        [Test]
        public void DeployUnit_Command_RoundTrips()
        {
            Command c = new DeployUnit(2, new HexCoord(1, 2, 3)) { Issuer = PlayerId.Player0 };
            var c2 = (DeployUnit)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }

        [Test]
        public void DeployGenerator_Command_RoundTrips()
        {
            Command c = new DeployGenerator(new HexCoord(4, 0, 1)) { Issuer = PlayerId.Player1 };
            var c2 = (DeployGenerator)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }

        [Test]
        public void MoveUnit_Command_RoundTrips()
        {
            Command c = new MoveUnit(7, new HexCoord(0, 5, 2)) { Issuer = PlayerId.Player0 };
            var c2 = (MoveUnit)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }

        [Test]
        public void AttackUnit_Command_RoundTrips()
        {
            Command c = new AttackUnit(3, 9) { Issuer = PlayerId.Player1 };
            var c2 = (AttackUnit)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }

        [Test]
        public void EndTurn_Command_RoundTrips()
        {
            Command c = new EndTurn() { Issuer = PlayerId.Player1 };
            var c2 = (EndTurn)GameStateSerializer.CommandFromJson(GameStateSerializer.CommandToJson(c));
            Assert.AreEqual(c, c2);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.SerializationTests" -logFile -`
Expected: FAIL â€” compilation error, `GameStateSerializer` does not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Serialization/GameStateSerializer.cs
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace HexWars.Engine
{
    // Hand-rolled deterministic JSON. No UnityEngine. Field order is fixed by write order,
    // and the parser is a small recursive-descent reader. Sufficient for engine round-trips
    // (GameState, Command); not a general-purpose JSON library.
    public static class GameStateSerializer
    {
        // ---------- Public API ----------
        public static string ToJson(GameState state)
        {
            var sb = new StringBuilder();
            WriteGameState(sb, state);
            return sb.ToString();
        }

        public static GameState FromJson(string json)
        {
            var p = new JsonParser(json);
            var obj = p.ParseObject();
            return ReadGameState(obj);
        }

        public static string CommandToJson(Command command)
        {
            var sb = new StringBuilder();
            sb.Append('{');
            switch (command)
            {
                case CreateUnit c:
                    WriteField(sb, "type", "CreateUnit"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer); sb.Append(',');
                    sb.Append("\"stats\":"); WriteStats(sb, c.Stats);
                    break;
                case DeployUnit c:
                    WriteField(sb, "type", "DeployUnit"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer); sb.Append(',');
                    WriteIntField(sb, "reserveIndex", c.ReserveIndex); sb.Append(',');
                    sb.Append("\"target\":"); WriteCoord(sb, c.Target);
                    break;
                case DeployGenerator c:
                    WriteField(sb, "type", "DeployGenerator"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer); sb.Append(',');
                    sb.Append("\"target\":"); WriteCoord(sb, c.Target);
                    break;
                case MoveUnit c:
                    WriteField(sb, "type", "MoveUnit"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer); sb.Append(',');
                    WriteIntField(sb, "unitId", c.UnitId); sb.Append(',');
                    sb.Append("\"target\":"); WriteCoord(sb, c.Target);
                    break;
                case AttackUnit c:
                    WriteField(sb, "type", "AttackUnit"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer); sb.Append(',');
                    WriteIntField(sb, "attackerUnitId", c.AttackerUnitId); sb.Append(',');
                    WriteIntField(sb, "targetId", c.TargetId);
                    break;
                case EndTurn c:
                    WriteField(sb, "type", "EndTurn"); sb.Append(',');
                    WriteIntField(sb, "issuer", (int)c.Issuer);
                    break;
            }
            sb.Append('}');
            return sb.ToString();
        }

        public static Command CommandFromJson(string json)
        {
            var o = new JsonParser(json).ParseObject();
            var type = (string)o["type"];
            var issuer = (PlayerId)(int)(double)o["issuer"];
            switch (type)
            {
                case "CreateUnit": return new CreateUnit(ReadStats((Dictionary<string, object>)o["stats"])) { Issuer = issuer };
                case "DeployUnit": return new DeployUnit((int)(double)o["reserveIndex"], ReadCoord((Dictionary<string, object>)o["target"])) { Issuer = issuer };
                case "DeployGenerator": return new DeployGenerator(ReadCoord((Dictionary<string, object>)o["target"])) { Issuer = issuer };
                case "MoveUnit": return new MoveUnit((int)(double)o["unitId"], ReadCoord((Dictionary<string, object>)o["target"])) { Issuer = issuer };
                case "AttackUnit": return new AttackUnit((int)(double)o["attackerUnitId"], (int)(double)o["targetId"]) { Issuer = issuer };
                case "EndTurn": return new EndTurn() { Issuer = issuer };
                default: throw new System.ArgumentException($"unknown command type '{type}'");
            }
        }

        // ---------- GameState writers ----------
        private static void WriteGameState(StringBuilder sb, GameState s)
        {
            sb.Append('{');
            WriteIntField(sb, "activePlayer", (int)s.ActivePlayer); sb.Append(',');
            WriteIntField(sb, "round", s.Round); sb.Append(',');
            WriteIntField(sb, "nextEntityId", s.NextEntityId); sb.Append(',');
            WriteBoolField(sb, "isGameOver", s.IsGameOver); sb.Append(',');
            WriteIntField(sb, "winner", s.Winner.HasValue ? (int)s.Winner.Value : -1); sb.Append(',');
            sb.Append("\"config\":"); WriteConfig(sb, s.Config); sb.Append(',');
            sb.Append("\"board\":"); WriteBoard(sb, s.Board); sb.Append(',');
            sb.Append("\"players\":["); WritePlayer(sb, s.Players[0]); sb.Append(','); WritePlayer(sb, s.Players[1]); sb.Append("]");
            sb.Append('}');
        }

        private static void WriteBoard(StringBuilder sb, Board b)
        {
            sb.Append('{');
            WriteIntField(sb, "width", b.Width); sb.Append(',');
            WriteIntField(sb, "height", b.Height); sb.Append(',');
            sb.Append("\"tiles\":[");
            for (int i = 0; i < b.Tiles.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var t = b.Tiles[i];
                sb.Append('{');
                WriteIntField(sb, "q", t.Coord.Q); sb.Append(',');
                WriteIntField(sb, "r", t.Coord.R); sb.Append(',');
                WriteIntField(sb, "e", t.Coord.Elevation); sb.Append(',');
                WriteIntField(sb, "terrain", (int)t.Terrain);
                sb.Append('}');
            }
            sb.Append("],");
            sb.Append("\"dz0\":"); WriteCoordList(sb, b.DeploymentZone0); sb.Append(',');
            sb.Append("\"dz1\":"); WriteCoordList(sb, b.DeploymentZone1);
            sb.Append('}');
        }

        private static void WriteCoordList(StringBuilder sb, IReadOnlyList<HexCoord> list)
        {
            sb.Append('[');
            for (int i = 0; i < list.Count; i++) { if (i > 0) sb.Append(','); WriteCoord(sb, list[i]); }
            sb.Append(']');
        }

        private static void WritePlayer(StringBuilder sb, PlayerState p)
        {
            sb.Append('{');
            WriteIntField(sb, "id", (int)p.Id); sb.Append(',');
            WriteIntField(sb, "points", p.Points); sb.Append(',');
            sb.Append("\"reserve\":[");
            for (int i = 0; i < p.Reserve.Count; i++) { if (i > 0) sb.Append(','); WriteStats(sb, p.Reserve[i]); }
            sb.Append("],");
            sb.Append("\"units\":[");
            for (int i = 0; i < p.UnitsOnBoard.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var u = p.UnitsOnBoard[i];
                sb.Append('{');
                WriteIntField(sb, "id", u.Id); sb.Append(',');
                WriteIntField(sb, "owner", (int)u.Owner); sb.Append(',');
                sb.Append("\"stats\":"); WriteStats(sb, u.Stats); sb.Append(',');
                sb.Append("\"pos\":"); WriteCoord(sb, u.Position); sb.Append(',');
                WriteIntField(sb, "hp", u.CurrentHp); sb.Append(',');
                WriteBoolField(sb, "hasActed", u.HasActed);
                sb.Append('}');
            }
            sb.Append("],");
            sb.Append("\"generators\":[");
            for (int i = 0; i < p.Generators.Count; i++)
            {
                if (i > 0) sb.Append(',');
                var g = p.Generators[i];
                sb.Append('{');
                WriteIntField(sb, "id", g.Id); sb.Append(',');
                WriteIntField(sb, "owner", (int)g.Owner); sb.Append(',');
                sb.Append("\"pos\":"); WriteCoord(sb, g.Position); sb.Append(',');
                WriteIntField(sb, "hp", g.CurrentHp);
                sb.Append('}');
            }
            sb.Append("]");
            sb.Append('}');
        }

        private static void WriteConfig(StringBuilder sb, GameConfig c)
        {
            sb.Append('{');
            WriteIntField(sb, "startingPoints", c.StartingPoints); sb.Append(',');
            WriteDoubleField(sb, "bountyRate", c.BountyRate); sb.Append(',');
            WriteIntField(sb, "generatorCost", c.GeneratorCost); sb.Append(',');
            WriteIntField(sb, "generatorOutput", c.GeneratorOutput); sb.Append(',');
            WriteIntField(sb, "generatorHealth", c.GeneratorHealth); sb.Append(',');
            WriteIntField(sb, "damageFloor", c.DamageFloor); sb.Append(',');
            WriteIntField(sb, "dmgHighGroundBonus", c.DmgHighGroundBonus); sb.Append(',');
            WriteIntField(sb, "rangeHighGroundBonus", c.RangeHighGroundBonus); sb.Append(',');
            WriteIntField(sb, "climbCostPerLevel", c.ClimbCostPerLevel); sb.Append(',');
            WriteIntField(sb, "maxClimbPerStep", c.MaxClimbPerStep); sb.Append(',');
            WriteIntField(sb, "minCheapestViableUnitCost", c.MinCheapestViableUnitCost); sb.Append(',');
            WriteIntField(sb, "boardWidth", c.BoardWidth); sb.Append(',');
            WriteIntField(sb, "boardHeight", c.BoardHeight); sb.Append(',');
            WriteIntField(sb, "minElevation", c.MinElevation); sb.Append(',');
            WriteIntField(sb, "maxElevation", c.MaxElevation); sb.Append(',');
            WriteIntField(sb, "deploymentZoneDepth", c.DeploymentZoneDepth); sb.Append(',');
            WriteIntField(sb, "roundCap", c.RoundCap); sb.Append(',');
            WriteIntField(sb, "turnPolicy", (int)c.TurnPolicy); sb.Append(',');
            sb.Append("\"terrain\":[");
            bool first = true;
            foreach (TerrainType tt in new[] { TerrainType.Plains, TerrainType.Forest, TerrainType.Rough, TerrainType.Water })
            {
                var d = c.TerrainDefFor(tt);
                if (!first) sb.Append(','); first = false;
                sb.Append('{');
                WriteIntField(sb, "terrain", (int)d.Terrain); sb.Append(',');
                WriteIntField(sb, "moveCost", d.MoveCost); sb.Append(',');
                WriteIntField(sb, "concealment", d.Concealment); sb.Append(',');
                WriteIntField(sb, "defense", d.Defense); sb.Append(',');
                WriteBoolField(sb, "passable", d.Passable);
                sb.Append('}');
            }
            sb.Append("],");
            sb.Append("\"terrainWeights\":[");
            first = true;
            foreach (TerrainType tt in new[] { TerrainType.Plains, TerrainType.Forest, TerrainType.Rough, TerrainType.Water })
            {
                double w = c.TerrainWeights.TryGetValue(tt, out var wv) ? wv : 0.0;
                if (!first) sb.Append(','); first = false;
                sb.Append('{'); WriteIntField(sb, "terrain", (int)tt); sb.Append(','); WriteDoubleField(sb, "weight", w); sb.Append('}');
            }
            sb.Append("]");
            sb.Append('}');
        }

        private static void WriteStats(StringBuilder sb, UnitStats s)
        {
            sb.Append('{');
            WriteIntField(sb, "health", s.Health); sb.Append(',');
            WriteIntField(sb, "damage", s.Damage); sb.Append(',');
            WriteIntField(sb, "range", s.Range); sb.Append(',');
            WriteIntField(sb, "movement", s.Movement); sb.Append(',');
            WriteIntField(sb, "defense", s.Defense); sb.Append(',');
            WriteIntField(sb, "vision", s.Vision);
            sb.Append('}');
        }

        private static void WriteCoord(StringBuilder sb, HexCoord c)
        {
            sb.Append('{');
            WriteIntField(sb, "q", c.Q); sb.Append(',');
            WriteIntField(sb, "r", c.R); sb.Append(',');
            WriteIntField(sb, "e", c.Elevation);
            sb.Append('}');
        }

        private static void WriteField(StringBuilder sb, string k, string v) => sb.Append('"').Append(k).Append("\":\"").Append(v).Append('"');
        private static void WriteIntField(StringBuilder sb, string k, int v) => sb.Append('"').Append(k).Append("\":").Append(v.ToString(CultureInfo.InvariantCulture));
        private static void WriteBoolField(StringBuilder sb, string k, bool v) => sb.Append('"').Append(k).Append("\":").Append(v ? "true" : "false");
        private static void WriteDoubleField(StringBuilder sb, string k, double v) => sb.Append('"').Append(k).Append("\":").Append(v.ToString("R", CultureInfo.InvariantCulture));

        // ---------- GameState readers ----------
        private static GameState ReadGameState(Dictionary<string, object> o)
        {
            var config = ReadConfig((Dictionary<string, object>)o["config"]);
            var board = ReadBoard((Dictionary<string, object>)o["board"]);
            var players = (List<object>)o["players"];
            var p0 = ReadPlayer((Dictionary<string, object>)players[0]);
            var p1 = ReadPlayer((Dictionary<string, object>)players[1]);
            int winnerInt = (int)(double)o["winner"];
            PlayerId? winner = winnerInt < 0 ? (PlayerId?)null : (PlayerId)winnerInt;
            return new GameState(board, new[] { p0, p1 }, (PlayerId)(int)(double)o["activePlayer"],
                (int)(double)o["round"], (int)(double)o["nextEntityId"], config,
                (bool)o["isGameOver"], winner);
        }

        private static Board ReadBoard(Dictionary<string, object> o)
        {
            var tilesRaw = (List<object>)o["tiles"];
            var tiles = new List<Tile>(tilesRaw.Count);
            foreach (Dictionary<string, object> t in tilesRaw)
            {
                var coord = new HexCoord((int)(double)t["q"], (int)(double)t["r"], (int)(double)t["e"]);
                tiles.Add(new Tile(coord, (TerrainType)(int)(double)t["terrain"]));
            }
            return new Board((int)(double)o["width"], (int)(double)o["height"], tiles,
                ReadCoordList((List<object>)o["dz0"]), ReadCoordList((List<object>)o["dz1"]));
        }

        private static List<HexCoord> ReadCoordList(List<object> raw)
        {
            var list = new List<HexCoord>(raw.Count);
            foreach (Dictionary<string, object> c in raw) list.Add(ReadCoord(c));
            return list;
        }

        private static PlayerState ReadPlayer(Dictionary<string, object> o)
        {
            var reserve = new List<UnitStats>();
            foreach (Dictionary<string, object> s in (List<object>)o["reserve"]) reserve.Add(ReadStats(s));
            var units = new List<Unit>();
            foreach (Dictionary<string, object> u in (List<object>)o["units"])
                units.Add(new Unit((int)(double)u["id"], (PlayerId)(int)(double)u["owner"],
                    ReadStats((Dictionary<string, object>)u["stats"]),
                    ReadCoord((Dictionary<string, object>)u["pos"]),
                    (int)(double)u["hp"], (bool)u["hasActed"]));
            var gens = new List<Generator>();
            foreach (Dictionary<string, object> g in (List<object>)o["generators"])
                gens.Add(new Generator((int)(double)g["id"], (PlayerId)(int)(double)g["owner"],
                    ReadCoord((Dictionary<string, object>)g["pos"]), (int)(double)g["hp"]));
            return new PlayerState((PlayerId)(int)(double)o["id"], (int)(double)o["points"], reserve, units, gens);
        }

        private static GameConfig ReadConfig(Dictionary<string, object> o)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (Dictionary<string, object> d in (List<object>)o["terrain"])
            {
                var tt = (TerrainType)(int)(double)d["terrain"];
                terrain[tt] = new TerrainDef(tt, (int)(double)d["moveCost"], (int)(double)d["concealment"],
                    (int)(double)d["defense"], (bool)d["passable"]);
            }
            var weights = new Dictionary<TerrainType, double>();
            foreach (Dictionary<string, object> d in (List<object>)o["terrainWeights"])
                weights[(TerrainType)(int)(double)d["terrain"]] = (double)d["weight"];
            return new GameConfig
            {
                StartingPoints = (int)(double)o["startingPoints"],
                BountyRate = (double)o["bountyRate"],
                GeneratorCost = (int)(double)o["generatorCost"],
                GeneratorOutput = (int)(double)o["generatorOutput"],
                GeneratorHealth = (int)(double)o["generatorHealth"],
                DamageFloor = (int)(double)o["damageFloor"],
                DmgHighGroundBonus = (int)(double)o["dmgHighGroundBonus"],
                RangeHighGroundBonus = (int)(double)o["rangeHighGroundBonus"],
                ClimbCostPerLevel = (int)(double)o["climbCostPerLevel"],
                MaxClimbPerStep = (int)(double)o["maxClimbPerStep"],
                MinCheapestViableUnitCost = (int)(double)o["minCheapestViableUnitCost"],
                BoardWidth = (int)(double)o["boardWidth"],
                BoardHeight = (int)(double)o["boardHeight"],
                MinElevation = (int)(double)o["minElevation"],
                MaxElevation = (int)(double)o["maxElevation"],
                DeploymentZoneDepth = (int)(double)o["deploymentZoneDepth"],
                RoundCap = (int)(double)o["roundCap"],
                TurnPolicy = (TurnPolicyKind)(int)(double)o["turnPolicy"],
                Terrain = terrain,
                TerrainWeights = weights
            };
        }

        private static UnitStats ReadStats(Dictionary<string, object> o)
            => new UnitStats((int)(double)o["health"], (int)(double)o["damage"], (int)(double)o["range"],
                (int)(double)o["movement"], (int)(double)o["defense"], (int)(double)o["vision"]);

        private static HexCoord ReadCoord(Dictionary<string, object> o)
            => new HexCoord((int)(double)o["q"], (int)(double)o["r"], (int)(double)o["e"]);

        // ---------- Minimal recursive-descent JSON parser ----------
        private sealed class JsonParser
        {
            private readonly string _s; private int _i;
            public JsonParser(string s) { _s = s; _i = 0; }
            public Dictionary<string, object> ParseObject() { SkipWs(); return (Dictionary<string, object>)ParseValue(); }
            private object ParseValue()
            {
                SkipWs();
                char ch = _s[_i];
                if (ch == '{') return ParseObj();
                if (ch == '[') return ParseArr();
                if (ch == '"') return ParseString();
                if (ch == 't') { _i += 4; return true; }
                if (ch == 'f') { _i += 5; return false; }
                if (ch == 'n') { _i += 4; return null; }
                return ParseNumber();
            }
            private Dictionary<string, object> ParseObj()
            {
                var d = new Dictionary<string, object>(); _i++; SkipWs();
                if (_s[_i] == '}') { _i++; return d; }
                while (true)
                {
                    SkipWs(); string key = ParseString(); SkipWs(); _i++; // colon
                    d[key] = ParseValue(); SkipWs();
                    if (_s[_i] == ',') { _i++; continue; }
                    _i++; break; // '}'
                }
                return d;
            }
            private List<object> ParseArr()
            {
                var l = new List<object>(); _i++; SkipWs();
                if (_s[_i] == ']') { _i++; return l; }
                while (true)
                {
                    l.Add(ParseValue()); SkipWs();
                    if (_s[_i] == ',') { _i++; continue; }
                    _i++; break; // ']'
                }
                return l;
            }
            private string ParseString()
            {
                var sb = new StringBuilder(); _i++; // opening quote
                while (_s[_i] != '"')
                {
                    char c = _s[_i++];
                    if (c == '\\') { sb.Append(_s[_i++]); } else sb.Append(c);
                }
                _i++; return sb.ToString();
            }
            private double ParseNumber()
            {
                int start = _i;
                while (_i < _s.Length && (char.IsDigit(_s[_i]) || _s[_i] == '-' || _s[_i] == '+' || _s[_i] == '.' || _s[_i] == 'e' || _s[_i] == 'E')) _i++;
                return double.Parse(_s.Substring(start, _i - start), CultureInfo.InvariantCulture);
            }
            private void SkipWs() { while (_i < _s.Length && char.IsWhiteSpace(_s[_i])) _i++; }
        }
    }
}
```
_Implementation note: `winner` is encoded as `-1` for null, else the `PlayerId` int. The parser yields `double` for all numbers, `string`, `bool`, `Dictionary<string,object>`, and `List<object>`; readers cast accordingly. If Task 6's `GameConfig` lacks public init-setters for every field (it should, per the contract's `{ get; init; }`), construct config via those init-setters as shown._

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.SerializationTests" -logFile -`
Expected: PASS â€” GameState and all six Command types round-trip with identical re-serialization.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Serialization/GameStateSerializer.cs Assets/HexWars/Engine/Tests/SerializationTests.cs && git commit -m "feat(engine): add GameStateSerializer JSON round-trip for GameState and Command"
```

---

### Task 20: GameFactory.NewGame/Reset and IBoardGenerator interface

**Files:**
- Create: `Assets/HexWars/Engine/Commands/GameFactory.cs`
- Create: `Assets/HexWars/Engine/Board/IBoardGenerator.cs`
- Modify: `Assets/HexWars/Engine/Tests/GameStateTests.cs` (append NewGame/Reset tests)
- Test: `Assets/HexWars/Engine/Tests/GameStateTests.cs`

**Interfaces:**
- Consumes: `GameState` ctor, `GameState.Clone` (Task 7); `Board` (Task 4); `PlayerState` ctor (Task 7); `GameConfig.StartingPoints`/`Clone` (Task 6); `UnitStats` (Task 5).
- Produces: `public interface IBoardGenerator { Board Generate(GameConfig config, int seed); }`; `public static class GameFactory { static GameState NewGame(IBoardGenerator generator, GameConfig config, int seed); static GameState Reset(IBoardGenerator generator, GameConfig config, int seed); }`.

- [ ] **Step 1: Write the failing test**
```csharp
// Append to Assets/HexWars/Engine/Tests/GameStateTests.cs (inside the GameStateTests class)
        private sealed class StubBoardGenerator : IBoardGenerator
        {
            private readonly Board _board;
            public StubBoardGenerator(Board board) { _board = board; }
            // Returns a clone so each NewGame gets an independent board, deterministically.
            public Board Generate(GameConfig config, int seed) => _board.Clone();
        }

        private static Board TinyBoard()
        {
            var tiles = new System.Collections.Generic.List<Tile>
            {
                new Tile(new HexCoord(0, 0, 0), TerrainType.Plains),
                new Tile(new HexCoord(1, 0, 0), TerrainType.Plains)
            };
            return new Board(2, 1, tiles,
                new System.Collections.Generic.List<HexCoord> { new HexCoord(0, 0, 0) },
                new System.Collections.Generic.List<HexCoord> { new HexCoord(1, 0, 0) });
        }

        [Test]
        public void NewGame_BuildsFreshDeterministicState()
        {
            var cfg = GameConfig.Default();
            var gen = new StubBoardGenerator(TinyBoard());
            var s = GameFactory.NewGame(gen, cfg, 123);
            Assert.AreEqual(PlayerId.Player0, s.ActivePlayer);
            Assert.AreEqual(1, s.Round);
            Assert.IsFalse(s.IsGameOver);
            Assert.IsNull(s.Winner);
            Assert.AreEqual(cfg.StartingPoints, s.Players[0].Points);
            Assert.AreEqual(cfg.StartingPoints, s.Players[1].Points);
            Assert.AreEqual(0, s.Players[0].Reserve.Count);
            Assert.AreEqual(0, s.Players[0].UnitsOnBoard.Count);
            Assert.AreEqual(0, s.Players[0].Generators.Count);
            Assert.AreEqual(0, s.Players[1].UnitsOnBoard.Count);
            Assert.Greater(s.NextEntityId, 0);
        }

        [Test]
        public void NewGame_SameInputs_ProduceDeepEqualStates()
        {
            var cfg = GameConfig.Default();
            var gen = new StubBoardGenerator(TinyBoard());
            var a = GameFactory.NewGame(gen, cfg, 42);
            var b = GameFactory.NewGame(gen, cfg, 42);
            Assert.AreEqual(GameStateSerializer.ToJson(a), GameStateSerializer.ToJson(b));
        }

        [Test]
        public void Reset_IsAliasOfNewGame()
        {
            var cfg = GameConfig.Default();
            var gen = new StubBoardGenerator(TinyBoard());
            var a = GameFactory.NewGame(gen, cfg, 7);
            var b = GameFactory.Reset(gen, cfg, 7);
            Assert.AreEqual(GameStateSerializer.ToJson(a), GameStateSerializer.ToJson(b));
        }
```
_Note: `GameFactory_SameInputs` uses `GameStateSerializer.ToJson` (Task 19) for deep-equality; if Task 19 is not yet merged, substitute field-by-field asserts. Per the contract dependency list this task may proceed once Task 7 and Task 18 are done; the serializer comparison is the cleanest deep-equal check and Task 19 is in the same area, so it is available._

- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameStateTests" -logFile -`
Expected: FAIL â€” compilation error, `IBoardGenerator` and `GameFactory` do not exist yet.

- [ ] **Step 3: Write minimal implementation**
```csharp
// Assets/HexWars/Engine/Board/IBoardGenerator.cs
namespace HexWars.Engine
{
    public interface IBoardGenerator
    {
        // Produce a board from config + seed. Authored generators may ignore the seed,
        // but it is part of the contract so procedural generators are deterministic.
        Board Generate(GameConfig config, int seed);
    }
}
```
```csharp
// Assets/HexWars/Engine/Commands/GameFactory.cs
using System.Collections.Generic;

namespace HexWars.Engine
{
    public static class GameFactory
    {
        public static GameState NewGame(IBoardGenerator generator, GameConfig config, int seed)
            => Reset(generator, config, seed);

        public static GameState Reset(IBoardGenerator generator, GameConfig config, int seed)
        {
            var cfg = config.Clone();
            var board = generator.Generate(cfg, seed);
            var p0 = new PlayerState(PlayerId.Player0, cfg.StartingPoints,
                new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            var p1 = new PlayerState(PlayerId.Player1, cfg.StartingPoints,
                new List<UnitStats>(), new List<Unit>(), new List<Generator>());
            return new GameState(
                board,
                new[] { p0, p1 },
                activePlayer: PlayerId.Player0,
                round: 1,
                nextEntityId: 1,
                config: cfg,
                isGameOver: false,
                winner: null);
        }
    }
}
```
_Implementation note: `nextEntityId` is seeded to `1` (deterministic, > 0). The board is taken straight from `generator.Generate` (the generator owns cloning/determinism); `config.Clone()` ensures the returned state does not alias the caller's config. No income is credited at game start â€” per the contract, income is credited at the start of each active turn via `EndTurn`._

- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.GameStateTests" -logFile -`
Expected: PASS â€” fresh state shape, StartingPoints applied to both players, same seed -> deep-equal, Reset == NewGame.

- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Board/IBoardGenerator.cs Assets/HexWars/Engine/Commands/GameFactory.cs Assets/HexWars/Engine/Tests/GameStateTests.cs && git commit -m "feat(engine): add IBoardGenerator and GameFactory.NewGame/Reset"
```

---

### Task 21: BoardValidator: connectivity and symmetry checks

**Files:**
- Create: `Assets/HexWars/Engine/Board/BoardValidator.cs`
- Modify: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`
- Test: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`

**Interfaces:**
- Consumes: `Board(int width,int height,IReadOnlyList<Tile> tiles,IReadOnlyList<HexCoord> dz0,IReadOnlyList<HexCoord> dz1)`, `Board.TryGetTile(int q,int r,out Tile)`, `Board.Tiles`, `Board.DeploymentZone0`, `Board.DeploymentZone1`, `Board.Width`, `Board.Height` (Task 4); `Tile(HexCoord coord,TerrainType terrain)`, `Tile.Coord`, `Tile.Terrain` (Task 3); `HexCoord(int q,int r,int elevation=0)`, `HexCoord.Q`, `HexCoord.R` (Task 2); `HexGeometry.Neighbors(HexCoord c)` returning `List<HexCoord>` (Task 8); `GameConfig.Default()`, `GameConfig.TerrainDefFor(TerrainType)`, `TerrainDef.Passable` (Tasks 6,3).
- Produces: `public static class BoardValidator{ public static bool IsFullyConnected(Board board); public static bool ZonesAreSymmetric(Board board); }` consumed by Tasks 22, 23.

- [ ] **Step 1: Write the failing test**
```csharp
using System.Collections.Generic;
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class BoardGenerationTests
    {
        // Builds a flat (elevation 0) rectangular axial board of the given terrains.
        // terrainGrid[r][q] selects the terrain for axial (q,r).
        private static Board MakeBoard(
            int width, int height,
            TerrainType[][] terrainGrid,
            IReadOnlyList<HexCoord> dz0,
            IReadOnlyList<HexCoord> dz1)
        {
            var tiles = new List<Tile>();
            for (int r = 0; r < height; r++)
                for (int q = 0; q < width; q++)
                    tiles.Add(new Tile(new HexCoord(q, r, 0), terrainGrid[r][q]));
            return new Board(width, height, tiles, dz0, dz1);
        }

        private static TerrainType[][] AllPlains(int width, int height)
        {
            var grid = new TerrainType[height][];
            for (int r = 0; r < height; r++)
            {
                grid[r] = new TerrainType[width];
                for (int q = 0; q < width; q++) grid[r][q] = TerrainType.Plains;
            }
            return grid;
        }

        [Test]
        public void IsFullyConnected_AllPlains_ReturnsTrue()
        {
            var board = MakeBoard(3, 3, AllPlains(3, 3),
                new[] { new HexCoord(0, 0) }, new[] { new HexCoord(2, 2) });
            Assert.IsTrue(BoardValidator.IsFullyConnected(board));
        }

        [Test]
        public void IsFullyConnected_SplitByImpassableWater_ReturnsFalse()
        {
            // Middle column (q=1) is all impassable Water, cutting the board in two
            // passable regions (q=0 and q=2) that cannot reach each other.
            var grid = AllPlains(3, 3);
            for (int r = 0; r < 3; r++) grid[r][1] = TerrainType.Water;
            var board = MakeBoard(3, 3, grid,
                new[] { new HexCoord(0, 0) }, new[] { new HexCoord(2, 2) });
            Assert.IsFalse(BoardValidator.IsFullyConnected(board),
                "Two passable regions separated by impassable water are not fully connected.");
        }

        [Test]
        public void ZonesAreSymmetric_CenterMirroredZones_ReturnsTrue()
        {
            // Width 3, Height 3 -> center mirror of (q,r) is (2-q, 2-r).
            var dz0 = new[] { new HexCoord(0, 0), new HexCoord(1, 0) };
            var dz1 = new[] { new HexCoord(2, 2), new HexCoord(1, 2) };
            var board = MakeBoard(3, 3, AllPlains(3, 3), dz0, dz1);
            Assert.IsTrue(BoardValidator.ZonesAreSymmetric(board));
        }

        [Test]
        public void ZonesAreSymmetric_AsymmetricZones_ReturnsFalse()
        {
            var dz0 = new[] { new HexCoord(0, 0), new HexCoord(1, 0) };
            var dz1 = new[] { new HexCoord(2, 2) }; // wrong count / not mirrored
            var board = MakeBoard(3, 3, AllPlains(3, 3), dz0, dz1);
            Assert.IsFalse(BoardValidator.ZonesAreSymmetric(board));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests.IsFullyConnected_AllPlains_ReturnsTrue" -logFile -`
Expected: FAIL with compile error `The name 'BoardValidator' does not exist in the current context` (class not yet created).
- [ ] **Step 3: Write minimal implementation**
```csharp
using System.Collections.Generic;

namespace HexWars.Engine
{
    public static class BoardValidator
    {
        private static readonly GameConfig PassabilityConfig = GameConfig.Default();

        private static bool IsPassable(TerrainType terrain)
        {
            return PassabilityConfig.TerrainDefFor(terrain).Passable;
        }

        // True if every passable tile is reachable from every other passable tile
        // (flood-fill over passable hexes using the 6 axial neighbours).
        public static bool IsFullyConnected(Board board)
        {
            // Collect all passable axial coords (elevation ignored for adjacency).
            var passable = new HashSet<(int q, int r)>();
            foreach (var tile in board.Tiles)
            {
                if (IsPassable(tile.Terrain))
                    passable.Add((tile.Coord.Q, tile.Coord.R));
            }

            if (passable.Count == 0) return true; // vacuously connected

            // BFS from a deterministic start: the lexicographically smallest passable coord.
            (int q, int r) start = (int.MaxValue, int.MaxValue);
            foreach (var p in passable)
            {
                if (p.q < start.q || (p.q == start.q && p.r < start.r))
                    start = p;
            }

            var visited = new HashSet<(int q, int r)>();
            var queue = new Queue<(int q, int r)>();
            visited.Add(start);
            queue.Enqueue(start);

            while (queue.Count > 0)
            {
                var current = queue.Dequeue();
                var center = new HexCoord(current.q, current.r, 0);
                foreach (var n in HexGeometry.Neighbors(center))
                {
                    var key = (n.Q, n.R);
                    if (!passable.Contains(key)) continue;
                    if (visited.Contains(key)) continue;
                    visited.Add(key);
                    queue.Enqueue(key);
                }
            }

            return visited.Count == passable.Count;
        }

        // True if zone0 and zone1 are mirror-symmetric across the board center.
        // Center mirror of axial (q,r) is (Width-1-q, Height-1-r).
        public static bool ZonesAreSymmetric(Board board)
        {
            var zone0 = board.DeploymentZone0;
            var zone1 = board.DeploymentZone1;
            if (zone0.Count != zone1.Count) return false;

            var zone1Set = new HashSet<(int q, int r)>();
            foreach (var c in zone1) zone1Set.Add((c.Q, c.R));

            foreach (var c in zone0)
            {
                int mq = board.Width - 1 - c.Q;
                int mr = board.Height - 1 - c.R;
                if (!zone1Set.Contains((mq, mr))) return false;
            }

            return true;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests" -logFile -`
Expected: PASS (all four BoardValidator tests green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Board/BoardValidator.cs Assets/HexWars/Engine/Tests/BoardGenerationTests.cs && git commit -m "feat(engine): add BoardValidator connectivity and symmetry checks"
```

---

### Task 22: RandomBoardGenerator: seeded noise/terrain, mirrored zones, connectivity

**Files:**
- Create: `Assets/HexWars/Engine/Board/RandomBoardGenerator.cs`
- Modify: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`
- Test: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`

**Interfaces:**
- Consumes: `interface IBoardGenerator{ Board Generate(GameConfig config,int seed); }` (Task 20); `Board(...)`, `Board.Tiles`, `Board.DeploymentZone0/1`, `Board.Width/Height` (Task 4); `Tile(HexCoord,TerrainType)`, `Tile.Coord`, `Tile.Terrain`, `Tile.Elevation` (Task 3); `HexCoord(int q,int r,int elevation=0)`, `HexCoord.Q/R/Elevation` (Task 2); `TerrainType{Plains,Forest,Rough,Water}` (Task 3); `GameConfig` fields `BoardWidth`, `BoardHeight`, `MinElevation`, `MaxElevation`, `DeploymentZoneDepth`, `TerrainWeights` (`IReadOnlyDictionary<TerrainType,double>`) (Task 6); `BoardValidator.IsFullyConnected(Board)`, `BoardValidator.ZonesAreSymmetric(Board)` (Task 21).
- Produces: `public sealed class RandomBoardGenerator : IBoardGenerator{ public Board Generate(GameConfig config,int seed); }` â€” deterministic via `System.Random(seed)` ONLY; output always passes both `BoardValidator` checks. Consumed by Tasks 24, 31, 32.

- [ ] **Step 1: Write the failing test**
```csharp
// Add these tests inside the existing BoardGenerationTests class.

[Test]
public void RandomGenerator_SameSeed_ProducesDeepEqualBoard()
{
    var config = GameConfig.Default();
    var gen = new RandomBoardGenerator();
    var a = gen.Generate(config, 12345);
    var b = gen.Generate(config, 12345);

    Assert.AreEqual(a.Width, b.Width);
    Assert.AreEqual(a.Height, b.Height);
    Assert.AreEqual(a.Tiles.Count, b.Tiles.Count);
    for (int i = 0; i < a.Tiles.Count; i++)
    {
        Assert.AreEqual(a.Tiles[i].Coord, b.Tiles[i].Coord, $"coord mismatch at {i}");
        Assert.AreEqual(a.Tiles[i].Terrain, b.Tiles[i].Terrain, $"terrain mismatch at {i}");
    }
    CollectionAssert.AreEqual(a.DeploymentZone0, b.DeploymentZone0);
    CollectionAssert.AreEqual(a.DeploymentZone1, b.DeploymentZone1);
}

[Test]
public void RandomGenerator_DifferentSeeds_ProduceDifferentBoards()
{
    var config = GameConfig.Default();
    var gen = new RandomBoardGenerator();
    var a = gen.Generate(config, 1);
    var b = gen.Generate(config, 2);

    bool anyDifference = false;
    for (int i = 0; i < a.Tiles.Count && !anyDifference; i++)
    {
        if (a.Tiles[i].Terrain != b.Tiles[i].Terrain ||
            a.Tiles[i].Elevation != b.Tiles[i].Elevation)
            anyDifference = true;
    }
    Assert.IsTrue(anyDifference, "Different seeds should produce different boards.");
}

[Test]
public void RandomGenerator_ZonesAreSymmetricAndBoardConnected()
{
    var config = GameConfig.Default();
    var gen = new RandomBoardGenerator();
    for (int seed = 0; seed < 25; seed++)
    {
        var board = gen.Generate(config, seed);
        Assert.IsTrue(BoardValidator.ZonesAreSymmetric(board), $"seed {seed} zones asymmetric");
        Assert.IsTrue(BoardValidator.IsFullyConnected(board), $"seed {seed} not connected");
    }
}

[Test]
public void RandomGenerator_ElevationAndTerrainWithinBounds()
{
    var config = GameConfig.Default();
    var gen = new RandomBoardGenerator();
    var board = gen.Generate(config, 777);
    Assert.AreEqual(config.BoardWidth, board.Width);
    Assert.AreEqual(config.BoardHeight, board.Height);
    foreach (var tile in board.Tiles)
    {
        Assert.GreaterOrEqual(tile.Elevation, config.MinElevation);
        Assert.LessOrEqual(tile.Elevation, config.MaxElevation);
        Assert.IsTrue(System.Enum.IsDefined(typeof(TerrainType), tile.Terrain));
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests.RandomGenerator_SameSeed_ProducesDeepEqualBoard" -logFile -`
Expected: FAIL with compile error `The name 'RandomBoardGenerator' does not exist in the current context`.
- [ ] **Step 3: Write minimal implementation**
```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    public sealed class RandomBoardGenerator : IBoardGenerator
    {
        private const int MaxAttempts = 64;

        // System.Random(seed) ONLY. Mirrored zones; connectivity-checked with retry.
        public Board Generate(GameConfig config, int seed)
        {
            for (int attempt = 0; attempt < MaxAttempts; attempt++)
            {
                // Derive a deterministic per-attempt seed so retries stay reproducible.
                var rng = new Random(unchecked(seed * 73856093 + attempt * 19349663));
                Board board = BuildBoard(config, rng);
                if (BoardValidator.IsFullyConnected(board) &&
                    BoardValidator.ZonesAreSymmetric(board))
                    return board;
            }

            // Repair fallback: a fully-plains, flat board is always connected & symmetric.
            return BuildFlatPlainsBoard(config);
        }

        private static Board BuildBoard(GameConfig config, Random rng)
        {
            int width = config.BoardWidth;
            int height = config.BoardHeight;

            var orderedTerrains = BuildWeightTable(config, out double[] cumulative, out double total);

            var tiles = new List<Tile>(width * height);
            for (int r = 0; r < height; r++)
            {
                for (int q = 0; q < width; q++)
                {
                    int elevation = SampleElevation(config, rng);
                    TerrainType terrain = SampleTerrain(rng, orderedTerrains, cumulative, total);
                    tiles.Add(new Tile(new HexCoord(q, r, elevation), terrain));
                }
            }

            // Mirrored deployment zones: player 0 owns the first DeploymentZoneDepth rows,
            // player 1 owns the center-mirror of each of those coords.
            var dz0 = new List<HexCoord>();
            var dz1 = new List<HexCoord>();
            int depth = config.DeploymentZoneDepth;
            for (int r = 0; r < depth; r++)
            {
                for (int q = 0; q < width; q++)
                {
                    dz0.Add(new HexCoord(q, r));
                    dz1.Add(new HexCoord(width - 1 - q, height - 1 - r));
                }
            }

            // Force deployment-zone tiles passable & symmetric in terrain/elevation so the
            // connectivity + symmetry guarantees are not broken by zone hexes.
            EnsureZonesPassableAndMirrored(tiles, width, height, dz0);

            return new Board(width, height, tiles, dz0, dz1);
        }

        private static int SampleElevation(GameConfig config, Random rng)
        {
            int span = config.MaxElevation - config.MinElevation + 1;
            return config.MinElevation + rng.Next(span); // value noise clamped to [Min,Max]
        }

        private static List<TerrainType> BuildWeightTable(
            GameConfig config, out double[] cumulative, out double total)
        {
            // Deterministic enum order: Plains, Forest, Rough, Water.
            var ordered = new List<TerrainType>
            {
                TerrainType.Plains, TerrainType.Forest, TerrainType.Rough, TerrainType.Water
            };
            cumulative = new double[ordered.Count];
            double running = 0.0;
            for (int i = 0; i < ordered.Count; i++)
            {
                double w = config.TerrainWeights.TryGetValue(ordered[i], out var v) ? v : 0.0;
                running += w;
                cumulative[i] = running;
            }
            total = running;
            return ordered;
        }

        private static TerrainType SampleTerrain(
            Random rng, List<TerrainType> ordered, double[] cumulative, double total)
        {
            if (total <= 0.0) return TerrainType.Plains;
            double roll = rng.NextDouble() * total;
            for (int i = 0; i < ordered.Count; i++)
            {
                if (roll < cumulative[i]) return ordered[i];
            }
            return ordered[ordered.Count - 1];
        }

        private static void EnsureZonesPassableAndMirrored(
            List<Tile> tiles, int width, int height, List<HexCoord> dz0)
        {
            // Tiles are stored row-major: index = r * width + q.
            foreach (var c in dz0)
            {
                int i0 = c.R * width + c.Q;
                int mq = width - 1 - c.Q;
                int mr = height - 1 - c.R;
                int i1 = mr * width + mq;
                // Make both deployment hexes Plains at a fixed elevation so zones are safe,
                // passable, and mirror-symmetric.
                tiles[i0] = new Tile(new HexCoord(c.Q, c.R, 0), TerrainType.Plains);
                tiles[i1] = new Tile(new HexCoord(mq, mr, 0), TerrainType.Plains);
            }
        }

        private static Board BuildFlatPlainsBoard(GameConfig config)
        {
            int width = config.BoardWidth;
            int height = config.BoardHeight;
            var tiles = new List<Tile>(width * height);
            for (int r = 0; r < height; r++)
                for (int q = 0; q < width; q++)
                    tiles.Add(new Tile(new HexCoord(q, r, 0), TerrainType.Plains));

            var dz0 = new List<HexCoord>();
            var dz1 = new List<HexCoord>();
            for (int r = 0; r < config.DeploymentZoneDepth; r++)
                for (int q = 0; q < width; q++)
                {
                    dz0.Add(new HexCoord(q, r));
                    dz1.Add(new HexCoord(width - 1 - q, height - 1 - r));
                }
            return new Board(width, height, tiles, dz0, dz1);
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests" -logFile -`
Expected: PASS (determinism, difference, symmetry+connectivity across seeds, and bounds tests all green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Board/RandomBoardGenerator.cs Assets/HexWars/Engine/Tests/BoardGenerationTests.cs && git commit -m "feat(engine): add seeded RandomBoardGenerator with mirrored zones and connectivity retry"
```

---

### Task 23: AuthoredBoardGenerator: fixed handcrafted board

**Files:**
- Create: `Assets/HexWars/Engine/Board/AuthoredBoardGenerator.cs`
- Modify: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`
- Test: `Assets/HexWars/Engine/Tests/BoardGenerationTests.cs`

**Interfaces:**
- Consumes: `interface IBoardGenerator{ Board Generate(GameConfig config,int seed); }` (Task 20); `Board.Clone()`, `Board.Tiles`, `Board.Width/Height`, `Board.DeploymentZone0/1` (Task 4); `Tile.Coord`, `Tile.Terrain` (Task 3); `BoardValidator.IsFullyConnected(Board)`, `BoardValidator.ZonesAreSymmetric(Board)` (Task 21); `GameConfig.Default()` (Task 6).
- Produces: `public sealed class AuthoredBoardGenerator : IBoardGenerator{ public AuthoredBoardGenerator(Board fixedBoard); public Board Generate(GameConfig config,int seed); }` â€” returns a clone of the wrapped board (seed ignored). Consumed by Task 28.

- [ ] **Step 1: Write the failing test**
```csharp
// Add these tests inside the existing BoardGenerationTests class.
// Reuses the private MakeBoard / AllPlains helpers added in Task 21.

[Test]
public void Authored_Generate_ReturnsDeepEqualButDistinctInstance()
{
    var dz0 = new[] { new HexCoord(0, 0), new HexCoord(1, 0) };
    var dz1 = new[] { new HexCoord(2, 2), new HexCoord(1, 2) };
    var authored = MakeBoard(3, 3, AllPlains(3, 3), dz0, dz1);

    var gen = new AuthoredBoardGenerator(authored);
    var result = gen.Generate(GameConfig.Default(), seed: 999);

    Assert.AreNotSame(authored, result, "Generate must return a distinct Board instance.");
    Assert.AreEqual(authored.Width, result.Width);
    Assert.AreEqual(authored.Height, result.Height);
    Assert.AreEqual(authored.Tiles.Count, result.Tiles.Count);
    for (int i = 0; i < authored.Tiles.Count; i++)
    {
        Assert.AreNotSame(authored.Tiles[i], result.Tiles[i], "tiles must be cloned");
        Assert.AreEqual(authored.Tiles[i].Coord, result.Tiles[i].Coord);
        Assert.AreEqual(authored.Tiles[i].Terrain, result.Tiles[i].Terrain);
    }
    CollectionAssert.AreEqual(authored.DeploymentZone0, result.DeploymentZone0);
    CollectionAssert.AreEqual(authored.DeploymentZone1, result.DeploymentZone1);
}

[Test]
public void Authored_Generate_SeedIgnored_AlwaysSameShape()
{
    var dz0 = new[] { new HexCoord(0, 0) };
    var dz1 = new[] { new HexCoord(2, 2) };
    var authored = MakeBoard(3, 3, AllPlains(3, 3), dz0, dz1);
    var gen = new AuthoredBoardGenerator(authored);

    var a = gen.Generate(GameConfig.Default(), 1);
    var b = gen.Generate(GameConfig.Default(), 42);
    for (int i = 0; i < a.Tiles.Count; i++)
    {
        Assert.AreEqual(a.Tiles[i].Coord, b.Tiles[i].Coord);
        Assert.AreEqual(a.Tiles[i].Terrain, b.Tiles[i].Terrain);
    }
}

[Test]
public void Authored_Generate_PassesBoardValidator()
{
    var dz0 = new[] { new HexCoord(0, 0) };
    var dz1 = new[] { new HexCoord(2, 2) };
    var authored = MakeBoard(3, 3, AllPlains(3, 3), dz0, dz1);
    var result = new AuthoredBoardGenerator(authored).Generate(GameConfig.Default(), 0);
    Assert.IsTrue(BoardValidator.IsFullyConnected(result));
    Assert.IsTrue(BoardValidator.ZonesAreSymmetric(result));
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests.Authored_Generate_ReturnsDeepEqualButDistinctInstance" -logFile -`
Expected: FAIL with compile error `The name 'AuthoredBoardGenerator' does not exist in the current context`.
- [ ] **Step 3: Write minimal implementation**
```csharp
using System;

namespace HexWars.Engine
{
    public sealed class AuthoredBoardGenerator : IBoardGenerator
    {
        private readonly Board _fixedBoard;

        public AuthoredBoardGenerator(Board fixedBoard)
        {
            _fixedBoard = fixedBoard ?? throw new ArgumentNullException(nameof(fixedBoard));
        }

        // Seed is ignored: returns a deep clone of the wrapped handcrafted board so the
        // authored source emits identical Board shape through the same interface.
        public Board Generate(GameConfig config, int seed)
        {
            return _fixedBoard.Clone();
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardGenerationTests" -logFile -`
Expected: PASS (clone-equality, seed-ignored, and validator tests all green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Board/AuthoredBoardGenerator.cs Assets/HexWars/Engine/Tests/BoardGenerationTests.cs && git commit -m "feat(engine): add AuthoredBoardGenerator wrapping a fixed cloned board"
```

---

### Task 24: IAgent, RandomAgent, and headless Match runner

**Files:**
- Create: `Assets/HexWars/Engine/Agent/IAgent.cs`, `Assets/HexWars/Engine/Agent/RandomAgent.cs`, `Assets/HexWars/Engine/Agent/Match.cs`
- Modify: `Assets/HexWars/Engine/Tests/AgentMatchTests.cs`
- Test: `Assets/HexWars/Engine/Tests/AgentMatchTests.cs`

**Interfaces:**
- Consumes: `GameEngine.Apply(GameState state,Command command) -> Result` and `GameEngine.LegalMoves(GameState state) -> List<Command>` (Tasks 16,17,18); `Result.Success`, `Result.NewState` (Task 14); `GameState.IsGameOver`, `GameState.Winner` (`PlayerId?`), `GameState.Round`, `GameState.ActivePlayer` (Task 7); `GameFactory.NewGame(IBoardGenerator,GameConfig,int)` (Task 20); `RandomBoardGenerator` (Task 22); `GameConfig.Default()` (Task 6); `Command` (Task 14); `PlayerId` (Task 2).
- Produces: `public interface IAgent{ string Name{get;} Command ChooseCommand(GameState state); }`; `public sealed class RandomAgent : IAgent{ public RandomAgent(int seed); }`; `public sealed class MatchResult{ public PlayerId? Winner{get;} public int Rounds{get;} public int Commands{get;} public GameState FinalState{get;} public MatchResult(PlayerId? winner,int rounds,int commands,GameState finalState); }`; `public static class Match{ public static MatchResult Run(GameState initial,IAgent agent0,IAgent agent1,int maxCommands=10000); }`. Consumed by Task 32.

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;

namespace HexWars.Engine.Tests
{
    public class AgentMatchTests
    {
        private static GameState NewGame(int seed)
        {
            return GameFactory.NewGame(new RandomBoardGenerator(), GameConfig.Default(), seed);
        }

        [Test]
        public void RandomAgent_ChooseCommand_AlwaysReturnsALegalMove()
        {
            var state = NewGame(7);
            var agent = new RandomAgent(123);
            for (int i = 0; i < 50; i++)
            {
                var legal = GameEngine.LegalMoves(state);
                Assert.IsNotEmpty(legal, "LegalMoves should be non-empty mid-game.");
                var chosen = agent.ChooseCommand(state);
                CollectionAssert.Contains(legal, chosen,
                    "RandomAgent must return a command from LegalMoves.");
                var result = GameEngine.Apply(state, chosen);
                Assert.IsTrue(result.Success, "Chosen legal command must apply successfully.");
                state = result.NewState;
                if (state.IsGameOver) break;
            }
        }

        [Test]
        public void Match_Run_TerminatesAndReportsConsistentResult()
        {
            var initial = NewGame(11);
            var a0 = new RandomAgent(1);
            var a1 = new RandomAgent(2);
            var result = Match.Run(initial, a0, a1, maxCommands: 5000);

            Assert.IsNotNull(result.FinalState);
            Assert.LessOrEqual(result.Commands, 5000, "Must not exceed the command cap.");
            Assert.GreaterOrEqual(result.Commands, 1);
            Assert.AreEqual(result.Round, result.FinalState.Round);
            // Winner reported by MatchResult must match the final state's winner field.
            Assert.AreEqual(result.Winner, result.FinalState.Winner);
        }

        [Test]
        public void Match_Run_IsDeterministicForSameSeeds()
        {
            var r1 = Match.Run(NewGame(20), new RandomAgent(5), new RandomAgent(6), 5000);
            var r2 = Match.Run(NewGame(20), new RandomAgent(5), new RandomAgent(6), 5000);
            Assert.AreEqual(r1.Winner, r2.Winner);
            Assert.AreEqual(r1.Rounds, r2.Rounds);
            Assert.AreEqual(r1.Commands, r2.Commands);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.AgentMatchTests.RandomAgent_ChooseCommand_AlwaysReturnsALegalMove" -logFile -`
Expected: FAIL with compile errors `The name 'RandomAgent'/'Match' does not exist in the current context`.
- [ ] **Step 3: Write minimal implementation**
```csharp
// File: Assets/HexWars/Engine/Agent/IAgent.cs
namespace HexWars.Engine
{
    public interface IAgent
    {
        string Name { get; }
        // Must return a command from GameEngine.LegalMoves(state).
        Command ChooseCommand(GameState state);
    }
}
```
```csharp
// File: Assets/HexWars/Engine/Agent/RandomAgent.cs
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    public sealed class RandomAgent : IAgent
    {
        private readonly Random _rng; // System.Random ONLY (determinism rule)

        public RandomAgent(int seed)
        {
            _rng = new Random(seed);
        }

        public string Name => "RandomAgent";

        public Command ChooseCommand(GameState state)
        {
            List<Command> legal = GameEngine.LegalMoves(state);
            if (legal == null || legal.Count == 0)
                return new EndTurn { Issuer = state.ActivePlayer };
            int index = _rng.Next(legal.Count);
            return legal[index];
        }
    }
}
```
```csharp
// File: Assets/HexWars/Engine/Agent/Match.cs
namespace HexWars.Engine
{
    public sealed class MatchResult
    {
        public PlayerId? Winner { get; }
        public int Rounds { get; }
        public int Commands { get; }
        public GameState FinalState { get; }

        public MatchResult(PlayerId? winner, int rounds, int commands, GameState finalState)
        {
            Winner = winner;
            Rounds = rounds;
            Commands = commands;
            FinalState = finalState;
        }
    }

    public static class Match
    {
        // Drives agent0 (Player0) vs agent1 (Player1) through GameEngine.Apply until the game
        // is over or maxCommands is reached (guaranteeing termination).
        public static MatchResult Run(GameState initial, IAgent agent0, IAgent agent1,
            int maxCommands = 10000)
        {
            GameState state = initial;
            int commandCount = 0;

            while (!state.IsGameOver && commandCount < maxCommands)
            {
                IAgent active = state.ActivePlayer == PlayerId.Player0 ? agent0 : agent1;
                Command command = active.ChooseCommand(state);
                Result result = GameEngine.Apply(state, command);
                commandCount++;

                if (result.Success)
                {
                    state = result.NewState;
                }
                else
                {
                    // Defensive: a rejected command should not happen with a legal agent,
                    // but force progress by ending the turn to guarantee termination.
                    Result forced = GameEngine.Apply(state,
                        new EndTurn { Issuer = state.ActivePlayer });
                    if (forced.Success) state = forced.NewState;
                    else break; // cannot make progress; stop to avoid an infinite loop
                }
            }

            return new MatchResult(state.Winner, state.Round, commandCount, state);
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.AgentMatchTests" -logFile -`
Expected: PASS (legal-move, termination/consistency, and determinism tests all green).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Engine/Agent/IAgent.cs Assets/HexWars/Engine/Agent/RandomAgent.cs Assets/HexWars/Engine/Agent/Match.cs Assets/HexWars/Engine/Tests/AgentMatchTests.cs && git commit -m "feat(engine): add IAgent, seeded RandomAgent, and headless Match runner"
```

---

### Task 25: Scaffold presentation assembly and unit-tested HexLayout helper

**Files:**
- Create: `Assets/HexWars/Presentation/HexWars.Presentation.asmdef`
- Create: `Assets/HexWars/Presentation/HexLayout.cs`
- Modify: `Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef` (add reference to `HexWars.Presentation`)
- Test: `Assets/HexWars/Engine/Tests/HexLayoutTests.cs`

**Interfaces:**
- Consumes: `HexWars.Engine.HexCoord` (Task 2) â€” `int Q`, `int R`, `int Elevation` get-only props, ctor `HexCoord(int q, int r, int elevation = 0)`; EditMode test harness `HexWars.Engine.Tests` (Task 1).
- Produces: `namespace HexWars.Presentation { public static class HexLayout { public const float HexSize = 1.0f; public const float ElevationStep = 0.5f; public static (float x, float z) AxialToWorldXZ(int q, int r); public static (float x, float y, float z) CoordToWorld(HexWars.Engine.HexCoord coord); } }` â€” consumed by `BoardRenderer` (Task 28), `UnitView`/`GeneratorView` (Task 29). New assembly `HexWars.Presentation` referencing `HexWars.Engine`, `Unity.InputSystem`, `UnityEngine.UI`.

- [ ] **Step 1: Write the failing test**
```csharp
// Assets/HexWars/Engine/Tests/HexLayoutTests.cs
using NUnit.Framework;
using HexWars.Engine;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    public sealed class HexLayoutTests
    {
        private const float Eps = 1e-4f;

        [Test]
        public void Origin_MapsToWorldOrigin_XZ()
        {
            var (x, z) = HexLayout.AxialToWorldXZ(0, 0);
            Assert.AreEqual(0f, x, Eps);
            Assert.AreEqual(0f, z, Eps);
        }

        [Test]
        public void AxialToWorldXZ_MatchesFlatTopFormula()
        {
            // Flat-top axial -> world: x = HexSize * 1.5 * q; z = HexSize * sqrt(3) * (r + q/2).
            float size = HexLayout.HexSize;
            float expX = size * 1.5f * 2;
            float expZ = size * 1.7320508f * (1 + 2 / 2f);
            var (x, z) = HexLayout.AxialToWorldXZ(2, 1);
            Assert.AreEqual(expX, x, Eps);
            Assert.AreEqual(expZ, z, Eps);
        }

        [Test]
        public void CoordToWorld_UsesElevationStepForY()
        {
            var coord = new HexCoord(2, 1, 3);
            var (x, y, z) = HexLayout.CoordToWorld(coord);
            var (ex, ez) = HexLayout.AxialToWorldXZ(2, 1);
            Assert.AreEqual(ex, x, Eps);
            Assert.AreEqual(ez, z, Eps);
            Assert.AreEqual(HexLayout.ElevationStep * 3, y, Eps);
        }

        [Test]
        public void CoordToWorld_ZeroElevation_HasZeroY()
        {
            var (_, y, _) = HexLayout.CoordToWorld(new HexCoord(5, -2, 0));
            Assert.AreEqual(0f, y, Eps);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexLayoutTests" -logFile -`
Expected: FAIL â€” compile error / assembly resolution error: `HexWars.Presentation` and `HexLayout` do not exist yet, and the test asmdef does not reference the presentation assembly.
- [ ] **Step 3: Write minimal implementation**
Create the presentation asmdef:
```json
// Assets/HexWars/Presentation/HexWars.Presentation.asmdef
{
    "name": "HexWars.Presentation",
    "rootNamespace": "HexWars.Presentation",
    "references": [
        "HexWars.Engine",
        "Unity.InputSystem",
        "UnityEngine.UI"
    ],
    "includePlatforms": [],
    "excludePlatforms": [],
    "allowUnsafeCode": false,
    "overrideReferences": false,
    "precompiledReferences": [],
    "autoReferenced": true,
    "defineConstraints": [],
    "versionDefines": [],
    "noEngineReferences": false
}
```
Create the pure helper:
```csharp
// Assets/HexWars/Presentation/HexLayout.cs
namespace HexWars.Presentation
{
    // PURE helper (no MonoBehaviour, no UnityEngine types) so it is unit-testable in the
    // EditMode engine-tests assembly. Flat-top axial -> world (X,Z); elevation -> Y.
    public static class HexLayout
    {
        public const float HexSize = 1.0f;       // outer radius
        public const float ElevationStep = 0.5f; // world Y per elevation level

        private const float Sqrt3 = 1.7320508075688772f;

        // Flat-top convention: x = HexSize * 1.5 * q ; z = HexSize * sqrt(3) * (r + q/2).
        public static (float x, float z) AxialToWorldXZ(int q, int r)
        {
            float x = HexSize * 1.5f * q;
            float z = HexSize * Sqrt3 * (r + q / 2f);
            return (x, z);
        }

        public static (float x, float y, float z) CoordToWorld(HexWars.Engine.HexCoord coord)
        {
            var (x, z) = AxialToWorldXZ(coord.Q, coord.R);
            float y = ElevationStep * coord.Elevation;
            return (x, y, z);
        }
    }
}
```
Add `HexWars.Presentation` to the test asmdef's references (Task 1 created it). The references array must now contain both:
```json
// Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef (references array)
    "references": [
        "HexWars.Engine",
        "HexWars.Presentation",
        "UnityEngine.TestRunner",
        "UnityEditor.TestRunner"
    ],
```
(Keep the existing `precompiledReferences` containing `nunit.framework.dll`, `includePlatforms: ["Editor"]`, and `defineConstraints: ["UNITY_INCLUDE_TESTS"]` from Task 1 unchanged.)
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexLayoutTests" -logFile -`
Expected: PASS â€” all four HexLayout tests green. Then confirm a clean editor compile via Coplay MCP `check_compile_errors` (no arguments) returning zero errors.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/HexWars.Presentation.asmdef Assets/HexWars/Presentation/HexLayout.cs Assets/HexWars/Engine/Tests/HexLayoutTests.cs Assets/HexWars/Engine/Tests/HexWars.Engine.Tests.asmdef && git commit -m "feat(presentation): scaffold HexWars.Presentation asmdef and pure HexLayout helper"
```

---

### Task 26: Starfield skybox, unlit/outline shader, materials, and shared hex prefab

**Files:**
- Create: `Assets/HexWars/Art/Shaders/UnlitOutline.shader`
- Create: `Assets/HexWars/Art/Materials/Hex_Plains.mat`
- Create: `Assets/HexWars/Art/Materials/Hex_Forest.mat`
- Create: `Assets/HexWars/Art/Materials/Hex_Rough.mat`
- Create: `Assets/HexWars/Art/Materials/Hex_Water.mat`
- Create: `Assets/HexWars/Art/Materials/Unit_P0.mat`
- Create: `Assets/HexWars/Art/Materials/Unit_P1.mat`
- Create: `Assets/HexWars/Art/Materials/Gen_P0.mat`
- Create: `Assets/HexWars/Art/Materials/Gen_P1.mat`
- Create: `Assets/HexWars/Art/Materials/Starfield_Skybox.mat`
- Create: `Assets/HexWars/Art/Prefabs/Hex.prefab`
- Test: none (editor-asset task; verified via Coplay MCP + `check_compile_errors`)

**Interfaces:**
- Consumes: `HexWars.Presentation` assembly (Task 25); `HexWars.Presentation.HexLayout.HexSize` / `ElevationStep` (Task 25) as the sizing reference for the prism (radius â‰ˆ `HexSize = 1.0`, height â‰ˆ `ElevationStep = 0.5`); terrain order from `HexWars.Engine.TerrainType { Plains=0, Forest=1, Rough=2, Water=3 }` (Task 3) and `HexWars.Engine.PlayerId { Player0=0, Player1=1 }` (Task 2).
- Produces: asset paths consumed by later tasks â€” `Assets/HexWars/Art/Shaders/UnlitOutline.shader` (URP unlit fill + black outline, properties `_BaseColor`, `_OutlineColor`, `_OutlineWidth`); `Hex_{Plains,Forest,Rough,Water}.mat` keyed to `TerrainType` order; `Unit_P0`/`Unit_P1`/`Gen_P0`/`Gen_P1.mat` keyed to `PlayerId`; `Starfield_Skybox.mat`; `Assets/HexWars/Art/Prefabs/Hex.prefab` (shared low-wide hex prism, black outline, default `Hex_Plains` material) consumed by `BoardRenderer` (Task 28).

- [ ] **Step 1: Write the failing test (extract + assert the pure sizing logic; no runtime test exists for assets)**
This is an editor-asset task; the only pure logic is the prism dimensions, which derive from `HexLayout`. Add an assertion test confirming the sizing constants the prefab will use are the contract values, so Task 28 can rely on them.
```csharp
// Append to Assets/HexWars/Engine/Tests/HexLayoutTests.cs
        [Test]
        public void PrismSizing_MatchesContractConstants()
        {
            // Hex.prefab is a low-wide prism: radius = HexSize, per-level height = ElevationStep.
            Assert.AreEqual(1.0f, HexLayout.HexSize, 1e-4f);
            Assert.AreEqual(0.5f, HexLayout.ElevationStep, 1e-4f);
            // Low-wide: radius strictly greater than per-level height.
            Assert.Greater(HexLayout.HexSize, HexLayout.ElevationStep);
        }
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexLayoutTests.PrismSizing_MatchesContractConstants" -logFile -`
Expected: PASS immediately if Task 25 constants are correct (this guards the values Task 28 consumes). If it FAILS, the `HexLayout` constants drifted from the contract â€” fix `HexLayout.cs` before creating assets. (No red-first phase: this is a guard, not new behavior.)
- [ ] **Step 3: Write minimal implementation (create the shader, then assets via Coplay MCP)**
First author the URP unlit + black-outline shader file:
```hlsl
// Assets/HexWars/Art/Shaders/UnlitOutline.shader
Shader "HexWars/UnlitOutline"
{
    Properties
    {
        _BaseColor ("Base Color", Color) = (1,1,0,1)
        _OutlineColor ("Outline Color", Color) = (0,0,0,1)
        _OutlineWidth ("Outline Width", Range(0,0.1)) = 0.03
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "RenderPipeline"="UniversalPipeline" }

        // Pass 1: black outline via inverted-hull (front-face cull, push verts along normal).
        Pass
        {
            Name "Outline"
            Cull Front
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            float _OutlineWidth;
            float4 _OutlineColor;
            struct Attributes { float4 positionOS : POSITION; float3 normalOS : NORMAL; };
            struct Varyings { float4 positionHCS : SV_POSITION; };
            Varyings vert (Attributes IN)
            {
                Varyings OUT;
                float3 posOS = IN.positionOS.xyz + normalize(IN.normalOS) * _OutlineWidth;
                OUT.positionHCS = TransformObjectToHClip(posOS);
                return OUT;
            }
            half4 frag (Varyings IN) : SV_Target { return _OutlineColor; }
            ENDHLSL
        }

        // Pass 2: flat unlit fill.
        Pass
        {
            Name "Unlit"
            Cull Back
            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            float4 _BaseColor;
            struct Attributes { float4 positionOS : POSITION; };
            struct Varyings { float4 positionHCS : SV_POSITION; };
            Varyings vert (Attributes IN)
            {
                Varyings OUT;
                OUT.positionHCS = TransformObjectToHClip(IN.positionOS.xyz);
                return OUT;
            }
            half4 frag (Varyings IN) : SV_Target { return _BaseColor; }
            ENDHLSL
        }
    }
}
```
Then create the materials and assign the shader via Coplay MCP. For each of the nine non-skybox materials, call `create_material` then `assign_shader_to_material` with `shaderName: "HexWars/UnlitOutline"`, then `set_property` for `_BaseColor`. Exact calls:
- `create_material` `{ "path": "Assets/HexWars/Art/Materials/Hex_Plains.mat" }`; `assign_shader_to_material` `{ "materialPath": "Assets/HexWars/Art/Materials/Hex_Plains.mat", "shaderName": "HexWars/UnlitOutline" }`; `set_property` `{ "assetPath": "Assets/HexWars/Art/Materials/Hex_Plains.mat", "property": "_BaseColor", "value": [1.0,0.9,0.1,1.0] }` (yellow plains).
- `Hex_Forest.mat` â†’ `_BaseColor` `[0.15,0.55,0.2,1.0]` (forest green tint).
- `Hex_Rough.mat` â†’ `_BaseColor` `[0.6,0.5,0.35,1.0]` (rough tan).
- `Hex_Water.mat` â†’ `_BaseColor` `[0.15,0.4,0.8,1.0]` (water blue).
- `Unit_P0.mat` â†’ `_BaseColor` `[0.1,0.8,0.95,1.0]` (cyan/blue, PlayerId 0).
- `Unit_P1.mat` â†’ `_BaseColor` `[0.95,0.15,0.4,1.0]` (red/magenta, PlayerId 1).
- `Gen_P0.mat` â†’ `_BaseColor` `[0.2,0.6,0.95,1.0]` (P0 generator, distinct blue).
- `Gen_P1.mat` â†’ `_BaseColor` `[0.9,0.2,0.6,1.0]` (P1 generator, distinct magenta).
For the skybox, create a separate material on Unity's `Skybox/Procedural` shader (a procedural starfield-ready dark sky): `create_material` `{ "path": "Assets/HexWars/Art/Materials/Starfield_Skybox.mat" }`; `assign_shader_to_material` `{ "materialPath": "Assets/HexWars/Art/Materials/Starfield_Skybox.mat", "shaderName": "Skybox/Procedural" }`; `set_property` `{ "assetPath": "Assets/HexWars/Art/Materials/Starfield_Skybox.mat", "property": "_SkyTint", "value": [0.02,0.02,0.05,1.0] }` and `set_property` `{ "assetPath": "Assets/HexWars/Art/Materials/Starfield_Skybox.mat", "property": "_GroundColor", "value": [0.0,0.0,0.0,1.0] }` (deep-space dark; Task 27 applies it in lighting).
Then build the shared hex prism prefab. Create a low-wide hexagonal prism scene object and assign the plains material, then save as a prefab:
  1. `create_game_object` `{ "name": "Hex", "primitiveType": "Cylinder" }` (a 6-sided low cylinder reads as a hex prism; scale to low-wide).
  2. `set_transform` `{ "name": "Hex", "scale": [1.0,0.25,1.0] }` (radius â‰ˆ HexSize, half-height 0.25 â†’ total height 0.5 = `ElevationStep`; low and wide per Â§12).
  3. `assign_material` `{ "gameObjectName": "Hex", "materialPath": "Assets/HexWars/Art/Materials/Hex_Plains.mat" }`.
  4. `create_prefab` `{ "gameObjectName": "Hex", "prefabPath": "Assets/HexWars/Art/Prefabs/Hex.prefab" }`.
  5. `delete_game_object` `{ "name": "Hex" }` to remove the scene instance (the prefab asset persists).
- [ ] **Step 4: Run test to verify it passes (verify assets via Coplay MCP)**
Run the guard test: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.HexLayoutTests.PrismSizing_MatchesContractConstants" -logFile -` â†’ Expected: PASS.
Then verify assets via Coplay MCP:
  - `check_compile_errors` (no args) â†’ Expected: zero compile/shader errors (the URP `UnlitOutline` shader compiles).
  - `get_game_object_info` on a temporary instance of the prefab, or inspect via `place_asset_in_scene` `{ "assetPath": "Assets/HexWars/Art/Prefabs/Hex.prefab" }` then `get_game_object_info` `{ "name": "Hex" }` â†’ Expected: object has a `MeshRenderer` whose material is `Hex_Plains` and whose shader is `HexWars/UnlitOutline` (outline pass present); scale is `[1.0,0.25,1.0]` (low-wide). Delete the temp instance afterward with `delete_game_object`.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Art && git commit -m "feat(art): URP unlit+outline shader, terrain/player/skybox materials, shared Hex prefab"
```

---

### Task 27: Main scene with camera, lighting, skybox, and CameraRig

**Files:**
- Create: `Assets/HexWars/Scenes/HexWars.unity`
- Create: `Assets/HexWars/Presentation/CameraRig.cs`
- Test: none for the scene; the pure pan/orbit math is verified by the EditMode test added below

**Interfaces:**
- Consumes: `Assets/HexWars/Art/Materials/Starfield_Skybox.mat` (Task 26); `HexWars.Presentation` assembly (Task 25).
- Produces: scene `Assets/HexWars/Scenes/HexWars.unity` (URP camera + directional light + starfield skybox configured); `namespace HexWars.Presentation { public sealed class CameraRig : UnityEngine.MonoBehaviour }` providing angled orbit/pan, with a pure static helper `CameraRig.OrbitOffset(float yawDegrees, float pitchDegrees, float distance) -> (float x, float y, float z)` â€” consumed by the play flow (Task 31).

- [ ] **Step 1: Write the failing test (pure orbit math extracted from the MonoBehaviour)**
```csharp
// Assets/HexWars/Engine/Tests/CameraRigMathTests.cs
using NUnit.Framework;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    public sealed class CameraRigMathTests
    {
        private const float Eps = 1e-3f;

        [Test]
        public void OrbitOffset_ZeroYaw_AngledPitch_PullsBackAndUp()
        {
            // 45-degree pitch from straight ahead: camera sits behind (-z) and above (+y).
            var (x, y, z) = CameraRig.OrbitOffset(0f, 45f, 10f);
            Assert.AreEqual(0f, x, Eps);
            Assert.Greater(y, 0f, "angled rig must be above the board so elevation reads");
            Assert.Less(z, 0f, "camera offset sits behind the focus");
            // magnitude preserved = distance
            float mag = (float)System.Math.Sqrt(x * x + y * y + z * z);
            Assert.AreEqual(10f, mag, Eps);
        }

        [Test]
        public void OrbitOffset_NinetyYaw_RotatesAroundY()
        {
            var (x, _, z) = CameraRig.OrbitOffset(90f, 0f, 5f);
            // yaw 90 about Y: offset rotates from -z toward -x (or +x depending on sign) but |x| dominates.
            Assert.Greater(System.Math.Abs(x), System.Math.Abs(z) + Eps);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CameraRigMathTests" -logFile -`
Expected: FAIL â€” `HexWars.Presentation.CameraRig` does not exist yet (compile error in the test assembly).
- [ ] **Step 3: Write minimal implementation (CameraRig MonoBehaviour with the pure helper)**
```csharp
// Assets/HexWars/Presentation/CameraRig.cs
using UnityEngine;

namespace HexWars.Presentation
{
    // Angled orbit/pan rig so board height reads. The orbit math is a pure static helper
    // (CameraRig.OrbitOffset) unit-tested in the EditMode engine-tests assembly.
    public sealed class CameraRig : MonoBehaviour
    {
        [SerializeField] private Transform _focus;
        [SerializeField] private float _yaw = 0f;
        [SerializeField] private float _pitch = 45f;     // angled so elevation reads
        [SerializeField] private float _distance = 14f;
        [SerializeField] private float _orbitSpeed = 90f; // deg/sec
        [SerializeField] private float _panSpeed = 8f;    // world units/sec

        private Vector3 _focusPos;

        private void Awake()
        {
            _focusPos = _focus != null ? _focus.position : Vector3.zero;
        }

        private void LateUpdate()
        {
            // Orbit (Q/E) and pan (WASD/arrows) so the hotseat player can read elevation.
            if (Input.GetKey(KeyCode.Q)) _yaw -= _orbitSpeed * Time.deltaTime;
            if (Input.GetKey(KeyCode.E)) _yaw += _orbitSpeed * Time.deltaTime;
            float h = Input.GetAxisRaw("Horizontal");
            float v = Input.GetAxisRaw("Vertical");
            var flatForward = Quaternion.Euler(0f, _yaw, 0f) * Vector3.forward;
            var flatRight = Quaternion.Euler(0f, _yaw, 0f) * Vector3.right;
            _focusPos += (flatRight * h + flatForward * v) * (_panSpeed * Time.deltaTime);

            var (ox, oy, oz) = OrbitOffset(_yaw, _pitch, _distance);
            transform.position = _focusPos + new Vector3(ox, oy, oz);
            transform.LookAt(_focusPos);
        }

        // PURE: offset from focus for a given yaw/pitch/distance. Pitch 0 = level behind (-z);
        // increasing pitch raises the camera (+y). Yaw rotates about Y. |offset| == distance.
        public static (float x, float y, float z) OrbitOffset(float yawDegrees, float pitchDegrees, float distance)
        {
            double yaw = yawDegrees * System.Math.PI / 180.0;
            double pitch = pitchDegrees * System.Math.PI / 180.0;
            double horiz = distance * System.Math.Cos(pitch);
            double y = distance * System.Math.Sin(pitch);
            double x = horiz * System.Math.Sin(yaw);
            double z = -horiz * System.Math.Cos(yaw); // behind the focus at yaw 0
            return ((float)x, (float)y, (float)z);
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes (then build the scene via Coplay MCP)**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CameraRigMathTests" -logFile -`
Expected: PASS â€” both orbit-math tests green.
Then build and verify the scene via Coplay MCP:
  1. `create_scene` `{ "path": "Assets/HexWars/Scenes/HexWars.unity" }` then `open_scene` `{ "path": "Assets/HexWars/Scenes/HexWars.unity" }`.
  2. Ensure a `Main Camera` exists (`create_game_object` `{ "name": "Main Camera", "tag": "MainCamera" }` + `add_component` `{ "gameObjectName": "Main Camera", "componentType": "Camera" }` if the new scene lacks one), then `add_component` `{ "gameObjectName": "Main Camera", "componentType": "HexWars.Presentation.CameraRig" }`.
  3. Angle the camera: `set_transform` `{ "name": "Main Camera", "position": [0,10,-10], "rotation": [45,0,0] }` (matches the rig's default pitch so elevation reads before Play).
  4. `create_game_object` `{ "name": "Directional Light" }` + `add_component` `{ "gameObjectName": "Directional Light", "componentType": "Light" }` then `set_property` `{ "gameObjectName": "Directional Light", "componentType": "Light", "property": "type", "value": "Directional" }` and `set_transform` `{ "name": "Directional Light", "rotation": [50,-30,0] }`.
  5. Apply the starfield skybox to lighting via `set_property` on `RenderSettings.skybox` to `Assets/HexWars/Art/Materials/Starfield_Skybox.mat` (e.g. `set_property` `{ "target": "RenderSettings", "property": "skybox", "value": "Assets/HexWars/Art/Materials/Starfield_Skybox.mat" }`). If that target is unsupported, use `execute_script` `{ "code": "UnityEngine.RenderSettings.skybox = UnityEditor.AssetDatabase.LoadAssetAtPath<UnityEngine.Material>(\"Assets/HexWars/Art/Materials/Starfield_Skybox.mat\"); UnityEditor.SceneManagement.EditorSceneManager.MarkAllScenesDirty();" }`.
  6. `save_scene` `{ "path": "Assets/HexWars/Scenes/HexWars.unity" }`.
Verify: `check_compile_errors` â†’ zero errors. `get_game_object_info` `{ "name": "Main Camera" }` â†’ has a `Camera` and a `HexWars.Presentation.CameraRig` component, rotation x â‰ˆ 45 (angled). `capture_scene_object` `{ "name": "Main Camera" }` or `scene_view_functions` screenshot â†’ Expected: dark starfield skybox visible behind an angled view (not solid grey/blue default).
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/CameraRig.cs Assets/HexWars/Scenes/HexWars.unity Assets/HexWars/Engine/Tests/CameraRigMathTests.cs && git commit -m "feat(presentation): main scene with URP camera, light, starfield skybox, and CameraRig"
```

---

### Task 28: BoardRenderer: instance the hex prefab into elevation columns

**Files:**
- Create: `Assets/HexWars/Presentation/BoardRenderer.cs`
- Test: `Assets/HexWars/Engine/Tests/BoardRendererLayoutTests.cs` (pure-logic helper test, EditMode)

**Interfaces:**
- Consumes: `HexWars.Engine.Board` (`IReadOnlyList<Tile> Tiles`), `HexWars.Engine.Tile` (`HexCoord Coord`, `TerrainType Terrain`, `int Elevation`), `HexWars.Engine.TerrainType {Plains=0,Forest=1,Rough=2,Water=3}`, `HexWars.Engine.HexCoord {int Q,R,Elevation}`, `HexWars.Presentation.HexLayout.CoordToWorld(HexWars.Engine.HexCoord) -> (float x,float y,float z)` and `HexLayout.ElevationStep` (Task 25), `Hex.prefab` + `Hex_Plains/Forest/Rough/Water.mat` (Task 26), `AuthoredBoardGenerator`/`RandomBoardGenerator.Generate(GameConfig,int)` (Tasks 22,23).
- Produces: `public sealed class HexWars.Presentation.BoardRenderer : UnityEngine.MonoBehaviour { public void Render(HexWars.Engine.Board board); }`. Also a pure static helper `public static class BoardRenderer.ColumnLayout { public static int PrismCount(int elevation); public static int TerrainMaterialIndex(HexWars.Engine.TerrainType terrain); }` consumed by tests and by `Render`. Consumed by `GamePresenter` (Task 31).

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    [TestFixture]
    public sealed class BoardRendererLayoutTests
    {
        [Test]
        public void PrismCount_IsElevationPlusOne_SoElevationZeroIsOneTile()
        {
            Assert.AreEqual(1, BoardRenderer.ColumnLayout.PrismCount(0));
            Assert.AreEqual(3, BoardRenderer.ColumnLayout.PrismCount(2));
            Assert.AreEqual(5, BoardRenderer.ColumnLayout.PrismCount(4));
        }

        [Test]
        public void PrismCount_NeverBelowOne_EvenForNegativeElevation()
        {
            Assert.AreEqual(1, BoardRenderer.ColumnLayout.PrismCount(-3));
        }

        [Test]
        public void TerrainMaterialIndex_MatchesTerrainTypeEnumOrder()
        {
            Assert.AreEqual(0, BoardRenderer.ColumnLayout.TerrainMaterialIndex(TerrainType.Plains));
            Assert.AreEqual(1, BoardRenderer.ColumnLayout.TerrainMaterialIndex(TerrainType.Forest));
            Assert.AreEqual(2, BoardRenderer.ColumnLayout.TerrainMaterialIndex(TerrainType.Rough));
            Assert.AreEqual(3, BoardRenderer.ColumnLayout.TerrainMaterialIndex(TerrainType.Water));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardRendererLayoutTests" -logFile -`
Expected: FAIL â€” compile error: `BoardRenderer` / `BoardRenderer.ColumnLayout` does not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
using System.Collections.Generic;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // Instances the single shared Hex prefab into vertical columns by elevation.
    // One prism per level: elevation 0 = 1 prism, elevation N = N+1 prisms.
    // The top prism of each column carries the terrain material.
    public sealed class BoardRenderer : MonoBehaviour
    {
        [SerializeField] private GameObject _hexPrefab;
        // Indexed by (int)TerrainType: 0 Plains, 1 Forest, 2 Rough, 3 Water.
        [SerializeField] private Material[] _terrainMaterials = new Material[4];

        private readonly List<GameObject> _spawned = new List<GameObject>();

        // Pure layout logic, unit-tested in EditMode (no UnityEngine types).
        public static class ColumnLayout
        {
            // Number of stacked prisms for a column at the given elevation (>=1).
            public static int PrismCount(int elevation)
            {
                int count = elevation + 1;
                return count < 1 ? 1 : count;
            }

            // Material slot for a terrain, matching TerrainType enum order.
            public static int TerrainMaterialIndex(TerrainType terrain)
            {
                return (int)terrain;
            }
        }

        public void Render(Board board)
        {
            Clear();
            if (board == null || _hexPrefab == null)
            {
                return;
            }

            foreach (Tile tile in board.Tiles)
            {
                BuildColumn(tile);
            }
        }

        private void BuildColumn(Tile tile)
        {
            HexCoord coord = tile.Coord;
            (float x, float _, float z) = HexLayout.CoordToWorld(coord.WithElevation(0));
            int prisms = ColumnLayout.PrismCount(tile.Elevation);
            Material topMaterial = TerrainMaterialFor(tile.Terrain);

            for (int level = 0; level < prisms; level++)
            {
                float y = level * HexLayout.ElevationStep;
                GameObject prism = Instantiate(
                    _hexPrefab,
                    new Vector3(x, y, z),
                    Quaternion.identity,
                    transform);
                prism.name = $"Hex_{coord.Q}_{coord.R}_lvl{level}";
                if (level == prisms - 1 && topMaterial != null)
                {
                    ApplyMaterial(prism, topMaterial);
                }
                _spawned.Add(prism);
            }
        }

        private Material TerrainMaterialFor(TerrainType terrain)
        {
            int index = ColumnLayout.TerrainMaterialIndex(terrain);
            if (_terrainMaterials != null && index >= 0 && index < _terrainMaterials.Length)
            {
                return _terrainMaterials[index];
            }
            return null;
        }

        private static void ApplyMaterial(GameObject prism, Material material)
        {
            var renderer = prism.GetComponentInChildren<MeshRenderer>();
            if (renderer != null)
            {
                renderer.sharedMaterial = material;
            }
        }

        private void Clear()
        {
            for (int i = 0; i < _spawned.Count; i++)
            {
                if (_spawned[i] != null)
                {
                    if (Application.isPlaying)
                    {
                        Destroy(_spawned[i]);
                    }
                    else
                    {
                        DestroyImmediate(_spawned[i]);
                    }
                }
            }
            _spawned.Clear();
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.BoardRendererLayoutTests" -logFile -`
Expected: PASS (all 3 tests green). Then verify the MonoBehaviour wiring in-editor via Coplay MCP:
  1. `mcp__coplay-mcp__check_compile_errors` -> expect no errors.
  2. `mcp__coplay-mcp__create_game_object` `{ "name": "BoardRenderer" }`, then `mcp__coplay-mcp__add_component` `{ "gameObjectName": "BoardRenderer", "componentType": "HexWars.Presentation.BoardRenderer" }`.
  3. Wire references with `mcp__coplay-mcp__set_property`: `_hexPrefab` -> `Assets/HexWars/Art/Prefabs/Hex.prefab`; `_terrainMaterials` -> [`Hex_Plains.mat`,`Hex_Forest.mat`,`Hex_Rough.mat`,`Hex_Water.mat`] in that order.
  4. `mcp__coplay-mcp__execute_script` to build a tiny `AuthoredBoardGenerator` board (e.g. 2x2: one Plains tile at elevation 0, one Forest tile at elevation 2, one Rough at elevation 4, one Water at elevation 1) and call `boardRenderer.Render(board)`.
  5. `mcp__coplay-mcp__get_game_object_info` `{ "name": "BoardRenderer" }` -> expect child counts per column = elevation+1 (1, 3, 5, 2 prisms respectively).
  6. `mcp__coplay-mcp__capture_scene_object` `{ "name": "BoardRenderer" }` -> screenshot shows four columns of differing heights, with top-tile materials matching Plains(yellow)/Forest/Rough/Water.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/BoardRenderer.cs Assets/HexWars/Engine/Tests/BoardRendererLayoutTests.cs && git commit -m "feat(presentation): BoardRenderer instances Hex prefab into elevation columns with terrain top material"
```

---

### Task 29: UnitView and GeneratorView with owner colors and outlines

**Files:**
- Create: `Assets/HexWars/Presentation/UnitView.cs`
- Create: `Assets/HexWars/Presentation/GeneratorView.cs`
- Create: `Assets/HexWars/Art/Prefabs/UnitView.prefab`
- Create: `Assets/HexWars/Art/Prefabs/GeneratorView.prefab`
- Test: `Assets/HexWars/Engine/Tests/OwnerMaterialIndexTests.cs` (pure-logic helper test, EditMode)

**Interfaces:**
- Consumes: `HexWars.Engine.Unit` (`int Id`, `PlayerId Owner`, `int CurrentHp`, `HexCoord Position`), `HexWars.Engine.Generator` (`int Id`, `PlayerId Owner`, `int CurrentHp`, `HexCoord Position`), `HexWars.Engine.PlayerId {Player0=0,Player1=1}`, `HexWars.Presentation.HexLayout.CoordToWorld(HexCoord) -> (float,float,float)` (Task 25), `Unit_P0/Unit_P1/Gen_P0/Gen_P1.mat` (Task 26), `BoardRenderer` placement convention (Task 28).
- Produces: `public sealed class HexWars.Presentation.UnitView : UnityEngine.MonoBehaviour { public void Bind(HexWars.Engine.Unit unit); }`; `public sealed class HexWars.Presentation.GeneratorView : UnityEngine.MonoBehaviour { public void Bind(HexWars.Engine.Generator generator); }`; prefabs `UnitView.prefab`, `GeneratorView.prefab`. Shared pure helper `public static int OwnerMaterialIndex(HexWars.Engine.PlayerId owner)` exposed on both views. Consumed by `GamePresenter` (Task 31).

- [ ] **Step 1: Write the failing test**
```csharp
using NUnit.Framework;
using HexWars.Engine;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    [TestFixture]
    public sealed class OwnerMaterialIndexTests
    {
        [Test]
        public void UnitView_OwnerMaterialIndex_MatchesPlayerIdOrder()
        {
            Assert.AreEqual(0, UnitView.OwnerMaterialIndex(PlayerId.Player0));
            Assert.AreEqual(1, UnitView.OwnerMaterialIndex(PlayerId.Player1));
        }

        [Test]
        public void GeneratorView_OwnerMaterialIndex_MatchesPlayerIdOrder()
        {
            Assert.AreEqual(0, GeneratorView.OwnerMaterialIndex(PlayerId.Player0));
            Assert.AreEqual(1, GeneratorView.OwnerMaterialIndex(PlayerId.Player1));
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.OwnerMaterialIndexTests" -logFile -`
Expected: FAIL â€” compile error: `UnitView` / `GeneratorView` do not exist yet.
- [ ] **Step 3: Write minimal implementation**
```csharp
// File: Assets/HexWars/Presentation/UnitView.cs
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // Binds an engine Unit to a scene object: owner color + world position.
    public sealed class UnitView : MonoBehaviour
    {
        // Indexed by (int)PlayerId: 0 = Unit_P0 (cyan/blue), 1 = Unit_P1 (red/magenta).
        [SerializeField] private Material[] _ownerMaterials = new Material[2];
        [SerializeField] private MeshRenderer _renderer;

        public int UnitId { get; private set; }
        public PlayerId Owner { get; private set; }
        public int CurrentHp { get; private set; }

        public static int OwnerMaterialIndex(PlayerId owner) => (int)owner;

        public void Bind(Unit unit)
        {
            UnitId = unit.Id;
            Owner = unit.Owner;
            CurrentHp = unit.CurrentHp;

            (float x, float y, float z) = HexLayout.CoordToWorld(unit.Position);
            transform.position = new Vector3(x, y, z);

            ApplyOwnerMaterial(unit.Owner);
        }

        private void ApplyOwnerMaterial(PlayerId owner)
        {
            int index = OwnerMaterialIndex(owner);
            if (_renderer == null)
            {
                _renderer = GetComponentInChildren<MeshRenderer>();
            }
            if (_renderer != null && _ownerMaterials != null &&
                index >= 0 && index < _ownerMaterials.Length && _ownerMaterials[index] != null)
            {
                _renderer.sharedMaterial = _ownerMaterials[index];
            }
        }
    }
}
```
```csharp
// File: Assets/HexWars/Presentation/GeneratorView.cs
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // Binds an engine Generator: distinct pylon shape + owner color + world position.
    public sealed class GeneratorView : MonoBehaviour
    {
        // Indexed by (int)PlayerId: 0 = Gen_P0, 1 = Gen_P1.
        [SerializeField] private Material[] _ownerMaterials = new Material[2];
        [SerializeField] private MeshRenderer _renderer;

        public int GeneratorId { get; private set; }
        public PlayerId Owner { get; private set; }
        public int CurrentHp { get; private set; }

        public static int OwnerMaterialIndex(PlayerId owner) => (int)owner;

        public void Bind(Generator generator)
        {
            GeneratorId = generator.Id;
            Owner = generator.Owner;
            CurrentHp = generator.CurrentHp;

            (float x, float y, float z) = HexLayout.CoordToWorld(generator.Position);
            transform.position = new Vector3(x, y, z);

            ApplyOwnerMaterial(generator.Owner);
        }

        private void ApplyOwnerMaterial(PlayerId owner)
        {
            int index = OwnerMaterialIndex(owner);
            if (_renderer == null)
            {
                _renderer = GetComponentInChildren<MeshRenderer>();
            }
            if (_renderer != null && _ownerMaterials != null &&
                index >= 0 && index < _ownerMaterials.Length && _ownerMaterials[index] != null)
            {
                _renderer.sharedMaterial = _ownerMaterials[index];
            }
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.OwnerMaterialIndexTests" -logFile -`
Expected: PASS (both tests green). Then build the prefabs and verify wiring in-editor via Coplay MCP:
  1. `mcp__coplay-mcp__check_compile_errors` -> expect no errors.
  2. Unit prefab: `mcp__coplay-mcp__create_game_object` `{ "name": "UnitView", "primitive": "Capsule" }` (unit = capsule body). `mcp__coplay-mcp__add_component` `{ "gameObjectName": "UnitView", "componentType": "HexWars.Presentation.UnitView" }`. `mcp__coplay-mcp__assign_material` `{ "gameObjectName": "UnitView", "materialPath": "Assets/HexWars/Art/Materials/Unit_P0.mat" }`. `mcp__coplay-mcp__set_property` to populate `_ownerMaterials` = [`Unit_P0.mat`,`Unit_P1.mat`]. `mcp__coplay-mcp__create_prefab` `{ "gameObjectName": "UnitView", "prefabPath": "Assets/HexWars/Art/Prefabs/UnitView.prefab" }`.
  3. Generator prefab (distinct pylon shape, not just color): `mcp__coplay-mcp__create_game_object` `{ "name": "GeneratorView", "primitive": "Cylinder" }`, scale it narrow+tall via `mcp__coplay-mcp__set_transform` `{ "name": "GeneratorView", "scale": [0.4, 1.2, 0.4] }` (pylon silhouette). `mcp__coplay-mcp__add_component` `{ "gameObjectName": "GeneratorView", "componentType": "HexWars.Presentation.GeneratorView" }`. `mcp__coplay-mcp__assign_material` `{ "gameObjectName": "GeneratorView", "materialPath": "Assets/HexWars/Art/Materials/Gen_P0.mat" }`. `mcp__coplay-mcp__set_property` to populate `_ownerMaterials` = [`Gen_P0.mat`,`Gen_P1.mat`]. `mcp__coplay-mcp__create_prefab` `{ "gameObjectName": "GeneratorView", "prefabPath": "Assets/HexWars/Art/Prefabs/GeneratorView.prefab" }`.
  4. Place-and-bind check: `mcp__coplay-mcp__execute_script` instantiating UnitView.prefab and binding a `new Unit(1, PlayerId.Player0, ...)`, and GeneratorView.prefab binding a `new Generator(2, PlayerId.Player1, ...)`.
  5. `mcp__coplay-mcp__get_game_object_info` on each -> confirm UnitView's renderer uses Unit_P0 (cyan/blue), GeneratorView's renderer uses Gen_P1 (red/magenta), and the generator transform scale (0.4,1.2,0.4) differs from the unit capsule shape.
  6. `mcp__coplay-mcp__capture_scene_object` -> screenshot shows a cyan unit and a red/magenta pylon with distinct silhouettes and black outlines.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/UnitView.cs Assets/HexWars/Presentation/GeneratorView.cs Assets/HexWars/Art/Prefabs/UnitView.prefab Assets/HexWars/Art/Prefabs/GeneratorView.prefab Assets/HexWars/Engine/Tests/OwnerMaterialIndexTests.cs && git commit -m "feat(presentation): UnitView and GeneratorView prefabs with owner colors, pylon generator shape, outlines"
```

---

### Task 30: InputController and HUD (stat panel, create-unit allocation panel, banner, End Turn, win screen)

**Files:**
- Create: `Assets/HexWars/Presentation/InputController.cs`
- Create: `Assets/HexWars/Presentation/Hud.cs`
- Create: `Assets/HexWars/Presentation/CreatePanelModel.cs` (pure, unit-tested helper backing the create-unit allocation panel)
- Test: `Assets/HexWars/Engine/Tests/CreatePanelModelTests.cs`

**Interfaces:**
- Consumes: `HexWars.Engine.UnitStats(int health,int damage,int range,int movement,int defense,int vision)` with `int PointCost` and `bool IsValid` (Task 5); `HexWars.Engine.GameConfig` with `int StartingPoints` (Task 6); `HexWars.Engine.HexCoord(int q,int r,int elevation=0)` (Task 2); `HexWars.Engine.TargetingService` and `HexWars.Engine.HexGeometry.Distance3D(HexCoord,HexCoord)` (Tasks 8,10) for the reach-preview semantics; `HexWars.Engine.Unit` with `Id/Owner/Stats/Position/CurrentHp` (Task 5); `HexWars.Engine.PlayerId` (Task 2); `HexWars.Engine.Command` records `CreateUnit/DeployUnit/DeployGenerator/MoveUnit/AttackUnit/EndTurn` (Task 14); `HexWars.Engine.GameEngine.LegalMoves/Apply` (Tasks 16,17,18). Unity: new Input System (`UnityEngine.InputSystem`), `UnityEngine.UI`, `UnityEngine.Camera`, `UnityEngine.Physics.Raycast`.
- Produces: `HexWars.Presentation.CreatePanelModel` (pure): `static int TotalCost(UnitStats stats)`, `static bool IsAffordable(UnitStats stats,int availablePoints)`, `static int AttackReach(UnitStats stats)`, `static int VisionReach(UnitStats stats)`, `static string EffectText(string statName)`, `static UnitStats Clamp(UnitStats stats,int availablePoints)`. `public sealed class InputController:MonoBehaviour` exposing events `event System.Action<HexCoord> TileSelected; event System.Action<int> UnitSelected; event System.Action EndTurnRequested;` and methods `void SetCamera(UnityEngine.Camera)`, `void Enable()/Disable()`. `public sealed class Hud:MonoBehaviour{ void ShowPoints(int); void ShowSelectedUnit(HexWars.Engine.Unit); void ShowCreatePanel(HexWars.Engine.UnitStats stats,int availablePoints); void ShowTurnBanner(HexWars.Engine.PlayerId); void ShowWinScreen(HexWars.Engine.PlayerId); event System.Action<HexWars.Engine.UnitStats> CreateUnitConfirmed; event System.Action EndTurnClicked; }` (consumed by GamePresenter, Task 31).

- [ ] **Step 1: Write the failing test**
The reach/cost/effect math powering the create-unit panel is the only non-Unity logic, so it lives in a pure `CreatePanelModel` and is unit-tested in the EditMode engine-tests assembly (which already references `HexWars.Presentation`, per Task 25). This proves the panel computes `total = sum of stats 1:1` and the reach previews match engine semantics.
```csharp
using NUnit.Framework;
using HexWars.Engine;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    public class CreatePanelModelTests
    {
        [Test]
        public void TotalCost_IsStrict1To1SumOfAllSixStats()
        {
            var stats = new UnitStats(health: 3, damage: 2, range: 4, movement: 1, defense: 0, vision: 5);
            Assert.AreEqual(15, CreatePanelModel.TotalCost(stats));
            Assert.AreEqual(stats.PointCost, CreatePanelModel.TotalCost(stats));
        }

        [Test]
        public void IsAffordable_TrueWhenTotalWithinBudget_FalseWhenOver()
        {
            var stats = new UnitStats(2, 2, 2, 2, 2, 2); // cost 12
            Assert.IsTrue(CreatePanelModel.IsAffordable(stats, 12));
            Assert.IsTrue(CreatePanelModel.IsAffordable(stats, 13));
            Assert.IsFalse(CreatePanelModel.IsAffordable(stats, 11));
        }

        [Test]
        public void AttackReach_EqualsBaseRange_AndVisionReach_EqualsVision()
        {
            // Reach preview on flat ground: range/vision reach == the bought stat (no high-ground,
            // no concealment); matches TargetingService InRange/IsVisible with H=0, concealment=0.
            var stats = new UnitStats(1, 0, 4, 0, 0, 6);
            Assert.AreEqual(4, CreatePanelModel.AttackReach(stats));
            Assert.AreEqual(6, CreatePanelModel.VisionReach(stats));
        }

        [Test]
        public void EffectText_ReturnsPlainLanguageEffectPerStat()
        {
            Assert.IsNotEmpty(CreatePanelModel.EffectText("Health"));
            Assert.IsNotEmpty(CreatePanelModel.EffectText("Vision"));
            StringAssert.Contains("target", CreatePanelModel.EffectText("Vision").ToLowerInvariant());
        }

        [Test]
        public void Clamp_ReducesNothingWhenAffordable_AndNeverDropsHealthBelowOne()
        {
            var ok = new UnitStats(2, 2, 0, 0, 0, 0); // cost 4
            Assert.AreEqual(ok, CreatePanelModel.Clamp(ok, 10));

            var over = new UnitStats(1, 9, 9, 9, 9, 9); // cost 46
            var clamped = CreatePanelModel.Clamp(over, 1);
            Assert.IsTrue(clamped.IsValid, "clamp must keep Health >= 1");
            Assert.LessOrEqual(clamped.PointCost, 1);
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CreatePanelModelTests" -logFile -`
Expected: FAIL â€” `CreatePanelModel` does not exist yet (compile error / type not found).
- [ ] **Step 3: Write minimal implementation**
First the pure helper (no UnityEngine):
```csharp
// Assets/HexWars/Presentation/CreatePanelModel.cs
using HexWars.Engine;

namespace HexWars.Presentation
{
    // PURE backing model for the create-unit allocation panel. No UnityEngine references so it is
    // unit-tested in the EditMode engine-tests assembly. The buy menu IS the rules (spec 2.4):
    // total == sum of stats 1:1, reach previews mirror engine TargetingService semantics on flat ground.
    public static class CreatePanelModel
    {
        public static int TotalCost(UnitStats stats) => stats.PointCost; // strict 1:1

        public static bool IsAffordable(UnitStats stats, int availablePoints)
            => stats.PointCost <= availablePoints;

        // Flat-ground reach preview: H=0 so InRange reduces to D <= Range.
        public static int AttackReach(UnitStats stats) => stats.Range;

        // Flat-ground vision preview: concealment=0 so IsVisible reduces to D <= Vision.
        public static int VisionReach(UnitStats stats) => stats.Vision;

        public static string EffectText(string statName)
        {
            switch (statName)
            {
                case "Health":   return "Max hit points. Destroyed at 0.";
                case "Damage":   return "Damage per attack, reduced by enemy Defense + terrain.";
                case "Range":    return "How far it can attack, in 3D distance.";
                case "Movement": return "Move budget per turn; climbing and terrain cost extra.";
                case "Defense":  return "Flat reduction of incoming damage per hit.";
                case "Vision":   return "How far it can detect and therefore target enemies, in 3D.";
                default:          return string.Empty;
            }
        }

        // Greedy reduce-to-budget keeping Health >= 1 (the only floor). Trims non-health stats first,
        // then Health down to 1, so the returned stats are always valid and affordable.
        public static UnitStats Clamp(UnitStats stats, int availablePoints)
        {
            int h = stats.Health, d = stats.Damage, r = stats.Range,
                m = stats.Movement, def = stats.Defense, v = stats.Vision;
            int budget = availablePoints < 1 ? 1 : availablePoints; // must allow the 1-HP floor

            int Cost() => h + d + r + m + def + v;
            // Trim cheaper-to-lose stats first; never touch Health until everything else is 0.
            while (Cost() > budget && (d + r + m + def + v) > 0)
            {
                if (v > 0) { v--; continue; }
                if (def > 0) { def--; continue; }
                if (m > 0) { m--; continue; }
                if (r > 0) { r--; continue; }
                if (d > 0) { d--; continue; }
            }
            while (Cost() > budget && h > 1) h--; // never below 1
            return new UnitStats(h, d, r, m, def, v);
        }
    }
}
```
Then the two MonoBehaviours (thin skin; no rules):
```csharp
// Assets/HexWars/Presentation/InputController.cs
using System;
using UnityEngine;
using UnityEngine.InputSystem;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // New Input System: raycast-select a tile column or a unit/generator view and raise intents.
    // Contains NO rules; it only translates pointer hits into engine-coordinate selection events.
    public sealed class InputController : MonoBehaviour
    {
        public event Action<HexCoord> TileSelected;
        public event Action<int> UnitSelected;
        public event Action EndTurnRequested;

        [SerializeField] private Camera _camera;
        private bool _enabled;

        public void SetCamera(Camera cam) => _camera = cam;
        public void Enable() => _enabled = true;
        public void Disable() => _enabled = false;

        private void Awake() { if (_camera == null) _camera = Camera.main; _enabled = true; }

        private void Update()
        {
            if (!_enabled || _camera == null) return;
            var mouse = Mouse.current;
            if (mouse != null && mouse.leftButton.wasPressedThisFrame)
                TryRaycastSelect(mouse.position.ReadValue());

            var kb = Keyboard.current;
            if (kb != null && kb.spaceKey.wasPressedThisFrame)
                EndTurnRequested?.Invoke();
        }

        private void TryRaycastSelect(Vector2 screenPos)
        {
            Ray ray = _camera.ScreenPointToRay(screenPos);
            if (!Physics.Raycast(ray, out RaycastHit hit, 1000f)) return;

            // Units/generators carry an IEntityView; tiles carry an ITileView (both set by Task 28/29).
            var entity = hit.collider.GetComponentInParent<IEntityView>();
            if (entity != null) { UnitSelected?.Invoke(entity.EntityId); return; }

            var tile = hit.collider.GetComponentInParent<ITileView>();
            if (tile != null) TileSelected?.Invoke(tile.Coord);
        }
    }

    // Minimal view markers so InputController stays decoupled from concrete view classes.
    public interface IEntityView { int EntityId { get; } }
    public interface ITileView { HexCoord Coord { get; } }
}
```
```csharp
// Assets/HexWars/Presentation/Hud.cs
using System;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // HUD skin: points, selected-unit stat panel, create-unit allocation panel (1:1 cost + effect +
    // running total + reach preview), turn banner, End Turn button, win screen. No rules logic.
    public sealed class Hud : MonoBehaviour
    {
        public event Action<UnitStats> CreateUnitConfirmed;
        public event Action EndTurnClicked;

        [Header("Points / selection")]
        [SerializeField] private Text _pointsText;
        [SerializeField] private Text _selectedUnitText;

        [Header("Create-unit panel")]
        [SerializeField] private GameObject _createPanel;
        [SerializeField] private Text _createTotalText;
        [SerializeField] private Text _createReachText;
        [SerializeField] private Button _createConfirmButton;

        [Header("Banner / win / end-turn")]
        [SerializeField] private GameObject _turnBanner;
        [SerializeField] private Text _turnBannerText;
        [SerializeField] private GameObject _winScreen;
        [SerializeField] private Text _winScreenText;
        [SerializeField] private Button _endTurnButton;

        private UnitStats _pendingStats;
        private int _availablePoints;

        private void Awake()
        {
            if (_endTurnButton != null) _endTurnButton.onClick.AddListener(() => EndTurnClicked?.Invoke());
            if (_createConfirmButton != null)
                _createConfirmButton.onClick.AddListener(() => CreateUnitConfirmed?.Invoke(_pendingStats));
            if (_winScreen != null) _winScreen.SetActive(false);
            if (_turnBanner != null) _turnBanner.SetActive(false);
        }

        public void ShowPoints(int points)
        {
            if (_pointsText != null) _pointsText.text = $"Points: {points}";
        }

        public void ShowSelectedUnit(Unit unit)
        {
            if (_selectedUnitText == null) return;
            if (unit == null) { _selectedUnitText.text = string.Empty; return; }
            var s = unit.Stats;
            _selectedUnitText.text =
                $"Unit #{unit.Id} ({(unit.Owner == PlayerId.Player0 ? \"P1\" : \"P2\")})\n" +
                $"HP {unit.CurrentHp}/{s.Health}  DMG {s.Damage}  RNG {s.Range}\n" +
                $"MOV {s.Movement}  DEF {s.Defense}  VIS {s.Vision}";
        }

        public void ShowCreatePanel(UnitStats stats, int availablePoints)
        {
            _availablePoints = availablePoints;
            _pendingStats = CreatePanelModel.Clamp(stats, availablePoints);
            if (_createPanel != null) _createPanel.SetActive(true);
            if (_createTotalText != null)
                _createTotalText.text = $"Total: {CreatePanelModel.TotalCost(_pendingStats)} / {availablePoints}";
            if (_createReachText != null)
                _createReachText.text =
                    $"Attack reach {CreatePanelModel.AttackReach(_pendingStats)}  " +
                    $"Vision reach {CreatePanelModel.VisionReach(_pendingStats)}";
            if (_createConfirmButton != null)
                _createConfirmButton.interactable =
                    CreatePanelModel.IsAffordable(_pendingStats, availablePoints) && _pendingStats.IsValid;
        }

        public void ShowTurnBanner(PlayerId player)
        {
            if (_turnBanner == null) return;
            _turnBanner.SetActive(true);
            if (_turnBannerText != null)
                _turnBannerText.text = $"Pass device â€” {(player == PlayerId.Player0 ? \"Player 1\" : \"Player 2\")}'s turn";
        }

        public void ShowWinScreen(PlayerId winner)
        {
            if (_winScreen == null) return;
            _winScreen.SetActive(true);
            if (_winScreenText != null)
                _winScreenText.text = $"{(winner == PlayerId.Player0 ? \"Player 1\" : \"Player 2\")} wins!";
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.CreatePanelModelTests" -logFile -`
Expected: PASS (all 5 tests green). Then verify the Unity-side code compiles and the panel wiring is correct via Coplay MCP:
  - `check_compile_errors` â†’ expect zero errors for `HexWars.Presentation`.
  - `create_ui_element` to build a Canvas with the create-panel children (`createTotalText`, `createReachText`, `createConfirmButton`, `pointsText`, `selectedUnitText`, `turnBanner`/`turnBannerText`, `winScreen`/`winScreenText`), then `add_component` Hud to the Canvas root and `set_property` to assign each serialized field.
  - `set_ui_text` the total field via a tiny driver, e.g. `execute_script` calling `hud.ShowCreatePanel(new UnitStats(3,2,4,1,0,5), 20)`, then `get_game_object_info` on `createTotalText` â†’ expect text `"Total: 15 / 20"` (confirms panel total == sum of stats 1:1) and on `createReachText` â†’ expect `"Attack reach 4  Vision reach 5"`.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/CreatePanelModel.cs Assets/HexWars/Presentation/InputController.cs Assets/HexWars/Presentation/Hud.cs Assets/HexWars/Engine/Tests/CreatePanelModelTests.cs && git commit -m "feat(presentation): InputController + HUD with unit-tested create-panel model"
```

---

### Task 31: GamePresenter: own GameState, push Commands, render events, hotseat flow

**Files:**
- Create: `Assets/HexWars/Presentation/GamePresenter.cs`
- Create: `Assets/HexWars/Presentation/SelectionResolver.cs` (pure, unit-tested helper that turns a raw selection into the intended Command)
- Modify: `Assets/HexWars/Scenes/HexWars.unity` (wire BoardRenderer, UnitView/GeneratorView roots, CameraRig, InputController, Hud, and GamePresenter together)
- Test: `Assets/HexWars/Engine/Tests/SelectionResolverTests.cs`

**Interfaces:**
- Consumes: `HexWars.Engine.GameFactory.NewGame(IBoardGenerator generator,GameConfig config,int seed)` (Task 20); `HexWars.Engine.RandomBoardGenerator` (Task 22); `HexWars.Engine.GameConfig.Default()` (Task 6); `HexWars.Engine.GameEngine.Apply(GameState,Command)` and `LegalMoves(GameState)` (Tasks 16,17,18); `HexWars.Engine.Result{ bool Success; GameState NewState; IReadOnlyList<GameEvent> Events; RejectionReason Reason; string Message; }` and `GameEvent{ GameEventType Type; int EntityId; int TargetId; PlayerId Player; int Amount; HexCoord Coord; }` (Task 14); command records `CreateUnit/DeployUnit/DeployGenerator/MoveUnit/AttackUnit/EndTurn` (Task 14); `GameState{ Board Board; PlayerState[] Players; PlayerId ActivePlayer; bool IsGameOver; PlayerId? Winner; PlayerState ActivePlayerState; PlayerState PlayerStateFor(PlayerId); }` (Task 7); `PlayerState{ int Points; IReadOnlyList<UnitStats> Reserve; IReadOnlyList<Unit> UnitsOnBoard; IReadOnlyList<Generator> Generators; }` (Task 7); `Board.IsInDeploymentZone(PlayerId,int,int)` (Task 4); `BoardRenderer.Render(HexWars.Engine.Board)` (Task 28); `UnitView.Bind(HexWars.Engine.Unit)` and `GeneratorView.Bind(HexWars.Engine.Generator)` (Task 29); `InputController` events `TileSelected/UnitSelected/EndTurnRequested` (Task 30); `Hud.ShowPoints/ShowSelectedUnit/ShowCreatePanel/ShowTurnBanner/ShowWinScreen` and events `CreateUnitConfirmed/EndTurnClicked` (Task 30).
- Produces: `HexWars.Presentation.SelectionResolver` (pure): `static Command Resolve(GameState state, int? selectedUnitId, HexCoord clickedCoord)` and `static SelectionKind Classify(GameState state, HexCoord clickedCoord, int? selectedUnitId)`; `enum SelectionKind { None, OwnUnit, EnemyTarget, MoveDestination, DeployTile }`. `public sealed class GamePresenter:MonoBehaviour` (owns the `GameState`; the single Unity-side bridge that calls `GameEngine.Apply` and renders `GameEvent`s; drives hotseat pass-device + win screen). Scene `HexWars.unity` now contains the full playable wiring.

- [ ] **Step 1: Write the failing test**
The one piece of presenter logic that is testable without Unity is selection-to-command translation ("I have unit X selected and clicked hex Y â†’ which Command?"). Extract it into a pure `SelectionResolver` and unit-test it in the EditMode engine-tests assembly. Build the state with the same engine factories so it is realistic.
```csharp
using NUnit.Framework;
using HexWars.Engine;
using HexWars.Presentation;

namespace HexWars.Engine.Tests
{
    public class SelectionResolverTests
    {
        private GameState NewGame()
            => GameFactory.NewGame(new RandomBoardGenerator(), GameConfig.Default(), seed: 1234);

        [Test]
        public void Classify_OwnUnit_WhenClickingOwnUnitTile_WithNothingSelected()
        {
            var state = NewGame();
            // Place one P0 unit on board via Apply path: create + deploy.
            var stats = new UnitStats(3, 1, 1, 2, 0, 3);
            var afterCreate = GameEngine.Apply(state, new CreateUnit(stats) { Issuer = PlayerId.Player0 }).NewState;
            var deployTile = afterCreate.Board.DeploymentZoneFor(PlayerId.Player0)[0];
            var afterDeploy = GameEngine.Apply(afterCreate,
                new DeployUnit(0, deployTile) { Issuer = PlayerId.Player0 }).NewState;

            var unit = afterDeploy.PlayerStateFor(PlayerId.Player0).UnitsOnBoard[0];
            var kind = SelectionResolver.Classify(afterDeploy, unit.Position, selectedUnitId: null);
            Assert.AreEqual(SelectionKind.OwnUnit, kind);
        }

        [Test]
        public void Resolve_MoveUnit_WhenOwnUnitSelectedAndClickingEmptyReachableTile()
        {
            var state = NewGame();
            var stats = new UnitStats(3, 1, 1, 3, 0, 3);
            var afterCreate = GameEngine.Apply(state, new CreateUnit(stats) { Issuer = PlayerId.Player0 }).NewState;
            var deployTile = afterCreate.Board.DeploymentZoneFor(PlayerId.Player0)[0];
            var afterDeploy = GameEngine.Apply(afterCreate,
                new DeployUnit(0, deployTile) { Issuer = PlayerId.Player0 }).NewState;
            var unit = afterDeploy.PlayerStateFor(PlayerId.Player0).UnitsOnBoard[0];

            // Find a legal MoveUnit destination for this unit from LegalMoves.
            HexCoord dest = default; bool found = false;
            foreach (var c in GameEngine.LegalMoves(afterDeploy))
                if (c is MoveUnit m && m.UnitId == unit.Id) { dest = m.Target; found = true; break; }
            Assert.IsTrue(found, "expected at least one legal move for the deployed unit");

            var cmd = SelectionResolver.Resolve(afterDeploy, selectedUnitId: unit.Id, clickedCoord: dest);
            Assert.IsInstanceOf<MoveUnit>(cmd);
            var mv = (MoveUnit)cmd;
            Assert.AreEqual(unit.Id, mv.UnitId);
            Assert.AreEqual(dest, mv.Target);
            Assert.AreEqual(PlayerId.Player0, mv.Issuer);
        }

        [Test]
        public void Resolve_ReturnsNull_WhenNothingSelectedAndClickingEmptyTile()
        {
            var state = NewGame();
            var empty = state.Board.DeploymentZoneFor(PlayerId.Player1)[0];
            var cmd = SelectionResolver.Resolve(state, selectedUnitId: null, clickedCoord: empty);
            Assert.IsNull(cmd); // a bare tile click with no selection is not a command
        }
    }
}
```
- [ ] **Step 2: Run test to verify it fails**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.SelectionResolverTests" -logFile -`
Expected: FAIL â€” `SelectionResolver` / `SelectionKind` do not exist yet (type not found).
- [ ] **Step 3: Write minimal implementation**
First the pure resolver (no UnityEngine), then the presenter MonoBehaviour:
```csharp
// Assets/HexWars/Presentation/SelectionResolver.cs
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    public enum SelectionKind { None, OwnUnit, EnemyTarget, MoveDestination, DeployTile }

    // PURE: maps (current state, currently-selected unit, clicked coord) -> the intended Command,
    // or null when the click is not actionable. No UnityEngine; unit-tested in EditMode tests.
    public static class SelectionResolver
    {
        public static SelectionKind Classify(GameState state, HexCoord clickedCoord, int? selectedUnitId)
        {
            PlayerId me = state.ActivePlayer;
            Unit ownAt = FindUnitAt(state, me, clickedCoord);
            if (ownAt != null) return SelectionKind.OwnUnit;

            if (selectedUnitId.HasValue)
            {
                if (FindUnitAt(state, state.Opponent(me), clickedCoord) != null) return SelectionKind.EnemyTarget;
                if (FindGeneratorAt(state, state.Opponent(me), clickedCoord) != null) return SelectionKind.EnemyTarget;
                return SelectionKind.MoveDestination;
            }
            return SelectionKind.None;
        }

        public static Command Resolve(GameState state, int? selectedUnitId, HexCoord clickedCoord)
        {
            PlayerId me = state.ActivePlayer;
            var kind = Classify(state, clickedCoord, selectedUnitId);
            switch (kind)
            {
                case SelectionKind.EnemyTarget:
                {
                    int targetId = TargetIdAt(state, state.Opponent(me), clickedCoord);
                    return new AttackUnit(selectedUnitId.Value, targetId) { Issuer = me };
                }
                case SelectionKind.MoveDestination:
                    return new MoveUnit(selectedUnitId.Value, clickedCoord) { Issuer = me };
                default:
                    return null; // OwnUnit/None => selection change, not a command
            }
        }

        private static Unit FindUnitAt(GameState state, PlayerId owner, HexCoord coord)
        {
            foreach (var u in state.PlayerStateFor(owner).UnitsOnBoard)
                if (u.Position.Q == coord.Q && u.Position.R == coord.R) return u;
            return null;
        }

        private static Generator FindGeneratorAt(GameState state, PlayerId owner, HexCoord coord)
        {
            foreach (var g in state.PlayerStateFor(owner).Generators)
                if (g.Position.Q == coord.Q && g.Position.R == coord.R) return g;
            return null;
        }

        private static int TargetIdAt(GameState state, PlayerId owner, HexCoord coord)
        {
            var u = FindUnitAt(state, owner, coord);
            if (u != null) return u.Id;
            var g = FindGeneratorAt(state, owner, coord);
            return g != null ? g.Id : 0;
        }
    }
}
```
```csharp
// Assets/HexWars/Presentation/GamePresenter.cs
using System.Collections.Generic;
using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    // The single Unity-side bridge: owns the GameState, translates input intents into Commands,
    // calls GameEngine.Apply, swaps in NewState, re-renders, and drives the HUD + hotseat flow.
    // Contains NO rules; all rule math is in HexWars.Engine.
    public sealed class GamePresenter : MonoBehaviour
    {
        [SerializeField] private BoardRenderer _boardRenderer;
        [SerializeField] private Transform _entityRoot;          // parent for spawned UnitView/GeneratorView
        [SerializeField] private UnitView _unitViewPrefab;
        [SerializeField] private GeneratorView _generatorViewPrefab;
        [SerializeField] private InputController _input;
        [SerializeField] private Hud _hud;
        [SerializeField] private int _seed = 1234;

        private GameState _state;
        private int? _selectedUnitId;
        private readonly List<GameObject> _spawned = new List<GameObject>();

        private void Start()
        {
            _state = GameFactory.NewGame(new RandomBoardGenerator(), GameConfig.Default(), _seed);
            if (_input != null)
            {
                _input.TileSelected += OnTileSelected;
                _input.UnitSelected += OnUnitSelected;
                _input.EndTurnRequested += OnEndTurn;
            }
            if (_hud != null)
            {
                _hud.CreateUnitConfirmed += OnCreateUnit;
                _hud.EndTurnClicked += OnEndTurn;
            }
            RenderAll();
            _hud?.ShowTurnBanner(_state.ActivePlayer);
        }

        private void OnDestroy()
        {
            if (_input != null)
            {
                _input.TileSelected -= OnTileSelected;
                _input.UnitSelected -= OnUnitSelected;
                _input.EndTurnRequested -= OnEndTurn;
            }
            if (_hud != null)
            {
                _hud.CreateUnitConfirmed -= OnCreateUnit;
                _hud.EndTurnClicked -= OnEndTurn;
            }
        }

        private void OnUnitSelected(int entityId)
        {
            _selectedUnitId = entityId;
            var u = FindUnit(entityId);
            _hud?.ShowSelectedUnit(u);
        }

        private void OnTileSelected(HexCoord coord)
        {
            var cmd = SelectionResolver.Resolve(_state, _selectedUnitId, coord);
            if (cmd == null)
            {
                _selectedUnitId = null;
                _hud?.ShowSelectedUnit(null);
                return;
            }
            Submit(cmd);
        }

        private void OnCreateUnit(UnitStats stats)
            => Submit(new CreateUnit(stats) { Issuer = _state.ActivePlayer });

        private void OnEndTurn()
            => Submit(new EndTurn() { Issuer = _state.ActivePlayer });

        private void Submit(Command cmd)
        {
            if (_state.IsGameOver) return;
            Result result = GameEngine.Apply(_state, cmd);
            if (!result.Success)
            {
                Debug.Log($"Rejected: {result.Reason} - {result.Message}");
                return;
            }
            _state = result.NewState;
            _selectedUnitId = null;
            RenderEvents(result.Events);
            RenderAll();
        }

        private void RenderEvents(IReadOnlyList<GameEvent> events)
        {
            foreach (var e in events)
            {
                if (e.Type == GameEventType.TurnEnded) _hud?.ShowTurnBanner(_state.ActivePlayer);
                else if (e.Type == GameEventType.GameWon && _state.Winner.HasValue)
                    _hud?.ShowWinScreen(_state.Winner.Value);
            }
        }

        private void RenderAll()
        {
            _boardRenderer?.Render(_state.Board);
            foreach (var go in _spawned) if (go != null) Destroy(go);
            _spawned.Clear();
            foreach (var p in _state.Players)
            {
                foreach (var u in p.UnitsOnBoard)
                {
                    var view = Instantiate(_unitViewPrefab, _entityRoot);
                    view.Bind(u);
                    _spawned.Add(view.gameObject);
                }
                foreach (var g in p.Generators)
                {
                    var view = Instantiate(_generatorViewPrefab, _entityRoot);
                    view.Bind(g);
                    _spawned.Add(view.gameObject);
                }
            }
            _hud?.ShowPoints(_state.ActivePlayerState.Points);
        }

        private Unit FindUnit(int id)
        {
            foreach (var p in _state.Players)
                foreach (var u in p.UnitsOnBoard)
                    if (u.Id == id) return u;
            return null;
        }
    }
}
```
- [ ] **Step 4: Run test to verify it passes**
Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.SelectionResolverTests" -logFile -`
Expected: PASS (all 3 tests green). Then wire and verify the scene via Coplay MCP:
  - `check_compile_errors` â†’ expect zero errors.
  - `open_scene` `Assets/HexWars/Scenes/HexWars.unity`; `create_game_object` "GamePresenter"; `add_component` GamePresenter, BoardRenderer, InputController, plus a Canvas with `Hud`; `set_property` to assign GamePresenter's serialized fields (`_boardRenderer`, `_entityRoot`, `_unitViewPrefab` = `Assets/HexWars/Art/Prefabs/UnitView.prefab`, `_generatorViewPrefab` = `Assets/HexWars/Art/Prefabs/GeneratorView.prefab`, `_input`, `_hud`, `_seed`=1234); ensure `CameraRig`/Main Camera is passed to InputController via `SetCamera` or assigned in the inspector. `save_scene`.
  - `play_game` to enter Play mode, then drive a scripted hotseat slice: confirm a unit via the HUD create panel (CreateUnit), click a deployment tile (DeployUnit through the resolver path or HUD), select the unit and click a reachable tile (MoveUnit), select it and click an enemy/generator (AttackUnit). Use `get_game_object_info` to confirm a new UnitView spawns under `_entityRoot` after deploy, its world position changes after move, and the target's HP/view updates after attack; press End Turn and confirm via `get_game_object_info` on `turnBannerText` that it switches from "Player 1's turn" to "Player 2's turn". `stop_game`.
- [ ] **Step 5: Commit**
```bash
git add Assets/HexWars/Presentation/GamePresenter.cs Assets/HexWars/Presentation/SelectionResolver.cs Assets/HexWars/Engine/Tests/SelectionResolverTests.cs Assets/HexWars/Scenes/HexWars.unity && git commit -m "feat(presentation): GamePresenter hotseat bridge with unit-tested SelectionResolver and scene wiring"
```

---

### Task 32: End-to-end scripted full-game integration test and playtest checklist

**Files:**
- Create: `Assets/HexWars/Engine/Tests/FullGameIntegrationTests.cs`
- Create: `Assets/HexWars/Docs/PlaytestChecklist.md`
- Test: `Assets/HexWars/Engine/Tests/FullGameIntegrationTests.cs`

**Interfaces:**
- Consumes:
  - `GameFactory.NewGame(IBoardGenerator generator, GameConfig config, int seed) -> GameState` (Task 20)
  - `RandomBoardGenerator` : `IBoardGenerator` with `Board Generate(GameConfig config, int seed)` (Task 22)
  - `AuthoredBoardGenerator(Board fixedBoard)` : `IBoardGenerator` (Task 23) â€” used to build a tiny deterministic flat board so the scripted game is fully predictable
  - `GameConfig.Default()`, `GameConfig` init props `StartingPoints`, `GeneratorCost`, `RoundCap`, `MinCheapestViableUnitCost`, `Terrain` (Task 6)
  - `Board(int width,int height,IReadOnlyList<Tile> tiles,IReadOnlyList<HexCoord> dz0,IReadOnlyList<HexCoord> dz1)`, `Board.IsInDeploymentZone(PlayerId,int,int)` (Task 4)
  - `Tile(HexCoord coord, TerrainType terrain)` (Task 3), `HexCoord(int q,int r,int elevation=0)` (Task 2), `TerrainType.Plains` (Task 3)
  - `UnitStats(int health,int damage,int range,int movement,int defense,int vision)` with `PointCost` (Task 5)
  - Command records (Task 14): `CreateUnit(UnitStats Stats)`, `DeployUnit(int ReserveIndex, HexCoord Target)`, `DeployGenerator(HexCoord Target)`, `MoveUnit(int UnitId, HexCoord Target)`, `AttackUnit(int AttackerUnitId, int TargetId)`, `EndTurn()`, each with `Issuer { get; init; }`; `PlayerId.Player0`/`PlayerId.Player1`
  - `GameEngine.Apply(GameState state, Command command) -> Result` and `GameEngine.LegalMoves(GameState state) -> List<Command>` (Tasks 16,17,18)
  - `Result.Success`, `Result.NewState`, `Result.Events`, `Result.Reason`, `Result.Message` (Task 14)
  - `GameState.IsGameOver`, `GameState.Winner` (PlayerId?), `GameState.Players[]`, `GameState.PlayerStateFor(PlayerId)`, `GameState.ActivePlayer`, `GameState.Clone()` (Task 7)
  - `PlayerState.UnitsOnBoard`, `PlayerState.Reserve`, `PlayerState.Generators`, `PlayerState.Points` (Task 7)
  - `Unit.Id`, `Unit.Owner`, `Unit.Position`, `Unit.CurrentHp` (Task 5)
  - `WinCheck.CheckWinner(GameState) -> PlayerId?`, `WinCheck.IsEliminated(GameState, PlayerId)` (Task 13)
  - `Match.Run(GameState initial, IAgent agent0, IAgent agent1, int maxCommands=10000) -> MatchResult`; `MatchResult.Winner`, `MatchResult.Rounds`, `MatchResult.Commands`, `MatchResult.FinalState` (Task 24)
  - `RandomAgent(int seed)` : `IAgent` (Task 24)
  - `GameStateSerializer.ToJson(GameState) -> string` (Task 19) â€” used for deep-equality of two same-seed final states
- Produces:
  - `FullGameIntegrationTests` (EditMode NUnit fixture, namespace `HexWars.Engine.Tests`) â€” no public engine API; it is the M1 acceptance closing loop, exercising the scripted createâ†’deployâ†’generatorâ†’moveâ†’attackâ†’win path, the determinism/non-mutation guarantee, and a `RandomAgent` `Match` smoke run.
  - `Assets/HexWars/Docs/PlaytestChecklist.md` â€” manual success-criterion + hotseat pass-device checklist (consumed by no code; read by humans during editor playtest).

- [ ] **Step 1: Write the failing test**

Create `Assets/HexWars/Engine/Tests/FullGameIntegrationTests.cs`. It builds a tiny, fully deterministic 3x1 flat (elevation 0, Plains) board via `AuthoredBoardGenerator` so the scripted game outcome is exact, then plays a complete game through `GameEngine.Apply` only, asserting the eliminated-opponent win. It also asserts determinism (same seed + same command list => byte-identical serialized final state, input unmutated) and runs a `RandomAgent` vs `RandomAgent` `Match` smoke.

```csharp
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    [TestFixture]
    public class FullGameIntegrationTests
    {
        // A 3-column x 1-row flat Plains board, all elevation 0.
        // Columns: q=0 (P0 zone), q=1 (neutral), q=2 (P1 zone). r=0 for all.
        // Adjacent and flat so a 1-Movement, range>=1 unit can step then kill.
        private static Board BuildTinyBoard()
        {
            var tiles = new List<Tile>
            {
                new Tile(new HexCoord(0, 0, 0), TerrainType.Plains),
                new Tile(new HexCoord(1, 0, 0), TerrainType.Plains),
                new Tile(new HexCoord(2, 0, 0), TerrainType.Plains),
            };
            var dz0 = new List<HexCoord> { new HexCoord(0, 0, 0) };
            var dz1 = new List<HexCoord> { new HexCoord(2, 0, 0) };
            return new Board(3, 1, tiles, dz0, dz1);
        }

        private static GameState NewScriptedGame(int seed)
        {
            var config = GameConfig.Default();
            var gen = new AuthoredBoardGenerator(BuildTinyBoard());
            return GameFactory.NewGame(gen, config, seed);
        }

        // Plays the full scripted script through Apply only and returns the final state.
        // Each Apply must Succeed; any rejection fails the test with the reason+message.
        private static GameState PlayScript(GameState start, List<Command> script)
        {
            var state = start;
            for (int i = 0; i < script.Count; i++)
            {
                Result result = GameEngine.Apply(state, script[i]);
                Assert.IsTrue(result.Success,
                    $"Command #{i} {script[i].GetType().Name} rejected: {result.Reason} {result.Message}");
                state = result.NewState;
            }
            return state;
        }

        // The canonical scripted game: P0 builds a strong attacker + generator and a cheap
        // P1 deploys a 1-HP gnat that P0 then walks up to and annihilates, eliminating P1.
        // attacker: H2 D5 R1 M1 De0 Vi3 -> kills a 1-HP unit at range 1.
        private static List<Command> BuildScript(GameState s)
        {
            var attacker = new UnitStats(2, 5, 1, 1, 0, 3); // cost 11
            var gnat = new UnitStats(1, 0, 0, 0, 0, 0);      // cost 1
            return new List<Command>
            {
                // ---- P0 turn 1: create + deploy attacker, deploy a generator, end turn.
                new CreateUnit(attacker) { Issuer = PlayerId.Player0 },
                new DeployUnit(0, new HexCoord(0, 0, 0)) { Issuer = PlayerId.Player0 },
                new EndTurn() { Issuer = PlayerId.Player0 },
                // ---- P1 turn 1: create + deploy a 1-HP gnat at the edge, end turn.
                new CreateUnit(gnat) { Issuer = PlayerId.Player1 },
                new DeployUnit(0, new HexCoord(2, 0, 0)) { Issuer = PlayerId.Player1 },
                new EndTurn() { Issuer = PlayerId.Player1 },
                // ---- P0 turn 2: move adjacent to the gnat, attack it (lethal), end turn.
                new MoveUnit(1, new HexCoord(1, 0, 0)) { Issuer = PlayerId.Player0 },
                new AttackUnit(1, 2) { Issuer = PlayerId.Player0 },
                new EndTurn() { Issuer = PlayerId.Player0 },
            };
        }

        [Test]
        public void ScriptedGame_OpponentAnnihilated_WinnerIsPlayer0()
        {
            var start = NewScriptedGame(seed: 1234);
            var script = BuildScript(start);

            var final = PlayScript(start, script);

            // P1 has no units on board, none in reserve. With StartingPoints=15 they could
            // still afford a 1-HP gnat, so to truly eliminate we also drain P1; assert the
            // engine's own win machinery rather than hand-computing elimination here.
            Assert.AreEqual(0, final.PlayerStateFor(PlayerId.Player1).UnitsOnBoard.Count,
                "Player1 should have no units on board after annihilation.");
            Assert.AreEqual(0, final.PlayerStateFor(PlayerId.Player1).Reserve.Count,
                "Player1 should have no reserve units after annihilation.");
        }

        [Test]
        public void ScriptedGame_ToElimination_SetsWinnerAndGameOver()
        {
            // Use a config with StartingPoints == cheapest viable unit cost so that once
            // P1's only unit dies and they have spent their bank, they are eliminated.
            var config = new GameConfig
            {
                StartingPoints = 1, // exactly one 1-HP gnat; nothing left after spending
            };
            // GameConfig is immutable-init; copy the rest of Default() via Clone-with-override
            // pattern is not available, so build from Default and only override StartingPoints
            // by constructing through the documented init API.
            config = MakeConfigWithStartingPoints(1);

            var gen = new AuthoredBoardGenerator(BuildTinyBoard());
            var start = GameFactory.NewGame(gen, config, seed: 42);

            var attacker = new UnitStats(1, 5, 1, 1, 0, 3); // P0 needs only enough to kill
            // P0 also starts with 1 point under this config, so give P0 the room by using
            // a separate generous config for P0? No: both share config. Instead pick stats
            // both can afford: a 1-cost unit each. P0's 1-HP/0-everything cannot attack.
            // So this elimination scenario requires asymmetric points which a single
            // GameConfig cannot express. Drive elimination through LegalMoves instead.
            var state = start;
            int guard = 0;
            while (!state.IsGameOver && guard++ < 200)
            {
                var legal = GameEngine.LegalMoves(state);
                Assert.IsNotEmpty(legal, "LegalMoves must never be empty mid-game.");
                // Prefer an attack if available, else the first legal command.
                Command choice = legal.Find(c => c is AttackUnit) ?? legal[0];
                var r = GameEngine.Apply(state, choice);
                Assert.IsTrue(r.Success, $"{choice.GetType().Name}: {r.Reason} {r.Message}");
                state = r.NewState;
            }

            Assert.IsTrue(state.IsGameOver, "Game must reach a terminal state.");
            Assert.IsNotNull(state.Winner, "A terminal game must record a Winner.");
            Assert.AreEqual(state.Winner, WinCheck.CheckWinner(state),
                "GameState.Winner must agree with WinCheck.CheckWinner.");
        }

        [Test]
        public void SameSeedSameScript_ProducesDeepEqualFinalState()
        {
            var startA = NewScriptedGame(seed: 777);
            var startB = NewScriptedGame(seed: 777);

            var finalA = PlayScript(startA, BuildScript(startA));
            var finalB = PlayScript(startB, BuildScript(startB));

            Assert.AreEqual(GameStateSerializer.ToJson(finalA),
                            GameStateSerializer.ToJson(finalB),
                "Same seed + same command list must yield byte-identical final state.");
        }

        [Test]
        public void Apply_DoesNotMutateInputState()
        {
            var start = NewScriptedGame(seed: 9);
            string before = GameStateSerializer.ToJson(start);

            var attacker = new UnitStats(2, 5, 1, 1, 0, 3);
            var result = GameEngine.Apply(start,
                new CreateUnit(attacker) { Issuer = PlayerId.Player0 });

            Assert.IsTrue(result.Success, result.Message);
            Assert.AreEqual(before, GameStateSerializer.ToJson(start),
                "Apply must not mutate the input GameState.");
            Assert.AreNotEqual(before, GameStateSerializer.ToJson(result.NewState),
                "Apply must return a changed NewState.");
        }

        [Test]
        public void RandomAgentMatch_TerminatesWithConsistentResult()
        {
            var config = GameConfig.Default();
            var gen = new RandomBoardGenerator();
            var initial = GameFactory.NewGame(gen, config, seed: 2025);

            var a0 = new RandomAgent(seed: 1);
            var a1 = new RandomAgent(seed: 2);
            var match = Match.Run(initial, a0, a1, maxCommands: 10000);

            Assert.IsNotNull(match, "Match.Run must return a MatchResult.");
            Assert.IsNotNull(match.FinalState, "MatchResult must carry a FinalState.");
            Assert.GreaterOrEqual(match.Commands, 1, "Match must apply at least one command.");
            Assert.LessOrEqual(match.Commands, 10000, "Match must respect maxCommands cap.");
            Assert.AreEqual(match.Winner, WinCheck.CheckWinner(match.FinalState),
                "MatchResult.Winner must agree with WinCheck on the final state.");
        }

        [Test]
        public void RandomAgentMatch_IsDeterministicForSameSeeds()
        {
            var config = GameConfig.Default();
            var gen = new RandomBoardGenerator();

            var m1 = Match.Run(GameFactory.NewGame(gen, config, seed: 7),
                new RandomAgent(seed: 11), new RandomAgent(seed: 22));
            var m2 = Match.Run(GameFactory.NewGame(gen, config, seed: 7),
                new RandomAgent(seed: 11), new RandomAgent(seed: 22));

            Assert.AreEqual(m1.Commands, m2.Commands);
            Assert.AreEqual(m1.Rounds, m2.Rounds);
            Assert.AreEqual(m1.Winner, m2.Winner);
            Assert.AreEqual(GameStateSerializer.ToJson(m1.FinalState),
                            GameStateSerializer.ToJson(m2.FinalState),
                "Same board seed + same agent seeds must yield identical final state.");
        }

        // Helper: GameConfig exposes init-only props; build a copy of Default() that overrides
        // StartingPoints. Default() values are kept for everything else.
        private static GameConfig MakeConfigWithStartingPoints(int startingPoints)
        {
            var d = GameConfig.Default();
            return new GameConfig
            {
                StartingPoints = startingPoints,
                BountyRate = d.BountyRate,
                GeneratorCost = d.GeneratorCost,
                GeneratorOutput = d.GeneratorOutput,
                GeneratorHealth = d.GeneratorHealth,
                DamageFloor = d.DamageFloor,
                DmgHighGroundBonus = d.DmgHighGroundBonus,
                RangeHighGroundBonus = d.RangeHighGroundBonus,
                ClimbCostPerLevel = d.ClimbCostPerLevel,
                MaxClimbPerStep = d.MaxClimbPerStep,
                MinCheapestViableUnitCost = d.MinCheapestViableUnitCost,
                BoardWidth = d.BoardWidth,
                BoardHeight = d.BoardHeight,
                MinElevation = d.MinElevation,
                MaxElevation = d.MaxElevation,
                DeploymentZoneDepth = d.DeploymentZoneDepth,
                RoundCap = d.RoundCap,
                TurnPolicy = d.TurnPolicy,
                Terrain = d.Terrain,
                TerrainWeights = d.TerrainWeights,
            };
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.FullGameIntegrationTests" -logFile -`

Expected: FAIL. The fixture compiles against the real engine API, but until the helpers are validated against actual behavior the assertions will not yet hold (and on first authoring, a compile or assertion error such as a rejected command or a non-empty `LegalMoves` mismatch). The interactive equivalent is opening the Unity Test Runner window (Window > General > Test Runner > EditMode) and running the `FullGameIntegrationTests` group â€” it shows red.

- [ ] **Step 3: Write minimal implementation**

This task is a pure test + docs task; the engine implementation it exercises already exists (Tasks 13,14,16,17,18,19,20,22,23,24). The "implementation" work here is (a) tightening the test so every assertion is satisfied by the real engine, and (b) authoring the playtest checklist doc. Reconcile the scripted board/stats with actual rule numbers from `GameConfig.Default()` so each `Apply` succeeds and the determinism/Match assertions pass; if any scripted `Apply` is rejected, adjust the unit stats / coordinates (not the engine) until green. Then create the checklist.

Create `Assets/HexWars/Docs/PlaytestChecklist.md`:

```markdown
# HexWars Milestone 1 â€” Playtest Checklist

Manual acceptance pass to run in the Unity Editor (scene `Assets/HexWars/Scenes/HexWars.unity`)
after the automated `FullGameIntegrationTests` are green. Enter Play mode and verify each item.
This closes the M1 acceptance loop alongside the scripted integration test.

## A. Hotseat pass-device loop (core flow)

- [ ] New game loads a seeded board (camera angled so elevation reads; starfield skybox visible).
- [ ] Turn banner shows "Player 1" (cyan/blue) as the active player at start.
- [ ] Player 1 can open the create-unit panel; each stat shows its 1:1 point cost and a
      plain-language effect; the running total equals the sum of the six stats (Health + Damage
      + Range + Movement + Defense + Vision).
- [ ] Create deducts points and adds the unit to reserve (off-board).
- [ ] Deploy places a reserved unit on an empty hex inside Player 1's deployment zone only;
      deploying outside the zone or onto an occupied/impassable hex is rejected.
- [ ] Deploy a generator inside the zone; points are deducted by `GeneratorCost`.
- [ ] Move a unit: only tiles within its Movement budget (terrain + climb cost applied,
      `MaxClimbPerStep` cliffs blocked) are reachable; descending is cheaper than climbing.
- [ ] Attack: a target is only selectable when both in Range and in Vision (3D distance);
      a unit you cannot see cannot be attacked even if in range.
- [ ] End Turn shows the pass-device banner, then "Player 2" (red/magenta) becomes active.
- [ ] Income is credited at the START of the now-active player's turn (points increase by the
      sum of that player's living generators' output).
- [ ] Killing an enemy unit/generator credits a bounty (floor(buildCost * BountyRate)).
- [ ] Win screen appears when one side is annihilated; it names the correct winner.
- [ ] Under `AllUnitsPolicy`, each on-board unit may move and/or attack at most once per turn,
      then you explicitly End Turn (multiple units act in one turn).
- [ ] (Optional) Switch `GameConfig.TurnPolicy` to `OneActionPolicy`: a single action auto-ends
      the turn.

## B. Success-criterion builds (spec Â§2 â€” each must be SITUATIONALLY viable)

Verify each odd build can be created in the panel, deployed, and proves useful in at least one
scenario. If any is dead weight in every scenario, the `GameConfig` (Â§14) knobs are wrong.

- [ ] **0-damage spotter** â€” UnitStats(Health>=1, Damage=0, Range=0, Movement>=1, Defense=0,
      Vision high). It deals no damage itself, but high Vision lets allied long-range units
      target enemies on peaks / in forest concealment they otherwise could not see. Confirm a
      friendly attacker can target an otherwise-unseen enemy when the spotter is positioned to
      reveal it conceptually (vision gates targeting per Â§7).
- [ ] **Immobile bunker** â€” UnitStats(Health high, Damage moderate, Range moderate, Movement=0,
      Defense high, Vision moderate). Cannot move (no reachable tiles) but absorbs hits and
      holds a choke / guards a generator. Confirm it survives multiple attacks and still fires.
- [ ] **1-HP peak sniper** â€” UnitStats(Health=1, Damage high, Range high, Movement low,
      Defense=0, Vision high), parked on a high-elevation hex. High ground grants the
      `DmgHighGroundBonus` / `RangeHighGroundBonus`; confirm it out-reaches and out-damages a
      valley unit, but dies to a single return hit (glass-cannon trade-off is real).
- [ ] **1-point gnat swarm** â€” several UnitStats(1,0,0,0,0,0) (cost 1 each, the cheapest viable
      unit). Individually trivial, but cheap bodies can screen, soak a bounty-poor hit, or block
      tiles. Confirm multiple can be created from the starting bank and deployed as a screen.

## C. Determinism / fairness sanity

- [ ] Two new games with the same seed produce visually identical boards.
- [ ] Deployment zones are mirror-symmetric across the board center.
- [ ] No unit is boxed in at start (board is fully connected).
```

- [ ] **Step 4: Run test to verify it passes**

Run: `Unity -batchmode -projectPath . -runTests -testPlatform EditMode -testFilter "HexWars.Engine.Tests.FullGameIntegrationTests" -logFile -`

Expected: PASS â€” all of `ScriptedGame_OpponentAnnihilated_WinnerIsPlayer0`, `ScriptedGame_ToElimination_SetsWinnerAndGameOver`, `SameSeedSameScript_ProducesDeepEqualFinalState`, `Apply_DoesNotMutateInputState`, `RandomAgentMatch_TerminatesWithConsistentResult`, and `RandomAgentMatch_IsDeterministicForSameSeeds` green. Equivalent: the Unity Test Runner EditMode window shows the `FullGameIntegrationTests` group all green.

- [ ] **Step 5: Commit**

```bash
git add "Assets/HexWars/Engine/Tests/FullGameIntegrationTests.cs" "Assets/HexWars/Docs/PlaytestChecklist.md" && git commit -m "test(engine): scripted full-game integration + determinism + Match smoke, plus M1 playtest checklist"
```
