# Territory Control — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add persistent per-hex *control* (capture by a unit standing on a hex), a `CaptureHex` command, its legal-move enumeration, and configurable **multiply-selectable win conditions** `{Annihilation, Economy, Score}` to the headless engine — leaving generators/economy-tick (Phase 2) and territory value + UI (Phase 3) out.

**Architecture:** Control is stored on the immutable `Board` (a `HexCoord → PlayerId` map), so it rides through every existing `GameState` rebuild via `state.Board` and only `CaptureHex` needs to build a new board — no `GameState` ctor changes. Win conditions become a `[Flags]` config the existing `WinCheck.IsTerminal`/`Resolve` honor. Score is a weighted composite over kills/points/army/territory; the "kills" term needs a small `DestroyedValue` tally on `PlayerState`.

**Tech Stack:** C# `netstandard2.1` engine (`engine/HexWars.Engine`), NUnit tests (`engine/HexWars.Engine.Tests`, target `net8.0`), built with `dotnet`.

## Global Constraints

- Engine project `HexWars.Engine` targets **netstandard2.1**, uses **`System.Random` only** (no `UnityEngine`), and must stay deterministic.
- Tests are **NUnit**: `[Test]`, `Assert.That(actual, Is.EqualTo(expected))`. Run with `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`.
- **Backward compatibility:** the existing 145 tests must stay green. New `GameConfig`/`PlayerState`/`Board` constructor parameters are **optional, added last**, and defaults preserve today's behavior (`WinConditions` defaults to `Annihilation` only).
- **No git attribution** in commit messages (no Co-Authored-By / tool credits).
- **Plugins DLL:** after engine changes, rebuild and copy `engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll` to `Assets/HexWars/Plugins/HexWars.Engine.dll` (final task) so Unity keeps compiling. Phase 1 adds no Unity-side calls, so this is hygiene, not a blocker.
- Build the engine with `dotnet build engine/HexWars.Engine -c Release`. The Windows dotnet is at `/c/Program Files/dotnet/dotnet.exe` from Git Bash; from PowerShell just use `dotnet`.

---

### Task 1: Per-hex control state on `Board`

**Files:**
- Modify: `engine/HexWars.Engine/Board.cs`
- Test: `engine/HexWars.Engine.Tests/BoardControlTests.cs` (create)

**Interfaces:**
- Produces: `Board.Controller(HexCoord) -> PlayerId?` (null = neutral); `Board.WithControl(HexCoord, PlayerId) -> Board` (new board, original unchanged); `Board.ControlledCount(PlayerId) -> int`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/BoardControlTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardControlTests
    {
        static Board OneTile() => new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });

        [Test]
        public void Hexes_StartNeutral()
        {
            var b = OneTile();
            Assert.That(b.Controller(new HexCoord(0, 0)), Is.Null);
            Assert.That(b.ControlledCount(PlayerId.Player0), Is.EqualTo(0));
        }

        [Test]
        public void WithControl_SetsController_AndIsImmutable()
        {
            var b = OneTile();
            var b2 = b.WithControl(new HexCoord(0, 0), PlayerId.Player1);

            Assert.That(b2.Controller(new HexCoord(0, 0)), Is.EqualTo(PlayerId.Player1));
            Assert.That(b2.ControlledCount(PlayerId.Player1), Is.EqualTo(1));
            Assert.That(b.Controller(new HexCoord(0, 0)), Is.Null, "original board must be unchanged");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: FAIL — `Board` has no `Controller`/`WithControl`/`ControlledCount`.

- [ ] **Step 3: Implement on `Board`**

In `engine/HexWars.Engine/Board.cs`, add a field next to `_zone1`:

```csharp
        private readonly Dictionary<HexCoord, PlayerId> _control;
```

Change the constructor signature and body to accept control (add the param **last**, keep existing params):

```csharp
        public Board(
            IEnumerable<Tile> tiles,
            IReadOnlyCollection<HexCoord>? zone0 = null,
            IReadOnlyCollection<HexCoord>? zone1 = null,
            IReadOnlyDictionary<HexCoord, PlayerId>? control = null)
        {
            _tiles = new Dictionary<HexCoord, Tile>();
            foreach (var tile in tiles)
                _tiles[tile.Coord] = tile;

            _zone0 = zone0 != null ? new HashSet<HexCoord>(zone0) : Empty;
            _zone1 = zone1 != null ? new HashSet<HexCoord>(zone1) : Empty;
            _control = control != null
                ? new Dictionary<HexCoord, PlayerId>(control)
                : new Dictionary<HexCoord, PlayerId>();
        }
```

Add these members (after `IsInDeploymentZone`):

```csharp
        /// <summary>Who currently controls this hex, or null if neutral.</summary>
        public PlayerId? Controller(HexCoord coord) =>
            _control.TryGetValue(coord, out var p) ? p : (PlayerId?)null;

        /// <summary>How many hexes the player controls.</summary>
        public int ControlledCount(PlayerId player)
        {
            int n = 0;
            foreach (var kv in _control) if (kv.Value == player) n++;
            return n;
        }

        /// <summary>A new board identical to this one but with <paramref name="coord"/> controlled by
        /// <paramref name="owner"/>. Immutable — this board is unchanged.</summary>
        public Board WithControl(HexCoord coord, PlayerId owner)
        {
            var control = new Dictionary<HexCoord, PlayerId>(_control) { [coord] = owner };
            return new Board(_tiles.Values, _zone0, _zone1, control);
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter BoardControlTests`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Board.cs engine/HexWars.Engine.Tests/BoardControlTests.cs
git commit -m "feat(engine): per-hex control state on Board (Controller/WithControl/ControlledCount)"
```

---

### Task 2: Win-condition config + capture/economy/score knobs

**Files:**
- Create: `engine/HexWars.Engine/WinBy.cs`
- Modify: `engine/HexWars.Engine/GameConfig.cs`
- Test: `engine/HexWars.Engine.Tests/TerritoryConfigTests.cs` (create)

**Interfaces:**
- Produces: `[Flags] enum WinBy { None=0, Annihilation=1, Economy=2, Score=4 }`; new `GameConfig` properties `WinConditions` (default `WinBy.Annihilation`), `CaptureCost` (default 3), `EconomyWinThreshold` (default 200), `ScoreKills`/`ScorePoints`/`ScoreArmy`/`ScoreTerritory` (all default 1); `GameConfig.Default(biomesEnabled, turnPolicy, winConditions, captureCost, economyWinThreshold, scoreKills, scorePoints, scoreArmy, scoreTerritory)` passthrough.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/TerritoryConfigTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TerritoryConfigTests
    {
        [Test]
        public void Default_IsAnnihilationOnly_WithControlDefaults()
        {
            var c = GameConfig.Default();
            Assert.That(c.WinConditions, Is.EqualTo(WinBy.Annihilation));
            Assert.That(c.CaptureCost, Is.EqualTo(3));
            Assert.That(c.EconomyWinThreshold, Is.EqualTo(200));
            Assert.That(c.ScoreKills, Is.EqualTo(1));
        }

        [Test]
        public void Default_PassesThroughWinConditions()
        {
            var c = GameConfig.Default(winConditions: WinBy.Economy | WinBy.Score, captureCost: 5);
            Assert.That(c.WinConditions, Is.EqualTo(WinBy.Economy | WinBy.Score));
            Assert.That(c.CaptureCost, Is.EqualTo(5));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter TerritoryConfigTests`
Expected: FAIL — `WinBy` and the new properties don't exist.

- [ ] **Step 3: Create `WinBy`**

Create `engine/HexWars.Engine/WinBy.cs`:

```csharp
using System;

namespace HexWars.Engine
{
    /// <summary>Multiply-selectable win conditions. Annihilation and Economy are instant (checked every
    /// command); Score resolves at the round cap. Any subset may be combined.</summary>
    [Flags]
    public enum WinBy
    {
        None = 0,
        Annihilation = 1,
        Economy = 2,
        Score = 4,
    }
}
```

- [ ] **Step 4: Add the config fields**

In `engine/HexWars.Engine/GameConfig.cs`, add properties (next to `BiomesEnabled`):

```csharp
        /// <summary>Which win conditions are active (any combination). Default: annihilation only.</summary>
        public WinBy WinConditions { get; }
        /// <summary>Flat cost to capture a hex (Phase 1; value-scaling comes with generators in Phase 2).</summary>
        public int CaptureCost { get; }
        /// <summary>Banked-points threshold for an Economy win.</summary>
        public int EconomyWinThreshold { get; }
        /// <summary>Score-composite weights (see WinCheck.Score).</summary>
        public int ScoreKills { get; }
        public int ScorePoints { get; }
        public int ScoreArmy { get; }
        public int ScoreTerritory { get; }
```

Add the constructor parameters (**after** `bool biomesEnabled = true`):

```csharp
            bool biomesEnabled = true,
            WinBy winConditions = WinBy.Annihilation,
            int captureCost = 3,
            int economyWinThreshold = 200,
            int scoreKills = 1,
            int scorePoints = 1,
            int scoreArmy = 1,
            int scoreTerritory = 1)
```

And assign them in the constructor body (after `BiomesEnabled = biomesEnabled;`):

```csharp
            WinConditions = winConditions;
            CaptureCost = captureCost;
            EconomyWinThreshold = economyWinThreshold;
            ScoreKills = scoreKills;
            ScorePoints = scorePoints;
            ScoreArmy = scoreArmy;
            ScoreTerritory = scoreTerritory;
```

Update `Default(...)` to accept and forward them. Change its signature and the closing constructor call:

```csharp
        public static GameConfig Default(bool biomesEnabled = true, ITurnPolicy? turnPolicy = null,
            WinBy winConditions = WinBy.Annihilation, int captureCost = 3, int economyWinThreshold = 200,
            int scoreKills = 1, int scorePoints = 1, int scoreArmy = 1, int scoreTerritory = 1) =>
            new GameConfig(new Dictionary<TerrainType, TerrainDef>
        {
            { TerrainType.Plains, new TerrainDef(moveCost: 1, concealment: 0, defense: 0, passable: true) },
            { TerrainType.Forest, new TerrainDef(moveCost: 2, concealment: 2, defense: 1, passable: true) },
            { TerrainType.Rough,  new TerrainDef(moveCost: 2, concealment: 1, defense: 1, passable: true) },
            { TerrainType.Water,  new TerrainDef(moveCost: 3, concealment: 0, defense: 0, passable: true) },
        }, turnPolicy: turnPolicy, biomesEnabled: biomesEnabled, winConditions: winConditions,
           captureCost: captureCost, economyWinThreshold: economyWinThreshold,
           scoreKills: scoreKills, scorePoints: scorePoints, scoreArmy: scoreArmy, scoreTerritory: scoreTerritory);
```

- [ ] **Step 5: Run tests to verify pass**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter TerritoryConfigTests`
Expected: PASS (2 tests). Also run the full suite to confirm no regression: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo` → 147 passed.

- [ ] **Step 6: Commit**

```bash
git add engine/HexWars.Engine/WinBy.cs engine/HexWars.Engine/GameConfig.cs engine/HexWars.Engine.Tests/TerritoryConfigTests.cs
git commit -m "feat(engine): WinBy flags + capture/economy/score config knobs on GameConfig (defaults preserve annihilation-only)"
```

---

### Task 3: `CaptureHex` command + handler + legal-move enumeration

**Files:**
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/RejectionReason.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/LegalMoves.cs`
- Test: `engine/HexWars.Engine.Tests/CaptureHexTests.cs` (create)

**Interfaces:**
- Consumes: `Board.WithControl`/`Controller` (Task 1); `GameConfig.CaptureCost` (Task 2); `PlayerState.WithPoints`; `Result.Ok`/`Reject`.
- Produces: `record CaptureHex(PlayerId Issuer, HexCoord Cell) : Command`; `RejectionReason.NoUnitOnHex`, `RejectionReason.AlreadyControlled`; a `CaptureHex` entry in `LegalMoves.For`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/CaptureHexTests.cs`:

```csharp
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class CaptureHexTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has one unit on A with `points`; board is a 1x3 strip so A exists. Plains, biomes off.
        static GameState State(int points, PlayerId active = PlayerId.Player0, PlayerId? controlA = null)
        {
            var tiles = new[]
            {
                new Tile(A, 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            };
            Board board = new Board(tiles);
            if (controlA != null) board = board.WithControl(A, controlA.Value);
            var p0 = new PlayerState(PlayerId.Player0, points, null,
                new[] { new Unit(1, PlayerId.Player0, S, A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, points);
            return new GameState(board, GameConfig.Default(biomesEnabled: false),
                new[] { p0, p1 }, active, round: 2, nextEntityId: 99);
        }

        [Test]
        public void Capture_FlipsControl_AndDeductsCost()
        {
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(10 - 3)); // default CaptureCost
        }

        [Test]
        public void Capture_RejectsWhenNoUnitOnHex()
        {
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, new HexCoord(1, 0)));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.NoUnitOnHex));
        }

        [Test]
        public void Capture_RejectsWhenAlreadyControlled()
        {
            var r = GameEngine.Apply(State(10, controlA: PlayerId.Player0), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.AlreadyControlled));
        }

        [Test]
        public void Capture_RejectsWhenBroke()
        {
            var r = GameEngine.Apply(State(2), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void Steal_RecapturesEnemyControlledHex()
        {
            var r = GameEngine.Apply(State(10, controlA: PlayerId.Player1), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void LegalMoves_IncludeCaptureOfGarrisonedUncontrolledHex()
        {
            var moves = LegalMoves.For(State(10));
            Assert.That(moves, Has.Some.Matches<Command>(m => m is CaptureHex ch && ch.Cell == A));
        }

        [Test]
        public void LegalMoves_ExcludeCaptureWhenAlreadyControlled()
        {
            var moves = LegalMoves.For(State(10, controlA: PlayerId.Player0));
            Assert.That(moves, Has.None.Matches<Command>(m => m is CaptureHex));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter CaptureHexTests`
Expected: FAIL — `CaptureHex`, the rejection reasons, the handler, and the legal-move don't exist (compile errors are an acceptable "fail" here).

- [ ] **Step 3: Add the command**

In `engine/HexWars.Engine/Command.cs`, add (before `EndTurn`):

```csharp
    /// <summary>Take control of the hex the issuer's unit stands on, paying the capture cost. Control is
    /// persistent (kept after the unit leaves) until an enemy unit recaptures it.</summary>
    public sealed record CaptureHex(PlayerId Issuer, HexCoord Cell) : Command(Issuer);
```

- [ ] **Step 4: Add the rejection reasons**

In `engine/HexWars.Engine/RejectionReason.cs`, add two values (before the closing brace of the enum):

```csharp
        NoUnitOnHex,
        AlreadyControlled,
```

- [ ] **Step 5: Add the handler + dispatch**

In `engine/HexWars.Engine/GameEngine.cs`, add a case to `Dispatch` (after the `AttackUnit` case):

```csharp
                case CaptureHex c: return ApplyCaptureHex(state, c);
```

Add the handler method (place after `ApplyAttackUnit`):

```csharp
        private static Result ApplyCaptureHex(GameState state, CaptureHex c)
        {
            if (!state.Board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);

            var player = state.Player(c.Issuer);
            bool hasUnit = false;
            foreach (var u in player.UnitsOnBoard)
                if (u.IsAlive && u.Cell == c.Cell) { hasUnit = true; break; }
            if (!hasUnit) return Result.Reject(state, RejectionReason.NoUnitOnHex);

            if (state.Board.Controller(c.Cell) == c.Issuer)
                return Result.Reject(state, RejectionReason.AlreadyControlled);

            if (player.Points < state.Config.CaptureCost)
                return Result.Reject(state, RejectionReason.InsufficientPoints);

            var players = state.Players.ToArray();
            players[(int)c.Issuer] = player.WithPoints(player.Points - state.Config.CaptureCost);
            var newBoard = state.Board.WithControl(c.Cell, c.Issuer);

            return Result.Ok(new GameState(newBoard, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, state.IsGameOver, state.Winner,
                state.MovedUnitIds, state.AttackedUnitIds));
        }
```

- [ ] **Step 6: Enumerate in LegalMoves**

In `engine/HexWars.Engine/LegalMoves.cs`, inside the `foreach (var unit in player.UnitsOnBoard)` loop (after the attack block, before its closing brace), add:

```csharp
                if (board.Controller(unit.Cell) != me
                    && player.Points >= state.Config.CaptureCost)
                    moves.Add(new CaptureHex(me, unit.Cell));
```

- [ ] **Step 7: Run tests to verify pass**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter CaptureHexTests`
Expected: PASS (7 tests). Note: "garrison-safe" is covered implicitly — capture requires the issuer's own unit on the hex, and one-unit-per-column means an enemy can never stand on your garrisoned hex to capture it.

- [ ] **Step 8: Commit**

```bash
git add engine/HexWars.Engine/Command.cs engine/HexWars.Engine/RejectionReason.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine.Tests/CaptureHexTests.cs
git commit -m "feat(engine): CaptureHex command + handler + legal-move (unit on hex, flat cost, persistent, steal by recapture)"
```

---

### Task 4: Kill tally (`PlayerState.DestroyedValue`)

**Files:**
- Modify: `engine/HexWars.Engine/PlayerState.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Test: `engine/HexWars.Engine.Tests/KillTallyTests.cs` (create)

**Interfaces:**
- Produces: `PlayerState.DestroyedValue` (int, cumulative enemy value destroyed); `PlayerState.WithDestroyed(int delta) -> PlayerState`. `WithPoints` preserves it. `ApplyAttackUnit` adds the killed entity's value to the attacker's `DestroyedValue`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/KillTallyTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class KillTallyTests
    {
        [Test]
        public void NewPlayer_HasZeroDestroyedValue()
        {
            Assert.That(new PlayerState(PlayerId.Player0, 0).DestroyedValue, Is.EqualTo(0));
        }

        [Test]
        public void WithPoints_PreservesDestroyedValue()
        {
            var p = new PlayerState(PlayerId.Player0, 5).WithDestroyed(7);
            Assert.That(p.WithPoints(99).DestroyedValue, Is.EqualTo(7));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter KillTallyTests`
Expected: FAIL — `DestroyedValue`/`WithDestroyed` don't exist.

- [ ] **Step 3: Add the field to `PlayerState`**

In `engine/HexWars.Engine/PlayerState.cs`, add the property (after `Generators`):

```csharp
        /// <summary>Cumulative point value of enemy entities this player has destroyed (for Score).</summary>
        public int DestroyedValue { get; }
```

Add the constructor parameter (**last**) and assignment:

```csharp
            IReadOnlyList<Generator>? generators = null,
            int destroyedValue = 0)
        {
            Id = id;
            Points = points;
            Barracks = barracks ?? NoBarracks;
            UnitsOnBoard = unitsOnBoard ?? NoUnits;
            Generators = generators ?? NoGenerators;
            DestroyedValue = destroyedValue;
        }
```

Update `WithPoints` and add `WithDestroyed`:

```csharp
        public PlayerState WithPoints(int points) =>
            new PlayerState(Id, points, Barracks, UnitsOnBoard, Generators, DestroyedValue);

        public PlayerState WithDestroyed(int delta) =>
            new PlayerState(Id, Points, Barracks, UnitsOnBoard, Generators, DestroyedValue + delta);
```

- [ ] **Step 4: Preserve the tally across the GameEngine reconstructions**

In `engine/HexWars.Engine/GameEngine.cs`, every place that rebuilds a `PlayerState` from an existing one must pass its `DestroyedValue` (the new last argument). Make these edits:

- `ApplyCreateUnit`: `new PlayerState(player.Id, player.Points - fee, barracks, player.UnitsOnBoard, player.Generators, player.DestroyedValue)`
- `ApplyDeployGenerator`: `new PlayerState(player.Id, player.Points - state.Config.GeneratorCost, player.Barracks, player.UnitsOnBoard, generators, player.DestroyedValue)`
- `ApplyDeployUnit`: `new PlayerState(player.Id, player.Points - cost, player.Barracks, units, player.Generators, player.DestroyedValue)`
- `ApplyMoveUnit`: `new PlayerState(player.Id, player.Points, player.Barracks, units, player.Generators, player.DestroyedValue)`
- `ApplyAttackUnit`, the enemy rebuild — **both** branches preserve the enemy's tally:
  - unit branch: `new PlayerState(enemy.Id, enemy.Points, enemy.Barracks, units, enemy.Generators, enemy.DestroyedValue)`
  - generator branch: `new PlayerState(enemy.Id, enemy.Points, enemy.Barracks, enemy.UnitsOnBoard, gens, enemy.DestroyedValue)`

- [ ] **Step 5: Credit kills to the attacker**

In `engine/HexWars.Engine/GameEngine.cs` `ApplyAttackUnit`, replace the `newPlayer` assignment:

```csharp
            var newPlayer = killed
                ? player.WithPoints(player.Points + CombatResolver.Bounty(buildCost, state.Config))
                       .WithDestroyed(buildCost)
                : player;
```

(`buildCost` is already the killed entity's point value — unit `PointCost` or `GeneratorCost`.)

- [ ] **Step 6: Run tests to verify pass + no regression**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: PASS — KillTallyTests (2) green and all prior tests still green (combat/attack tests unaffected because the tally defaults to 0 and is preserved).

- [ ] **Step 7: Commit**

```bash
git add engine/HexWars.Engine/PlayerState.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine.Tests/KillTallyTests.cs
git commit -m "feat(engine): PlayerState.DestroyedValue kill tally (credited on kill, preserved across rebuilds) for the Score composite"
```

---

### Task 5: `WinCheck.Score` composite

**Files:**
- Modify: `engine/HexWars.Engine/WinCheck.cs`
- Test: `engine/HexWars.Engine.Tests/ScoreTests.cs` (create)

**Interfaces:**
- Consumes: `GameConfig` score weights (Task 2); `PlayerState.DestroyedValue` (Task 4); `Board.ControlledCount` (Task 1).
- Produces: `WinCheck.Score(GameState, PlayerId) -> int` = `ScoreKills·destroyed + ScorePoints·points + ScoreArmy·armyValue + ScoreTerritory·controlledHexCount`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/ScoreTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ScoreTests
    {
        [Test]
        public void Score_IsWeightedSum_OfKillsPointsArmyTerritory()
        {
            var unit = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1); // PointCost computed from stats
            var p0 = new PlayerState(PlayerId.Player0, 10, null,
                new[] { new Unit(1, PlayerId.Player0, unit, new HexCoord(0, 0), 0) }).WithDestroyed(4);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) })
                .WithControl(new HexCoord(0, 0), PlayerId.Player0);
            // all weights default to 1
            var s = new GameState(board, GameConfig.Default(), new[] { p0, p1 },
                PlayerId.Player0, 1, 9);

            int expected = 4 /*kills*/ + 10 /*points*/ + unit.PointCost /*army*/ + 1 /*territory*/;
            Assert.That(WinCheck.Score(s, PlayerId.Player0), Is.EqualTo(expected));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter ScoreTests`
Expected: FAIL — `WinCheck.Score` doesn't exist.

- [ ] **Step 3: Implement `Score`**

In `engine/HexWars.Engine/WinCheck.cs`, add (after `Evaluate`):

```csharp
        /// <summary>The configurable Score composite: a weighted sum of kills (destroyed enemy value),
        /// banked points, surviving army value, and controlled-hex count. Weights live in
        /// <see cref="GameConfig"/>. Phase 1 counts controlled hexes flat; per-hex value arrives in Phase 3.
        /// This is the shared "who's winning" value the Score win-check reads (and the RL reward will too).</summary>
        public static int Score(GameState state, PlayerId player)
        {
            var p = state.Player(player);
            var cfg = state.Config;

            int army = 0;
            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) army += u.Stats.PointCost;

            int territory = state.Board.ControlledCount(player);

            return cfg.ScoreKills * p.DestroyedValue
                 + cfg.ScorePoints * p.Points
                 + cfg.ScoreArmy * army
                 + cfg.ScoreTerritory * territory;
        }
```

- [ ] **Step 4: Run test to verify pass**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter ScoreTests`
Expected: PASS (1 test).

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/WinCheck.cs engine/HexWars.Engine.Tests/ScoreTests.cs
git commit -m "feat(engine): WinCheck.Score weighted composite (kills + points + army + territory), the shared score/value function"
```

---

### Task 6: Configurable win-condition resolution

**Files:**
- Modify: `engine/HexWars.Engine/WinCheck.cs`
- Test: `engine/HexWars.Engine.Tests/WinConditionTests.cs` (create)

**Interfaces:**
- Consumes: `GameConfig.WinConditions`/`EconomyWinThreshold` (Task 2); `WinCheck.Score`/`IsEliminated` (existing/Task 5).
- Produces: `IsTerminal`/`Resolve` honoring the active `WinBy` set: Annihilation + Economy are instant; Score decides at the round cap; the round cap is always a terminal backstop.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/WinConditionTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class WinConditionTests
    {
        static GameState With(GameConfig cfg, int p0Points, int p1Points, int round)
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            // both sides have a living unit so nobody is "eliminated"
            var u0 = new Unit(1, PlayerId.Player0, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), new HexCoord(0, 0), 0);
            var pl0 = new PlayerState(PlayerId.Player0, p0Points, null, new[] { u0 });
            var pl1 = new PlayerState(PlayerId.Player1, p1Points, null, new[] { u1 });
            return new GameState(board, cfg, new[] { pl0, pl1 }, PlayerId.Player0, round, 9);
        }

        [Test]
        public void Economy_InstantWin_AtThreshold()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Economy, economyWinThreshold: 50);
            var s = With(cfg, p0Points: 60, p1Points: 0, round: 5);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void Economy_NotTerminal_BelowThreshold()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Economy, economyWinThreshold: 50);
            var s = With(cfg, p0Points: 10, p1Points: 10, round: 5);
            Assert.That(WinCheck.IsTerminal(s), Is.False);
        }

        [Test]
        public void Score_DecidesAtRoundCap_WhenEnabled()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Score, scorePoints: 1,
                scoreKills: 0, scoreArmy: 0, scoreTerritory: 0);
            var s = With(cfg, p0Points: 30, p1Points: 5, round: cfg.RoundCap);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.EqualTo(PlayerId.Player0)); // higher score
        }

        [Test]
        public void Cap_IsDraw_WhenScoreOff()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Annihilation);
            var s = With(cfg, p0Points: 30, p1Points: 5, round: cfg.RoundCap);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.Null); // annihilation-only: cap = draw, as today
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter WinConditionTests`
Expected: FAIL — current `IsTerminal`/`Resolve` ignore `WinConditions`/Economy/Score.

- [ ] **Step 3: Rewrite `IsTerminal` and `Resolve`**

In `engine/HexWars.Engine/WinCheck.cs`, replace the `IsTerminal` and `Resolve` methods with:

```csharp
        /// <summary>Whether the game has ended: an instant win condition fired, or the round cap reached.</summary>
        public static bool IsTerminal(GameState state)
        {
            var win = state.Config.WinConditions;

            if ((win & WinBy.Annihilation) != 0 && state.Round >= 2 &&
                (IsEliminated(state, PlayerId.Player0) || IsEliminated(state, PlayerId.Player1)))
                return true;

            if ((win & WinBy.Economy) != 0 && (EconomyWinner(state) != null))
                return true;

            if (state.Round >= state.Config.RoundCap)
                return true;

            return false;
        }

        /// <summary>The winner, or null on a draw / continuing game. Instant conditions (annihilation,
        /// economy) resolve immediately in priority order; if none fired and the round cap is reached, the
        /// Score condition (if enabled) decides by higher score, else it is a draw.</summary>
        public static PlayerId? Resolve(GameState state)
        {
            var win = state.Config.WinConditions;

            if ((win & WinBy.Annihilation) != 0 && state.Round >= 2)
            {
                bool e0 = IsEliminated(state, PlayerId.Player0);
                bool e1 = IsEliminated(state, PlayerId.Player1);
                if (e0 && !e1) return PlayerId.Player1;
                if (e1 && !e0) return PlayerId.Player0;
            }

            if ((win & WinBy.Economy) != 0)
            {
                var ew = EconomyWinner(state);
                if (ew != null) return ew;
            }

            if (state.Round >= state.Config.RoundCap && (win & WinBy.Score) != 0)
            {
                int s0 = Score(state, PlayerId.Player0);
                int s1 = Score(state, PlayerId.Player1);
                if (s0 > s1) return PlayerId.Player0;
                if (s1 > s0) return PlayerId.Player1;
            }

            return null; // draw / continue
        }

        /// <summary>The player at or past the Economy threshold (higher points if both), or null.</summary>
        private static PlayerId? EconomyWinner(GameState state)
        {
            int t = state.Config.EconomyWinThreshold;
            int p0 = state.Player(PlayerId.Player0).Points;
            int p1 = state.Player(PlayerId.Player1).Points;
            bool a = p0 >= t, b = p1 >= t;
            if (a && (!b || p0 >= p1)) return PlayerId.Player0;
            if (b) return PlayerId.Player1;
            return null;
        }
```

- [ ] **Step 4: Run the full suite**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: PASS — WinConditionTests (4) green, and the existing `WinCheckTests` still green (default `WinConditions = Annihilation` reproduces today's annihilation-only + cap-is-draw behavior exactly).

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/WinCheck.cs engine/HexWars.Engine.Tests/WinConditionTests.cs
git commit -m "feat(engine): configurable multiply-selectable win conditions — annihilation/economy instant, score at the cap; default stays annihilation-only"
```

---

### Task 7: Rebuild, re-sync the Plugins DLL, full green

**Files:**
- Modify (binary): `Assets/HexWars/Plugins/HexWars.Engine.dll`

**Interfaces:** none (verification + Unity hygiene).

- [ ] **Step 1: Full engine test suite**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: PASS — original 145 + the new tests (BoardControl 2, TerritoryConfig 2, CaptureHex 7, KillTally 2, Score 1, WinCondition 4) all green.

- [ ] **Step 2: Build the engine for Unity (netstandard2.1, Release)**

Run: `dotnet build engine/HexWars.Engine -c Release --nologo`
Expected: `Build succeeded. 0 Error(s)`.

- [ ] **Step 3: Sync the Plugins DLL**

Run: `cp -f engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll Assets/HexWars/Plugins/HexWars.Engine.dll`
Expected: file copied (Unity recompiles cleanly on focus; Phase 1 adds no Unity calls so nothing should break).

- [ ] **Step 4: Commit (no-op if DLL is gitignored)**

```bash
git add -A engine docs
git commit -m "chore(engine): territory-control Phase 1 — full suite green; Plugins engine DLL re-synced" || echo "nothing to commit"
```

---

## Self-Review

- **Spec coverage (Phase 1 scope):** per-hex control state → Task 1; `Capture` command + handler + steal/garrison → Task 3; LegalMoves enumeration → Task 3; configurable multiply-selectable win conditions with instant-vs-timed resolution → Tasks 2 + 6; Score composite incl. kills → Tasks 4 + 5; shared score/value function → Task 5 (`WinCheck.Score` public; RewardShaping wiring deferred with RL per spec §8). Generators/economy-tick (Phase 2) and territory per-hex value + UI/replay serialization (Phase 3) are intentionally excluded.
- **Placeholder scan:** no TBD/TODO; every code step has full code and an exact `dotnet test`/`dotnet build` command with expected result.
- **Type consistency:** `WinBy` flags, `GameConfig.{WinConditions,CaptureCost,EconomyWinThreshold,Score*}`, `Board.{Controller,WithControl,ControlledCount}`, `PlayerState.{DestroyedValue,WithDestroyed}`, `CaptureHex(PlayerId,HexCoord)`, `RejectionReason.{NoUnitOnHex,AlreadyControlled}`, and `WinCheck.Score` are defined before they are consumed and named identically across tasks.
- **Backward-compat check:** all new ctor params are optional and last; `GameConfig.Default()` keeps `WinConditions = Annihilation` so `WinCheck` behaves exactly as today — the existing 145 tests are expected to stay green (verified in Tasks 4, 6, 7).

## Notes for Phase 2 / 3 (out of scope here)

- **Phase 2:** generators (build at value-tied cost / steal), the economy tick (income − upkeep) at the turn boundary, capture cost scaling to value, node economy.
- **Phase 3:** per-hex territory value (biome-derived), control overlay + income/score UI, AI (`Capture`/`BuildGenerator`) support, and `ReplayFile` serialization of control + win config so territory games record/replay faithfully.
