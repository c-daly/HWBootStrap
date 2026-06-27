# Territory Control — Phase 2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the territory economy real — generators carry a 0–1 strength, you build them on hexes you control, capturing a generator's hex steals it, and each turn you net income minus upkeep.

**Architecture:** Generators stay in `PlayerState.Generators` but ownership now follows control: building adds a generator to your list on a hex you control, and `CaptureHex` transfers an enemy generator on the captured hex to you — so income (`Σ Output × strength` over your generators) is implicitly control-based. Units may now share a column with a generator (so you can garrison/capture a generator hex). Capture cost on a generator hex scales with its income; the per-turn tick credits income − upkeep.

**Tech Stack:** C# `netstandard2.1` engine (`engine/HexWars.Engine`), NUnit tests (`engine/HexWars.Engine.Tests`, `net8.0`), built with `dotnet`.

## Global Constraints

- Engine `HexWars.Engine` targets **netstandard2.1**, **`System.Random` only**, no `UnityEngine`, deterministic.
- Tests are **NUnit**: `[Test]`, `Assert.That(actual, Is.EqualTo(expected))`. Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo` (focused: append `--filter ClassName`). Windows dotnet from Git Bash: `/c/Program Files/dotnet/dotnet.exe`.
- **Backward compatibility:** the existing **163** tests must stay green. New `GameConfig`/`Generator` constructor parameters are **optional, added last**; the `Generator` strength defaults to **1.0** so existing constructions are unaffected. The economy tick changes `ApplyEndTurn`'s credit from `income` to `income − upkeep`; with the default `GeneratorOutput = 1` and no generators in existing tests, income and upkeep are both 0, so behavior is unchanged.
- **No git attribution** in commit messages.
- **Plugins DLL:** after engine changes, rebuild and copy `engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll` to `Assets/HexWars/Plugins/HexWars.Engine.dll` (final task).
- **Decisions locked for Phase 2** (from the spec + design review): ownership follows control (transfer on capture); units may co-locate with a generator; built generators are strength 1.0; neutral pre-placed nodes and the 0–1 strength *spread* are deferred to Phase 3.

---

### Task 1: Generator strength

**Files:**
- Modify: `engine/HexWars.Engine/Generator.cs`
- Test: `engine/HexWars.Engine.Tests/GeneratorStrengthTests.cs` (create)

**Interfaces:**
- Produces: `Generator(..., double strength = 1.0)` ctor; `Generator.Strength` (double); `Generator.WithOwner(PlayerId) -> Generator` (preserves cell/elevation/hp/strength). `WithDamage` preserves `Strength`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/GeneratorStrengthTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GeneratorStrengthTests
    {
        static readonly HexCoord C = new HexCoord(0, 0);

        [Test]
        public void Strength_DefaultsToOne()
        {
            var g = new Generator(1, PlayerId.Player0, C, 0, 3);
            Assert.That(g.Strength, Is.EqualTo(1.0));
        }

        [Test]
        public void Strength_StoredAndPreservedAcrossDamageAndOwnerChange()
        {
            var g = new Generator(1, PlayerId.Player0, C, 0, 3, 0.5);
            Assert.That(g.Strength, Is.EqualTo(0.5));
            Assert.That(g.WithDamage(1).Strength, Is.EqualTo(0.5));
            var owned = g.WithOwner(PlayerId.Player1);
            Assert.That(owned.Owner, Is.EqualTo(PlayerId.Player1));
            Assert.That(owned.Strength, Is.EqualTo(0.5));
            Assert.That(owned.Cell, Is.EqualTo(C));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter GeneratorStrengthTests`
Expected: FAIL — no `Strength`/`WithOwner`.

- [ ] **Step 3: Implement**

Replace the body of `engine/HexWars.Engine/Generator.cs` (keep the namespace + doc comment) so the struct carries `Strength`:

```csharp
    public readonly struct Generator
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }
        public double Strength { get; }

        public Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp, double strength = 1.0)
            : this(id, owner, cell, elevation, maxHp, maxHp, strength) { }

        private Generator(int id, PlayerId owner, HexCoord cell, int elevation, int maxHp, int currentHp, double strength)
        {
            Id = id;
            Owner = owner;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
            Strength = strength;
        }

        public bool IsAlive => CurrentHp > 0;

        public Generator WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Generator(Id, Owner, Cell, Elevation, hp, hp, Strength);
        }

        /// <summary>The same generator re-owned by <paramref name="owner"/> (used when a hex is captured).</summary>
        public Generator WithOwner(PlayerId owner) =>
            new Generator(Id, owner, Cell, Elevation, CurrentHp, CurrentHp, Strength);
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter GeneratorStrengthTests`
Expected: PASS (2). Then full suite — `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo` — expect 165 (163 + 2); existing generator constructions compile because `strength` defaults to 1.0.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Generator.cs engine/HexWars.Engine.Tests/GeneratorStrengthTests.cs
git commit -m "feat(engine): Generator.Strength (0-1, default 1.0) + WithOwner; preserved across damage/owner-change"
```

---

### Task 2: Phase-2 economy config knobs

**Files:**
- Modify: `engine/HexWars.Engine/GameConfig.cs`
- Test: `engine/HexWars.Engine.Tests/Phase2ConfigTests.cs` (create)

**Interfaces:**
- Produces: `GameConfig.UpkeepFactor` (double, default 0.25), `GameConfig.CaptureFactor` (double, default 4.0), `GameConfig.BuildFactor` (double, default 4.0); `Default(... , upkeepFactor, captureFactor, buildFactor)` passthrough.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/Phase2ConfigTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class Phase2ConfigTests
    {
        [Test]
        public void Default_HasPhase2Factors()
        {
            var c = GameConfig.Default();
            Assert.That(c.UpkeepFactor, Is.EqualTo(0.25));
            Assert.That(c.CaptureFactor, Is.EqualTo(4.0));
            Assert.That(c.BuildFactor, Is.EqualTo(4.0));
        }

        [Test]
        public void Default_PassesThroughFactors()
        {
            var c = GameConfig.Default(upkeepFactor: 0.5, captureFactor: 6.0, buildFactor: 3.0);
            Assert.That(c.UpkeepFactor, Is.EqualTo(0.5));
            Assert.That(c.CaptureFactor, Is.EqualTo(6.0));
            Assert.That(c.BuildFactor, Is.EqualTo(3.0));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter Phase2ConfigTests`
Expected: FAIL — properties don't exist.

- [ ] **Step 3: Add the fields**

In `engine/HexWars.Engine/GameConfig.cs`, add properties after `ScoreTerritory`:

```csharp
        /// <summary>Per-turn upkeep as a fraction of a player's generator income.</summary>
        public double UpkeepFactor { get; }
        /// <summary>Capture cost on a generator hex = max(CaptureCost, round(CaptureFactor × that generator's income)).</summary>
        public double CaptureFactor { get; }
        /// <summary>Build cost of a generator = round(BuildFactor × GeneratorOutput × strength).</summary>
        public double BuildFactor { get; }
```

Add constructor parameters (**after** `int scoreTerritory = 1`):

```csharp
            int scoreTerritory = 1,
            double upkeepFactor = 0.25,
            double captureFactor = 4.0,
            double buildFactor = 4.0)
```

Add assignments (after `ScoreTerritory = scoreTerritory;`):

```csharp
            UpkeepFactor = upkeepFactor;
            CaptureFactor = captureFactor;
            BuildFactor = buildFactor;
```

Update `Default(...)` — add the three params at the end of its signature and forward them. Change the signature line:

```csharp
        public static GameConfig Default(bool biomesEnabled = true, ITurnPolicy? turnPolicy = null,
            WinBy winConditions = WinBy.Annihilation, int captureCost = 3, int economyWinThreshold = 200,
            int scoreKills = 1, int scorePoints = 1, int scoreArmy = 1, int scoreTerritory = 1,
            double upkeepFactor = 0.25, double captureFactor = 4.0, double buildFactor = 4.0) =>
```

And extend the trailing constructor call's named arguments (after `scoreTerritory: scoreTerritory`):

```csharp
           scoreKills: scoreKills, scorePoints: scorePoints, scoreArmy: scoreArmy, scoreTerritory: scoreTerritory,
           upkeepFactor: upkeepFactor, captureFactor: captureFactor, buildFactor: buildFactor);
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter Phase2ConfigTests` → PASS (2). Full suite → 167.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/GameConfig.cs engine/HexWars.Engine.Tests/Phase2ConfigTests.cs
git commit -m "feat(engine): Phase-2 economy config — UpkeepFactor/CaptureFactor/BuildFactor (defaults 0.25/4.0/4.0)"
```

---

### Task 3: Strength-weighted income + upkeep

**Files:**
- Modify: `engine/HexWars.Engine/Economy.cs`
- Test: `engine/HexWars.Engine.Tests/EconomyIncomeTests.cs` (create)

**Interfaces:**
- Consumes: `Generator.Strength` (Task 1), `GameConfig.GeneratorOutput`/`UpkeepFactor` (Task 2).
- Produces: `Economy.Income(GameState, PlayerId) -> int` = `round(Σ_aliveGenerators GeneratorOutput × strength)`; `Economy.Upkeep(GameState, PlayerId) -> int` = `round(Income × UpkeepFactor)`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/EconomyIncomeTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class EconomyIncomeTests
    {
        // One player with two generators (strength 1.0 and 0.5), GeneratorOutput=10, UpkeepFactor=0.25.
        static GameState State()
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            var gens = new[]
            {
                new Generator(1, PlayerId.Player0, new HexCoord(0, 0), 0, 3, 1.0),
                new Generator(2, PlayerId.Player0, new HexCoord(1, 0), 0, 3, 0.5),
            };
            var p0 = new PlayerState(PlayerId.Player0, 0, null, null, gens);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = new GameConfig(new System.Collections.Generic.Dictionary<TerrainType, TerrainDef>(),
                generatorOutput: 10, upkeepFactor: 0.25);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 9);
        }

        [Test]
        public void Income_SumsOutputTimesStrength()
        {
            // 10*1.0 + 10*0.5 = 15
            Assert.That(Economy.Income(State(), PlayerId.Player0), Is.EqualTo(15));
        }

        [Test]
        public void Upkeep_IsFractionOfIncome()
        {
            // round(15 * 0.25) = 4 (AwayFromZero: 3.75 -> 4)
            Assert.That(Economy.Upkeep(State(), PlayerId.Player0), Is.EqualTo(4));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter EconomyIncomeTests`
Expected: FAIL — income ignores strength; `Upkeep` doesn't exist.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/Economy.cs`, replace `Income` and add `Upkeep`:

```csharp
        /// <summary>Income this turn = round(Σ over the player's living generators of GeneratorOutput × strength).
        /// Ownership follows control (captures transfer generators), so this is control-based income.</summary>
        public static int Income(GameState state, PlayerId player)
        {
            double income = 0;
            foreach (var g in state.Player(player).Generators)
                if (g.IsAlive) income += state.Config.GeneratorOutput * g.Strength;
            return (int)System.Math.Round(income, System.MidpointRounding.AwayFromZero);
        }

        /// <summary>Per-turn upkeep = round(income × UpkeepFactor).</summary>
        public static int Upkeep(GameState state, PlayerId player)
            => (int)System.Math.Round(Income(state, player) * state.Config.UpkeepFactor,
                                      System.MidpointRounding.AwayFromZero);
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter EconomyIncomeTests` → PASS (2). Full suite → 169 (existing tests have no generators, so income/upkeep stay 0).

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Economy.cs engine/HexWars.Engine.Tests/EconomyIncomeTests.cs
git commit -m "feat(engine): strength-weighted generator income + Economy.Upkeep (fraction of income)"
```

---

### Task 4: Units may garrison generator hexes

**Files:**
- Modify: `engine/HexWars.Engine/MovementService.cs`
- Test: `engine/HexWars.Engine.Tests/GarrisonMovementTests.cs` (create)

**Interfaces:**
- Produces: a unit can move onto a hex that holds a (friendly or enemy) generator; only living **units** block movement now.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/GarrisonMovementTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GarrisonMovementTests
    {
        [Test]
        public void Unit_CanReachAGeneratorHex()
        {
            var a = new HexCoord(0, 0);
            var b = new HexCoord(1, 0);
            var board = new Board(new[]
            {
                new Tile(a, 0, TerrainType.Plains),
                new Tile(b, 0, TerrainType.Plains),
            });
            var mover = new Unit(1, PlayerId.Player0, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), a, 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, null, new[] { mover });
            // enemy generator sits on b
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(2, PlayerId.Player1, b, 0, 3) });
            var s = new GameState(board, GameConfig.Default(biomesEnabled: false), new[] { p0, p1 },
                PlayerId.Player0, 1, 9);

            var reachable = MovementService.ReachableTiles(s, mover);
            Assert.That(reachable, Does.Contain(b), "a unit may garrison a generator hex");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter GarrisonMovementTests`
Expected: FAIL — `OccupiedCells` counts generators, so `b` is excluded.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/MovementService.cs`, change `OccupiedCells` to count **only units** (remove the generator loop):

```csharp
        private static HashSet<HexCoord> OccupiedCells(GameState state, Unit mover)
        {
            var set = new HashSet<HexCoord>();
            foreach (var player in state.Players)
                foreach (var u in player.UnitsOnBoard)
                    if (u.IsAlive) set.Add(u.Cell);
            set.Remove(mover.Cell);
            return set;
        }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter GarrisonMovementTests` → PASS (1). Full suite → 170. (Existing movement tests don't place generators, so they're unaffected.)

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/MovementService.cs engine/HexWars.Engine.Tests/GarrisonMovementTests.cs
git commit -m "feat(engine): units may share a column with a generator (garrison) — only units block movement now"
```

---

### Task 5: `BuildGenerator` command + handler + legal moves

**Files:**
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/RejectionReason.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/LegalMoves.cs`
- Test: `engine/HexWars.Engine.Tests/BuildGeneratorTests.cs` (create)

**Interfaces:**
- Consumes: `Board.Controller` (Phase 1), `GameConfig.BuildFactor`/`GeneratorOutput` (Task 2), `Generator(..., strength)` (Task 1), `PlayerState.DestroyedValue` (Phase 1), `WithPlayer` (existing GameEngine helper).
- Produces: `record BuildGenerator(PlayerId Issuer, HexCoord Cell) : Command`; `RejectionReason.HexNotControlled`; a `BuildGenerator` entry in `LegalMoves.For`.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/BuildGeneratorTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BuildGeneratorTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 controls A and has a unit there, with `points`. GeneratorOutput=10, BuildFactor=4 -> build cost 40.
        static GameState State(int points, bool control = true, bool generatorAlready = false)
        {
            Board board = new Board(new[]
            {
                new Tile(A, 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            });
            if (control) board = board.WithControl(A, PlayerId.Player0);
            var gens = generatorAlready
                ? new[] { new Generator(5, PlayerId.Player0, A, 0, 3) }
                : null;
            var p0 = new PlayerState(PlayerId.Player0, points, null, new[] { new Unit(1, PlayerId.Player0, S, A, 0) }, gens);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, buildFactor: 4.0);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void Build_PlacesGenerator_AndDeductsCost()
        {
            var r = GameEngine.Apply(State(50), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            var gens = r.NewState.Player(PlayerId.Player0).Generators;
            Assert.That(gens.Count, Is.EqualTo(1));
            Assert.That(gens[0].Cell, Is.EqualTo(A));
            Assert.That(gens[0].Strength, Is.EqualTo(1.0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(50 - 40)); // BuildFactor*Output
        }

        [Test]
        public void Build_RejectsWhenHexNotControlled()
        {
            var r = GameEngine.Apply(State(50, control: false), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.HexNotControlled));
        }

        [Test]
        public void Build_RejectsWhenGeneratorAlreadyThere()
        {
            var r = GameEngine.Apply(State(50, generatorAlready: true), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TileOccupied));
        }

        [Test]
        public void Build_RejectsWhenBroke()
        {
            var r = GameEngine.Apply(State(10), new BuildGenerator(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void LegalMoves_IncludeBuildOnControlledGarrisonedHex()
        {
            var moves = LegalMoves.For(State(50));
            Assert.That(moves, Has.Some.Matches<Command>(m => m is BuildGenerator b && b.Cell == A));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter BuildGeneratorTests`
Expected: FAIL — `BuildGenerator`, `HexNotControlled`, the handler, and the legal-move don't exist.

- [ ] **Step 3: Add the command**

In `engine/HexWars.Engine/Command.cs`, add (after `CaptureHex`):

```csharp
    /// <summary>Build a generator (full strength) on a hex the issuer controls, paying the build cost.</summary>
    public sealed record BuildGenerator(PlayerId Issuer, HexCoord Cell) : Command(Issuer);
```

- [ ] **Step 4: Add the rejection reason**

In `engine/HexWars.Engine/RejectionReason.cs`, add (before the closing brace of the enum):

```csharp
        HexNotControlled,
```

- [ ] **Step 5: Add the handler + dispatch + helpers**

In `engine/HexWars.Engine/GameEngine.cs`, add a `Dispatch` case (after the `CaptureHex` case):

```csharp
                case BuildGenerator c: return ApplyBuildGenerator(state, c);
```

Add the handler + helpers (place after `ApplyCaptureHex`):

```csharp
        private static Result ApplyBuildGenerator(GameState state, BuildGenerator c)
        {
            if (!state.Board.Contains(c.Cell)) return Result.Reject(state, RejectionReason.TileNotFound);
            if (state.Board.Controller(c.Cell) != c.Issuer)
                return Result.Reject(state, RejectionReason.HexNotControlled);
            if (HasGeneratorAt(state, c.Cell)) return Result.Reject(state, RejectionReason.TileOccupied);

            var player = state.Player(c.Issuer);
            int cost = BuildCost(state.Config);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var tile = state.Board.TileAt(c.Cell);
            var gen = new Generator(state.NextEntityId, c.Issuer, c.Cell, tile.Elevation, state.Config.GeneratorHealth, 1.0);
            var generators = new List<Generator>(player.Generators) { gen };
            var updated = new PlayerState(player.Id, player.Points - cost, player.Barracks,
                                          player.UnitsOnBoard, generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated, state.NextEntityId + 1));
        }

        /// <summary>Build cost of a full-strength generator = round(BuildFactor × GeneratorOutput).</summary>
        private static int BuildCost(GameConfig cfg) =>
            (int)System.Math.Round(cfg.BuildFactor * cfg.GeneratorOutput, System.MidpointRounding.AwayFromZero);

        private static bool HasGeneratorAt(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == coord) return true;
            return false;
        }
```

- [ ] **Step 6: Enumerate in LegalMoves**

In `engine/HexWars.Engine/LegalMoves.cs`, inside the `foreach (var unit in player.UnitsOnBoard)` loop, right after the existing `CaptureHex` enumeration line, add a build option at the unit's hex when the player controls it, no generator is there, and it's affordable:

```csharp
                if (board.Controller(unit.Cell) == me
                    && !HasGeneratorAt(state, unit.Cell)
                    && player.Points >= BuildCostFor(state.Config))
                    moves.Add(new BuildGenerator(me, unit.Cell));
```

Add two private helpers to `LegalMoves` (after the existing `IsOccupied` helper):

```csharp
        private static bool HasGeneratorAt(GameState state, HexCoord coord)
        {
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == coord) return true;
            return false;
        }

        private static int BuildCostFor(GameConfig cfg) =>
            (int)System.Math.Round(cfg.BuildFactor * cfg.GeneratorOutput, System.MidpointRounding.AwayFromZero);
```

- [ ] **Step 7: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter BuildGeneratorTests` → PASS (5). Full suite → 175.

- [ ] **Step 8: Commit**

```bash
git add engine/HexWars.Engine/Command.cs engine/HexWars.Engine/RejectionReason.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine.Tests/BuildGeneratorTests.cs
git commit -m "feat(engine): BuildGenerator command + handler + legal-move (build full-strength generator on a controlled hex, cost = BuildFactor x Output)"
```

---

### Task 6: Capture scales cost + steals the generator

**Files:**
- Modify: `engine/HexWars.Engine/GameEngine.cs` (`ApplyCaptureHex` + two helpers)
- Test: `engine/HexWars.Engine.Tests/CaptureStealTests.cs` (create)

**Interfaces:**
- Consumes: `GameConfig.CaptureFactor`/`CaptureCost`/`GeneratorOutput` (Tasks 1/2), `Generator.WithOwner` (Task 1), `Board.WithControl` (Phase 1).
- Produces: `ApplyCaptureHex` now charges `max(CaptureCost, round(CaptureFactor × the hex generator's income))` and, on success, transfers an enemy generator on the captured hex to the capturer.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/CaptureStealTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class CaptureStealTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has a unit on A; P1 controls A and has a strength-1 generator there. Output=10, CaptureFactor=4, CaptureCost=3.
        static GameState State(int p0Points)
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) }).WithControl(A, PlayerId.Player1);
            var p0 = new PlayerState(PlayerId.Player0, p0Points, null, new[] { new Unit(1, PlayerId.Player0, S, A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(7, PlayerId.Player1, A, 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, captureFactor: 4.0, captureCost: 3);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void Capture_OfGeneratorHex_ScalesCost_AndStealsGenerator()
        {
            var r = GameEngine.Apply(State(100), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            // cost = max(3, round(4 * (10*1.0))) = 40
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(100 - 40));
            // generator moved from P1 to P0, re-owned
            Assert.That(r.NewState.Player(PlayerId.Player1).Generators.Count(g => g.IsAlive), Is.EqualTo(0));
            var stolen = r.NewState.Player(PlayerId.Player0).Generators.Single();
            Assert.That(stolen.Id, Is.EqualTo(7));
            Assert.That(stolen.Owner, Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void Capture_OfGeneratorHex_RejectsWhenCannotAffordScaledCost()
        {
            // can afford the flat CaptureCost (3) but not the scaled 40
            var r = GameEngine.Apply(State(10), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter CaptureStealTests`
Expected: FAIL — current `ApplyCaptureHex` charges the flat cost and doesn't transfer the generator.

- [ ] **Step 3: Rewrite `ApplyCaptureHex` + add helpers**

In `engine/HexWars.Engine/GameEngine.cs`, replace the `ApplyCaptureHex` method with:

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

            int cost = CaptureCostFor(state, c.Cell);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var players = state.Players.ToArray();

            // steal: transfer an enemy generator on the captured hex to the capturer
            var enemy = state.Opponent(c.Issuer);
            int gi = IndexOfGeneratorAt(enemy, c.Cell);
            if (gi >= 0)
            {
                var stolen = enemy.Generators[gi].WithOwner(c.Issuer);
                var enemyGens = new List<Generator>(enemy.Generators);
                enemyGens.RemoveAt(gi);
                players[(int)enemy.Id] = new PlayerState(enemy.Id, enemy.Points, enemy.Barracks,
                                                         enemy.UnitsOnBoard, enemyGens, enemy.DestroyedValue);
                var myGens = new List<Generator>(player.Generators) { stolen };
                player = new PlayerState(player.Id, player.Points, player.Barracks,
                                         player.UnitsOnBoard, myGens, player.DestroyedValue);
            }

            players[(int)c.Issuer] = player.WithPoints(player.Points - cost);
            var newBoard = state.Board.WithControl(c.Cell, c.Issuer);

            return Result.Ok(new GameState(newBoard, state.Config, players, state.ActivePlayer,
                state.Round, state.NextEntityId, state.IsGameOver, state.Winner,
                state.MovedUnitIds, state.AttackedUnitIds));
        }

        /// <summary>Capture cost for a hex: flat CaptureCost, or — if a generator sits here — scaled by its
        /// income: max(CaptureCost, round(CaptureFactor × round(GeneratorOutput × strength))).</summary>
        private static int CaptureCostFor(GameState state, HexCoord cell)
        {
            int flat = state.Config.CaptureCost;
            foreach (var p in state.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == cell)
                    {
                        int income = (int)System.Math.Round(state.Config.GeneratorOutput * g.Strength,
                                                            System.MidpointRounding.AwayFromZero);
                        int scaled = (int)System.Math.Round(state.Config.CaptureFactor * income,
                                                            System.MidpointRounding.AwayFromZero);
                        return System.Math.Max(flat, scaled);
                    }
            return flat;
        }

        private static int IndexOfGeneratorAt(PlayerState p, HexCoord cell)
        {
            for (int i = 0; i < p.Generators.Count; i++)
                if (p.Generators[i].IsAlive && p.Generators[i].Cell == cell) return i;
            return -1;
        }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter CaptureStealTests` → PASS (2). Then full suite → expect 177. The Phase-1 `CaptureHexTests` (plain hexes, no generator) still pass: with no generator on the cell, `CaptureCostFor` returns the flat `CaptureCost` and no transfer happens — identical to before.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine.Tests/CaptureStealTests.cs
git commit -m "feat(engine): capturing a generator hex scales cost to its income and steals the generator (transfers ownership)"
```

---

### Task 7: Economy tick = income − upkeep

**Files:**
- Modify: `engine/HexWars.Engine/GameEngine.cs` (`ApplyEndTurn`)
- Test: `engine/HexWars.Engine.Tests/EconomyTickTests.cs` (create)

**Interfaces:**
- Consumes: `Economy.Income`/`Economy.Upkeep` (Task 3).
- Produces: on `EndTurn`, the player whose turn begins is credited `income − upkeep` (clamped so points never go below 0).

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/EconomyTickTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class EconomyTickTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);

        // P1 begins next turn with a strength-1 generator. Output=10, UpkeepFactor=0.25 -> net +8 (10 - round(2.5)=3 -> 7).
        static GameState State(int p1StartPoints)
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) });
            var p0 = new PlayerState(PlayerId.Player0, 0,
                null, new[] { new Unit(1, PlayerId.Player0, new UnitStats(5,3,2,3,2,1,1,2,1), A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, p1StartPoints, null, null,
                new[] { new Generator(2, PlayerId.Player1, new HexCoord(1, 0), 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, upkeepFactor: 0.25);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void EndTurn_CreditsIncomeMinusUpkeep()
        {
            var r = GameEngine.Apply(State(5), new EndTurn(PlayerId.Player0));
            Assert.That(r.Success, Is.True);
            // income 10, upkeep round(10*0.25)=3 (2.5 -> 3 AwayFromZero), net +7 -> 5 + 7 = 12
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.EqualTo(12));
        }

        [Test]
        public void EndTurn_PointsNeverGoNegative()
        {
            // huge upkeep via UpkeepFactor would exceed points; clamp at 0
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) });
            var p0 = new PlayerState(PlayerId.Player0, 0, null, new[] { new Unit(1, PlayerId.Player0, new UnitStats(5,3,2,3,2,1,1,2,1), A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(2, PlayerId.Player1, new HexCoord(1, 0), 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, upkeepFactor: 5.0); // upkeep 50 >> income 10
            var s = new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
            var r = GameEngine.Apply(s, new EndTurn(PlayerId.Player0));
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.EqualTo(0));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter EconomyTickTests`
Expected: FAIL — `ApplyEndTurn` credits gross income (no upkeep, no clamp).

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/GameEngine.cs` `ApplyEndTurn`, replace the income credit. Change:

```csharp
            int income = Economy.Income(state, next);
            var players = state.Players.ToArray();
            var np = players[(int)next];
            players[(int)next] = np.WithPoints(np.Points + income);
```

to:

```csharp
            int net = Economy.Income(state, next) - Economy.Upkeep(state, next);
            var players = state.Players.ToArray();
            var np = players[(int)next];
            players[(int)next] = np.WithPoints(System.Math.Max(0, np.Points + net));
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter EconomyTickTests` → PASS (2). Full suite → 179. Existing tests have no generators (income = upkeep = 0, net = 0) and non-negative points, so the clamp and the `−upkeep` are no-ops for them.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine.Tests/EconomyTickTests.cs
git commit -m "feat(engine): per-turn economy tick credits income minus upkeep (clamped at 0)"
```

---

### Task 8: Rebuild, re-sync the Plugins DLL, full green

**Files:**
- Modify (binary): `Assets/HexWars/Plugins/HexWars.Engine.dll`

**Interfaces:** none (verification + Unity hygiene).

- [ ] **Step 1: Full engine test suite**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: PASS — the 163 Phase-1 baseline + the new Phase-2 tests (Generator 2, Phase2Config 2, EconomyIncome 2, Garrison 1, BuildGenerator 5, CaptureSteal 2, EconomyTick 2 = 16) → 179 green.

- [ ] **Step 2: Build the engine for Unity**

Run: `dotnet build engine/HexWars.Engine -c Release --nologo`
Expected: `Build succeeded. 0 Error(s)`.

- [ ] **Step 3: Sync the Plugins DLL**

Run: `cp -f engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll Assets/HexWars/Plugins/HexWars.Engine.dll`
Expected: file copied.

- [ ] **Step 4: Commit (no-op if the DLL is gitignored)**

```bash
git add -A engine docs
git commit -m "chore(engine): territory-control Phase 2 — full suite green; Plugins engine DLL re-synced" || echo "nothing to commit"
```

---

## Self-Review

- **Spec coverage (Phase 2 scope):** generator strength → Task 1; strength-weighted, control-based income + upkeep → Task 3; `BuildGenerator` (controlled hex, cost ∝ value) → Task 5; capture cost scaling + steal/transfer → Task 6; economy tick (income − upkeep) → Task 7; the enabling "units garrison generators" rule (needed for capture/steal) → Task 4; config knobs → Task 2. Deferred per plan: neutral pre-placed nodes, the 0–1 strength spread (built = 1.0), territory per-hex value, UI, AI, replay serialization.
- **Placeholder scan:** none — every code step has full code and an exact `dotnet test`/`dotnet build` command with expected counts.
- **Type consistency:** `Generator(...strength=1.0)`/`Strength`/`WithOwner`; `GameConfig.{UpkeepFactor,CaptureFactor,BuildFactor}`; `Economy.Income`(strength-weighted)/`Economy.Upkeep`; `BuildGenerator(PlayerId,HexCoord)`; `RejectionReason.HexNotControlled`; `ApplyBuildGenerator`/`BuildCost`/`HasGeneratorAt`/`CaptureCostFor`/`IndexOfGeneratorAt` — defined before use and named identically across tasks. `HasGeneratorAt` exists in both `GameEngine` (Task 5) and `LegalMoves` (Task 5) as separate private helpers (each file needs its own; intentional, not a name clash).
- **Backward-compat:** new params optional/last with behavior-preserving defaults; no existing test places a generator, so income/upkeep/transfer/garrison changes are inert for them. The `Generator` strength default (1.0) keeps every existing construction compiling and behaving as before.

## Notes for Phase 3 (out of scope here)

Per-hex territory value (biome-derived) + the 0–1 strength spread on neutral pre-placed nodes; board-gen seeding of neutral generators; control/income/score UI overlays; AI (`Capture`/`BuildGenerator`) support; `ReplayFile` serialization of control + generators' strength + the win/economy config so territory games record and replay faithfully; and folding `WinCheck.Score`'s value function together with `RewardShaping` when RL adopts this mode.
