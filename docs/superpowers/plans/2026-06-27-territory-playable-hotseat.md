# Territory Control — Playable Hotseat (Phase 3, Slice 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.
> **Execution note:** Part A (engine, Tasks 1–4) is standard TDD and subagent-executable. Part B (presentation, Tasks 6–9) is Unity UI that cannot be unit-tested — its "test" step is a compile check (`mcp__coplay-mcp__check_compile_errors`) plus the manual play-through in Task 10. Build Part B directly in the session against a running Unity, not via headless subagents.

**Goal:** Make the already-built territory engine playable in a two-human hotseat game — control overlay, click-to-claim/build, an economy HUD, and the claim-tempo rule that makes you vulnerable as you expand.

**Architecture:** Add two gated `GameConfig` flags (`TerritoryMode`, `ClaimEndsTurn`) plus a batch control-seed helper to the engine; gate the new rules (deploy-on-controlled, claim-is-turn-exclusive) behind them so the 184 existing tests are untouched. Then layer presentation: tint controlled hexes by owner, click-the-hex to claim/build, an economy readout, and a `GameBootstrap` toggle that builds the territory config and seeds each player's home zone.

**Tech Stack:** C# `netstandard2.1` engine (`engine/HexWars.Engine`) + NUnit tests (`engine/HexWars.Engine.Tests`, net8.0); Unity 6 presentation (`Assets/HexWars/Presentation`) compiled against `Assets/HexWars/Plugins/HexWars.Engine.dll`.

## Global Constraints

- Engine targets **netstandard2.1**, **`System.Random` only**, no `UnityEngine`, deterministic.
- All new behavior is **gated** by `GameConfig.TerritoryMode` (default false) so the base annihilation game and the **184 existing tests stay green**. `ClaimEndsTurn` defaults **true**.
- New `GameConfig` ctor/`Default` params are **optional and added last**.
- NUnit: `Assert.That(actual, Is.EqualTo(expected))`. Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo` (focused: append `--filter ClassName`). Windows dotnet from Git Bash: `/c/Program Files/dotnet/dotnet.exe`.
- **No git attribution** in commit messages.
- After engine changes, rebuild Release and copy `engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll` → `Assets/HexWars/Plugins/HexWars.Engine.dll`, or Unity drops to Safe Mode.
- Reused engine APIs already on `main`: `Board.Controller/WithControl/ControlledCount`, `CaptureHex`+`ApplyCaptureHex`, `BuildGenerator`, `Economy.Income/Upkeep`, `GameConfig` cost dials, `GameState.MovedUnitIds/AttackedUnitIds`, `WinCheck.Score`, `RejectionReason.HexNotControlled` (added in Phase 2).

---

## Part A — Engine (TDD)

### Task 1: `TerritoryMode` + `ClaimEndsTurn` config flags (and `startingPoints` passthrough)

**Files:**
- Modify: `engine/HexWars.Engine/GameConfig.cs`
- Test: `engine/HexWars.Engine.Tests/Phase3ConfigTests.cs` (create)

**Interfaces:**
- Produces: `GameConfig.TerritoryMode` (bool, default false), `GameConfig.ClaimEndsTurn` (bool, default true); `Default(... , startingPoints, territoryMode, claimEndsTurn)` passthrough (startingPoints default 12).

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/Phase3ConfigTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class Phase3ConfigTests
    {
        [Test]
        public void Default_TerritoryDefaultsOff_ClaimEndsTurnOn()
        {
            var c = GameConfig.Default();
            Assert.That(c.TerritoryMode, Is.False);
            Assert.That(c.ClaimEndsTurn, Is.True);
        }

        [Test]
        public void Default_PassesThroughTerritoryFlagsAndStartingPoints()
        {
            var c = GameConfig.Default(startingPoints: 40, territoryMode: true, claimEndsTurn: false);
            Assert.That(c.StartingPoints, Is.EqualTo(40));
            Assert.That(c.TerritoryMode, Is.True);
            Assert.That(c.ClaimEndsTurn, Is.False);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter Phase3ConfigTests`
Expected: FAIL — `TerritoryMode`/`ClaimEndsTurn` don't exist and `Default` has no `startingPoints`/`territoryMode`/`claimEndsTurn` params.

- [ ] **Step 3: Add the properties**

In `engine/HexWars.Engine/GameConfig.cs`, after the `BuildFactor` property:

```csharp
        /// <summary>When true, the territory rules apply: deploy/build require control, and (with
        /// ClaimEndsTurn) claiming is a turn-exclusive opening action. Default false = base game.</summary>
        public bool TerritoryMode { get; }
        /// <summary>When true (and TerritoryMode), a claim must be the turn's first army action and
        /// immediately ends the turn — the vulnerable-as-you-expand tempo. Default true.</summary>
        public bool ClaimEndsTurn { get; }
```

- [ ] **Step 4: Add constructor params + assignments**

Change the constructor's final params from `double buildFactor = 4.0)` to:

```csharp
            double buildFactor = 4.0,
            bool territoryMode = false,
            bool claimEndsTurn = true)
```

After `BuildFactor = buildFactor;` add:

```csharp
            TerritoryMode = territoryMode;
            ClaimEndsTurn = claimEndsTurn;
```

- [ ] **Step 5: Thread through `Default`**

Change `Default`'s signature final params from `int generatorOutput = 1)` to:

```csharp
            int generatorOutput = 1,
            int startingPoints = 12,
            bool territoryMode = false,
            bool claimEndsTurn = true)
```

In the `new GameConfig(...)` call inside `Default`, add `startingPoints: startingPoints,` near the front of the named args (it maps to the ctor's `startingPoints` param) and append to the trailing args:

```csharp
           generatorOutput: generatorOutput, startingPoints: startingPoints,
           territoryMode: territoryMode, claimEndsTurn: claimEndsTurn);
```

(The ctor already has `int startingPoints = 12` as its 2nd param; `Default` simply forwards it now.)

- [ ] **Step 6: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter Phase3ConfigTests` → PASS (2). Full suite → 186 (184 + 2).

- [ ] **Step 7: Commit**

```bash
git add engine/HexWars.Engine/GameConfig.cs engine/HexWars.Engine.Tests/Phase3ConfigTests.cs
git commit -m "feat(engine): GameConfig.TerritoryMode + ClaimEndsTurn flags (gated, defaults off/on) + startingPoints passthrough in Default"
```

---

### Task 2: `Board.WithControl` batch seed helper

**Files:**
- Modify: `engine/HexWars.Engine/Board.cs`
- Test: `engine/HexWars.Engine.Tests/BoardSeedControlTests.cs` (create)

**Interfaces:**
- Produces: `Board.WithControl(IEnumerable<HexCoord> coords, PlayerId owner) -> Board` (immutable; sets all the given hexes to `owner`).

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/BoardSeedControlTests.cs`:

```csharp
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardSeedControlTests
    {
        [Test]
        public void WithControl_Batch_SeedsAllHexes_LeavesOriginalUnchanged()
        {
            var a = new HexCoord(0, 0);
            var b = new HexCoord(1, 0);
            var board = new Board(new[] { new Tile(a, 0, TerrainType.Plains), new Tile(b, 0, TerrainType.Plains) });

            var seeded = board.WithControl(new List<HexCoord> { a, b }, PlayerId.Player1);

            Assert.That(seeded.Controller(a), Is.EqualTo(PlayerId.Player1));
            Assert.That(seeded.Controller(b), Is.EqualTo(PlayerId.Player1));
            Assert.That(seeded.ControlledCount(PlayerId.Player1), Is.EqualTo(2));
            Assert.That(board.Controller(a), Is.Null, "original board is unchanged");
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter BoardSeedControlTests`
Expected: FAIL — no batch `WithControl` overload.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/Board.cs`, after the existing single-hex `WithControl`:

```csharp
        /// <summary>A new board with every hex in <paramref name="coords"/> controlled by
        /// <paramref name="owner"/>. Immutable — this board is unchanged. (Used to seed home zones.)</summary>
        public Board WithControl(System.Collections.Generic.IEnumerable<HexCoord> coords, PlayerId owner)
        {
            var control = new Dictionary<HexCoord, PlayerId>(_control);
            foreach (var c in coords) control[c] = owner;
            return new Board(_tiles.Values, _zone0, _zone1, control);
        }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter BoardSeedControlTests` → PASS (1). Full suite → 187.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Board.cs engine/HexWars.Engine.Tests/BoardSeedControlTests.cs
git commit -m "feat(engine): Board.WithControl(IEnumerable<HexCoord>, owner) batch seed helper (for home-zone control)"
```

---

### Task 3: `DeployUnit` requires control in `TerritoryMode`

**Files:**
- Modify: `engine/HexWars.Engine/GameEngine.cs` (`ApplyDeployUnit`)
- Test: `engine/HexWars.Engine.Tests/TerritoryDeployTests.cs` (create)

**Interfaces:**
- Consumes: `GameConfig.TerritoryMode` (Task 1), `Board.Controller`/`WithControl` (Task 2 + Phase 1), `RejectionReason.HexNotControlled` (Phase 2).
- Produces: in `TerritoryMode`, `DeployUnit` rejects (`HexNotControlled`) unless the target hex is controlled by the issuer; off-mode keeps the deployment-zone rule.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/TerritoryDeployTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TerritoryDeployTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly HexCoord B = new HexCoord(1, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has a barracks template + 100 pts. A is controlled by P0; B is neutral. No deployment zones set.
        static GameState State()
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains), new Tile(B, 0, TerrainType.Plains) })
                .WithControl(A, PlayerId.Player0);
            var p0 = new PlayerState(PlayerId.Player0, 100, new[] { S });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 9);
        }

        [Test]
        public void Deploy_OnControlledHex_Succeeds()
        {
            var r = GameEngine.Apply(State(), new DeployUnit(PlayerId.Player0, 0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Player(PlayerId.Player0).UnitsOnBoard.Count, Is.EqualTo(1));
        }

        [Test]
        public void Deploy_OnUncontrolledHex_Rejected()
        {
            var r = GameEngine.Apply(State(), new DeployUnit(PlayerId.Player0, 0, B));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.HexNotControlled));
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter TerritoryDeployTests`
Expected: FAIL — `DeployUnit` still checks the deployment zone (B isn't in any zone, A isn't either), so both reject with `OutsideDeploymentZone`.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/GameEngine.cs` `ApplyDeployUnit`, replace the single deployment-zone check line:

```csharp
            if (!board.IsInDeploymentZone(c.Issuer, c.Cell)) return Result.Reject(state, RejectionReason.OutsideDeploymentZone);
```

with a mode-conditional:

```csharp
            if (state.Config.TerritoryMode)
            {
                if (board.Controller(c.Cell) != c.Issuer) return Result.Reject(state, RejectionReason.HexNotControlled);
            }
            else
            {
                if (!board.IsInDeploymentZone(c.Issuer, c.Cell)) return Result.Reject(state, RejectionReason.OutsideDeploymentZone);
            }
```

- [ ] **Step 4: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter TerritoryDeployTests` → PASS (2). Full suite → 189. Existing deploy tests (off-mode) still hit the deployment-zone branch unchanged.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine.Tests/TerritoryDeployTests.cs
git commit -m "feat(engine): in TerritoryMode, DeployUnit requires a controlled hex (HexNotControlled); off-mode unchanged"
```

---

### Task 4: Claim tempo — `CaptureHex` must be first + ends the turn

**Files:**
- Modify: `engine/HexWars.Engine/RejectionReason.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs` (`Apply` auto-end, `ApplyCaptureHex`)
- Modify: `engine/HexWars.Engine/LegalMoves.cs` (capture enumeration gate)
- Test: `engine/HexWars.Engine.Tests/ClaimTempoTests.cs` (create)

**Interfaces:**
- Consumes: `GameConfig.TerritoryMode`/`ClaimEndsTurn` (Task 1), `GameState.MovedUnitIds`/`AttackedUnitIds`, existing `ApplyCaptureHex`.
- Produces: `RejectionReason.MustClaimFirst`; in `TerritoryMode && ClaimEndsTurn`, `CaptureHex` rejects unless no unit has moved/attacked this turn, and ends the turn on success; `LegalMoves` omits `CaptureHex` once an army action has occurred.

- [ ] **Step 1: Write the failing test**

Create `engine/HexWars.Engine.Tests/ClaimTempoTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ClaimTempoTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly HexCoord B = new HexCoord(1, 0);
        static readonly HexCoord C = new HexCoord(2, 0);
        static readonly UnitStats S = new UnitStats(8, 3, 2, 4, 2, 1, 1, 3, 1);

        // P0 has a unit on A (a neutral hex) and another on B; whole-army policy; territory mode, claim ends turn.
        static GameState State()
        {
            var board = new Board(new[]
            {
                new Tile(A, 0, TerrainType.Plains), new Tile(B, 0, TerrainType.Plains), new Tile(C, 0, TerrainType.Plains),
            });
            var p0 = new PlayerState(PlayerId.Player0, 50, null, new[]
            {
                new Unit(1, PlayerId.Player0, S, A, 0),
                new Unit(2, PlayerId.Player0, S, B, 0),
            });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true, captureCost: 3);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 9);
        }

        [Test]
        public void Claim_AsFirstAction_Succeeds_AndEndsTurn()
        {
            var r = GameEngine.Apply(State(), new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Board.Controller(A), Is.EqualTo(PlayerId.Player0));
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player1), "claim ends the turn");
        }

        [Test]
        public void Claim_AfterMoving_Rejected()
        {
            var moved = GameEngine.Apply(State(), new MoveUnit(PlayerId.Player0, 2, C));
            Assert.That(moved.Success, Is.True);
            Assert.That(moved.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0), "whole-army: still P0's turn");

            var r = GameEngine.Apply(moved.NewState, new CaptureHex(PlayerId.Player0, A));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.MustClaimFirst));
        }

        [Test]
        public void LegalMoves_OmitsCapture_AfterAnArmyAction()
        {
            var moved = GameEngine.Apply(State(), new MoveUnit(PlayerId.Player0, 2, C));
            var moves = LegalMoves.For(moved.NewState);
            Assert.That(moves.Any(m => m is CaptureHex), Is.False);

            var fresh = LegalMoves.For(State());
            Assert.That(fresh.Any(m => m is CaptureHex c && c.Cell == A), Is.True);
        }
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter ClaimTempoTests`
Expected: FAIL — `MustClaimFirst` doesn't exist; claim doesn't end the turn; capture is still enumerated after a move.

- [ ] **Step 3: Add the rejection reason**

In `engine/HexWars.Engine/RejectionReason.cs`, add to the enum (before the closing brace):

```csharp
        MustClaimFirst,
```

- [ ] **Step 4: Gate the handler (`ApplyCaptureHex`)**

In `engine/HexWars.Engine/GameEngine.cs` `ApplyCaptureHex`, immediately after the `AlreadyControlled` check and before the cost is computed, add:

```csharp
            if (state.Config.TerritoryMode && state.Config.ClaimEndsTurn
                && (state.MovedUnitIds.Count > 0 || state.AttackedUnitIds.Count > 0))
                return Result.Reject(state, RejectionReason.MustClaimFirst);
```

- [ ] **Step 5: Auto-end the turn after a claim**

In `engine/HexWars.Engine/GameEngine.cs` `Apply`, change the auto-end condition:

```csharp
            if (!newState.IsGameOver && !(command is EndTurn)
                && newState.Config.TurnPolicy.AutoEndTurnAfter(command))
```

to also end the turn after a territory claim:

```csharp
            if (!newState.IsGameOver && !(command is EndTurn)
                && (newState.Config.TurnPolicy.AutoEndTurnAfter(command)
                    || (newState.Config.TerritoryMode && newState.Config.ClaimEndsTurn && command is CaptureHex)))
```

- [ ] **Step 6: Gate the LegalMoves enumeration**

In `engine/HexWars.Engine/LegalMoves.cs`, just before the `foreach (var unit in player.UnitsOnBoard)` loop, compute whether claiming is currently legal:

```csharp
            bool claimLegal = !(state.Config.TerritoryMode && state.Config.ClaimEndsTurn)
                || (state.MovedUnitIds.Count == 0 && state.AttackedUnitIds.Count == 0);
```

Then change the existing capture enumeration guard from:

```csharp
                if (board.Controller(unit.Cell) != me
                    && player.Points >= CaptureCostFor(state, unit.Cell))
                    moves.Add(new CaptureHex(me, unit.Cell));
```

to:

```csharp
                if (claimLegal
                    && board.Controller(unit.Cell) != me
                    && player.Points >= CaptureCostFor(state, unit.Cell))
                    moves.Add(new CaptureHex(me, unit.Cell));
```

- [ ] **Step 7: Run tests**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo --filter ClaimTempoTests` → PASS (3). Full suite → 192. Existing capture tests (off-mode, or TerritoryMode-off) are unaffected: the `MustClaimFirst` gate and the auto-end only fire when `TerritoryMode && ClaimEndsTurn`, and `claimLegal` is `true` whenever that combination is off.

- [ ] **Step 8: Commit**

```bash
git add engine/HexWars.Engine/RejectionReason.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine.Tests/ClaimTempoTests.cs
git commit -m "feat(engine): claim tempo — in TerritoryMode+ClaimEndsTurn, CaptureHex must be the turn's first action (MustClaimFirst) and ends the turn; LegalMoves gates it"
```

---

### Task 5: Rebuild engine + re-sync the Plugins DLL

**Files:** Modify (binary): `Assets/HexWars/Plugins/HexWars.Engine.dll`

**Interfaces:** none — this exposes Tasks 1–4's new API (`TerritoryMode`, `ClaimEndsTurn`, batch `WithControl`, `MustClaimFirst`) to Unity so Part B compiles.

- [ ] **Step 1: Full engine suite green**

Run: `dotnet test engine/HexWars.Engine.Tests -c Debug --nologo`
Expected: PASS — 192 (184 baseline + Phase3Config 2 + BoardSeedControl 1 + TerritoryDeploy 2 + ClaimTempo 3).

- [ ] **Step 2: Build Release**

Run: `dotnet build engine/HexWars.Engine -c Release --nologo`
Expected: `Build succeeded. 0 Error(s)`.

- [ ] **Step 3: Sync the DLL**

Run: `cp -f engine/HexWars.Engine/bin/Release/netstandard2.1/HexWars.Engine.dll Assets/HexWars/Plugins/HexWars.Engine.dll`

- [ ] **Step 4: Verify Unity compiles**

Use `mcp__coplay-mcp__check_compile_errors`. Expected: No compile errors.

- [ ] **Step 5: Commit (no-op if the DLL is gitignored)**

```bash
git add -A engine docs && git commit -m "chore(engine): territory-playable engine rules complete; Plugins DLL re-synced" || echo "nothing to commit"
```

---

## Part B — Presentation (Unity; verify by compile + play, not unit tests)

> For each Part B task: make the edit, run `mcp__coplay-mcp__check_compile_errors` (expect none), then confirm the behavior in Play. The slice's real acceptance is Task 10.

### Task 6: `GameBootstrap` — territory mode toggle, seeded home control, starting points

**Files:** Modify `Assets/HexWars/Presentation/GameBootstrap.cs`

**Interfaces:**
- Consumes: `GameConfig.Default(..., winConditions, startingPoints, territoryMode)`, `Board.WithControl(IEnumerable<HexCoord>, PlayerId)`, `WinBy`.
- Produces: a runnable territory game — config built with `TerritoryMode`, both home zones seeded as controlled, players start with `TerritoryStartingPoints`.

- [ ] **Step 1: Add Inspector fields**

After the `OneActionPerTurn` field block, add:

```csharp
        [Tooltip("On = territory mode: control gates deploy/build, claiming a hex is a turn-exclusive action. Takes effect on a new game.")]
        public bool TerritoryMode = false;
        [Tooltip("Points each player starts with in territory mode (you need these to claim/build).")]
        public int TerritoryStartingPoints = 40;
```

- [ ] **Step 2: Build the territory config and seed control in `NewGame()`**

Replace the config + board setup in `NewGame()`:

```csharp
            var config = GameConfig.Default(biomesEnabled: BiomesEnabled,
                                            turnPolicy: OneActionPerTurn ? new OneActionPolicy() : null);
            var genConfig = new BoardGenConfig(Width, Height, MaxElevation, ZoneDepth, FlatChance,
                                               PlainsWeight, ForestWeight, RoughWeight, WaterWeight);
            var board = new RandomBoardGenerator(genConfig).Generate(Seed);
```

with:

```csharp
            var config = TerritoryMode
                ? GameConfig.Default(biomesEnabled: BiomesEnabled,
                                     turnPolicy: OneActionPerTurn ? new OneActionPolicy() : null,
                                     winConditions: WinBy.Economy | WinBy.Annihilation,
                                     startingPoints: TerritoryStartingPoints,
                                     territoryMode: true)
                : GameConfig.Default(biomesEnabled: BiomesEnabled,
                                     turnPolicy: OneActionPerTurn ? new OneActionPolicy() : null);
            var genConfig = new BoardGenConfig(Width, Height, MaxElevation, ZoneDepth, FlatChance,
                                               PlainsWeight, ForestWeight, RoughWeight, WaterWeight);
            var board = new RandomBoardGenerator(genConfig).Generate(Seed);
            if (TerritoryMode)
            {
                board = board.WithControl(board.DeploymentZone(PlayerId.Player0), PlayerId.Player0);
                board = board.WithControl(board.DeploymentZone(PlayerId.Player1), PlayerId.Player1);
            }
```

- [ ] **Step 3: Give starting points in `BuildPlayer`**

Change the `BuildPlayer` calls in `NewGame()`:

```csharp
            var p0 = BuildPlayer(board, PlayerId.Player0, ref nextId);
            var p1 = BuildPlayer(board, PlayerId.Player1, ref nextId);
```

to pass the starting points:

```csharp
            int startPts = TerritoryMode ? config.StartingPoints : 0;
            var p0 = BuildPlayer(board, PlayerId.Player0, startPts, ref nextId);
            var p1 = BuildPlayer(board, PlayerId.Player1, startPts, ref nextId);
```

Change `BuildPlayer`'s signature and both `new PlayerState(...)` lines:

```csharp
        PlayerState BuildPlayer(Board board, PlayerId id, int startingPoints, ref int nextId)
        {
            if (!DemoPieces)
                return new PlayerState(id, startingPoints);
            ...
            return new PlayerState(id, startingPoints, unitsOnBoard: units);
        }
```

(The `...` body — the `flatZone`/`demos`/`units` block — is unchanged.)

- [ ] **Step 4: Verify**

`check_compile_errors` → none. Play with `TerritoryMode` on: the game starts, P0/P1 home zones exist, each player shows `TerritoryStartingPoints` points in the HUD banner. (Overlay/claim come next.)

- [ ] **Step 5: Commit**

```bash
git add Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(ui): GameBootstrap TerritoryMode toggle — territory config (Economy+Annihilation), seeded home-zone control, starting points"
```

---

### Task 7: Control overlay — controlled hexes carry the owner's color

**Files:** Modify `Assets/HexWars/Presentation/BoardRenderer.cs`

**Interfaces:**
- Consumes: `GameState.Board`, `Board.Controller`, `Board.Tiles`.
- Produces: a per-controlled-hex colored cap rebuilt on every `RenderEntities(state)` call (which `GameBootstrap` already invokes on new-game and every applied command).

- [ ] **Step 1: Render control caps inside `RenderEntities`**

At the end of `RenderEntities(GameState state)` (after the players loop), add:

```csharp
            ClearChild("Control");
            var controlRoot = ChildRoot("Control");
            foreach (var tile in state.Board.Tiles)
            {
                var owner = state.Board.Controller(tile.Coord);
                if (owner == null) continue;
                BuildControlCap(controlRoot.transform, tile.Coord, tile.Elevation, owner.Value);
            }
```

- [ ] **Step 2: Add the cap builder + a transparent material helper**

Add these methods to `BoardRenderer`:

```csharp
        void BuildControlCap(Transform parent, HexCoord cell, int elevation, PlayerId owner)
        {
            var w = HexLayout.ToWorld(cell, HexSize);
            var cap = GameObject.CreatePrimitive(PrimitiveType.Quad);
            cap.name = "ControlCap";
            DestroyImmediate(cap.GetComponent<Collider>());
            cap.transform.SetParent(parent, false);
            cap.transform.localPosition = new Vector3((float)w.x, TopY(elevation) + 0.03f, (float)w.z);
            cap.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            cap.transform.localScale = Vector3.one * (HexSize * 1.3f);
            var c = owner == PlayerId.Player0
                ? new Color(0.27f, 0.68f, 1f, 0.40f)   // cyan, semi-transparent
                : new Color(0.92f, 0.28f, 0.28f, 0.40f); // red
            var mr = cap.GetComponent<MeshRenderer>();
            mr.sharedMaterial = TransparentColor(c);
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }

        Material TransparentColor(Color c)
        {
            var sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (sh == null) sh = Shader.Find("Unlit/Color");
            var m = new Material(sh);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            m.color = c;
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
            m.SetFloat("_Surface", 1f);                                  // transparent
            m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)RenderQueue.Transparent;
            return m;
        }
```

- [ ] **Step 2b: Clear the Control root on a fresh board**

In `Render(Board board)` (the column builder), also clear stale caps so a brand-new game starts clean: after `ClearChild("Columns");` add `ClearChild("Control");`.

- [ ] **Step 3: Verify**

`check_compile_errors` → none. Play with TerritoryMode on: each player's home-zone hexes are tinted their color; neutral hexes are untinted. Claim a hex (Task 8) and the cap appears on it. Tune the alpha / inset (`+0.03f`, `1.3f`, `0.40f`) by eye if it z-fights or reads weakly.

- [ ] **Step 4: Commit**

```bash
git add Assets/HexWars/Presentation/BoardRenderer.cs
git commit -m "feat(ui): control overlay — controlled hexes carry a translucent owner-colored cap, refreshed each state change"
```

---

### Task 8: Click-the-hex claim/build input + a selected-unit hint

**Files:** Modify `Assets/HexWars/Presentation/UnitInputController.cs`

**Interfaces:**
- Consumes: `GameConfig.TerritoryMode`, `Board.Controller`, `CaptureHex`, `BuildGenerator`, generator presence on a cell.
- Produces: clicking the selected unit's own hex issues `CaptureHex` (if you don't control it) or `BuildGenerator` (if you control it and it's empty); a hint line states the action + cost.

- [ ] **Step 1: Handle a click on the selected unit's own hex**

In `HandleClick`, before the final `Select(unit);`, add the claim/build branch:

```csharp
            // territory: click your selected unit's OWN hex to claim it (if not yours) or build on it
            if (ownSelected && unit == null && tile != null
                && _game.State.Config.TerritoryMode && tile.Coord == _selected.Unit.Cell)
            {
                var st = _game.State;
                var cell = _selected.Unit.Cell;
                if (st.Board.Controller(cell) != active)
                    _game.TryApply(new CaptureHex(active, cell));        // claim / convert (ends the turn)
                else if (!HasGeneratorOn(st, cell))
                    _game.TryApply(new BuildGenerator(active, cell));    // build on owned empty hex
                ReacquireSelection();
                return;
            }
```

Add the helper:

```csharp
        static bool HasGeneratorOn(GameState s, HexCoord cell)
        {
            foreach (var p in s.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == cell) return true;
            return false;
        }
```

- [ ] **Step 2: Add a hint label**

Add a field `Text _hint;` (add `using UnityEngine.UI;` at the top). Build it once in `Start()` after the existing finds:

```csharp
            _hint = MakeHintLabel();
```

Add the builder:

```csharp
        UnityEngine.UI.Text MakeHintLabel()
        {
            var canvasGo = new GameObject("HintCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 450;
            var go = new GameObject("Hint");
            go.transform.SetParent(canvasGo.transform, false);
            var t = go.AddComponent<UnityEngine.UI.Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = 18; t.color = new Color(1f, 0.95f, 0.6f); t.alignment = TextAnchor.LowerCenter;
            var rt = t.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0f); rt.anchorMax = new Vector2(0.5f, 0f);
            rt.pivot = new Vector2(0.5f, 0f); rt.sizeDelta = new Vector2(700f, 30f);
            rt.anchoredPosition = new Vector2(0f, 70f);
            return t;
        }
```

- [ ] **Step 3: Update the hint each frame**

At the end of `Update()`, add:

```csharp
            if (_hint != null) _hint.text = HintText();
```

Add the hint logic:

```csharp
        string HintText()
        {
            if (ReadOnly || _game == null || _game.State == null) return "";
            var st = _game.State;
            if (!st.Config.TerritoryMode || _selected == null) return "";
            if (_selected.Unit.Owner != st.ActivePlayer) return "";
            var cell = _selected.Unit.Cell;
            bool actedAlready = st.MovedUnitIds.Count > 0 || st.AttackedUnitIds.Count > 0;
            if (st.Board.Controller(cell) != st.ActivePlayer)
            {
                if (st.Config.ClaimEndsTurn && actedAlready)
                    return "Can't claim — your army already acted this turn";
                return "Click this hex to CLAIM it (ends your turn)";
            }
            if (!HasGeneratorOn(st, cell))
                return "Click this hex to BUILD a generator";
            return "";
        }
```

(The exact capture cost is in the engine's `CaptureCostFor`/`BuildCost`, which are private; the hint shows the action and that claiming ends the turn. A precise cost number can be added later by exposing those helpers — out of scope here.)

- [ ] **Step 4: Verify**

`check_compile_errors` → none. Play: select a unit on a neutral hex → hint reads "Click this hex to CLAIM it (ends your turn)"; click → it's claimed, turn passes, the control cap appears. Select a unit on an owned empty hex → "Click this hex to BUILD a generator"; click → a generator pylon appears and points drop. After moving a unit, the claim hint switches to the "already acted" message and the claim click is rejected (logged in the EventConsole).

- [ ] **Step 5: Commit**

```bash
git add Assets/HexWars/Presentation/UnitInputController.cs
git commit -m "feat(ui): click-the-hex to claim/build in territory mode + a selected-unit hint line"
```

---

### Task 9: Economy/score HUD

**Files:** Modify `Assets/HexWars/Presentation/GameHud.cs`

**Interfaces:**
- Consumes: `Economy.Income`/`Economy.Upkeep`, `Board.ControlledCount`, `WinCheck.Score`, `GameConfig.TerritoryMode`.
- Produces: in territory mode, the banner shows both players' points, net income, controlled-hex count, and score.

- [ ] **Step 1: Extend `Refresh()`**

Replace the body of `Refresh()`:

```csharp
        void Refresh()
        {
            if (_game == null || _game.State == null) return;
            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            bool p0 = s.ActivePlayer == PlayerId.Player0;
            int who = p0 ? 1 : 2;
            _banner.color = p0 ? new Color(0.4f, 0.8f, 1f) : new Color(1f, 0.45f, 0.45f);

            if (!s.Config.TerritoryMode)
            {
                _banner.text = $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")})     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";
                return;
            }

            _banner.text =
                $"P{who}'s turn  Round {s.Round}     " +
                $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}";
        }

        static string Stat(GameState s, PlayerId id)
        {
            int net = Economy.Income(s, id) - Economy.Upkeep(s, id);
            int pts = s.Player(id).Points;
            int hexes = s.Board.ControlledCount(id);
            int score = WinCheck.Score(s, id);
            string sign = net >= 0 ? "+" : "";
            return $"{pts}p ({sign}{net}/t)  {hexes} hex  score {score}";
        }
```

- [ ] **Step 2: Verify**

`check_compile_errors` → none. Play in territory mode: the banner shows both sides' points, net income per turn, controlled-hex count, and score; values update as you claim hexes and build generators.

- [ ] **Step 3: Commit**

```bash
git add Assets/HexWars/Presentation/GameHud.cs
git commit -m "feat(ui): territory economy HUD — per-player points, net income, controlled hexes, score"
```

---

### Task 10: Hotseat play-through acceptance

**Files:** none (verification).

**Interfaces:** none — this is the slice's acceptance gate.

- [ ] **Step 1: Set up.** In the `HexWars` scene, set `GameBootstrap.TerritoryMode = true` (leave `VsAI` off for hotseat), enter Play.

- [ ] **Step 2: Verify the full loop:**
  - Each player's home zone is tinted their color; the middle is neutral.
  - HUD shows both sides starting with ~40 points and 0 net income.
  - Move a unit toward a neutral hex over a turn or two (whole-army moves work as before).
  - With a unit on a neutral hex, select it → hint "Click to CLAIM (ends your turn)" → click → control cap appears, turn passes to the other player.
  - Back on your turn, select the unit on your now-controlled hex → "Click to BUILD a generator" → click → pylon appears, points drop by the build cost.
  - End a couple of turns → net income ticks your points up (minus upkeep) in the HUD.
  - Confirm the tempo rule: after moving/attacking, the claim hint shows "already acted" and a claim click is rejected (EventConsole logs `MustClaimFirst`).
  - Reach the `EconomyWinThreshold` (or wipe the enemy) → the game ends with the right winner.

- [ ] **Step 3: Confirm the base game is untouched.** Set `TerritoryMode = false`, Play: no control caps, deploy uses the deployment zone, capture isn't offered, HUD shows the classic banner.

- [ ] **Step 4 (optional): Commit a note** if any tuning constants were changed during play (starting points, factors).

---

## Self-Review

- **Spec coverage:** TerritoryMode/ClaimEndsTurn flags → T1; control-gated deploy → T3; claim tempo (first + ends turn) + LegalMoves gate → T4; seed home-zone control helper → T2 (used in T6); starting points → T1+T6; control overlay (owner color) → T7; click-the-hex claim/build + hint → T8; economy/score HUD → T9; mode enable → T6; DLL re-sync → T5; manual acceptance → T10. The "everything is a dial" table is satisfied by existing config + T1's two new flags. Non-goals (setup screen, AI, territory per-hex value/biomes/neutral nodes, replay) are absent (YAGNI).
- **Placeholder scan:** none — engine steps carry full code + exact `dotnet` commands with expected counts; presentation steps carry full code + a compile check + concrete play assertions. The only deliberately-deferred detail (exact cost number in the hint) is called out, not a silent gap.
- **Type consistency:** `TerritoryMode`/`ClaimEndsTurn`/`startingPoints` (T1) used identically in T3/T4/T6; `Board.WithControl(IEnumerable<HexCoord>, PlayerId)` (T2) called in T6; `RejectionReason.MustClaimFirst` (T4) asserted in T4 tests; `HasGeneratorOn` defined in T8; `Economy.Income/Upkeep`, `Board.ControlledCount`, `WinCheck.Score` (existing) used in T9. Auto-end edit in T4 matches the `Apply` flow in the current `GameEngine.cs`.
- **Backward-compat:** every engine change is gated by `TerritoryMode` (default false) and, for the tempo, additionally `ClaimEndsTurn`; off-mode paths are byte-unchanged, so the 184 existing tests stay green.
