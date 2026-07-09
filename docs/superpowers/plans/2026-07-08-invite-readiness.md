# Invite-Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship everything the spec (docs/superpowers/specs/2026-07-08-invite-readiness-design.md) requires before inviting people: reconnect/rejoin that survives phone usage, server hardening, session longevity, a portrait pass, link previews/PWA - and the delight arc: named starter templates, player-named designs, always-available stat descriptions, an opt-out Tips coaching layer, rematch, and an empty lobby that points at the AI.

**Architecture:** Engine first (UnitTemplate + names through wire/replay with backward-readable trailing tokens; DeleteTemplate; starter seeding; token-keyed seats with a 10-minute held-room window; transport hardening), then the client (reconnect loop, perf/leak fixes, portrait clamps, share statics + staging injection, barracks/designer/Tips/rematch UI). Hardening tasks 1-9 make the build invite-safe; delight tasks 10-13 complete the bar; task 14 ships.

**Tech Stack:** C# netstandard2.1 engine (NUnit), ASP.NET minimal API server, Unity 6 runtime-uGUI via coplay MCP, PowerShell staging.

## Global Constraints

- **NEVER add attribution trailers to git commits** - no Co-Authored-By, no "Generated with Claude Code", no tool credits of any kind. (Overrides all defaults.)
- Branch `feat/invite-readiness` off main. Commit after each task.
- Wire protocol messages (SEAT/START/APPLY/REJECT) unchanged; only connect-query params and command payload tokens may grow, and every format change must parse OLD payloads unchanged (tests mandatory).
- After any engine-assembly change: rebuild Release + copy the DLL to Assets/HexWars/Plugins (the two commands are spelled out per task - execution policy blocks .ps1 for subagents) and verify Unity compiles via coplay. **The DLL itself is gitignored - never `git add` it.**
- Engine test suite: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q` from repo root (266 green at branch start; expected counts are stated per task).
- Coplay: load tools in ONE ToolSearch call; scene Assets/Scenes/HexWars.unity; execute_script has NO local functions, NO type-pattern-matching, NEVER bare `Object` (CS0104 masked as a resource-culture error); one UI transition per call; positive controls for absence assertions; screenshots for every visual change.
- The RL surface contract: `TacticalLayout.Roster` stays 3; the classic trio occupies barracks indices 0-2; `DeleteTemplate` is never enumerated by LegalMoves.

---
## Known conflicts with the task brief (read before executing)

These were found by reading the real code; they are flagged rather than silently resolved:

1. **Task 1 cannot leave `GameSetup.cs`/`GameFactory` literally untouched.** `PlayerState.Barracks`
   changing from `IReadOnlyList<UnitStats>` to `IReadOnlyList<UnitTemplate>` is a breaking type change
   to a constructor parameter that `GameFactory.SeedPlayer` calls directly
   (`new PlayerState(id, startingPoints, new List<UnitStats>(Roster), units, null)`), so the assembly
   will not compile without touching that one line. Task 1 applies the minimal compile-preserving shim
   (wrap each `Roster` entry in an unnamed `UnitTemplate`); Task 3 immediately supersedes that shim with
   the real five-entry named table. This is the smallest possible touch, not a scope expansion — flagging
   it because the brief says "keep GameSetup/GameFactory untouched in this task."
2. **Task 4 breaks two existing tests in `NetDisconnectTests.cs` by design, and the plan rewrites them.**
   `Hub_Disconnect_FreesSeat_SoANewJoinerIsSeatedNotTurnedAway` and
   `Hub_Disconnect_LastMember_ResetsRoomForAFreshGame` currently assert that `MatchHub.Disconnect` frees
   the departing connection's seat immediately (today's per-socket-id behavior) and that an emptied room
   resets instantly. Token-keyed seats + the 10-minute hold window are the *opposite* contract: a
   started room's seats must stay reserved to their tokens (not freed to the next comer) so the *same*
   token can reclaim them, and a started-then-emptied room must be held, not reset. Task 4 rewrites both
   tests to assert the new contract (a different token gets `SEAT FULL`; a held room stays held) rather
   than leaving permanently-red tests in the suite. This is a deliberate behavior change the spec calls
   for, not a silent deviation — flagging it because it touches pre-existing test files not named in the
   brief's per-task file lists.
3. **`DeleteTemplate`'s base-call syntax.** The brief's snippet is
   `public sealed record DeleteTemplate(PlayerId Issuer, int TemplateIndex) : Command;` — but the real
   `Command` is `public abstract record Command(PlayerId Issuer);` (a positional record), so every
   subrecord in the codebase calls the positional base ctor, e.g.
   `public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats) : Command(Issuer);`. `: Command;`
   alone does not compile. Task 2 uses `: Command(Issuer)`, matching every other command in `Command.cs`.
4. **`GameEngine.Apply`'s auto-end-turn guard needs an explicit `DeleteTemplate` exclusion, not just
   "don't touch `MovedUnitIds`/`AttackedUnitIds`."** `OneActionPolicy.AutoEndTurnAfter` ends the turn
   after *any* single non-`EndTurn` command unconditionally (proven by the existing test
   `OneActionPolicy_AutoEndsTurn_AfterASingleAction`, which auto-ends the turn after a bare `CreateUnit`
   — a command that also never touches those two sets). Without an explicit
   `&& !(command is DeleteTemplate)` alongside the existing `&& !(command is EndTurn)` in
   `GameEngine.Apply`, deleting a template under `OneActionPolicy` (or after a `KActionsPolicy` budget is
   already spent) would end the player's turn — directly violating "free action, no turn consumption."
   Task 2 adds that guard explicitly; see Task 2 Step 2.

---

### Task 1: UnitTemplate + names end to end

**Files:**
- Create: `engine/HexWars.Engine/UnitTemplate.cs`
- Create: `engine/HexWars.Engine.Tests/UnitTemplateTests.cs`
- Modify: `engine/HexWars.Engine/Unit.cs`
- Modify: `engine/HexWars.Engine/PlayerState.cs`
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/LegalMoves.cs`
- Modify: `engine/HexWars.Engine/WinCheck.cs`
- Modify: `engine/HexWars.Engine/Net/CommandWire.cs`
- Modify: `engine/HexWars.Engine/ReplayFile.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalLayout.cs`
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs` (compile-preserving shim only — see conflict #1 above; Task 3 replaces this)
- Modify: `engine/HexWars.Engine.Tests/UnitTests.cs`
- Modify: `engine/HexWars.Engine.Tests/CommandWireTests.cs`
- Modify: `engine/HexWars.Engine.Tests/ReplayFileTests.cs`
- Modify: `engine/HexWars.Engine.Tests/LegalMovesTests.cs`
- Modify: `engine/HexWars.Engine.Tests/WinCheckTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TerritoryDeployTests.cs`
- Modify: `Assets/HexWars/Plugins/HexWars.Engine.dll` (rebuild + copy, final step)
- Audited, NO CHANGE NEEDED: `engine/HexWars.Engine/GreedyAgent.cs` (`player.Barracks.Count == 0` and
  `new CreateUnit(me, fighter)` both compile unchanged — `Count` works on any `IReadOnlyList<T>`, and
  `CreateUnit`'s new `Name` parameter defaults to `""`), `engine/HexWars.Engine/RandomAgent.cs`
  (`new CreateUnit(me, stats)` — same default-parameter reasoning), `engine/HexWars.Engine/HoarderAgent.cs`
  (contains no `Barracks`/`UnitStats` reference at all).

**Interfaces:**
- Produces: `public readonly struct UnitTemplate { public readonly string Name; public readonly UnitStats Stats; public UnitTemplate(string name, UnitStats stats); public static string Sanitize(string? raw); }`
- Produces: `PlayerState.Barracks : IReadOnlyList<UnitTemplate>` (was `IReadOnlyList<UnitStats>`).
- Produces: `Unit.Name : string`, `Unit.DisplayName : string` (`Roles.Dominant(Stats).ToString()` fallback when `Name` is empty), `Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, string name = "")`.
- Produces: `public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats, string Name = "") : Command(Issuer);`
- Produces (internal, reused by `ReplayFile`): `CommandWire.EncodeName(string)`, `CommandWire.DecodeName(string)`.
- Consumes: `Roles.Dominant(UnitStats)` (existing, unchanged), `Economy.DeployCost(UnitStats, GameConfig)` (existing, unchanged — call sites now pass `.Stats`).

- [ ] **Step 1: RED — `UnitTemplate` doesn't exist yet**

Create `engine/HexWars.Engine.Tests/UnitTemplateTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>UnitTemplate = name + stats, the barracks entry. Sanitize is the engine-boundary
    /// gate: no client can wire an unparseable or abusive-length name (spec §5/§7).</summary>
    public class UnitTemplateTests
    {
        [Test]
        public void Ctor_StoresNameAndStats()
        {
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var t = new UnitTemplate("Longshot", stats);
            Assert.That(t.Name, Is.EqualTo("Longshot"));
            Assert.That(t.Stats, Is.EqualTo(stats));
        }

        [Test]
        public void Sanitize_TrimsLeadingAndTrailingWhitespace() =>
            Assert.That(UnitTemplate.Sanitize("  Doom Turtle  "), Is.EqualTo("Doom Turtle"));

        [Test]
        public void Sanitize_CapsAtTwentyCharacters() =>
            Assert.That(UnitTemplate.Sanitize(new string('A', 30)), Is.EqualTo(new string('A', 20)));

        [Test]
        public void Sanitize_StripsDisallowedCharacters() =>
            Assert.That(UnitTemplate.Sanitize("<Doom>Turtle!!"), Is.EqualTo("DoomTurtle"));

        [Test]
        public void Sanitize_NullOrEmpty_ReturnsEmptyString()
        {
            Assert.That(UnitTemplate.Sanitize(null), Is.EqualTo(""));
            Assert.That(UnitTemplate.Sanitize(""), Is.EqualTo(""));
        }

        [Test]
        public void Sanitize_KeepsWhitelistedPunctuation() =>
            Assert.That(UnitTemplate.Sanitize("Mama's Boy_2-nd"), Is.EqualTo("Mama's Boy_2-nd"));
    }
}
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build error — `error CS0246: The type or namespace name 'UnitTemplate' could not be found`.

- [ ] **Step 2: GREEN — add `UnitTemplate.cs`**

Create `engine/HexWars.Engine/UnitTemplate.cs`:

```csharp
using System.Text;

namespace HexWars.Engine
{
    /// <summary>
    /// A reusable barracks blueprint: a name (shown in the tooltip/UI) plus the purchased stat line.
    /// Deploying a template clones <see cref="Stats"/> onto a new <see cref="Unit"/> and copies
    /// <see cref="Name"/> onto it — <see cref="Unit.DisplayName"/> falls back to the dominant role
    /// when Name is empty. Immutable.
    /// </summary>
    public readonly struct UnitTemplate
    {
        public readonly string Name;
        public readonly UnitStats Stats;

        public UnitTemplate(string name, UnitStats stats)
        {
            Name = name;
            Stats = stats;
        }

        /// <summary>Sanitize a raw (possibly null, possibly attacker-supplied) name at the engine
        /// boundary: trim, keep only <c>[A-Za-z0-9 _-']</c>, cap at 20 characters. Null/empty/fully
        /// stripped input becomes "" — callers fall back to the dominant-role label for display.</summary>
        public static string Sanitize(string? raw)
        {
            if (string.IsNullOrEmpty(raw)) return "";
            var sb = new StringBuilder(20);
            foreach (char ch in raw.Trim())
            {
                if (sb.Length == 20) break;
                bool allowed = (ch >= 'A' && ch <= 'Z') || (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9')
                               || ch == ' ' || ch == '_' || ch == '-' || ch == '\'';
                if (allowed) sb.Append(ch);
            }
            return sb.ToString();
        }
    }
}
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   272, Skipped:     0, Total:   272` (266 + 6 new).

- [ ] **Step 3: RED — `Unit` has no `Name`/`DisplayName` yet**

Append to `engine/HexWars.Engine.Tests/UnitTests.cs` (after `WithCell_MovesTheUnit_KeepingHp`, inside the existing `UnitTests` class, before its closing brace):

```csharp

        [Test]
        public void DisplayName_ReturnsName_WhenSet()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Doom Turtle");
            Assert.That(u.DisplayName, Is.EqualTo("Doom Turtle"));
        }

        [Test]
        public void DisplayName_FallsBackToDominantRole_WhenNameEmpty()
        {
            var brute = new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1); // dominant stat: Health -> Brute
            var u = new Unit(1, PlayerId.Player0, brute, new HexCoord(0, 0), 0);
            Assert.That(u.Name, Is.EqualTo(""));
            Assert.That(u.DisplayName, Is.EqualTo("Brute"));
        }

        [Test]
        public void WithDamage_PreservesName()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Recon");
            Assert.That(u.WithDamage(2).Name, Is.EqualTo("Recon"));
        }

        [Test]
        public void WithCell_PreservesName()
        {
            var u = new Unit(1, PlayerId.Player0, Hp(5), new HexCoord(0, 0), 0, "Recon");
            Assert.That(u.WithCell(new HexCoord(1, 1), 0).Name, Is.EqualTo("Recon"));
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build error — `error CS7036: There is no argument given that corresponds to the required
formal parameter` / `CS1729: 'Unit' does not contain a constructor that takes 6 arguments` (the 5-arg
`Unit` ctor has no `name` parameter yet, and `DisplayName` doesn't exist).

- [ ] **Step 4: GREEN — `Unit` gains `Name`/`DisplayName`**

Old (`engine/HexWars.Engine/Unit.cs`, full file):

```csharp
namespace HexWars.Engine
{
    /// <summary>
    /// An on-board unit: its purchased <see cref="Stats"/>, owner, 3D position
    /// (<see cref="Cell"/> = q,r column + its own <see cref="Elevation"/>), and current health.
    /// Immutable — mutations return new copies, so <c>Apply</c> can fork state without side effects.
    /// </summary>
    public readonly struct Unit
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public UnitStats Stats { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }

        /// <summary>Create a fresh unit at full health.</summary>
        public Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation)
            : this(id, owner, stats, cell, elevation, stats.Health) { }

        private Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, int currentHp)
        {
            Id = id;
            Owner = owner;
            Stats = stats;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
        }

        public bool IsAlive => CurrentHp > 0;

        /// <summary>A copy with <paramref name="amount"/> damage applied (clamped at 0 HP).</summary>
        public Unit WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Unit(Id, Owner, Stats, Cell, Elevation, hp);
        }

        /// <summary>A copy moved to a new 3D position, keeping current health.</summary>
        public Unit WithCell(HexCoord cell, int elevation) =>
            new Unit(Id, Owner, Stats, cell, elevation, CurrentHp);
    }
}
```

New (full file):

```csharp
namespace HexWars.Engine
{
    /// <summary>
    /// An on-board unit: its purchased <see cref="Stats"/>, owner, 3D position
    /// (<see cref="Cell"/> = q,r column + its own <see cref="Elevation"/>), current health, and the
    /// <see cref="Name"/> copied from the barracks template it was deployed from (empty for units
    /// seeded directly, e.g. the starting army). Immutable — mutations return new copies, so
    /// <c>Apply</c> can fork state without side effects.
    /// </summary>
    public readonly struct Unit
    {
        public int Id { get; }
        public PlayerId Owner { get; }
        public UnitStats Stats { get; }
        public HexCoord Cell { get; }
        public int Elevation { get; }
        public int CurrentHp { get; }
        public string Name { get; }

        /// <summary>Create a fresh unit at full health. <paramref name="name"/> defaults to "" (no
        /// template name) — see <see cref="DisplayName"/> for the fallback shown to a player.</summary>
        public Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, string name = "")
            : this(id, owner, stats, cell, elevation, stats.Health, name) { }

        private Unit(int id, PlayerId owner, UnitStats stats, HexCoord cell, int elevation, int currentHp, string name)
        {
            Id = id;
            Owner = owner;
            Stats = stats;
            Cell = cell;
            Elevation = elevation;
            CurrentHp = currentHp;
            Name = name;
        }

        public bool IsAlive => CurrentHp > 0;

        /// <summary>The name to show a player: the template Name if it has one, else the dominant-role
        /// label (so an unnamed unit still reads as something, never a blank).</summary>
        public string DisplayName => string.IsNullOrEmpty(Name) ? Roles.Dominant(Stats).ToString() : Name;

        /// <summary>A copy with <paramref name="amount"/> damage applied (clamped at 0 HP).</summary>
        public Unit WithDamage(int amount)
        {
            int hp = CurrentHp - amount;
            if (hp < 0) hp = 0;
            return new Unit(Id, Owner, Stats, Cell, Elevation, hp, Name);
        }

        /// <summary>A copy moved to a new 3D position, keeping current health.</summary>
        public Unit WithCell(HexCoord cell, int elevation) =>
            new Unit(Id, Owner, Stats, cell, elevation, CurrentHp, Name);
    }
}
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   276, Skipped:     0, Total:   276` (272 + 4 new).

- [ ] **Step 5: RED — the barracks type change, wire format, and replay format (one wide step)**

`PlayerState.Barracks` changing type is atomic across the assembly: `GameEngine`, `LegalMoves`,
`WinCheck`, `CommandWire`, `ReplayFile`, `TacticalLayout`, and `GameSetup` all reference it, and C#
compiles the whole assembly together — there is no intermediate state where some call sites are
migrated and others aren't. This step adds every test that exercises the target (post-migration) shape;
it will not compile against today's production code, which is the expected RED.

Add three tests to `engine/HexWars.Engine.Tests/CommandWireTests.cs` (after the existing
`CreateUnit_RoundTrips` test, before `DeployUnit_RoundTrips`):

```csharp
        [Test]
        public void CreateUnit_RoundTrips_WithName() =>
            RoundTrips(new CreateUnit(P1, new UnitStats(3, 4, 1, 2, 1, 2, 1, 3, 1), "Doom Turtle"));

        [Test]
        public void CreateUnit_Read_OldFormatMissingNameToken_DefaultsToEmptyName()
        {
            var cmd = (CreateUnit)CommandWire.Read("C 1 3 4 1 2 1 2 1 3 1");
            Assert.That(cmd.Name, Is.EqualTo(""));
        }

        [Test]
        public void CreateUnit_Write_EncodesSpacesAsUnderscores()
        {
            var wire = CommandWire.Write(new CreateUnit(P0, new UnitStats(1, 0, 0, 0, 0, 0, 0, 0, 0), "Doom Turtle"));
            Assert.That(wire, Does.EndWith(" Doom_Turtle"));
        }
```

In `engine/HexWars.Engine.Tests/ReplayFileTests.cs`: add `using System;` to the top of the file (needed
for `StringComparison` below), change the `RichStartState_RoundTrips` test's barracks construction and
add a name assertion, and add two new tests.

Old (top of file):
```csharp
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;
```
New:
```csharp
using System;
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;
```

Old (inside `RichStartState_RoundTrips`):
```csharp
            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation);
            var g0 = new Generator(2, P0, z0[1], board.TileAt(z0[1]).Elevation, 10);
            var p0 = new PlayerState(P0, 15, new[] { stats }, new[] { u0 }, new[] { g0 });
```
New:
```csharp
            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation);
            var g0 = new Generator(2, P0, z0[1], board.TileAt(z0[1]).Elevation, 10);
            var p0 = new PlayerState(P0, 15, new[] { new UnitTemplate("Vanguard", stats) }, new[] { u0 }, new[] { g0 });
```

Old (end of `RichStartState_RoundTrips`):
```csharp
            Assert.That(rp0.Barracks.Count, Is.EqualTo(1));
            Assert.That(s.Player(P1).UnitsOnBoard.Count, Is.EqualTo(1));
        }
    }
}
```
New:
```csharp
            Assert.That(rp0.Barracks.Count, Is.EqualTo(1));
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo("Vanguard"));
            Assert.That(s.Player(P1).UnitsOnBoard.Count, Is.EqualTo(1));
        }

        [Test]
        public void UnitAndBarracks_Name_RoundTrip()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var z0 = new List<HexCoord>(board.DeploymentZone(P0));

            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation, "Doom Turtle");
            var p0 = new PlayerState(P0, 10, new[] { new UnitTemplate("Longshot", stats) }, new[] { u0 });
            var p1 = new PlayerState(P1, 10);
            var start = new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 2);

            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;
            var rp0 = s.Player(P0);

            Assert.That(rp0.UnitsOnBoard[0].Name, Is.EqualTo("Doom Turtle"));
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo("Longshot"));
        }

        [Test]
        public void OldFormatReplay_MissingNameTokens_DefaultsToEmptyNames()
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(7);
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var z0 = new List<HexCoord>(board.DeploymentZone(P0));

            var u0 = new Unit(1, P0, stats, z0[0], board.TileAt(z0[0]).Elevation, "Recon");
            var p0 = new PlayerState(P0, 10, new[] { new UnitTemplate("Vanguard", stats) }, new[] { u0 });
            var p1 = new PlayerState(P1, 10);
            var start = new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 2);

            string modern = ReplayFile.Write(start, new List<Command>());

            // Simulate a pre-name-feature payload by dropping the trailing name token from each U/B
            // line — exactly what a file written before this feature shipped looks like.
            var oldLines = modern.Replace("\r\n", "\n").Split('\n');
            for (int i = 0; i < oldLines.Length; i++)
            {
                if (oldLines[i].StartsWith("U ", StringComparison.Ordinal) || oldLines[i].StartsWith("B ", StringComparison.Ordinal))
                {
                    int lastSpace = oldLines[i].LastIndexOf(' ');
                    oldLines[i] = oldLines[i].Substring(0, lastSpace);
                }
            }
            string old = string.Join("\n", oldLines);

            var s = ReplayFile.Read(old).Start;
            var rp0 = s.Player(P0);
            Assert.That(rp0.UnitsOnBoard[0].Name, Is.EqualTo(""), "old payloads with no trailing unit-name token default to \"\"");
            Assert.That(rp0.Barracks[0].Name, Is.EqualTo(""), "old payloads with no trailing barracks-name token default to \"\"");
            Assert.That(rp0.UnitsOnBoard[0].Stats.Damage, Is.EqualTo(3), "everything else still parses");
        }
    }
}
```

In `engine/HexWars.Engine.Tests/LegalMovesTests.cs`, old:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 5, barracks: new[] { TestStates.Cost(2) }, unitsOnBoard: new[] { myUnit });
```
New:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 5, barracks: new[] { new UnitTemplate("", TestStates.Cost(2)) }, unitsOnBoard: new[] { myUnit });
```

In `engine/HexWars.Engine.Tests/WinCheckTests.cs`, three call sites. Old:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 5,
                barracks: new[] { Cost(3) }, unitsOnBoard: new[] { unit }, generators: new[] { gen });
```
New:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 5,
                barracks: new[] { new UnitTemplate("", Cost(3)) }, unitsOnBoard: new[] { unit }, generators: new[] { gen });
```
Old:
```csharp
            var s = State(new PlayerState(PlayerId.Player0, 1, barracks: new[] { Cost(1) }),
                          new PlayerState(PlayerId.Player1, 0), 2);
```
New:
```csharp
            var s = State(new PlayerState(PlayerId.Player0, 1, barracks: new[] { new UnitTemplate("", Cost(1)) }),
                          new PlayerState(PlayerId.Player1, 0), 2);
```
Old:
```csharp
            var s = State(new PlayerState(PlayerId.Player0, 0, barracks: new[] { Cost(1) }),
                          new PlayerState(PlayerId.Player1, 0), 2);
```
New:
```csharp
            var s = State(new PlayerState(PlayerId.Player0, 0, barracks: new[] { new UnitTemplate("", Cost(1)) }),
                          new PlayerState(PlayerId.Player1, 0), 2);
```

In `engine/HexWars.Engine.Tests/TerritoryDeployTests.cs`, old:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 100, new[] { S });
```
New:
```csharp
            var p0 = new PlayerState(PlayerId.Player0, 100, new[] { new UnitTemplate("", S) });
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build errors — e.g. `CS1503: Argument 1: cannot convert from 'HexWars.Engine.UnitTemplate[]'
to 'System.Collections.Generic.IReadOnlyList<HexWars.Engine.UnitStats>'` and
`CS1739: 'CreateUnit' does not contain a parameter named 'Name'` (production code still expects
`IReadOnlyList<UnitStats>` barracks and a 2-arg `CreateUnit`).

- [ ] **Step 6: GREEN — migrate `PlayerState.Barracks` and every consumer**

`engine/HexWars.Engine/PlayerState.cs` — old (full file):
```csharp
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// One player's economy and forces: banked <see cref="Points"/>, designed unit templates in the
    /// <see cref="Barracks"/> (reusable blueprints — deploying a clone does NOT consume them),
    /// on-board <see cref="UnitsOnBoard"/>, and <see cref="Generators"/>. Immutable.
    /// </summary>
    public sealed class PlayerState
    {
        private static readonly IReadOnlyList<UnitStats> NoBarracks = new UnitStats[0];
        private static readonly IReadOnlyList<Unit> NoUnits = new Unit[0];
        private static readonly IReadOnlyList<Generator> NoGenerators = new Generator[0];

        public PlayerId Id { get; }
        public int Points { get; }
        public IReadOnlyList<UnitStats> Barracks { get; }
        public IReadOnlyList<Unit> UnitsOnBoard { get; }
        public IReadOnlyList<Generator> Generators { get; }

        /// <summary>Cumulative point value of enemy entities this player has destroyed (for Score).</summary>
        public int DestroyedValue { get; }

        public PlayerState(
            PlayerId id,
            int points,
            IReadOnlyList<UnitStats>? barracks = null,
            IReadOnlyList<Unit>? unitsOnBoard = null,
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

        public PlayerState WithPoints(int points) =>
            new PlayerState(Id, points, Barracks, UnitsOnBoard, Generators, DestroyedValue);

        public PlayerState WithDestroyed(int delta) =>
            new PlayerState(Id, Points, Barracks, UnitsOnBoard, Generators, DestroyedValue + delta);
    }
}
```
New (full file):
```csharp
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// One player's economy and forces: banked <see cref="Points"/>, designed unit templates in the
    /// <see cref="Barracks"/> (reusable blueprints — deploying a clone does NOT consume them),
    /// on-board <see cref="UnitsOnBoard"/>, and <see cref="Generators"/>. Immutable.
    /// </summary>
    public sealed class PlayerState
    {
        private static readonly IReadOnlyList<UnitTemplate> NoBarracks = new UnitTemplate[0];
        private static readonly IReadOnlyList<Unit> NoUnits = new Unit[0];
        private static readonly IReadOnlyList<Generator> NoGenerators = new Generator[0];

        public PlayerId Id { get; }
        public int Points { get; }
        public IReadOnlyList<UnitTemplate> Barracks { get; }
        public IReadOnlyList<Unit> UnitsOnBoard { get; }
        public IReadOnlyList<Generator> Generators { get; }

        /// <summary>Cumulative point value of enemy entities this player has destroyed (for Score).</summary>
        public int DestroyedValue { get; }

        public PlayerState(
            PlayerId id,
            int points,
            IReadOnlyList<UnitTemplate>? barracks = null,
            IReadOnlyList<Unit>? unitsOnBoard = null,
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

        public PlayerState WithPoints(int points) =>
            new PlayerState(Id, points, Barracks, UnitsOnBoard, Generators, DestroyedValue);

        public PlayerState WithDestroyed(int delta) =>
            new PlayerState(Id, Points, Barracks, UnitsOnBoard, Generators, DestroyedValue + delta);
    }
}
```

`engine/HexWars.Engine/Command.cs` — old line:
```csharp
    public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats) : Command(Issuer);
```
New line:
```csharp
    /// <summary>Design and pay for a unit; it goes to the issuer's reserve (off-board). Name is
    /// sanitized at the engine boundary (UnitTemplate.Sanitize); an empty/omitted name is legal — the
    /// dominant-role fallback happens on the deployed Unit's DisplayName, not here.</summary>
    public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats, string Name = "") : Command(Issuer);
```
(This replaces the existing single-line `CreateUnit` record and its preceding doc comment
`/// <summary>Design and pay for a unit; it goes to the issuer's reserve (off-board).</summary>`.)

`engine/HexWars.Engine/GameEngine.cs` — `ApplyCreateUnit`, old:
```csharp
        private static Result ApplyCreateUnit(GameState state, CreateUnit c)
        {
            if (c.Stats.Health < 1) return Result.Reject(state, RejectionReason.InvalidStats);

            var player = state.Player(c.Issuer);
            int fee = state.Config.DesignFee;
            if (player.Points < fee) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var barracks = new List<UnitStats>(player.Barracks) { c.Stats }; // reusable template
            var updated = new PlayerState(player.Id, player.Points - fee, barracks,
                                          player.UnitsOnBoard, player.Generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated));
        }
```
New:
```csharp
        private static Result ApplyCreateUnit(GameState state, CreateUnit c)
        {
            if (c.Stats.Health < 1) return Result.Reject(state, RejectionReason.InvalidStats);

            var player = state.Player(c.Issuer);
            int fee = state.Config.DesignFee;
            if (player.Points < fee) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var template = new UnitTemplate(UnitTemplate.Sanitize(c.Name), c.Stats);
            var barracks = new List<UnitTemplate>(player.Barracks) { template }; // reusable template
            var updated = new PlayerState(player.Id, player.Points - fee, barracks,
                                          player.UnitsOnBoard, player.Generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated));
        }
```

`ApplyDeployUnit`, old:
```csharp
            var stats = player.Barracks[c.TemplateIndex];
            int cost = Economy.DeployCost(stats, state.Config);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var unit = new Unit(state.NextEntityId, c.Issuer, stats, c.Cell, tile.Elevation);
            var units = new List<Unit>(player.UnitsOnBoard) { unit };
```
New:
```csharp
            var template = player.Barracks[c.TemplateIndex];
            int cost = Economy.DeployCost(template.Stats, state.Config);
            if (player.Points < cost) return Result.Reject(state, RejectionReason.InsufficientPoints);

            var unit = new Unit(state.NextEntityId, c.Issuer, template.Stats, c.Cell, tile.Elevation, template.Name);
            var units = new List<Unit>(player.UnitsOnBoard) { unit };
```

`engine/HexWars.Engine/LegalMoves.cs` — old:
```csharp
            for (int i = 0; i < player.Barracks.Count; i++)
            {
                if (player.Points < Economy.DeployCost(player.Barracks[i], state.Config)) continue;
                foreach (var coord in emptyZone)
                    moves.Add(new DeployUnit(me, i, coord));
            }
```
New:
```csharp
            for (int i = 0; i < player.Barracks.Count; i++)
            {
                if (player.Points < Economy.DeployCost(player.Barracks[i].Stats, state.Config)) continue;
                foreach (var coord in emptyZone)
                    moves.Add(new DeployUnit(me, i, coord));
            }
```

`engine/HexWars.Engine/WinCheck.cs` — old:
```csharp
            foreach (var stats in p.Barracks)
                if (p.Points >= Economy.DeployCost(stats, state.Config)) return false; // can redeploy
```
New:
```csharp
            foreach (var t in p.Barracks)
                if (p.Points >= Economy.DeployCost(t.Stats, state.Config)) return false; // can redeploy
```

`engine/HexWars.Engine/Net/CommandWire.cs` — full file, old (as read from the repo):
```csharp
using System;
using System.Globalization;

namespace HexWars.Engine
{
    /// <summary>
    /// Dependency-free, drift-free wire-format for a single <see cref="Command"/>: one line of
    /// space-separated tokens. The SAME code serializes on the authoritative server and deserializes on
    /// each client, so a relayed move reconstructs identically everywhere. Covers every command type
    /// (the format <see cref="ReplayFile"/> pioneered for replays, completed with the territory commands).
    /// </summary>
    public static class CommandWire
    {
        public static string Write(Command c)
        {
            switch (c)
            {
                case MoveUnit m:        return $"M {(int)m.Issuer} {m.UnitId} {m.Dest.Q} {m.Dest.R}";
                case AttackUnit a:      return $"A {(int)a.Issuer} {a.AttackerId} {a.TargetId}";
                case EndTurn e:         return $"E {(int)e.Issuer}";
                case CreateUnit cu:     return $"C {(int)cu.Issuer} {WriteStats(cu.Stats)}";
                case DeployUnit d:      return $"D {(int)d.Issuer} {d.TemplateIndex} {d.Cell.Q} {d.Cell.R}";
                case DeployGenerator g: return $"N {(int)g.Issuer} {g.Cell.Q} {g.Cell.R}";
                case CaptureHex h:      return $"H {(int)h.Issuer} {h.Cell.Q} {h.Cell.R}";
                case BuildGenerator b:  return $"B {(int)b.Issuer} {b.Cell.Q} {b.Cell.R}";
                default: throw new FormatException("unknown command " + c.GetType().Name);
            }
        }

        public static Command Read(string line)
        {
            var p = line.Split(' ');
            var issuer = (PlayerId)I(p[1]);
            switch (p[0])
            {
                case "M": return new MoveUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "A": return new AttackUnit(issuer, I(p[2]), I(p[3]));
                case "E": return new EndTurn(issuer);
                case "C": return new CreateUnit(issuer, ReadStats(p, 2));
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "N": return new DeployGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "H": return new CaptureHex(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "B": return new BuildGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                default: throw new FormatException("unknown command token " + p[0]);
            }
        }

        internal static string WriteStats(UnitStats s) =>
            $"{s.Health} {s.Damage} {s.Defense} {s.Movement} {s.VerticalMovement} {s.Range} {s.RangeArc} {s.Vision} {s.VisionArc}";

        internal static UnitStats ReadStats(string[] p, int o) =>
            new UnitStats(I(p[o]), I(p[o + 1]), I(p[o + 2]), I(p[o + 3]), I(p[o + 4]),
                          I(p[o + 5]), I(p[o + 6]), I(p[o + 7]), I(p[o + 8]));

        private static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    }
}
```
New (full file — adds the trailing name token to `CreateUnit`'s line, and the `EncodeName`/`DecodeName`
helpers `ReplayFile` reuses; `TryRead` is NOT added here — that's Task 5):
```csharp
using System;
using System.Globalization;

namespace HexWars.Engine
{
    /// <summary>
    /// Dependency-free, drift-free wire-format for a single <see cref="Command"/>: one line of
    /// space-separated tokens. The SAME code serializes on the authoritative server and deserializes on
    /// each client, so a relayed move reconstructs identically everywhere. Covers every command type
    /// (the format <see cref="ReplayFile"/> pioneered for replays, completed with the territory commands).
    /// </summary>
    public static class CommandWire
    {
        public static string Write(Command c)
        {
            switch (c)
            {
                case MoveUnit m:        return $"M {(int)m.Issuer} {m.UnitId} {m.Dest.Q} {m.Dest.R}";
                case AttackUnit a:      return $"A {(int)a.Issuer} {a.AttackerId} {a.TargetId}";
                case EndTurn e:         return $"E {(int)e.Issuer}";
                case CreateUnit cu:     return $"C {(int)cu.Issuer} {WriteStats(cu.Stats)} {EncodeName(cu.Name)}";
                case DeployUnit d:      return $"D {(int)d.Issuer} {d.TemplateIndex} {d.Cell.Q} {d.Cell.R}";
                case DeployGenerator g: return $"N {(int)g.Issuer} {g.Cell.Q} {g.Cell.R}";
                case CaptureHex h:      return $"H {(int)h.Issuer} {h.Cell.Q} {h.Cell.R}";
                case BuildGenerator b:  return $"B {(int)b.Issuer} {b.Cell.Q} {b.Cell.R}";
                default: throw new FormatException("unknown command " + c.GetType().Name);
            }
        }

        public static Command Read(string line)
        {
            var p = line.Split(' ');
            var issuer = (PlayerId)I(p[1]);
            switch (p[0])
            {
                case "M": return new MoveUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "A": return new AttackUnit(issuer, I(p[2]), I(p[3]));
                case "E": return new EndTurn(issuer);
                case "C": return new CreateUnit(issuer, ReadStats(p, 2), p.Length > 11 ? DecodeName(p[11]) : "");
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "N": return new DeployGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "H": return new CaptureHex(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "B": return new BuildGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                default: throw new FormatException("unknown command token " + p[0]);
            }
        }

        internal static string WriteStats(UnitStats s) =>
            $"{s.Health} {s.Damage} {s.Defense} {s.Movement} {s.VerticalMovement} {s.Range} {s.RangeArc} {s.Vision} {s.VisionArc}";

        internal static UnitStats ReadStats(string[] p, int o) =>
            new UnitStats(I(p[o]), I(p[o + 1]), I(p[o + 2]), I(p[o + 3]), I(p[o + 4]),
                          I(p[o + 5]), I(p[o + 6]), I(p[o + 7]), I(p[o + 8]));

        /// <summary>Wire-encode a sanitized name: spaces become underscores so it survives the
        /// space-separated line (names never contain any other whitespace after Sanitize). Empty
        /// name encodes to "" — on write this leaves a trailing space, so Split(' ') still yields an
        /// (empty-string) final token; readers use that to distinguish "new format, no name" from
        /// "old format, no token at all" (see Read/DecodeName call sites).</summary>
        internal static string EncodeName(string name) => string.IsNullOrEmpty(name) ? "" : name.Replace(' ', '_');

        /// <summary>Inverse of <see cref="EncodeName"/>.</summary>
        internal static string DecodeName(string token) => string.IsNullOrEmpty(token) ? "" : token.Replace('_', ' ');

        private static int I(string s) => int.Parse(s, CultureInfo.InvariantCulture);
    }
}
```

`engine/HexWars.Engine/ReplayFile.cs` — `WritePlayer`, old:
```csharp
        private static void WritePlayer(StringBuilder sb, PlayerState p)
        {
            sb.Append("PLAYER ").Append((int)p.Id).Append(' ').Append(p.Points)
              .Append(' ').Append(p.UnitsOnBoard.Count).Append(' ').Append(p.Generators.Count)
              .Append(' ').Append(p.Barracks.Count).Append('\n');

            foreach (var u in p.UnitsOnBoard)
            {
                sb.Append("U ").Append(u.Id).Append(' ').Append((int)u.Owner).Append(' ');
                AppendStats(sb, u.Stats);
                sb.Append(' ').Append(u.Cell.Q).Append(' ').Append(u.Cell.R).Append(' ').Append(u.Elevation).Append('\n');
            }
            foreach (var g in p.Generators)
                sb.Append("G ").Append(g.Id).Append(' ').Append((int)g.Owner).Append(' ')
                  .Append(g.Cell.Q).Append(' ').Append(g.Cell.R).Append(' ').Append(g.Elevation).Append(' ').Append(g.CurrentHp).Append('\n');
            foreach (var b in p.Barracks)
            {
                sb.Append("B ");
                AppendStats(sb, b);
                sb.Append('\n');
            }
        }
```
New:
```csharp
        private static void WritePlayer(StringBuilder sb, PlayerState p)
        {
            sb.Append("PLAYER ").Append((int)p.Id).Append(' ').Append(p.Points)
              .Append(' ').Append(p.UnitsOnBoard.Count).Append(' ').Append(p.Generators.Count)
              .Append(' ').Append(p.Barracks.Count).Append('\n');

            foreach (var u in p.UnitsOnBoard)
            {
                sb.Append("U ").Append(u.Id).Append(' ').Append((int)u.Owner).Append(' ');
                AppendStats(sb, u.Stats);
                sb.Append(' ').Append(u.Cell.Q).Append(' ').Append(u.Cell.R).Append(' ').Append(u.Elevation)
                  .Append(' ').Append(CommandWire.EncodeName(u.Name)).Append('\n');
            }
            foreach (var g in p.Generators)
                sb.Append("G ").Append(g.Id).Append(' ').Append((int)g.Owner).Append(' ')
                  .Append(g.Cell.Q).Append(' ').Append(g.Cell.R).Append(' ').Append(g.Elevation).Append(' ').Append(g.CurrentHp).Append('\n');
            foreach (var b in p.Barracks)
            {
                sb.Append("B ");
                AppendStats(sb, b.Stats);
                sb.Append(' ').Append(CommandWire.EncodeName(b.Name));
                sb.Append('\n');
            }
        }
```

`ReadPlayer`, old:
```csharp
        private static PlayerState ReadPlayer(Func<string> next, PlayerId expected)
        {
            var head = next().Split(' ');           // PLAYER pid points unitCount genCount barracksCount
            int points = I(head[2]);
            int units = I(head[3]), gens = I(head[4]), barr = I(head[5]);

            var unitList = new List<Unit>(units);
            var genList = new List<Generator>(gens);
            var barracks = new List<UnitStats>(barr);

            for (int i = 0; i < units; i++)
            {
                var p = next().Split(' ');           // U id owner <9 stats> q r elev
                int id = I(p[1]); var owner = (PlayerId)I(p[2]);
                var stats = ReadStats(p, 3);
                unitList.Add(new Unit(id, owner, stats, new HexCoord(I(p[12]), I(p[13])), I(p[14])));
            }
            for (int i = 0; i < gens; i++)
            {
                var p = next().Split(' ');           // G id owner q r elev hp
                genList.Add(new Generator(I(p[1]), (PlayerId)I(p[2]), new HexCoord(I(p[3]), I(p[4])), I(p[5]), I(p[6])));
            }
            for (int i = 0; i < barr; i++)
                barracks.Add(ReadStats(next().Split(' '), 1)); // B <9 stats>

            return new PlayerState(expected, points, barracks, unitList, genList);
        }
```
New:
```csharp
        private static PlayerState ReadPlayer(Func<string> next, PlayerId expected)
        {
            var head = next().Split(' ');           // PLAYER pid points unitCount genCount barracksCount
            int points = I(head[2]);
            int units = I(head[3]), gens = I(head[4]), barr = I(head[5]);

            var unitList = new List<Unit>(units);
            var genList = new List<Generator>(gens);
            var barracks = new List<UnitTemplate>(barr);

            for (int i = 0; i < units; i++)
            {
                var p = next().Split(' ');           // U id owner <9 stats> q r elev [name]
                int id = I(p[1]); var owner = (PlayerId)I(p[2]);
                var stats = ReadStats(p, 3);
                string name = p.Length > 15 ? CommandWire.DecodeName(p[15]) : ""; // old payloads: no name token
                unitList.Add(new Unit(id, owner, stats, new HexCoord(I(p[12]), I(p[13])), I(p[14]), name));
            }
            for (int i = 0; i < gens; i++)
            {
                var p = next().Split(' ');           // G id owner q r elev hp
                genList.Add(new Generator(I(p[1]), (PlayerId)I(p[2]), new HexCoord(I(p[3]), I(p[4])), I(p[5]), I(p[6])));
            }
            for (int i = 0; i < barr; i++)
            {
                var p = next().Split(' ');            // B <9 stats> [name]
                var stats = ReadStats(p, 1);
                string name = p.Length > 10 ? CommandWire.DecodeName(p[10]) : ""; // old payloads: no name token
                barracks.Add(new UnitTemplate(name, stats));
            }

            return new PlayerState(expected, points, barracks, unitList, genList);
        }
```
(`AppendStats`/`ReadStats` helpers at the bottom of `ReplayFile.cs` are unchanged — only their call
sites above now pass `b.Stats` instead of `b`.)

`engine/HexWars.Engine/Rl/TacticalLayout.cs` — old:
```csharp
            // seed each side's barracks with the roster types so they can DEPLOY reinforcements from bounty
            var templates = new List<UnitStats>(RosterStats);
            var p0 = new PlayerState(PlayerId.Player0, 0, templates, u0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, templates, u1, null);
```
New:
```csharp
            // seed each side's barracks with the roster types so they can DEPLOY reinforcements from bounty
            var templates = new List<UnitTemplate>();
            foreach (var s in RosterStats) templates.Add(new UnitTemplate("", s));
            var p0 = new PlayerState(PlayerId.Player0, 0, templates, u0, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, templates, u1, null);
```
(`RosterStats` itself stays `IReadOnlyList<UnitStats>` — `TacticalCoding.RoleOf` compares raw stats
fields and is untouched, matching spec §5's RL surface note.)

`engine/HexWars.Engine/Net/GameSetup.cs` (`GameFactory.SeedPlayer`) — old (last lines of the method):
```csharp
            var units = new List<Unit>();
            for (int i = 0; i < army.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, army[i], flat[i], 0));
            // pre-seed the barracks with the default roster so players can deploy without designing first
            return new PlayerState(id, startingPoints, new List<UnitStats>(Roster), units, null);
        }
```
New (minimal compile-preserving shim — see conflict #1; Task 3 replaces this body entirely):
```csharp
            var units = new List<Unit>();
            for (int i = 0; i < army.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, army[i], flat[i], 0));
            // pre-seed the barracks with the default roster so players can deploy without designing first.
            // TODO(Task 3): replaced by the five-entry named starter set (GameFactory.StarterTemplates).
            var barracks = new List<UnitTemplate>();
            foreach (var s in Roster) barracks.Add(new UnitTemplate("", s));
            return new PlayerState(id, startingPoints, barracks, units, null);
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   281, Skipped:     0, Total:   281` (276 + 5 new: 3 CommandWireTests + 2 ReplayFileTests new tests; `RichStartState_RoundTrips` was modified, not added).

- [ ] **Step 6b: Presentation compile shim (one line)**

The Barracks type change breaks exactly one call site in Assets/ (verified by grep — `CreateUnit`'s
name parameter has a default, so `DesignPanel.cs:92` keeps compiling). In
`Assets/HexWars/Presentation/BarracksPanel.cs` line ~113, change:

```csharp
                var stats = p.Barracks[i];
```
to
```csharp
                var stats = p.Barracks[i].Stats;   // Task 10 rebuilds this row properly (name + delete)
```

Without this shim, the Unity compile check below fails and Tasks 6-9's compile gates would be
red through no fault of their own.

- [ ] **Step 7: Rebuild the Unity DLL and verify compilation**

```powershell
& "C:\Program Files\dotnet\dotnet.exe" build -c Release engine\HexWars.Engine\HexWars.Engine.csproj
Copy-Item engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll Assets\HexWars\Plugins\ -Force
```
Then coplay `check_compile_errors` — expect zero errors (Unity must not be in Safe Mode).

- [ ] **Step 8: Commit**

```bash
git add engine/HexWars.Engine/UnitTemplate.cs engine/HexWars.Engine/Unit.cs engine/HexWars.Engine/PlayerState.cs engine/HexWars.Engine/Command.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/LegalMoves.cs engine/HexWars.Engine/WinCheck.cs engine/HexWars.Engine/Net/CommandWire.cs engine/HexWars.Engine/ReplayFile.cs engine/HexWars.Engine/Rl/TacticalLayout.cs engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine.Tests/UnitTemplateTests.cs engine/HexWars.Engine.Tests/UnitTests.cs engine/HexWars.Engine.Tests/CommandWireTests.cs engine/HexWars.Engine.Tests/ReplayFileTests.cs engine/HexWars.Engine.Tests/LegalMovesTests.cs engine/HexWars.Engine.Tests/WinCheckTests.cs engine/HexWars.Engine.Tests/TerritoryDeployTests.csgit commit -m "feat(engine): UnitTemplate (name+stats) replaces bare UnitStats in the barracks - CreateUnit/DeployUnit/Unit carry names end to end, CommandWire+ReplayFile grow a trailing name token that old payloads parse unchanged"
```

---

### Task 2: DeleteTemplate command

**Files:**
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/Net/CommandWire.cs`
- Create: `engine/HexWars.Engine.Tests/DeleteTemplateTests.cs`
- Modify: `engine/HexWars.Engine.Tests/CommandWireTests.cs`
- Modify: `Assets/HexWars/Plugins/HexWars.Engine.dll` (rebuild + copy, final step)

**Interfaces:**
- Produces: `public sealed record DeleteTemplate(PlayerId Issuer, int TemplateIndex) : Command(Issuer);` (see conflict #3 for the base-call syntax correction).
- Produces: `CommandWire` wire letter `"X"` — `Write`: `$"X {(int)x.Issuer} {x.TemplateIndex}"`; `Read` case `"X"`: `new DeleteTemplate(issuer, I(p[2]))`. Letters already in use: `M A E C D N H B` — `X` is free.
- Consumes: `PlayerState.Barracks : IReadOnlyList<UnitTemplate>` (Task 1), `RejectionReason.TemplateNotFound` (existing), `ITurnPolicy`/`GameEngine.Apply`'s auto-end-turn guard (existing — Task 2 adds the `DeleteTemplate` exclusion described in conflict #4), `LegalMoves.For` (existing — Task 2 adds no case for `DeleteTemplate`, so it is never enumerated by construction).

- [ ] **Step 1: RED — `DeleteTemplate` doesn't exist yet**

Create `engine/HexWars.Engine.Tests/DeleteTemplateTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>DeleteTemplate: an administrative barracks edit, not a game move — free, doesn't touch
    /// the turn budget (not even under OneActionPolicy, which ends the turn after ANY other single
    /// action — see TurnPolicyTests.OneActionPolicy_AutoEndsTurn_AfterASingleAction), and is never
    /// offered by LegalMoves (keeps RL action masks untouched, per spec §5).</summary>
    public class DeleteTemplateTests
    {
        private static UnitTemplate T(string name, int cost) => new UnitTemplate(name, TestStates.Cost(cost));

        private static GameState TwoTemplates(int points = 12, GameConfig? cfg = null) =>
            new GameState(
                new Board(new[]
                {
                    new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                    new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) }),
                cfg ?? GameConfig.Default(),
                new[]
                {
                    new PlayerState(PlayerId.Player0, points, new[] { T("Alpha", 2), T("Beta", 3) }),
                    new PlayerState(PlayerId.Player1, points),
                },
                PlayerId.Player0, 1, 1);

        [Test]
        public void Delete_RemovesTemplateAtIndex_NonMutating()
        {
            var state = TwoTemplates();
            var r = GameEngine.Apply(state, new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            var barracks = r.NewState.Player(PlayerId.Player0).Barracks;
            Assert.That(barracks.Count, Is.EqualTo(1));
            Assert.That(barracks[0].Name, Is.EqualTo("Beta"));
            Assert.That(state.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(2), "original untouched");
        }

        [Test]
        public void Delete_ShiftsLaterIndices()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, 0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Barracks[0].Name, Is.EqualTo("Beta"));
        }

        [Test]
        public void Delete_Rejects_IndexTooLarge()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, 2));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TemplateNotFound));
        }

        [Test]
        public void Delete_Rejects_NegativeIndex()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, -1));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TemplateNotFound));
        }

        [Test]
        public void Delete_IsFree_DoesNotSpendPoints()
        {
            var r = GameEngine.Apply(TwoTemplates(points: 7), new DeleteTemplate(PlayerId.Player0, 0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(7));
        }

        [Test]
        public void Delete_DoesNotEndTurn_UnderOneActionPolicy()
        {
            var r = GameEngine.Apply(TwoTemplates(cfg: GameConfig.Default(turnPolicy: new OneActionPolicy())),
                new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0),
                "a CreateUnit alone ends a OneActionPolicy turn (see TurnPolicyTests) — DeleteTemplate must not");
        }

        [Test]
        public void Delete_DoesNotCountTowardKActionsPolicy_AfterBudgetSpent()
        {
            // Player0 has one board unit under K=1; simulate a state where the K=1 budget is already
            // spent (as if a move just happened, without an EndTurn) and confirm DeleteTemplate right
            // there does not itself force a further end-turn.
            var cfg = GameConfig.Default(turnPolicy: new KActionsPolicy(1));
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });
            var unit = new Unit(1, PlayerId.Player0, TestStates.Stats(health: 3, movement: 1), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 10, new[] { T("Alpha", 2) }, new[] { unit });
            var p1 = new PlayerState(PlayerId.Player1, 10);
            var spent = new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 2,
                movedUnitIds: new[] { 1 });

            var r = GameEngine.Apply(spent, new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0),
                "DeleteTemplate must not trigger KActionsPolicy's already-at-budget auto-end");
        }

        [Test]
        public void Delete_NeverEnumeratedByLegalMoves()
        {
            var moves = LegalMoves.For(TwoTemplates());
            Assert.That(moves.OfType<DeleteTemplate>(), Is.Empty);
        }
    }
}
```

Add one round-trip test to `engine/HexWars.Engine.Tests/CommandWireTests.cs` (after `BuildGenerator_RoundTrips`, before `Read_UnknownToken_Throws`):

```csharp
        [Test] public void DeleteTemplate_RoundTrips() => RoundTrips(new DeleteTemplate(P0, 1));
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build error — `error CS0246: The type or namespace name 'DeleteTemplate' could not be found`.

- [ ] **Step 2: GREEN — add the command, dispatch, handler, and the auto-end-turn exclusion**

`engine/HexWars.Engine/Command.cs` — old (the `CreateUnit` line, as Task 1 left it):
```csharp
    /// <summary>Design and pay for a unit; it goes to the issuer's reserve (off-board). Name is
    /// sanitized at the engine boundary (UnitTemplate.Sanitize); an empty/omitted name is legal — the
    /// dominant-role fallback happens on the deployed Unit's DisplayName, not here.</summary>
    public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats, string Name = "") : Command(Issuer);
```
New (adds `DeleteTemplate` right after it):
```csharp
    /// <summary>Design and pay for a unit; it goes to the issuer's reserve (off-board). Name is
    /// sanitized at the engine boundary (UnitTemplate.Sanitize); an empty/omitted name is legal — the
    /// dominant-role fallback happens on the deployed Unit's DisplayName, not here.</summary>
    public sealed record CreateUnit(PlayerId Issuer, UnitStats Stats, string Name = "") : Command(Issuer);

    /// <summary>Delete a barracks template by index — a free administrative edit, not a game move: no
    /// points, no turn action (see GameEngine.Apply's auto-end-turn guard), and never enumerated by
    /// LegalMoves (keeps RL action masks untouched).</summary>
    public sealed record DeleteTemplate(PlayerId Issuer, int TemplateIndex) : Command(Issuer);
```

`engine/HexWars.Engine/GameEngine.cs` — the `Dispatch` switch, old:
```csharp
        private static Result Dispatch(GameState state, Command command)
        {
            switch (command)
            {
                case CreateUnit c: return ApplyCreateUnit(state, c);
                case DeployGenerator c: return ApplyDeployGenerator(state, c);
                case DeployUnit c: return ApplyDeployUnit(state, c);
                case MoveUnit c: return ApplyMoveUnit(state, c);
                case AttackUnit c: return ApplyAttackUnit(state, c);
                case CaptureHex c: return ApplyCaptureHex(state, c);
                case BuildGenerator c: return ApplyBuildGenerator(state, c);
                case EndTurn c: return ApplyEndTurn(state, c);
                default: return Result.Reject(state, RejectionReason.None);
            }
        }
```
New:
```csharp
        private static Result Dispatch(GameState state, Command command)
        {
            switch (command)
            {
                case CreateUnit c: return ApplyCreateUnit(state, c);
                case DeleteTemplate c: return ApplyDeleteTemplate(state, c);
                case DeployGenerator c: return ApplyDeployGenerator(state, c);
                case DeployUnit c: return ApplyDeployUnit(state, c);
                case MoveUnit c: return ApplyMoveUnit(state, c);
                case AttackUnit c: return ApplyAttackUnit(state, c);
                case CaptureHex c: return ApplyCaptureHex(state, c);
                case BuildGenerator c: return ApplyBuildGenerator(state, c);
                case EndTurn c: return ApplyEndTurn(state, c);
                default: return Result.Reject(state, RejectionReason.None);
            }
        }
```

The auto-end-turn guard in `Apply`, old:
```csharp
            // One-action turn policies auto-end the turn after a single non-EndTurn action.
            if (!newState.IsGameOver && !(command is EndTurn)
                && (newState.Config.TurnPolicy.AutoEndTurnAfter(command, newState)
                    || (newState.Config.TerritoryMode && newState.Config.ClaimEndsTurn && command is CaptureHex)))
            {
                newState = Finalize(ApplyEndTurn(newState, new EndTurn(command.Issuer)).NewState);
            }
```
New:
```csharp
            // One-action turn policies auto-end the turn after a single non-EndTurn action.
            // DeleteTemplate is an administrative barracks edit, not a game move — it must never
            // consume a turn action or trigger an auto-end (see DeleteTemplateTests).
            if (!newState.IsGameOver && !(command is EndTurn) && !(command is DeleteTemplate)
                && (newState.Config.TurnPolicy.AutoEndTurnAfter(command, newState)
                    || (newState.Config.TerritoryMode && newState.Config.ClaimEndsTurn && command is CaptureHex)))
            {
                newState = Finalize(ApplyEndTurn(newState, new EndTurn(command.Issuer)).NewState);
            }
```

Add `ApplyDeleteTemplate` immediately after `ApplyCreateUnit` (i.e. insert before the
`private static Result ApplyDeployGenerator(GameState state, DeployGenerator c)` method):
```csharp
        private static Result ApplyDeleteTemplate(GameState state, DeleteTemplate c)
        {
            var player = state.Player(c.Issuer);
            if (c.TemplateIndex < 0 || c.TemplateIndex >= player.Barracks.Count)
                return Result.Reject(state, RejectionReason.TemplateNotFound);

            var barracks = new List<UnitTemplate>(player.Barracks);
            barracks.RemoveAt(c.TemplateIndex);
            var updated = new PlayerState(player.Id, player.Points, barracks,
                                          player.UnitsOnBoard, player.Generators, player.DestroyedValue);
            return Result.Ok(WithPlayer(state, updated));
        }
```

`engine/HexWars.Engine/Net/CommandWire.cs` — `Write`, old:
```csharp
                case CreateUnit cu:     return $"C {(int)cu.Issuer} {WriteStats(cu.Stats)} {EncodeName(cu.Name)}";
                case DeployUnit d:      return $"D {(int)d.Issuer} {d.TemplateIndex} {d.Cell.Q} {d.Cell.R}";
```
New:
```csharp
                case CreateUnit cu:     return $"C {(int)cu.Issuer} {WriteStats(cu.Stats)} {EncodeName(cu.Name)}";
                case DeleteTemplate x:  return $"X {(int)x.Issuer} {x.TemplateIndex}";
                case DeployUnit d:      return $"D {(int)d.Issuer} {d.TemplateIndex} {d.Cell.Q} {d.Cell.R}";
```
`Read`, old:
```csharp
                case "C": return new CreateUnit(issuer, ReadStats(p, 2), p.Length > 11 ? DecodeName(p[11]) : "");
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
```
New:
```csharp
                case "C": return new CreateUnit(issuer, ReadStats(p, 2), p.Length > 11 ? DecodeName(p[11]) : "");
                case "X": return new DeleteTemplate(issuer, I(p[2]));
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   290, Skipped:     0, Total:   290` (281 + 9 new: 8 DeleteTemplateTests + 1 CommandWireTests).

- [ ] **Step 3: Rebuild the Unity DLL and verify compilation**

```powershell
& "C:\Program Files\dotnet\dotnet.exe" build -c Release engine\HexWars.Engine\HexWars.Engine.csproj
Copy-Item engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll Assets\HexWars\Plugins\ -Force
```
Then coplay `check_compile_errors` — expect zero errors.

- [ ] **Step 4: Commit**

```bash
git add engine/HexWars.Engine/Command.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/Net/CommandWire.cs engine/HexWars.Engine.Tests/DeleteTemplateTests.cs engine/HexWars.Engine.Tests/CommandWireTests.csgit commit -m "feat(engine): DeleteTemplate command - free barracks edit, excluded from turn-policy auto-end and never enumerated by LegalMoves so RL action masks stay untouched"
```

---

### Task 3: Starter seeding

**Files:**
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs`
- Create: `engine/HexWars.Engine.Tests/GameFactoryStarterTemplatesTests.cs`
- Modify: `engine/HexWars.Engine.Tests/ReplayFileTests.cs`
- Modify: `Assets/HexWars/Plugins/HexWars.Engine.dll` (rebuild + copy, final step)

**Interfaces:**
- Produces: `GameFactory` seeds every player's `PlayerState.Barracks` (both `GameMode.Annihilation` and `GameMode.Territory`) with five `UnitTemplate`s named exactly `"Brute"`, `"Striker"`, `"Sniper"`, `"Artillery"`, `"Scout"` in that index order (0-4), stats verbatim from spec §5's table.
- Consumes: `UnitTemplate` (Task 1), `DeleteTemplate`/`GameEngine.Apply` (Task 2 — used by the "deletion doesn't reseed" test), `GameFactory.Roster` (existing, unchanged — still drives on-board starting-army composition via `BuildArmy`, kept fully separate from the new barracks table so `ArmyCompositionTests` and self-play balance are unaffected).

- [ ] **Step 1: RED — GameFactory still only seeds three unnamed templates (Task 1's shim)**

Create `engine/HexWars.Engine.Tests/GameFactoryStarterTemplatesTests.cs`:

```csharp
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Every game's barracks starts pre-loaded with five named example designs (spec §5):
    /// deployable turn one, deletable per-game, teaching that a statline is a concept. The first three
    /// match the existing starting-army roster exactly (RL contract: TacticalLayout keeps its own
    /// roster, so this only affects human/AI matches built through GameFactory).</summary>
    public class GameFactoryStarterTemplatesTests
    {
        private static readonly (string Name, UnitStats Stats)[] Expected =
        {
            ("Brute",     new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1)),
            ("Striker",   new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1)),
            ("Sniper",    new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1)),
            ("Artillery", new UnitStats(3, 6, 0, 0, 0, 5, 2, 2, 1)),
            ("Scout",     new UnitStats(2, 0, 0, 4, 3, 0, 0, 7, 2)),
        };

        [Test]
        public void Build_Annihilation_SeedsFiveNamedTemplates_BothPlayers()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
                Assert.That(state.Player(pid).Barracks.Count, Is.EqualTo(5));
        }

        [Test]
        public void Build_Territory_SeedsFiveNamedTemplates_BothPlayers()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7));
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
                Assert.That(state.Player(pid).Barracks.Count, Is.EqualTo(5));
        }

        [Test]
        public void Build_StarterTemplates_MatchSpecTable_Exactly()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var barracks = state.Player(PlayerId.Player0).Barracks;
            for (int i = 0; i < Expected.Length; i++)
            {
                Assert.That(barracks[i].Name, Is.EqualTo(Expected[i].Name), $"slot {i} name");
                Assert.That(barracks[i].Stats, Is.EqualTo(Expected[i].Stats), $"slot {i} stats");
            }
        }

        [Test]
        public void Build_ClassicTrio_AtIndicesZeroToTwo_MatchesOnBoardRoster()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7,
                armySize: 3, brutes: 1, strikers: 1, snipers: 1));
            var barracks = state.Player(PlayerId.Player0).Barracks;
            var onBoard = state.Player(PlayerId.Player0).UnitsOnBoard.OrderBy(u => u.Id).ToArray();

            Assert.That(barracks[0].Stats, Is.EqualTo(onBoard[0].Stats), "Brute matches the on-board Brute");
            Assert.That(barracks[1].Stats, Is.EqualTo(onBoard[1].Stats), "Striker matches the on-board Striker");
            Assert.That(barracks[2].Stats, Is.EqualTo(onBoard[2].Stats), "Sniper matches the on-board Sniper");
        }

        [Test]
        public void DeleteTemplate_DuringGame_DoesNotReseed()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var afterDelete = GameEngine.Apply(state, new DeleteTemplate(PlayerId.Player0, 4)).NewState; // remove Scout
            Assert.That(afterDelete.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(4));

            var afterRound = GameEngine.Apply(
                GameEngine.Apply(afterDelete, new EndTurn(PlayerId.Player0)).NewState,
                new EndTurn(PlayerId.Player1)).NewState;
            Assert.That(afterRound.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(4),
                "a full round must not re-seed the deleted template back in");
        }
    }
}
```

Add one round-trip test to `engine/HexWars.Engine.Tests/ReplayFileTests.cs` (after `RichStartState_RoundTrips`,
before the class's closing brace):

```csharp

        [Test]
        public void SeededStartState_RoundTrips_WithNames()
        {
            var start = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;

            var expected = new[] { "Brute", "Striker", "Sniper", "Artillery", "Scout" };
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                var barracks = s.Player(pid).Barracks;
                Assert.That(barracks.Count, Is.EqualTo(5));
                for (int i = 0; i < 5; i++)
                    Assert.That(barracks[i].Name, Is.EqualTo(expected[i]), $"{pid} slot {i}");
            }
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: assertion failures (not build failures — `UnitTemplate`/`Barracks` already compile from
Task 1) — e.g. `Build_Annihilation_SeedsFiveNamedTemplates_BothPlayers` fails with
`Expected: 5 But was: 3` (Task 1's shim still only seeds the three unnamed `Roster` entries).

- [ ] **Step 2: GREEN — GameFactory seeds the five named starter templates**

`engine/HexWars.Engine/Net/GameSetup.cs` — old (as Task 1 left it: the `Roster` field is unchanged;
`SeedPlayer`'s tail is Task 1's shim):
```csharp
        static readonly UnitStats[] Roster =
        {
            new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1), // Brute
            new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1), // Striker
            new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1), // Sniper
        };
```
New (adds `StarterTemplates` right after `Roster`, `Roster` itself untouched):
```csharp
        static readonly UnitStats[] Roster =
        {
            new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1), // Brute
            new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1), // Striker
            new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1), // Sniper
        };

        // Barracks starter set (invite-readiness spec §5): five named example designs, deployable turn
        // one, deletable per-game. The first three intentionally match Roster[0..2] exactly — the
        // classic trio stays at indices 0-2, matching the on-board starting army and the RL contract
        // (TacticalLayout keeps its own separate roster, so this table never touches the RL surface).
        static readonly UnitTemplate[] StarterTemplates =
        {
            new UnitTemplate("Brute",     Roster[0]),
            new UnitTemplate("Striker",   Roster[1]),
            new UnitTemplate("Sniper",    Roster[2]),
            new UnitTemplate("Artillery", new UnitStats(3, 6, 0, 0, 0, 5, 2, 2, 1)),
            new UnitTemplate("Scout",     new UnitStats(2, 0, 0, 4, 3, 0, 0, 7, 2)),
        };
```

`SeedPlayer`, old (Task 1's shim tail):
```csharp
            var units = new List<Unit>();
            for (int i = 0; i < army.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, army[i], flat[i], 0));
            // pre-seed the barracks with the default roster so players can deploy without designing first.
            // TODO(Task 3): replaced by the five-entry named starter set (GameFactory.StarterTemplates).
            var barracks = new List<UnitTemplate>();
            foreach (var s in Roster) barracks.Add(new UnitTemplate("", s));
            return new PlayerState(id, startingPoints, barracks, units, null);
        }
```
New:
```csharp
            var units = new List<Unit>();
            for (int i = 0; i < army.Length && i < flat.Count; i++)
                units.Add(new Unit(nextId++, id, army[i], flat[i], 0));
            // pre-seed the barracks with the named starter set so players can deploy without designing first
            return new PlayerState(id, startingPoints, new List<UnitTemplate>(StarterTemplates), units, null);
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   296, Skipped:     0, Total:   296` (290 + 6 new: 5 GameFactoryStarterTemplatesTests + 1 ReplayFileTests).

- [ ] **Step 3: Rebuild the Unity DLL and verify compilation**

```powershell
& "C:\Program Files\dotnet\dotnet.exe" build -c Release engine\HexWars.Engine\HexWars.Engine.csproj
Copy-Item engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll Assets\HexWars\Plugins\ -Force
```
Then coplay `check_compile_errors` — expect zero errors.

- [ ] **Step 4: Commit**

```bash
git add engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine.Tests/GameFactoryStarterTemplatesTests.cs engine/HexWars.Engine.Tests/ReplayFileTests.csgit commit -m "feat(engine): GameFactory seeds every barracks with the five named starter templates (Brute/Striker/Sniper/Artillery/Scout) - classic trio stays at indices 0-2 for the RL contract"
```

---

### Task 4: Token rejoin + room hold

**Files:**
- Modify: `engine/HexWars.Engine/Net/MatchHub.cs`
- Modify: `engine/HexWars.Engine/Net/GameSession.cs` (parameter rename only — see conflict-adjacent note below; no signature/type change)
- Modify: `engine/HexWars.NetServer/Program.cs`
- Modify: `engine/HexWars.NetServer/SelfTest.cs`
- Create: `engine/HexWars.Engine.Tests/TokenRejoinTests.cs`
- Modify: `engine/HexWars.Engine.Tests/NetDisconnectTests.cs` (rewrites 2 tests — see conflict #2)
- Modify: `Assets/HexWars/Plugins/HexWars.Engine.dll` (rebuild + copy, final step)

**Interfaces:**
- Produces: `public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default, bool isPrivate = false, bool joinOnly = false, string? token = null)` on `MatchHub` — `token` defaults to `connectionId` when `null`, so every existing call site (which never passes `token`) keeps compiling and behaving identically for un-started rooms.
- Consumes: `GameSession.Join(string)` / `.Leave(string)` / `.Submit(string, Command)` (existing signatures, parameter renamed `connectionId` → `token` for clarity — no call-site break, confirmed no test uses a named `connectionId:` argument), `ReplayFile.Write` / `NetProtocol.Start` / `.Seat` / `.SeatFull` (existing, unchanged).
- Produces (room-internal, not exposed outside `MatchHub`): `Room.ConnToToken : Dictionary<string,string>`, `Room.EmptySinceTicks : long?`.

**Design note on why `Disconnect` no longer frees a seat (ties conflict #2 to the implementation):**
Today, `MatchHub.Disconnect` calls `room.Session.Leave(connectionId)`, which frees that seat for the
*next* connection to claim, whoever it is. Under token-keyed seats, doing that on every disconnect would
defeat the entire feature: if player B's socket ever drops and a stranger with a different token
connects to the room code first, the stranger would take B's seat before B can reconnect. So
`MatchHub.Disconnect` now only manages `Room.Members`/`Room.ConnToToken` bookkeeping and the
held-room-expiry clock — it never calls `Session.Leave`. `GameSession.Leave` itself is untouched (still
directly tested by `NetDisconnectTests.Session_Leave_FreesTheSeatForReuse`, which calls it in isolation);
`MatchHub` simply stops calling it. A started room's two seats are therefore permanently reserved to
their two tokens for the life of that `GameSession` — reclaimed only by presenting the same token, never
freed to a third party. An un-started room (nobody ever dealt START) still deletes itself the instant it
empties (unchanged from today), so there is no seat-reservation concern there at all.

- [ ] **Step 1: RED — `Connect` doesn't accept a `token` yet, and today's disconnect/reset tests assert the wrong contract**

Create `engine/HexWars.Engine.Tests/TokenRejoinTests.cs`:

```csharp
using System;
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Seats are keyed by a client-minted token, not the per-socket connection id, so a
    /// refresh/background/reconnect reclaims the same seat. A started room that drops to zero live
    /// connections is HELD (not deleted) for a 10-minute window — both players can blip through a
    /// network drop without losing the match; a stranger with a different token still can't steal a
    /// reserved seat. Un-started rooms (never dealt START) keep instant cleanup.</summary>
    public class TokenRejoinTests
    {
        private long _now;
        private MatchHub NewHub()
        {
            _now = TimeSpan.FromHours(1).Ticks;
            return new MatchHub(_ => TwoUnitGame(), () => _now);
        }

        private static GameState TwoUnitGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++) tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var u0 = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, stats, new HexCoord(4, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 10, null, new[] { u0 }, null);
            var p1 = new PlayerState(PlayerId.Player1, 10, null, new[] { u1 }, null);
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, 3);
        }

        [Test]
        public void SameToken_NewConnection_ReclaimsSeat_AndReceivesStart_MidGame()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");   // P0
            hub.Connect("R", "conn-b", token: "tok-b");   // P1 -> room Started

            hub.Disconnect("R", "conn-a");                // a's socket dies (tab backgrounded)
            var back = hub.Connect("R", "conn-a-2", token: "tok-a"); // a reconnects on a NEW socket, SAME token

            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message == "SEAT 0"),
                "the same token reclaims seat P0, not whatever seat is next free");
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message.StartsWith("START ")),
                "a reconnect into a started room gets a personal START re-deal");
        }

        [Test]
        public void DifferentToken_GetsSeatFull_WhenBothSeatsAreClaimed()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");                // a's socket drops, seat stays reserved to tok-a

            var stranger = hub.Connect("R", "conn-c", token: "tok-c");
            Assert.That(stranger, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-c" && o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void HeldRoom_ReconnectWithinTenMinutes_StillReclaimsSeat()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");                // room empty, started -> held, not removed

            _now += TimeSpan.FromMinutes(9).Ticks;
            var back = hub.Connect("R", "conn-a-2", token: "tok-a");
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message == "SEAT 0"),
                "9 minutes in, the hold window hasn't expired");
        }

        [Test]
        public void HeldRoom_ExpiresAfterTenMinutes_BecomesAFreshRoom()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");

            _now += TimeSpan.FromMinutes(11).Ticks;
            var fresh = hub.Connect("R", "conn-d", token: "tok-d");
            Assert.That(fresh, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-d" && o.Message == "SEAT 0"),
                "11 minutes in, the held room expired and a brand-new game was minted");
        }

        [Test]
        public void UnstartedRoom_StillCleansUpInstantly_NotHeld()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");    // only one seat ever taken — never Started
            hub.Disconnect("R", "conn-a");

            // no time advance at all: if the room were held like a started one, tok-a would still own P0
            var next = hub.Connect("R", "conn-e", token: "tok-e");
            Assert.That(next, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-e" && o.Message == "SEAT 0"),
                "an un-started room resets instantly, same as before this feature");
        }

        [Test]
        public void OpenGames_NeverLists_AHeldEmptyRoom()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");                // held, empty, started

            Assert.That(hub.OpenGames(), Is.Empty, "a held room is not a waiting host — never browsable");
        }
    }
}
```

Now rewrite the two conflicting tests in `engine/HexWars.Engine.Tests/NetDisconnectTests.cs` (conflict
#2). Old (the two `[Test]` methods, quoted from the real current file):
```csharp
        [Test]
        public void Hub_Disconnect_FreesSeat_SoANewJoinerIsSeatedNotTurnedAway()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a"); // P0
            hub.Connect("r", "b"); // P1 — room full
            hub.Disconnect("r", "a");
            var c = hub.Connect("r", "c");
            Assert.That(c, Has.Some.Matches<Outbound>(o => o.ConnectionId == "c" && o.Message == "SEAT 0"));
            Assert.That(c, Has.None.Matches<Outbound>(o => o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void Hub_Disconnect_LastMember_ResetsRoomForAFreshGame()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            hub.Disconnect("r", "a");
            hub.Disconnect("r", "b"); // room now empty
            var d = hub.Connect("r", "d");
            Assert.That(d, Has.Some.Matches<Outbound>(o => o.ConnectionId == "d" && o.Message == "SEAT 0"));
        }
```
New (same two slots, rewritten to the token-keyed/held-room contract — the old behavior these asserted
is exactly what Task 4 replaces; the new coverage of "same token reclaims" and "held window" lives in
`TokenRejoinTests` above):
```csharp
        [Test]
        public void Hub_Disconnect_OneOfTwo_SeatStaysReservedToItsToken_NotFreedForAnyComer()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a"); // P0, token defaults to "a"
            hub.Connect("r", "b"); // P1 — room full, Started
            hub.Disconnect("r", "a");
            var c = hub.Connect("r", "c"); // a stranger, a different token — a's seat is reserved, not freed
            Assert.That(c, Has.Some.Matches<Outbound>(o => o.ConnectionId == "c" && o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void Hub_Disconnect_LastMember_OfAStartedRoom_IsHeldNotReset()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            hub.Disconnect("r", "a");
            hub.Disconnect("r", "b"); // room now empty, but it Started — held, not reset
            var d = hub.Connect("r", "d"); // a stranger's token, no time advance
            Assert.That(d, Has.Some.Matches<Outbound>(o => o.ConnectionId == "d" && o.Message == NetProtocol.SeatFull),
                "a started room's seats survive both players dropping — reconnect must use the same token (see TokenRejoinTests)");
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build error — `error CS1739: 'MatchHub.Connect(string, string, GameSetup, bool, bool)'
does not have a parameter named 'token'` (the six `token:` call sites in `TokenRejoinTests.cs` don't
compile against today's `Connect` signature).

- [ ] **Step 2: GREEN — token-keyed seats + held-room sweep**

`engine/HexWars.Engine/Net/GameSession.cs` — old (the class doc comment + `Join`/`Leave`/`Submit`, quoted
from the real current file; `SubmitStatus`/`SubmitOutcome` above them are unchanged and omitted here):
```csharp
    /// <summary>
    /// One authoritative head-to-head match, independent of any transport. It seats two connections
    /// (P0 then P1), and on every command enforces the one rule the engine can't: a connection may only
    /// issue as the seat it actually holds (anti-impersonation). Everything else — turn order, legality,
    /// win/elimination — is delegated to <see cref="GameEngine.Apply"/>, the single source of truth.
    /// A WebSocket layer just turns sockets into <see cref="Join"/>/<see cref="Submit"/> calls and
    /// relays the accepted command (via <see cref="CommandWire"/>) to both seats.
    /// </summary>
    public sealed class GameSession
    {
        private readonly Dictionary<string, PlayerId> _seats = new Dictionary<string, PlayerId>();

        public GameState State { get; private set; }

        public GameSession(GameState start) { State = start; }

        /// <summary>Seat a connection as P0, then P1; a returning connection keeps its seat; null once full.</summary>
        public PlayerId? Join(string connectionId)
        {
            if (_seats.TryGetValue(connectionId, out var existing)) return existing;
            if (!_seats.ContainsValue(PlayerId.Player0)) return _seats[connectionId] = PlayerId.Player0;
            if (!_seats.ContainsValue(PlayerId.Player1)) return _seats[connectionId] = PlayerId.Player1;
            return null;
        }

        /// <summary>Release a connection's seat (on disconnect) so it can be re-taken by a reconnect.</summary>
        public void Leave(string connectionId) => _seats.Remove(connectionId);

        /// <summary>Validate the issuer owns its seat, then apply through the engine. On Accepted, advances State.</summary>
        public SubmitOutcome Submit(string connectionId, Command cmd)
        {
            if (!_seats.TryGetValue(connectionId, out var seat)) return SubmitOutcome.NoSeat();
            if (cmd.Issuer != seat) return SubmitOutcome.WrongSeat();

            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) return SubmitOutcome.Rejected(result.Reason);

            State = result.NewState;
            return SubmitOutcome.Accepted(State);
        }
    }
```
New (parameter rename only — `connectionId` → `token`; no signature/type/behavior change to `Join`,
`Leave`, or `Submit` themselves, only their doc comments and the class doc comment, so no call site
anywhere breaks):
```csharp
    /// <summary>
    /// One authoritative head-to-head match, independent of any transport. It seats two IDENTITIES
    /// (P0 then P1) — historically a raw socket connection id, now a client-minted <c>token</c> that
    /// survives a refresh/reconnect — and on every command enforces the one rule the engine can't: an
    /// identity may only issue as the seat it actually holds (anti-impersonation). Everything else —
    /// turn order, legality, win/elimination — is delegated to <see cref="GameEngine.Apply"/>, the
    /// single source of truth. <see cref="MatchHub"/> maps each live connection to a token and calls
    /// <see cref="Join"/>/<see cref="Submit"/> with that token, so a dropped-and-reconnected socket with
    /// the same token reclaims the same seat.
    /// </summary>
    public sealed class GameSession
    {
        private readonly Dictionary<string, PlayerId> _seats = new Dictionary<string, PlayerId>();

        public GameState State { get; private set; }

        public GameSession(GameState start) { State = start; }

        /// <summary>Seat a token as P0, then P1; a returning token keeps its seat; null once full.</summary>
        public PlayerId? Join(string token)
        {
            if (_seats.TryGetValue(token, out var existing)) return existing;
            if (!_seats.ContainsValue(PlayerId.Player0)) return _seats[token] = PlayerId.Player0;
            if (!_seats.ContainsValue(PlayerId.Player1)) return _seats[token] = PlayerId.Player1;
            return null;
        }

        /// <summary>Release a token's seat so it can be re-taken (by any token) on the next Join. Used
        /// only for un-started/lobby rooms — MatchHub never calls this for a Started room, since a
        /// started room's seats must survive both players' sockets dropping (see MatchHub.Disconnect).</summary>
        public void Leave(string token) => _seats.Remove(token);

        /// <summary>Validate the issuer owns its seat, then apply through the engine. On Accepted, advances State.</summary>
        public SubmitOutcome Submit(string token, Command cmd)
        {
            if (!_seats.TryGetValue(token, out var seat)) return SubmitOutcome.NoSeat();
            if (cmd.Issuer != seat) return SubmitOutcome.WrongSeat();

            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) return SubmitOutcome.Rejected(result.Reason);

            State = result.NewState;
            return SubmitOutcome.Accepted(State);
        }
    }
```

`engine/HexWars.Engine/Net/MatchHub.cs` — full file, old (as read from the repo):
```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>A message the transport should deliver to one connection.</summary>
    public readonly struct Outbound
    {
        public readonly string ConnectionId;
        public readonly string Message;
        public Outbound(string connectionId, string message) { ConnectionId = connectionId; Message = message; }
    }

    /// <summary>One joinable lobby entry: a public room with a host waiting and a game not yet begun.</summary>
    public readonly struct OpenGame
    {
        public readonly string Code;
        public readonly GameSetup Setup;
        public readonly int AgeSeconds;
        public OpenGame(string code, GameSetup setup, int ageSeconds) { Code = code; Setup = setup; AgeSeconds = ageSeconds; }
    }

    /// <summary>
    /// Routes connections into rooms and drives each room's <see cref="GameSession"/>. Pure logic: every
    /// method returns the <see cref="Outbound"/> messages a transport should send, so the entire server
    /// brain is unit-testable without a socket. A new room mints a fresh game via the injected factory;
    /// once both seats are filled it deals the start state to both; thereafter validated commands are
    /// broadcast to everyone and rejections go back to just the issuer.
    /// </summary>
    public sealed class MatchHub
    {
        private sealed class Room
        {
            public readonly GameSession Session;
            public readonly List<string> Members = new List<string>(); // seated connections, broadcast targets
            public readonly GameSetup Setup;       // the host's picks — shown in the lobby browser
            public readonly bool IsPrivate;        // private rooms are joinable by code/link only
            public readonly long CreatedAtTicks;
            public bool Started;                   // set when the start state is dealt; never cleared —
                                                   // a started room that drops to one member must not re-list
            public Room(GameState start, GameSetup setup, bool isPrivate, long nowTicks)
            { Session = new GameSession(start); Setup = setup; IsPrivate = isPrivate; CreatedAtTicks = nowTicks; }
        }

        private readonly Func<GameSetup, GameState> _newGame;
        private readonly Func<long> _now;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        /// <summary>The clock is injectable so lobby ages are exactly testable; production uses UTC.</summary>
        public MatchHub(Func<GameSetup, GameState> newGame, Func<long>? utcNowTicks = null)
        { _newGame = newGame; _now = utcNowTicks ?? (() => DateTime.UtcNow.Ticks); }

        /// <summary>Seat a connection. The first connection to a room creates it from <paramref name="setup"/>
        /// (the host's lobby picks); later joiners' setups are ignored — they join the host's game.
        /// <paramref name="joinOnly"/> marks a joiner (link/code/browser row, never a host): a joiner must
        /// never mint a room, since a typo'd code would otherwise create a phantom public game and strand
        /// the joiner with a seat in it. Missing room + joinOnly turns the connection away instead.</summary>
        public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default, bool isPrivate = false, bool joinOnly = false)
        {
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room))
            {
                if (joinOnly)
                {
                    outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                    return outs;
                }
                room = new Room(_newGame(setup), setup.Sanitized(), isPrivate, _now());
                _rooms[roomCode] = room;
            }

            var seat = room.Session.Join(connectionId);
            if (seat == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                return outs;
            }

            bool added = false;
            if (!room.Members.Contains(connectionId)) { room.Members.Add(connectionId); added = true; }
            outs.Add(new Outbound(connectionId, NetProtocol.Seat(seat.Value)));

            // The moment the second distinct player takes a seat, deal the authoritative start state to both.
            if (added && room.Members.Count == 2)
            {
                room.Started = true;
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                foreach (var m in room.Members) outs.Add(new Outbound(m, startMsg));
            }
            return outs;
        }

        /// <summary>The lobby browser's view: public rooms with a waiting host and no game started,
        /// newest first. Callers own thread-safety (the transport already serializes hub access).</summary>
        public IReadOnlyList<OpenGame> OpenGames()
        {
            var list = new List<OpenGame>();
            foreach (var kv in _rooms)
            {
                var r = kv.Value;
                if (r.IsPrivate || r.Started || r.Members.Count != 1) continue;
                int age = (int)((_now() - r.CreatedAtTicks) / TimeSpan.TicksPerSecond);
                list.Add(new OpenGame(kv.Key, r.Setup, age));
            }
            list.Sort((a, b) => a.AgeSeconds.CompareTo(b.AgeSeconds)); // newest (smallest age) first
            return list;
        }

        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            var cmd = CommandWire.Read(msg.Payload);
            var outcome = room.Session.Submit(connectionId, cmd);
            switch (outcome.Status)
            {
                case SubmitStatus.Accepted:
                    string applyMsg = NetProtocol.Apply(cmd);
                    foreach (var m in room.Members) outs.Add(new Outbound(m, applyMsg));
                    break;
                case SubmitStatus.Rejected:
                    outs.Add(new Outbound(connectionId, NetProtocol.Reject(outcome.Reason)));
                    break;
                default: // NoSeat / WrongSeat — tell only the offender, never touch the game
                    outs.Add(new Outbound(connectionId, "REJECT " + outcome.Status));
                    break;
            }
            return outs;
        }

        /// <summary>Free a dropped connection's seat; once the room is empty, reset it so the next pair gets a fresh game.</summary>
        public IReadOnlyList<Outbound> Disconnect(string roomCode, string connectionId)
        {
            if (_rooms.TryGetValue(roomCode, out var room))
            {
                room.Session.Leave(connectionId);
                room.Members.Remove(connectionId);
                if (room.Members.Count == 0) _rooms.Remove(roomCode);
            }
            return Array.Empty<Outbound>();
        }
    }
}
```
New (full file — token-keyed seats, per-room `ConnToToken` map, held-room sweep):
```csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>A message the transport should deliver to one connection.</summary>
    public readonly struct Outbound
    {
        public readonly string ConnectionId;
        public readonly string Message;
        public Outbound(string connectionId, string message) { ConnectionId = connectionId; Message = message; }
    }

    /// <summary>One joinable lobby entry: a public room with a host waiting and a game not yet begun.</summary>
    public readonly struct OpenGame
    {
        public readonly string Code;
        public readonly GameSetup Setup;
        public readonly int AgeSeconds;
        public OpenGame(string code, GameSetup setup, int ageSeconds) { Code = code; Setup = setup; AgeSeconds = ageSeconds; }
    }

    /// <summary>
    /// Routes connections into rooms and drives each room's <see cref="GameSession"/>. Pure logic: every
    /// method returns the <see cref="Outbound"/> messages a transport should send, so the entire server
    /// brain is unit-testable without a socket. A new room mints a fresh game via the injected factory;
    /// once both seats are filled it deals the start state to both; thereafter validated commands are
    /// broadcast to everyone and rejections go back to just the issuer.
    ///
    /// Seats are keyed by a client-minted TOKEN, not the transport connection id (a token survives a
    /// refresh/reconnect; a connection id does not). Each room keeps its own connection→token map so a
    /// dropped-and-reconnected socket with the same token reclaims the same <see cref="GameSession"/>
    /// seat. A room that has Started and drops to zero live connections is HELD (its Session — and so
    /// its token→seat assignments — kept alive) for <see cref="HoldWindowTicks"/> instead of being
    /// deleted, so both players can survive a simultaneous drop; an un-started room (never dealt START)
    /// still cleans up the instant it empties, as before this feature.
    /// </summary>
    public sealed class MatchHub
    {
        /// <summary>How long a Started room with zero live connections is kept alive for a reconnect.</summary>
        private static readonly long HoldWindowTicks = TimeSpan.FromMinutes(10).Ticks;

        private sealed class Room
        {
            public readonly GameSession Session;
            public readonly List<string> Members = new List<string>(); // seated connections, broadcast targets
            public readonly Dictionary<string, string> ConnToToken = new Dictionary<string, string>();
            public readonly GameSetup Setup;       // the host's picks — shown in the lobby browser
            public readonly bool IsPrivate;        // private rooms are joinable by code/link only
            public readonly long CreatedAtTicks;
            public bool Started;                   // set when the start state is dealt; never cleared —
                                                   // a started room that drops to one member must not re-list
            public long? EmptySinceTicks;          // set when a Started room's Members hits zero; null while
                                                   // occupied or un-started — the held-room expiry clock
            public Room(GameState start, GameSetup setup, bool isPrivate, long nowTicks)
            { Session = new GameSession(start); Setup = setup; IsPrivate = isPrivate; CreatedAtTicks = nowTicks; }
        }

        private readonly Func<GameSetup, GameState> _newGame;
        private readonly Func<long> _now;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        /// <summary>The clock is injectable so lobby ages (and hold-window expiry) are exactly testable;
        /// production uses UTC.</summary>
        public MatchHub(Func<GameSetup, GameState> newGame, Func<long>? utcNowTicks = null)
        { _newGame = newGame; _now = utcNowTicks ?? (() => DateTime.UtcNow.Ticks); }

        /// <summary>Drop any held room whose hold window has elapsed. Cheap (a linear scan of rooms);
        /// called opportunistically at the top of every public method — there are no timers in this
        /// pure, transport-agnostic hub.</summary>
        private void Sweep()
        {
            if (_rooms.Count == 0) return;
            long now = _now();
            List<string>? expired = null;
            foreach (var kv in _rooms)
                if (kv.Value.EmptySinceTicks is long since && now - since > HoldWindowTicks)
                    (expired ??= new List<string>()).Add(kv.Key);
            if (expired != null)
                foreach (var code in expired) _rooms.Remove(code);
        }

        /// <summary>Seat a connection under its identity token. The first connection to a room creates it
        /// from <paramref name="setup"/> (the host's lobby picks); later joiners' setups are ignored —
        /// they join the host's game. <paramref name="joinOnly"/> marks a joiner (link/code/browser row,
        /// never a host): a joiner must never mint a room, since a typo'd code would otherwise create a
        /// phantom public game and strand the joiner with a seat in it. Missing room + joinOnly turns the
        /// connection away instead. <paramref name="token"/> is the client's persistent identity (null
        /// falls back to <paramref name="connectionId"/> — a fresh/garbled token is a fresh identity, same
        /// as today's per-socket behaviour). Any connect into an already-Started room — including this
        /// same reconnect — gets a personal START re-deal of the current session state.</summary>
        public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default,
            bool isPrivate = false, bool joinOnly = false, string? token = null)
        {
            token ??= connectionId;
            Sweep();

            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room))
            {
                if (joinOnly)
                {
                    outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                    return outs;
                }
                room = new Room(_newGame(setup), setup.Sanitized(), isPrivate, _now());
                _rooms[roomCode] = room;
            }

            var seat = room.Session.Join(token);
            if (seat == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.SeatFull));
                return outs;
            }

            room.ConnToToken[connectionId] = token;
            room.EmptySinceTicks = null; // any successful connect cancels a pending hold-expiry
            bool added = false;
            if (!room.Members.Contains(connectionId)) { room.Members.Add(connectionId); added = true; }
            outs.Add(new Outbound(connectionId, NetProtocol.Seat(seat.Value)));

            if (room.Started)
            {
                // (Re)connect into an already-started room: a personal re-deal of the current state — the
                // same resync-by-replay mechanism used for the initial deal below, now also serving a
                // reconnect after a drop.
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                outs.Add(new Outbound(connectionId, startMsg));
            }
            else if (added && room.Members.Count == 2)
            {
                room.Started = true;
                string startMsg = NetProtocol.Start(ReplayFile.Write(room.Session.State, Array.Empty<Command>()));
                foreach (var m in room.Members) outs.Add(new Outbound(m, startMsg));
            }
            return outs;
        }

        /// <summary>The lobby browser's view: public rooms with a waiting host and no game started,
        /// newest first. Callers own thread-safety (the transport already serializes hub access).</summary>
        public IReadOnlyList<OpenGame> OpenGames()
        {
            Sweep();
            var list = new List<OpenGame>();
            foreach (var kv in _rooms)
            {
                var r = kv.Value;
                if (r.IsPrivate || r.Started || r.Members.Count != 1) continue;
                int age = (int)((_now() - r.CreatedAtTicks) / TimeSpan.TicksPerSecond);
                list.Add(new OpenGame(kv.Key, r.Setup, age));
            }
            list.Sort((a, b) => a.AgeSeconds.CompareTo(b.AgeSeconds)); // newest (smallest age) first
            return list;
        }

        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            Sweep();
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            var cmd = CommandWire.Read(msg.Payload);
            string token = room.ConnToToken.TryGetValue(connectionId, out var t) ? t : connectionId;
            var outcome = room.Session.Submit(token, cmd);
            switch (outcome.Status)
            {
                case SubmitStatus.Accepted:
                    string applyMsg = NetProtocol.Apply(cmd);
                    foreach (var m in room.Members) outs.Add(new Outbound(m, applyMsg));
                    break;
                case SubmitStatus.Rejected:
                    outs.Add(new Outbound(connectionId, NetProtocol.Reject(outcome.Reason)));
                    break;
                default: // NoSeat / WrongSeat — tell only the offender, never touch the game
                    outs.Add(new Outbound(connectionId, "REJECT " + outcome.Status));
                    break;
            }
            return outs;
        }

        /// <summary>Drop a connection. Its room's seat is NOT freed here — seats belong to tokens for the
        /// life of a Started room, so the same token can always reclaim it (see class doc). A Started room
        /// that drops to zero live connections starts its hold-window clock instead of being removed; an
        /// un-started room (nobody ever dealt START) is still removed the instant it empties.</summary>
        public IReadOnlyList<Outbound> Disconnect(string roomCode, string connectionId)
        {
            Sweep();
            if (_rooms.TryGetValue(roomCode, out var room))
            {
                room.Members.Remove(connectionId);
                room.ConnToToken.Remove(connectionId);
                if (room.Members.Count == 0)
                {
                    if (room.Started) room.EmptySinceTicks = _now();
                    else _rooms.Remove(roomCode);
                }
            }
            return Array.Empty<Outbound>();
        }
    }
}
```

`engine/HexWars.NetServer/Program.cs` — `Handle`, old (top of the method, as read from the repo):
```csharp
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
            bool joinOnly = ctx.Request.Query["join"].ToString() == "1";

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            Console.WriteLine($"[ws] CONNECT room={room} id={conn.Id[..8]} setup=({setup.Mode} {setup.Width}x{setup.Height} pts{setup.StartingPoints} seed{setup.Seed}) total={Conns.Count}");
            try
            {
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate, joinOnly)));
```
New (only the query-parsing and the `Hub.Connect` call change; the `while`/`finally` body below is
untouched here — Task 5 restructures `Locked`/`Dispatch` in that body):
```csharp
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
            bool joinOnly = ctx.Request.Query["join"].ToString() == "1";
            string? token = ctx.Request.Query["token"].ToString();
            if (string.IsNullOrWhiteSpace(token)) token = null; // absent/garbled -> fresh identity (today's behavior)

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            Console.WriteLine($"[ws] CONNECT room={room} id={conn.Id[..8]} setup=({setup.Mode} {setup.Width}x{setup.Height} pts{setup.StartingPoints} seed{setup.Seed}) total={Conns.Count}");
            try
            {
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate, joinOnly, token)));
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   302, Skipped:     0, Total:   302` (296 + 6 new `TokenRejoinTests`;
`NetDisconnectTests.cs` still has 3 tests total — 2 rewritten in place, not added).

- [ ] **Step 3: Rebuild the Unity DLL and verify compilation**

```powershell
& "C:\Program Files\dotnet\dotnet.exe" build -c Release engine\HexWars.Engine\HexWars.Engine.csproj
Copy-Item engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll Assets\HexWars\Plugins\ -Force
```
Then coplay `check_compile_errors` — expect zero errors.

- [ ] **Step 4: Extend the selftest with a kill-socket / reconnect-same-token scenario**

In `engine/HexWars.NetServer/SelfTest.cs`, insert a new scenario block right before `await app.StopAsync();`
(after the existing `joinOnlyTurnedAway` block, replacing the tail of `Run`). Old tail:
```csharp
                using var joiner = await Connect("ws://127.0.0.1:5234/ws?room=missing&join=1");
                string joinerSeat = await Recv(joiner);       // a joiner must never mint a room for a typo'd code
                bool joinOnlyTurnedAway = joinerSeat == "SEAT FULL";

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart && joinOnlyTurnedAway;

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}' lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart} joinOnly='{joinerSeat}'");

                await app.StopAsync();
                return ok ? 0 : 1;
```
New tail:
```csharp
                using var joiner = await Connect("ws://127.0.0.1:5234/ws?room=missing&join=1");
                string joinerSeat = await Recv(joiner);       // a joiner must never mint a room for a typo'd code
                bool joinOnlyTurnedAway = joinerSeat == "SEAT FULL";

                // Reconnect: kill A's socket, reconnect with the SAME token, and confirm the server
                // seats it back into P0 and re-deals START (the game must survive a background/refresh).
                using var ra = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA = await Recv(ra);               // SEAT 0
                using var rb = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-b");
                string rSeatB = await Recv(rb);                // SEAT 1
                string rStartA = await Recv(ra);
                string rStartB = await Recv(rb);

                ra.Abort();                                     // simulate a dead socket (no clean close)
                using var ra2 = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA2 = await Recv(ra2);                // SEAT 0 again — same token, same seat
                string rStartA2 = await Recv(ra2);               // personal START re-deal

                await Send(ra2, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string rApplyA2 = await Recv(ra2);
                string rApplyB = await Recv(rb);

                bool reconnectOk =
                    rSeatA == "SEAT 0" && rSeatB == "SEAT 1" &&
                    rStartA.StartsWith("START ") && rStartB.StartsWith("START ") &&
                    rSeatA2 == "SEAT 0" && rStartA2.StartsWith("START ") &&
                    rApplyA2 == "APPLY E 0" && rApplyB == "APPLY E 0";

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart && joinOnlyTurnedAway && reconnectOk;

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}' lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart} joinOnly='{joinerSeat}' reconnectOk={reconnectOk}");

                await app.StopAsync();
                return ok ? 0 : 1;
```

Run: `dotnet run --project engine/HexWars.NetServer -- selftest`
Expected: `SELFTEST PASS — two browsers can play head-to-head through this server`, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Net/MatchHub.cs engine/HexWars.Engine/Net/GameSession.cs engine/HexWars.NetServer/Program.cs engine/HexWars.NetServer/SelfTest.cs engine/HexWars.Engine.Tests/TokenRejoinTests.cs engine/HexWars.Engine.Tests/NetDisconnectTests.csgit commit -m "feat(net): token-keyed seats + held-room reconnect - a client-minted token (not the socket's connection id) owns a seat, so a refresh/background/reconnect reclaims the same game; a started room survives both sockets dropping for a 10-minute hold window"
```

---

### Task 5: Server hardening batch

**Files:**
- Modify: `engine/HexWars.Engine/Net/CommandWire.cs`
- Modify: `engine/HexWars.Engine/Net/MatchHub.cs`
- Modify: `engine/HexWars.Engine/Net/NetProtocol.cs`
- Modify: `engine/HexWars.NetServer/Program.cs`
- Modify: `engine/HexWars.Engine.Tests/CommandWireTests.cs`
- Modify: `engine/HexWars.Engine.Tests/MatchHubTests.cs`
- Modify: `engine/HexWars.Engine.Tests/NetProtocolTests.cs`
- Modify: `Assets/HexWars/Plugins/HexWars.Engine.dll` (rebuild + copy, final step)

**Interfaces:**
- Produces: `public static bool TryRead(string line, out Command? command)` on `CommandWire` — never throws; `false` + `command = null` on any malformed line.
- Produces: `public const string Malformed = "REJECT Malformed";` on `NetProtocol`.
- Consumes: `MatchHub.Receive` (Task 4's version) now calls `CommandWire.TryRead` instead of `CommandWire.Read` and replies `NetProtocol.Malformed` to the issuer on failure, instead of letting the `FormatException` propagate.
- `engine/HexWars.NetServer/Program.cs` gains a per-connection `Channel<string>` outbound queue, a 64 KB
  incoming-message cap, and an `Origin` vs request-`Host` check — none of this is covered by
  `HexWars.Engine.Tests` (there is no `HexWars.NetServer` test project; confirmed by listing `engine/*.csproj`).
  It is verified by the selftest (still PASS) and a manual smoke step; see Step 4's rationale for why no
  automated test is added for the two structural/transport-level pieces.

- [ ] **Step 1: RED — `TryRead` and the malformed-command path don't exist yet**

Add to `engine/HexWars.Engine.Tests/CommandWireTests.cs` (after `DeleteTemplate_RoundTrips`, before
`Read_UnknownToken_Throws`):

```csharp
        [Test]
        public void TryRead_ValidLine_ReturnsTrueAndCommand()
        {
            Assert.That(CommandWire.TryRead("E 0", out var cmd), Is.True);
            Assert.That(cmd, Is.EqualTo(new EndTurn(P0)));
        }

        [Test]
        public void TryRead_UnknownToken_ReturnsFalse()
        {
            Assert.That(CommandWire.TryRead("Z 0", out var cmd), Is.False);
            Assert.That(cmd, Is.Null);
        }

        [Test]
        public void TryRead_TruncatedLine_ReturnsFalse()
        {
            Assert.That(CommandWire.TryRead("M 0", out var cmd), Is.False); // MoveUnit needs more tokens
            Assert.That(cmd, Is.Null);
        }

        [Test]
        public void TryRead_EmptyString_ReturnsFalse()
        {
            Assert.That(CommandWire.TryRead("", out var cmd), Is.False);
            Assert.That(cmd, Is.Null);
        }

        [Test]
        public void TryRead_NonNumericIssuer_ReturnsFalse()
        {
            Assert.That(CommandWire.TryRead("E notanumber", out var cmd), Is.False);
            Assert.That(cmd, Is.Null);
        }
```

Add to `engine/HexWars.Engine.Tests/MatchHubTests.cs` (after `OutOfTurn_RejectsWithEngineReason`, before
`Connect_RoomBuiltFromHostSetup_NotJoiners`):

```csharp
        [Test]
        public void Receive_MalformedCommand_RejectsIssuerOnly_NoBroadcast()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            var outs = hub.Receive("r", "a", "CMD Z garbage");
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == NetProtocol.Malformed));
            Assert.That(outs, Has.None.Matches<Outbound>(o => o.Message.StartsWith("APPLY")));
        }
```

Add to `engine/HexWars.Engine.Tests/NetProtocolTests.cs` (after `Reject_CarriesReasonName`, before the
class's closing brace):

```csharp

        [Test]
        public void Malformed_IsAFixedRejectLine()
        {
            var msg = NetProtocol.Parse(NetProtocol.Malformed);
            Assert.That(msg.Type, Is.EqualTo("REJECT"));
            Assert.That(msg.Payload, Is.EqualTo("Malformed"));
        }
```

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected failure: build error — `error CS0117: 'CommandWire' does not contain a definition for 'TryRead'`
and `error CS0117: 'NetProtocol' does not contain a definition for 'Malformed'`.

- [ ] **Step 2: GREEN — `CommandWire.TryRead`, `NetProtocol.Malformed`, `MatchHub.Receive` tolerant parsing**

`engine/HexWars.Engine/Net/CommandWire.cs` — old (as Tasks 1+2 left `Read`/the section right after it):
```csharp
                case "X": return new DeleteTemplate(issuer, I(p[2]));
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "N": return new DeployGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "H": return new CaptureHex(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "B": return new BuildGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                default: throw new FormatException("unknown command token " + p[0]);
            }
        }

        internal static string WriteStats(UnitStats s) =>
```
New (adds `TryRead` right after `Read`, before `WriteStats`):
```csharp
                case "X": return new DeleteTemplate(issuer, I(p[2]));
                case "D": return new DeployUnit(issuer, I(p[2]), new HexCoord(I(p[3]), I(p[4])));
                case "N": return new DeployGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "H": return new CaptureHex(issuer, new HexCoord(I(p[2]), I(p[3])));
                case "B": return new BuildGenerator(issuer, new HexCoord(I(p[2]), I(p[3])));
                default: throw new FormatException("unknown command token " + p[0]);
            }
        }

        /// <summary>Reads a wire line and returns false (instead of throwing) on any malformed input —
        /// unknown token, truncated payload, non-numeric field. Used by the server so one garbled
        /// client message can never crash the room (spec §3, audit N4).</summary>
        public static bool TryRead(string line, out Command? command)
        {
            try
            {
                command = Read(line);
                return true;
            }
            catch
            {
                command = null;
                return false;
            }
        }

        internal static string WriteStats(UnitStats s) =>
```

`engine/HexWars.Engine/Net/NetProtocol.cs` — old:
```csharp
        /// <summary>The room is full; the connection is a spectator/turned away.</summary>
        public const string SeatFull = "SEAT FULL";
```
New:
```csharp
        /// <summary>The room is full; the connection is a spectator/turned away.</summary>
        public const string SeatFull = "SEAT FULL";
        /// <summary>A CMD payload that failed to parse (CommandWire.TryRead returned false) — sent only
        /// to the issuer, never broadcast.</summary>
        public const string Malformed = "REJECT Malformed";
```

`engine/HexWars.Engine/Net/MatchHub.cs` — `Receive`, old (Task 4's version):
```csharp
        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            Sweep();
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            var cmd = CommandWire.Read(msg.Payload);
            string token = room.ConnToToken.TryGetValue(connectionId, out var t) ? t : connectionId;
            var outcome = room.Session.Submit(token, cmd);
```
New:
```csharp
        public IReadOnlyList<Outbound> Receive(string roomCode, string connectionId, string raw)
        {
            Sweep();
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room)) return outs;

            var msg = NetProtocol.Parse(raw);
            if (msg.Type != "CMD") return outs; // v0: CMD is the only client→server message that does anything

            if (!CommandWire.TryRead(msg.Payload, out var cmd) || cmd == null)
            {
                outs.Add(new Outbound(connectionId, NetProtocol.Malformed));
                return outs;
            }

            string token = room.ConnToToken.TryGetValue(connectionId, out var t) ? t : connectionId;
            var outcome = room.Session.Submit(token, cmd);
```
(The `switch (outcome.Status)` block immediately below is unchanged — `cmd` is still in scope with the
same type, just resolved through `TryRead` now.)

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: `Passed!  - Failed:     0, Passed:   309, Skipped:     0, Total:   309` (302 + 7 new: 5 CommandWireTests + 1 MatchHubTests + 1 NetProtocolTests).

- [ ] **Step 3: Rebuild the Unity DLL and verify compilation**

```powershell
& "C:\Program Files\dotnet\dotnet.exe" build -c Release engine\HexWars.Engine\HexWars.Engine.csproj
Copy-Item engine\HexWars.Engine\bin\Release\netstandard2.1\HexWars.Engine.dll Assets\HexWars\Plugins\ -Force
```
Then coplay `check_compile_errors` — expect zero errors.

- [ ] **Step 4: NetServer transport hardening — ordered outbound queue, 64 KB cap, Origin check (no TDD cycle; see rationale below)**

These three pieces live entirely in `HexWars.NetServer`, which has no test project (confirmed:
`engine/*.csproj` lists only `HexWars.Engine`, `HexWars.Engine.Tests`, `HexWars.GymServer`, `HexWars.Sim`,
`HexWars.NetServer` — no `HexWars.NetServer.Tests`). The task's own test list only asks for "hub-level
TryRead garbage cases; selftest still PASS; note ordering race is structural (no practical test —
document why)" — that reasoning extends to the 64 KB cap and Origin check too: exercising them requires
a live Kestrel WebSocket handshake with custom headers / oversized frames, which is disproportionate
scaffolding for this repo's test infra to add in this task. They are implemented directly, verified by
the selftest (which must keep passing unchanged — it sends no `Origin` header and small messages, so
neither new check fires for it) plus a manual curl smoke step below.

**Ordering race, restructured.** Today, `Handle` computes `Hub.Connect`/`Hub.Receive` output under
`HubLock` (via `Locked`), then `await Dispatch(...)` the actual socket sends *after* the lock is
released. Two concurrent `Handle` calls (two different sockets acting near-simultaneously) can compute
their outbound messages under the lock in the correct serialized order, but their real `SendAsync` calls
happen concurrently afterward — nothing then guarantees a receiver sees them in that same order (audit
N2). The fix: give every `Conn` its own `Channel<string>` + one dedicated writer task, and make the hub
call sites enqueue synchronously *inside* the lock (no `await` needed for a channel `Write`), so the
serialization the lock already provides extends all the way to "which message landed in which
connection's send queue first." Actual socket I/O still happens on each connection's own background task,
but strictly in the order it was enqueued.

`engine/HexWars.NetServer/Program.cs` — old (full file, as Task 4 left it: only the `token` parsing/passthrough
in `Handle` changed from the pristine original; `Locked`/`Dispatch`/`Conns`/`Conn`/`Receive` are all
still in their original shape):
```csharp
using System.Collections.Concurrent;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;

namespace HexWars.NetServer
{
    /// <summary>
    /// Thin WebSocket adapter over <see cref="MatchHub"/>: turns sockets into Connect/Receive calls and
    /// routes the resulting <see cref="Outbound"/> messages back to the right connections. All game logic
    /// lives in the (unit-tested) engine; this file is just plumbing. Cloud-ready: binds 0.0.0.0:$PORT
    /// when a host injects PORT, and serves the WebGL client from wwwroot when present (single origin).
    /// Run `HexWars.NetServer selftest` to drive two in-process clients through a move and assert.
    /// </summary>
    public static class Program
    {
        static readonly ConcurrentDictionary<string, Conn> Conns = new();
        static readonly MatchHub Hub = new(GameFactory.Build);
        static readonly object HubLock = new();

        public static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "selftest") return await SelfTest.Run();

            var builder = WebApplication.CreateBuilder(args);
            var port = Environment.GetEnvironmentVariable("PORT");
            if (!string.IsNullOrWhiteSpace(port)) builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

            var app = builder.Build();
            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client (index.html) from wwwroot/ when a deploy copies it in
            // Unity WebGL ships .unityweb/.data/.wasm; without these mappings Kestrel 404s them.
            var types = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
            types.Mappings[".unityweb"] = "application/octet-stream"; // gzip payload; the loader decompresses (decompressionFallback)
            types.Mappings[".data"] = "application/octet-stream";
            types.Mappings[".wasm"] = "application/wasm";
            app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = types });
            app.MapGet("/healthz", () => "ok");
            // The lobby browser: open public games as JSON. Same origin as the WebGL client, no CORS.
            app.MapGet("/games", () =>
            {
                IReadOnlyList<OpenGame> open;
                lock (HubLock) open = Hub.OpenGames();
                return Results.Json(new
                {
                    games = open.Select(g => new
                    {
                        code = g.Code,
                        mode = g.Setup.Mode.ToString(),
                        width = g.Setup.Width,
                        height = g.Setup.Height,
                        fog = g.Setup.Fog,
                        pace = g.Setup.TurnActions,
                        army = g.Setup.ArmySize,
                        ageSeconds = g.AgeSeconds,
                    }).ToArray(),
                });
            });
            app.Map("/ws", Handle);
            await app.RunAsync();
            return 0;
        }

        /// <summary>Accept a socket, seat it in the room from ?room=, then pump messages until it closes.</summary>
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
            bool joinOnly = ctx.Request.Query["join"].ToString() == "1";
            string? token = ctx.Request.Query["token"].ToString();
            if (string.IsNullOrWhiteSpace(token)) token = null; // absent/garbled -> fresh identity (today's behavior)

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            Console.WriteLine($"[ws] CONNECT room={room} id={conn.Id[..8]} setup=({setup.Mode} {setup.Width}x{setup.Height} pts{setup.StartingPoints} seed{setup.Seed}) total={Conns.Count}");
            try
            {
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate, joinOnly, token)));
                while (socket.State == WebSocketState.Open)
                {
                    string? text = await Receive(socket);
                    if (text is null) break;          // closed / errored
                    if (text.Length == 0) continue;
                    Console.WriteLine($"[ws] RECV  room={room} id={conn.Id[..8]}: {text}");
                    await Dispatch(Locked(() => Hub.Receive(room, conn.Id, text)));
                }
            }
            finally
            {
                Console.WriteLine($"[ws] DISCONNECT room={room} id={conn.Id[..8]}");
                Locked(() => Hub.Disconnect(room, conn.Id)); // free the seat so a refresh/rejoin can re-take it
                Conns.TryRemove(conn.Id, out _);
                if (socket.State == WebSocketState.Open)
                    try { await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", default); } catch { }
            }
        }

        // MatchHub isn't thread-safe; serialize all access. Calls are synchronous and fast (no awaits).
        static IReadOnlyList<Outbound> Locked(Func<IReadOnlyList<Outbound>> f) { lock (HubLock) return f(); }

        static async Task Dispatch(IReadOnlyList<Outbound> outs)
        {
            foreach (var o in outs)
            {
                Console.WriteLine($"[ws] SEND  -> {o.ConnectionId[..8]}: {(o.Message.Length > 60 ? o.Message[..60] + "…" : o.Message)}");
                if (Conns.TryGetValue(o.ConnectionId, out var c))
                    try { await c.Send(o.Message); } catch { /* drop a dead socket; cleanup happens on its own loop */ }
            }
        }

        static async Task<string?> Receive(WebSocket socket)
        {
            var buf = new byte[8192];
            using var ms = new MemoryStream();
            WebSocketReceiveResult res;
            do
            {
                try { res = await socket.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None); }
                catch { return null; }
                if (res.MessageType == WebSocketMessageType.Close) return null;
                ms.Write(buf, 0, res.Count);
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        /// <summary>Parse the host's lobby picks from the connect query (?setup=...); default if absent/bad.</summary>
        static GameSetup ParseSetup(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return GameSetup.Default;
            try { return GameSetup.Parse(raw); } catch { return GameSetup.Default; }
        }

        /// <summary>Uppercase alphanumerics only, capped at 16 — so "kq7kp", " KQ7KP " and a pasted
        /// URL fragment all land in the same room, and a hostile room string can't be huge.</summary>
        internal static string NormalizeRoom(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return "DEFAULT";
            var sb = new StringBuilder();
            foreach (char ch in raw.Trim().ToUpperInvariant())
            {
                if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) sb.Append(ch);
                if (sb.Length == 16) break;
            }
            return sb.Length == 0 ? "DEFAULT" : sb.ToString();
        }

        /// <summary>Selftest hook: a locked snapshot of the lobby list.</summary>
        internal static IReadOnlyList<OpenGame> OpenGamesSnapshot() { lock (HubLock) return Hub.OpenGames(); }
    }

    /// <summary>One live connection: its id + socket, with sends serialized (one SendAsync at a time).</summary>
    sealed class Conn
    {
        public readonly string Id;
        public readonly WebSocket Socket;
        readonly SemaphoreSlim _send = new(1, 1);

        public Conn(string id, WebSocket socket) { Id = id; Socket = socket; }

        public async Task Send(string msg)
        {
            await _send.WaitAsync();
            try { await Socket.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None); }
            finally { _send.Release(); }
        }
    }
}
```
New (full file — per-connection `Channel<string>` outbound queue with enqueue-inside-the-lock, 64 KB
incoming cap, `Origin` vs `Host` check before `AcceptWebSocketAsync`):
```csharp
using System.Collections.Concurrent;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading.Channels;
using HexWars.Engine;

namespace HexWars.NetServer
{
    /// <summary>
    /// Thin WebSocket adapter over <see cref="MatchHub"/>: turns sockets into Connect/Receive calls and
    /// routes the resulting <see cref="Outbound"/> messages back to the right connections. All game logic
    /// lives in the (unit-tested) engine; this file is just plumbing. Cloud-ready: binds 0.0.0.0:$PORT
    /// when a host injects PORT, and serves the WebGL client from wwwroot when present (single origin).
    /// Run `HexWars.NetServer selftest` to drive two in-process clients through a move and assert.
    /// </summary>
    public static class Program
    {
        static readonly ConcurrentDictionary<string, Conn> Conns = new();
        static readonly MatchHub Hub = new(GameFactory.Build);
        static readonly object HubLock = new();
        const int MaxIncomingBytes = 64 * 1024;

        public static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "selftest") return await SelfTest.Run();

            var builder = WebApplication.CreateBuilder(args);
            var port = Environment.GetEnvironmentVariable("PORT");
            if (!string.IsNullOrWhiteSpace(port)) builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

            var app = builder.Build();
            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client (index.html) from wwwroot/ when a deploy copies it in
            // Unity WebGL ships .unityweb/.data/.wasm; without these mappings Kestrel 404s them.
            var types = new Microsoft.AspNetCore.StaticFiles.FileExtensionContentTypeProvider();
            types.Mappings[".unityweb"] = "application/octet-stream"; // gzip payload; the loader decompresses (decompressionFallback)
            types.Mappings[".data"] = "application/octet-stream";
            types.Mappings[".wasm"] = "application/wasm";
            app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = types });
            app.MapGet("/healthz", () => "ok");
            // The lobby browser: open public games as JSON. Same origin as the WebGL client, no CORS.
            app.MapGet("/games", () =>
            {
                IReadOnlyList<OpenGame> open;
                lock (HubLock) open = Hub.OpenGames();
                return Results.Json(new
                {
                    games = open.Select(g => new
                    {
                        code = g.Code,
                        mode = g.Setup.Mode.ToString(),
                        width = g.Setup.Width,
                        height = g.Setup.Height,
                        fog = g.Setup.Fog,
                        pace = g.Setup.TurnActions,
                        army = g.Setup.ArmySize,
                        ageSeconds = g.AgeSeconds,
                    }).ToArray(),
                });
            });
            app.Map("/ws", Handle);
            await app.RunAsync();
            return 0;
        }

        /// <summary>Accept a socket, seat it in the room from ?room=, then pump messages until it closes.</summary>
        internal static async Task Handle(HttpContext ctx)
        {
            if (!ctx.WebSockets.IsWebSocketRequest) { ctx.Response.StatusCode = 400; return; }
            if (!OriginAllowed(ctx)) { ctx.Response.StatusCode = 403; return; }
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
            bool joinOnly = ctx.Request.Query["join"].ToString() == "1";
            string? token = ctx.Request.Query["token"].ToString();
            if (string.IsNullOrWhiteSpace(token)) token = null; // absent/garbled -> fresh identity (today's behavior)

            var socket = await ctx.WebSockets.AcceptWebSocketAsync();
            var conn = new Conn(Guid.NewGuid().ToString("N"), socket);
            Conns[conn.Id] = conn;
            Console.WriteLine($"[ws] CONNECT room={room} id={conn.Id[..8]} setup=({setup.Mode} {setup.Width}x{setup.Height} pts{setup.StartingPoints} seed{setup.Seed}) total={Conns.Count}");
            try
            {
                Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate, joinOnly, token));
                while (socket.State == WebSocketState.Open)
                {
                    string? text = await Receive(socket);
                    if (text is null) break;          // closed / errored / over the size cap
                    if (text.Length == 0) continue;
                    Console.WriteLine($"[ws] RECV  room={room} id={conn.Id[..8]}: {text}");
                    Locked(() => Hub.Receive(room, conn.Id, text));
                }
            }
            finally
            {
                Console.WriteLine($"[ws] DISCONNECT room={room} id={conn.Id[..8]}");
                Locked(() => Hub.Disconnect(room, conn.Id));
                Conns.TryRemove(conn.Id, out _);
                conn.Close();
                if (socket.State == WebSocketState.Open)
                    try { await socket.CloseAsync(WebSocketCloseStatus.NormalClosure, "bye", default); } catch { }
            }
        }

        // MatchHub isn't thread-safe; serialize all access. The outbound Enqueue below is synchronous
        // (a Channel Write never blocks/awaits), so it runs INSIDE the lock along with the hub call —
        // this is what closes the APPLY-ordering race: two concurrent Handle() calls can no longer
        // interleave their actual sends, because "compute the outbound messages" and "hand them to each
        // connection's own ordered queue" are now one atomic, lock-serialized step (audit N2).
        static void Locked(Func<IReadOnlyList<Outbound>> f)
        {
            lock (HubLock)
            {
                var outs = f();
                foreach (var o in outs)
                {
                    Console.WriteLine($"[ws] SEND  -> {o.ConnectionId[..8]}: {(o.Message.Length > 60 ? o.Message[..60] + "…" : o.Message)}");
                    if (Conns.TryGetValue(o.ConnectionId, out var c)) c.Enqueue(o.Message);
                }
            }
        }

        static async Task<string?> Receive(WebSocket socket)
        {
            var buf = new byte[8192];
            using var ms = new MemoryStream();
            WebSocketReceiveResult res;
            do
            {
                try { res = await socket.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None); }
                catch { return null; }
                if (res.MessageType == WebSocketMessageType.Close) return null;
                ms.Write(buf, 0, res.Count);
                if (ms.Length > MaxIncomingBytes)
                {
                    try { await socket.CloseAsync(WebSocketCloseStatus.MessageTooBig, "message too large", CancellationToken.None); } catch { }
                    return null;
                }
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        /// <summary>When both an Origin header and a request Host are present, reject a mismatched
        /// Origin before Accept (audit M13) — closes cross-site WebSocket hijacking of a logged-in
        /// session. Absent/unparseable Origin (non-browser clients, the in-process selftest) is allowed
        /// through unchanged, matching the spec's "when both present" scope.</summary>
        static bool OriginAllowed(HttpContext ctx)
        {
            string origin = ctx.Request.Headers.Origin.ToString();
            string host = ctx.Request.Host.Value;
            if (string.IsNullOrEmpty(origin) || string.IsNullOrEmpty(host)) return true;
            if (!Uri.TryCreate(origin, UriKind.Absolute, out var originUri)) return true;
            string originAuthority = originUri.IsDefaultPort ? originUri.Host : $"{originUri.Host}:{originUri.Port}";
            return string.Equals(originAuthority, host, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>Parse the host's lobby picks from the connect query (?setup=...); default if absent/bad.</summary>
        static GameSetup ParseSetup(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return GameSetup.Default;
            try { return GameSetup.Parse(raw); } catch { return GameSetup.Default; }
        }

        /// <summary>Uppercase alphanumerics only, capped at 16 — so "kq7kp", " KQ7KP " and a pasted
        /// URL fragment all land in the same room, and a hostile room string can't be huge.</summary>
        internal static string NormalizeRoom(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return "DEFAULT";
            var sb = new StringBuilder();
            foreach (char ch in raw.Trim().ToUpperInvariant())
            {
                if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) sb.Append(ch);
                if (sb.Length == 16) break;
            }
            return sb.Length == 0 ? "DEFAULT" : sb.ToString();
        }

        /// <summary>Selftest hook: a locked snapshot of the lobby list.</summary>
        internal static IReadOnlyList<OpenGame> OpenGamesSnapshot() { lock (HubLock) return Hub.OpenGames(); }
    }

    /// <summary>One live connection: its id + socket, with an ordered per-connection outbound queue (a
    /// single writer task drains it) instead of a semaphore around SendAsync — Enqueue is synchronous
    /// and never blocks the hub-lock caller (see Program.Locked).</summary>
    sealed class Conn
    {
        public readonly string Id;
        public readonly WebSocket Socket;
        readonly Channel<string> _outbox = Channel.CreateUnbounded<string>();
        readonly Task _writer;

        public Conn(string id, WebSocket socket)
        {
            Id = id;
            Socket = socket;
            _writer = Task.Run(PumpAsync);
        }

        /// <summary>Enqueue a message for this connection's single writer task. Never blocks or throws —
        /// an unbounded channel Write always succeeds synchronously.</summary>
        public void Enqueue(string msg) => _outbox.Writer.TryWrite(msg);

        async Task PumpAsync()
        {
            await foreach (var msg in _outbox.Reader.ReadAllAsync())
            {
                try { await Socket.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None); }
                catch { /* dead socket; Handle's own receive loop notices and cleans up */ }
            }
        }

        /// <summary>Stop accepting new messages and let the writer task drain what's already queued.</summary>
        public void Close() => _outbox.Writer.TryComplete();
    }
}
```

- [ ] **Step 5: Verify — selftest + manual smoke**

Run: `dotnet run --project engine/HexWars.NetServer -- selftest`
Expected: `SELFTEST PASS — two browsers can play head-to-head through this server`, exit code 0 (unchanged
from Task 4's Step 4 — the selftest sends no `Origin` header and only small messages, so neither new
check fires; the channel-based outbound is transparent to it).

Manual smoke (documents the two pieces that have no automated coverage, per Step 4's rationale — this is
not a scripted step, just the verification the plan performs before committing):
1. Start the server: `dotnet run --project engine/HexWars.NetServer` (background). Note the bound port
   from the startup log (defaults to 5234 per the selftest constant / NetClient's dev default; use
   `PORT=<n>` to override).
2. `curl -i -H "Origin: https://evil.example.com" http://127.0.0.1:<port>/ws` → expect `403` (the
   WebSocket upgrade handshake is rejected before `Accept`; a plain `curl` without the WebSocket upgrade
   headers would 400 either way, so the Origin check is the thing being confirmed by the status code, not
   a full handshake).
3. Stop the server.

- [ ] **Step 6: Commit**

```bash
git add engine/HexWars.Engine/Net/CommandWire.cs engine/HexWars.Engine/Net/MatchHub.cs engine/HexWars.Engine/Net/NetProtocol.cs engine/HexWars.NetServer/Program.cs engine/HexWars.Engine.Tests/CommandWireTests.cs engine/HexWars.Engine.Tests/MatchHubTests.cs engine/HexWars.Engine.Tests/NetProtocolTests.csgit commit -m "feat(net): server hardening batch - per-connection ordered outbound queue, 64KB incoming cap, Origin check, CommandWire.TryRead + REJECT Malformed for garbage payloads"
```

---

## Client-half constraints (mechanisms validated live in the editor during drafting)

- **NEVER add attribution trailers to git commits** — no `Co-Authored-By`, no "Generated with Claude Code",
  no tool credits of any kind. (User's global CLAUDE.md; overrides all defaults.)
- **`engine/` is out of scope for every task in this plan.** The engine surface above is already landed; if
  a task discovers the live source disagrees with it, STOP and flag the conflict — do not patch `engine/`
  to match this plan.
- Unity work is verified in play mode via coplay `execute_script` on `Assets/Scenes/HexWars.unity`; every
  new/changed screen gets a screenshot (`capture_ui_canvas` / `capture_scene_object`) — visual verification
  is standing practice for this project, not optional polish.
- Coplay quirks (confirmed live against this project while drafting this plan):
  - `execute_script` takes a **file path** (`filePath`) to a `.cs` file containing a `public static` entry
    method, not an inline snippet — `methodName` defaults to `Execute` but any public static method name
    works. Scratch verification scripts belong in the harness scratchpad, not under `Assets/`.
  - No local functions, no type-pattern-matching (`is Foo f`) inside script bodies.
  - Never declare a bare `Object` — always `UnityEngine.Object` or `GameObject`/a concrete type (a bare
    `Object` silently resolves to a resource-culture type and the real error is masked as CS0104).
  - `Time.deltaTime` reads `0` inside `execute_script` (it doesn't run inside Unity's frame loop) — don't
    write verification code that depends on it advancing.
  - One UI transition per `execute_script` call (open a panel, screenshot, THEN close it in the next call)
    — chaining several transitions in one call races Unity's own layout/callback timing.
  - For "this must NOT appear" assertions, use a positive control: first prove the thing DOES appear in the
    case where it should (e.g. Tips on), then prove the same check reports absent when it should (Tips off).
    An absence check with no positive control can't distinguish "correctly hidden" from "the check is broken".
  - Outside Play Mode, `Screen.width`/`Screen.height` do **not** track the Game View's configured size at
    all (confirmed: reads a fixed default). The portrait mechanism below only works, and is only meant to be
    invoked, while playing.
- **Portrait verification mechanism (validated live for this plan — see Task 8 for the reusable script):**
  `UnityEditor.GameViewSizes` has no public API to add/select a custom resolution; reflection is required.
  Confirmed empirically in this project: adding a `FixedResolution` `GameViewSize` (390×844) via
  `GameViewSizeGroup.AddCustomSize` and selecting it via `GameView.SizeSelectionCallback(index, size)` (both
  non-public, found via reflection) makes `Screen.width`/`Screen.height` report **exactly** `390x844` during
  Play Mode when the target size is smaller than the Game View panel (it was, on the dev machine this was
  tested on) — and a `ScreenSpaceOverlay` canvas's own `RectTransform.rect` then reports canvas-space
  ≈815×1765 (confirmed reading `HudCanvas`'s rect at that Screen size, matching the `CanvasScaler`
  `ScaleWithScreenSize` / `matchWidthOrHeight=0.5` math: geometric-mean scale factor ≈0.478 px/canvas-unit
  against the 1600×900 reference). That low scale factor — not overflow — is the real portrait failure mode
  for most panels: a 20pt banner font renders at under 10 physical px. Panels with content sized to assume
  landscape headroom (the HUD banner's single-line status string in particular) wrap/truncate inside their
  fixed-height strip at that scale; `EventConsole` is a different case entirely (raw `OnGUI` pixel math, not
  `CanvasScaler`-mediated — its own scale button already covers ~86% of a 390-wide screen, matching the
  audit's F1/U1 finding exactly) and is already fixed in Task 7c. Resizing the Game View's `EditorWindow`
  via `.position` was tried first and does **not** work reliably (a docked window ignores explicit `.position`
  edits) — do not use that approach.
- The deployed WebGL build is always the online client (`Networked = true`); the editor default path
  (`Networked = false` → `NewGame()` hotseat) must keep working untouched.
- Work on branch `feat/invite-readiness` off `main`.
- Commit after each task.

---

### Task 6: Client reconnect — token identity, capped-backoff retry, status surface

**Files:**
- Modify: `Assets/HexWars/Presentation/NetClient.cs`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs`
- Modify: `Assets/HexWars/Presentation/GameHud.cs`

**Interfaces:**
- Consumes: server-side `?room=&token=` seating and started-room START re-deal (already landed, per header).
- Produces (nothing downstream in this plan depends on new public surface beyond what's listed, but it must
  stay exactly this shape since it's the reconnect contract):
  - `public static string NetClient.Token()` — mints/persists the 16-char per-browser token.
  - `public bool GameBootstrap.Reconnecting { get; }` — true while a started game's socket is retrying.
  - `internal void GameBootstrap.OnNetReconnecting(int attempt)` / `OnNetReconnected()` — called by
    `NetClient`; both raise `StateChanged` so `GameHud` repaints.

- [ ] **Step 1: `NetClient` — token, and replace the one-shot `Connect` with a self-driving lifecycle**

Read the current file first (already quoted above in full during drafting — 97 lines). Replace the block
from `WebSocket _ws;` through the end of the current `Connect` method:

```csharp
        WebSocket _ws;
        GameBootstrap _game;

        public PlayerId? Seat { get; private set; }
        public bool Connected { get; private set; }

        bool _closing;

        public async void Connect(GameBootstrap game, string room, string setupWire, bool isPrivate = false)
        {
            _game = game;
            string url = ServerWsUrl(room, setupWire, isPrivate);
            Debug.Log("[Net] connecting to " + url);
            _ws = new WebSocket(url);
            _ws.OnOpen += () => { Connected = true; Debug.Log("[Net] open"); };
            _ws.OnError += e => Debug.LogError("[Net] error: " + e);
            _ws.OnClose += c => { Connected = false; Debug.Log("[Net] closed: " + c); if (!_closing) _game.OnNetClosed(); };
            _ws.OnMessage += OnMessage;
            await _ws.Connect();
        }
```

with:

```csharp
        WebSocket _ws;
        GameBootstrap _game;

        public PlayerId? Seat { get; private set; }
        public bool Connected { get; private set; }

        bool _closing;
        int _attempt;                 // 0 = first-ever attempt; >0 = a retry after a drop
        string _room, _setupWire;
        bool _isPrivate;

        const string TokenPrefKey = "HexWars.SeatToken";
        const string TokenAlphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ23456789";
        static readonly float[] BackoffSeconds = { 1f, 2f, 4f, 8f, 15f }; // caps at 15s, then repeats forever

        /// <summary>One token per browser, minted once and kept in PlayerPrefs. The server seats by
        /// token, not by socket (spec §3), so presenting the same token — after a refresh, a background
        /// tab drop, or this class's own reconnect loop — reclaims the same seat.</summary>
        public static string Token()
        {
            string t = PlayerPrefs.GetString(TokenPrefKey, "");
            if (!string.IsNullOrEmpty(t)) return t;
            var chars = new char[16];
            for (int i = 0; i < chars.Length; i++) chars[i] = TokenAlphabet[UnityEngine.Random.Range(0, TokenAlphabet.Length)];
            t = new string(chars);
            PlayerPrefs.SetString(TokenPrefKey, t);
            PlayerPrefs.Save();
            return t;
        }

        /// <summary>Start the connection lifecycle for a room. Remembers the args so a dropped socket
        /// can retry with the exact same request.</summary>
        public void Connect(GameBootstrap game, string room, string setupWire, bool isPrivate = false)
        {
            _game = game;
            _room = room;
            _setupWire = setupWire;
            _isPrivate = isPrivate;
            StartCoroutine(Lifecycle());
        }

        /// <summary>Owns every connection attempt for this component's lifetime. A drop BEFORE a game
        /// started (still on the host/join screen) is never retried — un-started rooms clean up
        /// instantly server-side, so there's nothing left to reconnect into; that case keeps today's
        /// toast-and-stop behavior via <see cref="GameBootstrap.OnNetClosed"/>. A drop AFTER a game
        /// started retries with capped exponential backoff (1s, 2s, 4s, 8s, cap 15s) indefinitely —
        /// until it reconnects or this component is destroyed (Cancel / Main menu both call
        /// <c>Destroy(_net)</c>, which stops this coroutine along with everything else).</summary>
        IEnumerator Lifecycle()
        {
            while (true)
            {
                var open = OpenOnce();
                while (!open.IsCompleted) yield return null;
                if (_closing) yield break;

                if (_attempt == 0 && _game.State == null)
                {
                    _game.OnNetClosed();   // pre-start drop: existing toast + SetupForm status path, no retry
                    yield break;
                }

                _game.OnNetReconnecting(_attempt);
                float wait = BackoffSeconds[Mathf.Min(_attempt, BackoffSeconds.Length - 1)];
                _attempt++;
                yield return new WaitForSeconds(wait);
                if (_closing) yield break;
            }
        }

        /// <summary>One connection attempt. Resolves once that attempt's socket session ends — either
        /// it never opened, or it opened and later closed. NativeWebSocket's <c>Connect()</c> task runs
        /// the whole read loop internally, so awaiting it IS awaiting "this attempt is over".</summary>
        async System.Threading.Tasks.Task OpenOnce()
        {
            string url = ServerWsUrl(_room, _setupWire, _isPrivate, Token());
            Debug.Log("[Net] connecting to " + url);
            _ws = new WebSocket(url);
            _ws.OnOpen += () =>
            {
                Connected = true;
                Debug.Log("[Net] open");
                if (_attempt > 0) { _attempt = 0; _game.OnNetReconnected(); }
            };
            _ws.OnError += e => Debug.LogError("[Net] error: " + e);
            _ws.OnClose += c => { Connected = false; Debug.Log("[Net] closed: " + c); };
            _ws.OnMessage += OnMessage;
            await _ws.Connect();
        }
```

Add `using System.Collections;` to the file's usings (needed for `IEnumerator`) — check the current using
list (`System`, `System.Text`, `UnityEngine`, `NativeWebSocket`, `HexWars.Engine`) and insert it alongside
`System.Text`.

- [ ] **Step 2: `OnDestroy` — stop the coroutine explicitly (belt-and-suspenders; Unity would anyway)**

Replace:

```csharp
        async void OnDestroy()
        {
            _closing = true;             // deliberate teardown (Cancel / ReturnToMenu) — not an error
            if (_ws != null) await _ws.Close();
        }
```

with:

```csharp
        async void OnDestroy()
        {
            _closing = true;             // deliberate teardown (Cancel / ReturnToMenu) — not an error
            StopAllCoroutines();         // Unity would stop Lifecycle() on destroy anyway; explicit for clarity
            if (_ws != null) await _ws.Close();
        }
```

- [ ] **Step 3: `ServerWsUrl` — always append the token**

Replace:

```csharp
        /// <summary>Build the WebSocket URL for a room from the page origin: https://host → wss://host/ws?room=…
        /// (&amp;setup=… for the host). Falls back to ws://127.0.0.1:5234 in the editor (no page URL).</summary>
        static string ServerWsUrl(string room, string setupWire, bool isPrivate)
        {
            string origin = "ws://127.0.0.1:5234"; // dev default when there's no page URL (editor)
            string page = Application.absoluteURL;
            if (!string.IsNullOrEmpty(page))
            {
                try
                {
                    var uri = new Uri(page);
                    origin = (uri.Scheme == "https" ? "wss" : "ws") + "://" + uri.Authority;
                }
                catch { /* keep dev default */ }
            }
            string url = origin + "/ws?room=" + Uri.EscapeDataString(room);
            if (!string.IsNullOrEmpty(setupWire)) url += "&setup=" + Uri.EscapeDataString(setupWire);
            else url += "&join=1"; // a joiner (link/code/browser row) never carries setup — flag it so a
                                    // missing room turns the connection away instead of minting a phantom game
            if (isPrivate) url += "&private=1";
            return url;
        }
```

with:

```csharp
        /// <summary>Build the WebSocket URL for a room from the page origin: https://host → wss://host/ws?room=…
        /// (&amp;setup=… for the host). Falls back to ws://127.0.0.1:5234 in the editor (no page URL).
        /// <paramref name="token"/> is always appended — the server seats by token, not connection id.</summary>
        static string ServerWsUrl(string room, string setupWire, bool isPrivate, string token)
        {
            string origin = "ws://127.0.0.1:5234"; // dev default when there's no page URL (editor)
            string page = Application.absoluteURL;
            if (!string.IsNullOrEmpty(page))
            {
                try
                {
                    var uri = new Uri(page);
                    origin = (uri.Scheme == "https" ? "wss" : "ws") + "://" + uri.Authority;
                }
                catch { /* keep dev default */ }
            }
            string url = origin + "/ws?room=" + Uri.EscapeDataString(room);
            if (!string.IsNullOrEmpty(setupWire)) url += "&setup=" + Uri.EscapeDataString(setupWire);
            else url += "&join=1"; // a joiner (link/code/browser row) never carries setup — flag it so a
                                    // missing room turns the connection away instead of minting a phantom game
            if (isPrivate) url += "&private=1";
            url += "&token=" + Uri.EscapeDataString(token);
            return url;
        }
```

- [ ] **Step 4: Compile check**

coplay `check_compile_errors` → zero errors. (NetClient.cs alone won't compile yet — `GameBootstrap` doesn't
have `OnNetReconnecting`/`OnNetReconnected` until Step 5. Do Steps 1-3 and 5-7 together before checking.)

- [ ] **Step 5: `GameBootstrap` — `Reconnecting` state + callbacks**

Insert a new property between the existing `DemoMode` property and the `StateChanged` event:

```csharp
        /// <summary>True while the title-screen demo game (AI vs AI, muted, gameplay UI hidden) is
        /// running. Real-game starters clear it. HUD/panels early-out on it — see the spec's
        /// suppression contract.</summary>
        public bool DemoMode { get; private set; }

        /// <summary>True while a started game's socket dropped and NetClient is retrying with backoff.
        /// GameHud reads this to show a persistent status line; OnNetReconnecting Toasts once per drop
        /// episode (not once per attempt).</summary>
        public bool Reconnecting { get; private set; }

        /// <summary>Raised after the state changes (new game or applied command) so HUD can refresh.</summary>
        public event System.Action StateChanged;
```

Replace `CancelHosting`:

```csharp
        /// <summary>Host changed their mind while waiting: drop the socket and seat. State stays as it
        /// was (null before START), so the title/demo behind the form is untouched.</summary>
        public void CancelHosting()
        {
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
        }
```

with:

```csharp
        /// <summary>Host changed their mind while waiting: drop the socket and seat. State stays as it
        /// was (null before START), so the title/demo behind the form is untouched.</summary>
        public void CancelHosting()
        {
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
            Reconnecting = false;
        }
```

Replace `ReturnToMenu`:

```csharp
        public void ReturnToMenu()
        {
            Presenter?.ResetQueue();
            GameOverBanner.Dismiss();
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
            var ai = GetComponent<AiOpponent>();
            if (ai != null) Destroy(ai);
            State = null;
            StateChanged?.Invoke();
            GetComponent<SetupForm>()?.Close();
            GetComponent<GameBrowser>()?.Close();
            StartDemo();
            TitleScreen.Reopen(this);
        }
```

with:

```csharp
        public void ReturnToMenu()
        {
            Presenter?.ResetQueue();
            GameOverBanner.Dismiss();
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
            Reconnecting = false;
            var ai = GetComponent<AiOpponent>();
            if (ai != null) Destroy(ai);
            State = null;
            StateChanged?.Invoke();
            GetComponent<SetupForm>()?.Close();
            GetComponent<GameBrowser>()?.Close();
            StartDemo();
            TitleScreen.Reopen(this);
        }
```

- [ ] **Step 6: `GameBootstrap` — reconnect callbacks + defensive clear in `OnNetStart`**

Insert new methods between `OnNetClosed` and `OnNetStart`:

```csharp
        internal void OnNetClosed()
        {
            if (Networked && State == null && _net != null)
                Toast.Show("Connection lost — check the link and try again.");
            GetComponent<SetupForm>()?.OnConnectionLost();
        }

        /// <summary>A started game's socket dropped and NetClient is retrying with backoff. Called once
        /// per attempt (so a persistent status line can show progress); the Toast only fires transitioning
        /// INTO reconnecting, not on every retry, matching spec §7 ("every attempt updates the status
        /// line" — GameHud's banner text is that status line).</summary>
        internal void OnNetReconnecting(int attempt)
        {
            if (!Reconnecting) Toast.Show("Connection lost — reconnecting…", new Color(0.42f, 0.34f, 0.12f, 0.94f));
            Reconnecting = true;
            StateChanged?.Invoke();
        }

        /// <summary>The socket reopened. The server re-deals START right behind this (OnNetStart also
        /// clears Reconnecting, redundantly, so arrival order between the two can never leave it stuck).</summary>
        internal void OnNetReconnected()
        {
            Reconnecting = false;
            StateChanged?.Invoke();
        }

        /// <summary>The server dealt the authoritative start state — load and render it.</summary>
        internal void OnNetStart(string startStateText)
        {
            Reconnecting = false;      // a START re-deal (fresh join OR a reconnect) always means we're live
            Presenter?.ResetQueue();
            State = ReplayFile.Read(startStateText).Start;
```

(everything from `var renderer = GetComponent<BoardRenderer>();` to the end of the existing `OnNetStart`
body is unchanged — only the two new lines at the top are new.)

- [ ] **Step 7: `GameHud` — surface `Reconnecting` on the banner**

In `Refresh()`, the normal (non-game-over) branch currently ends:

```csharp
            _banner.text = s.Config.TerritoryMode
                ? $"P{who}'s turn{pace}{done}{armies}     Round {s.Round}     " +
                  $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}"
                : $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")}){pace}{done}{armies}     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";

            if (_endBtn != null) _endBtn.color = done.Length > 0 ? EndTurnUrge : EndTurnIdle;
```

Replace with:

```csharp
            _banner.text = s.Config.TerritoryMode
                ? $"P{who}'s turn{pace}{done}{armies}     Round {s.Round}     " +
                  $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}"
                : $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")}){pace}{done}{armies}     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";
            if (_game.Reconnecting) _banner.text = "⚠ Connection lost — reconnecting…     " + _banner.text;

            if (_endBtn != null) _endBtn.color = done.Length > 0 ? EndTurnUrge : EndTurnIdle;
```

- [ ] **Step 8: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 9: Verify against the local server — happy-path reconnect + rejoin**

Start the server in the background: `PowerShell -Command "$env:PORT='5234'; dotnet run --project engine/HexWars.NetServer"`
with `run_in_background: true`.

In the editor: enter Play Mode on `Assets/Scenes/HexWars.unity`. Each numbered item is its own
`execute_script` call (one transition per call, per the coplay quirk in Global Constraints):
1. Host a room: `var game = FindAnyObjectByType<GameBootstrap>(); game.StartNetGame("VERIFY6", new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Annihilation, 9, 7, 0, 7).ToWire()); return "hosting";`
2. Fill the second seat so the room STARTs, using a throwaway second `NetClient` pointed at the SAME
   `GameBootstrap` (its `OnMessage` unconditionally calls `_game.OnNetSeat(...)`/`OnNetStart(...)`, so a
   `null` game would NullReferenceException the instant SEAT arrives — it must be a real, if shared,
   instance):
   ```csharp
   var game = FindAnyObjectByType<GameBootstrap>();
   var secondGo = new UnityEngine.GameObject("SecondSeat");
   var second = secondGo.AddComponent<HexWars.Presentation.NetClient>();
   second.Connect(game, "VERIFY6", null);
   return "joining";
   ```
   Wait ~1s, then `execute_script`: assert `game.State != null` (START arrived) and `game.Seat.HasValue`.
   Immediately destroy the throwaway second connection so it can't also enter Task 6's retry loop and
   confuse the next steps: `UnityEngine.Object.Destroy(GameObject.Find("SecondSeat")); return "cleaned";`
   (this seat is now abandoned server-side, which is fine — the room stays `Started`, per the landed
   engine surface, and the test only needs the primary connection's behavior from here).
3. Kill the server process (find and stop the background PowerShell/dotnet process). `execute_script` after
   ~2s: assert `game.Reconnecting == true`. Screenshot the HUD — banner should read
   `⚠ Connection lost — reconnecting…     …` (this task's exact wording; Task 8 shortens it later once the
   portrait pass exists — verifying that shortened text is Task 8's own job, not this one).
4. Restart the server the same way. Wait ~3-5s (first backoff tier is 1s, so it should reconnect on the
   first or second attempt). `execute_script`: assert `game.Reconnecting == false` and `game.State != null`
   (the START re-deal landed). Screenshot the HUD — banner back to normal.
5. Seat-reclaim check: `execute_script` — destroy the primary `NetClient` component directly:
   `UnityEngine.Object.Destroy(FindAnyObjectByType<HexWars.Presentation.NetClient>()); return "destroyed";`.
6. `Destroy` is deferred to end of frame — wait ~0.5s, then reconnect fresh in a separate call:
   `var game = FindAnyObjectByType<GameBootstrap>(); game.StartNetGame("VERIFY6", null); return "rejoining";`
   (joiner path, no setupWire).
7. Wait ~1s, then assert the resulting `game.Seat` is the SAME value captured in step 2 — same token, same
   seat, proving `NetClient.Token()` persisted across the destroy/recreate (PlayerPrefs survives it) and the
   server's token-keyed seating reclaimed it.
8. Exit Play Mode. Stop the background server process.

- [ ] **Step 10: Verify Cancel/Main-menu abort the retry loop**

Repeat steps 1-3 above (host, second seat joins, kill server, confirm `Reconnecting == true`), then
`execute_script`: `FindAnyObjectByType<GameBootstrap>().ReturnToMenu(); return "menu";`. Wait ~2s past the
next backoff tier's deadline, then assert no exception was logged (`get_unity_logs`, filter for `[Net]` —
there should be no NEW `connecting to` lines after `ReturnToMenu` fired) and `Reconnecting == false`. Restart
the server for cleanliness if left down.

- [ ] **Step 11: Commit**

```bash
git add Assets/HexWars/Presentation/NetClient.cs Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/GameHud.cs
git commit -m "feat(net): client reconnect - token identity persisted in PlayerPrefs, capped-backoff retry loop for started games, GameHud/Toast status surface"
```

---

### Task 7: Session longevity — HP bars, tooltip cache, EventConsole cache, material caches

Audit P1/P2/F1/U1/P4, all in one task since they're independent small fixes across the same handful of files
and share one verification pass (3 back-to-back games, material/mesh counts sampled between them).

**Files:**
- Modify: `Assets/HexWars/Presentation/TokenStore.cs` (HP bars built once + `AddHull` single static material)
- Modify: `Assets/HexWars/Presentation/UnitTooltip.cs` (cache + char-loop line count)
- Modify: `Assets/HexWars/Presentation/EventConsole.cs` (cached GUIStyles/strings, portrait skip)
- Modify: `Assets/HexWars/Presentation/ActionPresenter.cs` (projectile tier materials cached)
- Modify: `Assets/HexWars/Presentation/ExplosionFx.cs` (flash via MaterialPropertyBlock, debris cached by tint)

**Interfaces:**
- Consumes: `BoardRenderer.UnlitColorMat`/`BlackMat` (unchanged signatures).
- Produces: no new public surface — every change here is an internal allocation fix. The one shape other
  tasks must not break: `TokenStore.RefreshToken` still calls a method named `RefreshHpBar(Transform, int, int)`
  and `AddHull(GameObject, float, float)` with the same signatures (their bodies change, callers don't).

- [ ] **Step 1: `TokenStore` — HP bars built once per token**

Read the current file (already quoted in full above during drafting — 213 lines). This is the audit's
sharpest leak: `RefreshHpBar` destroys and rebuilds two quads (and `MakeBarQuad` calls
`_board.UnlitColorMat(c)`, which allocates a brand-new `Material` — confirmed by reading `BoardRenderer.cs`:
`UnlitColor(Color c)` has no cache, `internal Material UnlitColorMat(Color c) => UnlitColor(c);`) on **every**
`Sync()` call, for every unit, whether its HP changed or not.

Add a small holder class (top of the file, inside the `HexWars.Presentation` namespace, above
`TokenStore`):

```csharp
    /// <summary>Persistent refs to one token's HP bar geometry, so RefreshHpBar can scale/position/tint
    /// in place instead of destroying and rebuilding two quads (and two Materials) every sync.</summary>
    sealed class HpBarRefs : MonoBehaviour
    {
        public Transform Fill;
        public MeshRenderer FillRenderer;
        public float BarWidth;
    }

    [RequireComponent(typeof(BoardRenderer))]
    public sealed class TokenStore : MonoBehaviour
```

(this replaces the existing `[RequireComponent(typeof(BoardRenderer))] public sealed class TokenStore :
MonoBehaviour` line — the attribute stays, `HpBarRefs` is inserted immediately above it.)

Add two fields near the top of `TokenStore` (alongside `_units`/`_generators`):

```csharp
        BoardRenderer _board;
        Transform _root;
        readonly Dictionary<int, GameObject> _units = new Dictionary<int, GameObject>();
        readonly Dictionary<int, GameObject> _generators = new Dictionary<int, GameObject>();

        static Material _hpBarMat;                 // ONE material for every background+fill quad, ever
        MaterialPropertyBlock _mpb;                 // per-renderer color override — reused across calls
```

Replace the HP-bar construction tail of `BuildToken`:

```csharp
            var bar = new GameObject("HpBar");
            bar.transform.SetParent(token.transform, false);
            bar.transform.localPosition = new Vector3(0f, 0.62f, 0f);
            bar.AddComponent<Billboard>();
            return token;
        }
```

with:

```csharp
            var bar = new GameObject("HpBar");
            bar.transform.SetParent(token.transform, false);
            bar.transform.localPosition = new Vector3(0f, 0.62f, 0f);
            bar.AddComponent<Billboard>();

            float hpBarW = _board.HexSize * 0.85f;
            _mpb = _mpb ?? new MaterialPropertyBlock();
            MakeBarQuad(bar.transform, 0f, 0f, hpBarW, 0.16f, new Color(0.18f, 0.03f, 0.03f)); // background — set once, never touched again
            var fillRenderer = MakeBarQuad(bar.transform, 0f, -0.01f, hpBarW, 0.11f, new Color(0.25f, 0.85f, 0.25f)); // placeholder tint — RefreshHpBar (called immediately after BuildToken) paints the real fraction/color
            var refs = bar.AddComponent<HpBarRefs>();
            refs.Fill = fillRenderer.transform;
            refs.FillRenderer = fillRenderer;
            refs.BarWidth = hpBarW;

            return token;
        }
```

Replace `RefreshHpBar` entirely:

```csharp
        void RefreshHpBar(Transform bar, int cur, int max)
        {
            for (int i = bar.childCount - 1; i >= 0; i--) Destroy(bar.GetChild(i).gameObject);
            float frac = max <= 0 ? 0f : Mathf.Clamp01((float)cur / max);
            float barW = _board.HexSize * 0.85f;
            MakeBarQuad(bar, 0f, 0f, barW, 0.16f, new Color(0.18f, 0.03f, 0.03f));
            float fw = Mathf.Max(0.001f, barW * frac);
            float fx = -barW * 0.5f + fw * 0.5f;
            var fill = Color.Lerp(new Color(0.85f, 0.2f, 0.12f), new Color(0.25f, 0.85f, 0.25f), frac);
            MakeBarQuad(bar, fx, -0.01f, fw, 0.11f, fill);
        }
```

with:

```csharp
        void RefreshHpBar(Transform bar, int cur, int max)
        {
            var refs = bar.GetComponent<HpBarRefs>();
            if (refs == null) return; // built by BuildToken; defensive only

            float frac = max <= 0 ? 0f : Mathf.Clamp01((float)cur / max);
            float fw = Mathf.Max(0.001f, refs.BarWidth * frac);
            float fx = -refs.BarWidth * 0.5f + fw * 0.5f;
            refs.Fill.localPosition = new Vector3(fx, -0.01f, 0f);
            refs.Fill.localScale = new Vector3(fw, 0.11f, 1f);

            var color = Color.Lerp(new Color(0.85f, 0.2f, 0.12f), new Color(0.25f, 0.85f, 0.25f), frac);
            TintQuad(refs.FillRenderer, color);
        }
```

Replace `MakeBarQuad` (signature changes — returns the `MeshRenderer` and applies its initial tint via the
property block instead of a per-call material):

```csharp
        void MakeBarQuad(Transform parent, float x, float zTowardCam, float w, float h, Color c)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = "Bar";
            DestroyImmediate(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            q.transform.localPosition = new Vector3(x, 0f, zTowardCam);
            q.transform.localScale = new Vector3(w, h, 1f);
            var mr = q.GetComponent<MeshRenderer>();
            mr.sharedMaterial = _board.UnlitColorMat(c);
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }
```

with:

```csharp
        MeshRenderer MakeBarQuad(Transform parent, float x, float zTowardCam, float w, float h, Color c)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = "Bar";
            DestroyImmediate(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            q.transform.localPosition = new Vector3(x, 0f, zTowardCam);
            q.transform.localScale = new Vector3(w, h, 1f);
            var mr = q.GetComponent<MeshRenderer>();
            mr.sharedMaterial = HpBarMaterial();
            mr.shadowCastingMode = ShadowCastingMode.Off;
            TintQuad(mr, c);
            return mr;
        }

        Material HpBarMaterial()
        {
            if (_hpBarMat == null) _hpBarMat = _board.UnlitColorMat(Color.white); // base color unused — every
                                                                                   // renderer tints via its own property block
            return _hpBarMat;
        }

        /// <summary>Per-renderer color override on the ONE shared HP-bar material — many bars can differ
        /// without a Material instance each (that was audit P1: two new Materials per unit per sync).</summary>
        void TintQuad(MeshRenderer mr, Color c)
        {
            _mpb.Clear();
            if (_hpBarMat.HasProperty("_BaseColor")) _mpb.SetColor("_BaseColor", c);
            _mpb.SetColor("_Color", c);
            mr.SetPropertyBlock(_mpb);
        }
```

- [ ] **Step 2: `TokenStore.AddHull` — one static hull material**

Replace:

```csharp
        void AddHull(GameObject host, float xz, float y)
        {
            var hull = new GameObject("Outline");
            hull.transform.SetParent(host.transform, false);
            hull.transform.localScale = new Vector3(xz, y, xz);
            hull.AddComponent<MeshFilter>().sharedMesh = host.GetComponent<MeshFilter>().sharedMesh;
            var mr = hull.AddComponent<MeshRenderer>();
            var m = new Material(_board.BlackMat);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 1f);
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }
```

with:

```csharp
        static Material _hullMat;

        void AddHull(GameObject host, float xz, float y)
        {
            var hull = new GameObject("Outline");
            hull.transform.SetParent(host.transform, false);
            hull.transform.localScale = new Vector3(xz, y, xz);
            hull.AddComponent<MeshFilter>().sharedMesh = host.GetComponent<MeshFilter>().sharedMesh;
            var mr = hull.AddComponent<MeshRenderer>();
            if (_hullMat == null)
            {
                _hullMat = new Material(_board.BlackMat);
                if (_hullMat.HasProperty("_Cull")) _hullMat.SetFloat("_Cull", 1f);
            }
            mr.sharedMaterial = _hullMat;
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }
```

(every token AND every generator pylon calls `AddHull` — both now share the one material instead of each
minting its own copy of `_board.BlackMat`.)

- [ ] **Step 3: `UnitTooltip` — cache and early-out, char-loop line count**

Replace the fields + `Show(Unit, Vector2, GameState)` method:

```csharp
        public void Show(Unit unit, Vector2 screenPos) => Show(unit, screenPos, null);

        /// <summary>Docked panel: <paramref name="screenPos"/> is ignored (kept for call-site
        /// compatibility). With <paramref name="state"/>, the active player's own units also get
        /// "this turn" lines: movement/climb budget still unspent and whether the attack is ready.</summary>
        public void Show(Unit unit, Vector2 screenPos, GameState state)
        {
            string text = Format(unit, state);
            _text.text = text;
            var prt = _panel.GetComponent<RectTransform>();
            prt.sizeDelta = new Vector2(Width, LineH * text.Split('\n').Length + 16f);
            _panel.SetActive(true);
        }
```

with:

```csharp
        int _cachedUnitId = -1;
        int _cachedHp = int.MinValue;
        bool _cachedMoved, _cachedAttacked, _cachedGameOver;

        public void Show(Unit unit, Vector2 screenPos) => Show(unit, screenPos, null);

        /// <summary>Docked panel: <paramref name="screenPos"/> is ignored (kept for call-site
        /// compatibility). With <paramref name="state"/>, the active player's own units also get
        /// "this turn" lines: movement/climb budget still unspent and whether the attack is ready.
        /// Called every frame a unit is hovered/selected (UnitInputController.Update), so it early-outs
        /// when nothing that affects the text has changed (audit P2 — this used to reformat and
        /// re-Split every frame regardless).</summary>
        public void Show(Unit unit, Vector2 screenPos, GameState state)
        {
            bool moved = false, attacked = false;
            if (state != null)
            {
                foreach (var id in state.MovedUnitIds) if (id == unit.Id) { moved = true; break; }
                foreach (var id in state.AttackedUnitIds) if (id == unit.Id) { attacked = true; break; }
            }
            bool gameOver = state != null && state.IsGameOver;
            bool unchanged = _panel.activeSelf && unit.Id == _cachedUnitId && unit.CurrentHp == _cachedHp
                            && moved == _cachedMoved && attacked == _cachedAttacked && gameOver == _cachedGameOver;
            if (unchanged) return;

            _cachedUnitId = unit.Id; _cachedHp = unit.CurrentHp;
            _cachedMoved = moved; _cachedAttacked = attacked; _cachedGameOver = gameOver;

            string text = Format(unit, state);
            _text.text = text;
            var prt = _panel.GetComponent<RectTransform>();
            prt.sizeDelta = new Vector2(Width, LineH * LineCount(text) + 16f);
            _panel.SetActive(true);
        }

        static int LineCount(string s)
        {
            int n = 1;
            for (int i = 0; i < s.Length; i++) if (s[i] == '\n') n++;
            return n;
        }
```

(`Hide()` is unchanged — it deactivates the panel, and `_panel.activeSelf` being false is part of the
`unchanged` check, so the next `Show()` after a `Hide()` always redraws even if the unit/hp/flags are
identical to what was last shown.)

- [ ] **Step 4: `EventConsole` — cached GUIStyles + strings, portrait skip**

Replace the field block:

```csharp
        const int MaxLines = 30;
        static EventConsole _inst;

        readonly Queue<string> _lines = new Queue<string>();
        GameState _state;
        bool _collapsed;
```

with:

```csharp
        const int MaxLines = 30;
        static EventConsole _inst;

        readonly Queue<string> _lines = new Queue<string>();
        GameState _state;
        bool _collapsed;

        // cached rendering data, rebuilt only when Report/Clear touch it — never per-OnGUI-frame (OnGUI
        // runs every frame, including while paused, so this was audit F1/U1's allocation source)
        GUIStyle _btnStyle, _h1Style, _h2Style, _h3Style, _logStyle;
        string _joinedLog = "";
        string _headerRound = "", _headerArmies = "", _headerSettings = "";
```

Replace `Report`:

```csharp
        /// <summary>Update the scoreboard to <paramref name="cur"/> and append any event lines.</summary>
        public static void Report(GameState cur, IEnumerable<string> events)
        {
            if (_inst == null) return;
            _inst._state = cur;
            if (events != null)
                foreach (var line in events)
                {
                    _inst._lines.Enqueue(line);
                    while (_inst._lines.Count > MaxLines) _inst._lines.Dequeue();
                }
        }
```

with:

```csharp
        /// <summary>Update the scoreboard to <paramref name="cur"/> and append any event lines. The
        /// header strings and the joined log text are rebuilt HERE (on a state/event change) instead of
        /// every OnGUI frame — Report fires on game turns/actions, not 60 times a second.</summary>
        public static void Report(GameState cur, IEnumerable<string> events)
        {
            if (_inst == null) return;
            _inst._state = cur;
            if (events != null)
            {
                bool changed = false;
                foreach (var line in events)
                {
                    _inst._lines.Enqueue(line);
                    changed = true;
                    while (_inst._lines.Count > MaxLines) _inst._lines.Dequeue();
                }
                if (changed) _inst._joinedLog = string.Join("\n", _inst._lines);
            }
            if (cur != null) _inst.RebuildHeader();
        }
```

Replace `Clear`:

```csharp
        /// <summary>Reset for a new game (clears the log; scoreboard refreshes on the next Report).</summary>
        public static void Clear()
        {
            if (_inst == null) return;
            _inst._lines.Clear();
            _inst._state = null;
        }
```

with:

```csharp
        /// <summary>Reset for a new game (clears the log; scoreboard refreshes on the next Report).</summary>
        public static void Clear()
        {
            if (_inst == null) return;
            _inst._lines.Clear();
            _inst._state = null;
            _inst._joinedLog = "";
            _inst._headerRound = _inst._headerArmies = _inst._headerSettings = "";
        }

        void RebuildHeader()
        {
            int u0 = AliveUnits(PlayerId.Player0), u1 = AliveUnits(PlayerId.Player1);
            int v0 = WinCheck.Evaluate(_state, PlayerId.Player0), v1 = WinCheck.Evaluate(_state, PlayerId.Player1);

            string result = "";
            if (_state.IsGameOver)
                result = _state.Winner == null ? "  ·  DRAW"
                       : (_state.Winner == PlayerId.Player0 ? "  ·  P1 WINS" : "  ·  P2 WINS");

            _headerRound = $"Round {_state.Round}{result}";
            _headerArmies = $"<color=#6FB1FF>P1</color>  {u0}u · {v0}v       <color=#FF7B6B>P2</color>  {u1}u · {v1}v";
            string mode = _state.Config.TerritoryMode ? "Territory" : "Annihilation";
            _headerSettings = $"{mode} · {_state.Board.Tiles.Count} tiles · {_state.Config.StartingPoints} start pts";
        }
```

Replace `OnGUI`:

```csharp
        void OnGUI()
        {
            var e = Event.current;
            if (e != null && e.type == EventType.KeyDown && e.keyCode == KeyCode.BackQuote) { _collapsed = !_collapsed; e.Use(); }

            // OnGUI doesn't DPI-scale, so it's tiny on 4K — scale the whole sidebar by screen height
            // (≈2x at 2160p) and draw in 1080p-logical coordinates.
            float s = Mathf.Max(1f, Screen.height / 1080f);
            var prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(s, s, 1f));
            DrawSidebar(Screen.width / s, Screen.height / s);
            GUI.matrix = prevMatrix;
        }
```

with:

```csharp
        void OnGUI()
        {
            // portrait: at 1080p-logical scale the 430-wide sidebar covers ~86% of a 390px-wide phone
            // screen (audit F1/U1) — the combat log stays a desktop/landscape feature, per spec §4.
            if (Screen.width < Screen.height) return;

            var e = Event.current;
            if (e != null && e.type == EventType.KeyDown && e.keyCode == KeyCode.BackQuote) { _collapsed = !_collapsed; e.Use(); }

            EnsureStyles();

            // OnGUI doesn't DPI-scale, so it's tiny on 4K — scale the whole sidebar by screen height
            // (≈2x at 2160p) and draw in 1080p-logical coordinates.
            float s = Mathf.Max(1f, Screen.height / 1080f);
            var prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(s, s, 1f));
            DrawSidebar(Screen.width / s, Screen.height / s);
            GUI.matrix = prevMatrix;
        }

        /// <summary>GUIStyle construction reads GUI.skin, which is only valid inside OnGUI — built once,
        /// the first time OnGUI actually runs (audit F1/U1: these were rebuilt every frame before).</summary>
        void EnsureStyles()
        {
            if (_btnStyle != null) return;
            _btnStyle = new GUIStyle(GUI.skin.button) { fontSize = 16 };
            _h1Style = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold, richText = true };
            _h1Style.normal.textColor = Color.white;
            _h2Style = new GUIStyle(GUI.skin.label) { fontSize = 19, richText = true };
            _h2Style.normal.textColor = new Color(0.9f, 0.92f, 0.95f);
            _h3Style = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = true };
            _h3Style.normal.textColor = new Color(0.62f, 0.66f, 0.74f);
            _logStyle = new GUIStyle(GUI.skin.label)
                { fontSize = 16, alignment = TextAnchor.LowerLeft, richText = true, wordWrap = true };
            _logStyle.normal.textColor = new Color(0.92f, 0.93f, 0.96f);
        }
```

Replace the `btn` local in `DrawSidebar` and the two `GUI.Button` calls that use it:

```csharp
        // Right-edge panel: a header scoreboard + a scrolling, color-coded narration of events.
        void DrawSidebar(float w, float h)
        {
            var btn = new GUIStyle(GUI.skin.button) { fontSize = 16 };

            // collapsed: just a small re-open tab top-right (also toggle with the ` key)
            if (_collapsed)
            {
                if (GUI.Button(new Rect(w - 96f, 6f, 90f, 30f), "◀ Log", btn)) _collapsed = false;
                return;
            }
            if (_state == null && _lines.Count == 0) return;

            const float pad = 12f, panelW = 430f;
            float x = w - panelW;

            var prevColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.78f);
            GUI.DrawTexture(new Rect(x, 0f, panelW, h), Texture2D.whiteTexture);
            GUI.color = prevColor;

            if (GUI.Button(new Rect(x + panelW - 36f, 6f, 30f, 28f), "▶", btn)) { _collapsed = true; return; }
```

with:

```csharp
        // Right-edge panel: a header scoreboard + a scrolling, color-coded narration of events.
        void DrawSidebar(float w, float h)
        {
            // collapsed: just a small re-open tab top-right (also toggle with the ` key)
            if (_collapsed)
            {
                if (GUI.Button(new Rect(w - 96f, 6f, 90f, 30f), "◀ Log", _btnStyle)) _collapsed = false;
                return;
            }
            if (_state == null && _lines.Count == 0) return;

            const float pad = 12f, panelW = 430f;
            float x = w - panelW;

            var prevColor = GUI.color;
            GUI.color = new Color(0f, 0f, 0f, 0.78f);
            GUI.DrawTexture(new Rect(x, 0f, panelW, h), Texture2D.whiteTexture);
            GUI.color = prevColor;

            if (GUI.Button(new Rect(x + panelW - 36f, 6f, 30f, 28f), "▶", _btnStyle)) { _collapsed = true; return; }
```

Replace `DrawHeader`:

```csharp
        float DrawHeader(float x, float y, float w)
        {
            int u0 = AliveUnits(PlayerId.Player0), u1 = AliveUnits(PlayerId.Player1);
            int v0 = WinCheck.Evaluate(_state, PlayerId.Player0), v1 = WinCheck.Evaluate(_state, PlayerId.Player1);

            string result = "";
            if (_state.IsGameOver)
                result = _state.Winner == null ? "  ·  DRAW"
                       : (_state.Winner == PlayerId.Player0 ? "  ·  P1 WINS" : "  ·  P2 WINS");

            var h1 = new GUIStyle(GUI.skin.label) { fontSize = 26, fontStyle = FontStyle.Bold, richText = true };
            h1.normal.textColor = Color.white;
            GUI.Label(new Rect(x, y, w, 34f), $"Round {_state.Round}{result}", h1);

            var h2 = new GUIStyle(GUI.skin.label) { fontSize = 19, richText = true };
            h2.normal.textColor = new Color(0.9f, 0.92f, 0.95f);
            GUI.Label(new Rect(x, y + 36f, w, 26f),
                      $"<color=#6FB1FF>P1</color>  {u0}u · {v0}v       <color=#FF7B6B>P2</color>  {u1}u · {v1}v", h2);

            // game settings readout (derived from the state — works for the joiner too)
            string mode = _state.Config.TerritoryMode ? "Territory" : "Annihilation";
            var h3 = new GUIStyle(GUI.skin.label) { fontSize = 15, richText = true };
            h3.normal.textColor = new Color(0.62f, 0.66f, 0.74f);
            GUI.Label(new Rect(x, y + 64f, w, 22f),
                      $"{mode} · {_state.Board.Tiles.Count} tiles · {_state.Config.StartingPoints} start pts", h3);

            return y + 90f;
        }
```

with:

```csharp
        float DrawHeader(float x, float y, float w)
        {
            GUI.Label(new Rect(x, y, w, 34f), _headerRound, _h1Style);
            GUI.Label(new Rect(x, y + 36f, w, 26f), _headerArmies, _h2Style);
            GUI.Label(new Rect(x, y + 64f, w, 22f), _headerSettings, _h3Style);
            return y + 90f;
        }
```

Replace `DrawLog`:

```csharp
        void DrawLog(float x, float y, float w, float h)
        {
            var style = new GUIStyle(GUI.skin.label)
            { fontSize = 16, alignment = TextAnchor.LowerLeft, richText = true, wordWrap = true };
            style.normal.textColor = new Color(0.92f, 0.93f, 0.96f);
            GUI.Label(new Rect(x, y, w, h), string.Join("\n", _lines), style);
        }
```

with:

```csharp
        void DrawLog(float x, float y, float w, float h)
        {
            GUI.Label(new Rect(x, y, w, h), _joinedLog, _logStyle);
        }
```

- [ ] **Step 5: `ActionPresenter` — projectile tier materials cached by damage tier**

The tier is computed in `PlayAttack` from the attacker's damage (`dmg`) into one of exactly three fixed
colors. Replace:

```csharp
            int dmg = attacker.Value.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            Color projColor = dmg >= 6 ? new Color(1f, 0.3f, 0.1f)
                            : dmg >= 3 ? new Color(1f, 0.65f, 0.2f)
                                       : new Color(1f, 0.95f, 0.5f);
```

with:

```csharp
            int dmg = attacker.Value.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            int projTier = dmg >= 6 ? 2 : dmg >= 3 ? 1 : 0;
            Color projColor = ProjectileTierColors[projTier];
```

Replace the call site immediately below (still inside `PlayAttack`):

```csharp
            SoundManager.Play(SoundKind.Attack);
            _presented = true;
            _projectile = MakeProjectile(from, projScale, projColor);
```

with:

```csharp
            SoundManager.Play(SoundKind.Attack);
            _presented = true;
            _projectile = MakeProjectile(from, projScale, projTier);
```

Replace `MakeProjectile` (signature changes from `Color color` to `int tier`; only one `_projectile` is
ever alive at a time in this class, so a tier-keyed cache is safe with no cross-instance color conflict):

```csharp
        GameObject MakeProjectile(Vector3 pos, float scale, Color color)
        {
            var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(p.GetComponent<Collider>());
            p.transform.position = pos;
            p.transform.localScale = Vector3.one * scale;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            m.color = color;
            var mr = p.GetComponent<MeshRenderer>();
            mr.sharedMaterial = m;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            var trail = p.AddComponent<TrailRenderer>();
            trail.time = 0.18f;
            trail.startWidth = scale * 0.9f;
            trail.endWidth = 0f;
            trail.material = m;
            trail.startColor = color;
            trail.endColor = new Color(color.r, color.g, color.b, 0f);
            trail.numCapVertices = 2;
            return p;
        }
```

with:

```csharp
        static readonly Color[] ProjectileTierColors =
            { new Color(1f, 0.95f, 0.5f), new Color(1f, 0.65f, 0.2f), new Color(1f, 0.3f, 0.1f) };
        static readonly Dictionary<int, Material> ProjectileMats = new Dictionary<int, Material>();

        static Material ProjectileMaterial(int tier)
        {
            if (ProjectileMats.TryGetValue(tier, out var m) && m != null) return m;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            m = new Material(unlit);
            var color = ProjectileTierColors[tier];
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            m.color = color;
            ProjectileMats[tier] = m;
            return m;
        }

        GameObject MakeProjectile(Vector3 pos, float scale, int tier)
        {
            var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(p.GetComponent<Collider>());
            p.transform.position = pos;
            p.transform.localScale = Vector3.one * scale;
            var mat = ProjectileMaterial(tier);
            var color = ProjectileTierColors[tier];
            var mr = p.GetComponent<MeshRenderer>();
            mr.sharedMaterial = mat;
            mr.shadowCastingMode = UnityEngine.Rendering.ShadowCastingMode.Off;
            var trail = p.AddComponent<TrailRenderer>();
            trail.time = 0.18f;
            trail.startWidth = scale * 0.9f;
            trail.endWidth = 0f;
            trail.material = mat;         // shared tier material — TrailRenderer tints via its own
                                           // startColor/endColor (per-component), never mutates the material
            trail.startColor = color;
            trail.endColor = new Color(color.r, color.g, color.b, 0f);
            trail.numCapVertices = 2;
            return p;
        }
```

`Dictionary<int, Material>` needs `using System.Collections.Generic;` — already present in this file (used
by `_queue`/collections elsewhere? check the existing usings: `System.Collections`, `System.Collections.Generic`,
`UnityEngine`, `HexWars.Engine` — `System.Collections.Generic` is already there).

- [ ] **Step 6: `ExplosionFx` — shared flash material via MaterialPropertyBlock, debris cached by tint**

Flash color animates every frame (near-white → tint over `Duration`); a naive tint-keyed material cache
would let two concurrent same-tint explosions fight over one Material's color each frame (this genuinely
happens — `ActionPresenter.OpponentGap` is 0.25s and `PlayClaim`'s wait is 0.15s, both shorter than
`ExplosionFx.Duration` = 0.6s, so back-to-back opponent actions or claim streaks overlap explosions in
time). The fix: ONE shared static material for every flash, with per-instance color applied through a
`MaterialPropertyBlock` on that instance's own renderer — the pattern already used for HP bars in Step 1.
Debris color is constant for its whole lifetime (only scale/position/rotation animate), so it has no such
conflict — a `Dictionary<Color, Material>` keyed by exact tint is safe and sufficient.

Add `using System.Collections.Generic;` to the file's usings (currently just `UnityEngine`,
`UnityEngine.Rendering`).

Replace the field block:

```csharp
        Color _tint = new Color(1f, 0.55f, 0.12f);
        float _scale = 1f;
        bool _withDebris = true;

        float _t;
        Transform _flash;
        Material _flashMat;
        Light _light;
        Transform[] _debris;
        Vector3[] _vel;
```

with:

```csharp
        Color _tint = new Color(1f, 0.55f, 0.12f);
        float _scale = 1f;
        bool _withDebris = true;

        float _t;
        Transform _flash;
        MeshRenderer _flashRenderer;
        MaterialPropertyBlock _flashBlock;
        Light _light;
        Transform[] _debris;
        Vector3[] _vel;

        static Material _sharedFlashMat;
        static readonly Dictionary<Color, Material> DebrisMats = new Dictionary<Color, Material>();

        static Material SharedFlashMat()
        {
            if (_sharedFlashMat != null) return _sharedFlashMat;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            _sharedFlashMat = new Material(unlit);
            return _sharedFlashMat;
        }

        static Material DebrisMat(Color tint)
        {
            if (DebrisMats.TryGetValue(tint, out var m) && m != null) return m;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            m = new Material(unlit);
            SetColor(m, tint);
            DebrisMats[tint] = m;
            return m;
        }
```

Replace `Start`:

```csharp
        void Start()
        {
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");

            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(s.GetComponent<Collider>());
            s.transform.SetParent(transform, false);
            s.transform.localScale = Vector3.one * 0.3f * _scale;
            _flash = s.transform;
            _flashMat = new Material(unlit);
            SetColor(_flashMat, new Color(1f, 0.9f, 0.5f));
            var smr = s.GetComponent<MeshRenderer>();
            smr.sharedMaterial = _flashMat;
            smr.shadowCastingMode = ShadowCastingMode.Off;

            var lgo = new GameObject("Flash");
            lgo.transform.SetParent(transform, false);
            _light = lgo.AddComponent<Light>();
            _light.type = LightType.Point;
            _light.color = new Color(1f, 0.6f, 0.25f);
            _light.range = 7f * _scale;
            _light.intensity = 9f * _scale;

            int n = _withDebris ? 8 : 0;
            _debris = new Transform[n];
            _vel = new Vector3[n];
            if (n > 0)
            {
                var debrisMat = new Material(unlit);
                SetColor(debrisMat, _tint);
                for (int i = 0; i < n; i++)
                {
                    var d = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    DestroyImmediate(d.GetComponent<Collider>());
                    d.transform.SetParent(transform, false);
                    d.transform.localScale = Vector3.one * Random.Range(0.12f, 0.24f) * _scale;
                    var mr = d.GetComponent<MeshRenderer>();
                    mr.sharedMaterial = debrisMat;
                    mr.shadowCastingMode = ShadowCastingMode.Off;
                    _debris[i] = d.transform;
                    float ang = i / (float)n * Mathf.PI * 2f;
                    _vel[i] = new Vector3(Mathf.Cos(ang), Random.Range(1.2f, 2.2f), Mathf.Sin(ang)) * Random.Range(2.5f, 4.5f) * _scale;
                }
            }
        }
```

with:

```csharp
        void Start()
        {
            var s = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(s.GetComponent<Collider>());
            s.transform.SetParent(transform, false);
            s.transform.localScale = Vector3.one * 0.3f * _scale;
            _flash = s.transform;
            var smr = s.GetComponent<MeshRenderer>();
            smr.sharedMaterial = SharedFlashMat();   // one material for every explosion in the game — per-
            smr.shadowCastingMode = ShadowCastingMode.Off;  // instance color goes through a property block below
            _flashRenderer = smr;
            _flashBlock = new MaterialPropertyBlock();
            SetFlashColor(new Color(1f, 0.9f, 0.5f));

            var lgo = new GameObject("Flash");
            lgo.transform.SetParent(transform, false);
            _light = lgo.AddComponent<Light>();
            _light.type = LightType.Point;
            _light.color = new Color(1f, 0.6f, 0.25f);
            _light.range = 7f * _scale;
            _light.intensity = 9f * _scale;

            int n = _withDebris ? 8 : 0;
            _debris = new Transform[n];
            _vel = new Vector3[n];
            if (n > 0)
            {
                var debrisMat = DebrisMat(_tint);    // constant color for this instance's whole lifetime —
                                                      // safe to share across every same-tint explosion
                for (int i = 0; i < n; i++)
                {
                    var d = GameObject.CreatePrimitive(PrimitiveType.Cube);
                    DestroyImmediate(d.GetComponent<Collider>());
                    d.transform.SetParent(transform, false);
                    d.transform.localScale = Vector3.one * Random.Range(0.12f, 0.24f) * _scale;
                    var mr = d.GetComponent<MeshRenderer>();
                    mr.sharedMaterial = debrisMat;
                    mr.shadowCastingMode = ShadowCastingMode.Off;
                    _debris[i] = d.transform;
                    float ang = i / (float)n * Mathf.PI * 2f;
                    _vel[i] = new Vector3(Mathf.Cos(ang), Random.Range(1.2f, 2.2f), Mathf.Sin(ang)) * Random.Range(2.5f, 4.5f) * _scale;
                }
            }
        }

        void SetFlashColor(Color c)
        {
            _flashBlock.Clear();
            if (_flashRenderer.sharedMaterial.HasProperty("_BaseColor")) _flashBlock.SetColor("_BaseColor", c);
            _flashBlock.SetColor("_Color", c);
            _flashRenderer.SetPropertyBlock(_flashBlock);
        }
```

Replace the one `SetColor(_flashMat, ...)` call in `Update`:

```csharp
            float peak = 2.4f * _scale;
            float s = p < 0.35f ? Mathf.Lerp(0.3f * _scale, peak, p / 0.35f) : Mathf.Lerp(peak, 0f, (p - 0.35f) / 0.65f);
            _flash.localScale = Vector3.one * s;
            SetColor(_flashMat, Color.Lerp(new Color(1f, 0.95f, 0.6f), _tint, p));
            _light.intensity = Mathf.Lerp(9f * _scale, 0f, p);
```

with:

```csharp
            float peak = 2.4f * _scale;
            float s = p < 0.35f ? Mathf.Lerp(0.3f * _scale, peak, p / 0.35f) : Mathf.Lerp(peak, 0f, (p - 0.35f) / 0.65f);
            _flash.localScale = Vector3.one * s;
            SetFlashColor(Color.Lerp(new Color(1f, 0.95f, 0.6f), _tint, p));
            _light.intensity = Mathf.Lerp(9f * _scale, 0f, p);
```

(the static `SetColor(Material, Color)` helper at the bottom of the file stays — `DebrisMat` still uses it.)

- [ ] **Step 7: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 8: Visual regression — HP bars pixel-comparable before/after**

Before making the Step 1 edit (or on a clean `git stash` of just this task), enter Play Mode, let a hotseat
game seed its demo units, screenshot a unit's HP bar close-up via `capture_scene_object` with the unit's
token path. After the edit, repeat the identical screenshot (same unit, same HP, same camera framing —
reuse `NewGame()`'s deterministic seed). Compare: background/fill size, position, and color must be pixel-
identical; only the underlying allocation pattern changed.

- [ ] **Step 9: Session-longevity proof — 3 back-to-back games, material/mesh counts plateau**

`execute_script`, one call per game plus a final sampling call:

```csharp
var counts = new System.Collections.Generic.List<string>();
var game = UnityEngine.Object.FindAnyObjectByType<HexWars.Presentation.GameBootstrap>();
for (int i = 0; i < 3; i++)
{
    game.StartLocalGame(new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Annihilation, 9, 7, 0, 7 + i), true);
    int mats = Resources.FindObjectsOfTypeAll<UnityEngine.Material>().Length;
    int meshes = Resources.FindObjectsOfTypeAll<UnityEngine.Mesh>().Length;
    counts.Add($"game{i}: mats={mats} meshes={meshes}");
}
return string.Join(" | ", counts);
```

Expected: `mats`/`meshes` counts for game 1→2→3 must plateau (game 2 and game 3 counts equal, or within a
small constant of game 1 — NOT climbing linearly with game count). A climbing count means something is
still leaking; if so, stop and diagnose before continuing (don't paper over it in verification).

- [ ] **Step 10: Tooltip content check**

`execute_script`: hover-simulate by calling `UnitInputController`'s tooltip path directly, or simpler —
select a unit in play mode (script a click via `UnitTooltip.Show` isn't directly reachable without a real
unit reference; instead find a live `Unit` from `GameBootstrap.State` and call
`FindAnyObjectByType<UnitTooltip>().Show(unit, Vector2.zero, game.State)` twice in a row with identical
state) and confirm the second call is a no-op (same text) while a THIRD call after `TryApply`-ing a move
(HP unchanged, moved flag now true) DOES update — read `GetComponentInChildren<UnityEngine.UI.Text>().text`
before/after and assert it changed exactly when the moved flag changed, matching the pre-cache behavior.

- [ ] **Step 11: Commit**

```bash
git add Assets/HexWars/Presentation/TokenStore.cs Assets/HexWars/Presentation/UnitTooltip.cs Assets/HexWars/Presentation/EventConsole.cs Assets/HexWars/Presentation/ActionPresenter.cs Assets/HexWars/Presentation/ExplosionFx.cs
git commit -m "perf(session): fix per-sync/per-frame allocators - HP bars built once with MaterialPropertyBlock tinting, tooltip caches unchanged content, EventConsole caches styles/strings and hides in portrait, projectile/explosion/hull materials cached"
```

---

### Task 8: Portrait pass — every screen usable at 390×844

**Files:**
- Create: `Assets/HexWars/Editor/PortraitGameView.cs` (verification utility — the mechanism validated below)
- Modify: `Assets/HexWars/Presentation/SetupForm.cs`
- Modify: `Assets/HexWars/Presentation/GameBrowser.cs`
- Modify: `Assets/HexWars/Presentation/TitleScreen.cs`
- Modify: `Assets/HexWars/Presentation/GameHud.cs`
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs`
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs`
- Modify: `Assets/HexWars/Presentation/UnitTooltip.cs`
- Modify: `Assets/HexWars/Presentation/GameOverBanner.cs`
- Modify: `Assets/HexWars/Presentation/Toast.cs`

**Interfaces:**
- Consumes: `UiKit.Canvas`/`SetRect` (unchanged).
- Produces: `HexWars.EditorTools.PortraitGameView.Enter()` / `.Exit()` — editor-only, called from
  `execute_script` during verification (this task and every later one that needs a portrait screenshot).

**Finding from drafting this task (verified live in this project, not theoretical — see the Global
Constraints section above for the mechanism): at 390×844 with the current shared `UiKit.Canvas` settings
(`referenceResolution = 1600×900`, `matchWidthOrHeight = 0.5`), the rendered `ScreenSpaceOverlay` canvas
reports ≈815×1765 canvas-space units — generous WIDTH-wise (most fixed-width panels numerically fit without
clipping) but at a low ≈0.478 px/canvas-unit scale factor, meaning `SizeBody` (16pt) text renders under
8 physical pixels tall. The concrete, verifiable clipping bugs this task fixes are real (SetupForm's status
label, HUD's long single-line banner string) but they are the minority case; the dominant portrait problem
at this project's current CanvasScaler settings is legibility, not overflow. Changing `matchWidthOrHeight`
globally would fix that more thoroughly but risks every landscape/desktop screen (all of them share one
`UiKit.Canvas` call) and is explicitly NOT what this task's fix list asks for (per-panel clamps, "same
docking, no layout redesign"). **This is flagged, not silently resolved** — the screenshot gate in Step 9
will make the legibility question visible; if it reads as genuinely unusable rather than merely dense, a
follow-up task to revisit `matchWidthOrHeight` (or a portrait-specific override) is the correct next step,
out of scope here.

- [ ] **Step 1: `PortraitGameView` — the verification utility (validated live while drafting this plan)**

Create `Assets/HexWars/Editor/PortraitGameView.cs`:

```csharp
using System;
using System.Reflection;
using UnityEditor;
using UnityEngine;

namespace HexWars.EditorTools
{
    /// <summary>
    /// Verification helper: forces the Game view to a fixed 390×844 logical resolution (iPhone-ish CSS
    /// px) so Screen.width/height — and therefore every CanvasScaler in the game — matches what a phone
    /// in portrait actually sees. Outside Play Mode, Screen.width/height don't track the Game view size
    /// at all (confirmed empirically), so this only matters, and is only meant to be called, while
    /// playing. GameViewSizes has no public API for adding/selecting a custom size, hence the reflection.
    /// Resizing the Game view's EditorWindow via .position does NOT work reliably (a docked window
    /// ignores explicit position edits) — this uses GameViewSizes' own fixed-resolution selection instead,
    /// which was confirmed to force Screen.width/height exactly to the target.
    /// </summary>
    public static class PortraitGameView
    {
        const string Label = "HexWars Portrait";
        const int W = 390, H = 844;
        const int RestoreIndex = 3; // "Full HD (1920x1080)" in the default size list

        static object _gvs, _group;

        public static string Enter()
        {
            var window = GameViewWindow();
            int idx = FindOrAddSize(out object size);
            SizeSelectionCallback(window, idx, size);
            window.Repaint();
            return $"Screen={Screen.width}x{Screen.height}";
        }

        /// <summary>Back to a normal desktop size when the sweep is done.</summary>
        public static string Exit()
        {
            var window = GameViewWindow();
            object size = GetGameViewSize(RestoreIndex);
            SizeSelectionCallback(window, RestoreIndex, size);
            window.Repaint();
            return $"Screen={Screen.width}x{Screen.height}";
        }

        static void EnsureGroup()
        {
            if (_group != null) return;
            var asm = typeof(Editor).Assembly;
            var sizesType = asm.GetType("UnityEditor.GameViewSizes");
            var single = typeof(ScriptableSingleton<>).MakeGenericType(sizesType);
            _gvs = single.GetProperty("instance", BindingFlags.Public | BindingFlags.Static).GetValue(null, null);
            var currentGroupType = sizesType.GetProperty("currentGroupType").GetValue(_gvs, null);
            _group = sizesType.GetMethod("GetGroup").Invoke(_gvs, new object[] { currentGroupType });
        }

        static object GetGameViewSize(int index)
        {
            EnsureGroup();
            return _group.GetType().GetMethod("GetGameViewSize").Invoke(_group, new object[] { index });
        }

        static int FindOrAddSize(out object size)
        {
            EnsureGroup();
            var groupType = _group.GetType();
            var texts = (string[])groupType.GetMethod("GetDisplayTexts").Invoke(_group, null);
            for (int i = 0; i < texts.Length; i++)
                if (texts[i].IndexOf(Label, StringComparison.Ordinal) >= 0)
                { size = GetGameViewSize(i); return i; }

            var asm = typeof(Editor).Assembly;
            var gameViewSizeType = asm.GetType("UnityEditor.GameViewSize");
            var sizeTypeEnum = asm.GetType("UnityEditor.GameViewSizeType");
            object fixedResolution = Enum.Parse(sizeTypeEnum, "FixedResolution");
            var ctor = gameViewSizeType.GetConstructor(new[] { sizeTypeEnum, typeof(int), typeof(int), typeof(string) });
            size = ctor.Invoke(new object[] { fixedResolution, W, H, Label });
            groupType.GetMethod("AddCustomSize").Invoke(_group, new object[] { size });

            texts = (string[])groupType.GetMethod("GetDisplayTexts").Invoke(_group, null);
            for (int i = 0; i < texts.Length; i++)
                if (texts[i].IndexOf(Label, StringComparison.Ordinal) >= 0) return i;
            return texts.Length - 1; // AddCustomSize appends — fallback only if the scan above ever misses
        }

        static EditorWindow GameViewWindow()
        {
            var gameViewType = typeof(Editor).Assembly.GetType("UnityEditor.GameView");
            return EditorWindow.GetWindow(gameViewType);
        }

        static void SizeSelectionCallback(EditorWindow window, int index, object size)
        {
            var method = window.GetType().GetMethod("SizeSelectionCallback",
                BindingFlags.Instance | BindingFlags.NonPublic | BindingFlags.Public);
            method.Invoke(window, new object[] { index, size });
        }
    }
}
```

- [ ] **Step 2: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 3: `SetupForm` — clamp card + status label to available width**

Read the current file (quoted in full above during drafting). Replace the start of `Build()`:

```csharp
        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("SetupCanvas", UiKit.OrderMenu, transform);

            _form = UiKit.Panel(_canvasGo.transform, "Form", UiKit.Surface).gameObject;
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(700f, 640f);
            frt.anchoredPosition = Vector2.zero;
```

with:

```csharp
        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("SetupCanvas", UiKit.OrderMenu, transform);

            float formW = Mathf.Min(700f, AvailWidth() - 40f); // GameRules' clamp pattern — a fixed
                                                                // 700-wide card must not overflow a narrow canvas
            _form = UiKit.Panel(_canvasGo.transform, "Form", UiKit.Surface).gameObject;
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(formW, 640f);
            frt.anchoredPosition = Vector2.zero;
```

Replace the `_status` construction (the concrete overflow this task actually catches: a fixed 1100-wide
label — wider than even the generous ≈815-unit portrait canvas — used for the room-code waiting screen):

```csharp
            _status = UiKit.Label(_canvasGo.transform, "", 0f, 0f, 1100f, 160f, UiKit.SizeHeading, TextAnchor.MiddleCenter);
            var srt = _status.GetComponent<RectTransform>();
            srt.anchorMin = srt.anchorMax = new Vector2(0.5f, 0.5f);
            srt.pivot = new Vector2(0.5f, 0.5f);
            srt.anchoredPosition = new Vector2(0f, 30f);

            BuildArmyPopup();
        }
```

with:

```csharp
            _status = UiKit.Label(_canvasGo.transform, "", 0f, 0f, Mathf.Min(1100f, AvailWidth() - 40f), 160f,
                                  UiKit.SizeHeading, TextAnchor.MiddleCenter);
            _status.horizontalOverflow = HorizontalWrapMode.Wrap; // was Overflow — the clamp above only
                                                                   // helps once long lines can actually wrap
            var srt = _status.GetComponent<RectTransform>();
            srt.anchorMin = srt.anchorMax = new Vector2(0.5f, 0.5f);
            srt.pivot = new Vector2(0.5f, 0.5f);
            srt.anchoredPosition = new Vector2(0f, 30f);

            BuildArmyPopup();
        }

        /// <summary>The canvas's actual rendered width (GameRules' clamp pattern) — a fixed layout must
        /// not overflow a narrow/portrait screen.</summary>
        float AvailWidth()
        {
            var rt = _canvasGo.GetComponent<RectTransform>();
            return rt != null && rt.rect.width > 0f ? rt.rect.width : 1200f;
        }
```

Apply the same clamp to the army popup card. Replace:

```csharp
            var card = UiKit.Panel(_armyPanel.transform, "Card", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(700f, 430f);
            crt.anchoredPosition = Vector2.zero;
```

with:

```csharp
            var card = UiKit.Panel(_armyPanel.transform, "Card", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(Mathf.Min(700f, AvailWidth() - 40f), 430f);
            crt.anchoredPosition = Vector2.zero;
```

- [ ] **Step 4: `GameBrowser` — clamp panel + rows to available width**

Replace `Build()`:

```csharp
        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("BrowserCanvas", UiKit.OrderMenu, transform);

            var panel = UiKit.Panel(_canvasGo.transform, "Panel", UiKit.Surface).gameObject;
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.pivot = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(760f, 640f);
            prt.anchoredPosition = Vector2.zero;

            UiKit.Label(panel.transform, "Open Games", 0f, -24f, 760f, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(panel.transform, "Back", -330f, -26f, 90f, 34f,
                         () => { Close(); TitleScreen.Reopen(_game); }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            UiKit.Button(panel.transform, "Refresh", 320f, -26f, 110f, 34f,
                         () => { StopAllCoroutines(); StartCoroutine(PollLoop()); }, // restart: fetch now, resume cadence — never two in-flight fetches
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);

            _status = UiKit.Label(panel.transform, "Loading…", 0f, -70f, 700f, 26f,
                                  UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            var lrt = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(lrt, 0f, -100f, 720f, 520f);
            _listRoot = listGo.transform;
        }
```

with:

```csharp
        float _panelW;

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("BrowserCanvas", UiKit.OrderMenu, transform);
            _panelW = Mathf.Min(760f, AvailWidth() - 40f); // GameRules' clamp pattern — unchanged (700/720
                                                            // math below) whenever the canvas is wide enough,
                                                            // same as it always was at desktop/landscape widths

            var panel = UiKit.Panel(_canvasGo.transform, "Panel", UiKit.Surface).gameObject;
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.pivot = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(_panelW, 640f);
            prt.anchoredPosition = Vector2.zero;

            UiKit.Label(panel.transform, "Open Games", 0f, -24f, _panelW, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(panel.transform, "Back", -_panelW * 0.5f + 55f, -26f, 90f, 34f,
                         () => { Close(); TitleScreen.Reopen(_game); }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            UiKit.Button(panel.transform, "Refresh", _panelW * 0.5f - 60f, -26f, 110f, 34f,
                         () => { StopAllCoroutines(); StartCoroutine(PollLoop()); }, // restart: fetch now, resume cadence — never two in-flight fetches
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);

            _status = UiKit.Label(panel.transform, "Loading…", 0f, -70f, _panelW - 60f, 26f,
                                  UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            var lrt = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(lrt, 0f, -100f, _panelW - 40f, 520f);
            _listRoot = listGo.transform;
        }

        float AvailWidth()
        {
            var rt = _canvasGo.GetComponent<RectTransform>();
            return rt != null && rt.rect.width > 0f ? rt.rect.width : 1200f;
        }
```

(at desktop/landscape widths `_panelW` still resolves to exactly `760f` — the row/list math below is
untouched in that case, only the WIDTHS below are re-derived from `_panelW` instead of the old literals
`700f`/`720f`, which are the exact same numbers when `_panelW == 760f`.)

Replace `BuildRow`'s width literals — the row button and detail card were `700f`, matching `_panelW - 60f`
at desktop widths:

```csharp
        float BuildRow(GameDto g, float y, bool expanded)
        {
            string age = g.ageSeconds < 60 ? $"{g.ageSeconds}s" : $"{g.ageSeconds / 60}m";
            string summary = $"{g.code}   ·   {g.mode} · {g.width}×{g.height}{(g.fog ? " · Fog" : "")}" +
                             $" · {PaceText(g.pace)} · {g.army} units · {age} ago";
            var code = g.code;
            var row = UiKit.Button(_listRoot, summary, 0f, y, 700f, 42f, () =>
            {
                _expandedCode = _expandedCode == code ? null : code;
                Rebuild();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            var rowText = row.GetComponentInChildren<Text>();
            rowText.alignment = TextAnchor.MiddleLeft;
            var trt = rowText.GetComponent<RectTransform>();
            trt.anchoredPosition = new Vector2(14f, trt.anchoredPosition.y);
            y -= 46f;

            if (expanded)
            {
                var card = UiKit.Panel(_listRoot, "Detail", new Color(0.09f, 0.11f, 0.18f, 1f)).gameObject;
                UiKit.SetRect(card.GetComponent<RectTransform>(), 0f, y, 700f, 96f);
                UiKit.Label(card.transform,
                            $"Mode {g.mode}    Map {g.width}×{g.height}    Fog {(g.fog ? "on" : "off")}\n" +
                            $"Pace {PaceText(g.pace)}    Army {g.army} units    Waiting {age}",
                            -80f, -12f, 520f, 72f, UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextDim);
                UiKit.Button(card.transform, "Join", 260f, -26f, 140f, 44f, () =>
                {
                    _status.text = $"Joining {g.code}…";
                    _game.StartNetGame(g.code, null);   // seat+start arrive via the normal net path
                }, UiKit.ButtonStyle.Cta, UiKit.SizeBody + 2);
                y -= 102f;
            }
            return y;
        }
```

with:

```csharp
        float BuildRow(GameDto g, float y, bool expanded)
        {
            float rowW = _panelW - 60f; // == 700f at desktop widths, same as the literal it replaces
            string age = g.ageSeconds < 60 ? $"{g.ageSeconds}s" : $"{g.ageSeconds / 60}m";
            string summary = $"{g.code}   ·   {g.mode} · {g.width}×{g.height}{(g.fog ? " · Fog" : "")}" +
                             $" · {PaceText(g.pace)} · {g.army} units · {age} ago";
            var code = g.code;
            var row = UiKit.Button(_listRoot, summary, 0f, y, rowW, 42f, () =>
            {
                _expandedCode = _expandedCode == code ? null : code;
                Rebuild();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            var rowText = row.GetComponentInChildren<Text>();
            rowText.alignment = TextAnchor.MiddleLeft;
            var trt = rowText.GetComponent<RectTransform>();
            trt.anchoredPosition = new Vector2(14f, trt.anchoredPosition.y);
            y -= 46f;

            if (expanded)
            {
                var card = UiKit.Panel(_listRoot, "Detail", new Color(0.09f, 0.11f, 0.18f, 1f)).gameObject;
                UiKit.SetRect(card.GetComponent<RectTransform>(), 0f, y, rowW, 96f);
                UiKit.Label(card.transform,
                            $"Mode {g.mode}    Map {g.width}×{g.height}    Fog {(g.fog ? "on" : "off")}\n" +
                            $"Pace {PaceText(g.pace)}    Army {g.army} units    Waiting {age}",
                            -80f, -12f, 520f, 72f, UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextDim);
                UiKit.Button(card.transform, "Join", rowW * 0.5f - 90f, -26f, 140f, 44f, () =>
                {
                    _status.text = $"Joining {g.code}…";
                    _game.StartNetGame(g.code, null);   // seat+start arrive via the normal net path
                }, UiKit.ButtonStyle.Cta, UiKit.SizeBody + 2);
                y -= 102f;
            }
            return y;
        }
```

Also apply the same clamp to the empty-state "Host Game" button's row — no width change needed there (it's
already narrower than any clamp would trigger), so `Rebuild()`'s empty branch is untouched here (Task 13
adds the "Play vs AI" button beside it).

- [ ] **Step 5: `TitleScreen` — menu column fits**

Replace the column sizing in `Build()`:

```csharp
            var crt = col.AddComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(520f, 640f);
            crt.anchoredPosition = new Vector2(0f, 20f);
```

with:

```csharp
            var crt = col.AddComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            var canvasRt = _canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            crt.sizeDelta = new Vector2(Mathf.Min(520f, availW - 40f), 640f);
            crt.anchoredPosition = new Vector2(0f, 20f);
```

(the buttons inside are 380 wide, well under any clamp this triggers — only the column plate/backdrop
width changes.)

- [ ] **Step 6: `GameHud` — banner drops the per-player stat block on narrow, keeps turn + points**

The long single-line status string (up to ~120 chars once round/points/barracks/armies/done are all
appended) is the confirmed real overflow: it wraps inside the fixed 46px banner strip and gets vertically
truncated. Replace the text-building block in `Refresh()`:

```csharp
            _banner.text = s.Config.TerritoryMode
                ? $"P{who}'s turn{pace}{done}{armies}     Round {s.Round}     " +
                  $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}"
                : $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")}){pace}{done}{armies}     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}";
            if (_game.Reconnecting) _banner.text = "⚠ Connection lost — reconnecting…     " + _banner.text;

            if (_endBtn != null) _endBtn.color = done.Length > 0 ? EndTurnUrge : EndTurnIdle;
```

with:

```csharp
            bool narrow = Screen.width < 700; // portrait phones; the full-detail banner only fits landscape
            _banner.text = s.Config.TerritoryMode
                ? (narrow
                    ? $"P{who}'s turn{pace}     Round {s.Round}     {p.Points} pts"
                    : $"P{who}'s turn{pace}{done}{armies}     Round {s.Round}     " +
                      $"P1 {Stat(s, PlayerId.Player0)}   |   P2 {Stat(s, PlayerId.Player1)}")
                : (narrow
                    ? $"Player {who}'s turn{pace}     {p.Points} pts     Round {s.Round}"
                    : $"Player {who}'s turn  (move {(p0 ? "cyan" : "red")}){pace}{done}{armies}     {p.Points} pts     Round {s.Round}     Barracks {p.Barracks.Count}");
            if (_game.Reconnecting) _banner.text = "⚠ Reconnecting…     " + _banner.text;

            if (_endBtn != null) _endBtn.color = done.Length > 0 ? EndTurnUrge : EndTurnIdle;
```

(shortened the reconnect prefix too on the same principle — the full "Connection lost — reconnecting…"
wording still fires from the Toast in Task 6, which isn't width-constrained the same way.)

- [ ] **Step 7: `BarracksPanel` / `DesignPanel` — clamp width, defensive Y clamp**

These are narrow fixed panels (230/270 units) that fit comfortably even in the tightest canvas this task
targets — the clamp here is defensive (a real bug only on a canvas narrower than ~300 units, not reachable
at 390×844 with the current CanvasScaler settings, confirmed via the portrait canvas-rect reading in this
task's header note) but is what the task's fix list asks for, and guards future narrower targets. In
`BarracksPanel.Build()`, replace:

```csharp
            const float w = 230f;
            var panel = UiKit.Panel(canvasGo.transform, "BarracksPanel", UiKit.Surface);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(w, 420f);
            prt.anchoredPosition = new Vector2(-8f, -58f);
```

with:

```csharp
            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            float w = Mathf.Min(230f, availW - 16f);
            var panel = UiKit.Panel(canvasGo.transform, "BarracksPanel", UiKit.Surface);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(w, 420f);
            prt.anchoredPosition = new Vector2(-8f, -58f);
```

(the rest of `Build()` already reads local `w`, so row/list widths derived from it stay consistent.)

In `DesignPanel.Build()`, replace:

```csharp
            const float w = 270f, rowH = 30f, top = 58f;
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowH * 9 + 120f);
            prt.anchoredPosition = new Vector2(8f, -top);
```

with:

```csharp
            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            const float rowH = 30f, top = 58f;
            float w = Mathf.Min(270f, availW - 16f);
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowH * 9 + 120f);
            prt.anchoredPosition = new Vector2(8f, -top);
```

(`w` is `const` today — dropping `const` is required since it's now computed; every other use of `w` in
the method already reads the local, so no further changes needed there. Task 11 adds a 10th row (Name) to
this panel — that edit lands after this one and must keep reading the same `w`.)

- [ ] **Step 8: `UnitTooltip` / `GameOverBanner` / `Toast` — clamp/wrap**

`UnitTooltip`: its fixed 230-wide panel is already narrower than `BarracksPanel`'s (which it docks under)
and needs no independent width clamp — it inherits safety from Barracks fitting. No source change here;
this is intentionally a no-op, documented so a reviewer doesn't wonder why the task list mentions tooltip
but no diff appears for it in this step.

`GameOverBanner`: the band's title/subtitle currently use `HorizontalWrapMode.Overflow`, which is fine at
desktop widths (the band is `sizeDelta = (0, 200)` — stretches full-width) but a 46pt "GAME OVER" title
can be too wide for a narrow band's readable area. Replace the `Text` helper:

```csharp
        static void Text(Transform parent, string s, Font font, int size, FontStyle style, Vector2 pos)
        {
            var go = new GameObject("Text");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.text = s;
            t.font = font; t.fontSize = size; t.fontStyle = style;
            t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Overflow; t.raycastTarget = false;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.sizeDelta = new Vector2(1200f, 60f);
            rt.anchoredPosition = pos;
        }
```

with:

```csharp
        static void Text(Transform parent, string s, Font font, int size, FontStyle style, Vector2 pos)
        {
            var go = new GameObject("Text");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.text = s;
            t.font = font; t.fontSize = size; t.fontStyle = style;
            t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.raycastTarget = false; // was Overflow — a
                                                                                      // narrow band must wrap, not run off-canvas
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0.5f); rt.anchorMax = new Vector2(1f, 0.5f); // stretch to the band's
            rt.offsetMin = new Vector2(20f, 0f); rt.offsetMax = new Vector2(-20f, 0f);  // own width, not a fixed 1200
            rt.sizeDelta = new Vector2(0f, 60f);
            rt.anchoredPosition = pos;
        }
```

(the band itself already stretches full-canvas-width via `anchorMin=(0,0.5), anchorMax=(1,0.5)` — the text
now stretches to match it instead of assuming a fixed 1200-wide canvas.)

`Toast`: the box is a fixed 580 wide, which fits the portrait canvas (≈815 units) but add the same
defensive clamp for narrower targets. Replace in `BuildUi()`:

```csharp
            var brt = _bg.GetComponent<RectTransform>();
            brt.anchorMin = brt.anchorMax = new Vector2(0.5f, 0f);
            brt.pivot = new Vector2(0.5f, 0f);
            brt.sizeDelta = new Vector2(580f, 50f);
            brt.anchoredPosition = new Vector2(0f, 170f);
```

with:

```csharp
            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            var brt = _bg.GetComponent<RectTransform>();
            brt.anchorMin = brt.anchorMax = new Vector2(0.5f, 0f);
            brt.pivot = new Vector2(0.5f, 0f);
            brt.sizeDelta = new Vector2(Mathf.Min(580f, availW - 40f), 50f);
            brt.anchoredPosition = new Vector2(0f, 170f);
```

- [ ] **Step 9: Compile check, then the portrait screenshot sweep (the acceptance gate)**

coplay `check_compile_errors` → zero errors.

Enter Play Mode. `execute_script` with `methodName: "Enter"` on `PortraitGameView.cs` — confirm the
returned string is `Screen=390x844`. Then, ONE transition per `execute_script` call (per the coplay quirk
in Global Constraints), sweep every screen and screenshot each:

1. Title screen (default boot state, demo running behind the menu) — `capture_ui_canvas` on `TitleCanvas`.
2. `SetupForm.Open(game, SetupForm.SetupMode.Host)` → screenshot `SetupCanvas`. Then open the army popup
   (`OpenArmy` isn't public — call `GetComponent<SetupForm>()` and drive it via the same button click path,
   or just assert `_armyPanel` exists and is inactive by default; screenshotting the popup open requires a
   second call that clicks the "Army:" button) → screenshot again.
3. Click "Create Game" → waiting screen (`ShowWaiting`) → screenshot (confirms the room-code label wraps
   instead of clipping).
4. Close, open `GameBrowser.Open(game)` → screenshot the empty state, then the "no open games" `Host Game`
   button is visible and tappable within frame.
5. Close, `TitleScreen.Reopen(game)`, start a local vs-AI game (`StartLocalGame(...)`) → screenshot `HudCanvas`
   (confirm the narrow banner text — turn+points only, no wrapped/clipped second line) and `BarracksCanvas`/
   `DesignCanvas`.
6. Hover/select a unit → screenshot the tooltip panel.
7. Force a game-over (`execute_script`: apply commands until `State.IsGameOver`, or directly construct a
   trivially-won state isn't available — instead drive `AttackUnit` commands against the AI's low-HP demo
   army, or simplest: call `GameHud`'s `ShowGameOver` path indirectly by ending the match through normal
   play for a couple of turns with a stacked army setup, e.g. `GameSetup` with `armySize` weighted so one
   side is trivially eliminated) → screenshot the `GameOverBanner` band.
8. Trigger a Toast (`Toast.Show("Verification toast — portrait width check", UiKit.Danger)`) → screenshot.
9. Open the rules card (`GameRules.Show(...)`, already portrait-safe per its existing clamp) → screenshot
   as a sanity check that Task 8 didn't regress the pattern it's copying from.

Confirm every screenshot: no text run off the visible 390-wide frame, no button center further than ~195
units from screen-center (i.e., nothing poking past the left/right edge).

`execute_script` with `methodName: "Exit"` on `PortraitGameView.cs` to restore a normal Game view size.
Exit Play Mode.

- [ ] **Step 10: Commit**

```bash
git add Assets/HexWars/Editor/PortraitGameView.cs Assets/HexWars/Presentation/SetupForm.cs Assets/HexWars/Presentation/GameBrowser.cs Assets/HexWars/Presentation/TitleScreen.cs Assets/HexWars/Presentation/GameHud.cs Assets/HexWars/Presentation/BarracksPanel.cs Assets/HexWars/Presentation/DesignPanel.cs Assets/HexWars/Presentation/GameOverBanner.cs Assets/HexWars/Presentation/Toast.cs
git commit -m "feat(ui): portrait pass - every screen clamps to available canvas width at 390x844; HUD banner drops the per-player stat block on narrow; PortraitGameView verification utility"
```

---

### Task 9: Share card + PWA statics

**Files:**
- Create: `engine/generate-share-assets.ps1` (procedural generator — committed as repo tooling, re-runnable)
- Create (by running the script above): `engine/HexWars.NetServer/wwwroot/manifest.json`,
  `engine/HexWars.NetServer/wwwroot/icon-192.png`, `engine/HexWars.NetServer/wwwroot/icon-512.png`,
  `engine/HexWars.NetServer/wwwroot/favicon.png`, `engine/HexWars.NetServer/wwwroot/preview.png`
- Modify: `engine/stage-webgl-deploy.ps1`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `GET /manifest.json`, `GET /icon-192.png`, `GET /icon-512.png`, `GET /favicon.png`,
  `GET /preview.png` served as static files by the existing NetServer (it already serves `wwwroot/` root-
  level files — confirmed by reading `stage-webgl-deploy.ps1`, see Step 1 below).

- [ ] **Step 1: Confirm statics survive staging (read before writing anything)**

Read `engine/stage-webgl-deploy.ps1` (already quoted in full above during drafting — 29 lines). It deletes
and recreates only `wwwroot/Build/` (`Remove-Item (Join-Path $dst "Build") -Recurse -Force` then
`Copy-Item ... Build ...`) and overwrites `wwwroot/index.html` (`Copy-Item ... index.html $dst -Force`). It
never touches any other root-level file under `wwwroot/`. **Confirmed: files committed directly under
`engine/HexWars.NetServer/wwwroot/` (not under `Build/`) survive every redeploy untouched — no fix needed
for the "must not delete wwwroot root-level files" requirement.** This is a finding, not an assumption —
flagged here per the instruction not to silently deviate from what was asked (the task asked to "confirm/
fix"; the confirm side is what's needed).

- [ ] **Step 2: Write the procedural asset generator**

Create `engine/generate-share-assets.ps1` (validated end-to-end while drafting this plan — ran it against a
scratch directory and visually confirmed the output: a clean hex-badge icon and an on-palette 1200×630
preview card with a hex-tile motif):

```powershell
# Generates the share-card / PWA static assets under engine/HexWars.NetServer/wwwroot/:
#   manifest.json, icon-192.png, icon-512.png, favicon.png, preview.png (1200x630 og:image)
# Procedural (no external art asset), on UiKit's palette (Bg #0A0E1C, Accent #45AEFF, CtaGreen #33845C,
# TextDim #9AA3B8). Re-run any time; overwrites in place. Run from repo root or anywhere (path is
# resolved relative to this script's own location).
Add-Type -AssemblyName System.Drawing
$ErrorActionPreference = "Stop"

$dst = Join-Path $PSScriptRoot "HexWars.NetServer\wwwroot"
if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Force -Path $dst | Out-Null }

$bg     = [System.Drawing.Color]::FromArgb(255, 0x0A, 0x0E, 0x1C)
$accent = [System.Drawing.Color]::FromArgb(255, 0x45, 0xAE, 0xFF)
$cta    = [System.Drawing.Color]::FromArgb(255, 0x33, 0x84, 0x5C)
$dim    = [System.Drawing.Color]::FromArgb(255, 0x9A, 0xA3, 0xB8)

function Get-HexPoints([double]$cx, [double]$cy, [double]$r, [double]$rotationDeg = 0) {
    $pts = New-Object System.Collections.Generic.List[System.Drawing.PointF]
    for ($i = 0; $i -lt 6; $i++) {
        $angle = [Math]::PI / 180.0 * (60 * $i + $rotationDeg - 90)
        $x = $cx + $r * [Math]::Cos($angle)
        $y = $cy + $r * [Math]::Sin($angle)
        $pts.Add((New-Object System.Drawing.PointF([float]$x, [float]$y)))
    }
    return $pts.ToArray()
}

function New-Icon([int]$size, [string]$path) {
    $bmp = New-Object System.Drawing.Bitmap($size, $size)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.Clear($bg)

    $cx = $size / 2.0; $cy = $size / 2.0
    $outerR = $size * 0.42
    $innerR = $size * 0.24

    $outerPen = New-Object System.Drawing.Pen($accent, [Math]::Max(2, $size * 0.035))
    $g.DrawPolygon($outerPen, (Get-HexPoints $cx $cy $outerR 0))

    $innerBrush = New-Object System.Drawing.SolidBrush($cta)
    $g.FillPolygon($innerBrush, (Get-HexPoints $cx $cy $innerR 0))

    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Png)
    $g.Dispose(); $bmp.Dispose()
}

New-Icon 192 (Join-Path $dst "icon-192.png")
New-Icon 512 (Join-Path $dst "icon-512.png")
New-Icon 64  (Join-Path $dst "favicon.png")

# preview.png — 1200x630 OpenGraph/Twitter card: wordmark + tagline over a scattered hex-tile motif
$pw = 1200; $ph = 630
$bmp = New-Object System.Drawing.Bitmap($pw, $ph)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear($bg)

$motifPen1 = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(40, $accent.R, $accent.G, $accent.B), 2)
$motifPen2 = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(30, $cta.R, $cta.G, $cta.B), 2)
$hr = 46.0
$stepX = $hr * 1.7; $stepY = $hr * 1.5
for ($row = -1; $row -lt ($ph / $stepY) + 2; $row++) {
    for ($col = -1; $col -lt ($pw / $stepX) + 2; $col++) {
        $x = $col * $stepX + (($row % 2) * $stepX * 0.5)
        $y = $row * $stepY
        $pen = if ((($row + $col) % 2) -eq 0) { $motifPen1 } else { $motifPen2 }
        $g.DrawPolygon($pen, (Get-HexPoints $x $y $hr 0))
    }
}

$titleFont = New-Object System.Drawing.Font("Arial", 92, [System.Drawing.FontStyle]::Bold)
$tagFont   = New-Object System.Drawing.Font("Arial", 30, [System.Drawing.FontStyle]::Regular)
$titleBrush = New-Object System.Drawing.SolidBrush($accent)
$tagBrush   = New-Object System.Drawing.SolidBrush($dim)

$g.DrawString("HEXWARS", $titleFont, $titleBrush, 90, 220)
$g.DrawString("hex-grid tactics - design an army, take the field", $tagFont, $tagBrush, 92, 340)

$bmp.Save((Join-Path $dst "preview.png"), [System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()

# manifest.json — "Add to Home Screen" / standalone PWA
$manifest = @'
{
  "name": "HexWars",
  "short_name": "HexWars",
  "display": "standalone",
  "start_url": "/",
  "background_color": "#0A0E1C",
  "theme_color": "#0A0E1C",
  "icons": [
    { "src": "/icon-192.png", "sizes": "192x192", "type": "image/png" },
    { "src": "/icon-512.png", "sizes": "512x512", "type": "image/png" }
  ]
}
'@
Set-Content (Join-Path $dst "manifest.json") -Value $manifest -Encoding utf8 -NoNewline

Write-Host "Generated manifest.json, icon-192.png, icon-512.png, favicon.png, preview.png under $dst"
Get-ChildItem $dst -Filter "*.png" | ForEach-Object { Write-Host "  $($_.Name): $($_.Length) bytes" }
```

- [ ] **Step 3: Run it**

```
powershell -ExecutionPolicy Bypass -File engine/generate-share-assets.ps1
```

Expected: `Generated manifest.json, icon-192.png, icon-512.png, favicon.png, preview.png under
...\wwwroot`, followed by four non-trivial byte counts (icon-192 ≈3 KB, icon-512 ≈9 KB, favicon ≈1 KB,
preview ≈80 KB — matches what was observed running this exact script during drafting). Visually spot-check
`preview.png` and `icon-512.png` (Read them as images) — hex badge icon, dark-palette OG card with the
HEXWARS wordmark and tagline over a faint hex-tile grid.

- [ ] **Step 4: `stage-webgl-deploy.ps1` — inject share-card tags after the cache-bust rewrite**

Replace the tail of the script (from the cache-bust comment to the end):

```powershell
# Cache-bust the payload URLs: builds ship under the SAME filenames, and Unity's IndexedDB cache
# keys by URL — after a redeploy a browser can pair an old cached .data with the new .wasm and die
# at boot ("RuntimeError: memory access out of bounds" in callMain). A per-deploy ?v= makes every
# build's URLs unique so old and new can never mix.
$v = (Get-FileHash (Join-Path $dst "Build\WebGL.data.unityweb") -Algorithm SHA256).Hash.Substring(0, 8).ToLower()
$idx = Join-Path $dst "index.html"
(Get-Content $idx -Raw) -replace '(Build/WebGL\.(?:data\.unityweb|framework\.js\.unityweb|wasm\.unityweb|loader\.js))', ('$1?v=' + $v) |
    Set-Content $idx -Encoding utf8 -NoNewline

Write-Host "Staged $src -> $dst with cache-bust v=$v  (now: git add/commit, then push from WSL)"
```

with:

```powershell
# Cache-bust the payload URLs: builds ship under the SAME filenames, and Unity's IndexedDB cache
# keys by URL — after a redeploy a browser can pair an old cached .data with the new .wasm and die
# at boot ("RuntimeError: memory access out of bounds" in callMain). A per-deploy ?v= makes every
# build's URLs unique so old and new can never mix.
$v = (Get-FileHash (Join-Path $dst "Build\WebGL.data.unityweb") -Algorithm SHA256).Hash.Substring(0, 8).ToLower()
$idx = Join-Path $dst "index.html"
$html = (Get-Content $idx -Raw) -replace '(Build/WebGL\.(?:data\.unityweb|framework\.js\.unityweb|wasm\.unityweb|loader\.js))', ('$1?v=' + $v)

# Share-card + PWA tags: the Unity template owns index.html, so staging is the one place post-build
# HTML edits happen — this extends the same rewrite step above rather than adding a second pass.
# Statics (manifest.json, icon-*.png, favicon.png, preview.png) live under wwwroot/ directly, generated
# by generate-share-assets.ps1; this script never deletes wwwroot root-level files (only wwwroot/Build/
# is removed+recreated, above), so they survive every redeploy without being re-copied here.
# Guarded on og:title so re-running this against an already-injected file — shouldn't happen, since
# index.html is always a fresh copy from $src above, but is cheap insurance — never double-inserts.
if ($html -notmatch 'og:title') {
    $headInject = @"
    <meta property="og:title" content="HexWars — hex-grid tactics" />
    <meta property="og:description" content="Design your army from raw points. Outbuild, outthink, dominate." />
    <meta property="og:image" content="https://hwbootstrap.onrender.com/preview.png" />
    <meta property="og:type" content="website" />
    <meta name="twitter:card" content="summary_large_image" />
    <meta name="twitter:title" content="HexWars — hex-grid tactics" />
    <meta name="twitter:description" content="Design your army from raw points. Outbuild, outthink, dominate." />
    <meta name="twitter:image" content="https://hwbootstrap.onrender.com/preview.png" />
    <link rel="manifest" href="/manifest.json" />
    <meta name="theme-color" content="#0A0E1C" />
    <link rel="apple-touch-icon" href="/icon-192.png" />
    <link rel="icon" type="image/png" href="/favicon.png" />
</head>
"@
    $html = $html -replace '</head>', $headInject
}

Set-Content $idx -Value $html -Encoding utf8 -NoNewline

Write-Host "Staged $src -> $dst with cache-bust v=$v  (now: git add/commit, then push from WSL)"
```

- [ ] **Step 5: Verify by actually staging**

This requires a real WebGL build to exist at `Build/WebGL` (`HexWars > Build WebGL` in the editor menu, per
the script's own error message, if one isn't already present from a prior task's deploy). Run:

```
powershell -File engine/stage-webgl-deploy.ps1
```

Then grep the staged file:

```
Select-String -Path engine/HexWars.NetServer/wwwroot/index.html -Pattern "og:title","og:image","twitter:card","rel=.manifest.","theme-color","apple-touch-icon","rel=.icon."
```

Expected: one match per pattern, `og:image` containing the literal `https://hwbootstrap.onrender.com/preview.png`.
Confirm the statics are still present and unchanged: `Get-ChildItem engine/HexWars.NetServer/wwwroot/*.png,
engine/HexWars.NetServer/wwwroot/manifest.json` lists all five files.

- [ ] **Step 6: Commit**

```bash
git add engine/generate-share-assets.ps1 engine/stage-webgl-deploy.ps1 engine/HexWars.NetServer/wwwroot/manifest.json engine/HexWars.NetServer/wwwroot/icon-192.png engine/HexWars.NetServer/wwwroot/icon-512.png engine/HexWars.NetServer/wwwroot/favicon.png engine/HexWars.NetServer/wwwroot/preview.png
git commit -m "feat(share): procedural PWA icons + OG preview card, manifest.json; staging injects share-card/PWA tags into index.html after its cache-bust rewrite"
```

---

### Task 10: Barracks names + delete

**Files:**
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs`

**Interfaces:**
- Consumes: `PlayerState.Barracks : IReadOnlyList<UnitTemplate>` (landed), `DeleteTemplate(PlayerId, int)`
  (landed).
- Produces: no new public surface — `IsDeploying`/`ReadOnly` keep their exact shape (Task 12 reads
  `IsDeploying`).

- [ ] **Step 1: Rows show the template's name + a delete button**

Read the current file (quoted in full above during drafting). Replace `Rebuild()`:

```csharp
        void Rebuild()
        {
            foreach (var r in _rows) Destroy(r.gameObject);
            _rows.Clear();
            if (_game == null) return;

            // hidden during the title demo and the connecting window (no state yet)
            if (_game.DemoMode || _game.State == null)
            {
                if (_canvasGo != null) _canvasGo.SetActive(false);
                return;
            }
            if (_canvasGo != null && !_canvasGo.activeSelf) _canvasGo.SetActive(true);

            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            if (_deployIndex >= p.Barracks.Count) _deployIndex = -1;

            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var stats = p.Barracks[i];
                int cost = Economy.DeployCost(stats, s.Config);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = UiKit.Button(_list, $"{Roles.Dominant(stats)}   deploy {cost}", 0f, -(4f + i * 34f), 214f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary);
                UiKit.SetToggled(row, selected);
                _rows.Add(row);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");
        }
```

with:

```csharp
        void Rebuild()
        {
            foreach (var r in _rows) Destroy(r.gameObject);
            _rows.Clear();
            if (_game == null) return;

            // hidden during the title demo and the connecting window (no state yet)
            if (_game.DemoMode || _game.State == null)
            {
                if (_canvasGo != null) _canvasGo.SetActive(false);
                return;
            }
            if (_canvasGo != null && !_canvasGo.activeSelf) _canvasGo.SetActive(true);

            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            if (_deployIndex >= p.Barracks.Count) _deployIndex = -1;

            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var template = p.Barracks[i];
                int cost = Economy.DeployCost(template.Stats, s.Config);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = UiKit.Button(_list, $"{template.Name}   deploy {cost}", -20f, -(4f + i * 34f), 170f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary);
                UiKit.SetToggled(row, selected);
                _rows.Add(row);

                var del = UiKit.Button(_list, "✕", 100f, -(4f + i * 34f), 32f, 30f,
                                       () => DeleteAt(idx), UiKit.ButtonStyle.Danger, 14);
                del.interactable = !ReadOnly;
                _rows.Add(del);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");
        }

        /// <summary>Delete a barracks template — free, no turn cost (DeleteTemplate is administrative,
        /// not a game move; it's not in LegalMoves). Bookkeeping mirrors spec §5: deleting the selected
        /// row clears deploy mode; deleting a row before the selected one shifts the selected index down
        /// so it still points at the same template after the barracks list re-indexes.</summary>
        void DeleteAt(int index)
        {
            if (ReadOnly || _game == null || _game.State == null) return;
            var seat = _game.State.ActivePlayer;
            if (!_game.TryApply(new DeleteTemplate(seat, index))) return;
            if (_deployIndex == index) _deployIndex = -1;
            else if (_deployIndex > index) _deployIndex--;
        }
```

(`_rows` already holds every row `Button` this panel builds and is fully cleared at the top of `Rebuild()`
— adding the delete buttons to the same list needs no new field.)

- [ ] **Step 2: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 3: Verify in play mode**

Enter Play Mode (hotseat boot — starter templates are seeded by `GameFactory`/`NewGame`'s player construction
per the landed engine surface). Each numbered item is its own `execute_script` call. `DeleteAt`/`Select` are
private, so every step below drives them the same way a real player does — through the actual row/✕
`Button.onClick` — rather than calling `TryApply(new DeleteTemplate(...))` directly (which would exercise
the engine command but skip `BarracksPanel`'s own `_deployIndex` bookkeeping entirely, the thing this task
actually needs proven). Rows and their ✕ buttons are built in matched pairs under
`BarracksCanvas/BarracksPanel/List`, so for template index `i` the row button is child `i*2` and its ✕ is
child `i*2+1`:

```csharp
var list = GameObject.Find("BarracksCanvas").transform.Find("BarracksPanel").Find("List");
```

1. Assert `FindAnyObjectByType<GameBootstrap>().State.Player(HexWars.Engine.PlayerId.Player0).Barracks.Count == 5`
   and the five `.Name` values are `Brute, Striker, Sniper, Artillery, Scout` in order (indices 0-4, per spec §5).
2. Screenshot `BarracksCanvas` — five named rows, each with a ✕.
3. `list.GetChild(2 * 2 + 1).GetComponent<UnityEngine.UI.Button>().onClick.Invoke();` (clicks template
   index 2's ✕ — Sniper). Assert `Barracks.Count == 4` and the remaining names are
   `Brute, Striker, Artillery, Scout` (Sniper gone, Artillery/Scout shifted down to indices 2-3).
   Screenshot the panel again — four rows, re-fetch `list` first since `Rebuild()` destroyed/rebuilt it.
4. `list.GetChild(0).GetComponent<UnityEngine.UI.Button>().onClick.Invoke();` (selects template 0 = Brute).
   Assert `FindAnyObjectByType<BarracksPanel>().IsDeploying == true`.
5. `list.GetChild(1).GetComponent<UnityEngine.UI.Button>().onClick.Invoke();` (clicks THAT SAME row's ✕ —
   deletes the selected template). Assert `IsDeploying == false` (deploy mode cleared) and `Barracks.Count == 3`
   with names `Striker, Artillery, Scout`.
6. Re-fetch `list`, click template index 0's row (`list.GetChild(0)`, now Striker) to select it, then apply
   a legal deploy: `game.TryApply(new HexWars.Engine.DeployUnit(HexWars.Engine.PlayerId.Player0, 0, <a
   deployment-zone cell from game.State.Board.DeploymentZone(PlayerId.Player0)>));` and confirm it returns
   `true` — proves indices stayed correct after the earlier shifts.
7. Exit Play Mode.

- [ ] **Step 4: Commit**

```bash
git add Assets/HexWars/Presentation/BarracksPanel.cs
git commit -m "feat(barracks): rows show the template's name (was the dominant-role string) with a per-row delete button; _deployIndex bookkeeping corrected on shifts"
```

---

### Task 11: Stat descriptions + designer name

**Files:**
- Create: `Assets/HexWars/Presentation/StatInfo.cs`
- Create: `Assets/HexWars/Presentation/TipBubble.cs`
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs`
- Modify: `Assets/HexWars/Presentation/UnitTooltip.cs`
- Modify: `Assets/HexWars/Presentation/GameRules.cs`

**Interfaces:**
- Consumes: `CreateUnit(PlayerId, UnitStats, string)` (landed), `UiKit.Panel`/`Button`/`Label`/`PromptText`.
- Produces (Task 12 consumes both):
  - `public static readonly (string Key, string Caption, string Full)[] StatInfo.All` — 9 entries, in the
    same order as `DesignPanel`'s existing `Names` array.
  - `public static void TipBubble.Show(string text, Vector2 screenPos, string cta = null, System.Action onCta = null)`
  - `public static void TipBubble.Dismiss()`

- [ ] **Step 1: `StatInfo` — the nine descriptions, verbatim**

Create `Assets/HexWars/Presentation/StatInfo.cs`:

```csharp
namespace HexWars.Presentation
{
    /// <summary>
    /// The nine stat descriptions (design spec §6) — each one mechanic + judgment. Always available
    /// regardless of the Tips toggle (spec: "Reference, not hand-holding"). <c>Caption</c> is a distilled
    /// one-liner used for the Designer's always-visible row captions when Tips is on (Task 12);
    /// <c>Full</c> is the spec's verbatim copy shown in the tap-to-see bubble (this task, via
    /// <see cref="TipBubble"/>). Order matches <see cref="DesignPanel"/>'s stat row order exactly.
    /// </summary>
    public static class StatInfo
    {
        public static readonly (string Key, string Caption, string Full)[] All =
        {
            ("Health",
             "Absorbs damage before dying — buy it to hold ground.",
             "How much damage it absorbs before dying. Buy it for units that must hold ground under fire; a 2-health unit dies to one mistake."),

            ("Damage",
             "Kill speed — enough Damage beats Defense stacking.",
             "Subtracted by the target's Defense; a landed hit always deals at least 1. This is kill speed — enough Damage makes Defense stacking pointless."),

            ("Defense",
             "Cuts incoming damage — strong vs swarms, weak vs one gun.",
             "Subtracted from every hit you take. Against a swarm of weak attackers it multiplies your effective health; against one big gun it's nearly worthless. Read the enemy's army first."),

            ("Movement",
             "Horizontal steps per turn — reach, escape, tempo.",
             "Horizontal steps per turn. Reach, escape, and tempo. Zero is a choice: an emplacement that never moves — position it like it matters, because it will never matter again."),

            ("Vertical Move",
             "Levels climbed per turn; down/level moves are free.",
             "How many levels it can climb per turn (descending and level moves are free). High ground adds damage and reach, so climbers take the positions that win fights."),

            ("Range",
             "How far it shoots; 0 means melee only.",
             "How far it shoots (0 = melee only). Outranging the enemy's answer is free damage; high ground extends it further."),

            ("Range Arc",
             "Levels it can fire upward; can lob over terrain.",
             "How many levels up it can fire — and anything above 0 can lob over blocking terrain (indirect fire). Your army still needs eyes on the target: batteries want spotters."),

            ("Vision",
             "How far it sees; you can only shoot what's seen.",
             "How far it sees. Sight is shared by your whole army, and you can only shoot what somebody sees. Under fog, information is the game — a cheap pair of eyes makes every gun longer."),

            ("Vision Arc",
             "How many levels up it can see.",
             "How many levels up it can see. Cliffs hide things; someone has to look over the edge."),
        };
    }
}
```

- [ ] **Step 2: `TipBubble` — the shared dismissible bubble**

Create `Assets/HexWars/Presentation/TipBubble.cs`:

```csharp
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// One shared dismissible info bubble. Used directly by stat-name tap targets (this task — always
    /// available, no CTA) and by <c>TipsService</c> for the Tips coaching layer (Task 12 — optionally with
    /// a CTA button, e.g. "Design your answer" opening the Designer). Only one bubble exists at a time —
    /// a new <see cref="Show"/> replaces whatever's already up. Tapping ANYWHERE (the bubble itself, or
    /// off it) dismisses it, per spec §6 ("tap anywhere or act → gone"); the CTA button, when present,
    /// sits on top and captures its own click first (dismiss + fire the action), never both.
    /// </summary>
    public static class TipBubble
    {
        public const string RootName = "TipBubbleCanvas";

        /// <summary><paramref name="screenPos"/> is a SCREEN-space position (physical/logical pixels,
        /// origin bottom-left) — the one coordinate space every caller can reach regardless of source:
        /// a UI element via <c>RectTransformUtility.WorldToScreenPoint(null, rt.TransformPoint(local))</c>
        /// (camera null — every UiKit canvas is ScreenSpaceOverlay), or a 3D scene object via
        /// <c>Camera.main.WorldToScreenPoint(worldPos)</c> (Task 12's unit-selection trigger uses this).
        /// Converted to this bubble's own canvas-local space via
        /// <c>RectTransformUtility.ScreenPointToLocalPointInRectangle</c> (camera null, same reason).
        /// Pass <c>new Vector2(Screen.width / 2f, Screen.height / 2f)</c> for a screen-centered bubble
        /// (Task 12's non-anchored triggers use this).</summary>
        public static void Show(string text, Vector2 screenPos, string cta = null, System.Action onCta = null)
        {
            Dismiss();

            var root = UiKit.Canvas(RootName, UiKit.OrderTooltip + 1, null);
            var canvasRt = root.GetComponent<RectTransform>();
            RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRt, screenPos, null, out Vector2 localAnchor);

            // full-screen invisible catcher BEHIND the card — taps anywhere off the card dismiss too
            var backdrop = new GameObject("Backdrop");
            backdrop.transform.SetParent(root.transform, false);
            var bdImg = backdrop.AddComponent<Image>();
            bdImg.color = new Color(0f, 0f, 0f, 0f);
            UiKit.Stretch(backdrop.GetComponent<RectTransform>());
            backdrop.AddComponent<Button>().onClick.AddListener(Dismiss);

            float w = 360f;
            float h = cta != null ? 170f : 120f;
            var card = UiKit.Panel(root.transform, "Bubble", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(w, h);
            crt.anchoredPosition = ClampToCanvas(localAnchor, canvasRt, w, h);
            card.AddComponent<Button>().onClick.AddListener(Dismiss); // tap the card's own body → dismiss

            var label = UiKit.Label(card.transform, text, 0f, -16f, w - 32f, h - (cta != null ? 60f : 32f),
                                    UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextMain);
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Overflow;

            if (cta != null)
                UiKit.Button(card.transform, cta, 0f, -(h - 46f), w - 60f, 38f,
                            () => { Dismiss(); onCta?.Invoke(); }, UiKit.ButtonStyle.Cta);
        }

        public static void Dismiss()
        {
            var old = GameObject.Find(RootName);
            if (old != null) Object.Destroy(old);
        }

        /// <summary>Keeps the bubble fully on-canvas even when anchored near a screen edge (a stat label
        /// near the panel edge, or an anchor-less call that defaults to screen center).</summary>
        static Vector2 ClampToCanvas(Vector2 anchorPos, RectTransform canvasRt, float w, float h)
        {
            float hw = canvasRt.rect.width * 0.5f, hh = canvasRt.rect.height * 0.5f;
            float x = Mathf.Clamp(anchorPos.x, -hw + w * 0.5f + 12f, hw - w * 0.5f - 12f);
            float y = Mathf.Clamp(anchorPos.y, -hh + h * 0.5f + 12f, hh - h * 0.5f - 12f);
            return new Vector2(x, y);
        }
    }
}
```

- [ ] **Step 3: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 4: `DesignPanel` — tappable stat labels + Name field**

Read the current file (quoted in full above during drafting). Replace the stat-row loop in `Build()`:

```csharp
            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowH);
                UiKit.Label(panel, Names[i], -63f, y, 120f, rowH, 15, TextAnchor.MiddleLeft);
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                int idx = i;
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
            }
            _valueLabels[0].text = "1";

            float sy = -(40f + 9 * rowH + 6f);
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
        }
```

with:

```csharp
            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowH);
                int idx = i;
                // the label itself is the tap target — a stat name button with a text-only look, opening
                // the verbatim description (spec §6: "always available, Tips or no Tips")
                var nameBtn = UiKit.Button(panel, Names[i], -63f, y, 120f, rowH, () =>
                {
                    Vector3 world = panel.TransformPoint(new Vector3(-63f, y, 0f));
                    Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(null, world); // camera
                                                                    // null — this canvas is ScreenSpaceOverlay
                    TipBubble.Show(StatInfo.All[idx].Full, screenPos);
                }, UiKit.ButtonStyle.Secondary, 15);
                nameBtn.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
            }
            for (int i = 0; i < 9; i++) _valueLabels[i].text = _stats[i].ToString(); // sync display to
                                                                                      // current _stats — matters
                                                                                      // once Task 12's Tips-toggle
                                                                                      // rebuild can re-run this Build()
                                                                                      // after the player has already
                                                                                      // spent points (was a bare
                                                                                      // "_valueLabels[0].text = "1";")

            float nameY = -(40f + 9 * rowH + 6f);
            UiKit.Label(panel, "Name", -63f, nameY, 60f, rowH, 15, TextAnchor.MiddleLeft);
            _nameBox = UiKit.Button(panel, PlaceholderText(), 23f, nameY, w - 110f, rowH, OnTapName,
                                    UiKit.ButtonStyle.Secondary, 14).GetComponentInChildren<Text>();
            ApplyNameDisplay(); // sets the grey placeholder color (UiKit.Button's own label defaults to white)

            float sy = nameY - rowH - 6f;
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
        }
```

Add fields and the name-entry plumbing. Replace the field block:

```csharp
        GameBootstrap _game;
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
```

with:

```csharp
        static readonly string[] NamePlaceholders = { "Doom Turtle", "Longshot", "Pathfinder" };

        GameBootstrap _game;
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
        Text _nameBox;
        string _name = "";
        int _placeholderIdx;
```

Add the name-entry methods and `PlaceholderText` after `Adjust`:

```csharp
        void Adjust(int i, int delta)
        {
            _stats[i] = Mathf.Max(i == 0 ? 1 : 0, _stats[i] + delta);
            _valueLabels[i].text = _stats[i].ToString();
            RefreshSummary();
        }

        string PlaceholderText() => NamePlaceholders[_placeholderIdx];

        /// <summary>Browser-prompt text entry (the established mobile pattern, same as join-by-code).
        /// Empty stays legal — CreateUnit/UnitTemplate.Sanitize defaults an empty name to the dominant
        /// role at the engine boundary.</summary>
        void OnTapName()
        {
            string typed = UiKit.PromptText("Name your unit", _name);
            if (typed == null) return; // cancelled, or no browser prompt available (editor) — leave as-is
            _name = typed.Trim();
            if (_name.Length == 0) RotatePlaceholder();
            ApplyNameDisplay();
        }

        /// <summary>Advances to the next example (Task 12's Tips-toggle rebuild also calls
        /// <see cref="ApplyNameDisplay"/> to resync the box's text/color after a rebuild, but must NOT
        /// rotate the placeholder just because the panel redrew — only an actual "went back to empty"
        /// user action should pick a new example, so the two are kept separate.</summary>
        void RotatePlaceholder() => _placeholderIdx = (_placeholderIdx + 1) % NamePlaceholders.Length;

        /// <summary>Pure display sync — safe to call any time the name box exists and needs to reflect
        /// current state (after typing, after Create, after a rebuild).</summary>
        void ApplyNameDisplay()
        {
            if (_name.Length > 0) { _nameBox.text = _name; _nameBox.color = UiKit.TextMain; }
            else { _nameBox.text = PlaceholderText(); _nameBox.color = UiKit.TextFaint; } // grey, per spec
        }
```

- [ ] **Step 5: `DesignPanel.OnCreate` — carries the name**

Replace:

```csharp
        void OnCreate()
        {
            if (_game == null || _game.State == null) return;
            _game.TryApply(new CreateUnit(_game.State.ActivePlayer, ToStats()));
        }
```

with:

```csharp
        void OnCreate()
        {
            if (_game == null || _game.State == null) return;
            if (_game.TryApply(new CreateUnit(_game.State.ActivePlayer, ToStats(), _name)))
            {
                _name = "";
                RotatePlaceholder(); // a fresh empty box next time shows a different example
                ApplyNameDisplay();
            }
        }
```

- [ ] **Step 6: `UnitTooltip` — header shows the deployed unit's name**

Replace the header line in `Format`:

```csharp
        static string Format(Unit u, GameState state)
        {
            var s = u.Stats;
            string role = Roles.Dominant(s).ToString();
            string owner = u.Owner == PlayerId.Player0 ? "Player 1" : "Player 2";
            string text =
                $"<b>{role}</b>  {s.PointCost} pts  ({owner})\n" +
```

with:

```csharp
        static string Format(Unit u, GameState state)
        {
            var s = u.Stats;
            string owner = u.Owner == PlayerId.Player0 ? "Player 1" : "Player 2";
            string text =
                $"<b>{u.DisplayName}</b>  {s.PointCost} pts  ({owner})\n" +
```

(`Roles.Dominant` is no longer referenced in this file — `role` was the only use; `Unit.DisplayName` per
the landed engine surface already falls back to the dominant-role string when a unit has no name, so a
starter-template deploy or an old-format spectated unit still reads correctly.)

Stat lines tappable in the tooltip is explicitly a stretch goal, not required: the docked layout is a
single multi-line `Text` block (see `Format`), not one `Text`/button per stat line — adding a per-line tap
target would mean rebuilding the tooltip as N separate `Text`+`Button` rows instead of one formatted
string, which changes its layout model non-trivially (and Task 7's caching in this same file assumes a
single string). **Flagging per instructions rather than forcing it**: the Designer (Step 4 above) is the
mandatory surface per spec §6 ("Tapping/hovering a stat name in the Designer OR the unit tooltip" — the
Designer alone satisfies "always available"); tooltip tap-targets are left for a follow-up if wanted.

- [ ] **Step 7: `GameRules` — UNITS section rewritten + new DESIGN YOUR OWN section**

Replace the `UNITS` block inside the `RulesText` constant:

```
UNITS
Each unit has: Health, Damage, Defense, Move, Vertical (climb), Range, and Vision.
  Brute    — tough melee (high HP/defense, range 1).
  Striker  — fast glass cannon (high damage, low HP).
  Sniper   — fragile but long range.
Hover (or touch) a unit in-game to see its full stats.
```

with:

```
UNITS
Brute, Striker, Sniper, Artillery, Scout — these are just ideas that come pre-loaded in your barracks
(delete them if you like). Any allocation you can imagine is a unit. Name it what it is.
Each unit has: Health, Damage, Defense, Move, Vertical (climb), Range, Range Arc, Vision, and Vision Arc.
Tap any stat name in the Designer, or hover/touch a unit in-game, to see what it does and why you'd buy it.

DESIGN YOUR OWN
Everything is points. See what your opponent built; build the answer.
  Create — open the Designer, put points into whichever stats fit the plan, name it, and it lands in
    your barracks as a reusable template.
  Deploy — pick a template in the barracks and place a paid clone of it — the template itself is never
    consumed, so deploy it again next turn or next game.
  Adapt — a kill earns bounty points. Spend them on the answer to what you just saw, not a repeat of
    what you already have.
```

(the file's other sections — GOAL, MODES, MOVING, COMBAT, TERRITORY & ECONOMY, FOG OF WAR, PACE, CONTROLS —
are unchanged; only the `UNITS` block is replaced and a new `DESIGN YOUR OWN` block is inserted
immediately after it, before `MOVING`.)

- [ ] **Step 8: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 9: Verify in play mode**

Enter Play Mode. Each numbered item is its own `execute_script` call:
1. Screenshot `DesignCanvas` — nine stat rows with name-labels rendered as buttons, a Name row beneath them
   showing a grey placeholder (one of the three rotating examples), then Create.
2. Click the "Health" label button (`FindAnyObjectByType<DesignPanel>()`'s row — locate via
   `GetComponentsInChildren<Button>()` and match by its child `Text.text == "Health"`). Screenshot —
   `TipBubbleCanvas` shows the verbatim Health description from `StatInfo.All[0].Full`.
3. Tap anywhere off the bubble (e.g. click the backdrop) — assert `GameObject.Find("TipBubbleCanvas") == null`.
4. Editor fallback check: click the Name row — `UiKit.PromptText` returns `null` outside WebGL, so assert
   the name box text is UNCHANGED (still showing a placeholder) and `_name` stays empty — the default-name
   fallback the spec calls for.
5. Click "Create (to Barracks)" with the name still empty; assert the deployed template's `.Name` equals
   its dominant role (engine-side `UnitTemplate.Sanitize` default) — confirms empty stays legal end to end.
6. Open the Rules card (`GameRules.Show(...)`) and screenshot — confirm the UNITS section reads the new
   copy and DESIGN YOUR OWN appears.
7. Exit Play Mode.

- [ ] **Step 10: Commit**

```bash
git add Assets/HexWars/Presentation/StatInfo.cs Assets/HexWars/Presentation/TipBubble.cs Assets/HexWars/Presentation/DesignPanel.cs Assets/HexWars/Presentation/UnitTooltip.cs Assets/HexWars/Presentation/GameRules.cs
git commit -m "feat(designer): stat-name tap targets show the spec's verbatim description (TipBubble), Name field wired into CreateUnit, tooltip header shows Unit.DisplayName; rules UNITS section reframes roles as examples + DESIGN YOUR OWN"
```

---

### Task 12: TipsService + toggle

**Files:**
- Create: `Assets/HexWars/Presentation/TipsService.cs`
- Modify: `Assets/HexWars/Presentation/TitleScreen.cs` (toggle button)
- Modify: `Assets/HexWars/Presentation/HelpOverlay.cs` (toggle button beside "?")
- Modify: `Assets/HexWars/Presentation/UnitInputController.cs` (first-selection trigger)
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs` (inline captions + `Highlight()` + rebuild-on-toggle)
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs` (affordable-deploy trigger)
- Modify: `Assets/HexWars/Presentation/GameHud.cs` (out-of-actions trigger, game-over-vs-AI trigger)
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (registry reset on new game, first-bounty detection)

**Interfaces:**
- Consumes: `TipBubble.Show(string, Vector2, string, Action)` (Task 11), `StatInfo.All` (Task 11),
  `AiOpponent` (existing — vs-AI detection, same pattern already used by `GameBootstrap.LocalHumanActs`/
  `WaitingHumanSeat`).
- Produces:
  - `public static bool TipsService.Enabled { get; set; }` — persisted, default on for a first-ever visit.
  - `public static void TipsService.NewGame()` — clears the once-per-game registry.
  - `public static void TipsService.Show(string id, string text, Vector2? screenPos = null, string cta = null, System.Action onCta = null)`.
  - `public static Button TipsService.BuildToggle(Transform parent, float x, float y)` — the reusable
    "Tips: On/Off" control (title screen + HelpOverlay both build one).
  - `public void DesignPanel.Highlight()` — the first-bounty CTA's target (the panel has no separate
    open/closed state to "open" — see the CTA's own note below for why this is the honest implementation).
  - **Forward note for Task 13**: the game-over-vs-AI trigger below is informational only in THIS task (no
    CTA — `GameBootstrap.Rematch()` doesn't exist yet, Task 13 adds it). Task 13's own `GameHud.cs` edit
    upgrades this exact call site to add the CTA once the Rematch button exists; Task 13's diff must quote
    the code this task leaves behind as its "old" text.

- [ ] **Step 1: `TipsService`**

Create `Assets/HexWars/Presentation/TipsService.cs`:

```csharp
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The Tips coaching layer's one owner (spec §6): a persisted opt-out flag, and a once-per-game
    /// trigger registry so each moment-suggestion fires at most once. Renders through <see cref="TipBubble"/>
    /// (one bubble at a time, tap-anywhere-dismiss — that discipline lives in TipBubble itself). Entirely
    /// inert when <see cref="Enabled"/> is false: callers never need to check it themselves.
    /// </summary>
    public static class TipsService
    {
        const string PrefKey = "HexWars.Tips";
        static bool? _enabled;
        static readonly HashSet<string> _firedThisGame = new HashSet<string>();

        /// <summary>Defaults ON for a first-ever visit (no key written yet); persists after that.</summary>
        public static bool Enabled
        {
            get
            {
                if (_enabled == null) _enabled = PlayerPrefs.GetInt(PrefKey, 1) != 0;
                return _enabled.Value;
            }
            set
            {
                _enabled = value;
                PlayerPrefs.SetInt(PrefKey, value ? 1 : 0);
                PlayerPrefs.Save();
                if (!value) TipBubble.Dismiss(); // switching off must not leave a bubble hanging (spec §7:
                                                  // "off forever once off" — includes whatever's on screen right now
            }
        }

        /// <summary>Clear the once-per-game registry. Called by GameBootstrap on every real new-game
        /// entry point (NOT on a Task 6 reconnect's START re-deal — that's the same game continuing).</summary>
        public static void NewGame() => _firedThisGame.Clear();

        /// <summary>Show a tip at most once per game per <paramref name="id"/>, only while Tips is on.
        /// A no-op otherwise (off, or already fired this game) — callers never branch on Enabled.</summary>
        public static void Show(string id, string text, Vector2? screenPos = null, string cta = null, System.Action onCta = null)
        {
            if (!Enabled) return;
            if (!_firedThisGame.Add(id)) return;
            var pos = screenPos ?? new Vector2(Screen.width / 2f, Screen.height / 2f);
            TipBubble.Show(text, pos, cta, onCta);
        }

        /// <summary>Small reusable "Tips: On/Off" control — the title screen (bottom corner) and the
        /// in-game "?" (HelpOverlay) each place one of these. One tap flips <see cref="Enabled"/> and
        /// relabels itself; callers position the returned Button however fits their screen.</summary>
        public static Button BuildToggle(Transform parent, float x, float y)
        {
            Button btn = null;
            btn = UiKit.Button(parent, "Tips: " + (Enabled ? "On" : "Off"), x, y, 110f, 34f, () =>
            {
                Enabled = !Enabled;
                btn.GetComponentInChildren<Text>().text = "Tips: " + (Enabled ? "On" : "Off");
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeCaption);
            return btn;
        }
    }
}
```

- [ ] **Step 2: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 3: Toggle UI — title screen + HelpOverlay**

In `TitleScreen.Build()`, add the toggle to the bottom-left screen corner (not inside the centered menu
column — it's a persistent screen-corner control). Replace the tail of `Build()`:

```csharp
            UiKit.Label(col.transform, "v" + Application.version + "   ·   two players, two browsers — share a room code",
                        0f, y - 6f, 520f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);
        }
```

with:

```csharp
            UiKit.Label(col.transform, "v" + Application.version + "   ·   two players, two browsers — share a room code",
                        0f, y - 6f, 520f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var tipsBtn = TipsService.BuildToggle(_canvasGo.transform, 0f, 0f);
            var trt = tipsBtn.GetComponent<RectTransform>();
            trt.anchorMin = trt.anchorMax = new Vector2(0f, 0f);
            trt.pivot = new Vector2(0f, 0f);
            trt.anchoredPosition = new Vector2(12f, 12f);
        }
```

In `HelpOverlay.Start()`, add the toggle beside the "?" button. Replace the tail of `Start()`:

```csharp
            // UiKit.Button anchors top-centre (SetRect); re-anchor to top-right afterward so the "?"
            // lands in the same corner (-12, -12) it always has.
            var q = UiKit.Button(canvasGo.transform, "?", 0f, 0f, 54f, 54f,
                                 () => GameRules.Show(canvasGo.transform, _font, 950),
                                 UiKit.ButtonStyle.Secondary, 26);
            var qrt = q.GetComponent<RectTransform>();
            qrt.anchorMin = qrt.anchorMax = new Vector2(1f, 1f);
            qrt.pivot = new Vector2(1f, 1f);
            qrt.anchoredPosition = new Vector2(-12f, -12f);
        }
```

with:

```csharp
            // UiKit.Button anchors top-centre (SetRect); re-anchor to top-right afterward so the "?"
            // lands in the same corner (-12, -12) it always has.
            var q = UiKit.Button(canvasGo.transform, "?", 0f, 0f, 54f, 54f,
                                 () => GameRules.Show(canvasGo.transform, _font, 950),
                                 UiKit.ButtonStyle.Secondary, 26);
            var qrt = q.GetComponent<RectTransform>();
            qrt.anchorMin = qrt.anchorMax = new Vector2(1f, 1f);
            qrt.pivot = new Vector2(1f, 1f);
            qrt.anchoredPosition = new Vector2(-12f, -12f);

            var tipsBtn = TipsService.BuildToggle(canvasGo.transform, 0f, 0f);
            var trt = tipsBtn.GetComponent<RectTransform>();
            trt.anchorMin = trt.anchorMax = new Vector2(1f, 1f);
            trt.pivot = new Vector2(1f, 1f);
            trt.anchoredPosition = new Vector2(-12f - 54f - 8f, -12f); // left of the "?" (54 wide, 8px gap)
        }
```

(this toggle lives on `HelpCanvas`, so it's hidden/shown by the exact same `Update()` guard that already
hides the "?" during the demo and the connecting window — no separate visibility logic needed.)

- [ ] **Step 4: First-selection trigger**

In `UnitInputController`, replace `Select`:

```csharp
        void Select(UnitView unit)
        {
            _selected = unit;
            _selectedId = unit != null ? unit.Unit.Id : -1;
            UpdateMarker();
        }
```

with:

```csharp
        void Select(UnitView unit)
        {
            _selected = unit;
            _selectedId = unit != null ? unit.Unit.Id : -1;
            UpdateMarker();

            // gated on !DemoMode: the title-screen demo's units are hoverable/clickable (this class isn't
            // demo-aware), but a Tips bubble popping up over the muted showcase would break DemoMode's
            // whole point (suppressed gameplay UI) — see GameBootstrap.DemoMode's doc comment.
            if (unit != null && _game != null && !_game.DemoMode && Camera.main != null)
            {
                Vector2 screenPos = Camera.main.WorldToScreenPoint(unit.transform.position);
                TipsService.Show("first-select", "Green hexes = where it can go. Red = what it can hit.", screenPos);
            }
        }
```

- [ ] **Step 5: `DesignPanel` — inline captions when Tips is on, `Highlight()`, rebuild on toggle**

Read the current file as Task 8 and Task 11 left it (both already applied). Replace the sizing block Task 8
left in `Build()`:

```csharp
            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            const float rowH = 30f, top = 58f;
            float w = Mathf.Min(270f, availW - 16f);
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowH * 9 + 120f);
            prt.anchoredPosition = new Vector2(8f, -top);
```

with:

```csharp
            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            const float rowH = 30f, top = 58f;
            float w = Mathf.Min(270f, availW - 16f);
            bool tipsOn = TipsService.Enabled;         // an inline caption line under each stat row while on
            float rowSlot = tipsOn ? rowH + 15f : rowH;
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowSlot * 9 + 120f);
            prt.anchoredPosition = new Vector2(8f, -top);
```

Replace the stat-row loop Task 11 left:

```csharp
            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowH);
                int idx = i;
                // the label itself is the tap target — a stat name button with a text-only look, opening
                // the verbatim description (spec §6: "always available, Tips or no Tips")
                var nameBtn = UiKit.Button(panel, Names[i], -63f, y, 120f, rowH, () =>
                {
                    Vector3 world = panel.TransformPoint(new Vector3(-63f, y, 0f));
                    Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(null, world); // camera
                                                                    // null — this canvas is ScreenSpaceOverlay
                    TipBubble.Show(StatInfo.All[idx].Full, screenPos);
                }, UiKit.ButtonStyle.Secondary, 15);
                nameBtn.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
            }
            for (int i = 0; i < 9; i++) _valueLabels[i].text = _stats[i].ToString(); // sync display to
                                                                                      // current _stats — matters
                                                                                      // once Task 12's Tips-toggle
                                                                                      // rebuild can re-run this Build()
                                                                                      // after the player has already
                                                                                      // spent points (was a bare
                                                                                      // "_valueLabels[0].text = "1";")

            float nameY = -(40f + 9 * rowH + 6f);
            UiKit.Label(panel, "Name", -63f, nameY, 60f, rowH, 15, TextAnchor.MiddleLeft);
            _nameBox = UiKit.Button(panel, PlaceholderText(), 23f, nameY, w - 110f, rowH, OnTapName,
                                    UiKit.ButtonStyle.Secondary, 14).GetComponentInChildren<Text>();
            ApplyNameDisplay(); // sets the grey placeholder color (UiKit.Button's own label defaults to white)

            float sy = nameY - rowH - 6f;
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
        }
```

with:

```csharp
            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowSlot);
                int idx = i;
                // the label itself is the tap target — a stat name button with a text-only look, opening
                // the verbatim description (spec §6: "always available, Tips or no Tips")
                var nameBtn = UiKit.Button(panel, Names[i], -63f, y, 120f, rowH, () =>
                {
                    Vector3 world = panel.TransformPoint(new Vector3(-63f, y, 0f));
                    Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(null, world); // camera
                                                                    // null — this canvas is ScreenSpaceOverlay
                    TipBubble.Show(StatInfo.All[idx].Full, screenPos);
                }, UiKit.ButtonStyle.Secondary, 15);
                nameBtn.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
                if (tipsOn) // spec §6: designer-opened stat rows show their one-line captions inline while Tips is on
                    UiKit.Label(panel, StatInfo.All[i].Caption, -63f, y - rowH + 2f, w - 90f, 15f,
                               11, TextAnchor.UpperLeft, UiKit.TextFaint);
            }
            for (int i = 0; i < 9; i++) _valueLabels[i].text = _stats[i].ToString(); // sync display to
                                                                                      // current _stats — matters
                                                                                      // once the Tips-toggle rebuild
                                                                                      // below can re-run this Build()
                                                                                      // after the player has already
                                                                                      // spent points

            float nameY = -(40f + 9 * rowSlot + 6f);
            UiKit.Label(panel, "Name", -63f, nameY, 60f, rowH, 15, TextAnchor.MiddleLeft);
            _nameBox = UiKit.Button(panel, PlaceholderText(), 23f, nameY, w - 110f, rowH, OnTapName,
                                    UiKit.ButtonStyle.Secondary, 14).GetComponentInChildren<Text>();
            ApplyNameDisplay(); // sets the grey placeholder color (UiKit.Button's own label defaults to white)

            float sy = nameY - rowH - 6f;
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
        }

        /// <summary>Called by GameBootstrap's first-bounty Tips CTA ("Design your answer"). This panel
        /// has no separate open/closed state to "open" — it's already visible whenever a game is active
        /// — so this draws the eye instead: a brief bright pulse on the panel background. Honest
        /// implementation of "opens the Designer" given the panel's always-visible design, not a fake
        /// no-op click handler.</summary>
        public void Highlight()
        {
            if (_canvasGo == null || !_canvasGo.activeSelf) return;
            StopAllCoroutines();
            StartCoroutine(PulseRoutine());
        }

        System.Collections.IEnumerator PulseRoutine()
        {
            var img = _canvasGo.transform.Find("DesignPanel")?.GetComponent<Image>();
            if (img == null) yield break;
            float t = 0f;
            while (t < 0.8f)
            {
                t += Time.deltaTime;
                img.color = Color.Lerp(UiKit.Accent, UiKit.Surface, t / 0.8f);
                yield return null;
            }
            img.color = UiKit.Surface;
        }
```

Add the rebuild-on-toggle field and hook it into `Start()`/`Update()`. Replace the field block:

```csharp
        static readonly string[] NamePlaceholders = { "Doom Turtle", "Longshot", "Pathfinder" };

        GameBootstrap _game;
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
        Text _nameBox;
        string _name = "";
        int _placeholderIdx;
```

with:

```csharp
        static readonly string[] NamePlaceholders = { "Doom Turtle", "Longshot", "Pathfinder" };

        GameBootstrap _game;
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
        Text _nameBox;
        string _name = "";
        int _placeholderIdx;
        bool _lastTipsEnabled;
```

Replace `Start()` and `Update()`:

```csharp
        void Start()
        {
            _stats[0] = 1; // Health >= 1
            _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            RefreshSummary();
        }

        // DesignPanel has no StateChanged-driven refresh, so poll: one SetActive per flip of the
        // hide condition (the equality guard keeps it from thrashing the canvas every frame), like
        // GameHud's guard. Hidden during the title demo and the connecting window (no state yet).
        void Update()
        {
            if (_game == null || _canvasGo == null) return;
            bool hidden = _game.DemoMode || _game.State == null;
            if (_canvasGo.activeSelf == hidden)
                _canvasGo.SetActive(!hidden);
        }
```

with:

```csharp
        void Start()
        {
            _stats[0] = 1; // Health >= 1
            _game = FindAnyObjectByType<GameBootstrap>();
            _lastTipsEnabled = TipsService.Enabled;
            Build();
            RefreshSummary();
        }

        // DesignPanel has no StateChanged-driven refresh, so poll: one SetActive per flip of the
        // hide condition (the equality guard keeps it from thrashing the canvas every frame), like
        // GameHud's guard. Hidden during the title demo and the connecting window (no state yet).
        // Also polls the Tips toggle: flipping it while the panel is already built must show/hide the
        // inline captions immediately, which needs a full rebuild (row spacing itself changes).
        void Update()
        {
            if (_game == null || _canvasGo == null) return;
            bool hidden = _game.DemoMode || _game.State == null;
            if (_canvasGo.activeSelf == hidden)
                _canvasGo.SetActive(!hidden);

            if (TipsService.Enabled != _lastTipsEnabled)
            {
                _lastTipsEnabled = TipsService.Enabled;
                Destroy(_canvasGo);
                Build();
                RefreshSummary();
                _canvasGo.SetActive(!hidden); // Build() always creates it active — restore the hidden state above
            }
        }
```

- [ ] **Step 6: `BarracksPanel` — affordable-deploy trigger**

Replace the tail of `Rebuild()`:

```csharp
            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var template = p.Barracks[i];
                int cost = Economy.DeployCost(template.Stats, s.Config);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = UiKit.Button(_list, $"{template.Name}   deploy {cost}", -20f, -(4f + i * 34f), 170f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary);
                UiKit.SetToggled(row, selected);
                _rows.Add(row);

                var del = UiKit.Button(_list, "✕", 100f, -(4f + i * 34f), 32f, 30f,
                                       () => DeleteAt(idx), UiKit.ButtonStyle.Danger, 14);
                del.interactable = !ReadOnly;
                _rows.Add(del);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");
        }
```

with:

```csharp
            int cheapest = int.MaxValue;
            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var template = p.Barracks[i];
                int cost = Economy.DeployCost(template.Stats, s.Config);
                cheapest = Mathf.Min(cheapest, cost);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = UiKit.Button(_list, $"{template.Name}   deploy {cost}", -20f, -(4f + i * 34f), 170f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary);
                UiKit.SetToggled(row, selected);
                _rows.Add(row);

                var del = UiKit.Button(_list, "✕", 100f, -(4f + i * 34f), 32f, 30f,
                                       () => DeleteAt(idx), UiKit.ButtonStyle.Danger, 14);
                del.interactable = !ReadOnly;
                _rows.Add(del);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");

            // spec §6: "First time points ≥ cheapest deploy cost with barracks open" — fires once per
            // game the moment it becomes true, whichever Rebuild() call (StateChanged-driven) sees it first.
            if (p.Barracks.Count > 0 && p.Points >= cheapest)
                TipsService.Show("can-afford-deploy", "Deploying costs the unit's points.");
        }
```

- [ ] **Step 7: `GameHud` — out-of-actions trigger, game-over-vs-AI trigger**

Replace the `done` computation in `Refresh()`:

```csharp
            string done = localHuman && !s.IsGameOver && !anyAction
                ? "     Nothing left to do - press End Turn"
                : "";
```

with:

```csharp
            string done = localHuman && !s.IsGameOver && !anyAction
                ? "     Nothing left to do - press End Turn"
                : "";
            if (done.Length > 0) TipsService.Show("out-of-actions", "Nothing left this turn — End Turn passes play.");
```

Replace the tail of `ShowGameOver`:

```csharp
            if (!_wasOver)
            {
                _wasOver = true;
                var accent = s.Winner == null ? new Color(0.25f, 0.27f, 0.33f, 0.96f)
                           : p0Won ? P0ToastBlue : P1ToastRed;
                StartCoroutine(ShowBannerWhenQuiet(result.ToUpperInvariant(), HowText(s), accent));
            }
        }
```

with:

```csharp
            if (!_wasOver)
            {
                _wasOver = true;
                var accent = s.Winner == null ? new Color(0.25f, 0.27f, 0.33f, 0.96f)
                           : p0Won ? P0ToastBlue : P1ToastRed;
                StartCoroutine(ShowBannerWhenQuiet(result.ToUpperInvariant(), HowText(s), accent));

                // spec §6: game-over nudge, vs AI only. Informational only for now — Task 13 adds
                // GameBootstrap.Rematch() and upgrades this exact call to a CTA that fires it.
                if (FindAnyObjectByType<AiOpponent>() != null)
                    TipsService.Show("game-over-rematch", "Run it back — you know what to build now.");
            }
        }
```

- [ ] **Step 8: `GameBootstrap` — registry reset on new game, first-bounty detection**

Insert `TipsService.NewGame();` as the first statement's neighbor in the three real new-game entry points
(never on a reconnect's START re-deal, which is the same game continuing — Task 6 already keeps that
distinct). In `NewGame()`:

```csharp
        public void NewGame()
        {
            EndDemo();
            Presenter?.ResetQueue();
            SetupEnvironment();
```

with:

```csharp
        public void NewGame()
        {
            EndDemo();
            TipsService.NewGame();
            Presenter?.ResetQueue();
            SetupEnvironment();
```

In `StartLocalGame`:

```csharp
        public void StartLocalGame(GameSetup setup, bool vsAi, AiLevel level = AiLevel.Hard)
        {
            EndDemo();
            Presenter?.ResetQueue();
            Networked = false; // play locally — TryApply applies here instead of going to the server
```

with:

```csharp
        public void StartLocalGame(GameSetup setup, bool vsAi, AiLevel level = AiLevel.Hard)
        {
            EndDemo();
            TipsService.NewGame();
            Presenter?.ResetQueue();
            Networked = false; // play locally — TryApply applies here instead of going to the server
```

In `StartNetGame`:

```csharp
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
```

with:

```csharp
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            TipsService.NewGame();
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
```

Add the first-bounty check, shared by both apply paths (`TryApply`'s local branch and `OnNetApply`). Add
this method near `OnNetApply`:

```csharp
        /// <summary>Spec §6's "first bounty earned" reveal: a kill (AttackUnit that increases the
        /// attacker's Points — CombatResolver awards bounty only on a kill, never a plain hit) issued by
        /// a seat this human controls fires the tip once per game, CTA drawing attention to the Designer.</summary>
        void CheckFirstBounty(GameState prev, Command cmd)
        {
            if (!(cmd is AttackUnit atk) || !IsLocalCommand(cmd)) return;
            int gained = State.Player(atk.Issuer).Points - prev.Player(atk.Issuer).Points;
            if (gained <= 0) return;
            TipsService.Show("first-bounty",
                $"You earned {gained} points. A wall? A sniper? Eyes that see everything? Design your answer.",
                cta: "Design your answer", onCta: OpenDesigner);
        }

        void OpenDesigner() => FindAnyObjectByType<DesignPanel>()?.Highlight();
```

Call it from `TryApply`. Replace:

```csharp
            var prev = State;
            State = result.NewState;
            if (!DemoMode) EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            StateChanged?.Invoke();
            return true;
        }
```

with:

```csharp
            var prev = State;
            State = result.NewState;
            if (!DemoMode) EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            if (!DemoMode) CheckFirstBounty(prev, cmd);
            StateChanged?.Invoke();
            return true;
        }
```

And from `OnNetApply` (the online path — `TryApply` returns early for `Networked` games and never reaches
the block above; this is the actual apply site for online play). Replace:

```csharp
        internal void OnNetApply(Command cmd)
        {
            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) { Debug.LogWarning("[Net] server move rejected locally: " + result.Reason); return; }
            var prev = State;
            State = result.NewState;
            EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            StateChanged?.Invoke();
        }
```

with:

```csharp
        internal void OnNetApply(Command cmd)
        {
            var result = GameEngine.Apply(State, cmd);
            if (!result.Success) { Debug.LogWarning("[Net] server move rejected locally: " + result.Reason); return; }
            var prev = State;
            State = result.NewState;
            EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            CheckFirstBounty(prev, cmd);
            StateChanged?.Invoke();
        }
```

- [ ] **Step 9: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 10: Verify — the full arc on fresh PlayerPrefs**

`execute_script`: `PlayerPrefs.DeleteKey("HexWars.Tips"); PlayerPrefs.DeleteKey("HexWars.SeatToken"); return "cleared";`
(simulates a genuine first-ever visit — Task 6's token key is unrelated but harmless to clear alongside).

Enter Play Mode. `execute_script`: assert `TipsService.Enabled == true` (first-visit default).

Each numbered item below is its OWN `execute_script` call (per the coplay quirk in Global Constraints —
one transition per call):

1. Start a vs-AI game (`StartLocalGame(new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Annihilation, 9, 7, 0, 7), true)`).
2. Select a unit (drive `UnitInputController`'s click path, or directly call the same effect via a script
   that finds a `UnitView` and simulates the tap — simplest is asserting the trigger fires by calling
   `TipsService.Show("first-select", ...)`'s call site indirectly is awkward from script; instead select
   via the real input path: raycast isn't scriptable without pointer simulation, so directly verify the
   MECHANISM instead — `execute_script`: `HexWars.Presentation.TipsService.Show("first-select", "test");
   return HexWars.Presentation.TipsService.Enabled;` then assert `GameObject.Find("TipBubbleCanvas") != null`.
   Screenshot it.
3. Dismiss (`TipBubble.Dismiss()`), then call `TipsService.Show("first-select", "test again")` a second
   time — assert `GameObject.Find("TipBubbleCanvas") == null` immediately (the once-per-game de-dup ate it;
   positive control already proved in step 2 that the mechanism fires when the id is fresh).
4. Open the Designer with Tips on — screenshot: 9 stat rows each with a grey caption line beneath.
5. Drive the game to a kill (script `AttackUnit` commands against the AI's demo army until one lands a kill
   on a unit the human's seat owns the attacker of) — assert the resulting Toast/bubble: screenshot
   `TipBubbleCanvas` showing "You earned N points…" with a "Design your answer" button. Click it — assert
   the bubble is gone AND (best-effort — the pulse is a coroutine) no exception logged.
6. Drive the game to a loss/win — screenshot the game-over banner area; confirm a `TipBubbleCanvas` reading
   "Run it back — you know what to build now." appeared (informational-only per this task, per the forward
   note above).
7. Exit Play Mode, re-enter, `execute_script`: `HexWars.Presentation.TipsService.Enabled = false; return "off";`.
   Repeat steps 1-2 conceptually (start a game, attempt each Show call directly) and assert
   `GameObject.Find("TipBubbleCanvas") == null` after EVERY `TipsService.Show(...)` call — Tips off means
   zero bubbles. **Positive control**: in the SAME state (Tips off), open the Designer and tap a stat name
   (Task 11's direct `TipBubble.Show` path) — assert a bubble STILL appears, proving stat descriptions stay
   "always available" independent of the toggle, and that the "zero bubbles" result above is a real effect
   of the Tips flag rather than something broken in `TipBubble` itself.
8. Toggle Tips back on via the title-screen or HelpOverlay button (screenshot both toggle controls once
   each, confirming their label reads "Tips: Off" then "Tips: On" after a click) and exit Play Mode.

- [ ] **Step 11: Commit**

```bash
git add Assets/HexWars/Presentation/TipsService.cs Assets/HexWars/Presentation/TitleScreen.cs Assets/HexWars/Presentation/HelpOverlay.cs Assets/HexWars/Presentation/UnitInputController.cs Assets/HexWars/Presentation/DesignPanel.cs Assets/HexWars/Presentation/BarracksPanel.cs Assets/HexWars/Presentation/GameHud.cs Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(tips): TipsService coaching layer - persisted opt-out toggle (title + in-game), once-per-game triggers for first selection, designer captions, affordable-deploy, out-of-actions, first-bounty reveal, and the vs-AI game-over nudge"
```

---

### Task 13: Rematch + empty lobby

**Files:**
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (`LastLocalSetup`/`LastLocalAi`/`RematchAvailable`/`Rematch()`)
- Modify: `Assets/HexWars/Presentation/GameOverBanner.cs` (optional Rematch button)
- Modify: `Assets/HexWars/Presentation/GameHud.cs` (wires Rematch into the banner + upgrades Task 12's tip)
- Modify: `Assets/HexWars/Presentation/GameBrowser.cs` (empty-state "Play vs AI")

**Interfaces:**
- Consumes: `SetupForm.Open(GameBootstrap, SetupForm.SetupMode.VsAi)` (existing), `GameSetup` (existing
  fields, per header).
- Produces: `GameBootstrap.RematchAvailable` (bool), `GameBootstrap.Rematch()` — no other task depends on
  these (this is the last task).

- [ ] **Step 1: `GameBootstrap` — remember the last local vs-AI setup, `Rematch()`**

Add properties near `DemoMode`/`Reconnecting`:

```csharp
        /// <summary>True while a started game's socket dropped and NetClient is retrying with backoff.
        /// GameHud reads this to show a persistent status line; OnNetReconnecting Toasts once per drop
        /// episode (not once per attempt).</summary>
        public bool Reconnecting { get; private set; }
```

with:

```csharp
        /// <summary>True while a started game's socket dropped and NetClient is retrying with backoff.
        /// GameHud reads this to show a persistent status line; OnNetReconnecting Toasts once per drop
        /// episode (not once per attempt).</summary>
        public bool Reconnecting { get; private set; }

        /// <summary>The setup/difficulty of the most recent LOCAL vs-AI game — null whenever the most
        /// recent game start was anything else (hotseat, online). Set only by StartLocalGame's vsAi
        /// path; cleared by NewGame()/StartNetGame() so a stale vs-AI setup can never make
        /// RematchAvailable true for a hotseat or online game that started afterward.</summary>
        public GameSetup? LastLocalSetup { get; private set; }
        public AiLevel LastLocalAi { get; private set; }

        /// <summary>Game-over banner shows Rematch only for a local vs-AI game (spec §6).</summary>
        public bool RematchAvailable => LastLocalSetup.HasValue;
```

Replace `NewGame()`'s first lines (already carries Task 12's `TipsService.NewGame()` insert):

```csharp
        public void NewGame()
        {
            EndDemo();
            TipsService.NewGame();
            Presenter?.ResetQueue();
            SetupEnvironment();
```

with:

```csharp
        public void NewGame()
        {
            EndDemo();
            TipsService.NewGame();
            LastLocalSetup = null; // hotseat isn't a vs-AI game — a stale Rematch target must not survive into it
            Presenter?.ResetQueue();
            SetupEnvironment();
```

Replace `StartNetGame`'s first lines:

```csharp
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            TipsService.NewGame();
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
```

with:

```csharp
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            TipsService.NewGame();
            LastLocalSetup = null; // online isn't a vs-AI game — same reasoning as NewGame() above
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
```

Replace `StartLocalGame`'s `vsAi` branch:

```csharp
            if (vsAi)
            {
                var ai = gameObject.AddComponent<AiOpponent>();
                ai.Level = level;
            }
        }
```

(this is the LAST occurrence of this exact block — `StartDemo()` builds a `SpectatorDriver`, not an
`AiOpponent`, so this pattern is unambiguous within the file) with:

```csharp
            if (vsAi)
            {
                var ai = gameObject.AddComponent<AiOpponent>();
                ai.Level = level;
                LastLocalSetup = setup;
                LastLocalAi = level;
            }
        }
```

Add `Rematch()` near `StartLocalGame`:

```csharp
        /// <summary>Game-over banner's Rematch button: same setup, fresh seed, instant restart. A no-op
        /// if the last game wasn't local vs-AI (defensive — the banner only shows the button when
        /// RematchAvailable is already true, so this guard should never actually trigger).</summary>
        public void Rematch()
        {
            if (!LastLocalSetup.HasValue) return;
            var s = LastLocalSetup.Value;
            var reseeded = new GameSetup(s.Mode, s.Width, s.Height, s.StartingPoints,
                                         UnityEngine.Random.Range(1, 99999), s.ArmySize, s.Brutes, s.Strikers,
                                         s.Snipers, s.TurnActions, s.Fog);
            StartLocalGame(reseeded, true, LastLocalAi);
        }
```

- [ ] **Step 2: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 3: `GameOverBanner` — optional Rematch button**

Replace `Show`'s signature and its button block:

```csharp
        public static void Show(string title, string subtitle, Color accent, System.Action onMainMenu = null)
        {
```

with:

```csharp
        public static void Show(string title, string subtitle, Color accent, System.Action onMainMenu = null, System.Action onRematch = null)
        {
```

Replace the `onMainMenu != null` button block at the end of `Show`:

```csharp
            if (onMainMenu != null)
            {
                // band uses a centre anchor (pivot 0.5,0.5); UiKit.Button anchors top-centre (pivot
                // 0.5,1), so the y offset is re-based from the band's centre to its top edge
                // (half its 200-tall rect) to land the button on the exact same pixels as before.
                UiKit.Button(band.transform, "Main menu", 0f, -140f, 220f, 44f,
                             () => { Dismiss(); onMainMenu(); }, UiKit.ButtonStyle.Secondary);
            }
        }
```

with:

```csharp
            // band uses a centre anchor (pivot 0.5,0.5); UiKit.Button anchors top-centre (pivot 0.5,1),
            // so the y offset is re-based from the band's centre to its top edge (half its 200-tall
            // rect) to land buttons on the exact same pixels the single Main-menu button always used.
            if (onRematch != null && onMainMenu != null)
            {
                UiKit.Button(band.transform, "Rematch", -120f, -140f, 220f, 44f,
                             () => { Dismiss(); onRematch(); }, UiKit.ButtonStyle.Cta);
                UiKit.Button(band.transform, "Main menu", 120f, -140f, 220f, 44f,
                             () => { Dismiss(); onMainMenu(); }, UiKit.ButtonStyle.Secondary);
            }
            else if (onMainMenu != null)
            {
                UiKit.Button(band.transform, "Main menu", 0f, -140f, 220f, 44f,
                             () => { Dismiss(); onMainMenu(); }, UiKit.ButtonStyle.Secondary);
            }
            else if (onRematch != null)
            {
                UiKit.Button(band.transform, "Rematch", 0f, -140f, 220f, 44f,
                             () => { Dismiss(); onRematch(); }, UiKit.ButtonStyle.Cta);
            }
        }
```

- [ ] **Step 4: `GameHud` — wire Rematch into the banner + upgrade Task 12's tip to a CTA**

Replace `ShowBannerWhenQuiet`:

```csharp
        System.Collections.IEnumerator ShowBannerWhenQuiet(string title, string how, Color accent)
        {
            var presenter = _game != null ? _game.Presenter : null;
            while (presenter != null && presenter.IsBusy) yield return null;
            GameOverBanner.Show(title, how, accent, onMainMenu: () => _game.ReturnToMenu());
        }
```

with:

```csharp
        System.Collections.IEnumerator ShowBannerWhenQuiet(string title, string how, Color accent)
        {
            var presenter = _game != null ? _game.Presenter : null;
            while (presenter != null && presenter.IsBusy) yield return null;
            System.Action onRematch = _game.RematchAvailable ? (System.Action)_game.Rematch : null;
            GameOverBanner.Show(title, how, accent, onMainMenu: () => _game.ReturnToMenu(), onRematch: onRematch);
        }
```

Replace the game-over tip Task 12 left in `ShowGameOver` (informational-only, per that task's forward
note — this is the promised upgrade):

```csharp
                // spec §6: game-over nudge, vs AI only. Informational only for now — Task 13 adds
                // GameBootstrap.Rematch() and upgrades this exact call to a CTA that fires it.
                if (FindAnyObjectByType<AiOpponent>() != null)
                    TipsService.Show("game-over-rematch", "Run it back — you know what to build now.");
```

with:

```csharp
                // spec §6: game-over nudge, vs AI only — now points at the real Rematch button.
                if (_game.RematchAvailable)
                    TipsService.Show("game-over-rematch", "Run it back — you know what to build now.",
                                     cta: "Rematch", onCta: _game.Rematch);
```

- [ ] **Step 5: `GameBrowser` — empty state offers Play vs AI beside Host Game**

Replace the empty branch in `Rebuild()`:

```csharp
            if (_lastGames.Length == 0)
            {
                _status.text = "No open games right now — host one!";
                UiKit.Button(_listRoot, "Host Game", 0f, -40f, 260f, 48f, () =>
                {
                    Close();
                    SetupForm.Open(_game, SetupForm.SetupMode.Host);
                }, UiKit.ButtonStyle.Cta);
                return;
            }
```

with:

```csharp
            if (_lastGames.Length == 0)
            {
                _status.text = "No open games right now — host one, or play the AI while you wait.";
                UiKit.Button(_listRoot, "Play vs AI", -140f, -40f, 260f, 48f, () =>
                {
                    Close();
                    SetupForm.Open(_game, SetupForm.SetupMode.VsAi);
                }, UiKit.ButtonStyle.Primary);
                UiKit.Button(_listRoot, "Host Game", 140f, -40f, 260f, 48f, () =>
                {
                    Close();
                    SetupForm.Open(_game, SetupForm.SetupMode.Host);
                }, UiKit.ButtonStyle.Cta);
                return;
            }
```

- [ ] **Step 6: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 7: Verify in play mode**

Enter Play Mode. `execute_script`, one transition per call:
1. Start a vs-AI game (`StartLocalGame(new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Annihilation, 9, 7, 0, 7), true, HexWars.Presentation.AiLevel.Easy)`); assert `RematchAvailable == true` and
   `LastLocalSetup.Value.Seed == 7`.
2. Drive the game to completion (script `AttackUnit`/`EndTurn` commands, or reduce to a trivial win via a
   lopsided `armySize`/role setup so it resolves in a few turns) — screenshot the game-over band: Rematch
   + Main menu both visible.
3. Click Rematch (`FindAnyObjectByType<GameBootstrap>().Rematch();` — exercises the same path the button
   fires). Assert: `State.IsGameOver == false`, `State.Config` matches the original mode/width/height/
   points/army composition, and the new `LastLocalSetup.Value.Seed != 7` (fresh seed). Screenshot the
   fresh board.
4. Start a hotseat game (`NewGame()`); assert `RematchAvailable == false`. Drive it to game-over; screenshot
   the band — Main menu only, no Rematch button. (Online: same assertion in spirit, `StartNetGame` also
   clears `LastLocalSetup` — covered by the code path, not re-verified live here since it needs a second
   client; Task 6's own verification already exercises the net path end to end.)
5. Return to menu (`ReturnToMenu()`), open `GameBrowser.Open(game)` with no games hosted — screenshot: both
   "Play vs AI" and "Host Game" visible in the empty state. Click the actual button (not a direct
   `SetupForm.Open` call — this proves the row is wired to the right mode, not a copy-paste onto Host's
   handler): `GameObject.Find("BrowserCanvas").transform.Find("Panel/List").GetChild(0)
   .GetComponent<UnityEngine.UI.Button>().onClick.Invoke();` (`List` is nested under `Panel` under
   `BrowserCanvas`, per `GameBrowser.Build()`; child 0 is "Play vs AI" — built first in the empty branch,
   "Host Game" is child 1). Assert a `SetupForm` component exists and its title text reads "Play vs AI"
   (confirms `SetupMode.VsAi`, not `Host`).
6. Exit Play Mode.

- [ ] **Step 8: Commit**

```bash
git add Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/GameOverBanner.cs Assets/HexWars/Presentation/GameHud.cs Assets/HexWars/Presentation/GameBrowser.cs
git commit -m "feat(delight): Rematch button on the game-over banner (vs-AI games only, same setup + fresh seed); empty lobby offers Play vs AI beside Host Game"
```

---
### Task 14: Final regression + ship

This task assumes Tasks 1-13 are all committed on the milestone branch (Tasks 1-5 above; Tasks 6-13 are
the client/Unity half — Designer name field, DeleteTemplate ✕ button, Tips service, portrait pass,
rematch, share/PWA — drafted separately, not in this file). It is a verification-and-release task, not a
TDD task: there is no new production code here, so there is no red/green cycle — every step is a gate
the branch must pass before it merges and ships.

**Files:** none created or modified by this task except `wwwroot/` (staged WebGL build output, committed
in Step 5) and the merge commit itself.

**Interfaces:** consumes the fully merged state of Tasks 1-13 (engine + client). Produces: a merged
`main` with a staged, deployable WebGL build, pushed to the remote.

- [ ] **Step 1: Full engine test suite + selftest**

```
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q
dotnet run --project engine/HexWars.NetServer -- selftest
```
Expected: all engine tests pass at whatever the cumulative total is after Tasks 1-13 (this file's
Tasks 1-5 alone bring the baseline from 266 to 309; Tasks 6-13 add client-side engine/session tests on
top of that — read the actual count off the branch, don't assume a number here). `SELFTEST PASS`, exit 0.
If either fails, stop — do not proceed to editor/build verification on a red suite.

- [ ] **Step 2: Editor regression (coplay-driven, per `hexwars-playmode-verification` conventions)**

Drive the editor via `mcp__coplay-mcp__execute_script` / `check_compile_errors` / `get_unity_logs` —
scene is `Assets/Scenes/HexWars.unity`. Verify, in order:
1. Zero compile errors on an idle, fully-compiled editor (`check_compile_errors`).
2. Hotseat boot: enter play mode, start a hotseat (local vs-AI or 2-local-player) game from the title
   flow, confirm the board renders, a unit is selectable, and the barracks shows the five named starter
   templates (Task 3) with a working ✕ delete (Task 2/Tasks 6-13's UI) that doesn't consume a turn.
3. Title demo: confirm the attract-mode/title demo still plays without error (no regression from any
   engine-assembly change across Tasks 1-5 — a stale/mismatched DLL would break this first).
4. `ReturnToMenu`: from mid-game, back out to the title screen and confirm no leaked state (a fresh game
   after `ReturnToMenu` seeds barracks correctly again, per Task 3).
5. Exit play mode; re-run `check_compile_errors` to confirm play mode didn't leave the editor dirty.

Screenshot each of steps 2-4 (`capture_scene_object`/`capture_ui_canvas` per the play-mode-verification
memory) as the evidence artifact for this step.

- [ ] **Step 3: WebGL build**

Before triggering: count existing `[WebGLBuild] result=` markers in BOTH
`C:\Users\cddal\HexWars\Logs\Editor.log` AND `%LOCALAPPDATA%\Unity\Editor\Editor.log` (the active log
location flips between the two — check both so a stale count from the wrong file isn't mistaken for "no
build yet").

Trigger the build via `EditorApplication.delayCall` ONLY when the editor is idle and fully compiled (a
build queued during compilation or play mode silently no-ops or corrupts the output). Then:
- Verify the log file being written to actually moved (grew) within 90 seconds of triggering — if
  neither log file is advancing, the delayCall never fired (editor was mid-domain-reload) and the trigger
  must be re-issued after confirming idle state.
- A full WebGL build takes roughly 7-8 minutes; an incremental build (no asset/script changes since the
  last one) can complete in seconds — don't assume a hang just because it returns fast.
- Confirm a new `[WebGLBuild] result=Success` marker appears in whichever log file is active.

- [ ] **Step 4: Stage the deploy**

```powershell
powershell -ExecutionPolicy Bypass -File engine\stage-webgl-deploy.ps1
```
(Run from the main session, not a subagent — matches the established pattern from prior deploy tasks.)
This copies the fresh `Build/` output into `wwwroot/`, rewrites `index.html` for cache-busting, and (per
spec §4) injects the OpenGraph/Twitter-card tags, favicon, `manifest.json`, and `theme-color` — this is
the one authority for post-build HTML edits, so Tasks 6-13's share/PWA work must have landed its static
assets (icons/screenshot) under `wwwroot/` outside `Build/` before this step, or the injected tags will
point at missing files. Spot-check `wwwroot/index.html` after staging for the OG tags and manifest link.

- [ ] **Step 5: Commit the staged build**

```bash
git add wwwroot
git commit -m "deploy: client build with invite-readiness (reconnect/rejoin, server hardening, named starter templates + deletion, session-longevity fixes, portrait pass, share/PWA, Tips coaching arc)"
```

- [ ] **Step 6: Merge to main and push**

```bash
git checkout main
git merge --no-ff feat/invite-readiness
```
Resolve any conflicts (there should be none if Tasks 1-13 all branched from an up-to-date `feat/invite-readiness`
and merged forward cleanly). Then push from WSL (per the `hexwars-deploy-and-git` memory — the Windows
shell has no deploy key):
```bash
wsl git -C /mnt/c/Users/cddal/HexWars push
```

- [ ] **Step 7: The user's live-smoke checklist (spec §8) — closing step, not automatable**

Hand this checklist to the user; it is the final gate before calling the milestone shipped:
- Phone backgrounding mid-game → auto-reconnect (no message, no manual refresh needed).
- Refresh mid-game → lands back in the same seat, same game (not a fresh room).
- Paste the link into a chat app → the preview card looks like a game (title, description, screenshot),
  not a bare dev URL.
- "Add to Home Screen" on a phone produces a proper icon/name (PWA manifest).
- First-ever visit with Tips on: the coaching arc fires through to the first-bounty Designer reveal
  ("You earned N points... Design your answer.") without ever blocking input or stacking bubbles.
- Toggling Tips off makes the coaching layer completely silent for the rest of the session.

---


---

## Assembly self-review (controller notes)

- **Spec coverage:** every spec section maps to a task — §3 reconnect/rejoin → T4 (server) + T6
  (client), §3 hardening batch → T5, §4 longevity/portrait/share → T7/T8/T9, §5 templates →
  T1/T2/T3 + T10, §6 teaching layer → T11/T12 + help copy in T11, rematch/empty-lobby → T13,
  §8 verification → per-task gates + T14. No orphaned requirements found.
- **Seam check (verified by cross-grep):** client tasks consume exactly the engine tasks'
  produced signatures — `UnitTemplate.Stats`, `CreateUnit(issuer, stats, name)` with defaulted
  name, `DeleteTemplate(issuer, index)`, `&token=` connect param, `TipBubble.Show(text, pos,
  cta, onCta)` produced by T11 and consumed by T12.
- **Compile-order risk resolved:** T1 Step 6b shims the one breaking presentation call site
  (BarracksPanel) so T6-T9's compile gates stay green before T10/T11 land the real UI.
- **The drafters' flagged conflicts stand as documented** in "Known conflicts" (GameFactory shim,
  NetDisconnectTests contract rewrite with un-started instant-cleanup explicitly preserved and
  tested, `: Command(Issuer)` syntax, DeleteTemplate auto-end-turn exclusion).
- **Portrait legibility note:** T8 fixes clipping and applies clamps; overall text scale on
  portrait (a `matchWidthOrHeight` question) is explicitly deferred — revisit only if live
  phone feedback says text is too small.
- The gitignored plugin DLL was removed from all commit steps (sync-only, per repo convention).
