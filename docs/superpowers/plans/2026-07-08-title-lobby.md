# Title Screen & Game Lobby Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The web build gets a title screen (main menu over a live AI-vs-AI demo game) and a real lobby: browse open games with their configs, join by match code, host public/private games, play vs AI — plus a shared UiKit that makes every menu look like one product.

**Architecture:** HTTP `GET /games` list endpoint on the existing NetServer + the existing WebSocket join path (Approach A per the spec). Client side: `UiKit` (style system) → demo mode on `GameBootstrap` → `SetupForm` (refactored LobbyPanel) → `GameBrowser` → `TitleScreen` wires it all. Engine changes are confined to `Net\` (MatchHub, GameSetup) + GameFactory.

**Tech Stack:** C# netstandard2.1 engine (NUnit tests), ASP.NET minimal API server, Unity 6 runtime-uGUI (no prefabs/assets), coplay MCP for play-mode verification.

**Spec:** `docs/superpowers/specs/2026-07-08-title-lobby-design.md`

## Global Constraints

- **NEVER add attribution trailers to git commits** — no `Co-Authored-By`, no "Generated with Claude Code", no tool credits of any kind. (User's global CLAUDE.md; overrides all defaults.)
- Engine assembly changes are confined to `engine/HexWars.Engine/Net/` + one line in `GameFactory.Build`; no game-rules changes; all 254 existing engine tests must stay green.
- **After any engine-assembly change:** run `powershell -File engine/build-to-unity.ps1` to re-sync `Assets/HexWars/Plugins/HexWars.Engine.dll`, then verify Unity compiles (`mcp__coplay-mcp__check_compile_errors`) — a stale/broken DLL boots Unity into Safe Mode and kills the Coplay MCP.
- Unity work is verified in play mode via coplay `execute_script` on `Assets/Scenes/HexWars.unity`; **every new/changed screen gets a screenshot** (capture_scene_object / capture_ui_canvas) — lobby screens are verified visually, per standing practice.
- Coplay `execute_script` quirks: no local functions, no type-pattern-matching in scripts; `Time.deltaTime` reads 0; use positive controls for absence assertions.
- The wire protocol (SEAT/START/APPLY/REJECT) must not change. `?room=` links must keep working (boot bypasses title → joins directly).
- The deployed WebGL build is always the online client (`Networked = true` on WebGL); the editor default path (`Networked = false` → `NewGame()` hotseat) must keep working untouched.
- Work on branch `feat/title-lobby` off main.
- Commit after each task. Final deploy: WebGL build → `powershell -File engine/stage-webgl-deploy.ps1` → commit wwwroot → push via WSL (`wsl git -C /mnt/c/Users/cddal/HexWars push` — Windows shell has no git key).

---

### Task 1: `GameSetup.Sanitized()` + clamp in `GameFactory.Build`

The public server currently builds whatever board a query string asks for (`?setup=0 9999 9999 …` → ~10⁸ tiles OOM under the global hub lock). Clamp every field at the one choke point all construction paths share.

**Files:**
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs` (add `Sanitized()`; call it at the top of `GameFactory.Build`)
- Test: `engine/HexWars.Engine.Tests/GameSetupSanitizeTests.cs` (create)

**Interfaces:**
- Produces: `public GameSetup Sanitized()` on `GameSetup`; `GameFactory.Build(setup)` self-sanitizes (later tasks rely on this — the server never clamps explicitly).

- [ ] **Step 1: Write the failing tests**

Create `engine/HexWars.Engine.Tests/GameSetupSanitizeTests.cs`:

```csharp
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Sanitized() clamps every field to the lobby form's own ranges so a hostile or
    /// corrupt ?setup= query can never OOM the server (audit finding N1). GameFactory.Build
    /// self-sanitizes, covering every construction path.</summary>
    public class GameSetupSanitizeTests
    {
        [Test]
        public void Sanitized_ClampsOversizedBoardAndArmy()
        {
            var s = new GameSetup(GameMode.Annihilation, 9999, 9999, 100000, 7, 500000, 99, 99, 99, 99).Sanitized();
            Assert.That(s.Width, Is.EqualTo(24));
            Assert.That(s.Height, Is.EqualTo(24));
            Assert.That(s.StartingPoints, Is.EqualTo(200));
            Assert.That(s.ArmySize, Is.EqualTo(12));
            Assert.That(s.Brutes, Is.EqualTo(12));
            Assert.That(s.Strikers, Is.EqualTo(12));
            Assert.That(s.Snipers, Is.EqualTo(12));
            Assert.That(s.TurnActions, Is.EqualTo(8));
        }

        [Test]
        public void Sanitized_ClampsUndersizedValues()
        {
            var s = new GameSetup((GameMode)99, 1, -5, -10, -3, 0, -1, -1, -1, -2).Sanitized();
            Assert.That((int)s.Mode, Is.InRange(0, 1));
            Assert.That(s.Width, Is.EqualTo(5));
            Assert.That(s.Height, Is.EqualTo(5));
            Assert.That(s.StartingPoints, Is.EqualTo(0));
            Assert.That(s.Seed, Is.EqualTo(1));
            Assert.That(s.ArmySize, Is.EqualTo(1));
            Assert.That(s.Brutes, Is.EqualTo(0));
            Assert.That(s.TurnActions, Is.EqualTo(0));
        }

        [Test]
        public void Sanitized_LegalValuesPassThroughUnchanged()
        {
            var input = new GameSetup(GameMode.Territory, 13, 9, 40, 1234, 5, 2, 2, 1, 3, fog: true);
            var s = input.Sanitized();
            Assert.That(s.ToWire(), Is.EqualTo(input.ToWire()));
        }

        [Test]
        public void GameFactoryBuild_SanitizesHostileSetup()
        {
            // must complete instantly with a clamped board, not build 9999x9999 tiles
            var state = GameFactory.Build(GameSetup.Parse("0 9999 9999 0 7"));
            Assert.That(state.Board.Tiles.Count, Is.EqualTo(24 * 24));
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

Run (from repo root): `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q --filter GameSetupSanitize`
Expected: compile error — `'GameSetup' does not contain a definition for 'Sanitized'`.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/Net/GameSetup.cs`, add inside the `GameSetup` type (after `Parse`, before the closing brace):

```csharp
        /// <summary>Every field clamped to the lobby form's own ranges. GameFactory.Build calls this,
        /// so no construction path — including a hostile ?setup= query on the public server — can
        /// request an absurd board or army (that was a one-request OOM before).</summary>
        public GameSetup Sanitized() => new GameSetup(
            (GameMode)Math.Clamp((int)Mode, 0, 1),
            Math.Clamp(Width, 5, 24),
            Math.Clamp(Height, 5, 24),
            Math.Clamp(StartingPoints, 0, 200),
            Math.Clamp(Seed, 1, 99999),
            Math.Clamp(ArmySize, 1, 12),
            Math.Clamp(Brutes, 0, 12),
            Math.Clamp(Strikers, 0, 12),
            Math.Clamp(Snipers, 0, 12),
            Math.Clamp(TurnActions, 0, 8),
            Fog);
```

Add `using System;` to the file's usings if not present. In the same file, first line of `GameFactory.Build(GameSetup setup)`:

```csharp
            setup = setup.Sanitized();
```

- [ ] **Step 4: Run tests**

`dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: all pass (254 existing + 4 new).

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine.Tests/GameSetupSanitizeTests.cs
git commit -m "feat(net): GameSetup.Sanitized clamps every field; GameFactory self-sanitizes - closes the unvalidated ?setup= OOM hole"
```

---

### Task 2: MatchHub lobby state — setup/privacy/started/age + `OpenGames()`

**Files:**
- Modify: `engine/HexWars.Engine/Net/MatchHub.cs`
- Test: `engine/HexWars.Engine.Tests/MatchHubLobbyTests.cs` (create)
- After GREEN: run `powershell -File engine/build-to-unity.ps1`, then coplay `check_compile_errors`.

**Interfaces:**
- Consumes: `GameSetup` (Task 1 unchanged surface).
- Produces (Task 3/7 rely on these exact signatures):
  - `public MatchHub(Func<GameSetup, GameState> newGame, Func<long> utcNowTicks = null)`
  - `public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default, bool isPrivate = false)`
  - `public IReadOnlyList<OpenGame> OpenGames()` — newest first
  - `public readonly struct OpenGame { public readonly string Code; public readonly GameSetup Setup; public readonly int AgeSeconds; }`

- [ ] **Step 1: Write the failing tests**

Create `engine/HexWars.Engine.Tests/MatchHubLobbyTests.cs`:

```csharp
using System;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>The lobby list: OpenGames() exposes rooms a browser can join — public, exactly one
    /// seated member, game not yet started. Age comes from an injected clock so tests are exact.</summary>
    public class MatchHubLobbyTests
    {
        private long _now;
        private MatchHub NewHub()
        {
            _now = TimeSpan.FromHours(1).Ticks;
            return new MatchHub(GameFactory.Build, () => _now);
        }

        [Test]
        public void WaitingPublicRoom_IsListed_WithSetupAndAge()
        {
            var hub = NewHub();
            hub.Connect("KQ7KP", "host", new GameSetup(GameMode.Territory, 13, 9, 40, 7, 5, 2, 2, 1, 3, fog: true));
            _now += TimeSpan.FromSeconds(120).Ticks;

            var open = hub.OpenGames();
            Assert.That(open.Count, Is.EqualTo(1));
            Assert.That(open[0].Code, Is.EqualTo("KQ7KP"));
            Assert.That(open[0].Setup.Mode, Is.EqualTo(GameMode.Territory));
            Assert.That(open[0].Setup.Width, Is.EqualTo(13));
            Assert.That(open[0].Setup.Fog, Is.True);
            Assert.That(open[0].AgeSeconds, Is.EqualTo(120));
        }

        [Test]
        public void PrivateRoom_NeverListed()
        {
            var hub = NewHub();
            hub.Connect("SECRET", "host", GameSetup.Default, isPrivate: true);
            Assert.That(hub.OpenGames(), Is.Empty);
        }

        [Test]
        public void StartedRoom_NotListed_EvenIfAMemberDrops()
        {
            var hub = NewHub();
            hub.Connect("R", "a");
            hub.Connect("R", "b");                       // second seat -> START dealt -> Started
            Assert.That(hub.OpenGames(), Is.Empty, "full room is not open");
            hub.Disconnect("R", "b");                    // back to one member, but the game began
            Assert.That(hub.OpenGames(), Is.Empty, "a started room must never re-list");
        }

        [Test]
        public void EmptiedRoom_IsRemoved_NotListed()
        {
            var hub = NewHub();
            hub.Connect("R", "a");
            hub.Disconnect("R", "a");
            Assert.That(hub.OpenGames(), Is.Empty);
        }

        [Test]
        public void OpenGames_NewestFirst()
        {
            var hub = NewHub();
            hub.Connect("OLD", "a");
            _now += TimeSpan.FromSeconds(60).Ticks;
            hub.Connect("NEW", "b");
            var open = hub.OpenGames();
            Assert.That(open.Select(g => g.Code), Is.EqualTo(new[] { "NEW", "OLD" }));
            Assert.That(open[0].AgeSeconds, Is.EqualTo(0));
            Assert.That(open[1].AgeSeconds, Is.EqualTo(60));
        }

        [Test]
        public void JoinerPrivacyFlag_Ignored_HostDecides()
        {
            var hub = NewHub();
            hub.Connect("R", "host", GameSetup.Default, isPrivate: false);
            var full = hub.Connect("R", "x", GameSetup.Default, isPrivate: true); // joiner flag must not flip the room
            // room is now full (started) so it's unlisted for THAT reason; verify via a fresh public room
            hub.Connect("R2", "h2", GameSetup.Default, isPrivate: false);
            Assert.That(hub.OpenGames().Select(g => g.Code), Is.EqualTo(new[] { "R2" }));
        }
    }
}
```

- [ ] **Step 2: Run to verify failure**

`dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q --filter MatchHubLobby`
Expected: compile error — no `OpenGames`, no 2-arg MatchHub ctor.

- [ ] **Step 3: Implement**

In `engine/HexWars.Engine/Net/MatchHub.cs`:

Add after the `Outbound` struct (file scope, same namespace):

```csharp
    /// <summary>One joinable lobby entry: a public room with a host waiting and a game not yet begun.</summary>
    public readonly struct OpenGame
    {
        public readonly string Code;
        public readonly GameSetup Setup;
        public readonly int AgeSeconds;
        public OpenGame(string code, GameSetup setup, int ageSeconds) { Code = code; Setup = setup; AgeSeconds = ageSeconds; }
    }
```

Replace the `Room` class with:

```csharp
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
```

Replace the ctor and `_newGame` field block with:

```csharp
        private readonly Func<GameSetup, GameState> _newGame;
        private readonly Func<long> _now;
        private readonly Dictionary<string, Room> _rooms = new Dictionary<string, Room>();

        /// <summary>The clock is injectable so lobby ages are exactly testable; production uses UTC.</summary>
        public MatchHub(Func<GameSetup, GameState> newGame, Func<long> utcNowTicks = null)
        { _newGame = newGame; _now = utcNowTicks ?? (() => DateTime.UtcNow.Ticks); }
```

Change `Connect`'s signature and room creation + start-dealing:

```csharp
        public IReadOnlyList<Outbound> Connect(string roomCode, string connectionId, GameSetup setup = default, bool isPrivate = false)
        {
            var outs = new List<Outbound>();
            if (!_rooms.TryGetValue(roomCode, out var room))
            {
                room = new Room(_newGame(setup), setup, isPrivate, _now());
                _rooms[roomCode] = room;
            }
```

(the rest of the method is unchanged except) — inside the `if (added && room.Members.Count == 2)` block, first line:

```csharp
                room.Started = true;
```

Add the query method after `Connect`:

```csharp
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
```

- [ ] **Step 4: Run the full engine suite**

`dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q`
Expected: all pass (existing MatchHubTests compile unchanged thanks to default params).

- [ ] **Step 5: Re-sync the Unity plugin DLL and verify Unity compiles**

```
powershell -File engine/build-to-unity.ps1
```
Then coplay `check_compile_errors` → expect zero errors (Unity must not be in Safe Mode).

- [ ] **Step 6: Commit**

```bash
git add engine/HexWars.Engine/Net/MatchHub.cs engine/HexWars.Engine.Tests/MatchHubLobbyTests.cs Assets/HexWars/Plugins/HexWars.Engine.dll
git commit -m "feat(net): MatchHub lobby state - rooms keep setup/privacy/started/age, OpenGames() lists joinable public rooms (injected clock)"
```

---

### Task 3: NetServer — `GET /games`, `?private=`, room-code normalization, selftest

**Files:**
- Modify: `engine/HexWars.NetServer/Program.cs`
- Modify: `engine/HexWars.NetServer/SelfTest.cs`

**Interfaces:**
- Consumes: `Hub.OpenGames()`, `Hub.Connect(room, id, setup, isPrivate)` (Task 2).
- Produces (Task 7's GameBrowser relies on this exact JSON): `GET /games` → `200 {"games":[{"code":"KQ7KP","mode":"Territory","width":13,"height":9,"fog":true,"pace":3,"army":5,"ageSeconds":120}]}`. WS query gains `&private=1`. Room codes normalized server-side: trim → uppercase → strip non-`[A-Z0-9]` → cap 16 chars → empty becomes `"default"` (uppercased to `DEFAULT` is fine — normalization applies to it too; the constant only matters as a fallback bucket).

- [ ] **Step 1: Implement server changes**

In `engine/HexWars.NetServer/Program.cs`:

After `app.MapGet("/healthz", ...)` add:

```csharp
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
```

Add `using System.Linq;` if missing (top of file).

In `Handle`, replace the two room lines with normalization + private flag:

```csharp
            string room = NormalizeRoom(ctx.Request.Query["room"].ToString());
            bool isPrivate = ctx.Request.Query["private"].ToString() == "1";
            var setup = ParseSetup(ctx.Request.Query["setup"].ToString());
```

and pass the flag through in the connect call:

```csharp
                await Dispatch(Locked(() => Hub.Connect(room, conn.Id, setup, isPrivate)));
```

Add the helper next to `ParseSetup`:

```csharp
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
```

- [ ] **Step 2: Extend the selftest**

The hub state (not MapGet plumbing) is what needs testing, so the selftest queries the hub through a small internal accessor rather than standing up the HTTP route. Add to `Program` (next to `Locked`):

```csharp
        /// <summary>Selftest hook: a locked snapshot of the lobby list.</summary>
        internal static IReadOnlyList<OpenGame> OpenGamesSnapshot() { lock (HubLock) return Hub.OpenGames(); }
```

Then in `SelfTest.Run`, after `string seatA = await Recv(a);` insert:

```csharp
                var lobby1 = Program.OpenGamesSnapshot();      // host waiting -> the room is browsable
                bool lobbyListsWaitingRoom = lobby1.Count == 1 && lobby1[0].Code == "TEST";
```

and after `string startB = await Recv(b);` insert:

```csharp
                var lobby2 = Program.OpenGamesSnapshot();      // both seated -> started -> unlisted
                bool lobbyEmptiesOnStart = lobby2.Count == 0;
```

and fold both into the `ok` conjunction:

```csharp
                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart;
```

and extend the FAIL message with `lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart}`. (`?room=test` normalizes to `TEST` — the assertion also proves normalization.)

- [ ] **Step 3: Run the selftest**

`dotnet run --project engine/HexWars.NetServer -- selftest`
Expected: `SELFTEST PASS …`, exit 0.

- [ ] **Step 4: Manual endpoint smoke**

Start the server in the background (`dotnet run --project engine/HexWars.NetServer`), then:
`curl -s http://127.0.0.1:5000/games` (or the port it binds; default launch profile) → expect `{"games":[]}`. Stop the server.
(If the default port differs, read it from the startup log — the exact port doesn't matter, the JSON shape does.)

- [ ] **Step 5: Commit**

```bash
git add engine/HexWars.NetServer/Program.cs engine/HexWars.NetServer/SelfTest.cs
git commit -m "feat(server): GET /games lobby list, ?private=1 rooms, room-code normalization; selftest covers list/unlist lifecycle"
```

---

### Task 4: `UiKit` — the shared style system

**Files:**
- Create: `Assets/HexWars/Presentation/UiKit.cs`

**Interfaces:**
- Produces (Tasks 5–9 build every screen on exactly these):

```csharp
public static class UiKit
{
    // palette
    public static readonly Color Bg, Surface, SurfaceBorder, Accent, AccentDim, CtaGreen, Danger,
                                 TextMain, TextDim, TextFaint, InputBg, InputText;
    // type scale
    public const int SizeTitle = 26, SizeHeading = 20, SizeBody = 16, SizeCaption = 13;
    // canvas sorting orders (one authority)
    public const int OrderHud = 500, OrderPanels = 700, OrderTooltip = 750, OrderToast = 800,
                     OrderBanner = 850, OrderRules = 900, OrderMenu = 1000;
    public static Font Font();
    public static void EnsureEventSystem();
    public static GameObject Canvas(string name, int sortingOrder, Transform parent);
    public static Sprite Rounded();                       // cached 9-sliced rounded-rect
    public static Image Panel(Transform parent, string name, Color color);       // stretched? no — bare image+sprite, caller sets rect
    public static void Stretch(RectTransform rt);
    public static void SetRect(RectTransform rt, float x, float y, float w, float h); // top-center anchor convention
    public static Text Label(Transform parent, string text, float x, float y, float w, float h,
                             int size, TextAnchor anchor, Color? color = null);
    public enum ButtonStyle { Primary, Secondary, Cta, Danger }
    public static Button Button(Transform parent, string label, float x, float y, float w, float h,
                                Action onClick, ButtonStyle style = ButtonStyle.Primary, int fontSize = 0);
    public static void SetToggled(Button b, bool on);     // selected-state tint for toggle buttons
    public static Text ValueBox(Transform parent, string label, float x, float y, float w, float h,
                                Func<int> get, Action<int> set, int min, int max);   // tap-to-type via browser prompt
}
```

- [ ] **Step 1: Write the file**

Create `Assets/HexWars/Presentation/UiKit.cs`:

```csharp
using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The single source of UI style: one canvas convention (1600x900 @ match 0.5), one palette, one
    /// type scale, one rounded-corner sprite, one widget factory. Every runtime menu/panel builds on
    /// this so the game reads as one product instead of nine hand-rolled forms. Zero assets: the
    /// rounded sprite is generated once at runtime (WebGL build stays art-free).
    /// </summary>
    public static class UiKit
    {
        // ---- palette ----
        public static readonly Color Bg            = Hex("0A0E1C");
        public static readonly Color Surface       = Hex("161B2C");
        public static readonly Color SurfaceBorder = Hex("2A3350");
        public static readonly Color Accent        = Hex("45AEFF");
        public static readonly Color AccentDim     = Hex("27476B");
        public static readonly Color CtaGreen      = Hex("33845C");
        public static readonly Color Danger        = Hex("B04040");
        public static readonly Color TextMain      = Color.white;
        public static readonly Color TextDim       = Hex("9AA3B8");
        public static readonly Color TextFaint     = Hex("6C7488");
        public static readonly Color InputBg       = Hex("EDF1F8");
        public static readonly Color InputText     = Hex("10131C");

        // ---- type scale ----
        public const int SizeTitle = 26, SizeHeading = 20, SizeBody = 16, SizeCaption = 13;

        // ---- canvas sorting orders (single authority; comments = who owns it) ----
        public const int OrderHud = 500;      // GameHud top bar
        public const int OrderPanels = 700;   // barracks / design side panels
        public const int OrderTooltip = 750;  // unit tooltip
        public const int OrderToast = 800;    // toasts
        public const int OrderBanner = 850;   // game-over band
        public const int OrderRules = 900;    // rules/help popup
        public const int OrderMenu = 1000;    // title / lobby screens

        static Color Hex(string rgb)
        {
            byte r = Convert.ToByte(rgb.Substring(0, 2), 16);
            byte g = Convert.ToByte(rgb.Substring(2, 2), 16);
            byte b = Convert.ToByte(rgb.Substring(4, 2), 16);
            return new Color32(r, g, b, 255);
        }

        public static Font Font()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }

        public static void EnsureEventSystem()
        {
            if (UnityEngine.Object.FindAnyObjectByType<EventSystem>() != null) return;
            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            var module = es.AddComponent<InputSystemUIInputModule>();
            module.AssignDefaultActions(); // without actions the module silently ignores UI input
        }

        /// <summary>ScreenSpaceOverlay canvas on the shared 1600x900 @ match 0.5 convention.</summary>
        public static GameObject Canvas(string name, int sortingOrder, Transform parent)
        {
            var go = new GameObject(name);
            if (parent != null) go.transform.SetParent(parent, false);
            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = sortingOrder;
            var scaler = go.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            go.AddComponent<GraphicRaycaster>();
            return go;
        }

        static Sprite _rounded;

        /// <summary>A 9-sliced rounded-rect sprite (generated once). Radius reads ~8px at reference scale.</summary>
        public static Sprite Rounded()
        {
            if (_rounded != null) return _rounded;
            const int s = 32, r = 10;
            var tex = new Texture2D(s, s, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[s * s];
            for (int y = 0; y < s; y++)
                for (int x = 0; x < s; x++)
                {
                    // distance outside the rounded-rect core; soft 1px edge for cheap anti-aliasing
                    float dx = Mathf.Max(0, Mathf.Max(r - x, x - (s - 1 - r)));
                    float dy = Mathf.Max(0, Mathf.Max(r - y, y - (s - 1 - r)));
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    float a = Mathf.Clamp01(r + 0.5f - d);
                    px[y * s + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            tex.SetPixels32(px);
            tex.Apply();
            _rounded = Sprite.Create(tex, new Rect(0, 0, s, s), new Vector2(0.5f, 0.5f), 100f, 0,
                                     SpriteMeshType.FullRect, new Vector4(r + 2, r + 2, r + 2, r + 2));
            return _rounded;
        }

        /// <summary>Rounded surface image. Caller positions it (SetRect/Stretch/manual anchors).</summary>
        public static Image Panel(Transform parent, string name, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.sprite = Rounded();
            img.type = Image.Type.Sliced;
            img.color = color;
            return img;
        }

        public static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        /// <summary>Top-center anchored (x, y, w, h) — the convention every form in the game uses.</summary>
        public static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, y);
        }

        public static Text Label(Transform parent, string text, float x, float y, float w, float h,
                                 int size, TextAnchor anchor, Color? color = null)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = Font(); t.fontSize = size; t.color = color ?? TextMain;
            t.alignment = anchor; t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            return t;
        }

        public enum ButtonStyle { Primary, Secondary, Cta, Danger }

        static Color BaseColor(ButtonStyle s) => s switch
        {
            ButtonStyle.Cta => CtaGreen,
            ButtonStyle.Danger => Danger,
            ButtonStyle.Secondary => new Color(0.13f, 0.16f, 0.24f, 1f),
            _ => AccentDim,
        };

        /// <summary>Rounded button with hover/pressed tints (uGUI ColorBlock, so states come free).</summary>
        public static Button Button(Transform parent, string label, float x, float y, float w, float h,
                                    Action onClick, ButtonStyle style = ButtonStyle.Primary, int fontSize = 0)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.sprite = Rounded();
            img.type = Image.Type.Sliced;
            img.color = Color.white; // tint comes from the ColorBlock so hover/pressed work
            var b = go.AddComponent<Button>();
            b.targetGraphic = img;
            var cb = b.colors;
            var baseC = BaseColor(style);
            cb.normalColor = baseC;
            cb.highlightedColor = baseC * 1.18f;
            cb.pressedColor = baseC * 0.82f;
            cb.selectedColor = baseC;
            cb.disabledColor = new Color(baseC.r, baseC.g, baseC.b, 0.35f);
            cb.fadeDuration = 0.08f;
            b.colors = cb;
            b.onClick.AddListener(() => onClick());
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            Label(go.transform, label, 0f, 0f, w, h,
                  fontSize > 0 ? fontSize : (style == ButtonStyle.Cta ? SizeHeading + 2 : SizeBody + 2),
                  TextAnchor.MiddleCenter);
            return b;
        }

        /// <summary>Selected-state tint for toggle-style buttons (mode pickers, checkboxes-as-buttons).</summary>
        public static void SetToggled(Button b, bool on)
        {
            var cb = b.colors;
            var baseC = on ? new Color(0.27f, 0.50f, 0.82f, 1f) : new Color(0.13f, 0.16f, 0.24f, 1f);
            cb.normalColor = baseC;
            cb.highlightedColor = baseC * 1.18f;
            cb.pressedColor = baseC * 0.82f;
            cb.selectedColor = baseC;
            b.colors = cb;
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        [System.Runtime.InteropServices.DllImport("__Internal")]
        static extern string HexWarsPrompt(string message, string current);
#endif
        /// <summary>Tap-to-type int prompt: browser prompt() on WebGL (the only reliable mobile
        /// keyboard), no-op in the editor (use the −/+ steppers there).</summary>
        public static int PromptInt(string label, int current)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            string s = HexWarsPrompt(label, current.ToString());
            return int.TryParse(s, out var v) ? v : current;
#else
            return current;
#endif
        }

        /// <summary>Same for free text (join-by-code). Returns null when unavailable/cancelled.</summary>
        public static string PromptText(string label, string current)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            return HexWarsPrompt(label, current);
#else
            return null;
#endif
        }

        /// <summary>Light input-style box showing an int; tap to type (WebGL), value clamped.</summary>
        public static Text ValueBox(Transform parent, string label, float x, float y, float w, float h,
                                    Func<int> get, Action<int> set, int min, int max)
        {
            var img = Panel(parent, "ValueBox", InputBg);
            var go = img.gameObject;
            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            var t = Label(go.transform, get().ToString(), 0f, 0f, w, h, SizeBody + 4, TextAnchor.MiddleCenter, InputText);
            btn.onClick.AddListener(() => { set(Mathf.Clamp(PromptInt(label, get()), min, max)); t.text = get().ToString(); });
            return t;
        }
    }
}
```

Note: `HexWarsPrompt` must accept being imported from two classes — the jslib function is global, multiple `DllImport`s of it are fine. `LobbyPanel`'s own import is deleted in Task 6.

- [ ] **Step 2: Compile check**

coplay `check_compile_errors` → zero errors.

- [ ] **Step 3: Visual smoke via coplay**

Enter play mode. `execute_script`: build a canvas with one of each widget and screenshot it:

```csharp
var canvas = HexWars.Presentation.UiKit.Canvas("UiKitSmoke", 2000, null);
HexWars.Presentation.UiKit.EnsureEventSystem();
var p = HexWars.Presentation.UiKit.Panel(canvas.transform, "Card", HexWars.Presentation.UiKit.Surface);
HexWars.Presentation.UiKit.SetRect(p.GetComponent<UnityEngine.RectTransform>(), 0f, -100f, 600f, 400f);
HexWars.Presentation.UiKit.Label(p.transform, "UiKit smoke", 0f, -20f, 600f, 40f, HexWars.Presentation.UiKit.SizeTitle, UnityEngine.TextAnchor.MiddleCenter);
HexWars.Presentation.UiKit.Button(p.transform, "Primary", -150f, -90f, 200f, 44f, () => {}, HexWars.Presentation.UiKit.ButtonStyle.Primary);
HexWars.Presentation.UiKit.Button(p.transform, "CTA", 150f, -90f, 200f, 44f, () => {}, HexWars.Presentation.UiKit.ButtonStyle.Cta);
return "built";
```

Screenshot via `capture_ui_canvas`, confirm rounded corners + palette, then destroy `UiKitSmoke` and exit play mode.

- [ ] **Step 4: Commit**

```bash
git add Assets/HexWars/Presentation/UiKit.cs Assets/HexWars/Presentation/UiKit.cs.meta
git commit -m "feat(ui): UiKit - one canvas convention, palette, type scale, rounded 9-slice, button states; the style system every menu builds on"
```

---

### Task 5: Demo mode — `GameBootstrap.StartDemo()` + gameplay-UI suppression

**Files:**
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs`
- Modify: `Assets/HexWars/Presentation/SoundManager.cs` (mute flag)
- Modify: `Assets/HexWars/Presentation/GameHud.cs` (hide while demo)
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs` (hide while demo)
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs` (hide while demo — *amendment: surfaced in the first demo screenshot*)
- Modify: `Assets/HexWars/Presentation/PauseToggle.cs` (skip while demo — its speed controls key off SpectatorDriver existing, which the demo always has)
- Modify: `Assets/HexWars/Presentation/HelpOverlay.cs` (hide the "?" while demo — the title menu has How to Play)

**Interfaces:**
- Consumes: `GameFactory.Build`, `SpectatorDriver`, `EventConsole.Clear`.
- Produces (Task 8's TitleScreen relies on these):
  - `public bool DemoMode { get; private set; }` on GameBootstrap
  - `public void StartDemo()` — muted Greedy-vs-Random showcase game, UI suppressed
  - `public static bool Muted` on SoundManager
  - Real-game starters (`NewGame`, `StartLocalGame`, `StartNetGame`) end demo mode themselves.

- [ ] **Step 1: SoundManager mute**

In `SoundManager`, add below `_clips`:

```csharp
        /// <summary>True while the title demo plays — the menu should be calm, not a battle radio.</summary>
        public static bool Muted;
```

and make `Play` early-out:

```csharp
        public static void Play(SoundKind kind)
        {
            if (Muted) return;
            Ensure();
            _src.PlayOneShot(Clip(kind));
        }
```

- [ ] **Step 2: GameBootstrap — DemoMode, StartDemo, EndDemo**

Add near the `State` property:

```csharp
        /// <summary>True while the title-screen demo game (AI vs AI, muted, gameplay UI hidden) is
        /// running. Real-game starters clear it. HUD/panels early-out on it — see the spec's
        /// suppression contract.</summary>
        public bool DemoMode { get; private set; }
```

Add these methods after `StartLocalGame`:

```csharp
        /// <summary>The title screen's living background: a muted Greedy-vs-Random match on a fresh
        /// standard map, driven by SpectatorDriver through the normal presenter path (camera glides
        /// and all), with every gameplay UI surface suppressed via <see cref="DemoMode"/>.
        /// Greedy-vs-Greedy is deliberately avoided: mirror matches draw ~93% as standoffs.</summary>
        public void StartDemo()
        {
            Presenter?.ResetQueue();
            Networked = false;
            DemoMode = true;
            SoundManager.Muted = true;
            var setup = new GameSetup(GameMode.Annihilation, 11, 8, 0,
                                      UnityEngine.Random.Range(1, 99999), 5, 2, 2, 1, 3);
            State = GameFactory.Build(setup);
            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(State.Board);
            renderer.RenderEntities(State, FogViewer());
            FindAnyObjectByType<CameraRig>()?.Frame();
            EventConsole.Clear();
            if (GetComponent<SpectatorDriver>() == null) gameObject.AddComponent<SpectatorDriver>();
            StateChanged?.Invoke();
        }

        /// <summary>Leave demo mode before a real game starts: drop the spectator driver, restore
        /// sound, and give input back (the driver had set ReadOnly).</summary>
        void EndDemo()
        {
            if (!DemoMode) return;
            DemoMode = false;
            SoundManager.Muted = false;
            var driver = GetComponent<SpectatorDriver>();
            if (driver != null) Destroy(driver);
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = false;
            var barracks = FindAnyObjectByType<BarracksPanel>();
            if (barracks != null) barracks.ReadOnly = false;
        }
```

Call `EndDemo();` as the FIRST line of `NewGame()`, `StartLocalGame(...)`, and `StartNetGame(...)`.

Gate the two demo-noise sources in `TryApply` (the demo drives through it):

```csharp
            var result = GameEngine.Apply(State, cmd);
            if (!result.Success)
            {
                Debug.Log($"[HexWars] {cmd.GetType().Name} rejected: {result.Reason}");
                if (!DemoMode) Toast.Show(Friendly(result.Reason.ToString()));
                return false;
            }
            var prev = State;
            State = result.NewState;
            if (!DemoMode) EventConsole.Report(State, CombatLog.Diff(prev, State, FogViewer()));
            Presenter.Enqueue(prev, cmd, State, IsLocalCommand(cmd));
            StateChanged?.Invoke();
            return true;
```

(Only the two `if (!DemoMode)` guards are new; everything else in `TryApply` stays.)

- [ ] **Step 3: GameHud hides while demo runs**

In `GameHud`, `Build()` starts with `var canvasGo = new GameObject("HudCanvas");` — promote it to a field: declare `GameObject _canvasGo;` next to `_banner`, assign `_canvasGo = canvasGo;` right after creation. Then at the top of `Refresh()`:

```csharp
            if (_game == null || _game.State == null) return;
            if (_canvasGo != null && _canvasGo.activeSelf == _game.DemoMode)
                _canvasGo.SetActive(!_game.DemoMode);
            if (_game.DemoMode) return;
```

(Hiding the canvas also silences the turn-handover toasts and the game-over banner, both of which only fire from `Refresh`/`ShowGameOver`.)

- [ ] **Step 4: BarracksPanel hides while demo runs**

In `BarracksPanel`, find the canvas GameObject it builds in its `Build()`/`Start()` (the GameObject holding its Canvas component; if it's a local, promote to a field `GameObject _canvasGo;` exactly like GameHud). At the top of `Rebuild()` (after its `_game == null` guard at line ~111):

```csharp
            if (_game != null && _game.DemoMode)
            {
                if (_canvasGo != null) _canvasGo.SetActive(false);
                return;
            }
            if (_canvasGo != null && !_canvasGo.activeSelf) _canvasGo.SetActive(true);
```

- [ ] **Step 5: Verify in play mode (coplay)**

Enter play mode (editor boots the normal hotseat `NewGame` — HUD visible). Then:

1. `execute_script`: `UnityEngine.Object.FindAnyObjectByType<HexWars.Presentation.GameBootstrap>().StartDemo(); return "demo";`
2. Wait ~5 s. `execute_script` checks: DemoMode true; a `SpectatorDriver` exists; `GameObject.Find("HudCanvas")` inactive (positive control: it was active before); `SoundManager.Muted` true; `State.Round` value captured.
3. Wait ~10 s more; re-read `State.Round`/command count — it advanced (the demo is actually playing, animated).
4. Screenshot the game view: board animating, NO HUD bar, NO barracks panel, no sidebar.
5. `execute_script`: `...StartLocalGame(new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Annihilation, 9, 7, 0, 7), false); return "real";` then assert DemoMode false, HudCanvas active again, `SoundManager.Muted` false, SpectatorDriver gone.
6. Exit play mode.

- [ ] **Step 6: Commit**

```bash
git add Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/SoundManager.cs Assets/HexWars/Presentation/GameHud.cs Assets/HexWars/Presentation/BarracksPanel.cs
git commit -m "feat(title): demo mode - StartDemo() runs a muted Greedy-vs-Random showcase with gameplay UI suppressed; real-game starters end it"
```

---

### Task 6: `SetupForm` — the host / vs-AI settings form (replaces LobbyPanel)

**Files:**
- Create: `Assets/HexWars/Presentation/SetupForm.cs`
- Delete: `Assets/HexWars/Presentation/LobbyPanel.cs` (+ its `.meta`)
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (net/local start signatures, cancel, seat-full toast, temporary boot wiring)
- Modify: `Assets/HexWars/Presentation/NetClient.cs` (`isPrivate` connect param)

**Interfaces:**
- Consumes: UiKit (Task 4 exact API), `GameBootstrap.StartLocalGame/StartNetGame`, `AiLevel`.
- Produces (Task 8 relies on):
  - `SetupForm.Open(GameBootstrap game, SetupForm.SetupMode mode)` — static; `enum SetupMode { Host, VsAi }`
  - `public void Close()` on SetupForm
  - `GameBootstrap.StartNetGame(string room, string setupWire, bool isPrivate = false)`
  - `GameBootstrap.StartLocalGame(GameSetup setup, bool vsAi, AiLevel level = AiLevel.Hard)`
  - `GameBootstrap.CancelHosting()` — tears down the socket + seat, leaves State null
  - `NetClient.Connect(GameBootstrap game, string room, string setupWire, bool isPrivate)`

- [ ] **Step 1: GameBootstrap + NetClient plumbing**

`NetClient.Connect` — add the flag and forward it:

```csharp
        public async void Connect(GameBootstrap game, string room, string setupWire, bool isPrivate = false)
        {
            _game = game;
            string url = ServerWsUrl(room, setupWire, isPrivate);
            ...
```

`ServerWsUrl` — new signature + one appended query:

```csharp
        static string ServerWsUrl(string room, string setupWire, bool isPrivate)
        {
            ... (unchanged body) ...
            string url = origin + "/ws?room=" + Uri.EscapeDataString(room);
            if (!string.IsNullOrEmpty(setupWire)) url += "&setup=" + Uri.EscapeDataString(setupWire);
            if (isPrivate) url += "&private=1";
            return url;
        }
```

`GameBootstrap`:

```csharp
        /// <summary>Connect to the server for a room. <paramref name="setupWire"/> is non-null only for
        /// the host (carries the lobby picks); a joiner passes null and gets the host's game.
        /// <paramref name="isPrivate"/> keeps the room out of the public browser list.</summary>
        public void StartNetGame(string room, string setupWire, bool isPrivate = false)
        {
            EndDemo();
            // the demo's state must not linger: panels dismiss on (State != null && !DemoMode), and
            // the authoritative state arrives later via START — until then there is no game here
            State = null;
            StateChanged?.Invoke();
            Networked = true;
            if (_net != null) { Destroy(_net); _net = null; }
            _net = gameObject.AddComponent<NetClient>();
            _net.Connect(this, room, setupWire, isPrivate);
        }

        /// <summary>Host changed their mind while waiting: drop the socket and seat. State stays as it
        /// was (null before START), so the title/demo behind the form is untouched.</summary>
        public void CancelHosting()
        {
            if (_net != null) { Destroy(_net); _net = null; }
            Seat = null;
        }
```

(`Networked = true;` — today only the WebGL define sets it; making the net starter own it lets the editor drive online flows too. The editor's default boot still sets nothing, so hotseat is unaffected.)

`StartLocalGame` — difficulty:

```csharp
        public void StartLocalGame(GameSetup setup, bool vsAi, AiLevel level = AiLevel.Hard)
        {
            EndDemo();
            Presenter?.ResetQueue();
            Networked = false;
            State = GameFactory.Build(setup);
            var renderer = GetComponent<BoardRenderer>();
            renderer.Render(State.Board);
            renderer.RenderEntities(State, FogViewer());
            FindAnyObjectByType<CameraRig>()?.Frame();
            EventConsole.Clear();
            EventConsole.Report(State, null);
            StateChanged?.Invoke();
            if (vsAi)
            {
                var ai = gameObject.AddComponent<AiOpponent>();
                ai.Level = level;
            }
        }
```

`OnNetSeatFull` — surface it instead of silently spectating:

```csharp
        internal void OnNetSeatFull()
        {
            Toast.Show("That game is already full.");
            CancelHosting();
        }
```

Connection-loss surfacing while waiting (spec §7): add to `GameBootstrap`:

```csharp
        /// <summary>The socket died before a match began (server down / network drop while hosting or
        /// joining). Mid-game drops are the reconnect follow-up (audit U2) — pre-game, a toast plus the
        /// waiting screen's Cancel is the whole story.</summary>
        internal void OnNetClosed()
        {
            if (Networked && State == null && _net != null)
                Toast.Show("Connection lost — check the link and try again.");
        }
```

and in `NetClient`, call it from the close handler — replace the `OnClose` lambda in `Connect` with:

```csharp
            _ws.OnClose += c => { Connected = false; Debug.Log("[Net] closed: " + c); if (!_closing) _game.OnNetClosed(); };
```

adding the field + flag set in `OnDestroy`:

```csharp
        bool _closing;

        async void OnDestroy()
        {
            _closing = true;             // deliberate teardown (Cancel / ReturnToMenu) — not an error
            if (_ws != null) await _ws.Close();
        }
```

Boot wiring (temporary until Task 8 adds TitleScreen): in `Start()`, replace `else gameObject.AddComponent<LobbyPanel>();` with `else { StartDemo(); SetupForm.Open(this, SetupForm.SetupMode.Host); }` and in `ReturnToMenu()` replace the LobbyPanel re-add block with `if (GetComponent<SetupForm>() == null) SetupForm.Open(this, SetupForm.SetupMode.Host);` (Task 8 rewires both to TitleScreen).

- [ ] **Step 2: Write SetupForm**

Create `Assets/HexWars/Presentation/SetupForm.cs` — the LobbyPanel form rebuilt on UiKit with modes. Full file:

```csharp
using System;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The game-settings form, opened from the title screen in one of two modes: <b>Host</b> (online —
    /// adds a Private toggle; Create connects and shows a waiting screen with the room code, share link
    /// and Cancel) and <b>VsAi</b> (adds a Difficulty row; Create starts the local game immediately).
    /// Numeric fields are tap-to-type value boxes (browser prompt — the only reliable mobile-WebGL
    /// keyboard) flanked by −/+ steppers. Removes itself when a real match starts.
    /// </summary>
    public sealed class SetupForm : MonoBehaviour
    {
        public enum SetupMode { Host, VsAi }

        GameBootstrap _game;
        SetupMode _mode;
        GameObject _canvasGo;
        GameObject _form;
        GameObject _armyPanel;
        Text _armyLabel;
        Text _status;
        GameObject _cancelBtn;

        GameMode _gameMode = GameMode.Annihilation;
        int _w = 9, _h = 7, _pts = 0, _seed = 7;
        int _armySize = 3, _brutes = 1, _strikers = 1, _snipers = 1;
        int _turnActions = 3;
        bool _fog = false;
        bool _private = false;
        AiLevel _ai = AiLevel.Hard;

        static readonly int[] PacePresets = { 3, 4, 0, 1 };
        static string PaceLabel(int k) => k <= 0 ? "whole army (fast)" : $"{k} action{(k == 1 ? "" : "s")}/turn";

        readonly System.Collections.Generic.List<(Button btn, Func<bool> selected)> _toggles
            = new System.Collections.Generic.List<(Button, Func<bool>)>();

        public static SetupForm Open(GameBootstrap game, SetupMode mode)
        {
            var existing = game.GetComponent<SetupForm>();
            if (existing != null) existing.Close();
            var form = game.gameObject.AddComponent<SetupForm>();
            form._game = game;
            form._mode = mode;
            return form; // Build runs in Start so _game/_mode are set first
        }

        void Start()
        {
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            _seed = UnityEngine.Random.Range(1, 9999);
            Build();
            RefreshToggles();
        }

        void Update()
        {
            // a real match started (host's START arrived, or the vs-AI game began) — this form is done
            if (_game != null && _game.State != null && !_game.DemoMode) Close();
        }

        public void Close()
        {
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

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

            float y = -24f;
            string title = _mode == SetupMode.Host ? "Host Online Game" : "Play vs AI";
            UiKit.Label(_form.transform, title, 0f, y, 700f, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(_form.transform, "Back", -300f, y - 2f, 90f, 34f, () => { Close(); TitleScreen.Reopen(_game); },
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            y -= 44f;
            UiKit.Label(_form.transform, "tap a value to type it, or use − / +", 0f, y, 700f, 22f,
                        UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint); y -= 40f;

            ToggleBtn("Annihilation", -95f, y, 180f, 38f, () => _gameMode == GameMode.Annihilation,
                      () => { _gameMode = GameMode.Annihilation; RefreshToggles(); });
            ToggleBtn("Territory", 100f, y, 160f, 38f, () => _gameMode == GameMode.Territory,
                      () => { _gameMode = GameMode.Territory; RefreshToggles(); });
            y -= 48f;

            NumberRow("Map width", y, () => _w, v => _w = v, 5, 24, 1); y -= 46f;
            NumberRow("Map height", y, () => _h, v => _h = v, 5, 24, 1); y -= 46f;
            NumberRow("Start points", y, () => _pts, v => _pts = v, 0, 200, 10); y -= 46f;

            UiKit.Label(_form.transform, "Seed", -245f, y, 210f, 38f, UiKit.SizeBody + 2, TextAnchor.MiddleLeft, UiKit.TextDim);
            var seedDisp = UiKit.ValueBox(_form.transform, "Seed", 60f, y, 130f, 38f, () => _seed, v => _seed = v, 1, 99999);
            UiKit.Button(_form.transform, "Reroll", 190f, y, 100f, 38f,
                         () => { _seed = UnityEngine.Random.Range(1, 9999); seedDisp.text = _seed.ToString(); },
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            y -= 48f;

            var armyBtn = UiKit.Button(_form.transform, "", 0f, y, 500f, 40f, OpenArmy, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            _armyLabel = armyBtn.GetComponentInChildren<Text>();
            _armyLabel.text = ArmySummary();
            y -= 48f;

            Text paceLabel = null;
            var paceBtn = UiKit.Button(_form.transform, "", 0f, y, 500f, 40f, () =>
            {
                int idx = Array.IndexOf(PacePresets, _turnActions);
                _turnActions = PacePresets[(idx + 1) % PacePresets.Length];
                if (paceLabel != null) paceLabel.text = "Pace:  " + PaceLabel(_turnActions) + "   ▸";
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            paceLabel = paceBtn.GetComponentInChildren<Text>();
            paceLabel.text = "Pace:  " + PaceLabel(_turnActions) + "   ▸";
            y -= 48f;

            if (_mode == SetupMode.Host)
            {
                ToggleBtn("Fog of war", -140f, y, 220f, 38f, () => _fog, () => { _fog = !_fog; RefreshToggles(); });
                ToggleBtn("Private (invite only)", 120f, y, 250f, 38f, () => _private, () => { _private = !_private; RefreshToggles(); });
            }
            else
            {
                ToggleBtn("Fog of war", -140f, y, 220f, 38f, () => _fog, () => { _fog = !_fog; RefreshToggles(); });
                ToggleBtn("AI: Hard", 120f, y, 250f, 38f, () => _ai == AiLevel.Hard, () =>
                {
                    _ai = _ai == AiLevel.Hard ? AiLevel.Easy : AiLevel.Hard;
                    RefreshToggles();
                    foreach (var (btn, sel) in _toggles)
                    {
                        var t = btn.GetComponentInChildren<Text>();
                        if (t != null && t.text.StartsWith("AI: ")) t.text = _ai == AiLevel.Hard ? "AI: Hard" : "AI: Easy";
                    }
                });
            }
            y -= 54f;

            string cta = _mode == SetupMode.Host ? "Create Game" : "Start Game";
            UiKit.Button(_form.transform, cta, 0f, y, 340f, 50f, OnCreate, UiKit.ButtonStyle.Cta);

            _status = UiKit.Label(_canvasGo.transform, "", 0f, 0f, 1100f, 160f, UiKit.SizeHeading, TextAnchor.MiddleCenter);
            var srt = _status.GetComponent<RectTransform>();
            srt.anchorMin = srt.anchorMax = new Vector2(0.5f, 0.5f);
            srt.pivot = new Vector2(0.5f, 0.5f);
            srt.anchoredPosition = new Vector2(0f, 30f);

            BuildArmyPopup();
        }

        string ArmySummary()
        {
            int spec = _brutes + _strikers + _snipers;
            if (spec <= 0) return $"Army:  {_armySize} random   ▸";
            string roles = $"{_brutes} Brute, {_strikers} Striker, {_snipers} Sniper";
            if (spec < _armySize) roles += " + random";
            return $"Army:  {roles}   ▸";
        }

        void BuildArmyPopup()
        {
            _armyPanel = new GameObject("ArmyPopup");
            _armyPanel.transform.SetParent(_canvasGo.transform, false);
            var prt = _armyPanel.AddComponent<RectTransform>();
            UiKit.Stretch(prt);

            var dim = UiKit.Panel(_armyPanel.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.75f));
            UiKit.Stretch(dim.GetComponent<RectTransform>());
            dim.sprite = null; // full-bleed dim, no rounding
            dim.gameObject.AddComponent<Button>().onClick.AddListener(CloseArmy);

            var card = UiKit.Panel(_armyPanel.transform, "Card", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(700f, 430f);
            crt.anchoredPosition = Vector2.zero;

            float y = -24f;
            UiKit.Label(card.transform, "Starting army", 0f, y, 700f, 34f, UiKit.SizeTitle - 3, TextAnchor.MiddleCenter); y -= 48f;
            NumberRowIn(card.transform, "Army size", y, () => _armySize, v => _armySize = v, 1, 12); y -= 46f;
            NumberRowIn(card.transform, "Brutes", y, () => _brutes, v => _brutes = v, 0, 12); y -= 46f;
            NumberRowIn(card.transform, "Strikers", y, () => _strikers, v => _strikers = v, 0, 12); y -= 46f;
            NumberRowIn(card.transform, "Snipers", y, () => _snipers, v => _snipers = v, 0, 12); y -= 44f;
            UiKit.Label(card.transform, "Leave roles at 0 for a random army; extra slots fill randomly.",
                        0f, y, 700f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint); y -= 40f;
            UiKit.Button(card.transform, "Done", 0f, y, 220f, 44f, CloseArmy, UiKit.ButtonStyle.Cta, UiKit.SizeHeading);

            _armyPanel.SetActive(false);
        }

        void OpenArmy() { if (_armyPanel != null) _armyPanel.SetActive(true); }

        void CloseArmy()
        {
            if (_armyPanel != null) _armyPanel.SetActive(false);
            if (_armyLabel != null) _armyLabel.text = ArmySummary();
        }

        void NumberRow(string label, float y, Func<int> get, Action<int> set, int min, int max, int step)
            => NumberRowOn(_form.transform, label, y, get, set, min, max, step);

        void NumberRowIn(Transform parent, string label, float y, Func<int> get, Action<int> set, int min, int max)
            => NumberRowOn(parent, label, y, get, set, min, max, 1);

        void NumberRowOn(Transform parent, string label, float y, Func<int> get, Action<int> set, int min, int max, int step)
        {
            UiKit.Label(parent, label, -245f, y, 210f, 38f, UiKit.SizeBody + 2, TextAnchor.MiddleLeft, UiKit.TextDim);
            Text disp = null;
            UiKit.Button(parent, "−", -10f, y, 54f, 38f,
                         () => { set(Mathf.Clamp(get() - step, min, max)); if (disp != null) disp.text = get().ToString(); },
                         UiKit.ButtonStyle.Secondary, 24);
            disp = UiKit.ValueBox(parent, label, 80f, y, 90f, 38f, get, set, min, max);
            UiKit.Button(parent, "+", 170f, y, 54f, 38f,
                         () => { set(Mathf.Clamp(get() + step, min, max)); if (disp != null) disp.text = get().ToString(); },
                         UiKit.ButtonStyle.Secondary, 24);
        }

        void ToggleBtn(string text, float x, float y, float w, float h, Func<bool> selected, Action onClick)
        {
            var b = UiKit.Button(_form.transform, text, x, y, w, h, onClick, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            _toggles.Add((b, selected));
        }

        void RefreshToggles()
        {
            foreach (var (btn, selected) in _toggles) UiKit.SetToggled(btn, selected());
        }

        void OnCreate()
        {
            var setup = new GameSetup(_gameMode, _w, _h, _pts, _seed,
                                      _armySize, _brutes, _strikers, _snipers, _turnActions, _fog);
            if (_mode == SetupMode.VsAi)
            {
                _game.StartLocalGame(setup, true, _ai);   // form dismisses via Update when State exists
                return;
            }

            string room = RandomCode();
            _game.StartNetGame(room, setup.ToWire(), _private);
            ShowWaiting(room);
        }

        void ShowWaiting(string room)
        {
            if (_form != null) _form.SetActive(false);
            if (_armyPanel != null) _armyPanel.SetActive(false);
            _status.text = $"Room code\n<size=64><b>{room}</b></size>\n\nWaiting for an opponent…\n" +
                           (_private ? "Private game — share the code or link below.\n" : "Your game is listed in Browse Games.\n") +
                           ShareUrl(room);
            _status.supportRichText = true;

            _cancelBtn = UiKit.Button(_canvasGo.transform, "Cancel", 0f, 0f, 200f, 44f, () =>
            {
                _game.CancelHosting();
                Close();
                TitleScreen.Reopen(_game);
            }, UiKit.ButtonStyle.Danger, UiKit.SizeBody + 2).gameObject;
            var crt = _cancelBtn.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.anchoredPosition = new Vector2(0f, -150f);
        }

        internal static string RandomCode()
        {
            const string alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
            var c = new char[5];
            for (int i = 0; i < c.Length; i++) c[i] = alphabet[UnityEngine.Random.Range(0, alphabet.Length)];
            return new string(c);
        }

        internal static string ShareUrl(string room)
        {
            string page = Application.absoluteURL;
            if (string.IsNullOrEmpty(page)) return "(this page) ?room=" + room;
            int q = page.IndexOf('?');
            if (q >= 0) page = page.Substring(0, q);
            return page + "?room=" + room;
        }
    }
}
```

Note the two `TitleScreen.Reopen(_game)` calls — TitleScreen doesn't exist until Task 8. **For this task**, add a minimal placeholder so the project compiles (Task 8 replaces it with the real class):

Create `Assets/HexWars/Presentation/TitleScreen.cs`:

```csharp
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Placeholder until the title-screen task lands: Reopen falls back to the Host form.</summary>
    public sealed class TitleScreen : MonoBehaviour
    {
        public static void Reopen(GameBootstrap game)
        {
            if (game.GetComponent<SetupForm>() == null) SetupForm.Open(game, SetupForm.SetupMode.Host);
        }
    }
}
```

Delete `Assets/HexWars/Presentation/LobbyPanel.cs` and `LobbyPanel.cs.meta`.

- [ ] **Step 3: Compile + play-mode verify (coplay)**

`check_compile_errors` → zero. Enter play mode, then:
1. `execute_script`: `var g = UnityEngine.Object.FindAnyObjectByType<HexWars.Presentation.GameBootstrap>(); g.StartDemo(); HexWars.Presentation.SetupForm.Open(g, HexWars.Presentation.SetupForm.SetupMode.VsAi); return "open";`
2. Screenshot: form over the demo board; Difficulty toggle visible, no Private toggle.
3. `execute_script` clicks Create via `onClick.Invoke()` on the CTA button (find `SetupCanvas` → walk to the "Start Game" button) → assert a real game started: `g.State != null && !g.DemoMode`, `AiOpponent` present with `Level` matching the toggle, form canvas gone.
4. Back to demo: `g.ReturnToMenu()` — expect (temporary) Host form; screenshot it: Private toggle visible, Difficulty absent.
5. Host flow (editor, local server): start `dotnet run --project engine/HexWars.NetServer` in the background. In the form, invoke Create → waiting screen shows a 5-char code + Cancel. Screenshot. Invoke Cancel → status gone, Host form returns (placeholder Reopen), `g.Seat == null`, NetClient destroyed. Stop the server. Exit play mode.

- [ ] **Step 4: Commit**

```bash
git add -A Assets/HexWars/Presentation
git commit -m "feat(lobby): SetupForm replaces LobbyPanel - UiKit styling, Host/VsAi modes, Private toggle, AI difficulty, waiting screen with working Cancel"
```

---

### Task 7: `GameBrowser` — the open-games list

**Files:**
- Create: `Assets/HexWars/Presentation/GameBrowser.cs`

**Interfaces:**
- Consumes: UiKit; `GET /games` JSON (Task 3 exact shape); `GameBootstrap.StartNetGame(code, null)`; `TitleScreen.Reopen`.
- Produces (Task 8 relies on): `GameBrowser.Open(GameBootstrap game)` static; `public void Close()`.

- [ ] **Step 1: Write the file**

Create `Assets/HexWars/Presentation/GameBrowser.cs`:

```csharp
using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The lobby browser: polls the server's <c>GET /games</c> every 3 s while open and lists every
    /// public game waiting for an opponent — code + config summary per row, tap a row to expand the
    /// full configuration with a Join button. You never end up in a game you didn't want. Joining is
    /// the ordinary by-code path; if the seat filled in the race window the server's SEAT FULL toast
    /// fires and the list refreshes on the next poll.
    /// </summary>
    public sealed class GameBrowser : MonoBehaviour
    {
        const float PollSeconds = 3f;

        GameBootstrap _game;
        GameObject _canvasGo;
        Transform _listRoot;
        Text _status;
        string _expandedCode;         // row currently expanded to the full-config card
        GameDto[] _lastGames = Array.Empty<GameDto>();
        bool _fetchFailed;

        [Serializable] class GamesDto { public GameDto[] games; }
        [Serializable] public class GameDto
        {
            public string code; public string mode;
            public int width, height, pace, army, ageSeconds;
            public bool fog;
        }

        public static GameBrowser Open(GameBootstrap game)
        {
            var existing = game.GetComponent<GameBrowser>();
            if (existing != null) existing.Close();
            var b = game.gameObject.AddComponent<GameBrowser>();
            b._game = game;
            return b;
        }

        void Start()
        {
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            StartCoroutine(PollLoop());
        }

        void Update()
        {
            if (_game != null && _game.State != null && !_game.DemoMode) Close(); // a match started
        }

        public void Close()
        {
            StopAllCoroutines();
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

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
                         () => StartCoroutine(FetchOnce()), UiKit.ButtonStyle.Secondary, UiKit.SizeBody);

            _status = UiKit.Label(panel.transform, "Loading…", 0f, -70f, 700f, 26f,
                                  UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            var lrt = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(lrt, 0f, -100f, 720f, 520f);
            _listRoot = listGo.transform;
        }

        IEnumerator PollLoop()
        {
            while (true)
            {
                yield return FetchOnce();
                yield return new WaitForSeconds(PollSeconds);
            }
        }

        IEnumerator FetchOnce()
        {
            using (var req = UnityWebRequest.Get(GamesUrl()))
            {
                req.timeout = 4;
                yield return req.SendWebRequest();
                if (req.result != UnityWebRequest.Result.Success)
                {
                    _fetchFailed = true;
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                GamesDto dto = null;
                try { dto = JsonUtility.FromJson<GamesDto>(req.downloadHandler.text); }
                catch (Exception) { /* malformed = treat as fetch failure */ }
                if (dto == null || dto.games == null)
                {
                    _fetchFailed = true;
                    if (_status != null) _status.text = "Can't reach the server — retrying…";
                    yield break;
                }
                _fetchFailed = false;
                _lastGames = dto.games;
                Rebuild();
            }
        }

        void Rebuild()
        {
            if (_listRoot == null) return;
            for (int i = _listRoot.childCount - 1; i >= 0; i--) Destroy(_listRoot.GetChild(i).gameObject);

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

            _status.text = $"{_lastGames.Length} open game{(_lastGames.Length == 1 ? "" : "s")} — tap one for details";
            float y = -4f;
            foreach (var g in _lastGames)
            {
                bool expanded = g.code == _expandedCode;
                y = BuildRow(g, y, expanded);
            }
        }

        float BuildRow(GameDto g, float y, bool expanded)
        {
            string age = g.ageSeconds < 60 ? $"{g.ageSeconds}s" : $"{g.ageSeconds / 60}m";
            string summary = $"{g.code}   ·   {g.mode} · {g.width}×{g.height}{(g.fog ? " · Fog" : "")}" +
                             $" · {(g.pace <= 0 ? "whole army" : g.pace + " acts/turn")} · {g.army} units · {age} ago";
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
                            $"Pace {(g.pace <= 0 ? "whole army" : g.pace + " actions/turn")}    Army {g.army} units    Waiting {age}",
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

        static string GamesUrl()
        {
            string page = Application.absoluteURL;
            if (!string.IsNullOrEmpty(page))
            {
                try { var uri = new Uri(page); return uri.Scheme + "://" + uri.Authority + "/games"; }
                catch { }
            }
            return "http://127.0.0.1:5234/games"; // editor dev default — matches NetClient's fallback
        }
    }
}
```

- [ ] **Step 2: Compile + verify against the live local server (coplay)**

`check_compile_errors` → zero. Start the local server in the background: `dotnet run --project engine/HexWars.NetServer` (it binds 5234 per NetClient's dev default — if the launch profile differs, set `PORT=5234`). Then in play mode:

1. Empty state: `execute_script` — `var g = FindAnyObjectByType<GameBootstrap>(); g.StartDemo(); GameBrowser.Open(g); return "open";` (full namespaces as in Task 6). Wait 4 s. Screenshot: "No open games right now — host one!" + Host button.
2. Seed an open game: from the shell, run a tiny websocket host? No — simplest true-to-life seeding: `execute_script` a second connection is not possible in-editor. Instead use the server's own selftest client shape: run `dotnet run --project engine/HexWars.NetServer -- selftest` — NO (it runs its own server). Seed instead with curl-equivalent: a WebSocket client one-liner isn't shell-native, so seed via a **second editor path**: `execute_script`: `g.StartNetGame(HexWars.Presentation.SetupForm.RandomCode(), new HexWars.Engine.GameSetup(HexWars.Engine.GameMode.Territory, 13, 9, 40, 7, 5, 2, 2, 1, 3, true).ToWire(), false); return "hosting";` — the editor itself is now the waiting host. Then `CancelHosting` cannot run (that would unlist) — instead poll `/games` from the shell: `curl http://127.0.0.1:5234/games` → the hosted room appears with the exact config JSON.
3. The browser CAN'T show the editor's own game as joinable-by-self (one editor = one client) — so for the list-rendering check, screenshot the browser AFTER the curl-confirmed row appears in `_lastGames` (execute_script: `FindAnyObjectByType<GameBrowser>()` reflection-read `_lastGames.Length == 1`) and screenshot the row + expanded detail card (invoke the row button's onClick, screenshot again).
4. Join race handling is covered by OnNetSeatFull (Task 6) — not double-tested here.
5. `g.CancelHosting();` then next poll → list empties back to the empty state (assert via `_lastGames.Length == 0`). Stop the server; wait one poll → status shows "Can't reach the server — retrying…". Screenshot. Exit play mode.

- [ ] **Step 3: Commit**

```bash
git add Assets/HexWars/Presentation/GameBrowser.cs Assets/HexWars/Presentation/GameBrowser.cs.meta
git commit -m "feat(lobby): GameBrowser - polls /games, config-summary rows with expandable detail + Join, empty and error states"
```

---

### Task 8: `TitleScreen` — the main menu over the demo, boot wiring

**Files:**
- Modify (replace placeholder): `Assets/HexWars/Presentation/TitleScreen.cs`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs` (boot + ReturnToMenu wiring)

**Interfaces:**
- Consumes: UiKit, `GameBootstrap.StartDemo/DemoMode/StartNetGame`, `SetupForm.Open`, `GameBrowser.Open`, `GameRules.Show`, `UiKit.PromptText`.
- Produces: `TitleScreen.Reopen(GameBootstrap game)` (already referenced by Tasks 6–7 — same signature, now real: re-adds the TitleScreen component if missing); `GameBootstrap` boots WebGL to demo + title; `ReturnToMenu` → demo + title.

- [ ] **Step 1: Replace the placeholder with the real TitleScreen**

Full file `Assets/HexWars/Presentation/TitleScreen.cs`:

```csharp
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The front door: HEXWARS wordmark + the five main actions, drawn over the live demo game
    /// (StartDemo's muted AI-vs-AI match). Owns the demo lifecycle — when the demo game ends it
    /// waits a beat and starts a fresh one. Opening a sub-screen (Browse / Host / vs AI / Rules)
    /// hides this menu and keeps the demo running behind it; the sub-screens call
    /// <see cref="Reopen"/> to come back. Destroys itself when a real match starts.
    /// </summary>
    public sealed class TitleScreen : MonoBehaviour
    {
        const float DemoRestartDelay = 3f;

        GameBootstrap _game;
        GameObject _canvasGo;
        float _overSince = -1f;

        public static void Reopen(GameBootstrap game)
        {
            if (game.GetComponent<TitleScreen>() == null) game.gameObject.AddComponent<TitleScreen>();
        }

        void Start()
        {
            _game = GetComponent<GameBootstrap>();
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            Build();
        }

        void Update()
        {
            if (_game == null) return;

            // a real match started — the title is done (sub-screens dismiss themselves the same way)
            if (_game.State != null && !_game.DemoMode) { Close(); return; }

            // back on the title with no demo running (cancelled hosting, seat-full bounce, dropped
            // socket) — self-heal: the title always has a living background
            if (!_game.DemoMode && _game.State == null) { _game.StartDemo(); return; }

            // demo ended: hold the final board a beat, then roll a fresh demo
            if (_game.DemoMode && _game.State != null && _game.State.IsGameOver)
            {
                if (_overSince < 0f) _overSince = Time.unscaledTime;
                else if (Time.unscaledTime - _overSince >= DemoRestartDelay) { _overSince = -1f; _game.StartDemo(); }
            }
            else _overSince = -1f;
        }

        void Close()
        {
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void Hide()  // a sub-screen takes over; demo keeps playing behind it
        {
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("TitleCanvas", UiKit.OrderMenu, transform);

            // left-anchored column: the menu reads over the demo without hiding the action
            var col = new GameObject("Menu");
            col.transform.SetParent(_canvasGo.transform, false);
            var crt = col.AddComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(520f, 640f);
            crt.anchoredPosition = new Vector2(0f, 20f);

            var plate = UiKit.Panel(col.transform, "Plate", new Color(UiKit.Bg.r, UiKit.Bg.g, UiKit.Bg.b, 0.82f));
            UiKit.Stretch(plate.GetComponent<RectTransform>());
            plate.raycastTarget = false;

            var word = UiKit.Label(col.transform, "HEXWARS", 0f, -34f, 520f, 70f, 58, TextAnchor.MiddleCenter, UiKit.Accent);
            word.fontStyle = FontStyle.Bold;
            UiKit.Label(col.transform, "hex-grid tactics — design an army, take the field",
                        0f, -104f, 520f, 24f, UiKit.SizeBody, TextAnchor.MiddleCenter, UiKit.TextDim);

            float y = -170f;
            const float bw = 380f, bh = 52f, gap = 62f;
            UiKit.Button(col.transform, "Browse Games", 0f, y, bw, bh, () =>
            { Hide(); GameBrowser.Open(_game); }, UiKit.ButtonStyle.Cta); y -= gap;

            UiKit.Button(col.transform, "Host Game", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Host); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Join by Code", 0f, y, bw, bh, OnJoinByCode, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Play vs AI", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.VsAi); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "How to Play", 0f, y, bw, bh, () =>
            { GameRules.Show(_canvasGo.transform, UiKit.Font(), 1100); }, UiKit.ButtonStyle.Secondary); y -= gap;

            UiKit.Label(col.transform, "v" + Application.version + "   ·   two players, two browsers — share a room code",
                        0f, y - 6f, 520f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);
        }

        void OnJoinByCode()
        {
            string code = UiKit.PromptText("Enter the room code (e.g. KQ7KP)", "");
            if (code == null)
            {
                Toast.Show("Type-in needs the browser build — in the editor, use Browse Games.");
                return;
            }
            code = code.Trim().ToUpperInvariant();
            if (code.Length == 0) return; // cancelled
            Hide();
            _game.StartNetGame(code, null);   // SEAT/START arrive via the normal net path;
                                              // a full/unknown room toasts via OnNetSeatFull
        }
    }
}
```

- [ ] **Step 2: GameBootstrap boot + return wiring**

In `Start()` (Networked branch), replace the Task 6 temporary line with:

```csharp
                if (!string.IsNullOrEmpty(room)) StartNetGame(room, null); // opened via a shared ?room= link → join it
                else { StartDemo(); gameObject.AddComponent<TitleScreen>(); } // front door: demo + title menu
```

In `ReturnToMenu()`, replace the SetupForm re-add with:

```csharp
            GetComponent<SetupForm>()?.Close();
            GetComponent<GameBrowser>()?.Close();
            StartDemo();
            TitleScreen.Reopen(this);
```

(Note `State = null` currently precedes this in ReturnToMenu — `StartDemo` immediately builds the demo state, which is what the title wants; the `State = null; StateChanged` pair can stay for the HUD reset, with StartDemo following it.)

In `OnNetSeatFull`, after `CancelHosting();` add `TitleScreen.Reopen(this);` — a stale `?room=` link boot lands on the title with the toast showing.

- [ ] **Step 3: Play-mode verification (coplay)**

Enter play mode. Then:

1. `execute_script`: `var g = FindAnyObjectByType<GameBootstrap>(); g.StartDemo(); TitleScreen.Reopen(g); return "title";`
2. Screenshot: wordmark + five buttons over a live board; no HUD.
3. Demo-restart loop: `execute_script` force-ends the demo — apply EndTurns? Too slow. Instead reflection-set: build a finished state is complex — pragmatic check: `execute_script` sets the private `_overSince` via reflection to `Time.unscaledTime - 10f` AND temporarily forces game-over by… **skip force**; instead verify the restart logic with a fast demo: reflection-set `SpectatorDriver.SecondsPerAction = 0.01f` and `ActionPresenter` fast-forwards handle pacing; wait until `g.State.IsGameOver` (poll every 10 s, typical Greedy-vs-Random game ends in well under 2 min at 0.01s/action), then wait 4 s more and assert `g.State.IsGameOver == false` (a fresh demo started) and `g.DemoMode == true`.
4. Buttons: invoke each button's onClick via script (find `TitleCanvas/Menu` children in order): Browse → BrowserCanvas exists, TitleCanvas gone → `GameBrowser` Back → TitleCanvas back. Host → SetupForm (Private visible) → Back → title. Play vs AI → SetupForm (AI toggle visible) → Back → title. How to Play → rules popup appears over the title; screenshot; dismiss.
5. Join by Code in editor → toast about needing the browser build (PromptText returns null in editor) — assert toast object exists (positive control: `Toast.Show("x")` works).
6. Real-game dismissal: from the vs-AI form, invoke Start Game → title/panels all gone, HUD visible, `DemoMode` false.
7. Screenshot each screen along the way (title, browser, host form, vs-ai form, rules-over-title).
8. Exit play mode.

- [ ] **Step 4: Commit**

```bash
git add Assets/HexWars/Presentation/TitleScreen.cs Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(title): TitleScreen - main menu over the live demo, join-by-code, demo restart loop; boot + ReturnToMenu land on the title"
```

---

### Task 9: Restyle the remaining panels onto UiKit

Consistent skin, zero behavior change. Each panel keeps its logic; only construction code moves to UiKit calls (canvas convention, palette, rounded sprite, button states, font).

**Files (all Modify):**
- `Assets/HexWars/Presentation/GameHud.cs` — `Build()`: canvas → `UiKit.Canvas("HudCanvas", UiKit.OrderHud, transform)`; banner Image color → `new Color(UiKit.Bg.r, UiKit.Bg.g, UiKit.Bg.b, 0.88f)`; End Turn button → keep the raw Image+Button (it tints via `_endBtn.color` state logic) but set `sprite = UiKit.Rounded(); type = Sliced`; both `BuiltinFont()` bodies → `UiKit.Font()` (delete the local helper); `EnsureEventSystem` → `UiKit.EnsureEventSystem()` (delete the local copy and the now-unused `using UnityEngine.EventSystems; using UnityEngine.InputSystem.UI;`).
- `Assets/HexWars/Presentation/BarracksPanel.cs` — canvas preamble → `UiKit.Canvas("BarracksCanvas", UiKit.OrderPanels, transform)`; its local `Panel/Label/Button/BuiltinFont` helpers → UiKit equivalents (`UiKit.Panel` for the backing card, `UiKit.Label`, `UiKit.Button` with `ButtonStyle.Secondary` for rows — selected row via `UiKit.SetToggled(row, true)` instead of the gold Image color); delete the local helpers.
- `Assets/HexWars/Presentation/DesignPanel.cs` — same treatment (`UiKit.OrderPanels`; Create button → `ButtonStyle.Cta`).
- `Assets/HexWars/Presentation/UnitTooltip.cs` — canvas → `UiKit.Canvas("TooltipCanvas", UiKit.OrderTooltip, transform)`; background Image → `UiKit.Panel` surface at 0.95 alpha; font → `UiKit.Font()`.
- `Assets/HexWars/Presentation/Toast.cs` — canvas → `UiKit.OrderToast` via `UiKit.Canvas`; toast background gets `UiKit.Rounded()` sliced; font → `UiKit.Font()`. Keep its self-replacing/timing logic byte-for-byte.
- `Assets/HexWars/Presentation/GameOverBanner.cs` — canvas → `UiKit.Canvas(RootName, UiKit.OrderBanner, null)`; Main-menu button → `UiKit.Button(..., ButtonStyle.Secondary)`; fonts → `UiKit.Font()`.
- `Assets/HexWars/Presentation/GameRules.cs` — canvas/scaler lines → `UiKit.Canvas`; body font/colors → UiKit constants; close button → `UiKit.Button`. (Its sorting order is caller-passed today — `GameRules.Show(parent, font, order)`; keep the signature, callers already pass 1100/`OrderRules`-ish values.)
- `Assets/HexWars/Presentation/HelpOverlay.cs` — the "?" button → `UiKit.Button` secondary; DELETE the dead `Toggle()`/`BuildPanel()`/`HelpText` block (~65 lines, audit F13 — it documents removed mechanics).

**Interfaces:** consumes UiKit only; produces nothing new.

- [ ] **Step 1: Apply the per-file changes above.** Work file by file; after each file, coplay `check_compile_errors`.

- [ ] **Step 2: Play-mode regression sweep (coplay)**

Enter play mode (hotseat `NewGame` boots). Verify with screenshots:
1. HUD bar: banner text renders, End Turn clickable and turns green when nothing left to do (drive a quick EndTurn loop).
2. Barracks: create a design via DesignPanel (drive `OnCreate` with points seeded), rows render on UiKit style, selected row highlights, deploy flow works (select row → click a zone hex → unit appears).
3. Tooltip: hover a unit (or call `Show` directly) — rounded surface, docked position unchanged.
4. Toast: `Toast.Show("styled")` — rounded, positioned as before.
5. Game over: force a win (`execute_script` applies attacks or loads a near-dead state) → banner band + Main menu button → click → title screen (Task 8 wiring).
6. Rules popup from the title.
7. Screenshot each. Exit play mode.

- [ ] **Step 3: Commit**

```bash
git add -A Assets/HexWars/Presentation
git commit -m "style(ui): every panel on UiKit - one canvas convention, palette, rounded chrome, button states; HelpOverlay dead code removed"
```

---

### Task 10: Full regression, WebGL build, stage, ship

**Files:**
- Modify: `engine/HexWars.NetServer/wwwroot/*` (staged build output)
- No source changes expected.

- [ ] **Step 1: Full test suite + selftest**

```
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --nologo -v q
dotnet run --project engine/HexWars.NetServer -- selftest
```
Expected: all engine tests pass (254 + ~10 new), `SELFTEST PASS`.

- [ ] **Step 2: Editor regression (coplay)**

1. `check_compile_errors` → zero.
2. Play mode: default editor boot still lands in hotseat `NewGame` with HUD (Networked=false path untouched) — screenshot.
3. `ReturnToMenu` → title + demo. Exit play mode.

- [ ] **Step 3: WebGL build**

Count existing `[WebGLBuild]` markers in the Editor log FIRST (count-before-trigger — stale markers have burned us). Then run the build menu method headlessly via coplay `execute_script` (`HexWars.EditorTools.WebGLBuild.Build()` — the class in `Assets/HexWars/Editor/WebGLBuild.cs`) or the established background-invocation pattern from the previous milestone. Poll `Build/WebGL` on the filesystem; expect success in ~5–8 min, exit 0 markers in Editor.log, output ~23 MB.

- [ ] **Step 4: Stage + commit**

```
powershell -File engine/stage-webgl-deploy.ps1
git add engine/HexWars.NetServer/wwwroot
git commit -m "deploy: client build with title screen + game lobby (browse/host/join-by-code/vs-AI over live demo)"
```
`Data/Plugins/lib_burst_generated.wasm` regenerates during the build — leave it uncommitted (established practice).

- [ ] **Step 5: Merge + push (via WSL — Windows shell has no git key)**

Merge `feat/title-lobby` back to main (fast-forward expected), then:

```
wsl git -C /mnt/c/Users/cddal/HexWars push
```
If WSL can't push non-interactively (key passphrase), stop and report — the user pushes from WSL themselves. Render redeploys from the push; the user smoke-tests the live URL on mobile width when back.

---

## Self-review notes (run after drafting — issues found and fixed inline)

- Task 6's `TitleScreen.Reopen` forward-reference is resolved by the compile-placeholder created in Task 6 Step 2 and replaced wholesale in Task 8 — both define the same static signature.
- `SetupForm.RandomCode`/`ShareUrl` are `internal static` because Task 7's verification script and TitleScreen reuse them.
- `MatchHubTests` (existing) keep compiling: both new `Connect` params have defaults; the new ctor param defaults to null.
- Editor vs WebGL: `UiKit.PromptText` returns null in the editor → Join-by-Code degrades to a toast; all other flows are editor-verifiable.
- GameBrowser's editor fallback URL (127.0.0.1:5234) matches NetClient's dev default so editor testing hits the same local server for both the list and the join.
- `GameRules.Show(parent, font, order)` signature is unchanged — TitleScreen passes 1100 like LobbyPanel did.
- Spec §7 connection-lost-while-waiting is covered by `OnNetClosed` (Task 6) — deliberate teardown is
  excluded via NetClient's `_closing` flag so Cancel/ReturnToMenu never toast a fake error.
