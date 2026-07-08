# Title Screen & Game Lobby — Design

**Date:** 2026-07-08
**Milestone:** The web build gets a real front door: a title screen over a live demo game, and a
lobby where matching is easy — browse open games (with their configuration), join by match code,
host a game (public or private), or play the AI. Plus a professional visual pass over all
runtime menus/panels via a shared UI kit.
**Approach:** HTTP lobby-list endpoint + existing WebSocket join (approved over a full WebSocket
lobby protocol — smallest honest version; live push can be layered on later without rework).

## 1. Problem

The deployed WebGL build boots straight into a settings form (`LobbyPanel`). There is no title
screen, no way to discover games (joining works **only** by opening a shared `?room=` link — you
cannot type a code), no matchmaking of any kind, and the "waiting for opponent" screen is a dead
end with no cancel. Visually, every panel styles itself ad hoc: three different canvas coordinate
spaces, per-file colors, nine copies of the same widget-factory boilerplate — the result reads as
a prototype.

The server (`MatchHub` + `HexWars.NetServer`) already pairs the first two arrivals in a named
room and deals the authoritative start state; it just has no notion of *listing* rooms — it
discards the host's `GameSetup` after building the state and forgets when a room started.

## 2. Goals and non-goals

**Goals**

- **Title screen = main menu over a live demo.** A muted AI-vs-AI game (fog off, gameplay UI
  suppressed) plays behind the HEXWARS wordmark and five actions: Browse Games, Host Game,
  Join by Code, Play vs AI, How to Play. The demo restarts when it ends and never blocks the menu.
- **Game browser.** A list of open public games showing each game's configuration (mode, map
  size, fog, pace, army size, age); tap to expand the full config; Join takes the open seat.
  You never end up in a game you didn't want.
- **Join by match code.** Type a room code (browser prompt — the established mobile-WebGL text
  entry); joins exactly like a `?room=` link. Shared links keep working and still bypass the title.
- **Host public or private.** The existing settings form plus a Private toggle (private = never
  listed; joinable only by code/link). The waiting screen shows the room code prominently, the
  share link, and a working **Cancel**.
- **Play vs AI** through the same settings form, with a Difficulty row (Easy = Random,
  Hard = Greedy — exposing the `AiLevel` that exists in code but was never in the UI).
- **Professional visual pass** over all runtime menus/panels through one shared `UiKit`
  (single canvas convention, palette, typography scale, rounded corners, button states).
- **Server hardening folded in** (audit follow-through, now that hosting arbitrary configs is a
  first-class flow): `GameSetup` field clamps server-side; room-code normalization.

**Non-goals**

- No random/quick-match queue (superseded by the browser; can be added later as a one-tap
  "join newest open game").
- No live-push lobby updates (poll is fine at current scale); no lobby chat, player names, or
  accounts. Games are identified by room code + config only.
- No NetClient reconnect/backoff work (separate audit follow-up U2) beyond what the new flows
  need (cancel, join-failure surfacing).
- No engine *game-rules* changes. Engine work is confined to `Net\` (MatchHub/GameSetup) and is
  additive to the wire behavior.
- No changes to the RL/gym side.

## 3. Screen flow (web build)

```
boot ──?room= link──────────────────────────► join room directly (existing; kept)
  │
  └─► start demo game (muted, UI-suppressed) + TitleScreen overlay
        ├─ Browse Games ─► list (poll /games 3s) ─► row detail ─► Join ─► seat+start
        ├─ Host Game ────► SetupForm(Host: +Private) ─► waiting screen (code, link, Cancel)
        ├─ Join by Code ─► browser prompt ─► join ─► seat+start (bad/full code → toast, stay)
        ├─ Play vs AI ───► SetupForm(VsAi: +Difficulty) ─► local game + AiOpponent
        └─ How to Play ──► rules popup (restyled)
game over / pause menu ─► ReturnToMenu ─► title + fresh demo
```

Details:

- **Demo:** Greedy vs Greedy on a standard random map (fresh seed each run), fog off, sound
  muted, `SpectatorDriver` pacing, camera glides on. Suppressed while demo runs: GameHud,
  BarracksPanel, EventConsole sidebar, toasts, game-over banner, unit input (ReadOnly). On
  `IsGameOver`, hold ~3 s, restart with a new seed. Sub-screens overlay it; it keeps playing.
- **Browse list rows:** `CODE — Mode · W×H · Fog? · pace · N units · age` (e.g.
  `KQ7KP — Territory · 13×9 · Fog · 3 acts/turn · 5 units · 2m ago`), newest first. Tap →
  expanded card with every setting + Join. Empty state: "No open games right now — host one!"
  with a Host shortcut button. Manual refresh button alongside the 3 s poll.
- **Waiting screen (host):** room code in large type (readable aloud — that is the primary
  sharing mechanism), the full share URL displayed beneath it (no copy button — WebGL clipboard
  is unreliable; the code is the point), and Cancel → destroys the socket, clears the seat,
  returns to title.
- **Join races:** if the seat fills between list refresh and tap (or a stale link/code), the
  server answers `SEAT_FULL`; the client toasts "That game just filled", disconnects (no silent
  spectator limbo — this replaces today's silent-spectate behavior on full rooms), and refreshes
  the list.

## 4. Visual style — `UiKit`

One static class is the single source of style; every panel below is rebuilt on it. This
supersedes the nine copy-pasted widget factories and three canvas conventions (audit U8/F14).

- **Canvas:** ScreenSpaceOverlay, reference 1600×900, matchWidthOrHeight 0.5 — everywhere.
  Sorting orders documented in one place (HUD 100, panels 200, tooltip 300, toast 800,
  rules/help 900, title/lobby 1000, banner 1100).
- **Palette (constants):** background `#0A0E1C`; panel surface `#161B2C` + 1-px lighter border
  `#2A3350`; accent = player cyan `#45AEFF` family for selection/links; primary CTA green
  `#33845C`; danger `#B04040`; text white / `#9AA3B8` secondary / `#6C7488` caption.
- **Typography:** LegacyRuntime.ttf (zero-asset build), sizes 26 title / 20 heading / 16 body /
  13 caption. Consistent anchors, 12-px padding rhythm.
- **Chrome:** one procedurally generated 9-sliced rounded-rect sprite (built once, cached) used
  by panels and buttons; buttons get hover/pressed tints via ColorBlock; toggle-buttons get a
  filled accent treatment when selected.
- **API sketch:** `UiKit.Canvas(name, order)`, `Panel(parent, rect, style)`,
  `Label(parent, text, size, rect, anchor)`, `Button(parent, text, rect, onClick, style)`,
  `ToggleButton(...)`, `ValueBox(...)` (keeps the tap-to-type browser-prompt + −/+ pattern),
  `Row(...)` helpers, `EnsureEventSystem()`, palette/type constants.
- **Restyled surfaces:** TitleScreen, GameBrowser, SetupForm, waiting screen, GameRules popup,
  Toast, GameOverBanner, GameHud top bar, BarracksPanel, DesignPanel, UnitTooltip, HelpOverlay
  button. Same bones and behavior, consistent skin. (Existing panels keep their logic; their
  build code moves onto UiKit.)

## 5. Server & protocol

**MatchHub** (engine `Net\`, stays pure/testable; clock injected):

- `Room` gains `GameSetup Setup`, `bool IsPrivate`, `bool Started`, `long CreatedAtTicks`.
  `Started` is set when the start state is dealt (second seat filled) and never cleared — a
  started room that drops to one member must not re-list.
- Constructor takes `Func<long> utcNowTicks` (default `() => DateTime.UtcNow.Ticks`); tests
  inject a fixed clock.
- `Connect(roomCode, connectionId, setup, isPrivate)` — privacy recorded at room creation;
  joiners' flags ignored (host semantics, same as setup).
- New `IReadOnlyList<OpenGame> OpenGames()` — public + one member + not started, newest first.
  `OpenGame` = `(string Code, GameSetup Setup, int AgeSeconds)`.

**GameSetup:** add `GameSetup Sanitized()` clamping every field to the lobby form's own ranges —
Width/Height [5,24], StartingPoints [0,200], ArmySize [1,12], Brutes/Strikers/Snipers [0,12],
TurnActions [0,8], Seed [1,99999], Mode to defined enum values. `GameFactory.Build` calls it
(one line), so every construction path — server, lobby, tests — is covered. (Audit finding N1.)

**NetServer:**

- `/ws` reads `?private=1` and forwards to `Connect`; room codes normalized (trim, uppercase,
  strip non-alphanumerics, cap 16 chars; empty → "default"). (Audit M13, partial.)
- New `GET /games`: under the existing hub lock, project `OpenGames()` to JSON via
  System.Text.Json: `[{"code","mode","width","height","fog","pace","army","ageSeconds"}]`
  (`mode` as string, `pace` = TurnActions, `army` = ArmySize). Same origin as the client —
  no CORS.
- Wire protocol (SEAT/START/APPLY/REJECT) unchanged. Old clients keep working.

**Build chain note:** MatchHub/GameSetup live in the engine assembly →
`Assets/HexWars/Plugins/HexWars.Engine.dll` must be re-synced after engine-side changes or Unity
boots to Safe Mode (and takes the Coplay MCP down).

## 6. Client components

| Component | Role |
|---|---|
| `UiKit.cs` (new) | Style system + widget factory (section 4). No game logic. |
| `TitleScreen.cs` (new) | Wordmark + five-action menu + routing; owns demo lifecycle (start, watch `IsGameOver`, restart); join-by-code prompt; version footer. Dismisses when a real (non-demo) game exists. |
| `GameBrowser.cs` (new) | List panel: coroutine polls `{pageOrigin}/games` every 3 s while visible (UnityWebRequest), renders rows, expandable detail card, Join → `StartNetGame(code, null)`. Manual refresh; inline error status. |
| `SetupForm.cs` (LobbyPanel refactored + renamed) | The settings form in two modes: **Host** (adds Private toggle; Create → `StartNetGame(code, wire, isPrivate)` → waiting screen with Cancel) and **VsAi** (adds Difficulty row, default Hard; Create → `StartLocalGame(setup, ai, level)`). Keeps browser-prompt ValueBoxes and the army popup. |
| `GameBootstrap` (modified) | `DemoMode` flag + `StartDemo()`; WebGL boot → demo + TitleScreen (instead of LobbyPanel); `StartNetGame` gains the private flag (connect query); `ReturnToMenu` → title + fresh demo; suppression contract below. |
| `NetClient` (touched) | Connect URL gains `&private=1` when hosting private; `SEAT_FULL` surfaces to a callback the join flows toast on (replacing the silent spectator warning). |

**Demo suppression contract:** `GameBootstrap.DemoMode` is true from demo start until a real
game starts (any lobby-initiated game clears it). While true: `GameHud.Refresh` hides its root
(which also silences its turn-handover toasts), `BarracksPanel.Rebuild` hides,
`EventConsole.Report` ignores state (and draws nothing), `GameOverBanner` never shows,
`UnitInputController.ReadOnly` = true, `SoundManager` muted. Each is a one-line early-out
checking the flag — no new event plumbing. `Toast` itself is NOT suppressed: the title's own
flows (join failures, cancel confirmations) toast over the demo deliberately.

## 7. Error handling

- `/games` fetch failure or non-200: inline status in the browser panel ("Can't reach the
  server — retrying…"), poll continues; Join buttons disabled while errored.
- Join a full/vanished room (list race, stale code/link): toast "That game just filled" /
  "No such game", disconnect the socket, auto-refresh the list. Applies to `?room=` links too.
- Waiting screen Cancel: destroy NetClient, `Seat = null`, back to title (demo still running).
  Fixes the current dead end (audit U11).
- Server restart while browsing: poll errors surface as above; hosting/waiting sockets drop —
  the waiting screen shows "Connection lost" with a Back button (full reconnect remains the
  separate U2 follow-up).
- Demo failure of any kind (exception, instant game over): restart the demo; the title menu is
  never gated on demo health.
- Malformed `/games` JSON on the client: treated as fetch failure (status + retry), never throws.

## 8. Verification

- **Engine tests (new):** MatchHub — OpenGames lists a waiting public room; excludes private
  rooms, started rooms, full rooms, and emptied (removed) rooms; ordering newest-first;
  age uses the injected clock; joiner's setup/privacy ignored. GameSetup — Sanitized clamps
  every field (both directions), legal values pass through unchanged.
- **NetServer selftest** extended: host connects (public) → `/games` shows it; second client
  joins → list empties; private room never appears.
- **Play-mode (coplay, screenshots per screen** — lobby screens are verified visually per
  standing practice): title over a running demo (HUD/console absent, sound muted), Browse with
  a live row (needs the local NetServer running) and empty state, Host form + waiting screen +
  Cancel round-trip, vs-AI form → game starts with difficulty applied, rules popup, and a
  demo-restart observed after game over.
- **Regression:** hotseat editor path (Networked = false) still boots `NewGame` untouched;
  `?room=` link join; a full engine test run; existing 254 tests stay green.
- **Ship:** WebGL build → stage to `wwwroot` → commit → push (via WSL — Windows has no git
  key) → Render deploy → mobile-width smoke by the user.

## 9. Success criterion

A first-time visitor on a phone opens the page and, without instructions: watches a game
playing behind the title, browses open games with configs visible, joins one they like (or
types a friend's code, or hosts and shares), and the whole journey — title → lobby → seated
match — looks like one deliberately designed product rather than a developer prototype.

## 10. Risks

- **Demo perf on mobile** (a full game simulating behind the menu): mitigated by fog-off
  standard map, muted audio, suppressed UI; if needed, cap demo pacing (SpectatorDriver
  interval) — the menu never waits on it.
- **UiKit restyle regressions** in existing panels (barracks deploy flow, tooltip dock,
  HUD buttons): mitigated by keeping panel logic untouched (build-code-only changes) and the
  play-mode screenshot sweep.
- **Engine-assembly DLL drift** breaking the Unity project: the plugins-DLL re-sync step is an
  explicit plan task after any `Net\` change.
- **Room-code normalization vs existing links:** normalization is case-insensitive-friendly
  (uppercase both on create and join), and `RandomCode()` already emits the safe alphabet.
