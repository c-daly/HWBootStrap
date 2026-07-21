# Session Barracks and Tooltip QoL Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the complete human barracks survive new games for the current app session and expose every template's full statistics on hover, focus, or touch.

**Architecture:** An in-memory presentation cache owns each local human's template catalog. Game construction accepts explicit starting barracks, and online rooms collect a sanitized catalog from each seat before constructing the authoritative start state. A dedicated tooltip view renders immutable template details without intercepting selection.

**Tech Stack:** Unity/C#, immutable HexWars engine state, WebSocket protocol, NUnit.

## Global Constraints

- No persistence beyond the running app/browser session.
- Each cache begins with the five built-in templates.
- Creating adds; deleting any template removes it for the rest of the session.
- Exact name-and-stat duplicates appear once.
- AI barracks remain the default starter set.
- Online cache changes occur only after authoritative confirmation.
- Opponent barracks remain hidden by the existing shown-seat rules.
- After every C# edit, run Unity `check_compile_errors`.

---

### Task 1: Reusable barracks catalog primitives

**Files:**
- Create: `engine/HexWars.Engine/BarracksCatalog.cs`
- Modify: `engine/HexWars.Engine/Net/GameSetup.cs:69`
- Create: `engine/HexWars.Engine.Tests/BarracksCatalogTests.cs`
- Modify: `engine/HexWars.Engine.Tests/GameFactoryStarterTemplatesTests.cs`

**Interfaces:**
- Produces: `BarracksCatalog.DefaultTemplates`, `BarracksCatalog.Normalize(IEnumerable<UnitTemplate>, int maxTemplates)`.
- Produces: `GameFactory.Build(GameSetup setup, IReadOnlyList<UnitTemplate>? p0Barracks, IReadOnlyList<UnitTemplate>? p1Barracks, bool beginInDeployment = true)`; the deployment flag is consumed by the later deployment plan.

- [ ] **Step 1: Write failing engine tests for the exact five defaults, exact duplicate removal, name sanitization, invalid-health rejection, and a fixed protocol maximum of 64 templates.**
- [ ] **Step 2: Run the focused engine tests and confirm failure before implementation.**
- [ ] **Step 3: Move the starter table into `BarracksCatalog` and implement normalized value equality over name plus all nine stats.**

```csharp
public static IReadOnlyList<UnitTemplate> Normalize(IEnumerable<UnitTemplate> source, int maxTemplates = 64)
{
    var result = new List<UnitTemplate>();
    foreach (var raw in source ?? Array.Empty<UnitTemplate>())
    {
        var item = new UnitTemplate(UnitTemplate.Sanitize(raw.Name), raw.Stats);
        if (item.Stats.Health < 1 || result.Any(x => Same(x, item))) continue;
        if (result.Count == maxTemplates) break;
        result.Add(item);
    }
    return result;
}
```

- [ ] **Step 4: Extend `GameFactory.Build` to copy supplied barracks or defaults into each player without sharing mutable lists.**
- [ ] **Step 5: Run the full engine suite; expect existing starter-template behavior plus new custom-catalog cases to pass.**
- [ ] **Step 6: Commit.**

```powershell
git add engine/HexWars.Engine/BarracksCatalog.cs engine/HexWars.Engine/Net/GameSetup.cs engine/HexWars.Engine.Tests/BarracksCatalogTests.cs engine/HexWars.Engine.Tests/GameFactoryStarterTemplatesTests.cs
git commit -m "feat(engine): accept normalized starting barracks catalogs"
```

### Task 2: Local session cache and local game injection

**Files:**
- Create: `Assets/HexWars/Presentation/SessionBarracksCache.cs`
- Create: `Assets/HexWars/Presentation/SessionBarracksCache.cs.meta`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs:75`
- Modify: `Assets/HexWars/Presentation/DesignPanel.cs:195`
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs:197`
- Create: `Assets/HexWars/Tests/Editor/SessionBarracksCacheTests.cs`
- Create: `Assets/HexWars/Tests/Editor/SessionBarracksCacheTests.cs.meta`

**Interfaces:**
- Consumes: `BarracksCatalog.DefaultTemplates` and `Normalize`.
- Produces: `SessionBarracksCache.ForLocalPlayer(int localPlayer)`, `Add`, `RemoveAt`, `Snapshot`, and `ResetForTests`.

- [ ] **Step 1: Write EditMode tests for default initialization, add, deduplication, delete of built-in/custom entries, independent hotseat catalogs, snapshot copying, and test reset.**
- [ ] **Step 2: Run focused tests and confirm failure before implementation.**
- [ ] **Step 3: Implement a process-lifetime static cache containing two independent lists, initialized lazily from normalized defaults. Do not use `PlayerPrefs`.**
- [ ] **Step 4: Inject the cache through every real local new-game path.** In `StartLocalGame`, use cache 0 for the human in vs-AI, defaults for the AI, and caches 0/1 for hotseat. Update the legacy inspector-driven `NewGame()`/`BuildPlayer` path to make the same human-versus-AI or hotseat choice. The title-screen demo must continue using defaults and must never mutate a human cache.
- [ ] **Step 5: Update confirmed local create/delete flows.** Add only after successful `TryApply`; delete the same cache index only after successful apply. Online paths remain unchanged until Task 4.
- [ ] **Step 6: Run `check_compile_errors`, cache tests, and Play Mode new-game checks through both `StartLocalGame` and legacy `NewGame()`. Confirm a deleted starter and created template survive Return to Menu → New Game, but Editor Play restart restores defaults.**
- [ ] **Step 7: Commit.**

```powershell
git add Assets/HexWars/Presentation/SessionBarracksCache.cs Assets/HexWars/Presentation/SessionBarracksCache.cs.meta Assets/HexWars/Presentation/GameBootstrap.cs Assets/HexWars/Presentation/DesignPanel.cs Assets/HexWars/Presentation/BarracksPanel.cs Assets/HexWars/Tests/Editor/SessionBarracksCacheTests.cs Assets/HexWars/Tests/Editor/SessionBarracksCacheTests.cs.meta
git commit -m "feat(barracks): retain local templates for the app session"
```

### Task 3: Versioned barracks wire payload

**Files:**
- Create: `engine/HexWars.Engine/Net/BarracksWire.cs`
- Modify: `engine/HexWars.Engine/Net/NetProtocol.cs`
- Modify: `engine/HexWars.Engine.Tests/NetProtocolTests.cs`
- Create: `engine/HexWars.Engine.Tests/BarracksWireTests.cs`

**Interfaces:**
- Produces: `BarracksWire.Write(IReadOnlyList<UnitTemplate>)` and `BarracksWire.Read(string)`.
- Produces: `NetProtocol.Catalog(string payload)` and parsed message kind `CATALOG`.

- [ ] **Step 1: Write failing tests for round trips, spaces and punctuation in sanitized names, all nine stats, empty lists, 64-entry truncation, malformed records, and a maximum encoded payload of 32 KiB.**
- [ ] **Step 2: Run focused tests and confirm failure.**
- [ ] **Step 3: Implement a versioned, length-bounded text payload with only framework types already available to the netstandard Unity engine assembly. Encode each sanitized name as UTF-8 Base64, keep the nine integer stats as delimited invariant numbers, normalize after parsing, and reject a payload over 32 KiB before allocating its records. Do not add `System.Text.Json` or another package dependency to `HexWars.Engine`.**
- [ ] **Step 4: Add the protocol message and parsing tests without changing command wire encoding.**
- [ ] **Step 5: Run the complete engine suite; expect zero failures.**
- [ ] **Step 6: Commit.**

```powershell
git add engine/HexWars.Engine/Net/BarracksWire.cs engine/HexWars.Engine/Net/NetProtocol.cs engine/HexWars.Engine.Tests/BarracksWireTests.cs engine/HexWars.Engine.Tests/NetProtocolTests.cs
git commit -m "feat(net): add bounded starting barracks payload"
```

### Task 4: Authoritative online seat catalogs

**Files:**
- Modify: `engine/HexWars.Engine/Net/MatchHub.cs:100`
- Modify: `engine/HexWars.Engine/Net/GameSession.cs`
- Modify: `engine/HexWars.Engine.Tests/MatchHubTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TokenRejoinTests.cs`
- Modify: `engine/HexWars.NetServer/Program.cs`
- Modify: `Assets/HexWars/Presentation/NetClient.cs`
- Modify: `Assets/HexWars/Presentation/GameBootstrap.cs:206`

**Interfaces:**
- Consumes: `BarracksWire`, `GameFactory.Build(...p0Barracks, p1Barracks...)`, `SessionBarracksCache.Snapshot()`.
- Produces: a room lifecycle that constructs `Room.Start` only after both seated players have supplied normalized catalogs.

- [ ] **Step 1: Write MatchHub tests first.** Cover distinct P0/P1 catalogs, a missing/malformed catalog falling back to defaults, reconnect not replacing a started match, and third-party/spectator catalog rejection.
- [ ] **Step 2: Run focused tests and confirm the current eager room construction fails them.**
- [ ] **Step 3: Change the waiting room to store setup plus per-seat catalogs. `Connect` sends `SEAT` but never `START`; the room builds its authoritative state exactly once after both seated connections have submitted a valid or defaulted `CATALOG`. Preserve listing, private-room, token-rejoin, and start replay behavior.**
- [ ] **Step 4: Have `NetClient` send the local normalized catalog immediately after `SEAT`. A malformed catalog marks that seat received with server defaults; a missing catalog keeps the room waiting and surfaces a connection/setup error rather than silently starting with the wrong roster.** Do not put a large catalog in the WebSocket URL.
- [ ] **Step 5: On an online `APPLY CreateUnit/DeleteTemplate` issued by the local seat, update the local session cache from the confirmed new state/index. Never mutate it on optimistic send or `REJECT`.**
- [ ] **Step 6: Run engine, hub, reconnect, and server self-tests; rebuild/copy the engine DLL; run `check_compile_errors`.**
- [ ] **Step 7: Use two clients with different caches and verify each receives its own barracks on start and after reconnect.**
- [ ] **Step 8: Commit.**

```powershell
git add engine/HexWars.Engine/Net/MatchHub.cs engine/HexWars.Engine/Net/GameSession.cs engine/HexWars.Engine.Tests/MatchHubTests.cs engine/HexWars.Engine.Tests/TokenRejoinTests.cs engine/HexWars.NetServer/Program.cs Assets/HexWars/Presentation/NetClient.cs Assets/HexWars/Presentation/GameBootstrap.cs
git commit -m "feat(net): seed online seats from session barracks"
```

### Task 5: Non-modal barracks template tooltip

**Files:**
- Create: `Assets/HexWars/Presentation/BarracksTemplateTooltip.cs`
- Create: `Assets/HexWars/Presentation/BarracksTemplateTooltip.cs.meta`
- Modify: `Assets/HexWars/Presentation/BarracksPanel.cs:137`
- Create: `Assets/HexWars/Tests/Editor/BarracksTemplateTooltipTests.cs`
- Create: `Assets/HexWars/Tests/Editor/BarracksTemplateTooltipTests.cs.meta`

**Interfaces:**
- Consumes: `UnitTemplate`, `Roles.Dominant`, `Economy.DeployCost`, and current `GameConfig`.
- Produces: `BarracksTemplateTooltip.Show(RectTransform anchor, UnitTemplate template, GameConfig config)` and `Hide()`.

- [ ] **Step 1: Write EditMode tests for full name, role, point/deploy costs, all nine full labels, canvas-edge clamping, pointer exit, keyboard focus, and the touch info target.**
- [ ] **Step 2: Run focused tests and confirm failure before implementation.**
- [ ] **Step 3: Implement the tooltip as a collider-free, non-modal overlay at tooltip sort order. It must set child graphics `raycastTarget = false` so row selection remains unchanged.**
- [ ] **Step 4: Add pointer-enter/exit and select/deselect handlers to each barracks row plus a touch info button. Hide the tooltip during rebuild, panel close, game change, and row destruction.**
- [ ] **Step 5: Run `check_compile_errors` and presentation tests. In Play Mode hover every built-in and a long-name custom template at screen edges, then select/deploy through the same row. Inspect logs.**
- [ ] **Step 6: Commit.**

```powershell
git add Assets/HexWars/Presentation/BarracksTemplateTooltip.cs Assets/HexWars/Presentation/BarracksTemplateTooltip.cs.meta Assets/HexWars/Presentation/BarracksPanel.cs Assets/HexWars/Tests/Editor/BarracksTemplateTooltipTests.cs Assets/HexWars/Tests/Editor/BarracksTemplateTooltipTests.cs.meta
git commit -m "feat(barracks): show complete template stats on hover"
```

### Task 6: Integrated verification

**Files:**
- Modify only if verification exposes a defect in files already listed above.

- [ ] **Step 1: Run the complete engine and Unity EditMode suites; expect zero failures.**
- [ ] **Step 2: Test local vs-AI, local hotseat, online P0, and online P1 creation/deletion across at least two new games in one session.**
- [ ] **Step 3: Reload the browser/app and confirm all caches return to the five defaults.**
- [ ] **Step 4: Verify hover, keyboard focus, and touch info behavior at landscape and portrait sizes; ensure opponent barracks never render.**
- [ ] **Step 5: Inspect Unity and server logs, run `git diff --check`, and commit only narrowly scoped corrections.**
