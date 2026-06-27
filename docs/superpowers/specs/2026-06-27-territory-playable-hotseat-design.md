# HexWars — Territory Control, Playable Hotseat (Phase 3, Slice 1) — design

- **Date:** 2026-06-27
- **Status:** Draft for review
- **Parent spec:** `docs/superpowers/specs/2026-06-26-territory-control-mode-design.md` (this is the first shippable slice of that spec's Phase 3).
- **Scope:** Make territory mode **playable end-to-end in a hotseat game** — turn the finished territory engine (Phases 1 & 2, merged) into something you can see and play, with one new gameplay rule (the claim tempo) and the deploy/build-on-control gate.

## 1. Overview

The territory **engine** already exists: per-hex control, `CaptureHex`, generators with strength, control-based income + upkeep, `BuildGenerator`, capture cost-scaling + steal, economy tick, configurable win conditions. What's missing is (a) a small set of **mode rules** that make territory play distinct, and (b) the **presentation** to see and drive it. This slice delivers both, for a two-human hotseat game.

The defining mechanic: **you are vulnerable as you expand.** Claiming ground is a turn-exclusive commitment, so a turn spent expanding is a turn your army can't move or fight, and the claiming unit sits exposed on forward ground. Fight-now vs. expand-for-later, decided one turn at a time.

## 2. Mode rules (the new gameplay)

These rules are gated by a **`TerritoryMode`** flag on `GameConfig` (default off, so the base annihilation game is unchanged).

### 2.1 Control is the economic gate
You may **deploy units** and **build generators** only on a hex you **control**. (Today `BuildGenerator` already requires control; this slice adds the same requirement to `DeployUnit` — replacing the fixed deployment-zone rule when `TerritoryMode` is on.)

### 2.2 Claiming is a turn-exclusive opening action
Making a hex yours (the `CaptureHex` command) is, in `TerritoryMode`:
- **First-or-not-at-all:** legal only if you have not yet moved or attacked this turn. Once your army has acted, you cannot claim until next turn.
- **Turn-ending:** issuing a claim immediately ends your turn (regardless of turn policy).
- Unchanged from Phases 1–2: you must have a **living unit standing on the target hex** (positioned on an earlier turn); you pay the capture cost (flat on an empty hex, scaled on a generator hex); claiming an enemy generator hex **transfers the generator** (the steal).

So each turn you choose: **declare a claim** (your whole turn) **or play your army** (move/attack/build/deploy, then End Turn) — never both.

### 2.3 Taking enemy ground
Same command, same rule. Because of one-unit-per-column you cannot stand on a hex an enemy unit occupies, so converting enemy ground means: **clear the defender in combat** (so the hex frees up), **occupy it** (move a unit on, a later turn), then **convert it** (a claim turn). Converting an enemy-controlled generator hex inherits the generator.

## 3. Starting state

A territory game is set up so the loop can begin:
- Each player **starts controlling their home/deployment-zone hexes** (so they have legal ground to deploy from on turn 1).
- Each player starts with a **points pool** (`StartingPoints`, tunable; default for this mode ~40) and **no generators**.
- The rest of the board (notably the contested middle) is **neutral** (uncontrolled).
- Combat uses the **whole-army** turn policy (move/attack with your whole army on a normal turn); only claiming is turn-exclusive.

Loop: march a unit onto neutral ground → spend a turn to claim it → build a generator on it (a later turn) → income each turn (minus upkeep) → bank toward an Economy win, expand further, or raid/steal the enemy's generators.

## 4. Win conditions

Default for this mode: **Economy + Annihilation** (bank `EconomyWinThreshold` points to win, or eliminate the enemy). Score is available but off by default. All are the already-implemented `WinBy` flags; no new win logic.

## 5. Presentation

Thin, code-built UI consistent with the existing presentation layer.

- **Control overlay:** every hex you occupy (control) carries its **owner's color** on the top face, clearly readable at a glance — your color vs. the enemy's; neutral hexes are untinted. Lives in `BoardRenderer`/`TileView`, refreshed on `StateChanged`.
- **Claim/build input (click-the-hex):** select a unit (existing `UnitInputController`); a one-line hint states exactly what clicking its hex will do and the cost — e.g. "Click to **Claim (3)** — ends your turn", "Click to **Build generator (4)**", or why it's unavailable (not your turn / already acted / can't afford / already controlled). Clicking the unit's own hex issues `CaptureHex` if you don't control it, or `BuildGenerator` if you control it and it's empty. The claim affordance only appears when the claim is legal (turn not yet spent). The result (and any rejection reason) is logged to the existing `EventConsole`.
- **Economy/score HUD:** for each player, show points, income − upkeep per turn, controlled-hex count, and score. Extends the existing HUD/`EventConsole` scoreboard.
- **Mode enable:** a `TerritoryMode` toggle on `GameBootstrap` (alongside `BiomesEnabled`/`OneActionPerTurn`/`VsAI`) that builds the territory config, seeds home-zone control, and applies the starting points. (A full New-Game setup screen is the next slice.)

## 6. Engine vs. presentation split

- **Engine (`engine/HexWars.Engine`, netstandard2.1, NUnit):**
  - `GameConfig.TerritoryMode` flag (optional, last, default false).
  - `DeployUnit`: when `TerritoryMode`, require the target hex be controlled by the issuer (instead of the deployment-zone check); off-mode behavior unchanged.
  - `CaptureHex`: when `TerritoryMode`, reject unless it's the turn's first army action (no units moved/attacked yet) and end the turn on success. A new `RejectionReason` (e.g. `MustClaimFirst`). Off-mode capture behavior unchanged.
  - `LegalMoves`: enumerate `CaptureHex` only when the claim is currently legal (first-action) in `TerritoryMode`.
  - A small reusable helper to seed a player's control over a set of hexes at game start (used by setup + tests).
- **Presentation (`Assets/HexWars/Presentation`, Unity, not unit-tested):** control overlay, click-to-claim/build + hint, economy/score HUD, `GameBootstrap` territory setup. After engine changes, **re-sync the Plugins DLL** (`Assets/HexWars/Plugins/HexWars.Engine.dll`).

## 7. Testing

- **Engine (TDD, NUnit):** deploy rejected off-controlled / allowed on-controlled in `TerritoryMode`; deploy unchanged off-mode; claim rejected after an army action (`MustClaimFirst`) and allowed as first action; claim ends the turn; `LegalMoves` omits/includes `CaptureHex` per first-action legality; seed-control helper. Existing 184 tests stay green (all new behavior is gated by `TerritoryMode`, default off).
- **Presentation:** not unit-tested (Unity); verified by playing a hotseat territory game (claim → build → income → bank/steal) and by `check_compile_errors` after the DLL re-sync.

## 8. Non-goals (later Phase 3 slices)

New-Game setup screen (slice 2); AI playing the mode — `AiOpponent` using `CaptureHex`/`BuildGenerator` (slice 3); territory per-hex value, biome reintroduction, neutral pre-placed generators + the 0–1 strength spread (slice 4); replay serialization of control + mode config (slice 5); folding `WinCheck.Score` with the RL reward (slice 6).

## 9. Open questions / future

- **Starting points and economy tuning** (`StartingPoints`, `GeneratorOutput`, `EconomyWinThreshold`, capture/build/upkeep factors) need play-testing; defaults here are provisional.
- Whether building/deploying before a claim should also forfeit the claim (this slice defines "first" as "no unit has moved or attacked"); revisit if it feels off in play.
- Whether the claiming unit should get any defensive penalty/bonus while exposed (out of scope now).
