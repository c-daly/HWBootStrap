# HexWars — Territory Control Mode (design)

- **Date:** 2026-06-26
- **Status:** Draft for review
- **Scope:** A configurable secondary game mode that layers hex *control* and a coefficient-driven
  *economy* (generators / territory income) onto the existing combat engine.

## 1. Overview

Today HexWars is annihilation-only combat with a barracks economy funded by kill bounty. This mode adds a
second axis: **controlling hexes**. Players capture and hold ground; held hexes (or generators on them)
produce points; holding costs upkeep; and the enemy's generators can be stolen by capturing the ground
under them.

The mode is deliberately a **configurable family**, not a single ruleset. The existing annihilation game
is the degenerate case (Annihilation-only win, no control/economy). Three things are configurable:

1. **Win conditions** — any subset of `{Annihilation, Economy, Score}`.
2. **Economy style** — `Node` (discrete generators) or `Territory` (every controlled hex pays).
3. **A single coefficient table** — every point source and sink is `coefficient × underlying value`.

Design principle: **one control mechanic, one balancing model.** "Node vs territory" is a config/map
choice, not a different engine; every economic flow is a tunable coefficient over a value, so the whole
economy balances from a handful of dials.

## 2. Win conditions (configurable, multiply-selectable)

Independent flags; any subset may be enabled.

- **Annihilation** *(instant, checked every turn):* a side with no living units and no way to field one
  loses; the other wins. (Existing `WinCheck` rule.)
- **Economy** *(instant, checked every turn):* a side reaching `EconomyWinThreshold` banked points wins.
- **Score** *(timed, calculated):* resolves only at the round cap — higher score wins (tie = draw; with
  Score off, the cap is a draw, as today). The score is a **configurable weighted composite** (§6,
  "Score composite"), not a single stat — at minimum it counts the value of enemy units destroyed.

**Resolution:** instant conditions fire immediately, in whatever order they occur; if the game reaches
the round cap with no instant win, Score (if enabled) decides, else draw. Examples: `Annihilation+Economy`
= win two ways, sudden-death; `Economy+Score` = race to threshold, else highest score at the cap.

## 3. Hex control (new state)

A **controller** per hex: `None`, `Player0`, or `Player1`. Terrain `Tile` is immutable and terrain-only,
so control is **separate state** on `GameState` (e.g. a `HexControl` map keyed by `HexCoord`, updated
immutably like the rest of the engine).

- **Capture:** a unit standing on a hex issues a `Capture` command, pays the capture cost → control flips
  to that player and **persists** after the unit leaves (you keep paying upkeep).
- **Recapture / steal:** an enemy flips it by getting a unit onto it and capturing. Because of the
  one-unit-per-column rule, **a garrisoned hex can't be captured** — the enemy must kill the defender
  first, then move on and capture. Defending = garrisoning.
- Capturing a hex that already holds a generator **transfers the generator** (the steal): you inherit a
  working generator for the capture cost only — no rebuild — which makes raiding a strong enemy generator
  a high-value play.

## 4. Per-hex value and the two economy styles

Each hex has a **generation value `v ∈ [0,1]`**. Income from a controlled hex = `v × Output`.

- **Node economy:** value is concentrated in discrete **generators** on specific hexes; all other hexes
  are `v = 0`. Most of the map is plain; the generators are the prizes.
- **Territory economy:** value is **variable per hex** across the map, so *every* controlled hex pays its
  share. Per-hex value is **derived from biome/terrain** — a per-`TerrainType` value coefficient — once
  biomes are reintroduced; while biomes are off, fall back to a uniform or seeded value map. (This gives
  the parked biome system a concrete role to return for.)

Both styles use the identical control mechanic and coefficient table below; they differ only in how `v`
is distributed across the board.

## 5. Generators

A **generator** sits on a hex and carries a strength `s ∈ [0,1]` (its value `v`). Reuses the existing
`Generator` struct (owner, cell, HP) and `GeneratorOutput` config.

- **Build:** on a hex you control, issue `BuildGenerator`, paying `BuildFactor × value`. Strong = pricier.
- **Steal:** capture a hex that already has a neutral or enemy generator (§3) — capture cost only.
- Generators may be **pre-placed neutral** (contested starting prizes) and/or built; both coexist as the
  same object. Neutral and enemy generators are the contested objects of the map.

## 6. Economy — one coefficient table

Every point flow is `coefficient × underlying value`, all tunable (extends the existing `BountyRate`
pattern). Lives in config (a `TerritoryConfig` block, or added to `GameConfig`):

| Flow | Formula | Notes |
|---|---|---|
| **Income / turn** | `Σ over controlled hexes (v × Output)` | the gross |
| **Bounty (kill)** | `BountyRate × victim.PointCost` | *existing* |
| **Capture cost** | `max(CaptureFloor, CaptureFactor × hex income)` | flat `CaptureFloor` on `v=0` hexes (the entry toll that discourages sprawl); value-tied on generators |
| **Build cost** | `BuildFactor × generator income` | income = `s × Output`; scales with what you build |
| **Upkeep / turn** | `UpkeepFactor × hex income` | plain hex earns 0 → upkeep 0 (free to hold) |
| **Economic win** | bank ≥ `EconomyWinThreshold` | only if the Economy win is enabled |

Net payback of a generator ≈ `CaptureFactor / (1 − UpkeepFactor)` turns — a single intuitive dial.
Dimensionless coefficients keep the whole economy easy to balance.

### Score composite (when the Score win is enabled)

The Score is itself a tunable weighted sum of contributions — **at minimum the value of enemy units
destroyed** — same coefficient philosophy as the economy:

| Contribution | Metric | Weight |
|---|---|---|
| **Kills** | cumulative value of enemy units destroyed | `ScoreKills` |
| Banked points | current banked points | `ScorePoints` |
| Territory | `Σ controlled hex value` | `ScoreTerritory` |
| Army | `Σ surviving own unit value` | `ScoreArmy` |

`Score = ScoreKills·kills + ScorePoints·points + ScoreTerritory·territory + ScoreArmy·army`. The
"banked-points-only" case is just `ScorePoints = 1`, the rest 0; a kills-focused score sets `ScoreKills`
high. The Kills term needs a small running tally of destroyed enemy value per player (new state).

**RL-reward version.** One Score preset *is the RL reward function itself* — the same position-value /
advantage the agents optimize today (`RewardShaping.PositionValue` / `Advantage`: committed force +
points-weighted economy), extended with the kills/territory terms above. This deliberately makes the
game's "who's winning" and the AI's objective **one shared, weighted value function** instead of two
parallel definitions — so the score function lives in one place (engine `Score`/`PositionValue`) and is
consumed by both the Score win-check and the RL reward. It also means an agent trained on this mode is
optimizing exactly the score players see. (The kills/territory contributions are precisely the terms the
reward would gain for this mode.)

## 7. Turn integration

- **Economy tick** at the round (or turn) boundary: each player gains `income − upkeep`. Net can be
  negative if over-extended — the over-extension pressure.
- **New commands:** `Capture` (unit on hex) and `BuildGenerator` (on a controlled hex), alongside the
  reused `DeployGenerator`. Each is a normal command, so both turn policies (whole-army / one-action) and
  the AI path work unchanged.
- `LegalMoves` enumerates `Capture`/`BuildGenerator` when this mode is active.

## 8. Engine changes (high level)

- **State:** per-hex control + per-hex value on `GameState`/`Board` (immutable update).
- **Commands:** `Capture`, `BuildGenerator` (+ re-enable `DeployGenerator`), with `GameEngine.Apply`
  handlers and `LegalMoves` enumeration.
- **Economy tick:** applied at the turn/round boundary in the engine.
- **WinCheck:** evaluate the enabled win-condition subset (instant + timed).
- **Config:** the coefficient table + win-condition flags + economy style.
- **Replay:** serialize control state + the mode config (extend `ReplayFile` META, as biomes did).
- **Shared score/value function:** a single engine `Score`/`PositionValue` (weighted contributions) is
  consumed by both the Score win-check and `RewardShaping` — so the human score and the RL reward are one
  definition (§6, "RL-reward version").
- **RL/obs:** *training* agents on this mode is deferred (the RL setup stays on the combat mode for now);
  the scoring above is reward-ready, and a future obs would add control/value planes.

## 9. Mode selection

A mode flag plus the config block, exposed on `GameBootstrap` for the playable game (like the existing
`BiomesEnabled` / `OneActionPerTurn` toggles) and constructable headlessly for the Sim/tests.

## 10. Phasing

The full mode is sizeable; build in slices, each independently testable:

1. **Control + win config** — per-hex control state, the `Capture` command, configurable
   multiply-selectable win conditions (Annihilation/Economy/Score) with the resolution rule. No generators
   yet (capture is free or flat-cost; Economy/Score scored off banked points).
2. **Generators + economy** — `BuildGenerator`/steal, the coefficient table, the economy tick, node
   economy. Capture/build/upkeep costs active.
3. **Territory economy + UI** — variable per-hex value (biome-tied), and presentation: control overlay
   (hex tint by controller), income/upkeep readout, capture/build affordances, AI support for the new
   commands.

## 11. Open questions / future

- **UI/UX:** how control reads on the board (tint, borders), income/upkeep display, capture/build input.
- **AI:** greedy/RL support for `Capture`/`BuildGenerator` so the AI opponent plays this mode.
- **Score weights:** default coefficients for the Score composite (how much kills vs points vs territory
  vs army count) — needs play-testing to tune.
- **Biome reintroduction** drives territory per-hex value — coordinate with that work.
