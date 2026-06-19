# HexWars — Design Spec

**Date:** 2026-06-18
**Status:** Draft for review
**Engine target:** Unity 6 (6000.5.0f1), URP, Input System
**Milestone of this spec:** Milestone 1 — Hotseat vertical slice

---

## 1. Vision

HexWars is a turn-based tactics game played on a platform of hexes floating in space. The
rules are deliberately small; the depth is meant to be emergent. A single currency —
**points** — buys *everything*: you design each unit from scratch by pouring points into its
stats, you build income structures, and you fight to annihilate the enemy. Terrain and
elevation make position matter; a transparent economy makes every purchase a real trade-off.

Two players (milestone 1: both human, hotseat on one screen). Later milestones add a computer
opponent and networked multiplayer without re-architecting.

---

## 2. Design goals (north star)

**"Flat pricing, deep interactions."**

1. **The point system is obvious and intuitive.** Flat, transparent pricing: **1 point = 1
   step** of any stat. No nonlinear cost curves, no hidden formulas. A player can do the math
   in their head. *This is a hard design rule, not a default.*
2. **Strategic depth is emergent, never priced-in.** Complexity comes from how stats interact
   with each other and with the board — not from the price list. Because a unit is bought
   **entirely from zero**, every unit is a specialist with an obvious weakness, which is what
   makes odd specialists viable and surprising.
3. **No dominant build.** Extreme/quirky builds should be *situationally* strong, not a single
   boring meta. The structural counters in §6–§8 (vision gates range, movement gates
   positioning, attrition punishes glass, elevation/terrain soft-counter parity, bounties
   punish hoarding) keep the design space wide. Balancing = keeping it wide and counterable,
   not making all builds equal.
4. **Self-documenting — it's mostly what you buy.** Beyond a **thin framework** of structural
   rules (turn order and how many actions per turn — the `ITurnPolicy` of §3), a player should
   never need a manual. The **buy menu is the rules**: stat names *are* their effects, priced
   1:1, so understanding what you can purchase is understanding the game. The **visible board**
   carries the rest — elevation is literally height, terrain is literally what it looks like.
   This is a constraint on every future addition: **prefer expressing a mechanic as a buyable
   stat over inventing a special rule.** Corollaries already honored: one unified 3D distance
   formula instead of special-case line-of-sight occlusion (§6); no hidden modifiers or
   nonlinear costs; deterministic combat.

**Milestone-1 success criterion:** in playtest, several non-obvious builds (e.g. a 0-damage
spotter, an immobile bunker, a 1-HP peak sniper, a swarm of 1-point gnats) should each prove
*situationally* strong. If they are all dead weight, the config knobs (§14) are wrong.

**Everything is provisional — balance and fun win.** Every number, and where necessary every
rule, is subject to change in service of balance and fun. This is *why* the architecture pushes
all values into `GameConfig` (§11/§14) and hides turn structure behind a swappable `ITurnPolicy`
(§3): the design is built to be tuned and iterated, not frozen. Treat this document as the
current best hypothesis, not a contract.

> All numeric values in this spec are **placeholders flagged as tunable** and live in
> `GameConfig` (§11). They exist to make the design concrete, not to lock balance.

---

## 3. Core loop & turn structure

A floating hex platform. Two players alternate turns. On a turn a player chooses from this
action menu:

> **Create** a unit · **Deploy** a unit · **Deploy a generator** · **Move / Use** units

- **Currency:** a single resource, **points**. Spent to create units and deploy generators.
- **Income:** points earned each turn from **generators** (§8) and from **bounties** on kills.
- **Win condition:** annihilate the enemy (§10).

### Turn policy (configurable — A/B target)

How much happens in one turn is **not finalized**; it is implemented behind an `ITurnPolicy`
so both modes are playtestable without touching the rules:

- **`AllUnitsPolicy` (default for M1):** in any order during your turn you may create units you
  can afford, deploy reserved units / generators, and move and/or attack with **each on-board
  unit at most once** (move-then-attack allowed), then End Turn. Stays snappy as armies grow.
- **`OneActionPolicy`:** a turn is a single atomic action (one create, OR one deploy, OR one
  unit's move+attack, OR one generator deploy), then auto-pass. Chess-like and deliberate.
  Exact granularity is tunable.

Income is credited at the **start** of the active player's turn.

---

## 4. Units — pure point-buy from zero

A unit *is* the stats you bought. Everything starts at **0**; **1 point = 1 step**. A unit
must have **≥1 Health** to exist (the only floor).

| Stat | Effect |
|---|---|
| **Health** | Current/max HP. Unit is destroyed at 0. |
| **Damage** | Base damage per attack, reduced by the target's Defense (+terrain). |
| **Range** | How far it can attack — measured in 3D distance (§6). |
| **Movement** | Movement budget per turn; climbing and terrain cost extra (§6). |
| **Defense** | Flat reduction of incoming damage per hit. |
| **Vision** | How far it can *detect and therefore target*, in 3D. No free baseline. |

- **Create** (action): pay the unit's total point cost (sum of stats); the designed unit goes
  into your **reserve** (off-board).
- **Deploy** (separate action): place a reserved unit on an empty hex in your deployment zone.
  Deploy costs **no points** (the cost was paid at Create); it consumes a deploy action.

Create and Deploy are intentionally decoupled: the *economic* decision (what to build) is
separate from the *board-timing* decision (when/where to commit it).

---

## 5. Combat (deterministic — no randomness)

On your turn, choose one of your units, then choose an enemy that is **both in Range and in
Vision** (§6). Resolve:

```
D = distance3D(attacker, target)                       // see §6
H = max(0, attacker.elevation - target.elevation)      // high-ground advantage
inRange   = D <= attacker.Range + H * rangeHighGroundBonus
visible   = D + target.terrainConcealment <= attacker.Vision
damage    = max(damageFloor, attacker.Damage + H * dmgHighGroundBonus
                              - (target.Defense + target.terrainDefense))
```

Apply `damage` to the target's Health. At 0 HP the target is destroyed and the attacker's
owner collects a **bounty** = a configured portion of the destroyed unit's build cost (§8).
The defender retaliates on **their** turn (no automatic return fire). Same inputs always
produce the same result.

An attack may target an enemy **unit or generator** (generators have Health and are valid
targets — §8); the same range/vision/damage rules apply.

---

## 6. Elevation & terrain

Every hex carries **both** an integer `Elevation` **and** a `TerrainType`. They are
orthogonal: elevation is height; terrain is surface.

### Unified 3D distance & visibility

All reach is expressed through one readable metric — no special-case occlusion code:

```
distance3D(a, b) = hexDistance(a, b) + |a.elevation - b.elevation|
```

Range, Vision, high-ground bonuses, and terrain concealment are all just terms layered on
this single distance (see §5). A unit in a valley must invest more Range/Vision to reach a
peak; a unit on a peak naturally out-reaches the valley.

### Elevation effects (all four the design calls for)

1. **3D distance** — elevation difference adds to distance for Range and Vision.
2. **High-ground bonus** — attacking downhill adds damage and effective range per level of
   advantage (`H` above); attacking uphill is penalized implicitly via added distance.
3. **Climb cost** — entering a higher hex costs extra movement per level; descending is cheap;
   a `maxClimbPerStep` caps un-scalable cliffs.
4. **Sight = the Vision stat**, not terrain occlusion — keeps terrain "what you see is what
   you get."

### Terrain types (data-driven modifier table)

Adding a terrain type is editing a `GameConfig` table, not writing code. Starter set:

| Terrain | Move cost | Concealment (vision) | Defense | Passable |
|---|---|---|---|---|
| **Plains** | 1 | 0 | 0 | yes |
| **Forest** | 2 | +2 | +1 | yes |
| **Rough** | 2 | +1 | +1 | yes |
| **Water** | 3 (or impassable) | 0 | 0 | config |

Terrain affects **movement, sight, and defense**. (Range is purely 3D-distance + elevation.)

### Movement

```
costToEnter(hex) = hex.terrainMoveCost + max(0, climbLevels) * climbCostPerLevel
```

Entry is forbidden if the hex is impassable, occupied, or `climbLevels > maxClimbPerStep`.
A unit spends up to its `Movement` per turn. One unit/structure per hex.

---

## 7. Vision & targeting

There is **no default vision** — it is bought like any stat. Vision determines what a unit can
**detect and therefore target** (§5 `visible`). Elevation difference and the target's terrain
concealment both eat into vision, so spotting a peak unit or a unit in forest costs more
Vision investment.

**Milestone 1: full on-screen visibility (no fog of war)** so one person can comfortably play
both sides. Vision still gates *targeting* (you cannot attack what you cannot see) — it simply
does not hide the board during testing. Fog of war becomes a presentation-layer filter over
this same Vision computation in a later milestone.

---

## 8. Economy

- **Starting bank:** each player begins with a configured pool of points to bootstrap.
- **Generators:** on-board, **attackable** structures (cost ≈ 2 pts) that produce income each
  turn. They occupy a hex you control and have their own Health, so the enemy can raid your
  economy and you must defend it. Income each turn = sum of your living generators' output.
- **Bounties:** destroying an enemy unit pays a configured **portion of its build cost** —
  expensive units are juicy targets, and over-investing in one monolith is structurally risky.

---

## 9. Board

- **Composition:** the board is one shared hex prefab instanced over the engine's cell data.
  Engine-side a board is a set of cells, each with `{coordinate, elevation, terrainType}` plus
  per-player **deployment zones**.
- **Elevation = literal stacking:** a hex at elevation *N* is a column of *N* of the same hex
  prefab; the top hex carries the terrain material, the stacked sides read as contour bands.
  (Engine stores elevation as a single integer; the column is purely how `BoardRenderer`
  builds it.)
- **Procedural generation (in M1):** an `IBoardGenerator` with a **seeded**
  `RandomBoardGenerator` is the default board source — deterministic from a seed, so boards are
  reproducible, unit-testable, shareable, and network-syncable later. "New board on the fly" =
  new seed. The generator produces elevation (simple noise → ridges/valleys), terrain
  (weighted thresholds), **mirrored/symmetric deployment zones** for fairness, and passes a
  connectivity sanity check (no one boxed in).
- **Authored boards** remain supported via an alternate `IBoardGenerator`; both emit identical
  `Board` data and render through the same path.
- "Basic but fair" generation now; biome/balance sophistication is a future spec.

---

## 10. Win, elimination & stalemate

- **Win:** annihilate the opponent.
- **Elimination:** a player loses when they have **no units on the board, none in reserve, and
  cannot afford to build a new one** (banked points below the cheapest viable unit). This
  avoids a false win when a player's board is empty but they are still rich. Evaluated after
  each command/turn, after initial deployment has occurred.
- **Stalemate backstop:** a configured **round cap**; if reached, the winner is whoever has the
  higher **total value** (on-board + reserve units + generators + banked points).

---

## 11. Architecture — headless engine + thin Unity skin

### `HexWars.Engine` — plain C# assembly, **zero `UnityEngine` references**, fully unit-testable

- **Data:** `HexCoord` (axial/cube + `Elevation`), `TerrainType`, `Board` (cells + deployment
  zones), `UnitStats` (Health/Damage/Range/Movement/Defense/Vision), `Unit` (stats + owner +
  position + currentHP), `Generator`, `PlayerState` (points, reserve, on-board units),
  `GameState` (board, players, activePlayer, round).
- **Rules (pure functions):** `Distance3D`, pathing with climb + terrain cost and
  `maxClimbPerStep`, `TargetingService` (in Range *and* Vision, 3D), `CombatResolver`
  (formula in §5), `Economy` (income, bounty, costs), `LegalMoves`, `WinCheck`.
- **Commands:** one record per action — `CreateUnit`, `DeployUnit`, `DeployGenerator`,
  `MoveUnit`, `AttackUnit`, `EndTurn`. A single `Apply(GameState, Command) -> Result(events |
  rejection)` is the **only** mutation path. *This command type is the future network
  wire-format and the AI's output — present from day one.*
- **`ITurnPolicy`:** decides which commands are legal this turn (`AllUnitsPolicy` vs
  `OneActionPolicy`) — the §3 A/B toggle, swappable without touching rules.
- **`GameConfig` / ruleset object:** every tunable number (§14). **Balancing is data, not
  code.**

### `HexWars.Engine.Tests` — Unity Test Framework (edit-mode)

TDD the math: 3D distance, climb/terrain movement cost, targeting (range+vision+concealment),
combat (incl. high-ground & terrain defense), economy (income/bounty/costs), win/elimination,
command validation & rejection, and "generator always yields a valid, symmetric, connected
board."

### Unity presentation layer (MonoBehaviours — the "skin")

- `BoardRenderer` (instances the shared hex prefab into columns by elevation, materials by
  terrain), `UnitView` / `GeneratorView`, `CameraRig` (angled orbit/pan so height reads),
  `InputController` (select → issue `Command`s), `HUD` (points, selected-unit stat panel, the
  **create-unit point-allocation panel**, turn banner, End Turn), and a `GamePresenter` that
  owns the `GameState`, pushes `Command`s in, and renders engine events out.
- **The create-unit panel is the de facto tutorial** (per design goal §2.4): each stat shows
  its 1:1 cost and plain-language effect, and the running point total + a preview of the
  resulting reach (range/vision) make the rules legible purely by buying. No separate manual.
- **Hotseat:** both players share input; a turn banner / pass-device prompt indicates the
  active player.

### How the roadmap plugs in (no rework)

- **AI:** a module that reads `GameState` and emits `Command`s. No UI needed.
- **Networking:** sync `Command`s (or seeds + commands) via the Multiplayer Center package; the
  command model is already the wire format.
- **Fog of war:** a presentation filter over the existing Vision computation.
- **Procedural variety / authored maps:** more `IBoardGenerator`s producing the same `Board`.

---

## 12. Visual direction (M1)

- **Skybox:** starfield — the platform floats in deep space.
- **Hexes:** flat **yellow** fill, crisp **black outlines**, low wide prisms; elevation = a
  stacked column of the prefab. Terrain swaps the top hex's material/tint; black outline shared.
- **Units / generators:** must pop against yellow and each other — **Player 1 = cyan/blue,
  Player 2 = red/magenta**; generators a distinct shape (e.g. glowing pylon), not color alone.
  Black outlines to match the board's graphic style over the starfield.
- **Pipeline:** primitives + an unlit/outline shader. No art pipeline needed to start.

---

## 13. Milestone 1 scope

**In:**
- Full engine + rules (units, combat, elevation, terrain, vision, economy, turns, win check).
- Seeded `RandomBoardGenerator` (default) + an authored board option.
- Create/deploy units; deploy generators; move + attack with elevation/terrain/vision;
  bounties; income; win-by-annihilation.
- Hotseat 2-player, full on-screen visibility (no fog).
- Readable 3D presentation + create-unit point-allocation UI.
- Edit-mode engine tests.
- Default turn mode `AllUnitsPolicy`; `OneActionPolicy` available via `ITurnPolicy`.

**Out (future specs):** AI opponent · networked multiplayer · fog of war · advanced/biome
procedural generation · art & audio polish · save/load · campaign/meta.

---

## 14. Open & tunable parameters (`GameConfig`)

To finalize during implementation/playtest (suggested starting values in parentheses):

- Starting point bank (e.g. 10–20).
- Bounty rate — portion of build cost refunded on kill (e.g. 50%).
- Generator: cost (≈2), per-turn output (≈1), Health (≈3).
- Damage floor — minimum damage per hit (0 or 1).
- High-ground bonuses: `dmgHighGroundBonus`, `rangeHighGroundBonus` per level (e.g. +1 each).
- Climb: `climbCostPerLevel` (e.g. +1), `maxClimbPerStep` (e.g. 2).
- Terrain table values (move cost / concealment / defense / passable) per §6.
- Board: dimensions (≈9×9), elevation range (0–4), terrain weights, deployment-zone depth.
- Round cap for the stalemate backstop.
- Exact `OneActionPolicy` granularity.
- Vision as a single 3D stat vs split horizontal/vertical (default: single 3D).

---

## 15. Future roadmap (each its own spec → plan → build cycle)

1. **AI opponent** — command-emitting module over `GameState`.
2. **Networked multiplayer** — Multiplayer Center; sync seeds + commands.
3. **Fog of war** — vision-driven presentation filter (+ optional in hotseat).
4. **Advanced procedural generation** — biomes, balance-aware layouts.
5. **Polish** — art, audio, UX, save/load.
