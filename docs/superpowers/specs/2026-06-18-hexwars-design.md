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

Two players (milestone 1: both human, hotseat on one screen). **AI players are first-class
citizens.** A challenging AI is a core goal, and beyond it the engine exposes a **basic API any
agent can use** — so you can play against your own bot, or RL-train against the engine, purely
by reading game state and submitting commands. Humans, the built-in AI, and external/third-party
agents are all just clients of that one API (§11). The smart AI and the external/RL transport
ship in a later milestone, but **milestone 1 builds the engine foundations that make them
cheap** (serialization, headless self-play, deterministic search-able state). Networked
multiplayer rides the same command API.

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
   stat over inventing a special rule.** Corollaries already honored: separate flat
   horizontal/vertical reach instead of special-case line-of-sight occlusion (§6); no hidden
   modifiers or nonlinear costs; deterministic combat.
5. **Uniform 3D pricing — every capability, both axes, 1 point per hex.** The board is 3D (hexes
   + elevation), so every spatial capability is bought on *two* axes: a horizontal reach and a
   vertical reach, each a flat 1 point per hex/level. This holds for **Movement** (+ **Vertical
   Movement**), **Range** (+ **Range Arc**), **Vision** (+ **Vision Arc**), and **any capability
   added later** — e.g. communication would be 1 point per hex, just like everything else. You
   can do only what you bought; no axis has a free baseline.

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
| **Defense** | Flat reduction of incoming damage per hit. |
| **Movement** | Horizontal traversal budget per turn — hexes entered, paying terrain move cost (§6). |
| **Vertical Movement** | Ascent budget per turn — levels it can climb (1 pt = 1 level up). Descending / level moves are free. 0 = can't leave its elevation band. |
| **Range** | Horizontal firing reach — max hex distance to a target (§5). |
| **Range Arc** | Vertical firing reach — levels *up* it can shoot. Firing at/below its own level is unrestricted. |
| **Vision** | Horizontal sight — max hex distance it can detect a target. Shared army-wide (§7). |
| **Vision Arc** | Vertical sight — levels *up* it can see. Seeing at/below its own level is unrestricted. |

All nine stats are 1 point per step; a unit's `PointCost` is their sum. Naming: the vertical
*reach* stats are "Arc" (instantaneous reach upward) while vertical *traversal* is "Vertical
Movement" — the wording itself flags that moving up is a cost, not a reach (§6).

**You can only do what you bought.** A stat at 0 means the capability is simply *absent* — there
is no implicit baseline for anything: `Movement` 0 can't move across, `Vertical Movement` 0 can't
climb, `Range` 0 can't attack across, `Range Arc` 0 can't fire above its own level, `Vision` 0 is
horizontally blind, `Vision Arc` 0 can't see anything higher, `Defense` 0 takes full damage. Every
capability, on every axis, is earned by spending points. (A **bounder** = high Vertical Movement,
low Movement; an **aircraft** = high in both; a **turret** = zero of both.)

- **Create** (action): pay the unit's total point cost (sum of stats); the designed unit goes
  into your **reserve** (off-board).
- **Deploy** (separate action): place a reserved unit on an empty hex in your deployment zone.
  Deploy costs **no points** (the cost was paid at Create); it consumes a deploy action.

Create and Deploy are intentionally decoupled: the *economic* decision (what to build) is
separate from the *board-timing* decision (when/where to commit it).

---

## 5. Combat (deterministic — no randomness)

On your turn, choose one of your units, then choose an enemy target your unit can **reach
(Range)** and that your **army can see (Vision)** — sight is shared army-wide (§7). Resolve:

```
hd(a,b) = hexDistance(a, b)                              // horizontal hex distance, ignores elevation
up(a,b) = max(0, b.elevation - a.elevation)             // how many levels b is ABOVE a
H       = max(0, attacker.elevation - target.elevation) // attacker's high-ground advantage

inRange = hd(attacker,target) <= attacker.Range + H*rangeHighGroundBonus  // horizontal reach
          AND up(attacker,target) <= attacker.RangeArc                    // vertical reach (firing up)
visible = ANY friendly living unit f:                                     // ARMY-WIDE sight
              hd(f,target) + target.terrainConcealment <= f.Vision        //   horizontal sight
              AND up(f,target) <= f.VisionArc                             //   vertical sight (looking up)
damage  = max(damageFloor, attacker.Damage + H*dmgHighGroundBonus
                           - (target.Defense + target.terrainDefense))
```

A target may be attacked when `inRange && visible`. Apply `damage` to the target's Health. At 0
HP the target is destroyed and the **owning player** collects a **bounty** = a configured
portion of the destroyed unit's build cost (§8) — so a kill your *spotter* enabled still pays
you in full. The defender retaliates on **their** turn (no automatic return fire). Same inputs
always produce the same result.

An attack may target an enemy **unit or generator** (generators have Health and are valid
targets — §8); the same range/vision/damage rules apply.

---

## 6. Elevation & terrain

Every hex carries **both** an integer `Elevation` **and** a `TerrainType`. They are
orthogonal: elevation is height; terrain is surface.

### Reach: separate horizontal and vertical axes

Reach is two independent, readable axes — no special-case occlusion code:

- **Horizontal** = hex distance on the plane (`hexDistance`, ignoring elevation), gated by
  `Range` (firing) and `Vision` (sight).
- **Vertical** = elevation difference. Firing/seeing *upward* is gated by `Range Arc` /
  `Vision Arc` (levels above you); at or below your own level is unrestricted — looking and
  shooting *down* is free, reinforcing the high-ground advantage.

So a valley unit must buy `Range Arc` / `Vision Arc` to engage a peak, while a peak unit looks
and fires downhill for free. Terrain concealment adds to the *horizontal* sight requirement.

### Elevation effects (all the design calls for)

1. **Vertical reach is its own buy** — firing/seeing *up* is gated by `Range Arc` / `Vision Arc`
   (levels above you); at or below your level is free (§5).
2. **High-ground bonus** — attacking downhill adds damage and horizontal range per level of
   advantage (`H` in §5). Holding the peak is real power.
3. **Vertical movement is its own buy** — climbing a higher hex spends `Vertical Movement`
   (1 per level up); descending / level moves are free. No hard cliff cap — your `Vertical
   Movement` budget *is* the limit (a bounder leaps a 5-stack; a turret can't step up one).
4. **Sight = the Vision / Vision Arc stats**, not terrain occlusion — keeps terrain "what you
   see is what you get."

### Terrain types (data-driven modifier table)

Adding a terrain type is editing a `GameConfig` table, not writing code. Starter set:

| Terrain | Move cost | Concealment (vision) | Defense | Passable |
|---|---|---|---|---|
| **Plains** | 1 | 0 | 0 | yes |
| **Forest** | 2 | +2 | +1 | yes |
| **Rough** | 2 | +1 | +1 | yes |
| **Water** | 3 (or impassable) | 0 | 0 | config |

Terrain affects **movement, sight, and defense**. (Range/Vision are horizontal hex distance;
vertical reach is the Arc stats.)

### Movement — two budgets (horizontal + vertical)

A unit has **two** per-turn budgets: **Movement** (horizontal) and **Vertical Movement**
(ascent). Entering an adjacent hex draws from both:

```
horizCost(hex)      = hex.terrainMoveCost                     // from Movement
vertCost(from, hex) = max(0, hex.elevation - from.elevation)  // from Vertical Movement (ascent only)
```

A step is allowed only if the hex is passable and unoccupied, AND remaining `Movement >=
horizCost` AND remaining `Vertical Movement >= vertCost`. Descending or moving level costs **0**
vertical. There is **no** hard cliff cap — the `Vertical Movement` budget is the only limit, so a
single step may climb several levels at once (hopping onto an adjacent **2-stack costs 2 Vertical
Movement** plus the hex's terrain cost). One unit/structure per hex.

This is deliberately *unlike* Range/Vision: shooting or seeing upward is instantaneous **reach**
(gated by the `Range Arc` / `Vision Arc` you bought), whereas moving upward is a **traversal cost**
(paid from `Vertical Movement`). Same uniform price — 1 point per level — different mechanic.

---

## 7. Vision & targeting

There is **no default vision** — it is bought like any stat, on two axes. **Sight is shared
across your entire army:** a target is *visible* (and therefore targetable, §5) if **any** of
your living units sees it — `hexDistance(spotter, target) + target.terrainConcealment ≤
spotter.Vision` (horizontal) **and** `target` is at most `spotter.VisionArc` levels above it
(seeing down is free). Range/Range Arc stay per-unit: a shooter must independently reach a
target the army can see. **Consequence:** a 0-damage, all-Vision **spotter** lifts the whole
army's reach and pays for itself through the kills it unlocks — a first-class build, not a
gimmick (the §2 success criterion).

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
  new seed. The generator mirrors the **entire board** — terrain *and* elevation — across center
  for fairness (not just the deployment zones), so neither side gets better ground. It produces
  elevation (simple noise → ridges/valleys), terrain (weighted thresholds), **mirror-symmetric
  deployment zones**, and passes a **passable-terrain connectivity check** (every deployment tile
  can reach the rest of the board over passable terrain — no zone walled off by impassable hexes).
  Elevation never traps a unit: any height is climbable given enough `Vertical Movement`, so
  reachability is a terrain-passability property, not an elevation one.
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
  zones), `UnitStats` (Health/Damage/Defense/Movement/VerticalMovement/Range/RangeArc/Vision/
  VisionArc), `Unit` (stats + owner + position + currentHP), `Generator`, `PlayerState` (points,
  reserve, on-board units), `GameState` (board, players, activePlayer, round).
- **Rules (pure functions):** `HexDistance`, pathing over two budgets (terrain cost from
  `Movement`, ascent cost from `VerticalMovement`), `TargetingService` (army-wide sight via
  `Vision`/`VisionArc` + per-unit `Range`/`RangeArc`), `CombatResolver` (formula in §5),
  `Economy` (income, bounty, costs),
  `LegalMoves`, `WinCheck`, and `Evaluate(GameState, player)` — a heuristic position score
  (reused by the §10 stalemate total-value calc and, later, as the AI's search heuristic).
- **Commands:** one record per action — `CreateUnit`, `DeployUnit`, `DeployGenerator`,
  `MoveUnit`, `AttackUnit`, `EndTurn`. A single `Apply(GameState, Command) -> Result(events |
  rejection)` is the **only** mutation path. **`Apply` is non-mutating** — it returns a *new*
  `GameState`, leaving the input untouched — and `GameState.Clone()` is cheap. Together with
  determinism this makes the engine **search-able**: an AI can fork the state, try a command,
  and score the result (minimax / MCTS) without side effects. *This command type is also the
  future network wire-format — present from day one.*
- **`ITurnPolicy`:** decides which commands are legal this turn (`AllUnitsPolicy` vs
  `OneActionPolicy`) — the §3 A/B toggle, swappable without touching rules.
- **`GameConfig` / ruleset object:** every tunable number (§14). **Balancing is data, not
  code.**

### `HexWars.Engine.Tests` — Unity Test Framework (edit-mode)

TDD the math: hex distance, two-budget movement (terrain + ascent), targeting (range+vision+concealment),
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

### Agent API — AI players are first-class (foundations in M1)

The engine *is* the game; the Unity UI, the future built-in AI, and any external/third-party
agent are all **clients of one basic API**. That API is the command model plus a small set of
primitives, all designed in milestone 1:

- **Drive:** read `GameState` (or a serialized snapshot), enumerate `LegalMoves`, submit a
  `Command` via the non-mutating `Apply`. Identical path for human, AI, and bot.
- **Headless `Match` runner + `IAgent`:** a pure-engine game loop that pits two `IAgent`s
  against each other with no Unity present. M1 ships `IAgent` and a trivial `RandomAgent`
  reference implementation (used in the integration test) to prove the API end-to-end.
- **RL-ready primitives:** `Reset` (new game from a seed) plus the `Apply` / `LegalMoves` /
  `WinCheck` / `Evaluate` quartet form a Gym-style loop — observation = serialized `GameState`,
  action space = `LegalMoves`, reward = derived from `Evaluate` / terminal, done = `WinCheck`.
- **Serializable `GameState` and `Command`s** (JSON) so an out-of-process agent — including a
  Python RL trainer — can exchange observations and actions over a simple transport.

Because `HexWars.Engine` is pure .NET (no Unity types), it runs fully headless for self-play and
training. Engine quality bars that keep this (and the §15 research-platform vision) viable:
**reproducible episodes** (a match is fully determined by `GameConfig` + seed + agent policies)
and **allocation-light `Apply`/`Clone`** so thousands of games simulate fast. The *built-in
challenging AI* and the *external/RL transport* (a local socket/JSON bridge or a Gym/PettingZoo
wrapper) are their own milestone (§15); M1 only lays these in-engine foundations.

### How the roadmap plugs in (no rework)

- **AI / external agents / RL:** all drive the engine through the Agent API above — no UI needed.
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
- **Agent-API foundations** (so AI players are first-class later, §11): non-mutating `Apply` +
  `GameState.Clone()`, `LegalMoves`, `Evaluate`, JSON-serializable `GameState`/`Command`, a
  headless `Match` runner + `IAgent` with a `RandomAgent` reference impl, and `Reset`/new-game.

**Out (future specs):** the *built-in challenging AI* itself and the *external/RL transport*
(socket/JSON bridge or Gym/PettingZoo wrapper) — M1 ships only the foundations above ·
networked multiplayer · fog of war · advanced/biome procedural generation · art & audio polish ·
save/load · campaign/meta · research-platform tooling.

---

## 14. Open & tunable parameters (`GameConfig`)

To finalize during implementation/playtest (suggested starting values in parentheses):

- Starting point bank (e.g. 10–20).
- Bounty rate — portion of build cost refunded on kill (e.g. 50%).
- Generator: cost (≈2), per-turn output (≈1), Health (≈3).
- Damage floor — minimum damage per hit (0 or 1).
- High-ground bonuses: `dmgHighGroundBonus`, `rangeHighGroundBonus` per level (e.g. +1 each).
- (Vertical Movement and the Arc reach stats cost a flat 1 point per level — uniform, not tunable.)
- Terrain table values (move cost / concealment / defense / passable) per §6.
- Board: dimensions (≈9×9), elevation range (0–4), terrain weights, deployment-zone depth.
- Round cap for the stalemate backstop.
- Exact `OneActionPolicy` granularity.
- `Range Arc` / `Vision Arc` magnitudes, and whether downhill firing/sight is ever capped
  (default: uncapped — at-or-below your level is free).
- Optional **intel-kickback** bonus for kills only your spotter could see (default: OFF — the
  shared economy already rewards recon; revisit only if pure spotters underperform in playtest).

---

## 15. Future roadmap (each its own spec → plan → build cycle)

1. **AI players & external/RL API (first-class)** — a challenging built-in AI (search over the
   deterministic engine + the `Evaluate` heuristic; difficulty via search depth / eval tuning),
   plus a *basic external API* so any agent can play or RL-train against the engine. Recommended
   transport: a language-agnostic local socket with line-delimited JSON, and/or a
   Gym/PettingZoo-style wrapper, built on the M1 serialization + `Match` foundations. Built-in
   AI, external bots, and humans all share the identical `Command` API.
2. **Research platform** — turn the agent API into a reusable multi-agent RL benchmark:
   versioned observation/action schemas, standard scenarios via `GameConfig`, batch/parallel
   self-play, metrics/logging, and reproducibility guarantees (config + seed + policies). A
   natural extension once the external API and a baseline AI exist; **not** required for play.
3. **Networked multiplayer** — Multiplayer Center; sync seeds + commands.
4. **Fog of war** — vision-driven presentation filter (+ optional in hotseat).
5. **Advanced procedural generation** — biomes, balance-aware layouts.
6. **Polish** — art, audio, UX, save/load.
