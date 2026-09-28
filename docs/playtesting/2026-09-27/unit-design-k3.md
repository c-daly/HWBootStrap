# Unit-design follow-up at browser pace K=3 — 2026-09-27

**The HP 3 / range 5 Sniper advantage survived the pace correction:** it won 11 of 12 equal-budget paired games at K=3, just as it did in the separate K=0 test. This supports a focused human balance experiment; it does not prove the custom design dominates expert play or every army composition.

## Scope correction and provenance

The original `unit-design` report tested the selectable K=0 whole-army turn mode from `GameSetup.Default`. The browser SetupForm instead initializes `_turnActions=3` at `Assets/HexWars/Presentation/SetupForm.cs:35` and randomizes its opening seed. This follow-up explicitly sets `turnActions:3` while repeating the same six fixed map seeds `[1,2,3,7,11,19]`. These are controlled seeds, not a sample of randomly chosen human matches.

Source is release main `1cd5f69`. Engine DLL SHA256: `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`. Maps remain 9x7 with elevation, biome modifiers disabled, damage floor 1, fog off, zero starting cash. No production rules changed. Planned biome and generator modes are outside this test; nothing here supports removing those systems.

The twelve manual actions in the earlier report are explicitly **constructed K=0 encounters**. They were not rerun here. In K=3, moving and then firing is possible only if the move does not exhaust the combat-action budget. Firing still ends that unit's movement.

## Frozen repeated suite

`criteria.md` was written before execution. All 42 games terminated by annihilation within 9 rounds; none hit the 40-round test limit. Every issued match command was accepted by `GameEngine.Apply`.

| Comparison | K=0 result, retained original | K=3 result |
|---|---|---|
| HP 3 / range 5 Snipers versus stock HP 2 / range 6 Snipers | Variant won 11/12 | Variant won 11/12 |
| Balanced fighters versus stock Strikers | Balanced won 8/12 | Balanced won 11/12 |
| Six cheap fighters versus three stock Strikers | Cheap won 1/12 | Cheap won 2/12 |
| Ordinary starter trio, six games | Seat 0 won 3/6 | Seat 0 won 5/6 |

The Sniper comparison uses three units per side, 60 points versus 60. The variant played all six seeds in each seat: **6/6 wins in seat 0 and 5/6 wins in seat 1** at K=3. Its K=0 result was 5/6 and 6/6 respectively. Do not pool the two pace settings as independent statistical evidence: they reuse maps and nearly identical controllers.

Synthetic armies were installed directly in starting slots with empty barracks. They were not bought through legal deployment commands, and their earned bounties were not spent. Every subsequent move and attack was legal. The six ordinary starter games kept real default barracks and legal bounty-funded purchases. The paired fighter tests also used equal 60-point budgets; complete stat lines are in `designs.json`.

## Controller limits matter more at K=3

The same heuristic still processes longer-range units first, favors a kill or immediate damage, mildly penalizes currently exposed positions, and otherwise approaches enemies. Adaptations were restricted to the active rule: stop after automatic handoff, never issue an old player's EndTurn, and do not score a final-action move as granting an immediate shot. `controller-k0-to-k3.diff` shows those changes and output metadata additions.

This is **not a K=3-optimized controller**. Its fixed ordering can spend two actions on a Sniper and the last on a Striker's move, leaving the Striker exposed with no immediate attack. It does not globally compare alternative sequences across all units or predict opponent move-and-shoot replies. Consequently, the ordinary-game finding that 10 of 12 starting Strikers died before firing is a usability and tactical-planning warning, not a clean measure of the Striker's intrinsic strength. Their two total attacks dealt only 4 effective damage. Seat 0's five wins are likewise too confounded and too few for a first-player balance conclusion.

The isolated Sniper comparisons avoid mixed-role ordering, since each side has three identical units, but deterministic unit ordering and myopic target choice still matter. This is a bounded design signal, not an optimal-play claim.

## Representative K=3 replay: health changes the exchange

In `sniper-health-7-base`, stock Snipers are seat 0 and HP 3 variants are seat 1:

1. Each army spends its first turn on three moves. The third move triggers automatic handoff; unlike K=0, the last unit cannot then fire.
2. Round 2: stock Sniper 1 moves to `(1,2)` and shoots variant 6 for 2 damage. The target survives on 1 HP. Stock Sniper 2 then moves to `(4,0)`, consuming the final action and passing the turn.
3. The variant army kills stock Sniper 2 with a single 2-damage hit, then uses its final action to reposition another unit. This is the one-hit versus two-hit health threshold in live play.
4. Round 3: the stock side uses elevation to deal 3 damage and kill one HP 3 variant outright, then finishes the previously injured variant. **Counterplay exists:** high ground can cancel the extra-health protection.
5. The remaining variant kills a stock Sniper. Round 4 becomes a final exchange: the last stock Sniper fires for 2, leaving the variant on 1 HP; the variant returns 2 damage and wins.

With the armies' seats swapped on the same seed, the variant also wins. That replay again includes a stock Sniper landing a lethal 3-damage elevation shot against HP 3; the advantage is situational, not invulnerability. Both complete traces are in `representative-replays.json`, with automatic handoffs explicitly marked.

## What this supports

- Prioritize a small human test of the third-health-point breakpoint. The equal-cost HP 3 / range 5 Sniper is a concrete candidate whose result survived both turn modes. It may lose value in larger spotter-supported formations where the extra range matters more.
- Test the Striker with a controller or human who deliberately reserves actions for its approach-and-fire sequence before changing its damage. Reducing its damage from 6 to 5 crosses a harmful breakpoint against the HP 7 / defense 2 Brute: two shots become three on level ground.
- Show the player when a proposed move consumes the final action, and whether the intended follow-up shot will therefore wait until the next turn. Combine that with lethal enemy-response cues.
- Do not infer cheap-unit dominance: the particular six-unit swarm lost 10/12 even at equal cost. Post-kill purchases, static blockers, generator play and future biome modes are separate questions.

This follow-up strengthens a specific health-breakpoint hypothesis. It does not justify a broad stat-pricing rewrite or removal of planned modes.
