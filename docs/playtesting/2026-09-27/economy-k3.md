# Economy and endgame follow-up: browser turn setting K3

The browser setup defaults to **3 acting units per turn**, not the whole-army `GameSetup.Default` preset used in the earlier 56-match batch. This separate, frozen 24-match follow-up uses K3, the browser's other initial settings, and fixed seeds 1–8. The real setup form randomizes its seed. Standard engine formation is used; manual placement is off, matching the initial form option. Sources: `Assets/HexWars/Presentation/SetupForm.cs:32–38,58–62,294–295`.

All games use the unchanged release engine from main `1cd5f6933702853ef1ed83a4f9f7f865175a5954`, DLL SHA256 `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`. No production changes, training, GUI automation, or live multiplayer commands were performed.

## High-priority finding confirmed at K3: banked reserves can deny annihilation

A player with no surviving units remains alive as long as any stock barracks template is affordable. They can repeatedly pass instead of deploying, keeping all their points and forcing a round-100 draw. This preserves an optional comeback but also rewards refusing a losing comeback.

**Minimal legal reproduction, seed 7:** load setup `0 9 7 0 7 3 1 1 1 3 0`, then apply the 16 commands in `minimal-stall-k3-seed7-bank0.txt`. Player 0 loses its last unit in round 3 but has 20 points. Player 1 has two units, 10 total HP, and 11 points. Player 0 can deploy but need not do so. Passing thereafter produces a draw at round 100 — 97 rounds after the army disappeared. The full trace is `k3-seed7-bank0.jsonl`.

**Counterplay:** at exactly this state, Player 0 has 84 legal deployment choices plus EndTurn. Player 1 has movement, two captures and EndTurn, but no attack target. Resuming stock reinforcement gives `E 1`, `D 0 3 2 2`, `E 0`, `M 1 4 3 2`, `A 1 7 8`: Player 1 wins in round 4. Thus voluntarily playing out a comeback loses, while refusing it guarantees a draw. A second branch, seed 3 with Player 1 banking, likewise loses after 5 further commands when choosing a stock reinforcement, versus drawing by passing. Exact snapshots and continuations are in `counterplay-results.json`.

| Frozen K3 batch | Games | Player 0 wins | Player 1 wins | Draws | Empty-army stalls |
|---|---:|---:|---:|---:|---:|
| Stock spender vs stock spender | 8 | 2 | 5 | 1 | 0 |
| Player 0 deliberately banks bounty | 8 | 1 | 2 | 5 | 5 |
| Player 1 deliberately banks bounty | 8 | 1 | 4 | 3 | 3 |

The **8/16 bank-strategy stalls** establish exposure under these fixed controllers; they are not estimated human draw rates or a general side-bias measurement. The rule reproduction itself does not depend on agent strength. Both bank controllers use only stock templates and start with zero points.

The root rule is `engine/HexWars.Engine/WinCheck.cs:49–59`: affordability prevents elimination indefinitely. At round 100, Annihilation without Score ends as a draw (`WinCheck.cs:74–75,101–108`).

**Smallest rule experiment:** allow a player who loses their last unit one own-turn opportunity to redeploy; ending that turn without an army then loses. Also distinguish an affordable design from an actually legal deployment location. This preserves comeback play while closing the refusal incentive. A round-cap score tiebreak alone does not fix 97 rounds of empty-board waiting. No tuning or code changes have been applied.

## Baseline draw is a different limitation

Seed 7 in stock-v-stock ends with one immobile Artillery on each side, no more affordable stock units, and no attacks taken after the fight stalls. The controller does not invent a new cheaper mobile design and blindly passes. `counterplay-results.json` records the two surviving pieces. This is **not** an empty-army refusal reproduction, and the single result does not prove an unavoidable game stalemate: the designer offers counterplay the fixed stock-only controller never tries. Keep this result in the baseline rather than silently dropping it.

## Scope and feature intent

The earlier `economy` directory contains 56 games in **selectable whole-army K0**, not the browser's initial K3 setting. Its raw evidence is retained and its report is corrected. The 10-point-design and immediate-fire observations from that batch remain K0-only evidence; this K3 batch did not retest those design matchups.

The user intends future modes to use generators and biomes. Preserve those systems. The earlier generator observation is a lower-priority **per-mode command/visibility consistency** issue: a mode should expose the generator actions it allows, and reject only actions disabled for that mode. It is not a recommendation to remove generators or biomes. No additional generator tests were run in this K3 batch.

## Harness and indispensable receipts

- `plan.json` and `counterplay-plan.json`: setup, seed bank, controller and caps frozen before the corresponding runs.
- `Program-batch.txt`: exact batch controller; `Program.cs`: counterplay and replay validation; `Economy.csproj`: isolated net10 project, no external packages.
- `results-k3.json`, `summary.json`: all 24 outcomes and bounded aggregates.
- `minimal-stall-k3-seed7-bank0.txt`, `k3-seed7-bank0.jsonl`: short reproduction and full draw trace.
- `minimal-stall-k3-seed3-bank1.txt`, `counterplay-results.json`: second seat reproduction and both continuation branches.
- `counterplay.log`: all 24 full game traces replayed from their starting GameSetup, every command accepted and every game terminal. No action-cap timeouts.

Run from this receipt directory with `dotnet run --project Economy.csproj -p:HexWarsEngineDll=/absolute/path/to/HexWars.Engine.dll`. The receipt root defaults to the working directory; override it with `HEXWARS_BALANCE_RECEIPTS` if necessary. To reproduce the batch without overwriting these receipts, copy the harness to a new directory, use `Program-batch.txt` as Program.cs, and run there.

The controller chooses immediate kills/damage, moves toward a shot or nearer enemy, and buys stock combat templates when affordable; bank mode instead saves its bounty. It does not search full replies or optimize manual placement. All 24 games are actual engine matches from public configurations, not injected-state fixtures, but this is headless scripted playtesting rather than human UX observation.
