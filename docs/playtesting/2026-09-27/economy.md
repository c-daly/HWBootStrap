# Economy and endgame playtest — release main 1cd5f69

Tested the actual engine DLL built from release main, SHA256 `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`, through legal `GameEngine.Apply` commands. All work is isolated in `/tmp/hexwars-balance-20260927/economy`; no game rules, repository source, training or remotes changed.

## Highest priority: an army-less player can deny a win by refusing reinforcements

**Ordinary public default; no custom setup or injected state required.** A player with zero units but enough saved points for any barracks template is never eliminated. They can keep passing until the default round-100 backstop awards a draw. The winning player has nothing to attack and no command that forces the opponent to deploy. This creates an explicit incentive to decline a losing comeback, rather than play it.

Concrete default-seed7 reproduction: `minimal-stall-repro.txt` contains25 legal commands from fresh `GameSetup.Default`. In round 4 Player 0 destroys Player 1's last unit. Player 0 still has2 units and11 points; Player 1 has0 units,20 points and only the 5 stock templates. `stockseed7-bank1.jsonl` then records passes through round 100 and a draw. This is96 rounds after apparent victory.

Counterplay check: at the empty-army position, Player 1 has 84 legal deployments and EndTurn. Player 0 has 16 moves, 2 captures and EndTurn, with no attack target. Replaying the same position and choosing a normal stock Artillery reinforcement results in Player 0 winning in round 5 after 5 more commands (`counterplay-results.json`). Thus the reserve supports real comeback play, but declining it grants a better result: draw instead of loss.

Bounded result:8/16 matches using stock catalogs plus deliberately saved bounty ended this way, across seeds1–8 with the bank strategy swapped between seats. Creating a free1-point inert template before fighting expands the exposure to11/16, since even one saved point prevents elimination. Baseline stock-v-stock controller finished all8 matches in rounds2–4 with0 draws. These are exploit-exposure counts for a deterministic test controller, **not human draw-rate estimates or balanced-agent win probabilities**.

Root rule: `engine/HexWars.Engine/WinCheck.cs:49–59` checks affordability, but has no requirement or deadline to redeploy. `WinCheck.cs:74–75,101–108` ends round 100 as a draw unless Score is enabled. The template can be entirely inert; the in-game designer begins at1 Health and0 other stats (`Assets/HexWars/Presentation/DesignPanel.cs:38–40`).

**Smallest design change to test:** retain one own-turn opportunity to redeploy after losing the last unit, then lose when ending that turn with no army. Check actual legal deployment availability too. A score tiebreak alone does not cure the96-round wait. Do not remove comeback reserves outright without deciding the intended comeback experience.

## Bounty and reinforcement tempo: worth testing, not proven overpowered

Stock armies cost61 points per side: Brute21, Striker20, Sniper20. A kill pays10 points; cheapest stock combat reinforcement is19-point Artillery, so two kills unlock an extra attacker. The reinforcement may attack on the same turn it arrives. The player already ahead in kills can turn that advantage into additional damage before the opponent responds.

In the8 baseline games, the first killer won7; the first reinforcement appeared in rounds1–3. This small set and my greedy controller cannot separate bounty amplification from positioning or turn order. The seed7 trace demonstrates the mechanism directly: Player 0's second kill grants20 points, `D 0 3 2 2` spends19 on Artillery, and the very next command `A 0 7 4` deals4 damage before the handoff. Player 1 later has the same opportunity and wins the ordinary baseline. Therefore this is **a tempo lever to compare**, not evidence that reinforcement itself should be removed.

Suggested smallest controlled experiment: compare current rules with new units unable to fire until their next turn, holding map, initial placement and strategy fixed. A separate frozen follow-up tested a10-point Raider (HP 1, damage 3, move 1, climb 1, range 2, arc 1, vision 1, visionArc 0) using the same controller against stock catalogs in16 seed/seatpairs. It won9/16, with0 draws; only1/16 results changed relative to the same-seed stock baseline (seed1, Player 1). This establishes an earlier reinforcement option but does not establish cheap-unit dominance. Avoid changing bounty rate and deployment readiness together in the first experiment.

Sources: `CombatResolver.cs:27–28`, `BarracksCatalog.cs:13–17`, `GameEngine.cs:196–227` and the immediate successful post-deploy attack in the trace.

## Separate lower-priority rule/API mismatch: hidden generators in Annihilation

**Raw-command reachability, not a visible ordinary Annihilation button.** Fresh default seed7, legal move/combat opener, Player 1 kills the Striker for10 points. Sending legacy command `N 1 <empty deployment hex>` then succeeds, costs2points, and creates a generator. After one turn cycle Player 1 has9 points (10−2+1income). Exact commands and state receipts are in `counterplay-results.json`, kind `default-bounty-raw-wire-generator`. No custom points or state were injected.

The Annihilation UI hides territory build controls (`UnitInputController.cs:712`), and `LegalMoves.cs:44` says generators were removed. Yet `DeployGenerator` remains accepted without checking a mode/GeneratorsEnabled restriction (`GameEngine.cs:176–193`). This means an alternate client can access economy behavior stock players cannot discover. Guard or retire the legacy command consistently if Annihilation is intended to be bounty-only. This was an engine-command proof; I did not submit it to a live server.

## Territory generators: correct an initial hypothesis

This part used explicit custom public options: Territory,40starting points,9x7, seeds7 and19. I built a4-point generator, moved the builder away, and directed an opposing unit to raid it. The seed19 attacker reached the generator in round3, attacks against the generator ID were rejected (`TargetNotEnemy`), but capture was legal at the opening of the next turn and transferred the generator to Player 1 for4points. Therefore **generators are capturable; do not report them as invulnerable/impassable**. They cannot be shot because the attack handler resolves unit IDs only (`GameEngine.cs:279–281`). The Generator type still describes an attackable structure, so the rule/presentation intent needs clarification before calling that a bug.

The seed7 approach failed to arrive within its50-command cap because the controller greedily refused an equal-distance detour around defending units. That is a controller limitation, not evidence that the generator cannot be reached. Retained in `counterplay-results.json`.

## Methods and receipts

- `plan.json`, `followup-plan-stock-bank.json`, `counterplay-plan.json` freeze seeds, setup, action caps and controller choices before each batch.
- 56 complete legal engine matches: 8 baseline; 16 with inert-template banking; 16 with only stock catalogs and banking; 16 with the 10-point Raider against stock. No injected-state games.
-1500-command cap, actual public round 100 rule. All 56 reached an engine terminal result, and all 56 full traces were replayed from their starting GameSetup with every command accepted (`counterplay.log`).
- `results-v1.json`, `results-stock-bank.json`, `results-cheap-reinforce.json`, `summary.json`, `*.jsonl` retain individual outcomes and full legal wire traces; `counterplay-results.json` retains branches and rejection receipts.
- `Program-v1.txt` is the first batch controller, `Program-v2.txt` the stock-only follow-up, `Program-cheap-reinforce.txt` the cheap-design follow-up; current `Program.cs` runs counterplay and replay validation. `compile-failure.log` preserves the initial harness-only compile error (used Health instead of CurrentHp), fixed before any game ran.
-Controller attacks available targets, prioritizing kills/damage; then pursues a move-to-shot/closer position; stock spender buys stock combat designs; bank mode deliberately saves money and refuses deployments. It does not model full opponent replies, optimize opening placement, or exhaustively plan pathfinding detours. The draw-stall reproduction is structural and does not depend on controller optimality; broader tactical win rates do.


## Running the receipt harness

From this folder, `dotnet run --project Economy.csproj` runs the counterplay branches and validates every included `*.jsonl` replay. It expects the engine DLL at `../engine/HexWars.Engine.dll` and uses net10.0. The receipts root defaults to the working directory and can be set with HEXWARS_BALANCE_RECEIPTS. Override the DLL path with `-p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll` if needed.

To reproduce each original batch without overwriting prior evidence, copy the folder to a new run directory, copy the appropriate archived Program-v1.txt / Program-v2.txt / Program-cheap-reinforce.txt to Program.cs, run from that new directory with HEXWARS_BALANCE_RECEIPTS unset, using the same command. Each corresponding plan.json records seeds and command caps. No external packages or repository writes are required.
