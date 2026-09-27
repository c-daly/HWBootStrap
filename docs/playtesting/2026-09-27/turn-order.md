# Turn order, movement, and pace playtest — 2026-09-27

The strongest finding is a reproducible **odd-width board symmetry defect**, followed by a **whole-army reinforcement chain that can erase the opposing army before its next turn**. This study does **not** establish a universal first- or second-player advantage. Controller choice reverses the apparent initiative result.

## Scope and evidence

- Current release source: main `1cd5f6933702853ef1ed83a4f9f7f865175a5954`.
- Fresh engine DLL supplied by parent: SHA256 `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`.
- Public `GameFactory.Build(GameSetup.Default)` rules: annihilation, 9x7, zero starting bank, three standard roles per side, no biome modifiers, damage floor1, max elevation2, no fog, whole-army turns. Board seed varies only as declared below. No custom designs, training, or production changes.
- Frozen protocol in `protocol.md`. Batch: seeds1–24, five controller matchups =120 current-rule matches. Separately: seeds1–12, three matchups, KActions2 =36 counterfactual matches. All156 reached an engine terminal state; zero illegal commands and zero external command-cap stops. 16,884 accepted batch commands recorded.
- **One additional match was chosen command-by-command by the playtesting subagent**, using legal-move/target inspection and a stated high-ground plan against the existing Greedy opponent. It completed in22 commands, round3. This is distinct from the156 bot simulations. Files `manual-vs-greedy.wire` and `.jsonl` contain the complete game.
- I inspected every command of the default-seed Greedy game and manual game, the default-seed tactical stalemate, and the seed21 reinforcement wipe. No GUI/audio/player-experience claims derive from these engine-only tests.

## 1. Fix board symmetry before interpreting balance percentages — high confidence, concrete defect

`RandomBoardGenerator.cs:36` mirrors the rectangular **offset** coordinates using `(width-1-col,height-1-row)`. On odd-width flat-top hex layouts this mapping does not preserve adjacency.

Exact 9x7 witness, in axial coordinates:

- `(1,0)` and `(2,0)` are adjacent: distance1.
- Their generated counterparts are `(7,3)` and `(6,2)`: distance2.

Every tested 9x7 map has96 directed neighbor relations that fail under the alleged mirror (48 undirected edges). Both unit formations and tile elevations pass the code's offset-mirror check, so simply checking those would miss the defect.

Opening reachable-cell totals, summed across the three units before any action:

| Width | Seeds | Seeds with unequal reach counts | P0 total | P1 total | Broken directed adjacencies per seed |
|---|---:|---:|---:|---:|---:|
| 8, exploratory control |24|0|818|818|0|
| **9, public default** |24|**21**|804|862|**96**|
| 10, exploratory control |24|0|830|830|0|

Default seed7 has27 options for P0 and31 for P1. P1 has more opening options on19 seeds; P0 has more on2;3 tie. Seeds4 and8 even have unequal corresponding cross-army pairwise distances because one role begins in a second column. These counts prove the symmetry defect; they do not measure its causal effect on game win rate.

**Suggested first change:** use a genuine hex-grid reflection/rotation for terrain and formation placement, or use an even-width default as an interim measure. Add an adjacency-preservation control plus mirrored reachable destinations, shot legality, and path-cost checks. Fixing only the spawn ordering will not fix terrain geometry. Keep these checks independent of controller performance.

Receipts: `symmetry.csv`, `symmetry-controls.txt`, `probes/Program.cs` (`symmetry`).

## 2. Whole-army + immediate bounty + fresh-unit activation creates large response gaps — high confidence example; incidence is controller-bound

Current rules let each existing unit move/fire, spend kill bounty immediately, and move/fire the purchased unit before passing. Two starting shooters cost20 each and have2HP. Any ordinary Brute/Sniper shot on level ground kills either shooter; every starter can kill one if it can legally target it. This makes one exposure a full lost activation plus bounty income.

In current-rule seed21 Greedy/Greedy, P0's round2 turn deletes all three opponents before P1's second turn:

1. Brute takes height2 and kills the enemy Sniper:10 bounty.
2. Existing Sniper kills the enemy Striker: bank20.
3. P0 buys a new Sniper for20, moves it, and fires at the remaining Brute for2.
4. P0's original Striker deals the remaining5, winning immediately.

A replay control omitting only the new Sniper's deployment/move/attack leaves that Brute alive at2HP. Thus the fresh unit's same-turn shot is necessary for this particular complete wipe; the receipt is not just a correlation.

Exact complete seed21 command sequence (`alpha-seed21.wire`, all engine-validated):

```text
M 0 1 3 -1
M 0 2 3 4
M 0 3 2 4
E 0
M 1 5 5 -2
M 1 4 5 4
M 1 6 6 -2
E 1
M 0 1 4 -1
A 0 1 6
A 0 3 5
D 0 2 2 5
M 0 7 4 4
M 0 2 5 3
A 0 7 4
A 0 2 4
```

Across120 current-rule bot games,58 included at least two kills in one uninterrupted turn; the maximum was3. The first killer won63/75 decisive games (84%). These are descriptive, controller-conditioned figures: skill, position, and controller mistakes confound first-kill correlation; it is not an estimate of the independent causal effect of bounty.

**Small first experiment:** fresh reinforcements can be placed immediately but cannot attack until their owner's next turn. Compare that single rule against delaying bounty to next turn. Do not simultaneously change unit stats, initiative, and economy. If response gaps remain, then compare activation policies.

Receipts: `replays/k0-greedy-greedy-seed21.jsonl`, `alpha-seed21.wire`, `alpha-control.txt`.

## 3. Elevation is tactically powerful even with biomes disabled — observed in manual play

My manual default-seed plan was to preserve the fragile shooters, move onto high ground, and use the Brute as a forward spotter. Every selected command was accepted. It won as P0 on round3 with all three starters alive: Brute3HP, Striker2HP, Sniper2HP.

- R1: Brute moved to`(3,-1)` height2; Sniper to`(2,2)` height2; Striker sheltered at`(2,0)` height0. Opponent advanced its whole army.
- R2: Striker climbed to`(3,2)` height2 and hit the opposing height2 Brute for4. My Brute moved to`(6,-2)` height1 and killed its Sniper. My Sniper moved to`(3,0)` height2 and chipped its Brute to2HP.
- I had hoped shared sight would also expose its Striker, but legal-target inspection showed it remained blocked. I used the available Brute shot. This demonstrates why range alone is insufficient: terrain line of sight still affects army spotting even when biome modifiers are off.
- Opponent moved its Brute down to`(4,0)` and hit my Brute for4 with its Striker. The opponent's greedy descent exposed it; it is a controller error, not forced play.
- R3: my Brute killed the Striker; my height2 Sniper killed the now-low Brute with2 damage after defense. Both kills occurred before a response.

Height grants extra range and damage simultaneously. A unit two levels below cannot fire or spot upward with the starters' vertical arcs1; shared spotting helps visibility but does not bypass the attacker's own vertical firing limit. Higher ground therefore affects both offensive output and retaliation availability. This creates meaningful positioning, but the magnitude is large relative to2HP shooters.

**Suggested experiment after symmetry:** compare one level's damage bonus being capped at1, or modest shooter durability increases at preserved total cost. These are hypotheses, not proven improvements. Preserve elevation distinctions in the UI/palette; decorative color should not imply biome mechanics.

References: `TargetingService.cs:27–37,41–51`, `CombatResolver.cs:13–24`, starter stats `Net/GameSetup.cs:72–76`. Exact manual state after every action: `manual-vs-greedy.jsonl`.

## 4. Initiative verdict changes with controller; K2 reduces wipes but is not yet a recommendation

| Current whole-army matchup | N | P0 wins | P1 wins | Draws at round100 |
|---|---:|---:|---:|---:|
| Greedy / Greedy |24|11|13|0|
| Charge / Charge |24|14|5|5|
| Hold / Hold |24|4|0|20|
| Charge / Hold |24|11|4|9|
| Hold / Charge |24|5|8|11|

Equal Greedy is nearly split. The tactical charge controller favors P0 here. Mixed styles favor charge in both seats (19 charge wins,9 hold wins,20 draws across48games), but there is no basis to call optimal play solved. Starting-board asymmetry is an additional confound.

Counterfactual K2, matched seeds1–12 and the same three charge-involving matchups:

- Current-rule subset:24 decisive games,12draws;17/36 games have a multi-kill turn.
- K2:21 decisive games,15draws;1/36 has a multi-kill turn.
- K2 seat results are11 P0 wins/10 P1 wins, but the controller mix and tiny paired sample preclude a fairness claim.
- This changes the meaning of a round and units' budget-refresh cadence, not just how long the opponent waits. Raw round counts are not comparable playtime measures.

**Recommendation:** do not change the default to K2 from these figures alone. First fix geometric symmetry and test the smaller reinforcement-activation intervention.

## 5. Stalemates need separate treatment from controller weakness

45/120 current-rule bot games drew at the engine's round100 cap.44 ended with living armies on both sides;25 had no attacks at all. Conservative one-ply threat scoring can overvalue holding a position and cannot plan multi-turn flanks. These rates are **not** human draw-rate estimates and do not demonstrate that aggression loses under optimal play.

A separate engine-rule issue appears in `k0-charge-hold-seed11`: P1 has no units after round7 but retains20 bank, enough for a standard template; it refuses deployment and the match draws at100. Final state is P0 two units +10 bank, P1 zero units +20 bank. `WinCheck.IsEliminated` treats an affordable reserve as survival indefinitely, and default annihilation resolves the round cap as a draw. The opponent cannot attack unspent bank. This provides an independent reproduction of the reserve-stall mechanism, separate from two living armies waiting.

**Suggested rule test:** when the board is empty, require a legal deployment within one own turn, otherwise lose; specify behavior when no deployment cell is legal. A passive-stockpile tie should not erase an achieved board victory. Separately consider an optional objective or no-progress rule for two live armies; do not infer its necessary form from weak-controller stalls.

References: `WinCheck.cs:49–59,74–75,101–109`; `replays/k0-charge-hold-seed11.jsonl`.

## Controller and interpretation limitations

- Greedy is existing production engine helper code, but ignores nonlethal health damage in its evaluation, can wander on ties, and is not a strong human proxy. BoundedSearch was deliberately not used because its fixed expansion budget and sorted command order can starve root alternatives.
- Charge/Hold are the same one-ply controller with different risk/approach weights. They favor actual damage, can move into shots, use health-weighted value, and estimate enemy next-turn reachable fire. Their threat estimate is conservative: it does not account for killing future attackers or coordinated multi-turn plans. Hold can refuse an affordable but exposed deployment.
- Seat swaps compare style, not a true isometric board swap; the odd-width defect blocks that interpretation. Independent fixed tie-break streams are used for the two seats.
- No custom unit designs, manual placement, fog, larger armies, Territory mode, human opponents, or network latency were tested. These156bot matches +1manual subagent match are an exploratory balance audit, not a statistically calibrated human win-rate study.
- No rules, assets, repository files, or remote branches were changed by this playtester.

## Reproduction

The parent supplied the fresh engine DLL at`/tmp/hexwars-balance-20260927/engine/HexWars.Engine.dll`. Scripts target installed .NET10 SDK and reference that DLL without rebuilding shared source.

```sh
dotnet run --project TurnOrder.csproj -c Release -p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll
dotnet run --project probes/Probes.csproj -c Release -p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll -- symmetry
dotnet run --project probes/Probes.csproj -c Release -p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll -- alpha
# Read-only replay validation and current terminal state of the manual match:
dotnet run --project probes/Probes.csproj -c Release -p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll -- manual
```

Run from the extracted turn-order directory; all outputs use the current working directory. The engine path is a configurable MSBuild property. First command rewrites batch receipts; preserve the supplied originals before rerunning. Manual commands are only appended when explicitly supplied after`manual`. The exploratory manual opponent resets its deterministic RNG at each response turn; its accepted command history is authoritative. Batch controllers maintain their RNG stream for the entire game. Full receipt index is `matches.csv`, `summary.json`, and `replays/`.
