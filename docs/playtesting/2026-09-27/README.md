# HexWars balance playtest — 27 September 2026

Three playtesters found two concrete rule defects and two useful balance experiments. **Fix the empty-army draw loophole and map symmetry first.** Then compare shooter durability and reinforcement timing separately. Biomes and generators remain planned features for other modes; this investigation does not propose removing those systems.

Rules were frozen at main `1cd5f6933702853ef1ed83a4f9f7f865175a5954`. This report changes no gameplay. The concurrent hex-color change is cosmetic.

## Setup correction

The browser setup form defaults to **K=3** and randomizes its seed. Its label is “Units acting per turn”, but moving and attacking with one unit uses two allowances. Multiple movement hops by that unit count once; deploying a unit counts zero. See `SetupForm.cs:35,60,147` and `ITurnPolicy.cs:49–61`.

The initial suites used `GameSetup.Default`: K=0, whole-army turns, seed 7 unless overridden. That is an available mode, **not the browser default**. After inspecting the actual browser, all three agents ran separate K=3 follow-ups and preserved the earlier results. Findings below prioritize the browser setting.

## Findings in priority order

| Priority | Finding | Evidence | Smallest next step |
|---|---|---|---|
| Fix | An army-less player can force a draw by refusing affordable reinforcements | K3, seed 7: army gone in round 3, 20 saved points, draw at round 100. Reproduced in 8/16 adversarial stock-catalog games at K3 and 8/16 at K0. | Allow one own turn to reinforce, then lose if ending it with no army; check legal deployment availability too. |
| Fix | Nine-column maps are not genuinely symmetric in hex geometry | 21/24 seeds have unequal opening move-option totals. Seed 7 gives 27 versus 31. Widths 8 and 10 match on all 24 control seeds. | Use an adjacency-preserving reflection for terrain and formations. An even-width default is a possible interim measure. |
| Experiment | The third health point is more valuable than another unsupported range point | Same-cost HP3/range5 Sniper beat stock HP2/range6 in 11/12 paired equal-budget games at K3, and 11/12 at K0. | Test that one-point trade with mixed armies and people before changing the roster. |
| Experiment | Kill rewards can fund a fresh attacker before the opponent responds | In 120 K3 games, fresh reinforcements fired before handoff in 34 games. A seed-5 example produces a second kill; that side still eventually loses. | Compare fresh units waiting until their next turn to fire, keeping bounty rate and other rules fixed. |
| Clarify | The setup label suggests a different turn budget | One unit moving and firing uses two allowances; a deployment alone uses none. | Rename the field and explain its budget. This may explain some unexpectedly early handoffs. |

## Exact browser-setting stall

The [16-command sequence](evidence/web-k3-stall-commands.txt) uses K=3, seed 7, the stock army/catalog and earned bounty. Player 1 (engine seat 0) loses its last unit in round 3 but retains 20 points. The opponent still has two units with 10 HP in total. Passing 193 more times produces a draw at round 100; the opponent cannot force deployment.

Replaying the same position and choosing a stock reinforcement instead loses in round 4 after five more commands. Refusing a losing comeback earns a better result. A round-100 score tiebreak alone leaves the long empty-board delay. A free inert one-point design widens the loophole, but is not required.

See [K3 economy analysis](economy-k3.md) and the separately labeled [K0 analysis](economy.md). The parent independently reproduced both stalls with all commands accepted.

## Map geometry

The generator reverses rectangular offset coordinates. On an odd-width hex layout this does not preserve adjacency: `(1,0)` to `(2,0)` has distance 1, while their alleged counterparts `(7,3)` and `(6,2)` have distance 2. Matching elevations and mirrored-looking pieces do not guarantee equivalent routes or firing geometry. This proves a defect, not a quantified win-rate advantage, and applies at both K=0 and K=3.

## Durability, counterplay and response gaps

Two HP is exactly one ordinary unarmored Sniper or Brute hit; three HP requires two such hits. A Sniper's own vision is only four, making extra range less useful without a spotter. Shared spotting and high-ground damage can change that comparison. The paired tests used three identical units per side, empty reserves and one bounded tactical controller.

Counterplay already exists: firing ends movement, Brutes punish unsupported Snipers, and destroying a spotter disables a visionless long-range gun. The tested cheap swarm lost 11/12 at K0 and 10/12 at K3. Cheap bounty-funded reinforcements won only 9/16 K0 trials. **This evidence does not support a blanket cheap-unit nerf.** See [K3 unit-design results](unit-design-k3.md) and [manual encounters/K0 results](unit-design.md).

K3 reduced games containing a multi-kill turn from 58/120 to 41/120 under matched controllers/seeds; the maximum observed uninterrupted kills fell from three to two. The exact [whole-army seed-21 wipe](evidence/reinforcement-wipe-commands.txt) is invalid under K3 because its opening three moves already pass the turn. It must not be described as a default-browser game. Fresh-unit activation itself remains possible at K3.

Controller choice reverses the apparent first/second-player advantage. Equal Greedy split 12/12 at K3; the charge controller favored the first seat, while conservative controllers produced many stalemates. There is no robust universal initiative verdict. See [K3 tempo analysis](turn-order-k3.md) and [whole-army/K2 controls](turn-order.md).

## What was played

| Track | Initial suite | Browser K3 follow-up |
|---|---|---|
| Turn order / tempo | 120 K0 games, 36 K2 games, one 22-command subagent-directed game against Greedy | 120 games, same five matchups and 24 seeds |
| Unit design | 6 ordinary games, 36 equal-budget custom-army encounters, 12 manual encounter actions | 42 games over the same seeds/designs; 934 accepted commands, 205 automatic handoffs |
| Economy / endgames | 56 complete games and 4 counterplay probes | 24 complete games plus targeted counterplay |

That is **440 batch games plus one subagent-directed complete game**. Full games reached engine endings within their declared caps. Synthetic armies were installed as controlled initial states, not purchased through a default starting menu. Two manual encounter actions intentionally tested rejection rules. A Territory raid probe hit its approach cap because its controller would not detour; a successful control showed generators are capturable.

These are engine-based playtests, not three people using the GUI. They provide exact rule reproductions and bounded strategic evidence, not human enjoyment measurements or optimal play. Banking counts measure exploit exposure under a deliberate strategy. Conservative bot draws and first-kill win rates are not causal balance estimates. Frozen criteria, controller limitations and correction history are retained.

## Mode boundaries

The engine accepts a legacy generator command in Annihilation while its normal UI hides it. This was reproduced with earned bounty through engine commands, not submitted to a live server. Preserve generator/biome functionality for intended modes and make command availability agree with each mode's UI and rules. Do not remove those systems globally.

## Evidence and reproduction

The parent independently replayed the 16-command K3 stall, 25-command K0 stall, 16-command K0 wipe and 22-command manual game, and checked the adjacency witness. All outcomes reproduced: [verification receipt](evidence/independent-verification-k3.json).

| Track | Browser K3 evidence | Corrected initial evidence |
|---|---|---|
| Turn order | [K3 archive](evidence/turn-order-k3-evidence.tar.gz) | [K0/K2 archive](evidence/turn-order-evidence.tar.gz) |
| Unit design | [K3 archive](evidence/unit-design-k3-evidence.tar.gz) | [K0 archive](evidence/unit-design-evidence.tar.gz) |
| Economy | [K3 archive](evidence/economy-k3-evidence.tar.gz) | [K0 archive](evidence/economy-evidence.tar.gz) |

Archives retain full traces, outcomes, frozen plans, source and reproduction instructions; compiled harnesses/caches are excluded. The [manifest](evidence/manifest.json) records hashes. Engine SHA256: `459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`.

Extract into fresh directories to preserve receipts. Each README documents its .NET 10 harness and `-p:HexWarsEngineDll=/absolute/path/to/HexWars.Engine.dll` override. Build the engine from the recorded release commit with the repository's .NET Standard toolchain. This run used the Windows targeting pack because the Linux host lacked it. No training, live matches, Steam accounts or remote deployment were used.
