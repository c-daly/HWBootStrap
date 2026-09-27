# HexWars balance playtest — 27 September 2026

Three independent playtesters found two reproducible rule defects and two useful balance
experiments. **Fix the empty-army draw loophole and map symmetry before tuning the roster.**
Then test reinforcement timing and shooter health separately.

The rules were frozen at main `1cd5f6933702853ef1ed83a4f9f7f865175a5954`.
This report changes no gameplay, unit costs, starting conditions or deployment settings.
The concurrent hex-color work is cosmetic and was excluded from these rules tests.

## Findings in priority order

| Priority | Finding | Evidence | Smallest next step |
|---|---|---|---|
| Fix | A player with no army can force a draw by refusing affordable reinforcements | Default seed 7: no units from round 4, but 20 saved points prevent elimination until a round-100 draw. Stock catalogs reproduced this in 8/16 adversarial matches. | Allow one own turn to reinforce, then lose if ending that turn with no army. Also check that a legal deployment exists. |
| Fix | The default nine-column map is not truly symmetric in hex geometry | 21/24 seeds have unequal opening move-option totals. Seed 7 gives 27 versus 31. Widths 8 and 10 have equal totals on all 24 control seeds. | Mirror the hex geometry and formations with an adjacency-preserving transform. An even-width default is a possible interim measure, not a complete board-generator fix. |
| Experiment | New reinforcements can join a kill chain before the opponent can respond | In a legal seed-21 game, two kills buy a new Sniper, which immediately moves and fires; all three opponents die in that turn. Omitting the fresh unit's actions leaves the enemy Brute alive. | Compare making new units wait until their next turn to fire. Keep bounty rate and other rules fixed. |
| Experiment | The third health point is disproportionately valuable for fragile shooters | A same-cost HP3/range5 Sniper beat the stock HP2/range6 Sniper in 11/12 equal-budget, seat-swapped encounters. | Test that one-point trade with mixed armies and people before changing the whole roster. |

### Why the draw loophole matters

The defeated player has an incentive to stop playing. In the same reproduced position,
choosing a normal reinforcement loses in round 5; declining to deploy earns a draw after
192 further end-turn commands. The opponent has no remaining target and cannot force
deployment. A score tiebreak at round 100 would still leave the long empty-board delay.

The [25-command reproduction](evidence/default-stall-commands.txt) starts from the ordinary
default game, uses only stock templates and spends actual kill rewards. A free inert
one-point design widens the loophole, but is not required. See [economy analysis](economy.md).

### Why the board needs correction before interpreting win rates

The generator reverses rectangular offset coordinates. With an odd number of columns,
that operation does not preserve hex adjacency: `(1,0)` to `(2,0)` has distance 1, while
their alleged counterparts `(7,3)` and `(6,2)` have distance 2. Mirrored-looking pieces
and matching elevations therefore do not guarantee equivalent routes or firing geometry.
This proves a geometry defect, not a quantified advantage in expert games.

The [turn-order analysis](turn-order.md) contains the odd/even-width controls, a complete
subagent-directed game, and the [reinforcement wipe sequence](evidence/reinforcement-wipe-commands.txt).

### What the roster tests suggest

Two health is exactly one ordinary unarmored Sniper or Brute hit. Three health requires
two such hits. Spending a point on otherwise unsupported range is less useful when a
Sniper's own vision is only four. This helps explain the HP3/range5 result, but shared
spotters and high-ground damage can change the matchup.

Counterplay already exists: firing ends movement, a Brute can punish an unsupported
Sniper, and killing a spotter disables a visionless long-range gun. The tested cheap
swarm lost 11/12 equal-budget encounters; cheap bounty-funded reinforcements won only
9/16 trials. **This study does not support a blanket cheap-unit nerf.**
See [unit-design analysis](unit-design.md) for exact designs and manual encounters.

## What was played

| Track | Actual play | Important boundary |
|---|---|---|
| Turn order / tempo | 120 current-rule bot games, 36 separately labeled K=2 games, plus one 22-command game chosen action-by-action by a subagent against Greedy | No custom unit designs; controller choice reverses the apparent first/second-player advantage. |
| Unit design / counterplay | 6 ordinary starter games, 36 equal-budget custom-army encounters, 12 manually selected encounter actions | Custom starting armies were installed as controlled scenarios, not purchased through a default starting menu. Two manual actions intentionally tested rejection rules. |
| Economy / endgames | 56 complete public-setup games and 4 counterplay probes | Deliberate banking/pass strategies measure exposure to an exploit, not expected human draw frequency. A custom Territory raid hit its approach cap because of controller pathfinding. |

That is **254 batch games plus one subagent-directed complete game**, alongside controlled
encounters and geometry checks. All full games reached engine endings within their declared
caps. The economy agent replay-validated all 56 complete traces. The parent independently
replayed the 25-command default stall, 16-command seed-21 wipe and 22-command manual game;
all commands were accepted and all reported outcomes reproduced. The
[verification receipt](evidence/independent-verification.json) records those checks.

These were engine-based playtests, not three people using the GUI. They provide exact
mechanical reproductions and bounded strategic evidence. They do not measure human
enjoyment, optimal play, or population win rates. Policies, seed sets, action caps and
counterfactuals are recorded in each track's frozen plans. The whole-army first-killer
win rate (63/75 decisive games) is descriptive, not an isolated causal estimate of bounty.

## Secondary issue

The engine still accepts a legacy generator command in Annihilation even though its
normal UI does not expose that economy action. This was reproduced with earned default-game
bounty through engine commands, not submitted to a live server. Treat it as a rule/API
consistency issue separate from ordinary player balance. Territory generators were
successfully captured; an early suspicion that they were impassable was disproved.

## Evidence and reproduction

The compressed bundles retain full legal command traces, original outcomes, frozen plans,
controller source and reproduction instructions. They exclude compiled harnesses and
build caches. The engine used for the run has SHA256
`459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312`.

- [Turn-order evidence](evidence/turn-order-evidence.tar.gz)
- [Unit-design evidence](evidence/unit-design-evidence.tar.gz)
- [Economy evidence](evidence/economy-evidence.tar.gz)
- [Archive and receipt hashes](evidence/manifest.json)

Extract each archive into a fresh directory to preserve the original outputs. Its README
describes running the isolated .NET 10 harness with
`-p:HexWarsEngineDll=/absolute/path/to/HexWars.Engine.dll`. Build that engine from the
recorded release commit using the repository's normal .NET Standard toolchain. The live
run used the installed Windows targeting pack; the Linux host lacked that pack. No
training jobs, live matches, Steam accounts or remote deployment were used.
