# Browser-default K3 follow-up — 2026-09-27

**Scope correction:** the actual browser setup starts with three combat actions per turn and randomizes the map seed. The earlier K0 report tested the selectable whole-army option and `GameSetup.Default`, not the default browser pace. `SetupForm.cs:35` initializes `_turnActions=3`, `:61` randomizes seed, and `:294–295` passes those values into the game setup. This follow-up changes only turnActions to3, leaving controllers, weights, tie-break streams, seeds1–24, and roster unchanged. It is a separately frozen120-game study, not a retuned rerun.

All120 games reached an engine terminal state with14,909 accepted commands, zero illegal batch commands and zero external caps. The single deliberately rejected command in the negative control is expected and reported separately. The engine snapshot is main1cd5f6933702853ef1ed83a4f9f7f865175a5954, DLL SHA256459f67ad83cd61576d034ca1f9fc25d3dc3974fa4c25f96d6257fe6d53817312.

## Matched outcomes

Each row has24 identical seeds per pace. P0 acts first. Draws occur at the actual engine round100 cap.

| Matchup | K0 P0/P1/draw | Browser K3 P0/P1/draw | K0 games with multi-kill turn | K3 games with multi-kill turn |
|---|---|---|---:|---:|
| Greedy / Greedy |11/13/0|12/12/0|22|21|
| Charge / Charge |14/5/5|14/6/4|18|9|
| Hold / Hold |4/0/20|3/0/21|1|1|
| Charge / Hold |11/4/9|10/0/14|11|6|
| Hold / Charge |5/8/11|2/9/13|6|4|
| **Total** |**45/30/45**|**41/27/52**|**58/120**|**41/120**|

The maximum kills in an uninterrupted turn fell from3 under K0 to2 under K3. First killer won46/68 decisive K3 games (67.6%) versus63/75 K0 games (84%). This is a matched descriptive comparison, not causal evidence that first kill alone determines victory. Controller behavior and position remain confounders.

Equal Greedy was12/12, while equal tactical Charge was14/6 with4draws. Thus the default browser pace still does not have a proven universal seat advantage. Conservative bots produced most draws. No empty-army reserve stall appeared among these120 K3 terminal results;52 games drew with living armies,28 without any attack. These are controller-conditioned rates, not expected human outcomes. The rule permitting reserve stalling remains in the engine and is independently reproduced in the earlier whole-army study and the team's economy study; absence here does not refute it.

## The earlier complete-army wipe does not transfer to K3

The exact K0 seed21 command sequence auto-passes after its third opening move under K3. Its following `E 0` is correctly rejected `NotYourTurn`. Therefore it is invalid to present that uninterrupted three-unit wipe as an example from the default web pace.

Receipt: `k3-mechanism-controls.txt`, negative-control section; original unchanged sequence retained as `alpha-seed21-k0-original.wire`.

## Fresh units still move and fire before the opponent responds

K3 counts movement/attacks but deployment itself does not consume a combat action. Across this K3 suite, newly deployed units fired before handoff36 times in34 matches, with24 kills. This is a real default-pace mechanism, not merely a hypothetical transfer from K0.

A complete37-command replay, Charge/Charge seed5, contains this round5 P1 sequence:

```text
A 1 4 7       # existing Brute kills enemy Sniper; bank10 becomes20, actions3->2
D 1 2 6 -3   # buy a fresh Sniper; actions stay2
M 1 8 4 -1   # new Sniper moves; actions2->1
A 1 8 3      # new Sniper kills the other enemy Sniper; automatic handoff
```

The opponent loses two Snipers without a response between them. The new unit appears off-board, moves into a firing position and attacks immediately. Nevertheless P0 ultimately wins this game on round7, showing that even this two-kill sequence is not an automatic victory. Full replay validation is in `k3-mechanism-controls.txt`; all accepted commands are in `fresh-activation-seed05.wire`; all states are in `replays/k3-charge-charge-seed05.jsonl`. Every observed fresh activation is indexed in `fresh-activation-receipts.json`.

**Small experiment worth trying:** reinforcements arrive now but cannot attack until their next own turn. Test this specifically under K3 and compare comeback opportunities as well as snowballing; the example is a trailing player's counterattack, so delaying reinforcements might harm recoverability as well as reduce response gaps. No such rule change was implemented.

## UI wording does not describe the rule accurately

`SetupForm.cs:147` calls this setting **“Units acting per turn.”** The engine's `KActionsPolicy` counts `MovedUnitIds.Count + AttackedUnitIds.Count` instead: one unit moving and then firing consumes two actions. Repeated movement hops by that same unit consume only one movement action in total; deployment consumes zero. This is neither a simple number of units nor a count of clicks.

Verified default K3, seed7:

```text
M 0 1 1 0   # first unit moves:2 remaining
M 0 1 2 0   # same unit moves again:still2 remaining
M 0 3 2 2   # second unit moves:1 remaining
```

A player interpreting the setup literally may expect three units to move and fire, then discover that the game hands over before all three have fired. **Suggested label:** “Combat actions per turn,” with compact help explaining one movement allocation plus one shot per unit, and deployment outside that budget. This is a clarity suggestion, not a balance-rule change.

## Conclusions for the actual browser default

1. **Odd-width geometric symmetry remains the strongest confirmed fairness defect.** K3 does not change the generated board. Earlier width8/9/10 controls and adjacency witness remain applicable to the browser's9x7 default.
2. **K3 already limits the whole-army response gap.** Do not recommend “switch from whole-army to interleaved turns” as though the browser were not already doing it. K2 is a separate optional counterfactual, not the current pace.
3. **Two-kill turns and fresh reinforcement shots remain possible and observed.** The compact activation rule is a candidate for a small experiment, with explicit comeback tradeoffs.
4. **Clarify the turn-budget label before further judging clunkiness.** The mismatch is concrete and explains why a player may experience an unexpected handoff.
5. **Avoid a global initiative claim or a human draw-rate claim.** Same one-ply-controller limitations apply as in the earlier report. No new manual subagent-versus-bot game was conducted at K3; this follow-up comprises120 bot games plus complete engine replay inspection and mechanism controls. The earlier manually directed game remains correctly labeled K0.

## Reproduction

See `README.md`. Both projects accept `-p:HexWarsEngineDll=/absolute/path/HexWars.Engine.dll`, and outputs use cwd. `protocol.md` was frozen before these results. `matched-comparison.json` contains exact row counts; all120 replays are retained. No production source, rules, or assets changed.
