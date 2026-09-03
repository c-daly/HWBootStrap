# Reduced-curriculum annihilation reward ablation: results

## Verdict

The current MaskablePPO setup does not learn reliable annihilation against Random
at 51,200 steps. Across all 18 models it produced 94 wins, 3 losses, and 803 draws
in 900 fresh held-out games. No individual model exceeded 10 wins in 50 games,
while Greedy won 21 of 50 on the same map/seat schedule.

Reward design matters, but it is not the dominant unresolved problem:

- Removing draw credit had exactly zero aggregate effect: 20/150 wins both with
  and without it. Its effect reversed between training seeds.
- Pure terminal reward was worst (6/150). Adding time pressure helped only
  modestly (10/150).
- Dense closing plus value shaping improved terminal-plus-time from 10/150 to
  20/150 on the paired schedule. This is the clearest component-level signal.
- The best aggregate condition removed closing and retained value shaping
  (24/150), but its advantage over full no-draw reward was small, inconsistent
  across training seeds, and not statistically secure.
- Every reward condition still showed aggressive PPO updates: mean final clip
  fractions ranged from 0.477 to 0.556 and mean approximate KL from 0.066 to
  0.100.

The next experiment should therefore be an optimizer/update-stability panel,
not another reward rewrite.

## Scope and protocol

These results apply only to the reduced tactical-v2 curriculum: 13x9 board,
three starting units per side, and a 100-round cap. They are not numerically
comparable to the historical 16x16, 12-unit runs.

- Algorithm: MaskablePPO / HexCNN with legal-action masking
- Opponent: Random
- Learner seat: alternating
- Training seeds: 11, 23, 47
- Budget: exactly 51,200 steps per model
- Evaluation seeds: 3,000,000 through 3,000,024
- Evaluation: reciprocal seats, 50 games per model
- Aggregate: 150 held-out games per condition
- Primary metric: engine-declared annihilation wins; draws never count as wins

The evaluation block was fresh and separate from the 2,000,000-series pilot
seeds. The protocol was written before panel results were inspected.

## Aggregate results

| Reward condition | Seed 11 W-L-D | Seed 23 W-L-D | Seed 47 W-L-D | Total W-L-D | Win rate | 95% Wilson interval |
|---|---:|---:|---:|---:|---:|---:|
| Full reward, draw credit 0.25 | 2-0-48 | 8-0-42 | 10-0-40 | 20-0-130 | 13.3% | 8.8%-19.7% |
| Full reward, no draw credit | 7-0-43 | 3-1-46 | 10-0-40 | 20-1-129 | 13.3% | 8.8%-19.7% |
| No draw credit, minus closing | 10-0-40 | 7-0-43 | 7-1-42 | 24-1-125 | 16.0% | 11.0%-22.7% |
| No draw credit, minus value delta | 6-0-44 | 6-1-43 | 2-0-48 | 14-1-135 | 9.3% | 5.6%-15.1% |
| Terminal plus time pressure | 4-0-46 | 3-0-47 | 3-0-47 | 10-0-140 | 6.7% | 3.7%-11.8% |
| Pure terminal reward | 2-0-48 | 3-0-47 | 1-0-49 | 6-0-144 | 4.0% | 1.8%-8.5% |

The Wilson intervals describe game-level sampling only. They do not account for
the hierarchy of 50 games nested inside each of only three trained models, so
training-seed consistency matters more than a naive interval ranking.

## Scripted controls

| Candidate | Games | W-L-D | Win rate |
|---|---:|---:|---:|
| Greedy vs Random, reciprocal | 50 | 21-0-29 | 42.0% |
| Random vs Random | 25 | 0-0-25 | 0.0% |

Random's inability to finish another Random policy explains why PPO losses are
nearly absent. "Never loses to Random" is not evidence of competence here.
Greedy demonstrates that the scenario is convertible, but even Greedy leaves
58% of games unresolved.

## Paired component comparisons

Each comparison pairs training seed, evaluation seed, and candidate seat. The
`left-only` and `right-only` columns count games won by one condition but not the
other.

| Question | Left vs right | Left-only | Right-only | Net left wins | Net by training seed | Exact sign p |
|---|---|---:|---:|---:|---:|---:|
| Remove draw credit | full no-draw vs full draw | 13 | 13 | 0 | +5, -5, 0 | 1.000 |
| Remove closing | minus-closing vs full no-draw | 17 | 13 | +4 | +3, +4, -3 | 0.585 |
| Retain value delta | full no-draw vs minus-value | 18 | 12 | +6 | +1, -3, +8 | 0.362 |
| Add time pressure | terminal-time vs terminal-only | 7 | 3 | +4 | +2, 0, +2 | 0.344 |
| Add closing and value shaping | full no-draw vs terminal-time | 13 | 3 | +10 | +3, 0, +7 | 0.021 |

The exact sign calculation treats paired games as independent and is therefore
optimistic because games share trained policies. It is included as a sensitivity
check, not as a substitute for more training seeds. The robust conclusions are
the large absolute failure rate, the null and seed-reversing draw-credit result,
and the consistent inferiority of sparse terminal reward.

## PPO update diagnostics at 51,200 steps

| Condition | Mean approximate KL | Mean clip fraction | Mean explained variance |
|---|---:|---:|---:|
| Full reward, draw credit 0.25 | 0.092 | 0.543 | -0.046 |
| Full reward, no draw credit | 0.100 | 0.556 | -0.134 |
| Minus closing | 0.087 | 0.513 | 0.051 |
| Minus value delta | 0.083 | 0.523 | 0.033 |
| Terminal plus time pressure | 0.066 | 0.484 | 0.211 |
| Pure terminal reward | 0.069 | 0.477 | -0.003 |

Between 48% and 56% of samples were clipped in the final PPO update across every
condition. Reward ablation did not remove the common optimization pathology.
Training reward is also not comparable across conditions because the reward
definitions differ; notably, pure-terminal runs reported positive training mean
reward while achieving only 1-3 held-out wins per model.

## Behavioral evidence from the strongest condition

Replay-backed diagnostics were captured for the first 10 reciprocal evaluation
seeds of `minus_closing`, training seed 11. The model scored 5-0-15 in those 20
games.

- All 15 draws reached round 100.
- All 15 matched the cycling diagnostic.
- 11/15 matched action waste.
- 4/15 were classified primarily as failed conversion.
- 5/15 matched balanced attrition; 2/15 matched damage stalemate.
- In draws the candidate averaged 2.13 kills and a positive normalized final
  advantage of 0.237.
- Its five wins took 8-68 rounds, averaging 34.8.

This is not a policy that failed to discover combat at all. It often damages and
out-values Random, then fails to locate, pursue, or finish the remaining force.
The missing capability is reliable conversion, exactly the behavior the primary
metric was designed to expose.

## Answers and next decision

1. **Were draws being mislabeled as wins?** No. All panel wins are engine-declared
   annihilations. Draw credit affected training reward only.
2. **Was positive draw credit the main cause?** No. Removing it was neutral in
   aggregate and reversed effect across seeds.
3. **Can terminal reward teach the rules by itself at this budget?** Not well.
   Pure terminal reward won 4% of games; time pressure raised that to only 6.7%.
4. **Does dense shaping help?** Yes, collectively. Closing plus value shaping
   doubled terminal-plus-time conversion, from 6.7% to 13.3%. The value term is
   directionally more useful than the closing term, but three seeds do not prove
   that individual attribution.
5. **Are these models competent?** No. The best condition reached 16% aggregate
   conversion and every individual model remained far below Greedy's 42%.
6. **What should change next?** Hold the reward fixed and ablate PPO update
   aggressiveness. Use `minus_closing` provisionally because it was best and does
   not pay merely for approaching, but retain `full_nodraw` as a confirmation arm
   to guard against winner's-curse selection.

A focused optimizer panel should compare the current defaults against a lower
learning rate, fewer epochs per rollout, and a KL-limited update, with the same
three training seeds and fresh reciprocal evaluation block. Do not change reward,
opponent, and optimizer simultaneously.

## Artifacts

- `PROTOCOL.md`: preregistered scope and conditions
- `panel.json`: machine-readable panel definition
- `aggregate.json`: per-model, aggregate, seat, diagnostic, control, and paired data
- `evaluations/`: all 18 fixed-checkpoint evaluation reports
- `controllers/`: immutable 51,200-step controller specifications
- `scenarios/`: the six materialized reward scenarios
- `evidence/minus-closing-seed11/`: retained traces and authoritative replays
- `../../run_annihilation_ablation_panel.py`: restart-safe PowerShell-callable runner
