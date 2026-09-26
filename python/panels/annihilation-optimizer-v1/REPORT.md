# Reduced-curriculum annihilation PPO optimizer panel: results

## Verdict

KL-limited PPO is the first optimizer change in this investigation that is
directionally useful, but annihilation is still not trainable to a competent
level at 51,200 steps.

- Adding `target_kl=0.02` to the existing PPO configuration raised held-out
  annihilation wins from 27/150 to 34/150. It improved or tied the default in
  every training seed: +4, +3, and 0 wins.
- Lowering the learning rate alone was nearly as strong at 33/150 and likewise
  improved or tied all three seeds.
- Applying every conservative control together did not help: the combined
  condition remained at 27/150 despite extremely low KL and clipping. Fewer
  epochs alone fell to 24/150.
- Greedy won 22/50 games (44%) on the same evaluation maps and reciprocal seat
  schedule. The best PPO condition won 22.7%; 116/150 games were still draws.
- Replay-backed inspection of the strongest individual target-KL model found
  that all 17 inspected draws reached round 100 and all 17 were cycling.

The actionable answer is therefore not "make PPO maximally conservative." Use
the existing learning rate with a KL stop as the provisional configuration and
test whether it continues improving beyond 51,200 steps. This panel does not
establish statistical proof from only three trained models, and it does not
solve the finishing behavior.

## Scope and protocol

These results apply only to the reduced tactical-v2 curriculum: 13x9 board,
three starting units per side, and a 100-round cap. They are not numerically
comparable to the historical 16x16, 12-unit runs.

- Algorithm: MaskablePPO / HexCNN with legal-action masking
- Opponent: Random
- Learner seat: alternating during training
- Training seeds: 11, 23, 47
- Budget: exactly 51,200 steps per model
- Evaluation seeds: 4,000,000 through 4,000,024
- Evaluation: reciprocal seats, 50 games per model
- Aggregate: 150 held-out games per condition
- Primary reward: zero draw credit, value-delta retained, closing reward removed
- Primary metric: engine-declared annihilation wins; draws never count as wins

The primary default controls reused the exact 51,200-step models from the reward
panel. Fifteen new CUDA models were trained for the non-default conditions. The
evaluation seed block was fresh and was not used in the pilot or reward panel.
The full no-draw reward was retained as a confirmation arm so that an optimizer
conclusion did not depend entirely on the reward-panel winner.

## Aggregate results

| PPO condition | Seed 11 W-L-D | Seed 23 W-L-D | Seed 47 W-L-D | Total W-L-D | Win rate | 95% Wilson interval |
|---|---:|---:|---:|---:|---:|---:|
| Current defaults | 10-0-40 | 7-0-43 | 10-0-40 | 27-0-123 | 18.0% | 12.7%-24.9% |
| Lower learning rate only | 12-0-38 | 11-0-39 | 10-0-40 | 33-0-117 | 22.0% | 16.1%-29.3% |
| Three epochs only | 5-0-45 | 8-0-42 | 11-0-39 | 24-0-126 | 16.0% | 11.0%-22.7% |
| Target KL 0.02 only | 14-0-36 | 10-0-40 | 10-0-40 | 34-0-116 | 22.7% | 16.7%-30.0% |
| Combined conservative update | 11-0-39 | 10-0-40 | 6-0-44 | 27-0-123 | 18.0% | 12.7%-24.9% |
| Full reward, current defaults | 9-0-41 | 5-0-45 | 10-0-40 | 24-0-126 | 16.0% | 11.0%-22.7% |
| Full reward, combined update | 10-0-40 | 8-1-41 | 7-0-43 | 25-1-124 | 16.7% | 11.6%-23.4% |

The Wilson intervals describe game-level sampling only. They do not account for
50 games being nested inside each of only three independently trained models.
The intervals overlap substantially, so the result is a direction for the next
experiment, not a claim that target KL is proven superior.

## Scripted control and seats

Greedy scored 22-0-28 in 50 games, or 44%, with 10 wins as player 0 and 12 as
player 1. This again establishes that the reduced scenario is convertible even
though Random makes losses rare and Greedy itself does not finish reliably.

The target-KL models produced 18 player-0 wins and 16 player-1 wins. Each model
was individually balanced: 8/6, 5/5, and 5/5. The lower-learning-rate condition
had almost the same aggregate result but was less reassuring by seat: its seed
47 model produced one player-0 win and nine player-1 wins. Target KL therefore
has the cleaner signal across both training seeds and seats.

## Paired comparisons

Each comparison pairs training seed, evaluation seed, and candidate seat. The
`left-only` and `right-only` columns count games won by one condition but not the
other.

| Question | Left vs right | Left-only | Right-only | Net left wins | Net by training seed | Exact sign p |
|---|---|---:|---:|---:|---:|---:|
| Lower learning rate | low LR vs default | 20 | 14 | +6 | +2, +4, 0 | 0.392 |
| Fewer optimization epochs | three epochs vs default | 19 | 22 | -3 | -5, +1, +1 | 0.755 |
| KL-limited updates | target KL vs default | 21 | 14 | +7 | +4, +3, 0 | 0.311 |
| Combined conservative update | combined vs default | 21 | 21 | 0 | +1, +3, -4 | 1.000 |
| Combined update confirmation | full combined vs full default | 17 | 16 | +1 | +1, +3, -3 | 1.000 |
| Reward choice under combined update | value-only vs full reward | 14 | 12 | +2 | +1, +2, -1 | 0.845 |

The exact sign calculation treats paired games as independent and is optimistic
because games share trained policies. The more credible evidence is that target
KL and lower learning rate improved or tied all three training seeds, whereas
the other interventions did not.

## PPO update diagnostics at 51,200 steps

| Condition | Mean approximate KL | Mean clip fraction | Mean explained variance | Mean updates | Mean steps/s |
|---|---:|---:|---:|---:|---:|
| Current defaults | 0.087 | 0.513 | 0.051 | 240 | 193 |
| Lower learning rate only | 0.014 | 0.128 | 0.086 | 240 | 272 |
| Three epochs only | 0.011 | 0.108 | 0.121 | 72 | 463 |
| Target KL 0.02 only | 0.021 | 0.140 | 0.065 | 114 | 438 |
| Combined conservative update | 0.0028 | 0.010 | 0.353 | 72 | 399 |
| Full reward, current defaults | 0.100 | 0.556 | -0.134 | 240 | 185 |
| Full reward, combined update | 0.0026 | 0.009 | 0.261 | 72 | 401 |

The target-KL runs performed 120, 110, and 112 optimizer updates, versus 240 for
the default runs. The KL stop therefore cut off roughly half of the nominal ten
epochs while retaining the default 3e-4 learning rate. It reduced final mean KL
from 0.087 to 0.021 and clip fraction from 0.513 to 0.140 while increasing wins.

Diagnostics are not themselves the objective. The combined condition's KL of
0.0028 and clip fraction of 0.010 look safest, yet it produced exactly the same
number of wins as the unstable default. Reducing epochs alone also improved the
diagnostics while reducing wins. The evidence supports a useful update-size
region, not a monotonic rule that smaller PPO updates are always better.

## Behavioral evidence from the strongest condition

Replay-backed diagnostics were captured for the first 10 reciprocal evaluation
seeds of the strongest individual model, target KL training seed 11. This
20-game evidence slice scored 3-0-17; the model's complete 50-game result was
14-0-36.

- All 17 draws reached round 100 and all 17 matched the cycling diagnostic.
- Cycling was the primary classification for 15 draws; failed conversion was
  primary for two.
- Fourteen draws matched action waste, three balanced attrition, two damage
  stalemate, and one avoidance.
- Draws ended with mean normalized candidate advantage 0.156, ranging from
  -0.207 to 0.492.
- The three wins took 22-81 rounds and averaged 46.3 rounds.

Target KL improves how often the model converts, but does not change the dominant
failure mode. The policy can fight and sometimes gain a large material advantage,
then repeats states or wastes turns instead of finding and killing the remaining
enemy.

## Answers and next decision

1. **Did PPO update instability matter?** Yes, directionally. Both target KL and
   lower learning rate improved or tied all training seeds while sharply reducing
   KL and clipping.
2. **Which intervention should be carried forward?** Target KL alone. It achieved
   the best aggregate conversion, remained balanced by seat, and lets PPO use the
   larger learning rate until an update actually becomes too large.
3. **Should all conservative settings be combined?** No. The combined condition
   under-updated by the behavioral metric and erased the gain.
4. **Is the model now competent against Random?** No. Target KL won 22.7% versus
   Greedy's 44%, and more than three quarters of its games were draws.
5. **Is target KL proven?** No. Three training seeds provide a consistent
   directional result, but not enough independent policies for a secure effect
   estimate.
6. **What experiment comes next?** Hold reward, opponent, learning rate, and
   `target_kl=0.02` fixed. Extend the same three-seed target-KL arm beyond 51,200
   steps to at least 100,352, retain rollout-aligned intermediate checkpoints,
   and evaluate every checkpoint on one new reciprocal seed block. This directly
   tests whether KL limiting prevents the late degradation previously observed
   in default PPO. Re-evaluate the existing 51,200 checkpoints on that same new
   block so checkpoint comparisons do not reuse this panel's selection data.

Do not change reward shaping or introduce Greedy training in that experiment.
If target-KL learning still plateaus in cycling draws, the next ablation should
target state/action representation or curriculum structure rather than paying
more for draws or adding another dense reward term.

## Artifacts

- `PROTOCOL.md`: preregistered scope and conditions
- `panel.json`: machine-readable panel definition
- `aggregate.json`: per-model, aggregate, seat, diagnostic, control, and paired data
- `evaluations/`: all 21 fixed-checkpoint evaluation reports
- `controllers/`: immutable controller specifications
- `../annihilation-reduced-v1/scenarios/`: the reused materialized reward scenarios
- `evidence/value-targetkl-seed11/`: retained traces and authoritative replays
- `historical-source/run_annihilation_optimizer_panel.py.txt`: exact source snapshot of the one-off runner
