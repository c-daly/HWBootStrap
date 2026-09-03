# Target-KL annihilation continuation: results

## Verdict

Target KL fixed the obvious PPO update pathology, but additional training on the
same task distribution did not improve annihilation conversion.

On one locked fresh reciprocal schedule, the three-seed aggregate moved from
52/150 wins at 51,200 steps, to 47/150 at 75,776, to 50/150 at 100,352. The final
checkpoint was two wins below the source checkpoint. Training-seed changes were
+3, -8, and +3 wins, so the flat aggregate conceals substantial behavioral drift.

The optimizer itself remained controlled: final mean approximate KL was 0.018
and clip fraction 0.081. This is not the earlier high-KL failure recurring. The
remaining problem is behavioral learning and retention. At 100,352 steps the
models still drew 100/150 games, and every draw in the final replay-backed
evidence slice reached round 100 and matched the cycling diagnostic.

More timesteps on this exact starting-state distribution are not justified by
these results. Keep target KL as the provisional PPO setting, but make the next
experiment a finishing/conversion curriculum rather than another straight
continuation.

## Scope and protocol

These results apply only to the reduced tactical-v2 curriculum: 13x9 board,
three starting units per side, and a 100-round cap. They are not numerically
comparable to the historical 16x16, 12-unit runs.

- Algorithm: MaskablePPO / HexCNN with legal-action masking
- PPO settings: learning rate 3e-4, ten nominal epochs, target KL 0.02
- Reward: zero draw credit, value-delta retained, closing reward removed
- Opponent: Random
- Learner seat: alternating
- Training seeds: 11, 23, 47
- Source checkpoint: 51,200 steps
- Continued checkpoints: 75,776 and 100,352 steps
- Evaluation seeds: 5,000,000 through 5,000,024
- Evaluation: reciprocal seats, 50 games per model/checkpoint
- Aggregate: 150 held-out games per checkpoint
- Primary metric: engine-declared annihilation wins; draws never count as wins

The protocol and exact checkpoint schedule were written before the 5,000,000
evaluation block was consumed. The existing 51,200 checkpoints were re-evaluated
on that same block, so checkpoint comparisons pair training seed, map seed, and
candidate seat.

## Aggregate learning curve

| Checkpoint | Seed 11 W-L-D | Seed 23 W-L-D | Seed 47 W-L-D | Total W-L-D | Win rate | 95% Wilson interval | P0/P1 wins |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 51,200 | 14-0-36 | 21-1-28 | 17-0-33 | 52-1-97 | 34.7% | 27.5%-42.6% | 23/29 |
| 75,776 | 17-0-33 | 13-0-37 | 17-0-33 | 47-0-103 | 31.3% | 24.5%-39.1% | 23/24 |
| 100,352 | 17-0-33 | 13-0-37 | 20-0-30 | 50-0-100 | 33.3% | 26.3%-41.2% | 22/28 |

Greedy scored 29-1-20 in 50 games on the same maps, or 58%, with 14 player-0
wins and 15 player-1 wins. The final PPO aggregate remained 24.7 percentage
points below that control and drew twice as often.

The Wilson intervals treat games as independent and are optimistic: all three
models reuse the same 25 generated maps, and only three independently trained
policies contribute to each checkpoint. Training-seed consistency and paired
movement are therefore more important than the nominal intervals.

## Paired checkpoint comparisons

| Later vs earlier | Later-only wins | Earlier-only wins | Net later wins | Net by training seed | Exact sign p |
|---|---:|---:|---:|---:|---:|
| 75,776 vs 51,200 | 23 | 28 | -5 | +3, -8, 0 | 0.576 |
| 100,352 vs 75,776 | 25 | 22 | +3 | 0, 0, +3 | 0.771 |
| 100,352 vs 51,200 | 20 | 22 | -2 | +3, -8, +3 | 0.878 |

The game-level exact tests are sensitivity checks, not trained-policy-level
proof. The useful result is the absence of a consistent positive learning curve:
seed 11 improved early, seed 47 improved late, and seed 23 lost eight wins and
did not recover.

## Evaluation-block sensitivity

The same 51,200 target-KL models scored 34/150 on the earlier 4,000,000-series
optimizer-panel maps and 52/150 on this preregistered 5,000,000-series block.
Greedy likewise moved from 22/50 to 29/50. Some evaluation blocks are materially
easier to convert than others.

This does not invalidate the learning-curve result because all three checkpoints
were paired on the new block. It does mean that a single 25-map block is not a
stable absolute estimate of competence. Future checkpoint promotion should use
multiple disjoint map blocks or a larger map sample; aggregate game count alone
overstates precision when maps are reused across trained policies.

## PPO diagnostics

| Checkpoint | Mean approximate KL | Mean clip fraction | Mean explained variance | Mean cumulative updates |
|---:|---:|---:|---:|---:|
| 51,200 | 0.0208 | 0.140 | 0.065 | 114.0 |
| 75,776 | 0.0153 | 0.052 | 0.085 | 149.7 |
| 100,352 | 0.0184 | 0.081 | 0.183 | 181.3 |

The KL stop continued to terminate optimization epochs: the models averaged
only about 181 cumulative updates by 100,352, far below the number ten complete
epochs would produce. Approximate KL and clipping stayed in the useful region
identified by the optimizer panel. The behavioral plateau cannot be repaired by
claiming that PPO simply became unstable again.

## Replay-backed behavioral evidence

Training seed 11 was traced on the first ten reciprocal evaluation seeds at all
three study checkpoints.

| Checkpoint | W-L-D | Draws at round 100 | Cycling draws | Action-waste draws | Mean draw advantage |
|---:|---:|---:|---:|---:|---:|
| 51,200 | 6-0-14 | 14/14 | 14/14 | 10/14 | 0.195 |
| 75,776 | 6-0-14 | 14/14 | 14/14 | 8/14 | 0.192 |
| 100,352 | 8-0-12 | 12/12 | 12/12 | 7/12 | 0.205 |

Seed 11 did improve modestly: failed-conversion classifications fell from three
to zero in this evidence slice, and wins rose from six to eight. But the
remaining draws did not become near-misses of a different kind. Every one still
cycled to the round cap. This local improvement did not generalize across the
three training seeds because seed 23 regressed by eight full-evaluation wins.

## Answers and next decision

1. **Did target KL prevent a return to explosive PPO updates?** Yes. KL and clip
   fraction remained controlled throughout the continuation.
2. **Did more training make annihilation more reliable?** No. Aggregate wins were
   essentially flat and finished two below the source checkpoint.
3. **Is there a clearly superior late checkpoint?** No. The 51,200 checkpoint was
   numerically highest on the preregistered block, but the differences were small,
   seed-dependent, and evaluation-block sensitivity was large.
4. **Should training continue unchanged beyond 100,352?** No. The experiment has
   stopped yielding a general learning signal.
5. **What did the policy actually learn?** It can engage, kill, and sometimes
   convert. It still cannot reliably turn surviving material or positional
   advantage into pursuit and annihilation before the round cap.
6. **What should change next?** Change the training-state distribution while
   holding target KL and the no-draw reward fixed. Build a conversion curriculum
   containing randomized late-game states such as 3-v-1, 2-v-1, and 1-v-1 at
   varied separation and terrain, still against Random. Success remains actual
   annihilation with time pressure; a draw receives no terminal credit. Mix those
   states with ordinary 3-v-3 starts so finishing is learned as a reusable skill
   rather than a replacement game.

That curriculum directly increases the frequency of the rare learning event the
current models miss: locating and killing the final enemy. It does so without
paying for draws, proximity, or merely holding a material advantage. Only after
conversion becomes reliable against Random should the opponent curriculum move
to Greedy.

## Artifacts

- `PROTOCOL.md`: preregistered continuation and evaluation schedule
- `panel.json`: machine-readable experiment definition
- `aggregate.json`: checkpoint, seed, seat, control, and paired outcomes
- `evaluations/`: nine fixed-checkpoint 50-game evaluations
- `controllers/`: immutable checkpoint controller specifications
- `evidence/`: replay-backed seed-11 evidence at all three checkpoints
- `../../run_annihilation_targetkl_continuation.py`: restart-safe runner
