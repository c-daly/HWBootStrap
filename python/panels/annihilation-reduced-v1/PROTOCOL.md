# Reduced-curriculum annihilation reward ablation protocol

## Scope

This panel studies the 13x9, three-unit tactical-v2 curriculum only. It must not
be numerically compared with the historical 16x16, 12-unit runs.

## Primary question

Which existing reward components improve held-out annihilation conversion against
Random, and which merely improve survival, material position, or training reward?

## Locked protocol

- Algorithm: MaskablePPO with HexCNN and legal-action masking
- Opponent: Random
- Learner seat: alternating
- Training seeds: 11, 23, 47
- Training budget: 51,200 environment steps per model
- Workers: 4
- Tracking: local metrics and TensorBoard for live inspection
- Checkpoint used: exactly 51,200 steps
- Evaluation seeds: 3,000,000 through 3,000,024
- Evaluation: both candidate seats for every seed, 50 games per model
- Aggregate: 150 held-out games per reward condition
- Primary metric: engine-declared annihilation win rate
- Draws are failures to convert and are never credited as wins
- Secondary checks: training-seed dispersion, seat asymmetry, draw rate, and PPO
  `approx_kl`, clip fraction, and explained variance

The evaluation block is fresh: it is separate from the 2,000,000--2,000,009
seeds used to inspect the two pilot runs.

## Conditions

1. `full_draw025`: original full reward, including draw credit 0.25.
2. `full_nodraw`: remove draw credit only.
3. `minus_closing`: from no-draw, remove closing-distance reward only.
4. `minus_value`: from no-draw, remove material/value-delta shaping only.
5. `terminal_time`: terminal result plus the existing per-decision cost.
6. `terminal_only`: pure terminal win/loss reward.

The panel intentionally does not change PPO hyperparameters. If all reward
conditions show unstable updates, optimizer stability becomes a separate panel
rather than a confounded explanation.

## Interpretation gate

A condition is not considered successful merely because it never loses to
Random. It must convert substantially more games into annihilation wins, do so
across training seeds and both seats, and avoid degradation hidden by aggregate
training reward.

The exact one-off runner is retained for review at
`historical-source/run_annihilation_ablation_panel.py.txt`. It is a historical
source snapshot, not a current entry point.
