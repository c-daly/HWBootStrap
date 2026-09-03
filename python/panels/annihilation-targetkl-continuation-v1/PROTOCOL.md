# Target-KL annihilation continuation protocol

## Question

Does the provisional target-KL PPO configuration continue improving beyond
51,200 steps, or does annihilation conversion plateau or degrade as it did in
the earlier default-PPO pilot?

## Locked scope

- Scenario: tactical-v2, 13x9 board, three starting units, 100-round cap
- Reward: zero draw credit, value-delta retained, closing reward removed
- Algorithm: MaskablePPO / HexCNN with legal-action masking
- PPO settings: learning rate 3e-4, 10 nominal epochs, target KL 0.02
- Opponent: Random; learner seat alternates
- Training seeds: 11, 23, 47
- Source checkpoint: 51,200 steps from the optimizer panel
- Continuation horizon: 100,352 absolute total steps
- Study checkpoints: 51,200, 75,776, and 100,352 steps
- Evaluation seeds: 5,000,000 through 5,000,024, both seats
- Games: 50 per model/checkpoint, 150 per checkpoint
- Primary metric: engine-declared annihilation wins; draws are non-wins
- Secondary metrics: training-seed/seat consistency and cycling-draw incidence
- Tracking: local and TensorBoard

The 75,776 checkpoint is the first completed rollout after the inherited 75,000
checkpoint threshold. The final horizon is exactly rollout-aligned. All three
checkpoints will be evaluated on the same fresh reciprocal schedule, including
the already-selected 51,200 checkpoint, so within-panel checkpoint comparisons
do not mix evaluation seed blocks.

## Interpretation

The continuation is useful only if held-out annihilation conversion improves or
at least remains stable across training seeds and seats. Training reward, KL,
clip fraction, or low loss to Random cannot substitute for wins.

The target-KL configuration is not considered competent merely for improving
over its own 51,200-step checkpoint. Greedy is evaluated on the same fresh maps
as a scenario-convertibility control. Replay-backed evidence is captured for
training seed 11 at every study checkpoint to determine whether checkpoint
movement changes the dominant cycling/failed-conversion behavior.

No reward, opponent, curriculum, architecture, or optimizer setting may change
inside this experiment.

Run from PowerShell with:

```powershell
.\python\winenv\Scripts\python.exe .\python\run_annihilation_targetkl_continuation.py
```
