# Reduced-curriculum annihilation PPO optimizer protocol

## Question

Can less aggressive PPO updates materially improve held-out annihilation
conversion against Random after reward ablation failed to produce a competent
policy?

## Locked scope

- Scenario: tactical-v2, 13x9 board, three starting units, 100-round cap
- Primary reward: zero draw credit, value-delta shaping retained, closing removed
- Confirmation reward: full no-draw shaping
- Algorithm: MaskablePPO / HexCNN with legal-action masking
- Opponent: Random; learner seat alternates
- Training seeds: 11, 23, 47
- Training checkpoint: exactly 51,200 steps
- Evaluation seeds: 4,000,000 through 4,000,024, both seats
- Games: 50 per model, 150 per condition
- Primary metric: engine-declared annihilation wins; draws are non-wins
- Tracking: local and TensorBoard

The default controls reuse the exact 51,200-step models from the reward panel.
All non-default conditions are new training runs. The evaluation seed block is
fresh and was not used in either the pilot or reward panel.

## Conditions

1. Current PPO defaults: learning rate 3e-4, 10 epochs, no target KL.
2. Lower learning rate only: 1e-4, 10 epochs, no target KL.
3. Fewer epochs only: 3e-4, 3 epochs, no target KL.
4. KL limit only: 3e-4, 10 epochs, target KL 0.02.
5. Combined conservative update: 1e-4, 3 epochs, target KL 0.02.
6. Default and combined updates repeated on full no-draw reward as a reward-choice
   confirmation.

## Interpretation

An optimizer change is useful only if it improves annihilation across training
seeds and seats while reducing approximate KL and clip fraction. A prettier
training curve or fewer losses to an opponent that almost never wins is not a
success criterion.

Run from PowerShell with:

```powershell
.\python\winenv\Scripts\python.exe .\python\run_annihilation_optimizer_panel.py
```
