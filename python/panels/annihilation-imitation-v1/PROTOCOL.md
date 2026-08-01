# Annihilation imitation v1 protocol

This directory preregisters demonstration collection, three independent pure behavioral clones and their gate, paired initialized/scratch PPO training, development evaluation, and one global PPO budget. It does not assign the final bank or authorize final evaluation.

Definitions are immutable and hash-bound. Every command validates `panel.json`, `seed-banks.json`, and the scenario before doing work. Stages resume in a sibling staging directory and publish by atomic rename only after manifests, hashes, counts, reciprocal schedules, and expected outputs validate.

## Locked data and models

- Greedy standard demonstrations: seeds 11,000,000?11,499,999; at least 100,000 retained decisions.
- Bounded-search conversion demonstrations: seeds 11,500,000?11,999,999; at least 50,000 retained decisions.
- BC validation: seeds 12,000,000?12,099,999; 100 standard reciprocal map pairs and 20 reciprocal map pairs for each of six near/far conversion profiles.
- Model seeds: 211, 223, 227. The sampler seed is recorded separately for each clone, even though this trainer's reviewed interface intentionally binds it to the model seed.
- Development: exactly seeds 16,000,000?16,000,099, both candidate seats, Random opponent, forced standard-3v3 profile.
- Final: seeds 17,000,000?17,000,249 remain unassigned.

The scenario samples 70% standard starts and 30% near/far conversions in basis points. Medium conversion profiles remain declared with zero weight. Authoritative outcomes are win +1, loss -1, draw 0. Shaping is 0.01, step penalty 0.005, closing 0, draw credit 0, and points 0.5.

## Commands

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py validate
python python/run_annihilation_imitation_panel.py collect
python python/run_annihilation_imitation_panel.py train-bc
python python/run_annihilation_imitation_panel.py evaluate-bc
```

A clone passes only when each seed wins at least 60 of 200 games and the panel wins at least 240 of 600. Any missing/duplicate seed-seat record, contract mismatch, changed definition provenance, missing loss/draw trace, or missing loss/draw replay fails the stage regardless of rates.


## Locked PPO panel

For model seeds 211, 223, and 227, train-ppo creates an actor-initialized
bc_ppo run and a scratch_ppo control. Each pair shares every online setting
and episode namespace; only the run identity and actor_init_source differ.
Episode seed bases are 13,000,000, 14,000,000, and 15,000,000 respectively.
Every run uses fresh MaskablePPO/HexCNN construction, Random, alternating seats,
51,200 requested timesteps, 12,800 checkpoint intervals, learning rate 0.0003,
10 epochs, and target KL 0.02. Resume sources are forbidden.

Published checkpoints must be strictly increasing, unique completed rollout
boundaries. Nominal budgets 12,800, 25,600, and 51,200 map to the first actual
published step at or beyond each nominal value. Both values and the checkpoint
hash are evidence.

evaluate-dev evaluates all 21 candidates: three pure clones plus both PPO
conditions for three seeds at three budgets. Every candidate receives the same
100 maps beginning at 16,000,000, both seats, Random, and forced standard-3v3.
Each of the 4,200 game records retains condition, model seed, nominal and actual
checkpoint, map seed, candidate seat, outcome, trace, and replay.

select-budget requires the complete development schedule and atomically writes
selection.json. It chooses one nominal budget for all three initialized PPO
seeds by pooled standard win rate, then higher worst-seed standard win rate,
higher pooled conversion win rate when conversion evidence exists, lower draw
rate, and earlier nominal budget. The output records each seed's rollout-aligned
actual step plus definition, development-table, and candidate-checkpoint hashes.

    python python/run_annihilation_imitation_panel.py train-ppo
    python python/run_annihilation_imitation_panel.py evaluate-dev
    python python/run_annihilation_imitation_panel.py select-budget
