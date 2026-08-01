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
10 epochs, and target KL 0.02. The production trainer caps each worker rollout
at 512 steps, so four workers publish on 2,048-step rollout boundaries. Resume
sources are forbidden.

Published checkpoints must be strictly increasing, unique completed rollout
boundaries. Nominal budgets 12,800, 25,600, and 51,200 map to the first actual
published step at or beyond each nominal value: 14,336, 26,624, and 51,200.
Both values, the canonical checkpoint/source/controller identity, and a digest
recomputed from the physical checkpoint are evidence.

Training uses a deterministic per-run .pending destination. An incomplete,
failed, stopped, or running pending run is scoped-cleaned and retrained without
resume; a completed pending run is validated and atomically published. A
bc_ppo run is accepted only when its full actor-only initializer provenance
matches the physical seed-specific clone checkpoint, fixtures, run manifest,
BC metadata, dataset manifest, contract, and encoding identities.

evaluate-dev evaluates all 21 candidates: three pure clones plus both PPO
conditions for three seeds at three budgets. Every candidate receives the same
100 maps beginning at 16,000,000, both seats, Random, and forced standard-3v3.
Each of the 4,200 game records retains condition, model seed, nominal and actual
checkpoint, checkpoint digest, controller and opponent identity, profile, map
seed, candidate seat, outcome, trace, and replay. Validation reopens every
physical per-map evaluation.json, verifies its controller/opponent/schedule,
and reconstructs the aggregate rather than trusting copied summary rows.
The validator accepts the full production controller metadata shape while
requiring the snapshot path, algorithm, and step; snapshot source identity is
bound through the canonical candidate snapshot specification and checkpoint
location. All 100 manifests are reopened for each of the 21 candidates.

select-budget requires the complete development schedule and atomically writes
selection.json. It chooses one nominal budget for all three initialized PPO
seeds by pooled standard win rate, then higher worst-seed standard win rate,
higher pooled conversion win rate when conversion evidence exists, lower draw
rate, and earlier nominal budget. The output records each seed's rollout-aligned
actual step plus definition and development-table hashes. Candidate checkpoint
hashes are recomputed immediately before publication from the canonical
physical checkpoint paths.

    python python/run_annihilation_imitation_panel.py train-ppo
    python python/run_annihilation_imitation_panel.py evaluate-dev
    python python/run_annihilation_imitation_panel.py select-budget
