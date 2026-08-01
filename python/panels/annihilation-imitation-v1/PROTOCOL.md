# Annihilation imitation v1 protocol

This directory preregisters demonstration collection, three independent pure behavioral clones and their gate, paired initialized/scratch PPO training, development evaluation, one global PPO budget, and a single-use final evaluation.

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
Final selection uses the same snapshot binding: controller metadata must match
kind, path, algorithm, and step, while source identity comes from the canonical
top-level source run and checkpoint-under-source/checkpoints relationship.

    python python/run_annihilation_imitation_panel.py train-ppo
    python python/run_annihilation_imitation_panel.py evaluate-dev
    python python/run_annihilation_imitation_panel.py select-budget


## Final seal, evaluation, and report

`freeze-final --incumbent-panel <path>` is legal only after the one global
checkpoint budget is complete. The atomic `final-seal.json` records the code
revision and dirty state; hashes of the panel, scenario, and still-immutable
seed-bank definitions; every dataset file; all six PPO run trees; sealed clone
gate and BC metric sources; three initialized checkpoints; three scratch
controls; and exactly three completed profiled-standard incumbent runs. The
incumbents are read from the production conversion aggregate's seed-keyed
`models.profiled_standard` entries (seeds 101, 113, and 127), then mapped to
explicit pairing slots 211, 223, and 227. Each fixed incumbent snapshot binds
the aggregate checkpoint digest to the completed run's `config.seed`,
algorithm, environment, contract hash, standard-only profile distribution,
checkpoint path, and checkpoint step. The seal's assignment snapshot flips
`final.assigned` to true without rewriting `seed-banks.json`, preserving all
earlier definition hashes. There is no unassign operation and a second freeze
is refused.

`evaluate-final` spends the sealed bank once. Each initialized model receives
maps 17,000,000 through 17,000,249 from both candidate seats against Random
with forced `standard-3v3`: exactly 500 games per model and 1,500 primary games
total. The seal hash is captured after initial validation. Immediately after
the last game and before publishing `final-evaluation.json`, the seal file must
still have that exact hash and every sealed file plus repository identity is
validated again. Publication is refused for missing or duplicated
seed/seat/model keys, either missing seat, a seed outside the bank, any
mid-evaluation sealed-input change, or an already completed final evaluation.

The preregistered primary gate passes only at both thresholds:

- at least 325 wins out of 500 for each of seeds 211, 223, and 227;
- at least 1,050 wins out of 1,500 pooled.

Every draw and loss remains a non-win regardless of material advantage or draw
classification. W/L/D counts and rates, Wilson 95-percent intervals, seat
summaries, rounds, decisions, action waste, peak material advantage, and draw
categories are recomputed from raw match rows. Scratch and incumbent
comparisons pair by model seed, map seed, and candidate seat and use the exact
two-sided sign test on discordant wins.

`report` places the primary gate table before every secondary result, followed
by clone, initialized PPO, scratch PPO, incumbent PPO, conversion, BC, learning
curve, compute, failure-trace, and limitation sections. Clone, conversion, BC,
learning, and compute values are derived from hash-sealed artifacts. Learning
curves contain the pooled standard win rate at every locked nominal budget for
both initialized and scratch PPO. Empty conversion evidence is an error rather
than a reported 0/0 result. Wall-clock timing is reported only when every
authoritative run records it; otherwise the aggregate and report explicitly
mark that timing unavailable. Comparator loss/draw intervals, seat summaries,
diagnostics, and draw categories come from the same aggregate as the Markdown
report.

Before aggregation, `report` requires a schema-v1 completed
`final-evaluation.json` whose schedule exactly equals the locked final
schedule and whose seal hash equals the current, fully revalidated seal.
The same captured seal is checked again before the publication pointer swap.

Publication writes `aggregate.json` and `REPORT.md` into an immutable
content-addressed directory under `.final-generations`. Readers first open the
single atomically replaced `final-publication.json` pointer and verify both
generation hashes. A failure while staging either file or before the pointer
swap leaves the prior reader-visible generation intact.

    python python/run_annihilation_imitation_panel.py freeze-final --incumbent-panel <path>
    python python/run_annihilation_imitation_panel.py evaluate-final
    python python/run_annihilation_imitation_panel.py report
