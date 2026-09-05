# Annihilation conversion curriculum protocol

## Question

Can target-KL PPO learn to convert material advantages into annihilation wins
against Random when 60% of its episodes provide mechanically valid late-game
pursuit practice, without sacrificing ordinary 3-v-3 play?

This is a diagnostic tactical-v2 experiment. It is not the tactical-v3 frozen
policy tournament and cannot promote a production AI.

## Frozen contract

- Board and rules: tactical-v2, 13x9, 100-round cap.
- Geometry: three controllable slots for every start profile.
- Placement policy: `profiled-seeded-v1`.
- Profile catalog: the exact ten entries in `panel.json`.
- Separation: closest opposing pair at distance 2-3 for near, 4-6 for medium,
  and at least 7 for far. `standard-3v3` alone uses the legacy mirrored start.
- Algorithm: MaskablePPO / HexCNN, learning rate 3e-4, ten nominal epochs,
  target KL 0.02.
- Reward: zero draw credit, zero closing reward, retained value delta and current
  per-command step penalty.
- Opponent: Random, with alternating learner seat.
- Training seeds: 101, 113, and 127.
- Training horizon: 51,200 rollout-aligned environment steps with four workers.

Both conditions declare the same profile catalog and therefore must have the
same model contract and encoding hashes. Only immutable scenario weights differ.
All weights are integer basis points and sum to 10,000.

The standard control samples `standard-3v3` at 10,000 basis points. The mixed
condition samples standard at 4,000 and each near/far conversion profile at
1,000. Medium profiles have zero training weight and are held-out interpolation
tests.

## Seed discipline

`seed-banks.json` is authority. The 6,000,000 bank is for implementation and
smoke testing. Locked standard and conversion evaluation seeds must not be used
until all compatibility and smoke gates pass. The entire 10,000,000-10,099,999
namespace remains untouched unless the mixed curriculum passes every panel gate.

Every evaluation map is played reciprocally. The requested profile is relative
to the candidate seat in each game; it is never implicitly relative to player 0.
Paired comparisons use training seed, map seed, profile, and candidate seat.

## Outcomes and evidence

The primary metric is engine-declared annihilation wins. A draw is never a win.
Report wins, losses, draws, conversion success, cycling and failed-conversion
draws, rounds to win, final health-adjusted advantage, action waste, and both
seat directions. Report environment steps, optimizer updates, completed episode
counts by profile, PPO KL/clip diagnostics, wall-clock training time, and
evaluation inference time.

Retain replay-backed evidence for every conversion draw, plus stratified standard
wins and Greedy controls. Evaluation rows must record forced profile, candidate
seat, map seed, scenario hash, contract hash, and encoding hash.

## Pass gate

The mixed condition passes only if every clause holds:

1. Aggregate conversion win rate improves by at least 15 percentage points.
2. Total conversion wins improve for each training seed.
3. Cycling-draw incidence falls by at least 25% relative.
4. Held-out medium-separation conversion improves.
5. Standard 3-v-3 win rate drops no more than five percentage points overall.
6. No training seed drops more than 15 percentage points on standard 3-v-3.
7. Loss rate rises no more than five percentage points.
8. Both candidate seats show the effect.

If the gate passes, continue mixed models to 100,352 steps and assign seeds from
the reserved confirmation namespace before consuming any of them. If it fails,
do not tune weights against the locked results. Diagnose the failed clause and
proceed to the gated algorithm-decision work in the authority plan.

## Stop conditions before training

Do not start the panel if legacy starts or hashes change, a profile can create an
invalid or wrong-seat state, profile selection is nondeterministic, provenance is
missing, control and mixed contracts differ, CUDA smoke artifacts are incomplete,
or any locked evaluation seed has been consumed during development.

Content hashes for all motivating reports, aggregates, source checkpoints, run
manifests, and source scenarios are frozen in `panel.json`.
