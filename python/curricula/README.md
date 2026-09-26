# Mixed closing and beacon curriculum

One shared tactical-v3 network, initialized from the original combat model.
`closing-beacon-replay-v1.json` declares the saved collections, task weights,
opponents, and reciprocal gameplay panels. No ML Lab selectors or normal-game
difficulty mappings change.

## First stage: replay, not new collection

The first experiment reuses the authenticated closing teacher-v2 collection and
both reach-cell collections. Every optimizer batch has one explicit scenario
identity. Three combat updates alternate with one beacon update, independent of
archive sizes. Small archives cycle through shuffled examples. Epoch length is
the ceiling of total archived training examples divided by batch size; a mixed
epoch is an update budget, not exactly one pass through every example.

Validation is never sampled into training. Each task is evaluated separately.
Weighted validation policy NLL controls early stopping and loss-best checkpoint
selection, **not** publication. The 3:1 mixture is an initial hypothesis.

The historical 4096-expansion/depth-8 teacher archive has its own read-only
compatibility path. Its schema-2 diagnostic sidecar is authenticated against each
decision; it is not confused with the later schema-2 actor-identity format.
The enclosing collection authenticates the historical actor. Existing collection
commands do not gain a new teacher or change behavior.

## Headless command

From this checkout's `python` directory, with the existing Windows Python/CUDA
environment and the matching Release GymServer build:

```text
python -m ml_lab.tactical_v3_curriculum_run --recipe curricula/closing-beacon-replay-v1.json --runs-root <existing-runs-directory> --run-name closing-beacon-seed236-replay-v1 --server <HexWars.GymServer.dll> --preflight
```

Remove `--preflight` to train. `--publish` permits creation of a new sibling
`<run-name>-model` package **only after both gameplay gates pass**. It never
replaces a source model or sets a normal-game difficulty. Without `--publish`,
even a successful candidate remains an experimental checkpoint.

The recipe has a 50-epoch maximum and five-epoch patience, with no wall-clock
deadline. Failed candidates retain their checkpoint, exact-resume state, metrics,
and evaluation evidence. This command is headless; it is not a new ML Lab Train
tab control, and does not attach a Unity training viewer.

## Observability and stop/retry

TensorBoard remains on the existing runs-root log directory. New events live in
`<run>/tensorboard`, including:

- `tasks/combat/steps/train/policy` and `tasks/beacon/steps/train/policy`:
  task-specific training NLL, emitted as batches finish.
- `tasks/<task>/epoch/validation/*`: full validation for that task each epoch.
- `epoch/validation/policy`: the 3:1 weighted diagnostic, not a gameplay score.
- `evaluation/<panel>/<task>/<controller>/wins`: cumulative gameplay wins;
  `evaluation-progress.json` records the corresponding games and seat schedule.

`training/checkpoints/best.pt` is the loss-best candidate. `training/last.pt`
preserves the current model, optimizer, RNGs, epoch history, and exact mixture
fingerprint after each completed epoch. Mixtures use resume format 2; existing
single-scenario resumes remain format 1. Changing weights or collections cannot
silently resume an old trajectory.

The normal run `control.json` stop requests are honored after an epoch's durable
checkpoint, or after a completed evaluation game. In this headless replay path,
both `stop_now` and `stop_after_checkpoint` finish the current epoch. An abrupt
process kill can lose only work since the last completed epoch, provided a first
checkpoint exists. Resume with the same arguments, a **new** run name, and
`--resume-from <stopped-run>`. The old run remains read-only. Evaluation restarts
from complete reciprocal panels rather than treating partial counters as results.

## Gameplay gate and next stage

The development panels reuse the previous 40 games per task, both seats. Each
seat must retain at least the original combat model's wins and win/draw score;
beacon must also improve its total wins over that source. A passing development
candidate must then pass independent 200-game-per-task confirmation panels.
Failed development candidates do not consume confirmation seeds. This is a
conservative screen, not a statistical proof of equivalence. Once confirmation
results influence subsequent choices, reserve fresh confirmation seeds again.

This first slice checks the final weighted-loss-best candidate. It does not
search all epochs for the best gameplay model. If that candidate fails, nothing
is published; inspect per-task behavior and revise the mixture, learning rate,
or checkpoint-selection strategy.

Next: collect fresh learner-controlled games from **both** tasks using their
respective teachers, authenticate and add those collections to the recipe, then
repeat mixed training. Do not return to long, task-exclusive training phases.
