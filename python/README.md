# HexWars ML Lab

HexWars training is a headless Stable-Baselines3 workflow with an optional Unity Editor control and viewing surface. Python launches one isolated `.NET` GymServer process per worker and exchanges newline-delimited JSON over stdin/stdout. Unity is not part of the training loop, so rendering, pausing an Arena match, or closing the Editor does not pace training.

The supported algorithm is `maskable_ppo`; `masked_dqn` is available for experiments and is explicitly marked experimental. Models produced here remain developer artifacts until they pass a separate official-AI promotion and runtime-packaging process.

## First setup

### Windows (recommended with the Unity ML Lab)

From the repository root in PowerShell:

```powershell
dotnet build .\engine\HexWars.GymServer -c Release
py -3 -m venv .\python\winenv
.\python\winenv\Scripts\python.exe -m pip install -r .\python\requirements.txt
.\python\winenv\Scripts\python.exe -m pip install tensorboard  # optional dashboard
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor
```

Open Unity and choose **HexWars > ML Lab**. The **Train** tab exposes the same configuration as the CLI and can validate, start, start-and-watch, resume, stop, and reconnect to a local run. The **Arena** tab can render two independently selected scripted, fixed, or live-run controllers.

### WSL2 / Ubuntu (fast headless CLI work)

Clone the repository into the WSL native filesystem rather than `/mnt/c`, then:

```bash
sudo apt-get update
sudo apt-get install -y dotnet-sdk-8.0 python3-venv
dotnet build engine/HexWars.GymServer -c Release
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r python/requirements.txt
python -m pip install tensorboard  # optional dashboard
python python/hexwars_ml.py doctor
```

There is no Windows/WSL networking requirement. Python launches the Linux GymServer itself and communicates over pipes. A run created in a WSL clone is not automatically visible to the Windows Unity project; copy the complete, stopped run directory if you want to inspect it there.

## Common commands

The examples below assume the Windows interpreter. In an activated WSL environment replace the first line with `python`.

```powershell
$hexwarsPython = ".\python\winenv\Scripts\python.exe"

# Check Python, SB3, GymServer, contract handshake, run-directory writes, and trackers.
& $hexwarsPython .\python\hexwars_ml.py doctor --tracker tensorboard --tracker wandb

# Measure headless engine throughput before selecting a worker count.
& $hexwarsPython .\python\hexwars_ml.py benchmark --games 20 --workers 1
& $hexwarsPython .\python\hexwars_ml.py benchmark --games 20 --workers 4

# Train against Greedy. Local records are always authoritative.
& $hexwarsPython .\python\hexwars_ml.py train --run ppo_greedy_seed11 --algorithm maskable_ppo --opponent greedy --timesteps 200000 --checkpoint-every 25000 --workers 4 --seed 11 --learner-seat alternating --tracker local --tracker tensorboard

# Observe a run without Unity.
& $hexwarsPython .\python\hexwars_ml.py status .\python\runs\ppo_greedy_seed11 --follow

# Request a safe stop after the next complete checkpoint.
& $hexwarsPython .\python\hexwars_ml.py stop .\python\runs\ppo_greedy_seed11 --after-checkpoint

# Resume a normal ML Lab run as a new, auditable run; its target is absolute.
& $hexwarsPython .\python\hexwars_ml.py resume .\python\runs\ppo_greedy_seed11 --run ppo_greedy_seed11_resume --timesteps 400000

# Reciprocal held-out evaluation: 30 seeds in each seat orientation, 60 games total.
& $hexwarsPython .\python\hexwars_ml.py evaluate --p0 run:.\python\runs\ppo_greedy_seed11 --p1 greedy --games 30 --seed-start 1000000 --both-seats --workers 4 --output .\python\runs\ppo_greedy_seed11\evaluation.json
```

Run names are 1–64 characters, start with a letter or number, and contain only letters, numbers, `.`, `_`, or `-`. Existing run directories are never overwritten.

Controller specifications accepted by evaluation and model tooling are `greedy`, `random`, and `run:PATH`. Standalone checkpoints and legacy `ppo:PATH` / `dqn:PATH` sources are rejected because they lack authoritative contract metadata.

## Tactical-v2 baseline evidence

In a fresh PowerShell session, first expose this checkout's package with `$env:PYTHONPATH = (Resolve-Path '.\python').Path`.

Run a reciprocal candidate baseline from the repository root in PowerShell:

```powershell
.\python\winenv\Scripts\python.exe -m ml_lab.cli evaluate --p0 run:python\runs\candidate --p1 greedy --games 500 --both-seats --workers 4 --environment tactical-v2 --evidence-dir python\evidence\candidate-vs-greedy
```

`--environment tactical-v2` is required for a scripted-only matchup to select the tactical-v2 contract explicitly. Evidence capture is opt-in: `--evidence-dir` implies `--capture-trace`, while `--capture-trace` by itself performs the same in-memory analysis without persisting trace or replay files. Existing evaluate commands without either flag retain their ordinary match schema and W/L/D arithmetic.

The command writes the normal `evaluation.json` to the candidate run directory when `--p0` is `run:PATH` and `--output` is omitted; `--output` can select a different report path. The report keeps the authoritative wins, losses, draws, rates, confidence intervals, seat totals, and per-match outcomes, and adds per-match summaries/classifications plus `evidence.draw_traces`, `evidence.control_traces`, and the `evidence.draw_categories` category totals.

With `--both-seats`, each held-out seed is paired: the candidate plays that seed once as player 0 and once as player 1. The schedule and filenames remain in seed/candidate-seat order even with multiple workers. Under the evidence directory, `traces\*.json` records accepted tactical transitions and `replays\*.replay` records the same game's portable engine replay; each retained pair shares a stem containing match index, seed, and candidate seat.

All draws are retained, including draws whose evidence is ambiguous. Non-draw controls use a deterministic rule: retain the first win and first loss encountered for each candidate seat, when those strata exist. Thus category totals describe draws only, while the controls provide reproducible comparisons against decisive games.

Treat classifications as diagnostic quarantine, not promotion arithmetic. A draw can carry multiple supported flags, but only the precedence-selected primary category contributes to `draw_categories`; evidence that supports no category remains `unclassified`. None of these labels, summaries, traces, or controls changes the reported winner, W/L/D totals, rates, confidence intervals, observations, masks, or rewards.

Retained traces are validated as a continuous before/command/after chain, and retained replays can reconstruct the reported final outcome through the engine replay reader. Repeating the same fixed-seed command produces the same ordered matches, retained filenames, summaries, classifications, and replay contents; only the report's `generated_at` timestamp is expected to differ.


## Training game templates and custom scenarios

Choose a checked-in template with `--template`, or pass a complete schema-v1 experiment file with `--scenario-file`; the two options are mutually exclusive. The selected scenario's `environment` must match `--environment`. If neither option is supplied, training selects that environment's `tactical-standard` or `adaptive-standard` template.

For a reviewed library template, run:

```powershell
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train `
  --run adaptive_large_seed31 `
  --environment adaptive-v1 `
  --template adaptive-large-battle `
  --algorithm maskable_ppo `
  --opponent greedy `
  --learner-seat alternating `
  --workers 4 `
  --timesteps 300000
```

For a one-off experiment file, run:

```powershell
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train `
  --run custom_scenario_seed32 `
  --environment adaptive-v1 `
  --scenario-file .\experiments\counter-artillery.json `
  --algorithm maskable_ppo `
  --opponent random `
  --learner-seat 1 `
  --workers 2 `
  --timesteps 100000
```

An intern should copy one complete entry from [`python/config/training-game-templates.json`](config/training-game-templates.json) into a standalone JSON object such as `experiments/counter-artillery.json`, give it a distinct `id` and `name`, and change only the intended fields. For an adaptive experiment, `adaptive.starting_army_budget` controls the private deployment budget and `episode.max_steps` is the training horizon. Keep the copied `environment` aligned with the CLI selection and retain every schema-v1 field; strict validation rejects missing or extra keys.

Before a full budget, use a new run name and `--timesteps 1000 --checkpoint-every 500 --workers 1 --tracker local`. After it completes, inspect both `python/runs/RUN/run.json` and `python/runs/RUN/scenario.json`: `run.json.scenario.path` must be `scenario.json`, the manifest's template ID/schema version must match the snapshot, and the contract observation/action dimensions must be the dimensions reported by the GymServer handshake for that snapshot.

`scenario.json` is immutable experiment provenance. Python writes its canonical snapshot before workers start and passes that run-local file to every worker. Do not edit it to tune or repair a run, and do not edit `run.json` to point elsewhere. Change the source experiment file or library entry, then launch a new smoke/full run under a new name. Promote only the exact reviewed checkpoint together with its unmodified `run.json`, `scenario.json`, evaluation, source commit, and approval.

The Unity equivalent is **HexWars > ML Lab > Train**: select **Environment** and **Template**, expand **Advanced game settings**, change the starting army budget and max steps if the hypothesis requires them, enter a new save-as template name/ID, and choose **Save as template**. Set the run, algorithm, opponent, learner seat, workers, timesteps, and checkpoint interval; review **Training preflight**, choose **Doctor**, then **Start & Watch**. Unity writes the edited working copy to a run-specific staging file and launches the same CLI with `--scenario-file`. Use **Open run folder** to inspect the immutable snapshot, and use the Arena only after a complete checkpoint exists. Watching is a separate rendered match and does not display or pace the learner's headless episodes.

### Tactical-v3 scenario training

Tactical-v3 uses the same Train-tab template workflow, but separates two choices that have different jobs:

- **Source model** supplies the initial learned weights. Select a completed structured model (or a continuation that resolves to its published model) and choose **Use selected model**.
- **Template** is the target game configuration used for new DAgger collection and training. **Standard** matches the current custom-only tactical-v3 scenario; **Full Roster** is the checked-in five-template scenario. Advanced settings can change the board, match rules, roster, and start-profile weights. The weights must total 10,000 basis points.

The source model does not force its old scenario onto the new run. ML Lab first checks package metadata against the target's encoding and capacity while allowing the target's full match/contract hash to differ. When you start, a foreground preflight asks the separate GymServer to authenticate the target scenario, loads and authenticates the complete source checkpoint package, verifies its model architecture, and does the same for any model opponent. Only a successful preflight starts the detached trainer and creates run artifacts. A failed preflight remains in the ML Lab command log and can be retried with the same run name.

For a new scenario, select **Tactical v3**, choose the closest template, edit only the intended fields, give a saved template an ID beginning with `tactical-v3-`, then select the source model and opponent. Fixed and live model opponents can use the same **Use selected model** lifecycle resolution as the source. A compatible tactical-v3 opponent may use its own network architecture; only the initialization source must match the continuation architecture exactly. Save advanced edits if they must survive closing ML Lab or a script reload. Leave TensorBoard enabled if you want live collection and epoch metrics, and use **Start & Watch** when you also want the separate Arena viewer. The run snapshots the edited target as its own `scenario.json` and records the source model's immutable identity and architecture separately; neither source artifact is overwritten.

The direct CLI equivalent accepts an explicit standalone target scenario:

```powershell
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train-structured `
  --run new_scenario_vs_greedy `
  --source-run .\python\runs\SOURCE_MODEL `
  --scenario-file .\python\config\annihilation-structured-imitation-v1.json `
  --opponent greedy `
  --train-labels 7500 `
  --validation-labels 3000 `
  --learner-seat alternating `
  --device cuda `
  --tracker local `
  --tracker tensorboard
```

Collection follows the target scenario's positive start-profile weights. With alternating seats, each selected profile is used for a reciprocal seat pair before the scheduler advances, so changing the scenario mix changes what is collected without changing the model selector or opponent controls.

### Beacon curriculum

The existing Tactical v3 template list now includes **Reach Beacon 1v1**
(`tactical-v3-reach-cell-v1`). Start with **Passive** as the opponent and
alternating learner seats. The flat 4-by-4 scenario asks the learner to reach a
seeded, initially unoccupied destination within eight rounds; killing the other
unit is not a substitute for reaching the beacon. The target is fixed for the
episode and carried in the existing move-candidate target reference, so model
tensor shapes and encoding/capacity hashes remain compatible.

The optional `tactical_v3.objective` configuration specifies `kind: reach_cell`,
`target_policy: seeded_farthest_reachable_unoccupied_v1`, and `radius: 0`.
Omitting it preserves the existing annihilation objective and legacy contract
hashes. Change board size, round limits, roster, or start distribution through
the existing template workflow; no new model-selection controls are needed.

For teacher-guided continuation, use a compatible source model with the existing
`train-structured` command and the saved reach scenario. Its shortest-path
teacher labels every unique visited learner decision with
`curriculum_reach_cell`; the learner still chooses the actions during collection.
Teacher depth and actual search expansions are zero because this objective uses
pathfinding, not the annihilation search teacher. The existing `train-outcome`
path also carries the objective through sampling, optimization, and checkpoint
validation, allowing later scratch or initialized outcome-learning experiments.

Judge a beacon pilot on held-out success rates **for each seat**, steps to reach
the goal, and repeated movement, not just teacher NLL. Then test retention on the
original closing scenarios before mixing beacon practice with approach/finish
tasks. A published beacon candidate is not automatically a replacement for the
best full-game model. Collection and epoch metrics use the existing TensorBoard
trackers; the learner's headless games remain separate from the Arena viewer.

## Tactical v2 rosters

`tactical-v2` is the default environment for a new tactical experiment: `train` resolves an omitted `--environment` to `tactical-v2`, not the legacy contract. `tactical-v1` remains supported as a frozen legacy contract; its templates, checkpoints, and encoding are compatible only with other tactical-v1 artifacts, never with tactical-v2. `doctor` and `benchmark` intentionally keep defaulting to `--environment tactical-v1` — they are environment diagnostics, not new-run launchers, so an existing invocation without an explicit flag keeps checking the contract it always has. Pass `--environment tactical-v2` explicitly to doctor-check or benchmark the new contract.

A tactical-v2 scenario's `tactical_v2` section carries its own roster instead of an army budget: `starting_unit_count` (1–12; `max_controllable_units` must equal it), `placement_policy` (`symmetric-random-v1`), and an explicit `templates` catalog — each entry a stable `id`, `name`, and full nine-field stat block. In **HexWars > ML Lab > Train**, selecting the tactical-v2 environment shows a **Tactical setup** box with a **Roster source** popup (local player 1 or 2) and a **Starting unit count** slider (1–12). **Refresh saved roster** rebuilds the working scenario's template list from that seat's five canonical barracks defaults plus any of that seat's saved custom designs, deduplicated against the defaults. A hand-authored `--scenario-file` has no live-roster equivalent; it must already list the exact templates it wants.

Whatever roster and count are current when a run starts — defaults, a refreshed saved-roster snapshot, or a hand-edited template list — are snapshotted verbatim into that run's `scenario.json` at launch, the same way board/rules/reward are. Nothing about a tactical-v2 run re-reads the live saved-roster cache or template library afterward: a resume reads its source run's `scenario.json`, and Arena/Start & Watch read the selected run's `scenario.json`, never a player's currently saved roster. Editing barracks designs after a run starts cannot change what that run trains or replays.

Each episode samples `starting_unit_count` templates **with replacement** from the roster, seeded from the episode seed, so the same seed always draws the same starting army for both seats — sampling is symmetric, not independent per side, and a small roster can and will repeat a template within one army. Across a representative seed set every roster entry, including a custom template, should appear in some sampled episode.

The roster and count are part of the contract identity: changing either the template list or `starting_unit_count` changes `encoding_hash`, so a checkpoint trained against one roster is not resume- or Arena-compatible with a differently rostered run, even at the same unit count. Treat a roster change like any other scenario change — a new run name, not an edit to an existing one.

`adaptive-v1` is unrelated to this roster mechanism: it keeps its own learned hidden-deployment flow and its own 1–24 `starting_unit_count`/budget knobs, unaffected by tactical-v2's roster source or sampling.

Every current run manifest records `environment`, `version`, and a lowercase SHA-256 `encoding_hash`. Inference and resume require all three values to match the selected engine environment exactly; observation/action geometry is checked as a separate guard. `contract_hash` remains the full run-contract identity and can legitimately differ between training and duel horizons, while `encoding_hash` covers only the semantics that determine what observations and actions mean. Manifests created before `encoding_hash` was introduced are intentionally rejected rather than guessed compatible. The Unity arena passes its engine-derived expected identity to `policy_server.py`, and a rejected live reload leaves the previously validated model active.

Runs created by `hexwars_ml.py` use absolute timestep targets. A manifest created by a legacy compatibility trainer can declare `config.timestep_mode: "additional"`; the resume command preserves that declaration, so inspect it before resuming an old run.

## Adaptive v1 intern experiment

Run this smoke experiment from the repository root. It deliberately selects the adaptive contract; omitting `--environment` selects the legacy `tactical-v1` contract. Do not mix checkpoints between those contracts.

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj -c Release
python python/hexwars_ml.py doctor --environment adaptive-v1 --json
python python/hexwars_ml.py train --run adaptive-smoke --environment adaptive-v1 --algorithm maskable_ppo --opponent greedy --timesteps 50000 --checkpoint-every 10000 --workers 4 --learner-seat alternating --tracker local
python python/hexwars_ml.py status python/runs/adaptive-smoke --json
python python/duel.py --environment adaptive-v1 --p0 run:python/runs/adaptive-smoke --p1 greedy --out replays/adaptive-smoke.replay
```

This is the concrete scenario the experiment exercises: the policy privately selects six affordable units and deployment cells, then confirms its hidden setup. During play it may redesign Custom A as a high-vision counter, deploy that template after earning enough points, and control the reinforcement through the lowest free unit slot. Python never reconstructs deployment, design, movement, attack, cost, fog, or turn legality; it uses the fixed action space and authoritative masks reported by the adaptive GymServer.

Adaptive training writes the exact diagnostic header to `adaptive_episodes.csv` for one worker, or to `adaptive_episodes.worker_N.csv` for multiple workers. Episode identities are globally unambiguous `worker:episode` values. Evaluation consumes worker files in numeric worker order and keeps these diagnostics separate from W-L-D and win rate.

In Unity, open **HexWars > ML Lab**, select **Adaptive v1**, choose `adaptive-smoke` as P1, and choose Greedy or another compatible adaptive run as P2. Start the Arena only after checking the resolved checkpoint identities and W-L-D summary. Hidden deployment is intentionally not rendered before both seats confirm. A run in progress, including `adaptive-smoke`, is an unfinished lab artifact. Regular players should receive only a separately reviewed and promoted official adaptive checkpoint through the existing AI-opponent packaging path; never expose a live or unfinished run as official game content.

## Where to go next

- [Architecture](../docs/ml/architecture.md): process boundaries, model contracts, and run layout.
- [Experiment guide](../docs/ml/experiment-guide.md): an intern-ready method and a complete `ppo_counter_run1` example.
- [Trackers](../docs/ml/trackers.md): local, TensorBoard, W&B, and custom adapters.
- [Troubleshooting](../docs/ml/troubleshooting.md): setup, process, model, and bridge failures.
- [Official AI boundary](../docs/ml/official-ai-boundary.md): why lab models are not selectable by regular players.

Legacy scripts remain for compatibility, but new experiments should use `hexwars_ml.py` so manifests, contracts, checkpoints, stop/resume behavior, evaluation, and Unity attachment remain consistent.
