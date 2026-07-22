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
