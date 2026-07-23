# ML Lab troubleshooting

Start with the structured preflight from the same interpreter and repository clone you intend to use:

```powershell
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor
```

In Unity, **HexWars > ML Lab > Train > Doctor** presents the same checks inline.

## Setup and startup

### Python or package check fails

Confirm that Unity's configured interpreter exists, then reinstall into that interpreter—not a different global Python:

```powershell
py -3 -m venv .\python\winenv
.\python\winenv\Scripts\python.exe -m pip install -r .\python\requirements.txt
```

In WSL, activate `.venv` and invoke `python`, not the Windows `python/winenv` executable.

### .NET or GymServer handshake fails

Install the repository's supported .NET SDK and rebuild Release:

```powershell
dotnet --version
dotnet build .\engine\HexWars.GymServer -c Release
```

If using `--server`, pass the exact built `.dll`. The default is `engine/HexWars.GymServer/bin/Release/net8.0/HexWars.GymServer.dll`. GymServer uses stdio; opening firewall ports or changing localhost settings cannot repair this path.

### Run directory is not writable or already exists

The tool deliberately never overwrites a run. Pick a new valid name or pass `--runs-root` to a writable experiment directory. A valid name starts with a letter/number, is at most 64 characters, and uses only letters, numbers, `.`, `_`, and `-`.

### Template or scenario validation fails

`--template` and `--scenario-file` are mutually exclusive. A template ID must exist in `python/config/training-game-templates.json`, and the selected document's `environment` must match `--environment`. A scenario file is one complete schema-v1 scenario object, not the entire library wrapper. Validation is strict: missing keys, extra keys, non-finite numbers, overlapping deployment zones, an impossible adaptive starting budget, or a non-positive `episode.max_steps` all stop launch.

Start from a known-good entry in the library, copy it to a new experiment file, and change only the hypothesis fields. For an adaptive budget/horizon experiment those are normally `adaptive.starting_army_budget` and `episode.max_steps`. Give the copy a distinct `id` and `name`, then validate it with a unique 1,000-timestep, one-worker smoke run before allocating the full budget.

### GymServer reports different scenario values

Python compares the requested scenario with the authoritative GymServer handshake before creating the durable run, including the scenario identity/schema, board, rules, horizon, rewards, adaptive settings, and resulting contract dimensions. A mismatch usually means the selected server binary is stale or `--server` points to a different build. Rebuild Release, confirm the repository/server path, and rerun the smoke; do not weaken validation.

```powershell
dotnet build .\engine\HexWars.GymServer\HexWars.GymServer.csproj -c Release
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run scenario-smoke-new-name --environment adaptive-v1 --scenario-file .\experiments\counter-artillery.json --algorithm maskable_ppo --opponent random --timesteps 1000 --checkpoint-every 500 --workers 1 --learner-seat alternating --tracker local
```

### A completed run has the wrong budget or horizon

Do not edit `python/runs/RUN/scenario.json` or redirect `run.json.scenario.path`. The snapshot is immutable provenance, and every worker was launched against it. Preserve the bad smoke as evidence long enough to record the finding, fix the source experiment file or template, and launch a new run name. Before promotion, verify that `run.json.scenario.path` is `scenario.json`, its template ID/schema version match that file, and the recorded observation/action dimensions match the handshake for the snapshot.

In Unity, reopen **HexWars > ML Lab > Train**, choose the environment/template, expand **Advanced game settings**, correct **Starting army budget** or **Max steps**, give the revision a new name/ID, and choose **Save as template**. Read **Training preflight**, run **Doctor**, and choose **Start & Watch** with a new smoke name. The Arena view is a separate checkpoint consumer; it cannot repair or redefine the training run.

## Training and status

### Status does not advance

Read `run.json.state`, `pid`, `latest_message`, `updated_at`, and the tail of `train.log`. Check operating-system CPU activity for that PID. A stale Editor panel can be refreshed/reconnected without restarting training. If the manifest is `failed`, preserve the last complete checkpoint and fix the logged root cause before starting a new run.

### Unity recompiled or closed during training

The run is durable and training is headless. Reopen the ML Lab and query/select the run directory. A live operating-system PID plus advancing manifest indicates the trainer survived. Arena attachment is optional and may be restarted independently.

### Stop appears delayed

`Stop after checkpoint` waits for a safe publication boundary. Its latency depends on rollout/update and checkpoint cadence. Use `Stop now` for the faster controlled request, understanding that it may not create another checkpoint. Do not kill GymServer workers merely because safe stop is finishing a rollout.

### Multiple workers are slower

This can be correct. Each worker adds a .NET process, pipe traffic, and CPU contention. Compare:

```powershell
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py benchmark --games 20 --workers 1
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py benchmark --games 20 --workers 2
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py benchmark --games 20 --workers 4
```

Use the best measured decisions/second under representative load. Leave CPU capacity for evaluation or Arena viewing if they run concurrently.

### CUDA is unavailable

CUDA is optional. `doctor` reports it as a non-required capability and training can use CPU. Verify that Torch matches the installed driver before selecting a CUDA device. For small networks/environment-heavy runs, GPU transfer overhead may not improve throughput; benchmark rather than assuming.

## Models, contracts, and resumes

### Observation/action or contract mismatch

Do not bypass the error. Inspect both the model and current server:

```powershell
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py inspect-model run:.\python\runs\MODEL_RUN
.\python\winenv\Scripts\python.exe .\python\hexwars_ml.py doctor
```

`doctor` and `inspect-model` report separate environment/model identities; they do not perform an all-fields comparison. Resume requires an exact source/current training contract. Arena and evaluation accept a supported encoding version with matching observation/action geometry and intentionally permit different reward/horizon hashes between training and duel environments. A hash difference therefore requires interpretation: expected reward/horizon differences may be viewable, while changed channel/action meaning needs a reviewed compatibility/migration path. Official promotion must apply stricter semantic evidence than the developer viewer.

### Old `game_prototype` model will not load

A standalone legacy zip may omit algorithm and contract metadata. Try an explicit trusted `ppo:PATH` or `dqn:PATH` inspection only for diagnosis. Do not rename a file to imply an algorithm, edit another run's manifest, or resume/promote an unversioned model. If the old environment semantics are still needed, reproduce them in an isolated compatibility branch and generate authoritative evaluation evidence.

### Resume rejected

Resume accepts a metadata-backed run/checkpoint whose declared algorithm and complete semantic contract match. It creates a new run. Normal ML Lab/CLI manifests use `timestep_mode: "absolute"`, so `--timesteps` is the final target. Legacy compatibility wrappers wrote `timestep_mode: "additional"`; resume preserves that field and interprets the value as extra steps. Inspect old `run.json` files before choosing a target. `masked_dqn` resume is intentionally unavailable until replay-buffer sidecars are persisted. A standalone trusted legacy resume is supported only by older explicit unsafe paths, not the normal ML Lab workflow.

### Watcher sees an old checkpoint

Check `run.json.latest_checkpoint` and `latest_checkpoint_step`. Live runs reload only between complete Arena games; they never switch weights mid-game. A fixed run/checkpoint is expected never to advance. Do not select a temporary/partial file by modification time.

## Arena and bridge

### Arena starts but a controller is unresolved

Use the displayed structured error and resolved seat metadata. Verify the path, `run.json`, latest checkpoint, explicit algorithm for a fixed zip, and contract. The Python policy server's stderr tail is surfaced in the ML Lab. Fix the model/path rather than falling back silently to filename guessing.

### Policy bridge disconnects or times out

Stop the Arena, ensure no orphaned policy server owns the selected files, then launch again. Bridge startup is asynchronous and bounded; a large first model load can take time but should not freeze Unity. Training is a separate process and should continue while the viewer reconnects.

### Watching changes training speed

Arena games are separate from training games, so playback pacing is not directly coupled. They still share CPU/GPU resources, however. Close/stop Arena and compare measured steps/second. For maximum headless throughput, train with Unity closed.

## Trackers and artifacts

### W&B or custom tracker is degraded

Local data remains authoritative. Read `run.json.tracker_status`, verify the module/credentials/service outside the run, and preserve `progress.csv`. Secrets belong in the provider's credential store or environment; tracker configuration containing token/secret/password/API-key fields is rejected.

### Disk usage grows

Checkpoints and optional replay/artifact mirrors dominate storage. Never delete files from a running or viewed run. After a decision, keep the evaluated checkpoint, manifests, metrics, logs, evaluation, and representative replays; prune redundant intermediate checkpoints only under an explicit retention policy. Do not commit generated `python/runs` content to source control.

## Reporting a reproducible problem

Include the code commit, OS, Python/.NET versions, exact command or Unity fields, `doctor` result, run name, `run.json` (after checking it contains no organization-sensitive paths), relevant `train.log` tail, controller identities, checkpoint step, and contract hash. For gameplay anomalies include the evaluation seed, seat orientation, and replay. Never attach credential stores or service tokens.
