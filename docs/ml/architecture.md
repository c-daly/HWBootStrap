# ML Lab architecture

## What runs where

```text
Unity Editor ML Lab ──launch/status/control──> Python hexwars_ml.py
                                                │
                                                ├── one GymServer process per worker
                                                │     JSONL over stdin/stdout
                                                │
                                                └── atomic run directory/checkpoints
                                                            │
Unity Arena <── PolicyBridge <── policy_server.py <─────────┘
```

Training and evaluation are headless. Unity starts and monitors Python when requested, but Python owns the SB3 learner and launches the pure-C# GymServer workers. The Arena is a separate checkpoint consumer that runs visualization games. It never renders the learner's active training episodes and therefore cannot slow, pause, or corrupt them.

The main responsibilities are:

| Component | Responsibility |
| --- | --- |
| `HexWars.GymServer` | Deterministic tactical environment, observations, legal-action masks, rewards, and game transitions |
| `python/ml_lab` | Strict template/scenario resolution, run lifecycle, controller resolution, SB3 adapters, vector workers, tracking, checkpoints, evaluation, and CLI |
| Unity **ML Lab / Train** | Select or edit a game template, preflight it with the engine contract, launch the same scenario-file CLI path, monitor, stop/resume, reconnect, and open its folder |
| `policy_server.py` + `PolicyBridge` | Load resolved checkpoints and answer inference requests for rendered games |
| Unity **ML Lab / Arena** | Play any two Greedy, Random, fixed-checkpoint, fixed-run, or live-run controllers |

## Terms

- **Algorithm**: the training method, currently supported `maskable_ppo` or experimental `masked_dqn`.
- **Policy/model**: learned parameters plus inference code.
- **Checkpoint**: an immutable, completely written `.zip` snapshot at a known training step.
- **Run**: one immutable experiment configuration and its evolving local artifacts.
- **Learner**: the policy whose weights are being updated.
- **Opponent**: a scripted controller or frozen model used to generate learner experience.
- **Seat**: Player 0 or Player 1. `alternating` changes the learner seat between episodes.
- **Contract**: versioned semantic description of observation/action sizes, board, roster, and rewards.
- **Evaluation**: deterministic held-out games, usually reciprocal across both seats.
- **Promotion**: an explicit decision to turn a validated checkpoint into a named lab candidate. This is not shipment as the official AI.

## Environment and model contract

`doctor` obtains and validates the shape of the GymServer handshake before a run starts. It is an environment health check; it does not compare that handshake with an arbitrary model. `run.json` and `params.json` record the resulting `EnvironmentContract`: version, semantic hash, observation size, action count, board description, roster, and rewards.

Compatibility is intentionally operation-specific. Resume requires the source manifest's complete contract to equal the new training environment, and candidate publication requires evaluation evidence for the exact checkpoint. Evaluation and Arena inference require a supported encoding version plus matching observation/action geometry; they intentionally allow training and duel contract hashes to differ because reward and horizon semantics differ between those environments. The displayed hashes remain provenance for humans and future, stricter promotion gates.

The semantic hash matters because two tensors can have the same length while their channels or actions mean different things. The current duel compatibility check assumes a shared supported encoding version and geometry; it is not proof that every semantic field is identical. Old `game_prototype` zips without a manifest are treated as legacy/unversioned. They may be inspected for a trusted qualitative test if their SB3 class and tensor geometry load, but they are not safe resume or official-AI inputs without an explicit compatibility record.

The current contract is a tactical training scenario, not the entire configurable game. Success here does not prove that a model understands arbitrary map sizes, custom units, economy, setup placement, or every turn rule.

## Scenario resolution and provenance

`python/config/training-game-templates.json` is the checked-in schema-v1 library shared by the CLI and Unity. A new run resolves exactly one input: `--template ID`, `--scenario-file PATH`, or the selected environment's standard template when both are omitted. Template and scenario-file selection are mutually exclusive, and the resolved document's `environment` must equal the CLI environment.

Resolution is deliberately two-stage. Python canonicalizes and strictly validates the document, materializes a temporary copy for a GymServer probe, and compares the authoritative handshake field by field with the requested board, rules, episode horizon, rewards, and adaptive settings. Only after that succeeds does it create the run and atomically write the canonical document to `RUN/scenario.json`. Every real worker receives that run-local path. Training rejects a worker whose resulting contract differs from the probe contract.

The snapshot is provenance, not a settings file. `run.json.scenario` records the relative path `scenario.json`, template ID, and schema version; `run.json.contract` records the handshake-derived observation and action dimensions and semantic identities. The snapshot explains the exact game that produced the experience even if the library entry later changes. Nothing should mutate or redirect it after run creation. To change starting budget, horizon, board, rules, or rewards, edit a source experiment file or a library working copy and create a differently named run.

Promotion consumes an exact checkpoint plus the unmodified manifest, scenario snapshot, evaluation, source commit, and human decision. A mutable template library entry, a Unity working copy, or a `latest` pointer is not sufficient provenance.

## Run directory

By default a run named `example` is written to `python/runs/example/`:

```text
example/
├── run.json                 authoritative intent, state, PID, step, contract, latest checkpoint
├── params.json              immutable resolved configuration and contract
├── scenario.json            immutable canonical game/scenario snapshot
├── control.json             stop request mailbox
├── progress.csv             normalized training metrics
├── monitor.csv              episode reward/length (plus worker-specific monitor files)
├── train.log                durable human-readable log
├── evaluation.json          latest held-out evaluation result
├── checkpoints/             atomically published step_XXXXXXXXX.zip files
└── replays/                 representative games when explicitly saved
```

`run.json` is the status API shared by CLI and Unity. Mutable fields are written atomically. Checkpoints are written to a temporary file, reopened and validated, then renamed into `checkpoints/`; only after that does `latest_checkpoint` advance. A fixed controller resolves one exact snapshot. A live-run controller resolves the newest complete snapshot only at a game boundary, never halfway through a game.

Run folders are intentionally not disposable caches. Keep the manifest, parameters, evaluation, and the checkpoints needed to reproduce a decision. Delete unneeded exploratory runs only after recording their conclusion; never delete or edit a run that is training or currently attached to an Arena.

## Process lifecycle

The Train tab runs the same stable CLI used in a terminal. Output streams are drained asynchronously and status updates are marshalled onto Unity's main thread. The Editor stores enough run/PID attachment information to query the durable manifest after script reload. Closing Unity does not deliberately terminate a headless trainer; reopen the ML Lab and select/query the run to reconnect.

`Stop after checkpoint` asks the trainer to finish the next safe checkpoint and exit. `Stop now` is still a controlled request, but may end before another checkpoint. A failure marks the manifest `failed`, retains the last complete checkpoint, and preserves the log tail. Optional tracker failures mark that tracker degraded and do not change the local training outcome.

## Performance model

Each worker owns a GymServer subprocess and deterministic seed stream. More workers can batch policy inference and environment collection, but they also add CPU/process/serialization overhead. Use `benchmark` with the same machine load and compare measured decisions per second before selecting a count. Do not infer that four workers are faster than one without the measurement.

For maximum speed, close Unity or leave Arena stopped, use the CLI, and choose a measured worker count. JSONL over process pipes is currently the deliberate transport. Replace it only if profiling identifies serialization as the actual bottleneck.
