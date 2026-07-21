# HexWars ML Experiment Integration Design

## Purpose

HexWars already has a functional tactical RL environment, Stable-Baselines3 trainers, frozen-policy self-play, checkpoint playback, and Unity rendering. The problem is integration: training, watching, evaluation, artifacts, and documentation are split across unrelated menus and scripts, and several interfaces still reflect the older `game_prototype` workflow.

This design makes the existing system coherent without rewriting its learning environment. A developer or intern selects an SB3 algorithm and an opponent model in one Unity Editor window, starts or resumes training, and watches completed checkpoints play in Unity as they are produced. Experimental models remain developer-only. A separate, explicit promotion path will eventually produce the one official player-facing AI.

## Verified current system

The current system has four distinct paths:

1. `AiOpponent` is the shipped gameplay controller and chooses pure-C# Random or Greedy. It does not load trained policies.
2. `TrainingLauncher` starts Windows `python/winenv` in a detached process and invokes either MaskablePPO or experimental masked DQN against Greedy or Random.
3. `TacticalEnv` and `DuelEnv` run inside `HexWars.GymServer`. Python communicates through JSON lines over stdin/stdout. The current contract is a fixed 13×9, three-role tactical scenario with 824 observation values and 1,054 discrete actions.
4. `ModelDuelDriver` and `policy_server.py` render trained models in Unity. Fixed model duels accept a model per seat. Live viewing exposes a checkpoint directory only for PPO in seat 0, even though the policy server can resolve and reload a model file or directory for either seat.

`selfplay.py` trains successive fresh learners against a frozen pool containing Greedy and previous checkpoints. It is not simultaneous two-learner optimization. The current regular trainer always trains one fixed learner seat. A reciprocal read-only evaluation of `run1` versus Greedy produced 44% wins as Player 1 and 6% as Player 2, while Greedy-versus-Greedy was balanced; the absent seat evaluation allowed that failure to pass unnoticed.

## Scope and decomposition

The first implementation project is the **Editor ML Lab** described below. It preserves the current observation encoding, action codec, reward shaping, tactical roster, GymServer process boundary, CNN, SB3 dependency, and model bridge. It may alternate the learner seat and unify launch code, but it does not change what the policy observes or which commands it can represent.

The **official runtime AI** is a follow-on project. It needs its own design once the supported gameplay scenario is chosen, because the current fixed tactical contract cannot honestly control arbitrary 5–64 boards, fog, territory, custom units, different army sizes, and every turn policy. Until a promoted runtime model meets that contract, Greedy remains the official gameplay AI.

## Editor ML Lab

One `HexWars ML` Editor window replaces the disconnected Start Training, Watch Live Training, and ad hoc model-selection flows. It has four sections.

### Learner

- Backend: SB3. The field exists in the run manifest, but SB3 is the only initial value.
- Algorithm: MaskablePPO (supported) or masked DQN (experimental).
- Initialization: fresh policy or a compatible checkpoint to resume.
- Run name, total steps, seed, checkpoint interval, and device preference.
- Learner-seat schedule: alternating by episode is the default. A fixed seat is allowed only as an explicitly labeled diagnostic option.

### Opponent

- Scripted Greedy or Random.
- A fixed trained-model checkpoint.
- A live run directory, resolved to its latest completely published checkpoint at an episode boundary.
- The opponent model may use any algorithm supported by the model adapter. Its type comes from metadata or model inspection, never filename text.

A fixed checkpoint remains frozen for the complete run. A live directory may advance only between episodes. Changing weights during an episode is forbidden.

### Run controls and status

- `Validate` performs the full preflight without starting training.
- `Start & Watch` launches training, waits for the first complete checkpoint, then enters continuous rendered arena play.
- `Start`, `Resume`, `Stop after checkpoint`, and `Watch/Stop watching` are independent controls.
- Status shows Python process state, current step, elapsed time, latest checkpoint, active opponent snapshot, learner seat, recent reward, evaluation W/L/D, and a short log tail.
- Errors appear inline. Editor dialogs are not used for normal validation or status.

The Editor owns the process handle rather than launching an opaque detached `cmd.exe` command. It drains output asynchronously, records the process ID in the run manifest, and can reattach to a still-running process after an assembly reload when the operating system confirms the PID and command line.

### Arena

Each arena seat independently accepts Greedy, Random, a fixed checkpoint, or a live run directory. Both live seats reload between games. The arena displays the resolved checkpoint path, algorithm, training step, schema status, and rolling W/L/D. Pause, speed, unit inspection, and representative replay saving reuse the current Unity presentation.

The first deliverable trains one learner against one selected scripted or trained opponent. Coordinated two-learner training is a later extension: each learner consumes immutable snapshots published by the other between episodes. Two optimizers never share live mutable weights or change an opponent mid-episode.

## Unified Python entry point

A thin Python command, `hexwars_ml.py`, provides stable subcommands over the existing modules:

- `doctor`: environment and contract smoke test.
- `train`: select algorithm and choose the scripted-opponent or frozen-model environment.
- `status`: print one run's state and optionally follow its manifest/log updates.
- `evaluate`: reciprocal held-out evaluation and replay selection.
- `inspect-model`: report algorithm, spaces, policy class, and source metadata.
- `publish-checkpoint`: atomically publish a completed checkpoint and update the run manifest.
- `benchmark`: measure headless reset/step throughput for candidate worker counts.

The existing trainers remain algorithm adapters initially. `train` routes scripted opponents through `HexWarsEnv` and trained opponents through `SelfPlayEnv`. Shared options, logging, manifests, callbacks, seat scheduling, and shutdown move into common modules so PPO and DQN do not silently diverge.

## Headless training and throughput

Training and evaluation remain fully headless. Python drives the pure C# engine inside `HexWars.GymServer`; no Unity process, scene, renderer, animation, or presentation delay participates in an episode. The Unity arena is an optional, independent consumer of completely published checkpoints and plays separate visualization matches. Closing, opening, pausing, or slowing the arena does not pause or pace the trainer.

The initial implementation supports a configurable number of parallel headless environments. Each worker owns an isolated GymServer subprocess and deterministic seed stream; SB3 receives vectorized observations and masks and batches policy work across them. MaskablePPO's multiprocessing contract requires `action_masks()` to live on the environment rather than depend on an `ActionMasker` callback, so the vector wrapper exposes that method directly.

`hexwars_ml.py benchmark` measures resets/second, decisions/second, CPU utilization, payload time, and policy-update time for worker counts from 1 through the configured ceiling. The ML Lab recommends a worker count from measured throughput rather than assuming every machine benefits from the same value. JSON-over-stdio remains the first implementation because it is simple and already works; a framed binary transport is considered only if profiling shows serialization is the limiting stage after vectorization.

The run manifest records worker count and measured throughput. Headless evaluation uses the same worker system and never launches Unity.

## Monitoring a headless run

Monitoring does not require Unity and does not turn training episodes into rendered games.

`hexwars_ml.py status RUN --follow` provides a compact terminal view of process state, completed steps, decisions/second, elapsed time and ETA, recent episode reward/length, algorithm-specific losses, current checkpoint, checkpoint age, and the latest held-out evaluation. It exits nonzero and prints the captured log tail when the trainer fails.

Training emits TensorBoard-compatible metrics under the run directory. An intern can run `tensorboard --logdir python/runs` to inspect reward, episode length, PPO KL/entropy/value loss or DQN loss/exploration rate, throughput, and evaluation curves in a browser. CSV/JSON artifacts remain the source of record; TensorBoard is a view over them rather than the only copy.

External experiment tracking is adapter-based rather than built into individual trainers. A small tracking interface receives normalized run-start, metric, checkpoint, evaluation, replay, failure, and run-end events. The built-in adapters are `local` (always enabled), TensorBoard, and Weights & Biases; later services can be registered without changing PPO, DQN, the evaluator, or the Unity Editor window. The ML Lab and CLI accept zero or more tracker names plus adapter-specific configuration. Secrets come from each service's normal environment or credential store and are never copied into manifests, logs, presets, or Unity project settings.

The local adapter is authoritative and training continues if an optional remote tracker is offline or fails. Remote adapters report a degraded status and retry from locally queued event/artifact metadata where the service supports it. W&B supports its normal online, offline, and disabled modes, with project/entity/group/tags exposed as optional run settings. Tracker integrations must not upload model checkpoints or replays unless artifact upload is explicitly enabled for that adapter.

At a configurable checkpoint cadence, a separate headless evaluator tests the latest complete checkpoint from both seats against the selected opponent and Greedy. It updates `evaluation.json` atomically and never reads a partial checkpoint. Evaluation can be disabled or assigned fewer workers when CPU contention would reduce training throughput. Training reward is never presented as a substitute for reciprocal W/L/D.

Unity may attach later to any local run directory and select `Watch Latest`. The arena consumes published checkpoints and plays separate visualization matches; starting or stopping it does not alter the trainer or evaluator. A run started from the CLI and one started from the ML Lab therefore have the same monitoring and attachment behavior.

## Run and model contracts

Every run directory contains:

- `run.json`: immutable experiment intent plus mutable status/latest-checkpoint fields written atomically.
- `params.json`: the complete GymServer handshake and resolved training arguments.
- `progress.csv` and `monitor.csv`: SB3 and episode metrics.
- `train.log`: structured human-readable log.
- `checkpoints/`: bounded checkpoint retention.
- `evaluation.json`: reciprocal evaluation results and promotion verdict.
- `replays/`: representative held-out wins, losses, and draws.

The manifest records backend, algorithm, policy class, engine commit, ML contract version, observation length, action count, board shape, roster identity, reward/config values, learner-seat schedule, opponent source and resolved snapshot, seeds, dependency versions, and checkpoint step.

New checkpoints are saved to a temporary path, validated by reopening them, then atomically renamed. Only after that rename does `run.json.latest_checkpoint` change. Watchers never select files by modification time while a trainer may still be writing them.

Legacy model zips remain usable if SB3 loads them and their exact observation/action spaces match the current GymServer handshake. They are displayed as `legacy/unversioned` and cannot be promoted to official without an explicit compatibility record.

The ML contract receives a monotonically increasing version. Contract tests hash the semantic channel/action table in addition to checking tensor dimensions, so a model fails clearly when meanings change while sizes remain equal.

## Training and evaluation policy

Training alternates the learner between Player 1 and Player 2 by episode. Because observations and actions are seat-relative, the network shape remains unchanged. Episode seeds are deterministic and recorded.

Evaluation is not inferred from training reward. Each candidate runs a fixed held-out seed suite:

- both seat orientations against Greedy;
- both seat orientations against the selected parent/opponent model;
- both orientations against the previous promoted champion when one exists;
- W/L/D, score, game length, illegal/fallback action count, and inference latency;
- confidence intervals or the exact game count, never a bare percentage without sample size.

The Editor shows evaluation separately from training curves. Promotion requires an explicit human action after the configured gates pass. No “latest checkpoint wins” rule exists.

MaskablePPO evaluation uses the mask-aware SB3 Contrib evaluation path; the library specifically warns that the base SB3 evaluation callback does not apply masks correctly. DQN remains experimental until its masked target calculation and reciprocal results meet the same gates.

## Official AI boundary

Arbitrary checkpoint selection is Editor-only. Regular players never see checkpoint paths, live run directories, experimental algorithms, or unfinished models.

A future promotion command will take one evaluated checkpoint and produce an immutable official artifact containing the inference model, ML contract manifest, evaluation report, and source run/commit. The build references only that artifact. Runtime inference validates output through the existing legal-action mask and `GameEngine.Apply`, and falls back to Greedy if the official artifact cannot load or is incompatible.

For WebGL, Python is training-only. The likely runtime path is exporting the policy to ONNX and using Unity Sentis locally. Unity 6000.5 officially provides Sentis 2.6.1, supports runtime inference across Unity platforms, and imports ONNX; SB3 documents policy export to ONNX. Exact export operators, WebGL latency, build size, and full-game model contract must be proven in the separate official-AI project before a trained policy replaces Greedy.

## Intern-facing documentation

Documentation is a tested deliverable. It includes:

- a diagram and explanation of Unity Editor → Python trainer → GymServer → checkpoint → policy server → Unity arena;
- a glossary distinguishing SB3, algorithm, policy/model, checkpoint, run, learner, opponent, seat, schema, evaluation, and promotion;
- a Windows-first installation matching the actual Editor workflow and a separate WSL/CLI guide;
- an exact `doctor` workflow covering Python, dependencies, Torch/CUDA or CPU fallback, .NET, GymServer build, reset/step, model load, and schema comparison;
- a headless-training section showing how to run with Unity closed, benchmark worker counts, select parallelism, and confirm that the optional arena is not pacing training;
- a monitoring section demonstrating `status --follow`, TensorBoard, reciprocal evaluation files, later Unity attachment, and which metric answers which question;
- a tracker section showing local-only, TensorBoard, and W&B online/offline setup, credential handling, explicit artifact-upload policy, and how to add another tracking adapter;
- checked-in smoke, PPO-versus-Greedy, PPO-versus-model, resume, live-opponent, and reciprocal-evaluation presets;
- an experiment method: hypothesis, one controlled change, train seeds, held-out seeds, watch, evaluate, save representative replays, and promote/reject;
- a run-directory and cleanup reference, including checkpoint retention and disk-size estimates;
- troubleshooting for missing venvs, stale dimensions, semantic schema mismatch, partial checkpoints, process crashes, CPU fallback, bridge loss, and old `game_prototype` models;
- an adapter guide for adding another SB3 algorithm with required mask, save/load, inspection, and evaluation tests.

### Worked experiment

The guide walks through a hypothetical experiment called `ppo_counter_run1`:

1. State the hypothesis: alternating-seat MaskablePPO trained against the frozen `run1` checkpoint will remove the observed Player 2 collapse without losing Player 1 strength.
2. Use `doctor`, then open the ML Lab and select SB3, MaskablePPO, fresh learner, `run1.zip` as opponent, alternating seats, a short smoke budget, and non-overlapping held-out evaluation seeds.
3. Click `Start & Watch`; identify which side is the learner, observe checkpoint reload boundaries, change playback speed, pause, inspect units, and save a replay that exposes a tactical failure.
4. Run reciprocal evaluation, compare per-seat results with the original 44%/6% baseline, and record a promote/reject decision.
5. Open the chosen checkpoint in the Editor arena against Greedy and its parent model to inspect behavior qualitatively.

The guide states the compatibility boundary: this is a rendered tactical scenario, not proof that the checkpoint can control arbitrary full-game setup options. If a human-versus-trained-model challenge tool is added later, it is an Editor-only qualitative test and is not exposed to regular players.

The documented smoke path should reach a watchable checkpoint within approximately 30 minutes after dependencies are installed.

## Error handling

Preflight failures do not start a process. Model load or schema errors name the expected and actual backend, algorithm, observation length, action count, and contract version. A training crash preserves the last complete checkpoint and marks the run `failed` with exit code and log tail. Stop requests finish the current save before shutdown. Watching may disconnect/reconnect without affecting training.

If a model outputs an illegal or invalid action during evaluation, the event is counted and the engine receives a safe End Turn fallback; any such fallback fails official promotion unless explicitly waived for a diagnostic run.

## Verification

Python tests cover manifest atomicity, legacy inspection, algorithm adapters, scripted/model opponent routing, alternating learner seats, vectorized action masks, disjoint worker seed streams, checkpoint retention, complete-file publication, crash/resume, mask-aware inference, reciprocal evaluation accounting, tracker event normalization, optional-tracker failure isolation, secret redaction, explicit artifact-upload consent, and arena independence from trainer pacing.

Engine contract tests pin observation channels, globals, action regions, mask/decode round trips, GymServer handshake, and semantic contract hash. The current deployed-reinforcement slot limitation is recorded as a known tactical-contract constraint rather than silently treated as full-game support.

Unity EditMode tests cover ML Lab validation, process argument construction, run discovery, status parsing, fixed/live seat specifications, reload boundaries, and incompatibility messages. Manual integration tests start a smoke run, watch its first and later checkpoints, stop/restart only the viewer, resume the trainer, render two live directories, and verify representative replay output.

No official AI artifact is created or shipped by this first project.

## References

- Unity Sentis 2.6.1 overview: https://docs.unity3d.com/Packages/com.unity.ai.inference@2.6/manual/index.html
- Stable-Baselines3 ONNX export guidance: https://stable-baselines3.readthedocs.io/en/v2.7.1/guide/export.html
- SB3 Contrib MaskablePPO evaluation requirements: https://sb3-contrib.readthedocs.io/en/v2.7.1/modules/ppo_mask.html
