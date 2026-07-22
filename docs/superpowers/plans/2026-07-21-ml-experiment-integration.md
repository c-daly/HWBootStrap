# HexWars ML Experiment Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn HexWars' existing GymServer, SB3 trainers, and Unity model-duel viewer into one reliable workflow where an experimenter can choose an algorithm and opponent, launch fast headless training, monitor it, and watch either seat use a fixed model or a live run.

**Architecture:** Keep `HexWars.GymServer` and the existing tactical observation/action contract as the authoritative environment. Add a thin Python orchestration package behind a single `hexwars_ml.py` CLI, with atomic run manifests, reusable controller/model resolution, tracker adapters, evaluation, and process control. Replace the detached Unity training launcher with an Editor-owned ML Lab that launches that CLI, polls its run files, and hands resolved seat specs to the existing `ModelDuelDriver`/`PolicyBridge` path. Keep unfinished experiments editor-only; shipping an official runtime model is explicitly outside this plan.

**Tech Stack:** Unity 6 Editor tooling and NUnit, C#/.NET 8 GymServer and NUnit, Python 3.10+, Gymnasium, Stable-Baselines3, sb3-contrib MaskablePPO, optional Weights & Biases, JSON/JSONL, CSV, and stdio subprocesses.

**Global constraints:**

- Preserve the current tactical board, roster, rewards, action masking, and JSONL GymServer protocol.
- Store all experiment truth locally under `python/runs/<run>/`; remote trackers are mirrors and may fail without stopping training.
- Treat Unity ML Lab control and visibility as a release-blocking requirement: a user must be able to configure, launch, watch progress/logs/metrics, stop, resume, and open a rendered live duel without leaving the Unity UI.
- Resolve model algorithms from run/checkpoint metadata; never infer an algorithm from a filename.
- Reload live-run opponents only between episodes or rendered games.
- Keep raw experiments and model selection out of player-facing builds and menus.
- Write checkpoints to temporary names, verify reload compatibility, and atomically publish them.
- Do not stage Unity-generated or pre-existing worktree changes.

---

## Task 1: Establish the Python test harness and atomic run contract

**Files:**

- Create: `python/tests/__init__.py`
- Create: `python/tests/test_run_contract.py`
- Create: `python/ml_lab/__init__.py`
- Create: `python/ml_lab/io.py`
- Create: `python/ml_lab/contracts.py`
- Modify: `python/requirements.txt`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing contract tests**

Cover run-name validation, atomic JSON writes, creation of the standard run tree, immutable contract fields, PID/control state, and atomic latest-checkpoint publication:

```python
def test_create_run_writes_complete_manifest(tmp_path):
    config = RunConfig(
        backend="stable_baselines3", algorithm="maskable_ppo", policy="hex_cnn",
        run_name="ppo_counter_run1", seed=17, total_timesteps=1_000,
        checkpoint_interval=100, workers=1, device="cpu",
        learner_seat="alternating", opponent={"kind": "scripted", "name": "greedy"},
        trackers=[{"kind": "local"}], resume_source=None,
    )
    run = create_run(tmp_path, config, contract)
    manifest = read_json(run / "run.json")
    assert manifest["state"] == "created"
    assert manifest["contract"]["observation_size"] == contract.observation_size
    assert (run / "checkpoints").is_dir()

def test_publish_checkpoint_rejects_incompatible_model(tmp_path):
    with pytest.raises(ContractMismatch):
        publish_checkpoint(
            source=tmp_path / "pending.zip",
            run_dir=tmp_path / "runs" / "counter",
            step=100,
            expected_contract=contract,
            inspector=lambda _: incompatible_model_info,
        )
```

- [ ] **Step 2: Run tests and confirm the module is missing**

Run: `python -m pytest python/tests/test_run_contract.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'ml_lab'`.

- [ ] **Step 3: Implement contract dataclasses and atomic file helpers**

`RunConfig` must include backend, algorithm, policy, run name, seed, total timesteps, checkpoint interval, worker count, device, learner-seat schedule, opponent spec, tracker specs, and resume source. `EnvironmentContract` must include semantic version/hash, observation/action sizes, board dimensions, roster, and reward configuration.

Create the standard tree and files:

```text
run.json
params.json
progress.csv
monitor.csv
train.log
control.json
checkpoints/
evaluation.json
replays/
```

- [ ] **Step 4: Add test-only dependency and ignore local experiment outputs**

Add `pytest>=8.0` to `python/requirements.txt`; ignore `python/runs/`, `python/winenv/`, TensorBoard event files, W&B local state, and generated replay files while retaining source fixtures.

- [ ] **Step 5: Run the focused tests**

Run: `python -m pytest python/tests/test_run_contract.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add .gitignore python/requirements.txt python/ml_lab python/tests
git commit -m "feat(ml): add atomic experiment run contract"
```

---

## Task 2: Make the GymServer handshake a versioned semantic contract

**Files:**

- Create: `engine/HexWars.Engine/Rl/MlContract.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`
- Create: `engine/HexWars.Engine.Tests/MlContractTests.cs`
- Modify: `python/hexwars_gym/env.py`
- Create: `python/tests/test_gym_client.py`

- [ ] **Step 1: Add failing .NET contract tests**

Assert that the contract is deterministic, includes the current `TacticalLayout` dimensions/action count, and changes hash when any semantic field changes.

- [ ] **Step 2: Run the focused .NET tests**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter MlContractTests`

Expected: FAIL because `MlContract` does not exist.

- [ ] **Step 3: Implement and expose the handshake**

Add `contract_version`, `contract_hash`, `board`, `roster`, and `reward` to the existing `spaces` response without removing existing keys. Hash canonical UTF-8 JSON using SHA-256.

- [ ] **Step 4: Add a failing Python client test**

Use a fake JSONL server process and assert `HexWarsEnv.contract` rejects missing or inconsistent semantic fields with a useful error.

- [ ] **Step 5: Implement client-side contract parsing**

Retain `spaces_info` for compatibility and expose a typed environment contract used by all new runs.

- [ ] **Step 6: Verify both suites**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "MlContractTests|TacticalEnvTests|DuelEnvTests"`

Run: `python -m pytest python/tests/test_gym_client.py -q`

Expected: PASS.

- [ ] **Step 7: Rebuild the Unity engine plugin and commit**

Run: `powershell -ExecutionPolicy Bypass -File engine/build-to-unity.ps1`

```bash
git add engine/HexWars.Engine/Rl/MlContract.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/MlContractTests.cs python/hexwars_gym/env.py python/tests/test_gym_client.py
git commit -m "feat(ml): version the tactical environment contract"
```

---

## Task 3: Unify controller and model resolution

**Files:**

- Create: `python/ml_lab/controllers.py`
- Create: `python/tests/test_controllers.py`
- Modify: `python/duel.py`
- Modify: `python/policy_server.py`
- Modify: `python/selfplay_env.py`

- [ ] **Step 1: Write failing resolver tests**

Cover `greedy`, `random`, a fixed `.zip`, a run directory, a checkpoints directory, missing metadata, incompatible contracts, and live resolution advancing only after `reload()`.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest python/tests/test_controllers.py -q`

Expected: FAIL because the resolver does not exist.

- [ ] **Step 3: Implement one controller specification**

Use explicit JSON-compatible specs:

```json
{"kind":"scripted","name":"greedy"}
{"kind":"checkpoint","path":"C:/HexWars/python/models/model.zip","algorithm":"maskable_ppo"}
{"kind":"run","path":"C:/HexWars/python/runs/run1","mode":"fixed|live"}
```

Keep parsing legacy `ppo:PATH` and `dqn:PATH` strings at the boundary, but normalize immediately. Load model type and contract from metadata and verify observation/action spaces before inference.

- [ ] **Step 4: Route duel, policy server, and self-play through the resolver**

`policy_server.py` must report each seat's resolved checkpoint, algorithm, timestep, and contract hash in its ready/reload replies. Directory reloads occur only on the explicit reload command.

- [ ] **Step 5: Run controller and legacy smoke tests**

Run: `python -m pytest python/tests/test_controllers.py -q`

Run: `python python/duel.py --help`

Expected: PASS and help lists both JSON/file/run inputs.

- [ ] **Step 6: Commit**

```bash
git add python/ml_lab/controllers.py python/tests/test_controllers.py python/duel.py python/policy_server.py python/selfplay_env.py
git commit -m "refactor(ml): unify scripted and trained controllers"
```

---

## Task 4: Add local, TensorBoard, W&B, and arbitrary tracker adapters

**Files:**

- Create: `python/ml_lab/tracking.py`
- Create: `python/tests/test_tracking.py`

- [ ] **Step 1: Write failing tracker tests**

Assert that local CSV is always enabled, optional tracker failures are recorded as degraded state without raising, W&B is lazy-imported, secrets are not serialized, and `module:function` adapters receive normalized metric events.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest python/tests/test_tracking.py -q`

- [ ] **Step 3: Implement tracker lifecycle and SB3 callback bridge**

Define `start_run`, `log_metrics`, `log_artifact`, and `finish`. TensorBoard uses SB3's logger output. W&B imports only when selected and takes project/entity/mode from non-secret config; authentication stays in environment or the user's W&B config. Custom adapters load through `importlib`.

- [ ] **Step 4: Verify graceful degradation**

Run: `python -m pytest python/tests/test_tracking.py -q`

Expected: PASS, including a deliberately failing fake remote tracker.

- [ ] **Step 5: Commit**

```bash
git add python/ml_lab/tracking.py python/tests/test_tracking.py
git commit -m "feat(ml): add extensible experiment tracking"
```

---

## Task 5: Build the unified headless training runner

**Files:**

- Create: `python/ml_lab/envs.py`
- Create: `python/ml_lab/algorithms.py`
- Create: `python/ml_lab/training.py`
- Create: `python/ml_lab/callbacks.py`
- Create: `python/hexwars_ml.py`
- Create: `python/ml_lab/cli.py`
- Create: `python/tests/test_algorithms.py`
- Create: `python/tests/test_training.py`
- Modify: `python/train_maskable_ppo.py`
- Modify: `python/train_dqn.py`
- Modify: `python/selfplay.py`

- [ ] **Step 1: Write failing algorithm and orchestration tests**

Use fake environments/models to assert algorithm selection, fresh/resume behavior, direct `action_masks()` availability through vector workers, deterministic worker seed streams, alternating learner seats, checkpoint validation/publication, status updates, and stop-after-checkpoint control.

- [ ] **Step 2: Run and confirm failure**

Run: `python -m pytest python/tests/test_algorithms.py python/tests/test_training.py -q`

- [ ] **Step 3: Implement algorithm adapters**

Support `maskable_ppo` as the verified default and `masked_dqn` as experimental. Centralize model create/load/predict/save/validate behavior and policy selection (`HexCNN` for PPO). Reject resume when algorithm or contract differs.

- [ ] **Step 4: Implement environment factories and workers**

Each worker owns its own GymServer process. Seeds follow `base_seed + worker_index + episode_index * worker_count`. Scripted and fixed/live model opponents share the controller resolver. Alternating learner seats is the default; fixed seat is available for diagnosis.

- [ ] **Step 5: Implement the training callback**

At rollout/checkpoint boundaries update `run.json` and `progress.csv`, publish reload-validated checkpoints atomically, invoke trackers, evaluate control state, and flush logs. On interruption mark the run stopped/failed without corrupting `latest`.

- [ ] **Step 6: Convert legacy scripts to compatibility wrappers**

Keep their existing command lines working, print a deprecation note, and call the unified runner instead of retaining three divergent training loops.

Add the initial `hexwars_ml.py train` command here so the real smoke run exercises the same entry point Unity will launch. The remaining operational subcommands are added in Task 6.

- [ ] **Step 7: Verify unit tests and a short real headless run**

Run: `python -m pytest python/tests/test_algorithms.py python/tests/test_training.py -q`

Run: `python python/hexwars_ml.py train --run smoke_ppo --algorithm maskable_ppo --opponent greedy --timesteps 64 --checkpoint-every 32 --workers 1 --device cpu`

Expected: run reaches `completed`, a reloadable checkpoint is published, and all GymServer child processes exit.

- [ ] **Step 8: Commit**

```bash
git add python/hexwars_ml.py python/ml_lab python/tests python/train_maskable_ppo.py python/train_dqn.py python/selfplay.py
git commit -m "feat(ml): unify headless SB3 training"
```

---

## Task 6: Add doctor, status/control, inspect, evaluate, benchmark, and publish commands

**Files:**

- Modify: `python/hexwars_ml.py`
- Modify: `python/ml_lab/cli.py`
- Create: `python/ml_lab/doctor.py`
- Create: `python/ml_lab/evaluation.py`
- Create: `python/ml_lab/benchmark.py`
- Create: `python/tests/test_cli.py`
- Create: `python/tests/test_evaluation.py`
- Modify: `python/winrate.py`

- [ ] **Step 1: Write failing CLI/evaluation tests**

Cover `doctor`, `train`, `resume`, `status --follow`, `stop --after-checkpoint`, `inspect-model`, `evaluate`, `benchmark`, and `publish-checkpoint`. Evaluation must use held-out deterministic seeds, both seat assignments, action masks, W/L/D totals, confidence intervals, opponent/checkpoint identity, and atomic `evaluation.json` writes.

- [ ] **Step 2: Implement the command surface**

All commands print human-readable output by default and one stable JSON object with `--json`. `doctor` validates Python packages, .NET, GymServer build/handshake, optional CUDA, optional trackers, and write access. `status --follow` tails local truth without requiring Unity or a remote service.

- [ ] **Step 3: Implement evaluation and benchmark**

Benchmark reports resets/s, decisions/s, elapsed time, CPU count, worker count, and protocol payload sizes. Evaluation reuses one server per worker and runs reciprocal seats for every seed.

- [ ] **Step 4: Implement explicit checkpoint publication**

Promotion within the lab means marking a compatible checkpoint as a named candidate artifact; it does not place it in a player build. Include evaluation evidence and source run identity in the candidate manifest.

- [ ] **Step 5: Verify commands**

Run: `python -m pytest python/tests/test_cli.py python/tests/test_evaluation.py -q`

Run: `python python/hexwars_ml.py doctor --json`

Run: `python python/hexwars_ml.py inspect-model python/runs/smoke_ppo --json`

Run: `python python/hexwars_ml.py evaluate --p0 python/runs/smoke_ppo --p1 greedy --games 4 --both-seats --json`

Expected: PASS and valid JSON from every command.

- [ ] **Step 6: Commit**

```bash
git add python/hexwars_ml.py python/ml_lab python/tests python/winrate.py
git commit -m "feat(ml): add experiment CLI and evaluation tools"
```

---

## Task 7: Add testable Unity Editor process and run models

**Files:**

- Create: `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- Create: `Assets/HexWars/Editor/MlLab/MlRunStatus.cs`
- Create: `Assets/HexWars/Editor/MlLab/MlCliProcess.cs`
- Create matching `.meta` files
- Modify: `Assets/HexWars/Editor/HexWars.Presentation.Editor.asmdef`
- Modify: `Assets/HexWars/Tests/Editor/HexWars.Presentation.Tests.asmdef`
- Create: `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`
- Create: `Assets/HexWars/Tests/Editor/MlRunStatusTests.cs`
- Create: `Assets/HexWars/Tests/Editor/MlCliProcessTests.cs`
- Create matching `.meta` files

- [ ] **Step 1: Write failing Editor tests**

Assert validation rules, exact argument quoting, algorithm/opponent/tracker serialization, JSON status parsing, bounded log retention, async stdout/stderr draining, process-exit state, PID reattachment metadata, and stop command construction. Do not launch Python in unit tests; inject a process adapter.

- [ ] **Step 2: Run focused EditMode tests**

Run Unity batchmode with `-testFilter 'HexWars.Presentation.Tests.Ml'`.

Expected: FAIL because the ML Lab classes do not exist.

- [ ] **Step 3: Implement the Editor-side models and process owner**

Use `System.Diagnostics.Process` with redirected asynchronous readers. Persist only safe UI preferences and active run path/PID in `SessionState`; never persist tracker secrets. On assembly reload, query `hexwars_ml.py status --json` and reattach monitoring even if the original `Process` object is gone.

- [ ] **Step 4: Run focused EditMode tests**

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add Assets/HexWars/Editor/MlLab Assets/HexWars/Editor/HexWars.Presentation.Editor.asmdef Assets/HexWars/Tests/Editor/MlLab* Assets/HexWars/Tests/Editor/HexWars.Presentation.Tests.asmdef
git commit -m "feat(editor): add ML Lab process foundation"
```

---

## Task 8: Replace the detached launcher with the Unity ML Lab

**Files:**

- Create: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Create matching `.meta`
- Modify: `Assets/HexWars/Editor/TrainingLauncher.cs`
- Modify: `Assets/HexWars/Editor/ReplayViewerMenu.cs`
- Create: `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`
- Create matching `.meta`

- [ ] **Step 1: Write failing state-transition tests**

Cover idle, validating, running, stopping, completed, failed, and externally-running states; form errors remain inline and no modal is required for ordinary use.

- [ ] **Step 2: Implement the Train tab**

Fields: run name; SB3 algorithm (`MaskablePPO`, experimental masked DQN); fresh/resume; timesteps; seed; checkpoint interval; workers; device; alternating/fixed learner seat; opponent kind and path; local/TensorBoard/W&B/custom trackers. Buttons: Doctor, Start, Start & Watch, Resume, Stop After Checkpoint, Stop Now, Open Run Folder. Show state, PID, step/target, elapsed time, latest checkpoint, latest evaluation, throughput, tracker degradation, and a bounded live log.

The window must remain the usable control surface after domain reload or Editor restart: scan known runs, restore the selected run, query its status, reconnect log/status polling, and issue control commands through the CLI even when Unity did not create the current Python process.

- [ ] **Step 3: Implement Start & Watch**

Launch training first, wait asynchronously for the first validated checkpoint, then launch the model duel with a live run spec. Never block `OnGUI` waiting for Python or model load.

- [ ] **Step 4: Retire the old menu without breaking discoverability**

Keep `HexWars/Start Training...` as an alias that opens the ML Lab. Add `HexWars/ML Lab` as the primary menu. Remove the detached `cmd.exe` and success/failure popup flow.

- [ ] **Step 5: Run Editor tests and compile check**

Run focused EditMode tests, then all `HexWars.Presentation.Tests` EditMode tests. Confirm zero Unity compile errors.

- [ ] **Step 6: Commit**

```bash
git add Assets/HexWars/Editor/MlLab Assets/HexWars/Editor/TrainingLauncher.cs Assets/HexWars/Editor/ReplayViewerMenu.cs Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs*
git commit -m "feat(editor): add integrated HexWars ML Lab"
```

---

## Task 9: Make arbitrary fixed/live model duels transparent and inspectable

**Files:**

- Modify: `Assets/HexWars/Presentation/PolicyBridge.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Create: `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`
- Create: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- Create matching `.meta` files

- [ ] **Step 1: Write failing bridge/configuration tests**

Test structured ready/action/reload/error messages, resolved model metadata for both seats, scripted/fixed/live combinations, and reload only between games.

- [ ] **Step 2: Replace substring/manual integer parsing with structured protocol DTOs**

The bridge exposes seat algorithm, checkpoint path, timestep, and contract hash. Errors include Python stderr tail and never leave a child process orphaned.

- [ ] **Step 3: Add the Arena tab**

Each seat independently selects Greedy, Random, fixed checkpoint, or live run. Show resolved model information before launch. Controls: launch, pause/resume, seconds per action, loop, and stop. During play show seed, winner tally, current seat, and the checkpoint used by each seat. Live runs reload between games only.

- [ ] **Step 4: Verify in EditMode and PlayMode**

Run bridge/configuration tests. Manually launch fixed-vs-fixed, fixed-vs-Greedy, and live-vs-live; confirm fog/game rules remain owned by `DuelEnv` and that both live seats advance only at game boundaries.

- [ ] **Step 5: Commit**

```bash
git add Assets/HexWars/Presentation/PolicyBridge.cs Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs* Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs*
git commit -m "feat(ml): add inspectable arbitrary model arena"
```

---

## Task 10: Write intern-grade experiment documentation

**Files:**

- Replace: `python/README.md`
- Create: `docs/ml/architecture.md`
- Create: `docs/ml/experiment-guide.md`
- Create: `docs/ml/trackers.md`
- Create: `docs/ml/troubleshooting.md`
- Create: `docs/ml/official-ai-boundary.md`

- [ ] **Step 1: Document setup and system boundaries**

Explain Unity Editor vs Python vs GymServer responsibilities, Windows and WSL setup, `doctor`, headless performance, contract compatibility, run layout, checkpoints, fixed vs live opponents, and why experiments are not player-selectable.

- [ ] **Step 2: Document a complete hypothetical experiment**

Use `ppo_counter_run1`: inspect `run1`; define the hypothesis that alternating-seat MaskablePPO can remove its known seat bias; configure reciprocal evaluation; launch headlessly or through ML Lab; monitor locally/TensorBoard/W&B; watch it against `run1`; interpret confidence-aware W/L/D results; and publish or reject a candidate. Include exact CLI and Editor steps and explain how that candidate may be used in the Editor game viewer without becoming the official player-facing AI.

- [ ] **Step 3: Document custom tracker and algorithm adapters**

Provide complete minimal adapter examples with signatures and tests, plus experiment naming, seeds, baseline selection, cleanup, and reproducibility guidance.

- [ ] **Step 4: Validate every command in the docs**

Run each `--help`, `doctor`, status, inspect, short train, evaluation, and benchmark example or its documented smoke-size equivalent.

- [ ] **Step 5: Commit**

```bash
git add python/README.md docs/ml
git commit -m "docs(ml): add experiment setup and operations guide"
```

---

## Task 11: End-to-end verification and delivery

**Files:**

- Modify only files required by failures found below.

- [ ] **Step 1: Run all Python tests**

Run: `python -m pytest python/tests -q`

Expected: PASS with no leaked GymServer/Python child processes.

- [ ] **Step 2: Run all engine tests**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

Expected: PASS.

- [ ] **Step 3: Run Unity EditMode tests and compile check**

Run all `HexWars.Presentation.Tests`; inspect Unity logs; confirm zero compile errors.

- [ ] **Step 4: Run an end-to-end smoke experiment**

From a clean temporary run name: doctor, train MaskablePPO headlessly, follow status, stop after checkpoint, resume, reciprocal evaluation, inspect, watch in Unity, and verify a live checkpoint reload between games.

Repeat the operational flow using only Unity ML Lab controls: configure and launch, observe live step/throughput/log updates, close and reopen the window, reconnect to the run, stop after a checkpoint, resume, and start the rendered viewer. This flow is required for completion; CLI-only success is insufficient.

- [ ] **Step 5: Benchmark one and multiple workers**

Record the commands and results in `docs/ml/experiment-guide.md`; do not claim multiple workers are faster unless the measured result demonstrates it.

- [ ] **Step 6: Check repository hygiene**

Run: `git diff --check`

Run: `rg -n "TODO|FIXME|placeholder|coming soon" python/ml_lab python/hexwars_ml.py docs/ml Assets/HexWars/Editor/MlLab`

Expected: no accidental placeholders and no generated runs, model archives, credentials, Unity caches, or unrelated worktree changes staged.

- [ ] **Step 7: Request code review, address findings, and re-run verification**

- [ ] **Step 8: Commit any verification fixes and push**

```bash
git push origin codex/qol-improvements
```

Expected: the branch advances on GitHub; no WebGL rebuild is required because this project adds editor/developer tooling, not a player-facing runtime feature.

---

## Deferred follow-on: official game AI

The official player-facing AI requires a separate design and implementation project: candidate promotion policy, deterministic export (likely ONNX), Unity Sentis/WebGL compatibility, runtime inference performance, model packaging/versioning, Greedy fallback, product difficulty behavior, and release validation. Nothing in this plan exposes arbitrary experiment checkpoints to regular players.
