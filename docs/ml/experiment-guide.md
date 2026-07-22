# Running HexWars ML experiments

This guide is written for an intern who can use a terminal and the Unity Editor but has not worked on this repository before. Complete the setup in [`python/README.md`](../../python/README.md), then use this method for every experiment.

## The experiment method

1. Write one falsifiable hypothesis and name the baseline it should beat.
2. Change one important variable at a time: algorithm, opponent, reward, seed group, or schedule.
3. Run `doctor` and inspect every model input before spending compute.
4. Choose training seeds now; reserve a non-overlapping held-out seed range for evaluation.
5. Start a short smoke run, confirm steps/checkpoints/metrics, then start the full budget under a new name.
6. Monitor local state first. Use TensorBoard or W&B as views, not as the only record.
7. Evaluate from both seats against the baseline and parent model.
8. Watch representative wins, losses, and draws in the Unity Arena. Qualitative play is diagnosis, not a win-rate substitute.
9. Record the result, confidence interval, limitations, code commit, and a promote/reject decision.

A useful name includes the algorithm, opponent or hypothesis, and seed, for example `ppo_counter_run1_s1701`. Never reuse a run name: the tool refuses to overwrite an existing directory. Keep experiment and held-out seeds disjoint; the default evaluation range begins at `1000000`.

## Local headless benchmark record

The following Task 11 verification was recorded on 2026-07-22 on the Windows development host reported by `benchmark` as 16 logical CPUs. Each command used 20 games and held the seed range and engine build constant:

```powershell
& $hexwarsPython .\python\hexwars_ml.py benchmark --games 20 --workers 1 --json
& $hexwarsPython .\python\hexwars_ml.py benchmark --games 20 --workers 2 --json
& $hexwarsPython .\python\hexwars_ml.py benchmark --games 20 --workers 4 --json
```

| Workers | Elapsed | Resets/second | Decisions/second |
| ---: | ---: | ---: | ---: |
| 1 | 3.250 s | 6.15 | 357.22 |
| 2 | 2.290 s | 8.73 | 506.90 |
| 4 | 1.827 s | 10.95 | 635.50 |

Four workers were fastest in this short local engine benchmark. That is evidence for this host and workload, not a universal default; repeat the commands after hardware, engine, policy, or background-load changes. A full training run also includes model-update cost and can have a different optimum.

## Worked scenario: `ppo_counter_run1`

### 1. Question and hypothesis

Assume `python/runs/run1` is an older model that appears strong from Player 0 but weak from Player 1. The experiment is:

> Alternating-seat MaskablePPO trained against a frozen `run1` snapshot will reduce seat bias without sacrificing its stronger orientation.

The controlled change is the alternating learner-seat schedule. The parent model remains frozen for the entire training run. The evidence will be reciprocal held-out W/L/D against `run1`, with results separated by orientation.

### 2. Inspect and validate

In PowerShell from the repository root:

```powershell
$hexwarsPython = ".\python\winenv\Scripts\python.exe"
& $hexwarsPython .\python\hexwars_ml.py doctor --tracker tensorboard
& $hexwarsPython .\python\hexwars_ml.py inspect-model run:.\python\runs\run1
```

Stop if `doctor` reports a missing dependency, GymServer failure, or unwritable run root. For `run1`, record the resolved algorithm, checkpoint path, step, observation size, action count, encoding version, and contract hash. `doctor` and `inspect-model` are separate checks and do not compare the current server's hash with the model. Resume later enforces an exact training-contract match; Arena/evaluation compatibility enforces supported encoding version and tensor geometry while permitting expected reward/horizon hash differences. Do not infer an algorithm from a filename.

Run a short smoke with a different name (for example `ppo_counter_run1_smoke`) and enough steps to publish one checkpoint. Delete it only after recording that startup, status, checkpoint publication, stop, and model inspection worked.

### 3A. Launch headlessly

```powershell
& $hexwarsPython .\python\hexwars_ml.py train `
  --run ppo_counter_run1 `
  --algorithm maskable_ppo `
  --opponent run:.\python\runs\run1 `
  --timesteps 300000 `
  --checkpoint-every 10000 `
  --workers 4 `
  --seed 1701 `
  --device auto `
  --learner-seat alternating `
  --tracker local `
  --tracker tensorboard
```

Use `--workers 4` only if `benchmark --workers 4` measured better throughput than the alternatives on this machine. The trainer has no Unity or rendering dependency and can run with the Editor closed.

### 3B. Or launch from Unity

Open **HexWars > ML Lab > Train** and enter the equivalent settings:

- Run: `ppo_counter_run1`
- Algorithm: `maskable_ppo`
- Opponent: **LiveRun**, path `python/runs/run1` (this is the Train tab's run-directory choice; `run:` resolves the current complete checkpoint as a frozen training opponent)
- Timesteps: `300000`; checkpoint interval: `10000`; seed: `1701`
- Learner seat: `alternating`; workers: the benchmarked count
- Trackers: local plus TensorBoard (and W&B if desired)

Choose **Doctor**, then **Start & Watch**. Training remains headless; Start & Watch separately opens rendered Arena games after a complete checkpoint exists. In the Arena, verify the learner run and exact checkpoint step shown for each seat. Checkpoints reload only between games. You may pause, change pacing, or stop watching without changing the trainer.

### 4. Monitor headless progress

In a second terminal:

```powershell
& $hexwarsPython .\python\hexwars_ml.py status .\python\runs\ppo_counter_run1 --follow
& $hexwarsPython -m tensorboard.main --logdir .\python\runs
```

To mirror the same run to W&B, install/login first as described in [trackers.md](trackers.md), then include `--tracker wandb --wandb-project hexwars --wandb-group counter-run1` when launching. The Train tab exposes those same non-secret settings. Monitor the W&B reward/optimization/throughput curves alongside local status, but use the local manifest and held-out `evaluation.json` for the final decision.

Read the metrics according to the question they answer:

| Evidence | What it answers | What it does not prove |
| --- | --- | --- |
| State, step, checkpoint age | Is the process alive and making durable progress? | Policy quality |
| Steps/second | Is collection throughput acceptable? | Better gameplay |
| Episode reward/length | Is the training signal changing? | Held-out win rate |
| PPO loss/KL/entropy | Is optimization numerically behaving? | Tactical strength by itself |
| Reciprocal W/L/D and Wilson interval | How often does it beat this named opponent on held-out seeds? | General full-game ability |
| Arena observation/replay | What tactical failure or style might explain a metric? | Statistical confidence |

The local run folder is authoritative if a remote dashboard disagrees or becomes unavailable. To stop safely:

```powershell
& $hexwarsPython .\python\hexwars_ml.py stop .\python\runs\ppo_counter_run1 --after-checkpoint
```

To continue later, create a new provenance-preserving run rather than altering the old one:

```powershell
& $hexwarsPython .\python\hexwars_ml.py resume .\python\runs\ppo_counter_run1 --run ppo_counter_run1_resume --timesteps 500000
```

Runs created through the ML Lab/CLI use an absolute total timestep target. A manifest created by a legacy compatibility wrapper may declare `config.timestep_mode: "additional"`; resume preserves it, so `500000` means 500,000 additional steps for that legacy source. Check the source manifest before resuming old runs. Experimental masked DQN resume is disabled until replay-buffer sidecars are durable.

### 5. Watch arbitrary models

In **ML Lab > Arena**, configure each seat independently. To diagnose this experiment:

- Seat 0: **Live Run**, `python/runs/ppo_counter_run1`
- Seat 1: **Fixed Run**, `python/runs/run1`

Live Run resolves the newest complete checkpoint at each game boundary. Fixed Run resolves one exact checkpoint and stays frozen. Reverse the seat assignments for the reciprocal visual comparison. The panel shows the resolved algorithm, exact checkpoint path/step, and contract identity; it also shows current seat/seed and rolling W/L/D. Hidden training episodes are never revealed—the Arena is generating separate presentation matches.

For two concurrently training policies, choose **Live Run** for both seats. Each side may advance independently between games. This watches two arbitrary published model streams; it does not couple their optimizers or let either read half-written weights.

### 6. Evaluate reciprocally

After a chosen checkpoint is complete:

```powershell
& $hexwarsPython .\python\hexwars_ml.py evaluate `
  --p0 run:.\python\runs\ppo_counter_run1 `
  --p1 run:.\python\runs\run1 `
  --games 50 `
  --seed-start 1000000 `
  --both-seats `
  --workers 4 `
  --output .\python\runs\ppo_counter_run1\evaluation.json
```

`--games 50 --both-seats` runs 50 held-out seeds in the specified orientation and 50 with the controllers swapped, for 100 total games. Report wins, losses, draws, the sample size, and the Wilson confidence interval—not only a percentage. Compare each seat orientation with the baseline. A result whose interval is broad or overlaps the rejection threshold calls for more held-out games, not optimistic rounding.

Also evaluate against Greedy to detect regression against the common scripted baseline. Use a separate output path while investigating or copy the prior signed-off report before replacing `evaluation.json`.

### 7. Decide and publish a lab candidate

Record:

- hypothesis and controlled change;
- source commit, Python/.NET dependency versions, algorithm, parent checkpoint, worker count, and seeds;
- final checkpoint step and contract hash;
- reciprocal per-seat W/L/D, total games, confidence intervals, and Greedy regression result;
- one representative success and failure, with a replay or exact seed/seat assignment;
- known limitation: fixed tactical contract rather than arbitrary full-game control;
- decision and reason.

If the evidence supports further Editor testing:

```powershell
& $hexwarsPython .\python\hexwars_ml.py publish-checkpoint .\python\runs\ppo_counter_run1 --name ppo-counter-candidate
```

This creates a named **Editor-lab candidate**, not the game's official AI. Open it in the Arena against `run1` and Greedy for qualitative analysis. Rejecting the experiment means preserving enough metadata/evaluation to avoid repeating it, then deleting only redundant large checkpoints according to the project's retention policy.

## Adding an algorithm adapter

Algorithms are registered in `python/ml_lab/algorithms.py` behind `AlgorithmAdapter`. An adapter declares `name`, `policy_name`, and `experimental`, and implements:

```python
class AlgorithmAdapter(Protocol):
    def create(self, env, **kwargs): ...
    def load(self, path: Path, *, env, device: str): ...
    def validate_model(self, model, expected_contract): ...
    def predict(self, model, observation, mask) -> int: ...
    def save(self, model, path: Path) -> Path: ...
    def inspect(self, path: Path, expected_contract) -> dict: ...
```

Register the explicit CLI/Unity name; do not guess it from a checkpoint filename. Legal-action masking must apply during both exploration and deterministic inference. The minimum tests cover registry metadata, unknown-algorithm rejection, create/load/save, model geometry and semantic contract rejection, legal-mask enforcement, checkpoint inspection, fixed and live controller resolution, reciprocal evaluation, and resume behavior. Keep a new adapter marked experimental until those tests and held-out evidence justify support.

A minimal SB3 adapter placed beside the existing adapters looks like this (the shared helpers perform durable saving and contract checks):

```python
@dataclass(frozen=True)
class ExperimentalPPOAdapter:
    name: str = "experimental_ppo"
    policy_name: str = "MlpPolicy"
    experimental: bool = True

    def create(self, env, *, spaces_info, seed, device, checkpoint_interval):
        del spaces_info, checkpoint_interval
        from sb3_contrib import MaskablePPO
        return MaskablePPO("MlpPolicy", env, seed=seed, device=device, verbose=1)

    def load(self, path, *, env, device):
        from sb3_contrib import MaskablePPO
        return MaskablePPO.load(path, env=env, device=device)

    def validate_model(self, model, expected_contract):
        _validate_geometry(model, expected_contract)

    def predict(self, model, observation, mask):
        action, _ = model.predict(observation, action_masks=mask, deterministic=True)
        return int(action)

    def save(self, model, path):
        return _save_sb3_model(model, path)

    def inspect(self, path, expected_contract):
        model = self.load(path, env=None, device="cpu")
        self.validate_model(model, expected_contract)
        return _checkpoint_info(expected_contract)
```

Register it in `get_algorithm_adapter` and add the exact name to the CLI choices. Then extend `python/ml_lab/controllers.py` so its algorithm validation, model loader, masked prediction, and fixed/live resolver metadata understand the same name; `policy_server.py` consumes that controller path for the Arena. Finally extend the Unity ML Lab enum serialization, CLI-value mapping, and fixed-checkpoint controller-spec mapping. A training-only dropdown entry is incomplete if evaluation or the Arena cannot resolve the resulting model. Begin with tests such as:

```python
def test_experimental_adapter_is_explicitly_labeled():
    adapter = get_algorithm_adapter("experimental_ppo")
    assert adapter.name == "experimental_ppo"
    assert adapter.experimental is True

def test_experimental_adapter_passes_the_legal_mask_to_inference(contract):
    seen = {}

    class FakeModel:
        def predict(self, observation, *, action_masks, deterministic):
            seen.update(observation=observation, action_masks=action_masks,
                        deterministic=deterministic)
            return int(np.flatnonzero(action_masks)[0]), None

    model = FakeModel()
    mask = np.array([False, True, False])
    action = ExperimentalPPOAdapter().predict(model, np.zeros(contract.observation_size), mask)
    assert np.array_equal(seen["action_masks"], mask)
    assert seen["deterministic"] is True
    assert mask[action]
```

Use the repository's real fake-model fixtures or equivalent; the essential assertion is that every returned action is legal and that an incompatible contract fails before play or resume.

## Reproducibility checklist

- Commit code before a long run and record the commit in the experiment notes.
- Preserve `run.json`, `params.json`, `progress.csv`, monitor files, logs, signed-off evaluation, and the evaluated checkpoint.
- Record training and held-out seed ranges and never tune on the held-out suite.
- Record CPU/GPU device and worker count; deterministic engine seeds do not make all GPU kernels bit-identical.
- Treat a remote tracker as a mirror. Credentials stay in its normal environment or credential store.
- Do not edit manifests or move checkpoints into another run directory to make incompatible provenance look valid.
- Never expose a lab run or candidate in a regular player menu.
