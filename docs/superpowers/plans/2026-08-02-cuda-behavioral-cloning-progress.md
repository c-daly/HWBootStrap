# CUDA Behavioral Cloning and Progress Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train annihilation-imitation behavioral clones on the locked CUDA device while publishing canonically verified CPU artifacts and emitting safe epoch-level progress.

**Architecture:** Extend the behavioral-cloning configuration with an explicit device and validate it before model construction. Optimize the existing production actor on CUDA, canonicalize the selected model onto CPU before fixture generation and publication, and expose progress through a callback whose panel implementation prints flushed JSON while the final clone retains validated epoch history.

**Tech Stack:** Python 3, PyTorch CUDA 13.0, Stable-Baselines3, sb3-contrib MaskablePPO, NumPy, pytest, existing atomic panel stages.

**Approved design:** [CUDA Behavioral Cloning and Progress Logging](../specs/2026-08-02-cuda-behavioral-cloning-progress-design.md)

## Global Constraints

- Use only `C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe`; never use uv or WSL.
- Set `PYTHONPATH=python` and `PYTHONDONTWRITEBYTECODE=1` for every Python command.
- The production annihilation-imitation panel locks behavioral cloning to `"cuda"` and fails closed when CUDA is unavailable.
- The smoke gate explicitly uses `"cpu"`.
- Training may use CUDA, but fixture logits, checkpoint saving, controller reload, and exact equality verification remain on CPU.
- Record requested and realized device provenance in the published clone.
- Emit one flushed structured event per completed epoch and one completion event; do not inspect live staging files for progress.
- Preserve dataset composition, sampler behavior, model seeds, optimizer, hyperparameters, architecture, clone gate, PPO protocol, and final gate.
- Do not weaken execution-identity, definition-hash, dataset-revision, or atomic-publication checks.
- Preserve the interrupted CPU dataset and staging until this correction passes independent review.
- Use TDD for every behavior change. Do not commit datasets, models, staging artifacts, Python bytecode, or evidence archives.
- Never add attribution trailers to commits.

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `python/ml_lab/imitation.py` | Explicit BC device contract, CUDA preflight, CPU canonical publication, progress events, and retained training history |
| `python/tests/test_imitation.py` | Unit/integration coverage for device validation, CUDA use, CPU publication, event schema, and history |
| `python/run_annihilation_imitation_panel.py` | Locked device threading, flushed JSON presentation, completed-clone physical validation, and CPU smoke wiring |
| `python/tests/test_annihilation_imitation_panel.py` | Panel-definition, logging, reuse, history, and failure-boundary tests |
| `python/panels/annihilation-imitation-v1/panel.json` | Immutable production `"device": "cuda"` definition |
| `python/panels/annihilation-imitation-v1/PROTOCOL.md` | Research method, hardware provenance, progress, and restart behavior |

---

### Task 1: Separate CUDA Optimization from CPU Publication

**Files:**
- Modify: `python/ml_lab/imitation.py:709-727,990-1171`
- Modify: `python/tests/test_imitation.py`

**Interfaces:**
- Produces: `BehavioralCloningConfig.device: str`.
- Produces: `resolve_behavioral_cloning_device(requested: str) -> dict[str, Any]`.
- Produces: `canonicalize_behavioral_clone_for_publication(model: Any) -> None`.
- Preserves: `train_behavioral_clone` returning `BehavioralCloningResult` and exact CPU reload verification.

- [ ] **Step 1: Write failing configuration and CUDA-preflight tests**

Add tests that make the desired contract explicit:

```python
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_behavioral_cloning_config_accepts_explicit_supported_device(device: str):
    assert BehavioralCloningConfig(device=device).device == device


@pytest.mark.parametrize("device", ["", "auto", "cuda:0", "mps", "CPU"])
def test_behavioral_cloning_config_rejects_unlocked_device(device: str):
    with pytest.raises(ValueError, match="device"):
        BehavioralCloningConfig(device=device)


def test_cuda_preflight_fails_closed_when_cuda_is_unavailable(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)

    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_behavioral_cloning_device("cuda")
```

Update every existing `BehavioralCloningConfig` constructor call in this test module to pass `device="cpu"`; the new field is explicit rather than defaulted.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "behavioral_cloning_config or cuda_preflight"
```

Expected: FAIL because `device` and `resolve_behavioral_cloning_device` do not exist.

- [ ] **Step 3: Implement the minimal explicit device resolver**

Add the field and resolver:

```python
@dataclass(frozen=True)
class BehavioralCloningConfig:
    model_seed: int = 0
    batch_size: int = 256
    learning_rate: float = 3e-4
    max_epochs: int = 50
    patience: int = 5
    device: str = ""

    def __post_init__(self) -> None:
        # Retain all existing validation.
        if self.device not in {"cpu", "cuda"}:
            raise ValueError(
                "behavioral-cloning device must be exactly 'cpu' or 'cuda'"
            )


def resolve_behavioral_cloning_device(requested: str) -> dict[str, Any]:
    import torch

    if requested not in {"cpu", "cuda"}:
        raise ValueError("unsupported behavioral-cloning device")
    if requested == "cuda":
        if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
            raise RuntimeError("behavioral-cloning CUDA device is unavailable")
        index = int(torch.cuda.current_device())
        return {
            "requested": "cuda",
            "resolved": f"cuda:{index}",
            "torch_version": str(torch.__version__),
            "cuda_runtime": str(torch.version.cuda),
            "device_index": index,
            "device_name": str(torch.cuda.get_device_name(index)),
        }
    return {
        "requested": "cpu",
        "resolved": "cpu",
        "torch_version": str(torch.__version__),
        "cuda_runtime": None,
        "device_index": None,
        "device_name": None,
    }
```

Resolve the device before `MaskablePPOAdapter.create` and pass
`device=config.device` instead of `device="cpu"`. Verify the created model's
parameter device type matches the request before constructing the BC optimizer.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing CUDA-threading and CPU-canonicalization tests**

Use the existing tiny environment/dataset fixtures and adapter monkeypatch pattern:

```python
class _CapturedDevice(RuntimeError):
    pass


def test_clone_trainer_passes_requested_device_to_production_adapter(
    clone_dataset: Path,
    clone_scenario,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    captured: dict[str, str] = {}

    def capture_create(
        self, env, *, spaces_info, seed, device, checkpoint_interval, options=None
    ):
        captured["device"] = device
        raise _CapturedDevice(device)

    monkeypatch.setattr(MaskablePPOAdapter, "create", capture_create)
    monkeypatch.setattr(
        imitation_module,
        "resolve_behavioral_cloning_device",
        lambda requested: {
            "requested": requested,
            "resolved": "cuda:0",
            "torch_version": "test",
            "cuda_runtime": "test",
            "device_index": 0,
            "device_name": "test-gpu",
        },
    )

    with pytest.raises(_CapturedDevice, match="cuda"):
        train_behavioral_clone(
            dataset=dataset,
            scenario=clone_scenario,
            env=_TinyCloneEnv(),
            contract=contract(),
            spaces_info={
                "channels": 1, "board_h": 1, "board_w": 1, "globals": 2,
            },
            run_dir=tmp_path / "bc",
            config=BehavioralCloningConfig(
                model_seed=211, batch_size=5, max_epochs=1, patience=1,
                device="cuda",
            ),
        )

    assert captured == {"device": "cuda"}
```

Also extend
`test_behavioral_clone_overfits_a_five_example_masked_dataset_and_publishes_a_resolvable_run`
with these concrete assertions after it loads `bc` and resolves `first`:

```python
assert bc["publication_device"] == "cpu"
assert bc["training_device"]["requested"] == "cpu"
assert bc["training_device"]["resolved"] == "cpu"
assert {
    parameter.device.type for parameter in first.model.policy.parameters()
} == {"cpu"}
```

The threading test stops at the adapter boundary and therefore runs on hosts
without CUDA; the later real-CUDA gate exercises actual hardware.

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "passes_requested_device or canonical_cpu_artifact"
```

Expected: FAIL because training still hardcodes CPU and publishes no device provenance.

- [ ] **Step 7: Implement CPU canonical publication**

After restoring `best_actor_state`, canonicalize before validation metrics,
fixture logits, or saving:

```python
def canonicalize_behavioral_clone_for_publication(model: Any) -> None:
    import torch

    model.policy.to(torch.device("cpu"))
    model.device = torch.device("cpu")
    devices = {parameter.device.type for parameter in model.policy.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError("behavioral-cloning publication model is not on CPU")
```

Store the resolver output as `training_device` and
`publication_device: "cpu"` in `bc.json`. Include `device` in the persisted
`bc_config` produced by `asdict(config)`. Keep
Keep `_verify_reload_identity` and its `torch.testing.assert_close` call at `rtol=0, atol=0` unchanged.

- [ ] **Step 8: Run Task 1 tests and the complete imitation module**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 1**

```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py
git diff --cached --check
git commit -m "Use explicit device for behavioral cloning"
```

---

### Task 2: Emit and Retain Epoch Progress

**Files:**
- Modify: `python/ml_lab/imitation.py:990-1171`
- Modify: `python/tests/test_imitation.py`

**Interfaces:**
- Adds the keyword-only argument `progress: Callable[[Mapping[str, Any]], None] | None = None` to `train_behavioral_clone`.
- Produces: schema-versioned `bc_epoch` and `bc_complete` mappings.
- Produces: `training-history.json` inside each completed clone.

- [ ] **Step 1: Write a failing epoch-event test**

Add a one-epoch test around the real tiny trainer:

```python
def test_clone_trainer_emits_finite_epoch_and_completion_progress(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    events: list[dict[str, Any]] = []
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "progress-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=1, patience=1,
            device="cpu",
        ),
        progress=events.append,
    )

    assert [event["event"] for event in events] == ["bc_epoch", "bc_complete"]
    epoch = events[0]
    assert epoch["schema_version"] == 1
    assert epoch["model_seed"] == 211
    assert epoch["device"] == "cpu"
    assert epoch["epoch"] == epoch["max_epochs"] == 1
    assert epoch["batches"] > 0
    assert epoch["examples"] > 0
    for key in (
        "mean_training_loss", "validation_nll", "top1_accuracy",
        "top3_accuracy", "top5_accuracy", "epoch_seconds",
        "elapsed_seconds", "examples_per_second",
    ):
        assert math.isfinite(epoch[key])
    assert epoch["epoch_seconds"] >= 0
    assert epoch["examples_per_second"] >= 0
    assert events[-1]["run_dir"] == str(result.run_dir.resolve())
```

- [ ] **Step 2: Run the event test and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "emits_finite_epoch"
```

Expected: FAIL because `progress` is not accepted.

- [ ] **Step 3: Implement the callback and event construction**

Use `time.perf_counter()`. Accumulate `float(loss.detach().cpu())` for every
batch, then emit after the existing validation metrics:

```python
event = {
    "schema_version": 1,
    "event": "bc_epoch",
    "model_seed": config.model_seed,
    "device": device_provenance["resolved"],
    "epoch": epoch,
    "max_epochs": config.max_epochs,
    "batches": steps_per_epoch,
    "examples": steps_per_epoch * config.batch_size,
    "mean_training_loss": float(sum(losses) / len(losses)),
    "validation_nll": float(validation_metrics.nll),
    "top1_accuracy": float(validation_metrics.top1_accuracy),
    "top3_accuracy": float(validation_metrics.top3_accuracy),
    "top5_accuracy": float(validation_metrics.top5_accuracy),
    "best_epoch": int(best_epoch),
    "best_validation_nll": float(best_nll),
    "epochs_without_improvement": int(epochs_without_improvement),
    "patience": int(config.patience),
    "epoch_seconds": float(epoch_elapsed),
    "elapsed_seconds": float(total_elapsed),
    "examples_per_second": float(examples / epoch_elapsed) if epoch_elapsed else 0.0,
}
history.append(event)
if progress is not None:
    progress(dict(event))
```

Validate that booleans do not occupy numeric fields, counts are non-negative
integers, metrics/times are finite, and rates are non-negative. Invoke the
callback synchronously so callback failure aborts publication.

After successful `os.replace(temporary, run_dir)`, emit:

```python
{
    "schema_version": 1,
    "event": "bc_complete",
    "model_seed": config.model_seed,
    "device": device_provenance["resolved"],
    "best_epoch": best_epoch,
    "epochs_trained": epochs_trained,
    "elapsed_seconds": total_elapsed,
    "run_dir": str(run_dir.resolve()),
}
```

- [ ] **Step 4: Run the event test and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing retained-history tests**

```python
def test_clone_publishes_complete_epoch_history(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "history-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=2, patience=2,
            device="cpu",
        ),
    )
    payload = json.loads(
        (result.run_dir / "training-history.json").read_text(encoding="utf-8")
    )
    assert payload["schema_version"] == 1
    assert payload["model_seed"] == 211
    assert payload["training_device"]["requested"] == "cpu"
    assert len(payload["epochs"]) == result.epochs_trained
    assert [row["epoch"] for row in payload["epochs"]] == list(
        range(1, result.epochs_trained + 1)
    )
    assert payload["epochs"][-1]["best_epoch"] == result.best_epoch
```

- [ ] **Step 6: Run the history test and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "publishes_complete_epoch_history"
```

Expected: FAIL because `training-history.json` is absent.

- [ ] **Step 7: Publish the compact history atomically**

Before run publication, write:

```python
atomic_write_json(
    temporary / "training-history.json",
    {
        "schema_version": 1,
        "model_seed": config.model_seed,
        "training_device": device_provenance,
        "publication_device": "cpu",
        "epochs": history,
    },
)
```

Add `"training-history.json"` to the exact published-file set in
`test_behavioral_clone_overfits_a_five_example_masked_dataset_and_publishes_a_resolvable_run`.
Do not write a live history file outside the temporary publication directory.
Live monitoring remains stdout-only.

- [ ] **Step 8: Run Task 2 tests and the complete imitation module**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q
```

Expected: all tests pass.

- [ ] **Step 9: Commit Task 2**

```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py
git diff --cached --check
git commit -m "Report behavioral cloning epoch progress"
```

---

### Task 3: Lock CUDA in the Panel and Verify the Real GPU Path

**Files:**
- Modify: `python/run_annihilation_imitation_panel.py:36-61,416-443,602-836,1352-1410,1793-1832,2171-2205,2453-2479`
- Modify: `python/tests/test_annihilation_imitation_panel.py`
- Modify: `python/panels/annihilation-imitation-v1/panel.json`
- Modify: `python/panels/annihilation-imitation-v1/PROTOCOL.md`
- Modify: `python/tests/test_imitation.py`

**Interfaces:**
- Produces: `emit_bc_progress(event: Mapping[str, Any]) -> None`.
- Threads: `panel["behavioral_cloning"]["device"]` into every production clone config.
- Extends `_validate_clone_run` with keyword-only `expected_device: str | None = None`.
- Requires: completed clones contain valid device provenance and `training-history.json`.
- Preserves: smoke BC with explicit `device="cpu"`.

- [ ] **Step 1: Write failing locked-definition and threading tests**

Update the expected BC definition and assert exact wiring:

```python
def test_locked_behavioral_cloning_uses_cuda():
    panel, _banks, _scenario = module.validate_definitions()
    assert panel["behavioral_cloning"] == {
        "batch_size": 256,
        "learning_rate": 0.0003,
        "max_epochs": 50,
        "patience": 5,
        "standard_fraction_basis_points": 7000,
        "device": "cuda",
    }


```

Extend
`test_clone_run_construction_uses_fresh_configs_and_distinct_destinations`
after its existing assertions; its `calls` list already contains the three
trainer invocations:

```python
assert [call["config"].device for call in calls] == ["cuda", "cuda", "cuda"]
assert all(call["progress"] is module.emit_bc_progress for call in calls)
```

Add `device="cpu"` to smoke config assertions.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_annihilation_imitation_panel.py -q -k "locked_behavioral_cloning_uses_cuda or threads_locked_device or smoke"
```

Expected: FAIL because the panel has no device field and clone configs do not receive one.

- [ ] **Step 3: Add the immutable panel field and thread it**

Add `"device": "cuda"` under `behavioral_cloning` in `panel.json` and in
the exact expected definition in `validate_definitions`. Pass
`device=bc["device"]` when constructing every production
`BehavioralCloningConfig`. Pass `device="cpu"` in the smoke schedule.

The panel hash changes by design. Do not edit the current dataset manifest or
execution identity to match it.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the Step 2 command.

Expected: PASS.

- [ ] **Step 5: Write failing flushed-JSON and physical-history tests**

For presentation, monkeypatch `builtins.print`:

```python
def test_emit_bc_progress_prints_one_sorted_flushed_json_object(monkeypatch):
    calls = []
    monkeypatch.setattr(
        builtins, "print",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    event = {"event": "bc_epoch", "schema_version": 1, "epoch": 1}
    module.emit_bc_progress(event)
    assert calls == [((json.dumps(event, sort_keys=True),), {"flush": True})]
```

Extend clone fixtures with `training-history.json` and device provenance.
Add rejection tests for:

- missing history;
- non-contiguous epochs;
- non-finite loss or timing;
- history seed/config mismatch;
- missing or malformed CUDA provenance;
- a completed clone whose `bc_config.device` differs from the locked panel.

- [ ] **Step 6: Run the new tests and verify RED**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_annihilation_imitation_panel.py -q -k "emit_bc_progress or training_history or device_provenance"
```

Expected: FAIL because presentation and physical validation are absent.

- [ ] **Step 7: Implement presentation and completed-clone validation**

Add:

```python
def emit_bc_progress(event: Mapping[str, Any]) -> None:
    print(json.dumps(dict(event), sort_keys=True), flush=True)
```

Pass `progress=emit_bc_progress` from `train_clone_runs` to the selected
trainer. Update test doubles to accept the callback.

In `_validate_clone_run`, require:

- `manifest["bc_config"]["device"]` is `"cpu"` or `"cuda"`;
- when `expected_device` is supplied, the config device equals it exactly;
- `bc["training_device"]["requested"]` equals that config device;
- `bc["publication_device"] == "cpu"`;
- `training-history.json` has schema 1, matching seed/device, and exactly
  `bc["epochs_trained"]` contiguous epoch rows;
- every required numeric event field is finite;
- epoch, batches, examples, best epoch, patience counters, and timing bounds
  are valid;
- the last history row's best epoch/NLL agree with `bc.json`.

Pass `expected_device=bc["device"]` at all three production validation calls
inside `train_clone_runs`. Pass `expected_device="cuda"` from
`_clone_metadata` and clone-gate physical validation. Pass
`expected_device=config.device` from smoke reuse, which remains explicitly
CPU.

- [ ] **Step 8: Update the protocol**

Document:

- locked CUDA optimization and fail-closed preflight;
- exact hardware/software provenance;
- CPU canonical publication and exact fixture verification;
- flushed per-epoch JSON fields;
- retained `training-history.json`;
- absence of a live progress file;
- requirement to recollect after the accepted code/definition change.

- [ ] **Step 9: Run focused panel and imitation tests**

Run:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py python/tests/test_annihilation_imitation_panel.py -q
```

Expected: all tests pass.

- [ ] **Step 10: Add and run a real-CUDA micro-gate**

Add a CUDA-marked test using the existing tiny dataset/environment. Skip only
when `torch.cuda.is_available()` is false:

```python
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_behavioral_clone_real_cuda_training_publishes_cpu_artifact(
    clone_dataset: Path, clone_scenario, tmp_path: Path
) -> None:
    dataset = load_imitation_dataset(clone_dataset, expected_contract=contract())
    events: list[dict[str, Any]] = []
    result = train_behavioral_clone(
        dataset=dataset,
        scenario=clone_scenario,
        env=_TinyCloneEnv(),
        contract=contract(),
        spaces_info={"channels": 1, "board_h": 1, "board_w": 1, "globals": 2},
        run_dir=tmp_path / "cuda-bc",
        config=BehavioralCloningConfig(
            model_seed=211, batch_size=5, max_epochs=1, patience=1,
            device="cuda",
        ),
        progress=events.append,
    )
    assert events[0]["device"].startswith("cuda:")
    bc = json.loads((result.run_dir / "bc.json").read_text(encoding="utf-8"))
    assert bc["training_device"]["device_name"] == torch.cuda.get_device_name(
        torch.cuda.current_device()
    )
    assert bc["publication_device"] == "cpu"
    resolved = ControllerResolver(contract()).resolve(f"run:{result.run_dir}")
    assert {
        parameter.device.type for parameter in resolved.model.policy.parameters()
    } == {"cpu"}
```

Run only this real hardware proof:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests/test_imitation.py -q -k "real_cuda_training"
```

Expected on this machine: 1 passed, not skipped. Confirm the event names
`NVIDIA GeForce RTX 5070`.

- [ ] **Step 11: Run complete regression gates**

Run independently:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONDONTWRITEBYTECODE='1'
& 'C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe' -m pytest python/tests -q

dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo

dotnet build engine\HexWars.GymServer\HexWars.GymServer.csproj --nologo
```

Restore only test-generated tracked Python bytecode if necessary. Run
`git diff --check` and verify no generated artifact is staged.

- [ ] **Step 12: Commit Task 3**

```powershell
git add python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py python/panels/annihilation-imitation-v1/panel.json python/panels/annihilation-imitation-v1/PROTOCOL.md python/tests/test_imitation.py
git diff --cached --check
git commit -m "Lock CUDA behavioral cloning with progress logs"
```

- [ ] **Step 13: Obtain independent review before experiment restart**

Generate a review package spanning the three implementation-task commits.
Require both spec compliance and code quality PASS. Fix every Critical or
Important finding through the SDD fix loop before touching the current dataset
or interrupted staging.

After review acceptance:

1. Move the current `ccbff0c537f8f282202fb05ff621e958b30da9d0de6d9e322109be2436b34a84`
   dataset and interrupted CPU staging intact into a new ignored evidence
   archive.
2. Run `validate` on the accepted clean commit.
3. Recollect the identity-bound dataset without opening live staging files.
4. Run `train-bc` and monitor the flushed epoch JSON from process output.
5. Continue the original Task 12 plan only if the clone gate passes.
