# Live Training Stochastic Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Start & Watch use native masked PPO sampling while every existing Arena, evaluation, and official-inference path remains deterministic by default.

**Architecture:** Add a validated `inference_mode` field to metadata-backed controller specifications and carry it through resolution, live reload, policy-server prediction, and resolved-seat metadata. Unity emits `stochastic` only from the dedicated live-training viewer helper; ordinary seat configurations continue to emit or imply `deterministic`.

**Tech Stack:** Python 3.14, Stable-Baselines3 2.9, sb3-contrib MaskablePPO, pytest, Unity 6000.5, C#, NUnit, JSONL policy bridge.

## Global Constraints

- Start & Watch samples Maskable PPO with `deterministic=False` and always supplies the legal-action mask.
- Manual Arena, command-line evaluation, official AI, and controller specs without `inference_mode` remain deterministic.
- Masked DQN remains masked argmax; this work does not invent a DQN viewing exploration mode.
- Live checkpoint reload preserves the original inference mode.
- Private deployment remains hidden.
- Unknown inference modes fail explicitly instead of falling back.
- The active training process and its run directory must not be stopped, rewritten, or committed.

---

### Task 1: Validated Python Controller Inference Mode

**Files:**
- Modify: `python/ml_lab/controllers.py:25-70, 77-160, 164-290, 415-426`
- Test: `python/tests/test_controllers.py`

**Interfaces:**
- Consumes: existing `ControllerSpec`, `ResolvedController.metadata()`, `ControllerBinding.reload()`, and `predict()`.
- Produces: `InferenceMode = Literal["deterministic", "stochastic"]`, `ControllerSpec.inference_mode`, metadata key `inference_mode`, and `predict(..., *, deterministic: bool = True)`.

- [ ] **Step 1: Write failing controller-spec tests**

Add tests that prove the default, explicit stochastic mode, rejection path, and live-reload preservation:

```python
def test_run_inference_mode_defaults_to_deterministic() -> None:
    spec = normalize_controller_spec({"kind": "run", "path": "run-a", "mode": "live"})
    assert spec.inference_mode == "deterministic"


def test_stochastic_run_inference_mode_survives_resolution_and_reload(
    tmp_path: Path, contract: EnvironmentContract, loader
) -> None:
    run = _write_run(tmp_path, contract)
    binding = ControllerResolver(contract, model_loader=loader).bind({
        "kind": "run", "path": str(run), "mode": "live",
        "inference_mode": "stochastic",
    })
    assert binding.resolved.spec.inference_mode == "stochastic"
    assert binding.resolved.metadata()["inference_mode"] == "stochastic"
    assert binding.reload() is False
    assert binding.resolved.spec.inference_mode == "stochastic"


def test_unknown_run_inference_mode_is_rejected() -> None:
    with pytest.raises(ControllerResolutionError, match="inference mode"):
        normalize_controller_spec({
            "kind": "run", "path": "run-a", "inference_mode": "epsilon",
        })
```

- [ ] **Step 2: Run the new spec tests and verify RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_controllers.py -q
```

Expected: failures report that `ControllerSpec` has no `inference_mode` and that `epsilon` was not rejected.

- [ ] **Step 3: Implement validated inference-mode propagation**

Add the field and parser:

```python
InferenceMode = Literal["deterministic", "stochastic"]

@dataclass(frozen=True)
class ControllerSpec:
    kind: Literal["scripted", "checkpoint", "run"]
    name: str | None = None
    path: Path | None = None
    algorithm: Algorithm | None = None
    mode: Literal["fixed", "live"] = "fixed"
    inference_mode: InferenceMode = "deterministic"


def _inference_mode_field(raw: Mapping[str, Any]) -> InferenceMode:
    value = raw.get("inference_mode", "deterministic")
    if value not in {"deterministic", "stochastic"}:
        raise ControllerResolutionError(
            "controller inference mode must be 'deterministic' or 'stochastic'"
        )
    return value
```

Use `_inference_mode_field(raw)` when normalizing `run` and `checkpoint` objects. Whenever resolution converts a checkpoint-directory spec into a run spec, copy both `mode=spec.mode` and `inference_mode=spec.inference_mode`. Add `"inference_mode": self.spec.inference_mode` to `ResolvedController.metadata()`.

- [ ] **Step 4: Run controller-spec tests and verify GREEN**

Run the Task 1 Step 2 command.

Expected: all `test_controllers.py` tests pass.

- [ ] **Step 5: Write failing masked-PPO prediction tests**

Import `predict` in `python/tests/test_controllers.py` and add:

```python
def test_maskable_ppo_prediction_defaults_to_deterministic_and_can_sample() -> None:
    calls: list[dict] = []

    class Model:
        def predict(self, observation, **kwargs):
            calls.append(kwargs)
            return np.int64(3), None

    observation = np.zeros(4, dtype=np.float32)
    mask = np.array([True, False, True, True])

    assert predict(Model(), "maskable_ppo", observation, mask) == 3
    assert predict(Model(), "maskable_ppo", observation, mask, deterministic=False) == 3
    assert calls == [
        {"action_masks": mask, "deterministic": True},
        {"action_masks": mask, "deterministic": False},
    ]
```

- [ ] **Step 6: Run the prediction test and verify RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_controllers.py -q
```

Expected: the second call fails because `predict()` does not accept `deterministic`.

- [ ] **Step 7: Add the prediction switch without changing DQN**

Change the signature and PPO call only:

```python
def predict(
    model: Any,
    algorithm: Algorithm,
    observation: np.ndarray,
    mask: np.ndarray,
    *,
    deterministic: bool = True,
) -> int:
    if algorithm == "maskable_ppo":
        action, _ = model.predict(
            observation, action_masks=mask, deterministic=deterministic
        )
        return int(action)
    # Keep the existing masked-DQN argmax implementation unchanged.
```

- [ ] **Step 8: Run Python controller tests and commit Task 1**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_controllers.py -q
git add python/ml_lab/controllers.py python/tests/test_controllers.py
git commit -m "feat(ml): add validated controller inference mode"
```

Expected: controller tests pass and the commit contains only the controller and its tests.

---

### Task 2: Policy Server Uses and Reports the Resolved Mode

**Files:**
- Modify: `python/policy_server.py:65-105, 185-198`
- Test: `python/tests/test_policy_server.py`

**Interfaces:**
- Consumes: `ResolvedController.spec.inference_mode`, `ResolvedController.metadata()`, and `predict(..., deterministic=...)` from Task 1.
- Produces: `predict_for_seat(seat, observation, mask) -> int`; ready/reload metadata includes `inference_mode` through the existing metadata path.

- [ ] **Step 1: Write failing policy-server routing and metadata tests**

Add:

```python
def test_predict_for_seat_uses_resolved_stochastic_mode(monkeypatch) -> None:
    from types import SimpleNamespace
    import policy_server

    calls = []
    resolved = SimpleNamespace(
        model=object(), algorithm="maskable_ppo",
        spec=SimpleNamespace(inference_mode="stochastic"),
    )
    seat = SimpleNamespace(resolved=resolved)
    monkeypatch.setattr(
        policy_server,
        "predict",
        lambda model, algorithm, obs, mask, *, deterministic: calls.append(deterministic) or 4,
    )

    assert policy_server.predict_for_seat(
        seat, np.zeros(3, dtype=np.float32), np.array([True, False])
    ) == 4
    assert calls == [False]
```

Extend each expected `seat_models()` dictionary in `test_seat_models_is_structured_and_stably_ordered` with `"inference_mode": "deterministic"`, and make `FakeSeat.metadata()` return that field.

- [ ] **Step 2: Run policy-server tests and verify RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_policy_server.py -q
```

Expected: `predict_for_seat` is missing before implementation.

- [ ] **Step 3: Route actions through the resolved mode**

Add:

```python
def predict_for_seat(seat, observation, mask) -> int:
    resolved = seat.resolved
    assert resolved.model is not None and resolved.algorithm is not None
    return predict(
        resolved.model,
        resolved.algorithm,
        observation,
        mask,
        deterministic=resolved.spec.inference_mode == "deterministic",
    )
```

Replace the inline `predict(...)` call in the JSONL action loop with `predict_for_seat(seat, obs, mask)`.

- [ ] **Step 4: Run policy-server and controller tests**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_policy_server.py python/tests/test_controllers.py -q
```

Expected: both files pass.

- [ ] **Step 5: Commit Task 2**

```powershell
git add python/policy_server.py python/tests/test_policy_server.py
git commit -m "feat(ml): route stochastic live viewer actions"
```

Expected: the commit contains only policy-server routing and tests.

---

### Task 3: Unity Emits Stochastic Mode Only for Live Training Watch

**Files:**
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs:8-47`
- Modify: `Assets/HexWars/Presentation/PolicyBridge.cs:12-25, 202-225, 340-360`
- Modify: `Assets/HexWars/Editor/ReplayViewerMenu.cs:89-125`
- Test: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- Test: `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`

**Interfaces:**
- Consumes: JSON controller field `inference_mode` and policy metadata field `inference_mode` from Tasks 1-2.
- Produces: `ModelInferenceMode`, `ModelSeatConfiguration.InferenceMode`, and `ReplayViewerMenu.BuildLiveTrainingSpec(string)`.

- [ ] **Step 1: Write failing Unity specification tests**

Add to `ModelDuelConfigurationTests.cs`:

```csharp
[Test]
public void LiveRunArenaDefaultsToDeterministicInference()
{
    var seat = new ModelSeatConfiguration {
        Kind = ModelControllerKind.LiveRun,
        Path = "C:/runs/arena",
    };

    Assert.That(seat.BuildSpec(), Does.Contain("\"inference_mode\":\"deterministic\""));
}

[Test]
public void LiveTrainingViewerExplicitlyRequestsStochasticInference()
{
    string spec = HexWars.Presentation.EditorTools.ReplayViewerMenu
        .BuildLiveTrainingSpec("C:/runs/training");

    Assert.That(spec, Does.Contain("\"mode\":\"live\""));
    Assert.That(spec, Does.Contain("\"inference_mode\":\"stochastic\""));
}
```

Extend `PolicyBridgeProtocolTests.ReadyMessage_ParsesStructuredMetadataForBothSeats` so seat 0 JSON includes `"inference_mode":"stochastic"` and assert:

```csharp
Assert.That(message.Seats[0].InferenceMode, Is.EqualTo("stochastic"));
```

- [ ] **Step 2: Run focused Unity tests and verify RED**

Run the two test fixtures in a Unity batch test process against a clean verifier checkout:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' `
  -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\stochastic-view-verifier' -accept-apiupdate `
  -runTests -testPlatform EditMode `
  -testFilter 'HexWars.Presentation.Tests.ModelDuelConfigurationTests;HexWars.Presentation.Tests.PolicyBridgeProtocolTests' `
  -testResults 'Logs\stochastic-view-red.xml' -logFile 'Logs\stochastic-view-red.log'
```

Expected: failures report missing `InferenceMode`, `BuildLiveTrainingSpec`, and `PolicySeatInfo.InferenceMode`.

- [ ] **Step 3: Implement Unity controller mode serialization**

In `ModelDuelDriver.cs`, add:

```csharp
public enum ModelInferenceMode { Deterministic, Stochastic }

public sealed class ModelSeatConfiguration
{
    public ModelControllerKind Kind = ModelControllerKind.Greedy;
    public string Path = string.Empty;
    public ModelInferenceMode InferenceMode = ModelInferenceMode.Deterministic;

    static string InferenceValue(ModelInferenceMode value) =>
        value == ModelInferenceMode.Stochastic ? "stochastic" : "deterministic";
}
```

Populate the live `RunSpec` with `inference_mode = InferenceValue(InferenceMode)` and add `public string inference_mode;` to `RunSpec`.

- [ ] **Step 4: Implement the dedicated live-training spec helper**

In `ReplayViewerMenu.cs`, add and use:

```csharp
public static string BuildLiveTrainingSpec(string runDirectory) =>
    new ModelSeatConfiguration
    {
        Kind = ModelControllerKind.LiveRun,
        Path = runDirectory,
        InferenceMode = ModelInferenceMode.Stochastic,
    }.BuildSpec();
```

Both `WatchLiveTraining()` and `WatchLiveRun(string)` must call this helper. Do not change `WatchModelDuel()`, `PickRunSpec()`, or ML Lab's manual Arena seat configuration.

- [ ] **Step 5: Parse resolved inference metadata in Unity**

Add `public string InferenceMode { get; internal set; }` to `PolicySeatInfo`, add `public string inference_mode;` to `SeatDto`, and populate it in `ConvertSeats`:

```csharp
InferenceMode = string.IsNullOrWhiteSpace(seat.inference_mode)
    ? "deterministic"
    : seat.inference_mode,
```

- [ ] **Step 6: Run focused Unity tests and compile check**

Repeat the Task 3 Step 2 command with `stochastic-view-green.xml` and `stochastic-view-green.log`.

Expected: both fixtures pass. Then point Coplay back to `C:\Users\cddal\HexWars`, allow Unity to recompile, and run `check_compile_errors`; expected: no compile errors.

- [ ] **Step 7: Commit Task 3**

```powershell
git add Assets/HexWars/Presentation/ModelDuelDriver.cs `
  Assets/HexWars/Presentation/PolicyBridge.cs `
  Assets/HexWars/Editor/ReplayViewerMenu.cs `
  Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs `
  Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs
git commit -m "feat(ml): sample policies in live training viewer"
```

Expected: only the Unity bridge/viewer implementation and its tests are committed.

---

### Task 4: End-to-End Verification and Publication

**Files:**
- Modify only if verification reveals a defect in Tasks 1-3.

**Interfaces:**
- Consumes: the complete stochastic live-run controller path.
- Produces: verified branch and pushed commits.

- [ ] **Step 1: Run the full Python suite**

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests -q
```

Expected: all Python tests pass.

- [ ] **Step 2: Run the full engine suite**

```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --no-restore
```

Expected: all engine tests pass.

- [ ] **Step 3: Run the full Unity EditMode suite in the verifier checkout**

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' `
  -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\stochastic-view-verifier' -accept-apiupdate `
  -runTests -testPlatform EditMode `
  -testResults 'Logs\stochastic-view-full.xml' -logFile 'Logs\stochastic-view-full.log'
```

Expected: all EditMode tests pass with zero failures.

- [ ] **Step 4: Smoke-test policy-server stochastic routing**

Use the active run only as a read-only model source. Start `policy_server.py` with a JSON live-run spec containing `"inference_mode":"stochastic"`, the adaptive expected environment/version/encoding hash from its `run.json`, send one valid masked action request from the adaptive GymServer contract, and close the server.

Expected: ready metadata reports `"inference_mode":"stochastic"`; the returned action is legal under the supplied mask; the active training process remains running.

- [ ] **Step 5: Verify repository hygiene**

```powershell
git diff --check
git status --short --branch
git log -5 --oneline
```

Expected: no source changes remain. A tracked `python/hexwars_gym/__pycache__/env.cpython-314.pyc` modification created by the active training process may remain unstaged and must not be committed.

- [ ] **Step 6: Push the verified branch**

```powershell
git push origin codex/ml-full-game-actions
```

Expected: `origin/codex/ml-full-game-actions` advances through all implementation commits.
