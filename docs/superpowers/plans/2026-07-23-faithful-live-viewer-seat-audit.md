# Faithful Live Viewer and Seat Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Start & Watch reproduce a run's scenario, opponent, learner seating, and fog perspective while recording enough durable episode data to prove alternating seats actually alternated.

**Architecture:** Python adds worker/episode/seed/seat columns to the authoritative monitor stream and status JSON summarizes seat counts. Unity parses one immutable `MlRunPresentationPlan` from `run.json` plus `scenario.json`; `ReplayViewerMenu` launches that plan without fallback. `ModelDuelDriver` uses the recorded scenario and reconfigures controller seats only at game boundaries, following the learner with the observer.

**Tech Stack:** Unity 6 Editor and runtime presentation; C# engine adapters; Python CSV/status CLI; NUnit; pytest.

## Global Constraints

- This plan begins only after `2026-07-23-training-game-template-pipeline.md` is complete.
- Start & Watch never substitutes Greedy for missing/unsupported metadata.
- Fixed seat `0` means learner P0; fixed seat `1` means learner P1.
- Viewer alternation starts learner-in-P0 and swaps after every completed presentation game.
- Training workers retain `(worker_index + episode_index) % 2`.
- The observer follows the learner and therefore never reveals the other seat's fog-hidden entities.
- Live models reload only at a completed-game boundary.
- Opponent pools cycle deterministically in the viewer and are labeled; this is not exact training-episode replay.
- Manual Arena retains its independent seat and observer controls.
- Legacy runs use a visible `legacy-default` scenario label and are never rewritten.

---

## File and Interface Map

### New files

- `Assets/HexWars/Editor/MlLab/MlRunPresentationPlan.cs` — strict run/scenario/resolved-opponent/seat parser and per-game plan generator.
- `Assets/HexWars/Tests/Editor/MlRunPresentationPlanTests.cs` — fixed/alternating/pool/legacy/failure tests.

### Modified files

- `python/ml_lab/contracts.py` — expanded monitor header.
- `python/ml_lab/controllers.py` — resolve internal metadata-backed checkpoint snapshots recorded with a run.
- `python/ml_lab/envs.py` — expose current assignment and record worker/episode/seed/seat.
- `python/ml_lab/cli.py` — include seat summary in status result.
- `python/tests/test_training.py`, `python/tests/test_cli.py`, `python/tests/test_run_contract.py` — durable audit tests.
- `Assets/HexWars/Editor/MlLab/MlRunStatus.cs` — parse Seat 0/Seat 1 counts and audit warnings.
- `Assets/HexWars/Editor/MlLab/MlLabWindow.cs` — display audit and launch strict run plans.
- `Assets/HexWars/Editor/ReplayViewerMenu.cs` — remove hard-coded Greedy and default environment fallback.
- `Assets/HexWars/Presentation/ModelDuelEnvironment.cs` — construct duel adapters from a `TrainingScenario`.
- `Assets/HexWars/Presentation/ModelDuelDriver.cs` — game-boundary presentation scheduling, bridge restart, observer following.
- `Assets/HexWars/Presentation/ModelArenaIdentity.cs` — learner/opponent role labels.
- `Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs` — render the role labels with the existing controller identity.
- Existing Unity EditMode tests for ML Lab, model duel, policy bridge, and arena identity.
- `docs/ml/architecture.md`, `docs/ml/experiment-guide.md`, `docs/ml/troubleshooting.md` — viewer semantics and audit interpretation.

### Task 1: Durable episode seat and seed audit

**Files:**
- Modify: `python/ml_lab/contracts.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/tests/test_training.py`
- Modify: `python/tests/test_run_contract.py`

**Interfaces:**
- Changes `MONITOR_HEADER` to:
  `worker_id,episode_index,episode_seed,learner_seat,episode_reward,episode_length,elapsed_seconds`
- Produces: `ScheduledEnvironment.current_assignment -> EpisodeAssignment`
- Produces: `EpisodeAssignment(worker_id, episode_index, seed, learner_seat)`.

- [ ] **Step 1: Write failing schedule and CSV tests**

```python
def test_monitor_records_assignment_for_each_completed_episode(tmp_path: Path) -> None:
    schedule = WorkerSchedule(base_seed=17, worker_index=1, worker_count=2)
    env = EpisodeMonitor(
        ScheduledEnvironment(schedule, build_one_step_env),
        tmp_path / "monitor.worker_1.csv",
        threading.Lock(),
        worker_id=1,
    )
    env.reset()
    env.step(0)
    env.reset()
    env.step(0)

    rows = list(csv.DictReader((tmp_path / "monitor.worker_1.csv").open()))
    assert [(row["worker_id"], row["episode_index"], row["episode_seed"], row["learner_seat"])
            for row in rows] == [
        ("1", "0", "18", "1"),
        ("1", "1", "20", "0"),
    ]
```

Add a four-worker test proving every worker alternates, the aggregate difference is at most one episode per worker, and fixed-seat schedules record only their configured seat.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_training.py python/tests/test_run_contract.py -q
```

Expected: current assignment metadata and CSV columns are absent.

- [ ] **Step 3: Implement an explicit assignment value**

```python
@dataclass(frozen=True)
class EpisodeAssignment:
    worker_id: int
    episode_index: int
    seed: int
    learner_seat: int
```

Change `WorkerSchedule.next_episode()` to return that type without changing its seed/seat formulas. `ScheduledEnvironment.reset()` stores the assignment used for the active episode. `EpisodeMonitor` reads it after reset and writes it with terminal metrics. Put the same fields into `info["episode"]`.

- [ ] **Step 4: Run all Python tests and commit**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests -q
```

Expected: all tests pass.

Commit:

```powershell
git add python/ml_lab/contracts.py python/ml_lab/envs.py python/tests/test_training.py python/tests/test_run_contract.py
git commit -m "feat(ml): record learner seats per episode"
```

### Task 2: Status seat summaries and Unity monitoring

**Files:**
- Modify: `python/ml_lab/cli.py`
- Modify: `python/tests/test_cli.py`
- Modify: `Assets/HexWars/Editor/MlLab/MlRunStatus.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlRunStatusTests.cs`

**Interfaces:**
- Produces status JSON:

```json
{
  "seat_audit": {
    "seat_0_episodes": 12,
    "seat_1_episodes": 11,
    "readable": true,
    "balanced": true,
    "warning": ""
  }
}
```

- Produces Unity properties: `Seat0Episodes`, `Seat1Episodes`, `SeatAuditReadable`, `SeatAuditWarning`.

- [ ] **Step 1: Add failing Python status aggregation tests**

Create two monitor worker files with interleaved seats, invoke `_run_result`, and assert numeric counts. Add malformed/missing-header coverage that returns `readable: false` with a path-specific warning instead of crashing status.

- [ ] **Step 2: Implement `read_seat_audit(run_dir, manifest)`**

Read exactly the manifest's `monitor_files`, count `learner_seat`, and compute:

```python
tolerance = max(1, int(manifest["config"].get("workers", 1)))
balanced = abs(seat_0 - seat_1) <= tolerance
```

Only emit imbalance warning for `learner_seat == "alternating"` and terminal run states. Fixed runs return counts without a warning.

- [ ] **Step 3: Add failing Unity parse/display tests**

```csharp
[Test]
public void ParseJson_ReadsSeatAudit()
{
    const string json = "{\"ok\":true,\"result\":{\"run\":{\"state\":\"running\"}," +
        "\"seat_audit\":{\"seat_0_episodes\":12,\"seat_1_episodes\":11," +
        "\"readable\":true,\"balanced\":true,\"warning\":\"\"}}}";
    MlRunStatus status = MlRunStatus.Parse(json);
    Assert.That(status.Seat0Episodes, Is.EqualTo(12));
    Assert.That(status.Seat1Episodes, Is.EqualTo(11));
    Assert.That(status.SeatAuditReadable, Is.True);
}
```

- [ ] **Step 4: Implement ML Lab status panel**

Display `Learner episodes (Seat 0 / Seat 1)` and the counts. Use an info help box for unreadable/in-progress audit and a warning box only for terminal material imbalance.

- [ ] **Step 5: Verify and commit**

Run all Python tests. Call Coplay `check_compile_errors`, then run `MlRunStatusTests`.

Commit:

```powershell
git add python/ml_lab/cli.py python/tests/test_cli.py Assets/HexWars/Editor/MlLab/MlRunStatus.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/MlRunStatusTests.cs
git commit -m "feat(ml): surface learner seat audit"
```

### Task 3: Strict run presentation plans

**Files:**
- Create: `Assets/HexWars/Editor/MlLab/MlRunPresentationPlan.cs`
- Create: `Assets/HexWars/Tests/Editor/MlRunPresentationPlanTests.cs`
- Modify: `Assets/HexWars/Editor/ReplayViewerMenu.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- Modify: `python/ml_lab/controllers.py`
- Modify: `python/tests/test_controllers.py`

**Interfaces:**
- Produces: `MlRunPresentationPlan.Load(runDirectory)`
- Produces: `MlPresentationGame PlanGame(int gameIndex)`
- `MlPresentationGame` contains `P0Spec`, `P1Spec`, `LearnerSeat`, `Observer`, `OpponentLabel`, `Scenario`.
- Produces internal controller spec: `{"kind":"snapshot","path":CHECKPOINT,"source_run":RUN,"algorithm":ALGORITHM,"step":STEP}`.

- [ ] **Step 1: Write failing plan tests**

```csharp
[TestCase("0", 0, 0)]
[TestCase("1", 0, 1)]
[TestCase("alternating", 0, 0)]
[TestCase("alternating", 1, 1)]
public void PlanGame_PlacesLearnerInRecordedSeat(string schedule, int game, int learnerSeat)
{
    string run = WriteRun(schedule, opponentJson: "{\"kind\":\"scripted\",\"name\":\"random\"}");
    var plan = MlRunPresentationPlan.Load(run);
    MlPresentationGame resolved = plan.PlanGame(game);
    Assert.That(resolved.LearnerSeat, Is.EqualTo(learnerSeat));
    Assert.That(resolved.Observer, Is.EqualTo(
        learnerSeat == 0 ? ModelDuelObserverSeat.Player1 : ModelDuelObserverSeat.Player2));
    Assert.That(learnerSeat == 0 ? resolved.P1Spec : resolved.P0Spec, Is.EqualTo("random"));
}

[Test]
public void MissingOpponentFailsInsteadOfFallingBackToGreedy()
{
    string run = WriteRun("0", opponentJson: null);
    Assert.That(() => MlRunPresentationPlan.Load(run),
        Throws.InvalidOperationException.With.Message.Contains("opponent"));
}
```

Add tests for Greedy, Random, exact fixed-checkpoint snapshots, live run, deterministic pool cycling, incompatible pool metadata, missing/invalid scenario, and visible `legacy-default`.

Add Python tests proving a `snapshot` controller loads contract metadata from `source_run/run.json`, requires `path` to be a recorded checkpoint inside that run, validates algorithm and step against the recorded snapshot, and rejects a standalone or escaped path. This internal kind is accepted by `policy_server.py` through `ControllerResolver`; do not add it to regular player menus.

- [ ] **Step 2: Run EditMode tests and confirm RED**

Expected: missing plan types.

- [ ] **Step 3: Implement strict manifest parsing**

Parse `run.json.config.learner_seat` and the top-level `opponent_snapshot` written during run creation. The learner spec is always `BuildLiveTrainingSpec(runDirectory)`. A fixed opponent uses its recorded metadata-backed checkpoint path and recorded source-run contract rather than re-resolving the newest checkpoint. A live opponent retains live mode. For pools:

```csharp
int opponentIndex = gameIndex % opponents.Count;
```

Do not sample and do not claim the sequence matches hidden training episodes. `Load` throws a path-specific `InvalidOperationException`; it never returns a default plan.

- [ ] **Step 4: Replace `WatchLiveRun` hard-coding**

Change the method to:

```csharp
public static void WatchLiveRun(string runDirectory)
{
    try
    {
        MlRunPresentationPlan plan = MlRunPresentationPlan.Load(runDirectory);
        LaunchDuel(PyDir(), plan);
    }
    catch (Exception error)
    {
        Debug.LogError("HexWars Start & Watch: " + error.Message);
    }
}
```

Delete `EnvironmentFromRun`'s catch-to-tactical behavior from this path. Manual Arena may retain its own selected environment.

- [ ] **Step 5: Compile, run tests, and commit**

Run `python/tests/test_controllers.py`. Call Coplay `check_compile_errors`; run `MlRunPresentationPlanTests` and `ModelDuelConfigurationTests`.

Commit the new `.meta` files too:

```powershell
git add python/ml_lab/controllers.py python/tests/test_controllers.py Assets/HexWars/Editor/MlLab/MlRunPresentationPlan.cs Assets/HexWars/Editor/MlLab/MlRunPresentationPlan.cs.meta Assets/HexWars/Editor/ReplayViewerMenu.cs Assets/HexWars/Tests/Editor/MlRunPresentationPlanTests.cs Assets/HexWars/Tests/Editor/MlRunPresentationPlanTests.cs.meta Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs
git commit -m "fix(ml): derive viewer games from run metadata"
```

### Task 4: Scenario-aware duel adapters

**Files:**
- Modify: `Assets/HexWars/Presentation/ModelDuelEnvironment.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`

**Interfaces:**
- Adds: `ModelDuelEnvironmentFactory.Create(TrainingScenario scenario)`
- Adds: `ModelDuelEnvironmentFactory.ContractIdentity(TrainingScenario scenario)`
- Keeps the environment-enum overloads for manual Arena defaults.
- Adds: `ModelDuelConfiguration.ScenarioRunPath`; blank means the selected environment's Standard scenario.

- [ ] **Step 1: Write failing custom-scenario adapter tests**

```csharp
[Test]
public void ScenarioFactory_UsesRecordedBoardAndEncoding()
{
    TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
    scenario.Board.Width = 24;
    scenario.Board.Height = 16;

    IModelDuelEnvironment duel = ModelDuelEnvironmentFactory.Create(scenario);

    Assert.That(duel.Contract.Board["width"], Is.EqualTo(24));
    Assert.That(duel.Contract.Board["height"], Is.EqualTo(16));
    Assert.That(duel.Contract.EncodingHash,
        Is.EqualTo(ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash));
}
```

Add adaptive starting-budget, max-steps, and invalid-scenario cases.

Add a manual-Arena test proving a selected run's `scenario.json` is loaded while P0/P1 controller choices and the observer remain independent.

Add a compatibility test proving both model seats are checked against the selected scenario's encoding hash before Play Mode. A run compatible with Standard but incompatible with the selected large-board scenario must be rejected with both expected and actual hashes.

- [ ] **Step 2: Implement constructor injection**

`TacticalModelDuelEnvironment` receives `EnvConfig`; `AdaptiveModelDuelEnvironment` receives `AdaptiveEnvConfig`. The factory builds once from the recorded scenario and uses the duel contract generated by that exact config.

In the Arena panel, add a run-scenario path plus `Use selected run scenario`. `LaunchArena` loads that scenario when present; otherwise it constructs Standard for the selected environment. It rejects an environment mismatch before entering Play Mode.

- [ ] **Step 3: Compile, test, and commit**

Call Coplay `check_compile_errors`; run all `ModelDuelConfigurationTests`.

Commit:

```powershell
git add Assets/HexWars/Presentation/ModelDuelEnvironment.cs Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs
git commit -m "feat(ml): replay recorded scenarios in Arena"
```

### Task 5: Boundary-safe alternating controller seats

**Files:**
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Presentation/ModelArenaIdentity.cs`
- Modify: `Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`

**Interfaces:**
- Adds driver field: `MlRunPresentationPlan PresentationPlan`.
- Adds pure transition: `MlPresentationGame NextPresentationGame(int gamesPlayed)`.
- Adds async boundary operation that disposes and restarts `PolicyBridge` when seat specs change.

- [ ] **Step 1: Write pure scheduling tests**

Test game 0/1/2 for fixed seats and alternating seats, observer changes, deterministic pool labels, and learner/opponent win-record projection.

- [ ] **Step 2: Write bridge-boundary tests**

Extract `ShouldReconfigure(previous, next, gameEnded)` and assert:

```csharp
Assert.That(ShouldReconfigure(game0, game1, gameEnded: false), Is.False);
Assert.That(ShouldReconfigure(game0, game1, gameEnded: true), Is.True);
Assert.That(ShouldReconfigure(game0, game0, gameEnded: true), Is.False);
```

Live reload tests must still prove no reload/reconfiguration occurs mid-game.

- [ ] **Step 3: Implement asynchronous game-boundary restart**

At the end-of-game rest boundary:

1. compute the next `MlPresentationGame`;
2. set `IsStarting = true` and stop actions;
3. dispose the prior bridge;
4. assign new P0/P1 specs, learner observer, and scenario;
5. start a new bridge with the scenario-derived encoding identity;
6. validate resolved models;
7. increment seed and begin the next game.

If restart fails, retain the error in the affected identity row and stop; never continue with an old model in the wrong seat.

- [ ] **Step 4: Label learner and opponent explicitly**

Identity rows display role plus seat:

```text
P1 · Learner · adaptive_large_seed31 · Maskable PPO · step 120,000
P2 · Opponent · Greedy
```

When seats swap, labels and learner-centric W/L/D update with the next game. The observer property must resolve from the active game's `LearnerSeat`, so fog rendering follows the learner.

- [ ] **Step 5: Compile, run all affected tests, and commit**

Call Coplay `check_compile_errors`, then run:

```text
HexWars.Presentation.Tests.ModelDuelConfigurationTests
HexWars.Presentation.Tests.ModelArenaIdentityTests
HexWars.Presentation.Tests.PolicyBridgeProtocolTests
HexWars.Presentation.Tests.MlRunPresentationPlanTests
```

Commit:

```powershell
git add Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Presentation/ModelArenaIdentity.cs Assets/HexWars/Presentation/ModelArenaIdentityOverlay.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs Assets/HexWars/Tests/Editor/ModelArenaIdentityTests.cs Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs
git commit -m "feat(ml): alternate live viewer seats safely"
```

### Task 6: ML Lab launch integration, smoke tests, and docs

**Files:**
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`
- Modify: `docs/ml/architecture.md`
- Modify: `docs/ml/experiment-guide.md`
- Modify: `docs/ml/troubleshooting.md`

**Interfaces:**
- Start & Watch consumes only the selected run path; the strict plan owns all reconstruction.
- Status displays the same seat schedule and opponent label used by the viewer.

- [ ] **Step 1: Add launch-state regression tests**

Verify Start & Watch waits for a checkpoint, launches exactly once, reports strict-plan errors in ML Lab, and does not mark `_watchLaunched` when plan loading fails.

- [ ] **Step 2: Implement error propagation**

Make `WatchLiveRun` return a result:

```csharp
public readonly struct MlViewerLaunchResult
{
    public readonly bool Success;
    public readonly string Error;
}
```

ML Lab applies `Error` to its state. Successful launch shows scenario, current learner seat, and current opponent.

- [ ] **Step 3: Run complete automated verification**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
& .\python\winenv\Scripts\python.exe -m pytest python/tests -q
```

Call Coplay `check_compile_errors`, then run all Unity EditMode tests.

Expected: all suites pass.

- [ ] **Step 4: Run a real alternating smoke test**

Start a two-worker, short adaptive run against Random with `learner-seat alternating`. Verify:

- `scenario.json` matches the selected template;
- each monitor row contains worker/episode/seed/seat;
- Seat 0/Seat 1 counts appear in Unity;
- Start & Watch begins learner P0 against Random;
- the next displayed game swaps learner to P1;
- the overlay follows the swap;
- a fog-hidden opponent unit is not rendered;
- checkpoint reload happens only between games.

- [ ] **Step 5: Document semantics and troubleshooting**

Document that presentation games use the same scenario/controller schedules but are not hidden training replays. Add exact diagnostics for missing scenario, unsupported opponent, encoding mismatch, bridge restart failure, and unreadable seat CSV.

- [ ] **Step 6: Commit and review**

```powershell
git add Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs docs/ml/architecture.md docs/ml/experiment-guide.md docs/ml/troubleshooting.md
git commit -m "docs(ml): document faithful live viewing"
git status --short
git diff --check HEAD~6..HEAD
```

Expected: only the pre-existing Python bytecode-cache modification remains uncommitted; no whitespace errors.
