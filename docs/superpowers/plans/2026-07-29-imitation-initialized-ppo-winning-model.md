# Imitation-Initialized PPO Winning Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce three independently initialized learned tactical-v2 policies that reliably convert advantages into annihilation wins against Random and pass the locked 65-percent-per-model, 70-percent-pooled milestone.

**Architecture:** Preserve the validated profiled-start and bounded-search research substrate, then record scripted teachers' accepted commands as exact pre-action tactical-v2 observations, legal masks, and action indices. Train the existing MaskablePPO actor by masked behavioral cloning, copy only actor-side parameters into a fresh PPO actor-critic, fine-tune under the locked 70/30 start distribution, and select one global checkpoint budget before a single sealed final evaluation.

**Tech Stack:** C#/.NET engine and GymServer, NUnit, Python 3, NumPy compressed shards, PyTorch, Gymnasium, Stable-Baselines3, sb3-contrib MaskablePPO, pytest.

**Approved design:** [From Draws to Annihilation](../specs/2026-07-29-imitation-initialized-ppo-winning-model-design.md)

## Global Constraints

- Keep the tactical-v2 observation size, flattened action geometry, legal-mask semantics, reward, Random opponent, and standard 13-by-9 3-v-3 milestone fixed.
- A success is an authoritative annihilation win. Every draw and loss is a failure.
- Collect at least 100,000 Greedy standard decisions and 50,000 bounded-search conversion decisions, stopping only after the reciprocal pair that crosses each threshold.
- Sample BC minibatches as 70 percent standard Greedy rows and 30 percent conversion-search rows.
- Use bounded search only on the six near/far conversion profiles, with expansion budget 512, maximum depth four, and its heuristic enabled.
- Do not train on the locked 99 planner-win/PPO-draw disagreement traces or any seed in the reserved 10,000,000 confirmation namespace.
- Use these disjoint namespaces exactly: Greedy demonstrations 11,000,000-11,499,999; search demonstrations 11,500,000-11,999,999; BC validation 12,000,000-12,099,999; PPO replicates 13,000,000-15,999,999; development 16,000,000-16,000,099; final 17,000,000-17,000,249.
- Use model seeds 211, 223, and 227 independently of episode-seed bases.
- Evaluate BC and PPO on 100 development maps from both seats. Select one global PPO budget across all three models.
- PPO budgets are the first completed rollouts at or beyond 12,800, 25,600, and 51,200 environment steps.
- PPO uses learning rate 0.0003, ten nominal epochs, and target KL 0.02.
- Do not proceed from BC to PPO unless pooled clone standard win rate is at least 40 percent, every clone is at least 30 percent, and all integrity checks pass.
- The final gate is at least 65 percent wins for each model over 500 reciprocal games and at least 70 percent pooled wins over 1,500 games.
- Keep final seeds sealed until dataset, hyperparameters, and the single global checkpoint budget are frozen.
- Every persisted artifact records code revision, dirty-state policy, scenario hash, contract hash, encoding hash, source hashes, and seed provenance.
- Use TDD for every behavior change. Commit only the files named by the task; never include generated datasets, models, replays, panel evidence, or Python bytecode.

## File and Responsibility Map

| File | Responsibility |
|---|---|
| `engine/HexWars.Engine/Rl/TacticalV2Config.cs` | Versioned profiled-start catalog and basis-point distribution |
| `engine/HexWars.Engine/Rl/TacticalV2Layout.cs` | Deterministic standard and conversion start construction |
| `engine/HexWars.Engine/BoundedSearchAgent.cs` | Deterministic authoritative conversion teacher |
| `engine/HexWars.Engine/Rl/TacticalV2Coding.cs` | Single authoritative command/action codec |
| `engine/HexWars.Engine/Rl/TacticalV2Demonstration.cs` | Pre-action decision DTO and opt-in buffered sink |
| `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs` | Accepted-command demonstration capture |
| `engine/HexWars.GymServer/Program.cs` | Bounded-search controller and demonstration RPC endpoints |
| `python/ml_lab/imitation.py` | Dataset schema, validation, sampling, BC, metrics, and actor fixtures |
| `python/collect_annihilation_demonstrations.py` | Restart-safe reciprocal teacher-data collection |
| `python/ml_lab/algorithms.py` | Fresh-model creation and explicit actor-only initialization |
| `python/ml_lab/contracts.py` | Recorded algorithm options and actor-initialization provenance |
| `python/ml_lab/training.py` | Actor-initialized PPO run construction |
| `python/run_annihilation_imitation_panel.py` | Staged experiment orchestration, gates, selection, and reporting |
| `python/config/annihilation-imitation-v1.json` | Locked profiled tactical-v2 70/30 scenario |
| `python/panels/annihilation-imitation-v1/PROTOCOL.md` | Human-readable preregistration and runbook |
| `python/panels/annihilation-imitation-v1/panel.json` | Machine-readable locked experiment definition |
| `python/panels/annihilation-imitation-v1/seed-banks.json` | Explicit namespaces and final-bank assignment state |

Generated data lives below `python/datasets/annihilation-imitation-v1/` and generated run/evaluation evidence below `python/panels/annihilation-imitation-v1/`. Add only small definitions and reports intentionally; model archives, numeric shards, raw traces, and replays remain ignored.

---

### Task 1: Integrate and Freeze Profiled Tactical-v2 Starts

**Files:**
- Modify: `engine/HexWars.Engine/Rl/TacticalV2Config.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2Layout.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2Env.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs`
- Modify: `python/ml_lab/scenarios.py`
- Modify: `python/ml_lab/protocol.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/ml_lab/evaluation.py`
- Modify: `python/tests/test_scenarios.py`
- Modify: `python/tests/test_evaluation.py`

**Interfaces:**
- Produces: `TacticalV2StartCatalog.ProfiledSeededV1()`, `TacticalV2StartDistribution.Select(int seed)`, `TacticalV2Layout.NewGame(int seed, TacticalV2StartProfile profile, PlayerId referenceSeat)`.
- Produces: `TacticalV2DuelEnv.Reset(..., string startProfileId, PlayerId referenceSeat)` and GymServer `duel_reset` fields `start_profile` and `reference_seat`.
- Produces: validated Python scenario fields `start_profiles` and `start_distribution`.

- [ ] **Step 1: Write failing engine tests for the exact versioned catalog**

```csharp
[Test]
public void ProfiledSeededV1_DeclaresExactCatalog()
{
    Assert.That(
        TacticalV2StartCatalog.ProfiledSeededV1().Select(p => p.Id),
        Is.EqualTo(new[]
        {
            "standard-3v3",
            "conversion-3v1-near", "conversion-3v1-medium", "conversion-3v1-far",
            "conversion-2v1-near", "conversion-2v1-medium", "conversion-2v1-far",
            "conversion-1v1-near", "conversion-1v1-medium", "conversion-1v1-far",
        }));
}
```

- [ ] **Step 2: Run the focused engine tests and confirm the missing types fail**

Run:
```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter "FullyQualifiedName~TacticalV2ConfigTests|FullyQualifiedName~TacticalV2EnvTests|FullyQualifiedName~TacticalV2DuelEnvTests"
```

Expected: FAIL because the profiled-start catalog and overloads are absent.

- [ ] **Step 3: Port the validated profiled-start types and deterministic constructor**

Use immutable `TacticalV2StartProfile` and `TacticalV2StartWeight` values. Validate the exact catalog, require weights to sum to 10,000 basis points, sort by ordinal profile ID before deterministic selection, and retain `symmetric-random-v1` unchanged. Conversion construction must independently sample both compositions and enforce the declared near/medium/far separation.

```csharp
public TacticalV2Start NewGame(
    int seed,
    TacticalV2StartProfile profile,
    PlayerId referenceSeat);

public View Reset(
    int seed,
    IAgent? controller0,
    IAgent? controller1,
    PlayerId learnerSeat,
    string startProfileId,
    PlayerId referenceSeat);
```

- [ ] **Step 4: Add deterministic and seat-symmetry tests**

Test identical seed/profile/reference-seat equality, disjoint reference-seat role reversal, exact live-unit counts, separation bounds, standard-profile legacy equivalence, and failure on undeclared profiles. Assert worker count does not change selected profile or constructed state.

- [ ] **Step 5: Add Python scenario and protocol tests**

```python
def test_profiled_scenario_requires_exact_catalog_and_10000_basis_points():
    scenario = profiled_scenario()
    validated = validate_scenario_document(scenario)
    assert sum(
        item["basis_points"]
        for item in validated["tactical_v2"]["start_distribution"]
    ) == 10_000

def test_duel_view_accepts_profile_and_reference_seat():
    obs, mask = validate_view_payload(
        {"obs": [0.0], "mask": [True], "start_profile": "conversion-1v1-far",
         "reference_seat": 1},
        observation_size=1,
        action_size=1,
    )
    assert obs.shape == (1,)
    assert mask.tolist() == [True]
```

- [ ] **Step 6: Run focused Python and engine tests**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_scenarios.py python/tests/test_evaluation.py -q
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter "FullyQualifiedName~TacticalV2"
```

- [ ] **Step 7: Commit the profiled-start integration**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Config.cs engine/HexWars.Engine/Rl/TacticalV2Layout.cs engine/HexWars.Engine/Rl/TacticalV2Env.cs engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs python/ml_lab/scenarios.py python/ml_lab/protocol.py python/ml_lab/envs.py python/ml_lab/evaluation.py python/tests/test_scenarios.py python/tests/test_evaluation.py
git commit -m "feat: integrate deterministic tactical-v2 start profiles"
```

---

### Task 2: Integrate the Bounded-search Conversion Teacher

**Files:**
- Create: `engine/HexWars.Engine/BoundedSearchAgent.cs`
- Create: `engine/HexWars.Engine.Tests/BoundedSearchAgentTests.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`

**Interfaces:**
- Consumes: authoritative `LegalMoves.For(GameState)` and `GameEngine.Apply(GameState, Command)`.
- Produces: `new BoundedSearchAgent(int expansionBudget = 512, int depth = 4, bool useHeuristic = true)`.
- Produces: GymServer scripted controller name `bounded-search`.

- [ ] **Step 1: Write failing bounded-search tests**

```csharp
[Test]
public void Decide_TakesAnAuthoritativeTerminalWin()
{
    GameState state = OneHitWinState();
    var teacher = new BoundedSearchAgent(expansionBudget: 64, depth: 3);
    Command selected = teacher.Decide(state);
    Result result = GameEngine.Apply(state, selected);
    Assert.That(result.Success, Is.True);
    Assert.That(result.NewState.Winner, Is.EqualTo(PlayerId.Player0));
}

[Test]
public void IdenticalTeachers_AreDeterministic()
{
    GameState state = NonterminalFixture();
    Assert.That(
        Describe(new BoundedSearchAgent().Decide(state)),
        Is.EqualTo(Describe(new BoundedSearchAgent().Decide(state))));
}
```

- [ ] **Step 2: Run the test and verify the missing teacher fails**

Run:
```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter FullyQualifiedName~BoundedSearchAgentTests
```

Expected: FAIL because `BoundedSearchAgent` is not present on the isolated branch.

- [ ] **Step 3: Port the validated deterministic search implementation**

Retain authoritative transitions, attack/move/deploy/EndTurn ordering, lexical tie-breaking, terminal values `+1/0/-1`, health-adjusted material, persistent-target pursuit, and the nonterminal clamp below absolute terminal value.

```csharp
public const int DefaultExpansionBudget = 512;
public const int DefaultDepth = 4;

public Command Decide(GameState state)
{
    // Enumerate ordered legal commands, allocate the fixed expansion budget,
    // recurse through GameEngine.Apply, and use deterministic tie breaks.
}
```

- [ ] **Step 4: Register only the locked server-side teacher form**

```csharp
if (spec == "bounded-search")
    return new BoundedSearchAgent(
        BoundedSearchAgent.DefaultExpansionBudget,
        BoundedSearchAgent.DefaultDepth,
        useHeuristic: true);
```

Do not add bounded search to the learned-controller resolver or normal product UI. It is a data-generation controller.

- [ ] **Step 5: Run focused and full engine tests**

Run:
```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter FullyQualifiedName~BoundedSearchAgentTests
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo
```

Expected: all tests pass, and the teacher's per-decision expansion count never exceeds 512.

- [ ] **Step 6: Commit the teacher**

```powershell
git add engine/HexWars.Engine/BoundedSearchAgent.cs engine/HexWars.Engine.Tests/BoundedSearchAgentTests.cs engine/HexWars.GymServer/Program.cs
git commit -m "feat: add bounded-search demonstration teacher"
```

---

### Task 3: Capture Exact Pre-action Teacher Labels

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV2Demonstration.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2Coding.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs`
- Modify: `python/ml_lab/evaluation.py`
- Modify: `python/tests/test_evaluation.py`

**Interfaces:**
- Produces: `TacticalV2Coding.TryEncode(Command, GameState, TacticalV2Layout, TacticalV2UnitRegistry, out int)`.
- Produces: `ITacticalV2DemonstrationSink`, `BufferedTacticalV2DemonstrationSink`, and immutable `TacticalV2Demonstration`.
- Produces: GymServer RPCs `duel_demo_enable` and `duel_demo_drain`.
- Produces: `DuelClient.enable_demonstrations(bool)` and `DuelClient.drain_demonstrations()`.

- [ ] **Step 1: Write failing codec round-trip tests**

```csharp
[Test]
public void EveryMaskedAction_RoundTripsThroughThePublicEncoder()
{
    TacticalFixture f = TacticalFixture.Standard(seed: 73);
    bool[] mask = TacticalV2Coding.Mask(f.State, f.Seat, f.Layout, f.Own);
    foreach (int action in Enumerable.Range(0, mask.Length).Where(i => mask[i]))
    {
        Command command = TacticalV2Coding.Decode(action, f.State, f.Seat, f.Layout, f.Own);
        Assert.That(
            TacticalV2Coding.TryEncode(command, f.State, f.Layout, f.Own, out int encoded),
            Is.True);
        Assert.That(encoded, Is.EqualTo(action));
    }
}
```

Also test that unsupported, wrong-seat, dead-unit, missing-target, and off-board commands return `false` without mapping to EndTurn.

- [ ] **Step 2: Run the codec test and confirm it fails**

Run:
```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter FullyQualifiedName~TacticalV2CodingTests
```

Expected: FAIL because the authoritative encoder is private.

- [ ] **Step 3: Expose a fail-closed wrapper around the existing encoder**

```csharp
public static bool TryEncode(
    Command command,
    GameState state,
    TacticalV2Layout layout,
    TacticalV2UnitRegistry own,
    out int action)
{
    if (command == null) throw new ArgumentNullException(nameof(command));
    if (state == null) throw new ArgumentNullException(nameof(state));
    if (layout == null) throw new ArgumentNullException(nameof(layout));
    ValidateRegistry(own, layout, nameof(own));
    int encoded = Encode(command, state, own, layout);
    if (encoded < 0 || encoded >= layout.ActionCount)
    {
        action = -1;
        return false;
    }
    action = encoded;
    return true;
}
```

`Mask` must call this wrapper so there remains exactly one command-to-index implementation.

- [ ] **Step 4: Write failing opt-in capture tests**

Cover one fully scripted Greedy-versus-Random episode and assert for every captured decision:

```csharp
TacticalV2DuelEnv.View before = env.Reset(
    seed: 91, controller0: null, controller1: new RandomAgent(92));
int action = FirstLegalNonEndTurn(before.ActionMask);
env.Step(action);
TacticalV2Demonstration item = sink.Drain().Single();
Assert.That(item.Seat, Is.EqualTo(0));
Assert.That(item.Action, Is.EqualTo(action));
Assert.That(item.LegalMask[item.Action], Is.True);
CollectionAssert.AreEqual(before.Observation, item.Observation);
CollectionAssert.AreEqual(before.ActionMask, item.LegalMask);
```

Run the same seed with capture disabled and enabled; compare replay text, accepted commands, terminal state, winner, and rewards.

- [ ] **Step 5: Implement the immutable decision DTO and buffered sink**

```csharp
public sealed class TacticalV2Demonstration
{
    public float[] Observation { get; }
    public bool[] LegalMask { get; }
    public int Action { get; }
    public int Seat { get; }
    public TacticalTraceCommand Command { get; }
}

public interface ITacticalV2DemonstrationSink
{
    void Reset();
    void Accepted(TacticalV2Demonstration decision);
}
```

Make `TacticalEvaluationTrace.ProjectCommand(Command)` internal and reuse it for the structured command projection; do not create a second command-kind mapping.

The buffered implementation defaults to disabled, copies the numeric arrays, drains in decision order, and clears on reset. It must not retain `GameState` references.

- [ ] **Step 6: Capture only successfully applied commands from their pre-action state**

In `TacticalV2DuelEnv.TryApply`, call `GameEngine.Apply` first. If it succeeds, project `before` using the acting and opposing registries before either registry is updated. Encode the accepted command, require `mask[action]`, send the immutable decision to the sink, and only then advance state and registries. Throw on any internal encode/mask mismatch because silently dropping a teacher action corrupts the dataset.

- [ ] **Step 7: Add versioned GymServer and Python client endpoints**

```json
{"cmd":"duel_demo_enable","enabled":true}
{"cmd":"duel_demo_drain"}
{"schema_version":1,"decisions":[{"Observation":[],"LegalMask":[],"Action":0,"Seat":0,"Command":{}}]}
```

```python
def enable_demonstrations(self, enabled: bool) -> None:
    response = self._rpc({"cmd": "duel_demo_enable", "enabled": enabled})
    if response != {"enabled": enabled}:
        raise ValueError("GymServer did not acknowledge demonstration capture")

def drain_demonstrations(self) -> list[dict[str, Any]]:
    payload = self._rpc({"cmd": "duel_demo_drain"})
    return validate_demonstration_payload(payload, self.contract)
```

Reject wrong schema versions, observation/mask lengths, illegal labels, invalid seats, and non-finite observations.

- [ ] **Step 8: Run focused tests and commit**

Run:
```powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo --filter "FullyQualifiedName~TacticalV2CodingTests|FullyQualifiedName~TacticalV2DuelEnvTests"
$env:PYTHONPATH='python'
python -m pytest python/tests/test_evaluation.py -q
```

Commit:
```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Demonstration.cs engine/HexWars.Engine/Rl/TacticalV2Coding.cs engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs python/ml_lab/evaluation.py python/tests/test_evaluation.py
git commit -m "feat: capture tactical-v2 teacher decisions"
```

---

### Task 4: Build Restart-safe Demonstration Collection

**Files:**
- Create: `python/ml_lab/imitation.py`
- Create: `python/collect_annihilation_demonstrations.py`
- Create: `python/tests/test_imitation.py`
- Create: `python/tests/test_collect_annihilation_demonstrations.py`
- Modify: `.gitignore`

**Interfaces:**
- Consumes: `DuelClient.enable_demonstrations`, `DuelClient.reset`, `DuelClient.drain_demonstrations`, and `DuelClient.save_replay`.
- Produces: `DemonstrationWriter.append_game(game: DemonstrationGame, decisions: Sequence[DecisionRow])`.
- Produces: `collect_partition(CollectionSpec) -> Path`.

- [ ] **Step 1: Write failing writer tests for atomic, normalized artifacts**

```python
def test_writer_commits_one_complete_game_atomically(tmp_path: Path):
    writer = DemonstrationWriter.create(tmp_path, contract=contract(), shard_rows=4)
    writer.append_game(game(seed=11_000_000), decisions(3))
    writer.close()

    manifest = read_json(tmp_path / "manifest.json")
    games = read_jsonl(tmp_path / "games.jsonl")
    assert manifest["decision_count"] == 3
    assert games[0]["seed"] == 11_000_000
    assert games[0]["row_count"] == 3
    assert all((tmp_path / item["path"]).is_file() for item in manifest["shards"])
```

Also simulate interruption before manifest replacement and verify reopening neither duplicates the game nor exposes an incomplete shard.

- [ ] **Step 2: Define the storage schema without adding dependencies**

Use compressed NumPy `.npz` shards of at most 4,096 rows:

```python
@dataclass(frozen=True)
class DecisionBatch:
    observations: np.ndarray       # float32 [N, observation_size]
    packed_masks: np.ndarray       # uint8 [N, ceil(action_size / 8)]
    actions: np.ndarray            # int64 [N]
    game_ids: np.ndarray           # int64 [N]
    decision_indices: np.ndarray   # int32 [N]
    seats: np.ndarray              # uint8 [N]
    action_kinds: np.ndarray       # uint8 [N]
```

`games.jsonl` owns seed, teacher seat, teacher type and parameters, opponent, profile, partition, replay path/hash, row span, outcome, and scenario/contract/encoding hashes. `manifest.json` owns schema version, code revision, dirty-state flag, source ranges, counts, and SHA-256 for every shard and replay. This normalization avoids repeating large provenance strings per row.

- [ ] **Step 3: Enforce namespace and integrity checks in the writer**

```python
FORBIDDEN_RANGES = (
    range(10_000_000, 10_100_000),
    range(16_000_000, 16_000_100),
    range(17_000_000, 17_000_250),
)

def validate_decision(row: Mapping[str, Any], contract: EnvironmentContract) -> None:
    observation = np.asarray(row["observation"], dtype=np.float32)
    mask = np.asarray(row["legal_mask"], dtype=bool)
    action = int(row["action"])
    if observation.shape != (contract.observation_size,) or not np.isfinite(observation).all():
        raise ValueError("invalid demonstration observation")
    if mask.shape != (contract.action_size,) or not mask[action]:
        raise ValueError("demonstration action is not legal")
```

Reject seed reuse across partitions, duplicate game keys, gaps in decision indices, inconsistent hashes, unknown profiles, wrong teacher/profile combinations, and replay hash mismatches.

- [ ] **Step 4: Write failing reciprocal-collection tests with a fake client**

Assert that a standard pair requests:

```python
[
    {"seed": 11_000_000, "p0": "greedy", "p1": "random",
     "start_profile": "standard-3v3", "reference_seat": 0},
    {"seed": 11_000_000, "p0": "random", "p1": "greedy",
     "start_profile": "standard-3v3", "reference_seat": 1},
]
```

Assert conversion pairs substitute `bounded-search`, use only the six near/far profiles, and set the teacher seat as `reference_seat`. The collector must retain only decisions whose `seat` equals the teacher seat.

- [ ] **Step 5: Implement deterministic collection and resume**

Expose:

```powershell
$env:PYTHONPATH='python'
python python/collect_annihilation_demonstrations.py --dataset python/datasets/annihilation-imitation-v1 --partition train --scenario python/config/annihilation-imitation-v1.json
```

Training collection stops after complete reciprocal pairs reach both locked decision thresholds. BC validation collection uses namespace 12,000,000-12,099,999 and collects 100 standard map pairs plus 20 map pairs for each of the six near/far profiles. A completed game key is `(partition, teacher, profile, seed, teacher_seat)`; resume skips only keys whose rows, replay, and hashes all validate.

Assign whole reciprocal pairs to workers by deterministic seed stride, merge results in sorted game-key order, and test that one-worker and four-worker collection produce identical commands, replay hashes, and logical dataset rows.

- [ ] **Step 6: Run collector and writer tests**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_imitation.py python/tests/test_collect_annihilation_demonstrations.py -q
```

Expected: tests pass without starting the real GymServer.

- [ ] **Step 7: Commit collection infrastructure**

```powershell
git add .gitignore python/ml_lab/imitation.py python/collect_annihilation_demonstrations.py python/tests/test_imitation.py python/tests/test_collect_annihilation_demonstrations.py
git commit -m "feat: collect validated imitation datasets"
```

---

### Task 5: Load, Validate, and Stratify Demonstration Shards

**Files:**
- Modify: `python/ml_lab/imitation.py`
- Modify: `python/tests/test_imitation.py`

**Interfaces:**
- Produces: `load_imitation_dataset(root: Path, expected_contract: EnvironmentContract) -> ImitationDataset`.
- Produces: `StratifiedDecisionSampler(dataset, standard_fraction=0.70, seed=...) `.
- Produces: `masked_cross_entropy(logits, legal_masks, actions)`.

- [ ] **Step 1: Write failing validation tests**

```python
def test_loader_rejects_contract_or_content_hash_mismatch(dataset_dir: Path):
    with pytest.raises(ContractMismatch):
        load_imitation_dataset(dataset_dir, expected_contract=other_contract())
    corrupt_first_shard(dataset_dir)
    with pytest.raises(ValueError, match="SHA-256"):
        load_imitation_dataset(dataset_dir, expected_contract=contract())

def test_loader_rejects_a_seed_shared_by_train_and_validation(dataset_dir: Path):
    duplicate_seed_across_partitions(dataset_dir)
    with pytest.raises(ValueError, match="partition"):
        load_imitation_dataset(dataset_dir, expected_contract=contract())
```

- [ ] **Step 2: Implement lazy compressed-shard indexing and mask unpacking**

Build one immutable index of `(shard, local_row)` pairs by source/profile/seat/action kind. Load compressed shards on demand, cache at most two decoded shards, and copy only selected rows into each batch. Unpack masks with:

```python
mask = np.unpackbits(packed, axis=1, count=contract.action_size, bitorder="little")
mask = mask.astype(bool, copy=False)
```

Assert every selected label remains legal after unpacking.

- [ ] **Step 3: Write the exact 70/30 sampler test**

```python
def test_sampler_emits_locked_source_ratio(dataset):
    sampler = StratifiedDecisionSampler(
        dataset, batch_size=100, standard_fraction=0.70, seed=211
    )
    batch = sampler.next_batch()
    assert (batch.sources == Source.GREEDY_STANDARD).sum() == 70
    assert (batch.sources == Source.SEARCH_CONVERSION).sum() == 30
```

For batch sizes not divisible by ten, use a deterministic residual accumulator so long-run exposure is exactly 70/30 rather than independently rounded per batch.

- [ ] **Step 4: Implement stable masked cross-entropy**

```python
def masked_cross_entropy(logits, legal_masks, actions):
    if logits.shape != legal_masks.shape:
        raise ValueError("logits and masks must have identical shape")
    if not legal_masks.gather(1, actions[:, None]).all():
        raise ValueError("teacher action is masked")
    masked_logits = logits.masked_fill(~legal_masks, torch.finfo(logits.dtype).min)
    return torch.nn.functional.cross_entropy(masked_logits, actions)
```

- [ ] **Step 5: Test deterministic order and synthetic overfit prerequisites**

Verify same sampler seed yields identical game/decision IDs, different seeds alter order without changing source counts, no batch crosses partitions, and a five-example fixture returns finite loss and gradients.

- [ ] **Step 6: Run tests and commit**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_imitation.py -q
```

Commit:
```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py
git commit -m "feat: validate and sample imitation shards"
```

---

### Task 6: Train and Publish Masked Behavioral Clones

**Files:**
- Modify: `python/ml_lab/imitation.py`
- Modify: `python/tests/test_imitation.py`

**Interfaces:**
- Produces: `BehavioralCloningConfig` and `train_behavioral_clone(...) -> BehavioralCloningResult`.
- Produces: metadata-backed clone run directories loadable as `run:PATH`.
- Produces: `actor-fixtures.npz` for exact transfer verification.

- [ ] **Step 1: Write a failing synthetic overfit test**

```python
def test_behavioral_clone_overfits_a_five_example_masked_dataset(tiny_env, tmp_path):
    result = train_behavioral_clone(
        dataset=five_rows_repeated(),
        env=tiny_env,
        contract=contract(),
        spaces_info=spaces(),
        run_dir=tmp_path / "bc",
        config=BehavioralCloningConfig(
            model_seed=211,
            batch_size=5,
            learning_rate=3e-4,
            max_epochs=200,
            patience=200,
        ),
    )
    assert result.validation.top1_accuracy == pytest.approx(1.0)
    assert result.validation.illegal_probability == pytest.approx(0.0)
```

- [ ] **Step 2: Build the clone from the production policy class**

Create a normal `MaskablePPO` model through `MaskablePPOAdapter.create` so BC and PPO use the same `HexCNN`, policy MLP, and action head. Do not create a parallel imitation-only network.

```python
actor_parameters = chain(
    model.policy.features_extractor.parameters(),
    model.policy.mlp_extractor.policy_net.parameters(),
    model.policy.action_net.parameters(),
)
optimizer = torch.optim.Adam(actor_parameters, lr=config.learning_rate)
```

Leave `mlp_extractor.value_net` and `value_net` out of the BC optimizer.

- [ ] **Step 3: Train through sb3-contrib's masked distribution path**

```python
distribution = model.policy.get_distribution(
    observation_tensor,
    action_masks=legal_mask_tensor,
)
loss = -distribution.log_prob(action_tensor).mean()
optimizer.zero_grad(set_to_none=True)
loss.backward()
torch.nn.utils.clip_grad_norm_(tuple(actor_parameters), max_norm=1.0)
optimizer.step()
```

Use batch size 256, learning rate `3e-4`, at most 50 epochs, and validation-NLL early stopping with patience five. Restore the best validation epoch before publishing.

- [ ] **Step 4: Add complete validation metrics**

For the whole game-level validation partition, report masked NLL; top-1, top-3, and top-5 legal accuracy; expected calibration error with ten equal-width confidence bins; predicted EndTurn probability; and counts/accuracy by teacher, profile, action kind, and seat.

```python
@dataclass(frozen=True)
class CloneMetrics:
    nll: float
    top1_accuracy: float
    top3_accuracy: float
    top5_accuracy: float
    expected_calibration_error: float
    mean_end_turn_probability: float
    illegal_probability: float
    strata: Mapping[str, Mapping[str, float | int]]
```

- [ ] **Step 5: Publish a contract-backed clone run**

Write `run.json`, `scenario.json`, `bc.json`, `metrics.json`, `checkpoints/step_000000000.zip`, and `actor-fixtures.npz` atomically. The run manifest uses algorithm `maskable_ppo`, policy `HexCNN`, step zero, the exact environment contract, dataset manifest hash, BC config, model seed, and best epoch.

Choose 32 validation rows deterministically by sorted `(game_id, decision_index)`, requiring at least one non-EndTurn action and both seats when available:

```python
np.savez_compressed(
    run_dir / "actor-fixtures.npz",
    observations=fixtures.observations.astype(np.float32),
    legal_masks=fixtures.legal_masks.astype(bool),
)
```

- [ ] **Step 6: Verify save/reload identity**

Load the published run through `ControllerResolver`, recompute masked logits on `actor-fixtures.npz`, and require `torch.testing.assert_close` with `rtol=0` and `atol=0` on CPU. The saved artifact must never contain BC optimizer state.

- [ ] **Step 7: Run tests and commit**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_imitation.py -q
```

Commit:
```powershell
git add python/ml_lab/imitation.py python/tests/test_imitation.py
git commit -m "feat: train masked behavioral clones"
```

---

### Task 7: Transfer Only the Actor into Fresh PPO

**Files:**
- Modify: `python/ml_lab/algorithms.py`
- Modify: `python/ml_lab/contracts.py`
- Modify: `python/ml_lab/training.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/ml_lab/cli.py`
- Modify: `python/tests/test_training.py`
- Modify: `python/tests/test_cli.py`
- Modify: `python/tests/test_controllers.py`

**Interfaces:**
- Adds: `RunConfig.algorithm_options: Mapping[str, Any]`.
- Adds: `RunConfig.actor_init_source: str | None`.
- Adds: `RunConfig.episode_seed_base: int | None`.
- Adds: `AlgorithmAdapter.initialize_actor(model, source_run, expected_contract, device) -> Mapping[str, Any]`.
- Keeps: resume and actor initialization as mutually exclusive operations.

- [ ] **Step 1: Write failing run-config and CLI tests**

```python
def test_actor_initialization_is_distinct_from_resume():
    config = run_config(actor_init_source="bc/run", resume_source="ppo/run")
    with pytest.raises(ValueError, match="mutually exclusive"):
        config.to_dict()

def test_cli_records_locked_ppo_options():
    args = parse_train(
        "--actor-init", "bc/run",
        "--learning-rate", "0.0003",
        "--ppo-epochs", "10",
        "--target-kl", "0.02",
        "--episode-seed-base", "13000000",
    )
    config = _training_config(args)
    assert config.algorithm_options == {
        "learning_rate": 0.0003,
        "n_epochs": 10,
        "target_kl": 0.02,
    }
    assert config.episode_seed_base == 13_000_000
```

- [ ] **Step 2: Validate and pass explicit algorithm options**

`MaskablePPOAdapter.create` accepts only `learning_rate`, `n_epochs`, and `target_kl`; it rejects unknown keys and supplies existing defaults when the mapping is empty. Pass validated values directly to `MaskablePPO` and preserve them in `run.json`. `TrainingEnvironmentFactory` passes `config.episode_seed_base if it is not None else config.seed` to `WorkerSchedule`, preserving legacy behavior while separating network initialization from episode assignment.

```python
allowed = {"learning_rate", "n_epochs", "target_kl"}
unknown = set(options) - allowed
if unknown:
    raise ValueError(f"unsupported MaskablePPO option {sorted(unknown)[0]!r}")
```

- [ ] **Step 3: Write a failing actor-transfer identity test**

```python
def test_actor_transfer_preserves_masked_logits_but_not_value_parameters(...):
    source = trained_clone()
    target = fresh_ppo(seed=999)
    source_values = clone_value_state(source)
    target_values_before = clone_value_state(target)

    provenance = adapter.initialize_actor(
        target, source_run=source.run_dir,
        expected_contract=contract(), device="cpu"
    )

    assert_masked_logits_equal(source, target, fixtures())
    assert clone_value_state(target) == target_values_before
    assert clone_value_state(target) != source_values
    assert target.policy.optimizer.state == {}
    assert provenance["source_checkpoint_sha256"] == sha256(source.checkpoint)
```

- [ ] **Step 4: Copy an explicit actor module map**

```python
ACTOR_MODULES = {
    "features_extractor": lambda policy: policy.features_extractor,
    "policy_net": lambda policy: policy.mlp_extractor.policy_net,
    "action_net": lambda policy: policy.action_net,
}
```

For every module, require identical state-dict keys, tensor shapes, and dtypes before `load_state_dict(strict=True)`. Do not copy `mlp_extractor.value_net`, `value_net`, any optimizer, rollout buffer, schedule progress, episode counters, or `num_timesteps`.

- [ ] **Step 5: Verify fixture logits during the transfer itself**

Load `actor-fixtures.npz` from the source run. Compute masked action logits before and after copying on the requested device, require exact CPU equality or documented device tolerance, and return the maximum absolute difference plus source hashes in provenance.

- [ ] **Step 6: Initialize after fresh creation, never through resume**

In `run_training`:

```python
model, resumed = create_or_resume_model(...)
if config.actor_init_source is not None:
    if resumed:
        raise ValueError("actor initialization cannot be combined with resume")
    provenance = adapter.initialize_actor(
        model,
        source_run=Path(config.actor_init_source),
        expected_contract=contract,
        device=config.device,
    )
    atomic_write_json(run_dir / "initialization.json", provenance)
```

Apply actor initialization before logger setup and before the first rollout. A later resume of this PPO run loads its whole PPO checkpoint normally and does not reapply BC.

- [ ] **Step 7: Run focused tests and commit**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_training.py python/tests/test_cli.py python/tests/test_controllers.py -q
```

Commit:
```powershell
git add python/ml_lab/algorithms.py python/ml_lab/contracts.py python/ml_lab/training.py python/ml_lab/envs.py python/ml_lab/cli.py python/tests/test_training.py python/tests/test_cli.py python/tests/test_controllers.py
git commit -m "feat: initialize fresh PPO actors from clones"
```

---

### Task 8: Lock the Scenario, Seed Banks, and Pure-clone Gate

**Files:**
- Create: `python/config/annihilation-imitation-v1.json`
- Create: `python/panels/annihilation-imitation-v1/PROTOCOL.md`
- Create: `python/panels/annihilation-imitation-v1/panel.json`
- Create: `python/panels/annihilation-imitation-v1/seed-banks.json`
- Create: `python/run_annihilation_imitation_panel.py`
- Create: `python/tests/test_annihilation_imitation_panel.py`

**Interfaces:**
- Produces: `validate_definitions() -> tuple[dict, dict, ResolvedScenario]`.
- Produces: `evaluate_clone_gate(clone_runs: Sequence[Path], ...) -> Mapping[str, Any]`.
- Produces: restart-safe panel commands `validate`, `collect`, `train-bc`, and `evaluate-bc`.

- [ ] **Step 1: Write failing immutable-definition tests**

```python
def test_locked_definitions_have_disjoint_exact_namespaces():
    panel, banks, scenario = validate_definitions()
    assert panel["model_seeds"] == [211, 223, 227]
    assert banks["final"]["start"] == 17_000_000
    assert banks["final"]["stop"] == 17_000_249
    assert banks["final"]["assigned"] is False
    assert_no_overlaps(banks)
    assert scenario.document["tactical_v2"]["start_distribution"] == locked_weights()
```

- [ ] **Step 2: Write the locked 70/30 scenario**

Use basis points:

```json
[
  {"profile_id":"standard-3v3","basis_points":7000},
  {"profile_id":"conversion-3v1-near","basis_points":500},
  {"profile_id":"conversion-3v1-medium","basis_points":0},
  {"profile_id":"conversion-3v1-far","basis_points":500},
  {"profile_id":"conversion-2v1-near","basis_points":500},
  {"profile_id":"conversion-2v1-medium","basis_points":0},
  {"profile_id":"conversion-2v1-far","basis_points":500},
  {"profile_id":"conversion-1v1-near","basis_points":500},
  {"profile_id":"conversion-1v1-medium","basis_points":0},
  {"profile_id":"conversion-1v1-far","basis_points":500}
]
```

Copy reward values from the approved profiled annihilation scenario: win `+1`, loss `-1`, draw `0`, shape scale `0.01`, step penalty `0.005`, closing weight `0`, and points weight `0.5`.

- [ ] **Step 3: Implement hash-checked staged commands**

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py validate
python python/run_annihilation_imitation_panel.py collect
python python/run_annihilation_imitation_panel.py train-bc
python python/run_annihilation_imitation_panel.py evaluate-bc
```

Every command reads and validates the same definition hashes. A stage writes to a sibling staging directory and atomically publishes only after counts, manifests, and expected outputs validate.

- [ ] **Step 4: Train all three clones without sharing initialization state**

For seeds 211, 223, and 227, rebuild a fresh production policy and BC optimizer, reuse only the immutable dataset, and write distinct run directories. Record the sampler seed separately from collection seeds.

- [ ] **Step 5: Evaluate the clone gate on development seeds**

Use exactly 100 maps starting at 16,000,000 and both candidate seats, against Random, standard profile only. Reuse `evaluate_controllers` and retain every draw/loss trace and replay. Gate code is literal:

```python
def clone_gate(per_seed_wins: Mapping[int, int]) -> bool:
    rates = {seed: wins / 200 for seed, wins in per_seed_wins.items()}
    return (
        all(rate >= 0.30 for rate in rates.values())
        and sum(per_seed_wins.values()) / 600 >= 0.40
    )
```

Integrity failures return a failed stage even if the rates pass.

- [ ] **Step 6: Test failure modes and idempotence**

Cover a 39.9-percent pooled failure, one 29.5-percent seed failure, missing reciprocal seat, duplicate seed/seat, mismatched contract, changed definition hash, and rerunning a completed stage without rewriting artifacts.

- [ ] **Step 7: Run tests and commit**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_annihilation_imitation_panel.py python/tests/test_scenarios.py -q
```

Commit:
```powershell
git add python/config/annihilation-imitation-v1.json python/panels/annihilation-imitation-v1/PROTOCOL.md python/panels/annihilation-imitation-v1/panel.json python/panels/annihilation-imitation-v1/seed-banks.json python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py
git commit -m "feat: lock imitation experiment protocol"
```

---

### Task 9: Train Initialized PPO, Controls, and Select One Global Budget

**Files:**
- Modify: `python/run_annihilation_imitation_panel.py`
- Modify: `python/tests/test_annihilation_imitation_panel.py`
- Modify: `python/panels/annihilation-imitation-v1/PROTOCOL.md`
- Modify: `python/panels/annihilation-imitation-v1/panel.json`

**Interfaces:**
- Produces: panel commands `train-ppo`, `evaluate-dev`, and `select-budget`.
- Produces: three actor-initialized runs and three from-scratch controls under identical online conditions.
- Produces: `selection.json` containing one global actual checkpoint step.

- [ ] **Step 1: Write failing run-matrix tests**

```python
def test_training_matrix_pairs_initialized_and_control_runs():
    runs = build_training_matrix(definitions())
    assert [(r.model_seed, r.episode_seed_base, r.condition) for r in runs] == [
        (211, 13_000_000, "bc_ppo"), (211, 13_000_000, "scratch_ppo"),
        (223, 14_000_000, "bc_ppo"), (223, 14_000_000, "scratch_ppo"),
        (227, 15_000_000, "bc_ppo"), (227, 15_000_000, "scratch_ppo"),
    ]
    assert all(r.scenario_sha256 == runs[0].scenario_sha256 for r in runs)
```

Assert the only config difference within each seed pair is `actor_init_source` and the condition/run name.

- [ ] **Step 2: Generate locked PPO run configurations**

```python
RunConfig(
    backend="sb3",
    algorithm="maskable_ppo",
    policy="HexCNN",
    run_name=run_name,
    seed=model_seed,
    episode_seed_base=episode_seed_base,
    total_timesteps=51_200,
    checkpoint_interval=12_800,
    workers=workers,
    device=device,
    learner_seat="alternating",
    opponent={"kind": "scripted", "name": "random"},
    trackers=[{"kind": "local"}],
    resume_source=None,
    environment="tactical-v2",
    algorithm_options={
        "learning_rate": 3e-4,
        "n_epochs": 10,
        "target_kl": 0.02,
    },
    actor_init_source=str(clone_run) if initialized else None,
)
```

Pass the replicate-specific episode base independently to `WorkerSchedule`; do not derive it from `model_seed`.

- [ ] **Step 3: Require rollout-aligned checkpoint identities**

For nominal budgets 12,800, 25,600, and 51,200, select the first published checkpoint with actual step at or above the nominal value. Record both values and reject missing, decreasing, duplicated, or non-rollout-boundary actual steps.

```python
def first_checkpoint_at_or_after(checkpoints, nominal):
    eligible = sorted(c for c in checkpoints if c.actual_step >= nominal)
    if not eligible:
        raise RuntimeError(f"no completed rollout reaches {nominal}")
    return eligible[0]
```

- [ ] **Step 4: Evaluate every condition/seed/budget on the same development schedule**

Each model receives the 100 maps beginning at 16,000,000 from both seats against Random under forced `standard-3v3`. Reuse the exact schedule for pure BC, BC-to-PPO, and scratch PPO. Store candidate seat, map seed, actual checkpoint step, outcome, trace, and replay.

- [ ] **Step 5: Write failing global-selection tests**

```python
def test_selection_uses_one_budget_for_all_model_seeds():
    selected = select_global_budget(fake_development_table())
    assert selected.nominal_step == 25_600
    assert set(selected.actual_steps) == {211, 223, 227}

def test_selection_tiebreak_order_is_locked():
    # Equal pooled standard wins: worst seed, conversion wins, draw rate, then earlier.
    assert select_global_budget(tied_table()).nominal_step == 12_800
```

- [ ] **Step 6: Implement the preregistered selection order**

Maximize pooled standard win rate, then higher worst-seed standard win rate, then pooled conversion win rate, then lower pooled draw rate, then earlier nominal budget. Selection returns one nominal budget; each model maps that nominal budget to its own recorded rollout-aligned actual step.

- [ ] **Step 7: Add restart-safe panel commands**

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py train-ppo
python python/run_annihilation_imitation_panel.py evaluate-dev
python python/run_annihilation_imitation_panel.py select-budget
```

`train-ppo` refuses to start unless the clone gate passed. `select-budget` refuses incomplete schedules and atomically writes `selection.json` with all input hashes.

- [ ] **Step 8: Run tests and commit**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_annihilation_imitation_panel.py python/tests/test_training.py -q
```

Commit:
```powershell
git add python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py python/panels/annihilation-imitation-v1/PROTOCOL.md python/panels/annihilation-imitation-v1/panel.json
git commit -m "feat: orchestrate imitation-initialized PPO"
```

---

### Task 10: Seal Final Evaluation and Produce the Course-project Report

**Files:**
- Modify: `python/run_annihilation_imitation_panel.py`
- Modify: `python/tests/test_annihilation_imitation_panel.py`
- Modify: `python/panels/annihilation-imitation-v1/PROTOCOL.md`
- Generated: `python/panels/annihilation-imitation-v1/aggregate.json`
- Generated: `python/panels/annihilation-imitation-v1/REPORT.md`

**Interfaces:**
- Produces: commands `freeze-final`, `evaluate-final`, and `report`.
- Produces: `apply_final_gate(matches) -> GateResult`.
- Consumes: `ml_lab.evaluation.wilson_interval`.

- [ ] **Step 1: Write failing final-seal tests**

```python
def test_final_bank_cannot_be_assigned_before_selection_is_frozen(panel_dir):
    with pytest.raises(RuntimeError, match="global checkpoint"):
        freeze_final(panel_dir)

def test_final_bank_is_single_use(panel_dir):
    freeze_complete_panel(panel_dir)
    freeze_final(panel_dir)
    with pytest.raises(RuntimeError, match="already assigned"):
        freeze_final(panel_dir)
```

The seal records code revision, clean/dirty state, all definition and dataset hashes, six training-run hashes, three selected initialized checkpoints, three selected control checkpoints, and the three incumbent comparator runs resolved from `--incumbent-panel`. It then flips `final.assigned` atomically and never supports unassign.

- [ ] **Step 2: Write boundary tests for the exact milestone**

```python
def test_final_gate_requires_each_seed_and_pooled_thresholds():
    assert apply_final_gate(wins={211: 325, 223: 325, 227: 400}, games=500).passed
    assert not apply_final_gate(wins={211: 324, 223: 400, 227: 400}, games=500).passed
    assert not apply_final_gate(wins={211: 325, 223: 325, 227: 399}, games=500).passed
```

The last assertion has 1,049 pooled wins and fails the required 1,050; all three per-seed thresholds alone are insufficient.

- [ ] **Step 3: Evaluate the complete reciprocal final bank once**

For every selected initialized model, use seeds 17,000,000 through 17,000,249, both candidate seats, Random opponent, and forced `standard-3v3`: exactly 500 games per model and 1,500 total. Refuse partial publication, duplicated keys, missing seats, seeds outside the bank, or a second invocation after completion.

- [ ] **Step 4: Compute primary and paired secondary statistics**

Report W/L/D counts and rates; Wilson 95-percent intervals for each; per-seat summaries; rounds, decisions, action waste, peak material advantage, and draw categories. Pair initialized outcomes with scratch and incumbent PPO by `(seed, candidate_seat)`; use an exact two-sided sign test on discordant wins.

```python
def exact_sign_test(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(left_only, right_only) + 1))
    return min(1.0, 2.0 * tail / (2 ** n))
```

- [ ] **Step 5: Generate a report that cannot substitute secondary metrics**

`REPORT.md` starts with the primary gate table for seeds 211, 223, 227 and pooled outcomes. It then reports clone, initialized PPO, scratch PPO, and incumbent comparisons; conversion performance; BC metrics; learning curves; compute; failure traces; and limitations. A lopsided draw remains in the draw column regardless of material diagnostics.

- [ ] **Step 6: Test atomic publication and report consistency**

Recompute every aggregate from the raw match table in the test, assert report numbers equal `aggregate.json`, and inject a publication failure to prove neither final file becomes visible alone.

- [ ] **Step 7: Run tests and commit evaluation code**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests/test_annihilation_imitation_panel.py python/tests/test_evaluation.py -q
```

Commit:
```powershell
git add python/run_annihilation_imitation_panel.py python/tests/test_annihilation_imitation_panel.py python/panels/annihilation-imitation-v1/PROTOCOL.md
git commit -m "feat: seal imitation milestone evaluation"
```

---

### Task 11: Run the End-to-end Smoke Gate

**Files:**
- Modify if a failure is found: only the task-owned implementation and matching test files above
- Generated: `python/panels/annihilation-imitation-v1/evidence/smoke/`

**Interfaces:**
- Exercises: real GymServer capture, replay, shard loading, BC, actor transfer, PPO rollout, checkpoint reload, and evaluation.

- [ ] **Step 1: Build the GymServer**

Run:
```powershell
dotnet build engine\HexWars.GymServer\HexWars.GymServer.csproj --nologo
```

Expected: exit code zero.

- [ ] **Step 2: Run the real smoke command**

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py smoke
```

The smoke command collects one reciprocal Greedy standard pair and one reciprocal bounded-search pair for each of the six near/far conversion profiles, trains a tiny BC run, transfers it into fresh PPO, completes one rollout, saves/reloads a checkpoint, evaluates two unused reciprocal maps, and verifies every replay and hash.

- [ ] **Step 3: Inspect the smoke manifest**

Require:

```text
reciprocal pairs = 7
games = 14
teacher labels > 0
masked labels = 0
round-trip mismatches = 0
replay mismatches = 0
actor fixture max error = 0 on CPU
PPO completed rollouts >= 1
evaluation games = 4
```

- [ ] **Step 4: Run full automated tests**

Run:
```powershell
$env:PYTHONPATH='python'
python -m pytest python/tests -q
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo
```

Expected: both commands exit zero with no failing tests.

- [ ] **Step 5: Commit only smoke fixes**

If the smoke exposed a defect, repeat its focused red/green test and commit the minimal fix. Do not commit the generated smoke dataset, models, traces, or replays.

---

### Task 12: Execute the Full Winning-model Experiment

**Files:**
- Generated: `python/datasets/annihilation-imitation-v1/`
- Generated: `python/panels/annihilation-imitation-v1/runs/`
- Generated: `python/panels/annihilation-imitation-v1/evaluations/`
- Generated: `python/panels/annihilation-imitation-v1/evidence/`
- Generated and publishable: `python/panels/annihilation-imitation-v1/aggregate.json`
- Generated and publishable: `python/panels/annihilation-imitation-v1/REPORT.md`

**Interfaces:**
- Consumes all preceding tasks.
- Produces the three selected learned-model run directories and the pass/fail research result.

- [ ] **Step 1: Record a clean execution identity**

Run:
```powershell
git status --short
git rev-parse HEAD
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py validate
```

Require a clean worktree and store the exact commit in the panel state before expensive collection.

- [ ] **Step 2: Collect and verify the frozen dataset**

Run:
```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py collect
```

Do not proceed until manifest counts exceed the two decision thresholds, all reciprocal pairs are complete, validation games occupy only their declared namespace, and every shard/replay hash revalidates.

- [ ] **Step 3: Train and gate the three pure clones**

Run:
```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py train-bc
python python/run_annihilation_imitation_panel.py evaluate-bc
```

If the gate fails, stop before PPO and publish the BC failure report. Follow the approved diagnosis: low supervised accuracy means inspect representation/data/optimization; high supervised accuracy with poor games triggers a separately designed DAgger condition.

- [ ] **Step 4: Train initialized and scratch PPO conditions**

Run:
```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py train-ppo
python python/run_annihilation_imitation_panel.py evaluate-dev
python python/run_annihilation_imitation_panel.py select-budget
```

Inspect `selection.json` for complete paired schedules and one global nominal budget. Do not choose different budgets per seed.

- [ ] **Step 5: Freeze comparator identities and the final bank**

Resolve the three already-trained incumbent PPO runs from the existing annihilation baseline panel:

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py freeze-final --incumbent-panel python/panels/annihilation-conversion-curriculum-v1
```

The command requires exactly three metadata-backed profiled-standard model runs in that panel, resolves and hashes them before assigning the bank, and fails closed on missing evidence or contract mismatch.

- [ ] **Step 6: Run the final bank exactly once and publish the report**

```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py evaluate-final
python python/run_annihilation_imitation_panel.py report
```

Read the primary table before interpreting diagnostics. The result passes only at 325 wins per model and 1,050 pooled wins or better.

- [ ] **Step 7: Verify evidence and commit the small result artifacts**

Run:
```powershell
$env:PYTHONPATH='python'
python python/run_annihilation_imitation_panel.py verify
git diff --check
```

If verification succeeds:

```powershell
git add python/panels/annihilation-imitation-v1/aggregate.json python/panels/annihilation-imitation-v1/REPORT.md
git commit -m "docs: report imitation-initialized PPO results"
```

Do not commit datasets, model archives, raw traces, or replays.
