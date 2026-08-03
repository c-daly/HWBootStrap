# Tactical-v2 Roster and Unit Count Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add tactical-v2 training with a selectable 1–12 starting-unit count, deterministic symmetric armies sampled from a selected player's defaults-plus-saved template catalog, while preserving tactical-v1 and adaptive-v1 behavior.

**Architecture:** Tactical-v2 is a new engine and wire contract that separates ordered template roles from stable controllable unit slots. Unity snapshots the selected session roster into `scenario.json`; Python and GymServer validate that immutable snapshot; each reset samples a symmetric composition with replacement from the snapshot using the episode seed.

**Tech Stack:** C# / .NET Standard 2.1 engine, .NET 8 GymServer, NUnit, Unity 6 Editor tooling and EditMode tests, Python 3.11+, Gymnasium, NumPy, pytest.

## Global Constraints

- Preserve tactical-v1 observation/action dimensions, hashes, checkpoints, scenarios, and Arena playback.
- Preserve adaptive-v1's current 1–24 starting-unit control and hidden deployment behavior.
- New tactical experiments default to `tactical-v2`; legacy `tactical-v1` remains explicitly selectable.
- Tactical-v2 starting-unit count is 1–12.
- The template selection space is the five canonical defaults plus every valid template saved by the selected local player.
- Normalize roster snapshots with `BarracksCatalog.Normalize`; do not read live Unity state from Python or GymServer.
- Sample with replacement using a seed-derived RNG and give both seats the same sampled composition.
- Snapshot template IDs, names, all nine stats, starting count, slot capacity, and placement-policy version into the immutable run scenario.
- Different counts, template order, names, or stats must produce different tactical-v2 encoding identities.
- Do not persist `SessionBarracksCache`, add manual per-role counts, add asymmetric armies, or implement exact placement/viewer presentation in this slice.
- Use `apply_patch` for edits.
- After every C# edit, run Coplay `check_compile_errors` and fix all errors before continuing.
- Run determinism-sensitive tests after episode-construction or action-codec changes.
- Never add attribution trailers or tool credits to commits or PR text.

## File and Interface Map

### New engine files

- `engine/HexWars.Engine/Rl/TacticalV2Config.cs` — versioned template catalog, validation, stable IDs, and seeded starting-army sampling.
- `engine/HexWars.Engine/Rl/TacticalV2Layout.cs` — board cells, action-region offsets, observation dimensions, and deterministic start-state construction.
- `engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs` — stable unit-slot and template-role identity for both initial units and reinforcements.
- `engine/HexWars.Engine/Rl/TacticalV2Coding.cs` — seat-relative observations, masks, and action encode/decode.
- `engine/HexWars.Engine/Rl/TacticalV2Env.cs` — single-learner tactical-v2 environment.
- `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs` — two-seat tactical-v2 duel environment used by GymServer and Unity Arena.

### New Unity Editor file

- `Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs` — selectable local-player roster snapshot and conversion to scenario templates.
- `Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs.meta` — Unity asset metadata.

### New tests

- `engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs`
- `Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs`
- `Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs.meta`

### Existing cross-stack files

- `engine/HexWars.Engine/Rl/MlContract.cs`
- `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- `engine/HexWars.Engine.Tests/MlContractTests.cs`
- `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs`
- `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs`
- `engine/HexWars.GymServer/ScenarioJson.cs`
- `engine/HexWars.GymServer/Program.cs`
- `python/config/training-game-templates.json`
- `python/ml_lab/scenarios.py`
- `python/ml_lab/cli.py`
- `python/ml_lab/contracts.py`
- `python/ml_lab/benchmark.py`
- `python/ml_lab/controllers.py`
- `python/ml_lab/doctor.py`
- `python/ml_lab/evaluation.py`
- `python/hexwars_gym/env.py`
- `python/selfplay_env.py`
- `python/duel.py`
- `python/policy_server.py`
- `python/tests/test_scenarios.py`
- `python/tests/test_cli.py`
- `python/tests/test_gym_client.py`
- `python/tests/test_run_contract.py`
- `python/tests/test_controllers.py`
- `python/tests/test_duel.py`
- `python/tests/test_evaluation.py`
- `python/tests/test_policy_server.py`
- `Assets/HexWars/Presentation/MlEnvironmentContract.cs`
- `Assets/HexWars/Presentation/ModelDuelEnvironment.cs`
- `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`
- `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`
- `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`
- `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`
- `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- `python/README.md`
- `docs/ml/architecture.md`
- `docs/ml/experiment-guide.md`

---

### Task 1: Tactical-v2 Catalog and Deterministic Army Sampling

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV2Config.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs`
- Modify: `engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

**Interfaces:**
- Produces: `TacticalV2Template(string id, UnitTemplate template)`
- Produces: `TacticalV2Config.Default()`
- Produces: `TacticalV2Config.Validate(): IReadOnlyList<string>`
- Produces: `TacticalV2Config.SampleStartingArmy(int seed): IReadOnlyList<TacticalV2Template>`
- Produces: `TacticalV2TemplateIds.From(UnitTemplate template): string`
- Consumes later: Training scenarios, layout construction, contract hashing, and Unity roster snapshots.

- [ ] **Step 1: Write failing catalog and sampling tests**

```csharp
[Test]
public void Default_UsesCanonicalCatalogAndThreeStartingSlots()
{
    TacticalV2Config config = TacticalV2Config.Default();

    Assert.That(config.Templates.Select(item => item.Template.Name),
        Is.EqualTo(BarracksCatalog.DefaultTemplates.Select(item => item.Name)));
    Assert.That(config.StartingUnitCount, Is.EqualTo(3));
    Assert.That(config.MaxControllableUnits, Is.EqualTo(3));
    Assert.That(config.Validate(), Is.Empty);
}

[Test]
public void SampleStartingArmy_IsSeededWithReplacementAndSymmetricInput()
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = 12;
    config.MaxControllableUnits = 12;

    string[] first = config.SampleStartingArmy(37).Select(item => item.Id).ToArray();
    string[] second = config.SampleStartingArmy(37).Select(item => item.Id).ToArray();

    Assert.That(second, Is.EqualTo(first));
    Assert.That(first, Has.Length.EqualTo(12));
    Assert.That(first.Distinct().Count(), Is.LessThan(12));
}

[TestCase(0)]
[TestCase(13)]
public void Validate_RejectsStartingCountsOutsideRegularGameLimit(int count)
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = count;
    config.MaxControllableUnits = count;

    Assert.That(config.Validate(),
        Has.Some.Contains("starting unit count must be between 1 and 12"));
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter TacticalV2ConfigTests
```

Expected: FAIL because `TacticalV2Config` and `TacticalV2Template` do not exist.

- [ ] **Step 3: Implement the catalog model, stable IDs, validation, and sampler**

Create these exact public shapes:

```csharp
public sealed class TacticalV2Template
{
    public TacticalV2Template(string id, UnitTemplate template);
    public string Id { get; }
    public UnitTemplate Template { get; }
}

public static class TacticalV2TemplateIds
{
    public static string From(UnitTemplate template);
}

public sealed class TacticalV2Config
{
    public BoardGenConfig BoardGen { get; set; }
    public GameConfig Game { get; set; }
    public IReadOnlyList<TacticalV2Template> Templates { get; set; }
    public int StartingUnitCount { get; set; }
    public int MaxControllableUnits { get; set; }
    public int MaxSteps { get; set; }
    public float ShapeScale { get; set; }
    public float StepPenalty { get; set; }
    public float ClosingWeight { get; set; }
    public float DrawCreditWeight { get; set; }
    public float PointsWeight { get; set; }
    public string PlacementPolicy { get; set; }

    public static TacticalV2Config Default();
    public IReadOnlyList<string> Validate();
    public IReadOnlyList<TacticalV2Template> SampleStartingArmy(int seed);
}
```

Implementation requirements:

```csharp
var rng = new Random(seed ^ 0x5A17);
while (result.Count < StartingUnitCount)
    result.Add(Templates[rng.Next(Templates.Count)]);
```

`TacticalV2TemplateIds.From` must sanitize the name, lowercase an ASCII slug, append a deterministic eight-hex SHA-256 prefix derived from the sanitized name plus all nine stats, and therefore distinguish equal names with different stats. Validate non-empty unique IDs, non-empty catalogs, `1 <= StartingUnitCount <= 12`, `MaxControllableUnits == StartingUnitCount`, and `PlacementPolicy == "symmetric-random-v1"`.

- [ ] **Step 4: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 5: Run focused and determinism tests**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TacticalV2ConfigTests|ArmyCompositionTests"
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Config.cs engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
git commit -m "feat(ml): define tactical v2 roster sampling"
```

---

### Task 2: Tactical-v2 Layout, Stable Slots, Observations, and Actions

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV2Layout.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV2Coding.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs`
- Modify: `engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

**Interfaces:**
- Consumes: `TacticalV2Config`, `TacticalV2Template`
- Produces: `TacticalV2Layout.NewGame(int seed): TacticalV2Start`
- Produces: `TacticalV2UnitRegistry.Initialize`, `ReleaseDead`, `RegisterDeployment`
- Produces: `TacticalV2Coding.Observe`, `Mask`, `Decode`
- Consumes later: Tactical-v2 single-agent and duel environments.

- [ ] **Step 1: Write failing geometry and action-region tests**

```csharp
[Test]
public void Layout_SeparatesTemplateRolesFromUnitSlots()
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = 7;
    config.MaxControllableUnits = 7;
    var layout = new TacticalV2Layout(config);
    int cells = config.BoardGen.Width * config.BoardGen.Height;

    Assert.That(layout.TemplateCount, Is.EqualTo(5));
    Assert.That(layout.UnitSlotCount, Is.EqualTo(7));
    Assert.That(layout.MoveOffset, Is.EqualTo(1));
    Assert.That(layout.AttackOffset, Is.EqualTo(1 + 7 * cells));
    Assert.That(layout.DeployOffset, Is.EqualTo(1 + 14 * cells));
    Assert.That(layout.ActionCount, Is.EqualTo(1 + (14 + 5) * cells));
    Assert.That(layout.ObservationChannels, Is.EqualTo(11));
}

[Test]
public void NewGame_SamplesOneCompositionForBothSeats()
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = 9;
    config.MaxControllableUnits = 9;
    TacticalV2Start start = new TacticalV2Layout(config).NewGame(41);

    Assert.That(start.TemplateIndices1, Is.EqualTo(start.TemplateIndices0));
    Assert.That(start.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(9));
    Assert.That(start.State.Player(PlayerId.Player1).UnitsOnBoard, Has.Count.EqualTo(9));
}
```

- [ ] **Step 2: Write failing slot identity tests**

Use two templates with identical stats but different names and IDs. Assert that initial units occupy their declared template planes, proving the codec does not recover role identity by comparing stats.

```csharp
[Test]
public void Observe_UsesRegisteredTemplateIdentityWhenStatsAreEqual()
{
    TacticalV2CodingFixture fixture =
        TacticalV2CodingFixture.WithEqualStatTemplates();
    float[] observation = TacticalV2Coding.Observe(
        fixture.Game, PlayerId.Player0, fixture.Layout,
        fixture.Slots0, fixture.Slots1);

    Assert.That(fixture.ValueAtFriendlyTemplatePlane(observation, 0), Is.GreaterThan(0f));
    Assert.That(fixture.ValueAtFriendlyTemplatePlane(observation, 1), Is.GreaterThan(0f));
}
```

Define `TacticalV2CodingFixture` as a private test helper in
`TacticalV2CodingTests.cs`. It must construct a two-template catalog whose
templates have identical stats but distinct IDs, initialize both registries
with explicit template indices, and expose
`ValueAtFriendlyTemplatePlane(float[] observation, int templateIndex)`.

- [ ] **Step 3: Run tests and verify they fail**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter TacticalV2CodingTests
```

Expected: FAIL because the tactical-v2 layout, registry, and coding types do not exist.

- [ ] **Step 4: Implement layout and start-state construction**

Create:

```csharp
public sealed class TacticalV2Start
{
    public GameState State { get; }
    public int[] TemplateIndices0 { get; }
    public int[] TemplateIndices1 { get; }
    public TacticalV2UnitRegistry Slots0 { get; }
    public TacticalV2UnitRegistry Slots1 { get; }
}
```

`TacticalV2Layout` must expose:

```csharp
public int TemplateCount { get; }
public int UnitSlotCount { get; }
public int CellCount { get; }
public int MoveOffset => 1;
public int AttackOffset => MoveOffset + UnitSlotCount * CellCount;
public int DeployOffset => AttackOffset + UnitSlotCount * CellCount;
public int ActionCount => DeployOffset + TemplateCount * CellCount;
public int ObservationChannels => 2 * TemplateCount + 1;
public int ObservationGlobals => TacticalV2Coding.Globals;
public int ObservationLength => ObservationChannels * CellCount + ObservationGlobals;
public HexCoord MirrorCell(HexCoord cell);
public TacticalV2Start NewGame(int seed);
```

Build one sampled template-index array and use it for both seats. Select Player 0's
starting cells from its deterministically sorted deployment zone, then derive every
Player 1 cell through `MirrorCell`, a 180-degree rotation around the board center.
Validate that each mirrored cell belongs to Player 1's deployment zone and is
distinct. Do not independently sort and zip the two deployment zones, because that
does not prove geometric symmetry. Reject a board whose smaller deployment zone has
fewer cells than `StartingUnitCount`.

Extend `NewGame_SamplesOneCompositionForBothSeats` to assert, for every starting
slot, that Player 1's cell equals `layout.MirrorCell(Player 0's cell)`.

- [ ] **Step 5: Implement the stable unit registry**

Create:

```csharp
public sealed class TacticalV2UnitRegistry
{
    public TacticalV2UnitRegistry(int capacity);
    public int Capacity { get; }
    public int UnitIdAt(int slot);
    public int TemplateIndexAt(int slot);
    public int SlotOf(int unitId);
    public void Initialize(IReadOnlyList<Unit> units, IReadOnlyList<int> templateIndices);
    public void ReleaseDead(GameState state, PlayerId seat);
    public void RegisterDeployment(GameState before, GameState after, PlayerId seat, int templateIndex);
}
```

`RegisterDeployment` identifies the one new living unit ID in `after`, assigns the lowest free slot, and throws if deployment exceeds capacity. Never infer template identity from `UnitStats`.

- [ ] **Step 6: Implement observation and action coding**

Create:

```csharp
public static class TacticalV2Coding
{
    public const int Globals = 5;
    public static float[] Observe(
        GameState state, PlayerId seat, TacticalV2Layout layout,
        TacticalV2UnitRegistry own, TacticalV2UnitRegistry foe);
    public static bool[] Mask(
        GameState state, PlayerId seat, TacticalV2Layout layout,
        TacticalV2UnitRegistry own);
    public static Command Decode(
        int action, GameState state, PlayerId seat, TacticalV2Layout layout,
        TacticalV2UnitRegistry own);
}
```

Use `MoveOffset`, `AttackOffset`, and `DeployOffset`; move/attack indices address stable unit slots, while deploy indices address template indices. Encode observation planes from registry template indices. Normalize alive counts by `UnitSlotCount`.

- [ ] **Step 7: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 8: Run focused and determinism tests**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TacticalV2CodingTests|MlContractTests"
```

Expected: PASS, including identical starts for identical seeds and different sampled compositions across the pinned representative seed set.

- [ ] **Step 9: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Layout.cs engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs engine/HexWars.Engine/Rl/TacticalV2Coding.cs engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
git commit -m "feat(ml): add tactical v2 action codec"
```

---

### Task 3: Tactical-v2 Single-Agent and Duel Environments

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV2Env.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs`
- Modify: `engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

**Interfaces:**
- Consumes: tactical-v2 config, layout, coding, and registries.
- Produces: GymServer-compatible single-agent reset/step/mask API.
- Produces: Arena/GymServer-compatible two-seat reset/step/view API.
- Consumes later: `Program.cs` and `ModelDuelEnvironmentFactory`.

- [ ] **Step 1: Write failing environment reset and reinforcement tests**

```csharp
[Test]
public void Reset_IsSymmetricAndReproducible()
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = config.MaxControllableUnits = 8;
    var first = new TacticalV2Env(seed => new GreedyAgent(seed), PlayerId.Player0, config);
    var second = new TacticalV2Env(seed => new GreedyAgent(seed), PlayerId.Player0, config);

    Assert.That(first.Reset(71), Is.EqualTo(second.Reset(71)));
    Assert.That(Signature(first.State, PlayerId.Player0),
        Is.EqualTo(Signature(first.State, PlayerId.Player1)));
}

[Test]
public void DeployAfterDeath_ReusesReleasedSlotWithChosenTemplateIdentity()
{
    TacticalV2EnvFixture fixture = TacticalV2EnvFixture.WithReleasedSlot();
    fixture.Environment.Step(fixture.DeployAction(template: 4, fixture.FreeCell));

    Assert.That(fixture.Environment.Slots0.TemplateIndexAt(fixture.ReleasedSlot),
        Is.EqualTo(4));
}
```

Define `TacticalV2EnvFixture` as a private test helper in
`TacticalV2EnvTests.cs`. It must create a legal state with one released slot,
one affordable catalog template at index 4, and one legal deployment cell; its
`DeployAction` helper calculates the action through the layout offsets instead
of embedding a magic integer.

- [ ] **Step 2: Run focused tests and verify they fail**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter TacticalV2EnvTests
```

Expected: FAIL because the environment types do not exist.

- [ ] **Step 3: Implement `TacticalV2Env`**

Mirror `TacticalEnv`'s opponent scheduling and reward shaping, but route reset, observe, mask, and decode through tactical-v2 types. After every accepted command:

1. retain the pre-command state;
2. apply the command;
3. release dead registry entries;
4. if the command is `DeployUnit`, register the new unit with its template index;
5. advance scripted opponents through the same registry update path.

Expose:

```csharp
public TacticalV2Config Config { get; }
public TacticalV2Layout Layout { get; }
public GameState State { get; }
public TacticalV2UnitRegistry Slots0 { get; }
public TacticalV2UnitRegistry Slots1 { get; }
public float[] Reset(int seed);
public StepResult Step(int action);
public bool[] LegalActionMask();
```

- [ ] **Step 4: Implement `TacticalV2DuelEnv`**

Mirror `DuelEnv.View` and controller behavior:

```csharp
public sealed class TacticalV2DuelEnv
{
    public sealed class View
    {
        public float[] Observation;
        public bool[] ActionMask;
        public PlayerId Seat;
        public float Reward;
        public PlayerId? Winner;
        public bool Terminated;
        public bool Truncated;
    }

    public TacticalV2DuelEnv(TacticalV2Config config);
    public TacticalV2Config Config { get; }
    public TacticalV2Layout Layout { get; }
    public GameState State { get; }
    public View Reset(int seed, IAgent controller0, IAgent controller1,
        PlayerId learnerSeat = PlayerId.Player0);
    public View Step(int action);
    public string ToReplay();
}
```

Every scripted and external command must update the same registries before the next view.

- [ ] **Step 5: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 6: Run focused, engine, and determinism tests**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TacticalV2EnvTests|TacticalV2CodingTests|TacticalEnvTests"
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Env.cs engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
git commit -m "feat(ml): run tactical v2 episodes"
```

---

### Task 4: Engine Contract, Training Scenario, and GymServer

**Files:**
- Modify: `engine/HexWars.Engine/Rl/MlContract.cs`
- Modify: `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- Modify: `engine/HexWars.Engine.Tests/MlContractTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs`
- Modify: `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs`
- Modify: `engine/HexWars.GymServer/ScenarioJson.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`

**Interfaces:**
- Produces: `MlContract.TacticalV2Version == "tactical-v2"`
- Produces: `MlContract.CreateTacticalV2(TacticalV2Config, MlEnvironmentKind)`
- Produces: `TrainingScenario.BuildTacticalV2(): TacticalV2Config`
- Produces: strict GymServer scenario parsing and tactical-v2 spaces/reset/step/duel commands.
- Consumes later: Python handshake parser and Unity scenario conversion.

- [ ] **Step 1: Write failing contract tests**

```csharp
[Test]
public void TacticalV2Contract_SeparatesSlotsAndTemplates()
{
    TacticalV2Config config = TacticalV2Config.Default();
    config.StartingUnitCount = config.MaxControllableUnits = 7;
    MlContract contract = MlContract.CreateTacticalV2(config);
    var layout = new TacticalV2Layout(config);

    Assert.That(contract.Version, Is.EqualTo("tactical-v2"));
    Assert.That(contract.ObservationSize, Is.EqualTo(layout.ObservationLength));
    Assert.That(contract.ActionSize, Is.EqualTo(layout.ActionCount));
    Assert.That(contract.Semantics["starting_unit_count"], Is.EqualTo(7));
    Assert.That(contract.Semantics["max_controllable_units"], Is.EqualTo(7));
}

[Test]
public void TacticalV1Contract_RemainsByteIdentical()
{
    TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
    scenario.Board.Width = 24;
    scenario.Board.Height = 24;
    scenario.Rules.StartingPoints = 50;
    MlContract contract = MlContract.Create(scenario.BuildTactical());

    Assert.That(contract.ContractHash, Is.EqualTo(
        "8794d90bde2455c77ba2a4c1c7a22f3fb60f5d4fdb7be766003269a8a5a08c33"));
    Assert.That(contract.EncodingHash, Is.EqualTo(
        "39c428c07a31de09137a8851c62b5e9ebc083af1729636ce8c833b59f450e49b"));
}
```

These literals come from an existing recorded tactical-v1 24×24 run. Do not
derive expected values from the new implementation.

- [ ] **Step 2: Write failing training-scenario tests**

```csharp
[Test]
public void TacticalV2Scenario_BuildsSavedCatalogAndCount()
{
    TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
    scenario.TacticalV2.StartingUnitCount = 6;
    scenario.TacticalV2.MaxControllableUnits = 6;
    scenario.TacticalV2.Templates.Add(CustomTemplate("custom-alpha", "Alpha"));

    TacticalV2Config config = scenario.BuildTacticalV2();

    Assert.That(config.StartingUnitCount, Is.EqualTo(6));
    Assert.That(config.Templates.Select(item => item.Id), Does.Contain("custom-alpha"));
}
```


- [ ] **Step 3: Run tests and verify they fail**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "MlContractTests|TrainingScenarioTests"
```

Expected: FAIL because tactical-v2 contract and scenario members do not exist.

- [ ] **Step 4: Add tactical-v2 contract identity**

Add:

```csharp
public const string TacticalV2Version = "tactical-v2";

public static MlContract CreateTacticalV2(
    TacticalV2Config config,
    MlEnvironmentKind environmentKind = MlEnvironmentKind.Tactical);
```

The `Semantics` object and canonical encoding JSON must include:

- ordered templates with `id`, `name`, `stats`, and point cost;
- `starting_unit_count`;
- `max_controllable_units`;
- `placement_policy`;
- move, attack, and deploy offsets/counts;
- ordered observation channel names;
- observation/action sizes.

Use environment kinds `"tactical"` and `"duel"` as tactical-v1 does; contract version distinguishes v1 from v2.

- [ ] **Step 5: Extend engine training scenarios**

Add DTOs:

```csharp
[Serializable]
public sealed class TrainingTacticalV2Config
{
    public int StartingUnitCount = 3;
    public int MaxControllableUnits = 3;
    public string PlacementPolicy = "symmetric-random-v1";
    public List<TrainingUnitTemplateConfig> Templates = new List<TrainingUnitTemplateConfig>();
}

[Serializable]
public sealed class TrainingUnitTemplateConfig
{
    public string Id = string.Empty;
    public string Name = string.Empty;
    public int Health;
    public int Damage;
    public int Defense;
    public int Movement;
    public int VerticalMovement;
    public int Range;
    public int RangeArc;
    public int Vision;
    public int VisionArc;
}
```

`TrainingScenario.CreateStandard("tactical-v2")` populates the five canonical defaults. Validation enforces environment-exclusive sections. `BuildTacticalV2()` maps the DTO without reading Unity state.

- [ ] **Step 6: Extend strict GymServer JSON parsing**

`ScenarioJson` must require exactly these tactical-v2 keys:

```json
"tactical_v2": {
  "starting_unit_count": 3,
  "max_controllable_units": 3,
  "placement_policy": "symmetric-random-v1",
  "templates": [
    {
      "id": "brute",
      "name": "Brute",
      "stats": {
        "health": 7,
        "damage": 2,
        "defense": 2,
        "movement": 3,
        "vertical_movement": 2,
        "range": 1,
        "range_arc": 1,
        "vision": 2,
        "vision_arc": 1
      }
    }
  ]
}
```

Reject missing/extra fields, duplicate IDs, invalid stats, wrong environment sections, and count/zone mismatches.

- [ ] **Step 7: Route GymServer commands**

Accept `"tactical-v2"` in argument validation. Instantiate `TacticalV2Env` and lazily create `TacticalV2DuelEnv`. Add a `TacticalV2Spaces` response containing the generic contract fields plus:

```csharp
tactical_v2 = contract.Semantics,
action_regions = contract.Semantics["action_regions"],
observation_channels = contract.Semantics["observation_channels"],
```

Route `spaces`, `reset`, `step`, `duel_spaces`, `duel_reset`, `duel_step`, and `duel_save` without altering tactical-v1/adaptive branches.

- [ ] **Step 8: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 9: Run engine and GymServer process tests**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "MlContractTests|TrainingScenarioTests|TacticalV2|CheckedInScenario"
```

Expected: PASS, including subprocess spaces/reset/step tests for counts 1 and 12 and unchanged tactical-v1 hashes.

- [ ] **Step 10: Commit**

```powershell
git add engine/HexWars.Engine/Rl/MlContract.cs engine/HexWars.Engine/Rl/TrainingScenario.cs engine/HexWars.Engine.Tests/MlContractTests.cs engine/HexWars.Engine.Tests/TrainingScenarioTests.cs engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs engine/HexWars.GymServer/ScenarioJson.cs engine/HexWars.GymServer/Program.cs
git commit -m "feat(ml): serve tactical v2 contracts"
```

---

### Task 5: Python Scenario, Handshake, and Training Support

**Files:**
- Modify: `python/config/training-game-templates.json`
- Modify: `python/ml_lab/scenarios.py`
- Modify: `python/ml_lab/cli.py`
- Modify: `python/ml_lab/contracts.py`
- Modify: `python/ml_lab/benchmark.py`
- Modify: `python/ml_lab/controllers.py`
- Modify: `python/ml_lab/doctor.py`
- Modify: `python/ml_lab/evaluation.py`
- Modify: `python/hexwars_gym/env.py`
- Modify: `python/selfplay_env.py`
- Modify: `python/duel.py`
- Modify: `python/policy_server.py`
- Modify: `python/tests/test_scenarios.py`
- Modify: `python/tests/test_cli.py`
- Modify: `python/tests/test_gym_client.py`
- Modify: `python/tests/test_run_contract.py`
- Modify: `python/tests/test_controllers.py`
- Modify: `python/tests/test_duel.py`
- Modify: `python/tests/test_evaluation.py`
- Modify: `python/tests/test_policy_server.py`

**Interfaces:**
- Consumes: GymServer tactical-v2 scenario and spaces JSON.
- Produces: strict `ResolvedScenario` support for tactical-v2.
- Produces: `parse_contract(... environment="tactical-v2")`.
- Produces: new-run default environment tactical-v2; explicit tactical-v1 remains supported.
- Consumes later: Unity's `--scenario-file` launch and policy-server compatibility.

- [ ] **Step 1: Write failing strict-scenario tests**

```python
def test_tactical_v2_scenario_preserves_roster_and_count():
    scenario = resolve_scenario(
        environment="tactical-v2",
        scenario_file=write_scenario(tactical_v2_document(starting_units=7)),
        template_id=None,
    )

    assert scenario.document["tactical_v2"]["starting_unit_count"] == 7
    assert [item["name"] for item in scenario.document["tactical_v2"]["templates"]][:2] == [
        "Brute", "Striker"
    ]


@pytest.mark.parametrize("count", [0, 13])
def test_tactical_v2_rejects_invalid_starting_count(count):
    document = tactical_v2_document(starting_units=count)
    with pytest.raises(ValueError, match="between 1 and 12"):
        validate_scenario_document(document)
```

- [ ] **Step 2: Write failing handshake geometry tests**

```python
def test_parse_tactical_v2_contract_uses_slots_and_templates():
    spaces = tactical_v2_spaces(template_count=5, unit_count=7, width=13, height=9)
    contract = parse_contract(spaces, environment="tactical-v2")

    cells = 13 * 9
    assert contract.action_size == 1 + (2 * 7 + 5) * cells
    assert contract.observation_size == (2 * 5 + 1) * cells + 5
    assert contract.semantics["starting_unit_count"] == 7
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```powershell
.\python\winenv\Scripts\python.exe -m pytest -q python/tests/test_scenarios.py python/tests/test_gym_client.py python/tests/test_cli.py
```

Expected: FAIL because tactical-v2 is unsupported.

- [ ] **Step 4: Extend scenario validation**

Add `"tactical-v2"` to supported environments and exact key sets. Validate template IDs/names/stats, catalog order, counts, placement policy, and deployment-zone capacity. Keep tactical-v1 and adaptive-v1 key sets unchanged.

Add checked-in entries:

- `tactical-v2-standard`
- `tactical-v2-long-battle`
- `tactical-v2-large-battle`

Each entry contains the five canonical defaults and defaults to three units/slots. Retain existing tactical-v1 entries for legacy selection.

- [ ] **Step 5: Extend CLI and run contracts**

Change new-run defaults from tactical-v1 to tactical-v2 in argument parsing and `RunConfig`, while preserving explicit `--environment tactical-v1`. Resume always reads the source run's recorded environment and scenario.

Every error listing supported environments must name all three:

```text
tactical-v1, tactical-v2, or adaptive-v1
```

- [ ] **Step 6: Validate tactical-v2 spaces**

Add `_validate_tactical_v2` to `python/hexwars_gym/env.py`. Require:

- `environment_kind` in `{"tactical", "duel"}`;
- `tactical_v2` semantics;
- template list matching `contract_roster`;
- action-region offsets matching `1 + (2 * slots + templates) * cells`;
- observation channels matching two planes per ordered template plus elevation;
- five globals;
- matching scenario ID/schema fields and lowercase hashes.

Set `expected_kind = "tactical"` for tactical-v1 and tactical-v2.

- [ ] **Step 7: Update every Python environment consumer**

Add tactical-v2 to the explicit supported-environment/version sets and exhaustive
kind mappings in:

- `python/duel.py`
- `python/policy_server.py`
- `python/selfplay_env.py`
- `python/ml_lab/benchmark.py`
- `python/ml_lab/controllers.py`
- `python/ml_lab/doctor.py`
- `python/ml_lab/evaluation.py`

For tactical-v2, use `"tactical"` for single-agent/doctor/benchmark handshakes and
`"duel"` for self-play/duel/evaluation handshakes. Keep adaptive mappings unchanged.
Controller compatibility remains exact on contract version and encoding hash; adding
tactical-v2 support must not make tactical-v1 checkpoints cross-compatible.

Add or update focused tests in `test_controllers.py`, `test_duel.py`,
`test_evaluation.py`, and `test_policy_server.py` so each consumer accepts
tactical-v2, sends `--environment tactical-v2`, requires the correct kind, and still
rejects unknown versions.

- [ ] **Step 8: Audit hard-coded environment branches**

Run:

```powershell
rg -n "tactical-v1|adaptive-v1|SUPPORTED_ENVIRONMENTS|SUPPORTED_ENCODING_VERSIONS" python --glob "*.py" --glob "!**/__pycache__/**"
```

Classify every result. Update code that enumerates supported environments or maps an
environment to a contract kind. Leave test fixtures and intentionally version-specific
adaptive behavior unchanged. This audit is required even if a file is not listed
above; add any newly discovered consumer and its focused test to this task before
committing.

- [ ] **Step 9: Run focused Python and live process tests**

Build GymServer first:

```powershell
dotnet build .\engine\HexWars.GymServer\HexWars.GymServer.csproj -c Release
.\python\winenv\Scripts\python.exe -m pytest -q python/tests/test_scenarios.py python/tests/test_gym_client.py python/tests/test_cli.py python/tests/test_run_contract.py python/tests/test_controllers.py python/tests/test_duel.py python/tests/test_evaluation.py python/tests/test_policy_server.py
```

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add python/config/training-game-templates.json python/ml_lab/scenarios.py python/ml_lab/cli.py python/ml_lab/contracts.py python/ml_lab/benchmark.py python/ml_lab/controllers.py python/ml_lab/doctor.py python/ml_lab/evaluation.py python/hexwars_gym/env.py python/selfplay_env.py python/duel.py python/policy_server.py python/tests/test_scenarios.py python/tests/test_cli.py python/tests/test_gym_client.py python/tests/test_run_contract.py python/tests/test_controllers.py python/tests/test_duel.py python/tests/test_evaluation.py python/tests/test_policy_server.py
git commit -m "feat(ml): train tactical v2 scenarios"
```

---

### Task 6: Unity Scenario DTOs and Saved-Roster Snapshots

**Files:**
- Create: `Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs`
- Create: `Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs.meta`
- Create: `Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs`
- Create: `Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs.meta`
- Modify: `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`

**Interfaces:**
- Produces: `MlTacticalRosterSource.AvailablePlayers`
- Produces: `MlTacticalRosterSource.Snapshot(int localPlayer): IReadOnlyList<MlTrainingUnitTemplate>`
- Produces: Unity tactical-v2 DTO/wire round trip.
- Consumes later: ML Lab roster selector and engine preflight conversion.

- [ ] **Step 1: Write failing roster-source tests**

```csharp
[Test]
public void Snapshot_IncludesDefaultsAndSelectedPlayersSavedTemplates()
{
    SessionBarracksCache.ResetForTests();
    SessionBarracksCache.ForLocalPlayer(1).Add(
        new UnitTemplate("Custom Alpha", new UnitStats(4, 3, 1, 3, 2, 2, 1, 4, 1)));

    IReadOnlyList<MlTrainingUnitTemplate> snapshot =
        MlTacticalRosterSource.Snapshot(1);

    Assert.That(snapshot.Select(item => item.Name),
        Does.StartWith(BarracksCatalog.DefaultTemplates.Select(item => item.Name)));
    Assert.That(snapshot.Select(item => item.Name), Does.Contain("Custom Alpha"));
    Assert.That(MlTacticalRosterSource.Snapshot(0).Select(item => item.Name),
        Does.Not.Contain("Custom Alpha"));
}
```

- [ ] **Step 2: Write failing scenario round-trip tests**

```csharp
[Test]
public void TacticalV2Scenario_RoundTripsTemplateIdentityAndCount()
{
    MlTrainingScenario scenario = Load("tactical-v2-standard");
    scenario.TacticalV2.StartingUnitCount = 7;
    scenario.TacticalV2.MaxControllableUnits = 7;
    scenario.TacticalV2.Templates =
        MlTacticalRosterSource.Snapshot(0).ToList();

    string json = MlTrainingScenarioFile.Serialize(scenario);
    MlTrainingScenario restored = WriteAndLoad(json);

    Assert.That(restored.TacticalV2.StartingUnitCount, Is.EqualTo(7));
    Assert.That(restored.TacticalV2.Templates.Select(item => item.Id),
        Is.EqualTo(scenario.TacticalV2.Templates.Select(item => item.Id)));
}
```

- [ ] **Step 3: Run Unity tests and verify they fail**

Run the Unity EditMode filter:

```text
HexWars.Presentation.Tests.MlTacticalRosterSourceTests
HexWars.Presentation.Tests.MlTrainingScenarioTests
```

Expected: FAIL because tactical-v2 Unity DTOs and roster source do not exist.

- [ ] **Step 4: Implement the roster-source boundary**

Create:

```csharp
public static class MlTacticalRosterSource
{
    public static IReadOnlyList<int> AvailablePlayers { get; }
    public static IReadOnlyList<MlTrainingUnitTemplate> Snapshot(int localPlayer);
}
```

`Snapshot` must:

1. validate local-player index;
2. begin with `BarracksCatalog.DefaultTemplates`;
3. append selected-cache entries not exactly equal to a canonical default;
4. normalize the combined catalog;
5. derive stable IDs with `TacticalV2TemplateIds.From`;
6. return a detached immutable snapshot.

- [ ] **Step 5: Add Unity tactical-v2 DTOs and strict JSON**

Add:

```csharp
public sealed class MlTrainingTacticalV2
{
    public int StartingUnitCount { get; set; }
    public int MaxControllableUnits { get; set; }
    public string PlacementPolicy { get; set; }
    public List<MlTrainingUnitTemplate> Templates { get; set; }
}

public sealed class MlTrainingUnitTemplate
{
    public string Id { get; set; }
    public string Name { get; set; }
    public MlTrainingUnitStats Stats { get; set; }
}

public sealed class MlTrainingUnitStats
{
    public int Health { get; set; }
    public int Damage { get; set; }
    public int Defense { get; set; }
    public int Movement { get; set; }
    public int VerticalMovement { get; set; }
    public int Range { get; set; }
    public int RangeArc { get; set; }
    public int Vision { get; set; }
    public int VisionArc { get; set; }
}
```

Extend clone, validation, library filtering, exact-key validation, wire serialization, and `MlTrainingScenarioPreflight.ToEngine`. Do not add tactical-v2 fields to tactical-v1 or adaptive JSON.

- [ ] **Step 6: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 7: Run focused Unity tests**

Run the same two EditMode filters.

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs Assets/HexWars/Editor/MlLab/MlTacticalRosterSource.cs.meta Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs Assets/HexWars/Tests/Editor/MlTacticalRosterSourceTests.cs.meta Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs
git commit -m "feat(ml): snapshot saved tactical rosters"
```

---

### Task 7: ML Lab Controls, Defaults, Preflight, and Arena

**Files:**
- Modify: `Assets/HexWars/Presentation/MlEnvironmentContract.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelEnvironment.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`

**Interfaces:**
- Consumes: Unity tactical-v2 DTOs, roster source, engine preflight, and duel environment.
- Produces: visible tactical-v2 starting-count and roster-source controls.
- Produces: `MlTrainingScenarioSession.RefreshTacticalRoster(int localPlayer)`.
- Produces: tactical-v2 as the new ML Lab default.
- Produces: Arena contract validation and playback for tactical-v2.

- [ ] **Step 1: Write failing environment/default tests**

```csharp
[Test]
public void NewConfig_DefaultsToTacticalV2()
{
    Assert.That(new MlLabConfig().Environment,
        Is.EqualTo(MlEnvironmentContract.TacticalV2));
    Assert.That(MlEnvironmentContracts.CliValue(MlEnvironmentContract.TacticalV2),
        Is.EqualTo("tactical-v2"));
}
```

- [ ] **Step 2: Write failing form-state tests**

```csharp
[Test]
public void TacticalV2RosterRefresh_UsesSelectedPlayerAndPreservesCount()
{
    SessionBarracksCache.ResetForTests();
    SessionBarracksCache.ForLocalPlayer(1).Add(Custom("Player Two Custom"));
    MlTrainingScenarioSession session =
        MlTrainingScenarioSession.Load(BuiltInLibraryPath);
    session.SelectEnvironment(MlEnvironmentContract.TacticalV2);
    session.WorkingCopy.TacticalV2.StartingUnitCount = 8;

    session.RefreshTacticalRoster(1);

    Assert.That(session.WorkingCopy.TacticalV2.StartingUnitCount, Is.EqualTo(8));
    Assert.That(session.WorkingCopy.TacticalV2.Templates.Select(item => item.Name),
        Does.Contain("Player Two Custom"));
}
```

- [ ] **Step 3: Write failing Arena adapter tests**

```csharp
[Test]
public void Factory_CreatesTacticalV2DuelWithMatchingIdentity()
{
    TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
    IModelDuelEnvironment duel = ModelDuelEnvironmentFactory.Create(scenario);

    Assert.That(duel.Environment, Is.EqualTo(MlEnvironmentContract.TacticalV2));
    Assert.That(duel.Contract.Version, Is.EqualTo("tactical-v2"));
    Assert.That(ModelDuelEnvironmentFactory.ContractIdentity(scenario).EncodingHash,
        Is.EqualTo(duel.Contract.EncodingHash));
}
```

- [ ] **Step 4: Run focused Unity tests and verify they fail**

Run these EditMode filters:

```text
HexWars.Presentation.Tests.MlLabConfigTests
HexWars.Presentation.Tests.MlLabWindowStateTests
HexWars.Presentation.Tests.ModelDuelConfigurationTests
```

Expected: FAIL because `TacticalV2` and roster-refresh APIs do not exist.

- [ ] **Step 5: Add environment selection and defaults**

Change:

```csharp
public enum MlEnvironmentContract
{
    TacticalV1,
    TacticalV2,
    AdaptiveV1,
}
```

Use an exhaustive switch in `CliValue`; never fall through unknown enum values to tactical-v1. Default `MlLabConfig`, `ModelDuelConfiguration`, and new training sessions to tactical-v2. Keep loaded tactical-v1 run metadata mapped to `TacticalV1`.

- [ ] **Step 6: Add tactical-v2 scenario controls**

In `DrawScenarioFields`, render a `Tactical setup` box for tactical-v2:

```text
Roster source: Local player 1 | Local player 2
Starting unit count: [1..12]
Available templates: N
Brute, Striker, Sniper, Artillery, Scout, ...
[Refresh saved roster]
```

Store the selected source as `[SerializeField] int _tacticalRosterPlayer`. Refresh calls:

```csharp
_scenarioSession.RefreshTacticalRoster(_tacticalRosterPlayer);
```

The working scenario, not live cache state, drives preflight and launch. Tactical-v1 shows a read-only legacy notice. Adaptive retains its existing deployment fields.

- [ ] **Step 7: Add preflight and run-summary semantics**

Preflight text for tactical-v2 must include:

```text
Starting units 8 · controllable slots 8 · templates 7
Roster source snapshotted · automatic symmetric placement
```

Run summaries read these values from run metadata/scenario, not current session cache.

- [ ] **Step 8: Add tactical-v2 Arena adapter**

Add `TacticalV2ModelDuelEnvironment : IModelDuelEnvironment` backed by `TacticalV2DuelEnv`. Treat tactical-v2 as immediately renderable like tactical-v1. Extend `ResolveScenario`, `ApplyPresentationGame`, and environment mapping with exhaustive branches.

- [ ] **Step 9: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 10: Run focused Unity tests**

Run the same three EditMode filters plus:

```text
HexWars.Presentation.Tests.MlTrainingScenarioTests
HexWars.Presentation.Tests.MlTacticalRosterSourceTests
```

Expected: PASS.

- [ ] **Step 11: Manually verify ML Lab state**

In Unity:

1. Open `HexWars > ML Lab`.
2. Select Train and tactical-v2.
3. Confirm `Starting unit count` is visible without selecting adaptive-v1.
4. Add a custom template to local player 2's session roster.
5. Select local player 2 and refresh.
6. Confirm defaults plus the custom template appear.
7. Set count to 8 and confirm preflight dimensions change.
8. Switch to adaptive-v1 and confirm its 1–24 control remains.
9. Switch to tactical-v1 and confirm it is labeled legacy/fixed.

- [ ] **Step 12: Commit**

```powershell
git add Assets/HexWars/Presentation/MlEnvironmentContract.cs Assets/HexWars/Presentation/ModelDuelEnvironment.cs Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Editor/MlLab/MlLabConfig.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/MlLabConfigTests.cs Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs
git commit -m "feat(ml): configure tactical v2 armies"
```

---

### Task 8: Documentation and End-to-End Verification

**Files:**
- Modify: `python/README.md`
- Modify: `docs/ml/architecture.md`
- Modify: `docs/ml/experiment-guide.md`
- Modify if contract troubleshooting changes: `docs/ml/troubleshooting.md`

**Interfaces:**
- Consumes: completed tactical-v2 implementation.
- Produces: user-facing training and compatibility guidance.

- [ ] **Step 1: Update documentation**

Document:

- tactical-v2 is the default for new tactical experiments;
- tactical-v1 is legacy and checkpoint-compatible only with tactical-v1;
- where to choose the local-player roster source and 1–12 unit count;
- defaults plus saved templates are snapshotted into each new run;
- random sampling is with replacement, seeded, and symmetric;
- roster/count changes create a new encoding identity;
- adaptive-v1 still uses learned hidden deployment and a 1–24 count;
- resumes and Arena use the run-local scenario, not the live roster cache.

- [ ] **Step 2: Check Unity compilation**

Run Coplay `check_compile_errors`.

Expected: no compile errors.

- [ ] **Step 3: Run the complete engine suite**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
```

Expected: PASS.

- [ ] **Step 4: Run the complete Python ML suite**

Run:

```powershell
.\python\winenv\Scripts\python.exe -m pytest -q python/tests
```

Expected: PASS.

- [ ] **Step 5: Run Unity EditMode tests**

Run all `HexWars.Presentation.Tests` EditMode tests.

Expected: PASS.

- [ ] **Step 6: Run determinism-sensitive PlayMode tests**

Run the project's PlayMode/determinism filters after the tactical reset and slot-allocation changes.

Expected: PASS with no unexplained baseline divergence. If a divergence occurs, root-cause symmetry, placement order, and seed consumption before considering a re-baseline.

- [ ] **Step 7: Run a live tactical-v2 smoke experiment**

From ML Lab:

1. choose local player 2's roster containing at least one custom template;
2. set tactical-v2 starting units to 8;
3. use a new run name;
4. run Doctor;
5. start 1 worker for 1,000 timesteps with a 500-step checkpoint;
6. inspect `run.json` and `scenario.json`;
7. verify exact roster IDs/names/stats and count 8 are recorded;
8. verify the custom template appears in sampled episodes across a representative seed set;
9. open the completed checkpoint in Arena;
10. confirm contract validation succeeds and both sides begin with eight units.

- [ ] **Step 8: Verify legacy and adaptive paths**

Run:

```powershell
.\python\winenv\Scripts\python.exe python\hexwars_ml.py doctor --environment tactical-v1
.\python\winenv\Scripts\python.exe python\hexwars_ml.py doctor --environment adaptive-v1
```

Expected: both succeed. Load one existing tactical-v1 checkpoint in Arena and verify its encoding hash remains accepted. Confirm adaptive-v1 still exposes and honors its starting-unit count.

- [ ] **Step 9: Review repository diff**

Run:

```powershell
git status --short
git diff --check
git log --oneline -8
```

Expected: only intentional source, test, template, and documentation changes; no generated `__pycache__`, run output, Unity Library files, scene changes, or unrelated user files.

- [ ] **Step 10: Commit documentation**

```powershell
git add python/README.md docs/ml/architecture.md docs/ml/experiment-guide.md docs/ml/troubleshooting.md
git commit -m "docs(ml): explain tactical v2 rosters"
```
