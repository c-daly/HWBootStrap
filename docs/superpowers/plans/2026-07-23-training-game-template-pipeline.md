# Training Game Template Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Unity and headless CLI users select or edit a versioned training-game template, validate it through all three stacks, and preserve the exact resolved scenario with every run.

**Architecture:** A shared JSON schema is authored in `python/config/training-game-templates.json`. Python resolves a template or explicit scenario into canonical JSON, probes the .NET GymServer contract, creates the immutable run snapshot, and then starts every worker using the run-local `scenario.json`. Unity owns a session working copy and editor controls, while the engine remains the authoritative constructor and validator of `EnvConfig` and `AdaptiveEnvConfig`.

**Tech Stack:** Unity 6 Editor IMGUI and `JsonUtility`; C#/.NET Standard 2.1 engine; .NET 8 GymServer with `System.Text.Json`; Python 3.14 dataclasses/JSON; Gymnasium; Stable-Baselines3; NUnit; pytest.

## Global Constraints

- The template-library schema version is exactly `1`.
- The repository template library is `python/config/training-game-templates.json`.
- Every new run contains an immutable canonical `scenario.json`.
- Adaptive v1 remains exactly 24 controllable slots, nine template slots, six fixed templates, and three custom templates.
- UI value `actions_per_turn: 0` means `AllUnitsPolicy`; positive values mean `KActionsPolicy`.
- Large valid boards produce warnings and resolved dimensions; they are not rejected by an arbitrary size cap.
- Exact resume requires the source run's scenario and complete contract.
- Cross-scenario fine-tuning is not implemented.
- Existing runs without `scenario.json` resolve visibly as `legacy-default` and are never rewritten.
- Optional trackers remain non-fatal and local files remain authoritative.
- Do not add new runtime dependencies to `HexWars.Engine`.

---

## File and Interface Map

### New files

- `engine/HexWars.Engine/Rl/TrainingScenario.cs` — engine-owned scenario DTO, validation, and builders for both RL environment configs.
- `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs` — authoritative construction, validation, contract, and geometry tests.
- `engine/HexWars.GymServer/ScenarioJson.cs` — strict JSON loading for `--scenario-file`.
- `python/config/training-game-templates.json` — checked-in schema-v1 authoring library.
- `python/ml_lab/scenarios.py` — Python schema validation, canonicalization, library lookup, snapshotting, and handshake comparison.
- `python/ml_lab/io.py` — add an atomic canonical-text writer beside the existing atomic JSON writer.
- `python/tests/test_scenarios.py` — Python scenario unit tests.
- `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs` — Unity DTO/store/session validation and conversion helpers.
- `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs` — Unity load/edit/save/CLI/preflight tests.

### Modified files

- `engine/HexWars.Engine/Rl/MlContract.cs` — make full-contract versus encoding compatibility behavior explicit and testable.
- `engine/HexWars.GymServer/Program.cs` — accept one scenario file and construct every environment from it.
- `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs` — process-level GymServer scenario handshake tests.
- `python/ml_lab/cli.py` — mutually exclusive `--scenario-file` / `--template` arguments and default resolution.
- `python/ml_lab/contracts.py` — scenario and resolved-opponent provenance in manifests and immutable run creation.
- `python/ml_lab/controllers.py` — export exact fixed/live opponent snapshots for run provenance.
- `python/ml_lab/training.py` — probe, snapshot, actual-worker construction sequence.
- `python/ml_lab/envs.py` — propagate the run-local scenario path to all workers.
- `python/hexwars_gym/env.py` — add `scenario_path` to GymServer command construction.
- `python/selfplay_env.py` — add the same scenario argument to duel workers.
- `python/tests/test_cli.py`, `python/tests/test_run_contract.py`, `python/tests/test_training.py`, `python/tests/test_gym_client.py`, `python/tests/test_duel.py`, `python/tests/test_controllers.py` — focused regression coverage.
- `Assets/HexWars/Editor/MlLab/MlLabConfig.cs` — scenario-aware validation and train arguments.
- `Assets/HexWars/Editor/MlLab/MlLabWindow.cs` — template selector, inline editor, save/reload, and preflight display.
- `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs` — CLI and validation regression tests.
- `python/README.md`, `docs/ml/architecture.md`, `docs/ml/experiment-guide.md`, `docs/ml/troubleshooting.md` — intern workflow and operational documentation.

## Exact Schema-v1 Field Set

Every resolved scenario has this shape:

```json
{
  "schema_version": 1,
  "id": "adaptive-standard",
  "name": "Standard",
  "environment": "adaptive-v1",
  "board": {
    "width": 13,
    "height": 9,
    "max_elevation": 4,
    "zone_depth": 3,
    "flat_chance": 0.6,
    "plains_weight": 70,
    "forest_weight": 15,
    "rough_weight": 10,
    "water_weight": 5
  },
  "rules": {
    "actions_per_turn": 0,
    "round_cap": 100,
    "starting_points": 12,
    "fog_of_war": true,
    "biomes_enabled": false,
    "bounty_rate": 0.5,
    "deploy_cost_multiplier": 1.0,
    "generator_cost": 2,
    "generator_output": 1,
    "generator_health": 3
  },
  "episode": {
    "max_steps": 900
  },
  "reward": {
    "intermediate_decision_penalty": 0.001,
    "deployment_completion_bonus": 0.0
  },
  "adaptive": {
    "starting_unit_count": 6,
    "starting_army_budget": 132,
    "max_design_point_cost": 24
  }
}
```

Tactical templates omit `adaptive` and instead use:

```json
{
  "shape_scale": 0.01,
  "step_penalty": 0.005,
  "closing_weight": 0.02,
  "draw_credit_weight": 0.25,
  "points_weight": 0.5
}
```

Code-grounded refinement from the design review: the five tactical shaping values apply only to `tactical-v1`. The current adaptive environment does not consume them; it consumes `intermediate_decision_penalty` and `deployment_completion_bonus`. ML Lab therefore shows the reward controls actually used by the selected environment rather than persisting inert values.

The library contains these six exact presets:

| ID | Environment | Board | Round cap | Max steps | Adaptive setup |
|---|---|---:|---:|---:|---|
| `tactical-standard` | `tactical-v1` | 13×9, zone 3 | 100 | 600 | omitted |
| `tactical-long-battle` | `tactical-v1` | 13×9, zone 3 | 200 | 1200 | omitted |
| `tactical-large-battle` | `tactical-v1` | 24×16, zone 4 | 150 | 1200 | omitted |
| `adaptive-standard` | `adaptive-v1` | 13×9, zone 3 | 100 | 900 | 6 units, 132 budget, 24 max design |
| `adaptive-long-battle` | `adaptive-v1` | 13×9, zone 3 | 200 | 1800 | 6 units, 132 budget, 24 max design |
| `adaptive-large-battle` | `adaptive-v1` | 24×16, zone 4 | 150 | 1800 | 6 units, 132 budget, 24 max design |

All other board, rules, and reward values in Long Battle and Large Battle equal their environment's Standard template.

### Task 1: Authoritative engine scenario construction

**Files:**
- Create: `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- Create: `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs`
- Modify: `engine/HexWars.Engine/Rl/MlContract.cs`

**Interfaces:**
- Produces: `TrainingScenario.Validate(): IReadOnlyList<string>`
- Produces: `TrainingScenario.BuildTactical(): EnvConfig`
- Produces: `TrainingScenario.BuildAdaptive(): AdaptiveEnvConfig`
- Produces: `TrainingScenario.CreateStandard(string environment, string id = "legacy-default"): TrainingScenario`
- Consumes later: GymServer and Unity pass a populated `TrainingScenario` to these methods.

- [ ] **Step 1: Write failing construction and validation tests**

Add tests that instantiate the DTO directly so engine behavior does not depend on either JSON parser:

```csharp
[Test]
public void TacticalScenario_BuildsEveryConfigurableValue()
{
    var scenario = TrainingScenario.CreateStandard("tactical-v1");
    scenario.Board.Width = 24;
    scenario.Board.Height = 16;
    scenario.Board.ZoneDepth = 4;
    scenario.Rules.ActionsPerTurn = 7;
    scenario.Rules.RoundCap = 150;
    scenario.Rules.FogOfWar = true;
    scenario.Episode.MaxSteps = 1200;
    scenario.TacticalReward.ShapeScale = 0.02f;

    EnvConfig config = scenario.BuildTactical();

    Assert.That(config.BoardGen.Width, Is.EqualTo(24));
    Assert.That(config.BoardGen.Height, Is.EqualTo(16));
    Assert.That(config.BoardGen.ZoneDepth, Is.EqualTo(4));
    Assert.That(config.Game.TurnPolicy.ActionsPerTurn, Is.EqualTo(7));
    Assert.That(config.Game.RoundCap, Is.EqualTo(150));
    Assert.That(config.Game.FogOfWar, Is.True);
    Assert.That(config.MaxSteps, Is.EqualTo(1200));
    Assert.That(config.ShapeScale, Is.EqualTo(0.02f));
}

[Test]
public void AdaptiveScenario_PreservesPinnedArchitecture()
{
    var scenario = TrainingScenario.CreateStandard("adaptive-v1");
    scenario.Adaptive.StartingUnitCount = 7;
    scenario.Adaptive.StartingArmyBudget = 160;

    AdaptiveEnvConfig config = scenario.BuildAdaptive();

    Assert.That(config.MaxControllableUnits, Is.EqualTo(24));
    Assert.That(config.Templates, Has.Count.EqualTo(9));
    Assert.That(config.FixedTemplateCount, Is.EqualTo(6));
    Assert.That(config.CustomTemplateCount, Is.EqualTo(3));
    Assert.That(config.StartingUnitCount, Is.EqualTo(7));
    Assert.That(config.StartingArmyBudget, Is.EqualTo(160));
}

[TestCase(0, 9, 3, "width")]
[TestCase(13, 0, 3, "height")]
[TestCase(7, 9, 4, "deployment zones overlap")]
public void Scenario_RejectsImpossibleGeometry(int width, int height, int zoneDepth, string message)
{
    var scenario = TrainingScenario.CreateStandard("tactical-v1");
    scenario.Board.Width = width;
    scenario.Board.Height = height;
    scenario.Board.ZoneDepth = zoneDepth;

    Assert.That(scenario.Validate(), Has.Some.Contains(message));
}

[TestCase(6, 3)]
[TestCase(7, 3)]
public void Scenario_AcceptsDisjointDeploymentZonesThatPartitionTheWidth(
    int width, int zoneDepth)
{
    var scenario = TrainingScenario.CreateStandard("tactical-v1");
    scenario.Board.Width = width;
    scenario.Board.ZoneDepth = zoneDepth;

    Assert.That(scenario.Validate(), Has.None.Contains("deployment zones overlap"));
}
```

Also test negative terrain weights, a zero terrain-weight sum, flat chance outside `[0,1]`, negative actions per turn, non-positive round/max steps, insufficient adaptive deployment cells, insufficient adaptive budget, and tactical/adaptive section mismatch.

- [ ] **Step 2: Run the focused engine tests and confirm RED**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter TrainingScenarioTests
```

Expected: compile failure because `TrainingScenario` does not exist.

- [ ] **Step 3: Implement the DTO, validation, and builders**

Use public fields so Unity `JsonUtility` can populate the same types. Keep JSON parsing outside the engine. Define `TrainingScenario`, `TrainingBoardConfig`, `TrainingRuleConfig`, `TrainingEpisodeConfig`, `TacticalRewardConfig`, `AdaptiveRewardConfig`, and `TrainingAdaptiveConfig`.

`TrainingScenario` exposes these exact methods:

```csharp
public IReadOnlyList<string> Validate()
public EnvConfig BuildTactical()
public AdaptiveEnvConfig BuildAdaptive()
public static TrainingScenario CreateStandard(
    string environment, string id = "legacy-default")
```

`CreateStandard` uses the values in the Exact Schema-v1 Field Set above. `Validate` emits one message per invalid field and enforces every condition enumerated in Step 1. Both builders throw `ArgumentException(string.Join("; ", errors))` when validation returns any error.

Construct the turn policy exactly as:

```csharp
ITurnPolicy turnPolicy = Rules.ActionsPerTurn == 0
    ? (ITurnPolicy)new AllUnitsPolicy()
    : new KActionsPolicy(Rules.ActionsPerTurn);
```

Construct `GameConfig` with an explicit standard terrain dictionary and named arguments for every exposed rule. Do not mutate `GameConfig.Default()` because it is immutable.

- [ ] **Step 4: Add contract-identity regression tests**

```csharp
[Test]
public void RewardOrHorizonChange_ChangesContractButNotEncodingIdentity()
{
    var first = TrainingScenario.CreateStandard("tactical-v1").BuildTactical();
    var changed = TrainingScenario.CreateStandard("tactical-v1").BuildTactical();
    changed.ShapeScale = 0.2f;
    changed.MaxSteps = 1200;

    MlContract a = MlContract.Create(first);
    MlContract b = MlContract.Create(changed);

    Assert.That(b.ContractHash, Is.Not.EqualTo(a.ContractHash));
    Assert.That(b.EncodingHash, Is.EqualTo(a.EncodingHash));
}

[Test]
public void GeometryChange_ChangesEncodingIdentityAndDimensions()
{
    var first = TrainingScenario.CreateStandard("adaptive-v1");
    var changed = TrainingScenario.CreateStandard("adaptive-v1");
    changed.Board.Width = 24;
    changed.Board.Height = 16;

    MlContract a = MlContract.CreateAdaptive(first.BuildAdaptive());
    MlContract b = MlContract.CreateAdaptive(changed.BuildAdaptive());

    Assert.That(b.EncodingHash, Is.Not.EqualTo(a.EncodingHash));
    Assert.That(b.ObservationSize, Is.Not.EqualTo(a.ObservationSize));
    Assert.That(b.ActionSize, Is.Not.EqualTo(a.ActionSize));
}
```

Keep the existing default encoding hashes stable. `NormalizeEncodingValue` already excludes environment kind, effective horizon, maximum steps, and adaptive reward knobs; the tests lock that behavior down. Do not broaden compatibility to economy, fog, biome, or procedural-distribution changes in this release. The immutable scenario and full contract preserve the evidence needed to define a wider transfer-compatible identity when fine-tuning is implemented.

- [ ] **Step 5: Run engine tests and commit**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
```

Expected: all engine tests pass.

Commit:

```powershell
git add engine/HexWars.Engine/Rl/TrainingScenario.cs engine/HexWars.Engine/Rl/MlContract.cs engine/HexWars.Engine.Tests/TrainingScenarioTests.cs
git commit -m "feat(ml): construct validated training scenarios"
```

### Task 2: GymServer scenario-file boundary

**Files:**
- Create: `engine/HexWars.GymServer/ScenarioJson.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`
- Modify: `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs`

**Interfaces:**
- Consumes: `TrainingScenario.BuildTactical()` / `BuildAdaptive()`
- Produces CLI: `HexWars.GymServer.dll --environment ENV --scenario-file PATH`
- Produces handshake fields: `scenario_id`, `scenario_schema_version`, existing contract fields.

- [ ] **Step 1: Add failing process tests**

Use the existing `ServerProcess` fixture:

```csharp
[Test]
public void GymServer_LoadsScenarioAndReportsResolvedContract()
{
    string scenario = WriteScenario(environment: "adaptive-v1", width: 24, height: 16, maxSteps: 1800);
    using var server = new ServerProcess(
        "--environment", "adaptive-v1", "--scenario-file", scenario);

    using JsonDocument spaces = server.Send(new { cmd = "spaces" });

    Assert.That(spaces.RootElement.GetProperty("scenario_id").GetString(), Is.EqualTo("test-large"));
    Assert.That(spaces.RootElement.GetProperty("scenario_schema_version").GetInt32(), Is.EqualTo(1));
    Assert.That(spaces.RootElement.GetProperty("board_w").GetInt32(), Is.EqualTo(24));
    Assert.That(spaces.RootElement.GetProperty("board_h").GetInt32(), Is.EqualTo(16));
    Assert.That(spaces.RootElement.GetProperty("max_steps").GetInt32(), Is.EqualTo(1800));
}
```

Add cases for malformed JSON, schema version `2`, environment mismatch, missing required environment-specific reward section, and impossible deployment.

- [ ] **Step 2: Run the process tests and confirm RED**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "AdaptiveDuelEnvTests"
```

Expected: new scenario assertions fail because GymServer ignores `--scenario-file`.

- [ ] **Step 3: Implement strict parsing and one-time construction**

`ScenarioJson.Load(path)` must deserialize snake_case JSON into private wire DTOs, require every schema-v1 field, map to `TrainingScenario`, call `Validate`, and throw one `InvalidDataException` containing joined field errors. In `Program.cs`:

```csharp
string? scenarioFile = null;
// parse --scenario-file exactly once; reject missing value
TrainingScenario scenario = scenarioFile == null
    ? TrainingScenario.CreateStandard(environment)
    : ScenarioJson.Load(scenarioFile);
if (!string.Equals(scenario.Environment, environment, StringComparison.Ordinal))
    throw new InvalidDataException("scenario environment does not match --environment");

EnvConfig? tacticalConfig = environment == "tactical-v1" ? scenario.BuildTactical() : null;
AdaptiveEnvConfig? adaptiveConfig = environment == "adaptive-v1" ? scenario.BuildAdaptive() : null;
```

Pass those same config objects to tactical, adaptive, and duel constructors. Never construct a default duel later in the command loop.

- [ ] **Step 4: Re-run tests and commit**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
```

Expected: all engine and GymServer process tests pass.

Commit:

```powershell
git add engine/HexWars.GymServer/ScenarioJson.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs
git commit -m "feat(ml): load scenarios in GymServer"
```

### Task 3: Python template resolution and canonical scenarios

**Files:**
- Create: `python/config/training-game-templates.json`
- Create: `python/ml_lab/scenarios.py`
- Create: `python/tests/test_scenarios.py`
- Modify: `python/ml_lab/io.py`
- Modify: `python/ml_lab/cli.py`
- Modify: `python/tests/test_cli.py`
- Modify: `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs`

**Interfaces:**
- Produces: `ResolvedScenario`
- Produces: `load_template_library(path: Path) -> list[ResolvedScenario]`
- Produces: `resolve_scenario(*, environment: str, scenario_file: Path | None, template_id: str | None, library_path: Path = DEFAULT_TEMPLATE_LIBRARY) -> ResolvedScenario`
- Produces: `validate_handshake(scenario: ResolvedScenario, spaces_info: Mapping[str, Any]) -> None`
- Produces: `ResolvedScenario.write(path: Path) -> None`
- Produces CLI: mutually exclusive `--scenario-file PATH` and `--template ID`.

- [ ] **Step 1: Write failing loader and CLI tests**

```python
def test_builtin_library_has_three_templates_per_environment() -> None:
    templates = load_template_library(DEFAULT_TEMPLATE_LIBRARY)
    assert [item.template_id for item in templates] == [
        "tactical-standard", "tactical-long-battle", "tactical-large-battle",
        "adaptive-standard", "adaptive-long-battle", "adaptive-large-battle",
    ]

def test_explicit_scenario_is_canonical_and_environment_checked(tmp_path: Path) -> None:
    path = write_scenario(tmp_path, environment="adaptive-v1")
    first = resolve_scenario(environment="adaptive-v1", scenario_file=path, template_id=None)
    second = resolve_scenario(environment="adaptive-v1", scenario_file=path, template_id=None)
    assert first.canonical_json == second.canonical_json
    with pytest.raises(ValueError, match="environment"):
        resolve_scenario(environment="tactical-v1", scenario_file=path, template_id=None)

def test_train_cli_rejects_template_and_file_together(parser) -> None:
    with pytest.raises(SystemExit):
        parser.parse_args([
            "train", "--run", "x", "--template", "tactical-standard",
            "--scenario-file", "custom.json",
        ])
```

Add parameterized schema tests for booleans accepted as integers, NaN/infinity, missing sections, wrong reward kind, invalid adaptive values, duplicate IDs, and library/template environment mismatch.

- [ ] **Step 2: Run tests and confirm RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_scenarios.py python/tests/test_cli.py -q
```

Expected: import/argument failures.

- [ ] **Step 3: Implement immutable resolution**

Use:

```python
@dataclass(frozen=True)
class ResolvedScenario:
    schema_version: int
    template_id: str
    name: str
    environment: str
    document: Mapping[str, Any]
    canonical_json: str

    def write(self, path: Path) -> None:
        atomic_write_text(path, self.canonical_json + "\n")
```

Add `atomic_write_text(path: Path, text: str)` to `ml_lab.io` using the same sibling-temp, flush, `os.fsync`, and `os.replace` pattern as `atomic_write_json`. Canonical JSON uses `sort_keys=True`, `separators=(",", ":")`, `allow_nan=False`. Deep-copy and freeze the validated document so later UI/CLI mutation cannot alter the snapshot. Default selection is `<environment-prefix>-standard`.

Create all six exact presets from the table above. Do not add a Small preset.

Add one parameterized .NET process test that reads all six entries from `python/config/training-game-templates.json`, writes each entry as a resolved scenario document, launches its declared environment, and verifies a successful spaces handshake. This prevents the checked-in authoring library from drifting away from the GymServer parser.

- [ ] **Step 4: Wire CLI scenario selection**

Add this argument group to `train`:

```python
scenario = train.add_mutually_exclusive_group()
scenario.add_argument("--scenario-file", type=Path)
scenario.add_argument("--template")
```

Resolve before `_training_config`, pass the `ResolvedScenario` to `run_training`, and ensure resume always loads the source run scenario (or a visible `legacy-default`) instead of honoring a new scenario selection.

- [ ] **Step 5: Run Python tests and commit**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_scenarios.py python/tests/test_cli.py -q
```

Expected: all focused tests pass.

Commit:

```powershell
git add python/config/training-game-templates.json python/ml_lab/scenarios.py python/ml_lab/io.py python/ml_lab/cli.py python/tests/test_scenarios.py python/tests/test_cli.py engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs
git commit -m "feat(ml): resolve versioned game templates"
```

### Task 4: Immutable run snapshots and run-local worker propagation

**Files:**
- Modify: `python/ml_lab/contracts.py`
- Modify: `python/ml_lab/controllers.py`
- Modify: `python/ml_lab/training.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/hexwars_gym/env.py`
- Modify: `python/selfplay_env.py`
- Modify: `python/tests/test_run_contract.py`
- Modify: `python/tests/test_training.py`
- Modify: `python/tests/test_gym_client.py`
- Modify: `python/tests/test_duel.py`
- Modify: `python/tests/test_controllers.py`

**Interfaces:**
- Consumes: `ResolvedScenario`
- Produces: `create_run(runs_root: Path, config: RunConfig, contract: EnvironmentContract, scenario: ResolvedScenario, *, opponent_snapshot: Mapping[str, Any]) -> Path`
- Produces: `TrainingEnvironmentFactory.probe(config, scenario_path) -> tuple[EnvironmentContract, Mapping[str, Any], Mapping[str, Any]]`
- Produces: `TrainingEnvironmentFactory(config, run_dir, scenario_path, opponent_snapshot)`
- Produces: `snapshot_opponents(config.opponent) -> Mapping[str, Any]`
- Produces: `materialized_scenario(scenario: ResolvedScenario, *, parent: Path)` context manager.
- Produces subprocess arg: `--scenario-file <run>/scenario.json`.

- [ ] **Step 1: Add failing provenance and worker tests**

```python
def test_create_run_snapshots_scenario_and_manifest_provenance(
    tmp_path: Path, config: RunConfig, contract: EnvironmentContract, scenario: ResolvedScenario
) -> None:
    opponent_snapshot = {"kind": "scripted", "name": "greedy"}
    run = create_run(
        tmp_path, config, contract, scenario,
        opponent_snapshot=opponent_snapshot,
    )
    assert (run / "scenario.json").read_text(encoding="utf-8") == scenario.canonical_json + "\n"
    manifest = read_json(run / "run.json")
    assert manifest["scenario"] == {
        "path": "scenario.json",
        "template_id": scenario.template_id,
        "schema_version": 1,
    }
    assert manifest["opponent_snapshot"] == opponent_snapshot

def test_every_real_worker_receives_run_local_scenario(tmp_path: Path, config: RunConfig) -> None:
    factory = CapturingTrainingEnvironmentFactory()
    run_training(config, runs_root=tmp_path, scenario=scenario(), environment_factory=factory)
    expected = tmp_path / config.run_name / "scenario.json"
    assert factory.worker_scenario_paths == [expected] * config.workers
    assert all(
        value == factory.probed_opponent_snapshot
        for value in factory.worker_opponent_snapshots
    )
```

Add a test that mutating the library after run creation does not change `scenario.json`, a probe/actual contract mismatch test, and legacy resume resolution.

Add controller-provenance tests asserting that scripted opponents retain their exact name, fixed runs retain the exact resolved checkpoint path/step/algorithm/contract identity, live runs retain their run path and live mode, and every pool entry is snapshotted in input order. Store this as top-level `opponent_snapshot` in `run.json`; keep `config.opponent` as the user's original request.

- [ ] **Step 2: Run focused tests and confirm RED**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests/test_run_contract.py python/tests/test_training.py python/tests/test_gym_client.py python/tests/test_duel.py -q
```

Expected: signature and missing-snapshot failures.

- [ ] **Step 3: Implement probe → create → worker ordering**

Materialize the canonical scenario in a temporary directory under `runs_root` for the probe. The real path must be:

```python
with materialized_scenario(scenario, parent=runs_root) as probe_path:
    probe_contract, probe_spaces, opponent_snapshot = environment_factory.probe(
        config, probe_path
    )
run_dir = create_run(
    runs_root, config, probe_contract, scenario,
    opponent_snapshot=opponent_snapshot,
)
env = environment_factory(
    config, run_dir, run_dir / "scenario.json", opponent_snapshot
)
if env.contract != probe_contract:
    env.close()
    raise ContractMismatch("worker contract changed after scenario snapshot")
```

The probe creates one worker without an episode monitor, reads the handshake, validates scenario values with `validate_handshake`, snapshots its resolved opponent bindings, and closes it. The actual vector workers are created only after the run-local file exists and consume those snapshots instead of resolving a fixed run's newest checkpoint again. A mismatch error names the scenario ID, field path, requested value, and authoritative handshake value.

Add `scenario_path: Path | None = None` to `HexWarsEnv` and `SelfPlayEnv`. Append `["--scenario-file", str(scenario_path)]` only when non-null so legacy callers keep default behavior.

- [ ] **Step 4: Preserve exact resume semantics**

`resume` reads `<source>/scenario.json`. If absent, create an in-memory Standard scenario with ID `legacy-default`, record that label in the new run, and do not write to the source. Exact contract validation remains unchanged.

- [ ] **Step 5: Run all Python tests and commit**

Run:

```powershell
& .\python\winenv\Scripts\python.exe -m pytest python/tests -q
```

Expected: all Python tests pass.

Commit:

```powershell
git add python/ml_lab/contracts.py python/ml_lab/controllers.py python/ml_lab/training.py python/ml_lab/envs.py python/hexwars_gym/env.py python/selfplay_env.py python/tests/test_run_contract.py python/tests/test_training.py python/tests/test_gym_client.py python/tests/test_duel.py python/tests/test_controllers.py
git commit -m "feat(ml): snapshot scenarios for every worker"
```

### Task 5: Unity scenario store and CLI bridge

**Files:**
- Create: `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`
- Create: `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`

**Interfaces:**
- Produces: `MlTrainingScenarioLibrary.Load(path)`
- Produces: `MlTrainingScenarioLibrary.Filter(environment)`
- Produces: `MlTrainingScenarioStore.WriteSessionScenario(projectRoot, scenario) -> string`
- Produces: `MlTrainingScenarioStore.SaveAsTemplate(path, scenario, overwrite)`.
- Changes: `MlLabConfig.BuildTrainArguments(string scenarioPath)`.

- [ ] **Step 1: Write failing Unity EditMode tests**

```csharp
[Test]
public void Library_LoadsAndFiltersBuiltInTemplates()
{
    var library = MlTrainingScenarioLibrary.Load(BuiltInLibraryPath);
    Assert.That(library.Filter(MlEnvironmentContract.AdaptiveV1)
        .Select(item => item.Id), Is.EqualTo(new[] {
            "adaptive-standard", "adaptive-long-battle", "adaptive-large-battle"
        }));
}

[Test]
public void SessionWriter_RoundTripsResolvedScenario()
{
    string path = MlTrainingScenarioStore.WriteSessionScenario(_projectRoot, _scenario);
    Assert.That(path, Does.StartWith(Path.Combine(_projectRoot, "Library")));
    Assert.That(MlTrainingScenarioFile.Load(path).Board.Width, Is.EqualTo(_scenario.Board.Width));
}

[Test]
public void BuildTrainArguments_UsesResolvedScenarioFile()
{
    var config = MlLabConfig.Default();
    string args = config.BuildTrainArguments(@"C:\project\Library\HexWars\MLLab\scenario.json");
    Assert.That(args, Does.Contain("--scenario-file \"C:\\project\\Library\\HexWars\\MLLab\\scenario.json\""));
}
```

Add validation tests for all three boundary classes, duplicate IDs, environment mismatch, and overwrite refusal.

- [ ] **Step 2: Run EditMode tests and confirm RED**

Run the Unity EditMode filter for `MlTrainingScenarioTests` and `MlLabConfigTests`.

Expected: compile failures for missing scenario types and new method signature.

- [ ] **Step 3: Implement DTO/store without dialogs**

Use serializable wire classes with snake_case field names for `JsonUtility`. Convert them to a session model with C#-style properties. Saving must:

1. serialize to a sibling `*.tmp`;
2. parse and validate the temporary file;
3. use `File.Replace(temp, target, null)` when the target exists;
4. use `File.Move(temp, target)` for first creation;
5. retain the session copy if any operation fails.

Expose overwrite as a boolean controlled by the window's inline two-click confirmation; do not open a modal.

- [ ] **Step 4: Check Unity compilation and run tests**

After the C# edits, call Coplay `check_compile_errors`. Fix every compiler error, then run:

```text
HexWars.Presentation.Tests.MlTrainingScenarioTests
HexWars.Presentation.Tests.MlLabConfigTests
```

Expected: all focused EditMode tests pass.

- [ ] **Step 5: Commit**

Include Unity-generated `.meta` files:

```powershell
git add Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs.meta Assets/HexWars/Editor/MlLab/MlLabConfig.cs Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs.meta Assets/HexWars/Tests/Editor/MlLabConfigTests.cs
git commit -m "feat(ml): bridge Unity training scenarios"
```

### Task 6: ML Lab template editor and preflight

**Files:**
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`

**Interfaces:**
- Consumes: Unity scenario library/store from Task 5.
- Produces UI state: selected template ID, session working copy, save name/ID, overwrite-armed flag.
- Produces launch file under `Library/HexWars/MLLab`.

- [ ] **Step 1: Extract and test deterministic UI state transitions**

Add a non-visual `MlTrainingScenarioSession` with tests:

```csharp
[Test]
public void EnvironmentChange_SelectsThatEnvironmentsStandardTemplate()
{
    var session = new MlTrainingScenarioSession(_library);
    session.SelectEnvironment(MlEnvironmentContract.AdaptiveV1);
    Assert.That(session.WorkingCopy.Id, Is.EqualTo("adaptive-standard"));
}

[Test]
public void EditIsSessionOnlyUntilSaveAndReloadDiscardsIt()
{
    var session = new MlTrainingScenarioSession(_library);
    session.WorkingCopy.Board.Width = 64;
    session.Reload();
    Assert.That(session.WorkingCopy.Board.Width, Is.EqualTo(13));
}
```

Also test Save As unique ID, inline overwrite arming, environment-filtered selection, and calculated action/observation dimensions.

- [ ] **Step 2: Run focused tests and confirm RED**

Run the two focused EditMode test classes.

Expected: missing session-state type.

- [ ] **Step 3: Implement the IMGUI controls**

In `DrawTrainingForm`:

- environment first;
- template popup filtered by environment;
- `Advanced game settings` foldout;
- grouped number/toggle fields for Board, Match rules, Episode, and the applicable reward type;
- Adaptive deployment group only for adaptive;
- inline `Template name`, `Template ID`, `Save as template`, `Reload templates`;
- a second inline `Confirm overwrite` click only after a collision;
- field errors in `HelpBox` and launch disabled when invalid.

Blank numeric controls are not possible with `EditorGUILayout.IntField`; retain the last valid numeric value while typing. Do not use `DisplayDialog`.

Show preflight values:

```text
Template · environment
Board W×H · zone depth · actions/turn (Whole team when 0)
Round cap · max steps · fog · opponent · learner seats
Observation N · actions M
Warning: large scenario may reduce headless throughput
```

Calculate dimensions by converting the working copy to `TrainingScenario` and calling the authoritative `MlContract.Create` or `MlContract.CreateAdaptive`; do not duplicate layout formulas in Editor code. Cover the displayed results with tests against engine defaults and the 24×16 presets.

- [ ] **Step 4: Write the scenario before Train and block invalid launch**

`StartTraining` validates the working copy, writes the temporary scenario, calls `BuildTrainArguments(path)`, and leaves `_state` in Failed with the exact path/error if either operation fails. Resume ignores the session editor and shows the source run scenario summary.

If the checked-in library cannot be read, keep the parse exception and exact library path in the session state, disable Train and Start & Watch, and render that error in a help box. Doctor remains available for dependency diagnosis. Do not synthesize compiled fallback templates.

- [ ] **Step 5: Compile, run all Unity EditMode tests, and commit**

Call Coplay `check_compile_errors`, then run all EditMode tests.

Expected: zero compile errors and all tests pass.

Commit:

```powershell
git add Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs
git commit -m "feat(ml): edit game templates in ML Lab"
```

### Task 7: Cross-stack verification and intern documentation

**Files:**
- Modify: `python/README.md`
- Modify: `docs/ml/architecture.md`
- Modify: `docs/ml/experiment-guide.md`
- Modify: `docs/ml/troubleshooting.md`

**Interfaces:**
- Documents both CLI forms and the Unity workflow.
- Establishes `scenario.json` as experiment provenance, not a mutable settings file.

- [ ] **Step 1: Add exact CLI examples**

Document:

```powershell
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train `
  --run adaptive_large_seed31 `
  --environment adaptive-v1 `
  --template adaptive-large-battle `
  --algorithm maskable_ppo `
  --opponent greedy `
  --learner-seat alternating `
  --workers 4 `
  --timesteps 300000
```

And:

```powershell
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train `
  --run custom_scenario_seed32 `
  --environment adaptive-v1 `
  --scenario-file .\experiments\counter-artillery.json `
  --algorithm maskable_ppo `
  --opponent random `
  --learner-seat 1 `
  --workers 2 `
  --timesteps 100000
```

Explain how an intern copies a library entry into an experiment file, changes starting budget and horizon, validates with a 1,000-step smoke run, inspects `run.json` plus `scenario.json`, watches it in Unity, and promotes only reviewed artifacts.

- [ ] **Step 2: Run cross-stack automated verification**

Run:

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
& .\python\winenv\Scripts\python.exe -m pytest python/tests -q
```

Call Coplay `check_compile_errors` and run all Unity EditMode tests.

Expected: all suites pass.

- [ ] **Step 3: Run two real smoke experiments**

Build GymServer Release, then run 1,000 timesteps each:

```powershell
dotnet build .\engine\HexWars.GymServer\HexWars.GymServer.csproj -c Release
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run scenario-smoke-tactical --environment tactical-v1 --template tactical-long-battle --algorithm maskable_ppo --opponent random --timesteps 1000 --checkpoint-every 500 --workers 1 --learner-seat alternating --tracker local
& .\python\winenv\Scripts\python.exe .\python\hexwars_ml.py train --run scenario-smoke-adaptive --environment adaptive-v1 --template adaptive-large-battle --algorithm maskable_ppo --opponent greedy --timesteps 1000 --checkpoint-every 500 --workers 1 --learner-seat alternating --tracker local
```

Expected: both complete, each manifest points to `scenario.json`, and handshake dimensions match its snapshot. Keep smoke runs untracked.

- [ ] **Step 4: Commit documentation**

```powershell
git add python/README.md docs/ml/architecture.md docs/ml/experiment-guide.md docs/ml/troubleshooting.md
git commit -m "docs(ml): explain training game templates"
```

- [ ] **Step 5: Final branch review**

Run:

```powershell
git status --short
git diff --check HEAD~7..HEAD
```

Expected: only the pre-existing `python/hexwars_gym/__pycache__/env.cpython-314.pyc` modification remains outside the commits; no whitespace errors.
