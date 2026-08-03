# Unity Profiled-Run Viewer Compatibility Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make a selected profiled tactical-v2 run watchable from a linked Unity worktree without changing its recorded scenario or interrupting active training.

**Architecture:** Extend Unity's strict training-scenario boundary to represent the engine's existing profiled-start catalog and distribution, then convert those values into the authoritative engine types. Route ReplayViewerMenu's interpreter lookup through the existing worktree-aware MlLabPaths resolver while keeping scripts and run data rooted in the current worktree, and use one exact inverse mapping for all three environment contracts.

**Tech Stack:** Unity 6.5 Editor C#, NUnit EditMode tests, HexWars.Engine tactical-v2 contracts, Windows Python virtual environment.

## Global Constraints

- Do not stop, restart, pause, or modify the active `bc223` trainer.
- Preserve strict JSON validation; unknown properties remain errors.
- Never replace a profiled scenario with a standard scenario or discard profile weights.
- Keep `policy_server.py`, GymServer, run metadata, and scenarios rooted in the current worktree.
- Resolve only the Python executable through the shared main-checkout environment when the local worktree environment is absent.
- After every C# edit, verify Unity compilation; because Coplay is absent in this worktree, use the live Editor log or a batch EditMode run as the compilation authority.
- Do not stage unrelated dirty worktree files.
- Keep the visible Unity Editor open. Run RED/GREEN checks in a disposable
  detached verification worktree so batch Unity never competes for the active
  project's lock; copy only the edited source/test files and the built
  `HexWars.Engine.dll` into that verifier before each run.

---

### Task 1: Profiled tactical-v2 scenario parity

**Files:**
- Create: `Assets/HexWars/Tests/Editor/Fixtures/ProfiledTacticalV2Scenario.json`
- Modify: `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs:90-190,256-262,480-660,1029-1084,1140-1165,1349-1374`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs:534-635`
- Test: `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`

**Interfaces:**
- Consumes: engine `TacticalV2StartProfile(string id, int learnerUnitCount, int opponentUnitCount, string separation)` and `TacticalV2StartWeight(string profileId, int basisPoints)`.
- Produces: `MlTrainingTacticalV2.StartProfiles`, `MlTrainingTacticalV2.StartDistribution`, and lossless `MlTrainingScenarioFile.Load/Serialize` behavior for `profiled-seeded-v1`.

- [ ] **Step 1: Write the failing profiled-scenario integration test**

Create `ProfiledTacticalV2Scenario.json` as a committed, byte-for-byte test
fixture copied from `python/config/annihilation-imitation-v1.json`. Add this
test to `MlTrainingScenarioTests`:

```csharp
[Test]
public void ProfiledTacticalV2Scenario_LoadsConvertsAndRoundTrips()
{
    string path = Path.Combine(
        "Assets", "HexWars", "Tests", "Editor", "Fixtures",
        "ProfiledTacticalV2Scenario.json");

    MlTrainingScenario loaded = MlTrainingScenarioFile.Load(path);
    TrainingScenario converted = MlTrainingScenarioPreflight.ToEngine(loaded);
    string roundTrip = MlTrainingScenarioFile.Serialize(loaded);
    MlTrainingScenario restored =
        MlTrainingScenarioFile.Parse(roundTrip, "profiled round trip");

    Assert.That(loaded.TacticalV2.PlacementPolicy, Is.EqualTo("profiled-seeded-v1"));
    Assert.That(loaded.TacticalV2.StartProfiles.Select(item => item.Id),
        Does.Contain("conversion-1v1-far"));
    Assert.That(loaded.TacticalV2.StartDistribution
        .Single(item => item.ProfileId == "standard-3v3").BasisPoints,
        Is.EqualTo(7000));
    Assert.That(converted.TacticalV2.StartProfiles, Has.Count.EqualTo(10));
    Assert.That(converted.TacticalV2.StartDistribution
        .Single(item => item.ProfileId == "standard-3v3").BasisPoints,
        Is.EqualTo(7000));
    Assert.That(restored.TacticalV2.StartProfiles.Select(item => item.Id),
        Is.EqualTo(loaded.TacticalV2.StartProfiles.Select(item => item.Id)));
    Assert.That(restored.TacticalV2.StartDistribution
        .Select(item => (item.ProfileId, item.BasisPoints)),
        Is.EqualTo(loaded.TacticalV2.StartDistribution
            .Select(item => (item.ProfileId, item.BasisPoints))));
}
```

This test catches removal of either strict-key support, wire mapping, semantic validation, engine conversion, or serialization.

- [ ] **Step 2: Run the focused test and verify RED**

Create `C:\Users\cddal\HexWars\.worktrees\unity-viewer-compat-verification`
as a detached worktree at the current branch tip, copy the new fixture and
test file into it, and copy `Assets/HexWars/Plugins/HexWars.Engine.dll` plus
its `.meta` file from the implementation worktree. Run:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\unity-viewer-compat-verification' -runTests -testPlatform EditMode -testFilter 'HexWars.Presentation.Tests.MlTrainingScenarioTests.ProfiledTacticalV2Scenario_LoadsConvertsAndRoundTrips' -testResults 'Logs\profiled-scenario-red.xml' -logFile 'Logs\profiled-scenario-red.log'
```

Expected: FAIL because strict validation reports unexpected
`scenario.tactical_v2.start_distribution` and `start_profiles`.

- [ ] **Step 3: Add Unity model types and semantic validation**

Add model types:

```csharp
public sealed class MlTrainingTacticalV2StartProfile
{
    public string Id { get; set; }
    public int LearnerUnitCount { get; set; }
    public int OpponentUnitCount { get; set; }
    public string Separation { get; set; }
}

public sealed class MlTrainingTacticalV2StartWeight
{
    public string ProfileId { get; set; }
    public int BasisPoints { get; set; }
}
```

Add `List<MlTrainingTacticalV2StartProfile> StartProfiles` and
`List<MlTrainingTacticalV2StartWeight> StartDistribution` to
`MlTrainingTacticalV2`.

Update `ValidateTacticalV2` so:

```csharp
if (TacticalV2.PlacementPolicy == "symmetric-random-v1")
{
    if ((TacticalV2.StartProfiles?.Count ?? 0) != 0)
        errors.Add("symmetric tactical-v2 placement must not declare start profiles");
    if ((TacticalV2.StartDistribution?.Count ?? 0) != 0)
        errors.Add("symmetric tactical-v2 placement must not declare a start distribution");
}
else if (TacticalV2.PlacementPolicy == "profiled-seeded-v1")
{
    if (TacticalV2.StartProfiles == null || TacticalV2.StartProfiles.Count == 0)
        errors.Add("profiled tactical-v2 placement requires start profiles");
    if (TacticalV2.StartDistribution == null || TacticalV2.StartDistribution.Count == 0)
        errors.Add("profiled tactical-v2 placement requires a start distribution");
}
else
{
    errors.Add(
        "tactical-v2 placement policy must be 'symmetric-random-v1' or 'profiled-seeded-v1'");
}
```

Also validate nonempty profile IDs/separations, positive learner/opponent counts,
nonempty distribution profile IDs, and nonnegative basis points. The engine
conversion in Step 5 remains authoritative for exact catalog and 10,000-point
sum validation.

- [ ] **Step 4: Extend strict JSON and wire round-trip support**

Use a base tactical-v2 key set:

```csharp
static readonly string[] TacticalV2Keys =
{
    "starting_unit_count", "max_controllable_units", "placement_policy", "templates",
};
static readonly string[] ProfiledTacticalV2Keys =
{
    "starting_unit_count", "max_controllable_units", "placement_policy",
    "start_profiles", "start_distribution", "templates",
};
static readonly string[] StartProfileKeys =
{
    "id", "learner_units", "opponent_units", "separation",
};
static readonly string[] StartWeightKeys = { "profile_id", "basis_points" };
```

In strict validation, read `placement_policy` first, select the exact key set,
and validate each nested array object and primitive.

Add serializable wire types with fields:

```csharp
sealed class MlTrainingTacticalV2StartProfileWire
{
    public string id;
    public int learner_units;
    public int opponent_units;
    public string separation;
}

sealed class MlTrainingTacticalV2StartWeightWire
{
    public string profile_id;
    public int basis_points;
}
```

Map both arrays in `MlTrainingTacticalV2Wire.ToModel` and `FromModel`.
For `symmetric-random-v1`, serialize the new arrays as `null` so the strict
standard schema remains unchanged. For `profiled-seeded-v1`, serialize the
complete arrays.

- [ ] **Step 5: Preserve profiles and weights in the engine conversion**

In `MlTrainingScenarioPreflight.ToEngine`, add:

```csharp
StartProfiles = (scenario.TacticalV2.StartProfiles ??
        new List<MlTrainingTacticalV2StartProfile>())
    .Select(item => new TacticalV2StartProfile(
        item.Id, item.LearnerUnitCount, item.OpponentUnitCount, item.Separation))
    .ToList(),
StartDistribution = (scenario.TacticalV2.StartDistribution ??
        new List<MlTrainingTacticalV2StartWeight>())
    .Select(item => new TacticalV2StartWeight(item.ProfileId, item.BasisPoints))
    .ToList(),
```

Keep the existing call to `converted.Validate()`; it proves the loaded catalog
matches the engine's versioned authority.

- [ ] **Step 6: Run the focused test and verify GREEN**

Run the same focused Unity command with
`profiled-scenario-green.xml` and `profiled-scenario-green.log`.

Expected: PASS, with no C# compilation errors.

- [ ] **Step 7: Run adjacent strict-scenario tests**

Run:

```powershell
& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\unity-viewer-compat-verification' -runTests -testPlatform EditMode -testFilter 'HexWars.Presentation.Tests.MlTrainingScenarioTests' -testResults 'Logs\profiled-scenario-suite.xml' -logFile 'Logs\profiled-scenario-suite.log'
```

Expected: all `MlTrainingScenarioTests` pass; unknown JSON fields remain
rejected.

- [ ] **Step 8: Commit Task 1 only**

```powershell
git add -- Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs Assets/HexWars/Tests/Editor/Fixtures/ProfiledTacticalV2Scenario.json
git diff --cached --check
git commit -m "fix: load profiled tactical scenarios in Unity"
```

### Task 2: Worktree-aware ReplayViewer resolution

**Files:**
- Modify: `Assets/HexWars/Editor/ReplayViewerMenu.cs:107-209,240-258`
- Modify: `Assets/HexWars/Presentation/MlEnvironmentContract.cs:11-23`
- Test: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs:18-64`

**Interfaces:**
- Consumes: `MlLabPaths.ResolvePythonExecutable(string projectRoot)`.
- Produces: `ReplayViewerMenu.ResolvePythonExecutable(string pythonDirectory)`,
  used by all viewer readiness and launch paths, and
  `MlEnvironmentContracts.Parse(string contractVersion)`, used by both viewer
  environment paths.

- [ ] **Step 1: Write the failing viewer resolver test**

Add to `ModelDuelConfigurationTests`:

```csharp
[Test]
public void ReplayViewerPython_UsesCommonRepositoryEnvironmentForLinkedWorktree()
{
    string scratch = Path.Combine(
        Path.GetTempPath(), "hexwars-viewer-python-" + Guid.NewGuid().ToString("N"));
    string repository = Path.Combine(scratch, "repo");
    string worktree = Path.Combine(scratch, "worktree");
    string gitDirectory = Path.Combine(repository, ".git", "worktrees", "feature");
    string expected = Path.Combine(
        repository, "python", "winenv", "Scripts", "python.exe");
    try
    {
        Directory.CreateDirectory(Path.GetDirectoryName(expected));
        File.WriteAllText(expected, string.Empty);
        Directory.CreateDirectory(gitDirectory);
        Directory.CreateDirectory(Path.Combine(worktree, "python"));
        File.WriteAllText(Path.Combine(worktree, ".git"), "gitdir: " + gitDirectory);
        File.WriteAllText(Path.Combine(gitDirectory, "commondir"), "../..");

        Assert.That(
            ReplayViewerMenu.ResolvePythonExecutable(Path.Combine(worktree, "python")),
            Is.EqualTo(expected));
    }
    finally
    {
        if (Directory.Exists(scratch)) Directory.Delete(scratch, recursive: true);
    }
}
```

```csharp
[Test]
public void ReplayViewerEnvironment_PreservesTacticalV2Contract()
{
    const string manifest =
        "{\"contract\":{\"version\":\"tactical-v2\"}}";

    Assert.That(
        ReplayViewerMenu.EnvironmentFromRunManifest(manifest),
        Is.EqualTo(MlEnvironmentContract.TacticalV2));
    Assert.That(
        MlEnvironmentContracts.Parse("tactical-v2"),
        Is.EqualTo(MlEnvironmentContract.TacticalV2));
    Assert.That(
        ReplayViewerMenu.EnvironmentFromScenario(
            TrainingScenario.CreateStandard("tactical-v2")),
        Is.EqualTo(MlEnvironmentContract.TacticalV2));
}
```
These tests catch duplicated local-only interpreter resolution and either
viewer path collapsing `tactical-v2` to `tactical-v1`.

- [ ] **Step 2: Run the focused test and verify RED**

Run Unity EditMode filtered to
`HexWars.Presentation.Tests.ModelDuelConfigurationTests`.

Expected: compile/test failure because
`ReplayViewerMenu.ResolvePythonExecutable`,
`ReplayViewerMenu.EnvironmentFromScenario`, and
`MlEnvironmentContracts.Parse` do not exist.

- [ ] **Step 3: Implement one viewer resolution boundary**

Add:

```csharp
public static string ResolvePythonExecutable(string pythonDirectory)
{
    if (string.IsNullOrWhiteSpace(pythonDirectory))
        throw new ArgumentException("Python directory is required.", nameof(pythonDirectory));
    DirectoryInfo project = Directory.GetParent(Path.GetFullPath(pythonDirectory));
    if (project == null)
        throw new InvalidOperationException(
            "Python directory does not have a project parent: " + pythonDirectory);
    return MlLabPaths.ResolvePythonExecutable(project.FullName);
}
```

Replace local-only `Path.Combine(pyDir, "winenv", "Scripts", "python.exe")`
construction in both `PyReady` and `LaunchDuel` with this method. Keep the
policy-server script at `Path.Combine(pyDir, "policy_server.py")`.

Add the exact inverse environment mapping:

```csharp
public static MlEnvironmentContract Parse(string contractVersion)
{
    switch (contractVersion)
    {
        case "tactical-v1": return MlEnvironmentContract.TacticalV1;
        case "adaptive-v1": return MlEnvironmentContract.AdaptiveV1;
        case "tactical-v2": return MlEnvironmentContract.TacticalV2;
        default: throw new ArgumentException(
            "Unknown ML environment contract: " + contractVersion,
            nameof(contractVersion));
    }
}

public static MlEnvironmentContract EnvironmentFromScenario(TrainingScenario scenario)
{
    if (scenario == null) throw new ArgumentNullException(nameof(scenario));
    return MlEnvironmentContracts.Parse(scenario.Environment);
}
```

Use the central parser in `EnvironmentFromRunManifest`, and use
`EnvironmentFromScenario(game.Scenario)` when `LaunchDuel(pyDir, plan)` maps
the presentation game. Do not retain a silent tactical-v1 fallback for unknown
metadata.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same `ModelDuelConfigurationTests` filter in the disposable verifier.

Expected: PASS, and the live Editor log contains no compiler errors.

- [ ] **Step 5: Run adjacent viewer tests**

Run Unity EditMode filtered to
`HexWars.Presentation.Tests.ModelDuelConfigurationTests`.

Expected: all tests pass.

- [ ] **Step 6: Commit Task 2 only**

```powershell
git add -- Assets/HexWars/Presentation/MlEnvironmentContract.cs Assets/HexWars/Editor/ReplayViewerMenu.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs
git diff --cached --check
git commit -m "fix: resolve shared Python for Unity viewer"
```

### Task 3: Integrated viewer verification

**Files:**
- Verify only; do not modify run metadata or active trainer files.

**Interfaces:**
- Consumes: completed Task 1 scenario parity and Task 2 viewer resolution.
- Produces: evidence that the selected profiled run reaches Arena launch without any of the three original boundary errors.

- [ ] **Step 1: Run the complete Unity EditMode suite**

Run Unity EditMode without a test filter and write
`Logs/viewer-compatibility-editmode.xml` plus
`Logs/viewer-compatibility-editmode.log`.

Expected: zero failed tests and zero compiler errors.

- [ ] **Step 2: Verify the original profiled run loads**

Use `MlRunPresentationPlan.Load` through the existing run-presentation tests
or an Editor test fixture pointed at:

```text
C:\Users\cddal\HexWars\python\runs\bc227-ppo-random-s227-20260802-v2
```

Expected: scenario ID `annihilation-imitation-v1`, placement policy
`profiled-seeded-v1`, and a valid checkpoint-backed presentation plan.

- [ ] **Step 3: Verify operational launch and active-run isolation**

In ML Lab, refresh and select
`bc227-ppo-random-s227-20260802-v2`, then choose Start & Watch or configure
Arena with the selected run. Confirm the Editor log no longer contains a fresh
`unexpected scenario.tactical_v2.start_distribution`,
`unexpected scenario.tactical_v2.start_profiles`, or
`Windows venv Python is unavailable` entry.

Separately query the `bc223` process and `run.json`; confirm it remains
running or has completed naturally, with no stop request written by this work.

- [ ] **Step 4: Record final evidence**

Report exact EditMode totals, the selected run/checkpoint, Editor log result,
and `bc223` state. Do not commit generated Unity logs or run artifacts.
