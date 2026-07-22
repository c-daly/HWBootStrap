# Adaptive Roster and Deployment ML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly selected `adaptive-v1` ML contract in which both headless training and Unity duels share a maskable hierarchical action codec, six fixed roles, three redesignable slots, stable reinforcement control, and hidden pregame placement.

**Architecture:** Preserve the existing `tactical-v1` classes and protocol as the legacy path. New adaptive C# classes own all phase state, masking, observation encoding, deployment secrecy, and engine command application; GymServer only selects and transports an environment, while Python only validates the reported contract and exposes it to SB3. Unity ML Lab selects the contract deliberately and the arena either skips deployment or renders only an explicitly selected seat-safe view.

**Tech Stack:** C# / .NET Standard 2.1 engine, NUnit / .NET 8 tests, JSONL GymServer, Python 3.11, Gymnasium, NumPy, pytest, Stable-Baselines3 / sb3-contrib MaskablePPO, Unity 6 EditMode tests.

## Global Constraints

- Keep `tactical-v1` spaces, hashes, commands, default CLI behavior, and existing checkpoints unchanged.
- The new version string is exactly `adaptive-v1`; its environment kinds are `adaptive_tactical` and `adaptive_duel`.
- Contract constants are: 9 templates (6 immutable + 3 mutable), 24 controllable unit slots per seat, 6 required starting units, 132 starting-army points, 24 maximum design points, and an intermediate-decision penalty of `0.001f`.
- The six fixed templates are Frontline `7,2,3,2,2,1,1,3,1`, Assault `3,6,0,3,2,2,1,3,1`, Marksman `2,3,0,2,2,6,1,5,1`, Artillery `3,6,0,1,1,5,2,3,1`, Recon `2,1,0,5,3,1,0,7,2`, and Support `4,3,2,3,2,3,1,4,1`.
- Those six fixed lines cost 131 points in total; the 132-point starting budget deliberately admits the baseline combined-arms deployment with one point of headroom while still constraining more expensive custom compositions.
- The three reset defaults are Custom A `4,3,1,3,2,2,1,3,1`, Custom B `5,2,2,2,2,3,1,3,1`, and Custom C `3,4,1,3,2,2,1,4,1`.
- Stat catalogs are Health `1..8`, Damage `0..8`, Defense `0..8`, Movement `0..6`, VerticalMovement `0..4`, Range `0..8`, RangeArc `0..4`, Vision `0..10`, and VisionArc `0..4`; a proposed line is legal only when its sum is at most 24.
- Python must never duplicate move, attack, deployment, design, cost, fog, or turn legality; it consumes masks produced by C#.
- No current enemy unit, target, starting placement, label, log entry, or highlight may reveal a fogged unit; before both deployment confirmations, no opponent placement may be observed at all.
- Only completed phase sequences mutate engine/setup state. Every non-root phase exposes Cancel, and an invalidated sequence clears to a legal root phase without applying a command.
- Use `apply_patch` for edits, run the Unity 6000.5.0f1 batch compile command after every C# implementation task, and save/verify any Unity scene value changed at runtime.
- Do not add packages and do not add attribution trailers to commits.

## File Map and Locked Interfaces

- Create `engine/HexWars.Engine/Rl/AdaptiveContractData.cs`: exact roster, catalogs, `AdaptiveEnvConfig`, and validation.
- Create `engine/HexWars.Engine/Rl/AdaptivePhase.cs`: phase enum, action-region constants, and per-seat pending decision state.
- Create `engine/HexWars.Engine/Rl/AdaptiveUnitSlots.cs`: stable living-unit slot allocation/release.
- Create `engine/HexWars.Engine/Rl/AdaptiveDeployment.cs`: hidden placement ledger, confirm rules, reveal, and scripted deployment policies.
- Create `engine/HexWars.Engine/Rl/AdaptiveLayout.cs`: stable cell order and action/observation dimensions.
- Create `engine/HexWars.Engine/Rl/AdaptiveCoding.cs`: the sole adaptive observer, mask builder, and phase decoder.
- Create `engine/HexWars.Engine/Rl/AdaptiveTacticalEnv.cs` and `engine/HexWars.Engine/Rl/AdaptiveDuelEnv.cs`: single-learner and two-controller orchestration over the same adaptive classes.
- Modify `engine/HexWars.Engine/Command.cs`, `GameConfig.cs`, `GameEngine.cs`, `Net/CommandWire.cs`, and `ReplayFile.cs`: atomic custom-slot replacement and replay-safe engine validation.
- Modify `engine/HexWars.Engine/Rl/MlContract.cs` and `engine/HexWars.GymServer/Program.cs`: explicit legacy/adaptive selection and complete handshake metadata.
- Modify `python/hexwars_gym/env.py`, `python/selfplay_env.py`, `python/ml_lab/contracts.py`, `python/ml_lab/cli.py`, `python/ml_lab/envs.py`, `python/ml_lab/controllers.py`, and tests: version-aware client, manifests, workers, resume/inference isolation, and diagnostics.
- Modify `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`, `MlLabWindow.cs`, `Assets/HexWars/Presentation/ModelDuelDriver.cs`, `PolicyBridge.cs`, and EditMode tests: explicit environment selection, preflight/run summary, and safe adaptive arena playback.
- Modify `python/README.md`: intern-facing adaptive experiment and trained-model playthrough.
- Create: `engine/HexWars.Engine.Tests/AdaptiveFixtures.cs`: deterministic adaptive boards/states and completed-sequence enumeration shared by adaptive tests.

---

### Task 1: Pin Adaptive Configuration and the Semantic Contract

**Files:**
- Create: `engine/HexWars.Engine/Rl/AdaptiveContractData.cs`
- Modify: `engine/HexWars.Engine/Rl/MlContract.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveContractTests.cs`

**Interfaces:**
- Produces: `AdaptiveEnvConfig.Default()`, `AdaptiveEnvConfig.Validate(Board)`, `AdaptiveContractData.Templates`, `AdaptiveContractData.StatValues`, and `MlContract.CreateAdaptive(AdaptiveEnvConfig, MlEnvironmentKind)`.
- Produces handshake fields later consumed verbatim by Python: `contract_version`, `contract_hash`, `environment_kind`, `adaptive`, `action_regions`, and `observation_channels`.
- Preserves: `MlContract.Create(EnvConfig, ...)` and `MlContract.CurrentVersion == "tactical-v1"`.

- [ ] **Step 1: Write contract tests before implementation**

```csharp
[Test]
public void AdaptiveDefaults_PinRosterBudgetsAndCatalogs()
{
    var c = AdaptiveEnvConfig.Default();
    Assert.That(c.Templates.Select(x => x.Name), Is.EqualTo(new[] {
        "Frontline", "Assault", "Marksman", "Artillery", "Recon", "Support",
        "Custom A", "Custom B", "Custom C" }));
    Assert.That(c.FixedTemplateCount, Is.EqualTo(6));
    Assert.That(c.CustomTemplateCount, Is.EqualTo(3));
    Assert.That(c.MaxControllableUnits, Is.EqualTo(24));
    Assert.That(c.StartingUnitCount, Is.EqualTo(6));
    Assert.That(c.StartingArmyBudget, Is.EqualTo(132));
    Assert.That(c.MaxDesignPointCost, Is.EqualTo(24));
    Assert.That(c.IntermediateDecisionPenalty, Is.EqualTo(0.001f));
    Assert.That(c.StatValues[AdaptiveStat.Vision], Is.EqualTo(Enumerable.Range(0, 11)));
}

[Test]
public void AdaptiveContract_IsDeterministicAndSeparatedFromLegacy()
{
    var a = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveTactical);
    var b = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveTactical);
    var legacy = MlContract.Create(new EnvConfig());
    Assert.That(a.Version, Is.EqualTo("adaptive-v1"));
    Assert.That(a.EnvironmentKind, Is.EqualTo("adaptive_tactical"));
    Assert.That(a.ContractHash, Is.EqualTo(b.ContractHash));
    Assert.That(a.ContractHash, Is.Not.EqualTo(legacy.ContractHash));
    Assert.That(a.Semantics["max_controllable_units"], Is.EqualTo(24));
}
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter AdaptiveContractTests`

Expected: compilation fails because `AdaptiveEnvConfig`, `AdaptiveStat`, `CreateAdaptive`, and adaptive enum values do not exist.

- [ ] **Step 3: Add exact immutable contract data and preflight validation**

```csharp
public enum AdaptiveStat { Health, Damage, Defense, Movement, VerticalMovement, Range, RangeArc, Vision, VisionArc }

public sealed class AdaptiveEnvConfig
{
    public BoardGenConfig BoardGen { get; set; } = BoardGenConfig.Default();
    public GameConfig Game { get; set; } = GameConfig.Default(biomesEnabled: false, fogOfWar: true);
    public IReadOnlyList<UnitTemplate> Templates { get; set; } = AdaptiveContractData.Templates;
    public IReadOnlyDictionary<AdaptiveStat, IReadOnlyList<int>> StatValues { get; set; } = AdaptiveContractData.StatValues;
    public int FixedTemplateCount { get; set; } = 6;
    public int CustomTemplateCount { get; set; } = 3;
    public int MaxControllableUnits { get; set; } = 24;
    public int StartingUnitCount { get; set; } = 6;
    public int StartingArmyBudget { get; set; } = 132;
    public int MaxDesignPointCost { get; set; } = 24;
    public int MaxSteps { get; set; } = 900;
    public float IntermediateDecisionPenalty { get; set; } = 0.001f;
    public float DeploymentCompletionBonus { get; set; } = 0f;
    public static AdaptiveEnvConfig Default() => new AdaptiveEnvConfig();

    public IReadOnlyList<string> Validate(Board board)
    {
        var errors = new List<string>();
        int cells = Math.Min(board.DeploymentZone(PlayerId.Player0).Count, board.DeploymentZone(PlayerId.Player1).Count);
        int cheapest = Templates.Min(t => t.Stats.PointCost);
        if (cells < StartingUnitCount) errors.Add($"starting deployment requires {StartingUnitCount} cells per seat but only {cells} are available");
        if (StartingArmyBudget < cheapest * StartingUnitCount) errors.Add($"starting deployment requires at least {cheapest * StartingUnitCount} points but only {StartingArmyBudget} are available");
        if (Templates.Count != FixedTemplateCount + CustomTemplateCount) errors.Add("adaptive roster must contain exactly 9 templates");
        if (MaxControllableUnits < StartingUnitCount) errors.Add("maximum controllable units must cover the starting army");
        return errors;
    }
}
```

Define `AdaptiveContractData.Templates` with the nine exact lines in Global Constraints, and define all nine integer catalogs with `Array.AsReadOnly(...)`. Extend `MlEnvironmentKind` without renaming legacy members, add an adaptive constructor path that hashes a fixed-order canonical JSON document containing every template name/stat line, custom flags, catalogs, budgets, penalty, phase table, region table, observation channel list, fog rule, and effective horizon, and expose that document through `MlContract.Semantics`.

- [ ] **Step 4: Verify both adaptive and legacy contract tests**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "AdaptiveContractTests|MlContractTests"`

Expected: all tests pass; the pre-existing tactical hash determinism assertions still pass.

- [ ] **Step 5: Compile Unity and commit the contract slice**

Run: `& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -quit -projectPath 'C:\Users\cddal\HexWars\.worktrees\ml-full-game-actions' -logFile 'Logs\adaptive-contract-compile.log'`

Expected: Unity exits 0 and `rg -n "error CS|Compilation failed" Logs/adaptive-contract-compile.log` returns no matches.

```bash
git add engine/HexWars.Engine/Rl/AdaptiveContractData.cs engine/HexWars.Engine/Rl/MlContract.cs engine/HexWars.Engine.Tests/AdaptiveContractTests.cs
git commit -m "feat(ml): define adaptive environment contract"
```

---

### Task 2: Implement the Hierarchical Phase Codec and Fog-Safe Observation

**Files:**
- Create: `engine/HexWars.Engine/Rl/AdaptivePhase.cs`
- Create: `engine/HexWars.Engine/Rl/AdaptiveLayout.cs`
- Create: `engine/HexWars.Engine/Rl/AdaptiveCoding.cs`
- Create: `engine/HexWars.Engine.Tests/AdaptiveFixtures.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveCodingTests.cs`

**Interfaces:**
- Consumes: `AdaptiveEnvConfig` and its exact templates/catalogs from Task 1.
- Produces: `AdaptiveDecisionState`, `AdaptiveLayout`, `AdaptiveCoding.Observe(...)`, `AdaptiveCoding.Mask(...)`, and `AdaptiveCoding.ApplyAction(...)`.
- `AdaptiveCoding.Observe(GameState? game, AdaptiveDeployment setup, PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots) -> float[]` is the only observation encoder.
- `AdaptiveCoding.Mask(GameState? game, AdaptiveDeployment setup, PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots) -> bool[]` is the only mask builder.
- `AdaptiveCoding.ApplyAction(int action, GameState? game, AdaptiveDeployment setup, PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots) -> AdaptiveTransition` is the only decoder.
- `AdaptiveTransition` exposes `Command? Command`, `bool MutatedSetup`, `bool Intermediate`, and `bool InvalidSequence`; environments apply a non-null `Command` through `GameEngine`.
- `AdaptiveFixtures` produces an `AdaptiveFixture` with `Game`, `Setup`, `Layout`, `Slots`, and `Decision`; an `AdaptiveDeploymentFixture` with `Deployment`, `Observe(PlayerId)`, `Place(PlayerId,int,HexCoord)`, `FirstLegalCell(PlayerId)`, `CanConfirm(PlayerId)`, and `Placements(PlayerId)`; plus `GameWithHiddenEnemy()`, `RevealedGame(int)`, `Deployment(int)`, `AtPhase(AdaptivePhase)`, `Units(params int[])`, `PlaceSixAffordable(AdaptiveDeploymentFixture,PlayerId)`, `CompletedMaskedSequences(AdaptiveFixture)`, `ApplySequence(AdaptiveFixture,IReadOnlyList<int>)`, and `PlayMaskedToEnd(AdaptiveDuelEnv,AdaptiveDuelEnv.View,int)`.

- [ ] **Step 1: Pin the phase and action-region table with failing tests**

```csharp
[Test]
public void Layout_PinsActionRegionsAndPhaseGlobals()
{
    var l = new AdaptiveLayout(AdaptiveEnvConfig.Default());
    Assert.That(l.CommandOffset, Is.EqualTo(0));
    Assert.That(l.CommandCount, Is.EqualTo(12));
    Assert.That(l.UnitOffset, Is.EqualTo(12));
    Assert.That(l.TemplateOffset, Is.EqualTo(36));
    Assert.That(l.CellOffset, Is.EqualTo(45));
    Assert.That(l.StatOffset, Is.EqualTo(162));
    Assert.That(l.ValueOffset, Is.EqualTo(171));
    Assert.That(l.ActionCount, Is.EqualTo(182));
    Assert.That(Enum.GetValues(typeof(AdaptivePhase)).Length, Is.EqualTo(14));
}

[Test]
public void Observe_HidesFoggedEnemyButKeepsFriendlyAndPublicTerrain()
{
    var fixture = AdaptiveFixtures.GameWithHiddenEnemy();
    var state = new AdaptiveDecisionState(PlayerId.Player0);
    var obs = AdaptiveCoding.Observe(fixture.Game, PlayerId.Player0, state, fixture.Layout, fixture.Slots);
    Assert.That(obs[fixture.Layout.EnemyUnitPlane(0) * fixture.Layout.CellCount + fixture.HiddenEnemyCell], Is.Zero);
    Assert.That(obs[fixture.Layout.FriendlyUnitPlane(0) * fixture.Layout.CellCount + fixture.FriendlyCell], Is.GreaterThan(0f));
    Assert.That(obs[fixture.Layout.ElevationPlane * fixture.Layout.CellCount + fixture.HiddenEnemyCell], Is.GreaterThanOrEqualTo(0f));
}
```

- [ ] **Step 2: Run the codec tests and verify they fail to compile**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter AdaptiveCodingTests`

Expected: compilation fails because adaptive phase/layout/coding types do not exist.

- [ ] **Step 3: Add the exact phase and action identifiers**

```csharp
public enum AdaptivePhase
{
    DeploymentRoot, DeploymentTemplate, DeploymentCell, DeploymentPlacedUnit, DeploymentMoveCell,
    GameplayRoot, GameplayUnit, GameplayUnitCommand, GameplayMoveCell, GameplayAttackCell,
    DesignSlot, DesignStat, DesignValue, DesignConfirm,
}

public enum AdaptiveCommandChoice
{
    Cancel = 0, EndTurn = 1, ChooseUnit = 2, DeployReinforcement = 3,
    RedesignCustom = 4, ConfirmDesign = 5, DeployStartingUnit = 6,
    RepositionStartingUnit = 7, RemoveStartingUnit = 8, ConfirmDeployment = 9,
    Move = 10, Attack = 11,
}

public sealed class AdaptiveDecisionState
{
    public PlayerId Seat { get; }
    public AdaptivePhase Phase { get; private set; } = AdaptivePhase.DeploymentRoot;
    public int PendingUnitSlot { get; private set; } = -1;
    public int PendingTemplateSlot { get; private set; } = -1;
    public int PendingStat { get; private set; } = -1;
    public int PendingValue { get; private set; } = -1;
    public HashSet<HexCoord> SeenCells { get; } = new HashSet<HexCoord>();
    public void Enter(AdaptivePhase phase) => Phase = phase;
    public void SelectUnit(int slot) => PendingUnitSlot = slot;
    public void SelectTemplate(int slot) => PendingTemplateSlot = slot;
    public void SelectStat(int stat) => PendingStat = stat;
    public void SelectValue(int value) => PendingValue = value;
    public void Clear(AdaptivePhase root) { Phase = root; PendingUnitSlot = PendingTemplateSlot = PendingStat = PendingValue = -1; }
}

public readonly struct AdaptiveTransition
{
    public Command? Command { get; }
    public bool MutatedSetup { get; }
    public bool Intermediate { get; }
    public bool InvalidSequence { get; }
    public AdaptiveTransition(Command? command, bool mutatedSetup, bool intermediate, bool invalidSequence)
    { Command = command; MutatedSetup = mutatedSetup; Intermediate = intermediate; InvalidSequence = invalidSequence; }
}
```

`AdaptiveLayout` must enumerate 13x9 cells exactly as `TacticalLayout` does and use these fixed regions: 12 commands, 24 unit slots, 9 templates, 117 cells, 9 stats, and 11 values. For non-default board dimensions, only `CellCount` changes; `CellOffset = 45`, `StatOffset = CellOffset + CellCount`, `ValueOffset = StatOffset + 9`, and `ActionCount = ValueOffset + 11`.

- [ ] **Step 4: Implement one source of truth for masks, decoding, and observations**

Implement `AdaptiveCoding.Mask` as a switch on the 14 phases. In every non-root phase, set `mask[(int)AdaptiveCommandChoice.Cancel] = true`. In `GameplayRoot`, expose EndTurn only for the active non-terminal seat, expose ChooseUnit only when at least one stable slot has a legal move/attack, expose DeployReinforcement only when a template is affordable, a legal deployment cell exists, and a unit slot is free, and expose RedesignCustom only when the design fee is affordable. Under fog, derive gameplay masks and completed move/attack commands from `LegalMoves.For` on a fresh seat-visible projection: retain public terrain and the acting seat's complete state, retain only currently visible enemy units/generators, and never mutate the authoritative state. Without fog, use `LegalMoves.For(game)` directly. This projection rule makes hidden-enemy presence and location observationally indistinguishable. The authoritative engine may therefore reject a projected move or reinforcement deployment because of hidden occupancy omitted from the projection, including an intermediate move-route blocker or occupied destination; a rejection caused solely by that removed hidden occupancy is the sole permitted completed-mask rejection. Never synthesize any other legality.

The observation channel order is exactly: elevation, terrain-plains, terrain-forest, terrain-rough, terrain-water, deployment-zone-self, current-visibility, previously-seen, 9 friendly-role HP planes, 9 visible-enemy-role HP planes, and 24 friendly-slot occupancy planes. Append globals in this order: own points, visible opponent points (zero under fog), round, own living count, visible foe count, remaining setup budget, unplaced count, 14 phase one-hot values, pending unit/template/stat/value normalized values, then 9 templates × (9 normalized stats + normalized cost + fixed flag). Hidden enemies never contribute to role planes, counts, target masks, or pending values.

`ApplyAction` changes only `AdaptiveDecisionState` for intermediate choices, returns a `Command` only after a gameplay move/attack/deploy/design sequence completes, and calls `state.Clear(root)` on Cancel or stale selections. A stale completed choice returns `InvalidSequence=true`, `Command=null`, and a legal root state.

- [ ] **Step 5: Add phase-round-trip and property tests**

```csharp
[TestCase(1, PlayerId.Player0)] [TestCase(7, PlayerId.Player0)] [TestCase(31, PlayerId.Player0)]
[TestCase(61, PlayerId.Player1)] [TestCase(97, PlayerId.Player1)]
public void EveryMaskedGameplaySequence_IsAcceptedOrPreciselyHiddenBlocked(int seed, PlayerId seat)
{
    var f = AdaptiveFixtures.RevealedGame(seed, seat);
    foreach (var sequence in AdaptiveFixtures.CompletedMaskedSequences(f))
    {
        var transition = AdaptiveFixtures.ApplySequence(f, sequence);
        Assert.That(transition.Command, Is.Not.Null);
        AssertAcceptedOrPreciselyHiddenBlocked(f.Game, transition.Command!);
    }
}

[Test]
public void EveryNonRootPhase_AlwaysOffersCancel()
{
    foreach (AdaptivePhase phase in Enum.GetValues(typeof(AdaptivePhase)))
    {
        if (phase == AdaptivePhase.GameplayRoot || phase == AdaptivePhase.DeploymentRoot) continue;
        var f = AdaptiveFixtures.AtPhase(phase);
        Assert.That(AdaptiveCoding.Mask(f.Game, f.Setup, f.Decision, f.Layout, f.Slots)[0], Is.True);
    }
}
```

`AssertAcceptedOrPreciselyHiddenBlocked` first applies the command to the authoritative state. Its only accepted rejection is a move or reinforcement deployment rejected for occupancy/range, which must succeed when applied to an independently constructed seat-visible projection and for which the projection removed hidden enemy occupancy. This covers a hidden unit on an intermediate move route as well as an occupied destination. Add an adversarial regression proving that changing hidden-enemy presence and location leaves both the observation and every gameplay phase mask identical.

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter AdaptiveCodingTests`

Expected: all phase-table, fog, Cancel, stale-sequence, and masked-sequence tests pass.

- [ ] **Step 6: Compile Unity and commit the codec slice**

Run the Unity batch compile command from Task 1 with log `Logs\adaptive-codec-compile.log`; expect exit 0 and no `error CS` or `Compilation failed` match.

```bash
git add engine/HexWars.Engine/Rl/AdaptivePhase.cs engine/HexWars.Engine/Rl/AdaptiveLayout.cs engine/HexWars.Engine/Rl/AdaptiveCoding.cs engine/HexWars.Engine.Tests/AdaptiveFixtures.cs engine/HexWars.Engine.Tests/AdaptiveCodingTests.cs
git commit -m "feat(ml): add adaptive hierarchical codec"
```

---

### Task 3: Add Hidden Pregame Deployment and Seeded Scripted Adapters

**Files:**
- Create: `engine/HexWars.Engine/Rl/AdaptiveDeployment.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveDeploymentTests.cs`

**Interfaces:**
- Consumes: deployment phases and template/cell regions from Task 2.
- Produces: `AdaptiveDeployment`, `DeploymentPlacement`, `IDeploymentPolicy`, `CombinedArmsDeploymentPolicy`, and `RandomDeploymentPolicy`.
- `AdaptiveDeployment.Reveal(PlayerId firstPlayer)` is the only construction path from hidden ledgers to round-one `GameState`.

- [ ] **Step 1: Write secrecy, validation, and determinism tests**

```csharp
[Test]
public void SecondSeatObservation_IsUnchangedByFirstSeatsHiddenPlacement()
{
    var a = AdaptiveFixtures.Deployment(seed: 11);
    var before = a.Observe(PlayerId.Player1);
    a.Place(PlayerId.Player0, template: 4, a.FirstLegalCell(PlayerId.Player0));
    var after = a.Observe(PlayerId.Player1);
    Assert.That(after, Is.EqualTo(before));
}

[Test]
public void Confirm_RequiresSixAffordableUniqueLegalPlacements()
{
    var d = AdaptiveFixtures.Deployment(seed: 12);
    Assert.That(d.CanConfirm(PlayerId.Player0), Is.False);
    AdaptiveFixtures.PlaceSixAffordable(d, PlayerId.Player0);
    Assert.That(d.CanConfirm(PlayerId.Player0), Is.True);
    Assert.That(d.TryPlace(PlayerId.Player0, 0, d.Placements(PlayerId.Player0)[0].Cell), Is.False);
}

[Test]
public void CombinedArmsPolicy_IsSeededAndDoesNotReadOpponentLedger()
{
    var a = AdaptiveFixtures.Deployment(seed: 13);
    var b = AdaptiveFixtures.Deployment(seed: 13);
    b.Place(PlayerId.Player1, 2, b.FirstLegalCell(PlayerId.Player1));
    var policy = new CombinedArmsDeploymentPolicy(99);
    Assert.That(policy.Choose(a.View(PlayerId.Player0)), Is.EqualTo(policy.Choose(b.View(PlayerId.Player0))));
}
```

- [ ] **Step 2: Run the focused tests and verify the red state**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter AdaptiveDeploymentTests`

Expected: compilation fails because deployment types do not exist.

- [ ] **Step 3: Implement per-seat hidden ledgers and atomic reveal**

```csharp
public readonly struct DeploymentPlacement
{
    public int Slot { get; }
    public int TemplateIndex { get; }
    public HexCoord Cell { get; }
    public DeploymentPlacement(int slot, int templateIndex, HexCoord cell)
    { Slot = slot; TemplateIndex = templateIndex; Cell = cell; }
}

public interface IDeploymentPolicy
{
    IReadOnlyList<DeploymentPlacement> Choose(AdaptiveDeploymentView view);
}

public sealed class AdaptiveDeploymentView
{
    public PlayerId Seat { get; }
    public Board Board { get; }
    public IReadOnlyList<UnitTemplate> Templates { get; }
    public IReadOnlyList<DeploymentPlacement> OwnPlacements { get; }
    public int RemainingBudget { get; }
    public int RequiredUnits { get; }
    // Constructor copies only these seat-safe values; there is intentionally no opponent ledger.
}

public sealed class AdaptiveDeployment
{
    public Board Board { get; }
    public bool IsRevealed => Confirmed(PlayerId.Player0) && Confirmed(PlayerId.Player1);
    public IReadOnlyList<DeploymentPlacement> Placements(PlayerId seat);
    public AdaptiveDeploymentView View(PlayerId seat);
    public bool TryPlace(PlayerId seat, int templateIndex, HexCoord cell);
    public bool TryMove(PlayerId seat, int placementSlot, HexCoord cell);
    public bool TryRemove(PlayerId seat, int placementSlot);
    public bool CanConfirm(PlayerId seat);
    public bool TryConfirm(PlayerId seat);
    public GameState Reveal(PlayerId firstPlayer);
}
```

`View(seat)` copies only public board/config values and that seat's placements, budget, and pool; it has no opponent-placement property. `TryPlace` assigns the lowest free placement slot, rejects occupied/impassable/out-of-zone cells, a seventh unit, or a line exceeding the 132-point budget. `TryMove` preserves the placement slot. `Reveal` requires both confirmations, assigns entity IDs deterministically by seat then placement slot, copies each seat's separately mutable nine-template barracks, and creates round 1 with the configured first player in one operation.

- [ ] **Step 4: Implement the two seeded deployment policies**

`RandomDeploymentPolicy` shuffles legal cells and legal affordable template choices with its constructor seed. `CombinedArmsDeploymentPolicy` chooses templates in the fixed order Frontline, Assault, Marksman, Artillery, Recon, Support, then scores cells without opponent input: Frontline/Assault favor greater forward depth, Marksman/Artillery favor elevation and rear depth, Recon favors forward depth and spacing, Support minimizes summed distance to already chosen friendly cells. Break score ties with a seeded random value and finally axial `Q,R` order.

- [ ] **Step 5: Run deployment tests, determinism tests, and commit**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "AdaptiveDeploymentTests|BoardSeedControlTests|ReplayTests"`

Expected: all tests pass; two identical seeds/policy choices produce byte-for-byte equal revealed starts; seat-one observations do not change when seat zero changes a hidden placement.

Run the Unity batch compile command from Task 1 with log `Logs\adaptive-deployment-compile.log`; expect exit 0 and no `error CS` or `Compilation failed` match.

```bash
git add engine/HexWars.Engine/Rl/AdaptiveDeployment.cs engine/HexWars.Engine.Tests/AdaptiveDeploymentTests.cs
git commit -m "feat(ml): add hidden adaptive deployment"
```

---

### Task 4: Make Custom Designs Atomic and Reinforcement Unit Slots Stable

**Files:**
- Create: `engine/HexWars.Engine/Rl/AdaptiveUnitSlots.cs`
- Modify: `engine/HexWars.Engine/Command.cs`
- Modify: `engine/HexWars.Engine/GameConfig.cs`
- Modify: `engine/HexWars.Engine/GameEngine.cs`
- Modify: `engine/HexWars.Engine/Net/CommandWire.cs`
- Modify: `engine/HexWars.Engine/ReplayFile.cs`
- Test: `engine/HexWars.Engine.Tests/ReplaceTemplateTests.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveUnitSlotsTests.cs`

**Interfaces:**
- Consumes: `AdaptiveCoding` returns `ReplaceTemplate` only for slots 6-8.
- Produces: `ReplaceTemplate(PlayerId Issuer, int TemplateIndex, UnitStats Stats, string Name = "")` and `AdaptiveUnitSlots`.
- `AdaptiveUnitSlots.Sync(GameState, PlayerId)` releases dead IDs and assigns every untracked living ID to the lowest free slot.

- [ ] **Step 1: Write failing engine and stable-slot tests**

```csharp
[Test]
public void ReplaceTemplate_IsAtomicChargesFeeAndKeepsIndex()
{
    var s = AdaptiveFixtures.RevealedGame(21).Game;
    var stats = new UnitStats(5, 4, 1, 3, 2, 2, 1, 4, 1);
    var r = GameEngine.Apply(s, new ReplaceTemplate(PlayerId.Player0, 6, stats, "Counter"));
    Assert.That(r.Success, Is.True);
    Assert.That(r.NewState.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(9));
    Assert.That(r.NewState.Player(PlayerId.Player0).Barracks[6].Name, Is.EqualTo("Counter"));
    Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(s.Player(PlayerId.Player0).Points - s.Config.DesignFee));
}

[Test]
public void Slots_ReleaseDeadUnitAndGiveLowestSlotToReinforcement()
{
    var slots = new AdaptiveUnitSlots(4);
    slots.Sync(AdaptiveFixtures.Units(10, 11, 12), PlayerId.Player0);
    slots.Sync(AdaptiveFixtures.Units(10, 12), PlayerId.Player0);
    slots.Sync(AdaptiveFixtures.Units(10, 12, 20), PlayerId.Player0);
    Assert.That(slots.UnitIdAt(1), Is.EqualTo(20));
}
```

- [ ] **Step 2: Run the focused tests and verify they fail**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "ReplaceTemplateTests|AdaptiveUnitSlotsTests"`

Expected: compilation fails because the command and slot table do not exist.

- [ ] **Step 3: Add engine-level design legality and atomic replacement**

Add `MaxDesignPointCost` to `GameConfig` with default `0` meaning unlimited, pass it through `GameConfig.Default`, and serialize it as `maxDesignCost=` in replay CONFIG. Use this validator for both `CreateUnit` and `ReplaceTemplate`:

```csharp
static bool ValidDesign(UnitStats s, int maxCost) =>
    s.Health >= 1 && s.Damage >= 0 && s.Defense >= 0 && s.Movement >= 0 &&
    s.VerticalMovement >= 0 && s.Range >= 0 && s.RangeArc >= 0 &&
    s.Vision >= 0 && s.VisionArc >= 0 && (maxCost <= 0 || s.PointCost <= maxCost);

private static Result ApplyReplaceTemplate(GameState state, ReplaceTemplate c)
{
    var player = state.Player(c.Issuer);
    if (c.TemplateIndex < 0 || c.TemplateIndex >= player.Barracks.Count)
        return Result.Reject(state, RejectionReason.TemplateNotFound);
    if (!ValidDesign(c.Stats, state.Config.MaxDesignPointCost))
        return Result.Reject(state, RejectionReason.InvalidStats);
    if (player.Points < state.Config.DesignFee)
        return Result.Reject(state, RejectionReason.InsufficientPoints);
    var barracks = new List<UnitTemplate>(player.Barracks);
    barracks[c.TemplateIndex] = new UnitTemplate(UnitTemplate.Sanitize(c.Name), c.Stats);
    var updated = new PlayerState(player.Id, player.Points - state.Config.DesignFee,
        barracks, player.UnitsOnBoard, player.Generators, player.DestroyedValue);
    return Result.Ok(WithPlayer(state, updated));
}
```

Update `AdaptiveEnvConfig.Default()` to construct its `GameConfig` with `maxDesignPointCost: 24`; this is the value used by the adaptive contract hash and by `GameEngine` at runtime.

Add `REPLACE <seat> <index> <9 stats> <encoded name>` to `CommandWire.Write/Read`; replay needs no new header because it already delegates command lines to `CommandWire`. Add round-trip tests for named and unnamed replacements.

- [ ] **Step 4: Implement stable capacity and wire it into adaptive masking**

```csharp
public sealed class AdaptiveUnitSlots
{
    readonly int[] _unitIds;
    public AdaptiveUnitSlots(int capacity) { _unitIds = Enumerable.Repeat(-1, capacity).ToArray(); }
    public int Capacity => _unitIds.Length;
    public int UnitIdAt(int slot) => slot >= 0 && slot < Capacity ? _unitIds[slot] : -1;
    public int SlotOf(int unitId) => Array.IndexOf(_unitIds, unitId);
    public bool HasFreeSlot => Array.IndexOf(_unitIds, -1) >= 0;
    public void Sync(GameState state, PlayerId seat);
}
```

`Sync` first clears IDs absent from `state.Player(seat).UnitsOnBoard`, then iterates living units ordered by entity ID and assigns unknown IDs to the lowest `-1` entry. Update `AdaptiveCoding.Mask` to mask reinforcement deployment when `HasFreeSlot` is false and to use `UnitIdAt` for move/attack decoding.

- [ ] **Step 5: Verify engine, replay, slot reuse, and fixed-slot immutability**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "ReplaceTemplateTests|AdaptiveUnitSlotsTests|ReplayFileTests|AdaptiveCodingTests"`

Expected: all tests pass, a replacement replays identically, fixed slots never receive a mask in `DesignSlot`, and a reinforcement becomes controllable in the lowest released unit slot.

Run the Unity batch compile command from Task 1 with log `Logs\adaptive-slots-compile.log`; expect exit 0 and no `error CS` or `Compilation failed` match.

```bash
git add engine/HexWars.Engine/Rl/AdaptiveUnitSlots.cs engine/HexWars.Engine/Command.cs engine/HexWars.Engine/GameConfig.cs engine/HexWars.Engine/GameEngine.cs engine/HexWars.Engine/Net/CommandWire.cs engine/HexWars.Engine/ReplayFile.cs engine/HexWars.Engine.Tests/ReplaceTemplateTests.cs engine/HexWars.Engine.Tests/AdaptiveUnitSlotsTests.cs engine/HexWars.Engine.Tests/AdaptiveCodingTests.cs
git commit -m "feat(engine): support adaptive unit redesign and slots"
```

---

### Task 5: Share Adaptive Environments and Select Them Explicitly in GymServer

**Files:**
- Create: `engine/HexWars.Engine/Rl/AdaptiveTacticalEnv.cs`
- Create: `engine/HexWars.Engine/Rl/AdaptiveDuelEnv.cs`
- Modify: `engine/HexWars.GymServer/Program.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveTacticalEnvTests.cs`
- Test: `engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs`

**Interfaces:**
- Consumes: one adaptive layout, deployment coordinator, phase state per seat, coding functions, and slot table per seat.
- Produces: the same public geometry/reset/step surface as legacy environments plus `AdaptiveDiagnostics` and `DeploymentComplete`.
- GymServer adds `--environment tactical-v1|adaptive-v1`, defaulting to `tactical-v1`; existing JSON command names remain unchanged.

- [ ] **Step 1: Write cross-environment and replay tests first**

```csharp
[Test]
public void TacticalAndDuelAdaptiveEnvironments_ReportIdenticalSpaces()
{
    var tactical = new AdaptiveTacticalEnv(s => new GreedyAgent(s), s => new CombinedArmsDeploymentPolicy(s));
    var duel = new AdaptiveDuelEnv();
    Assert.That(tactical.ActionCount, Is.EqualTo(duel.ActionCount));
    Assert.That(tactical.ObservationLength, Is.EqualTo(duel.ObservationLength));
    Assert.That(tactical.Contract.Version, Is.EqualTo("adaptive-v1"));
}

[Test]
public void TwoExternalPolicies_DeployFightAndReconstructReplay()
{
    var env = new AdaptiveDuelEnv();
    var view = env.Reset(33, null, null, null, null, PlayerId.Player0);
    view = AdaptiveFixtures.PlayMaskedToEnd(env, view, 34);
    var replay = new Replay(ReplayFile.Read(env.ToReplay()).Start, ReplayFile.Read(env.ToReplay()).Commands);
    Assert.That(replay.Final.Winner, Is.EqualTo(env.State.Winner));
}
```

- [ ] **Step 2: Run the adaptive environment tests and verify the red state**

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "AdaptiveTacticalEnvTests|AdaptiveDuelEnvTests"`

Expected: compilation fails because adaptive environment classes do not exist.

- [ ] **Step 3: Implement both orchestrators over shared state**

```csharp
public readonly struct AdaptiveDiagnostics
{
    public int DesignCount { get; }
    public int DistinctCustomTemplatesDeployed { get; }
    public int PregameDecisions { get; }
    public int InvalidSequences { get; }
    public bool DeploymentCompleted { get; }
}

public sealed class AdaptiveTacticalEnv
{
    public AdaptiveTacticalEnv(Func<int, IAgent> opponentFactory,
        Func<int, IDeploymentPolicy> deploymentFactory, PlayerId learningSeat = PlayerId.Player0,
        AdaptiveEnvConfig? config = null);
    public int ActionCount { get; }
    public int ObservationLength { get; }
    public MlContract Contract { get; }
    public float[] Reset(int seed);
    public StepResult Step(int action);
    public bool[] LegalActionMask();
}

public sealed class AdaptiveDuelEnv
{
    public AdaptiveDuelEnv(AdaptiveEnvConfig? config = null);
    public bool DeploymentComplete { get; }
    public GameState State { get; }
    public View Reset(int seed, IAgent? controller0, IAgent? controller1,
        IDeploymentPolicy? deployment0, IDeploymentPolicy? deployment1,
        PlayerId learnerSeat = PlayerId.Player0);
    public View Step(int action);
    public string ToReplay();
}
```

Each intermediate phase step increments the environment decision count, applies `-IntermediateDecisionPenalty`, and does not advance the engine turn. A completed design applies `ReplaceTemplate`; a completed move/attack/deploy/end applies its decoded engine command. On failed application, increment `InvalidSequences`, clear the acting seat to the appropriate root, apply no fallback command, and return a valid root mask. When a scripted seat is deploying, invoke its `IDeploymentPolicy` once and validate each placement through `AdaptiveDeployment`; when it is playing, retain the existing guarded `IAgent.Decide` loop. Capture `_start` immediately after atomic reveal so replay contains the revealed round-one armies plus gameplay/design commands.

- [ ] **Step 4: Add explicit GymServer selection without changing legacy defaults**

Parse `--environment` before constructing environments. With `tactical-v1`, execute the current code path byte-for-byte. With `adaptive-v1`, construct adaptive tactical/duel instances and return adaptive spaces. Add these fields to adaptive reset/step replies only:

```json
{"deployment_complete":false,"diagnostics":{"design_count":0,"distinct_custom_templates_deployed":0,"pregame_decisions":1,"invalid_sequences":0,"deployment_completed":false}}
```

The adaptive `spaces` response includes the complete `adaptive` and phase/region/channel metadata from `MlContract.Semantics`. Unknown environment values must write `unsupported environment 'value'` to stderr and exit 2 before reading stdin.

- [ ] **Step 5: Add a process-level legacy/adaptive smoke test and run all engine tests**

Add a test that starts the built GymServer twice, asserts no-argument `spaces.contract_version == tactical-v1`, asserts `--environment adaptive-v1` reports `adaptive-v1` and a different fixed action size, then resets and takes 100 masked adaptive actions without malformed masks or out-of-range errors.

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

Expected: all engine tests pass, including deterministic adaptive deployment and duel replay reconstruction.

Run the Unity batch compile command from Task 1 with log `Logs\adaptive-env-compile.log`; expect exit 0 and no `error CS` or `Compilation failed` match.

```bash
git add engine/HexWars.Engine/Rl/AdaptiveTacticalEnv.cs engine/HexWars.Engine/Rl/AdaptiveDuelEnv.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/AdaptiveTacticalEnvTests.cs engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs
git commit -m "feat(ml): expose adaptive headless environments"
```

---

### Task 6: Integrate Adaptive Training, Manifests, ML Lab, and Safe Arena Playback

**Files:**
- Modify: `python/hexwars_gym/env.py`
- Modify: `python/selfplay_env.py`
- Modify: `python/ml_lab/contracts.py`
- Modify: `python/ml_lab/cli.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/ml_lab/controllers.py`
- Modify: `python/ml_lab/evaluation.py`
- Modify: `python/tests/test_gym_client.py`
- Modify: `python/tests/test_cli.py`
- Modify: `python/tests/test_training.py`
- Modify: `python/tests/test_controllers.py`
- Modify: `python/tests/test_evaluation.py`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Presentation/PolicyBridge.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`
- Modify: `python/README.md`

**Interfaces:**
- Python `RunConfig.environment` is exactly `tactical-v1` or `adaptive-v1`; new runs default to `tactical-v1` and resumes inherit the source value.
- `HexWarsEnv(..., environment="tactical-v1")` and `SelfPlayEnv(..., environment="tactical-v1")` append `--environment` to GymServer.
- `EnvironmentContract.semantics` stores the complete adaptive semantic object and defaults to `{}` for legacy fixtures/manifests.
- Unity `MlEnvironmentContract` has `TacticalV1` and `AdaptiveV1`; ML Lab emits `--environment adaptive-v1` only when selected.
- Unity `ModelDuelPresentationState.ShouldRender(bool deploymentComplete) -> bool` returns false before reveal for adaptive games and true throughout tactical games.

- [ ] **Step 1: Write Python red tests for selection and strict compatibility**

```python
def test_adaptive_client_accepts_complete_contract_and_keeps_fixed_spaces(tmp_path):
    spaces = _valid_adaptive_spaces()
    env = HexWarsEnv(_fake_server(tmp_path, spaces), environment="adaptive-v1")
    assert env.contract.version == "adaptive-v1"
    assert env.action_space.n == spaces["n_actions"]
    assert env.contract.semantics["max_controllable_units"] == 24
    env.close()

def test_controller_rejects_legacy_checkpoint_for_adaptive_runtime(contract):
    adaptive = replace(contract, version="adaptive-v1", contract_hash="d" * 64)
    with pytest.raises(ControllerResolutionError, match="encoding version"):
        _validate_contract_compatibility(contract, adaptive)

def test_cli_records_explicit_adaptive_environment(runner):
    cli.main(["train", "--run", "adaptive-one", "--environment", "adaptive-v1"])
    assert runner.config.environment == "adaptive-v1"
```

- [ ] **Step 2: Run the focused Python tests and verify the red state**

Run: `python -m pytest python/tests/test_gym_client.py python/tests/test_cli.py python/tests/test_controllers.py -q`

Expected: failures show missing `environment`/`semantics` fields and unsupported `adaptive-v1` parsing.

- [ ] **Step 3: Make Python parsing version-aware and keep SB3 spaces stable**

```python
@dataclass(frozen=True)
class EnvironmentContract:
    version: str
    contract_hash: str
    observation_size: int
    action_size: int
    board: Mapping[str, Any]
    roster: list[str]
    reward: Mapping[str, Any]
    semantics: Mapping[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class RunConfig:
    # retain existing fields in their existing order
    environment: str = "tactical-v1"
```

Split `_parse_contract` into the shared hash/shape parser plus `_validate_tactical_v1` and `_validate_adaptive_v1`. Adaptive validation must require exactly 9 templates, 6 fixed, 3 custom, maximum 24 units, 14 named phases, all six action regions with offsets matching `n_actions`, every listed observation channel, and all nine stat catalogs. It must calculate no legal masks itself. Add `environment` to `params.json` through the existing `RunConfig.to_dict`; resumes whose older manifest lacks the field receive `tactical-v1`. Require exact version equality in controller compatibility so tactical and adaptive checkpoints cannot be resumed or inferred across contracts even if tensor sizes happen to match.

Update `TrainingEnvironmentFactory` and `SelfPlayEnv` to pass `config.environment`; both single-worker and subprocess-vector tests must assert every worker reports an identical adaptive contract and a boolean mask shaped `(workers, action_size)`.

- [ ] **Step 4: Record adaptive episode diagnostics without changing win-rate semantics**

Have both Python envs return GymServer's `diagnostics` in `info`. Extend `EpisodeMonitor` to sum the last values and create `adaptive_episodes.csv` only for adaptive runs with the exact header:

```python
ADAPTIVE_MONITOR_HEADER = [
    "episode", "design_count", "distinct_custom_templates_deployed",
    "deployment_completed", "invalid_sequences", "pregame_decisions",
]
```

Do not add these values to reward or win rate. Add pytest assertions that a completed adaptive episode writes one row and a tactical episode creates no adaptive file.

Extend `python/ml_lab/evaluation.py` so evaluation JSON reads this sidecar and reports `design_count`, `distinct_custom_templates_deployed`, `deployment_completion_rate`, `invalid_sequences`, and `average_pregame_decisions` beside—not inside—the existing W-L-D/win-rate fields. Add `test_evaluation.py` assertions that changing any adaptive diagnostic leaves the calculated win rate unchanged.

- [ ] **Step 5: Write Unity red tests for explicit ML Lab and arena contract selection**

```csharp
[Test]
public void BuildTrainArguments_EmitsAdaptiveEnvironmentOnlyWhenSelected()
{
    var c = MlLabConfig.Default();
    Assert.That(c.BuildTrainArguments(), Does.Contain("--environment tactical-v1"));
    c.Environment = MlEnvironmentContract.AdaptiveV1;
    Assert.That(c.BuildTrainArguments(), Does.Contain("--environment adaptive-v1"));
}

[Test]
public void AdaptiveArena_SkipsHiddenDeploymentAndRendersOnlyAfterReveal()
{
    var state = new ModelDuelPresentationState(MlEnvironmentContract.AdaptiveV1);
    Assert.That(state.ShouldRender(deploymentComplete: false), Is.False);
    Assert.That(state.ShouldRender(deploymentComplete: true), Is.True);
}
```

- [ ] **Step 6: Add ML Lab summaries, preflight, and arena routing**

Add an Environment dropdown to Train and Arena. For adaptive training, display a read-only preflight block with: `adaptive-v1`, six fixed roles, three custom slots, 24 maximum units, six starting units, 132 setup points, combined-arms scripted deployment, and hidden deployment. `BuildTrainArguments()` always emits the explicit environment. Existing run details read the manifest contract and display the same semantic values rather than current UI defaults.

Extend policy-server metadata and `PolicySeatInfo` with `contract_version`. Before `BeginGame`, `ModelDuelDriver` must require both model seats to match the selected environment; a scripted opponent inherits the selected environment. Construct `DuelEnv` for `tactical-v1` and `AdaptiveDuelEnv` for `adaptive-v1`. During adaptive deployment, continue requesting model actions but do not call `BoardRenderer.Render` or `RenderEntities`, do not emit `EventConsole` entries, and do not frame the camera. On `DeploymentComplete`, render the atomically revealed state once and resume normal paced combat. Live checkpoint reload remains between completed games only.

- [ ] **Step 7: Add the intern workflow to `python/README.md`**

Document this concrete experiment:

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj -c Release
python python/hexwars_ml.py doctor --json
python python/hexwars_ml.py train --run adaptive-smoke --environment adaptive-v1 --algorithm maskable_ppo --opponent greedy --timesteps 50000 --checkpoint-every 10000 --workers 4 --learner-seat alternating --tracker local
python python/hexwars_ml.py status python/runs/adaptive-smoke --json
python python/duel.py --environment adaptive-v1 --p0 run:python/runs/adaptive-smoke --p1 greedy --out replays/adaptive-smoke.replay
```

Explain the hypothetical scenario: the policy privately selects six affordable units and cells, confirms, redesigns Custom A when it needs a high-vision counter, deploys that template after earning enough points, and controls the reinforcement through the lowest free unit slot. Explain how to select Adaptive v1 in Unity ML Lab, choose the run as P1, choose Greedy or another compatible adaptive run as P2, start the Arena, inspect resolved checkpoints/W-L-D, and play a promoted official adaptive checkpoint through the existing AI-opponent path rather than exposing unfinished runs to regular players.

- [ ] **Step 8: Run Python, engine, and Unity verification**

Run: `python -m pytest python/tests -q`

Expected: all Python tests pass, including adaptive client, vector masks, manifests, diagnostics, legacy isolation, resume isolation, and controller geometry.

Run: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj`

Expected: all engine tests pass.

Run: `& 'C:\Program Files\Unity\Hub\Editor\6000.5.0f1\Editor\Unity.exe' -batchmode -nographics -projectPath 'C:\Users\cddal\HexWars\.worktrees\ml-full-game-actions' -runTests -testPlatform EditMode -testFilter 'HexWars.Presentation.Tests.MlLabConfigTests;HexWars.Presentation.Tests.ModelDuelConfigurationTests;HexWars.Presentation.Tests.PolicyBridgeProtocolTests' -testResults 'Logs\adaptive-editmode-results.xml' -logFile 'Logs\adaptive-editmode.log'`

Expected: all selected EditMode tests pass and Unity reports zero compiler errors.

- [ ] **Step 9: Perform the deterministic adaptive smoke and manual secrecy check**

Build GymServer Release, run 10,000 masked actions across seeds 0-31 with no invalid exposed sequence except the precisely classified authoritative rejection caused solely by hidden occupancy omitted from the seat-visible projection, train a 50,000-step four-worker MaskablePPO smoke run, and duel its first two validated checkpoints. Confirm the server reports an unchanged action/observation size after every reset, deployment completes, at least one replay reconstructs its final winner, and adaptive episode diagnostics remain separate from W-L-D.

In Unity, watch adaptive checkpoint vs Greedy and adaptive checkpoint vs checkpoint. Confirm nothing on screen/logs identifies either hidden army before reveal, no fogged enemy appears afterward, models/checkpoints reload only between games, and the arena renders normal combat after reveal.

- [ ] **Step 10: Commit the complete integration slice**

```bash
git add python/hexwars_gym/env.py python/selfplay_env.py python/ml_lab/contracts.py python/ml_lab/cli.py python/ml_lab/envs.py python/ml_lab/controllers.py python/ml_lab/evaluation.py python/tests/test_gym_client.py python/tests/test_cli.py python/tests/test_training.py python/tests/test_controllers.py python/tests/test_evaluation.py Assets/HexWars/Editor/MlLab/MlLabConfig.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Presentation/PolicyBridge.cs Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Tests/Editor/MlLabConfigTests.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs python/README.md
git commit -m "feat(ml): integrate adaptive training and arena"
```

## Completion Gate

- `tactical-v1` remains the default and loads every existing tactical run accepted before this work.
- `adaptive-v1` refuses tactical checkpoints before inference and records all semantic contract fields in `run.json` and `params.json`.
- Every masked completed sequence is accepted by `GameEngine` except a move or reinforcement deployment rejected solely because hidden occupancy was omitted from its seat-visible legality projection; all invalidated sequences return safely to a root mask.
- Both seats deploy six affordable units without observing the opponent; identical seeds/actions reproduce identical starts and games.
- Fixed templates cannot be redesigned, custom replacements preserve slots 6-8, dead unit slots are reused, and reinforcement deployment masks at 24 living units.
- Current observations and Unity presentation never reveal hidden deployment or fogged enemies.
- Headless single-agent training, self-play, Python duels, Unity duels, checkpoint reload, replay reconstruction, and separate adaptive diagnostics all use the same adaptive contract.
