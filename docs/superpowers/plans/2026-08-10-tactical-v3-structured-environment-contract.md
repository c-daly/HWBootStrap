# Tactical-v3 Structured Environment Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the engine and GymServer half of the new tactical-v3 contract: variable semantic observations, complete legal-command candidates, safe one-command projections, exact action resolution, bounded annihilation reward, strict scenario/contract identities, and deterministic JSONL interaction.

**Architecture:** Tactical-v3 reuses tactical-v2's seeded initial-match construction but does not reuse its flat observation, stable unit-slot action encoding, HexCNN geometry, or silent decode fallback. New focused engine interfaces create immutable structured observations and per-decision legal candidate frames; a tactical-v3 duel environment owns state, candidates, reward, replay, and scripted-controller rotation; a separate tactical-v3 contract and GymServer wire format expose ragged semantic tables without pretending they are a fixed Gym `Box`/`Discrete` pair.

**Tech Stack:** C# `netstandard2.1` deterministic engine, C# `net8.0` GymServer, NUnit engine/process tests, `System.Text.Json`, JSONL over stdin/stdout, Unity 6 compile verification through Coplay MCP.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-08-10-generalizable-structured-imitation-design.md` at commit `6161921`.
- This plan covers Project A only. PyTorch policy/training, corpus storage, DAgger collection, curriculum expansion, and ML Lab model loading receive separate plans.
- Add `tactical-v3`; do not modify tactical-v1, tactical-v2, adaptive-v1, or checkpoint compatibility.
- Initial mechanics are annihilation, no fog, the existing nine capabilities, and only `EndTurn`, `MoveUnit`, `AttackUnit`, and `DeployUnit`.
- Reuse `TacticalV2Layout.NewGame` only for seeded boards/start profiles. Do not use `TacticalV2Coding`, `TacticalV2UnitRegistry`, fixed action offsets, or slot capacity in tactical-v3 decisions.
- Template names, display names, runtime entity IDs, catalog positions, and absolute flat action indices are not learned features.
- Candidate IDs are decision-local opaque integers. Invalid, stale, or non-round-tripping candidate IDs throw; they never become `EndTurn`.
- All collections are immutable snapshots with deterministic ordering. Capacity overflow fails before a payload is returned; no row, edge, or candidate is truncated.
- Initial no-fog projection may apply one candidate to the immutable state exactly. Fog-enabled tactical-v3 scenarios fail validation in this plan.
- Initial imitation will not optimize reward, but the environment must emit the approved decomposed terminal reward: win `+1`, loss/draw/truncation `-1`, material adjustment bounded to `[-0.20,+0.20]`, and time pressure in `[-0.05,0]`.
- No new NuGet or Python dependencies.
- Outputs are explicitly `unsealed-experimental`; do not call or extend Task 11 sealed-evidence APIs.
- Keep engine files focused; do not add tactical-v3 logic to `TacticalV2Coding.cs` or `TacticalV2DuelEnv.cs`.
- Use TDD for Tasks 1-9: demonstrate the intended RED before production edits, then GREEN. Task 10 is an independent acceptance gate over the completed implementation.
- After every task that changes C#, run the focused .NET tests and Coplay `check_compile_errors`; fix and recheck before committing.
- After environment, transition, or replay changes, run determinism/replay tests, not only the new focused test.
- Git commits contain no attribution trailers or tool credits.

---

## Planned File Structure

### Engine files

- `engine/HexWars.Engine/Rl/TacticalV3Config.cs` — stage-one match, capacity, and reward configuration plus hard validation.
- `engine/HexWars.Engine/Rl/TacticalV3Schema.cs` — token/reference/relation/candidate enums and immutable primitive DTOs.
- `engine/HexWars.Engine/Rl/TacticalV3Capabilities.cs` — canonical nine capability definitions and typed interaction edges.
- `engine/HexWars.Engine/Rl/TacticalV3Observation.cs` — structured observation aggregate, memory interface, seat observation interface, and production source.
- `engine/HexWars.Engine/Rl/TacticalV3Candidates.cs` — complete command candidates, exact projection, decision frame, enumeration, and resolver interfaces/implementations.
- `engine/HexWars.Engine/Rl/TacticalV3Reward.cs` — health-adjusted terminal reward tracker and named breakdown.
- `engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs` — two-seat external/scripted environment, current frame, replay, and truncation.
- `engine/HexWars.Engine/Rl/TacticalV3Env.cs` — single-learner facade over the duel environment.
- `engine/HexWars.Engine/Rl/TacticalV3Contract.cs` — canonical semantic, match, and capacity identities.

### GymServer files

- `engine/HexWars.GymServer/TacticalV3Wire.cs` — explicit snake-case wire DTO projection; no anonymous deep schema.
- `engine/HexWars.GymServer/Program.cs` — tactical-v3 construction and `spaces/reset/step/duel_*` routing.
- `engine/HexWars.GymServer/ScenarioJson.cs` — strict tactical-v3 scenario parsing.

### Scenario and tests

- `python/config/annihilation-structured-imitation-v1.json` — fixed stage-one scenario with tactical-v3 capacity/reward fields.
- `engine/HexWars.Engine.Tests/TacticalV3SchemaTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3ObservationTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3CandidateTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3RewardTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3ScenarioTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3ContractTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs` ? shared independent test builders; tests must not reuse production comparison/hash helpers as their oracle.

---

### Task 1: Configuration, Capacity, and Primitive Schema

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3Config.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV3Schema.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV3Capabilities.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3SchemaTests.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: `TacticalV2Config`, `UnitStats`, `TerrainType`, `PlayerId`.
- Produces:

```csharp
public sealed class TacticalV3CapacityProfile
{
    public TacticalV3CapacityProfile(
        int maxCells, int maxUnits, int maxTemplates,
        int maxCapabilityDefinitions, int maxCapabilityAllocations,
        int maxRules, int maxMemoryRecords, int maxRelations, int maxCandidates);
    public int MaxCells { get; }
    public int MaxUnits { get; }
    public int MaxTemplates { get; }
    public int MaxCapabilityDefinitions { get; }
    public int MaxCapabilityAllocations { get; }
    public int MaxRules { get; }
    public int MaxMemoryRecords { get; }
    public int MaxRelations { get; }
    public int MaxCandidates { get; }
}

public sealed class TacticalV3RewardConfig
{
    public float TerminalWin { get; }
    public TacticalV3RewardConfig(
        float terminalWin, float terminalNonWin,
        float materialAdjustmentBound, float timePressureBound, float pointsWeight);
    public float TerminalNonWin { get; }
    public float MaterialAdjustmentBound { get; }
    public float TimePressureBound { get; }
    public float PointsWeight { get; }
}

public sealed class TacticalV3Config
{
    public TacticalV2Config Match { get; }
    public TacticalV3CapacityProfile Capacity { get; }
    public TacticalV3Config(
        TacticalV2Config match,
        TacticalV3CapacityProfile capacity,
        TacticalV3RewardConfig reward);
    public TacticalV3RewardConfig Reward { get; }
    public IReadOnlyList<string> Validate();
}

public readonly struct TacticalV3TokenRef
{
    public TacticalV3TableKind Table { get; }
    public int Row { get; }
}
```

- `TacticalV3Capabilities.All` returns exactly nine definitions ordered by `TacticalV3CapabilityKind`.
- `TacticalV3Capabilities.Relations` returns immutable semantic edges such as Damage `opposes` Health, Defense `reduces` Damage, and Range/RangeArc `enables_action` Attack.

`TacticalV3Fixtures.cs` begins with independent builders used throughout the plan:

```csharp
internal static class TacticalV3Fixtures
{
    public static GameConfig CloneGame(GameConfig source, bool? fogOfWar = null);
    public static TacticalV3CapacityProfile ExperimentalCapacity(
        int? maxCells = null, int? maxCandidates = null);
    public static TacticalV2Config Match(int width = 13, int height = 9, int seed = 17);
    public static TacticalV3Config Config(int width = 13, int height = 9, int seed = 17);
}
```

Later tasks extend this test-only file with focused state, terminal-state, reward-tracker, candidate, and environment builders. Production code must not reference it.

- [ ] **Step 1: Write the failing configuration and schema tests**

Add tests that require strict positive capacities, exact approved reward constants, rejection of fog/generators/non-annihilation, nine stable capability definitions, no names in schema DTOs, immutable collections, and type-exact token references.

```csharp
[Test]
public void StageOneConfig_RejectsFogAndWrongRewardBounds()
{
    TacticalV2Config match = TacticalV2Config.Default();
    match.Game = TacticalV3Fixtures.CloneGame(match.Game, fogOfWar: true);
    var config = new TacticalV3Config(match, TacticalV3Fixtures.ExperimentalCapacity(),
        new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

    Assert.That(config.Validate(), Has.Some.Contains("fog_of_war=false"));
    Assert.Throws<ArgumentException>(() =>
        new TacticalV3RewardConfig(+1f, -0.5f, 0.20f, 0.05f, 0.5f));
}

[Test]
public void CapabilityCatalog_IsStableAndContainsNoRosterIdentity()
{
    Assert.That(TacticalV3Capabilities.All.Select(x => x.Kind), Is.EqualTo(new[]
    {
        TacticalV3CapabilityKind.Health,
        TacticalV3CapabilityKind.Damage,
        TacticalV3CapabilityKind.Defense,
        TacticalV3CapabilityKind.Movement,
        TacticalV3CapabilityKind.VerticalMovement,
        TacticalV3CapabilityKind.Range,
        TacticalV3CapabilityKind.RangeArc,
        TacticalV3CapabilityKind.Vision,
        TacticalV3CapabilityKind.VisionArc,
    }));
    Assert.That(typeof(TacticalV3CapabilityDefinition).GetProperty("Name"), Is.Null);
}
```

- [ ] **Step 2: Run the new tests and capture RED**

Run:

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3SchemaTests" --no-restore
```

Expected: compilation fails because tactical-v3 schema/config types do not exist.

- [ ] **Step 3: Implement strict immutable primitives**

Use constructor validation with `type`-exact integer/float semantics already natural to C#. Copy every incoming list to an array and expose `Array.AsReadOnly`. `TacticalV3Config.Validate()` must verify:

```csharp
if (Match.Game.FogOfWar) errors.Add("tactical-v3 stage one requires fog_of_war=false");
if (Match.Game.WinConditions != WinBy.Annihilation) errors.Add("tactical-v3 stage one requires annihilation");
if (Match.Game.GeneratorsEnabled) errors.Add("tactical-v3 stage one requires generators disabled");
if (Reward.TerminalWin != 1f || Reward.TerminalNonWin != -1f)
    errors.Add("tactical-v3 terminal rewards must be +1/-1");
if (Reward.MaterialAdjustmentBound != 0.20f || Reward.TimePressureBound != 0.05f)
    errors.Add("tactical-v3 shaping bounds must be 0.20/0.05");
```

The experimental default capacity is:

```text
cells=512, units=64, templates=32, capability_definitions=128,
capability_allocations=2048, rules=128, memory=64,
relations=65536, candidates=32768
```

- [ ] **Step 4: Run focused tests and Unity compile check**

Run the focused .NET command from Step 2. Then use Coplay `check_compile_errors`; expected: no C# compile errors.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3Config.cs engine/HexWars.Engine/Rl/TacticalV3Schema.cs engine/HexWars.Engine/Rl/TacticalV3Capabilities.cs engine/HexWars.Engine.Tests/TacticalV3SchemaTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: define tactical-v3 structured schema"
```

---

### Task 2: Seat-Relative Structured Observation

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3Observation.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3ObservationTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: `TacticalV3Config`, `TacticalV3Capabilities`, `GameState`, `PlayerId`, `TacticalV2Layout.Cells` for canonical valid-cell ordering.
- Produces:

```csharp
public interface IObservationMemory
{
    IReadOnlyList<TacticalV3MemoryToken> Snapshot(PlayerId seat);
}

public interface ISeatObservationSource
{
    TacticalV3Observation Observe(GameState state, PlayerId seat, IObservationMemory memory);
}

public sealed class EmptyObservationMemory : IObservationMemory
{
    public static EmptyObservationMemory Instance { get; }
}

public sealed class TacticalV3SeatObservationSource : ISeatObservationSource
{
    public TacticalV3SeatObservationSource(TacticalV3Config config);
    public TacticalV3Observation Observe(GameState state, PlayerId seat, IObservationMemory memory);
}
```

`TacticalV3Observation` exposes immutable `Cells`, `Units`, `Templates`, `CapabilityDefinitions`, `CapabilityAllocations`, `Rules`, `Memory`, and `Relations`.

Extend the test helper with an internal immutable `TacticalV3Fixture` containing `State` and `Source`, and a `TacticalV3Fixtures.Standard(int seed)` factory. Each test builder constructs expected rows independently from the public contract rather than calling a production canonicalization helper.

- [ ] **Step 1: Write failing observation tests**

Cover:

- one cell row per valid board tile in deterministic axial order;
- self/opponent ownership relative to requested seat;
- current/max HP, cell reference, elevation, moved/attacked, movement spent, and point cost;
- nine capability allocations per unit and template;
- no template/unit names or engine IDs in public token DTOs;
- explicit rule rows for win condition, round/cap, turn policy, points, damage floor, elevation modifiers, bounty, deploy multiplier, fog, and design budget;
- six-neighbor, occupies, and has-capability relations with valid references;
- empty memory under `EmptyObservationMemory`;
- capacity overflow throws before returning an observation;
- calling for Player 1 swaps relative ownership/points without changing authoritative state.

```csharp
[Test]
public void Observe_RepresentsMechanicsNotRosterNamesOrEngineIds()
{
    TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 17);
    TacticalV3Observation observation = f.Source.Observe(
        f.State, PlayerId.Player0, EmptyObservationMemory.Instance);

    Assert.That(observation.Cells, Has.Count.EqualTo(f.State.Board.TileCount));
    Assert.That(observation.Units, Has.All.Property("CurrentHp").GreaterThan(0));
    Assert.That(observation.CapabilityAllocations.Count,
        Is.EqualTo(9 * (observation.Units.Count + observation.Templates.Count)));
    Assert.That(typeof(TacticalV3UnitToken).GetProperty("EngineId"), Is.Null);
    Assert.That(typeof(TacticalV3TemplateToken).GetProperty("Name"), Is.Null);
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ObservationTests" --no-restore
```

Expected: compilation fails because the observation source and token aggregates do not exist.

- [ ] **Step 3: Implement canonical seat-relative projection**

Implementation ordering is contractual:

```text
cells: TacticalV2Layout.Cells order
units: self living units by entity ID, then opponent living units by entity ID
templates: self barracks index order, then opponent barracks index order
capability definitions: TacticalV3CapabilityKind numeric order
allocations: owner token row, then capability kind
rules: TacticalV3RuleKind numeric order
relations: relation kind, source table/row, target table/row, then numeric features
```

Engine IDs may be used only in private dictionaries while constructing references. They are never copied into public tokens. Use relative owner enum values `Self` and `Opponent`, not raw `PlayerId` values.

- [ ] **Step 4: Run focused tests, existing board determinism, and compile check**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ObservationTests|FullyQualifiedName~BoardGenerationTests" --no-restore
```

Then run Coplay `check_compile_errors`; expected: no errors.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3Observation.cs engine/HexWars.Engine.Tests/TacticalV3ObservationTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: project tactical-v3 seat observations"
```

---

### Task 3: Legal Candidates, Exact Projection, and Resolver

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3Candidates.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3CandidateTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: `ISeatObservationSource`, immutable `GameState`, `LegalMoves.For`, `GameEngine.Apply`, `TacticalV3CapacityProfile`.
- Produces:

```csharp
public interface ICandidateProjector
{
    TacticalV3ProjectedDelta Project(
        GameState state, PlayerId seat, Command command, TacticalV3Observation observation);
}

public interface ILegalCandidateSource
{
    TacticalV3DecisionFrame CreateFrame(
        GameState state, PlayerId seat, IObservationMemory memory, long decisionId);
}

public interface IActionResolver
{
    Command Resolve(TacticalV3DecisionFrame frame, int candidateId, GameState currentState);
}

public sealed class TacticalV3DecisionFrame
{
    public long DecisionId { get; }
    public PlayerId Seat { get; }
    public TacticalV3Observation Observation { get; }
    public IReadOnlyList<TacticalV3Candidate> Candidates { get; }
}
```

The frame privately retains its source `GameState` reference and exact candidate-to-`Command` mapping. Public candidates contain only decision-local references and factual projection.

Extend `TacticalV3Fixture` with `Candidates` and `Resolver` built from public constructors. The test fixture may assemble dependencies, but it must not calculate expected legality, ordering, projection deltas, or resolution results with production helpers.

- [ ] **Step 1: Write failing candidate and adversarial resolver tests**

Require:

- candidate commands equal all `LegalMoves.For(state)` commands in the supported four-command vocabulary;
- unsupported legal command types cause an explicit configuration/contract failure rather than being dropped;
- deterministic order: Attack, Move, Deploy, EndTurn, then semantic actor/target/cell keys;
- each candidate has `candidate_id == row index` and the current `decision_id`;
- move projection reports source/destination, horizontal/vertical spend, points delta, round delta, and terminal state;
- attack projection reports HP delta, damage, lethality, bounty/points delta, and terminal state;
- deploy projection reports template/cell and points delta;
- projection does not mutate the source state;
- resolver rejects negative, out-of-range, wrong-decision, and stale-state selections;
- valid resolver output equals the original authoritative command exactly;
- overflow fails instead of trimming candidates.

```csharp
[Test]
public void Resolver_RejectsStaleFrameInsteadOfEndingTurn()
{
    TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
    TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
        f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);
    GameState changed = GameEngine.Apply(f.State,
        new EndTurn(f.State.ActivePlayer)).NewState;

    Assert.Throws<InvalidOperationException>(() =>
        f.Resolver.Resolve(frame, 0, changed));
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3CandidateTests" --no-restore
```

Expected: compilation fails because the candidate interfaces and implementations do not exist.

- [ ] **Step 3: Implement command projection and private resolution authority**

Use `GameEngine.Apply(state, command)` for exact no-fog projection. Reject any unsuccessful result because every source command came from `LegalMoves.For`. Compute deltas by comparing the immutable before/after states; do not duplicate combat formulas.

Use reference identity to close the in-process stale-state boundary:

```csharp
if (!ReferenceEquals(frame.SourceState, currentState))
    throw new InvalidOperationException("tactical-v3 candidate frame is stale");
if (candidateId < 0 || candidateId >= frame.Candidates.Count)
    throw new ArgumentOutOfRangeException(nameof(candidateId));
Command command = frame.CommandAt(candidateId);
if (!LegalMoves.For(currentState).Contains(command))
    throw new InvalidOperationException("tactical-v3 candidate no longer round-trips");
return command;
```

- [ ] **Step 4: Run focused tests, engine legality tests, and compile check**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3CandidateTests|FullyQualifiedName~LegalMovesTests|FullyQualifiedName~CombatResolverTests" --no-restore
```

Then run Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3Candidates.cs engine/HexWars.Engine.Tests/TacticalV3CandidateTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: add tactical-v3 command candidates"
```

---

### Task 4: Bounded Annihilation Reward Breakdown

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3Reward.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3RewardTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: `TacticalV3RewardConfig`, episode start/final `GameState`, learner seat, terminal/truncated flags.
- Produces:

```csharp
public interface IRewardContract
{
    void Reset(GameState initialState, PlayerId learnerSeat);
    TacticalV3RewardBreakdown Evaluate(
        GameState state, bool terminated, bool truncated);
}

public sealed class TacticalV3RewardBreakdown
{
    public float TerminalOutcome { get; }
    public float KnownHealthAdjustedMaterialProgress { get; }
    public float PublicResourceProgress { get; }
    public float TimePressure { get; }
    public float Total { get; }
    public bool Finalized { get; }
}
```

- [ ] **Step 1: Write failing reward tests**

Test win, loss, round-cap draw, step truncation, partial damage before kill, larger-army normalization, unchanged-state non-farming, and absolute bounds.

```csharp
[TestCase(true, 0,  1.0f)]
[TestCase(true, 1, -1.0f)]
[TestCase(false, -1, -1.0f)]
public void TerminalBase_UsesWinVersusNonWinOrdering(
    bool hasWinner, int winner, float expected)
{
    GameState final = TacticalV3Fixtures.Terminal(hasWinner ? (PlayerId?)winner : null);
    var reward = TacticalV3Fixtures.Tracker(PlayerId.Player0);
    TacticalV3RewardBreakdown value = reward.Evaluate(final, terminated: true, truncated: false);
    Assert.That(value.TerminalOutcome, Is.EqualTo(expected));
    Assert.That(value.Total, expected > 0 ? Is.GreaterThanOrEqualTo(0.75f) : Is.LessThanOrEqualTo(-0.75f));
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3RewardTests" --no-restore
```

Expected: compilation fails because the reward contract does not exist.

- [ ] **Step 3: Implement terminal-only bounded shaping**

Intermediate calls return an all-zero, non-finalized breakdown. On terminal or truncation:

```text
health_adjusted_value = sum(point_cost * current_hp / max_hp) + points_weight * banked_points
normalized_delta = (final_advantage - initial_advantage) / max(1, initial_total_value)
material = clamp(normalized_delta, -0.20, +0.20)
time = -0.05 * clamp((round - 1) / round_cap, 0, 1)
terminal = +1 only for learner annihilation win, otherwise -1
total = terminal + material + time
```

Set `PublicResourceProgress` to zero in stage one because banked points already enter the material value; this avoids double counting while preserving the named interface.

- [ ] **Step 4: Run focused tests and compile check**

Run Step 2's command and Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3Reward.cs engine/HexWars.Engine.Tests/TacticalV3RewardTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: add tactical-v3 annihilation reward"
```

---

### Task 5: Tactical-v3 Duel and Single-Learner Environments

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs`
- Create: `engine/HexWars.Engine/Rl/TacticalV3Env.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: Tasks 1-4 interfaces, `TacticalV2Layout.NewGame`, `IAgent`, `ReplayFile`, `DuelTransition`.
- Produces:

```csharp
public sealed class TacticalV3DuelEnv
{
    public TacticalV3DuelEnv(TacticalV3Config config);
    public TacticalV3View Reset(int seed, IAgent? controller0, IAgent? controller1,
        PlayerId learnerSeat = PlayerId.Player0);
    public TacticalV3View Reset(int seed, IAgent? controller0, IAgent? controller1,
        string startProfileId, PlayerId referenceSeat,
        PlayerId learnerSeat = PlayerId.Player0);
    public TacticalV3View Step(long decisionId, int candidateId);
    public string ToReplay();
    public GameState State { get; }
}

public sealed class TacticalV3View
{
    public TacticalV3DecisionFrame Decision { get; }
    public TacticalV3RewardBreakdown Reward { get; }
    public PlayerId Seat { get; }
    public int Winner { get; }
    public bool Terminated { get; }
    public bool Truncated { get; }
    public string StartProfileId { get; }
    public PlayerId ReferenceSeat { get; }
}

public sealed class TacticalV3Env
{
    public TacticalV3Env(
        Func<int, IAgent> opponentFactory,
        PlayerId learnerSeat,
        TacticalV3Config config);
    public TacticalV3View Reset(int seed);
    public TacticalV3View Step(long decisionId, int candidateId);
}
```

- [ ] **Step 1: Write failing environment tests**

Require:

- deterministic reset frames and candidate order for identical seeds;
- standard and every configured conversion profile;
- reciprocal external seats;
- exact selected candidate is the only externally applied command;
- no `TacticalV2UnitRegistry` cap prevents earned reinforcement deployment;
- scripted Random/Greedy commands pass through the same transition/replay path;
- an invalid external candidate throws before state/log mutation;
- each accepted authoritative command increments the global command count once;
- truncation occurs at `Match.MaxSteps`, not `MaxSteps * 2`;
- terminal/truncation reward is finalized once;
- replay reconstructs the same final state and writing it twice is byte-identical.

```csharp
[Test]
public void InvalidCandidate_DoesNotBecomeEndTurnOrMutateState()
{
    TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
    TacticalV3View view = env.Reset(31, null, new RandomAgent(9));
    GameState before = env.State;

    Assert.Throws<ArgumentOutOfRangeException>(() =>
        env.Step(view.Decision.DecisionId, view.Decision.Candidates.Count));
    Assert.That(env.State, Is.SameAs(before));
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3DuelEnvTests" --no-restore
```

Expected: compilation fails because tactical-v3 environments do not exist.

- [ ] **Step 3: Implement the environment loop**

Use `TacticalV2Layout(config.Match).NewGame(seed, profile, learnerSeat)` only at reset. Thereafter operate on raw immutable `GameState`; do not allocate tactical-v2 registries. Store the current decision frame and require both request values to match it:

```csharp
if (decisionId != _frame.DecisionId)
    throw new InvalidOperationException("tactical-v3 decision id is stale");
Command command = _resolver.Resolve(_frame, candidateId, _state);
ApplyAccepted(command);
AdvancePastInternalControllers();
return MakeView();
```

An invalid internal scripted command may use an explicitly logged recovery `EndTurn`; this is separate from external resolution and increments `InternalFallbackCount`. External candidate failures never recover.

- [ ] **Step 4: Run focused, replay, and determinism tests**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3DuelEnvTests|FullyQualifiedName~TacticalV2DuelEnvTests|FullyQualifiedName~ReplayTests|FullyQualifiedName~ReplayFileTests|FullyQualifiedName~BoardGenerationTests" --no-restore
```

Then run Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs engine/HexWars.Engine/Rl/TacticalV3Env.cs engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: run tactical-v3 structured duels"
```

---

### Task 6: Strict Tactical-v3 Scenario Loading

**Files:**
- Modify: `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- Modify: `engine/HexWars.GymServer/ScenarioJson.cs`
- Modify: `engine/HexWars.Engine/Rl/MlContract.cs`
- Create: `python/config/annihilation-structured-imitation-v1.json`
- Create: `engine/HexWars.Engine.Tests/TacticalV3ScenarioTests.cs`

**Interfaces:**
- Consumes: `TacticalV3Config`, existing board/rule/episode DTOs, existing tactical-v2 template/profile DTO shapes.
- Produces:

```csharp
public const string TacticalV3Version = "tactical-v3"; // on MlContract for shared environment naming

public TacticalV3Config TrainingScenario.BuildTacticalV3();
public TrainingTacticalV3Config TrainingScenario.TacticalV3;
public TrainingTacticalV3RewardConfig TrainingScenario.TacticalV3Reward;
```

`TrainingTacticalV3Config` owns starting counts, placement policy, templates, profiles, distribution, and nine capacity integers. Do not alias a `tactical_v2` JSON property under a new environment name.

- [ ] **Step 1: Write failing scenario tests**

Keep serializable scenario DTOs separate from immutable runtime types. Add `TrainingTacticalV3RewardConfig` with the five JSON fields, then map it to `TacticalV3RewardConfig` inside `BuildTacticalV3()`. `LoadCheckedIn` in the example below is a private test helper implemented in `TacticalV3ScenarioTests.cs`; it resolves the repository root and uses the production strict parser.

Test:

- checked-in JSON loads and builds the exact current 13x9/five-template profiled configuration;
- `environment` must be `tactical-v3` and `tactical_v3` must be present;
- tactical-v1/v2/adaptive sections and rewards are rejected when tactical-v3 is selected;
- unknown fields at root, reward, tactical-v3, capacity, template, profile, and distribution levels fail;
- fog, generators, overlapping zones, insufficient max steps, invalid reward constants, and undersized capacity fail before command processing;
- original three environment scenario files still parse unchanged.

```csharp
[Test]
public void CheckedInScenario_BuildsStageOneStructuredConfig()
{
    TrainingScenario scenario = LoadCheckedIn("annihilation-structured-imitation-v1.json");
    TacticalV3Config config = scenario.BuildTacticalV3();
    Assert.That(scenario.Environment, Is.EqualTo("tactical-v3"));
    Assert.That(config.Match.BoardGen.Width, Is.EqualTo(13));
    Assert.That(config.Match.Templates, Has.Count.EqualTo(5));
    Assert.That(config.Reward.TerminalNonWin, Is.EqualTo(-1f));
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ScenarioTests" --no-restore
```

Expected: parser rejects unknown `tactical-v3` environment/fields.

- [ ] **Step 3: Add strict scenario DTO and parser branches**

The checked-in JSON must copy the current `annihilation-imitation-v1` board/rules/templates/profiles, change the environment and section name, and use:

```json
"reward": {
  "terminal_win": 1.0,
  "terminal_non_win": -1.0,
  "material_adjustment_bound": 0.2,
  "time_pressure_bound": 0.05,
  "points_weight": 0.5
},
"capacity": {
  "max_cells": 512,
  "max_units": 64,
  "max_templates": 32,
  "max_capability_definitions": 128,
  "max_capability_allocations": 2048,
  "max_rules": 128,
  "max_memory_records": 64,
  "max_relations": 65536,
  "max_candidates": 32768
}
```

Medium conversion profiles remain declared even if their initial training weight is zero; evaluation selects profiles explicitly.

- [ ] **Step 4: Run focused and existing scenario tests; compile check**

Extract the mechanics mapping currently embedded in `BuildTacticalV2()` into a private shared builder, for example:

```csharp
private TacticalV2Config BuildTacticalMatch(
    TrainingTacticalV2Config source,
    float pointsWeight,
    float shapeScale,
    float stepPenalty,
    float closingWeight,
    float drawCreditWeight);
```

`BuildTacticalV2()` calls it with the existing tactical-v2 reward values. `BuildTacticalV3()` calls it with the tactical-v3 points weight and zeros for the four legacy tactical-v2 shaping fields, then wraps the result in `TacticalV3Config`. This avoids calling the public `BuildTacticalV2()` environment guard and prevents old reward shaping from leaking into tactical-v3.

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ScenarioTests|FullyQualifiedName~TrainingScenarioTests|FullyQualifiedName~AdaptiveDuelEnvTests.GymServer_" --no-restore
```

Then run Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TrainingScenario.cs engine/HexWars.Engine/Rl/MlContract.cs engine/HexWars.GymServer/ScenarioJson.cs python/config/annihilation-structured-imitation-v1.json engine/HexWars.Engine.Tests/TacticalV3ScenarioTests.cs
git commit -m "feat: load tactical-v3 training scenarios"
```

---

### Task 7: Separate Tactical-v3 Contract Identities

**Files:**
- Create: `engine/HexWars.Engine/Rl/TacticalV3Contract.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3ContractTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs`

**Interfaces:**
- Consumes: `TacticalV3Config`, exact schema/capability catalogs.
- Produces:

```csharp
public sealed class TacticalV3Contract
{
    public const int SchemaVersion = 1;
    public string Version { get; }
    public string EnvironmentKind { get; }
    public string ContractHash { get; }
    public string EncodingHash { get; }
    public string CapacityHash { get; }
    public IReadOnlyDictionary<string, object> Match { get; }
    public IReadOnlyDictionary<string, object> Encoding { get; }
    public IReadOnlyDictionary<string, object> Capacity { get; }

    public static TacticalV3Contract Create(
        TacticalV3Config config, MlEnvironmentKind environmentKind);
}
```

- [ ] **Step 1: Write failing identity and separation tests**

Require:

- identical config produces byte-identical canonical JSON and hashes;
- board size/stat/rule/reward changes change `ContractHash`;
- board size and stat values do not change `EncodingHash`;
- schema/relation/candidate semantic changes change `EncodingHash`;
- capacity-only change changes `CapacityHash` but neither encoding nor match-semantic contract hash;
- template/name-only presentation change changes no hash;
- environment kind changes contract hash;
- exact lower-case 64-hex outputs;
- no `obs_len`, `n_actions`, action offsets, or roster names in encoding identity.

```csharp
[Test]
public void BoardSizeChangesMatchButNotEncodingIdentity()
{
    TacticalV3Contract standard = TacticalV3Contract.Create(TacticalV3Fixtures.Config(13, 9), MlEnvironmentKind.Duel);
    TacticalV3Contract large = TacticalV3Contract.Create(TacticalV3Fixtures.Config(24, 16), MlEnvironmentKind.Duel);
    Assert.That(large.ContractHash, Is.Not.EqualTo(standard.ContractHash));
    Assert.That(large.EncodingHash, Is.EqualTo(standard.EncodingHash));
}
```

- [ ] **Step 2: Run tests and capture RED**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ContractTests" --no-restore
```

Expected: compilation fails because `TacticalV3Contract` does not exist.

- [ ] **Step 3: Implement manual canonical serialization**

Follow `MlContract`'s invariant formatting and SHA-256 helpers, but keep a dedicated tactical-v3 canonical builder. Sort object keys ordinally and serialize enums by stable schema strings. Never use reflection/property order as hash authority.

Identity boundaries:

```text
contract_hash = match mechanics + actual current templates/stat allocations + reward + encoding_hash
encoding_hash = table fields + enum values + relation kinds + capability descriptor schema + candidate/projection schema
capacity_hash = nine capacity integers only
```

- [ ] **Step 4: Run focused and existing contract tests; compile check**

```powershell
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3ContractTests|FullyQualifiedName~MlContractTests|FullyQualifiedName~AdaptiveContractTests" --no-restore
```

Then run Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3Contract.cs engine/HexWars.Engine.Tests/TacticalV3ContractTests.cs engine/HexWars.Engine.Tests/TacticalV3Fixtures.cs
git commit -m "feat: hash tactical-v3 semantic contracts"
```

---

### Task 8: Explicit Tactical-v3 GymServer Wire DTOs

**Files:**
- Create: `engine/HexWars.GymServer/TacticalV3Wire.cs`
- Create: `engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs`

**Interfaces:**
- Consumes: `TacticalV3Contract`, `TacticalV3View`, all structured engine DTOs.
- Produces:

```csharp
internal static class TacticalV3Wire
{
    public static object Spaces(TrainingScenario scenario, TacticalV3Contract contract);
    public static object View(TacticalV3View view);
}
```

Wire names are explicit snake case. `View` emits:

```json
{
  "decision_id": 12,
  "seat": 0,
  "observation": {
    "cells": [], "units": [], "templates": [],
    "capability_definitions": [], "capability_allocations": [],
    "rules": [], "memory": [], "relations": []
  },
  "candidates": [],
  "reward": {
    "terminal_outcome": 0.0,
    "known_health_adjusted_material_progress": 0.0,
    "public_resource_progress": 0.0,
    "time_pressure": 0.0,
    "total": 0.0,
    "finalized": false
  },
  "winner": -1,
  "terminated": false,
  "truncated": false,
  "start_profile": "standard-3v3",
  "reference_seat": 0
}
```

- [ ] **Step 1: Write failing wire-shape tests**

Use reflection to call internal wire helpers as existing GymServer tests do. Assert exact top-level and nested property sets, numeric types, deterministic array order, reference ranges, absence of `obs`, `mask`, `obs_len`, `n_actions`, names, and engine IDs, and exact `contract_hash`/`encoding_hash`/`capacity_hash` handoff.

- [ ] **Step 2: Build GymServer and run tests to capture RED**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3GymServerTests.Wire" --no-restore
```

Expected: test fails because `TacticalV3Wire` is missing.

- [ ] **Step 3: Implement explicit projection without round-trip JSON tricks**

Map every engine DTO to explicit wire records/anonymous leaves. Do not serialize an engine object and parse it back. Validate every `TacticalV3TokenRef` against table lengths before returning the object and call capacity validation again at the wire boundary.

- [ ] **Step 4: Run focused tests and compile check**

Run Step 2's build/tests, then Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.GymServer/TacticalV3Wire.cs engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs
git commit -m "feat: serialize tactical-v3 structured views"
```

---

### Task 9: GymServer Tactical-v3 Routing and Process Contract

**Files:**
- Modify: `engine/HexWars.GymServer/Program.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs`

**Interfaces:**
- Consumes: strict Task 6 scenario, Task 7 contract, Task 8 wire, `TacticalV3Env`, `TacticalV3DuelEnv`.
- Produces the following JSONL commands for `--environment tactical-v3`:

```text
{"cmd":"spaces"}
{"cmd":"reset","seed":123}
{"cmd":"step","decision_id":1,"candidate_id":4}
{"cmd":"duel_spaces"}
{"cmd":"duel_reset","seed":123,"p0":"external","p1":"random","learner":0,"start_profile":"standard-3v3","reference_seat":0}
{"cmd":"duel_step","decision_id":1,"candidate_id":4}
{"cmd":"duel_save","path":"tactical-v3-smoke.replay"}
{"cmd":"close"}
```

- [ ] **Step 1: Write failing subprocess tests**

Tests must launch the freshly built Debug GymServer DLL and prove:

- tactical-v3 rejects omitted/mismatched scenario files before reading commands;
- `spaces` reports structured schemas and no fixed flat geometry;
- reset/step and duel reset/step require both decision and candidate IDs;
- stale/negative/out-of-range selections exit with a named error and do not emit a successor view;
- two identical processes/seeds emit byte-identical JSON through a fixed command sequence;
- selected commands reconstruct the saved replay final state;
- standard and conversion profiles work from either learner seat;
- tactical-v1/v2/adaptive process payloads remain byte-shape compatible;
- evidence/DAgger RPCs explicitly reject tactical-v3 rather than entering Task 11 or tactical-v2 paths.

```csharp
[Test]
public void Process_StructuredStepRequiresCurrentDecisionIdentity()
{
    using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
    JsonElement reset = server.Request("{\"cmd\":\"reset\",\"seed\":41}");
    long decision = reset.GetProperty("decision_id").GetInt64();
    Assert.That(server.Reject(
        $"{{\"cmd\":\"step\",\"decision_id\":{decision - 1},\"candidate_id\":0}}"),
        Does.Contain("decision id is stale"));
}
```

`TacticalV3ServerProcess` and `CheckedInScenario` are private test helpers implemented in `TacticalV3GymServerTests.cs`; `Start` locates the freshly built Debug GymServer DLL and uses the checked-in scenario path relative to the repository root.

- [ ] **Step 2: Build and run process tests to capture RED**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3GymServerTests.Process" --no-restore
```

Expected: GymServer exits with unsupported environment `tactical-v3`.

- [ ] **Step 3: Add isolated tactical-v3 branches**

At startup, construct only the selected environment. For tactical-v3:

```csharp
TacticalV3Config? tacticalV3Config = environment == MlContract.TacticalV3Version
    ? scenario.BuildTacticalV3()
    : null;
TacticalV3Env? tacticalV3Env = tacticalV3Config == null
    ? null
    : new TacticalV3Env(opponentFactory, learningSeat, tacticalV3Config);
TacticalV3DuelEnv? tacticalV3Duel = null;
```

Route tactical-v3 before the existing tactical-v2 final `else` branches so null-forgiving tactical-v2 variables are never reached. Parse IDs with `GetInt64`/`GetInt32` and reject unknown fields using the same exact-field helpers introduced for strict evidence RPCs.

- [ ] **Step 4: Run process tests, complete GymServer regression, and compile check**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3GymServerTests|FullyQualifiedName~AdaptiveDuelEnvTests.GymServer_|FullyQualifiedName~TacticalV2DaggerTests.GymServer_" --no-restore
```

Then run Coplay `check_compile_errors`.

- [ ] **Step 5: Commit**

```powershell
git add engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs
git commit -m "feat: expose tactical-v3 through GymServer"
```

---

### Task 10: Cross-size, Determinism, Replay, and Final Project-A Gate

**Files:**
- Modify: `engine/HexWars.Engine.Tests/TacticalV3ObservationTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3CandidateTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs`
- Create: `.superpowers/sdd/2026-08-10-generalizable-structured-imitation/project-a-report.md`

**Interfaces:**
- Consumes: complete Project-A tactical-v3 contract.
- Produces: acceptance evidence that Project B can safely consume.

- [ ] **Step 1: Add the final independent conformance tests**

Add tests that do not reuse production comparison helpers to establish:

- one `TacticalV3SeatObservationSource`/schema handles 13x9 and 24x16 configs;
- both configs have the same `EncodingHash` but different match `ContractHash`;
- token references remain valid after every candidate projection and actual step;
- two same-seed runs emit the same observations, candidates, selected commands, reward, terminal state, and replay bytes;
- both seat perspectives are reflections/owner swaps on a symmetric start;
- a deliberately tiny capacity rejects before reset payload publication;
- externally applying every legal candidate from independently recreated identical states matches its projected delta;
- no public structured DTO type exposes `Name`, `DisplayName`, `EngineId`, `UnitId`, or raw `PlayerId` as a learned numeric feature;
- tactical-v1/v2/adaptive full contract suites remain green.

- [ ] **Step 2: Run the complete tactical-v3 suite**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
dotnet test physical-checkpoint-audit.slnx --filter "FullyQualifiedName~TacticalV3" --no-restore
```

Expected: all tactical-v3 tests pass with zero failures.

- [ ] **Step 3: Run full engine/GymServer regression and determinism-sensitive tests**

```powershell
dotnet test physical-checkpoint-audit.slnx --no-restore
```

Expected: all tests pass. Do not substitute a filtered suite for this command.

- [ ] **Step 4: Verify Unity compilation and runtime logs**

Use Coplay in this order:

```text
get_unity_editor_state
check_compile_errors
get_unity_logs
```

Expected: correct HexWars project connected, zero compile errors, and no tactical-v3-related exceptions. If the Editor is unavailable, record the missing Unity verification explicitly and do not claim the Project-A gate complete.

- [ ] **Step 5: Verify scope, whitespace, and exact run smoke**

```powershell
git diff --check
git status --short
dotnet run --project engine/HexWars.GymServer/HexWars.GymServer.csproj --no-build -- --environment tactical-v3 --scenario-file python/config/annihilation-structured-imitation-v1.json --opponent random --seat 0
```

Send `spaces`, `reset`, one legal `step`, and `close` as JSONL. Confirm structured output has decision-local candidates and no flat `obs`/`mask` fields.

- [ ] **Step 6: Write the Project-A completion report**

Create `.superpowers/sdd/2026-08-10-generalizable-structured-imitation/project-a-report.md` with:

```text
commits
files changed
public interfaces
contract/encoding/capacity hashes for the checked-in scenario
focused and full test commands with exact counts
Unity compile/log result
13x9 and 24x16 conformance result
known limitations: no model, no fog, no design, no DAgger, unsealed experimental
```

- [ ] **Step 7: Commit final conformance/report changes**

```powershell
git add engine/HexWars.Engine.Tests/TacticalV3ObservationTests.cs engine/HexWars.Engine.Tests/TacticalV3CandidateTests.cs engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs engine/HexWars.Engine.Tests/TacticalV3GymServerTests.cs .superpowers/sdd/2026-08-10-generalizable-structured-imitation/project-a-report.md
git commit -m "test: complete tactical-v3 contract gate"
```

---

## Project-A Completion Boundary

Project A is complete only when:

- a strict tactical-v3 scenario creates deterministic structured decisions;
- observations contain semantic cells, units, capabilities, rules, and relations with no roster-name identity;
- candidates equal the complete supported authoritative command set;
- exact no-fog projected deltas agree with authoritative transitions;
- stale/invalid selection fails without fallback or mutation;
- win/non-win reward ordering and shaping bounds hold;
- one schema runs 13x9 and 24x16 configurations;
- GymServer JSONL works for single and duel modes without changing older protocols;
- full .NET tests and Unity compilation are green;
- output is explicitly unsealed experimental evidence.

Only then should Project B plan implementation begin: Python structured DTO validation, ragged batching, token/relation network, variable candidate scorer, tiny-corpus overfit, checkpoint adapter, and ML Lab/Unity model publication.
