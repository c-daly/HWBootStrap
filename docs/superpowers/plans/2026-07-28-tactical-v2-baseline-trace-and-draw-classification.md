# Tactical-v2 Baseline Trace and Draw Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Add evaluation-only, replayable tactical-v2 transition evidence and deterministic draw classification without changing gameplay, policy observations, action masks, or rewards.

**Architecture:** TacticalV2DuelEnv reports accepted transitions through an injected sink whose default is a no-op. GymServer exposes that sink only through explicit duel-trace RPCs; ordinary reset and step payloads remain schema compatible. Python validates the diagnostic schema, derives episode metrics, classifies draws with recorded evidence, and writes deterministic per-match artifacts.

**Tech Stack:** C#/.NET engine, NUnit, JSONL GymServer protocol, Python dataclasses, NumPy, pytest.

## Global Constraints

- GameEngine, commands, combat, and the tactical-v2 observation/action contract do not change.
- Existing tactical-v1, tactical-v2, and adaptive-v1 checkpoints retain their current contract hashes.
- Diagnostic state never appears in observations, masks, rewards, memory, or candidate data.
- Transition capture is disabled by default and retains no state during training.
- Every captured command must reconstruct through the existing replay path.
- Paired seeds and reciprocal seats remain the official schedule.
- Draw classification includes evidence and an explicit unclassified category.
- Existing unrelated working-tree changes remain untouched.
- Never add attribution trailers to commits.

---

### Task 1: Accepted-transition sink boundary

**Files:**
- Create: engine/HexWars.Engine/Rl/IDuelTransitionSink.cs
- Modify: engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs
- Test: engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs

**Interfaces:**
- Consumes: DuelTransition(GameState previous, Command command, GameState resulting).
- Produces: IDuelTransitionSink.Reset, IDuelTransitionSink.Accepted, BufferedDuelTransitionSink.Enabled, and BufferedDuelTransitionSink.Drain.
- Compatibility: TacticalV2DuelEnv.CaptureTransitions and DrainTransitions keep their current behavior.

- [ ] **Step 1: Write failing sink tests**

~~~csharp
[Test]
public void InjectedSink_RecordsAcceptedCommandsOnlyWhileEnabled()
{
    var sink = new BufferedDuelTransitionSink { Enabled = true };
    var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), sink);

    env.Reset(11, new GreedyAgent(3), new GreedyAgent(4));

    Assert.That(sink.Drain(), Is.Not.Empty);
    Assert.That(sink.Drain(), Is.Empty);
}

[Test]
public void InjectedSink_DisabledByDefault_RetainsNothing()
{
    var sink = new BufferedDuelTransitionSink();
    var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), sink);

    env.Reset(11, new GreedyAgent(3), new GreedyAgent(4));

    Assert.That(sink.Drain(), Is.Empty);
}
~~~

Add an external-seat case proving a rejected command is absent and accepted commands remain ordered.

- [ ] **Step 2: Run the focused tests**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DuelEnvTests"
~~~

Expected: FAIL because the sink types and constructor overload do not exist.

- [ ] **Step 3: Implement the port and no-op/buffer implementations**

~~~csharp
using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    public interface IDuelTransitionSink
    {
        void Reset(GameState initialState);
        void Accepted(DuelTransition transition);
    }

    public sealed class NullDuelTransitionSink : IDuelTransitionSink
    {
        public static readonly NullDuelTransitionSink Instance = new NullDuelTransitionSink();
        private NullDuelTransitionSink() { }
        public void Reset(GameState initialState) { }
        public void Accepted(DuelTransition transition) { }
    }

    public sealed class BufferedDuelTransitionSink : IDuelTransitionSink
    {
        private readonly List<DuelTransition> _items = new List<DuelTransition>();
        public bool Enabled { get; set; }

        public void Reset(GameState initialState) => _items.Clear();

        public void Accepted(DuelTransition transition)
        {
            if (Enabled)
                _items.Add(transition ?? throw new ArgumentNullException(nameof(transition)));
        }

        public IReadOnlyList<DuelTransition> Drain()
        {
            var result = new List<DuelTransition>(_items);
            _items.Clear();
            return result;
        }
    }
}
~~~

- [ ] **Step 4: Inject the sink without removing legacy capture**

Add the field and overload:

~~~csharp
private readonly IDuelTransitionSink _transitionSink;

public TacticalV2DuelEnv(
    TacticalV2Config config,
    IDuelTransitionSink? transitionSink = null)
{
    if (config == null) throw new ArgumentNullException(nameof(config));
    _cfg = config;
    _layout = new TacticalV2Layout(_cfg);
    _transitionSink = transitionSink ?? NullDuelTransitionSink.Instance;
}
~~~

Call _transitionSink.Reset(_state) during Reset. In TryApply, after successful application:

~~~csharp
var transition = new DuelTransition(before, cmd, _state);
_transitionSink.Accepted(transition);
if (CaptureTransitions) _transitions.Add(transition);
~~~

- [ ] **Step 5: Verify focused and full engine tests**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DuelEnvTests"
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
~~~

Expected: PASS, including all existing CaptureTransitions tests.

- [ ] **Step 6: Commit**

~~~powershell
git add engine/HexWars.Engine/Rl/IDuelTransitionSink.cs engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs
git commit -m "feat: add duel transition sink boundary"
~~~

---

### Task 2: Diagnostic transition projection

**Files:**
- Create: engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs
- Create: engine/HexWars.Engine.Tests/TacticalEvaluationTraceTests.cs

**Interfaces:**
- Consumes: DuelTransition.
- Produces: TacticalEvaluationTrace.Project(DuelTransition) returning TacticalTraceTransition.
- Security: no policy-facing class may reference these DTOs.

- [ ] **Step 1: Write failing projector tests**

Build deterministic move, attack, and EndTurn transitions using TacticalV2Config.Default.

~~~csharp
private static DuelTransition FirstTransition(Predicate<DuelTransition> matches)
{
    for (int seed = 0; seed < 20; seed++)
    {
        var env = new TacticalV2DuelEnv(TacticalV2Config.Default());
        env.CaptureTransitions = true;
        env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
        foreach (DuelTransition transition in env.DrainTransitions())
            if (matches(transition)) return transition;
    }
    throw new AssertionException("expected matching transition across seeds 0..19");
}

[Test]
public void Project_AttackPreservesCommandAndMaterialChange()
{
    DuelTransition transition = FirstTransition(item => item.Command is AttackUnit);

    TacticalTraceTransition trace = TacticalEvaluationTrace.Project(transition);

    Assert.That(trace.Command.Kind, Is.EqualTo("attack"));
    Assert.That(trace.Command.ActorId, Is.Not.Null);
    Assert.That(trace.Command.TargetId, Is.Not.Null);
    int foe = 1 - trace.Command.Issuer;
    Assert.That(trace.After.Seats[foe].HealthAdjustedMaterial,
        Is.LessThan(trace.Before.Seats[foe].HealthAdjustedMaterial));
}

[Test]
public void Project_EndTurnRecordsProductiveAlternatives()
{
    DuelTransition transition = FirstTransition(item =>
        item.Command is EndTurn
        && LegalMoves.For(item.Previous).Any(command => !(command is EndTurn)));

    TacticalTraceTransition trace = TacticalEvaluationTrace.Project(transition);

    Assert.That(trace.Command.Kind, Is.EqualTo("end_turn"));
    Assert.That(trace.Before.ProductiveLegalActions, Is.GreaterThan(0));
}
~~~

- [ ] **Step 2: Confirm the tests fail**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalEvaluationTraceTests"
~~~

Expected: FAIL because the DTOs and projector do not exist.

- [ ] **Step 3: Define JSON-serializable DTOs**

~~~csharp
public sealed class TacticalTraceTransition
{
    public TacticalTraceState Before { get; set; } = null!;
    public TacticalTraceCommand Command { get; set; } = null!;
    public TacticalTraceState After { get; set; } = null!;
}

public sealed class TacticalTraceState
{
    public int Round { get; set; }
    public int ActiveSeat { get; set; }
    public bool IsGameOver { get; set; }
    public int? Winner { get; set; }
    public int ProductiveLegalActions { get; set; }
    public TacticalTraceSeat[] Seats { get; set; } = Array.Empty<TacticalTraceSeat>();
}

public sealed class TacticalTraceSeat
{
    public int Seat { get; set; }
    public int Points { get; set; }
    public int DestroyedValue { get; set; }
    public int AliveUnits { get; set; }
    public int CurrentHitPoints { get; set; }
    public int MaximumHitPoints { get; set; }
    public double HealthAdjustedMaterial { get; set; }
    public bool CanDamageEnemy { get; set; }
    public bool CanCurrentlyAttackEnemy { get; set; }
    public bool CanMove { get; set; }
    public TacticalTraceUnit[] Units { get; set; } = Array.Empty<TacticalTraceUnit>();
}

public sealed class TacticalTraceUnit
{
    public int Id { get; set; }
    public int Q { get; set; }
    public int R { get; set; }
    public int CurrentHp { get; set; }
    public int MaximumHp { get; set; }
    public int PointCost { get; set; }
    public int Damage { get; set; }
    public int Defense { get; set; }
    public int Movement { get; set; }
    public int VerticalMovement { get; set; }
    public int Range { get; set; }
    public bool Moved { get; set; }
    public bool Attacked { get; set; }
}

public sealed class TacticalTraceCommand
{
    public string Kind { get; set; } = "";
    public int Issuer { get; set; }
    public int? ActorId { get; set; }
    public int? TargetId { get; set; }
    public int? Q { get; set; }
    public int? R { get; set; }
}
~~~

TacticalTraceUnit has Id, Q, R, CurrentHp, MaximumHp, PointCost, Damage, Defense, Movement, VerticalMovement, Range, Moved, and Attacked. TacticalTraceCommand has Kind, Issuer, nullable ActorId, TargetId, Q, and R.

- [ ] **Step 4: Implement the pure projection**

Sort unit records by ID. Calculate health-adjusted material exactly:

~~~csharp
healthAdjustedMaterial +=
    unit.Stats.PointCost * unit.CurrentHp / (double)Math.Max(1, unit.Stats.Health);
~~~

ProductiveLegalActions is LegalMoves.For(state) excluding EndTurn for the active player. CanDamageEnemy is
true when a living unit has Damage greater than zero and the opponent has a living unit. Compute
CanCurrentlyAttackEnemy with TargetingService.CanTarget over living attacker/target pairs. CanMove is true
when a living unit has positive Movement or VerticalMovement.

Map EndTurn, MoveUnit, AttackUnit, DeployUnit, CaptureHex, and BuildGenerator explicitly. Unknown commands use their CLR type name and issuer without inventing actor or target fields.

- [ ] **Step 5: Verify projector and full engine tests**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalEvaluationTraceTests"
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
~~~

Expected: PASS with deterministic unit ordering.

- [ ] **Step 6: Commit**

~~~powershell
git add engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs engine/HexWars.Engine.Tests/TacticalEvaluationTraceTests.cs
git commit -m "feat: project tactical evaluation traces"
~~~

---

### Task 3: Quarantined GymServer trace RPCs

**Files:**
- Modify: engine/HexWars.GymServer/Program.cs
- Modify: engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs

**Interfaces:**
- Produces: duel_trace_enable(enabled: bool) and duel_trace_drain().
- Invariant: spaces, reset, and step payloads remain unchanged.

- [ ] **Step 1: Write failing server tests**

~~~csharp
[Test]
public void TacticalV2GymServer_TraceIsOptInAndSeparateFromStepPayload()
{
    using var server = new ServerProcess("--environment", "tactical-v2");

    using JsonDocument enabled =
        server.Exchange(new { cmd = "duel_trace_enable", enabled = true });
    Assert.That(enabled.RootElement.GetProperty("enabled").GetBoolean(), Is.True);

    using JsonDocument reset = server.Exchange(
        new { cmd = "duel_reset", seed = 41, p0 = "greedy", p1 = "greedy", learner = 0 });
    Assert.That(reset.RootElement.TryGetProperty("trace", out _), Is.False);

    using JsonDocument trace = server.Exchange(new { cmd = "duel_trace_drain" });
    Assert.That(trace.RootElement.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
    Assert.That(trace.RootElement.GetProperty("transitions").GetArrayLength(), Is.GreaterThan(0));
}
~~~

Also test disabling, resetting, and draining yields an empty array.

- [ ] **Step 2: Confirm the server tests fail**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "Name~TacticalV2GymServer_Trace"
~~~

Expected: FAIL because the RPC commands are unsupported.

- [ ] **Step 3: Inject one disabled buffer into duel construction**

Create one process-local BufferedDuelTransitionSink and pass it into every TacticalV2DuelEnv construction:

~~~csharp
var tacticalV2Trace = new BufferedDuelTransitionSink();

tacticalV2Duel ??=
    new TacticalV2DuelEnv(tacticalV2Config!, tacticalV2Trace);
~~~

Do not inject it into TacticalV2Env training instances.

- [ ] **Step 4: Implement both RPC commands**

~~~csharp
case "duel_trace_enable":
{
    if (environment != "tactical-v2")
        throw new InvalidDataException("duel trace is supported only for tactical-v2");
    bool enabled = root.GetProperty("enabled").GetBoolean();
    tacticalV2Trace.Enabled = enabled;
    if (!enabled) tacticalV2Trace.Drain();
    Send(new { enabled });
    break;
}

case "duel_trace_drain":
{
    if (environment != "tactical-v2")
        throw new InvalidDataException("duel trace is supported only for tactical-v2");
    var transitions = tacticalV2Trace.Drain()
        .Select(TacticalEvaluationTrace.Project)
        .ToArray();
    Send(new { schema_version = 1, transitions });
    break;
}
~~~

Trace data must never be included in duel_reset or duel_step responses.

- [ ] **Step 5: Verify server and full engine tests**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "Name~TacticalV2GymServer_Trace"
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
~~~

Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs
git commit -m "feat: expose evaluation-only tactical traces"
~~~

---

### Task 4: Strict Python trace contract

**Files:**
- Create: python/ml_lab/tactical_trace.py
- Create: python/tests/test_tactical_trace.py

**Interfaces:**
- Consumes: duel_trace_drain schema version 1.
- Produces: EpisodeTrace.from_payload and EpisodeTrace.to_dict.
- Failure behavior: malformed, non-finite, negative-count, invalid-seat, duplicate-unit, or discontinuous traces raise ValueError.

- [ ] **Step 1: Write failing parser tests**

~~~python
def test_episode_trace_rejects_discontinuous_transitions() -> None:
    payload = trace_payload()
    payload["transitions"][1]["before"]["round"] += 1

    with pytest.raises(ValueError, match="transition 1 does not chain"):
        EpisodeTrace.from_payload(payload)


def test_episode_trace_round_trips_canonical_payload() -> None:
    parsed = EpisodeTrace.from_payload(trace_payload())

    assert parsed.schema_version == 1
    assert parsed.transitions[0].command.kind == "move"
    assert parsed.to_dict() == trace_payload()
~~~

Also test invalid seats, duplicate unit IDs, negative HP, and non-finite material.

- [ ] **Step 2: Confirm import failure**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_tactical_trace.py -q
~~~

Expected: FAIL because ml_lab.tactical_trace does not exist.

- [ ] **Step 3: Define immutable records**

Create frozen UnitFrame, SeatFrame, StateFrame, CommandFrame, TransitionFrame, and EpisodeTrace dataclasses. Field names, including can_currently_attack_enemy, mirror the C# DTOs in snake_case. StateFrame.seats is exactly tuple[SeatFrame, SeatFrame]; EpisodeTrace.transitions is tuple[TransitionFrame, ...].

~~~python
@dataclass(frozen=True)
class UnitFrame:
    id: int
    q: int
    r: int
    current_hp: int
    maximum_hp: int
    point_cost: int
    damage: int
    defense: int
    movement: int
    vertical_movement: int
    range: int
    moved: bool
    attacked: bool

@dataclass(frozen=True)
class SeatFrame:
    seat: int
    points: int
    destroyed_value: int
    alive_units: int
    current_hit_points: int
    maximum_hit_points: int
    health_adjusted_material: float
    can_damage_enemy: bool
    can_currently_attack_enemy: bool
    can_move: bool
    units: tuple[UnitFrame, ...]

@dataclass(frozen=True)
class StateFrame:
    round: int
    active_seat: int
    is_game_over: bool
    winner: int | None
    productive_legal_actions: int
    seats: tuple[SeatFrame, SeatFrame]

@dataclass(frozen=True)
class CommandFrame:
    kind: str
    issuer: int
    actor_id: int | None
    target_id: int | None
    q: int | None
    r: int | None

@dataclass(frozen=True)
class TransitionFrame:
    before: StateFrame
    command: CommandFrame
    after: StateFrame

@dataclass(frozen=True)
class EpisodeTrace:
    schema_version: int
    transitions: tuple[TransitionFrame, ...]
~~~

EpisodeTrace exposes:

~~~python
@classmethod
def from_payload(cls, payload: Mapping[str, Any]) -> "EpisodeTrace":
    return cls(
        schema_version=_required_schema_version(payload),
        transitions=_parse_and_validate_transitions(payload),
    )

def to_dict(self) -> dict[str, Any]:
    return _canonical_trace_dict(self)
~~~

- [ ] **Step 4: Implement strict validation**

Reject bool where int is required; require math.isfinite for material; require seats exactly {0, 1}; require unique unit IDs within each seat; require non-negative counts and HP; and require each transition.after to equal the next transition.before.

Accept JsonSerializer PascalCase or camelCase only at the parser boundary. Emit only canonical snake_case from to_dict. An empty trace is transport-valid, but downstream episode summarization must reject it when capture was requested.

- [ ] **Step 5: Verify parser and existing protocol tests**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_tactical_trace.py python\tests\test_evaluation.py python\tests\test_gym_client.py -q
~~~

Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add python/ml_lab/tactical_trace.py python/tests/test_tactical_trace.py
git commit -m "feat: validate tactical evaluation traces"
~~~

---

### Task 5: Episode metrics and deterministic draw classification

**Files:**
- Create: python/ml_lab/draw_classification.py
- Create: python/tests/test_draw_classification.py

**Interfaces:**
- Consumes: EpisodeTrace, candidate_seat, terminated, truncated, and winner.
- Produces: summarize_episode and classify_draw.
- Categories: invalid_scenario, truncation, failed_conversion, damage_stalemate, mobility_stalemate, cycling, action_waste, avoidance, balanced_attrition, unclassified.

- [ ] **Step 1: Write one failing test per category**

~~~python
def test_lopsided_draw_is_failed_conversion() -> None:
    trace = episode_trace(
        material=((10.0, 10.0), (28.0, 4.0), (26.0, 3.0)),
        commands=("attack", "move"),
        final_can_damage=(True, True),
    )

    result = classify_draw(
        trace, candidate_seat=0, terminated=True, truncated=False, winner=None
    )

    assert result.primary == DrawCategory.FAILED_CONVERSION
    assert result.evidence["peak_normalized_advantage"] >= 0.35


def test_unsupported_pattern_remains_unclassified() -> None:
    trace = episode_trace(
        material=((10.0, 10.0), (10.0, 10.0)),
        commands=("move",),
        final_can_damage=(True, True),
    )

    result = classify_draw(
        trace, candidate_seat=0, terminated=True, truncated=False, winner=None
    )

    assert result.primary == DrawCategory.UNCLASSIFIED
~~~

Test overlapping failed-conversion and cycling evidence: failed_conversion is primary and cycling remains in flags.

- [ ] **Step 2: Confirm import failure**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_draw_classification.py -q
~~~

Expected: FAIL because ml_lab.draw_classification does not exist.

- [ ] **Step 3: Define classification records and thresholds**

~~~python
class DrawCategory(str, Enum):
    INVALID_SCENARIO = "invalid_scenario"
    TRUNCATION = "truncation"
    FAILED_CONVERSION = "failed_conversion"
    DAMAGE_STALEMATE = "damage_stalemate"
    MOBILITY_STALEMATE = "mobility_stalemate"
    CYCLING = "cycling"
    ACTION_WASTE = "action_waste"
    AVOIDANCE = "avoidance"
    BALANCED_ATTRITION = "balanced_attrition"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class DrawThresholds:
    decisive_advantage: float = 0.35
    balanced_advantage: float = 0.10
    cycle_repetitions: int = 3
    wasted_end_turns: int = 3
    wasted_end_turn_ratio: float = 0.25
~~~

EpisodeSummary records command and round counts, damage and kills by seat, EndTurns and wasted EndTurns by seat, peak/final normalized advantage, and maximum state repetition. DrawClassification records primary, all flags, and numeric evidence.

- [ ] **Step 4: Implement calculations and ordered evidence rules**

Normalize material advantage by max(initial combined health-adjusted material, 1.0). Count a wasted EndTurn only when command.kind is end_turn and before.productive_legal_actions is positive. Build a round-independent cycle key from active seat, points, and sorted living-unit tuples containing seat, ID, coordinates, HP, moved, and attacked.

Use this exact primary precedence:

~~~python
precedence = (
    DrawCategory.INVALID_SCENARIO,
    DrawCategory.TRUNCATION,
    DrawCategory.FAILED_CONVERSION,
    DrawCategory.DAMAGE_STALEMATE,
    DrawCategory.MOBILITY_STALEMATE,
    DrawCategory.CYCLING,
    DrawCategory.ACTION_WASTE,
    DrawCategory.AVOIDANCE,
    DrawCategory.BALANCED_ATTRITION,
)
~~~

Rules:

- invalid_scenario: either initial seat has zero living units.
- truncation: truncated is true and the engine did not report a terminal draw.
- failed_conversion: candidate peak advantage reaches decisive_advantage.
- damage_stalemate: both retain units and at least one seat cannot damage an enemy.
- mobility_stalemate: both retain damage-capable units, neither can move, and neither can currently attack.
- cycling: maximum repetition reaches cycle_repetitions.
- action_waste: candidate wasted EndTurns reach both thresholds.
- avoidance: total damage is zero.
- balanced_attrition: both dealt damage and absolute final advantage is within balanced_advantage.
- unclassified: no supported rule fired.

Never infer mobility stalemate from distance alone.

- [ ] **Step 5: Verify classifier and parser tests**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_draw_classification.py python\tests\test_tactical_trace.py -q
~~~

Expected: PASS.

- [ ] **Step 6: Commit**

~~~powershell
git add python/ml_lab/draw_classification.py python/tests/test_draw_classification.py
git commit -m "feat: classify tactical draw evidence"
~~~

---

### Task 6: Reciprocal evaluator evidence integration

**Files:**
- Modify: python/ml_lab/evaluation.py
- Modify: python/ml_lab/protocol.py
- Modify: python/tests/test_evaluation.py
- Modify: python/tests/test_protocol.py

**Interfaces:**
- DuelClient adds enable_trace, drain_trace, and save_replay.
- _play_game returns PlayedGame instead of an integer.
- evaluate_matchup adds evidence_dir: Path | None and capture_trace: bool.
- evaluate_controllers adds environment: str | None so scripted-only tactical-v2 matchups are explicit.
- Omitting both arguments preserves the current evaluation result schema.

- [ ] **Step 1: Extend FakeDuelClient and write failing tests**

Assert trace enable precedes reset, every draw retains trace and replay, the first win/loss in each candidate-seat stratum is retained as a control, worker completion does not affect selected controls, and an ordinary evaluation performs no trace RPC.

~~~python
def test_evaluation_writes_draw_evidence_and_controls(
    tmp_path: Path, contract: EnvironmentContract
) -> None:
    result = evaluate_matchup(
        candidate=_model_controller(tmp_path, contract, "candidate", 64),
        opponent=_model_controller(tmp_path, contract, "opponent", 96),
        games=2,
        seed_start=80_000,
        both_seats=True,
        workers=1,
        client_factory=trace_client_factory(outcomes=(0, -1, 1, -1)),
        evidence_dir=tmp_path / "evidence",
        capture_trace=True,
    )

    assert result["evidence"]["draw_traces"] == 2
    assert result["evidence"]["control_traces"] == 2
~~~

- [ ] **Step 2: Confirm evaluator failures**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_evaluation.py -q
~~~

Expected: FAIL because the new interfaces do not exist.

- [ ] **Step 3: Implement strict DuelClient trace methods**

~~~python
def enable_trace(self, enabled: bool) -> None:
    response = self._rpc({"cmd": "duel_trace_enable", "enabled": enabled})
    if response != {"enabled": enabled}:
        raise ValueError("duel trace enable response is invalid")

def drain_trace(self) -> EpisodeTrace:
    return EpisodeTrace.from_payload(self._rpc({"cmd": "duel_trace_drain"}))

def save_replay(self, path: Path) -> Path:
    response = self._rpc({"cmd": "duel_save", "path": str(path)})
    saved = Path(str(response.get("saved", "")))
    if saved != path:
        raise ValueError("duel save response path does not match request")
    return saved
~~~

Extend protocol.py only for these RPC responses. Do not alter validate_step_payload.

- [ ] **Step 4: Return a structured PlayedGame**

~~~python
@dataclass(frozen=True)
class PlayedGame:
    winner: int
    terminated: bool
    truncated: bool
    trace: EpisodeTrace | None
~~~

When capture is requested, enable before reset and drain after termination/truncation. A requested empty trace raises RuntimeError.

Add environment: str | None = None to evaluate_controllers. Validate it against SUPPORTED_ENVIRONMENTS. If a
resolved model controller has a contract whose version differs from the explicit environment, raise ValueError.
With no override, retain the current inference from controller contracts and tactical-v1 fallback:

selected_environment = environment or inferred_environment

If capture_trace is true and selected_environment is not tactical-v2, raise ValueError before starting a server.

- [ ] **Step 5: Write deterministic artifacts**

Use schedule index in filenames:

~~~python
stem = f"match-{index:06d}-seed-{seed}-candidate-seat-{candidate_seat}"
trace_path = evidence_dir / "traces" / f"{stem}.json"
replay_path = evidence_dir / "replays" / f"{stem}.replay"
~~~

Use atomic_write_json. Save every draw and the first win/loss in each candidate-seat stratum after schedule-order merge. Match records include terminated, truncated, summary, classification, trace_path, and replay_path only when evidence is enabled. W/L/D and Wilson arithmetic remain unchanged.

- [ ] **Step 6: Verify focused Python tests**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_evaluation.py python\tests\test_protocol.py python\tests\test_tactical_trace.py python\tests\test_draw_classification.py -q
~~~

Expected: PASS with deterministic worker ordering.

- [ ] **Step 7: Commit**

~~~powershell
git add python/ml_lab/evaluation.py python/ml_lab/protocol.py python/tests/test_evaluation.py python/tests/test_protocol.py
git commit -m "feat: preserve tactical evaluation evidence"
~~~

---

### Task 7: CLI, documentation, and game/research parity gate

**Files:**
- Modify: python/ml_lab/cli.py
- Modify: python/tests/test_cli.py
- Modify: python/README.md
- Modify: engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs

**Interfaces:**
- evaluate adds --capture-trace and --evidence-dir PATH.
- evaluate adds --environment with the existing supported environment names.
- --evidence-dir implies --capture-trace.
- Existing evaluate commands remain valid.

- [ ] **Step 1: Write failing CLI forwarding tests**

~~~python
def test_evidence_directory_enables_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_evaluate_controllers(*args, **kwargs):
        captured.update(kwargs)
        return {"wins": 0, "losses": 0, "draws": 1, "games": 1}

    monkeypatch.setattr("ml_lab.cli.evaluate_controllers", fake_evaluate_controllers)
    main([
        "evaluate", "--p0", "greedy", "--p1", "random", "--games", "1",
        "--environment", "tactical-v2",
        "--evidence-dir", str(tmp_path / "evidence"),
    ])

    assert captured["capture_trace"] is True
    assert captured["evidence_dir"] == tmp_path / "evidence"
    assert captured["environment"] == "tactical-v2"
~~~

- [ ] **Step 2: Confirm parser failure**

~~~powershell
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_cli.py -q
~~~

Expected: FAIL because the arguments are unknown.

- [ ] **Step 3: Add arguments and forwarding**

~~~python
evaluate.add_argument(
    "--capture-trace",
    action="store_true",
    help="capture evaluation-only tactical transition evidence",
)
evaluate.add_argument(
    "--evidence-dir",
    type=Path,
    help="write per-match traces and replays; implies --capture-trace",
)
evaluate.add_argument(
    "--environment",
    choices=["tactical-v1", "tactical-v2", "adaptive-v1"],
    help="explicit environment; required to select tactical-v2 for scripted-only matchups",
)
~~~

Forward environment=args.environment,
capture_trace=args.capture_trace or args.evidence_dir is not None, and
evidence_dir=args.evidence_dir.

- [ ] **Step 4: Add the real parity integration test**

The fixed-seed test must enable trace, run tactical-v2 greedy versus random, drain trace, save replay, reconstruct with ReplayFile.Read and Replay, and compare final winner/state. Repeat with trace disabled and assert identical commands and final state. This proves telemetry is passive.

- [ ] **Step 5: Document command and artifacts**

Add this single-line Windows example:

~~~powershell
.\python\winenv\Scripts\python.exe -m ml_lab.cli evaluate --p0 run:python\runs\candidate --p1 greedy --games 500 --both-seats --workers 4 --environment tactical-v2 --evidence-dir python\evidence\candidate-vs-greedy
~~~

Document evaluation.json, traces, replays, category totals, paired seeds, and diagnostic quarantine.

- [ ] **Step 6: Run final verification**

~~~powershell
dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
.\python\winenv\Scripts\python.exe -m pytest python\tests\test_tactical_trace.py python\tests\test_draw_classification.py python\tests\test_evaluation.py python\tests\test_protocol.py python\tests\test_cli.py -q
.\python\winenv\Scripts\python.exe -m ml_lab.cli evaluate --p0 greedy --p1 random --games 2 --both-seats --workers 1 --environment tactical-v2 --evidence-dir python\evidence\smoke
~~~

Expected: four reciprocal games; unchanged W/L/D arithmetic; every draw has trace/replay; controls follow the deterministic rule; traces chain; replays reconstruct; and a repeated run differs only in generated_at.

- [ ] **Step 7: Commit**

~~~powershell
git add python/ml_lab/cli.py python/tests/test_cli.py python/README.md engine/HexWars.Engine.Tests/AdaptiveDuelEnvTests.cs
git commit -m "feat: expose tactical baseline evidence capture"
~~~

---

## Exit Gate

- Capture remains disabled by default.
- Reset, step, observation, mask, and reward schemas are unchanged.
- Trace RPCs exist only on tactical-v2 duel evaluation.
- Malformed or discontinuous traces are rejected.
- Supported classifications include recorded evidence; ambiguity remains unclassified.
- Retained replays reconstruct reported outcomes.
- Trace-enabled and trace-disabled fixed-seed games are identical.
- Full engine and focused Python suites pass.

The next independent plan uses this evidence layer for the tournament matrix, hunter/evader controllers, historical checkpoints, aggregate reports, and counterfactual endgame forks. Tactical-v3 implementation starts only after that baseline is reproducible.
