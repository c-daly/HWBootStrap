# Task 11 Sealed-Engine Preflight and Operator CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a production-only GymServer evidence session that publishes physically authenticated Task 8 oracle-preflight evidence, then expose the complete selective-DAgger CLI, logging, reuse, and recovery safeguards required before Task 12 smoke verification.

**Architecture:** Python owns the frozen experiment, output transaction, and fresh GymServer process; GymServer owns repeated bounded-search execution, codec validation, trace/replay/benchmark bytes, and an ordered receipt chain. A new Task 8 schema-3 opener reconstructs the complete publication and is the only path that returns `evidence_class="sealed-engine"`; the existing callback-owned schema 2 remains permanently untrusted.

**Tech Stack:** Python 3.14, pytest, .NET 8, C# `netstandard2.1` engine code, GymServer JSONL protocol, `System.Text.Json`, SHA-256, immutable JSON publications, argparse, structured Python logging.

## Global Constraints

- Follow `docs/superpowers/specs/2026-08-09-task11-sealed-engine-preflight-design.md` exactly.
- Defend against pipeline substitution and callback/test-fixture promotion, not a malicious machine owner; do not add signing keys, remote services, or hardware attestation.
- The production preflight CLI must construct the concrete GymServer boundary internally. It may not accept evaluator, benchmark, codec, client, trust, seed, schedule, threshold, oracle, or final-bank overrides.
- Schema-2 evidence remains `mode="private-test-transcript"`, `evidence_class="untrusted-test-transcript"`, and `engine_authenticated=false`; no conversion or relabeling API is allowed.
- Schema-3 evidence is `mode="owned-gymserver-session"`, `evidence_class="sealed-engine"`, and `engine_authenticated=true` only after physical close, publication, and reopen.
- Preserve learner-action ownership of every environment transition. Repeated teacher queries are counterfactual and behaviorally passive.
- Do not change game rules, bounded-search behavior, rewards, observations, actions, DAgger eligibility, mixture ratios, optimizer settings, panel thresholds, or seed banks.
- Do not run the 480-game production preflight or any production DAgger training in this plan.
- Task 12 retains ownership of the exact four-collection/four-evaluation physical smoke, all-repository verification, and final branch acceptance.
- Use the active Python 3.14 environment. Record Python 3.11 compatibility as deferred; do not spend this plan on the older runtime.
- Keep generated evidence outside the repository and preserve exact containment, reparse-point rejection, atomic staging, diagnostics, and immutable reuse semantics.
- After every C# edit, run the affected .NET tests. Before completion, verify Unity health with Coplay, run `check_compile_errors`, and inspect Unity logs if compilation or runtime integration reports a failure.
- Never add attribution trailers to commits or pull-request text.

## File Responsibility Map

| File | Responsibility |
|---|---|
| `engine/HexWars.Engine/Rl/TacticalV2OraclePreflight.cs` | Behaviorally passive double-query oracle wrapper and immutable benchmark evidence DTOs. |
| `engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs` | Engine invariants plus real GymServer JSONL session tests. |
| `engine/HexWars.GymServer/OracleEvidenceSession.cs` | Exact session lifecycle, expected schedule, artifact descriptors, receipt sequencing, and hash chaining. |
| `engine/HexWars.GymServer/Program.cs` | Thin JSONL command parsing and wiring to the session component and tactical-v2 duel. |
| `python/ml_lab/evaluation.py` | Concrete `EngineEvidenceDuelClient`; exact protocol response parsing and artifact-byte decoding. |
| `python/tests/test_evaluation.py` | Client request/response, lifecycle, and fail-closed protocol tests. |
| `python/ml_lab/dagger.py` | Task 8 schema-3 types, physical opener, transactional producer, reuse, and diagnostics. |
| `python/tests/test_dagger.py` | Domain schema, canonical hash vectors, and physical mutation tests. |
| `python/run_annihilation_selective_dagger.py` | First-party adapter construction, Task 10 handoff, CLI, dispatch, logging, and operator guards. |
| `python/tests/test_annihilation_selective_dagger.py` | End-to-end stage, CLI, logging, reuse, recovery, and real-process integration tests. |
| `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md` | Exact commands, artifacts, trust meaning, recovery, smoke, and production-run instructions. |
| `docs/superpowers/plans/2026-08-03-selective-dagger-search-distillation.md` | Backlink marking this plan as the authoritative expanded Task 11 procedure. |

---

### Task 1: Add behaviorally passive repeated-oracle evidence in the engine

**Files:**

- Create: `engine/HexWars.Engine/Rl/TacticalV2OraclePreflight.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs`
- Reference: `engine/HexWars.Engine/Rl/TacticalV2Dagger.cs:19-110,251-360`
- Reference: `engine/HexWars.Engine/Rl/TacticalV2DecisionObserver.cs`

**Interfaces:**

- Consumes: `IActionOracle.Decide(TacticalV2DecisionContext)`, `TacticalV2OracleDecision`, and the existing `SelectiveDaggerObserver` eligibility boundary.
- Produces: `OraclePreflightActionOracle : IActionOracle`, `OraclePreflightBenchmarkRecord`, `IOraclePreflightBenchmarkSink`, and `BufferedOraclePreflightBenchmarkSink`.
- `OraclePreflightActionOracle.Decide` returns the first of two identical validated decisions, records both, and never calls `GameEngine.Apply` on the live state.

- [ ] **Step 1: Write failing immutable evidence and double-query tests**

Add tests named:

```csharp
[Test]
public void OraclePreflightActionOracle_QueriesTwiceAndReturnsFirstWithoutMutatingState()

[Test]
public void OraclePreflightActionOracle_RejectsDifferentRepeatedDecisions()

[Test]
public void OraclePreflightActionOracle_RejectsContextMutationBetweenQueries()

[Test]
public void OraclePreflightBenchmarkRecord_DefensivelyCopiesObservationMaskStateAndCommands()
```

Use a counting `IActionOracle`, a deterministic timestamp sequence, and the existing tactical-v2 fixture. Snapshot `TacticalEvaluationTrace.ProjectState(context.State)` before and after `Decide`; assert equality and assert the learner action remains the action later applied by `TacticalV2DuelEnv.Step`.

- [ ] **Step 2: Run the focused engine tests and verify RED**

Run:

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~OraclePreflight"
```

Expected: compile failure because the four new preflight evidence types do not exist. A fixture or build failure is not an acceptable RED.

- [ ] **Step 3: Implement the immutable evidence types**

Create these signatures:

```csharp
public interface IOraclePreflightBenchmarkSink
{
    void Reset();
    void Accepted(OraclePreflightBenchmarkRecord record);
}

public sealed class BufferedOraclePreflightBenchmarkSink : IOraclePreflightBenchmarkSink
{
    public void Reset();
    public void Accepted(OraclePreflightBenchmarkRecord record);
    public IReadOnlyList<OraclePreflightBenchmarkRecord> Drain();
}

public sealed class OraclePreflightBenchmarkRecord
{
    public string StateHash { get; }
    public int DecisionIndex { get; }
    public float[] Observation { get; }
    public bool[] LegalMask { get; }
    public TacticalTraceState State { get; }
    public TacticalV2OracleDecision First { get; }
    public TacticalV2OracleDecision Second { get; }
    public long FirstElapsedTicks { get; }
    public long SecondElapsedTicks { get; }
    public long ClockFrequency { get; }
}
```

Constructors must reject nulls, invalid counts/ticks/frequency, nonidentical decision semantics, masked actions, failed `TryEncode`/`Decode` equality, failed authoritative legality on cloned state, and a changed canonical state hash. All arrays, commands, and projected states must be defensively copied.

- [ ] **Step 4: Implement the repeated-oracle wrapper**

Use this public shape:

```csharp
public sealed class OraclePreflightActionOracle : IActionOracle
{
    public OraclePreflightActionOracle(
        IActionOracle inner,
        IOraclePreflightBenchmarkSink sink,
        Func<long> timestamp,
        long clockFrequency);

    public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context);
}
```

`Decide` must hash the complete projected state and frozen observation/mask before, between, and after the two inner calls; time each call separately; compare action, command, depth, budget, heuristic, and actual expansions; revalidate both decisions against the same snapshot; emit one immutable record; and return a defensive copy of the first decision. Production wiring later supplies `Stopwatch.GetTimestamp` and `Stopwatch.Frequency`.

- [ ] **Step 5: Run focused and existing DAgger engine tests**

Run:

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~OraclePreflight|FullyQualifiedName~TacticalV2DaggerTests"
```

Expected: all selected tests pass; existing selective-DAgger eligibility and learner-transition tests remain unchanged.

- [ ] **Step 6: Commit Task 1**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2OraclePreflight.cs engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs
git commit -m "feat: add authoritative oracle preflight evidence"
```

---

### Task 2: Add the GymServer evidence-session protocol and receipt chain

**Files:**

- Create: `engine/HexWars.GymServer/OracleEvidenceSession.cs`
- Modify: `engine/HexWars.GymServer/Program.cs:65-110,327-568`
- Modify: `engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs`
- Reference: `engine/HexWars.GymServer/ScenarioJson.cs`

**Interfaces:**

- Consumes: `OraclePreflightActionOracle`, `BufferedOraclePreflightBenchmarkSink`, `BufferedTacticalV2DaggerSink`, `BufferedDuelTransitionSink`, `TacticalV2DuelEnv`, and the expanded candidates-by-schedule request.
- Produces JSONL commands `duel_evidence_begin`, `duel_evidence_game_close`, and `duel_evidence_end`.
- Produces `OracleEvidenceSession`, which accepts only exact validated data from `Program.cs` and returns immutable response DTOs.

- [ ] **Step 1: Write failing real-process protocol tests**

Extend the existing GymServer process harness with tests named:

```csharp
[Test]
public void GymServer_EvidenceSessionEchoesNonceAndRejectsSecondBegin()

[Test]
public void GymServer_EvidenceSessionReturnsOrderedReceiptBoundToArtifacts()

[Test]
public void GymServer_EvidenceSessionRejectsPrematureEndWrongScheduleAndPostCloseCommands()

[Test]
public void GymServer_EvidenceSessionKeepsLearnerActionAsOnlyAppliedTransition()
```

The begin request fixture must contain the exact schema-1 fields from the design and an expanded ordered schedule entry for candidate 0/game 0. Assert the receipt contains sequence 1, the begin chain as `previous_receipt_sha256`, and correct trace/replay/benchmark descriptors.

- [ ] **Step 2: Build GymServer and verify the protocol tests fail**

Run:

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~GymServer_EvidenceSession"
```

Expected: GymServer replies with its unknown-command error for `duel_evidence_begin`.

- [ ] **Step 3: Implement canonical session DTOs and golden hashing**

Create `OracleEvidenceSession.cs` with these responsibilities and shapes:

```csharp
internal sealed class OracleEvidenceSession
{
    internal static OracleEvidenceBeginResponse Begin(
        OracleEvidenceBeginRequest request,
        OracleEvidenceRuntimeIdentity runtime);

    internal OracleEvidenceGameResponse CloseGame(
        OracleEvidenceGameContext context,
        byte[] traceUtf8,
        byte[] replayUtf8,
        byte[] benchmarkUtf8);

    internal OracleEvidenceEndResponse End();
}
```

Use an explicit `Utf8JsonWriter` field order for begin, receipt, and end hash bodies. Add a golden-vector process test that pins the SHA-256 for one literal begin body and one literal receipt body; Python will use the same vectors in Task 3. Reject unknown fields, non-lowercase 64-hex identities, non-64-hex nonce, duplicate candidate/game entries, noncanonical schedule order, invalid seats, and any schedule count other than `len(candidates) * len(preflight_schedule)`.

- [ ] **Step 4: Wire exact JSONL commands into Program.cs**

Add thin cases with exact field validation:

```csharp
case "duel_evidence_begin":
    // Parse exact request, verify tactical-v2 and physical identities,
    // freeze expanded schedule, install trace + DAgger + repeated-oracle sinks.
    break;

case "duel_evidence_game_close":
    // Require terminal expected game, serialize exact UTF-8 artifacts,
    // close through OracleEvidenceSession, drain only owned buffers.
    break;

case "duel_evidence_end":
    // Require complete schedule and no open game, freeze chain.
    break;
```

The game-close response carries each exact artifact as bounded base64 UTF-8 bytes plus SHA-256 and byte size. The receipt hashes those exact decoded bytes, not a Python reconstruction. Evidence mode forces trace and benchmark capture and does not permit ordinary drain commands to remove owned buffers before game close.

- [ ] **Step 5: Add failure and size-bound tests**

Test wrong nonce, mismatched contract/encoding/scenario/oracle, duplicated or skipped schedule entries, nonterminal close, second close, ordinary drain during evidence mode, oversized schedule, oversized artifact, and engine exit without end. Assert every failure leaves no successful end acknowledgement.

- [ ] **Step 6: Run GymServer and DAgger engine suites**

Run:

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~EvidenceSession|FullyQualifiedName~OraclePreflight|FullyQualifiedName~TacticalV2DaggerTests"
```

Expected: all selected tests pass.

- [ ] **Step 7: Commit Task 2**

```powershell
git add engine/HexWars.GymServer/OracleEvidenceSession.cs engine/HexWars.GymServer/Program.cs engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs
git commit -m "feat: seal GymServer preflight sessions"
```

---

### Task 3: Add the concrete Python engine-evidence client

**Files:**

- Modify: `python/ml_lab/evaluation.py:361-535`
- Modify: `python/tests/test_evaluation.py`
- Reference: `python/ml_lab/contracts.py`

**Interfaces:**

- Consumes: the exact Task 2 JSONL protocol.
- Produces immutable `EngineEvidenceArtifact`, `EngineEvidenceGame`, and `EngineEvidenceClosure` dataclasses plus `EngineEvidenceDuelClient`.
- `EngineEvidenceDuelClient` owns the subprocess through `DuelClient`; no caller supplies response DTOs or a trust flag.

- [ ] **Step 1: Write failing client lifecycle and parser tests**

Add these exact tests:

- `test_engine_evidence_client_sends_exact_begin_and_validates_ack` captures
  the request passed to `_rpc`, exact-compares every begin field, returns a
  literal valid acknowledgement, and asserts the client freezes it.
- `test_engine_evidence_client_decodes_exact_artifact_bytes_and_receipt`
  base64-encodes three literal byte strings, supplies their independent hashes
  and sizes, and asserts the returned dataclass owns exactly those bytes.
- `test_engine_evidence_client_rejects_wrong_nonce_sequence_hash_and_unknown_fields`
  parametrizes the four independent mutations and expects `ValueError`.
- `test_engine_evidence_client_requires_close_before_success` closes the
  subprocess without `duel_evidence_end` and asserts no closure exists.
- `test_engine_evidence_hash_vectors_match_gymserver_golden_values` hashes the
  literal Task 2 UTF-8 vectors and asserts their pinned digests.

Patch only `_rpc`; never patch the preflight publisher. Use the literal golden begin/receipt bodies from Task 2 and independently compute Python SHA-256 values.

- [ ] **Step 2: Run focused Python tests and verify RED**

Run from the worktree:

```powershell
$env:VIRTUAL_ENV='C:\Users\cddal\HexWars\python\winenv'
$env:UV_CACHE_DIR='C:\Users\cddal\HexWars\.uv-cache'
$env:PYTHONDONTWRITEBYTECODE='1'
uv run --active --no-project python -m pytest -q python/tests/test_evaluation.py -k "engine_evidence"
```

Expected: import or attribute failure because `EngineEvidenceDuelClient` does not exist.

- [ ] **Step 3: Implement strict immutable response types**

Add exact dataclasses:

```python
@dataclass(frozen=True)
class EngineEvidenceArtifact:
    payload: bytes
    sha256: str
    byte_size: int

@dataclass(frozen=True)
class EngineEvidenceGame:
    receipt: Mapping[str, Any]
    receipt_utf8: bytes
    trace: EngineEvidenceArtifact
    replay: EngineEvidenceArtifact
    benchmark: EngineEvidenceArtifact

@dataclass(frozen=True)
class EngineEvidenceClosure:
    begin: Mapping[str, Any]
    games: tuple[EngineEvidenceGame, ...]
    end: Mapping[str, Any]
```

Validate exact keys and JSON scalar types, strict lower-case hashes, bounded base64 decoding, byte sizes, receipt artifact descriptors, nonce/session/sequence/previous-hash continuity, and closing count/final hash.

- [ ] **Step 4: Implement EngineEvidenceDuelClient**

Use this shape:

Implement these exact methods on `EngineEvidenceDuelClient(DuelClient)`:

- `begin_evidence(request: Mapping[str, Any]) -> Mapping[str, Any]`
- `close_evidence_game() -> EngineEvidenceGame`
- `end_evidence() -> EngineEvidenceClosure`
- `__enter__() -> EngineEvidenceDuelClient`
- `__exit__(exc_type, exc, traceback) -> None`

The constructor remains `DuelClient(server_cmd, environment="tactical-v2")`. `begin_evidence` stores the exact frozen request and acknowledgement. `close_evidence_game` advances one expected expanded-schedule item. `end_evidence` returns a closure only after all expected games and a valid final acknowledgement. `close()` without `end_evidence()` must never synthesize a closure.

- [ ] **Step 5: Run focused and full evaluation-client tests**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_evaluation.py -k "engine_evidence or DuelClient"
uv run --active --no-project python -m pytest -q python/tests/test_evaluation.py
```

Expected: all tests pass.

- [ ] **Step 6: Commit Task 3**

```powershell
git add python/ml_lab/evaluation.py python/tests/test_evaluation.py
git commit -m "feat: add engine evidence client"
```

---

### Task 4: Add the strict Task 8 schema-3 physical opener

**Files:**

- Modify: `python/ml_lab/dagger.py:1611-1736,2115-2830`
- Modify: `python/tests/test_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py:5900-5970`
- Reference: `docs/superpowers/specs/2026-08-09-task11-sealed-engine-preflight-design.md`

**Interfaces:**

- Consumes: independently authored schema-3 fixture bytes and the Task 2/3 golden receipt format.
- Produces: `_open_oracle_preflight_v3(root, *, expected_identity, definition) -> OracleSpec` and dispatching `open_oracle_preflight_publication(root, *, definition, repository_identity_provider) -> OraclePreflightPublication`.
- Extends `OraclePreflightPublication` with `engine_session_id: str | None`,
  `engine_receipt_count: int`, and
  `engine_final_chain_sha256: str | None`. Schema 2 returns
  `None, 0, None`; schema 3 requires populated exact values.
- Preserves `_open_oracle_preflight_v2` and its untrusted return class unchanged.

- [ ] **Step 1: Write an independent minimal schema-3 fixture**

In tests, create literal begin, one-game receipt, end, trace, replay, benchmark, session, and manifest bytes without calling a production writer or production content-identity helper. Use a one-game test definition only at the private opener boundary; production panel validation remains frozen at 480 games.

- [ ] **Step 2: Write failing happy-path and trust-separation tests**

Add `test_open_oracle_preflight_v3_returns_only_physical_sealed_engine_evidence`,
`test_schema2_transcript_cannot_be_relabelled_as_schema3`, and
`test_public_opener_dispatches_schema3_and_preserves_schema2_untrusted_class`.
The first exact-compares the independently written fixture with the returned
physical identities; the second changes every visible trust string in schema 2
and still expects rejection; the third passes each physical schema to the
public dispatcher and asserts the two distinct evidence classes.

The happy path must assert exact selected oracle, physical root, overall content identity, engine session ID/root hash, and `evidence_class == "sealed-engine"`.

- [ ] **Step 3: Run the three tests and verify RED**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "preflight_v3 or schema3 or relabelled"
```

Expected: schema-3 rejection because only the schema-2 opener exists.

- [ ] **Step 4: Implement strict schema-3 parsing and inventory**

Add exact field sets for:

```python
_ENGINE_TRUST_V1_FIELDS
_PREFLIGHT_V3_MANIFEST_FIELDS
_ENGINE_SESSION_V1_FIELDS
_ENGINE_RECEIPT_V1_FIELDS
_ENGINE_ARTIFACT_DESCRIPTOR_FIELDS
```

Require manifest schema 3, completed status, exact recursive inventory, canonical contained POSIX paths, no reparses/symlinks, bounded file counts/sizes, exact descriptor bytes/hashes, exact begin/end fields, and no schema-2 trust keys such as `task_9_production_seal_required` in the schema-3 trust object.

- [ ] **Step 5: Implement physical reconstruction**

For every expanded schedule item, reopen its receipt and artifact bytes, compare nonce/session/sequence/previous hash, verify the canonical receipt hash, validate the trace/replay/benchmark semantics with the existing Task 8 validators, and recompute W/L/D, profile/seat coverage, diagnostics, deterministic pairs, codec failures, expansion metrics, throughput, eligibility, tie breaks, and selected oracle. End with a full inventory plus byte-for-byte final snapshot comparison.

- [ ] **Step 6: Add adversarial physical mutation tests**

Independently mutate and, where meaningful, reseal the outer manifest for:

- missing end acknowledgement;
- wrong nonce/session ID;
- duplicate, skipped, or reordered receipt;
- wrong previous hash or closing final hash;
- receipt copied from another session;
- trace/replay/benchmark byte mutation;
- receipt context changed to another seed/profile/seat/oracle;
- scenario/contract/encoding/repository/oracle-source drift;
- extra file, missing file, absolute path, `..`, reparse root, and nested Windows drive/ADS/reserved path;
- mutation during the final receipt or artifact reread.

Every case must raise `ValueError` before returning `OraclePreflightPublication`.

- [ ] **Step 7: Run focused Task 8 and DAgger suites**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "preflight or oracle or schema3 or sealed_engine"
```

Expected: all selected tests pass; schema-2 tests still assert untrusted evidence.

- [ ] **Step 8: Commit Task 4**

```powershell
git add python/ml_lab/dagger.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: authenticate sealed preflight publications"
```

---

### Task 5: Implement transactional production preflight, diagnostics, and reuse

**Files:**

- Modify: `python/ml_lab/dagger.py:2832-3620`
- Modify: `python/tests/test_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Reference: `python/ml_lab/evaluation.py`

**Interfaces:**

- Consumes: `PanelDefinition`, a repository-pinned `server_cmd`, frozen repository/dataset identity, and the existing lease/diagnostic helpers.
- Produces: `run_oracle_preflight(definition, *, output_root, server_cmd) -> OraclePreflightRun`.
- Produces private
  `_run_engine_preflight_game(session, *, learner, oracle, scheduled_duel) -> Mapping[str, Any]`
  and
  `_write_engine_preflight_game(staging, *, expected_context, learner_result, engine_game) -> Mapping[str, Any]`
  helpers so the game loop names below are defined in this task.
- `run_oracle_preflight` checks for exact physical reuse before constructing `EngineEvidenceDuelClient`; new publication constructs the concrete client lazily and requires a valid `EngineEvidenceClosure`.

- [ ] **Step 1: Write failing transactional production tests**

Add `test_run_oracle_preflight_publishes_schema3_only_after_engine_close`,
`test_run_oracle_preflight_reuse_starts_no_engine_process`,
`test_run_oracle_preflight_interruption_seals_diagnostics_not_destination`,
`test_run_oracle_preflight_constructs_only_concrete_engine_client`, and
`test_run_oracle_preflight_rolls_back_post_rename_physical_drift`.

Patch `evaluation.EngineEvidenceDuelClient` with a constructor spy whose
returned concrete instance has only `_rpc` patched at the protocol transport
seam. On reuse, make the constructor itself call `pytest.fail`. Do not patch
the publisher, physical opener, metrics, inventory, trust, or publication DTO.

- [ ] **Step 2: Run the transactional selector and verify RED**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "schema3 and (transaction or reuse or interruption or rollback or concrete)"
```

Expected: the current public function raises the Task 9/production-factory placeholder error.

- [ ] **Step 3: Implement the expanded frozen session request**

Build one canonical request from the validated definition containing candidates crossed with the 240-game schedule in candidate-major order, the complete schedule hash, panel/repository/scenario/contract/encoding/oracle identities, and a nonce from `secrets.token_hex(32)`. Require the repository to be clean before begin and again before publication.

Add the explicit run result:

```python
@dataclass(frozen=True)
class OraclePreflightRun:
    selected_oracle: OracleSpec
    publication: OraclePreflightPublication
    new_games: int
    new_engine_sessions: int
    reused: bool
```

An exact reuse returns `new_games=0`, `new_engine_sessions=0`, and
`reused=True` before the concrete client constructor is reached.

- [ ] **Step 4: Implement the production game loop**

For each expanded item:

```python
session.configure_dagger(
    enabled=True,
    depth=oracle.depth,
    expansion_budget=oracle.expansion_budget,
    use_heuristic=True,
)
result = _run_engine_preflight_game(
    session,
    learner=learner,
    oracle=oracle,
    scheduled_duel=scheduled_duel,
)
game = session.close_evidence_game()
record = _write_engine_preflight_game(
    staging,
    expected_context=expected_context,
    learner_result=result,
    engine_game=game,
)
```

Use the existing deterministic learner and scheduled-game logic. Derive all teacher decisions, codec evidence, trace, replay, benchmark, expansions, and trust only from `game`. After the last game, call `end_evidence`; compare the returned closure to every staged receipt before constructing summaries.

- [ ] **Step 5: Implement schema-3 transaction, diagnostic, and reuse rules**

Reuse the existing destination/staging/diagnostics/lease naming. On success: write session and manifest last, open staging physically, verify repository/dataset again, rename atomically, reopen destination, and compare content identity. On any pre-rename failure: move staging into a bounded diagnostic root after removing any would-be completion manifest. On any post-rename failure: move the newly created destination back to staging. Never roll back or mutate a pre-existing reusable destination.

- [ ] **Step 6: Add exact-compute and recovery assertions**

Assert first publication reports exactly 480 new games and one completed engine session. Exact reuse reports `new_games=0`, `new_epochs=0`, and zero GymServer RPCs/process starts. An incomplete session rerun uses a fresh nonce and does not reuse any receipt from diagnostics.

- [ ] **Step 7: Run Task 8 production and private regressions**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "preflight or oracle or execution_session or diagnostic"
```

Expected: production schema-3 tests and private schema-2 tests all pass.

- [ ] **Step 8: Commit Task 5**

```powershell
git add python/ml_lab/dagger.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: publish engine sealed oracle preflight"
```

---

### Task 6: Replace the Task 11 placeholder and authenticate Task 10 end to end

**Files:**

- Modify: `python/run_annihilation_selective_dagger.py:733-761,3207-3249,4277-4305`
- Modify: `python/tests/test_annihilation_selective_dagger.py:4700-4760,5900-5970`
- Reference: `python/ml_lab/dagger.py`
- Reference: `python/ml_lab/evaluation.py`

**Interfaces:**

- Consumes: repository-pinned `server_command()`, scenario path, `EngineEvidenceDuelClient`, and the public schema-3 opener.
- Produces: `run_sealed_oracle_preflight(*, definition, output_root) -> OraclePreflightRun` with no factory or server-command parameter.
- Produces
  `_validated_external_output_root(output_root, repository_root) -> Path`,
  which resolves the output parent, rejects repository-contained roots and
  reparses, and returns the canonical not-yet-created destination.
- `_open_development_preflight_evidence` accepts only the physical public opener result and still requires `evidence_class == "sealed-engine"`.

- [ ] **Step 1: Write failing public-call-contract tests**

Assert:

```python
parameters = inspect.signature(runner.run_sealed_oracle_preflight).parameters
assert set(parameters) == {"definition", "output_root", "repository_root"}
assert "execution_session_factory" not in parameters
assert "server_cmd" not in parameters
```

Also inspect the CLI-facing call graph and fail the test if `_run_oracle_preflight_for_test`, an evaluator callback, or a caller-provided trust object is reachable.

- [ ] **Step 2: Write a real physical Task 8-to-Task 10 fixture test**

Create a valid schema-3 publication from independently authored engine-session fixture bytes, then call `build_development_evaluation_definition` without patching `_open_development_preflight_evidence`. Assert the returned definition freezes the selected oracle, session/content identity, starting learner, repository, scenario, and complete Task 9 stable identity.

- [ ] **Step 3: Run focused tests and verify RED**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_annihilation_selective_dagger.py -k "sealed_preflight_public_contract or schema3_task10"
```

Expected: the current runner still requires `execution_session_factory`, or Task 10 rejects all physical preflights.

- [ ] **Step 4: Construct the concrete client internally**

Implement:

```python
def run_sealed_oracle_preflight(
    *,
    definition: dagger_domain.PanelDefinition,
    output_root: Path,
    repository_root: Path = _REPOSITORY_ROOT,
) -> dagger_domain.OraclePreflightRun:
    return dagger_domain.run_oracle_preflight(
        definition,
        output_root=_validated_external_output_root(
            output_root, repository_root,
        ),
        server_cmd=server_command(),
    )
```

`server_command()` is repository-owned and patchable by tests, but it is not
selectable through CLI arguments or this public runner signature. The domain
function owns lazy client construction so exact reuse launches no process.

- [ ] **Step 5: Replace Task 10's unconditional fail-closed placeholder**

Have `_open_development_preflight_evidence` call only `dagger.open_oracle_preflight_publication`. Map its physical schema-3 fields into `DevelopmentPreflightEvidence`; reject schema 2, missing session identity, repository-contained evidence roots, wrong starting learner, or any class other than `sealed-engine`. Preserve the final physical reread in both aggregate passes.

- [ ] **Step 6: Add substitution and drift regressions**

Prove Task 10 rejects a callback DTO claiming sealed evidence, a schema-2 manifest with changed trust strings, a copied receipt from another session, and a Task 8 mutation between definition construction and aggregate pass two.

- [ ] **Step 7: Run Task 10's bounded provenance selector**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_annihilation_selective_dagger.py -k "task10 and (preflight or definition or aggregate or physical or provenance)"
```

Expected: all selected tests pass; only explicit Windows symlink/reparse privilege tests may skip.

- [ ] **Step 8: Commit Task 6**

```powershell
git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: authenticate Task 10 engine preflight"
```

---

### Task 7: Complete the production CLI, structured logging, and operator recovery

**Files:**

- Modify: `python/run_annihilation_selective_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Reference: `python/run_annihilation_checkpoint_audit.py:23-60,278-310,470-496`

**Interfaces:**

- Produces: `server_command() -> list[str]`, `build_parser() -> argparse.ArgumentParser`, `configure_logging(output_root) -> logging.Logger`, `dispatch(args, logger) -> object`, and `main() -> int`.
- Produces named `run_baseline` and `run_report` physical stage wrappers,
  plus a deliberate fail-closed `run_smoke` guard that names Task 12 as its
  unavailable implementation boundary.
- Exposes only `prepare`, `validate`, `preflight`, `baseline`, `iteration --index {1,2,3}`, `evaluate`, `aggregate`, `report`, `smoke`, and `all`.
- Every stage reopens physical prerequisites before compute and returns a structured result used for progress/reuse logging.

- [ ] **Step 1: Write failing exact-parser tests**

Assert every subcommand, required `--output-root`, iteration choices, and rejection of unknown arguments. Assert no parser option can override panel path, seeds, schedules, thresholds, mixtures, oracle budgets, starting checkpoint, server command, trust class, or final-bank use.

- [ ] **Step 2: Write failing dispatch-order and dependency tests**

With stage functions patched, assert exact `all` order:

```text
prepare -> validate -> preflight -> baseline -> iteration 1 -> iteration 2
-> iteration 3 -> evaluate -> aggregate -> report
```

Make each stage fail once and assert no later callback runs. Individual commands must reopen and name a missing/incompatible prerequisite before any game, inference, or epoch callback.

- [ ] **Step 3: Write failing logging and interrupt-recovery tests**

Capture stdout and `<output-root>/selective-dagger.log`. Assert flushed events contain command, stage, candidate/iteration, games, labels, disagreements, expansions, throughput, elapsed, ETA, reuse counts, session ID/receipt sequence for preflight, and failure diagnostic path. Simulate collection, training, evaluation, and GymServer interruption; assert completed prerequisites reuse, partial stages remain inspectable, and no completed marker is published.

- [ ] **Step 4: Run CLI tests and verify RED**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_annihilation_selective_dagger.py -k "parser or dispatch or logging or operator or interrupt or dependency_guard"
```

Expected: parser/dispatch attributes are absent.

- [ ] **Step 5: Implement parser and repository-pinned server command**

Follow the checkpoint-audit runner's small structure. Use the exact
repository-owned command:

```python
_SCENARIO_PATH = (
    _REPOSITORY_ROOT / "python" / "config" /
    "annihilation-imitation-v1.json"
)

def server_command() -> list[str]:
    return [
        "dotnet",
        str(
            _REPOSITORY_ROOT / "engine" / "HexWars.GymServer" /
            "bin" / "Debug" / "net8.0" / "HexWars.GymServer.dll"
        ),
        "--scenario-file",
        str(_SCENARIO_PATH),
    ]
```

The parser must not accept an arbitrary executable or scenario. Build parser
subcommands exactly once and make `main()` configure logging only after
parsing the explicit output root.

- [ ] **Step 6: Implement dependency-aware dispatch**

Dispatch only to `run_prepare`, `run_validate`,
`run_sealed_oracle_preflight`, `run_baseline`, `run_iteration`,
`run_development_evaluation`, `publish_development_aggregate`,
`run_report`, and the Task 12-reserved `run_smoke` entry point.
`run_smoke` must raise a named Task 12 boundary error until Task 12 replaces
that guard with the exact physical smoke. `preflight` uses
`run_sealed_oracle_preflight`; `evaluate` evaluates baseline plus all three
actors on the locked schedule; `aggregate` and `report` perform read-only
reconstruction; `all` stops on first exception and returns nonzero. Create
`run_baseline` and `run_report` in this step around the existing physical
boundaries and test those exact names.

- [ ] **Step 7: Implement structured dual logging**

Use one logger named `hexwars.selective_dagger`, one stdout handler, and one UTF-8 file handler at `<output-root>/selective-dagger.log`. Set `propagate=False`, remove stale handlers on reconfiguration, format stable key-value events, and flush both handlers after each progress event. Do not treat log bytes as publication authority.

- [ ] **Step 8: Implement interrupt and reuse reporting**

Catch `KeyboardInterrupt` only at `main`; log the active stage and diagnostic/staging path, close the GymServer, and return exit code 130 without manufacturing completion. On exact reuse, log `new_games=0 new_epochs=0` and the reopened content identity.

- [ ] **Step 9: Run the complete runner test file**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_annihilation_selective_dagger.py
```

Expected: all tests pass; only explicit privilege-gated reparse tests may skip.

- [ ] **Step 10: Commit Task 7**

```powershell
git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: add selective DAgger operator CLI"
```

---

### Task 8: Prove the real process boundary, document operation, and verify Task 11

**Files:**

- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Modify: `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md`
- Modify: `docs/superpowers/plans/2026-08-03-selective-dagger-search-distillation.md`
- Verify: all files changed in Tasks 1-7

**Interfaces:**

- Consumes: the actual built GymServer, committed panel/scenario, schema-3 producer/opener, and CLI.
- Produces: one small deterministic real-process boundary regression and exact operator documentation.
- Does not execute Task 12 smoke or production seed banks.

- [ ] **Step 1: Write the failing real-process boundary test**

Add a test marked with the repository's existing real-GymServer convention.
Build/start the actual tactical-v2 GymServer, use a one-game nonproduction test
schedule, and run begin/reset/steps/game-close/end through the concrete client.
Pass the closure through the private schema-3 writer/opener core and assert:

```python
assert publication.evidence_class == "sealed-engine"
assert publication.engine_session_id == receipt["session_id"]
assert publication.engine_receipt_count == 1
assert publication.selected_oracle == expected_oracle
```

Run the same completed root again and assert no new process launch and byte-identical publication identity.

The one-game definition is accepted only by the private core used for boundary
testing. The public production dispatcher still validates the committed
480-game panel exactly; Task 12 owns its first small end-to-end public smoke.

- [ ] **Step 2: Run the real-process test and verify RED or first GREEN**

Run:

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj
uv run --active --no-project python -m pytest -q python/tests/test_annihilation_selective_dagger.py -k "real_engine_sealed_preflight" -s
```

Expected before final wiring: fail at the first missing/mismatched real protocol boundary. If Tasks 1-7 already make it pass, record that as first integration GREEN; do not weaken the test to manufacture a RED.

- [ ] **Step 3: Fix only real-boundary integration defects**

Correct protocol serialization, process cleanup, scenario paths, bounded sizes, or exact identity comparisons exposed by Step 2. Do not add a fake fallback, callback adapter, trust override, or test-only production branch.

- [ ] **Step 4: Update the panel protocol**

Document:

- integrity versus provenance and the non-hostile-machine trust model;
- schema 2 versus schema 3 and why relabeling is forbidden;
- exact CLI commands for prepare, validate, preflight, recovery, reuse, smoke, and all;
- artifact tree and evidence inspection commands;
- progress/log fields and diagnostic locations;
- CUDA required for production training and CPU used only by Task 12 smoke;
- Unity used to observe replays, not to perform training;
- the 480-game preflight and production DAgger run remain unauthorized until Task 12 and final review.

- [ ] **Step 5: Verify and preserve the original Task 11 backlink**

Confirm that the start of Task 11 in
`2026-08-03-selective-dagger-search-distillation.md` links to this
authoritative expanded procedure while preserving its command set and goals.
Correct any remaining stale statement that assigns the production seal to Task
9; Task 11 owns the seal.

- [ ] **Step 6: Run focused Python domain suites**

Run:

```powershell
uv run --active --no-project python -m pytest -q python/tests/test_evaluation.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
```

Expected: all tests pass; only explicit privilege-gated Windows reparse tests may skip.

- [ ] **Step 7: Run full engine tests and build GymServer**

Run:

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```

Expected: build and all engine tests pass.

- [ ] **Step 8: Verify Unity health and compilation**

Use Coplay `get_unity_editor_state` first. If connected, run `check_compile_errors` and require zero errors. If any error appears, read `get_unity_logs`, fix the cause, and recheck. No scene/default value changes are authorized, so do not save a scene.

- [ ] **Step 9: Run final static checks**

Run:

```powershell
git diff --check
git status --short
```

Inspect the diff for callback promotion, schema ambiguity, unbounded JSON/base64 reads, repository-contained evidence, test-only production branches, hidden seed/threshold overrides, and missing final rereads.

- [ ] **Step 10: Commit documentation and integration support**

```powershell
git add python/tests/test_annihilation_selective_dagger.py python/panels/annihilation-selective-dagger-v1/PROTOCOL.md docs/superpowers/plans/2026-08-03-selective-dagger-search-distillation.md
git commit -m "test: prove sealed preflight boundary"
```

- [ ] **Step 11: Request final Task 11 review**

Review the complete Task 11 range against the design, with explicit adversarial attention to:

1. any public callback/factory/DTO path capable of producing `sealed-engine`;
2. transcript freshness, complete schedule coverage, and receipt ordering;
3. artifact-byte, session, repository, scenario, contract, encoding, and oracle binding;
4. transaction rollback, incomplete-session diagnostics, and zero-process reuse;
5. Task 10 physical reopening on both aggregate passes; and
6. parser/logging/operator safeguards.

Do not begin Task 12 until this review is clean.

---

## Task 11 Completion Gate

Task 11 is complete only when all eight tasks are committed, the real-process boundary test passes, the complete selected Python and .NET suites pass, Unity compilation is clean when connected, the worktree contains no unintended generated files, and an independent final review finds no Critical or Important provenance or operator-safety defect.

The next separately authorized work is Task 12 exact physical smoke. Production oracle preflight and DAgger training remain out of scope until Task 12 and final branch review are accepted.
