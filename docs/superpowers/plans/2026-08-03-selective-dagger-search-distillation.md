# Selective DAgger Search Distillation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and smoke-test a restart-safe, three-iteration selective-DAgger experiment that distills deterministic bounded search into the audited seed-227 tactical-v2 actor only on learner-visited conversion, favorable, cycling, and wasted-EndTurn states.

**Architecture:** Add a passive, generic pre-action observer boundary to `TacticalV2DuelEnv`; implement selection and search behind `IActionOracle`; transport immutable evidence through the GymServer; store each train/validation collection as a hash-bound overlay; reuse the existing masked actor trainer through a generic source-mixture corpus; and orchestrate preflight, collection, training, paired evaluation, aggregation, and physical reuse from a focused panel runner. The learner action always advances the game, and neither selection nor search exists at inference time.

**Tech Stack:** C#/.NET 8 engine and GymServer, xUnit, Python 3.11+, NumPy, PyTorch, Stable-Baselines3/sb3-contrib MaskablePPO, pytest, `uv`, JSONL GymServer protocol, Unity 6/Coplay compile verification.

## Global Constraints

- Treat [the approved design](../specs/2026-08-03-selective-dagger-search-distillation-design.md) as authoritative. Do not change its seeds, thresholds, mixtures, learner checkpoint, teacher candidates, or decision rules while implementing.
- Preserve `tactical-v2`: its observation length, 1,288-action encoding, reward, transition order, and ordinary no-observer behavior must not change.
- The external learner action, never the oracle action, advances the environment. Search is research instrumentation and is absent in evaluation and publication.
- Keep the original imitation dataset immutable. DAgger training and validation collections are separate content-hashed overlays.
- Fail closed on illegal actions, codec mismatches, nondeterministic oracle decisions, seed overlap, partial reciprocal pairs, changed inputs, missing physical evidence, and dirty/replaced repository identities.
- Keep final seeds `17,000,000-17,000,249` untouched. Use only the namespaces frozen in the design and panel definition.
- Freeze these disjoint namespaces verbatim: iteration-train `18,000,000-18,099,999`, `18,100,000-18,199,999`, and `18,200,000-18,299,999`; oracle preflight `18,900,000-18,900,119`; smoke `18,990,000-18,990,009`; iteration-validation `19,000,000-19,009,999`, `19,010,000-19,019,999`, and `19,020,000-19,029,999`; reserved `19,030,000-19,099,999`; and development evaluation `20,000,000-20,000,099`.
- Use test-driven development. For every behavior change, first run the named focused test and observe the expected failure; after implementation, rerun it and the adjacent regression suite.
- After every C# edit, run the genuine .NET test command and Coplay `check_compile_errors`. Read Unity logs if a runtime check fails; do not infer runtime behavior from source alone.
- Publish generated stages by atomic rename only after reopening and physically validating every referenced file and SHA-256. Diagnostic staging data may remain after interruption but must never satisfy reuse.
- Do not commit generated datasets, checkpoints, traces, replays, stage directories, `aggregate.json`, `oracle-preflight.json`, or `REPORT.md`.
- Never add attribution trailers to commits or pull-request text.

## File Map

| Area | Files |
|---|---|
| Engine observer | `engine/HexWars.Engine/Rl/TacticalV2DecisionObserver.cs`, `TacticalV2DuelEnv.cs`, `TacticalV2UnitRegistry.cs` |
| Engine DAgger/search | `engine/HexWars.Engine/Rl/TacticalV2Dagger.cs`, `TacticalEvaluationTrace.cs`, `engine/HexWars.Engine/BoundedSearchAgent.cs` |
| Engine tests | `engine/HexWars.Engine.Tests/TacticalV2DecisionObserverTests.cs`, `TacticalV2DaggerTests.cs`, existing tactical-v2/search tests |
| Wire protocol | `engine/HexWars.GymServer/Program.cs`, `python/ml_lab/evaluation.py`, `python/tests/test_evaluation.py` |
| Data/training domain | `python/ml_lab/dagger.py`, `python/ml_lab/imitation.py`, `python/ml_lab/algorithms.py` |
| Python tests | `python/tests/test_dagger.py`, `test_imitation.py`, `test_algorithms.py` |
| Orchestration | `python/run_annihilation_selective_dagger.py`, `python/tests/test_annihilation_selective_dagger.py` |
| Frozen definitions | `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md`, `panel.json`, `seed-banks.json` |

---

### Task 1: Add an immutable, passive pre-action observer boundary

**Files:**

- Create: `engine/HexWars.Engine/Rl/TacticalV2DecisionObserver.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs:37-47,94-140`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs:15-106`
- Test: `engine/HexWars.Engine.Tests/TacticalV2DecisionObserverTests.cs`
- Regression: `engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs`

- [ ] **Step 1: Write failing registry-snapshot and observer-context tests**

Add tests proving that:

1. `TacticalV2UnitRegistry.Snapshot()` preserves every slot's unit/template pair and is unaffected when the live registry later releases or registers a unit.
2. `ITacticalV2DecisionObserver.Reset` receives the selected profile, reference seat, learner seat, and initial state before the first externally supplied decision.
3. `Observe` receives the exact pre-action state, observation, legal mask, integer action, decoded command, active seat, decision index, and independent registry snapshots.
4. Mutating arrays returned from a context property does not mutate the context.
5. Internal-controller actions and externally supplied actions for a seat other than the episode learner do not call `Observe`.

Use this public contract in the tests:

```csharp
public interface ITacticalV2DecisionObserver
{
    void Reset(TacticalV2EpisodeContext episode);
    void Observe(TacticalV2DecisionContext decision);
}
```

- [ ] **Step 2: Run the focused tests and confirm the missing types fail compilation**

Run:

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DecisionObserverTests"
```

Expected: compilation fails because the observer contracts and registry snapshot do not exist.

- [ ] **Step 3: Implement immutable episode and decision DTOs**

In `TacticalV2DecisionObserver.cs`, add sealed DTOs whose constructors validate and copy their inputs. The decision context must expose:

```csharp
GameState State
PlayerId Seat
int DecisionIndex
float[] Observation              // cloned on construction and access
bool[] LegalMask                 // cloned on construction and access
int LearnerAction
Command LearnerCommand
TacticalV2UnitRegistry OwnRegistry
TacticalV2UnitRegistry FoeRegistry
TacticalV2Layout Layout
```

The episode context must expose the initial state, selected profile ID, reference seat, learner seat, and points weight. Add `TacticalV2UnitRegistry.Snapshot()` using a private copy constructor so each context owns stable slot identity without exposing mutation of the environment's registries.

- [ ] **Step 4: Integrate the optional observer without changing legacy action semantics**

Add a nullable `DecisionObserver` property to `TacticalV2DuelEnv`. In `ResetFromStart`, call `Reset` after registries and metadata are installed but before `AdvancePastInternal`. In external `Step` only:

1. If an observer is installed, construct the authoritative observation and mask from `_state` and copied registries.
2. Reject an out-of-range or masked learner action instead of allowing `Decode`'s legacy EndTurn fallback.
3. Decode the command and verify `TryEncode` returns the identical action.
4. Call `Observe` synchronously only when the active external seat equals the episode learner seat.
5. Apply the already-decoded learner command through the existing `TryApply` path.

When `DecisionObserver` is null, retain the current decode/apply path exactly. Do not observe internal Random/search controller actions or the unstick fallback.

- [ ] **Step 5: Prove passive-observer equivalence and action ownership**

Add a reciprocal test that runs the same seeded scripted action sequence twice, once without an observer and once with a recorder that returns no action. Assert equality of commands, per-step observations, masks, rewards, terminal outcome, trace projections, and replay text. Add a malicious recorder holding a different command and prove the environment still applies the supplied learner command.

- [ ] **Step 6: Run engine regression and Unity compilation checks**

Run:

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DecisionObserverTests|FullyQualifiedName~TacticalV2DuelEnvTests"
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```

Then use Coplay `check_compile_errors`; expected: zero errors.

- [ ] **Step 7: Commit the observer boundary**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2DecisionObserver.cs engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs engine/HexWars.Engine.Tests/TacticalV2DecisionObserverTests.cs
git commit -m "feat: add tactical decision observer boundary"
```

---

### Task 2: Implement selective eligibility and the bounded-search oracle

**Files:**

- Create: `engine/HexWars.Engine/Rl/TacticalV2Dagger.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs:8-34`
- Modify: `engine/HexWars.Engine/BoundedSearchAgent.cs:13-29`
- Test: `engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs`
- Regression: `engine/HexWars.Engine.Tests/BoundedSearchAgentTests.cs`
- Reference parity: `python/ml_lab/draw_classification.py:213-242`

- [ ] **Step 1: Write failing tests for every eligibility reason**

Build small authoritative states and assert the complete flag set for:

- opponent living-unit count `<= 1`;
- positive normalized advantage, including partial HP and banked points;
- the second occurrence of the canonical state key, but not the first;
- EndTurn with at least one `TacticalEvaluationTrace.ProjectState(state).ProductiveLegalActions`;
- multiple simultaneous reasons;
- a repeated eligible state emitted only once per episode.

For normalized advantage, assert the exact formula:

```text
seat material = sum(point_cost * current_hp / maximum_hp for living units)
                + points_weight * banked_points
normalized advantage = (learner material - opponent material)
                       / max(1, initial material of both seats)
```

Also add a golden canonical-key fixture shared with the field order in Python's `_cycle_key`: active seat; both seats' `(points, destroyed_value)`; sorted controlled cells; sorted living-unit tuples including moved, attacked, and horizontal/vertical movement spent.

- [ ] **Step 2: Write failing oracle legality, determinism, and evidence tests**

The tests must require:

```csharp
public interface IActionOracle
{
    TacticalV2OracleDecision Decide(TacticalV2DecisionContext context);
}
```

For depth 4 with budgets 512 and 2,048, assert that the oracle returns a legal command, encodes to an unmasked action, reports `LastExpansionCount <= ExpansionBudget`, and returns identical command/action/evidence when queried twice on the same context. Assert construction rejects fog of war and unrecognized heuristic identities.

- [ ] **Step 3: Run the focused tests and confirm they fail**

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DaggerTests|FullyQualifiedName~BoundedSearchAgentTests"
```

Expected: the DAgger types are absent and bounded-search configuration is not inspectable.

- [ ] **Step 4: Add stable search metadata and reusable trace projection**

Expose read-only `ExpansionBudget`, `Depth`, `UseHeuristic`, and a constant heuristic identity such as `material-plus-pursuit-v1` from `BoundedSearchAgent`; do not alter move ordering or evaluation. Change `TacticalEvaluationTrace.ProjectState` from `private` to `internal` so eligibility uses the same productive-action diagnostic as evaluation.

- [ ] **Step 5: Implement the engine DAgger domain objects**

In `TacticalV2Dagger.cs`, add:

```csharp
[Flags]
public enum DaggerEligibilityReason
{
    None = 0,
    Conversion = 1,
    Favorable = 2,
    CycleWarning = 4,
    WastedEndTurn = 8,
}
```

Add immutable `TacticalV2OracleDecision` and `TacticalV2DaggerDecision` DTOs, `ITacticalV2DaggerSink`, and `BufferedTacticalV2DaggerSink`. A row must carry the pre-action arrays, learner and teacher action/command, all reasons, state hash, normalized advantage, opponent living count, productive legal-action count, seat/round/decision index, disagreement, and oracle depth/budget/heuristic/actual expansions.

Implement `BoundedSearchActionOracle` as the only concrete oracle. It must run against the context state, encode through the snapshotted registry, and throw if command legality, mask membership, or action round-trip fails.

- [ ] **Step 6: Implement `SelectiveDaggerObserver`**

The observer owns episode-local repetition counts and emitted-state hashes. On each decision it:

1. Computes the canonical key and increments its occurrence count.
2. Computes all four eligibility reasons before the learner action is applied.
3. Returns without search if the flag set is empty or the key was already emitted.
4. Queries the frozen oracle once.
5. Revalidates teacher and learner round-trips against the recorded mask.
6. Emits a cloned evidence row and marks the key emitted.

Selection diagnostics must never affect reward or replace the learner command.

- [ ] **Step 7: Run focused and full engine tests, then compile Unity**

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DaggerTests|FullyQualifiedName~BoundedSearchAgentTests|FullyQualifiedName~TacticalEvaluationTraceTests"
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
```

Run Coplay `check_compile_errors`; expected: zero errors.

- [ ] **Step 8: Commit engine selection and oracle support**

```powershell
git add engine/HexWars.Engine/Rl/TacticalV2Dagger.cs engine/HexWars.Engine/Rl/TacticalEvaluationTrace.cs engine/HexWars.Engine/BoundedSearchAgent.cs engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs engine/HexWars.Engine.Tests/BoundedSearchAgentTests.cs
git commit -m "feat: add selective DAgger search oracle"
```

---

### Task 3: Expose DAgger evidence through GymServer and `DuelClient`

**Files:**

- Modify: `engine/HexWars.GymServer/Program.cs:65-69,304-305,353-354,400-401,436-459`
- Modify: `python/ml_lab/evaluation.py:205-310`
- Test: `python/tests/test_evaluation.py:1710-1785`
- Test: `engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs`

- [ ] **Step 1: Write failing Python protocol tests**

Add tests for these exact requests:

```python
client.configure_dagger(
    enabled=True,
    depth=4,
    expansion_budget=512,
    use_heuristic=True,
)
client.drain_dagger()
```

Expected JSONL commands:

```json
{"cmd":"duel_dagger_configure","enabled":true,"depth":4,"expansion_budget":512,"use_heuristic":true}
{"cmd":"duel_dagger_drain"}
```

Cover exact acknowledgement fields, strict booleans/integers, schema version 1, required row keys, finite observation/diagnostics, boolean mask shape, legal learner/teacher actions, reason bitmask, 64-character lowercase state hash, and exact teacher evidence.

- [ ] **Step 2: Run the protocol tests and confirm missing methods fail**

```powershell
uv run pytest -q python/tests/test_evaluation.py -k dagger
```

- [ ] **Step 3: Add GymServer configuration and drain commands**

Instantiate one buffered DAgger sink beside `tacticalV2Demonstrations`. `duel_dagger_configure` must:

- be tactical-v2 duel-only;
- reject fog of war before enabling;
- validate depth, budget, and heuristic choice;
- install a new `SelectiveDaggerObserver` using `tacticalV2Config.PointsWeight` when enabled;
- clear and detach it when disabled;
- return the exact normalized configuration in its acknowledgement.

`duel_dagger_drain` returns an object with exactly `schema_version: 1` and a `decisions` array, then clears only the DAgger buffer. Keep ordinary demonstration capture independent.

- [ ] **Step 4: Add strict Python request/response validation**

Implement `DuelClient.configure_dagger` and `DuelClient.drain_dagger`, plus a standalone `validate_dagger_payload`. Reject coercible values (`True` as integer, strings as numbers), extra/missing keys, non-finite values, wrong contract array lengths, masked actions, and malformed commands before returning any row.

- [ ] **Step 5: Exercise the real JSONL server path**

Add a C# serialization test or a Python real-server integration test that resets a one-unit conversion game, takes one legal external action, drains DAgger evidence, and verifies the learner action remains the applied transition while the teacher action is only evidence.

- [ ] **Step 6: Run protocol, engine, and compile verification**

```powershell
uv run pytest -q python/tests/test_evaluation.py -k "dagger or demonstration"
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV2DaggerTests|FullyQualifiedName~TacticalV2DuelEnvTests"
```

Run Coplay `check_compile_errors`; expected: zero errors.

- [ ] **Step 7: Commit the transport boundary**

```powershell
git add engine/HexWars.GymServer/Program.cs python/ml_lab/evaluation.py python/tests/test_evaluation.py engine/HexWars.Engine.Tests/TacticalV2DaggerTests.cs
git commit -m "feat: expose DAgger evidence protocol"
```

---

### Task 4: Build immutable DAgger overlay storage and validation

**Files:**

- Create: `python/ml_lab/dagger.py`
- Create: `python/tests/test_dagger.py`
- Reference: `python/ml_lab/imitation.py:341-447,917-945`
- Reference: `python/ml_lab/io.py`

- [ ] **Step 1: Write failing schema and storage tests**

Cover:

- strict `OracleSpec`, `LearnerIdentity`, `DaggerRow`, `DaggerGame`, and `DaggerOverlay` parsing;
- observation shape/dtype/finiteness, packed mask length, action legality, command/action agreement, and disagreement consistency;
- complete eligibility reason preservation;
- unique `(game_id, decision_index)` and one canonical state hash per episode;
- distinct train and validation partitions;
- immutable `manifest.json` plus per-game `.npz` shards with SHA-256, row count, byte size, and content identity;
- a corrupt byte, missing shard, extra manifest field, checkpoint hash change, or duplicate row failing physical reopen;
- publication by atomic rename and reuse only after physical reopen.

- [ ] **Step 2: Run the focused test and confirm import failure**

```powershell
uv run pytest -q python/tests/test_dagger.py -k "overlay or schema"
```

- [ ] **Step 3: Implement strict domain DTOs and seed namespaces**

In `dagger.py`, use frozen dataclasses and explicit parsers. Define one authoritative seed-range table for oracle preflight, each train iteration, each validation iteration, smoke, reserved, and development evaluation. Add `require_seed_in_partition(seed, partition, iteration)` and a whole-definition overlap validator; never infer a partition from a numerical prefix.

- [ ] **Step 4: Implement staging, shard writing, and physical reopening**

Store policy tensors compactly in one shard per completed game:

```text
observations: float32 [N, observation_size]
packed_masks: uint8 [N, ceil(action_size / 8)]
actions: int32 [N]                 # teacher targets used by training
learner_actions: int32 [N]
seats, rounds, decision_indices: int32 [N]
reason_bits: uint8 [N]
state_hashes: fixed-width ASCII [N]
```

Put commands, profile/schedule fields, normalized advantage, oracle expansion evidence, learner/oracle identities, and scenario/contract/encoding/repository/panel hashes in strict per-game and overlay manifests. Publish only a completed reciprocal pair. `open_dagger_overlay` must re-hash and revalidate all shards rather than trusting `status: completed`.

- [ ] **Step 5: Add trace/replay evidence references**

Require every game manifest to reference a physically present trace JSON and replay file with hash and outcome metadata. The overlay loader opens both and verifies their declared seed, seats, profile, outcome, and transition count before accepting the game.

- [ ] **Step 6: Run focused tests and commit**

```powershell
uv run pytest -q python/tests/test_dagger.py -k "overlay or schema or seed"
```

```powershell
git add python/ml_lab/dagger.py python/tests/test_dagger.py
git commit -m "feat: add immutable DAgger overlays"
```

---

### Task 5: Implement deterministic selective collection and restart safety

**Files:**

- Modify: `python/ml_lab/dagger.py`
- Modify: `python/tests/test_dagger.py`
- Reference: `python/ml_lab/evaluation.py:205-344`
- Reference: `python/run_annihilation_checkpoint_audit.py:67-151`

- [ ] **Step 1: Write failing schedule and fake-runtime tests**

Use a fake `DuelClient` and fake deterministic learner to prove:

- train collection alternates the learner through both reciprocal seats for each map;
- profile scheduling is exactly 70% `standard-3v3` and 30% across declared conversion profiles, using deterministic residual accounting;
- the learner executes its prediction and the drained teacher action never enters `step`;
- all eligible rows, including agreements, are retained;
- duplicate state hashes within an episode are rejected;
- train stops after at least 20,000 valid labels only at a reciprocal-pair boundary;
- validation stops after at least 2,000 valid labels only at a pair boundary;
- train fails at 2,000 games and validation fails at 200 games when targets are unmet;
- exact completed pairs are reused with zero new games, while partial or identity-mismatched pairs are not.

- [ ] **Step 2: Run focused collection tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_dagger.py -k "schedule or collection or restart"
```

- [ ] **Step 3: Implement the reciprocal schedule**

Add immutable `CollectionDefinition` and `ScheduledDuel` records. Generate map/profile/reference-seat/learner-seat tuples from the frozen seed bank. A reciprocal pair shares map seed and profile and contains exactly learner seats 0 and 1 in stable order. Validate the whole schedule before opening GymServer.

Use one residual scheduler so every ten completed map pairs contain seven standard and three conversion assignments over the long run. Cycle declared conversion profiles in canonical panel order; observed label yield must not change the schedule.

- [ ] **Step 4: Implement game collection**

Implement `collect_selective_dagger` with keyword-only `definition: CollectionDefinition`, `learner: ResolvedController`, `oracle: OracleSpec`, `output_root: Path`, `client_factory: Callable[[], DuelClient]`, and `progress: Callable[[str], None]` parameters, returning `DaggerOverlay`.

For each game, configure DAgger before reset, run deterministic learner versus Random, enable/drain the authoritative trace, save the replay, drain DAgger exactly once, attach Python-owned identities, validate the complete game, then stage it. Reuse one server across games, but reset observer episode state at every duel reset.

- [ ] **Step 5: Add progress and failure diagnostics**

Flush a line after every game and pair containing games, labels, per-reason counts, disagreements, mean/max expansions, labels/second, elapsed time, and ETA. On failure, retain a diagnostic staging manifest with the exception, last complete pair, and physical files, but omit the completion marker.

- [ ] **Step 6: Implement exact reuse checks**

Before launching a runtime, reopen an existing overlay and compare repository, panel, scenario, contract, encoding, learner checkpoint/hash, oracle, dataset, schedule, target, and ceiling identities. Reuse returns the opened object and records `new_games=0`. Never append to or repair a completed overlay in place.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest -q python/tests/test_dagger.py -k "schedule or collection or restart or progress"
git add python/ml_lab/dagger.py python/tests/test_dagger.py
git commit -m "feat: collect selective DAgger labels"
```

---

### Task 6: Generalize materialization and sampling to a 49/21/30 corpus

**Files:**

- Modify: `python/ml_lab/imitation.py:336-339,449-465,695-747,917-949,1201-1415`
- Modify: `python/ml_lab/dagger.py`
- Modify: `python/tests/test_imitation.py`
- Modify: `python/tests/test_dagger.py`

- [ ] **Step 1: Write failing three-source sampler tests**

Add `Source.DAGGER_TARGETED` expectations and prove that a seeded batch sequence exposes exactly 49%, 21%, and 30% in the limit, including batch size 256 where fractions are non-integral. Assert deterministic residual carry, deterministic cycling, shuffled batch order, and cycling of an undersized DAgger pool without source-ratio drift. For the targeted source, prove every row is visited once in a seeded permutation before any row repeats and that profile, seat, reason, disagreement, and action kind do not change its probability.

Keep regression tests showing the legacy two-source sampler emits the existing 70/30 sequence byte-for-byte when no explicit mixture is passed.

- [ ] **Step 2: Write failing composite-corpus tests**

Materialize an original imitation fixture plus two train overlays and two validation overlays. Assert train contains base plus cumulative train overlays; validation contains only cumulative held-out targeted overlays; no validation identity appears in train; base shard paths/hashes are unchanged; targeted rows use teacher `actions`, not `learner_actions`; and overlay 1 remains when iteration 2 is added.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_imitation.py python/tests/test_dagger.py -k "mixture or corpus or sampler"
```

- [ ] **Step 4: Add the generic source-mixture sampler**

Add `DAGGER_TARGETED` to `Source` and implement `SourceMixtureSampler` over a `MaterializedImitationPartition`. Derive the original Greedy and search profile/seat/action-kind strata directly from its immutable `ImitationBatch`, avoiding the assumption that all rows share one shard root. Give `DAGGER_TARGETED` one seeded uniform row cycler over the entire targeted partition: it must not stratify or quality-weight targeted rows after the engine's per-episode state cap.

The constructor accepts an ordered immutable mapping whose finite positive values sum to 1.0. Maintain one residual accumulator per source:

```python
target[source] = batch_size * fraction[source] + carry[source]
count[source] = floor(target[source] + 1e-12)
while sum(count.values()) < batch_size:
    source = stable_argmax(target[source] - count[source])
    count[source] += 1
carry[source] = target[source] - count[source]
```

Assign every rounding remainder one row at a time by greatest residual with stable source-order tie-breaking, recompute each carry after the final allocation, then permute combined rows with the seeded RNG. Retain `StratifiedDecisionSampler` as a compatibility wrapper constructing the original 70/30 mapping.

- [ ] **Step 5: Add generic actor-supervision materialization**

```python
@dataclass(frozen=True)
class ActorSupervisionCorpus:
    training: MaterializedImitationPartition
    validation: MaterializedImitationPartition
    source_fractions: Mapping[Source, float]
    identity: Mapping[str, Any]
```

Factor the optimization body of `train_behavioral_clone` into an internal `train_actor_supervision` function consuming this corpus. Keep `train_behavioral_clone` as a behavior-preserving adapter for an `ImitationDataset` and the legacy 70/30 mixture.

In `dagger.py`, implement `build_dagger_corpus(base, train_overlays, validation_overlays)`: materialize the base once, decode overlay arrays into compatible `ImitationBatch` objects, concatenate without copying physical base shards, and freeze the 0.49/0.21/0.30 mapping. Use collision-proof row identities containing component hash, game ID, and decision index.

- [ ] **Step 6: Test source accounting and validation isolation**

Run at least 10,000 scheduled examples and assert exact residual-accounted totals. Use a sentinel validation action absent from train and prove it never appears in optimizer batches but does appear in validation metrics.

- [ ] **Step 7: Run imitation and DAgger regression suites**

```powershell
uv run pytest -q python/tests/test_imitation.py python/tests/test_dagger.py
```

- [ ] **Step 8: Commit the generic corpus boundary**

```powershell
git add python/ml_lab/imitation.py python/ml_lab/dagger.py python/tests/test_imitation.py python/tests/test_dagger.py
git commit -m "feat: train from mixed supervision corpora"
```

---

### Task 7: Warm-start and publish actor-only distillation safely

**Files:**

- Modify: `python/ml_lab/algorithms.py:35-58,152-260`
- Modify: `python/ml_lab/imitation.py:829-851,1201-1415`
- Modify: `python/ml_lab/dagger.py`
- Modify: `python/tests/test_algorithms.py`
- Modify: `python/tests/test_imitation.py`
- Modify: `python/tests/test_dagger.py`
- Reference: `python/ml_lab/controllers.py`

- [ ] **Step 1: Write failing actor-transfer tests for both source kinds**

Test the exact immutable PPO snapshot spec:

```json
{
  "kind": "snapshot",
  "path": "C:/Users/cddal/HexWars/python/runs/bc227-ppo-random-s227-20260802-v2/checkpoints/step_000038912.zip",
  "source_run": "C:/Users/cddal/HexWars/python/runs/bc227-ppo-random-s227-20260802-v2",
  "algorithm": "maskable_ppo",
  "step": 38912,
  "inference_mode": "deterministic"
}
```

and a preceding completed DAgger actor. Require exact contract/encoding geometry, checkpoint SHA-256, source-run containment, algorithm, policy class, step/name, and deterministic mode before any copy. Assert actor/shared-feature parameters match the source, value parameters retain newly created target values, and malformed sources leave the target untouched.

- [ ] **Step 2: Write failing distillation publication tests**

Require batch 256, learning rate `3e-4`, max 50 epochs, patience 5, seed 227, CUDA for production, actor-only gradients, targeted validation NLL, best-epoch restoration, canonical CPU save, physical reload, and exact fixture logits. Assert the manifest records source checkpoint/actor hashes, target initial/final actor hashes, unchanged value hash, corpus identities, hardware/software provenance, epoch timings, and source-mixture counts.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_algorithms.py python/tests/test_imitation.py python/tests/test_dagger.py -k "actor_transfer or warm_start or distill or publication"
```

- [ ] **Step 4: Factor a transactional actor-copy primitive**

In `algorithms.py`, extract compatible actor-module comparison and copy logic from `MaskablePPOAdapter.initialize_actor`. The new method accepts a controller-resolved source checkpoint, verifies every actor tensor/name before mutation, snapshots the target, copies only `policy.features_extractor`, `policy.mlp_extractor.policy_net`, and `policy.action_net`, then verifies hashes. Restore the target on error. Keep the existing BC-to-PPO API as a wrapper.

- [ ] **Step 5: Add warm-start support to actor supervision**

Have `train_actor_supervision` create the production MaskablePPO architecture, invoke the actor initializer before optimizer creation, capture untouched value parameters, and optimize only actor/shared-feature parameters. Do not load optimizer, scheduler, replay, PPO rollout, or value state.

Iteration 1 must use checkpoint SHA `ec20df88d980b4ec80d68d704eafa134600b87ee947019fd64e2b7cc84974561`; iterations 2 and 3 reference the preceding published DAgger actor by physical path and SHA.

- [ ] **Step 6: Reopen CPU publication and compare logits**

Before marking training complete, reopen the saved zip with `device="cpu"`, validate contract/encoding, and compare fixture logits with the existing exact canonical tolerance. Re-hash the physical checkpoint and atomically publish the manifest only after every check passes.

- [ ] **Step 7: Run training-domain regression**

```powershell
uv run pytest -q python/tests/test_algorithms.py python/tests/test_imitation.py python/tests/test_dagger.py
```

- [ ] **Step 8: Commit warm-start distillation**

```powershell
git add python/ml_lab/algorithms.py python/ml_lab/imitation.py python/ml_lab/dagger.py python/tests/test_algorithms.py python/tests/test_imitation.py python/tests/test_dagger.py
git commit -m "feat: warm start DAgger actor distillation"
```

---

### Task 8: Freeze panel definitions and implement oracle preflight

**Files:**

- Create: `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md`
- Create: `python/panels/annihilation-selective-dagger-v1/panel.json`
- Create: `python/panels/annihilation-selective-dagger-v1/seed-banks.json`
- Modify: `python/ml_lab/dagger.py`
- Modify: `python/tests/test_dagger.py`
- Create: `python/tests/test_annihilation_selective_dagger.py`

- [ ] **Step 1: Write failing strict-definition tests**

Require exact schema keys and values for the seed-227 checkpoint path/source/step/mode/SHA; original dataset path/hash; tactical-v2 scenario/contract/encoding/repository identities; profiles `conversion-3v1-near`, `conversion-3v1-far`, `conversion-2v1-near`, `conversion-2v1-far`, `conversion-1v1-near`, and `conversion-1v1-far`; oracle candidates `(4,512,true)` and `(4,2048,true)`; 20 consecutive maps per profile in canonical profile order across `18,900,000-18,900,119`, both seats; all train, validation, smoke, reserved, and development banks; and every locked target, ceiling, mixture, optimizer setting, and success threshold.

Reject unknown fields, overlaps, wrong counts, final-seed references, or a checkpoint/dataset whose physical hash differs.

- [ ] **Step 2: Write failing oracle-preflight tests**

With fake evaluator and oracle benchmark boundaries, assert exactly 240 games per candidate, identical schedules, two identical queries per sampled state, authoritative legal round-trips for every label, and in-process throughput. Enforce pooled wins `>= 0.85` and labels/second `>= 10`.

Test tie-break order: win rate, fewer cycling draws, higher throughput, smaller budget. If neither candidate passes, no collection or training callback may run.

- [ ] **Step 3: Run focused tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "definition or preflight or oracle"
```

- [ ] **Step 4: Write the frozen definition documents**

`PROTOCOL.md` explains the causal question, covariate shift/DAgger background, eligibility formulae, immutable overlays, actor-only masked cross-entropy, seed isolation, restart semantics, exact smoke, evaluation rules, and non-goals. `panel.json` owns model/data/oracle/training thresholds; `seed-banks.json` canonically describes every reserved range and reciprocal expansion.

- [ ] **Step 5: Implement strict loading and preflight execution**

Add `load_panel_definition`, `validate_panel_definition`, and `run_oracle_preflight` to `dagger.py`. Write per-game traces/replays and staged `oracle-preflight.json`; reopen and hash all evidence before publication. Record W/L/D, Wilson intervals, cycling/action-waste diagnostics, paired seats, decision determinism, round-trip failures, expansions, timings, and throughput per candidate plus the selected oracle.

- [ ] **Step 6: Add restart-safe preflight reuse**

An exact completed preflight returns its frozen `OracleSpec` and launches zero games. Any changed repository/definition/scenario/contract/encoding/teacher schedule requires a new output root; never overwrite incompatible completed evidence.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "definition or preflight or oracle"
git add python/panels/annihilation-selective-dagger-v1 python/ml_lab/dagger.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: define DAgger panel and oracle preflight"
```

---

### Task 9: Orchestrate cumulative three-iteration DAgger training

**Files:**

- Create: `python/run_annihilation_selective_dagger.py`
- Modify: `python/ml_lab/dagger.py`
- Modify: `python/tests/test_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Reference: `python/run_annihilation_checkpoint_audit.py:121-298,470-507`

- [ ] **Step 1: Write failing stage-order tests**

With injected preflight, collector, trainer, and evaluator boundaries, assert this exact order:

```text
prepare -> validate -> oracle preflight -> starting baseline
iteration k:
  held-out collection by incoming learner
  training collection by incoming learner
  cumulative corpus build from overlays 1..k
  actor-only distillation warm-started from incoming learner
  physical publication/reopen
```

Assert iteration 1 uses the audited step-38,912 snapshot, iteration 2 uses actor 1, and iteration 3 uses actor 2. A failure stops subsequent callbacks. Validation is collected before training and never enters the optimizer corpus.

- [ ] **Step 2: Write failing stage-identity/reuse tests**

Prove each stage identity binds definition bytes/hash, repository, scenario, contract, encoding, base dataset, selected oracle, incoming learner, cumulative overlay hashes, schedule, and optimizer config. Exact completed stages reopen with zero games/epochs. Missing/corrupt physical children or mismatched identities fail before downstream work.

- [ ] **Step 3: Run focused orchestration tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "iteration or stage or cumulative"
```

- [ ] **Step 4: Implement prepare and validate stages**

`prepare` copies canonical panel and seed-bank bytes into the output root without mutation, captures repository identity, and publishes once. `validate` reopens the audited learner checkpoint and base dataset, verifies all hashes/contracts/geometry/seed isolation, records resolved hardware/software, and publishes a strict validation manifest without launching games.

- [ ] **Step 5: Implement one iteration as a transactional pipeline**

Add `run_iteration(index: int, *, output_root: Path, dependencies: DaggerDependencies) -> IterationManifest` for indices 1 through 3 only. Resolve the incoming learner; collect its 2,000-label held-out overlay from the assigned 19m range with a 200-game ceiling; collect its 20,000-label training overlay from the assigned 18m range with a 2,000-game ceiling; build cumulative train/validation inputs; train with the locked settings; and reopen the CPU actor/checkpoint metadata.

Use deterministic staging paths beneath `iterations/iteration-{index}/.staging` and publish the iteration manifest only after every child is complete. Never silently continue a partly trained model.

- [ ] **Step 6: Verify cumulative provenance**

Iteration 2 must list overlay hashes 1-2 and iteration 3 hashes 1-3 for both train and held-out validation. Record exact row counts, source exposure, learner/teacher disagreement by reason, best epoch/NLL, checkpoint hash, source actor hash, and elapsed/throughput timings.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "iteration or stage or cumulative"
git add python/run_annihilation_selective_dagger.py python/ml_lab/dagger.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: orchestrate selective DAgger iterations"
```

---

### Task 10: Add paired development evaluation and locked decisions

**Files:**

- Modify: `python/ml_lab/dagger.py`
- Modify: `python/run_annihilation_selective_dagger.py`
- Modify: `python/tests/test_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Reference: `python/ml_lab/checkpoint_audit.py`
- Reference: `python/ml_lab/draw_classification.py`

- [ ] **Step 1: Write failing fixed-schedule evaluation tests**

Require the baseline and all three actors to play identical `standard-3v3` games on seeds `20,000,000-20,000,099`, both learner seats, deterministic model versus Random: exactly 200 games per candidate. Assert no observer/search API is enabled. Reject missing reciprocal games, profile drift, seed drift, model hash changes, or traces/replays that do not reopen.

- [ ] **Step 2: Write failing metric and paired-decision tests**

Using hand-built summaries, verify:

- W/L/D and Wilson 95% intervals;
- seat-specific rates and asymmetry;
- paired map-seat changes and exact sign tests;
- cycling/action-waste incidence and wasted-EndTurn counts/ratios;
- final/peak normalized advantage and rounds/decisions to victory;
- cumulative held-out teacher accuracy and disagreement by reason.

Candidate selection is highest win rate, then lower cycling, lower action waste, then earlier iteration. Success requires both `gain >= 0.20 or wins >= 0.65` and relative cycling reduction `>= 0.50`. A 20-point gain below 65% authorizes replication but does not meet the winning-model milestone. Encode the milestone separately as every replicate `>= 0.65` and pooled `>= 0.70`.

- [ ] **Step 3: Run focused evaluation tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "evaluation or decision or report"
```

- [ ] **Step 4: Implement physical evaluation and reuse**

Reuse the checkpoint-audit evaluation boundary rather than duplicating game semantics. Force Random, standard profile, deterministic inference, trace capture, replay save, and both seats. Classify retained traces with the existing draw/action-waste diagnostics. Stage one immutable result directory per candidate and reopen it before reuse.

- [ ] **Step 5: Implement aggregate and report construction**

Build `aggregate.json` only from reopened preflight, baseline, iteration manifests, overlays, checkpoints, traces, and replays. Render `REPORT.md` with causal question, frozen inputs, oracle selection, collection yields/reasons/disagreement, training curves/timings, supervised metrics, game outcomes, paired tests, chosen candidate, threshold decisions, and fixed interpretation branch.

Do not claim success from teacher accuracy alone. Explicitly report that final 17m seeds were not used and that three-replicate confirmation is a later experiment.

- [ ] **Step 6: Test reconstruction from retained evidence**

Delete only an in-memory cached summary in a fixture, rebuild aggregate/report from the physical artifacts, and assert canonical equality. Corrupt one trace/replay and require aggregation failure.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest -q python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py -k "evaluation or decision or report"
git add python/ml_lab/dagger.py python/run_annihilation_selective_dagger.py python/tests/test_dagger.py python/tests/test_annihilation_selective_dagger.py
git commit -m "feat: evaluate selective DAgger outcomes"
```

---

### Task 11: Complete the CLI, logging, and operator safeguards

> **Authoritative expanded procedure:** Implement this task with
> `docs/superpowers/plans/2026-08-09-task11-sealed-engine-preflight.md`.
> That plan adds the approved GymServer provenance boundary required by Task 10
> while preserving the command set, logging, reuse, and recovery goals below.

**Files:**

- Modify: `python/run_annihilation_selective_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Modify: `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md`

- [ ] **Step 1: Write failing parser and dispatch tests**

Expose only these subcommands:

```text
prepare
validate
preflight
baseline
iteration --index {1,2,3}
evaluate
aggregate
report
smoke
all
```

Every command requires an explicit output root; production commands use the committed panel directory and may not accept ad hoc seeds, thresholds, mixtures, teacher budgets, learner checkpoints, or final-bank switches. `all` dispatches in dependency order and stops on the first failure.

- [ ] **Step 2: Write failing logging and reuse tests**

Assert logs reach both stdout and `<output-root>/selective-dagger.log`, flush after every progress event, and include command, stage, candidate/iteration, games, labels, disagreements, expansions, throughput, elapsed, ETA, reuse counts, and failure path. A reuse run must visibly report zero new games and zero new epochs.

- [ ] **Step 3: Run focused runner tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_annihilation_selective_dagger.py -k "parser or dispatch or logging or reuse"
```

- [ ] **Step 4: Implement CLI dispatch and dependency guards**

Follow the small command/dispatch/logging structure in `run_annihilation_checkpoint_audit.py`; do not add DAgger branches to the older 5,000-line imitation panel runner. Each command reopens its prerequisites and fails with a named missing/incompatible stage before launching compute.

`evaluate` evaluates the baseline and every physically published iteration on the single fixed development schedule. `aggregate` and `report` are read-only reconstruction steps. `all` performs prepare, validate, preflight, baseline, iterations 1-3, evaluation, aggregate, and report.

- [ ] **Step 5: Document exact operator commands and artifacts**

Add PowerShell examples for validation, exact smoke, individual recovery commands, the full production run, and evidence inspection. State that production training requires CUDA, smoke deliberately trains one CPU epoch, Unity is for replay observation rather than training, and completed artifact reuse must launch no games.

- [ ] **Step 6: Test CLI errors and interrupt recovery**

Simulate failure during collection, training, and evaluation. Confirm logs/staging evidence remain inspectable, no completed marker exists, retry reuses completed prerequisite pairs/stages, and incompatible completed outputs require a different output root.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run pytest -q python/tests/test_annihilation_selective_dagger.py
git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py python/panels/annihilation-selective-dagger-v1/PROTOCOL.md
git commit -m "feat: add selective DAgger panel CLI"
```

---

### Task 12: Prove the exact physical smoke and complete verification

**Files:**

- Modify: `python/run_annihilation_selective_dagger.py`
- Modify: `python/tests/test_annihilation_selective_dagger.py`
- Modify: `python/panels/annihilation-selective-dagger-v1/PROTOCOL.md`
- Verify: all files changed in Tasks 1-11

- [ ] **Step 1: Write a failing exact-smoke schedule test**

Assert the `smoke` command does exactly:

1. collect `standard-3v3` seed `18,990,000` from learner seats 0 and 1;
2. collect `conversion-3v1-near` seed `18,990,001` from seats 0 and 1;
3. train one CPU epoch on accepted smoke labels and publish/reload the actor;
4. evaluate `standard-3v3` seeds `18,990,002` and `18,990,003`, each from seats 0 and 1;
5. rerun with zero newly launched collection/evaluation games and zero new training epochs.

The smoke oracle is the declared depth-4, 512-expansion, current-heuristic candidate; smoke verifies determinism and round-trip but does not pretend its four games satisfy the production preflight win-rate gate. Fail if collection yields no eligible label.

- [ ] **Step 2: Write failing physical-reuse and isolation tests**

Require exactly four collection and four evaluation replays/traces, one overlay, one CPU checkpoint, content hashes, and reopened manifests. Mark the smoke-only use of its training labels as validation explicitly and reject that alias in production. Assert no 17m, production 18m/19m, or 20m seed appears.

- [ ] **Step 3: Run the focused smoke tests and confirm failure**

```powershell
uv run pytest -q python/tests/test_annihilation_selective_dagger.py -k "smoke"
```

- [ ] **Step 4: Implement the smoke pipeline**

Route `smoke` through the same GymServer observer, overlay writer, actor initializer, masked loss, CPU publisher/reloader, deterministic controller, trace/replay writer, and physical validator as production. Parameterize only the frozen smoke schedule, one epoch, CPU device, small label target, and smoke-only validation alias; do not create a second implementation.

- [ ] **Step 5: Run the exact physical smoke twice**

```powershell
uv run python python/run_annihilation_selective_dagger.py --output-root python/runs/selective-dagger-smoke smoke
uv run python python/run_annihilation_selective_dagger.py --output-root python/runs/selective-dagger-smoke smoke
```

Expected first run: four collection games, one CPU epoch, four evaluation games, physically reopened actor. Expected second run: all stages reused, `new_games=0`, `new_epochs=0`. Inspect the log and manifests rather than inferring completion from process exit alone.

- [ ] **Step 6: Run all automated regressions**

```powershell
uv run pytest -q
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj
git diff --check
```

Expected: all Python and .NET tests pass and no whitespace errors are reported.

- [ ] **Step 7: Verify Unity health and compilation**

Use Coplay `get_unity_editor_state` first. If connected, run `check_compile_errors` and require zero errors. Run affected Unity EditMode tests through Coplay if shared engine imports or assemblies changed; read `get_unity_logs` for any failure. No scene/default value is intentionally changed, so no scene mutation should be saved.

- [ ] **Step 8: Audit artifacts and no-inference-search guarantee**

Open the smoke overlay/checkpoint/manifests/traces/replays and compare every recorded SHA. Search ordinary evaluation/publication code to prove it neither configures `duel_dagger_configure` nor instantiates `IActionOracle`. Confirm generated evidence is ignored by Git and the committed panel directory contains definitions/docs only.

- [ ] **Step 9: Commit smoke verification support**

```powershell
git add python/run_annihilation_selective_dagger.py python/tests/test_annihilation_selective_dagger.py python/panels/annihilation-selective-dagger-v1/PROTOCOL.md
git commit -m "test: prove selective DAgger physical smoke"
```

- [ ] **Step 10: Request code review and finish the branch**

Use `superpowers:requesting-code-review` on the complete branch, address findings with `superpowers:receiving-code-review`, rerun the affected focused tests and final verification, then use `superpowers:finishing-a-development-branch` to present merge/push/PR choices. Do not begin production preflight or DAgger collection as part of implementation.

---

## Specification Coverage Checklist

| Approved requirement | Implemented/tested in |
|---|---|
| Generic passive pre-action boundary; learner action owns transition | Tasks 1, 3 |
| Conversion, favorable, second-repetition, wasted-EndTurn selection | Task 2 |
| One canonical state per episode; all eligible labels | Tasks 2, 4, 5 |
| Deterministic depth-4 512-vs-2048 oracle preflight and gates | Tasks 2, 8 |
| Fog-off, legality, round-trip, expansions, throughput evidence | Tasks 2, 3, 8 |
| Immutable train/validation overlays and content hashes | Tasks 4, 5 |
| 20k/2k targets, 2,000/200 ceilings, reciprocal pairs, 70/30 profiles | Task 5 |
| Original data plus cumulative overlays; exact 49/21/30 exposure | Task 6 |
| Seed-227 checkpoint warm start; previous actor thereafter | Tasks 7, 9 |
| Actor-only masked CE; locked optimizer; CUDA train/CPU publish | Task 7 |
| Fixed 20m baseline/iteration schedule and paired diagnostics | Task 10 |
| Gain/cycling success rules and separate replication milestone | Task 10 |
| Structured logs, atomic publication, fail-closed physical reuse | Tasks 4, 5, 8, 9, 11 |
| Exact 4-game collection, 1-epoch CPU, 4-game evaluation smoke | Task 12 |
| Final 17m bank untouched; no inference-time search | Tasks 8, 10, 12 |

## Production Gate After This Plan

Implementation ends after the exact smoke and review are accepted. The next separately authorized operation is the 480-game oracle preflight. Only a passing selected oracle permits the newly measured 200-game baseline, three DAgger iterations, and their 600 evaluation games. Production compute must use a clean reviewed commit and a fresh output root whose frozen definitions match that commit.
