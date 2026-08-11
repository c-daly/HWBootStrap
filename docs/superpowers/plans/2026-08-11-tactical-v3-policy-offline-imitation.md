# Tactical-v3 Policy and Offline Imitation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Project B Python and Unity path that strictly consumes Project A tactical-v3 structured decisions, trains and publishes a custom variable-candidate PyTorch policy from a tiny immutable offline corpus, and selects exact candidate identities on 13x9 and 24x16 boards.

**Architecture:** A strict immutable Python contract parses the Project A `spaces` and decision JSON before any NumPy or Torch coercion. Focused corpus, batching, layer, model, objective, trainer, checkpoint, and controller modules keep structured learning out of the existing 2,370-line `imitation.py`; Unity Arena sends the same semantic decision shape to a CPU-only structured controller and applies the returned `(decision_id, candidate_id)` without a fallback action.

**Tech Stack:** Python 3.14.6 through `uv run --active --no-project`, PyTorch and NumPy already declared in `python/requirements.txt`, pytest 8+, Python stdlib JSONL/subprocess/hashlib/dataclasses, C# `netstandard2.1` engine, Unity 6000.5, NUnit/EditMode tests, JSONL policy bridge.

## Global Constraints

- The authoritative design is `docs/superpowers/specs/2026-08-10-generalizable-structured-imitation-design.md`; this Project B plan is decomposition item 2 and starts from reviewed Project A head `1b260449ee50b5c6ef8c7c01555b0b6d424ccbd4`.
- Project A is an immutable prerequisite: consume `tactical-v3`, its strict scenario, structured observations, candidate projections, and exact decision/candidate resolution; do not reintroduce tactical-v2 flat observations, unit slots, action offsets, or silent `EndTurn` fallback.
- Scope is only strict Python DTO/schema validation, GymServer client, semantic identities, one canonical example, one tiny immutable corpus, ragged typed batching, custom PyTorch policy/objectives/trainer/checkpoint, controller publication/loading, ML Lab/Unity Arena loading, and stage-zero conformance.
- Explicitly exclude full-game bounded-search collection, production DAgger, curriculum expansion, fog training, unit design, PPO/policy gradients, sealed evidence, checkpoint promotion, and any win-rate or tactical-competence claim. Those require subsequent plans.
- The smoke corpus may execute deterministic fixture trajectories only to obtain honest terminal labels. It stores at most eight training and four validation decisions, uses no search, has no append mode, and identifies its label source as `tiny-fixture-policy-v1`; it is not a teacher-collection pipeline.
- Use Python 3.14 through `uv run --active --no-project`. Before Python tests, run `uv run --active --no-project python --version` and require `Python 3.14.x`.
- Reuse the installed `torch`, `numpy`, and `pytest` packages. New structured modules must not import `stable-baselines3`/`sb3-contrib`, add PyTorch Geometric or graph libraries, disguise the schema as Gym `Discrete`/`Box`, or add any Python/NuGet/Unity package.
- Do not modify `python/requirements.txt`: it already declares `numpy>=1.24`, `torch>=2.1`, and `pytest>=8.0`.
- The model has no board width, board height, cell count, unit count, template count, candidate count, roster order, template name, entity ID, or permanent action dimension in architecture configuration or learned embeddings.
- `structured_imitation` is a new controller algorithm backed by a `.pt` state dictionary. It is not a MaskablePPO/DQN alias, and standalone checkpoints remain rejected because `run.json` is authoritative.
- All public DTOs are frozen dataclasses whose nested sequences are tuples and whose nested mappings are recursively read-only. Unknown/missing fields, booleans passed as integers, out-of-range integers, non-finite numbers, unknown enum values, invalid reference families/rows, stale decision IDs, duplicate candidate IDs, and nonterminal empty candidate sets fail before tensor creation.
- Runtime compatibility requires exact `contract_version == "tactical-v3"`, `encoding_hash`, and `capacity_hash`. `contract_hash` is recorded as the training match identity but may differ at 24x16 inference; this is necessary for one model to serve multiple match configurations.
- Recompute and compare the Project A encoding and capacity SHA-256 values from canonical sorted JSON. Do not attempt to make match-specific `contract_hash` a model geometry requirement.
- The checked-in Project A scenario identities are contract `0ae48260cde97bce9ed75975874676a262588b3ed17963cdb41d09d09d3088ce`, encoding `e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6`, and capacity `7aea1db4f008dc192e83811b2c13abd8ce2304d2a6a209f37f9847be5f367364`.
- Candidate IDs are opaque decision-local identities. The policy may return only a candidate present in the supplied frame and must return the unchanged frame `decision_id`; invalid selection never becomes another candidate.
- Padding has no semantic value. Every padded table, relation neighborhood, optional reference, target, and candidate carries an explicit boolean mask; all-masked nonterminal samples are errors.
- Row-order permutation plus consistent reference remapping must preserve physical candidate logits/actions; candidate-row permutation must only permute the matching logits. Adding padded rows or batching beside a larger board must not change valid logits.
- Cell coordinates are centered per sample before entering learned layers; local six-neighbor relations carry topology. There are no absolute-cell embeddings or flattening operations.
- Use custom PyTorch matrix/gather/softmax code only. Build padded incoming-neighbor tensors and use gather-based local/relational attention; do not use nondeterministic scatter accumulation or a graph dependency.
- Policy loss coefficient is exactly `1.0`. Configured terminal-outcome, win-within-horizon, and conditional remaining-turns coefficients are each finite/non-negative and their sum is at most `0.5`.
- Loss, logits, masks, gradients, gradient norm, and post-step parameters must be finite. Time-limit truncation is censored for remaining-victory targets, and remaining-turn loss is active only for wins with a defined remaining horizon.
- Set Python, NumPy, and Torch seeds; enable deterministic Torch algorithms; avoid dropout; use deterministic batch ordering; clip gradients; keep immutable validation partitions; restore the best finite validation epoch; never publish the latest epoch merely because it is latest.
- Training may run on CUDA, but publication always copies the selected model to CPU, removes optimizer state, reloads with `torch.load(..., map_location="cpu", weights_only=True)`, and verifies fixture logits within `rtol=1e-5, atol=1e-6` plus exact deterministic candidate identities. CPU save/load must reproduce logits exactly.
- Published runs live beneath `python/runs`, use existing schema-version-1 `run.json` controller conventions where compatible, name the algorithm `structured_imitation`, include strict architecture/loss/schema/corpus/scenario identities, and declare `evidence_status: "unsealed-experimental"`.
- Official metrics in this plan are parser/batching/model/training/inference checks only. Unity is behavioral inspection; no Unity observation is an official metric and no supervised metric is described as game performance.
- Use strict TDD in Tasks 1-12: add the named focused failing test, run the exact RED command and confirm the stated failure, write minimal production code, run the exact GREEN command, then commit. Task 13 is the independent final acceptance gate.
- Keep files focused. Do not add structured policy code to `python/ml_lab/imitation.py`, `python/ml_lab/dagger.py`, or the SB3 algorithm adapters.
- After every C# edit, rebuild/resync the engine plugin when applicable, run Coplay `check_compile_errors`, fix errors, and recheck. For Arena runtime work also inspect `get_unity_logs`.
- Git commits contain no attribution trailers or tool credits.

---

## Resolved Project A Wire Contract

The implementation must encode these actual `TacticalV3Wire.cs` names and semantic types, not inferred aliases:

```text
spaces = {
  scenario_id:string, scenario_schema_version:int32,
  contract_version:string, contract_hash:sha256, encoding_hash:sha256,
  capacity_hash:sha256, environment_kind:"tactical"|"duel",
  match:object, encoding:object, capacity:object
}

view = {
  decision_id:int64, seat:int32, observation:object, candidates:array,
  reward:object, winner:int32, terminated:bool, truncated:bool,
  start_profile:string, reference_seat:int32
}

cells = q:int32, r:int32, terrain:terrain_type, elevation:int32,
  self_deployment_zone:bool, opponent_deployment_zone:bool,
  controller:nullable_relative_owner, is_boundary:bool,
  currently_visible:bool, previously_observed:bool
units = owner:relative_owner, current_hp:int32, max_hp:int32, cell:token_ref,
  elevation:int32, moved:bool, attacked:bool,
  horizontal_movement_spent:int32, vertical_movement_spent:int32,
  point_cost:int32, deploy_cost:int32, currently_visible:bool
templates = owner:relative_owner, point_cost:int32, deploy_cost:int32,
  is_fixed:bool, is_deployable:bool
capability_definitions = kind:capability_kind
capability_allocations = owner:token_ref, definition:token_ref,
  capability:capability_kind, purchased_level:int32, effective_value:int32
rules = kind:rule_kind, int_value:int32, float_value:float32, bool_value:bool
memory = cell:token_ref, last_seen_round:int32, observation_age:int32,
  last_known_current_hp:int32, currently_visible:bool
relations = kind:relation_kind, source:token_ref, target:token_ref,
  int_feature:int32, float_feature:float32, bool_feature:bool
candidates = candidate_id:int32, decision_id:int64, kind:candidate_kind,
  actor:nullable_token_ref, target:nullable_token_ref,
  template:nullable_token_ref, cell:nullable_token_ref, projection:projected_delta
projected_delta = source_cell:nullable_token_ref, destination_cell:nullable_token_ref,
  template:nullable_token_ref, target:nullable_token_ref,
  horizontal_movement_spent:int32, vertical_movement_spent:int32,
  target_hp_delta:int32, damage:int32, is_lethal:bool,
  bounty_delta:int32, points_delta:int32, round_delta:int32, is_terminal:bool
reward = terminal_outcome:float32, known_health_adjusted_material_progress:float32,
  public_resource_progress:float32, time_pressure:float32, total:float32,
  finalized:bool
token_ref = table:table_kind, row:int32
```

Exact enum order is contractual:

```text
table_kind = cells, units, templates, capability_definitions,
  capability_allocations, rules, memory_records, relations, candidates
relative_owner = self, opponent
terrain_type = plains, forest, rough, water
relation_kind = neighbor, occupies, has_capability
capability_kind = health, damage, defense, movement, vertical_movement,
  range, range_arc, vision, vision_arc
candidate_kind = attack, move, deploy, end_turn
rule_kind = win_conditions, round, round_cap, actions_per_turn, starting_points,
  self_points, opponent_points, damage_floor, damage_high_ground_bonus,
  range_high_ground_bonus, bounty_rate, deploy_cost_multiplier, fog_of_war,
  max_design_point_cost, design_fee
```

---

## Planned File Structure

### Focused Python modules

- `python/ml_lab/tactical_v3_schema.py` — frozen wire DTOs, exact schema/enums, recursive freezing, reference validation, semantic identity validation, strict spaces/view/decision parsing.
- `python/ml_lab/tactical_v3_client.py` — owned GymServer JSONL subprocess with strict `spaces/reset/step/duel_*` requests and deterministic shutdown.
- `python/ml_lab/tactical_v3_corpus.py` — canonical structured example/target DTOs plus exclusive, content-addressed immutable corpus writer/loader.
- `python/ml_lab/tactical_v3_batching.py` — typed ragged tensor batches, masks, global reference remapping, incoming neighborhoods, and target collation.
- `python/ml_lab/tactical_v3_layers.py` — typed token encoders, centered coordinate encoder, local hex message passing, and typed gather-based relational attention.
- `python/ml_lab/tactical_v3_model.py` — shared state encoder, candidate projection encoder/scorer, outcome/horizon heads, and deterministic selection.
- `python/ml_lab/tactical_v3_objectives.py` — policy/outcome/horizon/remaining-turn losses and coefficient/target validation.
- `python/ml_lab/tactical_v3_training.py` — deterministic offline trainer, finite checks, immutable validation, best-state restoration, early stopping, and metrics/history.
- `python/ml_lab/tactical_v3_checkpoint.py` — strict weights-only `.pt` format, CPU canonicalization, inference fixtures, atomic unsealed run publication/reopen validation.
- `python/ml_lab/tactical_v3_controller.py` — structured contract compatibility, CPU controller load, exact candidate-identity inference adapter.
- `python/run_tactical_v3_imitation.py` — narrow `build-tiny-corpus`, `train`, and `validate-run` CLI; no teacher, DAgger, evaluation, or curriculum command.

### Python integration edits and tests

- Modify `python/ml_lab/controllers.py` only to recognize and dispatch the new manifest-backed algorithm without applying fixed SB3 geometry.
- Modify `python/policy_server.py` only to validate tactical-v3 expectations and route structured decisions to the new adapter.
- Create `python/tests/test_tactical_v3_schema.py`, `test_tactical_v3_client.py`, `test_tactical_v3_corpus.py`, `test_tactical_v3_batching.py`, `test_tactical_v3_layers.py`, `test_tactical_v3_model.py`, `test_tactical_v3_objectives.py`, `test_tactical_v3_training.py`, `test_tactical_v3_checkpoint.py`, `test_tactical_v3_controller.py`, and `test_tactical_v3_end_to_end.py`.
- Modify `python/tests/test_controllers.py` and `python/tests/test_policy_server.py` for legacy-regression plus tactical-v3 routing.
- Create `python/tests/fixtures/tactical_v3/seed-41-spaces.json`, `seed-41-decision.json`, `scenario-24x16.json`, and `tiny-corpus/{manifest.json,train.jsonl,validation.jsonl}`.

### Engine and Unity Arena integration

- Modify `engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs` and `engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs` to expose non-destructive transition draining for Arena playback.
- Modify `Assets/HexWars/Presentation/MlEnvironmentContract.cs` to add `TacticalV3` and capacity identity.
- Modify `Assets/HexWars/Presentation/ModelDuelEnvironment.cs` for a tactical-v3 decision/candidate adapter without changing legacy flat adapters.
- Create `Assets/HexWars/Presentation/TacticalV3PolicyPayload.cs` plus its Unity `.meta` — exact structured decision projection for `JsonUtility`.
- Modify `Assets/HexWars/Presentation/PolicyBridge.cs` to send structured decisions and parse exact returned identities.
- Modify `Assets/HexWars/Presentation/ModelDuelDriver.cs` to branch between legacy flat actions and tactical-v3 identities.
- Modify `Assets/HexWars/Presentation/ModelArenaIdentity.cs` only to display `structured_imitation` clearly.
- Modify `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`, `MlLabWindow.cs`, and `MlLabConfig.cs` to load tactical-v3 run scenarios into Arena while explicitly refusing SB3 Train-tab launch.
- Create `Assets/HexWars/Tests/Editor/TacticalV3PolicyPayloadTests.cs` plus its Unity `.meta`.
- Modify `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`, `ModelDuelConfigurationTests.cs`, `MlTrainingScenarioTests.cs`, `MlLabConfigTests.cs`, and `MlLabWindowStateTests.cs`.
### Evidence

- Create `docs/superpowers/reports/2026-08-11-generalizable-structured-imitation-project-b.md` with exact acceptance commands/results, identities, hashes, deterministic evidence, cross-size result, Unity status, and explicit claim limits.

---

### Task 1: Strict Semantic Identities and Immutable Wire DTOs

**Files:**
- Create: `python/ml_lab/tactical_v3_schema.py`
- Create: `python/tests/test_tactical_v3_schema.py`

**Interfaces:**
- Consumes: exact `spaces`/view contract above and Project A hashes.
- Produces:

```python
TableName = Literal["cells", "units", "templates", "capability_definitions",
    "capability_allocations", "rules", "memory_records", "relations", "candidates"]

@dataclass(frozen=True, slots=True)
class TokenRef:
    table: TableName
    row: int

@dataclass(frozen=True, slots=True)
class TacticalV3Observation:
    cells: tuple[CellToken, ...]
    units: tuple[UnitToken, ...]
    templates: tuple[TemplateToken, ...]
    capability_definitions: tuple[CapabilityDefinitionToken, ...]
    capability_allocations: tuple[CapabilityAllocationToken, ...]
    rules: tuple[RuleToken, ...]
    memory: tuple[MemoryToken, ...]
    relations: tuple[RelationToken, ...]

@dataclass(frozen=True, slots=True)
class TacticalV3Decision:
    decision_id: int
    seat: int
    observation: TacticalV3Observation
    candidates: tuple[Candidate, ...]

@dataclass(frozen=True, slots=True)
class TacticalV3SemanticIdentity:
    scenario_id: str
    scenario_schema_version: int
    contract_version: Literal["tactical-v3"]
    contract_hash: str
    encoding_hash: str
    capacity_hash: str
    environment_kind: Literal["tactical", "duel"]
    match: Mapping[str, object]
    encoding: Mapping[str, object]
    capacity: Mapping[str, int]

def parse_spaces(payload: object) -> TacticalV3SemanticIdentity
def parse_view(payload: object, identity: TacticalV3SemanticIdentity) -> TacticalV3View
def parse_decision(payload: object, identity: TacticalV3SemanticIdentity) -> TacticalV3Decision
def canonical_sha256(value: object) -> str
```

- [ ] **Step 1: Write failing exact-shape, type, immutability, hash, and reference tests**

```python
def test_spaces_and_view_parse_to_deeply_immutable_semantic_values() -> None:
    identity = parse_spaces(load_fixture("seed-41-spaces.json"))
    view = parse_view(load_fixture("seed-41-decision.json"), identity)
    assert identity.encoding_hash == canonical_sha256(identity.encoding)
    assert identity.capacity_hash == canonical_sha256(identity.capacity)
    assert view.decision.candidates[0].decision_id == view.decision.decision_id
    with pytest.raises(TypeError):
        identity.capacity["max_cells"] = 1

@pytest.mark.parametrize("mutation", [
    "unknown_field", "bool_candidate_id", "nan_rule_float", "wrong_ref_table",
    "ref_row_equal_to_length", "duplicate_candidate_id", "stale_candidate_decision",
])
def test_parser_rejects_malformed_wire_before_numeric_coercion(mutation: str) -> None:
    spaces, view = mutated_fixture(mutation)
    with pytest.raises((TypeError, ValueError), match=EXPECTED_ERRORS[mutation]):
        parse_view(view, parse_spaces(spaces))
```

- [ ] **Step 2: Run the schema test and confirm RED**

Run:

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_schema.py -q
```

Expected: collection fails with `ModuleNotFoundError: No module named 'ml_lab.tactical_v3_schema'`.

- [ ] **Step 3: Implement exact primitive validators and recursive freezing**

Use exact helpers; never call `int(value)` or `float(value)` before validation:

```python
def _int32(value: object, field: str) -> int:
    if type(value) is not int or not -(2**31) <= value < 2**31:
        raise TypeError(f"{field} must be an int32")
    return value

def _float32(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a float32 number")
    result = float(np.float32(value))
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result

def _exact_mapping(value: object, fields: frozenset[str], field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or frozenset(value) != fields:
        raise ValueError(f"{field} fields must be exactly {sorted(fields)}")
    return value
```

`canonical_sha256` uses `json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)` over recursively thawed strings/integers/booleans/lists/maps. Encoding and capacity contain no floating-point leaves, so this is byte-compatible with Project A's canonical serializer.

- [ ] **Step 4: Implement all DTO parsers and semantic family/range validation**

Validate the candidate family matrix exactly:

```python
CANDIDATE_REFERENCE_FAMILIES = {
    "attack": {"actor": "units", "target": "units", "template": None, "cell": None},
    "move": {"actor": "units", "target": None, "template": None, "cell": "cells"},
    "deploy": {"actor": None, "target": None, "template": "templates", "cell": "cells"},
    "end_turn": {"actor": None, "target": None, "template": None, "cell": None},
}

PROJECTED_REFERENCE_FAMILIES = {
    "attack": {"source_cell": "cells", "destination_cell": None, "template": None, "target": "units"},
    "move": {"source_cell": "cells", "destination_cell": "cells", "template": None, "target": None},
    "deploy": {"source_cell": None, "destination_cell": "cells", "template": "templates", "target": None},
    "end_turn": {"source_cell": None, "destination_cell": None, "template": None, "target": None},
}
```

Validate `neighbor: cells->cells`, `occupies: units->cells`, `has_capability: units|templates->capability_definitions`, allocation owner/definition families, every row range, exact candidate IDs `0..len(candidates)-1`, nonterminal non-empty candidates, and terminal empty candidates.

- [ ] **Step 5: Run GREEN and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_schema.py -q
git add python/ml_lab/tactical_v3_schema.py python/tests/test_tactical_v3_schema.py
git commit -m "feat: parse tactical-v3 structured decisions"
```

Expected: all schema tests pass.

---

### Task 2: Strict GymServer Client and Canonical Project A Fixtures

**Files:**
- Create: `python/ml_lab/tactical_v3_client.py`
- Create: `python/tests/test_tactical_v3_client.py`
- Create: `python/tests/fixtures/tactical_v3/seed-41-spaces.json`
- Create: `python/tests/fixtures/tactical_v3/seed-41-decision.json`
- Create: `python/tests/fixtures/tactical_v3/scenario-24x16.json`

**Interfaces:**
- Consumes: `parse_spaces`, `parse_view`, Project A GymServer commands.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class CandidateSelection:
    decision_id: int
    candidate_id: int

class TacticalV3GymClient:
    def __init__(self, server_cmd: Sequence[str], *, environment_kind: Literal["tactical", "duel"])
    @property
    def identity(self) -> TacticalV3SemanticIdentity
    def reset(self, seed: int) -> TacticalV3View
    def step(self, selection: CandidateSelection) -> TacticalV3View
    def duel_reset(self, seed: int, p0: str, p1: str, learner: int,
                   start_profile: str, reference_seat: int) -> TacticalV3View
    def duel_step(self, selection: CandidateSelection) -> TacticalV3View
    def save_replay(self, path: Path) -> Path
    def close(self) -> None
```

- [ ] **Step 1: Write failing fake-process and real-process tests**

```python
def test_client_sends_both_candidate_identity_fields_and_rejects_stale_reply(fake_server) -> None:
    with TacticalV3GymClient(fake_server.command, environment_kind="tactical") as client:
        view = client.reset(41)
        selected = CandidateSelection(view.decision.decision_id, view.decision.candidates[0].candidate_id)
        client.step(selected)
    assert fake_server.requests[-2] == {
        "cmd": "step", "decision_id": selected.decision_id,
        "candidate_id": selected.candidate_id,
    }

def test_real_server_13x9_and_24x16_share_encoding_not_match_hash(server_dll) -> None:
    standard = spaces_for(server_dll, CHECKED_IN_SCENARIO)
    large = spaces_for(server_dll, FIXTURES / "scenario-24x16.json")
    assert standard.encoding_hash == large.encoding_hash
    assert standard.capacity_hash == large.capacity_hash
    assert standard.contract_hash != large.contract_hash
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_client.py -q
```

Expected: import fails because the structured client is absent.

- [ ] **Step 3: Implement owned JSONL subprocess lifecycle and strict requests**

Launch with `stdin/stdout/stderr` text pipes, UTF-8, `creationflags=no_window_creationflags()`, exact scenario path, `--environment tactical-v3`, and the selected role. Serialize with `allow_nan=False` and compact separators; reject blank/non-object replies. On parser error or EOF, close stdin, terminate/reap the process, and include a bounded stderr tail. `close()` is idempotent and first attempts `{"cmd":"close"}`.

- [ ] **Step 4: Capture canonical fixtures from the freshly built server**

Add a private test helper gated by the exact environment value `HEXWARS_CAPTURE_TACTICAL_V3_FIXTURES=1`; normal test mode must never write. The helper writes only the three named paths with sorted, indented, `allow_nan=False` JSON. Create `scenario-24x16.json` first by changing only `id`, `name`, `board.width=24`, and `board.height=16` while retaining the capacity profile, then capture `spaces` and seed-41 `reset` from the freshly built server. Immediately rerun without the variable so checked-in bytes are validated rather than regenerated.

Run:

```powershell
$env:HEXWARS_CAPTURE_TACTICAL_V3_FIXTURES = "1"
try {
    uv run --active --no-project python -m pytest python/tests/test_tactical_v3_client.py -q
} finally {
    Remove-Item Env:HEXWARS_CAPTURE_TACTICAL_V3_FIXTURES
}
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_schema.py python/tests/test_tactical_v3_client.py -q
```

Expected: the canonical decision contains 117 cells, exact structured tables/candidates, and no `obs`, `mask`, `obs_len`, or `n_actions`; the 24x16 process emits 384 cells.

- [ ] **Step 5: Commit**

```powershell
git add python/ml_lab/tactical_v3_client.py python/tests/test_tactical_v3_client.py python/tests/fixtures/tactical_v3/seed-41-spaces.json python/tests/fixtures/tactical_v3/seed-41-decision.json python/tests/fixtures/tactical_v3/scenario-24x16.json
git commit -m "feat: add tactical-v3 GymServer client"
```

---

### Task 3: Canonical Structured Example and Tiny Immutable Corpus

**Files:**
- Create: `python/ml_lab/tactical_v3_corpus.py`
- Create: `python/tests/test_tactical_v3_corpus.py`
- Create: `python/tests/fixtures/tactical_v3/tiny-corpus/manifest.json`
- Create: `python/tests/fixtures/tactical_v3/tiny-corpus/train.jsonl`
- Create: `python/tests/fixtures/tactical_v3/tiny-corpus/validation.jsonl`
- Create: `python/run_tactical_v3_imitation.py`

**Interfaces:**
- Consumes: validated tactical-v3 decisions and exact schema identities.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TeacherEvidence:
    identity: str
    search_depth: int
    expansion_budget: int
    actual_expansions: int
    heuristic_identity: str
    confidence: float | None

@dataclass(frozen=True, slots=True)
class StructuredTarget:
    teacher_candidate_id: int
    terminal_outcome: Literal["win", "loss", "draw"]
    trajectory_index: int
    remaining_turns_to_victory: int | None
    truncated: bool

@dataclass(frozen=True, slots=True)
class StructuredExample:
    example_schema_version: Literal[1]
    decision: TacticalV3Decision
    target: StructuredTarget
    teacher: TeacherEvidence
    scenario_id: str
    contract_hash: str
    encoding_hash: str
    capacity_hash: str
    profile_id: str
    episode_seed: int
    learner_seat: int

@dataclass(frozen=True, slots=True)
class StructuredCorpus:
    root: Path
    identity: str
    train: tuple[StructuredExample, ...]
    validation: tuple[StructuredExample, ...]

def create_tiny_corpus(output: Path, server_cmd: Sequence[str]) -> StructuredCorpus
def load_corpus(root: Path, expected: TacticalV3SemanticIdentity) -> StructuredCorpus
```

- [ ] **Step 1: Write failing immutability, authenticity, and tamper tests**

```python
def test_tiny_corpus_is_exclusive_content_addressed_and_partitioned(tmp_path: Path, server_cmd) -> None:
    corpus = create_tiny_corpus(tmp_path / "corpus", server_cmd)
    assert 1 <= len(corpus.train) <= 8
    assert 1 <= len(corpus.validation) <= 4
    assert {row.episode_seed for row in corpus.train}.isdisjoint(
        row.episode_seed for row in corpus.validation
    )
    with pytest.raises(FileExistsError):
        create_tiny_corpus(corpus.root, server_cmd)

def test_loader_rejects_changed_row_bytes_even_when_json_is_valid(tiny_corpus_copy: Path) -> None:
    replace_one_candidate_id(tiny_corpus_copy / "train.jsonl")
    with pytest.raises(ValueError, match="SHA-256"):
        load_corpus(tiny_corpus_copy, EXPECTED_IDENTITY)
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_corpus.py -q
```

Expected: import fails because the corpus module is absent.

- [ ] **Step 3: Implement canonical rows and exclusive atomic corpus publication**

Write each row with `json.dumps(..., sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"`. Build in a temporary sibling directory; fsync files; record each file's relative path, byte length, SHA-256, row count, partition, seed list, schema identities, and `label_source: "tiny-fixture-policy-v1"`; reopen/validate the temporary corpus; publish with one directory rename. Reject symlinks/reparse points, extra files, path escapes, duplicate examples, cross-partition seeds, mismatched decision/candidate IDs, and manifest drift.

- [ ] **Step 4: Implement bounded smoke-corpus construction and CLI command**

The command is exact and deliberately narrow:

```powershell
uv run --active --no-project python python/run_tactical_v3_imitation.py build-tiny-corpus --server-dll engine/HexWars.GymServer/bin/Debug/net8.0/HexWars.GymServer.dll --scenario python/config/annihilation-structured-imitation-v1.json --output python/tests/fixtures/tactical_v3/tiny-corpus
```

For train seeds `(4101, 4102)` and validation seed `(5101,)`, run the single-agent learner against Random, choose the first non-`end_turn` candidate (otherwise the sole `end_turn`), retain only the first four learner decisions per seed, finish the deterministic fixture trajectory to obtain its true outcome, and backfill remaining learner decisions to victory only for wins. Evidence is exactly `TeacherEvidence("tiny-fixture-policy-v1", 0, 0, 0, "none", None)`. Refuse any configurable seed/count/teacher option.

- [ ] **Step 5: Capture, reopen, run GREEN, and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_schema.py python/tests/test_tactical_v3_client.py python/tests/test_tactical_v3_corpus.py -q
git add python/ml_lab/tactical_v3_corpus.py python/run_tactical_v3_imitation.py python/tests/test_tactical_v3_corpus.py python/tests/fixtures/tactical_v3/tiny-corpus
git commit -m "feat: add immutable tactical-v3 smoke corpus"
```

Expected: normal test mode authenticates checked-in bytes and never mutates them.

---

### Task 4: Ragged Typed Batching, Reference Remapping, and Masks

**Files:**
- Create: `python/ml_lab/tactical_v3_batching.py`
- Create: `python/tests/test_tactical_v3_batching.py`

**Interfaces:**
- Consumes: `StructuredExample` tuples.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class TokenTableBatch:
    numeric: torch.Tensor       # float32 [B, N, F]
    categorical: Mapping[str, torch.Tensor]  # int64 [B, N]
    boolean: Mapping[str, torch.Tensor]      # bool [B, N]
    mask: torch.Tensor          # bool [B, N]

@dataclass(frozen=True, slots=True)
class RelationNeighborhoodBatch:
    source_index: torch.Tensor  # int64 [B, Nnode, K]
    kind: torch.Tensor          # int64 [B, Nnode, K]
    int_feature: torch.Tensor   # int64 [B, Nnode, K]
    float_feature: torch.Tensor # float32 [B, Nnode, K]
    bool_feature: torch.Tensor  # bool [B, Nnode, K]
    mask: torch.Tensor          # bool [B, Nnode, K]

@dataclass(frozen=True, slots=True)
class CandidateBatch:
    candidate_id: torch.Tensor  # int64 [B, C]
    decision_id: torch.Tensor   # int64 [B, C]
    kind: torch.Tensor          # int64 [B, C]
    reference_index: torch.Tensor  # int64 [B, C, 8]
    reference_mask: torch.Tensor   # bool [B, C, 8]
    projection_integer: torch.Tensor  # int64 [B, C, 7]
    projection_boolean: torch.Tensor  # bool [B, C, 2]
    mask: torch.Tensor          # bool [B, C]

@dataclass(frozen=True, slots=True)
class RaggedBatch:
    tables: Mapping[str, TokenTableBatch]
    table_slices: Mapping[str, slice]
    node_mask: torch.Tensor
    neighborhoods: RelationNeighborhoodBatch
    candidates: CandidateBatch
    teacher_candidate_index: torch.Tensor
    terminal_outcome: torch.Tensor
    horizon_targets: torch.Tensor
    horizon_target_mask: torch.Tensor
    remaining_turns: torch.Tensor
    remaining_turns_mask: torch.Tensor

def collate_examples(examples: Sequence[StructuredExample], horizons: tuple[int, ...]) -> RaggedBatch
```

- [ ] **Step 1: Write failing mixed-size, remapping, padding, and malformed-target tests**

```python
def test_collate_remaps_every_reference_into_masked_global_nodes() -> None:
    batch = collate_examples([EXAMPLE_13X9, EXAMPLE_24X16], horizons=(4, 8, 16))
    assert batch.tables["cells"].mask.sum(dim=1).tolist() == [117, 384]
    for sample_index in range(2):
        valid = batch.candidates.reference_index[sample_index][
            batch.candidates.reference_mask[sample_index]
        ]
        assert torch.all(valid >= 0)
        assert torch.all(valid < batch.node_mask.shape[1])
        assert batch.node_mask[sample_index, valid].all()

@pytest.mark.parametrize("failure", ["invalid_ref", "teacher_missing", "nan", "all_masked"])
def test_collate_fails_closed_before_returning_tensors(failure: str) -> None:
    with pytest.raises(ValueError, match=FAILURE_TEXT[failure]):
        collate_examples([broken_example(failure)], horizons=(4, 8, 16))
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_batching.py -q
```

Expected: import fails because the batching module is absent.

- [ ] **Step 3: Implement typed table encoding and per-batch global reference maps**

Use batch-local padded maxima only. Concatenate table regions in the exact table order, build `(sample, table, row) -> global_index`, map absent optional references to index zero with `reference_mask=False`, and assert every present reference lands on a `node_mask=True` row. Center `q/r` over valid cells and divide by `max(1, max(abs(centered_q), abs(centered_r)))`; never expose raw row indices as numeric features.

- [ ] **Step 4: Build deterministic incoming neighborhoods without scatter**

Sort edges by `(destination_global_index, relation_kind, source_global_index, int_feature, float_feature, bool_feature)`, add reverse kinds explicitly, add derived allocation-owner and allocation-definition edges, pad each destination's incoming list to batch maximum `K`, and retain a boolean mask. Valid zero-degree nodes receive one masked dummy slot, never a semantic self-edge.

- [ ] **Step 5: Encode targets and run GREEN**

Map candidate identity to its row only after proving an exact unique match. Outcome indices are `loss=0, draw=1, win=2`. For horizon `h`, set a valid binary target only when the episode result and censoring permit it. Set `remaining_turns_mask=True` exactly for nontruncated wins with a defined positive remaining value.

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_batching.py -q
git add python/ml_lab/tactical_v3_batching.py python/tests/test_tactical_v3_batching.py
git commit -m "feat: batch ragged tactical-v3 decisions"
```

---
### Task 5: Typed Encoders, Local Hex Message Passing, and Relational Attention

**Files:**
- Create: `python/ml_lab/tactical_v3_layers.py`
- Create: `python/tests/test_tactical_v3_layers.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True, slots=True)
class TacticalV3ModelConfig:
    hidden_dim: int = 64
    categorical_dim: int = 16
    cell_message_rounds: int = 2
    relation_rounds: int = 2
    attention_heads: int = 4
    feed_forward_dim: int = 128
    candidate_hidden_dim: int = 128
    horizon_turns: tuple[int, ...] = (4, 8, 16)

class TypedTokenEncoders(nn.Module):
    def forward(self, batch: RaggedBatch) -> Tensor: ...

class LocalHexMessagePassing(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        cell_neighbor_index: Tensor,
        cell_neighbor_mask: Tensor,
        node_mask: Tensor,
    ) -> Tensor: ...

class TypedRelationalAttention(nn.Module):
    def forward(
        self,
        node_state: Tensor,
        incoming_source_index: Tensor,
        incoming_relation_kind: Tensor,
        incoming_int_feature: Tensor,
        incoming_float_feature: Tensor,
        incoming_bool_feature: Tensor,
        incoming_mask: Tensor,
        node_mask: Tensor,
    ) -> Tensor: ...
```

The table encoder owns distinct categorical embeddings and numeric projections per `TableKind`; booleans remain two-valued categories. `LocalHexMessagePassing` uses only the six topology neighbors supplied by the batch. `TypedRelationalAttention` gathers padded incoming sources, adds relation-kind and edge-feature encodings, and writes only valid destinations. Use stock PyTorch gather/reshape operations; do not add PyG, `torch_scatter`, or another graph dependency.

- [ ] **Step 1: Write the failing encoder tests**

```python
def test_centered_coordinates_are_translation_invariant() -> None: ...
def test_local_hex_layer_is_equivariant_to_cell_row_permutation() -> None: ...
def test_relational_layer_is_equivariant_to_typed_table_row_permutations() -> None: ...
def test_padding_cannot_change_any_valid_node_embedding() -> None: ...
def test_masked_neighbors_cannot_change_any_valid_destination() -> None: ...
def test_every_layer_rejects_nonfinite_inputs_before_attention() -> None: ...
```

Build each permuted batch by permuting the row and every reference to that row together. Run identical weights, undo the output permutation, and compare valid rows with `torch.testing.assert_close(..., rtol=0.0, atol=1e-6)`. For padding invariance, append extreme finite values under false masks; valid outputs must remain within the same tolerance.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_layers.py -q
```

Expected: collection fails because `tactical_v3_layers` is absent.

- [ ] **Step 3: Implement typed input encoders**

Create one path per `TableKind`, concatenate its fixed-width encoded fields, and project to `hidden_dim`. Encode `q/r` only through Task 4's centered/scaled coordinates. Validate `hidden_dim % attention_heads == 0`, positive dimensions/rounds, and strictly increasing positive horizons. Multiply every encoded table by its row mask immediately after projection and every residual block.

- [ ] **Step 4: Implement local and relational updates**

For each configured round, gather actual hex neighbors; compute safe masked mean/max summaries; update cell states with a shared residual MLP; gather typed incoming sources; add relation-kind and integer/float/bool edge encodings; apply masked multi-head attention and a masked feed-forward residual; then zero padded rows. An all-masked incoming set produces a finite zero message before the residual. Never use row-order board arithmetic, semantic dummy self-edges, in-place scatter, or fixed table capacities.

- [ ] **Step 5: Run GREEN and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_batching.py python/tests/test_tactical_v3_layers.py -q
git add python/ml_lab/tactical_v3_layers.py python/tests/test_tactical_v3_layers.py
git commit -m "feat: encode tactical-v3 relational state"
```

---
### Task 6: Shared Variable-Candidate Policy and Auxiliary Heads

**Files:**
- Create: `python/ml_lab/tactical_v3_model.py`
- Create: `python/tests/test_tactical_v3_model.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True, slots=True)
class PolicyOutput:
    candidate_logits: Tensor       # [batch, candidate]
    outcome_logits: Tensor         # [batch, 3], loss/draw/win
    horizon_logits: Tensor         # [batch, len(config.horizon_turns)]
    remaining_turns: Tensor        # [batch]

@dataclass(frozen=True, slots=True)
class CandidateIdentity:
    decision_id: int
    candidate_id: int

class TacticalV3Policy(nn.Module):
    config: TacticalV3ModelConfig

    def forward(self, batch: RaggedBatch) -> PolicyOutput: ...

    @torch.inference_mode()
    def select(self, batch: RaggedBatch) -> tuple[CandidateIdentity, ...]: ...
```

- [ ] **Step 1: Write failing policy tests**

```python
def test_candidate_permutation_permutes_logits_and_preserves_identity_selection() -> None: ...
def test_candidate_padding_cannot_change_valid_logits_or_argmax() -> None: ...
def test_batching_beside_24x16_cannot_change_13x9_logits_or_action() -> None: ...
def test_softmax_probability_is_zero_on_padding_and_sums_to_one_on_valid_rows() -> None: ...
def test_state_table_permutations_leave_candidate_logits_unchanged() -> None: ...
def test_projection_reference_changes_affect_only_the_referenced_candidate_path() -> None: ...
def test_all_masked_candidate_rows_raise_before_argmax() -> None: ...
def test_nonfinite_logits_raise_before_selection() -> None: ...
def test_selected_candidate_is_an_exact_member_of_each_input_candidate_set() -> None: ...
```

Use a deterministic tie and require the smallest `candidate_id`, not the first padded row, as the tie-break. Exercise candidate counts 1, 3, and 19 in one batch; no constructor argument may encode a maximum action count.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_model.py -q
```

Expected: collection fails because `tactical_v3_model` is absent.

- [ ] **Step 3: Implement state and candidate projection paths**

Pool each typed table with mask-aware mean plus max, concatenate per-table summaries with match/start-profile scalars, and project to a shared state vector. Build each candidate from its kind/scalars; gathered source/target unit, cell, allocation, definition, and tile embeddings; typed projection edge/value embeddings; and explicit presence bits for every optional reference. Use one candidate MLP for all kinds and rows. Do not flatten tables or candidate axes into model parameters.

- [ ] **Step 4: Implement scoring, heads, and selection**

Score every candidate through the same MLP over `[state, candidate, state * candidate]`; set padded logits to `-inf` only at the public output boundary. Outcome, horizon, and remaining-turn heads read the shared state. Reject a nonfinite valid logit/head, and reject samples with no valid candidates before argmax. Resolve ties by `(-logit, candidate_id)`; return the exact paired `decision_id/candidate_id` carried by the batch.

- [ ] **Step 5: Run GREEN and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_layers.py python/tests/test_tactical_v3_model.py -q
git add python/ml_lab/tactical_v3_model.py python/tests/test_tactical_v3_model.py
git commit -m "feat: score tactical-v3 legal candidates"
```

---
### Task 7: Policy, Outcome, Horizon, and Remaining-Turn Objectives

**Files:**
- Create: `python/ml_lab/tactical_v3_objectives.py`
- Create: `python/tests/test_tactical_v3_objectives.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True, slots=True)
class ObjectiveConfig:
    policy_coefficient: float = 1.0
    outcome_coefficient: float = 0.2
    horizon_coefficient: float = 0.2
    remaining_turns_coefficient: float = 0.1

@dataclass(frozen=True, slots=True)
class LossBreakdown:
    total: Tensor
    policy: Tensor
    outcome: Tensor
    horizon: Tensor
    remaining_turns: Tensor

def structured_imitation_loss(
    output: PolicyOutput,
    batch: RaggedBatch,
    config: ObjectiveConfig,
) -> LossBreakdown: ...
```

- [ ] **Step 1: Write failing objective tests**

```python
def test_auxiliary_coefficient_sum_must_not_exceed_policy_coefficient() -> None: ...
def test_policy_loss_ignores_padding_and_matches_manual_cross_entropy() -> None: ...
def test_outcome_loss_uses_loss_draw_win_target_order() -> None: ...
def test_horizon_loss_uses_only_uncensored_target_mask() -> None: ...
def test_remaining_turns_loss_uses_only_nontruncated_wins() -> None: ...
def test_empty_auxiliary_masks_produce_differentiable_finite_zeroes() -> None: ...
def test_nonfinite_component_or_total_raises() -> None: ...
def test_default_loss_backpropagates_finite_scorer_and_encoder_gradients() -> None: ...
```

Assert the default auxiliary coefficient sum is `0.5 <= 1.0`. Reject negative or nonfinite coefficients, a nonpositive policy coefficient, and `outcome + horizon + remaining_turns > policy`.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_objectives.py -q
```

Expected: collection fails because `tactical_v3_objectives` is absent.

- [ ] **Step 3: Implement masked objectives**

Use categorical cross-entropy over valid candidate logits for policy and fixed `loss/draw/win` order for outcome. Use BCE-with-logits only where `horizon_target_mask=True`, and Smooth L1 only where `remaining_turns_mask=True`. An empty auxiliary mask returns `output_tensor.sum() * 0.0`, preserving device and autograd. Validate target shapes/ranges and `torch.isfinite` for every component and the weighted total.

- [ ] **Step 4: Run GREEN and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_objectives.py -q
git add python/ml_lab/tactical_v3_objectives.py python/tests/test_tactical_v3_objectives.py
git commit -m "feat: add tactical-v3 imitation objectives"
```

---
### Task 8: Deterministic Offline Trainer, Finite Checks, and Early Stopping

**Files:**
- Create: `python/ml_lab/tactical_v3_training.py`
- Create: `python/tests/test_tactical_v3_training.py`
- Modify: `python/run_tactical_v3_imitation.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True, slots=True)
class TrainerConfig:
    seed: int = 227
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_epochs: int = 400
    patience_epochs: int = 100
    gradient_clip_norm: float = 1.0
    device: str = "cpu"

@dataclass(frozen=True, slots=True)
class EpochMetrics:
    epoch: int
    train: Mapping[str, float]
    validation: Mapping[str, float]
    validation_policy_nll: float
    improved: bool

@dataclass(frozen=True, slots=True)
class TrainingResult:
    model: TacticalV3Policy
    best_epoch: int
    best_validation_policy_nll: float
    stopped_early: bool
    history: tuple[EpochMetrics, ...]

def train_offline(
    train_examples: tuple[StructuredExample, ...],
    validation_examples: tuple[StructuredExample, ...],
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
) -> TrainingResult: ...
```

- [ ] **Step 1: Write failing trainer tests**

```python
def test_two_cpu_runs_have_identical_history_weights_logits_and_actions() -> None: ...
def test_validation_examples_are_never_seen_by_optimizer_or_shuffle() -> None: ...
def test_best_validation_policy_nll_state_is_restored() -> None: ...
def test_patience_stops_after_exact_number_of_nonimproving_epochs() -> None: ...
def test_nonfinite_loss_fails_before_backward() -> None: ...
def test_nonfinite_gradient_fails_before_optimizer_step() -> None: ...
def test_nonfinite_parameter_fails_immediately_after_optimizer_step() -> None: ...
def test_train_rejects_empty_or_overlapping_splits() -> None: ...
```

Instrument a tiny model/optimizer in the finite tests and assert its parameters are byte-identical before and after every rejected step. Compare two independent successful CPU runs with exact history dictionaries and `torch.equal` state tensors; compare logits and selected identities too.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_training.py -q
```

Expected: collection fails because `tactical_v3_training` is absent.

- [ ] **Step 3: Implement deterministic ownership**

Validate all trainer fields. Seed Python, NumPy, and PyTorch; enable `torch.use_deterministic_algorithms(True)`; configure deterministic cuDNN when CUDA is requested. Use a private seeded `torch.Generator` for epoch permutations, canonical corpus order before shuffling, and `DataLoader(num_workers=0)`. Construct `AdamW(..., weight_decay=0.0)`. Never mutate, re-split, or shuffle the frozen validation tuple.

- [ ] **Step 4: Implement guarded training and early stopping**

For each batch, check model outputs and total/component losses before backward, every gradient before clipping, clipped norm, and every parameter after `optimizer.step()`. Raise a contextual `FloatingPointError` with epoch/batch/tensor name at the first failure. Evaluate without gradients in canonical validation order. The selection metric is mean validation policy NLL only; auxiliary losses remain logged diagnostics. Treat improvement as strictly more than `1e-12`, snapshot a detached CPU clone of every state tensor, stop after exactly `patience_epochs` consecutive nonimproving epochs, then restore the best snapshot.

- [ ] **Step 5: Add the narrow train CLI and run GREEN**

Add `train --corpus --run-dir --seed --device`; all architecture/objective/trainer defaults come from the frozen configuration classes and are written to metadata. Do not add collection, DAgger, curriculum, evaluation, or game-play flags.

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_training.py -q
uv run --active --no-project python python/run_tactical_v3_imitation.py train --help
git add python/ml_lab/tactical_v3_training.py python/tests/test_tactical_v3_training.py python/run_tactical_v3_imitation.py
git commit -m "feat: train tactical-v3 policy deterministically"
```

---
### Task 9: Strict Checkpoints, Deterministic Save/Load, and CPU Publication

**Files:**
- Create: `python/ml_lab/tactical_v3_checkpoint.py`
- Create: `python/tests/test_tactical_v3_checkpoint.py`
- Modify: `python/run_tactical_v3_imitation.py`

**Interfaces introduced in this task:**

```python
@dataclass(frozen=True, slots=True)
class StructuredCheckpointMetadata:
    format_version: int
    algorithm: Literal["structured_imitation"]
    identity: TacticalV3SemanticIdentity
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig
    corpus_sha256: str
    model_state_sha256: str
    best_epoch: int
    best_validation_policy_nll: float
    published_device: Literal["cpu"]

@dataclass(frozen=True, slots=True)
class StructuredInferenceFixture:
    examples: tuple[StructuredExample, ...]
    valid_candidate_logits: tuple[tuple[float, ...], ...]
    selected_identities: tuple[CandidateIdentity, ...]

@dataclass(frozen=True, slots=True)
class LoadedStructuredPolicy:
    model: TacticalV3Policy
    metadata: StructuredCheckpointMetadata
    fixture: StructuredInferenceFixture

def save_structured_checkpoint(
    path: Path,
    model: TacticalV3Policy,
    metadata: StructuredCheckpointMetadata,
    fixture_examples: tuple[StructuredExample, ...],
) -> Path: ...

def load_structured_checkpoint(
    path: Path,
    expected_encoding_hash: str,
    expected_capacity_hash: str,
) -> LoadedStructuredPolicy: ...

def publish_structured_run(
    run_dir: Path,
    result: TrainingResult,
    corpus: StructuredCorpus,
    scenario_path: Path,
) -> Path: ...

def validate_structured_run(run_dir: Path) -> LoadedStructuredPolicy: ...
```

- [ ] **Step 1: Write failing checkpoint and publication tests**

```python
def test_checkpoint_contains_only_whitelisted_plain_values_and_cpu_tensors() -> None: ...
def test_load_uses_weights_only_and_rejects_unknown_missing_or_wrong_typed_keys() -> None: ...
def test_load_rejects_wrong_encoding_capacity_corpus_or_state_hash() -> None: ...
def test_two_cpu_saves_are_semantically_identical_after_strict_load() -> None: ...
def test_cpu_save_load_preserves_logits_and_actions_exactly() -> None: ...
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_training_publication_matches_cpu_within_declared_tolerance() -> None: ...
def test_publish_is_atomic_refuses_overwrite_and_leaves_run_unsealed() -> None: ...
def test_validate_run_replays_immutable_inference_fixture() -> None: ...
```

The CUDA test trains the same tiny corpus on CUDA, publishes CPU tensors, and compares CPU-reloaded valid logits with `rtol=1e-5, atol=1e-6`; selected candidate identities must be exact. CPU round trips require `rtol=0.0, atol=0.0` and exact actions.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_checkpoint.py -q
```

Expected: collection fails because `tactical_v3_checkpoint` is absent.

- [ ] **Step 3: Implement a strict weights-only format**

Write exactly four top-level keys: `format_version`, `metadata`, `state_dict`, and `inference_fixture`. Metadata and fixtures contain only recursively validated `dict[str, plain scalar/list/dict]` values; state values are detached, contiguous CPU tensors. Load with `torch.load(..., map_location="cpu", weights_only=True)`, recursively reject unknown/missing keys and non-whitelisted types, reconstruct frozen DTO/config objects, instantiate from saved model config, and call `load_state_dict(..., strict=True)`.

Compute `model_state_sha256` over sorted parameter name, dtype, shape, and contiguous bytes, not over container serialization bytes. Save by exclusive temporary file plus `os.replace`; reject a pre-existing final path. After load, re-hash state, re-batch the stored structured fixtures, require exact CPU logits/actions, and require exact expected encoding/capacity hashes before returning the model.

- [ ] **Step 4: Publish the experimental run without sealing**

Publish a new directory containing `run.json`, `scenario.json`, `corpus-manifest.json`, `metrics.jsonl`, `inference-fixture.json`, and `checkpoints/best.pt`. Build the manifest fields from validated values rather than string templates:

```python
run_manifest = {
    "schema_version": 1,
    "state": "completed",
    "evidence_status": "unsealed-experimental",
    "config": {"algorithm": "structured_imitation"},
    "contract": metadata.identity.to_manifest_dict(),
    "latest_checkpoint": "checkpoints/best.pt",
    "latest_checkpoint_step": metadata.best_epoch,
    "dataset_manifest_sha256": metadata.corpus_sha256,
    "best_epoch": metadata.best_epoch,
    "best_validation_policy_nll": metadata.best_validation_policy_nll,
}
```

`to_manifest_dict()` includes `environment="tactical-v3"`, `version="tactical-v3"`, the role-valued `environment_kind`, `contract_hash`, `encoding_hash`, and `capacity_hash`. Copy and reparse the scenario with canonical JSON. Refuse an existing run directory, symlinks, non-relative manifest artifacts, hash disagreement, or any evidence status other than `unsealed-experimental`. Publish all files in a sibling temporary directory, validate there, then atomically rename once.

- [ ] **Step 5: Wire CLI validation, run GREEN, and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_checkpoint.py -q
uv run --active --no-project python python/run_tactical_v3_imitation.py validate-run --help
git add python/ml_lab/tactical_v3_checkpoint.py python/tests/test_tactical_v3_checkpoint.py python/run_tactical_v3_imitation.py
git commit -m "feat: publish tactical-v3 policy checkpoints"
```

---
### Task 10: Structured Controller Resolution and Policy-Server Identity Routing

**Files:**
- Create: `python/ml_lab/tactical_v3_controller.py`
- Create: `python/tests/test_tactical_v3_controller.py`
- Modify: `python/ml_lab/controllers.py`
- Modify: `python/policy_server.py`
- Modify: `python/tests/test_controllers.py`
- Modify: `python/tests/test_policy_server.py`

**Interfaces introduced or extended in this task:**

```python
# tactical_v3_controller.py
@dataclass(frozen=True, slots=True)
class StructuredController:
    run_dir: Path
    checkpoint_path: Path
    policy: TacticalV3Policy
    identity: TacticalV3SemanticIdentity

def load_structured_controller(
    run_dir: Path,
    expected_encoding_hash: str,
    expected_capacity_hash: str,
) -> StructuredController: ...

def select_candidate(
    controller: StructuredController,
    view: TacticalV3View,
) -> CandidateIdentity: ...

# controllers.py
Algorithm = Literal["maskable_ppo", "masked_dqn", "structured_imitation"]
ControllerContract = EnvironmentContract | TacticalV3SemanticIdentity

# policy_server.py
@dataclass(frozen=True, slots=True)
class PolicyExpectation:
    environment: str
    version: str
    encoding_hash: str
    capacity_hash: str | None = None
```

- [ ] **Step 1: Write failing controller and server tests**

Add focused tests for manifest-only loading, `.pt` containment, exact encoding/capacity matching before tensor load, forced CPU/eval/inference mode, deterministic logits/actions after two loads, and exact legal candidate identity round-trip. Extend existing tests to prove the two legacy algorithms still require `.zip` plus fixed observation/action geometry.

Add policy-server subprocess tests for:

```python
def test_tactical_v3_requires_expected_capacity_hash() -> None: ...
def test_legacy_expectation_rejects_capacity_hash() -> None: ...
def test_tactical_v3_request_returns_decision_and_candidate_identity() -> None: ...
def test_tactical_v3_request_rejects_flat_or_mixed_payloads() -> None: ...
def test_structured_response_is_deterministic_across_server_restarts() -> None: ...
def test_wrong_encoding_or_capacity_fails_before_ready() -> None: ...
```

The structured input has exactly `{"seat": int, "decision": object}`; the output has exactly `{"decision_id": int, "candidate_id": int}`. Legacy input/output remain exactly `{"seat","obs","mask"}` and `{"action"}`.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_controller.py python/tests/test_controllers.py python/tests/test_policy_server.py -q
```

Expected: tactical-v3 imports/assertions fail while existing legacy cases remain green.

- [ ] **Step 3: Add structured run dispatch without SB3 geometry**

Recognize `structured_imitation` only from an authoritative `run.json`; do not add an alias or permit standalone `.pt` controller specs. In `ControllerResolver._resolve_run`, branch before the legacy suffix/geometry path: require a contained `checkpoints/*.pt`, parse the structured semantic identity including capacity hash, call `load_structured_controller`, and leave legacy `observation_size/action_size` unset. Keep `_model_geometry`, Gymnasium spaces, and SB3 loaders unreachable for this algorithm. Include algorithm and all three semantic hashes in ready metadata.

- [ ] **Step 4: Route the strict structured protocol**

Accept `--expected-capacity-hash`. Tactical-v3 expectations require it; legacy expectations reject it. `validate_resolved_contract` compares environment, version, encoding hash, and capacity hash before ready. After strict JSON DTO parsing, require request seat to match `view.seat`, batch one view, call `select_candidate`, prove the result's decision id equals the request and candidate id exists exactly once in `view.candidates`, then serialize the identity response. Preserve reload semantics and validate a live replacement completely before swapping.

- [ ] **Step 5: Run GREEN and commit**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_controller.py python/tests/test_controllers.py python/tests/test_policy_server.py -q
git add python/ml_lab/tactical_v3_controller.py python/ml_lab/controllers.py python/policy_server.py python/tests/test_tactical_v3_controller.py python/tests/test_controllers.py python/tests/test_policy_server.py
git commit -m "feat: load tactical-v3 structured controllers"
```

---
### Task 11: Engine Transition Drain and Unity Tactical-v3 Arena Bridge

**Files:**
- Modify: `engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs`
- Modify: `engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs`
- Modify: `Assets/HexWars/Presentation/MlEnvironmentContract.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelEnvironment.cs`
- Create: `Assets/HexWars/Presentation/TacticalV3PolicyPayload.cs`
- Create: `Assets/HexWars/Presentation/TacticalV3PolicyPayload.cs.meta`
- Modify: `Assets/HexWars/Presentation/PolicyBridge.cs`
- Modify: `Assets/HexWars/Presentation/ModelDuelDriver.cs`
- Modify: `Assets/HexWars/Presentation/ModelArenaIdentity.cs`
- Create: `Assets/HexWars/Tests/Editor/TacticalV3PolicyPayloadTests.cs`
- Create: `Assets/HexWars/Tests/Editor/TacticalV3PolicyPayloadTests.cs.meta`
- Modify: `Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs`

**Interfaces introduced or refactored in this task:**

```csharp
// TacticalV3DuelEnv.cs
public IReadOnlyList<DuelTransition> DrainTransitions();

// ModelDuelEnvironment.cs
public interface IModelDuelEnvironment
{
    MlEnvironmentContract Environment { get; }
    ModelDuelContractIdentity ContractIdentity { get; }
    GameState CurrentState { get; }
    bool CaptureTransitions { get; set; }
    ModelDuelView Reset(int seed, IAgent controller0, IAgent controller1);
    ModelDuelView Continue();
    IReadOnlyList<DuelTransition> DrainTransitions();
}

public interface ILegacyModelDuelEnvironment : IModelDuelEnvironment
{
    MlContract Contract { get; }
    ModelDuelView Step(int action);
}

public interface IStructuredModelDuelEnvironment : IModelDuelEnvironment
{
    TacticalV3Contract StructuredContract { get; }
    ModelDuelView Step(long decisionId, int candidateId);
}

// TacticalV3PolicyPayload.cs
[Serializable]
public sealed class TacticalV3PolicyRequestDto
{
    public int seat;
    public TacticalV3ViewDto decision;
}

[Serializable]
public sealed class TacticalV3ViewDto
{
    public long decision_id;
    public int seat;
    public TacticalV3ObservationDto observation;
    public TacticalV3CandidateDto[] candidates;
    public TacticalV3RewardDto reward;
    public int winner;
    public bool terminated;
    public bool truncated;
    public string start_profile;
    public int reference_seat;
}

public static class TacticalV3PolicyPayload
{
    public static TacticalV3ViewDto From(TacticalV3View view);
}

// PolicyBridge.cs
public PolicyCandidateResult ActStructured(int seat, TacticalV3ViewDto decision);
```

`TacticalV3ObservationDto`, candidate/projection DTOs, and reward DTO use the exact snake_case fields and scalar/array types enumerated in ?Resolved Project A Wire Contract?; there is no generic dictionary, reflection serializer, flattened observation, or mask.

- [ ] **Step 1: Write engine and Unity RED tests**

Engine tests assert drain-before-reset fails; reset starts empty; accepted external and scripted commands appear exactly once in the next drain; a second drain is empty; draining never changes `ToReplay()`, state, decision id, or truncation count; reset clears both history and cursor.

Unity EditMode tests assert `MlEnvironmentContracts.Parse("tactical-v3")`; semantic identity includes capacity hash; factory builds a tactical-v3 structured adapter without legacy `MlContract` geometry; payload JSON matches the checked-in seed-41 wire fixture's exact keys/types; all observation references/candidate projections survive; and `PolicyBridge` parses only a matching `decision_id/candidate_id`. Preserve all legacy bridge/configuration tests.

- [ ] **Step 2: Run RED**

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV3DuelEnvTests" --no-restore
```

Then run the Unity EditMode tests through Coplay. Expected: the drain and tactical-v3 presentation interfaces are absent.

- [ ] **Step 3: Add a non-destructive transition cursor**

Add `_drainedTransitionCount`. Reset it to zero whenever `_transitions` is cleared. `DrainTransitions()` snapshots `_transitions[_drainedTransitionCount..]`, advances only the cursor, and never removes replay history. Keep decision ids, maximum-step accounting, and `ToReplay()` based on the full transition list. Return detached `DuelTransition` snapshots under the same mutation-safety rules as existing environment state.

- [ ] **Step 4: Add the structured Unity adapter and exact DTO projection**

Add `TacticalV3` to `MlEnvironmentContract`, add optional `CapacityHash` to `ModelDuelContractIdentity` and `PolicySeatInfo`, and split legacy fixed-geometry versus structured stepping through the interfaces above. `TacticalV3ModelDuelEnvironment` owns `TacticalV3DuelEnv`, returns its `TacticalV3View` in `ModelDuelView`, throws on legacy integer action stepping, and delegates identity stepping exactly to `Step(long,int)`. Existing adapters implement `ILegacyModelDuelEnvironment` unchanged semantically.

Project every Project A field explicitly into serializable DTO arrays. Validate non-null tables, finite floats, unique/in-range references, exact candidate-id row identity, and `view.Seat == view.Decision.Seat` before serialization. Keep Unity field names equal to wire names rather than relying on naming conversion.

- [ ] **Step 5: Route candidate identities through the bridge and driver**

Pass capacity hash to `policy_server.py` only for tactical-v3. `ActStructured` writes `{"seat":seat,"decision":...}`, parses `decision_id` as `long` and `candidate_id` as `int`, rejects error/missing/extra identity fields, and requires the returned decision id to equal the sent decision. In `ModelDuelDriver`, branch on `IStructuredModelDuelEnvironment`, prove candidate membership before `Step`, and retain the existing legacy `Act/Step(int)` route. Label `structured_imitation` as a structured PyTorch policy in Arena identity text.

- [ ] **Step 6: Run GREEN, sync the engine DLL, check Unity, and commit**

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter "FullyQualifiedName~TacticalV3DuelEnvTests|FullyQualifiedName~TacticalV2DuelEnvTests|FullyQualifiedName~ReplayTests|FullyQualifiedName~ReplayFileTests" --no-restore
powershell -ExecutionPolicy Bypass -File engine/build-to-unity.ps1
```

Use Coplay against this exact worktree: `set_unity_project_root`, wait for import, `check_compile_errors`, run `TacticalV3PolicyPayloadTests`, `PolicyBridgeProtocolTests`, and `ModelDuelConfigurationTests`, then `get_unity_logs`. Expected: zero compile/test errors and no new tactical-v3 exceptions.

```powershell
git add engine/HexWars.Engine/Rl/TacticalV3DuelEnv.cs engine/HexWars.Engine.Tests/TacticalV3DuelEnvTests.cs Assets/HexWars/Presentation/MlEnvironmentContract.cs Assets/HexWars/Presentation/ModelDuelEnvironment.cs Assets/HexWars/Presentation/TacticalV3PolicyPayload.cs Assets/HexWars/Presentation/TacticalV3PolicyPayload.cs.meta Assets/HexWars/Presentation/PolicyBridge.cs Assets/HexWars/Presentation/ModelDuelDriver.cs Assets/HexWars/Presentation/ModelArenaIdentity.cs Assets/HexWars/Tests/Editor/TacticalV3PolicyPayloadTests.cs Assets/HexWars/Tests/Editor/TacticalV3PolicyPayloadTests.cs.meta Assets/HexWars/Tests/Editor/PolicyBridgeProtocolTests.cs Assets/HexWars/Tests/Editor/ModelDuelConfigurationTests.cs
git commit -m "feat: run tactical-v3 policies in Unity Arena"
```

---
### Task 12: ML Lab Tactical-v3 Scenario and Experimental Run Loading

**Files:**
- Modify: `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- Modify: `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabConfigTests.cs`
- Modify: `Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs`

**Interfaces introduced or extended in this task:**

```csharp
public sealed class MlTrainingScenario
{
    public MlTacticalV3Reward TacticalV3Reward { get; set; }
    public MlTrainingTacticalV3 TacticalV3 { get; set; }
}

public sealed class MlTacticalV3Reward
{
    public float TerminalWin { get; set; }
    public float TerminalNonWin { get; set; }
    public float MaterialAdjustmentBound { get; set; }
    public float TimePressureBound { get; set; }
    public float PointsWeight { get; set; }
}

public sealed class MlTrainingTacticalV3
{
    public int StartingUnitCount { get; set; }
    public int MaxControllableUnits { get; set; }
    public string PlacementPolicy { get; set; }
    public MlTrainingTacticalV3Capacity Capacity { get; set; }
    public List<MlTrainingUnitTemplate> Templates { get; set; }
    public List<MlTrainingTacticalV2StartProfile> StartProfiles { get; set; }
    public List<MlTrainingTacticalV2StartWeight> StartDistribution { get; set; }
}

public sealed class MlTrainingTacticalV3Capacity
{
    public int MaxCells { get; set; }
    public int MaxUnits { get; set; }
    public int MaxTemplates { get; set; }
    public int MaxCapabilityDefinitions { get; set; }
    public int MaxCapabilityAllocations { get; set; }
    public int MaxRules { get; set; }
    public int MaxMemoryRecords { get; set; }
    public int MaxRelations { get; set; }
    public int MaxCandidates { get; set; }
}

public sealed class MlTrainingScenarioPreflight
{
    public bool UsesStructuredCandidates { get; }
    public ModelDuelContractIdentity ContractIdentity { get; }
    public int? ObservationSize { get; }
    public int? ActionSize { get; }
    public static TrainingScenario ToEngine(MlTrainingScenario scenario);
}
```

- [ ] **Step 1: Write failing strict-scenario, run-loading, and Train-tab tests**

Load `python/config/annihilation-structured-imitation-v1.json` and assert all reward, capacity, template, profile, distribution, and board values survive `Parse -> Serialize -> Parse -> ToEngine`. Assert exact-key validation rejects a tactical-v3 scenario with any missing/extra/mistyped/nonfinite field, invalid capacity, nonzero fog, unsupported unit/design section, invalid profile weight sum, or mismatched section for another environment.

Add tests that an Arena run with algorithm `structured_imitation`, `.pt` checkpoint, scenario copy, matching contract/encoding/capacity hashes, and `evidence_status="unsealed-experimental"` produces `MlArenaLaunchPlan`; wrong/missing hashes, any other evidence status, traversal, or a fixed-geometry contract fail. Assert ML Lab configuration validation and `BuildTrainArguments` explicitly reject tactical-v3 rather than generating an SB3 command. Existing v1/v2/adaptive scenario, Train, and Arena tests must remain unchanged.

- [ ] **Step 2: Run RED**

Run the three affected EditMode test classes with Coplay. Expected: tactical-v3 parsing/loading cases fail, while legacy regression cases remain green.

- [ ] **Step 3: Extend strict JSON and engine conversion**

Add tactical-v3 exact-key sets for top level, `reward`, `tactical_v3`, and `capacity`. Reuse the existing template/profile/distribution value types only where their JSON semantics are identical. Require reward finiteness and documented bounds, all nine positive capacity values, `max_controllable_units == starting_unit_count`, valid profiled placement, unique ids, distribution total 10,000 basis points, fog/biomes false, and absence of adaptive/tactical-v1/v2-only sections.

Map every property explicitly into Project A's `TrainingScenario.TacticalV3` and reward objects; call `BuildTacticalV3()` in preflight. Set `UsesStructuredCandidates=True`, fill `ContractIdentity` from `TacticalV3Contract`, and leave both fixed geometry properties null. Do not invent an observation size or action count for display.

- [ ] **Step 4: Load structured runs only into Arena**

Extend Arena manifest DTOs with `algorithm`, `contract_hash`, and `capacity_hash`. Resolve the selected run's contained `scenario.json` and `checkpoints/best.pt`, require unsealed experimental state, and compare environment/version/contract/encoding/capacity identities before altering the active configuration. Pass the unchanged `run:PATH` controller spec to `PolicyBridge`. Display ?variable structured candidates? and the three semantic hashes.

Keep tactical-v3 out of SB3 Train-tab environment/algorithm choices. If an imported config requests it, show a specific offline-imitation instruction and throw before any process launch. Do not add corpus collection, teacher selection, DAgger, curriculum, or win-rate controls.

- [ ] **Step 5: Run GREEN, inspect logs, and commit**

Use Coplay: `check_compile_errors`; run `MlTrainingScenarioTests`, `MlLabConfigTests`, `MlLabWindowStateTests`, and the Task 11 bridge/configuration classes; then `get_unity_logs`. Expected: zero compile/test errors and no new ML Lab/Arena exceptions.

```powershell
git add Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs Assets/HexWars/Editor/MlLab/MlLabWindow.cs Assets/HexWars/Editor/MlLab/MlLabConfig.cs Assets/HexWars/Tests/Editor/MlTrainingScenarioTests.cs Assets/HexWars/Tests/Editor/MlLabConfigTests.cs Assets/HexWars/Tests/Editor/MlLabWindowStateTests.cs
git commit -m "feat: load tactical-v3 runs in ML Lab"
```

---
### Task 13: Tiny Overfit, Cross-size Identity Round-trip, and Final Acceptance Gate

**Files:**
- Create: `python/tests/test_tactical_v3_end_to_end.py`
- Create: `docs/superpowers/reports/2026-08-11-generalizable-structured-imitation-project-b.md`

**Interfaces exercised in this task:**

```python
def overfit_metrics(
    controller: StructuredController,
    examples: tuple[StructuredExample, ...],
) -> Mapping[str, float]: ...
# Required keys: policy_nll, policy_accuracy, finite_logit_count.

def assert_gymserver_identity_round_trip(
    client: TacticalV3GymClient,
    controller: StructuredController,
    seed: int,
) -> CandidateIdentity: ...
```

Keep these as test helpers, not production API. They use the same strict parser, batcher, controller, and GymServer client as publication.

- [ ] **Step 1: Write the failing end-to-end tests**

```python
def test_tiny_corpus_overfits_deterministically() -> None: ...
def test_13x9_gymserver_policy_server_candidate_identity_round_trip() -> None: ...
def test_same_checkpoint_infers_legally_on_24x16_without_rebuild() -> None: ...
def test_two_publications_reload_with_identical_logits_and_actions() -> None: ...
```

Train two independent CPU models from the frozen tiny corpus and seed 227. Require identical histories, state hashes, CPU logits, and selected identities. On both train and validation examples require `policy_accuracy == 1.0`, `policy_nll < 0.02`, and all valid logits finite. These are fixture-overfit checks only.

For 13x9, reset seed 41 through the real GymServer, send its exact structured view through a real `policy_server.py` subprocess, prove the response identity is in the legal candidate list exactly once, call GymServer `step(decision_id, candidate_id)`, and require a new decision or terminal view. For 24x16, use `scenario-24x16.json`, the same loaded model instance/configuration and semantic encoding/capacity hashes, and require finite logits plus a legal selected identity. Do not compare or require equal match `contract_hash`: board/match configuration legitimately changes while schema identities remain compatible.

- [ ] **Step 2: Run RED**

```powershell
dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_end_to_end.py -q
```

Expected: the new tests fail until all preceding task interfaces are present; do not weaken thresholds or replace the subprocess paths with mocks.

- [ ] **Step 3: Complete the narrow integration harness and run GREEN**

Use temporary exclusive run directories inside pytest. Bound subprocess startup/read/exit waits, capture stderr, and close both JSONL processes in `finally`. Keep the 24x16 board dimensions out of tensor/model constructors and assert the model parameter count and `TacticalV3ModelConfig` are unchanged across sizes.

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_end_to_end.py -q
```

Expected: four tests pass with the exact overfit and identity assertions.

- [ ] **Step 4: Publish and reopen one experimental run under `python/runs`**

Start with a path that does not exist; publication must refuse reuse rather than deleting it.

```powershell
uv run --active --no-project python python/run_tactical_v3_imitation.py train --corpus python/tests/fixtures/tactical_v3/tiny-corpus --run-dir python/runs/tactical-v3-policy-project-b-acceptance --seed 227 --device cpu
uv run --active --no-project python python/run_tactical_v3_imitation.py validate-run --run-dir python/runs/tactical-v3-policy-project-b-acceptance
```

Expected: the run is `state=completed`, `evidence_status=unsealed-experimental`, algorithm `structured_imitation`, checkpoint `checkpoints/best.pt`, exact corpus/contract/encoding/capacity/state hashes, CPU publication, replayed fixture logits/actions, best epoch, and policy NLL below 0.02. The ignored run is experimental evidence and is not added to git.

- [ ] **Step 5: Run focused and full Python acceptance**

```powershell
uv run --active --no-project python --version
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_schema.py python/tests/test_tactical_v3_client.py python/tests/test_tactical_v3_corpus.py python/tests/test_tactical_v3_batching.py python/tests/test_tactical_v3_layers.py python/tests/test_tactical_v3_model.py python/tests/test_tactical_v3_objectives.py python/tests/test_tactical_v3_training.py python/tests/test_tactical_v3_checkpoint.py python/tests/test_tactical_v3_controller.py python/tests/test_tactical_v3_end_to_end.py python/tests/test_controllers.py python/tests/test_policy_server.py -q
uv run --active --no-project python -m pytest python/tests -q
```

Expected: Python reports 3.14.x and both suites pass. Record exact counts, duration, CUDA availability, skipped CUDA tolerance test if applicable, and the 13x9/24x16 result; a skip is permitted only for unavailable CUDA, never for CPU determinism or cross-size inference.

- [ ] **Step 6: Run full engine and Unity acceptance**

```powershell
dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --no-restore
powershell -ExecutionPolicy Bypass -File engine/build-to-unity.ps1
```

Use Coplay against this worktree in order: `set_unity_project_root`; wait for import; `check_compile_errors`; run the six affected EditMode classes from Tasks 11?12; run the ML Lab selected-run test with `python/runs/tactical-v3-policy-project-b-acceptance`; `get_unity_logs`; then `check_compile_errors` again. Expected: full .NET green, zero Unity compilation/test errors, the selected structured run loads into Arena with its capacity identity, and no new exceptions/errors in logs. If the Editor is unavailable, report the missing gate and do not claim completion.

- [ ] **Step 7: Write the Project B evidence report**

Record reviewed Project A base `1b260449`; changed files; public interfaces; Python/.NET/Unity commands and exact results; corpus/run/state hashes; best epoch/NLL/accuracy; deterministic save/load/logit/action evidence; GPU-to-CPU tolerance result or CUDA skip; legal 13x9 round-trip; 24x16 inference; and unsealed publication path.

The report must state that this is only tiny-corpus offline-imitation plumbing. It makes no full-game quality or win-rate claim and does not include full-game bounded-search collection, production DAgger, curriculum expansion, fog, or unit design. Name those as subsequent projects, not deferred work inside Project B.

- [ ] **Step 8: Run the final scope and hygiene gate, then commit**

Invoke `superpowers:verification-before-completion`, rerun any command it identifies as stale, and inspect every result before claiming success.

```powershell
git diff --check
git status --short
git diff --name-only 1b260449...HEAD
rg -n "bounded.search|DAgger|curriculum|fog|unit design|win.rate" python/ml_lab/tactical_v3_*.py python/run_tactical_v3_imitation.py
```

Expected: no whitespace errors; only planned files; the final search is empty; no dependency file changed; no fixed observation/action shape, SB3/PyG wrapper, corpus mutation, sealed claim, or production collection command exists.

```powershell
git add python/tests/test_tactical_v3_end_to_end.py docs/superpowers/reports/2026-08-11-generalizable-structured-imitation-project-b.md
git commit -m "test: complete tactical-v3 policy gate"
git status --short
```

Expected: final status is clean.

## Final Acceptance Criteria

Project B is complete only when all of the following are evidenced in the report:

- strict frozen DTO parsing rejects unknown/missing/wrong-typed/nonfinite fields and invalid references;
- contract, encoding, and capacity identities are recorded, with cross-board compatibility gated by semantic encoding/capacity rather than a fixed match hash;
- the canonical example and train/validation corpus are immutable and content-addressed;
- ragged batching, remapping, masks, local hex message passing, and typed relational attention pass permutation/equivariance and padding-invariance tests;
- the shared scorer selects exact legal identities for variable candidate counts and fails closed for all-masked/nonfinite inputs;
- auxiliary coefficient sum never exceeds the policy coefficient, censoring is correct, and every training boundary has finite checks;
- CPU training/save/load/logits/actions are deterministic; CUDA publication, when available, meets the declared CPU tolerance and exact action criterion;
- the checkpoint loads weights-only, publishes CPU tensors atomically, replays fixtures, and remains explicitly unsealed;
- the real 13x9 GymServer/policy-server/GymServer path accepts the selected identity and the same checkpoint infers legally on 24x16 without architecture changes;
- ML Lab loads the structured run into Unity Arena while its SB3 Train tab refuses tactical-v3;
- full Python and engine suites plus Unity compile/tests/log inspection are green; and
- the evidence makes no full-game behavior, bounded-search, DAgger, curriculum, fog, unit-design, or win-rate claim.

---
