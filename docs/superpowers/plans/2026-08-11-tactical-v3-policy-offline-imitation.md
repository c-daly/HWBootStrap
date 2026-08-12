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
RelativeOwner = Literal["self", "opponent"]
TerrainTypeName = Literal["plains", "forest", "rough", "water"]
RelationKind = Literal["neighbor", "occupies", "has_capability"]
CandidateKind = Literal["attack", "move", "deploy", "end_turn"]
CapabilityKind = Literal["health", "damage", "defense", "movement",
    "vertical_movement", "range", "range_arc", "vision", "vision_arc"]
RuleKind = Literal["win_conditions", "round", "round_cap", "actions_per_turn",
    "starting_points", "self_points", "opponent_points", "damage_floor",
    "damage_high_ground_bonus", "range_high_ground_bonus", "bounty_rate",
    "deploy_cost_multiplier", "fog_of_war", "max_design_point_cost", "design_fee"]

@dataclass(frozen=True, slots=True)
class TokenRef:
    table: TableName
    row: int

@dataclass(frozen=True, slots=True)
class CellToken:
    q: int
    r: int
    terrain: TerrainTypeName
    elevation: int
    self_deployment_zone: bool
    opponent_deployment_zone: bool
    controller: RelativeOwner | None
    is_boundary: bool
    currently_visible: bool
    previously_observed: bool

@dataclass(frozen=True, slots=True)
class UnitToken:
    owner: RelativeOwner
    current_hp: int
    max_hp: int
    cell: TokenRef
    elevation: int
    moved: bool
    attacked: bool
    horizontal_movement_spent: int
    vertical_movement_spent: int
    point_cost: int
    deploy_cost: int
    currently_visible: bool

@dataclass(frozen=True, slots=True)
class TemplateToken:
    owner: RelativeOwner
    point_cost: int
    deploy_cost: int
    is_fixed: bool
    is_deployable: bool

@dataclass(frozen=True, slots=True)
class CapabilityDefinitionToken:
    kind: CapabilityKind

@dataclass(frozen=True, slots=True)
class CapabilityAllocationToken:
    owner: TokenRef
    definition: TokenRef
    capability: CapabilityKind
    purchased_level: int
    effective_value: int

@dataclass(frozen=True, slots=True)
class RuleToken:
    kind: RuleKind
    int_value: int
    float_value: float
    bool_value: bool

@dataclass(frozen=True, slots=True)
class MemoryToken:
    cell: TokenRef
    last_seen_round: int
    observation_age: int
    last_known_current_hp: int
    currently_visible: bool

@dataclass(frozen=True, slots=True)
class RelationToken:
    kind: RelationKind
    source: TokenRef
    target: TokenRef
    int_feature: int
    float_feature: float
    bool_feature: bool

@dataclass(frozen=True, slots=True)
class ProjectedDelta:
    source_cell: TokenRef | None
    destination_cell: TokenRef | None
    template: TokenRef | None
    target: TokenRef | None
    horizontal_movement_spent: int
    vertical_movement_spent: int
    target_hp_delta: int
    damage: int
    is_lethal: bool
    bounty_delta: int
    points_delta: int
    round_delta: int
    is_terminal: bool

@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: int
    decision_id: int
    kind: CandidateKind
    actor: TokenRef | None
    target: TokenRef | None
    template: TokenRef | None
    cell: TokenRef | None
    projection: ProjectedDelta

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
class TacticalV3Reward:
    terminal_outcome: float
    known_health_adjusted_material_progress: float
    public_resource_progress: float
    time_pressure: float
    total: float
    finalized: bool

@dataclass(frozen=True, slots=True)
class TacticalV3View:
    decision: TacticalV3Decision
    reward: TacticalV3Reward
    winner: int
    terminated: bool
    truncated: bool
    start_profile: str
    reference_seat: int

    @property
    def seat(self) -> int:
        return self.decision.seat

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

The in-memory identity below deliberately proves strict shape, hash, reference, and freezing behavior without reading artifacts owned by a later task. Task 2 is the first gate against canonical Project A fixture bytes.

- [ ] **Step 1: Write failing exact-shape, type, immutability, hash, and reference tests**

```python
def raw_canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def minimal_spaces_payload() -> dict[str, object]:
    encoding = {
        "schema_version": 1,
        "version": "tactical-v3",
        "hex_offset_layout": "odd-q",
        "token_reference_schema": ["table:table_kind", "row:int32"],
    }
    capacity = {
        "max_cells": 4,
        "max_units": 4,
        "max_templates": 4,
        "max_capability_definitions": 9,
        "max_capability_allocations": 16,
        "max_rules": 16,
        "max_memory_records": 4,
        "max_relations": 16,
        "max_candidates": 16,
    }
    return {
        "scenario_id": "in-memory-schema-test",
        "scenario_schema_version": 1,
        "contract_version": "tactical-v3",
        "contract_hash": "a" * 64,
        "encoding_hash": raw_canonical_sha256(encoding),
        "capacity_hash": raw_canonical_sha256(capacity),
        "environment_kind": "tactical",
        "match": {"board": {"width": 1, "height": 1}, "max_steps": 8},
        "encoding": encoding,
        "capacity": capacity,
    }

def minimal_view_payload() -> dict[str, object]:
    cell_ref = {"table": "cells", "row": 0}
    unit_ref = {"table": "units", "row": 0}
    definition_ref = {"table": "capability_definitions", "row": 0}
    projection = {
        "source_cell": cell_ref.copy(),
        "destination_cell": cell_ref.copy(),
        "template": None,
        "target": None,
        "horizontal_movement_spent": 1,
        "vertical_movement_spent": 0,
        "target_hp_delta": 0,
        "damage": 0,
        "is_lethal": False,
        "bounty_delta": 0,
        "points_delta": 0,
        "round_delta": 0,
        "is_terminal": False,
    }
    return {
        "decision_id": 7,
        "seat": 0,
        "observation": {
            "cells": [{
                "q": 0, "r": 0, "terrain": "plains", "elevation": 0,
                "self_deployment_zone": True, "opponent_deployment_zone": False,
                "controller": "self", "is_boundary": True,
                "currently_visible": True, "previously_observed": True,
            }],
            "units": [{
                "owner": "self", "current_hp": 2, "max_hp": 2,
                "cell": cell_ref.copy(), "elevation": 0,
                "moved": False, "attacked": False,
                "horizontal_movement_spent": 0, "vertical_movement_spent": 0,
                "point_cost": 3, "deploy_cost": 3, "currently_visible": True,
            }],
            "templates": [{
                "owner": "self", "point_cost": 3, "deploy_cost": 3,
                "is_fixed": True, "is_deployable": True,
            }],
            "capability_definitions": [{"kind": "health"}],
            "capability_allocations": [{
                "owner": unit_ref.copy(), "definition": definition_ref,
                "capability": "health", "purchased_level": 2, "effective_value": 2,
            }],
            "rules": [{
                "kind": "round", "int_value": 1,
                "float_value": 0.0, "bool_value": False,
            }],
            "memory": [],
            "relations": [{
                "kind": "occupies", "source": unit_ref.copy(), "target": cell_ref.copy(),
                "int_feature": 0, "float_feature": 0.0, "bool_feature": False,
            }],
        },
        "candidates": [{
            "candidate_id": 0, "decision_id": 7, "kind": "move",
            "actor": unit_ref.copy(), "target": None, "template": None,
            "cell": cell_ref.copy(), "projection": projection,
        }],
        "reward": {
            "terminal_outcome": 0.0,
            "known_health_adjusted_material_progress": 0.0,
            "public_resource_progress": 0.0,
            "time_pressure": 0.0,
            "total": 0.0,
            "finalized": False,
        },
        "winner": -1,
        "terminated": False,
        "truncated": False,
        "start_profile": "in-memory-1v1",
        "reference_seat": 0,
    }

EXPECTED_ERRORS = {
    "unknown_field": "view fields",
    "bool_candidate_id": "candidate_id must be an int32",
    "nan_rule_float": "rules\\[0\\].float_value must be finite",
    "wrong_ref_table": "move.actor references incompatible table",
    "ref_row_equal_to_length": "cells\\[1\\] of 1",
    "duplicate_candidate_id": "candidate ids must be exactly",
    "stale_candidate_decision": "candidate decision_id does not match",
}

def mutated_payload(mutation: str) -> tuple[dict[str, object], dict[str, object]]:
    spaces = minimal_spaces_payload()
    view = minimal_view_payload()
    if mutation == "unknown_field":
        view["extra"] = 1
    elif mutation == "bool_candidate_id":
        view["candidates"][0]["candidate_id"] = True
    elif mutation == "nan_rule_float":
        view["observation"]["rules"][0]["float_value"] = float("nan")
    elif mutation == "wrong_ref_table":
        view["candidates"][0]["actor"] = {"table": "cells", "row": 0}
    elif mutation == "ref_row_equal_to_length":
        view["observation"]["units"][0]["cell"] = {"table": "cells", "row": 1}
    elif mutation == "duplicate_candidate_id":
        view["candidates"].append(copy.deepcopy(view["candidates"][0]))
    elif mutation == "stale_candidate_decision":
        view["candidates"][0]["decision_id"] = 8
    else:
        raise AssertionError(f"unknown mutation {mutation}")
    return spaces, view

def test_spaces_and_view_parse_to_deeply_immutable_semantic_values() -> None:
    spaces_payload = minimal_spaces_payload()
    view_payload = minimal_view_payload()
    identity = parse_spaces(spaces_payload)
    view = parse_view(view_payload, identity)
    assert identity.encoding_hash == raw_canonical_sha256(spaces_payload["encoding"])
    assert identity.capacity_hash == raw_canonical_sha256(spaces_payload["capacity"])
    assert view.seat == view.decision.seat == 0
    assert view.decision == parse_decision(view_payload, identity)
    assert view.decision.candidates[0].decision_id == view.decision.decision_id == 7
    assert view.reward.finalized is False
    with pytest.raises(TypeError):
        identity.capacity["max_cells"] = 1
    with pytest.raises(AttributeError):
        view.decision.candidates.append(view.decision.candidates[0])

@pytest.mark.parametrize("mutation", tuple(EXPECTED_ERRORS))
def test_parser_rejects_malformed_wire_before_numeric_coercion(mutation: str) -> None:
    spaces, view = mutated_payload(mutation)
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
    cell_neighbor_index: torch.Tensor  # int64 [B, Ncell, 6]
    cell_neighbor_mask: torch.Tensor   # bool [B, Ncell, 6]
    neighborhoods: RelationNeighborhoodBatch
    candidates: CandidateBatch
    teacher_candidate_index: torch.Tensor
    terminal_outcome: torch.Tensor
    horizon_targets: torch.Tensor
    horizon_target_mask: torch.Tensor
    remaining_turns: torch.Tensor
    remaining_turns_mask: torch.Tensor
def collate_decisions(decisions: Sequence[TacticalV3Decision], horizons: tuple[int, ...]) -> RaggedBatch: ...

def collate_examples(examples: Sequence[StructuredExample], horizons: tuple[int, ...]) -> RaggedBatch: ...
```

- [ ] **Step 1: Write failing mixed-size, remapping, padding, and malformed-target tests**

```python
def test_collate_remaps_every_reference_and_hex_neighbor_into_masked_global_nodes() -> None:
    examples = (EXAMPLE_13X9, EXAMPLE_24X16)
    batch = collate_examples(examples, horizons=(4, 8, 16))
    assert batch.tables["cells"].mask.sum(dim=1).tolist() == [117, 384]
    cells_slice = batch.table_slices["cells"]
    assert batch.cell_neighbor_index.shape == (2, 384, 6)
    assert batch.cell_neighbor_mask.shape == (2, 384, 6)
    for sample_index, (example, cell_count) in enumerate(
        zip(examples, (117, 384), strict=True)
    ):
        valid = batch.candidates.reference_index[sample_index][
            batch.candidates.reference_mask[sample_index]
        ]
        assert torch.all(valid >= 0)
        assert torch.all(valid < batch.node_mask.shape[1])
        assert batch.node_mask[sample_index, valid].all()
        neighbor_index = batch.cell_neighbor_index[sample_index]
        neighbor_mask = batch.cell_neighbor_mask[sample_index]
        valid_neighbors = neighbor_index[neighbor_mask]
        assert torch.all(valid_neighbors >= cells_slice.start)
        assert torch.all(valid_neighbors < cells_slice.stop)
        assert batch.node_mask[sample_index, valid_neighbors].all()
        assert not neighbor_mask[cell_count:].any()
        assert torch.all(neighbor_index[~neighbor_mask] == 0)
        for destination_row in range(cell_count):
            actual_sources = neighbor_index[destination_row][
                neighbor_mask[destination_row]
            ].tolist()
            expected_sources = sorted(
                cells_slice.start + relation.source.row
                for relation in example.decision.observation.relations
                if relation.kind == "neighbor" and relation.target.row == destination_row
            )
            assert actual_sources == expected_sources
            assert len(actual_sources) <= 6

@pytest.mark.parametrize("failure", ["invalid_ref", "teacher_missing", "nan", "all_masked"])
def test_collate_fails_closed_before_returning_tensors(failure: str) -> None:
    with pytest.raises(ValueError, match=FAILURE_TEXT[failure]):
        collate_examples([broken_example(failure)], horizons=(4, 8, 16))

def assert_tensor_fields_equal(left: object, right: object) -> None:
    assert type(left) is type(right)
    for field in dataclasses.fields(left):
        left_value = getattr(left, field.name)
        right_value = getattr(right, field.name)
        if isinstance(left_value, Mapping):
            assert left_value.keys() == right_value.keys()
            for key in left_value:
                torch.testing.assert_close(left_value[key], right_value[key], rtol=0.0, atol=0.0)
        else:
            torch.testing.assert_close(left_value, right_value, rtol=0.0, atol=0.0)

def test_collate_decisions_matches_features_without_fabricating_targets() -> None:
    supervised = collate_examples([EXAMPLE_13X9], horizons=(4, 8, 16))
    inference = collate_decisions([EXAMPLE_13X9.decision], horizons=(4, 8, 16))
    assert supervised.tables.keys() == inference.tables.keys()
    for table_name in supervised.tables:
        assert_tensor_fields_equal(supervised.tables[table_name], inference.tables[table_name])
    assert supervised.table_slices == inference.table_slices
    torch.testing.assert_close(supervised.node_mask, inference.node_mask, rtol=0.0, atol=0.0)
    assert torch.equal(supervised.cell_neighbor_index, inference.cell_neighbor_index)
    assert torch.equal(supervised.cell_neighbor_mask, inference.cell_neighbor_mask)
    assert_tensor_fields_equal(supervised.neighborhoods, inference.neighborhoods)
    assert_tensor_fields_equal(supervised.candidates, inference.candidates)
    assert inference.teacher_candidate_index.tolist() == [-1]
    assert inference.terminal_outcome.tolist() == [-1]
    assert not inference.horizon_target_mask.any()
    assert not inference.remaining_turns_mask.any()
    assert torch.count_nonzero(inference.horizon_targets) == 0
    assert torch.count_nonzero(inference.remaining_turns) == 0
```

- [ ] **Step 2: Run tests and confirm RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_batching.py -q
```

Expected: import fails because the batching module is absent.

- [ ] **Step 3: Implement typed table encoding and per-batch global reference maps**

Use batch-local padded maxima only. Concatenate table regions in the exact table order, build `(sample, table, row) -> global_index`, map absent optional references to index zero with `reference_mask=False`, and assert every present reference lands on a `node_mask=True` row. Center `q/r` over valid cells and divide by `max(1, max(abs(centered_q), abs(centered_r)))`; never expose raw row indices as numeric features.

- [ ] **Step 4: Build deterministic incoming neighborhoods without scatter**

Consume only the schema-authenticated `neighbor: cells -> cells` relations for local topology. For each relation, append the remapped global source-cell index to its target cell row, require unique sources and at most six sources per valid destination, sort sources by global index, and pad to exactly six slots with `index=0, mask=False`. Padded destination rows contain six masked dummy slots. Every true-masked value must be in the canonical cells slice and point to `node_mask=True`; do not infer adjacency from coordinates or table row order.

Independently sort relational-attention edges by `(destination_global_index, relation_kind, source_global_index, int_feature, float_feature, bool_feature)`, add reverse kinds explicitly, add derived allocation-owner and allocation-definition edges, pad each destination's incoming list to batch maximum `K`, and retain a boolean mask. Valid zero-degree nodes receive one masked dummy slot, never a semantic self-edge.

- [ ] **Step 5: Encode targets and run GREEN**

Share one state/candidate collation path between `collate_examples` and `collate_decisions`. Map candidate identity to its row only after proving an exact unique match. Outcome indices are `loss=0, draw=1, win=2`. For horizon `h`, set a valid binary target only when the episode result and censoring permit it. Set `remaining_turns_mask=True` exactly for nontruncated wins with a defined positive remaining value.

`collate_decisions` is target-free: use `teacher_candidate_index=-1`, `terminal_outcome=-1`, zero target values, and all-false target masks. It must still perform the same reference, finiteness, and nonempty-candidate validation as supervised collation. `structured_imitation_loss` rejects a batch containing either sentinel, so inference cannot accidentally enter training.

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
        cells_slice: slice,
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

The table encoder owns distinct categorical embeddings and numeric projections per `TableKind`; booleans remain two-valued categories. `LocalHexMessagePassing` receives `batch.cell_neighbor_index`, `batch.cell_neighbor_mask`, `batch.node_mask`, and `batch.table_slices["cells"]`; it gathers only those authenticated neighbors, returns a full `[B, Nnode, H]` state, and replaces only the canonical cells slice. `TypedRelationalAttention` gathers padded incoming sources, adds relation-kind and edge-feature encodings, and writes only valid destinations. Use stock PyTorch gather/reshape operations; do not add PyG, `torch_scatter`, or another graph dependency.

- [ ] **Step 1: Write the failing encoder tests**

```python
COORDINATE_FEATURE_SLICE = slice(0, 2)

def test_centered_coordinates_are_translation_invariant() -> None:
    base = canonical_example("13x9")
    shifted = translate_cell_coordinates(base, dq=17, dr=-9)
    left = collate_examples((base,), horizons=(4, 8, 16))
    right = collate_examples((shifted,), horizons=(4, 8, 16))
    torch.testing.assert_close(
        left.tables["cells"].numeric[..., COORDINATE_FEATURE_SLICE],
        right.tables["cells"].numeric[..., COORDINATE_FEATURE_SLICE],
        rtol=0.0,
        atol=0.0,
    )


def test_local_hex_reads_batch_neighbors_and_writes_only_the_cells_slice() -> None:
    case = make_layer_case(seed=17)
    encoded = case.token_encoders(case.batch)
    actual = case.local_hex(
        encoded,
        case.batch.cell_neighbor_index,
        case.batch.cell_neighbor_mask,
        case.batch.node_mask,
        case.batch.table_slices["cells"],
    )
    assert actual.shape == encoded.shape
    non_cell_mask = case.batch.node_mask.clone()
    non_cell_mask[:, case.batch.table_slices["cells"]] = False
    torch.testing.assert_close(
        actual[non_cell_mask], encoded[non_cell_mask], rtol=0.0, atol=0.0
    )

def test_local_hex_layer_is_equivariant_to_cell_row_permutation() -> None:
    case = make_layer_case(seed=19)
    permuted, inverse = permute_table_and_remap(case.batch, "cells", seed=23)
    actual = run_local_stack(case, permuted)
    restored = undo_table_rows(actual, permuted.table_slices["cells"], inverse)
    torch.testing.assert_close(
        restored[:, case.batch.table_slices["cells"]],
        run_local_stack(case, case.batch)[:, case.batch.table_slices["cells"]],
        rtol=0.0,
        atol=1e-6,
    )

def test_relational_layer_is_equivariant_to_typed_table_row_permutations() -> None:
    case = make_layer_case(seed=29)
    for table in ("units", "templates", "capability_definitions", "capability_allocations"):
        permuted, inverse = permute_table_and_remap(case.batch, table, seed=31)
        actual = run_encoder_stack(case, permuted)
        restored = undo_table_rows(actual, permuted.table_slices[table], inverse)
        torch.testing.assert_close(
            restored[:, case.batch.table_slices[table]],
            run_encoder_stack(case, case.batch)[:, case.batch.table_slices[table]],
            rtol=0.0,
            atol=1e-6,
        )

def test_padding_cannot_change_any_valid_node_embedding() -> None:
    case = make_layer_case(seed=37)
    padded = append_masked_padding(case.batch, fill=1_000_000.0)
    expected = run_encoder_stack(case, case.batch)
    actual = run_encoder_stack(case, padded)[:, : expected.shape[1]]
    torch.testing.assert_close(actual[case.batch.node_mask], expected[case.batch.node_mask],
                               rtol=0.0, atol=1e-6)

def test_masked_neighbors_cannot_change_any_valid_destination() -> None:
    case = make_layer_case(seed=41)
    mutated = replace_masked_neighbor_payload(case.batch, fill=1_000_000.0)
    torch.testing.assert_close(
        run_encoder_stack(case, mutated)[case.batch.node_mask],
        run_encoder_stack(case, case.batch)[case.batch.node_mask],
        rtol=0.0,
        atol=1e-6,
    )

def test_every_layer_rejects_nonfinite_inputs_before_attention() -> None:
    case = make_layer_case(seed=43)
    bad = replace_first_valid_numeric(case.batch, value=float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite.*tables.cells.numeric"):
        run_encoder_stack(case, bad)
```
Helper contracts for this test file:

```python
@dataclass(frozen=True, slots=True)
class LayerTestCase:
    batch: RaggedBatch
    token_encoders: TypedTokenEncoders
    local_hex: LocalHexMessagePassing
    relational: TypedRelationalAttention

def canonical_example(size: Literal["13x9"]) -> StructuredExample: ...
def translate_cell_coordinates(example: StructuredExample, dq: int, dr: int) -> StructuredExample: ...
def make_layer_case(seed: int) -> LayerTestCase: ...
def permute_table_and_remap(batch: RaggedBatch, table: str, seed: int) -> tuple[RaggedBatch, Tensor]: ...
def undo_table_rows(state: Tensor, table_slice: slice, inverse: Tensor) -> Tensor: ...
def run_local_stack(case: LayerTestCase, batch: RaggedBatch) -> Tensor:
    node_state = case.token_encoders(batch)
    return case.local_hex(
        node_state,
        batch.cell_neighbor_index,
        batch.cell_neighbor_mask,
        batch.node_mask,
        batch.table_slices["cells"],
    )
def run_encoder_stack(case: LayerTestCase, batch: RaggedBatch) -> Tensor: ...
def append_masked_padding(batch: RaggedBatch, fill: float) -> RaggedBatch: ...
def replace_masked_neighbor_payload(batch: RaggedBatch, fill: float) -> RaggedBatch: ...
def replace_first_valid_numeric(batch: RaggedBatch, value: float) -> RaggedBatch: ...
```

`canonical_example` loads the immutable seed-41 example through the real corpus/schema loaders. `translate_cell_coordinates` replaces every cell `q/r` by `q+dq/r+dr` without changing rows or references. `make_layer_case` seeds Torch, constructs the default config and all three eval-mode layers, and owns one canonical batch. `permute_table_and_remap` uses `torch.randperm(..., generator=torch.Generator().manual_seed(seed))` over valid rows, moves their feature rows, remaps `cell_neighbor_index` destination rows and global source indices, and remaps table slices, relational neighborhoods, allocations, and candidate/projection references; its returned inverse restores physical row order. `run_local_stack` passes the two dedicated neighbor tensors and canonical cells slice directly to `LocalHexMessagePassing`; it never reconstructs board adjacency from row positions. `run_encoder_stack` adds relational rounds. The padding helpers rebuild frozen batches, set every added cell-neighbor and relational-neighbor mask false, use index zero only beneath false masks, and alter only false-masked payloads. `replace_masked_neighbor_payload` changes only payload values under `cell_neighbor_mask=False` or `neighborhoods.mask=False`, leaving both masks unchanged. `replace_first_valid_numeric` changes the first true-masked cell numeric value and records that field name in the expected exception.

Build each permuted batch by permuting the row and every reference to that row together. Run identical weights, undo the output permutation, and compare valid rows with `torch.testing.assert_close(..., rtol=0.0, atol=1e-6)`. For padding invariance, append extreme finite values under false masks; valid outputs must remain within the same tolerance.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_layers.py -q
```

Expected: collection fails because `tactical_v3_layers` is absent.

- [ ] **Step 3: Implement typed input encoders**

Create one path per `TableKind`, concatenate its fixed-width encoded fields, and project to `hidden_dim`. Encode `q/r` only through Task 4's centered/scaled coordinates. Validate `hidden_dim % attention_heads == 0`, positive dimensions/rounds, and strictly increasing positive horizons. Multiply every encoded table by its row mask immediately after projection and every residual block.

- [ ] **Step 4: Implement local and relational updates**

For each configured round, pass the exact Task 4 `cell_neighbor_index` and `cell_neighbor_mask` tensors plus `batch.table_slices["cells"]` to the local layer, gather global neighbor states, compute safe masked mean/max summaries, and replace only `node_state[:, cells_slice, :]` through a shared residual MLP. Then gather typed incoming sources, add relation-kind and integer/float/bool edge encodings, apply masked multi-head attention and a masked feed-forward residual, and zero padded rows. An all-masked incoming set produces a finite zero message before the residual. Never reconstruct neighbors with row-order board arithmetic, create semantic dummy self-edges, use in-place scatter, or assume fixed table capacities.

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

    def __init__(self, config: TacticalV3ModelConfig) -> None: ...
    def forward(self, batch: RaggedBatch) -> PolicyOutput: ...

    @torch.inference_mode()
    def select(self, batch: RaggedBatch) -> tuple[CandidateIdentity, ...]: ...
```

- [ ] **Step 1: Write failing policy tests**

```python
def test_policy_output_shapes_and_finiteness() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=47)
    output = case.policy(case.batch)
    assert output.candidate_logits.shape == (3, 19)
    assert output.outcome_logits.shape == (3, 3)
    assert output.horizon_logits.shape == (3, 3)
    assert output.remaining_turns.shape == (3,)
    assert torch.isfinite(output.candidate_logits[case.batch.candidates.mask]).all()
    assert torch.isneginf(output.candidate_logits[~case.batch.candidates.mask]).all()
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        assert torch.isfinite(getattr(output, name)).all(), name

def test_candidate_permutation_permutes_logits_and_preserves_identity_selection() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=53)
    permuted, inverse = permute_candidate_rows(case.batch, seed=59)
    expected = case.policy(case.batch).candidate_logits
    actual = restore_candidate_rows(case.policy(permuted).candidate_logits, inverse)
    torch.testing.assert_close(actual[case.batch.candidates.mask],
                               expected[case.batch.candidates.mask], rtol=0.0, atol=1e-6)
    assert case.policy.select(permuted) == case.policy.select(case.batch)

def test_candidate_padding_cannot_change_valid_logits_or_argmax() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=61)
    padded = append_candidate_padding(case.batch, rows=11, fill=1_000_000.0)
    expected = case.policy(case.batch)
    actual = case.policy(padded)
    torch.testing.assert_close(
        actual.candidate_logits[:, : expected.candidate_logits.shape[1]][case.batch.candidates.mask],
        expected.candidate_logits[case.batch.candidates.mask],
        rtol=0.0,
        atol=0.0,
    )
    assert_auxiliary_heads_equal(actual, expected, atol=0.0)
    assert case.policy.select(padded) == case.policy.select(case.batch)

def test_batch_shape_padding_beside_synthetic_384_cell_state_is_invariant() -> None:
    example = canonical_model_example()
    synthetic_large = expand_synthetic_cells_for_batch_shape(example, total_cells=384)
    policy = seeded_policy(seed=67)
    single = collate_examples((example,), horizons=policy.config.horizon_turns)
    mixed = collate_examples((example, synthetic_large), horizons=policy.config.horizon_turns)
    expected = policy(single)
    actual = policy(mixed)
    torch.testing.assert_close(
        actual.candidate_logits[0, : single.candidates.mask.shape[1]],
        expected.candidate_logits[0],
        rtol=0.0,
        atol=1e-6,
    )
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        torch.testing.assert_close(
            getattr(actual, name)[0], getattr(expected, name)[0], rtol=0.0, atol=1e-6
        )
    assert policy.select(mixed)[0] == policy.select(single)[0]

def test_softmax_probability_is_zero_on_padding_and_sums_to_one_on_valid_rows() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=71)
    probabilities = torch.softmax(case.policy(case.batch).candidate_logits, dim=-1)
    assert torch.equal(probabilities[~case.batch.candidates.mask],
                       torch.zeros_like(probabilities[~case.batch.candidates.mask]))
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(3),
                               rtol=0.0, atol=1e-7)

def test_state_table_permutations_leave_policy_output_unchanged() -> None:
    case = make_policy_case(candidate_counts=(19,), seed=73)
    expected = case.policy(case.batch)
    for table in ("cells", "units", "templates", "capability_definitions",
                  "capability_allocations", "rules", "memory", "relations"):
        permuted, _inverse = permute_model_table_and_remap(case.batch, table, seed=79)
        actual = case.policy(permuted)
        torch.testing.assert_close(actual.candidate_logits, expected.candidate_logits,
                                   rtol=0.0, atol=1e-6)
        assert_auxiliary_heads_equal(actual, expected, atol=1e-6)

def test_projection_reference_changes_affect_only_the_referenced_candidate_path() -> None:
    case = make_reference_sensitive_policy_case(seed=83)
    candidate_row, alternate_cell = movable_projection_case(case.batch)
    changed = retarget_projection(case.batch, candidate_row, alternate_cell)
    before = case.policy(case.batch).candidate_logits[0]
    after = case.policy(changed).candidate_logits[0]
    other = case.batch.candidates.mask[0].clone()
    other[candidate_row] = False
    torch.testing.assert_close(after[other], before[other], rtol=0.0, atol=0.0)
    assert not torch.equal(after[candidate_row], before[candidate_row])

def test_all_masked_candidate_rows_raise_before_argmax() -> None:
    case = make_policy_case(candidate_counts=(3,), seed=89)
    with pytest.raises(ValueError, match="sample 0 has no valid candidates"):
        case.policy.select(mask_all_candidates(case.batch))

@pytest.mark.parametrize(
    "field", ("candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns")
)
def test_nonfinite_policy_output_raises_before_selection(field: str) -> None:
    case = make_policy_case(candidate_counts=(3,), seed=97)
    handle = inject_nan_into_policy_output(case.policy, field)
    try:
        with pytest.raises(FloatingPointError, match=field):
            case.policy(case.batch)
    finally:
        handle.remove()

def test_selected_candidate_is_an_exact_member_of_each_constructed_candidate_set() -> None:
    case = make_policy_case(candidate_counts=(1, 3, 19), seed=101)
    force_equal_candidate_logits(case.policy)
    selections = case.policy.select(case.batch)
    for sample, selection in enumerate(selections):
        constructed = candidate_identity_set(case.batch, sample)
        assert (selection.decision_id, selection.candidate_id) in constructed
        assert selection.candidate_id == min(candidate_id for _, candidate_id in constructed)
```
Helper contracts for this test file:

```python
def canonical_model_example() -> StructuredExample: ...

@dataclass(frozen=True, slots=True)
class PolicyTestCase:
    policy: TacticalV3Policy
    batch: RaggedBatch

def make_cardinality_stress_example(
    example: StructuredExample, candidate_count: int,
) -> StructuredExample:
    if candidate_count <= 0 or not example.decision.candidates:
        raise ValueError("cardinality stress data requires positive count and a source candidate")
    decision_id = example.decision.decision_id
    source = example.decision.candidates
    candidates = tuple(
        dataclasses.replace(
            source[index % len(source)], candidate_id=index, decision_id=decision_id
        )
        for index in range(candidate_count)
    )
    return dataclasses.replace(
        example,
        decision=dataclasses.replace(example.decision, candidates=candidates),
        target=dataclasses.replace(example.target, teacher_candidate_id=0),
    )

def make_policy_case(candidate_counts: tuple[int, ...], seed: int) -> PolicyTestCase:
    canonical = canonical_model_example()
    examples = tuple(
        make_cardinality_stress_example(canonical, count) for count in candidate_counts
    )
    policy = seeded_policy(seed)
    batch = collate_examples(examples, horizons=policy.config.horizon_turns)
    return PolicyTestCase(policy=policy, batch=batch)

def seeded_policy(seed: int) -> TacticalV3Policy:
    torch.manual_seed(seed)
    return TacticalV3Policy(TacticalV3ModelConfig()).cpu().eval()

def assert_auxiliary_heads_equal(
    actual: PolicyOutput, expected: PolicyOutput, atol: float,
) -> None:
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), rtol=0.0, atol=atol
        )

def expand_synthetic_cells_for_batch_shape(
    example: StructuredExample, total_cells: int,
) -> StructuredExample: ...
def permute_candidate_rows(batch: RaggedBatch, seed: int) -> tuple[RaggedBatch, tuple[Tensor, ...]]: ...
def restore_candidate_rows(logits: Tensor, inverse: tuple[Tensor, ...]) -> Tensor: ...
def append_candidate_padding(batch: RaggedBatch, rows: int, fill: float) -> RaggedBatch: ...
def permute_model_table_and_remap(batch: RaggedBatch, table: str, seed: int) -> tuple[RaggedBatch, Tensor]: ...
def make_reference_sensitive_policy_case(seed: int) -> PolicyTestCase: ...
def movable_projection_case(batch: RaggedBatch) -> tuple[int, int]: ...
def retarget_projection(batch: RaggedBatch, candidate_row: int, cell_row: int) -> RaggedBatch: ...
def mask_all_candidates(batch: RaggedBatch) -> RaggedBatch: ...
def inject_nan_into_policy_output(
    policy: TacticalV3Policy,
    field: Literal["candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns"],
) -> RemovableHandle: ...
def force_equal_candidate_logits(policy: TacticalV3Policy) -> None: ...
def candidate_identity_set(batch: RaggedBatch, sample: int) -> set[tuple[int, int]]: ...
```

The canonical loader uses the immutable seed-41 example. `make_cardinality_stress_example` cycles/copies its already validated candidate payloads, changes only `candidate_id` and `decision_id`, assigns exact contiguous IDs `0..count-1`, and preserves every actor/target/template/cell/projection reference. These are test-only cardinality stress rows, not claims that the engine emitted 1, 3, or 19 distinct legal actions. `make_policy_case` selects constructed candidate zero as the target, collates the constructed examples, and returns one CPU eval-mode policy created after `torch.manual_seed(seed)`.

`expand_synthetic_cells_for_batch_shape` appends unreferenced, `currently_visible=False` plains cells with unique `(1000+i, -1000-i)` coordinates until the exact requested total; all original tokens/candidates remain unchanged. It tests ragged batch shape and padding only and is not a real 24x16 match; Task 13 owns real cross-size legality. Candidate/table permutation helpers use private seeded generators, remap every dependent reference including Task 4 neighbor tensors, and return inverse row orders. Padding adds only false-masked rows. The reference-sensitive case sets the projection-destination coordinate path and final scorer weight to one and other scorer weights to zero; `movable_projection_case` returns the first move plus a different valid cell.

`inject_nan_into_policy_output` maps each field to the final module that produces it (`candidate_scorer`, `outcome_head`, `horizon_head`, or `remaining_turns_head`), installs a removable forward hook that clones the tensor and replaces its first valid scalar with NaN, and returns the handle. Equal-logit setup zeros the candidate scorer's final weight and bias. `candidate_identity_set` reads only true-masked rows from the constructed batch and returns their exact integer decision/candidate pairs.

Use a deterministic tie and require the smallest `candidate_id`, not the first padded row, as the tie-break. Exercise candidate counts 1, 3, and 19 in one batch; no constructor argument may encode a maximum action count.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_model.py -q
```

Expected: collection fails because `tactical_v3_model` is absent.

- [ ] **Step 3: Implement state and candidate projection paths**

Pool only the encoded typed state-table regions in `batch.table_slices` with mask-aware mean plus max, concatenate those table summaries, and project them to a shared state vector. Rules, cells, and units already represent current match state; do not add match, semantic-identity, or start-profile metadata to `RaggedBatch`. Exact identity/profile compatibility remains external in the checkpoint and controller tasks. Build each candidate from its kind/scalars; gathered source/target unit, cell, allocation, definition, and tile embeddings; typed projection edge/value embeddings; and explicit presence bits for every optional reference. Use one candidate MLP for all kinds and rows. Do not flatten tables or candidate axes into model parameters.

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
def test_policy_coefficient_is_exactly_one_and_auxiliary_sum_is_at_most_half() -> None:
    default = ObjectiveConfig()
    assert (default.outcome_coefficient + default.horizon_coefficient
            + default.remaining_turns_coefficient) == pytest.approx(0.5)
    assert default.policy_coefficient == 1.0
    assert default.outcome_coefficient + default.horizon_coefficient + default.remaining_turns_coefficient <= 0.5
    for value in (0.0, 0.5, 1.0000001):
        with pytest.raises(ValueError, match="policy_coefficient must be exactly 1.0"):
            ObjectiveConfig(policy_coefficient=value)

@pytest.mark.parametrize(
    "field", ("policy_coefficient", "outcome_coefficient",
              "horizon_coefficient", "remaining_turns_coefficient")
)
@pytest.mark.parametrize("value", (float("nan"), float("inf"), float("-inf"), -0.1))
def test_every_coefficient_rejects_nonfinite_and_negative_values(
    field: str, value: float,
) -> None:
    with pytest.raises(ValueError, match=field):
        ObjectiveConfig(**{field: value})

def test_policy_coefficient_rejects_nonfinite_and_nonexact_values() -> None:
    for value in (float("nan"), float("inf"), float("-inf"), -1.0, 0.0, 0.5, 2.0):
        with pytest.raises(ValueError, match="policy_coefficient must be exactly 1.0"):
            ObjectiveConfig(policy_coefficient=value)

def test_auxiliary_coefficient_sum_cannot_exceed_policy_coefficient() -> None:
    with pytest.raises(ValueError, match="auxiliary coefficient sum"):
        ObjectiveConfig(outcome_coefficient=0.6, horizon_coefficient=0.3,
                        remaining_turns_coefficient=0.2)

MALFORMED_OBJECTIVE_ERRORS = {
    "candidate_logits_shape": "candidate_logits shape",
    "candidate_mask_shape": "candidate mask shape",
    "outcome_logits_shape": "outcome_logits shape",
    "horizon_count": "horizon count",
    "horizon_mask_shape": "horizon_target_mask shape",
    "remaining_shape": "remaining_turns shape",
    "remaining_mask_shape": "remaining_turns_mask shape",
    "teacher_shape": "teacher_candidate_index shape",
    "outcome_target_shape": "terminal_outcome shape",
    "horizon_target_shape": "horizon_targets shape",
    "remaining_target_shape": "batch.remaining_turns shape",
    "teacher_out_of_range": "teacher_candidate_index.*out of range",
    "teacher_padded": "teacher_candidate_index.*padded",
    "outcome_target": "terminal_outcome.*0..2",
    "horizon_target": "horizon_targets.*binary",
    "remaining_target": "remaining_turns.*positive",
}

def test_target_free_collate_decisions_batch_is_rejected() -> None:
    examples = objective_examples()
    batch = collate_decisions(
        tuple(example.decision for example in examples), horizons=(4, 8, 16)
    )
    assert batch.teacher_candidate_index.tolist() == [-1, -1]
    assert batch.terminal_outcome.tolist() == [-1, -1]
    with pytest.raises(
        ValueError,
        match="target-free.*teacher_candidate_index=-1.*terminal_outcome=-1",
    ):
        structured_imitation_loss(
            finite_output_for_batch(batch), batch, ObjectiveConfig()
        )

@pytest.mark.parametrize("failure", tuple(MALFORMED_OBJECTIVE_ERRORS))
def test_shapes_and_target_ranges_fail_closed_before_loss_math(failure: str) -> None:
    output, batch = malformed_objective_case(failure)
    with pytest.raises(
        ValueError, match=MALFORMED_OBJECTIVE_ERRORS[failure]
    ):
        structured_imitation_loss(output, batch, ObjectiveConfig())

def test_policy_loss_ignores_padding_and_matches_manual_cross_entropy() -> None:
    output, batch = make_objective_case()
    changed = replace_padded_candidate_logits(output, value=1_000_000.0)
    expected = F.cross_entropy(valid_candidate_matrix(output, batch),
                               batch.teacher_candidate_index)
    actual = structured_imitation_loss(changed, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.policy, expected, rtol=0.0, atol=1e-7)

def test_outcome_loss_uses_loss_draw_win_target_order() -> None:
    output, batch = make_objective_case()
    expected = F.cross_entropy(output.outcome_logits, torch.tensor([0, 2]))
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.outcome, expected, rtol=0.0, atol=1e-7)

def test_horizon_loss_uses_only_uncensored_target_mask() -> None:
    output, batch = make_objective_case()
    expected = F.binary_cross_entropy_with_logits(
        output.horizon_logits[batch.horizon_target_mask],
        batch.horizon_targets[batch.horizon_target_mask],
    )
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.horizon, expected, rtol=0.0, atol=1e-7)

def test_remaining_turns_loss_uses_only_nontruncated_wins() -> None:
    output, batch = make_objective_case()
    expected = F.smooth_l1_loss(
        output.remaining_turns[batch.remaining_turns_mask],
        batch.remaining_turns[batch.remaining_turns_mask],
    )
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())
    torch.testing.assert_close(actual.remaining_turns, expected, rtol=0.0, atol=1e-7)

def test_weighted_total_is_exact_coefficient_combination() -> None:
    output, batch = make_objective_case()
    config = ObjectiveConfig(
        policy_coefficient=1.0,
        outcome_coefficient=0.1,
        horizon_coefficient=0.2,
        remaining_turns_coefficient=0.15,
    )
    actual = structured_imitation_loss(output, batch, config)
    expected = (actual.policy + 0.1 * actual.outcome + 0.2 * actual.horizon
                + 0.15 * actual.remaining_turns)
    torch.testing.assert_close(actual.total, expected, rtol=0.0, atol=0.0)

def test_empty_auxiliary_masks_produce_differentiable_finite_zeroes() -> None:
    output, batch = make_objective_case()
    empty = clear_auxiliary_masks(batch)
    actual = structured_imitation_loss(output, empty, ObjectiveConfig())
    assert actual.horizon.item() == 0.0
    assert actual.remaining_turns.item() == 0.0
    assert actual.horizon.requires_grad and actual.remaining_turns.requires_grad
    (actual.horizon + actual.remaining_turns).backward()
    assert torch.isfinite(output.horizon_logits.grad).all()
    assert torch.isfinite(output.remaining_turns.grad).all()

def test_padded_negative_infinity_is_allowed_and_total_remains_finite() -> None:
    output, batch = make_objective_case()
    assert torch.isneginf(output.candidate_logits[~batch.candidates.mask]).all()
    actual = structured_imitation_loss(output, batch, ObjectiveConfig())

    assert torch.isfinite(actual.total)
@pytest.mark.parametrize(
    ("field", "value"),
    (("candidate_logits", float("nan")),
     ("candidate_logits", float("inf")),
     ("candidate_logits", float("-inf")),
     ("outcome_logits", float("inf")),
     ("horizon_logits", float("-inf")),
     ("remaining_turns", float("nan"))),
)
def test_each_nonfinite_output_component_fails_with_named_error(
    field: str, value: float,
) -> None:
    output, batch = make_objective_case()
    bad = replace_output_component(output, field, value)
    with pytest.raises(FloatingPointError, match=field):
        structured_imitation_loss(bad, batch, ObjectiveConfig())


def test_default_loss_backpropagates_finite_scorer_and_encoder_gradients() -> None:
    model, batch = make_gradient_case(seed=103)
    loss = structured_imitation_loss(model(batch), batch, ObjectiveConfig()).total
    loss.backward()
    gradients = named_required_gradients(model, prefixes=("encoders.", "candidate_scorer."))
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in gradients.values())
```
Helper contracts for this test file:

```python
def objective_examples() -> tuple[StructuredExample, StructuredExample]: ...

def finite_output_for_batch(batch: RaggedBatch) -> PolicyOutput:
    batch_size, candidate_count = batch.candidates.mask.shape
    horizon_count = batch.horizon_targets.shape[1]
    candidate_logits = torch.zeros(batch_size, candidate_count).masked_fill(
        ~batch.candidates.mask, float("-inf")
    )
    return PolicyOutput(
        candidate_logits=candidate_logits.requires_grad_(True),
        outcome_logits=torch.zeros(batch_size, 3, requires_grad=True),
        horizon_logits=torch.zeros(batch_size, horizon_count, requires_grad=True),
        remaining_turns=torch.zeros(batch_size, requires_grad=True),
    )

def malformed_objective_case(failure: str) -> tuple[PolicyOutput, RaggedBatch]:
    output, batch = make_objective_case()
    if failure == "candidate_logits_shape":
        output = dataclasses.replace(output, candidate_logits=output.candidate_logits[:, :-1])
    elif failure == "candidate_mask_shape":
        candidates = dataclasses.replace(
            batch.candidates, mask=batch.candidates.mask[:, :-1]
        )
        batch = dataclasses.replace(batch, candidates=candidates)
    elif failure == "outcome_logits_shape":
        output = dataclasses.replace(output, outcome_logits=output.outcome_logits[:, :2])
    elif failure == "horizon_count":
        output = dataclasses.replace(output, horizon_logits=output.horizon_logits[:, :-1])
    elif failure == "horizon_mask_shape":
        batch = dataclasses.replace(
            batch, horizon_target_mask=batch.horizon_target_mask[:, :-1]
        )
    elif failure == "remaining_shape":
        output = dataclasses.replace(
            output, remaining_turns=output.remaining_turns.unsqueeze(1)
        )
    elif failure == "remaining_mask_shape":
        batch = dataclasses.replace(
            batch, remaining_turns_mask=batch.remaining_turns_mask.unsqueeze(1)
        )
    elif failure == "teacher_shape":
        batch = dataclasses.replace(
            batch, teacher_candidate_index=batch.teacher_candidate_index.unsqueeze(1)
        )
    elif failure == "outcome_target_shape":
        batch = dataclasses.replace(
            batch, terminal_outcome=batch.terminal_outcome.unsqueeze(1)
        )
    elif failure == "horizon_target_shape":
        batch = dataclasses.replace(
            batch, horizon_targets=batch.horizon_targets[:, :-1]
        )
    elif failure == "remaining_target_shape":
        batch = dataclasses.replace(
            batch, remaining_turns=batch.remaining_turns.unsqueeze(1)
        )
    elif failure in {"teacher_out_of_range", "teacher_padded"}:
        target = batch.teacher_candidate_index.clone()
        target[0] = batch.candidates.mask.shape[1] if failure.endswith("range") else 2
        batch = dataclasses.replace(batch, teacher_candidate_index=target)
    elif failure == "outcome_target":
        target = batch.terminal_outcome.clone()
        target[0] = 3
        batch = dataclasses.replace(batch, terminal_outcome=target)
    elif failure == "horizon_target":
        target = batch.horizon_targets.clone()
        target[0, 0] = 2.0
        batch = dataclasses.replace(batch, horizon_targets=target)
    elif failure == "remaining_target":
        target = batch.remaining_turns.clone()
        target[0] = -1.0
        batch = dataclasses.replace(batch, remaining_turns=target)
    else:
        raise AssertionError(f"unknown malformed objective case {failure}")
    return output, batch

def replace_output_component(
    output: PolicyOutput,
    field: Literal["candidate_logits", "outcome_logits", "horizon_logits", "remaining_turns"],
    value: float,
) -> PolicyOutput:
    changed = getattr(output, field).detach().clone()
    changed.reshape(-1)[0] = value
    changed.requires_grad_(True)
    return dataclasses.replace(output, **{field: changed})

def make_objective_case() -> tuple[PolicyOutput, RaggedBatch]: ...
def replace_padded_candidate_logits(output: PolicyOutput, value: float) -> PolicyOutput: ...
def valid_candidate_matrix(output: PolicyOutput, batch: RaggedBatch) -> Tensor: ...
def clear_auxiliary_masks(batch: RaggedBatch) -> RaggedBatch: ...
def make_gradient_case(seed: int) -> tuple[TacticalV3Policy, RaggedBatch]: ...
def named_required_gradients(model: nn.Module, prefixes: tuple[str, ...]) -> dict[str, Tensor | None]: ...
```

`objective_examples` loads train row zero and validation row zero from `python/tests/fixtures/tactical_v3/tiny-corpus` through the real corpus loader using the semantic identity parsed from `python/tests/fixtures/tactical_v3/seed-41-spaces.json`; it returns exactly two immutable examples. `finite_output_for_batch` derives `B`, `C`, and horizon count only from Task 4 tensors, emits finite zero heads, and places `-inf` only under false candidate masks. `malformed_objective_case` starts from the canonical objective case and changes only the field named by its exhaustive branch; every branch body above is the required mutation.

`make_objective_case` returns two samples with candidate logits `[[2,0,-inf],[0,1,2]]`, masks `[[T,T,F],[T,T,T]]`, and targets `[0,2]`; outcome logits `[[2,1,0],[0,1,2]]` with targets `[loss,win]`; horizon logits `[[0,2,-2],[1,-1,3]]`, targets `[[1,0,0],[0,1,1]]`, and mask `[[T,F,F],[F,T,F]]`; remaining predictions `[4,7]`, targets `[5,9]`, and mask `[T,F]`. All output tensors are independent `requires_grad=True` leaves. Replacement helpers use `dataclasses.replace`, mutate only the named masked/valid positions, and leave the source frozen values unchanged. `valid_candidate_matrix` clones logits and restores `-inf` at false masks. `make_gradient_case` collates two immutable corpus rows and constructs the default seeded CPU model; `named_required_gradients` returns every trainable parameter whose name begins with either exact prefix.

Policy coefficient is exactly `1.0`; reject every other value. Reject negative/nonfinite auxiliary coefficients and `outcome + horizon + remaining_turns > 0.5`.

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_objectives.py -q
```

Expected: collection fails because `tactical_v3_objectives` is absent.

- [ ] **Step 3: Implement masked objectives**

Reject target-free inference sentinels before loss computation. Use categorical cross-entropy over valid candidate logits for policy and fixed `loss/draw/win` order for outcome. Use BCE-with-logits only where `horizon_target_mask=True`, and Smooth L1 only where `remaining_turns_mask=True`. An empty auxiliary mask returns `output_tensor.sum() * 0.0`, preserving device and autograd. Validate every output/batch shape, target range, valid logit/head value, component, coefficient, and weighted total before returning.

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

**Interfaces:**

```python
from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import torch

from ml_lab.tactical_v3_corpus import StructuredExample
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig


def _canonical_device(value: object) -> str:
    if type(value) is not str:
        raise ValueError("device must be a built-in str")
    if value == "cpu":
        return "cpu"
    match = re.fullmatch(r"cuda(?::([0-9]+))?", value)
    if match is None:
        raise ValueError(
            "device must be exactly cpu, cuda, or cuda:<nonnegative decimal index>"
        )
    if not torch.cuda.is_available():
        raise ValueError("device requests CUDA but CUDA is unavailable")
    index = (
        torch.cuda.current_device()
        if match.group(1) is None
        else int(match.group(1))
    )
    if index < 0 or index >= torch.cuda.device_count():
        raise ValueError(f"device CUDA index {index} is unavailable")
    return f"cuda:{index}"


@dataclass(frozen=True, slots=True)
class TrainerConfig:
    seed: int = 227
    batch_size: int = 4
    learning_rate: float = 3e-4
    max_epochs: int = 400
    patience_epochs: int = 100
    gradient_clip_norm: float = 1.0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("seed", "batch_size", "max_epochs", "patience_epochs"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if type(value) is not int or value < minimum:
                qualifier = "nonnegative" if name == "seed" else "positive"
                raise ValueError(f"{name} must be a {qualifier} built-in int")
        for name in ("learning_rate", "gradient_clip_norm"):
            value = getattr(self, name)
            if type(value) is not float or not math.isfinite(value) or value <= 0.0:
                raise ValueError(
                    f"{name} must be a finite positive built-in float"
                )
        object.__setattr__(self, "device", _canonical_device(self.device))


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
    history: tuple


def train_offline(
    train_examples: tuple,
    validation_examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
) -> TrainingResult:
    return _train_offline_impl(train_examples, validation_examples, model_config, objective_config, trainer_config)

def _canonical_example_key(example: StructuredExample) -> tuple[str, int, int, str, int]:
    if type(example) is not StructuredExample:
        raise TypeError("example must be StructuredExample")
    return example.scenario_id, example.episode_seed, example.learner_seat, example.profile_id, example.decision.decision_id
```

The following private helpers are the concrete trainer contract. `_batch_to_device` reconstructs every nested dataclass and mapping rather than mutating or partially replacing a batch. `_after_backward`, `_clip_grad_norm`, `_after_optimizer_step`, and `_validation_batch_losses` are deliberately narrow test seams called by the real loops; none owns aggregation, best-epoch selection, or early stopping.

```python
from __future__ import annotations

import dataclasses
import math
import random
from collections.abc import Iterable, Mapping
from types import MappingProxyType

import numpy as np
import torch
from torch import Tensor, nn

from ml_lab.tactical_v3_batching import (
    CandidateBatch,
    RaggedBatch,
    RelationNeighborhoodBatch,
    TokenTableBatch,
    collate_examples,
)
from ml_lab.tactical_v3_corpus import StructuredExample
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import PolicyOutput, TacticalV3Policy
from ml_lab.tactical_v3_objectives import LossBreakdown, ObjectiveConfig, structured_imitation_loss


def _batch_to_device(batch: RaggedBatch, device: torch.device) -> RaggedBatch:
    if type(batch) is not RaggedBatch or type(device) is not torch.device:
        raise TypeError("_batch_to_device requires RaggedBatch and torch.device")

    def move(value: Tensor) -> Tensor:
        return value.to(device=device)

    tables = MappingProxyType({
        name: TokenTableBatch(
            numeric=move(table.numeric),
            categorical=MappingProxyType({
                field_name: move(value)
                for field_name, value in table.categorical.items()
            }),
            boolean=MappingProxyType({
                field_name: move(value)
                for field_name, value in table.boolean.items()
            }),
            mask=move(table.mask),
        )
        for name, table in batch.tables.items()
    })
    neighborhoods = RelationNeighborhoodBatch(
        source_index=move(batch.neighborhoods.source_index),
        kind=move(batch.neighborhoods.kind),
        int_feature=move(batch.neighborhoods.int_feature),
        float_feature=move(batch.neighborhoods.float_feature),
        bool_feature=move(batch.neighborhoods.bool_feature),
        mask=move(batch.neighborhoods.mask),
    )
    candidates = CandidateBatch(
        candidate_id=move(batch.candidates.candidate_id),
        decision_id=move(batch.candidates.decision_id),
        kind=move(batch.candidates.kind),
        reference_index=move(batch.candidates.reference_index),
        reference_mask=move(batch.candidates.reference_mask),
        projection_integer=move(batch.candidates.projection_integer),
        projection_boolean=move(batch.candidates.projection_boolean),
        mask=move(batch.candidates.mask),
    )
    return RaggedBatch(
        tables=tables,
        table_slices=MappingProxyType(dict(batch.table_slices)),
        node_mask=move(batch.node_mask),
        cell_neighbor_index=move(batch.cell_neighbor_index),
        cell_neighbor_mask=move(batch.cell_neighbor_mask),
        neighborhoods=neighborhoods,
        candidates=candidates,
        teacher_candidate_index=move(batch.teacher_candidate_index),
        terminal_outcome=move(batch.terminal_outcome),
        horizon_targets=move(batch.horizon_targets),
        horizon_target_mask=move(batch.horizon_target_mask),
        remaining_turns=move(batch.remaining_turns),
        remaining_turns_mask=move(batch.remaining_turns_mask),
    )


def _after_backward(
    model: TacticalV3Policy, *, epoch: int, batch_index: int,
) -> None:
    del model, epoch, batch_index


def _clip_grad_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


def _after_optimizer_step(
    model: TacticalV3Policy, *, epoch: int, batch_index: int,
) -> None:
    del model, epoch, batch_index


def _validation_batch_losses(
    model: TacticalV3Policy,
    batch: RaggedBatch,
    objective_config: ObjectiveConfig,
    *,
    epoch: int,
    batch_index: int,
) -> LossBreakdown:
    device = next(model.parameters()).device
    context = f"epoch={epoch} validation_batch={batch_index}"
    output = model(batch)
    _validate_policy_output(output, batch, device, context)
    losses = structured_imitation_loss(output, batch, objective_config)
    return losses

METRIC_KEYS = ("total", "policy", "outcome", "horizon", "remaining_turns")


def _collate_training_batch(
    examples: tuple, horizons: tuple,
) -> RaggedBatch:
    return collate_examples(examples, horizons)


def _evaluate_validation(
    model: TacticalV3Policy,
    examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    batch_size: int,
    device: torch.device,
    *,
    epoch: int,
) -> tuple[Mapping[str, float], float]:
    weighted = {name: 0.0 for name in METRIC_KEYS}
    example_count = 0
    model.eval()
    with torch.no_grad():
        for batch_index, start in enumerate(range(0, len(examples), batch_size)):
            rows = examples[start:start + batch_size]
            batch = _batch_to_device(
                collate_examples(rows, model_config.horizon_turns), device
            )
            _validate_batch_contract(
                batch, device, f"epoch={epoch} validation_batch={batch_index}"
            )
            losses = _validation_batch_losses(
                model, batch, objective_config,
                epoch=epoch, batch_index=batch_index,
            )
            _validate_losses(
                losses, device, f"epoch={epoch} validation_batch={batch_index}"
            )
            for name in METRIC_KEYS:
                value = getattr(losses, name)
                if (
                    value.ndim != 0
                    or value.device != device
                    or not bool(torch.isfinite(value))
                ):
                    raise FloatingPointError(
                        f"epoch={epoch} validation_batch={batch_index} loss.{name}"
                    )
                contribution = float(value.detach().item()) * len(rows)
                if not math.isfinite(contribution):
                    raise FloatingPointError(
                        f"epoch={epoch} validation_batch={batch_index} weighted loss.{name}"
                    )
                weighted[name] += contribution
            example_count += len(rows)
    metrics = MappingProxyType({
        name: float(weighted[name] / example_count) for name in METRIC_KEYS
    })
    return metrics, metrics["policy"]

def _canonical_split(
    examples: tuple, label: str,
) -> tuple[StructuredExample, ...]:
    if type(examples) is not tuple:
        raise TypeError(f"{label} split must be an immutable tuple")
    if not examples:
        raise ValueError(f"{label} split must be non-empty")
    keyed = tuple((_canonical_example_key(example), example) for example in examples)
    keys = tuple(key for key, _ in keyed)
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate {label} example")
    return tuple(example for _, example in sorted(keyed, key=lambda item: item[0]))


def _named_batch_tensors(value: object, path: str = "batch"):
    if isinstance(value, Tensor):
        yield path, value
    elif dataclasses.is_dataclass(value) and not isinstance(value, type):
        for field_info in dataclasses.fields(value):
            yield from _named_batch_tensors(
                getattr(value, field_info.name), f"{path}.{field_info.name}"
            )
    elif isinstance(value, Mapping):
        for name, nested in value.items():
            yield from _named_batch_tensors(nested, f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for index, nested in enumerate(value):
            yield from _named_batch_tensors(nested, f"{path}[{index}]")


def _validate_batch_contract(
    batch: RaggedBatch, device: torch.device, context: str,
) -> None:
    if type(batch) is not RaggedBatch:
        raise TypeError(f"{context} batch must be RaggedBatch")
    for path, value in _named_batch_tensors(batch):
        if value.device != device:
            raise ValueError(f"{context} {path} device")
        if value.dtype not in {torch.float32, torch.int64, torch.bool}:
            raise ValueError(f"{context} {path} dtype")
        if path.endswith("mask") and value.dtype != torch.bool:
            field = "candidate_mask" if path == "batch.candidates.mask" else path
            raise FloatingPointError(f"{context} {field}")


def _validate_policy_output(
    output: PolicyOutput, batch: RaggedBatch, device: torch.device, context: str,
) -> None:
    if type(output) is not PolicyOutput:
        raise TypeError(f"{context} output must be PolicyOutput")
    batch_size = int(batch.node_mask.shape[0])
    expected = {
        "candidate_logits": tuple(batch.candidates.mask.shape),
        "outcome_logits": (batch_size, 3),
        "horizon_logits": tuple(batch.horizon_targets.shape),
        "remaining_turns": (batch_size,),
    }
    for name, shape in expected.items():
        value = getattr(output, name)
        if (
            not isinstance(value, Tensor)
            or value.device != device
            or value.dtype != torch.float32
            or tuple(value.shape) != shape
        ):
            raise ValueError(f"{context} {name} contract")
    logits = output.candidate_logits
    valid = batch.candidates.mask
    if not bool(torch.isfinite(logits[valid]).all()):
        raise FloatingPointError(f"{context} candidate_logits")
    if not bool(torch.isneginf(logits[~valid]).all()):
        raise FloatingPointError(f"{context} candidate_logits.padding")
    for name in ("outcome_logits", "horizon_logits", "remaining_turns"):
        if not bool(torch.isfinite(getattr(output, name)).all()):
            raise FloatingPointError(f"{context} {name}")


def _validate_losses(
    losses: LossBreakdown, device: torch.device, context: str,
) -> None:
    if type(losses) is not LossBreakdown:
        raise TypeError(f"{context} losses must be LossBreakdown")
    for name in METRIC_KEYS:
        value = getattr(losses, name)
        if (
            not isinstance(value, Tensor)
            or value.ndim != 0
            or value.device != device
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value))
        ):
            raise FloatingPointError(f"{context} loss.{name}")


def _frozen_metrics(weighted: Mapping[str, float], count: int) -> Mapping[str, float]:
    values = {
        name: float(weighted[name] / count)
        for name in METRIC_KEYS
    }
    if any(type(value) is not float or not math.isfinite(value) for value in values.values()):
        raise FloatingPointError("epoch metrics are nonfinite")
    return MappingProxyType(values)


def _snapshot_state(model: TacticalV3Policy) -> Mapping[str, Tensor]:
    return MappingProxyType({
        name: value.detach().to(device="cpu").contiguous().clone()
        for name, value in model.state_dict().items()
    })


def _restore_state(
    model: TacticalV3Policy, state: Mapping[str, Tensor], device: torch.device,
) -> None:
    model.load_state_dict(
        {name: value.to(device=device) for name, value in state.items()},
        strict=True,
    )


def _train_offline_impl(
    train_examples: tuple,
    validation_examples: tuple,
    model_config: TacticalV3ModelConfig,
    objective_config: ObjectiveConfig,
    trainer_config: TrainerConfig,
) -> TrainingResult:
    if type(train_examples) is not tuple:
        raise TypeError("training split must be an immutable tuple")
    if type(validation_examples) is not tuple:
        raise TypeError("validation split must be an immutable tuple")
    if not train_examples:
        raise ValueError("training split must be non-empty")
    if not validation_examples:
        raise ValueError("validation split must be non-empty")
    if type(model_config) is not TacticalV3ModelConfig:
        raise TypeError("model_config must be TacticalV3ModelConfig")
    if type(objective_config) is not ObjectiveConfig:
        raise TypeError("objective_config must be ObjectiveConfig")
    if type(trainer_config) is not TrainerConfig:
        raise TypeError("trainer_config must be TrainerConfig")
    train_rows = _canonical_split(train_examples, "training")
    validation_rows = _canonical_split(validation_examples, "validation")
    train_keys = {_canonical_example_key(example) for example in train_rows}
    validation_keys = {_canonical_example_key(example) for example in validation_rows}
    if train_keys & validation_keys:
        raise ValueError("splits overlap")

    random.seed(trainer_config.seed)
    np.random.seed(trainer_config.seed % (2**32))
    torch_seed = trainer_config.seed % (2**63 - 1)
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(torch_seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(trainer_config.device)
    model = TacticalV3Policy(model_config).to(device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=trainer_config.learning_rate, weight_decay=0.0
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(torch_seed)

    history: list[EpochMetrics] = []
    best_epoch = -1
    best_nll = math.inf
    best_state: Mapping[str, Tensor] | None = None
    epochs_without_improvement = 0
    stopped_early = False

    for epoch in range(trainer_config.max_epochs):
        permutation = torch.randperm(len(train_rows), generator=generator).tolist()
        train_weighted = {name: 0.0 for name in METRIC_KEYS}
        train_count = 0
        model.train()
        for batch_index, start in enumerate(
            range(0, len(permutation), trainer_config.batch_size)
        ):
            indices = permutation[start:start + trainer_config.batch_size]
            rows = tuple(train_rows[index] for index in indices)
            batch = _batch_to_device(
                _collate_training_batch(rows, model_config.horizon_turns), device
            )
            context = f"epoch={epoch} batch={batch_index}"
            _validate_batch_contract(batch, device, context)
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            _validate_policy_output(output, batch, device, context)
            losses = structured_imitation_loss(output, batch, objective_config)
            _validate_losses(losses, device, context)
            losses.total.backward()
            _after_backward(model, epoch=epoch, batch_index=batch_index)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and not bool(
                    torch.isfinite(parameter.grad).all()
                ):
                    raise FloatingPointError(f"{context} gradient={name}")
            gradient_norm = _clip_grad_norm(
                model.parameters(), trainer_config.gradient_clip_norm
            )
            if (
                not isinstance(gradient_norm, Tensor)
                or gradient_norm.ndim != 0
                or gradient_norm.device != device
                or not bool(torch.isfinite(gradient_norm))
            ):
                raise FloatingPointError(f"{context} gradient_norm")
            optimizer.step()
            _after_optimizer_step(model, epoch=epoch, batch_index=batch_index)
            for name, parameter in model.named_parameters():
                if not bool(torch.isfinite(parameter).all()):
                    raise FloatingPointError(f"{context} parameter={name}")
            for name in METRIC_KEYS:
                contribution = float(getattr(losses, name).detach().item()) * len(rows)
                if not math.isfinite(contribution):
                    raise FloatingPointError(f"{context} weighted loss.{name}")
                train_weighted[name] += contribution
            train_count += len(rows)

        train_metrics = _frozen_metrics(train_weighted, train_count)
        validation_metrics, candidate_nll = _evaluate_validation(
            model,
            validation_rows,
            model_config,
            objective_config,
            trainer_config.batch_size,
            device,
            epoch=epoch,
        )
        if type(candidate_nll) is not float or not math.isfinite(candidate_nll):
            raise FloatingPointError(f"epoch={epoch} validation policy")
        improved = candidate_nll < best_nll - 1e-12
        if improved:
            best_epoch = epoch
            best_nll = candidate_nll
            best_state = _snapshot_state(model)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        history.append(EpochMetrics(
            epoch=epoch,
            train=MappingProxyType(dict(train_metrics)),
            validation=MappingProxyType(dict(validation_metrics)),
            validation_policy_nll=candidate_nll,
            improved=improved,
        ))
        if epochs_without_improvement >= trainer_config.patience_epochs:
            stopped_early = True
            break

    if best_state is None or best_epoch < 0:
        raise RuntimeError("training did not produce a best state")
    _restore_state(model, best_state, device)
    model.eval()
    return TrainingResult(
        model=model,
        best_epoch=best_epoch,
        best_validation_policy_nll=float(best_nll),
        stopped_early=stopped_early,
        history=tuple(history),
    )
```

`TrainerConfig.__post_init__` accepts seed `0`; `seed` is otherwise a nonnegative built-in `int`, and `batch_size`, `max_epochs`, and `patience_epochs` are positive built-in `int`s. `learning_rate` and `gradient_clip_norm` are finite positive built-in `float`s. Device input is exactly `cpu`, `cuda`, or `cuda:<nonnegative decimal index>`. After checking availability, bare `cuda` resolves through `torch.cuda.current_device()` and is stored as canonical `cuda:<current-index>`; explicit CUDA indices are range-checked and stored indexed. Downstream helpers receive `torch.device(trainer_config.device)`, avoiding comparisons between unindexed `cuda` and indexed tensor devices. Every field rejects bool, Tensor, NumPy scalar, and the wrong built-in numeric/string class. The invalid matrix below is normative.

Before seeding or model construction, reject empty splits, duplicate canonical keys within either split, and equal keys across splits. Sort both immutable tuples by `_canonical_example_key`; one private CPU `torch.Generator`, seeded once with `TrainerConfig.seed`, produces exactly one `torch.randperm` per train epoch. Collate each selected slice in that order. Validation receives the sorted validation tuple in canonical contiguous batches and never consumes the generator. SHA-256 identities are trace labels only, never ordering or overlap keys.

The training loop checks all nested batch tensors are on the model device while preserving float32/int64/bool dtypes and checks every mask has bool dtype. Candidate logits must be finite at `candidates.mask` and exactly negative infinity at false positions. All auxiliary heads and all five `LossBreakdown` scalar tensors must be finite on the model device. After `total.backward()`, call `_after_backward`, allow `None` gradients, reject every nonfinite non-`None` gradient, call `_clip_grad_norm`, reject a nonfinite returned norm, call `AdamW.step`, call `_after_optimizer_step`, and immediately reject every nonfinite parameter. Errors contain `epoch=<n> batch=<n> <field>`; a post-step parameter failure is intentionally non-rollback.

Train and validation metrics accumulate each detached batch loss multiplied by the actual batch length and divide by the total example count. `_evaluate_validation` calls `_validation_batch_losses` for each canonical batch; this is the only scripted validation seam, so aggregation and epoch selection remain real. The exact comparator is `candidate_nll < best_nll - 1e-12`; equality within that tolerance is non-improvement. Restore the detached best state on the configured device. `EpochMetrics.train` and `.validation` are new `MappingProxyType` mappings in exact key order `total`, `policy`, `outcome`, `horizon`, `remaining_turns`; values and `validation_policy_nll` are finite built-in floats, and `validation_policy_nll == validation["policy"]` exactly. Do not checkpoint, publish, create run artifacts, modify the CLI, or write metadata in Task 8.

- [ ] **Step 1: Write failing trainer tests and complete helper code**

```python
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Iterator, Literal, Mapping

import numpy as np
import pytest
import torch
from torch import Tensor, nn

import ml_lab.tactical_v3_training as training
from ml_lab.tactical_v3_batching import CandidateBatch, RaggedBatch, RelationNeighborhoodBatch, TokenTableBatch, collate_examples
from ml_lab.tactical_v3_corpus import StructuredExample, load_corpus
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import CandidateIdentity, PolicyOutput, TacticalV3Policy
from ml_lab.tactical_v3_objectives import LossBreakdown, ObjectiveConfig, structured_imitation_loss
from ml_lab.tactical_v3_schema import parse_spaces
from ml_lab.tactical_v3_training import EpochMetrics, TrainerConfig, TrainingResult, train_offline

FIXTURES = Path(__file__).parent / "fixtures" / "tactical_v3"
METRIC_KEYS = ("total", "policy", "outcome", "horizon", "remaining_turns")

@dataclass(frozen=True, slots=True)
class TrainerTestCase:
    train: tuple
    validation: tuple
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig
    trainer_config: TrainerConfig

@dataclass(frozen=True, slots=True)
class FaultResult:
    error: BaseException
    optimizer_steps: int
    before_state_sha256: str
    after_state_sha256: str
    result: TrainingResult | None

@dataclass(slots=True)
class TrainingTrace:
    optimizer_orders: list[tuple] = field(default_factory=list)
    validation_orders: list[tuple] = field(default_factory=list)

def stable_example_identity(example: StructuredExample) -> str:
    key = training._canonical_example_key(example)
    return hashlib.sha256(repr(key).encode("utf-8")).hexdigest()

def make_trainer_case(*, device: str = "cpu", max_epochs: int = 6, batch_size: int = 4, patience_epochs: int = 6) -> TrainerTestCase:
    spaces = parse_spaces(json.loads((FIXTURES / "seed-41-spaces.json").read_text(encoding="utf-8")))
    corpus = load_corpus(FIXTURES / "tiny-corpus", spaces)
    return TrainerTestCase(corpus.train, corpus.validation, TacticalV3ModelConfig(), ObjectiveConfig(), TrainerConfig(seed=227, batch_size=batch_size, max_epochs=max_epochs, patience_epochs=patience_epochs, device=device))

def run_training_case(case: TrainerTestCase) -> TrainingResult:
    return train_offline(case.train, case.validation, case.model_config, case.objective_config, case.trainer_config)

def assert_state_dict_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None:
    assert tuple(left) == tuple(right)
    for name in left:
        assert left[name].dtype == right[name].dtype
        assert left[name].device == right[name].device
        assert left[name].shape == right[name].shape
        assert torch.equal(left[name], right[name]), name

def state_dict_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8")); digest.update(str(value.dtype).encode("ascii")); digest.update(repr(tuple(value.shape)).encode("ascii")); digest.update(value.numpy().tobytes())
    return digest.hexdigest()

def iter_batch_tensors(batch: RaggedBatch) -> Iterator[Tensor]:
    def walk(value: object) -> Iterator[Tensor]:
        if isinstance(value, Tensor):
            yield value
        elif dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field_info in dataclasses.fields(value):
                yield from walk(getattr(value, field_info.name))
        elif isinstance(value, Mapping):
            for nested in value.values():
                yield from walk(nested)
        elif isinstance(value, (tuple, list)):
            for nested in value:
                yield from walk(nested)
    yield from walk(batch)

def mapping_proxy_batch(batch: RaggedBatch) -> RaggedBatch:
    tables = MappingProxyType({
        name: TokenTableBatch(
            table.numeric,
            MappingProxyType(dict(table.categorical)),
            MappingProxyType(dict(table.boolean)),
            table.mask,
        )
        for name, table in batch.tables.items()
    })
    return dataclasses.replace(
        batch, tables=tables,
        table_slices=MappingProxyType(dict(batch.table_slices)),
    )

@contextmanager
def capture_training_trace(trace: TrainingTrace) -> Iterator[None]:
    original_collate = training._collate_training_batch
    original_evaluate = training._evaluate_validation

    def traced_collate(
        rows: tuple, horizons: tuple,
    ) -> RaggedBatch:
        trace.optimizer_orders.append(
            tuple(stable_example_identity(row) for row in rows)
        )
        return original_collate(rows, horizons)

    def traced_evaluate(
        model: TacticalV3Policy,
        rows: tuple,
        model_config: TacticalV3ModelConfig,
        objective_config: ObjectiveConfig,
        batch_size: int,
        device: torch.device,
        *,
        epoch: int,
    ) -> tuple[Mapping[str, float], float]:
        trace.validation_orders.append(
            tuple(stable_example_identity(row) for row in rows)
        )
        return original_evaluate(
            model, rows, model_config, objective_config, batch_size, device,
            epoch=epoch,
        )

    training._collate_training_batch = traced_collate
    training._evaluate_validation = traced_evaluate
    try:
        yield
    finally:
        training._collate_training_batch = original_collate
        training._evaluate_validation = original_evaluate


def scripted_validation_batch_losses(
    script: tuple,
    states: dict[int, str],
    observed: list[tuple[int, int, int]],
) -> Callable:
    def compute(
        model: TacticalV3Policy,
        batch: RaggedBatch,
        objective_config: ObjectiveConfig,
        *,
        epoch: int,
        batch_index: int,
    ) -> LossBreakdown:
        del objective_config
        states.setdefault(epoch, state_dict_sha256(model.state_dict()))
        observed.append((epoch, batch_index, int(batch.node_mask.shape[0])))
        value = torch.tensor(
            script[epoch][batch_index],
            dtype=torch.float32,
            device=next(model.parameters()).device,
        )
        return LossBreakdown(value, value, value, value, value)

    return compute


def assert_valid_padded_negative_infinity_case() -> None:
    case = make_trainer_case(
        max_epochs=1, batch_size=2, patience_epochs=1
    )
    batch = collate_examples(case.train[:2], case.model_config.horizon_turns)
    assert bool((~batch.candidates.mask).any())
    torch.manual_seed(case.trainer_config.seed)
    model = TacticalV3Policy(case.model_config).eval()
    with torch.no_grad():
        output = model(batch)
    assert torch.isfinite(
        output.candidate_logits[batch.candidates.mask]
    ).all()
    assert torch.isneginf(
        output.candidate_logits[~batch.candidates.mask]
    ).all()
    assert torch.isfinite(output.outcome_logits).all()
    assert torch.isfinite(output.horizon_logits).all()
    assert torch.isfinite(output.remaining_turns).all()
    result = run_training_case(case)
    assert tuple(metric.epoch for metric in result.history) == (0,)


def run_fault_case(
    stage: Literal[
        "valid_logit_nan", "valid_logit_neg_inf", "outcome", "horizon",
        "remaining", "policy", "outcome_loss", "horizon_loss",
        "remaining_loss", "total", "mask", "gradient", "clip", "parameter",
    ],
) -> FaultResult:
    case = make_trainer_case(max_epochs=1, patience_epochs=1)
    steps = 0
    captured_model: TacticalV3Policy | None = None
    before = ""
    original_init = TacticalV3Policy.__init__
    original_forward = TacticalV3Policy.forward
    original_loss = training.structured_imitation_loss
    original_transfer = training._batch_to_device
    original_after_backward = training._after_backward
    original_clip = training._clip_grad_norm
    original_after_step = training._after_optimizer_step

    def initialize(
        model: TacticalV3Policy, config: TacticalV3ModelConfig,
    ) -> None:
        nonlocal captured_model, before
        original_init(model, config)
        captured_model = model
        before = state_dict_sha256(model.state_dict())

    def forward(model: TacticalV3Policy, batch: RaggedBatch) -> PolicyOutput:
        output = original_forward(model, batch)
        if stage in {"valid_logit_nan", "valid_logit_neg_inf"}:
            logits = output.candidate_logits.clone()
            row = int(torch.nonzero(
                batch.candidates.mask[0], as_tuple=False
            )[0, 0])
            logits[0, row] = (
                float("nan") if stage == "valid_logit_nan" else float("-inf")
            )
            return dataclasses.replace(output, candidate_logits=logits)
        field_name = {
            "outcome": "outcome_logits",
            "horizon": "horizon_logits",
            "remaining": "remaining_turns",
        }.get(stage)
        if field_name is None:
            return output
        changed = getattr(output, field_name).clone()
        changed.reshape(-1)[0] = float("nan")
        return dataclasses.replace(output, **{field_name: changed})

    def loss(
        output: PolicyOutput, batch: RaggedBatch, config: ObjectiveConfig,
    ) -> LossBreakdown:
        value = original_loss(output, batch, config)
        field_name = {
            "policy": "policy",
            "outcome_loss": "outcome",
            "horizon_loss": "horizon",
            "remaining_loss": "remaining_turns",
            "total": "total",
        }.get(stage)
        if field_name is None:
            return value
        return dataclasses.replace(
            value,
            **{field_name: getattr(value, field_name) * float("nan")},
        )

    def transfer(batch: RaggedBatch, device: torch.device) -> RaggedBatch:
        moved = original_transfer(batch, device)
        if stage != "mask":
            return moved
        return dataclasses.replace(
            moved,
            candidates=dataclasses.replace(
                moved.candidates,
                mask=moved.candidates.mask.to(torch.int64),
            ),
        )

    def after_backward(
        model: TacticalV3Policy, *, epoch: int, batch_index: int,
    ) -> None:
        original_after_backward(model, epoch=epoch, batch_index=batch_index)
        if stage == "gradient":
            parameter = next(
                value for value in model.parameters() if value.grad is not None
            )
            with torch.no_grad():
                parameter.grad.reshape(-1)[0] = float("nan")

    def clip(parameters: Iterable[nn.Parameter], max_norm: float) -> Tensor:
        result = original_clip(parameters, max_norm)
        if stage == "clip":
            return torch.tensor(float("inf"), device=result.device)
        return result

    def after_step(
        model: TacticalV3Policy, *, epoch: int, batch_index: int,
    ) -> None:
        nonlocal steps
        original_after_step(model, epoch=epoch, batch_index=batch_index)
        steps += 1
        if stage == "parameter":
            parameter = next(model.parameters())
            with torch.no_grad():
                parameter.reshape(-1)[0] = float("inf")

    TacticalV3Policy.__init__ = initialize
    TacticalV3Policy.forward = forward
    training.structured_imitation_loss = loss
    training._batch_to_device = transfer
    training._after_backward = after_backward
    training._clip_grad_norm = clip
    training._after_optimizer_step = after_step
    try:
        try:
            run_training_case(case)
        except BaseException as caught:
            error = caught
        else:
            error = AssertionError("fault did not fail")
    finally:
        TacticalV3Policy.__init__ = original_init
        TacticalV3Policy.forward = original_forward
        training.structured_imitation_loss = original_loss
        training._batch_to_device = original_transfer
        training._after_backward = original_after_backward
        training._clip_grad_norm = original_clip
        training._after_optimizer_step = original_after_step
    assert captured_model is not None and before
    after = state_dict_sha256(captured_model.state_dict())
    return FaultResult(error, steps, before, after, None)
```

```python
INTEGER_FIELDS = ("seed", "batch_size", "max_epochs", "patience_epochs")
POSITIVE_INTEGER_FIELDS = ("batch_size", "max_epochs", "patience_epochs")
FLOAT_FIELDS = ("learning_rate", "gradient_clip_norm")
CONFIG_INVALID_CASES: tuple = (
    *((field, True) for field in (*INTEGER_FIELDS, *FLOAT_FIELDS, "device")),
    *((field, torch.tensor(1)) for field in INTEGER_FIELDS),
    *((field, np.int64(1)) for field in INTEGER_FIELDS),
    *((field, 1.0) for field in INTEGER_FIELDS),
    ("seed", -1),
    *((field, value) for field in POSITIVE_INTEGER_FIELDS for value in (0, -1)),
    *((field, torch.tensor(1.0)) for field in FLOAT_FIELDS),
    *((field, np.float64(1.0)) for field in FLOAT_FIELDS),
    *((field, 1) for field in FLOAT_FIELDS),
    *((field, value) for field in FLOAT_FIELDS for value in (
        0.0, -1.0, float("nan"), float("inf"), float("-inf"),
    )),
    ("device", torch.tensor(0)),
    ("device", np.str_("cpu")),
    ("device", torch.device("cpu")),
    *(("device", value) for value in (
        "", " cpu", "cpu:0", "mps", "CUDA", "cuda:x", "cuda:-1", "cuda:0:1",
    )),
)


def test_seed_zero_is_the_only_nonpositive_integer_exception() -> None:
    assert TrainerConfig(seed=0).seed == 0


def test_public_train_offline_interface_binds_a_callable() -> None:
    assert train_offline is training.train_offline
    assert callable(train_offline)

@pytest.mark.parametrize(("field", "value"), CONFIG_INVALID_CASES)
def test_every_config_field_rejects_its_invalid_type_and_domain_matrix(
    field: str, value: object,
) -> None:
    with pytest.raises(ValueError, match=field):
        TrainerConfig(**{field: value})


def test_cuda_device_availability_index_and_bare_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="device.*CUDA.*unavailable"):
        TrainerConfig(device="cuda")
    with pytest.raises(ValueError, match="device.*CUDA.*unavailable"):
        TrainerConfig(device="cuda:0")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    bare = TrainerConfig(device="cuda")
    assert bare.device == "cuda:1"
    assert torch.device(bare.device) == torch.device("cuda:1")
    explicit = TrainerConfig(device="cuda:0")
    assert explicit.device == "cuda:0"
    assert torch.device(explicit.device) == torch.device("cuda:0")
    with pytest.raises(ValueError, match="device.*index 2.*unavailable"):
        TrainerConfig(device="cuda:2")


def test_recursive_mapping_proxy_transfer_preserves_every_tensor_and_dtype() -> None:
    case = make_trainer_case()
    source = mapping_proxy_batch(collate_examples(
        case.validation[:2], case.model_config.horizon_turns
    ))
    source_tensors = tuple(iter_batch_tensors(source))
    expected_count = (
        sum(
            2 + len(table.categorical) + len(table.boolean)
            for table in source.tables.values()
        )
        + len(dataclasses.fields(RelationNeighborhoodBatch))
        + len(dataclasses.fields(CandidateBatch))
        + sum(
            isinstance(getattr(source, field_info.name), Tensor)
            for field_info in dataclasses.fields(RaggedBatch)
        )
    )
    assert len(source_tensors) == expected_count
    snapshots = tuple(value.clone() for value in source_tensors)
    moved = training._batch_to_device(source, torch.device("cpu"))
    moved_tensors = tuple(iter_batch_tensors(moved))
    assert type(moved.tables) is MappingProxyType
    assert type(moved.table_slices) is MappingProxyType
    assert moved.table_slices == source.table_slices
    assert moved.table_slices is not source.table_slices
    assert all(
        type(table.categorical) is MappingProxyType
        and type(table.boolean) is MappingProxyType
        for table in moved.tables.values()
    )
    with pytest.raises(TypeError):
        moved.tables["cells"] = moved.tables["cells"]  # type: ignore[index]
    assert len(source_tensors) == len(moved_tensors)
    for original, snapshot, transferred in zip(
        source_tensors, snapshots, moved_tensors, strict=True
    ):
        assert original.device.type == "cpu"
        assert transferred.device.type == "cpu"
        assert original.dtype == transferred.dtype
        assert torch.equal(original, snapshot)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("requested", ("cuda", "cuda:0"))
def test_cuda_training_uses_canonical_index_for_all_tensors_and_selection(
    requested: str,
) -> None:
    expected_index = torch.cuda.current_device() if requested == "cuda" else 0
    canonical = f"cuda:{expected_index}"
    case = make_trainer_case(
        device=requested, max_epochs=1, patience_epochs=1
    )
    assert case.trainer_config.device == canonical
    device = torch.device(case.trainer_config.device)
    assert device == torch.device(canonical)
    assert device.type == "cuda" and device.index == expected_index
    result = run_training_case(case)
    source = mapping_proxy_batch(collate_examples(
        case.validation[:1], case.model_config.horizon_turns
    ))
    batch = training._batch_to_device(source, device)
    assert all(tensor.device == device for tensor in iter_batch_tensors(batch))
    assert all(tensor.device.type == "cpu" for tensor in iter_batch_tensors(source))
    assert all(
        parameter.device == device and parameter.dtype == torch.float32
        for parameter in result.model.parameters()
    )
    output = result.model(batch)
    assert all(
        getattr(output, field_info.name).device == device
        for field_info in dataclasses.fields(PolicyOutput)
    )
    selected = result.model.select(batch)[0]
    legal = {
        (
            int(batch.candidates.decision_id[0, row]),
            int(batch.candidates.candidate_id[0, row]),
        )
        for row in torch.nonzero(
            batch.candidates.mask[0], as_tuple=False
        ).flatten()
    }
    assert (selected.decision_id, selected.candidate_id) in legal
def test_reversed_inputs_preserve_history_state_logits_actions_and_trace() -> None:
    case = make_trainer_case(max_epochs=2, patience_epochs=2)
    left_trace = TrainingTrace()
    right_trace = TrainingTrace()
    with capture_training_trace(left_trace):
        left = run_training_case(case)
    reversed_case = dataclasses.replace(
        case,
        train=tuple(reversed(case.train)),
        validation=tuple(reversed(case.validation)),
    )
    with capture_training_trace(right_trace):
        right = run_training_case(reversed_case)
    batch = collate_examples(case.validation, case.model_config.horizon_turns)
    assert left.history == right.history
    assert_state_dict_equal(left.model.state_dict(), right.model.state_dict())
    torch.testing.assert_close(
        left.model(batch).candidate_logits,
        right.model(batch).candidate_logits,
        rtol=0.0,
        atol=0.0,
    )
    assert left.model.select(batch) == right.model.select(batch)
    assert left_trace.optimizer_orders == right_trace.optimizer_orders
    assert left_trace.validation_orders == right_trace.validation_orders


def test_empty_duplicate_and_cross_split_keys_fail_before_model_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_trainer_case()

    def forbid_model_construction(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("model constructed before split validation")

    monkeypatch.setattr(
        training.TacticalV3Policy, "__init__", forbid_model_construction
    )
    invalid = (
        (dataclasses.replace(case, train=()), "training split must be non-empty"),
        (dataclasses.replace(case, validation=()), "validation split must be non-empty"),
        (
            dataclasses.replace(
                case, train=(case.train[0], dataclasses.replace(case.train[0]))
            ),
            "duplicate training example",
        ),
        (
            dataclasses.replace(
                case,
                validation=(case.validation[0], dataclasses.replace(case.validation[0])),
            ),
            "duplicate validation example",
        ),
        (
            dataclasses.replace(
                case, validation=(dataclasses.replace(case.train[0]),)
            ),
            "splits overlap",
        ),
    )
    for invalid_case, message in invalid:
        with pytest.raises(ValueError, match=message):
            run_training_case(invalid_case)


def test_real_weighted_validation_changes_ranking_and_restores_epoch_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_trainer_case(max_epochs=2, batch_size=2, patience_epochs=2)
    case = dataclasses.replace(base, validation=base.validation[:3])
    script = ((0.0, 1.5), (1.0, 0.0))
    states: dict[int, str] = {}
    observed: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        training,
        "_validation_batch_losses",
        scripted_validation_batch_losses(script, states, observed),
    )
    result = run_training_case(case)
    weighted = (0.5, 2.0 / 3.0)
    batch_means = tuple(sum(epoch_values) / 2.0 for epoch_values in script)
    assert weighted[0] < weighted[1]
    assert batch_means[1] < batch_means[0]
    assert observed == [
        (0, 0, 2), (0, 1, 1),
        (1, 0, 2), (1, 1, 1),
    ]
    assert tuple(
        metric.validation_policy_nll for metric in result.history
    ) == pytest.approx(weighted)
    assert tuple(
        metric.validation["policy"] for metric in result.history
    ) == pytest.approx(weighted)
    assert tuple(metric.improved for metric in result.history) == (True, False)
    assert result.best_epoch == 0
    assert result.best_validation_policy_nll == pytest.approx(weighted[0])
    assert state_dict_sha256(result.model.state_dict()) == states[0]


def test_real_loop_exact_history_patience_and_best_state_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = make_trainer_case(max_epochs=8, batch_size=2, patience_epochs=3)
    case = dataclasses.replace(base, validation=base.validation[:3])
    nlls = (0.8, 0.2, 0.4, 0.4, 0.4)
    script = tuple((value, value) for value in nlls)
    states: dict[int, str] = {}
    observed: list[tuple[int, int, int]] = []
    monkeypatch.setattr(
        training,
        "_validation_batch_losses",
        scripted_validation_batch_losses(script, states, observed),
    )
    result = run_training_case(case)
    assert tuple(
        metric.validation_policy_nll for metric in result.history
    ) == nlls
    assert tuple(metric.validation["policy"] for metric in result.history) == nlls
    assert tuple(metric.improved for metric in result.history) == (
        True, True, False, False, False,
    )
    assert tuple(metric.epoch for metric in result.history) == (0, 1, 2, 3, 4)
    assert result.best_epoch == 1
    assert result.best_validation_policy_nll == 0.2
    assert result.stopped_early
    assert state_dict_sha256(result.model.state_dict()) == states[1]


def test_padded_negative_inf_control_and_full_finite_fault_matrix() -> None:
    assert_valid_padded_negative_infinity_case()
    matrix = (
        ("valid_logit_nan", "candidate_logits", 0),
        ("valid_logit_neg_inf", "candidate_logits", 0),
        ("outcome", "outcome_logits", 0),
        ("horizon", "horizon_logits", 0),
        ("remaining", "remaining_turns", 0),
        ("policy", "loss.policy", 0),
        ("outcome_loss", "loss.outcome", 0),
        ("horizon_loss", "loss.horizon", 0),
        ("remaining_loss", "loss.remaining_turns", 0),
        ("total", "loss.total", 0),
        ("mask", "candidate_mask", 0),
        ("gradient", "gradient=", 0),
        ("clip", "gradient_norm", 0),
        ("parameter", "parameter=", 1),
    )
    for stage, field_name, expected_steps in matrix:
        fault = run_fault_case(stage)
        assert isinstance(fault.error, FloatingPointError)
        assert f"epoch=0 batch=0 {field_name}" in str(fault.error)
        assert fault.optimizer_steps == expected_steps
        if expected_steps == 0:
            assert fault.after_state_sha256 == fault.before_state_sha256
        else:
            assert fault.after_state_sha256 != fault.before_state_sha256


def test_epoch_metrics_are_exact_immutable_plain_float_maps() -> None:
    metric = run_training_case(make_trainer_case(
        max_epochs=1, patience_epochs=1
    )).history[0]
    assert type(metric.epoch) is int
    assert type(metric.improved) is bool
    assert type(metric.train) is MappingProxyType
    assert type(metric.validation) is MappingProxyType
    assert tuple(metric.train) == METRIC_KEYS
    assert tuple(metric.validation) == METRIC_KEYS
    assert all(
        type(value) is float and math.isfinite(value)
        for value in (*metric.train.values(), *metric.validation.values())
    )
    assert type(metric.validation_policy_nll) is float
    assert metric.validation_policy_nll == metric.validation["policy"]
    with pytest.raises(TypeError):
        metric.train["policy"] = 0.0  # type: ignore[index]
    with pytest.raises(TypeError):
        metric.validation["policy"] = 0.0  # type: ignore[index]
```

- [ ] **Step 2: Run RED**

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_training.py -q
```

- [ ] **Step 3: Implement and run GREEN**

Implement the exact interfaces and loop ordering above. Seed Python `random`, NumPy, CPU Torch, and every available CUDA generator from `TrainerConfig.seed`; enable deterministic Torch algorithms before model construction; use `AdamW` and no dropout. Each epoch consumes one CPU permutation, trains its contiguous batches through `_collate_training_batch`, validates exactly once through `_evaluate_validation`, appends one immutable `EpochMetrics`, updates or retains the detached best-state clone with the exact comparator, and stops only at the declared patience count. Load the best state before constructing `TrainingResult`; do not move it to CPU because Task 9 owns CPU publication.

```powershell
uv run --active --no-project python -m pytest python/tests/test_tactical_v3_training.py -q
git add python/ml_lab/tactical_v3_training.py python/tests/test_tactical_v3_training.py
git commit -m "feat: train tactical-v3 policy deterministically"
```

---
### Task 9: Strict Checkpoints, Deterministic Save/Load, and CPU Publication

**Files:**
- Create: `python/ml_lab/tactical_v3_checkpoint.py`
- Create: `python/tests/test_tactical_v3_checkpoint.py`
- Modify: `python/run_tactical_v3_imitation.py`

**Task-8 CLI handoff (owned and tested in Task 9 Step 1):** Task 9 alone adds persistent `train`. Change the entry point to `def main(argv: Sequence[str] | None = None) -> int`; add `train --corpus <Path> --scenario <Path> --run-dir <Path> --seed <int> --device <str>`. The command parses the scenario identity, loads the corpus against it, constructs default `TacticalV3ModelConfig` and `ObjectiveConfig` plus `TrainerConfig(seed=args.seed, device=args.device)`, calls `train_offline`, then calls `publish_structured_run(args.run_dir, result, corpus, args.scenario)` exactly once. CLI code does not open, create, write, rename, or delete any run artifact; publication owns all artifacts.
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

- [ ] **Step 1: Write failing CLI handoff, checkpoint, and publication tests**

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from ml_lab.tactical_v3_corpus import StructuredCorpus, load_corpus
from ml_lab.tactical_v3_layers import TacticalV3ModelConfig
from ml_lab.tactical_v3_model import TacticalV3Policy
from ml_lab.tactical_v3_objectives import ObjectiveConfig
from ml_lab.tactical_v3_schema import TacticalV3SemanticIdentity, parse_spaces
from ml_lab.tactical_v3_training import TrainerConfig, TrainingResult

FIXTURES = Path(__file__).parent / "fixtures" / "tactical_v3"


@dataclass(frozen=True, slots=True)
class TrainCliCase:
    identity: TacticalV3SemanticIdentity
    corpus: StructuredCorpus
    result: TrainingResult
    corpus_root: Path
    scenario_path: Path
    model_config: TacticalV3ModelConfig
    objective_config: ObjectiveConfig


def make_train_cli_case() -> TrainCliCase:
    scenario_path = FIXTURES / "seed-41-spaces.json"
    identity = parse_spaces(json.loads(
        scenario_path.read_text(encoding="utf-8")
    ))
    corpus_root = FIXTURES / "tiny-corpus"
    corpus = load_corpus(corpus_root, identity)
    model_config = TacticalV3ModelConfig()
    objective_config = ObjectiveConfig()
    model = TacticalV3Policy(model_config).eval()
    result = TrainingResult(
        model=model,
        best_epoch=0,
        best_validation_policy_nll=0.0,
        stopped_early=False,
        history=(),
    )
    return TrainCliCase(
        identity, corpus, result, corpus_root, scenario_path,
        model_config, objective_config,
    )


def test_train_cli_calls_real_sequence_then_only_publisher_owns_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import run_tactical_v3_imitation as cli

    case = make_train_cli_case()
    calls: list[str] = []
    scenario_payload = json.loads(case.scenario_path.read_text(encoding="utf-8"))
    run_dir = tmp_path / "run"
    assert not run_dir.exists()

    def fake_parse(payload: object) -> TacticalV3SemanticIdentity:
        assert payload == scenario_payload
        calls.append("parse")
        return case.identity

    def fake_load(
        root: Path, expected: TacticalV3SemanticIdentity,
    ) -> StructuredCorpus:
        assert root == case.corpus_root
        assert expected == case.identity
        calls.append("load")
        return case.corpus

    def fake_train(
        train_examples: tuple,
        validation_examples: tuple,
        model_config: TacticalV3ModelConfig,
        objective_config: ObjectiveConfig,
        trainer_config: TrainerConfig,
    ) -> TrainingResult:
        assert train_examples == case.corpus.train
        assert validation_examples == case.corpus.validation
        assert model_config == case.model_config
        assert objective_config == case.objective_config
        assert trainer_config == TrainerConfig(seed=0, device="cpu")
        calls.append("train")
        return case.result

    def fake_publish(
        destination: Path,
        result: TrainingResult,
        corpus: StructuredCorpus,
        scenario_path: Path,
    ) -> Path:
        assert destination == run_dir
        assert result is case.result
        assert corpus is case.corpus
        assert scenario_path == case.scenario_path
        calls.append("publish")
        return destination

    monkeypatch.setattr(cli, "parse_spaces", fake_parse)
    monkeypatch.setattr(cli, "load_corpus", fake_load)
    monkeypatch.setattr(cli, "train_offline", fake_train)
    monkeypatch.setattr(cli, "publish_structured_run", fake_publish)
    assert cli.main([
        "train",
        "--corpus", str(case.corpus_root),
        "--scenario", str(case.scenario_path),
        "--run-dir", str(run_dir),
        "--seed", "0",
        "--device", "cpu",
    ]) == 0
    assert calls == ["parse", "load", "train", "publish"]
    assert not run_dir.exists()
```

```python
def test_checkpoint_contains_only_whitelisted_plain_values_and_cpu_tensors(
    tmp_path: Path,
) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    path = save_checkpoint_case(case, tmp_path / "model.pt")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {"format_version", "metadata", "state_dict", "inference_fixture"}
    assert_checkpoint_value_whitelist(raw)
    assert all(tensor.device.type == "cpu" and tensor.is_contiguous()
               for tensor in raw["state_dict"].values())

def test_load_uses_weights_only_and_rejects_unknown_missing_or_wrong_typed_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    path = save_checkpoint_case(case, tmp_path / "model.pt")
    calls = spy_torch_load(monkeypatch)
    load_structured_checkpoint(path, case.identity.encoding_hash,
                               case.identity.capacity_hash)
    assert calls == [{"map_location": "cpu", "weights_only": True}]
    for mutation, message in (
        ("unknown_top_level", "checkpoint fields"),
        ("missing_state_dict", "checkpoint fields"),
        ("bool_best_epoch", "metadata.best_epoch"),
    ):
        bad = write_checkpoint_variant(path, tmp_path / f"{mutation}.pt", mutation)
        with pytest.raises((TypeError, ValueError), match=message):
            load_structured_checkpoint(bad, case.identity.encoding_hash,
                                       case.identity.capacity_hash)

def test_load_rejects_wrong_encoding_capacity_corpus_or_state_hash(tmp_path: Path) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    path = save_checkpoint_case(case, tmp_path / "model.pt")
    with pytest.raises(ValueError, match="encoding hash"):
        load_structured_checkpoint(path, "0" * 64, case.identity.capacity_hash)
    with pytest.raises(ValueError, match="capacity hash"):
        load_structured_checkpoint(path, case.identity.encoding_hash, "0" * 64)
    state_bad = write_checkpoint_variant(path, tmp_path / "state-bad.pt", "state_tensor")
    with pytest.raises(ValueError, match="model state SHA-256"):
        load_structured_checkpoint(state_bad, case.identity.encoding_hash,
                                   case.identity.capacity_hash)
    run_dir = publish_checkpoint_case(case, tmp_path / "run")
    mutate_json_field(run_dir / "run.json", "dataset_manifest_sha256", "0" * 64)
    with pytest.raises(ValueError, match="corpus SHA-256"):
        validate_structured_run(run_dir)

def test_two_cpu_saves_are_semantically_identical_after_strict_load(tmp_path: Path) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    first = load_checkpoint_case(case, save_checkpoint_case(case, tmp_path / "a.pt"))
    second = load_checkpoint_case(case, save_checkpoint_case(case, tmp_path / "b.pt"))
    assert first.metadata == second.metadata
    assert first.fixture == second.fixture
    assert_state_dict_equal(first.model.state_dict(), second.model.state_dict())

def test_cpu_save_load_preserves_logits_and_actions_exactly(tmp_path: Path) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    loaded = load_checkpoint_case(case, save_checkpoint_case(case, tmp_path / "model.pt"))
    logits, actions = fixture_logits_and_actions(loaded.model, case.fixture_examples)
    for actual, expected in zip(logits, case.fixture_logits, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert actions == case.fixture_actions

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_training_publication_matches_cpu_within_declared_tolerance(
    tmp_path: Path,
) -> None:
    case = make_checkpoint_case(tmp_path, device="cuda")
    loaded = load_checkpoint_case(case, save_checkpoint_case(case, tmp_path / "model.pt"))
    assert next(loaded.model.parameters()).device.type == "cpu"
    logits, actions = fixture_logits_and_actions(loaded.model, case.fixture_examples)
    for actual, expected in zip(logits, case.fixture_logits, strict=True):
        torch.testing.assert_close(actual, expected.cpu(), rtol=1e-5, atol=1e-6)
    assert actions == case.fixture_actions

def test_publish_is_atomic_refuses_overwrite_and_leaves_run_unsealed(
    tmp_path: Path,
) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    failed = tmp_path / "failed-run"
    with inject_publish_failure("after_checkpoint"):
        with pytest.raises(RuntimeError, match="injected after_checkpoint"):
            publish_checkpoint_case(case, failed)
    assert not failed.exists()
    assert not tuple(tmp_path.glob(".failed-run.tmp-*"))
    run_dir = publish_checkpoint_case(case, tmp_path / "run")
    manifest = read_json(run_dir / "run.json")
    assert manifest["evidence_status"] == "unsealed-experimental"
    assert manifest["config"]["algorithm"] == "structured_imitation"
    with pytest.raises(FileExistsError):
        publish_checkpoint_case(case, run_dir)

def test_validate_run_replays_immutable_inference_fixture(tmp_path: Path) -> None:
    case = make_checkpoint_case(tmp_path, device="cpu")
    run_dir = publish_checkpoint_case(case, tmp_path / "run")
    before = sha256_tree(run_dir)
    loaded = validate_structured_run(run_dir)
    logits, actions = fixture_logits_and_actions(loaded.model, case.fixture_examples)
    assert tuple(tuple(row.tolist()) for row in logits) == loaded.fixture.valid_candidate_logits
    assert actions == loaded.fixture.selected_identities
    assert sha256_tree(run_dir) == before
```
Helper contracts for this test file:

```python
@dataclass(frozen=True, slots=True)
class CheckpointTestCase:
    identity: TacticalV3SemanticIdentity
    result: TrainingResult
    corpus: StructuredCorpus
    scenario_path: Path
    metadata: StructuredCheckpointMetadata
    fixture_examples: tuple[StructuredExample, ...]
    fixture_logits: tuple[Tensor, ...]
    fixture_actions: tuple[CandidateIdentity, ...]

def make_checkpoint_case(tmp_path: Path, device: Literal["cpu", "cuda"]) -> CheckpointTestCase: ...
def save_checkpoint_case(case: CheckpointTestCase, path: Path) -> Path: ...
def load_checkpoint_case(case: CheckpointTestCase, path: Path) -> LoadedStructuredPolicy: ...
def publish_checkpoint_case(case: CheckpointTestCase, run_dir: Path) -> Path: ...
def assert_checkpoint_value_whitelist(value: object) -> None: ...
def spy_torch_load(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]: ...
def write_checkpoint_variant(source: Path, destination: Path, mutation: str) -> Path: ...
def mutate_json_field(path: Path, field: str, value: object) -> None: ...
def assert_state_dict_equal(left: Mapping[str, Tensor], right: Mapping[str, Tensor]) -> None: ...
def fixture_logits_and_actions(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
) -> tuple[tuple[Tensor, ...], tuple[CandidateIdentity, ...]]: ...
@contextmanager
def inject_publish_failure(stage: Literal["after_checkpoint"]) -> Iterator[None]: ...
def sha256_tree(root: Path) -> tuple[tuple[str, str], ...]: ...
```

The ellipsis declarations in the immediately preceding fence are retained only for Task 9's pre-existing general checkpoint-test helpers: they are outside the persistent-train CLI fixture and call path, and their concrete behavior is fully constrained by the following paragraph and checkpoint tests. The Task 9 CLI fence above is independently complete, declares every import and fixture path it uses, and calls none of these general checkpoint helpers.

The case loader trains the immutable tiny corpus with seed 227 on the requested device, selects the first two canonical validation rows as fixtures, computes valid-only logits and identities before publication, and builds metadata from the real corpus/state/semantic hashes. Save/load/publish helpers call only the public interfaces in this task with the case's exact expected identities. The whitelist recursively permits plain JSON scalars/lists/dicts outside `state_dict` and contiguous tensors only inside it. The load spy delegates to the original function and records only `map_location` and `weights_only`. Variant mutations respectively add a top-level key, remove `state_dict`, replace `best_epoch` with `True`, or add one to the first tensor without changing its recorded hash. JSON mutation rewrites one top-level field atomically for a deliberate tamper test. The injected publication failure fires after temporary checkpoint creation and requires cleanup before rethrow. Tree hashing returns sorted relative paths plus SHA-256 and never follows links.

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
def test_tactical_v3_requires_expected_capacity_hash(tmp_path: Path) -> None:
    case = make_policy_server_case(tmp_path)
    args = without_argument(case.args, "--expected-capacity-hash")
    result = run_policy_server_until_exit(args)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "--expected-capacity-hash is required for tactical-v3" in result.stderr

def test_legacy_expectation_rejects_capacity_hash() -> None:
    with pytest.raises(ValueError, match="capacity hash is valid only for tactical-v3"):
        PolicyExpectation(
            environment="tactical-v1",
            version="tactical-v1",
            encoding_hash="a" * 64,
            capacity_hash="b" * 64,
        )

def test_tactical_v3_request_returns_decision_and_candidate_identity(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    with start_policy_server(case.args) as server:
        assert server.ready["ready"] is True
        response = server.request({"seat": case.seat, "decision": case.view_payload})
    assert set(response) == {"decision_id", "candidate_id"}
    assert response["decision_id"] == case.view_payload["decision_id"]
    assert (response["decision_id"], response["candidate_id"]) in case.legal_identities

def test_tactical_v3_request_rejects_flat_or_mixed_payloads(tmp_path: Path) -> None:
    case = make_policy_server_case(tmp_path)
    invalid = (
        {"seat": case.seat, "obs": [0.0], "mask": [True]},
        {"seat": case.seat, "decision": case.view_payload,
         "obs": [0.0], "mask": [True]},
        {"seat": case.seat, "decision": case.view_payload, "extra": 1},
    )
    with start_policy_server(case.args) as server:
        for request in invalid:
            response = server.request(request)
            assert set(response) == {"error"}
            assert "structured policy request fields" in response["error"]

def test_structured_response_is_deterministic_across_server_restarts(
    tmp_path: Path,
) -> None:
    case = make_policy_server_case(tmp_path)
    first = request_once(case.args, {"seat": case.seat, "decision": case.view_payload})
    second = request_once(case.args, {"seat": case.seat, "decision": case.view_payload})
    assert first == second
    assert (first["decision_id"], first["candidate_id"]) in case.legal_identities

def test_wrong_encoding_or_capacity_fails_before_ready(tmp_path: Path) -> None:
    case = make_policy_server_case(tmp_path)
    for flag in ("--expected-encoding-hash", "--expected-capacity-hash"):
        args = replace_argument(case.args, flag, "0" * 64)
        result = run_policy_server_until_exit(args)
        assert result.returncode != 0
        assert result.stdout == ""
        assert "does not match expected" in result.stderr
```
Helper contracts for this test file:

```python
@dataclass(frozen=True, slots=True)
class PolicyServerCase:
    args: tuple[str, ...]
    seat: int
    view_payload: Mapping[str, object]
    legal_identities: frozenset[tuple[int, int]]

class PolicyServerProcess(AbstractContextManager["PolicyServerProcess"]):
    ready: Mapping[str, object]
    def request(self, payload: Mapping[str, object]) -> Mapping[str, object]: ...
    def close(self) -> None: ...

def make_policy_server_case(tmp_path: Path) -> PolicyServerCase: ...
def without_argument(args: tuple[str, ...], flag: str) -> tuple[str, ...]: ...
def replace_argument(args: tuple[str, ...], flag: str, value: str) -> tuple[str, ...]: ...
def run_policy_server_until_exit(args: tuple[str, ...]) -> subprocess.CompletedProcess[str]: ...
def start_policy_server(args: tuple[str, ...]) -> PolicyServerProcess: ...
def request_once(
    args: tuple[str, ...],
    payload: Mapping[str, object],
) -> Mapping[str, object]: ...
```

The case helper publishes a minimal valid CPU structured run through Task 9's public publication path, thaws the canonical seed-41 view, derives legal identities from its exact candidates, and builds argv with `sys.executable`, `python/policy_server.py`, `--p0 run:PATH`, and all four exact expectation flags. Argument helpers require the named flag exactly once and remove or replace its following value. The exit helper uses UTF-8 text pipes, a 30-second timeout, and returns captured stdout/stderr. The context helper starts the same command, reads exactly one JSON ready line, exposes request/reply JSONL, bounds stderr, and always sends close then terminates/reaps in `__exit__`. `request_once` opens that context, requires ready, sends one payload, returns its decoded reply, and closes.

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

`TacticalV3ObservationDto`, candidate/projection DTOs, and reward DTO use the exact snake_case fields and scalar/array types enumerated in `Resolved Project A Wire Contract`; there is no generic dictionary, reflection serializer, flattened observation, or mask.

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

Extend Arena manifest DTOs with `algorithm`, `contract_hash`, and `capacity_hash`. Resolve the selected run's contained `scenario.json` and `checkpoints/best.pt`, require unsealed experimental state, and compare environment/version/contract/encoding/capacity identities before altering the active configuration. Pass the unchanged `run:PATH` controller spec to `PolicyBridge`. Display "variable structured candidates" and the three semantic hashes.

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

def round_trip_via_policy_server(
    client: TacticalV3GymClient,
    run_dir: Path,
    identity: TacticalV3SemanticIdentity,
    seed: int,
) -> tuple[CandidateIdentity, TacticalV3View, TacticalV3View]: ...
```

Keep these as test helpers, not production API. They use the same strict parser, batcher, controller, and GymServer client as publication.

- [ ] **Step 1: Write the failing end-to-end tests**

```python
def test_tiny_corpus_overfits_deterministically(
    tmp_path: Path, server_dll: Path,
) -> None:
    case = make_end_to_end_case(tmp_path, server_dll)
    assert case.first_result.history == case.second_result.history
    assert state_dict_sha256(case.first_result.model.state_dict()) == state_dict_sha256(
        case.second_result.model.state_dict()
    )
    for examples in (case.corpus.train, case.corpus.validation):
        first = overfit_metrics(case.first_controller, examples)
        second = overfit_metrics(case.second_controller, examples)
        assert first == second
        assert first["policy_accuracy"] == 1.0
        assert first["policy_nll"] < 0.02
        assert first["finite_logit_count"] == total_valid_candidates(examples)

def test_13x9_gymserver_policy_server_candidate_identity_round_trip(
    tmp_path: Path, server_dll: Path,
) -> None:
    case = make_end_to_end_case(tmp_path, server_dll)
    command = gymserver_command(server_dll, CHECKED_IN_SCENARIO, role="tactical")
    with TacticalV3GymClient(command, environment_kind="tactical") as client:
        selection, before, after = round_trip_via_policy_server(
            client, case.first_run_dir, case.identity, seed=41
        )
    matches = [candidate for candidate in before.decision.candidates
               if candidate.candidate_id == selection.candidate_id]
    assert len(matches) == 1
    assert selection.decision_id == before.decision.decision_id
    assert after.terminated or after.truncated or (
        after.decision.decision_id != before.decision.decision_id
    )

def test_same_checkpoint_infers_legally_on_24x16_without_rebuild(
    tmp_path: Path, server_dll: Path,
) -> None:
    case = make_end_to_end_case(tmp_path, server_dll)
    before_parameter_count = sum(parameter.numel()
                                 for parameter in case.first_controller.policy.parameters())
    command = gymserver_command(server_dll, SCENARIO_24X16, role="tactical")
    with TacticalV3GymClient(command, environment_kind="tactical") as client:
        view = client.reset(41)
        assert client.identity.encoding_hash == case.identity.encoding_hash
        assert client.identity.capacity_hash == case.identity.capacity_hash
        assert client.identity.contract_hash != case.identity.contract_hash
        logits, selection = controller_logits_and_selection(case.first_controller, view)
    assert torch.isfinite(logits).all()
    assert selection.candidate_id in {
        candidate.candidate_id for candidate in view.decision.candidates
    }
    assert sum(parameter.numel()
               for parameter in case.first_controller.policy.parameters()) == before_parameter_count
    assert case.first_controller.policy.config == case.second_controller.policy.config

def test_two_publications_reload_with_identical_logits_and_actions(
    tmp_path: Path, server_dll: Path,
) -> None:
    case = make_end_to_end_case(tmp_path, server_dll)
    first = validate_structured_run(case.first_run_dir)
    second = validate_structured_run(case.second_run_dir)
    assert first.metadata.model_state_sha256 == second.metadata.model_state_sha256
    for examples in (case.corpus.train, case.corpus.validation):
        first_logits, first_actions = controller_fixture_outputs(first.model, examples)
        second_logits, second_actions = controller_fixture_outputs(second.model, examples)
        for left, right in zip(first_logits, second_logits, strict=True):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
        assert first_actions == second_actions
```
Helper contracts for this test file:

```python
@dataclass(frozen=True, slots=True)
class EndToEndCase:
    corpus: StructuredCorpus
    identity: TacticalV3SemanticIdentity
    first_result: TrainingResult
    second_result: TrainingResult
    first_run_dir: Path
    second_run_dir: Path
    first_controller: StructuredController
    second_controller: StructuredController

def make_end_to_end_case(tmp_path: Path, server_dll: Path) -> EndToEndCase: ...
def total_valid_candidates(examples: tuple[StructuredExample, ...]) -> int: ...
def gymserver_command(
    server_dll: Path,
    scenario: Path,
    role: Literal["tactical", "duel"],
) -> tuple[str, ...]: ...
def round_trip_via_policy_server(
    client: TacticalV3GymClient,
    run_dir: Path,
    identity: TacticalV3SemanticIdentity,
    seed: int,
) -> tuple[CandidateIdentity, TacticalV3View, TacticalV3View]: ...
def controller_logits_and_selection(
    controller: StructuredController,
    view: TacticalV3View,
) -> tuple[Tensor, CandidateIdentity]: ...
def controller_fixture_outputs(
    model: TacticalV3Policy,
    examples: tuple[StructuredExample, ...],
) -> tuple[tuple[Tensor, ...], tuple[CandidateIdentity, ...]]: ...
```

`make_end_to_end_case` builds/loads the checked-in corpus, runs `train_offline` twice from freshly constructed seed-227 CPU configs, publishes to two absent sibling directories, validates both runs, and loads both structured controllers with exact encoding/capacity hashes. `overfit_metrics` collates examples in canonical order, runs eval/inference mode, and returns mean target NLL, exact target-row accuracy, and the number of finite true-masked logits. The GymServer argv is exactly `(dotnet, server_dll, --scenario, scenario, --environment, tactical-v3, --role, role)`. The round-trip helper resets the real client, projects the frozen view back to the exact Task 1 wire fields, starts the real policy server with all expectation flags, sends one structured request, requires an exact identity reply, converts it to `CandidateSelection`, and calls the real client step before closing both processes. Controller logit helpers use `collate_decisions` for target-free inference, return valid-only CPU logits, and preserve exact candidate identities.

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

Use Coplay against this worktree in order: `set_unity_project_root`; wait for import; `check_compile_errors`; run the six affected EditMode classes from Tasks 11-12; run the ML Lab selected-run test with `python/runs/tactical-v3-policy-project-b-acceptance`; `get_unity_logs`; then `check_compile_errors` again. Expected: full .NET green, zero Unity compilation/test errors, the selected structured run loads into Arena with its capacity identity, and no new exceptions/errors in logs. If the Editor is unavailable, report the missing gate and do not claim completion.

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
