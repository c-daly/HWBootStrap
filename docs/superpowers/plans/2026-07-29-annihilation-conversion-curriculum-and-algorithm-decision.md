# Annihilation Conversion Curriculum and Algorithm Decision Plan

**Goal:** Determine whether HexWars can reliably teach annihilation conversion
against Random by increasing practice on mechanically valid late-game pursuit
states, then use controlled algorithm comparisons to identify any remaining
exploration, persistence, search, or sample-reuse limitation.

**Architecture:** Extend tactical-v2 only as an experimental substrate. Keep the
model geometry at three unit slots while allowing a versioned catalog of
learner-relative start profiles with fewer live units. The catalog is model
semantics; per-run sampling weights are immutable scenario provenance and do not
change the observation/action encoding. Train fresh standard and curriculum
controls under the same new contract. Keep target-KL PPO as the incumbent,
measure conversion on profile-forced evaluation suites, and admit another
algorithm only when it tests a named failure hypothesis.

**Authority boundary:** This work validates training ideas for tactical-v3; it
does not turn tactical-v2 into the long-term general agent. Tactical-v3 remains
intentionally checkpoint-incompatible and owns variable entities, legal-command
candidate scoring, persistent intentions, auxiliary heads, and later self-play.

**Tech stack:** C#/.NET engine and GymServer, NUnit, JSONL protocol, Python 3.11+,
Gymnasium, SB3/sb3-contrib, PyTorch, pytest, TensorBoard, deterministic replay and
draw-classification tooling.

## Evidence motivating the plan

- Reward ablation did not make PPO competent. Dense value progress was useful;
  terminal-only reward was not.
- Target KL 0.02 improved 51,200-step conversion relative to default PPO and
  eliminated the extreme KL/clip pathology.
- Continuing those models to 100,352 steps did not improve the paired aggregate:
  52/150, 47/150, and 50/150 wins at 51,200, 75,776, and 100,352.
- Final mean approximate KL remained 0.018 and clip fraction 0.081, so the plateau
  was behavioral rather than a recurrence of optimizer explosion.
- Every inspected final-checkpoint draw reached round 100 and cycled.
- The same fixed models varied from 34/150 to 52/150 wins across two 25-map seed
  blocks. Future promotion decisions require more independent maps.

Inputs:

- `python/panels/annihilation-reduced-v1/REPORT.md`
- `python/panels/annihilation-optimizer-v1/REPORT.md`
- `python/panels/annihilation-targetkl-continuation-v1/REPORT.md`
- `docs/superpowers/specs/2026-07-28-tactical-v3-general-rl-design.md`

## Global constraints

- An annihilation win is the primary outcome. A draw is never counted as a win.
- Keep draw terminal credit at zero and closing-distance reward at zero.
- Keep value-delta shaping, target KL 0.02, learning rate 3e-4, and ten nominal
  PPO epochs unless a named algorithm condition explicitly changes them.
- Random remains the training opponent for the curriculum decision. Greedy is an
  evaluation control, not yet a training opponent.
- Keep `MaxControllableUnits = 3` for every profile. Empty live slots are valid;
  observation and action geometry must not vary with a start profile.
- Do not implement the curriculum by setting `StartingUnitCount` to one or two.
  The current equality between starting count and slot capacity would create a
  different encoding and make comparisons invalid.
- Existing tactical-v2 starts, scenarios, hashes, checkpoints, and replays remain
  readable. The legacy `symmetric-random-v1` path remains byte-identical.
- New profiled-start models use a new contract hash. Do not resume an old
  tactical-v2 checkpoint into the new contract.
- The standard-control and mixed-curriculum scenarios declare the same profile
  catalog and therefore the same model contract. Only immutable sampler weights
  differ.
- Start-distribution weights affect scenario/run provenance, not encoding hash.
  They must be inherited exactly on resume and included in panel manifests.
- Profile selection and construction are C# engine authority. Python sends seeds
  and explicit evaluation profile IDs; it never constructs a `GameState`.
- Profile construction must fail closed when a requested state is mechanically
  invalid or cannot contain a plausible learner path to positive damage.
- Evaluation can force only a profile declared by the active contract. There is
  no generic arbitrary-state injection RPC.
- Reciprocal evaluation means the profile's advantaged side follows the candidate
  controller, not always player 0.
- Algorithm comparisons use the same contracts, profile schedules, training
  seeds, and evaluation banks. Report environment steps, optimizer updates,
  wall-clock time, and inference/search cost; timesteps alone are not equal
  compute across algorithms.
- This six-model panel is a diagnostic decision experiment. It does not satisfy
  tactical-v3's official frozen-policy tournament requirement of roughly 500
  paired seeds per major matchup, and its report must not make that claim.
- Retaining tactical-v2's per-command step penalty is an isolation choice for
  this panel, not approval of that reward for tactical-v3. Tactical-v3's planned
  round/turn-based time pressure remains authoritative for the new contract.
- Existing unrelated working-tree changes remain untouched.

## Contract and scenario design

### Profile catalog

Add a profiled placement policy, `profiled-seeded-v1`. A catalog entry has the
following exact semantic fields:

```json
{
  "id": "conversion-2v1-far",
  "learner_units": 2,
  "opponent_units": 1,
  "separation": "far"
}
```

The shared catalog for the first panel contains:

```text
standard-3v3
conversion-3v1-near
conversion-3v1-medium
conversion-3v1-far
conversion-2v1-near
conversion-2v1-medium
conversion-2v1-far
conversion-1v1-near
conversion-1v1-medium
conversion-1v1-far
```

`standard-3v3` has `learner_units = 3`, `opponent_units = 3`, and
`separation = "legacy-mirrored"`; it uses the existing mirrored deployment-zone
constructor exactly. Only this exact profile may use `legacy-mirrored`.
Conversion profiles use deterministic learner-relative construction. Separation
bands are versioned engine semantics, not names interpreted independently by
Python. For this 13x9 panel, the minimum initial hex distance between opposing
units is 2-3 for `near`, 4-6 for `medium`, and at least 7 for `far`. A profile is
invalid if the board cannot satisfy its band.

Conversion armies sample learner and opponent templates independently with
replacement from the same declared template catalog using separate RNG domains.
The start record retains the exact ordered template IDs for both seats. Standard
3-v-3 keeps the old one-composition-for-both-seats behavior byte for byte.

### Distribution

Represent sampling weights as integer basis points summing to 10,000. Zero is
valid so a scenario can declare a profile for evaluation without sampling it in
training.

The control scenario uses:

```text
standard-3v3 = 10000
all conversion profiles = 0
```

The mixed scenario uses:

```text
standard-3v3 = 4000
each near/far conversion profile = 1000
each medium conversion profile = 0
```

This makes 60% of training episodes conversion practice, balanced over 3-v-1,
2-v-1, and 1-v-1. Medium separation is a declared but held-out interpolation
test. Profile sampling is a deterministic function of episode seed.

### Identity rules

- `contract_hash` includes the profile catalog and placement-policy version but
  excludes distribution weights.
- `encoding_hash` continues to depend on slot capacity, cells, channels,
  templates, and action regions; it must be identical between the two panel
  scenarios.
- `scenario.json` includes both catalog and weights. The panel manifest records
  its SHA-256 hash.
- Resume preserves the source scenario, including weights. A CLI attempt to
  override weights during resume is rejected.
- Evaluation output records forced profile ID, reference/candidate seat, scenario
  hash, contract hash, and encoding hash.

## File and interface map

### Engine and GymServer

- `engine/HexWars.Engine/Rl/TacticalV2Config.cs`
- `engine/HexWars.Engine/Rl/TacticalV2Layout.cs`
- `engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs`
- `engine/HexWars.Engine/Rl/TacticalV2Env.cs`
- `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs`
- `engine/HexWars.Engine/Rl/MlContract.cs`
- `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- `engine/HexWars.GymServer/ScenarioJson.cs`
- `engine/HexWars.GymServer/Program.cs`

### Engine tests

- `engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs`
- `engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs`
- `engine/HexWars.Engine.Tests/MlContractTests.cs`
- `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs`

### Python harness

- `python/hexwars_gym/env.py`
- `python/ml_lab/contracts.py`
- `python/ml_lab/scenarios.py`
- `python/ml_lab/envs.py`
- `python/ml_lab/evaluation.py`
- `python/ml_lab/cli.py`
- `python/ml_lab/protocol.py`
- `python/ml_lab/controllers.py`
- `python/ml_lab/algorithms.py`
- `python/config/training-game-templates.json`
- `python/config/experiments/`

### Python tests

- `python/tests/test_scenarios.py`
- `python/tests/test_gym_client.py`
- `python/tests/test_protocol.py`
- `python/tests/test_run_contract.py`
- `python/tests/test_training.py`
- `python/tests/test_evaluation.py`
- `python/tests/test_controllers.py`
- `python/tests/test_algorithms.py`

### Unity editor compatibility

- `Assets/HexWars/Editor/MlLab/MlTrainingScenario.cs`
- `Assets/HexWars/Editor/MlLab/MlLabConfig.cs`
- `Assets/HexWars/Editor/MlLab/MlLabWindow.cs`
- corresponding EditMode tests under `Assets/HexWars/Tests/Editor/`

The first implementation need not add a friendly curriculum authoring UI. Unity
must round-trip, validate, display, and launch checked-in profiled scenarios
without dropping fields. A richer editor surface is deferred until the research
result justifies it.

---

## Task 1: Freeze experiment identities and seed banks

**Files:**

- Create: `python/panels/annihilation-conversion-curriculum-v1/PROTOCOL.md`
- Create: `python/panels/annihilation-conversion-curriculum-v1/panel.json`
- Create: `python/panels/annihilation-conversion-curriculum-v1/seed-banks.json`

- [ ] Record SHA-256 hashes for the three prior panel aggregates, the three
  51,200 target-KL source checkpoints, their run manifests, and scenarios.
- [ ] Reserve fresh training seeds `101`, `113`, and `127` for both conditions.
- [ ] Reserve seed bank `6,000,000..6,000,009` for smoke/development only.
- [ ] Reserve standard evaluation map banks `7,000,000..7,000,024` and
  `8,000,000..8,000,024`. Both are used for the locked panel, yielding 50 unique
  standard maps and 100 reciprocal games per model.
- [ ] Reserve ten seeds per conversion profile beginning at `9,000,000`, with a
  disjoint 1,000-seed namespace per profile. Do not consume them during
  implementation.
- [ ] Reserve the `10,000,000` series as an untouched confirmation bank. It is
  consumed only after a condition passes the panel gate.
- [ ] Write all profile definitions, weights, metrics, stop/go gates, and paired
  comparison rules before training begins.

Expected result: rerunning the panel cannot silently substitute a checkpoint,
scenario, seed bank, or profile definition.

---

## Task 2: Add the profiled-start domain model

**Files:**

- Modify: `engine/HexWars.Engine/Rl/TacticalV2Config.cs`
- Test: `engine/HexWars.Engine.Tests/TacticalV2ConfigTests.cs`

**Interfaces:**

- Produce `TacticalV2StartProfile` with ID, learner/opponent counts, and
  separation band.
- Produce `TacticalV2StartDistribution` with integer basis-point weights and a
  deterministic `Select(seed)` method.
- Preserve `SampleStartingArmy(int seed)` for the legacy standard path.

- [ ] Write failing tests for unique non-empty profile IDs, counts between one
  and slot capacity, known separation values, exact 10,000 weight sum, allowed
  zero weights, and rejection of weights for undeclared profiles.
- [ ] Prove selection is deterministic and insensitive to dictionary iteration
  order by sorting on profile ID before cumulative selection.
- [ ] Keep legacy validation unchanged for `symmetric-random-v1`, including
  `StartingUnitCount == MaxControllableUnits`.
- [ ] Under `profiled-seeded-v1`, require `StartingUnitCount = 3` and
  `MaxControllableUnits = 3` for this contract, plus the exact declared catalog.
- [ ] Reject a catalog that contains only mechanically suspect conversion
  profiles and no standard profile.
- [ ] Run focused tests.

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter TacticalV2ConfigTests
```

Expected result: profile selection is pure, seeded, validated, and independent
of model geometry.

---

## Task 3: Construct deterministic, feasible conversion starts

**Files:**

- Modify: `engine/HexWars.Engine/Rl/TacticalV2Layout.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2UnitRegistry.cs` only if empty-slot
  initialization is not already correct
- Test: `engine/HexWars.Engine.Tests/TacticalV2CodingTests.cs`

**Interfaces:**

```csharp
TacticalV2Start NewGame(int seed);
TacticalV2Start NewGame(
    int seed,
    TacticalV2StartProfile profile,
    PlayerId learnerSeat);
```

- [ ] Snapshot representative legacy `NewGame(seed)` start states before editing:
  board, player state, units, registries, IDs, and replay start serialization.
- [ ] Keep `NewGame(seed)` as the byte-identical standard constructor.
- [ ] Build conversion starts from sorted passable cell candidates using a
  domain-separated RNG. Do not depend on hash-set/dictionary iteration order.
- [ ] Define the exact 2-3, 4-6, and 7-or-more distance bands in engine code and
  test boundary distances using the closest opposing-unit pair.
- [ ] Place the learner and opponent counts relative to `learnerSeat`; reciprocal
  construction must swap ownership without changing board generation.
- [ ] Sample conversion template compositions independently with replacement,
  using separate deterministic RNG domains for learner templates, opponent
  templates, and placement. Retain ordered IDs in `TacticalV2Start` diagnostics.
- [ ] Initialize both registries at capacity three while occupying only the live
  slots. Empty slots must observe as zero and remain eligible for a later legal
  deployment.
- [ ] Use authoritative engine calculations to require a plausible route to
  positive damage from at least one learner unit. Immobile, mutually harmless,
  disconnected, or immediately terminal starts are rejected during deterministic
  construction rather than admitted as training draws.
- [ ] Reject only mechanically impossible states. Do not filter merely difficult
  matchups. Record construction attempts and rejection reasons so feasibility
  filtering cannot silently turn the curriculum into easy-template selection.
- [ ] Bound deterministic rejection sampling. Exhaustion throws an error naming
  profile and seed; it never falls back to standard 3-v-3.
- [ ] Test all default template matchups and adversarial catalogs containing
  immobile or zero-damage units.
- [ ] Test stable unit IDs, slot release/redeployment, action-mask legality,
  observation length, action count, same-seed equality, different-seed variation,
  and both learner seats for every profile.
- [ ] Verify legacy replay-start bytes for the frozen standard seeds are unchanged.

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TacticalV2CodingTests|ReplayTests"
```

Expected result: conversion starts are reproducible and mechanically meaningful,
while the old constructor remains untouched in behavior.

---

## Task 4: Route profiles through training and duel environments

**Files:**

- Modify: `engine/HexWars.Engine/Rl/TacticalV2Env.cs`
- Modify: `engine/HexWars.Engine/Rl/TacticalV2DuelEnv.cs`
- Test: `engine/HexWars.Engine.Tests/TacticalV2EnvTests.cs`
- Test: `engine/HexWars.Engine.Tests/TacticalV2DuelEnvTests.cs`

- [ ] In the single-agent environment, select a training profile from the
  configured distribution during reset and construct it relative to `_seat`.
- [ ] Expose the selected profile ID as reset/episode diagnostics, never as an
  observation feature or reward component.
- [ ] Add an explicit duel reset overload accepting a declared profile ID and
  reference seat. Do not infer the reference seat from player 0 or reward
  perspective.
- [ ] Keep the existing duel reset overload standard and byte compatible.
- [ ] Ensure `_armyValue`, previous advantage, trace reset state, replay start,
  and both registries are initialized from the actual profiled state.
- [ ] Test alternating learner seats over multiple resets and prove the learner
  receives the declared live-unit advantage in both seats.
- [ ] Test that reward remains identical in definition: win +1, loss -1, zero
  draw terminal credit, value delta, and per-command time penalty as currently
  configured.
- [ ] Test forced undeclared profile rejection and standard default behavior.

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TacticalV2EnvTests|TacticalV2DuelEnvTests"
```

Expected result: the same profile semantics drive training, deterministic duel
evaluation, tracing, and replay.

---

## Task 5: Version scenario, wire, and contract provenance

**Files:**

- Modify: `engine/HexWars.Engine/Rl/TrainingScenario.cs`
- Modify: `engine/HexWars.Engine/Rl/MlContract.cs`
- Modify: `engine/HexWars.GymServer/ScenarioJson.cs`
- Modify: `python/ml_lab/scenarios.py`
- Modify: `python/config/training-game-templates.json`
- Modify: Unity editor scenario DTOs and tests
- Test: `engine/HexWars.Engine.Tests/TrainingScenarioTests.cs`
- Test: `engine/HexWars.Engine.Tests/MlContractTests.cs`
- Test: `python/tests/test_scenarios.py`

- [ ] Add strict JSON fields for profile catalog and distribution. Reject missing,
  duplicate, unknown, misspelled, non-integer, out-of-range, and non-summing
  values at C#, GymServer wire, Python, and Unity boundaries.
- [ ] Add checked-in `annihilation-profiled-standard.json` and
  `annihilation-profiled-mixed.json` experiment scenarios with identical board,
  rules, reward, template catalog, slot capacity, and profile catalog.
- [ ] Include placement version and ordered profile catalog in tactical-v2
  contract semantics and canonical contract JSON.
- [ ] Exclude only distribution weights from contract identity. Document why:
  they change reset sampling/provenance, not observation, legal-action semantics,
  reward, or transition rules for any declared profile.
- [ ] Prove both scenarios have identical contract and encoding hashes but
  different scenario SHA-256 hashes.
- [ ] Prove old standard scenarios retain their old contract and encoding hashes.
- [ ] Prove changing any profile count, separation definition, slot capacity,
  template, action region, or placement version changes the appropriate identity.
- [ ] Ensure resume reads the source scenario and cannot override distribution.
- [ ] Make Unity round-trip the new fields without adding a full authoring UI.

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --filter "TrainingScenarioTests|MlContractTests"
.\python\winenv\Scripts\python.exe -m pytest -q python/tests/test_scenarios.py python/tests/test_run_contract.py
```

Expected result: comparison models share a legitimate inference contract while
their different training distributions remain auditable.

---

## Task 6: Extend reset and evaluation protocols

**Files:**

- Modify: `engine/HexWars.GymServer/Program.cs`
- Modify: `python/hexwars_gym/env.py`
- Modify: `python/ml_lab/envs.py`
- Modify: `python/ml_lab/evaluation.py`
- Modify: `python/ml_lab/cli.py`
- Modify: `python/ml_lab/protocol.py`
- Modify: `python/ml_lab/contracts.py`
- Test: Python Gym, protocol, training, CLI, and evaluation tests

**Protocol changes:**

- Training `reset` returns `start_profile` selected by engine authority.
- Tactical-v2 `duel_reset` accepts optional `start_profile` and explicit
  `reference_seat`.
- Evaluation CLI accepts `--start-profile PROFILE_ID`; reciprocal evaluation sets
  `reference_seat` to the candidate seat for each game.

- [ ] Validate reset response profile IDs against the declared handshake catalog.
- [ ] Preserve legacy reset payload acceptance when the old placement policy is
  active.
- [ ] Add profile ID to episode info and a dedicated tactical-v2 episode sidecar
  rather than silently changing legacy monitor CSV meaning.
- [ ] Record profile counts by worker, learner seat, and outcome. At run completion,
  compare observed sampling frequencies with the configured distribution and
  warn on impossible/absent nonzero profiles.
- [ ] Record forced profile and reference seat in every evaluation match row and
  top-level schedule metadata.
- [ ] Refuse `--start-profile` for an environment or contract that does not
  declare it.
- [ ] Keep controller compatibility exact. Because both panel scenarios share
  one new contract, no unsafe encoding-only override is required.
- [ ] Preserve replay and trace capture for profiled starts.

```powershell
.\python\winenv\Scripts\python.exe -m pytest -q python/tests/test_gym_client.py python/tests/test_protocol.py python/tests/test_training.py python/tests/test_evaluation.py python/tests/test_cli.py
```

Expected result: training distributions and forced evaluation profiles are
observable, deterministic, and impossible to confuse with one another.

---

## Task 7: Prove the standard path and end-to-end smoke

- [ ] Run the complete engine and Python suites.
- [ ] Run Unity compile and relevant EditMode tests.
- [ ] Run the standard profiled scenario twice with the same seed and compare
  reset observation, legal mask, selected profile, episode result, trace, and
  replay bytes.
- [ ] Run every conversion profile for both candidate seats and verify declared
  counts and distance bands from authoritative traces.
- [ ] Run a 4,000-step target-KL CUDA smoke for each scenario under fresh run
  names.
- [ ] Inspect saved PPO objects and confirm learning rate 3e-4, ten nominal epochs,
  target KL 0.02, three-slot geometry, and matching contract hashes.
- [ ] Confirm TensorBoard, local progress, per-profile episode sidecar, checkpoint,
  run manifest, and scenario hash all exist.
- [ ] Evaluate smoke checkpoints only on the development seed bank. Do not touch
  locked panel banks.

```powershell
dotnet test .\engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj
.\python\winenv\Scripts\python.exe -m pytest -q python/tests
```

Stop if standard-path replay or hash compatibility fails. Do not begin training
while a profile can produce an invalid start, wrong-seat advantage, missing
provenance, or mismatched contract.

---

## Task 8: Run the six-model PPO curriculum panel

**Conditions:**

1. Profiled standard control, 100% standard starts.
2. Mixed conversion curriculum, 40% standard and 60% near/far conversion starts.

**Locked common settings:**

- MaskablePPO / HexCNN
- learning rate 3e-4
- ten nominal epochs
- target KL 0.02
- Random opponent
- alternating learner seat
- seeds 101, 113, 127
- 51,200 steps
- four workers
- local and TensorBoard tracking
- no draw terminal credit
- no closing reward
- existing value-delta shaping and step penalty

- [ ] Write a restart-safe runner that validates exact scenarios, hashes,
  algorithm options, run state, checkpoints, profile frequencies, and evaluation
  schedules before reuse.
- [ ] Train all six models to exactly 51,200 rollout-aligned steps.
- [ ] Evaluate each model on 50 standard maps in both seats: 100 games/model.
- [ ] Evaluate each model on ten maps for each of nine conversion profiles in
  both seats: 180 games/model.
- [ ] Run Greedy and Random controls on the same suites.
- [ ] Capture replay-backed evidence for every conversion draw plus stratified
  standard wins and Greedy controls.
- [ ] Report per-training-seed, profile, separation, unit-count, and seat outcomes.
- [ ] Pair comparisons by training seed, map seed, profile, and candidate seat.
- [ ] Report cycling, action waste, failed conversion, rounds to win, and final
  health-adjusted advantage.
- [ ] Report environment steps, completed episodes by profile, wall-clock time,
  PPO KL/clip/update counts, and evaluation inference time.
- [ ] Label the report as a diagnostic curriculum decision, not the tactical-v3
  official tournament or a production-AI promotion.

### Curriculum pass gate

The mixed condition passes only if all are true:

- aggregate conversion-suite win rate improves by at least 15 percentage points;
- total conversion wins improve for each of the three training seeds;
- cycling-draw incidence falls by at least 25% relative;
- held-out medium-separation conversion improves, not only trained near/far;
- standard 3-v-3 win rate is no more than five percentage points below control;
- no individual seed loses more than 15 percentage points on standard 3-v-3;
- loss rate does not rise by more than five percentage points; and
- both candidate seats show the effect.

If the gate passes, continue the mixed models to 100,352 and consume the reserved
10,000,000-series confirmation bank. If it fails, do not tune weights on the
locked evaluation results. Diagnose the failed clause and proceed to the
algorithm decision work package.

---

## Task 9: Add a bounded search baseline before another learned algorithm

**Purpose:** Test whether explicit lookahead and a persistent pursuit objective
can convert states that reactive PPO cycles in.

**Files:**

- Create: `engine/HexWars.Engine/BoundedSearchAgent.cs`
- Create: `engine/HexWars.Engine.Tests/BoundedSearchAgentTests.cs`
- Modify: `engine/HexWars.GymServer/Program.cs` to expose the named scripted
  controller only after its tests pass
- Modify: Python controller normalization/evaluation tests for the new scripted
  identity

**Entry gate:** Run this task if the curriculum fails its pass gate, passes but
remains materially below Greedy on conversion, or leaves cycling as the dominant
draw class. If curriculum PPO reaches the confirmation gate without those
failures, record this task as deferred rather than expanding scope automatically.

- [ ] Implement an evaluation-only bounded planner over authoritative legal
  commands and exact engine transitions. It must not alter game rules or policy
  observations.
- [ ] Start with conversion profiles where branching is bounded. Use a fixed
  simulation/expansion budget per decision and record actual expansions and wall
  time.
- [ ] Use terminal outcome as primary value. Any nonterminal heuristic is bounded,
  health-sensitive, documented, and separately ablated.
- [ ] Compare planner, Greedy, and PPO from identical profile/map/seat starts.
- [ ] Retain planner replays and top-branch diagnostics for every disagreement in
  which the planner wins and PPO draws.

Interpretation:

- Planner succeeds where PPO cycles: prioritize tactical-v3 candidate/afterstate
  scoring and persistent intention work.
- Planner also fails: inspect mechanical feasibility, round cap, and start/profile
  construction before blaming model-free learning.
- Planner offers no gain over Greedy: do not build a large AlphaZero-style system
  without stronger evidence.

This is a diagnostic ceiling, not a training condition, so do not rank it by PPO
timesteps.

---

## Task 10: Test imitation bootstrap as the first learned alternative

**Purpose:** Distinguish failure to discover pursuit behavior from failure to
optimize or retain it.

**Files:**

- Create: `python/ml_lab/imitation.py`
- Create: `python/collect_annihilation_demonstrations.py`
- Create: `python/tests/test_imitation.py`
- Modify: `python/ml_lab/algorithms.py` only through an explicit, tested
  policy-weight initialization boundary

**Entry gate:** Run this only when the bounded planner or demonstrator materially
outperforms from-scratch PPO on identical conversion states. Without that positive
control, behavioral cloning has no demonstrated pursuit knowledge worth copying.

- [ ] Generate a training-only demonstration dataset from Greedy and the bounded
  planner on the development/training seed namespace. Never include locked
  evaluation seeds.
- [ ] Store observation, legal mask, selected action, profile, seed, seat,
  controller identity, contract hash, and source replay for every row.
- [ ] Reject any action that does not round-trip through `TacticalV2Coding` and
  the authoritative engine.
- [ ] Add masked cross-entropy behavioral cloning for the same HexCNN actor used
  by PPO. Report held-out action accuracy but do not treat it as game competence.
- [ ] Initialize a fresh MaskablePPO model from cloned policy weights with a fresh
  value head/optimizer state, then train under the exact mixed curriculum.
- [ ] Compare three seeds against from-scratch mixed PPO on the same evaluation
  suites and compute budget.

Interpretation:

- BC-to-PPO improves conversion across seeds: sparse discovery/exploration is a
  major bottleneck.
- BC accuracy is high but game conversion is unchanged: imitation learned local
  action resemblance without strategic persistence.
- BC harms standard play: reduce demonstration dominance or move the relevant
  behavior into tactical-v3 auxiliary/intention training rather than rewarding
  it directly.

---

## Task 11: Admit DQN only after parity hardening

The current `masked_dqn` is not an acceptable algorithm comparison. It is marked
experimental, uses an MLP rather than the PPO HexCNN, exposes no algorithm
options, and cannot resume because replay-buffer sidecars are not persisted.

Before a DQN result is interpreted:

- [ ] Use the same spatial feature extractor or document and separately ablate
  the representation difference.
- [ ] Ensure legal masking applies to exploration, exploitation, bootstrap target
  selection, and evaluation.
- [ ] Persist replay buffer, exploration schedule, optimizer, RNG, and target
  network state with checkpoint provenance and safe resume.
- [ ] Add durable learning-rate, buffer, batch, learning-start, train-frequency,
  target-update, and exploration options.
- [ ] Add tests proving illegal actions cannot affect current or target Q maxima.
- [ ] Decide explicitly whether the condition is vanilla DQN, Double DQN, or a
  distributional variant; do not label a changing bundle simply "DQN."
- [ ] Compare by environment steps and wall-clock/updates, with three seeds and
  the same curriculum/evaluation banks.

Run DQN only if replay-based sample reuse remains a live hypothesis after the
curriculum, planner, and imitation results. Otherwise this work is deferred.

---

## Task 12: Algorithm and architecture decision record

Create:

- `python/panels/annihilation-conversion-curriculum-v1/REPORT.md`
- `docs/superpowers/decisions/2026-XX-XX-annihilation-learning-path.md`

The decision record answers:

1. Is annihilation conversion trainable under frequent valid finishing states?
2. Does the learned skill transfer to ordinary 3-v-3 starts and held-out medium
   separation?
3. Is the main remaining limitation exploration, persistent intention, reactive
   action selection, representation, or on-policy sample reuse?
4. Does PPO remain the tactical-v3 baseline optimizer?
5. Which evidence, if any, authorizes Greedy training or self-play?

### Final branches

**Curriculum PPO passes:** retain target-KL PPO as the baseline, confirm at
100,352 on the sealed bank, then introduce Greedy gradually while preserving
Random and conversion evaluation.

**Curriculum fails, planner succeeds:** stop tuning tactical-v2 reward/optimizer.
Advance tactical-v3 legal-candidate/afterstate scoring and persistent intention
stages. PPO may remain the optimizer around the new representation.

**Imitation succeeds:** retain imitation only as an initialization/data strategy,
not a reward substitute. Expand diverse demonstrators before Greedy opponent
training.

**Parity DQN succeeds:** run a confirmation panel with additional seeds and equal
compute accounting before changing the supported algorithm.

**Nothing succeeds:** treat the observation/action contract or task feasibility as
the blocker. Do not spend more compute on the current policy family.

## Completion definition

This plan is complete only when:

- profiled starts are deterministic, valid, replayable, and learner-relative;
- standard tactical-v2 behavior remains compatible;
- the standard and mixed scenarios share one new model contract but have distinct
  immutable provenance;
- six PPO models and all locked evaluations are complete or a documented hard
  gate stops them;
- replay evidence explains remaining draws;
- algorithm alternatives are admitted only after their parity prerequisites;
- a decision record selects the next learning path using wins and causal evidence,
  not TensorBoard aesthetics.
