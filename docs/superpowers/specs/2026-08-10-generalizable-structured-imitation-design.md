# Generalizable Structured Imitation for HexWars

**Status:** Approved design; **Date:** 2026-08-10
**Game-mode family:** Annihilation tactical play

## Purpose

HexWars needs a new imitation-learning path that can first learn reliably in a controlled setting and then
generalize across board sizes, army compositions, unit statistics, and eventually unit-design capabilities.
The current behavioral-cloning and selective-DAgger work is useful evidence, but it trains the fixed
`tactical-v2` HexCNN actor. That actor cannot establish the requested generality because its observation and
action dimensions and their meanings depend directly on one board, roster ordering, template count, and unit
slot count.

This design defines a new structured policy, action contract, imitation pipeline, curriculum, reward boundary,
and evaluation program. It implements the first practical portion of the broader
`2026-07-28-tactical-v3-general-rl-design.md` architecture. It does not modify or relabel the existing
selective-DAgger experiment, and it does not make Task 11 sealed evidence a prerequisite for explicitly
experimental training.

The central decision is:

> Build the general representation and variable-candidate policy from the beginning, but begin training on a
> stable 13x9 curriculum with the five existing unit templates. Add one controlled source of variation at a
> time after fixed-curriculum learning and regression gates pass.

This separates architectural generality from curriculum difficulty. The first checkpoint need not immediately
master every configuration, but it must not encode a board size or roster position into permanent network
geometry.

## Problem Statement

The observed tactical policy often wins material exchanges yet fails to convert the last few units into an
annihilation win. Recent PPO continuation also showed promising early performance followed by collapse. A new
imitation path should therefore address both distribution shift and representation quality without using PPO
to overwrite the learned actor.

DAgger alone is insufficient when the learner cannot observe the facts needed to reproduce the teacher. The
current `tactical-v2` contract has the following relevant limitations:

- `TacticalV2Layout` makes observation and action lengths functions of cell, template, and unit-slot counts.
- `TacticalV2Coding` represents unit occupancy through template-indexed HP planes but move and attack actions
  through stable unit slots.
- Unit statistics are absent from the observation. Template channel position stands in for mechanical meaning.
- The observation omits explicit per-unit moved, attacked, and movement-spent state.
- The flat action vector is `EndTurn` plus unit-slot-by-cell move and attack regions and a
  template-by-cell deployment region.
- Invalid action decoding can fall back to `EndTurn`, concealing representation or resolver failures as
  apparent policy passivity.
- `HexCNN` flattens the complete convolutional board result into a fixed-size linear layer.
- The selective-DAgger panel pins one 1,292-value observation, one 1,288-action encoding, one scenario, and one
  catalog. Its protocol explicitly makes no board-size, roster, mechanic, or unit-design generality claim.

The existing `adaptive-v1` environment demonstrates useful hierarchical command and unit-design concepts, but
it is not the target neural contract. It fixes 24 unit slots, nine templates, nine known statistics, eleven
stat values, and a board-cell action region whose size still varies with the board.

## Goals

- One checkpoint for the annihilation game-mode family across supported board sizes.
- Variable numbers of cells, units, templates, capability allocations, relations, memory records, and legal
  commands.
- Mechanical interpretation of unfamiliar unit stat allocations rather than roster-name or template-position
  memorization.
- Complete legal-command selection through a variable candidate set.
- Safe candidate successor information so the model can compare threat, opportunity, and conversion potential.
- Full-game teacher imitation followed by learner-state DAgger correction.
- Cumulative curricula that add variation without discarding previously mastered distributions.
- Annihilation win rate and conversion behavior, not supervised loss alone, as checkpoint-selection criteria.
- A run and controller format loadable by ML Lab and observable in Unity.
- A reward interface suitable for later RL without making reward optimization part of the initial imitation
  milestone.
- Explicit interfaces and conformance tests at every engine, transport, model, and publication boundary.

## Non-goals for the First Milestone

- PPO continuation or any other policy-gradient fine-tuning.
- Fog-of-war training, although empty memory and visibility tables are part of the schema.
- Autoregressive unit-design generation.
- Armor penetration, medicine, aircraft, or other new mechanics.
- Self-play or a population curriculum.
- Zero-shot competence on genuinely unknown mechanical semantics.
- Replacement or compatibility conversion of tactical-v1, tactical-v2, adaptive-v1, or their checkpoints.
- Production-sealed research evidence. Experimental outputs must be labeled unsealed and may later pass through
  an independently designed sealing workflow.

## Design Principles

### General interface, narrow initial distribution

The model contract is ragged and semantic from its first version. Stage-one data nevertheless uses the current
13x9 board, five templates, standard 3v3 profile, and existing conversion profiles. This makes optimization and
teacher failures diagnosable without hardcoding those values into the network.

### Engine facts, learned strategy

The engine owns legality, transition mechanics, pathfinding, targeting, visibility, costs, and damage. It may
provide these as factual token, relation, or candidate features. It does not assign a fixed strategic ranking
such as "nearest enemy is best" or "artillery is always the greatest threat."

### Complete commands, not permanent action neurons

The policy scores the legal commands available in the current state. Board size changes the number of
candidates, not the shape of a permanent action head.

### Outcome is distinct from diagnostics

A draw is a non-win. Useful actions and mechanically informative transitions inside a drawn game remain valid
imitation and auxiliary-learning examples. Diagnostic progress never relabels the episode as successful.

### Generality is an evaluation result

Training on variation is not evidence of generalization. Each curriculum stage reserves configurations and
combinations that are absent from training and reports them separately.

## High-Level Architecture

```text
authoritative GameState + observing seat + observation memory
    -> ISeatObservationSource
    -> structured seat observation

structured seat observation + authoritative legal-command enumeration
    -> ILegalCandidateSource
    -> variable complete-command candidates

candidate + seat-safe transition projection
    -> ICandidateProjector
    -> mechanical facts and projected successor delta

structured tokens + relations + candidates
    -> structured PyTorch policy
    -> masked distribution over candidates
    -> selected candidate identity

selected identity + unchanged state
    -> IActionResolver
    -> exact authoritative Command
```

The model is a new controller type. It is not stored as a disguised MaskablePPO checkpoint and does not depend
on SB3's fixed `Discrete(n)` action head.

## Structured Observation Contract

The logical observation contains variable-row typed tables:

```text
cells                   [Nc, Fc]
units                   [Nu, Fu]
templates               [Nt, Ft]
capability_definitions  [Nk, Fk]
capability_allocations  [Na, Fa]
rules                   [Nr, Fr]
memory                  [Nm, Fm]
relations               [Ne, Fe]
```

The feature width of each table is schema-versioned. Row counts vary by state. Runtime batching may pad to the
largest row count in a minibatch and must carry an explicit validity mask. Padding has no game meaning.

A separately versioned capacity profile declares maximum row counts accepted by one runtime deployment. A
match exceeding the capacity profile fails before reset. No object, relationship, or candidate is silently
truncated.

### Cell tokens

There is one token for every valid hex. Initial features include:

- axial coordinates in a seat-relative frame;
- terrain and elevation;
- deployment-zone, control, and objective status where applicable;
- current visibility and previously observed status;
- boundary context;
- references to occupying visible entities.

Cells form explicit six-neighbor edges. Coordinates are data, not learned absolute cell IDs. Translation and
seat-reflection tests enforce the intended symmetries.

### Unit tokens

Each friendly or seat-observable enemy unit receives one token with:

- owner relationship;
- current and maximum health;
- cell and elevation references;
- moved, attacked, and movement-spent state;
- point cost and deploy-equivalent cost context;
- visibility or last-observed status;
- links to its capability allocations.

Runtime entity IDs may link records within one episode but never receive learned type embeddings. Unit row order
has no strategic meaning.

### Template tokens

Each seat-available template exposes:

- capability allocations;
- point and deployment costs;
- fixed or custom status;
- deployment availability and mechanical restrictions.

Template names and catalog indices are provenance or presentation data. They are not policy features.

### Capability definition and allocation tokens

The first contract represents all existing nine unit statistics through capability definitions and allocation
records rather than permanent template-channel meanings. A definition includes:

- stable schema identity;
- passive, modifier, or active classification;
- cost rule;
- source and target domains;
- target type and timing descriptors;
- known effect family;
- action, cooldown, or usage constraints where applicable;
- typed interaction relationships such as `opposes`, `reduces`, `restores`, `enhances`, `requires`, and
  `enables_action`.

An allocation links one unit or template to a definition and supplies its purchased level and effective value.
The initial nine definitions are Health, Damage, Defense, Movement, VerticalMovement, Range, RangeArc, Vision,
and VisionArc.

The relationship graph is an inductive hint, not an engine formula or strategic rule. Adding a future mechanic
adds definition, allocation, relation, and action records. An old model may represent an unfamiliar definition
generically; useful behavior still requires relevant experience and fine-tuning.

### Rule and economy tokens

Rules that change legality, transitions, or objectives are observable. Initial records include:

- game mode and win condition;
- round and round cap;
- turn policy and action budget;
- damage and elevation modifiers;
- starting and current points;
- bounty and deployment-cost rules;
- board and terrain rules;
- fog state;
- design budget and fees when later enabled.

Otherwise identical positions under different transition rules must not alias.

### Memory tokens

The initial no-fog curriculum emits an empty memory table. The schema supports later exact last-observed records:
last-seen cell and round, observation age, last-known health and capabilities, currently-visible flag, and a
seat-safe conservative reachable envelope. Hidden current truth never enters memory.

## Spatial and Relational Processing

Each token type has a small type-specific encoder that projects raw features into a shared model width. The
result remains ordinary matrix computation. For example:

```text
cells         [117, Fc] -> CellEncoder       -> [117, D]
units           [6, Fu] -> UnitEncoder       -> [  6, D]
capabilities   [54, Fa] -> AllocationEncoder -> [ 54, D]
```

On another board the cell matrix may be `[384, D]`; the learned widths and weights are unchanged.

Cell representations use local hex-neighbor message passing. Units, templates, capabilities, and rules use
typed relational attention. Attention may receive relation features including:

- relative axial displacement and hex direction;
- hex, path, and movement-turn distance;
- elevation difference;
- visibility and line-of-fire state;
- movement and range margins;
- deterministic public damage and lethality;
- legal reachability or targetability;
- candidate participation.

Distance is evidence, not importance. Candidate-conditioned queries allow a distant long-range attacker to be
a major defensive threat while nearby weak units remain attractive attack opportunities.

Permutation tests must prove that shuffling token rows and consistently updating references only permutes the
corresponding representations and cannot change the selected physical command.

## Variable Legal-Candidate Action Contract

For a state `s`, the engine enumerates complete legal commands `C(s)`. Initial candidates cover:

- `EndTurn`;
- `MoveUnit(unit, destination)`;
- `AttackUnit(attacker, target)`;
- `DeployUnit(template, cell)`.

Every candidate contains a stable within-decision identity, command type, relevant token references, and
mechanical features. The candidate set must equal the authoritative engine legal-command set exactly after
applying the current environment's supported-command filter.

The policy applies one shared scoring function to every candidate and normalizes over the variable set:

```text
score_i = CandidateScorer(state_context, candidate_i, actor_i, target_i, projected_delta_i)
pi(a_i | s) = masked_softmax(score_i)
```

The selected identity resolves to the exact command represented. Out-of-range, stale, ambiguous, or
non-round-tripping identities are contract errors. They never fall back to `EndTurn`.

## Candidate Projection and State Comparison

`ICandidateProjector` supplies the policy with safe factual consequences of one candidate. Depending on command
type, the projection may include:

- actor and target state deltas;
- destination and movement expenditure;
- deterministic damage, remaining health, and kill status;
- points spent or earned;
- action-budget consumption;
- changed reachability, targeting, and threat relations;
- sparse successor token updates.

The learner therefore compares actions using current state and projected successor information rather than
memorizing a flat action address.

In the initial no-fog curriculum, one-command projection may be exact. Under later fog, projection uses only
the seat-visible model of the world. It cannot simulate through hidden occupancy or otherwise reveal hidden
truth. The authoritative engine remains the final transition validator.

Full successor encoding can be expensive. The implementation may use sparse deltas, batched successor
encoding, or a measured candidate shortlist, but any approximation must be explicit and evaluated. It may not
reintroduce fixed board dimensions or permanently exclude unusual long-range candidates.

## Teacher and Dataset Contract

### Teacher

The initial teacher is deterministic bounded search over the same authoritative legal commands. Each label
records:

- selected command identity;
- evaluated state identity;
- search depth and expansion budget;
- actual expansions;
- heuristic identity;
- candidate value or confidence evidence when the teacher implementation exposes it;
- engine, scenario, observation, capability, and candidate schema identities.

The teacher may use full authoritative state only while fog is disabled. Future fog training requires a
seat-information-consistent teacher or an explicitly privileged distillation experiment whose leakage boundary
is separately evaluated.

The existing `material-plus-pursuit-v1` search heuristic is a starting teacher, not an unquestioned strategic
oracle. It uses health-adjusted material but also a manually weighted nearest-target pursuit term. Teacher
performance must be measured directly, and low-confidence or inconsistent labels remain separately visible.

### Structured example

One example contains:

```text
structured seat observation
relation graph
complete legal candidates
projected candidate deltas
teacher-selected candidate identity
teacher evidence
episode/profile/configuration identity
terminal outcome and trajectory index
remaining victory horizon when defined
```

The canonical stored form is semantic and schema-versioned. It does not store tactical-v2 flat observations or
permanent action indices as authority.

### Corpus behavior

Stage one begins with reciprocal-seat games in which bounded search supplies the learner-seat decisions against
Random on standard 3v3 and every existing conversion profile. The corpus retains every valid teacher decision,
not only disagreements or late-game states. Search-versus-search games may be added as a separately identified
source but cannot silently replace the primary search-versus-Random distribution.

DAgger then collects learner-controlled games, asks the teacher to label states visited by the learner, and
appends those rows to a cumulative immutable corpus. Earlier curriculum strata remain available to every later
training stage. Sampling is stratified by source, profile, action type, phase, outcome, and conversion status so
large ordinary-game strata cannot erase rare endgame corrections.

No teacher search or scripted correction runs at policy inference time.

## Model and Training Objectives

The production policy is a custom PyTorch module with:

- typed token encoders;
- spatial/relational shared encoder;
- variable legal-candidate scorer;
- terminal outcome head;
- win-within-horizon heads;
- conditional remaining-turns-to-victory head;
- optional factual mechanics heads.

The primary actor loss is masked candidate cross-entropy:

```text
L_policy = -log pi(a_teacher | observation, legal_candidates)
```

Auxiliary tasks may include deterministic damage, lethality, reachability, targetability, next-visible-state,
terminal outcome, win within versioned response horizons, and remaining turns to victory conditional on a win.
The action loss has coefficient 1.0. The sum of configured auxiliary coefficients must not exceed 0.5, and all
coefficients are recorded in the run manifest. This keeps action imitation primary while allowing shared
mechanics learning.

Conditional time-to-victory loss is masked for losses and terminal draws. Time-limit truncations are censored,
not assigned arbitrary long victory labels.

Training uses deterministic seeds, gradient clipping, finite-loss checks, immutable validation partitions, and
early stopping. The latest epoch or DAgger iteration is never automatically promoted. Historical candidates
remain available for closed-loop comparison.

## Reward Contract

Initial structured imitation does not optimize environment reward. It optimizes teacher action likelihood and
auxiliary supervised labels. This prevents reward-gradient instability from overwriting the actor and makes it
explicit that changing reward cannot repair missing observation information.

The annihilation environment still exposes a decomposed reward contract for diagnostics and later RL:

```text
terminal_outcome
known_health_adjusted_material_progress
public_resource_progress
time_pressure
total
```

Terminal outcomes are:

```text
annihilation win = +1
loss             = -1
draw             = -1
```

Any later scalar progress contribution must be health-sensitive and normalized by match resource scale. Let
`T` be the terminal outcome above and `S` be the sum of every progress and time adjustment over the episode.
The reward contract requires `-0.25 <= S <= +0.25`, including a non-positive time-pressure contribution whose
absolute episode total is at most `0.05`. Thus a win return lies in `[+0.75,+1.25]` and every non-win lies in
`[-1.25,-0.75]`. If progress is distributed across transitions, the environment records the deterministic
remaining shaping budget as reward-contract state so the episode bound cannot be exceeded. Repeatedly holding
the same advantage earns nothing. Hidden state may not influence any component.

The current full-point-cost value for an injured living unit is replaced in progress diagnostics by:

```text
effective_value(unit) = deploy_equivalent_cost * current_hp / maximum_hp
```

Nearest-enemy closing reward is excluded. It prescribes a tactic and can reward suicidal or hidden-state pursuit.
Time pressure is based on completed turns or rounds and normalized by the round cap, not charged once per useful
command.

Detailed draw classes, threat estimates, cycles, action waste, and failed conversion remain diagnostics and
curriculum selectors rather than proliferating hand-authored reward terms.

## Staged Curriculum

### Stage 0: representation and learning smoke

- General structured schema and variable candidate policy are active.
- Use tiny deterministic fixtures and current 13x9 mechanics.
- Prove parser, batching, forward pass, loss, gradient, save/load, controller inference, and Unity publication.
- Overfit a tiny corpus before expensive collection.

### Stage 1: fixed-config tactical imitation

- Current 13x9 board.
- Existing five templates and nine capabilities.
- No fog.
- Standard 3v3 plus every existing conversion profile.
- Random as the primary closed-loop opponent, with selected Greedy and bounded-search comparisons.
- Full teacher corpus, behavioral cloning, then learner-state DAgger.

The practical gate is consistent annihilation wins against Random without unacceptable conversion-profile
failure.

### Stage 2: unfamiliar stat allocations

- Keep board and rules fixed.
- Procedurally vary allocations of the existing nine capabilities within legal budgets.
- Reserve complete stat combinations and composition combinations from training.
- Require retention of stage-one performance.

### Stage 3: variable board sizes

- Train on multiple sizes in mixed batches.
- Reserve at least one supported intermediate or extrapolative size entirely from training.
- Use the same checkpoint and capacity profile across sizes.
- Report each size separately and retain stages one and two.

### Stage 4: army and economy variation

- Vary unit count, template count, starting points, design/deployment budgets, terrain, and selected rule values.
- Reserve cross-factor combinations rather than testing only new individual values.

### Stage 5: atomic unit design

- Add a design/replace command candidate and an autoregressive capability-allocation decoder.
- Submit one complete, mechanically valid design as one authoritative command.
- Charge the configured design fee once per complete design.
- Reuse the same capability and context encoder used to recognize units.

### Stage 6: capability extensions and opponent population

- Add new capability descriptors through targeted curricula.
- Add historical policies, self-play, design specialists, hunters, evaders, and exploiters only after the
  preceding stages are stable.

Every curriculum stage uses cumulative data and a retention gate. Promotion requires new-stratum improvement
without exceeding configured regressions on previously accepted strata.

## Evaluation and Checkpoint Selection

### Stage-one banks

Stage one uses frozen, reciprocal-seat development, validation, and test seed banks covering:

- standard 3v3 versus Random;
- all existing conversion profiles versus Random;
- selected standard and conversion games versus Greedy and bounded search;
- held-out map seeds absent from collection and training.

Primary metrics are:

- W/L/D with confidence intervals;
- macro and seat-specific win rate;
- failed-conversion rate after decisive known advantage;
- conversion time;
- cycling and repeated-position incidence;
- premature `EndTurn` with productive legal candidates remaining;
- action-type distribution;
- teacher agreement, negative log-likelihood, and calibration;
- outcome and horizon-head calibration.

The first practical milestone is consistent wins against Random. Aggregate win rate may not conceal severe
failure in a conversion profile.

### Generalization banks

Later stages reserve and report:

- unseen stat allocations;
- unseen combinations of familiar capabilities;
- board sizes absent from training;
- unfamiliar army, point-budget, and rule combinations;
- unfamiliar opponent unit designs.

One checkpoint must service every tested board size. No board-size-specific checkpoint selection is permitted.

Supervised metrics are necessary diagnostics but cannot promote a checkpoint without closed-loop game evidence.
The best historical candidate is retained; performance collapse in a later epoch or iteration cannot overwrite
the last accepted actor.

## ML Lab and Unity Integration

The new controller/checkpoint adapter publishes experimental runs beneath `python/runs` with:

- model weights and strict architecture configuration;
- observation, capability, relation, candidate, and capacity schema identities;
- training and validation corpus identities;
- curriculum stage and sampled configuration ranges;
- optimizer and loss configuration;
- teacher identity and DAgger iteration;
- evaluation summaries and replay paths;
- explicit `unsealed-experimental` evidence status.

ML Lab resolves the new controller type, validates its schemas against the selected scenario, launches the same
authoritative GymServer/engine path, and provides Arena playback in Unity. Official evaluation remains headless
and deterministic; Unity observation is for behavioral inspection and replay, not the source of official
metrics.

## Interface Boundaries

### Engine process

- `ISeatObservationSource`: immutable seat-filtered structured state.
- `ILegalCandidateSource`: exact supported legal commands.
- `ICandidateProjector`: seat-safe factual one-command consequences.
- `IActionResolver`: exact candidate-to-command round trip.
- `IRewardContract`: decomposed objective and diagnostics.
- `IObservationMemory`: exact public history without hidden-state acquisition.

### GymServer boundary

A versioned `tactical-v3` handshake declares table schemas, relation schemas, candidate schema, capability schema,
capacity profile, and hashes. Payloads are strict and deterministic. Unknown fields fail until introduced by a
new schema version.

### Python runtime

- immutable structured DTOs;
- schema validator;
- ragged batcher;
- policy protocol;
- teacher/corpus protocols;
- trainer and checkpoint selector;
- controller/checkpoint adapter;
- evaluation and ML Lab publication adapters.

Implementations may change behind these interfaces without changing unrelated contracts.

## Failure and Integrity Rules

- Hidden authoritative state reaching observation, memory, candidate features, projection, or reward is a hard
  failure.
- Capacity overflow fails before reset; no row or candidate is dropped.
- Every relation and candidate reference is range-checked against its table and decision identity.
- Nonterminal supported states expose at least one candidate.
- Candidate sets exactly match supported authoritative legal commands.
- Candidate selection round-trips to the exact command represented against the unchanged state.
- Invalid decoding never becomes another action.
- Unrecognized capability semantic shapes fail explicitly; known generic shapes use an explicit unknown
  representation.
- NaN, infinity, all-masked samples, invalid target labels, and malformed padding fail before optimization.
- Checkpoint and runtime schema mismatches name the expected and actual identities.
- Deterministic failures record seed, seat, scenario, controller, schema identities, and replay path.

## Verification

### Engine and contract tests

- Cell, unit, template, capability, and candidate row permutation invariance/equivariance.
- Seat reflection and coordinate translation consistency where rules are symmetric.
- Multiple board sizes in one structured batch.
- No roster-name, template-index, entity-ID, or absolute-cell-ID strategic semantics.
- Explicit moved, attacked, movement-spent, action-budget, and rule-state distinction.
- Exact candidate-set equality with authoritative supported legal commands.
- Candidate identity and command round trip without fallback.
- Projected-delta agreement with authoritative one-command transitions in no-fog states.
- Fog projection and memory non-leakage fixtures before fog is enabled.
- Capacity overflow and invalid-reference failure.
- Scenario and schema hashes agree across engine, GymServer, Python, checkpoint, and replay paths.

### Reward tests

- Win is positive; loss and draw are negative.
- No permitted progress contribution can make a non-win outrank a win.
- Partial damage changes health-adjusted progress before a kill.
- Unchanged advantage cannot be farmed.
- Hidden movement cannot change reward.
- Multiple productive commands in one turn do not accumulate command-count time penalties.
- Reward components and draw diagnostics remain independently reported.

### Model tests

- Ragged padding and masks do not affect valid-token outputs.
- Permuting rows and references preserves physical candidate probabilities.
- Candidate probabilities normalize over exactly the valid candidates.
- Backpropagation produces finite actor and encoder gradients.
- Auxiliary loss never exceeds the configured aggregate coefficient bound.
- Tiny deterministic corpus can be intentionally overfit.
- Save/load reproduces logits and deterministic selected commands.
- CPU publication inference agrees with validated training-device fixtures within declared tolerance.

### End-to-end tests

- Structured engine payload parses identically in Python.
- A selected Python candidate reconstructs the exact authoritative command and replay transition.
- Short collection, BC, DAgger, evaluation, publication, reload, and Unity Arena smoke paths complete.
- Identical seeds and deterministic checkpoints reproduce command sequences.
- Worker count and ragged batch composition do not change deterministic evaluation results.
- One checkpoint runs at least two board sizes before cross-size learning claims.

## Implementation Decomposition

The approved design is implemented through four separate plans or plan sections with independent acceptance
gates:

1. **Structured environment contract:** engine interfaces, tactical-v3 schemas, candidate projection/resolution,
   GymServer transport, and conformance tests.
2. **Policy and offline imitation:** PyTorch model, batching, supervised corpus, trainer, checkpoint, controller,
   ML Lab publication, and tiny-corpus overfit.
3. **Teacher collection and DAgger:** full-game demonstrations, learner-state correction, cumulative immutable
   corpus, and fixed-curriculum closed-loop selection.
4. **Generalization curriculum:** stat-allocation, board-size, army/economy, design, and later capability stages.

No plan may bypass a prior acceptance gate merely because downstream training infrastructure exists.

## Acceptance Criteria for the First Practical Milestone

- The new controller uses structured tokens and a variable legal-candidate distribution; it has no fixed board
  flattening or permanent slot-by-cell action head.
- Unit mechanics are represented by actual capability definitions and allocations, not roster names or channel
  positions.
- Tiny-corpus overfit, deterministic save/load, and end-to-end candidate round trips pass.
- Full-game teacher BC and at least one learner-state DAgger iteration complete on the fixed stage-one curriculum.
- The accepted checkpoint consistently beats Random on frozen reciprocal-seat standard games and reports every
  conversion profile separately.
- The accepted checkpoint does not regress behind the best historical candidate merely because training
  continued.
- The same unmodified model performs valid inference on a second board size, even before cross-size competence
  is claimed.
- The published experimental run loads in ML Lab and can be watched in Unity Arena.
- Artifacts identify themselves as unsealed experimental evidence and do not depend on Task 11 completion.

## Design Outcome

The first new imitation model learns on a controlled distribution without being architecturally confined to
it. It represents the board, units, capabilities, rules, and legal commands as variable semantic records;
compares candidate successor consequences; learns first from complete teacher play and then from its own visited
states; treats draws as failures while preserving useful supervision; and expands through measured curricula
with retention gates. Unit design and new mechanics become later extensions of the same representation rather
than reasons to replace the model again.
