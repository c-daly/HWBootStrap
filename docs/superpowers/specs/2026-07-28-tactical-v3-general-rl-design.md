# Tactical-v3 General Reinforcement Learning Design

## Purpose

HexWars needs a tactical reinforcement-learning contract that can pursue the actual victory condition,
operate across board sizes, understand arbitrary unit designs, and eventually incorporate new purchasable
capabilities without rebuilding the model around a larger fixed vector.

The immediate behavioral defect is not general combat incompetence. The current tactical-v2 policy has not
lost to the random agent in observed play, but it draws frequently. Many draws occur after it has established
a decisive material advantage and then fails to find or finish the last survivors. Tactical-v3 therefore
treats annihilation conversion as a distinct learned skill and a first-class evaluation target.

This document defines the new model contract, its training objective, and the baseline evidence required
before implementation results are judged.

## Goals

- One checkpoint per game mode, never one checkpoint per board size.
- Whole-board spatial awareness across variable valid-hex counts.
- Variable numbers of units, templates, capabilities, rules, structures, and legal commands.
- Strategic interpretation of unit statistics rather than dependence on roster names or slot positions.
- Shared representations for commanding units and designing them.
- Pregame and midgame generation of complete, coherent unit designs.
- Safe ingestion of unfamiliar capability tokens, with fine-tuning required to learn genuinely new semantics.
- Persistent multiround intentions such as pursuing a distant or last-seen survivor.
- Annihilation as the primary objective, with useful learning signal preserved inside drawn games.
- Exact fog-of-war filtering and explicit last-observed memory without hidden-state leakage.
- Deterministic, replayable, statistically defensible evaluation.

## Non-goals

- A general agent query API. A future seat-filtered knowledge API remains useful but is deferred.
- Zero-shot mastery of mechanics the model has never experienced.
- A universal gameplay DSL capable of expressing every future rule.
- Designing armor penetration, medicine, aircraft, or other future mechanics in this work.
- Rewarding novelty, roster diversity, or particular unit archetypes for their own sake.
- Preserving tactical-v2 model compatibility. Tactical-v3 is an intentionally incompatible contract and
  requires new training.

## Verified Current Constraints

The current implementation informs the redesign:

- `UnitStats` is a fixed nine-field struct, and `PointCost` is the flat sum of all nine fields.
- Tactical-v2 uses fixed template HP planes and stable unit registry slots.
- Tactical-v2 writes all living enemies into its observation without applying fog visibility.
- Tactical-v2 exposes only points, round, and alive fractions as globals; remaining actions and per-unit
  action state are not observation features.
- Tactical-v2 flattens move and attack addresses over unit slots and board cells. `EndTurn` is always legal.
- Reward shaping counts each living unit at full point cost regardless of remaining health, rewards movement
  toward the nearest true-state enemy, charges a penalty per learner command, and gives material-based
  partial terminal credit at a round-cap draw.
- The existing Python evaluator runs deterministic held-out seeds in reciprocal seats and reports W/L/D with
  Wilson intervals, but it records no command-level tactical diagnostics.
- `TacticalV2DuelEnv` can already capture every accepted command transition on demand, and the replay system
  reconstructs authoritative final states.
- Existing model/controller validation requires exact observation and action sizes, so current checkpoints
  cannot establish cross-size generalization.

These facts make tactical-v2 useful as a behavioral baseline, not as an extensible foundation.

## Design Principles

### Structured rather than monolithic

The policy receives collections of typed objects and relationships instead of a fixed raster plus permanent
action neurons. A token is a fixed-width numeric row representing one object; the neural computation remains
ordinary matrix computation.

### Engine authority

The engine remains authoritative for legality, pathfinding, targeting, damage, visibility, costs, and state
transitions. The model learns strategic value, not a fallible approximation of command legality.

### Stable semantics, variable cardinality

Feature semantics remain fixed while row counts vary. Adding a unit, cell, template, or known-form capability
adds a row rather than a feature column. Runtime IDs link records during an episode but have no learned
strategic identity.

### Outcome first, progress informative

A draw is always a negative outcome. Good attacks, designs, and deployments inside a drawn game still provide
bounded learning signal. No amount of secondary progress may turn a draw into a positive episode or outweigh
an annihilation win.

### Generality is measured

Claims of generality require held-out board sizes, designs, rule configurations, and seeds. Training on a
varied distribution is not itself evidence of generalization.

### Interfaces over implementations

Every architectural boundary is an explicit, versioned interface. Here, interface means the appropriate
boundary mechanism: a C# interface inside the engine process, a serialized schema across the GymServer boundary,
a Python protocol between runtime components, or a named tensor contract between the environment and model. It
does not require a one-method interface around every private helper.

The logical ports are:

- seat-state source: produces an immutable, seat-filtered view without exposing authoritative hidden state;
- observation encoder: converts that view and its rule descriptors into typed token and relation collections;
- capacity profile and batcher: pads variable collections to configurable limits and supplies validity masks;
- legal-candidate source: enumerates complete legal commands from the current seat-visible contract;
- transition projector: produces an exact safe afterstate or sparse transition delta for a candidate;
- action resolver: round-trips a selected candidate identity to the exact authoritative command;
- reward contract: maps a transition to terminal and bounded progress components with a named breakdown;
- memory provider: supplies and updates last-observed records without acquiring unavailable information;
- policy interface: maps an observation/candidate batch to a masked candidate distribution and intentions;
- value/outcome interface: estimates return, terminal outcome, conversion probability, and victory horizon;
- design interface: maps contextual representations to an atomically valid unit-design proposal;
- rollout/storage interface: records versioned transitions without depending on one optimizer;
- optimizer interface: trains policy, value, and auxiliary heads without owning engine semantics;
- controller/checkpoint interface: validates schema identity and performs inference independently of training;
- evaluation sink: consumes replayable episodes and diagnostics without affecting observations or reward.

Dependencies point toward these contracts, never toward a downstream concrete implementation. Tactical-v1 and
tactical-v2 remain separate adapters. Tactical-v3 components may be replaced independently, and a new mechanic
adds an implementation or descriptor through the relevant port instead of requiring unrelated consumers to be
rewritten.

Interfaces remove the need to predict future implementations, not the need to specify meaning. Each port owns
explicit invariants, seat perspective, determinism rules, failure behavior, and a schema hash. The model sees
semantic records rather than C# object layouts, JSON property accidents, tensor offsets, or roster identities.

Actual row counts are match configuration. Batch capacities are separately configurable model/runtime values:
maximum cells, entities, templates, capabilities, relations, candidates, memory records, and design steps. The
observation and action spaces are derived from these capacities and the versioned feature schemas; users never
set observation length or action offsets independently. A match that exceeds the declared envelope fails
validation before reset. It never silently drops an entity or legal command.

This boundary grammar deliberately does not enumerate future mechanics. A future capability can provide its
descriptor, relationships, state records, legal candidates, and transition behavior through these ports. Existing
models may represent an unfamiliar descriptor generically; learning effective use still requires experience.

## Mode and Checkpoint Boundary

A model is trained for a game-mode family whose objective and command vocabulary are coherent, such as
annihilation tactical play or territory/economy play. Different board sizes are configuration values inside
that family and never select different checkpoints.

Within a mode, explicit rule tokens describe configurable values such as turn policy, action budget, round
cap, bounty, design fee, deployment multiplier, terrain mechanics, and fog. If a future rule changes the
objective or introduces a substantially different phase structure, it may justify a new mode checkpoint;
mere board dimensions do not.

## Observation Source

Tactical-v3 initially consumes a seat-filtered snapshot built directly from authoritative `GameState` plus
small per-seat observation memory:

```text
GameState + observing seat + ObservationMemory
    -> ISeatObservationSource
    -> structured tactical-v3 observation
```

`ISeatObservationSource` is an internal seam, not a public agent API. A future knowledge service may implement
the same seam without changing the neural contract.

Exact facts available to the seat belong in the snapshot or observation memory. Strategic inference remains
the policy's responsibility. For example, the environment records that an enemy was last seen at a cell on a
particular round; the policy infers where that enemy is likely heading.

## Token Observation Contract

The logical observation contains variable-row tables:

```text
cells          [Nc, Fc]
units          [Nu, Fu]
templates      [Nt, Ft]
capabilities   [Nk, Fk]
rules          [Nr, Fr]
memory         [Nm, Fm]
relations      [Ne, Fe]
```

`N*` values vary per state. `F*` widths and their semantics are contract-versioned. Training batches pad only
to the largest row count in that batch and carry explicit masks; padding has no game semantics.

### Cell tokens

There is one token for every valid hex in the observable board, not one token for every location in an
arbitrary bounding rectangle. Cell features include:

- axial coordinate representation in a seat-relative frame;
- terrain, elevation, control, visibility, and known structure state;
- boundary, deployment-region, and objective context where applicable;
- references or edges to occupying entities rather than duplicated complete unit records.

The policy receives the complete observable board cell by cell. Whole-board observation does not imply
all-to-all cell attention.

### Unit tokens

Unit records include:

- owner relation and stable within-episode linkage ID;
- current and maximum health;
- cell reference and elevation/domain/status information;
- complete capability allocations;
- movement already spent, moved/attacked state, and other rule-relevant per-turn state;
- current visibility or memory status.

Unit order is semantically meaningless. Permuting unit rows and updating their references must only permute
the corresponding internal representations and action candidates.

### Template tokens

Templates expose their actual capability allocation, cost, fixed/custom status, deployment availability, and
other mechanical restrictions. Roster names, ordering, and conventional roles carry no strategic meaning.

The roster is therefore an optional library of human-friendly examples and cached deployable designs. The
policy may deploy, ignore, evaluate, or replace eligible templates.

### Capability-definition tokens

A unit design is a variable collection of allocations linked to shared capability definitions. Current stats
such as Health, Damage, Defense, Movement, and Vision are represented through the same mechanism as future
capabilities.

Capability definitions contain structured descriptors that fit a stable meta-schema:

- cost and purchased level;
- passive, modifier, or active classification;
- source and target domains;
- target type and targeting constraints;
- known effect family and timing;
- action, cooldown, or use constraints where applicable;
- typed interaction edges.

Interaction edges include relations such as `enhances`, `opposes`, `reduces`, `restores`, `requires`, and
`enables_action`. Armor penetration could oppose Defense; a future medic capability could enhance or restore
Health. The engine formula remains authoritative. The relationship is an inductive hint, not a declaration
that the investment is strategically worthwhile.

An old model can map an unfamiliar capability to generic descriptors and safely give it little weight. Useful
transfer is expected only when the new mechanic is composed from descriptors the model already understands.
Fine-tuning and targeted experience are required for genuinely new semantics.

### Rule and economy tokens

Rules that change transitions, legality, reward meaning, or terminal conditions must be observed. At minimum:

- active game mode and win conditions;
- turn policy, actions per turn, and actions remaining;
- round and round cap;
- damage floor and relevant combat modifiers;
- banked points, bounties, design fees, deployment costs, and unit-design budget;
- terrain, fog, territory, generator, capture, and economy configuration when active.

Two visually identical boards with different remaining actions are different decision states and must not
share an identical observation.

### Fog and memory tokens

The observation never exposes hidden current state. When a visible enemy disappears, its exact last-observed
record becomes a memory token containing:

- last-seen cell and time;
- age since observation;
- last-observed health, capabilities, and action state;
- explicit uncertainty/currently-visible flags;
- a conservative reachable envelope derived only from known rules and observations.

Cells subsequently observed empty remove possibilities from that envelope. Runtime identity supports
continuity but never becomes a learned unit-type label.

Environment memory stores exact public history. A recurrent policy state may additionally learn beliefs such
as likely routes and intentions. The two mechanisms are complementary.

## Spatial Representation

Tokens are unordered; spatial structure is supplied explicitly.

For token pair `i, j`, attention may receive relative axial displacement, hex distance and direction,
elevation difference, neighborhood relation, visibility, reachability, or targeting relations. Spatial
relevance is capability- and command-conditioned; it is never a fixed rule that nearer objects are always
more important:

```text
attention(i, j)
    = semantic_similarity(i, j)
    + geometry_bias(i, j)
    + interaction_bias(capabilities_i, capabilities_j, relation_i_j, candidate)
```

Interaction features include range margin, movement-turn distance, line of sight, visibility, threat in both
directions, and whether the pair participates in a legal command. A target ten hexes away may be irrelevant to
a melee unit and immediately decisive to a unit with Range 10. Raw distance is evidence, not relevance by
itself.

Attention is directional: a friendly unit may attend strongly to distant enemy artillery because the enemy
threatens it, even when the friendly unit cannot return fire. Nearby weak enemies may dominate a local
occupancy or immediate-action head while the artillery dominates threat assessment and goal selection.
Multiple attention heads allow both facts to remain relevant instead of forcing the policy to choose one
universal ranking of objects.
These head roles are descriptive possibilities, not manually assigned jobs. Attention specialization, the
importance of each relation, and the tradeoff between immediate opportunity and residual threat are learned
from outcomes.

The entity encoder represents observed facts, not one final context-free strength ranking. Decision-specific
queries compute asymmetric relevance:

```text
attack_opportunity(enemy | my attacker)
    != defensive_threat(enemy | protected friendly asset)
    != counter_design_value(enemy | available budget and capabilities)
```

An attack query emphasizes killability, target value, retaliation, and opportunity cost. A defensive query
emphasizes what the enemy can reach or damage before a response and which friendly asset is at risk. A design
query emphasizes exploitable capability allocations and possible counters. The same enemy may rank differently
under each query. No fixed strength, threat, opportunity, or shots-to-kill heuristic selects the action. The
engine supplies mechanics and relations as facts; the learned policy and value function decide what those facts
mean in context.

The model uses seat-relative and relative geometry rather than learned absolute cell indices. Translating an
otherwise identical battle should not change its strategic representation. Boundary and objective features
retain the global context that translation alone cannot supply.

To avoid quadratic scaling on large boards:

- cell-to-cell attention is local or sparse;
- unit/entity sparsification takes the union of local neighbors, legal interactions, capability-conditioned
  threat/reach envelopes, active goals, and selected global links;
- every legal long-range interaction receives a direct path even when it lies outside the local cell window;
- global army/economy/rule tokens provide long-range communication;
- exact reachability and legality relations come from the engine;
- full all-to-all attention is reserved for bounded entity or summary sets where justified.

## Shared Policy and Value Architecture

A shared attention encoder processes cells, entities, templates, capabilities, rules, memory, and relations.
It feeds:

- a tactical command pointer;
- a persistent intention pointer;
- an atomic unit-design decoder;
- a value head;
- optional auxiliary prediction heads.

The design and command heads share representations because effective design depends on tactical context, and
effective command depends on recognizing the capabilities of arbitrary designs.

Rollout and optimization remain standard actor-critic phases. Network weights are fixed during rollout.
PPO later revisits stored observations and actions, computes advantages, and backpropagates policy and value
losses through the heads and shared attention encoder.

For transitions whose immediate result is knowable, candidate evaluation uses the exact successor observation
produced through the authoritative engine transition. The engine determines what each command does; the learned
network compares the resulting states. Conceptually, net desirability is opportunity created minus threat
remaining: a learned goal-conditioned afterstate value, not a hand-authored component score.

## Tactical Action Contract

### Legal-command candidates

The engine enumerates each concrete legal command and the adapter represents it as a candidate token:

```text
Move(actor, reachable cell)
Attack(actor, target entity)
Deploy(template, cell)
Claim(actor, cell)
Build(actor or controlled cell)
EndTurn
```

Each legal candidate pairs its command token with the seat-filtered observation of the state resulting from
that command. In notation, afterstate_i = Observe(Apply(current_state, command_i), evaluating_seat).

In a fully observed deterministic match this is an exact engine result, not a neural prediction. Under fog of
war, candidate construction may expose only consequences determined by the evaluating seat's information; it
must never obtain hidden-state knowledge by applying the command to an unfiltered authoritative state.

The learned scorer compares the current state, the candidate command, and its successor. Its preference is
Q-like: candidate_value_i = immediate_reward_i + gamma * V(afterstate_i).

The policy may learn this comparison directly from candidate and afterstate embeddings rather than using that
equation as a fixed action rule. The state value remains from the evaluating seat's perspective even if the
command ends the turn. A low successor value may reflect, for example, that two easy kills do not compensate
for leaving distant artillery a decisive response. That conclusion is learned from returns rather than encoded
as an artillery or threat heuristic.

State comparison is goal-conditioned. A locally safe or materially favorable afterstate is not valuable if it
cannot plausibly lead to annihilation. The shared encoder therefore also predicts:

- probability of eventual victory from the afterstate;
- probability of victory within each of several opponent-response horizons;
- remaining rounds and decision steps to victory, conditional on eventual victory.

Together these represent conversion potential: first whether the advantage can become victory, then how long
and how many opponent responses that conversion is likely to require. They avoid treating an unreachable last
survivor as a strong state merely because friendly material is dominant.

Candidate construction may also attach engine-derived facts such as distance, path cost, elevation relation,
deterministic damage, lethality, and action-budget consumption. These facts describe the transition but do not
prescribe its strategic value or reveal hidden information.

The policy scores the variable candidate matrix and samples from a masked categorical distribution. One
selected candidate becomes one authoritative engine command. Invalid decoding never silently falls back to
`EndTurn`; it is a contract error and diagnostic event.

Exact afterstate encoding can be expensive when the legal set is large. Implementations may batch successor
encodings, represent them as sparse state deltas, or use a learned command prior to shortlist candidates. Any
shortlisting is a measured approximation and must preserve a path for long-range and unusual legal commands;
it must not reintroduce fixed board dimensions or fixed unit slots.

### Persistent intention pointer

Immediate moves remain legal within the unit's current-turn movement budget. Separately, the policy may point
to a distant cell, entity, memory token, structure, or tactical relation as a multiround intention:

```text
goal = Pursue(last-seen enemy 12)
action = Move(unit 7, currently reachable cell 41 | goal)
```

The intention conditions immediate command scoring and persists until reached, invalidated, or explicitly
replanned. The engine never executes an entire multiround path blindly. Each world transition still uses a
fresh legal command.

The goal pointer is experimental and separable. Its value is measured through an ablation against an
otherwise identical pointer policy. No direct reward is paid merely for reducing distance to a goal; outcomes
must establish whether pursuit was wise.

### Atomic design decoder

Ordinary tactical commands are enumerable. Complete designs are combinatorial and use an internal
autoregressive decoder:

1. The command pointer selects an eligible design/replace operation and template slot.
2. The decoder repeatedly points to capability definitions and assigns budget amounts.
3. Engine masks enforce remaining budget and mechanical validity.
4. A stop decision completes one coherent design.
5. The full design is submitted as one atomic engine command; internal allocation decisions do not advance
   the game clock.

The combined action log probability is the command-selection log probability plus the decoder-step log
probabilities. Pregame and midgame design use the same decoder under different legal contexts.

Design generation must not recreate the adaptive-v1 behavior of paying a design fee for each single-stat
edit. One completed design pays the authoritative configured fee once.

## Variable-Size Training Mechanics

The semantic contract is ragged even if a specific tensor library requires padded minibatches. A global
maximum board size or action count must not become a learned identity or checkpoint-selection boundary.

Tactical-v3 requires a policy and rollout implementation that supports:

- masked variable-row observations;
- masked variable candidate sets;
- hierarchical/autoregressive action log probabilities;
- recurrent or explicit goal state;
- per-sample relation graphs;
- deterministic inference and replayable sampling.

The implementation plan will choose between extending the current Python stack and adopting a framework with
native structured/autoregressive distributions. Framework convenience must not weaken the contract back into
a permanent flattened action grid.

## Reward Contract

### Terminal objective

An annihilation win is positive, a loss is negative, and a round-cap draw is always negative for both seats.
Exact coefficients are chosen after baseline measurement, but the win/non-win ordering and draw bound are
contractual:

```text
annihilation win > every non-win outcome
every draw < 0
```

Draw and loss may be identical or separated numerically. Their relationship controls how much win probability
justifies risking defeat and is therefore a strategic risk-preference parameter, not a cosmetic constant. The
baseline must expose suicidal gambling, draw acceptance, and deliberate stalling before that value is frozen.

### Bounded progress signal

Drawn games contain valuable decisions. Progress uses a normalized, health-sensitive material measure rather
than full point cost for every living unit:

```text
effective unit value = deploy-equivalent value * current HP / maximum HP
```

Changes in known material, public damage, destroyed value, deployable resources, and useful committed force
may provide bounded secondary reward. Coefficients and normalization depend on match resource scale so that
the same model can train across different point budgets.

The complete progress contribution is bounded tightly enough that:

- no draw becomes positive;
- no material-rich draw outranks a win;
- repeated possession of the same advantage cannot be farmed;
- hidden movement cannot leak through reward.

Direct nearest-enemy closing reward is removed. It encodes a method, can reward suicidal pursuit, and can leak
true hidden positions.

### Draw severity and classification

Draw reward may include bounded adjustments for final known material margin and public no-progress duration,
while remaining negative. More detailed failure modes remain diagnostics rather than increasingly elaborate
hand-authored reward terms:

- avoidance/no contact;
- balanced attrition;
- failed conversion from decisive advantage;
- damage stalemate;
- mobility stalemate;
- action waste or premature EndTurn;
- cycling/repetition;
- step truncation;
- invalid or mechanically impossible scenario.

Failed-conversion states feed targeted curriculum rather than causing the successful early and middle-game
portion of the trajectory to be unlearned.

### Time pressure

Per-command step penalty is replaced by small game-time pressure tied to completed turns or rounds. Using
several productive commands inside one turn should not cost more merely because the policy exercised its
available actions. Coefficients must be audited for the opposite failure mode, intentionally accepting an
early loss to avoid later time costs.

### Auxiliary objectives

Auxiliary supervised or self-supervised losses may train the shared encoder from every transition without
altering policy return. Candidate tasks include:

- deterministic damage and lethality prediction;
- legal reachability and targetability prediction;
- next visible-state prediction;
- capability-interaction prediction;
- masked token reconstruction;
- terminal-outcome prediction;
- win-within-horizon prediction at several opponent-response horizons;
- remaining rounds and decision steps to victory, conditional on eventual victory.

Won trajectories provide exact remaining-time labels. For losses and terminal draws, the conditional
time-to-victory loss is masked; for time-limit truncations, horizons beyond the observed trajectory are treated
as censored rather than assigned an arbitrary large target. Outcome probability remains primary, so the model
cannot prefer a rare quick win over a reliably slower win merely because its conditional time estimate is
shorter.

Auxiliary metrics remain separate from W/L/D and reward reporting.

## Training Strategy

Training proceeds in measurable stages rather than introducing every novel component simultaneously:

1. Variable board/entity/template tokens, explicit rule state, legal-command and exact-afterstate scoring,
   value/outcome/time-to-victory heads, and revised reward using the current nine capabilities and existing
   commands.
2. Persistent intention pointer plus failed-conversion endgame curriculum.
3. Atomic full-template design decoder sharing the same encoder.
4. Capability-definition tokens and relationship graph, initially representing existing mechanics.
5. Population self-play and procedural design curricula across supported mode configurations.
6. Later mechanics as extension tests, not prerequisites.

The opponent population includes:

- random and greedy scripted controls;
- frozen historical checkpoints;
- current-policy mirrors;
- diverse design policies;
- pursuit-focused hunters and evasion-focused survivors;
- targeted exploiters for dominant unit designs.

Training samples board sizes, terrain, starting resources, unit counts, template counts, unit allocations, and
rule values within the mode. Evaluation reserves held-out combinations and sizes. Adding a new capability uses
targeted situations where it could matter, but never rewards using it merely because it is new.

## Baseline Work Package 0

Baseline work is completed before tactical-v3 performance claims. Exhaustive means predefined coverage,
paired reproducibility, causal intervention, and preserved evidence; it does not mean enumerating the infinite
configuration space.

### Frozen evidence

Create an immutable manifest containing checkpoint hashes, code revision, contract/scenario hashes,
hyperparameters, controller identities, inference mode, and fixed development/validation/test seed banks.

### Current-contract audit

Document observation omissions, action-mask and decoding behavior, reward decomposition, seat symmetry,
determinism, replay reconstruction, and the fixed-size boundary of tactical-v2.

### Engine-feasibility grid

Run reciprocal scripted matchups across representative boards, terrain, turn policies, point budgets, unit
counts, and combat compositions:

```text
random vs random
greedy vs random
random vs greedy
greedy vs greedy
hunter vs evader
```

This establishes whether annihilation is mechanically achievable before blaming a learned policy.

### Frozen-policy tournament

Evaluate the final tactical-v2 checkpoint against random, greedy, mirror, selected historical checkpoints,
hunter, and evader controllers. Use paired seeds and reciprocal seats. Major matchup cells target 500 paired
seeds, or 1,000 games, producing roughly a three-percentage-point worst-case 95% margin for a proportion near
50 percent.

Run deterministic inference as the primary official result and stochastic replications on a stratified seed
subset to measure policy uncertainty.

### Per-episode and per-command telemetry

Add evaluation-only transition analysis. The agent never observes these diagnostics. Record:

- outcome, seat, rounds, commands, and truncation reason;
- health-adjusted material advantage and peak advantage;
- first entry into several decisive-advantage thresholds;
- time from advantage to annihilation or draw;
- damage, kills, deployments, banked points, and surviving value;
- EndTurn choices, remaining action budget, and productive legal actions left;
- distance/path distance to the last survivor;
- current and eventual ability to reach and damage survivors;
- repeated positions, cycles, and no-progress intervals;
- policy entropy, EndTurn probability/rank, value estimate, and top candidates where available;
- reward decomposed by source.

Save every draw replay plus stratified win/loss controls. All reported episodes must reconstruct to the same
terminal state through the authoritative replay path.

### Draw classification

Classify avoidance, balanced attrition, failed conversion, damage stalemate, mobility stalemate, action waste,
cycling, truncation, and invalid scenarios. Preserve an explicit `unclassified` category; uncertain evidence
must not be forced into a confident label.

### Counterfactual endgame forks

Save snapshots at decisive advantage and continue identical states under controlled interventions:

- original policy;
- extended round cap;
- EndTurn suppression while an attack is legal;
- pursuit-waypoint heuristic;
- hunter replacement;
- repeated stochastic policy sampling.

These interventions distinguish insufficient time, bad incentives, missing persistence, and mechanical
impossibility.

### Learning-trajectory audit

Evaluate log-spaced historical checkpoints on the same seeds. Compare combat competence, conversion, action
entropy, value accuracy, episode length, PPO KL, clip fraction, explained variance, and seat asymmetry. This
shows whether pursuit was never learned, learned late, or forgotten.

### Baseline artifacts

The work package produces:

```text
manifest.json       identities, hashes, and seed banks
games.jsonl         one authoritative row per game
diagnostics/        detailed episode/transition summaries
replays/            all draws and stratified controls
summary.json        aggregate statistics and intervals
report.md           conclusions and causal evidence
```

## Primary Evaluation Metrics

- Annihilation W/L/D with confidence intervals.
- Seat-specific and macro-averaged configuration performance.
- Failed-conversion rate.
- Conversion probability as a function of peak normalized advantage.
- Median and distribution of rounds to convert.
- EndTurn waste rate.
- Mechanical versus behavioral draw rate.
- Performance on held-out board sizes, designs, rule values, and seeds.
- Unit-design best-response performance against concentrated and specialized opponents.
- Policy calibration, entropy, and value error in endgame states.

No single aggregate win rate may hide a severe regression in one board-size or configuration stratum.

## Error Handling and Contract Integrity

- Hidden state reaching observation, candidate features, reward, or memory is a hard contract failure.
- Nonterminal states must expose at least one legal candidate; normally `EndTurn` remains available.
- A selected candidate must round-trip to the exact command represented. Invalid candidates are reported and
  never converted silently to another command.
- Generated designs are masked and validated incrementally, then authoritatively validated atomically.
- Unknown capability descriptors use an explicit generic representation. Unsupported semantic shapes fail
  contract validation rather than being misread as a known capability.
- Ragged masks and relation references are range-checked before model inference.
- Observation/action/capability schema hashes are stored with every checkpoint and evaluation artifact.
- Deterministic evaluation failures include the seed, seat, controller identities, scenario hash, and replay
  path.

## Verification

### Contract tests

- Unit, template, capability, and candidate permutation invariance/equivariance.
- Translation and seat-reflection consistency where game rules are symmetric.
- Multiple board sizes in one training batch.
- No fixed roster-name or template-slot semantics.
- Fog filtering, last-seen aging, negative observation, and reachable-envelope correctness.
- Remaining-action and per-unit action state distinguish otherwise identical boards.
- Candidate set equals authoritative legal commands exactly.
- Candidate decode/encode round trips without fallback.
- Atomic design validity, budget masks, combined log probability, and one-fee behavior.
- Persistent goal survival, invalidation, and replanning.
- Every port passes conformance tests against at least one test double and its production implementation.
- Contract handshakes derive tensor geometry from schema and capacity values without duplicated offsets.
- Substituting an implementation behind one port does not change unrelated serialized or tensor contracts.
- Capacity overflow fails before reset; no entity, relation, memory record, or legal candidate is truncated.

### Reward tests

- Every draw remains negative under all bounded adjustments.
- No draw outranks a win and no progress term hides a loss.
- Damage changes health-sensitive material value before a kill.
- Hidden movement produces no reward signal.
- Repeating an unchanged advantage cannot farm reward.
- Multiple useful commands within one turn do not accumulate a per-command time penalty.
- Draw classification and reward diagnostics remain separate.

### Determinism and replay tests

- Identical seeds, policies, and inference modes reproduce command sequences.
- Recorded structured actions reconstruct authoritative replay winners and states.
- Variable-size batching and worker count do not change deterministic results.
- Physics/task or engine changes run the existing determinism-sensitive suites.

### Learning and performance tests

- Short smoke training proves finite losses, gradients, masks, and checkpoint reload.
- Pointer probabilities normalize over only legal candidates.
- Autoregressive design log probabilities are reproducible under replayed samples.
- Headless throughput and memory are measured by board/entity/candidate count.
- Ablations compare token/pointer baseline, persistent intention, revised reward, auxiliary heads, and design
  decoder independently.

## Deferred Extensions

- A seat-filtered query/knowledge API may later replace direct snapshot traversal behind
  `ISeatObservationSource`.
- Active information-gathering actions such as scans belong to game mechanics, not the deferred read-only API.
- Aircraft may later introduce movement domains or airborne status; jump and flight semantics are intentionally
  not fixed here.
- Medicine, armor penetration, and other capabilities will test the relationship graph after the existing
  capabilities work end to end.

## Design Outcome

Tactical-v3 is an object-centric, spatially structured actor-critic contract. It sees the complete observable
board, reasons over variable entities and capability relationships, points to legal commands, retains distant
intentions, and generates coherent unit designs. Its training signal recognizes useful progress without
mistaking a draw for success. Its baseline suite distinguishes combat ability, conversion failure, mechanical
stalemate, and optimization health before any new architecture is credited with improvement.
