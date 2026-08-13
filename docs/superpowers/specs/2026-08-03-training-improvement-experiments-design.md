# Training Improvement Experiments

## Status

Approved conversational design. This specification defines the next diagnostic
training sequence; it does not authorize consuming final evaluation seeds,
launching full production panels, or implementing the complete tactical-v3
contract.

## Problem

HexWars tactical policies routinely gain material advantages against Random but
often fail to convert those advantages into annihilation. More training steps,
reward adjustments, and a static conversion curriculum have not reliably solved
the problem.

The current evidence is unusually informative:

- Profiled standard PPO won 381 of 540 conversion games (70.6%). Mixed
  conversion PPO won 396 of 540 (73.3%), a gain of only 2.78 percentage points.
- Bounded search won 168 of 180 conversion games (93.3%) and reduced cycling
  draws to eight. Terminal-only search still won 154 games (85.6%).
- The three pure behavioral clones failed their standard-game gate. Seeds 211,
  223, and 227 won 26, 59, and 58 of 200 games respectively, for a pooled 23.8%
  win rate. Their failures were overwhelmingly draws.
- One BC-initialized PPO seed reached 51,036 environment steps with target KL
  0.02 and stable observed KL, but it stopped without a physical 51,200-step
  checkpoint and has not received the required checkpoint-trajectory audit.
- The separate `bc223b3a12` run is not evidence for annihilation training. It
  used the legacy standard scenario, draw terminal credit 0.25, closing reward
  0.02, no target-KL option, one worker, and a 30-million-step target. Near its
  stopping point its approximate KL repeatedly reached roughly 0.24-0.57 and
  its clip fraction roughly 0.25-0.41.

Together these results suggest three distinct hypotheses:

1. PPO may improve, preserve, or erase parts of the cloned policy over training.
2. Behavioral cloning may fail primarily because it sees teacher-visited states
   rather than the states induced by learner mistakes.
3. The fixed 1,288-way tactical-v2 action head may be a poor representation for
   comparing variable legal commands and their strategic consequences.

The experiment sequence isolates those hypotheses in order.

## Goal and success criterion

The immediate milestone remains a policy that consistently beats Random through
annihilation rather than lopsided draws. A candidate is provisionally successful
when it wins at least 65% of standard development games in each replicated seed
and at least 70% pooled, with reciprocal seats and cycling reported separately.

Development thresholds select the next experiment; they do not constitute a
production promotion. Any final claim requires untouched seed banks and the
existing frozen-evidence process.

## Non-goals

- No broad PPO hyperparameter sweep.
- No further reward shaping in tactical-v2.
- No resumption or interpretation of the unstable legacy 30-million-step run.
- No self-play, Greedy training opponent, or DQN comparison in this sequence.
- No complete tactical-v3 implementation before the smaller causal experiments
  have answered their questions.
- No use of final or confirmation seed namespaces during development.

## Experiment 1: physical-checkpoint audit

### Question

Does target-KL PPO improve, preserve, or destroy the behavior present in the
seed-227 behavioral clone?

### Candidates

- The physical seed-227 pure-clone archive.
- Every physical checkpoint already published by its BC-to-PPO run.
- A contract-compatible from-scratch PPO reference where one physically exists.
- Random and bounded-search controls as outcome anchors.

The stopped in-memory state at 51,036 steps is not a checkpoint. It must not be
reconstructed and described as the same trajectory. A later 51,200-step run is
a fresh, explicitly identified replicate.

### Evaluation schedule

Each candidate plays the same 100 standard development maps in both seats, for
200 games per candidate. The schedule uses the established
`16,000,000-16,000,099` development namespace. It records map seed, candidate
seat, checkpoint digest, scenario and contract identities, outcome, replay, and
tactical trace.

This incomplete seed-227 audit is explicitly exploratory. It cannot replace the
locked imitation panel's 21-candidate development schedule, global-budget
selection, scratch controls, or three-seed final gate.

### Metrics

- W/L/D and Wilson intervals.
- Paired map-seat changes between successive checkpoints.
- Cycling and action-waste draw incidence.
- Rounds and decisions to annihilation.
- Final and peak health-adjusted advantage.
- Candidate EndTurn rank/probability when productive commands remain, when the
  policy interface exposes those quantities.
- Seat asymmetry.

### Decisions

- If PPO reaches at least 65% wins and improves consistently over the clone,
  reproduce the condition with seeds 211 and 223.
- If an earlier checkpoint exceeds a later checkpoint by at least ten percentage
  points, test PPO with a retained imitation constraint.
- If every checkpoint remains below 50% wins, or cycling remains the dominant
  result, proceed directly to Experiment 2.

## Experiment 2: DAgger search distillation

### Question

Can learner-state teacher supervision rescue the current tactical-v2
representation?

Ordinary behavioral cloning trains on observations produced by the teacher's
state distribution. DAgger (Dataset Aggregation) rolls out the learner and asks
the teacher to label the states induced by learner decisions:

```text
D <- D union {(learner observation, legal mask, teacher action)}
```

This directly tests covariate shift without changing reward or action geometry.

### Seed isolation

The DAgger panel reserves new, disjoint namespaces before collection:

- `18,000,000-18,999,999`: learner rollouts and teacher labeling, subdivided by
  model seed and aggregation iteration in the immutable panel manifest;
- `19,000,000-19,099,999`: held-out supervised validation states; and
- `20,000,000-20,000,099`: reciprocal closed-loop development games.

The existing `17,000,000-17,000,249` final bank remains untouched and unassigned
by these experiments.

### Collection loop

1. Initialize from the best candidate selected by Experiment 1.
2. Roll out against Random using 70% standard and 30% declared conversion starts.
3. At every learner decision, query bounded search on the same seat-visible
   information for its preferred legal action.
4. Execute the learner action, not the teacher action, so subsequent states remain
   the learner's true closed-loop distribution.
5. Collect 20,000 valid labeled decisions per iteration for three iterations.
6. Retrain the existing HexCNN actor on the aggregate dataset.
7. Evaluate after every iteration on the fixed development schedule.

The first experiment uses one model seed. A successful treatment is then repeated
with three independently initialized seeds.

### Dataset contract

Every row contains:

- observation and legal mask;
- learner and teacher actions;
- disagreement flag;
- authoritative command identity and round-trip evidence;
- state hash, profile, map seed, episode seed, and seat;
- teacher identity, parameters, and search budget;
- scenario, contract, encoding, code, and dataset identities.

Teacher actions must be legal under the recorded mask and round-trip to the
authoritative command. Any mismatch fails collection rather than falling back to
EndTurn.

Repeated state hashes are capped within an episode and weighted so cycles do not
fill the dataset with hundreds of equivalent labels. The aggregate retains the
original teacher demonstrations in every training mixture to protect opening and
combat behavior while adding recovery behavior.

Search scores or root rankings may be recorded as diagnostics. The first causal
test trains on the selected teacher action only; score-based objectives remain a
separate ablation.

### Gate

The one-seed experiment succeeds only if it produces both:

- at least a 20-percentage-point standard-game improvement over its starting
  policy or at least 65% standard wins; and
- at least a 50% relative reduction in cycling-draw incidence.

Interpretation:

- Strong improvement authorizes three-seed replication and then a short,
  target-KL PPO fine-tune.
- High held-out teacher-action accuracy without closed-loop improvement indicates
  that the flat action representation is the likely bottleneck.
- Poor held-out teacher-action accuracy motivates representation/capacity analysis
  before any PPO continuation.

### PPO retention experiment

Only after DAgger produces a competent closed-loop policy, compare:

- target-KL PPO without imitation retention; and
- target-KL PPO with a decaying auxiliary masked-imitation or policy-KL loss.

Both use identical seeds, rollouts, checkpoints, and evaluation maps. This
separates useful RL correction from catastrophic forgetting.

## Experiment 3: candidate/afterstate architecture

### Entry condition

Enter this experiment when DAgger labels are learnable but the flat tactical-v2
policy still fails to convert games. It implements only tactical-v3 stage one.

### Contract

For each seat-visible state `s`, engine authority enumerates all concrete legal
commands `A(s)`. For every candidate `a_i`, the transition projector produces
the exact safe seat-visible afterstate:

```text
afterstate_i = Observe(Apply(s, a_i), evaluating_seat)
```

The policy scores the variable candidate set:

```text
score_i = f(current_state, command_i, afterstate_i)
policy = softmax(scores over legal candidates)
```

This lets the network learn comparisons such as immediate opportunity versus
remaining threat without hardcoded artillery, strength, pursuit, or
shots-to-kill rules.

### Minimum architecture

- Variable cell, unit, template, and rule tokens.
- Seat-relative spatial and interaction relations.
- Authoritative legal-command candidate tokens.
- Exact afterstate encodings or validated sparse transition deltas.
- A shared attention encoder.
- Masked candidate policy scores.
- State value, eventual-win probability, win-within-horizon probabilities, and
  conditional decisions/rounds-to-victory heads.

The first slice excludes unit design, capability-definition generalization, fog
memory, and persistent intentions. Those remain independently testable stages.

### Variable-size requirement

Actual board cells, entities, templates, relations, and legal candidates are
variable-row collections. Implementations may pad minibatches to a configurable
capacity and supply masks, but board dimensions never select a checkpoint. A
match exceeding the declared capacity envelope fails before reset; no legal
candidate or entity is silently truncated.

### Training sequence

1. Distill bounded search on learner-visited states, recording rankings or
   scores when the teacher can expose them faithfully.
2. Evaluate closed-loop play against Random across multiple development board
   sizes and reciprocal seats.
3. Fine-tune with actor-critic learning while retaining a measured decaying
   distillation loss if unregularized RL erases competence.

### Required ablations

- Current-state plus command token versus current-state, command, and afterstate.
- Flat tactical-v2 action head versus variable candidate scorer.
- Candidate scorer without versus with a persistent intention pointer.

Persistent intention is added only if candidate scoring wins battles but still
cycles against reachable survivors. It must not receive a direct distance reward.

## Data flow and interfaces

The experiments use the existing research boundaries:

```text
Engine state
  -> seat-visible observation
  -> legal-candidate source
  -> optional exact transition projector
  -> policy/controller
  -> authoritative action resolver
  -> transition, reward breakdown, trace, and replay
```

Teacher-label generation is quarantined from policy inference. Full authoritative
state may support offline diagnostics only when every teacher input is also
available through the learner's seat-visible contract for the no-fog experiment.
No teacher-only fact enters the policy observation or reward.

Each logical boundary retains its own schema and identity. Scenario, contract,
encoding, checkpoint, dataset, teacher, evaluation schedule, and code identities
must be present in reusable artifacts.

## Error handling and reproducibility

- Illegal or non-round-tripping labels fail closed.
- Candidate enumeration must equal the authoritative legal-command set.
- Nonterminal states must expose at least one candidate.
- Invalid decoding never falls back to EndTurn.
- Duplicate or aliased evaluation rows are rejected.
- Interrupted stages retain diagnostics but cannot appear complete.
- Evaluation reuse requires physical artifact reopening and digest validation.
- Learner, teacher, and evaluation seed namespaces remain disjoint.
- Final and confirmation banks remain untouched until a development gate passes.

## Verification

Before compute-heavy runs:

- unit-test DAgger aggregation, state-hash caps, masks, action round trips, and
  restart behavior;
- prove deterministic teacher labels for fixed state and search budget;
- prove evaluation schedules are reciprocal and seed-disjoint;
- prove candidate permutation equivariance and legal-set completeness;
- prove multiple board sizes can share one model batch and checkpoint contract;
- run full engine, Python, and Unity checks required by the changed boundaries;
- run a small end-to-end smoke that reopens every physical artifact.

## Decision outcome

The sequence produces one of three useful conclusions:

1. Existing BC-to-PPO already learns annihilation, so replicate it cleanly.
2. DAgger repairs the policy, establishing covariate shift as a major cause and
   providing learner-state data for tactical-v3.
3. DAgger labels are learnable but closed-loop performance remains poor, providing
   direct evidence to replace the flat action head with candidate/afterstate
   scoring.

No outcome authorizes returning to unbounded PPO training without evaluation.
