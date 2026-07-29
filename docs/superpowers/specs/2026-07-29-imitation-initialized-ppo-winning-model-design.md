# From Draws to Annihilation

## Imitation-Initialized PPO for a Winning HexWars Policy

- **Document type:** Graduate-level reinforcement-learning course project and implementation design
- **Status:** Design approved; implementation pending
- **Date:** 2026-07-29
- **Primary milestone:** Consistently defeat Random in standard tactical-v2 3-v-3 games
- **Audience:** Students who have completed a machine-learning course but may be new to reinforcement learning

## Abstract

HexWars is a configurable, turn-based tactics game and reinforcement-learning
research environment. The current tactical-v2 MaskablePPO agents learn useful
combat behavior and almost never lose to a Random opponent, but they convert too
few favorable positions into annihilation victories. Most failures are round-cap
draws, frequently caused by repeated or cyclic behavior after the agent has
already obtained a material advantage.

Prior experiments tested reward ablations, PPO optimizer changes, longer
training, and a curriculum containing mechanically valid late-game conversion
states. These interventions improved some measurements but did not eliminate the
central behavior. In contrast, deterministic scripted controllers demonstrate
that the environment is winnable: Greedy wins 90.0 percent of the conversion
suite, and a depth-four bounded-search controller wins 93.3 percent. The search
controller also supplies many cases in which it wins from a start where PPO
draws.

This project uses those controllers as teachers. First, behavioral cloning (BC)
trains the existing HexCNN actor to imitate legal teacher actions. Second,
Maskable Proximal Policy Optimization (PPO) fine-tunes the cloned policy using
environment return. BC addresses behavioral discovery: it supplies dense labels
for pursuit, combat, and conversion. PPO addresses the limitations of imitation:
it can improve beyond the teachers and optimize actual game outcomes under the
state distribution induced by the learned policy.

The first milestone is intentionally narrower than the long-term general-agent
goal. Three independently trained BC-to-PPO pipelines must each win at least 65
percent of 500 held-out reciprocal games against Random, and their pooled win
rate must reach at least 70 percent. Draws and losses are both failures. The
project holds reward, action semantics, and the tactical-v2 model contract fixed
so that the causal intervention is the source of policy initialization.

---

## 1. Learning objectives and prerequisites

### 1.1 Expected prerequisites

The reader is assumed to know:

- vectors, matrices, probability distributions, and expectations;
- gradients, backpropagation, and stochastic gradient descent;
- neural-network training, train/validation/test splits, and overfitting;
- multiclass classification and cross-entropy loss;
- convolutional neural networks at an introductory level.

No prior reinforcement-learning course is assumed.

### 1.2 Learning objectives

After studying and completing the project, a student should be able to:

1. formulate a turn-based tactics problem as an episodic Markov decision process;
2. distinguish environment state, policy observation, legal-action mask, action,
   reward, return, value, action value, and advantage;
3. derive masked behavioral-cloning loss from multiclass cross-entropy;
4. explain actor-critic learning, generalized advantage estimation, and PPO's
   clipped surrogate objective;
5. explain why supervised action accuracy does not imply closed-loop game
   competence;
6. identify covariate shift in sequential imitation and motivate DAgger-style
   data aggregation;
7. design disjoint demonstration, development, and final-evaluation seed banks;
8. use paired reciprocal evaluation and confidence intervals to distinguish
   robust improvement from lucky maps, seats, or training seeds;
9. interpret a negative experimental gate without discarding useful evidence;
10. connect algorithmic abstractions to a real engine's observation, action,
    reward, replay, and provenance interfaces.

---

## 2. Problem statement

### 2.1 The behavioral failure

The practical problem is not that the learned policy is uniformly incompetent.
It can fight, gain material advantages, and avoid losing to Random. The failure
is conversion: after reducing the enemy to a small surviving force, the policy
often stops making progress, ends turns unproductively, or cycles among repeated
states until the round cap.

This distinction matters. A model that wins early exchanges but draws at round
100 has learned useful local behavior without learning a reliable closed-loop
strategy for annihilation. Treating every draw trajectory as wholly useless
would throw away good combat decisions; treating a lopsided draw as success
would teach the wrong terminal objective.

For the first milestone:

```text
success = authoritative annihilation victory
failure = draw or loss
```

Material advantage, damage, distance, and draw category are diagnostics. They do
not replace the primary outcome.

### 2.2 Why another long PPO run is not the default answer

The empirical sequence already rules out several simple explanations:

- terminal-only reward did not produce a competent policy;
- dense material progress was useful, but did not solve conversion;
- target-KL PPO reduced extreme update pathology;
- extending target-KL models to approximately 100,000 environment steps did not
  improve the paired aggregate;
- conversion-focused curricula produced substantially more late-game practice
  but only a small, inconsistent win-rate gain;
- cycling remained the dominant draw class.

The key inference is not that PPO can never work. It is that more on-policy data
from the same initial policy family has low expected information value. A
different intervention should give the actor access to behavior that PPO has
failed to discover reliably.

### 2.3 Scope boundary

This is a shortest-path project for producing a winning learned model under the
current tactical-v2 contract. It is not the final general HexWars agent.

In scope:

- standard 13-by-9 tactical-v2 3-v-3 games;
- the existing fixed template catalog and fixed action geometry;
- Random as the training opponent and primary milestone opponent;
- Greedy and bounded search as demonstration teachers;
- behavioral cloning followed by MaskablePPO;
- three training replicates and sealed reciprocal evaluation.

Out of scope for this milestone:

- arbitrary board-size generalization;
- tactical-v3 token and candidate architectures;
- unit-design generation;
- fog-of-war memory;
- self-play or Greedy-opponent training;
- DQN;
- reward redesign;
- AlphaZero-style learned search.

These exclusions are experimental controls, not claims that the deferred ideas
are unimportant.

---

## 3. Empirical motivation

### 3.1 Conversion curriculum

The profiled curriculum compared standard-trained and mixed-conversion PPO at
51,200 steps across three training seeds. Aggregate conversion results were:

| Condition | Conversion W-L-D | Win rate | Cycling draws |
|---|---:|---:|---:|
| Profiled standard PPO | 381-36-123 | 70.6% | 123 |
| Mixed conversion PPO | 396-25-119 | 73.3% | 119 |

The absolute win-rate gain was 2.78 percentage points, far below the locked
15-point gate. One training seed regressed. The result demonstrates that valid
finishing-state practice alone did not reliably install the missing behavior.

### 3.2 Bounded-search positive control

The conversion suite was also evaluated with deterministic scripted controllers:

| Controller | Conversion W-L-D | Win rate | Cycling draws |
|---|---:|---:|---:|
| Bounded search | 168-4-8 | 93.3% | 8 |
| Terminal-only search ablation | 154-2-24 | 85.6% | 24 |
| Greedy | 162-11-7 | 90.0% | 7 |
| Mixed PPO, three-seed aggregate | 396-25-119 | 73.3% | 119 |

Bounded search used an expansion budget of 512 and maximum depth four. It made
2,612 decisions, expanded 1,008,170 authoritative transitions, and averaged
approximately 9 ms of search time per decision. The retained evidence includes
99 planner-win/PPO-draw disagreements.

### 3.3 Interpreting the search gate correctly

The search panel's locked *composite* gate failed because adding its nonterminal
pursuit heuristic improved win rate over terminal-only search by 7.8 points,
below a preregistered 15-point attribution threshold. That failure must remain
recorded. It means the experiment did not independently establish persistent
intention as the dominant causal factor.

It does **not** mean the teacher is weak:

- search exceeded mixed PPO by 20.0 percentage points;
- search passed the absolute performance, seat, medium-profile, cycling, and
  planner-win/PPO-draw clauses;
- terminal-only search itself strongly outperformed PPO.

The original scientific sequence required an explicit reconciliation before
imitation. This document supplies it: the user's current objective is the
fastest defensible path to a winning *learned* model, not isolated attribution
of the pursuit heuristic. Material teacher outperformance is sufficient to test
whether policy initialization is the bottleneck.

The 99 locked disagreements justify the teacher but remain evaluation evidence.
They must never enter the demonstration dataset.

---

## 4. HexWars as a reinforcement-learning problem

### 4.1 Episodic Markov decision process

An episodic Markov decision process (MDP) is a tuple

$$
\mathcal{M} = (\mathcal{S}, \mathcal{A}, P, R, \gamma, \rho_0),
$$

where:

- $\mathcal{S}$ is the set of authoritative game states;
- $\mathcal{A}$ is the action set;
- $P(s' \mid s,a)$ is the transition model implemented by the engine;
- $R(s,a,s')$ is the scalar reward;
- $\gamma \in [0,1]$ is the discount factor;
- $\rho_0$ is the seeded initial-state distribution.

At time $t$, the learner receives an observation $o_t = \phi(s_t,p)$ from the
perspective of seat $p$, selects action $a_t$, receives reward $r_t$, and
transitions to $s_{t+1}$.

The engine state and policy observation are not interchangeable. The engine
contains everything needed to enforce rules and reconstruct replay. The policy
receives only the versioned observation contract. Evaluation-only diagnostics
may inspect full state, but they must not leak into observation or reward.

### 4.2 Current tactical-v2 observation

For $T$ templates and $N$ board cells, tactical-v2 emits a flat vector that
conceptually contains:

- $T$ friendly template-specific hit-point planes;
- $T$ enemy template-specific hit-point planes;
- one elevation plane;
- five scalar globals: friendly points, enemy points, normalized round,
  friendly alive fraction, and enemy alive fraction.

The Python HexCNN reshapes the board portion to channels-by-height-by-width,
applies two 3-by-3 convolutional layers with 32 and 64 channels, flattens the
result, concatenates the five globals, and produces a 256-dimensional feature
vector for policy and value processing.

This is spatial, but fixed-size. It is appropriate for the tactical-v2 milestone
and intentionally not the long-term variable-board tactical-v3 representation.

### 4.3 Current tactical-v2 action space

The flattened action space contains four regions:

1. EndTurn;
2. move: controllable-unit slot by destination cell;
3. attack: controllable-unit slot by target cell;
4. deploy: template index by destination cell.

The slot registry gives stable identity to a bounded number of living units.
Many flattened indices are illegal in a particular state. The engine therefore
emits a Boolean legal-action mask $m_t$.

Define the legal set

$$
\mathcal{A}(o_t,m_t) = \{a \in \mathcal{A} : m_t(a)=1\}.
$$

If the actor produces logits $z_\theta(o_t,a)$, the masked policy is

$$
\pi_\theta(a \mid o_t,m_t)
=
\frac{m_t(a)\exp z_\theta(o_t,a)}
{\sum_b m_t(b)\exp z_\theta(o_t,b)}.
$$

Illegal actions receive zero probability. The mask is part of action selection
and log-probability computation; it is not an optional postprocessing filter.

### 4.4 Return, value, action value, and advantage

The discounted return from time $t$ is

$$
G_t = \sum_{k=0}^{T-t-1}\gamma^k r_{t+k}.
$$

Three related functions answer different questions:

$$
V^\pi(o_t) = \mathbb{E}_\pi[G_t \mid o_t],
$$

$$
Q^\pi(o_t,a_t) = \mathbb{E}_\pi[G_t \mid o_t,a_t],
$$

$$
A^\pi(o_t,a_t) = Q^\pi(o_t,a_t)-V^\pi(o_t).
$$

$V^\pi$ asks how promising the current observation is under the policy.
$Q^\pi$ asks how promising it is to choose a particular action and then follow
the policy. $A^\pi$ asks whether that action is better or worse than the
policy's usual choice in the same observation.

This formalizes the earlier "threat versus opportunity" intuition. Destroying
two weak units may have positive immediate material value, but its $Q$ value can
still be low if the successor exposes the acting force to decisive artillery.
The comparison belongs in expected return, not in a hardcoded artillery rule.

### 4.5 Reward held fixed for this project

The profiled annihilation experiments use:

- terminal win: $+1$;
- terminal loss: $-1$;
- terminal draw credit: $0$;
- value-advantage delta scale: $0.01$;
- per-command step penalty: $0.005$;
- closing-distance weight: $0$;
- banked-points weight in position value: $0.5$.

Ignoring terminal reward, the learner step reward is

$$
r_t^{shape}
= 0.01\left(A_{t+1}^{material}-A_t^{material}\right)-0.005.
$$

The current position-value calculation counts living units at full point cost
and banked points at half weight. This is an acknowledged limitation, but it is
held fixed so that imitation initialization is the experimental intervention.

---

## 5. Policy-gradient and actor-critic background

### 5.1 The policy-gradient idea

A stochastic policy $\pi_\theta(a\mid o,m)$ is a differentiable probability
distribution. The objective is expected discounted return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}\left[\sum_t\gamma^t r_t\right].
$$

The policy-gradient theorem leads to an estimator of the form

$$
\nabla_\theta J(\theta)
\approx
\frac{1}{B}\sum_t
\nabla_\theta\log\pi_\theta(a_t\mid o_t,m_t)\hat A_t.
$$

If $\hat A_t>0$, gradient ascent increases the probability of $a_t$. If
$\hat A_t<0$, it decreases the probability. The action mask changes the
normalized distribution and therefore must be present when computing
$\log\pi_\theta$.

### 5.2 Why a critic is useful

Raw Monte Carlo returns have high variance. An actor-critic method learns:

- an actor $\pi_\theta$, which selects actions;
- a critic $V_\psi$, which predicts expected return.

The one-step temporal-difference residual is

$$
\delta_t = r_t+\gamma V_\psi(o_{t+1})-V_\psi(o_t).
$$

Generalized Advantage Estimation (GAE) forms

$$
\hat A_t^{GAE(\gamma,\lambda)}
=
\sum_{l=0}^{T-t-1}(\gamma\lambda)^l\delta_{t+l}.
$$

$\lambda$ controls a bias-variance tradeoff. Values near one use longer return
information and usually reduce bias while increasing variance.

### 5.3 PPO's clipped objective

PPO collects trajectories under an old policy $\pi_{\theta_{old}}$ and reuses
them for several minibatch epochs. Define the importance ratio

$$
r_t(\theta)
=
\frac{\pi_\theta(a_t\mid o_t,m_t)}
{\pi_{\theta_{old}}(a_t\mid o_t,m_t)}.
$$

The clipped surrogate objective is

$$
L^{clip}(\theta)
=
\mathbb{E}_t
\left[
\min\left(
r_t(\theta)\hat A_t,
\operatorname{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
$$

The clipping term limits the incentive for a single batch to move the policy too
far. A practical actor-critic loss also includes a value regression term and an
entropy term:

$$
L
=
-L^{clip}
+c_v\mathbb{E}_t[(V_\psi(o_t)-\hat G_t)^2]
-c_H\mathbb{E}_t[\mathcal{H}(\pi_\theta(\cdot\mid o_t,m_t))].
$$

HexWars uses MaskablePPO so that collection, entropy, action probability, and
importance ratios are all defined over legal actions. The incumbent experiment
uses learning rate $3\times10^{-4}$, ten nominal epochs, and target KL $0.02$.
Target KL provides an additional early-stop signal when the new policy moves too
far from the rollout policy.

### 5.4 Why PPO can fail to discover annihilation

PPO receives an outcome signal only after a sequence of decisions. In a long
game, pursuit behavior must be sampled often enough, survive clipping and value
estimation noise, and remain useful across many maps. If a locally reasonable
reactive policy repeatedly avoids catastrophe, it may gather many trajectories
that end in zero-credit draws without discovering the coordinated sequence that
converts them.

More late-game starts shorten the credit-assignment path, but the curriculum
experiment shows that this alone did not reliably cross the discovery barrier.

---

## 6. Behavioral cloning and sequential imitation

### 6.1 Behavioral cloning as supervised learning

Let a teacher dataset be

$$
\mathcal{D}
=
\{(o_i,m_i,a_i^*,c_i)\}_{i=1}^N,
$$

where $c_i$ contains provenance such as seed, seat, profile, teacher identity,
contract hash, and replay.

Behavioral cloning minimizes masked negative log likelihood:

$$
L_{BC}(\theta)
=
-\frac{1}{N}\sum_{i=1}^N
\log\pi_\theta(a_i^*\mid o_i,m_i).
$$

This is ordinary multiclass cross-entropy over the legal action set. It supplies
a learning signal at every teacher decision rather than waiting for a terminal
win.

### 6.2 Why the critic is not cloned

Teacher actions label *what the teacher selected*. They do not label the
expected return of the learner's future policy. A value head trained as if
teacher choice implied a value target would be conceptually wrong.

The transfer boundary therefore copies:

- the HexCNN feature extractor;
- policy-side latent layers;
- the action-logit head.

It reinitializes:

- value-side latent layers and value head;
- PPO optimizer state;
- rollout and advantage state.

If the SB3 policy shares the feature extractor between actor and critic, the
fresh critic consumes cloned spatial features but begins with new value-specific
parameters.

### 6.3 Covariate shift

Supervised learning usually assumes train and test examples come from similar
distributions. Sequential policies violate this assumption. A small imitation
error changes the next state; that new state may never appear in teacher
trajectories; a second error moves the learner farther away.

If the teacher visits distribution $d_{\pi^*}$ but the learned policy induces
$d_{\pi_\theta}$, BC trains on one distribution and executes on another.

DAgger addresses this by:

1. rolling out the current learner;
2. asking the teacher to label states the learner actually visits;
3. aggregating those labels into the dataset;
4. retraining the policy.

DAgger is a contingency, not part of the first run. We add it only if action
accuracy is high on teacher data but closed-loop game performance remains poor.

### 6.4 Why BC is followed by PPO

Pure BC is limited by teacher quality and covariate shift. PPO fine-tuning uses
the true environment return and can:

- recover from learner-specific mistakes;
- improve actions for which teachers are suboptimal;
- trade local imitation accuracy for higher win probability;
- adapt the policy to its own state distribution.

The hybrid interpretation is:

```text
behavioral cloning supplies discovery and a useful prior;
PPO supplies outcome-based correction and potential improvement.
```

---

## 7. Teacher design

### 7.1 Greedy teacher

Greedy supplies complete standard 3-v-3 trajectories. It provides examples of:

- deployment;
- ordinary movement and attack;
- target selection;
- early- and mid-game combat;
- decisions from the same full-game start distribution used by the milestone.

Greedy wins only about half of the standard evaluation games, so it is not a
sufficient final policy. Its purpose is broad state coverage and a better
starting policy than random neural initialization.

### 7.2 Bounded-search teacher

Bounded search supplies high-quality conversion behavior. It enumerates
authoritative legal commands and applies exact engine transitions under a fixed
depth-four, 512-expansion budget. Terminal outcomes have primary value. A
bounded nonterminal heuristic adds health-sensitive material and progress toward
a persistent target.

The teacher is used on conversion profiles because their branching factor is
bounded enough for inexpensive data generation. Search is not embedded in the
learned policy and is not a runtime requirement after training.

### 7.3 Information boundary

The profiled tactical-v2 milestone has fog of war disabled. The search teacher's
authoritative state therefore does not contain hidden enemy information absent
from the learner observation. If fog is introduced later, demonstrations must be
generated by a seat-filtered teacher; otherwise identical observations could
receive inconsistent labels based on hidden state.

---

## 8. Proposed method

### 8.1 End-to-end flow

```mermaid
flowchart LR
  A[Seeded teacher games] --> B[Authoritative pre-action capture]
  B --> C[Validated demonstration shards]
  C --> D[Masked behavioral cloning]
  D --> E[Pure-clone evaluation]
  E --> F[Transfer actor into fresh PPO]
  F --> G[Standard-heavy PPO fine-tuning]
  G --> H[Development checkpoint selection]
  H --> I[Sealed reciprocal milestone evaluation]
```

### 8.2 Demonstration composition

Collection stops after completing the reciprocal pair that reaches at least:

- 100,000 Greedy decisions from standard 3-v-3 games;
- 50,000 bounded-search decisions from conversion profiles.

Imitation minibatches are sampled as:

- 70 percent standard Greedy rows;
- 30 percent conversion search rows.

The sampling ratio, rather than the raw file size, defines training exposure.

### 8.3 Seed namespaces

The new manifest reserves:

| Purpose | Namespace |
|---|---:|
| Greedy standard demonstrations | 11,000,000-11,499,999 |
| Search conversion demonstrations | 11,500,000-11,999,999 |
| BC validation games | 12,000,000-12,099,999 |
| PPO training replicate 1 episodes | 13,000,000-13,999,999 |
| PPO training replicate 2 episodes | 14,000,000-14,999,999 |
| PPO training replicate 3 episodes | 15,000,000-15,999,999 |
| Development evaluation maps | 16,000,000-16,000,099 |
| Final evaluation maps | 17,000,000-17,000,249 |

Model initialization seeds are 211, 223, and 227. The runner must expose model
seed and episode-seed base independently; a model seed must not silently
determine an overlapping incrementing map sequence.

The previously reserved 10,000,000 confirmation namespace remains untouched.

### 8.4 Dataset split

Rows are never randomly split across transitions. Entire map seeds belong to one
partition. This prevents adjacent states from one game appearing in both
training and validation.

The dataset consists of:

- compressed numeric shards containing float32 observations, packed masks, and
  integer actions;
- `games.jsonl` containing game and replay provenance;
- `manifest.json` containing schema, code revision, controller identities,
  contract and encoding hashes, seed ranges, counts, and file hashes.

### 8.5 BC training

Each of three BC runs uses the same frozen dataset but a different initialization
and minibatch order. Training reports:

- masked cross-entropy;
- legal top-1 and top-k action accuracy;
- accuracy by teacher, profile, action kind, and seat;
- predicted EndTurn probability;
- calibration diagnostics;
- checkpoint and manifest hashes.

Action accuracy is diagnostic. The pure-clone game gate is authoritative.

### 8.6 Pure-clone gate

Each clone is evaluated on 100 development maps from both seats, or 200 games.
Proceed to PPO if:

- pooled standard win rate is at least 40 percent;
- no clone's standard win rate is below 30 percent;
- all action, mask, replay, and contract checks pass.

Conversion win rate is reported but does not replace the standard-game gate.

Failure means the imitation pipeline has not transferred teacher behavior well
enough to justify PPO compute.

### 8.7 PPO fine-tuning

The new fine-tuning scenario shares the profiled tactical-v2 contract and uses:

- 70 percent standard 3-v-3 starts;
- 30 percent conversion starts, split equally among the six trained near/far
  profiles;
- medium-separation profiles declared but assigned zero training weight;
- Random opponent;
- unchanged annihilation reward;
- learning rate $3\times10^{-4}$;
- ten nominal PPO epochs;
- target KL $0.02$.

Evaluate rollout-aligned checkpoints near:

- 12,800 environment steps;
- 25,600 environment steps;
- 51,200 environment steps.

The exact saved step is the first completed rollout at or beyond each nominal
budget and is recorded explicitly.

### 8.8 Control condition

For causal attribution, train three from-scratch MaskablePPO controls under the
same 70/30 scenario, code revision, budgets, model seeds, and episode-seed
namespaces. The controls do not delay use of an already-passing learned model,
but they are required for the course-project conclusion.



### 8.9 Model-based, offline, and on-policy roles

Three modes of learning or decision-making appear in this project:

| Component | Uses engine transitions while deciding? | Learns from a fixed dataset? | Learns from its own rollouts? |
|---|---:|---:|---:|
| Bounded-search teacher | Yes | No | No |
| Behavioral-cloning actor | No | Yes | No |
| PPO actor-critic | No at inference | No | Yes |

Bounded search is **model-based** because it calls the known transition model
while choosing an action. The cloned neural actor is **model-free at
inference**: one forward pass maps observation and mask to action probabilities.
BC is **offline supervised learning** over a fixed teacher dataset. PPO is
**on-policy reinforcement learning** because its updates use trajectories
collected by the current or very recent policy.

The distinction explains why search cost does not become deployment cost. Search
spends computation once to produce labels; the student compresses those choices
into network parameters.

### 8.10 Algorithm walkthrough

The primary pipeline can be written as the following high-level algorithm.

```text
Inputs:
  authoritative environment E
  Greedy teacher g
  bounded-search teacher b
  disjoint seed banks
  BC/PPO replicate seeds {211, 223, 227}

1. Collect demonstrations
   D_standard   <- actions selected by g on standard 3-v-3 training seeds
   D_conversion <- actions selected by b on conversion training seeds
   validate every (observation, legal mask, action, replay) row

2. For each replicate seed j
   initialize actor parameters theta_j

   repeat for BC epochs
     sample a minibatch with 70% D_standard and 30% D_conversion
     apply each legal mask to the action logits
     L_BC <- mean negative log probability of the teacher actions
     theta_j <- optimizer_step(theta_j, gradient(L_BC))

   evaluate the pure clone on development maps

   construct a fresh MaskablePPO actor-critic
   copy theta_j into the encoder and actor
   initialize fresh value-specific parameters and optimizer state

   repeat until 51,200 environment steps or an earlier stopping decision
     collect legal masked trajectories against Random
     compute returns and generalized advantages
     optimize PPO clipped policy, value, and entropy objectives
     evaluate rollout-aligned checkpoints on development maps

3. Select one global checkpoint budget across all three replicates
4. Freeze configuration and evaluate once on the final reciprocal bank
5. Apply the 65%-per-seed and 70%-pooled annihilation-win gate
```

During BC, a single update follows the familiar supervised-learning sequence:

1. the HexCNN produces a feature vector;
2. the actor produces one logit per flattened action;
3. illegal logits are replaced by a value representing negative infinity;
4. softmax produces a categorical distribution over legal actions;
5. negative log probability of the teacher action becomes the loss;
6. backpropagation moves probability mass toward the teacher action.

During PPO, the actor generates its own data. The critic estimates returns, GAE
turns temporal-difference residuals into lower-variance advantage estimates, and
the clipped ratio limits how far each batch can move the policy. Thus BC and PPO
change the same actor through different evidence:

$$
\text{BC evidence}=\text{teacher chose this legal action},
$$

$$
\text{PPO evidence}=\text{this sampled action produced better or worse return
than expected}.
$$

Students should notice that neither statement says an action is intrinsically
good. BC quality depends on the teacher and state distribution. PPO quality
depends on exploration, credit assignment, value estimation, and the reward
contract.

---


## 9. System and interface design

### 9.1 Engine command encoder

`TacticalV2Coding` already has the authoritative private encoder used to
construct the legal mask. Implementation exposes that logic through one narrow,
tested interface rather than creating a second mapping.

Required invariants for every legal action $a$ and command $c$:

$$
\operatorname{Encode}(\operatorname{Decode}(a))=a,
$$

$$
\operatorname{Decode}(\operatorname{Encode}(c))=c.
$$

Equality is semantic: issuer, unit identity, target identity or destination, and
template identity must match.

### 9.2 Pre-action demonstration capture

When a scripted controller selects a command, the recorder captures before
applying it:

| Field | Meaning |
|---|---|
| observation | Exact policy-facing float vector |
| legal_mask | Exact policy-facing legal-action mask |
| action | Authoritative encoded command index |
| command | Structured command for audit |
| teacher | Greedy or bounded-search identity and parameters |
| seed | Map/start seed |
| seat | Acting seat |
| start_profile | Standard or conversion profile |
| decision_index | Monotone index within game |
| contract_hash | Observation/action/reward semantics identity |
| encoding_hash | Tensor geometry identity |
| scenario_hash | Training-distribution provenance |
| replay_path | Authoritative replay |

The recorder is research infrastructure. Enabling it must not alter action
selection, transition order, reward, or terminal outcome.

### 9.3 Python dataset boundary

`python/ml_lab/imitation.py` owns:

- manifest and shard DTOs;
- validation and deterministic loading;
- 70/30 stratified minibatch sampling;
- masked cross-entropy;
- BC checkpoint writing;
- actor transfer into MaskablePPO.

`python/collect_annihilation_demonstrations.py` owns orchestration but not
game semantics. It requests seeded games and persists validated rows.

### 9.4 Actor-transfer boundary

The algorithm adapter gains one explicit method that:

1. constructs a fresh MaskablePPO model under the target contract;
2. validates parameter names and tensor shapes;
3. copies the HexCNN and policy parameters;
4. reinitializes value-specific parameters;
5. creates a new optimizer;
6. verifies cloned logits on frozen fixtures;
7. writes transfer provenance.

Missing, extra, or shape-mismatched parameters are hard errors. Partial silent
loading is forbidden.

### 9.5 Panel runner

The experiment runner is restart-safe and validates:

- exact code revision and dirty-state policy;
- scenario, contract, encoding, dataset, and checkpoint hashes;
- disjoint seed namespaces;
- complete reciprocal schedules;
- rollout-aligned checkpoint identity;
- final-bank single-use rules;
- evidence publication as an atomic transaction.

---

## 10. Evaluation protocol

### 10.1 Development selection

At clone initialization and each PPO budget, evaluate every seed on the same 100
development maps from both seats. Select one global budget for all three seeds
using pooled standard-game win rate. Do not select a different best step for each
seed.

Tie-breakers, in order:

1. higher worst-seed standard win rate;
2. higher pooled conversion win rate;
3. lower pooled draw rate;
4. earlier checkpoint.

### 10.2 Final milestone

After the configuration and global checkpoint budget are frozen, evaluate each
model on 250 previously unused maps from both seats:

$$
250\text{ maps}\times2\text{ seats}=500\text{ games per model}.
$$

The milestone passes only if:

- seed 211 wins at least 65 percent;
- seed 223 wins at least 65 percent;
- seed 227 wins at least 65 percent;
- pooled wins across 1,500 games are at least 70 percent.

Every draw and loss is a non-win. No material or draw-severity credit changes the
gate.

### 10.3 Statistical reporting

Report:

- W/L/D counts and rates;
- Wilson 95 percent intervals for win, loss, and draw rates;
- seat-specific rates;
- paired map/seat comparisons with current PPO and from-scratch PPO;
- exact sign tests on discordant paired outcomes;
- per-template-composition and map summaries;
- draw categories, rounds, decisions, action waste, and peak material advantage.

The preregistered primary statistic is annihilation win rate against Random.
Secondary metrics explain failures and must not be substituted after observing
the primary result.

### 10.4 Evidence

Retain:

- every draw replay and trace;
- every loss replay and trace;
- stratified winning controls;
- pure-clone and PPO checkpoint identities;
- decomposed reward and action statistics;
- BC validation predictions;
- final aggregate and human-readable report.

---

## 11. Hypotheses and interpretation

### H1: Imitation solves behavioral discovery

Prediction: pure clones materially outperform current PPO on standard games and
show far fewer conversion cycles.

Evidence for H1: pure-clone game win rate, not merely teacher-action accuracy.

### H2: PPO improves or preserves the cloned policy

Prediction: BC-to-PPO reaches the 65/70 milestone and exceeds both pure BC and
from-scratch PPO.

Evidence for H2: paired reciprocal outcomes under equal online environment steps.

### H3: Closed-loop distribution shift is the remaining failure

Pattern: validation action accuracy is high, but pure-clone win rate is low or
errors compound after the first deviation.

Response: generate a DAgger-style dataset by rolling out the learner and
obtaining teacher labels on learner-visited training states.

### H4: PPO catastrophically erases teacher behavior

Pattern: the pure clone plays well, but later PPO checkpoints lose win rate,
increase cycling, or sharply increase EndTurn probability.

Response: first select the earlier global checkpoint. If no PPO checkpoint
preserves the clone, design a separate, preregistered auxiliary-imitation
condition rather than tuning coefficients against the final bank.

### H5: The tactical-v2 representation is the bottleneck

Pattern: BC cannot reproduce a demonstrably strong teacher despite correct
actions, masks, and sufficient data; or BC/PPO remain reactive and cycle after
DAgger.

Response: stop tuning tactical-v2 and advance tactical-v3 legal-candidate,
afterstate, and persistent-intention representations.

---

## 12. Verification and testing

### 12.1 Engine tests

- legal command encode/decode bijection;
- illegal and unrepresentable commands fail closed;
- observation and mask captured before the selected action;
- captured action is legal and reproduces the teacher command;
- recorder on/off produces identical commands, states, rewards, and outcomes;
- worker count does not change deterministic teacher games;
- replay reconstructs every recorded terminal state.

### 12.2 Dataset tests

- schema and hash validation;
- no seed appears in more than one partition;
- locked evaluation namespaces are rejected;
- no invalid or masked action is admitted;
- shards are deterministic and content-hashed;
- stratified sampler produces the declared 70/30 exposure;
- interrupted collection resumes without duplicated games or rows.

### 12.3 Learning tests

- masked probabilities sum to one over legal actions;
- illegal actions have zero probability and zero sampling frequency;
- a small synthetic dataset can be overfit;
- save/reload preserves logits;
- actor transfer preserves frozen-fixture logits;
- value-specific parameters and optimizer state are fresh;
- BC and PPO runs are reproducible for fixed seeds within documented numerical
  tolerance.

### 12.4 End-to-end smoke

Before full collection:

1. collect one reciprocal Greedy standard pair;
2. collect one reciprocal search pair for every conversion profile;
3. train BC on a tiny shard;
4. transfer the actor into PPO;
5. run one rollout and checkpoint;
6. reload and evaluate;
7. verify replay and evidence publication.

---

## 13. Risks and limitations

### 13.1 Teacher ceiling

Greedy standard play wins only about half its games. Search is strongest on
reduced conversion profiles. The combined dataset may teach good components
without teaching a complete 70-percent standard policy. PPO is expected to
compose and improve those components, but that is a hypothesis.

### 13.2 Conflicting labels

Greedy and search may select different actions in similar observations. Teacher
identity is provenance, not an observation feature. The fixed 70/30 sampler
defines the aggregate target. Future work may use quality-weighted or
cost-to-go labels, but the first condition uses plain masked cross-entropy.

### 13.3 Action aliases and EndTurn fallback

The current decoder maps invalid indices to EndTurn, which could hide a broken
demonstration pipeline. Dataset capture uses only authoritative encoded legal
commands and rejects any round-trip mismatch. It never infers a label by
decoding arbitrary indices.

### 13.4 Evaluation leakage

The strongest qualitative evidence is tempting training material. Using the 99
locked disagreements would invalidate the experiment. They remain sealed; new
training-only seeds generate analogous examples.

### 13.5 Fixed-size milestone

Success proves a learned policy can consistently beat Random under one
tactical-v2 standard contract. It does not establish board-size, roster,
mechanic, or unit-design generalization. Those belong to tactical-v3.

### 13.6 Compute comparison

Teacher search compute is used only to create offline labels. Runtime inference
cost for the learned policy remains neural inference. Reports separately state
teacher generation cost, BC optimizer cost, PPO environment steps, PPO updates,
wall-clock time, and final inference cost.

---

## 14. Deliverables

The project produces:

```text
docs/superpowers/specs/
  2026-07-29-imitation-initialized-ppo-winning-model-design.md

python/datasets/annihilation-imitation-v1/
  manifest.json
  games.jsonl
  shards/

python/panels/annihilation-imitation-v1/
  PROTOCOL.md
  panel.json
  seed-banks.json
  aggregate.json
  REPORT.md
  evaluations/
  evidence/

python/ml_lab/imitation.py
python/collect_annihilation_demonstrations.py
python/run_annihilation_imitation_panel.py
python/tests/test_imitation.py
```

Engine and protocol changes add authoritative demonstration capture and tests.

---

## 15. Questions for students

1. Why can a policy have high behavioral-cloning action accuracy but low win
   rate?
2. Why must the legal mask be applied before softmax rather than after sampling?
3. Which parts of the state-to-action mapping can BC learn without any reward?
4. Why does the demonstration dataset not provide a trustworthy initial value
   head?
5. What evidence distinguishes failure to discover pursuit from failure to
   represent pursuit?
6. Why is splitting individual transitions randomly weaker than splitting by
   complete game seed?
7. What causal claim does the from-scratch PPO control permit?
8. Why is selecting a separate best checkpoint for each training seed less
   convincing than selecting one global budget?
9. If BC-to-PPO wins 75 percent against Random but fails on a larger board, which
   project claim is falsified and which remains valid?
10. How would fog of war change the teacher information boundary?

---

## 16. Repository grounding

Relevant authoritative or design sources include:

- `engine/HexWars.Engine/Rl/TacticalV2Coding.cs`: observation, mask, command
  encoder, and action decoder;
- `engine/HexWars.Engine/Rl/TacticalV2Env.cs`: learner transition and reward
  timing;
- `engine/HexWars.Engine/Rl/RewardShaping.cs`: material advantage and draw
  helpers;
- `python/hex_cnn.py`: current spatial feature extractor;
- `python/ml_lab/algorithms.py`: MaskablePPO construction and prediction;
- `python/ml_lab/evaluation.py`: reciprocal evaluation and evidence;
- `python/panels/annihilation-conversion-curriculum-v1/REPORT.md`:
  curriculum evidence;
- `python/panels/annihilation-bounded-search-v1/REPORT.md`: teacher
  positive-control evidence;
- `docs/superpowers/specs/2026-07-28-tactical-v3-general-rl-design.md`:
  long-term architecture boundary.

As of the design date, the profiled-start and bounded-search substrate exists in
the research workspace but is not yet cleanly integrated with the merged
evidence branch. Preserving and integrating that substrate is Phase 0 of the
implementation plan.

---

## 17. References

1. Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. (2017).
   [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

2. Schulman, J., Moritz, P., Levine, S., Jordan, M., and Abbeel, P. (2015).
   [High-Dimensional Continuous Control Using Generalized Advantage
   Estimation](https://arxiv.org/abs/1506.02438).

3. Ross, S., Gordon, G., and Bagnell, D. (2011).
   [A Reduction of Imitation Learning and Structured Prediction to No-Regret
   Online Learning](https://proceedings.mlr.press/v15/ross11a.html).

4. Huang, S., and Ontanon, S. (2020).
   [A Closer Look at Invalid Action Masking in Policy Gradient
   Algorithms](https://arxiv.org/abs/2006.14171).

5. Sutton, R. S., and Barto, A. G. (2018). [*Reinforcement Learning: An
   Introduction*, second edition](https://mitpress.mit.edu/9780262039246/reinforcement-learning/).
   MIT Press.

---

## 18. Design decision

For the first winning-model milestone, implement behavioral cloning from Greedy
standard-game and bounded-search conversion demonstrations, then fine-tune the
cloned actor with MaskablePPO under a fixed standard-heavy curriculum.

Do not respond to a failed run by silently tuning reward, demonstration mixture,
PPO algorithm, opponent, or final seed bank. Follow the preregistered failure
branches, preserve negative results, and keep the primary question simple:

> Does supplying demonstrably effective actions before on-policy training
> produce a learned policy that consistently annihilates Random?
