# Annihilation learning-path decision

**Date:** 2026-07-29

**Status:** Accepted

**Scope:** Tactical-v2 annihilation diagnostics and the next research architecture; not production-AI promotion

## Decision

Stop tuning tactical-v2 reward weights, curriculum weights, PPO duration, or PPO
optimizer settings. Advance to a tactical-v3 research tranche centered on
authoritative legal-action candidates, afterstate scoring, and a small persistent
intention/target representation.

Keep target-KL MaskablePPO only as the provisional optimizer around that new
representation. This does not confirm PPO as the best algorithm; it avoids
changing optimizer and representation simultaneously when the evidence points
primarily at reactive action selection and short-horizon representation.

Do not consume the 10,000,000-series confirmation bank. Do not begin Greedy
opponent training, self-play, DQN parity work, imitation bootstrap, or a larger
AlphaZero-style search system in this tranche.

## Evidence

The curriculum panel trained six fresh 51,200-step MaskablePPO models, three on
profiled standard starts and three on a mixed standard/conversion curriculum. It
then ran 2,240 locked games, including reciprocal standard and nine-profile
conversion suites plus Greedy and Random controls. Draws received no win credit.

Mixed training preserved standard play but failed its preregistered curriculum
gate:

- conversion wins moved from 381/540 (70.6%) to 396/540 (73.3%), a 2.78-point gain rather than 15 points;
- per-seed conversion changes were +15, +13, and -13 wins;
- cycling draws moved only from 123 to 119 rather than falling 25 percent;
- standard wins improved from 61/300 to 68/300, and conversion losses fell from 36 to 25.

The bounded-search panel then ran 360 locked games over the same nine conversion
profiles and reciprocal seats, retaining every trace, replay, expansion count,
and top-branch diagnostic. Its composite gate remained a FAIL because the
persistent-pursuit heuristic improved over terminal-only search by 7.8 points,
below a locked 15-point attribution threshold. The performance evidence was
nevertheless decisive:

| Controller | W-L-D | Win rate | Cycling draws |
|---|---:|---:|---:|
| Bounded search | 168-4-8 | 93.3% | 8 |
| Terminal-only bounded search | 154-2-24 | 85.6% | 24 |
| Greedy | 162-11-7 | 90.0% | 7 |
| Mixed PPO, three-seed aggregate | 396-25-119 | 73.3% | 119 |

Bounded search passed every locked performance clause: aggregate wins, a
20-point improvement over mixed PPO, cycling at 4.4%, both seats above 92%, and
held-out medium profiles at 88.3%. It retained 99 planner-win/PPO-draw
disagreements. Against the three PPO seeds, paired planner-only wins versus
PPO-only wins were 31-4, 39-1, and 46-3. The exact sign-test results were all
strongly inconsistent with equal paired win behavior.

The pursuit heuristic's increment was smaller but real within this panel:
bounded search had 17 win-only pairs versus three for terminal-only search
(`p=0.0026`). Terminal-only search already winning 85.6% means the non-learning
controller package accounts for most of the positive-control gap; pursuit value
chiefly reduced residual cycling. That package includes authoritative candidate
enumeration, depth-four terminal detection, fixed command ordering, and stable
tie-breaking. Because nonterminal branches in the ablation all tie at zero, the
panel does not isolate lookahead from ordering and tie-breaking.

Search numerically exceeded Greedy 168 to 162, but the paired win-only comparison
was 9-3 (`p=0.146`). This does not establish a reliable advantage over Greedy and
does not justify scaling search into a product architecture.

## Answers to the plan's decision questions

1. **Is annihilation conversion trainable under frequent valid finishing states?**
   Partly. PPO can learn substantial conversion competence, but the mixed
   curriculum did not reliably improve it across seeds or eliminate cycling.
2. **Does the skill transfer to ordinary 3-v-3 and held-out medium separation?**
   Mixed training did not harm ordinary 3-v-3, and held-out medium conversion
   improved modestly. The planner's 88.3% medium result proves those starts are
   mechanically solvable; it is not evidence that PPO learned the same transfer.
3. **What is the main remaining limitation?** Reactive action selection and
   short-horizon representation are the leading explanation. Mechanical
   feasibility is not the blocker. Persistent pursuit contributes, but this
   panel did not establish it as the dominant cause. The data do not isolate
   on-policy sample reuse as the problem.
4. **Does PPO remain the tactical-v3 baseline optimizer?** Provisionally, yes,
   solely to isolate the representation change. It has no confirmation or
   production status.
5. **What authorizes Greedy training or self-play?** Nothing in these panels.
   Greedy remains an evaluation control. Opponent escalation would confound the
   representation experiment and is deferred.

## Why imitation and DQN are deferred

The authority plan permits imitation when a demonstrator materially outperforms
from-scratch PPO, and the planner meets that broad condition. The locked planner
panel imposed a stricter automatic-entry rule that also required a 15-point
pursuit-heuristic increment; that composite rule failed. More importantly, the
terminal-only result implicates the candidate/afterstate controller package as
the larger effect, although it does not separately identify lookahead, ordering,
and tie-breaking. Building behavioral cloning now would test whether the current
reactive policy can mimic planner actions, while leaving the implicated
representation unchanged. Tactical-v3 is the more direct causal test.

DQN remains deferred because its current implementation lacks representation,
masking, checkpoint, replay-buffer, and resume parity. The evidence does not yet
name replay-based sample reuse as the bottleneck, so parity hardening would be an
expensive algorithm change without a live causal hypothesis.

## Next tranche gate

The tactical-v3 tranche must remain narrow and reversible:

- enumerate candidates from authoritative `LegalMoves.For` and derive afterstates only through `GameEngine.Apply`;
- preserve exact tactical-v2 rules, profile construction, reciprocal seat policy, and replayability;
- expose bounded afterstate features that include health-sensitive material and pursuit progress without copying the search controller into the policy;
- add an explicit, inspectable intention/target state whose reset, retarget, and invalidation semantics are tested;
- compare a tactical-v3 PPO condition against the locked tactical-v2 mixed PPO on development/training namespaces before assigning any new confirmation seeds;
- require gains across seeds, both seats, ordinary 3-v-3, and held-out medium profiles before confirmation;
- retain disagreement traces and account for environment steps, optimizer updates, wall time, and inference cost.

If tactical-v3 does not materially reduce the planner/PPO disagreement set, stop
and reassess the observation/action contract before admitting imitation or a new
learned algorithm.

## Limitations

The PPO training episode-seed ranges overlapped heavily across training seeds, so
the three models are independent optimizer/policy-initialization replicates but
not independent map-exposure replicates. The realized mixed exposure was 42.55%
standard and 57.45% conversion, with far starts overrepresented relative to the
nominal mix. This limitation was frozen before locked evaluation and weakens
claims about replicate independence; it does not explain away a 2.78-point,
seed-inconsistent curriculum effect or the planner's 20-point advantage.

All results here use the profiled 13x9 tactical-v2 setting with at most three
controllable units. They must not be numerically compared to the user's earlier
16x16, 12-unit runs as if they were the same task.
