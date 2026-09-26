# Annihilation bounded-search v1

> Evaluation-only diagnostic ceiling; not a trainable controller or production-AI promotion.

**Positive-control gate:** FAIL

The locked composite gate failed solely because the pursuit heuristic improved win rate over terminal-only search by 7.8%, below the preregistered 15-point attribution threshold. That FAIL is preserved. It is not evidence that bounded lookahead failed: the planner passed every performance clause, won 93.3 percent, and exceeded mixed PPO by 20.0 points.

## Outcomes

| Controller | W-L-D | Win rate | Cycling draws |
|---|---:|---:|---:|
| Bounded search | 168-4-8 | 93.3% | 8 |
| Terminal-only ablation | 154-2-24 | 85.6% | 24 |
| Greedy | 162-11-7 | 90.0% | 7 |
| Mixed PPO, three-seed aggregate | 396-25-119 | 73.3% | 119 |

## Gate clauses

- PASS — aggregate win rate
- PASS — win-rate improvement over mixed PPO
- PASS — cycling draw incidence
- PASS — win rate in each candidate seat
- PASS — held-out medium win rate
- FAIL — improvement over terminal-only search
- PASS — retained planner-win/PPO-draw disagreements: 99

## Conversion profiles

| Profile | Search W-L-D | Terminal-only W-L-D | Greedy W-L-D |
|---|---:|---:|---:|
| conversion-1v1-far | 18-0-2 | 11-0-9 | 15-3-2 |
| conversion-1v1-medium | 13-3-4 | 10-1-9 | 13-7-0 |
| conversion-1v1-near | 18-1-1 | 17-1-2 | 19-1-0 |
| conversion-2v1-far | 19-0-1 | 18-0-2 | 17-0-3 |
| conversion-2v1-medium | 20-0-0 | 20-0-0 | 19-0-1 |
| conversion-2v1-near | 20-0-0 | 18-0-2 | 20-0-0 |
| conversion-3v1-far | 20-0-0 | 20-0-0 | 20-0-0 |
| conversion-3v1-medium | 20-0-0 | 20-0-0 | 19-0-1 |
| conversion-3v1-near | 20-0-0 | 20-0-0 | 20-0-0 |

## Search compute

The planner made 2,612 decisions and expanded 1,008,170 authoritative transitions in 23.55 search-seconds. Mean cost was 386.0 expansions and 9.01 ms per decision.

## Attribution and Greedy comparison

Search had 17 win-only pairs versus 3 for terminal-only search (exact sign p=0.0026). The increment is real in this panel, but smaller than the locked 15-point magnitude requirement. The strong 85.6 percent terminal-only result says the non-learning controller package accounts for most of the positive-control gap: authoritative candidate enumeration, depth-four terminal detection, fixed command ordering, and stable tie-breaking. Because nonterminal branches in that ablation all tie at zero, this panel does not isolate lookahead from ordering and tie-breaking. The persistent-pursuit value mainly reduces residual cycling.

Search had 9 win-only pairs versus 3 for Greedy (exact sign p=0.1460). Its numerical 168-to-162 edge does not establish a reliable gain over Greedy and does not justify a large AlphaZero-style system.

## Decision

Explicit bounded lookahead is a positive control for states where PPO cycles. Stop tactical-v2 reward/optimizer tuning and prioritize tactical-v3 candidate/afterstate scoring. The pursuit heuristic helps, but its incremental effect missed the locked magnitude threshold, so this panel does not independently establish persistent intention as the dominant cause.

Under this panel's stricter locked composite rule, imitation is not automatically authorized. The authority plan's broader material-outperformance condition is satisfied, so proceeding to imitation requires an explicit reconciliation of those two rules rather than relabeling this gate.

The 10,000,000-series confirmation namespace remains unassigned and unconsumed.
