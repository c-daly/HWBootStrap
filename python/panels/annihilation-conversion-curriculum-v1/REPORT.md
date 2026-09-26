# Annihilation Conversion Curriculum v1

> Diagnostic curriculum decision panel; not a tactical-v3 official tournament or production-AI promotion.

**Curriculum gate:** FAIL

The mixed curriculum did not produce the preregistered effect. Conversion wins rose from 381/540 (70.6%) to 396/540 (73.3%), only 2.78% in absolute rate. Cycling draws fell from 123 to 119, while the gate required a 25% relative reduction.

This is a useful negative result rather than a null run: mixed training preserved ordinary 3-v-3 play, improved held-out medium conversion, reduced losses, and moved both seats in the favorable direction. But the aggregate gain was small, seed 127 regressed, and cycling remained the dominant failure. The plan therefore authorizes the bounded-search diagnostic; it does not authorize weight tuning, 100,352-step confirmation, Greedy training, self-play, or a DQN comparison.

## Gate clauses

- FAIL — aggregate conversion win rate improves by at least 15 percentage points
- FAIL — conversion wins improve for every training seed
- FAIL — cycling draw incidence falls by at least 25 percent relative
- PASS — held-out medium separation conversion improves
- PASS — standard win rate drop is at most 5 percentage points
- PASS — no seed loses more than 15 percentage points on standard
- PASS — conversion loss rate rises by at most 5 percentage points
- PASS — conversion effect appears in both candidate seats

## Aggregate outcomes

| Controller / condition | Standard W-L-D | Conversion W-L-D |
|---|---:|---:|
| Profiled standard PPO | 61-0-239 | 381-36-123 |
| Mixed conversion PPO | 68-0-232 | 396-25-119 |
| Greedy control | 48-0-52 | 162-11-7 |
| Random control | 0-0-100 | 105-25-50 |

## Per-seed outcomes

| Condition | Seed | Standard W-L-D | Conversion W-L-D |
|---|---:|---:|---:|
| profiled_standard | 101 | 16-0-84 | 126-12-42 |
| profiled_standard | 113 | 19-0-81 | 117-15-48 |
| profiled_standard | 127 | 26-0-74 | 138-9-33 |
| mixed_conversion | 101 | 20-0-80 | 141-8-31 |
| mixed_conversion | 113 | 24-0-76 | 130-10-40 |
| mixed_conversion | 127 | 24-0-76 | 125-7-48 |

## Paired conversion comparison

| Scope | Control-only wins | Mixed-only wins | Net mixed wins | Exact sign p |
|---|---:|---:|---:|---:|
| All seeds | 49 | 64 | +15 | 0.1876 |
| Seed 101 | 14 | 29 | +15 | 0.0315 |
| Seed 113 | 13 | 26 | +13 | 0.0533 |
| Seed 127 | 22 | 9 | -13 | 0.0294 |

## Conversion profiles

| Profile | Standard-trained W-L-D | Mixed-trained W-L-D | Net mixed wins |
|---|---:|---:|---:|
| conversion-3v1-near | 47-0-13 | 50-0-10 | +3 |
| conversion-3v1-medium | 55-0-5 | 54-0-6 | -1 |
| conversion-3v1-far | 55-1-4 | 52-0-8 | -3 |
| conversion-2v1-near | 47-1-12 | 46-0-14 | -1 |
| conversion-2v1-medium | 48-0-12 | 52-0-8 | +4 |
| conversion-2v1-far | 45-2-13 | 39-2-19 | -6 |
| conversion-1v1-near | 39-5-16 | 44-5-11 | +5 |
| conversion-1v1-medium | 19-22-19 | 23-14-23 | +4 |
| conversion-1v1-far | 26-5-29 | 36-4-20 | +10 |

## Training and compute

| Condition | Seed | Episodes | Seconds | Final KL | Final clip | Updates |
|---|---:|---:|---:|---:|---:|---:|
| profiled_standard | 101 | 168 | 129.0 | 0.0157 | 0.106 | 115 |
| profiled_standard | 113 | 168 | 129.9 | 0.0211 | 0.147 | 126 |
| profiled_standard | 127 | 171 | 128.8 | 0.0193 | 0.132 | 117 |
| mixed_conversion | 101 | 277 | 148.2 | 0.0217 | 0.171 | 116 |
| mixed_conversion | 113 | 260 | 148.1 | 0.0193 | 0.158 | 119 |
| mixed_conversion | 127 | 255 | 153.7 | 0.0227 | 0.062 | 123 |

## Training-schedule limitation

The three models remain independent optimizer and policy-initialization replicates under paired treatments, but they are not independent map-exposure replicates because their incrementing episode-seed ranges overlap heavily.

The realized mixed treatment is 42.55 percent standard and 57.45 percent conversion. Far profiles received 36.36 percent of episodes and near profiles 21.09 percent, rather than the nominal 30/30 split. Locked results must be reported against realized exposure and must not be used to retune weights.

This limitation was frozen before locked evaluation. It does not explain away the failed gate: the realized treatment still supplied substantially more valid conversion practice, yet produced only a small and inconsistent gain.

## Decision

Do not consume the 10,000,000-series confirmation bank and do not tune curriculum weights against these results. Proceed to the bounded-search baseline from Task 9 to test whether explicit lookahead and persistent pursuit can convert the same states where PPO cycles. Another learned algorithm remains gated on that diagnostic.
