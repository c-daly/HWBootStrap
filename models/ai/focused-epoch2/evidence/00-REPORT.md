# Targeted training experiment

Candidate failed a preregistered gameplay gate; retain it as an unpromoted experiment.

The integrity audit passed for 1802 files. Selected checkpoint: `ec2c56a392a88b909d573c8d65f52ea578b8f2cf512a1884eebdccc80fdbd932`.

Training completed 4 epochs and 528 optimizer steps. The selected checkpoint is epoch 4 by the frozen validation-NLL rule.

On 3700 retained archived validation examples, source NLL was 1.576220 and candidate NLL was 1.533905. This measures imitation of archived labels; gameplay acceptance is determined separately below.

| Archived validation metric | Source | Candidate | Denominator |
|---|---:|---:|---:|
| teacher_agreement | 1951 | 1954 | 3700 |
| positive_agreement | 137 | 154 | 193 |
| lethal_agreement | 65 | 67 | 72 |
| move_agreement | 1170 | 1151 | 2681 |
| deploy_agreement | 6 | 11 | 182 |
| lethal_selected | 74 | 75 | 80 |
| positive_selected | 170 | 192 | 223 |

## Development: failed

| Controller | W | D | L | W + 0.5 D | Round-cap draws | Step truncations |
|---|---:|---:|---:|---:|---:|---:|
| focused-epoch2 | 62 | 12 | 22 | 68 | 12 | 0 |
| closing-mirror | 54 | 24 | 18 | 66 | 24 | 0 |
| targeted-candidate | 57 | 13 | 26 | 63.5 | 13 | 0 |
| greedy-control | 54 | 10 | 32 | 59 | 10 | 0 |

Failed preregistered gates:

- `seat-0-retention-vs-focused-epoch2`: candidate-minus-baseline wins -1, integer score -1.
- `seat-1-retention-vs-focused-epoch2`: candidate-minus-baseline wins -4, integer score -8.
- `seat-1-retention-vs-closing-mirror`: candidate-minus-baseline wins -2, integer score -9.
- `standard-3v3-retention-vs-focused-epoch2`: candidate-minus-baseline wins -1, integer score -3.
- `scenario-custom-13x9-hills-retention-vs-focused-epoch2`: candidate-minus-baseline wins -2, integer score -4.
- `scenario-custom-18x12-flat-retention-vs-focused-epoch2`: candidate-minus-baseline wins -1, integer score -2.
- `scenario-default-13x9-flat-retention-vs-focused-epoch2`: candidate-minus-baseline wins -2, integer score -4.
- `development-improvement`: candidate-minus-baseline wins -5, integer score -9.

Confirmation bank: no confirmation artifacts or recorded execution; reserved.

## Limits

- Label agreement and validation NLL measure imitation of retained archived teacher labels, not tactical optimality or gameplay strength.
- Validation targets with selected zero-damage attacks were excluded by the frozen recipe; scores describe that retained distribution.
- The experiment uses fixed-template combat with bounty income and redeployment, at most three initial units, and no fog, capture, generators, territory or custom unit design.
- The original exposed 96-game panel and selected counterfactuals are diagnostic evidence, not confirmation.
- Finite panel acceptance is not a statistical superiority or equivalence claim and does not complete the historical combat/beacon retention gate.
- Movement tags are contextual proxies, not proven productive or safe actions.
- Archived combat roster has three mobile templates; default zero-damage Scout and immobile Artillery are absent.
- A zero-damage attack may be a legal expendable command before another unit attacks; exclusion is a preregistered training choice, not an optimality proof.
- Teacher search budgets and labels are retained exactly; no replacement actions or outcome-selected training games.
- Canonical-key uniqueness does not assert optimal labels or absence of semantically equivalent states under different game histories.
