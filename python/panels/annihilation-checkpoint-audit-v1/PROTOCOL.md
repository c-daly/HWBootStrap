# Annihilation Physical Checkpoint Audit v1

## Research question

Does the physically retained seed-227 behavioral-cloning-to-PPO checkpoint trajectory improve annihilation performance against Random, and does the trajectory indicate replication, retained-imitation PPO, DAgger, or an inconclusive follow-up? This audit is exploratory. It does not replace the locked imitation panel and cannot promote a production model.

## Candidates

The `prepare` command discovers candidates only from supplied physical files and freezes their byte identities. The required trajectory is the pure behavioral clone at step 0 followed by every physically present BC-initialized PPO checkpoint in increasing step order. Random and bounded-search are scripted anchors. A compatible scratch-PPO run is optional; when it is not supplied, the omission is explicit. Manifest-only, partial, or inferred checkpoints are never candidates. In particular, an in-memory stop count does not create a 51,200-step checkpoint.

## Schedule

Each candidate plays 100 maps, seeds 16,000,000 through 16,000,099, against Random under `standard-3v3`. Each map is reciprocal: the candidate plays once as player 0 and once as player 1, for exactly 200 ordered games per candidate. The environment and version are `tactical-v2`. This published schedule has no CLI override.

## Metrics

- W/L/D are candidate-perspective counts over 200 games; win uncertainty is a 95% Wilson interval.
- Seat delta is candidate-as-player-0 win rate minus candidate-as-player-1 win rate.
- Draw pathology reports cycling and action-waste counts/incidence plus mutually exclusive primary draw categories.
- Win speed reports round and decision-count mean, median, and p90 only among wins; no wins means unavailable, not zero.
- Health-adjusted advantage reports final and peak normalized advantage over all games and draws.
- Successive trajectory checkpoints use the same `(map seed, candidate seat)` keys to form a paired outcome-transition table and exact sign test.
- EndTurn waste is observable from traces. EndTurn action rank/probability is unavailable because the integer-action inference boundary does not expose policy ranks or probabilities.
- Every match retains and hashes its trace and replay. Candidate checkpoints, source manifests, source scenarios, the frozen definition, runtime contract, and aggregate are also physically identified.

## Decision precedence

The aggregate records every clause, then selects exactly one next experiment in this order:

1. A later PPO checkpoint losing at least 20 of 200 wins (10 percentage points) relative to an earlier PPO checkpoint selects `test_retained_imitation_constraint`.
2. Otherwise, a PPO checkpoint with at least 130 wins and a nondecreasing trajectory with some improvement through the earliest qualifier selects `replicate_seeds_211_223`.
3. Otherwise, all PPO checkpoints below 100 wins or cycling-dominant latest PPO evidence selects `proceed_to_dagger`.
4. Otherwise, select `inconclusive_review_trajectory`.

## Seed isolation

The 16m bank belongs only to this exploratory audit. The 17m bank remains untouched. DAgger banks 18m, 19m, and 20m are reserved and not consumed by this audit. No final-evaluation or locked-panel seeds are read into the decision.

## Commands

Run from the repository root, supplying canonical physical run directories:

```powershell
uv run python python/run_annihilation_checkpoint_audit.py prepare --clone-run CLONE --ppo-run PPO --output-root OUTPUT
uv run python python/run_annihilation_checkpoint_audit.py validate --output-root OUTPUT
uv run python python/run_annihilation_checkpoint_audit.py evaluate --output-root OUTPUT --workers 4
uv run python python/run_annihilation_checkpoint_audit.py aggregate --output-root OUTPUT
uv run python python/run_annihilation_checkpoint_audit.py report --output-root OUTPUT
uv run python python/run_annihilation_checkpoint_audit.py all --clone-run CLONE --ppo-run PPO --output-root OUTPUT --workers 4
```

Add `--scratch-run PATH` only when a physical compatible scratch run is intentionally included. The `all` command executes prepare, validate, evaluate, aggregate, and report in order and stops on the first failure.

## Output tree

```text
OUTPUT/
  definition.json
  manifest.json
  audit.log
  candidates/<candidate-id>/map-<seed>/
    evaluation.json
    evidence/<trace-and-replay-artifacts>
  audit.json
  report.md
```

`definition.json` freezes discovery. `manifest.json` binds repository, scenario, source, runtime, and definition identities. `audit.json` is reproducible only by reopening all physical map evidence. `report.md` is rendered from that revalidated aggregate, never trusted as an independent data source.

## Recovery

Evaluation is restart-safe at the candidate/map boundary. Rerun the identical command after interruption: valid completed maps are reopened, hashed, and reused; only missing maps run. A malformed existing map, unexpected file, changed source/checkpoint, incompatible runtime, or mismatched frozen definition fails closed. Do not delete or silently overwrite the discrepancy. Investigate it, preserve `audit.log`, and restart only after the physical identity issue is understood. Aggregate and report likewise reopen and validate every map.

## Full contract and encoding compatibility

Source full-contract hashes may legitimately differ because they capture more than tensor layout and can come from different run snapshots. They remain recorded separately and are never rewritten as equal. Compatibility instead requires every learned source and the single evaluation runtime to agree on tactical-v2 environment/version, encoding hash, observation size, action size, and board geometry. Thus provenance stays exact while the model/runtime interface is demonstrably compatible.
