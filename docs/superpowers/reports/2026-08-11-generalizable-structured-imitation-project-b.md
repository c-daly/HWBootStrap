# Project B Report: Generalizable Tactical-v3 Structured Imitation

## Outcome

Project B now trains, publishes, validates, reloads, serves, and runs a tactical-v3 structured-imitation policy through the real GymServer/policy-server boundary and Unity Arena. The published evidence remains explicitly unsealed-experimental; this task does not claim full-game strength or a win-rate result.

The final run format is schema version 2:

- scenario.json is the canonical Arena/training scenario.
- policy-identity.json is the authenticated semantic spaces identity used by the corpus, checkpoint, controller, and policy server.
- run.json contract summarizes the policy identity and declares policy_identity as policy-identity.json.
- Unity derives the Arena match contract from scenario.json, requires policy environment/version/kind plus matching encoding/capacity hashes, and permits a distinct policy contract hash because match provenance and board geometry may differ.

The split identity design and implementation plan are committed separately in 44ec2b4.

## Implementation

- Added the bounded Task 13 end-to-end module covering deterministic double training/publication, exact reload logits/actions, a real 13x9 GymServer-to-policy-server-to-step round trip, and legal 24x16 inference without rebuilding the model.
- Added the measured CLI smoke training configuration used by both tests and the real CPU acceptance publication: hidden dimension 16, batch size 8, learning rate 0.005, 80 epochs/patience, seed 227.
- Hardened publication and validation around an exact seven-entry inventory, canonical JSON, a contained fixed policy sidecar, corpus/checkpoint/fixture identity equality, and atomic no-overwrite publication.
- Updated structured controller and policy-server resolution to require schema 2 and read policy-identity.json before checkpoint tensor loading. Legacy SB3 runs remain schema 1.
- Updated Unity ML Lab admission for schema 2, contained scenario/manifest/checkpoint/policy files, Duel policy kind, SHA-256 syntax, and encoding/capacity compatibility. A 24x16 Arena scenario with the 13x9-trained policy contract is explicitly accepted.
- Tactical-v3 remains offline-imitation-only in Train and loadable through Arena.

## TDD Evidence

- Initial end-to-end harness RED timed out at 900 seconds because training was repeated per test. A measured module-scoped fixture reduced the four-test gate to about five minutes without weakening assertions.
- The initial real CLI publication was structurally valid but failed the quality threshold under the former default configuration: validation accuracy 0.75 and NLL 0.67248. The measured shared CLI/test configuration produced 1.0 accuracy and NLL below 0.02.
- The first live Unity selected-run probe failed with missing scenario.environment, proving that the schema-1 publisher had overloaded scenario.json with a spaces payload.
- The split-identity Unity RED failed exactly because policy and Arena contract hashes differed. Schema 2 then passed the same behavioral test.
- The first schema-2 end-to-end rerun passed 3/4 and failed only because the policy-server resolver still required schema 1. After making resolver schema selection algorithm-aware and reading the sidecar, the full gate passed 4/4.
- Explicit missing, repointed, mismatched, and pre-checkpoint policy-sidecar cases now fail closed in Python and Unity.

## Acceptance Artifact

Path: python/runs/tactical-v3-policy-project-b-acceptance-v3

- State: completed
- Evidence: unsealed-experimental
- Schema: 2
- Best epoch: 38
- Train: accuracy 1.0, policy NLL 1.8030275441560661e-06, 234 finite valid logits
- Validation: accuracy 1.0, policy NLL 6.824638603575295e-06, 56 finite valid logits
- Corpus SHA-256: cc4ebbbd5c230c8797c84155c542e9cbf39074fa03f04fd521c316649b04c123
- Model-state SHA-256: a755b8e82a8678e44732110034ab2382e82ce8ab1b3fbe918ff7e61195e78ad8
- Checkpoint-file SHA-256: 7ac5b32d679738829a4fb1ca96e461aed6f7d891fe11a301d497f61a12f5dde3
- Scenario SHA-256: 868df700953836eb86c958ca9557bf1133efd6ba4b52363ffa7bb4fdba4e003c
- Policy-identity SHA-256: 82c3d8beba2c4869300a38ad0844e7021b6b94dd021afb16b12df7f8021ceea7
- Policy contract: bac4af4d4b8e68466ffaf37c2721f98129edc93b90f529999ba45463cd921437
- Encoding: e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6
- Capacity: 7aea1db4f008dc192e83811b2c13abd8ce2304d2a6a209f37f9847be5f367364

The real validate-run CLI exited 0. Earlier acceptance v1/v2 directories were preserved unchanged; v3 was a fresh atomic sibling publication.

## Verification

- Python 3.14.6 focused structured-policy gate: 551 passed, 6 skipped in 856.35 seconds.
- Final checkpoint/controller sidecar gate: 76 passed, 6 skipped in 96.65 seconds.
- End-to-end gate after schema-2 resolver fix: 4 passed in 339.47 seconds.
- Engine suite: 995 passed, 0 failed/skipped in 207.5 seconds.
- Release build-to-Unity: 0 errors; 5 existing nullable warnings.
- Unity 6000.5 exact-worktree EditMode gate after review hardening: 191 passed, 0 failed/skipped in 3.31 seconds.
- Temporary exact-artifact Unity admission probe: the actual ignored v3 run passed 1/1 under the final strict validator and produced its exact run:PATH spec; the environment-specific test was removed afterward.
- Live Coplay health: exact project connected, play mode off, hasCompilationErrors=false; check_compile_errors returned No compile errors.
- Unity console contained no task error; only an unrelated unsupported-toolbar warning.
- CUDA was available. The standalone real CUDA training/publication test passed 1/1 in 8.67 seconds: GPU-reloaded logits matched CPU at rtol 1e-5 and atol 1e-6, and selected actions matched exactly. The prescribed deterministic acceptance publication remained CPU.

Six focused Python skips and 21 full-suite skips are the existing Windows privilege-only symlink cases.

## Full-suite Limitation

The full Python suite was attempted and returned 1908 passed, 21 skipped, and 64 failed. The failures are outside Task 13 and reproduce independently in Project A annihilation selective-DAgger panel validation. Root-cause audit found stale physical identity inputs:

- one seed-bank file differs only through Windows CRLF checkout conversion;
- the declared BoundedSearchAgent.cs hash matches neither current bytes nor the Git blob;
- the external 3966-file dataset aggregate has the same count and byte size but a different content digest.

Transient normalization of the seed-bank advanced the suite but could not repair the stale source and external-dataset identities. Those inputs were restored, no Project A oracle/dataset was modified, and the worktree remained clean outside Task 13 scope.

## Boundaries

This evidence establishes deterministic offline overfit on the authenticated tiny corpus, exact checkpoint reload behavior, real transport/action identity, and capacity-compatible cross-size inference. It does not establish full-game policy quality, bounded-search full-game collection, production DAgger, curriculum performance, fog-of-war behavior, unit-design generalization, or promotion/sealing readiness.

## Independent Review Fix

- Unity policy-identity admission now strictly parses the exact full sidecar, rejects duplicate/extra/missing properties, and recomputes encoding, capacity, and contract hashes over canonical parsed bodies.
- The focused behavioral RED rejected none of four minimal/body-tamper/extra/duplicate cases (0/4); after the fix the same matrix passed 4/4.
- The final combined Unity gate passed 191/191 with no compile or unhandled errors.
