# Task 11 Sealed-Engine Preflight Design

**Date:** 2026-08-09

**Status:** Approved design amendment for Task 11 of the selective-DAgger plan

**Related documents:**

- `docs/superpowers/specs/2026-08-03-selective-dagger-search-distillation-design.md`
- `docs/superpowers/plans/2026-08-03-selective-dagger-search-distillation.md`
- `docs/superpowers/specs/2026-08-09-task10-owned-overlay-evidence-design.md`
- `docs/superpowers/plans/2026-08-09-task10-breaker-repair.md`

## Problem

Task 8 currently has a strict schema-2 physical opener. It validates exact
inventory, retained traces and replays, benchmark records, schedules, metrics,
and deterministic oracle selection. That proves integrity: the publication is
self-consistent and has not changed.

The schema-2 producer is nevertheless callback-owned. Its execution-trust
declaration is permanently `untrusted-test-transcript`, with
`engine_authenticated=false`. The files do not prove that the real HexWars
GymServer executed the games, bounded-search queries, and codec round-trips.
Task 10 therefore correctly fails closed rather than converting a callback DTO
or a self-authored wrapper into `sealed-engine` evidence.

Task 11 must provide the missing production provenance boundary before the CLI
can launch a research-grade preflight or downstream DAgger training.

## Trust Model

The seal protects against accidental or deliberate substitution inside the
training pipeline: callbacks, test fixtures, stale transcripts, copied games,
reordered records, or self-authored metadata must not be accepted as engine
evidence.

The seal does not defend against a malicious user who controls the local
machine, executable, repository, and evidence files. That stronger problem
would require an external signing authority, protected keys, or hardware-backed
attestation and is outside HexWars' current research-platform requirements.

The seal is therefore a first-party process and protocol provenance guarantee,
not a security signature.

## Goals

1. Prove that production Task 8 evidence crossed a fresh, runner-owned
   HexWars GymServer process.
2. Bind every scheduled game and its trace, replay, benchmark, oracle, codec,
   contract, encoding, scenario, and repository identities into an ordered
   engine session.
3. Make `sealed-engine` constructible only by the concrete production adapter;
   public CLI commands must expose no evaluator, benchmark, codec, client, or
   trust injection seam.
4. Retain strict physical reopening, deterministic reuse, diagnostics, and
   restart-safe stage boundaries.
5. Preserve the existing schema-2 callback path for fast tests while ensuring
   that it remains permanently untrusted.
6. Supply the authenticated Task 8 input that Task 9 and Task 10 already
   require.

## Non-Goals

- Cryptographic signing, remote attestation, protected keys, or defense against
  the machine owner.
- Changes to game rules, bounded-search policy, rewards, observations, actions,
  DAgger eligibility, mixture ratios, or optimizer settings.
- Running the 480-game production oracle preflight.
- Running any production DAgger collection or training.
- Replacing Task 12's exact physical smoke and final full-suite verification.
- Restoring Python 3.11 compatibility. Task 11 targets the project's active
  Python 3.14 training environments; older-version compatibility is deferred.

## Chosen Architecture

Use an engine-session transcript. Python owns the experiment and process;
GymServer owns authoritative execution and receipt issuance.

```text
Python production CLI                         HexWars GymServer
---------------------                         -----------------
freeze inputs and output root
launch fresh process
generate fresh nonce
             duel_evidence_begin ----------> validate tactical-v2 identities
                                    <-------- session acknowledgement

for every frozen scheduled game:
  send reset/actions -----------------------> apply authoritative transitions
                                               query bounded search twice
                                               check codec round-trip
                                               retain trace/replay/benchmark
             duel_evidence_game_close ------> freeze payloads and receipt
                                    <-------- payloads + chained receipt
  validate and write exact returned bytes

             duel_evidence_end ------------> freeze final chain
                                    <-------- closing acknowledgement
reopen all physical evidence
atomically publish oracle-preflight.json
```

The production path does not call the schema-2 callback-owned executor. Shared
pure computations such as schedule construction, metric recomputation, and
oracle selection may be factored and reused, but the trust-bearing producer and
opener remain schema-distinct.

## Engine Session Protocol

### Begin

The production runner launches a fresh GymServer process and sends
`duel_evidence_begin` before any preflight game. The exact request contains:

- `cmd="duel_evidence_begin"`
- `schema_version=1`
- `purpose="oracle-preflight"`
- a fresh 256-bit lowercase hexadecimal nonce
- the complete canonical preflight schedule plus its SHA-256 identity
- the frozen panel SHA-256 identity
- repository commit and clean-tree identity
- scenario SHA-256
- contract and encoding SHA-256 identities
- bounded-search source and heuristic identities

GymServer accepts the command only for `tactical-v2`, only when no evidence
session is active, and only when its authoritative environment, scenario,
contract, encoding, and oracle identities match the request. It validates and
freezes the complete schedule. Panel and repository identities are external
experiment claims: the engine binds them into every receipt while Python's
physical validators authenticate them against the repository. The engine
returns an exact acknowledgement containing the echoed nonce, a fresh session
ID, the frozen schedule identity, the authoritative
environment/contract/encoding/oracle identities, sequence zero, and the
initial chain hash.

The session ID and nonce provide freshness and cross-run separation. They are
not secrets.

### Authoritative Game Execution

Production preflight games use the ordinary authoritative reset and step
mechanisms, but evidence mode installs a dedicated preflight observer. For each
eligible benchmark sample, the observer:

1. captures the canonical state, observation, legal mask, and decision index;
2. queries the configured bounded-search oracle twice without changing state;
3. records both commands, encoded actions, actual expansion counts, and elapsed
   timings;
4. encodes and decodes both actions through the authoritative tactical-v2
   codec;
5. verifies legality and exact command round-trip; and
6. leaves the learner-selected action as the only action applied to the game.

Python may select learner actions and drive the episode, but it cannot supply
teacher decisions, benchmark results, codec evidence, traces, replays, or trust
metadata to the production publisher.

### Game Close and Receipt Chain

After a scheduled game terminates, Python sends
`duel_evidence_game_close`. GymServer freezes and returns the exact trace,
replay, and benchmark payloads plus an engine receipt. The receipt contains:

- schema version, session ID, nonce, sequence number, and previous receipt hash;
- candidate index and complete scheduled-duel identity;
- oracle, scenario, contract, encoding, and engine protocol identities;
- terminal outcome, transition count, benchmark sample count, and expansion
  totals;
- SHA-256 identities and byte sizes for the returned trace, replay, and
  benchmark payloads; and
- its own canonical content hash.

The receipt hash becomes the previous hash for the next scheduled game.
GymServer rejects game close before terminal state, a second close for one game,
out-of-order schedule context, or a game whose configured oracle differs from
the session request.

Python validates each acknowledgement and receipt immediately and writes the
exact returned payloads without semantic reconstruction. The physical opener
later performs an independent validation.

### End

After the complete frozen schedule, Python sends `duel_evidence_end`.
GymServer accepts it only when every expected scheduled game has one receipt
and no game is open. The closing acknowledgement contains the session ID,
nonce, receipt count, final receipt-chain hash, and a closing content hash.
Closing freezes the session; further evidence commands are rejected.

An engine exit, broken pipe, or interruption before the closing acknowledgement
can never produce a completed sealed publication.

## Physical Publication Schema

Production preflight uses `oracle-preflight.json` schema version 3 and exact
inventory:

```text
oracle-preflight/
|-- oracle-preflight.json
|-- engine-evidence/
|   |-- session.json
|   `-- receipts/
|       `-- candidate-<budget>/game-<index>.json
`-- games/
    `-- candidate-<budget>/
        |-- game-<index>.trace.json
        |-- game-<index>.replay.json
        `-- game-<index>.benchmark.json
```

`session.json` contains the exact begin acknowledgement, an ordered inventory
of receipt descriptors, the exact closing acknowledgement, and the final chain
identity. Each descriptor records a contained canonical relative path, byte
size, and SHA-256. `oracle-preflight.json` binds the session descriptor,
candidate summaries, selected oracle, complete game artifact inventory, frozen
input identity, and overall content identity.

The production execution-trust object is exact and schema-specific:

```json
{
  "schema_version": 1,
  "mode": "owned-gymserver-session",
  "evidence_class": "sealed-engine",
  "engine_authenticated": true,
  "engine_evidence_root": "engine-evidence/session.json"
}
```

No public conversion function accepts a schema-2 publication and emits this
object. The only writer receives a completed concrete engine session owned by
the production runner.

## Physical Opener

The public sealed preflight opener must:

1. resolve-contain the publication root without following reparse points;
2. recompute the expected panel, repository, dataset, scenario, schedule,
   learner, contract, encoding, and oracle-source identities;
3. require the exact production schema and exact recursive inventory;
4. validate the begin and closing acknowledgements against the expected
   identities;
5. reopen every receipt in canonical schedule order and rebuild the hash chain;
6. match every receipt's payload hashes and sizes to the retained trace,
   replay, and benchmark bytes;
7. semantically reopen the trace, replay, benchmark records, authoritative
   commands, legality evidence, and terminal result;
8. recompute per-candidate metrics, gates, tie breaks, and selected oracle from
   retained physical evidence;
9. compare the recomputed values to the manifest; and
10. reread all load-bearing files and inventory before returning.

Only this successful opener returns an `OraclePreflightPublication` whose
`evidence_class` is `sealed-engine`. The schema-2 opener continues returning
`untrusted-test-transcript`. Task 10 must use the public dispatcher and retain
its explicit `sealed-engine` requirement.

## CLI and Dependency Boundary

Task 11 retains the planned commands:

```text
prepare
validate
preflight
baseline
iteration --index {1,2,3}
evaluate
aggregate
report
smoke
all
```

Every command requires an explicit output root. `preflight` constructs the
concrete GymServer adapter internally from the committed panel and production
server command. Production CLI parsing exposes no seed, schedule, threshold,
oracle, callback, client, trust, or final-bank override. `all` runs stages in
dependency order and stops at the first failure.

Task 9 and Task 10 receive only the reopened physical publication. They do not
accept an in-memory seal DTO or callback assertion.

## Failure, Diagnostics, and Reuse

- Destination and staging roots may not coexist.
- Any protocol, identity, repository, schedule, receipt, artifact, or final
  reread mismatch fails closed before publication.
- An incomplete session is never resumed or sealed. Its staging files and
  GymServer stderr are moved into a bounded timestamped diagnostic directory.
- A retry launches a fresh process with a fresh nonce and reruns preflight.
- A completed preflight is reusable only after the sealed physical opener
  succeeds against current frozen inputs.
- Successful reuse launches no GymServer process and reports zero new games and
  zero new epochs.
- An incompatible completed destination requires a new output root; it is not
  overwritten or silently repaired.
- Downstream stages cannot run when the preflight is missing, untrusted, or
  incompatible.

## Logging

Task 11 logs to stdout and `<output-root>/selective-dagger.log`, flushing every
progress event. Preflight events include command, stage, output root,
repository identity, process start, session ID, candidate/game progress,
receipt sequence and chain identity, games, samples, expansions, throughput,
elapsed time, ETA, physical publication/reopen result, reuse counts, failure
stage, diagnostic path, and process exit information.

The log is operational evidence, not an authority. The physical opener trusts
only the immutable publication.

## Verification Strategy

### Engine unit tests

- exact begin/end protocol fields and tactical-v2-only enforcement;
- nonce echo, one active session, and immutable frozen identity;
- monotonic sequences and correct receipt-chain construction;
- authoritative repeated oracle queries and codec round-trip;
- learner action remains the applied transition;
- rejection of wrong schedule, wrong oracle, duplicate close, premature close,
  and post-close commands.

### Python unit and adversarial tests

- production public APIs expose no callback or trust injection;
- schema-2 callback evidence cannot be relabeled or consumed by Task 9/10;
- missing closing acknowledgement rejects publication;
- wrong nonce/session/sequence/previous hash rejects;
- deleted, duplicated, reordered, or cross-session receipts reject;
- trace, replay, benchmark, schedule, oracle, scenario, contract, encoding, or
  repository mutations reject;
- a self-consistent forged outer manifest still rejects;
- interrupted publication preserves diagnostics without a completed marker;
- complete reuse performs zero engine launches, games, and epochs;
- Task 10 accepts the real sealed physical publication and still rejects test
  transcripts.

### Real process integration

A small deterministic GymServer integration starts the actual process, creates
one evidence session, runs a minimal tactical-v2 game, closes it, reopens the
physical publication, and proves that the engine-issued receipt binds the
retained artifacts. This is a Task 11 boundary test, not the Task 12 exact
physical smoke or the 480-game production preflight.

## Acceptance Criteria

Task 11's sealed-engine amendment is complete only when:

1. the real GymServer is the sole source of production preflight teacher,
   codec, trace, replay, benchmark, and receipt evidence;
2. a completed physical production publication reopens as `sealed-engine`;
3. callback and fixture publications remain explicitly untrusted;
4. Task 10 accepts the real publication without a private seam;
5. all adversarial substitution and mutation tests fail closed;
6. CLI, logging, diagnostics, dependency guards, and zero-compute reuse satisfy
   the original Task 11 requirements; and
7. the focused engine, Python, and real-process integration suites pass under
   the active Python 3.14 and .NET 8 environments.

Production compute remains unauthorized until Task 12's exact smoke, complete
verification, and final branch review are accepted.
