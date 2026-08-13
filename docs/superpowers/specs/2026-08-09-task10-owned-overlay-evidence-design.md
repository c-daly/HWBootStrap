# Task 10 Evaluator-Owned Overlay Evidence Design

## Context

Task 10 supervised evaluation originally reopened cumulative validation-overlay
directories owned by Task 9. Repeated reads detected many mutation boundaries,
but they could not establish an atomic view of multiple independently mutable
directory trees. For every finite multi-root scan, some root has an earliest
last read and may change while a later root is read.

The user approved replacing that impossible guarantee with evaluator-owned,
content-addressed evidence on 2026-08-09.

## Decision

Before inference, the supervised evaluator serializes the already authenticated
overlay inventories into one deterministic bundle owned by the supervised
publication. The bundle is a single regular artifact whose filename is its
SHA-256 digest. It contains each cumulative overlay under an ordered,
content-identity-qualified prefix and preserves the exact directory/file
inventory captured by the Task 9 physical opener.

The evaluator reopens the bundle, validates its outer hash and exact archive
schema, safely materializes it into a private temporary directory, invokes the
existing first-party DAgger overlay opener on every materialized overlay, and
derives supervised rows from those reopened copies. Predictions, metrics, reuse,
and aggregate publication depend on the owned bundle, never on the original
Task 9 paths after capture.

## Publication schema

The supervised manifest and evidence schemas advance to version 2. The
publication contains exactly:

```text
manifest.json
evidence.json
predictions.json
metrics.json
owned-overlays/
  <bundle-sha256>.zip
```

The manifest describes the bundle with relative path, SHA-256, and byte size.
The evidence artifact records, in cumulative order, each source root as
provenance, its Task 9 content identity, and its exact bundle prefix. Absolute
source roots are claims bound into the publication identity; owned relative
paths are the physical authority.

Archive entries are deterministic and exact: fixed metadata, sorted paths, no
duplicate names, no absolute paths, no `..`, no symlink/reparse encodings, and
no entries outside the declared overlay prefixes. Extraction is manual into a
new private temporary root; `extractall` is not used.

## Data flow

For a new publication:

1. Authenticate the original Task 9 overlay roots and capture their exact byte
   inventories.
2. Create the supervised staging directory.
3. Build the deterministic bundle only from captured bytes; do not reread the
   sources while bundling.
4. Reopen the bundle through the strict bundle decoder and the existing DAgger
   overlay opener.
5. Require every owned content identity and projected supervised row to match
   the authenticated source evidence and frozen Task 10 definition.
6. Run pre/post predictions and compute metrics from owned rows.
7. Write schema-2 evidence and manifest artifacts, validate the complete staging
   publication, and atomically rename it.
8. Reopen the published bundle and all local artifacts before returning.

For reuse and aggregate passes, source roots need not exist. The caller's
declared source-root sequence must equal the recorded provenance sequence, but
all labels, reasons, metrics, and content identities are reconstructed from the
owned bundle.

## Integrity contract

The supervised publication is the sole authorized writer of its bundle. Atomic
rename publishes it once; production code never mutates it afterward. The
returned evidence is a snapshot bound to the exact bundle bytes and hash read
during that open. A non-cooperating process that writes publication files is
external corruption: the next physical open must reject it, but no reader claims
it can prevent a write that occurs after its last read.

This contract is both stronger and more honest than claiming simultaneous
stability of several foreign directories. It gives every reported metric a
retained, portable physical source while acknowledging the ordinary filesystem
boundary.

## Failure and restart behavior

- Any source inconsistency while its inventory is captured produces either an
  invalid owned bundle or a content-identity mismatch and fails before
  publication.
- Any malformed, truncated, extra-entry, traversal, duplicate, wrong-hash, or
  wrong-prefix bundle fails closed.
- Predictor or validation failure leaves only the deterministic staging path;
  it cannot be mistaken for a completed result.
- Reuse launches zero inference and does not open original Task 9 overlay roots.
- Existing post-rename rollback semantics remain unchanged.

## Verification

Tests must prove:

- a completed supervised publication reopens after every original overlay is
  deleted;
- mutation of an original overlay after bundle capture cannot change reopened
  rows or metrics;
- bundle byte mutation, path traversal, duplicates, reparse encodings, extra
  entries, wrong content identities, and unowned top-level entries fail;
- bundle materialization round-trips the exact Task 9 overlay inventory and
  uses the existing physical DAgger opener;
- reuse performs zero inference and zero original-overlay reads;
- aggregate pass 1 and pass 2 reconstruct identical supervised evidence from
  bundle bytes and reject a bundle mutation between passes;
- Windows symlink tests skip only for WinError 1314.
