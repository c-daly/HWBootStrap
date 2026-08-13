# Tactical-v3 Split Run Identity Design

## Problem

Structured publication currently writes the realized Gym spaces identity to scenario.json. Python can validate that artifact, but ML Lab requires scenario.json to use the strict training-scenario schema. Replacing the file alone would then break Python's exact policy/checkpoint identity checks because the seed-specific policy contract (bac4...) intentionally differs from the Arena scenario contract (0ae4...).

## Design

Structured runs use schema version 2 and contain two explicit artifacts:

- scenario.json is the canonical ML Lab training scenario used to construct Arena matches.
- policy-identity.json is the canonical full TacticalV3SemanticIdentity spaces payload used to authenticate the corpus, checkpoint, inference fixture, and runtime policy compatibility.

run.json.contract remains the policy identity summary and adds policy_identity: policy-identity.json. The strict inventory includes the new sidecar. Python validates the manifest, sidecar, corpus, checkpoint metadata, and fixture against the same policy identity before tensor loading. The publisher takes the training scenario path and policy identity as distinct keyword-only inputs so they cannot be reversed.

The CLI keeps --scenario for the canonical training scenario and requires --policy-spaces for the realized policy identity. Existing schema-v1 runs remain immutable and are not migrated or overwritten; acceptance publishes a fresh sibling run.

## Unity Compatibility

ML Lab continues to parse scenario.json through its strict training-scenario loader. For tactical-v3 seats it requires the policy manifest to declare tactical-v3/duel, a well-formed contract hash, and encoding/capacity hashes equal to the Arena scenario. It does not require the policy match contract hash to equal the Arena match contract hash because board/match provenance legitimately differs while model geometry remains compatible.

ML Lab also requires the declared policy-identity.json to be a contained regular file. It verifies the sidecar's top-level environment/version/kind and contract/encoding/capacity hashes against run.json without deserializing a checkpoint.

## Rejected Alternatives

- Teaching Unity to derive a training scenario from spaces duplicates the schema and loses the exact authored launch scenario.
- Changing only run.json cannot repair the incompatible scenario.json shape or preserve Python's exact checkpoint/corpus identity boundary.

## Verification

TDD adds publication, validation, controller pre-load, CLI wiring, E2E, and ML Lab regressions. A fresh schema-v2 run must validate in Python, load through live Unity/Coplay with the intentionally different contract hashes, retain exact encoding/capacity identity, and preserve deterministic 13x9 plus legal 24x16 inference. The old v1/v2 acceptance artifacts are read-only evidence.

