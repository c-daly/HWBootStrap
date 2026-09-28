# Gameplay AI and research artifacts

The project deliberately selects a trained gameplay package through
`Assets/StreamingAssets/ai-models.json`. The initial default difficulty is Hard, mapped to `focused-epoch2`.
See [AI model configuration](../operations/ai-model-configuration.md) for the one
project/deployment mapping shared by normal gameplay, title demonstration, and
unpinned developer duels. Players choose your configured difficulty names, not
model files or training runs.

The ML Lab remains a developer experiment surface. Publishing or training a lab
checkpoint never changes the gameplay catalog. Explicit lab controller choices
also remain pinned when the project default changes.

## Package and supported contract

The focused package preserves the exact source checkpoint and its evaluated
provenance. Its manifest marks completion of the **inference export**, while the
copied source manifest continues to report stopped training. It does not turn a
stopped run into completed training or reverse any research retention-gate result.

The playable adapter supports Annihilation without fog, bounded by the tactical-v3
state and candidate capacities. Setup and runtime checks reject unsupported
configurations. Evidence from the retained reciprocal Greedy panels supports this
limited deployment choice; it does not establish performance across every game
mode, roster, board, or player strategy.

Desktop inference uses an asynchronous Python policy process. WebGL sends the
same structured decisions to a bounded hosted Python process; Python and model
weights are not executed in the browser. Both routes load the same checkpoint,
validate its contract and identity, replay its saved CPU fixture, and return an
action that must still pass the engine's legal-action checks.

This is a different runtime choice from the earlier proposed ONNX/Unity Inference
Engine path. A future native export would need numerical/action agreement and
platform validation before replacing the current Python runtime.

## Changing the official selection

1. Evaluate a candidate on frozen reciprocal suites and retain its failures,
   regressions, provenance, and timing results.
2. Export an immutable package with an exact checkpoint digest and completed
   evidence. Preserve the source training state and older packages.
3. Add the package's contract and identity to the shared catalog and deliberately
   map a difficulty to its ID; choose the default difficulty separately.
4. Validate local and hosted inference, scene persistence, browser integration,
   responsiveness, latency, memory, cancellation, and error recovery for the
   intended release. A passing checkpoint fixture alone is not platform testing.
5. Rebuild/stage the client when its integration changes, then release through the
   normal deployment workflow. Keep a catalog rollback available.

Catalog revisions and checkpoint hashes are pinned for a match. Initialization,
queueing and inference have deadlines. Cancellation discards in-flight policy
processes when needed to avoid consuming a late response as the next decision.
Errors remain visible; the game does not silently replace the trained opponent
with Greedy. Mapping a difficulty to `greedy` in the project catalog is an explicit rollback.
