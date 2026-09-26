# Experiment panel retention

Completed panels keep the small artifacts needed to understand the decision:
the protocol, locked manifest and seed bank, aggregate metrics, report, and any
small validation fixture referenced by the manifest.

Per-match evaluations, controller snapshots, logs, replays, traces, and other
generated evidence stay local and are ignored by Git. They can be regenerated
for a new panel, but copying gigabytes of machine-specific output into the
repository does not make a result more reproducible.

Hashes beside normal source paths record the bytes used when a panel ran; they
are not assertions that today's file at that path is still byte-identical.
Likewise, hashes for omitted binaries, checkpoints, and per-match evidence keep
the historical identity auditable without treating generated files as source.

The July 2026 tactical-v2 panel reports are historical decision evidence. Their
one-off runners and focused tests targeted the CLI and source identities present
at the time, so exact copies live under each panel's `historical-source/`
directory with a `.py.txt` suffix. They are reviewable records, not maintained
entry points or active tests. A new experiment should use a new panel identity
and the current ML Lab lifecycle rather than mutating or rerunning a completed
panel in place.
