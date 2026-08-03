# Training Game Templates and Faithful Live Viewer

## Purpose

HexWars ML Lab must let experimenters choose the game configuration used by
headless training. A completed run must retain an immutable copy of that
configuration, and Start & Watch must reproduce the run's scenario, opponent,
and learner-seat schedule rather than silently launching learner-as-P0 against
Greedy.

Fine-tuning across scenarios is intentionally out of scope. The design must not
foreclose it: scenario snapshots retain both the complete contract identity and
the encoding identity needed to distinguish exact resume compatibility from
future geometry-compatible transfer.

## Current behavior and root cause

ML Lab currently exposes run and optimizer settings but only passes an
environment name (`tactical-v1` or `adaptive-v1`) to Python. GymServer constructs
`EnvConfig` or `AdaptiveEnvConfig` from hard-coded defaults.

`ReplayViewerMenu.WatchLiveRun` currently places the learner run in P0 and passes
the literal `"greedy"` for P1. It does not read `config.learner_seat` or
`config.opponent` from `run.json`. This is why Start & Watch can show a different
seat assignment and opponent from the training run.

Python's `WorkerSchedule` does alternate each worker when
`learner_seat == "alternating"`, and tests exercise the schedule and environment
rebuild. Episode files do not currently record the learner seat, so a real run
cannot be audited from durable metrics.

## Template library

The repository ships a versioned template library at:

`python/config/training-game-templates.json`

The top-level document contains:

- `schema_version`: initially `1`;
- `templates`: an array of uniquely identified training game templates.

Each template contains:

- `id`: stable CLI-safe identity;
- `name`: human-readable label;
- `environment`: `tactical-v1` or `adaptive-v1`;
- `board`: procedural-board values;
- `rules`: gameplay/economy values;
- `episode`: truncation and horizon values;
- `reward`: reward-shaping values;
- `adaptive`: adaptive-only deployment/design values, omitted for tactical
  templates.

The initial library contains environment-specific variants of:

- **Standard**: the exact current defaults;
- **Long Battle**: standard board geometry with longer round and step limits;
- **Large Battle**: larger geometry and proportionate deployment/horizon values.

There is no smaller-board preset. Large geometry is permitted, but the UI shows
the resolved observation/action dimensions and a performance warning.

## ML Lab workflow

The Train tab gains a **Game template** selector filtered by the selected
environment. Selecting a template creates a session working copy. A grouped
advanced foldout edits that copy:

- **Board**: width, height, maximum elevation, deployment-zone depth, flat
  chance, and terrain weights.
- **Match rules**: actions per turn (`0` in the UI means whole team), round cap,
  maximum environment steps, starting points, fog, biomes, bounty rate,
  deployment cost, and generator values.
- **Reward shaping**: shaping scale, step penalty, closing reward, draw credit,
  and points-value weight.
- **Adaptive deployment**: starting unit count, starting army budget, and
  maximum design point cost.

Adaptive encoding constants remain locked in this version:

- 24 controllable unit slots;
- nine template slots;
- six fixed templates and three custom templates.

Changing those constants requires a new action/observation encoding rather than
a different training scenario.

Unsaved advanced edits remain in the Unity Editor session. **Save as template**
writes a new named entry to the JSON library. Saving uses an atomic replacement,
requires a unique ID, and requires explicit overwrite confirmation for an
existing ID. **Reload templates** discards the session copy and reloads the
library.

The resolved preflight panel displays:

- template name and environment;
- board/rule/deployment summary;
- observation size and action count;
- learner-seat schedule;
- opponent or opponent-pool schedule;
- any performance or compatibility warnings.

## Resolution and data flow

The template library is authoring data. Training never depends on it after a run
has been created.

1. Unity loads a template and applies session overrides.
2. Unity writes the resolved working copy to a temporary scenario JSON file
   under the project's Unity `Library` directory.
3. ML Lab passes that file to the Python CLI with an explicit scenario-file
   argument. Headless CLI users may instead select a template ID from the
   repository library or provide their own scenario file.
4. Python parses and validates the scenario before creating the run.
5. Python writes the canonical resolved scenario to
   `<run>/scenario.json` and records its relative path, template ID, and schema
   version in `run.json`.
6. Every GymServer worker receives the run-local resolved scenario path.
7. GymServer parses and validates it, constructs the appropriate environment
   configuration, and returns its authoritative contract handshake.
8. Python verifies that the handshake matches the requested scenario before
   model creation.

The run-local scenario is immutable provenance. Later template-library edits do
not affect the experiment.

## Contract and compatibility semantics

The existing `EnvironmentContract` remains authoritative:

- `contract_hash` identifies the complete resolved scenario, including rules,
  reward, horizon, roster, and geometry.
- `encoding_hash` identifies the observation/action encoding and model geometry.

Resume continues the same experiment and requires the exact recorded scenario
and contract. No cross-scenario fine-tuning workflow is added now.

The separation between contract and encoding identities is retained so a future
fine-tune command can explicitly allow compatible rule/reward changes while
rejecting geometry, roster-slot, or encoding changes. This release does not
silently relax resume validation.

Existing runs that predate `scenario.json` resolve to the current standard
defaults and are visibly labeled `legacy-default`. They are not rewritten.

## Validation

Validation occurs at three boundaries:

1. **Unity** provides immediate, field-specific feedback and disables launch.
2. **Python** validates schema, types, ranges, environment-specific fields, and
   canonicalization before creating durable run state.
3. **GymServer/.NET** validates the authoritative constructed configuration and
   rejects impossible combinations before training.

Validation includes:

- positive board dimensions and legal elevation/terrain values;
- non-overlapping deployment zones;
- sufficient deployment cells for adaptive starting units;
- sufficient adaptive starting budget for the required unit count;
- valid actions-per-turn, round, and step limits;
- adaptive design consistency;
- checked observation/action geometry;
- opponent model encoding compatibility.

Large but valid boards are allowed. ML Lab warns about the estimated tensor and
action-space size rather than imposing a small arbitrary cap.

An unreadable or invalid template library disables template-dependent training
controls and displays the exact path and parse error. No hidden compiled fallback
duplicates the JSON defaults.

## Faithful Start & Watch behavior

Start & Watch resolves a presentation plan from the selected run's immutable
metadata:

- the run-local scenario;
- `config.learner_seat`;
- `config.opponent`;
- the live learner run controller.

For a fixed learner seat:

- learner seat `0` launches the live learner as P0 and the configured opponent
  as P1;
- learner seat `1` launches the configured opponent as P0 and the live learner
  as P1.

For `alternating`, learner and opponent swap seats after every presentation
game, beginning with the learner in P0. Training workers continue to phase their
own alternating schedules by worker ID so simultaneous workers do not all begin
in the same seat. The observer follows the learner, including fog-of-war
perspective. Labels explicitly identify learner, opponent, P0/P1, controller
identity, and checkpoint step.

A single configured opponent is reproduced exactly:

- Greedy remains Greedy;
- Random remains Random;
- fixed run remains fixed at the resolved training identity;
- live run reloads only at game boundaries.

For an opponent pool, the viewer cycles deterministically through pool entries
between presentation games and displays the current entry. The viewer does not
claim to replay an actual hidden training episode; it generates separate Arena
games using the same scenario and schedules.

Manual Arena retains independent seat and observer controls. When launching two
model runs, it rejects incompatible encoding identities. Compatible models may
play under an explicitly selected run scenario; Start & Watch always uses the
learner run's recorded scenario.

If the viewer cannot reconstruct the scenario, opponent, or seat schedule, it
fails with a clear error. It never falls back to Greedy.

## Auditable seat alternation

Every monitored episode records:

- worker ID and per-worker episode index;
- learner seat;
- episode seed;
- episode reward and length;
- adaptive diagnostics when applicable.

ML Lab displays cumulative Seat 0 and Seat 1 episode counts for the selected run.
For alternating schedules, it warns if the durable counts cannot be read or if a
completed run is materially imbalanced beyond the expected one-episode
per-worker boundary. Fixed-seat runs are labeled explicitly rather than shown as
an alternation failure.

This makes alternation observable in real runs while preserving deterministic,
disjoint worker seed streams.

## Error handling

- Invalid template library: show the path and parse error; disable launch.
- Invalid working scenario: show field errors; do not invoke Python.
- Python/.NET disagreement: fail run creation before model initialization and
  report both requested and authoritative identities.
- Missing run-local scenario: use visible `legacy-default` behavior only for
  older runs.
- Incompatible opponent: reject before training or Arena launch.
- Missing or unsupported opponent metadata in Start & Watch: fail; do not
  substitute Greedy.
- Template save failure: leave the original library untouched and retain the
  session working copy.

## Testing

### .NET engine and GymServer

- Parse each built-in template and construct both environment types.
- Reject malformed schemas, invalid ranges, overlapping zones, insufficient
  deployment capacity/budget, and adaptive structural changes.
- Verify scenario values appear in the contract handshake.
- Verify changes that affect geometry update encoding identity and dimensions.
- Verify rule/reward-only changes update the full contract while preserving the
  appropriate encoding identity.

### Python

- Load templates by ID and from an explicit file.
- Canonicalize and snapshot resolved scenarios into new runs.
- Pass the same run-local scenario to every worker.
- Reject handshake/config disagreement before model creation.
- Preserve exact scenario provenance on resume.
- Record learner seat and seed in episode monitoring.
- Verify one-worker and multi-worker alternating schedules produce auditable
  balanced seat counts.

### Unity Editor

- Load, filter, edit, reload, and atomically save templates.
- Validate fields and show resolved geometry.
- Build CLI arguments containing the resolved scenario file.
- Parse run scenario/opponent/seat metadata.
- Verify fixed Seat 0, fixed Seat 1, and alternating presentation plans.
- Verify the observer follows the learner for alternating games.
- Verify opponent pools cycle and labels update.
- Regression-test that Start & Watch never hard-codes Greedy.

### Cross-stack smoke tests

- Start short tactical and adaptive runs from non-default templates.
- Confirm all workers report the same authoritative scenario contract.
- Confirm run snapshots remain unchanged after editing the template library.
- Watch fixed-seat and alternating runs and verify displayed seats, opponent,
  scenario, fog perspective, and checkpoint reload boundaries.

## Out of scope

- Cross-scenario fine-tuning or transfer learning.
- Changing adaptive slot counts or the action/observation encoding.
- Replaying the exact hidden games used for training.
- General player-facing scenario selection outside ML Lab/Arena.
