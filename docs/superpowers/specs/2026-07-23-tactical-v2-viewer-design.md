# Tactical-v2 Setup and Faithful ML Viewer Design

**Date:** 2026-07-23
**Status:** Approved design

## Objective

Make new tactical experiments configurable and reproducible while making the Unity ML viewer easy to follow.

The work has two independent implementation slices:

1. Add a `tactical-v2` environment contract with configurable starting armies and placements.
2. Present every authoritative game command in an omniscient, paced Unity viewer with point totals and an optional fog-of-war indicator.

Headless training remains unrestricted by viewer pacing. Model observations and action masks retain their configured fog-of-war limits.

## Why tactical-v2 Is Required

The `tactical-v1` codec assumes that one roster entry is simultaneously:

- one distinct unit role;
- one starting unit;
- one controllable unit slot; and
- one barracks template.

That encoding cannot represent repeated unit types, unequal starting counts, or independently reusable reinforcement slots without ambiguity. Changing those semantics in place would silently change the meaning of existing checkpoints.

`tactical-v2` separates template roles from controllable unit slots. `tactical-v1` remains available for existing checkpoints and recorded runs. New tactical experiments use `tactical-v2` by default.

## Tactical-v2 Contract

### Template catalogue

Each scenario defines a non-empty template catalogue. Every template has:

- a stable, scenario-unique ID;
- a display name; and
- all nine unit stats.

Observation role planes and reinforcement choices use template IDs. Repeating a template in a starting army does not duplicate its observation role plane.

### Controllable slots

Each scenario defines `max_controllable_units` per player. Movement and attack actions address stable unit slots, while deployment actions address template IDs.

The action layout is:

1. End turn;
2. move: controllable slot × board cell;
3. attack: controllable slot × board cell; and
4. deploy: template × board cell.

The observation layout uses one friendly and one enemy plane per template, plus the existing board and scalar data. Multiple units of the same template share the corresponding role plane at their distinct cells.

When a unit dies, its slot becomes available. A reinforcement receives a deterministic free slot and is controllable on later decisions. Slot allocation is identical in training, GymServer, and Unity duels.

### Starting armies

The scenario stores explicit P1 and P2 starting-unit instances. Each instance references a template ID and, in exact mode, a board column and row.

The two armies may:

- contain repeated templates;
- have different compositions;
- have different unit counts; and
- use independent positions.

Neither army may exceed `max_controllable_units`.

### Placement modes

`automatic` placement deterministically assigns the declared unit instances to valid cells in the owning deployment zone. The same scenario, placement-policy version, and episode seed produce the same board and placement. Automatic placement is resolved independently for every episode rather than frozen to the seed used by the first game.

`exact` placement requires a coordinate for every starting unit. Coordinates use rectangular board column and row values; the engine converts them to axial coordinates. Each coordinate must:

- exist on the generated board;
- be in the owning player's deployment zone; and
- be unique within the starting state.

Elevation comes from the generated board tile.

### Mirroring

The editor provides **Mirror P1 to P2**, enabled by default.

When enabled:

- P2 composition follows P1 composition; and
- exact P2 positions are the 180-degree board-centre rotation of P1 positions.

When disabled, P2 composition and positions become independently editable.

The resolved run snapshot always writes both final army lists. In exact mode it also writes every final coordinate, including generated mirrored coordinates. In automatic mode it writes the placement-policy name and version; workers reproduce coordinates from that policy and each episode seed. The `mirror` flag remains as provenance, but workers consume the explicit resolved army data. A run therefore does not depend on future editor mirroring behavior.

## ML Lab Scenario UI

The existing game-settings editor gains a **Tactical setup** section when `tactical-v2` is selected.

### Template editor

The template table supports:

- add and remove;
- stable ID;
- display name; and
- numeric fields for all nine stats.

Validation is inline and does not use modal input prompts.

### Composition editor

P1 composition rows contain a template dropdown and numeric count. The UI expands counts into starting-unit instances when resolving the scenario.

With mirroring enabled, P2 shows a read-only mirrored summary. Disabling mirroring exposes the same editable controls for P2.

### Placement editor

The placement selector offers **Automatic** and **Exact**.

Exact mode shows one row per expanded unit instance:

- unit label;
- template;
- column; and
- row.

Mirrored P2 rows are read-only. Independently configured P2 rows are editable.

The editor reports duplicate cells, cells outside the deployment zone, missing placements, unknown templates, slot-limit violations, and other schema errors beside the relevant section. Invalid scenarios cannot be saved as templates or launched.

## Run Data and Compatibility

The scenario snapshot is written into the run directory before worker launch. Run metadata records:

- environment and contract version;
- scenario ID and schema version;
- board configuration;
- template catalogue and role order;
- maximum controllable slots;
- both resolved starting armies;
- exact coordinates or the automatic placement-policy name and version;
- observation and action sizes;
- contract hash; and
- encoding hash.

Checkpoint selection rejects mismatched environment versions, board/layout dimensions, template ordering or stats, controllable-slot counts, and encoding hashes.

`tactical-v1` remains loadable and viewable. It does not gain the new configurable-army controls. Existing adaptive contracts and checkpoints remain unchanged.

## Authoritative Viewer Transitions

Every duel environment exposes the accepted command transitions produced by a step. A transition contains:

- previous immutable game state;
- accepted command; and
- resulting immutable game state.

The transition list includes commands auto-played by scripted Greedy or Random controllers. Invalid actions do not produce presentation transitions.

The Unity viewer consumes these transitions directly. It does not infer attacks or movement from a coarse before/after diff.

## Viewer Playback

The Unity viewer is an omniscient spectator:

- every unit is rendered;
- every accepted move is animated;
- every accepted attack shows its projectile, impact, damage, and kill effects;
- deployments, captures, generator construction, and point awards use the existing gameplay presentation;
- transitions remain ordered; and
- each visible animation finishes before the next transition begins.

The viewer does not request another policy action while presentation transitions are queued. This pacing affects only the separate Unity viewing duel. The headless training process continues independently at full speed.

When tactical-v2 fog of war is enabled, observation role planes omit enemy units outside the acting model's current visibility. Legal-action masks remain derived from the same fog-constrained game rules. The omniscient presentation state is never passed into policy inference.

The presentation layer tracks a `PresentedState`. Board entities, point totals, active-player indicators, and event text read from that state rather than from a simulation state that may already be several commands ahead.

## Player Point Totals

The existing P1 and P2 identity rows continuously display each player's points. Values update when the corresponding presented transition completes.

The rows continue to show:

- active player;
- learner/opponent role;
- controller and algorithm;
- checkpoint;
- run step;
- win/loss record; and
- status.

Portrait and landscape layouts retain their existing truncation behavior.

## Fog-of-War Indicator

The viewer adds **Fog overlay: Off / P1 / P2**.

- **Off** is the default omniscient presentation.
- **P1** shades cells outside P1's current visibility.
- **P2** shades cells outside P2's current visibility.

The viewer remains omniscient in all three modes. Units hidden from the selected model remain present but receive a clear dimmed or marked treatment, so the operator can distinguish “spectator can see this” from “selected model can see this.”

The overlay is computed from `PresentedState` using the engine's authoritative visibility rules. It advances with the animation queue. It never changes observations, masks, policy inputs, or simulation state. When fog of war is disabled, the selector reports that all cells are visible.

## Validation and Failure Behavior

A tactical-v2 run fails before worker launch if:

- template IDs are missing or duplicated;
- template stats are invalid;
- a composition references an unknown template;
- counts are negative or an army is empty;
- an army exceeds the controllable-slot limit;
- exact positions are missing, duplicated, outside the board, or outside the correct deployment zone;
- mirrored resolved data does not match the expected rotation; or
- scenario and checkpoint contract metadata differ.

Errors identify the scenario field and player responsible. Unity and Python must report the same invalid condition in user-readable terms.

The viewer stops with an explicit presentation error if an authoritative transition cannot be queued or rendered. It must not silently skip a command and continue with misleading visuals.

## Testing

### Engine and contract tests

- Preserve `tactical-v1` action, observation, and checkpoint semantics.
- Verify tactical-v2 action and observation dimensions.
- Verify repeated templates share role planes without sharing unit slots.
- Verify asymmetric starting armies.
- Verify deterministic automatic placement.
- Verify exact and mirrored placement.
- Reject invalid coordinates, duplicate cells, unknown templates, and slot overflow.
- Verify slot release after death and deterministic reinforcement slot assignment.
- Verify tactical-v2 fog-limited observations.

### Cross-runtime tests

- Parse and validate the same tactical-v2 scenario in Unity, GymServer, and Python.
- Assert matching contract and encoding hashes.
- Assert matching resolved armies, placements, action sizes, and observation sizes.
- Snapshot the exact scenario into run metadata and reload it for Arena viewing.

### Viewer tests

- A scripted opponent command batch produces one ordered transition per accepted command.
- Every attack transition reaches the animation queue.
- The next policy action waits until queued presentation completes.
- Point totals follow `PresentedState`.
- Omniscient rendering includes units hidden from P1/P2 observations.
- P1 and P2 fog overlays match engine visibility without hiding spectator information.
- Adaptive deployment remains hidden until its atomic reveal, after which omniscient playback begins.

### Verification

Before completion:

- run engine and GymServer .NET tests;
- run the complete Python ML suite;
- run Unity EditMode tests;
- run focused Unity PlayMode viewer tests;
- confirm Unity has no compilation errors; and
- manually watch a tactical-v2 duel containing attacks, point awards, asymmetric armies, exact placement, and both fog overlays.

## Delivery Order

1. Tactical-v2 engine contract and scenario schema.
2. Python/GymServer parsing, handshake, metadata, and training support.
3. ML Lab tactical setup editor and validation.
4. Authoritative transition capture.
5. Paced omniscient viewer playback and point totals.
6. Fog overlay selector.
7. Documentation, migration notes, and end-to-end verification.
