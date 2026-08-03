# Tactical-v2 Training Roster and Unit Count Design

**Date:** 2026-07-24

## Goal

Let ML Lab users choose the number of starting units for both `tactical-v2`
and `adaptive-v1`. Configurable tactical training draws those units from the
chosen player's available unit templates: the canonical defaults plus every
valid template that player has saved.

Adaptive training keeps its existing deployment behavior and its existing
1–24 starting-unit range.

## Contract Version

This feature is the starting-army implementation slice of `tactical-v2` and
refines `2026-07-23-tactical-v2-viewer-design.md`. Tactical-v1 cannot support
this feature without silently changing existing checkpoint meanings: its codec
assumes one roster entry is simultaneously one template role, one starting
unit, one controllable slot, and one deploy choice.

Tactical-v1 remains available for existing checkpoints and recorded runs but
does not gain configurable armies. New tactical experiments default to
`tactical-v2`, and checkpoints never cross those environment versions.

For this slice, tactical-v2 uses a snapshotted template selection space, one
starting-unit count, and seeded symmetric random composition with automatic
placement. Manual asymmetric compositions and exact placement remain outside
this slice.

## Current Problem

`tactical-v1` always builds `EnvConfig.DefaultRoster()`, which contains three
hard-coded units. ML Lab only renders `Starting unit count` when the selected
environment is `adaptive-v1`.

The tactical codec also uses one value, `Roster`, for three different concepts:

- available template roles;
- starting controllable unit slots; and
- deployable barracks template indices.

That coupling worked only while all three counts were exactly three. Duplicating
templates to obtain larger armies would create redundant observation planes and
duplicate deployment actions, so the counts must be separated.

## User Experience

The Train tab exposes `Starting unit count` for tactical-v2 and adaptive-v1.

- Tactical-v2 accepts 1–12, matching the regular game's current army-size limit.
- Adaptive-v1 accepts 1–24, preserving its current contract.
- Tactical-v2 exposes a roster source that identifies the player roster to use.
  The source is never implicitly hard-coded to local player 0 or 1.
- The tactical roster preview lists the normalized selection space: canonical
  defaults plus all valid templates saved by the selected player.
- Refresh reloads the selected player's current session roster.
- Validation prevents launch when the selected roster is unavailable or empty.

Changing either the tactical unit count or template pool updates preflight
contract dimensions and identity before launch.

## Roster Source and Snapshot

The current saved-template mechanism is `SessionBarracksCache`, with one
process-lifetime catalog per local player. ML Lab lets the user choose which
available local-player catalog supplies a tactical-v2 experiment.

Unity obtains a snapshot of the chosen player's current saved-template catalog.
`BarracksCatalog.Normalize` remains the authority for sanitizing, validating,
deduplicating, and limiting that catalog. Canonical defaults are included even
when the selected cache has been edited; saved templates follow the defaults in
stable catalog order.

ML Lab copies the normalized templates into the working training scenario.
When training starts, the exact templates and starting-unit count are serialized
into the session scenario and then into the run-local immutable
`scenario.json`. Headless Python and GymServer processes never read live Unity
session state.

Resume and Arena use the run-local scenario snapshot. Later changes to a
player's saved roster cannot change an existing run.

The standalone CLI continues to work without Unity. Checked-in tactical-v2
templates contain the canonical default catalog explicitly; custom scenario
files may provide a different normalized catalog.

This slice does not add persistence to `SessionBarracksCache`.

## Tactical Episode Construction

The tactical-v2 configuration separates:

- `Templates`: the ordered, normalized template selection space;
- `StartingUnitCount`: the number of units initially created per seat; and
- `MaxControllableUnits`: the stable unit-slot capacity, equal to
  `StartingUnitCount` in this slice.

At reset, a seed-derived RNG samples `StartingUnitCount` templates from the
scenario's selection space. Sampling uses the regular game's current random-fill
semantics, including sampling with replacement. Both seats receive the same
sampled composition and deterministic placement order, preserving the regular
game's symmetric-army fairness and episode reproducibility.

The random army composition may change between episode seeds. The ordered
template selection space and action/observation meanings do not.

## Tactical Observation and Action Contract

Template roles and unit slots become separate layout dimensions.

- Observation role planes remain keyed to the ordered template catalog.
- Friendly and enemy units are encoded on the plane matching their template
  stats.
- Alive-unit global values normalize against `StartingUnitCount`.
- Move and attack regions allocate `MaxControllableUnits` unit slots.
- Deploy actions allocate one region per available template, not one region per
  starting unit.
- A dead unit releases its stable slot for a later reinforcement.

The action count is:

`1 + (2 × max controllable units + template count) × board cells`

The observation count is:

`(2 × template count + 1) × board cells + tactical globals`

Different counts, template ordering, names, or stats produce distinct contract
and encoding identities. Checkpoint loading rejects every mismatch.

## Scenario Schema and Validation

The versioned tactical-v2 scenario contains:

- `starting_unit_count`;
- `max_controllable_units`;
- the ordered unit-template catalog, including stable ID, name, and all nine stats;
- the automatic placement policy name and version.

Validation is enforced independently in Unity, Python, and GymServer:

- tactical-v2 count is between 1 and 12;
- `max_controllable_units` equals the starting count for this slice;
- template catalog is present, non-empty, normalized, and within the protocol
  template limit;
- template IDs are present and unique;
- every template has valid health and serializable stats;
- the deployment zone has enough cells for the requested count;
- adaptive scenarios reject tactical-only configuration;
- tactical-v1 scenarios reject tactical-v2 configuration;
- tactical-v2 scenarios reject adaptive-only configuration.

Checked-in tactical-v2 templates contain the canonical default catalog. Strict
JSON parsers continue rejecting missing or unknown schema fields.

## Data Flow

1. ML Lab selects tactical-v2, a game template, player roster source, and
   starting-unit count.
2. Unity normalizes and copies the selected player's defaults-plus-saved
   templates into the working scenario.
3. Unity preflight converts the working scenario to the engine contract and
   displays the resulting dimensions and identity.
4. Start writes the complete session scenario and passes it to the Python CLI.
5. Python validates and copies the canonical scenario to the new run directory.
6. Every worker passes the same run-local scenario to GymServer.
7. GymServer validates the scenario and builds the tactical-v2 environment.
8. Each reset samples a symmetric starting composition from the snapshotted
   template pool using that episode's seed.

Adaptive-v1 follows its existing data flow with its own starting-unit count.

## Error Handling

ML Lab disables Start and shows an inline validation error when the roster
source cannot be read, normalization yields no templates, the count is outside
its environment's range, or the board cannot place the requested units.

CLI and GymServer failures name the exact invalid field. A failed validation
does not create or partially overwrite a run. Existing runs remain immutable.

## Testing

Engine tests cover:

- tactical-v1 dimensions and hashes remain unchanged;
- deterministic sampling for identical seeds;
- differing sampled compositions across a representative seed set;
- identical compositions for both seats;
- counts 1 and 12;
- sampling with replacement;
- saved/custom template participation;
- template count independent from unit-slot count;
- mask, encode, and decode boundaries for move, attack, and deploy regions;
- distinct hashes for changed counts or catalogs;
- slot release and deterministic reinforcement reassignment;
- insufficient deployment-zone rejection.

Unity EditMode tests cover:

- tactical-v2 and adaptive-v1 count controls;
- tactical-v1 remains explicitly legacy and fixed;
- roster-source selection without a hard-coded player;
- defaults plus saved templates in the preview and working scenario;
- scenario snapshot round-trip;
- launch disabled for invalid or unavailable rosters;
- resume using the immutable source-run scenario.

Python and process tests cover:

- strict tactical-v2 schema validation;
- checked-in tactical-v2 templates;
- custom roster round-trip into GymServer;
- run-local scenario preservation across workers;
- contract handshake dimensions and hashes;
- tactical-v1 checkpoint compatibility remains unchanged.

After every C# change, Unity compile errors are checked. The focused engine,
Unity EditMode, Python, GymServer process, and determinism-sensitive PlayMode
tests run before completion.

## Non-Goals

- Persisting `SessionBarracksCache` beyond its existing process lifetime.
- Changing the regular game's army-size limit or random-fill behavior.
- Teaching adaptive agents to complete deployment.
- Loading tactical-v1 checkpoints into tactical-v2.
- Manual per-role counts, asymmetric armies, or exact placements in this slice.
- Implementing the tactical-v2 viewer presentation work described in the
  broader 2026-07-23 design.
