# Current: cleaner tactical interface and machine units

The [implemented tactical interface](gameplay-visual-language.md) gives the board more space,
keeps the squad compact, and places full stats and combat arithmetic behind **Unit details**.
The [latest control pass](smooth-movement.md) makes movement a single click with **Undo move**,
keeps current targets visible during route previews, and lets squad cards skip spent units.
Attacks retain a damage preview, with no double-click timing requirement. The eight existing appearances now use recognizable
machine forms, and the dark palette and H-in-hex mark remain.

The latest pass adds distinct team hulls and portraits, mirrored backline starts, and automatic
range markers. Biome colors and decoration are hidden while biome rules are disabled.
Enable **Place starting units** in setup to arrange each army anywhere in its starting area before
pressing **Ready**. Escape immediately dismisses help and the designer; coaching is smaller and brief.

The [tabletop sound pass](tabletop-audio.md) adds quiet mechanical cues, shorter weapon tails and separate
effects, ambience and music controls. With the preview server running,
[audition the changes](http://localhost:8196/sound-study/).

Open the [WebGL player instructions](webgl-preview.md) or the
[Windows game launcher](<Open Tactical Game.cmd>). The [Windows build notes](windows-preview.md)
and [latest validation receipt](evidence/polish-integration-checks.json) describe what was tested.

The [integration review](integration-review.md) records the final audio, Escape and coaching fixes.

The [original tactical study](concept/tactical.html) remains an illustrative browser fixture.
The material below records earlier explorations and does not override the current direction.

# HexWars player polish study

**Branch:** `codex/player-polish-20260924`

**Baseline:** `7048f42`, the multiplayer feature in PR #21

**Study status:** historical proposals and interactive references; the playable implementation is linked above.

## Latest: dark unit workshop

Following your preference for the dark theme and simpler, more elegant units, start with the
**[new interactive unit workshop](concept/units.html)**. It has eight distinct forms, porcelain and
graphite finishes, automatic or explicit art selection, board/scale previews, and saved designs
that retain their art. [Windows launcher](<concept/Open Unit Workshop.cmd>).

See the [unit design decisions and integration notes](unit-art-direction.md), or compare the
[porcelain collection](evidence/unit-collection-porcelain.png) and
[graphite collection](evidence/unit-collection-graphite.png). The dark direction now takes priority
over the initial warm-palette recommendation below.

## Initial exploration

Start with the [recommendations](recommendations.md). Open [the visual study](concept/index.html)
directly in a browser to compare three directions, inspect a damage preview, and adjust a unit's
point allocation. It uses local HTML, CSS and SVG, requires no packages or network access, and
does not connect to Steam or the game engine. Its example names, stats and maps are illustrative.

The initial recommendation was **The Workshop**: keep the floating hex battlefield, make the pieces feel
like deliberately constructed objects, and make designing a counter the game's central promise.
Pair that visual direction with a guided first encounter and an exact attack forecast before
expanding the content offering.

The document includes **24 ranked suggestions**, three art directions, concrete copy examples,
a first playable slice, effort/dependency notes, official Steam references and a proposed player
study. The concept includes interactive damage comparisons, three editable example designs and
a three-step teaching sequence. Its browser controls and desktop/mobile layouts were checked.

If opening HTML is inconvenient, compare the captured [Workshop](evidence/concept-workshop-desktop.png),
[Atlas](evidence/concept-atlas-desktop.png) and [Foundry](evidence/concept-orbital-desktop.png) studies.

The review uses current source plus a hands-on look at the checked-in WebGL build from commit
`39c4aa2` (2026-08-30). That build predates the Steam client integration. See the
[evidence and validation notes](evidence.md) for the precise observation boundary.
