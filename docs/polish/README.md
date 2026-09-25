# Latest: gameplay and visual language

The [new tactical interface study](gameplay-visual-language.md) responds to the request for
clearer gameplay cues, a more attractive UI and less abstract units. Open the
[interactive preview](concept/tactical.html) or the
[Windows launcher](<concept/Open Tactical Study.cmd>). It explores a focused unit panel,
movement/attack forecasts and eight recognizable machine forms. This is a browser design
study; the current playable Unity build below is unchanged.

# Playable graphite update

The graphite direction is now implemented in Unity. Start with the [WebGL build](webgl-preview.md)
or [Windows preview](windows-preview.md) for the playable art picker and validation boundaries.
The studies below document the earlier exploration.

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
