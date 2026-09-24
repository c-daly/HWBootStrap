# HexWars player polish study

**Branch:** `codex/player-polish-20260924`

**Baseline:** `7048f42`, the multiplayer feature in PR #21

**Status:** proposals and an interactive visual study; no Unity gameplay changes

Start with the [recommendations](recommendations.md). Open [the visual study](concept/index.html)
directly in a browser to compare three directions, inspect a damage preview, and adjust a unit's
point allocation. It uses local HTML, CSS and SVG, requires no packages or network access, and
does not connect to Steam or the game engine. Its example names, stats and maps are illustrative.

My recommendation is **The Workshop**: keep the floating hex battlefield, make the pieces feel
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
