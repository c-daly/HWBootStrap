# Evidence and validation boundaries

## Current source

The study branch starts at `7048f421697d0fccf96c106d30d889576fc4dcbd`, the head of multiplayer
PR #21 when this review began. It is an isolated worktree; the multiplayer branch and the dirty
root checkout were left intact. This branch adds review documents, an offline HTML/CSS/SVG
concept, and evidence. It does not modify C#, scenes, gameplay rules, package configuration or
training state.

The source review covered title/setup, designer/barracks, movement and attack previews, unit
tooltips and role icons, HUD/end-turn/result/rematch flow, tips/rules, camera-facing presentation,
sound/settings, Steam lobby UI, opponent startup, and combat math. File links in the recommendation
table are the current evidence, rather than the older release audit's inventory.

Some older gaps have advanced: Steam lobby presentation and durable server recovery now exist,
and Unity PlayMode tests have been run on the multiplayer feature. Those local checks still do
not establish a deployed Steam game or an enjoyable first session. The trained-opponent choice
is now a real desktop setup option backed by a Python bridge; the earlier Random/Easy versus
Greedy/Hard summary would be incomplete for this source snapshot.

## Packaged player inspection

Served the repository's `engine/HexWars.NetServer/wwwroot` on `127.0.0.1:8174` and opened it in
headless system Chrome with SwiftShader at 1600×900. Inspected the title, opened Play vs AI,
started the default local match, opened the pause menu and returned to title. This was a visual
and navigation inspection, **not** a completed match or a native-player qualification.

The checked-in WebGL build was last changed at `39c4aa2` on 2026-08-30. Its UI therefore predates
the current Steam integration; it is not a screenshot of `7048f42` compiled into a new player.
The exact inspected loader/framework/data/wasm and page hashes are in
[legacy-webgl-sha256.json](evidence/legacy-webgl-sha256.json).

| Capture | Direct observation | Limit |
|---|---|---|
| [Legacy title](evidence/legacy-webgl-title.png) | A central translucent menu overlays the demo battlefield; navy/blue, built-in font and a generic tactics subtitle dominate. | Legacy route labels differ from the current Steam title source. |
| [Legacy setup](evidence/legacy-webgl-setup.png) | Raw dimensions, seed, army and action-count controls precede the match; a board tooltip remains visible behind the form. | Does not establish whether every current modal has the same hover behavior. |
| [Legacy match opening](evidence/legacy-webgl-match.png) | Nine design stats, five barracks entries and a zero-point bank appear together; the board occupies a relatively small central area. | One default setup and camera state; no claim about every map or resolution. |

The old build logged 446 missing-component errors during the observation, involving
SphereCollider, CapsuleCollider and MeshCollider. The [message counts](evidence/legacy-browser-errors-summary.json)
are retained; the complete repeated log is compressed locally at
`Library/PolishReview-20260924/legacy-browser-errors.json.gz`. A stripping/package issue is a
plausible follow-up hypothesis, not an established root cause. Do not use these captures as
release-quality evidence. This review did not repair or rebuild that artifact.

Audio files and routing were inspected in source, but a listening/mix evaluation was not performed.
No sound-quality judgment or claim about how the supplied clips were produced is made.

## Concept validation

Open [concept/index.html](concept/index.html) directly in a browser. It requires no server,
installation, external fonts, downloaded images, tracking, credentials or Steam connection.
The SVG geometry is authored for this study. All names and positions are illustrative.

Screenshots:

- [The Workshop](evidence/concept-workshop-desktop.png)
- [Field Atlas](evidence/concept-atlas-desktop.png)
- [Orbital Foundry](evidence/concept-orbital-desktop.png)
- [Workshop design comparison](evidence/concept-workshop-design.png)
- [1280×720 layout](evidence/concept-workshop-720.png)
- [390×844 layout](evidence/concept-workshop-mobile.png)

The captured pages include review controls above the interface and explanatory material below it;
their full-page height exceeds the viewport. These are not proposed full-screen Unity dimensions.
Small-screen stacking makes the **study** readable; it does not establish phone, controller or
Steam Deck gameplay support.

The [browser receipt](evidence/concept-checks.json) records seven successful verification groups:

1. Three direction controls and captured desktop layouts.
2. Damage/remaining-health comparison for the two documented fixture states.
3. Three template totals, increment/decrement controls, health floor, upper input clamp and all
   nine stats. The study's upper input limit of 12 is an interface-demo bound, not a game rule.
4. Three teaching moments and return to the beginning.
5. No horizontal overflow across 27 combinations of three directions, three screens and three
   viewports: 1280×720, 2560×1440 and 390×844. The main captures also use 1600×900.
6. Keyboard activation of the screen selector and reduced-motion preference handling.
7. Zero page/console errors and zero HTTP(S) requests from the concept.

The first verification script expected lowercase rendered lesson text despite CSS uppercase
styling; it was corrected to check DOM text, then the full browser verification passed. The
concept needed no behavior change for that assertion. This is a scoped browser check, not a
comprehensive accessibility audit. Production font scaling, contrast, controller interaction,
animation and built-player performance still need their own acceptance work.

JavaScript syntax and local document links were also checked. No .NET or Unity suite was rerun:
this branch changes no runtime game code. Earlier multiplayer test results remain documented in
their own report; they are not counted as validation of these proposed polish features.

## Product hypotheses

The Workshop preference, first-session priorities, effort sizes and player-study thresholds are
design judgments supported by the observations above. They are not measured outcomes. No external
player cohort, willingness-to-pay test, wishlist experiment, Steam launch or revenue analysis was
performed. Steam references in [the recommendations](recommendations.md) link to the official
storefronts/documentation checked on 2026-09-24 and are used for bounded design lessons.
