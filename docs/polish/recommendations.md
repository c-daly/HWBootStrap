# Make designing the counter the reason to play

**Direction update:** the owner prefers the dark theme and more elegant unit forms. The
[new unit-art study](unit-art-direction.md) develops that direction and demonstrates art selection
in the creator. It supersedes the warm visual recommendation in this initial comparison; the
prioritized UX proposals below remain available for review.

**Review date:** 2026-09-24. **Proposal baseline:** `7048f42`. All changes below are suggestions,
not implemented product features. Default audience hypothesis: approachable tactics with enough
depth for competitive play. There is no player-research or sales evidence yet for that positioning.

HexWars has a better hook than its current presentation communicates: you can build an odd little
specialist, give up something important to afford it, and win because it was the right tool for
this particular board. The point economy, deterministic combat, shared vision, elevation and
editable unit templates all support that promise. Lead with **“Design the army. Find the counter.”**

The strongest Steam-facing product hypothesis is a compact tactics workshop with a satisfying
solo starting point and friend matches. A game that only feels worthwhile when public matchmaking
is busy has a harder first-player problem. Start by proving that building a counter is enjoyable
against a reliable local opponent. Multiplayer then extends that experience.

## What is actually here

This is an extension of the existing game, not a request to replace it. The source links below
refer to this branch. The [evidence notes](evidence.md) distinguish source observations from the
older packaged build and design judgments.

| Surface | Existing foundation | Opportunity |
|---|---|---|
| Entry | [TitleScreen](../../Assets/HexWars/Presentation/TitleScreen.cs) has a live demo, local/AI/online routes and rules. Steam puts Quick Match first. | Give a first-time player one reliable way to learn the hook; make customization secondary. |
| Setup | [SetupForm](../../Assets/HexWars/Presentation/SetupForm.cs) exposes dimensions, seed, army, action count, fog and opponent. | Offer curated presets with a board preview before the raw configuration form. |
| Design | [DesignPanel](../../Assets/HexWars/Presentation/DesignPanel.cs) has nine point-buy stats, names, role feedback, example designs and stat help. | Turn numeric editing into understandable trade-offs and an immediate board experiment. |
| Battlefield | [UnitInputController](../../Assets/HexWars/Presentation/UnitInputController.cs), [AttackPreviewTargets](../../Assets/HexWars/Presentation/AttackPreviewTargets.cs) and movement controllers already show routes and legal targets, including targets after a proposed move. | Add exact damage and explain why a tempting action is unavailable. Do not rebuild existing reach previews. |
| Pieces | [TokenStore](../../Assets/HexWars/Presentation/TokenStore.cs) uses discs, role symbols and HP bars. [RoleIcons](../../Assets/HexWars/Presentation/RoleIcons.cs) distinguishes eight roles. | Preserve that useful symbol language; give pieces a recognizable silhouette and clearer readiness state. |
| Teaching | [TipsService](../../Assets/HexWars/Presentation/TipsService.cs), stat help, a rules reference and the first-bounty “Design your answer” prompt exist. | Arrange those assets into a short encounter where the player discovers a counter by doing it. |
| Results | [GameOverBanner](../../Assets/HexWars/Presentation/GameOverBanner.cs) supports board inspection and a local vs-AI rematch. [GameBootstrap](../../Assets/HexWars/Presentation/GameBootstrap.cs) rematches with a new seed. | Offer retrying the same position, editing a design and an agreed friend rematch as distinct choices. |
| Sound | [SoundManager](../../Assets/HexWars/Presentation/SoundManager.cs) layers procedural effects with 17 supplied clips, including music and weapon variants. | Tune the mix, timing and meaning of cues. Do not assume existing audio needs replacing or infer its provenance from its sound. |
| Persistence | [SessionBarracksCache](../../Assets/HexWars/Presentation/SessionBarracksCache.cs) deliberately lasts only for the session. Online durable recovery is now locally tested. | Save named designs and a local match; online server durability alone is not a local Continue button. |
| Opponents | [SetupForm](../../Assets/HexWars/Presentation/SetupForm.cs) labels the desktop choices “AI: Greedy” and “AI: Trained model.” [AiOpponent](../../Assets/HexWars/Presentation/AiOpponent.cs) starts the trained policy through a separate Python bridge. | Ship dependable player-facing difficulty; keep experimental opponents in an explicitly separate developer route until packaged and qualified. |

## Three coherent visual directions

The [interactive study](concept/index.html) compares palette, typography, surface treatment and
information hierarchy on the same illustrative board. It is a direction study, not final art or
a working Unity implementation. Its shared geometry deliberately makes those choices comparable.

| Direction | Visual and audio vocabulary | Why it fits | Cost and risk |
|---|---|---|---|
| **A. The Workshop — recommended** | Warm ivory, charcoal, oxidized teal and vermilion; printed labels; ceramic or painted metal pieces; precise mechanical clicks and restrained room tone. | Makes point allocation feel like building an object. Offers a strong small-scale identity without requiring a large world or character cast. | Medium. Needs coherent material/lighting work and a small original piece set. Keep the floating battlefield visible so it retains HexWars' identity. |
| **B. Field Atlas** | Parchment, forest green, rust; contour bands and survey marks; tactical counters; pencil-like route lines and dry percussion. | Supports legibility, elevation and thoughtful competitive play. Fits the existing discs and symbols with the smallest art expansion. | Small–medium. Can read as a utility/map editor unless pieces, motion and a distinctive wordmark provide warmth. |
| **C. Orbital Foundry** | Graphite, chalk, safety amber and ice blue; modular machines; landing pads, altitude marks, a dark orbital void; motor and relay sounds. | Extends the existing floating-in-space setting and makes unit construction literal. | Large if modeled machines and animation are required. Avoid filling every empty space with glowing panels and tiny technical labels. |

Choose one visual grammar before buying or producing a large asset set. A minimum direction test
is one map, three readable pieces, the unit designer and one complete attack sequence. View it at
normal play scale and thumbnail scale, with the logo hidden. Ask what the game is about and what
the pieces do; do not ask whether participants think AI made it.

## What makes it feel deliberately made

“AI-generated” is not a visual diagnosis. The observable risks here are generic copy, equally
weighted controls, weakly distinguished materials, and an interface that reveals configuration
before helping the player make a decision. A consistent, edited game can use procedural assets.

- **Build an identifiable visual vocabulary.** Keep the role symbols, select a consistent stroke
  weight and contrast, and repeat them on pieces, workshop examples and attack forecasts. A role
  is a summary of a stat allocation, not a new class with hidden bonuses. Custom hybrids must remain
  legible; never let a silhouette promise range or armor the stats do not have.
- **Spend detail where the player looks.** Distinguish the selected piece, its target and height
  difference. Simplify stars, particles and nonessential panel chrome. The existing WebGL scene
  gives much of the screen to empty space while nine-stat editing and barracks occupy the sides.
- **Write about choices and consequences.** Prefer a specific drawback to three heroic adjectives.
  For a zero-movement sniper, “Long reach. Cannot reposition.” tells the player something useful.
  Compute claims from stats or validated authored templates; do not generate tactical assertions
  from a language model during a match.
- **Use a small, intentional type system.** One expressive title treatment, a highly readable UI
  face, and aligned numerals. The current `UiKit` uses Unity's built-in legacy font. Test weights,
  long names and real sizes before procuring fonts; the study uses installed system fonts only.
- **Let actions land.** Selection, movement, damage, defeat and income should have different visual
  and sonic signatures. Keep exact state changes readable at fast animation speed. Review supplied
  audio by listening in context before deciding what additional recording is needed.
- **Retain useful restraint.** No invented lore paragraphs in tooltips, constant victory sparkles,
  random unit barks, or extra currencies solely to make the game appear larger.

### Copy examples to test

| Existing wording | Proposed wording or treatment | Reason |
|---|---|---|
| “hex-grid tactics — design an army, take the field” | “Design the army. Find the counter.” | Leads with the action that differentiates the game. Test comprehension, not just preference. |
| “Play vs AI” | “Solo skirmish”; first launch leads with “Learn by playing” | Describes the experience and separates learning from configuration. |
| “AI: Greedy” / “AI: Trained model” | A qualified “Standard” opponent with a brief behavioral description | Names must follow actual difficulty testing; relabeling alone does not improve the opponent. |
| “Create (to Barracks)” | “Save design”; show “Deployment: 17 points” separately | Separates creating a template from paying to field a unit. Respect configurable design fees. |
| “Out of movement reach” | “Needs 3 movement; 2 left” or “Cannot climb this height” | Use the authoritative failure reason. Do not guess from distance alone. |
| “Match service not configured” | “Online play is not available in this build.” + “Solo skirmish” | An honest placeholder state with a useful next action; keep diagnostics in support details. |
| “Rematch” for the existing fresh-seed behavior | “New battlefield”; add “Retry this battle” separately | Makes it clear whether the map and initial state will change. |

## Prioritized suggestion bank

Effort is relative to one experienced developer working with this code: **S** is a focused change,
**M** spans multiple surfaces, **L** is a new subsystem or substantial content/asset work. These
are sizing hypotheses, not delivery estimates. “Now” means candidate for the first polish slice,
not authorization to implement this whole table.

| ID / priority | Suggestion and player value | Smallest useful version | Effort / dependency | Evidence to earn the next step |
|---|---|---|---|---|
| P01 Now | **One clear first-session route.** Get to a meaningful move before a settings form. | First-launch Learn by playing; returning players get Solo skirmish and Play with a friend. Custom match retains the existing knobs. | M; curated default encounter | At least 6/8 new testers reach their first useful move unaided within 90 seconds. |
| P02 Now | **Explain the result before the click.** Confidence makes tactical decisions satisfying. | Target card: damage, target HP after hit, elevation/terrain/defense breakdown. Existing route and target highlights stay. | M; authoritative combat and visibility adapters | Every fixture matches engine output; 6/8 testers correctly predict two contrasting attacks. |
| P03 Now | **Fit the battlefield to the available play area.** Make pieces worth looking at. | A Fit board control, saved camera preference, context-driven side panels, stable selected-piece inspector; suppress board hover/tooltip updates behind setup and other modals. | M; camera/panel layout | No key cells obscured at 1280×720, 1920×1080 or 2560×1440; no camera jump during an action or stray tooltip behind a modal. |
| P04 Now | **Curated quick setup.** Help a player choose an experience. | Three verified presets with a map thumbnail, objective and army summary; Advanced opens the existing form. Label duration only after measuring it. | S–M; suitable seeds | Testers can describe how two presets differ without reading the manual. |
| P05 Now | **Design by comparison.** Make the nine stats understandable. | Start from existing templates; show old/new capability and point delta, useful drawbacks, and a tiny range/climb diagram. Keep all nine stats accessible. | M; stat help already exists | 6/8 testers make a purposeful modification and correctly name its cost or lost capability. |
| P06 Now | **Useful unavailable states.** Answer “why can't I?” before frustration. | Show why a unit cannot deploy, climb or attack; distinguish banked points, template cost and spent actions. Hide neither the cause nor the recovery action. | M; legality reasons | Cases cover occupied hex, insufficient funds, spent attack, vertical reach and unseen target without leaking fog state. |
| P07 Next | **Legible pieces and teams.** Identify role, owner and readiness quickly. | Refine existing eight symbols; add a small role silhouette set, team rim patterns and separate move/attack pips. | M; chosen art direction | In grayscale and at game scale, 6/8 testers identify team and relevant role correctly. |
| P08 Next | **A tactical lesson that sells the hook.** Teach by finding a counter. | Three authored encounters: use high ground; spot for a ranged specialist; modify a design to solve a new constraint. Each offers an immediate retry. | M–L; P02/P05, deterministic fixtures | Testers explain one trade-off in their own words and voluntarily try a second design. |
| P09 Next | **Keep the player's inventions.** Named designs are personal investment. | A local saved-design shelf with rename, duplicate, archive and versioned import/export. Preserve templates across restart. | M; persistence/schema validation | Restart and version-migration round trips preserve names/stats; invalid designs get actionable errors. |
| P10 Next | **Resume a local game.** Respect interrupted sessions. | Explicit Save and return to menu plus Continue; include initial setup, journal, opponent identity and version. | L; local persistence and AI resumability | Close/relaunch reproduces board, turn, fog and next legal action. Test older/incompatible saves honestly. |
| P11 Now | **A desktop shell that feels finished.** Basic comfort before more modes. | Title/in-game settings, Quit, resizable window, text scale, motion/flash controls and separate music/effects levels. | M; settings/input | Keyboard-only route through menus; no clipping with enlarged text and long labels; settings survive restart. |
| P12 Next | **The opponent's turn tells a story.** Keep decisions understandable. | Reuse action presentation; configurable speed, skip-to-current-state, and a short “moved / attacked / gained” recap linked to visible cells. | M; presentation queue | Fast mode preserves final state and input gating; fog never reveals unseen actions. |
| P13 Next | **A result screen with a next idea.** Help a loss become another experiment. | Existing result/inspect-board plus three factual events, Retry this battle and Edit army. Later add a mutually accepted friend rematch. | M; replay/setup preservation | 4/8 testers voluntarily start another attempt without a facilitator prompt; retain the unchanged-board inspection route. |
| P14 Next | **Sound with functional hierarchy.** Give important outcomes weight. | Mix the existing clips; distinguish damage absorbed, damage dealt and destruction; duck ambience for turn cues; support independent volumes. | S–M; listening pass, P11 | Players notice the turn cue without staring at the HUD; muted play retains every essential cue. |
| P15 Next | **A dependable solo challenge.** Make difficulty mean something. | Qualify the shipped local opponent, name difficulty from observed behavior, add a separate teaching opponent if needed. Keep research labels and Python setup out of the normal route. | M–L; built-player trials | Clean-machine solo match completes; novices have room to learn and experienced players can identify exploitable weaknesses. |
| P16 Explore | **A small scenario collection.** Offer value when no friend is online. | Six authored battles using existing rules, each about a distinct constraint; optional mastery objectives that do not affect PvP stats. | L; P08/P15 | At least three encounters elicit different strategies; retire ones solved by the same opening. |
| P17 Explore | **Challenge a friend with a design.** Make experiments shareable. | Versioned local design code and a reproducible scenario code; later a results comparison. | M–L; P09, validation and fog-safe export | Recipient reproduces the setup; mismatched engine/config is detected. Do not expose hidden live-match information. |
| P18 Explore | **Weekly workshop challenge.** Give players a common conversation. | An archived, replayable curated seed with a fixed budget; no streak penalty and no new rules each week. | M plus recurring curation | Track distinct returning participants across four trials; continue only if the content effort is justified. |
| P19 Explore | **A visible series with a friend.** Make a session more than an isolated match. | Best-of-three, mutually accepted rematches and alternate first player/map sides. | M–L; live Steam qualification, fairness evidence | Both players understand series state, decline gracefully and recover from interruption. |
| P20 Later | **Mastery achievements.** Encourage discovering the system. | A few achievements for meaningful combinations or completed authored challenges; no repeated-click grind or competitive power unlocks. | M; frozen events, Steam identity | Each achievement teaches or celebrates a recognizable tactic and cannot trigger on an invalid/replayed event. |
| P21 Now | **A distinctive visual opening.** Make one screenshot communicate the game. | Selected piece, reachable route, target forecast and an open comparison of two designs; consistent logo/material treatment. | M; P02/P05/P07 | After five seconds, 6/8 unfamiliar viewers can describe unit construction plus tactics, not just “a hex board.” |
| P22 Next | **Online waiting with honest choices.** Make a small community comfortable. | Explain search/ready/reconnect states, show elapsed waiting, keep Cancel and friend invite clear; offer solo only after safely leaving the queue. | M; existing Steam coordinator | No lost lobby, unexpected auto-join, false population count or hidden bot substitution. |
| P23 Later | **Replay as a learning tool.** Make losses inspectable. | Step through one's own completed match with move/attack annotations and a final-board bookmark. | L; durable journal, visibility, version compatibility | Replayed public information matches what that viewer was allowed to see; mismatched versions fail clearly. |
| P24 Next | **A stronger Steam demonstration.** Let footage prove the promise. | One short trailer scene: inspect threat, modify specialist, deploy, execute counter. A compact solo demo with one optional friend route. | M–L; a qualified native slice | Uncoached players reach and explain the central loop; store footage represents the actual delivered build. |

## First playable polish slice

Build a thin version of **P01 + P02 + P03 + P05**, with the minimum P11 controls needed to test
comfortably. Scope it to one curated local encounter and existing rules. Its demonstration is:

1. Enter Learn by playing. See the objective, your three existing pieces and the board, with no
   need to choose seed, dimensions or turn policy.
2. Select a piece. See movement and attack availability using the current legal-action paths.
3. Preview a target. Read exact damage and one useful explanation. Moving to higher ground changes
   the forecast because the engine changed it.
4. Reach a situation where modifying a saved template solves an understandable problem. The
   existing first-bounty teaching hook is a good insertion point when the encounter grants funds.
5. Finish or retry. Inspect the board, identify what the new design achieved, and immediately try
   the same battle with a different design.

Implementation order: authoritative forecast adapter and fixtures; focused board/inspector layout;
template comparison; authored encounter wrapper; then a coherent art/audio pass on that slice.
Keep defaults and mode rules under the existing configuration. Do not turn a UX experiment into
a balance change, model-training run, or new multiplayer protocol.

### Forecast contract

Use `CombatResolver.ComputeDamage` and the same legality/terrain inputs as `GameEngine.Apply`.
Include damage floor, zero-damage noncombatants, height difference, target defense and terrain.
Target HP display clamps at zero. Mark a rejected action with its real reason; zero damage is not
automatically an illegal attack. No counterattack, critical chance, line-of-sight rule or future
human intent should be invented for presentation. Display only information authorized by the
viewer/fog model, including when previewing a move to a new hex.

The visual study's worked example uses damage 5, one level of high ground at +1, defense 2 and
terrain defense 1: **3 damage**, leaving a target with 5 HP at **2 HP**. Removing the height bonus
makes that **2 damage / 3 HP remaining**. It illustrates the hierarchy; the browser study is not
an engine oracle and must not be copied into Unity as a second combat implementation.

## Making a plausible Steam proposition

Use references as design lessons, not revenue forecasts. Their storefronts do not establish that
copying their features or art direction would sell HexWars.

| Reference / official source, checked 2026-09-24 | Concrete lesson | HexWars-specific application |
|---|---|---|
| [Into the Breach](https://store.steampowered.com/app/590380/Into_the_Breach/) foregrounds telegraphed attacks and counterplay. | A tactics promise can be communicated through a visible decision. | Show known outcomes clearly. Human opponents' next moves remain unknown; do not transplant its perfect enemy-intent information into PvP. |
| [The Battle of Polytopia](https://store.steampowered.com/app/874390/The_Battle_of_Polytopia/) presents a named world, distinct tribes and configurable maps across solo and multiplayer. | Consistent personality gives simple geometry something memorable to represent. | Name authored scenarios and example designs; build a visual vocabulary around construction rather than adding sixteen factions. |
| [Steam graphical asset rules](https://partner.steamgames.com/doc/store/assets/rules) distinguish capsule, library hero and logo content. | A strong identity must work across several surfaces with different constraints. | Test a recognizable piece and readable game name at capsule size. Keep promotional copy out of base capsule art and words out of the library hero. |
| [Steam demo guidance](https://partner.steamgames.com/doc/store/application/demos) provides routes from a demo back to the full game's store page. | The demo should establish the game's value and offer a natural next step. | Put a full-game/wishlist route after the demonstration and in its menu, once real app/store configuration exists. |

My proposed trailer opening is **20–30 seconds as an editing experiment**, not a claimed match
length: reveal an awkward enemy position, show one stat trade-off, deploy the answer, then show
the resulting attack. Use captured gameplay from the qualified build. The browser concept and
legacy WebGL captures in this branch are review material, not finished Steam marketing assets.

Do not choose a price from these references or from test counts. First establish a native solo
slice, the content it includes today, the quality of repeat play, and clean-machine operation.
Actual Steam/Render multiplayer, controller support, Steam Deck claims and local save/resume
remain separate validation work. PR #21's local process tests do not close those gates.

## Player study and decision rules

Recruit a first cohort of eight unfamiliar players, roughly half occasional tactics players and
half experienced players. This is a diagnostic cohort, not a market estimate. Observe individually
with consent, use a fixed encounter and the same instructions, and record assistance rather than
quietly rescuing them. Do not infer commercial demand from a small convenience sample.

Record first useful move time, unexpected outcomes, whether they can explain a design trade-off,
completion/abandonment, and whether they choose another attempt without being prompted. Ask
“What were you trying to do?” and “What do you expect this attack to do?” rather than pitching
the feature. Give time for a real choice to stop; asking “Want another?” biases that measure.

The thresholds in P01/P02/P05/P13/P21 are **proposed stop/go criteria**, not measured results or
industry benchmarks. Agree and freeze them before that cohort; retain every attempt. Repeated
fundamental confusion, hidden-information leakage, a blocked first match, or unrecoverable save
loss stops expansion. Change the relevant slice, document what changed, then test a fresh cohort.
Only after this comprehension pass should a larger demo test examine repeat sessions, wishlists
and demand, with channels and denominators reported.

## Scope to defer

Defer a large campaign, additional currencies, power progression in PvP, a live-service calendar,
ranked seasons, many factions, Workshop integration, generative dialogue and a broad asset
shopping pass. They can all create work without proving the current loop. A curated challenge
collection and saved designs are the stronger early replayability candidates.

The next decision is which visual direction to develop and which small slice to implement.
The recommendation remains The Workshop plus the first-playable slice above; it preserves the
flat point-buy identity while making that identity much easier to see, understand and share.
