# Invite-Readiness — Design

**Date:** 2026-07-08
**Milestone:** Everything that must ship before inviting people to play: the game survives real-world
phone usage (reconnect, portrait, long sessions), looks legitimate when shared (link previews, PWA),
and delivers the core delight — *surprised by the simplicity of the one-point economy, then impressed
that the ways to assemble those points are effectively infinite, and that judicious choices dominate.*
**Play pattern:** organized friend sessions first, drop-in via shared link as the goal. Hosting will be
self-hosted or a paid tier (decided separately), so the server is assumed not to sleep — cold-start
mitigation is out of scope.
**Sequencing:** one milestone, hardening tasks first — after the hardening half lands, the build is
invite-safe (an early-invite fallback point); the delight half completes the bar this spec sets.

## 1. Problem

Three gaps stand between the current build and inviting people:

1. **Fragility on phones.** Backgrounding the tab drops the WebSocket and the game silently dies —
   no message, no retry, no rejoin (audit U2). Seats are keyed by per-socket GUIDs, so even a manual
   refresh cannot reclaim a game (audit M3); an emptied room is deleted instantly, so a double drop
   destroys the match. Sessions degrade: HP bars leak two Materials per unit per sync (audit P1),
   the tooltip allocates every frame (P2), and the event console allocates every OnGUI call and
   covers ~86% of a portrait screen (F1/U1). Several screens were designed landscape-first.
2. **Naked link.** Pasting the URL into a chat produces no preview card; there is no icon, no
   manifest — it reads as a dev artifact, not a game.
3. **The depth is invisible.** The unit designer — the game's best idea — is never introduced. The
   barracks starts empty, the help screen never mentions designing units, nothing marks the moment
   a player first holds bounty points. Nominal roles (Brute/Striker/Sniper) read as fixed classes
   when they are just example allocations. A newcomer can finish three games without discovering
   that a statline is anything they can imagine.

## 2. Goals and non-goals

**Goals**

- **Reconnect and rejoin.** A dropped socket (tab background, network blip, refresh) recovers into
  the same seat and the same game automatically. Started games survive short full-disconnects.
- **Server hardening** (audit follow-through): per-connection ordered sends, incoming message cap,
  origin check, tolerant command parsing.
- **Session longevity:** the known per-session leaks and per-frame allocators fixed so a phone can
  play games back-to-back without progressive stutter.
- **Portrait-phone pass:** every screen usable at ~390 CSS px width; the event console stays out of
  the way on portrait.
- **Shareable:** OpenGraph/Twitter preview card, favicon, PWA manifest ("Add to Home Screen").
- **The delight arc, mechanized:**
  - *Starter templates:* every game's barracks comes pre-loaded with named example designs —
    deployable turn one, deletable per-game, back next game. Examples are chosen to teach that a
    statline is a concept ("Artillery" = Movement 0 + long range).
  - *Named designs:* players name what they create; the name lives on the template and on every
    unit deployed from it (tooltip shows "Artillery", not just a role icon).
  - *Stat descriptions, always available:* tap/hover any stat name in the Designer or unit tooltip →
    what it does mechanically + why you would spend points on it. Reference, not hand-holding —
    available regardless of the Tips toggle.
  - *Tips toggle:* an opt-out coaching layer (defaults on for a first-ever visit, one tap to kill,
    persists). State-keyed bubbles plus moment-suggestions — above all the first-bounty designer
    reveal. One bubble at a time, never blocks input, never appears when off.
  - *Rematch* (vs AI) on the game-over banner; *empty lobby* offers Play vs AI beside "host one";
    help screen gains a DESIGN YOUR OWN section and reframes roles as example templates.
  - *Imported audio (added 2026-07-08, user-supplied assets):* title-screen music
    (Deep_In_Space, loops under the demo, fades when a real game starts — title-only by explicit
    decision), a quiet in-game ambient bed (Ambiant_Loop), a pneumatic-door one-shot for unit
    deploys (new SoundKind.Deploy with the old procedural click as fallback), and a faint
    computer hum while the Designer is open (Computer_loop). Machin_Loop deliberately unused
    (generator hum wants positional audio — later). *Weapon sounds (second user pack, Scifi Guns
    SFX):* attacks play asset shots matched to the projectile's existing damage tier — light/mid/
    heavy = Gun1/Gun3/Gun5 families (size-ordered; remappable by ear after live play), each
    rotating 4 variants randomly so repeated attacks never sound stamped; a gun-rack clip plays
    when a design is created into the barracks (SoundKind.Design). Deaths deliberately keep the
    procedural explosion (no explosion in the pack; the synth boom is the strongest procedural
    sound). Extracted role-named: AttackLight/Mid/Heavy_0-3 + CreateRacked; both vendor packs
    deleted after extraction. Music/ambience live on dedicated looping
    AudioSources beside the one-shot SFX source; music ignores the demo's SFX-mute. All combat
    SFX stay procedural; every asset-backed sound keeps a synthesized fallback. The four used
    clips live under Assets/HexWars/Resources/Audio as role-named copies (TitleMusic, AmbientBed,
    DeployDoor, DesignerHum — tracked); the 1.2 GB vendor kits were deleted after extraction
    (user-authorized; .gitignore entries remain as re-import protection). WebGL constraint: no
    Streaming load type — music is CompressedInMemory Vorbis.

**Non-goals**

- No accounts, persistence, match history, replays gallery, or leaderboard (the storage milestone).
- No scripted tutorial scenario (the first vs-AI game with Tips on is the guided game).
- No unit-type documentation — stats are described, archetypes are not (the design space makes
  archetype descriptions wrong by construction).
- No cold-start mitigation (hosting decision pending), no async play, no turn timers.
- No RL/gym retraining work — but engine changes must keep the RL surface consistent (see §5).

## 3. Reconnect & rejoin

**Identity:** the client mints a random 16-char token once per browser (PlayerPrefs) and sends it on
every connect: `?room=X&token=T` (+ existing `setup`/`private`/`join` params). The server seats by
token, not by connection id.

**Server (engine `Net\`):**

- `GameSession` seats keyed by token (the connection→token mapping lives in the room). A `Join` with
  a known token reclaims its seat regardless of the socket it arrives on.
- A started room whose connections all drop is **held for 10 minutes** (injected clock; swept
  opportunistically on any hub call) instead of deleted instantly — both players can refresh
  through a net blip without losing the match. Un-started rooms keep today's instant cleanup.
- Any (re)connect into a **started** room re-deals `START` with the session's current state to that
  connection only (the resync-by-replay mechanism that already exists for seat refill).
- Wire protocol unchanged: SEAT/START/APPLY/REJECT as-is; `token` is a connect-query param like
  `private`.

**Client (`NetClient`):**

- On unexpected close: surface "Connection lost — reconnecting…" (toast + the waiting/HUD status
  line where one exists) and retry with capped exponential backoff (1s, 2s, 4s … cap 15s,
  indefinitely — the player can always leave via Main menu). On reconnect the server re-deals
  START; `OnNetStart` already rebuilds from it. Deliberate teardown (`_closing`) never retries.
- A mid-game rejoin replaces local state wholesale via the START re-deal; the presenter resets
  (existing `ResetQueue` path) — no attempt to replay missed actions as animations.

**Hardening batch (same server files, same tasks):** per-connection ordered outbound queue
(`Channel<string>`, enqueued under the hub lock, drained by one writer task per socket — closes the
APPLY-ordering race, audit N2); incoming message cap 64 KB (N3); `Origin` header checked against the
request host when both present (M13); `CommandWire.TryRead` + `REJECT Malformed` for garbage
payloads (N4).

## 4. Session longevity + portrait + shareability

- **HP bars** (audit P1): built once per token (background + fill quad); refresh scales/positions
  the fill and tints via `MaterialPropertyBlock` on one shared material. No per-sync
  destroy/create, no material instantiation.
- **Tooltip** (P2): cache `(unitId, hp, moved, attacked)`; skip re-format when unchanged; count
  lines without `Split`.
- **Event console** (F1/U1): GUIStyles + joined log string cached (rebuilt only in `Report`/
  `Clear`); hidden entirely on portrait aspect (< 1.0 width/height) — the combat log remains
  available on desktop/landscape where it fits.
- **Effect materials** (P4): projectile tiers, explosion flash/debris, and token hull materials
  cached statically (the demo restart loop made these unbounded; meshes were fixed in the lobby
  milestone, materials were not).
- **Portrait pass:** every screen verified and fixed at 390×844 logical: title menu, browser (rows
  must not overflow), setup form (clamp card width to available width, the GameRules pattern),
  waiting screen, HUD banner (text truncation rules), barracks/designer (clamp width and vertical
  position to fit — same docking, no layout redesign), tooltip, game-over banner, rules card.
- **Share/PWA:** OpenGraph + Twitter-card tags (title, description, a real screenshot), favicon,
  `manifest.json` + icons, `theme-color`. Injected at staging time by `stage-webgl-deploy.ps1`
  (the Unity template owns index.html; staging already rewrites it for cache-busting, so the
  injection lives there — one authority for post-build HTML edits). Icons/screenshot live as
  static files under `wwwroot/` outside `Build/`.

## 5. Engine: named templates, starter set, deletion

**Template = name + stats.** `UnitTemplate` (readonly struct: `string Name`, `UnitStats Stats`)
replaces bare `UnitStats` in the barracks (`PlayerState.Barracks : IReadOnlyList<UnitTemplate>`).
`CreateUnit` gains the name; `DeployUnit` copies the template's name onto the spawned unit;
`Unit.Name` defaults to the dominant-role string when absent. Names are sanitized at the engine
boundary: trimmed, length-capped at 20, restricted to `[A-Za-z0-9 \-']`, empty → dominant role.
(Underscore is deliberately NOT allowed: the wire encoding maps spaces↔underscores, so allowing
literal underscores would make names lossy across a round-trip — the whitelist keeps the encoding
bijective.)

**Wire/replay compatibility (the one format-touching change):** `CreateUnit`'s wire line carries the
name as the final token with spaces encoded as underscores; wherever `ReplayFile` currently encodes
barracks entries and units, those records gain a trailing name token the same way. Readers treat a
missing trailing token as "no name" (old payloads and replays parse unchanged); writers always emit
it. Old-payload round-trip tests are mandatory. `CommandWire.TryRead` (from §3 hardening) covers
malformed names.

**Starter templates.** `GameFactory.Build` seeds every player's barracks with five named examples
(deployable immediately, deletable per-game):

| Name | H | D | Def | Mv | Vt | Rg | RgArc | Vis | VisArc | Teaches |
|---|---|---|---|---|---|---|---|---|---|---|
| Brute | 7 | 2 | 2 | 3 | 2 | 1 | 1 | 2 | 1 | the wall |
| Striker | 2 | 6 | 0 | 3 | 2 | 2 | 1 | 3 | 1 | the glass cannon |
| Sniper | 2 | 2 | 0 | 2 | 2 | 6 | 1 | 4 | 1 | reach |
| Artillery | 3 | 6 | 0 | 0 | 0 | 5 | 2 | 2 | 1 | a statline is a concept: it never moves — you emplace it; arc 2 lobs over terrain but needs a spotter |
| Scout | 2 | 0 | 0 | 4 | 3 | 0 | 0 | 7 | 2 | information as a weapon; harmless and sees everything |

(The first three match the existing starting-army roster exactly; costs derive from `PointCost` as
today. Artillery/Scout are examples, not balance claims — the self-play harness can vet them later.)

**Deletion.** New `DeleteTemplate(issuer, templateIndex)` command: validated (index in range), free
(no points, no turn action), **not enumerated by `LegalMoves`** (an administrative action, not a
game move — this also keeps RL action masks untouched). Barracks UI gets a per-row ✕ with the
existing selected-row (`_deployIndex`) bookkeeping corrected on shifts.

**RL surface note:** `TacticalLayout.Roster` stays 3; the seeded list puts the classic trio at
indices 0-2, so the RL deploy region now addresses live templates from step one (previously dead
until an agent Created). Agents never issue `DeleteTemplate` (not enumerated), so indices never
shift under a training episode. Artillery/Scout at indices 3-4 are outside the RL action space by
design. `RoleOf` observation matching is untouched (it reads stats, not names).

## 6. The teaching layer

**Stat descriptions — always available, Tips or no Tips.** Tapping/hovering a stat *name* in the
Designer or the unit tooltip shows its bubble. The nine descriptions (verbatim product copy; each is
mechanic + judgment):

1. **Health** — How much damage it absorbs before dying. Buy it for units that must hold ground
   under fire; a 2-health unit dies to one mistake.
2. **Damage** — Subtracted by the target's Defense; a landed hit always deals at least 1. This is
   kill speed — enough Damage makes Defense stacking pointless.
3. **Defense** — Subtracted from every hit you take. Against a swarm of weak attackers it multiplies
   your effective health; against one big gun it's nearly worthless. Read the enemy's army first.
4. **Movement** — Horizontal steps per turn. Reach, escape, and tempo. Zero is a choice: an
   emplacement that never moves — position it like it matters, because it will never matter again.
5. **Vertical** — How many levels it can climb per turn (descending and level moves are free). High
   ground adds damage and reach, so climbers take the positions that win fights.
6. **Range** — How far it shoots (0 = melee only). Outranging the enemy's answer is free damage;
   high ground extends it further.
7. **Range Arc** — How many levels *up* it can fire — and anything above 0 can lob over blocking
   terrain (indirect fire). Your army still needs eyes on the target: batteries want spotters.
8. **Vision** — How far it sees. Sight is shared by your whole army, and you can only shoot what
   somebody sees. Under fog, information is the game — a cheap pair of eyes makes every gun longer.
9. **Vision Arc** — How many levels up it can see. Cliffs hide things; someone has to look over the
   edge.

**Tips toggle.** Persisted per browser (PlayerPrefs); first-ever visit defaults **on**; toggle lives
on the title screen and beside the in-game "?"; off = the coaching layer is completely inert. One
`TipsService`: owns the flag, renders exactly one dismissible bubble at a time (tap anywhere or act
→ gone), never blocks input, each trigger fires at most once per game. Triggers:

- First unit selection → "Green hexes = where it can go. Red = what it can hit."
- Designer opened → stat rows show their one-line captions inline (the §6 descriptions, short form).
- First time points ≥ cheapest deploy cost with barracks open → "Deploying costs the unit's points."
- Out of actions with turn not ended → "Nothing left this turn — End Turn passes play."
- **First bounty earned (the reveal)** → "You earned N points. A wall? A sniper? Eyes that see
  everything? **Design your answer.**" — with a button that opens the Designer.
- Game over (vs AI) → "Run it back — you know what to build now." pointing at Rematch.

**Copy voice everywhere:** the templates are *examples of imagination applied to points* — help
screen's unit section becomes: *"Brute, Striker, Sniper, Artillery, Scout — these are just ideas that
come pre-loaded in your barracks (delete them if you like). Any allocation you can imagine is a
unit. Name it what it is."* plus a new **DESIGN YOUR OWN** section explaining create → deploy →
adapt: *"Everything is points. See what your opponent built; build the answer."*

**Rematch:** game-over banner gains a Rematch button in vs-AI games only — same setup, fresh seed,
instant restart. **Empty lobby:** the browser's empty state offers **Play vs AI** beside "Host Game."
**Designer name field:** a Name box (browser-prompt text entry, the established mobile pattern) whose
placeholder rotates through evocative examples ("Doom Turtle", "Longshot", "Pathfinder") to model the
naming habit; empty stays legal (defaults to dominant role).

## 7. Error handling

- Reconnect loop: backoff capped at 15 s, runs until success or deliberate exit; every attempt
  updates the status line; Main menu (CancelHosting path) always works mid-reconnect.
- Token absent/garbled → server treats the connect as a fresh identity (today's behavior); a
  reclaim into a room whose hold expired → SEAT FULL → existing toast path.
- Held-room sweep runs on hub calls only (no timers in the pure hub); expiry uses the injected clock.
- Name sanitization at the engine boundary means no client can wire an unparseable or abusive-length
  name; the UI additionally pre-trims.
- Tips bubbles self-dismiss on scene teardown; the service never holds references across games.

## 8. Verification

- Engine tests: template naming round-trip (CommandWire + ReplayFile old-and-new payloads),
  DeleteTemplate validation + index shifts, starter seeding (all modes), token-keyed rejoin
  (same token/new connection reclaims its seat; hold-window expiry via injected clock; un-started
  rooms still clean up instantly), sanitization edge cases. NetServer selftest: kill a client
  socket mid-game, reconnect with the same token, receive START re-deal, play on.
- Play-mode (coplay, screenshots): starter templates visible turn one with names; delete flow;
  designer name entry (editor fallback: default name); stat-description bubbles; each Tips trigger
  (and their absence with Tips off); rematch; empty-lobby vs-AI button; portrait-layout sweep at
  390×844 for every screen; HP-bar visual identical after the rebuild-once change.
- Session-longevity proof: a scripted 3-game back-to-back run with material/mesh counts sampled
  between games (counts must plateau, not climb).
- Live smoke (user): phone backgrounding mid-game → auto-reconnect; refresh mid-game → same seat;
  link paste into a chat shows the preview card; Add to Home Screen; first-visit Tips arc through
  the bounty reveal; Tips off stays off.

## 9. Success criterion

A stranger on a phone, alone at 11pm, taps a link from a group chat: the preview card looked like a
game, the title is alive, the empty lobby points them at the AI, and within two games they have
earned points, been shown the Designer at the exact moment it matters, built something they named
themselves, and won with it. Meanwhile a friend who backgrounded the tab during an organized match
comes back to the same game without noticing anything happened.

## 10. Risks

- **Replay/wire format change** (names): mitigated by trailing-token-with-default design + explicit
  old-payload tests; this is the only format-touching change and it is backward-readable.
- **Rejoin state divergence:** eliminated by wholesale START re-deal (never incremental catch-up).
- **Seeded templates change game feel** (deployable options from turn one, notably in Territory
  where starting points exist): intended; the examples are the product. Balance is explicitly not
  asserted — the self-play harness can measure Artillery/Scout later.
- **Tips annoyance:** every bubble dismissible, one at a time, once per game, global kill switch,
  off forever once off.
- **Portrait work touching every screen:** mitigated by the screenshot sweep at a fixed logical
  size as the acceptance gate.
