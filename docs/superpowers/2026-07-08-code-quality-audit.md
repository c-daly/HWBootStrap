# HexWars Code Quality & Efficiency Audit

**Date:** 2026-07-08 (at main `3b8db00`, post opponent-turn-presentation merge)
**Method:** four parallel subsystem reviews (engine core, engine API/RL/net, presentation gameplay,
presentation UI/drivers), highest-severity findings independently re-verified against source.
**Scope:** ~8,600 lines project C# + ~670 lines Python. Focus: performance (including latent) and
architecture/SOLID. Gameplay-rule correctness was out of scope (254 engine tests cover it), but
several correctness bugs surfaced anyway and are included.

Findings marked ✔ were independently verified by a second read of the cited source.

---

## Overall verdict

The architecture is genuinely good — better than typical for a codebase this age. The functional
core (immutable `GameState`, single `Apply` path, pure static rule services), the strict layering
(engine has zero Unity/net dependencies; `MatchHub` is transport-free and unit-testable; the RL
observation/action codec is shared between training, duel, and Unity so encodings can't drift),
and the single presentation pipeline (every input source funnels through `ActionPresenter`) are
all the right shapes, and the doc-comment discipline is exceptional.

The defects are **systemic rather than random** — four recurring themes:

1. **The engine pays immutability's copy costs while collecting none of its caching benefits.**
   No per-state occupancy/vision/reachability caches, so every query is a linear scan and army
   vision is recomputed cubically. Dominant cost at RL scale (millions of steps).
2. **Unity resource lifecycle.** Procedural materials/meshes created per-event and never
   destroyed, plus steady per-frame allocation in the input/tooltip/console loop — on WebGL's
   stop-the-world Boehm GC this guarantees progressively worsening hitches over a session.
3. **Rule knowledge duplicated across boundaries, enforced only by discipline** — and it has
   already cracked in four places (territory-deploy legality, replay config, seat gating,
   fog-gate mirroring).
4. **The public NetServer trusts remote input more than a deployed service can afford.**

None of the fixes require architectural change; most have in-repo precedents (the codebase
already demonstrates the correct caching patterns — the leaks are the sites that missed them).

---

## Top findings (calibrated, cross-subsystem)

| # | Finding | Severity | Where |
|---|---------|----------|-------|
| 1 | Unclamped `?setup=` → one-request OOM DoS on the public server | Critical | GameSetup.cs:40, NetServer Program.cs |
| 2 | RL slot map frozen at reset — deployed units permanently inert to trained agents | Critical (RL mission) | TacticalLayout.cs:61, TacticalCoding.cs:95,134 |
| 3 | HP bars rebuilt + 2 Materials leaked per unit per Sync | Critical (WebGL) | TokenStore.cs:161-184, BoardRenderer.cs:103 |
| 4 | Army vision recomputed per attacker×target — cubic LegalMoves | Critical (RL scale) | LegalMoves.cs:51, TargetingService.cs:14 |
| 5 | Movement BFS allocation storm + full BFS re-run per Apply(MoveUnit) | Critical (RL scale) | MovementService.cs, GameEngine.cs:163 |
| 6 | No per-state occupancy cache — every query is a linear entity scan | Critical (RL scale) | LegalMoves.cs:82, GameEngine.cs:373 |
| 7 | LegalMoves/GameEngine territory-deploy divergence — wrong RL action masks | Important | LegalMoves.cs:23 vs GameEngine.cs:125 |
| 8 | NetClient: no reconnect/timeout/user-facing state — dropped socket silently kills online games | Important (product) | NetClient.cs:24-41 |
| 9 | End Turn & Create Unit buttons not seat-gated — human can act as the AI | Important | GameHud.cs:117, DesignPanel.cs:81 |
| 10 | EventConsole IMGUI + tooltip formatting: ~250-550 KB/s steady garbage during live play | Important (WebGL) | EventConsole.cs:53, UnitTooltip.cs:75, UnitInputController.cs:71 |
| 11 | Replay CONFIG misses 8 of 28 knobs — silent playback divergence | Important | ReplayFile.cs:115 vs GameConfig.cs |
| 12 | APPLY ordering race in the relay (send outside the hub lock) | Important | NetServer Program.cs:79-89 |

---

## 1. Engine core (`engine/HexWars.Engine`)

### Critical — performance (all dominate RL/self-play throughput)

**E1. ✔ Cubic army-vision recompute in LegalMoves** — `LegalMoves.cs:51-56` calls
`TargetingService.CanTarget` per attacker×target; `IsVisibleToArmy` (TargetingService.cs:41-55)
is attacker-independent yet loops all friendly spotters with an O(distance) LOS walk each time.
O(U² × E × D) per LegalMoves call, which runs every step for action masking. *Fix:* compute
visibility once per target before the unit loop; longer term a per-state visible-set cache
(fog rendering uses the same predicate, so it pays twice).

**E2. ✔ Movement BFS allocation storm + redundant re-run** — per `ReachableCosts` call (every
living unit, every LegalMoves; again inside every `Apply(MoveUnit)`, GameEngine.cs:163):
`HexCoord.Neighbors()` allocates `HexCoord[6]` per dequeued node (MovementService.cs:48);
`AddToFrontier` allocates a closure per insertion via `RemoveAll(p => ...)` (:99); `OccupiedCells`
rebuilds an identical HashSet per unit (:72-80); plus per-call dictionaries/queues. ~60-100 KB
gen0 garbage per LegalMoves call → tens of GB per training run. GreedyAgent's probe-every-move
style multiplies the Apply-side BFS ~15×. *Fix:* direction-table loop instead of `Neighbors()`,
manual reverse-loop instead of `RemoveAll`, hoist the occupied set, pooled scratch buffers,
cache `ReachableCosts` per (state, unit).

**E3. ✔ No spatial caching on the immutable state** — `IsOccupied` (duplicated at
LegalMoves.cs:82-90 and GameEngine.cs:373-381) scans every entity of both players; called per
deployment-zone cell, per capture check, and per board tile under BuildAnywhere. The state is
deeply immutable, so a lazily built occupancy map (and later vision/reachability caches) computed
once per state makes all of these O(1). **The single biggest structural perf lever in the repo.**

### Important

**E4. Board.WithControl rebuilds the whole board** (Board.cs:64-68) — re-hashes all tiles, rebuilds
both zone sets, copies the control dict twice, on every capture (and every capture GreedyAgent
probes). *Fix:* private aliasing ctor that shares `_tiles`/zones.

**E5. Dictionary-backed tile store + double lookups** — `Contains` → `TileAt` doubles every probe
in BFS/LOS/targeting inner loops (MovementService.cs:50-52, LineOfSight.cs:20-23); the terrain
table is a `Dictionary<TerrainType,...>` (GameConfig.cs:164) where enum-keyed dictionaries have
historically boxed per lookup under Unity Mono/IL2CPP. *Fix:* `TryGetTile`, `TerrainDef[]` by
enum index; longer term a dense `Tile[]` over the bounding box.

**E6. ✔ LegalMoves/GameEngine rule divergence — live bug** — the deploy handler
(GameEngine.cs:125-132) is control-based in TerritoryMode; the enumerator (LegalMoves.cs:23)
is always zone-based. In territory mode the enumerator (a) never offers deploys onto captured
hexes the handler accepts and (b) offers deploys the handler rejects — silently skewing self-play
stats and making RL action masks wrong. `CaptureCostFor`, `BuildCostFor`, `IsOccupied`,
`HasGeneratorAt` are likewise copy-pasted between the two files, the mirror enforced by nothing.
*Fix:* one shared internal `Rules` module + a property test ("every enumerated move Applies; no
rejected move is enumerable").

**E7. ClaimEndsTurn hardcoded in Apply, bypassing the ITurnPolicy seam** (GameEngine.cs:24-27) —
the seam exists precisely for this; wrap as a policy decorator. (OCP violation with three homes
for turn-structure rules.)

**E8. ✔ Replay CONFIG serializes 20 of 28 knobs** (ReplayFile.cs:115-140) — BountyRate,
GeneratorCost, GeneratorHealth, DmgHighGroundBonus, RangeHighGroundBonus, RoundCap, DesignFee,
DeployCostMultiplier all silently reconstruct as defaults → replays of non-default games diverge
on playback. Symptom of **E9**: `GameConfig`'s 28-param ctor + 22-param `Default` + two hand-kept
serializer lists — every knob touches four sites. *Fix:* record with init-only props + one
(key, getter, setter) table shared by write/read.

**E10. Economy.ControlledHexes is O(tiles) duplicating O(controlled) Board.ControlledCount**
(Economy.cs:20-26 vs Board.cs:55-60), sitting in GreedyAgent's per-candidate scorer (~47k needless
hash lookups per territory decision); `ApplyEndTurn` also computes income twice (Upkeep recalls
Income).

**E11. Unbounded barracks growth** — `CreateUnit` has no cap/dedup (GameEngine.cs:91 O(n) list
copy → O(n²) cumulative); RandomAgent creates on ~20% of decisions, so LegalMoves lists explode
to 10,000+ DeployUnit records per call late in random-agent matches, and `WinCheck.IsEliminated`
iterates the whole barracks after every command. *Fix:* engine-level `MaxBarracksSize` or dedup —
also protects the RL action space.

### Minor
`Result` as class = 1 alloc per Apply (make readonly struct) · `AttackedUnitIds.Contains` relies
on LINQ's runtime HashSet fast path (expose `HasAttacked(id)`) · `HoarderAgent.cs:23` nullable
violation · unknown command rejected with `RejectionReason.None`; `GeneratorNotFound`/
`IllegalForPolicy` never produced · stale "generators removed" comment (LegalMoves.cs:36) +
`DeployGenerator` handler that no enumerator can reach · BFS keys frontier by cell-only and reads
start elevation from the tile, not `unit.Elevation` — needs (cell, elevation) nodes for the 3D
battlespace plan · missing `WithUnits`/`WithGenerators`/`WithBarracks` withers force 13 long
positional ctor calls · `Simulator.RunBatch` is single-threaded despite embarrassingly parallel
seeded games (near-linear `Parallel.For` win for balance sweeps) · replay `Read` throws bare
`IndexOutOfRangeException` on truncation; core `ReplayFile` depends on `Net\CommandWire` (codec
should live in core); `Match.RunCore` silently swallows rejected agent commands (a rejection
counter would have caught E6).

---

## 2. Engine API surface — Net / RL / Sim / Python

### Critical

**N1. ✔ Unvalidated setup dimensions = one-request remote DoS** — `GameSetup.Parse`
(GameSetup.cs:40-47) TryParses but never clamps magnitude; the NetServer feeds `?setup=` from an
unauthenticated WebSocket straight to `GameFactory.Build` **inside the global hub lock**.
`?setup=0 9999 9999 0 7` → ~10⁸ dictionary-backed tiles (GBs); huge `ArmySize` likewise. OOM or
minutes of CPU while holding the lock every room shares → total outage of the deployed Render
service from one anonymous request. *Fix (afternoon):* clamp in Parse — Width/Height ∈ [3,32],
ArmySize ∈ [1,12], StartingPoints ∈ [0,1000], TurnActions ∈ [0,32].

**R1. ✔ Frozen slot map: deployed units are permanently inert to trained agents** — the slot map
is built once from the starting armies (TacticalLayout.cs:61-69) and assigned only at Reset
(TacticalEnv.cs:94, DuelEnv.cs:53-54). `Encode` returns -1 for a deployed unit's MoveUnit/
AttackUnit (TacticalCoding.cs:134-142) and `Mask` silently drops them (:95-96). Meanwhile deploy
actions ARE encodable and the reward shaping (`PointsWeight`) actively incentivizes deploying —
**training teaches agents to buy units they can never move or attack with**, and it also poisons
any RL-based evaluation of the barracks economy. Failure is invisible: no error, just missing
mask bits. *Fix:* re-derive slot→unitId each turn from living units in canonical order (also
recycles death-freed slots), or grow the map on deploy; either way assert loudly when Encode
returns -1 for an engine-legal move.

### Important

**N2. APPLY ordering race + head-of-line blocking** (NetServer Program.cs:79-89) — hub mutation is
serialized but Outbounds are sent after the lock is released; two handler tasks can deliver APPLY
B before APPLY A to the same client → local engine rejects B → silent desync for the rest of the
game. Surfaces only under fast interleaved play. *Fix:* per-connection ordered `Channel<string>`,
enqueue inside the lock, drain per-connection.

**N3. Unbounded incoming message buffering** (Program.cs:91-104) — fragments accumulate into a
MemoryStream with no cap; protocol needs tens of bytes. Cap at 64 KB and close.

**N4. Unguarded CommandWire.Read** (CommandWire.cs:30-46, called at MatchHub.cs:74) — garbage
payload throws through the hub, killing the connection and bypassing the REJECT path. Add
`TryRead` + `REJECT Malformed`.

**R2. JSON-text transport of full obs+mask every training step** (hexwars_gym/env.py:35-57) —
~761 floats + ~1,000 bools as JSON text (~10-15 KB) per step, `json.loads` → list → ndarray twice;
~0.2-0.5 ms/step of pure transport ≈ 25-50% training-throughput tax. *Fix:* length-prefixed binary
frames (`np.frombuffer` on float32 + packed bitmask); keep JSON for the handshake.

**R3. Single synchronous env** (train_maskable_ppo.py:52, selfplay.py:45, train_dqn.py:61) — no
`SubprocVecEnv`; the design is already process-isolated and seed-parameterized, so N=8 envs is a
~5-8× samples/sec win compounding with R2.

**R4. Per-step encoder churn** (TacticalCoding.cs:29-38, 89-98) — fresh obs/mask arrays per step,
full `LegalMoves.For` materialization for the mask, and the **immutable elevation plane recomputed
every step** (2×N dictionary lookups + division for data fixed at reset). *Fix:* snapshot the
elevation plane at Reset, caller-owned buffers, allocation-free `LegalMoves.ForEach` overload.

### Minor
Global hub lock serializes all rooms with boardgen inside it (per-room locks; generate outside) ·
fog is client-trust-only — relay broadcasts full APPLY/START regardless of fog (document the trust
model; per-seat filtering is a design decision to make deliberately) · seat identity is a
per-socket GUID — double-drop + reversed rejoin swaps armies (rejoin token) · per-message
`Console.WriteLine` with payload contents on the hot path · env guard-exhaustion and rejected
actions are silent (surface `IllegalAction`/`OpponentStalled` in info) · TacticalEnv/DuelEnv
duplicate ~40 lines of step orchestration (the one place training/duel can still drift) ·
`RoleOf` structurally scans 9 fields per unit per observation; identical templates alias ·
python `close()` lacks `wait()`; trainers lack try/finally; policy_server dies on malformed lines ·
`duel.predict` imports inside the call · Sim sweeps single-threaded (same as E-minor) · no origin
check / room-code validation on `/ws`.

---

## 3. Presentation — gameplay core

### Critical (WebGL lifecycle/GC)

**P1. ✔ HP bars destroyed and rebuilt with two leaked Materials per unit per Sync**
(TokenStore.cs:161-184) — `RefreshHpBar` destroys both bar quads and `MakeBarQuad` calls
`_board.UnlitColorMat(c)` → `UnlitColor` does `Shader.Find` + `new Material` unconditionally
(BoardRenderer.cs:103-112). `Destroy(gameObject)` never destroys code-created Materials; Sync runs
≥2× per presented action. ~150-action game × 10 units ≈ **6,000 leaked Materials** + 6,000
CreatePrimitive(Quad)-with-MeshCollider create/destroy pairs. Worst object-lifecycle bug in the
repo; heap and GC scan time grow monotonically over a session. *Fix:* build the bar once per
token; refresh = scale/position the fill quad + `MaterialPropertyBlock` color on one shared
material. (Was already on the post-merge follow-up list as "RefreshHpBar quad churn" — it is
worse than the ledger entry suggests because of the material leak.)

**P2. ✔ Tooltip formats every frame while a unit is hovered *or selected***
(UnitInputController.cs:71-72 → UnitTooltip.cs:75-113) — ~8 interpolations + `Split('\n')` per
frame; a selected unit keeps this running all turn ≈ 90-150 KB/s steady garbage. *Fix:* cache
(unitId, hp, spent, attacked) and early-out; count lines without Split; ideally event-driven.

### Important

**P3. Per-frame closure + label allocation in the territory action button**
(UnitInputController.cs:322-369, called unconditionally from Update) — recompute only on
StateChanged/selection change.

**P4. Effect materials created per-event, never destroyed** — projectile: `Shader.Find` + new
Material per attack (ActionPresenter.cs:379-381); explosion: 2 materials per spawn
(ExplosionFx.cs:35-62), spawned per impact/kill/claim; token hull material per token per fog
flip (TokenStore.cs:206). *Fix:* cache like the in-repo precedents (`_iconMats`, `_controlMats`).

**P5. Skybox texture (1.5 MB GPU + 2 MB temp) rebuilt every NewGame, previous orphaned**
(GameBootstrap.cs:416-436) — sibling `BrightReflection()` IS statically cached; clear oversight.

**P6. Procedural meshes never shared or destroyed** (BoardRenderer.cs:180-190, HexMesh.cs) —
~500 meshes per board build for ≤6 distinct shapes; `ClearChild` leaks them all on rebuild
(per-game, not per-action — frequency acceptable, leak isn't). Static ring mesh + prism-by-height
cache.

**P7. ~450-600 renderers for a 13×9 board** (BuildColumn) — dominant per-frame draw-call load on
mobile WebGL. `CombineMeshes` per terrain material → ~6-8 draw calls; move tint to vertex color
or overlay quads.

**P8. UpdateControlTint: string-based `Find("Fill")` + 3 GetComponents × 117 columns on every
committed action** (BoardRenderer.cs:87-101) — cache `(TileView, MeshRenderer)` at build; diff
prev/next control (presenter already holds both states).

**P9. ✔ FogViewerFor calls FindAnyObjectByType<AiOpponent>() ~4× per action**
(GameBootstrap.cs:229) — the same class already uses `GetComponent` correctly for the same lookup
(known ledger follow-up; scene-scan cost quantified now).

**P10. Fog visibility flips destroy/rebuild whole tokens (~8 GOs + materials each)**
(TokenStore.cs:68-104) — scout-heavy fog games churn constantly. `SetActive(false)` +
reactivate; hard-destroy only actual deaths.

**P11. GameBootstrap is a god class (438 lines, ~8 responsibilities)** — session lifecycle,
environment art, demo armies, net glue, fog/seat policy, rejection copy. The
load-render-frame-notify sequence is triplicated verbatim (NewGame/StartLocalGame/OnNetStart);
TryApply/OnNetApply duplicate the apply-report-enqueue block. *Fix:* extract EnvironmentSetup,
`PresentState()`, SessionPolicy (FogViewerFor/IsLocalCommand/WaitingHumanSeat), move `Friendly()`
to Toast. (SRP; also the root enabler of the seat-gating bugs in §4.)

**P12. ActionSite re-implements each Play-handler's fog gate by mirrored duplication**
(ActionPresenter.cs:318-361) — the exact bug class the fog review existed to catch, held together
by comments. Compute one `PresentationPlan` per item; camera and handler both consume it.

### Minor
Sync allocates 2 HashSets + a List per call (reusable fields) · uncached `WaitForSeconds` ·
boxed `IReadOnlyCollection` enumerators in HasActed/Contains · `CombatFx.Report` allocates a
`PlayerId[]` per call · AiOpponent/SpectatorDriver duplicate the pacing loop; ReadOnly assigned
per frame · one `Billboard.LateUpdate` per HP bar with a per-frame scene-scan fallback when no
MainCamera tag · CameraRig writes the transform every frame when idle · UnitInputController uses
`FindObjectsByType<UnitView>` when TokenStore has an O(1) id lookup · `??`/`?.` on
UnityEngine.Object (fake-null hazard) at GameBootstrap.cs:74/89/96 etc. · duplicated math/palette:
HexPath.LerpRound mirrors LineOfSight's, CombatFx.Top mirrors TokenStore.CellTop, player colors
hard-coded in 3 files · opponent EndTurn gets a double pacing gap (0.5 s vs 0.25 s — intended?) ·
CreatePrimitive-then-DestroyImmediate-collider as standard pattern (Quad = cooked MeshCollider) ·
real-time point Light per explosion · selection-marker material color set per frame · stale
"force recompile" comment in UnitInputController.cs:8.

---

## 4. Presentation — UI & drivers

### Critical

**U1. EventConsole IMGUI sidebar allocates during live play** (EventConsole.cs:53-130) — 5
GUIStyles + interpolated strings + `string.Join` over 30 lines per OnGUI call, ≥2 calls/frame;
`AutoCreate` is unconditional and `GameBootstrap.TryApply` feeds it state, so its early-out never
fires again after the first action ≈ 150-400 KB/s in live mobile matches. Also fixed 430-logical
width covers ~86% of a portrait phone screen. *Fix:* cache styles + joined string (rebuild only in
Report/Clear), or port to uGUI; default collapsed on mobile; clamp width.

**U2. ✔ NetClient: no reconnect, no timeout, no surfaced state** (NetClient.cs:24-41) — mobile
tab-background routinely drops the socket; OnClose flips a bool and logs. Player returns to a
frozen board, no message, no retry. `Send` silently drops when not open while
`GameBootstrap.TryApply` returns true anyway; `async void` Connect swallows failures →
lobby waits forever. **Most likely real-world failure mode of the whole product.** *Fix:* toast +
bounded exponential-backoff reconnect (server already reseats/replays via SEAT/START); `Send` →
bool + toast on failure.

### Important

**U3. ✔ End Turn is not seat-gated** (GameHud.cs:117-120) — issues `EndTurn(ActivePlayer)`, so
clicking during the AI's turn truncates the AI's turn (misclick hazard and a deliberate exploit —
deny the AI its remaining actions every turn). Online is saved only by the seat check in TryApply.

**U4. ✔ Create Unit is not seat-gated** (DesignPanel.cs:81-85) — vs AI, clicking during the AI's
turn banks your design into the **AI's** barracks and spends its points; online it's a silent
no-op. AiOpponent ReadOnly-gates input+barracks but DesignPanel has no ReadOnly at all.
*Fix for U3/U4:* one `IsLocalHumanTurn` on GameBootstrap consulted by every command-issuing
widget (seat logic currently exists in 3 places with different subsets — GameHud.LocalHumanActs,
IsLocalCommand, WaitingHumanSeat).

**U5. ✔ BarracksPanel renders the ACTIVE player's barracks** (BarracksPanel.cs:113-114) — online,
during the opponent's turn your sidebar lists their undeployed template designs (role + cost):
an information leak the fog design explicitly cares about (CombatLog fogs enemy deploys; this
hands over their build strategy every turn). Also means you can't browse your own barracks while
waiting. *Fix (one line):* `s.Player(_game.Seat ?? s.ActivePlayer)`.

**U6. BarracksPanel destroys/recreates every row on every StateChanged** (BarracksPanel.cs:107) —
fires per command including opponent playback at ~3 actions/s; list only changes on
Create/Deploy/turn-flip. Diff and early-out.

**U7. GameHud.Refresh materializes the full LegalMoves list + 2 scene scans per state change**
(GameHud.cs:146-153, 243, 256) — hundreds of command allocations to answer "any action left?";
add a lazy `LegalMoves.Any(state, predicate)` and cache the AiOpponent reference.

**U8. Nine copies of the uGUI widget-factory boilerplate** (~350 of 2,000 lines) — `BuiltinFont`
verbatim ×4 (+5 inline), `EnsureEventSystem` ×2, three Label conventions differing in anchor and
y-sign. Code-built UI is fine at this scale **only** with the factory extracted (`UiKit`) — the
pattern is at its copy-paste breaking point. Related: three different CanvasScaler reference
resolutions, and UnitTooltip hard-codes the barracks panel's rect (`-486f // barracks panel is
(-8,-58) + 420 tall`).

**U9. Panels discover GameBootstrap via FindAnyObjectByType and bind to the concrete class** —
no interface seam, nothing UI-side is testable without a full scene + engine; seat logic smeared
across call sites is the direct cause of U3/U4. (DIP; the panels literally live on the same
GameObject — `GetComponent` or a constructor-style Init would do.)

**U10. PolicyBridge blocks the main thread on unbounded ReadLine** (PolicyBridge.cs:46-52) — a
hung torch import freezes the editor with no escape; `{"error":...}` replies parse garbage; any
stray print() on the server's stdout desyncs the protocol. Editor/dev tool, so Important not
Critical.

**U11. Lobby "waiting" is a dead end** (LobbyPanel.cs:225-237) — no Cancel; combined with U2,
connection failure strands the host until page reload; room-full only logs.

### Minor
HelpOverlay carries ~65 lines of dead code (Toggle/BuildPanel/HelpText — never invoked, documents
removed mechanics) · BarracksPanel rows overflow the panel after 10 templates (unbounded template
count, no mask/scroll) · LobbyPanel polls `State != null` per frame; PauseToggle runs 3×
FindAnyObjectByType every 0.5 s forever, even in live WebGL games where none of its targets exist ·
NetClient.OnMessage parses server frames unguarded (`int.Parse`/`CommandWire.Read` throw inside
the WS callback) · ReplayPlayer.LoadText NREs before Start and doesn't catch parse failures ·
WebGLBuild doc header says compression disabled, code sets Gzip (stale doc on the build entry
point) · TrainingLauncher interpolates the run name unquoted into a cmd line (space/metachar
breaks it) · GameHud `_lastActive` goes stale across games (turn-handover toast can be suppressed
in game 2).

---

## Strengths (verified across all four reviews)

- **Functional core done right:** immutable state, single Apply, structural sharing, no
  exceptions on the hot path (`Result`/`RejectionReason`), sealed-record commands doubling as
  the wire format. Exactly the right substrate for search/RL.
- **Determinism discipline end to end:** seeded RNG only, no statics/globals in the engine,
  culture-invariant serialization, replay = start + commands, mirror-symmetric board gen.
- **Real seams:** `ITurnPolicy`, `IBoardGenerator`, `IAgent`; `MatchHub` pure and testable;
  `TacticalLayout`/`TacticalCoding` shared by training, duel, and Unity so encodings can't drift
  (the most common RL-integration failure, designed out).
- **Presentation pipeline:** every input source through one queued presenter with correct
  FastForward/Commit re-sync semantics; TokenStore's diff Sync self-heals interrupted animations.
- **Event hygiene:** all StateChanged subscribers pair += with -= in OnDestroy.
- **WebGL-aware engineering:** custom unlit icon shader vs stripped variants, shader
  force-inclusion in the build script, procedural audio/textures (tiny build), browser prompt()
  for mobile text input, DispatchMessageQueue compiled out on WebGL, wss-from-origin derivation.
- **Caching discipline exists where designed in** (RoleIcons, _iconMats/_controlMats/_matcap,
  SoundManager clips, _reflection) — every leak found is a site that missed the established
  in-repo pattern, so fixes have precedents.
- **Exceptional doc comments** — design intent, not mechanics, including known compromises.

---

## Recommended sequencing

1. **Now (before/with the next deploy):** N1 setup clamps (+ N3 message cap, N4 TryRead) — the
   server is public today. U3/U4 seat gating (user-visible exploit vs AI). U5 barracks leak
   (one line).
2. **Before the next long training run:** R1 slot map (currently mis-training every run that
   involves deploys), then E1/E2/E3 as one "engine caching + zero-alloc" batch, then R2 binary
   framing + R3 vectorized envs (multiplicative wins).
3. **Next WebGL build:** P1 HP bars, P2/U1 steady GC (tooltip + EventConsole), P4/P5 material/
   skybox leaks. These four end the progressive-hitching failure mode.
4. **Product robustness:** U2 NetClient reconnect + U11 lobby cancel; N2 ordered outbound queues.
5. **Structural (schedule, don't rush):** E6 shared Rules module + property test, E9 GameConfig
   record + table-driven replay serialization (fixes E8), P11 GameBootstrap split + one
   IsLocalHumanTurn seam (U9), U8 UiKit extraction, P12 PresentationPlan.
