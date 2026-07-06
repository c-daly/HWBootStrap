# Opponent-Turn Presentation — Design

**Date:** 2026-07-05
**Milestone:** Juice & feel — the opponent's turn becomes a legible, animated, paced sequence.
**Approach:** Persistent tokens + one unified action presenter (Approach B; approved over an
animation shim on the rebuild renderer, and over a full view-model rewrite).

## 1. Problem

`BoardRenderer` destroys and rebuilds every unit token on each state change. Local actions feel
fine only because `UnitInputController` plays a tween *before* the state applies
(`MoveSeq`/`AttackSeq`). AI and server-echoed opponent actions skip that path entirely, so
opponent units **teleport**. With fog on, half the game's events are illegible non-events.
There is no camera feedback, and damage popups fire on state change — before the shot visually
lands.

The wire already delivers what a presenter needs: `GameBootstrap.OnNetApply` receives opponent
commands one at a time and applies them through the local engine, so `(prevState, cmd,
nextState)` is available for every action. **This milestone is presentation-only: zero engine,
wire, or Python changes.**

## 2. Goals and non-goals

**Goals**

- Every opponent action (AI, remote, spectator/replay) plays as an animated, paced sequence
  through the same pipeline as local actions.
- Under fog, hidden attacks read as a **tracer from the dark**: real bearing, origin clamped to
  the fog boundary, muzzle never rendered.
- Camera pans to off-screen opponent actions; kills get a micro-shake.
- Damage popups synchronize with visual impact.
- Modest own-action juice: ~50 ms hit-pause on attack impact; landing squash on deploy and
  move-end.
- **Refused clicks say why** *(added 2026-07-06 from playtest feedback)*: every presentation-side
  guard that silently eats a click — spent unit, out-of-range / no-LOS / unseen target,
  unreachable or occupied hex, invalid build tap, orders during the opponent's turn — surfaces a
  short reason toast through the existing `Toast`. Engine-side rejections already do; this
  closes the gap for clicks that never reach the engine.

**Non-goals**

- No new art direction, unit models, music, or ambient life (future polish specs).
- No event feed (the combat log / event console already covers it).
- No engine or `CommandWire` changes; no changes to the RL/gym side.
- No animator controllers / prefab view-model layer (Approach C, rejected as overkill).

## 3. Architecture

Three presentation-side pieces.

### 3.1 `TokenStore` — persistent unit tokens

A registry `unit id → token GameObject` replaces destroy-and-rebuild for units.

- Token visuals are exactly today's (disc + role icon billboard + box collider + `UnitView`),
  built once per unit and kept.
- `Sync(state, viewerSeat)` diffs: spawns tokens for newly present units, despawns dead ones,
  snaps survivors to true position/elevation, updates dim/bright spent-state materials in place.
- Under fog, "present" means *alive and visible to the viewer*. Enemy tokens fade in/out at the
  vision boundary instead of popping.
- `BoardRenderer` keeps tiles, terrain, and territory tint (rebuild-based is fine there — cheap,
  non-animated) and delegates all unit rendering to the store.

### 3.2 `ActionPresenter` — the single animation pipeline

A queue of `(prevState, cmd, nextState, isLocal)`. Every applied command — local click, AI,
server echo, spectator/replay driver — is pushed here instead of triggering an immediate
re-render. A coroutine drains it; per action type:

| Action | Presentation |
|---|---|
| Move | Token tweens hop-by-hop along the path (SmoothStep, ~0.3 s/hop). |
| Attack | Projectile arc attacker→target (existing power-scaled projectile lifted out of `UnitInputController`), impact explosion, damage popup **at impact**. |
| Deploy | Drop-in with a small landing squash. |
| Claim/Build | Territory tint pulse + existing chime. |
| EndTurn | Turn-handover beat. |

Pacing: ~0.25 s pause between opponent actions; local actions play with no added gap. After each
action animates, `TokenStore.Sync(next)` commits visual truth and the tile/territory layer
re-renders (its updates ride the action commit, since state changes no longer trigger an
immediate board re-render) — an interrupted animation still lands the board on the correct
state.

### 3.3 Rewired `GameBootstrap`

- `TryApply` (local) and `OnNetApply` (remote) both push to the presenter.
- **Engine state commits immediately; only visuals lag.** `StateChanged` fires instantly, so the
  HUD, barracks panel, and combat log stay real-time and authoritative.
- `UnitInputController`'s pre-animation special case (`MoveSeq`/`AttackSeq`) is deleted — that
  local/remote split is the root cause of the teleporting.
- `CombatFx.Report`'s popup timing moves into the presenter (popups at impact); its state-diff
  damage/kill detection and breakdown text are reused.

Free win: `SpectatorDriver` (AI-vs-AI spectating) drives games through `TryApply`, so it becomes
fully animated with no extra work. `ReplayPlayer` and `ModelDuelDriver` render reconstructed
states directly (scrubbing needs instant snaps) and stay on the un-animated facade; animating
sequential replay playback is a possible follow-up.

## 4. Fog rules

Viewer = local seat; visibility computed from vision in `prev`/`next`.

- **Visible move:** animates normally.
- **Partially visible move:** the token animates only the hops the viewer can see — fades in at
  the first visible hop, fades out after the last. Pathing through the dark is never shown.
- **Attack from an unseen unit:** tracer spawns at the fog boundary along the true
  attacker→target line — real bearing, clamped origin, muzzle never rendered. Impact, explosion,
  and popup are normal. (Deliberate, small intel leak: bearing is counter-play information —
  fits the fog-as-balance-knob philosophy.)
- **Fully invisible actions** (unseen moves in the dark, hidden claims): skipped with **zero
  time cost**. Pacing must not create a timing side-channel about activity in the fog beyond
  what the game already reveals.

## 5. Input during playback

No input lockout. Local commands validate against true engine state, which is always current.
If a local action touches a unit with queued animations, the queue **fast-forwards**
(snap-commits) first. No desync window: visuals may lag, truth never does.

## 6. Camera

- During opponent actions only: if the action is off-screen, `CameraRig` smoothly pans focus
  toward it before playing; if visible, no movement.
- Any player camera input cancels the auto-pan immediately.
- Kills: short camera micro-shake alongside the existing explosion. No shake on ordinary hits —
  on mobile WebGL, restraint reads better than wobble.

## 7. Verification

- Engine tests untouched (no engine changes to make).
- Play-mode verification via coplay `execute_script` on `Assets/Scenes/HexWars.unity`: scripted
  AI-vs-AI spectator game with fog ON, watching for the three failure classes — teleporting
  tokens, popups before impact, anything animating in unseen territory.
- Explicit regression checks on the quiet details the `BoardRenderer` refactor could break:
  spent-unit dimming, click colliders, docked unit stats, territory tint.
- Denial sweep: deliberately click every refused action and read its reason toast; hotseat must
  show no "opponent's turn" notice (the active player is whoever holds the mouse).
- WebGL build + live smoke on Render, including mobile.

## 8. Success criterion

Watch a full game against the AI with fog on and **narrate the opponent's turn from the screen
alone** — every visible action legible, every hidden attack a tracer from the dark, nothing
teleporting. Frame pacing stays smooth on mobile WebGL.

## 9. Risks

- `BoardRenderer` refactor regressing quiet details (dimming, colliders, docked stats, tint) —
  mitigated by the explicit play-mode checklist above.
- Fog edge cases (units revealed/hidden mid-action) — mitigated by `Sync(next)` always
  committing truth after each action.
- WebGL perf: expected to *improve* (diffing allocates less than rebuild-everything).
