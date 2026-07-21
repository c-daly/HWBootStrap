# Setup, Barracks, and Deployment QoL Design

## Scope

This change improves match setup, removes browser input prompts, carries barracks choices across games during one running app session, makes template statistics easy to inspect, and adds a deployment phase before round one. Audio is out of scope because volume controls already exist. The intelligent movement-preview work is tracked separately.

## Inline input and setup limits

All editable values use ordinary Unity UI input fields populated with their current defaults. The project no longer uses `window.prompt` for setup values, room codes, seeds, army composition, or template names. Enter or focus loss commits an edit; Escape restores its previous committed text. Validation appears beside the field and never opens a modal alert.

A blank numeric field commits as `0` only when zero has a defined meaning for that field. For example, a blank turn-action limit becomes `0`, meaning no automatic limit. A blank width or height is invalid, remains visibly marked, and prevents match creation until corrected. Validation occurs on commit and match creation rather than on every keystroke.

Board width and height accept `5` through `64`, enforced by both the form and `GameSetup.Sanitized()`. The turn-action field accepts any non-negative 32-bit integer. It retains the current `KActionsPolicy` meaning exactly; this project does not change the turn model to distinct-unit activation. The normal turn-action default remains `3`, while `0` explicitly selects whole-team/unlimited play. The existing preset cycle and numeric plus/minus steppers are removed.

Room codes are entered inline on the title screen. Template names are entered inline in the designer. After all callers migrate, the WebGL browser-prompt bridge is removed.

## Session barracks cache

The running client owns an in-memory barracks cache for each locally controlled player. Each cache begins with the five built-in starter templates. A server-confirmed or locally accepted template creation adds the sanitized template to the appropriate cache. Deleting any template, including a built-in template, removes it from that cache. Exact name-and-stat duplicates appear once.

Starting another game copies the cache into the corresponding human player's new `PlayerState`. The cache follows the human rather than a fixed player number, so a player seated as Player 2 receives the same barracks. Local hotseat keeps one cache per local player. AI players continue to use the default starter templates. Closing or reloading the app clears every cache and restores the built-in defaults; no disk, browser storage, account storage, or cross-session migration is included.

When hosting or joining an online game, the client sends its sanitized cached templates as seat-initialization data. The authoritative server validates the count, names, and stats before attaching them to that seat. Barracks edits are copied into the cache only after the authoritative apply succeeds, so rejected or optimistic commands cannot corrupt the next game's catalog. Protocol limits bound template count and payload size.

## Barracks stat information

Every barracks row has a non-modal information surface. Pointer hover or keyboard focus shows an anchored card containing the full template name, dominant role, point cost, deploy cost under the current match configuration, and all nine statistics with full labels: Health, Damage, Defense, Movement, Vertical Move, Range, Range Arc, Vision, and Vision Arc.

The card remains while the row is hovered or focused, disappears on exit, and never intercepts the row's select action. Touch layouts expose the same card through a small information target because touch has no hover. This information surface is not an input prompt and does not dim or block the board.

## Pregame deployment

`GameState` gains an explicit deployment phase and readiness state for both players. New matches still create and place the configured starting armies automatically, but combat commands, economy actions, and turn-budget accounting remain disabled until deployment completes.

The automatic layout no longer places the army in role-grouped order. It interleaves roles across the formation and uses independent deterministic seed derivations for the two players, avoiding predictable mirrored formations while preserving reproducible matches and replays.

During deployment, a player may repeatedly select one of their starting units and relocate it at no cost to any passable, unoccupied hex in their existing deployment zone. The UI highlights that zone and valid destinations. Units may not leave the zone, overlap another entity, attack, claim, build, deploy reinforcements, or edit the opposing army. Ready is final for that army.

Online players may arrange simultaneously. The server accepts deployment commands from either unready seat, validates them authoritatively, and broadcasts the resulting state. Opposing units remain hidden during deployment regardless of the match's fog setting. In local hotseat, players deploy sequentially with an obscuring handoff screen between them. A local AI may accept or deterministically adjust its mixed automatic formation and then marks itself ready. When both players are ready, the game enters normal play at round one with Player 1 active and empty movement/attack bookkeeping.

Deployment phase and readiness are included in network state and replay serialization. Older replay payloads without this information load as already in normal play, preserving compatibility.

## Error handling and authority

Client-side input feedback is immediate but never authoritative. `GameSetup.Sanitized()`, online template validation, and deployment-command validation remain server/engine boundaries. Malformed room codes, invalid names, oversized template payloads, invalid coordinates, occupied cells, and commands submitted after Ready are rejected without changing cached or match state. Rejections use inline messages or the existing toast system, not browser dialogs.

## Verification

Engine tests cover setup sanitization through `64`, unrestricted non-negative turn limits, blank-field parsing policy, template-cache add/delete/deduplication, online catalog validation, mixed deterministic formations, every deployment-command rejection, simultaneous readiness, transition to round one, replay compatibility, and preservation of existing turn-policy semantics.

Unity EditMode tests cover input commit/cancel/validation behavior, title/designer inline fields, barracks hover/focus/touch information, cache ownership by the local human seat, valid-zone highlighting, hotseat handoff, and input suppression during deployment. Manual PlayMode/WebGL checks cover keyboard focus, mobile soft-keyboard entry, a full local/AI deployment, two-browser simultaneous deployment, return-to-menu/new-game barracks persistence, and session reset after browser reload.

## Non-goals

- Persisting barracks across app or browser sessions.
- Replacing the current action-counting turn policy with distinct-unit activations.
- Changing combat, fog, deployment-zone geometry, unit design costs, or reinforcement rules after round one.
- Making opponent barracks visible.
- Adding or changing audio controls.
