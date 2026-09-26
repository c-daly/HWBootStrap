# Easier battlefield controls

Branch: `codex/smooth-movement-20260926`, based on the complete web build at `619515a`.

- Select a piece and click a reachable hex to move. Hover previews the route and its cost.
  Starting placement also takes one click, and can be rearranged freely before Ready.
- **Undo move**, or **Ctrl+Z / Command+Z**, restores the most recent move, elevation, movement
  budgets and action allowance. Both players see the same result after the server accepts it.
- Gold **IN RANGE** markers stay visible for the selected piece while examining another route.
  Muted **AFTER MOVE** labels identify additional targets available from the proposed destination.
- Click an enemy to inspect the damage, then click that enemy again or use **Fire**. There is no
  double-click timing requirement. Changing targets or pressing Escape does not fire.
- Clicking a spent squad card selects a matching unit with a legal move or attack remaining.
  Clicking a piece on the board selects that exact piece; when every matching unit is spent,
  the card still lets you inspect it. Selecting another piece returns the controls to movement.

Only the latest movement can be undone. Another accepted action, including an attack or a turn
handoff, makes it final. A subsequent move replaces the previous undo checkpoint. Undo is disabled
with fog of war so it cannot be used for free scouting. Moves that exhaust the turn allowance still
hand play to the opponent; their hover preview says that they cannot be undone.

Undo is an authoritative engine command, recorded alongside moves in the command log. Reconnecting
rebuilds the same undo availability by replaying the log. A client waits for acknowledgement before
changing the board, and a rejected or duplicate undo cannot rewind either player's state.

Deploy this branch's matching server and rebuilt WebGL client together. Existing browser sessions
must reload after deployment to understand the new command. See [web deployment](webgl-preview.md).
The previous audio changes and the H mark are included. No live deployment or PR merge is performed
by this work.

Validation and screenshots will be recorded with the completed build.
