# Model Arena Identity Overlay Design

## Purpose

The ML Lab currently reports resolved models in its Editor window, but a person watching the Unity Game view cannot tell which controller owns each army, which checkpoint is loaded, or how the matchup is going. The arena needs a compact, always-visible identity display that remains accurate across live checkpoint reloads.

## Display

A dedicated model-arena overlay appears only while `ModelDuelDriver` is active. It is independent of the collapsible combat log, so hiding the log does not hide matchup identity.

The overlay contains one row per seat:

- `P1` in the existing cyan player color and `P2` in the existing red player color;
- an active-turn marker beside the seat currently choosing an action;
- controller identity: `Greedy`, `Random`, fixed checkpoint name, or live/fixed run name;
- model algorithm when applicable;
- resolved checkpoint filename and training step when applicable;
- that seat's current-session `W-L-D` record and win percentage.

For a session with `w` wins, `l` losses, and `d` draws, win percentage is `w / (w + l + d)`. Before any completed game it displays an em dash rather than dividing by zero. P1 and P2 records are mirrored from the same results: a P1 win is a P2 loss and draws count for both.

The concise arena label uses the run directory name and checkpoint filename rather than a full filesystem path. The existing ML Lab runtime panel retains the full resolved path and contract hash for diagnosis.

## Data flow

`ModelDuelDriver` remains the source of truth for seat configuration, resolved `PolicySeatInfo`, current seat, and session results. A small immutable presentation snapshot converts those values into display strings. The overlay reads that snapshot and has no knowledge of Python protocol details.

At bridge startup, model rows transition from `loading` to their resolved checkpoint. After a completed game, the driver requests live reloads. If a newer checkpoint is accepted, the next game's row changes before the first action; otherwise the existing checkpoint label remains. A checkpoint never changes mid-game.

Scripted seats always show their configured controller name and no algorithm, step, or checkpoint placeholder. Fixed and live model seats both show the actually resolved checkpoint, not merely the requested path.

## Layout

In landscape, the overlay is a compact two-row strip anchored away from the right-side combat log and the bottom playback controls. Each row keeps player color, identity, and record visually grouped. Long run or checkpoint names are middle-truncated while the step and record remain visible.

In portrait or narrow Game views, the overlay stacks the identity and record onto two short lines per seat and omits the redundant checkpoint extension before hiding meaningful data. It must remain readable at the same logical scaling used by the existing playback UI.

## Error handling

While a model is resolving, the row says `loading checkpoint`. If bridge startup or reload fails, the row retains the last successfully resolved checkpoint and adds a concise `reload failed` state; the detailed error remains in the Unity log and ML Lab status. Missing metadata displays `unknown algorithm` or `step unknown` rather than inventing values.

## Verification

EditMode tests cover scripted labels, fixed checkpoint labels, live run labels, active-seat markers, long-name truncation, zero-game records, win/loss mirroring, draw accounting, and win-percentage formatting. Driver tests verify that the display snapshot updates after bridge resolution and reload without changing during a game.

Manual verification launches Greedy versus a fixed checkpoint and two live runs, collapses the combat log, changes Game-view aspect ratios, and confirms that controller identity, checkpoint step, active seat, and cumulative record stay visible and accurate.

