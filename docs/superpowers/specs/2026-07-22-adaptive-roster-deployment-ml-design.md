# Adaptive Roster and Deployment ML Design

## Purpose

The current tactical ML contract starts both armies in fixed, predictable cells and limits the policy to moving, attacking, and deploying copies of three baked-in roles. This project expands that contract so training can teach combined-arms composition, situational unit design, and hidden pregame deployment without creating one enormous flat action space.

This is a new, incompatible ML contract. Existing tactical-v1 checkpoints remain inspectable and playable through the existing environment, but they cannot be resumed or silently loaded into the new contract.

## Scope

The first adaptive contract supports:

- six curated, named unit roles with distinct tactical uses;
- three custom barracks slots that the policy may redesign during a match;
- deployment from any legal cell in the acting player's starting area;
- a hidden pregame deployment phase in which each player independently places its starting army;
- the existing move, attack, reinforcement-deploy, and end-turn behavior after deployment;
- the same shared codec for headless training and Unity model duels.

It does not initially expose arbitrary board dimensions, every human setup option, continuous-valued actions, simultaneous mutable policies, or a player-facing unfinished model selector. Those remain later contract versions or promotion work.

## Roster

Each player receives nine barracks slots. Slots 0 through 5 are immutable curated templates: durable frontline, close-range damage, long-range precision, artillery, reconnaissance, and a mobile support/generalist role. Their exact `UnitStats`, names, and costs are contract data and therefore participate in the semantic contract hash.

Slots 6 through 8 are mutable custom templates. At reset they contain conservative general-purpose defaults so every slot is immediately usable. A design action replaces one custom slot's complete template; it does not append a tenth slot. Fixed roles cannot be redesigned.

The environment offers a finite catalog of legal stat changes rather than treating nine stats as simultaneous continuous values. Unit design is a short phase sequence: choose custom slot, choose stat category, choose the new legal value, and confirm. At every stage, the action mask exposes only values accepted by `GameEngine`, including cost and stat-budget constraints. Cancelling returns to the normal command phase without changing the template.

This retains meaningful flexibility while keeping each policy decision small and maskable. The three independent custom slots let the policy preserve successful designs while adapting another slot to a new need.

## Hierarchical action protocol

Python continues to expose a single `Discrete(n)` action space because SB3 MaskablePPO expects a fixed discrete space and mask. The C# engine interprets that discrete choice through a per-seat decision phase stored by the RL environment, not in authoritative `GameState`.

The root phase contains commands such as end turn, choose a unit, deploy a template, redesign a custom slot, and confirm pregame deployment. Selecting a command that needs parameters transitions to another decision phase. Subsequent phases select one item from a fixed maximum-sized region: unit slot, template slot, board cell, stat category, or stat value. Invalid and unused entries are masked.

Only a completed sequence applies a `Command` to `GameEngine`. Intermediate choices produce zero game-time advancement and a small configurable decision penalty so the learner cannot profit from cycling through menus. Cancel is legal in every non-root phase and clears the pending selection.

The observation adds one-hot decision-phase globals and normalized pending selections. This makes an action's meaning Markovian: the model can always tell which parameter it is currently choosing. The semantic contract records phase names, region offsets, maximum counts, and meanings so equal tensor sizes with different semantics are rejected.

## Unit identity and reinforcement capacity

The current codec assigns one permanent action slot to each initial unit, so reinforcements cannot be controlled once every slot is occupied. The adaptive contract replaces that mapping with a fixed maximum controllable-unit table per seat. Living units occupy stable slots for their lifetime; a dead unit releases its slot; newly deployed units receive the lowest free slot.

The maximum is contract data and must cover the configured tactical army plus plausible reinforcements. When no slot is available, reinforcement deployment is masked even if the game economy would otherwise allow it. This limitation is explicit in the run manifest and Unity ML Lab rather than hidden in fallback behavior.

## Hidden pregame deployment

Each episode begins in a deployment phase before round one. Both seats receive the same roster and army budget, but each observes only its own placements, its own starting zone, public terrain, and public setup values. Opponent placements are absent from observations until both seats confirm.

Deployment is sequential internally for deterministic training, but information is simultaneous: the second policy receives no data produced by the first policy's hidden choices. A seat selects a roster/template slot and then a legal empty starting-zone cell. It may reposition an already placed unit, remove it back to the unplaced pool, or confirm once all required units are placed and the budget is valid.

When both seats confirm, the environment atomically reveals the armies, creates the normal round-one `GameState`, and begins with the configured first player. The same seed, policy actions, and configuration reproduce the same deployment and match.

Scripted opponents need a deployment adapter. The first adapter distributes a combined-arms roster across legal zone cells using seeded scoring for terrain, spacing, range, and frontline depth. It must not read the learner's hidden placements. Random remains available as a seeded diagnostic deployment policy.

## Observation and fog of war

The adaptive observation keeps seat-relative board planes and adds:

- visibility and previously-seen terrain planes required by the selected fog configuration;
- per-role and custom-slot unit planes for friendly and currently visible enemy units;
- barracks template stats, normalized costs, custom/fixed flags, and occupied unit-slot summaries;
- deployment pool and remaining setup budget;
- decision phase and pending parameter selections;
- the existing points, round, army-count, and relevant rules globals.

Enemy units hidden by fog are never encoded as current units, legal targets, deployment hints, or Unity arena labels. The pregame deployment phase applies the stricter rule that all opposing placements are hidden until both players confirm, even if starting zones would otherwise be visible.

## Environment boundary and compatibility

`TacticalEnv` and `DuelEnv` continue sharing one layout, observation encoder, phase state, mask builder, and decoder. Python remains a transport and training layer; it does not duplicate legality rules. GymServer reports both the legacy tactical-v1 environment and the adaptive environment by explicit kind so old runs remain reproducible.

The adaptive contract receives a new version and hash. A run manifest records curated templates, custom defaults, stat value catalogs, maximum controllable units, deployment rules, observation channels, phase table, and action-region meanings. Model inspection reports a clear compatibility error before inference when any field differs.

Training runs use the adaptive environment only when explicitly selected. Existing presets continue selecting the legacy environment until adaptive smoke and reciprocal evaluation tests pass.

## Rewards

Normal combat and terminal rewards remain recognizable so results can be compared with tactical-v1. Additional shaping is limited to:

- a small penalty per intermediate design/deployment decision;
- no reward for merely creating or repeatedly replacing a template;
- normal economic/army-value consequences once a design is deployed;
- a deployment completion bonus only if experiments show policies otherwise refuse to confirm, recorded as contract data when enabled.

Evaluation separately reports design count, distinct custom templates deployed, deployment completion rate, illegal/fallback actions, and average pregame decisions. These diagnostics are never folded into win rate.

## Unity ML Lab and arena

The ML Lab exposes the environment contract as a deliberate selection. Adaptive runs display the curated roster, custom-slot count, maximum units, starting-army budget, and deployment policy in preflight and run details. Training remains headless; the Unity arena renders independent games from completed checkpoints.

During a watched adaptive game, Unity shows the hidden deployment process only from the selected observer's legal perspective or skips directly to reveal. It must not reveal the opposing deployment through unit objects, logs, highlights, camera framing, or model metadata.

## Error handling

If a pending action becomes illegal because state changed, the environment clears the phase, records an invalid-sequence diagnostic, applies no command, and returns a legal root mask. A policy can never cause an out-of-range lookup or bypass `GameEngine` validation.

If a seat cannot complete deployment because configuration exceeds available cells or budget, preflight rejects the run with the exact required and available counts. If every non-root option is masked unexpectedly, cancel remains legal. If no root gameplay action is legal, end turn remains the safe fallback.

## Verification

Engine tests pin the complete phase/action table, mask/decode round trips, custom-slot replacement, fixed-slot immutability, unit-slot reuse, deployment hiding, confirm rules, deterministic setup, fog-safe observations, and semantic contract hash. Property-style tests sample masked actions across many seeds and assert that every exposed completed sequence is accepted by `GameEngine`.

GymServer and Python tests verify explicit environment selection, stable spaces, vectorized masks, run-manifest compatibility, legacy checkpoint isolation, and headless throughput. Duel tests run two external controllers through deployment and combat and reconstruct the final replay.

Unity EditMode tests cover ML Lab summaries and observer-safe deployment presentation. Manual verification watches two adaptive checkpoints deploy and play several games, confirms that live reloading occurs only between games, and checks that no hidden enemy placement or fogged unit appears.

