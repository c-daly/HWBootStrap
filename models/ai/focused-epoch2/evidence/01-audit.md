# Anchored candidate: development rejected

The independent offline audit passed. The candidate remains unpromoted. The same selected checkpoint was used throughout the evaluated phase(s).

## Development: failed

| Controller | Wins | Draws | Losses | 2W+D |
|---|---:|---:|---:|---:|
| focused-epoch2 | 56 | 15 | 25 | 127 |
| closing-mirror | 52 | 19 | 25 | 123 |
| anchored-candidate | 58 | 14 | 24 | 130 |
| greedy-control | 54 | 5 | 37 | 113 |

Failed preregistered gates: seat-1-retention-vs-focused-epoch2, scenario-default-24x16-hills-retention-vs-focused-epoch2, development-improvement.

Against the paired focused baseline: +2 wins and +3 integer score; 8 improved outcomes and 6 worsened outcomes. These changes do not identify which actions caused the result.

## Evidence and limits

The audit authenticated 384 exact scheduled game receipts and 768 trace/replay files, matching initial physical states across controllers, terminal outcomes, controller aggregates, and every fixed gate. Source, runtime, protected artifacts, checkpoint, preflight, and master bindings remained unchanged.

The qualified 2048-expansion teacher scored 38W/3D/7L against Greedy on its separate fresh 48-game panel versus the legacy teacher’s 31W/6D/11L. That teacher result justified collecting labels; it does not guarantee that imitation reproduces the teacher’s gameplay.

This candidate used 511 fresh training rows and 265 heldout rows across four equally weighted tasks, with the focused policy fixed as a 0.5 KL reference. Four epochs/32 updates selected epoch4 before gameplay. Heldout task-weighted label NLL improved 3.330034→3.021113, while exact label agreement changed 58→54/265. Positive-attack selection changed 52→54/74 and lethal selection 31→33/41. These label diagnostics are separate from the game results above.

Confirmation: no confirmation artifacts or recorded execution; bank remains reserved.

This is tactical combat with fixed templates, bounty income and redeployment. It does not test fog, generators, capture, territory, custom unit design, or the historical mixed-task acceptance gate. No model promotion, runtime integration, publication or further training was performed by this audit.

Full machine-readable evidence: [audit.json](audit.json). Training detail: [TRAINING.md](TRAINING.md).
