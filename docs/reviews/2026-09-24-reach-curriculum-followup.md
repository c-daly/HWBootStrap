# Reach curriculum review follow-up

PR #20 retains the existing reach objective, fixed target, teacher identity, scenario hashes and publication gates. These guards prevent invalid inputs from producing misleading collection evidence:

- Reach catalogs require positive movement for every template. Starting armies sample with replacement, so one immobile template can fill an entire learner roster. Both scenario loading and runtime configuration reject this before reset. Legacy annihilation catalogs still accept immobile units.
- If every legal move leaves the fixed beacon unreachable, teacher selection raises an explicit error without emitting a label or advancing the episode. This can happen when a moving opponent occupies the beacon or blocks the only path. The target is not moved, opponents are not restricted, and no arbitrary shortest-path label is recorded.
- When units have no legal moves, the existing end-turn behavior remains. Passive-opponent reachable episodes keep their prior teacher decisions and provenance.

The standard passive Reach Beacon template remains appropriate for this shortest-path teacher. A moving-opponent run must treat an unreachable-target error as an incomplete collection, not as a successful or publishable run. No saved collection was relabeled, training run restarted, checkpoint promoted, or criterion relaxed in this follow-up.

Validation receipts are retained locally under `Library/PrFollowup-20260924/`. The focused suite covers all-immobile and mixed catalogs, both learner seats with an opponent legally occupying the fixed target, and unchanged passive-opponent behavior. The final full engine/GymServer suite passed **1,047 tests, zero failed or skipped**, on .NET 8.0.31. Unity 6000.5.0f1 compiled the isolated project with the updated Release engine DLL and exited 0 with no C# compiler errors. This branch has no PlayMode test assembly; the full engine suite includes determinism and GymServer process checks. No interactive Arena rendering is claimed.

An earlier full engine run reported one failure in its legacy “valid reach scenario” fixture because that combat-derived fixture included immobile artillery. The fixture now removes immobile artillery when testing a valid reach configuration, and a separate test requires the original mixed catalog to fail during scenario loading. Original failing receipts are retained.
