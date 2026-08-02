# Task 11 report: end-to-end smoke gate

## Outcome

The real isolated smoke gate completed and atomically published
`python/panels/annihilation-imitation-v1/evidence/smoke/smoke.json`. The
generated dataset, model archives, traces, replays, and preserved failed-attempt
diagnostics remain local and uncommitted.

The published manifest records:

- seven reciprocal pairs and 14 games;
- 1,815 teacher labels, all legal under their recorded masks;
- zero mask, action round-trip, and replay mismatches;
- one BC epoch using training rows as a smoke-only validation view;
- exact CPU actor transfer (`maximum_absolute_logit_difference = 0.0`);
- two PPO timesteps, one completed physical rollout, and a reloaded checkpoint;
- four evaluation games over two unused maps from both seats.

The manifest binds 57 physical artifacts, including the runtime scenario and
its provenance. `stage.json` binds the completed stage to the panel, scenario, and
seed-bank definition hashes.

## Command

The successful run used the required repository interpreter:

```powershell
$env:PYTHONPATH='python'
C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe python\run_annihilation_imitation_panel.py smoke
```

Re-running the command reopens every recorded artifact and returns the completed
stage with `reused = true`; it does not recollect, retrain, or reevaluate.

## RED/GREEN defect evidence

Every behavior change was driven by a focused failing test before its fix:

1. The panel initially had no `smoke` CLI boundary. Parser, exact schedule,
   manifest, isolation, atomic-publication, command-dispatch, and composition
   tests failed before the command and pipeline were added.
2. The panel launched stale Release GymServer bytes after a Debug build. The
   server-command regression failed until the panel used the required Debug
   output.
3. Real capture exposed accepted tactical-v2 `CreateUnit` and `DeployUnit`
   commands outside the fixed action codec. Production-profile tests failed on
   the exact seed transitions until tactical-v2 closed creation, capture, and
   generator command families while retaining the fixed deploy catalog.
4. A round-cap draw serialized `winner: null`. Engine and GymServer wire tests
   failed until tactical-v2 used the protocol's integer `-1` sentinel.
5. BC capture and PPO training have intentionally different full contract
   hashes because their duel/tactical horizons differ. Actor initialization
   rejected this compatible source until it relied on the existing resolver's
   environment/version/encoding/geometry checks and recorded the actual source
   contract hashes.
6. Restart tests failed until completed BC output was physically validated and
   reused, completed PPO output was reopened, and a failed PPO attempt was moved
   by `os.replace` to deterministic sibling `.smoke.recovery` outside the
   publishable tree. No failed artifact was deleted or entered `smoke.json`.
7. Evidence-chain tests failed until the dataset was reopened against the BC's
   exact source contract, PPO compatibility was rechecked, and every actor
   initialization source hash was recomputed. A final callable-boundary
   regression caught and repaired a missing evidence-collector function header
   before publication.

Focused GREEN evidence included the three recovery/source-chain tests, the
callable-boundary regression, ten panel smoke tests, the full dataset audit,
and compatible actor-transfer regression.

## Verification

- `dotnet build engine\HexWars.GymServer\HexWars.GymServer.csproj --nologo`: passed, zero warnings and zero errors.
- `$env:PYTHONPATH='python'; C:\Users\cddal\HexWars\python\winenv\Scripts\python.exe -m pytest python\tests -q`: passed, 572 tests and one expected scenario auto-raise warning.
- `dotnet test engine\HexWars.Engine.Tests\HexWars.Engine.Tests.csproj --nologo`: passed, 653 tests.
- `git diff --check`: passed.

## Interpretation limit

The one-epoch BC reuses its training decisions as validation rows only to drive
the trainer and transfer boundaries. Those metrics are not held-out evidence
and make no claim about policy quality or generalization. The smoke proves the
end-to-end mechanics and evidence integrity, not that the full experiment's
winning-model gate has passed.
