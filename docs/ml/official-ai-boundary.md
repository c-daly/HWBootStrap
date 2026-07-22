# From experiment to the official HexWars AI

The ML Lab is a developer experiment surface. Its arbitrary checkpoint paths, live run directories, tracker settings, experimental algorithms, and candidate artifacts must not appear in regular player menus or builds. Today, Greedy remains the safe gameplay controller unless and until an official trained artifact is deliberately shipped.

## Why a lab checkpoint is not the official AI

The current learned-policy contract describes a fixed tactical scenario. The regular game permits broader board sizes, fog, economy, unit design, rosters, deployment, and turn options. A model can achieve an excellent held-out score while lacking observations/actions for those systems. It can also be too slow or large for WebGL, depend on Python/SB3, fail a platform operator, or regress a difficulty/accessibility expectation.

`publish-checkpoint` therefore creates only a named Editor-lab candidate. It does not modify player settings, game scenes, WebGL assets, or the official opponent selector.

## Proposed promotion pipeline

An official-AI project should make each boundary explicit:

1. **Choose the supported gameplay contract.** Specify board/roster/setup/economy/fog/turn coverage, observation/action semantics, legal masks, reward-independent evaluation, and difficulty requirements.
2. **Select an evaluated candidate.** Require reciprocal held-out suites against Greedy, its parent, the previous champion, and relevant scripted scenarios; report W/L/D, sample size, confidence intervals, illegal/fallback count, game length, and inference latency.
3. **Freeze provenance.** Package source run/checkpoint, code commit, engine and contract versions, dependency versions, evaluation report, and human approval. Never reference `latest` or a mutable run directory.
4. **Export a runtime model.** Convert the supported policy to a versioned ONNX (or another reviewed portable format) and prove numerical/action agreement against the Python source on a golden observation corpus.
5. **Import with Unity inference.** Use a pinned Unity Sentis/Inference Engine package and an immutable project asset. Validate every output through the current legal-action mask and `GameEngine.Apply`.
6. **Qualify platforms.** Measure WebGL download/build-size impact, browser memory, warmup, per-decision latency, mobile/desktop behavior, deterministic fallbacks, and unsupported operator/backend behavior.
7. **Integrate difficulty.** Prefer separately evaluated fixed artifacts or well-defined decision-time behavior. Do not expose arbitrary experiments as a difficulty dropdown.
8. **Release deliberately.** Version the official artifact and compatibility manifest, add build/scene tests, canary it, and keep a rollback path.

## Runtime package

The eventual immutable package should contain at least:

```text
OfficialAi/<version>/
├── model.onnx
├── contract.json
├── provenance.json
├── evaluation.json
├── golden-observations.json
└── release-notes.md
```

`provenance.json` identifies the source run, exact checkpoint step/hash, algorithm/policy, training code commit, dependencies, exporter, and approval. `contract.json` pins semantic observation/action mappings, not only dimensions. The build references the explicit version.

Python remains a training/evaluation dependency and is not bundled into WebGL. The Unity runtime loads only the promoted portable artifact. Before adopting Sentis/Inference Engine, verify the exact package version supported by the repository's Unity release and test the exported operators on every target platform.

## Safety and fallback

Runtime inference is advisory until the engine accepts a legal action. Required defenses are:

- validate model and contract identity at load;
- bound initialization and decision time;
- mask illegal actions before selection and validate again through `GameEngine.Apply`;
- count and log invalid outputs in development/telemetry where permitted;
- fall back to Greedy (or another deterministic safe controller) when load, inference, timing, or validation fails;
- never reveal fog-hidden state in the observation or debugging presentation;
- keep a remote-disable/rollback strategy appropriate to the release channel.

Any illegal/fallback action in the official evaluation suite should fail promotion unless a documented diagnostic waiver exists. A fallback is resilience, not a way to hide an incompatible model.

## Definition of official-ready

A candidate is not official-ready until all of these are true:

- the full intended gameplay contract exists and has semantic-hash tests;
- reciprocal evaluation gates and confidence thresholds are written before testing and pass;
- golden Python-versus-Unity inference agreement passes;
- WebGL and other target performance/build-size budgets pass on representative hardware;
- difficulty and player experience are playtested;
- load/inference/fallback/rollback tests pass;
- the package is immutable, versioned, reviewed, and intentionally referenced by the build.

Until then, experiments remain available only under Unity Editor developer tooling. Regular players may select the supported official opponent/difficulty, never a filesystem path, unfinished model, live run, algorithm, tracker, or training control.
