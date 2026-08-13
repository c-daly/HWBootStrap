# Project A: Generalizable Structured Imitation — Completion Report

Date: 2026-08-11

Status: complete as unsealed experimental evidence. Project B may consume the structured contract, subject to the limitations below.

## Commits

- `53e9dfb` feat: define tactical-v3 structured schema
- `1895505` docs: target real tactical-v3 test project
- `5011a1b` docs: move tactical fixture seed to state creation
- `83ba441` test: cover tactical-v3 config constraints
- `1b2193a` feat: project tactical-v3 seat observations
- `708ba73` fix: reflect tactical-v3 seat coordinates
- `6187c8f` feat: add tactical-v3 command candidates
- `f512b2a` docs: bind tactical resolver to decision identity
- `5df1193` fix: enforce tactical-v3 decision identity
- `6f7af4a` test: tighten tactical-v3 candidate surface guard
- `d20bbf5` feat: add tactical-v3 annihilation reward
- `7071422` fix: freeze tactical-v3 terminal reward
- `d1e9b38` feat: run tactical-v3 structured duels
- `3cbbe0e` fix: preserve tactical-v3 episode authority
- `ca6a65b` fix: freeze tactical-v3 facade profile selection
- `77e3979` feat: load tactical-v3 training scenarios
- `ec5f8ba` fix: harden tactical-v3 scenario validation
- `c50c7f8` fix: align tactical-v3 structural capacities
- `74aa005` feat: hash tactical-v3 semantic contracts
- `49c85c1` feat: serialize tactical-v3 structured views
- `13e8bb6` fix: validate tactical-v3 wire evidence
- `62ff7c6` feat: expose tactical-v3 through GymServer
- `0e0375f` fix: harden tactical-v3 GymServer contract
- `1dcc4e8` fix: validate tactical-v3 request value types
- `3f675b3` docs: keep project-a report in tracked docs
- `be92556073636d2f104c741390b00ec1605bf43c` test: complete tactical-v3 contract gate
- Review hardening: `test: harden tactical-v3 conformance proofs` (the current commit; its hash is unavoidably self-referential here).
- Final-review closure: `fix: close tactical-v3 final review findings` (the current commit; its hash is unavoidably self-referential here).

## Acceptance evidence

- GymServer build: `dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore` — succeeded with 0 warnings and 0 errors.
- Review-hardening selector: the six new/strengthened conformance tests passed 6, failed 0, skipped 0.
- Complete tactical-v3 selector: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter FullyQualifiedName~TacticalV3 --no-restore` — 269 passed, 0 failed, 0 skipped.
- Full engine/GymServer regression: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --no-restore` — final exact run passed 982, failed 0, skipped 0.
- Two prior exact full-suite runs each passed 980 and timed out in two different existing GymServer rejection cases at their fixed 10-second process-exit wait. Every named case passed when isolated (7/7 and 2/2), the tactical-v3 cases passed in the 269-test selector, and the final unchanged exact run passed 982/982. No timeout or production workaround was added.
- Live Unity/Coplay root: `C:\Users\cddal\HexWars\.worktrees\physical-checkpoint-audit`.
- Live Unity state: `playMode=false`, `hasCompilationErrors=false`.
- Live Unity compile result: `No compile errors`.
- Live Unity log result: empty 200-entry console query; no tactical-v3 exception was present.
- JSONL smoke: `spaces`, `reset` with seed 41, legal `step` selecting decision 0/candidate 0, then `close`.
- Smoke result: 10 reset candidates; selected kind `move`; successor decision 1 with 9 candidates; close exit 0.
- Smoke schema result: no top-level flat `obs` or `mask` in spaces, reset, or step; structured observation tables and decision-local candidates were present.

## Final-review closure evidence

- Focused TDD: 11 confirmed regressions failed before their production fixes and then passed 11, failed 0.
- Direct runtime configuration now rejects every stage-one path that can emit unsupported capture mechanics, requires positive template health, and validates the exact point-cost, deploy-cost, bounty, and reachable-points bounds needed to keep integer mechanics safe.
- JSON tactical-v3 scenarios reject zero-health templates while legacy tactical-v2 validation retains its prior zero-health behavior.
- Candidate and projected-delta cell references use the same canonical coordinate-to-row authority as observation cells; two reversed-board-insertion seat regressions prove coordinate stability.
- Defensive reward evaluation remains finite and within the published shaping/total bounds even when handed an invalid zero-health state.
- Every finished frame, including exact-step truncation, exposes zero candidates; a further step is rejected without changing replay commands.
- `TacticalV3Wire.View` validates all eight observation-table counts and candidate count against the configured capacity before serialization. Live GymServer callers pass the validated scenario capacity explicitly.
- The authoritative rectangular topology comment now correctly names odd-q.
- Final tactical-v3 selector: 278 passed, 0 failed, 0 skipped.
- Final full engine/GymServer suite: 991 passed, 0 failed, 0 skipped.
- Final GymServer build: succeeded with 0 warnings and 0 errors.
- Final structured JSONL smoke: `spaces`, seed-41 `reset`, decision-0/candidate-0 `move`, successor decision 1, then `close`; reset had 10 candidates, successor had 9, structured rows were present, flat `obs`/`mask` were absent, and exit was 0.
- Final Coplay state: `playMode=false`, `hasCompilationErrors=false`.
- Final Coplay compile result: `No compile errors`.
- Final Coplay 200-entry log query: empty.

## Checked-in scenario identity

Scenario: `python/config/annihilation-structured-imitation-v1.json`

- Environment: `tactical-v3` / tactical single-agent role.
- Board: 13x9.
- Contract hash: `0ae48260cde97bce9ed75975874676a262588b3ed17963cdb41d09d09d3088ce`.
- Encoding hash: `e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6`.
- Capacity hash: `7aea1db4f008dc192e83811b2c13abd8ce2304d2a6a209f37f9847be5f367364`.
- Non-empty start-profile and start-distribution rows are pinned by exact public property sets, numeric types, values, and ordinal profile order.

## Cross-size and independent conformance

- The same `TacticalV3SeatObservationSource` and structured DTO schema produced 117 cell rows for 13x9 and 384 cell rows for 24x16.
- The two sizes have the same encoding hash and capacity hash, while their match contract hashes differ.
- Both seat views on an asymmetric terrain/elevation/control/deployment sentinel board were checked with an independent odd-q offset/axial reflection oracle, plus self/opponent owner and deployment-zone swaps. Production `MirrorCell` is not used as expected authority.
- For every legal candidate in a distinguishable asymmetric all-four-kind state, the hidden authoritative command was obtained through `IActionResolver.Resolve` and applied to one independently recreated state. A command reconstructed solely from public candidate/token rows was applied to another. Complete successor config, policy, terrain, board, control, zones, players, resources, barracks, units, generators, turn, bookkeeping, terminal, and winner fields matched before the public projected delta was independently checked against that successor.
- Observation, relation, candidate, and projected-delta token references were range-checked and constrained to their semantic expected table families after projection and immediately after every accepted environment step.
- Two seed-149 trajectories matched complete public observations, candidates, selected command fields, rewards, terminal/truncation state, and exact replay text for exactly 10 positive commands; both ended `terminated=false`, `truncated=true`.
- A cycle-safe learned-DTO graph traversal rooted at observation, candidate, projection, and token-reference types inspects public instance properties and fields through nullable, array, collection, generic, and nested DTO paths. It rejects `Name`, `DisplayName`, `EngineId`, `UnitId`, and any raw `PlayerId`; adversarial and safe DTOs self-characterize every path.
- A scenario with `max_cells=1` was rejected at startup with empty stdout, before reset payload publication.
- Reversing the authoritative board tile insertion order does not change either seat's observation coordinates, candidate cell coordinates, projected source-cell coordinates, or projected destination-cell coordinates.
- Tactical-v3 JSON and direct runtime boundaries both require template health of at least 1; defensive reward handling prevents non-finite output if an invalid state bypasses those boundaries.
- Truncated frames and terminal frames both have empty candidate lists, matching the step-rejection lifecycle.
- The GymServer view boundary rejects every configured table-capacity overflow before payload construction.
- Legacy payload-shape contract tests remained green for tactical-v1, adaptive-v1, and tactical-v2 as part of the final 982-test suite.

## Public interfaces

- Schema and semantics: `TacticalV3TableKind`, `TacticalV3TokenRef`, capability/action/relation enums and descriptors.
- Observation: `TacticalV3Observation`, its eight structured token/definition tables, `IObservationMemory`, `ISeatObservationSource`, and `TacticalV3SeatObservationSource`.
- Decisions: `TacticalV3Candidate`, `TacticalV3ProjectedDelta`, `TacticalV3DecisionFrame`, `ILegalCandidateSource`, `ICandidateProjector`, `IActionResolver`, and their tactical-v3 implementations.
- Reward and environments: `TacticalV3RewardBreakdown`, `IRewardContract`, `TacticalV3Reward`, `TacticalV3DuelEnv`, `TacticalV3Env`, and `TacticalV3View`.
- Learned-feature boundary: `TacticalV3View.Seat` remains control-plane envelope metadata and is explicitly outside the learned observation/candidate/projection/token graph.
- Contract/configuration: `TacticalV3CapacityProfile`, `TacticalV3RewardConfig`, `TacticalV3Config`, and `TacticalV3Contract` with independent contract, encoding, and capacity hashes.
- GymServer protocol: tactical-v3 `spaces`, `reset`, `step`, `duel_spaces`, `duel_reset`, `duel_step`, `duel_save`, and `close`; structured selections require exact decision/candidate identity.

## Project-A tracked file set

- Plan and scenario: `docs/superpowers/plans/2026-08-10-tactical-v3-structured-environment-contract.md`; `python/config/annihilation-structured-imitation-v1.json`.
- Engine: `engine/HexWars.Engine/ReplayFile.cs`; `engine/HexWars.Engine/Rl/MlContract.cs`; `TacticalV3Candidates.cs`; `TacticalV3Capabilities.cs`; `TacticalV3Config.cs`; `TacticalV3Contract.cs`; `TacticalV3DuelEnv.cs`; `TacticalV3Env.cs`; `TacticalV3Observation.cs`; `TacticalV3Reward.cs`; `TacticalV3Schema.cs`; and `TrainingScenario.cs` in the same `Rl` directory.
- GymServer: `engine/HexWars.GymServer/Program.cs`; `ScenarioJson.cs`; `TacticalV3Wire.cs`.
- Tests: `engine/HexWars.Engine.Tests/ReplayFileTests.cs`; `TacticalV3CandidateTests.cs`; `TacticalV3ContractTests.cs`; `TacticalV3DuelEnvTests.cs`; `TacticalV3Fixtures.cs`; `TacticalV3GymServerTests.cs`; `TacticalV3ObservationTests.cs`; `TacticalV3RewardTests.cs`; `TacticalV3ScenarioTests.cs`; `TacticalV3SchemaTests.cs`.
- Final-fix evidence: `.superpowers/sdd/2026-08-10-tactical-v3-structured-environment-contract/final-fix-report.md`.
- Final acceptance report: `docs/superpowers/reports/2026-08-10-generalizable-structured-imitation-project-a.md`.

## Known limitations

- No model: Project B model implementation, ragged batching, overfit proof, and checkpoint adapter are not part of this gate.
- No fog: stage-one tactical-v3 requires `fog_of_war=false`; memory rows remain an explicit but unused extension surface.
- No design: unit-design actions and design-time learning are outside Project A.
- No DAgger: tactical-v3 explicitly rejects DAgger and evidence-session RPCs that remain tactical-v2-only.
- Unsealed experimental: hashes and exact schemas fail closed on drift, but this contract is not a production-sealed compatibility promise.
- Deferred Task 6–9 hardening: additional candidate-kind self-characterization, deeper nested-map mutation coverage, per-token-row golden hardening, and legacy golden-provenance documentation remain follow-up work.
- Legacy harness timing: fixed 10-second GymServer rejection waits were intermittently exceeded during two full-suite attempts even though every rotated case passed in isolation and the final exact suite was green. The hardening commit does not claim that legacy timing concern is closed.
