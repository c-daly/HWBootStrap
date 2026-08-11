# Project A: Generalizable Structured Imitation — Completion Report

Date: 2026-08-11

Status: complete as unsealed experimental evidence. Project B may consume the structured contract, subject to the limitations below.

## Commits

- `53e9dfb` feat: define tactical-v3 structured schema
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
- Final acceptance: `test: complete tactical-v3 contract gate` (this commit).

## Acceptance evidence

- GymServer build: `dotnet build engine/HexWars.GymServer/HexWars.GymServer.csproj --no-restore` — succeeded with 0 warnings and 0 errors.
- Complete tactical-v3 selector: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --filter FullyQualifiedName~TacticalV3 --no-restore` — 267 passed, 0 failed, 0 skipped.
- Full engine/GymServer regression: `dotnet test engine/HexWars.Engine.Tests/HexWars.Engine.Tests.csproj --no-restore` — 980 passed, 0 failed, 0 skipped.
- Live Unity/Coplay root: `C:\Users\cddal\HexWars\.worktrees\physical-checkpoint-audit`.
- Live Unity state: `playMode=false`, `hasCompilationErrors=false`.
- Live Unity compile result: `No compile errors`.
- Live Unity log result: empty 200-entry console query; no tactical-v3 exception was present.
- JSONL smoke: `spaces`, `reset` with seed 41, legal `step` selecting decision 0/candidate 0, then `close`.
- Smoke result: 10 reset candidates; selected kind `move`; successor decision 1 with 9 candidates; close exit 0.
- Smoke schema result: no top-level flat `obs` or `mask` in spaces, reset, or step; structured observation tables and decision-local candidates were present.

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
- Both seat views on a symmetric start were checked as coordinate reflections with self/opponent owner and deployment-zone swaps.
- Every legal candidate in an all-four-kind state was reconstructed from public row references, applied to a freshly recreated state with `GameEngine.Apply`, and compared field-by-field with its projected delta.
- All observation, relation, candidate, and projected-delta token references were range-checked after candidate projection and after every accepted environment step.
- Two same-seed trajectories matched complete public observations, candidates, selected command fields, rewards, terminal/truncation state, and exact replay text.
- Learned structured row DTOs are guarded against `Name`, `DisplayName`, `EngineId`, `UnitId`, and raw `PlayerId` features.
- A scenario with `max_cells=1` was rejected at startup with empty stdout, before reset payload publication.
- Legacy payload-shape contract tests remained green for tactical-v1, adaptive-v1, and tactical-v2 as part of the 980-test suite.

## Public interfaces

- Schema and semantics: `TacticalV3TableKind`, `TacticalV3TokenRef`, capability/action/relation enums and descriptors.
- Observation: `TacticalV3Observation`, its eight structured token/definition tables, `IObservationMemory`, `ISeatObservationSource`, and `TacticalV3SeatObservationSource`.
- Decisions: `TacticalV3Candidate`, `TacticalV3ProjectedDelta`, `TacticalV3DecisionFrame`, `ILegalCandidateSource`, `ICandidateProjector`, `IActionResolver`, and their tactical-v3 implementations.
- Reward and environments: `TacticalV3RewardBreakdown`, `IRewardContract`, `TacticalV3Reward`, `TacticalV3DuelEnv`, `TacticalV3Env`, and `TacticalV3View`.
- Contract/configuration: `TacticalV3CapacityProfile`, `TacticalV3RewardConfig`, `TacticalV3Config`, and `TacticalV3Contract` with independent contract, encoding, and capacity hashes.
- GymServer protocol: tactical-v3 `spaces`, `reset`, `step`, `duel_spaces`, `duel_reset`, `duel_step`, `duel_save`, and `close`; structured selections require exact decision/candidate identity.

## Project-A tracked file set

- Plan and scenario: `docs/superpowers/plans/2026-08-10-tactical-v3-structured-environment-contract.md`; `python/config/annihilation-structured-imitation-v1.json`.
- Engine: `engine/HexWars.Engine/ReplayFile.cs`; `engine/HexWars.Engine/Rl/MlContract.cs`; `TacticalV3Candidates.cs`; `TacticalV3Capabilities.cs`; `TacticalV3Config.cs`; `TacticalV3Contract.cs`; `TacticalV3DuelEnv.cs`; `TacticalV3Env.cs`; `TacticalV3Observation.cs`; `TacticalV3Reward.cs`; `TacticalV3Schema.cs`; and `TrainingScenario.cs` in the same `Rl` directory.
- GymServer: `engine/HexWars.GymServer/Program.cs`; `ScenarioJson.cs`; `TacticalV3Wire.cs`.
- Tests: `engine/HexWars.Engine.Tests/ReplayFileTests.cs`; `TacticalV3CandidateTests.cs`; `TacticalV3ContractTests.cs`; `TacticalV3DuelEnvTests.cs`; `TacticalV3Fixtures.cs`; `TacticalV3GymServerTests.cs`; `TacticalV3ObservationTests.cs`; `TacticalV3RewardTests.cs`; `TacticalV3ScenarioTests.cs`; `TacticalV3SchemaTests.cs`.
- Final acceptance report: `docs/superpowers/reports/2026-08-10-generalizable-structured-imitation-project-a.md`.

## Known limitations

- No model: Project B model implementation, ragged batching, overfit proof, and checkpoint adapter are not part of this gate.
- No fog: stage-one tactical-v3 requires `fog_of_war=false`; memory rows remain an explicit but unused extension surface.
- No design: unit-design actions and design-time learning are outside Project A.
- No DAgger: tactical-v3 explicitly rejects DAgger and evidence-session RPCs that remain tactical-v2-only.
- Unsealed experimental: hashes and exact schemas fail closed on drift, but this contract is not a production-sealed compatibility promise.
