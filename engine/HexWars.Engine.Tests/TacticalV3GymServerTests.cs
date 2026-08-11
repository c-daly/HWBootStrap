using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class TacticalV3GymServerTests
    {
        private static string CheckedInScenario => RepositoryPath(
            "python", "config", "annihilation-structured-imitation-v1.json");

        private static string MismatchedScenario => RepositoryPath(
            "python", "config", "annihilation-imitation-v1.json");

        public enum WrongReferenceFamily
        {
            UnitCell,
            UnknownTable,
            MemoryCell,
            AllocationOwner,
            AllocationDefinition,
            AllocationCapability,
            CandidateActor,
            CandidateTarget,
            CandidateTemplate,
            CandidateCell,
            ProjectionSourceCell,
            ProjectionDestinationCell,
            ProjectionTemplate,
            ProjectionTarget,
            NeighborSource,
            NeighborTarget,
            OccupiesSource,
            OccupiesTarget,
            HasCapabilitySource,
            HasCapabilityTarget,
        }

        public enum ContractMismatch
        {
            MatchDrift, CapacityDrift, UnrelatedValid, InvalidRole,
            EncodingDrift, EnvironmentRoleHashDrift,
        }

        [Test]
        public void Wire_SpacesEmitsOnlyStructuredContractAndExactHashes()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard(
                MlContract.TacticalV3Version, "wire-shape-scenario");
            scenario.Name = "scenario-name-must-not-leak";
            scenario.TacticalV3.Templates[0].Id = "engine-template-id-must-not-leak";
            scenario.TacticalV3.Templates[0].Name = "engine-template-name-must-not-leak";
            TacticalV3Config config = scenario.BuildTacticalV3();
            TacticalV3Contract contract = TacticalV3Contract.Create(config, MlEnvironmentKind.Duel);

            object wire = InvokeWire("Spaces", scenario, contract);
            JsonElement json = JsonSerializer.SerializeToElement(wire);

            AssertProperties(json,
                "scenario_id", "scenario_schema_version", "contract_version",
                "contract_hash", "encoding_hash", "capacity_hash", "environment_kind",
                "match", "encoding", "capacity");
            Assert.That(json.GetProperty("scenario_id").GetString(), Is.EqualTo(scenario.Id));
            Assert.That(json.GetProperty("scenario_schema_version").GetInt32(),
                Is.EqualTo(scenario.SchemaVersion));
            Assert.That(json.GetProperty("contract_version").GetString(), Is.EqualTo(contract.Version));
            Assert.That(json.GetProperty("contract_hash").GetString(), Is.EqualTo(contract.ContractHash));
            Assert.That(json.GetProperty("encoding_hash").GetString(), Is.EqualTo(contract.EncodingHash));
            Assert.That(json.GetProperty("capacity_hash").GetString(), Is.EqualTo(contract.CapacityHash));
            Assert.That(json.GetProperty("environment_kind").GetString(),
                Is.EqualTo(contract.EnvironmentKind));
            AssertIndependentSpacesSchema(json);

            Assert.That(Property(wire, "scenario_schema_version"), Is.TypeOf<int>());
            string raw = json.GetRawText();
            Assert.That(raw, Does.Not.Contain("obs_len"));
            Assert.That(raw, Does.Not.Contain("n_actions"));
            Assert.That(raw, Does.Not.Contain("\"obs\""));
            Assert.That(raw, Does.Not.Contain("\"mask\""));
            Assert.That(raw, Does.Not.Contain(scenario.Name));
            Assert.That(raw, Does.Not.Contain("engine-template-id-must-not-leak"));
            Assert.That(raw, Does.Not.Contain("engine-template-name-must-not-leak"));
        }

        [Test]
        public void Wire_ViewEmitsExactSnakeCaseSchemaNumericTypesAndDeterministicOrder()
        {
            TacticalV3View firstView = ViewWithMemory(seed: 41);
            TacticalV3View secondView = ViewWithMemory(seed: 41);

            object firstWire = InvokeWire("View", firstView);
            object secondWire = InvokeWire("View", secondView);
            JsonElement json = JsonSerializer.SerializeToElement(firstWire);
            string raw = JsonSerializer.Serialize(firstWire);

            Assert.That(raw, Is.EqualTo(JsonSerializer.Serialize(secondWire)));
            AssertProperties(json,
                "decision_id", "seat", "observation", "candidates", "reward", "winner",
                "terminated", "truncated", "start_profile", "reference_seat");
            Assert.That(Property(firstWire, "decision_id"), Is.TypeOf<long>());
            Assert.That(Property(firstWire, "seat"), Is.TypeOf<int>());
            Assert.That(Property(firstWire, "winner"), Is.TypeOf<int>());
            Assert.That(json.GetProperty("decision_id").GetInt64(),
                Is.EqualTo(firstView.Decision.DecisionId));
            Assert.That(json.GetProperty("seat").GetInt32(), Is.EqualTo((int)firstView.Seat));
            Assert.That(json.GetProperty("start_profile").GetString(), Is.EqualTo("standard-3v3"));
            Assert.That(json.GetProperty("reference_seat").GetInt32(), Is.EqualTo(0));

            JsonElement observation = json.GetProperty("observation");
            AssertProperties(observation,
                "cells", "units", "templates", "capability_definitions",
                "capability_allocations", "rules", "memory", "relations");
            AssertEveryRow(observation.GetProperty("cells"),
                "q", "r", "terrain", "elevation", "self_deployment_zone",
                "opponent_deployment_zone", "controller", "is_boundary",
                "currently_visible", "previously_observed");
            AssertEveryRow(observation.GetProperty("units"),
                "owner", "current_hp", "max_hp", "cell", "elevation", "moved", "attacked",
                "horizontal_movement_spent", "vertical_movement_spent", "point_cost",
                "deploy_cost", "currently_visible");
            AssertEveryRow(observation.GetProperty("templates"),
                "owner", "point_cost", "deploy_cost", "is_fixed", "is_deployable");
            AssertEveryRow(observation.GetProperty("capability_definitions"), "kind");
            AssertEveryRow(observation.GetProperty("capability_allocations"),
                "owner", "definition", "capability", "purchased_level", "effective_value");
            AssertEveryRow(observation.GetProperty("rules"),
                "kind", "int_value", "float_value", "bool_value");
            AssertEveryRow(observation.GetProperty("memory"),
                "cell", "last_seen_round", "observation_age", "last_known_current_hp",
                "currently_visible");
            AssertEveryRow(observation.GetProperty("relations"),
                "kind", "source", "target", "int_feature", "float_feature", "bool_feature");

            JsonElement candidates = json.GetProperty("candidates");
            AssertEveryRow(candidates,
                "candidate_id", "decision_id", "kind", "actor", "target", "template", "cell",
                "projection");
            foreach (JsonElement candidate in candidates.EnumerateArray())
                AssertProperties(candidate.GetProperty("projection"),
                    "source_cell", "destination_cell", "template", "target",
                    "horizontal_movement_spent", "vertical_movement_spent", "target_hp_delta",
                    "damage", "is_lethal", "bounty_delta", "points_delta", "round_delta",
                    "is_terminal");

            JsonElement reward = json.GetProperty("reward");
            AssertProperties(reward,
                "terminal_outcome", "known_health_adjusted_material_progress",
                "public_resource_progress", "time_pressure", "total", "finalized");
            object rewardWire = Property(firstWire, "reward");
            Assert.That(Property(rewardWire, "terminal_outcome"), Is.TypeOf<float>());
            Assert.That(Property(rewardWire, "known_health_adjusted_material_progress"),
                Is.TypeOf<float>());
            Assert.That(Property(rewardWire, "public_resource_progress"), Is.TypeOf<float>());
            Assert.That(Property(rewardWire, "time_pressure"), Is.TypeOf<float>());
            Assert.That(Property(rewardWire, "total"), Is.TypeOf<float>());

            object observationWire = Property(firstWire, "observation");
            object ruleWire = First(Property(observationWire, "rules"));
            object relationWire = First(Property(observationWire, "relations"));
            Assert.That(Property(ruleWire, "int_value"), Is.TypeOf<int>());
            Assert.That(Property(ruleWire, "float_value"), Is.TypeOf<float>());
            Assert.That(Property(relationWire, "int_feature"), Is.TypeOf<int>());
            Assert.That(Property(relationWire, "float_feature"), Is.TypeOf<float>());

            string[] definitionOrder = observation.GetProperty("capability_definitions")
                .EnumerateArray().Select(row => row.GetProperty("kind").GetString()!).ToArray();
            Assert.That(definitionOrder, Is.EqualTo(new[]
            {
                "health", "damage", "defense", "movement", "vertical_movement",
                "range", "range_arc", "vision", "vision_arc",
            }));
            int[] candidateOrder = candidates.EnumerateArray()
                .Select(row => row.GetProperty("candidate_id").GetInt32()).ToArray();
            Assert.That(candidateOrder, Is.EqualTo(Enumerable.Range(0, candidateOrder.Length)));
            AssertReferencesAreInRange(json);

            Assert.That(raw, Does.Not.Contain("\"obs\""));
            Assert.That(raw, Does.Not.Contain("\"mask\""));
            Assert.That(raw, Does.Not.Contain("obs_len"));
            Assert.That(raw, Does.Not.Contain("n_actions"));
            Assert.That(raw, Does.Not.Contain("unit_id"));
            Assert.That(raw, Does.Not.Contain("entity_id"));
            Assert.That(raw, Does.Not.Contain("template_id"));
            Assert.That(raw, Does.Not.Contain("\"name\""));
        }

        [TestCaseSource(nameof(WrongReferenceFamilies))]
        public void Wire_ViewRejectsInRangeWrongTableForEverySemanticReference(
            WrongReferenceFamily family)
        {
            TacticalV3View view = ViewWithMemory(seed: 42);
            ApplyWrongReference(view, family);

            TargetInvocationException exception = Assert.Throws<TargetInvocationException>(
                () => InvokeWire("View", view))!;
            Assert.That(exception.InnerException, Is.TypeOf<InvalidOperationException>());
            Assert.That(exception.InnerException!.Message, Does.Contain("semantic"));
        }

        [TestCaseSource(nameof(ContractMismatches))]
        public void Wire_SpacesRejectsStaleUnrelatedOrMalformedContractEvidence(
            ContractMismatch mismatch)
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard(
                MlContract.TacticalV3Version, "contract-evidence");
            TacticalV3Contract contract = TacticalV3Contract.Create(
                scenario.BuildTacticalV3(), MlEnvironmentKind.Duel);

            switch (mismatch)
            {
                case ContractMismatch.MatchDrift:
                    scenario.Rules.StartingPoints++;
                    break;
                case ContractMismatch.CapacityDrift:
                    scenario.TacticalV3.Capacity.MaxCells++;
                    break;
                case ContractMismatch.UnrelatedValid:
                    contract = TacticalV3Contract.Create(
                        TacticalV3Fixtures.Config(width: 12, height: 9), MlEnvironmentKind.Duel);
                    break;
                case ContractMismatch.InvalidRole:
                    contract = CloneContract(contract, environmentKind: "spectator");
                    break;
                case ContractMismatch.EncodingDrift:
                    contract = CloneContract(contract, encodingHash: new string('0', 64));
                    break;
                case ContractMismatch.EnvironmentRoleHashDrift:
                    contract = CloneContract(contract, environmentKind: "tactical");
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(mismatch));
            }

            TargetInvocationException exception = Assert.Throws<TargetInvocationException>(
                () => InvokeWire("Spaces", scenario, contract))!;
            Assert.That(exception.InnerException, Is.TypeOf<InvalidOperationException>());
            Assert.That(exception.InnerException!.Message, Does.Contain("contract evidence"));
        }

        [Test]
        public void Wire_ViewAllowsTerminalFrameWithNoCandidates()
        {
            TacticalV3View view = ViewWithMemory(seed: 44);
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.Candidates),
                Array.AsReadOnly(Array.Empty<TacticalV3Candidate>()));
            SetAutoProperty(view, nameof(TacticalV3View.Winner), 1);
            SetAutoProperty(view, nameof(TacticalV3View.Terminated), true);

            JsonElement json = JsonSerializer.SerializeToElement(InvokeWire("View", view));

            Assert.That(json.GetProperty("candidates").GetArrayLength(), Is.Zero);
            Assert.That(json.GetProperty("winner").GetInt32(), Is.EqualTo(1));
            Assert.That(json.GetProperty("terminated").GetBoolean(), Is.True);
        }

        [Test]
        public void Wire_ViewPreservesPlayer1ControllersWinnerAndEveryNumericLeafType()
        {
            TacticalV3View view = ViewWithMemory(
                seed: 46, learnerSeat: PlayerId.Player1, referenceSeat: PlayerId.Player1);
            TacticalV3Observation original = view.Decision.Observation;
            TacticalV3CellToken[] cells = original.Cells.ToArray();
            cells[0] = CellWithController(cells[0], TacticalV3RelativeOwner.Self);
            cells[1] = CellWithController(cells[1], TacticalV3RelativeOwner.Opponent);
            TacticalV3RelationToken[] relations = original.Relations.ToArray();
            TacticalV3RelationToken relation = relations[0];
            relations[0] = new TacticalV3RelationToken(
                relation.Kind, relation.Source, relation.Target,
                intFeature: 7, floatFeature: 1.25f, boolFeature: true);
            ReplaceObservation(view, cells: cells, relations: relations);

            long decisionId = (long)int.MaxValue + 41L;
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.DecisionId), decisionId);
            foreach (TacticalV3Candidate candidate in view.Decision.Candidates)
                SetAutoProperty(candidate, nameof(TacticalV3Candidate.DecisionId), decisionId);
            SetAutoProperty(view, nameof(TacticalV3View.Reward), new TacticalV3RewardBreakdown(
                0.125f, -0.25f, 0.375f, -0.5f, 0.625f, finalized: true));
            SetAutoProperty(view, nameof(TacticalV3View.Winner), 1);

            object wire = InvokeWire("View", view);
            JsonElement json = JsonSerializer.SerializeToElement(wire);

            Assert.That(json.GetProperty("decision_id").GetInt64(), Is.EqualTo(decisionId));
            Assert.That(json.GetProperty("decision_id").TryGetInt32(out _), Is.False);
            Assert.That(json.GetProperty("seat").GetInt32(), Is.EqualTo(1));
            Assert.That(json.GetProperty("reference_seat").GetInt32(), Is.EqualTo(1));
            Assert.That(json.GetProperty("winner").GetInt32(), Is.EqualTo(1));
            JsonElement jsonCells = json.GetProperty("observation").GetProperty("cells");
            Assert.That(jsonCells[0].GetProperty("controller").GetString(), Is.EqualTo("self"));
            Assert.That(jsonCells[1].GetProperty("controller").GetString(), Is.EqualTo("opponent"));
            Assert.That(json.GetProperty("reward").GetProperty("terminal_outcome")
                .TryGetInt32(out _), Is.False);
            AssertNumericWireTypes(wire);
            AssertNumericJsonTypes(json);
        }

        [Test]
        public void Wire_ViewRejectsTokenReferenceOutsideTargetTable()
        {
            TacticalV3View view = ViewWithMemory(seed: 43);
            TacticalV3Observation original = view.Decision.Observation;
            TacticalV3UnitToken source = original.Units[0];
            TacticalV3UnitToken invalid = new TacticalV3UnitToken(
                source.Owner, source.CurrentHp, source.MaxHp,
                new TacticalV3TokenRef(TacticalV3TableKind.Cells, original.Cells.Count),
                source.Elevation, source.Moved, source.Attacked, source.HorizontalMovementSpent,
                source.VerticalMovementSpent, source.PointCost, source.DeployCost,
                source.CurrentlyVisible);
            TacticalV3UnitToken[] units = original.Units.ToArray();
            units[0] = invalid;
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.Observation),
                new TacticalV3Observation(
                    original.Cells, units, original.Templates, original.CapabilityDefinitions,
                    original.CapabilityAllocations, original.Rules, original.Memory,
                    original.Relations));

            TargetInvocationException exception = Assert.Throws<TargetInvocationException>(
                () => InvokeWire("View", view))!;
            Assert.That(exception.InnerException, Is.TypeOf<InvalidOperationException>());
            Assert.That(exception.InnerException!.Message, Does.Contain("token reference"));
        }

        [Test]
        public void Wire_SpacesRevalidatesScenarioCapacityAtBoundary()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard(MlContract.TacticalV3Version);
            TacticalV3Contract contract = TacticalV3Contract.Create(
                scenario.BuildTacticalV3(), MlEnvironmentKind.Tactical);
            scenario.TacticalV3.Capacity.MaxCells = 1;

            TargetInvocationException exception = Assert.Throws<TargetInvocationException>(
                () => InvokeWire("Spaces", scenario, contract))!;
            Assert.That(exception.InnerException, Is.TypeOf<ArgumentException>());
            Assert.That(exception.InnerException!.Message, Does.Contain("capacity"));
        }

        [Test]
        public void Wire_ViewRejectsEveryConfiguredTableCapacityOverflow()
        {
            TacticalV3View view = ViewWithMemory(seed: 47);
            TacticalV3Observation original = view.Decision.Observation;
            var memory = new[]
            {
                original.Memory[0],
                new TacticalV3MemoryToken(
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, 1),
                    lastSeenRound: 1, observationAge: 0, lastKnownCurrentHp: 2,
                    currentlyVisible: true),
            };
            ReplaceObservation(view, memory: memory);
            original = view.Decision.Observation;
            var cases = new[]
            {
                ("cells", TacticalV3Fixtures.ExperimentalCapacity(maxCells: original.Cells.Count - 1)),
                ("units", TacticalV3Fixtures.ExperimentalCapacity(maxUnits: original.Units.Count - 1)),
                ("templates", TacticalV3Fixtures.ExperimentalCapacity(maxTemplates: original.Templates.Count - 1)),
                ("capability_definitions", TacticalV3Fixtures.ExperimentalCapacity(
                    maxCapabilityDefinitions: original.CapabilityDefinitions.Count - 1)),
                ("capability_allocations", TacticalV3Fixtures.ExperimentalCapacity(
                    maxCapabilityAllocations: original.CapabilityAllocations.Count - 1)),
                ("rules", TacticalV3Fixtures.ExperimentalCapacity(maxRules: original.Rules.Count - 1)),
                ("memory", TacticalV3Fixtures.ExperimentalCapacity(maxMemoryRecords: original.Memory.Count - 1)),
                ("relations", TacticalV3Fixtures.ExperimentalCapacity(maxRelations: original.Relations.Count - 1)),
                ("candidates", TacticalV3Fixtures.ExperimentalCapacity(
                    maxCandidates: view.Decision.Candidates.Count - 1)),
            };

            foreach ((string table, TacticalV3CapacityProfile capacity) in cases)
            {
                TargetInvocationException exception = Assert.Throws<TargetInvocationException>(
                    () => InvokeWire("View", view, capacity), table)!;
                Assert.That(exception.InnerException, Is.TypeOf<InvalidOperationException>(), table);
                Assert.That(exception.InnerException!.Message,
                    Does.Contain("capacity").And.Contain(table), table);
            }
        }

        [Test]
        public void Process_TacticalV3RejectsOmittedScenarioBeforeReadingCommands()
        {
            string error = TacticalV3ServerProcess.RejectStartup(
                "--environment", MlContract.TacticalV3Version);

            Assert.That(error, Does.Contain("tactical-v3 requires --scenario-file"));
        }

        [Test]
        public void Process_TacticalV3RejectsMismatchedScenarioBeforeReadingCommands()
        {
            string error = TacticalV3ServerProcess.RejectStartup(
                "--environment", MlContract.TacticalV3Version,
                "--scenario-file", MismatchedScenario);

            Assert.That(error, Does.Contain("scenario environment does not match --environment"));
        }

        [Test]
        public void Process_TinyCapacityRejectsBeforeResetPayloadPublication()
        {
            string path = Path.Combine(
                TestContext.CurrentContext.WorkDirectory,
                "tactical-v3-tiny-capacity-" + Guid.NewGuid().ToString("N") + ".json");
            try
            {
                string source = File.ReadAllText(CheckedInScenario);
                string capacityMarker = (char)34 + "max_cells" + (char)34 + ": 512";
                string tiny = source.Replace(
                    capacityMarker, (char)34 + "max_cells" + (char)34 + ": 1");
                Assert.That(tiny, Is.Not.EqualTo(source));
                File.WriteAllText(path, tiny, new UTF8Encoding(false));

                string error = TacticalV3ServerProcess.RejectStartup(
                    "--environment", MlContract.TacticalV3Version,
                    "--scenario-file", path);

                Assert.That(error,
                    Does.Contain("tactical-v3 max cells capacity is smaller than the board"));
            }
            finally
            {
                if (File.Exists(path)) File.Delete(path);
            }
        }

        [Test]
        public void Process_SpacesReportsStructuredSchemasWithoutFlatGeometry()
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);

            JsonElement spaces = server.Request("{\"cmd\":\"spaces\"}");

            AssertProperties(spaces,
                "scenario_id", "scenario_schema_version", "contract_version",
                "contract_hash", "encoding_hash", "capacity_hash", "environment_kind",
                "match", "encoding", "capacity");
            Assert.That(spaces.GetProperty("contract_version").GetString(),
                Is.EqualTo(MlContract.TacticalV3Version));
            Assert.That(spaces.GetProperty("environment_kind").GetString(), Is.EqualTo("tactical"));
            Assert.That(spaces.GetProperty("encoding").GetProperty("tables").ValueKind,
                Is.EqualTo(JsonValueKind.Object));
            Assert.That(spaces.GetProperty("encoding").GetProperty("candidate_schema").ValueKind,
                Is.EqualTo(JsonValueKind.Object));
            Assert.That(spaces.TryGetProperty("obs_len", out _), Is.False);
            Assert.That(spaces.TryGetProperty("n_actions", out _), Is.False);
            Assert.That(spaces.TryGetProperty("channels", out _), Is.False);
            Assert.That(spaces.TryGetProperty("board_h", out _), Is.False);
            Assert.That(spaces.TryGetProperty("board_w", out _), Is.False);
        }

        [Test]
        public void Process_CheckedInProfiledScenarioPinsExactHashesAndProfileSchemas()
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement spaces = server.Request(JsonSerializer.Serialize(new { cmd = "spaces" }));

            Assert.Multiple(() =>
            {
                Assert.That(spaces.GetProperty("contract_hash").GetString(),
                    Is.EqualTo("0ae48260cde97bce9ed75975874676a262588b3ed17963cdb41d09d09d3088ce"));
                Assert.That(spaces.GetProperty("encoding_hash").GetString(),
                    Is.EqualTo("e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6"));
                Assert.That(spaces.GetProperty("capacity_hash").GetString(),
                    Is.EqualTo("7aea1db4f008dc192e83811b2c13abd8ce2304d2a6a209f37f9847be5f367364"));
                Assert.That(spaces.GetProperty("match").GetProperty("board")
                    .GetProperty("width").GetInt32(), Is.EqualTo(13));
                Assert.That(spaces.GetProperty("match").GetProperty("board")
                    .GetProperty("height").GetInt32(), Is.EqualTo(9));
            });
            AssertPinnedProfileSchemas(spaces.GetProperty("match"));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void Process_ResetAndStepViewsCarryDecisionAndCandidateIdentity(bool duel)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = duel
                ? server.Request(JsonSerializer.Serialize(new
                {
                    cmd = "duel_reset", seed = 41, p0 = "external", p1 = "external",
                    learner = 0, start_profile = "standard-3v3", reference_seat = 0,
                }))
                : server.Request("{\"cmd\":\"reset\",\"seed\":41}");
            AssertViewIdentities(reset);
            long decisionId = reset.GetProperty("decision_id").GetInt64();
            int candidateId = reset.GetProperty("candidates")[0]
                .GetProperty("candidate_id").GetInt32();

            JsonElement next = server.Request(JsonSerializer.Serialize(new
            {
                cmd = duel ? "duel_step" : "step",
                decision_id = decisionId,
                candidate_id = candidateId,
            }));

            AssertViewIdentities(next);
        }

        [TestCase("step", "decision_id")]
        [TestCase("step", "candidate_id")]
        [TestCase("step", "unknown")]
        [TestCase("duel_step", "decision_id")]
        [TestCase("duel_step", "candidate_id")]
        [TestCase("duel_step", "unknown")]
        public void Process_StructuredSelectionsRequireExactIdentityFields(
            string command, string mutation)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = ResetForSelection(server, command);
            long decisionId = reset.GetProperty("decision_id").GetInt64();
            int candidateId = reset.GetProperty("candidates")[0]
                .GetProperty("candidate_id").GetInt32();
            string request = mutation switch
            {
                "decision_id" => JsonSerializer.Serialize(new { cmd = command, candidate_id = candidateId }),
                "candidate_id" => JsonSerializer.Serialize(new { cmd = command, decision_id = decisionId }),
                "unknown" => JsonSerializer.Serialize(new
                {
                    cmd = command, decision_id = decisionId, candidate_id = candidateId, extra = 1,
                }),
                _ => throw new ArgumentOutOfRangeException(nameof(mutation)),
            };

            string error = server.Reject(request);

            Assert.That(error, Does.Contain("unknown or missing fields"));
        }

        [TestCase("step", "stale")]
        [TestCase("step", "negative")]
        [TestCase("step", "out_of_range")]
        [TestCase("duel_step", "stale")]
        [TestCase("duel_step", "negative")]
        [TestCase("duel_step", "out_of_range")]
        public void Process_InvalidStructuredSelectionEmitsNamedErrorWithoutSuccessorView(
            string command, string mutation)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = ResetForSelection(server, command);
            long decisionId = reset.GetProperty("decision_id").GetInt64();
            int candidateId = reset.GetProperty("candidates")[0]
                .GetProperty("candidate_id").GetInt32();
            if (mutation == "stale") decisionId--;
            else if (mutation == "negative") candidateId = -1;
            else if (mutation == "out_of_range") candidateId = int.MaxValue;
            else throw new ArgumentOutOfRangeException(nameof(mutation));

            string error = server.Reject(JsonSerializer.Serialize(new
            {
                cmd = command, decision_id = decisionId, candidate_id = candidateId,
            }));

            Assert.That(error, mutation == "stale"
                ? Does.Contain("decision id is stale")
                : Does.Contain("candidate id is out of range"));
        }

        [TestCase("step")]
        [TestCase("duel_step")]
        public void Process_StructuredSelectionParsesDecisionIdentityAsInt64(string command)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = ResetForSelection(server, command);
            int candidateId = reset.GetProperty("candidates")[0]
                .GetProperty("candidate_id").GetInt32();

            string error = server.Reject(JsonSerializer.Serialize(new
            {
                cmd = command, decision_id = (long)int.MaxValue + 1L,
                candidate_id = candidateId,
            }));

            Assert.That(error, Does.Contain("decision id is stale"));
        }

        [TestCase("step")]
        [TestCase("duel_step")]
        public void Process_StructuredSelectionRejectsCandidateIdentityOutsideInt32(string command)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = ResetForSelection(server, command);
            long decisionId = reset.GetProperty("decision_id").GetInt64();

            string error = server.Reject(JsonSerializer.Serialize(new
            {
                cmd = command,
                decision_id = decisionId,
                candidate_id = (long)int.MaxValue + 1L,
            }));

            Assert.That(error, Does.Contain("Int32"));
        }

        [Test]
        public void Process_IdenticalSeedsAndCommandsEmitByteIdenticalJson()
        {
            using var first = TacticalV3ServerProcess.Start(CheckedInScenario);
            using var second = TacticalV3ServerProcess.Start(CheckedInScenario);
            AssertSameReply(first, second, "{\"cmd\":\"spaces\"}");
            string reset = AssertSameReply(
                first, second, "{\"cmd\":\"reset\",\"seed\":123}");
            int endTurn = CandidateId(reset, "end_turn");
            AssertSameReply(first, second, JsonSerializer.Serialize(new
            {
                cmd = "step", decision_id = 0L, candidate_id = endTurn,
            }));
            AssertSameReply(first, second, "{\"cmd\":\"duel_spaces\"}");
            string duelReset = AssertSameReply(first, second, JsonSerializer.Serialize(new
            {
                cmd = "duel_reset", seed = 123, p0 = "external", p1 = "random",
                learner = 0, start_profile = "standard-3v3", reference_seat = 0,
            }));
            int duelEndTurn = CandidateId(duelReset, "end_turn");
            AssertSameReply(first, second, JsonSerializer.Serialize(new
            {
                cmd = "duel_step", decision_id = 0L, candidate_id = duelEndTurn,
            }));
        }

        [Test]
        public void Process_SelectedCommandReconstructsSavedReplayFinalState()
        {
            string replayPath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "tactical-v3-process-" + Guid.NewGuid().ToString("N") + ".replay");
            try
            {
                using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
                JsonElement reset = server.Request(JsonSerializer.Serialize(new
                {
                    cmd = "duel_reset", seed = 71, p0 = "external", p1 = "external",
                    learner = 0, start_profile = "standard-3v3", reference_seat = 0,
                }));
                JsonElement selected = reset.GetProperty("candidates").EnumerateArray()
                    .First(candidate => candidate.GetProperty("kind").GetString() == "move");
                long decisionId = reset.GetProperty("decision_id").GetInt64();
                int candidateId = selected.GetProperty("candidate_id").GetInt32();
                JsonElement successor = server.Request(JsonSerializer.Serialize(new
                {
                    cmd = "duel_step", decision_id = decisionId, candidate_id = candidateId,
                }));
                server.Request(JsonSerializer.Serialize(new { cmd = "duel_save", path = replayPath }));

                TacticalV3Config independentConfig = LoadStrictScenario().BuildTacticalV3();
                var independent = new TacticalV3DuelEnv(independentConfig);
                TacticalV3View independentReset = independent.Reset(
                    71, null, null, "standard-3v3", PlayerId.Player0, PlayerId.Player0);
                TacticalV3Candidate independentCandidate = independentReset.Decision.Candidates
                    .Single(candidate => candidate.CandidateId == candidateId);
                Assert.That(independentCandidate.Kind, Is.EqualTo(TacticalV3CandidateKind.Move));
                TacticalV3View independentSuccessor = independent.Step(
                    independentReset.Decision.DecisionId, independentCandidate.CandidateId);

                ReplayData data = ReplayFile.Read(File.ReadAllText(replayPath));
                var replay = new Replay(data.Start, data.Commands);
                var command = (MoveUnit)data.Commands.Single();
                Unit finalActor = replay.Final.Player(command.Issuer).UnitsOnBoard
                    .Single(unit => unit.Id == command.UnitId);
                JsonElement destinationReference = selected.GetProperty("projection")
                    .GetProperty("destination_cell");
                JsonElement destination = reset.GetProperty("observation").GetProperty("cells")
                    [destinationReference.GetProperty("row").GetInt32()];

                Assert.Multiple(() =>
                {
                    Assert.That(command.Dest.Q, Is.EqualTo(destination.GetProperty("q").GetInt32()));
                    Assert.That(command.Dest.R, Is.EqualTo(destination.GetProperty("r").GetInt32()));
                    Assert.That(finalActor.Cell, Is.EqualTo(command.Dest));
                    Assert.That(data.Commands, Has.Count.EqualTo(1));
                    Assert.That(successor.GetProperty("decision_id").GetInt64(), Is.EqualTo(1));
                    Assert.That(independentSuccessor.Decision.DecisionId,
                        Is.EqualTo(successor.GetProperty("decision_id").GetInt64()));
                });
                AssertAuthoritativeGameStatesEqual(independent.State, replay.Final);
            }
            finally
            {
                if (File.Exists(replayPath)) File.Delete(replayPath);
            }
        }

        [TestCase("standard-3v3", 0, 3, 3)]
        [TestCase("standard-3v3", 1, 3, 3)]
        [TestCase("conversion-3v1-near", 0, 3, 1)]
        [TestCase("conversion-3v1-near", 1, 1, 3)]
        public void Process_DuelProfilesWorkFromEitherLearnerSeat(
            string profile, int learner, int expectedSelf, int expectedOpponent)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement reset = server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_reset", seed = 83, p0 = "external", p1 = "external",
                learner, start_profile = profile, reference_seat = learner,
            }));
            JsonElement units = reset.GetProperty("observation").GetProperty("units");
            int self = units.EnumerateArray().Count(unit =>
                unit.GetProperty("owner").GetString() == "self");
            int opponent = units.EnumerateArray().Count(unit =>
                unit.GetProperty("owner").GetString() == "opponent");

            Assert.Multiple(() =>
            {
                Assert.That(reset.GetProperty("start_profile").GetString(), Is.EqualTo(profile));
                Assert.That(reset.GetProperty("reference_seat").GetInt32(), Is.EqualTo(learner));
                Assert.That(self, Is.EqualTo(expectedSelf));
                Assert.That(opponent, Is.EqualTo(expectedOpponent));
            });
            long decisionId = reset.GetProperty("decision_id").GetInt64();
            int candidateId = reset.GetProperty("candidates")[0]
                .GetProperty("candidate_id").GetInt32();
            AssertViewIdentities(server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_step", decision_id = decisionId, candidate_id = candidateId,
            })));
        }

        [TestCase("tactical-v1",
            "c14020cd08e2ea02596939a55fb6235dfd6822d4de2cfc3445ecc14e78c1a0aa")]
        [TestCase("adaptive-v1",
            "f307cfec91605431175c36c1ed8e6a90b3442cacb8b9315720b01bad8a01e405")]
        [TestCase("tactical-v2",
            "09ab67ceba29b59208a93d6985ab90a1a7f93872ab0c82c058868ae3ed2ce01f")]
        public void Process_LegacyEnvironmentPayloadShapesRemainCompatible(
            string environment, string expectedShapeHash)
        {
            Assert.That(LegacySequenceShapeHash(environment),
                Is.EqualTo(expectedShapeHash), environment);
        }

        [TestCase("duel_dagger_configure", "duel DAgger is supported only for tactical-v2")]
        [TestCase("duel_dagger_drain", "duel DAgger is supported only for tactical-v2")]
        [TestCase("duel_evidence_begin", "evidence sessions are supported only for tactical-v2")]
        [TestCase("duel_evidence_game_close", "evidence sessions are supported only for tactical-v2")]
        [TestCase("duel_evidence_end", "evidence sessions are supported only for tactical-v2")]
        public void Process_TacticalV3ExplicitlyRejectsDaggerAndEvidenceRpcs(
            string command, string expectedError)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            string request = command == "duel_dagger_configure"
                ? JsonSerializer.Serialize(new
                {
                    cmd = command, enabled = true, depth = 4,
                    expansion_budget = 512, use_heuristic = true,
                })
                : JsonSerializer.Serialize(new { cmd = command });

            string error = server.Reject(request);

            Assert.That(error, Does.Contain(expectedError));
        }

        [TestCase("spaces")]
        [TestCase("reset")]
        [TestCase("step")]
        [TestCase("duel_spaces")]
        [TestCase("duel_reset")]
        [TestCase("duel_step")]
        [TestCase("duel_save")]
        [TestCase("close")]
        public void Process_TacticalV3EveryCommandRejectsUnknownFields(string command)
        {
            string savePath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "strict-command-" + Guid.NewGuid().ToString("N") + ".replay");
            try
            {
                using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
                string valid = TacticalV3CommandRequest(server, command, savePath);
                string malformed = valid.Substring(0, valid.Length - 1) +
                    ",\"unexpected\":true}";

                string error = server.RejectCommand(malformed);

                Assert.That(error, Does.Contain(
                    $"tactical-v3 {command} has unknown or missing fields"));
            }
            finally
            {
                if (File.Exists(savePath)) File.Delete(savePath);
            }
        }

        [TestCase("spaces")]
        [TestCase("reset")]
        [TestCase("step")]
        [TestCase("duel_spaces")]
        [TestCase("duel_reset")]
        [TestCase("duel_step")]
        [TestCase("duel_save")]
        [TestCase("close")]
        public void Process_TacticalV3EveryCommandRejectsDuplicateRootProperties(string command)
        {
            string savePath = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "duplicate-command-" + Guid.NewGuid().ToString("N") + ".replay");
            try
            {
                using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
                string valid = TacticalV3CommandRequest(server, command, savePath);
                string malformed = "{\"cmd\":\"" + command + "\"," + valid.Substring(1);

                string error = server.RejectCommand(malformed);

                Assert.That(error, Does.Contain("duplicate"));
            }
            finally
            {
                if (File.Exists(savePath)) File.Delete(savePath);
            }
        }

        [TestCase("spaces", "cmd", "null", "a non-empty string")]
        [TestCase("spaces", "cmd", "\"\"", "a non-empty string")]
        [TestCase("reset", "seed", "\"1\"", "an Int32 number")]
        [TestCase("reset", "seed", "2147483648", "an Int32 number")]
        [TestCase("step", "decision_id", "9223372036854775808", "an Int64 number")]
        [TestCase("step", "candidate_id", "false", "an Int32 number")]
        [TestCase("step", "candidate_id", "2147483648", "an Int32 number")]
        [TestCase("duel_spaces", "cmd", "[]", "a non-empty string")]
        [TestCase("duel_reset", "seed", "{}", "an Int32 number")]
        [TestCase("duel_reset", "p0", "null", "a non-empty string")]
        [TestCase("duel_reset", "p0", "\"\"", "a non-empty string")]
        [TestCase("duel_reset", "p1", "null", "a non-empty string")]
        [TestCase("duel_reset", "learner", "1.5", "an Int32 number")]
        [TestCase("duel_reset", "start_profile", "null", "a non-empty string")]
        [TestCase("duel_reset", "start_profile", "\"\"", "a non-empty string")]
        [TestCase("duel_reset", "start_profile", "[]", "a non-empty string")]
        [TestCase("duel_reset", "reference_seat", "\"0\"", "an Int32 number")]
        [TestCase("duel_step", "decision_id", "{}", "an Int64 number")]
        [TestCase("duel_step", "candidate_id", "null", "an Int32 number")]
        [TestCase("duel_save", "path", "null", "a non-empty string")]
        [TestCase("duel_save", "path", "{}", "a non-empty string")]
        [TestCase("duel_save", "path", "\"\"", "a non-empty string")]
        [TestCase("close", "cmd", "false", "a non-empty string")]
        public void Process_TacticalV3RejectsNullAndWrongKindFieldValues(
            string command, string field, string invalidJson, string expectedType)
        {
            string workDirectory = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "invalid-field-kind-" + Guid.NewGuid().ToString("N"));
            Directory.CreateDirectory(workDirectory);
            try
            {
                using var server = TacticalV3ServerProcess.StartInWorkingDirectory(
                    CheckedInScenario, workDirectory);
                string savePath = Path.Combine(workDirectory, "explicit.replay");
                string valid = TacticalV3CommandRequest(server, command, savePath);
                string malformed = ReplaceRootFieldValue(valid, field, invalidJson);

                string error = server.RejectCommand(malformed);

                string expected = field == "cmd"
                    ? $"tactical-v3 field 'cmd' must be {expectedType}"
                    : $"tactical-v3 {command} field '{field}' must be {expectedType}";
                Assert.Multiple(() =>
                {
                    Assert.That(error, Does.Contain(expected));
                    Assert.That(Directory.EnumerateFiles(workDirectory, "*.replay",
                        SearchOption.AllDirectories), Is.Empty,
                        "invalid field values must not create a replay");
                });
            }
            finally
            {
                if (Directory.Exists(workDirectory))
                    Directory.Delete(workDirectory, recursive: true);
            }
        }

        [Test]
        public void Process_TacticalV3RejectsUnknownCommandWithoutResponse()
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);

            string error = server.RejectCommand("{\"cmd\":\"not_a_command\"}");

            Assert.That(error, Does.Contain("tactical-v3 unknown command 'not_a_command'"));
        }

        [TestCase(false, false, true)]
        [TestCase(true, true, true)]
        [TestCase(true, false, false)]
        [TestCase(false, true, false)]
        public void Process_DuelResetRequiresProfileAndReferenceSeatTogether(
            bool includeProfile, bool includeReferenceSeat, bool accepted)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            var request = new Dictionary<string, object?>
            {
                ["cmd"] = "duel_reset", ["seed"] = 47,
                ["p0"] = "external", ["p1"] = "external", ["learner"] = 0,
            };
            if (includeProfile) request["start_profile"] = "standard-3v3";
            if (includeReferenceSeat) request["reference_seat"] = 0;
            string json = JsonSerializer.Serialize(request);

            if (accepted)
            {
                AssertViewIdentities(server.Request(json));
                return;
            }

            string error = server.RejectCommand(json);
            Assert.That(error, Does.Contain("reference_seat"));
        }

        [TestCase("p0")]
        [TestCase("p1")]
        public void Process_DuelResetRejectsUnknownControllerSpecPerSeat(string seat)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            var request = new Dictionary<string, object?>
            {
                ["cmd"] = "duel_reset", ["seed"] = 53,
                ["p0"] = "external", ["p1"] = "external", ["learner"] = 0,
                ["start_profile"] = "standard-3v3", ["reference_seat"] = 0,
            };
            request[seat] = "typo-controller";

            string error = server.RejectCommand(JsonSerializer.Serialize(request));

            Assert.That(error, Does.Contain(
                $"tactical-v3 duel_reset {seat} controller 'typo-controller' is unsupported"));
        }

        [TestCase(false)]
        [TestCase(true)]
        public void Process_DuelSaveBeforeSuccessfulResetRejectsWithoutFilesystemMutation(
            bool querySpacesFirst)
        {
            string parent = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "save-before-reset-" + Guid.NewGuid().ToString("N"));
            string replayPath = Path.Combine(parent, "nested", "should-not-exist.replay");
            try
            {
                using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
                if (querySpacesFirst) server.Request("{\"cmd\":\"duel_spaces\"}");

                string error = server.RejectCommand(JsonSerializer.Serialize(new
                {
                    cmd = "duel_save", path = replayPath,
                }));

                Assert.Multiple(() =>
                {
                    Assert.That(error, Does.Contain(
                        "tactical-v3 duel_save requires a successful duel_reset"));
                    Assert.That(Directory.Exists(parent), Is.False);
                    Assert.That(File.Exists(replayPath), Is.False);
                });
            }
            finally
            {
                if (Directory.Exists(parent)) Directory.Delete(parent, recursive: true);
            }
        }

        [Test]
        public void Process_CloseExitsZeroWithoutResponse()
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            server.CloseAndAssert();
        }

        [TestCase(0)]
        [TestCase(1)]
        public void Process_TerminalRewardIsRelativeToLearnerSeat(int learner)
        {
            using var server = TacticalV3ServerProcess.Start(CheckedInScenario);
            JsonElement terminal = server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_reset", seed = 59, p0 = "greedy", p1 = "greedy",
                learner, start_profile = "conversion-1v1-near", reference_seat = 0,
            }));
            int winner = terminal.GetProperty("winner").GetInt32();

            Assert.Multiple(() =>
            {
                Assert.That(terminal.GetProperty("terminated").GetBoolean(), Is.True);
                Assert.That(winner, Is.AnyOf(0, 1));
                Assert.That(terminal.GetProperty("reward").GetProperty("terminal_outcome")
                    .GetSingle(), Is.EqualTo(winner == learner ? 1f : -1f));
            });
        }

        private static string ReplaceRootFieldValue(
            string requestJson, string field, string invalidJson)
        {
            var request = JsonSerializer.Deserialize<Dictionary<string, JsonElement>>(
                requestJson)!;
            using JsonDocument invalid = JsonDocument.Parse(invalidJson);
            request[field] = invalid.RootElement.Clone();
            return JsonSerializer.Serialize(request);
        }

        private static string TacticalV3CommandRequest(
            TacticalV3ServerProcess server, string command, string savePath)
        {
            switch (command)
            {
                case "spaces":
                    return "{\"cmd\":\"spaces\"}";
                case "reset":
                    return "{\"cmd\":\"reset\",\"seed\":43}";
                case "step":
                {
                    JsonElement reset = server.Request("{\"cmd\":\"reset\",\"seed\":43}");
                    return JsonSerializer.Serialize(new
                    {
                        cmd = "step",
                        decision_id = reset.GetProperty("decision_id").GetInt64(),
                        candidate_id = reset.GetProperty("candidates")[0]
                            .GetProperty("candidate_id").GetInt32(),
                    });
                }
                case "duel_spaces":
                    return "{\"cmd\":\"duel_spaces\"}";
                case "duel_reset":
                    return JsonSerializer.Serialize(new
                    {
                        cmd = "duel_reset", seed = 43,
                        p0 = "external", p1 = "external", learner = 0,
                        start_profile = "standard-3v3", reference_seat = 0,
                    });
                case "duel_step":
                {
                    JsonElement reset = ResetForSelection(server, command);
                    return JsonSerializer.Serialize(new
                    {
                        cmd = "duel_step",
                        decision_id = reset.GetProperty("decision_id").GetInt64(),
                        candidate_id = reset.GetProperty("candidates")[0]
                            .GetProperty("candidate_id").GetInt32(),
                    });
                }
                case "duel_save":
                    server.Request(JsonSerializer.Serialize(new
                    {
                        cmd = "duel_reset", seed = 43,
                        p0 = "external", p1 = "external", learner = 0,
                        start_profile = "standard-3v3", reference_seat = 0,
                    }));
                    return JsonSerializer.Serialize(new { cmd = "duel_save", path = savePath });
                case "close":
                    return "{\"cmd\":\"close\"}";
                default:
                    throw new ArgumentOutOfRangeException(nameof(command));
            }
        }

        private static JsonElement ResetForSelection(
            TacticalV3ServerProcess server, string command) => command == "duel_step"
            ? server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_reset", seed = 41, p0 = "external", p1 = "external",
                learner = 0, start_profile = "standard-3v3", reference_seat = 0,
            }))
            : server.Request("{\"cmd\":\"reset\",\"seed\":41}");

        private static void AssertViewIdentities(JsonElement view)
        {
            long decisionId = view.GetProperty("decision_id").GetInt64();
            Assert.That(view.GetProperty("decision_id").ValueKind, Is.EqualTo(JsonValueKind.Number));
            Assert.That(view.GetProperty("candidates").GetArrayLength(), Is.GreaterThan(0));
            foreach (JsonElement candidate in view.GetProperty("candidates").EnumerateArray())
            {
                Assert.That(candidate.GetProperty("decision_id").GetInt64(), Is.EqualTo(decisionId));
                Assert.That(candidate.GetProperty("candidate_id").TryGetInt32(out _), Is.True);
            }
        }

        private static void AssertAuthoritativeGameStatesEqual(
            GameState expected, GameState actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.ActivePlayer, Is.EqualTo(expected.ActivePlayer));
                Assert.That(actual.Round, Is.EqualTo(expected.Round));
                Assert.That(actual.NextEntityId, Is.EqualTo(expected.NextEntityId));
                Assert.That(actual.IsGameOver, Is.EqualTo(expected.IsGameOver));
                Assert.That(actual.Winner, Is.EqualTo(expected.Winner));
                Assert.That(actual.MovedUnitIds.OrderBy(id => id),
                    Is.EqualTo(expected.MovedUnitIds.OrderBy(id => id)));
                Assert.That(actual.AttackedUnitIds.OrderBy(id => id),
                    Is.EqualTo(expected.AttackedUnitIds.OrderBy(id => id)));
                Assert.That(actual.MovementSpent.OrderBy(pair => pair.Key)
                        .Select(pair => (pair.Key, pair.Value.H, pair.Value.V)),
                    Is.EqualTo(expected.MovementSpent.OrderBy(pair => pair.Key)
                        .Select(pair => (pair.Key, pair.Value.H, pair.Value.V))));

                AssertGameConfigsEqual(expected.Config, actual.Config);
                Assert.That(actual.Config.TurnPolicy.RemainingActions(actual),
                    Is.EqualTo(expected.Config.TurnPolicy.RemainingActions(expected)));

                Tile[] expectedTiles = expected.Board.Tiles
                    .OrderBy(tile => tile.Coord.Q).ThenBy(tile => tile.Coord.R).ToArray();
                Tile[] actualTiles = actual.Board.Tiles
                    .OrderBy(tile => tile.Coord.Q).ThenBy(tile => tile.Coord.R).ToArray();
                Assert.That(actualTiles.Length, Is.EqualTo(expectedTiles.Length));
                for (int index = 0; index < expectedTiles.Length; index++)
                {
                    Assert.That(actualTiles[index].Coord, Is.EqualTo(expectedTiles[index].Coord));
                    Assert.That(actualTiles[index].Elevation,
                        Is.EqualTo(expectedTiles[index].Elevation));
                    Assert.That(actualTiles[index].Terrain,
                        Is.EqualTo(expectedTiles[index].Terrain));
                    Assert.That(actual.Board.Controller(actualTiles[index].Coord),
                        Is.EqualTo(expected.Board.Controller(expectedTiles[index].Coord)));
                }

                foreach (PlayerId seat in new[] { PlayerId.Player0, PlayerId.Player1 })
                {
                    Assert.That(actual.Board.DeploymentZone(seat),
                        Is.EquivalentTo(expected.Board.DeploymentZone(seat)));
                    AssertPlayersEqual(expected.Player(seat), actual.Player(seat));
                }
            });
        }

        private static void AssertGameConfigsEqual(GameConfig expected, GameConfig actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.StartingPoints, Is.EqualTo(expected.StartingPoints));
                Assert.That(actual.BountyRate, Is.EqualTo(expected.BountyRate));
                Assert.That(actual.GeneratorCost, Is.EqualTo(expected.GeneratorCost));
                Assert.That(actual.GeneratorOutput, Is.EqualTo(expected.GeneratorOutput));
                Assert.That(actual.GeneratorHealth, Is.EqualTo(expected.GeneratorHealth));
                Assert.That(actual.DamageFloor, Is.EqualTo(expected.DamageFloor));
                Assert.That(actual.DmgHighGroundBonus, Is.EqualTo(expected.DmgHighGroundBonus));
                Assert.That(actual.RangeHighGroundBonus, Is.EqualTo(expected.RangeHighGroundBonus));
                Assert.That(actual.RoundCap, Is.EqualTo(expected.RoundCap));
                Assert.That(actual.DesignFee, Is.EqualTo(expected.DesignFee));
                Assert.That(actual.MaxDesignPointCost, Is.EqualTo(expected.MaxDesignPointCost));
                Assert.That(actual.FixedTemplateCount, Is.EqualTo(expected.FixedTemplateCount));
                Assert.That(actual.TemplateSlotCount, Is.EqualTo(expected.TemplateSlotCount));
                Assert.That(actual.DeployCostMultiplier, Is.EqualTo(expected.DeployCostMultiplier));
                Assert.That(actual.TurnPolicy.GetType(), Is.EqualTo(expected.TurnPolicy.GetType()));
                Assert.That(actual.TurnPolicy.ActionsPerTurn,
                    Is.EqualTo(expected.TurnPolicy.ActionsPerTurn));
                Assert.That(actual.BiomesEnabled, Is.EqualTo(expected.BiomesEnabled));
                Assert.That(actual.WinConditions, Is.EqualTo(expected.WinConditions));
                Assert.That(actual.CaptureCost, Is.EqualTo(expected.CaptureCost));
                Assert.That(actual.EconomyWinThreshold, Is.EqualTo(expected.EconomyWinThreshold));
                Assert.That(actual.ScoreKills, Is.EqualTo(expected.ScoreKills));
                Assert.That(actual.ScorePoints, Is.EqualTo(expected.ScorePoints));
                Assert.That(actual.ScoreArmy, Is.EqualTo(expected.ScoreArmy));
                Assert.That(actual.ScoreTerritory, Is.EqualTo(expected.ScoreTerritory));
                Assert.That(actual.UpkeepFactor, Is.EqualTo(expected.UpkeepFactor));
                Assert.That(actual.CaptureFactor, Is.EqualTo(expected.CaptureFactor));
                Assert.That(actual.BuildFactor, Is.EqualTo(expected.BuildFactor));
                Assert.That(actual.TerritoryMode, Is.EqualTo(expected.TerritoryMode));
                Assert.That(actual.ClaimEndsTurn, Is.EqualTo(expected.ClaimEndsTurn));
                Assert.That(actual.BuildAnywhere, Is.EqualTo(expected.BuildAnywhere));
                Assert.That(actual.TerritoryIncome, Is.EqualTo(expected.TerritoryIncome));
                Assert.That(actual.GeneratorsEnabled, Is.EqualTo(expected.GeneratorsEnabled));
                Assert.That(actual.PointDecay, Is.EqualTo(expected.PointDecay));
                Assert.That(actual.FogOfWar, Is.EqualTo(expected.FogOfWar));
                foreach (TerrainType terrain in Enum.GetValues(typeof(TerrainType)))
                {
                    TerrainDef expectedTerrain = expected.Terrain(terrain);
                    TerrainDef actualTerrain = actual.Terrain(terrain);
                    Assert.That(actualTerrain.MoveCost, Is.EqualTo(expectedTerrain.MoveCost));
                    Assert.That(actualTerrain.Concealment,
                        Is.EqualTo(expectedTerrain.Concealment));
                    Assert.That(actualTerrain.Defense, Is.EqualTo(expectedTerrain.Defense));
                    Assert.That(actualTerrain.Passable, Is.EqualTo(expectedTerrain.Passable));
                }
            });
        }

        private static void AssertPlayersEqual(PlayerState expected, PlayerState actual)
        {
            Assert.That(actual.Id, Is.EqualTo(expected.Id));
            Assert.That(actual.Points, Is.EqualTo(expected.Points));
            Assert.That(actual.DestroyedValue, Is.EqualTo(expected.DestroyedValue));
            Assert.That(actual.Barracks.Count, Is.EqualTo(expected.Barracks.Count));
            for (int index = 0; index < expected.Barracks.Count; index++)
            {
                Assert.That(actual.Barracks[index].Name,
                    Is.EqualTo(expected.Barracks[index].Name));
                AssertUnitStatsEqual(expected.Barracks[index].Stats,
                    actual.Barracks[index].Stats);
            }

            Unit[] expectedUnits = expected.UnitsOnBoard.OrderBy(unit => unit.Id).ToArray();
            Unit[] actualUnits = actual.UnitsOnBoard.OrderBy(unit => unit.Id).ToArray();
            Assert.That(actualUnits.Length, Is.EqualTo(expectedUnits.Length));
            for (int index = 0; index < expectedUnits.Length; index++)
            {
                Unit expectedUnit = expectedUnits[index];
                Unit actualUnit = actualUnits[index];
                Assert.That(actualUnit.Id, Is.EqualTo(expectedUnit.Id));
                Assert.That(actualUnit.Owner, Is.EqualTo(expectedUnit.Owner));
                Assert.That(actualUnit.Cell, Is.EqualTo(expectedUnit.Cell));
                Assert.That(actualUnit.Elevation, Is.EqualTo(expectedUnit.Elevation));
                Assert.That(actualUnit.CurrentHp, Is.EqualTo(expectedUnit.CurrentHp));
                Assert.That(actualUnit.Name, Is.EqualTo(expectedUnit.Name));
                AssertUnitStatsEqual(expectedUnit.Stats, actualUnit.Stats);
            }

            Generator[] expectedGenerators = expected.Generators
                .OrderBy(generator => generator.Id).ToArray();
            Generator[] actualGenerators = actual.Generators
                .OrderBy(generator => generator.Id).ToArray();
            Assert.That(actualGenerators.Length, Is.EqualTo(expectedGenerators.Length));
            for (int index = 0; index < expectedGenerators.Length; index++)
            {
                Generator expectedGenerator = expectedGenerators[index];
                Generator actualGenerator = actualGenerators[index];
                Assert.That(actualGenerator.Id, Is.EqualTo(expectedGenerator.Id));
                Assert.That(actualGenerator.Owner, Is.EqualTo(expectedGenerator.Owner));
                Assert.That(actualGenerator.Cell, Is.EqualTo(expectedGenerator.Cell));
                Assert.That(actualGenerator.Elevation, Is.EqualTo(expectedGenerator.Elevation));
                Assert.That(actualGenerator.CurrentHp, Is.EqualTo(expectedGenerator.CurrentHp));
                Assert.That(actualGenerator.Strength, Is.EqualTo(expectedGenerator.Strength));
            }
        }

        private static void AssertUnitStatsEqual(UnitStats expected, UnitStats actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Health, Is.EqualTo(expected.Health));
                Assert.That(actual.Damage, Is.EqualTo(expected.Damage));
                Assert.That(actual.Defense, Is.EqualTo(expected.Defense));
                Assert.That(actual.Movement, Is.EqualTo(expected.Movement));
                Assert.That(actual.VerticalMovement, Is.EqualTo(expected.VerticalMovement));
                Assert.That(actual.Range, Is.EqualTo(expected.Range));
                Assert.That(actual.RangeArc, Is.EqualTo(expected.RangeArc));
                Assert.That(actual.Vision, Is.EqualTo(expected.Vision));
                Assert.That(actual.VisionArc, Is.EqualTo(expected.VisionArc));
            });
        }

        private static string AssertSameReply(
            TacticalV3ServerProcess first, TacticalV3ServerProcess second, string request)
        {
            string firstReply = first.RequestRaw(request);
            string secondReply = second.RequestRaw(request);
            Assert.That(secondReply, Is.EqualTo(firstReply), request);
            return firstReply;
        }

        private static int CandidateId(string view, string kind)
        {
            using JsonDocument parsed = JsonDocument.Parse(view);
            return parsed.RootElement.GetProperty("candidates").EnumerateArray()
                .Single(candidate => candidate.GetProperty("kind").GetString() == kind)
                .GetProperty("candidate_id").GetInt32();
        }

        private static string LegacySequenceShapeHash(string environment)
        {
            using var server = TacticalV3ServerProcess.StartEnvironment(environment);
            var replies = new List<JsonElement>
            {
                server.Request("{\"cmd\":\"spaces\"}"),
            };

            JsonElement reset = server.Request("{\"cmd\":\"reset\",\"seed\":29}");
            replies.Add(reset);
            replies.Add(server.Request(JsonSerializer.Serialize(new
            {
                cmd = "step",
                action = FirstLegalAction(reset),
            })));
            replies.Add(server.Request("{\"cmd\":\"duel_spaces\"}"));
            JsonElement duelReset = server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_reset",
                seed = 31,
                p0 = "external",
                p1 = "external",
                learner = 0,
            }));
            replies.Add(duelReset);
            replies.Add(server.Request(JsonSerializer.Serialize(new
            {
                cmd = "duel_step",
                action = FirstLegalAction(duelReset),
            })));

            string normalized = string.Join("\n", replies.Select(JsonShape));
            byte[] hash = SHA256.HashData(Encoding.UTF8.GetBytes(normalized));
            return Convert.ToHexString(hash).ToLowerInvariant();
        }

        private static int FirstLegalAction(JsonElement view)
        {
            int action = 0;
            foreach (JsonElement legal in view.GetProperty("mask").EnumerateArray())
            {
                if (legal.GetBoolean()) return action;
                action++;
            }

            throw new InvalidDataException("legacy reset emitted no legal action");
        }

        private static string JsonShape(JsonElement value) => value.ValueKind switch
        {
            JsonValueKind.Object => "{" + string.Join(",",
                value.EnumerateObject().Select(property =>
                    property.Name + ":" + JsonShape(property.Value))) + "}",
            JsonValueKind.Array => JsonArrayShape(value),
            JsonValueKind.String => "string",
            JsonValueKind.Number => "number",
            JsonValueKind.True or JsonValueKind.False => "bool",
            JsonValueKind.Null => "null",
            _ => value.ValueKind.ToString(),
        };

        private static string JsonArrayShape(JsonElement value)
        {
            var runs = new List<string>();
            string? current = null;
            int count = 0;
            foreach (JsonElement element in value.EnumerateArray())
            {
                string next = JsonShape(element);
                if (current == next)
                {
                    count++;
                    continue;
                }

                if (current != null) runs.Add(count + "*" + current);
                current = next;
                count = 1;
            }

            if (current != null) runs.Add(count + "*" + current);
            return "[" + value.GetArrayLength() + ":" + string.Join(",", runs) + "]";
        }

        private static string RepositoryPath(params string[] parts)
        {
            string root = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", ".."));
            return Path.Combine(new[] { root }.Concat(parts).ToArray());
        }

        private static IEnumerable<WrongReferenceFamily> WrongReferenceFamilies =>
            Enum.GetValues(typeof(WrongReferenceFamily)).Cast<WrongReferenceFamily>();

        private static IEnumerable<ContractMismatch> ContractMismatches =>
            Enum.GetValues(typeof(ContractMismatch)).Cast<ContractMismatch>();

        private static void ApplyWrongReference(
            TacticalV3View view, WrongReferenceFamily family)
        {
            TacticalV3Observation observation = view.Decision.Observation;
            TacticalV3TokenRef cells = new TacticalV3TokenRef(TacticalV3TableKind.Cells, 0);
            TacticalV3TokenRef units = new TacticalV3TokenRef(TacticalV3TableKind.Units, 0);
            TacticalV3TokenRef definitions = new TacticalV3TokenRef(
                TacticalV3TableKind.CapabilityDefinitions, 0);
            TacticalV3UnitToken unit = observation.Units[0];
            TacticalV3MemoryToken memory = observation.Memory[0];
            TacticalV3CapabilityAllocationToken allocation = observation.CapabilityAllocations[0];
            TacticalV3Candidate candidate = view.Decision.Candidates[0];
            TacticalV3ProjectedDelta projection = candidate.Projection;
            TacticalV3RelationToken neighbor = observation.Relations.First(
                item => item.Kind == TacticalV3RelationKind.Neighbor);
            TacticalV3RelationToken occupies = observation.Relations.First(
                item => item.Kind == TacticalV3RelationKind.Occupies);
            TacticalV3RelationToken hasCapability = observation.Relations.First(
                item => item.Kind == TacticalV3RelationKind.HasCapability);

            switch (family)
            {
                case WrongReferenceFamily.UnitCell:
                    SetAutoProperty(unit, nameof(TacticalV3UnitToken.Cell), units);
                    break;
                case WrongReferenceFamily.UnknownTable:
                    SetAutoProperty(unit, nameof(TacticalV3UnitToken.Cell),
                        new TacticalV3TokenRef((TacticalV3TableKind)999, 0));
                    break;
                case WrongReferenceFamily.MemoryCell:
                    SetAutoProperty(memory, nameof(TacticalV3MemoryToken.Cell), units);
                    break;
                case WrongReferenceFamily.AllocationOwner:
                    SetAutoProperty(allocation, nameof(TacticalV3CapabilityAllocationToken.Owner), definitions);
                    break;
                case WrongReferenceFamily.AllocationDefinition:
                    SetAutoProperty(allocation, nameof(TacticalV3CapabilityAllocationToken.Definition), units);
                    break;
                case WrongReferenceFamily.AllocationCapability:
                    SetAutoProperty(allocation, nameof(TacticalV3CapabilityAllocationToken.Capability),
                        allocation.Capability == TacticalV3CapabilityKind.Health
                            ? TacticalV3CapabilityKind.Damage
                            : TacticalV3CapabilityKind.Health);
                    break;
                case WrongReferenceFamily.CandidateActor:
                    SetAutoProperty(candidate, nameof(TacticalV3Candidate.Actor), (TacticalV3TokenRef?)cells);
                    break;
                case WrongReferenceFamily.CandidateTarget:
                    SetAutoProperty(candidate, nameof(TacticalV3Candidate.Target), (TacticalV3TokenRef?)cells);
                    break;
                case WrongReferenceFamily.CandidateTemplate:
                    SetAutoProperty(candidate, nameof(TacticalV3Candidate.Template), (TacticalV3TokenRef?)cells);
                    break;
                case WrongReferenceFamily.CandidateCell:
                    SetAutoProperty(candidate, nameof(TacticalV3Candidate.Cell), (TacticalV3TokenRef?)units);
                    break;
                case WrongReferenceFamily.ProjectionSourceCell:
                    SetAutoProperty(projection, nameof(TacticalV3ProjectedDelta.SourceCell),
                        (TacticalV3TokenRef?)units);
                    break;
                case WrongReferenceFamily.ProjectionDestinationCell:
                    SetAutoProperty(projection, nameof(TacticalV3ProjectedDelta.DestinationCell),
                        (TacticalV3TokenRef?)units);
                    break;
                case WrongReferenceFamily.ProjectionTemplate:
                    SetAutoProperty(projection, nameof(TacticalV3ProjectedDelta.Template),
                        (TacticalV3TokenRef?)cells);
                    break;
                case WrongReferenceFamily.ProjectionTarget:
                    SetAutoProperty(projection, nameof(TacticalV3ProjectedDelta.Target),
                        (TacticalV3TokenRef?)cells);
                    break;
                case WrongReferenceFamily.NeighborSource:
                    SetAutoProperty(neighbor, nameof(TacticalV3RelationToken.Source), units);
                    break;
                case WrongReferenceFamily.NeighborTarget:
                    SetAutoProperty(neighbor, nameof(TacticalV3RelationToken.Target), units);
                    break;
                case WrongReferenceFamily.OccupiesSource:
                    SetAutoProperty(occupies, nameof(TacticalV3RelationToken.Source), cells);
                    break;
                case WrongReferenceFamily.OccupiesTarget:
                    SetAutoProperty(occupies, nameof(TacticalV3RelationToken.Target), units);
                    break;
                case WrongReferenceFamily.HasCapabilitySource:
                    SetAutoProperty(hasCapability, nameof(TacticalV3RelationToken.Source), cells);
                    break;
                case WrongReferenceFamily.HasCapabilityTarget:
                    SetAutoProperty(hasCapability, nameof(TacticalV3RelationToken.Target), units);
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(family));
            }
        }

        private static TacticalV3Contract CloneContract(
            TacticalV3Contract source,
            string? environmentKind = null,
            string? encodingHash = null)
        {
            var clone = (TacticalV3Contract)RuntimeHelpers.GetUninitializedObject(
                typeof(TacticalV3Contract));
            SetAutoProperty(clone, nameof(TacticalV3Contract.Version), source.Version);
            SetAutoProperty(clone, nameof(TacticalV3Contract.EnvironmentKind),
                environmentKind ?? source.EnvironmentKind);
            SetAutoProperty(clone, nameof(TacticalV3Contract.ContractHash), source.ContractHash);
            SetAutoProperty(clone, nameof(TacticalV3Contract.EncodingHash),
                encodingHash ?? source.EncodingHash);
            SetAutoProperty(clone, nameof(TacticalV3Contract.CapacityHash), source.CapacityHash);
            SetAutoProperty(clone, nameof(TacticalV3Contract.Match), source.Match);
            SetAutoProperty(clone, nameof(TacticalV3Contract.Encoding), source.Encoding);
            SetAutoProperty(clone, nameof(TacticalV3Contract.Capacity), source.Capacity);
            return clone;
        }

        private static TacticalV3CellToken CellWithController(
            TacticalV3CellToken cell, TacticalV3RelativeOwner controller) =>
            new TacticalV3CellToken(
                cell.Q, cell.R, cell.Terrain, cell.Elevation,
                cell.SelfDeploymentZone, cell.OpponentDeploymentZone, controller,
                cell.IsBoundary, cell.CurrentlyVisible, cell.PreviouslyObserved);

        private static void ReplaceObservation(
            TacticalV3View view,
            IReadOnlyList<TacticalV3CellToken>? cells = null,
            IReadOnlyList<TacticalV3RelationToken>? relations = null,
            IReadOnlyList<TacticalV3MemoryToken>? memory = null)
        {
            TacticalV3Observation original = view.Decision.Observation;
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.Observation),
                new TacticalV3Observation(
                    cells ?? original.Cells, original.Units, original.Templates,
                    original.CapabilityDefinitions, original.CapabilityAllocations,
                    original.Rules, memory ?? original.Memory, relations ?? original.Relations));
        }

        private static TacticalV3View ViewWithMemory(
            int seed,
            PlayerId learnerSeat = PlayerId.Player0,
            PlayerId referenceSeat = PlayerId.Player0)
        {
            var env = new TacticalV3DuelEnv(TacticalV3Fixtures.ProfiledConfig());
            IAgent? controller0 = learnerSeat == PlayerId.Player1
                ? new RandomAgent(seed + 1000)
                : null;
            TacticalV3View view = env.Reset(
                seed, controller0, null, "standard-3v3", referenceSeat, learnerSeat);
            TacticalV3Observation original = view.Decision.Observation;
            var memory = new[]
            {
                new TacticalV3MemoryToken(
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, 0),
                    lastSeenRound: 1, observationAge: 0, lastKnownCurrentHp: 2,
                    currentlyVisible: true),
            };
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.Observation),
                new TacticalV3Observation(
                    original.Cells, original.Units, original.Templates,
                    original.CapabilityDefinitions, original.CapabilityAllocations,
                    original.Rules, memory, original.Relations));
            return view;
        }

        private static TrainingScenario LoadStrictScenario()
        {
            string gymServerDll = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));
            Assembly assembly = Assembly.LoadFrom(gymServerDll);
            Type scenarioJson = assembly.GetType(
                "HexWars.GymServer.ScenarioJson", throwOnError: true)!;
            MethodInfo load = scenarioJson.GetMethod(
                "Load", BindingFlags.Public | BindingFlags.Static)!;
            return (TrainingScenario)load.Invoke(null, new object[] { CheckedInScenario })!;
        }

        private static object InvokeWire(string methodName, params object[] arguments)
        {
            string gymServerDll = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));
            Assembly assembly = Assembly.LoadFrom(gymServerDll);
            Type wireType = assembly.GetType("HexWars.GymServer.TacticalV3Wire", throwOnError: true)!;
            MethodInfo method = wireType.GetMethods(BindingFlags.Public | BindingFlags.Static)
                .Single(candidate => candidate.Name == methodName &&
                    candidate.GetParameters().Length == arguments.Length);
            return method.Invoke(null, arguments)!;
        }

        private static void AssertIndependentSpacesSchema(JsonElement spaces)
        {
            JsonElement match = spaces.GetProperty("match");
            AssertProperties(match,
                "board", "game", "max_controllable_units", "max_steps", "placement_policy",
                "reward", "start_distribution", "start_profiles", "starting_unit_count",
                "templates");
            JsonElement board = match.GetProperty("board");
            AssertProperties(board,
                "flat_chance", "forest_weight", "height", "hex_offset_layout", "max_elevation",
                "plains_weight", "rough_weight", "water_weight", "width", "zone_depth");
            Assert.That(board.GetProperty("hex_offset_layout").GetString(), Is.EqualTo("odd-q"));
            AssertJsonKinds(board, JsonValueKind.Number,
                "flat_chance", "forest_weight", "height", "max_elevation", "plains_weight",
                "rough_weight", "water_weight", "width", "zone_depth");

            JsonElement game = match.GetProperty("game");
            AssertProperties(game,
                "actions_per_turn", "biomes_enabled", "bounty_rate", "build_anywhere",
                "build_factor", "capture_cost", "capture_factor", "claim_ends_turn",
                "damage_floor", "deploy_cost_multiplier", "design_fee",
                "dmg_high_ground_bonus", "economy_win_threshold", "fixed_template_count",
                "fog_of_war", "generator_cost", "generator_health", "generator_output",
                "generators_enabled", "max_design_point_cost", "point_decay",
                "range_high_ground_bonus", "round_cap", "score_army", "score_kills",
                "score_points", "score_territory", "starting_points", "template_slot_count",
                "terrain", "territory_income", "territory_mode", "turn_policy", "upkeep_factor",
                "win_conditions");
            AssertJsonKinds(game, JsonValueKind.Number,
                "bounty_rate", "build_factor", "capture_cost", "capture_factor", "damage_floor",
                "deploy_cost_multiplier", "design_fee", "dmg_high_ground_bonus",
                "economy_win_threshold", "fixed_template_count", "generator_cost",
                "generator_health", "generator_output", "max_design_point_cost", "point_decay",
                "range_high_ground_bonus", "round_cap", "score_army", "score_kills",
                "score_points", "score_territory", "starting_points", "template_slot_count",
                "territory_income", "upkeep_factor");
            AssertJsonBooleanFields(game, "biomes_enabled", "build_anywhere", "claim_ends_turn",
                "generators_enabled", "fog_of_war", "territory_mode");
            Assert.That(game.GetProperty("turn_policy").ValueKind, Is.EqualTo(JsonValueKind.String));
            Assert.That(game.GetProperty("win_conditions").ValueKind, Is.EqualTo(JsonValueKind.Array));
            AssertStringArray(game.GetProperty("win_conditions"), "annihilation");
            JsonElement terrain = game.GetProperty("terrain");
            AssertProperties(terrain, "forest", "plains", "rough", "water");
            foreach (JsonProperty item in terrain.EnumerateObject())
            {
                AssertProperties(item.Value, "concealment", "defense", "move_cost", "passable");
                AssertJsonKinds(item.Value, JsonValueKind.Number,
                    "concealment", "defense", "move_cost");
                Assert.That(item.Value.GetProperty("passable").ValueKind,
                    Is.AnyOf(JsonValueKind.True, JsonValueKind.False));
            }

            AssertProperties(match.GetProperty("reward"),
                "material_adjustment_bound", "points_weight", "terminal_non_win",
                "terminal_win", "time_pressure_bound");
            Assert.That(match.GetProperty("start_distribution").GetArrayLength(), Is.Zero);
            Assert.That(match.GetProperty("start_profiles").GetArrayLength(), Is.Zero);
            JsonElement templates = match.GetProperty("templates");
            Assert.That(templates.GetArrayLength(), Is.EqualTo(5));
            foreach (JsonElement template in templates.EnumerateArray())
            {
                AssertProperties(template, "capability_allocations");
                JsonElement allocations = template.GetProperty("capability_allocations");
                Assert.That(allocations.GetArrayLength(), Is.EqualTo(9));
                Assert.That(allocations.EnumerateArray()
                    .Select(item => item.GetProperty("capability").GetString()).ToArray(),
                    Is.EqualTo(new[]
                    {
                        "health", "damage", "defense", "movement", "vertical_movement",
                        "range", "range_arc", "vision", "vision_arc",
                    }));
                foreach (JsonElement allocation in allocations.EnumerateArray())
                {
                    AssertProperties(allocation,
                        "capability", "effective_value", "purchased_level");
                    Assert.That(allocation.GetProperty("effective_value").TryGetInt32(out _), Is.True);
                    Assert.That(allocation.GetProperty("purchased_level").TryGetInt32(out _), Is.True);
                }
            }

            JsonElement capacity = spaces.GetProperty("capacity");
            AssertProperties(capacity,
                "max_cells", "max_units", "max_templates", "max_capability_definitions",
                "max_capability_allocations", "max_rules", "max_memory_records",
                "max_relations", "max_candidates");
            Assert.That(capacity.EnumerateObject().All(item => item.Value.TryGetInt32(out _)), Is.True);
            Assert.That(capacity.GetProperty("max_cells").GetInt32(), Is.EqualTo(512));
            Assert.That(capacity.GetProperty("max_units").GetInt32(), Is.EqualTo(64));
            Assert.That(capacity.GetProperty("max_templates").GetInt32(), Is.EqualTo(32));
            Assert.That(capacity.GetProperty("max_capability_definitions").GetInt32(), Is.EqualTo(128));
            Assert.That(capacity.GetProperty("max_capability_allocations").GetInt32(), Is.EqualTo(2048));
            Assert.That(capacity.GetProperty("max_rules").GetInt32(), Is.EqualTo(128));
            Assert.That(capacity.GetProperty("max_memory_records").GetInt32(), Is.EqualTo(64));
            Assert.That(capacity.GetProperty("max_relations").GetInt32(), Is.EqualTo(65536));
            Assert.That(capacity.GetProperty("max_candidates").GetInt32(), Is.EqualTo(32768));

            JsonElement encoding = spaces.GetProperty("encoding");
            AssertProperties(encoding,
                "capability_descriptors", "candidate_schema", "enums", "hex_offset_layout",
                "schema_version", "tables", "token_reference_schema", "version");
            Assert.That(encoding.GetProperty("hex_offset_layout").GetString(), Is.EqualTo("odd-q"));
            Assert.That(encoding.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
            Assert.That(encoding.GetProperty("version").GetString(), Is.EqualTo("tactical-v3"));
            AssertStringArray(encoding.GetProperty("token_reference_schema"),
                "table:table_kind", "row:int32");

            JsonElement tables = encoding.GetProperty("tables");
            AssertProperties(tables,
                "cells", "units", "templates", "capability_definitions",
                "capability_allocations", "rules", "memory", "relations", "candidates");
            AssertStringArray(tables.GetProperty("cells"),
                "q:int32", "r:int32", "terrain:terrain_type", "elevation:int32",
                "self_deployment_zone:bool", "opponent_deployment_zone:bool",
                "controller:nullable_relative_owner", "is_boundary:bool",
                "currently_visible:bool", "previously_observed:bool");
            AssertStringArray(tables.GetProperty("units"),
                "owner:relative_owner", "current_hp:int32", "max_hp:int32", "cell:token_ref",
                "elevation:int32", "moved:bool", "attacked:bool",
                "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                "point_cost:int32", "deploy_cost:int32", "currently_visible:bool");
            AssertStringArray(tables.GetProperty("templates"),
                "owner:relative_owner", "point_cost:int32", "deploy_cost:int32",
                "is_fixed:bool", "is_deployable:bool");
            AssertStringArray(tables.GetProperty("capability_definitions"), "kind:capability_kind");
            AssertStringArray(tables.GetProperty("capability_allocations"),
                "owner:token_ref", "definition:token_ref", "capability:capability_kind",
                "purchased_level:int32", "effective_value:int32");
            AssertStringArray(tables.GetProperty("rules"),
                "kind:rule_kind", "int_value:int32", "float_value:float32", "bool_value:bool");
            AssertStringArray(tables.GetProperty("memory"),
                "cell:token_ref", "last_seen_round:int32", "observation_age:int32",
                "last_known_current_hp:int32", "currently_visible:bool");
            AssertStringArray(tables.GetProperty("relations"),
                "kind:relation_kind", "source:token_ref", "target:token_ref",
                "int_feature:int32", "float_feature:float32", "bool_feature:bool");

            JsonElement candidateSchema = encoding.GetProperty("candidate_schema");
            AssertProperties(candidateSchema, "fields", "projection_fields");
            AssertStringArray(candidateSchema.GetProperty("fields"),
                "candidate_id:int32", "decision_id:int64", "kind:candidate_kind",
                "actor:nullable_token_ref", "target:nullable_token_ref",
                "template:nullable_token_ref", "cell:nullable_token_ref",
                "projection:projected_delta");
            AssertStringArray(candidateSchema.GetProperty("projection_fields"),
                "source_cell:nullable_token_ref", "destination_cell:nullable_token_ref",
                "template:nullable_token_ref", "target:nullable_token_ref",
                "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                "target_hp_delta:int32", "damage:int32", "is_lethal:bool",
                "bounty_delta:int32", "points_delta:int32", "round_delta:int32",
                "is_terminal:bool");
            AssertStringArray(tables.GetProperty("candidates"),
                candidateSchema.GetProperty("fields").EnumerateArray()
                    .Select(item => item.GetString()!).ToArray());

            AssertIndependentEnumAndCapabilityCatalogs(encoding);
        }

        private static void AssertIndependentEnumAndCapabilityCatalogs(JsonElement encoding)
        {
            JsonElement enums = encoding.GetProperty("enums");
            AssertProperties(enums,
                "table_kind", "relative_owner", "terrain_type", "rule_kind", "relation_kind",
                "capability_kind", "action_kind", "capability_relation_kind", "candidate_kind",
                "win_condition");
            AssertStringArray(enums.GetProperty("table_kind"),
                "cells", "units", "templates", "capability_definitions",
                "capability_allocations", "rules", "memory_records", "relations", "candidates");
            AssertStringArray(enums.GetProperty("relative_owner"), "self", "opponent");
            AssertStringArray(enums.GetProperty("terrain_type"),
                "plains", "forest", "rough", "water");
            AssertStringArray(enums.GetProperty("rule_kind"),
                "win_conditions", "round", "round_cap", "actions_per_turn", "starting_points",
                "self_points", "opponent_points", "damage_floor", "damage_high_ground_bonus",
                "range_high_ground_bonus", "bounty_rate", "deploy_cost_multiplier",
                "fog_of_war", "max_design_point_cost", "design_fee");
            AssertStringArray(enums.GetProperty("relation_kind"),
                "neighbor", "occupies", "has_capability");
            AssertStringArray(enums.GetProperty("capability_kind"),
                "health", "damage", "defense", "movement", "vertical_movement",
                "range", "range_arc", "vision", "vision_arc");
            AssertStringArray(enums.GetProperty("action_kind"),
                "move", "attack", "deploy", "end_turn");
            AssertStringArray(enums.GetProperty("capability_relation_kind"),
                "opposes", "reduces", "enables_action");
            AssertStringArray(enums.GetProperty("candidate_kind"),
                "attack", "move", "deploy", "end_turn");
            AssertStringArray(enums.GetProperty("win_condition"),
                "none", "annihilation", "economy", "score");

            JsonElement capabilities = encoding.GetProperty("capability_descriptors");
            AssertProperties(capabilities,
                "definition_fields", "definitions", "relation_fields", "relations");
            AssertStringArray(capabilities.GetProperty("definition_fields"),
                "kind:capability_kind");
            AssertStringArray(capabilities.GetProperty("definitions"),
                "health", "damage", "defense", "movement", "vertical_movement",
                "range", "range_arc", "vision", "vision_arc");
            AssertStringArray(capabilities.GetProperty("relation_fields"),
                "source:capability_kind", "kind:capability_relation_kind",
                "target:capability_or_action");
            JsonElement[] relations = capabilities.GetProperty("relations").EnumerateArray().ToArray();
            Assert.That(relations, Has.Length.EqualTo(4));
            string[] expected =
            {
                "damage|opposes|capability:health",
                "defense|reduces|capability:damage",
                "range|enables_action|action:attack",
                "range_arc|enables_action|action:attack",
            };
            Assert.That(relations.Select(item =>
            {
                AssertProperties(item, "source", "kind", "target");
                return item.GetProperty("source").GetString() + "|" +
                    item.GetProperty("kind").GetString() + "|" +
                    item.GetProperty("target").GetString();
            }).ToArray(), Is.EqualTo(expected));
        }

        private static void AssertPinnedProfileSchemas(JsonElement match)
        {
            AssertPinnedStartProfiles(match.GetProperty("start_profiles"));
            AssertPinnedStartDistribution(match.GetProperty("start_distribution"));
        }

        private static void AssertPinnedStartProfiles(JsonElement profiles)
        {
            foreach (JsonElement profile in profiles.EnumerateArray())
            {
                AssertProperties(profile,
                    "id", "learner_unit_count", "opponent_unit_count", "separation");
                Assert.That(profile.GetProperty("learner_unit_count").TryGetInt32(out _), Is.True);
                Assert.That(profile.GetProperty("opponent_unit_count").TryGetInt32(out _), Is.True);
            }
            string[] actual = profiles.EnumerateArray().Select(profile =>
                profile.GetProperty("id").GetString() + "|" +
                profile.GetProperty("learner_unit_count").GetInt32() + "|" +
                profile.GetProperty("opponent_unit_count").GetInt32() + "|" +
                profile.GetProperty("separation").GetString()).ToArray();
            Assert.That(actual, Is.EqualTo(new[]
            {
                "conversion-1v1-far|1|1|far",
                "conversion-1v1-medium|1|1|medium",
                "conversion-1v1-near|1|1|near",
                "conversion-2v1-far|2|1|far",
                "conversion-2v1-medium|2|1|medium",
                "conversion-2v1-near|2|1|near",
                "conversion-3v1-far|3|1|far",
                "conversion-3v1-medium|3|1|medium",
                "conversion-3v1-near|3|1|near",
                "standard-3v3|3|3|legacy-mirrored",
            }));
        }

        private static void AssertPinnedStartDistribution(JsonElement distribution)
        {
            foreach (JsonElement weight in distribution.EnumerateArray())
            {
                AssertProperties(weight, "profile_id", "basis_points");
                Assert.That(weight.GetProperty("basis_points").TryGetInt32(out _), Is.True);
            }
            string[] actual = distribution.EnumerateArray().Select(weight =>
                weight.GetProperty("profile_id").GetString() + "|" +
                weight.GetProperty("basis_points").GetInt32()).ToArray();
            Assert.That(actual, Is.EqualTo(new[]
            {
                "conversion-1v1-far|500",
                "conversion-1v1-medium|0",
                "conversion-1v1-near|500",
                "conversion-2v1-far|500",
                "conversion-2v1-medium|0",
                "conversion-2v1-near|500",
                "conversion-3v1-far|500",
                "conversion-3v1-medium|0",
                "conversion-3v1-near|500",
                "standard-3v3|7000",
            }));
        }

        private static void AssertNumericWireTypes(object wire)
        {
            AssertTypes<long>(wire, "decision_id");
            AssertTypes<int>(wire, "seat", "winner", "reference_seat");
            object observation = Property(wire, "observation");
            AssertTypes<int>(First(Property(observation, "cells")), "q", "r", "elevation");
            AssertTypes<int>(First(Property(observation, "units")),
                "current_hp", "max_hp", "elevation", "horizontal_movement_spent",
                "vertical_movement_spent", "point_cost", "deploy_cost");
            AssertTypes<int>(First(Property(observation, "templates")),
                "point_cost", "deploy_cost");
            AssertTypes<int>(First(Property(observation, "capability_allocations")),
                "purchased_level", "effective_value");
            object rule = First(Property(observation, "rules"));
            AssertTypes<int>(rule, "int_value");
            AssertTypes<float>(rule, "float_value");
            AssertTypes<int>(First(Property(observation, "memory")),
                "last_seen_round", "observation_age", "last_known_current_hp");
            object relation = First(Property(observation, "relations"));
            AssertTypes<int>(relation, "int_feature");
            AssertTypes<float>(relation, "float_feature");
            object candidate = First(Property(wire, "candidates"));
            AssertTypes<int>(candidate, "candidate_id");
            AssertTypes<long>(candidate, "decision_id");
            AssertTypes<int>(Property(candidate, "projection"),
                "horizontal_movement_spent", "vertical_movement_spent", "target_hp_delta",
                "damage", "bounty_delta", "points_delta", "round_delta");
            AssertTypes<float>(Property(wire, "reward"),
                "terminal_outcome", "known_health_adjusted_material_progress",
                "public_resource_progress", "time_pressure", "total");
        }

        private static void AssertNumericJsonTypes(JsonElement view)
        {
            Assert.That(view.GetProperty("decision_id").TryGetInt64(out _), Is.True);
            AssertJsonInt32Fields(view, "seat", "winner", "reference_seat");
            JsonElement observation = view.GetProperty("observation");
            AssertJsonInt32Fields(observation.GetProperty("cells")[0], "q", "r", "elevation");
            AssertJsonInt32Fields(observation.GetProperty("units")[0],
                "current_hp", "max_hp", "elevation", "horizontal_movement_spent",
                "vertical_movement_spent", "point_cost", "deploy_cost");
            AssertJsonInt32Fields(observation.GetProperty("templates")[0],
                "point_cost", "deploy_cost");
            AssertJsonInt32Fields(observation.GetProperty("capability_allocations")[0],
                "purchased_level", "effective_value");
            AssertJsonInt32Fields(observation.GetProperty("rules")[0], "int_value");
            Assert.That(observation.GetProperty("rules").EnumerateArray()
                .Single(item => item.GetProperty("kind").GetString() == "bounty_rate")
                .GetProperty("float_value").TryGetInt32(out _), Is.False);
            AssertJsonInt32Fields(observation.GetProperty("memory")[0],
                "last_seen_round", "observation_age", "last_known_current_hp");
            AssertJsonInt32Fields(observation.GetProperty("relations")[0], "int_feature");
            Assert.That(observation.GetProperty("relations")[0].GetProperty("float_feature")
                .TryGetInt32(out _), Is.False);
            JsonElement candidate = view.GetProperty("candidates")[0];
            AssertJsonInt32Fields(candidate, "candidate_id");
            Assert.That(candidate.GetProperty("decision_id").TryGetInt64(out _), Is.True);
            Assert.That(candidate.GetProperty("decision_id").TryGetInt32(out _), Is.False);
            AssertJsonInt32Fields(candidate.GetProperty("projection"),
                "horizontal_movement_spent", "vertical_movement_spent", "target_hp_delta",
                "damage", "bounty_delta", "points_delta", "round_delta");
            JsonElement reward = view.GetProperty("reward");
            foreach (string name in new[]
            {
                "terminal_outcome", "known_health_adjusted_material_progress",
                "public_resource_progress", "time_pressure", "total",
            })
            {
                Assert.That(reward.GetProperty(name).ValueKind, Is.EqualTo(JsonValueKind.Number));
                Assert.That(reward.GetProperty(name).TryGetInt32(out _), Is.False, name);
                _ = reward.GetProperty(name).GetSingle();
            }
        }

        private static void AssertTypes<T>(object target, params string[] names)
        {
            foreach (string name in names)
                Assert.That(Property(target, name), Is.TypeOf<T>(), name);
        }

        private static void AssertJsonInt32Fields(JsonElement target, params string[] names)
        {
            foreach (string name in names)
            {
                Assert.That(target.GetProperty(name).ValueKind, Is.EqualTo(JsonValueKind.Number), name);
                Assert.That(target.GetProperty(name).TryGetInt32(out _), Is.True, name);
            }
        }

        private static void AssertJsonBooleanFields(JsonElement target, params string[] names)
        {
            foreach (string name in names)
            {
                Assert.That(target.GetProperty(name).ValueKind,
                    Is.AnyOf(JsonValueKind.True, JsonValueKind.False), name);
            }
        }

        private static void AssertJsonKinds(
            JsonElement target, JsonValueKind kind, params string[] names)
        {
            foreach (string name in names)
                Assert.That(target.GetProperty(name).ValueKind, Is.EqualTo(kind), name);
        }

        private static void AssertStringArray(JsonElement value, params string[] expected)
        {
            Assert.That(value.ValueKind, Is.EqualTo(JsonValueKind.Array));
            Assert.That(value.EnumerateArray().Select(item => item.GetString()).ToArray(),
                Is.EqualTo(expected));
        }

        private static void AssertReferencesAreInRange(JsonElement view)
        {
            JsonElement observation = view.GetProperty("observation");
            var lengths = new Dictionary<string, int>(StringComparer.Ordinal)
            {
                ["cells"] = observation.GetProperty("cells").GetArrayLength(),
                ["units"] = observation.GetProperty("units").GetArrayLength(),
                ["templates"] = observation.GetProperty("templates").GetArrayLength(),
                ["capability_definitions"] = observation.GetProperty("capability_definitions").GetArrayLength(),
                ["capability_allocations"] = observation.GetProperty("capability_allocations").GetArrayLength(),
                ["rules"] = observation.GetProperty("rules").GetArrayLength(),
                ["memory_records"] = observation.GetProperty("memory").GetArrayLength(),
                ["relations"] = observation.GetProperty("relations").GetArrayLength(),
                ["candidates"] = view.GetProperty("candidates").GetArrayLength(),
            };

            foreach (JsonElement row in observation.GetProperty("units").EnumerateArray())
                AssertReference(row.GetProperty("cell"), lengths);
            foreach (JsonElement row in observation.GetProperty("capability_allocations").EnumerateArray())
            {
                AssertReference(row.GetProperty("owner"), lengths);
                AssertReference(row.GetProperty("definition"), lengths);
            }
            foreach (JsonElement row in observation.GetProperty("memory").EnumerateArray())
                AssertReference(row.GetProperty("cell"), lengths);
            foreach (JsonElement row in observation.GetProperty("relations").EnumerateArray())
            {
                AssertReference(row.GetProperty("source"), lengths);
                AssertReference(row.GetProperty("target"), lengths);
            }
            foreach (JsonElement row in view.GetProperty("candidates").EnumerateArray())
            {
                AssertReference(row.GetProperty("actor"), lengths);
                AssertReference(row.GetProperty("target"), lengths);
                AssertReference(row.GetProperty("template"), lengths);
                AssertReference(row.GetProperty("cell"), lengths);
                JsonElement projection = row.GetProperty("projection");
                AssertReference(projection.GetProperty("source_cell"), lengths);
                AssertReference(projection.GetProperty("destination_cell"), lengths);
                AssertReference(projection.GetProperty("template"), lengths);
                AssertReference(projection.GetProperty("target"), lengths);
            }
        }

        private static void AssertReference(
            JsonElement reference, IReadOnlyDictionary<string, int> lengths)
        {
            if (reference.ValueKind == JsonValueKind.Null) return;
            AssertProperties(reference, "table", "row");
            string table = reference.GetProperty("table").GetString()!;
            int row = reference.GetProperty("row").GetInt32();
            Assert.That(lengths.ContainsKey(table), Is.True, table);
            Assert.That(row, Is.GreaterThanOrEqualTo(0));
            Assert.That(row, Is.LessThan(lengths[table]), $"{table}[{row}]");
        }

        private sealed class TacticalV3ServerProcess : IDisposable
        {
            private readonly Process _process;

            private static string ServerDll => Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));

            private TacticalV3ServerProcess(string? workingDirectory, params string[] args)
            {
                Assert.That(File.Exists(ServerDll), Is.True,
                    $"GymServer was not built at {ServerDll}");
                var start = new ProcessStartInfo("dotnet")
                {
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                if (workingDirectory != null)
                    start.WorkingDirectory = workingDirectory;
                start.ArgumentList.Add(ServerDll);
                foreach (string argument in args) start.ArgumentList.Add(argument);
                _process = Process.Start(start)!;
            }

            public static TacticalV3ServerProcess Start(string scenario) =>
                new TacticalV3ServerProcess(null,
                    "--environment", MlContract.TacticalV3Version,
                    "--scenario-file", scenario);

            public static TacticalV3ServerProcess StartInWorkingDirectory(
                string scenario, string workingDirectory) =>
                new TacticalV3ServerProcess(workingDirectory,
                    "--environment", MlContract.TacticalV3Version,
                    "--scenario-file", scenario);

            public static TacticalV3ServerProcess StartEnvironment(string environment) =>
                new TacticalV3ServerProcess(null, "--environment", environment);

            public static string RejectStartup(params string[] args)
            {
                Assert.That(File.Exists(ServerDll), Is.True,
                    $"GymServer was not built at {ServerDll}");
                var start = new ProcessStartInfo("dotnet")
                {
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                };
                start.ArgumentList.Add(ServerDll);
                foreach (string argument in args) start.ArgumentList.Add(argument);
                using Process process = Process.Start(start)!;
                try
                {
                    Assert.That(process.WaitForExit(10000), Is.True,
                        "GymServer should reject tactical-v3 startup before reading commands");
                    string output = process.StandardOutput.ReadToEnd();
                    string error = process.StandardError.ReadToEnd();
                    Assert.That(process.ExitCode, Is.Not.EqualTo(0));
                    Assert.That(output, Is.Empty,
                        "startup rejection must not emit a command response");
                    return error;
                }
                finally
                {
                    if (!process.HasExited) process.Kill(entireProcessTree: true);
                }
            }

            public JsonElement Request(string request)
            {
                using JsonDocument response = JsonDocument.Parse(RequestRaw(request));
                return response.RootElement.Clone();
            }

            public string RequestRaw(string request)
            {
                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();
                var pending = _process.StandardOutput.ReadLineAsync();
                if (!pending.Wait(TimeSpan.FromSeconds(10)))
                    Assert.Fail("GymServer did not reply to the tactical-v3 request");
                string? line = pending.Result;
                if (line == null)
                    Assert.Fail("GymServer exited without a reply: " +
                        _process.StandardError.ReadToEnd());
                return line!;
            }

            public string Reject(string request)
            {
                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();
                Assert.That(_process.WaitForExit(10000), Is.True,
                    "GymServer did not reject the tactical-v3 request");
                string output = _process.StandardOutput.ReadToEnd();
                string error = _process.StandardError.ReadToEnd();
                Assert.That(_process.ExitCode, Is.Not.EqualTo(0));
                Assert.That(output, Is.Empty,
                    "an invalid selection must not emit a successor view");
                return error;
            }

            public string RejectCommand(string request)
            {
                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();
                var pending = _process.StandardOutput.ReadLineAsync();
                if (pending.Wait(TimeSpan.FromSeconds(2)) && pending.Result != null)
                    Assert.Fail("malformed tactical-v3 command emitted a response (" +
                        pending.Result.Length + " bytes)");

                Assert.That(_process.WaitForExit(10000), Is.True,
                    "GymServer did not reject the malformed tactical-v3 command");
                Assert.That(pending.Wait(TimeSpan.FromSeconds(1)), Is.True,
                    "GymServer stdout did not close after rejecting the command");
                string? line = pending.Result;
                string output = _process.StandardOutput.ReadToEnd();
                string error = _process.StandardError.ReadToEnd();
                Assert.Multiple(() =>
                {
                    Assert.That(_process.ExitCode, Is.Not.EqualTo(0));
                    Assert.That(line, Is.Null,
                        "a malformed command must not emit a response line");
                    Assert.That(output, Is.Empty,
                        "a malformed command must not emit trailing output");
                });
                return error;
            }

            public void CloseAndAssert()
            {
                _process.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                _process.StandardInput.Flush();
                Assert.That(_process.WaitForExit(5000), Is.True,
                    "GymServer did not exit after close");
                string output = _process.StandardOutput.ReadToEnd();
                string error = _process.StandardError.ReadToEnd();
                Assert.Multiple(() =>
                {
                    Assert.That(_process.ExitCode, Is.Zero);
                    Assert.That(output, Is.Empty, "close must not emit a response");
                    Assert.That(error, Is.Empty, "normal close must not emit an error");
                });
            }

            public void Dispose()
            {
                if (_process.HasExited)
                {
                    _process.Dispose();
                    return;
                }

                _process.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                _process.StandardInput.Flush();
                if (!_process.WaitForExit(5000)) _process.Kill(entireProcessTree: true);
                _process.Dispose();
            }
        }

        private static void AssertEveryRow(JsonElement rows, params string[] names)
        {
            Assert.That(rows.ValueKind, Is.EqualTo(JsonValueKind.Array));
            Assert.That(rows.GetArrayLength(), Is.GreaterThan(0), string.Join(",", names));
            foreach (JsonElement row in rows.EnumerateArray()) AssertProperties(row, names);
        }

        private static void AssertProperties(JsonElement value, params string[] expected)
        {
            string[] actual = value.EnumerateObject()
                .Select(property => property.Name)
                .OrderBy(name => name, StringComparer.Ordinal)
                .ToArray();
            Assert.That(actual, Is.EqualTo(expected.OrderBy(name => name, StringComparer.Ordinal)));
        }

        private static object Property(object target, string name) =>
            target.GetType().GetProperty(name, BindingFlags.Instance | BindingFlags.Public)!
                .GetValue(target)!;

        private static object First(object sequence) => ((IEnumerable)sequence).Cast<object>().First();

        private static void SetAutoProperty<T>(object target, string propertyName, T value)
        {
            FieldInfo field = target.GetType().GetField(
                $"<{propertyName}>k__BackingField", BindingFlags.Instance | BindingFlags.NonPublic)!;
            field.SetValue(target, value);
        }
    }
}
