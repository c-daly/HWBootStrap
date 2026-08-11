using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.CompilerServices;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class TacticalV3GymServerTests
    {
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
            IReadOnlyList<TacticalV3RelationToken>? relations = null)
        {
            TacticalV3Observation original = view.Decision.Observation;
            SetAutoProperty(view.Decision, nameof(TacticalV3DecisionFrame.Observation),
                new TacticalV3Observation(
                    cells ?? original.Cells, original.Units, original.Templates,
                    original.CapabilityDefinitions, original.CapabilityAllocations,
                    original.Rules, original.Memory, relations ?? original.Relations));
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

        private static object InvokeWire(string methodName, params object[] arguments)
        {
            string gymServerDll = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));
            Assembly assembly = Assembly.LoadFrom(gymServerDll);
            Type wireType = assembly.GetType("HexWars.GymServer.TacticalV3Wire", throwOnError: true)!;
            MethodInfo method = wireType.GetMethod(
                methodName, BindingFlags.Public | BindingFlags.Static)!;
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
