using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class TacticalV3GymServerTests
    {
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
            Assert.That(json.GetProperty("match").GetRawText(),
                Is.EqualTo(JsonSerializer.Serialize(contract.Match)));
            Assert.That(json.GetProperty("encoding").GetRawText(),
                Is.EqualTo(JsonSerializer.Serialize(contract.Encoding)));
            Assert.That(json.GetProperty("capacity").GetRawText(),
                Is.EqualTo(JsonSerializer.Serialize(contract.Capacity)));

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

        private static TacticalV3View ViewWithMemory(int seed)
        {
            var env = new TacticalV3DuelEnv(TacticalV3Fixtures.ProfiledConfig());
            TacticalV3View view = env.Reset(
                seed, null, null, "standard-3v3", PlayerId.Player0, PlayerId.Player0);
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
