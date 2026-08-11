using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.ExceptionServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    [TestFixture]
    public class TacticalV3ScenarioTests
    {
        [Test]
        public void ValidTacticalV3Document_LoadsAndBuilds()
        {
            TrainingScenario scenario = LoadTemporary(ValidTacticalV3Json().ToJsonString());

            Assert.That(scenario.BuildTacticalV3().Validate(), Is.Empty);
        }

        [Test]
        public void CheckedInScenario_BuildsStageOneStructuredConfig()
        {
            TrainingScenario scenario = LoadCheckedIn("annihilation-structured-imitation-v1.json");
            TacticalV3Config config = scenario.BuildTacticalV3();

            Assert.Multiple(() =>
            {
                Assert.That(scenario.SchemaVersion, Is.EqualTo(1));
                Assert.That(scenario.Id, Is.EqualTo("annihilation-structured-imitation-v1"));
                Assert.That(scenario.Name, Is.EqualTo("Annihilation structured imitation 70/30"));
                Assert.That(scenario.Environment, Is.EqualTo("tactical-v3"));
                Assert.That(config.Validate(), Is.Empty);
                Assert.That((
                    scenario.Board.Width, scenario.Board.Height, scenario.Board.MaxElevation,
                    scenario.Board.ZoneDepth, scenario.Board.FlatChance, scenario.Board.PlainsWeight,
                    scenario.Board.ForestWeight, scenario.Board.RoughWeight, scenario.Board.WaterWeight),
                    Is.EqualTo((13, 9, 4, 3, 0.6, 70, 15, 10, 5)));
                Assert.That((
                    config.Match.BoardGen.Width, config.Match.BoardGen.Height,
                    config.Match.BoardGen.MaxElevation, config.Match.BoardGen.ZoneDepth,
                    config.Match.BoardGen.FlatChance, config.Match.BoardGen.PlainsWeight,
                    config.Match.BoardGen.ForestWeight, config.Match.BoardGen.RoughWeight,
                    config.Match.BoardGen.WaterWeight), Is.EqualTo((13, 9, 4, 3, 0.6, 70, 15, 10, 5)));
                Assert.That((
                    scenario.Rules.ActionsPerTurn, scenario.Rules.RoundCap,
                    scenario.Rules.StartingPoints, scenario.Rules.FogOfWar,
                    scenario.Rules.BiomesEnabled, scenario.Rules.BountyRate,
                    scenario.Rules.DeployCostMultiplier, scenario.Rules.GeneratorCost,
                    scenario.Rules.GeneratorOutput, scenario.Rules.GeneratorHealth),
                    Is.EqualTo((0, 100, 12, false, false, 0.5, 1.0, 2, 1, 3)));
                Assert.That((
                    config.Match.Game.StartingPoints, config.Match.Game.RoundCap,
                    config.Match.Game.FogOfWar, config.Match.Game.BiomesEnabled,
                    config.Match.Game.BountyRate, config.Match.Game.DeployCostMultiplier,
                    config.Match.Game.GeneratorCost, config.Match.Game.GeneratorOutput,
                    config.Match.Game.GeneratorHealth, config.Match.Game.GeneratorsEnabled),
                    Is.EqualTo((12, 100, false, false, 0.5, 1.0, 2, 1, 3, false)));
                Assert.That(config.Match.Game.TurnPolicy, Is.TypeOf<AllUnitsPolicy>());
                Assert.That(config.Match.Game.WinConditions, Is.EqualTo(WinBy.Annihilation));
                Assert.That(config.Match.MaxSteps, Is.EqualTo(808));
                Assert.That(config.Match.Templates.Select(template => (
                    template.Id, template.Template.Name,
                    template.Template.Stats.Health, template.Template.Stats.Damage,
                    template.Template.Stats.Defense, template.Template.Stats.Movement,
                    template.Template.Stats.VerticalMovement, template.Template.Stats.Range,
                    template.Template.Stats.RangeArc, template.Template.Stats.Vision,
                    template.Template.Stats.VisionArc)), Is.EqualTo(new[]
                {
                    ("brute-85597320", "Brute", 7, 2, 2, 3, 2, 1, 1, 2, 1),
                    ("striker-0d7b6999", "Striker", 2, 6, 0, 3, 2, 2, 1, 3, 1),
                    ("sniper-d065c02a", "Sniper", 2, 2, 0, 2, 2, 6, 1, 4, 1),
                    ("artillery-27c01722", "Artillery", 3, 6, 0, 0, 0, 5, 2, 2, 1),
                    ("scout-d3503dfa", "Scout", 2, 0, 0, 4, 3, 0, 0, 7, 2),
                }));
                Assert.That(config.Match.PlacementPolicy, Is.EqualTo("profiled-seeded-v1"));
                Assert.That(config.Match.StartingUnitCount, Is.EqualTo(3));
                Assert.That(config.Match.MaxControllableUnits, Is.EqualTo(3));
                Assert.That(config.Match.StartProfiles.Select(profile => (
                    profile.Id, profile.LearnerUnitCount, profile.OpponentUnitCount,
                    profile.Separation)), Is.EqualTo(new[]
                {
                    ("standard-3v3", 3, 3, "legacy-mirrored"),
                    ("conversion-3v1-near", 3, 1, "near"), ("conversion-3v1-medium", 3, 1, "medium"),
                    ("conversion-3v1-far", 3, 1, "far"), ("conversion-2v1-near", 2, 1, "near"),
                    ("conversion-2v1-medium", 2, 1, "medium"), ("conversion-2v1-far", 2, 1, "far"),
                    ("conversion-1v1-near", 1, 1, "near"), ("conversion-1v1-medium", 1, 1, "medium"),
                    ("conversion-1v1-far", 1, 1, "far"),
                }));
                Assert.That(config.Match.StartDistribution.Weights.Select(weight =>
                    (weight.ProfileId, weight.BasisPoints)), Is.EqualTo(new[]
                {
                    ("standard-3v3", 7000),
                    ("conversion-3v1-near", 500), ("conversion-3v1-medium", 0), ("conversion-3v1-far", 500),
                    ("conversion-2v1-near", 500), ("conversion-2v1-medium", 0), ("conversion-2v1-far", 500),
                    ("conversion-1v1-near", 500), ("conversion-1v1-medium", 0), ("conversion-1v1-far", 500),
                }));
                Assert.That(config.Match.ShapeScale, Is.Zero);
                Assert.That(config.Match.StepPenalty, Is.Zero);
                Assert.That(config.Match.ClosingWeight, Is.Zero);
                Assert.That(config.Match.DrawCreditWeight, Is.Zero);
                Assert.That(config.Match.PointsWeight, Is.EqualTo(0.5f));
                Assert.That(config.Reward.TerminalWin, Is.EqualTo(1f));
                Assert.That(config.Reward.TerminalNonWin, Is.EqualTo(-1f));
                Assert.That(config.Reward.MaterialAdjustmentBound, Is.EqualTo(0.2f));
                Assert.That(config.Reward.TimePressureBound, Is.EqualTo(0.05f));
                Assert.That(config.Reward.PointsWeight, Is.EqualTo(0.5f));
                Assert.That(CapacityValues(config.Capacity), Is.EqualTo(
                    new[] { 512, 64, 32, 128, 2048, 128, 64, 65536, 32768 }));
            });
        }

        [TestCase("missing_tactical_v3")]
        [TestCase("wrong_environment")]
        [TestCase("adaptive")]
        [TestCase("tactical_v2")]
        public void TacticalV3Scenario_RejectsMissingOrForeignEnvironmentSections(string mutation)
        {
            JsonObject scenario = ValidTacticalV3Json();
            switch (mutation)
            {
                case "missing_tactical_v3":
                    scenario.Remove("tactical_v3");
                    break;
                case "wrong_environment":
                    scenario["environment"] = "tactical-v2";
                    break;
                case "adaptive":
                    scenario["adaptive"] = new JsonObject
                    {
                        ["starting_unit_count"] = 6,
                        ["starting_army_budget"] = 132,
                        ["max_design_point_cost"] = 24,
                    };
                    break;
                case "tactical_v2":
                    JsonObject tacticalV2 = (JsonObject)scenario["tactical_v3"]!.DeepClone();
                    tacticalV2.Remove("capacity");
                    scenario["tactical_v2"] = tacticalV2;
                    break;
                default:
                    throw new AssertionException("unknown mutation " + mutation);
            }

            AssertRejected(scenario);
        }

        [TestCase("tactical")]
        [TestCase("adaptive")]
        public void TacticalV3Scenario_RejectsLegacyRewardShapes(string rewardKind)
        {
            JsonObject scenario = ValidTacticalV3Json();
            scenario["reward"] = rewardKind == "tactical"
                ? new JsonObject
                {
                    ["shape_scale"] = 0.01,
                    ["step_penalty"] = 0.005,
                    ["closing_weight"] = 0.02,
                    ["draw_credit_weight"] = 0.25,
                    ["points_weight"] = 0.5,
                }
                : new JsonObject
                {
                    ["intermediate_decision_penalty"] = 0.001,
                    ["deployment_completion_bonus"] = 0.0,
                };

            AssertRejected(scenario);
        }

        [TestCase("root")]
        [TestCase("reward")]
        [TestCase("tactical_v3")]
        [TestCase("capacity")]
        [TestCase("template")]
        [TestCase("profile")]
        [TestCase("distribution")]
        public void TacticalV3Scenario_RejectsUnknownFieldsAtEveryOwnedNestingLevel(string level)
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            switch (level)
            {
                case "root": scenario["future_root"] = true; break;
                case "reward": ((JsonObject)scenario["reward"]!)["future_reward"] = true; break;
                case "tactical_v3": tacticalV3["future_tactical_v3"] = true; break;
                case "capacity": ((JsonObject)tacticalV3["capacity"]!)["future_capacity"] = true; break;
                case "template": ((JsonObject)((JsonArray)tacticalV3["templates"]!)[0]!)["future_template"] = true; break;
                case "profile": ((JsonObject)((JsonArray)tacticalV3["start_profiles"]!)[0]!)["future_profile"] = true; break;
                case "distribution": ((JsonObject)((JsonArray)tacticalV3["start_distribution"]!)[0]!)["future_distribution"] = true; break;
                default: throw new AssertionException("unknown level " + level);
            }

            AssertRejected(scenario);
        }

        [TestCase("root")]
        [TestCase("nested")]
        public void TacticalV3Scenario_RejectsDuplicateJsonProperties(string level)
        {
            string json = ValidTacticalV3Json().ToJsonString();
            if (level == "root")
                json = json.Replace(
                    "\"schema_version\":1", "\"schema_version\":1,\"schema_version\":1");
            else
                json = json.Replace(
                    "\"points_weight\":0.5", "\"points_weight\":0.5,\"points_weight\":0.5");

            Assert.Throws<InvalidDataException>(() => LoadTemporary(json));
        }

        [TestCase("template")]
        [TestCase("profile")]
        [TestCase("distribution")]
        public void TacticalV3Scenario_JsonRejectsNullCollectionElements(string collection)
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            string property = collection == "template" ? "templates" :
                collection == "profile" ? "start_profiles" : "start_distribution";
            ((JsonArray)tacticalV3[property]!)[0] = null;

            AssertRejected(scenario);
        }

        [TestCase("config")]
        [TestCase("reward")]
        public void TacticalV3Scenario_DirectApiRejectsNullSections(string section)
        {
            TrainingScenario scenario =
                TrainingScenario.CreateStandard(MlContract.TacticalV3Version);
            if (section == "config") scenario.TacticalV3 = null!;
            else scenario.TacticalV3Reward = null!;

            Assert.That(scenario.Validate(),
                Has.Some.Contains("requires a tactical-v3"));
            Assert.Throws<ArgumentException>(() => scenario.BuildTacticalV3());
        }

        [TestCase("templates")]
        [TestCase("start_profiles")]
        [TestCase("start_distribution")]
        public void TacticalV3Scenario_DirectApiRejectsNullCollections(string collection)
        {
            TrainingScenario scenario = LoadCheckedIn("annihilation-structured-imitation-v1.json");
            if (collection == "templates") scenario.TacticalV3.Templates = null!;
            else if (collection == "start_profiles") scenario.TacticalV3.StartProfiles = null!;
            else scenario.TacticalV3.StartDistribution = null!;

            Assert.That(scenario.Validate(), Has.Some.Contains("required"));
            Assert.Throws<ArgumentException>(() => scenario.BuildTacticalV3());
        }

        [TestCase("template")]
        [TestCase("profile")]
        [TestCase("distribution")]
        public void TacticalV3Scenario_DirectApiRejectsNullCollectionElements(string collection)
        {
            TrainingScenario scenario = LoadCheckedIn("annihilation-structured-imitation-v1.json");
            if (collection == "template") scenario.TacticalV3.Templates[0] = null!;
            else if (collection == "profile") scenario.TacticalV3.StartProfiles[0] = null!;
            else scenario.TacticalV3.StartDistribution[0] = null!;

            Assert.That(scenario.Validate(), Has.Some.Contains("null"));
            Assert.Throws<ArgumentException>(() => scenario.BuildTacticalV3());
        }

        [Test]
        public void TacticalV3Scenario_JsonRejectsSemanticallyInvalidProfileCounts()
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            ((JsonObject)((JsonArray)tacticalV3["start_profiles"]!)[0]!)["learner_units"] = 0;

            AssertRejected(scenario);
        }


        [Test]
        public void TacticalV3Scenario_RejectsOverflowedStepBudget()
        {
            JsonObject scenario = ValidTacticalV3Json();
            ((JsonObject)scenario["rules"]!)["round_cap"] = 268435456;
            ((JsonObject)scenario["episode"]!)["max_steps"] = 1;

            AssertRejected(scenario);
        }

        [Test]
        public void TacticalV3Scenario_DirectApiRejectsOverflowingProfilesOnSymmetricPlacement()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard(MlContract.TacticalV3Version);
            scenario.TacticalV3.StartProfiles.Add(
                new TacticalV2StartProfile("overflow", int.MaxValue, 1, "near"));
            scenario.TacticalV3.StartDistribution.Add(
                new TacticalV2StartWeight("overflow", 10000));

            Assert.That(scenario.Validate(),
                Has.Some.Contains("symmetric-random-v1").And.Contains("start profile"));
            Assert.Throws<ArgumentException>(() => scenario.BuildTacticalV3());
        }

        [Test]
        public void TacticalV3Scenario_JsonRejectsProfileDataOnSymmetricPlacement()
        {
            JsonObject scenario = ValidTacticalV3Json();
            ((JsonObject)scenario["tactical_v3"]!)["placement_policy"] = "symmetric-random-v1";

            AssertRejected(scenario);
        }


        [Test]
        public void TacticalV3Scenario_AcceptsExactStructuralCapacityForSixUnits()
        {
            // 616 adjacency + 6 occupancy + 9*(6 unit + 10 template rows) = 766 relations.
            // 1 end + 5*27 deploy + 6*117 move + 6*5 ordered attack = 868 candidates.
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            JsonObject capacity = (JsonObject)tacticalV3["capacity"]!;
            capacity["max_units"] = 6;
            capacity["max_relations"] = 766;
            capacity["max_candidates"] = 868;

            Assert.That(LoadTemporary(scenario.ToJsonString()).BuildTacticalV3().Validate(), Is.Empty);
        }

        [TestCase("max_relations", 1345)]
        [TestCase("max_candidates", 11655)]
        public void TacticalV3Scenario_RejectsOneBelowCheckedInStructuralCapacity(
            string field, int value)
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            ((JsonObject)tacticalV3["capacity"]!)[field] = value;

            AssertRejected(scenario);
        }

        [TestCase("max_relations", 1346)]
        [TestCase("max_candidates", 11656)]
        public void TacticalV3Scenario_AcceptsExactCheckedInStructuralCapacity(
            string field, int value)
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            ((JsonObject)tacticalV3["capacity"]!)[field] = value;

            Assert.That(LoadTemporary(scenario.ToJsonString()).BuildTacticalV3().Validate(), Is.Empty);
        }


        [TestCase("fog")]
        [TestCase("generators")]
        [TestCase("overlapping_zones")]
        [TestCase("insufficient_steps")]
        [TestCase("invalid_reward")]
        [TestCase("undersized_cells")]
        [TestCase("undersized_templates")]
        public void TacticalV3Scenario_RejectsInvalidStageOneConfigurationBeforeUse(string mutation)
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v3"]!;
            switch (mutation)
            {
                case "fog": ((JsonObject)scenario["rules"]!)["fog_of_war"] = true; break;
                case "generators": ((JsonObject)scenario["rules"]!)["generators_enabled"] = true; break;
                case "overlapping_zones": ((JsonObject)scenario["board"]!)["zone_depth"] = 7; break;
                case "insufficient_steps": ((JsonObject)scenario["episode"]!)["max_steps"] = 799; break;
                case "invalid_reward": ((JsonObject)scenario["reward"]!)["terminal_non_win"] = 0.0; break;
                case "undersized_cells": ((JsonObject)tacticalV3["capacity"]!)["max_cells"] = 116; break;
                case "undersized_templates": ((JsonObject)tacticalV3["capacity"]!)["max_templates"] = 4; break;
                default: throw new AssertionException("unknown mutation " + mutation);
            }

            AssertRejected(scenario);
        }

        [Test]
        public void TacticalV3Scenario_JsonRejectsZeroHealthTemplate()
        {
            JsonObject scenario = ValidTacticalV3Json();
            JsonArray templates = (JsonArray)((JsonObject)scenario["tactical_v3"]!)["templates"]!;
            JsonObject stats = (JsonObject)((JsonObject)templates[0]!)["stats"]!;
            stats["health"] = 0;

            AssertRejected(scenario);
        }

        [Test]
        public void LegacyCheckedInScenarios_StillParseUnchanged()
        {
            var expected = new List<string>();
            var actual = new List<string>();
            using JsonDocument library = JsonDocument.Parse(File.ReadAllText(
                ConfigPath("training-game-templates.json")));
            foreach (JsonElement template in library.RootElement.GetProperty("templates").EnumerateArray())
            {
                expected.Add(template.GetProperty("id").GetString()!);
                actual.Add(LoadTemporary(template.GetRawText()).Id);
            }

            expected.Add("annihilation-imitation-v1");
            actual.Add(LoadCheckedIn("annihilation-imitation-v1.json").Id);
            Assert.That(actual, Is.EqualTo(expected));
        }


        private static TrainingScenario LoadCheckedIn(string fileName) => Load(ConfigPath(fileName));

        private static TrainingScenario LoadTemporary(string content)
        {
            string path = Path.Combine(TestContext.CurrentContext.WorkDirectory,
                "tactical-v3-scenario-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(path, content);
            try
            {
                return Load(path);
            }
            finally
            {
                File.Delete(path);
            }
        }

        private static TrainingScenario Load(string path)
        {
            string gymServerDll = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));
            Assembly assembly = Assembly.LoadFrom(gymServerDll);
            Type parserType = assembly.GetType("HexWars.GymServer.ScenarioJson", throwOnError: true)!;
            MethodInfo load = parserType.GetMethod("Load", BindingFlags.Public | BindingFlags.Static)!;
            try
            {
                return (TrainingScenario)load.Invoke(null, new object[] { path })!;
            }
            catch (TargetInvocationException exception) when (exception.InnerException != null)
            {
                ExceptionDispatchInfo.Capture(exception.InnerException).Throw();
                throw;
            }
        }

        private static void AssertRejected(JsonObject scenario)
        {
            Assert.Throws<InvalidDataException>(() => LoadTemporary(scenario.ToJsonString()));
        }

        private static JsonObject ValidTacticalV3Json()
        {
            return (JsonObject)JsonNode.Parse(
                """
                {
                  "schema_version": 1,
                  "id": "valid-tactical-v3",
                  "name": "Valid Tactical V3",
                  "environment": "tactical-v3",
                  "board": {"width":13,"height":9,"max_elevation":4,"zone_depth":3,"flat_chance":0.6,"plains_weight":70,"forest_weight":15,"rough_weight":10,"water_weight":5},
                  "rules": {"actions_per_turn":0,"round_cap":100,"starting_points":12,"fog_of_war":false,"biomes_enabled":false,"bounty_rate":0.5,"deploy_cost_multiplier":1.0,"generator_cost":2,"generator_output":1,"generator_health":3},
                  "episode": {"max_steps":808},
                  "reward": {"terminal_win":1.0,"terminal_non_win":-1.0,"material_adjustment_bound":0.2,"time_pressure_bound":0.05,"points_weight":0.5},
                  "tactical_v3": {
                    "starting_unit_count": 3,
                    "max_controllable_units": 3,
                    "placement_policy": "profiled-seeded-v1",
                    "capacity": {"max_cells":512,"max_units":64,"max_templates":32,"max_capability_definitions":128,"max_capability_allocations":2048,"max_rules":128,"max_memory_records":64,"max_relations":65536,"max_candidates":32768},
                    "templates": [
                      {"id":"brute-85597320","name":"Brute","stats":{"health":7,"damage":2,"defense":2,"movement":3,"vertical_movement":2,"range":1,"range_arc":1,"vision":2,"vision_arc":1}},
                      {"id":"striker-0d7b6999","name":"Striker","stats":{"health":2,"damage":6,"defense":0,"movement":3,"vertical_movement":2,"range":2,"range_arc":1,"vision":3,"vision_arc":1}},
                      {"id":"sniper-d065c02a","name":"Sniper","stats":{"health":2,"damage":2,"defense":0,"movement":2,"vertical_movement":2,"range":6,"range_arc":1,"vision":4,"vision_arc":1}},
                      {"id":"artillery-27c01722","name":"Artillery","stats":{"health":3,"damage":6,"defense":0,"movement":0,"vertical_movement":0,"range":5,"range_arc":2,"vision":2,"vision_arc":1}},
                      {"id":"scout-d3503dfa","name":"Scout","stats":{"health":2,"damage":0,"defense":0,"movement":4,"vertical_movement":3,"range":0,"range_arc":0,"vision":7,"vision_arc":2}}
                    ],
                    "start_profiles": [
                      {"id":"standard-3v3","learner_units":3,"opponent_units":3,"separation":"legacy-mirrored"},
                      {"id":"conversion-3v1-near","learner_units":3,"opponent_units":1,"separation":"near"},
                      {"id":"conversion-3v1-medium","learner_units":3,"opponent_units":1,"separation":"medium"},
                      {"id":"conversion-3v1-far","learner_units":3,"opponent_units":1,"separation":"far"},
                      {"id":"conversion-2v1-near","learner_units":2,"opponent_units":1,"separation":"near"},
                      {"id":"conversion-2v1-medium","learner_units":2,"opponent_units":1,"separation":"medium"},
                      {"id":"conversion-2v1-far","learner_units":2,"opponent_units":1,"separation":"far"},
                      {"id":"conversion-1v1-near","learner_units":1,"opponent_units":1,"separation":"near"},
                      {"id":"conversion-1v1-medium","learner_units":1,"opponent_units":1,"separation":"medium"},
                      {"id":"conversion-1v1-far","learner_units":1,"opponent_units":1,"separation":"far"}
                    ],
                    "start_distribution": [
                      {"profile_id":"standard-3v3","basis_points":7000},
                      {"profile_id":"conversion-3v1-near","basis_points":500},
                      {"profile_id":"conversion-3v1-medium","basis_points":0},
                      {"profile_id":"conversion-3v1-far","basis_points":500},
                      {"profile_id":"conversion-2v1-near","basis_points":500},
                      {"profile_id":"conversion-2v1-medium","basis_points":0},
                      {"profile_id":"conversion-2v1-far","basis_points":500},
                      {"profile_id":"conversion-1v1-near","basis_points":500},
                      {"profile_id":"conversion-1v1-medium","basis_points":0},
                      {"profile_id":"conversion-1v1-far","basis_points":500}
                    ]
                  }
                }
                """)!;
        }

        private static int[] CapacityValues(TacticalV3CapacityProfile capacity) => new[]
        {
            capacity.MaxCells,
            capacity.MaxUnits,
            capacity.MaxTemplates,
            capacity.MaxCapabilityDefinitions,
            capacity.MaxCapabilityAllocations,
            capacity.MaxRules,
            capacity.MaxMemoryRecords,
            capacity.MaxRelations,
            capacity.MaxCandidates,
        };

        private static string ConfigPath(string fileName) => Path.GetFullPath(Path.Combine(
            TestContext.CurrentContext.TestDirectory,
            "..", "..", "..", "..", "..", "python", "config", fileName));
    }
}
