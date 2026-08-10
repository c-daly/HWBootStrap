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
                Assert.That(scenario.Id, Is.EqualTo("annihilation-structured-imitation-v1"));
                Assert.That(scenario.Environment, Is.EqualTo("tactical-v3"));
                Assert.That(config.Validate(), Is.Empty);
                Assert.That(config.Match.BoardGen.Width, Is.EqualTo(13));
                Assert.That(config.Match.BoardGen.Height, Is.EqualTo(9));
                Assert.That(config.Match.MaxSteps, Is.EqualTo(808));
                Assert.That(config.Match.Game.FogOfWar, Is.False);
                Assert.That(config.Match.Game.GeneratorsEnabled, Is.False);
                Assert.That(config.Match.Templates.Select(template => template.Id), Is.EqualTo(new[]
                {
                    "brute-85597320", "striker-0d7b6999", "sniper-d065c02a",
                    "artillery-27c01722", "scout-d3503dfa",
                }));
                Assert.That(config.Match.PlacementPolicy, Is.EqualTo("profiled-seeded-v1"));
                Assert.That(config.Match.StartProfiles.Select(profile => profile.Id), Is.EqualTo(new[]
                {
                    "standard-3v3",
                    "conversion-3v1-near", "conversion-3v1-medium", "conversion-3v1-far",
                    "conversion-2v1-near", "conversion-2v1-medium", "conversion-2v1-far",
                    "conversion-1v1-near", "conversion-1v1-medium", "conversion-1v1-far",
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
            JsonObject scenario = (JsonObject)JsonNode.Parse(
                File.ReadAllText(ConfigPath("annihilation-imitation-v1.json")))!;
            scenario["id"] = "valid-tactical-v3";
            scenario["name"] = "Valid Tactical V3";
            scenario["environment"] = "tactical-v3";
            scenario["reward"] = new JsonObject
            {
                ["terminal_win"] = 1.0,
                ["terminal_non_win"] = -1.0,
                ["material_adjustment_bound"] = 0.2,
                ["time_pressure_bound"] = 0.05,
                ["points_weight"] = 0.5,
            };
            JsonObject tacticalV3 = (JsonObject)scenario["tactical_v2"]!.DeepClone();
            tacticalV3["capacity"] = new JsonObject
            {
                ["max_cells"] = 512,
                ["max_units"] = 64,
                ["max_templates"] = 32,
                ["max_capability_definitions"] = 128,
                ["max_capability_allocations"] = 2048,
                ["max_rules"] = 128,
                ["max_memory_records"] = 64,
                ["max_relations"] = 65536,
                ["max_candidates"] = 32768,
            };
            scenario.Remove("tactical_v2");
            scenario["tactical_v3"] = tacticalV3;
            return scenario;
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
