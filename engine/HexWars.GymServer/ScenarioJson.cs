using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using HexWars.Engine.Rl;

namespace HexWars.GymServer
{
    internal static class ScenarioJson
    {
        private static readonly JsonSerializerOptions Options = new JsonSerializerOptions
        {
            PropertyNameCaseInsensitive = false,
            ReadCommentHandling = JsonCommentHandling.Disallow,
            UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        };

        public static TrainingScenario Load(string path)
        {
            ScenarioWire? wire;
            try
            {
                wire = JsonSerializer.Deserialize<ScenarioWire>(File.ReadAllText(path), Options);
            }
            catch (JsonException exception)
            {
                throw new InvalidDataException("invalid scenario JSON: " + exception.Message, exception);
            }
            catch (IOException exception)
            {
                throw new InvalidDataException($"could not read scenario file '{path}': {exception.Message}", exception);
            }

            var errors = new List<string>();
            if (wire == null)
            {
                errors.Add("scenario is required");
            }
            else
            {
                ValidateRequiredFields(wire, errors);
            }

            if (errors.Count > 0) throw new InvalidDataException(string.Join("; ", errors));

            TrainingScenario scenario = Map(wire!);
            errors.AddRange(scenario.Validate());
            if (errors.Count > 0) throw new InvalidDataException(string.Join("; ", errors));

            return scenario;
        }

        private static void ValidateRequiredFields(ScenarioWire wire, List<string> errors)
        {
            Require(wire.SchemaVersion, "schema_version", errors);
            RequireText(wire.Id, "id", errors);
            RequireText(wire.Name, "name", errors);
            RequireText(wire.Environment, "environment", errors);
            Require(wire.Board, "board section", errors);
            Require(wire.Rules, "rules section", errors);
            Require(wire.Episode, "episode section", errors);
            Require(wire.Reward, "reward section", errors);

            if (wire.Board != null)
            {
                Require(wire.Board.Width, "board.width", errors);
                Require(wire.Board.Height, "board.height", errors);
                Require(wire.Board.MaxElevation, "board.max_elevation", errors);
                Require(wire.Board.ZoneDepth, "board.zone_depth", errors);
                Require(wire.Board.FlatChance, "board.flat_chance", errors);
                Require(wire.Board.PlainsWeight, "board.plains_weight", errors);
                Require(wire.Board.ForestWeight, "board.forest_weight", errors);
                Require(wire.Board.RoughWeight, "board.rough_weight", errors);
                Require(wire.Board.WaterWeight, "board.water_weight", errors);
            }

            if (wire.Rules != null)
            {
                Require(wire.Rules.ActionsPerTurn, "rules.actions_per_turn", errors);
                Require(wire.Rules.RoundCap, "rules.round_cap", errors);
                Require(wire.Rules.StartingPoints, "rules.starting_points", errors);
                Require(wire.Rules.FogOfWar, "rules.fog_of_war", errors);
                Require(wire.Rules.BiomesEnabled, "rules.biomes_enabled", errors);
                Require(wire.Rules.BountyRate, "rules.bounty_rate", errors);
                Require(wire.Rules.DeployCostMultiplier, "rules.deploy_cost_multiplier", errors);
                Require(wire.Rules.GeneratorCost, "rules.generator_cost", errors);
                Require(wire.Rules.GeneratorOutput, "rules.generator_output", errors);
                Require(wire.Rules.GeneratorHealth, "rules.generator_health", errors);
            }

            if (wire.Episode != null) Require(wire.Episode.MaxSteps, "episode.max_steps", errors);

            if (wire.Environment == MlContract.CurrentVersion)
            {
                TacticalRewardWire? reward = DeserializeReward<TacticalRewardWire>(wire.Reward, errors);
                if (reward != null)
                {
                    Require(reward.ShapeScale, "reward.shape_scale", errors);
                    Require(reward.StepPenalty, "reward.step_penalty", errors);
                    Require(reward.ClosingWeight, "reward.closing_weight", errors);
                    Require(reward.DrawCreditWeight, "reward.draw_credit_weight", errors);
                    Require(reward.PointsWeight, "reward.points_weight", errors);
                }
                RequireAbsent(wire.Adaptive, "adaptive section is not valid for tactical-v1", errors);
                RequireAbsent(wire.TacticalV2, "tactical-v2 section is not valid for tactical-v1", errors);
            }
            else if (wire.Environment == MlContract.AdaptiveVersion)
            {
                AdaptiveRewardWire? reward = DeserializeReward<AdaptiveRewardWire>(wire.Reward, errors);
                if (reward != null)
                {
                    Require(reward.IntermediateDecisionPenalty, "reward.intermediate_decision_penalty", errors);
                    Require(reward.DeploymentCompletionBonus, "reward.deployment_completion_bonus", errors);
                }

                Require(wire.Adaptive, "adaptive section", errors);
                if (wire.Adaptive != null)
                {
                    Require(wire.Adaptive.StartingUnitCount, "adaptive.starting_unit_count", errors);
                    Require(wire.Adaptive.StartingArmyBudget, "adaptive.starting_army_budget", errors);
                    Require(wire.Adaptive.MaxDesignPointCost, "adaptive.max_design_point_cost", errors);
                }
                RequireAbsent(wire.TacticalV2, "tactical-v2 section is not valid for adaptive-v1", errors);
            }
            else if (wire.Environment == MlContract.TacticalV2Version)
            {
                TacticalRewardWire? reward = DeserializeReward<TacticalRewardWire>(wire.Reward, errors);
                if (reward != null)
                {
                    Require(reward.ShapeScale, "reward.shape_scale", errors);
                    Require(reward.StepPenalty, "reward.step_penalty", errors);
                    Require(reward.ClosingWeight, "reward.closing_weight", errors);
                    Require(reward.DrawCreditWeight, "reward.draw_credit_weight", errors);
                    Require(reward.PointsWeight, "reward.points_weight", errors);
                }
                RequireAbsent(wire.Adaptive, "adaptive section is not valid for tactical-v2", errors);

                Require(wire.TacticalV2, "tactical_v2 section", errors);
                if (wire.TacticalV2 != null)
                {
                    Require(wire.TacticalV2.StartingUnitCount, "tactical_v2.starting_unit_count", errors);
                    Require(wire.TacticalV2.MaxControllableUnits, "tactical_v2.max_controllable_units", errors);
                    RequireText(wire.TacticalV2.PlacementPolicy, "tactical_v2.placement_policy", errors);

                    if (wire.TacticalV2.Templates == null || wire.TacticalV2.Templates.Count == 0)
                    {
                        errors.Add("tactical_v2.templates is required");
                    }
                    else
                    {
                        for (int i = 0; i < wire.TacticalV2.Templates.Count; i++)
                            ValidateTacticalV2TemplateWire(wire.TacticalV2.Templates[i], $"tactical_v2.templates[{i}]", errors);
                    }
                }
            }
        }

        private static void ValidateTacticalV2TemplateWire(
            TacticalV2TemplateWire template, string prefix, List<string> errors)
        {
            RequireText(template.Id, prefix + ".id", errors);
            RequireText(template.Name, prefix + ".name", errors);
            Require(template.Stats, prefix + ".stats", errors);
            if (template.Stats == null) return;

            Require(template.Stats.Health, prefix + ".stats.health", errors);
            Require(template.Stats.Damage, prefix + ".stats.damage", errors);
            Require(template.Stats.Defense, prefix + ".stats.defense", errors);
            Require(template.Stats.Movement, prefix + ".stats.movement", errors);
            Require(template.Stats.VerticalMovement, prefix + ".stats.vertical_movement", errors);
            Require(template.Stats.Range, prefix + ".stats.range", errors);
            Require(template.Stats.RangeArc, prefix + ".stats.range_arc", errors);
            Require(template.Stats.Vision, prefix + ".stats.vision", errors);
            Require(template.Stats.VisionArc, prefix + ".stats.vision_arc", errors);
        }

        private static T? DeserializeReward<T>(JsonElement? reward, List<string> errors) where T : class
        {
            if (reward == null) return null;
            try
            {
                return reward.Value.Deserialize<T>(Options);
            }
            catch (JsonException exception)
            {
                errors.Add("reward is invalid: " + exception.Message);
                return null;
            }
        }

        private static TrainingScenario Map(ScenarioWire wire)
        {
            var scenario = new TrainingScenario
            {
                SchemaVersion = wire.SchemaVersion!.Value,
                Id = wire.Id!,
                Name = wire.Name!,
                Environment = wire.Environment!,
                Board = new TrainingBoardConfig
                {
                    Width = wire.Board!.Width!.Value,
                    Height = wire.Board.Height!.Value,
                    MaxElevation = wire.Board.MaxElevation!.Value,
                    ZoneDepth = wire.Board.ZoneDepth!.Value,
                    FlatChance = wire.Board.FlatChance!.Value,
                    PlainsWeight = wire.Board.PlainsWeight!.Value,
                    ForestWeight = wire.Board.ForestWeight!.Value,
                    RoughWeight = wire.Board.RoughWeight!.Value,
                    WaterWeight = wire.Board.WaterWeight!.Value,
                },
                Rules = new TrainingRuleConfig
                {
                    ActionsPerTurn = wire.Rules!.ActionsPerTurn!.Value,
                    RoundCap = wire.Rules.RoundCap!.Value,
                    StartingPoints = wire.Rules.StartingPoints!.Value,
                    FogOfWar = wire.Rules.FogOfWar!.Value,
                    BiomesEnabled = wire.Rules.BiomesEnabled!.Value,
                    BountyRate = wire.Rules.BountyRate!.Value,
                    DeployCostMultiplier = wire.Rules.DeployCostMultiplier!.Value,
                    GeneratorCost = wire.Rules.GeneratorCost!.Value,
                    GeneratorOutput = wire.Rules.GeneratorOutput!.Value,
                    GeneratorHealth = wire.Rules.GeneratorHealth!.Value,
                },
                Episode = new TrainingEpisodeConfig { MaxSteps = wire.Episode!.MaxSteps!.Value },
            };

            if (wire.Environment == MlContract.CurrentVersion)
            {
                TacticalRewardWire reward = wire.Reward!.Value.Deserialize<TacticalRewardWire>(Options)!;
                scenario.TacticalReward = new TacticalRewardConfig
                {
                    ShapeScale = reward.ShapeScale!.Value,
                    StepPenalty = reward.StepPenalty!.Value,
                    ClosingWeight = reward.ClosingWeight!.Value,
                    DrawCreditWeight = reward.DrawCreditWeight!.Value,
                    PointsWeight = reward.PointsWeight!.Value,
                };
            }
            else if (wire.Environment == MlContract.AdaptiveVersion)
            {
                AdaptiveRewardWire reward = wire.Reward!.Value.Deserialize<AdaptiveRewardWire>(Options)!;
                scenario.AdaptiveReward = new AdaptiveRewardConfig
                {
                    IntermediateDecisionPenalty = reward.IntermediateDecisionPenalty!.Value,
                    DeploymentCompletionBonus = reward.DeploymentCompletionBonus!.Value,
                };
                scenario.Adaptive = new TrainingAdaptiveConfig
                {
                    StartingUnitCount = wire.Adaptive!.StartingUnitCount!.Value,
                    StartingArmyBudget = wire.Adaptive.StartingArmyBudget!.Value,
                    MaxDesignPointCost = wire.Adaptive.MaxDesignPointCost!.Value,
                };
            }
            else if (wire.Environment == MlContract.TacticalV2Version)
            {
                TacticalRewardWire reward = wire.Reward!.Value.Deserialize<TacticalRewardWire>(Options)!;
                scenario.TacticalReward = new TacticalRewardConfig
                {
                    ShapeScale = reward.ShapeScale!.Value,
                    StepPenalty = reward.StepPenalty!.Value,
                    ClosingWeight = reward.ClosingWeight!.Value,
                    DrawCreditWeight = reward.DrawCreditWeight!.Value,
                    PointsWeight = reward.PointsWeight!.Value,
                };
                scenario.TacticalV2 = new TrainingTacticalV2Config
                {
                    StartingUnitCount = wire.TacticalV2!.StartingUnitCount!.Value,
                    MaxControllableUnits = wire.TacticalV2.MaxControllableUnits!.Value,
                    PlacementPolicy = wire.TacticalV2.PlacementPolicy!,
                    Templates = wire.TacticalV2.Templates!.Select(MapTacticalV2Template).ToList(),
                };
            }

            return scenario;
        }

        private static TrainingUnitTemplateConfig MapTacticalV2Template(TacticalV2TemplateWire template) =>
            new TrainingUnitTemplateConfig
            {
                Id = template.Id!,
                Name = template.Name!,
                Health = template.Stats!.Health!.Value,
                Damage = template.Stats.Damage!.Value,
                Defense = template.Stats.Defense!.Value,
                Movement = template.Stats.Movement!.Value,
                VerticalMovement = template.Stats.VerticalMovement!.Value,
                Range = template.Stats.Range!.Value,
                RangeArc = template.Stats.RangeArc!.Value,
                Vision = template.Stats.Vision!.Value,
                VisionArc = template.Stats.VisionArc!.Value,
            };

        private static void Require(object? value, string field, List<string> errors)
        {
            if (value == null) errors.Add(field + " is required");
        }

        private static void RequireText(string? value, string field, List<string> errors)
        {
            if (string.IsNullOrWhiteSpace(value)) errors.Add(field + " is required");
        }

        private static void RequireAbsent(object? value, string error, List<string> errors)
        {
            if (value != null) errors.Add(error);
        }

        private sealed class ScenarioWire
        {
            [JsonPropertyName("schema_version")] public int? SchemaVersion { get; set; }
            [JsonPropertyName("id")] public string? Id { get; set; }
            [JsonPropertyName("name")] public string? Name { get; set; }
            [JsonPropertyName("environment")] public string? Environment { get; set; }
            [JsonPropertyName("board")] public BoardWire? Board { get; set; }
            [JsonPropertyName("rules")] public RulesWire? Rules { get; set; }
            [JsonPropertyName("episode")] public EpisodeWire? Episode { get; set; }
            [JsonPropertyName("reward")] public JsonElement? Reward { get; set; }
            [JsonPropertyName("adaptive")] public AdaptiveWire? Adaptive { get; set; }
            [JsonPropertyName("tactical_v2")] public TacticalV2Wire? TacticalV2 { get; set; }
        }

        private sealed class BoardWire
        {
            [JsonPropertyName("width")] public int? Width { get; set; }
            [JsonPropertyName("height")] public int? Height { get; set; }
            [JsonPropertyName("max_elevation")] public int? MaxElevation { get; set; }
            [JsonPropertyName("zone_depth")] public int? ZoneDepth { get; set; }
            [JsonPropertyName("flat_chance")] public double? FlatChance { get; set; }
            [JsonPropertyName("plains_weight")] public int? PlainsWeight { get; set; }
            [JsonPropertyName("forest_weight")] public int? ForestWeight { get; set; }
            [JsonPropertyName("rough_weight")] public int? RoughWeight { get; set; }
            [JsonPropertyName("water_weight")] public int? WaterWeight { get; set; }
        }

        private sealed class RulesWire
        {
            [JsonPropertyName("actions_per_turn")] public int? ActionsPerTurn { get; set; }
            [JsonPropertyName("round_cap")] public int? RoundCap { get; set; }
            [JsonPropertyName("starting_points")] public int? StartingPoints { get; set; }
            [JsonPropertyName("fog_of_war")] public bool? FogOfWar { get; set; }
            [JsonPropertyName("biomes_enabled")] public bool? BiomesEnabled { get; set; }
            [JsonPropertyName("bounty_rate")] public double? BountyRate { get; set; }
            [JsonPropertyName("deploy_cost_multiplier")] public double? DeployCostMultiplier { get; set; }
            [JsonPropertyName("generator_cost")] public int? GeneratorCost { get; set; }
            [JsonPropertyName("generator_output")] public int? GeneratorOutput { get; set; }
            [JsonPropertyName("generator_health")] public int? GeneratorHealth { get; set; }
        }

        private sealed class EpisodeWire
        {
            [JsonPropertyName("max_steps")] public int? MaxSteps { get; set; }
        }

        private sealed class TacticalRewardWire
        {
            [JsonPropertyName("shape_scale")] public float? ShapeScale { get; set; }
            [JsonPropertyName("step_penalty")] public float? StepPenalty { get; set; }
            [JsonPropertyName("closing_weight")] public float? ClosingWeight { get; set; }
            [JsonPropertyName("draw_credit_weight")] public float? DrawCreditWeight { get; set; }
            [JsonPropertyName("points_weight")] public float? PointsWeight { get; set; }
        }

        private sealed class AdaptiveRewardWire
        {
            [JsonPropertyName("intermediate_decision_penalty")] public float? IntermediateDecisionPenalty { get; set; }
            [JsonPropertyName("deployment_completion_bonus")] public float? DeploymentCompletionBonus { get; set; }
        }

        private sealed class AdaptiveWire
        {
            [JsonPropertyName("starting_unit_count")] public int? StartingUnitCount { get; set; }
            [JsonPropertyName("starting_army_budget")] public int? StartingArmyBudget { get; set; }
            [JsonPropertyName("max_design_point_cost")] public int? MaxDesignPointCost { get; set; }
        }

        private sealed class TacticalV2Wire
        {
            [JsonPropertyName("starting_unit_count")] public int? StartingUnitCount { get; set; }
            [JsonPropertyName("max_controllable_units")] public int? MaxControllableUnits { get; set; }
            [JsonPropertyName("placement_policy")] public string? PlacementPolicy { get; set; }
            [JsonPropertyName("templates")] public List<TacticalV2TemplateWire>? Templates { get; set; }
        }

        private sealed class TacticalV2TemplateWire
        {
            [JsonPropertyName("id")] public string? Id { get; set; }
            [JsonPropertyName("name")] public string? Name { get; set; }
            [JsonPropertyName("stats")] public TacticalV2StatsWire? Stats { get; set; }
        }

        private sealed class TacticalV2StatsWire
        {
            [JsonPropertyName("health")] public int? Health { get; set; }
            [JsonPropertyName("damage")] public int? Damage { get; set; }
            [JsonPropertyName("defense")] public int? Defense { get; set; }
            [JsonPropertyName("movement")] public int? Movement { get; set; }
            [JsonPropertyName("vertical_movement")] public int? VerticalMovement { get; set; }
            [JsonPropertyName("range")] public int? Range { get; set; }
            [JsonPropertyName("range_arc")] public int? RangeArc { get; set; }
            [JsonPropertyName("vision")] public int? Vision { get; set; }
            [JsonPropertyName("vision_arc")] public int? VisionArc { get; set; }
        }
    }
}
