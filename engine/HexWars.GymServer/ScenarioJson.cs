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
                RequireAbsent(wire.TacticalV3, "tactical-v3 section is not valid for tactical-v1", errors);
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
                RequireAbsent(wire.TacticalV3, "tactical-v3 section is not valid for adaptive-v1", errors);
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
                RequireAbsent(wire.TacticalV3, "tactical-v3 section is not valid for tactical-v2", errors);

                Require(wire.TacticalV2, "tactical_v2 section", errors);
                if (wire.TacticalV2 != null)
                {
                    Require(wire.TacticalV2.StartingUnitCount, "tactical_v2.starting_unit_count", errors);
                    Require(wire.TacticalV2.MaxControllableUnits, "tactical_v2.max_controllable_units", errors);
                    RequireText(wire.TacticalV2.PlacementPolicy, "tactical_v2.placement_policy", errors);
                    if (wire.TacticalV2.PlacementPolicy == "profiled-seeded-v1")
                    {
                        Require(wire.TacticalV2.StartProfiles, "tactical_v2.start_profiles", errors);
                        Require(wire.TacticalV2.StartDistribution, "tactical_v2.start_distribution", errors);
                        if (wire.TacticalV2.StartProfiles != null)
                            for (int i = 0; i < wire.TacticalV2.StartProfiles.Count; i++)
                            {
                                TacticalV2Wire.TacticalV2StartProfileWire profile = wire.TacticalV2.StartProfiles[i];
                                RequireText(profile.Id, $"tactical_v2.start_profiles[{i}].id", errors);
                                Require(profile.LearnerUnitCount, $"tactical_v2.start_profiles[{i}].learner_units", errors);
                                Require(profile.OpponentUnitCount, $"tactical_v2.start_profiles[{i}].opponent_units", errors);
                                RequireText(profile.Separation, $"tactical_v2.start_profiles[{i}].separation", errors);
                            }
                        if (wire.TacticalV2.StartDistribution != null)
                            for (int i = 0; i < wire.TacticalV2.StartDistribution.Count; i++)
                                ValidateTacticalV2StartWeightWire(wire.TacticalV2.StartDistribution[i], i, errors);

                    }
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
            else if (wire.Environment == MlContract.TacticalV3Version)
            {
                TacticalV3RewardWire? reward = DeserializeReward<TacticalV3RewardWire>(wire.Reward, errors);
                if (reward != null)
                {
                    Require(reward.TerminalWin, "reward.terminal_win", errors);
                    Require(reward.TerminalNonWin, "reward.terminal_non_win", errors);
                    Require(reward.MaterialAdjustmentBound, "reward.material_adjustment_bound", errors);
                    Require(reward.TimePressureBound, "reward.time_pressure_bound", errors);
                    Require(reward.PointsWeight, "reward.points_weight", errors);
                }

                RequireAbsent(wire.Adaptive, "adaptive section is not valid for tactical-v3", errors);
                RequireAbsent(wire.TacticalV2, "tactical-v2 section is not valid for tactical-v3", errors);
                Require(wire.TacticalV3, "tactical_v3 section", errors);
                if (wire.TacticalV3 != null)
                {
                    TacticalV3Wire source = wire.TacticalV3;
                    Require(source.StartingUnitCount, "tactical_v3.starting_unit_count", errors);
                    Require(source.MaxControllableUnits, "tactical_v3.max_controllable_units", errors);
                    RequireText(source.PlacementPolicy, "tactical_v3.placement_policy", errors);
                    Require(source.StartProfiles, "tactical_v3.start_profiles", errors);
                    Require(source.StartDistribution, "tactical_v3.start_distribution", errors);
                    if (source.StartProfiles != null)
                        for (int i = 0; i < source.StartProfiles.Count; i++)
                        {
                            TacticalV2Wire.TacticalV2StartProfileWire profile = source.StartProfiles[i];
                            RequireText(profile.Id, $"tactical_v3.start_profiles[{i}].id", errors);
                            Require(profile.LearnerUnitCount, $"tactical_v3.start_profiles[{i}].learner_units", errors);
                            Require(profile.OpponentUnitCount, $"tactical_v3.start_profiles[{i}].opponent_units", errors);
                            RequireText(profile.Separation, $"tactical_v3.start_profiles[{i}].separation", errors);
                        }
                    if (source.StartDistribution != null)
                        for (int i = 0; i < source.StartDistribution.Count; i++)
                        {
                            TacticalV2Wire.TacticalV2StartWeightWire weight = source.StartDistribution[i];
                            RequireText(weight.ProfileId, $"tactical_v3.start_distribution[{i}].profile_id", errors);
                            Require(weight.BasisPoints, $"tactical_v3.start_distribution[{i}].basis_points", errors);
                        }

                    if (source.Templates == null || source.Templates.Count == 0)
                    {
                        errors.Add("tactical_v3.templates is required");
                    }
                    else
                    {
                        for (int i = 0; i < source.Templates.Count; i++)
                            ValidateTacticalV2TemplateWire(
                                source.Templates[i], $"tactical_v3.templates[{i}]", errors);
                    }

                    Require(source.Capacity, "tactical_v3.capacity", errors);
                    if (source.Capacity != null)
                    {
                        Require(source.Capacity.MaxCells, "tactical_v3.capacity.max_cells", errors);
                        Require(source.Capacity.MaxUnits, "tactical_v3.capacity.max_units", errors);
                        Require(source.Capacity.MaxTemplates, "tactical_v3.capacity.max_templates", errors);
                        Require(source.Capacity.MaxCapabilityDefinitions,
                            "tactical_v3.capacity.max_capability_definitions", errors);
                        Require(source.Capacity.MaxCapabilityAllocations,
                            "tactical_v3.capacity.max_capability_allocations", errors);
                        Require(source.Capacity.MaxRules, "tactical_v3.capacity.max_rules", errors);
                        Require(source.Capacity.MaxMemoryRecords,
                            "tactical_v3.capacity.max_memory_records", errors);
                        Require(source.Capacity.MaxRelations, "tactical_v3.capacity.max_relations", errors);
                        Require(source.Capacity.MaxCandidates, "tactical_v3.capacity.max_candidates", errors);
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
                    StartProfiles = wire.TacticalV2.StartProfiles?.Select(MapTacticalV2StartProfile).ToList() ?? new List<TacticalV2StartProfile>(),
                    StartDistribution = wire.TacticalV2.StartDistribution?.Select(MapTacticalV2StartWeight).ToList() ?? new List<TacticalV2StartWeight>(),
                    Templates = wire.TacticalV2.Templates!.Select(MapTacticalV2Template).ToList(),
                };
            }
            else if (wire.Environment == MlContract.TacticalV3Version)
            {
                TacticalV3RewardWire reward = wire.Reward!.Value.Deserialize<TacticalV3RewardWire>(Options)!;
                TacticalV3Wire source = wire.TacticalV3!;
                scenario.TacticalV3Reward = new TrainingTacticalV3RewardConfig
                {
                    TerminalWin = reward.TerminalWin!.Value,
                    TerminalNonWin = reward.TerminalNonWin!.Value,
                    MaterialAdjustmentBound = reward.MaterialAdjustmentBound!.Value,
                    TimePressureBound = reward.TimePressureBound!.Value,
                    PointsWeight = reward.PointsWeight!.Value,
                };
                scenario.TacticalV3 = new TrainingTacticalV3Config
                {
                    StartingUnitCount = source.StartingUnitCount!.Value,
                    MaxControllableUnits = source.MaxControllableUnits!.Value,
                    PlacementPolicy = source.PlacementPolicy!,
                    StartProfiles = source.StartProfiles!.Select(MapTacticalV2StartProfile).ToList(),
                    StartDistribution = source.StartDistribution!.Select(MapTacticalV2StartWeight).ToList(),
                    Templates = source.Templates!.Select(MapTacticalV2Template).ToList(),
                    Capacity = new TrainingTacticalV3CapacityConfig
                    {
                        MaxCells = source.Capacity!.MaxCells!.Value,
                        MaxUnits = source.Capacity.MaxUnits!.Value,
                        MaxTemplates = source.Capacity.MaxTemplates!.Value,
                        MaxCapabilityDefinitions = source.Capacity.MaxCapabilityDefinitions!.Value,
                        MaxCapabilityAllocations = source.Capacity.MaxCapabilityAllocations!.Value,
                        MaxRules = source.Capacity.MaxRules!.Value,
                        MaxMemoryRecords = source.Capacity.MaxMemoryRecords!.Value,
                        MaxRelations = source.Capacity.MaxRelations!.Value,
                        MaxCandidates = source.Capacity.MaxCandidates!.Value,
                    },
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
        private static TacticalV2StartProfile MapTacticalV2StartProfile(TacticalV2Wire.TacticalV2StartProfileWire profile) =>
            new TacticalV2StartProfile(profile.Id!, profile.LearnerUnitCount!.Value,
                profile.OpponentUnitCount!.Value, profile.Separation!);

        private static TacticalV2StartWeight MapTacticalV2StartWeight(TacticalV2Wire.TacticalV2StartWeightWire weight) =>
            new TacticalV2StartWeight(weight.ProfileId!, weight.BasisPoints!.Value);

        private static void ValidateTacticalV2StartWeightWire(
            TacticalV2Wire.TacticalV2StartWeightWire weight, int index, List<string> errors)
        {
            RequireText(weight.ProfileId, $"tactical_v2.start_distribution[{index}].profile_id", errors);
            Require(weight.BasisPoints, $"tactical_v2.start_distribution[{index}].basis_points", errors);
        }

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
            [JsonPropertyName("tactical_v3")] public TacticalV3Wire? TacticalV3 { get; set; }
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
        private sealed class TacticalV3RewardWire
        {
            [JsonPropertyName("terminal_win")] public float? TerminalWin { get; set; }
            [JsonPropertyName("terminal_non_win")] public float? TerminalNonWin { get; set; }
            [JsonPropertyName("material_adjustment_bound")] public float? MaterialAdjustmentBound { get; set; }
            [JsonPropertyName("time_pressure_bound")] public float? TimePressureBound { get; set; }
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
            [JsonPropertyName("start_profiles")] public List<TacticalV2StartProfileWire>? StartProfiles { get; set; }
        public sealed class TacticalV2StartProfileWire
        {
            [JsonPropertyName("id")] public string? Id { get; set; }
            [JsonPropertyName("learner_units")] public int? LearnerUnitCount { get; set; }
            [JsonPropertyName("opponent_units")] public int? OpponentUnitCount { get; set; }
            [JsonPropertyName("separation")] public string? Separation { get; set; }
        }

        public sealed class TacticalV2StartWeightWire
        {
            [JsonPropertyName("profile_id")] public string? ProfileId { get; set; }
            [JsonPropertyName("basis_points")] public int? BasisPoints { get; set; }
        }
            [JsonPropertyName("start_distribution")] public List<TacticalV2StartWeightWire>? StartDistribution { get; set; }
            [JsonPropertyName("templates")] public List<TacticalV2TemplateWire>? Templates { get; set; }
        }
        private sealed class TacticalV3Wire
        {
            [JsonPropertyName("starting_unit_count")] public int? StartingUnitCount { get; set; }
            [JsonPropertyName("max_controllable_units")] public int? MaxControllableUnits { get; set; }
            [JsonPropertyName("placement_policy")] public string? PlacementPolicy { get; set; }
            [JsonPropertyName("start_profiles")]
            public List<TacticalV2Wire.TacticalV2StartProfileWire>? StartProfiles { get; set; }
            [JsonPropertyName("start_distribution")]
            public List<TacticalV2Wire.TacticalV2StartWeightWire>? StartDistribution { get; set; }
            [JsonPropertyName("templates")] public List<TacticalV2TemplateWire>? Templates { get; set; }
            [JsonPropertyName("capacity")] public TacticalV3CapacityWire? Capacity { get; set; }
        }

        private sealed class TacticalV3CapacityWire
        {
            [JsonPropertyName("max_cells")] public int? MaxCells { get; set; }
            [JsonPropertyName("max_units")] public int? MaxUnits { get; set; }
            [JsonPropertyName("max_templates")] public int? MaxTemplates { get; set; }
            [JsonPropertyName("max_capability_definitions")] public int? MaxCapabilityDefinitions { get; set; }
            [JsonPropertyName("max_capability_allocations")] public int? MaxCapabilityAllocations { get; set; }
            [JsonPropertyName("max_rules")] public int? MaxRules { get; set; }
            [JsonPropertyName("max_memory_records")] public int? MaxMemoryRecords { get; set; }
            [JsonPropertyName("max_relations")] public int? MaxRelations { get; set; }
            [JsonPropertyName("max_candidates")] public int? MaxCandidates { get; set; }
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
