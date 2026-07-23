using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Globalization;
using System.Linq;
using System.Security.Cryptography;
using System.Text;

namespace HexWars.Engine.Rl
{
    public enum MlEnvironmentKind
    {
        Tactical,
        Duel,
        AdaptiveTactical,
        AdaptiveDuel,
    }

    /// <summary>
    /// The versioned meaning of the tactical observation, action and reward spaces. The JSON hashed here is
    /// deliberately assembled in a fixed order rather than reflected from configuration objects, so its SHA-256
    /// is stable across runtimes and changes whenever a known semantic input changes.
    /// </summary>
    public sealed class MlContract
    {
        public const string CurrentVersion = "tactical-v1";
        public const string AdaptiveVersion = "adaptive-v1";

        private static readonly IReadOnlyList<string> AdaptivePhases = Array.AsReadOnly(new[]
        {
            "deployment_root", "deployment_template", "deployment_cell", "deployment_placed_unit",
            "deployment_move_cell", "gameplay_root", "gameplay_unit", "gameplay_unit_command",
            "gameplay_move_cell", "gameplay_attack_cell", "design_slot", "design_stat", "design_value",
            "design_confirm",
        });

        private MlContract(
            string version,
            int observationSize,
            int actionSize,
            IReadOnlyDictionary<string, object> board,
            IReadOnlyList<string> roster,
            IReadOnlyDictionary<string, object> reward,
            IReadOnlyDictionary<string, object> semantics,
            string environmentKind,
            string contractHash,
            string encodingHash)
        {
            Version = version;
            ObservationSize = observationSize;
            ActionSize = actionSize;
            Board = board;
            Roster = roster;
            Reward = reward;
            Semantics = semantics;
            EnvironmentKind = environmentKind;
            ContractHash = contractHash;
            EncodingHash = encodingHash;
        }

        public string Version { get; }
        public string ContractHash { get; }
        public string EncodingHash { get; }
        public int ObservationSize { get; }
        public int ActionSize { get; }
        public IReadOnlyDictionary<string, object> Board { get; }
        public IReadOnlyList<string> Roster { get; }
        public IReadOnlyDictionary<string, object> Reward { get; }
        public IReadOnlyDictionary<string, object> Semantics { get; }
        public string EnvironmentKind { get; }

        public static MlContract Create(EnvConfig config, MlEnvironmentKind environmentKind = MlEnvironmentKind.Tactical)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            TacticalLayout.ValidateDimensions(config);
            var layout = new TacticalLayout(config);
            var kind = EnvironmentKindName(environmentKind);
            var maxSteps = EffectiveMaxSteps(config, environmentKind);
            var board = BoardValues(config, kind, maxSteps);
            var roster = RosterValues(config.Roster);
            var reward = RewardValues(config);
            var canonical = CanonicalJson(config, layout, roster, kind, maxSteps);
            return new MlContract(
                CurrentVersion,
                layout.ObservationLength,
                layout.ActionCount,
                board,
                roster,
                reward,
                new Dictionary<string, object>(),
                kind,
                Sha256(canonical),
                Sha256(CanonicalEncodingJson(CurrentVersion, layout.ObservationLength,
                    layout.ActionCount, board, roster, new Dictionary<string, object>())));
        }

        public static MlContract CreateAdaptive(
            AdaptiveEnvConfig config,
            MlEnvironmentKind environmentKind = MlEnvironmentKind.AdaptiveTactical)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));
            var designErrors = config.ValidateDesignRuleConsistency();
            if (designErrors.Count > 0)
                throw new ArgumentException(string.Join("; ", designErrors), nameof(config));

            string kind = AdaptiveEnvironmentKindName(environmentKind);
            int maxSteps = AdaptiveEffectiveMaxSteps(config, environmentKind);
            int cellCount = checked(config.BoardGen.Width * config.BoardGen.Height);
            var actionRegions = AdaptiveActionRegions(config, cellCount);
            int actionSize = RegionEnd(actionRegions, "value");
            var observationChannels = AdaptiveObservationChannels(config);
            int globals = 7 + AdaptivePhases.Count + 4 + config.Templates.Count * 11;
            int observationSize = checked(observationChannels.Count * cellCount + globals);
            var board = BoardValues(
                new EnvConfig { BoardGen = config.BoardGen, Game = config.Game },
                kind,
                maxSteps);
            var roster = AdaptiveRosterValues(config.Templates);
            var reward = AdaptiveRewardValues(config);
            var semantics = AdaptiveSemantics(
                config,
                board,
                actionRegions,
                observationChannels,
                actionSize,
                observationSize,
                maxSteps,
                kind);
            var canonical = CanonicalAdaptiveJson(
                semantics,
                roster,
                reward,
                kind,
                actionSize,
                observationSize);

            return new MlContract(
                AdaptiveVersion,
                observationSize,
                actionSize,
                board,
                roster,
                reward,
                semantics,
                kind,
                Sha256(canonical),
                Sha256(CanonicalEncodingJson(AdaptiveVersion, observationSize,
                    actionSize, board, roster, semantics)));
        }

        private static string AdaptiveEnvironmentKindName(MlEnvironmentKind kind) => kind switch
        {
            MlEnvironmentKind.AdaptiveTactical => "adaptive_tactical",
            MlEnvironmentKind.AdaptiveDuel => "adaptive_duel",
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static int AdaptiveEffectiveMaxSteps(AdaptiveEnvConfig config, MlEnvironmentKind kind) => kind switch
        {
            MlEnvironmentKind.AdaptiveTactical => config.MaxSteps,
            MlEnvironmentKind.AdaptiveDuel => checked(config.MaxSteps * 2),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static string EnvironmentKindName(MlEnvironmentKind kind) => kind switch
        {
            MlEnvironmentKind.Tactical => "tactical",
            MlEnvironmentKind.Duel => "duel",
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static int EffectiveMaxSteps(EnvConfig config, MlEnvironmentKind kind) => kind switch
        {
            MlEnvironmentKind.Tactical => config.MaxSteps,
            MlEnvironmentKind.Duel => checked(config.MaxSteps * 2),
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static IReadOnlyDictionary<string, object> AdaptiveActionRegions(
            AdaptiveEnvConfig config,
            int cellCount)
        {
            int commandOffset = 0;
            int unitOffset = commandOffset + 12;
            int templateOffset = unitOffset + config.MaxControllableUnits;
            int cellOffset = templateOffset + config.Templates.Count;
            int statOffset = cellOffset + cellCount;
            int valueOffset = statOffset + Enum.GetValues(typeof(AdaptiveStat)).Length;
            int valueCount = config.StatValues.Count == 0 ? 0 : config.StatValues.Values.Max(values => values.Count);

            return ReadOnlyMap(new Dictionary<string, object>
            {
                ["command"] = Region(commandOffset, 12),
                ["unit"] = Region(unitOffset, config.MaxControllableUnits),
                ["template"] = Region(templateOffset, config.Templates.Count),
                ["cell"] = Region(cellOffset, cellCount),
                ["stat"] = Region(statOffset, Enum.GetValues(typeof(AdaptiveStat)).Length),
                ["value"] = Region(valueOffset, valueCount),
            });
        }

        private static IReadOnlyDictionary<string, object> Region(int offset, int count) =>
            ReadOnlyMap(new Dictionary<string, object>
            {
                ["offset"] = offset,
                ["count"] = count,
            });

        private static int RegionEnd(IReadOnlyDictionary<string, object> regions, string name)
        {
            var region = (IReadOnlyDictionary<string, object>)regions[name];
            return (int)region["offset"] + (int)region["count"];
        }

        private static IReadOnlyList<string> AdaptiveObservationChannels(AdaptiveEnvConfig config)
        {
            var channels = new List<string>
            {
                "elevation", "terrain_plains", "terrain_forest", "terrain_rough", "terrain_water",
                "deployment_zone_self", "current_visibility", "previously_seen",
            };
            for (int i = 0; i < config.Templates.Count; i++) channels.Add($"friendly_role_hp_{i}");
            for (int i = 0; i < config.Templates.Count; i++) channels.Add($"visible_enemy_role_hp_{i}");
            for (int i = 0; i < config.MaxControllableUnits; i++) channels.Add($"friendly_slot_occupancy_{i}");
            return Array.AsReadOnly(channels.ToArray());
        }

        private static IReadOnlyList<string> AdaptiveRosterValues(IReadOnlyList<UnitTemplate> templates)
        {
            var values = new string[templates.Count];
            for (int i = 0; i < templates.Count; i++)
                values[i] = templates[i].Name + ":" + RosterEntry(templates[i].Stats);
            return Array.AsReadOnly(values);
        }

        private static IReadOnlyDictionary<string, object> AdaptiveRewardValues(AdaptiveEnvConfig config) =>
            ReadOnlyMap(new Dictionary<string, object>
            {
                ["intermediate_decision_penalty"] = config.IntermediateDecisionPenalty,
                ["deployment_completion_bonus"] = config.DeploymentCompletionBonus,
                ["terminal_win"] = 1f,
                ["terminal_loss"] = -1f,
            });

        private static IReadOnlyDictionary<string, object> AdaptiveSemantics(
            AdaptiveEnvConfig config,
            IReadOnlyDictionary<string, object> board,
            IReadOnlyDictionary<string, object> actionRegions,
            IReadOnlyList<string> observationChannels,
            int actionSize,
            int observationSize,
            int maxSteps,
            string environmentKind)
        {
            var templates = new object[config.Templates.Count];
            for (int i = 0; i < config.Templates.Count; i++)
            {
                var template = config.Templates[i];
                templates[i] = ReadOnlyMap(new Dictionary<string, object>
                {
                    ["slot"] = i,
                    ["name"] = template.Name,
                    ["stats"] = Array.AsReadOnly(StatValues(template.Stats)),
                    ["cost"] = template.Stats.PointCost,
                    ["fixed"] = i < config.FixedTemplateCount,
                });
            }

            var statValues = new Dictionary<string, object>();
            foreach (AdaptiveStat stat in Enum.GetValues(typeof(AdaptiveStat)))
            {
                config.StatValues.TryGetValue(stat, out var values);
                statValues[AdaptiveStatName(stat)] = values ?? Array.AsReadOnly(Array.Empty<int>());
            }

            return ReadOnlyMap(new Dictionary<string, object>
            {
                ["adaptive"] = true,
                ["contract_version"] = AdaptiveVersion,
                ["environment_kind"] = environmentKind,
                ["fixed_template_count"] = config.FixedTemplateCount,
                ["custom_template_count"] = config.CustomTemplateCount,
                ["max_controllable_units"] = config.MaxControllableUnits,
                ["starting_unit_count"] = config.StartingUnitCount,
                ["starting_army_budget"] = config.StartingArmyBudget,
                ["max_design_point_cost"] = config.MaxDesignPointCost,
                ["intermediate_decision_penalty"] = config.IntermediateDecisionPenalty,
                ["deployment_completion_bonus"] = config.DeploymentCompletionBonus,
                ["effective_horizon"] = maxSteps,
                ["fog_rule"] = "hide_current_enemy_units_and_all_opponent_deployment_until_both_confirm;"
                    + "derive_action_masks_from_seat_visible_projection;"
                    + "authoritative_hidden_blocker_rejection_is_only_allowed_mask_rejection",
                ["templates"] = Array.AsReadOnly(templates),
                ["stat_values"] = ReadOnlyMap(statValues),
                ["phases"] = AdaptivePhases,
                ["action_regions"] = actionRegions,
                ["observation_channels"] = observationChannels,
                ["action_size"] = actionSize,
                ["observation_size"] = observationSize,
                ["board"] = board,
            });
        }

        private static int[] StatValues(UnitStats stats) => new[]
        {
            stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
            stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
        };

        private static string AdaptiveStatName(AdaptiveStat stat) => stat switch
        {
            AdaptiveStat.Health => "health",
            AdaptiveStat.Damage => "damage",
            AdaptiveStat.Defense => "defense",
            AdaptiveStat.Movement => "movement",
            AdaptiveStat.VerticalMovement => "vertical_movement",
            AdaptiveStat.Range => "range",
            AdaptiveStat.RangeArc => "range_arc",
            AdaptiveStat.Vision => "vision",
            AdaptiveStat.VisionArc => "vision_arc",
            _ => throw new ArgumentOutOfRangeException(nameof(stat)),
        };

        private static IReadOnlyDictionary<string, object> ReadOnlyMap(Dictionary<string, object> values) =>
            new ReadOnlyDictionary<string, object>(values);

        private static IReadOnlyDictionary<string, object> BoardValues(EnvConfig config, string environmentKind, int maxSteps)
        {
            var b = config.BoardGen;
            var g = config.Game;
            return new Dictionary<string, object>
            {
                ["width"] = b.Width,
                ["height"] = b.Height,
                ["max_elevation"] = b.MaxElevation,
                ["max_steps"] = maxSteps,
                ["zone_depth"] = b.ZoneDepth,
                ["flat_chance"] = b.FlatChance,
                ["plains_weight"] = b.PlainsWeight,
                ["forest_weight"] = b.ForestWeight,
                ["rough_weight"] = b.RoughWeight,
                ["water_weight"] = b.WaterWeight,
                ["plains"] = TerrainValues(g.Terrain(TerrainType.Plains)),
                ["forest"] = TerrainValues(g.Terrain(TerrainType.Forest)),
                ["rough"] = TerrainValues(g.Terrain(TerrainType.Rough)),
                ["water"] = TerrainValues(g.Terrain(TerrainType.Water)),
                ["biomes_enabled"] = g.BiomesEnabled,
                ["starting_points"] = g.StartingPoints,
                ["bounty_rate"] = g.BountyRate,
                ["generator_cost"] = g.GeneratorCost,
                ["generator_output"] = g.GeneratorOutput,
                ["generator_health"] = g.GeneratorHealth,
                ["damage_floor"] = g.DamageFloor,
                ["dmg_high_ground_bonus"] = g.DmgHighGroundBonus,
                ["range_high_ground_bonus"] = g.RangeHighGroundBonus,
                ["round_cap"] = g.RoundCap,
                ["design_fee"] = g.DesignFee,
                ["deploy_cost_multiplier"] = g.DeployCostMultiplier,
                ["turn_policy"] = g.TurnPolicy.GetType().FullName ?? g.TurnPolicy.GetType().Name,
                ["actions_per_turn"] = g.TurnPolicy.ActionsPerTurn ?? -1,
                ["win_conditions"] = (int)g.WinConditions,
                ["capture_cost"] = g.CaptureCost,
                ["economy_win_threshold"] = g.EconomyWinThreshold,
                ["score_kills"] = g.ScoreKills,
                ["score_points"] = g.ScorePoints,
                ["score_army"] = g.ScoreArmy,
                ["score_territory"] = g.ScoreTerritory,
                ["upkeep_factor"] = g.UpkeepFactor,
                ["capture_factor"] = g.CaptureFactor,
                ["build_factor"] = g.BuildFactor,
                ["territory_mode"] = g.TerritoryMode,
                ["claim_ends_turn"] = g.ClaimEndsTurn,
                ["build_anywhere"] = g.BuildAnywhere,
                ["territory_income"] = g.TerritoryIncome,
                ["generators_enabled"] = g.GeneratorsEnabled,
                ["point_decay"] = g.PointDecay,
                ["fog_of_war"] = g.FogOfWar,
                ["environment_kind"] = environmentKind,
            };
        }

        private static IReadOnlyList<string> RosterValues(IReadOnlyList<UnitStats> roster)
        {
            var result = new List<string>(roster.Count);
            for (int i = 0; i < roster.Count; i++) result.Add(RosterEntry(roster[i]));
            return result;
        }

        private static IReadOnlyDictionary<string, object> TerrainValues(TerrainDef terrain) =>
            new Dictionary<string, object>
            {
                ["move_cost"] = terrain.MoveCost,
                ["concealment"] = terrain.Concealment,
                ["defense"] = terrain.Defense,
                ["passable"] = terrain.Passable,
            };

        private static IReadOnlyDictionary<string, object> RewardValues(EnvConfig config) =>
            new Dictionary<string, object>
            {
                ["shape_scale"] = config.ShapeScale,
                ["step_penalty"] = config.StepPenalty,
                ["closing_weight"] = config.ClosingWeight,
                ["draw_credit_weight"] = config.DrawCreditWeight,
                ["points_weight"] = config.PointsWeight,
                ["terminal_win"] = 1f,
                ["terminal_loss"] = -1f,
            };

        private static string CanonicalAdaptiveJson(
            IReadOnlyDictionary<string, object> semantics,
            IReadOnlyList<string> roster,
            IReadOnlyDictionary<string, object> reward,
            string environmentKind,
            int actionSize,
            int observationSize)
        {
            var document = ReadOnlyMap(new Dictionary<string, object>
            {
                ["action_size"] = actionSize,
                ["contract_version"] = AdaptiveVersion,
                ["environment_kind"] = environmentKind,
                ["observation_size"] = observationSize,
                ["reward"] = reward,
                ["roster"] = roster,
                ["semantics"] = semantics,
            });
            return AppendCanonicalValue(new StringBuilder(), document).ToString();
        }

        private static string CanonicalEncodingJson(
            string version,
            int observationSize,
            int actionSize,
            IReadOnlyDictionary<string, object> board,
            IReadOnlyList<string> roster,
            IReadOnlyDictionary<string, object> semantics)
        {
            var document = ReadOnlyMap(new Dictionary<string, object>
            {
                ["action_size"] = actionSize,
                ["board"] = NormalizeEncodingValue(board),
                ["contract_version"] = version,
                ["observation_size"] = observationSize,
                ["roster"] = roster,
                ["semantics"] = NormalizeEncodingValue(semantics),
            });
            return AppendCanonicalValue(new StringBuilder(), document).ToString();
        }

        // Encoding identity is deliberately narrower than the full run contract: inference may transfer
        // across environment role, episode horizon, and reward shaping, but not across board, rules, or
        // adaptive architecture changes. Keep this exclusion list intentionally small until fine-tuning
        // compatibility has an explicit contract of its own.
        private static object NormalizeEncodingValue(object value)
        {
            if (value is IReadOnlyDictionary<string, object> dictionary)
            {
                var normalized = new Dictionary<string, object>();
                foreach (var pair in dictionary)
                {
                    if (pair.Key == "environment_kind" || pair.Key == "effective_horizon"
                        || pair.Key == "max_steps" || pair.Key == "intermediate_decision_penalty"
                        || pair.Key == "deployment_completion_bonus") continue;
                    normalized[pair.Key] = NormalizeEncodingValue(pair.Value);
                }
                return ReadOnlyMap(normalized);
            }
            if (value is IEnumerable sequence && value is not string)
            {
                var normalized = new List<object?>();
                foreach (object? item in sequence)
                    normalized.Add(item == null ? null : NormalizeEncodingValue(item));
                return Array.AsReadOnly(normalized.ToArray());
            }
            return value;
        }

        private static StringBuilder AppendCanonicalValue(StringBuilder text, object? value)
        {
            switch (value)
            {
                case null:
                    return text.Append("null");
                case string stringValue:
                    return text.AppendJsonString(stringValue);
                case bool boolValue:
                    return text.AppendBool(boolValue);
                case int intValue:
                    return text.Append(intValue);
                case long longValue:
                    return text.Append(longValue);
                case float floatValue:
                    return text.AppendNumber(floatValue);
                case double doubleValue:
                    return text.AppendNumber(doubleValue);
                case IReadOnlyDictionary<string, object> dictionary:
                    text.Append('{');
                    bool firstProperty = true;
                    foreach (string key in dictionary.Keys.OrderBy(key => key, StringComparer.Ordinal))
                    {
                        if (!firstProperty) text.Append(',');
                        firstProperty = false;
                        text.AppendJsonString(key).Append(':');
                        AppendCanonicalValue(text, dictionary[key]);
                    }
                    return text.Append('}');
                case IEnumerable sequence:
                    text.Append('[');
                    bool firstItem = true;
                    foreach (object? item in sequence)
                    {
                        if (!firstItem) text.Append(',');
                        firstItem = false;
                        AppendCanonicalValue(text, item);
                    }
                    return text.Append(']');
                default:
                    throw new InvalidOperationException($"Unsupported adaptive contract value type {value.GetType().FullName}");
            }
        }

        private static string CanonicalJson(EnvConfig config, TacticalLayout layout, IReadOnlyList<string> roster,
            string environmentKind, int maxSteps)
        {
            var b = config.BoardGen;
            var g = config.Game;
            var text = new StringBuilder();
            text.Append("{\"action_size\":").Append(layout.ActionCount)
                .Append(",\"board\":{")
                .Append("\"actions_per_turn\":").Append(g.TurnPolicy.ActionsPerTurn ?? -1)
                .Append(",\"biomes_enabled\":").AppendBool(g.BiomesEnabled)
                .Append(",\"bounty_rate\":").AppendNumber(g.BountyRate)
                .Append(",\"build_anywhere\":").AppendBool(g.BuildAnywhere)
                .Append(",\"build_factor\":").AppendNumber(g.BuildFactor)
                .Append(",\"capture_cost\":").Append(g.CaptureCost)
                .Append(",\"capture_factor\":").AppendNumber(g.CaptureFactor)
                .Append(",\"claim_ends_turn\":").AppendBool(g.ClaimEndsTurn)
                .Append(",\"damage_floor\":").Append(g.DamageFloor)
                .Append(",\"deploy_cost_multiplier\":").AppendNumber(g.DeployCostMultiplier)
                .Append(",\"design_fee\":").Append(g.DesignFee)
                .Append(",\"dmg_high_ground_bonus\":").Append(g.DmgHighGroundBonus)
                .Append(",\"economy_win_threshold\":").Append(g.EconomyWinThreshold)
                .Append(",\"flat_chance\":").AppendNumber(b.FlatChance)
                .Append(",\"fog_of_war\":").AppendBool(g.FogOfWar)
                .Append(",\"environment_kind\":").AppendJsonString(environmentKind)
                .Append(",\"forest\":").AppendTerrain(g.Terrain(TerrainType.Forest))
                .Append(",\"forest_weight\":").Append(b.ForestWeight)
                .Append(",\"generator_cost\":").Append(g.GeneratorCost)
                .Append(",\"generator_health\":").Append(g.GeneratorHealth)
                .Append(",\"generator_output\":").Append(g.GeneratorOutput)
                .Append(",\"generators_enabled\":").AppendBool(g.GeneratorsEnabled)
                .Append(",\"height\":").Append(b.Height)
                .Append(",\"max_elevation\":").Append(b.MaxElevation)
                .Append(",\"max_steps\":").Append(maxSteps)
                .Append(",\"plains_weight\":").Append(b.PlainsWeight)
                .Append(",\"plains\":").AppendTerrain(g.Terrain(TerrainType.Plains))
                .Append(",\"point_decay\":").AppendNumber(g.PointDecay)
                .Append(",\"range_high_ground_bonus\":").Append(g.RangeHighGroundBonus)
                .Append(",\"rough_weight\":").Append(b.RoughWeight)
                .Append(",\"rough\":").AppendTerrain(g.Terrain(TerrainType.Rough))
                .Append(",\"round_cap\":").Append(g.RoundCap)
                .Append(",\"score_army\":").Append(g.ScoreArmy)
                .Append(",\"score_kills\":").Append(g.ScoreKills)
                .Append(",\"score_points\":").Append(g.ScorePoints)
                .Append(",\"score_territory\":").Append(g.ScoreTerritory)
                .Append(",\"starting_points\":").Append(g.StartingPoints)
                .Append(",\"territory_income\":").Append(g.TerritoryIncome)
                .Append(",\"territory_mode\":").AppendBool(g.TerritoryMode)
                .Append(",\"turn_policy\":").AppendJsonString(g.TurnPolicy.GetType().FullName ?? g.TurnPolicy.GetType().Name)
                .Append(",\"upkeep_factor\":").AppendNumber(g.UpkeepFactor)
                .Append(",\"water_weight\":").Append(b.WaterWeight)
                .Append(",\"water\":").AppendTerrain(g.Terrain(TerrainType.Water))
                .Append(",\"width\":").Append(b.Width)
                .Append(",\"win_conditions\":").Append((int)g.WinConditions)
                .Append(",\"zone_depth\":").Append(b.ZoneDepth)
                .Append("},\"contract_version\":").AppendJsonString(CurrentVersion)
                .Append(",\"environment_kind\":").AppendJsonString(environmentKind)
                .Append(",\"observation_size\":").Append(layout.ObservationLength)
                .Append(",\"reward\":{")
                .Append("\"closing_weight\":").AppendNumber(config.ClosingWeight)
                .Append(",\"draw_credit_weight\":").AppendNumber(config.DrawCreditWeight)
                .Append(",\"points_weight\":").AppendNumber(config.PointsWeight)
                .Append(",\"shape_scale\":").AppendNumber(config.ShapeScale)
                .Append(",\"step_penalty\":").AppendNumber(config.StepPenalty)
                .Append(",\"terminal_loss\":-1,\"terminal_win\":1},\"roster\":[");

            for (int i = 0; i < roster.Count; i++)
            {
                if (i > 0) text.Append(',');
                text.AppendJsonString(roster[i]);
            }
            return text.Append("]}").ToString();
        }

        private static string RosterEntry(UnitStats stats) => string.Join(",", new[]
        {
            stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
            stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
        });

        private static string Sha256(string canonicalJson)
        {
            using var sha = SHA256.Create();
            var hash = sha.ComputeHash(Encoding.UTF8.GetBytes(canonicalJson));
            var text = new StringBuilder(hash.Length * 2);
            foreach (var value in hash) text.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            return text.ToString();
        }
    }

    internal static class MlContractStringBuilderExtensions
    {
        public static StringBuilder AppendBool(this StringBuilder text, bool value) => text.Append(value ? "true" : "false");
        public static StringBuilder AppendNumber(this StringBuilder text, double value) => text.Append(value.ToString("R", CultureInfo.InvariantCulture));
        public static StringBuilder AppendNumber(this StringBuilder text, float value) => text.Append(value.ToString("R", CultureInfo.InvariantCulture));
        public static StringBuilder AppendTerrain(this StringBuilder text, TerrainDef terrain) => text
            .Append("{\"concealment\":").Append(terrain.Concealment)
            .Append(",\"defense\":").Append(terrain.Defense)
            .Append(",\"move_cost\":").Append(terrain.MoveCost)
            .Append(",\"passable\":").AppendBool(terrain.Passable)
            .Append('}');
        public static StringBuilder AppendJsonString(this StringBuilder text, string value)
        {
            text.Append('"');
            foreach (var ch in value)
            {
                switch (ch)
                {
                    case '\\': text.Append("\\\\"); break;
                    case '"': text.Append("\\\""); break;
                    case '\n': text.Append("\\n"); break;
                    case '\r': text.Append("\\r"); break;
                    case '\t': text.Append("\\t"); break;
                    default:
                        if (ch < ' ') text.Append("\\u").Append(((int)ch).ToString("x4", CultureInfo.InvariantCulture));
                        else text.Append(ch);
                        break;
                }
            }
            return text.Append('"');
        }
    }
}
