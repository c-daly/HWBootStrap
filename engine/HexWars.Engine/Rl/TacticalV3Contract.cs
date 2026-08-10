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
    /// <summary>
    /// Independent identities for tactical-v3 match semantics, structured encoding semantics, and
    /// deployment capacity. Schema authority is explicit here; runtime reflection and property order
    /// never participate in any hash.
    /// </summary>
    public sealed class TacticalV3Contract
    {
        private const string CurrentVersion = "tactical-v3";

        private TacticalV3Contract(
            string environmentKind,
            string contractHash,
            string encodingHash,
            string capacityHash,
            IReadOnlyDictionary<string, object> match,
            IReadOnlyDictionary<string, object> encoding,
            IReadOnlyDictionary<string, object> capacity)
        {
            Version = CurrentVersion;
            EnvironmentKind = environmentKind;
            ContractHash = contractHash;
            EncodingHash = encodingHash;
            CapacityHash = capacityHash;
            Match = match;
            Encoding = encoding;
            Capacity = capacity;
        }

        public const int SchemaVersion = 1;

        public string Version { get; }
        public string EnvironmentKind { get; }
        public string ContractHash { get; }
        public string EncodingHash { get; }
        public string CapacityHash { get; }
        public IReadOnlyDictionary<string, object> Match { get; }
        public IReadOnlyDictionary<string, object> Encoding { get; }
        public IReadOnlyDictionary<string, object> Capacity { get; }

        public static TacticalV3Contract Create(
            TacticalV3Config config, MlEnvironmentKind environmentKind)
        {
            if (config == null) throw new ArgumentNullException(nameof(config));

            string kind = EnvironmentKindName(environmentKind);
            IReadOnlyDictionary<string, object> encoding = EncodingValues();
            string encodingHash = Sha256(CanonicalJson(encoding));
            IReadOnlyDictionary<string, object> capacity = CapacityValues(config.Capacity);
            string capacityHash = Sha256(CanonicalJson(capacity));
            IReadOnlyDictionary<string, object> match = MatchValues(config);
            var contract = Map(new Dictionary<string, object>
            {
                ["encoding_hash"] = encodingHash,
                ["environment_kind"] = kind,
                ["match"] = match,
                ["schema_version"] = SchemaVersion,
                ["version"] = CurrentVersion,
            });

            return new TacticalV3Contract(
                kind,
                Sha256(CanonicalJson(contract)),
                encodingHash,
                capacityHash,
                match,
                encoding,
                capacity);
        }

        private static IReadOnlyDictionary<string, object> MatchValues(TacticalV3Config config)
        {
            TacticalV2Config match = config.Match;
            return Map(new Dictionary<string, object>
            {
                ["board"] = BoardValues(match.BoardGen),
                ["game"] = GameValues(match.Game),
                ["max_controllable_units"] = match.MaxControllableUnits,
                ["max_steps"] = match.MaxSteps,
                ["placement_policy"] = match.PlacementPolicy,
                ["reward"] = RewardValues(config.Reward),
                ["start_distribution"] = StartDistributionValues(match.StartDistribution),
                ["start_profiles"] = StartProfileValues(match.StartProfiles),
                ["starting_unit_count"] = match.StartingUnitCount,
                ["templates"] = TemplateValues(match.Templates),
            });
        }

        private static IReadOnlyDictionary<string, object> BoardValues(BoardGenConfig board) =>
            Map(new Dictionary<string, object>
            {
                ["flat_chance"] = board.FlatChance,
                ["forest_weight"] = board.ForestWeight,
                ["height"] = board.Height,
                ["hex_offset_layout"] = "odd-q",
                ["max_elevation"] = board.MaxElevation,
                ["plains_weight"] = board.PlainsWeight,
                ["rough_weight"] = board.RoughWeight,
                ["water_weight"] = board.WaterWeight,
                ["width"] = board.Width,
                ["zone_depth"] = board.ZoneDepth,
            });

        private static IReadOnlyDictionary<string, object> GameValues(GameConfig game) =>
            Map(new Dictionary<string, object>
            {
                ["actions_per_turn"] = game.TurnPolicy.ActionsPerTurn.HasValue
                    ? (object)game.TurnPolicy.ActionsPerTurn.Value
                    : null!,
                ["biomes_enabled"] = game.BiomesEnabled,
                ["bounty_rate"] = game.BountyRate,
                ["build_anywhere"] = game.BuildAnywhere,
                ["build_factor"] = game.BuildFactor,
                ["capture_cost"] = game.CaptureCost,
                ["capture_factor"] = game.CaptureFactor,
                ["claim_ends_turn"] = game.ClaimEndsTurn,
                ["damage_floor"] = game.DamageFloor,
                ["deploy_cost_multiplier"] = game.DeployCostMultiplier,
                ["design_fee"] = game.DesignFee,
                ["dmg_high_ground_bonus"] = game.DmgHighGroundBonus,
                ["economy_win_threshold"] = game.EconomyWinThreshold,
                ["fixed_template_count"] = game.FixedTemplateCount,
                ["fog_of_war"] = game.FogOfWar,
                ["generator_cost"] = game.GeneratorCost,
                ["generator_health"] = game.GeneratorHealth,
                ["generator_output"] = game.GeneratorOutput,
                ["generators_enabled"] = game.GeneratorsEnabled,
                ["max_design_point_cost"] = game.MaxDesignPointCost,
                ["point_decay"] = game.PointDecay,
                ["range_high_ground_bonus"] = game.RangeHighGroundBonus,
                ["round_cap"] = game.RoundCap,
                ["score_army"] = game.ScoreArmy,
                ["score_kills"] = game.ScoreKills,
                ["score_points"] = game.ScorePoints,
                ["score_territory"] = game.ScoreTerritory,
                ["starting_points"] = game.StartingPoints,
                ["template_slot_count"] = game.TemplateSlotCount,
                ["terrain"] = TerrainValues(game),
                ["territory_income"] = game.TerritoryIncome,
                ["territory_mode"] = game.TerritoryMode,
                ["turn_policy"] = TurnPolicyName(game.TurnPolicy),
                ["upkeep_factor"] = game.UpkeepFactor,
                ["win_conditions"] = WinConditionValues(game.WinConditions),
            });

        private static IReadOnlyDictionary<string, object> TerrainValues(GameConfig game) =>
            Map(new Dictionary<string, object>
            {
                ["forest"] = TerrainValue(game.Terrain(TerrainType.Forest)),
                ["plains"] = TerrainValue(game.Terrain(TerrainType.Plains)),
                ["rough"] = TerrainValue(game.Terrain(TerrainType.Rough)),
                ["water"] = TerrainValue(game.Terrain(TerrainType.Water)),
            });

        private static IReadOnlyDictionary<string, object> TerrainValue(TerrainDef terrain) =>
            Map(new Dictionary<string, object>
            {
                ["concealment"] = terrain.Concealment,
                ["defense"] = terrain.Defense,
                ["move_cost"] = terrain.MoveCost,
                ["passable"] = terrain.Passable,
            });

        private static IReadOnlyDictionary<string, object> RewardValues(TacticalV3RewardConfig reward) =>
            Map(new Dictionary<string, object>
            {
                ["material_adjustment_bound"] = reward.MaterialAdjustmentBound,
                ["points_weight"] = reward.PointsWeight,
                ["terminal_non_win"] = reward.TerminalNonWin,
                ["terminal_win"] = reward.TerminalWin,
                ["time_pressure_bound"] = reward.TimePressureBound,
            });

        private static IReadOnlyList<object> TemplateValues(
            IReadOnlyList<TacticalV2Template> templates)
        {
            if (templates == null) throw new ArgumentException("template catalog must not be null");
            var values = new object[templates.Count];
            for (int index = 0; index < templates.Count; index++)
            {
                TacticalV2Template template = templates[index] ??
                    throw new ArgumentException("template catalog must not contain null entries");
                values[index] = Map(new Dictionary<string, object>
                {
                    ["capability_allocations"] = CapabilityAllocationValues(template.Template.Stats),
                });
            }
            return Array.AsReadOnly(values);
        }

        private static IReadOnlyList<object> CapabilityAllocationValues(UnitStats stats)
        {
            IReadOnlyList<TacticalV3CapabilityDefinition> definitions = TacticalV3Capabilities.All;
            var values = new object[definitions.Count];
            for (int index = 0; index < definitions.Count; index++)
            {
                TacticalV3CapabilityKind kind = definitions[index].Kind;
                int value = CapabilityValue(kind, stats);
                values[index] = Map(new Dictionary<string, object>
                {
                    ["capability"] = CapabilityKindName(kind),
                    ["effective_value"] = value,
                    ["purchased_level"] = value,
                });
            }
            return Array.AsReadOnly(values);
        }

        private static IReadOnlyList<object> StartProfileValues(
            IReadOnlyList<TacticalV2StartProfile> profiles)
        {
            if (profiles == null) throw new ArgumentException("start profile catalog must not be null");
            return Array.AsReadOnly(profiles
                .OrderBy(profile => profile.Id, StringComparer.Ordinal)
                .Select(profile => (object)Map(new Dictionary<string, object>
                {
                    ["id"] = profile.Id,
                    ["learner_unit_count"] = profile.LearnerUnitCount,
                    ["opponent_unit_count"] = profile.OpponentUnitCount,
                    ["separation"] = profile.Separation,
                }))
                .ToArray());
        }

        private static IReadOnlyList<object> StartDistributionValues(
            TacticalV2StartDistribution distribution)
        {
            if (distribution == null) throw new ArgumentException("start distribution must not be null");
            return Array.AsReadOnly(distribution.Weights
                .OrderBy(weight => weight.ProfileId, StringComparer.Ordinal)
                .Select(weight => (object)Map(new Dictionary<string, object>
                {
                    ["basis_points"] = weight.BasisPoints,
                    ["profile_id"] = weight.ProfileId,
                }))
                .ToArray());
        }

        private static IReadOnlyDictionary<string, object> CapacityValues(
            TacticalV3CapacityProfile capacity) =>
            Map(new Dictionary<string, object>
            {
                ["max_cells"] = capacity.MaxCells,
                ["max_units"] = capacity.MaxUnits,
                ["max_templates"] = capacity.MaxTemplates,
                ["max_capability_definitions"] = capacity.MaxCapabilityDefinitions,
                ["max_capability_allocations"] = capacity.MaxCapabilityAllocations,
                ["max_rules"] = capacity.MaxRules,
                ["max_memory_records"] = capacity.MaxMemoryRecords,
                ["max_relations"] = capacity.MaxRelations,
                ["max_candidates"] = capacity.MaxCandidates,
            });

        private static IReadOnlyDictionary<string, object> EncodingValues()
        {
            IReadOnlyList<string> candidateFields = Strings(
                "candidate_id:int32", "decision_id:int64", "kind:candidate_kind",
                "actor:nullable_token_ref", "target:nullable_token_ref",
                "template:nullable_token_ref", "cell:nullable_token_ref",
                "projection:projected_delta");
            IReadOnlyList<string> projectionFields = Strings(
                "source_cell:nullable_token_ref", "destination_cell:nullable_token_ref",
                "template:nullable_token_ref", "target:nullable_token_ref",
                "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                "target_hp_delta:int32", "damage:int32", "is_lethal:bool",
                "bounty_delta:int32", "points_delta:int32", "round_delta:int32", "is_terminal:bool");

            var tables = Map(new Dictionary<string, object>
            {
                ["cells"] = Strings(
                    "q:int32", "r:int32", "terrain:terrain_type", "elevation:int32",
                    "self_deployment_zone:bool", "opponent_deployment_zone:bool",
                    "controller:nullable_relative_owner", "is_boundary:bool",
                    "currently_visible:bool", "previously_observed:bool"),
                ["units"] = Strings(
                    "owner:relative_owner", "current_hp:int32", "max_hp:int32", "cell:token_ref",
                    "elevation:int32", "moved:bool", "attacked:bool",
                    "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                    "point_cost:int32", "deploy_cost:int32", "currently_visible:bool"),
                ["templates"] = Strings(
                    "owner:relative_owner", "point_cost:int32", "deploy_cost:int32",
                    "is_fixed:bool", "is_deployable:bool"),
                ["capability_definitions"] = Strings("kind:capability_kind"),
                ["capability_allocations"] = Strings(
                    "owner:token_ref", "definition:token_ref", "capability:capability_kind",
                    "purchased_level:int32", "effective_value:int32"),
                ["rules"] = Strings(
                    "kind:rule_kind", "int_value:int32", "float_value:float32", "bool_value:bool"),
                ["memory"] = Strings(
                    "cell:token_ref", "last_seen_round:int32", "observation_age:int32",
                    "last_known_current_hp:int32", "currently_visible:bool"),
                ["relations"] = Strings(
                    "kind:relation_kind", "source:token_ref", "target:token_ref",
                    "int_feature:int32", "float_feature:float32", "bool_feature:bool"),
                ["candidates"] = candidateFields,
            });

            var enums = Map(new Dictionary<string, object>
            {
                ["table_kind"] = Strings(
                    "cells", "units", "templates", "capability_definitions",
                    "capability_allocations", "rules", "memory_records", "relations", "candidates"),
                ["relative_owner"] = Strings("self", "opponent"),
                ["terrain_type"] = Strings("plains", "forest", "rough", "water"),
                ["rule_kind"] = Strings(
                    "win_conditions", "round", "round_cap", "actions_per_turn", "starting_points",
                    "self_points", "opponent_points", "damage_floor", "damage_high_ground_bonus",
                    "range_high_ground_bonus", "bounty_rate", "deploy_cost_multiplier",
                    "fog_of_war", "max_design_point_cost", "design_fee"),
                ["relation_kind"] = Strings("neighbor", "occupies", "has_capability"),
                ["capability_kind"] = Strings(
                    "health", "damage", "defense", "movement", "vertical_movement",
                    "range", "range_arc", "vision", "vision_arc"),
                ["action_kind"] = Strings("move", "attack", "deploy", "end_turn"),
                ["capability_relation_kind"] = Strings("opposes", "reduces", "enables_action"),
                ["candidate_kind"] = Strings("attack", "move", "deploy", "end_turn"),
                ["win_condition"] = Strings("none", "annihilation", "economy", "score"),
            });

            return Map(new Dictionary<string, object>
            {
                ["capability_descriptors"] = CapabilityDescriptorValues(),
                ["candidate_schema"] = Map(new Dictionary<string, object>
                {
                    ["fields"] = candidateFields,
                    ["projection_fields"] = projectionFields,
                }),
                ["enums"] = enums,
                ["hex_offset_layout"] = "odd-q",
                ["schema_version"] = SchemaVersion,
                ["tables"] = tables,
                ["token_reference_schema"] = Strings("table:table_kind", "row:int32"),
                ["version"] = CurrentVersion,
            });
        }

        private static IReadOnlyDictionary<string, object> CapabilityDescriptorValues()
        {
            IReadOnlyList<TacticalV3CapabilityDefinition> definitions = TacticalV3Capabilities.All;
            var definitionNames = new string[definitions.Count];
            for (int index = 0; index < definitions.Count; index++)
                definitionNames[index] = CapabilityKindName(definitions[index].Kind);

            IReadOnlyList<TacticalV3CapabilityRelation> relations = TacticalV3Capabilities.Relations;
            var relationValues = new object[relations.Count];
            for (int index = 0; index < relations.Count; index++)
            {
                TacticalV3CapabilityRelation relation = relations[index];
                relationValues[index] = Map(new Dictionary<string, object>
                {
                    ["kind"] = CapabilityRelationKindName(relation.Kind),
                    ["source"] = CapabilityKindName(relation.Source),
                    ["target"] = RelationTargetName(relation.Target),
                });
            }

            return Map(new Dictionary<string, object>
            {
                ["definition_fields"] = Strings("kind:capability_kind"),
                ["definitions"] = Array.AsReadOnly(definitionNames),
                ["relation_fields"] = Strings(
                    "source:capability_kind", "kind:capability_relation_kind",
                    "target:capability_or_action"),
                ["relations"] = Array.AsReadOnly(relationValues),
            });
        }

        private static string EnvironmentKindName(MlEnvironmentKind kind) => kind switch
        {
            MlEnvironmentKind.Tactical => "tactical",
            MlEnvironmentKind.Duel => "duel",
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static string TurnPolicyName(ITurnPolicy policy) => policy switch
        {
            AllUnitsPolicy _ => "all_units",
            OneActionPolicy _ => "one_action",
            KActionsPolicy _ => "k_actions",
            _ => throw new ArgumentException(
                $"unsupported tactical-v3 turn policy {policy.GetType().FullName}", nameof(policy)),
        };

        private static IReadOnlyList<string> WinConditionValues(WinBy conditions)
        {
            const WinBy known = WinBy.Annihilation | WinBy.Economy | WinBy.Score;
            if ((conditions & ~known) != 0)
                throw new ArgumentOutOfRangeException(nameof(conditions));

            var values = new List<string>();
            if (conditions == WinBy.None) values.Add("none");
            if ((conditions & WinBy.Annihilation) != 0) values.Add("annihilation");
            if ((conditions & WinBy.Economy) != 0) values.Add("economy");
            if ((conditions & WinBy.Score) != 0) values.Add("score");
            return Array.AsReadOnly(values.ToArray());
        }

        private static int CapabilityValue(TacticalV3CapabilityKind kind, UnitStats stats) => kind switch
        {
            TacticalV3CapabilityKind.Health => stats.Health,
            TacticalV3CapabilityKind.Damage => stats.Damage,
            TacticalV3CapabilityKind.Defense => stats.Defense,
            TacticalV3CapabilityKind.Movement => stats.Movement,
            TacticalV3CapabilityKind.VerticalMovement => stats.VerticalMovement,
            TacticalV3CapabilityKind.Range => stats.Range,
            TacticalV3CapabilityKind.RangeArc => stats.RangeArc,
            TacticalV3CapabilityKind.Vision => stats.Vision,
            TacticalV3CapabilityKind.VisionArc => stats.VisionArc,
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static string CapabilityKindName(TacticalV3CapabilityKind kind) => kind switch
        {
            TacticalV3CapabilityKind.Health => "health",
            TacticalV3CapabilityKind.Damage => "damage",
            TacticalV3CapabilityKind.Defense => "defense",
            TacticalV3CapabilityKind.Movement => "movement",
            TacticalV3CapabilityKind.VerticalMovement => "vertical_movement",
            TacticalV3CapabilityKind.Range => "range",
            TacticalV3CapabilityKind.RangeArc => "range_arc",
            TacticalV3CapabilityKind.Vision => "vision",
            TacticalV3CapabilityKind.VisionArc => "vision_arc",
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static string CapabilityRelationKindName(
            TacticalV3CapabilityRelationKind kind) => kind switch
        {
            TacticalV3CapabilityRelationKind.Opposes => "opposes",
            TacticalV3CapabilityRelationKind.Reduces => "reduces",
            TacticalV3CapabilityRelationKind.EnablesAction => "enables_action",
            _ => throw new ArgumentOutOfRangeException(nameof(kind)),
        };

        private static string RelationTargetName(TacticalV3RelationTarget target)
        {
            if (target == TacticalV3CapabilityKind.Health) return "capability:health";
            if (target == TacticalV3CapabilityKind.Damage) return "capability:damage";
            if (target == TacticalV3CapabilityKind.Defense) return "capability:defense";
            if (target == TacticalV3CapabilityKind.Movement) return "capability:movement";
            if (target == TacticalV3CapabilityKind.VerticalMovement) return "capability:vertical_movement";
            if (target == TacticalV3CapabilityKind.Range) return "capability:range";
            if (target == TacticalV3CapabilityKind.RangeArc) return "capability:range_arc";
            if (target == TacticalV3CapabilityKind.Vision) return "capability:vision";
            if (target == TacticalV3CapabilityKind.VisionArc) return "capability:vision_arc";
            if (target == TacticalV3ActionKind.Move) return "action:move";
            if (target == TacticalV3ActionKind.Attack) return "action:attack";
            if (target == TacticalV3ActionKind.Deploy) return "action:deploy";
            if (target == TacticalV3ActionKind.EndTurn) return "action:end_turn";
            throw new ArgumentException("unsupported tactical-v3 capability relation target", nameof(target));
        }

        private static IReadOnlyList<string> Strings(params string[] values) =>
            Array.AsReadOnly(values);

        private static IReadOnlyDictionary<string, object> Map(
            IDictionary<string, object> values) =>
            new ReadOnlyDictionary<string, object>(
                new Dictionary<string, object>(values, StringComparer.Ordinal));

        private static string CanonicalJson(object? value) =>
            AppendCanonicalValue(new StringBuilder(), value).ToString();

        private static StringBuilder AppendCanonicalValue(StringBuilder text, object? value)
        {
            switch (value)
            {
                case null:
                    return text.Append("null");
                case string stringValue:
                    return AppendJsonString(text, stringValue);
                case bool boolValue:
                    return text.Append(boolValue ? "true" : "false");
                case int intValue:
                    return text.Append(intValue.ToString(CultureInfo.InvariantCulture));
                case long longValue:
                    return text.Append(longValue.ToString(CultureInfo.InvariantCulture));
                case float floatValue:
                    return text.Append(floatValue.ToString("R", CultureInfo.InvariantCulture));
                case double doubleValue:
                    return text.Append(doubleValue.ToString("R", CultureInfo.InvariantCulture));
                case IReadOnlyDictionary<string, object> dictionary:
                    text.Append('{');
                    bool firstProperty = true;
                    foreach (string key in dictionary.Keys.OrderBy(key => key, StringComparer.Ordinal))
                    {
                        if (!firstProperty) text.Append(',');
                        firstProperty = false;
                        AppendJsonString(text, key).Append(':');
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
                    throw new InvalidOperationException(
                        $"Unsupported tactical-v3 contract value type {value.GetType().FullName}");
            }
        }

        private static StringBuilder AppendJsonString(StringBuilder text, string value)
        {
            text.Append('"');
            foreach (char character in value)
            {
                switch (character)
                {
                    case '\\':
                        text.Append("\\\\");
                        break;
                    case '"':
                        text.Append("\\\"");
                        break;
                    case '\n':
                        text.Append("\\n");
                        break;
                    case '\r':
                        text.Append("\\r");
                        break;
                    case '\t':
                        text.Append("\\t");
                        break;
                    default:
                        if (character < ' ')
                        {
                            text.Append("\\u");
                            text.Append(((int)character).ToString("x4", CultureInfo.InvariantCulture));
                        }
                        else
                        {
                            text.Append(character);
                        }
                        break;
                }
            }
            return text.Append('"');
        }

        private static string Sha256(string canonicalJson)
        {
            using SHA256 sha = SHA256.Create();
            byte[] hash = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(canonicalJson));
            var text = new StringBuilder(hash.Length * 2);
            foreach (byte value in hash)
                text.Append(value.ToString("x2", CultureInfo.InvariantCulture));
            return text.ToString();
        }
    }
}
