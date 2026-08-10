using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3ContractTests
    {
        [Test]
        public void Create_IsDeterministicAndHashesExactPublicPayloads()
        {
            TacticalV3Contract first = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(), MlEnvironmentKind.Duel);
            TacticalV3Contract second = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(), MlEnvironmentKind.Duel);

            Assert.That(TacticalV3Contract.SchemaVersion, Is.EqualTo(1));
            Assert.That(first.Version, Is.EqualTo("tactical-v3"));
            Assert.That(first.EnvironmentKind, Is.EqualTo("duel"));
            Assert.That(Canonical(first.Match), Is.EqualTo(Canonical(second.Match)));
            Assert.That(Canonical(first.Encoding), Is.EqualTo(Canonical(second.Encoding)));
            Assert.That(Canonical(first.Capacity), Is.EqualTo(Canonical(second.Capacity)));
            Assert.That(first.ContractHash, Is.EqualTo(second.ContractHash));
            Assert.That(first.EncodingHash, Is.EqualTo(second.EncodingHash));
            Assert.That(first.CapacityHash, Is.EqualTo(second.CapacityHash));
            Assert.That(first.EncodingHash, Is.EqualTo(Hash(first.Encoding)));
            Assert.That(first.CapacityHash, Is.EqualTo(Hash(first.Capacity)));
            Assert.That(first.ContractHash, Is.EqualTo(Hash(ContractPayload(first))));
            Assert.That(first.ContractHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(first.EncodingHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(first.CapacityHash, Does.Match("^[0-9a-f]{64}$"));
        }

        [Test]
        public void BoardSizeChangesMatchButNotEncodingIdentity()
        {
            TacticalV3Contract standard = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(13, 9), MlEnvironmentKind.Duel);
            TacticalV3Contract large = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(24, 16), MlEnvironmentKind.Duel);

            Assert.That(large.ContractHash, Is.Not.EqualTo(standard.ContractHash));
            Assert.That(large.EncodingHash, Is.EqualTo(standard.EncodingHash));
            Assert.That(large.CapacityHash, Is.EqualTo(standard.CapacityHash));
        }

        [Test]
        public void TemplateStatChangesMatchButNotEncodingIdentity()
        {
            TacticalV3Config baselineConfig = TacticalV3Fixtures.Config();
            TacticalV2Template source = baselineConfig.Match.Templates[0];
            UnitStats stats = source.Template.Stats;
            var changedStats = new UnitStats(
                stats.Health + 1, stats.Damage, stats.Defense, stats.Movement,
                stats.VerticalMovement, stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc);
            TacticalV2Template[] templates = baselineConfig.Match.Templates.ToArray();
            templates[0] = new TacticalV2Template(source.Id,
                new UnitTemplate(source.Template.Name, changedStats));
            TacticalV3Config changedConfig = new TacticalV3Config(
                TacticalV3Fixtures.CloneMatch(baselineConfig.Match, templates: templates),
                baselineConfig.Capacity, baselineConfig.Reward);

            TacticalV3Contract baseline = TacticalV3Contract.Create(baselineConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract changed = TacticalV3Contract.Create(changedConfig, MlEnvironmentKind.Duel);

            Assert.That(changed.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changed.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changed.CapacityHash, Is.EqualTo(baseline.CapacityHash));
        }

        [Test]
        public void RuleChangesMatchButNotEncodingIdentity()
        {
            TacticalV3Config baselineConfig = TacticalV3Fixtures.Config();
            GameConfig changedGame = TacticalV3Fixtures.CloneGame(
                baselineConfig.Match.Game,
                startingPoints: baselineConfig.Match.Game.StartingPoints + 1);
            TacticalV3Config changedConfig = new TacticalV3Config(
                TacticalV3Fixtures.CloneMatch(baselineConfig.Match, game: changedGame),
                baselineConfig.Capacity, baselineConfig.Reward);

            TacticalV3Contract baseline = TacticalV3Contract.Create(baselineConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract changed = TacticalV3Contract.Create(changedConfig, MlEnvironmentKind.Duel);

            Assert.That(changed.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changed.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changed.CapacityHash, Is.EqualTo(baseline.CapacityHash));
        }

        [Test]
        public void RewardChangesMatchButNotEncodingIdentity()
        {
            TacticalV3Config baselineConfig = TacticalV3Fixtures.Config();
            TacticalV3Config changedConfig = new TacticalV3Config(
                baselineConfig.Match, baselineConfig.Capacity,
                TacticalV3Fixtures.UncheckedReward(materialAdjustmentBound: 0.19f));

            TacticalV3Contract baseline = TacticalV3Contract.Create(baselineConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract changed = TacticalV3Contract.Create(changedConfig, MlEnvironmentKind.Duel);

            Assert.That(changed.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changed.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changed.CapacityHash, Is.EqualTo(baseline.CapacityHash));
        }

        [TestCase(0, "max_cells")]
        [TestCase(1, "max_units")]
        [TestCase(2, "max_templates")]
        [TestCase(3, "max_capability_definitions")]
        [TestCase(4, "max_capability_allocations")]
        [TestCase(5, "max_rules")]
        [TestCase(6, "max_memory_records")]
        [TestCase(7, "max_relations")]
        [TestCase(8, "max_candidates")]
        public void EachCapacityIntegerChangesOnlyCapacityIdentity(
            int changedIndex, string changedKey)
        {
            TacticalV3Config baselineConfig = TacticalV3Fixtures.Config();
            TacticalV3CapacityProfile capacity = baselineConfig.Capacity;
            int[] values =
            {
                capacity.MaxCells, capacity.MaxUnits, capacity.MaxTemplates,
                capacity.MaxCapabilityDefinitions, capacity.MaxCapabilityAllocations,
                capacity.MaxRules, capacity.MaxMemoryRecords, capacity.MaxRelations,
                capacity.MaxCandidates,
            };
            values[changedIndex]++;
            TacticalV3Config changedConfig = new TacticalV3Config(
                baselineConfig.Match,
                new TacticalV3CapacityProfile(
                    values[0], values[1], values[2], values[3], values[4],
                    values[5], values[6], values[7], values[8]),
                baselineConfig.Reward);

            TacticalV3Contract baseline = TacticalV3Contract.Create(baselineConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract changed = TacticalV3Contract.Create(changedConfig, MlEnvironmentKind.Duel);

            string[] keys =
            {
                "max_cells", "max_units", "max_templates", "max_capability_definitions",
                "max_capability_allocations", "max_rules", "max_memory_records",
                "max_relations", "max_candidates",
            };
            Assert.That(changed.CapacityHash, Is.Not.EqualTo(baseline.CapacityHash));
            Assert.That(changed.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changed.ContractHash, Is.EqualTo(baseline.ContractHash));
            Assert.That(changed.Capacity.Keys, Is.EquivalentTo(keys));
            Assert.That(changed.Capacity[changedKey],
                Is.EqualTo((int)baseline.Capacity[changedKey] + 1));
            for (int index = 0; index < keys.Length; index++)
                Assert.That(changed.Capacity[keys[index]], Is.EqualTo(values[index]), keys[index]);
        }

        [Test]
        public void TemplatePresentationMetadataChangesNoIdentity()
        {
            TacticalV3Config baselineConfig = TacticalV3Fixtures.Config();
            TacticalV2Template[] templates = baselineConfig.Match.Templates.ToArray();
            templates[0] = new TacticalV2Template(
                "presentation-only-id",
                new UnitTemplate("Presentation-only name", templates[0].Template.Stats));
            TacticalV3Config changedConfig = new TacticalV3Config(
                TacticalV3Fixtures.CloneMatch(baselineConfig.Match, templates: templates),
                baselineConfig.Capacity, baselineConfig.Reward);

            TacticalV3Contract baseline = TacticalV3Contract.Create(baselineConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract changed = TacticalV3Contract.Create(changedConfig, MlEnvironmentKind.Duel);

            Assert.That(changed.ContractHash, Is.EqualTo(baseline.ContractHash));
            Assert.That(changed.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changed.CapacityHash, Is.EqualTo(baseline.CapacityHash));
        }

        [Test]
        public void EnvironmentKindChangesOnlyMatchContractIdentity()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();

            TacticalV3Contract tactical = TacticalV3Contract.Create(config, MlEnvironmentKind.Tactical);
            TacticalV3Contract duel = TacticalV3Contract.Create(config, MlEnvironmentKind.Duel);

            Assert.That(tactical.EnvironmentKind, Is.EqualTo("tactical"));
            Assert.That(duel.EnvironmentKind, Is.EqualTo("duel"));
            Assert.That(duel.ContractHash, Is.Not.EqualTo(tactical.ContractHash));
            Assert.That(duel.EncodingHash, Is.EqualTo(tactical.EncodingHash));
            Assert.That(duel.CapacityHash, Is.EqualTo(tactical.CapacityHash));
            Assert.That(Canonical(duel.Match), Is.EqualTo(Canonical(tactical.Match)));
        }

        [Test]
        public void EncodingSchema_IsExplicitInductiveAndFreeOfFlatGeometry()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            TacticalV3Contract contract = TacticalV3Contract.Create(
                config, MlEnvironmentKind.Duel);
            var tables = (IReadOnlyDictionary<string, object>)contract.Encoding["tables"];
            var enums = (IReadOnlyDictionary<string, object>)contract.Encoding["enums"];
            var capabilities =
                (IReadOnlyDictionary<string, object>)contract.Encoding["capability_descriptors"];
            var candidates =
                (IReadOnlyDictionary<string, object>)contract.Encoding["candidate_schema"];

            Assert.That(contract.Encoding["hex_offset_layout"], Is.EqualTo("odd-q"));
            Assert.That((IReadOnlyList<string>)contract.Encoding["token_reference_schema"], Is.EqualTo(new[]
            {
                "table:table_kind", "row:int32",
            }));
            Assert.That(tables.Keys, Is.EquivalentTo(new[]
            {
                "cells", "units", "templates", "capability_definitions",
                "capability_allocations", "rules", "memory", "relations", "candidates",
            }));
            Assert.That((IReadOnlyList<string>)tables["cells"], Is.EqualTo(new[]
            {
                "q:int32", "r:int32", "terrain:terrain_type", "elevation:int32",
                "self_deployment_zone:bool", "opponent_deployment_zone:bool",
                "controller:nullable_relative_owner", "is_boundary:bool",
                "currently_visible:bool", "previously_observed:bool",
            }));
            Assert.That((IReadOnlyList<string>)tables["units"], Is.EqualTo(new[]
            {
                "owner:relative_owner", "current_hp:int32", "max_hp:int32", "cell:token_ref",
                "elevation:int32", "moved:bool", "attacked:bool",
                "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                "point_cost:int32", "deploy_cost:int32", "currently_visible:bool",
            }));
            Assert.That((IReadOnlyList<string>)tables["templates"], Is.EqualTo(new[]
            {
                "owner:relative_owner", "point_cost:int32", "deploy_cost:int32",
                "is_fixed:bool", "is_deployable:bool",
            }));
            Assert.That((IReadOnlyList<string>)tables["capability_definitions"], Is.EqualTo(new[]
            {
                "kind:capability_kind",
            }));
            Assert.That((IReadOnlyList<string>)tables["capability_allocations"], Is.EqualTo(new[]
            {
                "owner:token_ref", "definition:token_ref", "capability:capability_kind",
                "purchased_level:int32", "effective_value:int32",
            }));
            Assert.That((IReadOnlyList<string>)tables["rules"], Is.EqualTo(new[]
            {
                "kind:rule_kind", "int_value:int32", "float_value:float32", "bool_value:bool",
            }));
            Assert.That((IReadOnlyList<string>)tables["memory"], Is.EqualTo(new[]
            {
                "cell:token_ref", "last_seen_round:int32", "observation_age:int32",
                "last_known_current_hp:int32", "currently_visible:bool",
            }));
            Assert.That((IReadOnlyList<string>)tables["relations"], Is.EqualTo(new[]
            {
                "kind:relation_kind", "source:token_ref", "target:token_ref",
                "int_feature:int32", "float_feature:float32", "bool_feature:bool",
            }));

            Assert.That(enums.Keys, Is.EquivalentTo(new[]
            {
                "table_kind", "relative_owner", "terrain_type", "rule_kind", "relation_kind",
                "capability_kind", "action_kind", "capability_relation_kind", "candidate_kind",
                "win_condition",
            }));
            Assert.That((IReadOnlyList<string>)enums["table_kind"], Is.EqualTo(new[]
            {
                "cells", "units", "templates", "capability_definitions", "capability_allocations",
                "rules", "memory_records", "relations", "candidates",
            }));
            Assert.That((IReadOnlyList<string>)enums["capability_kind"], Is.EqualTo(new[]
            {
                "health", "damage", "defense", "movement", "vertical_movement",
                "range", "range_arc", "vision", "vision_arc",
            }));
            Assert.That((IReadOnlyList<string>)enums["relative_owner"], Is.EqualTo(new[]
            {
                "self", "opponent",
            }));
            Assert.That((IReadOnlyList<string>)enums["terrain_type"], Is.EqualTo(new[]
            {
                "plains", "forest", "rough", "water",
            }));
            Assert.That((IReadOnlyList<string>)enums["rule_kind"], Is.EqualTo(new[]
            {
                "win_conditions", "round", "round_cap", "actions_per_turn", "starting_points",
                "self_points", "opponent_points", "damage_floor", "damage_high_ground_bonus",
                "range_high_ground_bonus", "bounty_rate", "deploy_cost_multiplier",
                "fog_of_war", "max_design_point_cost", "design_fee",
            }));
            Assert.That((IReadOnlyList<string>)enums["relation_kind"], Is.EqualTo(new[]
            {
                "neighbor", "occupies", "has_capability",
            }));
            Assert.That((IReadOnlyList<string>)enums["action_kind"], Is.EqualTo(new[]
            {
                "move", "attack", "deploy", "end_turn",
            }));
            Assert.That((IReadOnlyList<string>)enums["capability_relation_kind"], Is.EqualTo(new[]
            {
                "opposes", "reduces", "enables_action",
            }));
            Assert.That((IReadOnlyList<string>)enums["candidate_kind"], Is.EqualTo(new[]
            {
                "attack", "move", "deploy", "end_turn",
            }));
            Assert.That((IReadOnlyList<string>)enums["win_condition"], Is.EqualTo(new[]
            {
                "none", "annihilation", "economy", "score",
            }));
            Assert.That((IReadOnlyList<string>)capabilities["definition_fields"], Is.EqualTo(new[]
            {
                "kind:capability_kind",
            }));
            Assert.That((IReadOnlyList<string>)capabilities["definitions"], Is.EqualTo(new[]
            {
                "health", "damage", "defense", "movement", "vertical_movement",
                "range", "range_arc", "vision", "vision_arc",
            }));
            Assert.That((IReadOnlyList<string>)capabilities["relation_fields"], Is.EqualTo(new[]
            {
                "source:capability_kind", "kind:capability_relation_kind",
                "target:capability_or_action",
            }));
            Assert.That(Canonical(capabilities["relations"]), Is.EqualTo(
                "[{\"kind\":\"opposes\",\"source\":\"damage\",\"target\":\"capability:health\"}," +
                "{\"kind\":\"reduces\",\"source\":\"defense\",\"target\":\"capability:damage\"}," +
                "{\"kind\":\"enables_action\",\"source\":\"range\",\"target\":\"action:attack\"}," +
                "{\"kind\":\"enables_action\",\"source\":\"range_arc\",\"target\":\"action:attack\"}]"));

            string[] candidateFields =
            {
                "candidate_id:int32", "decision_id:int64", "kind:candidate_kind",
                "actor:nullable_token_ref", "target:nullable_token_ref",
                "template:nullable_token_ref", "cell:nullable_token_ref",
                "projection:projected_delta",
            };
            Assert.That((IReadOnlyList<string>)candidates["fields"], Is.EqualTo(candidateFields));
            Assert.That((IReadOnlyList<string>)tables["candidates"], Is.EqualTo(candidateFields));
            Assert.That((IReadOnlyList<string>)candidates["projection_fields"], Is.EqualTo(new[]
            {
                "source_cell:nullable_token_ref", "destination_cell:nullable_token_ref",
                "template:nullable_token_ref", "target:nullable_token_ref",
                "horizontal_movement_spent:int32", "vertical_movement_spent:int32",
                "target_hp_delta:int32", "damage:int32", "is_lethal:bool",
                "bounty_delta:int32", "points_delta:int32", "round_delta:int32", "is_terminal:bool",
            }));

            string json = Canonical(contract.Encoding);
            Assert.That(json, Does.Not.Contain("obs_len"));
            Assert.That(json, Does.Not.Contain("n_actions"));
            Assert.That(json, Does.Not.Contain("action_offset"));
            Assert.That(json, Does.Not.Contain("action_regions"));
            foreach (TacticalV2Template template in config.Match.Templates)
            {
                Assert.That(json, Does.Not.Contain(template.Id), template.Id);
                Assert.That(json, Does.Not.Contain(template.Template.Name), template.Template.Name);
            }
        }

        [Test]
        public void SchemaVersionMutationChangesEncodingIdentity()
        {
            TacticalV3Contract contract = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(), MlEnvironmentKind.Duel);
            Dictionary<string, object> mutated = CloneMap(contract.Encoding);
            mutated["schema_version"] = TacticalV3Contract.SchemaVersion + 1;

            Assert.That(Hash(mutated), Is.Not.EqualTo(contract.EncodingHash));
        }

        [Test]
        public void CapabilityRelationMutationChangesEncodingIdentity()
        {
            TacticalV3Contract contract = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(), MlEnvironmentKind.Duel);
            Dictionary<string, object> mutated = CloneMap(contract.Encoding);
            var capabilities = CloneMap(
                (IReadOnlyDictionary<string, object>)mutated["capability_descriptors"]);
            capabilities["relations"] = Array.AsReadOnly(new object[]
            {
                new Dictionary<string, object>
                {
                    ["source"] = "damage",
                    ["kind"] = "enhances",
                    ["target"] = "capability:health",
                },
            });
            mutated["capability_descriptors"] = capabilities;

            Assert.That(Hash(mutated), Is.Not.EqualTo(contract.EncodingHash));
        }

        [Test]
        public void CandidateProjectionMutationChangesEncodingIdentity()
        {
            TacticalV3Contract contract = TacticalV3Contract.Create(
                TacticalV3Fixtures.Config(), MlEnvironmentKind.Duel);
            Dictionary<string, object> mutated = CloneMap(contract.Encoding);
            var candidates = CloneMap(
                (IReadOnlyDictionary<string, object>)mutated["candidate_schema"]);
            candidates["projection_fields"] = Array.AsReadOnly(new[]
            {
                "damage:int64",
            });
            mutated["candidate_schema"] = candidates;

            Assert.That(Hash(mutated), Is.Not.EqualTo(contract.EncodingHash));
        }

        private static IReadOnlyDictionary<string, object> ContractPayload(TacticalV3Contract contract) =>
            new Dictionary<string, object>
            {
                ["encoding_hash"] = contract.EncodingHash,
                ["environment_kind"] = contract.EnvironmentKind,
                ["match"] = contract.Match,
                ["schema_version"] = TacticalV3Contract.SchemaVersion,
                ["version"] = contract.Version,
            };

        private static Dictionary<string, object> CloneMap(
            IReadOnlyDictionary<string, object> source) =>
            source.ToDictionary(pair => pair.Key, pair => pair.Value, StringComparer.Ordinal);

        private static string Hash(object value)
        {
            using SHA256 sha = SHA256.Create();
            byte[] digest = sha.ComputeHash(Encoding.UTF8.GetBytes(Canonical(value)));
            return Convert.ToHexString(digest).ToLowerInvariant();
        }

        private static string Canonical(object? value)
        {
            using var stream = new MemoryStream();
            using (var writer = new Utf8JsonWriter(stream))
            {
                WriteCanonical(writer, value);
            }
            return Encoding.UTF8.GetString(stream.ToArray());
        }

        private static void WriteCanonical(Utf8JsonWriter writer, object? value)
        {
            switch (value)
            {
                case null:
                    writer.WriteNullValue();
                    return;
                case string text:
                    writer.WriteStringValue(text);
                    return;
                case bool flag:
                    writer.WriteBooleanValue(flag);
                    return;
                case int number:
                    writer.WriteNumberValue(number);
                    return;
                case long number:
                    writer.WriteNumberValue(number);
                    return;
                case float number:
                    writer.WriteNumberValue(number);
                    return;
                case double number:
                    writer.WriteNumberValue(number);
                    return;
                case IReadOnlyDictionary<string, object> dictionary:
                    writer.WriteStartObject();
                    foreach (string key in dictionary.Keys.OrderBy(key => key, StringComparer.Ordinal))
                    {
                        writer.WritePropertyName(key);
                        WriteCanonical(writer, dictionary[key]);
                    }
                    writer.WriteEndObject();
                    return;
                case IEnumerable sequence:
                    writer.WriteStartArray();
                    foreach (object? item in sequence) WriteCanonical(writer, item);
                    writer.WriteEndArray();
                    return;
                default:
                    throw new AssertionException(
                        $"Unsupported independent canonical value {value.GetType().FullName}");
            }
        }
    }
}
