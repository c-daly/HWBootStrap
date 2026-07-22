using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AdaptiveContractTests
    {
        [Test]
        public void AdaptiveDefaults_PinRosterBudgetsAndCatalogs()
        {
            var c = AdaptiveEnvConfig.Default();

            Assert.That(c.Templates.Select(x => x.Name), Is.EqualTo(new[]
            {
                "Frontline", "Assault", "Marksman", "Artillery", "Recon", "Support",
                "Custom A", "Custom B", "Custom C",
            }));
            Assert.That(c.Templates.Select(x => StatLine(x.Stats)), Is.EqualTo(new[]
            {
                "7,2,3,2,2,1,1,3,1", "3,6,0,3,2,2,1,3,1", "2,3,0,2,2,6,1,5,1",
                "3,6,0,1,1,5,2,3,1", "2,1,0,5,3,1,0,7,2", "4,3,2,3,2,3,1,4,1",
                "4,3,1,3,2,2,1,3,1", "5,2,2,2,2,3,1,3,1", "3,4,1,3,2,2,1,4,1",
            }));
            Assert.That(c.FixedTemplateCount, Is.EqualTo(6));
            Assert.That(c.CustomTemplateCount, Is.EqualTo(3));
            Assert.That(c.MaxControllableUnits, Is.EqualTo(24));
            Assert.That(c.StartingUnitCount, Is.EqualTo(6));
            Assert.That(c.StartingArmyBudget, Is.EqualTo(132));
            Assert.That(c.MaxDesignPointCost, Is.EqualTo(24));
            Assert.That(c.MaxSteps, Is.EqualTo(900));
            Assert.That(c.IntermediateDecisionPenalty, Is.EqualTo(0.001f));
            Assert.That(c.DeploymentCompletionBonus, Is.Zero);
            Assert.That(c.Game.BiomesEnabled, Is.False);
            Assert.That(c.Game.FogOfWar, Is.True);

            Assert.That(c.StatValues[AdaptiveStat.Health], Is.EqualTo(Enumerable.Range(1, 8)));
            Assert.That(c.StatValues[AdaptiveStat.Damage], Is.EqualTo(Enumerable.Range(0, 9)));
            Assert.That(c.StatValues[AdaptiveStat.Defense], Is.EqualTo(Enumerable.Range(0, 9)));
            Assert.That(c.StatValues[AdaptiveStat.Movement], Is.EqualTo(Enumerable.Range(0, 7)));
            Assert.That(c.StatValues[AdaptiveStat.VerticalMovement], Is.EqualTo(Enumerable.Range(0, 5)));
            Assert.That(c.StatValues[AdaptiveStat.Range], Is.EqualTo(Enumerable.Range(0, 9)));
            Assert.That(c.StatValues[AdaptiveStat.RangeArc], Is.EqualTo(Enumerable.Range(0, 5)));
            Assert.That(c.StatValues[AdaptiveStat.Vision], Is.EqualTo(Enumerable.Range(0, 11)));
            Assert.That(c.StatValues[AdaptiveStat.VisionArc], Is.EqualTo(Enumerable.Range(0, 5)));
            Assert.That((ICollection<int>)c.StatValues[AdaptiveStat.Vision], Has.Property("IsReadOnly").True);
        }

        [Test]
        public void Validate_ReportsEveryUnsatisfiedAdaptivePreflightInvariant()
        {
            var cells = Enumerable.Range(0, 5).Select(q => new HexCoord(q, 0)).ToArray();
            var board = new Board(
                cells.Select(c => new Tile(c, 0, TerrainType.Plains)),
                zone0: cells,
                zone1: cells);
            var c = AdaptiveEnvConfig.Default();
            c.StartingArmyBudget = 1;
            c.Templates = c.Templates.Take(8).ToArray();
            c.MaxControllableUnits = 5;

            var errors = c.Validate(board);

            Assert.That(errors, Has.Count.EqualTo(5));
            Assert.That(errors[0], Does.Contain("runtime template slot count").And.Contain("template roster"));
            Assert.That(errors[1], Does.Contain("requires 6 cells per seat").And.Contain("only 5"));
            Assert.That(errors[2], Does.Contain("starting deployment requires at least").And.Contain("only 1"));
            Assert.That(errors[3], Is.EqualTo("adaptive roster must contain exactly 9 templates"));
            Assert.That(errors[4], Is.EqualTo("maximum controllable units must cover the starting army"));
        }

        [Test]
        public void AdaptiveContract_IsDeterministicAndSeparatedFromLegacy()
        {
            var a = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveTactical);
            var b = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveTactical);
            var duel = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveDuel);
            var legacy = MlContract.Create(new EnvConfig());

            Assert.That(MlContract.CurrentVersion, Is.EqualTo("tactical-v1"));
            Assert.That(a.Version, Is.EqualTo("adaptive-v1"));
            Assert.That(a.EnvironmentKind, Is.EqualTo("adaptive_tactical"));
            Assert.That(duel.EnvironmentKind, Is.EqualTo("adaptive_duel"));
            Assert.That(a.ContractHash, Is.EqualTo(b.ContractHash));
            Assert.That(a.ContractHash, Is.Not.EqualTo(duel.ContractHash));
            Assert.That(a.ContractHash, Is.Not.EqualTo(legacy.ContractHash));
            Assert.That(a.ActionSize, Is.EqualTo(182));
            Assert.That(a.ObservationSize, Is.EqualTo(5974));
            Assert.That(a.Semantics["max_controllable_units"], Is.EqualTo(24));
            Assert.That(a.Semantics["effective_horizon"], Is.EqualTo(900));
            Assert.That(a.Semantics["fog_rule"], Is.EqualTo(
                "hide_current_enemy_units_and_all_opponent_deployment_until_both_confirm;" +
                "derive_action_masks_from_seat_visible_projection;" +
                "authoritative_hidden_blocker_rejection_is_only_allowed_mask_rejection"));
            Assert.That((IReadOnlyList<string>)a.Semantics["phases"], Is.EqualTo(new[]
            {
                "deployment_root", "deployment_template", "deployment_cell", "deployment_placed_unit",
                "deployment_move_cell", "gameplay_root", "gameplay_unit", "gameplay_unit_command",
                "gameplay_move_cell", "gameplay_attack_cell", "design_slot", "design_stat", "design_value",
                "design_confirm",
            }));

            var regions = (IReadOnlyDictionary<string, object>)a.Semantics["action_regions"];
            AssertRegion(regions, "command", 0, 12);
            AssertRegion(regions, "unit", 12, 24);
            AssertRegion(regions, "template", 36, 9);
            AssertRegion(regions, "cell", 45, 117);
            AssertRegion(regions, "stat", 162, 9);
            AssertRegion(regions, "value", 171, 11);

            var channels = new List<string>
            {
                "elevation", "terrain_plains", "terrain_forest", "terrain_rough", "terrain_water",
                "deployment_zone_self", "current_visibility", "previously_seen",
            };
            for (int i = 0; i < 9; i++) channels.Add($"friendly_role_hp_{i}");
            for (int i = 0; i < 9; i++) channels.Add($"visible_enemy_role_hp_{i}");
            for (int i = 0; i < 24; i++) channels.Add($"friendly_slot_occupancy_{i}");
            Assert.That((IReadOnlyList<string>)a.Semantics["observation_channels"], Is.EqualTo(channels));
        }

        [Test]
        public void AdaptiveContract_HashChangesWithAdaptiveSemanticInputs()
        {
            var baseline = MlContract.CreateAdaptive(AdaptiveEnvConfig.Default(), MlEnvironmentKind.AdaptiveTactical);
            var changedBudget = AdaptiveEnvConfig.Default();
            changedBudget.StartingArmyBudget++;
            var changedPenalty = AdaptiveEnvConfig.Default();
            changedPenalty.IntermediateDecisionPenalty = 0.002f;
            var changedHorizon = AdaptiveEnvConfig.Default();
            changedHorizon.MaxSteps++;
            var changedBoard = AdaptiveEnvConfig.Default();
            changedBoard.BoardGen = new BoardGenConfig(width: 14);
            var changedTemplate = AdaptiveEnvConfig.Default();
            changedTemplate.Templates = changedTemplate.Templates
                .Select((template, i) => i == 8
                    ? new UnitTemplate(template.Name, new UnitStats(4, 4, 1, 3, 2, 2, 1, 4, 1))
                    : template)
                .ToArray();
            var changedCatalog = AdaptiveEnvConfig.Default();
            changedCatalog.StatValues = new Dictionary<AdaptiveStat, IReadOnlyList<int>>(changedCatalog.StatValues)
            {
                [AdaptiveStat.Vision] = Array.AsReadOnly(Enumerable.Range(0, 10).ToArray()),
            };

            Assert.That(MlContract.CreateAdaptive(changedBudget, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(MlContract.CreateAdaptive(changedPenalty, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(MlContract.CreateAdaptive(changedHorizon, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(MlContract.CreateAdaptive(changedBoard, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(MlContract.CreateAdaptive(changedTemplate, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(MlContract.CreateAdaptive(changedCatalog, MlEnvironmentKind.AdaptiveTactical).ContractHash,
                Is.Not.EqualTo(baseline.ContractHash));
        }

        [Test]
        public void LegacyContractHashes_AreCharacterizedBeforeAdaptiveWork()
        {
            Assert.That(MlContract.Create(new EnvConfig(), MlEnvironmentKind.Tactical).ContractHash,
                Is.EqualTo("a2fd8714e5b25ffa7e648e801894a1efeb27a88966946ac1ff8393c89347bf77"));
            Assert.That(MlContract.Create(new EnvConfig(), MlEnvironmentKind.Duel).ContractHash,
                Is.EqualTo("dd78cc337a59753a05a102379f21ef13fea4aba08b60b676c6cf14f5ece2f971"));
        }

        private static void AssertRegion(
            IReadOnlyDictionary<string, object> regions,
            string name,
            int offset,
            int count)
        {
            var region = (IReadOnlyDictionary<string, object>)regions[name];
            Assert.That(region["offset"], Is.EqualTo(offset), name + " offset");
            Assert.That(region["count"], Is.EqualTo(count), name + " count");
        }

        private static string StatLine(UnitStats stats) => string.Join(",", new[]
        {
            stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
            stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
        });
    }
}
