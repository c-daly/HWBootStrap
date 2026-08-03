using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class MlContractTests
    {
        [Test]
        public void Create_IsDeterministicAndMatchesTacticalLayout()
        {
            var config = new EnvConfig();
            var layout = new TacticalLayout(config);

            var first = MlContract.Create(config);
            var second = MlContract.Create(config);

            Assert.That(first.ContractHash, Is.EqualTo(second.ContractHash));
            Assert.That(first.EncodingHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(first.EncodingHash, Is.EqualTo(second.EncodingHash));
            Assert.That(first.ObservationSize, Is.EqualTo(layout.ObservationLength));
            Assert.That(first.ActionSize, Is.EqualTo(layout.ActionCount));
            Assert.That(first.Board["width"], Is.EqualTo(layout.BoardW));
            Assert.That(first.Board["height"], Is.EqualTo(layout.BoardH));
        }

        [Test]
        public void Create_ChangesHashForBoardRosterAndRewardSemantics()
        {
            var baseline = MlContract.Create(new EnvConfig());
            var changedBoard = MlContract.Create(new EnvConfig
            {
                BoardGen = new BoardGenConfig(width: 14),
            });
            var changedRoster = MlContract.Create(new EnvConfig
            {
                Roster = new List<UnitStats>
                {
                    new UnitStats(6, 3, 2, 3, 2, 1, 1, 2, 1),
                    new UnitStats(3, 5, 0, 3, 2, 2, 1, 3, 1),
                    new UnitStats(2, 2, 0, 4, 3, 1, 0, 5, 2),
                },
            });
            var changedReward = MlContract.Create(new EnvConfig { ClosingWeight = 0.03f });
            var changedHorizon = MlContract.Create(new EnvConfig { MaxSteps = 601 });

            Assert.That(changedBoard.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedRoster.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedReward.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedHorizon.ContractHash, Is.Not.EqualTo(baseline.ContractHash));
            Assert.That(changedBoard.EncodingHash, Is.Not.EqualTo(baseline.EncodingHash));
            Assert.That(changedRoster.EncodingHash, Is.Not.EqualTo(baseline.EncodingHash));
            Assert.That(changedReward.EncodingHash, Is.EqualTo(baseline.EncodingHash));
            Assert.That(changedHorizon.EncodingHash, Is.EqualTo(baseline.EncodingHash));
        }

        [Test]
        public void Create_UsesDistinctContractsAndEffectiveHorizonsForTacticalAndDuelModes()
        {
            var config = new EnvConfig { MaxSteps = 123 };

            var tactical = MlContract.Create(config, MlEnvironmentKind.Tactical);
            var duel = MlContract.Create(config, MlEnvironmentKind.Duel);

            Assert.That(tactical.EnvironmentKind, Is.EqualTo("tactical"));
            Assert.That(duel.EnvironmentKind, Is.EqualTo("duel"));
            Assert.That(tactical.Board["max_steps"], Is.EqualTo(123));
            Assert.That(duel.Board["max_steps"], Is.EqualTo(246));
            Assert.That(duel.ContractHash, Is.Not.EqualTo(tactical.ContractHash));
            Assert.That(duel.EncodingHash, Is.EqualTo(tactical.EncodingHash));
        }

        [Test]
        public void TacticalV2Contract_SeparatesSlotsAndTemplates()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = config.MaxControllableUnits = 7;
            MlContract contract = MlContract.CreateTacticalV2(config);
            var layout = new TacticalV2Layout(config);

            Assert.That(contract.Version, Is.EqualTo("tactical-v2"));
            Assert.That(contract.ObservationSize, Is.EqualTo(layout.ObservationLength));
            Assert.That(contract.ActionSize, Is.EqualTo(layout.ActionCount));
            Assert.That(contract.Semantics["starting_unit_count"], Is.EqualTo(7));
            Assert.That(contract.Semantics["max_controllable_units"], Is.EqualTo(7));
        }

        [Test]
        public void TacticalV2ProfiledContract_EmitsDeclaredProfilesAndDistribution()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.PlacementPolicy = "profiled-seeded-v1";
            config.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1();
            config.StartDistribution = new TacticalV2StartDistribution(config.StartProfiles.Select(profile =>
                new TacticalV2StartWeight(profile.Id, profile.Id == "conversion-1v1-far" ? 10000 : 0)));

            MlContract contract = MlContract.CreateTacticalV2(config, MlEnvironmentKind.Duel);

            var profiles = (IReadOnlyList<object>)contract.Semantics["start_profiles"];
            var distribution = (IReadOnlyList<object>)contract.Semantics["start_distribution"];
            Assert.That(profiles.Count, Is.EqualTo(10));
            Assert.That(distribution.Count, Is.EqualTo(10));
            Assert.That(contract.Semantics["placement_policy"], Is.EqualTo("profiled-seeded-v1"));
        }

        [Test]
        public void TacticalV1Contract_RemainsByteIdentical()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v1");
            scenario.Board.Width = 24;
            scenario.Board.Height = 24;
            scenario.Rules.StartingPoints = 50;
            MlContract contract = MlContract.Create(scenario.BuildTactical());

            Assert.That(contract.ContractHash, Is.EqualTo(
                "8794d90bde2455c77ba2a4c1c7a22f3fb60f5d4fdb7be766003269a8a5a08c33"));
            Assert.That(contract.EncodingHash, Is.EqualTo(
                "39c428c07a31de09137a8851c62b5e9ebc083af1729636ce8c833b59f450e49b"));
        }

        [Test]
        public void TacticalV2Contract_IsDeterministic()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            MlContract first = MlContract.CreateTacticalV2(config);
            MlContract second = MlContract.CreateTacticalV2(config);

            Assert.That(first.ContractHash, Is.EqualTo(second.ContractHash));
            Assert.That(first.EncodingHash, Does.Match("^[0-9a-f]{64}$"));
            Assert.That(first.EncodingHash, Is.EqualTo(second.EncodingHash));
        }

        [Test]
        public void TacticalV2Contract_DuelSharesEncodingIdentityButNotContractIdentity()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            MlContract tactical = MlContract.CreateTacticalV2(config, MlEnvironmentKind.Tactical);
            MlContract duel = MlContract.CreateTacticalV2(config, MlEnvironmentKind.Duel);

            Assert.That(tactical.EnvironmentKind, Is.EqualTo("tactical"));
            Assert.That(duel.EnvironmentKind, Is.EqualTo("duel"));
            Assert.That(duel.ContractHash, Is.Not.EqualTo(tactical.ContractHash));
            Assert.That(duel.EncodingHash, Is.EqualTo(tactical.EncodingHash));
        }

        [Test]
        public void TacticalV2Contract_EncodingIdentityChangesWithCountOrderNameAndStats()
        {
            TacticalV2Config baseline = TacticalV2Config.Default();
            MlContract baselineContract = MlContract.CreateTacticalV2(baseline);
            List<TacticalV2Template> templates = baseline.Templates.ToList();

            TacticalV2Config differentCount = TacticalV2Config.Default();
            differentCount.StartingUnitCount = differentCount.MaxControllableUnits = 5;
            MlContract differentCountContract = MlContract.CreateTacticalV2(differentCount);

            TacticalV2Config reordered = TacticalV2Config.Default();
            reordered.Templates = new List<TacticalV2Template> { templates[1], templates[0] }
                .Concat(templates.Skip(2)).ToList();
            MlContract reorderedContract = MlContract.CreateTacticalV2(reordered);

            TacticalV2Config renamed = TacticalV2Config.Default();
            UnitTemplate firstTemplate = templates[0].Template;
            var renamedList = new List<TacticalV2Template>(templates)
            {
                [0] = new TacticalV2Template(templates[0].Id, new UnitTemplate("Renamed", firstTemplate.Stats)),
            };
            renamed.Templates = renamedList;
            MlContract renamedContract = MlContract.CreateTacticalV2(renamed);

            TacticalV2Config restatted = TacticalV2Config.Default();
            var boostedStats = new UnitStats(
                firstTemplate.Stats.Health + 1, firstTemplate.Stats.Damage, firstTemplate.Stats.Defense,
                firstTemplate.Stats.Movement, firstTemplate.Stats.VerticalMovement, firstTemplate.Stats.Range,
                firstTemplate.Stats.RangeArc, firstTemplate.Stats.Vision, firstTemplate.Stats.VisionArc);
            var restattedList = new List<TacticalV2Template>(templates)
            {
                [0] = new TacticalV2Template(templates[0].Id, new UnitTemplate(firstTemplate.Name, boostedStats)),
            };
            restatted.Templates = restattedList;
            MlContract restattedContract = MlContract.CreateTacticalV2(restatted);

            Assert.That(differentCountContract.EncodingHash, Is.Not.EqualTo(baselineContract.EncodingHash));
            Assert.That(reorderedContract.EncodingHash, Is.Not.EqualTo(baselineContract.EncodingHash));
            Assert.That(renamedContract.EncodingHash, Is.Not.EqualTo(baselineContract.EncodingHash));
            Assert.That(restattedContract.EncodingHash, Is.Not.EqualTo(baselineContract.EncodingHash));
        }

        [Test]
        public void TacticalV2Contract_EncodingHashIsIndependentOfRewardShaping()
        {
            // Mirrors Create_ChangesHashForBoardRosterAndRewardSemantics' tactical-v1 reward case:
            // reward fields are part of the run's ContractHash (so a reload can tell reward configs
            // apart) but deliberately excluded from EncodingHash (so a policy trained under one
            // reward shape can be reused as another's warm start without a spurious mismatch).
            TacticalV2Config baseline = TacticalV2Config.Default();
            MlContract baselineContract = MlContract.CreateTacticalV2(baseline);

            TacticalV2Config changedReward = TacticalV2Config.Default();
            changedReward.ShapeScale = baseline.ShapeScale + 1f;
            changedReward.StepPenalty = baseline.StepPenalty + 1f;
            changedReward.ClosingWeight = baseline.ClosingWeight + 1f;
            changedReward.DrawCreditWeight = baseline.DrawCreditWeight + 1f;
            changedReward.PointsWeight = baseline.PointsWeight + 1f;
            MlContract changedRewardContract = MlContract.CreateTacticalV2(changedReward);

            Assert.That(changedRewardContract.ContractHash, Is.Not.EqualTo(baselineContract.ContractHash));
            Assert.That(changedRewardContract.EncodingHash, Is.EqualTo(baselineContract.EncodingHash));
        }
    }
}
