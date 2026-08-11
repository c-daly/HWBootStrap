using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3SchemaTests
    {
        [TestCase(0, 0)]
        [TestCase(0, -1)]
        [TestCase(1, 0)]
        [TestCase(1, -1)]
        [TestCase(2, 0)]
        [TestCase(2, -1)]
        [TestCase(3, 0)]
        [TestCase(3, -1)]
        [TestCase(4, 0)]
        [TestCase(4, -1)]
        [TestCase(5, 0)]
        [TestCase(5, -1)]
        [TestCase(6, 0)]
        [TestCase(6, -1)]
        [TestCase(7, 0)]
        [TestCase(7, -1)]
        [TestCase(8, 0)]
        [TestCase(8, -1)]
        public void CapacityProfile_RejectsEveryNonPositiveBound(int invalidIndex, int invalidValue)
        {
            var capacities = new[] { 512, 64, 32, 128, 2048, 128, 64, 65536, 32768 };
            capacities[invalidIndex] = invalidValue;

            Assert.Throws<ArgumentOutOfRangeException>(() => new TacticalV3CapacityProfile(
                capacities[0], capacities[1], capacities[2], capacities[3], capacities[4],
                capacities[5], capacities[6], capacities[7], capacities[8]));
        }

        [Test]
        public void StageOneConfig_RejectsFogAndWrongRewardBounds()
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.Game = TacticalV3Fixtures.CloneGame(match.Game, fogOfWar: true);
            var config = new TacticalV3Config(match, TacticalV3Fixtures.ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

            Assert.That(config.Validate(), Has.Some.Contains("fog_of_war=false"));
            Assert.Throws<ArgumentException>(() => new TacticalV3RewardConfig(+0.9f, -1f, 0.20f, 0.05f, 0.5f));
            Assert.Throws<ArgumentException>(() => new TacticalV3RewardConfig(+1f, -0.5f, 0.20f, 0.05f, 0.5f));
            Assert.Throws<ArgumentException>(() => new TacticalV3RewardConfig(+1f, -1f, 0.19f, 0.05f, 0.5f));
            Assert.Throws<ArgumentException>(() => new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.04f, 0.5f));
            Assert.Throws<ArgumentException>(() => new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.4f));
        }

        [Test]
        public void StageOneConfig_RequiresAnnihilationAndDisabledGenerators()
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.Game = GameConfig.Default(
                biomesEnabled: false,
                winConditions: WinBy.Score,
                generatorsEnabled: true,
                fixedTemplateCount: BarracksCatalog.DefaultTemplates.Count,
                templateSlotCount: BarracksCatalog.DefaultTemplates.Count);

            TacticalV3Config config = new TacticalV3Config(
                match, TacticalV3Fixtures.ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

            Assert.That(config.Validate(), Has.Some.Contains("requires annihilation"));
            Assert.That(config.Validate(), Has.Some.Contains("requires generators disabled"));
        }

        [TestCase(3, false)]
        [TestCase(int.MaxValue, true)]
        public void StageOneConfig_RejectsCaptureMechanicsBeforeEnvironmentConstruction(
            int captureCost, bool territoryMode)
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.Game = TacticalV3Fixtures.CloneGame(
                match.Game, captureCost: captureCost, territoryMode: territoryMode);
            TacticalV3Config config = Config(match);

            Assert.That(config.Validate(), Has.Some.Contains("capture"));
            Assert.Throws<ArgumentException>(() => new TacticalV3SeatObservationSource(config));
            Assert.Throws<ArgumentException>(() => new TacticalV3DuelEnv(config));
        }

        [Test]
        public void StageOneConfig_RejectsPointsThatCouldReachTheCaptureSentinel()
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.Game = TacticalV3Fixtures.CloneGame(
                match.Game, startingPoints: int.MaxValue - 1, bountyRate: 0.5);
            TacticalV3Config config = Config(match);

            Assert.That(config.Validate(),
                Has.Some.Contains("points").And.Some.Contains("capture"));
            Assert.Throws<ArgumentException>(() => new TacticalV3DuelEnv(config));
        }

        [Test]
        public void StageOneConfig_RejectsZeroHealthAndOverflowingTemplatePointCost()
        {
            TacticalV2Config source = TacticalV2Config.Default();
            var zeroHealth = new TacticalV2Template("zero-health",
                new UnitTemplate("Zero", new UnitStats(0, 1, 0, 1, 0, 1, 0, 1, 0)));
            var overflowing = new TacticalV2Template("overflowing",
                new UnitTemplate("Overflow", new UnitStats(
                    int.MaxValue, 1, 0, 0, 0, 0, 0, 0, 0)));
            TacticalV2Config match = TacticalV3Fixtures.CloneMatch(
                source, templates: Array.AsReadOnly(new[] { zeroHealth, overflowing }));
            TacticalV3Config config = Config(match);

            Assert.That(config.Validate(),
                Has.Some.Contains("health").And.Some.Contains("point cost"));
            Assert.Throws<ArgumentException>(() => new TacticalV3SeatObservationSource(config));
        }

        private static TacticalV3Config Config(TacticalV2Config match) => new TacticalV3Config(
            match,
            TacticalV3Fixtures.ExperimentalCapacity(),
            new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

        [Test]
        public void CapabilityCatalog_IsStableAndContainsNoRosterIdentity()
        {
            Assert.That(TacticalV3Capabilities.All.Select(x => x.Kind), Is.EqualTo(new[]
            {
                TacticalV3CapabilityKind.Health,
                TacticalV3CapabilityKind.Damage,
                TacticalV3CapabilityKind.Defense,
                TacticalV3CapabilityKind.Movement,
                TacticalV3CapabilityKind.VerticalMovement,
                TacticalV3CapabilityKind.Range,
                TacticalV3CapabilityKind.RangeArc,
                TacticalV3CapabilityKind.Vision,
                TacticalV3CapabilityKind.VisionArc,
            }));
            Assert.That(typeof(TacticalV3CapabilityDefinition).GetProperty("Name"), Is.Null);
            Assert.That(typeof(TacticalV3CapabilityRelation).GetProperty("Name"), Is.Null);
        }

        [Test]
        public void CapabilityCatalog_ExposesImmutableSemanticRelationSnapshot()
        {
            IReadOnlyList<TacticalV3CapabilityRelation> relations = TacticalV3Capabilities.Relations;

            Assert.That(relations.Any(relation =>
                relation.Source == TacticalV3CapabilityKind.Damage &&
                relation.Kind == TacticalV3CapabilityRelationKind.Opposes &&
                relation.Target == TacticalV3CapabilityKind.Health), Is.True);
            Assert.That(relations.Any(relation =>
                relation.Source == TacticalV3CapabilityKind.Defense &&
                relation.Kind == TacticalV3CapabilityRelationKind.Reduces &&
                relation.Target == TacticalV3CapabilityKind.Damage), Is.True);
            Assert.That(relations.Count(relation =>
                relation.Kind == TacticalV3CapabilityRelationKind.EnablesAction &&
                relation.Target == TacticalV3ActionKind.Attack), Is.EqualTo(2));
            Assert.Throws<NotSupportedException>(() =>
                ((IList<TacticalV3CapabilityRelation>)relations)[0] = relations[0]);
        }

        [Test]
        public void TokenReferences_AreTableKindAndIntegerRowOnly()
        {
            var token = new TacticalV3TokenRef(TacticalV3TableKind.Units, 7);

            Assert.That(token.Table, Is.EqualTo(TacticalV3TableKind.Units));
            Assert.That(token.Row, Is.EqualTo(7));
            Assert.That(typeof(TacticalV3TokenRef).GetProperty(nameof(TacticalV3TokenRef.Table))!.PropertyType,
                Is.EqualTo(typeof(TacticalV3TableKind)));
            Assert.That(typeof(TacticalV3TokenRef).GetProperty(nameof(TacticalV3TokenRef.Row))!.PropertyType,
                Is.EqualTo(typeof(int)));
            Assert.That(typeof(TacticalV3TokenRef).GetProperty("Name"), Is.Null);
        }
    }
}
