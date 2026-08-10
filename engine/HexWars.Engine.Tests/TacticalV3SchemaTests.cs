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
        [Test]
        public void CapacityProfile_RejectsEveryNonPositiveBound()
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new TacticalV3CapacityProfile(
                maxCells: 0,
                maxUnits: 64,
                maxTemplates: 32,
                maxCapabilityDefinitions: 128,
                maxCapabilityAllocations: 2048,
                maxRules: 128,
                maxMemoryRecords: 64,
                maxRelations: 65536,
                maxCandidates: 32768));
        }

        [Test]
        public void StageOneConfig_RejectsFogAndWrongRewardBounds()
        {
            TacticalV2Config match = TacticalV2Config.Default();
            match.Game = TacticalV3Fixtures.CloneGame(match.Game, fogOfWar: true);
            var config = new TacticalV3Config(match, TacticalV3Fixtures.ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));

            Assert.That(config.Validate(), Has.Some.Contains("fog_of_war=false"));
            Assert.Throws<ArgumentException>(() =>
                new TacticalV3RewardConfig(+1f, -0.5f, 0.20f, 0.05f, 0.5f));
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
