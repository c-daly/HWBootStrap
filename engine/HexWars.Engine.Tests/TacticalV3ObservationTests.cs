using System;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3ObservationTests
    {
        [Test]
        public void Observe_RepresentsMechanicsNotRosterNamesOrEngineIds()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Cells, Has.Count.EqualTo(fixture.State.Board.TileCount));
            Assert.That(typeof(TacticalV3UnitToken).GetProperty("EngineId"), Is.Null);
            Assert.That(typeof(TacticalV3UnitToken).GetProperty("Name"), Is.Null);
            Assert.That(typeof(TacticalV3TemplateToken).GetProperty("EngineId"), Is.Null);
            Assert.That(typeof(TacticalV3TemplateToken).GetProperty("Name"), Is.Null);
            Assert.Throws<NotSupportedException>(() =>
                ((System.Collections.IList)observation.Cells)[0] = observation.Cells[0]);
        }

        [Test]
        public void Observe_OrdersSeatRelativeRowsAndIncludesAllNineCapabilities()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);
            TacticalV2Layout layout = new TacticalV2Layout(TacticalV3Fixtures.Config().Match);

            Assert.That(observation.Cells.Select(cell => (cell.Q, cell.R)),
                Is.EqualTo(layout.Cells.Select(cell => (cell.Q, cell.R))));
            Assert.That(observation.Units.Take(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(observation.Units.Skip(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Opponent));
            Assert.That(observation.Units[0].CurrentHp, Is.EqualTo(observation.Units[0].MaxHp));
            Assert.That(observation.Units[0].Cell.Table, Is.EqualTo(TacticalV3TableKind.Cells));
            Assert.That(observation.Units[0].Moved, Is.False);
            Assert.That(observation.Units[0].Attacked, Is.False);
            Assert.That(observation.Units[0].HorizontalMovementSpent, Is.EqualTo(0));
            Assert.That(observation.Units[0].VerticalMovementSpent, Is.EqualTo(0));
            Assert.That(observation.Units[0].PointCost, Is.EqualTo(
                fixture.State.Player(PlayerId.Player0).UnitsOnBoard[0].Stats.PointCost));
            Assert.That(observation.Templates.Take(5).Select(template => template.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(observation.Templates.Skip(5).Select(template => template.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Opponent));
            Assert.That(observation.CapabilityAllocations.Count,
                Is.EqualTo(9 * (observation.Units.Count + observation.Templates.Count)));
            Assert.That(observation.CapabilityAllocations.Take(9).Select(row => row.Capability), Is.EqualTo(
                TacticalV3Capabilities.All.Select(definition => definition.Kind)));
        }

        [Test]
        public void Observe_ExposesExplicitRulesAndOnlyValidRelationsWithEmptyMemory()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Rules.Select(rule => rule.Kind), Is.EqualTo(new[]
            {
                TacticalV3RuleKind.WinConditions,
                TacticalV3RuleKind.Round,
                TacticalV3RuleKind.RoundCap,
                TacticalV3RuleKind.ActionsPerTurn,
                TacticalV3RuleKind.StartingPoints,
                TacticalV3RuleKind.SelfPoints,
                TacticalV3RuleKind.OpponentPoints,
                TacticalV3RuleKind.DamageFloor,
                TacticalV3RuleKind.DamageHighGroundBonus,
                TacticalV3RuleKind.RangeHighGroundBonus,
                TacticalV3RuleKind.BountyRate,
                TacticalV3RuleKind.DeployCostMultiplier,
                TacticalV3RuleKind.FogOfWar,
                TacticalV3RuleKind.MaxDesignPointCost,
                TacticalV3RuleKind.DesignFee,
            }));
            Assert.That(Rule(observation, TacticalV3RuleKind.WinConditions).IntValue,
                Is.EqualTo((int)WinBy.Annihilation));
            Assert.That(Rule(observation, TacticalV3RuleKind.FogOfWar).BoolValue, Is.False);
            Assert.That(observation.Memory, Is.Empty);
            Assert.That(observation.Relations.Any(row => row.Kind == TacticalV3RelationKind.Neighbor), Is.True);
            Assert.That(observation.Relations.Count(row => row.Kind == TacticalV3RelationKind.Occupies),
                Is.EqualTo(observation.Units.Count));
            Assert.That(observation.Relations.Count(row => row.Kind == TacticalV3RelationKind.HasCapability),
                Is.EqualTo(observation.CapabilityAllocations.Count));
            Assert.That(observation.Relations.All(row => row.Source.Row >= 0 && row.Target.Row >= 0), Is.True);
        }

        [Test]
        public void Observe_SwapsRelativeOwnershipAndPointsWithoutChangingAuthoritativeState()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player1, EmptyObservationMemory.Instance);

            Assert.That(observation.Units.Take(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(Rule(observation, TacticalV3RuleKind.SelfPoints).IntValue,
                Is.EqualTo(fixture.State.Player(PlayerId.Player1).Points));
            Assert.That(Rule(observation, TacticalV3RuleKind.OpponentPoints).IntValue,
                Is.EqualTo(fixture.State.Player(PlayerId.Player0).Points));
            Assert.That(fixture.State.Player(PlayerId.Player0).Id, Is.EqualTo(PlayerId.Player0));
            Assert.That(fixture.State.Player(PlayerId.Player1).Id, Is.EqualTo(PlayerId.Player1));
        }

        [Test]
        public void Observe_ThrowsBeforeReturningWhenCapacityIsExceeded()
        {
            TacticalV3Config config = new TacticalV3Config(
                TacticalV3Fixtures.Match(),
                new TacticalV3CapacityProfile(1, 64, 32, 128, 2048, 128, 64, 65536, 32768),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));
            TacticalV2Layout layout = new TacticalV2Layout(config.Match);
            TacticalV3SeatObservationSource source = new TacticalV3SeatObservationSource(config);

            Assert.Throws<InvalidOperationException>(() => source.Observe(
                layout.NewGame(17).State, PlayerId.Player0, EmptyObservationMemory.Instance));
        }

        [Test]
        public void Observe_ExcludesDeadUnitsWithoutDisturbingCapabilityAllocationOwnership()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            PlayerState original = fixture.State.Player(PlayerId.Player0);
            Unit[] units = original.UnitsOnBoard.ToArray();
            units[0] = units[0].WithDamage(units[0].CurrentHp);
            var player0 = new PlayerState(PlayerId.Player0, original.Points, original.Barracks, units,
                original.Generators, original.DestroyedValue);
            var state = new GameState(fixture.State.Board, fixture.State.Config,
                new[] { player0, fixture.State.Player(PlayerId.Player1) }, fixture.State.ActivePlayer,
                fixture.State.Round, fixture.State.NextEntityId, fixture.State.IsGameOver, fixture.State.Winner,
                fixture.State.MovedUnitIds, fixture.State.AttackedUnitIds, fixture.State.MovementSpent);

            TacticalV3Observation observation = fixture.Source.Observe(
                state, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Units, Has.Count.EqualTo(5));
            Assert.That(observation.CapabilityAllocations.Count,
                Is.EqualTo(9 * (observation.Units.Count + observation.Templates.Count)));
        }

        private static TacticalV3RuleToken Rule(TacticalV3Observation observation, TacticalV3RuleKind kind) =>
            observation.Rules.Single(rule => rule.Kind == kind);
    }
}
