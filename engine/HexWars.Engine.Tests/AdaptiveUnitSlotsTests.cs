using System;
using System.Collections.Generic;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AdaptiveUnitSlotsTests
    {
        [Test]
        public void Sync_ReleasesMissingUnitAndGivesLowestSlotToReinforcement()
        {
            var slots = new AdaptiveUnitSlots(4);
            slots.Sync(State(10, 11, 12), PlayerId.Player0);
            slots.Sync(State(10, 12), PlayerId.Player0);

            slots.Sync(State(10, 12, 20), PlayerId.Player0);

            Assert.That(slots.UnitIdAt(0), Is.EqualTo(10));
            Assert.That(slots.UnitIdAt(1), Is.EqualTo(20));
            Assert.That(slots.UnitIdAt(2), Is.EqualTo(12));
            Assert.That(slots.UnitIdAt(3), Is.EqualTo(-1));
            Assert.That(slots.SlotOf(20), Is.EqualTo(1));
            Assert.That(slots.HasFreeSlot, Is.True);
            Assert.That(slots.HasOverflow, Is.False);
        }

        [Test]
        public void Sync_InitialAssignmentIsDeterministicByEntityId()
        {
            var slots = new AdaptiveUnitSlots(4);

            slots.Sync(State(40, 10, 30, 20), PlayerId.Player0);

            Assert.That(new[]
            {
                slots.UnitIdAt(0), slots.UnitIdAt(1), slots.UnitIdAt(2), slots.UnitIdAt(3),
            }, Is.EqualTo(new[] { 10, 20, 30, 40 }));
        }

        [Test]
        public void Sync_OverflowIsExplicitAndNeverRemapsExistingLivingUnits()
        {
            var slots = new AdaptiveUnitSlots(2);
            slots.Sync(State(20, 30), PlayerId.Player0);

            slots.Sync(State(10, 20, 30, 40), PlayerId.Player0);

            Assert.That(slots.UnitIdAt(0), Is.EqualTo(20));
            Assert.That(slots.UnitIdAt(1), Is.EqualTo(30));
            Assert.That(slots.HasFreeSlot, Is.False);
            Assert.That(slots.HasOverflow, Is.True);
            Assert.That(slots.OverflowCount, Is.EqualTo(2));
            Assert.That(slots.SlotOf(10), Is.EqualTo(-1));
            Assert.That(slots.SlotOf(40), Is.EqualTo(-1));
        }

        [Test]
        public void Sync_IgnoresDeadUnitsEvenWhenStillPresentInState()
        {
            var slots = new AdaptiveUnitSlots(2);
            slots.Sync(State(10, 20), PlayerId.Player0);

            slots.Sync(StateWithDeadUnit(10, 20), PlayerId.Player0);

            Assert.That(slots.UnitIdAt(0), Is.EqualTo(-1));
            Assert.That(slots.UnitIdAt(1), Is.EqualTo(20));
        }

        [Test]
        public void Constructor_RejectsNegativeCapacity()
        {
            Assert.That(() => new AdaptiveUnitSlots(-1), Throws.TypeOf<ArgumentOutOfRangeException>());
        }

        [Test]
        public void ZeroCapacity_ReportsEveryLivingUnitAsOverflow()
        {
            var slots = new AdaptiveUnitSlots(0);

            slots.Sync(State(3, 1, 2), PlayerId.Player0);

            Assert.That(slots.Capacity, Is.EqualTo(0));
            Assert.That(slots.HasFreeSlot, Is.False);
            Assert.That(slots.HasOverflow, Is.True);
            Assert.That(slots.OverflowCount, Is.EqualTo(3));
            Assert.That(slots.UnitIdAt(0), Is.EqualTo(-1));
        }

        private static GameState State(params int[] unitIds)
        {
            var board = Board();
            var units = new List<Unit>();
            foreach (int id in unitIds)
                units.Add(new Unit(id, PlayerId.Player0, Stats(), new HexCoord(0, 0), 0));
            return Game(board, units);
        }

        private static GameState StateWithDeadUnit(int deadId, int livingId)
        {
            var board = Board();
            var dead = new Unit(deadId, PlayerId.Player0, Stats(), new HexCoord(0, 0), 0).WithDamage(2);
            var living = new Unit(livingId, PlayerId.Player0, Stats(), new HexCoord(0, 0), 0);
            return Game(board, new[] { dead, living });
        }

        private static GameState Game(Board board, IReadOnlyList<Unit> units)
        {
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: units),
                new PlayerState(PlayerId.Player1, 0),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 100);
        }

        private static Board Board() => new Board(new[]
        {
            new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
        }, new[] { new HexCoord(0, 0) }, Array.Empty<HexCoord>());

        private static UnitStats Stats() => new UnitStats(2, 1, 0, 1, 0, 0, 0, 1, 0);
    }
}
