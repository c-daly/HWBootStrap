using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class MovementTooltipFormatterTests
    {
        [Test]
        public void FormatRoute_ReportsCostsAndRemainingBudgets()
        {
            var route = new MovementRoute(new[]
            {
                new HexCoord(0, 0),
                new HexCoord(1, 0),
            }, 2, 1, 3, 2);

            Assert.That(MovementTooltipFormatter.FormatRoute(route), Is.EqualTo(
                "Route: 2 move · 1 climb · leaves 3 move / 2 climb"));
        }

        [Test]
        public void FormatMovementStatus_AttackClosesMovementWithoutShowingRawRemainder()
        {
            var cell = new HexCoord(0, 0);
            var board = new Board(new[] { new Tile(cell, 0, TerrainType.Plains) });
            var stats = new UnitStats(health: 5, damage: 1, defense: 0,
                movement: 4, verticalMovement: 2, range: 1, rangeArc: 0,
                vision: 2, visionArc: 0);
            var unit = new Unit(1, PlayerId.Player0, stats, cell, 0);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit }),
                new PlayerState(PlayerId.Player1, 0),
            }, PlayerId.Player0, 1, 2,
            attackedUnitIds: new[] { unit.Id },
            movementSpent: new Dictionary<int, (int H, int V)> { [unit.Id] = (1, 0) });

            Assert.That(MovementTooltipFormatter.FormatMovementStatus(unit, state),
                Is.EqualTo("Movement ended by attack"));
        }

        [Test]
        public void FormatMovementStatus_ReportsActualRemainingBudgetsBeforeAttack()
        {
            var cell = new HexCoord(0, 0);
            var board = new Board(new[] { new Tile(cell, 0, TerrainType.Plains) });
            var stats = new UnitStats(health: 5, damage: 1, defense: 0,
                movement: 4, verticalMovement: 2, range: 1, rangeArc: 0,
                vision: 2, visionArc: 0);
            var unit = new Unit(1, PlayerId.Player0, stats, cell, 0);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { unit }),
                new PlayerState(PlayerId.Player1, 0),
            }, PlayerId.Player0, 1, 2,
            movementSpent: new Dictionary<int, (int H, int V)> { [unit.Id] = (1, 1) });

            Assert.That(MovementTooltipFormatter.FormatMovementStatus(unit, state),
                Is.EqualTo("Move 3/4   Climb 1/2"));
        }
    }
}
