using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class MovementPresentationRouteTests
    {
        [Test]
        public void Resolve_UsesLegalDetourInsteadOfBlockedStraightLine()
        {
            var start = new HexCoord(0, 0);
            var blocked = new HexCoord(1, 0);
            var aroundFirst = new HexCoord(0, 1);
            var aroundSecond = new HexCoord(1, 1);
            var destination = new HexCoord(2, 0);
            var board = new Board(new[]
            {
                new Tile(start, 0, TerrainType.Plains),
                new Tile(blocked, 0, TerrainType.Plains),
                new Tile(aroundFirst, 0, TerrainType.Plains),
                new Tile(aroundSecond, 0, TerrainType.Plains),
                new Tile(destination, 0, TerrainType.Plains),
            });
            var moverStats = new UnitStats(health: 5, damage: 1, defense: 0,
                movement: 3, verticalMovement: 0, range: 1, rangeArc: 0,
                vision: 2, visionArc: 0);
            var blockerStats = new UnitStats(health: 5, damage: 0, defense: 0,
                movement: 0, verticalMovement: 0, range: 0, rangeArc: 0,
                vision: 0, visionArc: 0);
            var mover = new Unit(1, PlayerId.Player0, moverStats, start, 0);
            var blocker = new Unit(2, PlayerId.Player1, blockerStats, blocked, 0);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { mover }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { blocker }),
            }, PlayerId.Player0, 1, 3);
            var command = new MoveUnit(PlayerId.Player0, mover.Id, destination);

            var resolved = MovementPresentationRoute.Resolve(state, command);
            var authoritative = MovementService.Routes(state, mover)[destination].Cells;

            Assert.That(resolved, Is.EqualTo(authoritative));
            Assert.That(resolved, Is.EqualTo(new[] { start, aroundFirst, aroundSecond, destination }));
            Assert.That(resolved, Does.Not.Contain(blocked));
        }

        [Test]
        public void Resolve_ReturnsEmptyWhenUnitOrRouteIsMissing()
        {
            var start = new HexCoord(0, 0);
            var board = new Board(new[] { new Tile(start, 0, TerrainType.Plains) });
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0),
                new PlayerState(PlayerId.Player1, 0),
            }, PlayerId.Player0, 1, 2);

            Assert.That(MovementPresentationRoute.Resolve(
                state, new MoveUnit(PlayerId.Player0, 999, new HexCoord(4, 4))), Is.Empty);
        }
    }
}
