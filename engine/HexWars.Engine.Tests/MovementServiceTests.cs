using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class MovementServiceTests
    {
        private static UnitStats Mover(int movement, int verticalMovement) =>
            new UnitStats(health: 1, damage: 0, defense: 0,
                          movement: movement, verticalMovement: verticalMovement,
                          range: 0, rangeArc: 0, vision: 0, visionArc: 0);

        // A line of columns along r = 0: (q, elevation, terrain).
        private static Board Line(params (int q, int elev, TerrainType t)[] cols) =>
            new Board(cols.Select(c => new Tile(new HexCoord(c.q, 0), c.elev, c.t)).ToArray());

        private static GameState StateWith(Board board, params Unit[] units)
        {
            var p0 = new PlayerState(PlayerId.Player0, 0,
                unitsOnBoard: units.Where(u => u.Owner == PlayerId.Player0).ToArray());
            var p1 = new PlayerState(PlayerId.Player1, 0,
                unitsOnBoard: units.Where(u => u.Owner == PlayerId.Player1).ToArray());
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, 100);
        }

        [Test]
        public void FlatPlains_ReachesWithinHorizontalBudget_NotItsOwnCell()
        {
            var board = Line((0, 0, TerrainType.Plains), (1, 0, TerrainType.Plains),
                             (2, 0, TerrainType.Plains), (3, 0, TerrainType.Plains), (4, 0, TerrainType.Plains));
            var unit = new Unit(1, PlayerId.Player0, Mover(2, 0), new HexCoord(2, 0), 0);

            var reach = MovementService.ReachableTiles(StateWith(board, unit), unit);

            Assert.That(reach, Does.Contain(new HexCoord(1, 0)));
            Assert.That(reach, Does.Contain(new HexCoord(0, 0)));
            Assert.That(reach, Does.Contain(new HexCoord(3, 0)));
            Assert.That(reach, Does.Contain(new HexCoord(4, 0)));
            Assert.That(reach, Does.Not.Contain(new HexCoord(2, 0)));
        }

        [Test]
        public void ZeroVerticalMovement_CannotClimb()
        {
            var board = Line((0, 0, TerrainType.Plains), (1, 2, TerrainType.Plains));
            var unit = new Unit(1, PlayerId.Player0, Mover(5, 0), new HexCoord(0, 0), 0);
            Assert.That(MovementService.ReachableTiles(StateWith(board, unit), unit),
                        Does.Not.Contain(new HexCoord(1, 0)));
        }

        [Test]
        public void EnoughVerticalMovement_AllowsClimb()
        {
            var board = Line((0, 0, TerrainType.Plains), (1, 2, TerrainType.Plains));
            var unit = new Unit(1, PlayerId.Player0, Mover(1, 2), new HexCoord(0, 0), 0);
            Assert.That(MovementService.ReachableTiles(StateWith(board, unit), unit),
                        Does.Contain(new HexCoord(1, 0)));
        }

        [Test]
        public void InsufficientVerticalMovement_BlocksClimb()
        {
            var board = Line((0, 0, TerrainType.Plains), (1, 2, TerrainType.Plains));
            var unit = new Unit(1, PlayerId.Player0, Mover(5, 1), new HexCoord(0, 0), 0);
            Assert.That(MovementService.ReachableTiles(StateWith(board, unit), unit),
                        Does.Not.Contain(new HexCoord(1, 0)));
        }

        [Test]
        public void Descent_CostsNoVerticalMovement()
        {
            var board = Line((0, 2, TerrainType.Plains), (1, 0, TerrainType.Plains));
            var unit = new Unit(1, PlayerId.Player0, Mover(1, 0), new HexCoord(0, 0), 2);
            Assert.That(MovementService.ReachableTiles(StateWith(board, unit), unit),
                        Does.Contain(new HexCoord(1, 0)));
        }

        [Test]
        public void OccupiedTiles_AreNotReachable()
        {
            var board = Line((0, 0, TerrainType.Plains), (1, 0, TerrainType.Plains), (2, 0, TerrainType.Plains));
            var mover = new Unit(1, PlayerId.Player0, Mover(2, 0), new HexCoord(0, 0), 0);
            var blocker = new Unit(2, PlayerId.Player1, Mover(0, 0), new HexCoord(1, 0), 0);
            Assert.That(MovementService.ReachableTiles(StateWith(board, mover, blocker), mover),
                        Does.Not.Contain(new HexCoord(1, 0)));
        }
    }
}
