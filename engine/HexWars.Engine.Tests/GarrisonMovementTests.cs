using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GarrisonMovementTests
    {
        [Test]
        public void Unit_CanReachAGeneratorHex()
        {
            var a = new HexCoord(0, 0);
            var b = new HexCoord(1, 0);
            var board = new Board(new[]
            {
                new Tile(a, 0, TerrainType.Plains),
                new Tile(b, 0, TerrainType.Plains),
            });
            var mover = new Unit(1, PlayerId.Player0, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), a, 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, null, new[] { mover });
            // enemy generator sits on b
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(2, PlayerId.Player1, b, 0, 3) });
            var s = new GameState(board, GameConfig.Default(biomesEnabled: false), new[] { p0, p1 },
                PlayerId.Player0, 1, 9);

            var reachable = MovementService.ReachableTiles(s, mover);
            Assert.That(reachable, Does.Contain(b), "a unit may garrison a generator hex");
        }
    }
}
