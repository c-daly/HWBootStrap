using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ScoreTests
    {
        [Test]
        public void Score_IsWeightedSum_OfKillsPointsArmyTerritory()
        {
            var unit = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1); // PointCost computed from stats
            var p0 = new PlayerState(PlayerId.Player0, 10, null,
                new[] { new Unit(1, PlayerId.Player0, unit, new HexCoord(0, 0), 0) }).WithDestroyed(4);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) })
                .WithControl(new HexCoord(0, 0), PlayerId.Player0);
            // all weights default to 1
            var s = new GameState(board, GameConfig.Default(), new[] { p0, p1 },
                PlayerId.Player0, 1, 9);

            int expected = 4 /*kills*/ + 10 /*points*/ + unit.PointCost /*army*/ + 1 /*territory*/;
            Assert.That(WinCheck.Score(s, PlayerId.Player0), Is.EqualTo(expected));
        }
    }
}
