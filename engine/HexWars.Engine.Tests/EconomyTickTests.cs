using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class EconomyTickTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);

        // P1 begins next turn with a strength-1 generator. Output=10, UpkeepFactor=0.25 -> net +8 (10 - round(2.5)=3 -> 7).
        static GameState State(int p1StartPoints)
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) });
            var p0 = new PlayerState(PlayerId.Player0, 0,
                null, new[] { new Unit(1, PlayerId.Player0, new UnitStats(5,3,2,3,2,1,1,2,1), A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, p1StartPoints, null, null,
                new[] { new Generator(2, PlayerId.Player1, new HexCoord(1, 0), 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, upkeepFactor: 0.25);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
        }

        [Test]
        public void EndTurn_CreditsIncomeMinusUpkeep()
        {
            var r = GameEngine.Apply(State(5), new EndTurn(PlayerId.Player0));
            Assert.That(r.Success, Is.True);
            // income 10, upkeep round(10*0.25)=3 (2.5 -> 3 AwayFromZero), net +7 -> 5 + 7 = 12
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.EqualTo(12));
        }

        [Test]
        public void EndTurn_PointsNeverGoNegative()
        {
            // huge upkeep via UpkeepFactor would exceed points; clamp at 0
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains) });
            var p0 = new PlayerState(PlayerId.Player0, 0, null, new[] { new Unit(1, PlayerId.Player0, new UnitStats(5,3,2,3,2,1,1,2,1), A, 0) });
            var p1 = new PlayerState(PlayerId.Player1, 0, null, null,
                new[] { new Generator(2, PlayerId.Player1, new HexCoord(1, 0), 0, 3, 1.0) });
            var cfg = GameConfig.Default(biomesEnabled: false, generatorOutput: 10, upkeepFactor: 5.0); // upkeep 50 >> income 10
            var s = new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 99);
            var r = GameEngine.Apply(s, new EndTurn(PlayerId.Player0));
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.EqualTo(0));
        }
    }
}
