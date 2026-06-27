using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class EconomyIncomeTests
    {
        // One player with two generators (strength 1.0 and 0.5), GeneratorOutput=10, UpkeepFactor=0.25.
        static GameState State()
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            var gens = new[]
            {
                new Generator(1, PlayerId.Player0, new HexCoord(0, 0), 0, 3, 1.0),
                new Generator(2, PlayerId.Player0, new HexCoord(1, 0), 0, 3, 0.5),
            };
            var p0 = new PlayerState(PlayerId.Player0, 0, null, null, gens);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = new GameConfig(new System.Collections.Generic.Dictionary<TerrainType, TerrainDef>(),
                generatorOutput: 10, upkeepFactor: 0.25);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 9);
        }

        [Test]
        public void Income_SumsOutputTimesStrength()
        {
            // 10*1.0 + 10*0.5 = 15
            Assert.That(Economy.Income(State(), PlayerId.Player0), Is.EqualTo(15));
        }

        [Test]
        public void Upkeep_IsFractionOfIncome()
        {
            // round(15 * 0.25) = 4 (AwayFromZero: 3.75 -> 4)
            Assert.That(Economy.Upkeep(State(), PlayerId.Player0), Is.EqualTo(4));
        }
    }
}
