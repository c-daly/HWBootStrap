using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class EconomyTests
    {
        private static GameState StateWith(params Generator[] gens)
        {
            var p0 = new PlayerState(PlayerId.Player0, 0, generators: gens);
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, 100);
        }

        [Test]
        public void Income_CountsLivingGeneratorsTimesOutput()
        {
            var g1 = new Generator(1, PlayerId.Player0, new HexCoord(0, 0), 0, 3);
            var g2 = new Generator(2, PlayerId.Player0, new HexCoord(1, 0), 0, 3);
            Assert.That(Economy.Income(StateWith(g1, g2), PlayerId.Player0), Is.EqualTo(2)); // 2 * output 1
        }

        [Test]
        public void Income_IgnoresDeadGenerators()
        {
            var alive = new Generator(1, PlayerId.Player0, new HexCoord(0, 0), 0, 3);
            var dead = new Generator(2, PlayerId.Player0, new HexCoord(1, 0), 0, 3).WithDamage(3);
            Assert.That(Economy.Income(StateWith(alive, dead), PlayerId.Player0), Is.EqualTo(1));
        }

        [Test]
        public void Income_IsZero_WithNoGenerators()
        {
            Assert.That(Economy.Income(StateWith(), PlayerId.Player0), Is.EqualTo(0));
        }
    }
}
