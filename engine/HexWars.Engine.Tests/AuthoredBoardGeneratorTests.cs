using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AuthoredBoardGeneratorTests
    {
        [Test]
        public void Generate_ReturnsTheAuthoredBoard_RegardlessOfSeed()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 1, TerrainType.Forest),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });

            var gen = new AuthoredBoardGenerator(board);

            Assert.That(gen.Generate(1), Is.SameAs(board));
            Assert.That(gen.Generate(999), Is.SameAs(board));
        }
    }
}
