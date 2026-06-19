using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardTests
    {
        private static Board TwoColumns()
        {
            var tiles = new[]
            {
                new Tile(new HexCoord(0, 0), elevation: 0, terrain: TerrainType.Plains),
                new Tile(new HexCoord(1, 0), elevation: 2, terrain: TerrainType.Forest),
            };
            return new Board(tiles);
        }

        [Test]
        public void TileAt_ReturnsTheTileForAColumn()
        {
            var tile = TwoColumns().TileAt(new HexCoord(1, 0));
            Assert.That(tile.Elevation, Is.EqualTo(2));
            Assert.That(tile.Terrain, Is.EqualTo(TerrainType.Forest));
        }

        [Test]
        public void Contains_IsTrueForExistingColumn_FalseOtherwise()
        {
            var board = TwoColumns();
            Assert.That(board.Contains(new HexCoord(0, 0)), Is.True);
            Assert.That(board.Contains(new HexCoord(5, 5)), Is.False);
        }

        [Test]
        public void TileCount_ReflectsAllColumns()
        {
            Assert.That(TwoColumns().TileCount, Is.EqualTo(2));
        }
    }
}
