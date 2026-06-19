using System.Linq;
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

        [Test]
        public void DeploymentZone_IsPerPlayer()
        {
            var tiles = new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(5, 0), 0, TerrainType.Plains),
            };
            var board = new Board(tiles,
                zone0: new[] { new HexCoord(0, 0) },
                zone1: new[] { new HexCoord(5, 0) });

            Assert.That(board.IsInDeploymentZone(PlayerId.Player0, new HexCoord(0, 0)), Is.True);
            Assert.That(board.IsInDeploymentZone(PlayerId.Player0, new HexCoord(5, 0)), Is.False);
            Assert.That(board.IsInDeploymentZone(PlayerId.Player1, new HexCoord(5, 0)), Is.True);
        }

        [Test]
        public void Tiles_EnumeratesEveryColumn()
        {
            var board = TwoColumns();
            Assert.That(board.Tiles.Count, Is.EqualTo(2));
            Assert.That(board.Tiles.Select(t => t.Elevation).OrderBy(e => e).ToArray(),
                        Is.EqualTo(new[] { 0, 2 }));
        }
    }
}
