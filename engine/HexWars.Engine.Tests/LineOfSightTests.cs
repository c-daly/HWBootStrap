using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class LineOfSightTests
    {
        // A straight row of columns along +Q with the given elevations.
        private static Board Row(params int[] elevations)
        {
            var tiles = new Tile[elevations.Length];
            for (int q = 0; q < elevations.Length; q++)
                tiles[q] = new Tile(new HexCoord(q, 0), elevations[q], TerrainType.Plains);
            return new Board(tiles);
        }

        private static readonly HexCoord A = new HexCoord(0, 0);
        private static HexCoord End(int q) => new HexCoord(q, 0);

        [Test]
        public void FlatGround_IsClear() =>
            Assert.That(LineOfSight.IsClear(Row(0, 0, 0, 0), A, 0, End(3), 0), Is.True);

        [Test]
        public void TallColumnBetween_Blocks() =>
            Assert.That(LineOfSight.IsClear(Row(0, 3, 0, 0), A, 0, End(3), 0), Is.False);

        [Test]
        public void HighGround_SeesOverLowHill() =>
            Assert.That(LineOfSight.IsClear(Row(0, 1, 0, 0), A, 4, End(3), 0), Is.True);

        [Test]
        public void Adjacent_IsAlwaysClear() =>
            Assert.That(LineOfSight.IsClear(Row(0, 9), A, 0, End(1), 9), Is.True);
    }
}
