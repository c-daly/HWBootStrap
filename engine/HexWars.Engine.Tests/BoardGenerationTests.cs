using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardGenerationTests
    {
        private const int W = 8, H = 6, MaxElev = 4, ZoneDepth = 2;
        private static RandomBoardGenerator Gen() => new RandomBoardGenerator(W, H, MaxElev, ZoneDepth);

        [Test]
        public void Generate_ProducesEveryColumn()
        {
            Assert.That(Gen().Generate(seed: 123).TileCount, Is.EqualTo(W * H));
        }

        [Test]
        public void Generate_IsDeterministic_ForTheSameSeed()
        {
            var a = Gen().Generate(123);
            var b = Gen().Generate(123);
            for (int r = 0; r < H; r++)
                for (int q = 0; q < W; q++)
                {
                    var c = new HexCoord(q, r);
                    Assert.That(b.TileAt(c).Elevation, Is.EqualTo(a.TileAt(c).Elevation));
                    Assert.That(b.TileAt(c).Terrain, Is.EqualTo(a.TileAt(c).Terrain));
                }
        }

        [Test]
        public void Generate_IsMirrorSymmetric_InTerrainAndElevation()
        {
            var board = Gen().Generate(99);
            for (int r = 0; r < H; r++)
                for (int q = 0; q < W; q++)
                {
                    var c = new HexCoord(q, r);
                    var m = new HexCoord(W - 1 - q, H - 1 - r);
                    Assert.That(board.TileAt(c).Elevation, Is.EqualTo(board.TileAt(m).Elevation));
                    Assert.That(board.TileAt(c).Terrain, Is.EqualTo(board.TileAt(m).Terrain));
                }
        }

        [Test]
        public void Generate_GivesEachPlayerANonEmptyDisjointZone()
        {
            var board = Gen().Generate(5);
            var z0 = board.DeploymentZone(PlayerId.Player0);
            var z1 = board.DeploymentZone(PlayerId.Player1);
            Assert.That(z0, Is.Not.Empty);
            Assert.That(z1, Is.Not.Empty);
            Assert.That(z0.Intersect(z1).Any(), Is.False);
        }

        [Test]
        public void Generate_ElevationsWithinRange()
        {
            var board = Gen().Generate(42);
            for (int r = 0; r < H; r++)
                for (int q = 0; q < W; q++)
                    Assert.That(board.TileAt(new HexCoord(q, r)).Elevation, Is.InRange(0, MaxElev));
        }
    }
}
