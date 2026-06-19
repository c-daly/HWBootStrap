using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BoardGenerationTests
    {
        private const int W = 8, H = 6, MaxElev = 4, ZoneDepth = 2;
        private static RandomBoardGenerator Gen() =>
            new RandomBoardGenerator(new BoardGenConfig(W, H, MaxElev, ZoneDepth));

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
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++)
                {
                    var c = HexLayout.OffsetToAxial(col, row);
                    Assert.That(b.TileAt(c).Elevation, Is.EqualTo(a.TileAt(c).Elevation));
                    Assert.That(b.TileAt(c).Terrain, Is.EqualTo(a.TileAt(c).Terrain));
                }
        }

        [Test]
        public void Generate_IsMirrorSymmetric_InTerrainAndElevation()
        {
            var board = Gen().Generate(99);
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++)
                {
                    var c = HexLayout.OffsetToAxial(col, row);
                    var m = HexLayout.OffsetToAxial(W - 1 - col, H - 1 - row);
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
            for (int row = 0; row < H; row++)
                for (int col = 0; col < W; col++)
                    Assert.That(board.TileAt(HexLayout.OffsetToAxial(col, row)).Elevation, Is.InRange(0, MaxElev));
        }

        [Test]
        public void Generate_IsMostlyFlatGround()
        {
            foreach (var seed in new[] { 1, 42, 123, 777 })
            {
                var board = Gen().Generate(seed);
                int flat = board.Tiles.Count(t => t.Elevation == 0);
                Assert.That(flat, Is.GreaterThanOrEqualTo(board.TileCount / 2),
                    $"seed {seed}: expected a majority of flat (elevation 0) tiles");
            }
        }

        [Test]
        public void Generate_GroundUnit_CanRoamALargeFlatArea()
        {
            var board = Gen().Generate(123);
            var start = board.Tiles.First(t => t.Elevation == 0).Coord;
            var groundUnit = new Unit(1, PlayerId.Player0,
                new UnitStats(1, 0, 0, movement: 50, verticalMovement: 0, 0, 0, 0, 0), start, 0);
            var state = new GameState(board, GameConfig.Default(),
                new[] { new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { groundUnit }),
                        new PlayerState(PlayerId.Player1, 0) },
                PlayerId.Player0, 1, 100);

            var reach = MovementService.ReachableTiles(state, groundUnit);
            Assert.That(reach.Count, Is.GreaterThanOrEqualTo(board.TileCount / 3),
                "a 0-vertical-movement unit should roam a large connected flat area");
        }

        [Test]
        public void Config_FlatChance1_ProducesAllFlatGround()
        {
            var board = new RandomBoardGenerator(
                new BoardGenConfig(6, 4, maxElevation: 4, zoneDepth: 1, flatChance: 1.0)).Generate(5);
            Assert.That(board.Tiles.All(t => t.Elevation == 0), Is.True);
        }

        [Test]
        public void OffsetToAxial_StaggersOddColumns()
        {
            Assert.That(HexLayout.OffsetToAxial(0, 0), Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(HexLayout.OffsetToAxial(1, 0), Is.EqualTo(new HexCoord(1, 0)));
            Assert.That(HexLayout.OffsetToAxial(2, 0), Is.EqualTo(new HexCoord(2, -1)));
        }

        [Test]
        public void Generate_HasRectangularFootprint_NotSheared()
        {
            var board = Gen().Generate(7);
            double minZ = double.MaxValue, maxZ = double.MinValue;
            foreach (var t in board.Tiles)
            {
                var (_, z) = HexLayout.ToWorld(t.Coord, 1.0);
                minZ = System.Math.Min(minZ, z);
                maxZ = System.Math.Max(maxZ, z);
            }
            double sqrt3 = System.Math.Sqrt(3.0);
            // rectangle: z-span ~ (H-1+0.5) rows. A sheared parallelogram would be ~ (H-1+(W-1)/2) rows.
            Assert.That(maxZ - minZ, Is.LessThan(H * sqrt3));
        }

        [Test]
        public void Config_TerrainWeights_DriveTerrainSelection()
        {
            var allPlains = new RandomBoardGenerator(new BoardGenConfig(6, 4, flatChance: 1.0,
                plainsWeight: 1, forestWeight: 0, roughWeight: 0, waterWeight: 0)).Generate(5);
            Assert.That(allPlains.Tiles.All(t => t.Terrain == TerrainType.Plains), Is.True);

            var allWater = new RandomBoardGenerator(new BoardGenConfig(6, 4, flatChance: 1.0,
                plainsWeight: 0, forestWeight: 0, roughWeight: 0, waterWeight: 1)).Generate(5);
            Assert.That(allWater.Tiles.All(t => t.Terrain == TerrainType.Water), Is.True);
        }
    }
}
