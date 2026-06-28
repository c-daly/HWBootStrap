using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// Playable games must not box units in: terrain is inert (biomes off, so no impassable-cost tiles for a
    /// mv-2 unit) and no adjacent elevation jump exceeds the roster's vertical budget (2), so any neighbour
    /// is climbable. Regression guard for the "sniper can't move up a hex" bug.
    /// </summary>
    public class MovementBoardTests
    {
        [Test]
        public void Build_BiomesOff_AndNoUnclimbableCliffs()
        {
            var st = GameFactory.Build(GameSetup.Default);
            Assert.That(st.Config.BiomesEnabled, Is.False, "terrain should be inert in playable games");

            foreach (var t in st.Board.Tiles)
                foreach (var n in t.Coord.Neighbors())
                    if (st.Board.Contains(n))
                        Assert.That(System.Math.Abs(st.Board.TileAt(n).Elevation - t.Elevation), Is.LessThanOrEqualTo(2),
                            $"adjacent elevation jump too steep at ({t.Coord.Q},{t.Coord.R})->({n.Q},{n.R})");
        }
    }
}
