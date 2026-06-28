using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The lobby's output: a GameSetup round-trips over the wire, and GameFactory turns it into the right
    /// kind of start state (ruleset, board size, seeded army, starting points).
    /// </summary>
    public class GameSetupTests
    {
        private const PlayerId P0 = PlayerId.Player0;

        [Test]
        public void Wire_RoundTrips()
        {
            var s = new GameSetup(GameMode.Territory, 13, 9, 40, 7);
            Assert.That(GameSetup.Parse(s.ToWire()), Is.EqualTo(s));
        }

        [Test]
        public void Build_Annihilation_NotTerritory_HasArmy_ZeroPoints()
        {
            var st = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 0, 7));
            Assert.That(st.Config.TerritoryMode, Is.False);
            Assert.That(st.Player(P0).UnitsOnBoard.Count, Is.GreaterThan(0));
            Assert.That(st.Player(P0).Points, Is.EqualTo(0));
        }

        [Test]
        public void Build_Territory_HasTerritoryRules_AndStartingPoints()
        {
            var st = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 8, 40, 3));
            Assert.That(st.Config.TerritoryMode, Is.True);
            Assert.That(st.Player(P0).Points, Is.EqualTo(40));
        }

        [Test]
        public void Build_BoardSize_RespectsWidthHeight()
        {
            var st = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 0, 7));
            Assert.That(st.Board.Tiles.Count, Is.EqualTo(9 * 7));
        }

        [Test]
        public void Build_SameSeed_SameBoard()
        {
            var a = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 0, 42));
            var b = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 0, 42));
            foreach (var t in a.Board.Tiles)
                Assert.That(b.Board.TileAt(t.Coord).Elevation, Is.EqualTo(t.Elevation));
        }
    }
}
