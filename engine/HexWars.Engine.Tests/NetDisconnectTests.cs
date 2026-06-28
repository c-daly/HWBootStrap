using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// A dropped connection must release its seat, or refreshes/reconnects pile up "ghost" players and
    /// real clients get turned away with SEAT FULL. When the last player leaves, the room resets so the
    /// next pair gets a fresh game rather than resuming a stale one.
    /// </summary>
    public class NetDisconnectTests
    {
        private const PlayerId P0 = PlayerId.Player0;
        private const PlayerId P1 = PlayerId.Player1;

        private static GameState TwoUnitGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++) tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var u0 = new Unit(1, P0, stats, new HexCoord(0, 0), 0);
            var u1 = new Unit(2, P1, stats, new HexCoord(4, 0), 0);
            var p0 = new PlayerState(P0, 10, null, new[] { u0 }, null);
            var p1 = new PlayerState(P1, 10, null, new[] { u1 }, null);
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, P0, 1, 3);
        }

        [Test]
        public void Session_Leave_FreesTheSeatForReuse()
        {
            var s = new GameSession(TwoUnitGame());
            Assert.That(s.Join("a"), Is.EqualTo(P0));
            Assert.That(s.Join("b"), Is.EqualTo(P1));
            s.Leave("a");
            Assert.That(s.Join("c"), Is.EqualTo(P0), "a's freed seat is available again");
        }

        [Test]
        public void Hub_Disconnect_FreesSeat_SoANewJoinerIsSeatedNotTurnedAway()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a"); // P0
            hub.Connect("r", "b"); // P1 — room full
            hub.Disconnect("r", "a");
            var c = hub.Connect("r", "c");
            Assert.That(c, Has.Some.Matches<Outbound>(o => o.ConnectionId == "c" && o.Message == "SEAT 0"));
            Assert.That(c, Has.None.Matches<Outbound>(o => o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void Hub_Disconnect_LastMember_ResetsRoomForAFreshGame()
        {
            var hub = new MatchHub(_ => TwoUnitGame());
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            hub.Disconnect("r", "a");
            hub.Disconnect("r", "b"); // room now empty
            var d = hub.Connect("r", "d");
            Assert.That(d, Has.Some.Matches<Outbound>(o => o.ConnectionId == "d" && o.Message == "SEAT 0"));
        }
    }
}
