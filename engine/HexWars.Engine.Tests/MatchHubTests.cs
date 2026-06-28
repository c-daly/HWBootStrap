using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The room router: maps connections to seats in a room, deals the start state once both seats are
    /// present, and broadcasts validated moves to everyone (rejections go only to the issuer). It returns
    /// the messages a transport should send, so the whole server brain is testable without a socket.
    /// </summary>
    public class MatchHubTests
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

        private static MatchHub NewHub() => new MatchHub(TwoUnitGame);

        [Test]
        public void SecondJoin_SeatsBothAndDealsStartToEveryone()
        {
            var hub = NewHub();
            var a = hub.Connect("r", "a");
            Assert.That(a, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == "SEAT 0"));
            Assert.That(a, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")), "no start until both seated");

            var b = hub.Connect("r", "b");
            Assert.That(b, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message == "SEAT 1"));
            Assert.That(b, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message.StartsWith("START ")));
            Assert.That(b, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message.StartsWith("START ")));
        }

        [Test]
        public void ThirdJoin_IsTurnedAway()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            var c = hub.Connect("r", "c");
            Assert.That(c, Has.Some.Matches<Outbound>(o => o.ConnectionId == "c" && o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void SeparateRoomCodes_AreIsolated()
        {
            var hub = NewHub();
            Assert.That(hub.Connect("r1", "a"), Has.Some.Matches<Outbound>(o => o.Message == "SEAT 0"));
            Assert.That(hub.Connect("r2", "b"), Has.Some.Matches<Outbound>(o => o.Message == "SEAT 0"), "a fresh room seats P0 again");
        }

        [Test]
        public void ValidCommand_BroadcastsApplyToBothSeats()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            var outs = hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0)));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == "APPLY E 0"));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message == "APPLY E 0"));
        }

        [Test]
        public void Impersonation_RejectsIssuerOnly_NoBroadcast()
        {
            var hub = NewHub();
            hub.Connect("r", "a"); // P0
            hub.Connect("r", "b"); // P1
            var outs = hub.Receive("r", "b", NetProtocol.Cmd(new EndTurn(P0))); // b(P1) issues as P0
            Assert.That(outs, Has.None.Matches<Outbound>(o => o.Message.StartsWith("APPLY")));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message.StartsWith("REJECT")));
        }

        [Test]
        public void OutOfTurn_RejectsWithEngineReason()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0))); // now P1's turn
            var outs = hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0)));
            Assert.That(outs, Has.Some.Matches<Outbound>(
                o => o.ConnectionId == "a" && o.Message == NetProtocol.Reject(RejectionReason.NotYourTurn)));
        }
    }
}
