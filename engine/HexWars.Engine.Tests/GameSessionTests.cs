using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The authoritative, transport-agnostic multiplayer session: it seats two connections (P0/P1),
    /// blocks a connection from issuing commands as the other seat (anti-impersonation), and otherwise
    /// defers legality/turn-order to the engine. This is the server's brain; a WebSocket layer just
    /// pipes strings into it. No sockets here so it's unit-testable.
    /// </summary>
    public class GameSessionTests
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
        public void Join_SeatsTwoThenFull()
        {
            var s = new GameSession(TwoUnitGame());
            Assert.That(s.Join("a"), Is.EqualTo(P0));
            Assert.That(s.Join("b"), Is.EqualTo(P1));
            Assert.That(s.Join("c"), Is.Null, "third joiner has no seat");
        }

        [Test]
        public void Join_SameConnectionKeepsItsSeat()
        {
            var s = new GameSession(TwoUnitGame());
            Assert.That(s.Join("a"), Is.EqualTo(P0));
            Assert.That(s.Join("a"), Is.EqualTo(P0), "rejoin returns the same seat, not a new one");
            Assert.That(s.Join("b"), Is.EqualTo(P1));
        }

        [Test]
        public void Submit_FromUnseatedConnection_NoSeat()
        {
            var s = new GameSession(TwoUnitGame());
            s.Join("a");
            var r = s.Submit("stranger", new EndTurn(P0));
            Assert.That(r.Status, Is.EqualTo(SubmitStatus.NoSeat));
        }

        [Test]
        public void Submit_ImpersonatingOtherSeat_WrongSeat()
        {
            var s = new GameSession(TwoUnitGame());
            s.Join("a"); // P0
            s.Join("b"); // P1
            // b holds P1 but tries to issue as P0 — must be blocked BEFORE the engine ever sees it
            var r = s.Submit("b", new EndTurn(P0));
            Assert.That(r.Status, Is.EqualTo(SubmitStatus.WrongSeat));
            Assert.That(s.State.ActivePlayer, Is.EqualTo(P0), "state untouched by a rejected impersonation");
        }

        [Test]
        public void Submit_LegalOnYourTurn_AcceptedAndAdvances()
        {
            var s = new GameSession(TwoUnitGame());
            s.Join("a"); // P0
            s.Join("b"); // P1
            var r = s.Submit("a", new EndTurn(P0));
            Assert.That(r.Status, Is.EqualTo(SubmitStatus.Accepted));
            Assert.That(s.State.ActivePlayer, Is.EqualTo(P1), "turn passed to the other seat");
        }

        [Test]
        public void Submit_OnOpponentsTurn_RejectedByEngine()
        {
            var s = new GameSession(TwoUnitGame());
            s.Join("a"); // P0
            s.Join("b"); // P1
            s.Submit("a", new EndTurn(P0)); // now P1's turn
            var r = s.Submit("a", new EndTurn(P0)); // P0 acts out of turn (not impersonation — own seat)
            Assert.That(r.Status, Is.EqualTo(SubmitStatus.Rejected));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.NotYourTurn));
        }
    }
}
