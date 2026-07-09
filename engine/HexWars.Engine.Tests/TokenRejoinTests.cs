using System;
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Seats are keyed by a client-minted token, not the per-socket connection id, so a
    /// refresh/background/reconnect reclaims the same seat. A started room that drops to zero live
    /// connections is HELD (not deleted) for a 10-minute window — both players can blip through a
    /// network drop without losing the match; a stranger with a different token still can't steal a
    /// reserved seat. Un-started rooms (never dealt START) keep instant cleanup.</summary>
    public class TokenRejoinTests
    {
        private long _now;
        private MatchHub NewHub()
        {
            _now = TimeSpan.FromHours(1).Ticks;
            return new MatchHub(_ => TwoUnitGame(), () => _now);
        }

        private static GameState TwoUnitGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q < 5; q++) tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(4, 0) });
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var u0 = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, stats, new HexCoord(4, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 10, null, new[] { u0 }, null);
            var p1 = new PlayerState(PlayerId.Player1, 10, null, new[] { u1 }, null);
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, 3);
        }

        [Test]
        public void SameToken_NewConnection_ReclaimsSeat_AndReceivesStart_MidGame()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");   // P0
            hub.Connect("R", "conn-b", token: "tok-b");   // P1 -> room Started

            hub.Disconnect("R", "conn-a");                // a's socket dies (tab backgrounded)
            var back = hub.Connect("R", "conn-a-2", token: "tok-a"); // a reconnects on a NEW socket, SAME token

            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message == "SEAT 0"),
                "the same token reclaims seat P0, not whatever seat is next free");
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message.StartsWith("START ")),
                "a reconnect into a started room gets a personal START re-deal");
        }

        [Test]
        public void DifferentToken_GetsSeatFull_WhenBothSeatsAreClaimed()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");                // a's socket drops, seat stays reserved to tok-a

            var stranger = hub.Connect("R", "conn-c", token: "tok-c");
            Assert.That(stranger, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-c" && o.Message == NetProtocol.SeatFull));
        }

        [Test]
        public void HeldRoom_ReconnectWithinTenMinutes_StillReclaimsSeat()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");                // room empty, started -> held, not removed

            _now += TimeSpan.FromMinutes(9).Ticks;
            var back = hub.Connect("R", "conn-a-2", token: "tok-a");
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message == "SEAT 0"),
                "9 minutes in, the hold window hasn't expired");
        }

        [Test]
        public void HeldRoom_ExpiresAfterTenMinutes_BecomesAFreshRoom()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");

            _now += TimeSpan.FromMinutes(11).Ticks;
            var fresh = hub.Connect("R", "conn-d", token: "tok-d");
            Assert.That(fresh, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-d" && o.Message == "SEAT 0"),
                "11 minutes in, the held room expired and a brand-new game was minted");
        }

        [Test]
        public void UnstartedRoom_StillCleansUpInstantly_NotHeld()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");    // only one seat ever taken — never Started
            hub.Disconnect("R", "conn-a");

            // no time advance at all: if the room were held like a started one, tok-a would still own P0
            var next = hub.Connect("R", "conn-e", token: "tok-e");
            Assert.That(next, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-e" && o.Message == "SEAT 0"),
                "an un-started room resets instantly, same as before this feature");
        }

        [Test]
        public void OpenGames_NeverLists_AHeldEmptyRoom()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");                // held, empty, started

            Assert.That(hub.OpenGames(), Is.Empty, "a held room is not a waiting host — never browsable");
        }
    }
}
