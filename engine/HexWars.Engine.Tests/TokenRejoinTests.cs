using System;
using System.Collections.Generic;
using System.Linq;
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
        public void TwoTabsSameToken_BeforeOpponent_DoesNotStart()
        {
            var hub = NewHub();
            var t1 = hub.Connect("R", "conn-a1", token: "tok-a");   // host's first tab -> P0
            var t2 = hub.Connect("R", "conn-a2", token: "tok-a");   // second tab, SAME identity, no disconnect

            Assert.That(t1, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START ")));
            Assert.That(t2, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START ")),
                "two tabs of one player are one seat — the game must not start against yourself");
            Assert.That(hub.OpenGames(), Has.Count.EqualTo(1),
                "still a lone waiting host — the room stays browsable however many tabs the host has open");

            var joined = hub.Connect("R", "conn-b", token: "tok-b"); // a real opponent -> P1, NOW it starts
            Assert.That(joined, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a1" && o.Message.StartsWith("START ")));
            Assert.That(joined, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a2" && o.Message.StartsWith("START ")));
            Assert.That(joined, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-b" && o.Message.StartsWith("START ")),
                "START is dealt to every live connection, duplicate tabs included");
        }

        [Test]
        public void JoinOnly_ReconnectIntoHeldRoom_ReclaimsSeat()
        {
            var hub = NewHub();
            hub.Connect("R", "conn-a", token: "tok-a");
            hub.Connect("R", "conn-b", token: "tok-b");
            hub.Disconnect("R", "conn-a");
            hub.Disconnect("R", "conn-b");                // room empty, started -> held

            _now += TimeSpan.FromMinutes(5).Ticks;
            // A reconnecting client arrives via the join link/code path (joinOnly) — the held room
            // still exists, so joinOnly must not turn it away; the token reclaims its seat.
            var back = hub.Connect("R", "conn-a-2", joinOnly: true, token: "tok-a");
            Assert.That(back, Has.None.Matches<Outbound>(o => o.Message == NetProtocol.SeatFull));
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message == "SEAT 0"));
            Assert.That(back, Has.Some.Matches<Outbound>(o => o.ConnectionId == "conn-a-2" && o.Message.StartsWith("START ")),
                "a joinOnly reconnect into a held room gets its seat and a personal START re-deal");
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

        // Two units 3 hexes apart on a flat plains line so a move + an in-range attack are both legal
        // without needing deployment zones or terrain variety.
        private static GameState AdjacentUnitGame()
        {
            var tiles = new List<Tile>();
            for (int q = 0; q <= 3; q++) tiles.Add(new Tile(new HexCoord(q, 0), 0, TerrainType.Plains));
            var board = new Board(tiles, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(3, 0) });
            var stats = new UnitStats(health: 5, damage: 3, defense: 1, movement: 2, verticalMovement: 1,
                                       range: 2, rangeArc: 0, vision: 3, visionArc: 0);
            var u0 = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, stats, new HexCoord(3, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 10, null, new[] { u0 }, null);
            var p1 = new PlayerState(PlayerId.Player1, 10, null, new[] { u1 }, null);
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, 1, 3);
        }

        /// <summary>C1: a reconnect re-deal must reconstruct the EXACT mid-game state, not the fresh
        /// start state. Moves + damages a unit before the drop, reconnects the same token, then
        /// fast-forwards the re-dealt ReplayData through the same engine the client uses (exactly what
        /// GameBootstrap.OnNetStart now does) and checks the result against a state built by applying
        /// the identical commands directly — the same computation the held GameSession performed, so
        /// this is a field-level check that the re-deal lost nothing (HP, MovedUnitIds, MovementSpent).</summary>
        [Test]
        public void Reconnect_ReDeal_ReconstructsDamageAndTurnTracking_Exactly()
        {
            var hub = new MatchHub(_ => AdjacentUnitGame(), () => _now);
            hub.Connect("R", "conn-a", token: "tok-a");   // P0
            hub.Connect("R", "conn-b", token: "tok-b");   // P1 -> Started

            var move = new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0));
            var attack = new AttackUnit(PlayerId.Player0, 1, 2);
            var afterMove = hub.Receive("R", "conn-a", NetProtocol.Cmd(move));
            Assert.That(afterMove, Has.None.Matches<Outbound>(o => o.Message.StartsWith("REJECT")), "move must be accepted");
            var afterAttack = hub.Receive("R", "conn-a", NetProtocol.Cmd(attack));
            Assert.That(afterAttack, Has.None.Matches<Outbound>(o => o.Message.StartsWith("REJECT")), "attack must be accepted");

            hub.Disconnect("R", "conn-a");                                // a's socket dies mid-game, after damage
            var back = hub.Connect("R", "conn-a-2", token: "tok-a");      // reconnects on a new socket, same token

            var startOut = back.Single(o => o.ConnectionId == "conn-a-2" && o.Message.StartsWith("START "));
            var replay = ReplayFile.Read(startOut.Message.Substring("START ".Length));

            // Fast-forward exactly like GameBootstrap.OnNetStart does: Start, then every logged command.
            var replayed = replay.Start;
            foreach (var cmd in replay.Commands)
            {
                var r = GameEngine.Apply(replayed, cmd);
                Assert.That(r.Success, Is.True, $"replayed command must still apply cleanly: {cmd}");
                replayed = r.NewState;
            }

            // The same two commands applied directly to a fresh copy of the same start state — what the
            // held GameSession itself computed, independently reproduced.
            var direct = GameEngine.Apply(AdjacentUnitGame(), move);
            direct = GameEngine.Apply(direct.NewState, attack);
            var expected = direct.NewState;

            Assert.That(replay.Commands, Has.Count.EqualTo(2), "the log carries exactly the move + the attack");

            var replayedTarget = replayed.Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 2);
            var expectedTarget = expected.Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 2);
            Assert.That(replayedTarget.CurrentHp, Is.EqualTo(expectedTarget.CurrentHp),
                "the re-dealt replay must reconstruct the damaged unit's HP exactly, not full health");
            Assert.That(replayedTarget.CurrentHp, Is.LessThan(expectedTarget.Stats.Health),
                "sanity: the unit is actually damaged in this scenario");

            Assert.That(replayed.MovedUnitIds, Is.EquivalentTo(expected.MovedUnitIds));
            Assert.That(replayed.AttackedUnitIds, Is.EquivalentTo(expected.AttackedUnitIds));
            Assert.That(replayed.MovementSpent.Keys, Is.EquivalentTo(expected.MovementSpent.Keys));
            foreach (var id in expected.MovementSpent.Keys)
                Assert.That(replayed.MovementSpent[id], Is.EqualTo(expected.MovementSpent[id]),
                    $"MovementSpent for unit {id} must survive the re-deal exactly");
        }
    }
}
