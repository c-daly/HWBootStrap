using System.Collections.Generic;
using System.Linq;
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

        private static MatchHub NewHub() => new MatchHub(
            _ => TwoUnitGame(),
            newCatalogGame: (_, p0, p1) => WithBarracks(TwoUnitGame(), p0, p1));

        private static GameState WithBarracks(GameState state,
            IReadOnlyList<UnitTemplate> p0Barracks, IReadOnlyList<UnitTemplate> p1Barracks)
        {
            var p0 = state.Player(P0);
            var p1 = state.Player(P1);
            return new GameState(state.Board, state.Config, new[]
            {
                new PlayerState(P0, p0.Points, p0Barracks, p0.UnitsOnBoard, p0.Generators, p0.DestroyedValue),
                new PlayerState(P1, p1.Points, p1Barracks, p1.UnitsOnBoard, p1.Generators, p1.DestroyedValue),
            }, state.ActivePlayer, state.Round, state.NextEntityId);
        }

        private static UnitTemplate Template(string name, int health) =>
            new UnitTemplate(name, new UnitStats(health, 1, 2, 3, 4, 5, 6, 7, 8));

        private static IReadOnlyList<Outbound> Catalog(MatchHub hub, string connectionId,
            params UnitTemplate[] templates) =>
            hub.Receive("r", connectionId, NetProtocol.Catalog(BarracksWire.Write(templates)));

        private static void StartDefault(MatchHub hub)
        {
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            Catalog(hub, "a", BarracksCatalog.DefaultTemplates.ToArray());
            Catalog(hub, "b", BarracksCatalog.DefaultTemplates.ToArray());
        }

        [Test]
        public void Start_WaitsForBothSeatCatalogs_ThenDealsDistinctBarracksToEveryone()
        {
            var hub = NewHub();
            var a = hub.Connect("r", "a");
            Assert.That(a, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == "SEAT 0"));
            Assert.That(a, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")));

            var b = hub.Connect("r", "b");
            Assert.That(b, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message == "SEAT 1"));
            Assert.That(b, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")),
                "two occupied seats are not enough until their barracks arrive");

            Assert.That(Catalog(hub, "a", Template("Alpha", 2)),
                Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")));
            var started = Catalog(hub, "b", Template("Bravo", 3));

            Assert.That(started, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message.StartsWith("START ")));
            Assert.That(started, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message.StartsWith("START ")));
            var start = ReplayFile.Read(started.First(o => o.Message.StartsWith("START ")).Message.Substring("START ".Length)).Start;
            Assert.That(start.Player(P0).Barracks.Select(x => x.Name), Is.EqualTo(new[] { "Alpha" }));
            Assert.That(start.Player(P1).Barracks.Select(x => x.Name), Is.EqualTo(new[] { "Bravo" }));
        }

        [Test]
        public void PreStartCommand_IsRejectedWithVersionedSetupReason()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");

            var outs = hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0)));

            Assert.That(outs, Has.Exactly(1).Matches<Outbound>(o =>
                o.ConnectionId == "a" && o.Message == "REJECT CatalogV1Required"));
        }

        [Test]
        public void MalformedCatalog_CountsAsReceivedDefaults_AndCanStart()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");

            Assert.That(hub.Receive("r", "a", "CATALOG definitely-not-v1"),
                Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")));
            var started = Catalog(hub, "b", Template("Bravo", 3));

            var start = ReplayFile.Read(started.First(o => o.Message.StartsWith("START ")).Message.Substring("START ".Length)).Start;
            Assert.That(start.Player(P0).Barracks.Select(x => x.Name),
                Is.EqualTo(BarracksCatalog.DefaultTemplates.Select(x => x.Name)));
            Assert.That(start.Player(P1).Barracks.Select(x => x.Name), Is.EqualTo(new[] { "Bravo" }));
        }

        [Test]
        public void Catalog_AfterStart_IsRejectedWithoutReplacingTheMatch()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            Catalog(hub, "a", Template("Alpha", 2));
            Catalog(hub, "b", Template("Bravo", 3));

            var outs = Catalog(hub, "a", Template("Replacement", 9));

            Assert.That(outs, Has.Exactly(1).Matches<Outbound>(o =>
                o.ConnectionId == "a" && o.Message == "REJECT CatalogClosed"));
        }

        [Test]
        public void Catalog_FromUnseatedConnection_IsRejectedAndCannotStartRoom()
        {
            var hub = NewHub();
            hub.Connect("r", "a");
            hub.Connect("r", "b");
            hub.Connect("r", "stranger");
            Catalog(hub, "a", Template("Alpha", 2));

            var outsider = Catalog(hub, "stranger", Template("Intruder", 9));

            Assert.That(outsider, Has.Exactly(1).Matches<Outbound>(o =>
                o.ConnectionId == "stranger" && o.Message == "REJECT NoSeat"));
            Assert.That(outsider, Has.None.Matches<Outbound>(o => o.Message.StartsWith("START")));
        }

        [Test]
        public void GameFactory_IsCalledExactlyOnce_AfterBothCatalogsArrive()
        {
            int calls = 0;
            var hub = new MatchHub(
                _ => throw new AssertionException("legacy factory must not build a catalog-aware room"),
                newCatalogGame: (_, p0, p1) =>
                {
                    calls++;
                    return WithBarracks(TwoUnitGame(), p0, p1);
                });

            hub.Connect("r", "a");
            hub.Connect("r", "b");
            Assert.That(calls, Is.Zero);
            Catalog(hub, "a", Template("Alpha", 2));
            Assert.That(calls, Is.Zero);
            Catalog(hub, "b", Template("Bravo", 3));
            Assert.That(calls, Is.EqualTo(1));
            Catalog(hub, "a", Template("Replacement", 9));
            Assert.That(calls, Is.EqualTo(1));
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
            StartDefault(hub);
            var outs = hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0)));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == "APPLY E 0"));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message == "APPLY E 0"));
        }

        [Test]
        public void Impersonation_RejectsIssuerOnly_NoBroadcast()
        {
            var hub = NewHub();
            StartDefault(hub);
            var outs = hub.Receive("r", "b", NetProtocol.Cmd(new EndTurn(P0))); // b(P1) issues as P0
            Assert.That(outs, Has.None.Matches<Outbound>(o => o.Message.StartsWith("APPLY")));
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "b" && o.Message.StartsWith("REJECT")));
        }

        [Test]
        public void OutOfTurn_RejectsWithEngineReason()
        {
            var hub = NewHub();
            StartDefault(hub);
            hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0))); // now P1's turn
            var outs = hub.Receive("r", "a", NetProtocol.Cmd(new EndTurn(P0)));
            Assert.That(outs, Has.Some.Matches<Outbound>(
                o => o.ConnectionId == "a" && o.Message == NetProtocol.Reject(RejectionReason.NotYourTurn)));
        }

        [Test]
        public void Receive_MalformedCommand_RejectsIssuerOnly_NoBroadcast()
        {
            var hub = NewHub();
            StartDefault(hub);
            var outs = hub.Receive("r", "a", "CMD Z garbage");
            Assert.That(outs, Has.Some.Matches<Outbound>(o => o.ConnectionId == "a" && o.Message == NetProtocol.Malformed));
            Assert.That(outs, Has.None.Matches<Outbound>(o => o.Message.StartsWith("APPLY")));
        }

        [Test]
        public void Connect_RoomBuiltFromHostSetup_NotJoiners()
        {
            var hub = new MatchHub(GameFactory.Build); // the real factory turns the host's setup into the game
            hub.Connect("r", "host", new GameSetup(GameMode.Annihilation, 11, 8, 0, 1));
            hub.Connect("r", "guest", new GameSetup(GameMode.Annihilation, 5, 5, 0, 2));
            hub.Receive("r", "host", NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
            var outs = hub.Receive("r", "guest", NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
            var startMsg = outs.First(o => o.Message.StartsWith("START ")).Message;
            var state = ReplayFile.Read(startMsg.Substring("START ".Length)).Start;
            Assert.That(state.Board.Tiles.Count, Is.EqualTo(11 * 8), "room uses the host's board size, not the joiner's");
        }
    }
}
