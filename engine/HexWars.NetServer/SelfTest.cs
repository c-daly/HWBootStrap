using System.Linq;
using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;
using HexWars.NetServer.Hosting;

namespace HexWars.NetServer
{
    /// <summary>
    /// End-to-end proof with no Unity and no browser: spin up the real server in-process, connect two
    /// real WebSocket clients, and walk them through seat → start → a move → broadcast. Exit 0 on pass.
    /// </summary>
    static class SelfTest
    {
        const string Url = "http://127.0.0.1:5234";
        const string Ws = "ws://127.0.0.1:5234/ws?room=test";

        public static async Task<int> Run()
        {
            var builder = WebApplication.CreateBuilder();
            builder.WebHost.UseUrls(Url);
            builder.Logging.ClearProviders();
            var app = builder.Build();
            app.UseWebSockets();
            app.Map("/ws", LegacyWebSocketServer.Handle);
            await app.StartAsync();

            try
            {
                using var a = await Connect();
                string seatA = await Recv(a);               // SEAT 0 (room not full yet)
                string catalogRequestA = await Recv(a);     // CATALOG? while the room is waiting
                await Send(a, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));

                var lobby1 = LegacyWebSocketServer.OpenGamesSnapshot();      // host waiting -> the room is browsable
                bool lobbyListsWaitingRoom = lobby1.Count == 1 && lobby1[0].Code == "TEST";

                using var b = await Connect();
                string seatB = await Recv(b);               // SEAT 1
                string catalogRequestB = await Recv(b);
                await Send(b, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                string startA = await Recv(a);              // START ... (dealt to both once full)
                string startB = await Recv(b);

                var lobby2 = LegacyWebSocketServer.OpenGamesSnapshot();      // both seated -> started -> unlisted
                bool lobbyEmptiesOnStart = lobby2.Count == 0;

                await Send(a, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string applyA = await Recv(a);              // APPLY E 0 broadcast to both
                string applyB = await Recv(b);

                using var joiner = await Connect("ws://127.0.0.1:5234/ws?room=missing&join=1");
                string joinerSeat = await Recv(joiner);       // a joiner must never mint a room for a typo'd code
                bool joinOnlyTurnedAway = joinerSeat == "SEAT FULL";

                // Reconnect: kill A's socket, reconnect with the SAME token, and confirm the server
                // seats it back into P0 and re-deals START (the game must survive a background/refresh).
                using var ra = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA = await Recv(ra);               // SEAT 0
                string rCatalogRequestA = await Recv(ra);
                await Send(ra, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                using var rb = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-b");
                string rSeatB = await Recv(rb);                // SEAT 1
                string rCatalogRequestB = await Recv(rb);
                await Send(rb, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                string rStartA = await Recv(ra);
                string rStartB = await Recv(rb);

                // C1: damage a unit BEFORE the drop — the default (9x7, seed 7) army placement lets
                // P0's Striker (unit 2) close to (3,0) and land a legal hit on P1's Striker (unit 5);
                // exact coordinates come from GameFactory's deterministic seed, not a guess.
                await Send(ra, NetProtocol.Cmd(new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0))));
                string rMoveApplyA = await Recv(ra);
                string rMoveApplyB = await Recv(rb);
                await Send(ra, NetProtocol.Cmd(new AttackUnit(PlayerId.Player0, 2, 5)));
                string rAttackApplyA = await Recv(ra);
                string rAttackApplyB = await Recv(rb);

                ra.Abort();                                     // simulate a dead socket (no clean close)
                using var ra2 = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA2 = await Recv(ra2);                // SEAT 0 again — same token, same seat
                string rStartA2 = await Recv(ra2);               // personal START re-deal

                // The re-deal must carry the move + the attack (not a fresh, undamaged deal) — assert
                // both the command count AND that fast-forwarding it (exactly what GameBootstrap.OnNetStart
                // does client-side) lands on the SAME state an independent direct replay of the identical
                // two commands produces from the same fresh start — the same field-level cross-check
                // TokenRejoinTests uses engine-side, applied here end-to-end through the real socket.
                var reDealt = ReplayFile.Read(rStartA2.Substring("START ".Length));
                bool reDealCarriesBothCommands = reDealt.Commands.Count == 2;

                var fastForwarded = reDealt.Start;
                bool fastForwardOk = true;
                foreach (var cmd in reDealt.Commands)
                {
                    var fr = GameEngine.Apply(fastForwarded, cmd);
                    if (!fr.Success) { fastForwardOk = false; break; }
                    fastForwarded = fr.NewState;
                }

                var freshP0PointsBefore = GameFactory.Build(GameSetup.Default).Player(PlayerId.Player0).Points;
                var direct = GameEngine.Apply(GameFactory.Build(GameSetup.Default), new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));
                direct = GameEngine.Apply(direct.NewState, new AttackUnit(PlayerId.Player0, 2, 5));
                var expected = direct.NewState;

                bool targetPresenceMatches =
                    fastForwarded.Player(PlayerId.Player1).UnitsOnBoard.Any(u => u.Id == 5) ==
                    expected.Player(PlayerId.Player1).UnitsOnBoard.Any(u => u.Id == 5);
                bool pointsMatch = fastForwarded.Player(PlayerId.Player0).Points == expected.Player(PlayerId.Player0).Points;
                bool attackActuallyLanded = expected.Player(PlayerId.Player0).Points > freshP0PointsBefore // bounty proves the hit landed
                    || expected.Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 5).CurrentHp
                       < GameFactory.Build(GameSetup.Default).Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 5).CurrentHp;
                bool reDealReflectsDamage = fastForwardOk && targetPresenceMatches && pointsMatch && attackActuallyLanded;

                await Send(ra2, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string rApplyA2 = await Recv(ra2);
                string rApplyB = await Recv(rb);

                bool reconnectOk =
                    rSeatA == "SEAT 0" && rSeatB == "SEAT 1" &&
                    rCatalogRequestA == NetProtocol.CatalogRequest &&
                    rCatalogRequestB == NetProtocol.CatalogRequest &&
                    rStartA.StartsWith("START ") && rStartB.StartsWith("START ") &&
                    rMoveApplyA.StartsWith("APPLY M ") && rMoveApplyA == rMoveApplyB &&
                    rAttackApplyA.StartsWith("APPLY A ") && rAttackApplyA == rAttackApplyB &&
                    rSeatA2 == "SEAT 0" && rStartA2.StartsWith("START ") &&
                    reDealCarriesBothCommands && reDealReflectsDamage &&
                    rApplyA2 == "APPLY E 0" && rApplyB == "APPLY E 0";

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    catalogRequestA == NetProtocol.CatalogRequest &&
                    catalogRequestB == NetProtocol.CatalogRequest &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart && joinOnlyTurnedAway && reconnectOk;

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}' lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart} joinOnly='{joinerSeat}' reconnectOk={reconnectOk} reDealCarriesBothCommands={reDealCarriesBothCommands} reDealReflectsDamage={reDealReflectsDamage} moveApply='{rMoveApplyA}' attackApply='{rAttackApplyA}'");

                await app.StopAsync();
                return ok ? 0 : 1;
            }
            catch (Exception ex)
            {
                Console.WriteLine("SELFTEST ERROR " + ex);
                await app.StopAsync();
                return 1;
            }
        }

        static async Task<ClientWebSocket> Connect(string url = Ws)
        {
            var c = new ClientWebSocket();
            await c.ConnectAsync(new Uri(url), CancellationToken.None);
            return c;
        }

        static async Task Send(ClientWebSocket c, string msg) =>
            await c.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None);

        static async Task<string> Recv(ClientWebSocket c)
        {
            var buf = new byte[16384];
            using var ms = new MemoryStream();
            WebSocketReceiveResult res;
            do
            {
                res = await c.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None);
                ms.Write(buf, 0, res.Count);
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }
    }
}
