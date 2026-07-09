using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;

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
            app.Map("/ws", Program.Handle);
            await app.StartAsync();

            try
            {
                using var a = await Connect();
                string seatA = await Recv(a);               // SEAT 0 (room not full yet)

                var lobby1 = Program.OpenGamesSnapshot();      // host waiting -> the room is browsable
                bool lobbyListsWaitingRoom = lobby1.Count == 1 && lobby1[0].Code == "TEST";

                using var b = await Connect();
                string seatB = await Recv(b);               // SEAT 1
                string startA = await Recv(a);              // START ... (dealt to both once full)
                string startB = await Recv(b);

                var lobby2 = Program.OpenGamesSnapshot();      // both seated -> started -> unlisted
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
                using var rb = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-b");
                string rSeatB = await Recv(rb);                // SEAT 1
                string rStartA = await Recv(ra);
                string rStartB = await Recv(rb);

                ra.Abort();                                     // simulate a dead socket (no clean close)
                using var ra2 = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA2 = await Recv(ra2);                // SEAT 0 again — same token, same seat
                string rStartA2 = await Recv(ra2);               // personal START re-deal

                await Send(ra2, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string rApplyA2 = await Recv(ra2);
                string rApplyB = await Recv(rb);

                bool reconnectOk =
                    rSeatA == "SEAT 0" && rSeatB == "SEAT 1" &&
                    rStartA.StartsWith("START ") && rStartB.StartsWith("START ") &&
                    rSeatA2 == "SEAT 0" && rStartA2.StartsWith("START ") &&
                    rApplyA2 == "APPLY E 0" && rApplyB == "APPLY E 0";

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart && joinOnlyTurnedAway && reconnectOk;

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}' lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart} joinOnly='{joinerSeat}' reconnectOk={reconnectOk}");

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
