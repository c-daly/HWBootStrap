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

                using var b = await Connect();
                string seatB = await Recv(b);               // SEAT 1
                string startA = await Recv(a);              // START ... (dealt to both once full)
                string startB = await Recv(b);

                await Send(a, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string applyA = await Recv(a);              // APPLY E 0 broadcast to both
                string applyB = await Recv(b);

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0";

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}'");

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

        static async Task<ClientWebSocket> Connect()
        {
            var c = new ClientWebSocket();
            await c.ConnectAsync(new Uri(Ws), CancellationToken.None);
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
