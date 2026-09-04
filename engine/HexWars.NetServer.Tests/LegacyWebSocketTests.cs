using System.Net.WebSockets;
using System.Text;
using HexWars.Engine;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// End-to-end cover for the legacy /ws lobby through the real host: a socket really is upgraded, the
    /// hub really seats it, and the Origin check really honours ALLOWED_WEB_ORIGINS. The hub is a static
    /// singleton shared by every test in the process, so each test uses its own room code.
    /// </summary>
    [TestFixture]
    public class LegacyWebSocketTests
    {
        static WebApplicationFactory<Program> Factory(string? allowedWebOrigins = null) =>
            new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                if (allowedWebOrigins is not null)
                    builder.UseSetting("ALLOWED_WEB_ORIGINS", allowedWebOrigins);
            });

        static async Task<WebSocket> ConnectAsync(
            WebApplicationFactory<Program> factory, string room, string? origin)
        {
            var wsClient = factory.Server.CreateWebSocketClient();
            if (origin is not null)
                wsClient.ConfigureRequest = request => request.Headers["Origin"] = origin;
            return await wsClient.ConnectAsync(new Uri("ws://localhost/ws?room=" + room), CancellationToken.None);
        }

        static async Task<string> ReceiveAsync(WebSocket socket)
        {
            var buffer = new byte[65536];
            using var message = new MemoryStream();
            WebSocketReceiveResult result;
            do
            {
                result = await socket.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);
                message.Write(buffer, 0, result.Count);
            } while (!result.EndOfMessage);
            return Encoding.UTF8.GetString(message.ToArray());
        }

        static Task SendAsync(WebSocket socket, string text) => socket.SendAsync(
            Encoding.UTF8.GetBytes(text), WebSocketMessageType.Text, true, CancellationToken.None);

        [Test]
        public async Task SameHostOrigin_IsSeated()
        {
            using var factory = Factory();
            using var socket = await ConnectAsync(factory, "ORIGINSAME", "http://localhost");

            Assert.That(await ReceiveAsync(socket), Is.EqualTo("SEAT 0"));
        }

        [Test]
        public async Task NoOriginHeader_IsSeated()
        {
            using var factory = Factory();
            using var socket = await ConnectAsync(factory, "ORIGINNONE", origin: null);

            Assert.That(await ReceiveAsync(socket), Is.EqualTo("SEAT 0"));
        }

        /// <summary>The reviewer case: a portal on another domain, explicitly configured, was still
        /// refused because the check only ever compared the Origin with the request Host.</summary>
        [Test]
        public async Task ConfiguredCrossSiteOrigin_IsSeated()
        {
            using var factory = Factory("https://portal.invalid");
            using var socket = await ConnectAsync(factory, "ORIGINALLOW", "https://portal.invalid");

            Assert.That(await ReceiveAsync(socket), Is.EqualTo("SEAT 0"));
        }

        [Test]
        public async Task UnconfiguredCrossSiteOrigin_IsRefused()
        {
            using var factory = Factory("https://portal.invalid");

            var ex = Assert.CatchAsync(async () => await ConnectAsync(factory, "ORIGINDENY", "https://evil.invalid"));

            Assert.That(ex!.Message, Does.Contain("403"));
        }

        /// <summary>The hub end to end through the real host: two sockets are seated, both hand in a
        /// barracks catalog, and the server deals the same START replay to both.</summary>
        [Test]
        public async Task TwoClientsInOneRoom_AreSeatedAndDealtTheSameStart()
        {
            string catalog = NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
            using var factory = Factory();

            using var host = await ConnectAsync(factory, "PLAYTHROUGH", origin: null);
            Assert.That(await ReceiveAsync(host), Is.EqualTo(NetProtocol.Seat(PlayerId.Player0)));
            Assert.That(await ReceiveAsync(host), Is.EqualTo(NetProtocol.CatalogRequest));
            await SendAsync(host, catalog);

            using var guest = await ConnectAsync(factory, "PLAYTHROUGH", origin: null);
            Assert.That(await ReceiveAsync(guest), Is.EqualTo(NetProtocol.Seat(PlayerId.Player1)));
            Assert.That(await ReceiveAsync(guest), Is.EqualTo(NetProtocol.CatalogRequest));
            await SendAsync(guest, catalog);

            string hostStart = await ReceiveAsync(host);
            string guestStart = await ReceiveAsync(guest);

            Assert.That(hostStart, Does.StartWith("START "));
            Assert.That(guestStart, Is.EqualTo(hostStart));
        }

        [Test]
        public async Task CrossSiteOriginWithNothingConfigured_IsRefused()
        {
            using var factory = Factory();

            var ex = Assert.CatchAsync(async () => await ConnectAsync(factory, "ORIGINBARE", "https://evil.invalid"));

            Assert.That(ex!.Message, Does.Contain("403"));
        }
    }
}
