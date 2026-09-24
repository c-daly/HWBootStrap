using System.Net;
using System.Net.Http.Headers;
using System.Net.WebSockets;
using System.Net.Sockets;
using System.Text;
using HexWars.NetServer.Hosting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The request limits against real Kestrel on a real socket.
    ///
    /// Every other test in this suite runs on the in-memory test server, which is the right place for
    /// routing and handler behaviour and the wrong place for this: the transport limits are Kestrel
    /// settings, the server header is written by Kestrel, and a body with no declared length only exists
    /// once there is a wire to send it over. A test host cannot show any of that.
    /// </summary>
    [TestFixture]
    public class KestrelRequestLimitTests
    {
        WebApplication _app = null!;
        HttpClient _client = null!;
        Uri _origin = null!;

        [SetUp]
        public async Task StartOnALoopbackPort()
        {
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions
            {
                EnvironmentName = Environments.Development,
            });

            // Port zero: the operating system picks one, so a suite running in parallel with anything else
            // cannot collide on a hard-coded number.
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            builder.Logging.ClearProviders();
            builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
            {
                // The legacy lobby only: it needs no database and no Steam credentials, and this file is
                // about the transport rather than about anything either of those decides.
                ["LOBBY_PROVIDER"] = "Legacy",
                ["MATCH_BUILD_ID"] = "kestrel-test",
            });

            builder.AddHexWarsServer();
            _app = builder.Build();
            _app.UseHexWarsServer();

            await _app.StartAsync();

            _origin = new Uri(_app.Urls.First());
            _client = new HttpClient { BaseAddress = _origin };
        }

        [TearDown]
        public async Task Stop()
        {
            _client?.Dispose();
            if (_app is not null)
            {
                await _app.StopAsync();
                await _app.DisposeAsync();
            }
        }

        static HttpRequestMessage Chunked(int bytes)
        {
            var request = new HttpRequestMessage(HttpMethod.Post, "/games")
            {
                Content = new StreamContent(new UndeclaredLengthStream(new byte[bytes])),
            };

            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            request.Headers.TransferEncodingChunked = true;
            return request;
        }

        [TestCase("GET", "/healthz")]
        [TestCase("POST", "/games")]
        public async Task AnUnfinishedChunkedBodyHasAnIndependentDeadline(string method, string path)
        {
            using var socket = new TcpClient();
            await socket.ConnectAsync(_origin.Host, _origin.Port);
            await using NetworkStream stream = socket.GetStream();
            // A large enough first chunk to avoid the transport's minimum-rate deadline. The
            // application must stop waiting even though the client never sends the final chunk.
            string request = $"{method} {path} HTTP/1.1\r\nHost: {_origin.Authority}\r\n"
                + "Transfer-Encoding: chunked\r\n\r\n1000\r\n" + new string('x', 4096) + "\r\n";
            await stream.WriteAsync(Encoding.ASCII.GetBytes(request));
            using var deadline = new CancellationTokenSource(RequestLimits.BodyReadTimeout + TimeSpan.FromSeconds(3));
            using var reader = new StreamReader(stream);
            string? response = await reader.ReadLineAsync(deadline.Token);
            Assert.That(response, Does.Contain("408"));
        }

        [Test]
        public async Task ABodyWithNoDeclaredLengthOverTheCap_IsRefused()
        {
            using HttpRequestMessage request = Chunked((int)RequestLimits.MaxRequestBodyBytes + 4096);
            using HttpResponseMessage response = await _client.SendAsync(request);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge),
                "omitting Content-Length must not be a way around the cap");
            Assert.That(await response.Content.ReadAsStringAsync(), Does.Contain("invalid_request"));
        }

        [Test]
        public async Task ABodyWithADeclaredLengthOverTheCap_IsRefused()
        {
            using var content = new StringContent(
                new string('a', (int)RequestLimits.MaxRequestBodyBytes + 1),
                Encoding.UTF8, "application/json");
            using HttpResponseMessage response = await _client.PostAsync("/games", content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge));
        }

        [Test]
        public async Task ABodyWithNoDeclaredLengthInsideTheCap_IsNotRefusedForItsSize()
        {
            using HttpRequestMessage request = Chunked(1024);
            using HttpResponseMessage response = await _client.SendAsync(request);

            Assert.That(response.StatusCode, Is.Not.EqualTo(HttpStatusCode.RequestEntityTooLarge),
                "the cap is a ceiling, not a ban on chunked requests");
        }

        [Test]
        public async Task NoResponseNamesTheSoftwareServingIt()
        {
            using HttpResponseMessage response = await _client.GetAsync("/healthz");

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(response.Headers.Contains("Server"), Is.False,
                "the server header names the software and its version to whoever is writing the scanner");
        }

        [Test]
        public async Task AWebsocketUpgradeIsUnaffectedByTheBodyCap()
        {
            using var socket = new ClientWebSocket();
            var address = new Uri("ws://" + _origin.Authority + "/ws?room=LIMITS");

            await socket.ConnectAsync(address, CancellationToken.None);

            Assert.That(socket.State, Is.EqualTo(WebSocketState.Open),
                "an upgrade is a GET with no body, and the request-limit middleware must not touch it");

            // Aborted rather than closed politely: the v1 lobby ends its side as soon as the client stops
                // reading, so a close handshake here would fail on a socket that upgraded perfectly well.
            socket.Abort();
        }

        /// <summary>A stream that will not say how long it is, so HttpClient sends it chunked.</summary>
        sealed class UndeclaredLengthStream(byte[] content) : Stream
        {
            int _position;

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();

            public override long Position
            {
                get => throw new NotSupportedException();
                set => throw new NotSupportedException();
            }

            public override int Read(byte[] buffer, int offset, int count)
            {
                int take = Math.Min(count, content.Length - _position);
                if (take <= 0) return 0;

                Array.Copy(content, _position, buffer, offset, take);
                _position += take;
                return take;
            }

            public override void Flush() { }
            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
        }
    }
}
