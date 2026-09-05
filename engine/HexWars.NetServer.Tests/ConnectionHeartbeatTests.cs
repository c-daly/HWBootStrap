using System.Net.WebSockets;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The heartbeat is the only thing that happens on an idle match, which makes it the one loop on this
    /// host that must never be held up by anything.
    ///
    /// It does two jobs with very different costs. Pings and staleness are arithmetic over a dictionary;
    /// re-checking a credential is a database round trip, once per socket. Doing the second one inline
    /// would put every match on the host behind the slowest query in it - so these tests are about the
    /// second job never being allowed to delay the first.
    ///
    /// The clock is the real one, as it is in the endpoint tests: liveness is genuinely about elapsed time,
    /// and the injected TimeProvider here does not drive timers.
    /// </summary>
    [TestFixture]
    public sealed class ConnectionHeartbeatTests
    {
        static readonly Guid MatchId = Guid.Parse("2f6b9f0e-2c1f-4c5a-9d1e-1a2b3c4d5e6f");

        V2ConnectionRegistry _registry = null!;
        BlockingCredentialService _credentials = null!;
        ConnectionHeartbeatService _heartbeat = null!;
        readonly List<V2Connection> _connections = new();

        [TearDown]
        public async Task StopTheHeartbeat()
        {
            _credentials?.ReleaseAll();
            if (_heartbeat is not null) await _heartbeat.StopAsync(CancellationToken.None);
            _heartbeat?.Dispose();

            // Together, and each told its receive loop is over first. These fakes have no handler behind
            // them, so a close would otherwise wait out the unresponsive-peer window once per socket.
            foreach (V2Connection connection in _connections) connection.ReceiveLoopEnded();

            await Task.WhenAll(_connections.Select(c => c.CloseAsync(1000, "bye")));
            _connections.Clear();
        }

        void Start(int recheckSeconds)
        {
            var options = Options.Create(new MatchHostingOptions
            {
                HeartbeatIntervalSeconds = 1,
                StaleConnectionSeconds = 300,
                CredentialRecheckSeconds = recheckSeconds,
                MaxSocketsPerIp = 256,
            });

            _registry = new V2ConnectionRegistry(options, NullLogger<V2ConnectionRegistry>.Instance);
            _credentials = new BlockingCredentialService();

            var store = new InMemoryMatchStore();
            var coordinator = new DurableMatchCoordinator(
                store,
                _credentials,
                new JournalLiveMatchLoader(store),
                _registry,
                options,
                TimeProvider.System,
                NullLogger<DurableMatchCoordinator>.Instance);

            _heartbeat = new ConnectionHeartbeatService(
                _registry,
                coordinator,
                _credentials,
                options,
                TimeProvider.System,
                NullLogger<ConnectionHeartbeatService>.Instance);
        }

        /// <summary>One authenticated socket over a peer that accepts everything and says nothing.</summary>
        RecordingWebSocket Seat(int index)
        {
            var peer = new RecordingWebSocket();
            var connection = new V2Connection(
                "c" + index.ToString(), peer, "203.0.113.1", 256, 1024 * 1024, DateTimeOffset.UtcNow)
            {
                MatchId = MatchId,
                CredentialHash = new byte[32],
                CredentialExpiresAt = DateTimeOffset.UtcNow.AddHours(1),

                // Long ago, so every socket is due for a re-check on the very first tick.
                LastCredentialCheck = DateTimeOffset.UtcNow.AddHours(-1),
            };

            connection.CredentialHash[0] = (byte)index;
            _connections.Add(connection);
            _registry.Add(connection);
            return peer;
        }

        [Test]
        public async Task OneWedgedRecheck_StopsNeitherThePingsNorTheOtherRechecks()
        {
            Start(recheckSeconds: 5);

            var peers = new List<RecordingWebSocket>();
            for (var i = 0; i < 50; i++) peers.Add(Seat(i));

            // Seat 7 asks a store that never answers. Every other seat has to be unaffected by it, and so
            // does the ping the other 49 matches are relying on to stay connected.
            _credentials.BlockFor(7);

            await _heartbeat.StartAsync(CancellationToken.None);

            await WaitUntil(() => _credentials.Completed >= 49, TimeSpan.FromSeconds(15));
            await WaitUntil(() => peers.All(p => p.Sent.Any(m => m == "PING")), TimeSpan.FromSeconds(15));

            Assert.That(_credentials.Completed, Is.GreaterThanOrEqualTo(49),
                "one query that never returns must not be 50 queries that never return");
            Assert.That(_credentials.InFlightPeak,
                Is.LessThanOrEqualTo(ConnectionHeartbeatService.MaxConcurrentRechecks),
                "a host with a thousand sockets must not turn one heartbeat into a thousand queries");
            Assert.That(peers.Count(p => p.Sent.Contains("PING")), Is.EqualTo(50));
            Assert.That(peers[7].Sent.Contains("PING"), Is.True,
                "the socket whose re-check is stuck is still a socket, and still gets its ping");
        }

        [Test]
        public async Task ShuttingDown_CancelsARecheckThatIsStillWaiting()
        {
            Start(recheckSeconds: 5);
            Seat(0);
            _credentials.BlockFor(0);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Started >= 1, TimeSpan.FromSeconds(15));

            DateTimeOffset asked = DateTimeOffset.UtcNow;
            await _heartbeat.StopAsync(CancellationToken.None);

            Assert.That(DateTimeOffset.UtcNow - asked, Is.LessThan(TimeSpan.FromSeconds(5)),
                "shutdown must not wait out a query that is not coming back");
            await WaitUntil(() => _credentials.Cancelled >= 1, TimeSpan.FromSeconds(10));
            Assert.That(_credentials.Cancelled, Is.GreaterThanOrEqualTo(1),
                "the token handed to the store is a real one, so stopping the host stops the query");
        }

        static async Task WaitUntil(Func<bool> condition, TimeSpan within)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.Add(within);
            while (DateTimeOffset.UtcNow < deadline)
            {
                if (condition()) return;
                await Task.Delay(25);
            }
        }

        /// <summary>A credential service that answers instantly except for the seats a test wedges.</summary>
        sealed class BlockingCredentialService : IMatchCredentialService
        {
            readonly HashSet<byte> _blocked = new();
            readonly SemaphoreSlim _release = new(0);
            int _started;
            int _completed;
            int _cancelled;
            int _inFlight;
            int _inFlightPeak;

            public int Started => Volatile.Read(ref _started);
            public int Completed => Volatile.Read(ref _completed);
            public int Cancelled => Volatile.Read(ref _cancelled);
            public int InFlightPeak => Volatile.Read(ref _inFlightPeak);

            public void BlockFor(byte seat)
            {
                lock (_blocked) _blocked.Add(seat);
            }

            public void ReleaseAll() => _release.Release(int.MaxValue / 2);

            public async Task<bool> IsStillValidAsync(
                byte[] credentialHash, Guid matchId, DateTimeOffset now, CancellationToken ct)
            {
                Interlocked.Increment(ref _started);
                RecordPeak(Interlocked.Increment(ref _inFlight));

                bool blocked;
                lock (_blocked) blocked = _blocked.Contains(credentialHash[0]);

                try
                {
                    if (blocked) await _release.WaitAsync(ct).ConfigureAwait(false);

                    Interlocked.Increment(ref _completed);
                    return true;
                }
                catch (OperationCanceledException)
                {
                    Interlocked.Increment(ref _cancelled);
                    throw;
                }
                finally
                {
                    Interlocked.Decrement(ref _inFlight);
                }
            }

            void RecordPeak(int depth)
            {
                int seen = Volatile.Read(ref _inFlightPeak);
                while (depth > seen)
                {
                    int prior = Interlocked.CompareExchange(ref _inFlightPeak, depth, seen);
                    if (prior == seen) return;
                    seen = prior;
                }
            }

            public Task<IssuedCredential> IssueAsync(Guid matchId, string steamId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<CredentialValidation?> ValidateAsync(
                Guid matchId, string credential, CancellationToken ct) =>
                throw new NotSupportedException();
        }

        /// <summary>A peer that accepts every frame and never sends one.</summary>
        sealed class RecordingWebSocket : WebSocket
        {
            readonly List<string> _sent = new();

            public IReadOnlyList<string> Sent
            {
                get { lock (_sent) return _sent.ToArray(); }
            }

            public override WebSocketCloseStatus? CloseStatus => null;
            public override string? CloseStatusDescription => null;
            public override WebSocketState State => WebSocketState.Open;
            public override string? SubProtocol => null;

            public override void Abort() { }

            public override Task CloseAsync(
                WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
                Task.CompletedTask;

            public override Task CloseOutputAsync(
                WebSocketCloseStatus closeStatus, string? statusDescription, CancellationToken ct) =>
                Task.CompletedTask;

            public override void Dispose() { }

            public override Task<WebSocketReceiveResult> ReceiveAsync(
                ArraySegment<byte> buffer, CancellationToken ct) =>
                new TaskCompletionSource<WebSocketReceiveResult>().Task;

            public override Task SendAsync(
                ArraySegment<byte> buffer, WebSocketMessageType messageType, bool endOfMessage,
                CancellationToken ct)
            {
                lock (_sent) _sent.Add(System.Text.Encoding.UTF8.GetString(buffer.AsSpan()));
                return Task.CompletedTask;
            }
        }
    }
}
