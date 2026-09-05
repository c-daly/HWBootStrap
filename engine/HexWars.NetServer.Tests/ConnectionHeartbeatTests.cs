using System.Net.WebSockets;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.Logging;
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
        CapturingLoggerProvider _logs = null!;
        ILoggerFactory _loggers = null!;
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
            _loggers?.Dispose();
        }

        void Start(
            int recheckSeconds,
            int heartbeatSeconds = 1,
            int budget = MatchHostingOptions.DefaultMaxRechecksPerCadence,
            TimeSpan? checkTakes = null,
            IMatchCredentialService? credentials = null)
        {
            var options = Options.Create(new MatchHostingOptions
            {
                HeartbeatIntervalSeconds = heartbeatSeconds,
                StaleConnectionSeconds = 300,
                CredentialRecheckSeconds = recheckSeconds,
                MaxRechecksPerCadence = budget,
                MaxSocketsPerIp = 8192,
            });

            _logs = new CapturingLoggerProvider();
            _loggers = LoggerFactory.Create(builder => builder.AddProvider(_logs));

            _registry = new V2ConnectionRegistry(options, NullLogger<V2ConnectionRegistry>.Instance);
            _credentials = new BlockingCredentialService(checkTakes);
            IMatchCredentialService service = credentials ?? _credentials;

            var store = new InMemoryMatchStore();
            var coordinator = new DurableMatchCoordinator(
                store,
                service,
                new JournalLiveMatchLoader(store),
                _registry,
                options,
                TimeProvider.System,
                NullLogger<DurableMatchCoordinator>.Instance);

            _heartbeat = new ConnectionHeartbeatService(
                _registry,
                coordinator,
                service,
                options,
                TimeProvider.System,
                _loggers.CreateLogger<ConnectionHeartbeatService>());
        }

        /// <summary>One authenticated socket over a peer that accepts everything and says nothing.</summary>
        RecordingWebSocket Seat(int index) => Seat(index, due: true);

        RecordingWebSocket Seat(int index, bool due, DateTimeOffset? lastCheck = null)
        {
            var peer = new RecordingWebSocket();
            var connection = new V2Connection(
                "c" + index.ToString(), peer, "203.0.113.1", 256, 1024 * 1024, DateTimeOffset.UtcNow)
            {
                MatchId = MatchId,
                CredentialHash = new byte[32],
                CredentialExpiresAt = DateTimeOffset.UtcNow.AddHours(1),

                // Long ago, so the socket is due for a re-check on the very first tick.
                LastCredentialCheck = lastCheck ?? (due
                    ? DateTimeOffset.UtcNow.AddHours(-1)
                    : DateTimeOffset.UtcNow.AddHours(1)),
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

        [Test]
        public async Task AHundredDueSockets_AreAllCheckedInsideOneCadence()
        {
            // The regression this file exists for. Taking a slot per socket and stopping at the first
            // refusal meant exactly eight genuinely asynchronous checks ran per heartbeat, however many
            // were due: a slot freed a millisecond later sat idle until the next tick, and the hundredth
            // socket was re-checked minutes after the interval promised it would be.
            //
            // A hundred quarter-second checks over eight workers is about three seconds of work, which
            // fits inside one five-second cadence and takes thirteen cadences without a pool that refills.
            Start(recheckSeconds: 1, heartbeatSeconds: 5, checkTakes: TimeSpan.FromMilliseconds(200));

            for (var i = 0; i < 100; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Completed >= 100, TimeSpan.FromSeconds(20));

            AssertStartedOldestFirst(_credentials.StartOrder);
            Assert.That(_credentials.Completed, Is.EqualTo(100));
            Assert.That(_heartbeat.RecheckCadences, Is.EqualTo(1),
                "one pass, not thirteen: the workers keep pulling until the batch is drained");
            Assert.That(_credentials.InFlightPeak,
                Is.LessThanOrEqualTo(ConnectionHeartbeatService.MaxConcurrentRechecks),
                "and draining is still not the same as flooding the store");
        }

        [Test]
        public async Task OneWedgedCheckInABatch_DoesNotStopTheOtherNinetyNine()
        {
            Start(recheckSeconds: 1, heartbeatSeconds: 5, checkTakes: TimeSpan.FromMilliseconds(200));

            for (var i = 0; i < 100; i++) Seat(i);
            _credentials.BlockFor(7);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Completed >= 99, TimeSpan.FromSeconds(20));

            Assert.That(_credentials.Completed, Is.GreaterThanOrEqualTo(99),
                "one query that never returns holds one worker, not the batch");
        }

        [Test]
        public async Task SelectingTheBatch_LooksAtTheDueSocketsAndNotTheHost()
        {
            // Sorting every connection on the host to choose a few hundred is work proportional to the
            // whole host, repeated every heartbeat. The selection walks the DUE ones once instead, so a
            // host with five thousand quiet sockets pays for the forty that are asking to be checked.
            Start(recheckSeconds: 1, heartbeatSeconds: 5, budget: 32);

            for (var i = 0; i < 40; i++) Seat(i, due: true);
            for (var i = 0; i < 5000; i++) Seat(200 + i, due: false);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _heartbeat.RecheckCadences >= 1, TimeSpan.FromSeconds(20));
            await WaitUntil(() => _credentials.Started >= 32, TimeSpan.FromSeconds(20));

            Assert.That(_heartbeat.RecheckCandidatesConsidered, Is.EqualTo(40),
                "only the due sockets are offered to the batch");
            Assert.That(_credentials.Started, Is.EqualTo(32), "and the batch is one budget deep");
        }

        [Test]
        public async Task SocketsThatDidNotFitInACadence_AreTheOnesCheckedNext()
        {
            // Fairness is the whole reason the queue is ordered by last check. A socket the pass never
            // reached is deliberately not stamped, so it keeps the oldest check time on the host and goes
            // to the front - without that a busy host would re-check the same lucky sockets forever and
            // never notice a revoked credential on any of the others.
            const int budget = 32;
            const int seats = budget + 8;

            Start(recheckSeconds: 1, budget: budget);
            for (var i = 0; i < seats; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Seen.Count >= seats, TimeSpan.FromSeconds(20));

            Assert.That(_credentials.Seen, Has.Count.EqualTo(seats),
                "every socket is reached within a few cadences, not merely the first budget of them");

            foreach (V2Connection connection in _connections)
                Assert.That(connection.LastCredentialCheck,
                    Is.GreaterThan(DateTimeOffset.UtcNow.AddMinutes(-30)),
                    "a socket that was checked is stamped, and one that was skipped is not");
        }

        [Test]
        public async Task UnderSustainedOverload_ChecksStartStrictlyOldestFirst()
        {
            // The oldest SET is not the same as oldest FIRST. Handing the pool a batch in registry order
            // starts whichever sockets happen to sit near the front of the dictionary, and when a check
            // outlasts a cadence those same few are all that ever start - an hour-old socket behind them is
            // cancelled every time and never asked about at all.
            //
            // Seat i is stamped one second newer than seat i-1, so the age order is exactly the index
            // order and any gap in the started set is a socket that was skipped over an older one.
            DateTimeOffset oldest = DateTimeOffset.UtcNow.AddHours(-2);

            Start(recheckSeconds: 300, heartbeatSeconds: 1, budget: 256,
                checkTakes: TimeSpan.FromSeconds(3));

            foreach (int index in Shuffled(100))
                Seat(index, due: true, lastCheck: oldest.AddSeconds(index));

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _heartbeat.RecheckCadences >= 3, TimeSpan.FromSeconds(20));
            await _heartbeat.StopAsync(CancellationToken.None);

            IReadOnlySet<byte> started = _credentials.Seen;

            Assert.That(started, Is.Not.Empty);
            Assert.That(started.Count, Is.LessThan(100), "the checks outlast a cadence, so not all of them");
            Assert.That(started, Is.EquivalentTo(Enumerable.Range(0, started.Count).Select(i => (byte)i)),
                "the started sockets are the N oldest, with no younger one jumping the queue");

            AssertStartedOldestFirst(_credentials.StartOrder);
        }

        [Test]
        public async Task AStoreThatIgnoresCancellation_NeverLetsPassesPileUp()
        {
            // A cancellation token asks; it does not compel. A pass started once per tick against a client
            // that ignores the token would grow by eight workers a tick and never come down, so a tick that
            // finds the previous pass still running cancels it and starts none of its own.
            Start(recheckSeconds: 300, heartbeatSeconds: 1, budget: 256);
            _credentials.IgnoresCancellation = true;

            RecordingWebSocket peer = Seat(0);
            for (var i = 1; i < 40; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);

            var highest = 0;
            for (var sample = 0; sample < 60; sample++)
            {
                highest = Math.Max(highest, _heartbeat.RecheckBatchesInFlight);
                await Task.Delay(100);
            }

            Assert.That(highest, Is.LessThanOrEqualTo(1), "one pass at a time, however badly it is going");
            Assert.That(peer.Sent.Count(m => m == "PING"), Is.GreaterThanOrEqualTo(3),
                "and the pings that keep every match alive still go out every tick");
        }

        [Test]
        public async Task AStoreThatThrows_LeavesTheSocketsOpenAndTheHeartbeatRunning()
        {
            // A database having a bad minute is not evidence against any player. Closing live sockets over
            // it would turn an outage in one dependency into an outage in the game.
            Start(recheckSeconds: 1, heartbeatSeconds: 1);
            _credentials.Throws = true;

            RecordingWebSocket peer = Seat(0);
            for (var i = 1; i < 5; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Started >= 10, TimeSpan.FromSeconds(20));

            Assert.That(_connections.Any(c => c.IsClosed), Is.False, "no socket is closed over a failed query");
            Assert.That(peer.Sent.Count(m => m == "PING"), Is.GreaterThanOrEqualTo(2),
                "and the heartbeat keeps going rather than dying on the first exception");
            Assert.That(
                _logs.Messages.Count(m => m.Contains("could not be re-checked", StringComparison.Ordinal)),
                Is.GreaterThanOrEqualTo(1), "each failure is reported once");
        }

        [Test]
        public async Task AStoreThatIgnoresCancellation_CannotAccumulateOrphanedCalls()
        {
            // Abandoning an await frees the WORKER. It does not free the store: the call this host stopped
            // listening to is still running, still holding a connection, still costing something. Counting
            // passes or workers cannot see those, so without a ceiling on the calls themselves they grow by
            // up to eight a pass for as long as the store misbehaves.
            Start(recheckSeconds: 300, heartbeatSeconds: 1, budget: 256);
            _credentials.IgnoresCancellation = true;

            RecordingWebSocket peer = Seat(0);
            for (var i = 1; i < 100; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);

            var highestOutstanding = 0;
            for (var sample = 0; sample < 80; sample++)
            {
                highestOutstanding = Math.Max(highestOutstanding, _heartbeat.OutstandingChecks);
                Assert.That(_credentials.InFlightNow,
                    Is.LessThanOrEqualTo(ConnectionHeartbeatService.MaxOutstandingChecks),
                    "the store is never given more calls than the ceiling allows");
                await Task.Delay(100);
            }

            Assert.That(highestOutstanding,
                Is.LessThanOrEqualTo(ConnectionHeartbeatService.MaxOutstandingChecks));
            Assert.That(_credentials.InFlightPeak,
                Is.LessThanOrEqualTo(ConnectionHeartbeatService.MaxOutstandingChecks),
                "including across passes, which is what workers alone cannot bound");
            Assert.That(peer.Sent.Count(m => m == "PING"), Is.GreaterThanOrEqualTo(3),
                "and the pings that keep every match alive still go out every tick");
            Assert.That(_connections.Any(c => c.IsClosed), Is.False,
                "a store that will not answer closes nobody: re-checks pause, they do not convict");

            // And when the store comes back, every slot returns and re-checks resume.
            long before = _credentials.Started;
            _credentials.ReleaseAll();

            await WaitUntil(() => _heartbeat.OutstandingChecks == 0, TimeSpan.FromSeconds(20));
            Assert.That(_heartbeat.OutstandingChecks, Is.Zero, "every abandoned call gave its slot back");

            await WaitUntil(() => _credentials.Started > before, TimeSpan.FromSeconds(20));
            Assert.That(_credentials.Started, Is.GreaterThan(before), "and checking starts again");
        }

        [Test]
        public async Task WhenEveryPermitIsHeld_TheStarvationWarningIsWrittenOnce()
        {
            // A cancelled cadence releases every waiting worker at the same instant, and each of them has
            // the same thing to report. A throttle that reads and then writes lets all eight through and
            // buries the fact it was meant to report in eight copies of itself.
            Start(recheckSeconds: 300, heartbeatSeconds: 1, budget: 256);
            _credentials.IgnoresCancellation = true;

            for (var i = 0; i < 60; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);

            // Two passes fill the sixteen permits; from the third on, every worker waits and is cancelled.
            await WaitUntil(
                () => _logs.Any("credential store is not answering"), TimeSpan.FromSeconds(20));
            await Task.Delay(3000);

            Assert.That(
                _logs.Messages.Count(m =>
                    m.Contains("credential store is not answering", StringComparison.Ordinal)),
                Is.EqualTo(1),
                "one worker owns the line; the other seven that woke with it stay quiet");
        }

        [Test]
        public async Task AStoreThatThrowsBeforeItReturnsATask_StillGivesItsPermitBack()
        {
            // A synchronous throw never produces a task, so there is nothing for the completion
            // continuation to hang off. The permit has to come back on that path too, or a store failing
            // this way would exhaust the ceiling in sixteen checks and pause re-checks for good.
            var throwing = new SynchronouslyThrowingCredentialService();
            Start(recheckSeconds: 1, heartbeatSeconds: 1, credentials: throwing);

            for (var i = 0; i < 4; i++) Seat(i);

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => throwing.Calls >= 4, TimeSpan.FromSeconds(20));
            await WaitUntil(() => _heartbeat.OutstandingChecks == 0, TimeSpan.FromSeconds(10));

            Assert.That(_heartbeat.OutstandingChecks, Is.Zero, "every permit came back");
            Assert.That(_connections.Any(c => c.IsClosed), Is.False,
                "a store that will not answer closes nobody");
            Assert.That(
                _logs.Messages.Count(m =>
                    m.Contains("could not be re-checked", StringComparison.Ordinal)),
                Is.GreaterThanOrEqualTo(4), "and each failure is reported once");
        }

        [Test]
        public async Task ABudgetSmallerThanTheDueSet_TakesExactlyTheOldest()
        {
            // Registered shuffled, so passing this means the heap ordered them and not the dictionary.
            DateTimeOffset oldest = DateTimeOffset.UtcNow.AddHours(-2);

            Start(recheckSeconds: 300, heartbeatSeconds: 5, budget: 8);

            foreach (int index in Shuffled(20))
                Seat(index, due: true, lastCheck: oldest.AddSeconds(index));

            await _heartbeat.StartAsync(CancellationToken.None);
            await WaitUntil(() => _credentials.Started >= 8, TimeSpan.FromSeconds(20));

            // Read between ticks: the next one is five seconds away.
            Assert.That(_credentials.Seen,
                Is.EquivalentTo(Enumerable.Range(0, 8).Select(i => (byte)i)),
                "the eight oldest, and none of the twelve younger ones");

            AssertStartedOldestFirst(_credentials.StartOrder);
        }

        /// <summary>
        /// Asserts nothing started while something meaningfully older was still waiting.
        ///
        /// Not a strict sort: eight workers pull the head of the batch at once, so the eight oldest start
        /// in whatever order the scheduler runs them. What must hold is that no socket starts more than one
        /// worker-wave ahead of the oldest thing still unstarted.
        /// </summary>
        static void AssertStartedOldestFirst(IReadOnlyList<byte> order)
        {
            for (var position = 0; position < order.Count; position++)
                Assert.That(order[position],
                    Is.LessThan(position + ConnectionHeartbeatService.MaxConcurrentRechecks),
                    "seat " + order[position].ToString() + " started at position " + position.ToString()
                    + ", jumping ahead of older sockets that had not started");
        }

        /// <summary>0..count-1 in an order that is not the natural one, with a fixed seed so a failure can
        /// be reproduced.</summary>
        static IEnumerable<int> Shuffled(int count)
        {
            var indices = Enumerable.Range(0, count).ToArray();
            var random = new Random(20260905);

            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            return indices;
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
        sealed class BlockingCredentialService(TimeSpan? takes = null) : IMatchCredentialService
        {
            /// <summary>Answers with a task that never completes and pays no attention to the token: a
            /// store client that treats cancellation as a suggestion.</summary>
            public bool IgnoresCancellation { get; set; }

            /// <summary>Fails every call, the way a database that is down does.</summary>
            public bool Throws { get; set; }

            readonly HashSet<byte> _blocked = new();
            readonly SemaphoreSlim _release = new(0);
            int _started;
            int _completed;
            int _cancelled;
            int _inFlight;
            int _inFlightPeak;

            readonly HashSet<byte> _seen = new();
            readonly List<byte> _order = new();
            readonly TaskCompletionSource _ignored = new(TaskCreationOptions.RunContinuationsAsynchronously);

            public int Started => Volatile.Read(ref _started);

            /// <summary>Which seats have been asked about at least once.</summary>
            public IReadOnlySet<byte> Seen
            {
                get { lock (_seen) return new HashSet<byte>(_seen); }
            }
            public int Completed => Volatile.Read(ref _completed);
            public int Cancelled => Volatile.Read(ref _cancelled);
            public int InFlightPeak => Volatile.Read(ref _inFlightPeak);

            /// <summary>Calls this fake has been given and has not finished. It counts the ones the caller
            /// has stopped waiting for, which is the whole point: a worker moving on does not end a query.</summary>
            public int InFlightNow => Volatile.Read(ref _inFlight);

            /// <summary>The seats checks were started for, in the order they started.</summary>
            public IReadOnlyList<byte> StartOrder
            {
                get { lock (_order) return _order.ToArray(); }
            }

            public void BlockFor(byte seat)
            {
                lock (_blocked) _blocked.Add(seat);
            }

            public void ReleaseAll()
            {
                _ignored.TrySetResult();
                _release.Release(int.MaxValue / 2);
            }

            public async Task<bool> IsStillValidAsync(
                byte[] credentialHash, Guid matchId, DateTimeOffset now, CancellationToken ct)
            {
                Interlocked.Increment(ref _started);
                RecordPeak(Interlocked.Increment(ref _inFlight));
                lock (_seen) _seen.Add(credentialHash[0]);
                lock (_order) _order.Add(credentialHash[0]);

                if (Throws)
                {
                    Interlocked.Decrement(ref _inFlight);
                    throw new InvalidOperationException("the database is not answering");
                }

                if (IgnoresCancellation)
                {
                    // No token, on purpose. What is being tested is the caller giving up on it - and the
                    // ceiling that stops those abandoned calls from accumulating without bound.
                    await _ignored.Task.ConfigureAwait(false);
                }

                bool blocked;
                lock (_blocked) blocked = _blocked.Contains(credentialHash[0]);

                try
                {
                    if (blocked) await _release.WaitAsync(ct).ConfigureAwait(false);

                    // Genuinely asynchronous when a test asks for it. A check that completes synchronously
                    // hides the scheduling bug this file exists to catch: the slot is free again before the
                    // caller has looked at it, so a scheduler that never refills still looks busy.
                    else if (takes is TimeSpan delay) await Task.Delay(delay, ct).ConfigureAwait(false);

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

        /// <summary>A store client that throws where it stands, before it has a task to fail.</summary>
        sealed class SynchronouslyThrowingCredentialService : IMatchCredentialService
        {
            int _calls;

            public int Calls => Volatile.Read(ref _calls);

            // Deliberately NOT async: an async method turns a throw into a faulted task, which is the
            // path that already has a continuation to release the permit. This one has none.
            public Task<bool> IsStillValidAsync(
                byte[] credentialHash, Guid matchId, DateTimeOffset now, CancellationToken ct)
            {
                Interlocked.Increment(ref _calls);
                throw new InvalidOperationException("the database is not answering");
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
