using System.Collections.Concurrent;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// What the retention sweeper does when eviction does not work.
    ///
    /// The rows are the easy half and are covered by the store contract. This is about the other half: a
    /// match is abandoned in the database and its players are still connected to it, so an eviction that
    /// throws or never returns leaves people sitting in a game the record says is over. A loop that died on
    /// the first of those would stop applying the whole retention policy, silently, until the next deploy.
    /// </summary>
    [TestFixture]
    public class MatchRetentionServiceTests
    {
        static readonly DateTimeOffset Start = new(2026, 4, 1, 0, 0, 0, TimeSpan.Zero);

        FakeTimeProvider _clock = null!;
        RetiringStore _store = null!;
        RecordingEvictor _evictor = null!;
        MatchRetentionService _service = null!;

        [SetUp]
        public void Build()
        {
            _clock = new FakeTimeProvider(Start);
            _store = new RetiringStore();
            _evictor = new RecordingEvictor();
            _service = new MatchRetentionService(
                _store,
                _evictor,
                Options.Create(new MatchHostingOptions()),
                _clock,
                NullLogger<MatchRetentionService>.Instance);
        }

        static Guid Id(byte seed) => new(new byte[] { seed, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1 });

        [Test]
        public async Task AnEvictionThatThrows_LeavesTheOthersEvictedAndIsRetriedNextSweep()
        {
            Guid first = Id(1), stubborn = Id(2), last = Id(3);
            _store.Abandon(first, stubborn, last);
            _evictor.ThrowFor(stubborn, new InvalidOperationException("that socket would not close"));

            await _service.SweepOnceAsync(CancellationToken.None);

            Assert.That(_evictor.Evicted, Is.EquivalentTo(new[] { first, last }),
                "one match that would not close is not a reason to stop closing the others");
            Assert.That(_service.PendingEvictions, Is.EqualTo(1));

            // The next sweep finds nothing to abandon and still tries the id it could not close.
            _store.Abandon();
            _evictor.Clear();

            await _service.SweepOnceAsync(CancellationToken.None);

            Assert.That(_evictor.Offered, Does.Contain(stubborn), "the id is retried rather than lost");
            Assert.That(_evictor.Evicted, Does.Contain(stubborn));
            Assert.That(_service.PendingEvictions, Is.EqualTo(0), "and dropped once it worked");
        }

        [Test]
        public async Task AnEvictionThatNeverReturns_DoesNotHoldUpTheSweep()
        {
            Guid stuck = Id(7), other = Id(8);
            _store.Abandon(stuck, other);
            _evictor.BlockOn(stuck);

            Task<RetentionResult> sweep = _service.SweepOnceAsync(CancellationToken.None);

            // The deadline runs on the injected clock, so the wait is the test winding it forward rather
            // than the test sleeping for five seconds.
            await WaitUntil(() => _clock.ScheduledTimers > 0);
            _clock.Advance(MatchRetentionService.EvictionTimeout);

            await sweep.WaitAsync(TimeSpan.FromSeconds(20));

            Assert.That(_evictor.Evicted, Does.Contain(other),
                "the sweep moved past the match it could not close and finished the rest");
            Assert.That(_service.PendingEvictions, Is.EqualTo(1));

            _evictor.Unblock();
        }

        [Test]
        public async Task ShutdownIsNotDelayedByAnEvictionInFlight()
        {
            _store.Abandon(Id(9));
            _evictor.BlockOn(Id(9));

            await _service.StartAsync(CancellationToken.None);

            // One cadence, so the loop is inside an eviction that is never going to answer.
            await WaitUntil(() => _clock.ScheduledTimers > 0);
            _clock.Advance(TimeSpan.FromMinutes(MatchHostingOptions.DefaultRetentionSweepMinutes));
            await WaitUntil(() => _evictor.Offered.Count > 0);

            var stopping = System.Diagnostics.Stopwatch.StartNew();
            await _service.StopAsync(CancellationToken.None).WaitAsync(TimeSpan.FromSeconds(20));
            stopping.Stop();

            Assert.That(stopping.Elapsed, Is.LessThan(TimeSpan.FromSeconds(10)),
                "a wedged eviction must not hold a deploy open");

            _evictor.Unblock();
        }

        [Test]
        public async Task AStoreThatRefusesTheSweep_LeavesTheLoopRunning()
        {
            _store.Fail(new InvalidOperationException("the database went away"));

            RetentionResult first = await _service.SweepOnceAsync(CancellationToken.None);
            Assert.That(first.IsEmpty, Is.True);

            _store.Fail(null);
            _store.Abandon(Id(4));

            RetentionResult second = await _service.SweepOnceAsync(CancellationToken.None);
            Assert.That(second.Abandoned, Is.EqualTo(1));
            Assert.That(_evictor.Evicted, Does.Contain(Id(4)));
        }

        [Test]
        public async Task ThePendingSetIsBounded()
        {
            var many = new Guid[MatchRetentionService.MaxPendingEvictions + 10];
            for (var i = 0; i < many.Length; i++) many[i] = Guid.NewGuid();

            _store.Abandon(many);
            _evictor.ThrowForEverything(new InvalidOperationException("nothing closes today"));

            await _service.SweepOnceAsync(CancellationToken.None);

            Assert.That(_service.PendingEvictions, Is.EqualTo(MatchRetentionService.MaxPendingEvictions),
                "a record of what the host is failing to do must not itself become the outage");
        }

        static async Task WaitUntil(Func<bool> ready)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.AddSeconds(10);
            while (DateTimeOffset.UtcNow < deadline)
            {
                if (ready()) return;
                await Task.Delay(5);
            }

            Assert.Fail("the service never reached the state this test was waiting for");
        }

        /// <summary>An evictor a test can make fail in each of the ways a real one can.</summary>
        sealed class RecordingEvictor : IMatchEvictor
        {
            readonly ConcurrentDictionary<Guid, Exception> _throwFor = new();
            readonly ConcurrentDictionary<Guid, byte> _blockOn = new();
            readonly List<Guid> _offered = new();
            readonly List<Guid> _evicted = new();
            readonly object _gate = new();

            TaskCompletionSource _blocked = new(TaskCreationOptions.RunContinuationsAsynchronously);
            Exception? _throwForEverything;

            public IReadOnlyList<Guid> Offered
            {
                get { lock (_gate) return _offered.ToArray(); }
            }

            public IReadOnlyList<Guid> Evicted
            {
                get { lock (_gate) return _evicted.ToArray(); }
            }

            public void ThrowFor(Guid matchId, Exception failure) => _throwFor[matchId] = failure;

            public void ThrowForEverything(Exception failure) => _throwForEverything = failure;

            public void BlockOn(Guid matchId) => _blockOn[matchId] = 0;

            /// <summary>Ids this evictor answers to a store-status re-derivation, standing in for matches
            /// whose rows are terminal while this host still holds them.</summary>
            public List<Guid> Reaped { get; } = new();

            /// <summary>The token the last eviction was handed, so a test can see a deadline reach it.</summary>
            public CancellationToken LastToken { get; private set; }

            public void Unblock() => _blocked.TrySetResult();

            public void Clear()
            {
                lock (_gate)
                {
                    _offered.Clear();
                    _evicted.Clear();
                }

                _throwFor.Clear();
                _blockOn.Clear();
                _throwForEverything = null;
            }

            public async Task<IReadOnlyList<Guid>> EvictAsync(
                IEnumerable<Guid> matchIds, int closeStatus, string reason, CancellationToken ct)
            {
                LastToken = ct;

                foreach (Guid matchId in matchIds)
                {
                    lock (_gate) _offered.Add(matchId);

                    if (_throwForEverything is { } everything) throw everything;
                    if (_throwFor.TryGetValue(matchId, out Exception? failure)) throw failure;

                    if (_blockOn.ContainsKey(matchId))
                    {
                        await _blocked.Task.WaitAsync(ct).ConfigureAwait(false);
                        continue;
                    }

                    lock (_gate) _evicted.Add(matchId);
                }

                return Array.Empty<Guid>();
            }

            public Task<IReadOnlyList<Guid>> EvictReapedAsync(
                int closeStatus, string reason, CancellationToken ct) =>
                EvictAsync(Reaped.ToArray(), closeStatus, reason, ct);
        }

        /// <summary>A store whose only job is to answer one retention call the way a test wants.</summary>
        sealed class RetiringStore : IMatchStore
        {
            Guid[] _abandoned = Array.Empty<Guid>();
            Exception? _failure;

            public void Abandon(params Guid[] abandoned) => _abandoned = abandoned;

            public void Fail(Exception? failure) => _failure = failure;

            public Task<RetentionResult> ApplyRetentionAsync(
                RetentionPolicy policy, DateTimeOffset now, CancellationToken ct)
            {
                if (_failure is { } failure) throw failure;

                return Task.FromResult(
                    new RetentionResult(0, _abandoned.Length, 0, 0, _abandoned));
            }

            public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<bool> TryStartMatchAsync(
                Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<AppendResult> AppendCommandAsync(
                Guid matchId, int expectedSequence, string commandWire, string issuerSteamId,
                DateTimeOffset acceptedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<bool> TryCompleteMatchAsync(
                Guid matchId, MatchStatus terminal, int? winnerSeat, DateTimeOffset completedAt,
                CancellationToken ct) =>
                throw new NotSupportedException();

            public Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct) =>
                throw new NotSupportedException();

            public Task StoreJoinCredentialAsync(
                byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt,
                CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task RevokeJoinCredentialsAsync(
                Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct) =>
                throw new NotSupportedException();

            public Task<CredentialReplacement> ReplaceJoinCredentialAsync(
                byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt,
                DateTimeOffset now, CancellationToken ct, TimeSpan? allowTerminalWithin = null) =>
                throw new NotSupportedException();
        }
    }
}
