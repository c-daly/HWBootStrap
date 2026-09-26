using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// How the startup recovery pass behaves while the host around it is being taken apart.
    ///
    /// Stopping and disposing reach this service from two different places - the host stops hosted
    /// services, the container disposes singletons - and NOTHING in either contract says they arrive in
    /// that order. Disposing an IHost disposes its provider without stopping anything, which is what a
    /// synchronous host dispose does and what `using var host = builder.Build()` does. So both orders have
    /// to work, and both are asserted here rather than assumed.
    /// </summary>
    [TestFixture]
    public class RecoveryStartupServiceTests
    {
        static readonly DateTimeOffset Start = new(2026, 5, 1, 0, 0, 0, TimeSpan.Zero);

        FakeTimeProvider _clock = null!;
        RecoveryState _state = null!;

        [SetUp]
        public void Build()
        {
            _clock = new FakeTimeProvider(Start);
            _state = new RecoveryState();
        }

        /// <summary>A service whose first pass succeeds, so no retry loop is ever started.</summary>
        RecoveryStartupService WithNothingToVerify() =>
            new(_state, recovery: null, _clock, NullLogger<RecoveryStartupService>.Instance);

        /// <summary>A service whose first pass fails, so the background retry loop IS running.</summary>
        RecoveryStartupService WithAStoreThatWillNotAnswer()
        {
            var store = new UnreachableStore();
            var recovery = new MatchRecoveryService(
                store,
                Options.Create(new MatchHostingOptions()),
                _clock,
                new MatchMetrics(),
                NullLogger<MatchRecoveryService>.Instance);

            return new RecoveryStartupService(
                _state, recovery, _clock, NullLogger<RecoveryStartupService>.Instance);
        }

        [Test]
        public async Task StopThenDisposeThenStopAgain_IsSafe()
        {
            RecoveryStartupService service = WithNothingToVerify();
            await service.StartAsync(CancellationToken.None);

            await service.StopAsync(CancellationToken.None);
            service.Dispose();

            // The sequence a host teardown actually performs when it is torn down twice, which is what a
            // test factory disposed both synchronously and asynchronously does.
            Assert.DoesNotThrowAsync(() => service.StopAsync(CancellationToken.None));
            Assert.DoesNotThrow(service.Dispose);
        }

        [Test]
        public async Task DisposeThenStop_IsSafe()
        {
            RecoveryStartupService service = WithAStoreThatWillNotAnswer();
            await service.StartAsync(CancellationToken.None);

            Assert.That(service.RetryLoop, Is.Not.Null, "the retry loop has to be running, or this proves nothing");

            // Disposing an IHost disposes the container WITHOUT stopping hosted services, so this order is
            // not hypothetical - it is what a synchronous host dispose does.
            service.Dispose();

            Assert.DoesNotThrowAsync(() => service.StopAsync(CancellationToken.None));
        }

        [Test]
        public async Task DisposeWithoutStopping_UnwindsTheRetryLoop()
        {
            RecoveryStartupService service = WithAStoreThatWillNotAnswer();
            await service.StartAsync(CancellationToken.None);

            Task? retries = service.RetryLoop;
            Assert.That(retries, Is.Not.Null);
            Assert.That(retries!.IsCompleted, Is.False, "the loop is waiting on its backoff");

            // Disposing the source without cancelling it took the delay registration with it, so the loop
            // never woke again and was stranded for the life of the process, holding its host behind it.
            service.Dispose();

            await retries.WaitAsync(TimeSpan.FromSeconds(10));
            Assert.That(retries.IsCompletedSuccessfully, Is.True);
        }

        [Test]
        public async Task AHostDisposedWithoutBeingStopped_CanStillBeStopped()
        {
            // The same thing through the real hosting API rather than by poking the service, because the
            // ordering being asserted is the host contract and not a detail of this class.
            IHost host = new HostBuilder()
                .ConfigureServices(services =>
                {
                    services.AddSingleton(_state);
                    services.AddSingleton<TimeProvider>(_clock);
                    services.AddSingleton<IHostedService>(provider => new RecoveryStartupService(
                        provider.GetRequiredService<RecoveryState>(),
                        recovery: null,
                        provider.GetRequiredService<TimeProvider>(),
                        NullLogger<RecoveryStartupService>.Instance));
                })
                .Build();

            await host.StartAsync();

            host.Dispose();

            Assert.DoesNotThrowAsync(() => host.StopAsync());
        }

        /// <summary>A store whose open-match query never answers, so the first pass always fails.</summary>
        sealed class UnreachableStore : IMatchStore
        {
            public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct) =>
                Task.FromException<IReadOnlyList<Guid>>(
                    new InvalidOperationException("the database is not answering"));

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

            public Task<RetentionResult> ApplyRetentionAsync(
                RetentionPolicy policy, DateTimeOffset now, CancellationToken ct) =>
                throw new NotSupportedException();
        }
    }
}
