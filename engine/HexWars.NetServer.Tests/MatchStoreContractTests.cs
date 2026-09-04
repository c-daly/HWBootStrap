using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Everything an <see cref="IMatchStore"/> must do, written once and run against both implementations.
    ///
    /// The whole point of this class is that the in-memory double is not allowed to be a convenient lie: the
    /// coordinator tests that will lean on <see cref="InMemoryMatchStore"/> are only meaningful if it refuses
    /// the same appends, honours the same status edges and returns the same idempotency answers as Postgres.
    /// A behaviour that is only asserted for one of the two belongs in that one fixture alone.
    /// </summary>
    public abstract class MatchStoreContractTests
    {
        protected const string SetupWire = "annihilation 9 7 0 7 3 1 1 1 3 0";
        protected const string EngineVersion = "hexwars-engine/1";
        protected const string BuildId = "test-build";
        protected const int ProtocolVersion = 2;

        /// <summary>Whole seconds on purpose: Postgres stores timestamptz to the microsecond, so a test clock
        /// with sub-microsecond ticks would compare unequal after a round trip for reasons that say nothing
        /// about the store.</summary>
        protected static readonly DateTimeOffset Created = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);
        protected static readonly DateTimeOffset Started = Created.AddMinutes(1);
        protected static readonly DateTimeOffset Move1 = Created.AddMinutes(2);
        protected static readonly DateTimeOffset Move2 = Created.AddMinutes(3);
        protected static readonly DateTimeOffset Move3 = Created.AddMinutes(4);
        protected static readonly DateTimeOffset Ended = Created.AddMinutes(5);

        static int _identifiers;

        protected IMatchStore Store = null!;

        protected static CancellationToken Ct => CancellationToken.None;

        /// <summary>A store backed by empty storage. Called once per test.</summary>
        protected abstract Task<IMatchStore> CreateStoreAsync();

        [SetUp]
        public async Task StartFromAnEmptyStore() => Store = await CreateStoreAsync();

        /// <summary>Ids are unique across the whole run so a fixture that shares one database between tests
        /// still cannot collide on the open-lobby index.</summary>
        protected static string NextLobbyId() =>
            "1097752" + Interlocked.Increment(ref _identifiers).ToString("D11");

        protected static string NextSteamId() =>
            "7656119" + Interlocked.Increment(ref _identifiers).ToString("D10");

        protected static CreateMatchRequest Request(
            string lobbyId, string seat0, string seat1, DateTimeOffset? createdAt = null) =>
            new(lobbyId, SetupWire, EngineVersion, ProtocolVersion, BuildId,
                new[] { (seat0, 0), (seat1, 1) }, createdAt ?? Created);

        /// <summary>Creates a waiting match with two fresh players and hands back everything a test needs.</summary>
        protected async Task<(Guid MatchId, string Lobby, string Seat0, string Seat1)> NewWaitingMatchAsync(
            DateTimeOffset? createdAt = null)
        {
            string lobby = NextLobbyId(), seat0 = NextSteamId(), seat1 = NextSteamId();
            CreateMatchResult created =
                await Store.CreateMatchForLobbyAsync(Request(lobby, seat0, seat1, createdAt), Ct);
            return (created.Match.MatchId, lobby, seat0, seat1);
        }

        protected async Task<(Guid MatchId, string Lobby, string Seat0, string Seat1)> NewActiveMatchAsync(
            DateTimeOffset? createdAt = null)
        {
            var match = await NewWaitingMatchAsync(createdAt);
            bool started = await Store.TryStartMatchAsync(match.MatchId, "START-REPLAY", Started, Ct);
            Assert.That(started, Is.True, "the fixture could not start the match it just created");
            return match;
        }

        protected static byte[] Hash(byte seed)
        {
            var hash = new byte[32];
            for (int i = 0; i < hash.Length; i++) hash[i] = (byte)(seed + i);
            return hash;
        }

        // ---- creation --------------------------------------------------------

        [Test]
        public async Task CreateThenGet_ReturnsTheSameWaitingMatchWithBothSeats()
        {
            string lobby = NextLobbyId(), seat0 = NextSteamId(), seat1 = NextSteamId();

            CreateMatchResult result = await Store.CreateMatchForLobbyAsync(Request(lobby, seat0, seat1), Ct);

            Assert.That(result.Created, Is.True);

            PersistedMatch? stored = await Store.GetMatchAsync(result.Match.MatchId, Ct);
            Assert.That(stored, Is.Not.Null);
            Assert.That(stored, Is.EqualTo(result.Match), "the record returned by create must be the stored row");
            Assert.That(stored!.SteamLobbyId, Is.EqualTo(lobby));
            Assert.That(stored.Status, Is.EqualTo(MatchStatus.Waiting));
            Assert.That(stored.SetupWire, Is.EqualTo(SetupWire));
            Assert.That(stored.EngineVersion, Is.EqualTo(EngineVersion));
            Assert.That(stored.ProtocolVersion, Is.EqualTo(ProtocolVersion));
            Assert.That(stored.BuildId, Is.EqualTo(BuildId));
            Assert.That(stored.StartReplay, Is.Null);
            Assert.That(stored.StartedAt, Is.Null);
            Assert.That(stored.CompletedAt, Is.Null);
            Assert.That(stored.WinnerSeat, Is.Null);
            Assert.That(stored.CreatedAt, Is.EqualTo(Created));
            Assert.That(stored.LastActivityAt, Is.EqualTo(stored.CreatedAt));

            IReadOnlyList<PersistedPlayer> players = await Store.GetPlayersAsync(result.Match.MatchId, Ct);
            Assert.That(players.Select(p => p.Seat), Is.EqualTo(new[] { 0, 1 }));
            Assert.That(players.Select(p => p.SteamId), Is.EqualTo(new[] { seat0, seat1 }));
            Assert.That(players.Select(p => p.CatalogWire), Is.All.Null);
            Assert.That(players.Select(p => p.LastSeenAt), Is.All.Null);
            Assert.That(players.Select(p => p.JoinedAt), Is.All.EqualTo(Created));
            Assert.That(players.Select(p => p.MatchId), Is.All.EqualTo(result.Match.MatchId));

            PersistedPlayer? one = await Store.GetPlayerAsync(result.Match.MatchId, seat1, Ct);
            Assert.That(one, Is.Not.Null);
            Assert.That(one!.Seat, Is.EqualTo(1));

            Assert.That(await Store.GetMatchAsync(Guid.NewGuid(), Ct), Is.Null);
            Assert.That(await Store.GetPlayerAsync(result.Match.MatchId, "76561190000000000", Ct), Is.Null);
        }

        [Test]
        public async Task CreateTwiceForTheSameLobby_ReturnsTheMatchThatAlreadyExists()
        {
            string lobby = NextLobbyId(), seat0 = NextSteamId(), seat1 = NextSteamId();

            CreateMatchResult first = await Store.CreateMatchForLobbyAsync(Request(lobby, seat0, seat1), Ct);
            CreateMatchResult second = await Store.CreateMatchForLobbyAsync(
                Request(lobby, seat0, seat1, Created.AddMinutes(1)), Ct);

            Assert.That(second.Created, Is.False);
            Assert.That(second.Match.MatchId, Is.EqualTo(first.Match.MatchId));
            Assert.That(second.Match.CreatedAt, Is.EqualTo(Created), "the second call must not rewrite the row");

            PersistedMatch? open = await Store.FindOpenMatchForLobbyAsync(lobby, Ct);
            Assert.That(open, Is.Not.Null);
            Assert.That(open!.MatchId, Is.EqualTo(first.Match.MatchId));
            Assert.That(await Store.FindOpenMatchForLobbyAsync(NextLobbyId(), Ct), Is.Null);
        }

        [Test]
        public async Task CreateForADifferentLobby_MakesADifferentMatch()
        {
            CreateMatchResult first = await Store.CreateMatchForLobbyAsync(
                Request(NextLobbyId(), NextSteamId(), NextSteamId()), Ct);
            CreateMatchResult second = await Store.CreateMatchForLobbyAsync(
                Request(NextLobbyId(), NextSteamId(), NextSteamId()), Ct);

            Assert.That(second.Created, Is.True);
            Assert.That(second.Match.MatchId, Is.Not.EqualTo(first.Match.MatchId));
        }

        [Test]
        public async Task AfterTheMatchIsCompleted_TheSameLobbyCanHostANewOne()
        {
            string lobby = NextLobbyId();
            CreateMatchResult first = await Store.CreateMatchForLobbyAsync(
                Request(lobby, NextSteamId(), NextSteamId()), Ct);
            await Store.TryStartMatchAsync(first.Match.MatchId, "START-REPLAY", Started, Ct);
            Assert.That(
                await Store.TryCompleteMatchAsync(first.Match.MatchId, MatchStatus.Completed, 0, Ended, Ct),
                Is.True);

            Assert.That(await Store.FindOpenMatchForLobbyAsync(lobby, Ct), Is.Null);

            CreateMatchResult second = await Store.CreateMatchForLobbyAsync(
                Request(lobby, NextSteamId(), NextSteamId(), Ended.AddMinutes(1)), Ct);

            Assert.That(second.Created, Is.True);
            Assert.That(second.Match.MatchId, Is.Not.EqualTo(first.Match.MatchId));
            Assert.That((await Store.FindOpenMatchForLobbyAsync(lobby, Ct))!.MatchId,
                Is.EqualTo(second.Match.MatchId));
        }

        [Test]
        public void CreateWithASingleSeat_IsRejected()
        {
            var request = new CreateMatchRequest(NextLobbyId(), SetupWire, EngineVersion, ProtocolVersion,
                BuildId, new[] { (NextSteamId(), 0) }, Created);

            Assert.ThrowsAsync<ArgumentException>(() => Store.CreateMatchForLobbyAsync(request, Ct));
        }

        [Test]
        public void CreateWithTwoPlayersInOneSeat_IsRejected()
        {
            var request = new CreateMatchRequest(NextLobbyId(), SetupWire, EngineVersion, ProtocolVersion,
                BuildId, new[] { (NextSteamId(), 0), (NextSteamId(), 0) }, Created);

            Assert.ThrowsAsync<ArgumentException>(() => Store.CreateMatchForLobbyAsync(request, Ct));
        }

        [Test]
        public void CreateWithTheSamePlayerInBothSeats_IsRejected()
        {
            string steamId = NextSteamId();
            var request = new CreateMatchRequest(NextLobbyId(), SetupWire, EngineVersion, ProtocolVersion,
                BuildId, new[] { (steamId, 0), (steamId, 1) }, Created);

            Assert.ThrowsAsync<ArgumentException>(() => Store.CreateMatchForLobbyAsync(request, Ct));
        }

        // ---- catalogues ------------------------------------------------------

        [Test]
        public async Task SaveCatalog_IsVisibleInTheJournal()
        {
            var match = await NewWaitingMatchAsync();

            await Store.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-seat-0", Ct);

            MatchJournal? journal = await Store.LoadJournalAsync(match.MatchId, Ct);
            Assert.That(journal, Is.Not.Null);
            Assert.That(journal!.Match.Status, Is.EqualTo(MatchStatus.Waiting));
            Assert.That(journal.Commands, Is.Empty);
            Assert.That(journal.Players.Single(p => p.SteamId == match.Seat0).CatalogWire,
                Is.EqualTo("catalog-seat-0"));
            Assert.That(journal.Players.Single(p => p.SteamId == match.Seat1).CatalogWire, Is.Null);

            Assert.That(await Store.LoadJournalAsync(Guid.NewGuid(), Ct), Is.Null);
        }

        [Test]
        public async Task SaveCatalog_AfterTheMatchStarted_ChangesNothing()
        {
            var match = await NewWaitingMatchAsync();
            await Store.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-before-start", Ct);
            await Store.TryStartMatchAsync(match.MatchId, "START-REPLAY", Started, Ct);

            await Store.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-after-start", Ct);

            PersistedPlayer? player = await Store.GetPlayerAsync(match.MatchId, match.Seat0, Ct);
            Assert.That(player!.CatalogWire, Is.EqualTo("catalog-before-start"));
        }

        // ---- starting --------------------------------------------------------

        [Test]
        public async Task StartMatch_SucceedsOnceAndStoresTheStartReplay()
        {
            var match = await NewWaitingMatchAsync();

            Assert.That(await Store.TryStartMatchAsync(match.MatchId, "REPLAY-ONE", Started, Ct), Is.True);
            Assert.That(await Store.TryStartMatchAsync(match.MatchId, "REPLAY-TWO", Move1, Ct), Is.False);

            PersistedMatch? stored = await Store.GetMatchAsync(match.MatchId, Ct);
            Assert.That(stored!.Status, Is.EqualTo(MatchStatus.Active));
            Assert.That(stored.StartReplay, Is.EqualTo("REPLAY-ONE"));
            Assert.That(stored.StartedAt, Is.EqualTo(Started));
            Assert.That(stored.LastActivityAt, Is.EqualTo(Started));

            Assert.That(await Store.TryStartMatchAsync(Guid.NewGuid(), "REPLAY", Started, Ct), Is.False);
        }

        // ---- the command journal ---------------------------------------------

        [Test]
        public async Task AppendCommands_AreStoredInSequenceOrderWithTheirIssuers()
        {
            var match = await NewActiveMatchAsync();

            Assert.That((await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat0, Move1, Ct)),
                Is.EqualTo(new AppendResult(AppendStatus.Appended, 1)));
            Assert.That((await Store.AppendCommandAsync(match.MatchId, 2, "A 0 2 5", match.Seat0, Move2, Ct)),
                Is.EqualTo(new AppendResult(AppendStatus.Appended, 2)));
            Assert.That((await Store.AppendCommandAsync(match.MatchId, 3, "E 1", match.Seat1, Move3, Ct)),
                Is.EqualTo(new AppendResult(AppendStatus.Appended, 3)));

            MatchJournal? journal = await Store.LoadJournalAsync(match.MatchId, Ct);
            Assert.That(journal!.Commands.Select(c => c.Sequence), Is.EqualTo(new[] { 1, 2, 3 }));
            Assert.That(journal.Commands.Select(c => c.CommandWire),
                Is.EqualTo(new[] { "M 0 2 3 0", "A 0 2 5", "E 1" }));
            Assert.That(journal.Commands.Select(c => c.IssuerSteamId),
                Is.EqualTo(new[] { match.Seat0, match.Seat0, match.Seat1 }));
            Assert.That(journal.Commands.Select(c => c.AcceptedAt), Is.EqualTo(new[] { Move1, Move2, Move3 }));
            Assert.That(journal.Commands.Select(c => c.MatchId), Is.All.EqualTo(match.MatchId));
            Assert.That(journal.Match.LastActivityAt, Is.EqualTo(Move3));
        }

        [Test]
        public async Task AppendCommand_AheadOfTheNextSequence_Conflicts()
        {
            var match = await NewActiveMatchAsync();
            await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat0, Move1, Ct);

            AppendResult result =
                await Store.AppendCommandAsync(match.MatchId, 5, "E 0", match.Seat0, Move2, Ct);

            Assert.That(result.Status, Is.EqualTo(AppendStatus.Conflict));
            Assert.That(result.Sequence, Is.EqualTo(2), "a conflict reports the sequence the store expects next");
            Assert.That((await Store.LoadJournalAsync(match.MatchId, Ct))!.Commands, Has.Count.EqualTo(1));
        }

        [Test]
        public async Task AppendCommand_ReplayedExactly_IsAlreadyApplied()
        {
            var match = await NewActiveMatchAsync();
            await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat0, Move1, Ct);

            AppendResult retry =
                await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat0, Move2, Ct);

            Assert.That(retry, Is.EqualTo(new AppendResult(AppendStatus.AlreadyApplied, 1)));

            MatchJournal? journal = await Store.LoadJournalAsync(match.MatchId, Ct);
            Assert.That(journal!.Commands, Has.Count.EqualTo(1));
            Assert.That(journal.Commands[0].AcceptedAt, Is.EqualTo(Move1),
                "the retry must not rewrite the accepted timestamp of the stored command");
        }

        [Test]
        public async Task AppendCommand_ReusingASequenceForDifferentContent_Conflicts()
        {
            var match = await NewActiveMatchAsync();
            await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat0, Move1, Ct);

            AppendResult otherWire =
                await Store.AppendCommandAsync(match.MatchId, 1, "E 0", match.Seat0, Move2, Ct);
            AppendResult otherIssuer =
                await Store.AppendCommandAsync(match.MatchId, 1, "M 0 2 3 0", match.Seat1, Move2, Ct);

            Assert.That(otherWire.Status, Is.EqualTo(AppendStatus.Conflict));
            Assert.That(otherIssuer.Status, Is.EqualTo(AppendStatus.Conflict),
                "the issuer is part of what makes a retry the same command");
            Assert.That((await Store.LoadJournalAsync(match.MatchId, Ct))!.Commands[0].CommandWire,
                Is.EqualTo("M 0 2 3 0"));
        }

        [Test]
        public async Task AppendCommand_OnAMatchThatIsNotActive_IsRefused()
        {
            var waiting = await NewWaitingMatchAsync();

            AppendResult onWaiting =
                await Store.AppendCommandAsync(waiting.MatchId, 1, "E 0", waiting.Seat0, Move1, Ct);
            AppendResult onUnknown =
                await Store.AppendCommandAsync(Guid.NewGuid(), 1, "E 0", waiting.Seat0, Move1, Ct);

            Assert.That(onWaiting, Is.EqualTo(new AppendResult(AppendStatus.MatchNotActive, 1)));
            Assert.That(onUnknown.Status, Is.EqualTo(AppendStatus.MatchNotActive));
            Assert.That((await Store.LoadJournalAsync(waiting.MatchId, Ct))!.Commands, Is.Empty);
        }

        // ---- terminal statuses -----------------------------------------------

        [Test]
        public async Task CompleteMatch_FromActive_StoresTheWinnerAndTheTimestamp()
        {
            var match = await NewActiveMatchAsync();

            Assert.That(
                await Store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, 1, Ended, Ct), Is.True);

            PersistedMatch? stored = await Store.GetMatchAsync(match.MatchId, Ct);
            Assert.That(stored!.Status, Is.EqualTo(MatchStatus.Completed));
            Assert.That(stored.WinnerSeat, Is.EqualTo(1));
            Assert.That(stored.CompletedAt, Is.EqualTo(Ended));
            Assert.That(stored.LastActivityAt, Is.EqualTo(Ended));
        }

        [Test]
        public async Task CompleteMatch_FromATerminalStatus_IsRefused()
        {
            var match = await NewActiveMatchAsync();
            await Store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, 1, Ended, Ct);

            Assert.That(
                await Store.TryCompleteMatchAsync(
                    match.MatchId, MatchStatus.Abandoned, null, Ended.AddMinutes(1), Ct),
                Is.False);

            PersistedMatch? stored = await Store.GetMatchAsync(match.MatchId, Ct);
            Assert.That(stored!.Status, Is.EqualTo(MatchStatus.Completed));
            Assert.That(stored.WinnerSeat, Is.EqualTo(1), "a refused transition must not touch the row");
            Assert.That(stored.CompletedAt, Is.EqualTo(Ended));
        }

        [Test]
        public async Task ExpireMatch_FromWaiting_IsAllowed()
        {
            var match = await NewWaitingMatchAsync();

            Assert.That(
                await Store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Expired, null, Ended, Ct), Is.True);

            PersistedMatch? stored = await Store.GetMatchAsync(match.MatchId, Ct);
            Assert.That(stored!.Status, Is.EqualTo(MatchStatus.Expired));
            Assert.That(stored.WinnerSeat, Is.Null);
            Assert.That(stored.CompletedAt, Is.EqualTo(Ended));
        }

        [Test]
        public async Task CompleteMatch_FromWaiting_IsRefused()
        {
            var match = await NewWaitingMatchAsync();

            Assert.That(
                await Store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, 0, Ended, Ct), Is.False);
            Assert.That((await Store.GetMatchAsync(match.MatchId, Ct))!.Status,
                Is.EqualTo(MatchStatus.Waiting));
        }

        // ---- liveness --------------------------------------------------------

        [Test]
        public async Task Touch_AdvancesMatchActivityAndThePlayerLastSeen()
        {
            var match = await NewWaitingMatchAsync();

            await Store.TouchAsync(match.MatchId, match.Seat0, Move1, Ct);

            Assert.That((await Store.GetMatchAsync(match.MatchId, Ct))!.LastActivityAt, Is.EqualTo(Move1));
            Assert.That((await Store.GetPlayerAsync(match.MatchId, match.Seat0, Ct))!.LastSeenAt,
                Is.EqualTo(Move1));
            Assert.That((await Store.GetPlayerAsync(match.MatchId, match.Seat1, Ct))!.LastSeenAt, Is.Null);

            await Store.TouchAsync(match.MatchId, null, Move2, Ct);

            Assert.That((await Store.GetMatchAsync(match.MatchId, Ct))!.LastActivityAt, Is.EqualTo(Move2));
            Assert.That((await Store.GetPlayerAsync(match.MatchId, match.Seat0, Ct))!.LastSeenAt,
                Is.EqualTo(Move1), "a match-level touch must not claim the player was seen");
        }

        [Test]
        public async Task ListOpenMatchIds_ListsWaitingAndActiveInCreationOrder()
        {
            var waiting = await NewWaitingMatchAsync(Created);
            var active = await NewActiveMatchAsync(Created.AddMinutes(1));
            var expired = await NewWaitingMatchAsync(Created.AddMinutes(2));
            await Store.TryCompleteMatchAsync(expired.MatchId, MatchStatus.Expired, null, Ended, Ct);

            IReadOnlyList<Guid> open = await Store.ListOpenMatchIdsAsync(Ct);

            Assert.That(open, Is.EqualTo(new[] { waiting.MatchId, active.MatchId }));
        }

        // ---- join credentials ------------------------------------------------

        [Test]
        public async Task JoinCredential_RoundTripsByHash()
        {
            var match = await NewWaitingMatchAsync();
            byte[] hash = Hash(1);
            DateTimeOffset expiresAt = Created.AddMinutes(15);

            await Store.StoreJoinCredentialAsync(hash, match.MatchId, match.Seat0, expiresAt, Ct);

            JoinCredentialRecord? found = await Store.FindJoinCredentialAsync(Hash(1), Ct);
            Assert.That(found, Is.Not.Null);
            Assert.That(found!.CredentialHash, Is.EqualTo(hash));
            Assert.That(found.MatchId, Is.EqualTo(match.MatchId));
            Assert.That(found.SteamId, Is.EqualTo(match.Seat0));
            Assert.That(found.ExpiresAt, Is.EqualTo(expiresAt));
            Assert.That(found.RevokedAt, Is.Null);
        }

        [Test]
        public async Task JoinCredential_ThatHasExpired_IsStillReturned()
        {
            var match = await NewWaitingMatchAsync();
            byte[] hash = Hash(2);

            await Store.StoreJoinCredentialAsync(hash, match.MatchId, match.Seat0, Created.AddMinutes(-5), Ct);

            JoinCredentialRecord? found = await Store.FindJoinCredentialAsync(hash, Ct);
            Assert.That(found, Is.Not.Null, "expiry is decided by the credential service, never by the store");
            Assert.That(found!.ExpiresAt, Is.EqualTo(Created.AddMinutes(-5)));
        }

        [Test]
        public async Task RevokeJoinCredentials_StampsOnlyThatPlayersLiveCredentials()
        {
            var match = await NewWaitingMatchAsync();
            await Store.StoreJoinCredentialAsync(Hash(3), match.MatchId, match.Seat0, Created.AddMinutes(15), Ct);
            await Store.StoreJoinCredentialAsync(Hash(9), match.MatchId, match.Seat1, Created.AddMinutes(15), Ct);

            await Store.RevokeJoinCredentialsAsync(match.MatchId, match.Seat0, Move1, Ct);
            await Store.RevokeJoinCredentialsAsync(match.MatchId, match.Seat0, Move2, Ct);

            Assert.That((await Store.FindJoinCredentialAsync(Hash(3), Ct))!.RevokedAt, Is.EqualTo(Move1),
                "a second revoke must not move the timestamp of an already revoked credential");
            Assert.That((await Store.FindJoinCredentialAsync(Hash(9), Ct))!.RevokedAt, Is.Null);
        }

        [Test]
        public async Task FindJoinCredential_ForAnUnknownHash_IsNull()
        {
            var match = await NewWaitingMatchAsync();
            await Store.StoreJoinCredentialAsync(Hash(4), match.MatchId, match.Seat0, Created.AddMinutes(15), Ct);

            Assert.That(await Store.FindJoinCredentialAsync(Hash(5), Ct), Is.Null);
        }

        [Test]
        public async Task JoinCredential_WithAHashThatIsNotThirtyTwoBytes_IsRejected()
        {
            var match = await NewWaitingMatchAsync();
            byte[] tooShort = new byte[31];

            Assert.ThrowsAsync<ArgumentException>(() => Store.StoreJoinCredentialAsync(
                tooShort, match.MatchId, match.Seat0, Created.AddMinutes(15), Ct));
            Assert.ThrowsAsync<ArgumentException>(() => Store.FindJoinCredentialAsync(tooShort, Ct));
        }
    }

    /// <summary>The contract against the real database. Every test starts from a freshly migrated schema so
    /// no test can inherit rows, or a stale schema, from the one before it.</summary>
    [TestFixture]
    public sealed class PostgresMatchStoreTests : MatchStoreContractTests
    {
        protected override async Task<IMatchStore> CreateStoreAsync()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();
            return new PostgresMatchStore(database.DataSource, NullLogger<PostgresMatchStore>.Instance);
        }
    }

    /// <summary>The same contract against the test double the coordinator tests will use.</summary>
    [TestFixture]
    public sealed class InMemoryMatchStoreTests : MatchStoreContractTests
    {
        protected override Task<IMatchStore> CreateStoreAsync() =>
            Task.FromResult<IMatchStore>(new InMemoryMatchStore());
    }
}
