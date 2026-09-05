using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The parts of the store contract that only mean anything when two connections are talking to the same
    /// database at once: a journal read that must not tear, a catalogue save that must not overtake a start,
    /// and a lobby allocation that must survive the match it collided with closing underneath it.
    ///
    /// These live apart from <see cref="MatchStoreContractTests"/> deliberately. They are not statements
    /// about the interface, they are statements about the isolation levels and row locks the Postgres
    /// implementation uses, and the in-memory double has nothing to say about either.
    /// </summary>
    [TestFixture]
    public sealed class PostgresMatchStoreConcurrencyTests
    {
        const string SetupWire = "annihilation 9 7 0 7 3 1 1 1 3 0";
        const string EngineVersion = "hexwars-engine/1";
        const string BuildId = "test-build";
        const int ProtocolVersion = 2;

        static readonly DateTimeOffset Created = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);
        static readonly DateTimeOffset Started = Created.AddMinutes(1);
        static readonly DateTimeOffset Move1 = Created.AddMinutes(2);

        /// <summary>Long enough that a blocked call is really blocked rather than merely slow, short enough
        /// that the suite does not crawl.</summary>
        static readonly TimeSpan BlockedFor = TimeSpan.FromSeconds(1);

        static int _identifiers;

        PostgresTestDatabase _db = null!;
        PostgresMatchStore _store = null!;

        static CancellationToken Ct => CancellationToken.None;

        [SetUp]
        public async Task StartFromAFreshSchema()
        {
            _db = await PostgresTestDatabase.GetAsync();
            await _db.ResetAsync();
            await _db.ApplyMigrationsAsync();
            _store = new PostgresMatchStore(_db.DataSource, NullLogger<PostgresMatchStore>.Instance);
        }

        // ---- 1. the journal is read from one snapshot ---------------------------

        [Test]
        public async Task LoadJournal_ReadsOneSnapshot_SoAStartAndAnAppendMidReadCannotTearIt()
        {
            var match = await NewWaitingMatchAsync();

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);

            // Fires once the loader has read the match row and before it reads the rest: exactly the window a
            // read-committed loader would have taken three separate snapshots in.
            _store.OnJournalSnapshotForTests = async () =>
            {
                Assert.That(await other.TryStartMatchAsync(match.MatchId, "START-REPLAY", Started, Ct), Is.True);
                Assert.That(
                    (await other.AppendCommandAsync(match.MatchId, 1, "E 0", match.Seat0, Move1, Ct)).Status,
                    Is.EqualTo(AppendStatus.Appended));
            };

            MatchJournal? journal = await _store.LoadJournalAsync(match.MatchId, Ct);

            Assert.That(journal, Is.Not.Null);
            Assert.That(journal!.Match.Status == MatchStatus.Waiting && journal.Commands.Count > 0, Is.False,
                "a waiting match carrying journalled commands is a torn read: the row and the commands came "
                + "from different snapshots");

            if (journal.Match.Status == MatchStatus.Waiting)
            {
                Assert.That(journal.Match.StartReplay, Is.Null);
                Assert.That(journal.Commands, Is.Empty);
            }
            else
            {
                Assert.That(journal.Match.Status, Is.EqualTo(MatchStatus.Active));
                Assert.That(journal.Match.StartReplay, Is.EqualTo("START-REPLAY"));
                Assert.That(journal.Commands, Has.Count.EqualTo(1));
            }
        }

        [Test]
        public async Task LoadJournal_ForAMatchThatDoesNotExist_IsNull()
        {
            Assert.That(await _store.LoadJournalAsync(Guid.NewGuid(), Ct), Is.Null);
        }

        // ---- 2. catalogue saves are serialised with the start -------------------

        [Test]
        public async Task SaveCatalog_QueuesBehindTheMatchRowLock_AndNoOpsOnceTheStartCommits()
        {
            var match = await NewWaitingMatchAsync();
            await _store.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-before-start", Ct);

            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();
            await using NpgsqlTransaction holding = await holder.BeginTransactionAsync();

            await using (var takeLock = new NpgsqlCommand(
                "SELECT status FROM matches WHERE match_id = @matchId FOR UPDATE", holder, holding))
            {
                takeLock.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await takeLock.ExecuteScalarAsync(), Is.EqualTo("waiting"));
            }

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);
            Task save = other.SaveCatalogAsync(match.MatchId, match.Seat0, "catalog-after-start", Ct);

            Task first = await Task.WhenAny(save, Task.Delay(BlockedFor));
            Assert.That(first, Is.Not.SameAs(save),
                "a catalogue save must take the match row lock, so it cannot run while a start holds it");

            await using (var start = new NpgsqlCommand(
                "UPDATE matches SET status = 'active', start_replay = 'START-REPLAY', "
                + "started_at = now(), last_activity_at = now() WHERE match_id = @matchId", holder, holding))
            {
                start.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await start.ExecuteNonQueryAsync(), Is.EqualTo(1));
            }

            await holding.CommitAsync();
            await save;

            PersistedPlayer? player = await _store.GetPlayerAsync(match.MatchId, match.Seat0, Ct);
            Assert.That(player!.CatalogWire, Is.EqualTo("catalog-before-start"),
                "the save lost the race and must have done nothing at all");
        }

        // ---- 3. allocation survives a close/create race -------------------------

        [Test]
        public async Task CreateMatch_RetriesWhenTheMatchItCollidedWithClosesBeforeItCanBeReadBack()
        {
            string lobby = NextLobbyId();
            CreateMatchResult first = await _store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            int conflicts = 0;
            _store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(first.Match.MatchId);
            };

            CreateMatchResult second = await _store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            Assert.That(conflicts, Is.EqualTo(1), "the first insert must have collided exactly once");
            Assert.That(second.Created, Is.True, "the lobby was free again, so the retry must have allocated");
            Assert.That(second.Match.MatchId, Is.Not.EqualTo(first.Match.MatchId));
            Assert.That(second.Match.Status, Is.EqualTo(MatchStatus.Waiting));
        }

        [Test]
        public async Task CreateMatch_KeepsAllocatingWhileTheLobbyChurns_AndSucceedsOnceItStops()
        {
            string lobby = NextLobbyId();
            Guid open = (await _store.CreateMatchForLobbyAsync(Request(lobby), Ct)).Match.MatchId;

            int conflicts = 0, retries = 0;

            _store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(open);
            };

            // Three rounds of somebody else winning the race and then closing their match, and then the
            // lobby is left alone. Three collisions is well inside what a busy lobby reaches by accident, so
            // allocation has to come back with a match rather than an exception.
            _store.OnCreateRetryForTests = async () =>
            {
                retries++;
                if (retries <= 2) open = await InsertWaitingMatchAsync(lobby);
            };

            CreateMatchResult allocated = await _store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            Assert.That(conflicts, Is.EqualTo(3), "three collisions before the lobby was left alone");
            Assert.That(allocated.Created, Is.True, "the fourth insert had a free lobby and must have taken it");
            Assert.That(allocated.Match.Status, Is.EqualTo(MatchStatus.Waiting));
        }

        [Test]
        public async Task CreateMatch_KeepsRetryingUntilTheLobbyIsFree()
        {
            var logs = new CapturingLoggerProvider();
            using ILoggerFactory factory = LoggerFactory.Create(builder => builder.AddProvider(logs));
            var store = new PostgresMatchStore(_db.DataSource, factory.CreateLogger<PostgresMatchStore>());

            string lobby = NextLobbyId();
            Guid open = (await store.CreateMatchForLobbyAsync(Request(lobby), Ct)).Match.MatchId;

            int conflicts = 0, retries = 0;

            // Close the match the insert collided with, so the re-read finds nothing...
            store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(open);
            };

            // ...then open another one before the retry, so the next insert collides again. Twelve times,
            // which is past any bound the old code had, and then the lobby is left alone. No number of
            // collisions makes allocation give up: the thirteenth insert has a free lobby and must take it.
            store.OnCreateRetryForTests = async () =>
            {
                retries++;
                if (retries <= 11) open = await InsertWaitingMatchAsync(lobby);
            };

            CreateMatchResult allocated = await store.CreateMatchForLobbyAsync(Request(lobby), Ct);

            Assert.That(conflicts, Is.EqualTo(12), "twelve collisions before the lobby was left alone");
            Assert.That(allocated.Created, Is.True, "the lobby was free, so the insert must have taken it");
            Assert.That(allocated.Match.Status, Is.EqualTo(MatchStatus.Waiting));

            // Churn that deep is worth saying out loud, once every eight collisions rather than every one.
            string[] churnWarnings = logs.Messages
                .Where(m => m.Contains("keeps colliding during match allocation", StringComparison.Ordinal))
                .ToArray();
            Assert.That(churnWarnings, Has.Length.EqualTo(1),
                "twelve collisions crosses the eight mark exactly once");
            Assert.That(churnWarnings[0], Does.Contain(lobby),
                "the warning has to name the lobby or nobody can find the client doing it");
        }

        [Test]
        public async Task CreateMatch_StopsWhenCancelled_WhileTheLobbyChurns()
        {
            string lobby = NextLobbyId();
            Guid open = (await _store.CreateMatchForLobbyAsync(Request(lobby), Ct)).Match.MatchId;

            using var cancelling = new CancellationTokenSource();
            int conflicts = 0;

            // A lobby that never stops churning, so nothing inside the store can end this loop. The token
            // is the only bound there is, and it has to be honoured or a caller that gave up is still
            // holding a connection and spinning against the database.
            _store.OnCreateConflictForTests = async () =>
            {
                conflicts++;
                await ExpireAsync(open);
            };
            _store.OnCreateRetryForTests = async () =>
            {
                open = await InsertWaitingMatchAsync(lobby);
                if (conflicts >= 5) await cancelling.CancelAsync();
            };

            CreateMatchRequest request = Request(lobby);

            Assert.CatchAsync<OperationCanceledException>(
                () => _store.CreateMatchForLobbyAsync(request, cancelling.Token));

            Assert.That(conflicts, Is.EqualTo(5), "the token stopped it, and nothing else could have");
            Assert.That(await SeatCountAsync(request.Players[0].SteamId), Is.EqualTo(0L),
                "a cancelled allocation must not leave a match behind");
        }

        // ---- an append cannot outlive the game ----------------------------------

        [Test]
        public async Task AppendCommand_QueuedBehindACompletion_IsRefusedOnceThatCompletionCommits()
        {
            var match = await NewActiveMatchAsync();

            // Hold the match row, so the append cannot get past its own SELECT ... FOR UPDATE. This is the
            // interleaving that matters and the one a plain concurrent test cannot force: the append has
            // read nothing yet when the game ends underneath it.
            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();
            await using NpgsqlTransaction holding = await holder.BeginTransactionAsync();

            await using (var takeLock = new NpgsqlCommand(
                "SELECT status FROM matches WHERE match_id = @matchId FOR UPDATE", holder, holding))
            {
                takeLock.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await takeLock.ExecuteScalarAsync(), Is.EqualTo("active"));
            }

            Task<AppendResult> append =
                _store.AppendCommandAsync(match.MatchId, 1, "E 0", match.Seat0, Move1, Ct);

            Task first = await Task.WhenAny(append, Task.Delay(BlockedFor));
            Assert.That(first, Is.Not.SameAs(append),
                "an append must take the match row lock, so it cannot run while a completion holds it");

            await using (var complete = new NpgsqlCommand(
                "UPDATE matches SET status = 'completed', completed_at = now(), winner_seat = 0, "
                + "last_activity_at = now() WHERE match_id = @matchId", holder, holding))
            {
                complete.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await complete.ExecuteNonQueryAsync(), Is.EqualTo(1));
            }

            await holding.CommitAsync();

            Assert.That(await append, Is.EqualTo(new AppendResult(AppendStatus.MatchNotActive, 1)),
                "the append woke up in a finished game and must say so, not report an ordering problem");
            Assert.That((await _store.LoadJournalAsync(match.MatchId, Ct))!.Commands, Is.Empty,
                "a command must not land in a game that ended while it was waiting for the row");
        }

        // ---- one connection is enough ------------------------------------------

        [Test]
        public async Task StoreJoinCredential_RetriedOnAPoolOfOne_ReadsBackOnTheConnectionItAlreadyHas()
        {
            // A pool of exactly one is the smallest honest model of a pool under load: if the retry path
            // wants a second connection while holding the first, it can only wait for itself. On a real
            // deployment that is not a slow call, it is every caller on the credential path deadlocked.
            await using NpgsqlDataSource single = new NpgsqlDataSourceBuilder(
                _db.ConnectionString + ";Maximum Pool Size=1;Timeout=5").Build();
            var store = new PostgresMatchStore(single, NullLogger<PostgresMatchStore>.Instance);

            CreateMatchRequest request = Request(NextLobbyId());
            CreateMatchResult match = await store.CreateMatchForLobbyAsync(request, Ct);
            byte[] hash = Hash(11);

            await store.StoreJoinCredentialAsync(
                hash, match.Match.MatchId, request.Players[0].SteamId, Created.AddMinutes(15), Ct);

            // The same issue arriving twice, which is what a credential service does after a dropped reply.
            Assert.DoesNotThrowAsync(() => store.StoreJoinCredentialAsync(
                hash, match.Match.MatchId, request.Players[0].SteamId, Created.AddMinutes(30), Ct));

            JoinCredentialRecord stored = (await store.FindJoinCredentialAsync(hash, Ct))!;
            Assert.That(stored.SteamId, Is.EqualTo(request.Players[0].SteamId));
            Assert.That(stored.ExpiresAt, Is.EqualTo(Created.AddMinutes(15)),
                "a retry keeps the credential that was already issued, expiry and all");
        }

        [Test]
        public async Task StoreJoinCredential_ClashingWithAnotherSeatOnAPoolOfOne_IsStillRefusedPromptly()
        {
            await using NpgsqlDataSource single = new NpgsqlDataSourceBuilder(
                _db.ConnectionString + ";Maximum Pool Size=1;Timeout=5").Build();
            var store = new PostgresMatchStore(single, NullLogger<PostgresMatchStore>.Instance);

            CreateMatchRequest request = Request(NextLobbyId());
            CreateMatchResult match = await store.CreateMatchForLobbyAsync(request, Ct);
            byte[] hash = Hash(23);

            await store.StoreJoinCredentialAsync(
                hash, match.Match.MatchId, request.Players[0].SteamId, Created.AddMinutes(15), Ct);

            // The refusal comes from the same read-back, so it must not need a second connection either.
            Assert.ThrowsAsync<InvalidOperationException>(() => store.StoreJoinCredentialAsync(
                hash, match.Match.MatchId, request.Players[1].SteamId, Created.AddMinutes(15), Ct));
        }

        // ---- helpers ------------------------------------------------------------

        // ---- credential rotation is one transaction -----------------------------

        [Test]
        public async Task ASecondIssueForOneSeat_QueuesBehindTheFirst_LeavingExactlyOneLiveCredential()
        {
            var match = await NewWaitingMatchAsync();
            MatchCredentialService service = NewCredentialService(_store);
            IssuedCredential firstCredential = await service.IssueAsync(match.MatchId, match.Seat0, Ct);

            // The seat row rather than the match row: two reconnects for ONE seat are what must serialise,
            // and holding this is the only way to make them overlap on purpose rather than by luck.
            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();
            await using NpgsqlTransaction holding = await holder.BeginTransactionAsync();

            await using (var takeLock = new NpgsqlCommand(
                "SELECT seat FROM match_players WHERE match_id = @matchId AND steam_id = @steamId FOR UPDATE",
                holder, holding))
            {
                takeLock.Parameters.AddWithValue("matchId", match.MatchId);
                takeLock.Parameters.AddWithValue("steamId", match.Seat0);
                Assert.That(await takeLock.ExecuteScalarAsync(), Is.EqualTo(0));
            }

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);
            Task<IssuedCredential> second = NewCredentialService(other).IssueAsync(match.MatchId, match.Seat0, Ct);

            Task first = await Task.WhenAny(second, Task.Delay(BlockedFor));
            Assert.That(first, Is.Not.SameAs(second),
                "issuing must lock the seat, or two overlapping reconnects both revoke first and both then insert");

            // The discriminating assertion. Revoke-then-insert as two calls revokes on its own connection
            // and commits that immediately, so the player loses the credential they are holding and only
            // then blocks on the insert - and if the insert never lands they have none at all. One
            // transaction means nothing at all is visible until the whole replacement can commit.
            Assert.That(await service.ValidateAsync(match.MatchId, firstCredential.Credential, Ct), Is.Not.Null,
                "nothing may be revoked while the replacement is still waiting to commit");

            await holding.RollbackAsync();
            IssuedCredential secondCredential = await second;

            Assert.That(await LiveCredentialCountAsync(match.MatchId, match.Seat0), Is.EqualTo(1));
            Assert.That(await service.ValidateAsync(match.MatchId, firstCredential.Credential, Ct), Is.Null);
            Assert.That(await service.ValidateAsync(match.MatchId, secondCredential.Credential, Ct), Is.Not.Null);
        }

        [Test]
        public async Task AnIssueOverlappingTheMatchEnding_WaitsForItAndThenRefuses()
        {
            var match = await NewActiveMatchAsync();
            MatchCredentialService service = NewCredentialService(_store);
            IssuedCredential held = await service.IssueAsync(match.MatchId, match.Seat0, Ct);

            await using NpgsqlConnection holder = await _db.DataSource.OpenConnectionAsync();
            await using NpgsqlTransaction holding = await holder.BeginTransactionAsync();

            await using (var takeLock = new NpgsqlCommand(
                "SELECT status FROM matches WHERE match_id = @matchId FOR UPDATE", holder, holding))
            {
                takeLock.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await takeLock.ExecuteScalarAsync(), Is.EqualTo("active"));
            }

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);
            // The window is closed for this issue on purpose. What is being proved is that issuing takes
            // the match row lock and therefore sees a completion that is still in flight; leaving the
            // window open would have it granted for a different and correct reason, and prove nothing
            // about the lock.
            Task<IssuedCredential> issue = NewCredentialService(other, terminalWindowSeconds: 0)
                .IssueAsync(match.MatchId, match.Seat0, Ct);

            Task first = await Task.WhenAny(issue, Task.Delay(BlockedFor));
            Assert.That(first, Is.Not.SameAs(issue),
                "issuing must take the match row lock, or it cannot see a completion that is still in flight");

            await using (var complete = new NpgsqlCommand(
                "UPDATE matches SET status = \u0027completed\u0027, winner_seat = 0, completed_at = now(), "
                + "last_activity_at = now() WHERE match_id = @matchId", holder, holding))
            {
                complete.Parameters.AddWithValue("matchId", match.MatchId);
                Assert.That(await complete.ExecuteNonQueryAsync(), Is.EqualTo(1));
            }

            await holding.CommitAsync();

            var refused = Assert.ThrowsAsync<InvalidOperationException>(() => issue);
            Assert.That(refused!.Message, Is.EqualTo(MatchCredentialService.MatchNotOpenMessage));
            Assert.That(await LiveCredentialCountAsync(match.MatchId, match.Seat0), Is.EqualTo(1),
                "a refused issue must leave the credential the player already held exactly as it was");
            // The ending does not revoke it. A seat that missed the final APPLY still needs a way to learn
            // how the game finished, so the credential opens the match for the terminal reconnect window
            // and for nothing beyond it.
            Assert.That(await service.ValidateAsync(match.MatchId, held.Credential, Ct), Is.Not.Null,
                "inside the terminal window a finished match is still reachable");

            MatchCredentialService windowClosed = new(
                _store,
                Options.Create(new MatchHostingOptions
                {
                    JoinTokenTtlSeconds = 900,
                    TerminalReconnectSeconds = 0,
                }),
                new FakeTimeProvider(Created),
                NullLogger<MatchCredentialService>.Instance);

            Assert.That(await windowClosed.ValidateAsync(match.MatchId, held.Credential, Ct), Is.Null,
                "and with the window closed it stops working the moment the match is over");
        }

        [Test]
        public async Task TwoIssuesForOneSeatSubmittedTogether_LeaveExactlyOneUsableCredential()
        {
            // No held lock and no hook: two services on two pools, submitted together and left to
            // interleave however the database schedules them. It cannot prove the mechanism the way the
            // blocked test above does, but it is the shape the deployment actually produces - one player
            // reconnecting through two instances at the same moment - and the outcome must hold whatever
            // order they land in.
            var match = await NewWaitingMatchAsync();

            await using NpgsqlDataSource otherPool = NpgsqlDataSource.Create(_db.ConnectionString);
            var other = new PostgresMatchStore(otherPool, NullLogger<PostgresMatchStore>.Instance);

            MatchCredentialService first = NewCredentialService(_store);
            MatchCredentialService second = NewCredentialService(other);

            IssuedCredential[] issued = await Task.WhenAll(
                first.IssueAsync(match.MatchId, match.Seat0, Ct),
                second.IssueAsync(match.MatchId, match.Seat0, Ct));

            CredentialValidation?[] validations = await Task.WhenAll(
                first.ValidateAsync(match.MatchId, issued[0].Credential, Ct),
                first.ValidateAsync(match.MatchId, issued[1].Credential, Ct));

            Assert.That(validations.Count(validation => validation is not null), Is.EqualTo(1),
                "one seat may have one usable credential, whichever of the two issues committed last");
            Assert.That(await LiveCredentialCountAsync(match.MatchId, match.Seat0), Is.EqualTo(1));
        }

        [Test]
        public async Task AFailureAfterTheRevoke_RollsBackAndLeavesTheOldCredentialUsable()
        {
            // The window that made the two-call version dangerous: the old credential is already revoked
            // and the new one has not landed. One transaction means the player keeps what they had rather
            // than being left holding nothing.
            var match = await NewWaitingMatchAsync();
            MatchCredentialService service = NewCredentialService(_store);
            IssuedCredential held = await service.IssueAsync(match.MatchId, match.Seat0, Ct);

            _store.AfterRevokeForTests = () => throw new TimeoutException("the connection went away");

            Assert.ThrowsAsync<TimeoutException>(() => service.IssueAsync(match.MatchId, match.Seat0, Ct));

            _store.AfterRevokeForTests = null;

            Assert.That(await LiveCredentialCountAsync(match.MatchId, match.Seat0), Is.EqualTo(1));
            Assert.That(await service.ValidateAsync(match.MatchId, held.Credential, Ct), Is.Not.Null,
                "a failed reissue must not cost the player the credential they were already holding");
        }

        MatchCredentialService NewCredentialService(
            IMatchStore store,
            int terminalWindowSeconds = MatchHostingOptions.DefaultTerminalReconnectSeconds) => new(
            store,
            Options.Create(new MatchHostingOptions
            {
                JoinTokenTtlSeconds = 900,
                TerminalReconnectSeconds = terminalWindowSeconds,
            }),
            new FakeTimeProvider(Created),
            NullLogger<MatchCredentialService>.Instance);

        [Test]
        public async Task AMatchThatFinishesBeforeTheLock_CapsTheCredentialAtTheWindow()
        {
            // The caller decides on a full TTL, and the match ends before this transaction can take the
            // match row lock. Only the value computed under that lock knows the match is over, so only it
            // can be capped - a caller that read the status first read it before it changed.
            var match = await NewActiveMatchAsync();
            TimeSpan window = TimeSpan.FromMinutes(10);

            _store.BeforeLockForTests = async () =>
            {
                _store.BeforeLockForTests = null;
                Assert.That(
                    await _store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, 0, Move1, Ct),
                    Is.True);
            };

            CredentialReplacement replacement = await _store.ReplaceJoinCredentialAsync(
                Hash(80), match.MatchId, match.Seat0, Move1.AddMinutes(15), Move1, Ct, window);

            Assert.That(replacement.Replaced, Is.True);
            Assert.That(replacement.EffectiveExpiresAt, Is.EqualTo(Move1 + window),
                "the returned expiry is the one the row was locked against");

            JoinCredentialRecord stored = (await _store.FindJoinCredentialAsync(Hash(80), Ct))!;
            Assert.That(stored.ExpiresAt, Is.EqualTo(Move1 + window),
                "and it is what was actually written, not merely what was reported");
        }

        [Test]
        public async Task AnIssueForAMatchThatHasFinished_IsRefusedAndWritesNothing()
        {
            var match = await NewActiveMatchAsync();
            Assert.That(
                await _store.TryCompleteMatchAsync(match.MatchId, MatchStatus.Completed, 0, Move1, Ct), Is.True);

            // The window closed, so the only answer left is a refusal. Inside it a finished match is
            // joinable on purpose - that is what lets a seat come back and be shown how the game ended.
            var service = new MatchCredentialService(
                _store,
                Options.Create(new MatchHostingOptions
                {
                    JoinTokenTtlSeconds = 900,
                    TerminalReconnectSeconds = 0,
                }),
                new FakeTimeProvider(Created),
                NullLogger<MatchCredentialService>.Instance);

            var refused = Assert.ThrowsAsync<InvalidOperationException>(
                () => service.IssueAsync(match.MatchId, match.Seat0, Ct));

            Assert.That(refused!.Message, Is.EqualTo(MatchCredentialService.MatchNotOpenMessage));
            Assert.That(await LiveCredentialCountAsync(match.MatchId, match.Seat0), Is.Zero);
        }

        async Task<long> LiveCredentialCountAsync(Guid matchId, string steamId)
        {
            await using var command = _db.DataSource.CreateCommand(
                "SELECT count(*) FROM match_join_credentials "
                + "WHERE match_id = @matchId AND steam_id = @steamId AND revoked_at IS NULL");
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("steamId", steamId);
            return (long)(await command.ExecuteScalarAsync())!;
        }

        static string NextLobbyId() =>
            "1097752" + Interlocked.Increment(ref _identifiers).ToString("D11");

        static string NextSteamId() =>
            "7656119" + Interlocked.Increment(ref _identifiers).ToString("D10");

        static CreateMatchRequest Request(string lobbyId) =>
            new(lobbyId, SetupWire, EngineVersion, ProtocolVersion, BuildId,
                new[] { (NextSteamId(), 0), (NextSteamId(), 1) }, Created);

        static byte[] Hash(byte seed)
        {
            var hash = new byte[32];
            for (int i = 0; i < hash.Length; i++) hash[i] = (byte)(seed + i);
            return hash;
        }

        async Task<(Guid MatchId, string Lobby, string Seat0, string Seat1)> NewWaitingMatchAsync()
        {
            CreateMatchRequest request = Request(NextLobbyId());
            CreateMatchResult created = await _store.CreateMatchForLobbyAsync(request, Ct);
            return (created.Match.MatchId, request.SteamLobbyId, request.Players[0].SteamId,
                request.Players[1].SteamId);
        }

        async Task<(Guid MatchId, string Lobby, string Seat0, string Seat1)> NewActiveMatchAsync()
        {
            var match = await NewWaitingMatchAsync();
            bool started = await _store.TryStartMatchAsync(match.MatchId, "START-REPLAY", Started, Ct);
            Assert.That(started, Is.True, "the fixture could not start the match it just created");
            return match;
        }

        async Task ExpireAsync(Guid matchId)
        {
            await using var command = _db.DataSource.CreateCommand(
                "UPDATE matches SET status = 'expired', completed_at = now(), last_activity_at = now() "
                + "WHERE match_id = @matchId");
            command.Parameters.AddWithValue("matchId", matchId);
            await command.ExecuteNonQueryAsync();
        }

        async Task<long> SeatCountAsync(string steamId)
        {
            await using var command = _db.DataSource.CreateCommand(
                "SELECT count(*) FROM match_players WHERE steam_id = @steamId");
            command.Parameters.AddWithValue("steamId", steamId);
            return (long)(await command.ExecuteScalarAsync())!;
        }

        async Task<Guid> InsertWaitingMatchAsync(string lobbyId)
        {
            var matchId = Guid.NewGuid();
            await using var command = _db.DataSource.CreateCommand(
                "INSERT INTO matches (match_id, steam_lobby_id, status, setup_wire, engine_version, "
                + "protocol_version, build_id, created_at, last_activity_at) "
                + "VALUES (@matchId, @lobbyId, 'waiting', @setup, @engine, @protocol, @build, now(), now())");
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("lobbyId", lobbyId);
            command.Parameters.AddWithValue("setup", SetupWire);
            command.Parameters.AddWithValue("engine", EngineVersion);
            command.Parameters.AddWithValue("protocol", ProtocolVersion);
            command.Parameters.AddWithValue("build", BuildId);
            await command.ExecuteNonQueryAsync();
            return matchId;
        }
    }
}
