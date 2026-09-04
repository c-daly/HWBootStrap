using Npgsql;
using NpgsqlTypes;

namespace HexWars.NetServer.Persistence
{
    /// <summary>
    /// The match journal on Postgres.
    ///
    /// Every guarantee this class makes is enforced by the database rather than by C#, because two instances
    /// of the server can be doing this at the same time. Allocation is idempotent because of the partial
    /// unique index on open lobbies; append ordering holds because the INSERT carries its own
    /// "only if this is the next sequence" predicate and the row is locked first; status transitions are safe
    /// because the allowed edges live in the WHERE clause of a single UPDATE, not in a read-then-write.
    ///
    /// Timestamps go in as UTC timestamptz and come back as UTC, so nothing here depends on the timezone of
    /// the server or of the database session.
    ///
    /// last_activity_at and last_seen_at only ever move forward. Every write that bumps them takes the
    /// GREATEST of the stored value and the new one, because these are the columns the retention sweeper
    /// reads: two instances writing the same match, or one client retrying a dropped heartbeat, can land an
    /// older timestamp last, and a match dragged far enough into the past gets abandoned while it is still
    /// being played.
    /// </summary>
    public sealed class PostgresMatchStore(NpgsqlDataSource dataSource, ILogger<PostgresMatchStore> logger) : IMatchStore
    {
        /// <summary>The partial unique index from 001_match_journal. A unique violation naming it is not an
        /// error, it is the answer "someone else allocated this lobby first".</summary>
        public const string OpenLobbyIndex = "ux_matches_open_lobby";

        /// <summary>How many consecutive collisions pass before allocation says so out loud. It is a
        /// reporting interval, not a limit: the loop below has no limit, because any fixed one would throw
        /// on the attempt after a lobby had already gone quiet.</summary>
        public const int CollisionsPerWarning = 8;

        const string MatchColumns =
            "match_id, steam_lobby_id, status, setup_wire, start_replay, engine_version, protocol_version, "
            + "build_id, created_at, started_at, completed_at, last_activity_at, winner_seat";

        const string PlayerColumns = "match_id, steam_id, seat, catalog_wire, joined_at, last_seen_at";

        const string CommandColumns = "match_id, sequence, command_wire, accepted_at, issuer_steam_id";

        /// <summary>The two statuses that mean a match can still be joined or played, written as a SQL
        /// literal list so the only place these words are spelled is "MatchStatusText".</summary>
        static readonly string OpenStatuses =
            "('" + MatchStatusText.Waiting + "', '" + MatchStatusText.Active + "')";

        /// <summary>Test hook: runs once the journal read has taken its snapshot and before the rest
        /// of the journal is read, so a test can commit a start and an append into that window.</summary>
        internal Func<Task>? OnJournalSnapshotForTests { get; set; }

        /// <summary>Test hook: runs after the open-lobby index rejected an insert and before the open
        /// match is read back, so a test can close that match in between.</summary>
        internal Func<Task>? OnCreateConflictForTests { get; set; }

        /// <summary>Test hook: runs after the re-read found nothing and before the insert is retried,
        /// so a test can put a new open match in the way.</summary>
        internal Func<Task>? OnCreateRetryForTests { get; set; }

        // ---- allocation ------------------------------------------------------

        public async Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(request);
            MatchStoreGuard.ValidatePlayers(request.Players);

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);

            // A collision on the open-lobby index normally means somebody else allocated first, and the
            // answer is their match. But the match it collided with can reach a terminal status before we
            // read it back, and then there is no match to return and no reason not to allocate after all.
            // That is a legal sequence of events, not a failure, so it is retried rather than thrown.
            //
            // There is no attempt limit, deliberately. Every iteration ends in one of two answers that are
            // both correct - an insert that committed, or an open match that was really there - so a limit
            // only ever adds a third answer that is wrong: it throws on the attempt after the lobby went
            // quiet, when the very next insert would have succeeded. The token is what stops this loop,
            // which puts the bound where it belongs, with the caller.
            int collisions = 0;

            while (true)
            {
                ct.ThrowIfCancellationRequested();

                await using NpgsqlTransaction transaction =
                    await connection.BeginTransactionAsync(ct).ConfigureAwait(false);

                try
                {
                    PersistedMatch created =
                        await InsertMatchAsync(connection, transaction, request, ct).ConfigureAwait(false);
                    await transaction.CommitAsync(ct).ConfigureAwait(false);
                    return new CreateMatchResult(created, true);
                }
                catch (PostgresException ex)
                    when (ex.SqlState == PostgresErrorCodes.UniqueViolation && ex.ConstraintName == OpenLobbyIndex)
                {
                    await transaction.RollbackAsync(ct).ConfigureAwait(false);

                    if (OnCreateConflictForTests is not null) await OnCreateConflictForTests().ConfigureAwait(false);

                    PersistedMatch? existing =
                        await FindOpenMatchAsync(connection, null, request.SteamLobbyId, ct).ConfigureAwait(false);

                    if (existing is not null)
                    {
                        logger.LogInformation(
                            "Lobby already had match {MatchId} allocated; returning it instead of creating another.",
                            Short(existing.MatchId));
                        return new CreateMatchResult(existing, false);
                    }

                    collisions++;

                    if (collisions % CollisionsPerWarning == 0)
                        // Not a failure, but not normal either: a lobby being opened and closed faster than
                        // it can be read is worth an operator seeing, and naming it is the only way to find
                        // out which client is doing it.
                        logger.LogWarning(
                            "Steam lobby {SteamLobbyId} keeps colliding during match allocation: {Collisions} "
                            + "inserts in a row have hit a match that closed again before it could be read "
                            + "back. Still retrying.", request.SteamLobbyId, collisions);
                    else
                        logger.LogInformation(
                            "Lobby allocation collided with a match that had closed by the time it was read "
                            + "back; retrying (collision {Collisions}).", collisions);

                    if (OnCreateRetryForTests is not null) await OnCreateRetryForTests().ConfigureAwait(false);
                }
            }
        }

        static async Task<PersistedMatch> InsertMatchAsync(NpgsqlConnection connection,
            NpgsqlTransaction transaction, CreateMatchRequest request, CancellationToken ct)
        {
            PersistedMatch created;
            await using (var insert = new NpgsqlCommand(
                "INSERT INTO matches (" + MatchColumns + ") VALUES "
                + "(@matchId, @lobbyId, @status, @setupWire, NULL, @engineVersion, @protocolVersion, "
                + "@buildId, @createdAt, NULL, NULL, @createdAt, NULL) RETURNING " + MatchColumns,
                connection, transaction))
            {
                insert.Parameters.AddWithValue("matchId", Guid.NewGuid());
                insert.Parameters.AddWithValue("lobbyId", request.SteamLobbyId);
                insert.Parameters.AddWithValue("status", MatchStatusText.ToDb(MatchStatus.Waiting));
                insert.Parameters.AddWithValue("setupWire", request.SetupWire);
                insert.Parameters.AddWithValue("engineVersion", request.EngineVersion);
                insert.Parameters.AddWithValue("protocolVersion", request.ProtocolVersion);
                insert.Parameters.AddWithValue("buildId", request.BuildId);
                insert.Parameters.Add(Timestamp("createdAt", request.CreatedAt));

                await using NpgsqlDataReader reader = await insert.ExecuteReaderAsync(ct).ConfigureAwait(false);
                await reader.ReadAsync(ct).ConfigureAwait(false);
                created = ReadMatch(reader);
            }

            foreach ((string steamId, int seat) in request.Players)
            {
                await using var insertPlayer = new NpgsqlCommand(
                    "INSERT INTO match_players (" + PlayerColumns + ") "
                    + "VALUES (@matchId, @steamId, @seat, NULL, @joinedAt, NULL)", connection, transaction);
                insertPlayer.Parameters.AddWithValue("matchId", created.MatchId);
                insertPlayer.Parameters.AddWithValue("steamId", steamId);
                insertPlayer.Parameters.AddWithValue("seat", seat);
                insertPlayer.Parameters.Add(Timestamp("joinedAt", created.CreatedAt));
                await insertPlayer.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }

            return created;
        }

        // ---- reads -----------------------------------------------------------

        public async Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            return await ReadMatchAsync(connection, null, matchId, ct).ConfigureAwait(false);
        }

        public async Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            return await FindOpenMatchAsync(connection, null, steamLobbyId, ct).ConfigureAwait(false);
        }

        public async Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            return await ReadPlayersAsync(connection, null, matchId, ct).ConfigureAwait(false);
        }

        public async Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using var command = new NpgsqlCommand(
                "SELECT " + PlayerColumns + " FROM match_players WHERE match_id = @matchId AND steam_id = @steamId",
                connection);
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("steamId", steamId);

            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            return await reader.ReadAsync(ct).ConfigureAwait(false) ? ReadPlayer(reader) : null;
        }

        public async Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);

            // One snapshot for all three reads. Under read committed each query sees whatever is committed
            // when it starts, so a start or an append landing between them would hand the caller a journal
            // that never existed: a waiting match with commands in it, or an active one with none. Repeatable
            // read takes the snapshot at the first statement and keeps it for the rest of the transaction.
            await using NpgsqlTransaction transaction = await connection
                .BeginTransactionAsync(System.Data.IsolationLevel.RepeatableRead, ct).ConfigureAwait(false);

            PersistedMatch? match =
                await ReadMatchAsync(connection, transaction, matchId, ct).ConfigureAwait(false);

            if (OnJournalSnapshotForTests is not null) await OnJournalSnapshotForTests().ConfigureAwait(false);

            if (match is null)
            {
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                return null;
            }

            IReadOnlyList<PersistedPlayer> players =
                await ReadPlayersAsync(connection, transaction, matchId, ct).ConfigureAwait(false);
            IReadOnlyList<PersistedCommand> commands =
                await ReadCommandsAsync(connection, transaction, matchId, ct).ConfigureAwait(false);

            await transaction.CommitAsync(ct).ConfigureAwait(false);
            return new MatchJournal(match, players, commands);
        }

        public async Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using var command = new NpgsqlCommand(
                "SELECT match_id FROM matches WHERE status IN " + OpenStatuses + " ORDER BY created_at", connection);

            var ids = new List<Guid>();
            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            while (await reader.ReadAsync(ct).ConfigureAwait(false)) ids.Add(reader.GetGuid(0));
            return ids;
        }

        // ---- match lifecycle -------------------------------------------------

        public async Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct)
        {
            MatchStoreGuard.ValidateSteamId(steamId, nameof(steamId));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using NpgsqlTransaction transaction =
                await connection.BeginTransactionAsync(ct).ConfigureAwait(false);

            // FOR UPDATE on the match row is what serialises this against TryStartMatchAsync, whose UPDATE
            // wants the same lock. A save therefore either lands before the start and is picked up by the
            // read that builds the start replay, or it arrives after it and does nothing at all. An EXISTS
            // guard would not do: the start could commit between evaluating it and writing the row.
            string? status = await ReadStatusForUpdateAsync(connection, transaction, matchId, ct).ConfigureAwait(false);
            if (status is null || MatchStatusText.FromDb(status) != MatchStatus.Waiting)
            {
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                return;
            }

            int updated;
            await using (var save = new NpgsqlCommand(
                "UPDATE match_players SET catalog_wire = @catalogWire "
                + "WHERE match_id = @matchId AND steam_id = @steamId", connection, transaction))
            {
                save.Parameters.AddWithValue("catalogWire", catalogWire);
                save.Parameters.AddWithValue("matchId", matchId);
                save.Parameters.AddWithValue("steamId", steamId);
                updated = await save.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }

            if (updated == 0)
            {
                // Nobody by that Steam id holds a seat here. Bumping last_activity_at anyway would let a
                // stranger keep a dead lobby out of the reaper.
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                return;
            }

            // No caller timestamp reaches this method, so the database clock decides. It only has to keep the
            // reaper away from a lobby whose players are still choosing.
            await using (var touch = new NpgsqlCommand(
                "UPDATE matches SET last_activity_at = GREATEST(last_activity_at, now()) "
                + "WHERE match_id = @matchId", connection, transaction))
            {
                touch.Parameters.AddWithValue("matchId", matchId);
                await touch.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }

            await transaction.CommitAsync(ct).ConfigureAwait(false);
        }

        public async Task<bool> TryStartMatchAsync(Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct)
        {
            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using var command = new NpgsqlCommand(
                "UPDATE matches SET status = @active, start_replay = @startReplay, started_at = @startedAt, "
                + "last_activity_at = GREATEST(last_activity_at, @startedAt) "
                + "WHERE match_id = @matchId AND status = @waiting", connection);
            command.Parameters.AddWithValue("active", MatchStatusText.Active);
            command.Parameters.AddWithValue("startReplay", startReplay);
            command.Parameters.Add(Timestamp("startedAt", startedAt));
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("waiting", MatchStatusText.Waiting);

            return await command.ExecuteNonQueryAsync(ct).ConfigureAwait(false) == 1;
        }

        public async Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat, DateTimeOffset completedAt, CancellationToken ct)
        {
            MatchStoreGuard.ValidateWinnerSeat(winnerSeat, nameof(winnerSeat));

            // Only a completed match is scored. Expired and abandoned games ended without anyone winning, and
            // the schema will not hold a winner on one, so this is a refusal rather than a failed write.
            if (winnerSeat is not null && terminal != MatchStatus.Completed) return false;

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);

            // The allowed edges are the WHERE clause. An UPDATE that matches no row is the refusal, so there
            // is no window between deciding a transition is legal and performing it.
            await using var command = new NpgsqlCommand(
                "UPDATE matches SET status = @terminal, completed_at = @completedAt, winner_seat = @winnerSeat, "
                + "last_activity_at = GREATEST(last_activity_at, @completedAt) WHERE match_id = @matchId AND ("
                + "  (status = @active AND @terminal IN (@completed, @expired, @abandoned))"
                + "  OR (status = @waiting AND @terminal IN (@expired, @abandoned)))", connection);
            command.Parameters.AddWithValue("terminal", MatchStatusText.ToDb(terminal));
            command.Parameters.Add(Timestamp("completedAt", completedAt));
            command.Parameters.AddWithValue("winnerSeat", (object?)winnerSeat ?? DBNull.Value);
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("active", MatchStatusText.Active);
            command.Parameters.AddWithValue("waiting", MatchStatusText.Waiting);
            command.Parameters.AddWithValue("completed", MatchStatusText.Completed);
            command.Parameters.AddWithValue("expired", MatchStatusText.Expired);
            command.Parameters.AddWithValue("abandoned", MatchStatusText.Abandoned);

            return await command.ExecuteNonQueryAsync(ct).ConfigureAwait(false) == 1;
        }

        public async Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct)
        {
            if (steamId is not null) MatchStoreGuard.ValidateSteamId(steamId, nameof(steamId));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using var batch = new NpgsqlBatch(connection);

            var match = new NpgsqlBatchCommand(
                "UPDATE matches SET last_activity_at = GREATEST(last_activity_at, @seenAt) "
                + "WHERE match_id = @matchId");
            match.Parameters.Add(Timestamp("seenAt", seenAt));
            match.Parameters.AddWithValue("matchId", matchId);
            batch.BatchCommands.Add(match);

            if (steamId is not null)
            {
                var player = new NpgsqlBatchCommand(
                    "UPDATE match_players SET last_seen_at = GREATEST(COALESCE(last_seen_at, @seenAt), @seenAt) "
                    + "WHERE match_id = @matchId AND steam_id = @steamId");
                player.Parameters.Add(Timestamp("seenAt", seenAt));
                player.Parameters.AddWithValue("matchId", matchId);
                player.Parameters.AddWithValue("steamId", steamId);
                batch.BatchCommands.Add(player);
            }

            await batch.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
        }

        // ---- the command journal ---------------------------------------------

        public async Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence, string commandWire, string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct)
        {
            MatchStoreGuard.ValidateSteamId(issuerSteamId, nameof(issuerSteamId));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using NpgsqlTransaction transaction = await connection.BeginTransactionAsync(ct).ConfigureAwait(false);

            // FOR UPDATE serialises appends for this match against every other instance. Without it the
            // status check below and the INSERT that follows are two separate decisions, and a match that
            // completes in between would accept one more command.
            (string? status, bool issuerHoldsSeat) =
                await ReadStatusAndSeatForUpdateAsync(connection, transaction, matchId, issuerSteamId, ct)
                    .ConfigureAwait(false);

            if (status is null || MatchStatusText.FromDb(status) != MatchStatus.Active)
            {
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                return new AppendResult(AppendStatus.MatchNotActive, expectedSequence);
            }

            // Checked here rather than left to the foreign key: the INSERT below writes nothing at all when
            // the sequence is stale, and a seatless issuer would then be answered with an ordering result it
            // could act on instead of the refusal it deserves.
            if (!issuerHoldsSeat)
            {
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                throw new ArgumentException(MatchStoreGuard.NoSeatMessage, nameof(issuerSteamId));
            }

            int inserted;
            try
            {
                await using var insert = new NpgsqlCommand(
                    "INSERT INTO match_commands (" + CommandColumns + ") "
                    + "SELECT @matchId, @sequence, @commandWire, @acceptedAt, @issuerSteamId "
                    + "WHERE (SELECT COALESCE(MAX(sequence), 0) + 1 FROM match_commands WHERE match_id = @matchId) "
                    + "= @sequence", connection, transaction);
                insert.Parameters.AddWithValue("matchId", matchId);
                insert.Parameters.AddWithValue("sequence", expectedSequence);
                insert.Parameters.AddWithValue("commandWire", commandWire);
                insert.Parameters.Add(Timestamp("acceptedAt", acceptedAt));
                insert.Parameters.AddWithValue("issuerSteamId", issuerSteamId);
                inserted = await insert.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }
            catch (PostgresException ex) when (ex.SqlState == PostgresErrorCodes.ForeignKeyViolation)
            {
                // The issuer holds no seat in this match. Nobody could have submitted this command, so it is
                // a caller mistake rather than an ordering answer the caller could act on.
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                throw new ArgumentException(MatchStoreGuard.NoSeatMessage, nameof(issuerSteamId), ex);
            }
            catch (PostgresException ex) when (ex.SqlState == PostgresErrorCodes.UniqueViolation)
            {
                // Belt and braces: the row lock should have prevented this, but if the sequence is taken the
                // answer is exactly the same one the no-rows path computes.
                await transaction.RollbackAsync(ct).ConfigureAwait(false);
                return await ClassifyAppendAsync(
                    connection, null, matchId, expectedSequence, commandWire, issuerSteamId, ct).ConfigureAwait(false);
            }

            if (inserted == 1)
            {
                await using (var touch = new NpgsqlCommand(
                    "UPDATE matches SET last_activity_at = GREATEST(last_activity_at, @acceptedAt) "
                    + "WHERE match_id = @matchId", connection, transaction))
                {
                    touch.Parameters.Add(Timestamp("acceptedAt", acceptedAt));
                    touch.Parameters.AddWithValue("matchId", matchId);
                    await touch.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
                }

                await transaction.CommitAsync(ct).ConfigureAwait(false);
                return new AppendResult(AppendStatus.Appended, expectedSequence);
            }

            AppendResult classified = await ClassifyAppendAsync(
                connection, transaction, matchId, expectedSequence, commandWire, issuerSteamId, ct).ConfigureAwait(false);
            await transaction.RollbackAsync(ct).ConfigureAwait(false);

            if (classified.Status == AppendStatus.Conflict)
                logger.LogWarning(
                    "Rejected command at sequence {Requested} for match {MatchId}; the journal expects {Expected}.",
                    expectedSequence, Short(matchId), classified.Sequence);

            return classified;
        }

        /// <summary>Decides between AlreadyApplied and Conflict once we know the INSERT wrote nothing. The
        /// stored issuer is part of the comparison: the same wire from the other seat is a different command,
        /// not a retry.</summary>
        static async Task<AppendResult> ClassifyAppendAsync(NpgsqlConnection connection, NpgsqlTransaction? transaction,
            Guid matchId, int expectedSequence, string commandWire, string issuerSteamId, CancellationToken ct)
        {
            await using (var existing = new NpgsqlCommand(
                "SELECT command_wire, issuer_steam_id FROM match_commands "
                + "WHERE match_id = @matchId AND sequence = @sequence", connection, transaction))
            {
                existing.Parameters.AddWithValue("matchId", matchId);
                existing.Parameters.AddWithValue("sequence", expectedSequence);

                await using NpgsqlDataReader reader = await existing.ExecuteReaderAsync(ct).ConfigureAwait(false);
                if (await reader.ReadAsync(ct).ConfigureAwait(false)
                    && string.Equals(reader.GetString(0), commandWire, StringComparison.Ordinal)
                    && string.Equals(reader.GetString(1), issuerSteamId, StringComparison.Ordinal))
                    return new AppendResult(AppendStatus.AlreadyApplied, expectedSequence);
            }

            await using var next = new NpgsqlCommand(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM match_commands WHERE match_id = @matchId",
                connection, transaction);
            next.Parameters.AddWithValue("matchId", matchId);
            object? nextSequence = await next.ExecuteScalarAsync(ct).ConfigureAwait(false);

            return new AppendResult(AppendStatus.Conflict, Convert.ToInt32(nextSequence));
        }

        // ---- join credentials ------------------------------------------------

        public async Task StoreJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt, CancellationToken ct)
        {
            MatchStoreGuard.ValidateCredentialHash(credentialHash, nameof(credentialHash));
            MatchStoreGuard.ValidateSteamId(steamId, nameof(steamId));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);

            int inserted;
            try
            {
                // DO NOTHING rather than a bare insert: the credential service retries after a dropped
                // connection, and a retry that lost its answer must not turn a stored credential into an
                // error the caller reports as a failed issue.
                await using var command = new NpgsqlCommand(
                    "INSERT INTO match_join_credentials (credential_hash, match_id, steam_id, expires_at, revoked_at) "
                    + "VALUES (@credentialHash, @matchId, @steamId, @expiresAt, NULL) "
                    + "ON CONFLICT (credential_hash) DO NOTHING", connection);
                command.Parameters.AddWithValue("credentialHash", credentialHash);
                command.Parameters.AddWithValue("matchId", matchId);
                command.Parameters.AddWithValue("steamId", steamId);
                command.Parameters.Add(Timestamp("expiresAt", expiresAt));

                inserted = await command.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
            }
            catch (PostgresException ex) when (ex.SqlState == PostgresErrorCodes.ForeignKeyViolation)
            {
                throw new ArgumentException(MatchStoreGuard.NoSeatMessage, nameof(steamId), ex);
            }

            if (inserted == 1) return;

            // Read back on the connection we are already holding rather than through the public finder,
            // which would open a second one. A pool sized for the load can be entirely checked out by
            // callers sitting exactly here, and then every one of them is waiting for a connection that only
            // another one of them could release: the retry path would deadlock under the load it exists for.
            JoinCredentialRecord? stored =
                await ReadJoinCredentialAsync(connection, null, credentialHash, ct).ConfigureAwait(false);

            if (stored is not null
                && stored.MatchId == matchId
                && string.Equals(stored.SteamId, steamId, StringComparison.Ordinal))
                return;

            // Two different seats cannot share one credential: whoever presented it would be authenticated
            // as the wrong player. In practice this means a 32 byte random collision or a caller bug.
            throw new InvalidOperationException("credential hash already bound to another seat");
        }

        public async Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct)
        {
            MatchStoreGuard.ValidateCredentialHash(credentialHash, nameof(credentialHash));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            return await ReadJoinCredentialAsync(connection, null, credentialHash, ct).ConfigureAwait(false);
        }

        /// <summary>The credential read itself, on a connection the caller already owns. Everything that
        /// needs a credential row while holding a connection uses this; nothing inside this class opens a
        /// second connection to answer a question the first one could have.</summary>
        static async Task<JoinCredentialRecord?> ReadJoinCredentialAsync(NpgsqlConnection connection,
            NpgsqlTransaction? transaction, byte[] credentialHash, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT credential_hash, match_id, steam_id, expires_at, revoked_at "
                + "FROM match_join_credentials WHERE credential_hash = @credentialHash", connection, transaction);
            command.Parameters.AddWithValue("credentialHash", credentialHash);

            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            if (!await reader.ReadAsync(ct).ConfigureAwait(false)) return null;

            return new JoinCredentialRecord(
                reader.GetFieldValue<byte[]>(0),
                reader.GetGuid(1),
                reader.GetString(2),
                ReadTimestamp(reader, 3),
                reader.IsDBNull(4) ? null : ReadTimestamp(reader, 4));
        }

        public async Task RevokeJoinCredentialsAsync(Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct)
        {
            MatchStoreGuard.ValidateSteamId(steamId, nameof(steamId));

            await using NpgsqlConnection connection = await dataSource.OpenConnectionAsync(ct).ConfigureAwait(false);
            await using var command = new NpgsqlCommand(
                "UPDATE match_join_credentials SET revoked_at = @revokedAt "
                + "WHERE match_id = @matchId AND steam_id = @steamId AND revoked_at IS NULL", connection);
            command.Parameters.Add(Timestamp("revokedAt", revokedAt));
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("steamId", steamId);

            await command.ExecuteNonQueryAsync(ct).ConfigureAwait(false);
        }

        // ---- shared plumbing -------------------------------------------------

        static async Task<PersistedMatch?> ReadMatchAsync(NpgsqlConnection connection, NpgsqlTransaction? transaction,
            Guid matchId, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT " + MatchColumns + " FROM matches WHERE match_id = @matchId", connection, transaction);
            command.Parameters.AddWithValue("matchId", matchId);

            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            return await reader.ReadAsync(ct).ConfigureAwait(false) ? ReadMatch(reader) : null;
        }

        static async Task<PersistedMatch?> FindOpenMatchAsync(NpgsqlConnection connection, NpgsqlTransaction? transaction,
            string steamLobbyId, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT " + MatchColumns + " FROM matches "
                + "WHERE steam_lobby_id = @lobbyId AND status IN " + OpenStatuses, connection, transaction);
            command.Parameters.AddWithValue("lobbyId", steamLobbyId);

            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            return await reader.ReadAsync(ct).ConfigureAwait(false) ? ReadMatch(reader) : null;
        }

        static async Task<IReadOnlyList<PersistedPlayer>> ReadPlayersAsync(NpgsqlConnection connection,
            NpgsqlTransaction? transaction, Guid matchId, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT " + PlayerColumns + " FROM match_players WHERE match_id = @matchId ORDER BY seat",
                connection, transaction);
            command.Parameters.AddWithValue("matchId", matchId);

            var players = new List<PersistedPlayer>(2);
            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            while (await reader.ReadAsync(ct).ConfigureAwait(false)) players.Add(ReadPlayer(reader));
            return players;
        }

        static async Task<IReadOnlyList<PersistedCommand>> ReadCommandsAsync(NpgsqlConnection connection,
            NpgsqlTransaction? transaction, Guid matchId, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT " + CommandColumns + " FROM match_commands WHERE match_id = @matchId ORDER BY sequence",
                connection, transaction);
            command.Parameters.AddWithValue("matchId", matchId);

            var commands = new List<PersistedCommand>();
            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            while (await reader.ReadAsync(ct).ConfigureAwait(false))
                commands.Add(new PersistedCommand(
                    reader.GetGuid(0), reader.GetInt32(1), reader.GetString(2), ReadTimestamp(reader, 3),
                    reader.GetString(4)));
            return commands;
        }

        /// <summary>The match status and whether that Steam id holds a seat, read together under the same
        /// row lock so the two answers cannot come from different moments.</summary>
        static async Task<(string? Status, bool HoldsSeat)> ReadStatusAndSeatForUpdateAsync(
            NpgsqlConnection connection, NpgsqlTransaction transaction, Guid matchId, string steamId,
            CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT m.status, EXISTS (SELECT 1 FROM match_players p "
                + "WHERE p.match_id = m.match_id AND p.steam_id = @steamId) "
                + "FROM matches m WHERE m.match_id = @matchId FOR UPDATE", connection, transaction);
            command.Parameters.AddWithValue("matchId", matchId);
            command.Parameters.AddWithValue("steamId", steamId);

            await using NpgsqlDataReader reader = await command.ExecuteReaderAsync(ct).ConfigureAwait(false);
            if (!await reader.ReadAsync(ct).ConfigureAwait(false)) return (null, false);
            return (reader.GetString(0), reader.GetBoolean(1));
        }

        static async Task<string?> ReadStatusForUpdateAsync(NpgsqlConnection connection,
            NpgsqlTransaction transaction, Guid matchId, CancellationToken ct)
        {
            await using var command = new NpgsqlCommand(
                "SELECT status FROM matches WHERE match_id = @matchId FOR UPDATE", connection, transaction);
            command.Parameters.AddWithValue("matchId", matchId);
            return await command.ExecuteScalarAsync(ct).ConfigureAwait(false) as string;
        }

        static PersistedMatch ReadMatch(NpgsqlDataReader reader) => new(
            reader.GetGuid(0),
            reader.IsDBNull(1) ? null : reader.GetString(1),
            MatchStatusText.FromDb(reader.GetString(2)),
            reader.GetString(3),
            reader.IsDBNull(4) ? null : reader.GetString(4),
            reader.GetString(5),
            reader.GetInt32(6),
            reader.GetString(7),
            ReadTimestamp(reader, 8),
            reader.IsDBNull(9) ? null : ReadTimestamp(reader, 9),
            reader.IsDBNull(10) ? null : ReadTimestamp(reader, 10),
            ReadTimestamp(reader, 11),
            reader.IsDBNull(12) ? null : reader.GetInt32(12));

        static PersistedPlayer ReadPlayer(NpgsqlDataReader reader) => new(
            reader.GetGuid(0),
            reader.GetString(1),
            reader.GetInt32(2),
            reader.IsDBNull(3) ? null : reader.GetString(3),
            ReadTimestamp(reader, 4),
            reader.IsDBNull(5) ? null : ReadTimestamp(reader, 5));

        /// <summary>timestamptz always comes back as an instant; normalising to a zero offset here means no
        /// caller ever has to think about the timezone the session happened to be in.</summary>
        static DateTimeOffset ReadTimestamp(NpgsqlDataReader reader, int ordinal) =>
            reader.GetFieldValue<DateTimeOffset>(ordinal).ToUniversalTime();

        static NpgsqlParameter Timestamp(string name, DateTimeOffset value) =>
            new(name, NpgsqlDbType.TimestampTz) { Value = value.UtcDateTime };

        /// <summary>Match ids are logged eight characters wide: enough to follow one match through a log,
        /// short enough not to turn every line into a wall of hex.</summary>
        static string Short(Guid matchId) => matchId.ToString("N").Substring(0, 8);
    }
}
