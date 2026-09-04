using HexWars.NetServer.Persistence;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// An <see cref="IMatchStore"/> in a dictionary, for tests about the layers above persistence.
    ///
    /// It is held to the same contract as the real store by <c>MatchStoreContractTests</c>, which runs the
    /// same scenarios against both. That matters more than it sounds: a double that accepted an out-of-order
    /// append, or let a completed match be reopened, would turn every coordinator test that uses it into a
    /// test of a system nobody is going to deploy.
    ///
    /// Two deliberate mirrors of Postgres behaviour: timestamps are normalised to UTC and truncated to the
    /// microsecond, because timestamptz cannot hold a .NET tick; and the recorded rows are copied on the way
    /// in and out, so a caller mutating an array it passed in cannot reach into stored state.
    /// </summary>
    public sealed class InMemoryMatchStore : IMatchStore
    {
        const long TicksPerMicrosecond = 10;

        readonly object _gate = new();
        readonly Dictionary<Guid, MatchRow> _matches = new();
        readonly List<CredentialRow> _credentials = new();

        /// <summary>
        /// When set, the next write throws this and clears the field, so a test can make exactly one durable
        /// write fail and then watch what the caller does about it.
        ///
        /// "Write" means the seven methods that change stored state: create, save catalog, start, append,
        /// complete, store credential and revoke credentials. <see cref="TouchAsync"/> is deliberately not
        /// one of them: a liveness heartbeat failing is not the failure any coordinator test is about, and
        /// having it consume the injected exception would make those tests fragile.
        /// </summary>
        public Exception? InjectedWriteFailure { get; set; }

        /// <summary>How many of those seven write methods have run without throwing. It counts calls, not
        /// rows changed, so an idempotent create that returned an existing match still counts as one.</summary>
        public int WriteCount { get; private set; }

        // ---- allocation ------------------------------------------------------

        public Task<CreateMatchResult> CreateMatchForLobbyAsync(CreateMatchRequest request, CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(request);
            ValidatePlayers(request.Players);

            lock (_gate)
            {
                BeginWrite();

                MatchRow? open = _matches.Values.FirstOrDefault(m =>
                    string.Equals(m.SteamLobbyId, request.SteamLobbyId, StringComparison.Ordinal) && m.IsOpen);
                if (open is not null) return Task.FromResult(new CreateMatchResult(open.Snapshot(), false));

                DateTimeOffset createdAt = Stored(request.CreatedAt);
                var row = new MatchRow
                {
                    MatchId = Guid.NewGuid(),
                    SteamLobbyId = request.SteamLobbyId,
                    Status = MatchStatus.Waiting,
                    SetupWire = request.SetupWire,
                    EngineVersion = request.EngineVersion,
                    ProtocolVersion = request.ProtocolVersion,
                    BuildId = request.BuildId,
                    CreatedAt = createdAt,
                    LastActivityAt = createdAt
                };

                foreach ((string steamId, int seat) in request.Players)
                    row.Players.Add(new PlayerRow { SteamId = steamId, Seat = seat, JoinedAt = createdAt });
                row.Players.Sort((left, right) => left.Seat.CompareTo(right.Seat));

                _matches.Add(row.MatchId, row);
                return Task.FromResult(new CreateMatchResult(row.Snapshot(), true));
            }
        }

        // ---- reads -----------------------------------------------------------

        public Task<PersistedMatch?> GetMatchAsync(Guid matchId, CancellationToken ct)
        {
            lock (_gate)
            {
                return Task.FromResult(_matches.TryGetValue(matchId, out MatchRow? row) ? row.Snapshot() : null);
            }
        }

        public Task<PersistedMatch?> FindOpenMatchForLobbyAsync(string steamLobbyId, CancellationToken ct)
        {
            lock (_gate)
            {
                MatchRow? open = _matches.Values.FirstOrDefault(m =>
                    string.Equals(m.SteamLobbyId, steamLobbyId, StringComparison.Ordinal) && m.IsOpen);
                return Task.FromResult(open?.Snapshot());
            }
        }

        public Task<IReadOnlyList<PersistedPlayer>> GetPlayersAsync(Guid matchId, CancellationToken ct)
        {
            lock (_gate)
            {
                IReadOnlyList<PersistedPlayer> players = _matches.TryGetValue(matchId, out MatchRow? row)
                    ? row.Players.OrderBy(p => p.Seat).Select(p => p.Snapshot(matchId)).ToArray()
                    : Array.Empty<PersistedPlayer>();
                return Task.FromResult(players);
            }
        }

        public Task<PersistedPlayer?> GetPlayerAsync(Guid matchId, string steamId, CancellationToken ct)
        {
            lock (_gate)
            {
                PlayerRow? player = Player(matchId, steamId);
                return Task.FromResult(player?.Snapshot(matchId));
            }
        }

        public Task<MatchJournal?> LoadJournalAsync(Guid matchId, CancellationToken ct)
        {
            lock (_gate)
            {
                if (!_matches.TryGetValue(matchId, out MatchRow? row)) return Task.FromResult<MatchJournal?>(null);

                return Task.FromResult<MatchJournal?>(new MatchJournal(
                    row.Snapshot(),
                    row.Players.OrderBy(p => p.Seat).Select(p => p.Snapshot(matchId)).ToArray(),
                    row.Commands.OrderBy(c => c.Sequence).ToArray()));
            }
        }

        public Task<IReadOnlyList<Guid>> ListOpenMatchIdsAsync(CancellationToken ct)
        {
            lock (_gate)
            {
                IReadOnlyList<Guid> open = _matches.Values
                    .Where(m => m.IsOpen)
                    .OrderBy(m => m.CreatedAt)
                    .Select(m => m.MatchId)
                    .ToArray();
                return Task.FromResult(open);
            }
        }

        // ---- match lifecycle -------------------------------------------------

        public Task SaveCatalogAsync(Guid matchId, string steamId, string catalogWire, CancellationToken ct)
        {
            ValidateSteamId(steamId, nameof(steamId));

            lock (_gate)
            {
                BeginWrite();

                if (_matches.TryGetValue(matchId, out MatchRow? row) && row.Status == MatchStatus.Waiting)
                {
                    // Nobody by that Steam id holds a seat: a full no-op, activity timestamp included, or a
                    // stranger could keep a dead lobby away from the reaper.
                    PlayerRow? player = Player(matchId, steamId);
                    if (player is not null)
                    {
                        player.CatalogWire = catalogWire;
                        // Mirrors the now() the SQL uses: no caller timestamp reaches this method.
                        row.LastActivityAt = Stored(DateTimeOffset.UtcNow);
                    }
                }

                return Task.CompletedTask;
            }
        }

        public Task<bool> TryStartMatchAsync(Guid matchId, string startReplay, DateTimeOffset startedAt, CancellationToken ct)
        {
            lock (_gate)
            {
                BeginWrite();

                if (!_matches.TryGetValue(matchId, out MatchRow? row) || row.Status != MatchStatus.Waiting)
                    return Task.FromResult(false);

                row.Status = MatchStatus.Active;
                row.StartReplay = startReplay;
                row.StartedAt = Stored(startedAt);
                row.LastActivityAt = Stored(startedAt);
                return Task.FromResult(true);
            }
        }

        public Task<bool> TryCompleteMatchAsync(Guid matchId, MatchStatus terminal, int? winnerSeat, DateTimeOffset completedAt, CancellationToken ct)
        {
            ValidateWinnerSeat(winnerSeat, nameof(winnerSeat));

            lock (_gate)
            {
                BeginWrite();

                if (!_matches.TryGetValue(matchId, out MatchRow? row)) return Task.FromResult(false);

                bool allowed = terminal switch
                {
                    MatchStatus.Completed => row.Status == MatchStatus.Active,
                    MatchStatus.Expired or MatchStatus.Abandoned =>
                        row.Status is MatchStatus.Active or MatchStatus.Waiting,
                    _ => false
                };

                // Only a completed match is scored; the schema refuses a winner on any other status. A
                // completed match MAY still have a null winner, which is how a draw is recorded.
                if (winnerSeat is not null && terminal != MatchStatus.Completed) allowed = false;

                if (!allowed) return Task.FromResult(false);

                row.Status = terminal;
                row.WinnerSeat = winnerSeat;
                row.CompletedAt = Stored(completedAt);
                row.LastActivityAt = Stored(completedAt);
                return Task.FromResult(true);
            }
        }

        public Task TouchAsync(Guid matchId, string? steamId, DateTimeOffset seenAt, CancellationToken ct)
        {
            if (steamId is not null) ValidateSteamId(steamId, nameof(steamId));

            lock (_gate)
            {
                if (_matches.TryGetValue(matchId, out MatchRow? row)) row.LastActivityAt = Stored(seenAt);

                if (steamId is not null)
                {
                    PlayerRow? player = Player(matchId, steamId);
                    if (player is not null) player.LastSeenAt = Stored(seenAt);
                }

                return Task.CompletedTask;
            }
        }

        // ---- the command journal ---------------------------------------------

        public Task<AppendResult> AppendCommandAsync(Guid matchId, int expectedSequence, string commandWire, string issuerSteamId, DateTimeOffset acceptedAt, CancellationToken ct)
        {
            ValidateSteamId(issuerSteamId, nameof(issuerSteamId));

            lock (_gate)
            {
                BeginWrite();

                if (!_matches.TryGetValue(matchId, out MatchRow? row) || row.Status != MatchStatus.Active)
                    return Task.FromResult(new AppendResult(AppendStatus.MatchNotActive, expectedSequence));

                // The composite foreign key on match_commands: an issuer with no seat could never have had a
                // command accepted, and Postgres refuses the same insert.
                if (Player(matchId, issuerSteamId) is null)
                    throw new ArgumentException(MatchStoreGuard.NoSeatMessage, nameof(issuerSteamId));

                int next = row.Commands.Count == 0 ? 1 : row.Commands[^1].Sequence + 1;

                if (next == expectedSequence)
                {
                    row.Commands.Add(new PersistedCommand(
                        matchId, expectedSequence, commandWire, Stored(acceptedAt), issuerSteamId));
                    row.LastActivityAt = Stored(acceptedAt);
                    return Task.FromResult(new AppendResult(AppendStatus.Appended, expectedSequence));
                }

                PersistedCommand? existing = row.Commands.FirstOrDefault(c => c.Sequence == expectedSequence);
                if (existing is not null
                    && string.Equals(existing.CommandWire, commandWire, StringComparison.Ordinal)
                    && string.Equals(existing.IssuerSteamId, issuerSteamId, StringComparison.Ordinal))
                    return Task.FromResult(new AppendResult(AppendStatus.AlreadyApplied, expectedSequence));

                return Task.FromResult(new AppendResult(AppendStatus.Conflict, next));
            }
        }

        // ---- join credentials ------------------------------------------------

        public Task StoreJoinCredentialAsync(byte[] credentialHash, Guid matchId, string steamId, DateTimeOffset expiresAt, CancellationToken ct)
        {
            ValidateCredentialHash(credentialHash, nameof(credentialHash));
            ValidateSteamId(steamId, nameof(steamId));

            lock (_gate)
            {
                BeginWrite();

                // ON CONFLICT (credential_hash) DO NOTHING, then read back what is stored: the same seat
                // means an earlier attempt committed after all, so the retry is a success and the stored row
                // keeps the expiry it was issued with. This comes first because a skipped insert checks no
                // foreign key either.
                CredentialRow? clash = Credential(credentialHash);
                if (clash is not null)
                {
                    if (clash.MatchId == matchId
                        && string.Equals(clash.SteamId, steamId, StringComparison.Ordinal))
                        return Task.CompletedTask;

                    throw new InvalidOperationException("credential hash already bound to another seat");
                }

                // The composite foreign key: a credential belongs to a seat, so an unknown match and an
                // unknown player are the same refusal.
                if (Player(matchId, steamId) is null)
                    throw new ArgumentException(MatchStoreGuard.NoSeatMessage, nameof(steamId));

                _credentials.Add(new CredentialRow
                {
                    Hash = (byte[])credentialHash.Clone(),
                    MatchId = matchId,
                    SteamId = steamId,
                    ExpiresAt = Stored(expiresAt)
                });

                return Task.CompletedTask;
            }
        }

        public Task<JoinCredentialRecord?> FindJoinCredentialAsync(byte[] credentialHash, CancellationToken ct)
        {
            ValidateCredentialHash(credentialHash, nameof(credentialHash));

            lock (_gate)
            {
                return Task.FromResult(Credential(credentialHash)?.Snapshot());
            }
        }

        public Task RevokeJoinCredentialsAsync(Guid matchId, string steamId, DateTimeOffset revokedAt, CancellationToken ct)
        {
            ValidateSteamId(steamId, nameof(steamId));

            lock (_gate)
            {
                BeginWrite();

                foreach (CredentialRow credential in _credentials)
                    if (credential.MatchId == matchId
                        && string.Equals(credential.SteamId, steamId, StringComparison.Ordinal)
                        && credential.RevokedAt is null)
                        credential.RevokedAt = Stored(revokedAt);

                return Task.CompletedTask;
            }
        }

        // ---- internals -------------------------------------------------------

        /// <summary>Consumes an injected failure if one is armed, then counts the call.</summary>
        void BeginWrite()
        {
            Exception? failure = InjectedWriteFailure;
            if (failure is not null)
            {
                InjectedWriteFailure = null;
                throw failure;
            }

            WriteCount++;
        }

        PlayerRow? Player(Guid matchId, string steamId) =>
            _matches.TryGetValue(matchId, out MatchRow? row)
                ? row.Players.FirstOrDefault(p => string.Equals(p.SteamId, steamId, StringComparison.Ordinal))
                : null;

        CredentialRow? Credential(byte[] hash) =>
            _credentials.FirstOrDefault(c => c.Hash.SequenceEqual(hash));

        /// <summary>What Postgres would have stored: an instant, truncated to the microsecond timestamptz
        /// holds. Without this the double compares equal where the database would not.</summary>
        static DateTimeOffset Stored(DateTimeOffset value)
        {
            DateTime utc = value.UtcDateTime;
            return new DateTimeOffset(utc.AddTicks(-(utc.Ticks % TicksPerMicrosecond)), TimeSpan.Zero);
        }

        /// <summary>The argument checks the real store applies, called through the shared guard rather
        /// than reimplemented, so the double cannot drift into accepting a request Postgres rejects.</summary>
        static void ValidatePlayers(IReadOnlyList<(string SteamId, int Seat)> players) =>
            MatchStoreGuard.ValidatePlayers(players);

        static void ValidateCredentialHash(byte[] credentialHash, string parameterName) =>
            MatchStoreGuard.ValidateCredentialHash(credentialHash, parameterName);

        static void ValidateSteamId(string? steamId, string parameterName) =>
            MatchStoreGuard.ValidateSteamId(steamId, parameterName);

        static void ValidateWinnerSeat(int? winnerSeat, string parameterName) =>
            MatchStoreGuard.ValidateWinnerSeat(winnerSeat, parameterName);

        sealed class MatchRow
        {
            public Guid MatchId;
            public string? SteamLobbyId;
            public MatchStatus Status;
            public string SetupWire = string.Empty;
            public string? StartReplay;
            public string EngineVersion = string.Empty;
            public int ProtocolVersion;
            public string BuildId = string.Empty;
            public DateTimeOffset CreatedAt;
            public DateTimeOffset? StartedAt;
            public DateTimeOffset? CompletedAt;
            public DateTimeOffset LastActivityAt;
            public int? WinnerSeat;

            public readonly List<PlayerRow> Players = new();
            public readonly List<PersistedCommand> Commands = new();

            public bool IsOpen => Status is MatchStatus.Waiting or MatchStatus.Active;

            public PersistedMatch Snapshot() => new(MatchId, SteamLobbyId, Status, SetupWire, StartReplay,
                EngineVersion, ProtocolVersion, BuildId, CreatedAt, StartedAt, CompletedAt, LastActivityAt,
                WinnerSeat);
        }

        sealed class PlayerRow
        {
            public string SteamId = string.Empty;
            public int Seat;
            public string? CatalogWire;
            public DateTimeOffset JoinedAt;
            public DateTimeOffset? LastSeenAt;

            public PersistedPlayer Snapshot(Guid matchId) =>
                new(matchId, SteamId, Seat, CatalogWire, JoinedAt, LastSeenAt);
        }

        sealed class CredentialRow
        {
            public byte[] Hash = Array.Empty<byte>();
            public Guid MatchId;
            public string SteamId = string.Empty;
            public DateTimeOffset ExpiresAt;
            public DateTimeOffset? RevokedAt;

            public JoinCredentialRecord Snapshot() =>
                new((byte[])Hash.Clone(), MatchId, SteamId, ExpiresAt, RevokedAt);
        }
    }
}
