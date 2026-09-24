using System.Globalization;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Endpoints
{
    /// <summary>
    /// The two calls a Steam client makes before it opens a socket: allocate a match from a lobby, and
    /// pick up a credential for a match already allocated.
    ///
    /// The order of the checks in each handler is the security design, not an implementation detail.
    /// Nothing that costs a round trip to Valve happens before the cheap refusals, nothing that reads a
    /// lobby happens before the caller has been authenticated, and nothing is written to the database
    /// before the lobby has been validated. The one fact every later step depends on - who is calling -
    /// comes only from the ticket: no handler here reads a Steam id out of a request body, because a body
    /// is a claim and a ticket is evidence.
    /// </summary>
    public static class SteamMatchEndpoints
    {
        public const string CreateRoute = "/api/v1/steam/matches";
        public const string JoinRoute = "/api/v1/steam/matches/{matchId:guid}/join";

        public const string CreateRateLimitPolicy = "steam-create";
        public const string JoinRateLimitPolicy = "steam-join";

        /// <summary>The v2 websocket route these responses point a client at.</summary>
        public const string WebSocketPath = "/ws/v2";

        const string LoggerCategory = "HexWars.NetServer.Endpoints.SteamMatchEndpoints";

        /// <summary>A lobby id is digits, and a Steam lobby id is not an account id - so it is checked for
        /// shape here rather than through SteamId64, which would reject every real one.</summary>
        const int MaxLobbyIdLength = 20;

        /// <summary>Wide enough for a real GetAuthTicketForWebApi ticket and narrow enough that a body
        /// cannot be used to make this server do megabytes of work before it refuses.</summary>
        const int MinTicketLength = 2;
        const int MaxTicketLength = 8192;

        /// <summary>A GameSetup wire form is well under this; anything longer is not one.</summary>
        const int MaxRequestedSetupLength = 256;

        /// <summary>The partition key when the connection has no remote address, which under a test server
        /// or a misconfigured proxy is every request.</summary>
        public const string UnknownCaller = Hosting.CallerKey.Unknown;

        public static IEndpointRouteBuilder MapSteamMatchEndpoints(this IEndpointRouteBuilder app)
        {
            app.MapPost(CreateRoute, CreateAsync).RequireRateLimiting(CreateRateLimitPolicy);
            app.MapPost(JoinRoute, JoinAsync).RequireRateLimiting(JoinRateLimitPolicy);
            return app;
        }

        static async Task<IResult> CreateAsync(
            HttpContext http,
            ISteamWebApiClient steam,
            SteamLobbyValidator validator,
            IMatchStore store,
            IMatchCredentialService credentials,
            AuthFailureThrottle throttle,
            PlayerBlockList blockList,
            OpenMatchQuota quota,
            TimeProvider time,
            IOptions<MatchHostingOptions> hosting,
            IHostEnvironment environment,
            ILoggerFactory loggerFactory,
            CancellationToken ct)
        {
            ILogger logger = loggerFactory.CreateLogger(LoggerCategory);
            MatchHostingOptions options = hosting.Value;

            if (IsRefusedTransport(http, environment))
            {
                logger.LogWarning("Refused a match creation that arrived over plaintext http");
                return ApiErrors.InvalidRequestResult();
            }

            CreateSteamMatchRequest? request =
                await JsonBody.ReadAsync<CreateSteamMatchRequest>(http.Request, ct: ct).ConfigureAwait(false);

            if (request is null ||
                !IsWellFormedLobbyId(request.SteamLobbyId) ||
                !IsWellFormedTicket(request.Ticket) ||
                !IsWellFormedRequestedSetup(request.RequestedSetup))
            {
                return ApiErrors.InvalidRequestResult();
            }

            string caller = CallerKey(http);
            if (throttle.IsThrottled(caller)) return ApiErrors.RateLimitedResult();

            // Taken under the quota lock before anything is awaited, and taken as a SEAT rather than as an
            // answer to a question. Asking whether there is room and spending it later is check-then-act:
            // the room is granted in between, so a dozen concurrent creations from one address all read the
            // same low number and all proceed. It is also before the Steam round trips and before the write,
            // for the same reason the throttle is: a caller that has filled its share of the database costs
            // this server nothing to refuse.
            string bucket = OpenMatchQuota.BucketFor(http.Connection.RemoteIpAddress);
            bool reserved = quota.TryReserve(bucket, out QuotaLease lease);

            // Nothing has been written yet, so a lease that never gets past here is handed straight back.
            // The one moment that is neither is the create call itself: an exception from it may or may not
            // have left a row, and the two mistakes are not equal, so an ambiguous commit is charged.
            CreationOutcome outcome = CreationOutcome.NothingWritten;

            // The try starts IMMEDIATELY after the seat is taken, and everything else is inside it. Anything
            // between the two - opening a logging scope, one indexed read - can throw, and a seat stranded
            // by a throw is a seat nobody ever gets back.
            try
            {
                // Every line the rest of this call writes carries the lobby, including the ones written from
                // the catch blocks, which is where an operator actually goes looking.
                using IDisposable? lobbyScope = LogScopes.LobbyScope(logger, request.SteamLobbyId);

                // The one request still worth serving over the cap is a retry for a lobby that ALREADY has a
                // match, because that request writes nothing - it hands the caller back the match they were
                // already given. Refusing it would lock a player out of the game they just started because
                // their first response was lost. It costs one indexed read, and only over the cap.
                //
                // The row is CARRIED rather than discarded. Proving an allocation exists and then calling
                // create-or-get several awaits later is a race with its own ending: if the match went
                // terminal in between, the store INSERTS a new one and this request is holding no seat for
                // it - and committing a refused lease cannot take a row back.
                PersistedMatch? provenAllocation = null;
                if (!reserved)
                {
                    provenAllocation = await store
                        .FindOpenMatchForLobbyAsync(request.SteamLobbyId!, ct).ConfigureAwait(false);

                    if (provenAllocation is null)
                    {
                        logger.LogWarning(
                            "Refused a match creation: this address already allocated {Cap} matches this window",
                            options.MaxOpenMatchesPerIp);
                        return ApiErrors.RateLimitedResult();
                    }
                }

                SteamIdentity identity = await AuthenticateAsync(steam, throttle, caller, request.Ticket!, ct);

                if (IsRefusedAccount(blockList, identity, logger, "a match creation"))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status403Forbidden, ApiErrors.Blocked, ApiErrors.BlockedMessage);
                }

                if (!await steam.CheckAppOwnershipAsync(identity.SteamId, ct).ConfigureAwait(false))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status403Forbidden,
                        ApiErrors.OwnershipMissing,
                        SteamFailureMessages.OwnershipMissing);
                }

                SteamLobbySnapshot lobby =
                    await steam.GetLobbyDataAsync(request.SteamLobbyId!, ct).ConfigureAwait(false);

                // Everything from here on is server-derived. The validator is handed the authenticated
                // identity, never the body, so the seats it returns cannot be influenced by the caller.
                VerifiedLobby verified = validator.ValidateForMatchCreation(lobby, identity);

                if (!RequestedSetupStillMatches(request.RequestedSetup, verified.Setup))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, ApiErrors.SettingsChangedMessage);
                }

                var allocation = new CreateMatchRequest(
                    verified.LobbyId,
                    verified.Setup.ToWire(),
                    EngineContract.Version,
                    options.ProtocolVersion,
                    options.BuildId,
                    verified.Players,
                    time.GetUtcNow());

                CreateMatchResult result;

                if (provenAllocation is null)
                {
                    outcome = CreationOutcome.Ambiguous;
                    result = await store.CreateMatchForLobbyAsync(allocation, ct).ConfigureAwait(false);
                }
                else
                {
                    // This request holds no seat: it was let through only because a match already existed
                    // for the lobby. So it may not INSERT one, and create-or-get would. The allocation is
                    // re-read instead, and the answer decides which of two different requests this is.
                    PersistedMatch? stillOpen = await store
                        .FindOpenMatchForLobbyAsync(verified.LobbyId, ct).ConfigureAwait(false);

                    if (stillOpen is not null)
                    {
                        // Still the retry it presented itself as. Nothing is written and nothing is charged.
                        result = new CreateMatchResult(stillOpen, Created: false);
                    }
                    else
                    {
                        // The match it was retrying for has ended. This is now an allocation like any other
                        // and needs a seat of its own before a row can be written for it.
                        lease.Release();

                        if (!quota.TryReserve(bucket, out lease))
                        {
                            logger.LogWarning(
                                "Refused a match creation: the allocation it was retrying for has ended and "
                                + "this address already allocated {Cap} matches this window",
                                options.MaxOpenMatchesPerIp);
                            return ApiErrors.RateLimitedResult();
                        }

                        outcome = CreationOutcome.Ambiguous;
                        result = await store.CreateMatchForLobbyAsync(allocation, ct).ConfigureAwait(false);
                    }
                }

                // Charged only for a match this call actually allocated. A create that found the lobby
                // already had one wrote nothing, and the other seat asking for their credential is the normal
                // way that happens - billing them for it would lock a pair out of their own rematches.
                outcome = result.Created ? CreationOutcome.Created : CreationOutcome.NothingWritten;

                string requesterId = Canonical(identity.SteamId);
                int? seat = SeatOf(verified.Players, requesterId);

                if (!result.Created)
                {
                    // A match for this lobby already existed. The stored roster and setup are what the
                    // journal is keyed by, so they win - and if the lobby has moved on since, this request
                    // is not the retry it looks like. Handing back a credential anyway would seat the
                    // requester in a game the people now in the lobby are not playing.
                    IReadOnlyList<PersistedPlayer> existing = await store
                        .GetPlayersAsync(result.Match.MatchId, ct).ConfigureAwait(false);

                    if (!RosterMatches(existing, verified.Players) ||
                        !string.Equals(result.Match.SetupWire, verified.Setup.ToWire(), StringComparison.Ordinal))
                    {
                        logger.LogInformation(
                            "Refused an existing match {MatchId} for lobby {LobbyId}: the lobby no longer matches it",
                            Short(result.Match.MatchId), verified.LobbyId);
                        return ApiErrors.Failure(
                            StatusCodes.Status409Conflict,
                            ApiErrors.LobbyChanged,
                            SteamFailureMessages.LobbyChanged);
                    }
                }

                if (seat is null)
                {
                    // Unreachable while the validator seats the requester, which it does by construction;
                    // a refusal is still the right answer to a match this player would have no seat in.
                    return ApiErrors.Failure(
                        StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, SteamFailureMessages.LobbyChanged);
                }

                // The protocol the match was WRITTEN under, not the one this process happens to speak. A
                // deployment that has moved on cannot serve a match allocated by the previous one, and
                // saying so is better than issuing a credential for a socket the client cannot talk on.
                if (result.Match.ProtocolVersion != options.ProtocolVersion)
                {
                    logger.LogInformation(
                        "Refused match {MatchId}: it was allocated under protocol {Stored}, this server speaks {Current}",
                        Short(result.Match.MatchId), result.Match.ProtocolVersion, options.ProtocolVersion);
                    return ApiErrors.Failure(
                        StatusCodes.Status426UpgradeRequired,
                        ApiErrors.IncompatibleVersion,
                        SteamFailureMessages.IncompatibleVersion);
                }

                IssuedCredential credential = await credentials
                    .IssueAsync(result.Match.MatchId, requesterId, ct).ConfigureAwait(false);

                logger.LogInformation(
                    "Match created {MatchId} lobby {LobbyId} requester {Sid} seat {Seat}",
                    Short(result.Match.MatchId), verified.LobbyId,
                    SteamLogRedaction.HashSteamId(requesterId), seat.Value);

                return Results.Json(new CreateSteamMatchResponse(
                    result.Match.MatchId,
                    result.Match.ProtocolVersion,
                    WebsocketUrlFor(options.PublicBaseUrl),
                    seat.Value,
                    credential.Credential,
                    credential.ExpiresAt));
            }
            catch (SteamApiException failure)
            {
                logger.LogInformation(
                    "Steam refused a match creation for lobby {LobbyId}: {Failure} ({Detail})",
                    request.SteamLobbyId, failure.Failure, failure.Detail);
                return ApiErrors.From(failure);
            }
            catch (InvalidOperationException closed)
                when (closed.Message == MatchCredentialService.MatchNotOpenMessage)
            {
                // The match ended between the checks above and the credential being issued. Matched on the
                // one message rather than the exception type: any other InvalidOperationException from here
                // is a fault, and answering those with a cheerful 409 would hide it.
                return ApiErrors.Failure(
                    StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, ApiErrors.MatchEndedMessage);
            }
            catch (ArgumentException invalid)
            {
                // The stores reject a malformed or seatless Steam id this way, and so does the credential
                // service. It is a refusal, not a fault: answering 500 would tell a caller the server
                // broke when what actually happened is that their request could not be honoured.
                logger.LogWarning(
                    "Refused a match creation for lobby {LobbyId}: {Reason}",
                    request.SteamLobbyId, SteamLogRedaction.Redact(invalid.Message));
                return ApiErrors.InvalidRequestResult();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception storage)
            {
                // The exception is described rather than handed to the logger. Anything that reaches
                // here failed while holding something: a transport error quotes the URL it could not reach,
                // which for the Steam client is the publisher key and the ticket, and a connection failure
                // quotes DATABASE_URL. Describe keeps the stack trace and takes the values out.
                logger.LogRedactedError(
                    storage, "Match creation failed for lobby {LobbyId}", request.SteamLobbyId);
                return ApiErrors.UnavailableResult();
            }
            finally
            {
                // The single place the seat is settled, whichever of the dozen exits this handler took.
                // Everything before the create call wrote nothing, so the seat goes straight back; the call
                // itself may have left a row it could not tell us about, and that one is charged.
                if (outcome == CreationOutcome.NothingWritten) lease.Release();
                else lease.Commit();
            }
        }

        /// <summary>What the create call managed to do, which is what decides whether the caller is charged
        /// for it. <see cref="Ambiguous"/> is the window inside the store call: a row may exist and this
        /// process cannot say, so it is treated as one.</summary>
        enum CreationOutcome
        {
            NothingWritten,
            Ambiguous,
            Created,
        }

        static async Task<IResult> JoinAsync(
            Guid matchId,
            HttpContext http,
            ISteamWebApiClient steam,
            IMatchStore store,
            IMatchCredentialService credentials,
            AuthFailureThrottle throttle,
            PlayerBlockList blockList,
            IOptions<MatchHostingOptions> hosting,
            IHostEnvironment environment,
            TimeProvider time,
            ILoggerFactory loggerFactory,
            CancellationToken ct)
        {
            ILogger logger = loggerFactory.CreateLogger(LoggerCategory);
            MatchHostingOptions options = hosting.Value;

            // The match is known from the route, so every line this call writes can carry it.
            using IDisposable? matchScope = LogScopes.MatchScope(logger, matchId);

            if (IsRefusedTransport(http, environment))
            {
                logger.LogWarning("Refused a join that arrived over plaintext http");
                return ApiErrors.InvalidRequestResult();
            }

            JoinSteamMatchRequest? request =
                await JsonBody.ReadAsync<JoinSteamMatchRequest>(http.Request, ct: ct).ConfigureAwait(false);

            if (request is null || !IsWellFormedTicket(request.Ticket))
            {
                return ApiErrors.InvalidRequestResult();
            }

            try
            {
                // Authentication before the match lookup, deliberately. A caller with no valid ticket must
                // not be able to tell a match id that never existed from one that has finished, and a
                // throttled caller must cost this server no database read at all.
                string caller = CallerKey(http);
                if (throttle.IsThrottled(caller)) return ApiErrors.RateLimitedResult();

                SteamIdentity identity = await AuthenticateAsync(steam, throttle, caller, request.Ticket!, ct);

                if (IsRefusedAccount(blockList, identity, logger, "a join"))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status403Forbidden, ApiErrors.Blocked, ApiErrors.BlockedMessage);
                }

                PersistedMatch? match = await store.GetMatchAsync(matchId, ct).ConfigureAwait(false);
                if (match is null)
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status404NotFound, ApiErrors.NotFound, ApiErrors.NotFoundMessage);
                }

                // A finished match is refused, with one exception: a game that STARTED and ended within the
                // reconnect window is still joinable, so a seat that missed the final APPLY can get a
                // credential and be shown how it ended. The credential the service then issues is capped at
                // what is left of that window.
                if (match.Status is not (MatchStatus.Waiting or MatchStatus.Active)
                    && !IsInsideTheTerminalWindow(match, options, time.GetUtcNow()))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, ApiErrors.MatchEndedMessage);
                }

                // The protocol the match was written under. This check is only meaningful here and not at
                // allocation: a rolling deploy can leave a match from the previous version open, and the
                // client rejoining it is the one that finds out.
                if (match.ProtocolVersion != options.ProtocolVersion)
                {
                    logger.LogInformation(
                        "Refused a join at match {MatchId}: it speaks protocol {Stored}, this server speaks {Current}",
                        Short(matchId), match.ProtocolVersion, options.ProtocolVersion);
                    return ApiErrors.Failure(
                        StatusCodes.Status426UpgradeRequired,
                        ApiErrors.IncompatibleVersion,
                        SteamFailureMessages.IncompatibleVersion);
                }

                // Membership on join comes from the persisted roster, not from Steam. The lobby settled
                // who plays when the match was created; re-reading it now would let a lobby that has since
                // changed hands add someone to a game already in progress.
                string requesterId = Canonical(identity.SteamId);
                PersistedPlayer? player =
                    await store.GetPlayerAsync(matchId, requesterId, ct).ConfigureAwait(false);

                if (player is null)
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status403Forbidden,
                        ApiErrors.NotLobbyMember,
                        SteamFailureMessages.NotLobbyMember);
                }

                IssuedCredential credential =
                    await credentials.IssueAsync(matchId, requesterId, ct).ConfigureAwait(false);

                logger.LogInformation(
                    "Match joined {MatchId} requester {Sid} seat {Seat}",
                    Short(matchId), SteamLogRedaction.HashSteamId(requesterId), player.Seat);

                return Results.Json(new JoinSteamMatchResponse(
                    matchId,
                    match.ProtocolVersion,
                    WebsocketUrlFor(options.PublicBaseUrl),
                    player.Seat,
                    credential.Credential,
                    credential.ExpiresAt));
            }
            catch (SteamApiException failure)
            {
                logger.LogInformation(
                    "Steam refused a join at match {MatchId}: {Failure} ({Detail})",
                    Short(matchId), failure.Failure, failure.Detail);
                return ApiErrors.From(failure);
            }
            catch (InvalidOperationException closed)
                when (closed.Message == MatchCredentialService.MatchNotOpenMessage)
            {
                // The match finished between the status check above and the credential being issued. The
                // issuing transaction is the only place that window can be closed, so this is not a check
                // that could have been made earlier.
                logger.LogInformation("Match {MatchId} ended while a join was in flight", Short(matchId));
                return ApiErrors.Failure(
                    StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, ApiErrors.MatchEndedMessage);
            }
            catch (ArgumentException invalid)
            {
                logger.LogWarning(
                    "Refused a join at match {MatchId}: {Reason}",
                    Short(matchId), SteamLogRedaction.Redact(invalid.Message));
                return ApiErrors.InvalidRequestResult();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception storage)
            {
                logger.LogRedactedError(storage, "Join failed for match {MatchId}", Short(matchId));
                return ApiErrors.UnavailableResult();
            }
        }

        /// <summary>
        /// Presents the ticket to Steam, counting a rejection against the caller on the way out. Counted
        /// here rather than at the call site so neither endpoint can forget to.
        /// </summary>
        static async Task<SteamIdentity> AuthenticateAsync(
            ISteamWebApiClient steam,
            AuthFailureThrottle throttle,
            string caller,
            string ticket,
            CancellationToken ct)
        {
            try
            {
                return await steam.AuthenticateUserTicketAsync(ticket, ct).ConfigureAwait(false);
            }
            catch (SteamApiException rejected) when (rejected.Failure == SteamFailure.AuthenticationFailed)
            {
                throttle.RecordFailure(caller);
                throw;
            }
        }

        /// <summary>The rate-limit and throttle partition. Never anything from the request, which the
        /// caller controls, and always the same bucketing the other per-caller controls use - a limiter that
        /// counted raw addresses while the quota counted prefixes would be a limiter an IPv6 client walks
        /// straight past.</summary>
        public static string CallerKey(HttpContext http) => Hosting.CallerKey.From(http);

        /// <summary>Whether a match that is over is close enough to its ending for a seat to rejoin it. Only
        /// a match that actually started - one that expired while waiting has no game to show anybody.</summary>
        static bool IsInsideTheTerminalWindow(
            PersistedMatch match, MatchHostingOptions options, DateTimeOffset now)
        {
            if (match.StartReplay is null) return false;
            if (match.CompletedAt is not DateTimeOffset finishedAt) return false;
            if (options.TerminalReconnectSeconds <= 0) return false;

            // Strict, and matched by both stores: at exactly the closing instant there is no window left.
            // This answer is only ever a fast refusal, though - the store re-reads the clock under the row
            // lock, and its answer is the one that decides whether a credential exists.
            return now < finishedAt + TimeSpan.FromSeconds(options.TerminalReconnectSeconds);
        }

        /// <summary>
        /// The websocket URL for this deployment: the public base URL with an upgraded scheme and the v2
        /// path. Any path on the base URL is dropped, because /ws/v2 is mapped at the root of this server
        /// and a client that prefixed it would be asking for a route that does not exist.
        /// </summary>
        internal static string WebsocketUrlFor(Uri? publicBaseUrl)
        {
            if (publicBaseUrl is null) return WebSocketPath;

            string scheme =
                string.Equals(publicBaseUrl.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase)
                    ? "wss"
                    : "ws";

            string authority = publicBaseUrl.IsDefaultPort
                ? publicBaseUrl.Host
                : publicBaseUrl.Host + ":" + publicBaseUrl.Port.ToString(CultureInfo.InvariantCulture);

            return scheme + "://" + authority + WebSocketPath;
        }

        /// <summary>
        /// True unless the client named a setup and the lobby no longer carries it. Parsed strictly: a
        /// requestedSetup that cannot be read is not a match for anything, and treating an unreadable one
        /// as absent would let a client skip the check by sending noise.
        /// </summary>
        internal static bool RequestedSetupStillMatches(string? requestedSetup, GameSetup verified)
        {
            if (string.IsNullOrWhiteSpace(requestedSetup)) return true;

            return SteamLobbyRules.TryParseSetupStrict(requestedSetup, out GameSetup requested) &&
                   SteamLobbyRules.SetupEquals(requested, verified);
        }

        internal static bool IsWellFormedLobbyId(string? value)
        {
            if (string.IsNullOrEmpty(value) || value.Length > MaxLobbyIdLength) return false;

            foreach (char digit in value)
            {
                if (!char.IsAsciiDigit(digit)) return false;
            }

            return true;
        }

        internal static bool IsWellFormedTicket(string? value) =>
            value is { Length: >= MinTicketLength and <= MaxTicketLength } &&
            !string.IsNullOrWhiteSpace(value);

        internal static bool IsWellFormedRequestedSetup(string? value) =>
            value is null || value.Length <= MaxRequestedSetupLength;

        /// <summary>
        /// True for an account this server will not serve: one on the operator block list, or one Valve
        /// says this publisher has banned.
        ///
        /// A VAC ban is deliberately not a refusal. VAC is a signal for the games that consume it, and
        /// HexWars does not; refusing on it would bar players over something that happened in an unrelated
        /// title. It is logged, because an operator investigating a report will want to know.
        /// </summary>
        static bool IsRefusedAccount(
            PlayerBlockList blockList, SteamIdentity identity, ILogger logger, string what)
        {
            string handle = SteamLogRedaction.HashSteamId(identity.SteamId);

            if (identity.PublisherBanned)
            {
                logger.LogWarning("Refused {What} from publisher-banned account {Sid}", what, handle);
                return true;
            }

            if (blockList.IsBlocked(identity.SteamId))
            {
                logger.LogWarning("Refused {What} from blocked account {Sid}", what, handle);
                return true;
            }

            if (identity.VacBanned)
            {
                logger.LogInformation("Allowing {What} from VAC-banned account {Sid}", what, handle);
            }

            return false;
        }

        /// <summary>
        /// True when this request arrived over a transport a Steam ticket may not travel on.
        ///
        /// The body carries a Steam auth ticket and the response carries a join credential, so a
        /// plaintext hop hands both to anyone on the path - and the credential is a bearer token for a
        /// seat, which makes replay the whole attack. Only Production is guarded: Development and the
        /// test host speak http, and a rule that applied everywhere would make these endpoints
        /// unreachable exactly where they are exercised.
        ///
        /// Read after the forwarded-headers middleware, so behind a TLS-terminating proxy this is the
        /// scheme of the client leg rather than of the hop into this process.
        /// </summary>
        static bool IsRefusedTransport(HttpContext http, IHostEnvironment environment) =>
            environment.IsProduction() && !http.Request.IsHttps;

        static string Canonical(string steamId) =>
            SteamId64.TryNormalize(steamId, out string canonical) ? canonical : steamId.Trim();

        /// <summary>
        /// True when the stored roster is seat for seat the one the lobby just verified to. Compared as a
        /// set rather than in order: the store is free to return players in whatever order it likes, and a
        /// test that depended on that order would be pinning something the contract does not promise.
        /// </summary>
        internal static bool RosterMatches(
            IReadOnlyList<PersistedPlayer> stored, IReadOnlyList<(string SteamId, int Seat)> verified)
        {
            if (stored.Count != verified.Count) return false;

            foreach ((string steamId, int seat) in verified)
            {
                bool seated = false;
                foreach (PersistedPlayer player in stored)
                {
                    if (string.Equals(player.SteamId, steamId, StringComparison.Ordinal) && player.Seat == seat)
                    {
                        seated = true;
                        break;
                    }
                }

                if (!seated) return false;
            }

            return true;
        }

        static int? SeatOf(IReadOnlyList<(string SteamId, int Seat)> players, string steamId)
        {
            foreach ((string id, int seat) in players)
            {
                if (string.Equals(id, steamId, StringComparison.Ordinal)) return seat;
            }

            return null;
        }

        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}
