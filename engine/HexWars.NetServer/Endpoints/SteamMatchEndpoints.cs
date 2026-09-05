using System.Globalization;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Persistence;
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
        /// or a misconfigured proxy is every request. Named rather than empty so it reads in a log.</summary>
        public const string UnknownCaller = "unknown";

        public static IEndpointRouteBuilder MapSteamMatchEndpoints(this IEndpointRouteBuilder app)
        {
            app.MapPost(CreateRoute, CreateAsync).RequireRateLimiting(CreateRateLimitPolicy);
            app.MapPost(JoinRoute, JoinAsync).RequireRateLimiting(JoinRateLimitPolicy);
            return app;
        }

        static async Task<IResult> CreateAsync(
            CreateSteamMatchRequest? request,
            HttpContext http,
            ISteamWebApiClient steam,
            SteamLobbyValidator validator,
            IMatchStore store,
            IMatchCredentialService credentials,
            AuthFailureThrottle throttle,
            TimeProvider time,
            IOptions<MatchHostingOptions> hosting,
            ILoggerFactory loggerFactory,
            CancellationToken ct)
        {
            ILogger logger = loggerFactory.CreateLogger(LoggerCategory);
            MatchHostingOptions options = hosting.Value;

            if (request is null ||
                !IsWellFormedLobbyId(request.SteamLobbyId) ||
                !IsWellFormedTicket(request.Ticket) ||
                !IsWellFormedRequestedSetup(request.RequestedSetup))
            {
                return ApiErrors.InvalidRequestResult();
            }

            string caller = CallerKey(http);
            if (throttle.IsThrottled(caller)) return ApiErrors.RateLimitedResult();

            try
            {
                SteamIdentity identity = await AuthenticateAsync(steam, throttle, caller, request.Ticket!, ct);

                if (IsBlocked(options, identity.SteamId))
                {
                    logger.LogWarning(
                        "Refused a match creation from blocked account {Sid}",
                        SteamLogRedaction.HashSteamId(identity.SteamId));
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

                CreateMatchResult result = await store.CreateMatchForLobbyAsync(
                    new CreateMatchRequest(
                        verified.LobbyId,
                        verified.Setup.ToWire(),
                        EngineContract.Version,
                        options.ProtocolVersion,
                        options.BuildId,
                        verified.Players,
                        time.GetUtcNow()),
                    ct).ConfigureAwait(false);

                string requesterId = Canonical(identity.SteamId);
                int? seat = SeatOf(verified.Players, requesterId);

                if (!result.Created)
                {
                    // A match for this lobby already existed, so the seats it was created with win over
                    // the lobby snapshot just read: the roster may have changed since, and the stored one
                    // is the roster the journal is keyed by. A requester who is not in it is asking to
                    // join a match that moved on without them.
                    PersistedPlayer? player = await store
                        .GetPlayerAsync(result.Match.MatchId, requesterId, ct).ConfigureAwait(false);

                    if (player is null)
                    {
                        logger.LogInformation(
                            "Refused {Sid} an existing match {MatchId} for lobby {LobbyId}: no seat",
                            SteamLogRedaction.HashSteamId(requesterId), Short(result.Match.MatchId),
                            verified.LobbyId);
                        return ApiErrors.Failure(
                            StatusCodes.Status409Conflict,
                            ApiErrors.LobbyChanged,
                            SteamFailureMessages.LobbyChanged);
                    }

                    seat = player.Seat;
                }

                if (seat is null)
                {
                    // Unreachable while the validator seats the requester, which it does by construction;
                    // a refusal is still the right answer to a match this player would have no seat in.
                    return ApiErrors.Failure(
                        StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, SteamFailureMessages.LobbyChanged);
                }

                IssuedCredential credential = await credentials
                    .IssueAsync(result.Match.MatchId, requesterId, ct).ConfigureAwait(false);

                logger.LogInformation(
                    "Match created {MatchId} lobby {LobbyId} requester {Sid} seat {Seat}",
                    Short(result.Match.MatchId), verified.LobbyId,
                    SteamLogRedaction.HashSteamId(requesterId), seat.Value);

                return Results.Json(new CreateSteamMatchResponse(
                    result.Match.MatchId,
                    options.ProtocolVersion,
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
            catch (ArgumentException invalid)
            {
                // The stores reject a malformed or seatless Steam id this way, and so does the credential
                // service. It is a refusal, not a fault: answering 500 would tell a caller the server
                // broke when what actually happened is that their request could not be honoured.
                logger.LogWarning(
                    invalid, "Refused a match creation for lobby {LobbyId}: {Reason}",
                    request.SteamLobbyId, invalid.Message);
                return ApiErrors.InvalidRequestResult();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception storage)
            {
                // Storage is the only thing left that can throw here. The ticket is deliberately not in
                // scope of this line: an exception message may be echoed into a log sink verbatim.
                logger.LogError(
                    storage, "Match creation failed for lobby {LobbyId}", request.SteamLobbyId);
                return ApiErrors.UnavailableResult();
            }
        }

        static async Task<IResult> JoinAsync(
            Guid matchId,
            JoinSteamMatchRequest? request,
            HttpContext http,
            ISteamWebApiClient steam,
            IMatchStore store,
            IMatchCredentialService credentials,
            AuthFailureThrottle throttle,
            IOptions<MatchHostingOptions> hosting,
            ILoggerFactory loggerFactory,
            CancellationToken ct)
        {
            ILogger logger = loggerFactory.CreateLogger(LoggerCategory);
            MatchHostingOptions options = hosting.Value;

            if (request is null || !IsWellFormedTicket(request.Ticket))
            {
                return ApiErrors.InvalidRequestResult();
            }

            try
            {
                PersistedMatch? match = await store.GetMatchAsync(matchId, ct).ConfigureAwait(false);
                if (match is null)
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status404NotFound, ApiErrors.NotFound, ApiErrors.NotFoundMessage);
                }

                if (match.Status is not (MatchStatus.Waiting or MatchStatus.Active))
                {
                    return ApiErrors.Failure(
                        StatusCodes.Status409Conflict, ApiErrors.LobbyChanged, ApiErrors.MatchEndedMessage);
                }

                string caller = CallerKey(http);
                if (throttle.IsThrottled(caller)) return ApiErrors.RateLimitedResult();

                SteamIdentity identity = await AuthenticateAsync(steam, throttle, caller, request.Ticket!, ct);

                if (IsBlocked(options, identity.SteamId))
                {
                    logger.LogWarning(
                        "Refused a join from blocked account {Sid} at match {MatchId}",
                        SteamLogRedaction.HashSteamId(identity.SteamId), Short(matchId));
                    return ApiErrors.Failure(
                        StatusCodes.Status403Forbidden, ApiErrors.Blocked, ApiErrors.BlockedMessage);
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
                    options.ProtocolVersion,
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
            catch (ArgumentException invalid)
            {
                logger.LogWarning(
                    invalid, "Refused a join at match {MatchId}: {Reason}", Short(matchId), invalid.Message);
                return ApiErrors.InvalidRequestResult();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception storage)
            {
                logger.LogError(storage, "Join failed for match {MatchId}", Short(matchId));
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

        /// <summary>The rate-limit and throttle partition: the connection address, or a fixed key when
        /// there is none. Never anything from the request, which the caller controls.</summary>
        public static string CallerKey(HttpContext http) =>
            http.Connection.RemoteIpAddress?.ToString() ?? UnknownCaller;

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

        /// <summary>Compares canonically, so a blocked id configured with padding or in a non-canonical
        /// form still matches the account it was meant to name.</summary>
        internal static bool IsBlocked(MatchHostingOptions options, string steamId)
        {
            if (options.BlockedSteamIds.Length == 0) return false;

            string canonical = Canonical(steamId);
            foreach (string blocked in options.BlockedSteamIds)
            {
                if (string.Equals(Canonical(blocked), canonical, StringComparison.Ordinal)) return true;
            }

            return false;
        }

        static string Canonical(string steamId) =>
            SteamId64.TryNormalize(steamId, out string canonical) ? canonical : steamId.Trim();

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
