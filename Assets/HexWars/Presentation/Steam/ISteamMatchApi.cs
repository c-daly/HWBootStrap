#nullable enable
using System;

namespace HexWars.Presentation
{
    /// <summary>Error codes the match service returns in its JSON error body.</summary>
    public static class SteamMatchErrorCodes
    {
        public const string AuthenticationFailed = "authentication_failed";
        public const string OwnershipMissing = "ownership_missing";
        public const string NotLobbyMember = "not_lobby_member";
        public const string NotLobbyOwner = "not_lobby_owner";
        public const string LobbyChanged = "lobby_changed";
        public const string IncompatibleVersion = "incompatible_version";
        public const string ServiceUnavailable = "service_unavailable";
        public const string RateLimited = "rate_limited";
        public const string NotFound = "not_found";
        public const string Blocked = "blocked";
        public const string InvalidRequest = "invalid_request";
    }

    /// <summary>The outcome of one match-service call, success or failure.</summary>
    public sealed class SteamMatchApiResult
    {
        public SteamMatchApiResult(
            bool ok,
            long httpStatus,
            string? matchId,
            string? websocketUrl,
            string? joinCredential,
            int seat,
            string? errorCode,
            string? message)
        {
            Ok = ok;
            HttpStatus = httpStatus;
            MatchId = matchId;
            WebsocketUrl = websocketUrl;
            JoinCredential = joinCredential;
            Seat = seat;
            ErrorCode = errorCode;
            Message = message;
        }

        public bool Ok { get; }

        /// <summary>HTTP status code, or 0 when the request never reached the server.</summary>
        public long HttpStatus { get; }

        public string? MatchId { get; }

        public string? WebsocketUrl { get; }

        public string? JoinCredential { get; }

        public int Seat { get; }

        /// <summary>The <c>error</c> field of the failure body, or null on success.</summary>
        public string? ErrorCode { get; }

        /// <summary>The player-safe <c>message</c> field, or null.</summary>
        public string? Message { get; }

        public static SteamMatchApiResult Success(string matchId, string websocketUrl, string joinCredential, int seat)
        {
            return new SteamMatchApiResult(true, 200, matchId, websocketUrl, joinCredential, seat, null, null);
        }

        public static SteamMatchApiResult Failure(long httpStatus, string errorCode, string message)
        {
            return new SteamMatchApiResult(false, httpStatus, null, null, null, 0, errorCode, message);
        }

        /// <summary>The request never reached the match service (DNS, TLS, timeout, offline).</summary>
        public static SteamMatchApiResult NetworkFailure(string? message = null)
        {
            return new SteamMatchApiResult(false, 0, null, null, null, 0, null, message);
        }
    }

    /// <summary>Everything the game needs to open the match websocket.</summary>
    public sealed class SteamMatchTicket
    {
        public SteamMatchTicket(string? matchId, string? websocketUrl, string? joinCredential, int seat)
        {
            MatchId = matchId ?? string.Empty;
            WebsocketUrl = websocketUrl ?? string.Empty;
            JoinCredential = joinCredential ?? string.Empty;
            Seat = seat;
        }

        public string MatchId { get; }

        public string WebsocketUrl { get; }

        /// <summary>The single-use credential for the <c>AUTH</c> frame. Never log this.</summary>
        public string JoinCredential { get; }

        /// <summary>0 for the lobby owner, 1 for the other member.</summary>
        public int Seat { get; }
    }

    /// <summary>
    /// The two match-service calls the lobby flow makes. Implementations deliver every result on the
    /// main thread; <see cref="Cancel"/> abandons whatever is in flight.
    /// </summary>
    public interface ISteamMatchApi
    {
        /// <summary>Owner side: allocate a match for a lobby. POST /api/v1/steam/matches.</summary>
        void CreateMatch(string lobbyId, string ticketHex, string requestedSetupWire, Action<SteamMatchApiResult> onDone);

        /// <summary>Guest side: join an allocated match. POST /api/v1/steam/matches/{matchId}/join.</summary>
        void JoinMatch(string matchId, string ticketHex, Action<SteamMatchApiResult> onDone);

        /// <summary>Abandons any in-flight request. Pending callbacks must not fire afterwards.</summary>
        void Cancel();
    }
}
