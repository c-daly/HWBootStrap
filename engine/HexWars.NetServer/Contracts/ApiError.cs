using HexWars.NetServer.Steam;

namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// The one error body every Steam match endpoint returns: a stable machine code the client branches
    /// on, and a sentence a player is allowed to read.
    ///
    /// The split is the point. The code is contract - the Unity client maps it to a retry, a sign-in
    /// prompt or a dead end - while the message is prose that may be reworded without breaking anything.
    /// Nothing derived from an exception ever reaches the message: it is chosen from a fixed set here.
    /// </summary>
    public sealed record ApiError(string Error, string Message);

    /// <summary>
    /// The error vocabulary and the single mapping from a Steam failure onto it.
    ///
    /// Every endpoint funnels its refusals through this class so the wire codes cannot drift apart
    /// between create and join, and so a new <see cref="SteamFailure"/> has exactly one place to be
    /// answered rather than one per endpoint.
    /// </summary>
    public static class ApiErrors
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

        /// <summary>Deliberately says nothing about WHICH field was wrong. A malformed body is either a
        /// client bug, which a log line serves better than a response, or someone probing.</summary>
        public const string InvalidRequestMessage = "That request was not valid.";

        /// <summary>For an account this server has been told not to serve. It names no appeal route on
        /// purpose; support does that, an error body cannot.</summary>
        public const string BlockedMessage = "This account cannot start a match.";

        public const string NotFoundMessage = "That match no longer exists.";

        /// <summary>The lobby moved on between the client reading it and this server reading it.</summary>
        public const string SettingsChangedMessage = "The requested settings no longer match the lobby.";

        public const string MatchEndedMessage = "That match has ended.";

        public static IResult Failure(int statusCode, string error, string message) =>
            Results.Json(new ApiError(error, message), statusCode: statusCode);

        public static IResult InvalidRequestResult() =>
            Failure(StatusCodes.Status400BadRequest, InvalidRequest, InvalidRequestMessage);

        public static IResult RateLimitedResult() =>
            Failure(StatusCodes.Status429TooManyRequests, RateLimited, SteamFailureMessages.RateLimited);

        public static IResult UnavailableResult() =>
            Failure(StatusCodes.Status503ServiceUnavailable, ServiceUnavailable, SteamFailureMessages.ServiceUnavailable);

        /// <summary>
        /// The single translation from a Steam refusal to an HTTP one. The message is always the
        /// exception PlayerSafeMessage, never its Detail: Detail is written for an operator and may name
        /// an internal reason, and this string goes to the player.
        ///
        /// MalformedResponse deliberately lands on 503 rather than a 5xx that blames the client: a
        /// response we could not parse is our problem, and to the player it is indistinguishable from
        /// Steam being down.
        /// </summary>
        public static IResult From(SteamApiException exception)
        {
            (int status, string code) = exception.Failure switch
            {
                SteamFailure.AuthenticationFailed => (StatusCodes.Status401Unauthorized, AuthenticationFailed),
                SteamFailure.OwnershipMissing => (StatusCodes.Status403Forbidden, OwnershipMissing),
                SteamFailure.NotLobbyMember => (StatusCodes.Status403Forbidden, NotLobbyMember),
                SteamFailure.NotLobbyOwner => (StatusCodes.Status403Forbidden, NotLobbyOwner),
                SteamFailure.LobbyChanged => (StatusCodes.Status409Conflict, LobbyChanged),
                SteamFailure.IncompatibleVersion => (StatusCodes.Status426UpgradeRequired, IncompatibleVersion),
                SteamFailure.RateLimited => (StatusCodes.Status429TooManyRequests, RateLimited),
                SteamFailure.ServiceUnavailable => (StatusCodes.Status503ServiceUnavailable, ServiceUnavailable),
                SteamFailure.MalformedResponse => (StatusCodes.Status503ServiceUnavailable, ServiceUnavailable),
                _ => (StatusCodes.Status503ServiceUnavailable, ServiceUnavailable),
            };

            return Failure(status, code, exception.PlayerSafeMessage);
        }
    }
}
