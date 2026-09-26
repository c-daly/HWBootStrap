namespace HexWars.NetServer.Steam
{
    /// <summary>Every way a Steam-backed match request can fail, in the vocabulary the HTTP layer and the
    /// WebSocket layer both map onto player-visible codes.</summary>
    public enum SteamFailure
    {
        AuthenticationFailed,
        OwnershipMissing,
        NotLobbyMember,
        NotLobbyOwner,
        LobbyChanged,
        IncompatibleVersion,
        ServiceUnavailable,
        RateLimited,
        MalformedResponse,
    }

    /// <summary>
    /// The exact text a player is allowed to see for each failure. Nothing here names an internal
    /// component, a status code or a Steam endpoint: a malformed Valve response is our fault, so it reads
    /// to the player as an ordinary service outage rather than as something they can act on.
    /// </summary>
    public static class SteamFailureMessages
    {
        public const string AuthenticationFailed = "Steam sign-in could not be verified.";
        public const string OwnershipMissing = "This Steam account does not own HexWars.";
        public const string NotLobbyMember = "You are not a member of that lobby.";
        public const string NotLobbyOwner = "Only the lobby owner can start the match.";
        public const string LobbyChanged = "The lobby changed \u2014 check that both players are ready and try again.";
        public const string IncompatibleVersion = "Your game version is not compatible with this server.";
        public const string ServiceUnavailable = "The match service is temporarily unavailable \u2014 try again shortly.";
        public const string RateLimited = "Too many attempts \u2014 wait a moment and try again.";

        public static string For(SteamFailure failure) => failure switch
        {
            SteamFailure.AuthenticationFailed => AuthenticationFailed,
            SteamFailure.OwnershipMissing => OwnershipMissing,
            SteamFailure.NotLobbyMember => NotLobbyMember,
            SteamFailure.NotLobbyOwner => NotLobbyOwner,
            SteamFailure.LobbyChanged => LobbyChanged,
            SteamFailure.IncompatibleVersion => IncompatibleVersion,
            SteamFailure.RateLimited => RateLimited,
            SteamFailure.ServiceUnavailable => ServiceUnavailable,
            SteamFailure.MalformedResponse => ServiceUnavailable,
            _ => ServiceUnavailable,
        };
    }
}
