namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// The two identifiers every line about a match should carry, attached as logging scopes rather than
    /// repeated in each message.
    ///
    /// A scope is what makes a log searchable during an incident: an operator wants every line about one
    /// match, and those lines are written by four different components that have no reason to agree on
    /// message wording. Putting the id in the scope means each of them says it once, at the top of the
    /// operation, and every line underneath inherits it.
    ///
    /// Both values are deliberately small. A match id is truncated to its first eight characters, which is
    /// enough to follow one game through a log and not enough to be a handle on anything else; a lobby id is
    /// a Steam lobby rather than an account, so it is safe to write in full. NOTHING else belongs in a
    /// scope: a scope is attached to every line beneath it, so a secret in one is a secret in all of them.
    /// </summary>
    public static class LogScopes
    {
        /// <summary>The scope key for a match. Structured sinks index on it; the console renderer prints it.</summary>
        public const string MatchIdKey = "MatchId";

        public const string LobbyIdKey = "LobbyId";

        /// <summary>Enough of a match id to follow one game through a log.</summary>
        public const int MatchIdLength = 8;

        /// <summary>Opens a scope carrying the match id. Dispose it when the operation ends.</summary>
        public static IDisposable? MatchScope(ILogger logger, Guid matchId)
        {
            ArgumentNullException.ThrowIfNull(logger);

            return logger.BeginScope(new Dictionary<string, object> { [MatchIdKey] = ShortMatchId(matchId) });
        }

        /// <summary>Opens a scope carrying the lobby id.</summary>
        public static IDisposable? LobbyScope(ILogger logger, string? lobbyId)
        {
            ArgumentNullException.ThrowIfNull(logger);

            return logger.BeginScope(new Dictionary<string, object> { [LobbyIdKey] = lobbyId ?? string.Empty });
        }

        /// <summary>The form a match id takes in a log: the first eight characters of its N form.</summary>
        public static string ShortMatchId(Guid matchId) => matchId.ToString("N")[..MatchIdLength];
    }
}
