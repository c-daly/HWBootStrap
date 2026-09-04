namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The only exception the Steam client throws. It carries two separate strings on purpose: Detail is
    /// for operators and goes in logs, PlayerSafeMessage is for the person at the keyboard and goes on the
    /// wire. Detail must never contain a publisher key, an auth ticket, or a URL query string.
    /// </summary>
    public sealed class SteamApiException : Exception
    {
        public SteamApiException(SteamFailure failure, string detail, Exception? inner = null)
            : base(failure + ": " + detail, inner)
        {
            Failure = failure;
            Detail = detail;
            PlayerSafeMessage = SteamFailureMessages.For(failure);
        }

        public SteamFailure Failure { get; }

        /// <summary>Operator-facing reason. Redacted at the point of construction, never re-derived.</summary>
        public string Detail { get; }

        public string PlayerSafeMessage { get; }
    }
}
