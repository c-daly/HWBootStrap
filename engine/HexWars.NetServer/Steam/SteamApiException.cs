namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The only exception the Steam client throws. It carries two separate strings on purpose: Detail is
    /// for operators and goes in logs, PlayerSafeMessage is for the person at the keyboard and goes on the
    /// wire. Detail must never contain a publisher key, an auth ticket, or a URL query string.
    ///
    /// There is deliberately no way to attach an inner exception. ToString() renders an inner exception
    /// verbatim, and the transport exceptions this wraps quote the request they failed on - which is the
    /// URL carrying the publisher key and the auth ticket. Callers put a redacted summary of the cause in
    /// Detail instead, so no construction path can leak one back.
    /// </summary>
    public sealed class SteamApiException : Exception
    {
        public SteamApiException(SteamFailure failure, string detail)
            : base(failure + ": " + detail)
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
