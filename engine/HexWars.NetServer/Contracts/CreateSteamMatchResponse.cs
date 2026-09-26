namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// Everything a client needs to open the match websocket, and nothing else.
    ///
    /// <paramref name="Seat"/> is server-derived - it comes from lobby ownership, not from the request -
    /// and <paramref name="JoinCredential"/> exists only in this response: the server stores a hash of it
    /// and cannot reproduce the value, so a client that loses it must ask for a new one.
    /// </summary>
    public sealed record CreateSteamMatchResponse(
        Guid MatchId,
        int ProtocolVersion,
        string WebsocketUrl,
        int Seat,
        string JoinCredential,
        DateTimeOffset CredentialExpiresAt);
}
