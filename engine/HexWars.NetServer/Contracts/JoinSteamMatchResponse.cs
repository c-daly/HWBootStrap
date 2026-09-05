namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// The same shape as <see cref="CreateSteamMatchResponse"/>, deliberately: a client that has just been
    /// allocated a match and one that is rejoining an existing one both hold the same five facts, so the
    /// connect path downstream of either answer is one piece of code.
    ///
    /// It is a separate type rather than a reuse because the two endpoints are separately versioned: the
    /// day one of them needs another field, sharing a record would force the other to grow it too.
    /// </summary>
    public sealed record JoinSteamMatchResponse(
        Guid MatchId,
        int ProtocolVersion,
        string WebsocketUrl,
        int Seat,
        string JoinCredential,
        DateTimeOffset CredentialExpiresAt);
}
