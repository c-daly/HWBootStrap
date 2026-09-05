namespace HexWars.NetServer.Auth
{
    /// <summary>
    /// A join credential exactly as the player receives it, together with the moment it stops working.
    ///
    /// The credential string exists here and in the HTTP response that carries it to the client, and
    /// nowhere else: the server keeps only a SHA-256 hash of it and cannot reproduce this value.
    /// </summary>
    public sealed record IssuedCredential(string Credential, DateTimeOffset ExpiresAt);

    /// <summary>
    /// What a credential turned out to be: the match it belongs to, the Steam account behind it, the seat
    /// that account holds, and the identity of the credential itself.
    ///
    /// The hash rather than the credential, and it is here because a socket outlives its handshake. A
    /// credential that expires or is revoked while a game is being played has to stop working then rather
    /// than at the next reconnect, and the only way to ask that question later is to have kept the one
    /// value that identifies the credential without being usable as one.
    /// </summary>
    public sealed record CredentialValidation(
        Guid MatchId, string SteamId, int Seat, byte[] CredentialHash, DateTimeOffset ExpiresAt);

    /// <summary>
    /// Issues and checks the short-lived credential a client presents when it opens the match websocket.
    ///
    /// A bearer credential cannot be made impossible to steal, so this one is built to be worth very
    /// little when it is: it names one seat of one match, it expires, and issuing a new credential for a
    /// seat revokes the ones issued before it. The raw value is never stored and never logged.
    /// </summary>
    public interface IMatchCredentialService
    {
        /// <summary>
        /// Mints a credential for one seat, revoking whatever that seat still held.
        ///
        /// Revoking first is what makes a reconnect safe to repeat: a client that asks twice ends up with
        /// exactly one usable credential, and a credential that leaked out of an abandoned attempt is
        /// already dead by the time the second answer reaches the player.
        /// </summary>
        /// <exception cref="ArgumentException">The Steam id is malformed, or it holds no seat in this
        /// match. A credential is bound to a seat, so there is nothing to issue for a player who is not in
        /// the match, and handing back a token that could never validate would only move the failure to
        /// the websocket handshake.</exception>
        Task<IssuedCredential> IssueAsync(Guid matchId, string steamId, CancellationToken ct);

        /// <summary>
        /// The seat behind a credential, or null when the credential cannot be used for this match.
        ///
        /// Null covers every refusal on purpose - malformed, never issued, issued for another match,
        /// revoked, expired, or issued for a seat that no longer exists. The caller is answering an
        /// unauthenticated socket, so telling it which of those happened would only help a guesser.
        /// </summary>
        Task<CredentialValidation?> ValidateAsync(Guid matchId, string credential, CancellationToken ct);

        /// <summary>
        /// Whether a credential that already opened a socket is still one.
        ///
        /// Asked periodically for the whole life of a connection, because the handshake is a moment and a
        /// match is an hour. A credential revoked by the same player reconnecting elsewhere, or simply run
        /// out of time, leaves a socket that was legitimate when it opened and is not any more; without
        /// this the seat is held until the client chooses to let go of it.
        ///
        /// Takes the hash rather than the credential: nothing on this server keeps the credential, and
        /// nothing should need it to ask this.
        /// </summary>
        Task<bool> IsStillValidAsync(byte[] credentialHash, Guid matchId, DateTimeOffset now);
    }
}
