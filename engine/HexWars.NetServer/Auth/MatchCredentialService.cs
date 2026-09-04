using System.Security.Cryptography;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Auth
{
    /// <summary>
    /// The join credential, start to finish.
    ///
    /// Two decisions shape everything here. The credential is 256 bits of cryptographic randomness rather
    /// than a signed claim, because a bearer token that carries its own meaning has to be verified with a
    /// key the server keeps, and rotating that key would sign every live player out at once. And only the
    /// SHA-256 of it is stored, because the rows outlive the sessions: an operator reading the table, a
    /// backup, or a leaked dump then holds a list of matches rather than a set of working keys.
    ///
    /// Hashing is a bare SHA-256 rather than a password hash on purpose. The input is not a password, it
    /// is 32 uniformly random bytes, so there is no dictionary to slow down and nothing for a work factor
    /// to buy - while the cost of one would land on the websocket handshake of every reconnecting player.
    /// </summary>
    public sealed class MatchCredentialService(
        IMatchStore store,
        IOptions<MatchHostingOptions> options,
        TimeProvider time,
        ILogger<MatchCredentialService> logger) : IMatchCredentialService
    {
        public async Task<IssuedCredential> IssueAsync(Guid matchId, string steamId, CancellationToken ct)
        {
            var raw = new byte[CredentialEncoding.CredentialBytes];
            RandomNumberGenerator.Fill(raw);
            byte[] hash = SHA256.HashData(raw);

            DateTimeOffset now = time.GetUtcNow();
            DateTimeOffset expiresAt = now.AddSeconds(options.Value.JoinTokenTtlSeconds);

            // Revoking before storing is what makes a reconnect safe to repeat. The other order would leave
            // a window where the credential just handed to the player is already revoked, and a player who
            // asked twice would hold two live credentials for one seat.
            await store.RevokeJoinCredentialsAsync(matchId, steamId, now, ct).ConfigureAwait(false);
            await store.StoreJoinCredentialAsync(hash, matchId, steamId, expiresAt, ct).ConfigureAwait(false);

            // The credential itself never appears in a log line, at any level. Only the fact of it does.
            logger.LogInformation(
                "Issued a join credential for {Player} in match {Match}, valid until {ExpiresAt:o}",
                SteamLogRedaction.HashSteamId(steamId), Short(matchId), expiresAt);

            return new IssuedCredential(CredentialEncoding.ToBase64Url(raw), expiresAt);
        }

        public async Task<CredentialValidation?> ValidateAsync(Guid matchId, string credential, CancellationToken ct)
        {
            // Before the store, deliberately. This runs on an unauthenticated websocket frame, so a string
            // that cannot be a credential must cost a regex and not a query, or the handshake becomes a way
            // to point arbitrary traffic at the database.
            if (!CredentialEncoding.TryFromBase64Url(credential, out byte[] raw))
            {
                logger.LogDebug("Join credential for match {Match} refused: not the shape of one", Short(matchId));
                return null;
            }

            JoinCredentialRecord? issued =
                await store.FindJoinCredentialAsync(SHA256.HashData(raw), ct).ConfigureAwait(false);

            if (issued is null)
            {
                logger.LogDebug("Join credential for match {Match} refused: never issued", Short(matchId));
                return null;
            }

            if (issued.MatchId != matchId)
            {
                logger.LogDebug(
                    "Join credential refused: {Player} offered a credential for match {Issued} at match {Match}",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(issued.MatchId), Short(matchId));
                return null;
            }

            if (issued.RevokedAt is not null)
            {
                logger.LogDebug("Join credential for {Player} in match {Match} refused: revoked at {RevokedAt:o}",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(matchId), issued.RevokedAt);
                return null;
            }

            // Inclusive: a credential is dead at its expiry, not one tick after it.
            if (issued.ExpiresAt <= time.GetUtcNow())
            {
                logger.LogDebug("Join credential for {Player} in match {Match} refused: expired at {ExpiresAt:o}",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(matchId), issued.ExpiresAt);
                return null;
            }

            // The seat is read rather than trusted from the credential row, because the seat number is what
            // the caller acts on and a credential that outlived its seat must open nothing.
            PersistedPlayer? seat =
                await store.GetPlayerAsync(matchId, issued.SteamId, ct).ConfigureAwait(false);

            if (seat is null)
            {
                logger.LogDebug("Join credential for {Player} in match {Match} refused: the seat is gone",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(matchId));
                return null;
            }

            return new CredentialValidation(matchId, issued.SteamId, seat.Seat);
        }

        /// <summary>Match ids reach logs as their first eight hex characters: enough to follow one match
        /// through a log file, short enough that a line stays readable.</summary>
        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}
