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
        /// <summary>
        /// The message on the <see cref="InvalidOperationException"/> a refused issue carries. A constant
        /// rather than a bespoke exception type so callers can match this one refusal precisely: an
        /// endpoint that caught every InvalidOperationException would also turn a genuine store fault into
        /// a cheerful 409.
        /// </summary>
        public const string MatchNotOpenMessage = "match is not open";

        public async Task<IssuedCredential> IssueAsync(Guid matchId, string steamId, CancellationToken ct)
        {
            var raw = new byte[CredentialEncoding.CredentialBytes];
            RandomNumberGenerator.Fill(raw);
            byte[] hash = SHA256.HashData(raw);

            DateTimeOffset now = time.GetUtcNow();
            DateTimeOffset expiresAt = now.AddSeconds(options.Value.JoinTokenTtlSeconds);
            TimeSpan window = TerminalWindow;

            // One store call, not two. Revoking and storing separately leaves a window where a concurrent
            // reconnect ends with two live credentials for one seat, and a failure between the two destroys
            // the only credential the player had. The store does both inside a single transaction that also
            // refuses a match which finished while the request was in flight.
            // The full TTL is what is ASKED for. What is stored is what the store decides under the match
            // row lock: a match that finishes between this call being made and that lock being taken caps
            // the credential at what is left of the reconnect window, and only the transaction can know
            // that. Reading the status here first and capping from it would be a decision made on a value
            // that was already able to change.
            CredentialReplacement replacement = await store
                .ReplaceJoinCredentialAsync(hash, matchId, steamId, expiresAt, now, ct, window)
                .ConfigureAwait(false);

            if (!replacement.Replaced)
            {
                logger.LogInformation(
                    "Refused a join credential for {Player} in match {Match}: the match is no longer open",
                    SteamLogRedaction.HashSteamId(steamId), Short(matchId));
                throw new InvalidOperationException(MatchNotOpenMessage);
            }

            DateTimeOffset stored = replacement.EffectiveExpiresAt ?? expiresAt;

            // The credential itself never appears in a log line, at any level. Only the fact of it does.
            logger.LogInformation(
                "Issued a join credential for {Player} in match {Match}, valid until {ExpiresAt:o}",
                SteamLogRedaction.HashSteamId(steamId), Short(matchId), stored);

            return new IssuedCredential(CredentialEncoding.ToBase64Url(raw), stored);
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
            // A credential outlives the game it was issued for: it is valid for its full TTL, and nothing
            // revokes it when the match ends. Without this the websocket handshake would happily seat a
            // player into a finished match and then have to discover the problem afterwards.
            PersistedMatch? match = await store.GetMatchAsync(matchId, ct).ConfigureAwait(false);
            if (match is null || !StillReachable(match))
            {
                logger.LogDebug(
                    "Join credential for {Player} in match {Match} refused: the match is no longer open",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(matchId));
                return null;
            }

            PersistedPlayer? seat =
                await store.GetPlayerAsync(matchId, issued.SteamId, ct).ConfigureAwait(false);

            if (seat is null)
            {
                logger.LogDebug("Join credential for {Player} in match {Match} refused: the seat is gone",
                    SteamLogRedaction.HashSteamId(issued.SteamId), Short(matchId));
                return null;
            }

            return new CredentialValidation(
                matchId, issued.SteamId, seat.Seat, issued.CredentialHash, issued.ExpiresAt);
        }

        public async Task<bool> IsStillValidAsync(
            byte[] credentialHash, Guid matchId, DateTimeOffset now, CancellationToken ct)
        {
            ArgumentNullException.ThrowIfNull(credentialHash);

            JoinCredentialRecord? issued =
                await store.FindJoinCredentialAsync(credentialHash, ct).ConfigureAwait(false);

            if (issued is null || issued.MatchId != matchId) return false;
            if (issued.RevokedAt is not null) return false;
            if (issued.ExpiresAt <= now) return false;

            // And the window, independently of the credential. A credential stored before the match ended
            // carries an expiry that knows nothing about the ending, so a socket holding one would outlive
            // the window by however much of its TTL was left.
            PersistedMatch? match = await store.GetMatchAsync(matchId, ct).ConfigureAwait(false);
            if (match is null) return false;
            if (match.Status is MatchStatus.Waiting or MatchStatus.Active) return true;

            TimeSpan window = TerminalWindow;

            return window > TimeSpan.Zero
                && match.CompletedAt is DateTimeOffset finishedAt
                && now - finishedAt <= window;
        }

        /// <summary>How long after a match ends its seats can still get back in. Zero closes the window.</summary>
        TimeSpan TerminalWindow
        {
            get
            {
                int seconds = options.Value.TerminalReconnectSeconds;
                return seconds <= 0 ? TimeSpan.Zero : TimeSpan.FromSeconds(seconds);
            }
        }

        /// <summary>
        /// Whether a socket may still be opened into this match.
        ///
        /// Waiting and active are the game itself. A match that STARTED and has since ended stays reachable
        /// for MATCH_TERMINAL_RECONNECT_SECONDS afterwards, because the final APPLY is the frame most likely
        /// to be lost - it is broadcast at the instant the match becomes terminal - and a player whose
        /// socket dropped a moment earlier has no other way to learn how the game they were playing ended.
        ///
        /// Any terminal status, not only completed. A game the reaper abandoned or expired underneath its
        /// players ended just as definitely as one somebody won, and the seats deserve to be shown the same
        /// final position rather than a bare refusal.
        ///
        /// A match that never started is never reachable once it is over: there is no game in it to show.
        /// </summary>
        bool StillReachable(PersistedMatch match)
        {
            if (match.Status is MatchStatus.Waiting or MatchStatus.Active) return true;
            if (match.StartReplay is null) return false;
            if (match.CompletedAt is not DateTimeOffset completedAt) return false;

            TimeSpan window = TerminalWindow;
            if (window <= TimeSpan.Zero) return false;

            return time.GetUtcNow() - completedAt <= window;
        }

        /// <summary>Match ids reach logs as their first eight hex characters: enough to follow one match
        /// through a log file, short enough that a line stays readable.</summary>
        static string Short(Guid matchId) => matchId.ToString("N")[..8];
    }
}
