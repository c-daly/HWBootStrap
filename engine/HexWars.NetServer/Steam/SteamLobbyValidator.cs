using System.Globalization;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// Decides whether a Steam lobby may become a hosted match, and on what terms.
    ///
    /// The single most important property of this class is what it does NOT take: there is no
    /// request-supplied roster, no caller-supplied seat map and no caller-supplied setup parameter. The
    /// only inputs are the snapshot Valve returned and the options of this server, so a client cannot
    /// name its opponent, hand itself seat 0, or smuggle in a board it never advertised to the other
    /// player. The requester is the one thing the caller contributes, and it is only ever used to answer
    /// whether that account is the lobby owner; it can never add anyone to the match.
    ///
    /// The requester arrives as a <see cref="SteamIdentity"/> rather than a string, and ONLY an identity
    /// returned by <see cref="ISteamWebApiClient.AuthenticateUserTicketAsync"/> may be passed. That is the
    /// whole provenance rule: a bare string cannot say whether Valve vouched for it, so accepting one
    /// would let any caller that could reach this class claim to be any account in the lobby. Only
    /// <see cref="SteamIdentity.SteamId"/> decides who plays - never OwnerSteamId, which under Family
    /// Sharing belongs to whoever bought the licence rather than to whoever signed in.
    ///
    /// Every rejection is a <see cref="SteamApiException"/> whose Detail is a fixed operator-facing
    /// string. Details never carry Steam ids: these lines end up in logs, and a lobby id joined to two
    /// account ids is exactly what a log reader should not be handed for free.
    /// </summary>
    public sealed class SteamLobbyValidator(IOptions<SteamOptions> steam, IOptions<MatchHostingOptions> hosting)
    {
        /// <summary>
        /// Applies every lobby rule in order and returns the server-derived match terms. The order is
        /// deliberate: membership before ownership before version before shape, so someone who is not in
        /// the lobby learns nothing about its contents.
        /// </summary>
        /// <param name="requester">An identity Valve returned. Only its SteamId is read.</param>
        public VerifiedLobby ValidateForMatchCreation(SteamLobbySnapshot lobby, SteamIdentity requester)
        {
            var steamOptions = steam.Value;
            var hostingOptions = hosting.Value;

            // 1. Canonicalise every id first. A lobby whose ids we cannot parse is not a lobby we can seat
            //    players from, and it is far more likely to be a mid-flight change than an attack.
            if (!SteamId64.TryNormalize(lobby.OwnerSteamId, out var owner))
            {
                throw Fail(SteamFailure.LobbyChanged, "unparseable member id");
            }

            var members = new List<string>(lobby.Members.Count);
            var memberData = new Dictionary<string, IReadOnlyDictionary<string, string>>(StringComparer.Ordinal);
            foreach (var member in lobby.Members)
            {
                if (!SteamId64.TryNormalize(member.SteamId, out var id))
                {
                    throw Fail(SteamFailure.LobbyChanged, "unparseable member id");
                }

                members.Add(id);
                memberData[id] = member.Data;
            }

            // 2. The requester must be in the lobby before anything else about it is disclosed. It is
            //    the authenticated SteamId that is looked up, never the licence owner.
            if (!SteamId64.TryNormalize(requester.SteamId, out var requesterId) ||
                !members.Contains(requesterId, StringComparer.Ordinal))
            {
                throw Fail(SteamFailure.NotLobbyMember, "requester is not a lobby member");
            }

            // 3. Only the owner starts the match; the guest waits for hw_match to appear.
            if (!string.Equals(requesterId, owner, StringComparison.Ordinal))
            {
                throw Fail(SteamFailure.NotLobbyOwner, "requester is not the lobby owner");
            }

            // 4. A lobby for another App ID is not ours to host.
            var appRaw = Meta(lobby.Metadata, SteamLobbyKeys.App);
            if (appRaw is null ||
                !uint.TryParse(appRaw.Trim(), NumberStyles.None, CultureInfo.InvariantCulture, out var appId) ||
                appId != steamOptions.AppId)
            {
                throw Fail(SteamFailure.IncompatibleVersion, "app id mismatch");
            }

            // 5. Protocol version is an exact match: a v2 client and a v3 client cannot share a match.
            var protocolRaw = Meta(lobby.Metadata, SteamLobbyKeys.Protocol);
            if (protocolRaw is null ||
                !int.TryParse(protocolRaw.Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var protocol) ||
                protocol != hostingOptions.ProtocolVersion)
            {
                throw Fail(SteamFailure.IncompatibleVersion, "protocol");
            }

            // 6. The build allow-list is opt-in: empty means every build is welcome, which is the normal
            //    state. It exists so a bad client build can be shut out without a server redeploy.
            var build = Meta(lobby.Metadata, SteamLobbyKeys.Build);
            build = string.IsNullOrWhiteSpace(build) ? null : build;
            var allowedBuilds = hostingOptions.CompatibleClientBuilds;
            if (allowedBuilds is { Length: > 0 } &&
                (build is null || !allowedBuilds.Contains(build, StringComparer.Ordinal)))
            {
                throw Fail(SteamFailure.IncompatibleVersion, "build");
            }

            // 7. HexWars is strictly 1v1. Anything else means the lobby moved on without us.
            if (members.Count != 2)
            {
                throw Fail(SteamFailure.LobbyChanged, "member count");
            }

            // 8. Both players must still be ready at the moment the owner presses start.
            foreach (var id in members)
            {
                if (!memberData.TryGetValue(id, out var data) ||
                    !data.TryGetValue(SteamLobbyKeys.MemberReady, out var ready) ||
                    !string.Equals(ready, SteamLobbyKeys.ReadyTrue, StringComparison.Ordinal))
                {
                    throw Fail(SteamFailure.LobbyChanged, "not ready");
                }
            }

            // 9. An unknown ruleset is a client we do not understand, treated as a changed lobby rather
            //    than as a version problem: the setup, not the protocol, is what we cannot interpret.
            var ruleset = Meta(lobby.Metadata, SteamLobbyKeys.Ruleset) ?? string.Empty;
            if (!SteamLobbyRules.IsKnownRuleset(ruleset))
            {
                throw Fail(SteamFailure.LobbyChanged, "ruleset");
            }

            // 10. The setup is read from the lobby, never from the request, so both players have seen it.
            var setupRaw = Meta(lobby.Metadata, SteamLobbyKeys.Setup);
            if (string.IsNullOrWhiteSpace(setupRaw))
            {
                throw Fail(SteamFailure.LobbyChanged, "setup missing");
            }

            // 10b. Strictly, not with GameSetup.Parse: the engine parser substitutes a default for every
            //      field it cannot read, so an unreadable hw_setup would quietly become the default board
            //      and start a match neither player agreed to.
            if (!SteamLobbyRules.TryParseSetupStrict(setupRaw, out var setup))
            {
                throw Fail(SteamFailure.LobbyChanged, "setup malformed");
            }

            // 11. quick-v1 pins every field but the seed. The check runs on the value exactly as written
            //     rather than on the sanitized one, so the engine clamp cannot launder an out-of-range
            //     seed into a match that our own client would never have offered. Re-parsing is safe
            //     here and nowhere else: the strict parse above has already proved this string is the
            //     eleven integers ToWire emits, so Parse can no longer invent anything.
            var requested = GameSetup.Parse(setupRaw);
            if (string.Equals(ruleset, SteamLobbyRules.QuickRuleset, StringComparison.Ordinal) &&
                !SteamLobbyRules.IsQuickMatchSetup(requested))
            {
                throw Fail(SteamFailure.LobbyChanged, "setup does not match quick ruleset");
            }

            // 12. Seats come from lobby ownership alone: owner 0, the other member 1.
            string? guest = null;
            foreach (var id in members)
            {
                if (!string.Equals(id, owner, StringComparison.Ordinal))
                {
                    guest = id;
                    break;
                }
            }

            if (guest is null)
            {
                // Two entries that canonicalise to the same account: one player cannot fill both seats.
                throw Fail(SteamFailure.LobbyChanged, "duplicate member");
            }

            var players = new (string SteamId, int Seat)[] { (owner, 0), (guest, 1) };
            return new VerifiedLobby(lobby.LobbyId, owner, players, setup, ruleset, build);
        }

        /// <summary>
        /// Throws with <see cref="SteamFailure.NotLobbyMember"/> unless the account is currently in the
        /// lobby. Used on join, where ownership, readiness and setup were already settled by the owner.
        /// </summary>
        /// <param name="identity">An identity Valve returned. Only its SteamId is read.</param>
        public void EnsureMember(SteamLobbySnapshot lobby, SteamIdentity identity)
        {
            if (!SteamId64.TryNormalize(identity.SteamId, out var canonical))
            {
                throw Fail(SteamFailure.NotLobbyMember, "requester is not a lobby member");
            }

            foreach (var member in lobby.Members)
            {
                if (SteamId64.TryNormalize(member.SteamId, out var id) &&
                    string.Equals(id, canonical, StringComparison.Ordinal))
                {
                    return;
                }
            }

            throw Fail(SteamFailure.NotLobbyMember, "requester is not a lobby member");
        }

        static string? Meta(IReadOnlyDictionary<string, string> metadata, string key) =>
            metadata.TryGetValue(key, out var value) ? value : null;

        static SteamApiException Fail(SteamFailure failure, string detail) => new(failure, detail);
    }
}
