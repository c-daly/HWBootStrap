using System.Globalization;
using HexWars.NetServer.Steam;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// A scripted Steam partner API. Every answer is a dictionary entry a test wrote, so a test states
    /// what Valve said rather than what the endpoint should conclude from it, and nothing here can reach
    /// the network.
    ///
    /// The identities live behind ticket strings on purpose. That is the provenance rule made testable:
    /// an endpoint can only learn who is calling by presenting a ticket to this class, so a test can put
    /// a Steam id in this table and prove the seat that comes back followed the ticket rather than
    /// anything the request body claimed.
    /// </summary>
    public sealed class FakeSteamWebApiClient : ISteamWebApiClient
    {
        public const string OwnerSteamId = "76561198000000001";
        public const string GuestSteamId = "76561198000000002";

        /// <summary>A real account that is simply not in the lobby.</summary>
        public const string OutsiderSteamId = "76561198000000003";

        public const string OwnerTicket = "a1b2c3d4";
        public const string GuestTicket = "e5f60718";
        public const string OutsiderTicket = "99aabbcc";

        /// <summary>A lobby id, which is NOT an account id: digits, but not a SteamID64.</summary>
        public const string LobbyId = "109775240000000001";

        public const uint AppId = 480000;
        public const int ProtocolVersion = 2;
        public const string BuildId = "test-build";
        public const int QuickSeed = 4242;

        /// <summary>The one hw_setup a quick-v1 lobby on <see cref=\"QuickSeed\"/> may advertise.</summary>
        public static string QuickSetupWire => SteamLobbyRules.QuickMatchSetup(QuickSeed).ToWire();

        /// <summary>Ticket hex to the identity Valve vouches for. An unlisted ticket is AuthenticationFailed.</summary>
        public Dictionary<string, SteamIdentity> Tickets { get; } = new(StringComparer.Ordinal);

        /// <summary>The accounts that own the app. Anything else answers false.</summary>
        public HashSet<string> Ownership { get; } = new(StringComparer.Ordinal);

        /// <summary>Lobby id to snapshot. A missing lobby is LobbyChanged, exactly as the real client.</summary>
        public Dictionary<string, SteamLobbySnapshot> Lobbies { get; } = new(StringComparer.Ordinal);

        /// <summary>Thrown by whichever call comes next, once. Set it to inject an outage.</summary>
        public Exception? NextFailure { get; set; }

        public int AuthenticateCalls { get; private set; }

        public int OwnershipCalls { get; private set; }

        public int LobbyCalls { get; private set; }

        /// <summary>Zeroes the counters so a test can count only the calls it is about.</summary>
        public void ResetCounts()
        {
            AuthenticateCalls = 0;
            OwnershipCalls = 0;
            LobbyCalls = 0;
        }

        /// <summary>
        /// The happy path: three known tickets, all three accounts own the app, and one ready two-member
        /// quick-v1 lobby owned by <see cref=\"OwnerSteamId\"/>.
        /// </summary>
        public static FakeSteamWebApiClient Ready()
        {
            var fake = new FakeSteamWebApiClient();
            fake.Identify(OwnerTicket, OwnerSteamId);
            fake.Identify(GuestTicket, GuestSteamId);
            fake.Identify(OutsiderTicket, OutsiderSteamId);
            fake.Ownership.Add(OwnerSteamId);
            fake.Ownership.Add(GuestSteamId);
            fake.Ownership.Add(OutsiderSteamId);
            fake.Lobbies[LobbyId] = ReadyLobby();
            return fake;
        }

        public void Identify(string ticketHex, string steamId) =>
            Tickets[ticketHex] = new SteamIdentity(steamId, steamId, false, false);

        /// <summary>A lobby that passes every validator rule, with each rule exposed as a parameter so a
        /// test can break exactly one of them.</summary>
        public static SteamLobbySnapshot ReadyLobby(
            string lobbyId = LobbyId,
            bool guestReady = true,
            string? appId = null,
            string? protocol = null,
            string? build = null,
            string ruleset = SteamLobbyRules.QuickRuleset,
            string? setupWire = null,
            string ownerSteamId = OwnerSteamId,
            string guestSteamId = GuestSteamId)
        {
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [SteamLobbyKeys.App] = appId ?? AppId.ToString(CultureInfo.InvariantCulture),
                [SteamLobbyKeys.Protocol] = protocol ?? ProtocolVersion.ToString(CultureInfo.InvariantCulture),
                [SteamLobbyKeys.Build] = build ?? BuildId,
                [SteamLobbyKeys.Ruleset] = ruleset,
                [SteamLobbyKeys.Setup] = setupWire ?? QuickSetupWire,
            };

            var members = new SteamLobbyMember[]
            {
                Member(ownerSteamId, ready: true),
                Member(guestSteamId, guestReady),
            };

            return new SteamLobbySnapshot(lobbyId, ownerSteamId, members, metadata);
        }

        static SteamLobbyMember Member(string steamId, bool ready) =>
            new(steamId, new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [SteamLobbyKeys.MemberReady] = ready ? SteamLobbyKeys.ReadyTrue : "0",
            });

        public Task<SteamIdentity> AuthenticateUserTicketAsync(string ticketHex, CancellationToken ct)
        {
            AuthenticateCalls++;
            if (TakeFailure() is { } injected) return Task.FromException<SteamIdentity>(injected);

            return Tickets.TryGetValue(ticketHex, out SteamIdentity? identity)
                ? Task.FromResult(identity)
                : Task.FromException<SteamIdentity>(
                    new SteamApiException(SteamFailure.AuthenticationFailed, "ticket rejected"));
        }

        public Task<bool> CheckAppOwnershipAsync(string steamId, CancellationToken ct)
        {
            OwnershipCalls++;
            if (TakeFailure() is { } injected) return Task.FromException<bool>(injected);

            return Task.FromResult(Ownership.Contains(steamId));
        }

        public Task<SteamLobbySnapshot> GetLobbyDataAsync(string lobbyId, CancellationToken ct)
        {
            LobbyCalls++;
            if (TakeFailure() is { } injected) return Task.FromException<SteamLobbySnapshot>(injected);

            return Lobbies.TryGetValue(lobbyId, out SteamLobbySnapshot? lobby)
                ? Task.FromResult(lobby)
                : Task.FromException<SteamLobbySnapshot>(
                    new SteamApiException(SteamFailure.LobbyChanged, "lobby not found"));
        }

        Exception? TakeFailure()
        {
            Exception? failure = NextFailure;
            NextFailure = null;
            return failure;
        }
    }
}
