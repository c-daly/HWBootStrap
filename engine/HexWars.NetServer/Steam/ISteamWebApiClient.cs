namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The three Steam partner calls the match host needs. Every failure surfaces as
    /// <see cref="SteamApiException"/>, so callers branch on <see cref="SteamFailure"/> and never on a
    /// status code. Implementations must never reach the network from a test.
    /// </summary>
    public interface ISteamWebApiClient
    {
        /// <summary>Verifies a GetAuthTicketForWebApi ticket issued for the identity hexwars-match.</summary>
        Task<SteamIdentity> AuthenticateUserTicketAsync(string ticketHex, CancellationToken ct);

        /// <summary>True when the account holds a live license for the configured App ID.</summary>
        Task<bool> CheckAppOwnershipAsync(string steamId, CancellationToken ct);

        /// <summary>Reads lobby membership and metadata. A missing lobby is LobbyChanged, not an error.</summary>
        Task<SteamLobbySnapshot> GetLobbyDataAsync(string lobbyId, CancellationToken ct);
    }
}
