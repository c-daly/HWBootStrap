using HexWars.Engine;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// A Steam lobby that has passed every rule in <see cref="SteamLobbyValidator"/>. Everything on it is
    /// server-derived: the ids are canonical, the seats come from lobby ownership rather than from the
    /// request, and <see cref="Setup"/> is already sanitized, so a consumer can hand it straight to
    /// GameFactory and to the match store without re-checking anything.
    /// </summary>
    /// <param name="LobbyId">The Steam lobby id, exactly as the snapshot reported it.</param>
    /// <param name="OwnerSteamId">Canonical SteamID64 of the lobby owner, who always takes seat 0.</param>
    /// <param name="Players">Canonical id and seat for every member, owner first.</param>
    /// <param name="Setup">The sanitized GameSetup the match must be built from.</param>
    /// <param name="Ruleset">quick-v1 or custom.</param>
    /// <param name="ClientBuild">The client build the lobby advertised, or null when it advertised none.</param>
    public sealed record VerifiedLobby(
        string LobbyId,
        string OwnerSteamId,
        IReadOnlyList<(string SteamId, int Seat)> Players,
        GameSetup Setup,
        string Ruleset,
        string? ClientBuild);
}
