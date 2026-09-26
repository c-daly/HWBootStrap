namespace HexWars.NetServer.Steam
{
    /// <summary>One member of a Steam lobby plus their per-member data (hw_ready and friends).</summary>
    public sealed record SteamLobbyMember(string SteamId, IReadOnlyDictionary<string, string> Data);

    /// <summary>
    /// A point-in-time read of a Steam lobby. It is a snapshot in the strong sense: by the time the
    /// validator looks at it a player may already have left, which is exactly why every decision taken
    /// from it is re-checked before the match is written.
    /// </summary>
    public sealed record SteamLobbySnapshot(
        string LobbyId,
        string OwnerSteamId,
        IReadOnlyList<SteamLobbyMember> Members,
        IReadOnlyDictionary<string, string> Metadata);
}
