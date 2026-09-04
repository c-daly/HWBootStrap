namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// Who Valve says the bearer of an auth ticket is. OwnerSteamId differs from SteamId under Family
    /// Sharing: the ownership check has to run against the owner, the seat belongs to the player.
    /// </summary>
    public sealed record SteamIdentity(string SteamId, string OwnerSteamId, bool VacBanned, bool PublisherBanned);
}
