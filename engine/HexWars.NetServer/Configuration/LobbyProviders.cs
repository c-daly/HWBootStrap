namespace HexWars.NetServer.Configuration
{
    /// <summary>Which lobby front doors this process exposes. Flags so a single deployment can run the
    /// legacy WebGL room browser and the Steam match API side by side during the migration.</summary>
    [Flags]
    public enum LobbyProviders
    {
        None = 0,
        Legacy = 1,
        Steam = 2,
    }
}
