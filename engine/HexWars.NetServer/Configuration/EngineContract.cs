namespace HexWars.NetServer.Configuration
{
    /// <summary>The rules contract a stored replay was produced under. Bump <see cref="Version"/> whenever
    /// an engine change makes an existing journal replay to a different state, and keep every version that
    /// still replays identically in <see cref="SupportedVersions"/>.</summary>
    public static class EngineContract
    {
        public const string Version = "hexwars-engine/1";

        public static readonly IReadOnlySet<string> SupportedVersions = new HashSet<string>(StringComparer.Ordinal) { Version };
    }
}
