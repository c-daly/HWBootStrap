using System.Security.Cryptography;
using System.Text.Json;
using HexWars.NetServer.Persistence;
using Microsoft.Extensions.Hosting;

namespace HexWars.NetServer.Configuration
{
    /// <summary>
    /// What this process believes it is: environment, App ID, build, protocol, lobby providers, the exact
    /// engine build serving replays, and where the database lives. Printed by the describe-environment
    /// subcommand and logged once at startup so a deploy can be identified from a log line alone.
    /// Deliberately carries no secret: no publisher key, no connection string, no credentials.
    /// </summary>
    public sealed record EnvironmentReport(
        string Environment,
        uint? SteamAppId,
        string BuildId,
        int ProtocolVersion,
        string LobbyProvider,
        string EngineVersion,
        string EngineAssemblyHash,
        string DatabaseTarget,
        string? PublicBaseUrl)
    {
        static readonly JsonSerializerOptions JsonOptions = new()
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        };

        static readonly Lazy<string> CachedEngineHash = new(ComputeEngineAssemblyHash, isThreadSafe: true);

        public static EnvironmentReport Describe(SteamOptions steam, MatchHostingOptions match, IHostEnvironment env) =>
            new(
                env.EnvironmentName,
                steam.AppId == 0 ? null : steam.AppId,
                match.BuildId,
                match.ProtocolVersion,
                match.LobbyProvider.ToString(),
                EngineContract.Version,
                CachedEngineHash.Value,
                string.IsNullOrWhiteSpace(match.DatabaseUrl) ? "none" : DatabaseUrl.DescribeTarget(match.DatabaseUrl),
                match.PublicBaseUrl?.ToString());

        public string ToJson() => JsonSerializer.Serialize(this, JsonOptions);

        /// <summary>The full SHA-256 digest of the loaded HexWars.Engine assembly, formatted
        /// sha256:&lt;64 lowercase hex&gt;, or "unavailable" when the assembly has no on-disk location. The
        /// complete digest is what proves two servers replay a journal against identical rules; a truncated
        /// prefix is not a digest a replay build can be identified by years later.</summary>
        static string ComputeEngineAssemblyHash()
        {
            try
            {
                string path = typeof(HexWars.Engine.GameSetup).Assembly.Location;
                if (string.IsNullOrEmpty(path) || !File.Exists(path)) return "unavailable";
                using var stream = File.OpenRead(path);
                return "sha256:" + Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
            }
            catch (Exception)
            {
                return "unavailable";
            }
        }
    }
}
