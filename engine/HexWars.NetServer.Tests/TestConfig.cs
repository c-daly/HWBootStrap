using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;

namespace HexWars.NetServer.Tests
{
    /// <summary>A host environment that names itself whatever a test needs, without a content root on
    /// disk. Enough for HexWarsConfiguration.Read, which is pure parse-and-validate.</summary>
    internal sealed class TestHostEnvironment : IHostEnvironment
    {
        public string EnvironmentName { get; set; } = Environments.Development;
        public string ApplicationName { get; set; } = "HexWars.NetServer";
        public string ContentRootPath { get; set; } = AppContext.BaseDirectory;
        public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
    }

    /// <summary>Shared builders so every fixture feeds the configuration layer the same flat environment
    /// surface the real server reads. Every secret here is deliberately fake.</summary>
    internal static class TestConfig
    {
        public static IHostEnvironment Env(string name = "Development") =>
            new TestHostEnvironment { EnvironmentName = name };

        public static IConfiguration Config(IDictionary<string, string?> values) =>
            new ConfigurationBuilder().AddInMemoryCollection(values).Build();

        /// <summary>A complete, valid Steam-provider environment.</summary>
        public static Dictionary<string, string?> ValidSteamSettings() => new()
        {
            ["LOBBY_PROVIDER"] = "Steam",
            ["STEAM_APP_ID"] = "480000",
            ["STEAM_PUBLISHER_WEB_API_KEY"] = "fake-publisher-key",
            ["DATABASE_URL"] = "postgres://hexwars:s3cr3t@db.internal:5432/hexwars?sslmode=require",
            ["MATCH_PUBLIC_BASE_URL"] = "https://match.hexwars.invalid",
            ["MATCH_BUILD_ID"] = "build-42",
        };
    }
}
