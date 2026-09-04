using System.Net;
using System.Text.Json;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.FileProviders;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Covers the flat environment-variable configuration surface, its validation, the DATABASE_URL
    /// translation, the environment report, and a WebApplicationFactory smoke test proving the legacy
    /// endpoints are still wired exactly as before (and are unmapped when Legacy is switched off).
    /// </summary>
    [TestFixture]
    public class ConfigurationTests
    {
        sealed class TestEnvironment : IHostEnvironment
        {
            public string EnvironmentName { get; set; } = Environments.Development;
            public string ApplicationName { get; set; } = "HexWars.NetServer";
            public string ContentRootPath { get; set; } = AppContext.BaseDirectory;
            public IFileProvider ContentRootFileProvider { get; set; } = new NullFileProvider();
        }

        static IHostEnvironment Env(string name = "Development") => new TestEnvironment { EnvironmentName = name };

        static IConfiguration Config(IDictionary<string, string?> values) =>
            new ConfigurationBuilder().AddInMemoryCollection(values).Build();

        static Dictionary<string, string?> ValidSteamSettings() => new()
        {
            ["LOBBY_PROVIDER"] = "Steam",
            ["STEAM_APP_ID"] = "480000",
            ["STEAM_PUBLISHER_WEB_API_KEY"] = "fake-publisher-key",
            ["DATABASE_URL"] = "postgres://hexwars:s3cr3t@db.internal:5432/hexwars?sslmode=require",
            ["MATCH_PUBLIC_BASE_URL"] = "https://match.hexwars.invalid",
            ["MATCH_BUILD_ID"] = "build-42",
        };

        static ConfigurationResult Read(IDictionary<string, string?> values, string environment = "Development") =>
            HexWarsConfiguration.Read(Config(values), Env(environment));

        static string Joined(ConfigurationResult r) => string.Join(" | ", r.Errors);

        // ---- defaults ------------------------------------------------------

        [Test]
        public void LegacyDefaults_NothingConfigured_AreValid()
        {
            var result = Read(new Dictionary<string, string?>());

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.LobbyProvider, Is.EqualTo(LobbyProviders.Legacy));
            Assert.That(result.Match.ProtocolVersion, Is.EqualTo(2));
            Assert.That(result.Match.JoinTokenTtlSeconds, Is.EqualTo(900));
            Assert.That(result.Match.TrustForwardedHeaders, Is.False);
            Assert.That(result.Match.AllowedWebOrigins, Is.Empty);
            Assert.That(result.Steam.AppId, Is.EqualTo(0u));
            Assert.That(result.Steam.RequestTimeoutSeconds, Is.EqualTo(5));
            Assert.That(result.Steam.WebApiBaseUrl.ToString(), Does.StartWith("https://partner.steam-api.com"));
        }

        // ---- required keys -------------------------------------------------

        [Test]
        public void SteamProvider_WithNothingElse_NamesExactlyTheFiveRequiredKeys()
        {
            var result = Read(new Dictionary<string, string?> { ["LOBBY_PROVIDER"] = "Steam" });

            Assert.That(result.IsValid, Is.False);
            var keys = result.Errors.Select(e => e.Split(new[] { ": " }, StringSplitOptions.None)[0])
                                    .OrderBy(k => k, StringComparer.Ordinal).ToArray();
            Assert.That(keys, Is.EqualTo(new[]
            {
                "DATABASE_URL",
                "MATCH_BUILD_ID",
                "MATCH_PUBLIC_BASE_URL",
                "STEAM_APP_ID",
                "STEAM_PUBLISHER_WEB_API_KEY",
            }));
            Assert.That(result.Errors, Has.All.EndWith(": missing"));
        }

        [Test]
        public void Production_RequiresTheSameFiveKeysEvenWithoutTheSteamProvider()
        {
            var result = Read(new Dictionary<string, string?>(), "Production");

            Assert.That(result.IsValid, Is.False);
            Assert.That(result.Errors, Has.Count.EqualTo(5));
        }

        [Test]
        public void ValidSteamConfiguration_HasNoErrors()
        {
            var result = Read(ValidSteamSettings());
            Assert.That(result.IsValid, Is.True, Joined(result));
        }

        // ---- placeholders --------------------------------------------------

        [TestCase("changeme")]
        [TestCase("CHANGEME")]
        [TestCase("placeholder")]
        [TestCase("PlaceHolder")]
        [TestCase("your-steam-key")]
        [TestCase("YOUR-STEAM-KEY")]
        [TestCase("xxx")]
        [TestCase("XXX")]
        [TestCase("todo")]
        [TestCase("ToDo")]
        [TestCase("example")]
        [TestCase("EXAMPLE-KEY")]
        public void PlaceholderSecrets_AreRejectedCaseInsensitively(string value)
        {
            var settings = ValidSteamSettings();
            settings["STEAM_PUBLISHER_WEB_API_KEY"] = value;

            var result = Read(settings);

            Assert.That(result.Errors, Is.EqualTo(new[] { "STEAM_PUBLISHER_WEB_API_KEY: placeholder value" }));
        }

        [Test]
        public void PlaceholderDatabaseUrl_ReportsThePlaceholderAndNotAParseError()
        {
            var settings = ValidSteamSettings();
            settings["DATABASE_URL"] = "postgres://hexwars:changeme@db.internal:5432/hexwars";

            var result = Read(settings);

            Assert.That(result.Errors, Is.EqualTo(new[] { "DATABASE_URL: placeholder value" }));
        }

        [Test]
        public void PlaceholderBuildId_IsRejected()
        {
            var settings = ValidSteamSettings();
            settings["MATCH_BUILD_ID"] = "TODO-set-me";

            var result = Read(settings);

            Assert.That(result.Errors, Is.EqualTo(new[] { "MATCH_BUILD_ID: placeholder value" }));
        }

        [Test]
        public void ErrorMessages_NeverContainConfiguredValues()
        {
            var settings = ValidSteamSettings();
            settings["STEAM_PUBLISHER_WEB_API_KEY"] = "changeme-super-secret-9f2c";

            var result = Read(settings);

            Assert.That(Joined(result), Does.Not.Contain("super-secret"));
            Assert.That(Joined(result), Does.Not.Contain("9f2c"));
        }

        // ---- public base url -----------------------------------------------

        [Test]
        public void Production_RejectsPlainHttpPublicBaseUrl()
        {
            var settings = ValidSteamSettings();
            settings["MATCH_PUBLIC_BASE_URL"] = "http://match.hexwars.invalid";

            var result = Read(settings, "Production");

            Assert.That(result.Errors, Is.EqualTo(new[] { "MATCH_PUBLIC_BASE_URL: must use https in Production" }));
        }

        [Test]
        public void Development_AcceptsPlainHttpPublicBaseUrl()
        {
            var settings = ValidSteamSettings();
            settings["MATCH_PUBLIC_BASE_URL"] = "http://match.hexwars.invalid";

            var result = Read(settings, "Development");

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.PublicBaseUrl!.Scheme, Is.EqualTo("http"));
        }

        [Test]
        public void RelativePublicBaseUrl_IsRejected()
        {
            var settings = ValidSteamSettings();
            settings["MATCH_PUBLIC_BASE_URL"] = "/matches";

            var result = Read(settings);

            Assert.That(result.Errors, Is.EqualTo(new[] { "MATCH_PUBLIC_BASE_URL: must be an absolute http or https URL" }));
        }

        // ---- lobby providers -----------------------------------------------

        [Test]
        public void LobbyProvider_CommaSeparatedNames_CombineAsFlags()
        {
            var settings = ValidSteamSettings();
            settings["LOBBY_PROVIDER"] = "legacy,steam";

            var result = Read(settings);

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.LobbyProvider, Is.EqualTo(LobbyProviders.Legacy | LobbyProviders.Steam));
        }

        [Test]
        public void LobbyProvider_UnknownName_IsAValidationError()
        {
            var result = Read(new Dictionary<string, string?> { ["LOBBY_PROVIDER"] = "Bogus" });

            Assert.That(result.Errors, Is.EqualTo(new[] { "LOBBY_PROVIDER: unknown provider name" }));
            Assert.That(result.Match.LobbyProvider, Is.EqualTo(LobbyProviders.Legacy));
        }

        // ---- scalar parsing -------------------------------------------------

        [TestCase("59")]
        [TestCase("86401")]
        [TestCase("not-a-number")]
        public void JoinTokenTtl_OutsideTheAllowedRange_IsAValidationError(string raw)
        {
            var result = Read(new Dictionary<string, string?> { ["MATCH_JOIN_TOKEN_TTL_SECONDS"] = raw });

            Assert.That(result.Errors,
                Is.EqualTo(new[] { "MATCH_JOIN_TOKEN_TTL_SECONDS: must be an integer between 60 and 86400" }));
        }

        [Test]
        public void JoinTokenTtl_InRange_IsAccepted()
        {
            var result = Read(new Dictionary<string, string?> { ["MATCH_JOIN_TOKEN_TTL_SECONDS"] = "120" });

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.JoinTokenTtlSeconds, Is.EqualTo(120));
        }

        [TestCase("true", true)]
        [TestCase("TRUE", true)]
        [TestCase("1", true)]
        [TestCase("false", false)]
        [TestCase("0", false)]
        public void BooleanKeys_AcceptTrueFalseOneZero(string raw, bool expected)
        {
            var result = Read(new Dictionary<string, string?> { ["MATCH_TRUST_FORWARDED_HEADERS"] = raw });

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.TrustForwardedHeaders, Is.EqualTo(expected));
        }

        [Test]
        public void BooleanKeys_RejectAnythingElse()
        {
            var result = Read(new Dictionary<string, string?> { ["MATCH_TRUST_FORWARDED_HEADERS"] = "yes" });

            Assert.That(result.Errors,
                Is.EqualTo(new[] { "MATCH_TRUST_FORWARDED_HEADERS: must be true, false, 1 or 0" }));
        }

        [Test]
        public void CommaLists_AreTrimmedAndEmptyEntriesDropped()
        {
            var result = Read(new Dictionary<string, string?>
            {
                ["ALLOWED_WEB_ORIGINS"] = " https://a.invalid , ,https://b.invalid, ",
                ["MATCH_COMPATIBLE_CLIENT_BUILDS"] = "1.2.3,,1.2.4",
                ["MATCH_BLOCKED_STEAM_IDS"] = "76561190000000001, 76561190000000002",
            });

            Assert.That(result.IsValid, Is.True, Joined(result));
            Assert.That(result.Match.AllowedWebOrigins, Is.EqualTo(new[] { "https://a.invalid", "https://b.invalid" }));
            Assert.That(result.Match.CompatibleClientBuilds, Is.EqualTo(new[] { "1.2.3", "1.2.4" }));
            Assert.That(result.Match.BlockedSteamIds,
                Is.EqualTo(new[] { "76561190000000001", "76561190000000002" }));
        }

        [Test]
        public void MetricsToken_IsNullWhenAbsent()
        {
            Assert.That(Read(new Dictionary<string, string?>()).Match.MetricsToken, Is.Null);
            Assert.That(Read(new Dictionary<string, string?> { ["MATCH_METRICS_TOKEN"] = "abc" }).Match.MetricsToken,
                Is.EqualTo("abc"));
        }

        // ---- DATABASE_URL ---------------------------------------------------

        [Test]
        public void DatabaseUrl_DecodesUserInfoAndMapsRequireSslMode()
        {
            string cs = DatabaseUrl.ToNpgsqlConnectionString(
                "postgres://hex%40user:p%40ss%3Aw0rd@db.internal:6432/hexwars?sslmode=require");

            Assert.That(cs, Does.Contain("Host=db.internal"));
            Assert.That(cs, Does.Contain("Port=6432"));
            Assert.That(cs, Does.Contain("Database=hexwars"));
            Assert.That(cs, Does.Contain("Username=hex@user"));
            Assert.That(cs, Does.Contain("Password=p@ss:w0rd"));
            Assert.That(cs, Does.Contain("SSL Mode=Require"));
            Assert.That(cs, Does.Contain("Trust Server Certificate=true"));
        }

        [Test]
        public void DatabaseUrl_DefaultsThePortAndAcceptsThePostgresqlScheme()
        {
            string cs = DatabaseUrl.ToNpgsqlConnectionString("postgresql://u:p@localhost/hexwars");

            Assert.That(cs, Does.Contain("Host=localhost"));
            Assert.That(cs, Does.Contain("Port=5432"));
            Assert.That(cs, Does.Not.Contain("SSL Mode"));
        }

        [TestCase("prefer", "SSL Mode=Prefer")]
        [TestCase("PREFER", "SSL Mode=Prefer")]
        [TestCase("disable", "SSL Mode=Disable")]
        public void DatabaseUrl_MapsTheOtherSslModes(string mode, string expected)
        {
            string cs = DatabaseUrl.ToNpgsqlConnectionString("postgres://u:p@h.invalid:5432/d?sslmode=" + mode);

            Assert.That(cs, Does.Contain(expected));
            Assert.That(cs, Does.Not.Contain("Trust Server Certificate"));
        }

        [Test]
        public void DatabaseUrl_KeyValueConnectionStringPassesThroughUnchanged()
        {
            const string kv = "Host=db.internal;Port=5432;Database=hexwars;Username=hex;Password=p@ss";

            Assert.That(DatabaseUrl.ToNpgsqlConnectionString(kv), Is.EqualTo(kv));
        }

        [TestCase("")]
        [TestCase("   ")]
        [TestCase("not-a-connection-string")]
        [TestCase("mysql://u:p@h.invalid:3306/d")]
        [TestCase("postgres://")]
        public void DatabaseUrl_GarbageThrowsFormatException(string raw)
        {
            Assert.Throws<FormatException>(() => DatabaseUrl.ToNpgsqlConnectionString(raw));
        }

        [Test]
        public void DescribeTarget_ShowsHostPortDatabaseAndNeverThePassword()
        {
            string target = DatabaseUrl.DescribeTarget(
                "postgres://hexwars:s3cr3t@db.internal:6432/hexwars?sslmode=require");

            Assert.That(target, Is.EqualTo("db.internal:6432/hexwars"));
            Assert.That(target, Does.Not.Contain("s3cr3t"));
        }

        [Test]
        public void DescribeTarget_ReadsAKeyValueConnectionString()
        {
            Assert.That(
                DatabaseUrl.DescribeTarget("Host=db.internal;Port=6432;Database=hexwars;Password=s3cr3t"),
                Is.EqualTo("db.internal:6432/hexwars"));
        }

        [Test]
        public void DescribeTarget_ReportsNoneAndInvalidWithoutThrowing()
        {
            Assert.That(DatabaseUrl.DescribeTarget(""), Is.EqualTo("none"));
            Assert.That(DatabaseUrl.DescribeTarget("garbage"), Is.EqualTo("invalid"));
        }

        // ---- environment report ---------------------------------------------

        [Test]
        public void EnvironmentReport_ExposesIdentityAndNeverSecrets()
        {
            var result = Read(ValidSteamSettings());
            Assert.That(result.IsValid, Is.True, Joined(result));

            var report = EnvironmentReport.Describe(result.Steam, result.Match, Env("Development"));
            string json = report.ToJson();

            Assert.That(json, Does.Contain("480000"));
            Assert.That(json, Does.Contain("build-42"));
            Assert.That(json, Does.Contain("hexwars-engine/1"));
            Assert.That(json, Does.Contain("db.internal:5432/hexwars"));
            Assert.That(json, Does.Not.Contain("fake-publisher-key"));
            Assert.That(json, Does.Not.Contain("s3cr3t"));

            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;
            Assert.That(root.GetProperty("environment").GetString(), Is.EqualTo("Development"));
            Assert.That(root.GetProperty("steamAppId").GetUInt32(), Is.EqualTo(480000u));
            Assert.That(root.GetProperty("buildId").GetString(), Is.EqualTo("build-42"));
            Assert.That(root.GetProperty("protocolVersion").GetInt32(), Is.EqualTo(2));
            Assert.That(root.GetProperty("lobbyProvider").GetString(), Is.EqualTo("Steam"));
            Assert.That(root.GetProperty("engineVersion").GetString(), Is.EqualTo(EngineContract.Version));
            Assert.That(root.GetProperty("publicBaseUrl").GetString(),
                Does.StartWith("https://match.hexwars.invalid"));

            string hash = root.GetProperty("engineAssemblyHash").GetString()!;
            Assert.That(hash == "unavailable" || hash.Length == 16, Is.True, hash);
        }

        [Test]
        public void EnvironmentReport_WithoutADatabase_ReportsNone()
        {
            var result = Read(new Dictionary<string, string?>());
            var report = EnvironmentReport.Describe(result.Steam, result.Match, Env());

            Assert.That(report.DatabaseTarget, Is.EqualTo("none"));
            Assert.That(report.SteamAppId, Is.Null);
            Assert.That(report.LobbyProvider, Is.EqualTo("Legacy"));
            Assert.That(report.PublicBaseUrl, Is.Null);
            Assert.That(report.EngineVersion, Is.EqualTo(EngineContract.Version));
        }

        [Test]
        public void EngineContract_AdvertisesExactlyTheSupportedVersion()
        {
            Assert.That(EngineContract.Version, Is.EqualTo("hexwars-engine/1"));
            Assert.That(EngineContract.SupportedVersions, Is.EquivalentTo(new[] { EngineContract.Version }));
        }

        // ---- options registration -------------------------------------------

        [Test]
        public void AddHexWarsOptions_BindsAValidSteamConfiguration()
        {
            var services = new ServiceCollection();
            services.AddHexWarsOptions(Config(ValidSteamSettings()), Env());
            using var provider = services.BuildServiceProvider();

            Assert.That(provider.GetRequiredService<IOptions<SteamOptions>>().Value.AppId, Is.EqualTo(480000u));
            Assert.That(provider.GetRequiredService<IOptions<MatchHostingOptions>>().Value.LobbyProvider,
                Is.EqualTo(LobbyProviders.Steam));
        }

        [Test]
        public void AddHexWarsOptions_RejectsAnIncompleteSteamConfiguration()
        {
            var services = new ServiceCollection();
            services.AddHexWarsOptions(Config(new Dictionary<string, string?> { ["LOBBY_PROVIDER"] = "Steam" }), Env());
            using var provider = services.BuildServiceProvider();

            var steam = Assert.Throws<OptionsValidationException>(
                () => _ = provider.GetRequiredService<IOptions<SteamOptions>>().Value);
            Assert.That(string.Join(" | ", steam!.Failures), Does.Contain("STEAM_PUBLISHER_WEB_API_KEY: missing"));

            var match = Assert.Throws<OptionsValidationException>(
                () => _ = provider.GetRequiredService<IOptions<MatchHostingOptions>>().Value);
            Assert.That(string.Join(" | ", match!.Failures), Does.Contain("DATABASE_URL: missing"));
        }

        // ---- describe-environment entry point --------------------------------

        [Test]
        public void DescribeEnvironment_PrintsTheReportAndReturnsZero()
        {
            var stdout = new StringWriter();
            var stderr = new StringWriter();

            int code = Program.DescribeEnvironment(Config(ValidSteamSettings()), Env("Development"), stdout, stderr);

            Assert.That(code, Is.EqualTo(0));
            Assert.That(stderr.ToString(), Is.Empty);
            using var doc = JsonDocument.Parse(stdout.ToString());
            Assert.That(doc.RootElement.GetProperty("buildId").GetString(), Is.EqualTo("build-42"));
        }

        [Test]
        public void DescribeEnvironment_ReportsInvalidConfigurationOnStderrAndReturnsTwo()
        {
            var stdout = new StringWriter();
            var stderr = new StringWriter();

            int code = Program.DescribeEnvironment(
                Config(new Dictionary<string, string?> { ["LOBBY_PROVIDER"] = "Steam" }), Env("Development"),
                stdout, stderr);

            Assert.That(code, Is.EqualTo(2));
            Assert.That(stdout.ToString(), Is.Empty);
            string err = stderr.ToString();
            Assert.That(err, Does.Contain("CONFIGURATION INVALID"));
            Assert.That(err, Does.Contain("STEAM_PUBLISHER_WEB_API_KEY: missing"));
            Assert.That(err, Does.Contain("MATCH_BUILD_ID: missing"));
        }

        // ---- server composition smoke tests ----------------------------------

        [Test]
        public async Task LegacyDefaults_StillServeHealthzAndTheLobbyBrowser()
        {
            using var factory = new WebApplicationFactory<Program>();
            using var client = factory.CreateClient();

            Assert.That(await client.GetStringAsync("/healthz"), Is.EqualTo("ok"));

            var games = await client.GetAsync("/games");
            Assert.That(games.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            using var doc = JsonDocument.Parse(await games.Content.ReadAsStringAsync());
            Assert.That(doc.RootElement.TryGetProperty("games", out var array), Is.True);
            Assert.That(array.ValueKind, Is.EqualTo(JsonValueKind.Array));
        }

        [Test]
        public async Task SteamOnlyProvider_UnmapsTheLegacyLobbyButKeepsHealthz()
        {
            using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("LOBBY_PROVIDER", "Steam");
                builder.UseSetting("STEAM_APP_ID", "480000");
                builder.UseSetting("STEAM_PUBLISHER_WEB_API_KEY", "fake-publisher-key");
                builder.UseSetting("DATABASE_URL", "postgres://hexwars:s3cr3t@db.internal:5432/hexwars");
                builder.UseSetting("MATCH_PUBLIC_BASE_URL", "https://match.hexwars.invalid");
                builder.UseSetting("MATCH_BUILD_ID", "build-42");
                builder.ConfigureServices(services =>
                {
                    // This test is about which endpoints get mapped, not about storage. The DATABASE_URL
                    // above names a host that does not resolve, so drop the startup migration that would
                    // otherwise (correctly) refuse to boot. PostgresStartupTests covers that behaviour.
                    services.Remove(services.Single(d => d.ImplementationType == typeof(MigrationHostedService)));
                });
            });
            using var client = factory.CreateClient();

            Assert.That(await client.GetStringAsync("/healthz"), Is.EqualTo("ok"));

            var games = await client.GetAsync("/games");
            Assert.That(games.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
        }
    }
}
