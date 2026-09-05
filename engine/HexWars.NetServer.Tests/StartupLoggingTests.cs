using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.Logging;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The server prints an environment report once at startup so a deploy can be identified from a log
    /// line. This boots the real host with real-looking secrets and reads back every log line it wrote,
    /// which is the only way to prove the report is identifying the deployment without publishing the
    /// publisher key or the database password.
    /// </summary>
    [TestFixture]
    public class StartupLoggingTests
    {
        const string PublisherKey = "super-secret-key-value";
        const string DatabasePassword = "hunter2";

        [Test]
        public async Task StartupReport_IdentifiesTheDeploymentWithoutPrintingASecret()
        {
            var capture = new CapturingLoggerProvider();
            using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("LOBBY_PROVIDER", "Steam");
                builder.UseSetting("STEAM_APP_ID", "480000");
                builder.UseSetting("STEAM_PUBLISHER_WEB_API_KEY", PublisherKey);
                builder.UseSetting("DATABASE_URL",
                    "postgres://u:" + DatabasePassword + "@db.internal:5432/hexwars");
                builder.UseSetting("MATCH_PUBLIC_BASE_URL", "https://match.invalid/base");
                builder.UseSetting("MATCH_BUILD_ID", "build-42");
                builder.ConfigureLogging(logging => logging.AddProvider(capture));
                // The DATABASE_URL above names a host that does not resolve; this test is about the startup
                // report, not storage, so drop the startup migration that would (correctly) refuse to boot.
                builder.ConfigureServices(services =>
                    services.Remove(services.Single(d => d.ImplementationType == typeof(MigrationHostedService))));
            });
            using var client = factory.CreateClient();

            Assert.That(await client.GetStringAsync("/healthz"), Is.EqualTo("ok"));

            string all = string.Join("\n", capture.Messages);
            Assert.That(capture.Any("Environment report"), Is.True,
                "the startup report was never logged; captured: " + all);
            Assert.That(capture.Any("db.internal:5432/hexwars"), Is.True,
                "the report should still name the database target: " + all);
            Assert.That(capture.Any(PublisherKey), Is.False, "the publisher key reached a log line");
            Assert.That(capture.Any(DatabasePassword), Is.False, "the database password reached a log line");
        }
    }
}
