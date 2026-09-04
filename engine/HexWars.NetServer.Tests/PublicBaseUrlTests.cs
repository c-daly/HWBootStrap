using HexWars.NetServer.Configuration;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// MATCH_PUBLIC_BASE_URL is echoed into the environment report and logged once at startup, so it must
    /// never be able to carry a secret. Validation rejects userinfo, a query string and a fragment; the
    /// report renders scheme, authority and path only, so even a value that arrived some other way cannot
    /// put an API key or a token into a log line.
    /// </summary>
    [TestFixture]
    public class PublicBaseUrlTests
    {
        const string CredentialsAndQuery = "https://svc:APIKEY@match.invalid/base?token=QUERYSECRET";
        const string RejectionMessage =
            "MATCH_PUBLIC_BASE_URL: must not contain credentials, a query string, or a fragment";

        static ConfigurationResult Read(IDictionary<string, string?> values, string environment = "Production") =>
            HexWarsConfiguration.Read(TestConfig.Config(values), TestConfig.Env(environment));

        [TestCase(CredentialsAndQuery)]
        [TestCase("https://svc:APIKEY@match.invalid/base")]
        [TestCase("https://match.invalid/base?token=QUERYSECRET")]
        [TestCase("https://match.invalid/base#QUERYSECRET")]
        public void CredentialsQueryOrFragment_AreRejected(string raw)
        {
            var settings = TestConfig.ValidSteamSettings();
            settings["MATCH_PUBLIC_BASE_URL"] = raw;

            var result = Read(settings);

            Assert.That(result.Errors, Is.EqualTo(new[] { RejectionMessage }));
            Assert.That(result.Match.PublicBaseUrl, Is.Null);
        }

        [Test]
        public void CredentialsAreRejectedOutsideProductionToo()
        {
            var result = HexWarsConfiguration.Read(
                TestConfig.Config(new Dictionary<string, string?> { ["MATCH_PUBLIC_BASE_URL"] = CredentialsAndQuery }),
                TestConfig.Env("Development"));

            Assert.That(result.Errors, Is.EqualTo(new[] { RejectionMessage }));
        }

        [Test]
        public void CleanUrlWithAPath_IsAcceptedAndReportedVerbatim()
        {
            var settings = TestConfig.ValidSteamSettings();
            settings["MATCH_PUBLIC_BASE_URL"] = "https://match.invalid/base";

            var result = Read(settings);

            Assert.That(result.IsValid, Is.True, string.Join(" | ", result.Errors));
            var report = EnvironmentReport.Describe(result.Steam, result.Match, TestConfig.Env("Production"));
            Assert.That(report.PublicBaseUrl, Is.EqualTo("https://match.invalid/base"));
        }

        /// <summary>Defence in depth: the report must strip credentials, query and fragment itself rather
        /// than trusting that validation already refused them.</summary>
        [Test]
        public void Report_RendersSchemeAuthorityAndPathOnly()
        {
            var match = new MatchHostingOptions
            {
                BuildId = "build-42",
                PublicBaseUrl = new Uri("https://svc:APIKEY@match.invalid/base?token=QUERYSECRET#anchorsecret"),
            };

            var report = EnvironmentReport.Describe(new SteamOptions(), match, TestConfig.Env("Production"));
            string json = report.ToJson();

            Assert.That(report.PublicBaseUrl, Is.EqualTo("https://match.invalid/base"));
            Assert.That(json, Does.Not.Contain("APIKEY"));
            Assert.That(json, Does.Not.Contain("QUERYSECRET"));
            Assert.That(json, Does.Not.Contain("anchorsecret"));
        }
    }
}
