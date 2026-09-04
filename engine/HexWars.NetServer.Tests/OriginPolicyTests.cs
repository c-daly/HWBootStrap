using HexWars.NetServer.Hosting;
using Microsoft.AspNetCore.Http;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The Origin rule on its own: same-origin is always allowed, ALLOWED_WEB_ORIGINS widens it, and
    /// anything else is refused. Absent and unparseable Origins keep the pre-existing pass-through so
    /// non-browser clients and the in-process selftest are unaffected.
    /// </summary>
    [TestFixture]
    public class OriginPolicyTests
    {
        static HttpContext Request(string host, string? origin)
        {
            var context = new DefaultHttpContext();
            context.Request.Host = new HostString(host);
            if (origin is not null) context.Request.Headers["Origin"] = origin;
            return context;
        }

        static readonly string[] None = Array.Empty<string>();

        [Test]
        public void NoOriginHeader_IsAllowed() =>
            Assert.That(OriginPolicy.IsAllowed(Request("game.invalid", origin: null), None), Is.True);

        [Test]
        public void UnparseableOrigin_IsAllowed() =>
            Assert.That(OriginPolicy.IsAllowed(Request("game.invalid", "null"), None), Is.True);

        [TestCase("game.invalid", "https://game.invalid")]
        [TestCase("game.invalid", "http://GAME.invalid")]
        [TestCase("game.invalid:8080", "http://game.invalid:8080")]
        public void OriginMatchingTheRequestHost_IsAllowed(string host, string origin) =>
            Assert.That(OriginPolicy.IsAllowed(Request(host, origin), None), Is.True);

        [Test]
        public void CrossSiteOriginWithNothingConfigured_IsRefused() =>
            Assert.That(OriginPolicy.IsAllowed(Request("game.invalid", "https://evil.invalid"), None), Is.False);

        [TestCase("https://portal.invalid")]
        [TestCase("https://portal.invalid/")]
        [TestCase("  https://PORTAL.invalid  ")]
        [TestCase("https://portal.invalid:443")]
        public void ConfiguredOrigin_IsAllowed(string configured) =>
            Assert.That(
                OriginPolicy.IsAllowed(Request("game.invalid", "https://portal.invalid"), new[] { configured }),
                Is.True);

        [Test]
        public void ConfiguredOriginOnAnotherScheme_DoesNotMatch() =>
            Assert.That(
                OriginPolicy.IsAllowed(Request("game.invalid", "http://portal.invalid"),
                    new[] { "https://portal.invalid" }),
                Is.False);

        [Test]
        public void ConfiguredOriginOnAnotherPort_DoesNotMatch() =>
            Assert.That(
                OriginPolicy.IsAllowed(Request("game.invalid", "https://portal.invalid:8443"),
                    new[] { "https://portal.invalid" }),
                Is.False);

        [Test]
        public void OneMatchAmongSeveralConfiguredEntries_IsEnough() =>
            Assert.That(
                OriginPolicy.IsAllowed(Request("game.invalid", "https://b.invalid"),
                    new[] { "https://a.invalid", "not-a-url", "https://b.invalid" }),
                Is.True);
    }
}
