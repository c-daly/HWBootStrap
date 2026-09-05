#nullable enable
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// The protocol-v2 frames that are NOT part of the v1 message set: the AUTH handshake, its
    /// failure reply, the keepalive pair, the graceful-restart notice, and the retryable reject.
    /// These are exact wire strings: the server matches them byte for byte.
    /// </summary>
    [TestFixture]
    public sealed class SteamMatchProtocolTests
    {
        [Test]
        public void AuthFrame_IsMatchIdThenCredential()
        {
            Assert.That(SteamMatchProtocol.AuthFrame("m-1", "cred-1"), Is.EqualTo("AUTH m-1 cred-1"));
        }

        [Test]
        public void TryParseAuthFail_ReadsTheCode()
        {
            Assert.That(SteamMatchProtocol.TryParseAuthFail("AUTH FAIL expired", out var code), Is.True);
            Assert.That(code, Is.EqualTo("expired"));
        }

        [Test]
        public void TryParseAuthFail_WithoutACode_IsStillAnAuthFailure()
        {
            Assert.That(SteamMatchProtocol.TryParseAuthFail("AUTH FAIL", out var code), Is.True);
            Assert.That(code, Is.EqualTo(SteamMatchProtocol.AuthFailUnknown));
        }

        [Test]
        public void TryParseAuthFail_YieldsOnlyCodesFromTheFixedSet()
        {
            foreach (var known in new[] { "invalid", "expired", "unavailable", "protocol" })
            {
                Assert.That(SteamMatchProtocol.TryParseAuthFail("AUTH FAIL " + known, out var code), Is.True);
                Assert.That(code, Is.EqualTo(known));
            }
        }

        [Test]
        public void TryParseAuthFail_NeverPassesAnUntrustedPayloadThrough()
        {
            // The payload is whatever the far end sent. It reaches a Unity log and a player-facing
            // path, so it is mapped to a code we chose or to "unknown" - never echoed.
            var hostile = "AUTH FAIL 0A1B2C3D-join-credential\nSEAT 0";
            Assert.That(SteamMatchProtocol.TryParseAuthFail(hostile, out var code), Is.True);
            Assert.That(code, Is.EqualTo(SteamMatchProtocol.AuthFailUnknown));

            Assert.That(SteamMatchProtocol.TryParseAuthFail("AUTH FAIL Expired", out var cased), Is.True);
            Assert.That(cased, Is.EqualTo(SteamMatchProtocol.AuthFailUnknown), "the set is matched exactly");
        }

        [Test]
        public void TryParseAuthFail_IgnoresEveryOtherFrame()
        {
            Assert.That(SteamMatchProtocol.TryParseAuthFail("SEAT 0", out var seatCode), Is.False);
            Assert.That(seatCode, Is.Empty);
            Assert.That(SteamMatchProtocol.TryParseAuthFail("AUTH FAILURE odd", out _), Is.False);
            Assert.That(SteamMatchProtocol.TryParseAuthFail("", out _), Is.False);
            Assert.That(SteamMatchProtocol.TryParseAuthFail(null, out _), Is.False);
        }

        [Test]
        public void KeepaliveAndRestartFramesAreExact()
        {
            Assert.That(SteamMatchProtocol.AuthFailPrefix, Is.EqualTo("AUTH FAIL"));
            Assert.That(SteamMatchProtocol.Ping, Is.EqualTo("PING"));
            Assert.That(SteamMatchProtocol.Pong, Is.EqualTo("PONG"));
            Assert.That(SteamMatchProtocol.ServerRestart, Is.EqualTo("SERVER RESTART"));
        }

        [Test]
        public void IsTemporaryFailure_MatchesOnlyTheRetryableReject()
        {
            Assert.That(SteamMatchProtocol.IsTemporaryFailure("TemporaryFailure"), Is.True);
            Assert.That(SteamMatchProtocol.IsTemporaryFailure("NotYourTurn"), Is.False);
            Assert.That(SteamMatchProtocol.IsTemporaryFailure("temporaryfailure"), Is.False);
            Assert.That(SteamMatchProtocol.IsTemporaryFailure(""), Is.False);
            Assert.That(SteamMatchProtocol.IsTemporaryFailure(null), Is.False);
        }
    }
}
