#nullable enable
using Newtonsoft.Json.Linq;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// The pure half of the match-service client: the URLs it posts to, the exact JSON field names the
    /// server reads, how an HTTP status plus body becomes a <see cref="SteamMatchApiResult"/>, and how
    /// the shipped config asset decides whether Steam play is configured at all.
    /// </summary>
    [TestFixture]
    public sealed class SteamMatchApiContractsTests
    {
        const string BaseUrl = "https://match.hexwars.test";

        [Test]
        public void CreateMatchUrl_IsTheAllocationEndpoint()
        {
            Assert.That(SteamMatchApiContracts.CreateMatchUrl(BaseUrl),
                Is.EqualTo("https://match.hexwars.test/api/v1/steam/matches"));
        }

        [Test]
        public void CreateMatchUrl_IgnoresATrailingSlashOnTheBase()
        {
            Assert.That(SteamMatchApiContracts.CreateMatchUrl("https://match.hexwars.test/"),
                Is.EqualTo("https://match.hexwars.test/api/v1/steam/matches"));
        }

        [Test]
        public void JoinMatchUrl_AddsTheMatchIdAndJoin()
        {
            Assert.That(SteamMatchApiContracts.JoinMatchUrl(BaseUrl, "0b9c4f2a"),
                Is.EqualTo("https://match.hexwars.test/api/v1/steam/matches/0b9c4f2a/join"));
        }

        [Test]
        public void JoinMatchUrl_EscapesTheMatchId()
        {
            Assert.That(SteamMatchApiContracts.JoinMatchUrl(BaseUrl, "a b/c"),
                Is.EqualTo("https://match.hexwars.test/api/v1/steam/matches/a%20b%2Fc/join"));
        }

        [Test]
        public void CreateMatchBody_UsesTheServerFieldNames()
        {
            var body = JObject.Parse(SteamMatchApiContracts.CreateMatchBody(
                "109775240", "AABBCC", "Annihilation 9 7 0 42 3 1 1 1 3 0"));

            Assert.That((string?)body["steamLobbyId"], Is.EqualTo("109775240"));
            Assert.That((string?)body["ticket"], Is.EqualTo("AABBCC"));
            Assert.That((string?)body["requestedSetup"], Is.EqualTo("Annihilation 9 7 0 42 3 1 1 1 3 0"));
            Assert.That(body.Count, Is.EqualTo(3), "the create body carries exactly these three fields");
        }

        [Test]
        public void JoinMatchBody_CarriesOnlyTheTicket()
        {
            var body = JObject.Parse(SteamMatchApiContracts.JoinMatchBody("AABBCC"));

            Assert.That((string?)body["ticket"], Is.EqualTo("AABBCC"));
            Assert.That(body.Count, Is.EqualTo(1));
        }

        [Test]
        public void Parse_SuccessCarriesEverythingTheSocketNeeds()
        {
            var result = SteamMatchApiContracts.Parse(200,
                @"{""matchId"":""m-1"",""protocolVersion"":2,""websocketUrl"":""wss://match.hexwars.test/ws/v2"","
                + @"""seat"":1,""joinCredential"":""cred-1"",""credentialExpiresAt"":""2026-09-04T12:00:00Z""}", 2);

            Assert.That(result.Ok, Is.True);
            Assert.That(result.HttpStatus, Is.EqualTo(200));
            Assert.That(result.MatchId, Is.EqualTo("m-1"));
            Assert.That(result.WebsocketUrl, Is.EqualTo("wss://match.hexwars.test/ws/v2"));
            Assert.That(result.JoinCredential, Is.EqualTo("cred-1"));
            Assert.That(result.Seat, Is.EqualTo(1));
            Assert.That(result.ErrorCode, Is.Null);
        }

        [Test]
        public void Parse_SuccessWithoutACredentialIsMalformed()
        {
            var result = SteamMatchApiContracts.Parse(200,
                @"{""matchId"":""m-1"",""websocketUrl"":""wss://match.hexwars.test/ws/v2"",""seat"":0}", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.HttpStatus, Is.EqualTo(200));
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchApiContracts.MalformedErrorCode));
        }

        [Test]
        public void Parse_AuthenticationFailureKeepsTheCodeAndPlayerMessage()
        {
            var result = SteamMatchApiContracts.Parse(401,
                @"{""error"":""authentication_failed"",""message"":""Steam sign-in could not be verified.""}", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.HttpStatus, Is.EqualTo(401));
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchErrorCodes.AuthenticationFailed));
            Assert.That(result.Message, Is.EqualTo("Steam sign-in could not be verified."));
            Assert.That(result.MatchId, Is.Null);
            Assert.That(result.JoinCredential, Is.Null);
        }

        [Test]
        public void Parse_ServiceUnavailableKeepsItsCode()
        {
            var result = SteamMatchApiContracts.Parse(503,
                @"{""error"":""service_unavailable"",""message"":""The match service is temporarily unavailable.""}", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.HttpStatus, Is.EqualTo(503));
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchErrorCodes.ServiceUnavailable));
            Assert.That(result.Message, Is.EqualTo("The match service is temporarily unavailable."));
        }

        [Test]
        public void Parse_UnreadableErrorBodyIsMalformed()
        {
            var result = SteamMatchApiContracts.Parse(500, "<html>bad gateway</html>", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.HttpStatus, Is.EqualTo(500));
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchApiContracts.MalformedErrorCode));
            Assert.That(result.Message, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public void Parse_ErrorBodyWithoutACodeIsMalformed()
        {
            var result = SteamMatchApiContracts.Parse(409, @"{""message"":""no code here""}", 2);

            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchApiContracts.MalformedErrorCode));
        }

        [Test]
        public void Parse_StatusZeroIsANetworkFailure()
        {
            var result = SteamMatchApiContracts.Parse(0, "", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.HttpStatus, Is.EqualTo(0));
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchApiContracts.NetworkErrorCode));
            Assert.That(result.Message, Is.Not.Null.And.Not.Empty);
        }

        [Test]
        public void ParseJson_ShippedPlaceholderMeansSteamPlayIsNotConfigured()
        {
            var settings = SteamMatchConfig.ParseJson(@"{""matchBaseUrl"":""OWNER-INPUT"",""protocolVersion"":2}");

            Assert.That(settings.IsConfigured, Is.False);
            Assert.That(settings.BaseUrl, Is.Empty);
        }

        [Test]
        public void ParseJson_AnAbsoluteBaseUrlIsConfigured()
        {
            var settings = SteamMatchConfig.ParseJson(
                @"{""matchBaseUrl"":""https://match.hexwars.test/"",""protocolVersion"":2}");

            Assert.That(settings.IsConfigured, Is.True);
            Assert.That(settings.BaseUrl, Is.EqualTo("https://match.hexwars.test"));
            Assert.That(settings.ProtocolVersion, Is.EqualTo(2));
        }

        [Test]
        public void ParseJson_MissingProtocolVersionFallsBackToTwo()
        {
            var settings = SteamMatchConfig.ParseJson(@"{""matchBaseUrl"":""https://match.hexwars.test""}");

            Assert.That(settings.IsConfigured, Is.True);
            Assert.That(settings.ProtocolVersion, Is.EqualTo(SteamMatchConfig.DefaultProtocolVersion));
        }

        [Test]
        public void ParseJson_RelativeEmptyOrUnreadableIsNotConfigured()
        {
            Assert.That(SteamMatchConfig.ParseJson(@"{""matchBaseUrl"":""/api/v1""}").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.ParseJson(@"{""matchBaseUrl"":""""}").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.ParseJson("{}").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.ParseJson("not json at all").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.ParseJson(null).IsConfigured, Is.False);
        }

        [Test]
        public void FromBaseUrl_TakesHttpAndHttpsOnly()
        {
            Assert.That(SteamMatchConfig.FromBaseUrl("ftp://match.hexwars.test").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.FromBaseUrl("match.hexwars.test").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.FromBaseUrl("  https://match.hexwars.test/  ").BaseUrl,
                Is.EqualTo("https://match.hexwars.test"));
            Assert.That(SteamMatchConfig.FromBaseUrl("http://127.0.0.1:5234").IsConfigured, Is.True);
        }

        [Test]
        public void Parse_AReplyOnAnotherProtocolVersion_IsAVersionMismatch()
        {
            var mismatch = SteamMatchApiContracts.Parse(200,
                @"{""matchId"":""m-1"",""protocolVersion"":3,""websocketUrl"":""wss://match.hexwars.test/ws/v2"","
                + @"""seat"":0,""joinCredential"":""cred-1""}", 2);

            Assert.That(mismatch.Ok, Is.False);
            Assert.That(mismatch.ErrorCode, Is.EqualTo(SteamMatchErrorCodes.IncompatibleVersion));

            var missing = SteamMatchApiContracts.Parse(200,
                @"{""matchId"":""m-1"",""websocketUrl"":""wss://match.hexwars.test/ws/v2"","
                + @"""seat"":0,""joinCredential"":""cred-1""}", 2);

            Assert.That(missing.ErrorCode, Is.EqualTo(SteamMatchErrorCodes.IncompatibleVersion),
                "an absent protocolVersion reads as 0, which is not a version this build speaks");
        }

        [Test]
        public void Parse_AnUnencryptedSocketUrl_IsRefused()
        {
            var result = SteamMatchApiContracts.Parse(200,
                @"{""matchId"":""m-1"",""protocolVersion"":2,""websocketUrl"":""ws://match.hexwars.test/ws/v2"","
                + @"""seat"":0,""joinCredential"":""cred-1""}", 2);

            Assert.That(result.Ok, Is.False);
            Assert.That(result.ErrorCode, Is.EqualTo(SteamMatchApiContracts.InsecureTransportErrorCode));
            Assert.That(result.JoinCredential, Is.Null, "a credential must never ride a plaintext socket");
        }

        [Test]
        public void Parse_ALoopbackDevelopmentSocket_IsStillAccepted()
        {
            foreach (var host in new[] { "127.0.0.1:5234", "localhost:5234", "[::1]:5234" })
            {
                var result = SteamMatchApiContracts.Parse(200,
                    @"{""matchId"":""m-1"",""protocolVersion"":2,""websocketUrl"":""ws://" + host + @"/ws/v2"","
                    + @"""seat"":0,""joinCredential"":""cred-1""}", 2);

                Assert.That(result.Ok, Is.True, host);
            }
        }

        [Test]
        public void FromBaseUrl_RequiresHttpsAwayFromLoopback()
        {
            Assert.That(SteamMatchConfig.FromBaseUrl("http://match.hexwars.test").IsConfigured, Is.False);
            Assert.That(SteamMatchConfig.FromBaseUrl("https://match.hexwars.test").IsConfigured, Is.True);
            Assert.That(SteamMatchConfig.FromBaseUrl("http://localhost:5234").IsConfigured, Is.True);
            Assert.That(SteamMatchConfig.FromBaseUrl("http://[::1]:5234").IsConfigured, Is.True);
        }
    }
}
