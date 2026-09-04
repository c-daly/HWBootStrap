using System.Net;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// Every Steam Web API behaviour the match host depends on, driven entirely through
    /// <see cref="FakeSteamHandler"/>. No test here may open a socket: the base URL is a .invalid host so
    /// a bypassed fake fails loudly rather than calling Valve, and the retry delay is replaced by a
    /// recorder so the retry tests finish instantly.
    /// </summary>
    [TestFixture]
    public class SteamWebApiClientTests
    {
        const string Key = "test-key";
        const uint AppId = 480000u;
        const string BaseUrl = "https://partner.steam-api.invalid";
        const string Ticket = "0a1b2c3d";
        const string OwnerId = "76561197960287930";
        const string MemberId = "76561197985812219";
        const string LobbyId = "109775241010407638";

        // ---- JSON bodies ---------------------------------------------------

        const string AuthOk =
            """{"response":{"params":{"result":"OK","steamid":"76561197960287930","ownersteamid":"76561197985812219","vacbanned":false,"publisherbanned":false}}}""";

        // ownersteamid absent: the client must default it to steamid. Ids arrive as JSON numbers here,
        // which Valve does for some fields, and with leading whitespace that must be trimmed away.
        const string AuthOkNoOwner =
            """{"response":{"params":{"result":"OK","steamid":76561197960287930,"vacbanned":true,"publisherbanned":false}}}""";

        const string AuthError =
            """{"response":{"error":{"errorcode":101,"errordesc":"Invalid ticket"}}}""";

        const string OwnershipTrue =
            """{"appownership":{"ownsapp":true,"permanent":true,"timestamp":"2024-01-01T00:00:00Z","ownersteamid":"76561197960287930","sitelicense":false,"result":"OK"}}""";

        const string OwnershipFalse =
            """{"appownership":{"ownsapp":false,"permanent":false,"ownersteamid":"76561197960287930","result":"OK"}}""";

        const string OwnershipMissing = """{"result":{"status":1}}""";

        // The shape Valve documents for ILobbyMatchmakingService/GetLobbyData/v1: member_metadata rather
        // than member_data, key_name/key_value rather than key/value, and no steamid_lobby echoed back.
        const string LobbyValveShape =
            """{"response":{"appid":480000,"lobby_type":2,"steamid_owner":"76561197960287930","lobby_metadata":[{"key_name":"hw_app","key_value":"480000"},{"key_name":"hw_protocol","key_value":"2"}],"members":[{"steamid":"76561197960287930","member_metadata":[{"key_name":"hw_ready","key_value":"1"}]},{"steamid":"76561197985812219","member_metadata":[{"key_name":"hw_ready","key_value":"0"}]}]}}""";

        // The alternate shape: numeric ids, member_data as a JSON object, lobby_metadata as key/value.
        const string LobbyAlternateShape =
            """{"response":{"appid":480000,"steamid_lobby":109775241010407638,"lobby_type":2,"lobby_flags":0,"max_members":2,"steamid_owner":76561197960287930,"members":[{"steamid":76561197985812219,"member_data":{"hw_ready":"1"}}],"lobby_metadata":[{"key":"hw_app","value":"480000"},{"key":"hw_setup","value":"Annihilation 9 7 0 1234 3 1 1 1 3 0"}]}}""";

        const string LobbyEmpty = """{"response":{}}""";

        // ---- harness -------------------------------------------------------

        sealed class RecordingLogger<T> : ILogger<T>
        {
            public List<(LogLevel Level, string Message)> Entries { get; } = new();

            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter) =>
                Entries.Add((logLevel, formatter(state, exception)));
        }

        sealed class Harness : IDisposable
        {
            public FakeSteamHandler Handler { get; } = new();
            public RecordingLogger<SteamWebApiClient> Logger { get; } = new();
            public List<TimeSpan> Delays { get; } = new();
            public SteamWebApiClient Client { get; }

            readonly HttpClient _http;

            public Harness(int timeoutSeconds = 5)
            {
                _http = new HttpClient(Handler)
                {
                    BaseAddress = new Uri(BaseUrl + "/"),
                    Timeout = TimeSpan.FromSeconds(timeoutSeconds),
                };
                var options = Options.Create(new SteamOptions
                {
                    AppId = AppId,
                    PublisherWebApiKey = Key,
                    WebApiBaseUrl = new Uri(BaseUrl),
                    RequestTimeoutSeconds = timeoutSeconds,
                });
                Client = new SteamWebApiClient(_http, options, Logger)
                {
                    // Retries must not cost the suite real seconds; record what would have been slept.
                    DelayAsync = (delay, _) => { Delays.Add(delay); return Task.CompletedTask; },
                };
            }

            public void Dispose() => _http.Dispose();
        }

        static string QueryOf(Uri uri) => uri.Query;

        // ---- AuthenticateUserTicket ---------------------------------------

        [Test]
        public async Task Auth_HappyPath_NormalisesIdentity()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);

            var identity = await h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None);

            Assert.That(identity.SteamId, Is.EqualTo(OwnerId));
            Assert.That(identity.OwnerSteamId, Is.EqualTo(MemberId));
            Assert.That(identity.VacBanned, Is.False);
            Assert.That(identity.PublisherBanned, Is.False);
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(1));
        }

        [Test]
        public async Task Auth_WithoutOwnerSteamId_DefaultsOwnerToSteamId()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOkNoOwner);

            var identity = await h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None);

            Assert.That(identity.SteamId, Is.EqualTo(OwnerId));
            Assert.That(identity.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(identity.VacBanned, Is.True);
        }

        [Test]
        public void Auth_ResponseError_ThrowsAuthenticationFailed()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthError);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.AuthenticationFailed));
            Assert.That(ex.PlayerSafeMessage, Is.EqualTo("Steam sign-in could not be verified."));
            Assert.That(ex.Message, Does.StartWith("AuthenticationFailed: "));
            Assert.That(ex.Message, Does.Contain("101"));
        }

        [TestCase("nothex!!", TestName = "Auth_NonHexTicket_NeverReachesTheNetwork")]
        [TestCase("0a1b2", TestName = "Auth_OddLengthTicket_NeverReachesTheNetwork")]
        [TestCase("", TestName = "Auth_EmptyTicket_NeverReachesTheNetwork")]
        public void Auth_InvalidTicket_ThrowsWithoutAnyRequest(string ticket)
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.AuthenticationFailed));
            Assert.That(h.Handler.Requests, Is.Empty, "a malformed ticket must never reach Valve");
        }

        [Test]
        public void Auth_OversizedTicket_ThrowsWithoutAnyRequest()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);
            var oversized = new string((char)97, 5000);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(oversized, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.AuthenticationFailed));
            Assert.That(h.Handler.Requests, Is.Empty);
        }

        [Test]
        public async Task Auth_RequestUri_CarriesKeyAppIdTicketAndIdentity()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);

            await h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None);

            var query = QueryOf(h.Handler.Requests[0]);
            Assert.That(h.Handler.Requests[0].AbsolutePath, Is.EqualTo(FakeSteamHandler.AuthPath));
            Assert.That(query, Does.Contain("identity=hexwars-match"));
            Assert.That(query, Does.Contain("appid=480000"));
            Assert.That(query, Does.Contain("key=" + Key));
            Assert.That(query, Does.Contain("ticket=" + Ticket));
        }

        [Test]
        public void Auth_ServiceFailure_ExceptionLeaksNeitherKeyNorTicket()
        {
            using var h = new Harness();
            h.Handler.RespondStatus(FakeSteamHandler.AuthPath, HttpStatusCode.ServiceUnavailable);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.ServiceUnavailable));
            Assert.That(ex.ToString(), Does.Not.Contain(Key));
            Assert.That(ex.ToString(), Does.Not.Contain(Ticket));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(1), "ticket auth must never be retried");
        }

        [Test]
        public void Auth_Unauthorized_MapsToServiceUnavailableAndLogsAnOperatorError()
        {
            using var h = new Harness();
            h.Handler.RespondStatus(FakeSteamHandler.AuthPath, HttpStatusCode.Unauthorized);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.ServiceUnavailable));
            Assert.That(ex.PlayerSafeMessage,
                Is.EqualTo("The match service is temporarily unavailable \u2014 try again shortly."));
            Assert.That(
                h.Logger.Entries.Any(e => e.Level == LogLevel.Error && e.Message.Contains("Steam publisher key rejected")),
                Is.True, "a rejected publisher key is an operator problem and must be logged as an error");
        }

        [Test]
        public void Auth_RateLimited_MapsToRateLimitedWithoutRetrying()
        {
            using var h = new Harness();
            h.Handler.RespondStatus(FakeSteamHandler.AuthPath, HttpStatusCode.TooManyRequests);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.RateLimited));
            Assert.That(ex.PlayerSafeMessage, Is.EqualTo("Too many attempts \u2014 wait a moment and try again."));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(1));
            Assert.That(h.Delays, Is.Empty);
        }

        [Test]
        public void Auth_MalformedBody_MapsToMalformedResponse()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, "not json at all");

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.MalformedResponse));
            Assert.That(ex.PlayerSafeMessage,
                Is.EqualTo("The match service is temporarily unavailable \u2014 try again shortly."));
        }

        // ---- CheckAppOwnership ---------------------------------------------

        [Test]
        public async Task Ownership_OwnsApp_IsTrue()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.OwnershipPath, OwnershipTrue);

            Assert.That(await h.Client.CheckAppOwnershipAsync(OwnerId, CancellationToken.None), Is.True);

            var query = QueryOf(h.Handler.Requests[0]);
            Assert.That(h.Handler.Requests[0].AbsolutePath, Is.EqualTo(FakeSteamHandler.OwnershipPath));
            Assert.That(query, Does.Contain("steamid=" + OwnerId));
            Assert.That(query, Does.Contain("appid=480000"));
            Assert.That(query, Does.Contain("key=" + Key));
        }

        [Test]
        public async Task Ownership_DoesNotOwnApp_IsFalse()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.OwnershipPath, OwnershipFalse);

            Assert.That(await h.Client.CheckAppOwnershipAsync(OwnerId, CancellationToken.None), Is.False);
        }

        [Test]
        public void Ownership_MissingAppOwnership_MapsToMalformedResponse()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.OwnershipPath, OwnershipMissing);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.CheckAppOwnershipAsync(OwnerId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.MalformedResponse));
        }

        [Test]
        public void Ownership_NonSteamId_ThrowsWithoutAnyRequest()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.OwnershipPath, OwnershipTrue);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.CheckAppOwnershipAsync("not-a-steam-id", CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.AuthenticationFailed));
            Assert.That(h.Handler.Requests, Is.Empty);
        }

        [Test]
        public async Task Ownership_TransientFailureThenSuccess_IsRetried()
        {
            using var h = new Harness();
            h.Handler
                .RespondStatus(FakeSteamHandler.OwnershipPath, HttpStatusCode.BadGateway)
                .RespondJson(FakeSteamHandler.OwnershipPath, OwnershipTrue);

            Assert.That(await h.Client.CheckAppOwnershipAsync(OwnerId, CancellationToken.None), Is.True);
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(2));
            Assert.That(h.Delays, Has.Count.EqualTo(1));
        }

        // ---- GetLobbyData ---------------------------------------------------

        [Test]
        public async Task Lobby_ValveDocumentedShape_Parses()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.LobbyPath, LobbyValveShape);

            var snapshot = await h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None);

            // Valve does not echo steamid_lobby back, so the requested id has to survive the round trip.
            Assert.That(snapshot.LobbyId, Is.EqualTo(LobbyId));
            Assert.That(snapshot.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(snapshot.Members.Select(m => m.SteamId), Is.EqualTo(new[] { OwnerId, MemberId }));
            Assert.That(snapshot.Members[0].Data["hw_ready"], Is.EqualTo("1"));
            Assert.That(snapshot.Members[1].Data["hw_ready"], Is.EqualTo("0"));
            Assert.That(snapshot.Metadata["hw_app"], Is.EqualTo("480000"));
            Assert.That(snapshot.Metadata["hw_protocol"], Is.EqualTo("2"));

            var query = QueryOf(h.Handler.Requests[0]);
            Assert.That(h.Handler.Requests[0].AbsolutePath, Is.EqualTo(FakeSteamHandler.LobbyPath));
            Assert.That(query, Does.Contain("steamid_lobby=" + LobbyId));
            Assert.That(query, Does.Contain("appid=480000"));
        }

        [Test]
        public async Task Lobby_NumericIdsAndObjectShapedMemberData_Parse()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.LobbyPath, LobbyAlternateShape);

            var snapshot = await h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None);

            Assert.That(snapshot.LobbyId, Is.EqualTo(LobbyId));
            Assert.That(snapshot.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(snapshot.Members, Has.Count.EqualTo(1));
            Assert.That(snapshot.Members[0].SteamId, Is.EqualTo(MemberId));
            Assert.That(snapshot.Members[0].Data["hw_ready"], Is.EqualTo("1"));
            Assert.That(snapshot.Metadata["hw_app"], Is.EqualTo("480000"));
            Assert.That(snapshot.Metadata["hw_setup"], Is.EqualTo("Annihilation 9 7 0 1234 3 1 1 1 3 0"));
        }

        [Test]
        public async Task Lobby_TransientFailures_AreRetriedWithBoundedJitteredBackoff()
        {
            using var h = new Harness();
            h.Handler
                .RespondStatus(FakeSteamHandler.LobbyPath, HttpStatusCode.ServiceUnavailable)
                .RespondStatus(FakeSteamHandler.LobbyPath, HttpStatusCode.ServiceUnavailable)
                .RespondJson(FakeSteamHandler.LobbyPath, LobbyValveShape);

            var snapshot = await h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None);

            Assert.That(snapshot.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(3));
            Assert.That(h.Delays, Has.Count.EqualTo(2));
            Assert.That(h.Delays[0], Is.InRange(TimeSpan.FromMilliseconds(200), TimeSpan.FromMilliseconds(300)));
            Assert.That(h.Delays[1], Is.InRange(TimeSpan.FromMilliseconds(400), TimeSpan.FromMilliseconds(500)));
        }

        [Test]
        public void Lobby_AllAttemptsFail_MapsToServiceUnavailableAfterThreeRequests()
        {
            using var h = new Harness();
            h.Handler.RespondStatus(FakeSteamHandler.LobbyPath, HttpStatusCode.ServiceUnavailable);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.ServiceUnavailable));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(3), "one attempt plus two retries");
            Assert.That(ex.ToString(), Does.Not.Contain(Key));
        }

        [Test]
        public async Task Lobby_RateLimited_HonoursRetryAfterAndRetries()
        {
            using var h = new Harness();
            h.Handler
                .RespondRetryAfter(FakeSteamHandler.LobbyPath, HttpStatusCode.TooManyRequests, 1)
                .RespondJson(FakeSteamHandler.LobbyPath, LobbyValveShape);

            var snapshot = await h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None);

            Assert.That(snapshot.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(2));
            Assert.That(h.Delays, Has.Count.EqualTo(1));
            Assert.That(h.Delays[0], Is.EqualTo(TimeSpan.FromSeconds(1)));
        }

        [Test]
        public void Lobby_RateLimitedThroughout_MapsToRateLimited()
        {
            using var h = new Harness();
            h.Handler.RespondRetryAfter(FakeSteamHandler.LobbyPath, HttpStatusCode.TooManyRequests, 30);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.RateLimited));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(3));
            // Retry-After is capped so a hostile header cannot stall a request thread for 30 seconds.
            Assert.That(h.Delays, Is.All.EqualTo(TimeSpan.FromSeconds(2)));
        }

        [Test]
        public void Lobby_NotFound_MapsToLobbyChangedWithoutRetrying()
        {
            using var h = new Harness();
            h.Handler.RespondStatus(FakeSteamHandler.LobbyPath, HttpStatusCode.NotFound);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.LobbyChanged));
            Assert.That(ex.PlayerSafeMessage,
                Is.EqualTo("The lobby changed \u2014 check that both players are ready and try again."));
            Assert.That(h.Handler.Requests, Has.Count.EqualTo(1));
        }

        [Test]
        public void Lobby_EmptyResponse_MapsToLobbyChanged()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.LobbyPath, LobbyEmpty);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.LobbyChanged));
        }

        [Test]
        public void Lobby_MalformedJson_MapsToMalformedResponse()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.LobbyPath, "<html>gateway</html>");

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync(LobbyId, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.MalformedResponse));
        }

        [Test]
        public void Lobby_NonNumericLobbyId_ThrowsWithoutAnyRequest()
        {
            using var h = new Harness();
            h.Handler.RespondJson(FakeSteamHandler.LobbyPath, LobbyValveShape);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.GetLobbyDataAsync("lobby-one", CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.LobbyChanged));
            Assert.That(h.Handler.Requests, Is.Empty);
        }

        // ---- transport ------------------------------------------------------

        [Test]
        public void Timeout_MapsToServiceUnavailable()
        {
            using var h = new Harness(timeoutSeconds: 1);
            h.Handler.Delay = TimeSpan.FromSeconds(2);
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.ServiceUnavailable));
            Assert.That(ex.ToString(), Does.Not.Contain(Key));
        }

        [Test]
        public void CallerCancellation_IsNotSwallowed()
        {
            using var h = new Harness();
            h.Handler.Delay = TimeSpan.FromSeconds(10);
            h.Handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);
            using var cts = new CancellationTokenSource();
            cts.Cancel();

            Assert.ThrowsAsync<TaskCanceledException>(
                () => h.Client.AuthenticateUserTicketAsync(Ticket, cts.Token));
        }

        // ---- registration ----------------------------------------------------

        [Test]
        public void AddSteamWebApi_RegistersTheTypedClient()
        {
            var services = new ServiceCollection();
            services.AddOptions<SteamOptions>().Configure(o =>
            {
                o.AppId = AppId;
                o.PublisherWebApiKey = Key;
                o.WebApiBaseUrl = new Uri(BaseUrl);
                o.RequestTimeoutSeconds = 3;
            });
            services.AddSteamWebApi();

            using var provider = services.BuildServiceProvider();
            var client = provider.GetRequiredService<ISteamWebApiClient>();

            Assert.That(client, Is.InstanceOf<SteamWebApiClient>());
        }
    }

    /// <summary>
    /// The two pure helpers the rest of the Steam surface leans on: SteamID64 normalisation (every id in
    /// the system is a canonical decimal string, never a long) and log redaction (a publisher key or an
    /// auth ticket must not survive a trip through a log line).
    /// </summary>
    [TestFixture]
    public class SteamIdAndRedactionTests
    {
        [TestCase("76561197960287930", ExpectedResult = "76561197960287930")]
        [TestCase("  76561197960287930  ", ExpectedResult = "76561197960287930")]
        [TestCase("109775241010407638", ExpectedResult = "109775241010407638")]
        [TestCase("076561197960287930", ExpectedResult = "76561197960287930")]
        public string TryNormalize_AcceptsSteamId64(string raw)
        {
            Assert.That(SteamId64.TryNormalize(raw, out var canonical), Is.True);
            Assert.That(SteamId64.IsValid(raw), Is.True);
            return canonical;
        }

        [TestCase("7656119796028793", TestName = "TryNormalize_RejectsBelowTheIndividualBase")]
        [TestCase("abc", TestName = "TryNormalize_RejectsLetters")]
        [TestCase("", TestName = "TryNormalize_RejectsEmpty")]
        [TestCase(null, TestName = "TryNormalize_RejectsNull")]
        [TestCase("99999999999999999999999", TestName = "TryNormalize_RejectsOverflow")]
        [TestCase("7656119796028 7930", TestName = "TryNormalize_RejectsInnerWhitespace")]
        [TestCase("-76561197960287930", TestName = "TryNormalize_RejectsNegative")]
        [TestCase("7.6561197960287930e16", TestName = "TryNormalize_RejectsScientificNotation")]
        public void TryNormalize_RejectsAnythingElse(string? raw)
        {
            Assert.That(SteamId64.TryNormalize(raw, out var canonical), Is.False);
            Assert.That(canonical, Is.Empty);
            Assert.That(SteamId64.IsValid(raw), Is.False);
        }

        [Test]
        public void Redact_MasksKeyAndTicketAndTheWholeQueryString()
        {
            const string raw =
                "GET https://partner.steam-api.com/ISteamUserAuth/AuthenticateUserTicket/v1/?key=abc123&ticket=deadbeef failed";

            var redacted = SteamLogRedaction.Redact(raw);

            Assert.That(redacted, Does.Not.Contain("abc123"));
            Assert.That(redacted, Does.Not.Contain("deadbeef"));
            Assert.That(redacted, Does.Contain("AuthenticateUserTicket"));
            Assert.That(redacted, Does.Contain("failed"), "the surrounding message must survive");
        }

        [Test]
        public void Redact_MasksSecretParametersEvenWithoutAQueryString()
        {
            Assert.That(SteamLogRedaction.Redact("key=abc123"), Is.EqualTo("key=<redacted>"));
            Assert.That(SteamLogRedaction.Redact("ticket=DEADBEEF"), Is.EqualTo("ticket=<redacted>"));
        }

        [Test]
        public void Redact_LeavesOrdinaryTextAlone()
        {
            Assert.That(SteamLogRedaction.Redact("connection refused"), Is.EqualTo("connection refused"));
            Assert.That(SteamLogRedaction.Redact(string.Empty), Is.Empty);
        }

        [Test]
        public void HashSteamId_IsStableTwelveCharactersAndHidesTheId()
        {
            const string steamId = "76561197960287930";

            var hashed = SteamLogRedaction.HashSteamId(steamId);

            Assert.That(hashed, Has.Length.EqualTo(12));
            Assert.That(hashed, Does.StartWith("sid:"));
            Assert.That(hashed, Does.Not.Contain(steamId));
            Assert.That(SteamLogRedaction.HashSteamId(steamId), Is.EqualTo(hashed));
            Assert.That(SteamLogRedaction.HashSteamId("76561197960287931"), Is.Not.EqualTo(hashed));
        }

        [Test]
        public void FailureMessages_AreTheExactPlayerSafeStrings()
        {
            Assert.Multiple(() =>
            {
                Assert.That(SteamFailureMessages.For(SteamFailure.AuthenticationFailed),
                    Is.EqualTo("Steam sign-in could not be verified."));
                Assert.That(SteamFailureMessages.For(SteamFailure.OwnershipMissing),
                    Is.EqualTo("This Steam account does not own HexWars."));
                Assert.That(SteamFailureMessages.For(SteamFailure.NotLobbyMember),
                    Is.EqualTo("You are not a member of that lobby."));
                Assert.That(SteamFailureMessages.For(SteamFailure.NotLobbyOwner),
                    Is.EqualTo("Only the lobby owner can start the match."));
                Assert.That(SteamFailureMessages.For(SteamFailure.LobbyChanged),
                    Is.EqualTo("The lobby changed \u2014 check that both players are ready and try again."));
                Assert.That(SteamFailureMessages.For(SteamFailure.IncompatibleVersion),
                    Is.EqualTo("Your game version is not compatible with this server."));
                Assert.That(SteamFailureMessages.For(SteamFailure.ServiceUnavailable),
                    Is.EqualTo("The match service is temporarily unavailable \u2014 try again shortly."));
                Assert.That(SteamFailureMessages.For(SteamFailure.RateLimited),
                    Is.EqualTo("Too many attempts \u2014 wait a moment and try again."));
                // A malformed body is our problem, not the player problem: it reads as a service outage.
                Assert.That(SteamFailureMessages.For(SteamFailure.MalformedResponse),
                    Is.EqualTo("The match service is temporarily unavailable \u2014 try again shortly."));
            });
        }

        [Test]
        public void SteamApiException_CarriesFailureDetailAndPlayerSafeMessage()
        {
            var ex = new SteamApiException(SteamFailure.OwnershipMissing, "ownsapp was false");

            Assert.That(ex.Failure, Is.EqualTo(SteamFailure.OwnershipMissing));
            Assert.That(ex.Detail, Is.EqualTo("ownsapp was false"));
            Assert.That(ex.Message, Is.EqualTo("OwnershipMissing: ownsapp was false"));
            Assert.That(ex.PlayerSafeMessage, Is.EqualTo("This Steam account does not own HexWars."));
        }
    }
}
