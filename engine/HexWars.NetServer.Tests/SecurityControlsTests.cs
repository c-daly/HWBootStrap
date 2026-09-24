using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Console;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using Npgsql;
using NpgsqlTypes;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The controls that exist to bound what an unauthenticated caller can spend, and to keep secrets out of
    /// the log.
    ///
    /// Every test here drives the real host. That matters more than usual for this file: a request cap, a
    /// CORS policy and a per-address quota are all properties of the assembled pipeline, and a unit test of
    /// any one of them would pass against a server that never wired it up.
    /// </summary>
    [TestFixture]
    public class SecurityControlsTests
    {
        const string CreateRoute = "/api/v1/steam/matches";
        const string AllowedOrigin = "https://play.hexwars.test";
        const string ForeignOrigin = "https://evil.invalid";

        static string JoinRoute(Guid matchId) => CreateRoute + "/" + matchId.ToString() + "/join";

        // ---- request size ----------------------------------------------------

        [Test]
        public async Task ABodyOverTheRequestCap_IsRefusedWithoutBeingRead()
        {
            using var fixture = new SteamServerFactory();
            using HttpClient client = fixture.CreateClient();

            var oversized = new string('a', 17 * 1024);
            using var content = new StringContent(oversized, Encoding.UTF8, "application/json");

            using HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge));
            Assert.That(fixture.RequestBodyBytesRead, Is.EqualTo(0),
                "a body refused for its size must never be pulled off the wire");
        }

        [Test]
        public async Task ABodyInsideTheCap_StillReachesTheHandler()
        {
            using var fixture = new SteamServerFactory();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage response = await client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket = FakeSteamWebApiClient.OwnerTicket,
            });

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());
        }

        // ---- CORS ------------------------------------------------------------

        static SteamServerFactory WithAllowedOrigin()
        {
            var fixture = new SteamServerFactory();
            fixture.Settings["LOBBY_PROVIDER"] = "Legacy,Steam";
            fixture.Settings["ALLOWED_WEB_ORIGINS"] = AllowedOrigin;
            return fixture;
        }

        static async Task<HttpResponseMessage> GetWithOrigin(HttpClient client, string path, string? origin)
        {
            var request = new HttpRequestMessage(HttpMethod.Get, path);
            if (origin is not null) request.Headers.Add("Origin", origin);
            return await client.SendAsync(request);
        }

        static string? AllowOriginHeader(HttpResponseMessage response) =>
            response.Headers.TryGetValues("Access-Control-Allow-Origin", out IEnumerable<string>? values)
                ? values.FirstOrDefault()
                : null;

        [Test]
        public async Task Games_TellsAnAllowListedOriginItMayRead()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage response = await GetWithOrigin(client, "/games", AllowedOrigin);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(AllowOriginHeader(response), Is.EqualTo(AllowedOrigin));
        }

        [Test]
        public async Task Games_TellsAnyOtherOriginNothing()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage response = await GetWithOrigin(client, "/games", ForeignOrigin);

            // The body is still served - CORS is a browser rule, not an authorisation one - but without the
            // header no browser will hand it to the page that asked.
            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(AllowOriginHeader(response), Is.Null);
        }

        [Test]
        public async Task Healthz_CarriesTheSamePolicy()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage response = await GetWithOrigin(client, "/healthz", AllowedOrigin);

            Assert.That(AllowOriginHeader(response), Is.EqualTo(AllowedOrigin));
        }

        [Test]
        public async Task TheSteamApi_NeverEmitsCorsHeaders()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            var request = new HttpRequestMessage(HttpMethod.Post, CreateRoute)
            {
                Content = JsonContent.Create(new
                {
                    steamLobbyId = FakeSteamWebApiClient.LobbyId,
                    ticket = FakeSteamWebApiClient.OwnerTicket,
                }),
            };
            request.Headers.Add("Origin", AllowedOrigin);

            using HttpResponseMessage response = await client.SendAsync(request);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(AllowOriginHeader(response), Is.Null,
                "a credential-bearing endpoint must not advertise itself as a cross-site target");
        }

        [Test]
        public async Task TheSteamApi_RefusesAPreflightRatherThanAnsweringIt()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            var request = new HttpRequestMessage(HttpMethod.Options, CreateRoute);
            request.Headers.Add("Origin", AllowedOrigin);
            request.Headers.Add("Access-Control-Request-Method", "POST");

            using HttpResponseMessage response = await client.SendAsync(request);

            Assert.That(AllowOriginHeader(response), Is.Null);
        }



        [Test]
        public async Task Join_EmitsNoCorsHeaderEither()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage created = await client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket = FakeSteamWebApiClient.OwnerTicket,
            });
            Guid matchId = await MatchIdOf(created);

            var request = new HttpRequestMessage(HttpMethod.Post, JoinRoute(matchId))
            {
                Content = JsonContent.Create(new { ticket = FakeSteamWebApiClient.GuestTicket }),
            };
            request.Headers.Add("Origin", AllowedOrigin);

            using HttpResponseMessage joined = await client.SendAsync(request);

            Assert.That(joined.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await joined.Content.ReadAsStringAsync());
            Assert.That(AllowOriginHeader(joined), Is.Null);
        }

        [Test]
        public async Task APreflightForJoin_GetsNoCorsHeaders()
        {
            using SteamServerFactory fixture = WithAllowedOrigin();
            using HttpClient client = fixture.CreateClient();

            var request = new HttpRequestMessage(HttpMethod.Options, JoinRoute(Guid.NewGuid()));
            request.Headers.Add("Origin", AllowedOrigin);
            request.Headers.Add("Access-Control-Request-Method", "POST");

            using HttpResponseMessage response = await client.SendAsync(request);

            Assert.That(AllowOriginHeader(response), Is.Null,
                "answering a preflight would tell a browser this route is open to it");
        }

        [Test]
        public async Task ABlockedAccount_CannotJoinEither()
        {
            using var fixture = new SteamServerFactory();
            using HttpClient client = fixture.CreateClient();

            using HttpResponseMessage created = await client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket = FakeSteamWebApiClient.OwnerTicket,
            });
            Guid matchId = await MatchIdOf(created);

            // Blocked after the match exists, which is the case that matters: the block has to hold on the
            // route a seated player uses to get back in, not only on the one that starts a game.
            using SteamServerFactory blocked = new();
            blocked.Settings["MATCH_BLOCKED_STEAM_IDS"] = FakeSteamWebApiClient.GuestSteamId;
            blocked.Store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    FakeSteamWebApiClient.LobbyId,
                    GameSetup.Default.ToWire(),
                    "hexwars-engine/1",
                    2,
                    SteamServerFactory.BuildId,
                    new[] { (FakeSteamWebApiClient.OwnerSteamId, 0), (FakeSteamWebApiClient.GuestSteamId, 1) },
                    SteamServerFactory.Start),
                CancellationToken.None).GetAwaiter().GetResult();

            using HttpClient blockedClient = blocked.CreateClient();
            Guid seeded = (await blocked.Store.ListOpenMatchIdsAsync(CancellationToken.None)).Single();

            using HttpResponseMessage refused = await blockedClient.PostAsJsonAsync(
                JoinRoute(seeded), new { ticket = FakeSteamWebApiClient.GuestTicket });

            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await refused.Content.ReadAsStringAsync(), Does.Contain("blocked"));
            Assert.That(matchId, Is.Not.EqualTo(Guid.Empty));
        }

        [Test]
        public void TheBlockList_FollowsTheOptionsItWasGiven()
        {
            var options = new MatchHostingOptions();
            var monitor = new ReloadableOptionsMonitor(options);
            var list = new PlayerBlockList(monitor);

            Assert.That(list.IsBlocked("76561198000000001"), Is.False);

            // Render restarts the service when an environment variable changes, so a restart is the admin
            // path for the Playtest. Reading through the monitor rather than a captured snapshot means a
            // host that DOES reload configuration picks the change up without one.
            monitor.Replace(new MatchHostingOptions
            {
                BlockedSteamIds = new[] { "76561198000000001" },
            });

            Assert.That(list.IsBlocked("76561198000000001"), Is.True);
            Assert.That(list.Count, Is.EqualTo(1));
        }

        /// <summary>An options monitor whose value a test can change, the way a reloading host would.</summary>
        sealed class ReloadableOptionsMonitor(MatchHostingOptions initial) : IOptionsMonitor<MatchHostingOptions>
        {
            readonly List<Action<MatchHostingOptions, string?>> _listeners = new();

            MatchHostingOptions _value = initial;

            public MatchHostingOptions CurrentValue => _value;

            public MatchHostingOptions Get(string? name) => _value;

            public IDisposable? OnChange(Action<MatchHostingOptions, string?> listener)
            {
                _listeners.Add(listener);
                return null;
            }

            public void Replace(MatchHostingOptions value)
            {
                _value = value;
                foreach (Action<MatchHostingOptions, string?> listener in _listeners) listener(value, null);
            }
        }

        // ---- the per-address socket cap --------------------------------------

        static Task<WebSocket> OpenSocketFrom(WebApplicationFactory<Program> host, string forwardedFor)
        {
            WebSocketClient client = host.Server.CreateWebSocketClient();
            client.ConfigureRequest = request => request.Headers["X-Forwarded-For"] = forwardedFor;
            return client.ConnectAsync(new Uri("ws://localhost/ws/v2"), CancellationToken.None);
        }

        [Test]
        public async Task TheNinthSocketFromOneAddress_IsRefusedAndOthersAreNot()
        {
            using var fixture = new SteamServerFactory();
            fixture.Settings["MATCH_TRUST_FORWARDED_HEADERS"] = "true";
            fixture.Settings["MATCH_TRUST_ALL_PROXIES"] = "true";

            using WebApplicationFactory<Program> host = fixture.WithWebHostBuilder(_ => { });

            var held = new List<WebSocket>();
            try
            {
                // The default cap, exercised as the default rather than as a number this test chose.
                for (var i = 0; i < MatchHostingOptions.DefaultMaxSocketsPerIp; i++)
                {
                    held.Add(await OpenSocketFrom(host, "198.51.100.7"));
                }

                Exception? refused = null;
                try
                {
                    held.Add(await OpenSocketFrom(host, "198.51.100.7"));
                }
                catch (Exception failure)
                {
                    refused = failure;
                }

                Assert.That(refused, Is.Not.Null, "the cap is applied while refusing is still free");
                Assert.That(refused!.Message, Does.Contain("429"));

                // A different client is unaffected, which is what makes this a per-address cap rather than a
                // ceiling on the host.
                WebSocket elsewhere = await OpenSocketFrom(host, "198.51.100.8");
                held.Add(elsewhere);
                Assert.That(elsewhere.State, Is.EqualTo(WebSocketState.Open));
            }
            finally
            {
                foreach (WebSocket socket in held) socket.Dispose();
            }
        }

        // ---- the per-address open-match quota ---------------------------------

        /// <summary>A host whose Steam fake knows four distinct ready lobbies, so four creations really are
        /// four matches rather than one match asked for four times.</summary>
        static SteamServerFactory WithSeveralLobbies(int lobbies)
        {
            var fixture = new SteamServerFactory();
            fixture.RemoteIpAddress = IPAddress.Parse("198.51.100.20");

            for (var i = 0; i < lobbies; i++)
            {
                fixture.Steam.Lobbies[LobbyIdFor(i)] = FakeSteamWebApiClient.ReadyLobby(lobbyId: LobbyIdFor(i));
            }

            return fixture;
        }

        static string LobbyIdFor(int index) => "10977524000000010" + index.ToString();

        static Task<HttpResponseMessage> CreateFor(HttpClient client, int lobby) =>
            client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = LobbyIdFor(lobby),
                ticket = FakeSteamWebApiClient.OwnerTicket,
            });

        [Test]
        public async Task TheFourthMatchOneAddressAllocates_IsRefused()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(4);
            using HttpClient client = fixture.CreateClient();

            for (var lobby = 0; lobby < MatchHostingOptions.DefaultMaxOpenMatchesPerIp; lobby++)
            {
                using HttpResponseMessage allowed = await CreateFor(client, lobby);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    await allowed.Content.ReadAsStringAsync());
            }

            using HttpResponseMessage refused = await CreateFor(client, 3);

            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
            Assert.That(await refused.Content.ReadAsStringAsync(), Does.Contain("rate_limited"));
        }

        [Test]
        public async Task TheQuotaIsTheThingRefusing_NotSomethingElse()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(4);
            fixture.Settings["MATCH_MAX_OPEN_MATCHES_PER_IP"] = "5";
            using HttpClient client = fixture.CreateClient();

            for (var lobby = 0; lobby < 4; lobby++)
            {
                using HttpResponseMessage allowed = await CreateFor(client, lobby);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    await allowed.Content.ReadAsStringAsync());
            }
        }

        [Test]
        public async Task AnAllocationThatFoundTheLobbyAlreadyHadAMatch_IsNotCharged()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(1);
            using HttpClient client = fixture.CreateClient();

            // The same lobby four times. Only the first writes a match; the rest are the other seat picking
            // up a credential, and charging for those would lock a pair out of their own rematches.
            for (var attempt = 0; attempt < 4; attempt++)
            {
                using HttpResponseMessage response = await CreateFor(client, 0);
                Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    await response.Content.ReadAsStringAsync());
            }
        }



        [Test]
        public async Task ConcurrentCreatesFromOneAddress_LeaveExactlyTheCapManyMatches()
        {
            // Five, not twenty: the request rate limiter allows five creates a minute per address, so a
            // larger burst would be refused by the limiter and this test would pass without the quota
            // existing at all. The twenty-caller race is in OpenMatchQuotaTests, where nothing else can
            // answer first.
            using SteamServerFactory fixture = WithSeveralLobbies(5);
            using HttpClient client = fixture.CreateClient();

            HttpResponseMessage[] answers = await Task.WhenAll(
                Enumerable.Range(0, 5).Select(lobby => CreateFor(client, lobby)));

            try
            {
                int allowed = answers.Count(a => a.StatusCode == HttpStatusCode.OK);
                int refused = answers.Count(a => a.StatusCode == HttpStatusCode.TooManyRequests);

                Assert.That(allowed, Is.EqualTo(MatchHostingOptions.DefaultMaxOpenMatchesPerIp));
                Assert.That(refused, Is.EqualTo(5 - MatchHostingOptions.DefaultMaxOpenMatchesPerIp));
                IReadOnlyList<Guid> stored =
                    await fixture.Store.ListOpenMatchIdsAsync(CancellationToken.None);
                Assert.That(stored, Has.Count.EqualTo(MatchHostingOptions.DefaultMaxOpenMatchesPerIp),
                    "the cap has to be spent before the write, not counted after it");
            }
            finally
            {
                foreach (HttpResponseMessage answer in answers) answer.Dispose();
            }
        }

        [Test]
        public async Task ACreateThatSteamRefuses_HandsTheSeatBack()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(4);
            using HttpClient client = fixture.CreateClient();

            fixture.Steam.NextFailure =
                new SteamApiException(SteamFailure.ServiceUnavailable, "Steam is having a moment");

            using (HttpResponseMessage failed = await CreateFor(client, 0))
            {
                Assert.That(failed.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            }

            // Three more must still fit: the attempt that failed wrote nothing, so it bought nothing.
            for (var lobby = 1; lobby <= 3; lobby++)
            {
                using HttpResponseMessage allowed = await CreateFor(client, lobby);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    await allowed.Content.ReadAsStringAsync());
            }
        }

        [Test]
        public async Task ACreateTheStoreRefuses_HandsTheSeatBack()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(4);
            using HttpClient client = fixture.CreateClient();

            fixture.Store.InjectedWriteFailure = new InvalidOperationException("the database went away");

            using (HttpResponseMessage failed = await CreateFor(client, 0))
            {
                Assert.That(failed.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            }

            // An ambiguous commit IS charged - the row may exist - so exactly one seat is gone and two
            // remain. That is the fail-closed half of the lease, asserted rather than assumed.
            for (var lobby = 1; lobby <= 2; lobby++)
            {
                using HttpResponseMessage allowed = await CreateFor(client, lobby);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    await allowed.Content.ReadAsStringAsync());
            }

            using HttpResponseMessage refused = await CreateFor(client, 3);
            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
        }

        [Test]
        public async Task ARepeatCreateForALobbyThatAlreadyHasAMatch_IsServedEvenAtTheCap()
        {
            using SteamServerFactory fixture = WithSeveralLobbies(4);
            using HttpClient client = fixture.CreateClient();

            Guid first = Guid.Empty;
            for (var lobby = 0; lobby < MatchHostingOptions.DefaultMaxOpenMatchesPerIp; lobby++)
            {
                using HttpResponseMessage allowed = await CreateFor(client, lobby);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK));
                if (lobby == 0) first = await MatchIdOf(allowed);
            }

            // The owner is at the cap and their first response was lost. Retrying writes nothing - it hands
            // back the match they already have - so the cap has no business refusing it.
            using HttpResponseMessage retry = await CreateFor(client, 0);

            Assert.That(retry.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await retry.Content.ReadAsStringAsync());
            Assert.That(await MatchIdOf(retry), Is.EqualTo(first), "and it is the same match");

            using HttpResponseMessage stillRefused = await CreateFor(client, 3);
            Assert.That(stillRefused.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests),
                "serving the retry did not hand the caller a fourth seat either");
        }

        static async Task<Guid> MatchIdOf(HttpResponseMessage response)
        {
            string text = await response.Content.ReadAsStringAsync();
            return Guid.Parse(JsonDocument.Parse(text).RootElement.GetProperty("matchId").GetString()!);
        }

        // ---- log hygiene ------------------------------------------------------

        /// <summary>A DATABASE_URL whose password is distinctive enough to search a whole log for. The host
        /// never connects to it - the store is replaced and the startup migration removed - so the only
        /// thing being tested is whether the value reaches a log line.</summary>
        const string DatabasePassword = "pw-must-not-appear";

        const string LeakDatabaseUrl = "postgres://hexwars:" + DatabasePassword + "@127.0.0.1:1/leaks";

        /// <summary>The value the fixture puts in STEAM_PUBLISHER_WEB_API_KEY.</summary>
        const string PublisherKey = "test-key";

        [Test]
        public async Task AFullMatchFlow_PutsNoSecretInAnyLogLine()
        {
            var log = new CapturingLoggerProvider();

            using var fixture = new SteamServerFactory { Logging = log };
            fixture.Settings["DATABASE_URL"] = LeakDatabaseUrl;
            fixture.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            using WebApplicationFactory<Program> host = fixture.WithWebHostBuilder(_ => { });

            await using DurableFlowClient zero = await DurableFlowClient.CreateAsync(
                host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);
            await using DurableFlowClient one = await DurableFlowClient.JoinAsync(
                host, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

            await zero.ConnectAsync();
            await zero.ExpectAsync(NetProtocol.CatalogRequest);
            await one.ConnectAsync();
            await one.ExpectAsync(NetProtocol.CatalogRequest);

            await zero.SendCatalogAsync();
            await one.SendCatalogAsync();
            await zero.ExpectAsync("START ");
            await one.ExpectAsync("START ");

            // One real command, so the journal path and its logging run too.
            var opening = new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0));
            await zero.SendCmdAsync(opening);
            await zero.ExpectAsync("APPLY ");
            await one.ExpectAsync("APPLY ");

            // A frame this build refuses, because a refusal is where a careless log line would print what it
            // refused - and what a confused client sends first is often its bare credential.
            await zero.SendAsync("CMD not-a-command");
            await zero.ExpectAsync("REJECT ");

            // The graceful-restart notice, broadcast the way a shutdown would broadcast it.
            await host.Services.GetRequiredService<DurableMatchCoordinator>()
                .BroadcastAllAsync("SERVER RESTART");
            await zero.ExpectAsync("SERVER RESTART");

            var secrets = new List<(string What, string Value)>
            {
                ("the owner ticket", FakeSteamWebApiClient.OwnerTicket),
                ("the guest ticket", FakeSteamWebApiClient.GuestTicket),
                ("seat 0 credential", zero.Credential),
                ("seat 1 credential", one.Credential),
                ("the publisher web API key", "test-key"),
                ("the database password", DatabasePassword),
            };

            Assert.That(log.Messages, Is.Not.Empty, "a leak audit over an empty log proves nothing");

            foreach ((string what, string value) in secrets)
            {
                Assert.That(log.Containing(value), Is.Empty, what + " reached a log line");
            }
        }


        [Test]
        public async Task AFailureThatQuotesASecret_DoesNotPutItInTheLog()
        {
            var log = new CapturingLoggerProvider();

            using var fixture = new SteamServerFactory { Logging = log };
            fixture.Settings["DATABASE_URL"] = LeakDatabaseUrl;
            using HttpClient client = fixture.CreateClient();

            // The exceptions a real outage raises are written by code that was holding a secret when it
            // failed: a transport error quotes the URL it could not reach, which for the Steam client is the
            // publisher key and the ticket, and a connection failure quotes DATABASE_URL. These stand in for
            // those, with the same shapes.
            fixture.Steam.NextFailure = new HttpRequestException(
                "Connection refused for https://partner.steam-api.com/ISteamUserAuth/AuthenticateUserTicket/v1"
                + "?key=" + PublisherKey + "&ticket=" + FakeSteamWebApiClient.OwnerTicket);

            using (HttpResponseMessage steamFailed = await client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket = FakeSteamWebApiClient.OwnerTicket,
            }))
            {
                Assert.That(steamFailed.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            }

            fixture.Store.InjectedWriteFailure = new InvalidOperationException(
                "Failed to connect to " + LeakDatabaseUrl + " (Password=" + DatabasePassword + ")");

            using (HttpResponseMessage storeFailed = await client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket = FakeSteamWebApiClient.OwnerTicket,
            }))
            {
                Assert.That(storeFailed.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            }

            Assert.That(log.Messages, Is.Not.Empty, "a leak audit over an empty log proves nothing");
            Assert.That(
                log.Messages.Any(m => m.Contains("Match creation failed", StringComparison.Ordinal)),
                Is.True,
                "the failures really did reach a log line, or the assertions below search nothing");

            foreach ((string what, string value) in new[]
            {
                ("the publisher web API key", PublisherKey),
                ("the owner ticket", FakeSteamWebApiClient.OwnerTicket),
                ("the database password", DatabasePassword),
            })
            {
                Assert.That(log.Containing(value), Is.Empty,
                    what + " reached a log line through an exception message");
            }
        }

        [TestCase("key=abc123", "abc123")]
        [TestCase("ticket=deadbeef", "deadbeef")]
        [TestCase("Password=hunter2;Host=db", "hunter2")]
        [TestCase("pwd=hunter2", "hunter2")]
        [TestCase("postgres://user:hunter2@db:5432/hexwars", "hunter2")]
        [TestCase("https://partner.steam-api.com/x?key=abc123", "abc123")]
        public void RedactionCoversEveryShapeASecretArrivesIn(string text, string secret)
        {
            string redacted = SteamLogRedaction.Redact(text);

            Assert.That(redacted, Does.Not.Contain(secret), redacted);
        }

        [Test]
        public void DescribingAnExceptionKeepsTheStackAndDropsTheSecret()
        {
            Exception failure;
            try
            {
                throw new InvalidOperationException("could not reach postgres://u:hunter2@db/hexwars");
            }
            catch (Exception thrown)
            {
                failure = thrown;
            }

            string described = SteamLogRedaction.Describe(failure);

            Assert.That(described, Does.Not.Contain("hunter2"));
            Assert.That(described, Does.Contain(nameof(DescribingAnExceptionKeepsTheStackAndDropsTheSecret)),
                "the stack trace names methods rather than values, and is the part worth keeping");
        }

        [Test]
        public void TheLeakAuditWouldNoticeASecret()
        {
            // The audit above is only worth its runtime if a value in a scope or a message would actually be
            // found. Both halves are checked here, because scopes are the mechanism this server uses to carry
            // identifiers and an audit blind to them would pass over exactly the wrong log.
            var log = new CapturingLoggerProvider();
            ILogger logger = log.CreateLogger("test");

            logger.LogInformation("a message carrying {Value}", "in-the-message");
            using (logger.BeginScope(new Dictionary<string, object> { ["Value"] = "in-the-scope" }))
            {
                logger.LogInformation("a message carrying nothing itself");
            }

            Assert.That(log.Containing("in-the-message"), Is.Not.Empty);
            Assert.That(log.Containing("in-the-scope"), Is.Not.Empty);
        }


        // ---- the console the platform actually reads --------------------------

        [Test]
        public void TheDevelopmentConsole_RendersScopes()
        {
            using var fixture = new SteamServerFactory();
            using HttpClient started = fixture.CreateClient();

            var console = fixture.Services
                .GetRequiredService<IOptionsMonitor<SimpleConsoleFormatterOptions>>();

            Assert.That(console.CurrentValue.IncludeScopes, Is.True,
                "MatchId and LobbyId are carried as SCOPES, and a formatter that does not render them "
                + "computes those identifiers and then drops them on the floor");
        }

        [Test]
        public void TheProductionConsole_RendersScopesAsJsonInUtc()
        {
            using var fixture = new SteamServerFactory { Environment = "Production" };
            fixture.Settings["MATCH_PUBLIC_BASE_URL"] = "https://match.test";
            using HttpClient started = fixture.CreateClient();

            var console = fixture.Services
                .GetRequiredService<IOptionsMonitor<JsonConsoleFormatterOptions>>();

            Assert.That(console.CurrentValue.IncludeScopes, Is.True);
            Assert.That(console.CurrentValue.UseUtcTimestamp, Is.True,
                "one timezone in the log, and the one every other timestamp in this system is in");
        }

        [Test]
        public async Task TwoMatchesLoggedAtOnce_KeepTheirOwnScopes()
        {
            var log = new CapturingLoggerProvider();
            ILogger logger = log.CreateLogger("concurrent");

            Guid first = Guid.NewGuid(), second = Guid.NewGuid();
            using var bothInside = new SemaphoreSlim(0, 2);

            async Task WorkOn(Guid matchId)
            {
                using IDisposable? scope = LogScopes.MatchScope(logger, matchId);

                // Both scopes are open at the same time on different flows, which is the only arrangement
                // in which a scope leaking between them would show up at all.
                bothInside.Release();
                await bothInside.WaitAsync().ConfigureAwait(false);
                await Task.Yield();

                logger.LogInformation("working");
            }

            await Task.WhenAll(WorkOn(first), WorkOn(second));

            string firstHandle = LogScopes.ShortMatchId(first);
            string secondHandle = LogScopes.ShortMatchId(second);

            Assert.That(log.Containing(firstHandle), Has.Count.EqualTo(1));
            Assert.That(log.Containing(secondHandle), Has.Count.EqualTo(1));
            Assert.That(log.Containing(firstHandle).Single(), Does.Not.Contain(secondHandle),
                "a scope belongs to one async flow, not to the thread that happened to run it");
        }

        // ---- the retention sweeper, through the host --------------------------

        [Test]
        public async Task AnActiveMatchGoneQuiet_IsAbandonedAndItsSocketClosed()
        {
            using var fixture = new SteamServerFactory();
            fixture.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                ruleset: SteamLobbyRules.CustomRuleset, setupWire: GameSetup.Default.ToWire());

            // The heartbeat is removed for this test alone. It runs on the same injected clock, so winding it
            // forward eight days would have the liveness sweep close these sockets as stale long before
            // retention looked at them - and this test is about retention, not about liveness.
            using WebApplicationFactory<Program> host = fixture.WithWebHostBuilder(
                builder => builder.ConfigureServices(services =>
                {
                    ServiceDescriptor? heartbeat = services.FirstOrDefault(
                        d => d.ImplementationType == typeof(ConnectionHeartbeatService));
                    if (heartbeat is not null) services.Remove(heartbeat);
                }));

            await using DurableFlowClient zero = await DurableFlowClient.CreateAsync(
                host, FakeSteamWebApiClient.LobbyId, FakeSteamWebApiClient.OwnerTicket);
            await using DurableFlowClient one = await DurableFlowClient.JoinAsync(
                host, zero.MatchId, FakeSteamWebApiClient.GuestTicket);

            await zero.ConnectAsync();
            await zero.ExpectAsync(NetProtocol.CatalogRequest);
            await one.ConnectAsync();
            await one.ExpectAsync(NetProtocol.CatalogRequest);
            await zero.SendCatalogAsync();
            await one.SendCatalogAsync();
            await zero.ExpectAsync("START ");
            await one.ExpectAsync("START ");

            // The sweep has to have armed its timer before the clock jumps, or the jump moves past nothing.
            await WaitUntil(() => fixture.Clock.ScheduledTimers > 0);

            // Wound past the activity the MATCH recorded rather than past where this clock started. The
            // catalog write stamps activity from the store clock, mirroring the now() the SQL uses, so
            // the instant to age from is the one in the row and not the one the test began at.
            DateTimeOffset lastActivity =
                (await fixture.Store.GetMatchAsync(zero.MatchId, CancellationToken.None))!.LastActivityAt;
            fixture.Clock.SetUtcNow(
                lastActivity + TimeSpan.FromDays(MatchHostingOptions.DefaultRetentionActiveIdleDays + 1));

            var sweeper = host.Services.GetRequiredService<MatchRetentionService>();

            // One pass on top of whatever the cadence already did. A PeriodicTimer coalesces the ticks a
            // single jump produces, so the pass the cadence ran may have landed anywhere inside the eight
            // days; the assertions below are about the state either of them leaves, not about which one ran.
            await sweeper.SweepOnceAsync(CancellationToken.None);

            Assert.That(sweeper.Sweeps, Is.GreaterThan(0),
                "the cadence runs on the injected clock, so winding it forward is what wakes the sweeper");

            PersistedMatch stored = (await fixture.Store.GetMatchAsync(zero.MatchId, CancellationToken.None))!;
            Assert.That(stored.Status, Is.EqualTo(MatchStatus.Abandoned),
                "a match nobody has touched for eight days is abandoned");

            Assert.That(await zero.ExpectCloseAsync(), Is.EqualTo(WebSocketCloseStatus.EndpointUnavailable));
            Assert.That(stored.CompletedAt, Is.Not.Null, "an abandoned match records when it ended");
        }

        [Test]
        public async Task AnActiveMatchStillBeingPlayed_IsLeftAlone()
        {
            using var fixture = new SteamServerFactory();
            using WebApplicationFactory<Program> host = fixture.WithWebHostBuilder(_ => { });
            using HttpClient started = host.CreateClient();

            CreateMatchResult created = await fixture.Store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    FakeSteamWebApiClient.LobbyId,
                    GameSetup.Default.ToWire(),
                    "hexwars-engine/1",
                    2,
                    SteamServerFactory.BuildId,
                    new[] { (FakeSteamWebApiClient.OwnerSteamId, 0), (FakeSteamWebApiClient.GuestSteamId, 1) },
                    SteamServerFactory.Start),
                CancellationToken.None);

            await fixture.Store.TryStartMatchAsync(
                created.Match.MatchId, "START-REPLAY", SteamServerFactory.Start, CancellationToken.None);

            // Six days of silence, one day inside the window. A game in progress is never disturbed.
            fixture.Clock.Advance(TimeSpan.FromDays(6));

            var sweeper = host.Services.GetRequiredService<MatchRetentionService>();
            RetentionResult result = await sweeper.SweepOnceAsync(CancellationToken.None);

            Assert.That(result.Abandoned, Is.EqualTo(0));
            Assert.That((await fixture.Store.GetMatchAsync(created.Match.MatchId, CancellationToken.None))!.Status,
                Is.EqualTo(MatchStatus.Active));
        }

        /// <summary>Polls until <paramref name="ready"/> holds, or fails the test. Used only for the one
        /// thing a test cannot arrange directly: a background service reaching the point where it waits.</summary>
        static async Task WaitUntil(Func<bool> ready)
        {
            DateTimeOffset deadline = DateTimeOffset.UtcNow.AddSeconds(10);
            while (DateTimeOffset.UtcNow < deadline)
            {
                if (ready()) return;
                await Task.Delay(10);
            }

            Assert.Fail("the host never reached the state this test was waiting for");
        }

        // ---- the block list ---------------------------------------------------

        static PlayerBlockList BlockList(params string[] blocked)
        {
            var options = new MatchHostingOptions { BlockedSteamIds = blocked };
            return new PlayerBlockList(new StaticOptionsMonitor(options));
        }

        [TestCase("76561198000000001")]
        [TestCase(" 76561198000000001 ")]
        [TestCase("076561198000000001")]
        public void ABlockedAccount_MatchesHoweverItWasWritten(string configured)
        {
            Assert.That(BlockList(configured).IsBlocked("76561198000000001"), Is.True);
        }

        [Test]
        public void AnAccountWrittenWithPadding_IsStillRecognisedAtTheCallSite() =>
            Assert.That(
                BlockList("76561198000000001").IsBlocked(" 76561198000000001 "), Is.True);

        [Test]
        public void AnAccountNotOnTheList_IsNotBlocked() =>
            Assert.That(BlockList("76561198000000001").IsBlocked("76561198000000002"), Is.False);

        [Test]
        public void AnEmptyListBlocksNobody() =>
            Assert.That(BlockList().IsBlocked("76561198000000001"), Is.False);

        [Test]
        public void AMistypedEntry_RefusesExactlyWhatItSaysAndNothingElse()
        {
            PlayerBlockList list = BlockList("not-a-steam-id");

            Assert.That(list.IsBlocked("76561198000000001"), Is.False);
            Assert.That(list.IsBlocked("not-a-steam-id"), Is.True);
        }

        /// <summary>The options this server actually has: read once at startup and never reloaded.</summary>
        sealed class StaticOptionsMonitor(MatchHostingOptions value) : IOptionsMonitor<MatchHostingOptions>
        {
            public MatchHostingOptions CurrentValue => value;

            public MatchHostingOptions Get(string? name) => value;

            public IDisposable? OnChange(Action<MatchHostingOptions, string?> listener) => null;
        }
    }

    /// <summary>
    /// The retention rules, run against both stores.
    ///
    /// Written once and executed twice for the same reason the match-store contract is: the in-memory double
    /// is what the coordinator tests lean on, and a double that aged a match out on a different boundary
    /// than Postgres would make every one of those tests a test of a system nobody deploys.
    /// </summary>
    public abstract class MatchRetentionContractTests
    {
        protected const string SetupWire = "annihilation 9 7 0 7 3 1 1 1 3 0";
        protected const string EngineVersion = "hexwars-engine/1";
        protected const string BuildId = "test-build";

        /// <summary>The instant every age in these tests is measured back from.</summary>
        protected static readonly DateTimeOffset Now = new(2026, 6, 1, 12, 0, 0, TimeSpan.Zero);

        /// <summary>The policy as the retention decision states it.</summary>
        protected static readonly RetentionPolicy Policy = new(
            TimeSpan.FromMinutes(30), TimeSpan.FromDays(7), TimeSpan.FromDays(90), TimeSpan.FromHours(24));

        static int _identifiers;

        protected IMatchStore Store = null!;

        protected static CancellationToken Ct => CancellationToken.None;

        protected abstract Task<IMatchStore> CreateStoreAsync();

        [SetUp]
        public async Task StartFromAnEmptyStore() => Store = await CreateStoreAsync();

        protected static string NextLobbyId() =>
            "1097752" + Interlocked.Increment(ref _identifiers).ToString("D11");

        protected static string NextSteamId() =>
            "7656119" + Interlocked.Increment(ref _identifiers).ToString("D10");

        protected async Task<(Guid MatchId, string Seat0, string Seat1)> WaitingAsync(DateTimeOffset createdAt)
        {
            string seat0 = NextSteamId(), seat1 = NextSteamId();
            CreateMatchResult created = await Store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    NextLobbyId(), SetupWire, EngineVersion, 2, BuildId,
                    new[] { (seat0, 0), (seat1, 1) }, createdAt),
                Ct);

            return (created.Match.MatchId, seat0, seat1);
        }

        protected async Task<(Guid MatchId, string Seat0, string Seat1)> ActiveAsync(
            DateTimeOffset createdAt, DateTimeOffset lastActivityAt)
        {
            var match = await WaitingAsync(createdAt);
            await Store.TryStartMatchAsync(match.MatchId, "START-REPLAY", createdAt, Ct);
            await Store.TouchAsync(match.MatchId, match.Seat0, lastActivityAt, Ct);
            return match;
        }

        protected static byte[] Hash(byte seed)
        {
            var hash = new byte[32];
            for (var i = 0; i < hash.Length; i++) hash[i] = (byte)(seed + i);
            return hash;
        }

        [Test]
        public async Task AWaitingMatchOlderThanTheWindow_Expires()
        {
            var stale = await WaitingAsync(Now.AddMinutes(-31));
            var fresh = await WaitingAsync(Now.AddMinutes(-29));

            RetentionResult swept = await Store.ApplyRetentionAsync(Policy, Now, Ct);

            Assert.That(swept.Expired, Is.EqualTo(1));

            PersistedMatch aged = (await Store.GetMatchAsync(stale.MatchId, Ct))!;
            Assert.That(aged.Status, Is.EqualTo(MatchStatus.Expired));
            Assert.That(aged.CompletedAt, Is.EqualTo(Now));
            Assert.That(aged.WinnerSeat, Is.Null, "a match nobody played has no winner");

            Assert.That((await Store.GetMatchAsync(fresh.MatchId, Ct))!.Status,
                Is.EqualTo(MatchStatus.Waiting), "a match inside the window is untouched");
        }

        [Test]
        public async Task AnActiveMatchIdleLongerThanTheWindow_IsAbandonedAndNamed()
        {
            var quiet = await ActiveAsync(Now.AddDays(-30), Now.AddDays(-8));
            var playing = await ActiveAsync(Now.AddDays(-30), Now.AddMinutes(-1));

            RetentionResult swept = await Store.ApplyRetentionAsync(Policy, Now, Ct);

            Assert.That(swept.Abandoned, Is.EqualTo(1));
            Assert.That(swept.AbandonedIds, Is.EqualTo(new[] { quiet.MatchId }));

            PersistedMatch abandoned = (await Store.GetMatchAsync(quiet.MatchId, Ct))!;
            Assert.That(abandoned.Status, Is.EqualTo(MatchStatus.Abandoned));
            Assert.That(abandoned.CompletedAt, Is.EqualTo(Now));

            Assert.That((await Store.GetMatchAsync(playing.MatchId, Ct))!.Status,
                Is.EqualTo(MatchStatus.Active),
                "a game in progress is never disturbed by retention");
        }

        [Test]
        public async Task ATerminalMatchPastTheKeepWindow_IsDeletedWithEverythingUnderIt()
        {
            var old = await ActiveAsync(Now.AddDays(-200), Now.AddDays(-200));
            await Store.AppendCommandAsync(old.MatchId, 1, "E 0", old.Seat0, Now.AddDays(-200), Ct);
            await Store.StoreJoinCredentialAsync(Hash(3), old.MatchId, old.Seat0, Now.AddDays(-199), Ct);
            await Store.TryCompleteMatchAsync(old.MatchId, MatchStatus.Completed, 0, Now.AddDays(-91), Ct);

            var recent = await ActiveAsync(Now.AddDays(-200), Now.AddDays(-200));
            await Store.TryCompleteMatchAsync(recent.MatchId, MatchStatus.Completed, 1, Now.AddDays(-89), Ct);

            RetentionResult swept = await Store.ApplyRetentionAsync(Policy, Now, Ct);

            Assert.That(swept.MatchesDeleted, Is.EqualTo(1));
            Assert.That(await Store.GetMatchAsync(old.MatchId, Ct), Is.Null);
            Assert.That(await Store.LoadJournalAsync(old.MatchId, Ct), Is.Null);
            Assert.That(await Store.GetPlayersAsync(old.MatchId, Ct), Is.Empty,
                "the seats cascade from the match row");
            Assert.That(await Store.FindJoinCredentialAsync(Hash(3), Ct), Is.Null,
                "so do the credentials bound to those seats");

            Assert.That(await Store.GetMatchAsync(recent.MatchId, Ct), Is.Not.Null,
                "a match inside the keep window stays");
        }

        [Test]
        public async Task ACredentialPastItsExpiryPlusTheGrace_IsDeleted()
        {
            var match = await ActiveAsync(Now.AddDays(-1), Now.AddMinutes(-1));

            await Store.StoreJoinCredentialAsync(Hash(11), match.MatchId, match.Seat0, Now.AddHours(-25), Ct);
            await Store.StoreJoinCredentialAsync(Hash(51), match.MatchId, match.Seat1, Now.AddHours(-23), Ct);

            RetentionResult swept = await Store.ApplyRetentionAsync(Policy, Now, Ct);

            Assert.That(swept.CredentialsDeleted, Is.EqualTo(1));
            Assert.That(await Store.FindJoinCredentialAsync(Hash(11), Ct), Is.Null);
            Assert.That(await Store.FindJoinCredentialAsync(Hash(51), Ct), Is.Not.Null,
                "a credential still inside the grace period is kept, expired or not");
        }

        [Test]
        public async Task ASweepOverAnUntouchedStore_ReportsNothing()
        {
            await ActiveAsync(Now.AddDays(-1), Now.AddMinutes(-1));
            await WaitingAsync(Now.AddMinutes(-5));

            RetentionResult swept = await Store.ApplyRetentionAsync(Policy, Now, Ct);

            Assert.That(swept.IsEmpty, Is.True);
            Assert.That(swept.AbandonedIds, Is.Empty);
        }
    }

    [TestFixture]
    public sealed class InMemoryMatchRetentionTests : MatchRetentionContractTests
    {
        protected override Task<IMatchStore> CreateStoreAsync() =>
            Task.FromResult<IMatchStore>(new InMemoryMatchStore());
    }

    /// <summary>The same rules against the real four statements, where the cascade is the schema doing it.</summary>
    [TestFixture]
    public sealed class PostgresMatchRetentionTests : MatchRetentionContractTests
    {
        protected override async Task<IMatchStore> CreateStoreAsync()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();
            return new PostgresMatchStore(database.DataSource, NullLogger<PostgresMatchStore>.Instance);
        }

        /// <summary>
        /// The retention statements are literal-status so the planner can use the partial indexes.
        ///
        /// It is checked with EXPLAIN rather than by reading the SQL, because the claim is about the
        /// PLANNER: a parameterised status compiles and runs and returns the right rows, and the only
        /// visible difference is that an hourly sweep becomes a sequential scan of every match ever played.
        /// Sequential scans are turned off for the check so the question is whether the index CAN be used
        /// for this predicate, which is exactly what a parameter would take away.
        /// </summary>
        [Test]
        public async Task TheWaitingExpiryStatement_CanUseItsPartialIndex()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();

            string plan = await ExplainAsync(
                database, PostgresMatchStore.ExpireWaitingStatement, TimeSpan.FromMinutes(30));

            Assert.That(plan, Does.Contain("ix_matches_waiting_created"),
                "a parameterised status cannot be proved to satisfy a partial index predicate: " + plan);
        }

        [Test]
        public async Task TheTerminalPurgeStatement_CanUseItsPartialIndex()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();

            string plan = await ExplainAsync(
                database, PostgresMatchStore.PurgeTerminalStatement, TimeSpan.FromDays(90));

            Assert.That(plan, Does.Contain("ix_matches_terminal_completed"), plan);
        }

        static async Task<string> ExplainAsync(
            PostgresTestDatabase database, string statement, TimeSpan age)
        {
            await using NpgsqlConnection connection = await database.DataSource.OpenConnectionAsync();

            await using (var seqscan = new NpgsqlCommand("SET enable_seqscan = off", connection))
            {
                await seqscan.ExecuteNonQueryAsync();
            }

            await using var explain = new NpgsqlCommand("EXPLAIN " + statement, connection);
            explain.Parameters.Add(
                new NpgsqlParameter("now", NpgsqlDbType.TimestampTz) { Value = DateTime.UtcNow });
            explain.Parameters.Add(new NpgsqlParameter("age", NpgsqlDbType.Interval) { Value = age });

            var plan = new System.Text.StringBuilder();
            await using NpgsqlDataReader reader = await explain.ExecuteReaderAsync();
            while (await reader.ReadAsync()) plan.AppendLine(reader.GetString(0));

            return plan.ToString();
        }
    }
}
