using System.Net;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.DependencyInjection;
using Npgsql;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The Steam match allocation endpoints, driven through the real host over real HTTP.
    ///
    /// Every test here builds its own server. That is not tidiness: the rate limiter and the auth-failure
    /// throttle both partition on the caller address, which under the test server is the same address for
    /// every request, so a shared host would let one test spend another one\u2019s budget and the failure
    /// would land on whichever test happened to run fifth.
    /// </summary>
    [TestFixture]
    public class SteamMatchEndpointTests
    {
        const string CreateRoute = "/api/v1/steam/matches";
        const string UnknownTicket = "deadbeef";

        static string JoinRoute(Guid matchId) => CreateRoute + "/" + matchId.ToString() + "/join";

        static Task<HttpResponseMessage> Create(
            HttpClient client, string ticket, string? lobbyId = null, string? requestedSetup = null) =>
            client.PostAsJsonAsync(CreateRoute, new
            {
                steamLobbyId = lobbyId ?? FakeSteamWebApiClient.LobbyId,
                ticket,
                requestedSetup,
            });

        static Task<HttpResponseMessage> Join(HttpClient client, Guid matchId, string ticket) =>
            client.PostAsJsonAsync(JoinRoute(matchId), new { ticket });

        static async Task<HttpResponseMessage> CreateFrom(
            HttpClient client, string ticket, string forwardedFor)
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, CreateRoute)
            {
                Content = JsonContent.Create(new
                {
                    steamLobbyId = FakeSteamWebApiClient.LobbyId,
                    ticket,
                }),
            };
            request.Headers.Add("X-Forwarded-For", forwardedFor);
            return await client.SendAsync(request);
        }

        static async Task<HttpResponseMessage> JoinFrom(
            HttpClient client, Guid matchId, string ticket, string forwardedFor)
        {
            using var request = new HttpRequestMessage(HttpMethod.Post, JoinRoute(matchId))
            {
                Content = JsonContent.Create(new { ticket }),
            };
            request.Headers.Add("X-Forwarded-For", forwardedFor);
            return await client.SendAsync(request);
        }

        static async Task<JsonElement> Body(HttpResponseMessage response)
        {
            string text = await response.Content.ReadAsStringAsync();
            return JsonDocument.Parse(text).RootElement.Clone();
        }

        static async Task<string?> ErrorCode(HttpResponseMessage response) =>
            (await Body(response)).GetProperty("error").GetString();

        static async Task<Guid> CreatedMatchId(HttpResponseMessage response)
        {
            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());
            return Guid.Parse((await Body(response)).GetProperty("matchId").GetString()!);
        }

        // ---- create: the happy path ------------------------------------------------------------

        [Test]
        public async Task OwnerCreate_AnswersWithTheMatchTicketForSeatZero()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());

            JsonElement body = await Body(response);
            Assert.Multiple(() =>
            {
                Assert.That(Guid.TryParse(body.GetProperty("matchId").GetString(), out _), Is.True);
                Assert.That(body.GetProperty("protocolVersion").GetInt32(), Is.EqualTo(2));
                Assert.That(body.GetProperty("websocketUrl").GetString(),
                    Is.EqualTo(SteamServerFactory.WebsocketUrl));
                Assert.That(body.GetProperty("seat").GetInt32(), Is.EqualTo(0));
                Assert.That(body.GetProperty("joinCredential").GetString(),
                    Has.Length.EqualTo(CredentialEncoding.CredentialCharacters));
                Assert.That(
                    body.GetProperty("credentialExpiresAt").GetDateTimeOffset(),
                    Is.EqualTo(SteamServerFactory.Start.AddSeconds(SteamServerFactory.JoinTokenTtlSeconds)));
            });
        }

        [Test]
        public async Task TheResponseIsCamelCasedExactlyAsTheContractSays()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            JsonElement body = await Body(await Create(client, FakeSteamWebApiClient.OwnerTicket));

            Assert.That(
                body.EnumerateObject().Select(property => property.Name).ToArray(),
                Is.EquivalentTo(new[]
                {
                    "matchId", "protocolVersion", "websocketUrl", "seat", "joinCredential",
                    "credentialExpiresAt",
                }));
        }

        [Test]
        public async Task ASecondCreateReturnsTheSameMatchAndRetiresTheEarlierCredential()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            JsonElement first = await Body(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            JsonElement second = await Body(await Create(client, FakeSteamWebApiClient.OwnerTicket));

            string firstCredential = first.GetProperty("joinCredential").GetString()!;
            string secondCredential = second.GetProperty("joinCredential").GetString()!;
            var matchId = Guid.Parse(first.GetProperty("matchId").GetString()!);

            Assert.Multiple(() =>
            {
                Assert.That(second.GetProperty("matchId").GetString(),
                    Is.EqualTo(first.GetProperty("matchId").GetString()),
                    "a retry for the same lobby must not allocate a second match");
                Assert.That(secondCredential, Is.Not.EqualTo(firstCredential));
            });

            var credentials = factory.Services.GetRequiredService<IMatchCredentialService>();
            Assert.That(await credentials.ValidateAsync(matchId, firstCredential, default), Is.Null,
                "issuing a credential must revoke the one it replaces");
            Assert.That(await credentials.ValidateAsync(matchId, secondCredential, default), Is.Not.Null);
        }

        [Test]
        public async Task ARequestedSetupThatStillMatchesTheLobbyIsAccepted()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(
                client, FakeSteamWebApiClient.OwnerTicket,
                requestedSetup: FakeSteamWebApiClient.QuickSetupWire);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());
        }

        // ---- create: refusals ------------------------------------------------------------------

        [Test]
        public async Task AGuestCannotStartTheMatch()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("not_lobby_owner"));
        }

        [Test]
        public async Task ATicketFromOutsideTheLobbyIsRefusedAsNotAMember()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OutsiderTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("not_lobby_member"));
        }

        [Test]
        public async Task AnUnknownTicketIsUnauthorized()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, UnknownTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
            Assert.That(await ErrorCode(response), Is.EqualTo("authentication_failed"));
        }

        [Test]
        public async Task AnAccountThatDoesNotOwnTheGameIsForbidden()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.Ownership.Remove(FakeSteamWebApiClient.OwnerSteamId);
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("ownership_missing"));
            Assert.That(factory.Steam.LobbyCalls, Is.Zero,
                "the lobby must not be read for an account that does not own the game");
        }

        [Test]
        public async Task ALobbyAdvertisingAnotherAppRequiresAnUpgrade()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] =
                FakeSteamWebApiClient.ReadyLobby(appId: "999999");
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That((int)response.StatusCode, Is.EqualTo(426));
            Assert.That(await ErrorCode(response), Is.EqualTo("incompatible_version"));
        }

        [Test]
        public async Task ALobbyWhereTheGuestIsNotReadyIsAConflict()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] =
                FakeSteamWebApiClient.ReadyLobby(guestReady: false);
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
        }

        [Test]
        public async Task ALobbySteamNoLongerHasIsAConflict()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.Lobbies.Clear();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
        }

        [Test]
        public async Task SteamBeingDownIsAServiceOutageRatherThanAPlayerError()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.NextFailure =
                new SteamApiException(SteamFailure.ServiceUnavailable, "partner api unreachable");
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(await ErrorCode(response), Is.EqualTo("service_unavailable"));
        }

        [Test]
        public async Task AStoreFailureIsAServiceOutageAndLeavesNothingBehind()
        {
            using var factory = new SteamServerFactory();
            factory.Store.InjectedWriteFailure = new InvalidOperationException("the write did not land");
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(await ErrorCode(response), Is.EqualTo("service_unavailable"));
            Assert.That(factory.Store.WriteCount, Is.Zero, "nothing may have been written");
            Assert.That(
                await factory.Store.FindOpenMatchForLobbyAsync(FakeSteamWebApiClient.LobbyId, default),
                Is.Null);
        }

        [Test]
        public async Task ARequestedSetupThatNoLongerMatchesTheLobbyIsAConflict()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(
                client, FakeSteamWebApiClient.OwnerTicket,
                requestedSetup: SteamLobbyRules.QuickMatchSetup(1234).ToWire());

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
            Assert.That(factory.Store.WriteCount, Is.Zero);
        }

        [Test]
        public async Task AnEmptyTicketIsRefusedBeforeSteamIsAsked()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, string.Empty);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.Steam.AuthenticateCalls, Is.Zero);
        }

        [Test]
        public async Task ALobbyIdWithLettersInItIsRefusedBeforeSteamIsAsked()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response =
                await Create(client, FakeSteamWebApiClient.OwnerTicket, lobbyId: "10977524000000000a");

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.Steam.AuthenticateCalls, Is.Zero);
        }

        [Test]
        public async Task AnOversizedTicketIsRefusedBeforeSteamIsAsked()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, new string('a', 8193));

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.Steam.AuthenticateCalls, Is.Zero);
        }

        [Test]
        public async Task ABlockedAccountCannotStartAMatch()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["MATCH_BLOCKED_STEAM_IDS"] = FakeSteamWebApiClient.OwnerSteamId;
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("blocked"));
            Assert.That(factory.Store.WriteCount, Is.Zero);
        }

        // ---- join ------------------------------------------------------------------------------

        [Test]
        public async Task TheGuestJoinsIntoSeatOne()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            HttpResponseMessage response = await Join(client, matchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());

            JsonElement body = await Body(response);
            Assert.Multiple(() =>
            {
                Assert.That(body.GetProperty("seat").GetInt32(), Is.EqualTo(1));
                Assert.That(body.GetProperty("matchId").GetString(), Is.EqualTo(matchId.ToString()));
                Assert.That(body.GetProperty("protocolVersion").GetInt32(), Is.EqualTo(2));
                Assert.That(body.GetProperty("websocketUrl").GetString(),
                    Is.EqualTo(SteamServerFactory.WebsocketUrl));
                Assert.That(body.GetProperty("joinCredential").GetString(),
                    Has.Length.EqualTo(CredentialEncoding.CredentialCharacters));
            });
        }

        [Test]
        public async Task SomeoneWithNoSeatCannotJoin()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            HttpResponseMessage response =
                await Join(client, matchId, FakeSteamWebApiClient.OutsiderTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("not_lobby_member"));
        }

        [Test]
        public async Task JoiningAMatchThatWasNeverAllocatedIsNotFound()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response =
                await Join(client, Guid.NewGuid(), FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
            Assert.That(await ErrorCode(response), Is.EqualTo("not_found"));
        }

        [Test]
        public async Task JoiningAFinishedMatchIsAConflict()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            await factory.Store.TryStartMatchAsync(
                matchId, "replay", SteamServerFactory.Start, default);
            await factory.Store.TryCompleteMatchAsync(
                matchId, MatchStatus.Completed, 0, SteamServerFactory.Start, default);

            HttpResponseMessage response = await Join(client, matchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
        }

        // ---- throttling --------------------------------------------------------------------------

        [Test]
        public async Task TheSixthCreateInAWindowIsRateLimited()
        {
            using var factory = new SteamServerFactory();

            // Distinct lobbies, because the create endpoint is idempotent per lobby: repeating one lobby
            // would return the same match every time and could pass without the limiter existing.
            var lobbies = new string[6];
            for (int i = 0; i < lobbies.Length; i++)
            {
                lobbies[i] = "10977524000000010" + i.ToString();
                factory.Steam.Lobbies[lobbies[i]] = FakeSteamWebApiClient.ReadyLobby(lobbies[i]);
            }

            using HttpClient client = factory.CreateClient();

            for (int i = 0; i < 5; i++)
            {
                HttpResponseMessage allowed =
                    await Create(client, FakeSteamWebApiClient.OwnerTicket, lobbyId: lobbies[i]);
                Assert.That(allowed.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                    "request " + i + ": " + await allowed.Content.ReadAsStringAsync());
            }

            HttpResponseMessage refused =
                await Create(client, FakeSteamWebApiClient.OwnerTicket, lobbyId: lobbies[5]);

            Assert.That(refused.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
            Assert.That(await ErrorCode(refused), Is.EqualTo("rate_limited"));
        }

        [Test]
        public async Task AnEleventhBadTicketIsRefusedWithoutAskingSteam()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            factory.Steam.ResetCounts();

            for (int attempt = 1; attempt <= AuthFailureThrottle.MaxFailures; attempt++)
            {
                HttpResponseMessage rejected = await Join(client, matchId, UnknownTicket);
                Assert.That(rejected.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized),
                    "attempt " + attempt);
            }

            HttpResponseMessage throttled = await Join(client, matchId, UnknownTicket);

            Assert.That(throttled.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
            Assert.That(await ErrorCode(throttled), Is.EqualTo("rate_limited"));
            Assert.That(factory.Steam.AuthenticateCalls, Is.EqualTo(AuthFailureThrottle.MaxFailures),
                "the throttled attempt must be answered without a round trip to Valve");
        }

        [Test]
        public async Task BehindATrustedProxyEachForwardedAddressGetsItsOwnBudget()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["MATCH_TRUST_FORWARDED_HEADERS"] = "true";
            using HttpClient client = factory.CreateClient();

            const string abuser = "203.0.113.10";
            const string bystander = "198.51.100.7";

            Guid matchId = await CreatedMatchId(
                await CreateFrom(client, FakeSteamWebApiClient.OwnerTicket, abuser));

            for (int attempt = 1; attempt <= AuthFailureThrottle.MaxFailures; attempt++)
            {
                HttpResponseMessage rejected = await JoinFrom(client, matchId, UnknownTicket, abuser);
                Assert.That(rejected.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized),
                    "attempt " + attempt);
            }

            HttpResponseMessage throttled = await JoinFrom(client, matchId, UnknownTicket, abuser);
            HttpResponseMessage innocent = await JoinFrom(client, matchId, UnknownTicket, bystander);

            Assert.Multiple(() =>
            {
                Assert.That(throttled.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
                Assert.That(innocent.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized),
                    "one abusive client behind a proxy must not lock out everyone else behind it");
            });
        }

        [Test]
        public async Task WithoutTrustedProxiesAForwardedAddressBuysNothing()
        {
            // The default. Without it, any caller could pick their own rate-limit partition by writing a
            // header, which would make both the limiter and the throttle decorative.
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            const string abuser = "203.0.113.10";

            Guid matchId = await CreatedMatchId(
                await CreateFrom(client, FakeSteamWebApiClient.OwnerTicket, abuser));

            for (int attempt = 1; attempt <= AuthFailureThrottle.MaxFailures; attempt++)
            {
                await JoinFrom(client, matchId, UnknownTicket, abuser);
            }

            HttpResponseMessage renamed = await JoinFrom(client, matchId, UnknownTicket, "198.51.100.7");

            Assert.That(renamed.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
            Assert.That(await ErrorCode(renamed), Is.EqualTo("rate_limited"));
        }

        // ---- provenance and provider gating ------------------------------------------------------

        [Test]
        public async Task TheSeatFollowsTheTicketRatherThanAnythingInTheRequest()
        {
            using var factory = new SteamServerFactory();

            // One ticket string, pointed at whichever account the test wants. Nothing about the string
            // says who that is, and the request carries no Steam id at all, so the only way the server
            // can seat anyone is by asking Valve who presented it.
            const string ticket = "00ff00ff";
            factory.Steam.Identify(ticket, FakeSteamWebApiClient.OwnerSteamId);
            using HttpClient client = factory.CreateClient();

            string requestBody = JsonSerializer.Serialize(new
            {
                steamLobbyId = FakeSteamWebApiClient.LobbyId,
                ticket,
            });

            Assert.Multiple(() =>
            {
                Assert.That(requestBody, Does.Not.Contain(FakeSteamWebApiClient.OwnerSteamId));
                Assert.That(requestBody, Does.Not.Contain(FakeSteamWebApiClient.GuestSteamId));
            });

            using var content = new StringContent(requestBody, Encoding.UTF8, "application/json");
            HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());

            JsonElement body = await Body(response);
            Assert.That(body.GetProperty("seat").GetInt32(), Is.EqualTo(0));

            var matchId = Guid.Parse(body.GetProperty("matchId").GetString()!);
            var credentials = factory.Services.GetRequiredService<IMatchCredentialService>();
            CredentialValidation? validated = await credentials.ValidateAsync(
                matchId, body.GetProperty("joinCredential").GetString()!, default);

            Assert.That(validated, Is.Not.Null);
            Assert.That(validated!.SteamId, Is.EqualTo(FakeSteamWebApiClient.OwnerSteamId),
                "the credential was issued to the account the ticket resolved to");

            // The same bytes on the wire, a different account behind the ticket, a different verdict.
            factory.Steam.Identify(ticket, FakeSteamWebApiClient.GuestSteamId);
            using var repeat = new StringContent(requestBody, Encoding.UTF8, "application/json");
            HttpResponseMessage asGuest = await client.PostAsync(CreateRoute, repeat);

            Assert.That(asGuest.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(asGuest), Is.EqualTo("not_lobby_owner"));
        }

        [Test]
        public async Task WithoutTheSteamProviderTheEndpointsDoNotExist()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["LOBBY_PROVIDER"] = "Legacy";
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage create = await Create(client, FakeSteamWebApiClient.OwnerTicket);
            HttpResponseMessage join =
                await Join(client, Guid.NewGuid(), FakeSteamWebApiClient.OwnerTicket);

            Assert.Multiple(() =>
            {
                Assert.That(create.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
                Assert.That(join.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
            });
        }

        // ---- the real database -------------------------------------------------------------------

        [Test]
        public async Task AgainstPostgresACreateAndAJoinLeaveTheRowsTheProtocolPromises()
        {
            await using SteamServerFactory factory = await SteamServerFactory.PostgresAsync();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage createResponse = await Create(client, FakeSteamWebApiClient.OwnerTicket);
            Guid matchId = await CreatedMatchId(createResponse);
            JsonElement created = await Body(createResponse);

            HttpResponseMessage joinResponse = await Join(client, matchId, FakeSteamWebApiClient.GuestTicket);
            Assert.That(joinResponse.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await joinResponse.Content.ReadAsStringAsync());
            Assert.That((await Body(joinResponse)).GetProperty("seat").GetInt32(), Is.EqualTo(1));

            NpgsqlDataSource data = factory.Database!.DataSource;
            Assert.Multiple(async () =>
            {
                Assert.That(await Scalar<long>(data, "SELECT count(*) FROM matches"), Is.EqualTo(1));
                Assert.That(await Scalar<long>(data, "SELECT count(*) FROM match_players"), Is.EqualTo(2));
                Assert.That(
                    await Scalar<long>(data, "SELECT count(*) FROM match_join_credentials"),
                    Is.EqualTo(2));
                Assert.That(
                    await Scalar<string>(data, "SELECT steam_lobby_id FROM matches"),
                    Is.EqualTo(FakeSteamWebApiClient.LobbyId));
                Assert.That(await Scalar<string>(data, "SELECT status FROM matches"), Is.EqualTo("waiting"));
            });

            string credential = created.GetProperty("joinCredential").GetString()!;
            byte[] stored = await Scalar<byte[]>(
                data,
                "SELECT credential_hash FROM match_join_credentials WHERE steam_id = \u0027"
                + FakeSteamWebApiClient.OwnerSteamId + "\u0027");

            Assert.That(CredentialEncoding.TryFromBase64Url(credential, out byte[] raw), Is.True);
            Assert.That(stored, Is.Not.EqualTo(Encoding.UTF8.GetBytes(credential)),
                "the credential itself must never be what is stored");
            Assert.That(stored, Is.EqualTo(SHA256.HashData(raw)));
        }

        static async Task<T> Scalar<T>(NpgsqlDataSource data, string sql)
        {
            await using NpgsqlCommand command = data.CreateCommand(sql);
            object? value = await command.ExecuteScalarAsync();
            return (T)value!;
        }
    }
}
