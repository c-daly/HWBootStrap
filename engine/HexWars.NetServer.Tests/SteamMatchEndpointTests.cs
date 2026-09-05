using System.Net;
using System.Net.Http.Headers;
using System.Net.Http.Json;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Endpoints;
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
        public async Task ABodyThatIsNotJson_IsRefusedInTheFixedErrorShape()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            using var content = new StringContent("steamLobbyId=1", Encoding.UTF8, "text/plain");
            HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest),
                "the framework answers an unsupported content type with 415 and its own body shape");
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.Steam.AuthenticateCalls, Is.Zero);
        }

        [Test]
        public async Task ABodyThatIsBrokenJson_IsRefusedInTheFixedErrorShape()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            using var content = new StringContent("{\"ticket\":", Encoding.UTF8, "application/json");
            HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
        }

        [Test]
        public async Task AnEmptyBody_IsRefusedInTheFixedErrorShape()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            using var content = new StringContent(string.Empty, Encoding.UTF8, "application/json");
            HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
        }

        /// <summary>An otherwise valid request with a large unknown member. Unknown members are ignored by
        /// the serializer, which is exactly how a caller would smuggle work in: size is then the only
        /// reason left to refuse it.</summary>
        static string OversizedCreateBody() => JsonSerializer.Serialize(new
        {
            steamLobbyId = FakeSteamWebApiClient.LobbyId,
            ticket = FakeSteamWebApiClient.OwnerTicket,
            padding = new string((char)120, 20 * 1024),
        });

        [Test]
        public async Task AnOversizedBodyThatDeclaresItsLength_IsRefusedWithoutBeingReadAtAll()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            using var content = new StringContent(OversizedCreateBody(), Encoding.UTF8, "application/json");
            HttpResponseMessage response = await client.PostAsync(CreateRoute, content);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.RequestBodyBytesRead, Is.Zero,
                "a Content-Length past the cap is refused before a byte is pulled off the socket");
        }

        [Test]
        public async Task AnOversizedBodyThatHidesItsLength_IsRefusedOnceTheCapIsReached()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            // Chunked, so Content-Length says nothing and the cap has to be enforced by the read itself.
            // This is the case the header check cannot cover, and the one an abusive client would use.
            byte[] body = Encoding.UTF8.GetBytes(OversizedCreateBody());
            using var request = new HttpRequestMessage(HttpMethod.Post, CreateRoute)
            {
                Content = new StreamContent(new UndeclaredLengthStream(body)),
            };
            request.Content.Headers.ContentType = new MediaTypeHeaderValue("application/json");
            request.Headers.TransferEncodingChunked = true;

            HttpResponseMessage response = await client.SendAsync(request);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
            Assert.That(await ErrorCode(response), Is.EqualTo("invalid_request"));
            Assert.That(factory.RequestBodyBytesRead, Is.GreaterThan(0),
                "this body really did have to be read, or the bound below asserts nothing");
            Assert.That(factory.RequestBodyBytesRead, Is.LessThanOrEqualTo(JsonBody.DefaultMaxBytes + 1),
                "and the read must stop one byte past the cap rather than draining the whole body");
        }

        /// <summary>A stream that will not say how long it is, so HttpClient sends it chunked.</summary>
        sealed class UndeclaredLengthStream(byte[] content) : Stream
        {
            int _position;

            public override bool CanRead => true;
            public override bool CanSeek => false;
            public override bool CanWrite => false;
            public override long Length => throw new NotSupportedException();

            public override long Position
            {
                get => throw new NotSupportedException();
                set => throw new NotSupportedException();
            }

            public override int Read(byte[] buffer, int offset, int count)
            {
                int available = Math.Min(count, content.Length - _position);
                if (available <= 0) return 0;

                Array.Copy(content, _position, buffer, offset, available);
                _position += available;
                return available;
            }

            public override void Flush()
            {
            }

            public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
            public override void SetLength(long value) => throw new NotSupportedException();
            public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
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
        public async Task JoiningAnUnknownMatchWithABadTicket_IsUnauthorizedRatherThanNotFound()
        {
            // Authentication comes before the match lookup on purpose: a caller holding no valid ticket
            // must not be able to tell a match id that never existed from one that has finished, and a
            // throttled caller must not be able to make this server read the database at all.
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Join(client, Guid.NewGuid(), UnknownTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized));
            Assert.That(await ErrorCode(response), Is.EqualTo("authentication_failed"));
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

        // ---- an allocation that already exists ----------------------------------------------------

        [Test]
        public async Task AnExistingAllocationWhoseRosterHasChanged_IsAConflict()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));

            // The guest left and somebody else took the seat. The match already written names the first
            // guest, so honouring this request would hand the owner a credential for a game the person
            // now sitting in the lobby cannot join.
            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                guestSteamId: FakeSteamWebApiClient.OutsiderSteamId);

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict),
                await response.Content.ReadAsStringAsync());
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));

            IReadOnlyList<PersistedPlayer> players = await factory.Store.GetPlayersAsync(matchId, default);
            Assert.That(
                players.Select(player => player.SteamId),
                Does.Contain(FakeSteamWebApiClient.GuestSteamId),
                "the stored roster is the one the journal is keyed by and must not be quietly rewritten");
        }

        [Test]
        public async Task AnExistingAllocationWhoseSetupHasChanged_IsAConflict()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));

            factory.Steam.Lobbies[FakeSteamWebApiClient.LobbyId] = FakeSteamWebApiClient.ReadyLobby(
                setupWire: SteamLobbyRules.QuickMatchSetup(1234).ToWire());

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
        }

        [Test]
        public async Task AMatchThatEndsWhileTheJoinIsInFlight_IsAConflict()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));

            // Inside the seat lookup, which is after the join has already checked the status and before it
            // issues anything. Without the status check inside the issuing transaction this window hands
            // out a credential for a finished game.
            factory.Counting.BeforeGetPlayer = async (id, _) =>
            {
                await factory.Store.TryStartMatchAsync(id, "START-REPLAY", SteamServerFactory.Start, default);
                await factory.Store.TryCompleteMatchAsync(
                    id, MatchStatus.Completed, 0, SteamServerFactory.Start, default);
            };

            HttpResponseMessage response = await Join(client, matchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Conflict),
                await response.Content.ReadAsStringAsync());
            Assert.That(await ErrorCode(response), Is.EqualTo("lobby_changed"));
        }

        // ---- bans and protocol ---------------------------------------------------------------------

        [Test]
        public async Task APublisherBannedOwnerCannotStartAMatch()
        {
            using var factory = new SteamServerFactory();
            factory.Steam.Identify(
                FakeSteamWebApiClient.OwnerTicket, FakeSteamWebApiClient.OwnerSteamId, publisherBanned: true);
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("blocked"));
            Assert.That(factory.Steam.LobbyCalls, Is.Zero,
                "a banned account must be refused before this server spends a lobby read on them");
        }

        [Test]
        public async Task APublisherBannedGuestCannotJoin()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(await Create(client, FakeSteamWebApiClient.OwnerTicket));
            factory.Steam.Identify(
                FakeSteamWebApiClient.GuestTicket, FakeSteamWebApiClient.GuestSteamId, publisherBanned: true);

            HttpResponseMessage response = await Join(client, matchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.Forbidden));
            Assert.That(await ErrorCode(response), Is.EqualTo("blocked"));
        }

        [Test]
        public async Task AVacBannedPlayerIsStillAllowedToPlay()
        {
            // VAC is a signal for the games that consume it, and HexWars does not. Refusing on it would ban
            // players for something that happened in an unrelated title.
            using var factory = new SteamServerFactory();
            factory.Steam.Identify(
                FakeSteamWebApiClient.OwnerTicket, FakeSteamWebApiClient.OwnerSteamId, vacBanned: true);
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await Create(client, FakeSteamWebApiClient.OwnerTicket);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK),
                await response.Content.ReadAsStringAsync());
        }

        [Test]
        public async Task AMatchPersistedUnderAnotherProtocolCannotBeJoined()
        {
            using var factory = new SteamServerFactory();

            // A match this deployment did not write: the row carries protocol 1 and this server speaks 2,
            // so the credential it would hand back names a socket the client cannot talk on.
            CreateMatchResult seeded = await factory.Store.CreateMatchForLobbyAsync(
                new CreateMatchRequest(
                    FakeSteamWebApiClient.LobbyId,
                    FakeSteamWebApiClient.QuickSetupWire,
                    "hexwars-engine/1",
                    1,
                    SteamServerFactory.BuildId,
                    new[]
                    {
                        (FakeSteamWebApiClient.OwnerSteamId, 0),
                        (FakeSteamWebApiClient.GuestSteamId, 1),
                    },
                    SteamServerFactory.Start),
                default);

            using HttpClient client = factory.CreateClient();
            HttpResponseMessage response =
                await Join(client, seeded.Match.MatchId, FakeSteamWebApiClient.GuestTicket);

            Assert.That((int)response.StatusCode, Is.EqualTo(426), await response.Content.ReadAsStringAsync());
            Assert.That(await ErrorCode(response), Is.EqualTo("incompatible_version"));
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

        [Test]
        public async Task AForwardedAddressFromAnUntrustedPeer_IsIgnored()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["MATCH_TRUST_FORWARDED_HEADERS"] = "true";
            factory.Settings["MATCH_TRUSTED_PROXY_CIDRS"] = "10.4.0.0/16";
            factory.RemoteIpAddress = IPAddress.Parse("203.0.113.99");
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(
                await CreateFrom(client, FakeSteamWebApiClient.OwnerTicket, "198.51.100.1"));

            for (int attempt = 1; attempt <= AuthFailureThrottle.MaxFailures; attempt++)
            {
                await JoinFrom(client, matchId, UnknownTicket, "198.51.100.1");
            }

            // A different forwarded address, from the same untrusted peer. Because the header is not
            // believed, both requests partition on the peer, so this one is already spent.
            HttpResponseMessage renamed = await JoinFrom(client, matchId, UnknownTicket, "198.51.100.2");

            Assert.That(renamed.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
            Assert.That(await ErrorCode(renamed), Is.EqualTo("rate_limited"));
        }

        [Test]
        public async Task TrustingForwardedHeadersWithNoTrustedProxies_SaysSoAtStartup()
        {
            var captured = new CapturingLoggerProvider();
            using var factory = new SteamServerFactory();
            factory.Settings["MATCH_TRUST_FORWARDED_HEADERS"] = "true";
            factory.Logging = captured;
            using HttpClient client = factory.CreateClient();

            Assert.That(await client.GetStringAsync("/healthz"), Is.EqualTo("ok"));

            Assert.That(captured.Any("MATCH_TRUSTED_PROXY_CIDRS"), Is.True,
                "an empty trust list means any peer can name the client, which an operator has to be told");
        }

        [Test]
        public async Task AForwardedAddressFromATrustedProxy_IsHonoured()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["MATCH_TRUST_FORWARDED_HEADERS"] = "true";
            factory.Settings["MATCH_TRUSTED_PROXY_CIDRS"] = "10.4.0.0/16";
            factory.RemoteIpAddress = IPAddress.Parse("10.4.7.9");
            using HttpClient client = factory.CreateClient();

            Guid matchId = await CreatedMatchId(
                await CreateFrom(client, FakeSteamWebApiClient.OwnerTicket, "198.51.100.1"));

            for (int attempt = 1; attempt <= AuthFailureThrottle.MaxFailures; attempt++)
            {
                await JoinFrom(client, matchId, UnknownTicket, "198.51.100.1");
            }

            HttpResponseMessage sameClient = await JoinFrom(client, matchId, UnknownTicket, "198.51.100.1");
            HttpResponseMessage otherClient = await JoinFrom(client, matchId, UnknownTicket, "198.51.100.2");

            Assert.Multiple(() =>
            {
                Assert.That(sameClient.StatusCode, Is.EqualTo(HttpStatusCode.TooManyRequests));
                Assert.That(otherClient.StatusCode, Is.EqualTo(HttpStatusCode.Unauthorized),
                    "behind a trusted proxy the header names the client, so one abuser is not everyone");
            });
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

        [Test]
        public async Task TheAdvertisedWebsocketRouteAnswersInTheFixedErrorShape()
        {
            // Every create and join response points a client here. Until protocol v2 ships, following that
            // URL has to produce something the client can parse and act on rather than a 404 page.
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await client.GetAsync(SteamMatchEndpoints.WebSocketPath);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
            Assert.That(await ErrorCode(response), Is.EqualTo("service_unavailable"));
        }

        [Test]
        public async Task WithoutTheSteamProviderTheWebsocketRouteIsNotMappedEither()
        {
            using var factory = new SteamServerFactory();
            factory.Settings["LOBBY_PROVIDER"] = "Legacy";
            using HttpClient client = factory.CreateClient();

            HttpResponseMessage response = await client.GetAsync(SteamMatchEndpoints.WebSocketPath);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
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
