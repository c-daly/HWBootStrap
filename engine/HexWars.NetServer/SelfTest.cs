using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using HexWars.Engine;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Runtime;
using HexWars.NetServer.Steam;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Npgsql;

namespace HexWars.NetServer
{
    /// <summary>
    /// End-to-end proof with no Unity and no browser: spin up the real server in-process, connect two
    /// real WebSocket clients, and walk them through seat → start → a move → broadcast. Exit 0 on pass.
    /// </summary>
    static class SelfTest
    {
        const string Url = "http://127.0.0.1:5234";
        const string Ws = "ws://127.0.0.1:5234/ws?room=test";

        public static async Task<int> Run()
        {
            var builder = WebApplication.CreateBuilder();
            builder.WebHost.UseUrls(Url);
            builder.Logging.ClearProviders();
            var app = builder.Build();
            app.UseWebSockets();
            app.Map("/ws", LegacyWebSocketServer.Handle);
            await app.StartAsync();

            try
            {
                using var a = await Connect();
                string seatA = await Recv(a);               // SEAT 0 (room not full yet)
                string catalogRequestA = await Recv(a);     // CATALOG? while the room is waiting
                await Send(a, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));

                var lobby1 = LegacyWebSocketServer.OpenGamesSnapshot();      // host waiting -> the room is browsable
                bool lobbyListsWaitingRoom = lobby1.Count == 1 && lobby1[0].Code == "TEST";

                using var b = await Connect();
                string seatB = await Recv(b);               // SEAT 1
                string catalogRequestB = await Recv(b);
                await Send(b, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                string startA = await Recv(a);              // START ... (dealt to both once full)
                string startB = await Recv(b);

                var lobby2 = LegacyWebSocketServer.OpenGamesSnapshot();      // both seated -> started -> unlisted
                bool lobbyEmptiesOnStart = lobby2.Count == 0;

                await Send(a, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string applyA = await Recv(a);              // APPLY E 0 broadcast to both
                string applyB = await Recv(b);

                using var joiner = await Connect("ws://127.0.0.1:5234/ws?room=missing&join=1");
                string joinerSeat = await Recv(joiner);       // a joiner must never mint a room for a typo'd code
                bool joinOnlyTurnedAway = joinerSeat == "SEAT FULL";

                // Reconnect: kill A's socket, reconnect with the SAME token, and confirm the server
                // seats it back into P0 and re-deals START (the game must survive a background/refresh).
                using var ra = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA = await Recv(ra);               // SEAT 0
                string rCatalogRequestA = await Recv(ra);
                await Send(ra, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                using var rb = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-b");
                string rSeatB = await Recv(rb);                // SEAT 1
                string rCatalogRequestB = await Recv(rb);
                await Send(rb, NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates)));
                string rStartA = await Recv(ra);
                string rStartB = await Recv(rb);

                // C1: damage a unit BEFORE the drop — the default (9x7, seed 7) army placement lets
                // P0's Striker (unit 2) close to (3,0) and land a legal hit on P1's Striker (unit 5);
                // exact coordinates come from GameFactory's deterministic seed, not a guess.
                await Send(ra, NetProtocol.Cmd(new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0))));
                string rMoveApplyA = await Recv(ra);
                string rMoveApplyB = await Recv(rb);
                await Send(ra, NetProtocol.Cmd(new AttackUnit(PlayerId.Player0, 2, 5)));
                string rAttackApplyA = await Recv(ra);
                string rAttackApplyB = await Recv(rb);

                ra.Abort();                                     // simulate a dead socket (no clean close)
                using var ra2 = await Connect("ws://127.0.0.1:5234/ws?room=reconnect&token=tok-a");
                string rSeatA2 = await Recv(ra2);                // SEAT 0 again — same token, same seat
                string rStartA2 = await Recv(ra2);               // personal START re-deal

                // The re-deal must carry the move + the attack (not a fresh, undamaged deal) — assert
                // both the command count AND that fast-forwarding it (exactly what GameBootstrap.OnNetStart
                // does client-side) lands on the SAME state an independent direct replay of the identical
                // two commands produces from the same fresh start — the same field-level cross-check
                // TokenRejoinTests uses engine-side, applied here end-to-end through the real socket.
                var reDealt = ReplayFile.Read(rStartA2.Substring("START ".Length));
                bool reDealCarriesBothCommands = reDealt.Commands.Count == 2;

                var fastForwarded = reDealt.Start;
                bool fastForwardOk = true;
                foreach (var cmd in reDealt.Commands)
                {
                    var fr = GameEngine.Apply(fastForwarded, cmd);
                    if (!fr.Success) { fastForwardOk = false; break; }
                    fastForwarded = fr.NewState;
                }

                var freshP0PointsBefore = GameFactory.Build(GameSetup.Default).Player(PlayerId.Player0).Points;
                var direct = GameEngine.Apply(GameFactory.Build(GameSetup.Default), new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)));
                direct = GameEngine.Apply(direct.NewState, new AttackUnit(PlayerId.Player0, 2, 5));
                var expected = direct.NewState;

                bool targetPresenceMatches =
                    fastForwarded.Player(PlayerId.Player1).UnitsOnBoard.Any(u => u.Id == 5) ==
                    expected.Player(PlayerId.Player1).UnitsOnBoard.Any(u => u.Id == 5);
                bool pointsMatch = fastForwarded.Player(PlayerId.Player0).Points == expected.Player(PlayerId.Player0).Points;
                bool attackActuallyLanded = expected.Player(PlayerId.Player0).Points > freshP0PointsBefore // bounty proves the hit landed
                    || expected.Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 5).CurrentHp
                       < GameFactory.Build(GameSetup.Default).Player(PlayerId.Player1).UnitsOnBoard.Single(u => u.Id == 5).CurrentHp;
                bool reDealReflectsDamage = fastForwardOk && targetPresenceMatches && pointsMatch && attackActuallyLanded;

                await Send(ra2, NetProtocol.Cmd(new EndTurn(PlayerId.Player0)));
                string rApplyA2 = await Recv(ra2);
                string rApplyB = await Recv(rb);

                bool reconnectOk =
                    rSeatA == "SEAT 0" && rSeatB == "SEAT 1" &&
                    rCatalogRequestA == NetProtocol.CatalogRequest &&
                    rCatalogRequestB == NetProtocol.CatalogRequest &&
                    rStartA.StartsWith("START ") && rStartB.StartsWith("START ") &&
                    rMoveApplyA.StartsWith("APPLY M ") && rMoveApplyA == rMoveApplyB &&
                    rAttackApplyA.StartsWith("APPLY A ") && rAttackApplyA == rAttackApplyB &&
                    rSeatA2 == "SEAT 0" && rStartA2.StartsWith("START ") &&
                    reDealCarriesBothCommands && reDealReflectsDamage &&
                    rApplyA2 == "APPLY E 0" && rApplyB == "APPLY E 0";

                bool ok =
                    seatA == "SEAT 0" && seatB == "SEAT 1" &&
                    catalogRequestA == NetProtocol.CatalogRequest &&
                    catalogRequestB == NetProtocol.CatalogRequest &&
                    startA.StartsWith("START ") && startB.StartsWith("START ") &&
                    applyA == "APPLY E 0" && applyB == "APPLY E 0" &&
                    lobbyListsWaitingRoom && lobbyEmptiesOnStart && joinOnlyTurnedAway && reconnectOk;

                Console.WriteLine(ok
                    ? "SELFTEST PASS — two browsers can play head-to-head through this server"
                    : $"SELFTEST FAIL seatA='{seatA}' seatB='{seatB}' startA?={startA.StartsWith("START ")} applyA='{applyA}' applyB='{applyB}' lobby1={lobbyListsWaitingRoom} lobby2={lobbyEmptiesOnStart} joinOnly='{joinerSeat}' reconnectOk={reconnectOk} reDealCarriesBothCommands={reDealCarriesBothCommands} reDealReflectsDamage={reDealReflectsDamage} moveApply='{rMoveApplyA}' attackApply='{rAttackApplyA}'");

                await app.StopAsync();
                return ok ? 0 : 1;
            }
            catch (Exception ex)
            {
                Console.WriteLine("SELFTEST ERROR " + ex);
                await app.StopAsync();
                return 1;
            }
        }

        static async Task<ClientWebSocket> Connect(string url = Ws)
        {
            var c = new ClientWebSocket();
            await c.ConnectAsync(new Uri(url), CancellationToken.None);
            return c;
        }

        static async Task Send(ClientWebSocket c, string msg) =>
            await c.SendAsync(Encoding.UTF8.GetBytes(msg), WebSocketMessageType.Text, true, CancellationToken.None);

        static async Task<string> Recv(ClientWebSocket c)
        {
            var buf = new byte[16384];
            using var ms = new MemoryStream();
            WebSocketReceiveResult res;
            do
            {
                res = await c.ReceiveAsync(new ArraySegment<byte>(buf), CancellationToken.None);
                ms.Write(buf, 0, res.Count);
            } while (!res.EndOfMessage);
            return Encoding.UTF8.GetString(ms.ToArray());
        }

        // ---- the durable self-test -------------------------------------------

        /// <summary>A different port from the legacy self-test, so the two can run back to back.</summary>
        const string DurableUrl = "http://127.0.0.1:5235";
        const string DurableWs = "ws://127.0.0.1:5235/ws/v2";
        const string DurableCreateRoute = DurableUrl + "/api/v1/steam/matches";

        const string DurableDatabaseVariable = "HEXWARS_TEST_DATABASE_URL";

        internal const string SelfTestOwnerSteamId = "76561198000000001";
        internal const string SelfTestGuestSteamId = "76561198000000002";
        internal const string SelfTestOwnerTicket = "AA";
        internal const string SelfTestGuestTicket = "BB";
        internal const string SelfTestLobbyId = "109775240000000001";
        internal const string SelfTestAppId = "480000";
        internal const string SelfTestBuildId = "selftest-build";
        internal const string SelfTestProtocolVersion = "2";

        /// <summary>
        /// The same deterministic opening the rest of the repository uses: on seed 7 the Striker of Player0
        /// can close on one of Player1 and land a hit, so these three commands are legal and their effect on
        /// the state is real rather than a pass that changes nothing.
        /// </summary>
        static readonly Command[] DurableScript =
        {
            new MoveUnit(PlayerId.Player0, 2, new HexCoord(3, 0)),
            new AttackUnit(PlayerId.Player0, 2, 5),
            new EndTurn(PlayerId.Player0),
        };

        /// <summary>
        /// The restart proof, with no test runner and no Docker of its own: play a match on one process,
        /// stop it, start another on the same port over the same database, and carry on.
        ///
        /// Exit 3 rather than 0 when there is no database to run against. A self-test that quietly passed
        /// because it had nothing to test is worse than one that did not run, and a deploy pipeline needs to
        /// be able to tell those apart.
        /// </summary>
        public static async Task<int> RunDurable()
        {
            string? databaseUrl = Environment.GetEnvironmentVariable(DurableDatabaseVariable);
            if (string.IsNullOrWhiteSpace(databaseUrl))
            {
                Console.WriteLine("SELFTEST-DURABLE SKIPPED: set " + DurableDatabaseVariable);
                return 3;
            }

            // Before a connection is opened, because the first thing this does with the answer is drop the
            // public schema of whatever it was given, and there is no undo for that.
            if (!DisposableDatabaseGuard.IsDisposable(
                    databaseUrl, Environment.GetEnvironmentVariable, out string reason))
            {
                Console.WriteLine("SELFTEST-DURABLE REFUSED: "
                    + (DisposableDatabaseGuard.DatabaseName(databaseUrl) ?? "that target")
                    + " is not marked disposable");
                Console.WriteLine("  " + reason);
                return 3;
            }

            try
            {
                await ResetSchema(databaseUrl);
                return await DriveARestart(databaseUrl);
            }
            catch (Exception failure)
            {
                Console.WriteLine("SELFTEST-DURABLE FAIL " + failure);
                return 1;
            }
        }

        static async Task<int> DriveARestart(string databaseUrl)
        {
            var steam = new SelfTestSteamClient();
            Guid matchId;

            // ---- the process that played the opening ----
            WebApplication a = BuildDurableHost(databaseUrl, steam);
            await a.StartAsync();
            try
            {
                using var http = new HttpClient();

                (Guid allocated, int seatZero, string credentialZero) = await CreateMatch(http);
                matchId = allocated;
                (_, int seatOne, string credentialOne) =
                    await JoinMatch(http, matchId, SelfTestGuestTicket);

                if (seatZero != 0 || seatOne != 1)
                    throw new InvalidOperationException(
                        "seats came back as " + seatZero + " and " + seatOne + ", not 0 and 1");

                using ClientWebSocket zero = await SeatSocket(matchId, credentialZero, 0);
                await Expect(zero, NetProtocol.CatalogRequest);
                using ClientWebSocket one = await SeatSocket(matchId, credentialOne, 1);
                await Expect(one, NetProtocol.CatalogRequest);

                string catalog = NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
                await Send(zero, catalog);
                await Send(one, catalog);
                await Expect(zero, "START ");
                await Expect(one, "START ");

                foreach (Command command in DurableScript) await PlayOne(zero, one, command);

                RequireJournal(await Journal(databaseUrl, matchId), DurableScript.Length, "before the restart");
            }
            finally
            {
                // Stop rather than abandon: the port has to be free for the process that takes over, and a
                // graceful stop is the shutdown a deploy actually performs.
                await a.StopAsync();
                await a.DisposeAsync();
            }

            // ---- the process that took over ----
            WebApplication b = BuildDurableHost(databaseUrl, steam);
            await b.StartAsync();
            try
            {
                RecoveryState recovery = b.Services.GetRequiredService<RecoveryState>();
                if (recovery.Report is null)
                    throw new InvalidOperationException(
                        "startup recovery did not run: " + (recovery.Error?.Message ?? "no report"));
                if (recovery.Report.Verified != 1 || recovery.Report.Failed.Count != 0)
                    throw new InvalidOperationException(
                        "startup recovery verified " + recovery.Report.Verified + " and refused "
                        + recovery.Report.Failed.Count + ", expected 1 and 0");

                using var http = new HttpClient();

                // New credentials, from new tickets: nothing a client held before the restart is reused.
                (_, int seatZero, string credentialZero) =
                    await JoinMatch(http, matchId, SelfTestOwnerTicket);
                (_, int seatOne, string credentialOne) =
                    await JoinMatch(http, matchId, SelfTestGuestTicket);

                if (seatZero != 0 || seatOne != 1)
                    throw new InvalidOperationException(
                        "the seats did not survive the restart: " + seatZero + " and " + seatOne);

                using ClientWebSocket zero = await SeatSocket(matchId, credentialZero, 0);
                string reDeal = await Expect(zero, "START ");
                using ClientWebSocket one = await SeatSocket(matchId, credentialOne, 1);
                string reDealOne = await Expect(one, "START ");

                int replayed = CommandCount(reDeal);
                if (replayed != DurableScript.Length)
                    throw new InvalidOperationException(
                        "the re-deal carried " + replayed + " commands, expected " + DurableScript.Length);

                if (reDeal != reDealOne)
                    throw new InvalidOperationException("the two seats were dealt different games");

                string recovered = FastForward(reDeal);
                string expected = DirectReplayText(DurableScript.Length);
                if (recovered != expected)
                    throw new InvalidOperationException(
                        "the recovered position is not the one the commands produce from a fresh start");

                // The game carries on: a command from the other seat, applied on top of what was recovered.
                await PlayOne(zero, one, new EndTurn(PlayerId.Player1));

                RequireJournal(
                    await Journal(databaseUrl, matchId), DurableScript.Length + 1, "after the restart");
            }
            finally
            {
                await b.StopAsync();
                await b.DisposeAsync();
            }

            Console.WriteLine(
                "SELFTEST-DURABLE PASS - a match survived a process restart with its journal intact");
            return 0;
        }

        // ---- the host --------------------------------------------------------

        /// <summary>
        /// The real server, composed the way production composes it, with one edge replaced.
        ///
        /// Only the Steam partner API is swapped out - it is the one thing this process genuinely cannot
        /// own. Configuration binding, options validation, the migration runner, the endpoints, the
        /// credential service, the coordinator, the recovery pass and the socket are all the shipped code,
        /// which is the point: a self-test that reimplemented the pipeline would pass while the deployed
        /// server refused every request.
        /// </summary>
        static WebApplication BuildDurableHost(string databaseUrl, ISteamWebApiClient steam)
        {
            var builder = WebApplication.CreateBuilder(new WebApplicationOptions
            {
                // Named explicitly: describe-environment treats a missing environment as Production, and in
                // Production a plaintext MATCH_PUBLIC_BASE_URL is refused at startup - correctly.
                EnvironmentName = Environments.Development,
            });

            builder.WebHost.UseUrls(DurableUrl);
            builder.Logging.ClearProviders();

            builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
            {
                ["LOBBY_PROVIDER"] = "Steam",
                ["STEAM_APP_ID"] = SelfTestAppId,
                ["STEAM_PUBLISHER_WEB_API_KEY"] = "selftest-key",
                ["DATABASE_URL"] = databaseUrl,
                ["MATCH_PUBLIC_BASE_URL"] = DurableUrl,
                ["MATCH_BUILD_ID"] = SelfTestBuildId,
                ["MATCH_PROTOCOL_VERSION"] = SelfTestProtocolVersion,
            });

            builder.AddHexWarsServer();

            builder.Services.RemoveAll<ISteamWebApiClient>();
            builder.Services.AddSingleton(steam);

            WebApplication app = builder.Build();
            app.UseHexWarsServer();
            return app;
        }

        // ---- allocation over real HTTP ---------------------------------------

        static Task<(Guid MatchId, int Seat, string Credential)> CreateMatch(HttpClient http) =>
            PostForSeat(
                http,
                DurableCreateRoute,
                "{\"steamLobbyId\":\"" + SelfTestLobbyId + "\",\"ticket\":\"" + SelfTestOwnerTicket + "\"}",
                "create");

        static Task<(Guid MatchId, int Seat, string Credential)> JoinMatch(
            HttpClient http, Guid matchId, string ticket) =>
            PostForSeat(
                http,
                DurableCreateRoute + "/" + matchId.ToString() + "/join",
                "{\"ticket\":\"" + ticket + "\"}",
                "join");

        static async Task<(Guid MatchId, int Seat, string Credential)> PostForSeat(
            HttpClient http, string route, string body, string what)
        {
            using var content = new StringContent(body, Encoding.UTF8, "application/json");
            using HttpResponseMessage response = await http.PostAsync(route, content);

            string text = await response.Content.ReadAsStringAsync();
            if (!response.IsSuccessStatusCode)
                throw new InvalidOperationException(
                    what + " answered " + (int)response.StatusCode + ": " + text);

            JsonElement seat = JsonDocument.Parse(text).RootElement;
            return (
                Guid.Parse(seat.GetProperty("matchId").GetString()!),
                seat.GetProperty("seat").GetInt32(),
                seat.GetProperty("joinCredential").GetString()!);
        }

        // ---- the socket ------------------------------------------------------

        static async Task<ClientWebSocket> SeatSocket(Guid matchId, string credential, int seat)
        {
            ClientWebSocket socket = await Connect(DurableWs);
            await Send(socket, "AUTH " + matchId.ToString() + " " + credential);
            await Expect(socket, NetProtocol.Seat((PlayerId)seat));
            return socket;
        }

        /// <summary>The next frame, which must start with <paramref name="prefix"/>. PING is answered and
        /// skipped: the heartbeat is liveness, not part of any exchange being asserted on.</summary>
        static async Task<string> Expect(ClientWebSocket socket, string prefix)
        {
            while (true)
            {
                string frame = await Recv(socket);
                if (frame == "PING")
                {
                    await Send(socket, "PONG");
                    continue;
                }

                if (!frame.StartsWith(prefix, StringComparison.Ordinal))
                    throw new InvalidOperationException(
                        "expected a frame starting " + prefix + " and got " + frame);

                return frame;
            }
        }

        static async Task PlayOne(ClientWebSocket zero, ClientWebSocket one, Command command)
        {
            ClientWebSocket issuer = command.Issuer == PlayerId.Player0 ? zero : one;
            await Send(issuer, NetProtocol.Cmd(command));

            string broadcast = NetProtocol.Apply(command);
            string toZero = await Expect(zero, "APPLY ");
            string toOne = await Expect(one, "APPLY ");

            if (toZero != broadcast || toOne != broadcast)
                throw new InvalidOperationException(
                    "both seats must be told " + broadcast + ", got " + toZero + " and " + toOne);
        }

        // ---- the game, computed independently --------------------------------

        /// <summary>The state a fresh engine reaches by applying the first <paramref name="commands"/>
        /// scripted commands, with no server involved at all. Comparing the two halves of the restart
        /// against each other would pass for a server that dealt both of them the same wrong game.</summary>
        static string DirectReplayText(int commands)
        {
            IReadOnlyList<UnitTemplate> barracks =
                BarracksWire.Read(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
            GameState state = GameFactory.Build(GameSetup.Default, barracks, barracks);

            for (var i = 0; i < commands; i++)
            {
                Result applied = GameEngine.Apply(state, DurableScript[i]);
                if (!applied.Success)
                    throw new InvalidOperationException(
                        "the scripted game is not legal at step " + (i + 1) + ": " + applied.Reason);

                state = applied.NewState;
            }

            return ReplayFile.Write(state, Array.Empty<Command>());
        }

        /// <summary>What a client would be looking at after fast-forwarding a START frame.</summary>
        static string FastForward(string startFrame)
        {
            ReplayData dealt = ReplayFile.Read(startFrame.Substring("START ".Length));
            GameState state = dealt.Start;

            foreach (Command command in dealt.Commands)
            {
                Result applied = GameEngine.Apply(state, command);
                if (!applied.Success)
                    throw new InvalidOperationException(
                        "the re-deal does not replay: " + applied.Reason);

                state = applied.NewState;
            }

            return ReplayFile.Write(state, Array.Empty<Command>());
        }

        static int CommandCount(string startFrame) =>
            ReplayFile.Read(startFrame.Substring("START ".Length)).Commands.Count;

        // ---- the database, read directly -------------------------------------

        static async Task ResetSchema(string databaseUrl)
        {
            await using NpgsqlDataSource data =
                NpgsqlDataSource.Create(DatabaseUrl.ToNpgsqlConnectionString(databaseUrl));
            await using NpgsqlConnection connection = await data.OpenConnectionAsync();
            await using var reset = new NpgsqlCommand(
                "DROP SCHEMA IF EXISTS public CASCADE; CREATE SCHEMA public;", connection);

            await reset.ExecuteNonQueryAsync();
        }

        /// <summary>The command journal as it stands, read on a pool this self-test owns rather than
        /// through the server that wrote it.</summary>
        static async Task<IReadOnlyList<(int Sequence, string Wire)>> Journal(
            string databaseUrl, Guid matchId)
        {
            await using NpgsqlDataSource data =
                NpgsqlDataSource.Create(DatabaseUrl.ToNpgsqlConnectionString(databaseUrl));
            await using NpgsqlCommand read = data.CreateCommand(
                "SELECT sequence, command_wire FROM match_commands WHERE match_id = @match ORDER BY sequence");
            read.Parameters.AddWithValue("match", matchId);

            var rows = new List<(int Sequence, string Wire)>();
            await using NpgsqlDataReader reader = await read.ExecuteReaderAsync();
            while (await reader.ReadAsync()) rows.Add((reader.GetInt32(0), reader.GetString(1)));

            return rows;
        }

        /// <summary>Every acknowledged command, once each, in the order it was acknowledged.</summary>
        static void RequireJournal(
            IReadOnlyList<(int Sequence, string Wire)> journal, int expected, string when)
        {
            if (journal.Count != expected)
                throw new InvalidOperationException(
                    "the journal holds " + journal.Count + " command(s) " + when + ", expected " + expected);

            for (var i = 0; i < journal.Count; i++)
            {
                if (journal[i].Sequence != i + 1)
                    throw new InvalidOperationException(
                        "the journal is not contiguous " + when + ": row " + i + " holds sequence "
                        + journal[i].Sequence);

                string wire = i < DurableScript.Length
                    ? CommandWire.Write(DurableScript[i])
                    : CommandWire.Write(new EndTurn(PlayerId.Player1));

                if (journal[i].Wire != wire)
                    throw new InvalidOperationException(
                        "sequence " + journal[i].Sequence + " holds " + journal[i].Wire + " " + when
                        + ", expected " + wire);
            }
        }
    }

    /// <summary>
    /// The Steam partner API, scripted: two accounts, two tickets, and one ready lobby carrying the
    /// default setup under the custom ruleset.
    ///
    /// It lives here rather than in the test project because the server assembly cannot reference that
    /// project, and because this is the one dependency the self-test genuinely cannot own - everything
    /// else it exercises is the shipped code. Nothing here reaches the network, so the self-test runs on a
    /// laptop and in CI with no publisher key.
    ///
    /// The ruleset is custom rather than quick-v1 on purpose: quick-v1 pins every field but the seed, and
    /// the legality of the scripted opening is a fact about seed 7.
    /// </summary>
    internal sealed class SelfTestSteamClient : ISteamWebApiClient
    {
        public Task<SteamIdentity> AuthenticateUserTicketAsync(string ticketHex, CancellationToken ct) =>
            ticketHex switch
            {
                SelfTest.SelfTestOwnerTicket => Task.FromResult(Identity(SelfTest.SelfTestOwnerSteamId)),
                SelfTest.SelfTestGuestTicket => Task.FromResult(Identity(SelfTest.SelfTestGuestSteamId)),
                _ => Task.FromException<SteamIdentity>(
                    new SteamApiException(SteamFailure.AuthenticationFailed, "ticket rejected")),
            };

        public Task<bool> CheckAppOwnershipAsync(string steamId, CancellationToken ct) =>
            Task.FromResult(
                steamId == SelfTest.SelfTestOwnerSteamId || steamId == SelfTest.SelfTestGuestSteamId);

        public Task<SteamLobbySnapshot> GetLobbyDataAsync(string lobbyId, CancellationToken ct)
        {
            if (lobbyId != SelfTest.SelfTestLobbyId)
                return Task.FromException<SteamLobbySnapshot>(
                    new SteamApiException(SteamFailure.LobbyChanged, "lobby not found"));

            var metadata = new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [SteamLobbyKeys.App] = SelfTest.SelfTestAppId,
                [SteamLobbyKeys.Protocol] = SelfTest.SelfTestProtocolVersion,
                [SteamLobbyKeys.Build] = SelfTest.SelfTestBuildId,
                [SteamLobbyKeys.Ruleset] = SteamLobbyRules.CustomRuleset,
                [SteamLobbyKeys.Setup] = GameSetup.Default.ToWire(),
            };

            var members = new[]
            {
                Member(SelfTest.SelfTestOwnerSteamId),
                Member(SelfTest.SelfTestGuestSteamId),
            };

            return Task.FromResult(
                new SteamLobbySnapshot(lobbyId, SelfTest.SelfTestOwnerSteamId, members, metadata));
        }

        static SteamIdentity Identity(string steamId) => new(steamId, steamId, false, false);

        static SteamLobbyMember Member(string steamId) =>
            new(steamId, new Dictionary<string, string>(StringComparer.Ordinal)
            {
                [SteamLobbyKeys.MemberReady] = SteamLobbyKeys.ReadyTrue,
            });
    }
}
