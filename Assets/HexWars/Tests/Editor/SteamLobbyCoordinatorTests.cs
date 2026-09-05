#nullable enable
using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    [TestFixture]
    public sealed class SteamLobbyCoordinatorTests
    {
        const string LocalId = "76561197960287930";
        const string RemoteId = "76561197960287931";
        const string StrangerId = "76561197960287999";
        const uint AppId = 480000;
        const int Protocol = 2;
        const string Build = "1.4.2";
        const int Seed = 1234;
        const string Wss = "wss://match.invalid/ws/v2";

        /// <summary>
        /// A realistic starting clock. Deadlines are relative and anchored on the first Tick that
        /// observes them, so nothing here primes the clock: a test ticks once to start a deadline
        /// running, and again past its duration to fire it.
        /// </summary>
        const double Clock = 1000;

        FakeSteamLobbyClient _steam = null!;
        FakeSteamMatchApi _api = null!;
        SteamLobbyConfig _config = null!;
        List<SteamLobbyStatus> _statuses = null!;
        List<SteamMatchTicket> _tickets = null!;
        SteamLobbyCoordinator _sut = null!;

        [SetUp]
        public void SetUp()
        {
            _steam = new FakeSteamLobbyClient
            {
                LocalSteamId = LocalId,
                LocalDisplayName = "LocalPlayer",
                AppId = AppId,
                RemoteOwnerSteamId = RemoteId,
                RemoteOwnerDisplayName = "RemoteOwner",
            };
            _api = new FakeSteamMatchApi();
            _config = new SteamLobbyConfig
            {
                AppId = AppId,
                ProtocolVersion = Protocol,
                ClientBuild = Build,
                RollSeed = () => Seed,
                SearchTimeoutSeconds = 8,
                AllocationTimeoutSeconds = 15,
            };
            _statuses = new List<SteamLobbyStatus>();
            _tickets = new List<SteamMatchTicket>();
            _sut = new SteamLobbyCoordinator(_steam, _api, _config, s => _statuses.Add(s), t => _tickets.Add(t));
        }

        [TearDown]
        public void TearDown()
        {
            _sut.Dispose();
        }

        // ----- rules ---------------------------------------------------------------------------

        [Test]
        public void QuickMatchSetup_IsTheFixedQuickV1Ruleset()
        {
            var setup = SteamLobbyRules.QuickMatchSetup(4242);

            Assert.That(setup.ToWire(), Is.EqualTo("0 9 7 0 4242 3 1 1 1 3 0"));
            Assert.That(SteamLobbyRules.QuickMatchSetup(0).Seed, Is.EqualTo(SteamLobbyRules.MinSeed));
            Assert.That(SteamLobbyRules.QuickMatchSetup(1000000).Seed, Is.EqualTo(SteamLobbyRules.MaxSeed));
            Assert.That(SteamLobbyRules.SetupEquals(setup, GameSetup.Parse(setup.ToWire())), Is.True);
            Assert.That(SteamLobbyRules.SetupEquals(setup, SteamLobbyRules.QuickMatchSetup(4243)), Is.False);
        }

        [Test]
        public void CompatibilityAndSearchMetadata_UseTheSharedLobbyKeys()
        {
            var required = SteamLobbyRules.RequiredSearchMetadata(AppId, Protocol, SteamLobbyRules.QuickRuleset);

            Assert.That(required[SteamLobbyKeys.App], Is.EqualTo("480000"));
            Assert.That(required[SteamLobbyKeys.Protocol], Is.EqualTo("2"));
            Assert.That(required[SteamLobbyKeys.Ruleset], Is.EqualTo("quick-v1"));
            Assert.That(required.Count, Is.EqualTo(3), "the build id must not narrow the search");

            Assert.That(SteamLobbyRules.IsCompatible(Meta(), AppId, Protocol), Is.True);
            Assert.That(SteamLobbyRules.IsCompatible(Meta(), 481000, Protocol), Is.False);
            Assert.That(SteamLobbyRules.IsCompatible(Meta(), AppId, 3), Is.False);
            Assert.That(SteamLobbyRules.IsCompatible(new Dictionary<string, string>(), AppId, Protocol), Is.False);
        }

        [Test]
        public void EveryPhase_HasItsExactPlayerVisibleLine()
        {
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.Searching), Is.EqualTo("Searching for a match\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.CreatingLobby), Is.EqualTo("Creating lobby\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.WaitingForPlayer), Is.EqualTo("Waiting for a player\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.WaitingForReady), Is.EqualTo("Waiting for both players to ready up"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.RequestingTicket), Is.EqualTo("Signing in with Steam\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.AllocatingMatch), Is.EqualTo("Allocating server match\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.JoiningMatch), Is.EqualTo("Joining match\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.MatchReady), Is.EqualTo("Connecting\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.Reconnecting), Is.EqualTo("Reconnecting\u2026"));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.SteamUnavailable), Is.EqualTo("Steam is unavailable \u2014 start the game through Steam."));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.BackendUnavailable), Is.EqualTo("The match service is unavailable \u2014 try again."));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.VersionMismatch), Is.EqualTo("Your game version is out of date \u2014 update HexWars in Steam."));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.Failed), Is.EqualTo("Could not start the match."));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.Cancelled), Is.EqualTo("Cancelled."));
            Assert.That(SteamLobbyMessages.For(SteamLobbyPhase.Idle), Is.Empty);
        }

        // ----- steam availability --------------------------------------------------------------

        [Test]
        public void EveryOperation_WhenSteamIsUnavailable_ReportsSteamUnavailable()
        {
            _steam.IsAvailable = false;

            _sut.QuickMatch();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.SteamUnavailable));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.SteamUnavailable));

            _sut.HostGame(GameSetup.Default, SteamLobbyVisibility.Private);
            _sut.InviteFriend();
            _sut.JoinInvited("109775240000000009");
            _sut.SetReady(true);
            _sut.Retry();
            _sut.Reconnect("match-1");

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.SteamUnavailable));
            Assert.That(_statuses, Has.Count.EqualTo(1), "an unchanged phase must not republish");
            Assert.That(_steam.CreateLobbyCalls, Is.Zero);
            Assert.That(_steam.RequestLobbyListCalls, Is.Zero);
            Assert.That(_steam.JoinLobbyCalls, Is.Zero);
            Assert.That(_steam.RequestAuthTicketCalls, Is.Zero);
            Assert.That(_api.Calls, Is.Empty);
        }

        // ----- quick match ---------------------------------------------------------------------

        [Test]
        public void QuickMatch_JoinsTheFirstCompatibleOpenLobby()
        {
            _steam.AvailableLobbies.Add(OpenLobby("109775240000000101"));
            _steam.AvailableLobbies.Add(OpenLobby("109775240000000102"));

            _sut.QuickMatch();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Searching));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.Searching));
            Pump();

            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));
            Assert.That(_steam.CreateLobbyCalls, Is.Zero);
            Assert.That(_sut.Status.LobbyId, Is.EqualTo("109775240000000101"));
            Assert.That(_sut.Status.IsOwner, Is.False);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
        }

        [Test]
        public void QuickMatch_SkipsFullAndAlreadyAllocatedLobbies()
        {
            var full = Meta();
            var allocated = Meta();
            allocated[SteamLobbyKeys.Match] = "match-9";
            var otherRuleset = Meta();
            otherRuleset[SteamLobbyKeys.Ruleset] = SteamLobbyRules.CustomRuleset;

            _steam.AvailableLobbies.Add(new SteamLobbySearchResult("109775240000000201", full, 2));
            _steam.AvailableLobbies.Add(new SteamLobbySearchResult("109775240000000202", allocated, 1));
            _steam.AvailableLobbies.Add(new SteamLobbySearchResult("109775240000000203", otherRuleset, 1));
            _steam.AvailableLobbies.Add(OpenLobby("109775240000000204"));

            _sut.QuickMatch();
            Pump();

            Assert.That(_sut.Status.LobbyId, Is.EqualTo("109775240000000204"));
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));
        }

        [Test]
        public void QuickMatch_WithNoOpenLobbies_CreatesAPublicQuickMatchLobby()
        {
            _sut.QuickMatch();
            Pump();

            Assert.That(_steam.CreateLobbyCalls, Is.EqualTo(1));
            Assert.That(_steam.LastCreateVisibility, Is.EqualTo(SteamLobbyVisibility.Public));
            Assert.That(_steam.LastCreateMaxMembers, Is.EqualTo(2));

            var metadata = _steam.GetLobby(_sut.Status.LobbyId!)!.Metadata;
            Assert.That(metadata[SteamLobbyKeys.App], Is.EqualTo("480000"));
            Assert.That(metadata[SteamLobbyKeys.Protocol], Is.EqualTo("2"));
            Assert.That(metadata[SteamLobbyKeys.Build], Is.EqualTo(Build));
            Assert.That(metadata[SteamLobbyKeys.Ruleset], Is.EqualTo(SteamLobbyRules.QuickRuleset));
            Assert.That(metadata[SteamLobbyKeys.Setup], Is.EqualTo(SteamLobbyRules.QuickMatchSetup(Seed).ToWire()));
            Assert.That(metadata[SteamLobbyKeys.Name], Is.EqualTo("LocalPlayer"));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
            Assert.That(_sut.Status.IsOwner, Is.True);
            Assert.That(_sut.Status.OpponentName, Is.Null);
            Assert.That(Phases(), Is.EqualTo(new[]
            {
                SteamLobbyPhase.Searching, SteamLobbyPhase.CreatingLobby, SteamLobbyPhase.WaitingForPlayer,
            }));
        }

        [Test]
        public void QuickMatch_WhenTheSearchTimesOut_CreatesALobbyInstead()
        {
            _steam.AvailableLobbies.Add(OpenLobby("109775240000000301"));

            _sut.QuickMatch();
            _sut.Tick(Clock);            // the first tick starts the search deadline running
            _sut.Tick(Clock + 3);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Searching),
                "a deadline is counted from when it is first observed, not from an unprimed zero");

            _sut.Tick(Clock + 9);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.CreatingLobby));
            Pump();

            Assert.That(_steam.CreateLobbyCalls, Is.EqualTo(1));
            Assert.That(_steam.JoinLobbyCalls, Is.Zero, "the late search result must be ignored");
            Assert.That(_sut.Status.IsOwner, Is.True);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
        }

        // ----- hosting and invites -------------------------------------------------------------

        [Test]
        public void HostGame_CreatesALobbyWithTheCustomRulesetAndTheRequestedVisibility()
        {
            var setup = new GameSetup(GameMode.Territory, 11, 9, 20, 77, 5, 2, 2, 1, 2, true);

            _sut.HostGame(setup, SteamLobbyVisibility.Private);
            Pump();

            Assert.That(_steam.LastCreateVisibility, Is.EqualTo(SteamLobbyVisibility.Private));
            var metadata = _steam.GetLobby(_sut.Status.LobbyId!)!.Metadata;
            Assert.That(metadata[SteamLobbyKeys.Ruleset], Is.EqualTo(SteamLobbyRules.CustomRuleset));
            Assert.That(metadata[SteamLobbyKeys.Setup], Is.EqualTo(setup.Sanitized().ToWire()));
            Assert.That(_steam.RequestLobbyListCalls, Is.Zero);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
        }

        [Test]
        public void InviteFriend_CreatesAFriendsOnlyQuickLobbyAndOpensTheOverlay()
        {
            _sut.InviteFriend();
            Pump();

            var lobbyId = _sut.Status.LobbyId!;
            Assert.That(_steam.LastCreateVisibility, Is.EqualTo(SteamLobbyVisibility.FriendsOnly));
            Assert.That(_steam.OpenInviteOverlayCalls, Is.EqualTo(1));
            Assert.That(_steam.LastInviteOverlayLobbyId, Is.EqualTo(lobbyId));

            var metadata = _steam.GetLobby(lobbyId)!.Metadata;
            Assert.That(metadata[SteamLobbyKeys.Ruleset], Is.EqualTo(SteamLobbyRules.QuickRuleset));
            Assert.That(metadata[SteamLobbyKeys.Setup], Is.EqualTo(SteamLobbyRules.QuickMatchSetup(Seed).ToWire()));
            Assert.That(_steam.RequestLobbyListCalls, Is.Zero);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
        }

        [Test]
        public void AnAcceptedInvite_JoinsThatLobby()
        {
            const string lobbyId = "109775240000000801";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _steam.RaiseInviteAccepted(lobbyId);
            Pump();

            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
        }

        [Test]
        public void JoiningAnIncompatibleLobby_LeavesItAndReportsVersionMismatch()
        {
            const string lobbyId = "109775240000000401";
            var metadata = Meta();
            metadata[SteamLobbyKeys.Protocol] = "99";
            _steam.AvailableLobbies.Add(new SteamLobbySearchResult(lobbyId, metadata, 1));

            _sut.JoinInvited(lobbyId);
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.VersionMismatch));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.VersionMismatch));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));
            Assert.That(_sut.Status.LobbyId, Is.Null);
        }

        [Test]
        public void JoiningALobbyWithBothPlayersPresent_ReportsWaitingForReadyAndTheOpponent()
        {
            JoinRemoteLobby();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.WaitingForReady));
            Assert.That(_sut.Status.IsOwner, Is.False);
            Assert.That(_sut.Status.OpponentName, Is.EqualTo("RemoteOwner"));
            Assert.That(_sut.Status.CanReady, Is.True);
            Assert.That(_sut.Status.CanCancel, Is.True);
        }

        // ----- ready handshake and allocation --------------------------------------------------

        [Test]
        public void SetReady_WritesMemberDataAndTracksTheLocalFlag()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();

            _sut.SetReady(true);
            Assert.That(_sut.Status.LocalReady, Is.True);
            Pump();
            Assert.That(ReadyOf(lobbyId, LocalId), Is.EqualTo("1"));

            _sut.SetReady(false);
            Pump();
            Assert.That(_sut.Status.LocalReady, Is.False);
            Assert.That(ReadyOf(lobbyId, LocalId), Is.EqualTo("0"));
            Assert.That(_api.Calls, Is.Empty, "one sided readiness must not allocate");
        }

        [Test]
        public void WhenBothPlayersAreReady_TheOwnerAllocatesTheMatchAndPublishesIt()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _statuses.Clear();

            ReadyUpBothPlayers(lobbyId);

            Assert.That(_api.CreateMatchCalls, Is.EqualTo(1));
            var call = _api.Calls[0];
            Assert.That(call.LobbyId, Is.EqualTo(lobbyId));
            Assert.That(call.TicketHex, Is.EqualTo("0A1B2C3D"));
            Assert.That(call.SetupWire, Is.EqualTo(SteamLobbyRules.QuickMatchSetup(Seed).ToWire()));

            Assert.That(_steam.GetLobby(lobbyId)!.Metadata[SteamLobbyKeys.Match], Is.EqualTo("match-0001"));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.MatchReady));
            Assert.That(_tickets, Has.Count.EqualTo(1));
            Assert.That(_tickets[0].MatchId, Is.EqualTo("match-0001"));
            Assert.That(_tickets[0].WebsocketUrl, Is.EqualTo(Wss));
            Assert.That(_tickets[0].JoinCredential, Is.EqualTo("owner-credential"));
            Assert.That(_tickets[0].Seat, Is.EqualTo(0));
            Assert.That(Phases(), Is.EqualTo(new[]
            {
                SteamLobbyPhase.WaitingForReady, SteamLobbyPhase.RequestingTicket,
                SteamLobbyPhase.AllocatingMatch, SteamLobbyPhase.MatchReady,
            }));
        }

        [Test]
        public void AGuestSeeingTheMatchKey_JoinsTheAllocatedMatch()
        {
            var lobbyId = JoinRemoteLobby();
            _api.JoinResults.Enqueue(SteamMatchApiResult.Success("match-7", Wss, "guest-credential", 1));
            _statuses.Clear();

            _steam.SetRemoteLobbyData(lobbyId, SteamLobbyKeys.Match, "match-7");
            Pump();

            Assert.That(_api.JoinMatchCalls, Is.EqualTo(1));
            Assert.That(_api.CreateMatchCalls, Is.Zero, "a guest never allocates");
            Assert.That(_api.Calls[0].MatchId, Is.EqualTo("match-7"));
            Assert.That(_api.Calls[0].TicketHex, Is.EqualTo("0A1B2C3D"));
            Assert.That(_tickets, Has.Count.EqualTo(1));
            Assert.That(_tickets[0].Seat, Is.EqualTo(1));
            Assert.That(_tickets[0].JoinCredential, Is.EqualTo("guest-credential"));
            Assert.That(Phases(), Is.EqualTo(new[]
            {
                SteamLobbyPhase.RequestingTicket, SteamLobbyPhase.JoiningMatch, SteamLobbyPhase.MatchReady,
            }));
        }

        // ----- error mapping -------------------------------------------------------------------

        [Test]
        public void AnIncompatibleVersionResponse_ReportsVersionMismatch()
        {
            AllocateWith(SteamMatchApiResult.Failure(426, SteamMatchErrorCodes.IncompatibleVersion,
                "Your game version is not compatible with this server."));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.VersionMismatch));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.VersionMismatch));
        }

        [TestCase(503, SteamMatchErrorCodes.ServiceUnavailable)]
        [TestCase(429, SteamMatchErrorCodes.RateLimited)]
        [TestCase(0, null)]
        public void AMatchServiceOutage_ReportsBackendUnavailableAndOffersRetry(long status, string? code)
        {
            AllocateWith(code == null
                ? SteamMatchApiResult.NetworkFailure()
                : SteamMatchApiResult.Failure(status, code, "Try again shortly."));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);
        }

        [Test]
        public void ALobbyChangedResponse_ReturnsToWaitingForReadyWithTheServerMessage()
        {
            var lobbyId = AllocateWith(SteamMatchApiResult.Failure(409, SteamMatchErrorCodes.LobbyChanged,
                "The lobby changed. Check that both players are ready and try again."));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
            Assert.That(_sut.Status.Message, Is.EqualTo("The lobby changed. Check that both players are ready and try again."));
            Assert.That(_sut.Status.LocalReady, Is.False, "the owner drops its own ready so the retry is deliberate");
            Assert.That(ReadyOf(lobbyId, LocalId), Is.EqualTo("0"));
            Assert.That(_sut.Status.CanReady, Is.True);
        }

        [TestCase(401, SteamMatchErrorCodes.AuthenticationFailed, "Steam sign-in could not be verified.")]
        [TestCase(403, SteamMatchErrorCodes.OwnershipMissing, "This Steam account does not own HexWars.")]
        [TestCase(403, SteamMatchErrorCodes.NotLobbyOwner, "Only the lobby owner can start the match.")]
        [TestCase(403, SteamMatchErrorCodes.Blocked, "You cannot start a match right now.")]
        public void AnyOtherResponse_ReportsFailedWithTheServerMessage(long status, string code, string message)
        {
            AllocateWith(SteamMatchApiResult.Failure(status, code, message));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_sut.Status.Message, Is.EqualTo(message));
        }

        // ----- timeouts, cancel, retry ---------------------------------------------------------

        [Test]
        public void AnAllocationThatNeverAnswers_IsCancelledAndReportsBackendUnavailable()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _api.Deferred = true;

            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.AllocatingMatch));

            _sut.Tick(Clock);
            _sut.Tick(Clock + 16);

            Assert.That(_api.CancelCalls, Is.EqualTo(1));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);

            _api.CompletePending();
            Pump();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_tickets, Is.Empty);
        }

        [Test]
        public void Cancel_LeavesTheLobbyAndReturnsToIdleOnTheNextTick()
        {
            CreateOwnedLobbyWithOpponent();

            _sut.Cancel();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Cancelled));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.Cancelled));
            Assert.That(_sut.Status.LobbyId, Is.Null);
            Assert.That(_sut.Status.OpponentName, Is.Null);
            Assert.That(_api.CancelCalls, Is.EqualTo(1));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));

            _sut.Tick(Clock);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Idle));
            Assert.That(_sut.Status.CanCancel, Is.False);
        }

        [Test]
        public void Cancel_WorksFromTheSearchPhaseToo()
        {
            _steam.AvailableLobbies.Add(OpenLobby("109775240000000701"));
            _sut.QuickMatch();

            _sut.Cancel();
            Pump();
            _sut.Tick(Clock);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Idle));
            Assert.That(_steam.JoinLobbyCalls, Is.Zero, "the late search result must be ignored");
            Assert.That(_steam.CreateLobbyCalls, Is.Zero);
        }

        [Test]
        public void AnAuthTicketArrivingAfterCancel_IsIgnored()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _sut.SetReady(true);
            _steam.SetRemoteMemberData(lobbyId, RemoteId, SteamLobbyKeys.MemberReady, "1");
            _steam.Pump();
            _steam.Pump();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.RequestingTicket));

            _sut.Cancel();
            Pump();
            _sut.Tick(Clock);

            Assert.That(_api.Calls, Is.Empty);
            Assert.That(_tickets, Is.Empty);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Idle));
        }

        [Test]
        public void ASecondMatchKeyNotification_DoesNotStartASecondJoin()
        {
            var lobbyId = JoinRemoteLobby();
            _api.JoinResults.Enqueue(SteamMatchApiResult.Success("match-7", Wss, "guest-credential", 1));

            _steam.SetRemoteLobbyData(lobbyId, SteamLobbyKeys.Match, "match-7");
            Pump();
            Assert.That(_api.JoinMatchCalls, Is.EqualTo(1));

            _steam.SetRemoteLobbyData(lobbyId, SteamLobbyKeys.Match, "match-7");
            Pump();

            Assert.That(_api.JoinMatchCalls, Is.EqualTo(1));
            Assert.That(_tickets, Has.Count.EqualTo(1));
        }

        [Test]
        public void LobbyEventsForOtherLobbies_AreIgnored()
        {
            const string other = "109775240000000999";
            JoinRemoteLobby();
            _steam.AvailableLobbies.Add(OpenLobby(other));
            _statuses.Clear();

            _steam.SetRemoteLobbyData(other, SteamLobbyKeys.Match, "match-x");
            _steam.AddRemoteMember(other, StrangerId, "Stranger");
            _steam.RemoveRemoteMember(other, StrangerId);
            Pump();

            Assert.That(_api.Calls, Is.Empty);
            Assert.That(_statuses, Is.Empty);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
        }

        [Test]
        public void WhenTheOpponentLeaves_TheLobbyReturnsToWaitingForPlayerWithReadyFlagsCleared()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _sut.SetReady(true);
            Pump();
            Assert.That(_sut.Status.LocalReady, Is.True);

            _steam.RemoveRemoteMember(lobbyId, RemoteId);
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
            Assert.That(_sut.Status.LocalReady, Is.False);
            Assert.That(_sut.Status.RemoteReady, Is.False);
            Assert.That(_sut.Status.OpponentName, Is.Null);
            Assert.That(ReadyOf(lobbyId, LocalId), Is.EqualTo("0"));
        }

        [Test]
        public void WhenSteamPromotesUsToOwner_IsOwnerBecomesTrue()
        {
            var lobbyId = JoinRemoteLobby();
            Assert.That(_sut.Status.IsOwner, Is.False);

            _steam.SetLobbyOwner(lobbyId, LocalId);
            _steam.RemoveRemoteMember(lobbyId, RemoteId);
            Pump();

            Assert.That(_sut.Status.IsOwner, Is.True);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
        }

        [Test]
        public void Reconnect_RejoinsTheMatchWithAFreshCredential()
        {
            _api.JoinResults.Enqueue(SteamMatchApiResult.Success("match-42", Wss, "fresh-credential", 1));

            _sut.Reconnect("match-42");
            Pump();

            Assert.That(_api.JoinMatchCalls, Is.EqualTo(1));
            Assert.That(_api.Calls[0].MatchId, Is.EqualTo("match-42"));
            Assert.That(_tickets, Has.Count.EqualTo(1));
            Assert.That(_tickets[0].JoinCredential, Is.EqualTo("fresh-credential"));
            Assert.That(Phases(), Is.EqualTo(new[]
            {
                SteamLobbyPhase.RequestingTicket, SteamLobbyPhase.Reconnecting, SteamLobbyPhase.MatchReady,
            }));
        }

        [Test]
        public void Reconnect_MapsAFailureLikeEveryOtherApiCall()
        {
            _api.JoinResults.Enqueue(SteamMatchApiResult.NetworkFailure());

            _sut.Reconnect("match-42");
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_tickets, Is.Empty);
        }

        [Test]
        public void Retry_AfterABackendOutage_AllocatesAgainWithoutLeavingTheLobby()
        {
            var lobbyId = AllocateWith(SteamMatchApiResult.Failure(503, SteamMatchErrorCodes.ServiceUnavailable, "Try again shortly."));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));

            _sut.Retry();
            Pump();

            Assert.That(_steam.LeaveLobbyCalls, Is.Zero);
            Assert.That(_api.CreateMatchCalls, Is.EqualTo(2));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            Assert.That(_tickets, Has.Count.EqualTo(1));
        }

        [Test]
        public void AfterALobbyChangedResponse_ReadyingUpAgainAllocatesExactlyOnceMore()
        {
            AllocateWith(SteamMatchApiResult.Failure(409, SteamMatchErrorCodes.LobbyChanged, "The lobby changed."));
            Pump();
            Assert.That(_api.CreateMatchCalls, Is.EqualTo(1), "dropping our ready must stop a re-allocation loop");

            _sut.SetReady(true);
            Pump();

            Assert.That(_api.CreateMatchCalls, Is.EqualTo(2));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.MatchReady));
        }

        [Test]
        public void Retry_ClearsAStaleServerMessage()
        {
            AllocateWith(SteamMatchApiResult.Failure(409, SteamMatchErrorCodes.LobbyChanged, "The lobby changed."));
            Pump();

            _sut.Retry();
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.WaitingForReady));
            Assert.That(_api.CreateMatchCalls, Is.EqualTo(1));
        }

        [Test]
        public void EveryChangePublishesANewImmutableStatus()
        {
            _sut.QuickMatch();
            var first = _statuses[0];
            Pump();

            Assert.That(_statuses, Has.Count.GreaterThanOrEqualTo(3));
            Assert.That(first.Phase, Is.EqualTo(SteamLobbyPhase.Searching), "an earlier snapshot must not mutate");
            Assert.That(first.LobbyId, Is.Null);
            for (var i = 1; i < _statuses.Count; i++)
            {
                Assert.That(_statuses[i], Is.Not.SameAs(_statuses[i - 1]));
                Assert.That(_statuses[i].Matches(_statuses[i - 1]), Is.False, "no duplicate publication");
            }
            Assert.That(_sut.Status, Is.SameAs(_statuses[_statuses.Count - 1]));
        }

        [Test]
        public void AJoinIssuedBeforeTheSearchDeadline_IsNotOvertakenByIt()
        {
            const string lobbyId = "109775240000000901";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _sut.QuickMatch();
            _sut.Tick(Clock);
            _sut.Tick(Clock + 7.9);
            _steam.Pump();                 // the search result lands, and a join goes out
            _sut.Tick(Clock + 8);          // the moment the search deadline would have fired
            Pump();                        // the join completes

            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));
            Assert.That(_steam.CreateLobbyCalls, Is.Zero, "an in-flight join must end the search");
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            Assert.That(_steam.LeaveLobbyCalls, Is.Zero);
        }

        [Test]
        public void ALobbyJoinedAfterACancel_IsLeftAgain()
        {
            const string lobbyId = "109775240000000902";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _sut.QuickMatch();
            _steam.Pump();                 // the search result lands, and a join goes out
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));

            _sut.Cancel();
            Assert.That(_steam.LeaveLobbyCalls, Is.Zero, "nothing is held yet");

            Pump();                        // the join lands behind the cancel
            _sut.Tick(Clock);

            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1), "an abandoned join must not strand a lobby");
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Idle));
        }

        [Test]
        public void AnInviteAcceptedOutsideIdle_IsIgnored()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            var joinsBefore = _steam.JoinLobbyCalls;

            _steam.AvailableLobbies.Add(OpenLobby("109775240000000903"));
            _steam.RaiseInviteAccepted("109775240000000903");
            Pump();

            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(joinsBefore),
                "an invite must not tear down a match that is already under way");
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            Assert.That(_tickets, Has.Count.EqualTo(1));
        }

        [Test]
        public void WhenTheLobbyMetadataCannotBePublished_TheLobbyIsLeftAndTheFlowFails()
        {
            _steam.FailNextSetLobbyData = true;

            _sut.QuickMatch();
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.PublishFailed));
            Assert.That(_sut.Status.CanRetry, Is.True);
            Assert.That(_sut.Status.LobbyId, Is.Null);
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));
        }

        [Test]
        public void AMatchKeyWriteThatFailsTwice_AbandonsTheAllocationAndLeavesTheLobby()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _steam.FailSetLobbyDataForKey = SteamLobbyKeys.Match;
            var leftBefore = _steam.LeaveLobbyCalls;

            ReadyUpBothPlayers(lobbyId);
            Assert.That(_api.CreateMatchCalls, Is.EqualTo(1));
            Assert.That(_tickets, Is.Empty, "a guest cannot see a match nobody published");

            _sut.Tick(Clock);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.AllocatingMatch));

            _sut.Tick(Clock + 1);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_sut.Status.Message, Is.EqualTo(SteamLobbyMessages.PublishFailed));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(leftBefore + 1));
            Assert.That(_tickets, Is.Empty);
        }

        [Test]
        public void AMatchKeyWriteThatSucceedsOnRetry_StillHandsOverTheTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _steam.FailNextSetLobbyData = true;   // only the first hw_match write is refused

            ReadyUpBothPlayers(lobbyId);
            Assert.That(_tickets, Is.Empty);

            _sut.Tick(Clock);

            Assert.That(_tickets, Has.Count.EqualTo(1));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.MatchReady));
            Assert.That(_steam.GetLobby(lobbyId)!.Metadata[SteamLobbyKeys.Match], Is.Not.Empty);
        }

        [Test]
        public void EveryMatchServiceExchange_ReleasesItsAuthTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            var cancelsBefore = _steam.CancelAuthTicketCalls;

            ReadyUpBothPlayers(lobbyId);

            Assert.That(_steam.RequestAuthTicketCalls, Is.EqualTo(1));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(cancelsBefore + 1),
                "a Web API ticket is spent by one exchange and must not outlive it");
        }

        [Test]
        public void Dispose_ReleasesTheWholeSession()
        {
            _sut.QuickMatch();
            Pump();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));

            _sut.Dispose();

            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1));
            Assert.That(_api.CancelCalls, Is.EqualTo(1));
            Assert.That(_steam.HasEventSubscribers, Is.False);
        }

        [Test]
        public void Detach_StopsSteamEventsButKeepsTheLobby()
        {
            CreateOwnedLobbyWithOpponent();

            _sut.Detach();

            Assert.That(_steam.HasEventSubscribers, Is.False);
            Assert.That(_steam.LeaveLobbyCalls, Is.Zero, "the lobby is released by Dispose, not by Detach");

            _steam.AvailableLobbies.Add(OpenLobby("109775240000000904"));
            _steam.RaiseInviteAccepted("109775240000000904");
            Pump();
            Assert.That(_steam.JoinLobbyCalls, Is.Zero);
        }

        // ----- auth ticket release ---------------------------------------------------------------

        [Test]
        public void AnAllocationTimeout_ReleasesTheAuthTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _steam.AutoDeliverAuthTickets = false;       // Steam never answers the ticket request
            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.RequestingTicket));

            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1),
                "an abandoned exchange must not leave a Web API ticket live on the account");
        }

        [Test]
        public void AnOpponentLeavingDuringTheTicketRequest_ReleasesTheAuthTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _steam.AutoDeliverAuthTickets = false;
            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.RequestingTicket));

            _steam.RemoveRemoteMember(lobbyId, RemoteId);
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1));
        }

        [Test]
        public void ATicketSteamRefusedToIssue_StillReleasesTheAuthTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _steam.NextTicket = null;

            ReadyUpBothPlayers(lobbyId);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_api.Calls, Is.Empty);
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1),
                "a handle can be live even when no ticket body came back");
        }

        [Test]
        public void AMatchServiceFailure_ReleasesTheAuthTicketExactlyOnce()
        {
            AllocateWith(SteamMatchApiResult.Failure(500, "InternalError", "Something went wrong."));

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1));
        }

        [Test]
        public void AnAbandonedMatchKeyWrite_ReleasesTheAuthTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _api.CreateResults.Enqueue(SteamMatchApiResult.Success("match-1", Wss, "cred-1", 0));
            _steam.FailSetLobbyDataForKey = SteamLobbyKeys.Match;   // hw_match can never be published

            ReadyUpBothPlayers(lobbyId);
            _sut.Tick(Clock);
            _sut.Tick(Clock);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Failed));
            Assert.That(_tickets, Is.Empty, "the owner must not walk into a match the guest cannot see");
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1),
                "the exchange released its ticket when it completed, and abandoning the write adds none");
        }

        // ----- overlapping Steam call results --------------------------------------------------

        [Test]
        public void AJoinStillInFlightWhenTheNextOneStarts_LeavesTheAbandonedLobby()
        {
            const string first = "109775240000000801";
            const string second = "109775240000000802";
            _steam.AvailableLobbies.Add(OpenLobby(first));
            _steam.AvailableLobbies.Add(OpenLobby(second));

            _sut.JoinInvited(first);      // join A is in flight
            _sut.Cancel();
            _sut.JoinInvited(second);     // join B starts before A answered

            Pump();

            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(2),
                "the second join must not replace the first registration");
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1),
                "the late success of the abandoned join must not strand its lobby");
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(second), "the live join still completes");
        }

        // ----- abandoning a half-published match ------------------------------------------------

        [Test]
        public void AnOpponentLeavingBetweenAFailedMatchKeyWriteAndItsRetry_AbandonsThePublication()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _api.CreateResults.Enqueue(SteamMatchApiResult.Success("match-1", Wss, "cred-1", 0));
            _steam.FailSetLobbyDataForKey = SteamLobbyKeys.Match;

            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.AllocatingMatch));
            var cancelsAfterAllocation = _steam.CancelAuthTicketCalls;

            _steam.RemoveRemoteMember(lobbyId, RemoteId);
            Pump();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));

            // The write would now succeed, which is exactly the trap: a retry landing here would hand
            // the owner a ticket for a match the guest already walked out of.
            _steam.FailSetLobbyDataForKey = null;
            var writesBefore = _steam.SetLobbyDataCalls;
            _sut.Tick(Clock);
            _sut.Tick(Clock + 1);
            _sut.Tick(Clock + 2);

            Assert.That(_steam.SetLobbyDataCalls, Is.EqualTo(writesBefore), "hw_match must not be written");
            Assert.That(_tickets, Is.Empty, "no ticket for a match with nobody in it");
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForPlayer));
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId), "the lobby stays open for the next player");
            Assert.That(_steam.CancelAuthTicketCalls, Is.GreaterThan(cancelsAfterAllocation),
                "the abandoned allocation releases its ticket");
            Assert.That(_steam.GetLobby(lobbyId)!.Metadata.ContainsKey(SteamLobbyKeys.Match), Is.False);
        }

        // ----- Steam calls that were never issued ----------------------------------------------

        [Test]
        public void ACreateLobbyThatFailsOutright_ReportsBackendUnavailableAndOffersRetry()
        {
            _steam.FailNextCreateLobby = true;

            _sut.HostGame(GameSetup.Default, SteamLobbyVisibility.Public);
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);
            Assert.That(_sut.Status.LobbyId, Is.Null);
        }

        [Test]
        public void AJoinLobbyThatFailsOutright_ReportsBackendUnavailableAndOffersRetry()
        {
            const string lobbyId = "109775240000000811";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));
            _steam.FailNextJoinLobby = true;

            _sut.JoinInvited(lobbyId);
            Pump();

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);
        }

        [Test]
        public void ACreateLobbyThatNeverAnswers_TimesOutIntoBackendUnavailable()
        {
            _sut.HostGame(GameSetup.Default, SteamLobbyVisibility.Public);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.CreatingLobby));

            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds - 0.1);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.CreatingLobby));

            _sut.Tick(Clock + _config.AllocationTimeoutSeconds);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);

            Pump();   // the late create must not resurrect the abandoned attempt
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1), "the lobby Steam did create is not stranded");
        }

        // ----- starting an operation over one in flight -----------------------------------------

        [Test]
        public void StartingAnOperationDuringAnExchange_CancelsTheRequestAndReleasesTheTicket()
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _api.Deferred = true;
            ReadyUpBothPlayers(lobbyId);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.AllocatingMatch));
            Assert.That(_steam.CancelAuthTicketCalls, Is.Zero);

            _sut.Reconnect("match-9");

            Assert.That(_api.CancelCalls, Is.EqualTo(1), "the abandoned request must be cancelled");
            Assert.That(_steam.CancelAuthTicketCalls, Is.EqualTo(1),
                "the abandoned exchange must not keep its Web API ticket");
        }

        [Test]
        public void AJoinThatNeverAnswers_TimesOutIntoBackendUnavailable()
        {
            const string lobbyId = "109775240000000821";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _sut.QuickMatch();
            _steam.Pump();   // the search answers, a join goes out, and Steam never answers it
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.Searching));

            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds);

            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.CanRetry, Is.True);
            Assert.That(_steam.CreateLobbyCalls, Is.Zero,
                "hosting on top of a join that may still land would strand a lobby");

            Pump();   // the late success belongs to an abandoned attempt
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));
        }

        [Test]
        public void ALateJoinSuccessForTheLobbyTheRetryAlreadyHolds_IsNotLeft()
        {
            const string lobbyId = "109775240000000831";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _sut.QuickMatch();
            _steam.Pump();                    // the search answers and a join goes out
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));

            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));

            _sut.Retry();                     // the retry searches again and finds the same lobby
            _steam.PumpAt(1);                 // its search answers ahead of the abandoned join
            _steam.PumpAt(1);                 // and so does its own join
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(2));
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            var held = _sut.Status.Phase;

            Assert.That(_steam.PendingCallbackCount, Is.EqualTo(1),
                "the abandoned join is the one callback still outstanding");

            _steam.Pump();                    // only now does the abandoned join report success

            Assert.That(_steam.LeaveLobbyCalls, Is.Zero,
                "leaving would eject the player from the lobby the coordinator believes it holds");
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            Assert.That(_sut.Status.Phase, Is.EqualTo(held));
        }

        [Test]
        public void ALateJoinSuccessForTheLobbyBeingEntered_IsAdoptedAsTheCurrentJoin()
        {
            var lobbyId = AdoptALateJoinSuccess(false);

            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId),
                "a confirmed membership must have an owner, or the lobby is occupied for ever");
            Assert.That(_sut.Status.Phase, Is.AnyOf(SteamLobbyPhase.WaitingForPlayer,
                                                    SteamLobbyPhase.WaitingForReady));
            Assert.That(_steam.LeaveLobbyCalls, Is.Zero);

            // The replacement join is answered as far as this coordinator is concerned, so nothing
            // times out on top of the lobby it now holds.
            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds * 4);

            Assert.That(_sut.Status.Phase, Is.Not.EqualTo(SteamLobbyPhase.BackendUnavailable));
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
        }

        [Test]
        public void AReplacementJoinFailingAfterAdoption_StillKeepsTheLobby()
        {
            var lobbyId = AdoptALateJoinSuccess(true);
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            var held = _sut.Status.Phase;

            _steam.Pump();   // the replacement join reports its own failure

            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId),
                "the adopted membership must not be torn down by the answer it replaced");
            Assert.That(_sut.Status.Phase, Is.EqualTo(held));
            Assert.That(_steam.LeaveLobbyCalls, Is.Zero);
            Assert.That(_steam.CreateLobbyCalls, Is.Zero,
                "hosting on top of the lobby already held would strand it");
        }

        // ----- helpers -------------------------------------------------------------------------

        void Pump()
        {
            _steam.PumpAll();
        }

        /// <summary>
        /// A join times out, the retry finds the same lobby, and only then does the abandoned join
        /// report success. <paramref name="replacementFails"/> scripts the retry own join to fail.
        /// </summary>
        string AdoptALateJoinSuccess(bool replacementFails)
        {
            const string lobbyId = "109775240000000841";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));

            _sut.QuickMatch();
            _steam.Pump();                     // the search answers and the first join goes out
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(1));

            _sut.Tick(Clock);
            _sut.Tick(Clock + _config.AllocationTimeoutSeconds);
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.BackendUnavailable));

            _sut.Retry();
            _steam.FailNextJoinLobby = replacementFails;
            _steam.PumpAt(1);                  // the retry search answers ahead of the abandoned join
            Assert.That(_steam.JoinLobbyCalls, Is.EqualTo(2));
            Assert.That(_sut.Status.LobbyId, Is.Null, "nothing is held while both joins are in flight");

            _steam.Pump();                     // the abandoned join now reports success
            return lobbyId;
        }

        List<SteamLobbyPhase> Phases()
        {
            var phases = new List<SteamLobbyPhase>(_statuses.Count);
            foreach (var status in _statuses) phases.Add(status.Phase);
            return phases;
        }

        static Dictionary<string, string> Meta()
        {
            return new Dictionary<string, string>(System.StringComparer.Ordinal)
            {
                { SteamLobbyKeys.App, "480000" },
                { SteamLobbyKeys.Protocol, "2" },
                { SteamLobbyKeys.Build, Build },
                { SteamLobbyKeys.Ruleset, SteamLobbyRules.QuickRuleset },
                { SteamLobbyKeys.Setup, SteamLobbyRules.QuickMatchSetup(4242).ToWire() },
                { SteamLobbyKeys.Name, "RemoteOwner" },
            };
        }

        static SteamLobbySearchResult OpenLobby(string lobbyId)
        {
            return new SteamLobbySearchResult(lobbyId, Meta(), 1);
        }

        string CreateOwnedLobbyWithOpponent()
        {
            _sut.QuickMatch();
            Pump();
            var lobbyId = _sut.Status.LobbyId!;
            _steam.AddRemoteMember(lobbyId, RemoteId, "Rival");
            Pump();
            Assert.That(_sut.Status.Phase, Is.EqualTo(SteamLobbyPhase.WaitingForReady));
            Assert.That(_sut.Status.IsOwner, Is.True);
            Assert.That(_sut.Status.OpponentName, Is.EqualTo("Rival"));
            return lobbyId;
        }

        string JoinRemoteLobby()
        {
            const string lobbyId = "109775240000000601";
            _steam.AvailableLobbies.Add(OpenLobby(lobbyId));
            _sut.QuickMatch();
            Pump();
            Assert.That(_sut.Status.LobbyId, Is.EqualTo(lobbyId));
            return lobbyId;
        }

        void ReadyUpBothPlayers(string lobbyId)
        {
            _sut.SetReady(true);
            _steam.SetRemoteMemberData(lobbyId, RemoteId, SteamLobbyKeys.MemberReady, "1");
            Pump();
        }

        string AllocateWith(SteamMatchApiResult scripted)
        {
            var lobbyId = CreateOwnedLobbyWithOpponent();
            _api.CreateResults.Enqueue(scripted);
            _statuses.Clear();
            ReadyUpBothPlayers(lobbyId);
            return lobbyId;
        }

        string? ReadyOf(string lobbyId, string steamId)
        {
            var snapshot = _steam.GetLobby(lobbyId);
            if (snapshot == null) return null;
            foreach (var member in snapshot.Members)
            {
                if (!string.Equals(member.SteamId, steamId, System.StringComparison.Ordinal)) continue;
                string? ready;
                return member.Data.TryGetValue(SteamLobbyKeys.MemberReady, out ready) ? ready : null;
            }
            return null;
        }
    }
}
