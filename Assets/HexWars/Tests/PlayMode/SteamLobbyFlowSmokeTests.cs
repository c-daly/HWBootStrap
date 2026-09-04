#nullable enable
using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;

namespace HexWars.Presentation.PlayModeTests
{
    /// <summary>
    /// End-to-end cover for <see cref="SteamLobbyScreen"/> as a running MonoBehaviour: the coordinator is
    /// already unit tested in EditMode, so these tests are about the wiring the editor cannot check by
    /// compiling \u2014 that Start builds the panel and starts the flow, that LateUpdate ticks the coordinator,
    /// that the status line reads what the player should see, and that the buttons do what they say.
    /// <para>
    /// Both doubles are injected: <see cref="SteamRuntime.OverrideClientForTests"/> replaces Steam and
    /// <see cref="SteamLobbyScreen.ApiOverrideForTests"/> replaces the match service, so nothing here
    /// touches the network.
    /// </para>
    /// </summary>
    public sealed class SteamLobbyFlowSmokeTests
    {
        const string LocalId = "76561197960287930";
        const string RemoteId = "76561197960287931";
        const string LobbyId = "109775240000000101";
        const string Build = "1.4.2";
        const uint AppId = 480000;
        const int Protocol = 2;
        const string Wss = "wss://match.invalid/ws/v2";

        FakeSteamLobbyClient _steam = null!;
        FakeSteamMatchApi _api = null!;
        GameObject _host = null!;
        GameBootstrap _game = null!;

        [SetUp]
        public void SetUp()
        {
            // the handoff opens a websocket to an unroutable host on purpose; its failure is not the subject
            LogAssert.ignoreFailingMessages = true;

            _steam = new FakeSteamLobbyClient
            {
                LocalSteamId = LocalId,
                LocalDisplayName = "LocalPlayer",
                AppId = AppId,
                RemoteOwnerSteamId = RemoteId,
                RemoteOwnerDisplayName = "RemoteOwner",
            };
            _api = new FakeSteamMatchApi();
            SteamRuntime.OverrideClientForTests(_steam);
            SteamLobbyScreen.ApiOverrideForTests = _api;
            SteamMatchConfig.Invalidate();

            _host = new GameObject("SteamLobbyFlowHost");
            _game = _host.AddComponent<GameBootstrap>();
            // GameBootstrap.Start would deal a whole demo game; the lobby screen only needs the component,
            // and a disabled behaviour never gets its Start called
            _game.enabled = false;
        }

        [TearDown]
        public void TearDown()
        {
            SteamLobbyScreen.ApiOverrideForTests = null;
            if (_host != null) Object.DestroyImmediate(_host);
            _steam.Dispose();
            LogAssert.ignoreFailingMessages = false;
        }

        [UnityTest]
        public IEnumerator QuickMatch_WithAnOpenLobby_WalksTheStatusLineToTheMatchHandoff()
        {
            _steam.AvailableLobbies.Add(OpenLobby(LobbyId));

            var screen = SteamLobbyScreen.OpenQuickMatch(_game);
            yield return Settle(4);

            Assert.That(screen.CurrentStatusText, Is.EqualTo(SteamLobbyMessages.WaitingForReady));
            Assert.That(screen.CoordinatorForTests!.Status.OpponentName, Is.EqualTo("RemoteOwner"));

            // the owner readies up and allocates; this client is the guest that joins what it allocated
            _api.JoinResults.Enqueue(SteamMatchApiResult.Success("match-7", Wss, "guest-credential", 1));
            screen.ClickReadyForTests();
            _steam.SetRemoteLobbyData(LobbyId, SteamLobbyKeys.Match, "match-7");
            yield return Settle(4);

            Assert.That(screen.StatusTextsForTests, Has.Member(SteamLobbyMessages.Searching));
            Assert.That(screen.StatusTextsForTests, Has.Member(SteamLobbyMessages.WaitingForReady));
            Assert.That(screen.StatusTextsForTests, Has.Member(SteamLobbyMessages.JoiningMatch));
            Assert.That(screen.StatusTextsForTests, Has.Member(SteamLobbyMessages.MatchReady));
            Assert.That(_api.JoinMatchCalls, Is.EqualTo(1));
            Assert.That(_game.Networked, Is.True, "the ticket must reach GameBootstrap.StartSteamMatch");
            Assert.That(_game.GetComponent<SteamMatchConnection>(), Is.Not.Null);
        }

        [UnityTest]
        public IEnumerator Cancel_FromWaitingForPlayer_LeavesTheLobbyAndReturnsToTheTitle()
        {
            var screen = SteamLobbyScreen.OpenQuickMatch(_game);   // no open lobbies: it hosts one itself
            yield return Settle(4);

            Assert.That(screen.CurrentStatusText, Is.EqualTo(SteamLobbyMessages.WaitingForPlayer));
            Assert.That(_game.GetComponent<TitleScreen>(), Is.Null, "the title stepped aside to open this screen");

            screen.ClickCancelForTests();

            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1));
            Assert.That(_game.GetComponent<TitleScreen>(), Is.Not.Null);
        }

        [UnityTest]
        public IEnumerator AMatchServiceOutage_OffersRetry_AndRetrySucceedsOnceTheServiceIsBack()
        {
            _steam.AvailableLobbies.Add(OpenLobby(LobbyId));
            var screen = SteamLobbyScreen.OpenQuickMatch(_game);
            yield return Settle(4);

            _api.JoinResults.Enqueue(SteamMatchApiResult.Failure(
                503, SteamMatchErrorCodes.ServiceUnavailable, "Try again shortly."));
            screen.ClickReadyForTests();
            _steam.SetRemoteLobbyData(LobbyId, SteamLobbyKeys.Match, "match-7");
            yield return Settle(4);

            Assert.That(screen.CurrentStatusText, Does.Contain("unavailable"));
            Assert.That(screen.RetryOfferedForTests, Is.True);
            Assert.That(_game.Networked, Is.False);

            _api.JoinResults.Enqueue(SteamMatchApiResult.Success("match-7", Wss, "guest-credential", 1));
            screen.ClickRetryForTests();
            yield return Settle(4);

            Assert.That(_api.JoinMatchCalls, Is.EqualTo(2));
            Assert.That(screen.StatusTextsForTests, Has.Member(SteamLobbyMessages.MatchReady));
            Assert.That(_game.Networked, Is.True);
        }

        [UnityTest]
        public IEnumerator AnInviteToAnIncompatibleLobby_ShowsTheVersionMismatchLine()
        {
            var metadata = Metadata();
            metadata[SteamLobbyKeys.Protocol] = "99";
            _steam.AvailableLobbies.Add(new SteamLobbySearchResult(LobbyId, metadata, 1));

            var screen = SteamLobbyScreen.OpenInvited(_game, LobbyId);
            yield return Settle(4);

            Assert.That(screen.CurrentStatusText, Is.EqualTo(SteamLobbyMessages.VersionMismatch));
            Assert.That(_steam.LeaveLobbyCalls, Is.EqualTo(1), "an incompatible lobby must not be held");
            Assert.That(_game.Networked, Is.False);
        }

        // ----- helpers ---------------------------------------------------------------------------

        /// <summary>
        /// Runs <paramref name="frames"/> frames, delivering Steam callbacks at the top of each one just as
        /// <see cref="SteamRuntime"/> would; the screen ticks the coordinator in its own LateUpdate.
        /// </summary>
        IEnumerator Settle(int frames)
        {
            for (var i = 0; i < frames; i++)
            {
                yield return null;
                _steam.PumpAll();
            }
            yield return null;
        }

        static Dictionary<string, string> Metadata()
        {
            return new Dictionary<string, string>(System.StringComparer.Ordinal)
            {
                { SteamLobbyKeys.App, AppId.ToString(System.Globalization.CultureInfo.InvariantCulture) },
                { SteamLobbyKeys.Protocol, Protocol.ToString(System.Globalization.CultureInfo.InvariantCulture) },
                { SteamLobbyKeys.Build, Build },
                { SteamLobbyKeys.Ruleset, SteamLobbyRules.QuickRuleset },
                { SteamLobbyKeys.Setup, SteamLobbyRules.QuickMatchSetup(4242).ToWire() },
                { SteamLobbyKeys.Name, "RemoteOwner" },
            };
        }

        static SteamLobbySearchResult OpenLobby(string lobbyId)
        {
            return new SteamLobbySearchResult(lobbyId, Metadata(), 1);
        }
    }
}
