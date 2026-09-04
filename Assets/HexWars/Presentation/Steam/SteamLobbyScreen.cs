#nullable enable
using System;
using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The Steam matchmaking screen: one panel over the running demo that reports where the flow
    /// stands, lets both players ready up, and hands the finished ticket to <see cref="GameBootstrap"/>.
    /// <para>
    /// It owns no matchmaking logic of its own — every decision lives in
    /// <see cref="SteamLobbyCoordinator"/>, which is plain C# and unit tested. This class only builds
    /// the UI, pumps <see cref="SteamLobbyCoordinator.Tick"/> once per frame in <c>LateUpdate</c> (after
    /// <see cref="SteamRuntime"/> has pumped Steam callbacks in <c>Update</c>), and renders the latest
    /// <see cref="SteamLobbyStatus"/>.
    /// </para>
    /// <para>
    /// After the match starts the screen hides its canvas but stays alive as the credential broker:
    /// <see cref="SteamMatchConnection"/> asks it to reissue a join credential before a reconnect, which
    /// it answers by driving <see cref="SteamLobbyCoordinator.Reconnect"/>. It closes for good once the
    /// connection it serves is gone.
    /// </para>
    /// </summary>
    public sealed class SteamLobbyScreen : MonoBehaviour
    {
        /// <summary>Which entry point opened this screen. Decides the title and the first operation.</summary>
        enum Entry
        {
            QuickMatch,
            Invite,
            Host,
            Invited,
        }

        /// <summary>Shown when this build has no match-service URL at all (see <see cref="SteamMatchConfig"/>).</summary>
        public const string NotConfiguredMessage = "Match service not configured";

        /// <summary>Test seam: when set, the screen drives this API instead of attaching a live one.</summary>
        internal static ISteamMatchApi? ApiOverrideForTests;

        GameBootstrap _game = null!;
        GameObject? _canvasGo;
        SteamLobbyCoordinator? _coordinator;
        SteamMatchApiClient? _apiComponent;

        Entry _entry = Entry.QuickMatch;
        GameSetup _setup = GameSetup.Default;
        SteamLobbyVisibility _visibility = SteamLobbyVisibility.Public;
        string _invitedLobbyId = string.Empty;

        Text? _statusText;
        Text? _opponentText;
        Text? _readyStateText;
        Button? _readyButton;
        Button? _inviteButton;
        Button? _retryButton;
        Button? _cancelButton;
        Text? _readyButtonLabel;
        Text? _cancelButtonLabel;

        readonly List<string> _statusTexts = new List<string>();

        SteamLobbyStatus? _status;
        Action<SteamMatchTicket?>? _pendingRefresh;
        string _matchId = string.Empty;
        bool _handedOff;   // the ticket went to GameBootstrap; the UI is gone but the broker lives on
        bool _dead;        // Destroy is deferred to end of frame and a dying screen must stay quiet
        float _panelW = 620f;

        // ----- entry points ------------------------------------------------------------------------

        /// <summary>Find an open quick-match lobby, or host one when there is none.</summary>
        public static SteamLobbyScreen OpenQuickMatch(GameBootstrap game)
        {
            return Open(game, Entry.QuickMatch, GameSetup.Default, SteamLobbyVisibility.Public, string.Empty);
        }

        /// <summary>Create a friends-only lobby and open the Steam invite overlay on it.</summary>
        public static SteamLobbyScreen OpenInvite(GameBootstrap game)
        {
            return Open(game, Entry.Invite, GameSetup.Default, SteamLobbyVisibility.FriendsOnly, string.Empty);
        }

        /// <summary>Host a lobby that plays the configuration built in <see cref="SetupForm"/>.</summary>
        public static SteamLobbyScreen OpenHost(GameBootstrap game, GameSetup setup, SteamLobbyVisibility visibility)
        {
            return Open(game, Entry.Host, setup, visibility, string.Empty);
        }

        /// <summary>Join the lobby behind an accepted Steam invite.</summary>
        public static SteamLobbyScreen OpenInvited(GameBootstrap game, string lobbyId)
        {
            return Open(game, Entry.Invited, GameSetup.Default, SteamLobbyVisibility.Public, lobbyId);
        }

        static SteamLobbyScreen Open(GameBootstrap game, Entry entry, GameSetup setup,
                                     SteamLobbyVisibility visibility, string lobbyId)
        {
            var existing = game.GetComponent<SteamLobbyScreen>();
            if (existing != null) existing.Close();
            var screen = game.gameObject.AddComponent<SteamLobbyScreen>();
            screen._game = game;
            screen._entry = entry;
            screen._setup = setup;
            screen._visibility = visibility;
            screen._invitedLobbyId = lobbyId ?? string.Empty;
            return screen;
        }

        // ----- lifecycle ---------------------------------------------------------------------------

        void Start()
        {
            if (_game == null) _game = GetComponent<GameBootstrap>()!;
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>()!;
            Build();
            BeginFlow();
        }

        void Update()
        {
            if (_dead) return;

            if (_handedOff)
            {
                // the broker outlives the UI, but only for as long as the connection it serves
                if (_game == null || _game.GetComponent<SteamMatchConnection>() == null) ReleaseCoordinator();
                return;
            }

            // a real match started (a reconnect from elsewhere, say) — this screen is done
            if (_game != null && _game.State != null && !_game.DemoMode) { Close(); return; }

            if (DeviceInput.Allowed && Keyboard.current != null
                && Keyboard.current.escapeKey.wasPressedThisFrame)
            {
                OnCancel();
            }
        }

        /// <summary>Runs after <see cref="SteamRuntime"/> pumped Steam callbacks, so timeouts fire on fresh state.</summary>
        void LateUpdate()
        {
            if (_dead || _coordinator == null) return;
            _coordinator.Tick(Time.unscaledTimeAsDouble);
        }

        public void Close()
        {
            _dead = true;
            if (_canvasGo != null) Destroy(_canvasGo);
            _canvasGo = null;
            Destroy(this);
        }

        void OnDestroy()
        {
            _dead = true;
            var pending = _pendingRefresh;
            _pendingRefresh = null;
            if (pending != null) pending(null);   // never leave a reconnect waiting on a dead screen
            if (_coordinator != null)
            {
                // After a handoff the lobby still belongs to the running match, so only the Steam
                // subscriptions go; ReleaseAfterMatch (or ReleaseCoordinator) frees the rest when the
                // match itself ends. Any other teardown is the player leaving, so release everything.
                if (_handedOff) _coordinator.Detach();
                else _coordinator.Dispose();
                _coordinator = null;
            }
            if (_apiComponent != null) { Destroy(_apiComponent); _apiComponent = null; }
        }

        /// <summary>
        /// Frees a coordinator that stayed alive as the credential broker. Called from the
        /// return-to-title path, which is the last chance to leave the Steam lobby behind.
        /// </summary>
        internal static void ReleaseAfterMatch(GameBootstrap game)
        {
            if (game == null) return;
            var screen = game.GetComponent<SteamLobbyScreen>();
            if (screen != null) screen.ReleaseCoordinator();
        }

        void ReleaseCoordinator()
        {
            if (_coordinator != null) { _coordinator.Dispose(); _coordinator = null; }
            _handedOff = false;
            Close();
        }

        // ----- flow --------------------------------------------------------------------------------

        void BeginFlow()
        {
            var settings = SteamMatchConfig.Resolve();
            var api = ApiOverrideForTests;
            if (api == null)
            {
                // an unconfigured build must say so instead of failing at the first request
                if (!settings.IsConfigured) { ShowNotConfigured(); return; }
                _apiComponent = SteamMatchApiClient.Attach(gameObject, settings.BaseUrl, settings.ProtocolVersion);
                api = _apiComponent;
            }

            var steam = SteamRuntime.Client;
            var config = new SteamLobbyConfig
            {
                AppId = steam.AppId,
                ProtocolVersion = settings.ProtocolVersion,
                ClientBuild = Application.version,
                RollSeed = () => UnityEngine.Random.Range(SteamLobbyRules.MinSeed, SteamLobbyRules.MaxSeed + 1),
            };
            // No clock priming: coordinator deadlines are relative and start running on the first
            // LateUpdate Tick, so the game clock being long past zero here does not matter.
            _coordinator = new SteamLobbyCoordinator(steam, api, config, OnStatus, OnMatchReady);

            switch (_entry)
            {
                case Entry.Invite: _coordinator.InviteFriend(); break;
                case Entry.Host: _coordinator.HostGame(_setup, _visibility); break;
                case Entry.Invited: _coordinator.JoinInvited(_invitedLobbyId); break;
                default: _coordinator.QuickMatch(); break;
            }
        }

        void ShowNotConfigured()
        {
            _status = new SteamLobbyStatus(SteamLobbyPhase.BackendUnavailable, NotConfiguredMessage,
                                           null, null, false, false, false, null, false, false, false);
            _statusTexts.Add(NotConfiguredMessage);
            Refresh();
        }

        void OnStatus(SteamLobbyStatus status)
        {
            _status = status;
            _statusTexts.Add(TextFor(status));

            var pending = _pendingRefresh;
            if (pending != null && GivesUp(status.Phase))
            {
                _pendingRefresh = null;
                pending(null);
            }

            Refresh();
        }

        void OnMatchReady(SteamMatchTicket ticket)
        {
            if (_dead) return;
            _matchId = ticket.MatchId;

            // a reissue in flight consumes this ticket instead of starting a second match
            var pending = _pendingRefresh;
            if (pending != null) { _pendingRefresh = null; pending(ticket); return; }

            _handedOff = true;
            // The coordinator has done its job. Keep it for credential reissues, but take it off the
            // Steam events: from here an accepted invite belongs to the title screen alone.
            if (_coordinator != null) _coordinator.Detach();
            if (_canvasGo != null) _canvasGo.SetActive(false);
            _game.StartSteamMatch(ticket, RefreshCredential);
        }

        /// <summary>
        /// <see cref="SteamMatchConnection"/> asking for a fresh single-use join credential. Answers false
        /// when no reissue could start; otherwise a later <see cref="OnMatchReady"/> answers with the new
        /// ticket, and a terminal phase answers with null.
        /// </summary>
        bool RefreshCredential(Action<SteamMatchTicket?> onDone)
        {
            if (_dead || onDone == null || _coordinator == null) return false;
            if (string.IsNullOrEmpty(_matchId)) return false;
            _pendingRefresh = onDone;
            _coordinator.Reconnect(_matchId);
            return true;
        }

        static bool GivesUp(SteamLobbyPhase phase)
        {
            return phase == SteamLobbyPhase.Failed
                || phase == SteamLobbyPhase.VersionMismatch
                || phase == SteamLobbyPhase.BackendUnavailable
                || phase == SteamLobbyPhase.SteamUnavailable
                || phase == SteamLobbyPhase.Cancelled;
        }

        static string TextFor(SteamLobbyStatus status)
        {
            return string.IsNullOrEmpty(status.Message) ? SteamLobbyMessages.For(status.Phase) : status.Message;
        }

        // ----- buttons -----------------------------------------------------------------------------

        void OnReady()
        {
            if (_dead || _coordinator == null || _status == null) return;
            _coordinator.SetReady(!_status.LocalReady);
        }

        void OnInvite()
        {
            if (_dead || _status == null || string.IsNullOrEmpty(_status.LobbyId)) return;
            var client = SteamRuntime.ClientIfCreated;
            if (client != null) client.OpenInviteOverlay(_status.LobbyId!);
        }

        void OnRetry()
        {
            if (_dead || _coordinator == null) return;
            _coordinator.Retry();
        }

        void OnCancel()
        {
            if (_dead) return;
            if (_coordinator != null) _coordinator.Cancel();
            var game = _game;
            Close();
            TitleScreen.Reopen(game);
        }

        // ----- ui ----------------------------------------------------------------------------------

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("SteamLobbyCanvas", UiKit.OrderMenu, transform);
            _panelW = Mathf.Min(620f, AvailWidth() - 40f);   // the GameBrowser portrait clamp

            var panel = UiKit.Panel(_canvasGo.transform, "Panel", UiKit.Surface).gameObject;
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0.5f, 0.5f);
            prt.pivot = new Vector2(0.5f, 0.5f);
            prt.sizeDelta = new Vector2(_panelW, 380f);
            prt.anchoredPosition = Vector2.zero;

            UiKit.Label(panel.transform, Title(), 0f, -26f, _panelW, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);

            _statusText = UiKit.Label(panel.transform, string.Empty, 0f, -86f, _panelW - 60f, 34f,
                                      UiKit.SizeBody, TextAnchor.MiddleCenter, UiKit.TextDim);
            _statusText.horizontalOverflow = HorizontalWrapMode.Wrap;   // server messages can be long

            _opponentText = UiKit.Label(panel.transform, OpponentLine(null), 0f, -130f, _panelW - 60f, 26f,
                                        UiKit.SizeBody, TextAnchor.MiddleCenter);
            _readyStateText = UiKit.Label(panel.transform, string.Empty, 0f, -162f, _panelW - 60f, 22f,
                                          UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            _readyButton = UiKit.Button(panel.transform, "Ready", -110f, -202f, 200f, 46f, OnReady,
                                        UiKit.ButtonStyle.Cta);
            _readyButtonLabel = _readyButton.GetComponentInChildren<Text>();
            _inviteButton = UiKit.Button(panel.transform, "Invite Friend", 110f, -202f, 200f, 46f, OnInvite,
                                         UiKit.ButtonStyle.Primary);
            _retryButton = UiKit.Button(panel.transform, "Retry", -110f, -262f, 200f, 42f, OnRetry,
                                        UiKit.ButtonStyle.Primary);
            _cancelButton = UiKit.Button(panel.transform, "Cancel", 110f, -262f, 200f, 42f, OnCancel,
                                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            _cancelButtonLabel = _cancelButton.GetComponentInChildren<Text>();

            Refresh();
        }

        float AvailWidth()
        {
            var rt = _canvasGo != null ? _canvasGo.GetComponent<RectTransform>() : null;
            return rt != null && rt.rect.width > 0f ? rt.rect.width : 1200f;
        }

        string Title()
        {
            switch (_entry)
            {
                case Entry.Invite: return "Invite a Friend";
                case Entry.Invited: return "Joining a Friend";
                case Entry.Host: return "Host Game";
                default: return "Quick Match";
            }
        }

        static string OpponentLine(string? name)
        {
            return "Opponent: " + (string.IsNullOrEmpty(name) ? "—" : UiKit.Ellipsize(name!, 24));
        }

        void Refresh()
        {
            if (_dead) return;
            var status = _status;

            if (_statusText != null) _statusText.text = status == null ? string.Empty : TextFor(status);
            if (_opponentText != null) _opponentText.text = OpponentLine(status?.OpponentName);

            bool canReady = status != null && status.CanReady;
            bool showReady = status != null && (status.CanReady || status.LocalReady);
            if (_readyButton != null)
            {
                _readyButton.gameObject.SetActive(showReady);
                _readyButton.interactable = canReady;
                if (_readyButtonLabel != null)
                    _readyButtonLabel.text = status != null && status.LocalReady ? "Not ready" : "Ready";
            }
            if (_readyStateText != null)
            {
                _readyStateText.text = showReady && status != null
                    ? "You: " + ReadyWord(status.LocalReady) + "   ·   Opponent: " + ReadyWord(status.RemoteReady)
                    : string.Empty;
            }

            bool canInvite = status != null && status.IsOwner && !string.IsNullOrEmpty(status.LobbyId);
            if (_inviteButton != null) _inviteButton.gameObject.SetActive(canInvite);

            bool canRetry = status != null && status.CanRetry;
            if (_retryButton != null) _retryButton.gameObject.SetActive(canRetry);

            // Cancel/Back is always present; it centres itself when Retry is not showing
            if (_cancelButton != null)
                UiKit.SetRect(_cancelButton.GetComponent<RectTransform>(), canRetry ? 110f : 0f, -262f, 200f, 42f);
            if (_cancelButtonLabel != null)
                _cancelButtonLabel.text = status != null && status.CanCancel ? "Cancel" : "Back";
        }

        static string ReadyWord(bool ready)
        {
            return ready ? "ready" : "not ready";
        }

        // ----- test seams --------------------------------------------------------------------------

        /// <summary>The status line exactly as the player reads it.</summary>
        internal string CurrentStatusText { get { return _statusText != null ? _statusText.text : string.Empty; } }

        /// <summary>Every status line published so far, in order.</summary>
        internal IReadOnlyList<string> StatusTextsForTests { get { return _statusTexts; } }

        internal SteamLobbyCoordinator? CoordinatorForTests { get { return _coordinator; } }

        /// <summary>True when the button is on screen for the player to press.</summary>
        internal bool ReadyOfferedForTests { get { return Offered(_readyButton); } }

        internal bool InviteOfferedForTests { get { return Offered(_inviteButton); } }

        internal bool RetryOfferedForTests { get { return Offered(_retryButton); } }

        internal bool CancelOfferedForTests { get { return Offered(_cancelButton); } }

        // These fire the real onClick binding rather than the handler behind it, so a button wired to
        // the wrong callback fails the test. The tests take them as methods because the PlayMode
        // assembly does not reference UnityEngine.UI and so cannot name a Button of its own.
        internal void ClickReadyForTests() { Click(_readyButton); }

        internal void ClickInviteForTests() { Click(_inviteButton); }

        internal void ClickRetryForTests() { Click(_retryButton); }

        internal void ClickCancelForTests() { Click(_cancelButton); }

        static bool Offered(Button? button)
        {
            return button != null && button.gameObject.activeSelf;
        }

        static void Click(Button? button)
        {
            if (button != null) button.onClick.Invoke();
        }
    }
}
