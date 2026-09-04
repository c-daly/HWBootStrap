using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// The front door: HEXWARS wordmark + the six main actions, drawn over the live demo game
    /// (StartDemo's muted AI-vs-AI match). Owns the demo lifecycle — when the demo game ends it
    /// waits a beat and starts a fresh one. Opening a sub-screen (Browse / Host / vs AI / Hotseat / Rules)
    /// hides this menu and keeps the demo running behind it; the sub-screens call
    /// <see cref="Reopen"/> to come back. Destroys itself when a real match starts.
    /// </summary>
    public sealed class TitleScreen : MonoBehaviour
    {
        const float DemoRestartDelay = 3f;

        GameBootstrap _game;
        GameObject _canvasGo;
        InputField _roomCodeField;
        Text _roomCodeError;
        string _committedRoomCode = "";
        float _overSince = -1f;
        bool _steamBuild;      // evaluated once in Build() — the Steam menu replaces browse/join-by-code
        bool _steamSubscribed; // an accepted invite must reach exactly one live title screen
        bool _dead; // set the moment this screen closes/hides — Destroy is deferred to end-of-frame,
                    // and a dying component's Update must not fire the self-heal (it would StartDemo()
                    // mid-frame right after join-by-code's Hide()+StartNetGame, clobbering the connection)

        public static void Reopen(GameBootstrap game)
        {
            if (game.GetComponent<TitleScreen>() == null) game.gameObject.AddComponent<TitleScreen>();
        }

        void Start()
        {
            _game = GetComponent<GameBootstrap>();
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            if (SteamRuntime.IsSteamBuild)
            {
                // the client (and its per-frame pump) must exist before the first lobby call, and an
                // invite accepted from the overlay has to find a listener while we are the front door
                SteamRuntime.EnsureCreated();
                var client = SteamRuntime.ClientIfCreated;
                if (client != null)
                {
                    client.InviteAccepted += OnInviteAccepted;
                    _steamSubscribed = true;
                }
            }
            Build();
        }

        void Update()
        {
            if (_dead || _game == null) return;

            if (DeviceInput.Allowed && UiKit.InputOwnsFocus(_roomCodeField) && Keyboard.current != null)
            {
                if (Keyboard.current.escapeKey.wasPressedThisFrame)
                {
                    UiKit.MarkInputEscapeHandled();
                    RestoreRoomCodeEdit();
                    return;
                }
                if (Keyboard.current.enterKey.wasPressedThisFrame
                    || Keyboard.current.numpadEnterKey.wasPressedThisFrame)
                {
                    OnJoinByCode();
                    return;
                }
            }

            // a real match started — the title is done (sub-screens dismiss themselves the same way)
            if (_game.State != null && !_game.DemoMode) { Close(); return; }

            // back on the title with no demo running (cancelled hosting, seat-full bounce, dropped
            // socket) — self-heal: the title always has a living background. A connect may still be
            // in flight here (e.g. a fresh join-by-code racing this Update); a START arriving into a
            // demo would desync a seated match, so drop the socket first (CancelHosting is null-safe).
            if (!_game.DemoMode && _game.State == null) { _game.CancelHosting(); _game.StartDemo(); return; }

            // demo ended: hold the final board a beat, then roll a fresh demo
            if (_game.DemoMode && _game.State != null && _game.State.IsGameOver)
            {
                if (_overSince < 0f) _overSince = Time.unscaledTime;
                else if (Time.unscaledTime - _overSince >= DemoRestartDelay) { _overSince = -1f; _game.StartDemo(); }
            }
            else _overSince = -1f;
        }

        void Close()
        {
            _dead = true;
            UnsubscribeSteam();
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void OnDestroy() => UnsubscribeSteam(); // Destroy(this) is deferred; a destroyed screen must not hold the event

        void UnsubscribeSteam()
        {
            if (!_steamSubscribed) return;
            _steamSubscribed = false;
            // ClientIfCreated, not Client: during shutdown this must not build a new Steam client
            var client = SteamRuntime.ClientIfCreated;
            if (client != null) client.InviteAccepted -= OnInviteAccepted;
        }

        /// <summary>
        /// The one place an accepted invite opens a lobby. It is honoured only while the title is the
        /// front door: with a real match on screen an invite must not tear it down.
        /// </summary>
        void OnInviteAccepted(string lobbyId)
        {
            if (_dead || string.IsNullOrEmpty(lobbyId)) return;
            if (_game == null) return;
            if (_game.State != null && !_game.DemoMode) return;
            Hide();
            SteamLobbyScreen.OpenInvited(_game, lobbyId);
        }

        void Hide() => Close(); // sub-screen takeover — semantically "step aside", the demo keeps playing

        void Build()
        {
            _steamBuild = SteamRuntime.IsSteamBuild;
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("TitleCanvas", UiKit.OrderMenu, transform);

            // left-anchored column: the menu reads over the demo without hiding the action
            var col = new GameObject("Menu");
            col.transform.SetParent(_canvasGo.transform, false);
            var crt = col.AddComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            var canvasRt = _canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            crt.sizeDelta = new Vector2(Mathf.Min(520f, availW - 40f), 640f);
            crt.anchoredPosition = new Vector2(0f, 20f);

            var plate = UiKit.Panel(col.transform, "Plate", new Color(UiKit.Bg.r, UiKit.Bg.g, UiKit.Bg.b, 0.82f));
            UiKit.Stretch(plate.GetComponent<RectTransform>());
            plate.raycastTarget = false;

            var word = UiKit.Label(col.transform, "HEXWARS", 0f, -34f, 520f, 70f, 58, TextAnchor.MiddleCenter, UiKit.Accent);
            word.fontStyle = FontStyle.Bold;
            UiKit.Label(col.transform, "hex-grid tactics — design an army, take the field",
                        0f, -104f, 520f, 24f, UiKit.SizeBody, TextAnchor.MiddleCenter, UiKit.TextDim);

            float y = -170f;
            const float bw = 380f, bh = 52f, gap = 62f;
            if (_steamBuild)
            {
                // Steam owns matchmaking here: no server room codes, no public browser — friends and
                // quick match come through the lobby screen instead
                UiKit.Button(col.transform, "Quick Match", 0f, y, bw, bh, () =>
                { Hide(); SteamLobbyScreen.OpenQuickMatch(_game); }, UiKit.ButtonStyle.Cta); y -= gap;

                UiKit.Button(col.transform, "Invite Friend", 0f, y, bw, bh, () =>
                { Hide(); SteamLobbyScreen.OpenInvite(_game); }, UiKit.ButtonStyle.Primary); y -= gap;

                UiKit.Button(col.transform, "Host Game", 0f, y, bw, bh, () =>
                { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Host); }, UiKit.ButtonStyle.Primary); y -= gap;
            }
            else
            {
                UiKit.Button(col.transform, "Browse Games", 0f, y, bw, bh, () =>
                { Hide(); GameBrowser.Open(_game); }, UiKit.ButtonStyle.Cta); y -= gap;

                UiKit.Button(col.transform, "Host Game", 0f, y, bw, bh, () =>
                { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Host); }, UiKit.ButtonStyle.Primary); y -= gap;

                _roomCodeField = UiKit.InputField(col.transform, _committedRoomCode, -65f, y, 245f, bh,
                                                   "Room code");
                _roomCodeField.gameObject.name = "Room code";
                _roomCodeField.GetComponent<WebGlInputBridge>().CancelRequested += RestoreRoomCodeEdit;
                _roomCodeField.onSubmit.AddListener(_ => OnJoinByCode());
                _roomCodeError = UiKit.Label(col.transform, "", -65f, y - 39f, 245f, 18f,
                                             UiKit.SizeCaption, TextAnchor.MiddleLeft, UiKit.Danger);
                UiKit.Button(col.transform, "Join", 135f, y, 125f, bh, OnJoinByCode,
                             UiKit.ButtonStyle.Primary); y -= gap;
            }

            UiKit.Button(col.transform, "Play vs AI", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.VsAi); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Hotseat", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Hotseat); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "How to Play", 0f, y, bw, bh, () =>
            { GameRules.Show(_canvasGo.transform, UiKit.Font(), 1100); }, UiKit.ButtonStyle.Secondary); y -= gap;

            UiKit.Label(col.transform, "v" + Application.version + "   ·   local, AI, or online play",
                        0f, y - 6f, 520f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            var tipsBtn = TipsService.BuildToggle(_canvasGo.transform, 0f, 0f);
            var trt = tipsBtn.GetComponent<RectTransform>();
            trt.anchorMin = trt.anchorMax = new Vector2(0f, 0f);
            trt.pivot = new Vector2(0f, 0f);
            trt.anchoredPosition = new Vector2(12f, 12f);
        }

        void OnJoinByCode()
        {
            if (_dead) return; // InputField.onSubmit and the keyboard fallback may share one frame.
            string code = NormalizeRoomCode(_roomCodeField != null ? _roomCodeField.text : "");
            if (code.Length == 0)
            {
                if (_roomCodeError != null) _roomCodeError.text = "Enter a room code";
                return;
            }
            _committedRoomCode = code;
            _roomCodeField?.SetTextWithoutNotify(code);
            if (_roomCodeError != null) _roomCodeError.text = "";
            Hide();
            _game.StartNetGame(code, null);   // SEAT/START arrive via the normal net path;
                                              // a full/unknown room toasts via OnNetSeatFull
        }

        void RestoreRoomCodeEdit()
        {
            if (_roomCodeField == null) return;
            _roomCodeField.SetTextWithoutNotify(_committedRoomCode);
            if (_roomCodeError != null) _roomCodeError.text = "";
            _roomCodeField.DeactivateInputField();
            (EventSystem.current ?? FindAnyObjectByType<EventSystem>())?.SetSelectedGameObject(null);
        }

        internal static string NormalizeRoomCode(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw)) return "";
            var result = new System.Text.StringBuilder(16);
            foreach (char ch in raw.Trim().ToUpperInvariant())
            {
                if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) result.Append(ch);
                if (result.Length == 16) break;
            }
            return result.ToString();
        }
    }
}
