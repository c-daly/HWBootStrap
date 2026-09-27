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
        RectTransform _menu;
        readonly RectTransform[] _surround = new RectTransform[4];
        Text _demoStatus;
        Vector2 _layoutSize;
        Rect _demoViewport = new Rect(0, 0, 1, 1);
        internal Rect DemoViewport => _dead ? new Rect(0, 0, 1, 1) : _demoViewport;
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

            if (_demoStatus != null && _game.State != null)
                _demoStatus.text = $"AI exhibition   ·   Round {_game.State.Round}   ·   "
                    + (_game.State.IsGameOver ? "Match complete" : $"Player {(int)_game.State.ActivePlayer + 1} to move");

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

        void LateUpdate()
        {
            if (_dead || _game == null) return;
            // CanvasScaler updates in Update. During a browser resize its canvas can briefly report
            // raw pixels; wait for the scaled dimensions before laying out the menu and camera.
            Layout();
            ApplyDemoCamera();
        }

        void Close()
        {
            _dead = true;
            UnsubscribeSteam();
            RestoreCamera();
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void OnDestroy()
        {
            UnsubscribeSteam();
            RestoreCamera();
        }

        void RestoreCamera()
        {
            var camera = Camera.main;
            // Do not overwrite a viewport already assigned by the match HUD or a new title.
            if (camera != null && camera.rect == _demoViewport)
            {
                camera.rect = new Rect(0, 0, 1, 1);
                camera.GetComponent<CameraRig>()?.Frame();
            }
        }

        void ApplyDemoCamera()
        {
            var camera = Camera.main;
            if (!_game.DemoMode || camera == null || camera.rect == _demoViewport) return;
            camera.rect = _demoViewport;
            camera.GetComponent<CameraRig>()?.Frame();
        }

        void Layout()
        {
            if (_canvasGo == null || _menu == null) return;
            var size = ((RectTransform)_canvasGo.transform).rect.size;
            if (size.x <= 0 || size.y <= 0 || Vector2.Distance(size, _layoutSize) < 1) return;
            float scale = Mathf.Min(1f, (size.y - 90f) / 640f, (size.x - 48f) / 440f);
            float menuWidth = 440f * scale, menuHeight = 640f * scale;
            bool beside = size.x >= 1100f;
            float left = beside ? 48f + menuWidth : 16f;
            float bottom = beside ? 50f : menuHeight + 100f;
            float width = size.x - left - 20f, height = size.y - bottom - 80f;
            if (scale <= 0 || width <= 0 || height <= 0) return; // keep the last usable viewport during a transient resize
            _layoutSize = size;
            _menu.localScale = Vector3.one * scale;
            _menu.anchoredPosition = beside
                ? new Vector2(24f, -(size.y - menuHeight) * .5f)
                : new Vector2((size.x - menuWidth) * .5f, -(size.y - menuHeight - 68f));
            _demoViewport = new Rect(left / size.x, bottom / size.y, width / size.x, height / size.y);

            // A partial camera only clears its own viewport. Cover its surroundings each frame,
            // including after a resize, so moving text and the old board cannot leave trails.
            PlaceFromBottom(_surround[0], 0, 0, size.x, bottom);
            PlaceFromBottom(_surround[1], 0, bottom + height, size.x, 80f);
            PlaceFromBottom(_surround[2], 0, bottom, left, height);
            PlaceFromBottom(_surround[3], left + width, bottom, 20f, height);
            PlaceFromBottom(_demoStatus.rectTransform, left, bottom - 32f, width, 24f);
        }

        static void PlaceFromBottom(RectTransform rect, float x, float y, float width, float height)
        {
            rect.anchorMin = rect.anchorMax = rect.pivot = Vector2.zero;
            rect.anchoredPosition = new Vector2(x, y);
            rect.sizeDelta = new Vector2(width, height);
        }

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

            // The state check alone is not the whole gate. Before START a match socket is up with no
            // state behind it, and a lobby screen can be mid-flow with no state either; honouring the
            // invite in either case tears down live work and orphans the coordinator that holds the
            // Steam lobby and the auth ticket.
            if (!SteamInviteGate.CanAccept(_game.State != null, _game.DemoMode,
                                           _game.GetComponent<SteamMatchConnection>() != null,
                                           _game.GetComponent<SteamLobbyScreen>() != null)) return;

            Hide();
            SteamLobbyScreen.OpenInvited(_game, lobbyId);
        }

        void Hide() => Close(); // sub-screen takeover — semantically "step aside", the demo keeps playing

        void Build() => BuildForPlatform(SteamRuntime.IsSteamBuild);

        // Keep both menus testable in a Steam-enabled editor without initialising the Steam client.
        void BuildForPlatform(bool steamBuild)
        {
            _steamBuild = steamBuild;
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("TitleCanvas", UiKit.OrderMenu, transform);

            for (int i = 0; i < _surround.Length; i++)
            {
                var background = UiKit.Panel(_canvasGo.transform, "Demo surround", UiKit.Bg);
                background.sprite = null;
                background.raycastTarget = false;
                _surround[i] = background.rectTransform;
            }

            // left-anchored column: the menu reads over the demo without hiding the action
            var col = new GameObject("Menu");
            col.transform.SetParent(_canvasGo.transform, false);
            _menu = col.AddComponent<RectTransform>();
            _menu.anchorMin = _menu.anchorMax = _menu.pivot = new Vector2(0f, 1f);
            _menu.sizeDelta = new Vector2(440f, 640f);

            var brand = new GameObject("HexWars wordmark", typeof(RectTransform));
            brand.transform.SetParent(col.transform, false);
            var word = UiKit.Label(brand.transform, "HEXWARS", 28f, 0f, 330f, 70f, 52, TextAnchor.MiddleLeft, UiKit.Accent);
            word.fontStyle = FontStyle.Bold;
            float wordWidth = word.preferredWidth;
            UiKit.SetRect(word.rectTransform, 28f, 0f, wordWidth, 70f);
            UiKit.SetRect((RectTransform)brand.transform, 0f, -28f, wordWidth + 56f, 70f);
            HexBrandMark.Add(brand.transform, -wordWidth * .5f - 4f, -11f, 48f);
            UiKit.Label(col.transform, "Design an army. Take the field.",
                        0f, -104f, 400f, 24f, UiKit.SizeBody, TextAnchor.MiddleCenter, UiKit.TextDim);

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

                _roomCodeField = UiKit.InputField(col.transform, _committedRoomCode, -65f, y, 250f, bh,
                                                   "Room code");
                _roomCodeField.gameObject.name = "Room code";
                _roomCodeField.GetComponent<WebGlInputBridge>().CancelRequested += RestoreRoomCodeEdit;
                _roomCodeField.onSubmit.AddListener(_ => OnJoinByCode());
                _roomCodeError = UiKit.Label(col.transform, "", -65f, y - 39f, 245f, 18f,
                                             UiKit.SizeCaption, TextAnchor.MiddleLeft, UiKit.Danger);
                UiKit.Button(col.transform, "Join", 130f, y, 120f, bh, OnJoinByCode,
                             UiKit.ButtonStyle.Primary); y -= gap;
            }

            UiKit.Button(col.transform, "Play vs AI", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.VsAi); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Hotseat", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Hotseat); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "How to Play", 0f, y, bw, bh, () =>
            { GameRules.Show(_canvasGo.transform, UiKit.Font(), 1100); }, UiKit.ButtonStyle.Secondary); y -= gap;

            UiKit.Label(col.transform, "v" + Application.version + "   ·   local, AI, or online play",
                        0f, y - 6f, 400f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);

            _demoStatus = UiKit.Label(_canvasGo.transform, "AI exhibition", 0, 0, 400, 24,
                UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextDim);
            Layout();

            var collection = UiKit.Button(_canvasGo.transform, "Unit collection", 0, 0, 180, 42, () => GraphiteWorkshop.Open(_game), UiKit.ButtonStyle.Secondary, 17);
            var cr = collection.GetComponent<RectTransform>(); cr.anchorMin=cr.anchorMax=new Vector2(1,1); cr.pivot=new Vector2(1,1);cr.anchoredPosition=new Vector2(-20,-20);
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
