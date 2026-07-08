using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The front door: HEXWARS wordmark + the five main actions, drawn over the live demo game
    /// (StartDemo's muted AI-vs-AI match). Owns the demo lifecycle — when the demo game ends it
    /// waits a beat and starts a fresh one. Opening a sub-screen (Browse / Host / vs AI / Rules)
    /// hides this menu and keeps the demo running behind it; the sub-screens call
    /// <see cref="Reopen"/> to come back. Destroys itself when a real match starts.
    /// </summary>
    public sealed class TitleScreen : MonoBehaviour
    {
        const float DemoRestartDelay = 3f;

        GameBootstrap _game;
        GameObject _canvasGo;
        float _overSince = -1f;
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
            Build();
        }

        void Update()
        {
            if (_dead || _game == null) return;

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
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void Hide() => Close(); // sub-screen takeover — semantically "step aside", the demo keeps playing

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("TitleCanvas", UiKit.OrderMenu, transform);

            // left-anchored column: the menu reads over the demo without hiding the action
            var col = new GameObject("Menu");
            col.transform.SetParent(_canvasGo.transform, false);
            var crt = col.AddComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(520f, 640f);
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
            UiKit.Button(col.transform, "Browse Games", 0f, y, bw, bh, () =>
            { Hide(); GameBrowser.Open(_game); }, UiKit.ButtonStyle.Cta); y -= gap;

            UiKit.Button(col.transform, "Host Game", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.Host); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Join by Code", 0f, y, bw, bh, OnJoinByCode, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "Play vs AI", 0f, y, bw, bh, () =>
            { Hide(); SetupForm.Open(_game, SetupForm.SetupMode.VsAi); }, UiKit.ButtonStyle.Primary); y -= gap;

            UiKit.Button(col.transform, "How to Play", 0f, y, bw, bh, () =>
            { GameRules.Show(_canvasGo.transform, UiKit.Font(), 1100); }, UiKit.ButtonStyle.Secondary); y -= gap;

            UiKit.Label(col.transform, "v" + Application.version + "   ·   two players, two browsers — share a room code",
                        0f, y - 6f, 520f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint);
        }

        void OnJoinByCode()
        {
            string code = UiKit.PromptText("Enter the room code (e.g. KQ7KP)", "");
            if (code == null)
            {
                Toast.Show("Type-in needs the browser build — in the editor, use Browse Games.");
                return;
            }
            code = code.Trim().ToUpperInvariant();
            if (code.Length == 0) return; // cancelled
            Hide();
            _game.StartNetGame(code, null);   // SEAT/START arrive via the normal net path;
                                              // a full/unknown room toasts via OnNetSeatFull
        }
    }
}
