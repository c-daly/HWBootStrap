using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Online/offline setup screen — fully tap-based so it works on mobile WebGL (no text fields, which
    /// don't get a keyboard there). Steppers set map width/height and starting points, Reroll picks a seed,
    /// toggles choose mode and vs-AI. Create + vs-AI starts a local game; Create online shows a shareable
    /// link; joining is done by opening that link (no code typing). Removes itself once the match starts.
    /// </summary>
    public sealed class LobbyPanel : MonoBehaviour
    {
        GameBootstrap _game;
        Font _font;
        GameObject _canvasGo;
        GameObject _form;
        Text _status;

        GameMode _mode = GameMode.Annihilation;
        bool _vsAi;
        int _w = 9, _h = 7, _pts = 0, _seed = 7;

        readonly List<(Button btn, Func<bool> selected)> _toggles = new List<(Button, Func<bool>)>();

        void Start()
        {
            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _game = FindAnyObjectByType<GameBootstrap>();
            _seed = UnityEngine.Random.Range(1, 9999);
            Build();
            Refresh();
        }

        void Update()
        {
            if (_game != null && _game.State != null && _canvasGo != null) { Destroy(_canvasGo); Destroy(this); }
        }

        void Build()
        {
            _canvasGo = new GameObject("LobbyCanvas");
            _canvasGo.transform.SetParent(transform, false);
            var canvas = _canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 1000;
            var scaler = _canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            _canvasGo.AddComponent<GraphicRaycaster>();

            Stretch(Panel(_canvasGo.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.88f)).GetComponent<RectTransform>());

            _form = Panel(_canvasGo.transform, "Form", new Color(0.10f, 0.12f, 0.18f, 0.99f));
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(470f, 560f);
            frt.anchoredPosition = Vector2.zero;

            const float L = -215f, RX = 36f;
            float y = -22f;
            Label(_form.transform, "HexWars — New Game", 0f, y, 470f, 36f, 26, TextAnchor.MiddleCenter); y -= 54f;

            Label(_form.transform, "Mode", L, y, 150f, 32f, 17, TextAnchor.MiddleLeft);
            ToggleBtn(_form.transform, "Annihilation", RX, y, 150f, 34f, () => _mode == GameMode.Annihilation, () => { _mode = GameMode.Annihilation; Refresh(); });
            ToggleBtn(_form.transform, "Territory", RX + 158f, y, 120f, 34f, () => _mode == GameMode.Territory, () => { _mode = GameMode.Territory; Refresh(); });
            y -= 50f;

            Stepper(_form.transform, "Map width", y, () => _w, v => _w = v, 5, 24, 2); y -= 46f;
            Stepper(_form.transform, "Map height", y, () => _h, v => _h = v, 5, 24, 2); y -= 46f;
            Stepper(_form.transform, "Start points", y, () => _pts, v => _pts = v, 0, 200, 10); y -= 46f;

            Label(_form.transform, "Seed", L, y, 150f, 32f, 17, TextAnchor.MiddleLeft);
            var seedLabel = Label(_form.transform, _seed.ToString(), RX, y, 120f, 34f, 18, TextAnchor.MiddleLeft);
            Btn(_form.transform, "Reroll", RX + 120f, y, 100f, 34f, () => { _seed = UnityEngine.Random.Range(1, 9999); seedLabel.text = _seed.ToString(); });
            y -= 52f;

            ToggleBtn(_form.transform, "vs AI", RX, y, 130f, 34f, () => _vsAi, () => { _vsAi = !_vsAi; Refresh(); });
            Label(_form.transform, "(single player)", RX + 140f, y, 160f, 32f, 14, TextAnchor.MiddleLeft);
            y -= 56f;

            Btn(_form.transform, "Create Game", 0f, y, 300f, 48f, OnCreate, big: true); y -= 56f;
            Label(_form.transform, "To join: open the host's shared link.", 0f, y, 470f, 24f, 14, TextAnchor.MiddleCenter);

            _status = Label(_canvasGo.transform, "", 0f, -320f, 1100f, 80f, 18, TextAnchor.MiddleCenter);
        }

        void OnCreate()
        {
            var setup = new GameSetup(_mode, _w, _h, _pts, _seed);
            if (_vsAi)
            {
                _game.StartLocalGame(setup, true);
                ShowWaiting("Starting game vs AI…");
            }
            else
            {
                string room = RandomCode();
                _game.StartNetGame(room, setup.ToWire());
                ShowWaiting($"Waiting for opponent…  Open this link on the other device:\n{ShareUrl(room)}");
            }
        }

        void ShowWaiting(string msg)
        {
            if (_form != null) _form.SetActive(false);
            if (_status != null) _status.text = msg;
        }

        static string RandomCode()
        {
            const string alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
            var c = new char[5];
            for (int i = 0; i < c.Length; i++) c[i] = alphabet[UnityEngine.Random.Range(0, alphabet.Length)];
            return new string(c);
        }

        static string ShareUrl(string room)
        {
            string page = Application.absoluteURL;
            if (string.IsNullOrEmpty(page)) return "(this page) ?room=" + room;
            int q = page.IndexOf('?');
            if (q >= 0) page = page.Substring(0, q);
            return page + "?room=" + room;
        }

        // ---- ui helpers ----

        void Refresh()
        {
            foreach (var (btn, selected) in _toggles)
                btn.GetComponent<Image>().color = selected()
                    ? new Color(0.26f, 0.50f, 0.82f, 1f) : new Color(0.17f, 0.20f, 0.27f, 1f);
        }

        // a tap-based number control: label, [−] value [+]
        void Stepper(Transform parent, string label, float y, Func<int> get, Action<int> set, int min, int max, int step)
        {
            const float L = -215f, RX = 36f;
            Label(parent, label, L, y, 150f, 34f, 17, TextAnchor.MiddleLeft);
            var val = Label(parent, get().ToString(), RX + 54f, y, 70f, 34f, 19, TextAnchor.MiddleCenter);
            Btn(parent, "−", RX, y, 48f, 34f, () => { set(Mathf.Clamp(get() - step, min, max)); val.text = get().ToString(); });
            Btn(parent, "+", RX + 130f, y, 48f, 34f, () => { set(Mathf.Clamp(get() + step, min, max)); val.text = get().ToString(); });
        }

        GameObject Panel(Transform parent, string name, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            return go;
        }

        static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one; rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        Text Label(Transform parent, string text, float x, float y, float w, float h, int size, TextAnchor anchor)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = _font; t.fontSize = size; t.color = Color.white; t.alignment = anchor; t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Overflow; t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            return t;
        }

        Button Btn(Transform parent, string text, float x, float y, float w, float h, Action onClick, bool big = false)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = big ? new Color(0.20f, 0.52f, 0.32f, 1f) : new Color(0.22f, 0.36f, 0.58f, 1f);
            var b = go.AddComponent<Button>();
            b.onClick.AddListener(() => onClick());
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            Label(go.transform, text, 0f, 0f, w, h, big ? 22 : 18, TextAnchor.MiddleCenter);
            return b;
        }

        void ToggleBtn(Transform parent, string text, float x, float y, float w, float h, Func<bool> selected, Action onClick)
        {
            _toggles.Add((Btn(parent, text, x, y, w, h, onClick), selected));
        }

        static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, y);
        }
    }
}
