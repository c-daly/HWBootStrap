using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Online/offline setup screen. Pick mode, map width/height, starting points, and seed (all typeable),
    /// optionally toggle "vs AI", then Create or Join a room code. Create + vs-AI starts a local game vs the
    /// computer; Create online connects with the chosen <see cref="GameSetup"/> and shows a shareable link;
    /// Join connects to an existing room. Removes itself once the match starts.
    /// </summary>
    public sealed class LobbyPanel : MonoBehaviour
    {
        GameBootstrap _game;
        Font _font;
        GameObject _canvasGo;
        GameObject _form;
        Text _status;
        InputField _wIn, _hIn, _ptsIn, _seedIn, _joinIn;

        GameMode _mode = GameMode.Annihilation;
        bool _vsAi;

        readonly List<(Button btn, Func<bool> selected)> _toggles = new List<(Button, Func<bool>)>();

        void Start()
        {
            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _game = FindAnyObjectByType<GameBootstrap>();
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
            frt.sizeDelta = new Vector2(460f, 600f);
            frt.anchoredPosition = Vector2.zero;

            const float L = -210f, RX = 30f;  // label x, controls x
            float y = -24f;
            Label(_form.transform, "HexWars — New Game", 0f, y, 460f, 36f, 26, TextAnchor.MiddleCenter); y -= 56f;

            Label(_form.transform, "Mode", L, y, 150f, 30f, 17, TextAnchor.MiddleLeft);
            ToggleBtn(_form.transform, "Annihilation", RX, y, 150f, 32f, () => _mode == GameMode.Annihilation, () => { _mode = GameMode.Annihilation; Refresh(); });
            ToggleBtn(_form.transform, "Territory", RX + 158f, y, 120f, 32f, () => _mode == GameMode.Territory, () => { _mode = GameMode.Territory; Refresh(); });
            y -= 46f;

            Label(_form.transform, "Map width", L, y, 150f, 30f, 17, TextAnchor.MiddleLeft);
            _wIn = NumberInput(_form.transform, "9", RX, y, 90f); y -= 42f;
            Label(_form.transform, "Map height", L, y, 150f, 30f, 17, TextAnchor.MiddleLeft);
            _hIn = NumberInput(_form.transform, "7", RX, y, 90f); y -= 42f;
            Label(_form.transform, "Start points", L, y, 150f, 30f, 17, TextAnchor.MiddleLeft);
            _ptsIn = NumberInput(_form.transform, "0", RX, y, 90f); y -= 42f;

            Label(_form.transform, "Seed", L, y, 150f, 30f, 17, TextAnchor.MiddleLeft);
            _seedIn = NumberInput(_form.transform, UnityEngine.Random.Range(1, 9999).ToString(), RX, y, 90f);
            Btn(_form.transform, "Reroll", RX + 100f, y, 90f, 32f, () => _seedIn.text = UnityEngine.Random.Range(1, 9999).ToString());
            y -= 50f;

            ToggleBtn(_form.transform, "vs AI", RX, y, 120f, 32f, () => _vsAi, () => { _vsAi = !_vsAi; Refresh(); });
            Label(_form.transform, "(single player)", RX + 130f, y, 160f, 30f, 14, TextAnchor.MiddleLeft);
            y -= 54f;

            Btn(_form.transform, "Create Game", 0f, y, 280f, 44f, OnCreate, big: true); y -= 54f;

            Label(_form.transform, "— or join an online game —", 0f, y, 460f, 24f, 14, TextAnchor.MiddleCenter); y -= 38f;
            _joinIn = MakeInput(_form.transform, "", -70f, y, 190f, 34f, "room code", InputField.ContentType.Standard);
            Btn(_form.transform, "Join", 140f, y, 90f, 34f, OnJoin);

            _status = Label(_canvasGo.transform, "", 0f, -330f, 1000f, 70f, 18, TextAnchor.MiddleCenter);
        }

        void OnCreate()
        {
            int w = Mathf.Clamp(ParseInt(_wIn, 9), 5, 24);
            int h = Mathf.Clamp(ParseInt(_hIn, 7), 5, 24);
            int pts = Mathf.Clamp(ParseInt(_ptsIn, 0), 0, 999);
            int seed = ParseInt(_seedIn, UnityEngine.Random.Range(1, 9999));
            var setup = new GameSetup(_mode, w, h, pts, seed);

            if (_vsAi)
            {
                _game.StartLocalGame(setup, true);
                ShowWaiting("Starting game vs AI…");
            }
            else
            {
                string room = RandomCode();
                _game.StartNetGame(room, setup.ToWire());
                ShowWaiting($"Waiting for opponent…  Share this link:\n{ShareUrl(room)}");
            }
        }

        void OnJoin()
        {
            string room = (_joinIn != null ? _joinIn.text : "").Trim();
            if (string.IsNullOrEmpty(room)) { if (_status != null) _status.text = "Enter a room code to join."; return; }
            _game.StartNetGame(room, null);
            ShowWaiting($"Joining {room}…");
        }

        void ShowWaiting(string msg)
        {
            if (_form != null) _form.SetActive(false);
            if (_status != null) _status.text = msg;
        }

        static int ParseInt(InputField f, int fallback) => (f != null && int.TryParse(f.text, out var v)) ? v : fallback;

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
            Label(go.transform, text, 0f, 0f, w, h, big ? 22 : 16, TextAnchor.MiddleCenter);
            return b;
        }

        void ToggleBtn(Transform parent, string text, float x, float y, float w, float h, Func<bool> selected, Action onClick)
        {
            _toggles.Add((Btn(parent, text, x, y, w, h, onClick), selected));
        }

        InputField NumberInput(Transform parent, string initial, float x, float y, float w)
            => MakeInput(parent, initial, x, y, w, 32f, "", InputField.ContentType.IntegerNumber);

        InputField MakeInput(Transform parent, string initial, float x, float y, float w, float h, string placeholder, InputField.ContentType type)
        {
            var go = new GameObject("Input");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(1f, 1f, 1f, 0.14f);
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            var input = go.AddComponent<InputField>();
            input.contentType = type;

            var ph = Label(go.transform, placeholder, 0f, 0f, w, h, 15, TextAnchor.MiddleLeft);
            ph.color = new Color(1f, 1f, 1f, 0.4f);
            var phRt = ph.GetComponent<RectTransform>(); phRt.offsetMin = new Vector2(8f, 0f); phRt.offsetMax = new Vector2(-8f, 0f);

            var text = Label(go.transform, "", 0f, 0f, w, h, 16, TextAnchor.MiddleLeft);
            var txRt = text.GetComponent<RectTransform>(); txRt.offsetMin = new Vector2(8f, 0f); txRt.offsetMax = new Vector2(-8f, 0f);

            input.textComponent = text;
            input.placeholder = ph;
            input.text = initial;
            input.characterLimit = 8;
            return input;
        }

        static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);  // x from parent center, y down from parent top
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, y);
        }
    }
}
