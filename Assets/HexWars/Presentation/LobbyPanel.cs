using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Online/offline setup screen. Each numeric setting is an editable box (type a value) flanked by big
    /// −/+ buttons (tap to adjust — the mobile fallback when the soft keyboard doesn't appear). Toggles pick
    /// mode and vs-AI. Create + vs-AI starts a local game; Create online shows a shareable link; joining is
    /// done by opening that link. Removes itself once the match starts.
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
            scaler.referenceResolution = new Vector2(1200f, 800f);
            scaler.matchWidthOrHeight = 0f; // match width so the panel always fits the screen's width (phones)
            _canvasGo.AddComponent<GraphicRaycaster>();

            Stretch(Panel(_canvasGo.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.9f)).GetComponent<RectTransform>());

            _form = Panel(_canvasGo.transform, "Form", new Color(0.10f, 0.12f, 0.18f, 0.99f));
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(680f, 520f);
            frt.anchoredPosition = Vector2.zero;

            float y = -26f;
            Label(_form.transform, "HexWars — New Game", 0f, y, 680f, 38f, 26, TextAnchor.MiddleCenter); y -= 52f;

            ToggleBtn(_form.transform, "Annihilation", -90f, y, 170f, 38f, () => _mode == GameMode.Annihilation, () => { _mode = GameMode.Annihilation; Refresh(); });
            ToggleBtn(_form.transform, "Territory", 95f, y, 150f, 38f, () => _mode == GameMode.Territory, () => { _mode = GameMode.Territory; Refresh(); });
            y -= 50f;

            NumberRow("Map width", y, () => _w, v => _w = v, 5, 24, 1); y -= 48f;
            NumberRow("Map height", y, () => _h, v => _h = v, 5, 24, 1); y -= 48f;
            NumberRow("Start points", y, () => _pts, v => _pts = v, 0, 200, 10); y -= 48f;

            Label(_form.transform, "Seed", -235f, y, 210f, 38f, 18, TextAnchor.MiddleLeft);
            var seedBox = MakeInput(_form.transform, _seed.ToString(), 40f, y, 130f, 38f);
            seedBox.characterLimit = 5;
            seedBox.onEndEdit.AddListener(s => { if (int.TryParse(s, out var v)) _seed = v; seedBox.text = _seed.ToString(); });
            Btn(_form.transform, "Reroll", 165f, y, 110f, 38f, () => { _seed = UnityEngine.Random.Range(1, 9999); seedBox.text = _seed.ToString(); });
            y -= 52f;

            ToggleBtn(_form.transform, "vs AI (single player)", -90f, y, 260f, 38f, () => _vsAi, () => { _vsAi = !_vsAi; Refresh(); });
            y -= 56f;

            Btn(_form.transform, "Create Game", 0f, y, 320f, 50f, OnCreate, big: true); y -= 50f;
            Label(_form.transform, "To join a game, open the host's shared link.", 0f, y, 680f, 26f, 15, TextAnchor.MiddleCenter);

            _status = Label(_canvasGo.transform, "", 0f, 0f, 1100f, 120f, 19, TextAnchor.MiddleCenter);
            var srt = _status.GetComponent<RectTransform>();
            srt.anchorMin = srt.anchorMax = new Vector2(0.5f, 0.5f);
            srt.pivot = new Vector2(0.5f, 0.5f);
            srt.anchoredPosition = Vector2.zero;
        }

        // label + [−] [editable box] [+], all kept comfortably inside the panel
        void NumberRow(string label, float y, Func<int> get, Action<int> set, int min, int max, int step)
        {
            Label(_form.transform, label, -235f, y, 210f, 38f, 18, TextAnchor.MiddleLeft);
            InputField box = null;
            Btn(_form.transform, "-", -10f, y, 54f, 38f, () => { set(Mathf.Clamp(get() - step, min, max)); if (box != null) box.text = get().ToString(); }, glyph: 30);
            box = MakeInput(_form.transform, get().ToString(), 80f, y, 90f, 38f);
            var b = box;
            b.onEndEdit.AddListener(s => { int v = int.TryParse(s, out var p) ? p : get(); set(Mathf.Clamp(v, min, max)); b.text = get().ToString(); });
            Btn(_form.transform, "+", 170f, y, 54f, 38f, () => { set(Mathf.Clamp(get() + step, min, max)); if (box != null) box.text = get().ToString(); }, glyph: 30);
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
                ShowWaiting($"Waiting for opponent…\nOpen this link on the other device:\n{ShareUrl(room)}");
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

        // editable numeric field: light box, dark text in a stretched child (renders the value and is typeable)
        InputField MakeInput(Transform parent, string initial, float x, float y, float w, float h)
        {
            var go = new GameObject("Input");
            go.transform.SetParent(parent, false);
            var bg = go.AddComponent<Image>();
            bg.color = new Color(0.93f, 0.95f, 0.98f, 1f);
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            var input = go.AddComponent<InputField>();
            input.targetGraphic = bg;

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(go.transform, false);
            var text = textGo.AddComponent<Text>();
            text.font = _font; text.fontSize = 20; text.color = new Color(0.06f, 0.07f, 0.10f);
            text.alignment = TextAnchor.MiddleCenter; text.supportRichText = false;
            var trt = textGo.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(6f, 2f); trt.offsetMax = new Vector2(-6f, -2f);

            input.textComponent = text;
            input.contentType = InputField.ContentType.IntegerNumber;
            input.characterLimit = 4;
            input.text = initial;
            return input;
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

        Button Btn(Transform parent, string text, float x, float y, float w, float h, Action onClick, bool big = false, int glyph = 0)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = big ? new Color(0.20f, 0.52f, 0.32f, 1f) : new Color(0.22f, 0.36f, 0.58f, 1f);
            var b = go.AddComponent<Button>();
            b.onClick.AddListener(() => onClick());
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            Label(go.transform, text, 0f, 0f, w, h, glyph > 0 ? glyph : (big ? 22 : 18), TextAnchor.MiddleCenter);
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
