using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Online setup screen: pick mode / board size / starting points / seed, then Create (host) or Join a
    /// room code. Create connects with the chosen <see cref="GameSetup"/> and shows a shareable link; Join
    /// connects to an existing room. Removes itself once the match starts. GameBootstrap shows it only when
    /// there's no <c>?room=</c> in the page URL — a shared link skips straight to joining.
    /// </summary>
    public sealed class LobbyPanel : MonoBehaviour
    {
        GameBootstrap _game;
        Font _font;
        GameObject _canvasGo;
        GameObject _form;
        Text _status;
        InputField _joinCode;

        GameMode _mode = GameMode.Annihilation;
        int _w = 9, _h = 7;
        int _points = 0;
        int _seed = 7;

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
            if (_game != null && _game.State != null && _canvasGo != null) // game started — clear the lobby
            {
                Destroy(_canvasGo);
                Destroy(this);
            }
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

            // dim backdrop
            var dim = Panel(_canvasGo.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.85f));
            Stretch(dim.GetComponent<RectTransform>());

            _form = Panel(_canvasGo.transform, "Form", new Color(0.08f, 0.10f, 0.16f, 0.98f));
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = new Vector2(0.5f, 0.5f);
            frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(440f, 560f);
            frt.anchoredPosition = Vector2.zero;

            float y = -28f;
            Label(_form.transform, "HexWars — New Game", 0f, y, 440f, 34f, 24, TextAnchor.MiddleCenter); y -= 50f;

            Label(_form.transform, "Mode", -200f + 16f, y, 120f, 24f, 16, TextAnchor.MiddleLeft);
            ToggleBtn(_form.transform, "Annihilation", -8f, y, 150f, 30f, () => _mode == GameMode.Annihilation, () => { _mode = GameMode.Annihilation; Refresh(); });
            ToggleBtn(_form.transform, "Territory", 150f, y, 120f, 30f, () => _mode == GameMode.Territory, () => { _mode = GameMode.Territory; Refresh(); });
            y -= 44f;

            Label(_form.transform, "Board", -200f + 16f, y, 120f, 24f, 16, TextAnchor.MiddleLeft);
            ToggleBtn(_form.transform, "Small", -8f, y, 90f, 30f, () => _w == 9, () => { _w = 9; _h = 7; Refresh(); });
            ToggleBtn(_form.transform, "Medium", 86f, y, 90f, 30f, () => _w == 11, () => { _w = 11; _h = 8; Refresh(); });
            ToggleBtn(_form.transform, "Large", 180f, y, 90f, 30f, () => _w == 13, () => { _w = 13; _h = 9; Refresh(); });
            y -= 44f;

            Label(_form.transform, "Start points", -200f + 16f, y, 120f, 24f, 16, TextAnchor.MiddleLeft);
            ToggleBtn(_form.transform, "0", -8f, y, 70f, 30f, () => _points == 0, () => { _points = 0; Refresh(); });
            ToggleBtn(_form.transform, "20", 66f, y, 70f, 30f, () => _points == 20, () => { _points = 20; Refresh(); });
            ToggleBtn(_form.transform, "40", 140f, y, 70f, 30f, () => _points == 40, () => { _points = 40; Refresh(); });
            y -= 44f;

            Label(_form.transform, "Seed", -200f + 16f, y, 120f, 24f, 16, TextAnchor.MiddleLeft);
            var seedLabel = Label(_form.transform, _seed.ToString(), -8f, y, 120f, 30f, 16, TextAnchor.MiddleLeft);
            Btn(_form.transform, "Reroll", 120f, y, 90f, 30f, () => { _seed = UnityEngine.Random.Range(1, 9999); seedLabel.text = _seed.ToString(); });
            y -= 56f;

            Btn(_form.transform, "Create Game", 0f, y, 260f, 40f, OnCreate, big: true); y -= 56f;

            Label(_form.transform, "— or join a game —", 0f, y, 440f, 24f, 14, TextAnchor.MiddleCenter); y -= 36f;
            _joinCode = MakeInput(_form.transform, "", -60f, y, 180f, 32f, "room code");
            Btn(_form.transform, "Join", 130f, y, 90f, 32f, OnJoin);

            _status = Label(_canvasGo.transform, "", 0f, -300f, 900f, 60f, 18, TextAnchor.MiddleCenter);
        }

        void OnCreate()
        {
            string room = RandomCode();
            string setup = new GameSetup(_mode, _w, _h, _points, _seed).ToWire();
            _game.StartNetGame(room, setup);
            ShowWaiting($"Waiting for opponent…  Share this link:\n{ShareUrl(room)}");
        }

        void OnJoin()
        {
            string room = (_joinCode != null ? _joinCode.text : "").Trim();
            if (string.IsNullOrEmpty(room)) { if (_status != null) _status.text = "Enter a room code to join."; return; }
            _game.StartNetGame(room, null);
            ShowWaiting($"Joining {room}…");
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
            if (string.IsNullOrEmpty(page)) return "(open this page) ?room=" + room;
            int q = page.IndexOf('?');
            if (q >= 0) page = page.Substring(0, q);
            return page + "?room=" + room;
        }

        // ---- ui helpers ----

        void Refresh()
        {
            foreach (var (btn, selected) in _toggles)
            {
                var img = btn.GetComponent<Image>();
                img.color = selected() ? new Color(0.24f, 0.46f, 0.78f, 1f) : new Color(0.18f, 0.21f, 0.28f, 1f);
            }
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
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        Text Label(Transform parent, string text, float x, float y, float w, float h, int size, TextAnchor anchor)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = _font; t.fontSize = size; t.color = Color.white; t.alignment = anchor; t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Overflow; t.verticalOverflow = VerticalWrapMode.Overflow;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            return t;
        }

        Button Btn(Transform parent, string text, float x, float y, float w, float h, Action onClick, bool big = false)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = big ? new Color(0.20f, 0.50f, 0.30f, 1f) : new Color(0.20f, 0.34f, 0.55f, 1f);
            var b = go.AddComponent<Button>();
            b.onClick.AddListener(() => onClick());
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            Label(go.transform, text, 0f, 0f, w, h, big ? 20 : 15, TextAnchor.MiddleCenter);
            return b;
        }

        void ToggleBtn(Transform parent, string text, float x, float y, float w, float h, Func<bool> selected, Action onClick)
        {
            var b = Btn(parent, text, x, y, w, h, onClick);
            _toggles.Add((b, selected));
        }

        InputField MakeInput(Transform parent, string initial, float x, float y, float w, float h, string placeholder)
        {
            var go = new GameObject("Input");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(1f, 1f, 1f, 0.12f);
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            var input = go.AddComponent<InputField>();

            var ph = Label(go.transform, placeholder, 0f, 0f, w, h, 15, TextAnchor.MiddleLeft);
            ph.color = new Color(1f, 1f, 1f, 0.4f);
            var phRt = ph.GetComponent<RectTransform>(); phRt.offsetMin = new Vector2(8f, 0f); phRt.offsetMax = new Vector2(-8f, 0f);

            var text = Label(go.transform, "", 0f, 0f, w, h, 15, TextAnchor.MiddleLeft);
            var txRt = text.GetComponent<RectTransform>(); txRt.offsetMin = new Vector2(8f, 0f); txRt.offsetMax = new Vector2(-8f, 0f);

            input.textComponent = text;
            input.placeholder = ph;
            input.text = initial;
            input.characterLimit = 12;
            return input;
        }

        static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = new Vector2(0.5f, 1f);  // x relative to center, y down from top of parent
            rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, y);
        }
    }
}
