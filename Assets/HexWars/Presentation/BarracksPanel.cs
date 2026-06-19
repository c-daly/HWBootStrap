using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Right-side barracks: lists the active player's reusable templates. Select one, then click a
    /// deployment-zone hex to deploy a paid clone (the template is not consumed). Stays in deploy
    /// mode so you can place several; re-click the selected template to stop.
    /// </summary>
    public sealed class BarracksPanel : MonoBehaviour
    {
        GameBootstrap _game;
        Font _font;
        RectTransform _list;
        Text _hint;
        int _deployIndex = -1;
        readonly List<GameObject> _rows = new List<GameObject>();

        void Start()
        {
            _font = BuiltinFont();
            _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            if (_game != null) { _game.StateChanged += Rebuild; Rebuild(); }
        }

        void OnDestroy()
        {
            if (_game != null) _game.StateChanged -= Rebuild;
        }

        void Update()
        {
            if (_deployIndex < 0 || _game == null) return;
            var mouse = Mouse.current;
            var cam = Camera.main;
            if (mouse == null || cam == null || !mouse.leftButton.wasPressedThisFrame || IsOverUi()) return;

            var mp = mouse.position.ReadValue();
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
            {
                var tv = hit.collider.GetComponentInParent<TileView>();
                if (tv != null)
                    _game.TryApply(new DeployUnit(_game.State.ActivePlayer, _deployIndex, tv.Coord));
            }
        }

        void Build()
        {
            var canvasGo = new GameObject("BarracksCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 400;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            const float w = 230f;
            var panel = new GameObject("BarracksPanel");
            panel.transform.SetParent(canvasGo.transform, false);
            panel.AddComponent<Image>().color = new Color(0.06f, 0.07f, 0.11f, 0.9f);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(w, 420f);
            prt.anchoredPosition = new Vector2(-8f, -58f);

            Label(panel.transform, "Title", "BARRACKS", 12f, 8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);
            _hint = Label(panel.transform, "Hint", "Design a unit, then deploy it here.", 12f, 34f, w - 24f, 22f, 13, TextAnchor.MiddleLeft);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            _list = listGo.AddComponent<RectTransform>();
            _list.anchorMin = _list.anchorMax = new Vector2(0f, 1f);
            _list.pivot = new Vector2(0f, 1f);
            _list.sizeDelta = new Vector2(w, 360f);
            _list.anchoredPosition = new Vector2(0f, -60f);
        }

        void Rebuild()
        {
            foreach (var r in _rows) Destroy(r);
            _rows.Clear();
            if (_game == null || _game.State == null) return;

            var s = _game.State;
            var p = s.Player(s.ActivePlayer);
            if (_deployIndex >= p.Barracks.Count) _deployIndex = -1;

            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var stats = p.Barracks[i];
                int cost = Economy.DeployCost(stats, s.Config);
                bool selected = i == _deployIndex;
                int idx = i;
                var row = Button(_list, $"{Roles.Dominant(stats)}   deploy {cost}", 8f, 4f + i * 34f, 214f, 30f,
                                 () => Select(idx),
                                 selected ? new Color(0.85f, 0.7f, 0.15f, 1f) : new Color(0.20f, 0.34f, 0.55f, 1f));
                _rows.Add(row);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy (re-click to stop)."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");
        }

        void Select(int i)
        {
            _deployIndex = (_deployIndex == i) ? -1 : i; // toggle
            Rebuild();
        }

        static bool IsOverUi()
        {
            var es = EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }

        // ---- UGUI helpers ----

        Text Label(Transform parent, string name, string text, float x, float y, float w, float h, int size, TextAnchor anchor)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = _font; t.text = text; t.fontSize = size; t.color = Color.white; t.alignment = anchor;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, -y);
            return t;
        }

        GameObject Button(Transform parent, string label, float x, float y, float w, float h, UnityEngine.Events.UnityAction onClick, Color color)
        {
            var go = new GameObject("Row");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            go.AddComponent<Button>().onClick.AddListener(onClick);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, -y);

            var tg = new GameObject("Text");
            tg.transform.SetParent(go.transform, false);
            var t = tg.AddComponent<Text>();
            t.font = _font; t.text = label; t.fontSize = 14; t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            var trt = t.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
            trt.offsetMin = Vector2.zero; trt.offsetMax = Vector2.zero;
            return go;
        }

        static Font BuiltinFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }
    }
}
