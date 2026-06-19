using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Left-side panel to design a unit: +/- each of the 9 stats (Health floored at 1), see live
    /// PointCost + dominant role, and bank a free template to the active player's barracks.
    /// </summary>
    public sealed class DesignPanel : MonoBehaviour
    {
        static readonly string[] Names =
            { "Health", "Damage", "Defense", "Movement", "Vertical Move", "Range", "Range Arc", "Vision", "Vision Arc" };

        GameBootstrap _game;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
        Font _font;

        void Start()
        {
            _font = BuiltinFont();
            _stats[0] = 1; // Health >= 1
            _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            RefreshSummary();
        }

        void Build()
        {
            var canvasGo = new GameObject("DesignCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 400;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            const float w = 270f, rowH = 30f, top = 58f;
            var panel = Panel(canvasGo.transform, "DesignPanel", w, rowH * 9 + 120f, top);

            Label(panel, "Title", "DESIGN UNIT", 12f, 8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);

            for (int i = 0; i < 9; i++)
            {
                float y = 40f + i * rowH;
                Label(panel, "L" + i, Names[i], 12f, y, 120f, rowH, 15, TextAnchor.MiddleLeft);
                _valueLabels[i] = Label(panel, "V" + i, "0", 138f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                int idx = i;
                Button(panel, "-", 182f, y + 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                Button(panel, "+", 222f, y + 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
            }
            _valueLabels[0].text = "1";

            float sy = 40f + 9 * rowH + 6f;
            _summary = Label(panel, "Summary", "", 12f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            Button(panel, "Create (to Barracks)", 12f, sy + 30f, w - 24f, 30f, OnCreate);
        }

        void Adjust(int i, int delta)
        {
            _stats[i] = Mathf.Max(i == 0 ? 1 : 0, _stats[i] + delta);
            _valueLabels[i].text = _stats[i].ToString();
            RefreshSummary();
        }

        void RefreshSummary()
        {
            var s = ToStats();
            _summary.text = $"Cost {s.PointCost}   Role: {Roles.Dominant(s)}";
        }

        UnitStats ToStats() =>
            new UnitStats(_stats[0], _stats[1], _stats[2], _stats[3], _stats[4], _stats[5], _stats[6], _stats[7], _stats[8]);

        void OnCreate()
        {
            if (_game == null) return;
            _game.TryApply(new CreateUnit(_game.State.ActivePlayer, ToStats()));
        }

        // ---- UGUI helpers (top-left anchored) ----

        RectTransform Panel(Transform parent, string name, float w, float h, float top)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(0.06f, 0.07f, 0.11f, 0.9f);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(8f, -top);
            return rt;
        }

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

        void Button(Transform parent, string label, float x, float y, float w, float h, UnityEngine.Events.UnityAction onClick)
        {
            var go = new GameObject("Btn_" + label);
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(0.20f, 0.34f, 0.55f, 1f);
            go.AddComponent<Button>().onClick.AddListener(onClick);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0f, 1f);
            rt.pivot = new Vector2(0f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, -y);

            var tg = new GameObject("Text");
            tg.transform.SetParent(go.transform, false);
            var t = tg.AddComponent<Text>();
            t.font = _font; t.text = label; t.fontSize = 15; t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            var trt = t.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
            trt.offsetMin = Vector2.zero; trt.offsetMax = Vector2.zero;
        }

        static Font BuiltinFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }
    }
}
