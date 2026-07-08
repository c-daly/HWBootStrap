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
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;

        void Start()
        {
            _stats[0] = 1; // Health >= 1
            _game = FindAnyObjectByType<GameBootstrap>();
            Build();
            RefreshSummary();
        }

        // DesignPanel has no StateChanged-driven refresh, so poll: one SetActive per flip of the
        // hide condition (the equality guard keeps it from thrashing the canvas every frame), like
        // GameHud's guard. Hidden during the title demo and the connecting window (no state yet).
        void Update()
        {
            if (_game == null || _canvasGo == null) return;
            bool hidden = _game.DemoMode || _game.State == null;
            if (_canvasGo.activeSelf == hidden)
                _canvasGo.SetActive(!hidden);
        }

        void Build()
        {
            var canvasGo = UiKit.Canvas("DesignCanvas", UiKit.OrderPanels, transform);
            _canvasGo = canvasGo;

            const float w = 270f, rowH = 30f, top = 58f;
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowH * 9 + 120f);
            prt.anchoredPosition = new Vector2(8f, -top);
            var panel = panelImg.transform;

            UiKit.Label(panel, "DESIGN UNIT", 0f, -8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);

            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowH);
                UiKit.Label(panel, Names[i], -63f, y, 120f, rowH, 15, TextAnchor.MiddleLeft);
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                int idx = i;
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
            }
            _valueLabels[0].text = "1";

            float sy = -(40f + 9 * rowH + 6f);
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
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
            if (_game == null || _game.State == null) return;
            _game.TryApply(new CreateUnit(_game.State.ActivePlayer, ToStats()));
        }
    }
}
