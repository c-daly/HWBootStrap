using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// A screen-space tooltip that shows a hovered/selected unit's full capabilities. Builds its own
    /// canvas/panel/text programmatically so no scene wiring is needed.
    /// </summary>
    public sealed class UnitTooltip : MonoBehaviour
    {
        GameObject _panel;
        Text _text;

        void Awake()
        {
            var canvasGo = new GameObject("UnitTooltipCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 1000;
            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            _panel = new GameObject("Panel");
            _panel.transform.SetParent(canvasGo.transform, false);
            var img = _panel.AddComponent<Image>();
            img.color = new Color(0.05f, 0.06f, 0.10f, 0.86f);
            var prt = _panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = Vector2.zero; // bottom-left origin (matches screen coords)
            prt.pivot = new Vector2(0f, 1f);              // top-left, so it hangs below-right of the cursor
            prt.sizeDelta = new Vector2(252f, 170f);

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(_panel.transform, false);
            _text = textGo.AddComponent<Text>();
            _text.font = BuiltinFont();
            _text.fontSize = 16;
            _text.color = Color.white;
            _text.supportRichText = true;
            _text.alignment = TextAnchor.UpperLeft;
            _text.horizontalOverflow = HorizontalWrapMode.Overflow;
            _text.verticalOverflow = VerticalWrapMode.Overflow;
            var trt = _text.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero;
            trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(12f, 10f);
            trt.offsetMax = new Vector2(-12f, -10f);

            Hide();
        }

        static Font BuiltinFont()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }

        public void Show(Unit unit, Vector2 screenPos)
        {
            _text.text = Format(unit);
            var prt = _panel.GetComponent<RectTransform>();
            float x = Mathf.Min(screenPos.x + 16f, Screen.width - prt.sizeDelta.x - 4f);
            float y = Mathf.Max(screenPos.y - 16f, prt.sizeDelta.y + 4f);
            prt.anchoredPosition = new Vector2(x, y);
            _panel.SetActive(true);
        }

        public void Hide()
        {
            if (_panel != null) _panel.SetActive(false);
        }

        static string Format(Unit u)
        {
            var s = u.Stats;
            string role = Roles.Dominant(s).ToString();
            string owner = u.Owner == PlayerId.Player0 ? "Player 1" : "Player 2";
            return
                $"<b>{role}</b>   {s.PointCost} pts   ({owner})\n" +
                $"HP {u.CurrentHp}/{s.Health}\n" +
                $"Damage {s.Damage}    Defense {s.Defense}\n" +
                $"Move {s.Movement}    Vertical {s.VerticalMovement}\n" +
                $"Range {s.Range}    Range Arc {s.RangeArc}\n" +
                $"Vision {s.Vision}    Vision Arc {s.VisionArc}";
        }
    }
}
