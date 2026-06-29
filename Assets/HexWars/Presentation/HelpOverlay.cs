using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// A "?" button (top-right) that toggles a basic how-to-play panel. Auto-created in any scene that has a
    /// <see cref="GameBootstrap"/>, so players always have instructions one tap away. Tap outside the card or
    /// "Got it" to close.
    /// </summary>
    public sealed class HelpOverlay : MonoBehaviour
    {
        GameObject _panel;
        Font _font;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            if (FindAnyObjectByType<GameBootstrap>() == null) return;
            new GameObject("HelpOverlay").AddComponent<HelpOverlay>();
        }

        void Start()
        {
            _font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");

            var canvasGo = new GameObject("HelpCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 900; // above the HUD, below the lobby (1000)
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1200f, 800f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            var q = Btn(canvasGo.transform, "?", new Color(0.18f, 0.32f, 0.50f, 0.95f), 26, Toggle);
            var qrt = q.GetComponent<RectTransform>();
            qrt.anchorMin = qrt.anchorMax = new Vector2(1f, 1f);
            qrt.pivot = new Vector2(1f, 1f);
            qrt.sizeDelta = new Vector2(54f, 54f);
            qrt.anchoredPosition = new Vector2(-12f, -12f);

            BuildPanel(canvasGo.transform);
            _panel.SetActive(false);
        }

        void Toggle() { if (_panel != null) _panel.SetActive(!_panel.activeSelf); }

        void BuildPanel(Transform parent)
        {
            _panel = new GameObject("HelpPanel");
            _panel.transform.SetParent(parent, false);
            var prt = _panel.AddComponent<RectTransform>();
            prt.anchorMin = Vector2.zero; prt.anchorMax = Vector2.one; prt.offsetMin = Vector2.zero; prt.offsetMax = Vector2.zero;

            // dim backdrop — tap outside the card to close
            var dim = new GameObject("Dim");
            dim.transform.SetParent(_panel.transform, false);
            dim.AddComponent<Image>().color = new Color(0.02f, 0.03f, 0.06f, 0.85f);
            var dimRt = dim.GetComponent<RectTransform>();
            dimRt.anchorMin = Vector2.zero; dimRt.anchorMax = Vector2.one; dimRt.offsetMin = Vector2.zero; dimRt.offsetMax = Vector2.zero;
            dim.AddComponent<Button>().onClick.AddListener(() => _panel.SetActive(false));

            // card (its Image blocks taps so they don't fall through to the dim)
            var card = new GameObject("Card");
            card.transform.SetParent(_panel.transform, false);
            card.AddComponent<Image>().color = new Color(0.10f, 0.12f, 0.18f, 1f);
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(700f, 600f);
            crt.anchoredPosition = Vector2.zero;

            var text = new GameObject("Text").AddComponent<Text>();
            text.transform.SetParent(card.transform, false);
            text.font = _font; text.fontSize = 19; text.color = Color.white;
            text.alignment = TextAnchor.UpperLeft; text.raycastTarget = false;
            text.horizontalOverflow = HorizontalWrapMode.Wrap; text.verticalOverflow = VerticalWrapMode.Overflow;
            text.text = HelpText;
            var trt = text.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(30f, 74f); trt.offsetMax = new Vector2(-30f, -22f);

            var close = Btn(card.transform, "Got it", new Color(0.20f, 0.52f, 0.32f, 1f), 22, () => _panel.SetActive(false));
            var clrt = close.GetComponent<RectTransform>();
            clrt.anchorMin = clrt.anchorMax = new Vector2(0.5f, 0f);
            clrt.pivot = new Vector2(0.5f, 0f);
            clrt.sizeDelta = new Vector2(200f, 48f);
            clrt.anchoredPosition = new Vector2(0f, 16f);
        }

        const string HelpText =
@"HOW TO PLAY

CAMERA
  Phone: drag to pan, pinch to zoom
  Desktop: WASD / arrow keys to pan, scroll to zoom

YOUR UNITS
  Tap a unit to select it
  Tap a highlighted hex to MOVE there
  Tap an enemy within range to ATTACK

TERRITORY MODE
  Select a unit, then use the button at the bottom of
  the screen to:
    - CLAIM the hex it stands on (costs points, ends turn)
    - BUILD a generator on a hex you control
  Generators are your ONLY income, so build them early.
  Start a Territory game with about 40 points so you can
  afford the first claims and a generator.

Press End Turn (bottom-left) when you're done.";

        Button Btn(Transform parent, string label, Color color, int fontSize, System.Action onClick)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            var b = go.AddComponent<Button>();
            b.onClick.AddListener(() => onClick());
            var t = new GameObject("Label").AddComponent<Text>();
            t.transform.SetParent(go.transform, false);
            t.font = _font; t.fontSize = fontSize; t.color = Color.white;
            t.alignment = TextAnchor.MiddleCenter; t.raycastTarget = false;
            var lrt = t.GetComponent<RectTransform>();
            lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one; lrt.offsetMin = Vector2.zero; lrt.offsetMax = Vector2.zero;
            return b;
        }
    }
}
