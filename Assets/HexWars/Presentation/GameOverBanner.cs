using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The end-of-game announcement: a full-width band across the centre of the screen with the
    /// result and how it was won, over a light dim. Tap anywhere to dismiss and inspect the final
    /// board (the HUD banner keeps showing the result). Spawned by GameHud the moment the state
    /// turns game-over — previously the only signal was a rejection toast when you tried to act.
    /// </summary>
    public static class GameOverBanner
    {
        public const string RootName = "GameOverBanner";

        public static void Show(string title, string subtitle, Color accent)
        {
            Dismiss(); // never stack two

            var root = new GameObject(RootName);
            var canvas = root.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 850; // above HUD/toasts, below the tooltip/lobby
            var scaler = root.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            root.AddComponent<GraphicRaycaster>();

            // light dim; tapping anywhere dismisses so the final board stays inspectable
            var dim = new GameObject("Dim");
            dim.transform.SetParent(root.transform, false);
            dim.AddComponent<Image>().color = new Color(0.02f, 0.03f, 0.06f, 0.55f);
            var drt = dim.GetComponent<RectTransform>();
            drt.anchorMin = Vector2.zero; drt.anchorMax = Vector2.one;
            drt.offsetMin = Vector2.zero; drt.offsetMax = Vector2.zero;
            dim.AddComponent<Button>().onClick.AddListener(Dismiss);

            var band = new GameObject("Band");
            band.transform.SetParent(root.transform, false);
            var bandImg = band.AddComponent<Image>();
            bandImg.color = accent;
            bandImg.raycastTarget = false; // taps on the band fall through to the dim's dismiss button
            var brt = band.GetComponent<RectTransform>();
            brt.anchorMin = new Vector2(0f, 0.5f); brt.anchorMax = new Vector2(1f, 0.5f);
            brt.pivot = new Vector2(0.5f, 0.5f);
            brt.sizeDelta = new Vector2(0f, 150f);
            brt.anchoredPosition = Vector2.zero;

            var font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            Text(band.transform, title, font, 46, FontStyle.Bold, new Vector2(0f, 14f));
            Text(band.transform, subtitle + "   (tap to dismiss)", font, 19, FontStyle.Normal, new Vector2(0f, -38f));
        }

        public static void Dismiss()
        {
            var old = GameObject.Find(RootName);
            if (old != null) Object.Destroy(old);
        }

        static void Text(Transform parent, string s, Font font, int size, FontStyle style, Vector2 pos)
        {
            var go = new GameObject("Text");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.text = s;
            t.font = font; t.fontSize = size; t.fontStyle = style;
            t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Overflow; t.raycastTarget = false;
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 0.5f);
            rt.sizeDelta = new Vector2(1200f, 60f);
            rt.anchoredPosition = pos;
        }
    }
}
