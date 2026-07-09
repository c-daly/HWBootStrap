using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The end-of-game announcement: a full-width band across the centre of the screen with the
    /// result, how it was won, and a Main-menu button, over a light dim. Tap outside the button to
    /// dismiss and inspect the final board (the HUD banner keeps showing the result). Spawned by
    /// GameHud the moment the state turns game-over — previously the only signal was a rejection
    /// toast when you tried to act.
    /// </summary>
    public static class GameOverBanner
    {
        public const string RootName = "GameOverBanner";

        // taps within this window after showing are ignored — the click that ends the game (or a
        // reflexive next one) must not silently dismiss the result before the player registers it
        const float DismissGrace = 0.8f;
        static float _shownAt;

        public static void Show(string title, string subtitle, Color accent, System.Action onMainMenu = null)
        {
            Dismiss(); // never stack two
            _shownAt = Time.unscaledTime;

            var root = UiKit.Canvas(RootName, UiKit.OrderBanner, null);

            // light dim; tapping anywhere dismisses so the final board stays inspectable
            var dim = new GameObject("Dim");
            dim.transform.SetParent(root.transform, false);
            dim.AddComponent<Image>().color = new Color(0.02f, 0.03f, 0.06f, 0.55f);
            var drt = dim.GetComponent<RectTransform>();
            drt.anchorMin = Vector2.zero; drt.anchorMax = Vector2.one;
            drt.offsetMin = Vector2.zero; drt.offsetMax = Vector2.zero;
            dim.AddComponent<Button>().onClick.AddListener(() =>
            {
                if (Time.unscaledTime - _shownAt >= DismissGrace) Dismiss();
            });

            var band = new GameObject("Band");
            band.transform.SetParent(root.transform, false);
            var bandImg = band.AddComponent<Image>();
            bandImg.color = accent;
            bandImg.raycastTarget = false; // taps on the band fall through to the dim's dismiss button
            var brt = band.GetComponent<RectTransform>();
            brt.anchorMin = new Vector2(0f, 0.5f); brt.anchorMax = new Vector2(1f, 0.5f);
            brt.pivot = new Vector2(0.5f, 0.5f);
            brt.sizeDelta = new Vector2(0f, 200f);
            brt.anchoredPosition = Vector2.zero;

            var font = UiKit.Font();
            Text(band.transform, title, font, 46, FontStyle.Bold, new Vector2(0f, 42f));
            var subtitleText = Text(band.transform, subtitle + "   (tap outside to look at the board)", font, 19, FontStyle.Normal, new Vector2(0f, -6f));
            subtitleText.verticalOverflow = VerticalWrapMode.Overflow; // horizontalOverflow already wraps
                                                                        // into this fixed-height box; without
                                                                        // this a future two-line subtitle
                                                                        // would silently lose its 2nd line

            if (onMainMenu != null)
            {
                // band uses a centre anchor (pivot 0.5,0.5); UiKit.Button anchors top-centre (pivot
                // 0.5,1), so the y offset is re-based from the band's centre to its top edge
                // (half its 200-tall rect) to land the button on the exact same pixels as before.
                UiKit.Button(band.transform, "Main menu", 0f, -140f, 220f, 44f,
                             () => { Dismiss(); onMainMenu(); }, UiKit.ButtonStyle.Secondary);
            }
        }

        public static void Dismiss()
        {
            var old = GameObject.Find(RootName);
            if (old != null) Object.Destroy(old);
        }

        static Text Text(Transform parent, string s, Font font, int size, FontStyle style, Vector2 pos)
        {
            var go = new GameObject("Text");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.text = s;
            t.font = font; t.fontSize = size; t.fontStyle = style;
            t.color = Color.white; t.alignment = TextAnchor.MiddleCenter;
            t.horizontalOverflow = HorizontalWrapMode.Wrap; t.raycastTarget = false; // was Overflow — a
                                                                                      // narrow band must wrap, not run off-canvas
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0.5f); rt.anchorMax = new Vector2(1f, 0.5f); // stretch to the band's
            rt.offsetMin = new Vector2(20f, 0f); rt.offsetMax = new Vector2(-20f, 0f);  // own width, not a fixed 1200
            rt.sizeDelta = new Vector2(0f, 60f);
            rt.anchoredPosition = pos;
            return t;
        }
    }
}
