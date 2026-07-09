using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// One shared dismissible info bubble. Used directly by stat-name tap targets (this task — always
    /// available, no CTA) and by <c>TipsService</c> for the Tips coaching layer (Task 12 — optionally with
    /// a CTA button, e.g. "Design your answer" opening the Designer). Only one bubble exists at a time —
    /// a new <see cref="Show"/> replaces whatever's already up. Tapping ANYWHERE (the bubble itself, or
    /// off it) dismisses it, per spec §6 ("tap anywhere or act → gone"); the CTA button, when present,
    /// sits on top and captures its own click first (dismiss + fire the action), never both.
    /// </summary>
    public static class TipBubble
    {
        public const string RootName = "TipBubbleCanvas";

        /// <summary><paramref name="screenPos"/> is a SCREEN-space position (physical/logical pixels,
        /// origin bottom-left) — the one coordinate space every caller can reach regardless of source:
        /// a UI element via <c>RectTransformUtility.WorldToScreenPoint(null, rt.TransformPoint(local))</c>
        /// (camera null — every UiKit canvas is ScreenSpaceOverlay), or a 3D scene object via
        /// <c>Camera.main.WorldToScreenPoint(worldPos)</c> (Task 12's unit-selection trigger uses this).
        /// Converted to this bubble's own canvas-local space via
        /// <c>RectTransformUtility.ScreenPointToLocalPointInRectangle</c> (camera null, same reason).
        /// Pass <c>new Vector2(Screen.width / 2f, Screen.height / 2f)</c> for a screen-centered bubble
        /// (Task 12's non-anchored triggers use this).</summary>
        public static void Show(string text, Vector2 screenPos, string cta = null, System.Action onCta = null)
        {
            Dismiss();

            var root = UiKit.Canvas(RootName, UiKit.OrderTooltip + 1, null);
            var canvasRt = root.GetComponent<RectTransform>();
            RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRt, screenPos, null, out Vector2 localAnchor);

            // full-screen invisible catcher BEHIND the card — taps anywhere off the card dismiss too
            var backdrop = new GameObject("Backdrop");
            backdrop.transform.SetParent(root.transform, false);
            var bdImg = backdrop.AddComponent<Image>();
            bdImg.color = new Color(0f, 0f, 0f, 0f);
            UiKit.Stretch(backdrop.GetComponent<RectTransform>());
            backdrop.AddComponent<Button>().onClick.AddListener(Dismiss);

            float w = 360f;
            float h = cta != null ? 170f : 120f;
            var card = UiKit.Panel(root.transform, "Bubble", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(w, h);
            crt.anchoredPosition = ClampToCanvas(localAnchor, canvasRt, w, h);
            card.AddComponent<Button>().onClick.AddListener(Dismiss); // tap the card's own body → dismiss

            var label = UiKit.Label(card.transform, text, 0f, -16f, w - 32f, h - (cta != null ? 60f : 32f),
                                    UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextMain);
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Overflow;

            if (cta != null)
                UiKit.Button(card.transform, cta, 0f, -(h - 46f), w - 60f, 38f,
                            () => { Dismiss(); onCta?.Invoke(); }, UiKit.ButtonStyle.Cta);
        }

        public static void Dismiss()
        {
            var old = GameObject.Find(RootName);
            if (old != null) Object.Destroy(old);
        }

        /// <summary>Keeps the bubble fully on-canvas even when anchored near a screen edge (a stat label
        /// near the panel edge, or an anchor-less call that defaults to screen center).</summary>
        static Vector2 ClampToCanvas(Vector2 anchorPos, RectTransform canvasRt, float w, float h)
        {
            float hw = canvasRt.rect.width * 0.5f, hh = canvasRt.rect.height * 0.5f;
            float x = Mathf.Clamp(anchorPos.x, -hw + w * 0.5f + 12f, hw - w * 0.5f - 12f);
            float y = Mathf.Clamp(anchorPos.y, -hh + h * 0.5f + 12f, hh - h * 0.5f - 12f);
            return new Vector2(x, y);
        }
    }
}
