using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// One shared dismissible info bubble. Used directly by stat-name tap targets (Task 11 — always
    /// available, no CTA, modal) and by <c>TipsService</c> for the Tips coaching layer (Task 12 —
    /// optionally with a CTA button, e.g. "Design your answer" opening the Designer; always non-modal).
    /// Only one bubble exists at a time — a new <see cref="Show"/> replaces whatever's already up.
    ///
    /// <para><b>modal</b> (default true): a full-screen invisible catcher sits behind the card and
    /// consumes the dismissing tap — right for a deliberate stat-reference popup the player opened on
    /// purpose. <b>Non-modal</b> (Task 12's coaching bubbles) has NO backdrop and nothing raycast-blocking
    /// except the card itself: the card's own tap still dismisses it (and the CTA button still works),
    /// but a press anywhere else passes straight through un-consumed to whatever it would normally hit —
    /// a board-hex tap both dismisses the bubble AND performs the tap's own action, in the same press.
    /// See <see cref="OutsideTapDismiss"/>, which polls rather than intercepts.</para>
    ///
    /// <para>The card auto-sizes to the label's <c>preferredHeight</c> (the longest StatInfo entry,
    /// Defense at 178 chars, overflowed the old fixed 360x120 card) so the text always fits, capped to
    /// still fit on a portrait screen. This applies to both modes — the label text drives the height.</para>
    /// </summary>
    public static class TipBubble
    {
        public const string RootName = "TipBubbleCanvas";

        const float Width = 360f;
        const float NoCtaPad = 32f;   // 16px top + 16px bottom around the label when there's no CTA
        const float CtaBlock = 60f;   // vertical space reserved below the label for the CTA button/gap
        const float MinLabelH = 40f;  // floor so a one-line tip isn't a postage stamp
        const float SlackH = 2f;      // rounding cushion so GetWorldCorners never clips the last line

        /// <summary><paramref name="screenPos"/> is a SCREEN-space position (physical/logical pixels,
        /// origin bottom-left) — the one coordinate space every caller can reach regardless of source:
        /// a UI element via <c>RectTransformUtility.WorldToScreenPoint(null, rt.TransformPoint(local))</c>
        /// (camera null — every UiKit canvas is ScreenSpaceOverlay), or a 3D scene object via
        /// <c>Camera.main.WorldToScreenPoint(worldPos)</c> (Task 12's unit-selection trigger uses this).
        /// Converted to this bubble's own canvas-local space via
        /// <c>RectTransformUtility.ScreenPointToLocalPointInRectangle</c> (camera null, same reason).
        /// Pass <c>new Vector2(Screen.width / 2f, Screen.height / 2f)</c> for a screen-centered bubble
        /// (Task 12's non-anchored triggers use this). See the type doc for <paramref name="modal"/>.</summary>
        public static void Show(string text, Vector2 screenPos, string cta = null, System.Action onCta = null, bool modal = true)
        {
            Dismiss();

            var root = UiKit.Canvas(RootName, UiKit.OrderTooltip + 1, null);
            var canvasRt = root.GetComponent<RectTransform>();
            RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRt, screenPos, null, out Vector2 localAnchor);

            if (modal)
            {
                // full-screen invisible catcher BEHIND the card — taps anywhere off the card dismiss too
                // (and, deliberately, EAT that tap — right for a popup the player opened on purpose)
                var backdrop = new GameObject("Backdrop");
                backdrop.transform.SetParent(root.transform, false);
                var bdImg = backdrop.AddComponent<Image>();
                bdImg.color = new Color(0f, 0f, 0f, 0f);
                UiKit.Stretch(backdrop.GetComponent<RectTransform>());
                backdrop.AddComponent<Button>().onClick.AddListener(Dismiss);
            }

            var card = UiKit.Panel(root.transform, "Bubble", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);

            // measure first: Text.preferredHeight only depends on the label's OWN rect WIDTH (already set
            // below to Width - 32), never on the height passed here — so this placeholder height is moot.
            var label = UiKit.Label(card.transform, text, 0f, -16f, Width - 32f, MinLabelH,
                                    UiKit.SizeBody, TextAnchor.UpperLeft, UiKit.TextMain);
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Overflow;

            float pad = cta != null ? CtaBlock : NoCtaPad;
            float maxH = canvasRt.rect.height > 0f ? canvasRt.rect.height * 0.85f : 700f; // cap to fit portrait
            float textH = Mathf.Max(MinLabelH, label.preferredHeight) + SlackH;
            float h = Mathf.Min(textH + pad, maxH);

            crt.sizeDelta = new Vector2(Width, h);
            crt.anchoredPosition = ClampToCanvas(localAnchor, canvasRt, Width, h);

            var labelRt = label.GetComponent<RectTransform>();
            UiKit.SetRect(labelRt, 0f, -16f, Width - 32f, h - pad); // resync now h is known — matters
                                                                     // whenever the clamp above shrank it

            card.AddComponent<Button>().onClick.AddListener(Dismiss); // tap the card's own body → dismiss

            if (cta != null)
                UiKit.Button(card.transform, cta, 0f, -(h - 46f), Width - 60f, 38f,
                            () => { Dismiss(); onCta?.Invoke(); }, UiKit.ButtonStyle.Cta);

            if (!modal) card.AddComponent<OutsideTapDismiss>().Init(crt);
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

        /// <summary>Non-modal self-dismiss (Task 12 review, amendment 1). With no backdrop, nothing here
        /// blocks a board/world raycast or another UI element's own click — this only WATCHES. Every
        /// Update it polls whether a brand-new press this frame landed outside the card and, if so, calls
        /// Dismiss(). It never marks the input "used" (no OnPointerDown/EventSystem handler — a plain
        /// Pointer.current read), so whatever else is listening for that same press — UnitInputController's
        /// own tap loop included — sees it too, in the very same frame, and acts on it normally.</summary>
        sealed class OutsideTapDismiss : MonoBehaviour
        {
            RectTransform _card;
            int _armedFrame; // ignore the frame this bubble was created on (defensive — Update doesn't
                              // start firing until the next frame anyway, but this costs nothing)

            public void Init(RectTransform card)
            {
                _card = card;
                _armedFrame = Time.frameCount;
            }

            void Update()
            {
                if (_card == null || Time.frameCount <= _armedFrame) return;
                var pointer = DeviceInput.Allowed ? Pointer.current : null;
                if (pointer == null || !pointer.press.wasPressedThisFrame) return;
                Vector2 pos = pointer.position.ReadValue();
                if (!RectTransformUtility.RectangleContainsScreenPoint(_card, pos, null))
                    Dismiss();
            }
        }
    }
}
