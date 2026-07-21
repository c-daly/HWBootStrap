using System;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The single source of UI style: one canvas convention (1600x900 @ match 0.5), one palette, one
    /// type scale, one rounded-corner sprite, one widget factory. Every runtime menu/panel builds on
    /// this so the game reads as one product instead of nine hand-rolled forms. Zero assets: the
    /// rounded sprite is generated once at runtime (WebGL build stays art-free).
    /// </summary>
    public static class UiKit
    {
        // ---- palette ----
        public static readonly Color Bg            = Hex("0A0E1C");
        public static readonly Color Surface       = Hex("161B2C");
        public static readonly Color SurfaceBorder = Hex("2A3350");
        public static readonly Color Accent        = Hex("45AEFF");
        public static readonly Color AccentDim     = Hex("27476B");
        public static readonly Color CtaGreen      = Hex("33845C");
        public static readonly Color Danger        = Hex("B04040");
        public static readonly Color TextMain      = Color.white;
        public static readonly Color TextDim       = Hex("9AA3B8");
        public static readonly Color TextFaint     = Hex("6C7488");
        public static readonly Color InputBg       = Hex("EDF1F8");
        public static readonly Color InputText     = Hex("10131C");

        // ---- type scale ----
        public const int SizeTitle = 26, SizeHeading = 20, SizeBody = 16, SizeCaption = 13;

        // ---- canvas sorting orders (single authority; comments = who owns it) ----
        public const int OrderHud = 500;      // GameHud top bar
        public const int OrderPanels = 700;   // barracks / design side panels
        public const int OrderTooltip = 750;  // unit tooltip
        public const int OrderBanner = 850;   // game-over band
        public const int OrderRules = 900;    // rules/help popup
        public const int OrderEscape = 980;   // in-game escape menu — above rules, below title/lobby
        public const int OrderMenu = 1000;    // title / lobby screens
        public const int OrderToast = 1200;   // transient feedback outranks every screen, including
                                               // menus (join errors must be visible over the title/browser)

        static Color Hex(string rgb)
        {
            byte r = Convert.ToByte(rgb.Substring(0, 2), 16);
            byte g = Convert.ToByte(rgb.Substring(2, 2), 16);
            byte b = Convert.ToByte(rgb.Substring(4, 2), 16);
            return new Color32(r, g, b, 255);
        }

        public static Font Font()
        {
            var f = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            if (f == null) f = Resources.GetBuiltinResource<Font>("Arial.ttf");
            return f;
        }

        public static void EnsureEventSystem()
        {
            if (UnityEngine.Object.FindAnyObjectByType<EventSystem>() != null) return;
            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            var module = es.AddComponent<InputSystemUIInputModule>();
            module.AssignDefaultActions(); // without actions the module silently ignores UI input
        }

        /// <summary>ScreenSpaceOverlay canvas on the shared 1600x900 @ match 0.5 convention.</summary>
        public static GameObject Canvas(string name, int sortingOrder, Transform parent)
        {
            var go = new GameObject(name);
            if (parent != null) go.transform.SetParent(parent, false);
            var canvas = go.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = sortingOrder;
            var scaler = go.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            go.AddComponent<GraphicRaycaster>();
            // Portrait pass (Task 8): callers read this canvas's own RectTransform.rect.width
            // moments later (SetupForm/GameBrowser/TitleScreen/BarracksPanel/DesignPanel/Toast's
            // AvailWidth clamps) to size children against the actually-available width. Without this,
            // a freshly added CanvasScaler hasn't run its layout pass yet within the same synchronous
            // call, so .rect still reports raw Screen.width/height (390 in portrait) instead of the
            // scaled canvas-space size (~815 in portrait) — confirmed live: an unforced read clamped
            // SetupForm's 700-wide card down to 350, which then let its fixed-offset children (e.g.
            // the Back button at x=-300) overflow past the card's own shrunken edge. Forcing the
            // layout pass here fixes it at the source for every caller.
            UnityEngine.Canvas.ForceUpdateCanvases();
            return go;
        }

        static Sprite _rounded;

        /// <summary>A 9-sliced rounded-rect sprite (generated once). Radius reads ~8px at reference scale.</summary>
        public static Sprite Rounded()
        {
            if (_rounded != null) return _rounded;
            const int s = 32, r = 10;
            var tex = new Texture2D(s, s, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color32[s * s];
            for (int y = 0; y < s; y++)
                for (int x = 0; x < s; x++)
                {
                    // distance outside the rounded-rect core; soft 1px edge for cheap anti-aliasing
                    float dx = Mathf.Max(0, Mathf.Max(r - x, x - (s - 1 - r)));
                    float dy = Mathf.Max(0, Mathf.Max(r - y, y - (s - 1 - r)));
                    float d = Mathf.Sqrt(dx * dx + dy * dy);
                    float a = Mathf.Clamp01(r + 0.5f - d);
                    px[y * s + x] = new Color32(255, 255, 255, (byte)(a * 255f));
                }
            tex.SetPixels32(px);
            tex.Apply();
            _rounded = Sprite.Create(tex, new Rect(0, 0, s, s), new Vector2(0.5f, 0.5f), 100f, 0,
                                     SpriteMeshType.FullRect, new Vector4(r + 2, r + 2, r + 2, r + 2));
            return _rounded;
        }

        /// <summary>Rounded surface image. Caller positions it (SetRect/Stretch/manual anchors).</summary>
        public static Image Panel(Transform parent, string name, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.sprite = Rounded();
            img.type = Image.Type.Sliced;
            img.color = color;
            return img;
        }

        public static void Stretch(RectTransform rt)
        {
            rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
            rt.offsetMin = Vector2.zero; rt.offsetMax = Vector2.zero;
        }

        /// <summary>Top-center anchored (x, y, w, h) — the convention every form in the game uses.</summary>
        public static void SetRect(RectTransform rt, float x, float y, float w, float h)
        {
            rt.anchorMin = rt.anchorMax = new Vector2(0.5f, 1f);
            rt.pivot = new Vector2(0.5f, 1f);
            rt.sizeDelta = new Vector2(w, h);
            rt.anchoredPosition = new Vector2(x, y);
        }

        public static Text Label(Transform parent, string text, float x, float y, float w, float h,
                                 int size, TextAnchor anchor, Color? color = null)
        {
            var go = new GameObject("Label");
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = Font(); t.fontSize = size; t.color = color ?? TextMain;
            t.alignment = anchor; t.text = text;
            t.horizontalOverflow = HorizontalWrapMode.Overflow;
            t.verticalOverflow = VerticalWrapMode.Overflow;
            t.raycastTarget = false;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            return t;
        }

        public enum ButtonStyle { Primary, Secondary, Cta, Danger }

        static Color BaseColor(ButtonStyle s) => s switch
        {
            ButtonStyle.Cta => CtaGreen,
            ButtonStyle.Danger => Danger,
            ButtonStyle.Secondary => new Color(0.13f, 0.16f, 0.24f, 1f),
            _ => AccentDim,
        };

        /// <summary>Rounded button with hover/pressed tints (uGUI ColorBlock, so states come free).</summary>
        public static Button Button(Transform parent, string label, float x, float y, float w, float h,
                                    Action onClick, ButtonStyle style = ButtonStyle.Primary, int fontSize = 0)
        {
            var go = new GameObject("Button");
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.sprite = Rounded();
            img.type = Image.Type.Sliced;
            img.color = Color.white; // tint comes from the ColorBlock so hover/pressed work
            var b = go.AddComponent<Button>();
            b.targetGraphic = img;
            var cb = b.colors;
            var baseC = BaseColor(style);
            cb.normalColor = baseC;
            cb.highlightedColor = baseC * 1.18f;
            cb.pressedColor = baseC * 0.82f;
            cb.selectedColor = baseC;
            cb.disabledColor = new Color(baseC.r, baseC.g, baseC.b, 0.35f);
            cb.fadeDuration = 0.08f;
            b.colors = cb;
            b.onClick.AddListener(() => onClick());
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            Label(go.transform, label, 0f, 0f, w, h,
                  fontSize > 0 ? fontSize : (style == ButtonStyle.Cta ? SizeHeading + 2 : SizeBody + 2),
                  TextAnchor.MiddleCenter);
            return b;
        }

        /// <summary>A complete legacy uGUI input field, styled to match the game's light input boxes.</summary>
        public static InputField InputField(Transform parent, string initial, float x, float y,
                                            float w, float h, string placeholder = "")
        {
            EnsureEventSystem();

            var go = new GameObject("InputField");
            go.transform.SetParent(parent, false);
            var image = go.AddComponent<Image>();
            image.sprite = Rounded();
            image.type = Image.Type.Sliced;
            image.color = InputBg;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);

            var valueGo = new GameObject("Text");
            valueGo.transform.SetParent(go.transform, false);
            var valueText = valueGo.AddComponent<Text>();
            valueText.font = Font();
            valueText.fontSize = SizeBody + 2;
            valueText.color = InputText;
            valueText.alignment = TextAnchor.MiddleLeft;
            valueText.supportRichText = false;
            StretchWithInset(valueGo.GetComponent<RectTransform>(), 12f, 8f);

            var placeholderGo = new GameObject("Placeholder");
            placeholderGo.transform.SetParent(go.transform, false);
            var placeholderText = placeholderGo.AddComponent<Text>();
            placeholderText.font = Font();
            placeholderText.fontSize = SizeBody + 2;
            placeholderText.fontStyle = FontStyle.Italic;
            placeholderText.color = new Color(InputText.r, InputText.g, InputText.b, 0.45f);
            placeholderText.alignment = TextAnchor.MiddleLeft;
            placeholderText.supportRichText = false;
            placeholderText.text = placeholder ?? string.Empty;
            placeholderText.raycastTarget = false;
            StretchWithInset(placeholderGo.GetComponent<RectTransform>(), 12f, 8f);

            var field = go.AddComponent<InputField>();
            field.targetGraphic = image;
            var colors = field.colors;
            colors.normalColor = InputBg;
            colors.highlightedColor = Color.Lerp(InputBg, Color.white, 0.12f);
            colors.pressedColor = Color.Lerp(InputBg, Color.black, 0.08f);
            colors.selectedColor = InputBg;
            colors.disabledColor = new Color(InputBg.r, InputBg.g, InputBg.b, 0.55f);
            field.colors = colors;
            image.color = InputBg;
            field.textComponent = valueText;
            field.placeholder = placeholderText;
            field.lineType = UnityEngine.UI.InputField.LineType.SingleLine;
            field.text = initial ?? string.Empty;
            return field;
        }

        /// <summary>Numeric input plus committed-value bookkeeping and an inline validation message.</summary>
        public static InlineIntBinding IntField(Transform parent, int initial, float x, float y,
                                                float w, float h, int min, int max,
                                                bool blankMeansZero, Action<int> setter)
        {
            var field = InputField(parent, initial.ToString(), x, y, w, h);
            field.contentType = UnityEngine.UI.InputField.ContentType.IntegerNumber;
            var error = Label(parent, string.Empty, x, y - h - 2f, w, 20f,
                              SizeCaption, TextAnchor.UpperLeft, Danger);
            return new InlineIntBinding(field, error, initial, min, max, blankMeansZero, setter);
        }

        static void StretchWithInset(RectTransform rt, float horizontal, float vertical)
        {
            rt.anchorMin = Vector2.zero;
            rt.anchorMax = Vector2.one;
            rt.offsetMin = new Vector2(horizontal, vertical);
            rt.offsetMax = new Vector2(-horizontal, -vertical);
        }

        /// <summary>Selected-state tint for toggle-style buttons (mode pickers, checkboxes-as-buttons).</summary>
        public static void SetToggled(Button b, bool on)
        {
            var cb = b.colors;
            var baseC = on ? new Color(0.27f, 0.50f, 0.82f, 1f) : new Color(0.13f, 0.16f, 0.24f, 1f);
            cb.normalColor = baseC;
            cb.highlightedColor = baseC * 1.18f;
            cb.pressedColor = baseC * 0.82f;
            cb.selectedColor = baseC;
            b.colors = cb;
        }

        /// <summary>Character-budget truncation for player-authored names in fixed-width slots
        /// ("Maximilian Longname" → "Maximilian…"). uGUI Text has no per-string ellipsis mode, and
        /// names may be up to 20 chars (UnitTemplate.Sanitize) while row/header slots fit fewer.</summary>
        public static string Ellipsize(string s, int max)
        {
            if (string.IsNullOrEmpty(s) || s.Length <= max) return s;
            return s.Substring(0, max - 1).TrimEnd() + "…";
        }

#if UNITY_WEBGL && !UNITY_EDITOR
        [System.Runtime.InteropServices.DllImport("__Internal")]
        static extern string HexWarsPrompt(string message, string current);
#endif
        /// <summary>Tap-to-type int prompt: browser prompt() on WebGL (the only reliable mobile
        /// keyboard), no-op in the editor (use the −/+ steppers there).</summary>
        public static int PromptInt(string label, int current)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            string s = HexWarsPrompt(label, current.ToString());
            return int.TryParse(s, out var v) ? v : current;
#else
            return current;
#endif
        }

        /// <summary>Same for free text (join-by-code). Returns null when unavailable/cancelled.</summary>
        public static string PromptText(string label, string current)
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            return HexWarsPrompt(label, current);
#else
            return null;
#endif
        }

        /// <summary>Light input-style box showing an int; tap to type (WebGL), value clamped.</summary>
        public static Text ValueBox(Transform parent, string label, float x, float y, float w, float h,
                                    Func<int> get, Action<int> set, int min, int max)
        {
            var img = Panel(parent, "ValueBox", InputBg);
            var go = img.gameObject;
            var btn = go.AddComponent<Button>();
            btn.targetGraphic = img;
            SetRect(go.GetComponent<RectTransform>(), x, y, w, h);
            var t = Label(go.transform, get().ToString(), 0f, 0f, w, h, SizeBody + 4, TextAnchor.MiddleCenter, InputText);
            btn.onClick.AddListener(() => { set(Mathf.Clamp(PromptInt(label, get()), min, max)); t.text = get().ToString(); });
            return t;
        }
    }

    public sealed class InlineIntBinding
    {
        readonly int _min;
        readonly int _max;
        readonly bool _blankMeansZero;
        readonly Action<int> _setter;
        int _committed;
        bool _restoring;

        public InputField Field { get; }
        public Text Error { get; }

        internal InlineIntBinding(InputField field, Text error, int initial, int min, int max,
                                  bool blankMeansZero, Action<int> setter)
        {
            Field = field;
            Error = error;
            _committed = initial;
            _min = min;
            _max = max;
            _blankMeansZero = blankMeansZero;
            _setter = setter ?? (_ => { });
            Field.onEndEdit.AddListener(_ =>
            {
                if (_restoring) return;
                Commit();
            });
        }

        public bool Commit()
        {
            if (!InlineFieldRules.TryInt(Field.text, _min, _max, _blankMeansZero,
                                         out int value, out string error))
            {
                Error.text = error;
                return false;
            }

            _committed = value;
            Field.SetTextWithoutNotify(value.ToString());
            Error.text = string.Empty;
            _setter(value);
            return true;
        }

        public void Restore()
        {
            _restoring = true;
            Field.SetTextWithoutNotify(_committed.ToString());
            Error.text = string.Empty;
            Field.DeactivateInputField();
            _restoring = false;
        }

        public void SetCommittedValue(int value)
        {
            _committed = value;
            _restoring = false;
            Field.SetTextWithoutNotify(value.ToString());
            Error.text = string.Empty;
        }
    }
}
