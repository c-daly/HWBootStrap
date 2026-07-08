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
}
