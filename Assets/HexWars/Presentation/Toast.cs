using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// A brief on-screen message, bottom-centre, that auto-hides after a couple of seconds. Used to explain
    /// why an action was rejected (deploy/move/claim) and to announce turn handovers. Call <see cref="Show"/>;
    /// the default background is the rejection red, pass a colour for non-error messages.
    /// </summary>
    public sealed class Toast : MonoBehaviour
    {
        static readonly Color RejectionRed = new Color(0.55f, 0.12f, 0.12f, 0.94f);

        static Toast _inst;
        GameObject _bg;
        Image _bgImage;
        Text _text;
        float _until;

        public static void Show(string message) => Show(message, RejectionRed);

        /// <summary><paramref name="top"/> places the box just under the HUD banner instead of
        /// bottom-centre — used for turn announcements so they never cover the battlefield or the
        /// damage popups rising from it.</summary>
        public static void Show(string message, Color background, bool top = false, float seconds = 2.8f)
        {
            if (_inst == null)
            {
                var go = new GameObject("Toast");
                DontDestroyOnLoad(go);
                _inst = go.AddComponent<Toast>();
                _inst.BuildUi();
            }
            _inst.Place(top);
            _inst._bgImage.color = background;
            _inst._text.text = message;
            _inst._bg.SetActive(true);
            _inst._until = Time.unscaledTime + seconds;
        }

        void Place(bool top)
        {
            var brt = _bg.GetComponent<RectTransform>();
            if (top)
            {
                brt.anchorMin = brt.anchorMax = new Vector2(0.5f, 1f);
                brt.pivot = new Vector2(0.5f, 1f);
                brt.anchoredPosition = new Vector2(0f, -50f); // clear of the 46px banner
            }
            else
            {
                brt.anchorMin = brt.anchorMax = new Vector2(0.5f, 0f);
                brt.pivot = new Vector2(0.5f, 0f);
                brt.anchoredPosition = new Vector2(0f, 170f);
            }
        }

        void BuildUi()
        {
            var canvasGo = new GameObject("ToastCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 800;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1200f, 800f);
            scaler.matchWidthOrHeight = 0.5f;

            _bg = new GameObject("Bg");
            _bg.transform.SetParent(canvasGo.transform, false);
            _bgImage = _bg.AddComponent<Image>();
            _bgImage.color = RejectionRed;
            var brt = _bg.GetComponent<RectTransform>();
            brt.anchorMin = brt.anchorMax = new Vector2(0.5f, 0f);
            brt.pivot = new Vector2(0.5f, 0f);
            brt.sizeDelta = new Vector2(580f, 50f);
            brt.anchoredPosition = new Vector2(0f, 170f);

            var tGo = new GameObject("Text");
            tGo.transform.SetParent(_bg.transform, false);
            _text = tGo.AddComponent<Text>();
            _text.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _text.fontSize = 19; _text.color = Color.white; _text.alignment = TextAnchor.MiddleCenter;
            _text.horizontalOverflow = HorizontalWrapMode.Overflow;
            var trt = tGo.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero; trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(12f, 0f); trt.offsetMax = new Vector2(-12f, 0f);

            _bg.SetActive(false);
        }

        void Update()
        {
            if (_bg != null && _bg.activeSelf && Time.unscaledTime > _until) _bg.SetActive(false);
        }
    }
}
