using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// A brief on-screen message, bottom-centre, that auto-hides after a couple of seconds. Used to explain
    /// why an action was rejected (deploy/move/claim), which is otherwise silent. Call <see cref="Show"/>.
    /// </summary>
    public sealed class Toast : MonoBehaviour
    {
        static Toast _inst;
        GameObject _bg;
        Text _text;
        float _until;

        public static void Show(string message)
        {
            if (_inst == null)
            {
                var go = new GameObject("Toast");
                DontDestroyOnLoad(go);
                _inst = go.AddComponent<Toast>();
                _inst.BuildUi();
            }
            _inst._text.text = message;
            _inst._bg.SetActive(true);
            _inst._until = Time.unscaledTime + 2.8f;
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
            _bg.AddComponent<Image>().color = new Color(0.55f, 0.12f, 0.12f, 0.94f);
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
