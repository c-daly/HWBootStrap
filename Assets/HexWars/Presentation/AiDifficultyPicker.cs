using System;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>Player-facing names and available choices come exclusively from the deployment catalog.</summary>
    public static class AiDifficultyPicker
    {
        public static void Open(Transform parent, string selectedId, Action<string> choose)
        {
            var choices = AiModelSettings.Catalog.difficulties;
            var canvas = UiKit.Canvas("AiDifficultyPicker", UiKit.OrderMenu + 10, parent);
            // This overlay is nested inside SetupCanvas. Unlike a root canvas, Unity does not
            // resize its default 100x100 RectTransform to the viewport automatically.
            UiKit.Stretch((RectTransform)canvas.transform);
            var overlay = canvas.GetComponent<Canvas>();
            // Unity discards a nested canvas's order when overrideSorting is enabled;
            // set the order afterwards so the popup renders and receives clicks above SetupCanvas.
            overlay.overrideSorting = true;
            overlay.sortingOrder = UiKit.OrderMenu + 10;
            Canvas.ForceUpdateCanvases();
            void Close()
            {
                if (Application.isPlaying) UnityEngine.Object.Destroy(canvas);
                else UnityEngine.Object.DestroyImmediate(canvas);
            }
            var dim = UiKit.Panel(canvas.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.85f));
            UiKit.Stretch(dim.rectTransform);
            dim.gameObject.AddComponent<Button>().onClick.AddListener(Close);
            float height = Mathf.Min(550, 125 + choices.Length * 52);
            var card = UiKit.Panel(canvas.transform, "Choices", UiKit.Surface);
            var rect = card.rectTransform;
            rect.anchorMin = rect.anchorMax = rect.pivot = new Vector2(0.5f, 0.5f);
            rect.sizeDelta = new Vector2(Mathf.Min(440, ((RectTransform)canvas.transform).rect.width - 40), height);
            rect.anchoredPosition = Vector2.zero;
            UiKit.Label(card.transform, "Difficulty", 0, -20, rect.sizeDelta.x - 40, 36, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            var viewport = new GameObject("DifficultyList", typeof(RectTransform), typeof(RectMask2D), typeof(ScrollRect));
            viewport.transform.SetParent(card.transform, false);
            var vr = (RectTransform)viewport.transform;
            vr.anchorMin = new Vector2(0, 0); vr.anchorMax = new Vector2(1, 1);
            vr.offsetMin = new Vector2(20, 65); vr.offsetMax = new Vector2(-20, -65);
            var content = new GameObject("Content", typeof(RectTransform)); content.transform.SetParent(viewport.transform, false);
            var cr = (RectTransform)content.transform;
            cr.anchorMin = new Vector2(0, 1); cr.anchorMax = new Vector2(1, 1); cr.pivot = new Vector2(0.5f, 1);
            cr.sizeDelta = new Vector2(0, choices.Length * 52);
            var scroll = viewport.GetComponent<ScrollRect>(); scroll.content = cr; scroll.viewport = vr; scroll.horizontal = false;
            string selected = string.IsNullOrEmpty(selectedId) ? AiModelSettings.Catalog.default_difficulty_id : selectedId;
            for (int i = 0; i < choices.Length; i++)
            {
                var choice = choices[i];
                UiKit.Button(content.transform, choice.label, 0, -i * 52, rect.sizeDelta.x - 50, 44,
                    () => { choose(choice.id); Close(); }, choice.id == selected ? UiKit.ButtonStyle.Cta : UiKit.ButtonStyle.Secondary);
            }
            UiKit.Button(card.transform, "Cancel", 0, -height + 55, 180, 38, Close, UiKit.ButtonStyle.Secondary);
        }
    }
}
