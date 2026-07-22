using System.Text;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>A non-modal, non-raycasting stat card anchored to a barracks template row.</summary>
    public sealed class BarracksTemplateTooltip : MonoBehaviour
    {
        public const string RootName = "BarracksTemplateTooltipCanvas";
        const float Width = 320f, Height = 304f, Margin = 12f;

        GameObject _root;
        string _displayText = "";

        public bool IsVisible => _root != null && _root.activeSelf;
        public string DisplayText => _displayText;
        public GameObject TooltipRoot => _root;

        public void Show(RectTransform anchor, UnitTemplate template, GameConfig config)
        {
            if (anchor == null || config == null) return;
            Hide();
            _displayText = Format(template, config);
            _root = UiKit.Canvas(RootName, UiKit.OrderTooltip, transform);
            var group = _root.AddComponent<CanvasGroup>();
            group.interactable = false;
            group.blocksRaycasts = false;

            var canvasRt = _root.GetComponent<RectTransform>();
            var cardImage = UiKit.Panel(_root.transform, "TemplateStats", UiKit.Surface);
            cardImage.raycastTarget = false;
            var card = cardImage.rectTransform;
            card.anchorMin = card.anchorMax = new Vector2(0.5f, 0.5f);
            card.pivot = new Vector2(0.5f, 0.5f);
            card.sizeDelta = new Vector2(Width, Height);

            var label = UiKit.Label(card, _displayText, 0f, -14f, Width - 28f, Height - 28f,
                                    14, TextAnchor.UpperLeft, UiKit.TextMain);
            label.horizontalOverflow = HorizontalWrapMode.Wrap;
            label.verticalOverflow = VerticalWrapMode.Overflow;
            label.lineSpacing = 1.05f;

            Vector2 screenAnchor = RectTransformUtility.WorldToScreenPoint(null, anchor.position);
            RectTransformUtility.ScreenPointToLocalPointInRectangle(canvasRt, screenAnchor, null, out var localAnchor);
            float side = screenAnchor.x >= Screen.width * 0.5f ? -1f : 1f;
            localAnchor.x += side * (anchor.rect.width * 0.5f + Width * 0.5f + Margin);
            card.anchoredPosition = ClampToCanvasRect(localAnchor, canvasRt.rect, card.sizeDelta, Margin);
        }

        public void Hide()
        {
            if (_root == null) return;
            var old = _root;
            _root = null;
            old.SetActive(false);
            if (Application.isPlaying) Destroy(old); else DestroyImmediate(old);
        }

        void OnDisable() => Hide();
        void OnDestroy() => Hide();

        public static Vector2 ClampToCanvasRect(Vector2 desired, Rect canvasRect, Vector2 size, float margin)
        {
            float minX = canvasRect.xMin + size.x * 0.5f + margin;
            float maxX = canvasRect.xMax - size.x * 0.5f - margin;
            float minY = canvasRect.yMin + size.y * 0.5f + margin;
            float maxY = canvasRect.yMax - size.y * 0.5f - margin;
            return new Vector2(Mathf.Clamp(desired.x, minX, maxX), Mathf.Clamp(desired.y, minY, maxY));
        }

        static string Format(UnitTemplate template, GameConfig config)
        {
            var s = template.Stats;
            var role = Roles.Dominant(s);
            string name = string.IsNullOrEmpty(template.Name) ? role.ToString() : template.Name;
            var text = new StringBuilder(256);
            text.Append(name).Append('\n');
            text.Append("Role: ").Append(role).Append('\n');
            text.Append("Point cost: ").Append(s.PointCost).Append('\n');
            text.Append("Deploy cost: ").Append(Economy.DeployCost(s, config)).Append('\n');
            text.Append("Health: ").Append(s.Health).Append('\n');
            text.Append("Damage: ").Append(s.Damage).Append('\n');
            text.Append("Defense: ").Append(s.Defense).Append('\n');
            text.Append("Movement: ").Append(s.Movement).Append('\n');
            text.Append("Vertical Movement: ").Append(s.VerticalMovement).Append('\n');
            text.Append("Range: ").Append(s.Range).Append('\n');
            text.Append("Range Arc: ").Append(s.RangeArc).Append('\n');
            text.Append("Vision: ").Append(s.Vision).Append('\n');
            text.Append("Vision Arc: ").Append(s.VisionArc.ToString());
            return text.ToString();
        }
    }

    /// <summary>Forwards row hover and keyboard focus to the shared tooltip without blocking clicks.</summary>
    public sealed class BarracksTemplateTooltipTarget : MonoBehaviour,
        IPointerEnterHandler, IPointerExitHandler, ISelectHandler, IDeselectHandler
    {
        BarracksTemplateTooltip _tooltip;
        RectTransform _anchor;
        UnitTemplate _template;
        GameConfig _config;

        public void Init(BarracksTemplateTooltip tooltip, RectTransform anchor,
                         UnitTemplate template, GameConfig config)
        {
            _tooltip = tooltip;
            _anchor = anchor;
            _template = template;
            _config = config;
        }

        public void ShowInfo() => _tooltip?.Show(_anchor, _template, _config);
        public void OnPointerEnter(PointerEventData eventData) => ShowInfo();
        public void OnPointerExit(PointerEventData eventData) => _tooltip?.Hide();
        public void OnSelect(BaseEventData eventData) => ShowInfo();
        public void OnDeselect(BaseEventData eventData) => _tooltip?.Hide();
        void OnDisable() => _tooltip?.Hide();
        void OnDestroy() => _tooltip?.Hide();
    }
}
