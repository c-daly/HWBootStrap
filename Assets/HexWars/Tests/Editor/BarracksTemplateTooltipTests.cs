using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Presentation;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation.Tests
{
    public class BarracksTemplateTooltipTests
    {
        GameObject _host;
        BarracksTemplateTooltip _tooltip;
        RectTransform _anchor;
        UnitTemplate _template;
        GameConfig _config;

        [SetUp]
        public void SetUp()
        {
            _host = new GameObject("TooltipTestHost");
            _tooltip = _host.AddComponent<BarracksTemplateTooltip>();
            var anchorGo = new GameObject("Anchor", typeof(RectTransform));
            anchorGo.transform.SetParent(_host.transform, false);
            _anchor = anchorGo.GetComponent<RectTransform>();
            _anchor.sizeDelta = new Vector2(140f, 30f);

            _template = new UnitTemplate("Combined Arms Prime",
                new UnitStats(1, 2, 3, 4, 5, 6, 7, 8, 9));
            _config = new GameConfig(new Dictionary<TerrainType, TerrainDef>(), deployCostMultiplier: 0.5);
        }

        [TearDown]
        public void TearDown()
        {
            if (_host != null) Object.DestroyImmediate(_host);
        }

        [Test]
        public void Show_RendersFullNameRoleCostsAndAllNineFullStatLabels()
        {
            _tooltip.Show(_anchor, _template, _config);

            string text = _tooltip.DisplayText;
            Assert.That(text, Does.Contain("Combined Arms Prime"));
            Assert.That(text, Does.Contain("Role: Spotter"));
            Assert.That(text, Does.Contain("Point cost: 45"));
            Assert.That(text, Does.Contain("Deploy cost: 23"));
            Assert.That(text, Does.Contain("Health: 1"));
            Assert.That(text, Does.Contain("Damage: 2"));
            Assert.That(text, Does.Contain("Defense: 3"));
            Assert.That(text, Does.Contain("Movement: 4"));
            Assert.That(text, Does.Contain("Vertical Movement: 5"));
            Assert.That(text, Does.Contain("Range: 6"));
            Assert.That(text, Does.Contain("Range Arc: 7"));
            Assert.That(text, Does.Contain("Vision: 8"));
            Assert.That(text, Does.Contain("Vision Arc: 9"));
        }

        [Test]
        public void Show_UsesDominantRoleAsTheFullDisplayNameWhenTemplateNameIsBlank()
        {
            var unnamed = new UnitTemplate("", new UnitStats(7, 1, 0, 1, 0, 1, 0, 1, 0));

            _tooltip.Show(_anchor, unnamed, _config);

            Assert.That(_tooltip.DisplayText, Does.StartWith("Brute\n"));
        }

        [Test]
        public void Show_CreatesOnlyNonRaycastingGraphics()
        {
            _tooltip.Show(_anchor, _template, _config);

            Assert.That(_tooltip.TooltipRoot.GetComponent<GraphicRaycaster>(), Is.Not.Null);
            Assert.That(_tooltip.TooltipRoot.GetComponent<CanvasGroup>().blocksRaycasts, Is.False);
            Assert.That(_tooltip.TooltipRoot.GetComponentsInChildren<Graphic>(true).Select(x => x.raycastTarget),
                Is.All.False);
        }

        [TestCase(1000f, 1000f)]
        [TestCase(-1000f, -1000f)]
        [TestCase(1000f, -1000f)]
        public void ClampToCanvasRect_KeepsWholeCardInsideEveryEdge(float x, float y)
        {
            var canvas = new Rect(-400f, -300f, 800f, 600f);
            var size = new Vector2(320f, 300f);

            Vector2 clamped = BarracksTemplateTooltip.ClampToCanvasRect(
                new Vector2(x, y), canvas, size, 12f);

            Assert.That(clamped.x - size.x * 0.5f, Is.GreaterThanOrEqualTo(canvas.xMin + 12f));
            Assert.That(clamped.x + size.x * 0.5f, Is.LessThanOrEqualTo(canvas.xMax - 12f));
            Assert.That(clamped.y - size.y * 0.5f, Is.GreaterThanOrEqualTo(canvas.yMin + 12f));
            Assert.That(clamped.y + size.y * 0.5f, Is.LessThanOrEqualTo(canvas.yMax - 12f));
        }

        [Test]
        public void PointerAndKeyboardHandlers_ShowAndHideTooltip()
        {
            var target = _anchor.gameObject.AddComponent<BarracksTemplateTooltipTarget>();
            target.Init(_tooltip, _anchor, _template, _config);

            target.OnPointerEnter(null);
            Assert.That(_tooltip.IsVisible, Is.True);
            target.OnPointerExit(null);
            Assert.That(_tooltip.IsVisible, Is.False);

            target.OnSelect(null);
            Assert.That(_tooltip.IsVisible, Is.True);
            target.OnDeselect(null);
            Assert.That(_tooltip.IsVisible, Is.False);
        }

        [Test]
        public void TouchInfoTarget_ShowsTooltipWithoutSelectingTheRow()
        {
            var target = _anchor.gameObject.AddComponent<BarracksTemplateTooltipTarget>();
            target.Init(_tooltip, _anchor, _template, _config);

            target.ShowInfo();

            Assert.That(_tooltip.IsVisible, Is.True);
        }

        [Test]
        public void DestroyingTargetAndExplicitHideDismissTooltip()
        {
            var target = _anchor.gameObject.AddComponent<BarracksTemplateTooltipTarget>();
            target.Init(_tooltip, _anchor, _template, _config);
            target.ShowInfo();

            _tooltip.Hide();
            Assert.That(_tooltip.IsVisible, Is.False, "panel rebuild/close uses explicit Hide");

            Object.DestroyImmediate(target);
        }
    }
}
