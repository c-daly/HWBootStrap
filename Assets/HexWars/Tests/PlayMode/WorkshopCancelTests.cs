using System.Collections;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.TestTools;
using UnityEngine.UI;

namespace HexWars.Presentation.PlayModeTests
{
    public sealed class WorkshopCancelTests
    {
        GameObject _host, _ownedEvents;
        TacticalHud _hud;
        EscapeMenu _menu;
        InputField _field;

        [UnitySetUp]
        public IEnumerator SetUp()
        {
            _host = new GameObject("Workshop cancel fixture", typeof(GameBootstrap));
            var game = _host.GetComponent<GameBootstrap>(); game.enabled = false;
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            typeof(GameBootstrap).GetProperty("State").SetValue(game, new GameState(board,
                GameConfig.Default(), new[] { new PlayerState(PlayerId.Player0, 30),
                    new PlayerState(PlayerId.Player1, 30) }, PlayerId.Player0, 1, 1));
            _host.AddComponent<DesignPanel>();
            _hud = _host.AddComponent<TacticalHud>(); _menu = _host.AddComponent<EscapeMenu>();
            if (EventSystem.current == null) _ownedEvents = new GameObject("Workshop events", typeof(EventSystem));
            yield return null; yield return null;
            _hud.SetWorkshop(true);
            _field = _host.GetComponentsInChildren<InputField>().Single(f => f.name == "Unit name");
            _field.text = "Original"; _field.onEndEdit.Invoke(_field.text);
            EventSystem.current.SetSelectedGameObject(_field.gameObject);
            _field.ActivateInputField(); yield return null;
            _field.text = "Discard this draft";
        }

        [UnityTearDown]
        public IEnumerator TearDown()
        {
            EventSystem.current?.SetSelectedGameObject(null);
            Object.Destroy(_host);
            if (_ownedEvents != null) Object.Destroy(_ownedEvents);
            foreach (var hub in Object.FindObjectsByType<WebGlInputHub>(FindObjectsSortMode.None))
                Object.Destroy(hub.gameObject);
            yield return null;
        }

        [UnityTest]
        public IEnumerator DesktopEscapeCancelsTheEditAndImmediatelyDismissesDesigner()
        {
            Assert.That(_menu.HandleEscape(), Is.True);
            AssertCancelledAndClosed();
            yield return AssertCancelledOnReopen();
        }

        [UnityTest]
        public IEnumerator BrowserEscapeCancelsExactlyOnceBeforeDismissingDesigner()
        {
            int cancellations = 0;
            var bridge = _field.GetComponent<WebGlInputBridge>();
            bridge.CancelRequested += () => cancellations++;
            bridge.Receive("cancel", "Discard this draft");
            Assert.That(cancellations, Is.EqualTo(1));
            bridge.Receive("blur", "Stale browser draft");
            AssertCancelledAndClosed();
            yield return AssertCancelledOnReopen();
        }

        void AssertCancelledAndClosed()
        {
            Assert.That(_field.text, Is.EqualTo("Original"));
            Assert.That(_hud.WorkshopOpen, Is.False, "One Escape must close the designer immediately.");
            Assert.That(_field.gameObject.activeInHierarchy, Is.False);
            Assert.That(EventSystem.current.currentSelectedGameObject, Is.Null);
            Assert.That(GameObject.Find("EscapeMenuCanvas"), Is.Null);
        }

        IEnumerator AssertCancelledOnReopen()
        {
            yield return null;
            _hud.SetWorkshop(true);
            Assert.That(_field.text, Is.EqualTo("Original"), "Closing must not commit the discarded draft.");
        }
    }
}
