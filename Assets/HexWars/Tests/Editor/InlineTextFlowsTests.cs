using System;
using System.Reflection;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation.Tests
{
    public sealed class InlineTextFlowsTests
    {
        GameObject _gameObject;

        [TearDown]
        public void TearDown()
        {
            if (_gameObject != null) UnityEngine.Object.DestroyImmediate(_gameObject);
            foreach (var canvas in UnityEngine.Object.FindObjectsByType<Canvas>(FindObjectsSortMode.None))
                if (canvas != null) UnityEngine.Object.DestroyImmediate(canvas.gameObject);
            foreach (var hub in UnityEngine.Object.FindObjectsByType<WebGlInputHub>(FindObjectsSortMode.None))
                if (hub != null) UnityEngine.Object.DestroyImmediate(hub.gameObject);
            foreach (var eventSystem in UnityEngine.Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None))
                if (eventSystem != null) UnityEngine.Object.DestroyImmediate(eventSystem.gameObject);
        }

        [TestCase(" kq-7_kp ", "KQ7KP")]
        [TestCase("abcdefghijklmnopqrs", "ABCDEFGHIJKLMNOP")]
        [TestCase("room 42", "ROOM42")]
        public void RoomCodeNormalization_MatchesServerContract(string raw, string expected)
        {
            Assert.That(InvokeStatic<string>(typeof(TitleScreen), "NormalizeRoomCode", raw), Is.EqualTo(expected));
        }

        [Test]
        public void JoinByCode_UsesPersistentFieldAndEnterShowsInlineValidation()
        {
            var title = BuildTitle();
            var field = FindField(title, "Room code");
            var error = FindPrivate<Text>(title, "_roomCodeError");

            field.text = " --- ";
            Invoke(title, "OnJoinByCode");

            Assert.That(error.text, Is.Not.Empty);
            Assert.That(title.enabled, Is.True, "invalid input must leave the title screen open");
        }

        [Test]
        public void JoinByCode_SubmitEventUsesTheSameInlineValidation()
        {
            var title = BuildTitle();
            var field = FindField(title, "Room code");
            var error = FindPrivate<Text>(title, "_roomCodeError");

            field.text = " --- ";
            field.onSubmit.Invoke(field.text);

            Assert.That(error.text, Is.Not.Empty,
                "the native WebGL input bridge submits through InputField.onSubmit");
            Assert.That(title.enabled, Is.True);
        }

        [Test]
        public void JoinByCode_EscapeRestoresLastCommittedText()
        {
            var title = BuildTitle();
            var field = FindField(title, "Room code");
            field.text = "draft-code";
            Invoke(title, "RestoreRoomCodeEdit");

            Assert.That(field.text, Is.Empty);
            Assert.That(CurrentEventSystem().currentSelectedGameObject, Is.Null);

            field.text = "second-draft";
            var bridge = field.GetComponent<WebGlInputBridge>();
            bridge.OnSelect(null);
            Invoke(bridge, "Receive", "cancel", field.text);

            Assert.That(field.text, Is.Empty,
                "native Escape must restore the screen's committed room code, not the draft at focus");
        }

        [Test]
        public void Designer_EnterCommitsSanitizedNameAndEscapeRestoresIt()
        {
            var designer = BuildDesigner();
            var field = FindField(designer, "Unit name");
            string raw = "  ♥ Iron__Wolf-Prime1234567890 ";
            string expected = UnitTemplate.Sanitize(raw);

            field.text = raw;
            Invoke(designer, "CommitName");

            Assert.That(FindPrivate<string>(designer, "_name"), Is.EqualTo(expected));
            Assert.That(field.text, Is.EqualTo(expected));

            field.text = "discard this edit";
            Invoke(designer, "RestoreNameEdit");

            Assert.That(field.text, Is.EqualTo(expected));
            Assert.That(CurrentEventSystem().currentSelectedGameObject, Is.Null);

            field.text = "another discarded edit";
            var bridge = field.GetComponent<WebGlInputBridge>();
            bridge.OnSelect(null);
            Invoke(bridge, "Receive", "cancel", field.text);

            Assert.That(field.text, Is.EqualTo(expected),
                "native Escape must use the designer's committed-name restore behavior");
        }

        [Test]
        public void Designer_CommittedUnitNameAppearsInSummary()
        {
            var designer = BuildDesigner();
            var field = FindField(designer, "Unit name");
            string expected = UnitTemplate.Sanitize("  Iron Wolf  ");

            field.text = "  Iron Wolf  ";
            Invoke(designer, "CommitName");

            var summary = FindPrivate<Text>(designer, "_summary");
            Assert.That(summary.text, Does.Contain("Name: " + expected));
        }

        [Test]
        public void FocusHelpers_DistinguishTheSelectedFieldAndAnyActiveTextEntry()
        {
            var title = BuildTitle();
            var roomCode = FindField(title, "Room code");
            var otherGo = new GameObject("Other field", typeof(RectTransform), typeof(InputField));
            otherGo.transform.SetParent(_gameObject.transform, false);
            var other = otherGo.GetComponent<InputField>();
            var eventSystem = CurrentEventSystem();
            eventSystem.SetSelectedGameObject(other.gameObject);

            var owns = typeof(UiKit).GetMethod("InputOwnsFocus",
                BindingFlags.Static | BindingFlags.NonPublic);
            var any = typeof(UiKit).GetMethod("AnyInputOwnsFocus",
                BindingFlags.Static | BindingFlags.NonPublic);
            Assert.That(owns, Is.Not.Null, "keyboard handlers need a shared focus gate");
            Assert.That(any, Is.Not.Null, "the escape menu needs to yield to active text entry");
            Assert.That((bool)owns.Invoke(null, new object[] { roomCode }), Is.False);
            Assert.That((bool)any.Invoke(null, null), Is.True);

            eventSystem.SetSelectedGameObject(roomCode.gameObject);
            Assert.That((bool)owns.Invoke(null, new object[] { roomCode }), Is.True);
        }

        [Test]
        public void HandledInputEscape_RemainsConsumedAfterTheFieldClearsFocusThatFrame()
        {
            var title = BuildTitle();
            var roomCode = FindField(title, "Room code");
            var eventSystem = CurrentEventSystem();
            eventSystem.SetSelectedGameObject(roomCode.gameObject);

            var mark = typeof(UiKit).GetMethod("MarkInputEscapeHandled",
                BindingFlags.Static | BindingFlags.NonPublic);
            var any = typeof(UiKit).GetMethod("AnyInputOwnsFocus",
                BindingFlags.Static | BindingFlags.NonPublic);
            Assert.That(mark, Is.Not.Null, "text entry must consume Escape before clearing focus");
            Assert.That(any, Is.Not.Null);

            mark.Invoke(null, null);
            eventSystem.SetSelectedGameObject(null);
            Assert.That((bool)any.Invoke(null, null), Is.True,
                "the escape menu must not reopen during the same keypress frame");
        }

        TitleScreen BuildTitle()
        {
            _gameObject = new GameObject("Inline title test", typeof(BoardRenderer), typeof(GameBootstrap));
            var title = _gameObject.AddComponent<TitleScreen>();
            Invoke(title, "Start");
            return title;
        }

        DesignPanel BuildDesigner()
        {
            _gameObject = new GameObject("Inline designer test", typeof(BoardRenderer), typeof(GameBootstrap));
            var designer = _gameObject.AddComponent<DesignPanel>();
            Invoke(designer, "Start");
            return designer;
        }

        static InputField FindField(Component owner, string name)
        {
            var field = Array.Find(owner.GetComponentsInChildren<InputField>(true), item => item.gameObject.name == name);
            Assert.That(field, Is.Not.Null, $"Missing inline field '{name}'");
            return field;
        }

        static EventSystem CurrentEventSystem() =>
            EventSystem.current ?? UnityEngine.Object.FindAnyObjectByType<EventSystem>();

        static T FindPrivate<T>(object owner, string fieldName)
        {
            var field = owner.GetType().GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null, $"Missing field {fieldName}");
            return (T)field.GetValue(owner);
        }

        static T InvokeStatic<T>(Type type, string method, params object[] args)
        {
            var found = type.GetMethod(method, BindingFlags.Static | BindingFlags.NonPublic);
            Assert.That(found, Is.Not.Null, $"Missing method {method}");
            return (T)found.Invoke(null, args);
        }

        static void Invoke(object target, string method, params object[] args) =>
            target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, args);
    }
}
