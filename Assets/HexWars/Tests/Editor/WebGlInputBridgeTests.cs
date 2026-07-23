using NUnit.Framework;
using System.Reflection;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation.Tests
{
    public sealed class WebGlInputBridgeTests
    {
        GameObject _canvas;

        [SetUp]
        public void SetUp()
        {
            _canvas = new GameObject("Web input test canvas", typeof(Canvas));
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_canvas);
            foreach (var hub in Object.FindObjectsByType<WebGlInputHub>(FindObjectsSortMode.None))
                if (hub != null) Object.DestroyImmediate(hub.gameObject);
            foreach (var eventSystem in Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None))
                if (eventSystem != null) Object.DestroyImmediate(eventSystem.gameObject);
        }

        [Test]
        public void UiKitInputField_AttachesNativeBrowserBridge()
        {
            var field = UiKit.InputField(
                _canvas.transform, "Scout t", 0f, 0f, 240f, 42f, "Unit name");

            var bridge = field.GetComponent<WebGlInputBridge>();

            Assert.That(bridge, Is.Not.Null);
            Assert.That(bridge.Field, Is.SameAs(field));
        }

        [Test]
        public void HubRoutesNativeChangesAndSubmitToOnlyTheSelectedField()
        {
            var first = UiKit.InputField(_canvas.transform, "First", 0f, 0f, 240f, 42f);
            var second = UiKit.InputField(_canvas.transform, "Second", 0f, -50f, 240f, 42f);
            var firstBridge = first.GetComponent<WebGlInputBridge>();
            var secondBridge = second.GetComponent<WebGlInputBridge>();
            int submitted = 0;
            string submittedValue = null;
            first.onSubmit.AddListener(value =>
            {
                submitted++;
                submittedValue = value;
            });

            firstBridge.OnSelect(null);
            var hub = Object.FindAnyObjectByType<WebGlInputHub>();
            Assert.That(hub, Is.Not.Null);

            hub.OnNativeInput(
                "{\"id\":" + BridgeId(firstBridge) + ",\"kind\":\"change\",\"value\":\"First team\"}");
            hub.OnNativeInput(
                "{\"id\":" + BridgeId(firstBridge) + ",\"kind\":\"submit\",\"value\":\"Final t\"}");
            secondBridge.OnSelect(null);
            hub.OnNativeInput(
                "{\"id\":" + BridgeId(firstBridge) + ",\"kind\":\"change\",\"value\":\"Stale update\"}");
            hub.OnNativeInput(
                "{\"id\":" + BridgeId(secondBridge) + ",\"kind\":\"change\",\"value\":\"Second team\"}");

            Assert.That(first.text, Is.EqualTo("Final t"));
            Assert.That(second.text, Is.EqualTo("Second team"));
            Assert.That(submitted, Is.EqualTo(1));
            Assert.That(submittedValue, Is.EqualTo("Final t"));
        }

        [Test]
        public void NativeCancelRestoresTextFromWhenEditingBegan()
        {
            var field = UiKit.InputField(_canvas.transform, "Original", 0f, 0f, 240f, 42f);
            var bridge = field.GetComponent<WebGlInputBridge>();
            bridge.OnSelect(null);
            var hub = Object.FindAnyObjectByType<WebGlInputHub>();

            hub.OnNativeInput(
                "{\"id\":" + BridgeId(bridge) + ",\"kind\":\"change\",\"value\":\"Discard me\"}");
            hub.OnNativeInput(
                "{\"id\":" + BridgeId(bridge) + ",\"kind\":\"cancel\",\"value\":\"Original\"}");

            Assert.That(field.text, Is.EqualTo("Original"));
        }

        [Test]
        public void NativeChangesHonorIntegerValidationAndCharacterLimit()
        {
            var field = UiKit.InputField(_canvas.transform, "", 0f, 0f, 240f, 42f);
            field.contentType = InputField.ContentType.IntegerNumber;
            field.characterLimit = 3;
            var bridge = field.GetComponent<WebGlInputBridge>();
            bridge.OnSelect(null);
            var hub = Object.FindAnyObjectByType<WebGlInputHub>();

            hub.OnNativeInput(
                "{\"id\":" + BridgeId(bridge) + ",\"kind\":\"change\",\"value\":\"6t423\"}");

            Assert.That(field.text, Is.EqualTo("642"));
        }

        static int BridgeId(WebGlInputBridge bridge)
        {
            var field = typeof(WebGlInputBridge).GetField(
                "_bridgeId", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null);
            return (int)field.GetValue(bridge);
        }
    }
}
