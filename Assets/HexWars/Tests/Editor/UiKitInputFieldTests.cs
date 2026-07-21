using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation.Tests
{
    public class UiKitInputFieldTests
    {
        GameObject _canvas;

        [SetUp]
        public void SetUp()
        {
            _canvas = new GameObject("Test Canvas", typeof(Canvas));
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_canvas);
            foreach (var eventSystem in Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None))
                Object.DestroyImmediate(eventSystem.gameObject);
        }

        [Test]
        public void InputField_CreatesCompleteLegacyGraphWithPrefilledText()
        {
            var field = UiKit.InputField(_canvas.transform, "Room 42", 0f, -20f, 240f, 42f, "Room code");

            var image = field.GetComponent<Image>();
            var text = field.transform.Find("Text").GetComponent<Text>();
            var placeholder = field.transform.Find("Placeholder").GetComponent<Text>();

            Assert.That(image, Is.Not.Null);
            Assert.That(field.targetGraphic, Is.SameAs(image));
            Assert.That(field.textComponent, Is.SameAs(text));
            Assert.That(field.placeholder, Is.SameAs(placeholder));
            Assert.That(field.text, Is.EqualTo("Room 42"));
            Assert.That(placeholder.text, Is.EqualTo("Room code"));
            Assert.That(Object.FindAnyObjectByType<EventSystem>(), Is.Not.Null);
        }

        [Test]
        public void IntField_UsesIntegerInputAndCommitsOnlyValidValues()
        {
            int committed = -1;
            var binding = UiKit.IntField(_canvas.transform, 12, 0f, -20f, 160f, 42f,
                min: 5, max: 64, blankMeansZero: false, setter: value => committed = value);

            Assert.That(binding.Field.contentType, Is.EqualTo(InputField.ContentType.IntegerNumber));
            Assert.That(binding.Field.text, Is.EqualTo("12"));

            binding.Field.text = "65";
            Assert.That(binding.Commit(), Is.False);
            Assert.That(binding.Field.text, Is.EqualTo("65"));
            Assert.That(binding.Error.text, Is.EqualTo("Use 5\u201364"));
            Assert.That(committed, Is.EqualTo(-1));

            binding.Field.text = "64";
            Assert.That(binding.Commit(), Is.True);
            Assert.That(committed, Is.EqualTo(64));
            Assert.That(binding.Error.text, Is.Empty);
        }

        [Test]
        public void IntField_SetCommittedValueUpdatesDisplayAndEscapeRestoreSuppressesEndEdit()
        {
            int setterCalls = 0;
            var binding = UiKit.IntField(_canvas.transform, 12, 0f, -20f, 160f, 42f,
                min: 0, max: 64, blankMeansZero: true, setter: _ => setterCalls++);

            binding.SetCommittedValue(24);
            Assert.That(binding.Field.text, Is.EqualTo("24"));

            binding.Field.text = "99";
            binding.Restore(); // screen-level Escape handling calls this before focus changes
            Assert.That(binding.Field.text, Is.EqualTo("24"));

            binding.Field.onEndEdit.Invoke(binding.Field.text);
            Assert.That(setterCalls, Is.EqualTo(0));
        }
    }
}
