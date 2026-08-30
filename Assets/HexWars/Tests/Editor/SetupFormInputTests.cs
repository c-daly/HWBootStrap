using System.Collections;
using System.Linq;
using System.Reflection;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace HexWars.Presentation.Tests
{
    public sealed class SetupFormInputTests
    {
        GameObject _gameObject;
        SetupForm _form;

        [SetUp]
        public void SetUp()
        {
            _gameObject = new GameObject("Setup Test", typeof(BoardRenderer), typeof(GameBootstrap));
            _form = SetupForm.Open(_gameObject.GetComponent<GameBootstrap>(), SetupForm.SetupMode.VsAi);
            Invoke(_form, "Start");
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_gameObject);
            foreach (var canvas in Object.FindObjectsByType<Canvas>(FindObjectsSortMode.None))
                if (canvas != null) Object.DestroyImmediate(canvas.gameObject);
            foreach (var eventSystem in Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None))
                if (eventSystem != null) Object.DestroyImmediate(eventSystem.gameObject);
        }

        [Test]
        public void Build_CreatesPrefilledNumericInputsWithoutSteppersOrPacePreset()
        {
            var fields = _form.GetComponentsInChildren<InputField>(true)
                .ToDictionary(field => field.gameObject.name, field => field.text);

            Assert.That(fields["Map width"], Is.EqualTo("9"));
            Assert.That(fields["Map height"], Is.EqualTo("7"));
            Assert.That(fields["Start points"], Is.EqualTo("0"));
            Assert.That(fields["Seed"], Is.Not.Empty);
            Assert.That(fields["Army size"], Is.EqualTo("3"));
            Assert.That(fields["Brutes"], Is.EqualTo("1"));
            Assert.That(fields["Strikers"], Is.EqualTo("1"));
            Assert.That(fields["Snipers"], Is.EqualTo("1"));
            Assert.That(fields["Units acting per turn"], Is.EqualTo("3"));

            var buttonLabels = _form.GetComponentsInChildren<Button>(true)
                .Select(button => button.GetComponentInChildren<Text>()?.text ?? string.Empty)
                .ToArray();
            Assert.That(buttonLabels, Does.Not.Contain("+"));
            Assert.That(buttonLabels, Does.Not.Contain("−"));
            Assert.That(buttonLabels.Any(text => text.StartsWith("Pace:")), Is.False);
        }

        [Test]
        public void DifficultyChoiceOffersGreedyAndTrainedModelInsteadOfRandom()
        {
            Button choice = _form.GetComponentsInChildren<Button>(true)
                .Single(button =>
                    (button.GetComponentInChildren<Text>()?.text ?? string.Empty)
                    .StartsWith("AI: "));
            Assert.That(choice.GetComponentInChildren<Text>().text,
                Is.EqualTo("AI: Greedy"));

            choice.onClick.Invoke();

            Assert.That(choice.GetComponentInChildren<Text>().text,
                Is.EqualTo("AI: Trained model"));
        }

        [Test]
        public void WidthRejectsBlankAndTurnActionsAcceptsBlankAsUnlimited()
        {
            var width = Binding("Map width");
            width.Field.text = string.Empty;
            Assert.That(width.Commit(), Is.False);
            Assert.That(width.Error.text, Is.EqualTo("Required"));

            var actions = Binding("Units acting per turn");
            actions.Field.text = string.Empty;
            Assert.That(actions.Commit(), Is.True);
            Assert.That(actions.Field.text, Is.EqualTo("0"));
        }

        [Test]
        public void CreateIsBlockedWhileAnyFieldIsInvalid()
        {
            Binding("Map width").Field.text = string.Empty;

            Invoke(_form, "OnCreate");

            Assert.That(_gameObject.GetComponent<GameBootstrap>().State, Is.Null);
            Assert.That(Binding("Map width").Error.text, Is.EqualTo("Required"));
        }

        InlineIntBinding Binding(string fieldName)
        {
            var field = typeof(SetupForm).GetField("_bindings", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null);
            foreach (var item in (IEnumerable)field.GetValue(_form))
            {
                var binding = (InlineIntBinding)item;
                if (binding.Field.gameObject.name == fieldName) return binding;
            }
            Assert.Fail("Missing binding " + fieldName);
            return null;
        }

        static void Invoke(object target, string method) =>
            target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, null);
    }
}
