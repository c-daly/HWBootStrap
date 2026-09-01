using System.Collections;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.TestTools;
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

    public sealed class HotseatMenuTests
    {
        GameObject _gameObject;

        [TearDown]
        public void TearDown()
        {
            SessionBarracksCache.ResetForTests();
            if (_gameObject != null) Object.DestroyImmediate(_gameObject);
            foreach (var canvas in Object.FindObjectsByType<Canvas>(FindObjectsSortMode.None))
                if (canvas != null) Object.DestroyImmediate(canvas.gameObject);
            foreach (var eventSystem in Object.FindObjectsByType<EventSystem>(FindObjectsSortMode.None))
                if (eventSystem != null) Object.DestroyImmediate(eventSystem.gameObject);
        }

        [Test]
        public void TitleScreen_HotseatButtonOpensHotseatSetup()
        {
            var game = BuildGame();
            var title = _gameObject.AddComponent<TitleScreen>();
            Invoke(title, "Start");
            var hotseat = title.GetComponentsInChildren<Button>(true)
                .Single(button => button.GetComponentInChildren<Text>()?.text == "Hotseat");

            LogAssert.Expect(LogType.Error,
                new Regex("TitleCanvas: Destroy may not be called from edit mode!"));
            LogAssert.Expect(LogType.Error,
                new Regex("Hotseat menu test: Destroy may not be called from edit mode!"));
            hotseat.onClick.Invoke();

            var form = _gameObject.GetComponent<SetupForm>();
            Assert.That(form, Is.Not.Null);
            Assert.That(Mode(form), Is.EqualTo(SetupForm.SetupMode.Hotseat));
        }

        [Test]
        public void HotseatSetup_StartsTwoHumanLocalGameWithoutAiChoice()
        {
            var game = BuildGame();
            SessionBarracksCache.ForLocalPlayer(0).RemoveAt(0);
            SessionBarracksCache.ForLocalPlayer(1).RemoveAt(1);
            var form = SetupForm.Open(game, SetupForm.SetupMode.Hotseat);
            Invoke(form, "Start");
            var labels = form.GetComponentsInChildren<Text>(true)
                .Select(label => label.text)
                .ToArray();

            Assert.That(labels, Does.Contain("Hotseat Game"));
            Assert.That(labels, Does.Not.Contain("Fog of war"));
            Assert.That(labels, Does.Contain("Fog of war is off for shared-screen play"));
            Assert.That(labels.Any(text => text.StartsWith("AI: ")), Is.False);
            Assert.That(labels.Any(text => text.StartsWith("Private")), Is.False);

            Invoke(form, "OnCreate");

            Assert.That(game.State, Is.Not.Null);
            Assert.That(game.Networked, Is.False);
            Assert.That(_gameObject.GetComponent<AiOpponent>(), Is.Null);
            Assert.That(game.RematchAvailable, Is.False);
            Assert.That(game.State.ActivePlayer, Is.EqualTo(PlayerId.Player0));
            Assert.That(game.WaitingHumanSeat(), Is.Null);
            Assert.That(game.State.Config.FogOfWar, Is.False);
            Assert.That(game.State.Player(PlayerId.Player0).Barracks.Select(item => item.Name),
                Does.Not.Contain("Brute"));
            Assert.That(game.State.Player(PlayerId.Player0).Barracks.Select(item => item.Name),
                Does.Contain("Striker"));
            Assert.That(game.State.Player(PlayerId.Player1).Barracks.Select(item => item.Name),
                Does.Contain("Brute"));
            Assert.That(game.State.Player(PlayerId.Player1).Barracks.Select(item => item.Name),
                Does.Not.Contain("Striker"));
        }

        GameBootstrap BuildGame()
        {
            SessionBarracksCache.ResetForTests();
            _gameObject = new GameObject(
                "Hotseat menu test", typeof(BoardRenderer), typeof(GameBootstrap));
            var store = _gameObject.AddComponent<TokenStore>();
            Invoke(store, "Awake");
            return _gameObject.GetComponent<GameBootstrap>();
        }

        static SetupForm.SetupMode Mode(SetupForm form)
        {
            var field = typeof(SetupForm).GetField(
                "_mode", BindingFlags.Instance | BindingFlags.NonPublic);
            Assert.That(field, Is.Not.Null);
            return (SetupForm.SetupMode)field.GetValue(form);
        }

        static void Invoke(object target, string method) =>
            target.GetType().GetMethod(method, BindingFlags.Instance | BindingFlags.NonPublic)
                .Invoke(target, null);
    }
}
