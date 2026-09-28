using System;
using System.Collections;
using System.Threading;
using System.Threading.Tasks;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using Object = UnityEngine.Object;

namespace HexWars.Presentation.PlayModeTests
{
    /// <summary>Opt-in host integration: real configured package or hosted endpoint, never a fake model.</summary>
    public sealed class ConfiguredAiIntegrationTests
    {
        static void RequireOptIn()
        {
            if (Environment.GetEnvironmentVariable("HEXWARS_RUN_AI_INTEGRATION") != "1")
                Assert.Ignore("Set HEXWARS_RUN_AI_INTEGRATION=1 with an available configured runtime to run real AI integration.");
        }

        static IEnumerator Await(Task task, float seconds)
        {
            float deadline = Time.realtimeSinceStartup + seconds;
            while (!task.IsCompleted && Time.realtimeSinceStartup < deadline) yield return null;
            Assert.That(task.IsCompleted, Is.True, "Configured AI operation exceeded the bounded integration deadline.");
            task.GetAwaiter().GetResult();
        }

        [UnityTest]
        public IEnumerator ConfiguredSessionMakesAnAcceptedCommandAndCancelsItsNextRequest()
        {
            RequireOptIn();
            yield return Await(AiModelSettings.LoadAsync(refresh: true), 35);
            var selection = AiModelSettings.Capture();
            Assert.That(selection.Model.IsTrained, Is.True, "This integration control requires the configured trained model.");
            GameState state = GameFactory.BuildTacticalV3Compatible(GameSetup.Default);
            using (var session = new AiPolicySession(selection, PlayerId.Player0, 7))
            using (var cancel = new CancellationTokenSource())
            {
                var inference = session.DecideAsync(state, CancellationToken.None);
                yield return Await(inference, 165);
                Command selected = inference.Result;
                Assert.That(GameEngine.Apply(state, selected).Success, Is.True);
                Debug.Log("AI integration accepted: " + selection.Model.id + " checkpoint=" +
                    selection.Model.checkpoint_sha256 + " transport=" + (selection.UsesHttp ? "HTTP" : "local") +
                    " command=" + selected.GetType().Name);
                var pending = session.DecideAsync(state, cancel.Token);
                cancel.Cancel(); session.Dispose();
                float deadline = Time.realtimeSinceStartup + 5;
                while (!pending.IsCompleted && Time.realtimeSinceStartup < deadline) yield return null;
                Assert.That(pending.IsCompleted, Is.True, "Cancellation must release an in-flight policy request promptly.");
                Assert.That(pending.IsCanceled || pending.IsFaulted, Is.True, "A canceled request cannot return an applicable command.");
                if (pending.IsFaulted) _ = pending.Exception; // Observe the aborted transport's failure.
            }
        }

        [UnityTest]
        public IEnumerator TitleDemoAndLocalMatchUseTheConfiguredAiAndMenuCancelsLateCommands()
        {
            RequireOptIn();
            yield return Await(AiModelSettings.LoadAsync(refresh: true), 35);
            Assert.That(AiModelSettings.Capture().Model.IsTrained, Is.True);
            SessionBarracksCache.ResetForTests();
            var host = new GameObject("Configured AI integration", typeof(BoardRenderer), typeof(TokenStore), typeof(GameBootstrap));
            var game = host.GetComponent<GameBootstrap>(); game.enabled = false;
            var presenter = host.AddComponent<ActionPresenter>();
            typeof(GameBootstrap).GetProperty("Presenter").SetValue(game, presenter);
            bool oldMuted = SoundManager.Muted;
            try
            {
                game.StartDemo();
                var demoStart = game.State;
                float deadline = Time.realtimeSinceStartup + 165;
                while (ReferenceEquals(game.State, demoStart) && Time.realtimeSinceStartup < deadline) yield return null;
                Assert.That(game.State, Is.Not.SameAs(demoStart), "The real configured title AI must submit a legal command.");
                Assert.That(game.DemoMode, Is.True);
                game.StartLocalGame(GameSetup.Default, true, difficultyId: AiModelSettings.Capture().DifficultyId);
                yield return null; // Destroy the old spectator before checking the new controller.
                Assert.That(game.GetComponent<SpectatorDriver>(), Is.Null);
                var opponent = game.GetComponent<AiOpponent>();
                Assert.That(opponent.Selection.Model.id, Is.EqualTo(AiModelSettings.Capture().Model.id));
                Assert.That(opponent.Selection.Model.checkpoint_sha256, Is.EqualTo(AiModelSettings.Capture().Model.checkpoint_sha256));
                Assert.That(opponent.Level, Is.EqualTo(AiLevel.Configured));
                Assert.That(game.LastLocalDifficultyId, Is.EqualTo(opponent.Selection.DifficultyId));
                opponent.SecondsPerAction = 0;
                Assert.That(game.TryApply(new EndTurn(PlayerId.Player0)), Is.True);
                presenter.FastForward();
                var waiting = game.State;
                deadline = Time.realtimeSinceStartup + 165;
                while (ReferenceEquals(game.State, waiting) && Time.realtimeSinceStartup < deadline) yield return null;
                Assert.That(game.State, Is.Not.SameAs(waiting), "The configured local opponent must submit a legal command.");
                presenter.FastForward();
                if (game.State.ActivePlayer == PlayerId.Player0)
                {
                    Assert.That(game.TryApply(new EndTurn(PlayerId.Player0)), Is.True);
                    presenter.FastForward();
                }
                // Start an actual request synchronously, then leave before its asynchronous result can apply.
                var request = typeof(AiOpponent).GetMethod("RequestAction", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic);
                if (!opponent.IsThinking) request.Invoke(opponent, new object[] { game.State });
                game.ReturnToMenu();
                foreach (var demo in game.GetComponents<SpectatorDriver>()) demo.enabled = false;
                var menuState = game.State;
                deadline = Time.realtimeSinceStartup + 1;
                while (Time.realtimeSinceStartup < deadline)
                {
                    Assert.That(game.State, Is.SameAs(menuState), "The canceled match must never change the menu/demo state.");
                    yield return null;
                }
                Assert.That(game.GetComponent<AiOpponent>(), Is.Null);
                Assert.That(game.DemoMode, Is.True);
                Debug.Log("AI integration lifecycle passed: title demo, local configured opponent, menu cancellation.");
            }
            finally
            {
                Object.Destroy(host);
                SessionBarracksCache.ResetForTests();
                SoundManager.Muted = oldMuted;
            }
            yield return null;
        }
    }
}
