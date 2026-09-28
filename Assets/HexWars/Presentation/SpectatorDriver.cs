using System;
using System.Threading;
using HexWars.Engine;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>The title/watch match uses the same pinned configured policy on both seats.</summary>
    public sealed class SpectatorDriver : MonoBehaviour
    {
        public float SecondsPerAction = 0.35f;
        GameBootstrap _game;
        AiPolicySession[] _sessions;
        readonly CancellationTokenSource _lifetime = new CancellationTokenSource();
        float _timer;
        bool _pending, _failed;

        async void Start()
        {
            _game = GetComponent<GameBootstrap>();
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null) input.ReadOnly = true;
            var barracks = FindAnyObjectByType<BarracksPanel>();
            if (barracks != null) barracks.ReadOnly = true;
            try
            {
                await AiModelSettings.LoadAsync(refresh: true);
                if (this == null || _lifetime.IsCancellationRequested) return;
                var selection = AiModelSettings.Capture();
                _sessions = new[] { new AiPolicySession(selection, PlayerId.Player0, 1),
                                    new AiPolicySession(selection, PlayerId.Player1, 2) };
            }
            catch (Exception error) { if (this != null && !_lifetime.IsCancellationRequested) Fail(error); }
        }

        void Update()
        {
            if (_failed || _pending || _sessions == null || _game == null || _game.State == null || _game.State.IsGameOver) return;
            if (_game.Presenter != null && _game.Presenter.IsBusy) { _timer = 0; return; }
            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0;
            RequestAction(_game.State);
        }

        async void RequestAction(GameState state)
        {
            _pending = true;
            try
            {
                Command command = state.PlacingStartingUnits ? new FinishPlacement(state.ActivePlayer)
                    : await _sessions[(int)state.ActivePlayer].DecideAsync(state, _lifetime.Token);
                if (this == null || _lifetime.IsCancellationRequested) return;
                AiPolicySession.RequireCurrentState(state, _game.State);
                if (!_game.TryApply(command)) throw new InvalidOperationException("The demo AI command was rejected.");
            }
            catch (OperationCanceledException) { }
            catch (Exception error) { if (this != null && !_lifetime.IsCancellationRequested) Fail(error); }
            finally { _pending = false; }
        }

        void Fail(Exception error)
        {
            _failed = true;
            Debug.LogError("SpectatorDriver: configured AI unavailable. " + error.Message);
            Toast.Show("AI demo unavailable: " + error.Message);
            if (_sessions != null) foreach (var session in _sessions) session.Dispose();
        }

        void OnDestroy()
        {
            _lifetime.Cancel();
            if (_sessions != null) foreach (var session in _sessions) session.Dispose();
        }
        void OnDisable() => OnDestroy();

#if UNITY_EDITOR
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoAttachInEditor()
        {
            if (!UnityEditor.EditorPrefs.GetBool("HexWars.Spectate", false)) return;
            UnityEditor.EditorPrefs.SetBool("HexWars.Spectate", false);
            var game = FindAnyObjectByType<GameBootstrap>();
            if (game != null) game.StartCoroutine(StartWatch(game));
        }
        static System.Collections.IEnumerator StartWatch(GameBootstrap game)
        {
            yield return null;
            if (game != null) game.StartDemo();
        }
#endif
    }
}
