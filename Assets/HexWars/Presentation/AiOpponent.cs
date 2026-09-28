using System;
using System.Collections;
using System.Threading;
using HexWars.Engine;
using UnityEngine;

namespace HexWars.Presentation
{
    // Existing serialized explicit choices retain their meanings; Configured follows the shared catalog.
    public enum AiLevel { Easy = 0, Hard = 1, TrainedModel = 2, Configured = 3 }

    public sealed class AiOpponent : MonoBehaviour
    {
        public PlayerId AiSeat = PlayerId.Player1;
        public AiLevel Level = AiLevel.Configured;
        public float SecondsPerAction = 0.35f;
        public AiModelSelection Selection { get; set; }
        public string ModelLabel => _session?.ModelLabel ?? Selection?.DisplayName ?? "Loading AI…";
        public bool IsThinking => _pending;
        public bool UsesTrainedModel => _session?.IsTrained ?? Selection?.Model.IsTrained ?? AiModelSettings.UsesTrainedModel(Level);

        GameBootstrap _game;
        UnitInputController _input;
        BarracksPanel _barracks;
        AiPolicySession _session;
        readonly CancellationTokenSource _lifetime = new CancellationTokenSource();
        float _timer;
        bool _pending, _failed;

        async void Start()
        {
            _game = GetComponent<GameBootstrap>();
            _input = FindAnyObjectByType<UnitInputController>();
            _barracks = FindAnyObjectByType<BarracksPanel>();
            try
            {
                await AiModelSettings.LoadAsync();
                if (this == null || _lifetime.IsCancellationRequested) return;
                Selection = Selection ?? AiModelSettings.ForLevel(Level);
                _session = new AiPolicySession(Selection, AiSeat, 7);
                Debug.Log("AiOpponent: selected " + Selection.Model.id + " (" + Selection.Model.checkpoint_sha256 + ")");
            }
            catch (Exception error) { if (this != null && !_lifetime.IsCancellationRequested) FailModelMatch(error.Message); }
        }

        void Update()
        {
            if (_game == null || _game.State == null || _failed) return;
            GameState state = _game.State;
            bool aiTurn = !state.IsGameOver && state.ActivePlayer == AiSeat;
            if (_input != null) _input.ReadOnly = aiTurn;
            if (_barracks != null) _barracks.ReadOnly = aiTurn;
            if (!aiTurn || _pending || _session == null) return;
            if (_game.Presenter != null && _game.Presenter.IsBusy) { _timer = 0; return; }
            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0;
            if (state.PlacingStartingUnits) { _game.TryApply(new FinishPlacement(AiSeat)); return; }
            RequestAction(state);
        }

        async void RequestAction(GameState state)
        {
            _pending = true;
            try
            {
                Command command = await _session.DecideAsync(state, _lifetime.Token);
                if (this == null || _lifetime.IsCancellationRequested) return;
                AiPolicySession.RequireCurrentState(state, _game.State);
                if (!_game.TryApply(command)) throw new InvalidOperationException("The AI's selected legal command was rejected.");
            }
            catch (OperationCanceledException) { }
            catch (Exception error) { if (this != null && !_lifetime.IsCancellationRequested) FailModelMatch(error.Message); }
            finally { _pending = false; }
        }

        void FailModelMatch(string reason)
        {
            if (_failed) return;
            _failed = true;
            _lifetime.Cancel();
            _session?.Dispose();
            Debug.LogError("AiOpponent: configured AI stopped. " + reason);
            Toast.Show("AI unavailable: " + reason);
            _game?.ReturnToMenu();
        }

        void OnDestroy()
        {
            _lifetime.Cancel();
            _session?.Dispose();
            // The pending continuation may still inspect the cancellation source after destruction.
        }
        void OnDisable() => OnDestroy();

#if UNITY_EDITOR
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoAttachInEditor()
        {
            if (!UnityEditor.EditorPrefs.GetBool("HexWars.VsAI", false)) return;
            UnityEditor.EditorPrefs.SetBool("HexWars.VsAI", false);
            var game = FindAnyObjectByType<GameBootstrap>();
            if (game == null) return;
            var level = (AiLevel)UnityEditor.EditorPrefs.GetInt("HexWars.AiLevel", (int)AiLevel.Configured);
            game.StartCoroutine(StartDefaultGame(game, level));
        }

        static IEnumerator StartDefaultGame(GameBootstrap game, AiLevel level)
        {
            yield return null;
            var loading = AiModelSettings.LoadAsync(refresh: true);
            while (!loading.IsCompleted) yield return null;
            if (game == null) yield break;
            if (loading.IsFaulted) { Toast.Show(AiModelSettings.Error); yield break; }
            game.StartLocalGame(GameSetup.Default, true, level);
        }
#endif
    }
}
