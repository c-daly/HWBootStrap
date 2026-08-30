using System;
using System.Collections;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using HexWars.Engine;
using HexWars.Engine.Rl;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Normal-game opponent choices.  Easy remains as the WebGL-safe legacy value, Hard is Greedy,
    /// and TrainedModel uses the selected fixed tactical-v3 package through the developer Python
    /// bridge.  The in-game desktop menu presents Greedy/TrainedModel; WebGL retains Random/Greedy
    /// until an in-process exported policy exists.
    /// </summary>
    public enum AiLevel { Easy = 0, Hard = 1, TrainedModel = 2 }

    /// <summary>
    /// A single AI-controlled seat in the playable game, so a human can challenge the computer. On the AI
    /// seat's turn it steps a scripted agent through the normal <see cref="GameBootstrap.TryApply"/> path
    /// (board, HP bars, turn banner, event sidebar all update as usual); on the human's turn it hands
    /// control back. Human input is gated to the human's turn by toggling the input/barracks ReadOnly flags.
    /// </summary>
    public sealed class AiOpponent : MonoBehaviour
    {
        public PlayerId AiSeat = PlayerId.Player1; // the human plays the other seat
        public AiLevel Level = AiLevel.Hard;
        public float SecondsPerAction = 0.35f;

        GameBootstrap _game;
        IAgent _agent;
        UnitInputController _input;
        BarracksPanel _barracks;
        float _timer;
        PlayableModelAdapter _model;
        PolicyBridge _bridge;
        CancellationTokenSource _startupCancellation;
        bool _startupRequested;
        bool _modelReady;
        bool _modelFailed;
        long _decisionId;

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _agent = Level == AiLevel.Easy
                ? (IAgent)new RandomAgent(7)
                : new GreedyAgent(7);
            _input = FindAnyObjectByType<UnitInputController>();
            _barracks = FindAnyObjectByType<BarracksPanel>();
        }

        void Update()
        {
            if (_game == null || _game.State == null) return;
            var s = _game.State;
            if (Level == AiLevel.TrainedModel && !_startupRequested)
                BeginModelStartup(s);
            bool aiTurn = !s.IsGameOver && s.ActivePlayer == AiSeat;

            // the human can only issue commands on their own turn
            if (_input != null) _input.ReadOnly = aiTurn;
            if (_barracks != null) _barracks.ReadOnly = aiTurn;
            if (!aiTurn) return;
            if (_game.Presenter != null && _game.Presenter.IsBusy) { _timer = 0f; return; } // let the last action finish playing

            _timer += Time.deltaTime;
            if (_timer < SecondsPerAction) return;
            _timer = 0f;
            if (Level != AiLevel.TrainedModel)
            {
                _game.TryApply(_agent.Decide(s));
                return;
            }
            if (_modelFailed) return;
            if (!_modelReady) return;
            ApplyModelAction(s);
        }

        async void BeginModelStartup(GameState state)
        {
            _startupRequested = true;
#if UNITY_WEBGL && !UNITY_EDITOR
            FailModelMatch(
                "The trained opponent needs the desktop policy runtime and is unavailable in WebGL.");
            await Task.CompletedTask;
#else
            PolicyBridge bridge = null;
            try
            {
                _model = new PlayableModelAdapter(state, AiSeat);
                DirectoryInfo project = Directory.GetParent(Application.dataPath);
                if (project == null)
                    throw new InvalidOperationException("Unity project root could not be resolved");
                PlayableModelLaunch launch = PlayableModelResolver.Resolve(project.FullName);

                _startupCancellation = new CancellationTokenSource();
                bridge = new PolicyBridge();
                _bridge = bridge;
                string p0 = AiSeat == PlayerId.Player0 ? launch.ControllerSpec : null;
                string p1 = AiSeat == PlayerId.Player1 ? launch.ControllerSpec : null;
                bool ready = await bridge.StartAsync(
                    launch.PythonExecutable,
                    launch.ServerScript,
                    p0,
                    p1,
                    launch.WorkingDirectory,
                    _model.ContractIdentity.Environment,
                    _model.ContractIdentity.Version,
                    _model.ContractIdentity.EncodingHash,
                    _model.ContractIdentity.CapacityHash,
                    PolicyBridge.DefaultStartupTimeoutMs,
                    _startupCancellation.Token);

                if (this == null || !ReferenceEquals(_bridge, bridge))
                {
                    bridge.Dispose();
                    return;
                }
                if (!ready)
                {
                    FailModelMatch("The trained model could not be loaded.");
                    return;
                }

                var compatibility = ModelDuelContractCompatibility.Validate(
                    _model.ContractIdentity,
                    AiSeat == PlayerId.Player0,
                    bridge.Seat0,
                    AiSeat == PlayerId.Player1,
                    bridge.Seat1);
                if (compatibility.Count != 0)
                {
                    FailModelMatch(string.Join(" ", compatibility));
                    return;
                }

                _modelReady = true;
                Debug.Log("AiOpponent: trained model ready: " + launch.ModelName);
            }
            catch (OperationCanceledException)
            {
                bridge?.Dispose();
            }
            catch (Exception error)
            {
                if (this != null)
                    FailModelMatch(error.Message);
                else
                    bridge?.Dispose();
            }
#endif
        }

        void ApplyModelAction(GameState state)
        {
            try
            {
                TacticalV3DecisionFrame frame = _model.CreateFrame(
                    state, AiSeat, _decisionId++);
                PolicyCandidateResult selected = _bridge.ActStructured(
                    (int)AiSeat, TacticalV3PolicyPayload.From(frame));
                Command command = _model.Resolve(frame, selected, _game.State);
                if (!_game.TryApply(command))
                    throw new InvalidOperationException(
                        "the trained model's selected legal command was rejected");
            }
            catch (Exception error)
            {
                FailModelMatch(error.Message);
            }
        }

        void FailModelMatch(string reason)
        {
            if (_modelFailed) return;
            _modelFailed = true;
            _modelReady = false;
            _bridge?.Dispose();
            _bridge = null;
            string detail = string.IsNullOrWhiteSpace(reason)
                ? "unknown model error"
                : reason;
            Debug.LogError(
                "AiOpponent: trained-model match stopped. " + detail);
            Toast.Show("Trained model unavailable — returning to the menu.");
            _game?.ReturnToMenu();
        }

        void OnDestroy()
        {
            _startupCancellation?.Cancel();
            _startupCancellation?.Dispose();
            _startupCancellation = null;
            _bridge?.Dispose();
            _bridge = null;
        }

#if UNITY_EDITOR
        // The "HexWars > Play vs AI" menu sets these prefs, then this attaches on play — so the saved scene
        // is never modified (mirrors SpectatorDriver). In a build, use GameBootstrap's VsAI fields instead.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoAttachInEditor()
        {
            if (!UnityEditor.EditorPrefs.GetBool("HexWars.VsAI", false)) return;
            UnityEditor.EditorPrefs.SetBool("HexWars.VsAI", false);

            var game = FindAnyObjectByType<GameBootstrap>();
            if (game == null) return;
            var level = (AiLevel)UnityEditor.EditorPrefs.GetInt(
                "HexWars.AiLevel", (int)AiLevel.Hard);
            if (level == AiLevel.TrainedModel)
            {
                game.StartCoroutine(StartDefaultModelGame(game));
                return;
            }
            var ai = game.GetComponent<AiOpponent>() ?? game.gameObject.AddComponent<AiOpponent>();
            ai.Level = level;
        }

        static IEnumerator StartDefaultModelGame(GameBootstrap game)
        {
            // RuntimeInitializeOnLoad runs before GameBootstrap.Start.  Wait one frame so its scene
            // setup exists, then enter through the same model-safe local-game path as the title UI.
            yield return null;
            if (game != null)
                game.StartLocalGame(GameSetup.Default, true, AiLevel.TrainedModel);
        }
#endif
    }
}
