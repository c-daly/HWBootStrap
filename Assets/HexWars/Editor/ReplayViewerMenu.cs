using System;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;

namespace HexWars.Presentation.EditorTools
{
    public readonly struct MlViewerLaunchResult
    {
        MlViewerLaunchResult(
            bool success, string error, string scenario,
            string seatSchedule, int learnerSeat, string opponent)
        {
            Success = success;
            Error = error ?? string.Empty;
            Scenario = scenario ?? string.Empty;
            SeatSchedule = seatSchedule ?? string.Empty;
            LearnerSeat = learnerSeat;
            Opponent = opponent ?? string.Empty;
        }

        public readonly bool Success;
        public readonly string Error;
        public readonly string Scenario;
        public readonly string SeatSchedule;
        public readonly int LearnerSeat;
        public readonly string Opponent;

        public static MlViewerLaunchResult Succeeded(
            string scenario, int learnerSeat, string opponent,
            string seatSchedule = "unknown") =>
            new MlViewerLaunchResult(
                true, string.Empty, scenario, seatSchedule,
                learnerSeat, opponent);

        public static MlViewerLaunchResult Failed(string error) =>
            new MlViewerLaunchResult(
                false,
                string.IsNullOrWhiteSpace(error)
                    ? "Start & Watch failed."
                    : error,
                string.Empty, string.Empty, -1, string.Empty);
    }

    /// <summary>Editor entry point for the replay viewer: pick a recorded match file (written headless
    /// in WSL2 or by HexWars.Sim) and watch it play back, with speed + scrub controls.</summary>
    public static class ReplayViewerMenu
    {
        [MenuItem("HexWars/Replay/Open Replay File...")]
        public static void OpenReplay()
        {
            string path = EditorUtility.OpenFilePanel("Open HexWars replay", Application.dataPath, "replay,txt");
            if (string.IsNullOrEmpty(path)) return;

            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.02f, 0.05f);
            camGo.AddComponent<CameraRig>();

            var rep = new GameObject("Replay");
            rep.AddComponent<BoardRenderer>();
            rep.AddComponent<ReplayPlayer>().ReplayPath = path;
            // read-only hover + click-to-inspect over the replayed units (no GameBootstrap here, so no
            // commands fire); RequireComponent pulls in the UnitTooltip automatically
            rep.AddComponent<UnitInputController>().ReadOnly = true;

            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<InputSystemUIInputModule>();

            EditorApplication.EnterPlaymode();
        }

        [MenuItem("HexWars/Watch AI vs AI")]
        public static void WatchAiVsAi()
        {
            EditorSceneManager.OpenScene("Assets/Scenes/HexWars.unity", OpenSceneMode.Single);
            EditorPrefs.SetBool("HexWars.Spectate", true); // SpectatorDriver auto-attaches on play
            EditorApplication.EnterPlaymode();
        }

        // Challenge the computer: you play Player 1, the AI plays Player 2. AiOpponent auto-attaches on
        // play from these prefs (the saved scene is untouched). In a build, set GameBootstrap.VsAI instead.
        [MenuItem("HexWars/Play vs AI/Easy (Random)")]
        public static void PlayVsAiEasy() => PlayVsAi(AiLevel.Easy);

        [MenuItem("HexWars/Play vs AI/Hard (Greedy)")]
        public static void PlayVsAiHard() => PlayVsAi(AiLevel.Hard);

        static void PlayVsAi(AiLevel level)
        {
            EditorSceneManager.OpenScene("Assets/Scenes/HexWars.unity", OpenSceneMode.Single);
            EditorPrefs.SetBool("HexWars.VsAI", true);
            EditorPrefs.SetInt("HexWars.AiLevel", (int)level);
            EditorApplication.EnterPlaymode();
        }

        // Watch two metadata-backed runs (or a run vs greedy) fight once via policy_server.py.
        // Cancel either folder picker to use greedy for that seat.
        [MenuItem("HexWars/Watch Model Duel...")]
        public static void WatchModelDuel()
        {
            string pyDir = PyDir();
            if (!PyReady(pyDir)) return;
            string p0 = PickRunSpec("Seat 0 (Player 1) run — Cancel for greedy", pyDir,
                ModelControllerKind.FixedRun, out string run0);
            if (p0 == null) return;
            string p1 = PickRunSpec("Seat 1 (Player 2) run — Cancel for greedy", pyDir,
                ModelControllerKind.FixedRun, out string run1);
            if (p1 == null) return;
            if (!TryResolveDuelEnvironment(run0, run1, out MlEnvironmentContract environment)) return;
            LaunchDuel(pyDir, p0, p1, loop: false, environment: environment);
        }

        // Watch LIVE training: seat 0 = the newest checkpoint in a run's folder, reloaded between games as
        // training writes fresh ones, looping continuously, vs an opponent (Cancel = greedy). You watch the
        // agent visibly improve over the run.
        [MenuItem("HexWars/Watch Live Training...")]
        public static void WatchLiveTraining()
        {
            string pyDir = PyDir();
            if (!PyReady(pyDir)) return;
            string dir = EditorUtility.OpenFolderPanel(
                "Pick the learner's checkpoint folder (runs/<run>/checkpoints)",
                System.IO.Path.Combine(pyDir, "runs"), "");
            if (string.IsNullOrEmpty(dir)) return;
            string learnerRun = NormalizeRunDirectory(dir);
            if (learnerRun == null) return;
            string p1 = PickRunSpec("Opponent run — Cancel for greedy", pyDir,
                ModelControllerKind.FixedRun, out string opponentRun);
            if (p1 == null) return;
            if (!TryResolveDuelEnvironment(learnerRun, opponentRun,
                    out MlEnvironmentContract environment)) return;
            string learner = BuildLiveTrainingSpec(learnerRun);
            LaunchDuel(pyDir, learner, p1, loop: true, environment: environment);
        }

        /// <summary>Open a live run from ML Lab, resolving its algorithm from run metadata.</summary>
        public static MlViewerLaunchResult WatchLiveRun(string runDirectory)
        {
            try
            {
                MlRunPresentationPlan plan = MlRunPresentationPlan.Load(runDirectory);
                MlPresentationGame firstGame = plan.PlanGame(0);
                LaunchDuel(PyDir(), plan);
                return MlViewerLaunchResult.Succeeded(
                    firstGame.Scenario.Id,
                    firstGame.LearnerSeat,
                    firstGame.OpponentLabel,
                    plan.LearnerSeatSchedule);
            }
            catch (Exception error)
            {
                Debug.LogError("HexWars Start & Watch: " + error.Message);
                return MlViewerLaunchResult.Failed(error.Message);
            }
        }

        public static string BuildLiveTrainingSpec(string runDirectory) =>
            new ModelSeatConfiguration
            {
                Kind = ModelControllerKind.LiveRun,
                Path = runDirectory,
                InferenceMode = ModelInferenceMode.Stochastic,
            }.BuildSpec();

        static string PyDir() =>
            System.IO.Path.Combine(System.IO.Directory.GetParent(Application.dataPath).FullName, "python");

        static bool PyReady(string pyDir)
        {
            string pyExe = System.IO.Path.Combine(pyDir, "winenv", "Scripts", "python.exe");
            if (System.IO.File.Exists(pyExe)) return true;
            EditorUtility.DisplayDialog("HexWars", "Windows venv Python not found at:\n" + pyExe, "OK");
            return false;
        }

        public static void LaunchDuel(
            string pyDir, string p0, string p1, bool loop, int seed = 0, float secondsPerAction = 0.4f,
            MlEnvironmentContract environment = MlEnvironmentContract.TacticalV1,
            TrainingScenario scenario = null,
            MlPresentationSchedule presentationPlan = null)
        {
            string pyExe = System.IO.Path.Combine(pyDir, "winenv", "Scripts", "python.exe");
            string server = System.IO.Path.Combine(pyDir, "policy_server.py");

            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.02f, 0.05f);
            camGo.AddComponent<CameraRig>();
            EnsureSingleAudioListener(cam);

            var go = new GameObject("ModelDuel");
            go.AddComponent<BoardRenderer>();
            var d = go.AddComponent<ModelDuelDriver>();
            d.PythonExe = pyExe; d.ServerScript = server; d.WorkingDir = pyDir;
            d.P0Spec = p0; d.P1Spec = p1; d.Seed = seed; d.Loop = loop;
            d.Environment = environment;
            d.Scenario = scenario;
            d.PresentationPlan = presentationPlan;
            d.SecondsPerAction = secondsPerAction;
            go.AddComponent<UnitInputController>().ReadOnly = true; // read-only hover/inspect

            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<InputSystemUIInputModule>();

            EditorApplication.EnterPlaymode();
        }

        /// <summary>Root cause of the "There are no audio listeners in the scene" spam during arena
        /// playback: <see cref="LaunchDuel"/> always starts from a brand-new, empty scene
        /// (<see cref="EditorSceneManager.NewScene"/>), which never carries the one listener a normal
        /// game scene owns — so battle SFX (<see cref="SoundManager"/>) played into it are both
        /// inaudible and log every frame. Adds exactly one <see cref="AudioListener"/>, on the arena's
        /// own camera, and only when the scene doesn't already carry one anywhere — adding a second
        /// listener just flips that same warning into "There are 2 audio listeners" instead of curing
        /// it. Thin: the caller passes the already-created arena camera; this only decides whether to
        /// add the component.</summary>
        public static void EnsureSingleAudioListener(Camera camera)
        {
            if (camera == null) return;
            if (UnityEngine.Object.FindObjectsByType<AudioListener>(FindObjectsSortMode.None).Length > 0) return;
            camera.gameObject.AddComponent<AudioListener>();
        }

        public static void LaunchDuel(string pyDir, MlRunPresentationPlan plan)
        {
            if (plan == null) throw new ArgumentNullException(nameof(plan));
            if (!PyReady(pyDir))
                throw new InvalidOperationException(
                    "Windows venv Python is unavailable for Start & Watch.");
            MlPresentationGame game = plan.PlanGame(0);
            MlEnvironmentContract environment =
                game.Scenario.Environment == "adaptive-v1"
                    ? MlEnvironmentContract.AdaptiveV1
                    : MlEnvironmentContract.TacticalV1;
            LaunchDuel(
                pyDir,
                game.P0Spec,
                game.P1Spec,
                loop: true,
                environment: environment,
                scenario: game.Scenario,
                presentationPlan: plan.BuildRuntimeSchedule());
        }

        static string PickRunSpec(string title, string pyDir, ModelControllerKind kind,
            out string runDirectory)
        {
            string path = EditorUtility.OpenFolderPanel(title,
                System.IO.Path.Combine(pyDir, "runs"), "");
            if (string.IsNullOrEmpty(path))
            {
                runDirectory = null;
                return "greedy";
            }
            runDirectory = NormalizeRunDirectory(path);
            if (runDirectory == null) return null;
            return new ModelSeatConfiguration { Kind = kind, Path = runDirectory }.BuildSpec();
        }

        static string FindRunManifest(string modelPath)
        {
            var directory = System.IO.Directory.Exists(modelPath)
                ? new System.IO.DirectoryInfo(modelPath)
                : System.IO.Directory.GetParent(modelPath);
            for (int depth = 0; directory != null && depth < 2; depth++, directory = directory.Parent)
            {
                string manifest = System.IO.Path.Combine(directory.FullName, "run.json");
                if (System.IO.File.Exists(manifest)) return manifest;
            }
            return null;
        }

        static string NormalizeRunDirectory(string modelPath)
        {
            string manifest = FindRunManifest(modelPath);
            if (manifest == null)
            {
                UnityEngine.Debug.LogError("HexWars: live training requires a metadata-backed run.");
                return null;
            }
            return System.IO.Path.GetDirectoryName(manifest);
        }

        static MlEnvironmentContract EnvironmentFromRun(string runDirectory)
        {
            try
            {
                string json = System.IO.File.ReadAllText(System.IO.Path.Combine(runDirectory, "run.json"));
                return EnvironmentFromRunManifest(json);
            }
            catch (System.Exception) { return MlEnvironmentContract.TacticalV1; }
        }

        public static MlEnvironmentContract EnvironmentFromRunManifest(string json) =>
            MlEnvironmentSummary.FromRunManifest(json).ContractVersion == "adaptive-v1"
                ? MlEnvironmentContract.AdaptiveV1
                : MlEnvironmentContract.TacticalV1;

        static bool TryResolveDuelEnvironment(string run0, string run1,
            out MlEnvironmentContract environment)
        {
            environment = run0 != null ? EnvironmentFromRun(run0)
                : run1 != null ? EnvironmentFromRun(run1)
                : MlEnvironmentContract.TacticalV1;
            if (run0 == null || run1 == null || EnvironmentFromRun(run1) == environment) return true;
            EditorUtility.DisplayDialog("HexWars",
                "The selected runs use different environment contracts.", "OK");
            return false;
        }
    }
}
