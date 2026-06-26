using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using HexWars.Presentation;

namespace HexWars.Presentation.EditorTools
{
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

        // Watch two *trained models* (or a model vs greedy/random) fight once, via the Windows-venv
        // policy_server.py bridge. Pick a .zip per seat (Cancel = greedy).
        [MenuItem("HexWars/Watch Model Duel...")]
        public static void WatchModelDuel()
        {
            string pyDir = PyDir();
            if (!PyReady(pyDir)) return;
            string p0 = PickSpec("Seat 0 (Player 1) model — Cancel for greedy", pyDir);
            string p1 = PickSpec("Seat 1 (Player 2) model — Cancel for greedy", pyDir);
            LaunchDuel(pyDir, p0, p1, loop: false);
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
            string p1 = PickSpec("Opponent model — Cancel for greedy", pyDir);
            LaunchDuel(pyDir, "ppo:" + dir, p1, loop: true);
        }

        static string PyDir() =>
            System.IO.Path.Combine(System.IO.Directory.GetParent(Application.dataPath).FullName, "python");

        static bool PyReady(string pyDir)
        {
            string pyExe = System.IO.Path.Combine(pyDir, "winenv", "Scripts", "python.exe");
            if (System.IO.File.Exists(pyExe)) return true;
            EditorUtility.DisplayDialog("HexWars", "Windows venv Python not found at:\n" + pyExe, "OK");
            return false;
        }

        static void LaunchDuel(string pyDir, string p0, string p1, bool loop)
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

            var go = new GameObject("ModelDuel");
            go.AddComponent<BoardRenderer>();
            var d = go.AddComponent<ModelDuelDriver>();
            d.PythonExe = pyExe; d.ServerScript = server; d.WorkingDir = pyDir;
            d.P0Spec = p0; d.P1Spec = p1; d.Seed = 0; d.Loop = loop;
            go.AddComponent<UnitInputController>().ReadOnly = true; // read-only hover/inspect

            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<InputSystemUIInputModule>();

            EditorApplication.EnterPlaymode();
        }

        static string PickSpec(string title, string pyDir)
        {
            string path = EditorUtility.OpenFilePanel(title, pyDir, "zip");
            if (string.IsNullOrEmpty(path)) return "greedy";
            string prefix = System.IO.Path.GetFileNameWithoutExtension(path).ToLowerInvariant().Contains("dqn") ? "dqn:" : "ppo:";
            return prefix + path;
        }
    }
}
