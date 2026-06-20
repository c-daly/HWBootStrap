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

        // Watch two *trained models* (or a model vs greedy/random) fight, live, via the Windows-venv
        // policy_server.py bridge. Pick a .zip per seat (Cancel = greedy).
        [MenuItem("HexWars/Watch Model Duel...")]
        public static void WatchModelDuel()
        {
            string root = System.IO.Directory.GetParent(Application.dataPath).FullName;
            string pyDir = System.IO.Path.Combine(root, "python");
            string pyExe = System.IO.Path.Combine(pyDir, "winenv", "Scripts", "python.exe");
            string server = System.IO.Path.Combine(pyDir, "policy_server.py");
            if (!System.IO.File.Exists(pyExe))
            {
                EditorUtility.DisplayDialog("Model Duel", "Windows venv Python not found at:\n" + pyExe, "OK");
                return;
            }

            string p0 = PickSpec("Seat 0 (Player 1) model — Cancel for greedy", pyDir);
            string p1 = PickSpec("Seat 1 (Player 2) model — Cancel for greedy", pyDir);

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
            d.P0Spec = p0; d.P1Spec = p1; d.Seed = 0;
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
