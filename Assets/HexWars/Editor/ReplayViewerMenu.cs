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
    }
}
