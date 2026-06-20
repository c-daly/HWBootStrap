using UnityEngine;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// Global Space-bar pause. Toggles <see cref="Time.timeScale"/> between 1 and 0, which freezes
    /// everything driven by scaled time — unit animations, the AI / replay / model-duel steppers, and
    /// coroutines — so it works in <b>any</b> context. Auto-creates itself once per play session (and
    /// survives scene loads), so there's no per-scene wiring. Hovering/inspecting still works while paused.
    /// </summary>
    public sealed class PauseToggle : MonoBehaviour
    {
        bool _paused;
        float _resumeScale = 1f;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            var go = new GameObject("PauseToggle");
            go.AddComponent<PauseToggle>();
            DontDestroyOnLoad(go);
        }

        void Update()
        {
            var kb = Keyboard.current;
            if (kb != null && kb.spaceKey.wasPressedThisFrame)
                SetPaused(!_paused);
        }

        void SetPaused(bool paused)
        {
            if (paused == _paused) return;
            _paused = paused;
            if (_paused)
            {
                _resumeScale = Time.timeScale > 0f ? Time.timeScale : 1f;
                Time.timeScale = 0f;
            }
            else
            {
                Time.timeScale = _resumeScale;
            }
        }

        void OnApplicationQuit() => Time.timeScale = 1f; // don't leave the editor frozen after play

        void OnGUI()
        {
            if (!_paused) return;
            var style = new GUIStyle(GUI.skin.label)
            {
                fontSize = 22,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.UpperCenter,
            };
            style.normal.textColor = Color.white;
            GUI.Label(new Rect(0f, 12f, Screen.width, 40f), "PAUSED  —  Space to resume", style);
        }
    }
}
