using UnityEngine;
using UnityEngine.InputSystem;

namespace HexWars.Presentation
{
    /// <summary>
    /// Global playback controls for any context: <b>Space-bar pause</b> and a <b>speed slider</b>, both via
    /// <see cref="Time.timeScale"/> — so they scale every stepper (AI spectator, model duel, replay) and
    /// animations alike. Auto-created once per play session (no per-scene wiring), drawn with OnGUI so it
    /// keeps working while paused. Hovering/inspecting still works throughout.
    /// </summary>
    public sealed class PauseToggle : MonoBehaviour
    {
        bool _paused;
        float _speed = 1f;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            var go = new GameObject("PlaybackControls");
            go.AddComponent<PauseToggle>();
            DontDestroyOnLoad(go);
        }

        void Update()
        {
            var kb = Keyboard.current;
            if (kb != null && kb.spaceKey.wasPressedThisFrame) { _paused = !_paused; Apply(); }
        }

        void Apply() => Time.timeScale = _paused ? 0f : _speed;

        void OnApplicationQuit() => Time.timeScale = 1f; // don't leave the editor frozen/scaled after play

        void OnGUI()
        {
            // speed slider (top-right)
            const float w = 150f;
            float x = Screen.width - w - 64f, y = 12f;
            float v = GUI.HorizontalSlider(new Rect(x, y + 5f, w, 18f), _speed, 0.25f, 4f);
            v = Mathf.Round(v * 4f) / 4f; // snap to 0.25x steps
            if (!Mathf.Approximately(v, _speed)) { _speed = v; Apply(); }

            var label = new GUIStyle(GUI.skin.label) { alignment = TextAnchor.MiddleLeft };
            label.normal.textColor = Color.white;
            GUI.Label(new Rect(x + w + 6f, y, 56f, 24f), $"{_speed:0.00}x", label);

            if (_paused)
            {
                var p = new GUIStyle(GUI.skin.label) { fontSize = 22, fontStyle = FontStyle.Bold, alignment = TextAnchor.UpperCenter };
                p.normal.textColor = Color.white;
                GUI.Label(new Rect(0f, 40f, Screen.width, 40f), "PAUSED  —  Space to resume", p);
            }
        }
    }
}
