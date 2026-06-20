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
            // OnGUI doesn't DPI-scale — scale by screen height (≈2x at 4K), draw in 1080p-logical coords
            float s = Mathf.Max(1f, Screen.height / 1080f);
            var prevMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(s, s, 1f));
            float W = Screen.width / s;

            // speed slider (top-right)
            const float w = 190f;
            float x = W - w - 90f, y = 16f;
            var slabel = new GUIStyle(GUI.skin.label) { fontSize = 18, alignment = TextAnchor.MiddleLeft };
            slabel.normal.textColor = Color.white;
            GUI.Label(new Rect(x - 70f, y - 2f, 70f, 32f), "Speed", slabel);
            float v = GUI.HorizontalSlider(new Rect(x, y + 9f, w, 22f), _speed, 0.25f, 4f);
            v = Mathf.Round(v * 4f) / 4f; // snap to 0.25x steps
            if (!Mathf.Approximately(v, _speed)) { _speed = v; Apply(); }
            GUI.Label(new Rect(x + w + 8f, y - 2f, 70f, 32f), $"{_speed:0.00}x", slabel);

            if (_paused)
            {
                var p = new GUIStyle(GUI.skin.label) { fontSize = 30, fontStyle = FontStyle.Bold, alignment = TextAnchor.UpperCenter };
                p.normal.textColor = Color.white;
                GUI.Label(new Rect(0f, 52f, W, 50f), "PAUSED  —  Space to resume", p);
            }

            GUI.matrix = prevMatrix;
        }
    }
}
