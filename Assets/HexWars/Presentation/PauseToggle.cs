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
        public const float MinSpeed = 0.25f;
        public const float MaxSpeed = 16f;
        const float FineStepCeiling = 4f; // at/below this, snap to 0.25x steps; above it, whole numbers

        bool _paused;
        float _speed = 1f;

        /// <summary>Pure snap rule for the speed slider, extracted for testability: quarter-steps up to
        /// (and including) <see cref="FineStepCeiling"/>, whole-number steps above it — fine control
        /// matters most near 1x, coarse control is all that's useful once you're already at multiples of
        /// the base speed. Always clamped to <see cref="MinSpeed"/>/<see cref="MaxSpeed"/> first, so an
        /// out-of-range input (e.g. from a slider drag) can never snap outside the slider's own bounds.</summary>
        public static float SnapSpeed(float value)
        {
            value = Mathf.Clamp(value, MinSpeed, MaxSpeed);
            return value <= FineStepCeiling ? Mathf.Round(value * 4f) / 4f : Mathf.Round(value);
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            var go = new GameObject("PlaybackControls");
            go.AddComponent<PauseToggle>();
            DontDestroyOnLoad(go);
        }

        bool _show;     // only shown in playback contexts (replay / spectator / model-duel), not live play
        float _nextCheck;
        GameBootstrap _game; // cached: the title demo also runs a SpectatorDriver, but must show no chrome

        void Update()
        {
            if (Time.unscaledTime >= _nextCheck)
            {
                _nextCheck = Time.unscaledTime + 0.5f;
                if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
                bool demo = _game != null && _game.DemoMode;
                bool show = !demo
                         && (FindAnyObjectByType<ReplayPlayer>() != null
                          || FindAnyObjectByType<SpectatorDriver>() != null
                          || FindAnyObjectByType<ModelDuelDriver>() != null);
                if (_show && !show && _paused) { _paused = false; Apply(); } // left a playback context — don't leave it paused
                _show = show;
            }
            if (!_show) return;

            var kb = Keyboard.current;
            if (kb != null && kb.spaceKey.wasPressedThisFrame) { _paused = !_paused; Apply(); }
        }

        void Apply() => Time.timeScale = _paused ? 0f : _speed;

        void OnApplicationQuit() => Time.timeScale = 1f; // don't leave the editor frozen/scaled after play

        void OnGUI()
        {
            if (!_show) return; // hidden unless we're in a playback context (e.g. a replay)
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
            float v = GUI.HorizontalSlider(new Rect(x, y + 9f, w, 22f), _speed, MinSpeed, MaxSpeed);
            v = SnapSpeed(v);
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
