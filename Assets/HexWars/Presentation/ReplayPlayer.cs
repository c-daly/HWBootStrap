using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Plays back a recorded match (a <see cref="ReplayFile"/> written headless in WSL2 or anywhere)
    /// by driving the <see cref="BoardRenderer"/> through every reconstructed frame. Controls: play /
    /// pause, a speed slider, and a scrubber to jump to any frame — i.e. watch at any speed, forward
    /// or back. Set <see cref="ReplayPath"/> before Play, or call <see cref="LoadText"/> at runtime.
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class ReplayPlayer : MonoBehaviour
    {
        public string ReplayPath = "";
        public float FramesPerSecond = 3f;

        BoardRenderer _board;
        Replay _replay;
        int _frame;
        bool _playing;
        float _accum;

        Text _label;
        Slider _scrub;
        Text _playLabel;

        void Start()
        {
            _board = GetComponent<BoardRenderer>();
            EnsureEnvironment();
            BuildUi();

            if (!string.IsNullOrEmpty(ReplayPath) && File.Exists(ReplayPath))
                LoadText(File.ReadAllText(ReplayPath));
        }

        public void LoadText(string text)
        {
            var data = ReplayFile.Read(text);
            _replay = new Replay(data.Start, data.Commands);
            _board.Render(_replay.Frame(0).Board); // board is constant across frames
            _scrub.minValue = 0;
            _scrub.maxValue = Mathf.Max(0, _replay.FrameCount - 1);
            _frame = 0;
            _playing = _replay.FrameCount > 1;
            ShowFrame(0);

            var rig = FindAnyObjectByType<CameraRig>();
            if (rig != null) rig.Frame();
        }

        void Update()
        {
            if (_replay == null || !_playing) return;
            _accum += Time.deltaTime * Mathf.Max(0.01f, FramesPerSecond);
            while (_accum >= 1f && _frame < _replay.FrameCount - 1)
            {
                _accum -= 1f;
                ShowFrame(_frame + 1);
            }
            if (_frame >= _replay.FrameCount - 1) { _playing = false; UpdatePlayLabel(); }
        }

        void ShowFrame(int index)
        {
            if (_replay == null) return;
            _frame = Mathf.Clamp(index, 0, _replay.FrameCount - 1);
            var s = _replay.Frame(_frame);
            _board.RenderEntities(s);
            if (_scrub != null) _scrub.SetValueWithoutNotify(_frame);

            string status = s.IsGameOver
                ? (s.Winner == null ? "draw" : (s.Winner == PlayerId.Player0 ? "Player 1 wins" : "Player 2 wins"))
                : $"{(s.ActivePlayer == PlayerId.Player0 ? "Player 1" : "Player 2")}'s turn";
            if (_label != null)
                _label.text = $"Frame {_frame}/{_replay.FrameCount - 1}   Round {s.Round}   {status}";
        }

        // ---- UI ----

        void TogglePlay()
        {
            if (_replay == null) return;
            if (_frame >= _replay.FrameCount - 1) { ShowFrame(0); _accum = 0; }
            _playing = !_playing;
            UpdatePlayLabel();
        }

        void UpdatePlayLabel() { if (_playLabel != null) _playLabel.text = _playing ? "Pause" : "Play"; }

        void BuildUi()
        {
            EnsureEventSystem();

            var canvasGo = new GameObject("ReplayCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 600;
            var scaler = canvasGo.AddComponent<CanvasScaler>();
            scaler.uiScaleMode = CanvasScaler.ScaleMode.ScaleWithScreenSize;
            scaler.referenceResolution = new Vector2(1600f, 900f);
            scaler.matchWidthOrHeight = 0.5f;
            canvasGo.AddComponent<GraphicRaycaster>();

            var bar = Panel(canvasGo.transform, "Bar", new Color(0.05f, 0.06f, 0.10f, 0.9f));
            var brt = bar.GetComponent<RectTransform>();
            brt.anchorMin = new Vector2(0f, 0f);
            brt.anchorMax = new Vector2(1f, 0f);
            brt.pivot = new Vector2(0.5f, 0f);
            brt.sizeDelta = new Vector2(0f, 64f);
            brt.anchoredPosition = Vector2.zero;

            var playBtn = Button(bar.transform, new Vector2(0f, 0.5f), new Vector2(110f, 40f), new Vector2(12f, 0f), TogglePlay);
            _playLabel = playBtn.GetComponentInChildren<Text>();
            _playLabel.text = "Play";

            _scrub = MakeSlider(bar.transform);
            _scrub.onValueChanged.AddListener(v => { _playing = false; UpdatePlayLabel(); ShowFrame(Mathf.RoundToInt(v)); });

            var speed = MakeSpeedSlider(bar.transform);
            speed.onValueChanged.AddListener(v => FramesPerSecond = v);

            _label = Text(bar.transform, new Vector2(0f, 1f), new Vector2(700f, 24f), new Vector2(140f, -4f), "Replay");
            _label.alignment = TextAnchor.UpperLeft;
        }

        Slider MakeSlider(Transform parent)
        {
            var go = new GameObject("Scrub");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(0f, 0.5f); rt.anchorMax = new Vector2(0f, 0.5f); rt.pivot = new Vector2(0f, 0.5f);
            rt.sizeDelta = new Vector2(620f, 22f); rt.anchoredPosition = new Vector2(140f, 0f);
            return BuildSlider(go, 0f, 1f, 0f, true);
        }

        Slider MakeSpeedSlider(Transform parent)
        {
            var go = new GameObject("Speed");
            go.transform.SetParent(parent, false);
            var rt = go.AddComponent<RectTransform>();
            rt.anchorMin = new Vector2(1f, 0.5f); rt.anchorMax = new Vector2(1f, 0.5f); rt.pivot = new Vector2(1f, 0.5f);
            rt.sizeDelta = new Vector2(180f, 22f); rt.anchoredPosition = new Vector2(-12f, 0f);
            return BuildSlider(go, 0.5f, 12f, FramesPerSecond, false);
        }

        static Slider BuildSlider(GameObject go, float min, float max, float value, bool whole)
        {
            var slider = go.AddComponent<Slider>();
            slider.minValue = min; slider.maxValue = max; slider.wholeNumbers = whole; slider.value = value;

            var bg = new GameObject("BG"); bg.transform.SetParent(go.transform, false);
            var bgImg = bg.AddComponent<Image>(); bgImg.color = new Color(1, 1, 1, 0.2f);
            Stretch(bg.GetComponent<RectTransform>());

            var fillArea = new GameObject("Fill"); fillArea.transform.SetParent(go.transform, false);
            var fillImg = fillArea.AddComponent<Image>(); fillImg.color = new Color(0.4f, 0.8f, 1f, 0.9f);
            var fr = fillArea.GetComponent<RectTransform>();
            fr.anchorMin = new Vector2(0f, 0.25f); fr.anchorMax = new Vector2(1f, 0.75f); fr.offsetMin = Vector2.zero; fr.offsetMax = Vector2.zero;
            slider.fillRect = fr;

            var handle = new GameObject("Handle"); handle.transform.SetParent(go.transform, false);
            var hImg = handle.AddComponent<Image>(); hImg.color = Color.white;
            var hr = handle.GetComponent<RectTransform>(); hr.sizeDelta = new Vector2(14f, 24f);
            slider.handleRect = hr; slider.targetGraphic = hImg;
            return slider;
        }

        static void Stretch(RectTransform r) { r.anchorMin = Vector2.zero; r.anchorMax = Vector2.one; r.offsetMin = Vector2.zero; r.offsetMax = Vector2.zero; }

        static GameObject Panel(Transform parent, string name, Color color)
        {
            var go = new GameObject(name); go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = color;
            return go;
        }

        Button Button(Transform parent, Vector2 anchor, Vector2 size, Vector2 pos, UnityEngine.Events.UnityAction onClick)
        {
            var go = new GameObject("Button"); go.transform.SetParent(parent, false);
            go.AddComponent<Image>().color = new Color(0.20f, 0.34f, 0.55f, 1f);
            var btn = go.AddComponent<Button>(); btn.onClick.AddListener(onClick);
            var rt = go.GetComponent<RectTransform>();
            rt.anchorMin = anchor; rt.anchorMax = anchor; rt.pivot = anchor; rt.sizeDelta = size; rt.anchoredPosition = pos;
            var t = Text(go.transform, Vector2.zero, Vector2.zero, Vector2.zero, "");
            Stretch(t.GetComponent<RectTransform>()); t.alignment = TextAnchor.MiddleCenter;
            return btn;
        }

        static Text Text(Transform parent, Vector2 anchor, Vector2 size, Vector2 pos, string s)
        {
            var go = new GameObject("Text"); go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = 18; t.color = Color.white; t.text = s; t.alignment = TextAnchor.MiddleLeft;
            var rt = t.GetComponent<RectTransform>();
            rt.anchorMin = anchor; rt.anchorMax = anchor; rt.pivot = anchor; rt.sizeDelta = size; rt.anchoredPosition = pos;
            return t;
        }

        static void EnsureEventSystem()
        {
            if (FindAnyObjectByType<EventSystem>() != null) return;
            var es = new GameObject("EventSystem");
            es.AddComponent<EventSystem>();
            es.AddComponent<InputSystemUIInputModule>().AssignDefaultActions();
        }

        static void EnsureEnvironment()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.55f, 0.57f, 0.62f);
            if (FindAnyObjectByType<Light>() == null)
            {
                var l = new GameObject("KeyLight").AddComponent<Light>();
                l.type = LightType.Directional;
                l.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
                l.intensity = 1.1f; l.shadows = LightShadows.Soft;
            }
        }
    }
}
