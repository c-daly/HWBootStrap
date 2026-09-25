using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// In-game menu: Esc (desktop) or the corner "Menu" button (hosted by <see cref="HelpOverlay"/>'s
    /// corner cluster — mobile has no Esc) opens a modal with Resume / Leave game. Leaving returns to
    /// the title via <see cref="GameBootstrap.ReturnToMenu"/>; online that disconnects the socket, and
    /// the seat is token-held server-side, so rejoining from the lobby within the hold window resumes.
    /// </summary>
    [DefaultExecutionOrder(-1000)]
    public sealed class EscapeMenu : MonoBehaviour
    {
        GameBootstrap _game;
        GameObject _overlay;

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void AutoCreate()
        {
            if (FindAnyObjectByType<GameBootstrap>() == null) return;
            new GameObject("EscapeMenu").AddComponent<EscapeMenu>();
        }

        void Start() => _game = FindAnyObjectByType<GameBootstrap>();

        void Update()
        {
            var kb = DeviceInput.FocusProbe() ? Keyboard.current : null;
            if (kb != null && kb.escapeKey.wasPressedThisFrame && !UiKit.EscapeHandledThisFrame) HandleEscape();
        }

        public bool HandleEscape()
        {
            if (TryDismissContext()) { UiKit.MarkInputEscapeHandled(); return true; }
            if (UiKit.AnyInputOwnsFocus()) return false; // Ordinary fields keep their cancel-edit behavior.
            var input = FindAnyObjectByType<UnitInputController>();
            if (input != null && (input.Destination.HasValue || input.TargetId >= 0)) input.ClearPreview();
            else Toggle();
            UiKit.MarkInputEscapeHandled();
            return true;
        }

        public bool TryDismissContext()
        {
            // Dismiss exactly the topmost surface, immediately and without also opening the menu.
            var collection = FindAnyObjectByType<GraphiteWorkshop>();
            if (collection != null) { collection.Close(); return true; }
            var hud = _game != null ? _game.GetComponent<TacticalHud>() : FindAnyObjectByType<TacticalHud>();
            if (TacticalHud.ModalOpen && hud != null) { hud.CloseDialog(); return true; }
            var rules = GameObject.Find("RulesCanvas");
            if (rules != null) { rules.SetActive(false); Destroy(rules); return true; }
            if (_overlay != null) { Close(); return true; }
            if (GameObject.Find(GameOverBanner.RootName) != null) { GameOverBanner.Dismiss(); return true; }
            if (TipBubble.IsOpen) { TipBubble.Dismiss(); return true; }
            if (hud != null && hud.WorkshopOpen) { hud.DismissWorkshop(); return true; }
            return false;
        }

        public void Toggle()
        {
            if (_overlay != null) { Close(); return; }
            if (_game == null || _game.State == null || _game.DemoMode) return; // no game to leave on the title
            Open();
        }

        void Close()
        {
            if (_overlay != null) { _overlay.SetActive(false); Destroy(_overlay); }
            _overlay = null;
        }

        void Open()
        {
            _overlay = UiKit.Canvas("EscapeMenuCanvas", UiKit.OrderEscape, transform);

            // full-screen shield: dims the board and eats every click behind the menu
            var shield = new GameObject("Shield");
            shield.transform.SetParent(_overlay.transform, false);
            var img = shield.AddComponent<Image>();
            img.color = new Color(0f, 0f, 0f, 0.55f);
            var srt = shield.GetComponent<RectTransform>();
            srt.anchorMin = Vector2.zero;
            srt.anchorMax = Vector2.one;
            srt.offsetMin = srt.offsetMax = Vector2.zero;
            shield.AddComponent<Button>().onClick.AddListener(Close); // tap outside = resume

            var panel = UiKit.Panel(_overlay.transform, "Menu", UiKit.Surface);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = prt.pivot = new Vector2(0.5f, 0.5f);
            bool online = _game.Networked;
            prt.sizeDelta = new Vector2(340f, online ? 538f : 488f);

            UiKit.Label(panel.transform, "MENU", 0f, -18f, 300f, 30f, UiKit.SizeTitle, TextAnchor.MiddleCenter);

            _muteText = UiKit.Button(panel.transform, MuteLabel(), 0f, -60f, 280f, 32f, () =>
            {
                SoundSettings.MuteAll = !SoundSettings.MuteAll;
                _muteText.text = MuteLabel();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeCaption).GetComponentInChildren<Text>();
            SoundRow(panel.transform, "Master", -102f, () => SoundSettings.Volume, v => SoundSettings.Volume = v, true);
            SoundRow(panel.transform, "Effects", -138f, () => SoundSettings.Effects, v => SoundSettings.Effects = v, true);
            SoundRow(panel.transform, "Ambience", -174f, () => SoundSettings.Atmosphere, v => SoundSettings.Atmosphere = v, false);
            SoundRow(panel.transform, "Music", -210f, () => SoundSettings.Music, v => SoundSettings.Music = v, false);

            Text motion = null;
            motion = UiKit.Button(panel.transform, MotionLabel(), 0f, -258f, 280f, 36f, () =>
            {
                MotionSettings.Reduced = !MotionSettings.Reduced;
                if (MotionSettings.Reduced) _game.Presenter?.FastForward();
                motion.text = MotionLabel();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeCaption).GetComponentInChildren<Text>();
            UiKit.Button(panel.transform, "How to play", -73f, -304f, 134f, 36f,
                () => GameRules.Show(_overlay.transform, UiKit.Font(), UiKit.OrderEscape+10), UiKit.ButtonStyle.Secondary, UiKit.SizeCaption);
            TipsService.BuildToggle(panel.transform, 70f, -304f);
            UiKit.Button(panel.transform, "Resume", 0f, -358f, 280f, 44f, Close, UiKit.ButtonStyle.Cta);
            UiKit.Button(panel.transform, "Leave game", 0f, -410f, 280f, 44f, () =>
            {
                Close();
                _game.ReturnToMenu();
            }, UiKit.ButtonStyle.Danger);
            if (online)
                UiKit.Label(panel.transform,
                            "Leaving disconnects you - rejoin from the lobby\nwhile the room is held (about 10 minutes).",
                            0f, -464f, 320f, 40f, UiKit.SizeCaption, TextAnchor.UpperCenter, UiKit.TextDim);
        }

        Text _muteText;
        static string MotionLabel() => "Reduced motion: " + (MotionSettings.Reduced ? "On" : "Off");
        static string MuteLabel() => SoundSettings.MuteAll ? "Sound: Off" : "Sound: On";

        static void SoundRow(Transform parent, string name, float y, System.Func<float> get,
                             System.Action<float> set, bool preview)
        {
            UiKit.Label(parent, name, -92f, y, 90f, 30f, UiKit.SizeBody, TextAnchor.MiddleLeft);
            var value = UiKit.Label(parent, Percent(get()), 50f, y, 50f, 30f, UiKit.SizeBody, TextAnchor.MiddleCenter);
            void Adjust(float delta)
            {
                set(get() + delta); value.text = Percent(get());
                if (preview) SoundManager.Play(SoundKind.Select);
            }
            UiKit.Button(parent, "-", -5f, y, 36f, 30f, () => Adjust(-.1f), UiKit.ButtonStyle.Secondary);
            UiKit.Button(parent, "+", 105f, y, 36f, 30f, () => Adjust(.1f), UiKit.ButtonStyle.Secondary);
        }
        static string Percent(float value) => Mathf.RoundToInt(value * 100f) + "%";
    }
}
