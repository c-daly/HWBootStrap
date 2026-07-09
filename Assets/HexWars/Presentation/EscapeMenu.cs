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
            var kb = Keyboard.current;
            if (kb != null && kb.escapeKey.wasPressedThisFrame) Toggle();
        }

        public void Toggle()
        {
            if (_overlay != null) { Close(); return; }
            if (_game == null || _game.State == null || _game.DemoMode) return; // no game to leave on the title
            Open();
        }

        void Close()
        {
            if (_overlay != null) Destroy(_overlay);
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
            prt.sizeDelta = new Vector2(340f, online ? 306f : 256f);

            UiKit.Label(panel.transform, "MENU", 0f, -18f, 300f, 30f, UiKit.SizeTitle, TextAnchor.MiddleCenter);

            // sound row: mute toggle + volume steppers, all through SoundSettings (persisted master)
            Text volText = null;
            Text muteText = UiKit.Button(panel.transform, MuteLabel(), -90f, -64f, 100f, 40f, () =>
            {
                SoundSettings.MuteAll = !SoundSettings.MuteAll;
                RefreshSoundRow();
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeCaption).GetComponentInChildren<Text>();
            UiKit.Button(panel.transform, "-", 0f, -64f, 40f, 40f, () =>
            {
                SoundSettings.Volume -= 0.1f;
                RefreshSoundRow();
            }, UiKit.ButtonStyle.Secondary);
            volText = UiKit.Label(panel.transform, VolLabel(), 52f, -64f, 48f, 40f,
                                  UiKit.SizeBody, TextAnchor.MiddleCenter);
            UiKit.Button(panel.transform, "+", 104f, -64f, 40f, 40f, () =>
            {
                SoundSettings.Volume += 0.1f;
                RefreshSoundRow();
            }, UiKit.ButtonStyle.Secondary);
            _volText = volText;
            _muteText = muteText;

            UiKit.Button(panel.transform, "Resume", 0f, -120f, 280f, 44f, Close, UiKit.ButtonStyle.Cta);
            UiKit.Button(panel.transform, "Leave game", 0f, -174f, 280f, 44f, () =>
            {
                Close();
                _game.ReturnToMenu();
            }, UiKit.ButtonStyle.Danger);
            if (online)
                UiKit.Label(panel.transform,
                            "Leaving disconnects you - rejoin from the lobby\nwhile the room is held (about 10 minutes).",
                            0f, -228f, 320f, 40f, UiKit.SizeCaption, TextAnchor.UpperCenter, UiKit.TextDim);
        }

        Text _volText, _muteText;

        static string MuteLabel() => SoundSettings.MuteAll ? "Sound: Off" : "Sound: On";
        static string VolLabel() => Mathf.RoundToInt(SoundSettings.Volume * 100f) + "%";

        void RefreshSoundRow()
        {
            if (_volText != null) _volText.text = VolLabel();
            if (_muteText != null) _muteText.text = MuteLabel();
            SoundManager.Play(SoundKind.Move); // audible feedback at the new level (silent when muted)
        }
    }
}
