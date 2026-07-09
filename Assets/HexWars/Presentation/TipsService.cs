using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace HexWars.Presentation
{
    /// <summary>
    /// The Tips coaching layer's one owner (spec §6): a persisted opt-out flag, and a once-per-game
    /// trigger registry so each moment-suggestion fires at most once. Renders through <see cref="TipBubble"/>
    /// in NON-MODAL mode (spec: "never blocks input") — no backdrop, nothing raycast-blocking except the
    /// card itself, so a coaching bubble never eats the player's next tap; it self-dismisses on that tap
    /// instead (see <see cref="TipBubble.Show"/>'s <c>modal</c> parameter). Entirely inert when
    /// <see cref="Enabled"/> is false: callers never need to check it themselves.
    /// </summary>
    public static class TipsService
    {
        const string PrefKey = "HexWars.Tips";
        static bool? _enabled;
        static readonly HashSet<string> _firedThisGame = new HashSet<string>();

        /// <summary>Defaults ON for a first-ever visit (no key written yet); persists after that.</summary>
        public static bool Enabled
        {
            get
            {
                if (_enabled == null) _enabled = PlayerPrefs.GetInt(PrefKey, 1) != 0;
                return _enabled.Value;
            }
            set
            {
                _enabled = value;
                PlayerPrefs.SetInt(PrefKey, value ? 1 : 0);
                PlayerPrefs.Save();
                if (!value) TipBubble.Dismiss(); // switching off must not leave a bubble hanging (spec §7:
                                                  // "off forever once off" — includes whatever's on screen right now
            }
        }

        /// <summary>Clear the once-per-game registry. Called by GameBootstrap on every real new-game
        /// entry point (NOT on a Task 6 reconnect's START re-deal — that's the same game continuing).</summary>
        public static void NewGame() => _firedThisGame.Clear();

        /// <summary>Show a tip at most once per game per <paramref name="id"/>, only while Tips is on.
        /// A no-op otherwise (off, or already fired this game) — callers never branch on Enabled. Always
        /// non-modal: a coaching bubble mid-play must never consume the player's next tap (spec's "never
        /// blocks input"), unlike the stat-reference popups callers reach directly via TipBubble.Show.</summary>
        public static void Show(string id, string text, Vector2? screenPos = null, string cta = null, System.Action onCta = null)
        {
            if (!Enabled) return;
            if (!_firedThisGame.Add(id)) return;
            var pos = screenPos ?? new Vector2(Screen.width / 2f, Screen.height / 2f);
            TipBubble.Show(text, pos, cta, onCta, modal: false);
        }

        /// <summary>Small reusable "Tips: On/Off" control — the title screen (bottom corner) and the
        /// in-game "?" (HelpOverlay) each place one of these. One tap flips <see cref="Enabled"/> and
        /// relabels itself; callers position the returned Button however fits their screen. Also carries
        /// a <see cref="LabelSync"/> so it stays correct even when the OTHER toggle (or a direct
        /// TipsService.Enabled set) changes the flag — HelpOverlay's copy is built exactly once per play
        /// session (RuntimeInitializeOnLoadMethod), so without this it could go stale for the rest of the
        /// session the moment the title-screen copy (rebuilt fresh on every menu visit) is used instead.</summary>
        public static Button BuildToggle(Transform parent, float x, float y)
        {
            Button btn = null;
            btn = UiKit.Button(parent, "Tips: " + (Enabled ? "On" : "Off"), x, y, 110f, 34f, () =>
            {
                Enabled = !Enabled;
            }, UiKit.ButtonStyle.Secondary, UiKit.SizeCaption);
            btn.gameObject.AddComponent<LabelSync>();
            return btn;
        }

        /// <summary>Keeps a BuildToggle button's label truthful regardless of which toggle (or which
        /// direct Enabled set) last changed the flag. Cheap: one bool compare per frame, a Text write
        /// only on an actual flip.</summary>
        sealed class LabelSync : MonoBehaviour
        {
            Text _label;
            bool _last;

            void Awake()
            {
                _label = GetComponentInChildren<Text>();
                _last = !Enabled; // force the first Update to sync, whatever Enabled is right now
            }

            void Update()
            {
                if (_label == null || Enabled == _last) return;
                _last = Enabled;
                _label.text = "Tips: " + (_last ? "On" : "Off");
            }
        }
    }
}
