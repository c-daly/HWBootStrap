using System;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using UnityEngine.InputSystem;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The game-settings form, opened from the title screen in one of two modes: <b>Host</b> (online —
    /// adds a Private toggle; Create connects and shows a waiting screen with the room code, share link
    /// and Cancel) and <b>VsAi</b> (adds a Difficulty row; Create starts the local game immediately).
    /// Numeric settings use ordinary, prefilled in-game text fields with inline validation.
    /// Removes itself when a real match starts.
    /// </summary>
    public sealed class SetupForm : MonoBehaviour
    {
        public enum SetupMode { Host, VsAi }

        GameBootstrap _game;
        SetupMode _mode;
        GameObject _canvasGo;
        GameObject _form;
        GameObject _armyPanel;
        Text _armyLabel;
        Text _status;
        GameObject _cancelBtn;
        InlineIntBinding _seedBinding;

        GameMode _gameMode = GameMode.Annihilation;
        int _w = 9, _h = 7, _pts = 0, _seed = 7;
        int _armySize = 3, _brutes = 1, _strikers = 1, _snipers = 1;
        int _turnActions = 3;
        bool _fog = false;
        bool _private = false;
        AiLevel _ai = AiLevel.Hard;

        readonly System.Collections.Generic.List<(Button btn, Func<bool> selected)> _toggles
            = new System.Collections.Generic.List<(Button, Func<bool>)>();
        readonly System.Collections.Generic.List<InlineIntBinding> _bindings
            = new System.Collections.Generic.List<InlineIntBinding>();
        readonly System.Collections.Generic.List<InlineIntBinding> _armyBindings
            = new System.Collections.Generic.List<InlineIntBinding>();

        public static SetupForm Open(GameBootstrap game, SetupMode mode)
        {
            var existing = game.GetComponent<SetupForm>();
            if (existing != null) existing.Close();
            var form = game.gameObject.AddComponent<SetupForm>();
            form._game = game;
            form._mode = mode;
            return form; // Build runs in Start so _game/_mode are set first
        }

        void Start()
        {
            if (_game == null) _game = FindAnyObjectByType<GameBootstrap>();
            _seed = UnityEngine.Random.Range(1, 9999);
            Build();
            RefreshToggles();
        }

        void Update()
        {
            if (DeviceInput.Allowed && Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
            {
                var eventSystem = EventSystem.current ?? FindAnyObjectByType<EventSystem>();
                var selected = eventSystem != null ? eventSystem.currentSelectedGameObject : null;
                foreach (var binding in _bindings)
                {
                    if (selected == binding.Field.gameObject)
                    {
                        binding.Restore();
                        eventSystem.SetSelectedGameObject(null);
                        return;
                    }
                }
            }

            // a real match started (host's START arrived, or the vs-AI game began) — this form is done
            if (_game != null && _game.State != null && !_game.DemoMode) Close();
        }

        public void Close()
        {
            if (_canvasGo != null) Destroy(_canvasGo);
            Destroy(this);
        }

        void Build()
        {
            UiKit.EnsureEventSystem();
            _canvasGo = UiKit.Canvas("SetupCanvas", UiKit.OrderMenu, transform);

            float formW = Mathf.Min(700f, AvailWidth() - 40f); // GameRules' clamp pattern — a fixed
                                                                // 700-wide card must not overflow a narrow canvas
            _form = UiKit.Panel(_canvasGo.transform, "Form", UiKit.Surface).gameObject;
            var frt = _form.GetComponent<RectTransform>();
            frt.anchorMin = frt.anchorMax = new Vector2(0.5f, 0.5f);
            frt.pivot = new Vector2(0.5f, 0.5f);
            frt.sizeDelta = new Vector2(formW, 640f);
            frt.anchoredPosition = Vector2.zero;

            float y = -24f;
            string title = _mode == SetupMode.Host ? "Host Online Game" : "Play vs AI";
            UiKit.Label(_form.transform, title, 0f, y, 700f, 36f, UiKit.SizeTitle, TextAnchor.MiddleCenter);
            UiKit.Button(_form.transform, "Back", -300f, y - 2f, 90f, 34f, () => { Close(); TitleScreen.Reopen(_game); },
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            y -= 44f;
            UiKit.Label(_form.transform, "Tap a value and type the number you want", 0f, y, 700f, 22f,
                        UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint); y -= 40f;

            ToggleBtn("Annihilation", -95f, y, 180f, 38f, () => _gameMode == GameMode.Annihilation,
                      () => { _gameMode = GameMode.Annihilation; RefreshToggles(); });
            ToggleBtn("Territory", 100f, y, 160f, 38f, () => _gameMode == GameMode.Territory,
                      () => { _gameMode = GameMode.Territory; RefreshToggles(); });
            y -= 48f;

            NumberFieldRow(_form.transform, "Map width", y, _w, v => _w = v, 5, 64, false); y -= 46f;
            NumberFieldRow(_form.transform, "Map height", y, _h, v => _h = v, 5, 64, false); y -= 46f;
            NumberFieldRow(_form.transform, "Start points", y, _pts, v => _pts = v, 0, 200, true); y -= 46f;

            UiKit.Label(_form.transform, "Seed", -245f, y, 210f, 38f, UiKit.SizeBody + 2, TextAnchor.MiddleLeft, UiKit.TextDim);
            _seedBinding = UiKit.IntField(_form.transform, _seed, 60f, y, 130f, 38f,
                                          1, 99999, false, v => _seed = v);
            _seedBinding.Field.gameObject.name = "Seed";
            _bindings.Add(_seedBinding);
            UiKit.Button(_form.transform, "Reroll", 190f, y, 100f, 38f,
                         () =>
                         {
                             _seed = UnityEngine.Random.Range(1, 9999);
                             _seedBinding.SetCommittedValue(_seed);
                         },
                         UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            y -= 48f;

            var armyBtn = UiKit.Button(_form.transform, "", 0f, y, 500f, 40f, OpenArmy, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            _armyLabel = armyBtn.GetComponentInChildren<Text>();
            _armyLabel.text = ArmySummary();
            y -= 48f;

            NumberFieldRow(_form.transform, "Units acting per turn", y, _turnActions,
                           v => _turnActions = v, 0, int.MaxValue, true);
            UiKit.Label(_form.transform, "0 = whole team / unlimited", 155f, y - 28f, 300f, 18f,
                        UiKit.SizeCaption, TextAnchor.MiddleLeft, UiKit.TextFaint);
            y -= 48f;

            if (_mode == SetupMode.Host)
            {
                ToggleBtn("Fog of war", -140f, y, 220f, 38f, () => _fog, () => { _fog = !_fog; RefreshToggles(); });
                ToggleBtn("Private (invite only)", 120f, y, 250f, 38f, () => _private, () => { _private = !_private; RefreshToggles(); });
            }
            else
            {
                ToggleBtn("Fog of war", -140f, y, 220f, 38f, () => _fog, () => { _fog = !_fog; RefreshToggles(); });
                ToggleBtn(AiLabel(_ai), 120f, y, 250f, 38f, () => _ai == AiLevel.Hard, () =>
                {
                    _ai = _ai == AiLevel.Hard ? AlternateAiLevel() : AiLevel.Hard;
                    RefreshToggles();
                    foreach (var (btn, sel) in _toggles)
                    {
                        var t = btn.GetComponentInChildren<Text>();
                        if (t != null && t.text.StartsWith("AI: ")) t.text = AiLabel(_ai);
                    }
                });
            }
            y -= 54f;

            string cta = _mode == SetupMode.Host ? "Create Game" : "Start Game";
            UiKit.Button(_form.transform, cta, 0f, y, 340f, 50f, OnCreate, UiKit.ButtonStyle.Cta);

            _status = UiKit.Label(_canvasGo.transform, "", 0f, 0f, Mathf.Min(1100f, AvailWidth() - 40f), 160f,
                                  UiKit.SizeHeading, TextAnchor.MiddleCenter);
            _status.horizontalOverflow = HorizontalWrapMode.Wrap; // was Overflow — the clamp above only
                                                                   // helps once long lines can actually wrap
            var srt = _status.GetComponent<RectTransform>();
            srt.anchorMin = srt.anchorMax = new Vector2(0.5f, 0.5f);
            srt.pivot = new Vector2(0.5f, 0.5f);
            srt.anchoredPosition = new Vector2(0f, 30f);

            BuildArmyPopup();
        }

        /// <summary>The canvas's actual rendered width (GameRules' clamp pattern) — a fixed layout must
        /// not overflow a narrow/portrait screen.</summary>
        float AvailWidth()
        {
            var rt = _canvasGo.GetComponent<RectTransform>();
            return rt != null && rt.rect.width > 0f ? rt.rect.width : 1200f;
        }

        string ArmySummary()
        {
            int spec = _brutes + _strikers + _snipers;
            if (spec <= 0) return $"Army:  {_armySize} random   ▸";
            string roles = $"{_brutes} Brute, {_strikers} Striker, {_snipers} Sniper";
            if (spec < _armySize) roles += " + random";
            return $"Army:  {roles}   ▸";
        }

        void BuildArmyPopup()
        {
            _armyPanel = new GameObject("ArmyPopup");
            _armyPanel.transform.SetParent(_canvasGo.transform, false);
            var prt = _armyPanel.AddComponent<RectTransform>();
            UiKit.Stretch(prt);

            var dim = UiKit.Panel(_armyPanel.transform, "Dim", new Color(0.02f, 0.03f, 0.06f, 0.75f));
            UiKit.Stretch(dim.GetComponent<RectTransform>());
            dim.sprite = null; // full-bleed dim, no rounding
            dim.gameObject.AddComponent<Button>().onClick.AddListener(CloseArmy);

            var card = UiKit.Panel(_armyPanel.transform, "Card", UiKit.Surface).gameObject;
            var crt = card.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.sizeDelta = new Vector2(Mathf.Min(700f, AvailWidth() - 40f), 430f);
            crt.anchoredPosition = Vector2.zero;

            float y = -24f;
            UiKit.Label(card.transform, "Starting army", 0f, y, 700f, 34f, UiKit.SizeTitle - 3, TextAnchor.MiddleCenter); y -= 48f;
            NumberFieldRow(card.transform, "Army size", y, _armySize, v => _armySize = v, 1, 12, false, true); y -= 46f;
            NumberFieldRow(card.transform, "Brutes", y, _brutes, v => _brutes = v, 0, 12, true, true); y -= 46f;
            NumberFieldRow(card.transform, "Strikers", y, _strikers, v => _strikers = v, 0, 12, true, true); y -= 46f;
            NumberFieldRow(card.transform, "Snipers", y, _snipers, v => _snipers = v, 0, 12, true, true); y -= 44f;
            UiKit.Label(card.transform, "Leave roles at 0 for a random army; extra slots fill randomly.",
                        0f, y, 700f, 22f, UiKit.SizeCaption, TextAnchor.MiddleCenter, UiKit.TextFaint); y -= 40f;
            UiKit.Button(card.transform, "Done", 0f, y, 220f, 44f, CloseArmy, UiKit.ButtonStyle.Cta, UiKit.SizeHeading);

            _armyPanel.SetActive(false);
        }

        void OpenArmy() { if (_armyPanel != null) _armyPanel.SetActive(true); }

        void CloseArmy()
        {
            bool valid = true;
            foreach (var binding in _armyBindings) valid &= binding.Commit();
            if (!valid) return;
            if (_armyPanel != null) _armyPanel.SetActive(false);
            if (_armyLabel != null) _armyLabel.text = ArmySummary();
        }

        InlineIntBinding NumberFieldRow(Transform parent, string label, float y, int initial,
                                        Action<int> set, int min, int max, bool blankMeansZero,
                                        bool armyField = false)
        {
            UiKit.Label(parent, label, -245f, y, 210f, 38f, UiKit.SizeBody + 2, TextAnchor.MiddleLeft, UiKit.TextDim);
            var binding = UiKit.IntField(parent, initial, 80f, y, 150f, 38f,
                                         min, max, blankMeansZero, set);
            binding.Field.gameObject.name = label;
            _bindings.Add(binding);
            if (armyField) _armyBindings.Add(binding);
            return binding;
        }

        void ToggleBtn(string text, float x, float y, float w, float h, Func<bool> selected, Action onClick)
        {
            var b = UiKit.Button(_form.transform, text, x, y, w, h, onClick, UiKit.ButtonStyle.Secondary, UiKit.SizeBody);
            _toggles.Add((b, selected));
        }

        void RefreshToggles()
        {
            foreach (var (btn, selected) in _toggles) UiKit.SetToggled(btn, selected());
        }

        void OnCreate()
        {
            bool valid = true;
            foreach (var binding in _bindings) valid &= binding.Commit();
            if (!valid)
            {
                if (_armyBindings.Exists(binding => !string.IsNullOrEmpty(binding.Error.text))) OpenArmy();
                return;
            }

            var setup = new GameSetup(_gameMode, _w, _h, _pts, _seed,
                                      _armySize, _brutes, _strikers, _snipers, _turnActions, _fog);
            if (_mode == SetupMode.VsAi)
            {
                if (_ai == AiLevel.TrainedModel &&
                    !PlayableModelAdapter.Supports(setup, out string reason))
                {
                    Toast.Show(reason);
                    return;
                }
                try
                {
                    // StartLocalGame performs a full tactical-v3 observation preflight before it
                    // publishes the state, including cached barracks and table capacities.
                    _game.StartLocalGame(setup, true, _ai);
                }
                catch (Exception error) when (
                    _ai == AiLevel.TrainedModel &&
                    (error is ArgumentException || error is InvalidOperationException))
                {
                    Toast.Show(error.Message);
                    return;
                }
                // form dismisses via Update when State exists
                return;
            }

            string room = RandomCode();
            _game.StartNetGame(room, setup.ToWire(), _private);
            ShowWaiting(room);
        }

        void ShowWaiting(string room)
        {
            if (_form != null) _form.SetActive(false);
            if (_armyPanel != null) _armyPanel.SetActive(false);
            _status.text = $"Room code\n<size=64><b>{room}</b></size>\n\nWaiting for an opponent…\n" +
                           (_private ? "Private game — share the code or link below.\n" : "Your game is listed in Browse Games.\n") +
                           ShareUrl(room);
            _status.supportRichText = true;

            _cancelBtn = UiKit.Button(_canvasGo.transform, "Cancel", 0f, 0f, 200f, 44f, () =>
            {
                _game.CancelHosting();
                Close();
                TitleScreen.Reopen(_game);
            }, UiKit.ButtonStyle.Danger, UiKit.SizeBody + 2).gameObject;
            var crt = _cancelBtn.GetComponent<RectTransform>();
            crt.anchorMin = crt.anchorMax = new Vector2(0.5f, 0.5f);
            crt.pivot = new Vector2(0.5f, 0.5f);
            crt.anchoredPosition = new Vector2(0f, -150f);
        }

        /// <summary>The socket died while this form's waiting screen was up (host waiting for an
        /// opponent, or a join in flight). No-op before Create (the form is still showing, not the
        /// waiting screen) and after Close (nothing left to update) — the waiting screen is the only
        /// state where the form is hidden with status text already up.</summary>
        public void OnConnectionLost()
        {
            if (_form != null && !_form.activeSelf && _status != null && !string.IsNullOrEmpty(_status.text))
                _status.text = "Connection lost — Cancel and try again.";
        }

        internal static string RandomCode()
        {
            const string alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
            var c = new char[5];
            for (int i = 0; i < c.Length; i++) c[i] = alphabet[UnityEngine.Random.Range(0, alphabet.Length)];
            return new string(c);
        }

        internal static string ShareUrl(string room)
        {
            string page = Application.absoluteURL;
            if (string.IsNullOrEmpty(page)) return "(this page) ?room=" + room;
            int q = page.IndexOf('?');
            if (q >= 0) page = page.Substring(0, q);
            return page + "?room=" + room;
        }

        static AiLevel AlternateAiLevel()
        {
#if UNITY_WEBGL && !UNITY_EDITOR
            // Browser builds cannot spawn the separate Python policy server.  Keep the existing
            // shippable Random/Greedy pair until a frozen in-process model is promoted.
            return AiLevel.Easy;
#else
            return AiLevel.TrainedModel;
#endif
        }

        static string AiLabel(AiLevel level)
        {
            if (level == AiLevel.Hard) return "AI: Greedy";
            if (level == AiLevel.TrainedModel) return "AI: Trained model";
            return "AI: Random";
        }
    }
}
