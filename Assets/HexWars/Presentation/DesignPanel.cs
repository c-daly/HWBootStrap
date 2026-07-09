using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Left-side panel to design a unit: +/- each of the 9 stats (Health floored at 1), see live
    /// PointCost + dominant role, and bank a free template to the active player's barracks.
    /// </summary>
    public sealed class DesignPanel : MonoBehaviour
    {
        static readonly string[] Names =
            { "Health", "Damage", "Defense", "Movement", "Vertical Move", "Range", "Range Arc", "Vision", "Vision Arc" };

        static readonly string[] NamePlaceholders = { "Doom Turtle", "Longshot", "Pathfinder" };

        GameBootstrap _game;
        GameObject _canvasGo;
        readonly int[] _stats = new int[9];
        readonly Text[] _valueLabels = new Text[9];
        Text _summary;
        Text _nameBox;
        string _name = "";
        int _placeholderIdx;
        bool _lastTipsEnabled;

        void Start()
        {
            _stats[0] = 1; // Health >= 1
            _game = FindAnyObjectByType<GameBootstrap>();
            _lastTipsEnabled = TipsService.Enabled;
            Build();
            RefreshSummary();
        }

        // DesignPanel has no StateChanged-driven refresh, so poll: one SetActive per flip of the
        // hide condition (the equality guard keeps it from thrashing the canvas every frame), like
        // GameHud's guard. Hidden during the title demo and the connecting window (no state yet).
        // Also polls the Tips toggle: flipping it while the panel is already built must show/hide the
        // inline captions immediately, which needs a full rebuild (row spacing itself changes).
        void Update()
        {
            if (_game == null || _canvasGo == null) return;
            bool hidden = _game.DemoMode || _game.State == null;
            if (_canvasGo.activeSelf == hidden)
            {
                _canvasGo.SetActive(!hidden);
                if (hidden) SoundManager.StopDesignerHum(); else SoundManager.StartDesignerHum();
            }

            if (TipsService.Enabled != _lastTipsEnabled)
            {
                _lastTipsEnabled = TipsService.Enabled;
                Destroy(_canvasGo);
                Build();
                RefreshSummary();
                _canvasGo.SetActive(!hidden); // Build() always creates it active — restore the hidden state above
            }
        }

        void Build()
        {
            var canvasGo = UiKit.Canvas("DesignCanvas", UiKit.OrderPanels, transform);
            _canvasGo = canvasGo;

            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            const float rowH = 30f, top = 58f;
            float w = Mathf.Min(270f, availW - 16f);
            bool tipsOn = TipsService.Enabled;         // an inline caption line under each stat row while on
            float rowSlot = tipsOn ? rowH + 15f : rowH;
            var panelImg = UiKit.Panel(canvasGo.transform, "DesignPanel", UiKit.Surface);
            var prt = panelImg.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(0f, 1f);
            prt.pivot = new Vector2(0f, 1f);
            prt.sizeDelta = new Vector2(w, rowSlot * 9 + rowH + 126f); // 9 (slotted) stat rows + the Name
                                                               // row (fixed rowH, no caption) + 126f padding
            prt.anchoredPosition = new Vector2(8f, -top);
            var panel = panelImg.transform;

            UiKit.Label(panel, "DESIGN UNIT", 0f, -8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);

            for (int i = 0; i < 9; i++)
            {
                float y = -(40f + i * rowSlot);
                int idx = i;
                // the label itself is the tap target — a stat name button with a text-only look, opening
                // the verbatim description (spec §6: "always available, Tips or no Tips")
                var nameBtn = UiKit.Button(panel, Names[i], -63f, y, 120f, rowH, () =>
                {
                    Vector3 world = panel.TransformPoint(new Vector3(-63f, y, 0f));
                    Vector2 screenPos = RectTransformUtility.WorldToScreenPoint(null, world); // camera
                                                                    // null — this canvas is ScreenSpaceOverlay
                    TipBubble.Show(StatInfo.All[idx].Full, screenPos);
                }, UiKit.ButtonStyle.Secondary, 15);
                nameBtn.GetComponentInChildren<Text>().alignment = TextAnchor.MiddleLeft;
                _valueLabels[i] = UiKit.Label(panel, "0", 23f, y, 40f, rowH, 16, TextAnchor.MiddleCenter);
                UiKit.Button(panel, "-", 65f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, -1));
                UiKit.Button(panel, "+", 105f, y - 2f, 36f, rowH - 4f, () => Adjust(idx, +1));
                if (tipsOn) // spec §6: designer-opened stat rows show their one-line captions inline while Tips is on
                    // x=0, width w-24: same full-width-minus-padding convention as the "DESIGN UNIT" header
                    // and _summary below (UiKit.Label's anchor pivot is the label's CENTER, not its left
                    // edge — the original x=-63/w-90 pairing put the caption's left edge ~18px past the
                    // panel's own left border, clipping the first couple of characters of every caption).
                    UiKit.Label(panel, StatInfo.All[i].Caption, 0f, y - rowH + 2f, w - 24f, 15f,
                               11, TextAnchor.UpperLeft, UiKit.TextFaint);
            }
            for (int i = 0; i < 9; i++) _valueLabels[i].text = _stats[i].ToString(); // sync display to
                                                                                      // current _stats — matters
                                                                                      // once the Tips-toggle rebuild
                                                                                      // below can re-run this Build()
                                                                                      // after the player has already
                                                                                      // spent points

            float nameY = -(40f + 9 * rowSlot + 6f);
            UiKit.Label(panel, "Name", -63f, nameY, 60f, rowH, 15, TextAnchor.MiddleLeft);
            _nameBox = UiKit.Button(panel, PlaceholderText(), 23f, nameY, w - 110f, rowH, OnTapName,
                                    UiKit.ButtonStyle.Secondary, 14).GetComponentInChildren<Text>();
            ApplyNameDisplay(); // sets the grey placeholder color (UiKit.Button's own label defaults to white)

            float sy = nameY - rowH - 6f;
            _summary = UiKit.Label(panel, "", 0f, sy, w - 24f, 24f, 15, TextAnchor.MiddleLeft);
            UiKit.Button(panel, "Create (to Barracks)", 0f, sy - 30f, w - 24f, 30f, OnCreate, UiKit.ButtonStyle.Cta);
        }

        /// <summary>Called by GameBootstrap's first-bounty Tips CTA ("Design your answer"). This panel
        /// has no separate open/closed state to "open" — it's already visible whenever a game is active
        /// — so this draws the eye instead: a brief bright pulse on the panel background. Honest
        /// implementation of "opens the Designer" given the panel's always-visible design, not a fake
        /// no-op click handler.</summary>
        public void Highlight()
        {
            if (_canvasGo == null || !_canvasGo.activeSelf) return;
            StopAllCoroutines();
            StartCoroutine(PulseRoutine());
        }

        System.Collections.IEnumerator PulseRoutine()
        {
            var img = _canvasGo.transform.Find("DesignPanel")?.GetComponent<Image>();
            if (img == null) yield break;
            float t = 0f;
            while (t < 0.8f)
            {
                t += Time.deltaTime;
                img.color = Color.Lerp(UiKit.Accent, UiKit.Surface, t / 0.8f);
                yield return null;
            }
            img.color = UiKit.Surface;
        }

        void Adjust(int i, int delta)
        {
            _stats[i] = Mathf.Max(i == 0 ? 1 : 0, _stats[i] + delta);
            _valueLabels[i].text = _stats[i].ToString();
            RefreshSummary();
        }

        string PlaceholderText() => NamePlaceholders[_placeholderIdx];

        /// <summary>Browser-prompt text entry (the established mobile pattern, same as join-by-code).
        /// Empty stays legal — CreateUnit/UnitTemplate.Sanitize defaults an empty name to the dominant
        /// role at the engine boundary.</summary>
        void OnTapName()
        {
            string typed = UiKit.PromptText("Name your unit", _name);
            if (typed == null) return; // cancelled, or no browser prompt available (editor) — leave as-is
            _name = typed.Trim();
            if (_name.Length == 0) RotatePlaceholder();
            ApplyNameDisplay();
        }

        /// <summary>Advances to the next example (Task 12's Tips-toggle rebuild also calls
        /// <see cref="ApplyNameDisplay"/> to resync the box's text/color after a rebuild, but must NOT
        /// rotate the placeholder just because the panel redrew — only an actual "went back to empty"
        /// user action should pick a new example, so the two are kept separate.</summary>
        void RotatePlaceholder() => _placeholderIdx = (_placeholderIdx + 1) % NamePlaceholders.Length;

        /// <summary>Pure display sync — safe to call any time the name box exists and needs to reflect
        /// current state (after typing, after Create, after a rebuild).</summary>
        void ApplyNameDisplay()
        {
            if (_name.Length > 0) { _nameBox.text = _name; _nameBox.color = UiKit.TextMain; }
            else { _nameBox.text = PlaceholderText(); _nameBox.color = UiKit.TextFaint; } // grey, per spec
        }

        void RefreshSummary()
        {
            var s = ToStats();
            _summary.text = $"Cost {s.PointCost}   Role: {Roles.Dominant(s)}";
        }

        UnitStats ToStats() =>
            new UnitStats(_stats[0], _stats[1], _stats[2], _stats[3], _stats[4], _stats[5], _stats[6], _stats[7], _stats[8]);

        void OnCreate()
        {
            if (_game == null || _game.State == null) return;
            // Client-side sanitize before the command is even built — the engine still backstops this
            // (UnitTemplate.Sanitize runs again at the CreateUnit boundary), but doing it here too means
            // what's echoed back in APPLY / shown in the barracks matches what the player typed, instead
            // of silently differing only after a round-trip.
            string sanitized = UnitTemplate.Sanitize(_name);
            if (_game.TryApply(new CreateUnit(_game.State.ActivePlayer, ToStats(), sanitized)))
            {
                // TryApply's optimistic `true` for a Networked game isn't a server verdict — the server
                // may yet reject it, so the visible "it worked" cues (sound + clearing the name box)
                // must wait for the real APPLY/REJECT round-trip, not fire on the local guess.
                if (!_game.Networked)
                {
                    SoundManager.Play(SoundKind.Design);
                    _name = "";
                    RotatePlaceholder(); // a fresh empty box next time shows a different example
                    ApplyNameDisplay();
                }
            }
        }
    }
}
