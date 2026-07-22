using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.EventSystems;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Right-side barracks: lists the LOCAL HUMAN's reusable templates (see <see cref="ShownSeat"/> —
    /// NOT necessarily <c>State.ActivePlayer</c>: online, always your own seat, never the opponent's;
    /// vs-AI, always your seat, never the AI's). Select one, then click a deployment-zone hex to deploy
    /// a paid clone (the template is not consumed). Stays in deploy mode so you can place several;
    /// re-click the selected template to stop.
    /// </summary>
    public sealed class BarracksPanel : MonoBehaviour
    {
        GameBootstrap _game;
        GameObject _canvasGo;
        RectTransform _list;
        Text _hint;
        BarracksTemplateTooltip _tooltip;
        int _deployIndex = -1;
        readonly List<Button> _rows = new List<Button>();

        public bool IsDeploying => _deployIndex >= 0;

        /// <summary>Spectator mode: still shows the active player's barracks, but the human can't deploy
        /// (the AI is playing). Set by <see cref="SpectatorDriver"/>.</summary>
        public bool ReadOnly;

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _tooltip = GetComponent<BarracksTemplateTooltip>() ?? gameObject.AddComponent<BarracksTemplateTooltip>();
            Build();
            if (_game != null) { _game.StateChanged += Rebuild; Rebuild(); }
        }

        void OnDestroy()
        {
            _tooltip?.Hide();
            if (_game != null) _game.StateChanged -= Rebuild;
        }

        void Update()
        {
            if (ReadOnly || _deployIndex < 0 || _game == null) return; // spectating: no human deploys
            var pointer = Pointer.current; // mouse or touch
            var cam = Camera.main;
            if (pointer == null || cam == null || !pointer.press.wasReleasedThisFrame || IsOverUi()) return;

            // Deploy mode is easy to leave: only a successful deploy keeps it. Clicking a unit,
            // an illegal hex, or empty space hands control back to the board (it used to stay
            // sticky until the template was re-clicked).
            var mp = pointer.position.ReadValue();
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
            {
                if (hit.collider.GetComponentInParent<UnitView>() != null) { StopDeploying(); return; }

                var tv = hit.collider.GetComponentInParent<TileView>();
                if (tv == null
                    || !_game.TryApply(new DeployUnit(_game.State.ActivePlayer, _deployIndex, tv.Coord)))
                    StopDeploying();
            }
            else StopDeploying();
        }

        void StopDeploying()
        {
            _deployIndex = -1;
            Rebuild();
        }

        void Build()
        {
            var canvasGo = UiKit.Canvas("BarracksCanvas", UiKit.OrderPanels, transform);
            _canvasGo = canvasGo;

            var canvasRt = canvasGo.GetComponent<RectTransform>();
            float availW = canvasRt != null && canvasRt.rect.width > 0f ? canvasRt.rect.width : 1200f;
            float w = Mathf.Min(230f, availW - 16f);
            var panel = UiKit.Panel(canvasGo.transform, "BarracksPanel", UiKit.Surface);
            var prt = panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(w, 420f);
            prt.anchoredPosition = new Vector2(-8f, -58f);

            UiKit.Label(panel.transform, "BARRACKS", 0f, -8f, w - 24f, 24f, 18, TextAnchor.MiddleLeft);
            _hint = UiKit.Label(panel.transform, "Design a unit, then deploy it here.", 0f, -34f, w - 24f, 22f, 13, TextAnchor.MiddleLeft);

            var listGo = new GameObject("List");
            listGo.transform.SetParent(panel.transform, false);
            _list = listGo.AddComponent<RectTransform>();
            UiKit.SetRect(_list, 0f, -60f, w, 360f);
        }

        /// <summary>Which seat's barracks this panel shows — deliberately NOT always
        /// <c>State.ActivePlayer</c> (audit I3): online, your own seat always, so the opponent's
        /// designs are never shown (info leak) or deletable mid-their-turn; vs-AI, the human's seat
        /// always, so the AI's templates are never shown/deletable mid-AI-turn; hotseat/spectator
        /// (incl. <see cref="SpectatorDriver"/>'s AI-vs-AI demo, which has no <see cref="AiOpponent"/>
        /// component), the active player — there's only ever one human at the screen either way.</summary>
        PlayerId ShownSeat()
        {
            if (_game.Networked) return _game.Seat ?? _game.State.ActivePlayer;
            var ai = _game.GetComponent<AiOpponent>();
            if (ai != null) return ai.AiSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            return _game.State.ActivePlayer;
        }

        void Rebuild()
        {
            _tooltip?.Hide();
            foreach (var r in _rows) Destroy(r.gameObject);
            _rows.Clear();
            if (_game == null) return;

            // hidden during the title demo and the connecting window (no state yet)
            if (_game.DemoMode || _game.State == null)
            {
                if (_canvasGo != null) _canvasGo.SetActive(false);
                return;
            }
            if (_canvasGo != null && !_canvasGo.activeSelf) _canvasGo.SetActive(true);

            var s = _game.State;
            var seat = ShownSeat();
            // whether the shown seat is the one actually allowed to act right now — deploy/delete are
            // both your-turn-only at the engine (top-level Apply() rejects any other issuer), so the
            // shown seat being idle (vs-AI's AI turn, or an online opponent's turn) must disable both.
            bool isActiveHuman = seat == s.ActivePlayer;
            if (!isActiveHuman && _deployIndex >= 0) _deployIndex = -1; // drop a stale selection — can't deploy on someone else's turn

            var p = s.Player(seat);
            if (_deployIndex >= p.Barracks.Count) _deployIndex = -1;

            int cheapest = int.MaxValue;
            for (int i = 0; i < p.Barracks.Count; i++)
            {
                var template = p.Barracks[i];
                // Starter templates always have a name; player-created ones may not (blank until named).
                string name = string.IsNullOrEmpty(template.Name) ? Roles.Dominant(template.Stats).ToString() : template.Name;
                int cost = Economy.DeployCost(template.Stats, s.Config);
                cheapest = Mathf.Min(cheapest, cost);
                bool selected = i == _deployIndex;
                int idx = i;
                // Interactable is baked from isActiveHuman ONLY — deliberately NOT ReadOnly (final
                // review N1): AiOpponent flips ReadOnly in its Update, one frame AFTER the AI's final
                // EndTurn already ran this Rebuild (StateChanged fires synchronously inside TryApply),
                // and nothing rebuilds again until the human acts — baking !ReadOnly here left the
                // whole panel dead with silent clicks at the top of every human turn. isActiveHuman is
                // computed from the STATE (ShownSeat vs ActivePlayer), so the hand-back rebuild enables
                // the rows immediately. Spectators (ReadOnly, set once by SpectatorDriver) are stopped
                // by the LIVE guards at click time instead: Select/DeleteAt/Update all check ReadOnly.
                // Name and cost are separate texts so a 20-char player name can't shove "deploy N" out
                // of the 170px row: the name is ellipsized left, the cost rides right-aligned on top
                // (UiKit.Label never raycasts, so clicks land on the select button underneath).
                var row = UiKit.Button(_list, UiKit.Ellipsize(name, 11), -28f, -(4f + i * 34f), 150f, 30f,
                                       () => Select(idx), UiKit.ButtonStyle.Secondary, 14);
                var rowText = row.GetComponentInChildren<Text>();
                rowText.alignment = TextAnchor.MiddleLeft;
                UiKit.SetRect(rowText.rectTransform, 0f, 0f, 132f, 30f); // 9px side insets inside the button
                UiKit.Label(row.transform, $"deploy {cost}", 0f, 0f, 132f, 30f, 11,
                            TextAnchor.MiddleRight, UiKit.TextFaint);
                UiKit.SetToggled(row, selected);
                row.interactable = isActiveHuman;
                row.gameObject.AddComponent<BarracksTemplateTooltipTarget>()
                    .Init(_tooltip, row.GetComponent<RectTransform>(), template, s.Config);
                _rows.Add(row);

                // Explicit touch target opens info without selecting/deploying the template.
                var info = UiKit.Button(_list, "i", 62f, -(4f + i * 34f), 28f, 30f,
                                        () => _tooltip.Show(row.GetComponent<RectTransform>(), template, s.Config),
                                        UiKit.ButtonStyle.Secondary, 13);
                _rows.Add(info);

                var del = UiKit.Button(_list, "✕", 100f, -(4f + i * 34f), 32f, 30f,
                                       () => DeleteAt(idx), UiKit.ButtonStyle.Danger, 14);
                del.interactable = isActiveHuman;
                _rows.Add(del);
            }

            _hint.text = _deployIndex >= 0
                ? "Click a zone hex to deploy - anywhere else to stop."
                : (p.Barracks.Count == 0 ? "Design a unit, then deploy it here." : "Select a template to deploy.");

            // spec §6: "First time points ≥ cheapest deploy cost with barracks open" — fires once per
            // game the moment it becomes true, whichever Rebuild() call (StateChanged-driven) sees it first.
            // !s.IsGameOver guards a known trigger collision (Task 12 review): a winning kill can make a
            // deploy affordable the same frame the game ends, and TipsService.Show is last-wins — without
            // this guard the deploy tip would silently eat the game-over rematch nudge. A deploy tip on
            // the game-over screen is useless anyway, so the guard is a pure win. isActiveHuman guards a
            // second collision (audit I3): without it, an AI's own barracks affording a deploy could fire
            // this tip at a human who isn't even the one being shown/acting.
            if (!s.IsGameOver && isActiveHuman && p.Barracks.Count > 0 && p.Points >= cheapest)
                TipsService.Show("can-afford-deploy", "Deploying costs the unit's points.");
        }

        void Select(int i)
        {
            if (ReadOnly) return; // live spectator guard — rows render interactable (see Rebuild's N1 note)
            _deployIndex = (_deployIndex == i) ? -1 : i; // toggle
            Rebuild();
        }

        /// <summary>Delete a barracks template — free, no turn cost (DeleteTemplate is administrative,
        /// not a game move; it's not in LegalMoves). Bookkeeping mirrors spec §5: deleting the selected
        /// row clears deploy mode; deleting a row before the selected one shifts the selected index down
        /// so it still points at the same template after the barracks list re-indexes.</summary>
        void DeleteAt(int index)
        {
            if (ReadOnly || _game == null || _game.State == null) return;
            var seat = _game.State.ActivePlayer;
            if (!_game.TryApply(new DeleteTemplate(seat, index))) return;
            if (_deployIndex == index) _deployIndex = -1;
            else if (_deployIndex > index) _deployIndex--;
        }

        static bool IsOverUi()
        {
            var es = EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }
    }
}
