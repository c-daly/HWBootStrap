using UnityEngine;
using UnityEngine.UI;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// The hovered/selected unit's full capabilities, docked in a fixed panel under the barracks
    /// (top-right) — it used to follow the cursor, which covered the board right where you were
    /// looking. Builds its own canvas/panel/text programmatically so no scene wiring is needed.
    /// </summary>
    public sealed class UnitTooltip : MonoBehaviour
    {
        const float Width = 230f;   // matches the barracks panel above it
        const float LineH = 22f;

        GameObject _panel;
        Text _text;

        void Awake()
        {
            var canvasGo = UiKit.Canvas("TooltipCanvas", UiKit.OrderTooltip, transform);

            var img = UiKit.Panel(canvasGo.transform, "Panel",
                                  new Color(UiKit.Surface.r, UiKit.Surface.g, UiKit.Surface.b, 0.95f));
            _panel = img.gameObject;
            img.raycastTarget = false; // informational only — never eat board clicks
            var prt = _panel.GetComponent<RectTransform>();
            prt.anchorMin = prt.anchorMax = new Vector2(1f, 1f);
            prt.pivot = new Vector2(1f, 1f);
            prt.sizeDelta = new Vector2(Width, 200f);
            prt.anchoredPosition = new Vector2(-8f, -486f); // barracks panel is (-8,-58) + 420 tall

            var textGo = new GameObject("Text");
            textGo.transform.SetParent(_panel.transform, false);
            _text = textGo.AddComponent<Text>();
            _text.font = UiKit.Font();
            _text.fontSize = 14;
            _text.color = Color.white;
            _text.supportRichText = true;
            _text.alignment = TextAnchor.UpperLeft;
            _text.horizontalOverflow = HorizontalWrapMode.Wrap;
            _text.verticalOverflow = VerticalWrapMode.Overflow;
            _text.raycastTarget = false;
            var trt = _text.GetComponent<RectTransform>();
            trt.anchorMin = Vector2.zero;
            trt.anchorMax = Vector2.one;
            trt.offsetMin = new Vector2(12f, 8f);
            trt.offsetMax = new Vector2(-12f, -8f);

            Hide();
        }

        int _cachedUnitId = -1;
        int _cachedHp = int.MinValue;
        bool _cachedMoved, _cachedAttacked, _cachedGameOver, _cachedIsOwnersTurn;

        public void Show(Unit unit, Vector2 screenPos) => Show(unit, screenPos, null);

        /// <summary>Docked panel: <paramref name="screenPos"/> is ignored (kept for call-site
        /// compatibility). With <paramref name="state"/>, the active player's own units also get
        /// "this turn" lines: movement/climb budget still unspent and whether the attack is ready.
        /// Called every frame a unit is hovered/selected (UnitInputController.Update), so it early-outs
        /// when nothing that affects the text has changed (audit P2 — this used to reformat and
        /// re-Split every frame regardless).</summary>
        public void Show(Unit unit, Vector2 screenPos, GameState state)
        {
            bool moved = false, attacked = false;
            if (state != null)
            {
                foreach (var id in state.MovedUnitIds) if (id == unit.Id) { moved = true; break; }
                foreach (var id in state.AttackedUnitIds) if (id == unit.Id) { attacked = true; break; }
            }
            bool gameOver = state != null && state.IsGameOver;
            // owner-turn bit mirrors Format()'s render condition for the "This turn" block: EndTurn
            // resets MovedUnitIds/AttackedUnitIds, so across a turn handover every other key field is
            // identical for an un-acted unit while the correct text differs
            bool isOwnersTurn = state != null && unit.Owner == state.ActivePlayer;
            bool unchanged = _panel.activeSelf && unit.Id == _cachedUnitId && unit.CurrentHp == _cachedHp
                            && moved == _cachedMoved && attacked == _cachedAttacked && gameOver == _cachedGameOver
                            && isOwnersTurn == _cachedIsOwnersTurn;
            if (unchanged) return;

            _cachedUnitId = unit.Id; _cachedHp = unit.CurrentHp;
            _cachedMoved = moved; _cachedAttacked = attacked; _cachedGameOver = gameOver;
            _cachedIsOwnersTurn = isOwnersTurn;

            string text = Format(unit, state);
            _text.text = text;
            var prt = _panel.GetComponent<RectTransform>();
            prt.sizeDelta = new Vector2(Width, LineH * LineCount(text) + 16f);
            _panel.SetActive(true);
        }

        static int LineCount(string s)
        {
            int n = 1;
            for (int i = 0; i < s.Length; i++) if (s[i] == '\n') n++;
            return n;
        }

        public void Hide()
        {
            if (_panel != null) _panel.SetActive(false);
        }

        static string Format(Unit u, GameState state)
        {
            var s = u.Stats;
            string owner = u.Owner == PlayerId.Player0 ? "Player 1" : "Player 2";
            string text =
                $"<b>{u.DisplayName}</b>  {s.PointCost} pts  ({owner})\n" +
                $"HP {u.CurrentHp}/{s.Health}\n" +
                $"Damage {s.Damage}   Defense {s.Defense}\n" +
                $"Move {s.Movement}   Vertical {s.VerticalMovement}\n" +
                $"Range {s.Range}   Arc {s.RangeArc}\n" +
                $"Vision {s.Vision}   Arc {s.VisionArc}";

            if (state != null && !state.IsGameOver && u.Owner == state.ActivePlayer)
            {
                var spent = state.MovementSpent.TryGetValue(u.Id, out var sp) ? sp : (H: 0, V: 0);
                int moveLeft = Mathf.Max(0, s.Movement - spent.H);
                int climbLeft = Mathf.Max(0, s.VerticalMovement - spent.V);
                bool attacked = false;
                foreach (var id in state.AttackedUnitIds) if (id == u.Id) { attacked = true; break; }
                text += $"\n<color=#9FD68C>This turn:  Move {moveLeft}/{s.Movement}  Climb {climbLeft}/{s.VerticalMovement}" +
                        $"\nAttack {(attacked ? "used" : "ready")}</color>";
            }
            return text;
        }
    }
}
