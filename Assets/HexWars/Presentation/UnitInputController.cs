using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using HexWars.Engine;

// (timestamp touch to force a fresh Unity recompile — the file's content was already correct)
namespace HexWars.Presentation
{
    /// <summary>
    /// Mouse interaction for units. Hover shows a capability tooltip. Click your own unit to select
    /// (a marker floats above it). With one of your units selected: click an empty hex to <b>move</b>
    /// (slides there), or click an enemy to <b>attack</b> (fires a projectile; the target explodes on
    /// a kill). Animations play, then the engine command is applied.
    /// </summary>
    [RequireComponent(typeof(UnitTooltip))]
    public sealed class UnitInputController : MonoBehaviour
    {
        UnitTooltip _tooltip;
        GameBootstrap _game;
        BarracksPanel _barracks;
        BoardRenderer _board;
        UnitView _selected;
        int _selectedId = -1;
        GameObject _marker;
        Material _markerMat;
        GameObject _actionGo;
        Image _actionBg;
        Text _actionLabel;
        System.Action _actionOnClick;
        bool _buildMode; // territory: when on, taps place generators on any hex you control

        /// <summary>Spectator mode: hover tooltips and click-to-inspect still work, but no commands are
        /// issued (the AI is playing). Set by <see cref="SpectatorDriver"/> instead of disabling input.</summary>
        public bool ReadOnly;

        void Awake()
        {
            _tooltip = GetComponent<UnitTooltip>();
            BuildMarker();
        }

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _barracks = FindAnyObjectByType<BarracksPanel>();
            _board = FindAnyObjectByType<BoardRenderer>();
            MakeActionButton();
        }

        Vector2 _pressPos;
        bool _pressedOverUi;
        const float TapThreshold = 24f; // px; beyond this, a press is treated as a camera drag, not a tap

        void Update()
        {
            var pointer = Pointer.current; // mouse OR touch — one path for desktop and mobile
            var cam = Camera.main;
            if (pointer == null || cam == null) return;

            Vector2 mp = pointer.position.ReadValue();
            UnitView hoveredUnit = null;
            TileView hoveredTile = null;
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
            {
                hoveredUnit = hit.collider.GetComponentInParent<UnitView>();
                hoveredTile = hit.collider.GetComponentInParent<TileView>();
            }

            if (hoveredUnit != null) _tooltip.Show(hoveredUnit.Unit, mp, _game.State);
            else if (_selected != null) _tooltip.Show(_selected.Unit, mp, _game.State);
            else _tooltip.Hide();

            // act on a TAP (press + release without dragging) so a drag is free to pan the camera
            if (pointer.press.wasPressedThisFrame) { _pressPos = mp; _pressedOverUi = IsPointerOverUi(); }
            bool blocked = _barracks != null && _barracks.IsDeploying;
            if (pointer.press.wasReleasedThisFrame && !blocked && !_pressedOverUi
                && Vector2.Distance(mp, _pressPos) <= TapThreshold)
                HandleClick(hoveredUnit, hoveredTile);

            if (_selected != null && _marker.activeSelf)
            {
                var p = _selected.transform.position;
                float bob = Mathf.Sin(Time.time * 4f) * 0.08f;
                _marker.transform.position = new Vector3(p.x, p.y + 0.85f + bob, p.z);

                // yellow if this unit is yours to command this turn, gray otherwise
                bool mine = _game != null && _game.State != null && _selected.Unit.Owner == _game.State.ActivePlayer;
                var c = mine ? new Color(1f, 0.92f, 0.15f) : new Color(0.6f, 0.6f, 0.65f);
                if (_markerMat != null) { if (_markerMat.HasProperty("_BaseColor")) _markerMat.SetColor("_BaseColor", c); _markerMat.color = c; }
            }

            UpdateActionButton();
        }

        void HandleClick(UnitView unit, TileView tile)
        {
            if (ReadOnly) { Select(unit); return; } // spectating: inspect any unit, but issue no commands
            if (_game == null || _game.State == null) { Select(unit); return; }
            var active = _game.State.ActivePlayer;

            // build mode (territory): a tap places a generator on any empty hex you control
            if (_buildMode && _game.State.Config.TerritoryMode)
            {
                var bst = _game.State;
                HexCoord? target = tile != null ? (HexCoord?)tile.Coord : (unit != null ? (HexCoord?)unit.Unit.Cell : null);
                if (target.HasValue && bst.Board.Controller(target.Value) == active && !HasGeneratorOn(bst, target.Value))
                    _game.TryApply(new BuildGenerator(active, target.Value));
                return; // while building, taps only place generators (or do nothing)
            }

            bool ownSelected = _selected != null && _selected.Unit.Owner == active && _selected.Unit.IsAlive;

            // attack intent: only fire if not already attacked AND actually targetable (range/vision/LOS/arc)
            if (ownSelected && unit != null && unit.Unit.Owner != active)
            {
                if (!HasActed(_game.State.AttackedUnitIds, _selected.Unit.Id)
                    && TargetingService.CanTarget(_game.State, _selected.Unit, unit.Unit.Cell, unit.Unit.Elevation))
                    Issue(new AttackUnit(active, _selected.Unit.Id, unit.Unit.Id));
                return; // invalid / spent: nothing happens, keep selection
            }
            // territory claim/build is done via the explicit on-screen action button (UpdateActionButton),
            // never by tapping the hex — so a stray tap can't spend points or end your turn by accident.

            // move intent: only if not already moved AND the hex is reachable
            if (ownSelected && unit == null && tile != null)
            {
                if (!HasActed(_game.State.MovedUnitIds, _selected.Unit.Id)
                    && IsReachable(_selected.Unit, tile.Coord))
                    Issue(new MoveUnit(active, _selected.Unit.Id, tile.Coord));
                return;
            }
            Select(unit);
        }

        bool IsReachable(Unit unit, HexCoord dest)
        {
            foreach (var c in MovementService.ReachableTiles(_game.State, unit))
                if (c == dest) return true;
            return false;
        }

        static bool HasActed(System.Collections.Generic.IReadOnlyCollection<int> ids, int id)
        {
            foreach (var i in ids) if (i == id) return true;
            return false;
        }

        static bool HasGeneratorOn(GameState s, HexCoord cell)
        {
            foreach (var p in s.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == cell) return true;
            return false;
        }

        // once the acting unit has used both its move and attack, jump to the next unit with actions left
        void AutoAdvance()
        {
            if (_game == null || _game.State == null || _selectedId < 0) return;
            bool spent = HasActed(_game.State.MovedUnitIds, _selectedId) && HasActed(_game.State.AttackedUnitIds, _selectedId);
            if (spent) SelectNextActionable();
        }

        void SelectNextActionable()
        {
            var active = _game.State.ActivePlayer;
            foreach (var u in _game.State.Player(active).UnitsOnBoard)
            {
                if (!u.IsAlive) continue;
                bool spent = HasActed(_game.State.MovedUnitIds, u.Id) && HasActed(_game.State.AttackedUnitIds, u.Id);
                if (!spent) { SelectById(u.Id); return; }
            }
            Select(null); // every unit has acted this turn
        }

        void SelectById(int id)
        {
            _selectedId = id;
            _selected = null;
            foreach (var v in FindObjectsByType<UnitView>(FindObjectsSortMode.None))
                if (v.Unit.Id == id && v.Unit.IsAlive) { _selected = v; break; }
            UpdateMarker();
        }

        void Select(UnitView unit)
        {
            _selected = unit;
            _selectedId = unit != null ? unit.Unit.Id : -1;
            UpdateMarker();
        }

        /// <summary>Issue a command through the one presentation pipeline: finish any queued playback
        /// first (visuals catch up to truth), then apply. The presenter animates the result.</summary>
        void Issue(Command cmd)
        {
            _game.Presenter?.FastForward();
            _game.TryApply(cmd);
            ReacquireSelection();
            AutoAdvance();
        }

        void ReacquireSelection()
        {
            _selected = null;
            if (_selectedId >= 0)
                foreach (var v in FindObjectsByType<UnitView>(FindObjectsSortMode.None))
                    if (v.Unit.Id == _selectedId && v.Unit.IsAlive) { _selected = v; break; }
            if (_selected == null) _selectedId = -1;
            UpdateMarker();
        }

        static bool IsPointerOverUi()
        {
            var es = UnityEngine.EventSystems.EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }

        void BuildMarker()
        {
            _marker = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            _marker.name = "SelectionMarker";
            var col = _marker.GetComponent<Collider>();
            if (col != null) Destroy(col);
            _marker.transform.SetParent(transform, false);
            _marker.transform.localScale = Vector3.one * 0.42f;

            var mr = _marker.GetComponent<MeshRenderer>();
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            var yellow = new Color(1f, 0.92f, 0.15f);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", yellow);
            m.color = yellow;
            mr.sharedMaterial = m;
            _markerMat = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;

            _marker.SetActive(false);
        }

        void UpdateMarker()
        {
            if (_selected == null) { _marker.SetActive(false); return; }
            var p = _selected.transform.position;
            _marker.transform.position = new Vector3(p.x, p.y + 0.85f, p.z);
            _marker.SetActive(true);
        }

        void MakeActionButton()
        {
            var canvasGo = new GameObject("ActionCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 460;
            canvasGo.AddComponent<GraphicRaycaster>();

            _actionGo = new GameObject("ActionButton");
            _actionGo.transform.SetParent(canvasGo.transform, false);
            _actionBg = _actionGo.AddComponent<Image>();
            var btn = _actionGo.AddComponent<Button>();
            btn.onClick.AddListener(() => { if (_actionOnClick != null) _actionOnClick(); });
            var rt = _actionGo.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0f); rt.anchorMax = new Vector2(0.5f, 0f);
            rt.pivot = new Vector2(0.5f, 0f); rt.sizeDelta = new Vector2(400f, 56f);
            rt.anchoredPosition = new Vector2(0f, 92f);

            var labelGo = new GameObject("Label");
            labelGo.transform.SetParent(_actionGo.transform, false);
            _actionLabel = labelGo.AddComponent<Text>();
            _actionLabel.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            _actionLabel.fontSize = 18; _actionLabel.color = Color.white;
            _actionLabel.alignment = TextAnchor.MiddleCenter; _actionLabel.raycastTarget = false;
            var lrt = _actionLabel.GetComponent<RectTransform>();
            lrt.anchorMin = Vector2.zero; lrt.anchorMax = Vector2.one; lrt.offsetMin = Vector2.zero; lrt.offsetMax = Vector2.zero;

            _actionGo.SetActive(false);
        }

        // Shows the one relevant territory action (claim or build) for the selected unit, with its cost and
        // consequence; greyed (no-op) with a reason when it can't be done. The only path to claim/build.
        void UpdateActionButton()
        {
            var st = _game != null ? _game.State : null;
            if (st == null || ReadOnly || !st.Config.TerritoryMode || _selected == null
                || !_selected.Unit.IsAlive || _selected.Unit.Owner != st.ActivePlayer) { HideAction(); return; }

            var active = st.ActivePlayer;
            var cell = _selected.Unit.Cell;
            int points = st.Player(active).Points;
            bool actedAlready = st.MovedUnitIds.Count > 0 || st.AttackedUnitIds.Count > 0;
            int buildCost = Mathf.RoundToInt((float)(st.Config.BuildFactor * st.Config.GeneratorOutput));

            // build mode: tap any hex you control to place a generator there
            if (_buildMode)
            {
                ShowAction("Building — tap your hexes to place   ·   Done", () => _buildMode = false);
                return;
            }

            // claim the unit's own hex if it isn't yours yet
            if (st.Board.Controller(cell) != active)
            {
                int cost = st.Config.CaptureCost;
                if (st.Config.ClaimEndsTurn && actedAlready)
                    ShowAction("Claim hex  —  army already acted this turn", null);
                else if (points < cost)
                    ShowAction($"Claim hex  —  need {cost} pts (have {points})", null);
                else
                    ShowAction($"Claim hex   ·   {cost} pts, ends turn",
                               () => { _game.TryApply(new CaptureHex(active, cell)); ReacquireSelection(); });
                return;
            }

            // standing on your own territory: place generators on any hex you control
            if (points >= buildCost)
                ShowAction("Build generators   ·   tap your hexes", () => _buildMode = true);
            else
                ShowAction($"Build generator  —  need {buildCost} pts (have {points})", null);
        }

        void ShowAction(string label, System.Action onClick)
        {
            if (_actionGo == null) return;
            _actionGo.SetActive(true);
            _actionLabel.text = label;
            _actionOnClick = onClick;
            _actionBg.color = onClick != null ? new Color(0.20f, 0.52f, 0.32f, 0.97f) : new Color(0.32f, 0.34f, 0.38f, 0.92f);
        }

        void HideAction()
        {
            _actionOnClick = null;
            if (_actionGo != null) _actionGo.SetActive(false);
        }
    }
}
