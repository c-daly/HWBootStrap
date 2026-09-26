using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using HexWars.Engine;

// (timestamp touch to force a fresh Unity recompile — the file's content was already correct)
namespace HexWars.Presentation
{
    /// <summary>
    /// Click to move or place a machine; inspect damage before committing an attack.
    /// The engine commits state; the shared presenter animates the accepted result.
    /// </summary>
    [DefaultExecutionOrder(-10)]
    [RequireComponent(typeof(UnitTooltip))]
    public sealed class UnitInputController : MonoBehaviour
    {
        UnitTooltip _tooltip;
        GameBootstrap _game;
        ModelDuelDriver _duelDriver;
        BarracksPanel _barracks;
        BoardRenderer _board;
        MovementHighlightController _movementHighlights;
        AttackTargetHighlightController _attackHighlights;
        readonly MovementPreviewState _movementPreview = new MovementPreviewState();
        readonly Dictionary<HexCoord, MovementRoute> _routes = new Dictionary<HexCoord, MovementRoute>();
        GameState _routesState;
        int _routesUnitId = -1;
        UnitView _selected;
        int _selectedId = -1;
        GameObject _marker;
        Material _markerMat;
        GameObject _actionGo;
        Transform _actionCanvas;
        RectTransform _territoryActionHost;
        Image _actionBg;
        Text _actionLabel;
        System.Action _actionOnClick;
        bool _buildMode; // territory: when on, taps place generators on any hex you control

        /// <summary>Spectator mode: hover tooltips and click-to-inspect still work, but no commands are
        /// issued (the AI is playing). Set by <see cref="SpectatorDriver"/> instead of disabling input.</summary>
        public bool ReadOnly;
        public enum Intent { Move, Attack }
        public Intent Mode { get; private set; } = Intent.Move;
        readonly List<HexCoord> _placementCells = new List<HexCoord>();
        HexCoord? _placementDestination;
        public IReadOnlyList<HexCoord> PlacementCells => _placementCells;
        public int SelectedId => _selectedId;
        public int TargetId { get; private set; } = -1;
        public bool AwaitingServer { get; private set; }
        public event System.Action PresentationChanged;
        GameState _previewState;
        GameState _submittedState;
        public PlayerId Viewer => _game != null && _game.State != null ? (_game.FogViewer() ?? _game.State.ActivePlayer) : PlayerId.Player0;
        public Unit? SelectedUnit => _game != null ? TacticalForecast.FindVisible(_game.State, _selectedId, Viewer) : null;
        public MovementRoute LockedRoute => _movementPreview.TouchLocked ? PreviewRoute() : null;
        public MovementRoute HoverRoute => PreviewRoute();
        public HexCoord? HoverDestination => _movementPreview.Destination;
        public HexCoord? Destination => _placementDestination ?? (_movementPreview.TouchLocked ? _movementPreview.Destination : null);
        public IReadOnlyDictionary<HexCoord, MovementRoute> Routes => _routes;
        public bool CanCommand => _game != null && _game.State != null && !ReadOnly && !_game.DemoMode
            && !_game.State.IsGameOver && !_game.Reconnecting && !AwaitingServer && _game.WaitingHumanSeat() == null;
        public bool CanUndoMove => CanCommand && _game.State.BeforeLastMove != null;

        public bool UndoLastMove()
        {
            if (!CanUndoMove) return false;
            int id = _game.State.LastMovedUnitId;
            SelectById(id);
            return Submit(new UndoMove(_game.State.ActivePlayer));
        }

        public void SetMode(Intent mode)
        {
            Mode = mode;
            ClearPreview();
            RefreshMovementRoutes();
            RefreshMovementHighlights();
        }

        public void ClearPreview()
        {
            TargetId = -1;
            _previewState = null;
            _placementDestination = null;
            _movementPreview.Clear();
            RefreshMovementHighlights();
            PresentationChanged?.Invoke();
        }

        public bool PreviewMove(HexCoord cell)
        {
            RefreshMovementRoutes();
            if (_game != null && _game.State != null && _game.State.PlacingStartingUnits)
            {
                if (!CanCommand || !_placementCells.Contains(cell)) return false;
                Mode = Intent.Move; TargetId = -1; _placementDestination = cell; _previewState = _game.State;
                RefreshMovementHighlights(); PresentationChanged?.Invoke(); return true;
            }
            if (!CanCommand || !_routes.ContainsKey(cell) || !SelectedUnit.HasValue) return false;
            if (!GameEngine.Apply(_game.State, new MoveUnit(_game.State.ActivePlayer, _selectedId, cell)).Success) return false;
            Mode = Intent.Move; TargetId = -1;
            _movementPreview.Tap(cell, true);
            _previewState = _game.State;
            RefreshMovementHighlights();
            PresentationChanged?.Invoke();
            return true;
        }

        public bool PreviewAttack(int id)
        {
            if (!CanCommand || _game.State.PlacingStartingUnits || !TacticalForecast.TryCreate(_game.State, Viewer, _selectedId, id, out _)) return false;
            Mode = Intent.Attack; TargetId = id; _movementPreview.Clear();
            _previewState = _game.State;
            RefreshMovementHighlights();
            PresentationChanged?.Invoke();
            return true;
        }

        public bool ConfirmPreview()
        {
            if (!CanCommand || !ReferenceEquals(_previewState, _game.State)) { ClearPreview(); return false; }
            Command command = null;
            if (_game.State.PlacingStartingUnits && _placementDestination.HasValue)
                command = new PlaceStartingUnit(_game.State.ActivePlayer, _selectedId, _placementDestination.Value);
            else if (Mode == Intent.Move && Destination.HasValue && _routes.ContainsKey(Destination.Value))
                command = new MoveUnit(_game.State.ActivePlayer, _selectedId, Destination.Value);
            if (Mode == Intent.Attack && TacticalForecast.TryCreate(_game.State, Viewer, _selectedId, TargetId, out _))
                command = new AttackUnit(_game.State.ActivePlayer, _selectedId, TargetId);
            if (command == null) { ClearPreview(); return false; }
            return Submit(command);
        }

        public bool FinishStartingPlacement()
        {
            if (!CanCommand || !_game.State.PlacingStartingUnits) return false;
            return Submit(new FinishPlacement(_game.State.ActivePlayer));
        }

        bool Submit(Command command)
        {
            ClearPreview();
            _game.Presenter?.FastForward();
            AwaitingServer = _game.Networked;
            _submittedState = AwaitingServer ? _game.State : null;
            bool accepted = _game.TryApply(command);
            if (!accepted) ClearAwaitingServer();
            ReacquireSelection();
            PresentationChanged?.Invoke();
            return accepted;
        }

        void OnStateChanged()
        {
            // Seat/socket status also raises StateChanged. Only an authoritative replacement
            // (APPLY or START resync) settles a command submitted against this immutable state.
            if (!ReferenceEquals(_submittedState, _game.State)) ClearAwaitingServer();
            TargetId = -1; _previewState = null;
            ReacquireSelection();
            PresentationChanged?.Invoke();
        }

        void ClearAwaitingServer() { AwaitingServer = false; _submittedState = null; }
        void OnRejected() { ClearAwaitingServer(); ClearPreview(); ReacquireSelection(); }
        void OnDestroy()
        {
            if (_game != null) { _game.StateChanged -= OnStateChanged; _game.CommandRejected -= OnRejected; }
            if (_markerMat != null) Destroy(_markerMat);
        }


        void Awake()
        {
            _tooltip = GetComponent<UnitTooltip>();
            _duelDriver = GetComponent<ModelDuelDriver>(); // arena: sits on the same GameObject, no _game
            BuildMarker();
        }

        void Start()
        {
            _game = FindAnyObjectByType<GameBootstrap>();
            _barracks = FindAnyObjectByType<BarracksPanel>();
            _board = FindAnyObjectByType<BoardRenderer>();
            if (_board != null)
            {
                _movementHighlights = _board.GetComponent<MovementHighlightController>()
                    ?? _board.gameObject.AddComponent<MovementHighlightController>();
                _attackHighlights = _board.GetComponent<AttackTargetHighlightController>()
                    ?? _board.gameObject.AddComponent<AttackTargetHighlightController>();
            }
            MakeActionButton();
            if (_game != null) { _game.StateChanged += OnStateChanged; _game.CommandRejected += OnRejected; }

        }

        Vector2 _pressPos;
        bool _pressedOverUi;
        const float TapThreshold = 24f; // px; beyond this, a press is treated as a camera drag, not a tap

        void Update()
        {
            var keyboard = DeviceInput.Allowed && !UiKit.AnyInputOwnsFocus() ? Keyboard.current : null;
            if (keyboard != null && _game != null && !_game.DemoMode)
            {
                if (keyboard.mKey.wasPressedThisFrame) SetMode(Intent.Move);
                if (keyboard.fKey.wasPressedThisFrame) SetMode(Intent.Attack);
                if (keyboard.zKey.wasPressedThisFrame &&
                    (keyboard.ctrlKey.isPressed || keyboard.leftMetaKey.isPressed || keyboard.rightMetaKey.isPressed)
                    && GameObject.Find("EscapeMenuCanvas") == null && GameObject.Find("RulesCanvas") == null
                    && !(_game.GetComponent<TacticalHud>()?.WorkshopOpen ?? false)) UndoLastMove();
                if (keyboard.escapeKey.wasPressedThisFrame && (Destination.HasValue || TargetId >= 0))
                { UiKit.MarkInputEscapeHandled(); ClearPreview(); }
            }
            var pointer = DeviceInput.Allowed ? Pointer.current : null; // mouse OR touch — one path for desktop and mobile
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

            RefreshMovementRoutes();
            bool isTouch = pointer is Touchscreen;
            if (!isTouch && Mode == Intent.Move) UpdateDesktopMovementPreview(IsPointerOverUi() ? null : hoveredTile);

            GameState inspectionState = ResolveInspectionState(
                _game != null ? _game.State : null,
                _duelDriver != null ? _duelDriver.PresentedState : null);

            var previewRoute = PreviewRoute();
            if (_game != null && _game.GetComponent<TacticalHud>() != null && !_game.DemoMode) _tooltip.Hide();
            else if (hoveredUnit != null)
            {
                var hoveredRoute = hoveredUnit.Unit.Id == _selectedId ? previewRoute : null;
                if (inspectionState != null) _tooltip.Show(hoveredUnit.Unit, mp, inspectionState, hoveredRoute);
                else _tooltip.Hide();
            }
            else if (_selected != null)
            {
                if (inspectionState != null) _tooltip.Show(_selected.Unit, mp, inspectionState, previewRoute);
                else _tooltip.Hide();
            }
            else _tooltip.Hide();

            // act on a TAP (press + release without dragging) so a drag is free to pan the camera
            if (pointer.press.wasPressedThisFrame) { _pressPos = mp; _pressedOverUi = IsPointerOverUi(); }
            bool blocked = _barracks != null && _barracks.IsDeploying;
            if (pointer.press.wasReleasedThisFrame && !blocked && !_pressedOverUi
                && Vector2.Distance(mp, _pressPos) <= TapThreshold)
                HandleBoardTap(hoveredUnit, hoveredTile, mp, Time.unscaledTimeAsDouble);

            if (_selected != null && _marker.activeSelf)
            {
                var p = _selected.transform.position;
                _marker.transform.position = new Vector3(p.x, p.y + .04f, p.z);

            }

            UpdateActionButton();
        }

        // Movement is direct. An attack's second click confirms the same preview without a timing
        // requirement; state replacement and Escape still disarm that preview.
        void HandleBoardTap(UnitView unit, TileView tile, Vector2 screenPosition, double now)
        {
            HandleClick(unit, tile);
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
                if (!target.HasValue) return; // tapped past the board — no message
                // guard first: during the opponent's turn "active" is them, so the controller check below
                // would misreport YOUR OWN hex as not yours
                if (_game.WaitingHumanSeat() != null) { Toast.Show("Opponent's turn — waiting for it to finish"); return; }
                if (bst.Board.Controller(target.Value) != active) { Toast.Show("You don't control that hex"); return; }
                if (HasGeneratorOn(bst, target.Value)) { Toast.Show("That hex already has a generator"); return; }
                _game.TryApply(new BuildGenerator(active, target.Value));
                return; // while building, taps only place generators
            }

            var selected = SelectedUnit;
            bool ownSelected = selected.HasValue && selected.Value.Owner == active;

            // attack intent: only fire if not already attacked AND actually targetable (range/vision/LOS/arc)
            if (ownSelected && unit != null && unit.Unit.Owner != active)
            {
                if (Mode == Intent.Attack && TargetId == unit.Unit.Id)
                { ConfirmPreview(); return; }
                // A visible opponent that cannot be attacked is still useful to inspect.
                if (!PreviewAttack(unit.Unit.Id)) SelectById(unit.Unit.Id);
                return;
            }
            // territory claim/build is done via the explicit on-screen action button (UpdateActionButton),
            // never by tapping the hex — so a stray tap can't spend points or end your turn by accident.

            // move intent: repeated commands are legal while the engine route map has the destination
            if (ownSelected && unit == null && tile != null)
            {
                RefreshMovementRoutes();
                if (tile.Coord == selected.Value.Cell) return; // A habitual second click is harmless.
                if (_game.State.PlacingStartingUnits)
                {
                    if (PreviewMove(tile.Coord)) ConfirmPreview();
                    else Toast.Show("Choose an empty hex in your starting area");
                    return;
                }
                if (!_routes.ContainsKey(tile.Coord))
                {
                    CancelMovementPreview();
                    Toast.Show(IsOccupied(_game.State, tile.Coord)
                        ? "That hex is occupied"
                        : "Out of movement reach");
                    return;
                }

                if (PreviewMove(tile.Coord)) ConfirmPreview();
                else if (CanCommand) Toast.Show("That move is not available now");
                return;
            }
            NotifyIfWaiting(unit, tile);
            Select(unit);
        }

        /// <summary>Which <see cref="GameState"/> hover/inspection reads against: the live simulation
        /// state when a <see cref="GameBootstrap"/> is present (the normal desktop/mobile game), falling
        /// back to the arena's <see cref="ModelDuelDriver.PresentedState"/> (what is actually on screen,
        /// never the sim running ahead of presentation) when there is no <c>_game</c> — the arena
        /// GameObject never carries one. Null when neither source has a state yet (pre-initialize);
        /// callers hide the tooltip and do nothing in that case rather than showing a stateless one.</summary>
        public static GameState ResolveInspectionState(GameState gameState, GameState presentedState) =>
            gameState ?? presentedState;

        static bool HasActed(System.Collections.Generic.IReadOnlyCollection<int> ids, int id)
        {
            foreach (var i in ids) if (i == id) return true;
            return false;
        }

        void RefreshMovementRoutes()
        {
            if (!TryGetSelectedStateUnit(out var selectedUnit))
            {
                ClearMovementRoutes();
                return;
            }

            var state = _game.State;
            if (object.ReferenceEquals(_routesState, state) && _routesUnitId == selectedUnit.Id)
                return;

            _routes.Clear(); _placementCells.Clear(); _placementDestination = null;
            if (state.PlacingStartingUnits) _placementCells.AddRange(StartingPlacement.Cells(state, selectedUnit));
            else foreach (var pair in MovementService.Routes(state, selectedUnit))
                _routes[pair.Key] = pair.Value;
            _routesState = state;
            _routesUnitId = selectedUnit.Id;
            _movementPreview.Clear();
            RefreshMovementHighlights();
        }

        bool TryGetSelectedStateUnit(out Unit selectedUnit)
        {
            selectedUnit = default;
            if (_game == null || _game.State == null || _selectedId < 0
                || ReadOnly || _game.DemoMode || _game.State.IsGameOver || _game.Reconnecting || AwaitingServer || _buildMode
                || (_barracks != null && _barracks.IsDeploying)
                || _game.WaitingHumanSeat() != null)
                return false;

            var state = _game.State;
            foreach (var unit in state.Player(state.ActivePlayer).UnitsOnBoard)
            {
                if (unit.Id != _selectedId || !unit.IsAlive) continue;
                selectedUnit = unit;
                return true;
            }
            return false;
        }

        void UpdateDesktopMovementPreview(TileView hoveredTile)
        {
            if (_movementPreview.TouchLocked || Mode != Intent.Move) return;
            HexCoord? destination = hoveredTile != null && _routes.ContainsKey(hoveredTile.Coord)
                ? (HexCoord?)hoveredTile.Coord
                : null;
            if (_movementPreview.Destination == destination) return;
            _movementPreview.Hover(destination);
            RefreshMovementHighlights();
            PresentationChanged?.Invoke();
        }

        MovementRoute PreviewRoute()
        {
            if (_movementPreview.Destination.HasValue
                && _routes.TryGetValue(_movementPreview.Destination.Value, out var route))
                return route;
            return null;
        }

        void CancelMovementPreview()
        {
            _movementPreview.Clear();
            RefreshMovementHighlights();
        }

        void ClearMovementRoutes()
        {
            _placementCells.Clear(); _placementDestination = null;
            _routes.Clear();
            _routesState = null;
            _routesUnitId = -1;
            _movementPreview.Clear();
            if (_movementHighlights != null) _movementHighlights.Clear();
            if (_attackHighlights != null) _attackHighlights.Clear();
        }

        void RefreshMovementHighlights()
        {
            if (_movementHighlights == null || _game == null || _game.State == null)
                return;
            if (_game.State.PlacingStartingUnits)
            {
                _movementHighlights.ShowPlacement(_game.State, _placementCells, _placementDestination);
                _attackHighlights?.Clear(); return;
            }
            if (Mode == Intent.Move) _movementHighlights.Show(_game.State, _routes, _movementPreview.Destination);
            else _movementHighlights.Clear();

            if (_attackHighlights == null) return;
            if (!TryGetSelectedStateUnit(out var attacker))
            {
                _attackHighlights.Clear();
                return;
            }

            var viewer = _game.FogViewer() ?? attacker.Owner;
            var current = AttackPreviewTargets.Resolve(_game.State, attacker, null, viewer);
            var possible = _movementPreview.Destination.HasValue
                ? AttackPreviewTargets.Resolve(_game.State, attacker, _movementPreview.Destination, viewer) : null;
            _attackHighlights.ShowAvailable(current, possible);
            if (Mode == Intent.Attack && TacticalForecast.TryCreate(_game.State, viewer, attacker.Id, TargetId, out var shot))
                _attackHighlights.ShowShot(attacker, shot.Target, LineOfSight.IsClear(_game.State.Board, attacker.Cell, attacker.Elevation, shot.Target.Cell, shot.Target.Elevation));
        }

        static bool HasGeneratorOn(GameState s, HexCoord cell)
        {
            foreach (var p in s.Players)
                foreach (var g in p.Generators)
                    if (g.IsAlive && g.Cell == cell) return true;
            return false;
        }

        /// <summary>TargetingService.CanTarget's three ANDed predicates, asked one at a time so the
        /// toast can say WHICH one refused. Ends on a generic fallback so a future rules change can
        /// never make this method lie.</summary>
        static string WhyCannotTarget(GameState s, Unit attacker, Unit target)
        {
            if (!TargetingService.InRange(attacker, target.Cell, target.Elevation, s.Config)) return "Out of range";
            if (!TargetingService.IsVisibleToArmy(s, attacker.Owner, target.Cell, target.Elevation)) return "No friendly unit can see the target";
            if (!TargetingService.HasShot(s, attacker, target.Cell, target.Elevation)) return "No line of sight";
            return "Can't target that unit";
        }

        static bool IsOccupied(GameState s, HexCoord cell)
        {
            foreach (var p in s.Players)
                foreach (var u in p.UnitsOnBoard)
                    if (u.IsAlive && u.Cell == cell) return true;
            return false;
        }

        /// <summary>A click that reads as an order (a live unit of the waiting human's is selected and
        /// they tapped a hex or an enemy) while the opponent's turn plays out: say why nothing will
        /// happen. Never fires in hotseat. Selection/inspection still proceeds after the toast.</summary>
        void NotifyIfWaiting(UnitView unit, TileView tile)
        {
            var waiting = _game != null ? _game.WaitingHumanSeat() : null;
            if (waiting == null || _selected == null || !_selected.Unit.IsAlive || _selected.Unit.Owner != waiting.Value) return;
            bool looksLikeOrder = tile != null || (unit != null && unit.Unit.Owner != waiting.Value);
            if (looksLikeOrder) Toast.Show("Opponent's turn — waiting for it to finish");
        }

        // attacking closes movement, so an attacker is immediately finished for this turn
        void AutoAdvance()
        {
            if (_game == null || _game.State == null || _selectedId < 0) return;
            bool spent = HasActed(_game.State.AttackedUnitIds, _selectedId);
            if (spent) SelectNextActionable();
        }

        void SelectNextActionable()
        {
            var active = _game.State.ActivePlayer;
            foreach (var u in _game.State.Player(active).UnitsOnBoard)
            {
                if (!u.IsAlive) continue;
                bool spent = HasActed(_game.State.AttackedUnitIds, u.Id);
                if (!spent) { SelectById(u.Id); return; }
            }
            Select(null); // every unit has acted this turn
        }

        public void SelectById(int id, bool audible = false)
        {
            if (_game != null && !TacticalForecast.FindVisible(_game.State, id, Viewer).HasValue) return;
            if (audible && id != _selectedId) SoundManager.Play(SoundKind.Select);
            ClearPreview();
            ClearMovementRoutes();
            Mode = Intent.Move;
            _selectedId = id;
            _selected = CurrentView(id);
            UpdateMarker();
            RefreshMovementRoutes();
            PresentationChanged?.Invoke();
        }

        public bool UnitCanAct(Unit unit)
        {
            if (!CanCommand || unit.Owner != _game.State.ActivePlayer || !unit.IsAlive) return false;
            if (_game.State.PlacingStartingUnits) return StartingPlacement.Cells(_game.State, unit).Count > 0;
            if (TacticalForecast.HasAttacked(_game.State, unit.Id)) return false;
            return MovementService.Routes(_game.State, unit).Count > 0
                || AttackPreviewTargets.Resolve(_game.State, unit, null, unit.Owner).Count > 0;
        }

        public void SelectReadyUnitOfKind(int id)
        {
            var clicked = TacticalForecast.FindVisible(_game.State, id, Viewer);
            if (!clicked.HasValue) return;
            if (!UnitCanAct(clicked.Value))
                foreach (var candidate in _game.State.Player(clicked.Value.Owner).UnitsOnBoard)
                    if (candidate.DisplayName == clicked.Value.DisplayName && candidate.Stats.Equals(clicked.Value.Stats)
                        && UnitArt.Resolve(candidate.ArtId, candidate.Stats) == UnitArt.Resolve(clicked.Value.ArtId, clicked.Value.Stats)
                        && UnitCanAct(candidate))
                    { SelectById(candidate.Id, true); return; }
            SelectById(id, true);
        }

        void Select(UnitView unit)
        {
            if (unit != null && unit.Unit.Id != _selectedId) SoundManager.Play(SoundKind.Select);
            ClearPreview();
            ClearMovementRoutes();
            Mode = Intent.Move;
            _selected = unit;
            _selectedId = unit != null ? unit.Unit.Id : -1;
            UpdateMarker();
            RefreshMovementRoutes();

            PresentationChanged?.Invoke();
            // gated on !DemoMode: the title-screen demo's units are hoverable/clickable (this class isn't
            // demo-aware), but a Tips bubble popping up over the muted showcase would break DemoMode's
            // whole point (suppressed gameplay UI) — see GameBootstrap.DemoMode's doc comment.
            if (unit != null && _game != null && !_game.DemoMode && Camera.main != null)
            {
                Vector2 screenPos = Camera.main.WorldToScreenPoint(unit.transform.position);
                TipsService.Show("first-select",
                    "Click a highlighted hex to move. Gold brackets mark targets in range now. Click an enemy to preview damage, then click it again to fire. Undo move takes back your last move.",
                    screenPos);
            }
        }

        /// <summary>Issue a command through the one presentation pipeline: finish any queued playback
        /// first (visuals catch up to truth), then apply. The presenter animates the result.</summary>
        void Issue(Command cmd)
        {
            _game.Presenter?.FastForward();
            if (!_game.TryApply(cmd))
            {
                CancelMovementPreview();
                return;
            }
            ReacquireSelection();
            AutoAdvance();
        }

        // TokenStore owns the current match. Scene-wide ID searches can find a previous demo's
        // pending-destruction tokens during the frame a new match starts.
        UnitView CurrentView(int id)
        {
            var token = _board != null ? _board.GetComponent<TokenStore>()?.UnitToken(id) : null;
            return token != null ? token.GetComponent<UnitView>() : null;
        }

        void ReacquireSelection()
        {
            ClearMovementRoutes();
            _selected = null;
            if (_selectedId >= 0 && SelectedUnit.HasValue) _selected = CurrentView(_selectedId);
            if (_selected == null) _selectedId = -1;
            UpdateMarker();
            RefreshMovementRoutes();
        }

        static bool IsPointerOverUi()
        {
            var es = UnityEngine.EventSystems.EventSystem.current;
            return es != null && es.IsPointerOverGameObject();
        }

        void BuildMarker()
        {
            _marker = new GameObject("SelectionBrackets");
            _marker.transform.SetParent(transform, false);
            var shader = Shader.Find("Universal Render Pipeline/Unlit") ?? Shader.Find("Unlit/Color");
            _markerMat = new Material(shader);
            _markerMat.color = new Color(.96f, .96f, .87f);
            if (_markerMat.HasProperty("_BaseColor")) _markerMat.SetColor("_BaseColor", _markerMat.color);
            for (int i = 0; i < 4; i++)
            {
                float x = (i % 2 == 0 ? -1 : 1) * .68f, z = (i / 2 == 0 ? -1 : 1) * .68f;
                var go = new GameObject("Corner"); go.transform.SetParent(_marker.transform, false);
                var line = go.AddComponent<LineRenderer>(); line.useWorldSpace = false; line.positionCount = 3;
                line.SetPositions(new[] { new Vector3(x*.56f,0,z), new Vector3(x,0,z), new Vector3(x,0,z*.56f) });
                line.startWidth = line.endWidth = .035f; line.sharedMaterial = _markerMat;
                line.shadowCastingMode = ShadowCastingMode.Off;
            }
            _marker.SetActive(false);
        }

        void UpdateMarker()
        {
            if (_selected == null) { _marker.SetActive(false); return; }
            var p = _selected.transform.position;
            _marker.transform.position = new Vector3(p.x, p.y + .04f, p.z);
            _marker.SetActive(true);
        }

        void MakeActionButton()
        {
            var canvasGo = new GameObject("ActionCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            // Canvas replaces the initial Transform with a RectTransform. Cache its final
            // parent so HUD rebuilds can detach the button before destroying the old canvas.
            _actionCanvas = canvasGo.transform;
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
            SetTerritoryActionHost(_territoryActionHost);
        }

        internal void SetTerritoryActionHost(RectTransform host)
        {
            _territoryActionHost = host;
            if (_actionGo == null || (host == null && _actionCanvas == null)) return;
            var rt = (RectTransform)_actionGo.transform;
            rt.SetParent(host != null ? host : _actionCanvas, false);
            if (host != null)
            {
                rt.anchorMin = Vector2.zero; rt.anchorMax = Vector2.one;
                rt.offsetMin = rt.offsetMax = Vector2.zero;
            }
            else
            {
                rt.anchorMin = rt.anchorMax = rt.pivot = new Vector2(.5f, 0);
                rt.sizeDelta = new Vector2(400, 56); rt.anchoredPosition = new Vector2(0, 92);
            }
        }

        void SetBuildMode(bool enabled)
        {
            if (_buildMode == enabled) return;
            _buildMode = enabled;
            ClearMovementRoutes();
            if (!enabled) RefreshMovementRoutes();
        }

        // Shows the one relevant territory action (claim or build) for the selected unit, with its cost and
        // consequence; greyed (no-op) with a reason when it can't be done. The only path to claim/build.
        void UpdateActionButton()
        {
            var st = _game != null ? _game.State : null;
            if (st == null || st.PlacingStartingUnits || ReadOnly || !st.Config.TerritoryMode || _selected == null
                || !_selected.Unit.IsAlive || _selected.Unit.Owner != st.ActivePlayer) { HideAction(); return; }

            var active = st.ActivePlayer;
            var cell = _selected.Unit.Cell;
            int points = st.Player(active).Points;
            bool actedAlready = st.MovedUnitIds.Count > 0 || st.AttackedUnitIds.Count > 0;
            int buildCost = Mathf.RoundToInt((float)(st.Config.BuildFactor * st.Config.GeneratorOutput));

            // build mode: tap any hex you control to place a generator there
            if (_buildMode)
            {
                ShowAction("Building — tap your hexes to place   ·   Done", () => SetBuildMode(false));
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
                ShowAction("Build generators   ·   tap your hexes", () => SetBuildMode(true));
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
