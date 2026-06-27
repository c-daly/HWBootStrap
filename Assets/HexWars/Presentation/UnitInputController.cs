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
        bool _animating;
        Text _hint;

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
            _hint = MakeHintLabel();
        }

        void Update()
        {
            var mouse = Mouse.current;
            var cam = Camera.main;
            if (mouse == null || cam == null) return;

            Vector2 mp = mouse.position.ReadValue();
            UnitView hoveredUnit = null;
            TileView hoveredTile = null;
            if (Physics.Raycast(cam.ScreenPointToRay(mp), out var hit, 1000f))
            {
                hoveredUnit = hit.collider.GetComponentInParent<UnitView>();
                hoveredTile = hit.collider.GetComponentInParent<TileView>();
            }

            if (hoveredUnit != null) _tooltip.Show(hoveredUnit.Unit, mp);
            else if (_selected != null) _tooltip.Show(_selected.Unit, mp);
            else _tooltip.Hide();

            bool blocked = _animating || IsPointerOverUi() || (_barracks != null && _barracks.IsDeploying);
            if (mouse.leftButton.wasPressedThisFrame && !blocked)
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

            if (_hint != null) _hint.text = HintText();
        }

        void HandleClick(UnitView unit, TileView tile)
        {
            if (ReadOnly) { Select(unit); return; } // spectating: inspect any unit, but issue no commands
            if (_game == null || _game.State == null) { Select(unit); return; }
            var active = _game.State.ActivePlayer;
            bool ownSelected = _selected != null && _selected.Unit.Owner == active && _selected.Unit.IsAlive;

            // attack intent: only fire if not already attacked AND actually targetable (range/vision/LOS/arc)
            if (ownSelected && unit != null && unit.Unit.Owner != active)
            {
                if (!HasActed(_game.State.AttackedUnitIds, _selected.Unit.Id)
                    && TargetingService.CanTarget(_game.State, _selected.Unit, unit.Unit.Cell, unit.Unit.Elevation))
                    StartCoroutine(AttackSeq(_selected, unit));
                return; // invalid / spent: nothing happens, keep selection
            }
            // territory: click your selected unit's OWN hex to claim it (if not yours) or build on it
            if (ownSelected && unit == null && tile != null
                && _game.State.Config.TerritoryMode && tile.Coord == _selected.Unit.Cell)
            {
                var st = _game.State;
                var cell = _selected.Unit.Cell;
                if (st.Board.Controller(cell) != active)
                    _game.TryApply(new CaptureHex(active, cell));        // claim / convert (ends the turn)
                else if (!HasGeneratorOn(st, cell))
                    _game.TryApply(new BuildGenerator(active, cell));    // build on owned empty hex
                ReacquireSelection();
                return;
            }
            // move intent: only if not already moved AND the hex is reachable
            if (ownSelected && unit == null && tile != null)
            {
                if (!HasActed(_game.State.MovedUnitIds, _selected.Unit.Id)
                    && IsReachable(_selected.Unit, tile.Coord))
                    StartCoroutine(MoveSeq(_selected, tile.Coord));
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

        int UnitHp(int id)
        {
            foreach (var p in _game.State.Players)
                foreach (var u in p.UnitsOnBoard)
                    if (u.Id == id) return u.CurrentHp;
            return 0;
        }

        void Select(UnitView unit)
        {
            _selected = unit;
            _selectedId = unit != null ? unit.Unit.Id : -1;
            UpdateMarker();
        }

        IEnumerator MoveSeq(UnitView mover, HexCoord dest)
        {
            _animating = true;
            var tr = mover.transform;
            Vector3 from = tr.position, to = HexTopWorld(dest);
            for (float t = 0f; t < 0.3f; t += Time.deltaTime)
            {
                tr.position = Vector3.Lerp(from, to, Mathf.SmoothStep(0f, 1f, t / 0.3f));
                yield return null;
            }
            bool ok = _game.TryApply(new MoveUnit(_game.State.ActivePlayer, _selectedId, dest));
            if (!ok && _board != null) _board.RenderEntities(_game.State); // illegal: snap back to truth
            ReacquireSelection();
            AutoAdvance();
            _animating = false;
        }

        IEnumerator AttackSeq(UnitView attacker, UnitView target)
        {
            _animating = true;

            // attack VFX scales with the attacker's Damage: small fast bullet -> fat slow rocket
            int dmg = attacker.Unit.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            Color projColor = dmg >= 6 ? new Color(1f, 0.3f, 0.1f)
                            : dmg >= 3 ? new Color(1f, 0.65f, 0.2f)
                                       : new Color(1f, 0.95f, 0.5f);

            int targetId = target.Unit.Id;
            int hpBefore = target.Unit.CurrentHp;
            Vector3 targetPos = target.transform.position;
            Vector3 from = attacker.transform.position + Vector3.up * 0.4f;
            Vector3 to = targetPos + Vector3.up * 0.4f;

            // direct shot flies straight; an indirect (LOS-blocked) shot lobs over the obstacles
            bool directLos = LineOfSight.IsClear(_game.State.Board,
                attacker.Unit.Cell, attacker.Unit.Elevation, target.Unit.Cell, target.Unit.Elevation);
            float arc = directLos ? 0f : Mathf.Max(2.5f, Vector3.Distance(from, to) * 0.35f);
            float flightDur = Mathf.Lerp(0.45f, 0.85f, power) + Vector3.Distance(from, to) * 0.035f; // slower, weightier

            var proj = MakeProjectile(from, projScale, projColor);
            for (float t = 0f; t < flightDur; t += Time.deltaTime)
            {
                float f = t / flightDur;
                var pos = Vector3.Lerp(from, to, f);
                pos.y += Mathf.Sin(f * Mathf.PI) * arc;
                proj.transform.position = pos;
                yield return null;
            }
            Destroy(proj);

            bool ok = _game.TryApply(new AttackUnit(_game.State.ActivePlayer, _selectedId, targetId));
            if (ok)
            {
                int dealt = Mathf.Max(0, hpBefore - UnitHp(targetId));
                DamagePopup.Spawn(targetPos + Vector3.up * 1.1f, dealt.ToString(), new Color(1f, 0.92f, 0.4f));
                if (!IsUnitAlive(targetId))
                    ExplosionFx.Spawn(targetPos, new Color(0.95f, 0.45f, 0.18f), Mathf.Lerp(0.8f, 2.0f, power), true);
                else
                    ExplosionFx.Spawn(to, projColor, Mathf.Lerp(0.4f, 0.9f, power), false);
            }
            ReacquireSelection();
            AutoAdvance();
            _animating = false;
        }

        Vector3 HexTopWorld(HexCoord cell)
        {
            float hexSize = _board != null ? _board.HexSize : 1f;
            float levelH = _board != null ? _board.LevelHeight : 0.55f;
            int elev = _game.State.Board.TileAt(cell).Elevation;
            var w = HexLayout.ToWorld(cell, hexSize);
            return new Vector3((float)w.x, (elev + 1) * levelH, (float)w.z);
        }

        bool IsUnitAlive(int id)
        {
            foreach (var p in _game.State.Players)
                foreach (var u in p.UnitsOnBoard)
                    if (u.Id == id && u.IsAlive) return true;
            return false;
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

        GameObject MakeProjectile(Vector3 pos, float scale, Color color)
        {
            var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            DestroyImmediate(p.GetComponent<Collider>());
            p.transform.position = pos;
            p.transform.localScale = Vector3.one * scale;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", color);
            m.color = color;
            var mr = p.GetComponent<MeshRenderer>();
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;

            var trail = p.AddComponent<TrailRenderer>();
            trail.time = 0.18f;
            trail.startWidth = scale * 0.9f;
            trail.endWidth = 0f;
            trail.material = m;
            trail.startColor = color;
            trail.endColor = new Color(color.r, color.g, color.b, 0f);
            trail.numCapVertices = 2;
            return p;
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

        Text MakeHintLabel()
        {
            var canvasGo = new GameObject("HintCanvas");
            canvasGo.transform.SetParent(transform, false);
            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;
            canvas.sortingOrder = 450;
            var go = new GameObject("Hint");
            go.transform.SetParent(canvasGo.transform, false);
            var t = go.AddComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            t.fontSize = 18; t.color = new Color(1f, 0.95f, 0.6f); t.alignment = TextAnchor.LowerCenter;
            var rt = t.GetComponent<RectTransform>();
            rt.anchorMin = new Vector2(0.5f, 0f); rt.anchorMax = new Vector2(0.5f, 0f);
            rt.pivot = new Vector2(0.5f, 0f); rt.sizeDelta = new Vector2(700f, 30f);
            rt.anchoredPosition = new Vector2(0f, 70f);
            return t;
        }

        string HintText()
        {
            if (ReadOnly || _game == null || _game.State == null) return "";
            var st = _game.State;
            if (!st.Config.TerritoryMode || _selected == null) return "";
            if (_selected.Unit.Owner != st.ActivePlayer) return "";
            var cell = _selected.Unit.Cell;
            bool actedAlready = st.MovedUnitIds.Count > 0 || st.AttackedUnitIds.Count > 0;
            if (st.Board.Controller(cell) != st.ActivePlayer)
            {
                if (st.Config.ClaimEndsTurn && actedAlready)
                    return "Can't claim — your army already acted this turn";
                return "Click this hex to CLAIM it (ends your turn)";
            }
            if (!HasGeneratorOn(st, cell))
                return "Click this hex to BUILD a generator";
            return "";
        }
    }
}
