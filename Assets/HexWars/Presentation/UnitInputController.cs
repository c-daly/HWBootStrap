using System.Collections;
using UnityEngine;
using UnityEngine.InputSystem;
using UnityEngine.Rendering;
using HexWars.Engine;

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
        bool _animating;

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
            }
        }

        void HandleClick(UnitView unit, TileView tile)
        {
            if (_game == null) { Select(unit); return; }
            var active = _game.State.ActivePlayer;
            bool ownSelected = _selected != null && _selected.Unit.Owner == active && _selected.Unit.IsAlive;

            // attack intent: only fire if actually targetable (range + army vision + LOS/arc)
            if (ownSelected && unit != null && unit.Unit.Owner != active)
            {
                if (TargetingService.CanTarget(_game.State, _selected.Unit, unit.Unit.Cell, unit.Unit.Elevation))
                    StartCoroutine(AttackSeq(_selected, unit));
                return; // out of range / no shot: nothing happens, keep selection
            }
            // move intent: only move to a reachable hex
            if (ownSelected && unit == null && tile != null)
            {
                if (IsReachable(_selected.Unit, tile.Coord))
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
            _animating = false;
        }

        IEnumerator AttackSeq(UnitView attacker, UnitView target)
        {
            _animating = true;

            // attack VFX scales with the attacker's Damage: small fast bullet -> fat slow rocket
            int dmg = attacker.Unit.Stats.Damage;
            float power = Mathf.Clamp01(dmg / 8f);
            float projScale = Mathf.Lerp(0.14f, 0.5f, power);
            float flightDur = Mathf.Lerp(0.16f, 0.4f, power);
            Color projColor = dmg >= 6 ? new Color(1f, 0.3f, 0.1f)
                            : dmg >= 3 ? new Color(1f, 0.65f, 0.2f)
                                       : new Color(1f, 0.95f, 0.5f);

            int targetId = target.Unit.Id;
            Vector3 targetPos = target.transform.position;
            Vector3 from = attacker.transform.position + Vector3.up * 0.4f;
            Vector3 to = targetPos + Vector3.up * 0.4f;

            // direct shot flies straight; an indirect (LOS-blocked) shot lobs over the obstacles
            bool directLos = LineOfSight.IsClear(_game.State.Board,
                attacker.Unit.Cell, attacker.Unit.Elevation, target.Unit.Cell, target.Unit.Elevation);
            float arc = directLos ? 0f : Mathf.Max(2.5f, Vector3.Distance(from, to) * 0.35f);

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
                if (!IsUnitAlive(targetId))
                    ExplosionFx.Spawn(targetPos, new Color(0.95f, 0.45f, 0.18f), Mathf.Lerp(0.8f, 2.0f, power), true);
                else
                    ExplosionFx.Spawn(to, projColor, Mathf.Lerp(0.4f, 0.9f, power), false);
            }
            ReacquireSelection();
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
    }
}
