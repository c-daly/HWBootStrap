using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Persistent tokens for units and generators, keyed by engine id. Sync() diffs the given
    /// state against what exists: spawns the missing, destroys the gone, snaps survivors to
    /// truth (position, HP bar, spent-dim material, UnitView payload). Replaces BoardRenderer's
    /// destroy-and-rebuild so the ActionPresenter can tween tokens between states.
    /// </summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class TokenStore : MonoBehaviour
    {
        BoardRenderer _board;
        Transform _root;
        readonly Dictionary<int, GameObject> _units = new Dictionary<int, GameObject>();
        readonly Dictionary<int, GameObject> _generators = new Dictionary<int, GameObject>();

        void Awake() => _board = GetComponent<BoardRenderer>();

        Transform Root()
        {
            if (_root == null)
            {
                var go = new GameObject("Tokens");
                go.transform.SetParent(transform, false);
                _root = go.transform;
            }
            return _root;
        }

        public GameObject UnitToken(int unitId) => _units.TryGetValue(unitId, out var go) && go != null ? go : null;

        public Vector3 CellTop(HexCoord cell, int elevation)
        {
            var w = HexLayout.ToWorld(cell, _board.HexSize);
            return new Vector3((float)w.x, (elevation + 1) * _board.LevelHeight, (float)w.z);
        }

        public void Clear()
        {
            foreach (var go in _units.Values) if (go != null) Destroy(go);
            foreach (var go in _generators.Values) if (go != null) Destroy(go);
            _units.Clear();
            _generators.Clear();
        }

        /// <summary>Snap the world to <paramref name="state"/>. With fog and a viewer, the other
        /// army exists only where the viewer's army has vision (same rule as targeting).</summary>
        public void Sync(GameState state, PlayerId? viewer)
        {
            _board.EnsureMaterials();
            bool fog = viewer.HasValue && state.Config.FogOfWar;
            var liveUnits = new HashSet<int>();
            var liveGens = new HashSet<int>();

            foreach (var player in state.Players)
            {
                bool isActive = player.Id == state.ActivePlayer;
                bool hideUnseen = fog && player.Id != viewer.Value;

                foreach (var u in player.UnitsOnBoard)
                {
                    if (!u.IsAlive) continue;
                    if (hideUnseen && !TargetingService.IsVisibleToArmy(state, viewer.Value, u.Cell, u.Elevation))
                        continue;
                    liveUnits.Add(u.Id);
                    bool spent = isActive && Contains(state.MovedUnitIds, u.Id) && Contains(state.AttackedUnitIds, u.Id);
                    var mat = _board.UnitMat(player.Id, !isActive || spent);
                    if (!_units.TryGetValue(u.Id, out var token) || token == null)
                        _units[u.Id] = token = BuildToken(u);
                    RefreshToken(token, u, mat);
                }

                foreach (var g in player.Generators)
                {
                    if (!g.IsAlive) continue;
                    if (hideUnseen && !TargetingService.IsVisibleToArmy(state, viewer.Value, g.Cell, g.Elevation))
                        continue;
                    liveGens.Add(g.Id);
                    if (!_generators.TryGetValue(g.Id, out var pylon) || pylon == null)
                        _generators[g.Id] = pylon = BuildPylon(g);
                    pylon.GetComponent<MeshRenderer>().sharedMaterial = _board.UnitMat(player.Id, !isActive);
                }
            }

            Prune(_units, liveUnits);
            Prune(_generators, liveGens);
        }

        static void Prune(Dictionary<int, GameObject> map, HashSet<int> live)
        {
            List<int> dead = null;
            foreach (var kv in map)
                if (!live.Contains(kv.Key))
                {
                    (dead = dead ?? new List<int>()).Add(kv.Key);
                    if (kv.Value != null) Destroy(kv.Value);
                }
            if (dead != null) foreach (var id in dead) map.Remove(id);
        }

        static bool Contains(IReadOnlyCollection<int> ids, int id)
        {
            foreach (var i in ids) if (i == id) return true;
            return false;
        }

        // ---- construction (visuals identical to the old BoardRenderer builders) ----

        GameObject BuildToken(Unit unit)
        {
            float sizeFactor = Mathf.Clamp(0.6f + unit.Stats.PointCost * 0.04f, 0.6f, 1.4f);
            float radius = _board.HexSize * 0.7f * sizeFactor;

            var token = new GameObject("Unit_" + unit.Id);
            token.transform.SetParent(Root(), false);
            token.AddComponent<UnitView>();
            var box = token.AddComponent<BoxCollider>();
            box.center = new Vector3(0f, 0.35f, 0f);
            box.size = new Vector3(_board.HexSize * 1.3f, 0.9f, _board.HexSize * 1.3f);

            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Disc";
            DestroyImmediate(disc.GetComponent<Collider>());
            disc.transform.SetParent(token.transform, false);
            disc.transform.localPosition = new Vector3(0f, 0.18f, 0f);
            disc.transform.localScale = new Vector3(radius, 0.16f, radius);
            AddHull(disc, 1.16f, 1.05f);

            var icon = GameObject.CreatePrimitive(PrimitiveType.Quad);
            icon.name = "RoleIcon";
            DestroyImmediate(icon.GetComponent<Collider>());
            icon.transform.SetParent(token.transform, false);
            icon.transform.localPosition = new Vector3(0f, 0.345f, 0f);
            icon.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            icon.transform.localScale = Vector3.one * (radius * 0.9f);
            var mr = icon.GetComponent<MeshRenderer>();
            mr.sharedMaterial = _board.IconMatFor(Roles.Dominant(unit.Stats));
            mr.shadowCastingMode = ShadowCastingMode.Off;

            var bar = new GameObject("HpBar");
            bar.transform.SetParent(token.transform, false);
            bar.transform.localPosition = new Vector3(0f, 0.62f, 0f);
            bar.AddComponent<Billboard>();
            return token;
        }

        void RefreshToken(GameObject token, Unit unit, Material discMat)
        {
            token.transform.localPosition = CellTop(unit.Cell, unit.Elevation);
            token.transform.localScale = Vector3.one; // a fast-forward can kill a squash/pop tween mid-scale
            token.GetComponent<UnitView>().Unit = unit; // engine states are immutable: re-point every sync
            token.transform.Find("Disc").GetComponent<MeshRenderer>().sharedMaterial = discMat;
            RefreshHpBar(token.transform.Find("HpBar"), unit.CurrentHp, unit.Stats.Health);
        }

        void RefreshHpBar(Transform bar, int cur, int max)
        {
            for (int i = bar.childCount - 1; i >= 0; i--) Destroy(bar.GetChild(i).gameObject);
            float frac = max <= 0 ? 0f : Mathf.Clamp01((float)cur / max);
            float barW = _board.HexSize * 0.85f;
            MakeBarQuad(bar, 0f, 0f, barW, 0.16f, new Color(0.18f, 0.03f, 0.03f));
            float fw = Mathf.Max(0.001f, barW * frac);
            float fx = -barW * 0.5f + fw * 0.5f;
            var fill = Color.Lerp(new Color(0.85f, 0.2f, 0.12f), new Color(0.25f, 0.85f, 0.25f), frac);
            MakeBarQuad(bar, fx, -0.01f, fw, 0.11f, fill);
        }

        void MakeBarQuad(Transform parent, float x, float zTowardCam, float w, float h, Color c)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = "Bar";
            DestroyImmediate(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            q.transform.localPosition = new Vector3(x, 0f, zTowardCam);
            q.transform.localScale = new Vector3(w, h, 1f);
            var mr = q.GetComponent<MeshRenderer>();
            mr.sharedMaterial = _board.UnlitColorMat(c);
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }

        GameObject BuildPylon(Generator g)
        {
            var pylon = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pylon.name = "Generator_" + g.Id;
            DestroyImmediate(pylon.GetComponent<Collider>());
            pylon.transform.SetParent(Root(), false);
            var top = CellTop(g.Cell, g.Elevation);
            pylon.transform.localPosition = new Vector3(top.x, top.y + 0.45f, top.z);
            pylon.transform.localScale = new Vector3(_board.HexSize * 0.45f, 0.9f, _board.HexSize * 0.45f);
            AddHull(pylon, 1.12f, 1.06f);
            return pylon;
        }

        void AddHull(GameObject host, float xz, float y)
        {
            var hull = new GameObject("Outline");
            hull.transform.SetParent(host.transform, false);
            hull.transform.localScale = new Vector3(xz, y, xz);
            hull.AddComponent<MeshFilter>().sharedMesh = host.GetComponent<MeshFilter>().sharedMesh;
            var mr = hull.AddComponent<MeshRenderer>();
            var m = new Material(_board.BlackMat);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 1f);
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }
    }
}
