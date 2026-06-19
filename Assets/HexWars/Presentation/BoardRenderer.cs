using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Renders an engine <see cref="Board"/> as stacked, black-outlined hex columns (elevation = N
    /// stacked levels; a black collar at each level boundary + black vertical corner bars), and the
    /// on-board units/generators as tokens (cyan = Player 0, red = Player 1; units are flat discs,
    /// generators are tall pylons). Bodies are lit so the soft shadows read. Procedural for now.
    /// </summary>
    public sealed class BoardRenderer : MonoBehaviour
    {
        public float HexSize = 1f;
        public float LevelHeight = 0.55f;
        public float ColumnRadiusFactor = 0.9f;
        public float EdgeBarThickness = 0.012f;
        [Range(0f, 1f)] public float Metallic = 1f;      // hex bodies: full metal
        [Range(0f, 1f)] public float Smoothness = 0.72f; // crisp enough to read as metal, soft enough to avoid hotspots
        public bool Outlines = true;                     // black cel-style edges (off = realistic metal)

        Material _plains, _forest, _water, _rough, _black, _p0, _p1, _p0Dim, _p1Dim;
        readonly Dictionary<UnitRole, Material> _iconMats = new Dictionary<UnitRole, Material>();
        static Texture2D _matcap;

        static Texture2D MetalMatcap()
        {
            if (_matcap != null) return _matcap;
            const int N = 128;
            var tex = new Texture2D(N, N, TextureFormat.RGBA32, false)
            { wrapMode = TextureWrapMode.Clamp, filterMode = FilterMode.Bilinear };
            var cols = new Color[N * N];
            for (int y = 0; y < N; y++)
                for (int x = 0; x < N; x++)
                {
                    float u = x / (float)(N - 1), v = y / (float)(N - 1);
                    float nx = u * 2f - 1f, ny = v * 2f - 1f;
                    float r = Mathf.Sqrt(nx * nx + ny * ny);
                    // brushed steel: mid-tone so the colour tint reads, gentle top-light + soft sheen
                    float up = Mathf.Clamp01((ny + 1f) * 0.5f);
                    float body = Mathf.Lerp(0.26f, 0.80f, up);                      // more range = defined faces
                    float hx = nx + 0.30f, hy = ny - 0.45f;
                    float hl = Mathf.Exp(-(hx * hx + hy * hy) / 0.08f) * 0.45f;      // tighter sheen
                    float rim = Mathf.SmoothStep(0.84f, 1.0f, r) * 0.35f;
                    float c = Mathf.Clamp01(body + hl + rim);
                    cols[y * N + x] = new Color(c, c, c * 1.02f);
                }
            tex.SetPixels(cols);
            tex.Apply();
            _matcap = tex;
            return tex;
        }

        // ---- board ----

        public void Render(Board board)
        {
            EnsureMaterials();
            ClearChild("Columns");
            var columns = ChildRoot("Columns");
            foreach (var tile in board.Tiles)
                BuildColumn(columns.transform, tile);
        }

        // ---- units / generators ----

        public void RenderEntities(GameState state)
        {
            EnsureMaterials();
            ClearChild("Entities");
            var root = ChildRoot("Entities");

            foreach (var player in state.Players)
            {
                bool isActive = player.Id == state.ActivePlayer;
                var bright = player.Id == PlayerId.Player0 ? _p0 : _p1;
                var dim = player.Id == PlayerId.Player0 ? _p0Dim : _p1Dim;

                foreach (var u in player.UnitsOnBoard)
                {
                    if (!u.IsAlive) continue;
                    // dim if it's the opponent's, or it's yours but already spent (moved AND attacked)
                    bool spent = isActive && Contains(state.MovedUnitIds, u.Id) && Contains(state.AttackedUnitIds, u.Id);
                    BuildToken(root.transform, u, (!isActive || spent) ? dim : bright);
                }
                foreach (var g in player.Generators)
                    if (g.IsAlive) BuildPylon(root.transform, g.Cell, g.Elevation, isActive ? bright : dim);
            }
        }

        static bool Contains(System.Collections.Generic.IReadOnlyCollection<int> ids, int id)
        {
            foreach (var i in ids) if (i == id) return true;
            return false;
        }

        float TopY(int elevation) => (elevation + 1) * LevelHeight;

        void BuildToken(Transform parent, Unit unit, Material color)
        {
            var w = HexLayout.ToWorld(unit.Cell, HexSize);
            float topY = TopY(unit.Elevation);

            // size scales with total points spent (bigger build = bigger token)
            float sizeFactor = Mathf.Clamp(0.6f + unit.Stats.PointCost * 0.04f, 0.6f, 1.4f);
            float radius = HexSize * 0.7f * sizeFactor;

            // unscaled token root carries the data + a generous box collider for easy clicking
            var token = new GameObject("Unit_" + unit.Id);
            token.transform.SetParent(parent, false);
            token.transform.localPosition = new Vector3((float)w.x, topY, (float)w.z);
            token.AddComponent<UnitView>().Unit = unit;
            var box = token.AddComponent<BoxCollider>();
            box.center = new Vector3(0f, 0.35f, 0f);
            box.size = new Vector3(HexSize * 1.3f, 0.9f, HexSize * 1.3f);

            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Disc";
            DestroyImmediate(disc.GetComponent<Collider>()); // hitbox is the token's box
            disc.transform.SetParent(token.transform, false);
            disc.transform.localPosition = new Vector3(0f, 0.18f, 0f);
            disc.transform.localScale = new Vector3(radius, 0.16f, radius);
            disc.GetComponent<MeshRenderer>().sharedMaterial = color;
            AddHull(disc, 1.16f, 1.05f);

            // role icon badge, flat on top, facing up (double-sided so it always reads)
            var icon = GameObject.CreatePrimitive(PrimitiveType.Quad);
            icon.name = "RoleIcon";
            DestroyImmediate(icon.GetComponent<Collider>());
            icon.transform.SetParent(token.transform, false);
            icon.transform.localPosition = new Vector3(0f, 0.40f, 0f);
            icon.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            icon.transform.localScale = Vector3.one * (radius * 0.9f);
            var mr = icon.GetComponent<MeshRenderer>();
            mr.sharedMaterial = IconMaterial(Roles.Dominant(unit.Stats));
            mr.shadowCastingMode = ShadowCastingMode.Off;

            BuildHpBar(parent, (float)w.x, (float)w.z, topY, unit.CurrentHp, unit.Stats.Health);
        }

        void BuildHpBar(Transform parent, float wx, float wz, float topY, int cur, int max)
        {
            float frac = max <= 0 ? 0f : Mathf.Clamp01((float)cur / max);
            float barW = HexSize * 0.85f;
            float y = topY + 0.62f;
            MakeBarQuad(parent, wx, y, wz, barW, 0.15f, new Color(0.18f, 0.03f, 0.03f));
            float fw = Mathf.Max(0.001f, barW * frac);
            float fx = wx - barW * 0.5f + fw * 0.5f; // left-anchored fill
            var fill = Color.Lerp(new Color(0.85f, 0.2f, 0.12f), new Color(0.25f, 0.85f, 0.25f), frac);
            MakeBarQuad(parent, fx, y + 0.01f, wz, fw, 0.10f, fill);
        }

        void MakeBarQuad(Transform parent, float x, float y, float z, float w, float h, Color c)
        {
            var q = GameObject.CreatePrimitive(PrimitiveType.Quad);
            q.name = "HpBar";
            DestroyImmediate(q.GetComponent<Collider>());
            q.transform.SetParent(parent, false);
            q.transform.localPosition = new Vector3(x, y, z);
            q.transform.localRotation = Quaternion.Euler(90f, 0f, 0f);
            q.transform.localScale = new Vector3(w, h, 1f);
            var mr = q.GetComponent<MeshRenderer>();
            mr.sharedMaterial = UnlitColor(c);
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }

        Material UnlitColor(Color c)
        {
            var sh = Shader.Find("Universal Render Pipeline/Unlit");
            if (sh == null) sh = Shader.Find("Unlit/Color");
            var m = new Material(sh);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            m.color = c;
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
            return m;
        }

        Material IconMaterial(UnitRole role)
        {
            if (_iconMats.TryGetValue(role, out var m)) return m;
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Texture");
            m = new Material(unlit);
            var tex = RoleIcons.For(role);
            if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", tex);
            if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", tex);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f); // double-sided
            _iconMats[role] = m;
            return m;
        }

        void BuildPylon(Transform parent, HexCoord cell, int elevation, Material color)
        {
            var w = HexLayout.ToWorld(cell, HexSize);
            var pylon = GameObject.CreatePrimitive(PrimitiveType.Cube);
            pylon.name = "Generator";
            DestroyImmediate(pylon.GetComponent<Collider>());
            pylon.transform.SetParent(parent, false);
            pylon.transform.localPosition = new Vector3((float)w.x, TopY(elevation) + 0.45f, (float)w.z);
            pylon.transform.localScale = new Vector3(HexSize * 0.45f, 0.9f, HexSize * 0.45f);
            pylon.GetComponent<MeshRenderer>().sharedMaterial = color;
            AddHull(pylon, 1.12f, 1.06f);
        }

        void AddHull(GameObject host, float xz, float y)
        {
            var hull = new GameObject("Outline");
            hull.transform.SetParent(host.transform, false);
            hull.transform.localScale = new Vector3(xz, y, xz);
            hull.AddComponent<MeshFilter>().sharedMesh = host.GetComponent<MeshFilter>().sharedMesh;
            var mr = hull.AddComponent<MeshRenderer>();
            var m = new Material(_black);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 1f); // back faces only -> silhouette
            mr.sharedMaterial = m;
            mr.shadowCastingMode = ShadowCastingMode.Off;
        }

        // ---- internals ----

        void BuildColumn(Transform parent, Tile tile)
        {
            float R = HexSize * ColumnRadiusFactor;
            var w = HexLayout.ToWorld(tile.Coord, HexSize);
            int levels = tile.Elevation + 1;
            float htot = levels * LevelHeight;

            var col = new GameObject($"Hex_{tile.Coord.Q}_{tile.Coord.R}");
            col.transform.SetParent(parent, false);
            col.transform.localPosition = new Vector3((float)w.x, 0f, (float)w.z);
            col.AddComponent<TileView>().Coord = tile.Coord;
            var box = col.AddComponent<BoxCollider>();        // clickable for deploy/move targeting
            box.center = new Vector3(0f, htot * 0.5f, 0f);
            box.size = new Vector3(R * 1.6f, htot, R * 1.6f);

            var fill = new GameObject("Fill");
            fill.transform.SetParent(col.transform, false);
            fill.AddComponent<MeshFilter>().sharedMesh = HexMesh.Prism(R, htot);
            fill.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor(tile.Terrain);

            if (!Outlines) return;

            for (int i = 0; i <= levels; i++)
            {
                var collar = new GameObject("EdgeH" + i);
                collar.transform.SetParent(col.transform, false);
                collar.transform.localPosition = new Vector3(0f, i * LevelHeight, 0f);
                collar.AddComponent<MeshFilter>().sharedMesh = HexMesh.Ring(R * 1.015f, R * 0.99f);
                var mr = collar.AddComponent<MeshRenderer>();
                mr.sharedMaterial = _black;
                mr.shadowCastingMode = ShadowCastingMode.Off;
            }

            // (vertical corner bars removed — they read as protruding 'sticks'; the column
            // geometry + horizontal seams convey the edges)
        }

        GameObject ChildRoot(string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            return go;
        }

        void ClearChild(string name)
        {
            // Hide + rename immediately so no duplicate is visible (the "clone") and the next Find
            // can't return it again; then destroy (deferred in play, immediate in edit).
            var existing = transform.Find(name);
            while (existing != null)
            {
                var go = existing.gameObject;
                go.SetActive(false);
                go.name = name + "_dead";
                if (Application.isPlaying) Destroy(go); else DestroyImmediate(go);
                existing = transform.Find(name);
            }
        }

        Material MaterialFor(TerrainType t)
        {
            switch (t)
            {
                case TerrainType.Forest: return _forest;
                case TerrainType.Water: return _water;
                case TerrainType.Rough: return _rough;
                default: return _plains;
            }
        }

        void EnsureMaterials()
        {
            if (_plains != null) return;
            var lit = Shader.Find("Universal Render Pipeline/Lit");
            if (lit == null) lit = Shader.Find("Standard");
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");

            var matcapShader = Shader.Find("HexWars/Matcap");
            var matcap = MetalMatcap();

            // Hex bodies use a matcap (pre-lit metal ball mapped by view-normal) so even flat faces
            // read as shiny metal, tinted by terrain colour.
            Material Metal(Color c)
            {
                if (matcapShader != null)
                {
                    var mm = new Material(matcapShader);
                    mm.SetColor("_BaseColor", Color.Lerp(c, Color.white, 0.1f)); // keep colors saturated/sharp
                    mm.SetTexture("_Matcap", matcap);
                    return mm;
                }
                var m = new Material(lit); // fallback
                if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
                m.color = c;
                if (m.HasProperty("_Metallic")) m.SetFloat("_Metallic", Metallic);
                if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", Smoothness);
                if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
                return m;
            }
            Material Matte(Color c)
            {
                var m = new Material(lit);
                if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
                m.color = c;
                if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", 0.1f);
                if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
                return m;
            }
            _plains = Metal(new Color(1f, 0.82f, 0.10f));
            _forest = Metal(new Color(0.30f, 0.62f, 0.27f));
            _water  = Metal(new Color(0.24f, 0.58f, 0.85f));
            _rough  = Metal(new Color(0.80f, 0.71f, 0.47f));
            _p0     = Matte(new Color(0.27f, 0.68f, 1f));   // units stay matte for readability
            _p1     = Matte(new Color(0.92f, 0.28f, 0.28f));
            _p0Dim  = Matte(new Color(0.11f, 0.27f, 0.40f)); // dimmed = opponent's, or spent this turn
            _p1Dim  = Matte(new Color(0.37f, 0.11f, 0.11f));

            var seam = new Color(0.05f, 0.05f, 0.06f); // near-black panel seam (visible, defines hexes)
            _black = new Material(unlit);
            if (_black.HasProperty("_BaseColor")) _black.SetColor("_BaseColor", seam);
            _black.color = seam;
            if (_black.HasProperty("_Cull")) _black.SetFloat("_Cull", 0f);
        }
    }
}
