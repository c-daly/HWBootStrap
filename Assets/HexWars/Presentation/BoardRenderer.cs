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
        Material _fogMarkCell, _p0Fog, _p1Fog;
        readonly Dictionary<UnitRole, Material> _iconMats = new Dictionary<UnitRole, Material>();
        readonly Dictionary<(TerrainType, PlayerId), Material> _controlMats = new Dictionary<(TerrainType, PlayerId), Material>();
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
            var store = GetComponent<TokenStore>();
            if (store != null) store.Clear();
            ClearChild("Columns");
            ClearChild("Control");
            var columns = ChildRoot("Columns");
            foreach (var tile in board.Tiles)
                BuildColumn(columns.transform, tile);
        }

        // ---- units / generators ----

        /// <summary>Snap all entities to <paramref name="state"/> (facade over TokenStore; kept so
        /// replay/duel/bootstrap call sites are unchanged). With fog and a viewer, the other army's
        /// entities exist only where the viewer's army has vision.</summary>
        public void RenderEntities(GameState state, PlayerId? viewer = null)
        {
            EnsureMaterials();
            var store = GetComponent<TokenStore>();
            if (store == null) store = gameObject.AddComponent<TokenStore>();
            store.Sync(state, viewer);
            UpdateControlTint(state);
        }

        /// <summary>Tint controlled hexes toward their owner's colour (in-place material swap on the
        /// tile fill; rides every entity sync and each presenter action commit).</summary>
        public void UpdateControlTint(GameState state)
        {
            var cols = transform.Find("Columns");
            if (cols == null) return;
            foreach (Transform col in cols)
            {
                var tv = col.GetComponent<TileView>();
                var fillT = col.Find("Fill");
                if (tv == null || fillT == null) continue;
                var terrain = state.Board.TileAt(tv.Coord).Terrain;
                var owner = state.Board.Controller(tv.Coord);
                fillT.GetComponent<MeshRenderer>().sharedMaterial =
                    owner == null ? MaterialFor(terrain) : ControlTintMaterial(terrain, owner.Value);
            }
        }

        /// <summary>Spec §"Fog-of-War Indicator" (amended 2026-07-25): shade every cell in
        /// <paramref name="markedCells"/> — everything outside the acting player's current visibility —
        /// with a translucent overlay sitting above each column's tile fill, and dim the units standing
        /// in those cells. Deliberately a SEPARATE overlay object per column rather than another swap of
        /// the "Fill" renderer's material: <see cref="UpdateControlTint"/> already owns that swap
        /// (terrain material vs. owner-tinted material) and layering a second, independent concern onto
        /// the same renderer would make the two features fight over one material slot. An empty/null
        /// <paramref name="markedCells"/> hides every marking (fog off in config, or the viewer's toggle
        /// off) — the units underneath are untouched either way; they were never hidden, only (un)dimmed.</summary>
        public void UpdateFogMarking(IReadOnlyCollection<HexCoord> markedCells)
        {
            var cols = transform.Find("Columns");
            if (cols == null) return;
            HashSet<HexCoord> marked = markedCells == null || markedCells.Count == 0
                ? null
                : (markedCells as HashSet<HexCoord> ?? new HashSet<HexCoord>(markedCells));
            foreach (Transform col in cols)
            {
                var tv = col.GetComponent<TileView>();
                var mark = col.Find("FogMark");
                if (tv == null || mark == null) continue;
                mark.gameObject.SetActive(marked != null && marked.Contains(tv.Coord));
            }
            GetComponent<TokenStore>()?.ApplyFogDimming(marked);
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
            var tex = RoleIcons.For(role);

            // Preferred: custom HexWars/IconUnlit bakes alpha blending — no keyword variant for WebGL to strip.
            var iconShader = Shader.Find("HexWars/IconUnlit");
            if (iconShader != null)
            {
                m = new Material(iconShader);
                m.SetTexture("_MainTex", tex);
                _iconMats[role] = m;
                return m;
            }

            // Fallback (editor/desktop): URP/Unlit switched to transparent mode via keyword.
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Texture");
            m = new Material(unlit);
            if (m.HasProperty("_BaseMap")) m.SetTexture("_BaseMap", tex);
            if (m.HasProperty("_MainTex")) m.SetTexture("_MainTex", tex);
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f); // double-sided
            m.SetFloat("_Surface", 1f);
            m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)RenderQueue.Transparent;
            _iconMats[role] = m;
            return m;
        }

        // tint a controlled hex toward its owner's colour, keeping the matcap metal look (opaque → WebGL-safe)
        Material ControlTintMaterial(TerrainType terrain, PlayerId owner)
        {
            var key = (terrain, owner);
            if (_controlMats.TryGetValue(key, out var m)) return m;
            var baseMat = MaterialFor(terrain);
            m = new Material(baseMat); // clone: same matcap shader + texture
            Color baseC = baseMat.HasProperty("_BaseColor") ? baseMat.GetColor("_BaseColor") : baseMat.color;
            Color ownerC = owner == PlayerId.Player0 ? new Color(0.27f, 0.68f, 1f) : new Color(0.92f, 0.28f, 0.28f);
            Color tint = Color.Lerp(baseC, ownerC, 0.6f);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", tint);
            m.color = tint;
            return _controlMats[key] = m;
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

            // fog-marking overlay: a thin translucent cap just above the fill top, independent of the
            // Fill renderer's own material (see UpdateFogMarking) — starts hidden; UpdateFogMarking
            // turns it on per cell.
            var mark = new GameObject("FogMark");
            mark.transform.SetParent(col.transform, false);
            mark.transform.localPosition = new Vector3(0f, htot + 0.01f, 0f);
            mark.AddComponent<MeshFilter>().sharedMesh = HexMesh.Prism(R * 0.995f, 0.02f);
            var markRenderer = mark.AddComponent<MeshRenderer>();
            markRenderer.sharedMaterial = FogMarkCellMaterial();
            markRenderer.shadowCastingMode = ShadowCastingMode.Off;
            mark.SetActive(false);

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

        internal Material UnitMat(PlayerId owner, bool dim) =>
            owner == PlayerId.Player0 ? (dim ? _p0Dim : _p0) : (dim ? _p1Dim : _p1);
        internal Material IconMatFor(UnitRole role) => IconMaterial(role);
        internal Material BlackMat => _black;
        internal Material UnlitColorMat(Color c) => UnlitColor(c);

        /// <summary>Translucent dark cap shared by every "FogMark" cell overlay (spec §"Fog-of-War
        /// Indicator"). Same alpha-blend recipe as <see cref="IconMaterial"/>'s fallback path — proven
        /// WebGL-safe transparent unlit — just without a texture, so it reads as a flat shade.</summary>
        Material FogMarkCellMaterial()
        {
            if (_fogMarkCell != null) return _fogMarkCell;
            _fogMarkCell = TransparentUnlit(new Color(0.02f, 0.02f, 0.04f, 0.6f));
            return _fogMarkCell;
        }

        /// <summary>Translucent per-owner cap for a unit token standing in a marked cell — visually
        /// distinct from <see cref="UnitMat"/>'s spent/inactive dim (a different material swapped onto
        /// the disc itself) so an operator never confuses "not this player's turn" with "the acting
        /// model can't see this."</summary>
        internal Material FogUnitMat(PlayerId owner)
        {
            if (owner == PlayerId.Player0) return _p0Fog ?? (_p0Fog = TransparentUnlit(new Color(0.05f, 0.09f, 0.14f, 0.72f)));
            return _p1Fog ?? (_p1Fog = TransparentUnlit(new Color(0.14f, 0.06f, 0.06f, 0.72f)));
        }

        Material TransparentUnlit(Color c)
        {
            var unlit = Shader.Find("Universal Render Pipeline/Unlit");
            if (unlit == null) unlit = Shader.Find("Unlit/Color");
            var m = new Material(unlit);
            if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
            m.color = c;
            if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
            m.SetFloat("_Surface", 1f);
            m.SetFloat("_SrcBlend", (float)BlendMode.SrcAlpha);
            m.SetFloat("_DstBlend", (float)BlendMode.OneMinusSrcAlpha);
            m.SetFloat("_ZWrite", 0f);
            m.EnableKeyword("_SURFACE_TYPE_TRANSPARENT");
            m.renderQueue = (int)RenderQueue.Transparent;
            return m;
        }

        internal void EnsureMaterials()
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
