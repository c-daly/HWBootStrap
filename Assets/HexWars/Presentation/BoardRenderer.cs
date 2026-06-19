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
        public float EdgeBarThickness = 0.08f;

        Material _plains, _forest, _water, _rough, _black, _p0, _p1;

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
                var color = player.Id == PlayerId.Player0 ? _p0 : _p1;
                foreach (var u in player.UnitsOnBoard)
                    if (u.IsAlive) BuildToken(root.transform, u.Cell, u.Elevation, color);
                foreach (var g in player.Generators)
                    if (g.IsAlive) BuildPylon(root.transform, g.Cell, g.Elevation, color);
            }
        }

        float TopY(int elevation) => (elevation + 1) * LevelHeight;

        void BuildToken(Transform parent, HexCoord cell, int elevation, Material color)
        {
            var w = HexLayout.ToWorld(cell, HexSize);
            var disc = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            disc.name = "Unit";
            DestroyImmediate(disc.GetComponent<Collider>());
            disc.transform.SetParent(parent, false);
            disc.transform.localPosition = new Vector3((float)w.x, TopY(elevation) + 0.18f, (float)w.z);
            disc.transform.localScale = new Vector3(HexSize * 0.7f, 0.16f, HexSize * 0.7f);
            disc.GetComponent<MeshRenderer>().sharedMaterial = color;
            AddHull(disc, 1.16f, 1.05f);
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

            var fill = new GameObject("Fill");
            fill.transform.SetParent(col.transform, false);
            fill.AddComponent<MeshFilter>().sharedMesh = HexMesh.Prism(R, htot);
            fill.AddComponent<MeshRenderer>().sharedMaterial = MaterialFor(tile.Terrain);

            for (int i = 0; i <= levels; i++)
            {
                var collar = new GameObject("EdgeH" + i);
                collar.transform.SetParent(col.transform, false);
                collar.transform.localPosition = new Vector3(0f, i * LevelHeight, 0f);
                collar.AddComponent<MeshFilter>().sharedMesh = HexMesh.Ring(R * 1.07f, R * 0.93f);
                var mr = collar.AddComponent<MeshRenderer>();
                mr.sharedMaterial = _black;
                mr.shadowCastingMode = ShadowCastingMode.Off;
            }

            for (int k = 0; k < 6; k++)
            {
                float a = Mathf.Deg2Rad * (60f * k);
                var bar = GameObject.CreatePrimitive(PrimitiveType.Cube);
                bar.name = "EdgeV" + k;
                DestroyImmediate(bar.GetComponent<Collider>());
                bar.transform.SetParent(col.transform, false);
                bar.transform.localPosition = new Vector3(R * Mathf.Cos(a), htot * 0.5f, R * Mathf.Sin(a));
                bar.transform.localScale = new Vector3(EdgeBarThickness, htot, EdgeBarThickness);
                var mr = bar.GetComponent<MeshRenderer>();
                mr.sharedMaterial = _black;
                mr.shadowCastingMode = ShadowCastingMode.Off;
            }
        }

        GameObject ChildRoot(string name)
        {
            var go = new GameObject(name);
            go.transform.SetParent(transform, false);
            return go;
        }

        void ClearChild(string name)
        {
            var existing = transform.Find(name);
            if (existing == null) return;
            if (Application.isPlaying) Destroy(existing.gameObject);
            else DestroyImmediate(existing.gameObject);
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

            Material Body(Color c)
            {
                var m = new Material(lit);
                if (m.HasProperty("_BaseColor")) m.SetColor("_BaseColor", c);
                m.color = c;
                if (m.HasProperty("_Smoothness")) m.SetFloat("_Smoothness", 0.08f);
                if (m.HasProperty("_Cull")) m.SetFloat("_Cull", 0f);
                return m;
            }
            _plains = Body(new Color(1f, 0.82f, 0.10f));
            _forest = Body(new Color(0.30f, 0.62f, 0.27f));
            _water  = Body(new Color(0.24f, 0.58f, 0.85f));
            _rough  = Body(new Color(0.80f, 0.71f, 0.47f));
            _p0     = Body(new Color(0.27f, 0.68f, 1f));
            _p1     = Body(new Color(0.92f, 0.28f, 0.28f));

            _black = new Material(unlit);
            if (_black.HasProperty("_BaseColor")) _black.SetColor("_BaseColor", Color.black);
            _black.color = Color.black;
            if (_black.HasProperty("_Cull")) _black.SetFloat("_Cull", 0f);
        }
    }
}
