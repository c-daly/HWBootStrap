using UnityEngine;
using UnityEngine.Rendering;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Renders an engine <see cref="Board"/> as stacked, black-outlined hex columns (one column per
    /// tile; elevation = number of stacked levels). Bodies are lit (so soft shadows read); every
    /// horizontal level boundary gets a black collar and every vertical corner a black bar, so the
    /// stacking is visible. Procedural geometry for now — a shared prefab + outline shader can
    /// replace the internals later without changing this interface.
    /// </summary>
    public sealed class BoardRenderer : MonoBehaviour
    {
        public float HexSize = 1f;
        public float LevelHeight = 0.55f;
        public float ColumnRadiusFactor = 0.9f;
        public float EdgeBarThickness = 0.08f;

        Material _plains, _forest, _water, _rough, _black;

        public void Render(Board board)
        {
            EnsureMaterials();
            Clear();
            foreach (var tile in board.Tiles)
                BuildColumn(tile);
        }

        public void Clear()
        {
            for (int i = transform.childCount - 1; i >= 0; i--)
            {
                var go = transform.GetChild(i).gameObject;
                if (Application.isPlaying) Destroy(go); else DestroyImmediate(go);
            }
        }

        void BuildColumn(Tile tile)
        {
            float R = HexSize * ColumnRadiusFactor;
            var w = HexLayout.ToWorld(tile.Coord, HexSize);
            int levels = tile.Elevation + 1;
            float htot = levels * LevelHeight;

            var col = new GameObject($"Hex_{tile.Coord.Q}_{tile.Coord.R}");
            col.transform.SetParent(transform, false);
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

            _black = new Material(unlit);
            if (_black.HasProperty("_BaseColor")) _black.SetColor("_BaseColor", Color.black);
            _black.color = Color.black;
            if (_black.HasProperty("_Cull")) _black.SetFloat("_Cull", 0f);
        }
    }
}
