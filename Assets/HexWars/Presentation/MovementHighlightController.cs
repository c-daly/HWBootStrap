using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Renders engine-provided movement routes without calculating movement legality.</summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class MovementHighlightController : MonoBehaviour
    {
        readonly List<GameObject> _pool = new List<GameObject>();
        BoardRenderer _board;
        Transform _root;
        Mesh _thinRing;
        Mesh _strongRing;
        Material _reachableMaterial;
        Material _routeMaterial;
        Material _expensiveMaterial;
        Material _destinationMaterial;
        int _used;

        void Awake() => _board = GetComponent<BoardRenderer>();

        public void Show(GameState state,
                         IReadOnlyDictionary<HexCoord, MovementRoute> routes,
                         HexCoord? previewDestination)
        {
            EnsureResources();
            Clear();

            var destinations = new List<HexCoord>(routes.Keys);
            destinations.Sort(CompareCoords);
            foreach (var destination in destinations)
                AddRing(state, destination, MovementHighlightKind.Reachable);

            if (!previewDestination.HasValue
                || !routes.TryGetValue(previewDestination.Value, out var preview))
                return;

            foreach (var cell in preview.Cells)
            {
                var kind = MovementHighlightClassifier.Classify(
                    state, preview, cell, routes.ContainsKey(cell));
                AddRing(state, cell, kind);
            }
        }

        public void Clear()
        {
            for (int i = 0; i < _pool.Count; i++)
                if (_pool[i] != null) _pool[i].SetActive(false);
            _used = 0;
        }

        void EnsureResources()
        {
            if (_board == null) _board = GetComponent<BoardRenderer>();
            if (_root == null)
            {
                var existing = transform.Find("MovementHighlights");
                if (existing != null) _root = existing;
                else
                {
                    var root = new GameObject("MovementHighlights");
                    root.transform.SetParent(transform, false);
                    _root = root.transform;
                }
            }

            float radius = _board.HexSize * _board.ColumnRadiusFactor;
            if (_thinRing == null) _thinRing = HexMesh.Ring(radius * 0.94f, radius * 0.89f);
            if (_strongRing == null) _strongRing = HexMesh.Ring(radius * 0.91f, radius * 0.78f);
            if (_reachableMaterial == null)
            {
                _reachableMaterial = CreateMaterial(new Color(0.20f, 0.78f, 0.30f));
                _routeMaterial = CreateMaterial(new Color(1.00f, 0.88f, 0.20f));
                _expensiveMaterial = CreateMaterial(new Color(1.00f, 0.58f, 0.08f));
                _destinationMaterial = CreateMaterial(new Color(1.00f, 1.00f, 0.88f));
            }
        }

        Material CreateMaterial(Color color)
        {
            var shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null) shader = Shader.Find("Unlit/Color");
            var material = new Material(shader);
            if (material.HasProperty("_BaseColor")) material.SetColor("_BaseColor", color);
            material.color = color;
            if (material.HasProperty("_Cull")) material.SetFloat("_Cull", 0f);
            return material;
        }

        void AddRing(GameState state, HexCoord cell, MovementHighlightKind kind)
        {
            GameObject ring;
            if (_used < _pool.Count) ring = _pool[_used];
            else
            {
                ring = new GameObject("MovementHighlight");
                ring.transform.SetParent(_root, false);
                ring.AddComponent<MeshFilter>();
                var renderer = ring.AddComponent<MeshRenderer>();
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                _pool.Add(ring);
            }
            _used++;

            ring.name = "MovementHighlight_" + kind;
            var filter = ring.GetComponent<MeshFilter>();
            var meshRenderer = ring.GetComponent<MeshRenderer>();
            filter.sharedMesh = kind == MovementHighlightKind.Reachable ? _thinRing : _strongRing;
            meshRenderer.sharedMaterial = MaterialFor(kind);

            var world = HexLayout.ToWorld(cell, _board.HexSize);
            float top = (state.Board.TileAt(cell).Elevation + 1) * _board.LevelHeight;
            float lift = kind == MovementHighlightKind.Reachable ? 0.035f
                       : kind == MovementHighlightKind.Destination ? 0.065f
                       : 0.050f;
            ring.transform.localPosition = new Vector3((float)world.x, top + lift, (float)world.z);
            ring.SetActive(true);
        }

        Material MaterialFor(MovementHighlightKind kind)
        {
            switch (kind)
            {
                case MovementHighlightKind.Expensive: return _expensiveMaterial;
                case MovementHighlightKind.Destination: return _destinationMaterial;
                case MovementHighlightKind.Route: return _routeMaterial;
                default: return _reachableMaterial;
            }
        }

        static int CompareCoords(HexCoord a, HexCoord b)
        {
            int comparison = a.Q.CompareTo(b.Q);
            return comparison != 0 ? comparison : a.R.CompareTo(b.R);
        }

        void OnDestroy()
        {
            if (_reachableMaterial != null) Destroy(_reachableMaterial);
            if (_routeMaterial != null) Destroy(_routeMaterial);
            if (_expensiveMaterial != null) Destroy(_expensiveMaterial);
            if (_destinationMaterial != null) Destroy(_destinationMaterial);
        }
    }
}
