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
        LineRenderer _routeLine;

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
            {
                AddRing(state, destination, MovementHighlightKind.Reachable);
                // Large maps retain all outlines; avoid hundreds of unnecessary text meshes.
                if (destinations.Count <= 100)
                {
                    var ring = _pool[_used - 1];
                    var label = ring.GetComponentInChildren<TextMesh>(true);
                    if (label == null)
                    {
                        var go = new GameObject("Movement cost");go.transform.SetParent(ring.transform,false);
                        go.transform.localPosition = new Vector3(0,.14f,0);
                        label = go.AddComponent<TextMesh>();label.font=UiKit.Font();label.fontSize=48;
                        label.characterSize=.075f;label.anchor=TextAnchor.MiddleCenter;label.color=new Color(.8f,.98f,.9f);
                        go.GetComponent<MeshRenderer>().sharedMaterial=label.font.material;go.AddComponent<Billboard>();
                    }
                    label.text=routes[destination].HorizontalCost.ToString();label.gameObject.SetActive(true);
                }
            }

            if (!previewDestination.HasValue
                || !routes.TryGetValue(previewDestination.Value, out var preview))
                return;

            _routeLine.gameObject.SetActive(true);
            _routeLine.positionCount = preview.Cells.Count;
            for (int i=0;i<preview.Cells.Count;i++)
            {
                var c=preview.Cells[i];var w=HexLayout.ToWorld(c,_board.HexSize);
                _routeLine.SetPosition(i,new Vector3((float)w.x,(state.Board.TileAt(c).Elevation+1)*_board.LevelHeight+.11f,(float)w.z));
            }
            foreach (var cell in preview.Cells)
            {
                var kind = MovementHighlightClassifier.Classify(
                    state, preview, cell, routes.ContainsKey(cell));
                AddRing(state, cell, kind);
            }
        }

        public void ShowPlacement(GameState state, IReadOnlyList<HexCoord> cells, HexCoord? destination)
        {
            EnsureResources(); Clear();
            foreach (var cell in cells)
                AddRing(state, cell, destination == cell ? MovementHighlightKind.Destination : MovementHighlightKind.Reachable);
        }

        public void Clear()
        {
            for (int i = 0; i < _pool.Count; i++)
                if (_pool[i] != null) _pool[i].SetActive(false);
            _used = 0;
            if (_routeLine != null) _routeLine.gameObject.SetActive(false);
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
                _reachableMaterial = CreateMaterial(new Color(.40f, .72f, .63f));
                _routeMaterial = CreateMaterial(new Color(.69f, .94f, .83f));
                _expensiveMaterial = CreateMaterial(new Color(.35f, .63f, .57f));
                _destinationMaterial = CreateMaterial(new Color(1.00f, 1.00f, 0.88f));
            }
            if (_routeLine == null)
            {
                var go=new GameObject("Movement route");go.transform.SetParent(transform,false);
                _routeLine=go.AddComponent<LineRenderer>();_routeLine.useWorldSpace=false;
                _routeLine.startWidth=_routeLine.endWidth=.045f;_routeLine.sharedMaterial=_routeMaterial;
                _routeLine.shadowCastingMode=ShadowCastingMode.Off;
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

            var oldCost = ring.GetComponentInChildren<TextMesh>(true);
            if (oldCost != null) oldCost.gameObject.SetActive(false);
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
