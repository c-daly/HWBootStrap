using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Renders compact circular halos around targetable enemy units.</summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class AttackTargetHighlightController : MonoBehaviour
    {
        readonly List<GameObject> _pool = new List<GameObject>();
        BoardRenderer _board;
        Transform _root;
        Mesh _haloMesh;
        Material _haloMaterial;
        int _used;

        void Awake() => _board = GetComponent<BoardRenderer>();

        public void Show(IReadOnlyList<AttackPreviewTarget> targets)
        {
            EnsureResources();
            Clear();
            for (int i = 0; i < targets.Count; i++)
                AddHalo(targets[i]);
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
                var existing = transform.Find("AttackTargetHighlights");
                if (existing != null) _root = existing;
                else
                {
                    var root = new GameObject("AttackTargetHighlights");
                    root.transform.SetParent(transform, false);
                    _root = root.transform;
                }
            }

            if (_haloMesh == null)
            {
                float radius = _board.HexSize * 0.60f;
                _haloMesh = CircleRing(radius, radius * 0.72f, 40);
            }
            if (_haloMaterial == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                _haloMaterial = new Material(shader);
                var blue = new Color(0.10f, 0.68f, 1.00f);
                if (_haloMaterial.HasProperty("_BaseColor"))
                    _haloMaterial.SetColor("_BaseColor", blue);
                _haloMaterial.color = blue;
                if (_haloMaterial.HasProperty("_Cull")) _haloMaterial.SetFloat("_Cull", 0f);
            }
        }

        void AddHalo(AttackPreviewTarget target)
        {
            GameObject halo;
            if (_used < _pool.Count) halo = _pool[_used];
            else
            {
                halo = new GameObject("AttackTarget");
                halo.transform.SetParent(_root, false);
                halo.AddComponent<MeshFilter>();
                var renderer = halo.AddComponent<MeshRenderer>();
                renderer.shadowCastingMode = ShadowCastingMode.Off;
                renderer.receiveShadows = false;
                _pool.Add(halo);
            }
            _used++;

            halo.name = "AttackTarget_" + target.UnitId;
            halo.GetComponent<MeshFilter>().sharedMesh = _haloMesh;
            halo.GetComponent<MeshRenderer>().sharedMaterial = _haloMaterial;
            var world = HexLayout.ToWorld(target.Cell, _board.HexSize);
            float top = (target.Elevation + 1) * _board.LevelHeight;
            halo.transform.localPosition = new Vector3((float)world.x, top + 0.72f, (float)world.z);
            halo.SetActive(true);
        }

        static Mesh CircleRing(float outerRadius, float innerRadius, int segments)
        {
            var vertices = new Vector3[segments * 2];
            var triangles = new int[segments * 6];
            for (int i = 0; i < segments; i++)
            {
                float angle = Mathf.PI * 2f * i / segments;
                float x = Mathf.Cos(angle);
                float z = Mathf.Sin(angle);
                vertices[i * 2] = new Vector3(x * outerRadius, 0f, z * outerRadius);
                vertices[i * 2 + 1] = new Vector3(x * innerRadius, 0f, z * innerRadius);

                int next = (i + 1) % segments;
                int triangle = i * 6;
                triangles[triangle] = i * 2;
                triangles[triangle + 1] = next * 2;
                triangles[triangle + 2] = i * 2 + 1;
                triangles[triangle + 3] = i * 2 + 1;
                triangles[triangle + 4] = next * 2;
                triangles[triangle + 5] = next * 2 + 1;
            }

            var mesh = new Mesh { name = "AttackTargetHalo" };
            mesh.vertices = vertices;
            mesh.triangles = triangles;
            mesh.RecalculateBounds();
            return mesh;
        }

        void OnDestroy()
        {
            if (_haloMesh != null) Destroy(_haloMesh);
            if (_haloMaterial != null) Destroy(_haloMaterial);
        }
    }
}
