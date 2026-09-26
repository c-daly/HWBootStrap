using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Renders readable target brackets and labels before an enemy is selected.</summary>
    [RequireComponent(typeof(BoardRenderer))]
    public sealed class AttackTargetHighlightController : MonoBehaviour
    {
        readonly List<GameObject> _pool = new List<GameObject>();
        BoardRenderer _board;
        Transform _root;
        Mesh _haloMesh;
        Material _haloMaterial;
        int _used;
        LineRenderer _shot;

        void Awake() => _board = GetComponent<BoardRenderer>();

        public void Show(IReadOnlyList<AttackPreviewTarget> targets, bool afterMove = false)
        {
            EnsureResources();
            Clear();
            for (int i = 0; i < targets.Count; i++)
                AddHalo(targets[i], afterMove);
        }

        public void ShowAvailable(IReadOnlyList<AttackPreviewTarget> current, IReadOnlyList<AttackPreviewTarget> afterMove)
        {
            Show(current);
            if (afterMove == null) return;
            foreach (var target in afterMove)
            {
                bool alreadyShown = false;
                foreach (var available in current) if (available.UnitId == target.UnitId) { alreadyShown = true; break; }
                if (!alreadyShown) AddHalo(target, true);
            }
        }

        public void Clear()
        {
            for (int i = 0; i < _pool.Count; i++)
                if (_pool[i] != null) _pool[i].SetActive(false);
            _used = 0;
            if (_shot != null) _shot.gameObject.SetActive(false);
        }

        public void ShowShot(Unit from,Unit target,bool direct)
        {
            EnsureResources();
            if(_shot==null)
            {
                var go=new GameObject("Attack preview line");go.transform.SetParent(_root,false);_shot=go.AddComponent<LineRenderer>();
                _shot.sharedMaterial=_haloMaterial;_shot.useWorldSpace=false;_shot.startWidth=.025f;_shot.endWidth=.025f;
                _shot.shadowCastingMode=ShadowCastingMode.Off;
            }
            Vector3 Point(Unit u){var p=HexLayout.ToWorld(u.Cell,_board.HexSize);return new Vector3((float)p.x,(u.Elevation+1)*_board.LevelHeight+.50f,(float)p.z);}
            var a=Point(from);var b=Point(target);_shot.positionCount=25;
            float arc=direct?.25f:Mathf.Max(2.5f,Vector3.Distance(a,b)*.35f);
            for(int i=0;i<25;i++){float t=i/24f;_shot.SetPosition(i,Vector3.Lerp(a,b,t)+Vector3.up*(Mathf.Sin(t*Mathf.PI)*arc));}
            _shot.gameObject.SetActive(true);
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
                float radius = _board.HexSize * 0.85f;
                _haloMesh = CircleRing(radius, radius * 0.87f, 40);
            }
            if (_haloMaterial == null)
            {
                var shader = Shader.Find("Universal Render Pipeline/Unlit");
                if (shader == null) shader = Shader.Find("Unlit/Color");
                _haloMaterial = new Material(shader);
                var blue = new Color(1f, .90f, .61f); // Distinct from either team hull.
                if (_haloMaterial.HasProperty("_BaseColor"))
                    _haloMaterial.SetColor("_BaseColor", blue);
                _haloMaterial.color = blue;
                if (_haloMaterial.HasProperty("_Cull")) _haloMaterial.SetFloat("_Cull", 0f);
            }
        }

        void AddHalo(AttackPreviewTarget target, bool afterMove)
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
                var badge = new GameObject("Range marker");
                badge.transform.SetParent(halo.transform, false);
                badge.transform.localPosition = new Vector3(0, _board.HexSize * 1.86f, 0);
                badge.AddComponent<Billboard>();
                var label = badge.AddComponent<TextMesh>();
                label.font = UiKit.Font(); label.fontSize = 48; label.characterSize = .065f * _board.HexSize;
                label.anchor = TextAnchor.MiddleCenter; label.color = new Color(1f, .94f, .76f);
                badge.GetComponent<MeshRenderer>().sharedMaterial = label.font.material;
                _pool.Add(halo);
            }
            _used++;

            halo.name = "AttackTarget_" + target.UnitId;
            halo.GetComponent<MeshFilter>().sharedMesh = _haloMesh;
            halo.GetComponent<MeshRenderer>().sharedMaterial = _haloMaterial;
            var world = HexLayout.ToWorld(target.Cell, _board.HexSize);
            float top = (target.Elevation + 1) * _board.LevelHeight;
            halo.transform.localPosition = new Vector3((float)world.x, top + 0.075f, (float)world.z);
            var text = halo.GetComponentInChildren<TextMesh>(true);
            text.text = afterMove ? "AFTER MOVE" : "IN RANGE";
            text.color = afterMove ? new Color(.70f,.79f,.78f) : new Color(1f,.94f,.76f);
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

                // Four separated arcs carry target identity without covering the machine.
                if (i % 10 >= 5) continue;
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
