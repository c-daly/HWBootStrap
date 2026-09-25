using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Eight original, cached sculpted forms. Art never changes a unit's picking or rules.</summary>
    public static class GraphitePieces
    {
        static readonly Mesh[] Bodies = new Mesh[8];
        static readonly RenderTexture[,] Portraits = new RenderTexture[2, 8];
        static int _lastPortraitFrame = -1;
        static Mesh _foot, _rim, _brokenRim;
        static Material _graphite, _base, _mint, _amber;

        static Material Material(ref Material cache, Color color)
        {
            if (cache != null) return cache;
            cache = new Material(Shader.Find("HexWars/Matcap")) { name = "Graphite satin" };
            cache.SetColor("_BaseColor", color);
            cache.SetTexture("_Matcap", BoardRenderer.MetalMatcap());
            cache.SetFloat("_Sheen", .035f);
            return cache;
        }
        public static Color Mint => new Color32(128, 217, 196, 255);
        public static Color Amber => new Color32(240, 165, 126, 255);
        static Material BodyMaterial => Material(ref _graphite, new Color32(109, 132, 146, 255));
        static Material FootMaterial => Material(ref _base, new Color32(37, 52, 61, 255));
        static Material TeamMaterial(PlayerId owner) => owner == PlayerId.Player0
            ? Material(ref _mint, Mint) : Material(ref _amber, Amber);

        public static GameObject Build(string id, PlayerId owner, Transform parent)
        {
            int index = UnitArt.Index(id);
            var root = new GameObject("Art_" + UnitArt.Ids[index]);
            root.transform.SetParent(parent, false);
            if (_foot == null) _foot = Lathe(new[] { new Vector2(.48f, .03f), new Vector2(.56f, .08f), new Vector2(.56f, .15f), new Vector2(.48f, .21f) }, 32);
            if (_rim == null) _rim = Ring(.535f, .022f, 48, false);
            if (_brokenRim == null) _brokenRim = Ring(.535f, .022f, 48, true);
            Part(root.transform, "Foot", _foot, FootMaterial, Vector3.zero);
            Part(root.transform, "TeamRim", owner == PlayerId.Player0 ? _rim : _brokenRim, TeamMaterial(owner), new Vector3(0, .17f, 0));
            Part(root.transform, "Graphite", Body(index), BodyMaterial, Vector3.zero);
            return root;
        }

        static GameObject Part(Transform parent, string name, Mesh mesh, Material material, Vector3 position)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            go.transform.localPosition = position;
            go.AddComponent<MeshFilter>().sharedMesh = mesh;
            var renderer = go.AddComponent<MeshRenderer>();
            renderer.sharedMaterial = material;
            renderer.shadowCastingMode = ShadowCastingMode.Off;
            return go;
        }

        public static Mesh Body(int index)
        {
            if (Bodies[index] != null) return Bodies[index];
            var pieces = new List<CombineInstance>();
            void Add(Mesh mesh, Vector3 p, Vector3 scale, Vector3 rotation = default) => pieces.Add(new CombineInstance
                { mesh = mesh, transform = Matrix4x4.TRS(p, Quaternion.Euler(rotation), scale) });
            Mesh column = Lathe(new[] { new Vector2(.30f, 0), new Vector2(.38f, .08f), new Vector2(.38f, .60f), new Vector2(.25f, .72f) }, 8);
            switch (index)
            {
                case 0: // Relay: a balanced faceted core.
                    Add(column, new Vector3(0, .22f, 0), Vector3.one); break;
                case 1: // Atlas: wide shoulders, a planted lower body.
                    Add(column, new Vector3(0, .23f, 0), new Vector3(1.27f, .87f, 1));
                    Add(column, new Vector3(-.34f, .24f, 0), new Vector3(.48f, .64f, .75f));
                    Add(column, new Vector3(.34f, .24f, 0), new Vector3(.48f, .64f, .75f)); break;
                case 2: // Edge: one leaning blade.
                    Add(Prism(new[] { new Vector2(-.28f, .23f), new Vector2(.25f, .23f), new Vector2(.42f, 1.19f), new Vector2(.12f, 1.03f) }, .27f), Vector3.zero, Vector3.one); break;
                case 3: // Bastion: two shields with a legible central gap.
                    Add(column, new Vector3(-.23f, .23f, 0), new Vector3(.49f, 1.30f, 1));
                    Add(column, new Vector3(.23f, .23f, 0), new Vector3(.49f, 1.30f, 1)); break;
                case 4: // Glide: low swept skids.
                    var skid = Prism(new[] { new Vector2(-.44f, .23f), new Vector2(.46f, .23f), new Vector2(.28f, .62f), new Vector2(-.25f, .47f) }, .22f);
                    Add(skid, new Vector3(0, 0, -.26f), Vector3.one);
                    Add(skid, new Vector3(0, 0, .26f), Vector3.one);
                    Add(column, new Vector3(0, .26f, 0), new Vector3(.60f, .26f, .8f)); break;
                case 5: // Crux: arch with separated feet.
                    Add(column, new Vector3(-.29f, .23f, 0), new Vector3(.42f, .85f, .65f));
                    Add(column, new Vector3(.29f, .23f, 0), new Vector3(.42f, .85f, .65f));
                    Add(column, new Vector3(0, .73f, 0), new Vector3(1.17f, .25f, .65f)); break;
                case 6: // Lance: turret and one slender barrel.
                    Add(column, new Vector3(-.09f, .23f, 0), new Vector3(.91f, .55f, .91f));
                    Add(column, new Vector3(0, .52f, 0), new Vector3(.23f, .95f, .23f), new Vector3(0, 0, -78)); break;
                case 7: // Halo: a true opening through the ring.
                    Add(column, new Vector3(0, .23f, 0), new Vector3(.32f, .69f, .32f));
                    Add(Ring(.32f, .08f, 24, false), new Vector3(0, .90f, 0), Vector3.one, new Vector3(77, 0, -12)); break;
            }
            var result = new Mesh { name = UnitArt.Names[index] + " sculpt" };
            result.CombineMeshes(pieces.ToArray());
            var temporary = new HashSet<Mesh>();
            foreach (var part in pieces) if (part.mesh != column) temporary.Add(part.mesh);
            foreach (var mesh in temporary) Dispose(mesh);
            Dispose(column);
            result.RecalculateBounds();
            Bodies[index] = result;
            return result;
        }

        static Mesh Lathe(Vector2[] profile, int sides)
        {
            var v = new List<Vector3>(); var t = new List<int>();
            // Independent quad vertices keep the cut facets crisp.
            for (int j = 0; j < profile.Length - 1; j++)
                for (int i = 0; i < sides; i++)
                {
                    float a = i * Mathf.PI * 2 / sides, b = (i + 1) * Mathf.PI * 2 / sides;
                    Vector3 P(Vector2 p, float angle) => new Vector3(Mathf.Cos(angle) * p.x, p.y, Mathf.Sin(angle) * p.x);
                    Quad(v, t, P(profile[j], a), P(profile[j + 1], a), P(profile[j + 1], b), P(profile[j], b));
                }
            for (int i = 0; i < sides; i++)
            {
                float a = i * Mathf.PI * 2 / sides, b = (i + 1) * Mathf.PI * 2 / sides;
                var p = profile[profile.Length - 1];
                Triangle(v, t, new Vector3(0, p.y, 0), new Vector3(Mathf.Cos(b)*p.x,p.y,Mathf.Sin(b)*p.x), new Vector3(Mathf.Cos(a)*p.x,p.y,Mathf.Sin(a)*p.x));
            }
            return Mesh(v, t);
        }
        static Mesh Prism(Vector2[] profile, float depth)
        {
            var v = new List<Vector3>(); var t = new List<int>();
            Vector3 P(int i, float z) => new Vector3(profile[i].x, profile[i].y, z);
            for (int i=0;i<profile.Length;i++)
            { int j=(i+1)%profile.Length; Quad(v,t,P(i,-depth/2),P(j,-depth/2),P(j,depth/2),P(i,depth/2)); }
            for(int i=1;i<profile.Length-1;i++)
            { Triangle(v,t,P(0,depth/2),P(i,depth/2),P(i+1,depth/2)); Triangle(v,t,P(0,-depth/2),P(i+1,-depth/2),P(i,-depth/2)); }
            return Mesh(v,t);
        }
        static Mesh Ring(float radius, float tube, int steps, bool broken)
        {
            var v = new List<Vector3>(); var t = new List<int>();
            Vector3 P(int i, int j) { float a=i*2*Mathf.PI/steps,b=j*2*Mathf.PI/6; return new Vector3((radius+tube*Mathf.Cos(b))*Mathf.Cos(a),tube*Mathf.Sin(b),(radius+tube*Mathf.Cos(b))*Mathf.Sin(a)); }
            for(int i=0;i<steps;i++)
            { if(broken && i%8>=5) continue; for(int j=0;j<6;j++) Quad(v,t,P(i,j),P(i,j+1),P(i+1,j+1),P(i+1,j)); }
            return Mesh(v,t);
        }
        static void Triangle(List<Vector3> v,List<int> t,Vector3 a,Vector3 b,Vector3 c)
        { int n=v.Count;v.Add(a);v.Add(b);v.Add(c);t.Add(n);t.Add(n+1);t.Add(n+2); }
        static void Quad(List<Vector3> v,List<int> t,Vector3 a,Vector3 b,Vector3 c,Vector3 d)
        { Triangle(v,t,a,b,c);Triangle(v,t,a,c,d); }
        static Mesh Mesh(List<Vector3> v,List<int> t)
        { var m=new Mesh();m.SetVertices(v);m.SetTriangles(t,0);m.RecalculateNormals();m.RecalculateBounds();return m; }
        static void Dispose(Object o) { if(Application.isPlaying) Object.Destroy(o); else Object.DestroyImmediate(o); }

        /// <summary>Visible UI requests portraits lazily; at most one uncached portrait renders per frame.</summary>
        public static bool TryGetPortrait(int index, bool detailed, out Texture portrait)
        {
            int tier = detailed ? 1 : 0;
            portrait = Portraits[tier, index];
            if (portrait != null) return true;
            if (SystemInfo.graphicsDeviceType == GraphicsDeviceType.Null || _lastPortraitFrame == Time.frameCount)
                return false;
            _lastPortraitFrame = Time.frameCount;
            portrait = Portraits[tier, index] = RenderPortrait(index, detailed ? 512 : 128);
            return true;
        }

        // Keep the rendered image on the GPU: RawImage can use it directly without ReadPixels.
        static RenderTexture RenderPortrait(int index, int size)
        {
            var root = Build(UnitArt.Ids[index], PlayerId.Player0, null);
            root.transform.position = new Vector3(10000,10000,10000);
            foreach(var t in root.GetComponentsInChildren<Transform>()) t.gameObject.layer=31;
            var go = new GameObject("Piece portrait camera");
            var camera = go.AddComponent<Camera>();
            camera.enabled=false;camera.orthographic=true;camera.orthographicSize=.92f;
            camera.clearFlags=CameraClearFlags.SolidColor;camera.backgroundColor=new Color(0,0,0,0);camera.cullingMask=1<<31;
            camera.nearClipPlane=.1f;camera.farClipPlane=10;
            var center=root.transform.position+Vector3.up*.59f;
            go.transform.position=center+new Vector3(2.5f,2.3f,-4);
            go.transform.LookAt(center);
            var rt = new RenderTexture(size, size, 24, RenderTextureFormat.ARGB32)
            { name = UnitArt.Names[index] + " portrait " + size };
            var previous=RenderTexture.active;
            try
            {
                camera.targetTexture = rt;
                camera.Render();
                return rt;
            }
            catch { Dispose(rt); throw; }
            finally
            {
                RenderTexture.active = previous;
                camera.targetTexture = null;
                root.SetActive(false);
                Dispose(root);
                Dispose(go);
            }
        }
    }
}
