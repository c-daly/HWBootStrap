using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Eight original, cached machine forms. Art never changes a unit's picking or rules.</summary>
    public static class GraphitePieces
    {
        static readonly Mesh[] Bodies = new Mesh[8];
        static readonly Mesh[] RunningGear = new Mesh[8], Details = new Mesh[8], Lamps = new Mesh[8];
        static Material _rubber, _trim;
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
        static Material BodyMaterial => Material(ref _graphite, new Color32(132, 153, 165, 255));
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
            Part(root.transform, "RunningGear", RunningGear[index], Material(ref _rubber, new Color32(27, 39, 49, 255)), Vector3.zero);
            Part(root.transform, "MachinedEdges", Details[index], Material(ref _trim, new Color32(178, 193, 198, 255)), Vector3.zero);
            Part(root.transform, "Sensors", Lamps[index], TeamMaterial(owner), Vector3.zero);
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
            var body = new List<CombineInstance>(); var gear = new List<CombineInstance>();
            var trim = new List<CombineInstance>(); var lights = new List<CombineInstance>();
            var temporary = new HashSet<Mesh>();
            Mesh cube = Prism(new[] { new Vector2(-.5f,-.5f),new Vector2(.5f,-.5f),new Vector2(.5f,.5f),new Vector2(-.5f,.5f) },1);
            Mesh cylinder = Lathe(new[] { new Vector2(.5f,-.5f),new Vector2(.5f,.5f) },16);
            temporary.Add(cube); temporary.Add(cylinder);
            void Add(List<CombineInstance> parts, Mesh mesh, Vector3 p, Vector3 size, Vector3 rotation = default)
            { temporary.Add(mesh); parts.Add(new CombineInstance { mesh=mesh, transform=Matrix4x4.TRS(p,Quaternion.Euler(rotation),size) }); }
            void Box(List<CombineInstance> parts,float x,float y,float z,float w,float h,float d,Vector3 rotation=default)
                => Add(parts,cube,new Vector3(x,y,z),new Vector3(w,h,d),rotation);
            void Rod(List<CombineInstance> parts,Vector3 a,Vector3 b,float width)
            {
                parts.Add(new CombineInstance { mesh=cylinder,transform=Matrix4x4.TRS((a+b)*.5f,
                    Quaternion.FromToRotation(Vector3.up,(b-a).normalized),new Vector3(width,Vector3.Distance(a,b),width)) });
            }
            // Hull / chassis. All pieces keep the same independent picking collider and art identity.
            Box(body,0,.36f,0,.77f,.22f,.91f);
            Box(trim,0,.485f,-.21f,.54f,.03f,.32f);
            bool wheels=index==0 || index==4 || index==7;
            if(index==5)
            {
                // Four articulated legs, knee housings and broad feet.
                for(int x=-1;x<=1;x+=2) for(int z=-1;z<=1;z+=2)
                {
                    var hip=new Vector3(x*.24f,.48f,z*.23f);var knee=new Vector3(x*.51f,.32f,z*.32f);
                    var foot=new Vector3(x*.59f,.13f,z*.46f);
                    Rod(body,hip,knee,.14f);Rod(gear,knee,foot,.12f);
                    Add(trim,cylinder,knee,new Vector3(.19f,.12f,.19f),new Vector3(0,0,90));
                    Box(body,foot.x,.095f,foot.z,.24f,.08f,.30f);
                }
            }
            else if(wheels)
            {
                for(int x=-1;x<=1;x+=2) for(int z=-1;z<=1;z++)
                {
                    Add(gear,cylinder,new Vector3(x*.40f,.23f,z*.30f),new Vector3(.31f,.16f,.31f),new Vector3(0,0,90));
                    Add(trim,cylinder,new Vector3(x*.49f,.23f,z*.30f),new Vector3(.14f,.015f,.14f),new Vector3(0,0,90));
                }
            }
            else
            {
                for(int x=-1;x<=1;x+=2)
                {
                    Box(gear,x*.40f,.23f,0,.22f,.24f,1.03f);
                    for(int j=0;j<9;j++) Box(trim,x*.514f,.23f,-.45f+j*.112f,.017f,.16f,.028f);
                    Box(body,x*.40f,.38f,0,.24f,.035f,1.03f);
                }
            }
            // Two headlights and an owner strip stay visible even at board scale.
            Box(lights,-.25f,.40f,.465f,.09f,.055f,.02f);Box(lights,.25f,.40f,.465f,.09f,.055f,.02f);
            Box(lights,0,.35f,-.467f,.38f,.025f,.018f);
            switch(index)
            {
                case 0: // Relay: box cab and a jointed utility arm.
                    Box(body,-.08f,.66f,.11f,.43f,.39f,.44f);
                    Box(gear,-.08f,.72f,.34f,.32f,.17f,.012f);
                    Box(trim,-.08f,.87f,.11f,.46f,.025f,.47f);
                    Rod(body,new Vector3(.27f,.48f,-.23f),new Vector3(.30f,.91f,-.10f),.09f);
                    Rod(trim,new Vector3(.30f,.91f,-.10f),new Vector3(.30f,1.02f,.28f),.07f);
                    Rod(gear,new Vector3(.30f,1.02f,.28f),new Vector3(.30f,.88f,.35f),.055f);
                    break;
                case 1: // Atlas: broad, sloped armor and a narrow view slit.
                    Add(body,Prism(new[]{new Vector2(-.38f,.45f),new Vector2(.38f,.45f),new Vector2(.27f,.80f),new Vector2(-.27f,.80f)},.68f),Vector3.zero,Vector3.one);
                    Box(gear,0,.66f,.348f,.40f,.06f,.012f);Box(lights,0,.66f,.357f,.21f,.023f,.015f);
                    Box(trim,0,.81f,0,.45f,.02f,.48f);break;
                case 2: // Edge: a forward assault turret with paired short cannons.
                case 6: // Lance: longer barrel, recoil sleeve, muzzle opening.
                    Box(body,0,.60f,.02f,.43f,.23f,.40f);
                    Box(trim,0,.73f,.02f,.32f,.025f,.31f);
                    for(int j=0;j<(index==2?2:1);j++)
                    {
                        float x=index==2?(j==0?-.13f:.13f):0;
                        float end=index==2?.75f:1.04f;
                        Rod(body,new Vector3(x,.62f,.15f),new Vector3(x,.71f,end),index==2?.09f:.095f);
                        Rod(trim,new Vector3(x,.64f,.28f),new Vector3(x,.67f,.48f),.14f);
                        Box(gear,x,.711f,end+.015f,.065f,.065f,.016f);
                    }
                    break;
                case 3: // Bastion: broad armored shield on a tracked carrier.
                    Box(body,0,.70f,.29f,.80f,.70f,.16f,new Vector3(-9,0,0));
                    Box(trim,0,1.04f,.24f,.81f,.028f,.19f);
                    Box(gear,0,.78f,.37f,.46f,.055f,.018f);
                    for(int x=-1;x<=1;x+=2)Box(trim,x*.27f,.63f,.39f,.022f,.38f,.015f);
                    Box(lights,0,.79f,.388f,.26f,.026f,.014f);break;
                case 4: // Glide: light buggy with sloped cab, exposed wheels and whip antenna.
                    Add(body,Prism(new[]{new Vector2(-.25f,.45f),new Vector2(.25f,.45f),new Vector2(.17f,.69f),new Vector2(-.17f,.69f)},.43f),new Vector3(0,0,.02f),Vector3.one);
                    Box(gear,0,.58f,.246f,.32f,.12f,.013f);
                    Rod(trim,new Vector3(-.24f,.49f,-.27f),new Vector3(-.28f,1.04f,-.34f),.017f);
                    Box(lights,-.28f,1.045f,-.34f,.033f,.034f,.033f);break;
                case 5:
                    Box(body,0,.60f,0,.48f,.26f,.49f);
                    Box(trim,0,.742f,0,.32f,.024f,.35f);
                    Box(lights,0,.65f,.252f,.21f,.055f,.018f);break;
                case 7: // Halo: real parabolic dish, equipment cabinet and sensor mast.
                    Box(body,0,.54f,-.08f,.42f,.19f,.52f);
                    Rod(trim,new Vector3(0,.56f,0),new Vector3(0,.91f,0),.07f);
                    var dish=Lathe(new[]{new Vector2(0,0),new Vector2(.11f,.025f),new Vector2(.24f,.10f),new Vector2(.34f,.24f),new Vector2(.32f,.25f),new Vector2(.22f,.13f),new Vector2(.09f,.065f),new Vector2(0,.055f)},24);
                    Add(body,dish,new Vector3(0,.87f,0),Vector3.one,new Vector3(34,0,0));
                    Rod(trim,new Vector3(0,.94f,0),new Vector3(0,1.19f,.12f),.026f);
                    Box(lights,0,1.19f,.12f,.055f,.04f,.05f);break;
            }
            Mesh Combine(List<CombineInstance> parts,string name)
            { var mesh=new Mesh{name=UnitArt.Names[index]+" "+name};mesh.CombineMeshes(parts.ToArray());mesh.RecalculateBounds();return mesh; }
            Bodies[index]=Combine(body,"machine");RunningGear[index]=Combine(gear,"running gear");
            Details[index]=Combine(trim,"edges");Lamps[index]=Combine(lights,"sensors");
            foreach(var mesh in temporary)Dispose(mesh);
            return Bodies[index];
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
            for (int i = 0; profile[profile.Length - 1].x > 0 && i < sides; i++)
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
            go.transform.position=center+new Vector3(2.5f,2.3f,4);
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
