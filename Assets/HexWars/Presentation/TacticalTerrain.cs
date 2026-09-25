using System.Collections.Generic;
using HexWars.Engine;
using UnityEngine;
using UnityEngine.Rendering;

namespace HexWars.Presentation
{
    /// <summary>Small, collider-free surface details. Terrain rules remain in GameConfig.</summary>
    public static class TacticalTerrain
    {
        static readonly Dictionary<TerrainType,Mesh> Meshes = new Dictionary<TerrainType,Mesh>();
        static Material _material;
        public static void Add(Transform column,TerrainType terrain,float top,float radius)
        {
            if(terrain==TerrainType.Plains)return;
            if(_material==null)
            {
                _material=new Material(Shader.Find("Universal Render Pipeline/Unlit"));
                _material.SetColor("_BaseColor",new Color(.18f,.26f,.28f));
                _material.SetFloat("_Cull",0);
            }
            if(!Meshes.TryGetValue(terrain,out var mesh)) Meshes[terrain]=mesh=Build(terrain);
            var go=new GameObject("Terrain detail");go.transform.SetParent(column,false);
            go.transform.localPosition=new Vector3(0,top+.035f,0);go.transform.localScale=new Vector3(radius,1,radius);
            go.AddComponent<MeshFilter>().sharedMesh=mesh;
            var r=go.AddComponent<MeshRenderer>();r.sharedMaterial=_material;r.shadowCastingMode=ShadowCastingMode.Off;
        }
        static Mesh Build(TerrainType terrain)
        {
            var vertices=new List<Vector3>();var triangles=new List<int>();
            void Triangle(Vector3 a,Vector3 b,Vector3 c){int i=vertices.Count;vertices.Add(a);vertices.Add(b);vertices.Add(c);triangles.Add(i);triangles.Add(i+1);triangles.Add(i+2);}
            void Stroke(float x,float z,float x2,float z2)
            {
                var a=new Vector3(x,0,z);var b=new Vector3(x2,0,z2);var n=Vector3.Cross((b-a).normalized,Vector3.up)*.014f;
                Triangle(a+n,b+n,b-n);Triangle(a+n,b-n,a-n);
            }
            if(terrain==TerrainType.Water)
                for(int j=0;j<3;j++)for(int i=0;i<6;i++)
                {float x=-.42f+i*.14f,z=.40f+j*.12f;Stroke(x,z+Mathf.Sin(i*1.8f)*.035f,x+.14f,z+Mathf.Sin((i+1)*1.8f)*.035f);}
            else if(terrain==TerrainType.Rough)
            {Stroke(-.72f,-.24f,-.50f,-.35f);Stroke(-.5f,-.35f,-.32f,-.65f);Stroke(-.50f,-.35f,-.64f,-.56f);Stroke(.42f,.58f,.71f,.33f);}
            else if(terrain==TerrainType.Forest)
                for(int tree=0;tree<3;tree++)
                {
                    var p=new Vector3(-.60f+tree*.18f,0,.38f+(tree%2)*.13f);
                    for(int k=0;k<6;k++)
                    {float a=k*Mathf.PI/3,b=(k+1)*Mathf.PI/3;Triangle(p+new Vector3(Mathf.Cos(a)*.09f,0,Mathf.Sin(a)*.09f),p+Vector3.up*.22f,p+new Vector3(Mathf.Cos(b)*.09f,0,Mathf.Sin(b)*.09f));}
                }
            var m=new Mesh{name="Terrain "+terrain};m.SetVertices(vertices);m.SetTriangles(triangles,0);m.RecalculateNormals();m.RecalculateBounds();return m;
        }
    }
}
