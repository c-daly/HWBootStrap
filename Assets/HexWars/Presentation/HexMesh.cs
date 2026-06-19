using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Procedural flat-top hex meshes used by <see cref="BoardRenderer"/>.</summary>
    public static class HexMesh
    {
        /// <summary>A solid flat-top hex prism from y=0 to y=height.</summary>
        public static Mesh Prism(float radius, float height)
        {
            var mesh = new Mesh();
            var v = new Vector3[14];
            v[0] = new Vector3(0, height, 0);
            v[7] = new Vector3(0, 0, 0);
            for (int k = 0; k < 6; k++)
            {
                float a = Mathf.Deg2Rad * (60f * k);
                float x = radius * Mathf.Cos(a), z = radius * Mathf.Sin(a);
                v[1 + k] = new Vector3(x, height, z);
                v[8 + k] = new Vector3(x, 0, z);
            }
            var t = new List<int>();
            for (int k = 0; k < 6; k++) { t.Add(0); t.Add(1 + (k + 1) % 6); t.Add(1 + k); }
            for (int k = 0; k < 6; k++) { t.Add(7); t.Add(8 + k); t.Add(8 + (k + 1) % 6); }
            for (int k = 0; k < 6; k++)
            {
                int t0 = 1 + k, t1 = 1 + (k + 1) % 6, b0 = 8 + k, b1 = 8 + (k + 1) % 6;
                t.Add(t0); t.Add(t1); t.Add(b0);
                t.Add(t1); t.Add(b1); t.Add(b0);
            }
            mesh.vertices = v;
            mesh.triangles = t.ToArray();
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>A flat hexagonal annulus at y=0 (outer radius -> inner radius).</summary>
        public static Mesh Ring(float outerR, float innerR)
        {
            var mesh = new Mesh();
            var v = new Vector3[12];
            for (int k = 0; k < 6; k++)
            {
                float a = Mathf.Deg2Rad * (60f * k);
                float c = Mathf.Cos(a), si = Mathf.Sin(a);
                v[k] = new Vector3(outerR * c, 0, outerR * si);
                v[6 + k] = new Vector3(innerR * c, 0, innerR * si);
            }
            var t = new List<int>();
            for (int k = 0; k < 6; k++)
            {
                int o0 = k, o1 = (k + 1) % 6, i0 = 6 + k, i1 = 6 + (k + 1) % 6;
                t.Add(o0); t.Add(o1); t.Add(i1);
                t.Add(o0); t.Add(i1); t.Add(i0);
            }
            mesh.vertices = v;
            mesh.triangles = t.ToArray();
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            return mesh;
        }
    }
}
