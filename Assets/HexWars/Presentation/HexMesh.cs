using System.Collections.Generic;
using UnityEngine;

namespace HexWars.Presentation
{
    /// <summary>Procedural flat-top hex meshes used by <see cref="BoardRenderer"/>.</summary>
    public static class HexMesh
    {
        /// <summary>A solid flat-top hex prism from y=0 to y=height. Faceted: every face has its own
        /// vertices so normals stay flat per face (no smoothing 'swoop' across the flat faces).</summary>
        public static Mesh Prism(float radius, float height)
        {
            var top = new Vector3[6];
            var bot = new Vector3[6];
            for (int k = 0; k < 6; k++)
            {
                float a = Mathf.Deg2Rad * (60f * k);
                float x = radius * Mathf.Cos(a), z = radius * Mathf.Sin(a);
                top[k] = new Vector3(x, height, z);
                bot[k] = new Vector3(x, 0f, z);
            }

            var verts = new List<Vector3>();
            var tris = new List<int>();

            // top face (own verts)
            int tc = verts.Count; verts.Add(new Vector3(0, height, 0));
            int ts = verts.Count; for (int k = 0; k < 6; k++) verts.Add(top[k]);
            for (int k = 0; k < 6; k++) { tris.Add(tc); tris.Add(ts + (k + 1) % 6); tris.Add(ts + k); }

            // bottom face (own verts)
            int bc = verts.Count; verts.Add(new Vector3(0, 0, 0));
            int bs = verts.Count; for (int k = 0; k < 6; k++) verts.Add(bot[k]);
            for (int k = 0; k < 6; k++) { tris.Add(bc); tris.Add(bs + k); tris.Add(bs + (k + 1) % 6); }

            // side quads (own verts per quad)
            for (int k = 0; k < 6; k++)
            {
                int k1 = (k + 1) % 6;
                int bi = verts.Count;
                verts.Add(top[k]); verts.Add(top[k1]); verts.Add(bot[k]); verts.Add(bot[k1]);
                tris.Add(bi + 0); tris.Add(bi + 1); tris.Add(bi + 2);
                tris.Add(bi + 1); tris.Add(bi + 3); tris.Add(bi + 2);
            }

            var mesh = new Mesh();
            mesh.SetVertices(verts);
            mesh.SetTriangles(tris, 0);
            mesh.RecalculateNormals(); // non-shared verts => flat per-face normals
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
