using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public class GraphitePiecesTests
    {
        [Test]
        public void AllEightFormsShareCachedMeshesAndKeepPickingOnTheToken()
        {
            for(int i=0;i<8;i++)
            {
                var a=GraphitePieces.Build(UnitArt.Ids[i],PlayerId.Player0,null);
                var b=GraphitePieces.Build(UnitArt.Ids[i],PlayerId.Player1,null);
                try
                {
                    var mesh=a.transform.Find("Graphite").GetComponent<MeshFilter>().sharedMesh;
                    Assert.That(mesh,Is.SameAs(b.transform.Find("Graphite").GetComponent<MeshFilter>().sharedMesh));
                    Assert.That(mesh.vertexCount,Is.GreaterThan(20));
                    Assert.That(mesh.bounds.size.y,Is.GreaterThan(.25f));
                    Assert.That(a.GetComponentsInChildren<Collider>(),Is.Empty);
                    Assert.That(a.transform.Find("TeamRim").GetComponent<MeshFilter>().sharedMesh.triangles.Length,
                        Is.GreaterThan(b.transform.Find("TeamRim").GetComponent<MeshFilter>().sharedMesh.triangles.Length));
                }
                finally { Object.DestroyImmediate(a);Object.DestroyImmediate(b); }
            }
        }
        [Test]
        public void DesignerRetainsManualArtUntilExplicitRoleMatching()
        {
            var go=new GameObject("Designer test");var designer=go.AddComponent<DesignPanel>();
            try { designer.SelectArt("halo-01");Assert.That(designer.AppearanceSelection,Is.EqualTo("halo-01"));
                designer.SelectArt("");Assert.That(designer.AppearanceSelection,Is.Empty); }
            finally { Object.DestroyImmediate(go); }
        }
    }
}
