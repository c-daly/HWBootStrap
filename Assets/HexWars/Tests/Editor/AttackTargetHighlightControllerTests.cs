using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class AttackTargetHighlightControllerTests
    {
        [Test]
        public void Show_CreatesColliderFreeCircularHaloForEachTarget()
        {
            var boardObject = new GameObject("Board");
            try
            {
                boardObject.AddComponent<BoardRenderer>();
                var highlights = boardObject.AddComponent<AttackTargetHighlightController>();
                var targets = new List<AttackPreviewTarget>
                {
                    new AttackPreviewTarget(7, new HexCoord(0, 0), elevation: 0)
                };

                highlights.Show(targets);

                var root = boardObject.transform.Find("AttackTargetHighlights");
                Assert.That(root, Is.Not.Null);
                Assert.That(root.childCount, Is.EqualTo(1));
                var halo = root.GetChild(0);
                Assert.That(halo.gameObject.activeSelf, Is.True);
                Assert.That(halo.name, Is.EqualTo("AttackTarget_7"));
                Assert.That(halo.GetComponent<Collider>(), Is.Null);
                Assert.That(halo.GetComponent<MeshFilter>().sharedMesh.vertexCount, Is.GreaterThan(12));
                Assert.That(halo.localPosition.y,
                    Is.GreaterThan(boardObject.GetComponent<BoardRenderer>().LevelHeight + 0.70f),
                    "the halo must render above the unit disc, icon, and health bar");
            }
            finally
            {
                Object.DestroyImmediate(boardObject);
            }
        }
    }
}
