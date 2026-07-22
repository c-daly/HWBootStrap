using System.Reflection;
using HexWars.Engine;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class NetClientCatalogHandshakeTests
    {
        [Test]
        public void CatalogRequest_BuildsCurrentSeatCatalogWithoutTimingGuess()
        {
            SessionBarracksCache.ResetForTests();
            var gameObject = new GameObject("Catalog handshake test", typeof(NetClient));
            try
            {
                var client = gameObject.GetComponent<NetClient>();
                typeof(NetClient).GetProperty("Seat", BindingFlags.Instance | BindingFlags.Public)
                    .SetValue(client, PlayerId.Player0);
                var method = typeof(NetClient).GetMethod("StartingCatalogMessage",
                    BindingFlags.Instance | BindingFlags.NonPublic);
                Assert.That(method, Is.Not.Null, "catalog sending must be driven by the server request");

                string message = (string)method.Invoke(client, null);
                var parsed = NetProtocol.Parse(message);
                Assert.That(parsed.Type, Is.EqualTo("CATALOG"));
                Assert.That(BarracksWire.Read(parsed.Payload),
                    Has.Count.EqualTo(BarracksCatalog.DefaultTemplates.Count));
            }
            finally
            {
                SessionBarracksCache.ResetForTests();
                UnityEngine.Object.DestroyImmediate(gameObject);
            }
        }
    }
}
