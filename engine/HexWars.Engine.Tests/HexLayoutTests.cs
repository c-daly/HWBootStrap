using System;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class HexLayoutTests
    {
        [Test]
        public void ToWorld_Origin_IsAtZero()
        {
            var (x, z) = HexLayout.ToWorld(new HexCoord(0, 0), hexSize: 1.0);
            Assert.That(x, Is.EqualTo(0.0).Within(1e-9));
            Assert.That(z, Is.EqualTo(0.0).Within(1e-9));
        }

        [Test]
        public void ToWorld_ScalesWithHexSize()
        {
            var (x, z) = HexLayout.ToWorld(new HexCoord(1, 0), hexSize: 2.0);
            Assert.That(x, Is.EqualTo(1.5 * 1 * 2.0).Within(1e-9));
            Assert.That(z, Is.EqualTo(Math.Sqrt(3.0) * 2.0 * 0.5).Within(1e-9));
        }

        [Test]
        public void ToWorld_AdjacentColumns_AreSqrt3TimesSizeApart()
        {
            const double size = 1.0;
            var a = HexLayout.ToWorld(new HexCoord(0, 0), size);
            var b = HexLayout.ToWorld(new HexCoord(1, 0), size);
            double dist = Math.Sqrt(Math.Pow(a.x - b.x, 2) + Math.Pow(a.z - b.z, 2));
            Assert.That(dist, Is.EqualTo(Math.Sqrt(3.0) * size).Within(1e-9));
        }
    }
}
