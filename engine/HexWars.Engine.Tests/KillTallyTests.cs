using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class KillTallyTests
    {
        [Test]
        public void NewPlayer_HasZeroDestroyedValue()
        {
            Assert.That(new PlayerState(PlayerId.Player0, 0).DestroyedValue, Is.EqualTo(0));
        }

        [Test]
        public void WithPoints_PreservesDestroyedValue()
        {
            var p = new PlayerState(PlayerId.Player0, 5).WithDestroyed(7);
            Assert.That(p.WithPoints(99).DestroyedValue, Is.EqualTo(7));
        }
    }
}
