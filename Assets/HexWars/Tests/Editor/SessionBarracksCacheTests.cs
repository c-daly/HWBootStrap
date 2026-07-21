using System;
using System.Linq;
using HexWars.Engine;
using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class SessionBarracksCacheTests
    {
        [SetUp]
        public void SetUp() => SessionBarracksCache.ResetForTests();

        [TearDown]
        public void TearDown() => SessionBarracksCache.ResetForTests();

        [Test]
        public void ForLocalPlayer_LazilyStartsEachSeatWithCanonicalDefaults()
        {
            var p0 = SessionBarracksCache.ForLocalPlayer(0).Snapshot();
            var p1 = SessionBarracksCache.ForLocalPlayer(1).Snapshot();

            Assert.That(p0, Has.Count.EqualTo(BarracksCatalog.DefaultTemplates.Count));
            Assert.That(p1, Has.Count.EqualTo(BarracksCatalog.DefaultTemplates.Count));
            for (int i = 0; i < BarracksCatalog.DefaultTemplates.Count; i++)
            {
                Assert.That(BarracksCatalog.Same(p0[i], BarracksCatalog.DefaultTemplates[i]), Is.True);
                Assert.That(BarracksCatalog.Same(p1[i], BarracksCatalog.DefaultTemplates[i]), Is.True);
            }
        }

        [Test]
        public void Add_SanitizesAndRejectsOnlyExactDuplicates()
        {
            var cache = SessionBarracksCache.ForLocalPlayer(0);
            var stats = new UnitStats(3, 2, 1, 4, 5, 6, 7, 8, 9);

            Assert.That(cache.Add(new UnitTemplate("  Alpha_One!  ", stats)), Is.True);
            Assert.That(cache.Add(new UnitTemplate("AlphaOne", stats)), Is.False);
            Assert.That(cache.Add(new UnitTemplate("AlphaOne", new UnitStats(3, 3, 1, 4, 5, 6, 7, 8, 9))), Is.True);

            var added = cache.Snapshot().Skip(BarracksCatalog.DefaultTemplates.Count).ToArray();
            Assert.That(added.Select(x => x.Name), Is.EqualTo(new[] { "AlphaOne", "AlphaOne" }));
            Assert.That(added[0].Stats.Damage, Is.EqualTo(2));
            Assert.That(added[1].Stats.Damage, Is.EqualTo(3));
        }

        [Test]
        public void Add_RejectsInvalidHealthAndEntriesPastProtocolLimit()
        {
            var cache = SessionBarracksCache.ForLocalPlayer(0);
            Assert.That(cache.Add(new UnitTemplate("Invalid", Stats(0))), Is.False);

            int customSlots = BarracksCatalog.ProtocolMaximumTemplates - BarracksCatalog.DefaultTemplates.Count;
            for (int i = 0; i < customSlots; i++)
                Assert.That(cache.Add(new UnitTemplate("Custom " + i, Stats(i + 1))), Is.True);

            Assert.That(cache.Snapshot(), Has.Count.EqualTo(BarracksCatalog.ProtocolMaximumTemplates));
            Assert.That(cache.Add(new UnitTemplate("One Too Many", Stats(99))), Is.False);
        }

        [Test]
        public void RemoveAt_RemovesBuiltInAndCustomEntriesForTheSession()
        {
            var cache = SessionBarracksCache.ForLocalPlayer(0);
            Assert.That(cache.Add(new UnitTemplate("Custom", Stats(9))), Is.True);

            Assert.That(cache.RemoveAt(0), Is.True);
            Assert.That(cache.RemoveAt(cache.Snapshot().Count - 1), Is.True);

            Assert.That(cache.Snapshot().Select(x => x.Name),
                Is.EqualTo(BarracksCatalog.DefaultTemplates.Skip(1).Select(x => x.Name)));
            Assert.That(cache.RemoveAt(-1), Is.False);
            Assert.That(cache.RemoveAt(cache.Snapshot().Count), Is.False);
        }

        [Test]
        public void PlayerCaches_AreIndependent()
        {
            var p0 = SessionBarracksCache.ForLocalPlayer(0);
            var p1 = SessionBarracksCache.ForLocalPlayer(1);

            p0.RemoveAt(0);
            p1.Add(new UnitTemplate("P1 Custom", Stats(11)));

            Assert.That(p0.Snapshot().Select(x => x.Name), Does.Not.Contain("Brute"));
            Assert.That(p0.Snapshot().Select(x => x.Name), Does.Not.Contain("P1 Custom"));
            Assert.That(p1.Snapshot().Select(x => x.Name), Does.Contain("Brute"));
            Assert.That(p1.Snapshot().Select(x => x.Name), Does.Contain("P1 Custom"));
        }

        [Test]
        public void Snapshot_RemainsUnchangedWhenTheCacheChanges()
        {
            var cache = SessionBarracksCache.ForLocalPlayer(0);
            var snapshot = cache.Snapshot();

            cache.RemoveAt(0);

            Assert.That(snapshot, Has.Count.EqualTo(BarracksCatalog.DefaultTemplates.Count));
            Assert.That(snapshot[0].Name, Is.EqualTo("Brute"));
            Assert.That(cache.Snapshot(), Has.Count.EqualTo(BarracksCatalog.DefaultTemplates.Count - 1));
        }

        [Test]
        public void ResetForTests_DiscardsBothMutatedCaches()
        {
            SessionBarracksCache.ForLocalPlayer(0).RemoveAt(0);
            SessionBarracksCache.ForLocalPlayer(1).Add(new UnitTemplate("Custom", Stats(7)));

            SessionBarracksCache.ResetForTests();

            Assert.That(SessionBarracksCache.ForLocalPlayer(0).Snapshot().Select(x => x.Name),
                Is.EqualTo(BarracksCatalog.DefaultTemplates.Select(x => x.Name)));
            Assert.That(SessionBarracksCache.ForLocalPlayer(1).Snapshot().Select(x => x.Name),
                Is.EqualTo(BarracksCatalog.DefaultTemplates.Select(x => x.Name)));
        }

        [TestCase(-1)]
        [TestCase(2)]
        public void ForLocalPlayer_RejectsUnknownSeat(int seat)
        {
            Assert.That(() => SessionBarracksCache.ForLocalPlayer(seat),
                Throws.TypeOf<ArgumentOutOfRangeException>());
        }

        private static UnitStats Stats(int health) =>
            new UnitStats(health, 1, 2, 3, 4, 5, 6, 7, 8);
    }
}
