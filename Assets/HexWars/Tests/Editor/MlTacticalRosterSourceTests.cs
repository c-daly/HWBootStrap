using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Presentation;
using HexWars.Presentation.EditorTools.MlLab;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class MlTacticalRosterSourceTests
    {
        [TearDown]
        public void TearDown()
        {
            SessionBarracksCache.ResetForTests();
        }

        [Test]
        public void Snapshot_IncludesDefaultsAndSelectedPlayersSavedTemplates()
        {
            SessionBarracksCache.ResetForTests();
            SessionBarracksCache.ForLocalPlayer(1).Add(
                new UnitTemplate("Custom Alpha", new UnitStats(4, 3, 1, 3, 2, 2, 1, 4, 1)));

            IReadOnlyList<MlTrainingUnitTemplate> snapshot =
                MlTacticalRosterSource.Snapshot(1);

            List<string> defaultNames = BarracksCatalog.DefaultTemplates
                .Select(item => item.Name).ToList();
            List<string> snapshotNames = snapshot.Select(item => item.Name).ToList();
            Assert.That(snapshotNames.Take(defaultNames.Count), Is.EqualTo(defaultNames));
            Assert.That(snapshot.Select(item => item.Name), Does.Contain("Custom Alpha"));
            Assert.That(MlTacticalRosterSource.Snapshot(0).Select(item => item.Name),
                Does.Not.Contain("Custom Alpha"));
        }

        [Test]
        public void Snapshot_RejectsOutOfRangeLocalPlayer()
        {
            Assert.Throws<System.ArgumentOutOfRangeException>(
                () => MlTacticalRosterSource.Snapshot(2));
        }

        [Test]
        public void Snapshot_DerivesStableIdsMatchingEngineDerivation()
        {
            IReadOnlyList<MlTrainingUnitTemplate> snapshot = MlTacticalRosterSource.Snapshot(0);

            IEnumerable<string> expectedIds = BarracksCatalog.DefaultTemplates
                .Select(HexWars.Engine.Rl.TacticalV2TemplateIds.From);
            Assert.That(snapshot.Select(item => item.Id), Is.EqualTo(expectedIds));
        }

        [Test]
        public void Snapshot_ReturnsDetachedCopyUnaffectedByLaterCacheChanges()
        {
            SessionBarracksCache.ResetForTests();
            IReadOnlyList<MlTrainingUnitTemplate> before = MlTacticalRosterSource.Snapshot(0);

            SessionBarracksCache.ForLocalPlayer(0).Add(
                new UnitTemplate("Late Addition", new UnitStats(3, 3, 1, 3, 2, 2, 1, 3, 1)));

            Assert.That(before.Select(item => item.Name), Does.Not.Contain("Late Addition"));
        }
    }
}
