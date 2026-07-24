using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2ConfigTests
    {
        [Test]
        public void Default_UsesCanonicalCatalogAndThreeStartingSlots()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            Assert.That(config.Templates.Select(item => item.Template.Name),
                Is.EqualTo(BarracksCatalog.DefaultTemplates.Select(item => item.Name)));
            Assert.That(config.StartingUnitCount, Is.EqualTo(3));
            Assert.That(config.MaxControllableUnits, Is.EqualTo(3));
            Assert.That(config.Validate(), Is.Empty);
        }

        [Test]
        public void SampleStartingArmy_IsSeededWithReplacementAndSymmetricInput()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = 12;
            config.MaxControllableUnits = 12;

            string[] first = config.SampleStartingArmy(37).Select(item => item.Id).ToArray();
            string[] second = config.SampleStartingArmy(37).Select(item => item.Id).ToArray();

            Assert.That(second, Is.EqualTo(first));
            Assert.That(first, Has.Length.EqualTo(12));
            Assert.That(first.Distinct().Count(), Is.LessThan(12));
        }

        [TestCase(0)]
        [TestCase(13)]
        public void Validate_RejectsStartingCountsOutsideRegularGameLimit(int count)
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = count;
            config.MaxControllableUnits = count;

            Assert.That(config.Validate(),
                Has.Some.Contains("starting unit count must be between 1 and 12"));
        }
    }
}
