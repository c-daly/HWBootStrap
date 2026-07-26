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

        [TestCase(1, 100, 404)]
        [TestCase(3, 100, 808)]
        [TestCase(12, 100, 2626)]
        [TestCase(3, 200, 1608)]
        [TestCase(3, 150, 1208)]
        public void DefaultMaxSteps_DerivesFromRoundCapAndStartingUnitCount(
            int startingUnitCount, int roundCap, int expected)
        {
            // 100-round-rule: MaxSteps counts RL actions (move/attack/deploy/end-turn), never rounds. Each
            // round both seats can spend up to (startingUnitCount + 1) actions apiece (one per unit slot
            // plus an end-turn) — so reaching roundCap needs roundCap * 2 * (startingUnitCount + 1) actions,
            // plus one extra round's headroom so the RL step budget never pre-empts the engine's own
            // round-cap backstop.
            Assert.That(TacticalV2Config.DefaultMaxSteps(startingUnitCount, roundCap), Is.EqualTo(expected));
        }

        [TestCase(1, 100, 400)]
        [TestCase(3, 100, 800)]
        [TestCase(12, 100, 2600)]
        public void MinimumMaxSteps_IsExactlyRoundCapActionsWithNoHeadroom(
            int startingUnitCount, int roundCap, int expected)
        {
            Assert.That(TacticalV2Config.MinimumMaxSteps(startingUnitCount, roundCap), Is.EqualTo(expected));
        }

        [Test]
        public void Default_DerivesMaxStepsFromRoundCapNotAMagicConstant()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            Assert.That(config.MaxSteps,
                Is.EqualTo(TacticalV2Config.DefaultMaxSteps(config.StartingUnitCount, GameConfig.DefaultRoundCap)));
            Assert.That(config.Game.RoundCap, Is.EqualTo(GameConfig.DefaultRoundCap));
        }

        [Test]
        public void TemplateIds_PinsTheCanonicalBruteIdCheckedIntoTrainingTemplates()
        {
            // "brute-85597320" is checked into python/config/training-game-templates.json. If
            // TacticalV2TemplateIds.From's slug/hash algorithm ever changes, that literal silently
            // stops matching the catalog it was derived from — pin it here so the break is loud.
            UnitTemplate brute = BarracksCatalog.DefaultTemplates
                .Single(template => template.Name == "Brute");

            Assert.That(TacticalV2TemplateIds.From(brute), Is.EqualTo("brute-85597320"));
        }
    }
}
