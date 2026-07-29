using System.Collections.Generic;
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

        [Test]
        public void ProfiledSeededV1_AcceptsExactCatalogAndZeroWeightProfiles()
        {
            Assert.That(
                TacticalV2StartCatalog.ProfiledSeededV1().Select(profile => profile.Id),
                Is.EqualTo(new[]
                {
                    "standard-3v3",
                    "conversion-3v1-near", "conversion-3v1-medium", "conversion-3v1-far",
                    "conversion-2v1-near", "conversion-2v1-medium", "conversion-2v1-far",
                    "conversion-1v1-near", "conversion-1v1-medium", "conversion-1v1-far",
                }));

            TacticalV2Config config = ProfiledConfig(StandardOnlyWeights());

            Assert.That(config.Validate(), Is.Empty);
            Assert.That(config.StartDistribution.Weights.Count(item => item.BasisPoints == 0),
                Is.EqualTo(9));
        }

        [TestCase("")]
        [TestCase("standard-3v3")]
        public void ProfiledSeededV1_RejectsEmptyOrDuplicateProfileIds(string secondId)
        {
            var profiles = TacticalV2StartCatalog.ProfiledSeededV1().ToList();
            profiles[1] = new TacticalV2StartProfile(secondId, 3, 1, "near");
            TacticalV2Config config = ProfiledConfig(StandardOnlyWeights(), profiles);

            Assert.That(config.Validate(), secondId.Length == 0
                ? Has.Some.Contains("profile ids must not be empty")
                : Has.Some.Contains("duplicate start profile id 'standard-3v3'"));
        }

        [TestCase(0)]
        [TestCase(4)]
        public void ProfiledSeededV1_RejectsCountsOutsideSlotCapacity(int learnerUnits)
        {
            var profiles = TacticalV2StartCatalog.ProfiledSeededV1().ToList();
            profiles[1] = new TacticalV2StartProfile(
                profiles[1].Id, learnerUnits, profiles[1].OpponentUnitCount, profiles[1].Separation);
            TacticalV2Config config = ProfiledConfig(StandardOnlyWeights(), profiles);

            Assert.That(config.Validate(),
                Has.Some.Contains("learner unit count must be between 1 and max controllable units"));
        }

        [Test]
        public void ProfiledSeededV1_RejectsUnknownSeparation()
        {
            var profiles = TacticalV2StartCatalog.ProfiledSeededV1().ToList();
            profiles[1] = new TacticalV2StartProfile(
                profiles[1].Id, profiles[1].LearnerUnitCount, profiles[1].OpponentUnitCount, "adjacent");
            TacticalV2Config config = ProfiledConfig(StandardOnlyWeights(), profiles);

            Assert.That(config.Validate(), Has.Some.Contains("unknown separation 'adjacent'"));
        }

        [Test]
        public void ProfiledSeededV1_RequiresExactTenThousandBasisPointSum()
        {
            var weights = StandardOnlyWeights().ToList();
            weights[0] = new TacticalV2StartWeight("standard-3v3", 9999);
            TacticalV2Config config = ProfiledConfig(weights);

            Assert.That(config.Validate(),
                Has.Some.Contains("start distribution weights must sum to 10000 basis points"));
        }

        [Test]
        public void ProfiledSeededV1_RejectsWeightForUndeclaredProfile()
        {
            var weights = StandardOnlyWeights().ToList();
            weights[1] = new TacticalV2StartWeight("not-declared", 0);
            TacticalV2Config config = ProfiledConfig(weights);

            Assert.That(config.Validate(),
                Has.Some.Contains("weight references undeclared start profile 'not-declared'"));
        }

        [Test]
        public void ProfiledSeededV1_RequiresStandardProfileAndExactCatalog()
        {
            var profiles = TacticalV2StartCatalog.ProfiledSeededV1()
                .Where(profile => profile.Id != "standard-3v3")
                .ToArray();
            TacticalV2Config config = ProfiledConfig(
                StandardOnlyWeights().Where(weight => weight.ProfileId != "standard-3v3"), profiles);

            Assert.That(config.Validate(),
                Has.Some.Contains("profiled-seeded-v1 requires the exact versioned start profile catalog"));
        }

        [Test]
        public void ProfiledSeededV1_RequiresThreeStartingAndControllableSlots()
        {
            TacticalV2Config config = ProfiledConfig(StandardOnlyWeights());
            config.StartingUnitCount = 2;
            config.MaxControllableUnits = 2;

            Assert.That(config.Validate(), Has.Some.Contains(
                "profiled-seeded-v1 requires starting unit count and max controllable units to equal 3"));
        }

        [Test]
        public void DistributionSelection_IsDeterministicAndIndependentOfWeightIterationOrder()
        {
            var forward = new TacticalV2StartDistribution(new[]
            {
                new TacticalV2StartWeight("z-profile", 3400),
                new TacticalV2StartWeight("a-profile", 3300),
                new TacticalV2StartWeight("m-profile", 3300),
            });
            var reverse = new TacticalV2StartDistribution(new[]
            {
                new TacticalV2StartWeight("m-profile", 3300),
                new TacticalV2StartWeight("a-profile", 3300),
                new TacticalV2StartWeight("z-profile", 3400),
            });

            string[] first = Enumerable.Range(-20, 80).Select(forward.Select).ToArray();
            string[] second = Enumerable.Range(-20, 80).Select(reverse.Select).ToArray();

            Assert.That(second, Is.EqualTo(first));
            Assert.That(first.Distinct(), Is.EquivalentTo(new[] { "a-profile", "m-profile", "z-profile" }));
        }

        [Test]
        public void LegacySymmetricPolicy_KeepsEqualCountValidationAndIgnoresProfileFields()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartProfiles = new[] { new TacticalV2StartProfile("unused", 1, 1, "near") };
            config.StartDistribution = new TacticalV2StartDistribution(new[]
            {
                new TacticalV2StartWeight("unused", 10000),
            });

            Assert.That(config.Validate(), Is.Empty);

            config.MaxControllableUnits = 4;
            Assert.That(config.Validate(),
                Has.Some.Contains("max controllable units must equal starting unit count"));
        }

        private static TacticalV2Config ProfiledConfig(
            IEnumerable<TacticalV2StartWeight> weights,
            IReadOnlyList<TacticalV2StartProfile>? profiles = null)
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.PlacementPolicy = "profiled-seeded-v1";
            config.StartProfiles = profiles ?? TacticalV2StartCatalog.ProfiledSeededV1();
            config.StartDistribution = new TacticalV2StartDistribution(weights);
            return config;
        }

        private static IReadOnlyList<TacticalV2StartWeight> StandardOnlyWeights() =>
            TacticalV2StartCatalog.ProfiledSeededV1()
                .Select(profile => new TacticalV2StartWeight(
                    profile.Id, profile.Id == "standard-3v3" ? 10000 : 0))
                .ToArray();
    }
}
