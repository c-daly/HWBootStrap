using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class ModelDuelConfigurationTests
    {
        [TestCase(ModelControllerKind.Greedy, "", MlModelAlgorithm.MaskablePpo, "greedy")]
        [TestCase(ModelControllerKind.Random, "", MlModelAlgorithm.MaskablePpo, "random")]
        [TestCase(ModelControllerKind.FixedCheckpoint, "C:/models/a.zip", MlModelAlgorithm.MaskedDqn, "dqn:C:/models/a.zip")]
        public void SeatSpec_BuildsExplicitControllerIdentity(
            ModelControllerKind kind, string path, MlModelAlgorithm algorithm, string expected)
        {
            var seat = new ModelSeatConfiguration { Kind = kind, Path = path, Algorithm = algorithm };

            Assert.That(seat.BuildSpec(), Is.EqualTo(expected));
        }

        [Test]
        public void LiveRunSpec_IsStructuredAndExplicitlyLive()
        {
            var seat = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun, Path = "C:/runs/one" };

            string spec = seat.BuildSpec();

            Assert.That(spec, Does.Contain("\"kind\":\"run\""));
            Assert.That(spec, Does.Contain("\"mode\":\"live\""));
            Assert.That(spec, Does.Contain("C:/runs/one"));
        }

        [Test]
        public void Reload_IsAllowedOnlyAtGameBoundaryForLiveSeats()
        {
            var config = new ModelDuelConfiguration
            {
                P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.LiveRun, Path = "run-a" },
                P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy },
            };

            Assert.That(config.ShouldReload(gameEnded: false), Is.False);
            Assert.That(config.ShouldReload(gameEnded: true), Is.True);
        }

        [Test]
        public void LegacyCheckpointDirectory_RemainsLiveForCompatibility()
        {
            Assert.That(ModelDuelDriver.IsLiveRun("ppo:C:/runs/one/checkpoints", _ => true), Is.True);
            Assert.That(ModelDuelDriver.IsLiveRun("ppo:C:/runs/one/model.zip", _ => false), Is.False);
        }

        [Test]
        public void Validate_RejectsMissingModelPathsAndInvalidPacing()
        {
            var config = new ModelDuelConfiguration
            {
                P0 = new ModelSeatConfiguration { Kind = ModelControllerKind.FixedCheckpoint },
                SecondsPerAction = 0,
            };

            Assert.That(config.Validate(), Has.Some.Contains("Seat 0"));
            Assert.That(config.Validate(), Has.Some.Contains("pacing"));
        }
    }
}
