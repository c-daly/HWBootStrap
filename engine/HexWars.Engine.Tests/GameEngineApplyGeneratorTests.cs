using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyGeneratorTests
    {
        [Test]
        public void DeployGenerator_Places_PaysCost_IncrementsEntityId()
        {
            var r = GameEngine.Apply(TestStates.Fresh(p0Points: 5),
                new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));

            Assert.That(r.Success, Is.True);
            var p0 = r.NewState.Player(PlayerId.Player0);
            Assert.That(p0.Points, Is.EqualTo(3));               // 5 - GeneratorCost(2)
            Assert.That(p0.Generators.Count, Is.EqualTo(1));
            Assert.That(p0.Generators[0].Cell, Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(p0.Generators[0].CurrentHp, Is.EqualTo(3)); // GeneratorHealth
            Assert.That(r.NewState.NextEntityId, Is.EqualTo(2));
        }

        [Test]
        public void DeployGenerator_Rejects_OutsideDeploymentZone()
        {
            var r = GameEngine.Apply(TestStates.Fresh(),
                new DeployGenerator(PlayerId.Player0, new HexCoord(1, 0))); // Player1's zone
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.OutsideDeploymentZone));
        }

        [Test]
        public void DeployGenerator_Rejects_OnOccupiedTile()
        {
            var first = GameEngine.Apply(TestStates.Fresh(p0Points: 10),
                new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));
            var second = GameEngine.Apply(first.NewState,
                new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));
            Assert.That(second.Reason, Is.EqualTo(RejectionReason.TileOccupied));
        }

        [Test]
        public void DeployGenerator_Rejects_WhenCannotAfford()
        {
            var r = GameEngine.Apply(TestStates.Fresh(p0Points: 1),
                new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }
    }
}
