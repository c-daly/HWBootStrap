using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyDeployUnitTests
    {
        // Fresh state where Player0 has one reserved unit (cost 3) ready to deploy.
        private static GameState WithOneReserve(int points = 12) =>
            GameEngine.Apply(TestStates.Fresh(p0Points: points),
                new CreateUnit(PlayerId.Player0, TestStates.Cost(3))).NewState;

        [Test]
        public void DeployUnit_PlacesReserved_RemovesFromReserve_IncrementsId()
        {
            var r = GameEngine.Apply(WithOneReserve(), new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));

            Assert.That(r.Success, Is.True);
            var p0 = r.NewState.Player(PlayerId.Player0);
            Assert.That(p0.Reserve.Count, Is.EqualTo(0));
            Assert.That(p0.UnitsOnBoard.Count, Is.EqualTo(1));
            Assert.That(p0.UnitsOnBoard[0].Cell, Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(p0.UnitsOnBoard[0].CurrentHp, Is.EqualTo(3));
            Assert.That(r.NewState.NextEntityId, Is.EqualTo(2));
        }

        [Test]
        public void DeployUnit_Rejects_BadReserveIndex()
        {
            var r = GameEngine.Apply(WithOneReserve(), new DeployUnit(PlayerId.Player0, 5, new HexCoord(0, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.ReserveUnitNotFound));
        }

        [Test]
        public void DeployUnit_Rejects_OutsideDeploymentZone()
        {
            var r = GameEngine.Apply(WithOneReserve(), new DeployUnit(PlayerId.Player0, 0, new HexCoord(1, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.OutsideDeploymentZone));
        }

        [Test]
        public void DeployUnit_Rejects_OnOccupiedTile()
        {
            var withGen = GameEngine.Apply(WithOneReserve(), new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));
            var r = GameEngine.Apply(withGen.NewState, new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TileOccupied));
        }
    }
}
