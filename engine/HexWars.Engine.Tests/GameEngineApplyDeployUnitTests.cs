using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyDeployUnitTests
    {
        // Fresh state where Player0 has one barracks template (cost 3) ready to deploy.
        private static GameState WithOneTemplate(int points = 12) =>
            GameEngine.Apply(TestStates.Fresh(p0Points: points),
                new CreateUnit(PlayerId.Player0, TestStates.Cost(3))).NewState;

        // Two deployable tiles in Player0's zone, so we can deploy multiple clones.
        private static GameState WithTemplateAndRoom(int points = 12)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(0, 1), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0), new HexCoord(0, 1) }, zone1: new[] { new HexCoord(1, 0) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, points),
                new PlayerState(PlayerId.Player1, 12),
            };
            var state = new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
            return GameEngine.Apply(state, new CreateUnit(PlayerId.Player0, TestStates.Cost(3))).NewState;
        }

        [Test]
        public void DeployUnit_ClonesTemplate_PaysDeployCost_KeepsTemplate_IncrementsId()
        {
            var r = GameEngine.Apply(WithOneTemplate(), new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));

            Assert.That(r.Success, Is.True);
            var p0 = r.NewState.Player(PlayerId.Player0);
            Assert.That(p0.Barracks.Count, Is.EqualTo(1));         // template is NOT consumed
            Assert.That(p0.Points, Is.EqualTo(9));                 // 12 - deploy cost 3
            Assert.That(p0.UnitsOnBoard.Count, Is.EqualTo(1));
            Assert.That(p0.UnitsOnBoard[0].Cell, Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(p0.UnitsOnBoard[0].CurrentHp, Is.EqualTo(3));
            Assert.That(r.NewState.NextEntityId, Is.EqualTo(2));
        }

        [Test]
        public void DeployUnit_CanDeployMultipleClonesOfSameTemplate()
        {
            var first = GameEngine.Apply(WithTemplateAndRoom(), new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));
            var second = GameEngine.Apply(first.NewState, new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 1)));

            Assert.That(second.Success, Is.True);
            var p0 = second.NewState.Player(PlayerId.Player0);
            Assert.That(p0.UnitsOnBoard.Count, Is.EqualTo(2));     // two clones
            Assert.That(p0.Barracks.Count, Is.EqualTo(1));         // single reusable template
            Assert.That(p0.Points, Is.EqualTo(6));                 // 12 - 3 - 3
        }

        [Test]
        public void DeployUnit_Rejects_WhenCannotAffordDeployCost()
        {
            var r = GameEngine.Apply(WithOneTemplate(points: 2), new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void DeployUnit_Rejects_BadTemplateIndex()
        {
            var r = GameEngine.Apply(WithOneTemplate(), new DeployUnit(PlayerId.Player0, 5, new HexCoord(0, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TemplateNotFound));
        }

        [Test]
        public void DeployUnit_Rejects_OutsideDeploymentZone()
        {
            var r = GameEngine.Apply(WithOneTemplate(), new DeployUnit(PlayerId.Player0, 0, new HexCoord(1, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.OutsideDeploymentZone));
        }

        [Test]
        public void DeployUnit_Rejects_OnOccupiedTile()
        {
            var withGen = GameEngine.Apply(WithOneTemplate(), new DeployGenerator(PlayerId.Player0, new HexCoord(0, 0)));
            var r = GameEngine.Apply(withGen.NewState, new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TileOccupied));
        }
    }
}
