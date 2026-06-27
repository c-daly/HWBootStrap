using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TerritoryDeployTests
    {
        static readonly HexCoord A = new HexCoord(0, 0);
        static readonly HexCoord B = new HexCoord(1, 0);
        static readonly UnitStats S = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);

        // P0 has a barracks template + 100 pts. A is controlled by P0; B is neutral. No deployment zones set.
        static GameState State()
        {
            var board = new Board(new[] { new Tile(A, 0, TerrainType.Plains), new Tile(B, 0, TerrainType.Plains) })
                .WithControl(A, PlayerId.Player0);
            var p0 = new PlayerState(PlayerId.Player0, 100, new[] { S });
            var p1 = new PlayerState(PlayerId.Player1, 0);
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true);
            return new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 2, 9);
        }

        [Test]
        public void Deploy_OnControlledHex_Succeeds()
        {
            var r = GameEngine.Apply(State(), new DeployUnit(PlayerId.Player0, 0, A));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Player(PlayerId.Player0).UnitsOnBoard.Count, Is.EqualTo(1));
        }

        [Test]
        public void Deploy_OnUncontrolledHex_Rejected()
        {
            var r = GameEngine.Apply(State(), new DeployUnit(PlayerId.Player0, 0, B));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.HexNotControlled));
        }
    }
}
