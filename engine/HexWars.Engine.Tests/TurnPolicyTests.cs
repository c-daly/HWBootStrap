using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TurnPolicyTests
    {
        private static GameState FreshWith(ITurnPolicy policy)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(1, 0, 0, true) },
            };
            var cfg = new GameConfig(terrain, turnPolicy: policy);
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });
            var players = new[] { new PlayerState(PlayerId.Player0, 10), new PlayerState(PlayerId.Player1, 10) };
            return new GameState(board, cfg, players, PlayerId.Player0, 1, 1);
        }

        [Test]
        public void AllUnitsPolicy_DoesNotAutoEndTurn_AfterAnAction()
        {
            var r = GameEngine.Apply(FreshWith(new AllUnitsPolicy()),
                new CreateUnit(PlayerId.Player0, TestStates.Cost(2)));
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void OneActionPolicy_AutoEndsTurn_AfterASingleAction()
        {
            var r = GameEngine.Apply(FreshWith(new OneActionPolicy()),
                new CreateUnit(PlayerId.Player0, TestStates.Cost(2)));
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player1));
        }

        [Test]
        public void Default_IsAllUnitsPolicy()
        {
            Assert.That(GameConfig.Default().TurnPolicy, Is.TypeOf<AllUnitsPolicy>());
        }
    }
}
