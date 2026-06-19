using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyEndTurnTests
    {
        [Test]
        public void EndTurn_SwitchesActivePlayer_AndResetsActedSets()
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            var players = new[] { new PlayerState(PlayerId.Player0, 5), new PlayerState(PlayerId.Player1, 5) };
            var state = new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 100,
                movedUnitIds: new[] { 1 }, attackedUnitIds: new[] { 2 });

            var r = GameEngine.Apply(state, new EndTurn(PlayerId.Player0));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player1));
            Assert.That(r.NewState.MovedUnitIds, Is.Empty);
            Assert.That(r.NewState.AttackedUnitIds, Is.Empty);
        }

        [Test]
        public void EndTurn_AdvancesRound_OnlyWhenReturningToPlayer0()
        {
            var afterP0 = GameEngine.Apply(TestStates.Fresh(), new EndTurn(PlayerId.Player0)).NewState;
            Assert.That(afterP0.ActivePlayer, Is.EqualTo(PlayerId.Player1));
            Assert.That(afterP0.Round, Is.EqualTo(1));

            var afterP1 = GameEngine.Apply(afterP0, new EndTurn(PlayerId.Player1)).NewState;
            Assert.That(afterP1.ActivePlayer, Is.EqualTo(PlayerId.Player0));
            Assert.That(afterP1.Round, Is.EqualTo(2));
        }

        [Test]
        public void EndTurn_CreditsIncomeToTheNewActivePlayer()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            });
            var gen = new Generator(5, PlayerId.Player1, new HexCoord(1, 0), 0, 3);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0),
                new PlayerState(PlayerId.Player1, 4, generators: new[] { gen }),
            };
            var state = new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 100);

            var r = GameEngine.Apply(state, new EndTurn(PlayerId.Player0));
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.EqualTo(5)); // 4 + income 1
        }
    }
}
