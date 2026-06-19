using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyTests
    {
        private static GameState Fresh(int p0Points = 12, int p1Points = 12)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, p0Points),
                new PlayerState(PlayerId.Player1, p1Points),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, 1, 1);
        }

        private static UnitStats Cost(int c) => new UnitStats(c, 0, 0, 0, 0, 0, 0, 0, 0);

        [Test]
        public void CreateUnit_PaysPoints_AndAddsToReserve_NonMutating()
        {
            var state = Fresh(p0Points: 5);
            var result = GameEngine.Apply(state, new CreateUnit(PlayerId.Player0, Cost(3)));

            Assert.That(result.Success, Is.True);
            Assert.That(result.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(2));
            Assert.That(result.NewState.Player(PlayerId.Player0).Reserve.Count, Is.EqualTo(1));
            Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(5)); // original untouched
        }

        [Test]
        public void CreateUnit_Rejects_WhenCannotAfford()
        {
            var result = GameEngine.Apply(Fresh(p0Points: 2), new CreateUnit(PlayerId.Player0, Cost(3)));
            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
        }

        [Test]
        public void Apply_Rejects_WhenNotYourTurn()
        {
            var result = GameEngine.Apply(Fresh(), new CreateUnit(PlayerId.Player1, Cost(1)));
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.NotYourTurn));
        }

        [Test]
        public void CreateUnit_Rejects_InvalidStatsBelowOneHealth()
        {
            var bad = new UnitStats(0, 5, 0, 0, 0, 0, 0, 0, 0); // 0 health
            var result = GameEngine.Apply(Fresh(), new CreateUnit(PlayerId.Player0, bad));
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InvalidStats));
        }
    }
}
