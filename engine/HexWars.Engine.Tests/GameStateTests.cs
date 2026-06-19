using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameStateTests
    {
        private static GameState Minimal()
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, points: 10),
                new PlayerState(PlayerId.Player1, points: 7),
            };
            return new GameState(board, GameConfig.Default(), players,
                activePlayer: PlayerId.Player0, round: 1, nextEntityId: 1);
        }

        [Test]
        public void Player_ReturnsThatPlayersState()
        {
            Assert.That(Minimal().Player(PlayerId.Player1).Points, Is.EqualTo(7));
        }

        [Test]
        public void Opponent_ReturnsTheOtherPlayer()
        {
            Assert.That(Minimal().Opponent(PlayerId.Player0).Id, Is.EqualTo(PlayerId.Player1));
        }

        [Test]
        public void Clone_IsADistinctInstanceWithEqualFields()
        {
            var s = Minimal();
            var c = s.Clone();

            Assert.That(ReferenceEquals(s, c), Is.False);
            Assert.That(c.ActivePlayer, Is.EqualTo(s.ActivePlayer));
            Assert.That(c.Round, Is.EqualTo(s.Round));
            Assert.That(c.NextEntityId, Is.EqualTo(s.NextEntityId));
            Assert.That(c.Player(PlayerId.Player0).Points, Is.EqualTo(10));
        }
    }
}
