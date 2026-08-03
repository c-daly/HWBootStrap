using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class EventConsolePrivacyTests
    {
        [Test]
        public void ArmySummary_UnderFog_ShowsOnlyFixedObserversExactArmyTotals()
        {
            GameState state = State(fogOfWar: true);

            string player1 = EventConsole.FormatArmySummary(state, PlayerId.Player0);
            string player2 = EventConsole.FormatArmySummary(state, PlayerId.Player1);

            Assert.That(player1, Does.Contain("P1</color>  1u"));
            Assert.That(player1, Does.Contain("P2</color>  ?u · ?v"));
            Assert.That(player2, Does.Contain("P1</color>  ?u · ?v"));
            Assert.That(player2, Does.Contain("P2</color>  2u"));
        }

        [Test]
        public void ArmySummary_WithoutFog_PreservesTacticalExactTotals()
        {
            string summary = EventConsole.FormatArmySummary(State(fogOfWar: false), PlayerId.Player0);

            Assert.That(summary, Does.Contain("P1</color>  1u"));
            Assert.That(summary, Does.Contain("P2</color>  2u"));
            Assert.That(summary, Does.Not.Contain("?u"));
        }

        static GameState State(bool fogOfWar)
        {
            var board = new Board(Enumerable.Range(0, 3)
                .Select(q => new Tile(new HexCoord(q, 0), 0, TerrainType.Plains)).ToArray());
            Unit Unit(int id, PlayerId owner, int cell) => new Unit(id, owner,
                new UnitStats(5, 1, 0, 1, 1, 1, 1, 1, 1), new HexCoord(cell, 0), 0);
            return new GameState(board, GameConfig.Default(fogOfWar: fogOfWar), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { Unit(1, PlayerId.Player0, 0) }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[]
                {
                    Unit(2, PlayerId.Player1, 1), Unit(3, PlayerId.Player1, 2),
                }),
            }, PlayerId.Player0, round: 1, nextEntityId: 4);
        }
    }
}
