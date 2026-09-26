using System.Linq;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameFactoryFormationTests
    {
        [TestCase(5, 5, 7, 3)]
        [TestCase(6, 8, 42, 6)]
        [TestCase(9, 7, 7, 3)]
        [TestCase(13, 9, 236, 12)]
        [TestCase(12, 10, 99999, 12)]
        public void BothArmiesFillTheirBackEdgeFirstWithMirroredRoles(int width, int height, int seed, int size)
        {
            foreach (var mode in new[] { GameMode.Annihilation, GameMode.Territory })
            {
                var setup = new GameSetup(mode, width, height, 40, seed, armySize: size);
                Check(GameFactory.Build(setup), width, height);
                if (mode == GameMode.Annihilation)
                    Check(GameFactory.BuildTacticalV3Compatible(setup), width, height);
                else
                    Check(GameFactory.BuildTerritory(GameConfig.Default(territoryMode: true), width, height, seed), width, height);
            }
        }

        static void Check(GameState state, int width, int height)
        {
            var mine = state.Player(PlayerId.Player0).UnitsOnBoard.ToArray();
            var theirs = state.Player(PlayerId.Player1).UnitsOnBoard.ToArray();
            Assert.That(theirs.Length, Is.EqualTo(mine.Length));
            for (int i = 0; i < mine.Length; i++)
            {
                int col = mine[i].Cell.Q, row = mine[i].Cell.R + (col - (col & 1)) / 2;
                Assert.That(theirs[i].Cell, Is.EqualTo(HexLayout.OffsetToAxial(width - 1 - col, height - 1 - row)), $"role {i}");
                Assert.That(theirs[i].Stats, Is.EqualTo(mine[i].Stats));
            }
            foreach (var owner in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                var army = state.Player(owner).UnitsOnBoard;
                var used = army.Select(u => u.Cell).ToHashSet();
                var unfilled = state.Board.DeploymentZone(owner)
                    .Where(c => state.Board.TileAt(c).Elevation == 0 && !used.Contains(c)).ToArray();
                foreach (var unit in army)
                {
                    Assert.That(state.Board.IsInDeploymentZone(owner, unit.Cell), Is.True);
                    Assert.That(unit.Elevation, Is.Zero);
                    Assert.That(unfilled.Any(c => owner == PlayerId.Player0 ? c.Q < unit.Cell.Q : c.Q > unit.Cell.Q), Is.False,
                        "A forward rank must not be used while a flat backline cell is unfilled.");
                }
            }
        }
    }
}
