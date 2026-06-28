using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class WinConditionTests
    {
        static GameState With(GameConfig cfg, int p0Points, int p1Points, int round)
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            // both sides have a living unit so nobody is "eliminated"
            var u0 = new Unit(1, PlayerId.Player0, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), new HexCoord(0, 0), 0);
            var u1 = new Unit(2, PlayerId.Player1, new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1), new HexCoord(0, 0), 0);
            var pl0 = new PlayerState(PlayerId.Player0, p0Points, null, new[] { u0 });
            var pl1 = new PlayerState(PlayerId.Player1, p1Points, null, new[] { u1 });
            return new GameState(board, cfg, new[] { pl0, pl1 }, PlayerId.Player0, round, 9);
        }

        [Test]
        public void Economy_InstantWin_AtThreshold()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Economy, economyWinThreshold: 50);
            var s = With(cfg, p0Points: 60, p1Points: 0, round: 5);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void Economy_NotTerminal_BelowThreshold()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Economy, economyWinThreshold: 50);
            var s = With(cfg, p0Points: 10, p1Points: 10, round: 5);
            Assert.That(WinCheck.IsTerminal(s), Is.False);
        }

        [Test]
        public void Score_DecidesAtRoundCap_WhenEnabled()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Score, scorePoints: 1,
                scoreKills: 0, scoreArmy: 0, scoreTerritory: 0);
            var s = With(cfg, p0Points: 30, p1Points: 5, round: cfg.RoundCap);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.EqualTo(PlayerId.Player0)); // higher score
        }

        [Test]
        public void Cap_IsDraw_WhenScoreOff()
        {
            var cfg = GameConfig.Default(winConditions: WinBy.Annihilation);
            var s = With(cfg, p0Points: 30, p1Points: 5, round: cfg.RoundCap);
            Assert.That(WinCheck.IsTerminal(s), Is.True);
            Assert.That(WinCheck.Resolve(s), Is.Null); // annihilation-only: cap = draw, as today
        }
    }
}
