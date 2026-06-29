using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// Starting army is configurable: requested role counts are honoured, the rest of ArmySize is filled
    /// randomly (all-zero = fully random), and both sides get the same composition.
    /// Roles are identified by signature stats — Brute Health 7, Striker Damage 6, Sniper Range 6.
    /// </summary>
    public class ArmyCompositionTests
    {
        [Test]
        public void Build_RequestedCounts_HonouredAndFilledToSize()
        {
            var setup = new GameSetup(GameMode.Annihilation, 15, 13, 0, 7, armySize: 5, brutes: 2, strikers: 1, snipers: 0);
            var units = GameFactory.Build(setup).Player(PlayerId.Player0).UnitsOnBoard;

            Assert.That(units.Count, Is.EqualTo(5), "total == army size when the board has room");
            Assert.That(units.Count(u => u.Stats.Health == 7), Is.GreaterThanOrEqualTo(2), "at least the requested Brutes");
            Assert.That(units.Count(u => u.Stats.Damage == 6), Is.GreaterThanOrEqualTo(1), "at least the requested Strikers");
        }

        [Test]
        public void Build_AllZeroCounts_GivesRandomArmyOfSize()
        {
            var setup = new GameSetup(GameMode.Annihilation, 15, 13, 0, 7, armySize: 6, brutes: 0, strikers: 0, snipers: 0);
            Assert.That(GameFactory.Build(setup).Player(PlayerId.Player0).UnitsOnBoard.Count, Is.EqualTo(6));
        }

        [Test]
        public void Build_Army_IsSymmetric()
        {
            var setup = new GameSetup(GameMode.Annihilation, 15, 13, 0, 7, armySize: 4, brutes: 0, strikers: 0, snipers: 0);
            var st = GameFactory.Build(setup);
            string Sig(PlayerId id) => string.Join(",", st.Player(id).UnitsOnBoard
                .Select(u => $"{u.Stats.Health}/{u.Stats.Damage}/{u.Stats.Range}").OrderBy(s => s));
            Assert.That(Sig(PlayerId.Player0), Is.EqualTo(Sig(PlayerId.Player1)), "both sides get the same composition");
        }
    }
}
