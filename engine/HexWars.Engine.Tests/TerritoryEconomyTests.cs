using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The territory-economy config levers used by the balance experiments: passive per-hex income,
    /// disabling generators, and building on any controlled hex vs only under a unit.
    /// </summary>
    public class TerritoryEconomyTests
    {
        static GameConfig Cfg(bool buildAnywhere = false, int territoryIncome = 0, bool generatorsEnabled = true) =>
            GameConfig.Default(biomesEnabled: false, territoryMode: true, startingPoints: 50,
                generatorOutput: 0, territoryIncome: territoryIncome,
                generatorsEnabled: generatorsEnabled, buildAnywhere: buildAnywhere);

        [Test]
        public void Income_TerritoryIncome_PaysPerControlledHex()
        {
            var st = GameFactory.BuildTerritory(Cfg(territoryIncome: 2), 11, 9, 1);
            int hexes = Economy.ControlledHexes(st, PlayerId.Player0);
            Assert.That(hexes, Is.GreaterThan(0));
            Assert.That(Economy.Income(st, PlayerId.Player0), Is.EqualTo(2 * hexes));
        }

        [Test]
        public void LegalMoves_GeneratorsDisabled_OffersNoBuild()
        {
            var st = GameFactory.BuildTerritory(Cfg(generatorsEnabled: false), 11, 9, 1);
            foreach (var m in LegalMoves.For(st))
                Assert.That(m, Is.Not.InstanceOf<BuildGenerator>());
        }

        [Test]
        public void LegalMoves_BuildAnywhere_OffersBuildOnAHexWithNoUnit()
        {
            var st = GameFactory.BuildTerritory(Cfg(buildAnywhere: true), 11, 9, 1);
            bool foundEmpty = false;
            foreach (var m in LegalMoves.For(st))
                if (m is BuildGenerator b && !UnitAt(st, b.Cell)) { foundEmpty = true; break; }
            Assert.That(foundEmpty, Is.True, "build-anywhere should offer a generator on a controlled hex with no unit");
        }

        [Test]
        public void LegalMoves_OccupiedOnly_OffersBuildOnlyUnderUnits()
        {
            var st = GameFactory.BuildTerritory(Cfg(buildAnywhere: false), 11, 9, 1);
            foreach (var m in LegalMoves.For(st))
                if (m is BuildGenerator b)
                    Assert.That(UnitAt(st, b.Cell), Is.True, "without build-anywhere, builds only under a unit");
        }

        [Test]
        public void PointDecay_BleedsABankedSurplus()
        {
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true, pointDecay: 0.2, startingPoints: 100);
            var st = GameFactory.BuildTerritory(cfg, 11, 9, 1);
            int before = st.Player(PlayerId.Player1).Points; // P1 gets income+decay when P0 ends its turn
            var r = GameEngine.Apply(st, new EndTurn(PlayerId.Player0));
            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Player(PlayerId.Player1).Points, Is.LessThan(before),
                "a big bank with no generator income should shrink under decay");
        }

        [Test]
        public void KActionsPolicy_AutoEndsTurnAfterKActions()
        {
            var cfg = GameConfig.Default(biomesEnabled: false, territoryMode: true, turnPolicy: new KActionsPolicy(1));
            var st = GameFactory.BuildTerritory(cfg, 11, 9, 1);
            foreach (var u in st.Player(PlayerId.Player0).UnitsOnBoard)
            {
                HexCoord? dest = null;
                foreach (var c in MovementService.ReachableTiles(st, u)) { dest = c; break; }
                if (dest == null) continue;
                var r = GameEngine.Apply(st, new MoveUnit(PlayerId.Player0, u.Id, dest.Value));
                Assert.That(r.Success, Is.True);
                Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player1),
                    "K=1 should auto-end the turn after a single action");
                return;
            }
            Assert.Fail("no movable unit to test with");
        }

        static bool UnitAt(GameState st, HexCoord cell)
        {
            foreach (var u in st.Player(PlayerId.Player0).UnitsOnBoard)
                if (u.IsAlive && u.Cell == cell) return true;
            return false;
        }
    }
}
