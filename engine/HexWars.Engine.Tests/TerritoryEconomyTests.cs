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

        static bool UnitAt(GameState st, HexCoord cell)
        {
            foreach (var u in st.Player(PlayerId.Player0).UnitsOnBoard)
                if (u.IsAlive && u.Cell == cell) return true;
            return false;
        }
    }
}
