using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class BarracksCatalogTests
    {
        private static readonly (string Name, UnitStats Stats)[] ExpectedDefaults =
        {
            ("Brute",     new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1)),
            ("Striker",   new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1)),
            ("Sniper",    new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1)),
            ("Artillery", new UnitStats(3, 6, 0, 0, 0, 5, 2, 2, 1)),
            ("Scout",     new UnitStats(2, 0, 0, 4, 3, 0, 0, 7, 2)),
        };

        [Test]
        public void DefaultTemplates_MatchTheFiveSpecifiedTemplatesExactly()
        {
            Assert.That(BarracksCatalog.DefaultTemplates.Count, Is.EqualTo(ExpectedDefaults.Length));
            for (int i = 0; i < ExpectedDefaults.Length; i++)
            {
                Assert.That(BarracksCatalog.DefaultTemplates[i].Name, Is.EqualTo(ExpectedDefaults[i].Name), $"slot {i} name");
                Assert.That(BarracksCatalog.DefaultTemplates[i].Stats, Is.EqualTo(ExpectedDefaults[i].Stats), $"slot {i} stats");
            }
        }

        [Test]
        public void Normalize_RemovesOnlyExactNameAndNineStatDuplicates()
        {
            var baseStats = new UnitStats(1, 2, 3, 4, 5, 6, 7, 8, 9);
            var source = new[]
            {
                new UnitTemplate("Recruit", baseStats),
                new UnitTemplate("Recruit", baseStats),
                new UnitTemplate("Recruit", new UnitStats(2, 2, 3, 4, 5, 6, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 3, 3, 4, 5, 6, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 4, 4, 5, 6, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 5, 5, 6, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 4, 6, 6, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 4, 5, 7, 7, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 4, 5, 6, 8, 8, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 4, 5, 6, 7, 9, 9)),
                new UnitTemplate("Recruit", new UnitStats(1, 2, 3, 4, 5, 6, 7, 8, 10)),
                new UnitTemplate("Other", baseStats),
            };

            var normalized = BarracksCatalog.Normalize(source);

            Assert.That(normalized, Has.Count.EqualTo(11));
            Assert.That(normalized.Count(x => x.Name == "Recruit"), Is.EqualTo(10));
            Assert.That(normalized.Last().Name, Is.EqualTo("Other"));
        }

        [Test]
        public void Normalize_SanitizesNamesAndRejectsInvalidHealth()
        {
            var valid = new UnitStats(1, 2, 3, 4, 5, 6, 7, 8, 9);
            var invalid = new UnitStats(0, 2, 3, 4, 5, 6, 7, 8, 9);

            var normalized = BarracksCatalog.Normalize(new[]
            {
                new UnitTemplate("  Doom_Turtle!!  ", valid),
                new UnitTemplate("Invalid", invalid),
            });

            Assert.That(normalized, Has.Count.EqualTo(1));
            Assert.That(normalized[0].Name, Is.EqualTo("DoomTurtle"));
            Assert.That(normalized[0].Stats, Is.EqualTo(valid));
        }

        [Test]
        public void Normalize_EnforcesProtocolMaximumOfSixtyFourTemplates()
        {
            var source = Enumerable.Range(1, 65)
                .Select(i => new UnitTemplate($"Template {i}", new UnitStats(i, 0, 0, 0, 0, 0, 0, 0, 0)));

            var normalized = BarracksCatalog.Normalize(source, maxTemplates: 100);

            Assert.That(normalized, Has.Count.EqualTo(64));
            Assert.That(normalized[63].Stats.Health, Is.EqualTo(64));
        }
    }
}
