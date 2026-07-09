using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>UnitTemplate = name + stats, the barracks entry. Sanitize is the engine-boundary
    /// gate: no client can wire an unparseable or abusive-length name (spec §5/§7).</summary>
    public class UnitTemplateTests
    {
        [Test]
        public void Ctor_StoresNameAndStats()
        {
            var stats = new UnitStats(3, 3, 1, 2, 1, 1, 1, 2, 1);
            var t = new UnitTemplate("Longshot", stats);
            Assert.That(t.Name, Is.EqualTo("Longshot"));
            Assert.That(t.Stats, Is.EqualTo(stats));
        }

        [Test]
        public void Sanitize_TrimsLeadingAndTrailingWhitespace() =>
            Assert.That(UnitTemplate.Sanitize("  Doom Turtle  "), Is.EqualTo("Doom Turtle"));

        [Test]
        public void Sanitize_CapsAtTwentyCharacters() =>
            Assert.That(UnitTemplate.Sanitize(new string('A', 30)), Is.EqualTo(new string('A', 20)));

        [Test]
        public void Sanitize_StripsDisallowedCharacters() =>
            Assert.That(UnitTemplate.Sanitize("<Doom>Turtle!!"), Is.EqualTo("DoomTurtle"));

        [Test]
        public void Sanitize_NullOrEmpty_ReturnsEmptyString()
        {
            Assert.That(UnitTemplate.Sanitize(null), Is.EqualTo(""));
            Assert.That(UnitTemplate.Sanitize(""), Is.EqualTo(""));
        }

        [Test]
        public void Sanitize_KeepsWhitelistedPunctuation() =>
            Assert.That(UnitTemplate.Sanitize("Mama's Boy_2-nd"), Is.EqualTo("Mama's Boy_2-nd"));
    }
}
