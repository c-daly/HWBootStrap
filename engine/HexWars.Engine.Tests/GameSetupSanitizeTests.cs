using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Sanitized() clamps every field to the lobby form's own ranges so a hostile or
    /// corrupt ?setup= query can never OOM the server (audit finding N1). GameFactory.Build
    /// self-sanitizes, covering every construction path.</summary>
    public class GameSetupSanitizeTests
    {
        [Test]
        public void Sanitized_ClampsOversizedBoardAndArmy()
        {
            var s = new GameSetup(GameMode.Annihilation, 9999, 9999, 100000, 7, 500000, 99, 99, 99, 99).Sanitized();
            Assert.That(s.Width, Is.EqualTo(24));
            Assert.That(s.Height, Is.EqualTo(24));
            Assert.That(s.StartingPoints, Is.EqualTo(200));
            Assert.That(s.ArmySize, Is.EqualTo(12));
            Assert.That(s.Brutes, Is.EqualTo(12));
            Assert.That(s.Strikers, Is.EqualTo(12));
            Assert.That(s.Snipers, Is.EqualTo(12));
            Assert.That(s.TurnActions, Is.EqualTo(8));
        }

        [Test]
        public void Sanitized_ClampsUndersizedValues()
        {
            var s = new GameSetup((GameMode)99, 1, -5, -10, -3, 0, -1, -1, -1, -2).Sanitized();
            Assert.That((int)s.Mode, Is.InRange(0, 1));
            Assert.That(s.Width, Is.EqualTo(5));
            Assert.That(s.Height, Is.EqualTo(5));
            Assert.That(s.StartingPoints, Is.EqualTo(0));
            Assert.That(s.Seed, Is.EqualTo(1));
            Assert.That(s.ArmySize, Is.EqualTo(1));
            Assert.That(s.Brutes, Is.EqualTo(0));
            Assert.That(s.TurnActions, Is.EqualTo(0));
        }

        [Test]
        public void Sanitized_LegalValuesPassThroughUnchanged()
        {
            var input = new GameSetup(GameMode.Territory, 13, 9, 40, 1234, 5, 2, 2, 1, 3, fog: true);
            var s = input.Sanitized();
            Assert.That(s.ToWire(), Is.EqualTo(input.ToWire()));
        }

        [Test]
        public void GameFactoryBuild_SanitizesHostileSetup()
        {
            // must complete instantly with a clamped board, not build 9999x9999 tiles
            var state = GameFactory.Build(GameSetup.Parse("0 9999 9999 0 7"));
            Assert.That(state.Board.Tiles.Count, Is.EqualTo(24 * 24));
        }
    }
}
