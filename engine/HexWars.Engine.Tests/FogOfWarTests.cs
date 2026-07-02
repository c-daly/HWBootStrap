using System.Collections.Generic;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// Fog of war is a config toggle carried end-to-end: lobby wire (GameSetup) -> config -> the
    /// START/replay wire (ReplayFile CONFIG line). The hiding itself is presentation-side — the
    /// engine's army-vision rule (TargetingService.IsVisibleToArmy) already gates attacks.
    /// </summary>
    public class FogOfWarTests
    {
        [Test]
        public void Fog_DefaultsOff()
        {
            Assert.That(GameConfig.Default().FogOfWar, Is.False);
            Assert.That(GameFactory.Build(GameSetup.Default).Config.FogOfWar, Is.False);
        }

        [Test]
        public void GameSetup_Fog_RoundTripsTheLobbyWire()
        {
            var setup = new GameSetup(GameMode.Territory, 11, 9, 40, 7, turnActions: 3, fog: true);
            var parsed = GameSetup.Parse(setup.ToWire());
            Assert.That(parsed.Fog, Is.True);
            Assert.That(GameFactory.Build(parsed).Config.FogOfWar, Is.True);

            // an old 10-field wire line (no fog) parses with fog off
            var old = GameSetup.Parse("1 11 9 40 7 3 1 1 1 3");
            Assert.That(old.Fog, Is.False);
        }

        [Test]
        public void Fog_RoundTripsTheReplayWire()
        {
            var start = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7, fog: true));
            var s = ReplayFile.Read(ReplayFile.Write(start, new List<Command>())).Start;
            Assert.That(s.Config.FogOfWar, Is.True, "fog must survive the START payload");
        }
    }
}
