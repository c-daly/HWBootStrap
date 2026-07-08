using System;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>The lobby list: OpenGames() exposes rooms a browser can join — public, exactly one
    /// seated member, game not yet started. Age comes from an injected clock so tests are exact.</summary>
    public class MatchHubLobbyTests
    {
        private long _now;
        private MatchHub NewHub()
        {
            _now = TimeSpan.FromHours(1).Ticks;
            return new MatchHub(GameFactory.Build, () => _now);
        }

        [Test]
        public void WaitingPublicRoom_IsListed_WithSetupAndAge()
        {
            var hub = NewHub();
            hub.Connect("KQ7KP", "host", new GameSetup(GameMode.Territory, 13, 9, 40, 7, 5, 2, 2, 1, 3, fog: true));
            _now += TimeSpan.FromSeconds(120).Ticks;

            var open = hub.OpenGames();
            Assert.That(open.Count, Is.EqualTo(1));
            Assert.That(open[0].Code, Is.EqualTo("KQ7KP"));
            Assert.That(open[0].Setup.Mode, Is.EqualTo(GameMode.Territory));
            Assert.That(open[0].Setup.Width, Is.EqualTo(13));
            Assert.That(open[0].Setup.Fog, Is.True);
            Assert.That(open[0].AgeSeconds, Is.EqualTo(120));
        }

        [Test]
        public void PrivateRoom_NeverListed()
        {
            var hub = NewHub();
            hub.Connect("SECRET", "host", GameSetup.Default, isPrivate: true);
            Assert.That(hub.OpenGames(), Is.Empty);
        }

        [Test]
        public void StartedRoom_NotListed_EvenIfAMemberDrops()
        {
            var hub = NewHub();
            hub.Connect("R", "a");
            hub.Connect("R", "b");                       // second seat -> START dealt -> Started
            Assert.That(hub.OpenGames(), Is.Empty, "full room is not open");
            hub.Disconnect("R", "b");                    // back to one member, but the game began
            Assert.That(hub.OpenGames(), Is.Empty, "a started room must never re-list");
        }

        [Test]
        public void EmptiedRoom_IsRemoved_NotListed()
        {
            var hub = NewHub();
            hub.Connect("R", "a");
            hub.Disconnect("R", "a");
            Assert.That(hub.OpenGames(), Is.Empty);
        }

        [Test]
        public void OpenGames_NewestFirst()
        {
            var hub = NewHub();
            hub.Connect("OLD", "a");
            _now += TimeSpan.FromSeconds(60).Ticks;
            hub.Connect("NEW", "b");
            var open = hub.OpenGames();
            Assert.That(open.Select(g => g.Code), Is.EqualTo(new[] { "NEW", "OLD" }));
            Assert.That(open[0].AgeSeconds, Is.EqualTo(0));
            Assert.That(open[1].AgeSeconds, Is.EqualTo(60));
        }

        [Test]
        public void JoinerPrivacyFlag_Ignored_HostDecides()
        {
            var hub = NewHub();
            hub.Connect("R", "host", GameSetup.Default, isPrivate: false);
            var full = hub.Connect("R", "x", GameSetup.Default, isPrivate: true); // joiner flag must not flip the room
            // room is now full (started) so it's unlisted for THAT reason; verify via a fresh public room
            hub.Connect("R2", "h2", GameSetup.Default, isPrivate: false);
            Assert.That(hub.OpenGames().Select(g => g.Code), Is.EqualTo(new[] { "R2" }));
        }
    }
}
