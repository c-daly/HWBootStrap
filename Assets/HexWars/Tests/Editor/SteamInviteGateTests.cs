#nullable enable
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// An accepted Steam invite is honoured only while the title screen really is the front door.
    /// Every other case would destroy something the player can see: a live match, or a lobby screen
    /// whose coordinator holds a Steam lobby and an auth ticket nobody would ever release.
    /// </summary>
    [TestFixture]
    public sealed class SteamInviteGateTests
    {
        [Test]
        public void AnIdleTitle_AcceptsTheInvite()
        {
            Assert.That(SteamInviteGate.CanAccept(false, false, false, false), Is.True);
        }

        [Test]
        public void TheAttractDemo_IsNotAMatchAndStillAcceptsTheInvite()
        {
            Assert.That(SteamInviteGate.CanAccept(true, true, false, false), Is.True);
        }

        [Test]
        public void ARealGameOnScreen_RefusesTheInvite()
        {
            Assert.That(SteamInviteGate.CanAccept(true, false, false, false), Is.False);
        }

        [Test]
        public void AMatchSocketStillAttached_RefusesTheInvite()
        {
            // The pre-START case: the socket is up but no state exists yet, so the state check alone
            // would wave the invite through and tear the connection down under the player.
            Assert.That(SteamInviteGate.CanAccept(false, false, true, false), Is.False);
            Assert.That(SteamInviteGate.CanAccept(true, true, true, false), Is.False);
        }

        [Test]
        public void ALobbyScreenAlreadyRunning_RefusesTheInvite()
        {
            // Opening a second lobby screen orphans the first coordinator: an empty Steam lobby and a
            // live auth ticket with nothing left holding a reference to them.
            Assert.That(SteamInviteGate.CanAccept(false, false, false, true), Is.False);
            Assert.That(SteamInviteGate.CanAccept(true, true, false, true), Is.False);
        }
    }
}
