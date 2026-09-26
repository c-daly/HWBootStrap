#nullable enable
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    [TestFixture]
    public sealed class StartupRouteTests
    {
        [Test]
        public void ASteamBuild_AlwaysReachesTheTitleWithSteam()
        {
            // The regression this exists for: the checked-in scene has Networked off, so a native
            // Steam player used to fall through to the local hotseat game and never saw the title.
            Assert.That(StartupRoute.Decide(false, true, false, false), Is.EqualTo(StartupRoute.Route.TitleWithSteam));
            Assert.That(StartupRoute.Decide(false, true, true, false), Is.EqualTo(StartupRoute.Route.TitleWithSteam));
            Assert.That(StartupRoute.Decide(false, true, false, true), Is.EqualTo(StartupRoute.Route.TitleWithSteam));
            Assert.That(StartupRoute.Decide(false, true, true, true), Is.EqualTo(StartupRoute.Route.TitleWithSteam));
        }

        [Test]
        public void ABrowserBuild_KeepsTheLegacyNetworkedTitle()
        {
            Assert.That(StartupRoute.Decide(true, false, false, false), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
            Assert.That(StartupRoute.Decide(true, false, true, false), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
            Assert.That(StartupRoute.Decide(true, false, false, true), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
            Assert.That(StartupRoute.Decide(true, false, true, true), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
        }

        [Test]
        public void AScenesNetworkedFlagOrARoomToJoin_TakesTheLegacyNetworkedTitle()
        {
            Assert.That(StartupRoute.Decide(false, false, true, false), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
            Assert.That(StartupRoute.Decide(false, false, false, true), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
            Assert.That(StartupRoute.Decide(false, false, true, true), Is.EqualTo(StartupRoute.Route.TitleWithLegacyNetwork));
        }

        [Test]
        public void APlainOfflineBuild_StartsTheLocalGame()
        {
            Assert.That(StartupRoute.Decide(false, false, false, false), Is.EqualTo(StartupRoute.Route.LocalGame));
        }

        [Test]
        public void OnlyTheLegacyNetworkRouteWithARoom_AutoJoinsIt()
        {
            Assert.That(StartupRoute.AutoJoinsRoom(StartupRoute.Route.TitleWithLegacyNetwork, true), Is.True);
            Assert.That(StartupRoute.AutoJoinsRoom(StartupRoute.Route.TitleWithLegacyNetwork, false), Is.False);
            Assert.That(StartupRoute.AutoJoinsRoom(StartupRoute.Route.TitleWithSteam, true), Is.False);
            Assert.That(StartupRoute.AutoJoinsRoom(StartupRoute.Route.LocalGame, true), Is.False);
        }
    }
}
