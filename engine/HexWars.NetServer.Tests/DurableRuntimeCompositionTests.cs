using HexWars.NetServer.Runtime;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.Extensions.DependencyInjection;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The durable runtime has to come out of the real container, not just out of a constructor a test
    /// calls. It is the piece the websocket route will resolve per connection, so a registration that is
    /// missing, or one whose dependencies are only registered on the Postgres path, would fail at the first
    /// player rather than at startup.
    /// </summary>
    [TestFixture]
    public class DurableRuntimeCompositionTests
    {
        [Test]
        public void TheHostResolvesOneCoordinatorForTheWholeProcess()
        {
            using var factory = new SteamServerFactory();
            using HttpClient client = factory.CreateClient();

            var first = factory.Services.GetRequiredService<DurableMatchCoordinator>();
            var second = factory.Services.GetRequiredService<DurableMatchCoordinator>();

            Assert.That(first, Is.SameAs(second), "a per-request coordinator would hold no match at all");
            Assert.That(factory.Services.GetRequiredService<ILiveMatchLoader>(),
                Is.TypeOf<JournalLiveMatchLoader>());
            Assert.That(factory.Services.GetRequiredService<IConnectionSink>(), Is.Not.Null);
        }
    }
}
