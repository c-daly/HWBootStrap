using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class PolicyBridgeProtocolTests
    {
        [Test]
        public void ReadyMessage_ParsesStructuredMetadataForBothSeats()
        {
            const string json = "{\"ready\":true,\"model_seats\":[0,1],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"path\":\"a.zip\",\"algorithm\":\"maskable_ppo\",\"step\":64,\"contract_hash\":\"abc\"},{\"seat\":1,\"kind\":\"checkpoint\",\"path\":\"b.zip\",\"algorithm\":\"masked_dqn\",\"step\":96,\"contract_hash\":\"def\"}]}";

            var message = PolicyBridge.ParseReady(json);

            Assert.That(message.Ready, Is.True);
            Assert.That(message.Seats, Has.Length.EqualTo(2));
            Assert.That(message.Seats[0].Seat, Is.Zero);
            Assert.That(message.Seats[0].Algorithm, Is.EqualTo("maskable_ppo"));
            Assert.That(message.Seats[1].Step, Is.EqualTo(96));
        }

        [Test]
        public void ActionAndReloadMessages_ParseWithoutSubstringOrManualIntegerLogic()
        {
            Assert.That(PolicyBridge.ParseAction("{\"action\":123}").Action, Is.EqualTo(123));
            var reload = PolicyBridge.ParseReload("{\"reloaded\":[0],\"seat_models\":[{\"seat\":0,\"algorithm\":\"maskable_ppo\",\"step\":128}]}");
            Assert.That(reload.ReloadedSeats, Is.EqualTo(new[] { 0 }));
            Assert.That(reload.Seats[0].Step, Is.EqualTo(128));
        }

        [Test]
        public void ErrorMessage_PreservesServerErrorAndStderrTail()
        {
            var error = PolicyBridge.ParseAction("{\"error\":\"contract mismatch\"}");

            Assert.That(error.Error, Is.EqualTo("contract mismatch"));
            Assert.Throws<System.InvalidOperationException>(() => error.RequireAction("last stderr"),
                "contract mismatch\nlast stderr");
        }
    }
}
