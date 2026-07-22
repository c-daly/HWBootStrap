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
        public void SeatStepMetadata_DistinguishesMissingFromExplicitZero()
        {
            var missing = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0}]}"
            ).Seats[0];
            var zero = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"step\":0}]}"
            ).Seats[0];

            Assert.That(missing.HasStep, Is.False);
            Assert.That(zero.HasStep, Is.True);
            Assert.That(ModelArenaIdentity.Build("ppo:a.zip", "greedy", missing, null, -1, 0, 0, 0)[0].Step,
                Is.EqualTo("step unknown"));
            Assert.That(ModelArenaIdentity.Build("ppo:a.zip", "greedy", zero, null, -1, 0, 0, 0)[0].Step,
                Is.EqualTo("step 0"));
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
        public void SeatsAfterReload_RetainsCurrentSeatsOnErrorAndAdvancesOnSuccess()
        {
            var current = PolicyBridge.ParseReady(
                "{\"ready\":true,\"seat_models\":[{\"seat\":0,\"step\":128}]}"
            ).Seats;
            var failed = PolicyBridge.ParseReload(
                "{\"error\":\"reload failed\",\"seat_models\":[{\"seat\":0,\"step\":256}]}"
            );

            Assert.That(PolicyBridge.SeatsAfterReload(current, failed), Is.SameAs(current));

            var succeeded = PolicyBridge.ParseReload(
                "{\"reloaded\":[0],\"seat_models\":[{\"seat\":0,\"step\":256}]}"
            );
            var advanced = PolicyBridge.SeatsAfterReload(current, succeeded);
            Assert.That(advanced, Is.SameAs(succeeded.Seats));
            Assert.That(advanced[0].Step, Is.EqualTo(256));
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
