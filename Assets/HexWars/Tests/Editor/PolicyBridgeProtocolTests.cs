using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class PolicyBridgeProtocolTests
    {
        [Test]
        public void ReadyMessage_ParsesStructuredMetadataForBothSeats()
        {
            const string json = "{\"ready\":true,\"model_seats\":[0,1],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"inference_mode\":\"stochastic\",\"path\":\"a.zip\",\"algorithm\":\"maskable_ppo\",\"step\":64,\"environment\":\"adaptive-v1\",\"contract_version\":\"adaptive-v1\",\"contract_hash\":\"abc\",\"encoding_hash\":\"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"},{\"seat\":1,\"kind\":\"checkpoint\",\"path\":\"b.zip\",\"algorithm\":\"masked_dqn\",\"step\":96,\"environment\":\"tactical-v1\",\"contract_version\":\"tactical-v1\",\"contract_hash\":\"def\",\"encoding_hash\":\"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\"}]}";

            var message = PolicyBridge.ParseReady(json);

            Assert.That(message.Ready, Is.True);
            Assert.That(message.Seats, Has.Length.EqualTo(2));
            Assert.That(message.Seats[0].Seat, Is.Zero);
            Assert.That(message.Seats[0].Algorithm, Is.EqualTo("maskable_ppo"));
            Assert.That(message.Seats[0].InferenceMode, Is.EqualTo("stochastic"));
            Assert.That(message.Seats[0].ContractVersion, Is.EqualTo("adaptive-v1"));
            Assert.That(message.Seats[0].Environment, Is.EqualTo("adaptive-v1"));
            Assert.That(message.Seats[0].EncodingHash, Is.EqualTo(new string('a', 64)));
            Assert.That(message.Seats[1].Step, Is.EqualTo(96));
            Assert.That(message.Seats[1].ContractVersion, Is.EqualTo("tactical-v1"));
        }

        [Test]
        public void StartupArguments_DeclareExpectedEnvironmentVersionAndEncodingHash()
        {
            string arguments = PolicyBridge.BuildArguments(
                "policy_server.py", "run:C:/runs/a", null,
                "adaptive-v1", "adaptive-v1", new string('c', 64));

            Assert.That(arguments, Does.Contain("--expected-environment adaptive-v1"));
            Assert.That(arguments, Does.Contain("--expected-contract-version adaptive-v1"));
            Assert.That(arguments, Does.Contain("--expected-encoding-hash " + new string('c', 64)));
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
            var reload = PolicyBridge.ParseReload("{\"reloaded\":[0],\"seat_models\":[{\"seat\":0,\"algorithm\":\"maskable_ppo\",\"step\":128,\"contract_version\":\"adaptive-v1\"}]}");
            Assert.That(reload.ReloadedSeats, Is.EqualTo(new[] { 0 }));
            Assert.That(reload.Seats[0].Step, Is.EqualTo(128));
            Assert.That(reload.Seats[0].ContractVersion, Is.EqualTo("adaptive-v1"));
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

        [Test]
        public void AlternatingViewerNeverReconfiguresOrReloadsMidGame()
        {
            var config = new ModelDuelConfiguration
            {
                P0 = new ModelSeatConfiguration
                    { Kind = ModelControllerKind.LiveRun, Path = "run-a" },
                P1 = new ModelSeatConfiguration { Kind = ModelControllerKind.Greedy },
            };
            var scenario = HexWars.Engine.Rl.TrainingScenario.CreateStandard("tactical-v1");
            var previous = new MlPresentationGame(
                "learner", "greedy", 0, ModelDuelObserverSeat.Player1,
                "Greedy", scenario);
            var next = new MlPresentationGame(
                "greedy", "learner", 1, ModelDuelObserverSeat.Player2,
                "Greedy", scenario);

            Assert.That(config.ShouldReload(gameEnded: false), Is.False);
            Assert.That(ModelDuelDriver.ShouldReconfigure(previous, next, gameEnded: false),
                Is.False);
        }
    }
}
