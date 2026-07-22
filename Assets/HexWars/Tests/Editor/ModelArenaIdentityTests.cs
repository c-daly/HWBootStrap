using HexWars.Presentation;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class ModelArenaIdentityTests
    {
        [Test]
        public void Driver_AlwaysCarriesIndependentIdentityOverlay()
        {
            var go = new GameObject("arena", typeof(BoardRenderer), typeof(ModelDuelDriver));
            try
            {
                Assert.That(go.GetComponent<ModelArenaIdentityOverlay>(), Is.Not.Null);
                var driver = go.GetComponent<ModelDuelDriver>();
                driver.P0Spec = "greedy";
                driver.P1Spec = "random";
                Assert.That(driver.IdentitySnapshot()[0].Controller, Is.EqualTo("Greedy"));
            }
            finally { Object.DestroyImmediate(go); }
        }

        [Test]
        public void Build_LabelsScriptedSeatsAndMarksCurrentSeat()
        {
            var rows = ModelArenaIdentity.Build("greedy", "random", null, null, 1, 0, 0, 0);

            Assert.That(rows[0].Player, Is.EqualTo("P1"));
            Assert.That(rows[0].Controller, Is.EqualTo("Greedy"));
            Assert.That(rows[0].IsActive, Is.False);
            Assert.That(rows[1].Controller, Is.EqualTo("Random"));
            Assert.That(rows[1].IsActive, Is.True);
            Assert.That(rows[0].Record, Is.EqualTo("0-0-0 (—)"));
        }

        [Test]
        public void Build_UsesResolvedCheckpointAndMirrorsRecords()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"path\":\"C:/runs/alpha/checkpoints/model_20480_steps.zip\",\"algorithm\":\"maskable_ppo\",\"step\":20480}]}"
            ).Seats[0];

            var rows = ModelArenaIdentity.Build("run:C:/runs/alpha", "greedy", resolved, null, 0, 3, 1, 1);

            Assert.That(rows[0].Controller, Is.EqualTo("alpha"));
            Assert.That(rows[0].Checkpoint, Is.EqualTo("model_20480_steps.zip"));
            Assert.That(rows[0].Algorithm, Is.EqualTo("Maskable PPO"));
            Assert.That(rows[0].Step, Is.EqualTo("step 20,480"));
            Assert.That(rows[0].Record, Is.EqualTo("3-1-1 (60%)"));
            Assert.That(rows[1].Record, Is.EqualTo("1-3-1 (20%)"));
        }

        [Test]
        public void Build_HandlesIncompleteResolvedRunMetadata()
        {
            var resolved = PolicyBridge.ParseReady(
                "{\"ready\":true,\"model_seats\":[0],\"seat_models\":[{\"seat\":0,\"kind\":\"run\",\"algorithm\":\"maskable_ppo\"}]}"
            ).Seats[0];

            var row = ModelArenaIdentity.Build(null, "greedy", resolved, null, 0, 0, 0, 0)[0];

            Assert.That(row.Controller, Is.EqualTo("model"));
            Assert.That(row.Checkpoint, Is.EqualTo(string.Empty));
            Assert.That(row.Algorithm, Is.EqualTo("Maskable PPO"));
            Assert.That(row.Step, Is.EqualTo(string.Empty));
        }

        [Test]
        public void MiddleTruncate_PreservesBothEnds()
        {
            Assert.That(ModelArenaIdentity.MiddleTruncate("abcdefghijklmnop", 11), Is.EqualTo("abcd…klmnop"));
        }
    }
}
