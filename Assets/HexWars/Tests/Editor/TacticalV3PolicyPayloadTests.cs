using System;
using System.IO;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class TacticalV3PolicyPayloadTests
    {
        [Test]
        public void Seed41Payload_PreservesTheCheckedInWireFixture()
        {
            TacticalV3Config config =
                TrainingScenario.CreateStandard("tactical-v3").BuildTacticalV3();
            TacticalV3View view =
                new TacticalV3DuelEnv(config).Reset(41, null, null);

            TacticalV3ViewDto actual = TacticalV3PolicyPayload.From(view);
            string fixturePath = Path.Combine(
                Directory.GetParent(Application.dataPath).FullName,
                "python", "tests", "fixtures", "tactical_v3",
                "seed-41-decision.json");
            TacticalV3ViewDto expected =
                JsonUtility.FromJson<TacticalV3ViewDto>(File.ReadAllText(fixturePath));

            Assert.That(JsonUtility.ToJson(actual), Is.EqualTo(JsonUtility.ToJson(expected)));
        }

        [Test]
        public void Seed41Payload_PreservesEveryReferenceAndCandidateProjection()
        {
            TacticalV3View view = new TacticalV3DuelEnv(
                TrainingScenario.CreateStandard("tactical-v3").BuildTacticalV3())
                .Reset(41, null, null);

            TacticalV3ViewDto dto = TacticalV3PolicyPayload.From(view);

            Assert.That(dto.observation.units, Has.Length.EqualTo(
                view.Decision.Observation.Units.Count));
            Assert.That(dto.observation.capability_allocations, Has.Length.EqualTo(
                view.Decision.Observation.CapabilityAllocations.Count));
            Assert.That(dto.observation.memory, Has.Length.EqualTo(
                view.Decision.Observation.Memory.Count));
            Assert.That(dto.observation.relations, Has.Length.EqualTo(
                view.Decision.Observation.Relations.Count));
            Assert.That(dto.candidates, Has.Length.EqualTo(
                view.Decision.Candidates.Count));
            for (int row = 0; row < dto.candidates.Length; row++)
            {
                Assert.That(dto.candidates[row].candidate_id, Is.EqualTo(row));
                Assert.That(dto.candidates[row].decision_id,
                    Is.EqualTo(view.Decision.DecisionId));
                Assert.That(dto.candidates[row].projection, Is.Not.Null);
            }
        }
    }
}
