using System;
using System.IO;
using System.Reflection;
using HexWars.Engine.Rl;
using HexWars.Presentation;
using Newtonsoft.Json.Linq;
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
        [Test]
        public void StructuredRequest_PreservesNullControllerAndEveryTokenReference()
        {
            TacticalV3View view = new TacticalV3DuelEnv(
                TrainingScenario.CreateStandard("tactical-v3").BuildTacticalV3())
                .Reset(41, null, null);
            TacticalV3ViewDto decision = TacticalV3PolicyPayload.From(view);
            JObject request = JObject.Parse(PolicyJson.Serialize(
                new TacticalV3PolicyRequestDto
                {
                    seat = decision.seat,
                    decision = decision,
                }));

            Assert.That(request.Count, Is.EqualTo(2));
            Assert.That(request["seat"].Value<int>(), Is.EqualTo(decision.seat));
            Assert.That(request["decision"]["decision_id"].Value<long>(),
                Is.EqualTo(decision.decision_id));

            int unownedCell = Array.FindIndex(decision.observation.cells,
                cell => cell.controller == null);
            Assert.That(unownedCell, Is.GreaterThanOrEqualTo(0));
            Assert.That(request["decision"]["observation"]["cells"][unownedCell]
                ["controller"].Type, Is.EqualTo(JTokenType.Null));

            AssertTokenReferencesMatch(decision, request["decision"]);
        }

        static void AssertTokenReferencesMatch(object value, JToken json)
        {
            if (value is Array values)
            {
                var jsonValues = (JArray)json;
                Assert.That(jsonValues.Count, Is.EqualTo(values.Length));
                for (int index = 0; index < values.Length; index++)
                    AssertTokenReferencesMatch(values.GetValue(index), jsonValues[index]);
                return;
            }

            if (value == null) return;
            var jsonObject = (JObject)json;
            foreach (FieldInfo field in value.GetType().GetFields(
                BindingFlags.Instance | BindingFlags.Public))
            {
                object fieldValue = field.GetValue(value);
                JToken fieldJson = jsonObject[field.Name];
                if (field.FieldType == typeof(TacticalV3TokenReferenceDto))
                {
                    if (fieldValue == null)
                    {
                        Assert.That(fieldJson.Type, Is.EqualTo(JTokenType.Null),
                            field.Name + " must remain a JSON null");
                        continue;
                    }

                    var reference = (TacticalV3TokenReferenceDto)fieldValue;
                    var jsonReference = (JObject)fieldJson;
                    Assert.That(jsonReference.Count, Is.EqualTo(2));
                    Assert.That(jsonReference["table"].Value<string>(),
                        Is.EqualTo(reference.table));
                    Assert.That(jsonReference["row"].Value<int>(),
                        Is.EqualTo(reference.row));
                    continue;
                }

                if (fieldValue is Array || (fieldValue != null &&
                    field.FieldType.Namespace == "HexWars.Presentation"))
                    AssertTokenReferencesMatch(fieldValue, fieldJson);
            }
        }
    }
}
