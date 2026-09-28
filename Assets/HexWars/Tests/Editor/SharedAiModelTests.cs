using System;
using System.Threading;
using HexWars.Engine;
using HexWars.Engine.AI;
using Newtonsoft.Json;
using NUnit.Framework;
using UnityEngine;

namespace HexWars.Presentation.Tests
{
    public sealed class SharedAiModelTests
    {
        internal static AiModelCatalog Catalog() => new AiModelCatalog
        {
            default_difficulty_id = "hard",
            difficulties = new[]
            {
                new AiDifficultyDefinition { id="easy", label="Relaxed", model_id="random" },
                new AiDifficultyDefinition { id="normal", label="Standard", model_id="greedy" },
                new AiDifficultyDefinition { id="hard", label="Veteran", model_id="focused" },
            }, catalog_revision = new string('a', 64),
            models = new[]
            {
                new AiModelDefinition { id="focused", label="Focused model", kind="trained", package="focused",
                    checkpoint_sha256=AiModelSettings.Sha256(Array.Empty<byte>()), contract_version="tactical-v3",
                    encoding_hash=new string('b',64), capacity_hash=new string('c',64) },
                new AiModelDefinition { id="greedy", label="Greedy", kind="greedy" },
                new AiModelDefinition { id="random", label="Random", kind="random" },
            },
        };
        internal static AiModelSelection Selection() => new AiModelSelection(Catalog(), null, null);

        [Test]
        public void NewSelectionUsesDeploymentDefaultAndOwnsACopy()
        {
            var catalog = Catalog();
            var match = new AiModelSelection(catalog, null, null);
            catalog.default_difficulty_id = "normal";
            catalog.models[0].checkpoint_sha256 = new string('d',64);
            Assert.That(match.Model.id, Is.EqualTo("focused"));
            Assert.That(match.DifficultyId, Is.EqualTo("hard"));
            Assert.That(match.DisplayName, Is.EqualTo("Veteran"));
            Assert.That(match.Model.checkpoint_sha256, Is.EqualTo(AiModelSettings.Sha256(Array.Empty<byte>())));
            Assert.That(new AiModelSelection(catalog, null, null).Model.id, Is.EqualTo("greedy"));
            Assert.That(new AiModelSelection(catalog, "easy", null).Model.id, Is.EqualTo("random"));
        }

        [TestCase("difficulty_id")]
        [TestCase("model_id")]
        [TestCase("catalog_revision")]
        [TestCase("checkpoint_sha256")]
        [TestCase("decision_id")]
        [TestCase("candidate_id")]
        public void HostedReplyRejectsWrongPinsOrStaleIdentity(string field)
        {
            var selected = Selection();
            var json = Newtonsoft.Json.Linq.JObject.Parse(Reply(selected));
            json[field] = field == "decision_id" || field == "candidate_id"
                ? (Newtonsoft.Json.Linq.JToken)(-1) : "wrong";
            Assert.Throws<InvalidOperationException>(() => AiPolicySession.ParseHostedSelection(json.ToString(), selected, 7));
        }

        [Test]
        public void DifficultyMappingAndNamesAreEntirelyConfiguredAndLiveMatchesKeepTheirPins()
        {
            var catalog = Catalog();
            var original = new AiModelSelection(catalog, "easy", null);
            catalog.difficulties[0].label = "Custom name";
            catalog.difficulties[0].model_id = "focused";
            var next = new AiModelSelection(catalog, "easy", null);
            Assert.That(next.DisplayName, Is.EqualTo("Custom name"));
            Assert.That(next.Model.IsTrained, Is.True);
            Assert.That(original.DisplayName, Is.EqualTo("Relaxed"));
            Assert.That(original.Model.kind, Is.EqualTo("random"));
        }

        [Test]
        public void HostedReplyRequiresAllFieldsAndDoesNotAcceptUnknownFields()
        {
            var selected = Selection();
            Assert.That(AiPolicySession.ParseHostedSelection(Reply(selected), selected, 7).CandidateId, Is.EqualTo(3));
            var json = Newtonsoft.Json.Linq.JObject.Parse(Reply(selected));
            json.Remove("candidate_id");
            Assert.Throws<JsonSerializationException>(() => AiPolicySession.ParseHostedSelection(json.ToString(), selected, 7));
            json["candidate_id"] = 3; json["fallback"] = "greedy";
            Assert.Throws<JsonSerializationException>(() => AiPolicySession.ParseHostedSelection(json.ToString(), selected, 7));
        }

        [Test]
        public void ChangedStateDiscardsCompletedInference()
        {
            GameState start = GameFactory.BuildTacticalV3Compatible(GameSetup.Default);
            AiPolicySession.RequireCurrentState(start, start);
            GameState changed = GameEngine.Apply(start, new EndTurn(PlayerId.Player0)).NewState;
            Assert.Throws<OperationCanceledException>(() => AiPolicySession.RequireCurrentState(start, changed));
        }

        [Test]
        public void ExplicitScriptedSessionHonorsCancellationWithoutLaunchingPython()
        {
            var selection = new AiModelSelection(Catalog(), "normal", null);
            using (var session = new AiPolicySession(selection, PlayerId.Player0, 7))
            {
                var state = GameFactory.BuildTacticalV3Compatible(GameSetup.Default);
                Assert.ThrowsAsync(Is.InstanceOf<OperationCanceledException>(), async () =>
                    await session.DecideAsync(state, new CancellationToken(true)));
            }
        }

        [Test]
        public void BrowserUsesPageOriginAndDesktopEndpointRejectsCredentials()
        {
            Assert.That(AiModelSettings.ResolveServerUrl("https://game.example/play/?room=x", "https://other", true),
                Is.EqualTo("https://game.example"));
            Assert.That(AiModelSettings.ResolveServerUrl("", "http://localhost:8000/", false),
                Is.EqualTo("http://localhost:8000"));
            Assert.Throws<InvalidOperationException>(() => AiModelSettings.ResolveServerUrl("", "https://secret@example.com", false));
            Assert.Throws<InvalidOperationException>(() => AiModelSettings.ResolveServerUrl("file:///game", null, true));
        }

        [Test]
        public void CatalogLoadRecoversAfterFailureAndRefreshesOnlyAtExplicitBoundary()
        {
            var flags = System.Reflection.BindingFlags.Static | System.Reflection.BindingFlags.NonPublic;
            var fields = new[] { "_catalog", "_loading", "_serverUrl" };
            var previous = new object[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                var field = typeof(AiModelSettings).GetField(fields[i], flags);
                previous[i] = field.GetValue(null); field.SetValue(null, null);
            }
            var errorProperty = typeof(AiModelSettings).GetProperty("Error");
            object previousError = errorProperty.GetValue(null);
            string oldPath = Environment.GetEnvironmentVariable("HEXWARS_AI_CONFIG_PATH");
            string oldServer = Environment.GetEnvironmentVariable("HEXWARS_AI_SERVER_URL");
            string path = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "hexwars-catalog-" + Guid.NewGuid().ToString("N") + ".json");
            try
            {
                Environment.SetEnvironmentVariable("HEXWARS_AI_SERVER_URL", null);
                Environment.SetEnvironmentVariable("HEXWARS_AI_CONFIG_PATH", path);
                Assert.Throws<System.IO.FileNotFoundException>(() => AiModelSettings.LoadAsync().GetAwaiter().GetResult());
                Assert.That(AiModelSettings.IsLoaded, Is.False);
                Assert.That(AiModelSettings.Error, Does.Contain("unavailable"));
                var catalog = Catalog();
                // Serialize fields only, matching the canonical deployment catalog contract.
                string json = JsonUtility.ToJson(catalog);
                System.IO.File.WriteAllText(path, json);
                AiModelSettings.LoadAsync().GetAwaiter().GetResult();
                var liveMatch = AiModelSettings.Capture();
                Assert.That(liveMatch.Model.id, Is.EqualTo("focused"));
                Assert.That(AiModelSettings.Error, Is.Null);
                catalog.default_difficulty_id = "normal";
                System.IO.File.WriteAllText(path, JsonUtility.ToJson(catalog));
                AiModelSettings.LoadAsync().GetAwaiter().GetResult();
                Assert.That(AiModelSettings.Capture().Model.id, Is.EqualTo("focused"));
                AiModelSettings.LoadAsync(refresh: true).GetAwaiter().GetResult();
                Assert.That(AiModelSettings.Capture().Model.id, Is.EqualTo("greedy"));
                Assert.That(liveMatch.Model.id, Is.EqualTo("focused"));
                Assert.That(AiModelSettings.Capture().CatalogRevision, Is.Not.EqualTo(liveMatch.CatalogRevision));
            }
            finally
            {
                Environment.SetEnvironmentVariable("HEXWARS_AI_CONFIG_PATH", oldPath);
                Environment.SetEnvironmentVariable("HEXWARS_AI_SERVER_URL", oldServer);
                if (System.IO.File.Exists(path)) System.IO.File.Delete(path);
                for (int i = 0; i < fields.Length; i++) typeof(AiModelSettings).GetField(fields[i], flags).SetValue(null, previous[i]);
                errorProperty.GetSetMethod(true).Invoke(null, new[] { previousError });
            }
        }

        [Test]
        public void DesktopCatalogOverrideIsExplicitAndIndependentOfRuntimePackageRoot()
        {
            Assert.That(AiModelSettings.CatalogPath("assets", null), Is.EqualTo(System.IO.Path.Combine("assets", "ai-models.json")));
            string configured = System.IO.Path.Combine(System.IO.Path.GetTempPath(), "deployment.json");
            Assert.That(AiModelSettings.CatalogPath("assets", configured), Is.EqualTo(configured));
        }

        [Test]
        public void PackagedCheckpointIdentitySurvivesThePythonReadyHandshake()
        {
            string digest = new string('e', 64);
            var ready = PolicyBridge.ParseReady("{\"ready\":true,\"seat_models\":[{\"seat\":0,\"checkpoint_sha256\":\"" + digest + "\"}]}");
            Assert.That(ready.Seats[0].CheckpointSha256, Is.EqualTo(digest));
        }

        [Test]
        public void DefaultEntryPointsFollowCatalogAndExplicitLabSpecsRemainPinned()
        {
            var config = new ModelDuelConfiguration();
            Assert.That(config.P0.BuildSpec(), Is.EqualTo("configured"));
            Assert.That(config.P1.BuildSpec(), Is.EqualTo("configured"));
            Assert.That(config.Validate(), Is.Empty);
            var plan = EditorTools.MlLab.MlArenaLaunchPlan.Create(config);
            Assert.That(plan.P0Spec, Is.EqualTo("configured"));
            Assert.That(plan.P1Spec, Is.EqualTo("configured"));
            Assert.That(plan.Scenario.Environment, Is.EqualTo("tactical-v3"));
            Assert.That(new ModelSeatConfiguration { Kind=ModelControllerKind.FixedRun, Path="chosen" }.BuildSpec(), Is.EqualTo("run:chosen"));
            Assert.That(new ModelSeatConfiguration { Kind=ModelControllerKind.Greedy }.BuildSpec(), Is.EqualTo("greedy"));
        }

        static string Reply(AiModelSelection selected) => JsonConvert.SerializeObject(new
        {
            difficulty_id=selected.DifficultyId, model_id=selected.Model.id, catalog_revision=selected.CatalogRevision,
            checkpoint_sha256=selected.Model.checkpoint_sha256, decision_id=7, candidate_id=3,
        });
    }
}
