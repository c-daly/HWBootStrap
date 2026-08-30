using System;
using System.IO;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class PlayableModelOpponentTests
    {
        [Test]
        public void NormalNineBySevenMatch_ProducesStructuredFrameAndRoundTripsSelection()
        {
            GameState start = GameFactory.BuildTacticalV3Compatible(
                new GameSetup(GameMode.Annihilation, 9, 7, 0, 7));
            var adapter = new PlayableModelAdapter(start, PlayerId.Player1);
            Result passed = GameEngine.Apply(start, new EndTurn(PlayerId.Player0));
            Assert.That(passed.Success, Is.True);

            TacticalV3DecisionFrame frame = adapter.CreateFrame(
                passed.NewState, PlayerId.Player1, 41);
            TacticalV3ViewDto payload = TacticalV3PolicyPayload.From(frame);
            TacticalV3Candidate endTurn = frame.Candidates.Single(
                candidate => candidate.Kind == TacticalV3CandidateKind.EndTurn);
            PolicyCandidateResult selected = PolicyBridge.ParseStructuredAction(
                "{\"decision_id\":41,\"candidate_id\":" + endTurn.CandidateId + "}", 41);
            Command command = adapter.Resolve(frame, selected, passed.NewState);

            Assert.That(payload.observation.cells, Has.Length.EqualTo(63));
            Assert.That(payload.seat, Is.EqualTo((int)PlayerId.Player1));
            Assert.That(payload.reward.total, Is.Zero);
            Assert.That(payload.reward.finalized, Is.False);
            Assert.That(payload.start_profile, Is.EqualTo("playable-game"));
            Assert.That(command, Is.InstanceOf<EndTurn>());
            Assert.That(GameEngine.Apply(passed.NewState, command).Success, Is.True);
        }

        [Test]
        public void SelectionFailsClosedWhenTheLiveStateHasChanged()
        {
            GameState start = GameFactory.BuildTacticalV3Compatible(GameSetup.Default);
            var adapter = new PlayableModelAdapter(start, PlayerId.Player0);
            TacticalV3DecisionFrame frame = adapter.CreateFrame(
                start, PlayerId.Player0, 9);
            TacticalV3Candidate endTurn = frame.Candidates.Single(
                candidate => candidate.Kind == TacticalV3CandidateKind.EndTurn);
            PolicyCandidateResult selected = PolicyBridge.ParseStructuredAction(
                "{\"decision_id\":9,\"candidate_id\":" + endTurn.CandidateId + "}", 9);
            GameState changed = GameEngine.Apply(
                start, new EndTurn(PlayerId.Player0)).NewState;

            Assert.Throws<InvalidOperationException>(() =>
                adapter.Resolve(frame, selected, changed));
        }

        [Test]
        public void SetupCompatibilityRejectsTerritoryFogAndOversizedBoards()
        {
            Assert.That(PlayableModelAdapter.Supports(GameSetup.Default, out _), Is.True);
            Assert.That(PlayableModelAdapter.Supports(
                new GameSetup(GameMode.Territory, 9, 7, 0, 7), out string territory), Is.False);
            Assert.That(territory, Does.Contain("Annihilation"));
            Assert.That(PlayableModelAdapter.Supports(
                new GameSetup(GameMode.Annihilation, 9, 7, 0, 7, fog: true), out string fog), Is.False);
            Assert.That(fog, Does.Contain("fog"));
            Assert.That(PlayableModelAdapter.Supports(
                new GameSetup(GameMode.Annihilation, 64, 64, 0, 7), out string size), Is.False);
            Assert.That(size, Does.Contain("512"));
        }

        [Test]
        public void AdapterRejectsOrdinaryCaptureEnabledAnnihilationState()
        {
            GameState ordinary = GameFactory.Build(GameSetup.Default);
            Assert.Throws<InvalidOperationException>(() =>
                new PlayableModelAdapter(ordinary, PlayerId.Player1));
        }

        [Test]
        public void CapacityGuardStopsCommandsBeforeTheyOverflowTheNextObservation()
        {
            GameState state = GameFactory.BuildTacticalV3Compatible(GameSetup.Default);
            TacticalV3CapacityProfile capacity =
                TacticalV3CapacityProfile.ExperimentalDefault();
            var templates = Enumerable.Range(0, capacity.MaxTemplates)
                .Select(index => new UnitTemplate(
                    "Template " + index,
                    new UnitStats(index + 1, 0, 0, 0, 0, 0, 0, 0, 0)))
                .ToArray();
            PlayerState p0 = state.Player(PlayerId.Player0);
            var players = state.Players.ToArray();
            players[0] = new PlayerState(
                p0.Id, p0.Points, templates, p0.UnitsOnBoard, p0.Generators,
                p0.DestroyedValue);
            state = new GameState(
                state.Board, state.Config, players, state.ActivePlayer, state.Round,
                state.NextEntityId);

            Assert.That(PlayableModelAdapter.PreservesCapacity(
                state,
                new CreateUnit(
                    PlayerId.Player0,
                    new UnitStats(1, 0, 0, 0, 0, 0, 0, 0, 0)),
                out string reason), Is.False);
            Assert.That(reason, Does.Contain("32-template"));
            Assert.That(PlayableModelAdapter.PreservesCapacity(
                state, new EndTurn(PlayerId.Player0), out _), Is.True);
        }

        [Test]
        public void ResolverUsesTheSelectedLineageDeclaredSourcePolicy()
        {
            string root = CreateResolverProject();
            try
            {
                CreateCompletedModel(root, "declared-source");
                WriteInventory(root, published: null, source: "declared-source");

                PlayableModelLaunch launch = PlayableModelResolver.Resolve(root);

                Assert.That(launch.ModelName, Is.EqualTo("declared-source"));
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        [Test]
        public void ResolverDoesNotHideAMalformedSelectedLineageBehindThePinnedFallback()
        {
            string root = CreateResolverProject();
            try
            {
                CreateCompletedModel(root, PlayableModelResolver.PinnedCompletedModel);
                string selected = Path.Combine(
                    root, "python", "runs", PlayableModelResolver.SelectedLineage);
                Directory.CreateDirectory(selected);
                File.WriteAllText(Path.Combine(selected, "run.json"), "{}");

                Assert.Throws<InvalidDataException>(() =>
                    PlayableModelResolver.Resolve(root));
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        [Test]
        public void ResolverTreatsADeclaredPublicationAsAuthoritativeAndFailsClosed()
        {
            string root = CreateResolverProject();
            try
            {
                CreateCompletedModel(root, PlayableModelResolver.PinnedCompletedModel);
                CreateCompletedModel(root, "declared-source");
                WriteInventory(root, published: "missing-publication", source: "declared-source");

                Assert.Throws<DirectoryNotFoundException>(() =>
                    PlayableModelResolver.Resolve(root));
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        [Test]
        public void ResolverRejectsCheckpointTraversalBeforeLaunchingPython()
        {
            string root = CreateResolverProject();
            try
            {
                CreateCompletedModel(
                    root, "unsafe-checkpoint", latestCheckpoint: "../outside.pt");
                WriteInventory(root, published: "unsafe-checkpoint", source: null);

                Assert.Throws<InvalidDataException>(() =>
                    PlayableModelResolver.Resolve(root));
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        [Test]
        public void ResolverUsesPinnedModelOnlyWhenSelectedLineageIsAbsent()
        {
            string root = CreateResolverProject();
            try
            {
                CreateCompletedModel(root, PlayableModelResolver.PinnedCompletedModel);

                PlayableModelLaunch launch = PlayableModelResolver.Resolve(root);

                Assert.That(launch.ModelName,
                    Is.EqualTo(PlayableModelResolver.PinnedCompletedModel));
            }
            finally
            {
                Directory.Delete(root, recursive: true);
            }
        }

        static string CreateResolverProject()
        {
            string root = Path.Combine(
                Path.GetTempPath(), "HexWars-playable-model-" + Guid.NewGuid().ToString("N"));
            string python = Path.Combine(root, "python");
            string scripts = Path.Combine(python, "winenv", "Scripts");
            Directory.CreateDirectory(scripts);
            Directory.CreateDirectory(Path.Combine(python, "runs"));
            File.WriteAllText(Path.Combine(scripts, "python.exe"), string.Empty);
            File.WriteAllText(Path.Combine(python, "policy_server.py"), string.Empty);
            return root;
        }

        static void WriteInventory(string root, string published, string source)
        {
            string selected = Path.Combine(
                root, "python", "runs", PlayableModelResolver.SelectedLineage);
            Directory.CreateDirectory(selected);
            string publishedJson = published == null ? "null" : "\"" + Json(published) + "\"";
            string sourceJson = source == null
                ? "null"
                : "{\"run\":\"" + Json(source) + "\"}";
            File.WriteAllText(Path.Combine(selected, "run.json"),
                "{\"schema_version\":1," +
                "\"config\":{\"algorithm\":\"structured_dagger\"}," +
                "\"contract\":{\"environment\":\"tactical-v3\"}," +
                "\"published_run\":" + publishedJson + "," +
                "\"source_policy\":" + sourceJson + "}");
        }

        static void CreateCompletedModel(
            string root, string name, string latestCheckpoint = "checkpoints/best.pt")
        {
            const string contractHash =
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
            const string encodingHash =
                "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
            const string capacityHash =
                "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
            string run = Path.Combine(root, "python", "runs", name);
            Directory.CreateDirectory(Path.Combine(run, "checkpoints"));
            File.WriteAllText(Path.Combine(run, "checkpoints", "best.pt"), string.Empty);
            File.WriteAllText(Path.Combine(run, "policy-identity.json"),
                "{\"contract_version\":\"tactical-v3\"," +
                "\"environment_kind\":\"duel\"," +
                "\"contract_hash\":\"" + contractHash + "\"," +
                "\"encoding_hash\":\"" + encodingHash + "\"," +
                "\"capacity_hash\":\"" + capacityHash + "\"}");
            File.WriteAllText(Path.Combine(run, "run.json"),
                "{\"schema_version\":2,\"state\":\"completed\"," +
                "\"config\":{\"algorithm\":\"structured_imitation\"}," +
                "\"contract\":{\"environment\":\"tactical-v3\"," +
                "\"version\":\"tactical-v3\",\"environment_kind\":\"duel\"," +
                "\"contract_hash\":\"" + contractHash + "\"," +
                "\"encoding_hash\":\"" + encodingHash + "\"," +
                "\"capacity_hash\":\"" + capacityHash + "\"}," +
                "\"policy_identity\":\"policy-identity.json\"," +
                "\"latest_checkpoint\":\"" + Json(latestCheckpoint) + "\"}");
        }

        static string Json(string value) =>
            value.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }
}
