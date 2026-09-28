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
        public void ResolverLoadsOnlyTheCatalogPinnedPackageAndAuthenticatesItsCheckpoint()
        {
            string root = CreateResolverProject();
            try
            {
                var selection = SharedAiModelTests.Selection();
                PlayableModelLaunch launch = PlayableModelResolver.Resolve(root, selection, null, null);
                Assert.That(launch.RunDirectory, Is.EqualTo(Path.Combine(root, "models", "ai", "focused")));
                Assert.That(launch.ControllerSpec, Is.EqualTo("run:" + launch.RunDirectory));
                File.WriteAllText(Path.Combine(launch.RunDirectory, "checkpoints", "best.pt"), "changed");
                Assert.Throws<InvalidDataException>(() => PlayableModelResolver.Resolve(root, selection, null, null));
            }
            finally { Directory.Delete(root, true); }
        }

        [Test]
        public void MissingConfiguredPackageDoesNotFallBackToAnotherRun()
        {
            string root = CreateResolverProject();
            try
            {
                File.Delete(Path.Combine(root, "models", "ai", "focused", "run.json"));
                Directory.CreateDirectory(Path.Combine(root, "python", "runs", "some-other-completed-model"));
                Assert.Throws<FileNotFoundException>(() => PlayableModelResolver.Resolve(root, SharedAiModelTests.Selection(), null, null));
            }
            finally { Directory.Delete(root, true); }
        }

        [Test]
        public void ExplicitRuntimeAndPythonPathsWorkIndependentlyOfGitWorktreeMarkers()
        {
            string root = CreateResolverProject();
            try
            {
                File.WriteAllText(Path.Combine(root, ".git"), "gitdir: /unavailable/wsl/repo/.git/worktrees/other");
                string python = Path.Combine(root, "python", "winenv", "Scripts", "python.exe");
                var launch = PlayableModelResolver.Resolve(Path.Combine(root, "unrelated"), SharedAiModelTests.Selection(), root, python);
                Assert.That(launch.PythonExecutable, Is.EqualTo(python));
                Assert.That(launch.ServerScript, Is.EqualTo(Path.Combine(root, "python", "policy_server.py")));
            }
            finally { Directory.Delete(root, true); }
        }

        [TestCase("../outside")]
        [TestCase("/absolute")]
        [TestCase("models/../../outside")]
        public void PackageTraversalIsRejected(string relative)
        {
            Assert.Throws<ArgumentException>(() => PlayableModelResolver.ContainedPath(Path.GetTempPath(), relative));
        }

        static string CreateResolverProject()
        {
            string root = Path.Combine(Path.GetTempPath(), "HexWars-ai-" + Guid.NewGuid().ToString("N"));
            string scripts = Path.Combine(root, "python", "winenv", "Scripts");
            Directory.CreateDirectory(scripts);
            File.WriteAllText(Path.Combine(scripts, "python.exe"), string.Empty);
            File.WriteAllText(Path.Combine(root, "python", "policy_server.py"), string.Empty);
            string package = Path.Combine(root, "models", "ai", "focused");
            Directory.CreateDirectory(Path.Combine(package, "checkpoints"));
            File.WriteAllText(Path.Combine(package, "checkpoints", "best.pt"), string.Empty);
            File.WriteAllText(Path.Combine(package, "run.json"), "{}");
            File.WriteAllText(Path.Combine(package, "policy-identity.json"), "{}");
            return root;
        }
    }
}
