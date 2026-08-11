using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3DuelEnvTests
    {
        [Test]
        public void Reset_SameSeedProducesSameStateDecisionAndCandidateOrder()
        {
            TacticalV3DuelEnv first = TacticalV3Fixtures.Env();
            TacticalV3DuelEnv second = TacticalV3Fixtures.Env();

            TacticalV3View firstView = first.Reset(31, null, null);
            TacticalV3View secondView = second.Reset(31, null, null);

            Assert.Multiple(() =>
            {
                Assert.That(StateSignature(second.State), Is.EqualTo(StateSignature(first.State)));
                Assert.That(secondView.Decision.DecisionId, Is.EqualTo(0));
                Assert.That(secondView.Decision.DecisionId, Is.EqualTo(firstView.Decision.DecisionId));
                Assert.That(secondView.Seat, Is.EqualTo(firstView.Seat));
                Assert.That(secondView.Decision.Candidates.Select(CandidateKey),
                    Is.EqualTo(firstView.Decision.Candidates.Select(CandidateKey)));
            });
        }

        [Test]
        public void SameSeedTrajectoryMatchesObservationsCandidatesCommandsRewardsTerminalAndReplay()
        {
            TacticalV3Config firstConfig = TacticalV3Fixtures.Config();
            TacticalV3Config secondConfig = TacticalV3Fixtures.Config();
            firstConfig.Match.MaxSteps = 10;
            secondConfig.Match.MaxSteps = 10;
            var firstEnv = new TacticalV3DuelEnv(firstConfig);
            var secondEnv = new TacticalV3DuelEnv(secondConfig);
            TacticalV3View first = firstEnv.Reset(149, null, null);
            TacticalV3View second = secondEnv.Reset(149, null, null);

            Assert.Throws<AssertionException>(() => AssertReferenceValid(
                new TacticalV3TokenRef(
                    TacticalV3TableKind.Cells, first.Decision.Observation.Cells.Count),
                RowCounts(first)));
            Assert.Throws<AssertionException>(() => AssertReferenceValid(
                new TacticalV3TokenRef(TacticalV3TableKind.Units, 0),
                TacticalV3TableKind.Cells,
                RowCounts(first)));
            int step = 0;
            while (!first.Terminated && !first.Truncated)
            {
                AssertDeterministicView(first, second);
                AssertAllReferencesValid(first);
                AssertAllReferencesValid(second);
                int candidateId = (step * 17 + 3) % first.Decision.Candidates.Count;
                Assert.That(JsonSerializer.Serialize(second.Decision.Candidates[candidateId]),
                    Is.EqualTo(JsonSerializer.Serialize(first.Decision.Candidates[candidateId])));

                first = firstEnv.Step(first.Decision.DecisionId, candidateId);
                second = secondEnv.Step(second.Decision.DecisionId, candidateId);
                AssertAllReferencesValid(first);
                AssertAllReferencesValid(second);
                ReplayData firstReplay = ReplayFile.Read(firstEnv.ToReplay());
                ReplayData secondReplay = ReplayFile.Read(secondEnv.ToReplay());
                Assert.That(firstReplay.Commands, Has.Count.EqualTo(step + 1));
                AssertCommandsEqual(firstReplay.Commands[step], secondReplay.Commands[step]);
                step++;
            }

            AssertDeterministicView(first, second);
            AssertAllReferencesValid(first);
            AssertAllReferencesValid(second);
            Assert.Multiple(() =>
            {
                Assert.That(step, Is.EqualTo(10).And.GreaterThan(0));
                Assert.That(ReplayFile.Read(firstEnv.ToReplay()).Commands, Has.Count.EqualTo(10));
                Assert.That(first.Terminated, Is.False);
                Assert.That(first.Truncated, Is.True);
                Assert.That(second.Terminated, Is.False);
                Assert.That(second.Truncated, Is.True);
                Assert.That(firstEnv.ToReplay(), Is.EqualTo(secondEnv.ToReplay()));
            });
        }

        [Test]
        public void ProfiledReset_SupportsEveryDeclaredProfileForEitherReferenceSeat()
        {
            TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig();
            foreach (TacticalV2StartProfile profile in config.Match.StartProfiles)
            foreach (PlayerId referenceSeat in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);
                TacticalV3View view = env.Reset(
                    6_000_005, null, null, profile.Id, referenceSeat, learnerSeat: PlayerId.Player0);
                PlayerId opponent = Other(referenceSeat);

                Assert.Multiple(() =>
                {
                    Assert.That(env.State.Player(referenceSeat).UnitsOnBoard,
                        Has.Count.EqualTo(profile.LearnerUnitCount), profile.Id);
                    Assert.That(env.State.Player(opponent).UnitsOnBoard,
                        Has.Count.EqualTo(profile.OpponentUnitCount), profile.Id);
                    Assert.That(view.StartProfileId, Is.EqualTo(profile.Id));
                    Assert.That(view.ReferenceSeat, Is.EqualTo(referenceSeat));
                });
            }
        }

        [Test]
        public void BothNullControllers_LeaveReciprocalSeatsExternallyControllable()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View first = env.Reset(31, null, null);

            TacticalV3View second = EndTurn(env, first);
            TacticalV3View third = EndTurn(env, second);

            Assert.That(new[] { first.Seat, second.Seat, third.Seat }, Is.EqualTo(new[]
            {
                PlayerId.Player0,
                PlayerId.Player1,
                PlayerId.Player0,
            }));
        }

        [Test]
        public void Step_AppliesExactlyTheSelectedCandidateAsTheOnlyExternalCommand()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, null, null);
            TacticalV3Candidate selected = view.Decision.Candidates.First(candidate =>
                candidate.Kind == TacticalV3CandidateKind.Move);
            GameState before = env.State;
            Unit expectedActor = before.Player(view.Seat).UnitsOnBoard
                .Where(unit => unit.IsAlive)
                .OrderBy(unit => unit.Id)
                .ElementAt(selected.Actor!.Value.Row);
            TacticalV3CellToken expectedCell =
                view.Decision.Observation.Cells[selected.Cell!.Value.Row];
            Command expected = new MoveUnit(view.Seat, expectedActor.Id,
                new HexCoord(expectedCell.Q, expectedCell.R));

            TacticalV3View after = env.Step(view.Decision.DecisionId, selected.CandidateId);
            ReplayData replay = ReplayFile.Read(env.ToReplay());

            Assert.Multiple(() =>
            {
                Assert.That(replay.Commands, Has.Count.EqualTo(1));
                Assert.That(replay.Commands[0], Is.EqualTo(expected));
                Assert.That(after.Decision.DecisionId, Is.EqualTo(1));
            });
        }

        [Test]
        public void DeployCandidate_CanReinforceBeyondTacticalV2RegistryCapacity()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            config.Match.Game = NonDefaultGame(config.Match.Game, deployCostMultiplier: 0.0);
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);
            TacticalV3View view = env.Reset(2, null, null);
            TacticalV3Candidate deploy = view.Decision.Candidates.First(candidate =>
                candidate.Kind == TacticalV3CandidateKind.Deploy);

            Assert.That(env.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(3));

            view = env.Step(view.Decision.DecisionId, deploy.CandidateId);
            ReplayData data = ReplayFile.Read(env.ToReplay());
            Replay replay = new Replay(data.Start, data.Commands);

            Assert.Multiple(() =>
            {
                Assert.That(env.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(4),
                    "raw tactical-v3 legality must not allocate the tactical-v2 three-slot registry");
                Assert.That(StateSignature(replay.Final), Is.EqualTo(StateSignature(env.State)));
                Assert.That(data.Commands.Count, Is.EqualTo(view.Decision.DecisionId));
            });
        }

        [Test]
        public void ScriptedRandomAndGreedyCommandsShareTheReplayTransitionPath()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            config.Match.MaxSteps = 12;
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);

            TacticalV3View view = env.Reset(47, new RandomAgent(9), new GreedyAgent(10));
            string firstWrite = env.ToReplay();
            string secondWrite = env.ToReplay();
            ReplayData data = ReplayFile.Read(firstWrite);
            Replay replay = new Replay(data.Start, data.Commands);

            Assert.Multiple(() =>
            {
                Assert.That(view.Terminated || view.Truncated, Is.True);
                Assert.That(data.Commands, Is.Not.Empty);
                Assert.That(data.Commands.Select(command => command.Issuer).Distinct(),
                    Is.EquivalentTo(new[] { PlayerId.Player0, PlayerId.Player1 }));
                Assert.That(env.InternalFallbackCount, Is.Zero);
                Assert.That(StateSignature(replay.Final), Is.EqualTo(StateSignature(env.State)));
                Assert.That(secondWrite, Is.EqualTo(firstWrite));
            });
        }

        [Test]
        public void ExternalTerminalCommand_ReturnsFreshNonActionableTerminalFrame()
        {
            TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig("conversion-3v1-near");
            config.Match.MaxSteps = 2000;
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);
            TacticalV3View view = env.Reset(6_000_005, null, new ApproachAgent(),
                "conversion-3v1-near", PlayerId.Player0);

            for (int step = 0; step < config.Match.MaxSteps && !view.Terminated && !view.Truncated; step++)
            {
                TacticalV3Candidate progress = ProgressTowardOpponent(view);
                view = env.Step(view.Decision.DecisionId, progress.CandidateId);
            }

            Assert.That(view.Terminated, Is.True, "the external attack policy must reach annihilation");
            AssertTerminalFrameMatchesState(view, env.State, ReplayFile.Read(env.ToReplay()).Commands.Count);
        }

        [Test]
        public void AllInternalReset_ReturnsFreshNonActionableTerminalFrame()
        {
            TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig("conversion-1v1-near");
            config.Match.MaxSteps = 2000;
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);

            TacticalV3View view = env.Reset(6_000_005, new AggressiveAgent(), new AggressiveAgent(),
                "conversion-1v1-near", PlayerId.Player0);

            Assert.That(view.Terminated, Is.True, "internal combat agents must reach annihilation");
            Assert.That(env.InternalFallbackCount, Is.Zero);
            AssertTerminalFrameMatchesState(view, env.State, ReplayFile.Read(env.ToReplay()).Commands.Count);
        }

        [Test]
        public void StateBeforeReset_Throws()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();

            Assert.Throws<InvalidOperationException>(() => { _ = env.State; });
        }

        [Test]
        public void StateSnapshotMutation_CannotChangeEpisodeReplayOrResolverFreshness()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, null, null);
            string before = StateSignature(env.State);
            string replayBefore = env.ToReplay();
            TacticalV3Candidate selected = view.Decision.Candidates.First(candidate =>
                candidate.Kind == TacticalV3CandidateKind.Move);

            GameState exposed = env.State;
            IList<PlayerState> players = (IList<PlayerState>)exposed.Players;
            players[0] = new PlayerState(PlayerId.Player0, 999);
            ((HashSet<HexCoord>)exposed.Board.DeploymentZone(PlayerId.Player0)).Clear();

            Assert.That(StateSignature(env.State), Is.EqualTo(before));
            Assert.That(env.ToReplay(), Is.EqualTo(replayBefore));

            TacticalV3View after = env.Step(view.Decision.DecisionId, selected.CandidateId);
            ReplayData replay = ReplayFile.Read(env.ToReplay());
            Assert.Multiple(() =>
            {
                Assert.That(after.Decision.DecisionId, Is.EqualTo(1));
                Assert.That(replay.Commands, Has.Count.EqualTo(1));
                Assert.That(StateSignature(replay.Start), Is.EqualTo(
                    StateSignature(ReplayFile.Read(replayBefore).Start)));
            });
        }

        [Test]
        public void CallerOwnedTerrainDictionary_CannotMutateEnvironmentAfterConstruction()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            config.Match.Game = NonDefaultGame(config.Match.Game, deployCostMultiplier: 1.0);
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in Enum.GetValues(typeof(TerrainType)))
                terrain.Add(terrainType, config.Match.Game.Terrain(terrainType));
            int expectedMoveCost = terrain[TerrainType.Plains].MoveCost;
            config.Match.Game = GameWithTerrain(config.Match.Game, terrain);
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);

            terrain[TerrainType.Plains] = new TerrainDef(99, 8, 7, true);
            env.Reset(31, null, null);

            Assert.Multiple(() =>
            {
                Assert.That(env.State.Config.Terrain(TerrainType.Plains).MoveCost,
                    Is.EqualTo(expectedMoveCost));
                Assert.That(ReplayFile.Read(env.ToReplay()).Start.Config
                    .Terrain(TerrainType.Plains).MoveCost, Is.EqualTo(expectedMoveCost));
            });
        }

        [Test]
        public void MutatingScriptedAgent_CannotAlterAuthoritativeStateOutsideTransition()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, new MutatingAgent(), null);

            ReplayData replay = ReplayFile.Read(env.ToReplay());
            Assert.Multiple(() =>
            {
                Assert.That(view.Terminated, Is.False);
                Assert.That(view.Seat, Is.EqualTo(PlayerId.Player1));
                Assert.That(env.State.Player(PlayerId.Player0).Points, Is.EqualTo(0));
                Assert.That(env.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(3));
                Assert.That(env.State.Board.DeploymentZone(PlayerId.Player0), Is.Not.Empty);
                Assert.That(replay.Commands, Is.EqualTo(new Command[]
                {
                    new EndTurn(PlayerId.Player0),
                }));
                Assert.That(env.InternalFallbackCount, Is.Zero);
            });
        }

        [Test]
        public void MixedExternalAndScriptedCommands_TruncateAtExactGlobalMaxSteps()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            config.Match.MaxSteps = 5;
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);
            TacticalV3View view = env.Reset(31, null, new EndTurnAgent());

            while (!view.Terminated && !view.Truncated)
                view = EndTurn(env, view);

            Assert.Multiple(() =>
            {
                Assert.That(view.Terminated, Is.False);
                Assert.That(view.Truncated, Is.True);
                Assert.That(view.Decision.DecisionId, Is.EqualTo(config.Match.MaxSteps));
                Assert.That(ReplayFile.Read(env.ToReplay()).Commands,
                    Has.Count.EqualTo(config.Match.MaxSteps));
                Assert.That(env.InternalFallbackCount, Is.Zero);
            });
        }

        private static void AssertTerminalFrameMatchesState(
            TacticalV3View view, GameState state, int expectedCommandCount)
        {
            IEnumerable<string> ExpectedRows(PlayerId owner, TacticalV3RelativeOwner relative) =>
                state.Player(owner).UnitsOnBoard
                    .Where(unit => unit.IsAlive)
                    .OrderBy(unit => unit.Id)
                    .Select(unit => relative + "|" + unit.CurrentHp + "|" + unit.Stats.Health + "|" +
                        unit.Elevation + "|" + unit.Stats.PointCost);
            string[] expectedUnits = ExpectedRows(view.Seat, TacticalV3RelativeOwner.Self)
                .Concat(ExpectedRows(Other(view.Seat), TacticalV3RelativeOwner.Opponent))
                .ToArray();
            string[] actualUnits = view.Decision.Observation.Units.Select(unit =>
                unit.Owner + "|" + unit.CurrentHp + "|" + unit.MaxHp + "|" +
                unit.Elevation + "|" + unit.PointCost).ToArray();
            TacticalV3RuleToken round = view.Decision.Observation.Rules.Single(rule =>
                rule.Kind == TacticalV3RuleKind.Round);

            Assert.Multiple(() =>
            {
                Assert.That(view.Decision.DecisionId, Is.EqualTo(expectedCommandCount));
                Assert.That(view.Seat, Is.EqualTo(state.ActivePlayer));
                Assert.That(view.Decision.Seat, Is.EqualTo(state.ActivePlayer));
                Assert.That(view.Decision.Candidates, Is.Empty);
                Assert.That(actualUnits, Is.EqualTo(expectedUnits));
                Assert.That(round.IntValue, Is.EqualTo(state.Round));
                Assert.That(view.Reward.Finalized, Is.True);
            });
        }

        private static void AssertDeterministicView(TacticalV3View expected, TacticalV3View actual) =>
            Assert.That(JsonSerializer.Serialize(actual),
                Is.EqualTo(JsonSerializer.Serialize(expected)));

        private static IReadOnlyDictionary<TacticalV3TableKind, int> RowCounts(TacticalV3View view) =>
            new Dictionary<TacticalV3TableKind, int>
            {
                [TacticalV3TableKind.Cells] = view.Decision.Observation.Cells.Count,
                [TacticalV3TableKind.Units] = view.Decision.Observation.Units.Count,
                [TacticalV3TableKind.Templates] = view.Decision.Observation.Templates.Count,
                [TacticalV3TableKind.CapabilityDefinitions] =
                    view.Decision.Observation.CapabilityDefinitions.Count,
                [TacticalV3TableKind.CapabilityAllocations] =
                    view.Decision.Observation.CapabilityAllocations.Count,
                [TacticalV3TableKind.Rules] = view.Decision.Observation.Rules.Count,
                [TacticalV3TableKind.MemoryRecords] = view.Decision.Observation.Memory.Count,
                [TacticalV3TableKind.Relations] = view.Decision.Observation.Relations.Count,
                [TacticalV3TableKind.Candidates] = view.Decision.Candidates.Count,
            };

        private static void AssertAllReferencesValid(TacticalV3View view)
        {
            IReadOnlyDictionary<TacticalV3TableKind, int> counts = RowCounts(view);
            TacticalV3Observation observation = view.Decision.Observation;
            foreach (TacticalV3UnitToken unit in observation.Units)
                AssertReferenceValid(unit.Cell, TacticalV3TableKind.Cells, counts);
            foreach (TacticalV3CapabilityAllocationToken allocation in observation.CapabilityAllocations)
            {
                AssertReferenceValid(allocation.Owner,
                    new[] { TacticalV3TableKind.Units, TacticalV3TableKind.Templates }, counts);
                AssertReferenceValid(
                    allocation.Definition, TacticalV3TableKind.CapabilityDefinitions, counts);
            }
            foreach (TacticalV3MemoryToken memory in observation.Memory)
                AssertReferenceValid(memory.Cell, TacticalV3TableKind.Cells, counts);
            foreach (TacticalV3RelationToken relation in observation.Relations)
                AssertRelationReferencesValid(relation, counts);
            foreach (TacticalV3Candidate candidate in view.Decision.Candidates)
                AssertCandidateReferencesValid(candidate, counts);
        }

        private static void AssertRelationReferencesValid(
            TacticalV3RelationToken relation,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            switch (relation.Kind)
            {
                case TacticalV3RelationKind.Neighbor:
                    AssertReferenceValid(relation.Source, TacticalV3TableKind.Cells, counts);
                    AssertReferenceValid(relation.Target, TacticalV3TableKind.Cells, counts);
                    return;
                case TacticalV3RelationKind.Occupies:
                    AssertReferenceValid(relation.Source, TacticalV3TableKind.Units, counts);
                    AssertReferenceValid(relation.Target, TacticalV3TableKind.Cells, counts);
                    return;
                case TacticalV3RelationKind.HasCapability:
                    AssertReferenceValid(relation.Source,
                        new[] { TacticalV3TableKind.Units, TacticalV3TableKind.Templates }, counts);
                    AssertReferenceValid(
                        relation.Target, TacticalV3TableKind.CapabilityDefinitions, counts);
                    return;
                default:
                    throw new AssertionException("unexpected tactical-v3 relation kind " + relation.Kind);
            }
        }

        private static void AssertReferenceValid(
            TacticalV3TokenRef reference,
            TacticalV3TableKind expectedTable,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            Assert.That(reference.Table, Is.EqualTo(expectedTable));
            AssertReferenceValid(reference, counts);
        }

        private static void AssertReferenceValid(
            TacticalV3TokenRef? reference,
            TacticalV3TableKind expectedTable,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            if (reference.HasValue)
                AssertReferenceValid(reference.Value, expectedTable, counts);
        }

        private static void AssertReferenceValid(
            TacticalV3TokenRef reference,
            IReadOnlyCollection<TacticalV3TableKind> expectedTables,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            Assert.That(expectedTables, Does.Contain(reference.Table));
            AssertReferenceValid(reference, counts);
        }

        private static void AssertReferenceValid(
            TacticalV3TokenRef? reference,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            if (reference.HasValue) AssertReferenceValid(reference.Value, counts);
        }

        private static void AssertReferenceValid(
            TacticalV3TokenRef reference,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            Assert.That(counts.ContainsKey(reference.Table), Is.True);
            Assert.That(reference.Row, Is.GreaterThanOrEqualTo(0));
            Assert.That(reference.Row, Is.LessThan(counts[reference.Table]));
        }

        private static void AssertCandidateReferencesValid(
            TacticalV3Candidate candidate,
            IReadOnlyDictionary<TacticalV3TableKind, int> counts)
        {
            AssertReferenceValid(candidate.Actor, TacticalV3TableKind.Units, counts);
            AssertReferenceValid(candidate.Target, TacticalV3TableKind.Units, counts);
            AssertReferenceValid(candidate.Template, TacticalV3TableKind.Templates, counts);
            AssertReferenceValid(candidate.Cell, TacticalV3TableKind.Cells, counts);
            AssertReferenceValid(
                candidate.Projection.SourceCell, TacticalV3TableKind.Cells, counts);
            AssertReferenceValid(
                candidate.Projection.DestinationCell, TacticalV3TableKind.Cells, counts);
            AssertReferenceValid(
                candidate.Projection.Template, TacticalV3TableKind.Templates, counts);
            AssertReferenceValid(
                candidate.Projection.Target, TacticalV3TableKind.Units, counts);
        }

        private static void AssertCommandsEqual(Command expected, Command actual)
        {
            Assert.That(actual.GetType(), Is.SameAs(expected.GetType()));
            Assert.That(actual.Issuer, Is.EqualTo(expected.Issuer));
            if (expected is AttackUnit expectedAttack)
            {
                var actualAttack = (AttackUnit)actual;
                Assert.That(actualAttack.AttackerId, Is.EqualTo(expectedAttack.AttackerId));
                Assert.That(actualAttack.TargetId, Is.EqualTo(expectedAttack.TargetId));
                return;
            }
            if (expected is MoveUnit expectedMove)
            {
                var actualMove = (MoveUnit)actual;
                Assert.That(actualMove.UnitId, Is.EqualTo(expectedMove.UnitId));
                Assert.That(actualMove.Dest, Is.EqualTo(expectedMove.Dest));
                return;
            }
            if (expected is DeployUnit expectedDeploy)
            {
                var actualDeploy = (DeployUnit)actual;
                Assert.That(actualDeploy.TemplateIndex, Is.EqualTo(expectedDeploy.TemplateIndex));
                Assert.That(actualDeploy.Cell, Is.EqualTo(expectedDeploy.Cell));
                return;
            }
            Assert.That(expected, Is.TypeOf<EndTurn>());
        }

        [Test]
        public void InvalidCandidate_DoesNotBecomeEndTurnOrMutateState()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, null, new RandomAgent(9));
            GameState before = env.State;
            string replayBefore = env.ToReplay();

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                env.Step(view.Decision.DecisionId, view.Decision.Candidates.Count));

            Assert.Multiple(() =>
            {
                Assert.That(StateSignature(env.State), Is.EqualTo(StateSignature(before)));
                Assert.That(env.ToReplay(), Is.EqualTo(replayBefore));
            });
        }

        [Test]
        public void StaleDecisionId_DoesNotMutateStateReplayOrCommandCount()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, null, new RandomAgent(9));
            GameState before = env.State;
            string replayBefore = env.ToReplay();

            Assert.Throws<InvalidOperationException>(() =>
                env.Step(view.Decision.DecisionId + 1, 0));

            Assert.Multiple(() =>
            {
                Assert.That(StateSignature(env.State), Is.EqualTo(StateSignature(before)));
                Assert.That(env.ToReplay(), Is.EqualTo(replayBefore));
                Assert.That(ReplayFile.Read(env.ToReplay()).Commands,
                    Has.Count.EqualTo(view.Decision.DecisionId));
            });
        }

        [Test]
        public void AcceptedCommandsIncrementGlobalCountOnceAndTruncateAtMatchMaxSteps()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            config.Match.MaxSteps = 3;
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env(config);
            TacticalV3View view = env.Reset(31, null, null);

            for (int expectedCount = 1; expectedCount <= config.Match.MaxSteps; expectedCount++)
            {
                view = EndTurn(env, view);
                Assert.That(ReplayFile.Read(env.ToReplay()).Commands, Has.Count.EqualTo(expectedCount));
                Assert.That(view.Decision.DecisionId, Is.EqualTo(expectedCount));
            }

            Assert.Multiple(() =>
            {
                Assert.That(view.Terminated, Is.False);
                Assert.That(view.Truncated, Is.True);
                Assert.That(view.Reward.Finalized, Is.True);
            });

            TacticalV3RewardBreakdown finalized = view.Reward;
            TacticalV3View reset = env.Reset(31, null, null);
            Assert.Multiple(() =>
            {
                Assert.That(reset.Reward.Finalized, Is.False);
                Assert.That(reset.Reward, Is.Not.SameAs(finalized));
            });
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void SingleLearnerEnv_SelectsProfileRelativeToLearnerAndReturnsAfterOpponentTurn(
            PlayerId learnerSeat)
        {
            TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig("conversion-2v1-far");
            var env = new TacticalV3Env(_ => new EndTurnAgent(), learnerSeat, config);

            TacticalV3View first = env.Reset(6_000_005);
            TacticalV3View second = env.Step(first.Decision.DecisionId,
                first.Decision.Candidates.Single(candidate =>
                    candidate.Kind == TacticalV3CandidateKind.EndTurn).CandidateId);

            Assert.Multiple(() =>
            {
                Assert.That(first.Seat, Is.EqualTo(learnerSeat));
                Assert.That(first.StartProfileId, Is.EqualTo("conversion-2v1-far"));
                Assert.That(first.ReferenceSeat, Is.EqualTo(learnerSeat));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Self), Is.EqualTo(2));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Opponent), Is.EqualTo(1));
                Assert.That(second.Seat, Is.EqualTo(learnerSeat));
                Assert.That(second.Decision.DecisionId,
                    Is.EqualTo(learnerSeat == PlayerId.Player0 ? 2 : 3));
            });
        }

        [Test]
        public void SingleLearnerEnv_ResetUsesConstructionTimeProfileSnapshotAfterCallerMutation()
        {
            const int seed = 6_000_005;
            TacticalV3Config callerConfig =
                TacticalV3Fixtures.ProfiledConfig("conversion-2v1-far");
            var env = new TacticalV3Env(
                _ => new EndTurnAgent(), PlayerId.Player1, callerConfig);
            var control = new TacticalV3Env(
                _ => new EndTurnAgent(), PlayerId.Player1,
                TacticalV3Fixtures.ProfiledConfig("conversion-2v1-far"));

            callerConfig.Match.StartProfiles = Array.AsReadOnly(new[]
            {
                TacticalV2StartCatalog.ProfiledSeededV1().Single(profile =>
                    profile.Id == "conversion-1v1-near"),
            });
            callerConfig.Match.StartDistribution = new TacticalV2StartDistribution(new[]
            {
                new TacticalV2StartWeight("conversion-1v1-near", 10000),
            });

            TacticalV3View expected = control.Reset(seed);
            TacticalV3View first = env.Reset(seed);
            TacticalV3View second = env.Reset(seed);

            Assert.Multiple(() =>
            {
                Assert.That(first.StartProfileId, Is.EqualTo("conversion-2v1-far"));
                Assert.That(first.StartProfileId, Is.EqualTo(expected.StartProfileId));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Self), Is.EqualTo(2));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Opponent), Is.EqualTo(1));
                Assert.That(first.Decision.Candidates.Select(CandidateKey),
                    Is.EqualTo(expected.Decision.Candidates.Select(CandidateKey)));
                Assert.That(second.StartProfileId, Is.EqualTo(first.StartProfileId));
                Assert.That(second.Decision.Candidates.Select(CandidateKey),
                    Is.EqualTo(first.Decision.Candidates.Select(CandidateKey)));
            });
        }

        [Test]
        public void InvalidScriptedCommand_LogsOneEndTurnRecoveryAndFallbackCount()
        {
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(31, new InvalidAgent(), null);
            ReplayData data = ReplayFile.Read(env.ToReplay());

            Assert.Multiple(() =>
            {
                Assert.That(view.Seat, Is.EqualTo(PlayerId.Player1));
                Assert.That(env.InternalFallbackCount, Is.EqualTo(1));
                Assert.That(data.Commands, Has.Count.EqualTo(1));
                Assert.That(data.Commands[0], Is.EqualTo(new EndTurn(PlayerId.Player0)));
            });
        }

        private static TacticalV3View EndTurn(TacticalV3DuelEnv env, TacticalV3View view)
        {
            TacticalV3Candidate endTurn = view.Decision.Candidates.Single(candidate =>
                candidate.Kind == TacticalV3CandidateKind.EndTurn);
            return env.Step(view.Decision.DecisionId, endTurn.CandidateId);
        }

        private static TacticalV3Candidate ProgressTowardOpponent(TacticalV3View view)
        {
            TacticalV3Candidate? attack = view.Decision.Candidates
                .Where(candidate => candidate.Kind == TacticalV3CandidateKind.Attack &&
                    candidate.Projection.Damage > 0)
                .OrderByDescending(candidate => candidate.Projection.IsLethal)
                .ThenByDescending(candidate => candidate.Projection.Damage)
                .ThenBy(candidate => candidate.CandidateId)
                .FirstOrDefault();
            if (attack != null) return attack;

            TacticalV3Candidate? move = view.Decision.Candidates
                .Where(candidate => candidate.Kind == TacticalV3CandidateKind.Move)
                .OrderByDescending(candidate => ActorDamage(view.Decision.Observation, candidate))
                .ThenBy(candidate => OpponentDistance(view.Decision.Observation, candidate))
                .ThenBy(candidate => candidate.CandidateId)
                .FirstOrDefault();
            return move ?? view.Decision.Candidates.Single(candidate =>
                candidate.Kind == TacticalV3CandidateKind.EndTurn);
        }

        private static int ActorDamage(
            TacticalV3Observation observation,
            TacticalV3Candidate candidate) =>
            observation.CapabilityAllocations.Single(allocation =>
                allocation.Owner.Equals(candidate.Actor!.Value) &&
                allocation.Capability == TacticalV3CapabilityKind.Damage).EffectiveValue;

        private static int OpponentDistance(
            TacticalV3Observation observation,
            TacticalV3Candidate candidate)
        {
            TacticalV3CellToken destination =
                observation.Cells[candidate.Cell!.Value.Row];
            var destinationCell = new HexCoord(destination.Q, destination.R);
            return observation.Units
                .Where(unit => unit.Owner == TacticalV3RelativeOwner.Opponent)
                .Select(unit =>
                {
                    TacticalV3CellToken cell = observation.Cells[unit.Cell.Row];
                    return HexCoord.Distance(destinationCell, new HexCoord(cell.Q, cell.R));
                })
                .Min();
        }

        private static GameConfig NonDefaultGame(
            GameConfig source, double deployCostMultiplier)
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>();
            foreach (TerrainType terrainType in Enum.GetValues(typeof(TerrainType)))
                terrain.Add(terrainType, source.Terrain(terrainType));
            terrain[TerrainType.Forest] = new TerrainDef(
                moveCost: 3, concealment: 4, defense: 2, passable: true);

            return new GameConfig(
                terrain,
                startingPoints: 19,
                bountyRate: 0.75,
                generatorCost: 7,
                generatorOutput: 3,
                generatorHealth: 11,
                damageFloor: 2,
                dmgHighGroundBonus: 4,
                rangeHighGroundBonus: 3,
                roundCap: 17,
                designFee: source.DesignFee,
                deployCostMultiplier: deployCostMultiplier,
                turnPolicy: source.TurnPolicy,
                biomesEnabled: true,
                winConditions: source.WinConditions,
                captureCost: source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);
        }

        private static GameConfig GameWithTerrain(
            GameConfig source, IReadOnlyDictionary<TerrainType, TerrainDef> terrain) =>
            new GameConfig(
                terrain,
                startingPoints: source.StartingPoints,
                bountyRate: source.BountyRate,
                generatorCost: source.GeneratorCost,
                generatorOutput: source.GeneratorOutput,
                generatorHealth: source.GeneratorHealth,
                damageFloor: source.DamageFloor,
                dmgHighGroundBonus: source.DmgHighGroundBonus,
                rangeHighGroundBonus: source.RangeHighGroundBonus,
                roundCap: source.RoundCap,
                designFee: source.DesignFee,
                deployCostMultiplier: source.DeployCostMultiplier,
                turnPolicy: source.TurnPolicy,
                biomesEnabled: source.BiomesEnabled,
                winConditions: source.WinConditions,
                captureCost: source.CaptureCost,
                economyWinThreshold: source.EconomyWinThreshold,
                scoreKills: source.ScoreKills,
                scorePoints: source.ScorePoints,
                scoreArmy: source.ScoreArmy,
                scoreTerritory: source.ScoreTerritory,
                upkeepFactor: source.UpkeepFactor,
                captureFactor: source.CaptureFactor,
                buildFactor: source.BuildFactor,
                territoryMode: source.TerritoryMode,
                claimEndsTurn: source.ClaimEndsTurn,
                buildAnywhere: source.BuildAnywhere,
                territoryIncome: source.TerritoryIncome,
                generatorsEnabled: source.GeneratorsEnabled,
                pointDecay: source.PointDecay,
                fogOfWar: source.FogOfWar,
                maxDesignPointCost: source.MaxDesignPointCost,
                fixedTemplateCount: source.FixedTemplateCount,
                templateSlotCount: source.TemplateSlotCount);

        private static string StateSignature(GameState state)
        {
            var rows = new List<string>
            {
                state.ActivePlayer + "|" + state.Round + "|" + state.NextEntityId + "|" +
                state.IsGameOver + "|" + (state.Winner.HasValue ? state.Winner.Value.ToString() : "-"),
                "rules|" + state.Config.TurnPolicy.GetType().FullName + "|" +
                state.Config.StartingPoints + "|" + state.Config.BountyRate + "|" +
                state.Config.GeneratorCost + "|" + state.Config.GeneratorOutput + "|" +
                state.Config.GeneratorHealth + "|" + state.Config.DamageFloor + "|" +
                state.Config.DmgHighGroundBonus + "|" + state.Config.RangeHighGroundBonus + "|" +
                state.Config.RoundCap + "|" + state.Config.DesignFee + "|" +
                state.Config.MaxDesignPointCost + "|" + state.Config.FixedTemplateCount + "|" +
                state.Config.TemplateSlotCount + "|" + state.Config.DeployCostMultiplier + "|" +
                (state.Config.TurnPolicy.ActionsPerTurn ?? -1) + "|" + state.Config.BiomesEnabled + "|" +
                state.Config.WinConditions + "|" + state.Config.CaptureCost + "|" +
                state.Config.EconomyWinThreshold + "|" + state.Config.ScoreKills + "|" +
                state.Config.ScorePoints + "|" + state.Config.ScoreArmy + "|" +
                state.Config.ScoreTerritory + "|" + state.Config.UpkeepFactor + "|" +
                state.Config.CaptureFactor + "|" + state.Config.BuildFactor + "|" +
                state.Config.TerritoryMode + "|" + state.Config.ClaimEndsTurn + "|" +
                state.Config.BuildAnywhere + "|" + state.Config.TerritoryIncome + "|" +
                state.Config.GeneratorsEnabled + "|" + state.Config.PointDecay + "|" +
                state.Config.FogOfWar,
            };

            foreach (TerrainType terrainType in Enum.GetValues(typeof(TerrainType)))
            {
                TerrainDef terrain = state.Config.Terrain(terrainType);
                rows.Add("terrain|" + terrainType + "|" + terrain.MoveCost + "|" +
                    terrain.Concealment + "|" + terrain.Defense + "|" + terrain.Passable);
            }
            foreach (PlayerId seat in new[] { PlayerId.Player0, PlayerId.Player1 })
                rows.Add("zone|" + seat + "|" + string.Join(",", state.Board.DeploymentZone(seat)
                    .OrderBy(cell => cell.Q).ThenBy(cell => cell.R)
                    .Select(cell => cell.Q + ":" + cell.R)));

            foreach (Tile tile in state.Board.Tiles.OrderBy(item => item.Coord.Q).ThenBy(item => item.Coord.R))
                rows.Add("tile|" + tile.Coord.Q + "|" + tile.Coord.R + "|" + tile.Elevation + "|" +
                    tile.Terrain + "|" + (state.Board.Controller(tile.Coord)?.ToString() ?? "-"));
            foreach (PlayerId seat in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                PlayerState player = state.Player(seat);
                rows.Add("player|" + seat + "|" + player.Points + "|" + player.DestroyedValue);
                foreach (Unit unit in player.UnitsOnBoard.OrderBy(item => item.Id))
                    rows.Add("unit|" + unit.Id + "|" + unit.Owner + "|" + unit.CurrentHp + "|" +
                        unit.Cell.Q + "|" + unit.Cell.R + "|" + unit.Elevation + "|" + unit.Name + "|" +
                        Stats(unit.Stats));
                foreach (Generator generator in player.Generators.OrderBy(item => item.Id))
                    rows.Add("generator|" + generator.Id + "|" + generator.Owner + "|" +
                        generator.CurrentHp + "|" + generator.Cell.Q + "|" + generator.Cell.R + "|" +
                        generator.Elevation + "|" + generator.Strength);
                foreach (UnitTemplate template in player.Barracks)
                    rows.Add("barracks|" + seat + "|" + template.Name + "|" + Stats(template.Stats));
            }

            rows.Add("moved|" + string.Join(",", state.MovedUnitIds.OrderBy(id => id)));
            rows.Add("attacked|" + string.Join(",", state.AttackedUnitIds.OrderBy(id => id)));
            foreach (var movement in state.MovementSpent.OrderBy(item => item.Key))
                rows.Add("movement|" + movement.Key + "|" + movement.Value.H + "|" + movement.Value.V);
            return string.Join("\n", rows);
        }

        private static string Stats(UnitStats stats) => string.Join(",", new[]
        {
            stats.Health,
            stats.Damage,
            stats.Defense,
            stats.Movement,
            stats.VerticalMovement,
            stats.Range,
            stats.RangeArc,
            stats.Vision,
            stats.VisionArc,
        });

        private static string CandidateKey(TacticalV3Candidate candidate) => string.Join("|", new[]
        {
            candidate.CandidateId.ToString(),
            candidate.DecisionId.ToString(),
            candidate.Kind.ToString(),
            Token(candidate.Actor),
            Token(candidate.Target),
            Token(candidate.Template),
            Token(candidate.Cell),
        });

        private static string Token(TacticalV3TokenRef? token) => token.HasValue
            ? token.Value.Table + ":" + token.Value.Row
            : "-";


        private static PlayerId Other(PlayerId seat) =>
            seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        private sealed class ApproachAgent : IAgent
        {
            public Command Decide(GameState state)
            {
                PlayerState opponent = state.Opponent(state.ActivePlayer);
                MoveUnit? move = LegalMoves.For(state)
                    .OfType<MoveUnit>()
                    .OrderBy(command => opponent.UnitsOnBoard
                        .Where(unit => unit.IsAlive)
                        .Min(unit => HexCoord.Distance(command.Dest, unit.Cell)))
                    .ThenBy(command => command.UnitId)
                    .ThenBy(command => command.Dest.Q)
                    .ThenBy(command => command.Dest.R)
                    .FirstOrDefault();
                return move != null ? move : new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class AggressiveAgent : IAgent
        {
            public Command Decide(GameState state)
            {
                IReadOnlyList<Command> legal = LegalMoves.For(state);
                AttackUnit? attack = legal.OfType<AttackUnit>()
                    .OrderBy(command => command.TargetId)
                    .ThenBy(command => command.AttackerId)
                    .FirstOrDefault();
                if (attack != null) return attack;

                PlayerState opponent = state.Opponent(state.ActivePlayer);
                MoveUnit? move = legal.OfType<MoveUnit>()
                    .OrderBy(command => opponent.UnitsOnBoard
                        .Where(unit => unit.IsAlive)
                        .Min(unit => HexCoord.Distance(command.Dest, unit.Cell)))
                    .ThenBy(command => command.UnitId)
                    .ThenBy(command => command.Dest.Q)
                    .ThenBy(command => command.Dest.R)
                    .FirstOrDefault();
                return move != null ? move : new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class MutatingAgent : IAgent
        {
            public Command Decide(GameState state)
            {
                IList<PlayerState> players = (IList<PlayerState>)state.Players;
                players[(int)state.ActivePlayer] = new PlayerState(state.ActivePlayer, 999);
                ((HashSet<HexCoord>)state.Board.DeploymentZone(state.ActivePlayer)).Clear();
                return new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class InvalidAgent : IAgent
        {
            public Command Decide(GameState state) =>
                new MoveUnit(state.ActivePlayer, int.MaxValue, new HexCoord(0, 0));
        }

        private sealed class EndTurnAgent : IAgent
        {
            public Command Decide(GameState state) => new EndTurn(state.ActivePlayer);
        }
    }
}
