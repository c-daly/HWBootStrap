using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2DecisionObserverTests
    {
        [Test]
        public void RegistrySnapshot_PreservesEverySlotPairAfterLiveReleaseAndRegistration()
        {
            TacticalV2Start start = new TacticalV2Layout(TacticalV2Config.Default()).NewGame(17);
            TacticalV2UnitRegistry live = start.Slots0;
            TacticalV2UnitRegistry snapshot = live.Snapshot();
            int releasedId = live.UnitIdAt(0);
            var expectedSlots = Enumerable.Range(0, snapshot.Capacity)
                .Select(slot => (snapshot.UnitIdAt(slot), snapshot.TemplateIndexAt(slot)))
                .ToArray();

            PlayerState original = start.State.Player(PlayerId.Player0);
            PlayerState foe = start.State.Player(PlayerId.Player1);
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, original.Points, original.Barracks,
                    original.UnitsOnBoard.Where(unit => unit.Id != releasedId).ToArray(),
                    original.Generators, original.DestroyedValue),
                foe,
            };
            var withoutUnit = new GameState(start.State.Board, start.State.Config, players,
                start.State.ActivePlayer, start.State.Round, start.State.NextEntityId,
                start.State.IsGameOver, start.State.Winner, start.State.MovedUnitIds,
                start.State.AttackedUnitIds, start.State.MovementSpent);

            live.ReleaseDead(withoutUnit, PlayerId.Player0);

            PlayerState released = withoutUnit.Player(PlayerId.Player0);
            Unit source = released.UnitsOnBoard[0];
            var deployed = new Unit(start.State.NextEntityId, PlayerId.Player0, source.Stats,
                source.Cell, source.Elevation);
            var playersWithDeployment = new[]
            {
                new PlayerState(PlayerId.Player0, released.Points, released.Barracks,
                    released.UnitsOnBoard.Concat(new[] { deployed }).ToArray(),
                    released.Generators, released.DestroyedValue),
                foe,
            };
            var withDeployment = new GameState(start.State.Board, start.State.Config, playersWithDeployment,
                start.State.ActivePlayer, start.State.Round, start.State.NextEntityId + 1,
                start.State.IsGameOver, start.State.Winner, start.State.MovedUnitIds,
                start.State.AttackedUnitIds, start.State.MovementSpent);
            live.RegisterDeployment(withoutUnit, withDeployment, PlayerId.Player0, templateIndex: 42);

            Assert.Multiple(() =>
            {
                Assert.That(live.UnitIdAt(0), Is.EqualTo(deployed.Id));
                Assert.That(live.TemplateIndexAt(0), Is.EqualTo(42));
                for (int slot = 0; slot < snapshot.Capacity; slot++)
                {
                    Assert.That((snapshot.UnitIdAt(slot), snapshot.TemplateIndexAt(slot)),
                        Is.EqualTo(expectedSlots[slot]));
                }
            });
        }

        [Test]
        public void Reset_ProvidesEpisodeMetadataAndInitialStateBeforeFirstExternalDecision()
        {
            TacticalV2Config config = ProfiledConfig();
            var recorder = new Recorder();
            var env = new TacticalV2DuelEnv(config) { DecisionObserver = recorder };

            env.Reset(6000005, null, null, "conversion-2v1-far", PlayerId.Player1, PlayerId.Player0);

            TacticalV2EpisodeContext episode = recorder.Episodes.Single();
            Assert.Multiple(() =>
            {
                Assert.That(episode.InitialState, Is.SameAs(env.State));
                Assert.That(episode.SelectedStartProfileId, Is.EqualTo("conversion-2v1-far"));
                Assert.That(episode.ReferenceSeat, Is.EqualTo(PlayerId.Player1));
                Assert.That(episode.LearnerSeat, Is.EqualTo(PlayerId.Player0));
                Assert.That(episode.PointsWeight, Is.EqualTo(config.PointsWeight));
                Assert.That(recorder.Decisions, Is.Empty);
            });
        }

        [Test]
        public void Observe_ReceivesExactImmutablePreActionDecisionContext()
        {
            var recorder = new Recorder();
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default()) { DecisionObserver = recorder };
            TacticalV2DuelEnv.View before = env.Reset(91, null, null);
            int action = FirstLegalNonEndTurn(before.ActionMask);
            GameState stateBefore = env.State;
            TacticalV2Start tracker = env.Layout.NewGame(91);
            Command command = TacticalV2Coding.Decode(action, stateBefore, before.Seat, env.Layout, tracker.Slots0);

            env.Step(action);

            TacticalV2DecisionContext decision = recorder.Decisions.Single();
            Assert.Multiple(() =>
            {
                Assert.That(decision.State, Is.SameAs(stateBefore));
                Assert.That(decision.Seat, Is.EqualTo(PlayerId.Player0));
                Assert.That(decision.DecisionIndex, Is.EqualTo(0));
                Assert.That(decision.Observation, Is.EqualTo(before.Observation));
                Assert.That(decision.LegalMask, Is.EqualTo(before.ActionMask));
                Assert.That(decision.LearnerAction, Is.EqualTo(action));
                Assert.That(decision.LearnerCommand, Is.EqualTo(command));
                Assert.That(decision.OwnRegistry, Is.Not.SameAs(tracker.Slots0));
                Assert.That(decision.FoeRegistry, Is.Not.SameAs(tracker.Slots1));
                Assert.That(decision.OwnRegistry.UnitIdAt(0), Is.EqualTo(tracker.Slots0.UnitIdAt(0)));
                Assert.That(decision.FoeRegistry.TemplateIndexAt(0), Is.EqualTo(tracker.Slots1.TemplateIndexAt(0)));
            });

            float originalObservation = decision.Observation[0];
            bool originalMask = decision.LegalMask[action];
            float[] returnedObservation = decision.Observation;
            bool[] returnedMask = decision.LegalMask;
            returnedObservation[0] = originalObservation + 1f;
            returnedMask[action] = !originalMask;

            Assert.Multiple(() =>
            {
                Assert.That(decision.Observation[0], Is.EqualTo(originalObservation));
                Assert.That(decision.LegalMask[action], Is.EqualTo(originalMask));
            });
        }

        [Test]
        public void Observe_ExcludesInternalActionsAndExternalActionsForAnotherSeat()
        {
            var internalRecorder = new Recorder();
            var internalEnv = new TacticalV2DuelEnv(TacticalV2Config.Default())
            {
                DecisionObserver = internalRecorder,
            };
            TacticalV2DuelEnv.View afterInternal = internalEnv.Reset(
                11, new AlwaysEndTurnAgent(), null, learnerSeat: PlayerId.Player0);
            internalEnv.Step(0);

            var otherSeatRecorder = new Recorder();
            var otherSeatEnv = new TacticalV2DuelEnv(TacticalV2Config.Default())
            {
                DecisionObserver = otherSeatRecorder,
            };
            otherSeatEnv.Reset(12, null, null, learnerSeat: PlayerId.Player0);
            otherSeatEnv.Step(0);
            otherSeatEnv.Step(0);

            Assert.Multiple(() =>
            {
                Assert.That(afterInternal.Seat, Is.EqualTo(PlayerId.Player1));
                Assert.That(internalRecorder.Decisions, Is.Empty);
                Assert.That(otherSeatRecorder.Decisions, Has.Count.EqualTo(1));
                Assert.That(otherSeatRecorder.Decisions[0].Seat, Is.EqualTo(PlayerId.Player0));
            });
        }

        [Test]
        public void Observer_FailsClosedBeforeObservationForMaskedOrOutOfRangeAction()
        {
            var recorder = new Recorder();
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default()) { DecisionObserver = recorder };
            TacticalV2DuelEnv.View view = env.Reset(7, null, null);
            int masked = FirstMaskedAction(view.ActionMask);
            GameState before = env.State;

            Assert.Throws<InvalidOperationException>(() => env.Step(masked));
            Assert.Throws<InvalidOperationException>(() => env.Step(view.ActionMask.Length));
            Assert.Multiple(() =>
            {
                Assert.That(env.State, Is.SameAs(before));
                Assert.That(recorder.Decisions, Is.Empty);
            });
        }

        [Test]
        public void PassiveObserver_DoesNotChangeSuppliedActionOrEpisodeResults()
        {
            RecordedEpisode withoutObserver = PlayScriptedEpisode(observer: null);
            var malicious = new Recorder { HeldCommand = new EndTurn(PlayerId.Player0) };
            RecordedEpisode withObserver = PlayScriptedEpisode(malicious);

            Assert.Multiple(() =>
            {
                Assert.That(withObserver.Commands, Is.EqualTo(withoutObserver.Commands));
                Assert.That(withObserver.Observations, Is.EqualTo(withoutObserver.Observations));
                Assert.That(withObserver.Masks, Is.EqualTo(withoutObserver.Masks));
                Assert.That(withObserver.Rewards, Is.EqualTo(withoutObserver.Rewards));
                Assert.That(withObserver.Terminated, Is.EqualTo(withoutObserver.Terminated));
                Assert.That(withObserver.Truncated, Is.EqualTo(withoutObserver.Truncated));
                Assert.That(withObserver.Winner, Is.EqualTo(withoutObserver.Winner));
                Assert.That(withObserver.TraceProjections, Is.EqualTo(withoutObserver.TraceProjections));
                Assert.That(withObserver.Replay, Is.EqualTo(withoutObserver.Replay));
                Assert.That(malicious.Decisions.Select(item => item.LearnerCommand), Is.EqualTo(
                    withObserver.Commands.Where(item => item.Issuer == PlayerId.Player0)));
                Assert.That(malicious.Decisions.Any(item => !Equals(item.LearnerCommand, malicious.HeldCommand)), Is.True);
            });
        }

        private static RecordedEpisode PlayScriptedEpisode(ITacticalV2DecisionObserver? observer)
        {
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default())
            {
                DecisionObserver = observer,
                CaptureTransitions = true,
            };
            var random = new Random(21);
            TacticalV2DuelEnv.View view = env.Reset(21, null, new RandomAgent(22));
            var commands = new List<Command>();
            var observations = new List<float[]>();
            var masks = new List<bool[]>();
            var rewards = new List<float>();
            int guard = 0;
            var traceProjections = new List<string>();
            while (!view.Terminated && !view.Truncated)
            {
                int action = PickLegal(view.ActionMask, random);
                observations.Add(view.Observation);
                masks.Add(view.ActionMask);
                view = env.Step(action);
                rewards.Add(view.Reward);
                foreach (DuelTransition transition in env.DrainTransitions())
                {
                    commands.Add(transition.Command);
                    traceProjections.Add(JsonSerializer.Serialize(TacticalEvaluationTrace.Project(transition)));
                }
                Assert.That(++guard, Is.LessThan(10_000));
            }
            return new RecordedEpisode(
                commands, observations, masks, rewards, traceProjections, view, env.ToReplay());
        }

        private static TacticalV2Config ProfiledConfig()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.PlacementPolicy = "profiled-seeded-v1";
            config.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1();
            config.StartDistribution = new TacticalV2StartDistribution(config.StartProfiles.Select(profile =>
                new TacticalV2StartWeight(profile.Id, profile.Id == "conversion-2v1-far" ? 10000 : 0)));
            return config;
        }

        private static int FirstLegalNonEndTurn(bool[] mask)
        {
            for (int action = 1; action < mask.Length; action++) if (mask[action]) return action;
            Assert.Fail("expected legal non-EndTurn action");
            return -1;
        }

        private static int FirstMaskedAction(bool[] mask)
        {
            for (int action = 1; action < mask.Length; action++) if (!mask[action]) return action;
            Assert.Fail("expected masked action");
            return -1;
        }

        private static int PickLegal(bool[] mask, Random random)
        {
            var legal = new List<int>();
            for (int action = 0; action < mask.Length; action++) if (mask[action]) legal.Add(action);
            return legal[random.Next(legal.Count)];
        }

        private sealed class AlwaysEndTurnAgent : IAgent
        {
            public Command Decide(GameState state) => new EndTurn(state.ActivePlayer);
        }

        private sealed class Recorder : ITacticalV2DecisionObserver
        {
            public List<TacticalV2EpisodeContext> Episodes { get; } = new List<TacticalV2EpisodeContext>();
            public List<TacticalV2DecisionContext> Decisions { get; } = new List<TacticalV2DecisionContext>();
            public Command? HeldCommand { get; set; }
            public void Reset(TacticalV2EpisodeContext episode) => Episodes.Add(episode);
            public void Observe(TacticalV2DecisionContext decision) => Decisions.Add(decision);
        }

        private sealed class RecordedEpisode
        {
            public RecordedEpisode(IReadOnlyList<Command> commands, IReadOnlyList<float[]> observations,
                IReadOnlyList<bool[]> masks, IReadOnlyList<float> rewards,
                IReadOnlyList<string> traceProjections, TacticalV2DuelEnv.View view, string replay)
            {
                Commands = commands;
                Observations = observations;
                Masks = masks;
                Rewards = rewards;
                TraceProjections = traceProjections;
                Terminated = view.Terminated;
                Truncated = view.Truncated;
                Winner = view.Winner;
                Replay = replay;
            }
            public IReadOnlyList<Command> Commands { get; }
            public IReadOnlyList<float[]> Observations { get; }
            public IReadOnlyList<bool[]> Masks { get; }
            public IReadOnlyList<float> Rewards { get; }
            public IReadOnlyList<string> TraceProjections { get; }
            public bool Terminated { get; }
            public bool Truncated { get; }
            public int Winner { get; }
            public string Replay { get; }
        }
    }
}
