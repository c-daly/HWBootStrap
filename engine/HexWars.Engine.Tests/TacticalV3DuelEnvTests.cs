using System;
using System.Collections.Generic;
using System.Linq;
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
                Assert.That(StateText(second.State), Is.EqualTo(StateText(first.State)));
                Assert.That(secondView.Decision.DecisionId, Is.EqualTo(0));
                Assert.That(secondView.Decision.DecisionId, Is.EqualTo(firstView.Decision.DecisionId));
                Assert.That(secondView.Seat, Is.EqualTo(firstView.Seat));
                Assert.That(secondView.Decision.Candidates.Select(CandidateKey),
                    Is.EqualTo(firstView.Decision.Candidates.Select(CandidateKey)));
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
            Command expected = new TacticalV3ActionResolver().Resolve(
                view.Decision, view.Decision.DecisionId, selected.CandidateId, env.State);

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
            TacticalV3DuelEnv env = TacticalV3Fixtures.Env();
            TacticalV3View view = env.Reset(2, null, new ApproachAgent());

            Assert.That(env.State.Player(PlayerId.Player0).Points, Is.EqualTo(0));
            TacticalV3Candidate? deploy = null;
            for (int step = 0; step < 600 && deploy == null; step++)
            {
                deploy = view.Decision.Candidates.FirstOrDefault(candidate =>
                    candidate.Kind == TacticalV3CandidateKind.Deploy);
                if (deploy == null)
                {
                    Assert.That(view.Terminated || view.Truncated, Is.False,
                        "the learner must earn reinforcement points before the episode ends");
                    TacticalV3Candidate progress = ProgressTowardOpponent(view);
                    view = env.Step(view.Decision.DecisionId, progress.CandidateId);
                }
            }

            deploy = deploy ?? throw new AssertionException(
                "default tactical rules did not expose an earned reinforcement within 600 decisions; " +
                "points=" + env.State.Player(PlayerId.Player0).Points +
                ", round=" + env.State.Round +
                ", ownUnits=" + env.State.Player(PlayerId.Player0).UnitsOnBoard.Count +
                ", opponentUnits=" + env.State.Player(PlayerId.Player1).UnitsOnBoard.Count +
                ", candidates=" + string.Join(",", view.Decision.Candidates.Select(item => item.Kind)));
            Assert.Multiple(() =>
            {
                Assert.That(env.State.Player(PlayerId.Player0).Points, Is.GreaterThan(0));
                Assert.That(env.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(3));
            });

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
                Assert.That(env.State, Is.SameAs(before));
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
                Assert.That(env.State, Is.SameAs(before));
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

        [Test]
        public void SingleLearnerEnv_SelectsProfileRelativeToLearnerAndReturnsAfterOpponentTurn()
        {
            TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig("conversion-2v1-far");
            var env = new TacticalV3Env(_ => new EndTurnAgent(), PlayerId.Player1, config);

            TacticalV3View first = env.Reset(6_000_005);
            TacticalV3View second = env.Step(first.Decision.DecisionId,
                first.Decision.Candidates.Single(candidate =>
                    candidate.Kind == TacticalV3CandidateKind.EndTurn).CandidateId);

            Assert.Multiple(() =>
            {
                Assert.That(first.Seat, Is.EqualTo(PlayerId.Player1));
                Assert.That(first.StartProfileId, Is.EqualTo("conversion-2v1-far"));
                Assert.That(first.ReferenceSeat, Is.EqualTo(PlayerId.Player1));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Self), Is.EqualTo(2));
                Assert.That(first.Decision.Observation.Units.Count(unit =>
                    unit.Owner == TacticalV3RelativeOwner.Opponent), Is.EqualTo(1));
                Assert.That(second.Seat, Is.EqualTo(PlayerId.Player1));
                Assert.That(second.Decision.DecisionId, Is.EqualTo(3));
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

        private static string StateSignature(GameState state)
        {
            var rows = new List<string>
            {
                state.ActivePlayer + "|" + state.Round + "|" + state.NextEntityId + "|" +
                state.IsGameOver + "|" + (state.Winner.HasValue ? state.Winner.Value.ToString() : "-"),
                "rules|" + state.Config.BountyRate + "|" + state.Config.DeployCostMultiplier + "|" +
                state.Config.DamageFloor + "|" + state.Config.RoundCap,
            };

            foreach (Tile tile in state.Board.Tiles)
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
                        generator.Elevation);
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

        private static string StateText(GameState state) =>
            ReplayFile.Write(state, Array.Empty<Command>());

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
