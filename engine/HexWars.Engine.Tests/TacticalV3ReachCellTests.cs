using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class TacticalV3ReachCellTests
    {
        [Test]
        public void Objective_ChangesOnlyMatchIdentityAndLegacyShapeStaysPinned()
        {
            TacticalV3Config legacyConfig = TacticalV3Fixtures.Config();
            var reachConfig = new TacticalV3Config(
                legacyConfig.Match,
                legacyConfig.Capacity,
                legacyConfig.Reward,
                ReachObjective());

            TacticalV3Contract legacy = TacticalV3Contract.Create(
                legacyConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract reach = TacticalV3Contract.Create(
                reachConfig, MlEnvironmentKind.Duel);
            var objective = (IReadOnlyDictionary<string, object>)reach.Match["objective"];

            Assert.Multiple(() =>
            {
                Assert.That(legacy.Match.ContainsKey("objective"), Is.False);
                Assert.That(legacy.EncodingHash, Is.EqualTo(
                    "e7a62d698a5f516c72ca3d1269ebd4b1afc61e7950c8ff0aeb2716f80e45f4b6"));
                Assert.That(reach.ContractHash, Is.Not.EqualTo(legacy.ContractHash));
                Assert.That(reach.EncodingHash, Is.EqualTo(legacy.EncodingHash));
                Assert.That(reach.CapacityHash, Is.EqualTo(legacy.CapacityHash));
                Assert.That(objective.Keys,
                    Is.EquivalentTo(new[] { "kind", "target_policy", "radius" }));
                Assert.That(objective["kind"], Is.EqualTo("reach_cell"));
                Assert.That(objective["target_policy"], Is.EqualTo(
                    "seeded_farthest_reachable_unoccupied_v1"));
                Assert.That(objective["radius"], Is.EqualTo(0));
            });
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ReachTeacher_FollowsFrozenTargetAndTerminatesBeforeOpponentActs(
            PlayerId learnerSeat)
        {
            TacticalV3Config config = LineReachConfig(lethalRangedAttack: false);
            var opponent = new CountingPassiveAgent();
            var env = new TacticalV3DuelEnv(config);
            TacticalV3View view = ResetWithPassiveOpponent(
                env, seed: 73, learnerSeat, opponent);
            TacticalV3TokenRef frozenTarget = RequireMoveTarget(view);

            var duplicate = new TacticalV3DuelEnv(config);
            TacticalV3View duplicateView = ResetWithPassiveOpponent(
                duplicate, seed: 73, learnerSeat, new CountingPassiveAgent());
            Assert.That(RequireMoveTarget(duplicateView), Is.EqualTo(frozenTarget));

            HexCoord physicalTarget = new TacticalV2Layout(config.Match).Cells[frozenTarget.Row];
            Unit initialLearner = env.State.Player(learnerSeat).UnitsOnBoard.Single();
            Assert.Multiple(() =>
            {
                Assert.That(env.State.Players.SelectMany(player => player.UnitsOnBoard)
                    .Any(unit => unit.IsAlive && unit.Cell == physicalTarget), Is.False);
                Assert.That(HexCoord.Distance(initialLearner.Cell, physicalTarget), Is.EqualTo(5));
            });

            var teacher = new BoundedSearchAgent(512, 4, useHeuristic: true);
            bool completed = false;
            for (int guard = 0; guard < 32 && !view.Terminated; guard++)
            {
                TacticalV3Candidate[] moves = view.Decision.Candidates
                    .Where(candidate => candidate.Kind == TacticalV3CandidateKind.Move)
                    .ToArray();
                foreach (TacticalV3Candidate move in moves)
                {
                    Assert.That(move.Target, Is.EqualTo(frozenTarget));
                    Assert.That(move.Target!.Value.Table, Is.EqualTo(TacticalV3TableKind.Cells));
                }

                TacticalV3TeacherSelection first = env.SelectTeacherCandidate(teacher);
                TacticalV3TeacherSelection second = env.SelectTeacherCandidate(teacher);
                Assert.Multiple(() =>
                {
                    Assert.That(second.CandidateId, Is.EqualTo(first.CandidateId));
                    Assert.That(first.SearchDepth, Is.Zero);
                    Assert.That(first.ExpansionBudget, Is.EqualTo(512));
                    Assert.That(first.ActualExpansions, Is.Zero);
                    Assert.That(first.HeuristicIdentity,
                        Is.EqualTo("reach-cell-shortest-path-v1"));
                });

                TacticalV3Candidate selected = view.Decision.Candidates[first.CandidateId];
                if (moves.Length == 0)
                {
                    Assert.That(selected.Kind, Is.EqualTo(TacticalV3CandidateKind.EndTurn));
                }
                else
                {
                    TacticalV3Candidate expected = moves
                        .OrderBy(candidate => RemainingDistance(view, candidate, frozenTarget))
                        .ThenBy(candidate => candidate.CandidateId)
                        .First();
                    Assert.That(selected.CandidateId, Is.EqualTo(expected.CandidateId));
                    Assert.That(selected.Kind, Is.EqualTo(TacticalV3CandidateKind.Move));
                }

                bool reachesTarget = selected.Kind == TacticalV3CandidateKind.Move &&
                    selected.Cell.HasValue && selected.Cell.Value.Equals(frozenTarget);
                if (selected.Kind == TacticalV3CandidateKind.Move)
                {
                    Assert.That(selected.Projection.DestinationCell, Is.EqualTo(selected.Cell));
                    Assert.That(selected.Projection.IsTerminal, Is.EqualTo(reachesTarget));
                }

                int opponentCalls = opponent.Calls;
                view = env.Step(view.Decision.DecisionId, selected.CandidateId);
                if (!reachesTarget) continue;

                completed = true;
                Assert.Multiple(() =>
                {
                    Assert.That(opponent.Calls, Is.EqualTo(opponentCalls),
                        "the opponent must not advance after the reaching move");
                    Assert.That(view.Terminated, Is.True);
                    Assert.That(view.Truncated, Is.False);
                    Assert.That(view.Winner, Is.EqualTo((int)learnerSeat));
                    Assert.That(view.Reward.Finalized, Is.True);
                    Assert.That(view.Reward.TerminalOutcome, Is.EqualTo(1f));
                    Assert.That(view.Decision.Candidates, Is.Empty);
                    Assert.That(env.State.IsGameOver, Is.False);
                    Assert.That(env.State.Winner, Is.Null);
                    Assert.That(env.State.ActivePlayer, Is.EqualTo(learnerSeat));
                });
            }

            Assert.That(completed, Is.True, "shortest-path teacher did not reach the beacon");
        }

        [Test]
        public void UnderlyingAnnihilationBeforeBeacon_IsTerminalNonWin()
        {
            TacticalV3Config config = LineReachConfig(lethalRangedAttack: true);
            var env = new TacticalV3DuelEnv(config);
            TacticalV3View view = env.Reset(
                73, controller0: null, controller1: new PassiveAgent(),
                learnerSeat: PlayerId.Player0);
            TacticalV3TokenRef target = RequireMoveTarget(view);
            TacticalV3Candidate endTurn = view.Decision.Candidates.Single(candidate =>
                candidate.Kind == TacticalV3CandidateKind.EndTurn);
            view = env.Step(view.Decision.DecisionId, endTurn.CandidateId);
            TacticalV3Candidate attack = view.Decision.Candidates.Single(candidate =>
                candidate.Kind == TacticalV3CandidateKind.Attack);

            Assert.That(attack.Projection.IsTerminal, Is.True);
            Assert.That(attack.Target, Is.Not.EqualTo(target));

            TacticalV3View after = env.Step(view.Decision.DecisionId, attack.CandidateId);

            Assert.Multiple(() =>
            {
                Assert.That(env.State.IsGameOver, Is.True);
                Assert.That(env.State.Winner, Is.EqualTo(PlayerId.Player0));
                Assert.That(after.Terminated, Is.True);
                Assert.That(after.Truncated, Is.False);
                Assert.That(after.Winner, Is.EqualTo(-1));
                Assert.That(after.Reward.TerminalOutcome, Is.EqualTo(-1f));
                Assert.That(after.Decision.Candidates, Is.Empty);
            });
        }

        private static TacticalV3ObjectiveConfig ReachObjective() =>
            new TacticalV3ObjectiveConfig(
                TacticalV3ObjectiveConfig.ReachCellKind,
                TacticalV3ObjectiveConfig.SeededFarthestReachableUnoccupiedPolicy,
                radius: 0);

        private static TacticalV3Config LineReachConfig(bool lethalRangedAttack)
        {
            var stats = new UnitStats(
                health: lethalRangedAttack ? 1 : 5,
                damage: lethalRangedAttack ? 50 : 1,
                defense: 0,
                movement: 1,
                verticalMovement: 1,
                range: lethalRangedAttack ? 10 : 0,
                rangeArc: lethalRangedAttack ? 10 : 0,
                vision: 10,
                visionArc: 10);
            var match = new TacticalV2Config
            {
                BoardGen = new BoardGenConfig(
                    width: 7, height: 1, maxElevation: 1, zoneDepth: 1,
                    flatChance: 1.0, plainsWeight: 100,
                    forestWeight: 0, roughWeight: 0, waterWeight: 0),
                Game = GameConfig.Default(
                    biomesEnabled: false,
                    winConditions: WinBy.Annihilation,
                    captureCost: int.MaxValue,
                    territoryMode: false,
                    territoryIncome: 0,
                    generatorsEnabled: false,
                    fogOfWar: false,
                    fixedTemplateCount: 1,
                    templateSlotCount: 1),
                Templates = Array.AsReadOnly(new[]
                {
                    new TacticalV2Template(
                        "reach-runner-v1", new UnitTemplate("Reach Runner", stats)),
                }),
                StartingUnitCount = 1,
                MaxControllableUnits = 1,
                MaxSteps = 404,
                PointsWeight = 0.5f,
                PlacementPolicy = "symmetric-random-v1",
                StartProfiles = Array.Empty<TacticalV2StartProfile>(),
                StartDistribution = TacticalV2StartDistribution.Empty,
            };
            return new TacticalV3Config(
                match,
                TacticalV3Fixtures.ExperimentalCapacity(),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f),
                ReachObjective());
        }

        private static TacticalV3View ResetWithPassiveOpponent(
            TacticalV3DuelEnv env,
            int seed,
            PlayerId learnerSeat,
            CountingPassiveAgent opponent) =>
            learnerSeat == PlayerId.Player0
                ? env.Reset(seed, controller0: null, controller1: opponent, learnerSeat)
                : env.Reset(seed, controller0: opponent, controller1: null, learnerSeat);

        private static TacticalV3TokenRef RequireMoveTarget(TacticalV3View view) =>
            view.Decision.Candidates
                .First(candidate => candidate.Kind == TacticalV3CandidateKind.Move)
                .Target ?? throw new AssertionException("reach move target is missing");

        private static int RemainingDistance(
            TacticalV3View view,
            TacticalV3Candidate move,
            TacticalV3TokenRef target)
        {
            TacticalV3CellToken destination =
                view.Decision.Observation.Cells[move.Cell!.Value.Row];
            TacticalV3CellToken goal = view.Decision.Observation.Cells[target.Row];
            return HexCoord.Distance(
                new HexCoord(destination.Q, destination.R),
                new HexCoord(goal.Q, goal.R));
        }

        private sealed class CountingPassiveAgent : IAgent
        {
            public int Calls { get; private set; }

            public Command Decide(GameState state)
            {
                Calls++;
                return new EndTurn(state.ActivePlayer);
            }
        }
    }
}
