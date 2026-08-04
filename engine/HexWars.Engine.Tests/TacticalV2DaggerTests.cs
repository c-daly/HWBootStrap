using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class TacticalV2DaggerTests
    {
        [Test]
        public void Conversion_EmitsOnlyTheCompleteConversionReason()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            TacticalV2DaggerDecision row = ObserveOnce(fixture, fixture.FirstProductiveAction);

            Assert.Multiple(() =>
            {
                Assert.That(row.Reasons, Is.EqualTo(DaggerEligibilityReason.Conversion));
                Assert.That(row.OpponentLivingUnitCount, Is.EqualTo(1));
            });
        }

        [Test]
        public void Favorable_UsesPartialHpAndBankedPointsInTheExactFormula()
        {
            UnitStats ten = Stats(health: 4, damage: 1, movement: 1, range: 1, vision: 3);
            UnitStats six = Stats(health: 2, damage: 1, movement: 1, range: 1, vision: 1);
            Fixture initial = Fixture.Create(
                new[] { Spec(1, ten), Spec(2, six) },
                new[] { Spec(10, ten), Spec(11, six) }, 2, 2);
            // Initial material is (10 + 6 + 2*2) per seat: denominator 40.
            Fixture current = initial.WithState(
                new[] { Spec(1, ten, damageTaken: 2), Spec(2, six) },
                new[] { Spec(10, ten, damageTaken: 3), Spec(11, six) }, 5, 1);
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(new EchoOracle(), sink);
            observer.Reset(initial.Episode(pointsWeight: 2f));

            observer.Observe(current.Context(current.FirstProductiveAction));

            TacticalV2DaggerDecision row = sink.Drain().Single();
            // Current learner = 10*(2/4)+6+2*5 = 21; opponent = 10*(1/4)+6+2*1 = 10.5.
            Assert.Multiple(() =>
            {
                Assert.That(row.Reasons, Is.EqualTo(DaggerEligibilityReason.Favorable));
                Assert.That(row.NormalizedAdvantage, Is.EqualTo(0.2625d).Within(1e-12));
            });
        }

        [Test]
        public void CycleWarning_IsSecondOccurrenceNotFirst()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(2));
            var oracle = new EchoOracle();
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(oracle, sink);
            observer.Reset(fixture.Episode());
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);

            observer.Observe(context);
            Assert.That(sink.Drain(), Is.Empty);
            observer.Observe(context);

            Assert.Multiple(() =>
            {
                Assert.That(sink.Drain().Single().Reasons,
                    Is.EqualTo(DaggerEligibilityReason.CycleWarning));
                Assert.That(oracle.DecisionCount, Is.EqualTo(1));
            });
        }

        [Test]
        public void WastedEndTurn_UsesTheSharedTraceProductiveActionProjection()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(2));
            TacticalV2DaggerDecision row = ObserveOnce(fixture, learnerAction: 0);

            Assert.Multiple(() =>
            {
                Assert.That(row.Reasons, Is.EqualTo(DaggerEligibilityReason.WastedEndTurn));
                Assert.That(row.ProductiveLegalActionCount, Is.GreaterThan(0));
                Assert.That(row.ProductiveLegalActionCount,
                    Is.EqualTo(LegalMoves.For(fixture.State).Count(command => !(command is EndTurn))));
            });
        }

        [Test]
        public void Observe_RecordsAllSimultaneousReasonsBeforeEmission()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(1, weak: true));
            TacticalV2DaggerDecision row = ObserveOnce(fixture, learnerAction: 0);

            Assert.That(row.Reasons, Is.EqualTo(
                DaggerEligibilityReason.Conversion |
                DaggerEligibilityReason.Favorable |
                DaggerEligibilityReason.WastedEndTurn));
        }

        [Test]
        public void RepeatedEligibleState_QueriesAndEmitsAtMostOncePerEpisode()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            var oracle = new EchoOracle();
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(oracle, sink);
            observer.Reset(fixture.Episode());
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);

            observer.Observe(context);
            observer.Observe(context);
            observer.Observe(context);

            Assert.Multiple(() =>
            {
                Assert.That(sink.Drain(), Has.Count.EqualTo(1));
                Assert.That(oracle.DecisionCount, Is.EqualTo(1));
            });
        }

        [Test]
        public void CanonicalStateKey_MatchesPythonCycleKeyFieldOrderAndSorting()
        {
            UnitStats stats = Stats(health: 5, damage: 1, movement: 2,
                verticalMovement: 1, range: 1, vision: 1);
            var tiles = new[]
            {
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            };
            Board board = new Board(tiles)
                .WithControl(new HexCoord(2, 0), PlayerId.Player0)
                .WithControl(new HexCoord(0, 0), PlayerId.Player1);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 3, unitsOnBoard: new[]
                {
                    new Unit(30, PlayerId.Player0, stats, new HexCoord(2, 0), 0).WithDamage(5),
                    new Unit(10, PlayerId.Player0, stats, new HexCoord(1, 0), 0).WithDamage(2),
                }, destroyedValue: 7),
                new PlayerState(PlayerId.Player1, 5, unitsOnBoard: new[]
                {
                    new Unit(20, PlayerId.Player1, stats, new HexCoord(0, 0), 0).WithDamage(1),
                }, destroyedValue: 11),
            }, PlayerId.Player1, 9, 31,
                movedUnitIds: new[] { 10 }, attackedUnitIds: new[] { 20 },
                movementSpent: new Dictionary<int, (int H, int V)>
                {
                    [10] = (2, 1),
                    [20] = (0, 1),
                });

            Assert.That(SelectiveDaggerObserver.CanonicalStateKey(state), Is.EqualTo(
                "1|[(3,7),(5,11)]|[(0,0,1),(2,0,0)]|" +
                "[(0,10,1,0,3,1,0,2,1),(1,20,0,0,4,0,1,0,1)]"));
        }

        [TestCase(512)]
        [TestCase(2048)]
        public void BoundedOracle_ReturnsLegalDeterministicRoundTripWithExpansionEvidence(int budget)
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(73);
            TacticalV2DecisionContext context = Context(start, layout, learnerAction: 0);
            var oracle = new BoundedSearchActionOracle(
                config.Game, budget, 4, BoundedSearchAgent.HeuristicIdentity);

            TacticalV2OracleDecision first = oracle.Decide(context);
            TacticalV2OracleDecision second = oracle.Decide(context);
            bool[] mask = context.LegalMask;

            Assert.Multiple(() =>
            {
                Assert.That(LegalMoves.For(context.State), Does.Contain(first.Command));
                Assert.That(first.Action, Is.InRange(0, mask.Length - 1));
                Assert.That(mask[first.Action], Is.True);
                Assert.That(TacticalV2Coding.Decode(first.Action, context.State, context.Seat,
                    context.Layout, context.OwnRegistry), Is.EqualTo(first.Command));
                Assert.That(first.ActualExpansionCount,
                    Is.GreaterThan(0).And.LessThanOrEqualTo(budget));
                Assert.That(second.Command, Is.EqualTo(first.Command));
                Assert.That(second.Action, Is.EqualTo(first.Action));
                Assert.That(second.ActualExpansionCount, Is.EqualTo(first.ActualExpansionCount));
                Assert.That(second.Depth, Is.EqualTo(first.Depth));
                Assert.That(second.ExpansionBudget, Is.EqualTo(first.ExpansionBudget));
                Assert.That(second.HeuristicIdentity, Is.EqualTo(first.HeuristicIdentity));
            });
        }

        [Test]
        public void BoundedOracle_RejectsFogOfWarAndUnknownHeuristic()
        {
            Assert.Throws<ArgumentException>(() => new BoundedSearchActionOracle(
                GameConfig.Default(fogOfWar: true), 512, 4,
                BoundedSearchAgent.HeuristicIdentity));
            Assert.Throws<ArgumentException>(() => new BoundedSearchActionOracle(
                GameConfig.Default(), 512, 4, "unrecognized-v99"));
        }

        [Test]
        public void EvidenceRows_CloneArraysAndTraceCommandsAndCarryEveryDiagnostic()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            TacticalV2DaggerDecision row = ObserveOnce(fixture, fixture.FirstProductiveAction);
            float expectedObservation = row.Observation[0];
            bool expectedMask = row.LegalMask[0];
            string expectedLearnerKind = row.LearnerCommand.Kind;
            string expectedTeacherKind = row.TeacherCommand.Kind;

            float[] observation = row.Observation;
            bool[] mask = row.LegalMask;
            TacticalTraceCommand learner = row.LearnerCommand;
            TacticalTraceCommand teacher = row.TeacherCommand;
            observation[0] += 10f;
            mask[0] = !mask[0];
            learner.Kind = "mutated";
            teacher.Kind = "mutated";

            Assert.Multiple(() =>
            {
                Assert.That(row.Observation[0], Is.EqualTo(expectedObservation));
                Assert.That(row.LegalMask[0], Is.EqualTo(expectedMask));
                Assert.That(row.LearnerCommand.Kind, Is.EqualTo(expectedLearnerKind));
                Assert.That(row.TeacherCommand.Kind, Is.EqualTo(expectedTeacherKind));
                Assert.That(row.StateHash, Has.Length.EqualTo(64));
                Assert.That(row.Seat, Is.EqualTo((int)fixture.State.ActivePlayer));
                Assert.That(row.Round, Is.EqualTo(fixture.State.Round));
                Assert.That(row.DecisionIndex, Is.EqualTo(0));
                Assert.That(row.Disagreement, Is.False);
                Assert.That(row.OracleDepth, Is.EqualTo(4));
                Assert.That(row.OracleExpansionBudget, Is.EqualTo(512));
                Assert.That(row.OracleHeuristicIdentity,
                    Is.EqualTo(BoundedSearchAgent.HeuristicIdentity));
                Assert.That(row.OracleActualExpansionCount, Is.EqualTo(7));
            });
        }

        private static TacticalV2DaggerDecision ObserveOnce(Fixture fixture, int learnerAction)
        {
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(new EchoOracle(), sink);
            observer.Reset(fixture.Episode());
            observer.Observe(fixture.Context(learnerAction));
            return sink.Drain().Single();
        }

        private static UnitStats Stats(int health, int damage = 0, int movement = 1,
            int verticalMovement = 0, int range = 1, int vision = 1) =>
            new UnitStats(health, damage, 0, movement, verticalMovement, range, 0, vision, 0);

        private static UnitSpec Spec(int id, UnitStats stats, int damageTaken = 0) =>
            new UnitSpec(id, stats, damageTaken);

        private static UnitSpec[] Units(int count, bool weak = false)
        {
            UnitStats stats = weak ? Stats(2) : Stats(4, damage: 1, vision: 2);
            return Enumerable.Range(0, count).Select(index => Spec(index + 1, stats)).ToArray();
        }

        private static TacticalV2DecisionContext Context(
            TacticalV2Start start, TacticalV2Layout layout, int learnerAction)
        {
            PlayerId seat = start.State.ActivePlayer;
            TacticalV2UnitRegistry own = seat == PlayerId.Player0 ? start.Slots0 : start.Slots1;
            TacticalV2UnitRegistry foe = seat == PlayerId.Player0 ? start.Slots1 : start.Slots0;
            bool[] mask = TacticalV2Coding.Mask(start.State, seat, layout, own);
            Command command = TacticalV2Coding.Decode(learnerAction, start.State, seat, layout, own);
            return new TacticalV2DecisionContext(start.State, seat, 0,
                TacticalV2Coding.Observe(start.State, seat, layout, own, foe), mask,
                learnerAction, command, own, foe, layout);
        }

        private sealed class EchoOracle : IActionOracle
        {
            public int DecisionCount { get; private set; }

            public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
            {
                DecisionCount++;
                return new TacticalV2OracleDecision(context.LearnerAction, context.LearnerCommand,
                    depth: 4, expansionBudget: 512,
                    heuristicIdentity: BoundedSearchAgent.HeuristicIdentity,
                    actualExpansionCount: 7);
            }
        }

        private readonly struct UnitSpec
        {
            public UnitSpec(int id, UnitStats stats, int damageTaken)
            {
                Id = id;
                Stats = stats;
                DamageTaken = damageTaken;
            }

            public int Id { get; }
            public UnitStats Stats { get; }
            public int DamageTaken { get; }
        }

        private sealed class Fixture
        {
            private Fixture(TacticalV2Layout layout, GameState state,
                TacticalV2UnitRegistry slots0, TacticalV2UnitRegistry slots1)
            {
                Layout = layout;
                State = state;
                Slots0 = slots0;
                Slots1 = slots1;
            }

            public TacticalV2Layout Layout { get; }
            public GameState State { get; }
            public TacticalV2UnitRegistry Slots0 { get; }
            public TacticalV2UnitRegistry Slots1 { get; }

            public int FirstProductiveAction
            {
                get
                {
                    bool[] mask = TacticalV2Coding.Mask(State, PlayerId.Player0, Layout, Slots0);
                    for (int action = 1; action < mask.Length; action++)
                        if (mask[action]) return action;
                    Assert.Fail("fixture requires a productive tactical-v2 action");
                    return -1;
                }
            }

            public static Fixture Create(UnitSpec[] learner, UnitSpec[] opponent,
                int learnerPoints = 0, int opponentPoints = 0)
            {
                TacticalV2Config config = TacticalV2Config.Default();
                config.MaxControllableUnits = Math.Max(config.MaxControllableUnits,
                    Math.Max(learner.Length, opponent.Length));
                var layout = new TacticalV2Layout(config);
                TacticalV2Start start = layout.NewGame(91);
                Unit[] learnerUnits = MakeUnits(PlayerId.Player0, learner,
                    start.State.Player(PlayerId.Player0).UnitsOnBoard
                        .Select(unit => unit.Cell).ToArray());
                Unit[] opponentUnits = MakeUnits(PlayerId.Player1, opponent,
                    start.State.Player(PlayerId.Player1).UnitsOnBoard
                        .Select(unit => unit.Cell).ToArray());
                var state = new GameState(start.State.Board, start.State.Config, new[]
                {
                    new PlayerState(PlayerId.Player0, learnerPoints, unitsOnBoard: learnerUnits),
                    new PlayerState(PlayerId.Player1, opponentPoints, unitsOnBoard: opponentUnits),
                }, PlayerId.Player0, round: 3, nextEntityId: 100);
                return Build(layout, state, learnerUnits, opponentUnits);
            }

            public Fixture WithState(UnitSpec[] learner, UnitSpec[] opponent,
                int learnerPoints, int opponentPoints)
            {
                Unit[] learnerUnits = MakeUnits(PlayerId.Player0, learner,
                    State.Player(PlayerId.Player0).UnitsOnBoard.Select(unit => unit.Cell).ToArray());
                Unit[] opponentUnits = MakeUnits(PlayerId.Player1, opponent,
                    State.Player(PlayerId.Player1).UnitsOnBoard.Select(unit => unit.Cell).ToArray());
                var state = new GameState(State.Board, State.Config, new[]
                {
                    new PlayerState(PlayerId.Player0, learnerPoints, unitsOnBoard: learnerUnits),
                    new PlayerState(PlayerId.Player1, opponentPoints, unitsOnBoard: opponentUnits),
                }, PlayerId.Player0, State.Round, State.NextEntityId);
                return Build(Layout, state, learnerUnits, opponentUnits);
            }

            public TacticalV2EpisodeContext Episode(float pointsWeight = 0f) =>
                new TacticalV2EpisodeContext(State, "test-profile", PlayerId.Player0,
                    PlayerId.Player0, pointsWeight);

            public TacticalV2DecisionContext Context(int learnerAction)
            {
                bool[] mask = TacticalV2Coding.Mask(State, PlayerId.Player0, Layout, Slots0);
                Command command = TacticalV2Coding.Decode(
                    learnerAction, State, PlayerId.Player0, Layout, Slots0);
                return new TacticalV2DecisionContext(State, PlayerId.Player0, 0,
                    TacticalV2Coding.Observe(State, PlayerId.Player0, Layout, Slots0, Slots1),
                    mask, learnerAction, command, Slots0, Slots1, Layout);
            }

            private static Fixture Build(TacticalV2Layout layout, GameState state,
                Unit[] learnerUnits, Unit[] opponentUnits)
            {
                var slots0 = new TacticalV2UnitRegistry(layout.UnitSlotCount);
                var slots1 = new TacticalV2UnitRegistry(layout.UnitSlotCount);
                slots0.Initialize(learnerUnits,
                    Enumerable.Repeat(0, learnerUnits.Length).ToArray());
                slots1.Initialize(opponentUnits,
                    Enumerable.Repeat(0, opponentUnits.Length).ToArray());
                return new Fixture(layout, state, slots0, slots1);
            }

            private static Unit[] MakeUnits(PlayerId seat, UnitSpec[] specs, HexCoord[] cells) =>
                specs.Select((spec, index) =>
                {
                    Unit unit = new Unit(spec.Id + (seat == PlayerId.Player1 ? 50 : 0),
                        seat, spec.Stats, cells[index], 0);
                    return spec.DamageTaken == 0
                        ? unit
                        : unit.WithDamage(spec.DamageTaken);
                }).ToArray();
        }
    }
}
