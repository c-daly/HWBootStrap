using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Nodes;
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
        public void Observer_RejectsMaskedTeacherActionBeforeSinkEmission()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);
            bool[] mask = context.LegalMask;
            int maskedAction = Enumerable.Range(0, mask.Length).First(action => !mask[action]);
            var oracle = new FixedOracle(decision => OracleDecision(
                maskedAction, decision.LearnerCommand));
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(oracle, sink);
            observer.Reset(fixture.Episode());

            InvalidOperationException? error = Assert.Throws<InvalidOperationException>(
                () => observer.Observe(context));

            Assert.Multiple(() =>
            {
                Assert.That(error!.Message, Does.Contain("teacher action is masked"));
                Assert.That(oracle.DecisionCount, Is.EqualTo(1));
                Assert.That(sink.Drain(), Is.Empty);
            });
        }

        [Test]
        public void Observer_RejectsTeacherCommandActionMismatchBeforeSinkEmission()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);
            var oracle = new FixedOracle(decision => OracleDecision(
                decision.LearnerAction, new EndTurn(decision.Seat)));
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(oracle, sink);
            observer.Reset(fixture.Episode());

            InvalidOperationException? error = Assert.Throws<InvalidOperationException>(
                () => observer.Observe(context));

            Assert.Multiple(() =>
            {
                Assert.That(error!.Message,
                    Does.Contain("teacher command does not encode to its recorded action"));
                Assert.That(oracle.DecisionCount, Is.EqualTo(1));
                Assert.That(sink.Drain(), Is.Empty);
            });
        }

        [Test]
        public void Observer_RejectsLearnerCommandActionMismatchBeforeSinkEmission()
        {
            Fixture fixture = Fixture.Create(Units(1, weak: true), Units(1));
            TacticalV2DecisionContext valid = fixture.Context(fixture.FirstProductiveAction);
            var context = new TacticalV2DecisionContext(
                valid.State,
                valid.Seat,
                valid.DecisionIndex,
                valid.Observation,
                valid.LegalMask,
                valid.LearnerAction,
                new EndTurn(valid.Seat),
                valid.OwnRegistry,
                valid.FoeRegistry,
                valid.Layout);
            var oracle = new FixedOracle(decision =>
                OracleDecision(action: 0, new EndTurn(decision.Seat)));
            var sink = new BufferedTacticalV2DaggerSink { Enabled = true };
            var observer = new SelectiveDaggerObserver(oracle, sink);
            observer.Reset(fixture.Episode());

            InvalidOperationException? error = Assert.Throws<InvalidOperationException>(
                () => observer.Observe(context));

            Assert.Multiple(() =>
            {
                Assert.That(error!.Message,
                    Does.Contain("learner command does not encode to its recorded action"));
                Assert.That(oracle.DecisionCount, Is.EqualTo(1));
                Assert.That(sink.Drain(), Is.Empty);
            });
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

        [TestCase("""{"cmd":"duel_dagger_configure","enabled":1,"depth":4,"expansion_budget":512,"use_heuristic":true}""", "enabled flag")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":true,"expansion_budget":512,"use_heuristic":true}""", "depth")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":4,"expansion_budget":"512","use_heuristic":true}""", "expansion budget")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":0,"expansion_budget":512,"use_heuristic":true}""", "depth")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":4,"expansion_budget":512,"use_heuristic":false}""", "heuristic")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":4,"expansion_budget":512}""", "exactly")]
        [TestCase("""{"cmd":"duel_dagger_configure","enabled":true,"depth":4,"expansion_budget":512,"use_heuristic":true,"extra":1}""", "exactly")]
        public void GymServer_DaggerConfigureRejectsMalformedOrUnsupportedValues(
            string request, string expectedError)
        {
            using var server = new DaggerServerProcess("--environment", "tactical-v2");

            string error = server.ExchangeFailureRaw(request);

            Assert.That(error, Does.Contain(expectedError));
        }

        [TestCase("duel_dagger_configure")]
        [TestCase("duel_dagger_drain")]
        public void GymServer_DaggerCommandsRejectNonTacticalV2(string command)
        {
            using var server = new DaggerServerProcess("--environment", "adaptive-v1");
            object request = command == "duel_dagger_configure"
                ? new
                {
                    cmd = command,
                    enabled = true,
                    depth = 4,
                    expansion_budget = 512,
                    use_heuristic = true,
                }
                : new { cmd = command };

            string error = server.ExchangeFailure(request);

            Assert.That(error, Does.Contain(
                "duel DAgger is supported only for tactical-v2"));
        }

        [Test]
        public void GymServer_DaggerConfigureRejectsFogBeforeEnabling()
        {
            string scenario = WriteProfiledScenario(fogOfWar: true);
            try
            {
                using var server = new DaggerServerProcess(
                    "--environment", "tactical-v2", "--scenario-file", scenario);

                string error = server.ExchangeFailure(new
                {
                    cmd = "duel_dagger_configure",
                    enabled = true,
                    depth = 4,
                    expansion_budget = 512,
                    use_heuristic = true,
                });

                Assert.That(error, Does.Contain("requires fog_of_war=false"));
            }
            finally
            {
                File.Delete(scenario);
            }
        }

        [Test]
        public void GymServer_DaggerJsonlKeepsLearnerTransitionAndBuffersIndependent()
        {
            using var server = DaggerServerProcess.Profiled();

            using JsonDocument configured = server.Exchange(new
            {
                cmd = "duel_dagger_configure",
                enabled = true,
                depth = 4,
                expansion_budget = 512,
                use_heuristic = true,
            });
            Assert.That(
                configured.RootElement.EnumerateObject().Select(item => item.Name).ToArray(),
                Is.EquivalentTo(new[]
                {
                    "enabled", "depth", "expansion_budget", "use_heuristic",
                }));
            Assert.Multiple(() =>
            {
                Assert.That(configured.RootElement.GetProperty("enabled").GetBoolean(), Is.True);
                Assert.That(configured.RootElement.GetProperty("depth").GetInt32(), Is.EqualTo(4));
                Assert.That(configured.RootElement.GetProperty("expansion_budget").GetInt32(),
                    Is.EqualTo(512));
                Assert.That(configured.RootElement.GetProperty("use_heuristic").GetBoolean(), Is.True);
            });

            using JsonDocument traceEnabled = server.Exchange(
                new { cmd = "duel_trace_enable", enabled = true });
            using JsonDocument demoEnabled = server.Exchange(
                new { cmd = "duel_demo_enable", enabled = true });
            using JsonDocument reset = server.Exchange(new
            {
                cmd = "duel_reset",
                seed = 91,
                p0 = "external",
                p1 = "external",
                learner = 0,
                start_profile = "conversion-3v1-near",
                reference_seat = 0,
            });
            Assert.That(reset.RootElement.GetProperty("mask")[0].GetBoolean(), Is.True);

            using JsonDocument step = server.Exchange(new { cmd = "duel_step", action = 0 });
            using JsonDocument dagger = server.Exchange(new { cmd = "duel_dagger_drain" });
            Assert.That(
                dagger.RootElement.EnumerateObject().Select(item => item.Name).ToArray(),
                Is.EquivalentTo(new[] { "schema_version", "decisions" }));
            Assert.That(dagger.RootElement.GetProperty("schema_version").GetInt32(), Is.EqualTo(1));
            JsonElement decisions = dagger.RootElement.GetProperty("decisions");
            Assert.That(decisions.GetArrayLength(), Is.EqualTo(1));
            JsonElement row = decisions[0];
            Assert.That(
                row.EnumerateObject().Select(item => item.Name).ToArray(),
                Is.EquivalentTo(new[]
                {
                    "Observation", "LegalMask", "LearnerAction", "LearnerCommand",
                    "TeacherAction", "TeacherCommand", "Reasons", "StateHash",
                    "NormalizedAdvantage", "OpponentLivingUnitCount",
                    "ProductiveLegalActionCount", "Seat", "Round", "DecisionIndex",
                    "Disagreement", "OracleDepth", "OracleExpansionBudget",
                    "OracleHeuristicIdentity", "OracleActualExpansionCount",
                }));

            using JsonDocument trace = server.Exchange(new { cmd = "duel_trace_drain" });
            JsonElement transitions = trace.RootElement.GetProperty("transitions");
            Assert.That(transitions.GetArrayLength(), Is.EqualTo(1));
            JsonElement applied = transitions[0].GetProperty("Command");
            JsonElement learnerCommand = row.GetProperty("LearnerCommand");
            JsonElement teacherCommand = row.GetProperty("TeacherCommand");
            Assert.Multiple(() =>
            {
                Assert.That(row.GetProperty("LearnerAction").GetInt32(), Is.Zero);
                Assert.That(row.GetProperty("TeacherAction").GetInt32(), Is.Not.Zero);
                Assert.That(learnerCommand.GetRawText(), Is.EqualTo(applied.GetRawText()),
                    "the external learner command must own the transition");
                Assert.That(teacherCommand.GetRawText(), Is.Not.EqualTo(applied.GetRawText()),
                    "the teacher command must remain evidence only");
            });

            using JsonDocument demoAfterDaggerDrain =
                server.Exchange(new { cmd = "duel_demo_drain" });
            Assert.That(
                demoAfterDaggerDrain.RootElement.GetProperty("decisions").GetArrayLength(),
                Is.EqualTo(1), "draining DAgger must not clear demonstrations");
            using JsonDocument emptyDagger = server.Exchange(new { cmd = "duel_dagger_drain" });
            Assert.That(emptyDagger.RootElement.GetProperty("decisions").GetArrayLength(), Is.Zero);

            using JsonDocument resetBeforeDisable = server.Exchange(new
            {
                cmd = "duel_reset",
                seed = 92,
                p0 = "external",
                p1 = "external",
                learner = 0,
                start_profile = "conversion-3v1-near",
                reference_seat = 0,
            });
            Assert.That(
                resetBeforeDisable.RootElement.GetProperty("mask")[0].GetBoolean(), Is.True);
            using JsonDocument stepBeforeDisable =
                server.Exchange(new { cmd = "duel_step", action = 0 });

            using JsonDocument disabled = server.Exchange(new
            {
                cmd = "duel_dagger_configure",
                enabled = false,
                depth = 4,
                expansion_budget = 512,
                use_heuristic = true,
            });
            Assert.That(disabled.RootElement.GetProperty("enabled").GetBoolean(), Is.False);
            using JsonDocument clearedByDisable =
                server.Exchange(new { cmd = "duel_dagger_drain" });
            Assert.That(
                clearedByDisable.RootElement.GetProperty("decisions").GetArrayLength(),
                Is.Zero, "disabling DAgger must clear evidence buffered before disable");
            using JsonDocument demoSurvivesDisable =
                server.Exchange(new { cmd = "duel_demo_drain" });
            Assert.That(
                demoSurvivesDisable.RootElement.GetProperty("decisions").GetArrayLength(),
                Is.EqualTo(1), "disabling DAgger must not clear demonstrations");

            using JsonDocument resetAfterDisable = server.Exchange(new
            {
                cmd = "duel_reset",
                seed = 93,
                p0 = "external",
                p1 = "external",
                learner = 0,
                start_profile = "conversion-3v1-near",
                reference_seat = 0,
            });
            using JsonDocument stepAfterDisable =
                server.Exchange(new { cmd = "duel_step", action = 0 });
            using JsonDocument stillEmpty = server.Exchange(new { cmd = "duel_dagger_drain" });
            Assert.That(stillEmpty.RootElement.GetProperty("decisions").GetArrayLength(), Is.Zero);
            using JsonDocument demoStillIndependent =
                server.Exchange(new { cmd = "duel_demo_drain" });
            Assert.That(
                demoStillIndependent.RootElement.GetProperty("decisions").GetArrayLength(),
                Is.EqualTo(1), "disabling DAgger must not disable demonstrations");
        }


        [Test]
        public void OraclePreflightActionOracle_QueriesTwiceAndReturnsFirstWithoutMutatingState()
        {
            var inner = new CountingOracle(context => OracleDecision(
                context.LearnerAction, context.LearnerCommand));
            var sink = new BufferedOraclePreflightBenchmarkSink();
            var timestamps = new Queue<long>(new[] { 100L, 111L, 200L, 223L });
            var preflight = new OraclePreflightActionOracle(inner, sink,
                () => timestamps.Dequeue(), clockFrequency: 1_000);
            var observer = new PreflightObserver(preflight);
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default())
            {
                CaptureTransitions = true,
                DecisionObserver = observer,
            };
            TacticalV2DuelEnv.View view = env.Reset(91, null, null);
            int learnerAction = FirstLegalNonEndTurn(view.ActionMask);

            env.Step(learnerAction);

            OraclePreflightBenchmarkRecord record = sink.Drain().Single();
            DuelTransition applied = env.DrainTransitions().Single();
            Assert.Multiple(() =>
            {
                Assert.That(inner.DecisionCount, Is.EqualTo(2));
                Assert.That(JsonSerializer.Serialize(observer.Before),
                    Is.EqualTo(JsonSerializer.Serialize(observer.After)));
                Assert.That(observer.Result!.Action, Is.EqualTo(learnerAction));
                Assert.That(applied.Command, Is.EqualTo(observer.Context!.LearnerCommand));
                Assert.That(record.FirstElapsedTicks, Is.EqualTo(11));
                Assert.That(record.SecondElapsedTicks, Is.EqualTo(23));
            });
        }

        [Test]
        public void OraclePreflightActionOracle_RejectsDifferentRepeatedDecisions()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(2));
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);
            var inner = new SequenceOracle((decision, count) => count == 1
                ? OracleDecision(decision.LearnerAction, decision.LearnerCommand)
                : OracleDecision(0, TacticalV2Coding.Decode(0, decision.State,
                    decision.Seat, decision.Layout, decision.OwnRegistry)));
            var preflight = new OraclePreflightActionOracle(inner,
                new BufferedOraclePreflightBenchmarkSink(), () => 1L, clockFrequency: 1);

            Assert.That(() => preflight.Decide(context), Throws.TypeOf<InvalidOperationException>());
            Assert.That(inner.DecisionCount, Is.EqualTo(2));
        }

        [Test]
        public void OraclePreflightActionOracle_RejectsContextMutationBetweenQueries()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(2));
            var moved = new List<int>();
            var state = new GameState(fixture.State.Board, fixture.State.Config,
                fixture.State.Players, fixture.State.ActivePlayer, fixture.State.Round,
                fixture.State.NextEntityId, movedUnitIds: moved);
            bool[] mask = TacticalV2Coding.Mask(state, PlayerId.Player0, fixture.Layout, fixture.Slots0);
            int action = fixture.FirstProductiveAction;
            var context = new TacticalV2DecisionContext(state, PlayerId.Player0, 0,
                TacticalV2Coding.Observe(state, PlayerId.Player0, fixture.Layout,
                    fixture.Slots0, fixture.Slots1),
                mask, action, TacticalV2Coding.Decode(action, state, PlayerId.Player0,
                    fixture.Layout, fixture.Slots0), fixture.Slots0, fixture.Slots1, fixture.Layout);
            var inner = new SequenceOracle((decision, count) =>
            {
                if (count == 1) moved.Add(1);
                return OracleDecision(decision.LearnerAction, decision.LearnerCommand);
            });
            var preflight = new OraclePreflightActionOracle(inner,
                new BufferedOraclePreflightBenchmarkSink(), () => 1L, clockFrequency: 1);

            Assert.That(() => preflight.Decide(context), Throws.TypeOf<InvalidOperationException>());
            Assert.That(inner.DecisionCount, Is.EqualTo(1));
        }

        [Test]
        public void OraclePreflightBenchmarkRecord_DefensivelyCopiesObservationMaskStateAndCommands()
        {
            Fixture fixture = Fixture.Create(Units(2), Units(2));
            TacticalV2DecisionContext context = fixture.Context(fixture.FirstProductiveAction);
            float[] observation = context.Observation;
            bool[] mask = context.LegalMask;
            TacticalTraceState state = TacticalEvaluationTrace.Project(
                new DuelTransition(context.State, context.LearnerCommand, context.State)).Before;
            TacticalV2OracleDecision first = OracleDecision(context.LearnerAction,
                context.LearnerCommand);
            var record = new OraclePreflightBenchmarkRecord("state", context.DecisionIndex,
                observation, mask, state, first, first, 1L, 2L, 1_000L);

            observation[0] = observation[0] + 1f;
            mask[context.LearnerAction] = !mask[context.LearnerAction];
            state.Seats[0].Points = 99;

            Assert.Multiple(() =>
            {
                Assert.That(record.Observation[0], Is.Not.EqualTo(observation[0]));
                Assert.That(record.LegalMask[context.LearnerAction], Is.True);
                Assert.That(record.State.Seats[0].Points, Is.Not.EqualTo(99));
                Assert.That(record.First.Command, Is.Not.SameAs(first.Command));
                Assert.That(record.Second.Command, Is.Not.SameAs(first.Command));
            });
        }

        private static int FirstLegalNonEndTurn(bool[] mask)
        {
            for (int action = 1; action < mask.Length; action++)
                if (mask[action]) return action;
            Assert.Fail("test requires a legal non-EndTurn action");
            return -1;
        }

        private sealed class CountingOracle : IActionOracle
        {
            private readonly Func<TacticalV2DecisionContext, TacticalV2OracleDecision> _decide;

            public CountingOracle(Func<TacticalV2DecisionContext, TacticalV2OracleDecision> decide)
            {
                _decide = decide;
            }

            public int DecisionCount { get; private set; }

            public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
            {
                DecisionCount++;
                return _decide(context);
            }
        }

        private sealed class SequenceOracle : IActionOracle
        {
            private readonly Func<TacticalV2DecisionContext, int, TacticalV2OracleDecision> _decide;

            public SequenceOracle(
                Func<TacticalV2DecisionContext, int, TacticalV2OracleDecision> decide)
            {
                _decide = decide;
            }

            public int DecisionCount { get; private set; }

            public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
            {
                DecisionCount++;
                return _decide(context, DecisionCount);
            }
        }

        private sealed class PreflightObserver : ITacticalV2DecisionObserver
        {
            private readonly OraclePreflightActionOracle _preflight;

            public PreflightObserver(OraclePreflightActionOracle preflight)
            {
                _preflight = preflight;
            }

            public TacticalV2DecisionContext? Context { get; private set; }
            public TacticalV2OracleDecision? Result { get; private set; }
            public TacticalTraceState? Before { get; private set; }
            public TacticalTraceState? After { get; private set; }

            public void Reset(TacticalV2EpisodeContext episode) { }

            public void Observe(TacticalV2DecisionContext decision)
            {
                Context = decision;
                Before = TacticalEvaluationTrace.Project(
                    new DuelTransition(decision.State, decision.LearnerCommand, decision.State)).Before;
                Result = _preflight.Decide(decision);
                After = TacticalEvaluationTrace.Project(
                    new DuelTransition(decision.State, decision.LearnerCommand, decision.State)).Before;
            }
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

        private static TacticalV2OracleDecision OracleDecision(int action, Command command) =>
            new TacticalV2OracleDecision(action, command,
                depth: 4, expansionBudget: 512,
                heuristicIdentity: BoundedSearchAgent.HeuristicIdentity,
                actualExpansionCount: 7);

        private sealed class FixedOracle : IActionOracle
        {
            private readonly Func<TacticalV2DecisionContext, TacticalV2OracleDecision> _decide;

            public FixedOracle(
                Func<TacticalV2DecisionContext, TacticalV2OracleDecision> decide)
            {
                _decide = decide;
            }

            public int DecisionCount { get; private set; }

            public TacticalV2OracleDecision Decide(TacticalV2DecisionContext context)
            {
                DecisionCount++;
                return _decide(context);
            }
        }

        private static string WriteProfiledScenario(bool fogOfWar)
        {
            string source = Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "..", "python", "config",
                "annihilation-imitation-v1.json"));
            JsonNode scenario = JsonNode.Parse(File.ReadAllText(source))!;
            scenario["rules"]!["fog_of_war"] = fogOfWar;
            string path = Path.Combine(
                TestContext.CurrentContext.WorkDirectory,
                "dagger-profile-" + Guid.NewGuid().ToString("N") + ".json");
            File.WriteAllText(path, scenario.ToJsonString());
            return path;
        }

        private sealed class DaggerServerProcess : IDisposable
        {
            private readonly Process _process;
            private readonly string? _ownedScenario;
            private static string ServerDll => Path.GetFullPath(Path.Combine(
                TestContext.CurrentContext.TestDirectory,
                "..", "..", "..", "..", "HexWars.GymServer", "bin", "Debug", "net8.0",
                "HexWars.GymServer.dll"));

            public static DaggerServerProcess Profiled(bool fogOfWar = false)
            {
                string scenario = WriteProfiledScenario(fogOfWar);
                return new DaggerServerProcess(
                    scenario,
                    "--environment", "tactical-v2", "--scenario-file", scenario);
            }

            private DaggerServerProcess(string ownedScenario, params string[] args)
                : this(args)
            {
                _ownedScenario = ownedScenario;
            }


            public DaggerServerProcess(params string[] args)
            {
                Assert.That(File.Exists(ServerDll), Is.True,
                    $"GymServer was not built at {ServerDll}");
                string arguments = $"\"{ServerDll}\" " +
                    string.Join(" ", args.Select(value => $"\"{value}\""));
                _process = Process.Start(new ProcessStartInfo("dotnet", arguments)
                {
                    RedirectStandardInput = true,
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                })!;
            }

            public JsonDocument Exchange(object request) =>
                ExchangeRaw(JsonSerializer.Serialize(request));

            public JsonDocument ExchangeRaw(string request)
            {
                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();
                var pending = _process.StandardOutput.ReadLineAsync();
                if (!pending.Wait(TimeSpan.FromSeconds(5)))
                    Assert.Fail("GymServer did not reply to the DAgger request");
                string? line = pending.Result;
                if (line == null)
                    Assert.Fail(
                        $"GymServer exited without a reply: {_process.StandardError.ReadToEnd()}");
                return JsonDocument.Parse(line!);
            }

            public string ExchangeFailure(object request) =>
                ExchangeFailureRaw(JsonSerializer.Serialize(request));

            public string ExchangeFailureRaw(string request)
            {
                _process.StandardInput.WriteLine(request);
                _process.StandardInput.Flush();
                Assert.That(_process.WaitForExit(10000), Is.True,
                    "GymServer did not reject the DAgger request");
                Assert.That(_process.StandardOutput.ReadLine(), Is.Null);
                return _process.StandardError.ReadToEnd();
            }

            public void Dispose()
            {
                if (_process.HasExited)
                {
                    _process.Dispose();
                    if (_ownedScenario != null) File.Delete(_ownedScenario);
                    return;
                }
                _process.StandardInput.WriteLine("{\"cmd\":\"close\"}");
                _process.StandardInput.Flush();
                if (!_process.WaitForExit(5000)) _process.Kill(entireProcessTree: true);
                _process.Dispose();
                if (_ownedScenario != null) File.Delete(_ownedScenario);
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
