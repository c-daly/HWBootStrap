using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2DuelEnvTests
    {
        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void ProfiledReset_AppliesDeclaredAdvantageToExplicitReferenceSeat(PlayerId referenceSeat)
        {
            TacticalV2Config config = ProfiledConfig();
            var env = new TacticalV2DuelEnv(config);

            TacticalV2DuelEnv.View view = env.Reset(
                6000005,
                controller0: null,
                controller1: null,
                startProfileId: "conversion-2v1-far",
                referenceSeat: referenceSeat,
                learnerSeat: PlayerId.Player0);

            PlayerId foe = referenceSeat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            Assert.That(env.State.Player(referenceSeat).UnitsOnBoard, Has.Count.EqualTo(2));
            Assert.That(env.State.Player(foe).UnitsOnBoard, Has.Count.EqualTo(1));
            Assert.That(env.SelectedStartProfileId, Is.EqualTo("conversion-2v1-far"));
            Assert.That(env.ReferenceSeat, Is.EqualTo(referenceSeat));
            Assert.That(view.StartProfileId, Is.EqualTo("conversion-2v1-far"));
            Assert.That(view.ReferenceSeat, Is.EqualTo(referenceSeat));
        }

        private static TacticalV2Config ProfiledConfig()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.PlacementPolicy = "profiled-seeded-v1";
            config.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1();
            config.StartDistribution = new TacticalV2StartDistribution(config.StartProfiles.Select(profile =>
                new TacticalV2StartWeight(profile.Id, profile.Id == "standard-3v3" ? 10000 : 0)));
            return config;
        }
        private static TacticalV2Config ProductionProfiledConfig()
        {
            TrainingScenario scenario = TrainingScenario.CreateStandard("tactical-v2");
            scenario.TacticalV2.PlacementPolicy = "profiled-seeded-v1";
            scenario.TacticalV2.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1().ToList();
            scenario.TacticalV2.StartDistribution = scenario.TacticalV2.StartProfiles
                .Select(profile => new TacticalV2StartWeight(
                    profile.Id, profile.Id == "standard-3v3" ? 10000 : 0))
                .ToList();
            return scenario.BuildTacticalV2();
        }
        /// <summary>Regression: an internal scripted controller (Greedy) decides purely from raw engine
        /// legality — board cells and points, never the RL registry's synthetic per-seat capacity — so
        /// nothing stops it from proposing a DeployUnit once every registry slot already holds a living
        /// unit. Before the fix, TryApply forwarded that command straight to
        /// <see cref="TacticalV2UnitRegistry.RegisterDeployment"/>, which throws when the registry is
        /// full. Two fully internal Greedy controllers playing to completion, for every starting seed in
        /// this range, must never throw.</summary>
        [Test]
        public void TwoInternalGreedyControllers_PlayToEnd_NeverOverflowRegistryCapacity()
        {
            TacticalV2Config config = TacticalV2Config.Default();

            for (int seed = 0; seed < 40; seed++)
            {
                var env = new TacticalV2DuelEnv(config);
                var view = env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
                Assert.That(view.Terminated || view.Truncated, Is.True);
            }
        }

        /// <summary>100-round-rule correctness check (duel side): when the ENGINE ends the game at its own
        /// round cap, the duel view must report a terminal (not truncated) result independent of MaxSteps
        /// — even a MaxSteps so huge it could never fire. See the matching
        /// <see cref="TacticalV2EnvTests.RoundCapReachedByEngine_IsReportedAsTerminatedNeverTruncated_EvenWithHugeMaxSteps"/>.</summary>
        [Test]
        public void RoundCapReachedByEngine_IsReportedAsTerminatedNeverTruncated_EvenWithHugeMaxSteps()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.Game = new GameConfig(new Dictionary<TerrainType, TerrainDef>(), biomesEnabled: false, roundCap: 2);
            config.MaxSteps = 1_000_000; // deliberately absurd: proves truncation cannot be what ends this

            var env = new TacticalV2DuelEnv(config);
            env.Reset(0, controller0: null, controller1: new AlwaysEndTurnAgent(), learnerSeat: PlayerId.Player0);

            TacticalV2DuelEnv.View view = env.Step(0); // learner ends its own turn; internal seat ends -> round cap hit

            Assert.That(view.Terminated, Is.True);
            Assert.That(view.Truncated, Is.False);
        }

        [Test]
        public void FullyScriptedRoundCapDraw_UsesMinusOneWinnerSentinel()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.Game = new GameConfig(
                new Dictionary<TerrainType, TerrainDef>(),
                biomesEnabled: false,
                roundCap: 1,
                captureCost: int.MaxValue,
                generatorsEnabled: false,
                fixedTemplateCount: config.Templates.Count,
                templateSlotCount: config.Templates.Count);
            var env = new TacticalV2DuelEnv(config);

            TacticalV2DuelEnv.View view = env.Reset(
                73, new AlwaysEndTurnAgent(), new AlwaysEndTurnAgent());

            Assert.That(view.Terminated, Is.True);
            Assert.That(view.Truncated, Is.False);
            Assert.That(view.Winner, Is.EqualTo(-1));
        }

        private sealed class AlwaysEndTurnAgent : IAgent
        {
            public Command Decide(GameState state) => new EndTurn(state.ActivePlayer);
        }

        private static int PickLegal(bool[] mask, Random rng)
        {
            var legal = new List<int>();
            for (int i = 0; i < mask.Length; i++) if (mask[i]) legal.Add(i);
            return legal[rng.Next(legal.Count)];
        }

        /// <summary>Asserts the drained transitions are exactly the accepted commands in order — same
        /// count and same commands (record equality) as the replay log — and that consecutive
        /// transitions chain by reference (transition[i].Resulting IS transition[i+1].Previous), so a
        /// consumer can play them back as one continuous sequence. Also asserts transitions[0].Previous
        /// is the episode's actual start state (the playback anchor a viewer seeds from): GameState has
        /// no value equality, so re-serializing it alongside the replay's own commands and checking that
        /// reproduces the replay text byte-for-byte is the cheapest structural proof available.</summary>
        private static void AssertTransitionsMatchReplay(TacticalV2DuelEnv env, IReadOnlyList<DuelTransition> transitions)
        {
            string replayText = env.ToReplay();
            ReplayData data = ReplayFile.Read(replayText);
            Assert.That(transitions.Count, Is.EqualTo(data.Commands.Count));
            for (int i = 0; i < transitions.Count; i++)
                Assert.That(transitions[i].Command, Is.EqualTo(data.Commands[i]));
            for (int i = 0; i < transitions.Count - 1; i++)
                Assert.That(transitions[i].Resulting, Is.SameAs(transitions[i + 1].Previous));
            if (transitions.Count > 0)
            {
                Assert.That(transitions[transitions.Count - 1].Resulting, Is.SameAs(env.State));
                Assert.That(ReplayFile.Write(transitions[0].Previous, data.Commands), Is.EqualTo(replayText),
                    "transitions[0].Previous must be the episode's start state (the playback anchor)");
            }
        }

        [Test]
        public void TwoInternalGreedyControllers_ProduceOneOrderedTransitionPerAcceptedCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.CaptureTransitions = true;
            var view = env.Reset(3, new GreedyAgent(3), new GreedyAgent(4));
            Assert.That(view.Terminated || view.Truncated, Is.True);

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            AssertTransitionsMatchReplay(env, transitions);
        }

        [Test]
        public void ExternalAndInternalSteps_ProduceOneOrderedTransitionPerAcceptedCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.CaptureTransitions = true;
            var rng = new Random(11);
            var view = env.Reset(11, null, new GreedyAgent(3)); // seat0 external, seat1 internal

            int steps = 0;
            while (!view.Terminated && !view.Truncated && steps < 4000)
            {
                view = env.Step(PickLegal(view.ActionMask, rng));
                steps++;
            }
            Assert.That(view.Terminated || view.Truncated, Is.True);

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            Assert.That(transitions, Is.Not.Empty);
            AssertTransitionsMatchReplay(env, transitions);
        }

        /// <summary>Across a scripted-vs-scripted episode segment, at least one accepted transition must
        /// be an AttackUnit — the viewer needs these specifically to drive projectile/impact/kill
        /// presentation instead of inferring an attack from a coarse before/after diff.</summary>
        [Test]
        public void ScriptedVsScripted_AttackCommandsAppearAsAttackUnitTransitions()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            bool foundAttack = false;

            for (int seed = 0; seed < 20 && !foundAttack; seed++)
            {
                var env = new TacticalV2DuelEnv(config);
                env.CaptureTransitions = true;
                env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
                IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
                if (transitions.Any(t => t.Command is AttackUnit)) foundAttack = true;
            }

            Assert.That(foundAttack, Is.True, "expected at least one AttackUnit transition across seeds 0..19");
        }

        /// <summary>A masked-off move destination — decoded into a real MoveUnit for a slot that holds a
        /// living unit at the very start of the episode, so it can't degrade to the always-legal EndTurn
        /// fallback — is rejected by the engine and must not produce a transition or change state.</summary>
        [Test]
        public void RejectedExternalAction_ProducesNoTransition()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.CaptureTransitions = true;
            var view = env.Reset(7, null, null); // both seats external; nothing auto-plays

            TacticalV2Layout layout = env.Layout;
            int n = layout.CellCount;
            int rejectedAction = -1;
            for (int i = layout.MoveOffset; i < layout.AttackOffset; i++)
            {
                if (!view.ActionMask[i]) { rejectedAction = i; break; }
            }
            Assert.That(rejectedAction, Is.GreaterThanOrEqualTo(0),
                "expected at least one masked-off move destination at the start of the episode");

            GameState before = env.State;
            env.Step(rejectedAction);

            Assert.That(env.State, Is.SameAs(before), "a rejected command must not change engine state");
            Assert.That(env.DrainTransitions(), Is.Empty);
        }

        [Test]
        public void Reset_ClearsPendingTransitions()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.CaptureTransitions = true;
            env.Reset(3, new GreedyAgent(3), new GreedyAgent(4)); // plays to completion; NOT drained

            env.Reset(9, new GreedyAgent(9), new GreedyAgent(10)); // reset again without draining

            IReadOnlyList<DuelTransition> transitions = env.DrainTransitions();
            AssertTransitionsMatchReplay(env, transitions); // must reflect only the second game
        }

        [Test]
        public void DrainTransitions_EmptiesTheQueue()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            env.CaptureTransitions = true;
            env.Reset(3, new GreedyAgent(3), new GreedyAgent(4));

            Assert.That(env.DrainTransitions(), Is.Not.Empty);
            Assert.That(env.DrainTransitions(), Is.Empty);
        }

        /// <summary>With CaptureTransitions left at its default (false), a scripted episode segment
        /// still plays to completion and produces a valid replay, but nothing accumulates for
        /// DrainTransitions — proving headless training (which never touches the flag) pays no capture
        /// cost.</summary>
        [Test]
        public void CaptureTransitionsDefaultsOff_ScriptedEpisodeSegment_DrainsEmpty()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var env = new TacticalV2DuelEnv(config);
            var view = env.Reset(3, new GreedyAgent(3), new GreedyAgent(4)); // CaptureTransitions untouched

            Assert.That(view.Terminated || view.Truncated, Is.True);
            Assert.That(env.DrainTransitions(), Is.Empty, "capture is opt-in; training must pay nothing by default");
        }

        [Test]
        public void InjectedSink_RecordsAcceptedCommandsOnlyWhileEnabled()
        {
            var sink = new BufferedDuelTransitionSink { Enabled = true };
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), sink);

            env.Reset(11, new GreedyAgent(3), new GreedyAgent(4));

            Assert.That(sink.Drain(), Is.Not.Empty);
            Assert.That(sink.Drain(), Is.Empty);
        }

        [Test]
        public void InjectedSink_DisabledByDefault_RetainsNothing()
        {
            var sink = new BufferedDuelTransitionSink();
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), sink);

            env.Reset(11, new GreedyAgent(3), new GreedyAgent(4));

            Assert.That(sink.Drain(), Is.Empty);
        }

        [Test]
        public void InjectedSink_ExcludesRejectedExternalAction_AndKeepsAcceptedCommandsOrdered()
        {
            var sink = new BufferedDuelTransitionSink { Enabled = true };
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), sink);
            var view = env.Reset(7, null, null);

            TacticalV2Layout layout = env.Layout;
            int rejectedAction = -1;
            for (int i = layout.MoveOffset; i < layout.AttackOffset; i++)
            {
                if (!view.ActionMask[i]) { rejectedAction = i; break; }
            }
            Assert.That(rejectedAction, Is.GreaterThanOrEqualTo(0),
                "expected at least one masked-off move destination at the start of the episode");

            GameState start = env.State;
            env.Step(rejectedAction);
            Assert.That(env.State, Is.SameAs(start), "a rejected command must not change engine state");

            view = env.Step(PickLegal(view.ActionMask, new Random(29)));
            GameState afterFirstAccepted = env.State;
            env.Step(PickLegal(view.ActionMask, new Random(30)));

            IReadOnlyList<DuelTransition> transitions = sink.Drain();
            Assert.That(transitions.Count, Is.EqualTo(2), "only accepted external commands are forwarded to the sink");
            AssertTransitionsMatchReplay(env, transitions);
            Assert.That(transitions[0].Previous, Is.SameAs(start));
            Assert.That(transitions[0].Resulting, Is.SameAs(afterFirstAccepted));
        }

        [Test]
        public void BufferedDemonstrationSink_DefaultsDisabled()
        {
            var sink = new BufferedTacticalV2DemonstrationSink();
            sink.Accepted(new TacticalV2Demonstration(
                new[] { 1f }, new[] { true }, 0, 0,
                new TacticalTraceCommand { Kind = "end_turn", Issuer = 0 }));

            Assert.That(sink.Drain(), Is.Empty);
        }

        [Test]
        public void Demonstration_IsImmutable_AndBufferedSinkDrainsInDecisionOrder()
        {
            var observation = new[] { 0.25f, 0.75f };
            var legalMask = new[] { true, false };
            var command = new TacticalTraceCommand { Kind = "end_turn", Issuer = 0 };
            var first = new TacticalV2Demonstration(observation, legalMask, 0, 0, command);
            var second = new TacticalV2Demonstration(
                new[] { 0.5f, 0.5f }, new[] { false, true }, 1, 1,
                new TacticalTraceCommand { Kind = "move", Issuer = 1, ActorId = 7, Q = 2, R = 3 });

            observation[0] = 99f;
            legalMask[0] = false;
            command.Kind = "mutated";
            float[] returnedObservation = first.Observation;
            bool[] returnedMask = first.LegalMask;
            TacticalTraceCommand returnedCommand = first.Command;
            returnedObservation[0] = 88f;
            returnedMask[0] = false;
            returnedCommand.Kind = "also-mutated";

            Assert.That(first.Observation, Is.EqualTo(new[] { 0.25f, 0.75f }));
            Assert.That(first.LegalMask, Is.EqualTo(new[] { true, false }));
            Assert.That(first.Command.Kind, Is.EqualTo("end_turn"));

            var sink = new BufferedTacticalV2DemonstrationSink { Enabled = true };
            sink.Accepted(first);
            sink.Accepted(second);

            Assert.That(sink.Drain(), Is.EqualTo(new[] { first, second }));
            Assert.That(sink.Drain(), Is.Empty);
            sink.Accepted(first);
            sink.Reset();
            Assert.That(sink.Drain(), Is.Empty);
        }

        [Test]
        public void DemonstrationSink_CapturesAcceptedExternalActionFromItsExactPreActionView()
        {
            var sink = new BufferedTacticalV2DemonstrationSink { Enabled = true };
            var env = new TacticalV2DuelEnv(TacticalV2Config.Default(), demonstrationSink: sink);
            TacticalV2DuelEnv.View before = env.Reset(91, controller0: null, controller1: null);
            int action = FirstLegalNonEndTurn(before.ActionMask);
            GameState stateBefore = env.State;
            TacticalV2Start start = env.Layout.NewGame(91);
            Command accepted = TacticalV2Coding.Decode(
                action, stateBefore, before.Seat, env.Layout, start.Slots0);

            env.Step(action);

            TacticalV2Demonstration item = sink.Drain().Single();
            Assert.That(item.Seat, Is.EqualTo((int)PlayerId.Player0));
            Assert.That(item.Action, Is.EqualTo(action));
            Assert.That(item.LegalMask[item.Action], Is.True);
            Assert.That(item.Observation, Is.EqualTo(before.Observation));
            Assert.That(item.LegalMask, Is.EqualTo(before.ActionMask));
            AssertTraceCommandsEqual(
                TacticalEvaluationTrace.Project(new DuelTransition(stateBefore, accepted, env.State)).Command,
                item.Command);
        }

        [Test]
        public void DemonstrationCapture_IsTransparentForScriptedGreedyVersusRandomEpisode()
        {
            CapturedEpisode withoutCapture = PlayGreedyVersusRandom(capture: false);
            CapturedEpisode withCapture = PlayGreedyVersusRandom(capture: true);

            Assert.That(withCapture.Env.ToReplay(), Is.EqualTo(withoutCapture.Env.ToReplay()));
            Assert.That(
                withCapture.Transitions.Select(item => item.Command),
                Is.EqualTo(withoutCapture.Transitions.Select(item => item.Command)));
            Assert.That(withCapture.Rewards, Is.EqualTo(withoutCapture.Rewards));
            Assert.That(withCapture.View.Terminated, Is.EqualTo(withoutCapture.View.Terminated));
            Assert.That(withCapture.View.Truncated, Is.EqualTo(withoutCapture.View.Truncated));
            Assert.That(withCapture.View.Winner, Is.EqualTo(withoutCapture.View.Winner));
            AssertStatesEquivalent(withoutCapture.Env.State, withCapture.Env.State);

            Assert.That(withCapture.Demonstrations.Count, Is.EqualTo(withCapture.Transitions.Count));
            TacticalV2Start start = withCapture.Env.Layout.NewGame(91);
            for (int index = 0; index < withCapture.Demonstrations.Count; index++)
            {
                TacticalV2Demonstration decision = withCapture.Demonstrations[index];
                DuelTransition transition = withCapture.Transitions[index];
                PlayerId seat = transition.Command.Issuer;
                TacticalV2UnitRegistry own =
                    seat == PlayerId.Player0 ? start.Slots0 : start.Slots1;
                TacticalV2UnitRegistry foe =
                    seat == PlayerId.Player0 ? start.Slots1 : start.Slots0;

                Assert.That(decision.Seat, Is.EqualTo((int)seat));
                Assert.That(decision.Observation, Is.EqualTo(TacticalV2Coding.Observe(
                    transition.Previous, seat, withCapture.Env.Layout, own, foe)));
                Assert.That(decision.LegalMask, Is.EqualTo(TacticalV2Coding.Mask(
                    transition.Previous, seat, withCapture.Env.Layout, own)));
                Assert.That(TacticalV2Coding.TryEncode(
                    transition.Command, transition.Previous, withCapture.Env.Layout, own,
                    out int expectedAction), Is.True);
                Assert.That(decision.Action, Is.EqualTo(expectedAction));
                Assert.That(decision.LegalMask[decision.Action], Is.True);
                AssertTraceCommandsEqual(
                    TacticalEvaluationTrace.Project(transition).Command, decision.Command);

                start.Slots0.ReleaseDead(transition.Resulting, PlayerId.Player0);
                start.Slots1.ReleaseDead(transition.Resulting, PlayerId.Player1);
                if (transition.Command is DeployUnit deploy)
                {
                    own.RegisterDeployment(
                        transition.Previous, transition.Resulting, deploy.Issuer, deploy.TemplateIndex);
                }
            }
        }
        [Test]
        public void ProductionScenario_ClosesUnencodedEngineCommandFamilies()
        {
            TacticalV2Config config = ProductionProfiledConfig();

            Assert.Multiple(() =>
            {
                Assert.That(config.Game.CaptureCost, Is.EqualTo(int.MaxValue),
                    "CaptureHex has no tactical-v2 action label");
                Assert.That(config.Game.GeneratorsEnabled, Is.False,
                    "BuildGenerator has no tactical-v2 action label");
                Assert.That(config.Game.FixedTemplateCount, Is.EqualTo(config.Templates.Count),
                    "the tactical-v2 catalog must be an immutable fixed prefix");
                Assert.That(config.Game.TemplateSlotCount, Is.EqualTo(config.Templates.Count),
                    "CreateUnit and dynamic-template deploys have no tactical-v2 action labels");
            });
        }

        [Test]
        public void ProductionScenario_SmokeSeedAcceptsOnlyEncodedScriptedCommands()
        {
            const int seed = 11_000_000;
            TacticalV2Config config = ProductionProfiledConfig();
            var transitions = new BufferedDuelTransitionSink { Enabled = true };
            var env = new TacticalV2DuelEnv(config, transitions);
            TacticalV2StartProfile profile = config.StartProfiles.Single(
                item => item.Id == "standard-3v3");
            TacticalV2Start tracker = env.Layout.NewGame(seed, profile, PlayerId.Player0);

            TacticalV2DuelEnv.View view = env.Reset(
                seed,
                new GreedyAgent(seed * 2 + 1),
                new RandomAgent(seed * 2 + 2),
                profile.Id,
                PlayerId.Player0,
                PlayerId.Player0);
            Assert.That(view.Terminated || view.Truncated, Is.True);

            var unsupported = new List<string>();
            int index = 0;
            foreach (DuelTransition transition in transitions.Drain())
            {
                PlayerId seat = transition.Command.Issuer;
                TacticalV2UnitRegistry own =
                    seat == PlayerId.Player0 ? tracker.Slots0 : tracker.Slots1;
                if (!TacticalV2Coding.TryEncode(
                    transition.Command, transition.Previous, env.Layout, own, out _))
                {
                    unsupported.Add(
                        $"{index}:{transition.Command.GetType().Name}:" +
                        $"round={transition.Previous.Round}:seat={(int)seat}:" +
                        $"points={transition.Previous.Player(seat).Points}");
                }

                tracker.Slots0.ReleaseDead(transition.Resulting, PlayerId.Player0);
                tracker.Slots1.ReleaseDead(transition.Resulting, PlayerId.Player1);
                if (transition.Command is DeployUnit deploy)
                {
                    own.RegisterDeployment(
                        transition.Previous, transition.Resulting,
                        deploy.Issuer, deploy.TemplateIndex);
                }
                index++;
            }

            Assert.That(unsupported, Is.Empty,
                "scripted production episode accepted commands outside tactical-v2 encoding");
        }

        [Test]
        public void DemonstrationCapture_FailsHardWhenEngineAcceptsAnUnencodableCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.Game = GameConfig.Default(biomesEnabled: false, captureCost: 0);
            var sink = new BufferedTacticalV2DemonstrationSink { Enabled = true };
            var env = new TacticalV2DuelEnv(config, demonstrationSink: sink);

            Assert.Throws<InvalidOperationException>(() =>
                env.Reset(5, new ClaimFirstAgent(), new ClaimFirstAgent()));
        }

        [Test]
        public void DisabledDemonstrationSink_IsTransparentEvenForUnencodableAcceptedCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.Game = GameConfig.Default(biomesEnabled: false, captureCost: 0);
            var sink = new BufferedTacticalV2DemonstrationSink();
            var env = new TacticalV2DuelEnv(config, demonstrationSink: sink);

            TacticalV2DuelEnv.View view =
                env.Reset(5, new ClaimFirstAgent(), new ClaimFirstAgent());

            Assert.That(view.Terminated || view.Truncated, Is.True);
            Assert.That(sink.Drain(), Is.Empty);
        }

        private static CapturedEpisode PlayGreedyVersusRandom(bool capture)
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.Game = GameConfig.Default(
                biomesEnabled: false,
                captureCost: int.MaxValue,
                generatorsEnabled: false,
                fixedTemplateCount: config.Templates.Count,
                templateSlotCount: config.Templates.Count);
            var transitions = new BufferedDuelTransitionSink { Enabled = true };
            var demonstrations = new BufferedTacticalV2DemonstrationSink { Enabled = capture };
            var env = new TacticalV2DuelEnv(
                config, transitions,
                capture ? demonstrations : null);
            var greedy = new GreedyAgent(91);
            TacticalV2Start tracker = env.Layout.NewGame(91);
            var accepted = new List<DuelTransition>();
            TacticalV2DuelEnv.View view =
                env.Reset(91, controller0: null, controller1: new RandomAgent(92));
            var rewards = new List<float>();
            int steps = 0;
            while (!view.Terminated && !view.Truncated)
            {
                Command command = greedy.Decide(env.State);
                if (!TacticalV2Coding.TryEncode(
                    command, env.State, env.Layout, tracker.Slots0, out int action))
                {
                    action = 0;
                }
                view = env.Step(action);
                rewards.Add(view.Reward);
                foreach (DuelTransition transition in transitions.Drain())
                {
                    accepted.Add(transition);
                    tracker.Slots0.ReleaseDead(transition.Resulting, PlayerId.Player0);
                    tracker.Slots1.ReleaseDead(transition.Resulting, PlayerId.Player1);
                    if (transition.Command is DeployUnit deploy)
                    {
                        TacticalV2UnitRegistry registry =
                            deploy.Issuer == PlayerId.Player0 ? tracker.Slots0 : tracker.Slots1;
                        registry.RegisterDeployment(
                            transition.Previous, transition.Resulting,
                            deploy.Issuer, deploy.TemplateIndex);
                    }
                }
                Assert.That(++steps, Is.LessThan(10_000), "scripted episode did not terminate");
            }

            return new CapturedEpisode(
                env, view, rewards, accepted, demonstrations.Drain());
        }

        private static int FirstLegalNonEndTurn(bool[] mask)
        {
            for (int action = 1; action < mask.Length; action++)
                if (mask[action]) return action;
            Assert.Fail("expected a legal non-EndTurn action");
            return -1;
        }

        private static void AssertTraceCommandsEqual(
            TacticalTraceCommand expected, TacticalTraceCommand actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Kind, Is.EqualTo(expected.Kind));
                Assert.That(actual.Issuer, Is.EqualTo(expected.Issuer));
                Assert.That(actual.ActorId, Is.EqualTo(expected.ActorId));
                Assert.That(actual.TargetId, Is.EqualTo(expected.TargetId));
                Assert.That(actual.Q, Is.EqualTo(expected.Q));
                Assert.That(actual.R, Is.EqualTo(expected.R));
            });
        }

        private static void AssertStatesEquivalent(GameState expected, GameState actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Round, Is.EqualTo(expected.Round));
                Assert.That(actual.ActivePlayer, Is.EqualTo(expected.ActivePlayer));
                Assert.That(actual.IsGameOver, Is.EqualTo(expected.IsGameOver));
                Assert.That(actual.Winner, Is.EqualTo(expected.Winner));
                for (int seat = 0; seat < 2; seat++)
                {
                    PlayerState expectedPlayer = expected.Player((PlayerId)seat);
                    PlayerState actualPlayer = actual.Player((PlayerId)seat);
                    Assert.That(actualPlayer.Points, Is.EqualTo(expectedPlayer.Points));
                    Assert.That(actualPlayer.DestroyedValue, Is.EqualTo(expectedPlayer.DestroyedValue));
                    Assert.That(actualPlayer.UnitsOnBoard, Is.EqualTo(expectedPlayer.UnitsOnBoard));
                    Assert.That(actualPlayer.Generators, Is.EqualTo(expectedPlayer.Generators));
                }
            });
        }

        private sealed class ClaimFirstAgent : IAgent
        {
            public Command Decide(GameState state)
            {
                Command? capture = LegalMoves.For(state).FirstOrDefault(command => command is CaptureHex);
                if (capture != null) return capture;
                Command? moveToUncontrolled = LegalMoves.For(state)
                    .OfType<MoveUnit>()
                    .FirstOrDefault(move => state.Board.Controller(move.Dest) != state.ActivePlayer);
                return moveToUncontrolled ?? new EndTurn(state.ActivePlayer);
            }
        }

        private sealed class CapturedEpisode
        {
            public CapturedEpisode(
                TacticalV2DuelEnv env,
                TacticalV2DuelEnv.View view,
                IReadOnlyList<float> rewards,
                IReadOnlyList<DuelTransition> transitions,
                IReadOnlyList<TacticalV2Demonstration> demonstrations)
            {
                Env = env;
                View = view;
                Rewards = rewards;
                Transitions = transitions;
                Demonstrations = demonstrations;
            }

            public TacticalV2DuelEnv Env { get; }
            public TacticalV2DuelEnv.View View { get; }
            public IReadOnlyList<float> Rewards { get; }
            public IReadOnlyList<DuelTransition> Transitions { get; }
            public IReadOnlyList<TacticalV2Demonstration> Demonstrations { get; }
        }
    }
}
