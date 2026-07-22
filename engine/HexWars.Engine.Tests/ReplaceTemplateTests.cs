using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class ReplaceTemplateTests
    {
        private static readonly UnitStats ValidStats = new UnitStats(5, 4, 1, 3, 2, 2, 1, 4, 1);

        private static GameConfig Config(int designFee = 3, int maxDesignPointCost = 24,
            int fixedTemplateCount = 6) =>
            new GameConfig(new Dictionary<TerrainType, TerrainDef>
            {
                { TerrainType.Plains, new TerrainDef(1, 0, 0, true) },
            }, designFee: designFee, maxDesignPointCost: maxDesignPointCost,
               fixedTemplateCount: fixedTemplateCount, templateSlotCount: 9);

        private static GameState State(GameConfig? config = null, int points = 20, int templateCount = 9)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, new[] { new HexCoord(0, 0) }, new[] { new HexCoord(1, 0) });
            var templates = Enumerable.Range(0, templateCount)
                .Select(i => new UnitTemplate($"Template {i}", new UnitStats(2 + i, 1, 0, 1, 0, 0, 0, 1, 0)))
                .ToArray();
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, points, templates),
                new PlayerState(PlayerId.Player1, points, templates),
            };
            return new GameState(board, config ?? Config(), players, PlayerId.Player0, 1, 1);
        }

        [Test]
        public void ReplaceTemplate_IsAtomicChargesFeeKeepsIndexAndSanitizesName()
        {
            var state = State();

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, 6, ValidStats, "  <Counter> Unit!!  "));

            Assert.That(result.Success, Is.True);
            Assert.That(result.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(17));
            Assert.That(result.NewState.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(9));
            Assert.That(result.NewState.Player(PlayerId.Player0).Barracks[6].Name, Is.EqualTo("Counter Unit"));
            Assert.That(result.NewState.Player(PlayerId.Player0).Barracks[6].Stats, Is.EqualTo(ValidStats));
            Assert.That(result.NewState.Player(PlayerId.Player0).Barracks[5],
                Is.EqualTo(state.Player(PlayerId.Player0).Barracks[5]));
            Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(20), "input state remains unchanged");
            Assert.That(state.Player(PlayerId.Player0).Barracks[6].Name, Is.EqualTo("Template 6"));
        }

        [TestCase(-1)]
        [TestCase(0)]
        [TestCase(5)]
        [TestCase(9)]
        public void ReplaceTemplate_RejectsNonCustomSlotWithoutMutation(int index)
        {
            var state = State();

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, index, ValidStats, "Counter"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.NewState, Is.SameAs(state));
            Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(20));
            Assert.That(state.Player(PlayerId.Player0).Barracks.Select(t => t.Name),
                Is.EqualTo(Enumerable.Range(0, 9).Select(i => $"Template {i}")));
        }

        [Test]
        public void ReplaceTemplate_RejectsWhenDesignFeeIsUnaffordableWithoutCharging()
        {
            var state = State(points: 2);

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, 6, ValidStats, "Counter"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InsufficientPoints));
            Assert.That(result.NewState, Is.SameAs(state));
            Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(2));
        }

        [TestCaseSource(nameof(InvalidStats))]
        public void ReplaceTemplate_RejectsAnyInvalidStatWithoutMutation(UnitStats stats)
        {
            var state = State();

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, 6, stats, "Counter"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InvalidStats));
            Assert.That(result.NewState, Is.SameAs(state));
            Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(20));
        }

        [Test]
        public void ReplaceTemplate_RejectsDesignOverConfiguredMaximum()
        {
            var state = State(Config(maxDesignPointCost: 22));

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, 6, ValidStats, "Counter"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.InvalidStats));
            Assert.That(result.NewState, Is.SameAs(state));
        }

        [Test]
        public void CreateUnit_UsesSharedFullStatAndMaximumValidation()
        {
            var negative = GameEngine.Apply(State(),
                new CreateUnit(PlayerId.Player0, new UnitStats(1, -1, 0, 0, 0, 0, 0, 0, 0)));
            var overMaximum = GameEngine.Apply(State(Config(maxDesignPointCost: 2)),
                new CreateUnit(PlayerId.Player0, new UnitStats(3, 0, 0, 0, 0, 0, 0, 0, 0)));

            Assert.That(negative.Reason, Is.EqualTo(RejectionReason.InvalidStats));
            Assert.That(overMaximum.Reason, Is.EqualTo(RejectionReason.InvalidStats));
        }

        [Test]
        public void SharedValidation_RejectsPointCostOverflowEvenWhenMaximumIsUnlimited()
        {
            var huge = new UnitStats(int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue,
                int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue);
            var config = Config(maxDesignPointCost: 0);

            var create = GameEngine.Apply(State(config),
                new CreateUnit(PlayerId.Player0, huge, "Overflow"));
            var replace = GameEngine.Apply(State(config),
                new ReplaceTemplate(PlayerId.Player0, 6, huge, "Overflow"));

            Assert.That(create.Reason, Is.EqualTo(RejectionReason.InvalidStats));
            Assert.That(replace.Reason, Is.EqualTo(RejectionReason.InvalidStats));
        }

        [Test]
        public void AdaptiveReplacement_RejectsSlotOutsideConfiguredRoster()
        {
            var state = State(templateCount: 10);

            var result = GameEngine.Apply(state,
                new ReplaceTemplate(PlayerId.Player0, 9, ValidStats, "Escaped"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.NewState, Is.SameAs(state));
        }

        [Test]
        public void AdaptiveRoster_CreateCannotAppendPastConfiguredCapacity()
        {
            var state = State();

            var result = GameEngine.Apply(state, new CreateUnit(PlayerId.Player0,
                new UnitStats(1, 2, 0, 1, 0, 1, 0, 1, 0), "Extra"));

            Assert.That(result.Success, Is.False);
            Assert.That(result.Reason, Is.EqualTo(RejectionReason.BarracksFull));
            Assert.That(result.NewState, Is.SameAs(state));
        }

        [TestCase(0)]
        [TestCase(6)]
        [TestCase(8)]
        public void AdaptiveRoster_DeleteCannotShiftFixedOrCustomSlots(int index)
        {
            var state = State();

            var result = GameEngine.Apply(state, new DeleteTemplate(PlayerId.Player0, index));

            Assert.That(result.Success, Is.False);
            Assert.That(result.NewState, Is.SameAs(state));
            Assert.That(state.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(9));
        }

        [Test]
        public void GameConfig_DefaultMaximumIsUnlimited_AndAdaptiveDefaultIsTwentyFour()
        {
            Assert.That(GameConfig.Default().MaxDesignPointCost, Is.EqualTo(0));
            Assert.That(GameConfig.Default().FixedTemplateCount, Is.EqualTo(0));
            Assert.That(GameConfig.Default().TemplateSlotCount, Is.EqualTo(0));
            Assert.That(AdaptiveEnvConfig.Default().Game.MaxDesignPointCost, Is.EqualTo(24));
            Assert.That(AdaptiveEnvConfig.Default().Game.FixedTemplateCount, Is.EqualTo(6));
            Assert.That(AdaptiveEnvConfig.Default().Game.TemplateSlotCount, Is.EqualTo(9));
        }

        [Test]
        public void AdaptiveContract_RejectsSemanticDesignLimitsThatDivergeFromRuntimeRules()
        {
            var config = AdaptiveEnvConfig.Default();
            config.MaxDesignPointCost = 12;

            Assert.That(() => MlContract.CreateAdaptive(config, MlEnvironmentKind.AdaptiveTactical),
                Throws.ArgumentException.With.Message.Contains("maximum design point cost"));
        }

        [TestCase("")]
        [TestCase("Doom Turtle")]
        public void CommandWire_ReplaceTemplateRoundTripsNamedAndUnnamed(string name)
        {
            var command = new ReplaceTemplate(PlayerId.Player1, 8, ValidStats, name);

            var wire = CommandWire.Write(command);

            Assert.That(wire, Does.StartWith("REPLACE 1 8 "));
            Assert.That(CommandWire.Read(wire), Is.EqualTo(command));
        }

        [Test]
        public void Replay_RoundTripsReplacementCommandAndEffectiveDesignRules()
        {
            var start = State();
            var command = new ReplaceTemplate(PlayerId.Player0, 7, ValidStats, "Doom Turtle");

            var data = ReplayFile.Read(ReplayFile.Write(start, new Command[] { command }));
            var originalResult = GameEngine.Apply(start, command);
            var replayedResult = GameEngine.Apply(data.Start, data.Commands.Single());

            Assert.That(data.Start.Config.DesignFee, Is.EqualTo(3));
            Assert.That(data.Start.Config.MaxDesignPointCost, Is.EqualTo(24));
            Assert.That(data.Start.Config.FixedTemplateCount, Is.EqualTo(6));
            Assert.That(data.Start.Config.TemplateSlotCount, Is.EqualTo(9));
            Assert.That(data.Commands.Single(), Is.EqualTo(command));
            Assert.That(replayedResult.Success, Is.True);
            Assert.That(replayedResult.NewState.Player(PlayerId.Player0).Points,
                Is.EqualTo(originalResult.NewState.Player(PlayerId.Player0).Points));
            Assert.That(replayedResult.NewState.Player(PlayerId.Player0).Barracks[7],
                Is.EqualTo(originalResult.NewState.Player(PlayerId.Player0).Barracks[7]));
        }

        private static IEnumerable<UnitStats> InvalidStats()
        {
            yield return new UnitStats(0, 1, 1, 1, 1, 1, 1, 1, 1);
            yield return new UnitStats(1, -1, 1, 1, 1, 1, 1, 1, 1);
            yield return new UnitStats(1, 1, -1, 1, 1, 1, 1, 1, 1);
            yield return new UnitStats(1, 1, 1, -1, 1, 1, 1, 1, 1);
            yield return new UnitStats(1, 1, 1, 1, -1, 1, 1, 1, 1);
            yield return new UnitStats(1, 1, 1, 1, 1, -1, 1, 1, 1);
            yield return new UnitStats(1, 1, 1, 1, 1, 1, -1, 1, 1);
            yield return new UnitStats(1, 1, 1, 1, 1, 1, 1, -1, 1);
            yield return new UnitStats(1, 1, 1, 1, 1, 1, 1, 1, -1);
            yield return new UnitStats(int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue,
                int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue, int.MaxValue);
        }
    }
}
