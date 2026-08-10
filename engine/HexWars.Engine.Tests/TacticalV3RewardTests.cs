using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3RewardTests
    {
        [TestCase(true, 0, 1.0f)]
        [TestCase(true, 1, -1.0f)]
        [TestCase(false, -1, -1.0f)]
        public void TerminalBase_UsesWinVersusNonWinOrdering(
            bool hasWinner, int winner, float expected)
        {
            GameState initial = TacticalV3Fixtures.RewardStart();
            GameState final = TacticalV3Fixtures.Terminal(hasWinner ? (PlayerId?)winner : null);
            TacticalV3Reward reward = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0);

            TacticalV3RewardBreakdown value = reward.Evaluate(final, terminated: true, truncated: false);

            Assert.That(value.TerminalOutcome, Is.EqualTo(expected));
            Assert.That(value.Total, expected > 0 ? Is.GreaterThanOrEqualTo(0.75f) : Is.LessThanOrEqualTo(-0.75f));
            Assert.That(value.Finalized, Is.True);
        }

        [Test]
        public void RoundCapDraw_UsesSpecifiedRoundBasedTimePressure()
        {
            GameState initial = TacticalV3Fixtures.RewardStart();
            GameState final = TacticalV3Fixtures.AtRound(initial, initial.Config.RoundCap);
            TacticalV3RewardBreakdown value = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0)
                .Evaluate(final, terminated: true, truncated: false);

            Assert.That(value.TerminalOutcome, Is.EqualTo(-1f));
            Assert.That(value.TimePressure, Is.EqualTo(-0.0495f));
            Assert.That(value.Total, Is.EqualTo(-1.0495f));
        }

        [Test]
        public void StepTruncation_FinalizesAsNonWin()
        {
            GameState initial = TacticalV3Fixtures.RewardStart();
            TacticalV3RewardBreakdown value = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0)
                .Evaluate(initial, terminated: false, truncated: true);

            Assert.That(value.Finalized, Is.True);
            Assert.That(value.TerminalOutcome, Is.EqualTo(-1f));
            Assert.That(value.Total, Is.EqualTo(-1f));
        }

        [Test]
        public void TerminalPartialDamage_UsesHealthAdjustedMaterialProgressBeforeAKill()
        {
            GameState initial = TacticalV3Fixtures.RewardStart(unitCost: 10);
            GameState final = TacticalV3Fixtures.WithDamage(initial, PlayerId.Player1, damage: 5);
            TacticalV3RewardBreakdown value = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0)
                .Evaluate(final, terminated: false, truncated: true);

            Assert.That(value.KnownHealthAdjustedMaterialProgress, Is.EqualTo(0.20f));
            Assert.That(value.PublicResourceProgress, Is.Zero);
            Assert.That(value.Total, Is.EqualTo(-0.80f));
        }

        [Test]
        public void LargerArmy_NormalizesEquivalentDamageAgainstInitialTotalValue()
        {
            GameState small = TacticalV3Fixtures.RewardStart(unitCost: 10);
            GameState large = TacticalV3Fixtures.RewardStart(unitCost: 20);
            TacticalV3RewardBreakdown smallValue = TacticalV3Fixtures.Tracker(small, PlayerId.Player0)
                .Evaluate(TacticalV3Fixtures.WithDamage(small, PlayerId.Player1, damage: 2), false, true);
            TacticalV3RewardBreakdown largeValue = TacticalV3Fixtures.Tracker(large, PlayerId.Player0)
                .Evaluate(TacticalV3Fixtures.WithDamage(large, PlayerId.Player1, damage: 2), false, true);

            Assert.That(smallValue.KnownHealthAdjustedMaterialProgress, Is.EqualTo(0.10f));
            Assert.That(largeValue.KnownHealthAdjustedMaterialProgress, Is.EqualTo(0.05f));
        }

        [Test]
        public void IntermediateUnchangedState_IsNotFinalizedAndCannotFarmReward()
        {
            GameState initial = TacticalV3Fixtures.RewardStart();
            TacticalV3Reward reward = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0);

            TacticalV3RewardBreakdown first = reward.Evaluate(initial, terminated: false, truncated: false);
            TacticalV3RewardBreakdown second = reward.Evaluate(initial, terminated: false, truncated: false);

            Assert.That(first.Finalized, Is.False);
            Assert.That(first.Total, Is.Zero);
            Assert.That(second.Total, Is.Zero);
        }

        [Test]
        public void TerminalReward_ClampsEveryComponentAndTheAbsoluteTotalBounds()
        {
            GameState initial = TacticalV3Fixtures.RewardStart(unitCost: 10);
            GameState final = TacticalV3Fixtures.AtRound(
                TacticalV3Fixtures.WithDamage(initial, PlayerId.Player1, damage: 10), initial.Config.RoundCap);
            TacticalV3RewardBreakdown value = TacticalV3Fixtures.Tracker(initial, PlayerId.Player0)
                .Evaluate(final, terminated: true, truncated: false);

            Assert.That(value.KnownHealthAdjustedMaterialProgress, Is.EqualTo(0.20f));
            Assert.That(value.TimePressure, Is.EqualTo(-0.0495f));
            Assert.That(value.Total, Is.InRange(-1.25f, 1.20f));
        }
    }
}
