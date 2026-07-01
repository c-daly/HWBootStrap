using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>
    /// The action-budget surface of ITurnPolicy (ActionsPerTurn / RemainingActions) that the HUD
    /// reads to show "actions left" when a paced turn structure is active.
    /// </summary>
    public class TurnPolicyBudgetTests
    {
        static GameState Build(ITurnPolicy policy) =>
            GameFactory.BuildTerritory(
                GameConfig.Default(biomesEnabled: false, territoryMode: true, turnPolicy: policy), 11, 9, 1);

        [Test]
        public void AllUnitsPolicy_HasNoActionBudget()
        {
            var policy = new AllUnitsPolicy();
            var st = Build(policy);
            Assert.That(policy.ActionsPerTurn, Is.Null);
            Assert.That(policy.RemainingActions(st), Is.Null);
        }

        [Test]
        public void OneActionPolicy_BudgetIsOne()
        {
            var policy = new OneActionPolicy();
            var st = Build(policy);
            Assert.That(policy.ActionsPerTurn, Is.EqualTo(1));
            Assert.That(policy.RemainingActions(st), Is.EqualTo(1));
        }

        [Test]
        public void KActionsPolicy_CountsDownAsActionsAreTaken()
        {
            var policy = new KActionsPolicy(3);
            var st = Build(policy);
            Assert.That(policy.ActionsPerTurn, Is.EqualTo(3));
            Assert.That(policy.RemainingActions(st), Is.EqualTo(3));

            foreach (var u in st.Player(PlayerId.Player0).UnitsOnBoard)
            {
                HexCoord? dest = null;
                foreach (var c in MovementService.ReachableTiles(st, u)) { dest = c; break; }
                if (dest == null) continue;
                var r = GameEngine.Apply(st, new MoveUnit(PlayerId.Player0, u.Id, dest.Value));
                Assert.That(r.Success, Is.True);
                Assert.That(policy.RemainingActions(r.NewState), Is.EqualTo(2),
                    "one move out of a 3-action budget leaves 2");
                return;
            }
            Assert.Fail("no movable unit to test with");
        }
    }
}
