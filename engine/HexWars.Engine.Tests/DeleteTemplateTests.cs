using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>DeleteTemplate: an administrative barracks edit, not a game move — free, doesn't touch
    /// the turn budget (not even under OneActionPolicy, which ends the turn after ANY other single
    /// action — see TurnPolicyTests.OneActionPolicy_AutoEndsTurn_AfterASingleAction), and is never
    /// offered by LegalMoves (keeps RL action masks untouched, per spec §5).</summary>
    public class DeleteTemplateTests
    {
        private static UnitTemplate T(string name, int cost) => new UnitTemplate(name, TestStates.Cost(cost));

        private static GameState TwoTemplates(int points = 12, GameConfig? cfg = null) =>
            new GameState(
                new Board(new[]
                {
                    new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                    new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) }),
                cfg ?? GameConfig.Default(),
                new[]
                {
                    new PlayerState(PlayerId.Player0, points, new[] { T("Alpha", 2), T("Beta", 3) }),
                    new PlayerState(PlayerId.Player1, points),
                },
                PlayerId.Player0, 1, 1);

        [Test]
        public void Delete_RemovesTemplateAtIndex_NonMutating()
        {
            var state = TwoTemplates();
            var r = GameEngine.Apply(state, new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            var barracks = r.NewState.Player(PlayerId.Player0).Barracks;
            Assert.That(barracks.Count, Is.EqualTo(1));
            Assert.That(barracks[0].Name, Is.EqualTo("Beta"));
            Assert.That(state.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(2), "original untouched");
        }

        [Test]
        public void Delete_ShiftsLaterIndices()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, 0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Barracks[0].Name, Is.EqualTo("Beta"));
        }

        [Test]
        public void Delete_Rejects_IndexTooLarge()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, 2));
            Assert.That(r.Success, Is.False);
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TemplateNotFound));
        }

        [Test]
        public void Delete_Rejects_NegativeIndex()
        {
            var r = GameEngine.Apply(TwoTemplates(), new DeleteTemplate(PlayerId.Player0, -1));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TemplateNotFound));
        }

        [Test]
        public void Delete_IsFree_DoesNotSpendPoints()
        {
            var r = GameEngine.Apply(TwoTemplates(points: 7), new DeleteTemplate(PlayerId.Player0, 0));
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(7));
        }

        [Test]
        public void Delete_DoesNotEndTurn_UnderOneActionPolicy()
        {
            var r = GameEngine.Apply(TwoTemplates(cfg: GameConfig.Default(turnPolicy: new OneActionPolicy())),
                new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0),
                "a CreateUnit alone ends a OneActionPolicy turn (see TurnPolicyTests) — DeleteTemplate must not");
        }

        [Test]
        public void Delete_DoesNotCountTowardKActionsPolicy_AfterBudgetSpent()
        {
            // Player0 has one board unit under K=1; simulate a state where the K=1 budget is already
            // spent (as if a move just happened, without an EndTurn) and confirm DeleteTemplate right
            // there does not itself force a further end-turn.
            var cfg = GameConfig.Default(turnPolicy: new KActionsPolicy(1));
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0) }, zone1: new[] { new HexCoord(1, 0) });
            var unit = new Unit(1, PlayerId.Player0, TestStates.Stats(health: 3, movement: 1), new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 10, new[] { T("Alpha", 2) }, new[] { unit });
            var p1 = new PlayerState(PlayerId.Player1, 10);
            var spent = new GameState(board, cfg, new[] { p0, p1 }, PlayerId.Player0, 1, 2,
                movedUnitIds: new[] { 1 });

            var r = GameEngine.Apply(spent, new DeleteTemplate(PlayerId.Player0, 0));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.ActivePlayer, Is.EqualTo(PlayerId.Player0),
                "DeleteTemplate must not trigger KActionsPolicy's already-at-budget auto-end");
        }

        [Test]
        public void Delete_NeverEnumeratedByLegalMoves()
        {
            var moves = LegalMoves.For(TwoTemplates());
            Assert.That(moves.OfType<DeleteTemplate>(), Is.Empty);
        }
    }
}
