using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class UndoMoveTests
    {
        const PlayerId P0 = PlayerId.Player0, P1 = PlayerId.Player1;
        static GameState Start(ITurnPolicy? policy = null, bool fog = false)
        {
            var board = new Board(Enumerable.Range(0, 7).Select(q =>
                new Tile(new HexCoord(q, 0), q == 1 ? 1 : 0, TerrainType.Plains)).ToArray());
            var stats = new UnitStats(10, 2, 0, 4, 2, 6, 2, 7, 2);
            return new GameState(board, GameConfig.Default(biomesEnabled: false, turnPolicy: policy, fogOfWar: fog),
                new[] { new PlayerState(P0, 12, unitsOnBoard: new[] { new Unit(1, P0, stats, new HexCoord(0, 0), 0) }),
                    new PlayerState(P1, 12, unitsOnBoard: new[] { new Unit(2, P1, stats, new HexCoord(6, 0), 0) }) }, P0, 1, 3);
        }
        static GameState Apply(GameState state, Command command)
        {
            var result = GameEngine.Apply(state, command);
            Assert.That(result.Success, Is.True, result.Reason.ToString());
            return result.NewState;
        }

        [Test]
        public void UndoRestoresPositionElevationBothBudgetsAndActionAllowance()
        {
            var start = Start(new KActionsPolicy(3));
            var moved = Apply(start, new MoveUnit(P0, 1, new HexCoord(1, 0)));
            Assert.That(moved.MovementSpent[1], Is.EqualTo((1, 1)));
            Assert.That(moved.Config.TurnPolicy.RemainingActions(moved), Is.EqualTo(2));
            var restored = Apply(moved, new UndoMove(P0));
            Assert.That(restored.Player(P0).UnitsOnBoard[0].Cell, Is.EqualTo(new HexCoord(0, 0)));
            Assert.That(restored.Player(P0).UnitsOnBoard[0].Elevation, Is.Zero);
            Assert.That(restored.MovementSpent, Is.Empty);
            Assert.That(restored.MovedUnitIds, Is.Empty);
            Assert.That(restored.Config.TurnPolicy.RemainingActions(restored), Is.EqualTo(3));
            Assert.That(restored.Player(P0).Points, Is.EqualTo(12));
            Assert.That(restored.Player(P1).UnitsOnBoard[0].CurrentHp, Is.EqualTo(10));
            Assert.That(restored.BeforeLastMove, Is.Null);
            Assert.That(moved.Player(P0).UnitsOnBoard[0].Cell, Is.EqualTo(new HexCoord(1, 0)), "Undo must not mutate a retained state.");
        }

        [Test]
        public void UndoOnlyRestoresTheLastHopAndCannotBeRepeated()
        {
            var first = Apply(Start(), new MoveUnit(P0, 1, new HexCoord(1, 0)));
            var second = Apply(first, new MoveUnit(P0, 1, new HexCoord(2, 0)));
            var restored = Apply(second, new UndoMove(P0));
            Assert.That(restored.Player(P0).UnitsOnBoard[0].Cell, Is.EqualTo(new HexCoord(1, 0)));
            Assert.That(restored.MovementSpent[1], Is.EqualTo((1, 1)));
            Assert.That(GameEngine.Apply(restored, new UndoMove(P0)).Reason, Is.EqualTo(RejectionReason.NoMoveToUndo));
        }

        [TestCase("attack")]
        [TestCase("design")]
        [TestCase("turn")]
        public void AnotherAcceptedActionClosesUndo(string action)
        {
            var moved = Apply(Start(), new MoveUnit(P0, 1, new HexCoord(1, 0)));
            Command next = action == "attack" ? new AttackUnit(P0, 1, 2) :
                action == "design" ? new CreateUnit(P0, new UnitStats(1, 1, 0, 1, 0, 1, 0, 1, 0)) : new EndTurn(P0);
            var after = Apply(moved, next);
            Assert.That(after.BeforeLastMove, Is.Null);
            Assert.That(GameEngine.Apply(after, new UndoMove(after.ActivePlayer)).Success, Is.False);
            if (action == "attack") Assert.That(after.Player(P1).UnitsOnBoard[0].CurrentHp, Is.LessThan(10));
        }

        [Test]
        public void RejectedActionDoesNotDiscardUndoAndTheOtherSeatCannotUseIt()
        {
            var moved = Apply(Start(), new MoveUnit(P0, 1, new HexCoord(1, 0)));
            var rejected = GameEngine.Apply(moved, new MoveUnit(P0, 1, new HexCoord(99, 0)));
            Assert.That(rejected.Success, Is.False);
            Assert.That(rejected.NewState, Is.SameAs(moved));
            Assert.That(GameEngine.Apply(moved, new UndoMove(P1)).Reason, Is.EqualTo(RejectionReason.NotYourTurn));
            Assert.That(Apply(moved, new UndoMove(P0)).BeforeLastMove, Is.Null);
        }

        [Test]
        public void FogAndAutomaticTurnHandoffsCannotBeReversed()
        {
            foreach (var start in new[] { Start(fog: true), Start(new OneActionPolicy()), Start(new KActionsPolicy(1)) })
            {
                var moved = Apply(start, new MoveUnit(P0, 1, new HexCoord(1, 0)));
                Assert.That(moved.BeforeLastMove, Is.Null);
                Assert.That(GameEngine.Apply(moved, new UndoMove(moved.ActivePlayer)).Success, Is.False);
            }
        }

        [Test]
        public void WireAndReconnectReplayPreserveUndoAndRejectForgedSeats()
        {
            Assert.That(CommandWire.Read(CommandWire.Write(new UndoMove(P0))), Is.EqualTo(new UndoMove(P0)));
            Assert.That(CommandWire.TryRead("UNDO 0 extra", out _), Is.False);
            var hub = new MatchHub(_ => Start());
            hub.Connect("undo", "a", token: "alice"); hub.Connect("undo", "b", token: "bob");
            string catalog = NetProtocol.Catalog(BarracksWire.Write(BarracksCatalog.DefaultTemplates));
            hub.Receive("undo", "a", catalog); hub.Receive("undo", "b", catalog);
            var move = new MoveUnit(P0, 1, new HexCoord(1, 0));
            var applied = hub.Receive("undo", "a", NetProtocol.Cmd(move));
            Assert.That(applied.Select(x => x.Message), Is.EqualTo(new[] { NetProtocol.Apply(move), NetProtocol.Apply(move) }));
            Assert.That(hub.Receive("undo", "b", "CMD UNDO 0").Single().Message, Does.StartWith("REJECT"));
            hub.Disconnect("undo", "a");
            var reconnect = hub.Connect("undo", "a2", token: "alice");
            var data = ReplayFile.Read(reconnect.Single(x => x.Message.StartsWith("START ")).Message.Substring(6));
            var client = new Replay(data.Start, data.Commands).Final;
            Assert.That(client.BeforeLastMove, Is.Not.Null);
            var undone = hub.Receive("undo", "a2", "CMD UNDO 0");
            Assert.That(undone.Select(x => x.Message), Is.EqualTo(new[] { "APPLY UNDO 0", "APPLY UNDO 0" }));
            client = Apply(client, new UndoMove(P0));
            var saved = hub.Connect("undo", "a3", token: "alice");
            data = ReplayFile.Read(saved.Single(x => x.Message.StartsWith("START ")).Message.Substring(6));
            Assert.That(data.Commands.Count, Is.EqualTo(2));
            var recovered = new Replay(data.Start, data.Commands).Final;
            Assert.That(recovered.Player(P0).UnitsOnBoard[0].Cell, Is.EqualTo(client.Player(P0).UnitsOnBoard[0].Cell));
            Assert.That(recovered.MovementSpent, Is.Empty);
            Assert.That(recovered.BeforeLastMove, Is.Null);
        }
    }
}
