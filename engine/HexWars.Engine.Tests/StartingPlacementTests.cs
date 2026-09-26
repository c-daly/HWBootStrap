using System.Linq;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class StartingPlacementTests
    {
        static GameState NewGame(GameMode mode = GameMode.Annihilation) => GameFactory.Build(
            new GameSetup(mode, 9, 7, 40, 7, turnActions: 1, manualPlacement: true));

        [TestCase(GameMode.Annihilation)]
        [TestCase(GameMode.Territory)]
        public void BothPlayersArrangeAnywhereInTheirZoneBeforeBattleWithoutSpendingResources(GameMode mode)
        {
            var state = NewGame(mode);
            foreach (var owner in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                Assert.That(state.PlacingStartingUnits, Is.True);
                Assert.That(state.ActivePlayer, Is.EqualTo(owner));
                var unit = state.Player(owner).UnitsOnBoard.First();
                // Placement is free even on elevated tiles or beyond this unit's normal move budget.
                var cell = StartingPlacement.Cells(state, unit).OrderByDescending(c => state.Board.TileAt(c).Elevation).First();
                Assert.That(state.Board.TileAt(cell).Elevation, Is.GreaterThan(0));
                var placed = GameEngine.Apply(state, new PlaceStartingUnit(owner, unit.Id, cell));
                Assert.That(placed.Success, Is.True); state = placed.NewState;
                Assert.That(state.Player(owner).UnitsOnBoard.First().Cell, Is.EqualTo(cell));
                Assert.That(state.Player(owner).UnitsOnBoard.First().Elevation, Is.EqualTo(state.Board.TileAt(cell).Elevation));
                Assert.That(state.ActivePlayer, Is.EqualTo(owner), "K=1 must not end the placement turn.");
                Assert.That(state.Player(owner).Points, Is.EqualTo(40));
                Assert.That(state.Round, Is.EqualTo(1));
                Assert.That(state.MovementSpent, Is.Empty); Assert.That(state.AttackedUnitIds, Is.Empty);
                state = GameEngine.Apply(state, new FinishPlacement(owner)).NewState;
            }
            Assert.That(state.PlacingStartingUnits, Is.False);
            Assert.That(state.ActivePlayer, Is.EqualTo(PlayerId.Player0));
            Assert.That(state.Round, Is.EqualTo(1));
            Assert.That(state.Players.Select(p => p.Points), Is.EqualTo(new[] { 40, 40 }));
            Assert.That(GameEngine.Apply(state, new FinishPlacement(PlayerId.Player0)).Reason,
                Is.EqualTo(RejectionReason.PlacementAlreadyFinished));
        }

        [Test]
        public void PlacementRejectsCombatWrongOwnersOccupiedAndOutsideCells()
        {
            var state = NewGame(); var a = state.Players[0].UnitsOnBoard.First(); var b = state.Players[1].UnitsOnBoard.First();
            Assert.That(GameEngine.Apply(state, new AttackUnit(PlayerId.Player0, a.Id, b.Id)).Reason, Is.EqualTo(RejectionReason.PlacementOnly));
            Assert.That(GameEngine.Apply(state, new EndTurn(PlayerId.Player0)).Reason, Is.EqualTo(RejectionReason.PlacementOnly));
            Assert.That(GameEngine.Apply(state, new FinishPlacement(PlayerId.Player1)).Reason, Is.EqualTo(RejectionReason.NotYourTurn));
            var free = StartingPlacement.Cells(state, a).First();
            Assert.That(GameEngine.Apply(state, new PlaceStartingUnit(PlayerId.Player0, b.Id, free)).Reason, Is.EqualTo(RejectionReason.UnitNotFound));
            Assert.That(GameEngine.Apply(state, new PlaceStartingUnit(PlayerId.Player0, a.Id, state.Players[0].UnitsOnBoard.Last().Cell)).Reason, Is.EqualTo(RejectionReason.TileOccupied));
            Assert.That(GameEngine.Apply(state, new PlaceStartingUnit(PlayerId.Player0, a.Id, b.Cell)).Reason, Is.EqualTo(RejectionReason.OutsideDeploymentZone));
            Assert.That(LegalMoves.For(state).All(c => c is PlaceStartingUnit || c is FinishPlacement), Is.True);
            foreach (var command in LegalMoves.For(state)) Assert.That(GameEngine.Apply(state, command).Success, Is.True);
        }

        [Test]
        public void SetupCommandsAndReplayPreservePlacementForNetworkResync()
        {
            var setup = new GameSetup(GameMode.Annihilation, 9, 7, 40, 7, manualPlacement: true);
            Assert.That(GameSetup.Parse(setup.ToWire()).Sanitized().ManualPlacement, Is.True);
            Assert.That(GameSetup.Parse("0 9 7 40 7 3 1 1 1 0 0").ManualPlacement, Is.False);
            var start = GameFactory.Build(setup); var a = start.Players[0].UnitsOnBoard.First();
            Command[] commands = { new PlaceStartingUnit(PlayerId.Player0, a.Id, StartingPlacement.Cells(start, a).First()), new FinishPlacement(PlayerId.Player0), new FinishPlacement(PlayerId.Player1) };
            foreach (var command in commands) Assert.That(CommandWire.Read(CommandWire.Write(command)), Is.EqualTo(command));
            var replay = ReplayFile.Read(ReplayFile.Write(start, commands));
            Assert.That(replay.Start.PlacingStartingUnits, Is.True); Assert.That(replay.Start.Clone().PlacingStartingUnits, Is.True);
            var state = replay.Start;
            foreach (var command in replay.Commands) { var result = GameEngine.Apply(state, command); Assert.That(result.Success, Is.True); state = result.NewState; }
            Assert.That(state.PlacingStartingUnits, Is.False);
            Assert.That(state.Players[0].UnitsOnBoard.First().Cell, Is.EqualTo(((PlaceStartingUnit)commands[0]).Cell));
        }

        [Test]
        public void AutomaticSetupRemainsImmediateAndExplicitOverrideSkipsPlacement()
        {
            Assert.That(GameFactory.Build(GameSetup.Default).PlacingStartingUnits, Is.False);
            var setup = new GameSetup(GameMode.Annihilation, 9, 7, 0, 7, manualPlacement: true);
            Assert.That(GameFactory.Build(setup, null, null, beginInDeployment: false).PlacingStartingUnits, Is.False);
        }
    }
}
