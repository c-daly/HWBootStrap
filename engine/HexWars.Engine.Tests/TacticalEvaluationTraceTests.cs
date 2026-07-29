using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalEvaluationTraceTests
    {
        [Test]
        public void Project_AttackPreservesCommandAndMaterialChange()
        {
            DuelTransition transition = FirstTransition(item => item.Command is AttackUnit);

            TacticalTraceTransition trace = TacticalEvaluationTrace.Project(transition);

            Assert.That(trace.Command.Kind, Is.EqualTo("attack"));
            Assert.That(trace.Command.ActorId, Is.Not.Null);
            Assert.That(trace.Command.TargetId, Is.Not.Null);
            int foe = 1 - trace.Command.Issuer;
            Assert.That(trace.After.Seats[foe].HealthAdjustedMaterial,
                Is.LessThan(trace.Before.Seats[foe].HealthAdjustedMaterial));
        }

        [Test]
        public void Project_MovePreservesActorAndDestination()
        {
            DuelTransition transition = FirstTransition(item => item.Command is MoveUnit);

            TacticalTraceTransition trace = TacticalEvaluationTrace.Project(transition);
            var command = (MoveUnit)transition.Command;

            Assert.That(trace.Command.Kind, Is.EqualTo("move"));
            Assert.That(trace.Command.ActorId, Is.EqualTo(command.UnitId));
            Assert.That(trace.Command.Q, Is.EqualTo(command.Dest.Q));
            Assert.That(trace.Command.R, Is.EqualTo(command.Dest.R));
        }

        [Test]
        public void Project_EndTurnRecordsProductiveAlternatives()
        {
            DuelTransition transition = FirstTransition(item =>
                item.Command is EndTurn
                && LegalMoves.For(item.Previous).Any(command => !(command is EndTurn)));

            TacticalTraceTransition trace = TacticalEvaluationTrace.Project(transition);

            Assert.That(trace.Command.Kind, Is.EqualTo("end_turn"));
            Assert.That(trace.Before.ProductiveLegalActions, Is.GreaterThan(0));
        }

        [Test]
        public void Project_SortsUnitsAndCalculatesHealthAdjustedMaterial()
        {
            GameState state = StateWithUnsortedDamagedUnits();

            TacticalTraceTransition trace = TacticalEvaluationTrace.Project(
                new DuelTransition(state, new EndTurn(PlayerId.Player0), state));

            TacticalTraceSeat seat = trace.Before.Seats[0];
            Assert.That(seat.Units.Select(unit => unit.Id), Is.EqualTo(new[] { 2, 7 }));
            Assert.That(seat.CurrentHitPoints, Is.EqualTo(9));
            Assert.That(seat.MaximumHitPoints, Is.EqualTo(14));
            Assert.That(seat.HealthAdjustedMaterial, Is.EqualTo(16.5d));
        }

        [Test]
        public void Project_PreservesMovementExpenditureAndBoardControl()
        {
            GameState source = StateWithUnsortedDamagedUnits();
            GameState state = new GameState(
                source.Board.WithControl(new HexCoord(1, 0), PlayerId.Player1),
                source.Config,
                source.Players,
                source.ActivePlayer,
                source.Round,
                source.NextEntityId,
                movedUnitIds: new[] { 7 },
                movementSpent: new Dictionary<int, (int H, int V)> { [7] = (2, 1) });

            TacticalTraceState trace = TacticalEvaluationTrace.Project(
                new DuelTransition(state, new EndTurn(PlayerId.Player0), state)).Before;

            TacticalTraceUnit unit = trace.Seats[0].Units.Single(item => item.Id == 7);
            Assert.That(unit.MovementSpentH, Is.EqualTo(2));
            Assert.That(unit.MovementSpentV, Is.EqualTo(1));
            Assert.That(trace.ControlledHexes.Select(item => (item.Q, item.R, item.Controller)),
                Is.EqualTo(new[] { (1, 0, 1) }));
        }

        [Test]
        public void Project_MapsSupportedCommandsAndLeavesUnknownFieldsEmpty()
        {
            GameState state = StateWithUnsortedDamagedUnits();
            var cases = new (Command Command, string Kind, int? ActorId, int? TargetId, int? Q, int? R)[]
            {
                (new EndTurn(PlayerId.Player0), "end_turn", null, null, null, null),
                (new MoveUnit(PlayerId.Player0, 7, new HexCoord(3, -2)), "move", 7, null, 3, -2),
                (new AttackUnit(PlayerId.Player0, 7, 99), "attack", 7, 99, null, null),
                (new DeployUnit(PlayerId.Player0, 1, new HexCoord(3, -2)), "deploy", null, null, 3, -2),
                (new CaptureHex(PlayerId.Player0, new HexCoord(3, -2)), "capture", null, null, 3, -2),
                (new BuildGenerator(PlayerId.Player0, new HexCoord(3, -2)), "build_generator", null, null, 3, -2),
                (new CreateUnit(PlayerId.Player0, UnitStats(health: 1)), "CreateUnit", null, null, null, null),
            };

            foreach (var item in cases)
            {
                TacticalTraceCommand command = TacticalEvaluationTrace.Project(
                    new DuelTransition(state, item.Command, state)).Command;

                Assert.That(command.Kind, Is.EqualTo(item.Kind));
                Assert.That(command.Issuer, Is.EqualTo((int)PlayerId.Player0));
                Assert.That(command.ActorId, Is.EqualTo(item.ActorId));
                Assert.That(command.TargetId, Is.EqualTo(item.TargetId));
                Assert.That(command.Q, Is.EqualTo(item.Q));
                Assert.That(command.R, Is.EqualTo(item.R));
            }
        }

        [Test]
        public void Project_UsesTargetingServiceForCurrentAttackPotential()
        {
            HexCoord attackerCell = new HexCoord(0, 0);
            HexCoord targetCell = new HexCoord(2, 0);
            var attacker = new Unit(1, PlayerId.Player0,
                UnitStats(health: 4, damage: 2, range: 3, vision: 3), attackerCell, 0);
            var target = new Unit(2, PlayerId.Player1, UnitStats(health: 4), targetCell, 0);
            var board = new Board(new[]
            {
                new Tile(attackerCell, 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 2, TerrainType.Plains),
                new Tile(targetCell, 0, TerrainType.Plains),
            });
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { attacker }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { target }),
            }, PlayerId.Player0, 1, 3);

            Assert.That(TargetingService.CanTarget(state, attacker, target.Cell, target.Elevation), Is.False);

            TacticalTraceSeat seat = TacticalEvaluationTrace.Project(
                new DuelTransition(state, new EndTurn(PlayerId.Player0), state)).Before.Seats[0];

            Assert.That(seat.CanDamageEnemy, Is.True);
            Assert.That(seat.CanCurrentlyAttackEnemy, Is.False);
        }

        [Test]
        public void TraceDtosSerializeWithPublicDataProperties()
        {
            GameState state = StateWithUnsortedDamagedUnits();

            string json = JsonSerializer.Serialize(TacticalEvaluationTrace.Project(
                new DuelTransition(state, new EndTurn(PlayerId.Player0), state)));

            Assert.That(json, Does.Contain("\"Before\""));
            Assert.That(json, Does.Contain("\"HealthAdjustedMaterial\""));
            Assert.That(json, Does.Contain("\"end_turn\""));
        }

        private static DuelTransition FirstTransition(Predicate<DuelTransition> matches)
        {
            for (int seed = 0; seed < 20; seed++)
            {
                var env = new TacticalV2DuelEnv(TacticalV2Config.Default());
                env.CaptureTransitions = true;
                env.Reset(seed, new GreedyAgent(seed), new GreedyAgent(seed + 1));
                foreach (DuelTransition transition in env.DrainTransitions())
                    if (matches(transition)) return transition;
            }
            throw new AssertionException("expected matching transition across seeds 0..19");
        }

        private static GameState StateWithUnsortedDamagedUnits()
        {
            var first = new Unit(7, PlayerId.Player0,
                UnitStats(health: 10, damage: 2, movement: 1, range: 2, vision: 2), new HexCoord(0, 0), 0)
                .WithDamage(5);
            var second = new Unit(2, PlayerId.Player0,
                UnitStats(health: 4, damage: 1, verticalMovement: 1, range: 1, vision: 1), new HexCoord(1, 0), 0);
            var enemy = new Unit(9, PlayerId.Player1, UnitStats(health: 5), new HexCoord(2, 0), 0);
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            });
            return new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 6, unitsOnBoard: new[] { first, second }, destroyedValue: 4),
                new PlayerState(PlayerId.Player1, 3, unitsOnBoard: new[] { enemy }, destroyedValue: 2),
            }, PlayerId.Player0, 4, 10);
        }

        private static UnitStats UnitStats(int health, int damage = 0, int defense = 0, int movement = 0,
            int verticalMovement = 0, int range = 0, int rangeArc = 0, int vision = 0, int visionArc = 0) =>
            new UnitStats(health, damage, defense, movement, verticalMovement, range, rangeArc, vision, visionArc);
    }
}
