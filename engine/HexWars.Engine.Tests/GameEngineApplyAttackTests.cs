using System;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class GameEngineApplyAttackTests
    {
        // Player0 attacker (id 1) on a 3-plains line; Player1 holds the given units/generators.
        private static GameState Scene(UnitStats attacker, Unit[]? enemyUnits = null,
                                       Generator[]? enemyGens = null, int round = 1)
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            });
            var a = new Unit(1, PlayerId.Player0, attacker, new HexCoord(0, 0), 0);
            var p0 = new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { a });
            var p1 = new PlayerState(PlayerId.Player1, 0,
                unitsOnBoard: enemyUnits ?? Array.Empty<Unit>(),
                generators: enemyGens ?? Array.Empty<Generator>());
            return new GameState(board, GameConfig.Default(), new[] { p0, p1 }, PlayerId.Player0, round, 100);
        }

        private static UnitStats Atk(int damage, int range = 2, int vision = 2) =>
            TestStates.Stats(damage: damage, range: range, vision: vision);

        [Test]
        public void AttackUnit_DealsDamage_NoReturnFire_MarksAttacked()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 5), new HexCoord(1, 0), 0);
            var r = GameEngine.Apply(Scene(Atk(3), new[] { enemy }), new AttackUnit(PlayerId.Player0, 1, 2));

            Assert.That(r.Success, Is.True);
            Assert.That(r.NewState.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp, Is.EqualTo(2));
            Assert.That(r.NewState.Player(PlayerId.Player0).UnitsOnBoard.Single().CurrentHp, Is.EqualTo(1)); // attacker unscathed
            Assert.That(r.NewState.AttackedUnitIds, Does.Contain(1));
        }

        [Test]
        public void AttackUnit_Kills_AwardsBounty_RemovesTarget()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Cost(3), new HexCoord(1, 0), 0); // 3 hp, cost 3
            var r = GameEngine.Apply(Scene(Atk(5), new[] { enemy }), new AttackUnit(PlayerId.Player0, 1, 2));

            Assert.That(r.NewState.Player(PlayerId.Player1).UnitsOnBoard, Is.Empty);
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(1)); // floor(3 * 0.5)
        }

        [Test]
        public void AttackUnit_Rejects_TargetOutOfRange()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 5), new HexCoord(2, 0), 0);
            var r = GameEngine.Apply(Scene(Atk(3, range: 1)), new AttackUnit(PlayerId.Player0, 1, 2));
            // target placed via enemyUnits param:
            r = GameEngine.Apply(Scene(Atk(3, range: 1), new[] { enemy }), new AttackUnit(PlayerId.Player0, 1, 2));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TargetNotInRange));
        }

        [Test]
        public void AttackUnit_Rejects_TargetNotVisible()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 5), new HexCoord(1, 0), 0);
            var r = GameEngine.Apply(Scene(Atk(3, range: 5, vision: 0), new[] { enemy }),
                new AttackUnit(PlayerId.Player0, 1, 2));
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TargetNotVisible));
        }

        [Test]
        public void AttackUnit_Rejects_FriendlyOrUnknownTarget()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 5), new HexCoord(1, 0), 0);
            var r = GameEngine.Apply(Scene(Atk(3), new[] { enemy }), new AttackUnit(PlayerId.Player0, 1, 1)); // own id
            Assert.That(r.Reason, Is.EqualTo(RejectionReason.TargetNotEnemy));
        }

        [Test]
        public void AttackUnit_Rejects_WhenAlreadyAttacked()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 10), new HexCoord(1, 0), 0);
            var first = GameEngine.Apply(Scene(Atk(3), new[] { enemy }), new AttackUnit(PlayerId.Player0, 1, 2));
            var second = GameEngine.Apply(first.NewState, new AttackUnit(PlayerId.Player0, 1, 2));
            Assert.That(second.Reason, Is.EqualTo(RejectionReason.UnitAlreadyAttacked));
        }

        [Test]
        public void AttackUnit_CanDestroyGenerator_AndPaysBounty()
        {
            var gen = new Generator(3, PlayerId.Player1, new HexCoord(1, 0), 0, 3);
            var r = GameEngine.Apply(Scene(Atk(5), enemyGens: new[] { gen }), new AttackUnit(PlayerId.Player0, 1, 3));

            Assert.That(r.NewState.Player(PlayerId.Player1).Generators, Is.Empty);
            Assert.That(r.NewState.Player(PlayerId.Player0).Points, Is.EqualTo(1)); // floor(GeneratorCost 2 * 0.5)
        }

        [Test]
        public void AttackUnit_KillingLastEnemyUnit_EndsGame_AfterOpening()
        {
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Cost(2), new HexCoord(1, 0), 0);
            var r = GameEngine.Apply(Scene(Atk(5), new[] { enemy }, round: 2), new AttackUnit(PlayerId.Player0, 1, 2));

            Assert.That(r.NewState.IsGameOver, Is.True);
            Assert.That(r.NewState.Winner, Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void MoveThenAttack_RemainsLegal()
        {
            var stats = TestStates.Stats(health: 5, damage: 1, movement: 2, range: 2, vision: 3);
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 10), new HexCoord(2, 0), 0);
            var initial = Scene(stats, new[] { enemy });

            var moved = GameEngine.Apply(initial,
                new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0)));
            var attacked = GameEngine.Apply(moved.NewState,
                new AttackUnit(PlayerId.Player0, 1, 2));

            Assert.That(moved.Success, Is.True);
            Assert.That(attacked.Success, Is.True);
        }

        [Test]
        public void AttackThenMove_IsRejected_AndRoutesAreClosedWithoutFakingSpentPoints()
        {
            var stats = TestStates.Stats(health: 5, damage: 1, movement: 2, range: 2, vision: 3);
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 10), new HexCoord(2, 0), 0);
            var attacked = GameEngine.Apply(Scene(stats, new[] { enemy }),
                new AttackUnit(PlayerId.Player0, 1, 2));
            var attacker = attacked.NewState.Player(PlayerId.Player0).UnitsOnBoard.Single();

            var move = GameEngine.Apply(attacked.NewState,
                new MoveUnit(PlayerId.Player0, 1, new HexCoord(1, 0)));

            Assert.Multiple(() =>
            {
                Assert.That(attacked.Success, Is.True);
                Assert.That(move.Success, Is.False);
                Assert.That(move.Reason.ToString(), Is.EqualTo("MovementEndedByAttack"));
                Assert.That(MovementService.Routes(attacked.NewState, attacker), Is.Empty);
                Assert.That(attacked.NewState.MovementSpent, Is.Empty);
            });
        }

        [Test]
        public void NewTurn_RestoresRoutesAfterAttack()
        {
            var stats = TestStates.Stats(health: 5, damage: 1, movement: 2, range: 2, vision: 3);
            var enemy = new Unit(2, PlayerId.Player1, TestStates.Stats(health: 10), new HexCoord(2, 0), 0);
            var state = GameEngine.Apply(Scene(stats, new[] { enemy }),
                new AttackUnit(PlayerId.Player0, 1, 2)).NewState;
            state = GameEngine.Apply(state, new EndTurn(PlayerId.Player0)).NewState;
            state = GameEngine.Apply(state, new EndTurn(PlayerId.Player1)).NewState;
            var attacker = state.Player(PlayerId.Player0).UnitsOnBoard.Single();

            Assert.That(MovementService.Routes(state, attacker), Does.ContainKey(new HexCoord(1, 0)));
        }
    }
}
