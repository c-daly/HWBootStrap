using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public sealed class AttackPreviewTargetsTests
    {
        [Test]
        public void Resolve_FromCurrentPosition_ReturnsOnlyTargetableEnemyUnits()
        {
            var board = LineBoard(4);
            var attacker = Unit(1, PlayerId.Player0, cell: 0, range: 2, vision: 4);
            var nearEnemy = Unit(2, PlayerId.Player1, cell: 2);
            var farEnemy = Unit(3, PlayerId.Player1, cell: 3);
            var state = State(board, new[] { attacker }, new[] { nearEnemy, farEnemy });

            var targets = AttackPreviewTargets.Resolve(state, attacker, null, PlayerId.Player0);

            Assert.That(targets.Select(target => target.UnitId), Is.EqualTo(new[] { nearEnemy.Id }));
        }

        [Test]
        public void Resolve_FromPreviewDestination_RecomputesTargetsAsIfAttackerMoved()
        {
            var board = LineBoard(4);
            var attacker = Unit(1, PlayerId.Player0, cell: 0, range: 1, vision: 4);
            var nearEnemy = Unit(2, PlayerId.Player1, cell: 1);
            var farEnemy = Unit(3, PlayerId.Player1, cell: 3);
            var state = State(board, new[] { attacker }, new[] { nearEnemy, farEnemy });

            var targets = AttackPreviewTargets.Resolve(
                state, attacker, new HexCoord(2, 0), PlayerId.Player0);

            Assert.That(targets.Select(target => target.UnitId),
                Is.EqualTo(new[] { nearEnemy.Id, farEnemy.Id }));
        }

        [Test]
        public void Resolve_UnderFog_DoesNotRevealTargetsHiddenBeforePreviewMove()
        {
            var board = LineBoard(4);
            var attacker = Unit(1, PlayerId.Player0, cell: 0, range: 4, vision: 1);
            var visibleEnemy = Unit(2, PlayerId.Player1, cell: 1);
            var hiddenEnemy = Unit(3, PlayerId.Player1, cell: 3);
            var state = State(
                board, new[] { attacker }, new[] { visibleEnemy, hiddenEnemy }, fogOfWar: true);

            var targets = AttackPreviewTargets.Resolve(
                state, attacker, new HexCoord(2, 0), PlayerId.Player0);

            Assert.That(targets.Select(target => target.UnitId),
                Is.EqualTo(new[] { visibleEnemy.Id }));
        }

        [Test]
        public void Resolve_AfterAttackerHasFired_ReturnsNoTargets()
        {
            var board = LineBoard(2);
            var attacker = Unit(1, PlayerId.Player0, cell: 0, range: 1, vision: 2);
            var enemy = Unit(2, PlayerId.Player1, cell: 1);
            var state = State(
                board, new[] { attacker }, new[] { enemy }, attackedUnitIds: new[] { attacker.Id });

            var targets = AttackPreviewTargets.Resolve(state, attacker, null, PlayerId.Player0);

            Assert.That(targets, Is.Empty);
        }

        [Test]
        public void Resolve_WithOffBoardPreviewDestination_ReturnsNoTargets()
        {
            var board = LineBoard(2);
            var attacker = Unit(1, PlayerId.Player0, cell: 0, range: 4, vision: 4);
            var enemy = Unit(2, PlayerId.Player1, cell: 1);
            var state = State(board, new[] { attacker }, new[] { enemy });

            var targets = AttackPreviewTargets.Resolve(
                state, attacker, new HexCoord(-1, 0), PlayerId.Player0);

            Assert.That(targets, Is.Empty);
        }

        static Board LineBoard(int length) =>
            new Board(Enumerable.Range(0, length)
                .Select(q => new Tile(new HexCoord(q, 0), 0, TerrainType.Plains))
                .ToArray());

        static Unit Unit(int id, PlayerId owner, int cell, int range = 0, int vision = 0) =>
            new Unit(id, owner,
                new UnitStats(health: 5, damage: 1, defense: 0, movement: 4,
                    verticalMovement: 2, range: range, rangeArc: 0,
                    vision: vision, visionArc: 0),
                new HexCoord(cell, 0), 0);

        static GameState State(
            Board board,
            Unit[] mine,
            Unit[] enemies,
            bool fogOfWar = false,
            int[] attackedUnitIds = null) =>
            new GameState(board, GameConfig.Default(fogOfWar: fogOfWar),
                new[]
                {
                    new PlayerState(PlayerId.Player0, 0, unitsOnBoard: mine),
                    new PlayerState(PlayerId.Player1, 0, unitsOnBoard: enemies)
                },
                PlayerId.Player0, round: 1, nextEntityId: 100,
                attackedUnitIds: attackedUnitIds);
    }
}
