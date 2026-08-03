using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class BoundedSearchAgentTests
    {
        [Test]
        public void Decide_TakesAnAuthoritativeTerminalWin()
        {
            GameState state = OneHitWinState();
            var teacher = new BoundedSearchAgent(expansionBudget: 64, depth: 3);

            Command selected = teacher.Decide(state);
            Result result = GameEngine.Apply(state, selected);

            Assert.That(result.Success, Is.True);
            Assert.That(result.NewState.Winner, Is.EqualTo(PlayerId.Player0));
        }

        [Test]
        public void IdenticalTeachers_AreDeterministic()
        {
            GameState state = NonterminalFixture();

            Assert.That(
                Describe(new BoundedSearchAgent().Decide(state)),
                Is.EqualTo(Describe(new BoundedSearchAgent().Decide(state))));
        }

        [Test]
        public void Decide_StopsExpandingAtConfiguredBudget()
        {
            var teacher = new BoundedSearchAgent(expansionBudget: 2, depth: 4);

            teacher.Decide(NonterminalFixture());

            Assert.That(teacher.LastExpansionCount, Is.GreaterThan(0));
            Assert.That(teacher.LastExpansionCount, Is.LessThanOrEqualTo(2));
        }

        [Test]
        public void Decide_PrefersAttackBeforeMoveAndEndTurn_WhenNonterminalScoresTie()
        {
            var teacher = new BoundedSearchAgent(expansionBudget: 64, depth: 1, useHeuristic: false);

            Command selected = teacher.Decide(AttackAndMoveFixture());

            Assert.That(selected, Is.TypeOf<AttackUnit>());
        }

        [Test]
        public void Decide_UsesLexicalCommandTieBreak_ForEquivalentAttacks()
        {
            var teacher = new BoundedSearchAgent(expansionBudget: 64, depth: 1, useHeuristic: false);

            Command selected = teacher.Decide(LexicalAttackFixture());

            Assert.That(selected, Is.EqualTo(new AttackUnit(PlayerId.Player0, 1, 2)));
        }

        private static GameState OneHitWinState()
        {
            UnitStats attacker = TestStates.Stats(health: 3, damage: 3, movement: 1, range: 1, vision: 2);
            UnitStats target = TestStates.Stats(health: 1, movement: 1, range: 1, vision: 2);
            return State(
                new[] { new Unit(1, PlayerId.Player0, attacker, new HexCoord(0, 0), 0) },
                new[] { new Unit(2, PlayerId.Player1, target, new HexCoord(1, 0), 0) },
                round: 2);
        }

        private static GameState NonterminalFixture()
        {
            UnitStats stats = TestStates.Stats(health: 4, damage: 1, movement: 1, range: 1, vision: 2);
            return State(
                new[] { new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0) },
                new[] { new Unit(2, PlayerId.Player1, stats, new HexCoord(2, 0), 0) });
        }

        private static GameState AttackAndMoveFixture()
        {
            UnitStats attacker = TestStates.Stats(health: 4, damage: 0, movement: 1, range: 1, vision: 2);
            UnitStats target = TestStates.Stats(health: 4, movement: 1, range: 1, vision: 2);
            return State(
                new[] { new Unit(1, PlayerId.Player0, attacker, new HexCoord(0, 0), 0) },
                new[] { new Unit(2, PlayerId.Player1, target, new HexCoord(1, 0), 0) });
        }

        private static GameState LexicalAttackFixture()
        {
            UnitStats attacker = TestStates.Stats(health: 4, damage: 0, movement: 1, range: 1, vision: 2);
            UnitStats target = TestStates.Stats(health: 4, movement: 1, range: 1, vision: 2);
            return State(
                new[]
                {
                    new Unit(1, PlayerId.Player0, attacker, new HexCoord(0, 0), 0),
                    new Unit(3, PlayerId.Player0, attacker, new HexCoord(0, 1), 0),
                },
                new[]
                {
                    new Unit(2, PlayerId.Player1, target, new HexCoord(1, 0), 0),
                    new Unit(4, PlayerId.Player1, target, new HexCoord(1, 1), 0),
                });
        }

        private static GameState State(IReadOnlyList<Unit> p0Units, IReadOnlyList<Unit> p1Units, int round = 1)
        {
            var tiles = Enumerable.Range(0, 3)
                .SelectMany(q => Enumerable.Range(0, 2)
                    .Select(r => new Tile(new HexCoord(q, r), 0, TerrainType.Plains)))
                .ToArray();
            var board = new Board(tiles,
                zone0: new[] { new HexCoord(0, 0), new HexCoord(0, 1) },
                zone1: new[] { new HexCoord(2, 0), new HexCoord(2, 1) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0, null, p0Units, null),
                new PlayerState(PlayerId.Player1, 0, null, p1Units, null),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, round, 5);
        }

        private static string Describe(Command command) => command switch
        {
            AttackUnit attack => $"attack:{attack.AttackerId}:{attack.TargetId}",
            MoveUnit move => $"move:{move.UnitId}:{move.Dest.Q}:{move.Dest.R}",
            DeployUnit deploy => $"deploy:{deploy.TemplateIndex}:{deploy.Cell.Q}:{deploy.Cell.R}",
            EndTurn => "end-turn",
            _ => command.ToString() ?? string.Empty,
        };
    }
}
