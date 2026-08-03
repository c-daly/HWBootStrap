using System;
using System.Collections.Generic;
using System.Globalization;

namespace HexWars.Engine
{
    /// <summary>
    /// A deterministic, fixed-budget minimax teacher for generating conversion demonstrations.
    /// It deliberately searches only through the engine's public legal-command and transition APIs.
    /// </summary>
    public sealed class BoundedSearchAgent : IAgent
    {
        public const int DefaultExpansionBudget = 512;
        public const int DefaultDepth = 4;

        private const double NonterminalLimit = 0.999;
        private readonly int _expansionBudget;
        private readonly int _depth;
        private readonly bool _useHeuristic;

        /// <summary>The number of authoritative transitions expanded by the most recent decision.</summary>
        public int LastExpansionCount { get; private set; }

        public BoundedSearchAgent(int expansionBudget = DefaultExpansionBudget, int depth = DefaultDepth, bool useHeuristic = true)
        {
            if (expansionBudget < 1) throw new ArgumentOutOfRangeException(nameof(expansionBudget));
            if (depth < 1) throw new ArgumentOutOfRangeException(nameof(depth));
            _expansionBudget = expansionBudget;
            _depth = depth;
            _useHeuristic = useHeuristic;
        }

        public Command Decide(GameState state)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            List<Command> commands = OrderedLegalMoves(state);
            if (commands.Count == 0) throw new InvalidOperationException("Cannot choose a command from a terminal state.");

            PlayerId rootPlayer = state.ActivePlayer;
            Command best = commands[0];
            double bestValue = double.NegativeInfinity;
            LastExpansionCount = 0;
            foreach (Command command in commands)
            {
                if (LastExpansionCount >= _expansionBudget) break;
                Result result = GameEngine.Apply(state, command);
                LastExpansionCount++;
                if (!result.Success) continue;

                double value = Search(result.NewState, rootPlayer, _depth - 1);
                if (value > bestValue)
                {
                    bestValue = value;
                    best = command;
                }
            }
            return best;
        }

        private double Search(GameState state, PlayerId rootPlayer, int remainingDepth)
        {
            if (state.IsGameOver) return TerminalValue(state, rootPlayer);
            if (remainingDepth <= 0 || LastExpansionCount >= _expansionBudget) return NonterminalValue(state, rootPlayer);

            List<Command> commands = OrderedLegalMoves(state);
            if (commands.Count == 0) return NonterminalValue(state, rootPlayer);

            bool maximize = state.ActivePlayer == rootPlayer;
            double best = maximize ? double.NegativeInfinity : double.PositiveInfinity;
            bool expanded = false;
            foreach (Command command in commands)
            {
                if (LastExpansionCount >= _expansionBudget) break;
                Result result = GameEngine.Apply(state, command);
                LastExpansionCount++;
                if (!result.Success) continue;

                expanded = true;
                double value = Search(result.NewState, rootPlayer, remainingDepth - 1);
                if (maximize)
                {
                    if (value > best) best = value;
                }
                else if (value < best)
                {
                    best = value;
                }
            }
            return expanded ? best : NonterminalValue(state, rootPlayer);
        }

        private double NonterminalValue(GameState state, PlayerId rootPlayer)
        {
            if (!_useHeuristic) return 0.0;
            PlayerId opponent = rootPlayer == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            double material = Material(state, rootPlayer) - Material(state, opponent);
            double pursuit = PersistentTargetPursuit(state, rootPlayer) - PersistentTargetPursuit(state, opponent);
            double value = (material + pursuit * 0.25) / 40.0;
            return Math.Max(-NonterminalLimit, Math.Min(NonterminalLimit, value));
        }

        private static double TerminalValue(GameState state, PlayerId rootPlayer)
        {
            if (state.Winner == null) return 0.0;
            return state.Winner == rootPlayer ? 1.0 : -1.0;
        }

        private static double Material(GameState state, PlayerId playerId)
        {
            PlayerState player = state.Player(playerId);
            double material = player.Points * 0.25;
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.IsAlive) material += unit.Stats.PointCost * (double)unit.CurrentHp / unit.Stats.Health;
            foreach (Generator generator in player.Generators)
                if (generator.IsAlive) material += state.Config.GeneratorOutput * generator.Strength * 2.0;
            return material;
        }

        private static double PersistentTargetPursuit(GameState state, PlayerId playerId)
        {
            PlayerState player = state.Player(playerId);
            PlayerState opponent = state.Opponent(playerId);
            double pursuit = 0.0;
            foreach (Unit unit in player.UnitsOnBoard)
            {
                if (!unit.IsAlive) continue;
                int distance = NearestEnemyDistance(unit.Cell, opponent);
                if (distance >= 0) pursuit += 1.0 / (1.0 + distance);
            }
            return pursuit;
        }

        private static int NearestEnemyDistance(HexCoord from, PlayerState opponent)
        {
            int best = -1;
            foreach (Unit enemy in opponent.UnitsOnBoard)
                if (enemy.IsAlive) best = Nearest(best, HexCoord.Distance(from, enemy.Cell));
            foreach (Generator enemy in opponent.Generators)
                if (enemy.IsAlive) best = Nearest(best, HexCoord.Distance(from, enemy.Cell));
            return best;
        }

        private static int Nearest(int current, int candidate) => current < 0 || candidate < current ? candidate : current;

        private static List<Command> OrderedLegalMoves(GameState state)
        {
            var commands = new List<Command>(LegalMoves.For(state));
            commands.Sort((left, right) => string.CompareOrdinal(CommandKey(left), CommandKey(right)));
            return commands;
        }

        private static string CommandKey(Command command)
        {
            switch (command)
            {
                case AttackUnit attack: return "0:attack:" + Number(attack.AttackerId) + ":" + Number(attack.TargetId);
                case MoveUnit move: return "1:move:" + Number(move.UnitId) + ":" + Number(move.Dest.Q) + ":" + Number(move.Dest.R);
                case DeployUnit deploy: return "2:deploy:" + Number(deploy.TemplateIndex) + ":" + Number(deploy.Cell.Q) + ":" + Number(deploy.Cell.R);
                case CaptureHex capture: return "3:capture:" + Number(capture.Cell.Q) + ":" + Number(capture.Cell.R);
                case BuildGenerator build: return "4:build:" + Number(build.Cell.Q) + ":" + Number(build.Cell.R);
                case EndTurn: return "5:end-turn";
                default: return "6:" + command.GetType().FullName;
            }
        }

        private static string Number(int value) => value.ToString(CultureInfo.InvariantCulture);
    }
}
