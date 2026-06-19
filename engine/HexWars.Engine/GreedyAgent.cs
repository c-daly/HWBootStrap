using System;

namespace HexWars.Engine
{
    /// <summary>
    /// A one-ply greedy heuristic agent: it bootstraps an army by designing one balanced fighter, then
    /// each step simulates every legal command and keeps the one that most improves its position. The
    /// score values committed force above banked points (so it actually deploys and fights, instead of
    /// hoarding, since deploy cost is value-neutral), counts kills via the opponent's lost value, and
    /// adds a small reward for closing on the enemy. A much stronger balance baseline than RandomAgent.
    /// </summary>
    public sealed class GreedyAgent : IAgent
    {
        private readonly Random _rng;
        private readonly double _pointsWeight; // banked points are worth less than units on the board
        private readonly double _aggression;   // per-hex reward for closing on the nearest enemy

        public GreedyAgent(int seed, double pointsWeight = 0.5, double aggression = 0.08)
        {
            _rng = new Random(seed);
            _pointsWeight = pointsWeight;
            _aggression = aggression;
        }

        public Command Decide(GameState state)
        {
            var me = state.ActivePlayer;
            var player = state.Player(me);

            // bootstrap: design one fighter if the barracks is empty and we can field it
            if (player.Barracks.Count == 0)
            {
                var fighter = Fighter();
                int cost = state.Config.DesignFee + Economy.DeployCost(fighter, state.Config);
                if (player.Points >= cost) return new CreateUnit(me, fighter);
            }

            var moves = LegalMoves.For(state); // includes EndTurn, so there is always a fallback
            Command best = new EndTurn(me);
            double bestScore = double.NegativeInfinity;
            foreach (var cmd in moves)
            {
                var result = GameEngine.Apply(state, cmd);
                if (!result.Success) continue;
                double sc = Score(result.NewState, me) + _rng.NextDouble() * 1e-6; // tiny tie-break
                if (sc > bestScore) { bestScore = sc; best = cmd; }
            }
            return best;
        }

        // balanced ~14-pt all-rounder: durable, hits, mobile, a little vertical reach + sight
        private static UnitStats Fighter() =>
            new UnitStats(health: 3, damage: 3, defense: 0, movement: 2, verticalMovement: 1,
                          range: 1, rangeArc: 1, vision: 2, visionArc: 1);

        private double Score(GameState s, PlayerId me)
        {
            var opp = me == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            double score = Side(s, me) - Side(s, opp);

            var myP = s.Player(me);
            var opP = s.Player(opp);
            foreach (var u in myP.UnitsOnBoard)
            {
                if (!u.IsAlive) continue;
                int d = NearestEnemyDist(u.Cell, opP);
                if (d >= 0) score -= _aggression * d;
            }
            return score;
        }

        private double Side(GameState s, PlayerId p)
        {
            var ps = s.Player(p);
            double v = _pointsWeight * ps.Points;
            foreach (var u in ps.UnitsOnBoard) if (u.IsAlive) v += u.Stats.PointCost;
            foreach (var g in ps.Generators) if (g.IsAlive) v += s.Config.GeneratorCost;
            return v;
        }

        private static int NearestEnemyDist(HexCoord from, PlayerState opp)
        {
            int best = -1;
            foreach (var e in opp.UnitsOnBoard)
                if (e.IsAlive) { int d = HexCoord.Distance(from, e.Cell); if (best < 0 || d < best) best = d; }
            foreach (var g in opp.Generators)
                if (g.IsAlive) { int d = HexCoord.Distance(from, g.Cell); if (best < 0 || d < best) best = d; }
            return best;
        }
    }
}
