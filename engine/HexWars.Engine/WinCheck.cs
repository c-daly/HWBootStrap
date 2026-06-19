namespace HexWars.Engine
{
    /// <summary>
    /// Pure win / elimination / stalemate rules, plus the <see cref="Evaluate"/> total-value score
    /// (reused by the round-cap tie-break and, later, as an AI heuristic).
    /// </summary>
    public static class WinCheck
    {
        private static readonly UnitStats MinimalUnit = new UnitStats(1, 0, 0, 0, 0, 0, 0, 0, 0);

        /// <summary>Total value of a player's position: banked points + on-board units + generators.
        /// Barracks templates are free reusable blueprints, so they add no value.</summary>
        public static int Evaluate(GameState state, PlayerId player)
        {
            var p = state.Player(player);
            int value = p.Points;

            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) value += u.Stats.PointCost;

            foreach (var g in p.Generators)
                if (g.IsAlive) value += state.Config.GeneratorCost;

            return value;
        }

        /// <summary>A player is eliminated with no living board units and too few points to field a unit
        /// — either by designing+deploying a minimal one, or deploying a cheaper existing template.</summary>
        public static bool IsEliminated(GameState state, PlayerId player)
        {
            var p = state.Player(player);

            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) return false;

            int cheapest = state.Config.DesignFee + Economy.DeployCost(MinimalUnit, state.Config);
            foreach (var stats in p.Barracks)
            {
                int c = Economy.DeployCost(stats, state.Config);
                if (c < cheapest) cheapest = c;
            }
            return p.Points < cheapest;
        }

        /// <summary>Whether the game has ended (a player eliminated after the opening, or the round cap
        /// reached). Use with <see cref="Resolve"/> to get the winner (which may be null on a draw).</summary>
        public static bool IsTerminal(GameState state)
        {
            if (state.Round >= 2 &&
                (IsEliminated(state, PlayerId.Player0) || IsEliminated(state, PlayerId.Player1)))
                return true;
            if (state.Round >= state.Config.RoundCap)
                return true;
            return false;
        }

        /// <summary>
        /// The winner, or null if the game continues / is a draw. Elimination is not declared during
        /// the opening round (round 1) so a player isn't lost before deploying. At the round cap the
        /// higher total value wins (equal = draw).
        /// </summary>
        public static PlayerId? Resolve(GameState state)
        {
            if (state.Round >= 2)
            {
                bool e0 = IsEliminated(state, PlayerId.Player0);
                bool e1 = IsEliminated(state, PlayerId.Player1);
                if (e0 && !e1) return PlayerId.Player1;
                if (e1 && !e0) return PlayerId.Player0;
                if (e0 && e1) return null; // mutual annihilation = draw
            }

            if (state.Round >= state.Config.RoundCap)
            {
                int v0 = Evaluate(state, PlayerId.Player0);
                int v1 = Evaluate(state, PlayerId.Player1);
                if (v0 > v1) return PlayerId.Player0;
                if (v1 > v0) return PlayerId.Player1;
                return null; // tie
            }

            return null;
        }
    }
}
