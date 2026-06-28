namespace HexWars.Engine
{
    /// <summary>
    /// Pure win / elimination / stalemate rules, plus the <see cref="Evaluate"/> total-value score
    /// (reused by the round-cap tie-break and, later, as an AI heuristic).
    /// </summary>
    public static class WinCheck
    {
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

        /// <summary>The configurable Score composite: a weighted sum of kills (destroyed enemy value),
        /// banked points, surviving army value, and controlled-hex count. Weights live in
        /// <see cref="GameConfig"/>. Phase 1 counts controlled hexes flat; per-hex value arrives in Phase 3.
        /// This is the shared "who's winning" value the Score win-check reads (and the RL reward will too).</summary>
        public static int Score(GameState state, PlayerId player)
        {
            var p = state.Player(player);
            var cfg = state.Config;

            int army = 0;
            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) army += u.Stats.PointCost;

            int territory = state.Board.ControlledCount(player);

            return cfg.ScoreKills * p.DestroyedValue
                 + cfg.ScorePoints * p.Points
                 + cfg.ScoreArmy * army
                 + cfg.ScoreTerritory * territory;
        }

        /// <summary>A player is eliminated (annihilated) when it has no living board units and cannot
        /// redeploy any existing barracks template (so a bounty-funded comeback is still possible, but an
        /// empty board with no affordable template — or no template at all — is a loss).</summary>
        public static bool IsEliminated(GameState state, PlayerId player)
        {
            var p = state.Player(player);

            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) return false; // still has an army

            foreach (var stats in p.Barracks)
                if (p.Points >= Economy.DeployCost(stats, state.Config)) return false; // can redeploy

            return true; // no units and nothing it can field
        }

        /// <summary>Whether the game has ended: an instant win condition fired, or the round cap reached.</summary>
        public static bool IsTerminal(GameState state)
        {
            var win = state.Config.WinConditions;

            if ((win & WinBy.Annihilation) != 0 && state.Round >= 2 &&
                (IsEliminated(state, PlayerId.Player0) || IsEliminated(state, PlayerId.Player1)))
                return true;

            if ((win & WinBy.Economy) != 0 && (EconomyWinner(state) != null))
                return true;

            if (state.Round >= state.Config.RoundCap)
                return true;

            return false;
        }

        /// <summary>The winner, or null on a draw / continuing game. Instant conditions (annihilation,
        /// economy) resolve immediately in priority order; if none fired and the round cap is reached, the
        /// Score condition (if enabled) decides by higher score, else it is a draw.</summary>
        public static PlayerId? Resolve(GameState state)
        {
            var win = state.Config.WinConditions;

            if ((win & WinBy.Annihilation) != 0 && state.Round >= 2)
            {
                bool e0 = IsEliminated(state, PlayerId.Player0);
                bool e1 = IsEliminated(state, PlayerId.Player1);
                if (e0 && !e1) return PlayerId.Player1;
                if (e1 && !e0) return PlayerId.Player0;
            }

            if ((win & WinBy.Economy) != 0)
            {
                var ew = EconomyWinner(state);
                if (ew != null) return ew;
            }

            if (state.Round >= state.Config.RoundCap && (win & WinBy.Score) != 0)
            {
                int s0 = Score(state, PlayerId.Player0);
                int s1 = Score(state, PlayerId.Player1);
                if (s0 > s1) return PlayerId.Player0;
                if (s1 > s0) return PlayerId.Player1;
            }

            return null; // draw / continue
        }

        /// <summary>The player at or past the Economy threshold (higher points if both), or null.</summary>
        private static PlayerId? EconomyWinner(GameState state)
        {
            int t = state.Config.EconomyWinThreshold;
            int p0 = state.Player(PlayerId.Player0).Points;
            int p1 = state.Player(PlayerId.Player1).Points;
            bool a = p0 >= t, b = p1 >= t;
            if (a && (!b || p0 >= p1)) return PlayerId.Player0;
            if (b) return PlayerId.Player1;
            return null;
        }
    }
}
