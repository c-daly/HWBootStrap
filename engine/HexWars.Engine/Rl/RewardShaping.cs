namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Reward-shaping helpers shared by <see cref="TacticalEnv"/> and <see cref="DuelEnv"/> so training and
    /// self-play use an identical signal. Two ideas beyond the basic value-advantage shaping:
    ///  • an active-piece closing term — when you MOVE a unit, reward it for reducing its own distance to the
    ///    nearest enemy (measured at your move, before the opponent replies: undiluted, no enemy-move noise), and
    ///  • partial terminal credit at the round cap proportional to net value advantage, so inflicting more
    ///    damage than you take pays even without a full wipeout (the game still only *wins* by annihilation).
    /// </summary>
    internal static class RewardShaping
    {
        public static float Advantage(GameState s, PlayerId me, PlayerId foe)
            => WinCheck.Evaluate(s, me) - WinCheck.Evaluate(s, foe);

        /// <summary>Hex distance from <paramref name="cell"/> to the nearest living enemy unit, or -1 if none.</summary>
        public static int NearestEnemyDist(GameState s, PlayerId foe, HexCoord cell)
        {
            int best = -1;
            foreach (var e in s.Player(foe).UnitsOnBoard)
                if (e.IsAlive)
                {
                    int d = HexCoord.Distance(cell, e.Cell);
                    if (best < 0 || d < best) best = d;
                }
            return best;
        }

        /// <summary>Distance from the living unit with <paramref name="unitId"/> (owned by <paramref name="owner"/>)
        /// to its nearest enemy, or -1 if the unit is gone or there are no enemies.</summary>
        public static int GapOfUnit(GameState s, int unitId, PlayerId owner, PlayerId foe)
        {
            foreach (var u in s.Player(owner).UnitsOnBoard)
                if (u.Id == unitId && u.IsAlive)
                    return NearestEnemyDist(s, foe, u.Cell);
            return -1;
        }

        /// <summary>Partial reward at a draw (round cap): net value advantage normalized by the starting
        /// army value, clamped to [-1,1], times <paramref name="weight"/>. Ahead on kills pays up to
        /// +weight; behind pays down to -weight; dead even = 0.</summary>
        public static float DrawCredit(GameState s, PlayerId me, PlayerId foe, float armyValue, float weight)
        {
            if (armyValue <= 0f) return 0f;
            float a = Advantage(s, me, foe) / armyValue;
            if (a > 1f) a = 1f;
            else if (a < -1f) a = -1f;
            return weight * a;
        }
    }
}
