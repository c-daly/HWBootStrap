namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Reward-shaping helpers shared by <see cref="TacticalEnv"/> and <see cref="DuelEnv"/> so training and
    /// self-play use an identical signal. Two ideas beyond the basic value-advantage shaping:
    ///  • a closing term (reward reducing the gap to the enemy) to break passive standoffs, and
    ///  • partial terminal credit at the round cap proportional to net value advantage, so inflicting more
    ///    damage than you take pays even without a full wipeout (the game still only *wins* by annihilation).
    /// </summary>
    internal static class RewardShaping
    {
        public static float Advantage(GameState s, PlayerId me, PlayerId foe)
            => WinCheck.Evaluate(s, me) - WinCheck.Evaluate(s, foe);

        /// <summary>Average hex distance from each of <paramref name="me"/>'s living units to its nearest
        /// living enemy unit (0 if either side has none). Lower = more engaged.</summary>
        public static float AvgGap(GameState s, PlayerId me, PlayerId foe)
        {
            var mine = s.Player(me);
            var their = s.Player(foe);
            int count = 0;
            long sum = 0;
            foreach (var u in mine.UnitsOnBoard)
            {
                if (!u.IsAlive) continue;
                int best = -1;
                foreach (var e in their.UnitsOnBoard)
                    if (e.IsAlive)
                    {
                        int d = HexCoord.Distance(u.Cell, e.Cell);
                        if (best < 0 || d < best) best = d;
                    }
                if (best >= 0) { sum += best; count++; }
            }
            return count == 0 ? 0f : (float)sum / count;
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
