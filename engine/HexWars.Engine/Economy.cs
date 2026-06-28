namespace HexWars.Engine
{
    /// <summary>Pure economy rules: per-turn generator income (bounties live in <see cref="CombatResolver"/>).</summary>
    public static class Economy
    {
        /// <summary>Income this turn = round(Σ over the player's living generators of GeneratorOutput × strength).
        /// Ownership follows control (captures transfer generators), so this is control-based income.</summary>
        public static int Income(GameState state, PlayerId player)
        {
            double income = 0;
            foreach (var g in state.Player(player).Generators)
                if (g.IsAlive) income += state.Config.GeneratorOutput * g.Strength;
            return (int)System.Math.Round(income, System.MidpointRounding.AwayFromZero);
        }

        /// <summary>Per-turn upkeep = round(income × UpkeepFactor).</summary>
        public static int Upkeep(GameState state, PlayerId player)
            => (int)System.Math.Round(Income(state, player) * state.Config.UpkeepFactor,
                                      System.MidpointRounding.AwayFromZero);

        /// <summary>Points to deploy one clone of a design = round(PointCost × DeployCostMultiplier).</summary>
        public static int DeployCost(UnitStats stats, GameConfig config)
            => (int)System.Math.Round(stats.PointCost * config.DeployCostMultiplier,
                                      System.MidpointRounding.AwayFromZero);
    }
}
