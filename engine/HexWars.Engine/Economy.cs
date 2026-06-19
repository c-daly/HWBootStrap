namespace HexWars.Engine
{
    /// <summary>Pure economy rules: per-turn generator income (bounties live in <see cref="CombatResolver"/>).</summary>
    public static class Economy
    {
        /// <summary>Income a player earns this turn = (living generators) × GeneratorOutput.</summary>
        public static int Income(GameState state, PlayerId player)
        {
            int living = 0;
            foreach (var g in state.Player(player).Generators)
                if (g.IsAlive) living++;
            return living * state.Config.GeneratorOutput;
        }

        /// <summary>Points to deploy one clone of a design = round(PointCost × DeployCostMultiplier).</summary>
        public static int DeployCost(UnitStats stats, GameConfig config)
            => (int)System.Math.Round(stats.PointCost * config.DeployCostMultiplier,
                                      System.MidpointRounding.AwayFromZero);
    }
}
