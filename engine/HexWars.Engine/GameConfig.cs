using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// All tunable rules values — balancing is data, not code. Placeholder defaults; every value
    /// is provisional (balance and fun win). Vertical movement / vertical reach cost a flat 1 per
    /// level by design (uniform 1pt/hex), so they are not tunables here.
    /// </summary>
    public sealed class GameConfig
    {
        private readonly IReadOnlyDictionary<TerrainType, TerrainDef> _terrain;

        /// <summary>Points each player starts with.</summary>
        public int StartingPoints { get; }

        /// <summary>Fraction of a destroyed unit's build cost paid to the killer's owner.</summary>
        public double BountyRate { get; }

        public int GeneratorCost { get; }
        public int GeneratorOutput { get; }
        public int GeneratorHealth { get; }

        /// <summary>Minimum damage a hit deals after defense (e.g. 0 or 1).</summary>
        public int DamageFloor { get; }

        /// <summary>Bonus damage per level of attacker high-ground advantage.</summary>
        public int DmgHighGroundBonus { get; }

        /// <summary>Bonus horizontal range per level of attacker high-ground advantage.</summary>
        public int RangeHighGroundBonus { get; }

        /// <summary>Round at which the stalemate backstop ends the game by total value.</summary>
        public int RoundCap { get; }

        /// <summary>The turn-structure rule (default <see cref="AllUnitsPolicy"/>).</summary>
        public ITurnPolicy TurnPolicy { get; }

        public GameConfig(
            IReadOnlyDictionary<TerrainType, TerrainDef> terrain,
            int startingPoints = 12,
            double bountyRate = 0.5,
            int generatorCost = 2,
            int generatorOutput = 1,
            int generatorHealth = 3,
            int damageFloor = 0,
            int dmgHighGroundBonus = 1,
            int rangeHighGroundBonus = 1,
            int roundCap = 40,
            ITurnPolicy? turnPolicy = null)
        {
            _terrain = terrain;
            StartingPoints = startingPoints;
            BountyRate = bountyRate;
            GeneratorCost = generatorCost;
            GeneratorOutput = generatorOutput;
            GeneratorHealth = generatorHealth;
            DamageFloor = damageFloor;
            DmgHighGroundBonus = dmgHighGroundBonus;
            RangeHighGroundBonus = rangeHighGroundBonus;
            RoundCap = roundCap;
            TurnPolicy = turnPolicy ?? new AllUnitsPolicy();
        }

        /// <summary>Modifier table for the given terrain.</summary>
        public TerrainDef Terrain(TerrainType type) => _terrain[type];

        /// <summary>Default ruleset (placeholder values, all tunable).</summary>
        public static GameConfig Default() => new GameConfig(new Dictionary<TerrainType, TerrainDef>
        {
            { TerrainType.Plains, new TerrainDef(moveCost: 1, concealment: 0, defense: 0, passable: true) },
            { TerrainType.Forest, new TerrainDef(moveCost: 2, concealment: 2, defense: 1, passable: true) },
            { TerrainType.Rough,  new TerrainDef(moveCost: 2, concealment: 1, defense: 1, passable: true) },
            { TerrainType.Water,  new TerrainDef(moveCost: 3, concealment: 0, defense: 0, passable: true) },
        });
    }
}
