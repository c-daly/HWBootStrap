using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// All tunable rules values live here so balancing is data, not code. Built up as the rules
    /// that consume each knob are implemented (TDD). Starts with the terrain table.
    /// </summary>
    public sealed class GameConfig
    {
        private readonly IReadOnlyDictionary<TerrainType, TerrainDef> _terrain;

        public GameConfig(IReadOnlyDictionary<TerrainType, TerrainDef> terrain)
        {
            _terrain = terrain;
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
