using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Seeded procedural board generator driven by a <see cref="BoardGenConfig"/>. Produces a
    /// width×height parallelogram of hex columns whose terrain AND elevation are mirror-symmetric
    /// across centre (so neither side gets better ground), mostly flat (per FlatChance) so units
    /// without Vertical Movement can roam, with each player's deployment zone on their flank.
    /// Deterministic: same seed → same board (System.Random only — never UnityEngine.Random).
    /// </summary>
    public sealed class RandomBoardGenerator : IBoardGenerator
    {
        private readonly BoardGenConfig _cfg;

        public RandomBoardGenerator(BoardGenConfig config)
        {
            _cfg = config;
        }

        public Board Generate(int seed)
        {
            var rng = new Random(seed);
            var elevation = new Dictionary<HexCoord, int>();
            var terrain = new Dictionary<HexCoord, TerrainType>();

            // Lay the W×H grid out in offset coordinates -> axial, so the footprint is a rectangle
            // (alternate columns nudged by half) rather than a sheared parallelogram.
            for (int row = 0; row < _cfg.Height; row++)
                for (int col = 0; col < _cfg.Width; col++)
                {
                    var cell = HexLayout.OffsetToAxial(col, row);
                    if (elevation.ContainsKey(cell)) continue;

                    var mirror = HexLayout.OffsetToAxial(_cfg.Width - 1 - col, _cfg.Height - 1 - row);
                    int e = rng.NextDouble() < _cfg.FlatChance ? 0 : 1 + rng.Next(_cfg.MaxElevation);
                    var t = WeightedTerrain(rng);

                    elevation[cell] = e; terrain[cell] = t;
                    elevation[mirror] = e; terrain[mirror] = t;
                }

            var tiles = new List<Tile>();
            var zone0 = new List<HexCoord>();
            var zone1 = new List<HexCoord>();
            for (int row = 0; row < _cfg.Height; row++)
                for (int col = 0; col < _cfg.Width; col++)
                {
                    var cell = HexLayout.OffsetToAxial(col, row);
                    tiles.Add(new Tile(cell, elevation[cell], terrain[cell]));
                    if (col < _cfg.ZoneDepth) zone0.Add(cell);
                    if (col >= _cfg.Width - _cfg.ZoneDepth) zone1.Add(cell);
                }

            return new Board(tiles, zone0, zone1);
        }

        private TerrainType WeightedTerrain(Random rng)
        {
            int total = _cfg.PlainsWeight + _cfg.ForestWeight + _cfg.RoughWeight + _cfg.WaterWeight;
            if (total <= 0) return TerrainType.Plains;

            int roll = rng.Next(total);
            if (roll < _cfg.PlainsWeight) return TerrainType.Plains;
            roll -= _cfg.PlainsWeight;
            if (roll < _cfg.ForestWeight) return TerrainType.Forest;
            roll -= _cfg.ForestWeight;
            if (roll < _cfg.RoughWeight) return TerrainType.Rough;
            return TerrainType.Water;
        }
    }
}
