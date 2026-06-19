using System;
using System.Collections.Generic;

namespace HexWars.Engine
{
    /// <summary>
    /// Seeded procedural board generator. Produces a width×height parallelogram of hex columns whose
    /// terrain AND elevation are mirror-symmetric across the centre (so neither side gets better
    /// ground), with each player's deployment zone on their flank. Deterministic: same seed → same
    /// board (System.Random only — never UnityEngine.Random).
    /// </summary>
    public sealed class RandomBoardGenerator : IBoardGenerator
    {
        private readonly int _width;
        private readonly int _height;
        private readonly int _maxElevation;
        private readonly int _zoneDepth;
        private readonly double _flatChance;

        /// <param name="flatChance">Probability a tile is flat ground (elevation 0). High by default so
        /// most of the board is a connected flat area that units with no Vertical Movement can roam;
        /// the rest is scattered raised terrain.</param>
        public RandomBoardGenerator(int width, int height, int maxElevation, int zoneDepth, double flatChance = 0.6)
        {
            _width = width;
            _height = height;
            _maxElevation = maxElevation;
            _zoneDepth = zoneDepth;
            _flatChance = flatChance;
        }

        public Board Generate(int seed)
        {
            var rng = new Random(seed);
            var elevation = new Dictionary<HexCoord, int>();
            var terrain = new Dictionary<HexCoord, TerrainType>();

            // Draw each mirror-pair once, assigning identical values to both halves.
            for (int r = 0; r < _height; r++)
                for (int q = 0; q < _width; q++)
                {
                    var cell = new HexCoord(q, r);
                    if (elevation.ContainsKey(cell)) continue;

                    var mirror = new HexCoord(_width - 1 - q, _height - 1 - r);
                    int e = rng.NextDouble() < _flatChance ? 0 : 1 + rng.Next(_maxElevation);
                    var t = WeightedTerrain(rng);

                    elevation[cell] = e; terrain[cell] = t;
                    elevation[mirror] = e; terrain[mirror] = t;
                }

            var tiles = new List<Tile>();
            var zone0 = new List<HexCoord>();
            var zone1 = new List<HexCoord>();
            for (int r = 0; r < _height; r++)
                for (int q = 0; q < _width; q++)
                {
                    var cell = new HexCoord(q, r);
                    tiles.Add(new Tile(cell, elevation[cell], terrain[cell]));
                    if (q < _zoneDepth) zone0.Add(cell);
                    if (q >= _width - _zoneDepth) zone1.Add(cell);
                }

            return new Board(tiles, zone0, zone1);
        }

        private static TerrainType WeightedTerrain(Random rng)
        {
            int roll = rng.Next(100);
            if (roll < 70) return TerrainType.Plains;
            if (roll < 85) return TerrainType.Forest;
            if (roll < 95) return TerrainType.Rough;
            return TerrainType.Water;
        }
    }
}
