namespace HexWars.Engine
{
    /// <summary>
    /// Tunable parameters for procedural board generation (separate from gameplay <see cref="GameConfig"/>).
    /// All values are configurable; terrain is chosen by the relative weights.
    /// </summary>
    public sealed class BoardGenConfig
    {
        public int Width { get; }
        public int Height { get; }
        public int MaxElevation { get; }
        public int ZoneDepth { get; }

        /// <summary>Probability a tile is flat ground (elevation 0). Higher = more roamable flat area.</summary>
        public double FlatChance { get; }

        public int PlainsWeight { get; }
        public int ForestWeight { get; }
        public int RoughWeight { get; }
        public int WaterWeight { get; }

        public BoardGenConfig(
            int width = 9,
            int height = 7,
            int maxElevation = 4,
            int zoneDepth = 2,
            double flatChance = 0.6,
            int plainsWeight = 70,
            int forestWeight = 15,
            int roughWeight = 10,
            int waterWeight = 5)
        {
            Width = width;
            Height = height;
            MaxElevation = maxElevation;
            ZoneDepth = zoneDepth;
            FlatChance = flatChance;
            PlainsWeight = plainsWeight;
            ForestWeight = forestWeight;
            RoughWeight = roughWeight;
            WaterWeight = waterWeight;
        }

        public static BoardGenConfig Default() => new BoardGenConfig();
    }
}
