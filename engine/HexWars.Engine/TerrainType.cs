namespace HexWars.Engine
{
    /// <summary>Surface type of a hex (orthogonal to its elevation). Drives the per-terrain
    /// modifier table in <see cref="GameConfig"/>.</summary>
    public enum TerrainType
    {
        Plains = 0,
        Forest = 1,
        Rough = 2,
        Water = 3,
    }
}
