using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Smoke test that the presentation assembly can see the engine DLL. If this compiles in Unity,
    /// the engine is correctly wired in. Delete once real presentation code references the engine.
    /// </summary>
    internal static class EngineLink
    {
        public static (double x, double z) WorldOf(HexCoord coord, double hexSize) =>
            HexLayout.ToWorld(coord, hexSize);
    }
}
