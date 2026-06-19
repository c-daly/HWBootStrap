namespace HexWars.Engine
{
    /// <summary>Produces a <see cref="Board"/>. Random and authored sources implement this so the
    /// rest of the engine/presentation is agnostic to where a board came from.</summary>
    public interface IBoardGenerator
    {
        Board Generate(int seed);
    }
}
