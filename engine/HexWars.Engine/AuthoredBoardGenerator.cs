namespace HexWars.Engine
{
    /// <summary>An <see cref="IBoardGenerator"/> that always returns a fixed, hand-authored board
    /// (the seed is ignored). Lets authored and procedural boards flow through the same path.</summary>
    public sealed class AuthoredBoardGenerator : IBoardGenerator
    {
        private readonly Board _board;

        public AuthoredBoardGenerator(Board board)
        {
            _board = board;
        }

        public Board Generate(int seed) => _board;
    }
}
