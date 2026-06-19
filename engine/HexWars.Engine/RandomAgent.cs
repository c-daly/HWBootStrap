using System;

namespace HexWars.Engine
{
    /// <summary>
    /// Reference agent: plays deterministically from a seed (so episodes are reproducible). It often
    /// ends its turn (keeping rounds advancing), sometimes designs a random unit template (capped so
    /// it stays affordable to deploy), and otherwise picks a random legal move — including deploying
    /// clones of its templates. Useful as a self-play baseline and a smoke test.
    /// </summary>
    public sealed class RandomAgent : IAgent
    {
        private readonly Random _rng;

        public RandomAgent(int seed)
        {
            _rng = new Random(seed);
        }

        public Command Decide(GameState state)
        {
            var me = state.ActivePlayer;
            var player = state.Player(me);

            int roll = _rng.Next(100);
            if (roll < 25) return new EndTurn(me);                       // end turns often -> rounds advance
            if (roll < 45 && player.Points >= 1) return RandomCreate(me, player.Points);

            var moves = LegalMoves.For(state);
            return moves[_rng.Next(moves.Count)];                        // always non-empty (includes EndTurn)
        }

        private Command RandomCreate(PlayerId me, int budget)
        {
            int spend = 1 + _rng.Next(Math.Min(budget, 8)); // affordable, capped design size
            var s = new int[9];
            s[0] = 1;                                        // >= 1 health so the unit can exist
            for (int remaining = spend - 1; remaining > 0; remaining--)
                s[_rng.Next(9)]++;

            var stats = new UnitStats(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], s[8]);
            return new CreateUnit(me, stats);
        }
    }
}
