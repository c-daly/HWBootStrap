using System;

namespace HexWars.Engine
{
    /// <summary>
    /// A turtle/banker baseline: it never spends points (no deploy / build / claim) and never advances —
    /// it only holds position and attacks whatever it can already reach, banking all income. Used to
    /// validate point decay: without decay it amasses a useless war chest; with decay that chest bleeds,
    /// and either way a spending opponent should beat it.
    /// </summary>
    public sealed class HoarderAgent : IAgent
    {
        private readonly Random _rng;

        public HoarderAgent(int seed) => _rng = new Random(seed);

        public Command Decide(GameState state)
        {
            var me = state.ActivePlayer;

            // take the attack that deals the most damage; otherwise end the turn (no moving, no spending)
            Command best = null;
            double bestScore = double.NegativeInfinity;
            foreach (var cmd in LegalMoves.For(state))
            {
                if (!(cmd is AttackUnit)) continue;
                var r = GameEngine.Apply(state, cmd);
                if (!r.Success) continue;
                double dmg = EnemyHp(state, me) - EnemyHp(r.NewState, me) + _rng.NextDouble() * 1e-6;
                if (dmg > bestScore) { bestScore = dmg; best = cmd; }
            }
            return best ?? new EndTurn(me);
        }

        private static int EnemyHp(GameState s, PlayerId me)
        {
            var opp = me == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            int hp = 0;
            foreach (var u in s.Player(opp).UnitsOnBoard) if (u.IsAlive) hp += u.CurrentHp;
            return hp;
        }
    }
}
