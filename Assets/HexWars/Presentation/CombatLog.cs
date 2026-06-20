using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Turns a game-state transition (prev -> cur) into human-readable event lines — kills, damage,
    /// deploys, unit designs, round changes — by diffing the two states. Command-agnostic, so it works in
    /// any viewer (playable game, AI spectator, model duel, replay) without needing the command itself.
    /// </summary>
    public static class CombatLog
    {
        public static List<string> Diff(GameState prev, GameState cur)
        {
            var lines = new List<string>();
            if (prev == null || cur == null) return lines;

            if (cur.Round > prev.Round) lines.Add($"— Round {cur.Round} —");

            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                var before = AliveUnits(prev.Player(pid));
                var after = AliveUnits(cur.Player(pid));

                foreach (var kv in before)                          // alive before, gone now = destroyed
                    if (!after.ContainsKey(kv.Key))
                        lines.Add($"{Name(pid)} unit {kv.Key} destroyed");

                foreach (var kv in after)                           // present now, absent before = deployed
                    if (!before.ContainsKey(kv.Key))
                        lines.Add($"{Name(pid)} deployed unit {kv.Key} at {Cell(kv.Value.Cell)}");

                foreach (var kv in after)                           // survived but lost HP = took damage
                    if (before.TryGetValue(kv.Key, out var u0) && kv.Value.CurrentHp < u0.CurrentHp)
                        lines.Add($"{Name(pid)} unit {kv.Key} took {u0.CurrentHp - kv.Value.CurrentHp} (HP {kv.Value.CurrentHp})");

                if (cur.Player(pid).Barracks.Count > prev.Player(pid).Barracks.Count)
                    lines.Add($"{Name(pid)} designed a unit");
            }

            return lines;
        }

        static Dictionary<int, Unit> AliveUnits(PlayerState p)
        {
            var d = new Dictionary<int, Unit>();
            foreach (var u in p.UnitsOnBoard)
                if (u.IsAlive) d[u.Id] = u;
            return d;
        }

        static string Name(PlayerId p) => p == PlayerId.Player0 ? "P1" : "P2";
        static string Cell(HexCoord c) => $"({c.Q},{c.R})";
    }
}
