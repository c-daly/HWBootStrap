using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Turns a game-state transition (prev -> cur) into human-readable, color-coded event lines — moves,
    /// attacks/damage, kills, deploys, designs, round changes — by diffing the two states. Units are named
    /// by their DOMINANT-trait role (<see cref="Roles.Dominant"/>, the same classification the board icon
    /// uses) so it scales to arbitrary player-designed units, plus their cell so you can find them on the
    /// board. The acting side is taken from prev.ActivePlayer (the player whose command produced the
    /// transition), so attacks can be attributed without needing the command. Works in any viewer.
    /// </summary>
    public static class CombatLog
    {
        const string C0 = "#6FB1FF"; // Player0 = P1 (blue)
        const string C1 = "#FF7B6B"; // Player1 = P2 (red)

        public static List<string> Diff(GameState prev, GameState cur)
        {
            var lines = new List<string>();
            if (prev == null || cur == null) return lines;

            if (cur.Round > prev.Round) lines.Add($"<b>— Round {cur.Round} —</b>");

            var actor = prev.ActivePlayer; // whose command produced this transition

            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                var before = AliveUnits(prev.Player(pid));
                var after = AliveUnits(cur.Player(pid));

                foreach (var kv in before)                          // alive before, gone now = destroyed
                    if (!after.ContainsKey(kv.Key))
                    {
                        var u = kv.Value;
                        lines.Add(pid == actor
                            ? $"{Tag(pid)}'s {Role(u)} at {Cell(u.Cell)} destroyed"
                            : $"{Tag(actor)} destroyed {Tag(pid)}'s {Role(u)} at {Cell(u.Cell)}");
                    }

                foreach (var kv in after)                           // present now, absent before = deployed
                    if (!before.ContainsKey(kv.Key))
                        lines.Add($"{Tag(pid)} deployed a {Role(kv.Value)} at {Cell(kv.Value.Cell)}");

                foreach (var kv in after)                           // survived: took damage and/or moved
                    if (before.TryGetValue(kv.Key, out var u0))
                    {
                        var u = kv.Value;
                        if (u.CurrentHp < u0.CurrentHp)
                            lines.Add($"{Tag(actor)} hit {Tag(pid)}'s {Role(u)} at {Cell(u.Cell)} for {u0.CurrentHp - u.CurrentHp} (HP {u.CurrentHp})");
                        else if (!u.Cell.Equals(u0.Cell))
                            lines.Add($"{Tag(pid)} moved a {Role(u)} {Cell(u0.Cell)}→{Cell(u.Cell)}");
                    }

                if (cur.Player(pid).Barracks.Count > prev.Player(pid).Barracks.Count)
                    lines.Add($"{Tag(pid)} designed a unit");
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

        static string Role(Unit u) => Roles.Dominant(u.Stats).ToString();
        static string Tag(PlayerId p) => p == PlayerId.Player0 ? $"<color={C0}>P1</color>" : $"<color={C1}>P2</color>";
        static string Cell(HexCoord c) => $"({c.Q},{c.R})";
    }
}
