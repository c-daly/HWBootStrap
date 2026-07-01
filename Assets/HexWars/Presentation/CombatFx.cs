using UnityEngine;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>
    /// Damage numbers and kill explosions, spawned by diffing an applied state transition (prev -> now).
    /// Driven from GameBootstrap on every state change, so the numbers are authoritative on every path:
    /// local commands, the AI's attacks, and server-echoed moves online. (The old path read the target's
    /// HP synchronously after sending the command — online that ran before the server echo and always
    /// showed 0.) When the command is an attack, the popup also explains the result: how much the hit
    /// actually dealt, what high ground added, and what defense absorbed.
    /// </summary>
    public static class CombatFx
    {
        static readonly Color HitYellow = new Color(1f, 0.92f, 0.4f);
        static readonly Color KillOrange = new Color(0.95f, 0.45f, 0.18f);

        public static void Report(GameState prev, GameState now, BoardRenderer board, Command cmd)
        {
            if (prev == null || now == null || board == null) return;

            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
            {
                foreach (var u0 in prev.Player(pid).UnitsOnBoard)
                {
                    if (!u0.IsAlive) continue;

                    bool found = false;
                    foreach (var u1 in now.Player(pid).UnitsOnBoard)
                    {
                        if (u1.Id != u0.Id || !u1.IsAlive) continue;
                        found = true;
                        if (u1.CurrentHp < u0.CurrentHp)
                            DamagePopup.Spawn(Top(board, u1.Cell, u1.Elevation) + Vector3.up * 1.1f,
                                              (u0.CurrentHp - u1.CurrentHp) + Breakdown(prev, cmd, u0),
                                              HitYellow);
                        break;
                    }

                    if (!found) // alive before, gone now = destroyed
                    {
                        var at = Top(board, u0.Cell, u0.Elevation);
                        DamagePopup.Spawn(at + Vector3.up * 1.1f, u0.CurrentHp + Breakdown(prev, cmd, u0), KillOrange);
                        ExplosionFx.Spawn(at, KillOrange, 1.4f, true);
                    }
                }
            }
        }

        /// <summary>How the hit resolved, recomputed from the pre-attack state: "(+H high ground, B blocked)".
        /// Empty when the command isn't an attack on this unit or nothing modified the raw damage.</summary>
        static string Breakdown(GameState prev, Command cmd, Unit target)
        {
            if (!(cmd is AttackUnit atk) || atk.TargetId != target.Id) return "";

            foreach (var a in prev.Player(atk.Issuer).UnitsOnBoard)
            {
                if (a.Id != atk.AttackerId || !a.IsAlive) continue;

                int high = Mathf.Max(0, a.Elevation - target.Elevation) * prev.Config.DmgHighGroundBonus;
                int defense = target.Stats.Defense
                            + prev.Config.Terrain(prev.Board.TileAt(target.Cell).Terrain).Defense;
                int incoming = a.Stats.Damage + high;
                int dealt = CombatResolver.ComputeDamage(a.Stats.Damage, a.Elevation, target.Elevation,
                                                         defense, prev.Config);
                int blocked = Mathf.Max(0, incoming - dealt);

                if (high == 0 && blocked == 0) return ""; // clean direct hit — the number says it all
                string parts = "";
                if (high > 0) parts = "+" + high + " high ground";
                if (blocked > 0) parts += (parts.Length > 0 ? ", " : "") + blocked + " blocked";
                return "\n(" + parts + ")";
            }
            return "";
        }

        static Vector3 Top(BoardRenderer board, HexCoord cell, int elevation)
        {
            var w = HexLayout.ToWorld(cell, board.HexSize);
            return new Vector3((float)w.x, (elevation + 1) * board.LevelHeight, (float)w.z);
        }
    }
}
