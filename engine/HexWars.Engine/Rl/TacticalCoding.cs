using System;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Seat-relative observation encoding + action codec for the tactical RL setup. Stateless and
    /// parameterized by seat, so the single-agent env and the two-agent duel env share one source of
    /// truth. Observation is from the given seat's point of view ("my" vs "enemy"); actions are
    /// 1 (EndTurn) + (move | attack) × unit-slot × cell, masked against the engine's legal moves.
    /// </summary>
    public static class TacticalCoding
    {
        // Spatial observation, biomes off: one HP plane per (owner × roster role) + one elevation plane,
        // emitted channel-major as [channel][cell] (cells are row-major over the board), so the Python side
        // reshapes the board part to (C, H, W) for a CNN. A few scalar globals follow. C = 2·roster + 1.
        public const int Globals = 5; // myPoints, foePoints, round, myCount, foeCount

        /// <summary>Observation channels: my-role + enemy-role HP planes (2·roster) plus an elevation plane.</summary>
        public static int Channels(int roster) => 2 * roster + 1;

        private static PlayerId Other(PlayerId p) => p == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        // ---- observation ----

        public static float[] Observe(GameState s, PlayerId seat, TacticalLayout L)
        {
            var foe = Other(seat);
            int N = L.CellCount, R = L.Roster, C = 2 * R + 1;
            var obs = new float[C * N + Globals];
            var board = s.Board;
            float maxElev = Math.Max(1, L.BoardGen.MaxElevation);

            int elevPlane = 2 * R; // last channel
            for (int i = 0; i < N; i++)
            {
                if (!board.Contains(L.Cells[i])) continue;
                obs[elevPlane * N + i] = Math.Min(1f, board.TileAt(L.Cells[i]).Elevation / maxElev);
            }

            WriteUnits(obs, s.Player(seat), 0, N, L); // my units  -> role planes 0..R-1
            WriteUnits(obs, s.Player(foe), R, N, L);  // enemy units -> role planes R..2R-1

            int g = C * N;
            obs[g + 0] = Math.Min(1f, s.Player(seat).Points / 50f);
            obs[g + 1] = Math.Min(1f, s.Player(foe).Points / 50f);
            obs[g + 2] = Math.Min(1f, s.Round / (float)Math.Max(1, L.Game.RoundCap));
            obs[g + 3] = Math.Min(1f, AliveUnits(s.Player(seat)) / (float)Math.Max(1, L.Roster));
            obs[g + 4] = Math.Min(1f, AliveUnits(s.Player(foe)) / (float)Math.Max(1, L.Roster));
            return obs;
        }

        // Light up plane (planeBase + role) at the unit's cell with its HP fraction. Role-keyed (not slot),
        // so a deployed reinforcement shows up on its template's plane.
        private static void WriteUnits(float[] obs, PlayerState p, int planeBase, int N, TacticalLayout L)
        {
            foreach (var u in p.UnitsOnBoard)
            {
                if (!u.IsAlive || !L.CellIndex.TryGetValue(u.Cell, out int i)) continue;
                int role = RoleOf(u.Stats, L);
                if (role < 0) continue;
                obs[(planeBase + role) * N + i] = Math.Min(1f, u.CurrentHp / (float)Math.Max(1, u.Stats.Health));
            }
        }

        /// <summary>The roster-role index (0..R-1) whose template matches these stats, or -1 if none
        /// (shouldn't happen in the fixed-roster tactical setup; deployed units copy a template's stats).</summary>
        private static int RoleOf(UnitStats st, TacticalLayout L)
        {
            for (int r = 0; r < L.RosterStats.Count; r++)
            {
                var t = L.RosterStats[r];
                if (t.Health == st.Health && t.Damage == st.Damage && t.Defense == st.Defense &&
                    t.Movement == st.Movement && t.VerticalMovement == st.VerticalMovement &&
                    t.Range == st.Range && t.RangeArc == st.RangeArc &&
                    t.Vision == st.Vision && t.VisionArc == st.VisionArc) return r;
            }
            return -1;
        }

        private static int AliveUnits(PlayerState p)
        {
            int c = 0;
            foreach (var u in p.UnitsOnBoard) if (u.IsAlive) c++;
            return c;
        }

        // ---- action mask / codec ----

        public static bool[] Mask(GameState s, PlayerId seat, int[] slotToUnitId, TacticalLayout L)
        {
            var mask = new bool[L.ActionCount];
            mask[0] = true; // EndTurn always available
            foreach (var cmd in LegalMoves.For(s))
            {
                int ix = Encode(cmd, s, seat, slotToUnitId, L);
                if (ix >= 0 && ix < mask.Length) mask[ix] = true;
            }
            return mask;
        }

        public static Command? Decode(int action, GameState s, PlayerId seat, int[] slotToUnitId, TacticalLayout L)
        {
            if (action <= 0) return new EndTurn(seat);

            int n = L.CellCount, r = L.Roster;
            int a = action - 1;
            int region = a / (r * n);   // 0 = move, 1 = attack, 2 = deploy
            a %= r * n;
            int idx = a / n, cell = a % n;   // idx = unit slot (move/attack) or barracks template (deploy)
            if (idx < 0 || idx >= r || cell < 0 || cell >= n) return null;

            var coord = L.Cells[cell];

            if (region == 2) // deploy a copy of barracks template `idx` onto `coord` (engine validates cost/zone)
                return new DeployUnit(seat, idx, coord);

            int unitId = slotToUnitId[idx];
            if (unitId < 0 || !IsLivingUnit(s, seat, unitId)) return null;
            if (region == 0) return new MoveUnit(seat, unitId, coord);

            int targetId = EnemyEntityAt(s, seat, coord);
            return targetId < 0 ? null : new AttackUnit(seat, unitId, targetId);
        }

        private static int Encode(Command cmd, GameState s, PlayerId seat, int[] slotToUnitId, TacticalLayout L)
        {
            int n = L.CellCount, r = L.Roster;
            switch (cmd)
            {
                case EndTurn _:
                    return 0;
                case MoveUnit m:
                {
                    int slot = SlotOf(slotToUnitId, m.UnitId);
                    if (slot < 0 || !L.CellIndex.TryGetValue(m.Dest, out int cell)) return -1;
                    return 1 + slot * n + cell;
                }
                case AttackUnit at:
                {
                    int slot = SlotOf(slotToUnitId, at.AttackerId);
                    var tc = CellOfEntity(s, at.TargetId);
                    if (slot < 0 || tc == null || !L.CellIndex.TryGetValue(tc.Value, out int cell)) return -1;
                    return 1 + r * n + slot * n + cell;
                }
                case DeployUnit d:
                {
                    if (d.TemplateIndex < 0 || d.TemplateIndex >= r) return -1; // only the seeded roster templates are codable
                    if (!L.CellIndex.TryGetValue(d.Cell, out int cell)) return -1;
                    return 1 + 2 * r * n + d.TemplateIndex * n + cell;
                }
                default:
                    return -1;
            }
        }

        private static int SlotOf(int[] slotToUnitId, int unitId)
        {
            for (int i = 0; i < slotToUnitId.Length; i++) if (slotToUnitId[i] == unitId) return i;
            return -1;
        }

        private static bool IsLivingUnit(GameState s, PlayerId seat, int unitId)
        {
            foreach (var u in s.Player(seat).UnitsOnBoard)
                if (u.Id == unitId && u.IsAlive) return true;
            return false;
        }

        private static int EnemyEntityAt(GameState s, PlayerId seat, HexCoord coord)
        {
            var foe = s.Player(Other(seat));
            foreach (var u in foe.UnitsOnBoard) if (u.IsAlive && u.Cell == coord) return u.Id;
            foreach (var g in foe.Generators) if (g.IsAlive && g.Cell == coord) return g.Id;
            return -1;
        }

        private static HexCoord? CellOfEntity(GameState s, int id)
        {
            foreach (var p in s.Players)
            {
                foreach (var u in p.UnitsOnBoard) if (u.Id == id && u.IsAlive) return u.Cell;
                foreach (var g in p.Generators) if (g.Id == id && g.IsAlive) return g.Cell;
            }
            return null;
        }
    }
}
