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
        public const int PerCell = 12; // elev, 4×terrain, myUnit, enemyUnit, myGen, enemyGen, hp, dmg, range
        public const int Globals = 5;  // myPoints, foePoints, round, myCount, foeCount

        private static PlayerId Other(PlayerId p) => p == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        // ---- observation ----

        public static float[] Observe(GameState s, PlayerId seat, TacticalLayout L)
        {
            var foe = Other(seat);
            var obs = new float[L.ObservationLength];
            var board = s.Board;
            float maxElev = Math.Max(1, L.BoardGen.MaxElevation);

            for (int i = 0; i < L.CellCount; i++)
            {
                if (!board.Contains(L.Cells[i])) continue;
                var t = board.TileAt(L.Cells[i]);
                int o = i * PerCell;
                obs[o] = Math.Min(1f, t.Elevation / maxElev);
                switch (t.Terrain)
                {
                    case TerrainType.Plains: obs[o + 1] = 1f; break;
                    case TerrainType.Forest: obs[o + 2] = 1f; break;
                    case TerrainType.Rough: obs[o + 3] = 1f; break;
                    case TerrainType.Water: obs[o + 4] = 1f; break;
                }
            }

            WriteEntities(obs, s.Player(seat), true, L);
            WriteEntities(obs, s.Player(foe), false, L);

            int g = L.CellCount * PerCell;
            obs[g + 0] = Math.Min(1f, s.Player(seat).Points / 50f);
            obs[g + 1] = Math.Min(1f, s.Player(foe).Points / 50f);
            obs[g + 2] = Math.Min(1f, s.Round / (float)Math.Max(1, L.Game.RoundCap));
            obs[g + 3] = Math.Min(1f, AliveUnits(s.Player(seat)) / (float)Math.Max(1, L.Roster));
            obs[g + 4] = Math.Min(1f, AliveUnits(s.Player(foe)) / (float)Math.Max(1, L.Roster));
            return obs;
        }

        private static void WriteEntities(float[] obs, PlayerState p, bool mine, TacticalLayout L)
        {
            foreach (var u in p.UnitsOnBoard)
            {
                if (!u.IsAlive || !L.CellIndex.TryGetValue(u.Cell, out int i)) continue;
                int o = i * PerCell;
                obs[o + (mine ? 5 : 6)] = 1f;
                obs[o + 9] = Math.Min(1f, u.CurrentHp / (float)Math.Max(1, u.Stats.Health));
                obs[o + 10] = Math.Min(1f, u.Stats.Damage / 10f);
                obs[o + 11] = Math.Min(1f, u.Stats.Range / 8f);
            }
            foreach (var gen in p.Generators)
            {
                if (!gen.IsAlive || !L.CellIndex.TryGetValue(gen.Cell, out int i)) continue;
                obs[i * PerCell + (mine ? 7 : 8)] = 1f;
            }
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
