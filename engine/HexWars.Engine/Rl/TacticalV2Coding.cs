using System;

namespace HexWars.Engine.Rl
{
    /// <summary>
    /// Seat-relative observation, mask, and action codec for tactical-v2. Move/attack action indices
    /// address stable <see cref="TacticalV2UnitRegistry"/> slots (WHICH controllable unit); deploy
    /// indices address ordered <see cref="TacticalV2Layout"/> template indices (WHAT kind of unit to
    /// bring on). Observation planes follow the same split: HP-plane identity comes from each
    /// registry's recorded <see cref="TacticalV2UnitRegistry.TemplateIndexAt"/>, never from comparing
    /// <see cref="UnitStats"/> — two catalog templates with an identical stat line must still map to
    /// distinct planes. Stateless and parameterized by seat, so a single-agent env and a duel env share
    /// one source of truth for what a trained policy sees.
    /// </summary>
    public static class TacticalV2Coding
    {
        public const int Globals = 5; // myPoints, foePoints, round, myAliveFraction, foeAliveFraction

        private static PlayerId Other(PlayerId seat) => seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        // ---- observation ----

        public static float[] Observe(GameState state, PlayerId seat, TacticalV2Layout layout,
            TacticalV2UnitRegistry own, TacticalV2UnitRegistry foe)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            ValidateRegistry(own, layout, nameof(own));
            ValidateRegistry(foe, layout, nameof(foe));

            PlayerId foeSeat = Other(seat);
            int n = layout.CellCount;
            int t = layout.TemplateCount;
            var obs = new float[layout.ObservationLength];
            var board = state.Board;
            float maxElevation = Math.Max(1, layout.BoardGen.MaxElevation);

            int elevationPlane = 2 * t;
            for (int i = 0; i < n; i++)
            {
                if (!board.Contains(layout.Cells[i])) continue;
                obs[elevationPlane * n + i] = Clamp01(board.TileAt(layout.Cells[i]).Elevation / maxElevation);
            }

            WriteUnits(obs, state, seat, own, 0, n, layout);    // friendly units -> planes 0..T-1
            WriteUnits(obs, state, foeSeat, foe, t, n, layout); // enemy units    -> planes T..2T-1

            int g = layout.ObservationChannels * n;
            obs[g + 0] = Clamp01(state.Player(seat).Points / 50f);
            obs[g + 1] = Clamp01(state.Player(foeSeat).Points / 50f);
            obs[g + 2] = Clamp01(state.Round / (float)Math.Max(1, layout.Game.RoundCap));
            obs[g + 3] = Clamp01(AliveCount(state.Player(seat)) / (float)Math.Max(1, layout.UnitSlotCount));
            obs[g + 4] = Clamp01(AliveCount(state.Player(foeSeat)) / (float)Math.Max(1, layout.UnitSlotCount));
            return obs;
        }

        // Light up plane (planeBase + registered template index) at each tracked unit's cell with its
        // HP fraction. Slot-and-registry keyed, not stats-keyed, so a role never gets confused with
        // another that happens to share a stat line.
        private static void WriteUnits(float[] obs, GameState state, PlayerId seat, TacticalV2UnitRegistry registry,
            int planeBase, int n, TacticalV2Layout layout)
        {
            var player = state.Player(seat);
            for (int slot = 0; slot < registry.Capacity; slot++)
            {
                int unitId = registry.UnitIdAt(slot);
                if (unitId < 0) continue;
                int templateIndex = registry.TemplateIndexAt(slot);
                if (templateIndex < 0 || templateIndex >= layout.TemplateCount) continue;

                Unit? unit = FindLivingUnit(player, unitId);
                if (unit == null || !layout.CellIndex.TryGetValue(unit.Value.Cell, out int cell)) continue;

                obs[(planeBase + templateIndex) * n + cell] =
                    Clamp01(unit.Value.CurrentHp / (float)Math.Max(1, unit.Value.Stats.Health));
            }
        }

        private static int AliveCount(PlayerState player)
        {
            int count = 0;
            foreach (var unit in player.UnitsOnBoard) if (unit.IsAlive) count++;
            return count;
        }

        private static Unit? FindLivingUnit(PlayerState player, int unitId)
        {
            foreach (var unit in player.UnitsOnBoard)
                if (unit.Id == unitId && unit.IsAlive) return unit;
            return null;
        }

        private static float Clamp01(float value) => Math.Max(0f, Math.Min(1f, value));

        // ---- action mask / codec ----

        public static bool[] Mask(GameState state, PlayerId seat, TacticalV2Layout layout, TacticalV2UnitRegistry own)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            ValidateRegistry(own, layout, nameof(own));

            var mask = new bool[layout.ActionCount];
            mask[0] = true; // EndTurn always available
            foreach (var command in LegalMoves.For(state))
            {
                if (TryEncode(command, state, layout, own, out int index)) mask[index] = true;
            }

            // The engine itself has no live-unit cap — only points and board legality gate a
            // DeployUnit — so LegalMoves can hand back deploys the registry has no slot left to
            // track. Gate the whole deploy region on registry capacity here, or RegisterDeployment
            // throws once the command is applied.
            if (!own.HasFreeSlot)
                for (int i = layout.DeployOffset; i < layout.ActionCount; i++) mask[i] = false;

            return mask;
        }

        /// <summary>Encodes an actual currently legal command. Invalid, wrong-seat, stale, and unsupported
        /// commands fail closed rather than degrading to EndTurn.</summary>
        public static bool TryEncode(
            Command command,
            GameState state,
            TacticalV2Layout layout,
            TacticalV2UnitRegistry own,
            out int action)
        {
            if (command == null) throw new ArgumentNullException(nameof(command));
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            ValidateRegistry(own, layout, nameof(own));

            action = -1;
            if (command.Issuer != state.ActivePlayer) return false;
            if (command is DeployUnit && !own.HasFreeSlot) return false;

            bool legal = false;
            foreach (Command candidate in LegalMoves.For(state))
            {
                if (candidate.Equals(command))
                {
                    legal = true;
                    break;
                }
            }
            if (!legal) return false;

            int encoded = Encode(command, state, own, layout);
            if (encoded < 0 || encoded >= layout.ActionCount) return false;
            action = encoded;
            return true;
        }
        public static Command Decode(int action, GameState state, PlayerId seat, TacticalV2Layout layout,
            TacticalV2UnitRegistry own)
        {
            if (state == null) throw new ArgumentNullException(nameof(state));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            ValidateRegistry(own, layout, nameof(own));

            if (action < layout.MoveOffset) return new EndTurn(seat);

            int n = layout.CellCount;
            Command? decoded;
            if (action < layout.AttackOffset)
                decoded = DecodeUnitCell(action - layout.MoveOffset, n, layout, own, state, seat, isAttack: false);
            else if (action < layout.DeployOffset)
                decoded = DecodeUnitCell(action - layout.AttackOffset, n, layout, own, state, seat, isAttack: true);
            else if (action < layout.ActionCount)
                decoded = DecodeDeploy(action - layout.DeployOffset, n, layout, seat, own);
            else
                decoded = null;

            return decoded ?? new EndTurn(seat);
        }

        private static Command? DecodeUnitCell(int offset, int n, TacticalV2Layout layout, TacticalV2UnitRegistry own,
            GameState state, PlayerId seat, bool isAttack)
        {
            if (n <= 0) return null;
            int slot = offset / n, cell = offset % n;
            if (slot < 0 || slot >= own.Capacity || cell < 0 || cell >= n) return null;

            int unitId = own.UnitIdAt(slot);
            if (unitId < 0 || !IsLivingUnit(state, seat, unitId)) return null;
            HexCoord coord = layout.Cells[cell];

            if (!isAttack) return new MoveUnit(seat, unitId, coord);

            int targetId = EnemyEntityAt(state, seat, coord);
            return targetId < 0 ? null : new AttackUnit(seat, unitId, targetId);
        }

        private static Command? DecodeDeploy(int offset, int n, TacticalV2Layout layout, PlayerId seat,
            TacticalV2UnitRegistry own)
        {
            if (n <= 0) return null;
            // Same registry-capacity gate as Mask: a deploy address can be structurally valid (in-range
            // template + cell) yet have nowhere to land once every slot holds a living unit, so decoding
            // it must degrade to EndTurn rather than let RegisterDeployment throw downstream.
            if (!own.HasFreeSlot) return null;
            int templateIndex = offset / n, cell = offset % n;
            if (templateIndex < 0 || templateIndex >= layout.TemplateCount || cell < 0 || cell >= n) return null;
            return new DeployUnit(seat, templateIndex, layout.Cells[cell]);
        }

        private static int Encode(Command command, GameState state, TacticalV2UnitRegistry own, TacticalV2Layout layout)
        {
            int n = layout.CellCount;
            switch (command)
            {
                case EndTurn _:
                    return 0;
                case MoveUnit move:
                {
                    int slot = own.SlotOf(move.UnitId);
                    if (slot < 0 || !layout.CellIndex.TryGetValue(move.Dest, out int cell)) return -1;
                    return layout.MoveOffset + slot * n + cell;
                }
                case AttackUnit attack:
                {
                    int slot = own.SlotOf(attack.AttackerId);
                    HexCoord targetCell = CellOfEntity(state, attack.TargetId, out bool found);
                    if (slot < 0 || !found || !layout.CellIndex.TryGetValue(targetCell, out int cell)) return -1;
                    return layout.AttackOffset + slot * n + cell;
                }
                case DeployUnit deploy:
                {
                    if (deploy.TemplateIndex < 0 || deploy.TemplateIndex >= layout.TemplateCount) return -1;
                    if (!layout.CellIndex.TryGetValue(deploy.Cell, out int cell)) return -1;
                    return layout.DeployOffset + deploy.TemplateIndex * n + cell;
                }
                default:
                    return -1;
            }
        }

        private static bool IsLivingUnit(GameState state, PlayerId seat, int unitId)
        {
            foreach (var unit in state.Player(seat).UnitsOnBoard)
                if (unit.Id == unitId && unit.IsAlive) return true;
            return false;
        }

        private static int EnemyEntityAt(GameState state, PlayerId seat, HexCoord coord)
        {
            var foe = state.Player(Other(seat));
            foreach (var unit in foe.UnitsOnBoard) if (unit.IsAlive && unit.Cell == coord) return unit.Id;
            foreach (var generator in foe.Generators) if (generator.IsAlive && generator.Cell == coord) return generator.Id;
            return -1;
        }

        private static HexCoord CellOfEntity(GameState state, int id, out bool found)
        {
            foreach (var player in state.Players)
            {
                foreach (var unit in player.UnitsOnBoard)
                    if (unit.Id == id && unit.IsAlive) { found = true; return unit.Cell; }
                foreach (var generator in player.Generators)
                    if (generator.Id == id && generator.IsAlive) { found = true; return generator.Cell; }
            }
            found = false;
            return default;
        }

        private static void ValidateRegistry(TacticalV2UnitRegistry registry, TacticalV2Layout layout, string paramName)
        {
            if (registry == null) throw new ArgumentNullException(paramName);
            if (registry.Capacity != layout.UnitSlotCount)
                throw new ArgumentException(
                    $"registry capacity {registry.Capacity} must equal layout unit slot count {layout.UnitSlotCount}",
                    paramName);
        }
    }
}
