using System;
using System.Collections.Generic;
using System.Linq;

namespace HexWars.Engine.Rl
{
    /// <summary>The sole adaptive-v1 observer, legal-mask builder, and hierarchical action decoder.</summary>
    public static class AdaptiveCoding
    {
        private static PlayerId Other(PlayerId seat) =>
            seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;

        public static float[] Observe(GameState? game, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            if (setup == null) throw new ArgumentNullException(nameof(setup));
            ValidateState(seat, decision, layout, slots);
            if (game != null) slots.Sync(game, seat);

            var board = game?.Board ?? setup.Board;
            int n = layout.CellCount;
            var observation = new float[layout.ObservationLength];
            float maxElevation = Math.Max(1, layout.BoardGen.MaxElevation);

            for (int cellIndex = 0; cellIndex < n; cellIndex++)
            {
                HexCoord cell = layout.Cells[cellIndex];
                if (!board.Contains(cell)) continue;
                var tile = board.TileAt(cell);
                observation[layout.ElevationPlane * n + cellIndex] =
                    Clamp01(tile.Elevation / maxElevation);
                observation[TerrainPlane(tile.Terrain, layout) * n + cellIndex] = 1f;
                if (board.IsInDeploymentZone(seat, cell))
                    observation[layout.DeploymentZonePlane * n + cellIndex] = 1f;
            }

            IReadOnlyList<UnitTemplate> ownTemplates;
            int ownPoints;
            int visibleFoePoints;
            int round;
            int ownLiving;
            int visibleFoes;
            int remainingBudget;
            int unplaced;

            if (game == null)
            {
                var view = setup.View(seat);
                ownTemplates = view.Templates;
                ownPoints = 0;
                visibleFoePoints = 0;
                round = 0;
                ownLiving = view.OwnPlacements.Count;
                visibleFoes = 0;
                remainingBudget = view.RemainingBudget;
                unplaced = Math.Max(0, view.RequiredUnits - view.OwnPlacements.Count);
                WriteDeploymentUnits(observation, view, layout);
            }
            else
            {
                ownTemplates = game.Player(seat).Barracks;
                ownPoints = game.Player(seat).Points;
                visibleFoePoints = game.Config.FogOfWar ? 0 : game.Player(Other(seat)).Points;
                round = game.Round;
                ownLiving = AliveCount(game.Player(seat));
                remainingBudget = 0;
                unplaced = 0;

                var currentlyVisible = CurrentVisibility(game, seat, layout);
                foreach (int cellIndex in currentlyVisible)
                    decision.SeenCells.Add(layout.Cells[cellIndex]);
                foreach (int cellIndex in currentlyVisible)
                    observation[layout.CurrentVisibilityPlane * n + cellIndex] = 1f;
                foreach (HexCoord seen in decision.SeenCells)
                    if (layout.CellIndex.TryGetValue(seen, out int cellIndex) && board.Contains(seen))
                        observation[layout.PreviouslySeenPlane * n + cellIndex] = 1f;

                WriteFriendlyUnits(observation, game, seat, layout, slots);
                visibleFoes = WriteVisibleEnemies(observation, game, seat, layout, currentlyVisible);
            }

            int g = layout.ObservationChannels * n;
            observation[g++] = Clamp01(ownPoints / 200f);
            observation[g++] = Clamp01(visibleFoePoints / 200f);
            observation[g++] = Clamp01(round / (float)Math.Max(1, layout.Game.RoundCap));
            observation[g++] = Clamp01(ownLiving / (float)Math.Max(1, layout.UnitCount));
            observation[g++] = Clamp01(visibleFoes / (float)Math.Max(1, layout.UnitCount));
            observation[g++] = Clamp01(remainingBudget / (float)Math.Max(1, layout.StartingArmyBudget));
            observation[g++] = Clamp01(unplaced / (float)Math.Max(1, layout.StartingUnitCount));

            for (int phase = 0; phase < AdaptiveLayout.PhaseTotal; phase++)
                observation[g++] = phase == (int)decision.Phase ? 1f : 0f;
            observation[g++] = NormalizeIndex(decision.PendingUnitSlot, layout.UnitCount);
            observation[g++] = NormalizeIndex(decision.PendingTemplateSlot, layout.TemplateCount);
            observation[g++] = NormalizeIndex(decision.PendingStat, AdaptiveLayout.StatTotal);
            observation[g++] = NormalizeIndex(decision.PendingValue, AdaptiveLayout.ValueTotal);

            for (int slot = 0; slot < layout.TemplateCount; slot++)
            {
                UnitStats stats = slot < ownTemplates.Count
                    ? ownTemplates[slot].Stats
                    : layout.Templates[slot].Stats;
                int[] statLine = StatLine(stats);
                for (int stat = 0; stat < statLine.Length; stat++)
                    observation[g++] = NormalizeStat(statLine[stat], (AdaptiveStat)stat, layout);
                observation[g++] = Clamp01(stats.PointCost / (float)Math.Max(1, layout.MaxDesignPointCost));
                observation[g++] = slot < layout.FixedTemplateCount ? 1f : 0f;
            }
            return observation;
        }

        public static bool[] Mask(GameState? game, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            if (setup == null) throw new ArgumentNullException(nameof(setup));
            ValidateState(seat, decision, layout, slots);
            if (game != null) slots.Sync(game, seat);
            var mask = new bool[layout.ActionCount];
            if (!IsRoot(decision.Phase)) mask[(int)AdaptiveCommandChoice.Cancel] = true;

            if (game == null && setup.Confirmed(seat)) return mask;
            if (game != null && (game.IsGameOver || game.ActivePlayer != seat)) return mask;

            IReadOnlyList<Command> legal = game == null
                ? Array.Empty<Command>()
                : SeatVisibleLegalMoves(game, seat);
            switch (decision.Phase)
            {
                case AdaptivePhase.DeploymentRoot:
                    MaskDeploymentRoot(mask, setup, seat);
                    break;
                case AdaptivePhase.DeploymentTemplate:
                    MaskTemplates(mask, game, setup, seat, decision, layout, slots, legal);
                    break;
                case AdaptivePhase.DeploymentCell:
                    MaskDeploymentCells(mask, game, setup, seat, decision, layout, legal);
                    break;
                case AdaptivePhase.DeploymentPlacedUnit:
                    MaskPlacedUnits(mask, setup, seat, decision, layout);
                    break;
                case AdaptivePhase.DeploymentMoveCell:
                    MaskDeploymentMoveCells(mask, setup, seat, decision, layout);
                    break;
                case AdaptivePhase.GameplayRoot:
                    MaskGameplayRoot(mask, game, seat, layout, slots, legal);
                    break;
                case AdaptivePhase.GameplayUnit:
                    MaskGameplayUnits(mask, seat, layout, slots, legal);
                    break;
                case AdaptivePhase.GameplayUnitCommand:
                    MaskUnitCommands(mask, seat, decision, slots, legal);
                    break;
                case AdaptivePhase.GameplayMoveCell:
                    MaskMoveCells(mask, seat, decision, layout, slots, legal);
                    break;
                case AdaptivePhase.GameplayAttackCell:
                    MaskAttackCells(mask, game, seat, decision, layout, slots, legal);
                    break;
                case AdaptivePhase.DesignSlot:
                    MaskDesignSlots(mask, game, seat, layout);
                    break;
                case AdaptivePhase.DesignStat:
                    if (ValidCustomSlot(game, seat, decision.PendingTemplateSlot, layout))
                        for (int stat = 0; stat < AdaptiveLayout.StatTotal; stat++)
                            mask[layout.StatOffset + stat] = true;
                    break;
                case AdaptivePhase.DesignValue:
                    MaskDesignValues(mask, game, seat, decision, layout);
                    break;
                case AdaptivePhase.DesignConfirm:
                    if (TryBuildReplacement(game, seat, decision, layout, out _))
                        mask[(int)AdaptiveCommandChoice.ConfirmDesign] = true;
                    break;
            }
            return mask;
        }

        public static AdaptiveTransition ApplyAction(int action, GameState? game, AdaptiveDeployment setup,
            PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            ValidateState(seat, decision, layout, slots);
            AdaptivePhase root = game == null ? AdaptivePhase.DeploymentRoot : AdaptivePhase.GameplayRoot;
            if (action < 0 || action >= layout.ActionCount)
                return Invalid(decision, root);

            bool[] mask = Mask(game, setup, seat, decision, layout, slots);
            if (!mask[action]) return Invalid(decision, root);
            if (action == (int)AdaptiveCommandChoice.Cancel && !IsRoot(decision.Phase))
            {
                decision.Clear(root);
                return new AdaptiveTransition(null, false, true, false);
            }

            switch (decision.Phase)
            {
                case AdaptivePhase.DeploymentRoot:
                    return ApplyDeploymentRoot(action, setup, seat, decision);
                case AdaptivePhase.DeploymentTemplate:
                    decision.SelectTemplate(action - layout.TemplateOffset);
                    decision.Enter(AdaptivePhase.DeploymentCell);
                    return Intermediate();
                case AdaptivePhase.DeploymentCell:
                    return ApplyDeploymentCell(action, game, setup, seat, decision, layout);
                case AdaptivePhase.DeploymentPlacedUnit:
                    return ApplyPlacedUnit(action, setup, seat, decision, layout);
                case AdaptivePhase.DeploymentMoveCell:
                    return ApplyDeploymentMoveCell(action, setup, seat, decision, layout);
                case AdaptivePhase.GameplayRoot:
                    return ApplyGameplayRoot(action, seat, decision);
                case AdaptivePhase.GameplayUnit:
                    decision.SelectUnit(action - layout.UnitOffset);
                    decision.Enter(AdaptivePhase.GameplayUnitCommand);
                    return Intermediate();
                case AdaptivePhase.GameplayUnitCommand:
                    decision.Enter(action == (int)AdaptiveCommandChoice.Move
                        ? AdaptivePhase.GameplayMoveCell
                        : AdaptivePhase.GameplayAttackCell);
                    return Intermediate();
                case AdaptivePhase.GameplayMoveCell:
                    return CompleteMove(action, game!, seat, decision, layout, slots);
                case AdaptivePhase.GameplayAttackCell:
                    return CompleteAttack(action, game!, seat, decision, layout, slots);
                case AdaptivePhase.DesignSlot:
                    decision.SelectTemplate(action - layout.TemplateOffset);
                    decision.Enter(AdaptivePhase.DesignStat);
                    return Intermediate();
                case AdaptivePhase.DesignStat:
                    decision.SelectStat(action - layout.StatOffset);
                    decision.Enter(AdaptivePhase.DesignValue);
                    return Intermediate();
                case AdaptivePhase.DesignValue:
                    var stat = (AdaptiveStat)decision.PendingStat;
                    decision.SelectValue(layout.StatValues[stat][action - layout.ValueOffset]);
                    decision.Enter(AdaptivePhase.DesignConfirm);
                    return Intermediate();
                case AdaptivePhase.DesignConfirm:
                    if (!TryBuildReplacement(game, seat, decision, layout, out var replacement))
                        return Invalid(decision, root);
                    decision.Clear(root);
                    return new AdaptiveTransition(replacement, false, false, false);
                default:
                    return Invalid(decision, root);
            }
        }

        private static void MaskDeploymentRoot(bool[] mask, AdaptiveDeployment setup, PlayerId seat)
        {
            var view = setup.View(seat);
            bool hasCell = LegalUnusedDeploymentCells(view).Any();
            bool canPlace = view.OwnPlacements.Count < view.RequiredUnits && hasCell
                && view.Templates.Any(t => t.Stats.PointCost <= view.RemainingBudget);
            mask[(int)AdaptiveCommandChoice.DeployStartingUnit] = canPlace;
            mask[(int)AdaptiveCommandChoice.RepositionStartingUnit] = view.OwnPlacements.Count > 0;
            mask[(int)AdaptiveCommandChoice.RemoveStartingUnit] = view.OwnPlacements.Count > 0;
            mask[(int)AdaptiveCommandChoice.ConfirmDeployment] = setup.CanConfirm(seat);
        }

        private static void MaskTemplates(bool[] mask, GameState? game, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots,
            IReadOnlyList<Command> legal)
        {
            if (game == null)
            {
                var view = setup.View(seat);
                if (!LegalUnusedDeploymentCells(view).Any()) return;
                for (int template = 0; template < Math.Min(layout.TemplateCount, view.Templates.Count); template++)
                    if (view.Templates[template].Stats.PointCost <= view.RemainingBudget)
                        mask[layout.TemplateOffset + template] = true;
                return;
            }
            if (!slots.HasFreeSlot) return;
            foreach (var deploy in legal.OfType<DeployUnit>().Where(x => x.Issuer == seat))
                if (deploy.TemplateIndex >= 0 && deploy.TemplateIndex < layout.TemplateCount)
                    mask[layout.TemplateOffset + deploy.TemplateIndex] = true;
        }

        private static void MaskDeploymentCells(bool[] mask, GameState? game, AdaptiveDeployment setup,
            PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout, IReadOnlyList<Command> legal)
        {
            int template = decision.PendingTemplateSlot;
            if (template < 0 || template >= layout.TemplateCount) return;
            if (game == null)
            {
                var view = setup.View(seat);
                if (template >= view.Templates.Count || view.Templates[template].Stats.PointCost > view.RemainingBudget)
                    return;
                foreach (HexCoord cell in LegalUnusedDeploymentCells(view))
                    if (layout.CellIndex.TryGetValue(cell, out int index)) mask[layout.CellOffset + index] = true;
                return;
            }
            foreach (var deploy in legal.OfType<DeployUnit>()
                         .Where(x => x.Issuer == seat && x.TemplateIndex == template))
                if (layout.CellIndex.TryGetValue(deploy.Cell, out int index))
                    mask[layout.CellOffset + index] = true;
        }

        private static void MaskPlacedUnits(bool[] mask, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            if (decision.PendingValue != (int)AdaptiveCommandChoice.RepositionStartingUnit
                && decision.PendingValue != (int)AdaptiveCommandChoice.RemoveStartingUnit) return;
            foreach (var placement in setup.View(seat).OwnPlacements)
                if (placement.Slot >= 0 && placement.Slot < layout.UnitCount)
                    mask[layout.UnitOffset + placement.Slot] = true;
        }

        private static void MaskDeploymentMoveCells(bool[] mask, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            if (decision.PendingValue != (int)AdaptiveCommandChoice.RepositionStartingUnit) return;
            var view = setup.View(seat);
            if (!view.OwnPlacements.Any(p => p.Slot == decision.PendingUnitSlot)) return;
            var occupied = new HashSet<HexCoord>(view.OwnPlacements
                .Where(p => p.Slot != decision.PendingUnitSlot).Select(p => p.Cell));
            foreach (HexCoord cell in view.Board.DeploymentZone(seat))
                if (view.Board.Contains(cell) && view.Game.Terrain(view.Board.TileAt(cell).Terrain).Passable
                    && !occupied.Contains(cell) && layout.CellIndex.TryGetValue(cell, out int index))
                    mask[layout.CellOffset + index] = true;
        }

        private static void MaskGameplayRoot(bool[] mask, GameState? game, PlayerId seat,
            AdaptiveLayout layout, AdaptiveUnitSlots slots, IReadOnlyList<Command> legal)
        {
            mask[(int)AdaptiveCommandChoice.EndTurn] = true;
            if (game == null) return;
            mask[(int)AdaptiveCommandChoice.ChooseUnit] = legal.Any(c =>
                c is MoveUnit m && m.Issuer == seat && slots.SlotOf(m.UnitId) >= 0
                || c is AttackUnit a && a.Issuer == seat && slots.SlotOf(a.AttackerId) >= 0);
            mask[(int)AdaptiveCommandChoice.DeployReinforcement] = slots.HasFreeSlot
                && legal.OfType<DeployUnit>().Any(d => d.Issuer == seat);
            mask[(int)AdaptiveCommandChoice.RedesignCustom] = game.Player(seat).Points >= game.Config.DesignFee
                && game.Player(seat).Barracks.Count > layout.FixedTemplateCount;
        }

        private static void MaskGameplayUnits(bool[] mask, PlayerId seat, AdaptiveLayout layout,
            AdaptiveUnitSlots slots, IReadOnlyList<Command> legal)
        {
            foreach (Command command in legal)
            {
                int id = command switch
                {
                    MoveUnit move when move.Issuer == seat => move.UnitId,
                    AttackUnit attack when attack.Issuer == seat => attack.AttackerId,
                    _ => -1,
                };
                int slot = slots.SlotOf(id);
                if (slot >= 0 && slot < layout.UnitCount) mask[layout.UnitOffset + slot] = true;
            }
        }

        private static void MaskUnitCommands(bool[] mask, PlayerId seat, AdaptiveDecisionState decision,
            AdaptiveUnitSlots slots, IReadOnlyList<Command> legal)
        {
            int unitId = slots.UnitIdAt(decision.PendingUnitSlot);
            if (unitId < 0) return;
            mask[(int)AdaptiveCommandChoice.Move] = legal.OfType<MoveUnit>()
                .Any(m => m.Issuer == seat && m.UnitId == unitId);
            mask[(int)AdaptiveCommandChoice.Attack] = legal.OfType<AttackUnit>()
                .Any(a => a.Issuer == seat && a.AttackerId == unitId);
        }

        private static void MaskMoveCells(bool[] mask, PlayerId seat, AdaptiveDecisionState decision,
            AdaptiveLayout layout, AdaptiveUnitSlots slots, IReadOnlyList<Command> legal)
        {
            int unitId = slots.UnitIdAt(decision.PendingUnitSlot);
            foreach (var move in legal.OfType<MoveUnit>().Where(m => m.Issuer == seat && m.UnitId == unitId))
                if (layout.CellIndex.TryGetValue(move.Dest, out int index)) mask[layout.CellOffset + index] = true;
        }

        private static void MaskAttackCells(bool[] mask, GameState? game, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots,
            IReadOnlyList<Command> legal)
        {
            if (game == null) return;
            int unitId = slots.UnitIdAt(decision.PendingUnitSlot);
            foreach (var attack in legal.OfType<AttackUnit>()
                         .Where(a => a.Issuer == seat && a.AttackerId == unitId))
            {
                Unit? target = FindLivingUnit(game.Player(Other(seat)), attack.TargetId);
                if (target.HasValue && layout.CellIndex.TryGetValue(target.Value.Cell, out int index))
                    mask[layout.CellOffset + index] = true;
            }
        }

        private static void MaskDesignSlots(bool[] mask, GameState? game, PlayerId seat, AdaptiveLayout layout)
        {
            if (game == null || game.Player(seat).Points < game.Config.DesignFee) return;
            int end = Math.Min(layout.TemplateCount, game.Player(seat).Barracks.Count);
            for (int slot = layout.FixedTemplateCount; slot < end; slot++)
                mask[layout.TemplateOffset + slot] = true;
        }

        private static void MaskDesignValues(bool[] mask, GameState? game, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            if (!ValidCustomSlot(game, seat, decision.PendingTemplateSlot, layout)
                || decision.PendingStat < 0 || decision.PendingStat >= AdaptiveLayout.StatTotal) return;
            var stat = (AdaptiveStat)decision.PendingStat;
            if (!layout.StatValues.TryGetValue(stat, out var values)) return;
            int count = Math.Min(values.Count, AdaptiveLayout.ValueTotal);
            for (int valueIndex = 0; valueIndex < count; valueIndex++)
            {
                var candidate = ReplaceStat(game!.Player(seat).Barracks[decision.PendingTemplateSlot].Stats,
                    stat, values[valueIndex]);
                if (candidate.PointCost <= layout.MaxDesignPointCost)
                    mask[layout.ValueOffset + valueIndex] = true;
            }
        }

        private static AdaptiveTransition ApplyDeploymentRoot(int action, AdaptiveDeployment setup,
            PlayerId seat, AdaptiveDecisionState decision)
        {
            switch ((AdaptiveCommandChoice)action)
            {
                case AdaptiveCommandChoice.DeployStartingUnit:
                    decision.Enter(AdaptivePhase.DeploymentTemplate);
                    return Intermediate();
                case AdaptiveCommandChoice.RepositionStartingUnit:
                case AdaptiveCommandChoice.RemoveStartingUnit:
                    decision.SelectValue(action);
                    decision.Enter(AdaptivePhase.DeploymentPlacedUnit);
                    return Intermediate();
                case AdaptiveCommandChoice.ConfirmDeployment:
                    bool confirmed = setup.TryConfirm(seat);
                    decision.Clear(AdaptivePhase.DeploymentRoot);
                    return confirmed
                        ? new AdaptiveTransition(null, true, true, false)
                        : new AdaptiveTransition(null, false, false, true);
                default:
                    return Invalid(decision, AdaptivePhase.DeploymentRoot);
            }
        }

        private static AdaptiveTransition ApplyDeploymentCell(int action, GameState? game,
            AdaptiveDeployment setup, PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            HexCoord cell = layout.Cells[action - layout.CellOffset];
            int template = decision.PendingTemplateSlot;
            if (game != null)
            {
                decision.Clear(AdaptivePhase.GameplayRoot);
                return new AdaptiveTransition(new DeployUnit(seat, template, cell), false, false, false);
            }
            bool placed = setup.TryPlace(seat, template, cell);
            decision.Clear(AdaptivePhase.DeploymentRoot);
            return placed
                ? new AdaptiveTransition(null, true, true, false)
                : new AdaptiveTransition(null, false, false, true);
        }

        private static AdaptiveTransition ApplyPlacedUnit(int action, AdaptiveDeployment setup, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            int slot = action - layout.UnitOffset;
            if (decision.PendingValue == (int)AdaptiveCommandChoice.RemoveStartingUnit)
            {
                bool removed = setup.TryRemove(seat, slot);
                decision.Clear(AdaptivePhase.DeploymentRoot);
                return removed
                    ? new AdaptiveTransition(null, true, true, false)
                    : new AdaptiveTransition(null, false, false, true);
            }
            decision.SelectUnit(slot);
            decision.Enter(AdaptivePhase.DeploymentMoveCell);
            return Intermediate();
        }

        private static AdaptiveTransition ApplyDeploymentMoveCell(int action, AdaptiveDeployment setup,
            PlayerId seat, AdaptiveDecisionState decision, AdaptiveLayout layout)
        {
            bool moved = setup.TryMove(seat, decision.PendingUnitSlot,
                layout.Cells[action - layout.CellOffset]);
            decision.Clear(AdaptivePhase.DeploymentRoot);
            return moved
                ? new AdaptiveTransition(null, true, true, false)
                : new AdaptiveTransition(null, false, false, true);
        }

        private static AdaptiveTransition ApplyGameplayRoot(int action, PlayerId seat,
            AdaptiveDecisionState decision)
        {
            switch ((AdaptiveCommandChoice)action)
            {
                case AdaptiveCommandChoice.EndTurn:
                    decision.Clear(AdaptivePhase.GameplayRoot);
                    return new AdaptiveTransition(new EndTurn(seat), false, false, false);
                case AdaptiveCommandChoice.ChooseUnit:
                    decision.Enter(AdaptivePhase.GameplayUnit);
                    return Intermediate();
                case AdaptiveCommandChoice.DeployReinforcement:
                    decision.Enter(AdaptivePhase.DeploymentTemplate);
                    return Intermediate();
                case AdaptiveCommandChoice.RedesignCustom:
                    decision.Enter(AdaptivePhase.DesignSlot);
                    return Intermediate();
                default:
                    return Invalid(decision, AdaptivePhase.GameplayRoot);
            }
        }

        private static AdaptiveTransition CompleteMove(int action, GameState game, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            int unitId = slots.UnitIdAt(decision.PendingUnitSlot);
            HexCoord cell = layout.Cells[action - layout.CellOffset];
            var command = SeatVisibleLegalMoves(game, seat).OfType<MoveUnit>()
                .FirstOrDefault(m => m.Issuer == seat && m.UnitId == unitId && m.Dest == cell);
            decision.Clear(AdaptivePhase.GameplayRoot);
            return command == null
                ? new AdaptiveTransition(null, false, false, true)
                : new AdaptiveTransition(command, false, false, false);
        }

        private static AdaptiveTransition CompleteAttack(int action, GameState game, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            int unitId = slots.UnitIdAt(decision.PendingUnitSlot);
            HexCoord cell = layout.Cells[action - layout.CellOffset];
            var command = SeatVisibleLegalMoves(game, seat).OfType<AttackUnit>()
                .FirstOrDefault(a => a.Issuer == seat && a.AttackerId == unitId
                    && FindLivingUnit(game.Player(Other(seat)), a.TargetId)?.Cell == cell);
            decision.Clear(AdaptivePhase.GameplayRoot);
            return command == null
                ? new AdaptiveTransition(null, false, false, true)
                : new AdaptiveTransition(command, false, false, false);
        }

        private static bool TryBuildReplacement(GameState? game, PlayerId seat,
            AdaptiveDecisionState decision, AdaptiveLayout layout, out ReplaceTemplate replacement)
        {
            replacement = null!;
            if (!ValidCustomSlot(game, seat, decision.PendingTemplateSlot, layout)
                || decision.PendingStat < 0 || decision.PendingStat >= AdaptiveLayout.StatTotal
                || game!.Player(seat).Points < game.Config.DesignFee) return false;
            var stat = (AdaptiveStat)decision.PendingStat;
            if (!layout.StatValues.TryGetValue(stat, out var values)
                || !values.Contains(decision.PendingValue)) return false;
            var old = game.Player(seat).Barracks[decision.PendingTemplateSlot];
            UnitStats stats = ReplaceStat(old.Stats, stat, decision.PendingValue);
            if (stats.PointCost > layout.MaxDesignPointCost) return false;
            replacement = new ReplaceTemplate(seat, decision.PendingTemplateSlot, stats, old.Name);
            return true;
        }

        private static bool ValidCustomSlot(GameState? game, PlayerId seat, int slot, AdaptiveLayout layout) =>
            game != null && slot >= layout.FixedTemplateCount && slot < layout.TemplateCount
            && slot < game.Player(seat).Barracks.Count;

        private static UnitStats ReplaceStat(UnitStats old, AdaptiveStat stat, int value) => new UnitStats(
            stat == AdaptiveStat.Health ? value : old.Health,
            stat == AdaptiveStat.Damage ? value : old.Damage,
            stat == AdaptiveStat.Defense ? value : old.Defense,
            stat == AdaptiveStat.Movement ? value : old.Movement,
            stat == AdaptiveStat.VerticalMovement ? value : old.VerticalMovement,
            stat == AdaptiveStat.Range ? value : old.Range,
            stat == AdaptiveStat.RangeArc ? value : old.RangeArc,
            stat == AdaptiveStat.Vision ? value : old.Vision,
            stat == AdaptiveStat.VisionArc ? value : old.VisionArc);

        private static IEnumerable<HexCoord> LegalUnusedDeploymentCells(AdaptiveDeploymentView view)
        {
            var occupied = new HashSet<HexCoord>(view.OwnPlacements.Select(p => p.Cell));
            return view.Board.DeploymentZone(view.Seat)
                .Where(cell => view.Board.Contains(cell)
                    && view.Game.Terrain(view.Board.TileAt(cell).Terrain).Passable
                    && !occupied.Contains(cell));
        }

        private static HashSet<int> CurrentVisibility(GameState game, PlayerId seat, AdaptiveLayout layout)
        {
            var visible = new HashSet<int>();
            for (int i = 0; i < layout.CellCount; i++)
            {
                HexCoord cell = layout.Cells[i];
                if (!game.Board.Contains(cell)) continue;
                if (!game.Config.FogOfWar || TargetingService.IsVisibleToArmy(
                        game, seat, cell, game.Board.TileAt(cell).Elevation))
                    visible.Add(i);
            }
            return visible;
        }

        private static void WriteDeploymentUnits(float[] observation, AdaptiveDeploymentView view,
            AdaptiveLayout layout)
        {
            int n = layout.CellCount;
            foreach (var placement in view.OwnPlacements)
            {
                if (!layout.CellIndex.TryGetValue(placement.Cell, out int cell)
                    || placement.TemplateIndex < 0 || placement.TemplateIndex >= layout.TemplateCount) continue;
                observation[layout.FriendlyUnitPlane(placement.TemplateIndex) * n + cell] = 1f;
                if (placement.Slot >= 0 && placement.Slot < layout.UnitCount)
                    observation[layout.FriendlySlotPlane(placement.Slot) * n + cell] = 1f;
            }
        }

        private static void WriteFriendlyUnits(float[] observation, GameState game, PlayerId seat,
            AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            int n = layout.CellCount;
            var player = game.Player(seat);
            foreach (Unit unit in player.UnitsOnBoard)
            {
                if (!unit.IsAlive || !layout.CellIndex.TryGetValue(unit.Cell, out int cell)) continue;
                int role = RoleOf(unit, player.Barracks, layout);
                if (role >= 0)
                    observation[layout.FriendlyUnitPlane(role) * n + cell] =
                        Clamp01(unit.CurrentHp / (float)Math.Max(1, unit.Stats.Health));
                int slot = slots.SlotOf(unit.Id);
                if (slot >= 0 && slot < layout.UnitCount)
                    observation[layout.FriendlySlotPlane(slot) * n + cell] = 1f;
            }
        }

        private static int WriteVisibleEnemies(float[] observation, GameState game, PlayerId seat,
            AdaptiveLayout layout, ISet<int> visible)
        {
            int count = 0;
            int n = layout.CellCount;
            var enemy = game.Player(Other(seat));
            foreach (Unit unit in enemy.UnitsOnBoard)
            {
                if (!unit.IsAlive || !layout.CellIndex.TryGetValue(unit.Cell, out int cell)
                    || !visible.Contains(cell)) continue;
                count++;
                int role = RoleOf(unit, enemy.Barracks, layout);
                if (role >= 0)
                    observation[layout.EnemyUnitPlane(role) * n + cell] =
                        Clamp01(unit.CurrentHp / (float)Math.Max(1, unit.Stats.Health));
            }
            return count;
        }

        private static int RoleOf(Unit unit, IReadOnlyList<UnitTemplate> templates, AdaptiveLayout layout)
        {
            for (int role = 0; role < Math.Min(layout.TemplateCount, templates.Count); role++)
                if (!string.IsNullOrEmpty(unit.Name)
                    && string.Equals(unit.Name, templates[role].Name, StringComparison.Ordinal)) return role;
            for (int role = 0; role < Math.Min(layout.TemplateCount, templates.Count); role++)
                if (SameStats(unit.Stats, templates[role].Stats)) return role;
            return -1;
        }

        private static bool SameStats(UnitStats a, UnitStats b) =>
            a.Health == b.Health && a.Damage == b.Damage && a.Defense == b.Defense
            && a.Movement == b.Movement && a.VerticalMovement == b.VerticalMovement
            && a.Range == b.Range && a.RangeArc == b.RangeArc
            && a.Vision == b.Vision && a.VisionArc == b.VisionArc;

        private static int AliveCount(PlayerState player) => player.UnitsOnBoard.Count(u => u.IsAlive);

        private static Unit? FindLivingUnit(PlayerState player, int unitId)
        {
            foreach (Unit unit in player.UnitsOnBoard)
                if (unit.Id == unitId && unit.IsAlive) return unit;
            return null;
        }

        /// <summary>Derives legality from the engine's LegalMoves over the requesting seat's visible
        /// projection. Hidden enemy occupancy is removed, so masks cannot act as a concealed-unit sensor.
        /// The authoritative engine remains final validator and may reject a move/deploy that bumps into
        /// a hidden blocker.</summary>
        private static IReadOnlyList<Command> SeatVisibleLegalMoves(GameState game, PlayerId seat)
        {
            if (!game.Config.FogOfWar) return LegalMoves.For(game);
            PlayerId foe = Other(seat);
            var players = game.Players.ToArray();
            var enemy = game.Player(foe);
            var visibleUnits = enemy.UnitsOnBoard.Where(unit => unit.IsAlive
                && TargetingService.IsVisibleToArmy(game, seat, unit.Cell, unit.Elevation)).ToArray();
            var visibleGenerators = enemy.Generators.Where(generator => generator.IsAlive
                && TargetingService.IsVisibleToArmy(game, seat, generator.Cell, generator.Elevation)).ToArray();
            players[(int)foe] = new PlayerState(foe, enemy.Points, enemy.Barracks,
                visibleUnits, visibleGenerators, enemy.DestroyedValue);
            var projection = new GameState(game.Board, game.Config, players, game.ActivePlayer,
                game.Round, game.NextEntityId, game.IsGameOver, game.Winner, game.MovedUnitIds,
                game.AttackedUnitIds, game.MovementSpent);
            return LegalMoves.For(projection);
        }

        private static int TerrainPlane(TerrainType terrain, AdaptiveLayout layout) => terrain switch
        {
            TerrainType.Plains => layout.PlainsPlane,
            TerrainType.Forest => layout.ForestPlane,
            TerrainType.Rough => layout.RoughPlane,
            TerrainType.Water => layout.WaterPlane,
            _ => layout.PlainsPlane,
        };

        private static int[] StatLine(UnitStats stats) => new[]
        {
            stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
            stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
        };

        private static float NormalizeStat(int value, AdaptiveStat stat, AdaptiveLayout layout)
        {
            var values = layout.StatValues[stat];
            int max = values.Count == 0 ? 1 : Math.Max(1, values.Max());
            return Clamp01(value / (float)max);
        }

        private static float NormalizeIndex(int value, int count) =>
            value < 0 ? 0f : Clamp01((value + 1) / (float)Math.Max(1, count));

        private static float Clamp01(float value) => Math.Max(0f, Math.Min(1f, value));
        private static bool IsRoot(AdaptivePhase phase) =>
            phase == AdaptivePhase.DeploymentRoot || phase == AdaptivePhase.GameplayRoot;

        private static void ValidateState(PlayerId seat, AdaptiveDecisionState decision,
            AdaptiveLayout layout, AdaptiveUnitSlots slots)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            if (layout == null) throw new ArgumentNullException(nameof(layout));
            if (slots == null) throw new ArgumentNullException(nameof(slots));
            if (decision.Seat != seat) throw new ArgumentException("decision state belongs to another seat", nameof(decision));
            if (slots.Capacity != layout.UnitCount)
                throw new ArgumentException($"slot capacity must equal adaptive layout unit count {layout.UnitCount}",
                    nameof(slots));
        }

        private static AdaptiveTransition Intermediate() => new AdaptiveTransition(null, false, true, false);

        private static AdaptiveTransition Invalid(AdaptiveDecisionState decision, AdaptivePhase root)
        {
            decision.Clear(root);
            return new AdaptiveTransition(null, false, false, true);
        }
    }
}
