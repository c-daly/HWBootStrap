using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    public enum AdaptivePhase
    {
        DeploymentRoot,
        DeploymentTemplate,
        DeploymentCell,
        DeploymentPlacedUnit,
        DeploymentMoveCell,
        GameplayRoot,
        GameplayUnit,
        GameplayUnitCommand,
        GameplayMoveCell,
        GameplayAttackCell,
        DesignSlot,
        DesignStat,
        DesignValue,
        DesignConfirm,
    }

    public enum AdaptiveCommandChoice
    {
        Cancel = 0,
        EndTurn = 1,
        ChooseUnit = 2,
        DeployReinforcement = 3,
        RedesignCustom = 4,
        ConfirmDesign = 5,
        DeployStartingUnit = 6,
        RepositionStartingUnit = 7,
        RemoveStartingUnit = 8,
        ConfirmDeployment = 9,
        Move = 10,
        Attack = 11,
    }

    /// <summary>Per-seat hierarchical decoder state. Deployment reposition/remove stores the selected
    /// root operation in PendingValue, so every fact that affects later decoding is present in the
    /// observation's four pending-value globals.</summary>
    public sealed class AdaptiveDecisionState
    {
        public PlayerId Seat { get; }
        public AdaptivePhase Phase { get; private set; }
        public int PendingUnitSlot { get; private set; } = -1;
        public int PendingTemplateSlot { get; private set; } = -1;
        public int PendingStat { get; private set; } = -1;
        public int PendingValue { get; private set; } = -1;
        public HashSet<HexCoord> SeenCells { get; } = new HashSet<HexCoord>();

        public AdaptiveDecisionState(PlayerId seat, AdaptivePhase phase = AdaptivePhase.DeploymentRoot)
        {
            Seat = seat;
            Phase = phase;
        }

        public void Enter(AdaptivePhase phase) => Phase = phase;
        public void SelectUnit(int slot) => PendingUnitSlot = slot;
        public void SelectTemplate(int slot) => PendingTemplateSlot = slot;
        public void SelectStat(int stat) => PendingStat = stat;
        public void SelectValue(int value) => PendingValue = value;

        public void Clear(AdaptivePhase root)
        {
            Phase = root;
            PendingUnitSlot = -1;
            PendingTemplateSlot = -1;
            PendingStat = -1;
            PendingValue = -1;
        }
    }

    public readonly struct AdaptiveTransition
    {
        public Command? Command { get; }
        public bool MutatedSetup { get; }
        public bool Intermediate { get; }
        public bool InvalidSequence { get; }

        public AdaptiveTransition(Command? command, bool mutatedSetup, bool intermediate, bool invalidSequence)
        {
            Command = command;
            MutatedSetup = mutatedSetup;
            Intermediate = intermediate;
            InvalidSequence = invalidSequence;
        }
    }
}
