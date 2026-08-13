using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    public enum TacticalV3CapabilityKind
    {
        Health = 0,
        Damage = 1,
        Defense = 2,
        Movement = 3,
        VerticalMovement = 4,
        Range = 5,
        RangeArc = 6,
        Vision = 7,
        VisionArc = 8,
    }

    public enum TacticalV3ActionKind
    {
        Move = 0,
        Attack = 1,
        Deploy = 2,
        EndTurn = 3,
    }

    public enum TacticalV3CapabilityRelationKind
    {
        Opposes = 0,
        Reduces = 1,
        EnablesAction = 2,
    }

    public sealed class TacticalV3CapabilityDefinition
    {
        public TacticalV3CapabilityDefinition(TacticalV3CapabilityKind kind) => Kind = kind;

        public TacticalV3CapabilityKind Kind { get; }
    }

    public readonly struct TacticalV3RelationTarget
    {
        private readonly TacticalV3CapabilityKind? _capability;
        private readonly TacticalV3ActionKind? _action;

        private TacticalV3RelationTarget(
            TacticalV3CapabilityKind? capability,
            TacticalV3ActionKind? action)
        {
            _capability = capability;
            _action = action;
        }

        public static implicit operator TacticalV3RelationTarget(TacticalV3CapabilityKind capability) =>
            new TacticalV3RelationTarget(capability, null);

        public static implicit operator TacticalV3RelationTarget(TacticalV3ActionKind action) =>
            new TacticalV3RelationTarget(null, action);

        public static bool operator ==(
            TacticalV3RelationTarget target,
            TacticalV3CapabilityKind capability) =>
            target._capability == capability;

        public static bool operator !=(
            TacticalV3RelationTarget target,
            TacticalV3CapabilityKind capability) =>
            !(target == capability);

        public static bool operator ==(
            TacticalV3RelationTarget target,
            TacticalV3ActionKind action) =>
            target._action == action;

        public static bool operator !=(
            TacticalV3RelationTarget target,
            TacticalV3ActionKind action) =>
            !(target == action);

        public override bool Equals(object? obj) =>
            obj is TacticalV3RelationTarget other &&
            _capability == other._capability &&
            _action == other._action;

        public override int GetHashCode() =>
            ((_capability.HasValue ? (int)_capability.Value + 1 : 0) * 397) ^
            (_action.HasValue ? (int)_action.Value + 1 : 0);
    }

    public sealed class TacticalV3CapabilityRelation
    {
        public TacticalV3CapabilityRelation(
            TacticalV3CapabilityKind source,
            TacticalV3CapabilityRelationKind kind,
            TacticalV3RelationTarget target)
        {
            if (kind == TacticalV3CapabilityRelationKind.EnablesAction &&
                !(target == TacticalV3ActionKind.Attack))
            {
                throw new ArgumentException("enables-action relations must target an action", nameof(target));
            }
            if (kind != TacticalV3CapabilityRelationKind.EnablesAction &&
                !(target == TacticalV3CapabilityKind.Health) &&
                !(target == TacticalV3CapabilityKind.Damage) &&
                !(target == TacticalV3CapabilityKind.Defense) &&
                !(target == TacticalV3CapabilityKind.Movement) &&
                !(target == TacticalV3CapabilityKind.VerticalMovement) &&
                !(target == TacticalV3CapabilityKind.Range) &&
                !(target == TacticalV3CapabilityKind.RangeArc) &&
                !(target == TacticalV3CapabilityKind.Vision) &&
                !(target == TacticalV3CapabilityKind.VisionArc))
            {
                throw new ArgumentException("capability relations must target a capability", nameof(target));
            }

            Source = source;
            Kind = kind;
            Target = target;
        }

        public TacticalV3CapabilityKind Source { get; }
        public TacticalV3CapabilityRelationKind Kind { get; }
        public TacticalV3RelationTarget Target { get; }
    }

    public static class TacticalV3Capabilities
    {
        private static readonly IReadOnlyList<TacticalV3CapabilityDefinition> Definitions =
            Array.AsReadOnly(new[]
            {
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Health),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Damage),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Defense),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Movement),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.VerticalMovement),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Range),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.RangeArc),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.Vision),
                new TacticalV3CapabilityDefinition(TacticalV3CapabilityKind.VisionArc),
            });

        private static readonly IReadOnlyList<TacticalV3CapabilityRelation> SemanticRelations =
            Array.AsReadOnly(new[]
            {
                new TacticalV3CapabilityRelation(
                    TacticalV3CapabilityKind.Damage,
                    TacticalV3CapabilityRelationKind.Opposes,
                    TacticalV3CapabilityKind.Health),
                new TacticalV3CapabilityRelation(
                    TacticalV3CapabilityKind.Defense,
                    TacticalV3CapabilityRelationKind.Reduces,
                    TacticalV3CapabilityKind.Damage),
                new TacticalV3CapabilityRelation(
                    TacticalV3CapabilityKind.Range,
                    TacticalV3CapabilityRelationKind.EnablesAction,
                    TacticalV3ActionKind.Attack),
                new TacticalV3CapabilityRelation(
                    TacticalV3CapabilityKind.RangeArc,
                    TacticalV3CapabilityRelationKind.EnablesAction,
                    TacticalV3ActionKind.Attack),
            });

        public static IReadOnlyList<TacticalV3CapabilityDefinition> All => Definitions;
        public static IReadOnlyList<TacticalV3CapabilityRelation> Relations => SemanticRelations;
    }
}
