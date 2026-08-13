using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;

namespace HexWars.Presentation
{
    [Serializable] public sealed class TacticalV3PolicyRequestDto
    {
        public int seat;
        public TacticalV3ViewDto decision;
    }

    [Serializable] public sealed class TacticalV3ViewDto
    {
        public long decision_id;
        public int seat;
        public TacticalV3ObservationDto observation;
        public TacticalV3CandidateDto[] candidates;
        public TacticalV3RewardDto reward;
        public int winner;
        public bool terminated;
        public bool truncated;
        public string start_profile;
        public int reference_seat;
    }

    [Serializable] public sealed class TacticalV3ObservationDto
    {
        public TacticalV3CellDto[] cells;
        public TacticalV3UnitDto[] units;
        public TacticalV3TemplateDto[] templates;
        public TacticalV3CapabilityDefinitionDto[] capability_definitions;
        public TacticalV3CapabilityAllocationDto[] capability_allocations;
        public TacticalV3RuleDto[] rules;
        public TacticalV3MemoryDto[] memory;
        public TacticalV3RelationDto[] relations;
    }

    [Serializable] public sealed class TacticalV3CellDto
    {
        public int q;
        public int r;
        public string terrain;
        public int elevation;
        public bool self_deployment_zone;
        public bool opponent_deployment_zone;
        public string controller;
        public bool is_boundary;
        public bool currently_visible;
        public bool previously_observed;
    }

    [Serializable] public sealed class TacticalV3UnitDto
    {
        public string owner;
        public int current_hp;
        public int max_hp;
        public TacticalV3TokenReferenceDto cell;
        public int elevation;
        public bool moved;
        public bool attacked;
        public int horizontal_movement_spent;
        public int vertical_movement_spent;
        public int point_cost;
        public int deploy_cost;
        public bool currently_visible;
    }

    [Serializable] public sealed class TacticalV3TemplateDto
    {
        public string owner;
        public int point_cost;
        public int deploy_cost;
        public bool is_fixed;
        public bool is_deployable;
    }

    [Serializable] public sealed class TacticalV3CapabilityDefinitionDto
    {
        public string kind;
    }

    [Serializable] public sealed class TacticalV3CapabilityAllocationDto
    {
        public TacticalV3TokenReferenceDto owner;
        public TacticalV3TokenReferenceDto definition;
        public string capability;
        public int purchased_level;
        public int effective_value;
    }

    [Serializable] public sealed class TacticalV3RuleDto
    {
        public string kind;
        public int int_value;
        public float float_value;
        public bool bool_value;
    }

    [Serializable] public sealed class TacticalV3MemoryDto
    {
        public TacticalV3TokenReferenceDto cell;
        public int last_seen_round;
        public int observation_age;
        public int last_known_current_hp;
        public bool currently_visible;
    }

    [Serializable] public sealed class TacticalV3RelationDto
    {
        public string kind;
        public TacticalV3TokenReferenceDto source;
        public TacticalV3TokenReferenceDto target;
        public int int_feature;
        public float float_feature;
        public bool bool_feature;
    }

    [Serializable] public sealed class TacticalV3CandidateDto
    {
        public int candidate_id;
        public long decision_id;
        public string kind;
        public TacticalV3TokenReferenceDto actor;
        public TacticalV3TokenReferenceDto target;
        public TacticalV3TokenReferenceDto template;
        public TacticalV3TokenReferenceDto cell;
        public TacticalV3ProjectedDeltaDto projection;
    }

    [Serializable] public sealed class TacticalV3ProjectedDeltaDto
    {
        public TacticalV3TokenReferenceDto source_cell;
        public TacticalV3TokenReferenceDto destination_cell;
        public TacticalV3TokenReferenceDto template;
        public TacticalV3TokenReferenceDto target;
        public int horizontal_movement_spent;
        public int vertical_movement_spent;
        public int target_hp_delta;
        public int damage;
        public bool is_lethal;
        public int bounty_delta;
        public int points_delta;
        public int round_delta;
        public bool is_terminal;
    }

    [Serializable] public sealed class TacticalV3TokenReferenceDto
    {
        public string table;
        public int row;
    }

    [Serializable] public sealed class TacticalV3RewardDto
    {
        public float terminal_outcome;
        public float known_health_adjusted_material_progress;
        public float public_resource_progress;
        public float time_pressure;
        public float total;
        public bool finalized;
    }

    public static class TacticalV3PolicyPayload
    {
        public static TacticalV3ViewDto From(TacticalV3View view)
        {
            if (view == null) throw new ArgumentNullException(nameof(view));
            TacticalV3DecisionFrame decision = view.Decision ??
                throw new InvalidOperationException("tactical-v3 decision is required");
            if (view.Seat != decision.Seat)
                throw new InvalidOperationException(
                    "tactical-v3 view seat does not match decision seat");
            TacticalV3Observation observation = decision.Observation ??
                throw new InvalidOperationException("tactical-v3 observation is required");
            Validate(observation, decision.Candidates, decision.DecisionId, view.Reward);

            return new TacticalV3ViewDto
            {
                decision_id = decision.DecisionId,
                seat = (int)view.Seat,
                observation = Observation(observation),
                candidates = decision.Candidates.Select(Candidate).ToArray(),
                reward = Reward(view.Reward),
                winner = view.Winner,
                terminated = view.Terminated,
                truncated = view.Truncated,
                start_profile = view.StartProfileId,
                reference_seat = (int)view.ReferenceSeat,
            };
        }

        static TacticalV3ObservationDto Observation(TacticalV3Observation value) =>
            new TacticalV3ObservationDto
            {
                cells = value.Cells.Select(cell => new TacticalV3CellDto
                {
                    q = cell.Q,
                    r = cell.R,
                    terrain = Terrain(cell.Terrain),
                    elevation = cell.Elevation,
                    self_deployment_zone = cell.SelfDeploymentZone,
                    opponent_deployment_zone = cell.OpponentDeploymentZone,
                    controller = cell.Controller.HasValue
                        ? Owner(cell.Controller.Value)
                        : null,
                    is_boundary = cell.IsBoundary,
                    currently_visible = cell.CurrentlyVisible,
                    previously_observed = cell.PreviouslyObserved,
                }).ToArray(),
                units = value.Units.Select(unit => new TacticalV3UnitDto
                {
                    owner = Owner(unit.Owner),
                    current_hp = unit.CurrentHp,
                    max_hp = unit.MaxHp,
                    cell = Reference(unit.Cell),
                    elevation = unit.Elevation,
                    moved = unit.Moved,
                    attacked = unit.Attacked,
                    horizontal_movement_spent = unit.HorizontalMovementSpent,
                    vertical_movement_spent = unit.VerticalMovementSpent,
                    point_cost = unit.PointCost,
                    deploy_cost = unit.DeployCost,
                    currently_visible = unit.CurrentlyVisible,
                }).ToArray(),
                templates = value.Templates.Select(template => new TacticalV3TemplateDto
                {
                    owner = Owner(template.Owner),
                    point_cost = template.PointCost,
                    deploy_cost = template.DeployCost,
                    is_fixed = template.IsFixed,
                    is_deployable = template.IsDeployable,
                }).ToArray(),
                capability_definitions = value.CapabilityDefinitions.Select(definition =>
                    new TacticalV3CapabilityDefinitionDto
                    {
                        kind = Capability(definition.Kind),
                    }).ToArray(),
                capability_allocations = value.CapabilityAllocations.Select(allocation =>
                    new TacticalV3CapabilityAllocationDto
                    {
                        owner = Reference(allocation.Owner),
                        definition = Reference(allocation.Definition),
                        capability = Capability(allocation.Capability),
                        purchased_level = allocation.PurchasedLevel,
                        effective_value = allocation.EffectiveValue,
                    }).ToArray(),
                rules = value.Rules.Select(rule => new TacticalV3RuleDto
                {
                    kind = Rule(rule.Kind),
                    int_value = rule.IntValue,
                    float_value = rule.FloatValue,
                    bool_value = rule.BoolValue,
                }).ToArray(),
                memory = value.Memory.Select(memory => new TacticalV3MemoryDto
                {
                    cell = Reference(memory.Cell),
                    last_seen_round = memory.LastSeenRound,
                    observation_age = memory.ObservationAge,
                    last_known_current_hp = memory.LastKnownCurrentHp,
                    currently_visible = memory.CurrentlyVisible,
                }).ToArray(),
                relations = value.Relations.Select(relation => new TacticalV3RelationDto
                {
                    kind = Relation(relation.Kind),
                    source = Reference(relation.Source),
                    target = Reference(relation.Target),
                    int_feature = relation.IntFeature,
                    float_feature = relation.FloatFeature,
                    bool_feature = relation.BoolFeature,
                }).ToArray(),
            };

        static TacticalV3CandidateDto Candidate(TacticalV3Candidate value) =>
            new TacticalV3CandidateDto
            {
                candidate_id = value.CandidateId,
                decision_id = value.DecisionId,
                kind = CandidateKind(value.Kind),
                actor = Reference(value.Actor),
                target = Reference(value.Target),
                template = Reference(value.Template),
                cell = Reference(value.Cell),
                projection = new TacticalV3ProjectedDeltaDto
                {
                    source_cell = Reference(value.Projection.SourceCell),
                    destination_cell = Reference(value.Projection.DestinationCell),
                    template = Reference(value.Projection.Template),
                    target = Reference(value.Projection.Target),
                    horizontal_movement_spent =
                        value.Projection.HorizontalMovementSpent,
                    vertical_movement_spent =
                        value.Projection.VerticalMovementSpent,
                    target_hp_delta = value.Projection.TargetHpDelta,
                    damage = value.Projection.Damage,
                    is_lethal = value.Projection.IsLethal,
                    bounty_delta = value.Projection.BountyDelta,
                    points_delta = value.Projection.PointsDelta,
                    round_delta = value.Projection.RoundDelta,
                    is_terminal = value.Projection.IsTerminal,
                },
            };

        static TacticalV3RewardDto Reward(TacticalV3RewardBreakdown value) =>
            new TacticalV3RewardDto
            {
                terminal_outcome = value.TerminalOutcome,
                known_health_adjusted_material_progress =
                    value.KnownHealthAdjustedMaterialProgress,
                public_resource_progress = value.PublicResourceProgress,
                time_pressure = value.TimePressure,
                total = value.Total,
                finalized = value.Finalized,
            };

        static TacticalV3TokenReferenceDto Reference(TacticalV3TokenRef value) =>
            new TacticalV3TokenReferenceDto
            {
                table = Table(value.Table),
                row = value.Row,
            };

        static TacticalV3TokenReferenceDto Reference(TacticalV3TokenRef? value) =>
            value.HasValue ? Reference(value.Value) : null;

        static void Validate(
            TacticalV3Observation observation,
            IReadOnlyList<TacticalV3Candidate> candidates,
            long decisionId,
            TacticalV3RewardBreakdown reward)
        {
            if (observation.Cells == null || observation.Units == null ||
                observation.Templates == null ||
                observation.CapabilityDefinitions == null ||
                observation.CapabilityAllocations == null ||
                observation.Rules == null || observation.Memory == null ||
                observation.Relations == null || candidates == null || reward == null)
                throw new InvalidOperationException(
                    "tactical-v3 payload tables and reward must not be null");

            for (int row = 0; row < candidates.Count; row++)
            {
                TacticalV3Candidate candidate = candidates[row] ??
                    throw new InvalidOperationException(
                        "tactical-v3 candidate rows must not be null");
                if (candidate.CandidateId != row)
                    throw new InvalidOperationException(
                        "tactical-v3 candidate id must equal its row");
                if (candidate.DecisionId != decisionId)
                    throw new InvalidOperationException(
                        "tactical-v3 candidate decision id mismatch");
                ValidateCandidate(candidate, observation, candidates.Count);
            }
            foreach (TacticalV3UnitToken unit in observation.Units)
                Require(unit.Cell, TacticalV3TableKind.Cells, observation,
                    candidates.Count, "unit.cell");
            foreach (TacticalV3CapabilityAllocationToken allocation
                in observation.CapabilityAllocations)
            {
                Require(allocation.Owner,
                    new[] { TacticalV3TableKind.Units, TacticalV3TableKind.Templates },
                    observation, candidates.Count, "allocation.owner");
                Require(allocation.Definition,
                    TacticalV3TableKind.CapabilityDefinitions, observation,
                    candidates.Count, "allocation.definition");
                if (observation.CapabilityDefinitions[allocation.Definition.Row].Kind !=
                    allocation.Capability)
                    throw new InvalidOperationException(
                        "tactical-v3 allocation capability mismatch");
            }
            foreach (TacticalV3MemoryToken memory in observation.Memory)
                Require(memory.Cell, TacticalV3TableKind.Cells, observation,
                    candidates.Count, "memory.cell");
            foreach (TacticalV3RelationToken relation in observation.Relations)
            {
                Require(relation.Source, observation, candidates.Count,
                    "relation.source");
                Require(relation.Target, observation, candidates.Count,
                    "relation.target");
                Finite(relation.FloatFeature, "relation.float_feature");
            }
            foreach (TacticalV3RuleToken rule in observation.Rules)
                Finite(rule.FloatValue, "rule.float_value");
            Finite(reward.TerminalOutcome, "reward.terminal_outcome");
            Finite(reward.KnownHealthAdjustedMaterialProgress,
                "reward.known_health_adjusted_material_progress");
            Finite(reward.PublicResourceProgress,
                "reward.public_resource_progress");
            Finite(reward.TimePressure, "reward.time_pressure");
            Finite(reward.Total, "reward.total");
        }

        static void ValidateCandidate(
            TacticalV3Candidate candidate,
            TacticalV3Observation observation,
            int candidateCount)
        {
            Require(candidate.Actor, observation, candidateCount, "candidate.actor");
            Require(candidate.Target, observation, candidateCount, "candidate.target");
            Require(candidate.Template, observation, candidateCount, "candidate.template");
            Require(candidate.Cell, observation, candidateCount, "candidate.cell");
            TacticalV3ProjectedDelta projection = candidate.Projection ??
                throw new InvalidOperationException(
                    "tactical-v3 candidate projection must not be null");
            Require(projection.SourceCell, observation, candidateCount,
                "projection.source_cell");
            Require(projection.DestinationCell, observation, candidateCount,
                "projection.destination_cell");
            Require(projection.Template, observation, candidateCount,
                "projection.template");
            Require(projection.Target, observation, candidateCount,
                "projection.target");
        }

        static void Require(
            TacticalV3TokenRef? reference,
            TacticalV3Observation observation,
            int candidateCount,
            string field)
        {
            if (reference.HasValue)
                Require(reference.Value, observation, candidateCount, field);
        }

        static void Require(
            TacticalV3TokenRef reference,
            TacticalV3TableKind expected,
            TacticalV3Observation observation,
            int candidateCount,
            string field) =>
            Require(reference, new[] { expected }, observation, candidateCount, field);

        static void Require(
            TacticalV3TokenRef reference,
            IReadOnlyCollection<TacticalV3TableKind> expected,
            TacticalV3Observation observation,
            int candidateCount,
            string field)
        {
            if (!expected.Contains(reference.Table))
                throw new InvalidOperationException(
                    "tactical-v3 " + field + " references the wrong table");
            Require(reference, observation, candidateCount, field);
        }

        static void Require(
            TacticalV3TokenRef reference,
            TacticalV3Observation observation,
            int candidateCount,
            string field)
        {
            int count;
            switch (reference.Table)
            {
                case TacticalV3TableKind.Cells:
                    count = observation.Cells.Count; break;
                case TacticalV3TableKind.Units:
                    count = observation.Units.Count; break;
                case TacticalV3TableKind.Templates:
                    count = observation.Templates.Count; break;
                case TacticalV3TableKind.CapabilityDefinitions:
                    count = observation.CapabilityDefinitions.Count; break;
                case TacticalV3TableKind.CapabilityAllocations:
                    count = observation.CapabilityAllocations.Count; break;
                case TacticalV3TableKind.Rules:
                    count = observation.Rules.Count; break;
                case TacticalV3TableKind.MemoryRecords:
                    count = observation.Memory.Count; break;
                case TacticalV3TableKind.Relations:
                    count = observation.Relations.Count; break;
                case TacticalV3TableKind.Candidates:
                    count = candidateCount; break;
                default:
                    throw new InvalidOperationException(
                        "tactical-v3 " + field + " references an unknown table");
            }
            if (reference.Row < 0 || reference.Row >= count)
                throw new InvalidOperationException(
                    "tactical-v3 " + field + " row is out of range");
        }

        static void Finite(float value, string field)
        {
            if (float.IsNaN(value) || float.IsInfinity(value))
                throw new InvalidOperationException(
                    "tactical-v3 " + field + " must be finite");
        }

        static string Table(TacticalV3TableKind value) => value switch
        {
            TacticalV3TableKind.Cells => "cells",
            TacticalV3TableKind.Units => "units",
            TacticalV3TableKind.Templates => "templates",
            TacticalV3TableKind.CapabilityDefinitions => "capability_definitions",
            TacticalV3TableKind.CapabilityAllocations => "capability_allocations",
            TacticalV3TableKind.Rules => "rules",
            TacticalV3TableKind.MemoryRecords => "memory_records",
            TacticalV3TableKind.Relations => "relations",
            TacticalV3TableKind.Candidates => "candidates",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string Owner(TacticalV3RelativeOwner value) => value switch
        {
            TacticalV3RelativeOwner.Self => "self",
            TacticalV3RelativeOwner.Opponent => "opponent",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string Terrain(TerrainType value) => value switch
        {
            TerrainType.Plains => "plains",
            TerrainType.Forest => "forest",
            TerrainType.Rough => "rough",
            TerrainType.Water => "water",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string Capability(TacticalV3CapabilityKind value) => value switch
        {
            TacticalV3CapabilityKind.Health => "health",
            TacticalV3CapabilityKind.Damage => "damage",
            TacticalV3CapabilityKind.Defense => "defense",
            TacticalV3CapabilityKind.Movement => "movement",
            TacticalV3CapabilityKind.VerticalMovement => "vertical_movement",
            TacticalV3CapabilityKind.Range => "range",
            TacticalV3CapabilityKind.RangeArc => "range_arc",
            TacticalV3CapabilityKind.Vision => "vision",
            TacticalV3CapabilityKind.VisionArc => "vision_arc",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string Rule(TacticalV3RuleKind value) => value switch
        {
            TacticalV3RuleKind.WinConditions => "win_conditions",
            TacticalV3RuleKind.Round => "round",
            TacticalV3RuleKind.RoundCap => "round_cap",
            TacticalV3RuleKind.ActionsPerTurn => "actions_per_turn",
            TacticalV3RuleKind.StartingPoints => "starting_points",
            TacticalV3RuleKind.SelfPoints => "self_points",
            TacticalV3RuleKind.OpponentPoints => "opponent_points",
            TacticalV3RuleKind.DamageFloor => "damage_floor",
            TacticalV3RuleKind.DamageHighGroundBonus =>
                "damage_high_ground_bonus",
            TacticalV3RuleKind.RangeHighGroundBonus =>
                "range_high_ground_bonus",
            TacticalV3RuleKind.BountyRate => "bounty_rate",
            TacticalV3RuleKind.DeployCostMultiplier =>
                "deploy_cost_multiplier",
            TacticalV3RuleKind.FogOfWar => "fog_of_war",
            TacticalV3RuleKind.MaxDesignPointCost => "max_design_point_cost",
            TacticalV3RuleKind.DesignFee => "design_fee",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string Relation(TacticalV3RelationKind value) => value switch
        {
            TacticalV3RelationKind.Neighbor => "neighbor",
            TacticalV3RelationKind.Occupies => "occupies",
            TacticalV3RelationKind.HasCapability => "has_capability",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        static string CandidateKind(TacticalV3CandidateKind value) => value switch
        {
            TacticalV3CandidateKind.Attack => "attack",
            TacticalV3CandidateKind.Move => "move",
            TacticalV3CandidateKind.Deploy => "deploy",
            TacticalV3CandidateKind.EndTurn => "end_turn",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };
    }
}
