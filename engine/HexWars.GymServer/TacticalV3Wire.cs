using System;
using System.Linq;
using HexWars.Engine.Rl;

namespace HexWars.GymServer
{
    internal static class TacticalV3Wire
    {
        public static object Spaces(TrainingScenario scenario, TacticalV3Contract contract)
        {
            if (scenario == null) throw new ArgumentNullException(nameof(scenario));
            if (contract == null) throw new ArgumentNullException(nameof(contract));

            // Re-run scenario and capacity validation at the serialization boundary. The returned
            // config is intentionally not serialized; the canonical contract is the wire authority.
            scenario.BuildTacticalV3();
            return new
            {
                scenario_id = scenario.Id,
                scenario_schema_version = scenario.SchemaVersion,
                contract_version = contract.Version,
                contract_hash = contract.ContractHash,
                encoding_hash = contract.EncodingHash,
                capacity_hash = contract.CapacityHash,
                environment_kind = contract.EnvironmentKind,
                match = contract.Match,
                encoding = contract.Encoding,
                capacity = contract.Capacity,
            };
        }

        public static object View(TacticalV3View view)
        {
            if (view == null) throw new ArgumentNullException(nameof(view));
            ValidateReferences(view);

            TacticalV3DecisionFrame decision = view.Decision;
            TacticalV3Observation observation = decision.Observation;
            TacticalV3RewardBreakdown reward = view.Reward;
            return new
            {
                decision_id = decision.DecisionId,
                seat = (int)view.Seat,
                observation = new
                {
                    cells = observation.Cells.Select(cell => new
                    {
                        q = cell.Q,
                        r = cell.R,
                        terrain = Terrain(cell.Terrain),
                        elevation = cell.Elevation,
                        self_deployment_zone = cell.SelfDeploymentZone,
                        opponent_deployment_zone = cell.OpponentDeploymentZone,
                        controller = cell.Controller.HasValue
                            ? RelativeOwner(cell.Controller.Value)
                            : null,
                        is_boundary = cell.IsBoundary,
                        currently_visible = cell.CurrentlyVisible,
                        previously_observed = cell.PreviouslyObserved,
                    }).ToArray(),
                    units = observation.Units.Select(unit => new
                    {
                        owner = RelativeOwner(unit.Owner),
                        current_hp = unit.CurrentHp,
                        max_hp = unit.MaxHp,
                        cell = TokenReference(unit.Cell),
                        elevation = unit.Elevation,
                        moved = unit.Moved,
                        attacked = unit.Attacked,
                        horizontal_movement_spent = unit.HorizontalMovementSpent,
                        vertical_movement_spent = unit.VerticalMovementSpent,
                        point_cost = unit.PointCost,
                        deploy_cost = unit.DeployCost,
                        currently_visible = unit.CurrentlyVisible,
                    }).ToArray(),
                    templates = observation.Templates.Select(template => new
                    {
                        owner = RelativeOwner(template.Owner),
                        point_cost = template.PointCost,
                        deploy_cost = template.DeployCost,
                        is_fixed = template.IsFixed,
                        is_deployable = template.IsDeployable,
                    }).ToArray(),
                    capability_definitions = observation.CapabilityDefinitions.Select(definition => new
                    {
                        kind = Capability(definition.Kind),
                    }).ToArray(),
                    capability_allocations = observation.CapabilityAllocations.Select(allocation => new
                    {
                        owner = TokenReference(allocation.Owner),
                        definition = TokenReference(allocation.Definition),
                        capability = Capability(allocation.Capability),
                        purchased_level = allocation.PurchasedLevel,
                        effective_value = allocation.EffectiveValue,
                    }).ToArray(),
                    rules = observation.Rules.Select(rule => new
                    {
                        kind = Rule(rule.Kind),
                        int_value = rule.IntValue,
                        float_value = rule.FloatValue,
                        bool_value = rule.BoolValue,
                    }).ToArray(),
                    memory = observation.Memory.Select(memory => new
                    {
                        cell = TokenReference(memory.Cell),
                        last_seen_round = memory.LastSeenRound,
                        observation_age = memory.ObservationAge,
                        last_known_current_hp = memory.LastKnownCurrentHp,
                        currently_visible = memory.CurrentlyVisible,
                    }).ToArray(),
                    relations = observation.Relations.Select(relation => new
                    {
                        kind = Relation(relation.Kind),
                        source = TokenReference(relation.Source),
                        target = TokenReference(relation.Target),
                        int_feature = relation.IntFeature,
                        float_feature = relation.FloatFeature,
                        bool_feature = relation.BoolFeature,
                    }).ToArray(),
                },
                candidates = decision.Candidates.Select(candidate => new
                {
                    candidate_id = candidate.CandidateId,
                    decision_id = candidate.DecisionId,
                    kind = Candidate(candidate.Kind),
                    actor = NullableTokenReference(candidate.Actor),
                    target = NullableTokenReference(candidate.Target),
                    template = NullableTokenReference(candidate.Template),
                    cell = NullableTokenReference(candidate.Cell),
                    projection = ProjectedDelta(candidate.Projection),
                }).ToArray(),
                reward = new
                {
                    terminal_outcome = reward.TerminalOutcome,
                    known_health_adjusted_material_progress =
                        reward.KnownHealthAdjustedMaterialProgress,
                    public_resource_progress = reward.PublicResourceProgress,
                    time_pressure = reward.TimePressure,
                    total = reward.Total,
                    finalized = reward.Finalized,
                },
                winner = view.Winner,
                terminated = view.Terminated,
                truncated = view.Truncated,
                start_profile = view.StartProfileId,
                reference_seat = (int)view.ReferenceSeat,
            };
        }

        private static object ProjectedDelta(TacticalV3ProjectedDelta projection) => new
        {
            source_cell = NullableTokenReference(projection.SourceCell),
            destination_cell = NullableTokenReference(projection.DestinationCell),
            template = NullableTokenReference(projection.Template),
            target = NullableTokenReference(projection.Target),
            horizontal_movement_spent = projection.HorizontalMovementSpent,
            vertical_movement_spent = projection.VerticalMovementSpent,
            target_hp_delta = projection.TargetHpDelta,
            damage = projection.Damage,
            is_lethal = projection.IsLethal,
            bounty_delta = projection.BountyDelta,
            points_delta = projection.PointsDelta,
            round_delta = projection.RoundDelta,
            is_terminal = projection.IsTerminal,
        };

        private static object TokenReference(TacticalV3TokenRef reference) => new
        {
            table = Table(reference.Table),
            row = reference.Row,
        };

        private static object? NullableTokenReference(TacticalV3TokenRef? reference) =>
            reference.HasValue ? TokenReference(reference.Value) : null;

        private static void ValidateReferences(TacticalV3View view)
        {
            TacticalV3Observation observation = view.Decision.Observation;
            int candidateCount = view.Decision.Candidates.Count;

            foreach (TacticalV3UnitToken unit in observation.Units)
                Validate(unit.Cell, observation, candidateCount);
            foreach (TacticalV3CapabilityAllocationToken allocation in observation.CapabilityAllocations)
            {
                Validate(allocation.Owner, observation, candidateCount);
                Validate(allocation.Definition, observation, candidateCount);
            }
            foreach (TacticalV3MemoryToken memory in observation.Memory)
                Validate(memory.Cell, observation, candidateCount);
            foreach (TacticalV3RelationToken relation in observation.Relations)
            {
                Validate(relation.Source, observation, candidateCount);
                Validate(relation.Target, observation, candidateCount);
            }
            foreach (TacticalV3Candidate candidate in view.Decision.Candidates)
            {
                Validate(candidate.Actor, observation, candidateCount);
                Validate(candidate.Target, observation, candidateCount);
                Validate(candidate.Template, observation, candidateCount);
                Validate(candidate.Cell, observation, candidateCount);
                TacticalV3ProjectedDelta projection = candidate.Projection;
                Validate(projection.SourceCell, observation, candidateCount);
                Validate(projection.DestinationCell, observation, candidateCount);
                Validate(projection.Template, observation, candidateCount);
                Validate(projection.Target, observation, candidateCount);
            }
        }

        private static void Validate(
            TacticalV3TokenRef? reference, TacticalV3Observation observation, int candidateCount)
        {
            if (reference.HasValue) Validate(reference.Value, observation, candidateCount);
        }

        private static void Validate(
            TacticalV3TokenRef reference, TacticalV3Observation observation, int candidateCount)
        {
            int length = reference.Table switch
            {
                TacticalV3TableKind.Cells => observation.Cells.Count,
                TacticalV3TableKind.Units => observation.Units.Count,
                TacticalV3TableKind.Templates => observation.Templates.Count,
                TacticalV3TableKind.CapabilityDefinitions => observation.CapabilityDefinitions.Count,
                TacticalV3TableKind.CapabilityAllocations => observation.CapabilityAllocations.Count,
                TacticalV3TableKind.Rules => observation.Rules.Count,
                TacticalV3TableKind.MemoryRecords => observation.Memory.Count,
                TacticalV3TableKind.Relations => observation.Relations.Count,
                TacticalV3TableKind.Candidates => candidateCount,
                _ => throw new InvalidOperationException(
                    "tactical-v3 token reference has an unknown table"),
            };
            if (reference.Row < 0 || reference.Row >= length)
                throw new InvalidOperationException(
                    "tactical-v3 token reference is outside table length: " +
                    Table(reference.Table) + "[" + reference.Row + "] of " + length);
        }

        private static string Table(TacticalV3TableKind value) => value switch
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

        private static string RelativeOwner(TacticalV3RelativeOwner value) => value switch
        {
            TacticalV3RelativeOwner.Self => "self",
            TacticalV3RelativeOwner.Opponent => "opponent",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        private static string Terrain(HexWars.Engine.TerrainType value) => value switch
        {
            HexWars.Engine.TerrainType.Plains => "plains",
            HexWars.Engine.TerrainType.Forest => "forest",
            HexWars.Engine.TerrainType.Rough => "rough",
            HexWars.Engine.TerrainType.Water => "water",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        private static string Capability(TacticalV3CapabilityKind value) => value switch
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

        private static string Rule(TacticalV3RuleKind value) => value switch
        {
            TacticalV3RuleKind.WinConditions => "win_conditions",
            TacticalV3RuleKind.Round => "round",
            TacticalV3RuleKind.RoundCap => "round_cap",
            TacticalV3RuleKind.ActionsPerTurn => "actions_per_turn",
            TacticalV3RuleKind.StartingPoints => "starting_points",
            TacticalV3RuleKind.SelfPoints => "self_points",
            TacticalV3RuleKind.OpponentPoints => "opponent_points",
            TacticalV3RuleKind.DamageFloor => "damage_floor",
            TacticalV3RuleKind.DamageHighGroundBonus => "damage_high_ground_bonus",
            TacticalV3RuleKind.RangeHighGroundBonus => "range_high_ground_bonus",
            TacticalV3RuleKind.BountyRate => "bounty_rate",
            TacticalV3RuleKind.DeployCostMultiplier => "deploy_cost_multiplier",
            TacticalV3RuleKind.FogOfWar => "fog_of_war",
            TacticalV3RuleKind.MaxDesignPointCost => "max_design_point_cost",
            TacticalV3RuleKind.DesignFee => "design_fee",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        private static string Relation(TacticalV3RelationKind value) => value switch
        {
            TacticalV3RelationKind.Neighbor => "neighbor",
            TacticalV3RelationKind.Occupies => "occupies",
            TacticalV3RelationKind.HasCapability => "has_capability",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };

        private static string Candidate(TacticalV3CandidateKind value) => value switch
        {
            TacticalV3CandidateKind.Attack => "attack",
            TacticalV3CandidateKind.Move => "move",
            TacticalV3CandidateKind.Deploy => "deploy",
            TacticalV3CandidateKind.EndTurn => "end_turn",
            _ => throw new ArgumentOutOfRangeException(nameof(value)),
        };
    }
}
