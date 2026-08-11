using System;
using System.Collections;
using System.Collections.Generic;
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
            TacticalV3Config config = scenario.BuildTacticalV3();
            MlEnvironmentKind environmentKind = ParseEnvironmentKind(contract.EnvironmentKind);
            TacticalV3Contract expected = TacticalV3Contract.Create(config, environmentKind);
            RequireContractEvidence(contract, expected);
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

        private static MlEnvironmentKind ParseEnvironmentKind(string environmentKind) =>
            environmentKind switch
            {
                "tactical" => MlEnvironmentKind.Tactical,
                "duel" => MlEnvironmentKind.Duel,
                _ => throw ContractEvidenceError(
                    "environment role must be exactly tactical or duel"),
            };

        private static void RequireContractEvidence(
            TacticalV3Contract supplied, TacticalV3Contract expected)
        {
            if (supplied.Version != expected.Version ||
                supplied.EnvironmentKind != expected.EnvironmentKind ||
                supplied.ContractHash != expected.ContractHash ||
                supplied.EncodingHash != expected.EncodingHash ||
                supplied.CapacityHash != expected.CapacityHash ||
                !ExactValue(supplied.Match, expected.Match) ||
                !ExactValue(supplied.Encoding, expected.Encoding) ||
                !ExactValue(supplied.Capacity, expected.Capacity))
            {
                throw ContractEvidenceError(
                    "supplied identity does not match the validated scenario and environment role");
            }
        }

        private static bool ExactValue(object? supplied, object? expected)
        {
            if (ReferenceEquals(supplied, expected)) return true;
            if (supplied == null || expected == null) return false;
            if (supplied is string suppliedString && expected is string expectedString)
                return suppliedString == expectedString;
            if (supplied is bool suppliedBool && expected is bool expectedBool)
                return suppliedBool == expectedBool;
            if (supplied is int suppliedInt && expected is int expectedInt)
                return suppliedInt == expectedInt;
            if (supplied is long suppliedLong && expected is long expectedLong)
                return suppliedLong == expectedLong;
            if (supplied is float suppliedFloat && expected is float expectedFloat)
                return suppliedFloat.Equals(expectedFloat);
            if (supplied is double suppliedDouble && expected is double expectedDouble)
                return suppliedDouble.Equals(expectedDouble);
            if (supplied is IReadOnlyDictionary<string, object> suppliedMap &&
                expected is IReadOnlyDictionary<string, object> expectedMap)
            {
                if (suppliedMap.Count != expectedMap.Count) return false;
                foreach (KeyValuePair<string, object> item in expectedMap)
                {
                    if (!suppliedMap.TryGetValue(item.Key, out object? suppliedValue) ||
                        !ExactValue(suppliedValue, item.Value))
                    {
                        return false;
                    }
                }
                return true;
            }
            if (supplied is IEnumerable suppliedSequence && expected is IEnumerable expectedSequence)
            {
                IEnumerator suppliedItems = suppliedSequence.GetEnumerator();
                IEnumerator expectedItems = expectedSequence.GetEnumerator();
                while (true)
                {
                    bool suppliedNext = suppliedItems.MoveNext();
                    bool expectedNext = expectedItems.MoveNext();
                    if (suppliedNext != expectedNext) return false;
                    if (!suppliedNext) return true;
                    if (!ExactValue(suppliedItems.Current, expectedItems.Current)) return false;
                }
            }
            return false;
        }

        private static InvalidOperationException ContractEvidenceError(string message) =>
            new InvalidOperationException("tactical-v3 contract evidence mismatch: " + message);

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
                RequireReference(unit.Cell, TacticalV3TableKind.Cells,
                    observation, candidateCount, "unit.cell");
            foreach (TacticalV3CapabilityAllocationToken allocation in observation.CapabilityAllocations)
            {
                RequireReference(allocation.Owner,
                    new[] { TacticalV3TableKind.Units, TacticalV3TableKind.Templates },
                    observation, candidateCount, "capability_allocation.owner");
                TacticalV3TokenRef definition = RequireReference(
                    allocation.Definition, TacticalV3TableKind.CapabilityDefinitions,
                    observation, candidateCount, "capability_allocation.definition");
                if (observation.CapabilityDefinitions[definition.Row].Kind != allocation.Capability)
                    throw SemanticError(
                        "capability_allocation.capability does not match its definition");
            }
            foreach (TacticalV3MemoryToken memory in observation.Memory)
                RequireReference(memory.Cell, TacticalV3TableKind.Cells,
                    observation, candidateCount, "memory.cell");
            foreach (TacticalV3RelationToken relation in observation.Relations)
                ValidateRelation(relation, observation, candidateCount);
            foreach (TacticalV3Candidate candidate in view.Decision.Candidates)
                ValidateCandidate(candidate, observation, candidateCount);
        }

        private static void ValidateRelation(
            TacticalV3RelationToken relation,
            TacticalV3Observation observation,
            int candidateCount)
        {
            switch (relation.Kind)
            {
                case TacticalV3RelationKind.Neighbor:
                    RequireReference(relation.Source, TacticalV3TableKind.Cells,
                        observation, candidateCount, "neighbor.source");
                    RequireReference(relation.Target, TacticalV3TableKind.Cells,
                        observation, candidateCount, "neighbor.target");
                    break;
                case TacticalV3RelationKind.Occupies:
                    RequireReference(relation.Source, TacticalV3TableKind.Units,
                        observation, candidateCount, "occupies.source");
                    RequireReference(relation.Target, TacticalV3TableKind.Cells,
                        observation, candidateCount, "occupies.target");
                    break;
                case TacticalV3RelationKind.HasCapability:
                    RequireReference(relation.Source,
                        new[] { TacticalV3TableKind.Units, TacticalV3TableKind.Templates },
                        observation, candidateCount, "has_capability.source");
                    RequireReference(relation.Target, TacticalV3TableKind.CapabilityDefinitions,
                        observation, candidateCount, "has_capability.target");
                    break;
                default:
                    throw SemanticError("relation.kind is unknown");
            }
        }

        private static void ValidateCandidate(
            TacticalV3Candidate candidate,
            TacticalV3Observation observation,
            int candidateCount)
        {
            TacticalV3ProjectedDelta projection = candidate.Projection;
            switch (candidate.Kind)
            {
                case TacticalV3CandidateKind.Attack:
                    RequireReference(candidate.Actor, TacticalV3TableKind.Units,
                        observation, candidateCount, "attack.actor");
                    RequireReference(candidate.Target, TacticalV3TableKind.Units,
                        observation, candidateCount, "attack.target");
                    RequireNull(candidate.Template, "attack.template");
                    RequireNull(candidate.Cell, "attack.cell");
                    RequireReference(projection.SourceCell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "attack.projection.source_cell");
                    RequireNull(projection.DestinationCell, "attack.projection.destination_cell");
                    RequireNull(projection.Template, "attack.projection.template");
                    RequireReference(projection.Target, TacticalV3TableKind.Units,
                        observation, candidateCount, "attack.projection.target");
                    break;
                case TacticalV3CandidateKind.Move:
                    RequireReference(candidate.Actor, TacticalV3TableKind.Units,
                        observation, candidateCount, "move.actor");
                    RequireNull(candidate.Target, "move.target");
                    RequireNull(candidate.Template, "move.template");
                    RequireReference(candidate.Cell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "move.cell");
                    RequireReference(projection.SourceCell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "move.projection.source_cell");
                    RequireReference(projection.DestinationCell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "move.projection.destination_cell");
                    RequireNull(projection.Template, "move.projection.template");
                    RequireNull(projection.Target, "move.projection.target");
                    break;
                case TacticalV3CandidateKind.Deploy:
                    RequireNull(candidate.Actor, "deploy.actor");
                    RequireNull(candidate.Target, "deploy.target");
                    RequireReference(candidate.Template, TacticalV3TableKind.Templates,
                        observation, candidateCount, "deploy.template");
                    RequireReference(candidate.Cell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "deploy.cell");
                    RequireNull(projection.SourceCell, "deploy.projection.source_cell");
                    RequireReference(projection.DestinationCell, TacticalV3TableKind.Cells,
                        observation, candidateCount, "deploy.projection.destination_cell");
                    RequireReference(projection.Template, TacticalV3TableKind.Templates,
                        observation, candidateCount, "deploy.projection.template");
                    RequireNull(projection.Target, "deploy.projection.target");
                    break;
                case TacticalV3CandidateKind.EndTurn:
                    RequireNull(candidate.Actor, "end_turn.actor");
                    RequireNull(candidate.Target, "end_turn.target");
                    RequireNull(candidate.Template, "end_turn.template");
                    RequireNull(candidate.Cell, "end_turn.cell");
                    RequireNull(projection.SourceCell, "end_turn.projection.source_cell");
                    RequireNull(projection.DestinationCell, "end_turn.projection.destination_cell");
                    RequireNull(projection.Template, "end_turn.projection.template");
                    RequireNull(projection.Target, "end_turn.projection.target");
                    break;
                default:
                    throw SemanticError("candidate.kind is unknown");
            }
        }

        private static TacticalV3TokenRef RequireReference(
            TacticalV3TokenRef? reference, TacticalV3TableKind expected,
            TacticalV3Observation observation, int candidateCount, string field)
        {
            if (!reference.HasValue) throw SemanticError(field + " is required");
            return RequireReference(reference.Value, expected, observation, candidateCount, field);
        }

        private static TacticalV3TokenRef RequireReference(
            TacticalV3TokenRef reference, TacticalV3TableKind expected,
            TacticalV3Observation observation, int candidateCount, string field) =>
            RequireReference(reference, new[] { expected }, observation, candidateCount, field);

        private static TacticalV3TokenRef RequireReference(
            TacticalV3TokenRef reference, IReadOnlyCollection<TacticalV3TableKind> expected,
            TacticalV3Observation observation, int candidateCount, string field)
        {
            if (!expected.Contains(reference.Table))
                throw SemanticError(field + " references incompatible table " + reference.Table +
                    " instead of " + string.Join(" or ", expected.Select(Table)));
            int length = TableLength(reference.Table, observation, candidateCount);
            if (reference.Row < 0 || reference.Row >= length)
                throw new InvalidOperationException(
                    "tactical-v3 token reference is outside table length: " +
                    Table(reference.Table) + "[" + reference.Row + "] of " + length);
            return reference;
        }

        private static int TableLength(
            TacticalV3TableKind table, TacticalV3Observation observation, int candidateCount) =>
            table switch
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
                _ => throw SemanticError("token reference has an unknown table"),
            };

        private static void RequireNull(TacticalV3TokenRef? reference, string field)
        {
            if (reference.HasValue) throw SemanticError(field + " must be null");
        }

        private static InvalidOperationException SemanticError(string message) =>
            new InvalidOperationException("tactical-v3 token semantic mismatch: " + message);

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
