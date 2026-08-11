using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3ObservationTests
    {
        [Test]
        public void Observe_RepresentsMechanicsNotRosterNamesOrEngineIds()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Cells, Has.Count.EqualTo(fixture.State.Board.TileCount));
            Assert.That(typeof(TacticalV3UnitToken).GetProperty("EngineId"), Is.Null);
            Assert.That(typeof(TacticalV3UnitToken).GetProperty("Name"), Is.Null);
            Assert.That(typeof(TacticalV3TemplateToken).GetProperty("EngineId"), Is.Null);
            Assert.That(typeof(TacticalV3TemplateToken).GetProperty("Name"), Is.Null);
            Assert.Throws<NotSupportedException>(() =>
                ((System.Collections.IList)observation.Cells)[0] = observation.Cells[0]);
        }

        [Test]
        public void Observe_OrdersSeatRelativeRowsAndIncludesAllNineCapabilities()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);
            TacticalV2Layout layout = new TacticalV2Layout(TacticalV3Fixtures.Config().Match);

            Assert.That(observation.Cells.Select(cell => (cell.Q, cell.R)),
                Is.EqualTo(layout.Cells.Select(cell => (cell.Q, cell.R))));
            Assert.That(observation.Units.Take(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(observation.Units.Skip(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Opponent));
            Assert.That(observation.Units[0].CurrentHp, Is.EqualTo(observation.Units[0].MaxHp));
            Assert.That(observation.Units[0].Cell.Table, Is.EqualTo(TacticalV3TableKind.Cells));
            Assert.That(observation.Units[0].Moved, Is.False);
            Assert.That(observation.Units[0].Attacked, Is.False);
            Assert.That(observation.Units[0].HorizontalMovementSpent, Is.EqualTo(0));
            Assert.That(observation.Units[0].VerticalMovementSpent, Is.EqualTo(0));
            Assert.That(observation.Units[0].PointCost, Is.EqualTo(
                fixture.State.Player(PlayerId.Player0).UnitsOnBoard[0].Stats.PointCost));
            Assert.That(observation.Templates.Take(5).Select(template => template.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(observation.Templates.Skip(5).Select(template => template.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Opponent));
            Assert.That(observation.CapabilityAllocations.Count,
                Is.EqualTo(9 * (observation.Units.Count + observation.Templates.Count)));
            Assert.That(observation.CapabilityAllocations.Take(9).Select(row => row.Capability), Is.EqualTo(
                TacticalV3Capabilities.All.Select(definition => definition.Kind)));
        }

        [Test]
        public void Observe_ExposesExplicitRulesAndOnlyValidRelationsWithEmptyMemory()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Rules.Select(rule => rule.Kind), Is.EqualTo(new[]
            {
                TacticalV3RuleKind.WinConditions,
                TacticalV3RuleKind.Round,
                TacticalV3RuleKind.RoundCap,
                TacticalV3RuleKind.ActionsPerTurn,
                TacticalV3RuleKind.StartingPoints,
                TacticalV3RuleKind.SelfPoints,
                TacticalV3RuleKind.OpponentPoints,
                TacticalV3RuleKind.DamageFloor,
                TacticalV3RuleKind.DamageHighGroundBonus,
                TacticalV3RuleKind.RangeHighGroundBonus,
                TacticalV3RuleKind.BountyRate,
                TacticalV3RuleKind.DeployCostMultiplier,
                TacticalV3RuleKind.FogOfWar,
                TacticalV3RuleKind.MaxDesignPointCost,
                TacticalV3RuleKind.DesignFee,
            }));
            Assert.That(Rule(observation, TacticalV3RuleKind.WinConditions).IntValue,
                Is.EqualTo((int)WinBy.Annihilation));
            Assert.That(Rule(observation, TacticalV3RuleKind.FogOfWar).BoolValue, Is.False);
            Assert.That(observation.Memory, Is.Empty);
            Assert.That(observation.Relations.Any(row => row.Kind == TacticalV3RelationKind.Neighbor), Is.True);
            Assert.That(observation.Relations.Count(row => row.Kind == TacticalV3RelationKind.Occupies),
                Is.EqualTo(observation.Units.Count));
            Assert.That(observation.Relations.Count(row => row.Kind == TacticalV3RelationKind.HasCapability),
                Is.EqualTo(observation.CapabilityAllocations.Count));
            Assert.That(observation.Relations.All(row => row.Source.Row >= 0 && row.Target.Row >= 0), Is.True);
        }

        [Test]
        public void Observe_ReflectsPlayer1CoordinatesAndDoesNotMutateAuthoritativeState()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            TacticalV2Layout layout = new TacticalV2Layout(TacticalV3Fixtures.Config().Match);
            string before = AuthoritativeStateFingerprint(fixture.State);

            TacticalV3Observation observation = fixture.Source.Observe(
                fixture.State, PlayerId.Player1, EmptyObservationMemory.Instance);

            Assert.That(observation.Cells.Select(cell => (cell.Q, cell.R)), Is.EqualTo(
                layout.Cells.Select(cell =>
                {
                    HexCoord reflected = IndependentReflectCell(
                        layout.BoardGen.Width,
                        layout.BoardGen.Height,
                        cell);
                    return (reflected.Q, reflected.R);
                })));
            Assert.That(observation.Units.Take(3).Select(unit => unit.Owner),
                Is.All.EqualTo(TacticalV3RelativeOwner.Self));
            Assert.That(Rule(observation, TacticalV3RuleKind.SelfPoints).IntValue,
                Is.EqualTo(fixture.State.Player(PlayerId.Player1).Points));
            Assert.That(Rule(observation, TacticalV3RuleKind.OpponentPoints).IntValue,
                Is.EqualTo(fixture.State.Player(PlayerId.Player0).Points));
            Assert.That(AuthoritativeStateFingerprint(fixture.State), Is.EqualTo(before));
        }

        [Test]
        public void Observe_ThrowsBeforeReturningWhenCapacityIsExceeded()
        {
            TacticalV3Config config = new TacticalV3Config(
                TacticalV3Fixtures.Match(),
                new TacticalV3CapacityProfile(1, 64, 32, 128, 2048, 128, 64, 65536, 32768),
                new TacticalV3RewardConfig(+1f, -1f, 0.20f, 0.05f, 0.5f));
            TacticalV2Layout layout = new TacticalV2Layout(config.Match);
            TacticalV3SeatObservationSource source = new TacticalV3SeatObservationSource(config);

            Assert.Throws<InvalidOperationException>(() => source.Observe(
                layout.NewGame(17).State, PlayerId.Player0, EmptyObservationMemory.Instance));
        }

        [Test]
        public void Observe_ExcludesDeadUnitsWithoutDisturbingCapabilityAllocationOwnership()
        {
            TacticalV3Fixture fixture = TacticalV3Fixture.Standard(seed: 17);
            PlayerState original = fixture.State.Player(PlayerId.Player0);
            Unit[] units = original.UnitsOnBoard.ToArray();
            units[0] = units[0].WithDamage(units[0].CurrentHp);
            var player0 = new PlayerState(PlayerId.Player0, original.Points, original.Barracks, units,
                original.Generators, original.DestroyedValue);
            var state = new GameState(fixture.State.Board, fixture.State.Config,
                new[] { player0, fixture.State.Player(PlayerId.Player1) }, fixture.State.ActivePlayer,
                fixture.State.Round, fixture.State.NextEntityId, fixture.State.IsGameOver, fixture.State.Winner,
                fixture.State.MovedUnitIds, fixture.State.AttackedUnitIds, fixture.State.MovementSpent);

            TacticalV3Observation observation = fixture.Source.Observe(
                state, PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.That(observation.Units, Has.Count.EqualTo(5));
            Assert.That(observation.CapabilityAllocations.Count,
                Is.EqualTo(9 * (observation.Units.Count + observation.Templates.Count)));
        }

        [Test]
        public void Observe_RelationsStayWithinIndependentBoundAcrossProfilesSeedsAndDenseCapacity()
        {
            var cases = new[]
            {
                (Profile: "standard-3v3", Seed: 17),
                (Profile: "conversion-1v1-near", Seed: 6_000_005),
                (Profile: "conversion-1v1-medium", Seed: 6_000_005),
                (Profile: "conversion-1v1-far", Seed: 6_000_005),
                (Profile: "conversion-2v1-far", Seed: 6_000_005),
            };
            foreach ((string profileId, int seed) in cases)
            {
                TacticalV3Config config = TacticalV3Fixtures.ProfiledConfig(profileId);
                TacticalV2StartProfile profile = config.Match.StartProfiles.Single(
                    item => item.Id == profileId);
                GameState state = new TacticalV2Layout(config.Match)
                    .NewGame(seed, profile, PlayerId.Player0).State;
                TacticalV3Observation observation = new TacticalV3SeatObservationSource(config)
                    .Observe(state, PlayerId.Player0, EmptyObservationMemory.Instance);

                Assert.That(observation.Relations.Count(
                    relation => relation.Kind == TacticalV3RelationKind.Neighbor),
                    Is.EqualTo(616), profileId);
                Assert.That(observation.Relations.Count, Is.LessThanOrEqualTo(1346), profileId);
            }

            TacticalV3Config denseConfig = TacticalV3Fixtures.CapacityBoundConfig();
            var denseSource = new TacticalV3SeatObservationSource(denseConfig);
            TacticalV3Observation dense = denseSource.Observe(
                TacticalV3Fixtures.DenseCapacityState(denseConfig),
                PlayerId.Player0, EmptyObservationMemory.Instance);

            Assert.Multiple(() =>
            {
                Assert.That(dense.Units, Has.Count.EqualTo(64));
                Assert.That(dense.Templates, Has.Count.EqualTo(10));
                Assert.That(dense.CapabilityAllocations, Has.Count.EqualTo(666));
                Assert.That(dense.Relations.Count(
                    relation => relation.Kind == TacticalV3RelationKind.Neighbor),
                    Is.EqualTo(616));
                Assert.That(dense.Relations, Has.Count.EqualTo(1346));
            });
            Assert.Throws<InvalidOperationException>(() => denseSource.Observe(
                TacticalV3Fixtures.DenseCapacityState(denseConfig, totalUnits: 65),
                PlayerId.Player0, EmptyObservationMemory.Instance));
        }

        [Test]
        public void Observe_OneStructuredSchemaHandles13x9And24x16WithStableEncodingIdentity()
        {
            TacticalV3Config standardConfig = TacticalV3Fixtures.Config(13, 9);
            TacticalV3Config largeConfig = TacticalV3Fixtures.Config(24, 16);
            var standardSource = new TacticalV3SeatObservationSource(standardConfig);
            var largeSource = new TacticalV3SeatObservationSource(largeConfig);

            TacticalV3Observation standard = standardSource.Observe(
                new TacticalV2Layout(standardConfig.Match).NewGame(101).State,
                PlayerId.Player0,
                EmptyObservationMemory.Instance);
            TacticalV3Observation large = largeSource.Observe(
                new TacticalV2Layout(largeConfig.Match).NewGame(101).State,
                PlayerId.Player0,
                EmptyObservationMemory.Instance);
            TacticalV3Contract standardContract = TacticalV3Contract.Create(
                standardConfig, MlEnvironmentKind.Duel);
            TacticalV3Contract largeContract = TacticalV3Contract.Create(
                largeConfig, MlEnvironmentKind.Duel);

            Assert.Multiple(() =>
            {
                Assert.That(standardSource.GetType(), Is.SameAs(largeSource.GetType()));
                Assert.That(standard.GetType(), Is.SameAs(large.GetType()));
                Assert.That(standard.Cells, Has.Count.EqualTo(13 * 9));
                Assert.That(large.Cells, Has.Count.EqualTo(24 * 16));
                Assert.That(standard.CapabilityDefinitions.Select(row => row.Kind),
                    Is.EqualTo(large.CapabilityDefinitions.Select(row => row.Kind)));
                Assert.That(largeContract.EncodingHash, Is.EqualTo(standardContract.EncodingHash));
                Assert.That(largeContract.ContractHash, Is.Not.EqualTo(standardContract.ContractHash));
                Assert.That(largeContract.CapacityHash, Is.EqualTo(standardContract.CapacityHash));
            });
        }

        [Test]
        public void Observe_AsymmetricBoardPerspectivesReflectCoordinatesAndSwapOwners()
        {
            TacticalV3Config config = TacticalV3Fixtures.Config();
            var layout = new TacticalV2Layout(config.Match);
            GameState state = AsymmetricPerspectiveState(layout, layout.NewGame(101).State);
            var source = new TacticalV3SeatObservationSource(config);
            TacticalV3Observation player0 = source.Observe(
                state, PlayerId.Player0, EmptyObservationMemory.Instance);
            TacticalV3Observation player1 = source.Observe(
                state, PlayerId.Player1, EmptyObservationMemory.Instance);

            AssertMirroredCellRows(
                config.Match.BoardGen.Width, config.Match.BoardGen.Height, player0, player1);
            AssertMirroredUnitRows(state, player0, player1);
            AssertMirroredTemplateRows(state, player0, player1);
            Assert.Multiple(() =>
            {
                Assert.That(
                    (player0.Cells[7].Terrain, player0.Cells[7].Elevation,
                        player0.Cells[7].SelfDeploymentZone, player0.Cells[7].OpponentDeploymentZone),
                    Is.EqualTo((TerrainType.Forest, 4, true, false)));
                Assert.That(
                    (player0.Cells[12].Terrain, player0.Cells[12].Elevation,
                        player0.Cells[12].SelfDeploymentZone, player0.Cells[12].OpponentDeploymentZone),
                    Is.EqualTo((TerrainType.Water, 1, false, true)));
                Assert.That(player0.Cells[23].Controller,
                    Is.EqualTo(TacticalV3RelativeOwner.Self));
                Assert.That(player1.Cells[23].Controller,
                    Is.EqualTo(TacticalV3RelativeOwner.Opponent));
                Assert.That(player0.Cells[72].Controller,
                    Is.EqualTo(TacticalV3RelativeOwner.Opponent));
                Assert.That(player1.Cells[72].Controller,
                    Is.EqualTo(TacticalV3RelativeOwner.Self));
            });
        }

        [Test]
        public void IndependentOddQReflection_MatchesHandDerivedEvenAndOddColumnCases()
        {
            Assert.That(IndependentReflectCell(13, 9, new HexCoord(0, 0)),
                Is.EqualTo(new HexCoord(12, 2)));
            Assert.That(IndependentReflectCell(13, 9, new HexCoord(1, 0)),
                Is.EqualTo(new HexCoord(11, 3)));
            Assert.That(IndependentReflectCell(13, 9, new HexCoord(11, 3)),
                Is.EqualTo(new HexCoord(1, 0)));
        }

        [Test]
        public void LearnedDtoGraph_RejectsPresentationEngineAndNestedRawSeatIdentity()
        {
            Type[] unsafeGraphs =
            {
                typeof(LeakyNamePath),
                typeof(LeakyDisplayNamePath),
                typeof(LeakyEngineIdPath),
                typeof(LeakyUnitIdPath),
                typeof(LeakyPublicFieldPath),
                typeof(LeakyDirectSeatPath),
                typeof(LeakyNullableSeatPath),
                typeof(LeakyCollectionSeatPath),
                typeof(LeakyArraySeatPath),
                typeof(LeakyGenericSeatPath),
                typeof(LeakyNestedSeatPath),
            };
            foreach (Type unsafeGraph in unsafeGraphs)
                Assert.That(LearnedRowSurfaceIsSafe(unsafeGraph), Is.False, unsafeGraph.Name);

            Assert.That(LearnedRowSurfaceIsSafe(typeof(SafeLearnedRoot)), Is.True);

            Type[] learnedRoots =
            {
                typeof(TacticalV3Observation),
                typeof(TacticalV3Candidate),
                typeof(TacticalV3ProjectedDelta),
                typeof(TacticalV3TokenRef),
            };
            foreach (Type root in learnedRoots)
                Assert.That(LearnedRowSurfaceIsSafe(root), Is.True, root.FullName);

            // TacticalV3View is the environment envelope, not a learned feature root. Seat is
            // intentionally retained there for control-plane routing and excluded from this graph.
            Assert.That(typeof(TacticalV3View).GetProperty(nameof(TacticalV3View.Seat))!.PropertyType,
                Is.EqualTo(typeof(PlayerId)));
            Assert.That(learnedRoots, Does.Not.Contain(typeof(TacticalV3View)));
        }

        private static void AssertMirroredCellRows(
            int width,
            int height,
            TacticalV3Observation player0,
            TacticalV3Observation player1)
        {
            for (int row = 0; row < player0.Cells.Count; row++)
                AssertMirroredCell(width, height, player0.Cells[row], player1.Cells[row], row);
        }

        private static void AssertMirroredCell(
            int width, int height, TacticalV3CellToken left, TacticalV3CellToken right, int row)
        {
            HexCoord reflected = IndependentReflectCell(width, height, new HexCoord(left.Q, left.R));
            Assert.Multiple(() =>
            {
                Assert.That((right.Q, right.R), Is.EqualTo((reflected.Q, reflected.R)));
                Assert.That(right.Terrain, Is.EqualTo(left.Terrain));
                Assert.That(right.Elevation, Is.EqualTo(left.Elevation));
                Assert.That(right.SelfDeploymentZone, Is.EqualTo(left.OpponentDeploymentZone));
                Assert.That(right.OpponentDeploymentZone, Is.EqualTo(left.SelfDeploymentZone));
                Assert.That(right.Controller, Is.EqualTo(SwapOwner(left.Controller)));
                Assert.That(right.IsBoundary, Is.EqualTo(left.IsBoundary));
                Assert.That(right.CurrentlyVisible, Is.EqualTo(left.CurrentlyVisible));
                Assert.That(right.PreviouslyObserved, Is.EqualTo(left.PreviouslyObserved));
            });
        }

        private static HexCoord IndependentReflectCell(int width, int height, HexCoord cell)
        {
            int sourceColumn = cell.Q;
            int sourceOffsetRow = cell.R + (sourceColumn - (sourceColumn & 1)) / 2;
            int reflectedColumn = width - 1 - sourceColumn;
            int reflectedOffsetRow = height - 1 - sourceOffsetRow;
            int reflectedAxialRow =
                reflectedOffsetRow - (reflectedColumn - (reflectedColumn & 1)) / 2;
            return new HexCoord(reflectedColumn, reflectedAxialRow);
        }

        private static GameState AsymmetricPerspectiveState(TacticalV2Layout layout, GameState source)
        {
            Tile[] tiles = source.Board.Tiles.Select(tile =>
                tile.Coord == layout.Cells[7] ? new Tile(tile.Coord, 4, TerrainType.Forest) :
                tile.Coord == layout.Cells[12] ? new Tile(tile.Coord, 1, TerrainType.Water) :
                tile.Coord == layout.Cells[39] ? new Tile(tile.Coord, 3, TerrainType.Rough) :
                tile).ToArray();
            var control = new System.Collections.Generic.Dictionary<HexCoord, PlayerId>
            {
                [layout.Cells[23]] = PlayerId.Player0,
                [layout.Cells[72]] = PlayerId.Player1,
            };
            var board = new Board(
                tiles,
                new[] { layout.Cells[7], layout.Cells[39] },
                new[] { layout.Cells[12], layout.Cells[83] },
                control);
            return new GameState(
                board, source.Config, source.Players, source.ActivePlayer, source.Round,
                source.NextEntityId, source.IsGameOver, source.Winner, source.MovedUnitIds,
                source.AttackedUnitIds, source.MovementSpent);
        }

        private static TacticalV3RelativeOwner? SwapOwner(TacticalV3RelativeOwner? owner) =>
            !owner.HasValue ? (TacticalV3RelativeOwner?)null :
            owner.Value == TacticalV3RelativeOwner.Self
                ? TacticalV3RelativeOwner.Opponent
                : TacticalV3RelativeOwner.Self;

        private static void AssertMirroredUnitRows(
            GameState state, TacticalV3Observation player0, TacticalV3Observation player1)
        {
            int player0Units = state.Player(PlayerId.Player0).UnitsOnBoard.Count(unit => unit.IsAlive);
            int player1Units = state.Player(PlayerId.Player1).UnitsOnBoard.Count(unit => unit.IsAlive);
            for (int row = 0; row < player0Units; row++)
                AssertSameUnitFacts(player0.Units[row], player1.Units[player1Units + row],
                    TacticalV3RelativeOwner.Opponent);
            for (int row = 0; row < player1Units; row++)
                AssertSameUnitFacts(player0.Units[player0Units + row], player1.Units[row],
                    TacticalV3RelativeOwner.Self);
        }

        private static void AssertSameUnitFacts(
            TacticalV3UnitToken expected,
            TacticalV3UnitToken actual,
            TacticalV3RelativeOwner expectedOwner)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Owner, Is.EqualTo(expectedOwner));
                Assert.That(actual.CurrentHp, Is.EqualTo(expected.CurrentHp));
                Assert.That(actual.MaxHp, Is.EqualTo(expected.MaxHp));
                Assert.That(actual.Cell, Is.EqualTo(expected.Cell));
                Assert.That(actual.Elevation, Is.EqualTo(expected.Elevation));
                Assert.That(actual.Moved, Is.EqualTo(expected.Moved));
                Assert.That(actual.Attacked, Is.EqualTo(expected.Attacked));
                Assert.That(actual.HorizontalMovementSpent, Is.EqualTo(expected.HorizontalMovementSpent));
                Assert.That(actual.VerticalMovementSpent, Is.EqualTo(expected.VerticalMovementSpent));
                Assert.That(actual.PointCost, Is.EqualTo(expected.PointCost));
                Assert.That(actual.DeployCost, Is.EqualTo(expected.DeployCost));
                Assert.That(actual.CurrentlyVisible, Is.EqualTo(expected.CurrentlyVisible));
            });
        }

        private static void AssertMirroredTemplateRows(
            GameState state, TacticalV3Observation player0, TacticalV3Observation player1)
        {
            int rows = state.Player(PlayerId.Player0).Barracks.Count;
            for (int row = 0; row < rows; row++)
            {
                AssertSameTemplateFacts(player0.Templates[row], player1.Templates[rows + row],
                    TacticalV3RelativeOwner.Opponent);
                AssertSameTemplateFacts(player0.Templates[rows + row], player1.Templates[row],
                    TacticalV3RelativeOwner.Self);
            }
        }

        private static void AssertSameTemplateFacts(
            TacticalV3TemplateToken expected,
            TacticalV3TemplateToken actual,
            TacticalV3RelativeOwner expectedOwner)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Owner, Is.EqualTo(expectedOwner));
                Assert.That(actual.PointCost, Is.EqualTo(expected.PointCost));
                Assert.That(actual.DeployCost, Is.EqualTo(expected.DeployCost));
                Assert.That(actual.IsFixed, Is.EqualTo(expected.IsFixed));
                Assert.That(actual.IsDeployable, Is.EqualTo(expected.IsDeployable));
            });
        }

        private static bool LearnedRowSurfaceIsSafe(Type rowType)
        {
            var forbiddenNames = new HashSet<string>(StringComparer.Ordinal)
            {
                "Name",
                "DisplayName",
                "EngineId",
                "UnitId",
            };
            var visited = new HashSet<Type>();
            var violations = new List<string>();

            void Visit(Type type, string path)
            {
                if (type == typeof(PlayerId))
                {
                    violations.Add(path + " uses raw PlayerId");
                    return;
                }

                Type? nullable = Nullable.GetUnderlyingType(type);
                if (nullable != null)
                {
                    Visit(nullable, path + "?");
                    return;
                }
                if (type.IsArray)
                {
                    Visit(type.GetElementType()!, path + "[]");
                    return;
                }
                if (type.IsGenericType)
                    foreach (Type argument in type.GetGenericArguments())
                        Visit(argument, path + "<" + argument.Name + ">");

                if (type.IsPrimitive || type.IsEnum || type == typeof(string) ||
                    type == typeof(decimal) || type.IsGenericParameter)
                    return;
                if (type.Namespace != null &&
                    type.Namespace.StartsWith("System", StringComparison.Ordinal))
                    return;
                if (!visited.Add(type)) return;

                const BindingFlags publicInstance = BindingFlags.Instance | BindingFlags.Public;
                foreach (PropertyInfo property in type.GetProperties(publicInstance))
                {
                    string memberPath = path + "." + property.Name;
                    if (forbiddenNames.Contains(property.Name))
                        violations.Add(memberPath + " has forbidden learned member name");
                    Visit(property.PropertyType, memberPath);
                }
                foreach (FieldInfo field in type.GetFields(publicInstance))
                {
                    string memberPath = path + "." + field.Name;
                    if (forbiddenNames.Contains(field.Name))
                        violations.Add(memberPath + " has forbidden learned member name");
                    Visit(field.FieldType, memberPath);
                }
            }

            Visit(rowType, rowType.Name);
            return violations.Count == 0;
        }

        private sealed class LeakyNamePath { public int Name { get; } = 1; }
        private sealed class LeakyDisplayNamePath { public int DisplayName { get; } = 1; }
        private sealed class LeakyEngineIdPath { public int EngineId { get; } = 1; }
        private sealed class LeakyUnitIdPath { public int UnitId { get; } = 1; }
        private sealed class LeakyPublicFieldPath { public PlayerId Seat = PlayerId.Player0; }
        private sealed class LeakyDirectSeatPath { public PlayerId Seat { get; } = PlayerId.Player0; }
        private sealed class LeakyNullableSeatPath { public PlayerId? Seat { get; } = PlayerId.Player0; }
        private sealed class LeakyCollectionSeatPath
        {
            public System.Collections.Generic.IReadOnlyList<PlayerId> Seats { get; } =
                Array.Empty<PlayerId>();
        }
        private sealed class LeakyArraySeatPath { public PlayerId[] Seats { get; } = Array.Empty<PlayerId>(); }
        private sealed class LeakyGenericSeatPath
        {
            public GenericNode<PlayerId> Node { get; } = new GenericNode<PlayerId>();
        }
        private sealed class LeakyNestedSeatPath { public NestedSeatNode Node { get; } = new NestedSeatNode(); }
        private sealed class NestedSeatNode { public PlayerId Seat { get; } = PlayerId.Player0; }
        private sealed class GenericNode<T> { public T? Value { get; set; } }
        private sealed class SafeLearnedRoot
        {
            public SafeLearnedNode Node { get; } = new SafeLearnedNode();
            public int[] Values { get; } = Array.Empty<int>();
        }
        private sealed class SafeLearnedNode { public int OwnerClass { get; } = 1; }

        private static string AuthoritativeStateFingerprint(GameState state)
        {
            string tiles = string.Join(";", state.Board.Tiles.OrderBy(tile => tile.Coord.Q).ThenBy(tile => tile.Coord.R)
                .Select(tile => $"{tile.Coord.Q},{tile.Coord.R},{tile.Terrain},{tile.Elevation},{state.Board.Controller(tile.Coord)}"));
            string zones = string.Join(";", new[] { PlayerId.Player0, PlayerId.Player1 }.Select(player =>
                $"{player}:{string.Join(",", state.Board.DeploymentZone(player).OrderBy(cell => cell.Q).ThenBy(cell => cell.R))}"));
            string players = string.Join(";", state.Players.Select(player =>
                $"{player.Id},{player.Points},{player.DestroyedValue}|" +
                string.Join(",", player.Barracks.Select(template => $"{template.Name}:{StatsFingerprint(template.Stats)}")) + "|" +
                string.Join(",", player.UnitsOnBoard.Select(unit =>
                    $"{unit.Id},{unit.Owner},{unit.Cell.Q},{unit.Cell.R},{unit.Elevation},{unit.CurrentHp},{unit.Name}:{StatsFingerprint(unit.Stats)}")) + "|" +
                string.Join(",", player.Generators.Select(generator =>
                    $"{generator.Id},{generator.Owner},{generator.Cell.Q},{generator.Cell.R},{generator.Elevation},{generator.CurrentHp},{generator.Strength}"))));
            string movement = string.Join(";", state.MovementSpent.OrderBy(item => item.Key)
                .Select(item => $"{item.Key}:{item.Value.H},{item.Value.V}"));
            return $"{tiles}|{zones}|{players}|{state.ActivePlayer}|{state.Round}|{state.NextEntityId}|" +
                $"{state.IsGameOver}|{state.Winner}|{string.Join(",", state.MovedUnitIds.OrderBy(id => id))}|" +
                $"{string.Join(",", state.AttackedUnitIds.OrderBy(id => id))}|{movement}";
        }

        private static string StatsFingerprint(UnitStats stats) => string.Join(",", new[]
        {
            stats.Health, stats.Damage, stats.Defense, stats.Movement, stats.VerticalMovement,
            stats.Range, stats.RangeArc, stats.Vision, stats.VisionArc,
        });

        private static TacticalV3RuleToken Rule(TacticalV3Observation observation, TacticalV3RuleKind kind) =>
            observation.Rules.Single(rule => rule.Kind == kind);
    }
}
