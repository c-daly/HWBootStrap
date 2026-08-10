using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV3CandidateTests
    {
        [Test]
        public void CreateFrame_EnumeratesEverySupportedLegalCommandAsAnOpaqueDecisionLocalRow()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);

            Assert.That(frame.DecisionId, Is.EqualTo(7));
            Assert.That(frame.Candidates.Select(candidate => candidate.CandidateId),
                Is.EqualTo(Enumerable.Range(0, frame.Candidates.Count)));
            Assert.That(frame.Candidates.Select(candidate => candidate.DecisionId), Is.All.EqualTo(7));
            Assert.That(frame.Candidates.Select(candidate => candidate.Kind), Is.Ordered);

            Command[] resolved = frame.Candidates.Select(candidate =>
                f.Resolver.Resolve(frame, frame.DecisionId, candidate.CandidateId, f.State)).ToArray();
            Command[] expected = ExpectedCommands(f.State);
            Assert.That(resolved, Is.EqualTo(expected));
            Assert.That(resolved.Length, Is.EqualTo(expected.Length));
        }

        [Test]
        public void Project_UsesAuthoritativeTransitionWithoutMutatingTheSourceState()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);

            TacticalV3Candidate move = frame.Candidates.First(candidate =>
                candidate.Kind == TacticalV3CandidateKind.Move);

            Assert.That(move.Projection.SourceCell.HasValue, Is.True);
            Assert.That(move.Projection.DestinationCell.HasValue, Is.True);
            Assert.That(move.Projection.HorizontalMovementSpent, Is.GreaterThan(0));
            Assert.That(move.Projection.PointsDelta, Is.Zero);
            Assert.That(move.Projection.RoundDelta, Is.Zero);
            Assert.That(f.State.MovementSpent, Is.Empty);
        }

        [Test]
        public void Resolver_RejectsStaleFrameInsteadOfEndingTurn()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);
            GameState changed = GameEngine.Apply(f.State,
                new EndTurn(f.State.ActivePlayer)).NewState;

            Assert.Throws<InvalidOperationException>(() =>
                f.Resolver.Resolve(frame, frame.DecisionId, 0, changed));
        }

        [Test]
        public void Resolver_RejectsInvalidCandidateIds()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);

            Assert.Throws<ArgumentOutOfRangeException>(() =>
                f.Resolver.Resolve(frame, frame.DecisionId, -1, f.State));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                f.Resolver.Resolve(frame, frame.DecisionId, frame.Candidates.Count, f.State));
        }

        [Test]
        public void CreateFrame_RejectsCandidateCapacityOverflowInsteadOfTrimming()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            var source = new TacticalV3LegalCandidateSource(f.Source,
                new TacticalV3CandidateProjector(), TacticalV3Fixtures.ExperimentalCapacity(maxCandidates: 1));

            Assert.Throws<InvalidOperationException>(() => source.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7));
        }
        [Test]
        public void Project_AttackReportsFactualDamageLethalityAndBountyWithoutMutatingState()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
            });
            var attacker = new Unit(1, PlayerId.Player0,
                TestStates.Stats(health: 1, damage: 5, range: 1, vision: 1), new HexCoord(0, 0), 0);
            var victim = new Unit(2, PlayerId.Player1, TestStates.Cost(3), new HexCoord(1, 0), 0);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0, unitsOnBoard: new[] { attacker }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[] { victim }),
            }, PlayerId.Player0, round: 2, nextEntityId: 3);

            TacticalV3ProjectedDelta delta = new TacticalV3CandidateProjector().Project(
                state, PlayerId.Player0, new AttackUnit(PlayerId.Player0, 1, 2), Observation(2, 2, 0));

            Assert.Multiple(() =>
            {
                Assert.That(delta.SourceCell, Is.EqualTo(
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, 0)));
                Assert.That(delta.TargetHpDelta, Is.EqualTo(-3));
                Assert.That(delta.Damage, Is.EqualTo(3));
                Assert.That(delta.IsLethal, Is.True);
                Assert.That(delta.BountyDelta, Is.EqualTo(1));
                Assert.That(delta.PointsDelta, Is.EqualTo(1));
                Assert.That(delta.IsTerminal, Is.True);
                Assert.That(state.Player(PlayerId.Player1).UnitsOnBoard.Single().CurrentHp, Is.EqualTo(3));
            });
        }

        [Test]
        public void Project_DeployReportsTemplateCellAndPointsDelta()
        {
            GameState baseState = TestStates.Fresh(p0Points: 12);
            var player0 = new PlayerState(PlayerId.Player0, 12,
                barracks: new[] { new UnitTemplate("", TestStates.Cost(3)) });
            var state = new GameState(baseState.Board, baseState.Config, new[]
            {
                player0,
                baseState.Player(PlayerId.Player1),
            }, PlayerId.Player0, 1, 1);

            TacticalV3ProjectedDelta delta = new TacticalV3CandidateProjector().Project(
                state, PlayerId.Player0, new DeployUnit(PlayerId.Player0, 0, new HexCoord(0, 0)),
                Observation(2, 0, 1));

            Assert.Multiple(() =>
            {
                Assert.That(delta.Template, Is.EqualTo(
                    new TacticalV3TokenRef(TacticalV3TableKind.Templates, 0)));
                Assert.That(delta.DestinationCell, Is.EqualTo(
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, 0)));
                Assert.That(delta.PointsDelta, Is.EqualTo(-3));
                Assert.That(state.Player(PlayerId.Player0).Points, Is.EqualTo(12));
            });
        }

        [Test]
        public void CreateFrame_RejectsUnsupportedLegalCommandInsteadOfDroppingIt()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
            });
            var unit = new Unit(1, PlayerId.Player0, TestStates.Stats(), new HexCoord(0, 0), 0);
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 100, unitsOnBoard: new[] { unit }),
                new PlayerState(PlayerId.Player1, 0),
            }, PlayerId.Player0, 1, 2);
            Assert.That(LegalMoves.For(state).OfType<CaptureHex>(), Is.Not.Empty);

            var source = new TacticalV3LegalCandidateSource(
                new FixedObservationSource(Observation(1, 1, 0)),
                TacticalV3Fixtures.ExperimentalCapacity());
            Assert.Throws<NotSupportedException>(() =>
                source.CreateFrame(state, PlayerId.Player0, EmptyObservationMemory.Instance, 7));
        }

        [Test]
        public void Resolver_RejectsWrongDecisionIdentity()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);

            IActionResolver resolver = f.Resolver;
            Assert.Throws<InvalidOperationException>(() =>
                resolver.Resolve(frame, 8, 0, f.State));
        }

        [Test]
        public void CreateFrame_OrdersWithinKindsByDecisionLocalRows()
        {
            GameState state = OrderingState();
            var source = new TacticalV3LegalCandidateSource(
                new FixedObservationSource(Observation(6, 4, 2)),
                TacticalV3Fixtures.ExperimentalCapacity());
            TacticalV3DecisionFrame frame = source.CreateFrame(
                state, PlayerId.Player0, EmptyObservationMemory.Instance, 11);

            Command[] resolved = frame.Candidates.Select(candidate =>
                new TacticalV3ActionResolver().Resolve(
                    frame, frame.DecisionId, candidate.CandidateId, state)).ToArray();

            Assert.That(resolved, Is.EqualTo(ExpectedCommands(state)));
            Assert.That(frame.Candidates.Where(candidate => candidate.Kind == TacticalV3CandidateKind.Move)
                .Select(candidate => candidate.Actor!.Value.Row + ":" + candidate.Cell!.Value.Row),
                Is.EqualTo(new[] { "0:0", "1:3" }));
            Assert.That(frame.Candidates.Where(candidate => candidate.Kind == TacticalV3CandidateKind.Deploy)
                .Select(candidate => candidate.Template!.Value.Row + ":" + candidate.Cell!.Value.Row),
                Is.EqualTo(new[] { "0:0", "0:3", "1:0", "1:3" }));
        }

        [Test]
        public void CandidateSurface_ExposesOnlyApprovedDecisionLocalFacts()
        {
            AssertPublicSurface(typeof(TacticalV3Candidate), CandidatePropertyContract());
            AssertPublicSurface(typeof(TacticalV3ProjectedDelta), ProjectedDeltaPropertyContract());
        }

        [Test]
        public void PublicSurfaceGuard_RejectsCommandSubtypesAndUnexpectedProperties()
        {
            Assert.That(LegacySurfaceGuardAccepts(typeof(LeakyCandidateSurface)), Is.True);
            Assert.Throws<AssertionException>(() =>
                AssertPublicSurface(typeof(LeakyCandidateSurface), CandidatePropertyContract()));
        }

        private static IReadOnlyDictionary<string, Type> CandidatePropertyContract() =>
            new Dictionary<string, Type>
            {
                ["CandidateId"] = typeof(int),
                ["DecisionId"] = typeof(long),
                ["Kind"] = typeof(TacticalV3CandidateKind),
                ["Actor"] = typeof(TacticalV3TokenRef?),
                ["Target"] = typeof(TacticalV3TokenRef?),
                ["Template"] = typeof(TacticalV3TokenRef?),
                ["Cell"] = typeof(TacticalV3TokenRef?),
                ["Projection"] = typeof(TacticalV3ProjectedDelta),
            };

        private static IReadOnlyDictionary<string, Type> ProjectedDeltaPropertyContract() =>
            new Dictionary<string, Type>
            {
                ["SourceCell"] = typeof(TacticalV3TokenRef?),
                ["DestinationCell"] = typeof(TacticalV3TokenRef?),
                ["Template"] = typeof(TacticalV3TokenRef?),
                ["Target"] = typeof(TacticalV3TokenRef?),
                ["HorizontalMovementSpent"] = typeof(int),
                ["VerticalMovementSpent"] = typeof(int),
                ["TargetHpDelta"] = typeof(int),
                ["Damage"] = typeof(int),
                ["IsLethal"] = typeof(bool),
                ["BountyDelta"] = typeof(int),
                ["PointsDelta"] = typeof(int),
                ["RoundDelta"] = typeof(int),
                ["IsTerminal"] = typeof(bool),
            };

        private static void AssertPublicSurface(
            Type surfaceType, IReadOnlyDictionary<string, Type> expectedProperties)
        {
            System.Reflection.PropertyInfo[] properties = surfaceType.GetProperties(
                System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.Public);
            Assert.That(properties.Select(property => property.Name).OrderBy(name => name),
                Is.EqualTo(expectedProperties.Keys.OrderBy(name => name)));

            foreach (System.Reflection.PropertyInfo property in properties)
            {
                Assert.That(expectedProperties.TryGetValue(property.Name, out Type? expectedType), Is.True,
                    surfaceType.Name + "." + property.Name);
                Assert.That(property.PropertyType, Is.EqualTo(expectedType),
                    surfaceType.Name + "." + property.Name);

                Type unwrapped = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
                Assert.That(typeof(Command).IsAssignableFrom(unwrapped), Is.False,
                    surfaceType.Name + "." + property.Name);
                Assert.That(unwrapped, Is.Not.EqualTo(typeof(PlayerId)),
                    surfaceType.Name + "." + property.Name);
            }
        }


        private static bool LegacySurfaceGuardAccepts(Type surfaceType)
        {
            string[] forbiddenNames =
            {
                "EngineId", "UnitId", "Name", "DisplayName",
            };

            foreach (System.Reflection.PropertyInfo property in surfaceType.GetProperties())
            {
                Type propertyType = Nullable.GetUnderlyingType(property.PropertyType) ?? property.PropertyType;
                if (forbiddenNames.Contains(property.Name) ||
                    propertyType == typeof(Command) ||
                    propertyType == typeof(PlayerId))
                    return false;
            }
            return true;
        }

        private sealed class LeakyCandidateSurface
        {
            public AttackUnit Attack { get; } = new AttackUnit(PlayerId.Player0, 1, 2);
            public int UnexpectedFutureProperty { get; } = 1;
        }


        private static Command[] ExpectedCommands(GameState state)
        {
            var unitRows = new Dictionary<int, int>();
            int unitRow = 0;
            foreach (Unit unit in state.Player(PlayerId.Player0).UnitsOnBoard
                .Where(unit => unit.IsAlive).OrderBy(unit => unit.Id))
                unitRows.Add(unit.Id, unitRow++);
            foreach (Unit unit in state.Player(PlayerId.Player1).UnitsOnBoard
                .Where(unit => unit.IsAlive).OrderBy(unit => unit.Id))
                unitRows.Add(unit.Id, unitRow++);

            var cellRows = new Dictionary<HexCoord, int>();
            int cellRow = 0;
            foreach (Tile tile in state.Board.Tiles)
                cellRows.Add(tile.Coord, cellRow++);

            return LegalMoves.For(state)
                .OrderBy(CommandKind)
                .ThenBy(command => PrimaryRow(command, unitRows))
                .ThenBy(command => SecondaryRow(command, unitRows, cellRows))
                .ToArray();
        }

        private static int CommandKind(Command command) =>
            command is AttackUnit ? 0 :
            command is MoveUnit ? 1 :
            command is DeployUnit ? 2 :
            command is EndTurn ? 3 :
            throw new AssertionException("unexpected legal command " + command.GetType().Name);

        private static int PrimaryRow(Command command, IReadOnlyDictionary<int, int> unitRows) =>
            command is AttackUnit attack ? unitRows[attack.AttackerId] :
            command is MoveUnit move ? unitRows[move.UnitId] :
            command is DeployUnit deploy ? deploy.TemplateIndex : -1;

        private static int SecondaryRow(
            Command command,
            IReadOnlyDictionary<int, int> unitRows,
            IReadOnlyDictionary<HexCoord, int> cellRows) =>
            command is AttackUnit attack ? unitRows[attack.TargetId] :
            command is MoveUnit move ? cellRows[move.Dest] :
            command is DeployUnit deploy ? cellRows[deploy.Cell] : -1;

        private static GameState OrderingState()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(3, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(5, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(4, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(2, 0), 0, TerrainType.Plains),
            }, zone0: new[] { new HexCoord(0, 0), new HexCoord(3, 0) });
            UnitStats stats = TestStates.Stats(health: 2, damage: 1, movement: 1, range: 5, vision: 5);
            return new GameState(board, GameConfig.Default(captureCost: int.MaxValue), new[]
            {
                new PlayerState(PlayerId.Player0, 20,
                    barracks: new[]
                    {
                        new UnitTemplate("", TestStates.Cost(3)),
                        new UnitTemplate("", TestStates.Cost(4)),
                    },
                    unitsOnBoard: new[]
                    {
                        new Unit(20, PlayerId.Player0, stats, new HexCoord(1, 0), 0),
                        new Unit(10, PlayerId.Player0, stats, new HexCoord(2, 0), 0),
                    }),
                new PlayerState(PlayerId.Player1, 0, unitsOnBoard: new[]
                {
                    new Unit(50, PlayerId.Player1, TestStates.Stats(health: 3), new HexCoord(5, 0), 0),
                    new Unit(40, PlayerId.Player1, TestStates.Stats(health: 3), new HexCoord(4, 0), 0),
                }),
            }, PlayerId.Player0, 1, 51);
        }


        private static TacticalV3Observation Observation(int cells, int units, int templates)
        {
            return new TacticalV3Observation(
                Enumerable.Range(0, cells).Select(index => new TacticalV3CellToken(
                    index, 0, TerrainType.Plains, 0, false, false, null, false, true, false)),
                Enumerable.Range(0, units).Select(index => new TacticalV3UnitToken(
                    TacticalV3RelativeOwner.Self, 1, 1,
                    new TacticalV3TokenRef(TacticalV3TableKind.Cells, 0), 0,
                    false, false, 0, 0, 1, 1, true)),
                Enumerable.Range(0, templates).Select(index => new TacticalV3TemplateToken(
                    TacticalV3RelativeOwner.Self, 3, 3, false, true)),
                Enumerable.Empty<TacticalV3CapabilityDefinition>(),
                Enumerable.Empty<TacticalV3CapabilityAllocationToken>(),
                Enumerable.Empty<TacticalV3RuleToken>(),
                Enumerable.Empty<TacticalV3MemoryToken>(),
                Enumerable.Empty<TacticalV3RelationToken>());
        }

        private sealed class FixedObservationSource : ISeatObservationSource
        {
            private readonly TacticalV3Observation _observation;

            public FixedObservationSource(TacticalV3Observation observation) => _observation = observation;

            public TacticalV3Observation Observe(
                GameState state, PlayerId seat, IObservationMemory memory) => _observation;
        }
    }
}
