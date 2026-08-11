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
        public void CreateFrame_CandidatesStayWithinIndependentUpperBoundAcrossProfilesSeedsAndUnitCapacity()
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
                var observations = new TacticalV3SeatObservationSource(config);
                var candidates = new TacticalV3LegalCandidateSource(
                    observations, new TacticalV3CandidateProjector(), config.Capacity);

                TacticalV3DecisionFrame frame = candidates.CreateFrame(
                    state, PlayerId.Player0, EmptyObservationMemory.Instance, 1);
                Assert.That(frame.Candidates.Count, Is.LessThanOrEqualTo(11656), profileId);
            }

            TacticalV3Config denseConfig = TacticalV3Fixtures.CapacityBoundConfig();
            GameState denseState = TacticalV3Fixtures.DenseCapacityState(denseConfig);
            var denseObservations = new TacticalV3SeatObservationSource(denseConfig);
            var denseCandidates = new TacticalV3LegalCandidateSource(
                denseObservations, new TacticalV3CandidateProjector(), denseConfig.Capacity);
            TacticalV3DecisionFrame denseFrame = denseCandidates.CreateFrame(
                denseState, PlayerId.Player0, EmptyObservationMemory.Instance, 2);

            Assert.Multiple(() =>
            {
                Assert.That(denseFrame.Observation.Units, Has.Count.EqualTo(64));
                Assert.That(denseFrame.Candidates.Count, Is.GreaterThan(496));
                Assert.That(denseFrame.Candidates.Count, Is.LessThanOrEqualTo(11656));
                Assert.That(denseFrame.Candidates.Any(
                    candidate => candidate.Kind == TacticalV3CandidateKind.Deploy), Is.True);
            });
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
            GameState state = AsymmetricOrderingState();
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

        [Test]
        public void EveryLegalPublicCandidateMatchesHiddenResolverCommandAndProjectedSuccessor()
        {
            GameState baseline = AsymmetricOrderingState();
            TacticalV3Observation observation = ProjectionObservation(baseline, PlayerId.Player0);
            var source = new TacticalV3LegalCandidateSource(
                new FixedObservationSource(observation), TacticalV3Fixtures.ExperimentalCapacity());
            TacticalV3DecisionFrame frame = source.CreateFrame(
                baseline, PlayerId.Player0, EmptyObservationMemory.Instance, 29);
            var resolver = new TacticalV3ActionResolver();

            Assert.That(frame.Candidates.Select(candidate => candidate.Kind).Distinct(), Is.EquivalentTo(new[]
            {
                TacticalV3CandidateKind.Attack,
                TacticalV3CandidateKind.Move,
                TacticalV3CandidateKind.Deploy,
                TacticalV3CandidateKind.EndTurn,
            }));
            foreach (TacticalV3Candidate candidate in frame.Candidates)
            {
                Command hiddenCommand = resolver.Resolve(
                    frame, frame.DecisionId, candidate.CandidateId, baseline);
                GameState hiddenBefore = AsymmetricOrderingState();
                GameState publicBefore = AsymmetricOrderingState();
                TacticalV3Observation publicObservation =
                    ProjectionObservation(publicBefore, PlayerId.Player0);
                AssertCandidateReferencesValid(candidate, publicObservation);
                Command publicCommand = Reconstruct(candidate, publicBefore, publicObservation);
                Result hiddenApplied = GameEngine.Apply(hiddenBefore, hiddenCommand);
                Result publicApplied = GameEngine.Apply(publicBefore, publicCommand);

                Assert.That(hiddenApplied.Success, Is.True, candidate.CandidateId.ToString());
                Assert.That(publicApplied.Success, Is.True, candidate.CandidateId.ToString());
                AssertCompleteGameStatesEqual(hiddenApplied.NewState, publicApplied.NewState);
                AssertProjectionMatchesAuthoritativeTransition(
                    candidate, publicCommand, publicBefore, hiddenApplied.NewState);
            }
        }

        [Test]
        public void CompleteSuccessorOracle_RejectsDistinctLegalDestinations()
        {
            GameState baseline = AsymmetricOrderingState();
            TacticalV3Observation observation = ProjectionObservation(baseline, PlayerId.Player0);
            var source = new TacticalV3LegalCandidateSource(
                new FixedObservationSource(observation), TacticalV3Fixtures.ExperimentalCapacity());
            TacticalV3DecisionFrame frame = source.CreateFrame(
                baseline, PlayerId.Player0, EmptyObservationMemory.Instance, 31);
            TacticalV3Candidate[] moves = frame.Candidates
                .Where(candidate => candidate.Kind == TacticalV3CandidateKind.Move)
                .Take(2)
                .ToArray();
            Assert.That(moves, Has.Length.EqualTo(2));

            Result first = GameEngine.Apply(
                AsymmetricOrderingState(),
                Reconstruct(moves[0], AsymmetricOrderingState(), observation));
            Result second = GameEngine.Apply(
                AsymmetricOrderingState(),
                Reconstruct(moves[1], AsymmetricOrderingState(), observation));

            Assert.That(() => AssertCompleteGameStatesEqual(first.NewState, second.NewState),
                Throws.InstanceOf<MultipleAssertException>());
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

        private static TacticalV3Observation ProjectionObservation(GameState state, PlayerId seat) =>
            new TacticalV3Observation(
                ProjectionCells(state),
                ProjectionUnits(state, seat),
                ProjectionTemplates(state, seat),
                Enumerable.Empty<TacticalV3CapabilityDefinition>(),
                Enumerable.Empty<TacticalV3CapabilityAllocationToken>(),
                Enumerable.Empty<TacticalV3RuleToken>(),
                Enumerable.Empty<TacticalV3MemoryToken>(),
                Enumerable.Empty<TacticalV3RelationToken>());

        private static IEnumerable<TacticalV3CellToken> ProjectionCells(GameState state) =>
            state.Board.Tiles.Select(tile => new TacticalV3CellToken(
                tile.Coord.Q, tile.Coord.R, tile.Terrain, tile.Elevation,
                false, false, null, false, true, false));

        private static IEnumerable<Unit> ProjectionUnitRows(GameState state, PlayerId seat) =>
            state.Player(seat).UnitsOnBoard.Where(unit => unit.IsAlive).OrderBy(unit => unit.Id)
                .Concat(state.Opponent(seat).UnitsOnBoard.Where(unit => unit.IsAlive).OrderBy(unit => unit.Id));

        private static IEnumerable<TacticalV3UnitToken> ProjectionUnits(GameState state, PlayerId seat) =>
            ProjectionUnitRows(state, seat).Select(unit => ProjectionUnit(state, seat, unit));

        private static TacticalV3UnitToken ProjectionUnit(GameState state, PlayerId seat, Unit unit)
        {
            (int horizontal, int vertical) = state.MovementSpent.TryGetValue(unit.Id, out var spent)
                ? spent
                : (0, 0);
            return new TacticalV3UnitToken(
                unit.Owner == seat ? TacticalV3RelativeOwner.Self : TacticalV3RelativeOwner.Opponent,
                unit.CurrentHp,
                unit.Stats.Health,
                new TacticalV3TokenRef(TacticalV3TableKind.Cells, CellRow(state, unit.Cell)),
                unit.Elevation,
                state.MovedUnitIds.Contains(unit.Id),
                state.AttackedUnitIds.Contains(unit.Id),
                horizontal,
                vertical,
                unit.Stats.PointCost,
                Economy.DeployCost(unit.Stats, state.Config),
                true);
        }

        private static IEnumerable<TacticalV3TemplateToken> ProjectionTemplates(
            GameState state, PlayerId seat) =>
            ProjectionTemplates(state, state.Player(seat), TacticalV3RelativeOwner.Self)
                .Concat(ProjectionTemplates(
                    state, state.Opponent(seat), TacticalV3RelativeOwner.Opponent));

        private static IEnumerable<TacticalV3TemplateToken> ProjectionTemplates(
            GameState state, PlayerState player, TacticalV3RelativeOwner owner) =>
            player.Barracks.Select((template, index) => new TacticalV3TemplateToken(
                owner,
                template.Stats.PointCost,
                Economy.DeployCost(template.Stats, state.Config),
                index < state.Config.FixedTemplateCount,
                player.Points >= Economy.DeployCost(template.Stats, state.Config)));

        private static int CellRow(GameState state, HexCoord cell) =>
            state.Board.Tiles.Select(tile => tile.Coord).ToList().IndexOf(cell);

        private static void AssertCandidateReferencesValid(
            TacticalV3Candidate candidate, TacticalV3Observation observation)
        {
            AssertReference(candidate.Actor, TacticalV3TableKind.Units, observation.Units.Count);
            AssertReference(candidate.Target, TacticalV3TableKind.Units, observation.Units.Count);
            AssertReference(candidate.Template, TacticalV3TableKind.Templates, observation.Templates.Count);
            AssertReference(candidate.Cell, TacticalV3TableKind.Cells, observation.Cells.Count);
            AssertReference(candidate.Projection.SourceCell,
                TacticalV3TableKind.Cells, observation.Cells.Count);
            AssertReference(candidate.Projection.DestinationCell,
                TacticalV3TableKind.Cells, observation.Cells.Count);
            AssertReference(candidate.Projection.Template,
                TacticalV3TableKind.Templates, observation.Templates.Count);
            AssertReference(candidate.Projection.Target,
                TacticalV3TableKind.Units, observation.Units.Count);
        }

        private static void AssertReference(
            TacticalV3TokenRef? reference, TacticalV3TableKind table, int rowCount)
        {
            if (!reference.HasValue) return;
            Assert.Multiple(() =>
            {
                Assert.That(reference.Value.Table, Is.EqualTo(table));
                Assert.That(reference.Value.Row, Is.GreaterThanOrEqualTo(0));
                Assert.That(reference.Value.Row, Is.LessThan(rowCount));
            });
        }

        private static Command Reconstruct(
            TacticalV3Candidate candidate,
            GameState state,
            TacticalV3Observation observation)
        {
            PlayerId seat = state.ActivePlayer;
            if (candidate.Kind == TacticalV3CandidateKind.Attack)
                return new AttackUnit(
                    seat,
                    UnitAt(state, seat, candidate.Actor!.Value.Row).Id,
                    UnitAt(state, seat, candidate.Target!.Value.Row).Id);
            if (candidate.Kind == TacticalV3CandidateKind.Move)
                return new MoveUnit(
                    seat,
                    UnitAt(state, seat, candidate.Actor!.Value.Row).Id,
                    CellAt(observation, candidate.Cell!.Value.Row));
            if (candidate.Kind == TacticalV3CandidateKind.Deploy)
                return new DeployUnit(
                    seat,
                    candidate.Template!.Value.Row,
                    CellAt(observation, candidate.Cell!.Value.Row));
            if (candidate.Kind == TacticalV3CandidateKind.EndTurn)
                return new EndTurn(seat);
            throw new ArgumentOutOfRangeException(nameof(candidate));
        }

        private static Unit UnitAt(GameState state, PlayerId seat, int row) =>
            ProjectionUnitRows(state, seat).ElementAt(row);

        private static HexCoord CellAt(TacticalV3Observation observation, int row) =>
            new HexCoord(observation.Cells[row].Q, observation.Cells[row].R);

        private static void AssertProjectionMatchesAuthoritativeTransition(
            TacticalV3Candidate candidate,
            Command command,
            GameState before,
            GameState after)
        {
            TacticalV3ProjectedDelta expected = ExpectedProjection(
                candidate, command, before, after);
            TacticalV3ProjectedDelta actual = candidate.Projection;
            Assert.Multiple(() =>
            {
                Assert.That(actual.SourceCell, Is.EqualTo(expected.SourceCell));
                Assert.That(actual.DestinationCell, Is.EqualTo(expected.DestinationCell));
                Assert.That(actual.Template, Is.EqualTo(expected.Template));
                Assert.That(actual.Target, Is.EqualTo(expected.Target));
                Assert.That(actual.HorizontalMovementSpent, Is.EqualTo(expected.HorizontalMovementSpent));
                Assert.That(actual.VerticalMovementSpent, Is.EqualTo(expected.VerticalMovementSpent));
                Assert.That(actual.TargetHpDelta, Is.EqualTo(expected.TargetHpDelta));
                Assert.That(actual.Damage, Is.EqualTo(expected.Damage));
                Assert.That(actual.IsLethal, Is.EqualTo(expected.IsLethal));
                Assert.That(actual.BountyDelta, Is.EqualTo(expected.BountyDelta));
                Assert.That(actual.PointsDelta, Is.EqualTo(expected.PointsDelta));
                Assert.That(actual.RoundDelta, Is.EqualTo(expected.RoundDelta));
                Assert.That(actual.IsTerminal, Is.EqualTo(expected.IsTerminal));
            });
        }

        private static void AssertCompleteGameStatesEqual(GameState expected, GameState actual)
        {
            AssertGameConfigsEqual(expected.Config, actual.Config);
            Assert.That(actual.Config.TurnPolicy.RemainingActions(actual),
                Is.EqualTo(expected.Config.TurnPolicy.RemainingActions(expected)));
            AssertBoardsEqual(expected.Board, actual.Board);
            Assert.That(actual.Players, Has.Count.EqualTo(expected.Players.Count));
            for (int player = 0; player < expected.Players.Count; player++)
                AssertPlayersEqual(expected.Players[player], actual.Players[player]);
            Assert.Multiple(() =>
            {
                Assert.That(actual.ActivePlayer, Is.EqualTo(expected.ActivePlayer));
                Assert.That(actual.Round, Is.EqualTo(expected.Round));
                Assert.That(actual.NextEntityId, Is.EqualTo(expected.NextEntityId));
                Assert.That(actual.IsGameOver, Is.EqualTo(expected.IsGameOver));
                Assert.That(actual.Winner, Is.EqualTo(expected.Winner));
                Assert.That(actual.MovedUnitIds.OrderBy(id => id),
                    Is.EqualTo(expected.MovedUnitIds.OrderBy(id => id)));
                Assert.That(actual.AttackedUnitIds.OrderBy(id => id),
                    Is.EqualTo(expected.AttackedUnitIds.OrderBy(id => id)));
                Assert.That(actual.MovementSpent.OrderBy(item => item.Key),
                    Is.EqualTo(expected.MovementSpent.OrderBy(item => item.Key)));
            });
        }

        private static void AssertBoardsEqual(Board expected, Board actual)
        {
            Tile[] expectedTiles = expected.Tiles.OrderBy(tile => tile.Coord.Q)
                .ThenBy(tile => tile.Coord.R).ToArray();
            Tile[] actualTiles = actual.Tiles.OrderBy(tile => tile.Coord.Q)
                .ThenBy(tile => tile.Coord.R).ToArray();
            Assert.That(actualTiles, Has.Length.EqualTo(expectedTiles.Length));
            for (int row = 0; row < expectedTiles.Length; row++)
            {
                Assert.Multiple(() =>
                {
                    Assert.That(actualTiles[row].Coord, Is.EqualTo(expectedTiles[row].Coord));
                    Assert.That(actualTiles[row].Terrain, Is.EqualTo(expectedTiles[row].Terrain));
                    Assert.That(actualTiles[row].Elevation, Is.EqualTo(expectedTiles[row].Elevation));
                    Assert.That(actual.Controller(actualTiles[row].Coord),
                        Is.EqualTo(expected.Controller(expectedTiles[row].Coord)));
                });
            }
            foreach (PlayerId player in new[] { PlayerId.Player0, PlayerId.Player1 })
                Assert.That(actual.DeploymentZone(player).OrderBy(cell => cell.Q).ThenBy(cell => cell.R),
                    Is.EqualTo(expected.DeploymentZone(player)
                        .OrderBy(cell => cell.Q).ThenBy(cell => cell.R)));
        }

        private static void AssertPlayersEqual(PlayerState expected, PlayerState actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Id, Is.EqualTo(expected.Id));
                Assert.That(actual.Points, Is.EqualTo(expected.Points));
                Assert.That(actual.DestroyedValue, Is.EqualTo(expected.DestroyedValue));
                Assert.That(actual.Barracks, Has.Count.EqualTo(expected.Barracks.Count));
                Assert.That(actual.UnitsOnBoard, Has.Count.EqualTo(expected.UnitsOnBoard.Count));
                Assert.That(actual.Generators, Has.Count.EqualTo(expected.Generators.Count));
            });
            for (int row = 0; row < expected.Barracks.Count; row++)
            {
                Assert.That(actual.Barracks[row].Name, Is.EqualTo(expected.Barracks[row].Name));
                AssertUnitStatsEqual(expected.Barracks[row].Stats, actual.Barracks[row].Stats);
            }
            for (int row = 0; row < expected.UnitsOnBoard.Count; row++)
            {
                Unit expectedUnit = expected.UnitsOnBoard[row];
                Unit actualUnit = actual.UnitsOnBoard[row];
                Assert.Multiple(() =>
                {
                    Assert.That(actualUnit.Id, Is.EqualTo(expectedUnit.Id));
                    Assert.That(actualUnit.Owner, Is.EqualTo(expectedUnit.Owner));
                    Assert.That(actualUnit.Cell, Is.EqualTo(expectedUnit.Cell));
                    Assert.That(actualUnit.Elevation, Is.EqualTo(expectedUnit.Elevation));
                    Assert.That(actualUnit.CurrentHp, Is.EqualTo(expectedUnit.CurrentHp));
                    Assert.That(actualUnit.Name, Is.EqualTo(expectedUnit.Name));
                });
                AssertUnitStatsEqual(expectedUnit.Stats, actualUnit.Stats);
            }
            for (int row = 0; row < expected.Generators.Count; row++)
            {
                Generator expectedGenerator = expected.Generators[row];
                Generator actualGenerator = actual.Generators[row];
                Assert.Multiple(() =>
                {
                    Assert.That(actualGenerator.Id, Is.EqualTo(expectedGenerator.Id));
                    Assert.That(actualGenerator.Owner, Is.EqualTo(expectedGenerator.Owner));
                    Assert.That(actualGenerator.Cell, Is.EqualTo(expectedGenerator.Cell));
                    Assert.That(actualGenerator.Elevation, Is.EqualTo(expectedGenerator.Elevation));
                    Assert.That(actualGenerator.CurrentHp, Is.EqualTo(expectedGenerator.CurrentHp));
                    Assert.That(actualGenerator.Strength, Is.EqualTo(expectedGenerator.Strength));
                });
            }
        }

        private static void AssertGameConfigsEqual(GameConfig expected, GameConfig actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.StartingPoints, Is.EqualTo(expected.StartingPoints));
                Assert.That(actual.BountyRate, Is.EqualTo(expected.BountyRate));
                Assert.That(actual.GeneratorCost, Is.EqualTo(expected.GeneratorCost));
                Assert.That(actual.GeneratorOutput, Is.EqualTo(expected.GeneratorOutput));
                Assert.That(actual.GeneratorHealth, Is.EqualTo(expected.GeneratorHealth));
                Assert.That(actual.DamageFloor, Is.EqualTo(expected.DamageFloor));
                Assert.That(actual.DmgHighGroundBonus, Is.EqualTo(expected.DmgHighGroundBonus));
                Assert.That(actual.RangeHighGroundBonus, Is.EqualTo(expected.RangeHighGroundBonus));
                Assert.That(actual.RoundCap, Is.EqualTo(expected.RoundCap));
                Assert.That(actual.DesignFee, Is.EqualTo(expected.DesignFee));
                Assert.That(actual.MaxDesignPointCost, Is.EqualTo(expected.MaxDesignPointCost));
                Assert.That(actual.FixedTemplateCount, Is.EqualTo(expected.FixedTemplateCount));
                Assert.That(actual.TemplateSlotCount, Is.EqualTo(expected.TemplateSlotCount));
                Assert.That(actual.DeployCostMultiplier, Is.EqualTo(expected.DeployCostMultiplier));
                Assert.That(actual.TurnPolicy.GetType(), Is.SameAs(expected.TurnPolicy.GetType()));
                Assert.That(actual.TurnPolicy.ActionsPerTurn,
                    Is.EqualTo(expected.TurnPolicy.ActionsPerTurn));
                Assert.That(actual.BiomesEnabled, Is.EqualTo(expected.BiomesEnabled));
                Assert.That(actual.WinConditions, Is.EqualTo(expected.WinConditions));
                Assert.That(actual.CaptureCost, Is.EqualTo(expected.CaptureCost));
                Assert.That(actual.EconomyWinThreshold, Is.EqualTo(expected.EconomyWinThreshold));
                Assert.That(actual.ScoreKills, Is.EqualTo(expected.ScoreKills));
                Assert.That(actual.ScorePoints, Is.EqualTo(expected.ScorePoints));
                Assert.That(actual.ScoreArmy, Is.EqualTo(expected.ScoreArmy));
                Assert.That(actual.ScoreTerritory, Is.EqualTo(expected.ScoreTerritory));
                Assert.That(actual.UpkeepFactor, Is.EqualTo(expected.UpkeepFactor));
                Assert.That(actual.CaptureFactor, Is.EqualTo(expected.CaptureFactor));
                Assert.That(actual.BuildFactor, Is.EqualTo(expected.BuildFactor));
                Assert.That(actual.TerritoryMode, Is.EqualTo(expected.TerritoryMode));
                Assert.That(actual.ClaimEndsTurn, Is.EqualTo(expected.ClaimEndsTurn));
                Assert.That(actual.BuildAnywhere, Is.EqualTo(expected.BuildAnywhere));
                Assert.That(actual.TerritoryIncome, Is.EqualTo(expected.TerritoryIncome));
                Assert.That(actual.GeneratorsEnabled, Is.EqualTo(expected.GeneratorsEnabled));
                Assert.That(actual.FogOfWar, Is.EqualTo(expected.FogOfWar));
                Assert.That(actual.PointDecay, Is.EqualTo(expected.PointDecay));
            });
            foreach (TerrainType terrain in Enum.GetValues(typeof(TerrainType)))
            {
                TerrainDef expectedTerrain = expected.Terrain(terrain);
                TerrainDef actualTerrain = actual.Terrain(terrain);
                Assert.Multiple(() =>
                {
                    Assert.That(actualTerrain.MoveCost, Is.EqualTo(expectedTerrain.MoveCost));
                    Assert.That(actualTerrain.Concealment, Is.EqualTo(expectedTerrain.Concealment));
                    Assert.That(actualTerrain.Defense, Is.EqualTo(expectedTerrain.Defense));
                    Assert.That(actualTerrain.Passable, Is.EqualTo(expectedTerrain.Passable));
                });
            }
        }

        private static void AssertUnitStatsEqual(UnitStats expected, UnitStats actual)
        {
            Assert.Multiple(() =>
            {
                Assert.That(actual.Health, Is.EqualTo(expected.Health));
                Assert.That(actual.Damage, Is.EqualTo(expected.Damage));
                Assert.That(actual.Defense, Is.EqualTo(expected.Defense));
                Assert.That(actual.Movement, Is.EqualTo(expected.Movement));
                Assert.That(actual.VerticalMovement, Is.EqualTo(expected.VerticalMovement));
                Assert.That(actual.Range, Is.EqualTo(expected.Range));
                Assert.That(actual.RangeArc, Is.EqualTo(expected.RangeArc));
                Assert.That(actual.Vision, Is.EqualTo(expected.Vision));
                Assert.That(actual.VisionArc, Is.EqualTo(expected.VisionArc));
            });
        }

        private static TacticalV3ProjectedDelta ExpectedProjection(
            TacticalV3Candidate candidate,
            Command command,
            GameState before,
            GameState after)
        {
            if (command is MoveUnit move)
                return ExpectedMove(candidate, move, before, after);
            if (command is AttackUnit attack)
                return ExpectedAttack(candidate, attack, before, after);
            if (command is DeployUnit)
                return Delta(before, after, command.Issuer,
                    destinationCell: candidate.Cell, template: candidate.Template);
            if (command is EndTurn)
                return Delta(before, after, command.Issuer);
            throw new ArgumentOutOfRangeException(nameof(command));
        }

        private static TacticalV3ProjectedDelta Delta(
            GameState before,
            GameState after,
            PlayerId seat,
            TacticalV3TokenRef? sourceCell = null,
            TacticalV3TokenRef? destinationCell = null,
            TacticalV3TokenRef? template = null,
            TacticalV3TokenRef? target = null,
            int horizontal = 0,
            int vertical = 0,
            int targetHpDelta = 0,
            int damage = 0,
            bool lethal = false,
            int bounty = 0) =>
            new TacticalV3ProjectedDelta(
                sourceCell, destinationCell, template, target,
                horizontal, vertical, targetHpDelta, damage, lethal, bounty,
                after.Player(seat).Points - before.Player(seat).Points,
                after.Round - before.Round,
                after.IsGameOver);

        private static TacticalV3ProjectedDelta ExpectedMove(
            TacticalV3Candidate candidate,
            MoveUnit move,
            GameState before,
            GameState after)
        {
            Unit actor = before.Player(move.Issuer).UnitsOnBoard.Single(unit => unit.Id == move.UnitId);
            (int beforeHorizontal, int beforeVertical) = Movement(before, move.UnitId);
            (int afterHorizontal, int afterVertical) = Movement(after, move.UnitId);
            return Delta(
                before,
                after,
                move.Issuer,
                sourceCell: new TacticalV3TokenRef(
                    TacticalV3TableKind.Cells, CellRow(before, actor.Cell)),
                destinationCell: candidate.Cell,
                horizontal: afterHorizontal - beforeHorizontal,
                vertical: afterVertical - beforeVertical);
        }

        private static (int Horizontal, int Vertical) Movement(GameState state, int unitId) =>
            state.MovementSpent.TryGetValue(unitId, out var spent)
                ? (spent.H, spent.V)
                : (0, 0);

        private static TacticalV3ProjectedDelta ExpectedAttack(
            TacticalV3Candidate candidate,
            AttackUnit attack,
            GameState before,
            GameState after)
        {
            Unit attacker = before.Player(attack.Issuer).UnitsOnBoard.Single(
                unit => unit.Id == attack.AttackerId);
            Unit target = before.Opponent(attack.Issuer).UnitsOnBoard.Single(
                unit => unit.Id == attack.TargetId);
            int afterHp = after.Opponent(attack.Issuer).UnitsOnBoard
                .Where(unit => unit.Id == attack.TargetId && unit.IsAlive)
                .Select(unit => unit.CurrentHp)
                .SingleOrDefault();
            bool lethal = afterHp == 0;
            int bounty = lethal
                ? after.Player(attack.Issuer).Points - before.Player(attack.Issuer).Points
                : 0;
            return Delta(
                before,
                after,
                attack.Issuer,
                sourceCell: new TacticalV3TokenRef(
                    TacticalV3TableKind.Cells, CellRow(before, attacker.Cell)),
                target: candidate.Target,
                targetHpDelta: afterHp - target.CurrentHp,
                damage: target.CurrentHp - afterHp,
                lethal: lethal,
                bounty: bounty);
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

        private static GameState AsymmetricOrderingState()
        {
            var board = new Board(new[]
            {
                new Tile(new HexCoord(3, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(1, 0), 1, TerrainType.Forest),
                new Tile(new HexCoord(5, 0), 3, TerrainType.Water),
                new Tile(new HexCoord(0, 0), 0, TerrainType.Plains),
                new Tile(new HexCoord(4, 0), 1, TerrainType.Rough),
                new Tile(new HexCoord(2, 0), 2, TerrainType.Forest),
            },
                zone0: new[] { new HexCoord(0, 0), new HexCoord(3, 0) },
                zone1: new[] { new HexCoord(4, 0), new HexCoord(5, 0) },
                control: new Dictionary<HexCoord, PlayerId>
                {
                    [new HexCoord(0, 0)] = PlayerId.Player0,
                    [new HexCoord(4, 0)] = PlayerId.Player1,
                });
            UnitStats actor20 = TestStates.Stats(
                health: 2, damage: 1, movement: 1, verticalMovement: 3,
                range: 5, rangeArc: 5, vision: 5, visionArc: 5);
            UnitStats actor10 = TestStates.Stats(
                health: 4, damage: 2, defense: 1, movement: 1, verticalMovement: 3,
                range: 4, rangeArc: 4, vision: 6, visionArc: 4);
            GameConfig config = GameConfig.Default(
                turnPolicy: new KActionsPolicy(6), captureCost: int.MaxValue,
                startingPoints: 17, damageFloor: 1);
            return new GameState(board, config, new[]
            {
                new PlayerState(PlayerId.Player0, 23,
                    barracks: new[]
                    {
                        new UnitTemplate("Scout", TestStates.Cost(3)),
                        new UnitTemplate("Bulwark", TestStates.Stats(
                            health: 2, defense: 2, movement: 1)),
                    },
                    unitsOnBoard: new[]
                    {
                        new Unit(20, PlayerId.Player0, actor20, new HexCoord(1, 0), 1, "Runner"),
                        new Unit(10, PlayerId.Player0, actor10, new HexCoord(2, 0), 2, "Archer"),
                    }, destroyedValue: 2),
                new PlayerState(PlayerId.Player1, 7, unitsOnBoard: new[]
                {
                    new Unit(50, PlayerId.Player1,
                        TestStates.Stats(health: 5, defense: 1), new HexCoord(5, 0), 3, "Tower"),
                    new Unit(40, PlayerId.Player1,
                        TestStates.Stats(health: 3), new HexCoord(4, 0), 1, "Guard").WithDamage(1),
                }, destroyedValue: 5),
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
