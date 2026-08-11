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

        [Test]
        public void Project_EveryLegalCandidateMatchesIndependentTransitionFromRecreatedState()
        {
            GameState baseline = OrderingState();
            TacticalV3Observation observation = ProjectionObservation(baseline, PlayerId.Player0);
            var source = new TacticalV3LegalCandidateSource(
                new FixedObservationSource(observation), TacticalV3Fixtures.ExperimentalCapacity());
            TacticalV3DecisionFrame frame = source.CreateFrame(
                baseline, PlayerId.Player0, EmptyObservationMemory.Instance, 29);

            Assert.That(frame.Candidates.Select(candidate => candidate.Kind).Distinct(), Is.EquivalentTo(new[]
            {
                TacticalV3CandidateKind.Attack,
                TacticalV3CandidateKind.Move,
                TacticalV3CandidateKind.Deploy,
                TacticalV3CandidateKind.EndTurn,
            }));
            foreach (TacticalV3Candidate candidate in frame.Candidates)
            {
                GameState independent = OrderingState();
                TacticalV3Observation independentObservation =
                    ProjectionObservation(independent, PlayerId.Player0);
                AssertCandidateReferencesValid(candidate, independentObservation);
                Command command = Reconstruct(candidate, independent, independentObservation);
                Result applied = GameEngine.Apply(independent, command);

                Assert.That(applied.Success, Is.True, candidate.CandidateId.ToString());
                AssertProjectionMatchesAuthoritativeTransition(
                    candidate, command, independent, applied.NewState);
            }
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
