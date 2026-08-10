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

            var resolved = new HashSet<Command>(frame.Candidates.Select(candidate =>
                f.Resolver.Resolve(frame, candidate.CandidateId, f.State)));
            Assert.That(resolved.SetEquals(LegalMoves.For(f.State)), Is.True);
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
                f.Resolver.Resolve(frame, 0, changed));
        }

        [Test]
        public void Resolver_RejectsInvalidCandidateIds()
        {
            TacticalV3Fixture f = TacticalV3Fixture.Standard(seed: 23);
            TacticalV3DecisionFrame frame = f.Candidates.CreateFrame(
                f.State, f.State.ActivePlayer, EmptyObservationMemory.Instance, 7);

            Assert.Throws<ArgumentOutOfRangeException>(() => f.Resolver.Resolve(frame, -1, f.State));
            Assert.Throws<ArgumentOutOfRangeException>(() =>
                f.Resolver.Resolve(frame, frame.Candidates.Count, f.State));
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

            Assert.Throws<InvalidOperationException>(() =>
                new TacticalV3ActionResolver().Resolve(frame, 8, 0, f.State));
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
