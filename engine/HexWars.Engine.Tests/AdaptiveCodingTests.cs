using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AdaptiveCodingTests
    {
        [Test]
        public void Layout_PinsActionRegionsAndPhaseGlobals()
        {
            var l = new AdaptiveLayout(AdaptiveEnvConfig.Default());

            Assert.That(l.CommandOffset, Is.EqualTo(0));
            Assert.That(l.CommandCount, Is.EqualTo(12));
            Assert.That(l.UnitOffset, Is.EqualTo(12));
            Assert.That(l.UnitCount, Is.EqualTo(24));
            Assert.That(l.TemplateOffset, Is.EqualTo(36));
            Assert.That(l.TemplateCount, Is.EqualTo(9));
            Assert.That(l.CellOffset, Is.EqualTo(45));
            Assert.That(l.StatOffset, Is.EqualTo(162));
            Assert.That(l.ValueOffset, Is.EqualTo(171));
            Assert.That(l.ActionCount, Is.EqualTo(182));
            Assert.That(l.ObservationChannels, Is.EqualTo(50));
            Assert.That(l.ObservationGlobals, Is.EqualTo(124));
            Assert.That(l.ObservationLength, Is.EqualTo(5974));
            Assert.That(Enum.GetValues(typeof(AdaptivePhase)).Length, Is.EqualTo(14));
        }

        [Test]
        public void Layout_NonDefaultDimensionsOnlyResizeCellRegionAndSpatialObservation()
        {
            var cfg = AdaptiveEnvConfig.Default();
            cfg.BoardGen = new BoardGenConfig(width: 5, height: 4);
            var l = new AdaptiveLayout(cfg);

            Assert.That(l.CellCount, Is.EqualTo(20));
            Assert.That(l.CellOffset, Is.EqualTo(45));
            Assert.That(l.StatOffset, Is.EqualTo(65));
            Assert.That(l.ValueOffset, Is.EqualTo(74));
            Assert.That(l.ActionCount, Is.EqualTo(85));
            Assert.That(l.ObservationLength, Is.EqualTo(1124));
        }

        [Test]
        public void Layout_GeometryIsUnaffectedByLaterConfigMutation()
        {
            var cfg = AdaptiveEnvConfig.Default();
            var layout = new AdaptiveLayout(cfg);

            cfg.MaxControllableUnits = 3;
            cfg.Templates = cfg.Templates.Take(2).ToArray();

            Assert.That(layout.UnitCount, Is.EqualTo(24));
            Assert.That(layout.TemplateCount, Is.EqualTo(9));
            Assert.That(layout.ActionCount, Is.EqualTo(182));
            Assert.That(layout.ObservationLength, Is.EqualTo(5974));
        }

        [Test]
        public void Layout_ExposesReadOnlyGeometryAndNoMutableConfig()
        {
            var layout = new AdaptiveLayout(AdaptiveEnvConfig.Default());

            Assert.That(typeof(AdaptiveLayout).GetProperty("Config"), Is.Null);
            Assert.That(() => ((ICollection<HexCoord>)layout.Cells).Add(new HexCoord(50, 50)),
                Throws.TypeOf<NotSupportedException>());
            Assert.That(() => ((IDictionary<HexCoord, int>)layout.CellIndex)
                    .Add(new HexCoord(50, 50), 999),
                Throws.TypeOf<NotSupportedException>());
            Assert.That(layout.ActionCount, Is.EqualTo(182));
        }

        [Test]
        public void Layout_RejectsActionRegionCountDriftFromAdaptiveV1()
        {
            var units = AdaptiveEnvConfig.Default();
            units.MaxControllableUnits = 25;
            var templates = AdaptiveEnvConfig.Default();
            templates.Templates = templates.Templates.Take(8).ToArray();

            Assert.That(() => new AdaptiveLayout(units),
                Throws.ArgumentException.With.Message.Contains("24 controllable unit slots"));
            Assert.That(() => new AdaptiveLayout(templates),
                Throws.ArgumentException.With.Message.Contains("9 template slots"));
        }

        [Test]
        public void Observe_HidesFoggedEnemyButKeepsFriendlyAndPublicTerrain()
        {
            var f = AdaptiveFixtures.GameWithHiddenEnemy();
            var obs = AdaptiveCoding.Observe(f.Game, f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);

            Assert.That(obs[f.Layout.EnemyUnitPlane(0) * f.Layout.CellCount + f.HiddenEnemyCell], Is.Zero);
            Assert.That(Enumerable.Range(0, 9).Sum(role =>
                obs[f.Layout.EnemyUnitPlane(role) * f.Layout.CellCount + f.HiddenEnemyCell]), Is.Zero);
            Assert.That(Enumerable.Range(0, 9).Sum(role =>
                obs[f.Layout.FriendlyUnitPlane(role) * f.Layout.CellCount + f.FriendlyCell]), Is.GreaterThan(0f));
            Assert.That(obs[f.Layout.ElevationPlane * f.Layout.CellCount + f.HiddenEnemyCell], Is.GreaterThanOrEqualTo(0f));
        }

        [Test]
        public void HiddenEnemyIdentityPointsAndStats_DoNotChangeObservationMaskOrMemory()
        {
            var f = AdaptiveFixtures.GameWithHiddenEnemy();
            var hidden = f.Game!.Player(PlayerId.Player1).UnitsOnBoard
                .First(u => f.Layout.CellIndex[u.Cell] == f.HiddenEnemyCell);
            var altered = new Unit(900, PlayerId.Player1,
                new UnitStats(8, 8, 8, 0, 0, 0, 0, 0, 0), hidden.Cell, hidden.Elevation, "Secret");
            var gameB = AdaptiveFixtures.WithEnemy(f.Game, altered, enemyPoints: 999);
            var stateA = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayRoot);
            var stateB = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayRoot);

            var obsA = AdaptiveCoding.Observe(f.Game, f.Setup, PlayerId.Player0, stateA, f.Layout, f.Slots);
            var obsB = AdaptiveCoding.Observe(gameB, f.Setup, PlayerId.Player0, stateB, f.Layout, f.Slots);
            var maskA = AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0, stateA, f.Layout, f.Slots);
            var maskB = AdaptiveCoding.Mask(gameB, f.Setup, PlayerId.Player0, stateB, f.Layout, f.Slots);

            Assert.That(obsB, Is.EqualTo(obsA));
            Assert.That(maskB, Is.EqualTo(maskA));
            Assert.That(stateB.SeenCells, Is.EquivalentTo(stateA.SeenCells));
            Assert.That(stateA.SeenCells, Does.Not.Contain(hidden.Cell));
        }

        [Test]
        public void HiddenEnemyPresenceAndLocation_DoNotChangeObservationOrGameplayMask()
        {
            var variants = AdaptiveFixtures.HiddenBlockerVariants();
            var baselineObservation = AdaptiveCoding.Observe(variants[0].Game, variants[0].Setup,
                PlayerId.Player0, variants[0].Decision, variants[0].Layout, variants[0].Slots);
            var baselineMask = AdaptiveCoding.Mask(variants[0].Game, variants[0].Setup,
                PlayerId.Player0, variants[0].Decision, variants[0].Layout, variants[0].Slots);

            var rawBaselineMoves = LegalMoves.For(variants[0].Game!).OfType<MoveUnit>()
                .Select(move => move.Dest).ToHashSet();
            var rawBlockedMoves = LegalMoves.For(variants[1].Game!).OfType<MoveUnit>()
                .Select(move => move.Dest).ToHashSet();
            var rawBaselineDeploys = LegalMoves.For(variants[0].Game!).OfType<DeployUnit>()
                .Select(deploy => deploy.Cell).ToHashSet();
            var rawBlockedDeploys = LegalMoves.For(variants[1].Game!).OfType<DeployUnit>()
                .Select(deploy => deploy.Cell).ToHashSet();
            Assert.That(rawBlockedMoves, Is.Not.EqualTo(rawBaselineMoves),
                "the authoritative movement list must actually exercise hidden occupancy");
            Assert.That(rawBlockedDeploys, Is.Not.EqualTo(rawBaselineDeploys),
                "the authoritative reinforcement list must actually exercise hidden occupancy");
            foreach (var variant in variants.Skip(1))
            {
                Assert.That(AdaptiveCoding.Observe(variant.Game, variant.Setup, PlayerId.Player0,
                    variant.Decision, variant.Layout, variant.Slots), Is.EqualTo(baselineObservation));
                Assert.That(AdaptiveCoding.Mask(variant.Game, variant.Setup, PlayerId.Player0,
                    variant.Decision, variant.Layout, variant.Slots), Is.EqualTo(baselineMask));

                var baselineMove = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayMoveCell);
                baselineMove.SelectUnit(0);
                var variantMove = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayMoveCell);
                variantMove.SelectUnit(0);
                Assert.That(AdaptiveCoding.Mask(variant.Game, variant.Setup, PlayerId.Player0,
                        variantMove, variant.Layout, variant.Slots),
                    Is.EqualTo(AdaptiveCoding.Mask(variants[0].Game, variants[0].Setup, PlayerId.Player0,
                        baselineMove, variants[0].Layout, variants[0].Slots)));

                var baselineDeploy = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.DeploymentCell);
                baselineDeploy.SelectTemplate(0);
                var variantDeploy = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.DeploymentCell);
                variantDeploy.SelectTemplate(0);
                Assert.That(AdaptiveCoding.Mask(variant.Game, variant.Setup, PlayerId.Player0,
                        variantDeploy, variant.Layout, variant.Slots),
                    Is.EqualTo(AdaptiveCoding.Mask(variants[0].Game, variants[0].Setup, PlayerId.Player0,
                        baselineDeploy, variants[0].Layout, variants[0].Slots)));
            }
        }

        [Test]
        public void DeploymentObservationAndMask_ReadOnlyOwnHiddenLedger()
        {
            var d = AdaptiveFixtures.Deployment(29);
            var cfg = AdaptiveEnvConfig.Default();
            var layout = new AdaptiveLayout(cfg);
            var slots = new AdaptiveUnitSlots(cfg.MaxControllableUnits);
            var beforeState = new AdaptiveDecisionState(PlayerId.Player0);
            var afterState = new AdaptiveDecisionState(PlayerId.Player0);
            var before = AdaptiveCoding.Observe(null, d.Deployment, PlayerId.Player0, beforeState, layout, slots);
            var beforeMask = AdaptiveCoding.Mask(null, d.Deployment, PlayerId.Player0, beforeState, layout, slots);

            Assert.That(d.Place(PlayerId.Player1, 4, d.FirstLegalCell(PlayerId.Player1)), Is.True);

            var after = AdaptiveCoding.Observe(null, d.Deployment, PlayerId.Player0, afterState, layout, slots);
            var afterMask = AdaptiveCoding.Mask(null, d.Deployment, PlayerId.Player0, afterState, layout, slots);
            Assert.That(after, Is.EqualTo(before));
            Assert.That(afterMask, Is.EqualTo(beforeMask));
        }

        [Test]
        public void PreviouslySeenPlanePersistsAfterCellLeavesCurrentVision()
        {
            var f = AdaptiveFixtures.GameWithHiddenEnemy();
            var game = f.Game!;
            var friendly = game.Player(PlayerId.Player0).UnitsOnBoard.First();
            int seenIndex = f.Layout.CellIndex[friendly.Cell];
            AdaptiveCoding.Observe(game, f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);

            var farCell = f.Layout.Cells.Last(c => game.Board.Contains(c));
            var moved = new Unit(friendly.Id, friendly.Owner, friendly.Stats, farCell,
                game.Board.TileAt(farCell).Elevation, friendly.Name);
            var p0 = game.Player(PlayerId.Player0);
            var players = game.Players.ToArray();
            players[0] = new PlayerState(PlayerId.Player0, p0.Points, p0.Barracks,
                new[] { moved }, p0.Generators, p0.DestroyedValue);
            var movedGame = new GameState(game.Board, game.Config, players, game.ActivePlayer, game.Round,
                game.NextEntityId);
            var obs = AdaptiveCoding.Observe(movedGame, f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);

            Assert.That(obs[f.Layout.PreviouslySeenPlane * f.Layout.CellCount + seenIndex], Is.EqualTo(1f));
        }

        [Test]
        public void EveryNonRootPhase_AlwaysOffersCancel()
        {
            foreach (AdaptivePhase phase in Enum.GetValues(typeof(AdaptivePhase)))
            {
                if (phase == AdaptivePhase.GameplayRoot || phase == AdaptivePhase.DeploymentRoot) continue;
                var f = AdaptiveFixtures.AtPhase(phase);
                Assert.That(AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                    f.Decision, f.Layout, f.Slots)[(int)AdaptiveCommandChoice.Cancel], Is.True, phase.ToString());
            }
        }

        [TestCase(1, PlayerId.Player0)]
        [TestCase(7, PlayerId.Player0)]
        [TestCase(31, PlayerId.Player0)]
        [TestCase(61, PlayerId.Player1)]
        [TestCase(97, PlayerId.Player1)]
        public void EveryMaskedGameplaySequence_IsAcceptedOrPreciselyHiddenBlocked(int seed, PlayerId seat)
        {
            var f = AdaptiveFixtures.RevealedGame(seed, seat);
            var sequences = AdaptiveFixtures.CompletedMaskedSequences(f);
            Assert.That(sequences, Is.Not.Empty);

            foreach (var sequence in sequences)
            {
                var transition = AdaptiveFixtures.ApplySequence(f, sequence);
                Assert.That(transition.Command, Is.Not.Null, string.Join(",", sequence));
                Assert.That(transition.InvalidSequence, Is.False, string.Join(",", sequence));
                AssertAcceptedOrPreciselyHiddenBlocked(f.Game!, transition.Command!, string.Join(",", sequence));
            }
        }

        [Test]
        public void InactiveAndTerminalGameplayRootsExposeNoEngineRejectedCommand()
        {
            var f = AdaptiveFixtures.RevealedGame(39);
            var inactiveState = new AdaptiveDecisionState(PlayerId.Player1, AdaptivePhase.GameplayRoot);
            var inactiveSlots = new AdaptiveUnitSlots(f.Layout.UnitCount);
            inactiveSlots.Sync(f.Game!, PlayerId.Player1);
            Assert.That(AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player1, inactiveState,
                f.Layout, inactiveSlots), Has.All.False);

            var game = f.Game!;
            var terminal = new GameState(game.Board, game.Config, game.Players, game.ActivePlayer,
                game.Round, game.NextEntityId, isGameOver: true, winner: PlayerId.Player0,
                movedUnitIds: game.MovedUnitIds, attackedUnitIds: game.AttackedUnitIds,
                movementSpent: game.MovementSpent);
            Assert.That(AdaptiveCoding.Mask(terminal, f.Setup, PlayerId.Player0, f.Decision,
                f.Layout, f.Slots), Has.All.False);
        }

        [Test]
        public void CancelAndOutOfRangeAction_ClearPendingStateToCorrectRoot()
        {
            var gameplay = AdaptiveFixtures.RevealedGame(41);
            gameplay.Decision.Enter(AdaptivePhase.DesignValue);
            gameplay.Decision.SelectUnit(3);
            gameplay.Decision.SelectTemplate(7);
            gameplay.Decision.SelectStat(2);
            var cancel = AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.Cancel, gameplay.Game,
                gameplay.Setup, PlayerId.Player0, gameplay.Decision, gameplay.Layout, gameplay.Slots);

            Assert.That(cancel.Command, Is.Null);
            Assert.That(cancel.Intermediate, Is.True);
            Assert.That(cancel.InvalidSequence, Is.False);
            AssertCleared(gameplay.Decision, AdaptivePhase.GameplayRoot);

            var deployment = AdaptiveFixtures.AtPhase(AdaptivePhase.DeploymentCell);
            deployment.Decision.SelectTemplate(0);
            var invalid = AdaptiveCoding.ApplyAction(deployment.Layout.ActionCount + 20, null,
                deployment.Setup, PlayerId.Player0, deployment.Decision, deployment.Layout, deployment.Slots);
            Assert.That(invalid.Command, Is.Null);
            Assert.That(invalid.InvalidSequence, Is.True);
            AssertCleared(deployment.Decision, AdaptivePhase.DeploymentRoot);
        }

        [Test]
        public void StaleUnitSelectionClearsWithoutEmittingCommand()
        {
            var f = AdaptiveFixtures.RevealedGame(43);
            f.Decision.Enter(AdaptivePhase.GameplayUnitCommand);
            f.Decision.SelectUnit(23);

            var transition = AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.Move, f.Game, f.Setup,
                PlayerId.Player0, f.Decision, f.Layout, f.Slots);

            Assert.That(transition.Command, Is.Null);
            Assert.That(transition.InvalidSequence, Is.True);
            AssertCleared(f.Decision, AdaptivePhase.GameplayRoot);
        }

        [Test]
        public void DeploymentRoot_PlacesRepositionsRemovesAndConfirmsThroughDeploymentApis()
        {
            var f = AdaptiveFixtures.AtPhase(AdaptivePhase.DeploymentRoot);
            var template = f.Layout.TemplateOffset;
            int firstCell = f.Layout.CellOffset + f.Layout.CellIndex[f.Setup.View(PlayerId.Player0).Board
                .DeploymentZone(PlayerId.Player0).OrderBy(c => c.Q).ThenBy(c => c.R).First()];

            AssertIntermediate(AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.DeployStartingUnit, null,
                f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots));
            AssertIntermediate(AdaptiveCoding.ApplyAction(template, null, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots));
            var placed = AdaptiveCoding.ApplyAction(firstCell, null, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            Assert.That(placed.MutatedSetup, Is.True);
            Assert.That(f.Setup.Placements(PlayerId.Player0), Has.Count.EqualTo(1));

            int secondCell = f.Layout.CellOffset + f.Layout.CellIndex[f.Setup.View(PlayerId.Player0).Board
                .DeploymentZone(PlayerId.Player0).OrderBy(c => c.Q).ThenBy(c => c.R).Skip(1).First()];
            AssertIntermediate(AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.RepositionStartingUnit, null,
                f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots));
            AssertIntermediate(AdaptiveCoding.ApplyAction(f.Layout.UnitOffset, null, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots));
            var moved = AdaptiveCoding.ApplyAction(secondCell, null, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            Assert.That(moved.MutatedSetup, Is.True);
            Assert.That(f.Setup.Placements(PlayerId.Player0).Single().Cell,
                Is.EqualTo(f.Layout.Cells[secondCell - f.Layout.CellOffset]));

            AssertIntermediate(AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.RemoveStartingUnit, null,
                f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots));
            var removed = AdaptiveCoding.ApplyAction(f.Layout.UnitOffset, null, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            Assert.That(removed.MutatedSetup, Is.True);
            Assert.That(f.Setup.Placements(PlayerId.Player0), Is.Empty);

            AdaptiveFixtures.PlaceSixAffordable(new AdaptiveDeploymentFixture(f.Setup), PlayerId.Player0);
            var confirmMask = AdaptiveCoding.Mask(null, f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            Assert.That(confirmMask[(int)AdaptiveCommandChoice.ConfirmDeployment], Is.True);
            var confirmed = AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.ConfirmDeployment, null,
                f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            Assert.That(confirmed.MutatedSetup, Is.True);
            Assert.That(f.Setup.Confirmed(PlayerId.Player0), Is.True);
        }

        [Test]
        public void ConfirmedDeploymentSeatHasNoFurtherSetupMutationMask()
        {
            var f = AdaptiveFixtures.AtPhase(AdaptivePhase.DeploymentRoot);
            AdaptiveFixtures.PlaceSixAffordable(new AdaptiveDeploymentFixture(f.Setup), PlayerId.Player0);
            Assert.That(f.Setup.TryConfirm(PlayerId.Player0), Is.True);

            Assert.That(AdaptiveCoding.Mask(null, f.Setup, PlayerId.Player0, f.Decision,
                f.Layout, f.Slots), Has.All.False);
            f.Decision.Enter(AdaptivePhase.DeploymentPlacedUnit);
            f.Decision.SelectValue((int)AdaptiveCommandChoice.RemoveStartingUnit);
            var nonRoot = AdaptiveCoding.Mask(null, f.Setup, PlayerId.Player0, f.Decision,
                f.Layout, f.Slots);
            Assert.That(nonRoot[(int)AdaptiveCommandChoice.Cancel], Is.True);
            Assert.That(nonRoot.Count(enabled => enabled), Is.EqualTo(1));
        }

        [Test]
        public void DeploymentPlacedUnit_OperationIsEncodedInObservedPendingValue()
        {
            var reposition = AdaptiveFixtures.AtPhase(AdaptivePhase.DeploymentRoot);
            var remove = AdaptiveFixtures.AtPhase(AdaptivePhase.DeploymentRoot);
            var repositionCell = reposition.Setup.View(PlayerId.Player0).Board.DeploymentZone(PlayerId.Player0)
                .OrderBy(c => c.Q).ThenBy(c => c.R).First();
            var removeCell = remove.Setup.View(PlayerId.Player0).Board.DeploymentZone(PlayerId.Player0)
                .OrderBy(c => c.Q).ThenBy(c => c.R).First();
            Assert.That(reposition.Setup.TryPlace(PlayerId.Player0, 0, repositionCell), Is.True);
            Assert.That(remove.Setup.TryPlace(PlayerId.Player0, 0, removeCell), Is.True);

            AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.RepositionStartingUnit, null,
                reposition.Setup, PlayerId.Player0, reposition.Decision, reposition.Layout, reposition.Slots);
            AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.RemoveStartingUnit, null,
                remove.Setup, PlayerId.Player0, remove.Decision, remove.Layout, remove.Slots);
            var repositionObservation = AdaptiveCoding.Observe(null, reposition.Setup, PlayerId.Player0,
                reposition.Decision, reposition.Layout, reposition.Slots);
            var removeObservation = AdaptiveCoding.Observe(null, remove.Setup, PlayerId.Player0,
                remove.Decision, remove.Layout, remove.Slots);

            Assert.That(reposition.Decision.Phase, Is.EqualTo(AdaptivePhase.DeploymentPlacedUnit));
            Assert.That(remove.Decision.Phase, Is.EqualTo(AdaptivePhase.DeploymentPlacedUnit));
            Assert.That(reposition.Decision.PendingValue,
                Is.EqualTo((int)AdaptiveCommandChoice.RepositionStartingUnit));
            Assert.That(remove.Decision.PendingValue,
                Is.EqualTo((int)AdaptiveCommandChoice.RemoveStartingUnit));
            Assert.That(removeObservation, Is.Not.EqualTo(repositionObservation),
                "the operation that changes decoding must be part of the observation");
            Assert.That(AdaptiveCoding.Mask(null, reposition.Setup, PlayerId.Player0,
                    reposition.Decision, reposition.Layout, reposition.Slots),
                Is.EqualTo(AdaptiveCoding.Mask(null, remove.Setup, PlayerId.Player0,
                    remove.Decision, remove.Layout, remove.Slots)),
                "the same placement-slot action is offered, while observed state identifies its meaning");
        }

        [Test]
        public void PendingIndexZeroIsObservablyDifferentFromUnset()
        {
            var f = AdaptiveFixtures.RevealedGame(37);
            var unset = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayUnitCommand);
            var selected = new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayUnitCommand);
            selected.SelectUnit(0);

            var unsetObservation = AdaptiveCoding.Observe(f.Game, f.Setup, PlayerId.Player0,
                unset, f.Layout, f.Slots);
            var selectedObservation = AdaptiveCoding.Observe(f.Game, f.Setup, PlayerId.Player0,
                selected, f.Layout, f.Slots);

            Assert.That(selectedObservation, Is.Not.EqualTo(unsetObservation));
            int pendingUnitGlobal = f.Layout.ObservationChannels * f.Layout.CellCount + 7 + 14;
            Assert.That(unsetObservation[pendingUnitGlobal], Is.Zero);
            Assert.That(selectedObservation[pendingUnitGlobal], Is.GreaterThan(0f));
        }

        [Test]
        public void HiddenEnemyNeverAppearsAsAttackTarget()
        {
            var f = AdaptiveFixtures.GameWithHiddenEnemy();
            int attackerSlot = Enumerable.Range(0, f.Slots.Capacity)
                .First(slot => f.Slots.UnitIdAt(slot) >= 0);
            f.Decision.Enter(AdaptivePhase.GameplayAttackCell);
            f.Decision.SelectUnit(attackerSlot);

            var mask = AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);

            Assert.That(mask[f.Layout.CellOffset + f.HiddenEnemyCell], Is.False);
        }

        [Test]
        public void ReinforcementCommandUsesLegalMovesAndNewUnitTakesLowestReleasedSlot()
        {
            var f = AdaptiveFixtures.RevealedGame(47);
            var before = f.Game!;
            int released = 1;
            int releasedId = f.Slots.UnitIdAt(released);
            var p0 = before.Player(PlayerId.Player0);
            var players = before.Players.ToArray();
            players[0] = new PlayerState(PlayerId.Player0, 200, p0.Barracks,
                p0.UnitsOnBoard.Where(u => u.Id != releasedId).ToArray(), p0.Generators, p0.DestroyedValue);
            f.Game = new GameState(before.Board, before.Config, players, before.ActivePlayer, before.Round,
                before.NextEntityId, before.IsGameOver, before.Winner, before.MovedUnitIds,
                before.AttackedUnitIds, before.MovementSpent);
            f.Slots.Sync(f.Game, PlayerId.Player0);
            Assert.That(f.Slots.UnitIdAt(released), Is.EqualTo(-1));

            var rootMask = AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            Assert.That(rootMask[(int)AdaptiveCommandChoice.DeployReinforcement], Is.True);
            AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.DeployReinforcement, f.Game, f.Setup,
                PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            int template = Enumerable.Range(f.Layout.TemplateOffset, f.Layout.TemplateCount)
                .First(i => AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                    f.Decision, f.Layout, f.Slots)[i]);
            AdaptiveCoding.ApplyAction(template, f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            int cell = Enumerable.Range(f.Layout.CellOffset, f.Layout.CellCount)
                .First(i => AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                    f.Decision, f.Layout, f.Slots)[i]);
            var transition = AdaptiveCoding.ApplyAction(cell, f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            var applied = GameEngine.Apply(f.Game, transition.Command!);

            Assert.That(applied.Success, Is.True);
            f.Slots.Sync(applied.NewState, PlayerId.Player0);
            Assert.That(f.Slots.UnitIdAt(released), Is.EqualTo(before.NextEntityId));
        }

        [Test]
        public void SlotCapacityMustMatchLayoutBeforeObservationCanTouchGlobals()
        {
            var f = AdaptiveFixtures.RevealedGame(59);
            var oversized = new AdaptiveUnitSlots(f.Layout.UnitCount + 1);

            Assert.That(() => AdaptiveCoding.Observe(f.Game, f.Setup, PlayerId.Player0,
                    f.Decision, f.Layout, oversized),
                Throws.ArgumentException.With.Message.Contains("slot capacity"));
            Assert.That(() => AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                    f.Decision, f.Layout, oversized),
                Throws.ArgumentException.With.Message.Contains("slot capacity"));
        }

        [Test]
        public void DesignFlowMasksOnlyCustomSlotsAndEmitsOneAtomicStatReplacement()
        {
            var f = AdaptiveFixtures.RevealedGame(53);
            AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.RedesignCustom, f.Game, f.Setup,
                PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            var slotMask = AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            Assert.That(Enumerable.Range(0, 6).All(i => !slotMask[f.Layout.TemplateOffset + i]), Is.True);
            Assert.That(Enumerable.Range(6, 3).All(i => slotMask[f.Layout.TemplateOffset + i]), Is.True);

            int selectedSlot = 6;
            int selectedStat = (int)AdaptiveStat.Vision;
            AdaptiveCoding.ApplyAction(f.Layout.TemplateOffset + selectedSlot, f.Game, f.Setup,
                PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            AdaptiveCoding.ApplyAction(f.Layout.StatOffset + selectedStat, f.Game, f.Setup,
                PlayerId.Player0, f.Decision, f.Layout, f.Slots);
            var valueMask = AdaptiveCoding.Mask(f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            int valueAction = Enumerable.Range(f.Layout.ValueOffset, f.Layout.ValueCount)
                .Where(i => valueMask[i]).Last();
            AdaptiveCoding.ApplyAction(valueAction, f.Game, f.Setup, PlayerId.Player0,
                f.Decision, f.Layout, f.Slots);
            var transition = AdaptiveCoding.ApplyAction((int)AdaptiveCommandChoice.ConfirmDesign, f.Game,
                f.Setup, PlayerId.Player0, f.Decision, f.Layout, f.Slots);

            Assert.That(transition.Command, Is.TypeOf<ReplaceTemplate>());
            var replace = (ReplaceTemplate)transition.Command!;
            Assert.That(replace.TemplateIndex, Is.EqualTo(6));
            Assert.That(replace.Stats.PointCost, Is.LessThanOrEqualTo(24));
            var old = f.Game!.Player(PlayerId.Player0).Barracks[6].Stats;
            Assert.That(new[] { replace.Stats.Health, replace.Stats.Damage, replace.Stats.Defense,
                replace.Stats.Movement, replace.Stats.VerticalMovement, replace.Stats.Range,
                replace.Stats.RangeArc, replace.Stats.VisionArc }, Is.EqualTo(new[] { old.Health,
                old.Damage, old.Defense, old.Movement, old.VerticalMovement, old.Range,
                old.RangeArc, old.VisionArc }));
            Assert.That(GameEngine.Apply(f.Game, replace).Success, Is.True);
        }

        private static void AssertIntermediate(AdaptiveTransition transition)
        {
            Assert.That(transition.Command, Is.Null);
            Assert.That(transition.Intermediate, Is.True);
            Assert.That(transition.InvalidSequence, Is.False);
        }

        private static void AssertCleared(AdaptiveDecisionState state, AdaptivePhase phase)
        {
            Assert.That(state.Phase, Is.EqualTo(phase));
            Assert.That(state.PendingUnitSlot, Is.EqualTo(-1));
            Assert.That(state.PendingTemplateSlot, Is.EqualTo(-1));
            Assert.That(state.PendingStat, Is.EqualTo(-1));
            Assert.That(state.PendingValue, Is.EqualTo(-1));
        }

        private static void AssertAcceptedOrPreciselyHiddenBlocked(GameState game, Command command,
            string sequence)
        {
            var applied = GameEngine.Apply(game, command);
            if (applied.Success) return;

            Assert.That(command, Is.TypeOf<MoveUnit>().Or.TypeOf<DeployUnit>(), sequence);
            Assert.That(applied.Reason, Is.EqualTo(command is MoveUnit
                ? RejectionReason.OutOfMovementRange
                : RejectionReason.TileOccupied), sequence);
            var visibleProjection = AdaptiveFixtures.WithoutHiddenEnemies(game, command.Issuer);
            Assert.That(GameEngine.Apply(visibleProjection, command).Success, Is.True,
                $"{sequence}: rejection must be caused solely by a removed hidden blocker");
            Assert.That(game.Opponent(command.Issuer).UnitsOnBoard.Any(unit => unit.IsAlive
                && !TargetingService.IsVisibleToArmy(game, command.Issuer, unit.Cell, unit.Elevation)), Is.True,
                $"{sequence}: no hidden blocker exists");
        }
    }
}
