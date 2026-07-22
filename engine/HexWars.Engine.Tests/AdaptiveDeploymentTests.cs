using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class AdaptiveDeploymentTests
    {
        private static readonly PlayerId P0 = PlayerId.Player0;
        private static readonly PlayerId P1 = PlayerId.Player1;

        [Test]
        public void SecondSeatView_IsUnchangedByFirstSeatsHiddenPlacement()
        {
            var deployment = NewDeployment(seed: 11);
            var before = ViewFingerprint(deployment.View(P1));

            Assert.That(deployment.TryPlace(P0, 4, FirstLegalCell(deployment, P0)), Is.True);

            Assert.That(ViewFingerprint(deployment.View(P1)), Is.EqualTo(before));
            Assert.That(typeof(AdaptiveDeploymentView).GetProperties().Select(p => p.Name),
                Has.None.Contains("Opponent"));
        }

        [Test]
        public void View_IsASeatSafeSnapshot_NotALiveLedgerReference()
        {
            var deployment = NewDeployment(seed: 21);
            var snapshot = deployment.View(P0);

            Assert.That(deployment.TryPlace(P0, 0, FirstLegalCell(deployment, P0)), Is.True);

            Assert.That(snapshot.OwnPlacements, Is.Empty);
            Assert.That(snapshot.RemainingBudget, Is.EqualTo(132));
            Assert.That(deployment.View(P0).OwnPlacements, Has.Count.EqualTo(1));
        }

        [Test]
        public void Confirm_RequiresExactlySixAffordableUniqueLegalPlacements()
        {
            var deployment = NewDeployment(seed: 12);

            Assert.That(deployment.CanConfirm(P0), Is.False);
            PlaceCombinedArms(deployment, P0);

            Assert.That(deployment.Placements(P0), Has.Count.EqualTo(6));
            Assert.That(deployment.Placements(P0).Select(p => p.Cell).Distinct().Count(), Is.EqualTo(6));
            Assert.That(deployment.View(P0).RemainingBudget, Is.EqualTo(1));
            Assert.That(deployment.CanConfirm(P0), Is.True);
            Assert.That(deployment.TryPlace(P0, 0, FirstUnusedLegalCell(deployment, P0)), Is.False,
                "a seventh starting unit is never legal");
        }

        [Test]
        public void Place_RejectsInvalidTemplateDuplicateCellOutsideZoneAndImpassableCell()
        {
            var (deployment, impassable, outside) = DeploymentWithImpassableZoneCell();
            var legal = FirstLegalCell(deployment, P0);

            Assert.That(deployment.TryPlace(P0, -1, legal), Is.False);
            Assert.That(deployment.TryPlace(P0, deployment.View(P0).Templates.Count, legal), Is.False);
            Assert.That(deployment.TryPlace(P0, 0, outside), Is.False);
            Assert.That(deployment.TryPlace(P0, 0, impassable), Is.False);
            Assert.That(deployment.TryPlace(P0, 0, legal), Is.True);
            Assert.That(deployment.TryPlace(P0, 1, legal), Is.False);
        }

        [Test]
        public void Place_RejectsACompositionWhosePointCostExceedsBudget()
        {
            var config = AdaptiveEnvConfig.Default();
            config.StartingArmyBudget = 120;
            var deployment = NewDeployment(seed: 22, config);
            var cells = LegalCells(deployment, P0).Take(6).ToArray();

            for (int i = 0; i < 5; i++)
                Assert.That(deployment.TryPlace(P0, 5, cells[i]), Is.True);

            Assert.That(deployment.TryPlace(P0, 5, cells[5]), Is.False);
            Assert.That(deployment.View(P0).RemainingBudget, Is.EqualTo(5));
            Assert.That(deployment.CanConfirm(P0), Is.False);
        }

        [Test]
        public void RemoveReusesLowestFreeSlot_AndMovePreservesSlot()
        {
            var deployment = NewDeployment(seed: 23);
            var cells = LegalCells(deployment, P0).Take(4).ToArray();
            Assert.That(deployment.TryPlace(P0, 0, cells[0]), Is.True);
            Assert.That(deployment.TryPlace(P0, 1, cells[1]), Is.True);
            Assert.That(deployment.TryPlace(P0, 2, cells[2]), Is.True);

            Assert.That(deployment.TryMove(P0, 1, cells[3]), Is.True);
            Assert.That(deployment.Placements(P0).Single(p => p.Slot == 1).Cell, Is.EqualTo(cells[3]));
            Assert.That(deployment.TryRemove(P0, 1), Is.True);
            Assert.That(deployment.TryPlace(P0, 3, cells[1]), Is.True);

            var replacement = deployment.Placements(P0).Single(p => p.Cell == cells[1]);
            Assert.That(replacement.Slot, Is.EqualTo(1));
            Assert.That(replacement.TemplateIndex, Is.EqualTo(3));
        }

        [Test]
        public void ConfirmFreezesOnlyThatSeatsLedger_AndRevealRequiresBothSeats()
        {
            var deployment = NewDeployment(seed: 24);
            PlaceCombinedArms(deployment, P0);
            PlaceCombinedArms(deployment, P1);

            Assert.That(deployment.TryConfirm(P0), Is.True);
            Assert.That(deployment.Confirmed(P0), Is.True);
            Assert.That(deployment.TryMove(P0, 0, deployment.Placements(P0)[1].Cell), Is.False);
            Assert.That(deployment.TryRemove(P0, 0), Is.False);
            Assert.That(deployment.TryConfirm(P0), Is.False);
            Assert.That(() => deployment.Reveal(P1), Throws.InvalidOperationException);
            Assert.That(deployment.TryMove(P1, 0, FirstUnusedLegalCell(deployment, P1)), Is.True,
                "the other hidden ledger remains editable until it confirms");
            Assert.That(deployment.TryConfirm(P1), Is.True);
        }

        [TestCase(PlayerId.Player0)]
        [TestCase(PlayerId.Player1)]
        public void Reveal_BuildsRoundOneWithRequestedFirstPlayerAndDeterministicIds(PlayerId firstPlayer)
        {
            var deployment = NewDeployment(seed: 25);
            PlaceCombinedArms(deployment, P0);
            PlaceCombinedArms(deployment, P1);
            Assert.That(deployment.TryConfirm(P0), Is.True);
            Assert.That(deployment.TryConfirm(P1), Is.True);

            var state = deployment.Reveal(firstPlayer);

            Assert.That(state.Round, Is.EqualTo(1));
            Assert.That(state.ActivePlayer, Is.EqualTo(firstPlayer));
            Assert.That(state.NextEntityId, Is.EqualTo(13));
            Assert.That(state.Player(P0).UnitsOnBoard.Select(u => u.Id), Is.EqualTo(Enumerable.Range(1, 6)));
            Assert.That(state.Player(P1).UnitsOnBoard.Select(u => u.Id), Is.EqualTo(Enumerable.Range(7, 6)));
            Assert.That(state.Player(P0).Points, Is.EqualTo(deployment.Game.StartingPoints));
            Assert.That(state.Player(P1).Points, Is.EqualTo(deployment.Game.StartingPoints));
            AssertRevealedPlacementsMatch(deployment, state, P0);
            AssertRevealedPlacementsMatch(deployment, state, P1);
        }

        [Test]
        public void Reveal_GivesEachSeatASeparateMutableNineTemplateBarracks()
        {
            var deployment = ConfirmedDeployment(seed: 26);

            var state = deployment.Reveal(P0);
            var barracks0 = (IList<UnitTemplate>)state.Player(P0).Barracks;
            var barracks1 = (IList<UnitTemplate>)state.Player(P1).Barracks;
            var originalP1 = barracks1[6];
            barracks0[6] = new UnitTemplate("Changed", new UnitStats(1, 0, 0, 0, 0, 0, 0, 0, 0));

            Assert.That(barracks0, Has.Count.EqualTo(9));
            Assert.That(barracks1, Has.Count.EqualTo(9));
            Assert.That(barracks0, Is.Not.SameAs(barracks1));
            Assert.That(barracks1[6].Name, Is.EqualTo(originalP1.Name));
            Assert.That(barracks1[6].Stats, Is.EqualTo(originalP1.Stats));
        }

        [Test]
        public void SameBoardAndPolicySeeds_ProduceByteIdenticalRevealedStarts()
        {
            string a = SeededRevealedStart(seed: 31, combinedSeed: 47, randomSeed: 53);
            string b = SeededRevealedStart(seed: 31, combinedSeed: 47, randomSeed: 53);

            Assert.That(b, Is.EqualTo(a));
        }

        [Test]
        public void RandomPolicy_IsRepeatableAffordableUniqueAndLegal()
        {
            var deployment = NewDeployment(seed: 32);
            var policy = new RandomDeploymentPolicy(71);

            var first = policy.Choose(deployment.View(P0));
            var second = policy.Choose(deployment.View(P0));

            Assert.That(PlacementFingerprint(second), Is.EqualTo(PlacementFingerprint(first)));
            AssertPolicyChoiceIsLegal(deployment.View(P0), first);
            ApplyPolicyChoice(deployment, P0, first);
            Assert.That(deployment.CanConfirm(P0), Is.True);
        }

        [Test]
        public void CombinedArmsPolicy_IsSeededAndDoesNotReadOpponentLedger()
        {
            var a = NewDeployment(seed: 13);
            var b = NewDeployment(seed: 13);
            Assert.That(b.TryPlace(P1, 2, FirstLegalCell(b, P1)), Is.True);
            var policy = new CombinedArmsDeploymentPolicy(99);

            var choiceA = policy.Choose(a.View(P0));
            var choiceB = policy.Choose(b.View(P0));

            Assert.That(PlacementFingerprint(choiceB), Is.EqualTo(PlacementFingerprint(choiceA)));
            Assert.That(choiceA.Select(p => p.TemplateIndex), Is.EqualTo(new[] { 0, 1, 2, 3, 4, 5 }));
            AssertPolicyChoiceIsLegal(a.View(P0), choiceA);
        }

        private static AdaptiveDeployment NewDeployment(int seed, AdaptiveEnvConfig? config = null)
        {
            config ??= AdaptiveEnvConfig.Default();
            var board = new RandomBoardGenerator(config.BoardGen).Generate(seed);
            return new AdaptiveDeployment(board, config);
        }

        private static AdaptiveDeployment ConfirmedDeployment(int seed)
        {
            var deployment = NewDeployment(seed);
            PlaceCombinedArms(deployment, P0);
            PlaceCombinedArms(deployment, P1);
            Assert.That(deployment.TryConfirm(P0), Is.True);
            Assert.That(deployment.TryConfirm(P1), Is.True);
            return deployment;
        }

        private static (AdaptiveDeployment Deployment, HexCoord Impassable, HexCoord Outside)
            DeploymentWithImpassableZoneCell()
        {
            var terrain = new Dictionary<TerrainType, TerrainDef>
            {
                [TerrainType.Plains] = new TerrainDef(1, 0, 0, true),
                [TerrainType.Forest] = new TerrainDef(2, 2, 1, true),
                [TerrainType.Rough] = new TerrainDef(2, 1, 1, true),
                [TerrainType.Water] = new TerrainDef(3, 0, 0, false),
            };
            var config = AdaptiveEnvConfig.Default();
            config.Game = new GameConfig(terrain, startingPoints: 17, biomesEnabled: true, fogOfWar: true);
            var zone0 = Enumerable.Range(0, 7).Select(r => new HexCoord(0, r)).ToArray();
            var zone1 = Enumerable.Range(0, 7).Select(r => new HexCoord(10, r)).ToArray();
            var impassable = zone0[0];
            var outside = new HexCoord(5, 0);
            var tiles = zone0.Select(c => new Tile(c, 0, c == impassable ? TerrainType.Water : TerrainType.Plains))
                .Concat(zone1.Select(c => new Tile(c, 0, TerrainType.Plains)))
                .Append(new Tile(outside, 0, TerrainType.Plains));
            return (new AdaptiveDeployment(new Board(tiles, zone0, zone1), config), impassable, outside);
        }

        private static IReadOnlyList<HexCoord> LegalCells(AdaptiveDeployment deployment, PlayerId seat) =>
            deployment.Board.DeploymentZone(seat)
                .Where(c => deployment.Game.Terrain(deployment.Board.TileAt(c).Terrain).Passable)
                .OrderBy(c => c.Q).ThenBy(c => c.R).ToArray();

        private static HexCoord FirstLegalCell(AdaptiveDeployment deployment, PlayerId seat) =>
            LegalCells(deployment, seat)[0];

        private static HexCoord FirstUnusedLegalCell(AdaptiveDeployment deployment, PlayerId seat)
        {
            var used = new HashSet<HexCoord>(deployment.Placements(seat).Select(p => p.Cell));
            return LegalCells(deployment, seat).First(c => !used.Contains(c));
        }

        private static void PlaceCombinedArms(AdaptiveDeployment deployment, PlayerId seat)
        {
            var cells = LegalCells(deployment, seat).Take(6).ToArray();
            for (int template = 0; template < 6; template++)
                Assert.That(deployment.TryPlace(seat, template, cells[template]), Is.True);
        }

        private static void ApplyPolicyChoice(AdaptiveDeployment deployment, PlayerId seat,
            IReadOnlyList<DeploymentPlacement> choice)
        {
            foreach (var placement in choice)
            {
                Assert.That(deployment.TryPlace(seat, placement.TemplateIndex, placement.Cell), Is.True);
                Assert.That(deployment.Placements(seat).Single(p => p.Cell == placement.Cell).Slot,
                    Is.EqualTo(placement.Slot));
            }
        }

        private static void AssertPolicyChoiceIsLegal(AdaptiveDeploymentView view,
            IReadOnlyList<DeploymentPlacement> choice)
        {
            Assert.That(choice, Has.Count.EqualTo(view.RequiredUnits - view.OwnPlacements.Count));
            Assert.That(choice.Select(p => p.Cell).Distinct().Count(), Is.EqualTo(choice.Count));
            Assert.That(choice.All(p => view.Board.IsInDeploymentZone(view.Seat, p.Cell)), Is.True);
            Assert.That(choice.All(p => view.Game.Terrain(view.Board.TileAt(p.Cell).Terrain).Passable), Is.True);
            Assert.That(choice.All(p => p.TemplateIndex >= 0 && p.TemplateIndex < view.Templates.Count), Is.True);
            Assert.That(choice.Sum(p => view.Templates[p.TemplateIndex].Stats.PointCost),
                Is.LessThanOrEqualTo(view.RemainingBudget));
        }

        private static void AssertRevealedPlacementsMatch(AdaptiveDeployment deployment, GameState state,
            PlayerId seat)
        {
            var expected = deployment.Placements(seat).OrderBy(p => p.Slot).ToArray();
            var actual = state.Player(seat).UnitsOnBoard;
            Assert.That(actual.Select(u => u.Cell), Is.EqualTo(expected.Select(p => p.Cell)));
            Assert.That(actual.Select(u => u.Name),
                Is.EqualTo(expected.Select(p => deployment.View(seat).Templates[p.TemplateIndex].Name)));
        }

        private static string SeededRevealedStart(int seed, int combinedSeed, int randomSeed)
        {
            var deployment = NewDeployment(seed);
            ApplyPolicyChoice(deployment, P0, new CombinedArmsDeploymentPolicy(combinedSeed).Choose(deployment.View(P0)));
            ApplyPolicyChoice(deployment, P1, new RandomDeploymentPolicy(randomSeed).Choose(deployment.View(P1)));
            Assert.That(deployment.TryConfirm(P0), Is.True);
            Assert.That(deployment.TryConfirm(P1), Is.True);
            return ReplayFile.Write(deployment.Reveal(P1), Array.Empty<Command>());
        }

        private static string ViewFingerprint(AdaptiveDeploymentView view)
        {
            var sb = new StringBuilder();
            sb.Append((int)view.Seat).Append('|').Append(view.RemainingBudget).Append('|')
                .Append(view.RequiredUnits).Append('|');
            foreach (var p in view.OwnPlacements.OrderBy(p => p.Slot))
                sb.Append(p.Slot).Append(',').Append(p.TemplateIndex).Append(',')
                    .Append(p.Cell.Q).Append(',').Append(p.Cell.R).Append(';');
            sb.Append('|');
            foreach (var t in view.Templates)
                sb.Append(t.Name).Append(':').Append(t.Stats.PointCost).Append(';');
            return sb.ToString();
        }

        private static string PlacementFingerprint(IEnumerable<DeploymentPlacement> placements) =>
            string.Join(";", placements.Select(p => $"{p.Slot},{p.TemplateIndex},{p.Cell.Q},{p.Cell.R}"));
    }
}
