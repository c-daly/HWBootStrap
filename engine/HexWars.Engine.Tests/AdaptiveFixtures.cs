using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public sealed class AdaptiveFixture
    {
        public GameState? Game { get; set; }
        public AdaptiveDeployment Setup { get; }
        public AdaptiveLayout Layout { get; }
        public AdaptiveUnitSlots Slots { get; }
        public AdaptiveDecisionState Decision { get; }
        public int FriendlyCell { get; set; } = -1;
        public int HiddenEnemyCell { get; set; } = -1;

        public AdaptiveFixture(GameState? game, AdaptiveDeployment setup, AdaptiveLayout layout,
            AdaptiveUnitSlots slots, AdaptiveDecisionState decision)
        {
            Game = game;
            Setup = setup;
            Layout = layout;
            Slots = slots;
            Decision = decision;
        }
    }

    public sealed class AdaptiveDeploymentFixture
    {
        public AdaptiveDeployment Deployment { get; }

        public AdaptiveDeploymentFixture(AdaptiveDeployment deployment) { Deployment = deployment; }

        public AdaptiveDeploymentView Observe(PlayerId seat) => Deployment.View(seat);
        public bool Place(PlayerId seat, int template, HexCoord cell) => Deployment.TryPlace(seat, template, cell);
        public bool CanConfirm(PlayerId seat) => Deployment.CanConfirm(seat);
        public IReadOnlyList<DeploymentPlacement> Placements(PlayerId seat) => Deployment.Placements(seat);

        public HexCoord FirstLegalCell(PlayerId seat)
        {
            var used = new HashSet<HexCoord>(Deployment.Placements(seat).Select(p => p.Cell));
            return Deployment.Board.DeploymentZone(seat)
                .Where(c => Deployment.Game.Terrain(Deployment.Board.TileAt(c).Terrain).Passable)
                .OrderBy(c => c.Q).ThenBy(c => c.R)
                .First(c => !used.Contains(c));
        }
    }

    public static class AdaptiveFixtures
    {
        public static AdaptiveFixture GameWithHiddenEnemy()
        {
            var f = RevealedGame(17);
            var friendly = f.Game!.Player(PlayerId.Player0).UnitsOnBoard.First();
            var hidden = f.Game.Player(PlayerId.Player1).UnitsOnBoard
                .First(u => !TargetingService.IsVisibleToArmy(f.Game, PlayerId.Player0, u.Cell, u.Elevation));
            f.FriendlyCell = f.Layout.CellIndex[friendly.Cell];
            f.HiddenEnemyCell = f.Layout.CellIndex[hidden.Cell];
            return f;
        }

        public static IReadOnlyList<AdaptiveFixture> HiddenBlockerVariants()
        {
            var terrain = Enum.GetValues(typeof(TerrainType)).Cast<TerrainType>()
                .ToDictionary(type => type, _ => new TerrainDef(1, 5, 0, true));
            var cfg = AdaptiveEnvConfig.Default();
            cfg.Game = new GameConfig(terrain, startingPoints: 200, biomesEnabled: true,
                fogOfWar: true, designFee: 2, maxDesignPointCost: 24,
                fixedTemplateCount: 6, templateSlotCount: 9);
            var layout = new AdaptiveLayout(cfg);
            var tiles = layout.Cells.Select(cell => new Tile(cell, 0, TerrainType.Forest)).ToArray();
            var zone0 = Enumerable.Range(0, 3).SelectMany(col => Enumerable.Range(0, cfg.BoardGen.Height)
                .Select(row => HexLayout.OffsetToAxial(col, row))).ToArray();
            var zone1 = Enumerable.Range(cfg.BoardGen.Width - 3, 3)
                .SelectMany(col => Enumerable.Range(0, cfg.BoardGen.Height)
                    .Select(row => HexLayout.OffsetToAxial(col, row))).ToArray();
            var board = new Board(tiles, zone0, zone1);
            var setup = new AdaptiveDeployment(board, cfg);
            HexCoord friendlyCell = HexLayout.OffsetToAxial(0, 4);
            var blockers = new[]
            {
                (HexCoord?)null,
                HexLayout.OffsetToAxial(2, 4),
                HexLayout.OffsetToAxial(2, 5),
            };
            var fixtures = new List<AdaptiveFixture>();
            foreach (HexCoord? blocker in blockers)
            {
                var friendly = new Unit(1, PlayerId.Player0, cfg.Templates[1].Stats,
                    friendlyCell, 0, cfg.Templates[1].Name);
                IReadOnlyList<Unit> enemies = blocker.HasValue
                    ? new[] { new Unit(2, PlayerId.Player1, cfg.Templates[0].Stats,
                        blocker.Value, 0, cfg.Templates[0].Name) }
                    : Array.Empty<Unit>();
                var players = new[]
                {
                    new PlayerState(PlayerId.Player0, 200, cfg.Templates, new[] { friendly }),
                    new PlayerState(PlayerId.Player1, 200, cfg.Templates, enemies),
                };
                var game = new GameState(board, cfg.Game, players, PlayerId.Player0, 1, 10);
                var slots = new AdaptiveUnitSlots(cfg.MaxControllableUnits);
                slots.Sync(game, PlayerId.Player0);
                fixtures.Add(new AdaptiveFixture(game, setup, layout, slots,
                    new AdaptiveDecisionState(PlayerId.Player0, AdaptivePhase.GameplayRoot)));
            }
            return fixtures;
        }

        public static AdaptiveFixture RevealedGame(int seed) => RevealedGame(seed, PlayerId.Player0);

        public static AdaptiveFixture RevealedGame(int seed, PlayerId seat)
        {
            var cfg = AdaptiveEnvConfig.Default();
            cfg.Game = GameConfig.Default(biomesEnabled: false, fogOfWar: true, startingPoints: 40,
                designFee: 2, maxDesignPointCost: 24, fixedTemplateCount: 6, templateSlotCount: 9);
            var board = new RandomBoardGenerator(cfg.BoardGen).Generate(seed);
            var setup = new AdaptiveDeployment(board, cfg);
            PlaceSixAffordable(new AdaptiveDeploymentFixture(setup), PlayerId.Player0);
            PlaceSixAffordable(new AdaptiveDeploymentFixture(setup), PlayerId.Player1);
            Assert.That(setup.TryConfirm(PlayerId.Player0), Is.True);
            Assert.That(setup.TryConfirm(PlayerId.Player1), Is.True);
            var game = setup.Reveal(seat);
            var layout = new AdaptiveLayout(cfg);
            var slots = new AdaptiveUnitSlots(cfg.MaxControllableUnits);
            slots.Sync(game, seat);
            return new AdaptiveFixture(game, setup, layout, slots,
                new AdaptiveDecisionState(seat, AdaptivePhase.GameplayRoot));
        }

        public static AdaptiveDeploymentFixture Deployment(int seed)
        {
            var cfg = AdaptiveEnvConfig.Default();
            var board = new RandomBoardGenerator(cfg.BoardGen).Generate(seed);
            return new AdaptiveDeploymentFixture(new AdaptiveDeployment(board, cfg));
        }

        public static AdaptiveFixture AtPhase(AdaptivePhase phase)
        {
            bool deployment = phase <= AdaptivePhase.DeploymentMoveCell;
            if (deployment)
            {
                var d = Deployment(23);
                var cfg = AdaptiveEnvConfig.Default();
                var state = new AdaptiveDecisionState(PlayerId.Player0);
                state.Enter(phase);
                return new AdaptiveFixture(null, d.Deployment, new AdaptiveLayout(cfg),
                    new AdaptiveUnitSlots(cfg.MaxControllableUnits), state);
            }

            var f = RevealedGame(23);
            f.Decision.Enter(phase);
            return f;
        }

        public static IReadOnlyList<Unit> Units(params int[] ids) => ids
            .Select((id, i) => new Unit(id, PlayerId.Player0, AdaptiveContractData.Templates[i % 9].Stats,
                new HexCoord(i, 0), 0, AdaptiveContractData.Templates[i % 9].Name))
            .ToArray();

        public static void PlaceSixAffordable(AdaptiveDeploymentFixture fixture, PlayerId seat)
        {
            for (int template = 0; template < 6; template++)
                Assert.That(fixture.Place(seat, template, fixture.FirstLegalCell(seat)), Is.True);
        }

        public static IReadOnlyList<IReadOnlyList<int>> CompletedMaskedSequences(AdaptiveFixture fixture)
        {
            var sequences = new List<IReadOnlyList<int>>();
            PlayerId seat = fixture.Decision.Seat;
            var roots = AdaptiveCoding.Mask(fixture.Game, fixture.Setup, seat,
                fixture.Decision, fixture.Layout, fixture.Slots);
            foreach (int action in Enabled(roots))
            {
                if (action == (int)AdaptiveCommandChoice.EndTurn)
                {
                    sequences.Add(new[] { action });
                    continue;
                }
                if (action == (int)AdaptiveCommandChoice.ChooseUnit)
                    ExpandUnitSequences(fixture, action, sequences);
                else if (action == (int)AdaptiveCommandChoice.DeployReinforcement)
                    ExpandTemplateCellSequences(fixture, action, sequences);
                else if (action == (int)AdaptiveCommandChoice.RedesignCustom)
                    ExpandDesignSequences(fixture, action, sequences);
            }
            fixture.Decision.Clear(AdaptivePhase.GameplayRoot);
            return sequences;
        }

        public static AdaptiveTransition ApplySequence(AdaptiveFixture fixture, IReadOnlyList<int> actions)
        {
            fixture.Decision.Clear(fixture.Game == null ? AdaptivePhase.DeploymentRoot : AdaptivePhase.GameplayRoot);
            AdaptiveTransition transition = default;
            foreach (int action in actions)
                transition = AdaptiveCoding.ApplyAction(action, fixture.Game, fixture.Setup, fixture.Decision.Seat,
                    fixture.Decision, fixture.Layout, fixture.Slots);
            return transition;
        }

        public static GameState WithEnemy(GameState source, Unit enemy, int enemyPoints)
        {
            var players = source.Players.ToArray();
            var old = source.Player(PlayerId.Player1);
            players[1] = new PlayerState(PlayerId.Player1, enemyPoints, old.Barracks,
                new[] { enemy }, old.Generators, old.DestroyedValue);
            return new GameState(source.Board, source.Config, players, source.ActivePlayer, source.Round,
                Math.Max(source.NextEntityId, enemy.Id + 1), source.IsGameOver, source.Winner,
                source.MovedUnitIds, source.AttackedUnitIds, source.MovementSpent);
        }

        public static GameState WithoutHiddenEnemies(GameState source, PlayerId seat)
        {
            var players = source.Players.ToArray();
            PlayerId foe = seat == PlayerId.Player0 ? PlayerId.Player1 : PlayerId.Player0;
            var enemy = source.Player(foe);
            players[(int)foe] = new PlayerState(foe, enemy.Points, enemy.Barracks,
                enemy.UnitsOnBoard.Where(unit => TargetingService.IsVisibleToArmy(
                    source, seat, unit.Cell, unit.Elevation)).ToArray(),
                enemy.Generators.Where(generator => TargetingService.IsVisibleToArmy(
                    source, seat, generator.Cell, generator.Elevation)).ToArray(),
                enemy.DestroyedValue);
            return new GameState(source.Board, source.Config, players, source.ActivePlayer, source.Round,
                source.NextEntityId, source.IsGameOver, source.Winner, source.MovedUnitIds,
                source.AttackedUnitIds, source.MovementSpent);
        }

        private static void ExpandUnitSequences(AdaptiveFixture f, int root,
            ICollection<IReadOnlyList<int>> sequences)
        {
            PlayerId seat = f.Decision.Seat;
            f.Decision.Clear(AdaptivePhase.GameplayRoot);
            AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
            foreach (int unit in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                         f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.UnitOffset))
            {
                f.Decision.Clear(AdaptivePhase.GameplayRoot);
                AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                AdaptiveCoding.ApplyAction(unit, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                foreach (int command in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                             f.Decision, f.Layout, f.Slots)).Where(x => x >= (int)AdaptiveCommandChoice.Move))
                {
                    f.Decision.Clear(AdaptivePhase.GameplayRoot);
                    AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    AdaptiveCoding.ApplyAction(unit, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    AdaptiveCoding.ApplyAction(command, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    foreach (int cell in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                                 f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.CellOffset))
                        sequences.Add(new[] { root, unit, command, cell });
                }
            }
        }

        private static void ExpandTemplateCellSequences(AdaptiveFixture f, int root,
            ICollection<IReadOnlyList<int>> sequences)
        {
            PlayerId seat = f.Decision.Seat;
            f.Decision.Clear(AdaptivePhase.GameplayRoot);
            AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
            foreach (int template in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                         f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.TemplateOffset))
            {
                f.Decision.Clear(AdaptivePhase.GameplayRoot);
                AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                AdaptiveCoding.ApplyAction(template, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                foreach (int cell in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                             f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.CellOffset))
                    sequences.Add(new[] { root, template, cell });
            }
        }

        private static void ExpandDesignSequences(AdaptiveFixture f, int root,
            ICollection<IReadOnlyList<int>> sequences)
        {
            PlayerId seat = f.Decision.Seat;
            f.Decision.Clear(AdaptivePhase.GameplayRoot);
            AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
            foreach (int slot in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                         f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.TemplateOffset))
            {
                f.Decision.Clear(AdaptivePhase.GameplayRoot);
                AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                AdaptiveCoding.ApplyAction(slot, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                foreach (int stat in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                             f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.StatOffset))
                {
                    f.Decision.Clear(AdaptivePhase.GameplayRoot);
                    AdaptiveCoding.ApplyAction(root, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    AdaptiveCoding.ApplyAction(slot, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    AdaptiveCoding.ApplyAction(stat, f.Game, f.Setup, seat, f.Decision, f.Layout, f.Slots);
                    foreach (int value in Enabled(AdaptiveCoding.Mask(f.Game, f.Setup, seat,
                                 f.Decision, f.Layout, f.Slots)).Where(x => x >= f.Layout.ValueOffset))
                        sequences.Add(new[] { root, slot, stat, value, (int)AdaptiveCommandChoice.ConfirmDesign });
                }
            }
        }

        private static IEnumerable<int> Enabled(bool[] mask) => Enumerable.Range(0, mask.Length).Where(i => mask[i]);
    }
}
