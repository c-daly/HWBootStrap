using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2CodingTests
    {
        // ---- Step 1: geometry and action-region layout ----

        [Test]
        public void Layout_SeparatesTemplateRolesFromUnitSlots()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = 7;
            config.MaxControllableUnits = 7;
            var layout = new TacticalV2Layout(config);
            int cells = config.BoardGen.Width * config.BoardGen.Height;

            Assert.That(layout.TemplateCount, Is.EqualTo(5));
            Assert.That(layout.UnitSlotCount, Is.EqualTo(7));
            Assert.That(layout.MoveOffset, Is.EqualTo(1));
            Assert.That(layout.AttackOffset, Is.EqualTo(1 + 7 * cells));
            Assert.That(layout.DeployOffset, Is.EqualTo(1 + 14 * cells));
            Assert.That(layout.ActionCount, Is.EqualTo(1 + (14 + 5) * cells));
            Assert.That(layout.ObservationChannels, Is.EqualTo(11));
        }

        [Test]
        public void NewGame_SamplesOneCompositionForBothSeats()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.StartingUnitCount = 9;
            config.MaxControllableUnits = 9;
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(41);

            Assert.That(start.TemplateIndices1, Is.EqualTo(start.TemplateIndices0));
            Assert.That(start.State.Player(PlayerId.Player0).UnitsOnBoard, Has.Count.EqualTo(9));
            Assert.That(start.State.Player(PlayerId.Player1).UnitsOnBoard, Has.Count.EqualTo(9));

            // Step 4 extension: Player 1's cells are not independently sampled — every starting slot's
            // Player 1 cell must be the 180-degree mirror of the same slot's Player 0 cell, proving the
            // two armies are geometric mirrors rather than merely similarly distributed.
            var p0Units = start.State.Player(PlayerId.Player0).UnitsOnBoard;
            var p1Units = start.State.Player(PlayerId.Player1).UnitsOnBoard;
            for (int i = 0; i < p0Units.Count; i++)
                Assert.That(p1Units[i].Cell, Is.EqualTo(layout.MirrorCell(p0Units[i].Cell)),
                    $"slot {i}: Player 1's cell must be the mirror of Player 0's cell");
        }

        [Test]
        public void NewGame_SameSeedProducesIdenticalStarts()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);

            TacticalV2Start a = layout.NewGame(99);
            TacticalV2Start b = layout.NewGame(99);

            Assert.That(a.TemplateIndices0, Is.EqualTo(b.TemplateIndices0));
            var aUnits0 = a.State.Player(PlayerId.Player0).UnitsOnBoard;
            var bUnits0 = b.State.Player(PlayerId.Player0).UnitsOnBoard;
            for (int i = 0; i < aUnits0.Count; i++)
                Assert.That(bUnits0[i].Cell, Is.EqualTo(aUnits0[i].Cell));
        }

        [Test]
        public void NewGame_RejectsBoardWhoseDeploymentZoneIsSmallerThanStartingUnitCount()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.BoardGen = new BoardGenConfig(width: 13, height: 9, zoneDepth: 1); // 1×9 = 9 cells/side
            config.StartingUnitCount = 10;
            config.MaxControllableUnits = 10;
            var layout = new TacticalV2Layout(config);

            Assert.Throws<InvalidOperationException>(() => layout.NewGame(1));
        }

        // ---- Step 2: slot identity is registry-tracked, never inferred from stats ----

        [Test]
        public void Observe_UsesRegisteredTemplateIdentityWhenStatsAreEqual()
        {
            TacticalV2CodingFixture fixture =
                TacticalV2CodingFixture.WithEqualStatTemplates();
            float[] observation = TacticalV2Coding.Observe(
                fixture.Game, PlayerId.Player0, fixture.Layout,
                fixture.Slots0, fixture.Slots1);

            Assert.That(fixture.ValueAtFriendlyTemplatePlane(observation, 0), Is.GreaterThan(0f));
            Assert.That(fixture.ValueAtFriendlyTemplatePlane(observation, 1), Is.GreaterThan(0f));
        }

        // ---- Contract coupling: observation_channels names must describe Observe's actual plane order ----

        [Test]
        public void Observe_NonzeroPlaneMatchesContractChannelNameForFriendlyAndEnemyTemplate()
        {
            // MlContract.CreateTacticalV2 hand-builds an observation_channels name list that is
            // supposed to describe TacticalV2Coding.Observe's plane order exactly (nothing in the
            // types ties them together). Prove the coupling by locating each plane purely by name
            // — never by re-deriving the offset formula here — and checking Observe actually wrote
            // a nonzero HP fraction there for a unit of that template.
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(11);
            MlContract contract = MlContract.CreateTacticalV2(config);
            var channels = (IReadOnlyList<string>)contract.Semantics["observation_channels"];

            float[] observation = TacticalV2Coding.Observe(
                start.State, PlayerId.Player0, layout, start.Slots0, start.Slots1);
            int n = layout.CellCount;

            int friendlyTemplateIndex = start.TemplateIndices0[0];
            Unit friendlyUnit = start.State.Player(PlayerId.Player0).UnitsOnBoard[0];
            int friendlyPlane = channels.ToList().IndexOf($"friendly_role_hp_{friendlyTemplateIndex}");
            Assert.That(friendlyPlane, Is.GreaterThanOrEqualTo(0),
                "contract's observation_channels must name the friendly plane for this template");
            int friendlyCell = layout.CellIndex[friendlyUnit.Cell];
            Assert.That(observation[friendlyPlane * n + friendlyCell], Is.GreaterThan(0f));

            int enemyTemplateIndex = start.TemplateIndices1[0];
            Unit enemyUnit = start.State.Player(PlayerId.Player1).UnitsOnBoard[0];
            int enemyPlane = channels.ToList().IndexOf($"visible_enemy_role_hp_{enemyTemplateIndex}");
            Assert.That(enemyPlane, Is.GreaterThanOrEqualTo(0),
                "contract's observation_channels must name the enemy plane for this template");
            int enemyCell = layout.CellIndex[enemyUnit.Cell];
            Assert.That(observation[enemyPlane * n + enemyCell], Is.GreaterThan(0f));
        }

        // ---- Coding: mask/decode round-trip against the engine's own legal-move enumeration ----

        [Test]
        public void Mask_TrueCountMatchesLegalMoveCount()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(7);

            bool[] mask = TacticalV2Coding.Mask(start.State, PlayerId.Player0, layout, start.Slots0);
            int legalCount = LegalMoves.For(start.State).Count;

            Assert.That(mask.Count(selected => selected), Is.EqualTo(legalCount));
        }

        [Test]
        public void Decode_EveryMaskedActionRoundTripsToALegalCommand()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(7);

            bool[] mask = TacticalV2Coding.Mask(start.State, PlayerId.Player0, layout, start.Slots0);
            IReadOnlyList<Command> legal = LegalMoves.For(start.State);

            for (int action = 0; action < mask.Length; action++)
            {
                if (!mask[action]) continue;
                Command decoded = TacticalV2Coding.Decode(action, start.State, PlayerId.Player0, layout, start.Slots0);
                Assert.That(legal, Does.Contain(decoded), $"action {action} decoded to a command absent from LegalMoves");
            }
        }

        [Test]
        public void EveryMaskedAction_RoundTripsThroughThePublicEncoder()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(73);
            bool[] mask = TacticalV2Coding.Mask(start.State, PlayerId.Player0, layout, start.Slots0);

            for (int action = 0; action < mask.Length; action++)
            {
                if (!mask[action]) continue;
                Command command = TacticalV2Coding.Decode(action, start.State, PlayerId.Player0, layout, start.Slots0);

                Assert.That(TacticalV2Coding.TryEncode(command, start.State, layout, start.Slots0, out int encoded),
                    Is.True, $"masked action {action} must have a public encoding");
                Assert.That(encoded, Is.EqualTo(action));
            }
        }

        [Test]
        public void TryEncode_RejectsUnsupportedWrongSeatDeadMissingAndOffBoardCommands()
        {
            TacticalV2CodingFixture fixture = TacticalV2CodingFixture.WithEqualStatTemplates();
            Unit first = fixture.Game.Player(PlayerId.Player0).UnitsOnBoard[0];
            Unit second = fixture.Game.Player(PlayerId.Player0).UnitsOnBoard[1];
            Unit dead = first.WithDamage(first.CurrentHp);
            var deadRegistry = new TacticalV2UnitRegistry(fixture.Layout.UnitSlotCount);
            deadRegistry.Initialize(new[] { dead, second }, new[] { 0, 1 });
            var p0 = fixture.Game.Player(PlayerId.Player0);
            var p1 = fixture.Game.Player(PlayerId.Player1);
            var deadState = new GameState(
                fixture.Game.Board,
                fixture.Game.Config,
                new[]
                {
                    new PlayerState(PlayerId.Player0, p0.Points, p0.Barracks, new[] { dead, second }, p0.Generators),
                    p1,
                },
                PlayerId.Player0,
                fixture.Game.Round,
                fixture.Game.Config.RoundCap);

            (Command Command, GameState State, TacticalV2UnitRegistry Registry)[] rejected =
            {
                (new CreateUnit(PlayerId.Player0, first.Stats), fixture.Game, fixture.Slots0),
                (new EndTurn(PlayerId.Player1), fixture.Game, fixture.Slots0),
                (new MoveUnit(PlayerId.Player0, dead.Id, fixture.Layout.Cells[0]), deadState, deadRegistry),
                (new MoveUnit(PlayerId.Player0, 9999, fixture.Layout.Cells[0]), fixture.Game, fixture.Slots0),
                (new AttackUnit(PlayerId.Player0, first.Id, 9999), fixture.Game, fixture.Slots0),
                (new MoveUnit(PlayerId.Player0, first.Id, new HexCoord(-100, -100)),
                    fixture.Game, fixture.Slots0),
            };

            foreach (var item in rejected)
            {
                Assert.That(TacticalV2Coding.TryEncode(
                    item.Command, item.State, fixture.Layout, item.Registry, out int encoded), Is.False);
                Assert.That(encoded, Is.EqualTo(-1), "rejected commands must not map to EndTurn");
            }
        }

        [Test]
        public void Decode_DeployActionAddressesTemplateIndexAndCell()
        {
            // DeployUnit is never affordable at a fresh reset (players start at 0 points) and, by
            // design, MaxControllableUnits == StartingUnitCount means the starting roster already
            // fills every registry slot — so it never appears in the Mask/Decode round-trip test
            // above. Cover its offset math directly instead, against a registry with a free slot
            // (DecodeDeploy never consults `state`, only registry capacity, so an otherwise-empty
            // registry of matching capacity is a faithful probe of the offset math alone).
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(7);
            var registryWithFreeSlot = new TacticalV2UnitRegistry(layout.UnitSlotCount);

            int templateIndex = 2;
            int cellIndex = 5;
            int action = layout.DeployOffset + templateIndex * layout.CellCount + cellIndex;

            Command decoded =
                TacticalV2Coding.Decode(action, start.State, PlayerId.Player0, layout, registryWithFreeSlot);

            Assert.That(decoded, Is.EqualTo(new DeployUnit(PlayerId.Player0, templateIndex, layout.Cells[cellIndex])));
        }

        // ---- Capacity gate: deploy region must be masked/decoded out when the registry is full ----

        [Test]
        public void Mask_HidesDeployRegionWhenRegistryHasNoFreeSlot()
        {
            // The fixture's two registry slots already hold a living unit each (capacity == 2,
            // both seeded by Initialize). Give the player plenty of points so a deploy would
            // otherwise be affordable and placeable in the deployment zone's remaining empty
            // cells, then prove the mask still refuses every deploy index.
            TacticalV2CodingFixture fixture = TacticalV2CodingFixture.WithEqualStatTemplates(points: 1000);

            bool[] mask = TacticalV2Coding.Mask(fixture.Game, PlayerId.Player0, fixture.Layout, fixture.Slots0);

            for (int action = fixture.Layout.DeployOffset; action < fixture.Layout.ActionCount; action++)
                Assert.That(mask[action], Is.False,
                    $"deploy action {action} should be illegal once every registry slot holds a living unit");
        }

        [Test]
        public void Decode_DeployActionFallsBackToEndTurnWhenRegistryHasNoFreeSlot()
        {
            TacticalV2CodingFixture fixture = TacticalV2CodingFixture.WithEqualStatTemplates(points: 1000);
            int action = fixture.Layout.DeployOffset; // templateIndex 0, cell 0 — structurally valid deploy address

            Command decoded =
                TacticalV2Coding.Decode(action, fixture.Game, PlayerId.Player0, fixture.Layout, fixture.Slots0);

            Assert.That(decoded, Is.EqualTo(new EndTurn(PlayerId.Player0)));
        }

        [Test]
        public void Decode_OutOfRangeOrNonPositiveActionsFallBackToEndTurn()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            var layout = new TacticalV2Layout(config);
            TacticalV2Start start = layout.NewGame(7);

            Assert.That(TacticalV2Coding.Decode(0, start.State, PlayerId.Player0, layout, start.Slots0),
                Is.EqualTo(new EndTurn(PlayerId.Player0)));
            Assert.That(TacticalV2Coding.Decode(-5, start.State, PlayerId.Player0, layout, start.Slots0),
                Is.EqualTo(new EndTurn(PlayerId.Player0)));
            Assert.That(TacticalV2Coding.Decode(layout.ActionCount, start.State, PlayerId.Player0, layout, start.Slots0),
                Is.EqualTo(new EndTurn(PlayerId.Player0)));
        }

        // ---- TacticalV2UnitRegistry: stable slot identity ----

        [Test]
        public void Initialize_AssignsUnitsToSlotsInGivenOrder()
        {
            var stats = new UnitStats(3, 1, 0, 2, 1, 1, 0, 2, 1);
            var units = new[]
            {
                new Unit(11, PlayerId.Player0, stats, new HexCoord(0, 0), 0),
                new Unit(12, PlayerId.Player0, stats, new HexCoord(1, 0), 0),
            };
            var registry = new TacticalV2UnitRegistry(3);

            registry.Initialize(units, new[] { 2, 0 });

            Assert.That(registry.UnitIdAt(0), Is.EqualTo(11));
            Assert.That(registry.TemplateIndexAt(0), Is.EqualTo(2));
            Assert.That(registry.UnitIdAt(1), Is.EqualTo(12));
            Assert.That(registry.TemplateIndexAt(1), Is.EqualTo(0));
            Assert.That(registry.UnitIdAt(2), Is.EqualTo(-1));
            Assert.That(registry.SlotOf(12), Is.EqualTo(1));
        }

        [Test]
        public void RegisterDeployment_ClaimsLowestFreeSlotAndTracksTemplateIndex()
        {
            var stats = new UnitStats(3, 1, 0, 2, 1, 1, 0, 2, 1);
            var before = MakePlayer0State(Array.Empty<Unit>());
            var deployed = new Unit(21, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var after = MakePlayer0State(new[] { deployed });

            var registry = new TacticalV2UnitRegistry(2);
            registry.RegisterDeployment(before, after, PlayerId.Player0, templateIndex: 3);

            Assert.That(registry.SlotOf(21), Is.EqualTo(0));
            Assert.That(registry.TemplateIndexAt(0), Is.EqualTo(3));
        }

        [Test]
        public void RegisterDeployment_ThrowsWhenCapacityExceeded()
        {
            var stats = new UnitStats(3, 1, 0, 2, 1, 1, 0, 2, 1);
            var existing = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var registry = new TacticalV2UnitRegistry(1);
            registry.Initialize(new[] { existing }, new[] { 0 });

            var before = MakePlayer0State(new[] { existing });
            var deployed = new Unit(2, PlayerId.Player0, stats, new HexCoord(1, 0), 0);
            var after = MakePlayer0State(new[] { existing, deployed });

            Assert.Throws<InvalidOperationException>(() =>
                registry.RegisterDeployment(before, after, PlayerId.Player0, templateIndex: 0));
        }

        [Test]
        public void ReleaseDead_FreesSlotWithoutDisturbingOtherSlots()
        {
            var stats = new UnitStats(3, 1, 0, 2, 1, 1, 0, 2, 1);
            var alive = new Unit(1, PlayerId.Player0, stats, new HexCoord(0, 0), 0);
            var dying = new Unit(2, PlayerId.Player0, stats, new HexCoord(1, 0), 0);
            var registry = new TacticalV2UnitRegistry(3);
            registry.Initialize(new[] { alive, dying }, new[] { 0, 1 });

            var afterDeath = MakePlayer0State(new[] { alive }); // `dying` removed from the board (killed)
            registry.ReleaseDead(afterDeath, PlayerId.Player0);

            Assert.That(registry.SlotOf(1), Is.EqualTo(0));
            Assert.That(registry.UnitIdAt(1), Is.EqualTo(-1));
            Assert.That(registry.TemplateIndexAt(1), Is.EqualTo(-1));
        }

        private static GameState MakePlayer0State(IReadOnlyList<Unit> player0Units)
        {
            var board = new RandomBoardGenerator(BoardGenConfig.Default()).Generate(1);
            var p0 = new PlayerState(PlayerId.Player0, 0, Array.Empty<UnitTemplate>(), player0Units, null);
            var p1 = new PlayerState(PlayerId.Player1, 0, Array.Empty<UnitTemplate>(), Array.Empty<Unit>(), null);
            return new GameState(board, GameConfig.Default(biomesEnabled: false),
                new PlayerState[] { p0, p1 }, PlayerId.Player0, 1, 100);
        }

        /// <summary>Private helper: a two-template catalog whose templates share an identical stat line
        /// but distinct catalog ids, so any test built on it proves the codec resolves plane identity
        /// from the registry's recorded template index — never by comparing <see cref="UnitStats"/>.</summary>
        private sealed class TacticalV2CodingFixture
        {
            public GameState Game { get; }
            public TacticalV2Layout Layout { get; }
            public TacticalV2UnitRegistry Slots0 { get; }
            public TacticalV2UnitRegistry Slots1 { get; }

            private readonly IReadOnlyList<HexCoord> _cellByTemplateIndex;

            private TacticalV2CodingFixture(GameState game, TacticalV2Layout layout,
                TacticalV2UnitRegistry slots0, TacticalV2UnitRegistry slots1,
                IReadOnlyList<HexCoord> cellByTemplateIndex)
            {
                Game = game;
                Layout = layout;
                Slots0 = slots0;
                Slots1 = slots1;
                _cellByTemplateIndex = cellByTemplateIndex;
            }

            public static TacticalV2CodingFixture WithEqualStatTemplates(int points = 0)
            {
                var sharedStats = new UnitStats(5, 3, 2, 3, 2, 1, 1, 2, 1);
                var templates = new[]
                {
                    new TacticalV2Template("alpha-0001", new UnitTemplate("Alpha", sharedStats)),
                    new TacticalV2Template("bravo-0002", new UnitTemplate("Bravo", sharedStats)),
                };

                TacticalV2Config config = TacticalV2Config.Default();
                config.Templates = templates;
                config.StartingUnitCount = 2;
                config.MaxControllableUnits = 2;

                var layout = new TacticalV2Layout(config);
                var board = new RandomBoardGenerator(config.BoardGen).Generate(1);
                var zone0 = new List<HexCoord>(board.DeploymentZone(PlayerId.Player0));
                zone0.Sort((x, y) => x.Q != y.Q ? x.Q.CompareTo(y.Q) : x.R.CompareTo(y.R));
                HexCoord cellForTemplate0 = zone0[0];
                HexCoord cellForTemplate1 = zone0[1];

                var barracks = new List<UnitTemplate> { templates[0].Template, templates[1].Template };
                var unit0 = new Unit(1, PlayerId.Player0, templates[0].Template.Stats, cellForTemplate0,
                    board.TileAt(cellForTemplate0).Elevation, templates[0].Template.Name);
                var unit1 = new Unit(2, PlayerId.Player0, templates[1].Template.Stats, cellForTemplate1,
                    board.TileAt(cellForTemplate1).Elevation, templates[1].Template.Name);
                var p0 = new PlayerState(PlayerId.Player0, points, barracks, new[] { unit0, unit1 }, null);
                var p1 = new PlayerState(PlayerId.Player1, 0, barracks, Array.Empty<Unit>(), null);
                var state = new GameState(board, config.Game, new PlayerState[] { p0, p1 }, PlayerId.Player0, 1, 3);

                var slots0 = new TacticalV2UnitRegistry(2);
                slots0.Initialize(new[] { unit0, unit1 }, new[] { 0, 1 });
                var slots1 = new TacticalV2UnitRegistry(2);
                slots1.Initialize(Array.Empty<Unit>(), Array.Empty<int>());

                return new TacticalV2CodingFixture(state, layout, slots0, slots1,
                    new[] { cellForTemplate0, cellForTemplate1 });
            }

            public float ValueAtFriendlyTemplatePlane(float[] observation, int templateIndex)
            {
                int cellIndex = Layout.CellIndex[_cellByTemplateIndex[templateIndex]];
                int n = Layout.CellCount;
                return observation[templateIndex * n + cellIndex];
            }
        }
    }
}
