using System;
using System.Collections.Generic;
using System.Linq;
using HexWars.Engine;
using HexWars.Engine.Rl;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    public class TacticalV2ProfiledStartTests
    {
        [Test]
        public void SameSeedProfileAndReferenceSeat_ConstructsIdenticalStateAndGeometry()
        {
            var layout = new TacticalV2Layout(Config());
            TacticalV2StartProfile profile = Profile(layout, "conversion-2v1-far");
            TacticalV2Start first = layout.NewGame(6000005, profile, PlayerId.Player0);
            TacticalV2Start second = layout.NewGame(6000005, profile, PlayerId.Player0);
            Assert.That(State(first.State), Is.EqualTo(State(second.State)));
            Assert.That(Cells(first.State, PlayerId.Player0), Is.EqualTo(Cells(second.State, PlayerId.Player0)));
            Assert.That(Cells(first.State, PlayerId.Player1), Is.EqualTo(Cells(second.State, PlayerId.Player1)));
        }

        [Test]
        public void ReferenceSeatRoleReversal_SwapsActualOccupiedGeometry()
        {
            var layout = new TacticalV2Layout(Config());
            TacticalV2StartProfile profile = Profile(layout, "conversion-2v1-far");
            TacticalV2Start p0 = layout.NewGame(6000005, profile, PlayerId.Player0);
            TacticalV2Start p1 = layout.NewGame(6000005, profile, PlayerId.Player1);
            Assert.That(Cells(p0.State, PlayerId.Player0), Is.EqualTo(Cells(p1.State, PlayerId.Player1)));
            Assert.That(Cells(p0.State, PlayerId.Player1), Is.EqualTo(Cells(p1.State, PlayerId.Player0)));
        }

        [TestCase("conversion-1v1-near", 2, 3)]
        [TestCase("conversion-1v1-medium", 4, 6)]
        [TestCase("conversion-1v1-far", 7, int.MaxValue)]
        public void ConversionProfiles_RespectDistanceBands(string id, int minimum, int maximum)
        {
            var layout = new TacticalV2Layout(Config());
            TacticalV2Start start = layout.NewGame(6000005, Profile(layout, id), PlayerId.Player0);
            int closest = Closest(start.State);
            Assert.That(closest, Is.GreaterThanOrEqualTo(minimum));
            Assert.That(closest, Is.LessThanOrEqualTo(maximum));
        }

        [Test]
        public void StandardProfile_IsExactlyEquivalentToLegacySymmetricConstruction()
        {
            var layout = new TacticalV2Layout(Config());
            TacticalV2Start legacy = layout.NewGame(71);
            TacticalV2Start profiled = layout.NewGame(71, Profile(layout, "standard-3v3"), PlayerId.Player1);
            Assert.That(State(profiled.State), Is.EqualTo(State(legacy.State)));
            Assert.That(Cells(profiled.State, PlayerId.Player0), Is.EqualTo(Cells(legacy.State, PlayerId.Player0)));
        }

        [Test]
        public void UndeclaredProfile_FailsAtLayoutAndResetApi()
        {
            TacticalV2Config config = Config();
            var layout = new TacticalV2Layout(config);
            var duel = new TacticalV2DuelEnv(config);
            var undeclared = new TacticalV2StartProfile("not-declared", 1, 1, "near");
            Assert.Throws<ArgumentException>(() => layout.NewGame(71, undeclared, PlayerId.Player0));
            Assert.Throws<ArgumentException>(() => duel.Reset(71, null, null, "not-declared", PlayerId.Player0));
        }

        [Test]
        public void WorkerCountInvariant_UsesSameSelectedProfileAndConstructedStateForSameEpisodeSeed()
        {
            TacticalV2Config config = Config();
            int seed = 6000005;
            var oneWorker = new TacticalV2Env(_ => new GreedyAgent(0), PlayerId.Player0, config);
            var fourWorkers = new TacticalV2Env(_ => new GreedyAgent(3), PlayerId.Player0, config);
            oneWorker.Reset(seed);
            fourWorkers.Reset(seed);
            Assert.That(oneWorker.SelectedStartProfileId, Is.EqualTo(config.StartDistribution.Select(seed)));
            Assert.That(fourWorkers.SelectedStartProfileId, Is.EqualTo(oneWorker.SelectedStartProfileId));
            Assert.That(State(fourWorkers.State), Is.EqualTo(State(oneWorker.State)));
            Assert.That(Cells(fourWorkers.State, PlayerId.Player0), Is.EqualTo(Cells(oneWorker.State, PlayerId.Player0)));
        }

        private static TacticalV2Config Config()
        {
            TacticalV2Config config = TacticalV2Config.Default();
            config.PlacementPolicy = "profiled-seeded-v1";
            config.StartProfiles = TacticalV2StartCatalog.ProfiledSeededV1();
            config.StartDistribution = new TacticalV2StartDistribution(config.StartProfiles.Select(profile =>
                new TacticalV2StartWeight(profile.Id, profile.Id == "conversion-2v1-far" ? 10000 : 0)));
            return config;
        }

        private static TacticalV2StartProfile Profile(TacticalV2Layout layout, string id) =>
            TacticalV2StartCatalog.ProfiledSeededV1().Single(profile => profile.Id == id);

        private static string State(GameState state) => ReplayFile.Write(state, Array.Empty<Command>());

        private static List<HexCoord> Cells(GameState state, PlayerId seat) => state.Player(seat).UnitsOnBoard
            .Where(unit => unit.IsAlive).Select(unit => unit.Cell)
            .OrderBy(cell => cell.Q).ThenBy(cell => cell.R).ToList();

        private static int Closest(GameState state) => state.Player(PlayerId.Player0).UnitsOnBoard
            .SelectMany(left => state.Player(PlayerId.Player1).UnitsOnBoard
                .Select(right => HexCoord.Distance(left.Cell, right.Cell))).Min();
    }
}
