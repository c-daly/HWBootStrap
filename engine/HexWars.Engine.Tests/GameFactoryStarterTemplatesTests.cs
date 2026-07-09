using System.Linq;
using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Engine.Tests
{
    /// <summary>Every game's barracks starts pre-loaded with five named example designs (spec §5):
    /// deployable turn one, deletable per-game, teaching that a statline is a concept. The first three
    /// match the existing starting-army roster exactly (RL contract: TacticalLayout keeps its own
    /// roster, so this only affects human/AI matches built through GameFactory).</summary>
    public class GameFactoryStarterTemplatesTests
    {
        private static readonly (string Name, UnitStats Stats)[] Expected =
        {
            ("Brute",     new UnitStats(7, 2, 2, 3, 2, 1, 1, 2, 1)),
            ("Striker",   new UnitStats(2, 6, 0, 3, 2, 2, 1, 3, 1)),
            ("Sniper",    new UnitStats(2, 2, 0, 2, 2, 6, 1, 4, 1)),
            ("Artillery", new UnitStats(3, 6, 0, 0, 0, 5, 2, 2, 1)),
            ("Scout",     new UnitStats(2, 0, 0, 4, 3, 0, 0, 7, 2)),
        };

        [Test]
        public void Build_Annihilation_SeedsFiveNamedTemplates_BothPlayers()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
                Assert.That(state.Player(pid).Barracks.Count, Is.EqualTo(5));
        }

        [Test]
        public void Build_Territory_SeedsFiveNamedTemplates_BothPlayers()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Territory, 11, 9, 40, 7));
            foreach (var pid in new[] { PlayerId.Player0, PlayerId.Player1 })
                Assert.That(state.Player(pid).Barracks.Count, Is.EqualTo(5));
        }

        [Test]
        public void Build_StarterTemplates_MatchSpecTable_Exactly()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var barracks = state.Player(PlayerId.Player0).Barracks;
            for (int i = 0; i < Expected.Length; i++)
            {
                Assert.That(barracks[i].Name, Is.EqualTo(Expected[i].Name), $"slot {i} name");
                Assert.That(barracks[i].Stats, Is.EqualTo(Expected[i].Stats), $"slot {i} stats");
            }
        }

        [Test]
        public void Build_ClassicTrio_AtIndicesZeroToTwo_MatchesOnBoardRoster()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7,
                armySize: 3, brutes: 1, strikers: 1, snipers: 1));
            var barracks = state.Player(PlayerId.Player0).Barracks;
            var onBoard = state.Player(PlayerId.Player0).UnitsOnBoard.OrderBy(u => u.Id).ToArray();

            Assert.That(barracks[0].Stats, Is.EqualTo(onBoard[0].Stats), "Brute matches the on-board Brute");
            Assert.That(barracks[1].Stats, Is.EqualTo(onBoard[1].Stats), "Striker matches the on-board Striker");
            Assert.That(barracks[2].Stats, Is.EqualTo(onBoard[2].Stats), "Sniper matches the on-board Sniper");
        }

        [Test]
        public void DeleteTemplate_DuringGame_DoesNotReseed()
        {
            var state = GameFactory.Build(new GameSetup(GameMode.Annihilation, 9, 7, 12, 7));
            var afterDelete = GameEngine.Apply(state, new DeleteTemplate(PlayerId.Player0, 4)).NewState; // remove Scout
            Assert.That(afterDelete.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(4));

            var afterRound = GameEngine.Apply(
                GameEngine.Apply(afterDelete, new EndTurn(PlayerId.Player0)).NewState,
                new EndTurn(PlayerId.Player1)).NewState;
            Assert.That(afterRound.Player(PlayerId.Player0).Barracks.Count, Is.EqualTo(4),
                "a full round must not re-seed the deleted template back in");
        }
    }
}
