using HexWars.Engine;
using HexWars.Presentation;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>Covers the pure "which state does hover/inspection read against" seam
    /// (<see cref="UnitInputController.ResolveInspectionState"/>) extracted to fix the arena hover bug:
    /// the arena GameObject has no <c>GameBootstrap</c>, so the old direct <c>_game.State</c> read threw
    /// before the tooltip could ever show. Mouse-driven hover and window-level behavior are exercised
    /// manually (see the batch report), not here.</summary>
    public sealed class UnitInputControllerTests
    {
        [Test]
        public void ResolveInspectionState_PrefersGameStateOverPresentedState()
        {
            GameState gameState = MinimalState();
            GameState presentedState = MinimalState();

            Assert.That(
                UnitInputController.ResolveInspectionState(gameState, presentedState),
                Is.SameAs(gameState));
        }

        [Test]
        public void ResolveInspectionState_FallsBackToPresentedStateWhenNoGameState()
        {
            GameState presentedState = MinimalState();

            Assert.That(
                UnitInputController.ResolveInspectionState(null, presentedState),
                Is.SameAs(presentedState));
        }

        [Test]
        public void ResolveInspectionState_NullWhenNeitherSourceHasAState()
        {
            Assert.That(UnitInputController.ResolveInspectionState(null, null), Is.Null);
        }

        static GameState MinimalState()
        {
            var board = new Board(new[] { new Tile(new HexCoord(0, 0), 0, TerrainType.Plains) });
            var players = new[]
            {
                new PlayerState(PlayerId.Player0, 0),
                new PlayerState(PlayerId.Player1, 0),
            };
            return new GameState(board, GameConfig.Default(), players, PlayerId.Player0, round: 1, nextEntityId: 1);
        }
    }
}
