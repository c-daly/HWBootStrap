using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class MovementHighlightClassifierTests
    {
        [Test]
        public void Classify_DistinguishesReachableRouteExpensiveDestinationAndNone()
        {
            var start = new HexCoord(0, 0);
            var normal = new HexCoord(1, 0);
            var expensive = new HexCoord(2, 0);
            var destination = new HexCoord(3, 0);
            var reachableOnly = new HexCoord(0, 1);
            var board = new Board(new[]
            {
                new Tile(start, 0, TerrainType.Plains),
                new Tile(normal, 0, TerrainType.Plains),
                new Tile(expensive, 1, TerrainType.Forest),
                new Tile(destination, 2, TerrainType.Forest),
                new Tile(reachableOnly, 0, TerrainType.Plains),
            });
            var state = new GameState(board, GameConfig.Default(), new[]
            {
                new PlayerState(PlayerId.Player0, 0),
                new PlayerState(PlayerId.Player1, 0),
            }, PlayerId.Player0, 1, 1);
            var route = new MovementRoute(
                new[] { start, normal, expensive, destination }, 5, 2, 0, 0);

            Assert.Multiple(() =>
            {
                Assert.That(MovementHighlightClassifier.Classify(state, route, reachableOnly, true),
                    Is.EqualTo(MovementHighlightKind.Reachable));
                Assert.That(MovementHighlightClassifier.Classify(state, route, start, false),
                    Is.EqualTo(MovementHighlightKind.Route));
                Assert.That(MovementHighlightClassifier.Classify(state, route, normal, false),
                    Is.EqualTo(MovementHighlightKind.Route));
                Assert.That(MovementHighlightClassifier.Classify(state, route, expensive, false),
                    Is.EqualTo(MovementHighlightKind.Expensive));
                Assert.That(MovementHighlightClassifier.Classify(state, route, destination, false),
                    Is.EqualTo(MovementHighlightKind.Destination));
                Assert.That(MovementHighlightClassifier.Classify(state, route,
                    new HexCoord(99, 99), false), Is.EqualTo(MovementHighlightKind.None));
            });
        }
    }
}
