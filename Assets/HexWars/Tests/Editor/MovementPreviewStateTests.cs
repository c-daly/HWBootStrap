using HexWars.Engine;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    public class MovementPreviewStateTests
    {
        static readonly HexCoord A = new HexCoord(1, 0);
        static readonly HexCoord B = new HexCoord(2, 0);

        [Test]
        public void Hover_PreviewsReachableDestination_AndLeavingClearsIt()
        {
            var state = new MovementPreviewState();

            state.Hover(A);
            Assert.That(state.Destination, Is.EqualTo(A));

            state.Hover(null);
            Assert.That(state.Destination, Is.Null);
        }

        [Test]
        public void FirstTouchTapPreviews_SecondSameTapConfirms()
        {
            var state = new MovementPreviewState();

            Assert.That(state.Tap(A, reachable: true), Is.EqualTo(MovementPreviewDecision.Preview));
            Assert.That(state.Destination, Is.EqualTo(A));
            Assert.That(state.TouchLocked, Is.True);
            Assert.That(state.Tap(A, reachable: true), Is.EqualTo(MovementPreviewDecision.Confirm));
        }

        [Test]
        public void TouchingAnotherReachableDestinationSwitchesPreview()
        {
            var state = new MovementPreviewState();
            state.Tap(A, reachable: true);

            Assert.That(state.Tap(B, reachable: true), Is.EqualTo(MovementPreviewDecision.Preview));
            Assert.That(state.Destination, Is.EqualTo(B));
        }

        [Test]
        public void UnreachableTapClearsPreview()
        {
            var state = new MovementPreviewState();
            state.Tap(A, reachable: true);

            Assert.That(state.Tap(B, reachable: false), Is.EqualTo(MovementPreviewDecision.None));
            Assert.That(state.Destination, Is.Null);
            Assert.That(state.TouchLocked, Is.False);
        }

        [Test]
        public void ClearResetsDesktopAndTouchState()
        {
            var state = new MovementPreviewState();
            state.Tap(A, reachable: true);

            state.Clear();

            Assert.That(state.Destination, Is.Null);
            Assert.That(state.TouchLocked, Is.False);
        }
    }
}
