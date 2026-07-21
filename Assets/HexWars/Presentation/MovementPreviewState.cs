using HexWars.Engine;

namespace HexWars.Presentation
{
    public enum MovementPreviewDecision { None, Preview, Confirm }

    public sealed class MovementPreviewState
    {
        public HexCoord? Destination { get; private set; }
        public bool TouchLocked { get; private set; }

        public void Hover(HexCoord? destination)
        {
            if (!TouchLocked) Destination = destination;
        }

        public MovementPreviewDecision Tap(HexCoord destination, bool reachable)
        {
            if (!reachable)
            {
                Clear();
                return MovementPreviewDecision.None;
            }

            if (TouchLocked && Destination == destination)
                return MovementPreviewDecision.Confirm;

            Destination = destination;
            TouchLocked = true;
            return MovementPreviewDecision.Preview;
        }

        public void Clear()
        {
            Destination = null;
            TouchLocked = false;
        }
    }
}
