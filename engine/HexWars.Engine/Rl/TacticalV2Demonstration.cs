using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    /// <summary>An immutable tactical-v2 teacher decision captured from the exact state visible
    /// immediately before its accepted command was applied.</summary>
    public sealed class TacticalV2Demonstration
    {
        private readonly float[] _observation;
        private readonly bool[] _legalMask;
        private readonly TacticalTraceCommand _command;

        public TacticalV2Demonstration(
            float[] observation,
            bool[] legalMask,
            int action,
            int seat,
            TacticalTraceCommand command)
        {
            if (observation == null) throw new ArgumentNullException(nameof(observation));
            if (legalMask == null) throw new ArgumentNullException(nameof(legalMask));
            if (command == null) throw new ArgumentNullException(nameof(command));
            _observation = (float[])observation.Clone();
            _legalMask = (bool[])legalMask.Clone();
            Action = action;
            Seat = seat;
            _command = Copy(command);
        }

        public float[] Observation => (float[])_observation.Clone();
        public bool[] LegalMask => (bool[])_legalMask.Clone();
        public int Action { get; }
        public int Seat { get; }
        public TacticalTraceCommand Command => Copy(_command);

        private static TacticalTraceCommand Copy(TacticalTraceCommand source) =>
            new TacticalTraceCommand
            {
                Kind = source.Kind,
                Issuer = source.Issuer,
                ActorId = source.ActorId,
                TargetId = source.TargetId,
                Q = source.Q,
                R = source.R,
            };
    }

    public interface ITacticalV2DemonstrationSink
    {
        bool Enabled { get; }
        void Reset();
        void Accepted(TacticalV2Demonstration decision);
    }

    public sealed class BufferedTacticalV2DemonstrationSink : ITacticalV2DemonstrationSink
    {
        private readonly List<TacticalV2Demonstration> _items =
            new List<TacticalV2Demonstration>();

        public bool Enabled { get; set; }

        public void Reset() => _items.Clear();

        public void Accepted(TacticalV2Demonstration decision)
        {
            if (decision == null) throw new ArgumentNullException(nameof(decision));
            if (Enabled) _items.Add(decision);
        }

        public IReadOnlyList<TacticalV2Demonstration> Drain()
        {
            var result = new List<TacticalV2Demonstration>(_items);
            _items.Clear();
            return result;
        }
    }
}
