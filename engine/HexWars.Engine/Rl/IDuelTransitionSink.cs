using System;
using System.Collections.Generic;

namespace HexWars.Engine.Rl
{
    public interface IDuelTransitionSink
    {
        void Reset(GameState initialState);
        void Accepted(DuelTransition transition);
    }

    public sealed class NullDuelTransitionSink : IDuelTransitionSink
    {
        public static readonly NullDuelTransitionSink Instance = new NullDuelTransitionSink();
        private NullDuelTransitionSink() { }
        public void Reset(GameState initialState) { }
        public void Accepted(DuelTransition transition) { }
    }

    public sealed class BufferedDuelTransitionSink : IDuelTransitionSink
    {
        private readonly List<DuelTransition> _items = new List<DuelTransition>();
        public bool Enabled { get; set; }

        public void Reset(GameState initialState) => _items.Clear();

        public void Accepted(DuelTransition transition)
        {
            if (Enabled)
                _items.Add(transition);
        }

        public IReadOnlyList<DuelTransition> Drain()
        {
            var result = new List<DuelTransition>(_items);
            _items.Clear();
            return result;
        }
    }
}
