using System.Collections.Concurrent;
using HexWars.NetServer.Runtime;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// Every live protocol-v2 socket on this host, and the <see cref="IConnectionSink"/> the coordinator
    /// broadcasts through.
    ///
    /// One object under both names on purpose. The coordinator wants somewhere to put a frame addressed to
    /// a connection id and nothing else; the socket route and the heartbeat want the connections
    /// themselves. Splitting them would mean two dictionaries that have to agree, and the moment they did
    /// not, a frame would be queued for a socket that had already gone.
    /// </summary>
    public sealed class V2ConnectionRegistry(ILogger<V2ConnectionRegistry> logger) : IConnectionSink
    {
        readonly ConcurrentDictionary<string, V2Connection> _connections = new(StringComparer.Ordinal);

        int _maxQueueDepth;

        /// <summary>Sockets currently registered, authenticated or not.</summary>
        public int Count => _connections.Count;

        /// <summary>
        /// The deepest any outbound queue on this host has been observed. Metrics only, and deliberately a
        /// high-water mark rather than a current reading: back pressure that only shows up in a burst is
        /// invisible to anything sampled on an interval.
        /// </summary>
        public int MaxQueueDepth => Volatile.Read(ref _maxQueueDepth);

        internal void Add(V2Connection connection) => _connections[connection.Id] = connection;

        internal void Remove(string connectionId)
        {
            if (_connections.TryRemove(connectionId, out V2Connection? connection))
                Observe(connection.MaxQueueDepth);
        }

        internal bool TryGet(string connectionId, out V2Connection? connection) =>
            _connections.TryGetValue(connectionId, out connection);

        /// <summary>A point-in-time copy, so a caller can close connections while walking it.</summary>
        internal IReadOnlyCollection<V2Connection> Snapshot() => _connections.Values.ToArray();

        /// <summary>How many sockets one address is holding. The material for a per-address cap, and worth
        /// having before one is needed: a count nobody kept is a count nobody can act on.</summary>
        public int CountForIp(string ip)
        {
            int count = 0;
            foreach (V2Connection connection in _connections.Values)
                if (string.Equals(connection.RemoteIp, ip, StringComparison.Ordinal)) count++;

            return count;
        }

        /// <summary>Queues one frame for one connection. A connection that has gone away is not an error:
        /// the coordinator decides what to send before it can know whether the socket is still there.</summary>
        public void Send(string connectionId, string message)
        {
            if (!_connections.TryGetValue(connectionId, out V2Connection? connection)) return;

            // Read before the attempt, because a full queue makes TryEnqueue close the connection: asking
            // afterwards would report every dropped frame as a slow client, including the ones dropped for
            // a socket that was already on its way out.
            bool alreadyClosing = connection.IsClosed;

            if (connection.TryEnqueue(message))
            {
                Observe(connection.QueueDepth);
                return;
            }

            // TryEnqueue has already asked the connection to close; this is only the record of why.
            if (alreadyClosing) return;

            logger.LogWarning(
                "Closed a socket that stopped reading: its outbound queue was full at {Depth} frames",
                connection.QueueDepth);
        }

        /// <summary>Asks for a connection to be closed. Returns immediately: the coordinator calls this
        /// while holding a match gate, and the close itself waits on a socket.</summary>
        public void Close(string connectionId, int closeStatus, string reason)
        {
            if (!_connections.TryGetValue(connectionId, out V2Connection? connection)) return;

            _ = connection.CloseAsync(closeStatus, reason);
        }

        void Observe(int depth)
        {
            int seen = Volatile.Read(ref _maxQueueDepth);
            while (depth > seen)
            {
                int prior = Interlocked.CompareExchange(ref _maxQueueDepth, depth, seen);
                if (prior == seen) return;
                seen = prior;
            }
        }
    }
}
