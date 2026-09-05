using System.Collections.Concurrent;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Runtime;
using Microsoft.Extensions.Options;

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
    public sealed class V2ConnectionRegistry(
        IOptions<MatchHostingOptions> options, ILogger<V2ConnectionRegistry> logger) : IConnectionSink
    {
        readonly ConcurrentDictionary<string, V2Connection> _connections = new(StringComparer.Ordinal);

        /// <summary>Sockets held or promised per address. Counting the accepted ones after the fact is not
        /// a cap: a hundred upgrades that arrive together all read the same low count and all pass.</summary>
        readonly ConcurrentDictionary<string, int> _perAddress = new(StringComparer.Ordinal);

        readonly int _maxPerAddress = options.Value.MaxSocketsPerIp;

        int _maxQueueDepth;

        /// <summary>Sockets currently registered, authenticated or not.</summary>
        public int Count => _connections.Count;

        /// <summary>
        /// The deepest any outbound queue on this host has been observed. Metrics only, and deliberately a
        /// high-water mark rather than a current reading: back pressure that only shows up in a burst is
        /// invisible to anything sampled on an interval.
        /// </summary>
        public int MaxQueueDepth => Volatile.Read(ref _maxQueueDepth);

        /// <summary>
        /// Claims one slot for an address, or refuses.
        ///
        /// Taken BEFORE the upgrade and released by <see cref="Remove"/>, so the count a caller is judged
        /// against includes the sockets that are still being accepted. A check-then-accept against a count
        /// of live connections is not a cap at all: every upgrade in a simultaneous burst reads the same
        /// number and every one of them passes it.
        /// </summary>
        public bool TryReserve(string ip)
        {
            ArgumentNullException.ThrowIfNull(ip);

            while (true)
            {
                if (!_perAddress.TryGetValue(ip, out int held))
                {
                    if (_maxPerAddress < 1) return false;
                    if (_perAddress.TryAdd(ip, 1)) return true;
                    continue;
                }

                if (held >= _maxPerAddress) return false;
                if (_perAddress.TryUpdate(ip, held + 1, held)) return true;
            }
        }

        /// <summary>Hands a reservation back. Called when the upgrade never became a connection; a socket
        /// that did become one gives its slot back through <see cref="Remove"/> instead.</summary>
        public void Release(string ip)
        {
            ArgumentNullException.ThrowIfNull(ip);

            while (true)
            {
                if (!_perAddress.TryGetValue(ip, out int held)) return;

                if (held <= 1)
                {
                    if (_perAddress.TryRemove(new KeyValuePair<string, int>(ip, held))) return;
                    continue;
                }

                if (_perAddress.TryUpdate(ip, held - 1, held)) return;
            }
        }

        /// <summary>Takes ownership of the reservation already made for this connection address.</summary>
        internal void Add(V2Connection connection) => _connections[connection.Id] = connection;

        internal void Remove(string connectionId)
        {
            if (!_connections.TryRemove(connectionId, out V2Connection? connection)) return;

            Observe(connection.MaxQueueDepth);
            Release(connection.RemoteIp);
        }

        internal bool TryGet(string connectionId, out V2Connection? connection) =>
            _connections.TryGetValue(connectionId, out connection);

        /// <summary>A point-in-time copy, so a caller can close connections while walking it.</summary>
        internal IReadOnlyCollection<V2Connection> Snapshot() => _connections.Values.ToArray();

        /// <summary>How many slots one address holds, reservations that have not yet become sockets
        /// included. That is the number the cap is enforced on, so it is the number worth reporting.</summary>
        public int CountForIp(string ip) => _perAddress.TryGetValue(ip, out int held) ? held : 0;

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
