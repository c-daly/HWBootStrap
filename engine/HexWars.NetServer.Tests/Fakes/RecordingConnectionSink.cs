using HexWars.NetServer.Runtime;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// The socket layer as a list.
    ///
    /// Order is the whole point of it. The durable rule the coordinator exists to keep is that a command is
    /// in the journal before anybody is told it happened, and an assertion about that is an assertion about
    /// sequence: which message reached which connection, and when relative to the writes. <see cref="OnSend"/>
    /// goes one step further and runs INSIDE the send, so a test can read the store at the exact instant a
    /// frame leaves rather than afterwards, when a later write could cover for a missing earlier one.
    /// </summary>
    public sealed class RecordingConnectionSink : IConnectionSink
    {
        readonly object _gate = new();
        readonly List<(string ConnectionId, string Message)> _sent = new();
        readonly List<(string ConnectionId, int CloseStatus, string Reason)> _closed = new();

        /// <summary>Runs while the send is happening, before the coordinator regains control.</summary>
        public Action<string, string>? OnSend { get; set; }

        /// <summary>Every frame sent, oldest first.</summary>
        public IReadOnlyList<(string ConnectionId, string Message)> Sent
        {
            get { lock (_gate) return _sent.ToArray(); }
        }

        /// <summary>Every close asked for, oldest first.</summary>
        public IReadOnlyList<(string ConnectionId, int CloseStatus, string Reason)> Closed
        {
            get { lock (_gate) return _closed.ToArray(); }
        }

        public void Send(string connectionId, string message)
        {
            lock (_gate) _sent.Add((connectionId, message));
            OnSend?.Invoke(connectionId, message);
        }

        public void Close(string connectionId, int closeStatus, string reason)
        {
            lock (_gate) _closed.Add((connectionId, closeStatus, reason));
        }

        /// <summary>What one connection was told, in order.</summary>
        public IReadOnlyList<string> MessagesFor(string connectionId)
        {
            lock (_gate)
            {
                return _sent
                    .Where(s => string.Equals(s.ConnectionId, connectionId, StringComparison.Ordinal))
                    .Select(s => s.Message)
                    .ToArray();
            }
        }

        /// <summary>Forgets everything recorded so far, so a test can assert about one step in isolation.</summary>
        public void Clear()
        {
            lock (_gate)
            {
                _sent.Clear();
                _closed.Clear();
            }
        }
    }
}
