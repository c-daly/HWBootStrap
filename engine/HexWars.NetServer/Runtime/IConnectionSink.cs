namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// Where the coordinator puts the frames it decides to send.
    ///
    /// The coordinator is the whole v2 brain and owns no socket, for the same reason the legacy
    /// <see cref="HexWars.Engine.MatchHub"/> returns messages instead of writing them: a rule about ordering
    /// - the journal first, the broadcast second - is only testable if the broadcast is something a test can
    /// stand in front of. The websocket layer implements this over the outbound channel of each connection;
    /// tests implement it as a list.
    ///
    /// Both methods are deliberately synchronous and must not block. They are called while the per-match gate
    /// is held, so an implementation that waited on a network write would stall every other player in that
    /// match; the real one hands the frame to a bounded per-connection channel and returns.
    /// </summary>
    public interface IConnectionSink
    {
        /// <summary>Queues one frame for one connection. A connection that has gone away is not an error.</summary>
        void Send(string connectionId, string message);

        /// <summary>Asks for a connection to be closed with a WebSocket close status and reason.</summary>
        void Close(string connectionId, int closeStatus, string reason);
    }
}
