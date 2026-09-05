#nullable enable
using System;
using System.Threading.Tasks;

namespace HexWars.Presentation
{
    /// <summary>
    /// The socket one <see cref="SteamMatchSession"/> connection attempt runs on.
    /// <para>
    /// Every call and every event carries the attempt id it belongs to, and that is the whole point
    /// of the interface. A socket that is being torn down keeps firing for a while, and without an id
    /// on the event there is no way to tell a dying attempt apart from the live one: attempt A could
    /// close, seat or authenticate attempt B. The session ignores any id that is not
    /// <see cref="SteamMatchSession.CurrentAttempt"/>, so a late frame can no longer do that.
    /// </para>
    /// <para>
    /// Implementations may raise the events from any thread; <see cref="SteamMatchSocketPump"/> is
    /// what puts them back on the main thread.
    /// </para>
    /// </summary>
    public interface ISteamSocketDriver
    {
        /// <summary>Starts an attempt. Any socket left from an earlier attempt is abandoned first.</summary>
        void Open(string url, int attemptId);

        /// <summary>Sends one frame, but only while <paramref name="attemptId"/> still owns the socket.</summary>
        void Send(int attemptId, string text);

        /// <summary>
        /// Closes the attempt socket. The task completes when the close is done, so the caller can wait
        /// for it (bounded) before opening the next attempt. Closing an attempt that no longer owns the
        /// socket is a no-op that completes at once.
        /// </summary>
        Task CloseAsync(int attemptId);

        /// <summary>The attempt socket opened.</summary>
        event Action<int> Opened;

        /// <summary>One inbound frame, with the attempt that received it.</summary>
        event Action<int, string> Message;

        /// <summary>The attempt socket closed, with the close description.</summary>
        event Action<int, string> Closed;

        /// <summary>A socket error. Diagnostic only: a close follows when it was fatal.</summary>
        event Action<int, string> Error;
    }
}
