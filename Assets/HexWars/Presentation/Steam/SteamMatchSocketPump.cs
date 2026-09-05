#nullable enable
using System;
using System.Collections.Generic;

namespace HexWars.Presentation
{
    /// <summary>
    /// Carries socket events from an <see cref="ISteamSocketDriver"/> to a
    /// <see cref="SteamMatchSession"/>, keeping the attempt id on every one of them.
    /// <para>
    /// The driver may raise its events from any thread, so nothing here touches the session directly:
    /// events are queued under a lock and <see cref="Pump"/> replays them, in order, on whichever
    /// thread calls it. That is the single place the session is fed, which is what keeps every Unity
    /// call that follows from an output on the main thread.
    /// </para>
    /// <para>
    /// This is deliberately plain C# so the dispatch that ships is the dispatch the tests exercise: an
    /// event from an abandoned attempt has to be provably incapable of seating or closing a live one.
    /// </para>
    /// </summary>
    public sealed class SteamMatchSocketPump : IDisposable
    {
        enum EventKind
        {
            Opened,
            Message,
            Closed,
            Error,
        }

        readonly struct SocketEvent
        {
            public SocketEvent(EventKind kind, int attemptId, string text)
            {
                Kind = kind;
                AttemptId = attemptId;
                Text = text;
            }

            public readonly EventKind Kind;
            public readonly int AttemptId;
            public readonly string Text;
        }

        readonly SteamMatchSession _session;
        readonly ISteamSocketDriver _driver;
        readonly Action<string>? _log;
        readonly Queue<SocketEvent> _queue = new Queue<SocketEvent>();
        readonly object _gate = new object();

        bool _disposed;

        /// <param name="log">Where socket errors go. Null discards them.</param>
        public SteamMatchSocketPump(SteamMatchSession session, ISteamSocketDriver driver, Action<string>? log = null)
        {
            if (session == null) throw new ArgumentNullException(nameof(session));
            if (driver == null) throw new ArgumentNullException(nameof(driver));

            _session = session;
            _driver = driver;
            _log = log;

            _driver.Opened += OnOpened;
            _driver.Message += OnMessage;
            _driver.Closed += OnClosed;
            _driver.Error += OnError;
        }

        /// <summary>How many driver events are waiting for the next <see cref="Pump"/>.</summary>
        public int QueuedEvents
        {
            get { lock (_gate) { return _queue.Count; } }
        }

        /// <summary>Replays every queued driver event into the session, oldest first.</summary>
        public void Pump()
        {
            while (true)
            {
                SocketEvent next;
                lock (_gate)
                {
                    if (_queue.Count == 0) return;
                    next = _queue.Dequeue();
                }

                switch (next.Kind)
                {
                    case EventKind.Opened:
                        _session.Opened(next.AttemptId);
                        break;
                    case EventKind.Message:
                        _session.Frame(next.AttemptId, next.Text);
                        break;
                    case EventKind.Closed:
                        _session.Closed(next.AttemptId);
                        break;
                    case EventKind.Error:
                        if (_log != null) _log(next.Text);
                        break;
                }
            }
        }

        /// <summary>Unsubscribes and drops everything still queued. Nothing reaches the session after this.</summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            _driver.Opened -= OnOpened;
            _driver.Message -= OnMessage;
            _driver.Closed -= OnClosed;
            _driver.Error -= OnError;

            lock (_gate) { _queue.Clear(); }
        }

        void OnOpened(int attemptId)
        {
            Enqueue(new SocketEvent(EventKind.Opened, attemptId, string.Empty));
        }

        void OnMessage(int attemptId, string text)
        {
            Enqueue(new SocketEvent(EventKind.Message, attemptId, text ?? string.Empty));
        }

        void OnClosed(int attemptId, string reason)
        {
            Enqueue(new SocketEvent(EventKind.Closed, attemptId, reason ?? string.Empty));
        }

        void OnError(int attemptId, string message)
        {
            Enqueue(new SocketEvent(EventKind.Error, attemptId,
                "attempt " + attemptId + ": " + (message ?? string.Empty)));
        }

        void Enqueue(SocketEvent socketEvent)
        {
            lock (_gate)
            {
                if (_disposed) return;
                _queue.Enqueue(socketEvent);
            }
        }
    }
}
