using System.Net.WebSockets;
using System.Text;
using System.Threading.Channels;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// One live protocol-v2 socket: its identity, its liveness stamp, and an ordered outbound queue that a
    /// single writer task drains.
    ///
    /// The queue is BOUNDED, which is the one real difference from the legacy <c>Conn</c>. The coordinator
    /// hands frames over synchronously while holding a match gate, so enqueueing must never wait; with an
    /// unbounded queue that promise is kept by letting a client which has stopped reading accumulate frames
    /// until the process runs out of memory, and every other match on the host pays for it. Bounded turns
    /// that into a decision: a connection whose queue fills is not slow, it is gone, and it is closed with
    /// 1008 so the client reconnects and resyncs from START rather than resuming a stream with a hole in it.
    /// </summary>
    internal sealed class V2Connection
    {
        /// <summary>The close status a client that stopped reading gets.</summary>
        public const int SlowClientCloseStatus = 1008;

        public const string SlowClientReason = "slow client";

        /// <summary>
        /// How long a close waits for the writer to finish the frame it is on.
        ///
        /// Waiting at all is what gets the last frame out - AUTH FAIL, or the APPLY that ended the game, is
        /// usually the one still in flight when the close is decided. Waiting without a deadline would make
        /// every close hostage to the one client that is not reading, which is the exact stall the bound
        /// above exists to end.
        /// </summary>
        internal static readonly TimeSpan CloseDrainWindow = TimeSpan.FromSeconds(2);

        readonly Channel<string> _outbound;
        readonly Task _writer;
        readonly TaskCompletionSource _closed = new(TaskCreationOptions.RunContinuationsAsynchronously);

        long _lastInboundTicks;
        int _queueDepth;
        int _maxQueueDepth;
        int _closing;

        // Written once by the socket task and read by the heartbeat, so the guid is published behind a
        // volatile flag rather than through a Guid? field: a nullable struct is two fields, and a reader
        // that saw the flag before the value would read a torn match id.
        Guid _matchId;
        volatile bool _authenticated;

        public V2Connection(
            string id, WebSocket webSocket, string remoteIp, int outboundQueueCapacity, DateTimeOffset now)
        {
            Id = id;
            WebSocket = webSocket;
            RemoteIp = remoteIp;
            _lastInboundTicks = now.UtcTicks;
            _outbound = Channel.CreateBounded<string>(new BoundedChannelOptions(outboundQueueCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait,
                SingleReader = true,
                SingleWriter = false,
            });
            _writer = Task.Run(PumpAsync);
        }

        public string Id { get; }

        public WebSocket WebSocket { get; }

        /// <summary>The peer address, after forwarded headers, or <c>unknown</c> under a test host.</summary>
        public string RemoteIp { get; }

        /// <summary>When this socket last said anything at all. Any inbound frame counts, PONG included:
        /// the question this answers is whether the client is still there, not what it had to say.</summary>
        public DateTimeOffset LastInbound
        {
            get => new(Interlocked.Read(ref _lastInboundTicks), TimeSpan.Zero);
            set => Interlocked.Exchange(ref _lastInboundTicks, value.UtcTicks);
        }

        /// <summary>The match this socket authenticated into, or null while it is still unauthenticated.</summary>
        public Guid? MatchId
        {
            get => _authenticated ? _matchId : null;
            set
            {
                if (value is null)
                {
                    _authenticated = false;
                    return;
                }

                _matchId = value.Value;
                _authenticated = true;
            }
        }

        public bool IsAuthenticated => _authenticated;

        /// <summary>True once a close has been asked for, whether or not it has finished.</summary>
        public bool IsClosed => Volatile.Read(ref _closing) != 0;

        /// <summary>Completes when the close has finished. Lets a caller that asked for a close - or a test
        /// that provoked one - wait for it without polling the socket state.</summary>
        public Task Closed => _closed.Task;

        /// <summary>Frames queued and not yet written.</summary>
        public int QueueDepth => Volatile.Read(ref _queueDepth);

        /// <summary>The deepest this connection's queue has ever been. Metrics only.</summary>
        public int MaxQueueDepth => Volatile.Read(ref _maxQueueDepth);

        /// <summary>
        /// Queues one frame. Synchronous and non-blocking by contract: the coordinator calls this while
        /// holding the per-match gate, so anything that waited here would stall the other seat.
        ///
        /// False means the frame was not queued - the connection is already closing, or its queue is full.
        /// A full queue also closes the connection: there is no way to deliver this frame and no honest way
        /// to skip it, so the socket ends and the client resyncs.
        /// </summary>
        public bool TryEnqueue(string message)
        {
            if (IsClosed) return false;

            if (!_outbound.Writer.TryWrite(message))
            {
                _ = CloseAsync(SlowClientCloseStatus, SlowClientReason);
                return false;
            }

            RecordDepth(Interlocked.Increment(ref _queueDepth));
            return true;
        }

        /// <summary>
        /// Closes the socket once. Every later call - and there will be several, because the receive loop,
        /// the heartbeat and the sink can all decide to close the same connection - waits for the first
        /// one rather than racing it onto a socket that allows a single outstanding send.
        /// </summary>
        public async Task CloseAsync(int status, string reason)
        {
            if (Interlocked.Exchange(ref _closing, 1) != 0)
            {
                await _closed.Task.ConfigureAwait(false);
                return;
            }

            try
            {
                _outbound.Writer.TryComplete();

                using var drained = new CancellationTokenSource();
                Task deadline = Task.Delay(CloseDrainWindow, drained.Token);
                Task finished = await Task.WhenAny(_writer, deadline).ConfigureAwait(false);
                drained.Cancel();

                if (!ReferenceEquals(finished, _writer))
                {
                    // The writer is still inside a send to a client that is not reading it. There is no
                    // close handshake to be had with a peer in that state.
                    WebSocket.Abort();
                    return;
                }

                if (WebSocket.State is WebSocketState.Open or WebSocketState.CloseReceived)
                {
                    await WebSocket
                        .CloseOutputAsync((WebSocketCloseStatus)status, reason, CancellationToken.None)
                        .ConfigureAwait(false);
                }
            }
            catch
            {
                // A socket that will not close politely is closed rudely. Nothing above this can act on
                // the difference, and the receive loop is already unwinding.
                try { WebSocket.Abort(); } catch { }
            }
            finally
            {
                _closed.TrySetResult();
            }
        }

        async Task PumpAsync()
        {
            try
            {
                await foreach (string message in _outbound.Reader.ReadAllAsync().ConfigureAwait(false))
                {
                    Interlocked.Decrement(ref _queueDepth);

                    try
                    {
                        await WebSocket.SendAsync(
                                Encoding.UTF8.GetBytes(message),
                                WebSocketMessageType.Text,
                                endOfMessage: true,
                                CancellationToken.None)
                            .ConfigureAwait(false);
                    }
                    catch
                    {
                        // The socket is gone. The receive loop is learning the same thing from the other
                        // side; a background task that threw here would only lose the frame louder.
                        break;
                    }
                }
            }
            catch
            {
            }
        }

        void RecordDepth(int depth)
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
