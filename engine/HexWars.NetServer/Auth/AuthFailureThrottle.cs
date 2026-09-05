using System.Collections.Concurrent;

namespace HexWars.NetServer.Auth
{
    /// <summary>
    /// Counts failed ticket authentications per caller and stops answering one that keeps failing.
    ///
    /// This is not the request rate limiter and does not replace it. The rate limiter bounds how often
    /// anyone may ask; this bounds how often anyone may be WRONG, which is the shape of ticket guessing
    /// and of a client stuck in a retry loop on a ticket Steam will never accept. Each failure costs this
    /// server a round trip to Valve, so the cheap answer has to arrive before that call, not after it.
    ///
    /// A fixed window rather than a decaying score: a player who genuinely cannot sign in should get a
    /// bounded lockout that ends at a predictable moment, not one that lengthens every time their client
    /// retries. Failures are counted, successes are not: a caller doing legitimate work never approaches
    /// the limit, so there is nothing to reset.
    /// </summary>
    public sealed class AuthFailureThrottle(TimeProvider time)
    {
        /// <summary>Failures inside one window before the caller is refused for the rest of it.</summary>
        public const int MaxFailures = 10;

        /// <summary>How long a window lasts, and therefore the longest a caller can be locked out.</summary>
        public static readonly TimeSpan Window = TimeSpan.FromMinutes(5);

        /// <summary>
        /// The most callers tracked at once. The map is keyed by an address the caller effectively chooses
        /// - an IPv6 client has a whole /64 to walk through - so without a ceiling a stream of failures
        /// from fresh addresses is an unbounded allocation this process can be made to perform.
        /// </summary>
        public const int MaxTrackedCallers = 10_000;

        /// <summary>How often the map is walked looking for elapsed windows. Sweeping on every failure
        /// makes each one cost the size of the map, which is the wrong shape when the map is largest.
        /// </summary>
        public static readonly TimeSpan SweepInterval = TimeSpan.FromMinutes(1);

        /// <summary>Evicted in one pass when the map is full and sweeping freed nothing. A batch rather
        /// than one entry, so the scan is paid once per batch rather than once per insertion.</summary>
        const int EvictionBatch = MaxTrackedCallers / 10;

        readonly ConcurrentDictionary<string, Counter> _counters = new(StringComparer.Ordinal);

        readonly object _maintenanceGate = new();

        DateTimeOffset _lastSweep;
        bool _swept;

        /// <summary>Callers currently tracked. Never above <see cref="MaxTrackedCallers"/>.</summary>
        internal int TrackedCallers => _counters.Count;

        /// <summary>How many times the periodic sweep has actually run, for asserting it is time-gated.</summary>
        internal int SweepCount { get; private set; }

        /// <summary>The failures recorded for one caller, and when their current window opened.</summary>
        sealed class Counter
        {
            public DateTimeOffset StartedAt;
            public int Failures;
        }

        /// <summary>Records one failed authentication for <paramref name="key"/>.</summary>
        public void RecordFailure(string key)
        {
            DateTimeOffset now = time.GetUtcNow();

            // Before the insert, so the ceiling is a ceiling rather than a high-water mark.
            Maintain(now, admittingNewCaller: !_counters.ContainsKey(key));

            Counter counter = _counters.GetOrAdd(key, _ => new Counter { StartedAt = now });

            lock (counter)
            {
                if (now - counter.StartedAt >= Window)
                {
                    counter.StartedAt = now;
                    counter.Failures = 0;
                }

                counter.Failures++;
            }
        }

        /// <summary>True while <paramref name="key"/> has spent its failures for the current window.</summary>
        public bool IsThrottled(string key)
        {
            if (!_counters.TryGetValue(key, out Counter? counter)) return false;

            DateTimeOffset now = time.GetUtcNow();
            lock (counter)
            {
                // An elapsed window is dropped rather than reset, so a caller who stopped failing stops
                // costing memory. A racing RecordFailure simply re-adds it.
                if (now - counter.StartedAt >= Window)
                {
                    _counters.TryRemove(new KeyValuePair<string, Counter>(key, counter));
                    return false;
                }

                return counter.Failures >= MaxFailures;
            }
        }

        /// <summary>
        /// Keeps the map small enough to be cheap and bounded enough to be safe.
        ///
        /// Two separate jobs. The periodic sweep drops elapsed windows and runs at most once a
        /// <see cref="SweepInterval"/>, because sweeping on every failure makes each failure cost the size
        /// of the map - worst exactly when the map is biggest, which is when this server is under the load
        /// the throttle exists for. The eviction is the hard ceiling and runs whenever admitting a caller
        /// would exceed it, sweeping first and dropping the oldest windows only if that freed nothing.
        ///
        /// Opportunistic rather than on a timer: a background timer would keep this object alive and have
        /// to be disposed, and the only call that grows the map is the one that can also shrink it.
        /// </summary>
        void Maintain(DateTimeOffset now, bool admittingNewCaller)
        {
            bool full = admittingNewCaller && _counters.Count >= MaxTrackedCallers;
            bool due = !_swept || now - _lastSweep >= SweepInterval;
            if (!full && !due) return;

            lock (_maintenanceGate)
            {
                // Re-checked under the lock: several threads can arrive here together, and the sweep only
                // needs to happen once for all of them.
                full = admittingNewCaller && _counters.Count >= MaxTrackedCallers;
                due = !_swept || now - _lastSweep >= SweepInterval;
                if (!full && !due) return;

                if (!_swept)
                {
                    // The first call establishes when the clock started rather than sweeping an empty map.
                    _lastSweep = now;
                    _swept = true;
                    if (!full) return;
                }

                if (due)
                {
                    _lastSweep = now;
                    SweepCount++;
                }

                DropElapsed(now);

                if (!admittingNewCaller || _counters.Count < MaxTrackedCallers) return;

                EvictOldest();
            }
        }

        void DropElapsed(DateTimeOffset now)
        {
            foreach (KeyValuePair<string, Counter> entry in _counters)
            {
                lock (entry.Value)
                {
                    if (now - entry.Value.StartedAt >= Window) _counters.TryRemove(entry);
                }
            }
        }

        /// <summary>Drops the batch whose windows opened longest ago. Oldest first because those are the
        /// callers closest to ageing out anyway, so the least information is lost.</summary>
        void EvictOldest()
        {
            KeyValuePair<string, Counter>[] oldest = _counters
                .OrderBy(entry => entry.Value.StartedAt)
                .Take(EvictionBatch)
                .ToArray();

            foreach (KeyValuePair<string, Counter> entry in oldest) _counters.TryRemove(entry);
        }
    }
}
