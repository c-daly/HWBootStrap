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

        /// <summary>Failures recorded against the live entry for this caller, or null when none is tracked.
        /// The public surface answers only "throttled or not", which cannot tell a count of zero from a
        /// count that was written somewhere the dictionary is no longer looking.</summary>
        internal int? FailuresFor(string key)
        {
            if (!_counters.TryGetValue(key, out Counter? counter)) return null;
            lock (counter) return counter.Failures;
        }

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

            // Maintenance before the counter is fetched, never after. The sweep drops entries whose
            // window has elapsed, which is precisely the entry a failure at the window boundary is about
            // to increment: fetching first meant the count landed on an object the dictionary had already
            // let go, and the caller got one free bad ticket at the start of every window.
            Maintain(now);

            while (true)
            {
                // A caller already tracked cannot grow the map, so it never has to wait on the gate. Only
                // an admission does, because checking for room and inserting are two halves of one
                // decision and a ceiling enforced across a gap is not a ceiling.
                Counter counter = _counters.TryGetValue(key, out Counter? tracked) ? tracked : Admit(key, now);

                lock (counter)
                {
                    if (now - counter.StartedAt >= Window)
                    {
                        counter.StartedAt = now;
                        counter.Failures = 0;
                    }

                    counter.Failures++;
                }

                // Maintenance on another thread can still have swept this entry between the fetch and the
                // increment. That is rare rather than impossible, and a lost failure is the one outcome
                // this class exists to prevent, so the write is confirmed against the live map and redone
                // if it landed on a detached counter.
                if (_counters.TryGetValue(key, out Counter? live) && ReferenceEquals(live, counter)) return;
            }
        }

        /// <summary>True while <paramref name="key"/> has spent its failures for the current window.</summary>
        public bool IsThrottled(string key)
        {
            if (!_counters.TryGetValue(key, out Counter? counter)) return false;

            DateTimeOffset now = time.GetUtcNow();
            lock (counter)
            {
                // An elapsed window is simply not a lockout. This deliberately does NOT drop the entry:
                // removing it here would take it out from under a RecordFailure that had already fetched
                // it, and that failure would then be counted on a detached object and lost. Reclaiming the
                // memory is a job for the sweep, which runs before anything fetches a counter.
                if (now - counter.StartedAt >= Window) return false;

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
        /// <summary>
        /// Admits a caller the map has not seen, making room first if there is none. The gate spans the
        /// check and the insert together, which is the whole point: they are the two halves of one
        /// decision, and a ceiling enforced across a gap is not a ceiling.
        /// </summary>
        Counter Admit(string key, DateTimeOffset now)
        {
            lock (_maintenanceGate)
            {
                // Another thread may have admitted this same caller while this one waited, in which case
                // the map did not grow and there is nothing to make room for.
                if (_counters.TryGetValue(key, out Counter? admitted)) return admitted;

                MaintainLocked(now, admittingNewCaller: true);

                var counter = new Counter { StartedAt = now };
                _counters[key] = counter;
                return counter;
            }
        }

        /// <summary>Runs the periodic sweep if it is due. Never makes room: only an admission can grow
        /// the map, and that path makes its own.</summary>
        void Maintain(DateTimeOffset now)
        {
            if (_swept && now - _lastSweep < SweepInterval) return;

            lock (_maintenanceGate)
            {
                MaintainLocked(now, admittingNewCaller: false);
            }
        }

        void MaintainLocked(DateTimeOffset now, bool admittingNewCaller)
        {
            // Re-checked under the gate: several threads can arrive together, and the sweep only needs to
            // happen once for all of them.
            bool full = admittingNewCaller && _counters.Count >= MaxTrackedCallers;
            bool due = !_swept || now - _lastSweep >= SweepInterval;
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
