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

        /// <summary>Entries above this many trigger an opportunistic sweep. High enough that ordinary
        /// traffic never pays for one, low enough that a scan of the map stays cheap.</summary>
        const int SweepThreshold = 1024;

        readonly ConcurrentDictionary<string, Counter> _counters = new(StringComparer.Ordinal);

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

            Sweep(now);
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
        /// Drops elapsed windows once the map has grown enough to be worth walking. Opportunistic rather
        /// than timed on purpose: a background timer would keep this object alive and would have to be
        /// disposed, and the only call that grows the map is the one that also prunes it.
        /// </summary>
        void Sweep(DateTimeOffset now)
        {
            if (_counters.Count <= SweepThreshold) return;

            foreach (KeyValuePair<string, Counter> entry in _counters)
            {
                lock (entry.Value)
                {
                    if (now - entry.Value.StartedAt >= Window)
                    {
                        _counters.TryRemove(entry);
                    }
                }
            }
        }
    }
}
