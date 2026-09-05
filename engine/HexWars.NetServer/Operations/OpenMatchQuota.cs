using System.Collections.Concurrent;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// How many matches one address may actually allocate inside a window.
    ///
    /// This is not the request rate limiter and does not replace it. The limiter bounds how OFTEN a caller
    /// may ask, over a one-minute window, which stops a burst; it says nothing about how much durable state
    /// a patient caller leaves behind. A match row is cheap to create and lives for the whole retention
    /// window, so five creations a minute sustained is a database filled by one client that never once broke
    /// the rate limit. This counts the creations that SUCCEEDED, which is the thing that costs storage.
    ///
    /// Counted in memory and lost on restart, deliberately: no IP address is written to the database, and
    /// the retention decision says so in as many words.
    ///
    /// The map is bounded the same way the auth-failure throttle is bounded and for the same reason: it is
    /// keyed by an address the caller effectively chooses, so without a ceiling a stream of creations from
    /// fresh addresses is an unbounded allocation this process can be made to perform.
    /// </summary>
    public sealed class OpenMatchQuota(IOptions<MatchHostingOptions> options, TimeProvider time)
    {
        /// <summary>The most callers tracked at once.</summary>
        public const int MaxTrackedCallers = 10_000;

        /// <summary>Evicted in one pass when the map is full and dropping elapsed windows freed nothing.</summary>
        const int EvictionBatch = MaxTrackedCallers / 10;

        readonly ConcurrentDictionary<string, Window> _windows = new(StringComparer.Ordinal);
        readonly object _gate = new();

        /// <summary>Callers currently tracked. Never above <see cref="MaxTrackedCallers"/>.</summary>
        internal int TrackedCallers => _windows.Count;

        /// <summary>
        /// True while <paramref name="caller"/> may still allocate a match.
        ///
        /// Asked before the work rather than after it, so a caller over the quota costs this server no Steam
        /// round trip and no database write at all.
        /// </summary>
        public bool HasHeadroom(string caller)
        {
            if (!_windows.TryGetValue(caller, out Window? window)) return true;

            DateTimeOffset now = time.GetUtcNow();
            lock (window)
            {
                if (now - window.StartedAt >= MatchHostingOptions.OpenMatchWindow) return true;

                return window.Created < options.Value.MaxOpenMatchesPerIp;
            }
        }

        /// <summary>Records one match this caller actually got. Only successful allocations are counted: a
        /// refusal leaves nothing behind, so charging for one would turn a Steam outage into a lockout.</summary>
        public void RecordCreation(string caller)
        {
            DateTimeOffset now = time.GetUtcNow();

            while (true)
            {
                Window window = _windows.TryGetValue(caller, out Window? tracked) ? tracked : Admit(caller, now);

                lock (window)
                {
                    if (now - window.StartedAt >= MatchHostingOptions.OpenMatchWindow)
                    {
                        window.StartedAt = now;
                        window.Created = 0;
                    }

                    window.Created++;
                }

                // Maintenance on another thread can have dropped this entry between the fetch and the
                // increment. A lost creation is the one outcome this class exists to prevent, so the write is
                // confirmed against the live map and redone if it landed on a detached window.
                if (_windows.TryGetValue(caller, out Window? live) && ReferenceEquals(live, window)) return;
            }
        }

        /// <summary>Admits a caller the map has not seen, making room first if there is none. The gate spans
        /// the check and the insert together: a ceiling enforced across a gap is not a ceiling.</summary>
        Window Admit(string caller, DateTimeOffset now)
        {
            lock (_gate)
            {
                if (_windows.TryGetValue(caller, out Window? admitted)) return admitted;

                if (_windows.Count >= MaxTrackedCallers)
                {
                    DropElapsed(now);
                    if (_windows.Count >= MaxTrackedCallers) EvictOldest();
                }

                var window = new Window { StartedAt = now };
                _windows[caller] = window;
                return window;
            }
        }

        void DropElapsed(DateTimeOffset now)
        {
            foreach (KeyValuePair<string, Window> entry in _windows)
            {
                lock (entry.Value)
                {
                    if (now - entry.Value.StartedAt >= MatchHostingOptions.OpenMatchWindow)
                        _windows.TryRemove(entry);
                }
            }
        }

        /// <summary>Drops the batch whose windows opened longest ago: those are the callers closest to ageing
        /// out anyway, so the least is lost.</summary>
        void EvictOldest()
        {
            KeyValuePair<string, Window>[] oldest = _windows
                .OrderBy(entry => entry.Value.StartedAt)
                .Take(EvictionBatch)
                .ToArray();

            foreach (KeyValuePair<string, Window> entry in oldest) _windows.TryRemove(entry);
        }

        sealed class Window
        {
            public DateTimeOffset StartedAt;
            public int Created;
        }
    }
}
