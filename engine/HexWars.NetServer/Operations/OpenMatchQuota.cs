using System.Globalization;
using System.Net;
using System.Net.Sockets;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// How many matches one address may actually allocate inside a rolling window.
    ///
    /// This is not the request rate limiter and does not replace it. The limiter bounds how OFTEN a caller
    /// may ask, over a one-minute window, which stops a burst; it says nothing about how much durable state
    /// a patient caller leaves behind. A match row is cheap to create and lives for the whole retention
    /// window, so five creations a minute sustained is a database filled by one client that never once broke
    /// the rate limit. This counts the creations that SUCCEEDED, which is the thing that costs storage.
    ///
    /// Admission is a RESERVATION, not a question. Asking whether there is room and then spending it later
    /// is check-then-act: the room is granted between the two, so a dozen concurrent creations from one
    /// address all read the same low number and all proceed. A caller takes a lease under the lock before
    /// anything is awaited, and the lease is the seat - it is what other requests cannot have.
    ///
    /// The window is genuinely rolling. Each key keeps the timestamps of its last few creations and the
    /// count is taken over the trailing window, so a fixed window cannot be walked by waiting for a boundary
    /// and then spending the whole budget again a second later.
    ///
    /// Counted in memory and lost on restart, deliberately: no IP address is written to the database, and
    /// the retention decision says so in as many words.
    /// </summary>
    public sealed class OpenMatchQuota(
        IOptions<MatchHostingOptions> options, TimeProvider time, ILogger<OpenMatchQuota> logger)
    {
        /// <summary>
        /// The most callers tracked at once.
        ///
        /// The map is keyed by an address the caller effectively chooses, so it needs a ceiling. What it does
        /// NOT do at the ceiling is evict: dropping a live window hands that caller its whole budget back,
        /// which turns memory pressure into a way of buying quota. A saturated map refuses callers it has
        /// never seen and keeps serving the ones it is already tracking.
        /// </summary>
        public const int MaxTrackedCallers = 10_000;

        /// <summary>How often saturation is allowed to say so. It is a condition, not an event, and a line
        /// per refused request would be the loudest thing in the log exactly when it is least useful.</summary>
        static readonly TimeSpan SaturationLogInterval = TimeSpan.FromMinutes(1);

        /// <summary>IPv6 is handed out by the /64, not by the address. A client with a routed prefix can
        /// otherwise walk through 18 quintillion addresses, one per match, and never meet this cap.</summary>
        public const int IPv6PrefixBits = 64;

        readonly Dictionary<string, Bucket> _buckets = new(StringComparer.Ordinal);
        readonly object _gate = new();

        DateTimeOffset _lastSaturationLog;
        bool _hasLoggedSaturation;

        /// <summary>Callers currently tracked. Never above <see cref="MaxTrackedCallers"/>.</summary>
        internal int TrackedCallers
        {
            get { lock (_gate) return _buckets.Count; }
        }

        /// <summary>Creations this key has made inside the trailing window, for tests and for a log line that
        /// wants to say how full a caller is without saying who they are.</summary>
        internal int CreationsWithinWindow(string key)
        {
            lock (_gate)
            {
                if (!_buckets.TryGetValue(key, out Bucket? bucket)) return 0;

                Prune(bucket, time.GetUtcNow());
                return bucket.Created.Count;
            }
        }

        /// <summary>
        /// The bucket an address falls in.
        ///
        /// Two normalisations, both of which close a way around the cap rather than being tidiness. An IPv4
        /// address that arrived over IPv6 is written ::ffff:a.b.c.d and would otherwise be a second, free
        /// bucket for the same client. And an IPv6 client is not one address: a routed /64 is the smallest
        /// unit an ISP hands out, so that is the unit a per-address cap has to count.
        /// </summary>
        public static string BucketFor(IPAddress? address)
        {
            if (address is null) return SteamMatchEndpointsCallerUnknown;

            if (address.IsIPv4MappedToIPv6) address = address.MapToIPv4();

            if (address.AddressFamily != AddressFamily.InterNetworkV6)
                return address.ToString();

            byte[] bytes = address.GetAddressBytes();
            for (var i = IPv6PrefixBits / 8; i < bytes.Length; i++) bytes[i] = 0;

            return new IPAddress(bytes).ToString()
                + "/" + IPv6PrefixBits.ToString(CultureInfo.InvariantCulture);
        }

        /// <summary>The key a connection with no address falls in. Named rather than empty so it reads in a
        /// log, and spelled here so this class does not depend on the endpoints it serves.</summary>
        internal const string SteamMatchEndpointsCallerUnknown = "unknown";

        /// <summary>
        /// Takes one seat of this caller budget, or refuses.
        ///
        /// The whole decision happens under the lock, so N callers racing get N different answers and exactly
        /// the cap many leases exist at once. A reservation counts against the cap while it is outstanding,
        /// which is what stops concurrent requests from all being told there is room for one more.
        /// </summary>
        /// <returns>True when a seat was taken. The lease must be disposed either way; a lease that was
        /// refused is inert.</returns>
        public bool TryReserve(string key, out QuotaLease lease)
        {
            ArgumentNullException.ThrowIfNull(key);

            DateTimeOffset now = time.GetUtcNow();
            int cap = options.Value.MaxOpenMatchesPerIp;

            lock (_gate)
            {
                if (!_buckets.TryGetValue(key, out Bucket? bucket))
                {
                    SweepLocked(now);

                    if (_buckets.Count >= MaxTrackedCallers)
                    {
                        LogSaturationLocked(now);
                        lease = QuotaLease.Refused(this, key);
                        return false;
                    }

                    bucket = new Bucket();
                    _buckets[key] = bucket;
                }

                Prune(bucket, now);

                if (bucket.Created.Count + bucket.Outstanding >= cap)
                {
                    lease = QuotaLease.Refused(this, key);
                    return false;
                }

                bucket.Outstanding++;
                lease = QuotaLease.Reserved(this, key);
                return true;
            }
        }

        /// <summary>Charges the caller for a match that now exists, and hands back the seat.</summary>
        internal void Commit(string key, bool reserved)
        {
            DateTimeOffset now = time.GetUtcNow();
            int cap = options.Value.MaxOpenMatchesPerIp;

            lock (_gate)
            {
                if (!_buckets.TryGetValue(key, out Bucket? bucket))
                {
                    // A match was created for a caller this map is not tracking, which happens only when the
                    // map was saturated when the request began. Admit it if there is room now; drop the
                    // charge if there still is not, because a bucket that cannot be created cannot be counted.
                    if (_buckets.Count >= MaxTrackedCallers) return;

                    bucket = new Bucket();
                    _buckets[key] = bucket;
                }

                if (reserved && bucket.Outstanding > 0) bucket.Outstanding--;

                Prune(bucket, now);

                // At most cap timestamps are kept: the question this class answers is whether there are cap
                // creations inside the window, and the older ones cannot change that answer.
                while (bucket.Created.Count >= cap && bucket.Created.Count > 0) bucket.Created.Dequeue();

                bucket.Created.Enqueue(now);
                bucket.Newest = now;
            }
        }

        /// <summary>Hands back a seat that did not become a match. Nothing is charged.</summary>
        internal void Release(string key, bool reserved)
        {
            if (!reserved) return;

            lock (_gate)
            {
                if (_buckets.TryGetValue(key, out Bucket? bucket) && bucket.Outstanding > 0)
                    bucket.Outstanding--;
            }
        }

        /// <summary>Drops keys whose newest creation has aged out and which hold no outstanding lease. A
        /// bucket with a live lease is never dropped: its seat would go with it.</summary>
        void SweepLocked(DateTimeOffset now)
        {
            DateTimeOffset cutoff = now - MatchHostingOptions.OpenMatchWindow;
            List<string>? stale = null;

            foreach (KeyValuePair<string, Bucket> entry in _buckets)
            {
                if (entry.Value.Outstanding > 0) continue;
                if (entry.Value.Created.Count > 0 && entry.Value.Newest > cutoff) continue;

                (stale ??= new List<string>()).Add(entry.Key);
            }

            if (stale is null) return;

            foreach (string key in stale) _buckets.Remove(key);
        }

        void LogSaturationLocked(DateTimeOffset now)
        {
            if (_hasLoggedSaturation && now - _lastSaturationLog < SaturationLogInterval) return;

            _lastSaturationLog = now;
            _hasLoggedSaturation = true;

            logger.LogWarning(
                "The open-match quota is tracking its ceiling of {Callers} callers; addresses it has not seen "
                + "are being refused until a window ages out",
                MaxTrackedCallers);
        }

        static void Prune(Bucket bucket, DateTimeOffset now)
        {
            DateTimeOffset cutoff = now - MatchHostingOptions.OpenMatchWindow;

            while (bucket.Created.Count > 0 && bucket.Created.Peek() <= cutoff) bucket.Created.Dequeue();
        }

        sealed class Bucket
        {
            /// <summary>When this caller last created a match, oldest first, never more than the cap.</summary>
            public readonly Queue<DateTimeOffset> Created = new();

            /// <summary>Leases taken and not yet committed or released.</summary>
            public int Outstanding;

            /// <summary>The newest creation, kept separately so the sweep can judge a bucket without
            /// pruning it.</summary>
            public DateTimeOffset Newest;
        }
    }

    /// <summary>
    /// One seat of an address budget, held for the length of a request.
    ///
    /// Disposal COMMITS a lease that was neither committed nor released, which is the fail-closed half of
    /// this design. A request that fell out of the handler in a way nobody anticipated may or may not have
    /// created a match, and the two mistakes are not equal: charging for a match that does not exist costs
    /// one caller one slot for ten minutes, while forgetting a match that does exist is a cap that quietly
    /// stops counting.
    /// </summary>
    public sealed class QuotaLease : IDisposable
    {
        readonly string _key;
        readonly bool _reserved;

        OpenMatchQuota? _quota;

        QuotaLease(OpenMatchQuota? quota, string key, bool reserved)
        {
            _quota = quota;
            _key = key;
            _reserved = reserved;
        }

        internal static QuotaLease Reserved(OpenMatchQuota quota, string key) => new(quota, key, true);

        /// <summary>A lease for a caller that was refused a seat. It holds nothing, but it can still be
        /// committed: a request that went on to create a match anyway has to be charged for it.</summary>
        internal static QuotaLease Refused(OpenMatchQuota quota, string key) => new(quota, key, false);

        /// <summary>A lease that belongs to no quota, for a path that never took one.</summary>
        public static QuotaLease None { get; } = new(null, string.Empty, false);

        /// <summary>True while this lease has been neither committed nor released.</summary>
        internal bool IsOutstanding => _quota is not null;

        /// <summary>Charges this caller for a match that now exists.</summary>
        public void Commit()
        {
            OpenMatchQuota? quota = _quota;
            _quota = null;
            quota?.Commit(_key, _reserved);
        }

        /// <summary>Hands the seat back. The request created nothing, so nothing is charged.</summary>
        public void Release()
        {
            OpenMatchQuota? quota = _quota;
            _quota = null;
            quota?.Release(_key, _reserved);
        }

        /// <summary>Commits a lease nobody settled. See the type remarks: fail closed.</summary>
        public void Dispose() => Commit();
    }
}
