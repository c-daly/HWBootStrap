using System.Collections.Concurrent;
using System.Diagnostics.Metrics;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// Everything the meter has been told, as one object.
    ///
    /// Counters are totals since this process started, not rates: a match host is restarted often enough
    /// that a rate computed here would be computed over the wrong window. Whatever scrapes this subtracts
    /// two readings, which is the only place the window is known.
    /// </summary>
    public sealed record MatchMetricsSnapshot(
        long MatchesCreated,
        long CommandsCommitted,
        long CommandsRejected,
        IReadOnlyDictionary<string, long> RejectionsByReason,
        long DatabaseFailures,
        long SteamFailures,
        IReadOnlyDictionary<string, long> SteamFailuresByKind,
        long AuthFailures,
        long Reconnects,
        long RecoveryFailures,
        int LiveMatches,
        int OpenSockets,
        int MaxOutboundQueueDepth,
        long CommitCount,
        double CommitMillisTotal,
        double CommitMillisMax,
        long BroadcastCount,
        double BroadcastMillisTotal,
        double BroadcastMillisMax);

    /// <summary>
    /// The instruments this server publishes, and an in-process reading of them.
    ///
    /// Two audiences, one set of measurements. A deployment with a collector attaches to the meter by name
    /// and gets everything; a deployment with nothing but a log gets the same numbers through Snapshot,
    /// aggregated here by a MeterListener of our own. Keeping one source is the point - a snapshot computed
    /// from private fields alongside the instruments would be two counters that disagree the first time
    /// somebody instruments a new path and updates only one of them.
    ///
    /// Registered on every deployment, including the legacy one that has no coordinator at all. The gauges
    /// read through delegates that are wired only when there is something to read, so a host with no
    /// durable runtime reports zero rather than refusing to start.
    /// </summary>
    public sealed class MatchMetrics : IDisposable
    {
        public const string MeterName = "HexWars.MatchServer";

        public const string MatchesCreatedName = "hexwars.matches.created";
        public const string CommandsCommittedName = "hexwars.commands.committed";
        public const string CommandsRejectedName = "hexwars.commands.rejected";
        public const string DatabaseFailuresName = "hexwars.db.failures";
        public const string SteamFailuresName = "hexwars.steam.failures";
        public const string AuthFailuresName = "hexwars.auth.failures";
        public const string ReconnectsName = "hexwars.reconnects";
        public const string RecoveryFailuresName = "hexwars.recovery.failures";

        public const string LiveMatchesName = "hexwars.matches.live";
        public const string OpenSocketsName = "hexwars.sockets.open";
        public const string OutboundQueueMaxName = "hexwars.outbound.queue.max";

        public const string CommitMillisName = "hexwars.command.commit.ms";
        public const string BroadcastMillisName = "hexwars.command.broadcast.ms";

        /// <summary>Why a command was refused. The engine reason names, plus the protocol level ones.</summary>
        public const string ReasonTag = "reason";

        /// <summary>Which Steam refusal it was.</summary>
        public const string FailureTag = "failure";

        readonly Meter _meter = new(MeterName);
        readonly MeterListener _listener = new();

        readonly ConcurrentDictionary<string, long> _counters = new(StringComparer.Ordinal);
        readonly ConcurrentDictionary<string, ConcurrentDictionary<string, long>> _tagged =
            new(StringComparer.Ordinal);
        readonly ConcurrentDictionary<string, Distribution> _distributions = new(StringComparer.Ordinal);
        readonly ConcurrentDictionary<string, int> _gauges = new(StringComparer.Ordinal);

        readonly Counter<long> _matchesCreated;
        readonly Counter<long> _commandsCommitted;
        readonly Counter<long> _commandsRejected;
        readonly Counter<long> _databaseFailures;
        readonly Counter<long> _steamFailures;
        readonly Counter<long> _authFailures;
        readonly Counter<long> _reconnects;
        readonly Counter<long> _recoveryFailures;
        readonly Histogram<double> _commitMillis;
        readonly Histogram<double> _broadcastMillis;

        public MatchMetrics()
        {
            _matchesCreated = _meter.CreateCounter<long>(MatchesCreatedName);
            _commandsCommitted = _meter.CreateCounter<long>(CommandsCommittedName);
            _commandsRejected = _meter.CreateCounter<long>(CommandsRejectedName);
            _databaseFailures = _meter.CreateCounter<long>(DatabaseFailuresName);
            _steamFailures = _meter.CreateCounter<long>(SteamFailuresName);
            _authFailures = _meter.CreateCounter<long>(AuthFailuresName);
            _reconnects = _meter.CreateCounter<long>(ReconnectsName);
            _recoveryFailures = _meter.CreateCounter<long>(RecoveryFailuresName);

            _commitMillis = _meter.CreateHistogram<double>(CommitMillisName, "ms");
            _broadcastMillis = _meter.CreateHistogram<double>(BroadcastMillisName, "ms");

            _meter.CreateObservableGauge(LiveMatchesName, () => LiveMatches?.Invoke() ?? 0);
            _meter.CreateObservableGauge(OpenSocketsName, () => OpenSockets?.Invoke() ?? 0);
            _meter.CreateObservableGauge(OutboundQueueMaxName, () => MaxOutboundQueueDepth?.Invoke() ?? 0);

            // Only this meter. A process that hosts two of these - which every test run does - must not see
            // one snapshot answer with another host measurements.
            _listener.InstrumentPublished = (instrument, listener) =>
            {
                if (ReferenceEquals(instrument.Meter, _meter)) listener.EnableMeasurementEvents(instrument);
            };

            _listener.SetMeasurementEventCallback<long>(
                (instrument, measurement, tags, _) => Record(instrument, measurement, tags));
            _listener.SetMeasurementEventCallback<int>(
                (instrument, measurement, tags, _) => Record(instrument, measurement, tags));
            _listener.SetMeasurementEventCallback<double>(
                (instrument, measurement, tags, _) => Record(instrument, measurement, tags));

            _listener.Start();
        }

        /// <summary>Matches this process is holding. Wired once the durable runtime exists; null before.</summary>
        public Func<int>? LiveMatches { get; set; }

        /// <summary>Live v2 sockets.</summary>
        public Func<int>? OpenSockets { get; set; }

        /// <summary>The deepest any outbound queue has been, which is what a slow client looks like.</summary>
        public Func<int>? MaxOutboundQueueDepth { get; set; }

        // ---- what the server records -----------------------------------------

        public void MatchCreated() => _matchesCreated.Add(1);

        public void CommandCommitted() => _commandsCommitted.Add(1);

        public void CommandRejected(string reason) =>
            _commandsRejected.Add(1, new KeyValuePair<string, object?>(ReasonTag, reason));

        public void DatabaseFailure() => _databaseFailures.Add(1);

        public void SteamFailure(string failure) =>
            _steamFailures.Add(1, new KeyValuePair<string, object?>(FailureTag, failure));

        public void AuthFailure() => _authFailures.Add(1);

        /// <summary>A seat that had a socket recently taking one again.</summary>
        public void Reconnect() => _reconnects.Add(1);

        /// <summary>Matches the recovery pass refused, or one for a pass that could not run at all.</summary>
        public void RecoveryFailure(long matches = 1)
        {
            if (matches > 0) _recoveryFailures.Add(matches);
        }

        /// <summary>How long the durable append took.</summary>
        public void RecordCommit(TimeSpan elapsed) => _commitMillis.Record(elapsed.TotalMilliseconds);

        /// <summary>Accepted to the last seat having the frame queued.</summary>
        public void RecordBroadcast(TimeSpan elapsed) => _broadcastMillis.Record(elapsed.TotalMilliseconds);

        // ---- what an operator reads ------------------------------------------

        public MatchMetricsSnapshot Snapshot()
        {
            // The gauges have no value until something asks for one.
            _listener.RecordObservableInstruments();

            (long CommitCount, double CommitTotal, double CommitMax) commit = Read(CommitMillisName);
            (long BroadcastCount, double BroadcastTotal, double BroadcastMax) broadcast = Read(BroadcastMillisName);

            return new MatchMetricsSnapshot(
                Count(MatchesCreatedName),
                Count(CommandsCommittedName),
                Count(CommandsRejectedName),
                Tagged(CommandsRejectedName),
                Count(DatabaseFailuresName),
                Count(SteamFailuresName),
                Tagged(SteamFailuresName),
                Count(AuthFailuresName),
                Count(ReconnectsName),
                Count(RecoveryFailuresName),
                Gauge(LiveMatchesName),
                Gauge(OpenSocketsName),
                Gauge(OutboundQueueMaxName),
                commit.CommitCount,
                commit.CommitTotal,
                commit.CommitMax,
                broadcast.BroadcastCount,
                broadcast.BroadcastTotal,
                broadcast.BroadcastMax);
        }

        public void Dispose()
        {
            _listener.Dispose();
            _meter.Dispose();
        }

        // ---- internals --------------------------------------------------------

        void Record(Instrument instrument, double measurement, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            if (instrument is Histogram<double>)
            {
                _distributions.GetOrAdd(instrument.Name, _ => new Distribution()).Add(measurement);
                return;
            }

            if (instrument.IsObservable)
            {
                _gauges[instrument.Name] = (int)measurement;
                return;
            }

            var delta = (long)measurement;
            _counters.AddOrUpdate(instrument.Name, delta, (_, held) => held + delta);

            string? tagged = null;
            foreach (KeyValuePair<string, object?> tag in tags)
            {
                if (tag.Key == ReasonTag || tag.Key == FailureTag) tagged = tag.Value?.ToString();
            }

            if (tagged is null) return;

            _tagged.GetOrAdd(instrument.Name, _ => new ConcurrentDictionary<string, long>(StringComparer.Ordinal))
                .AddOrUpdate(tagged, delta, (_, held) => held + delta);
        }

        long Count(string name) => _counters.TryGetValue(name, out long held) ? held : 0;

        int Gauge(string name) => _gauges.TryGetValue(name, out int held) ? held : 0;

        IReadOnlyDictionary<string, long> Tagged(string name) =>
            _tagged.TryGetValue(name, out ConcurrentDictionary<string, long>? held)
                ? new Dictionary<string, long>(held, StringComparer.Ordinal)
                : new Dictionary<string, long>(StringComparer.Ordinal);

        (long Count, double Total, double Max) Read(string name) =>
            _distributions.TryGetValue(name, out Distribution? held) ? held.Read() : (0, 0, 0);

        /// <summary>Count, total and worst, which is what a histogram is worth without a collector: the
        /// mean says whether commits are slow and the maximum says whether one of them was.</summary>
        sealed class Distribution
        {
            readonly object _gate = new();

            long _count;
            double _total;
            double _max;

            public void Add(double value)
            {
                lock (_gate)
                {
                    _count++;
                    _total += value;
                    if (value > _max) _max = value;
                }
            }

            public (long Count, double Total, double Max) Read()
            {
                lock (_gate) return (_count, _total, _max);
            }
        }
    }
}
