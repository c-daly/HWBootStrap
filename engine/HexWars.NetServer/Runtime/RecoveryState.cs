namespace HexWars.NetServer.Runtime
{
    /// <summary>
    /// What the startup recovery pass found, for anything that needs to know - which today means readiness.
    ///
    /// A singleton holding three fields rather than a method on the hosted service, because the two have
    /// different lifetimes as far as the rest of the process is concerned: the hosted service runs once and
    /// is of no further interest, while the answer it produced is consulted on every readiness probe for as
    /// long as this host lives.
    ///
    /// <see cref="Completed"/> is deliberately separate from both of the others. A pass that has not run yet
    /// and a pass that ran and found nothing wrong are the same shape and opposite answers, and a readiness
    /// endpoint that could not tell them apart would report ready before it had checked anything.
    /// </summary>
    public sealed class RecoveryState
    {
        readonly object _gate = new();

        bool _completed;
        RecoveryReport? _report;
        Exception? _error;

        /// <summary>Whether the pass has finished, however it finished.</summary>
        public bool Completed
        {
            get { lock (_gate) return _completed; }
        }

        /// <summary>What the pass found, or null when it could not run.</summary>
        public RecoveryReport? Report
        {
            get { lock (_gate) return _report; }
        }

        /// <summary>What stopped the pass, or null when it ran. A store that will not answer, in practice.</summary>
        public Exception? Error
        {
            get { lock (_gate) return _error; }
        }

        public void RecordReport(RecoveryReport report)
        {
            ArgumentNullException.ThrowIfNull(report);

            lock (_gate)
            {
                _report = report;
                _error = null;
                _completed = true;
            }
        }

        /// <summary>
        /// Records an attempt that could not run, leaving the pass unfinished.
        ///
        /// <see cref="Completed"/> stays false because a failure that will be retried is not a verdict: the
        /// host does not yet know what it is hosting, and saying it had finished would let readiness report
        /// an answer nobody has reached. <see cref="Error"/> carries the most recent reason so an operator
        /// can see what it is still failing on.
        /// </summary>
        public void RecordFailure(Exception error)
        {
            ArgumentNullException.ThrowIfNull(error);

            lock (_gate)
            {
                _report = null;
                _error = error;
                _completed = false;
            }
        }
    }
}
