namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// A clock a test moves by hand, including the timers scheduled against it.
    ///
    /// Expiry and backoff are the two interesting things about this server that take time to happen, and a
    /// test that waited for either would be slow or would have to shrink the production value until the
    /// assertion stopped being about it. Injecting the clock lets the same test use the real interval and
    /// still land exactly on the boundary, where the interesting bug lives.
    ///
    /// Timers are driven rather than left to the base implementation, which would quietly use the system
    /// clock: a retry loop waiting thirty seconds on <see cref="TimeProvider"/> would then take thirty real
    /// seconds however far a test wound this clock forward. <see cref="Advance"/> fires everything that
    /// falls due on the way, in order, moving the clock to each due time as it goes - so a callback that
    /// reads the clock sees the moment it was scheduled for and not the end of the jump.
    /// </summary>
    public sealed class FakeTimeProvider(DateTimeOffset utcNow) : TimeProvider
    {
        readonly object _gate = new();
        readonly List<FakeTimer> _timers = new();

        DateTimeOffset _utcNow = utcNow;

        public override DateTimeOffset GetUtcNow()
        {
            lock (_gate) return _utcNow;
        }

        public void Advance(TimeSpan delta) => SetUtcNow(GetUtcNow().Add(delta));

        /// <summary>
        /// Timers armed and waiting on this clock.
        ///
        /// A test that winds the clock forward before the code under test has scheduled its wait has
        /// simply moved the clock past nothing: the timer is created afterwards, due relative to the new
        /// now, and never fires. Waiting for this to rise first is how a test says it is ready to jump.
        /// </summary>
        public int ScheduledTimers
        {
            get { lock (_gate) return _timers.Count(timer => timer.DueAt is not null); }
        }

        public void SetUtcNow(DateTimeOffset value)
        {
            // One timer at a time, oldest due first, with the clock set to its due moment before it runs.
            // Firing them all at the end of the jump would let a callback that schedules another timer
            // schedule it in the past.
            while (true)
            {
                FakeTimer? next;

                lock (_gate)
                {
                    next = null;

                    foreach (FakeTimer timer in _timers)
                    {
                        if (timer.DueAt is not DateTimeOffset due || due > value) continue;
                        if (next is null || due < next.DueAt) next = timer;
                    }

                    if (next is null)
                    {
                        if (value > _utcNow) _utcNow = value;
                        return;
                    }

                    _utcNow = next.DueAt!.Value;
                    next.Rearm(_utcNow);
                }

                // Outside the lock: a callback is free to read this clock, or schedule against it.
                next.Fire();
            }
        }

        public override ITimer CreateTimer(
            TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
        {
            ArgumentNullException.ThrowIfNull(callback);

            var timer = new FakeTimer(this, callback, state);

            lock (_gate) _timers.Add(timer);

            timer.Change(dueTime, period);
            return timer;
        }

        void Forget(FakeTimer timer)
        {
            lock (_gate) _timers.Remove(timer);
        }

        sealed class FakeTimer(FakeTimeProvider clock, TimerCallback callback, object? state) : ITimer
        {
            TimeSpan _period = Timeout.InfiniteTimeSpan;

            /// <summary>When this timer next fires, or null when it is not armed.</summary>
            public DateTimeOffset? DueAt { get; private set; }

            public bool Change(TimeSpan dueTime, TimeSpan period)
            {
                lock (clock._gate)
                {
                    _period = period;
                    DueAt = dueTime == Timeout.InfiniteTimeSpan ? null : clock._utcNow + dueTime;
                }

                return true;
            }

            /// <summary>Called under the clock lock, as the timer is about to fire.</summary>
            public void Rearm(DateTimeOffset now) =>
                DueAt = _period == Timeout.InfiniteTimeSpan || _period <= TimeSpan.Zero
                    ? null
                    : now + _period;

            public void Fire() => callback(state);

            public void Dispose()
            {
                lock (clock._gate) DueAt = null;
                clock.Forget(this);
            }

            public ValueTask DisposeAsync()
            {
                Dispose();
                return ValueTask.CompletedTask;
            }
        }
    }
}
