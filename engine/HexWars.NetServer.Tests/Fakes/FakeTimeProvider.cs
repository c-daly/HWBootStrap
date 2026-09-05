namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// A clock a test moves by hand.
    ///
    /// Expiry is the only interesting thing about a join credential that takes time to happen, and a test
    /// that waited for it would either be slow or would have to shrink the TTL until the assertion stopped
    /// being about the production value. Injecting the clock instead lets the same test use a realistic TTL
    /// and still land exactly on the boundary, where the interesting bug lives.
    ///
    /// Timers are OPT IN, through <see cref="VirtualTimers"/>. Most of this suite schedules nothing and
    /// wants the real ones - the heartbeat is genuinely about elapsed seconds, and silently virtualising
    /// its ticks would stop those tests testing anything. A test that has to drive a backoff, where the
    /// production delays are minutes, turns them on and winds the clock instead of waiting.
    /// </summary>
    public sealed class FakeTimeProvider(DateTimeOffset utcNow) : TimeProvider
    {
        readonly object _gate = new();
        readonly List<VirtualTimer> _timers = new();

        DateTimeOffset _utcNow = utcNow;
        long _timersCreated;

        /// <summary>How many timers this clock has handed out, ever. A test that has to know a backoff is
        /// waiting before it winds the clock on watches this rather than racing the loop that schedules
        /// it.</summary>
        public long TimersCreated
        {
            get { lock (_gate) return _timersCreated; }
        }

        /// <summary>Timers that exist and have not been disposed.</summary>
        public int PendingTimers
        {
            get { lock (_gate) return _timers.Count; }
        }

        /// <summary>
        /// When set, CreateTimer hands out timers this clock owns and Advance fires the ones that came due.
        /// Off by default, which leaves the base implementation and its real timers. Set it before the host
        /// that will schedule anything starts.
        /// </summary>
        public bool VirtualTimers { get; set; }

        public override DateTimeOffset GetUtcNow()
        {
            lock (_gate) return _utcNow;
        }

        public void Advance(TimeSpan delta) => SetUtcNow(GetUtcNow().Add(delta));

        public void SetUtcNow(DateTimeOffset value)
        {
            List<VirtualTimer> due;

            lock (_gate)
            {
                _utcNow = value;
                if (!VirtualTimers) return;

                due = new List<VirtualTimer>();
                foreach (VirtualTimer timer in _timers)
                {
                    if (timer.TakeIfDue(value)) due.Add(timer);
                }
            }

            // Outside the lock, and off this thread: a callback may create or dispose a timer, and a
            // PeriodicTimer continuation resumed inline would put whatever awaits it onto the thread that
            // is driving the test.
            foreach (VirtualTimer timer in due) timer.Fire();
        }

        public override ITimer CreateTimer(
            TimerCallback callback, object? state, TimeSpan dueTime, TimeSpan period)
        {
            if (!VirtualTimers) return base.CreateTimer(callback, state, dueTime, period);

            var timer = new VirtualTimer(this, callback, state);

            lock (_gate)
            {
                _timers.Add(timer);
                _timersCreated++;
            }

            timer.Change(dueTime, period);
            return timer;
        }

        void Forget(VirtualTimer timer)
        {
            lock (_gate) _timers.Remove(timer);
        }

        /// <summary>A timer that only ever fires because a test said the clock had moved.</summary>
        sealed class VirtualTimer(FakeTimeProvider owner, TimerCallback callback, object? state) : ITimer
        {
            DateTimeOffset? _due;
            TimeSpan _period = Timeout.InfiniteTimeSpan;

            public bool Change(TimeSpan dueTime, TimeSpan period)
            {
                lock (owner._gate)
                {
                    _period = period;
                    _due = dueTime == Timeout.InfiniteTimeSpan ? null : owner._utcNow.Add(dueTime);
                }

                return true;
            }

            /// <summary>Called with the clock lock held. True when this timer now owes a callback.</summary>
            internal bool TakeIfDue(DateTimeOffset now)
            {
                if (_due is null || _due > now) return false;

                _due = _period == Timeout.InfiniteTimeSpan || _period <= TimeSpan.Zero
                    ? null
                    : now.Add(_period);

                return true;
            }

            internal void Fire() => ThreadPool.UnsafeQueueUserWorkItem(_ => callback(state), null);

            public void Dispose() => owner.Forget(this);

            public ValueTask DisposeAsync()
            {
                Dispose();
                return ValueTask.CompletedTask;
            }
        }
    }
}
