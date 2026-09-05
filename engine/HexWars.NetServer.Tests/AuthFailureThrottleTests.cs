using HexWars.NetServer.Auth;
using HexWars.NetServer.Tests.Fakes;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The throttle in isolation, for the one dimension an HTTP test cannot reach: time. A test that
    /// waited five minutes for a window to close would not be run, and one that shrank the window to make
    /// waiting bearable would stop testing the production value.
    /// </summary>
    [TestFixture]
    public class AuthFailureThrottleTests
    {
        static readonly DateTimeOffset Start = new(2026, 1, 1, 0, 0, 0, TimeSpan.Zero);

        const string Caller = "203.0.113.10";

        [Test]
        public void ACallerWithNoHistoryIsNotThrottled()
        {
            var throttle = new AuthFailureThrottle(new FakeTimeProvider(Start));

            Assert.That(throttle.IsThrottled(Caller), Is.False);
        }

        [Test]
        public void TheLimitBitesOnTheConfiguredFailureAndNotBefore()
        {
            var throttle = new AuthFailureThrottle(new FakeTimeProvider(Start));

            for (int failure = 1; failure < AuthFailureThrottle.MaxFailures; failure++)
            {
                throttle.RecordFailure(Caller);
                Assert.That(throttle.IsThrottled(Caller), Is.False, "after failure " + failure);
            }

            throttle.RecordFailure(Caller);
            Assert.That(throttle.IsThrottled(Caller), Is.True);
        }

        [Test]
        public void TheLockoutEndsWhenTheWindowDoes()
        {
            var clock = new FakeTimeProvider(Start);
            var throttle = new AuthFailureThrottle(clock);

            for (int failure = 0; failure < AuthFailureThrottle.MaxFailures; failure++)
            {
                throttle.RecordFailure(Caller);
            }

            // One tick short of the window is still inside it: a lockout that ended early would be a
            // lockout an attacker could simply wait out faster than the configuration says.
            clock.SetUtcNow(Start + AuthFailureThrottle.Window - TimeSpan.FromTicks(1));
            Assert.That(throttle.IsThrottled(Caller), Is.True);

            clock.SetUtcNow(Start + AuthFailureThrottle.Window);
            Assert.That(throttle.IsThrottled(Caller), Is.False);
        }

        [Test]
        public void TheFirstFailureOfANewWindowIsActuallyCounted()
        {
            // Crossing the window boundary is exactly when the sweep drops the old entry, so a failure
            // recorded here can land on a counter that is no longer the dictionary\u0027s. It reads as one free
            // bad ticket per window: the lockout arrives on the eleventh attempt rather than the tenth.
            var clock = new FakeTimeProvider(Start);
            var throttle = new AuthFailureThrottle(clock);

            for (int failure = 0; failure < 3; failure++) throttle.RecordFailure(Caller);

            clock.SetUtcNow(Start + AuthFailureThrottle.Window);
            throttle.RecordFailure(Caller);

            Assert.That(throttle.FailuresFor(Caller), Is.EqualTo(1),
                "the new window has to start at the failure that opened it, not at nothing");

            for (int failure = 2; failure <= AuthFailureThrottle.MaxFailures; failure++)
            {
                Assert.That(throttle.IsThrottled(Caller), Is.False, "before failure " + failure);
                throttle.RecordFailure(Caller);
            }

            Assert.That(throttle.IsThrottled(Caller), Is.True,
                "the limit must bite on the tenth failure of the new window, not the eleventh");
        }

        [Test]
        public void AFreshWindowStartsCountingAgainFromZero()
        {
            var clock = new FakeTimeProvider(Start);
            var throttle = new AuthFailureThrottle(clock);

            for (int failure = 0; failure < AuthFailureThrottle.MaxFailures; failure++)
            {
                throttle.RecordFailure(Caller);
            }

            clock.SetUtcNow(Start + AuthFailureThrottle.Window);
            throttle.RecordFailure(Caller);

            Assert.That(throttle.IsThrottled(Caller), Is.False,
                "the failures from the closed window must not carry into the new one");
        }

        [Test]
        public void TheMapNeverGrowsPastItsCeiling()
        {
            // Twice the ceiling of distinct callers, which is what a client walking an IPv6 /64 looks
            // like. Without a bound this is an allocation anyone can make this process perform.
            var throttle = new AuthFailureThrottle(new FakeTimeProvider(Start));

            for (int caller = 0; caller < 2 * AuthFailureThrottle.MaxTrackedCallers; caller++)
            {
                throttle.RecordFailure("10.0." + (caller / 256) + "." + (caller % 256) + ":" + caller);
                Assert.That(throttle.TrackedCallers, Is.LessThanOrEqualTo(AuthFailureThrottle.MaxTrackedCallers));
            }

            Assert.That(throttle.TrackedCallers, Is.LessThanOrEqualTo(AuthFailureThrottle.MaxTrackedCallers));
        }

        [Test]
        public void TheCeilingHoldsWhenCallersArriveAtOnce()
        {
            // Checking the ceiling and then inserting are two steps, so a single-threaded test can only
            // ever show the map is small on average. Filling it to just under the ceiling and then letting
            // many threads admit fresh callers together is what turns "evict then add" into a claim about
            // the ceiling rather than about the scheduler.
            var throttle = new AuthFailureThrottle(new FakeTimeProvider(Start));

            for (int caller = 0; caller < AuthFailureThrottle.MaxTrackedCallers - 100; caller++)
            {
                throttle.RecordFailure("seed-" + caller);
            }

            Parallel.For(0, 2_000, caller => throttle.RecordFailure("racing-" + caller));

            Assert.That(throttle.TrackedCallers, Is.LessThanOrEqualTo(AuthFailureThrottle.MaxTrackedCallers));
        }

        [Test]
        public void TheSweepIsTimeGatedRatherThanRunningOnEveryFailure()
        {
            var clock = new FakeTimeProvider(Start);
            var throttle = new AuthFailureThrottle(clock);

            for (int failure = 0; failure < 500; failure++) throttle.RecordFailure(Caller + failure);

            Assert.That(throttle.SweepCount, Is.Zero,
                "sweeping per failure costs the size of the map on every failure, worst when it is biggest");

            clock.SetUtcNow(Start + AuthFailureThrottle.SweepInterval);
            throttle.RecordFailure(Caller);
            throttle.RecordFailure(Caller);

            Assert.That(throttle.SweepCount, Is.EqualTo(1), "and once the interval has passed, exactly once");
        }

        [Test]
        public void ASweptCallerIsForgottenRatherThanKept()
        {
            var clock = new FakeTimeProvider(Start);
            var throttle = new AuthFailureThrottle(clock);
            throttle.RecordFailure(Caller);

            clock.SetUtcNow(Start + AuthFailureThrottle.Window + AuthFailureThrottle.SweepInterval);
            throttle.RecordFailure("198.51.100.7");

            Assert.That(throttle.TrackedCallers, Is.EqualTo(1),
                "the caller whose window closed must not still be occupying a slot");
        }

        [Test]
        public void OneCallerCannotSpendAnotherCallerBudget()
        {
            var throttle = new AuthFailureThrottle(new FakeTimeProvider(Start));

            for (int failure = 0; failure < AuthFailureThrottle.MaxFailures; failure++)
            {
                throttle.RecordFailure(Caller);
            }

            Assert.Multiple(() =>
            {
                Assert.That(throttle.IsThrottled(Caller), Is.True);
                Assert.That(throttle.IsThrottled("198.51.100.7"), Is.False);
            });
        }
    }
}
