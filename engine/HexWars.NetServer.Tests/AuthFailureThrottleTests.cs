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
