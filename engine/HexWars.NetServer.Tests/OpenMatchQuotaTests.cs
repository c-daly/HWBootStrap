using System.Net;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The per-address match quota on its own: admission under concurrency, the rolling window, what a
    /// saturated map does, and which addresses share a bucket.
    ///
    /// These are unit tests because the interesting failures are races and boundaries. An endpoint test can
    /// show that the cap refuses a fourth request; only this can show that twenty callers arriving together
    /// get twenty different answers, which is the difference between a reservation and a question.
    /// </summary>
    [TestFixture]
    public class OpenMatchQuotaTests
    {
        static readonly DateTimeOffset Start = new(2026, 3, 1, 12, 0, 0, TimeSpan.Zero);

        const string Caller = "198.51.100.4";

        FakeTimeProvider _clock = null!;
        OpenMatchQuota _quota = null!;

        [SetUp]
        public void Build() => _quota = Quota(cap: 3);

        OpenMatchQuota Quota(int cap)
        {
            _clock = new FakeTimeProvider(Start);
            var options = new MatchHostingOptions { MaxOpenMatchesPerIp = cap };
            return new OpenMatchQuota(
                Options.Create(options), _clock, NullLogger<OpenMatchQuota>.Instance);
        }

        /// <summary>Takes a seat and charges it, the way a create that allocated a match does.</summary>
        void Create(string caller = Caller)
        {
            Assert.That(_quota.TryReserve(caller, out QuotaLease lease), Is.True);
            lease.Commit();
        }

        // ---- admission --------------------------------------------------------

        [Test]
        public void TwentyCallersArrivingTogether_GetExactlyTheCapManySeats()
        {
            var granted = 0;
            var leases = new QuotaLease[20];
            using var ready = new ManualResetEventSlim();

            var racers = new Task[leases.Length];
            for (var i = 0; i < racers.Length; i++)
            {
                int slot = i;
                racers[i] = Task.Run(() =>
                {
                    ready.Wait();
                    if (_quota.TryReserve(Caller, out QuotaLease lease))
                    {
                        leases[slot] = lease;
                        Interlocked.Increment(ref granted);
                    }
                });
            }

            ready.Set();
            Task.WaitAll(racers);

            Assert.That(granted, Is.EqualTo(3),
                "a reservation is a seat other requests cannot have; asking whether there is room would let "
                + "every one of these read the same low number and proceed");

            foreach (QuotaLease lease in leases) lease?.Commit();
            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(3));
        }

        [Test]
        public void AnOutstandingReservation_CountsAgainstTheCapBeforeItIsCommitted()
        {
            Assert.That(_quota.TryReserve(Caller, out QuotaLease first), Is.True);
            Assert.That(_quota.TryReserve(Caller, out QuotaLease second), Is.True);
            Assert.That(_quota.TryReserve(Caller, out QuotaLease third), Is.True);

            Assert.That(_quota.TryReserve(Caller, out _), Is.False,
                "three seats are held even though nothing has been charged yet");

            first.Release();
            Assert.That(_quota.TryReserve(Caller, out _), Is.True, "a released seat is available again");

            second.Release();
            third.Release();
        }

        [Test]
        public void AReleasedLease_ChargesNothing()
        {
            Assert.That(_quota.TryReserve(Caller, out QuotaLease lease), Is.True);
            lease.Release();

            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(0));

            Create();
            Create();
            Create();
            Assert.That(_quota.TryReserve(Caller, out _), Is.False,
                "the released attempt bought nothing, so three creations still fill the budget");
        }

        [Test]
        public void ALeaseNobodySettled_IsCharged()
        {
            using (QuotaLease abandoned = Reserve()) { }

            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(1),
                "a request that ended without saying what it did may have left a match behind, and the "
                + "expensive mistake is the one where the cap quietly stops counting");
        }

        QuotaLease Reserve()
        {
            Assert.That(_quota.TryReserve(Caller, out QuotaLease lease), Is.True);
            return lease;
        }

        [Test]
        public void ARefusedLeaseThatCreatedAMatchAnyway_IsStillCharged()
        {
            Create();
            Create();
            Create();

            Assert.That(_quota.TryReserve(Caller, out QuotaLease refused), Is.False);

            // The idempotent-retry path runs on without a seat, and can still find the match it was
            // retrying has ended and allocate a new one. That has to be charged or the cap has a hole.
            refused.Commit();

            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(3));
            Assert.That(_quota.TryReserve(Caller, out _), Is.False);
        }

        // ---- the rolling window ----------------------------------------------

        [Test]
        public void TheWindowRolls_RatherThanResettingOnABoundary()
        {
            Create();                                  // t0
            _clock.Advance(TimeSpan.FromMinutes(9));
            Create();                                  // t0 + 9m
            Create();                                  // t0 + 9m

            Assert.That(_quota.TryReserve(Caller, out _), Is.False, "three creations inside ten minutes");

            _clock.Advance(TimeSpan.FromMinutes(1) + TimeSpan.FromSeconds(30));   // t0 + 10m30s
            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(2),
                "only the creation at t0 has aged out");

            Create();                                  // t0 + 10m30s

            // A FIXED window would have restarted at t0 + 10m and this would be the first of a fresh three.
            Assert.That(_quota.TryReserve(Caller, out _), Is.False,
                "the two creations from t0 + 9m are still inside the trailing ten minutes");

            _clock.Advance(TimeSpan.FromMinutes(10));
            Assert.That(_quota.TryReserve(Caller, out _), Is.True, "everything has now aged out");
        }

        [Test]
        public void ACreationExactlyOnTheWindowEdge_HasAgedOut()
        {
            Create();
            _clock.Advance(MatchHostingOptions.OpenMatchWindow);

            Assert.That(_quota.CreationsWithinWindow(Caller), Is.EqualTo(0));
        }

        // ---- saturation -------------------------------------------------------

        [Test]
        public void ASaturatedMap_RefusesNewCallersAndKeepsServingTheOnesItHas()
        {
            _quota = Quota(cap: 2);

            for (var i = 0; i < OpenMatchQuota.MaxTrackedCallers; i++)
            {
                Create("10." + (i / 65536) + "." + (i / 256 % 256) + "." + (i % 256));
            }

            Assert.That(_quota.TrackedCallers, Is.EqualTo(OpenMatchQuota.MaxTrackedCallers));

            Assert.That(_quota.TryReserve("203.0.113.9", out _), Is.False,
                "an address the map has never seen is refused rather than admitted by evicting somebody");

            Assert.That(_quota.TryReserve("10.0.0.0", out QuotaLease known), Is.True,
                "a caller already tracked still has its second seat");
            known.Release();

            Assert.That(_quota.TrackedCallers, Is.EqualTo(OpenMatchQuota.MaxTrackedCallers),
                "and refusing did not grow the map either");
        }

        [Test]
        public void ASaturatedMapRecovers_OnceAWindowHasAgedOut()
        {
            _quota = Quota(cap: 1);

            for (var i = 0; i < OpenMatchQuota.MaxTrackedCallers; i++)
            {
                Create("10." + (i / 65536) + "." + (i / 256 % 256) + "." + (i % 256));
            }

            Assert.That(_quota.TryReserve("203.0.113.9", out _), Is.False);

            _clock.Advance(MatchHostingOptions.OpenMatchWindow + TimeSpan.FromSeconds(1));

            Assert.That(_quota.TryReserve("203.0.113.9", out QuotaLease admitted), Is.True);
            admitted.Release();
        }

        // ---- buckets ----------------------------------------------------------

        [Test]
        public void AnIPv4AddressArrivingOverIPv6_SharesItsBucket() =>
            Assert.That(
                OpenMatchQuota.BucketFor(IPAddress.Parse("::ffff:10.0.0.1")),
                Is.EqualTo(OpenMatchQuota.BucketFor(IPAddress.Parse("10.0.0.1"))));

        [Test]
        public void TwoAddressesInOneIPv6Prefix_ShareABucket() =>
            Assert.That(
                OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:abcd:1234::1")),
                Is.EqualTo(OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:abcd:1234:ffff:ffff:ffff:ffff"))));

        [Test]
        public void TwoDifferentIPv6Prefixes_DoNotShareABucket() =>
            Assert.That(
                OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:abcd:1234::1")),
                Is.Not.EqualTo(OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:abcd:1235::1"))));

        [Test]
        public void AConnectionWithNoAddress_HasABucketOfItsOwn() =>
            Assert.That(OpenMatchQuota.BucketFor(null), Is.EqualTo("unknown"));

        [Test]
        public void AnIPv6ClientCannotBuyQuotaByChangingAddress()
        {
            _quota = Quota(cap: 3);

            for (var host = 1; host <= 3; host++)
            {
                Create(OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:1:1::" + host)));
            }

            Assert.That(
                _quota.TryReserve(OpenMatchQuota.BucketFor(IPAddress.Parse("2001:db8:1:1::99")), out _),
                Is.False,
                "a routed prefix is one client, not eighteen quintillion of them");
        }
    }
}
