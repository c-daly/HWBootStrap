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
    /// Only <see cref="GetUtcNow"/> is overridden: nothing under test schedules a timer, and a fake timer
    /// nobody exercises would be code that can only ever be wrong.
    /// </summary>
    public sealed class FakeTimeProvider(DateTimeOffset utcNow) : TimeProvider
    {
        DateTimeOffset _utcNow = utcNow;

        public override DateTimeOffset GetUtcNow() => _utcNow;

        public void Advance(TimeSpan delta) => _utcNow = _utcNow.Add(delta);

        public void SetUtcNow(DateTimeOffset value) => _utcNow = value;
    }
}
