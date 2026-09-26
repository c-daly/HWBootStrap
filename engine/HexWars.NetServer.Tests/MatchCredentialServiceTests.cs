using System.Security.Cryptography;
using System.Text.RegularExpressions;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// What a join credential has to be worth, and what it must never be.
    ///
    /// Two of these tests are about what does NOT happen. The store must not learn the credential, only a
    /// hash of it, so that a database dump is not a list of live sessions. And a string that cannot be a
    /// credential must not reach the store at all, because the websocket handshake is unauthenticated and
    /// a lookup per junk frame would let anyone aim the internet at the database through it.
    /// </summary>
    [TestFixture]
    public sealed class MatchCredentialServiceTests
    {
        const string SetupWire = "annihilation 9 7 0 7 3 1 1 1 3 0";
        const string EngineVersion = "hexwars-engine/1";
        const string BuildId = "test-build";
        const int ProtocolVersion = 2;

        /// <summary>Whole seconds on purpose: both stores keep timestamps to the microsecond, so a clock
        /// carrying .NET ticks would compare unequal after a round trip for reasons that have nothing to do
        /// with credentials.</summary>
        static readonly DateTimeOffset Origin = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);

        static int _identifiers;

        static CancellationToken Ct => CancellationToken.None;

        InMemoryMatchStore _storage = null!;
        CountingMatchStore _store = null!;
        FakeTimeProvider _clock = null!;
        MatchCredentialService _service = null!;
        Guid _matchId;
        string _seat0 = null!;
        string _seat1 = null!;

        [SetUp]
        public async Task CreateAWaitingMatchWithTwoSeats()
        {
            _storage = new InMemoryMatchStore();
            _store = new CountingMatchStore(_storage);
            _clock = new FakeTimeProvider(Origin);
            _service = NewService(MatchHostingOptions.DefaultJoinTokenTtlSeconds);

            _seat0 = NextSteamId();
            _seat1 = NextSteamId();
            _matchId = (await _storage.CreateMatchForLobbyAsync(Request(_seat0, _seat1), Ct)).Match.MatchId;
            _store.ResetCounts();
        }

        MatchCredentialService NewService(int ttlSeconds) => new(
            _store,
            Options.Create(new MatchHostingOptions { JoinTokenTtlSeconds = ttlSeconds }),
            _clock,
            NullLogger<MatchCredentialService>.Instance);

        static CreateMatchRequest Request(string seat0, string seat1) => new(
            NextLobbyId(), SetupWire, EngineVersion, ProtocolVersion, BuildId,
            new[] { (seat0, 0), (seat1, 1) }, Origin);

        static string NextLobbyId() => "1097752" + Interlocked.Increment(ref _identifiers).ToString("D11");

        static string NextSteamId() => "7656119" + Interlocked.Increment(ref _identifiers).ToString("D10");

        [Test]
        public async Task AnIssuedCredential_ValidatesToTheSeatItWasIssuedFor()
        {
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat1, Ct);

            CredentialValidation? seat = await _service.ValidateAsync(_matchId, issued.Credential, Ct);

            Assert.That(seat, Is.Not.Null);
            Assert.That(seat!.MatchId, Is.EqualTo(_matchId));
            Assert.That(seat.SteamId, Is.EqualTo(_seat1));
            Assert.That(seat.Seat, Is.EqualTo(1));
        }

        [Test]
        public async Task AnIssuedCredential_IsFortyThreeCharactersOfBase64Url()
        {
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            Assert.That(issued.Credential, Has.Length.EqualTo(43));
            Assert.That(Regex.IsMatch(issued.Credential, "^[A-Za-z0-9_-]+$"), Is.True,
                "the credential travels in JSON and in a websocket frame, so it must need no escaping");
        }

        [Test]
        public async Task TwoCredentialsForTheSameSeat_AreNeverTheSameString()
        {
            var seen = new HashSet<string>(StringComparer.Ordinal);

            for (int i = 0; i < 32; i++)
                Assert.That(seen.Add((await _service.IssueAsync(_matchId, _seat0, Ct)).Credential), Is.True,
                    "a repeated credential would mean the randomness is not");
        }

        [Test]
        public async Task Issuing_StoresTheHashAndNeverTheCredential()
        {
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            Assert.That(CredentialEncoding.TryFromBase64Url(issued.Credential, out byte[] raw), Is.True);
            byte[] expected = SHA256.HashData(raw);

            JoinCredentialRecord? stored = await _storage.FindJoinCredentialAsync(expected, Ct);
            Assert.That(stored, Is.Not.Null, "the row is found by the SHA-256 of the credential bytes");
            Assert.That(stored!.CredentialHash, Is.EqualTo(expected));
            Assert.That(stored.CredentialHash, Is.Not.EqualTo(raw), "what is stored is not the secret");
            Assert.That(stored.MatchId, Is.EqualTo(_matchId));
            Assert.That(stored.SteamId, Is.EqualTo(_seat0));
            Assert.That(await _storage.FindJoinCredentialAsync(raw, Ct), Is.Null,
                "the credential bytes themselves are not a key into the store");
        }

        [Test]
        public async Task ACredentialIssuedForAnotherMatch_DoesNotOpenThisOne()
        {
            string strangerSeat0 = NextSteamId(), strangerSeat1 = NextSteamId();
            Guid other = (await _storage.CreateMatchForLobbyAsync(
                Request(strangerSeat0, strangerSeat1), Ct)).Match.MatchId;

            IssuedCredential issued = await _service.IssueAsync(other, strangerSeat0, Ct);

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Null);
            Assert.That(await _service.ValidateAsync(other, issued.Credential, Ct), Is.Not.Null,
                "the same credential still works for the match it names");
        }

        [Test]
        public async Task ACredentialWhoseSeatHasGone_DoesNotValidate()
        {
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);
            _store.HideSeats = true;

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Null);
        }

        [Test]
        public async Task TheExpiryFollowsTheConfiguredTtl_AndTheCredentialDiesOnIt()
        {
            _service = NewService(120);

            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);
            Assert.That(issued.ExpiresAt, Is.EqualTo(Origin.AddSeconds(120)));

            _clock.Advance(TimeSpan.FromSeconds(119));
            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Not.Null,
                "a credential works right up to its expiry");

            _clock.Advance(TimeSpan.FromSeconds(1));
            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Null,
                "and is dead at it");
        }

        [Test]
        public async Task IssuingAgain_RevokesWhatTheSeatAlreadyHeld()
        {
            IssuedCredential first = await _service.IssueAsync(_matchId, _seat0, Ct);
            _clock.Advance(TimeSpan.FromSeconds(5));
            IssuedCredential second = await _service.IssueAsync(_matchId, _seat0, Ct);

            Assert.That(second.Credential, Is.Not.EqualTo(first.Credential));
            Assert.That(await _service.ValidateAsync(_matchId, first.Credential, Ct), Is.Null,
                "the credential an abandoned reconnect left behind is dead");
            Assert.That(await _service.ValidateAsync(_matchId, second.Credential, Ct), Is.Not.Null);
        }

        [Test]
        public async Task IssuingForOneSeat_LeavesTheOtherSeatAlone()
        {
            IssuedCredential zero = await _service.IssueAsync(_matchId, _seat0, Ct);
            IssuedCredential one = await _service.IssueAsync(_matchId, _seat1, Ct);
            IssuedCredential oneAgain = await _service.IssueAsync(_matchId, _seat1, Ct);

            Assert.That((await _service.ValidateAsync(_matchId, zero.Credential, Ct))?.Seat, Is.EqualTo(0),
                "one player reconnecting must not sign the other one out");
            Assert.That(await _service.ValidateAsync(_matchId, one.Credential, Ct), Is.Null);
            Assert.That((await _service.ValidateAsync(_matchId, oneAgain.Credential, Ct))?.Seat, Is.EqualTo(1));
        }

        [Test]
        public void IssuingForASteamIdThatHoldsNoSeat_IsAnArgumentException()
        {
            string stranger = NextSteamId();

            Assert.ThrowsAsync<ArgumentException>(() => _service.IssueAsync(_matchId, stranger, Ct),
                "a credential is bound to a seat, so there is nothing to mint for a player who has none");
        }

        [Test]
        public async Task AWellFormedCredentialThatWasNeverIssued_IsRefusedAfterOneLookup()
        {
            string never = CredentialEncoding.ToBase64Url(
                RandomNumberGenerator.GetBytes(CredentialEncoding.CredentialBytes));
            _store.ResetCounts();

            Assert.That(await _service.ValidateAsync(_matchId, never, Ct), Is.Null);
            Assert.That(_store.Reads, Is.EqualTo(1), "one hash lookup, and nothing after it");
            Assert.That(_store.Writes, Is.Zero, "validating never writes");
        }

        [TestCaseSource(nameof(CredentialsThatCannotBeCredentials))]
        public async Task AMalformedCredential_IsRefusedWithoutReachingTheStore(string credential)
        {
            await _service.IssueAsync(_matchId, _seat0, Ct);
            _store.ResetCounts();

            Assert.That(await _service.ValidateAsync(_matchId, credential, Ct), Is.Null);
            Assert.That(_store.Calls, Is.Zero,
                "a string that cannot be a credential must not become a database query");
        }

        [Test]
        public async Task IssuingLongAfterAMatchHasEnded_IsRefusedRatherThanStored()
        {
            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Completed, 0, Origin, Ct);

            _clock.Advance(
                TimeSpan.FromSeconds(MatchHostingOptions.DefaultTerminalReconnectSeconds + 60));

            var refused = Assert.ThrowsAsync<InvalidOperationException>(
                () => _service.IssueAsync(_matchId, _seat0, Ct));

            Assert.That(refused!.Message, Is.EqualTo(MatchCredentialService.MatchNotOpenMessage));
        }

        [Test]
        public async Task IssuingInsideTheTerminalWindow_HandsBackACredentialCappedAtTheWindow()
        {
            // A seat that missed the final APPLY has to be able to ask for a way back in, and the way back
            // in has to be no longer than the window it was granted under: a credential that outlived it
            // would be a working key to a match nobody can play.
            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Completed, 0, Origin, Ct);

            _clock.Advance(TimeSpan.FromMinutes(5));

            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            Assert.That(issued.ExpiresAt,
                Is.EqualTo(Origin.AddSeconds(MatchHostingOptions.DefaultTerminalReconnectSeconds)),
                "the credential ends when the window does, not 15 minutes after it was asked for");
            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Not.Null);
        }

        [Test]
        public async Task IssuingForAMatchThatEndedWithoutStarting_IsRefused()
        {
            // Nothing started, so there is no final position to come back for. The window exists to deal an
            // ending, and a match that expired while waiting for barracks has none.
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Expired, null, Origin, Ct);

            var refused = Assert.ThrowsAsync<InvalidOperationException>(
                () => _service.IssueAsync(_matchId, _seat0, Ct));

            Assert.That(refused!.Message, Is.EqualTo(MatchCredentialService.MatchNotOpenMessage));
        }

        [Test]
        public async Task ACredentialKeepsValidatingBrieflyAfterTheMatchEnds()
        {
            // The last APPLY of a game is the frame most likely to be lost: it goes out at the instant the
            // match becomes terminal. A player whose socket dropped a moment earlier has no other way to
            // learn how the game they were playing ended, so the credential still opens the match for a
            // short while afterwards.
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Completed, 0, Origin, Ct);

            _clock.Advance(TimeSpan.FromSeconds(30));

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Not.Null);
        }

        [Test]
        public async Task ACredentialStopsValidatingOnceTheTerminalWindowHasPassed()
        {
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Completed, 0, Origin, Ct);

            // Past the window and still inside the credential TTL, so the refusal is about the match and
            // not about the credential.
            _clock.Advance(
                TimeSpan.FromSeconds(MatchHostingOptions.DefaultTerminalReconnectSeconds + 60));

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Null,
                "the window is for learning how the game ended, and it closes");
        }

        [Test]
        public async Task ALiveCredentialStopsBeingValidWhenTheTerminalWindowCloses()
        {
            // The expiry stored on a credential issued while the match was still being played knows
            // nothing about the ending that came later, so a socket holding one would outlive the window
            // by whatever was left of its TTL. The re-check has to judge the match as well as the token.
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);
            Assert.That(CredentialEncoding.TryFromBase64Url(issued.Credential, out byte[] raw), Is.True);
            byte[] hash = SHA256.HashData(raw);

            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Completed, 0, Origin, Ct);

            Assert.That(
                await _service.IsStillValidAsync(hash, _matchId, Origin.AddMinutes(5), Ct), Is.True);

            DateTimeOffset past = Origin.AddSeconds(
                MatchHostingOptions.DefaultTerminalReconnectSeconds + 60);

            Assert.That(past, Is.LessThan(issued.ExpiresAt),
                "the credential itself is still unexpired, so the refusal can only be about the window");
            Assert.That(await _service.IsStillValidAsync(hash, _matchId, past, Ct), Is.False);

            // And the boundary itself is the closing instant, not the last usable one - the same strict
            // test the handshake and both stores apply.
            DateTimeOffset boundary =
                Origin.AddSeconds(MatchHostingOptions.DefaultTerminalReconnectSeconds);

            Assert.That(
                await _service.IsStillValidAsync(hash, _matchId, boundary.AddTicks(-1), Ct), Is.True);
            Assert.That(await _service.IsStillValidAsync(hash, _matchId, boundary, Ct), Is.False);
        }

        [Test]
        public async Task ACredentialValidatesIntoAnAbandonedMatchThatHadStarted()
        {
            // A game the reaper abandoned underneath its players ended just as definitely as one somebody
            // won, and the seats deserve to be shown the same final position rather than a bare refusal.
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            await _storage.TryStartMatchAsync(_matchId, "START-REPLAY", Origin, Ct);
            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Abandoned, null, Origin, Ct);

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Not.Null);
        }

        [Test]
        public async Task ALiveCredentialIsInvalidTheMomentAMatchThatNeverStartedExpires()
        {
            // The window belongs to a match that was PLAYED. One that expired while still waiting for
            // barracks has no game in it, so there is nothing for a socket to come back and be shown - and
            // a socket still holding a credential for it is holding nothing.
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);
            Assert.That(CredentialEncoding.TryFromBase64Url(issued.Credential, out byte[] raw), Is.True);
            byte[] hash = SHA256.HashData(raw);

            Assert.That(await _service.IsStillValidAsync(hash, _matchId, Origin, Ct), Is.True);

            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Expired, null, Origin, Ct);

            Assert.That(await _service.IsStillValidAsync(hash, _matchId, Origin, Ct), Is.False,
                "immediately, not after the reconnect window it was never entitled to");
        }

        [Test]
        public async Task ACredentialNeverValidatesIntoAMatchThatEndedWithoutStarting()
        {
            // No start replay means no game was ever dealt, so there is no ending to show anybody and the
            // window has nothing to be for.
            IssuedCredential issued = await _service.IssueAsync(_matchId, _seat0, Ct);

            await _storage.TryCompleteMatchAsync(_matchId, MatchStatus.Expired, null, Origin, Ct);

            Assert.That(await _service.ValidateAsync(_matchId, issued.Credential, Ct), Is.Null);
        }

        [TestCaseSource(nameof(CredentialsThatCannotBeCredentials))]
        public void CredentialEncoding_RefusesAnythingThatIsNotACanonicalCredential(string text)
        {
            Assert.That(CredentialEncoding.TryFromBase64Url(text, out byte[] raw), Is.False);
            Assert.That(raw, Is.Empty, "a refusal hands back nothing to dereference");
        }

        [Test]
        public void CredentialEncoding_RoundTripsThirtyTwoRandomBytes()
        {
            for (int i = 0; i < 200; i++)
            {
                byte[] raw = RandomNumberGenerator.GetBytes(CredentialEncoding.CredentialBytes);
                string text = CredentialEncoding.ToBase64Url(raw);

                Assert.That(text, Has.Length.EqualTo(CredentialEncoding.CredentialCharacters));
                Assert.That(Regex.IsMatch(text, "^[A-Za-z0-9_-]+$"), Is.True);
                Assert.That(CredentialEncoding.TryFromBase64Url(text, out byte[] back), Is.True);
                Assert.That(back, Is.EqualTo(raw));
            }
        }

        static IEnumerable<TestCaseData> CredentialsThatCannotBeCredentials()
        {
            const string Alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

            byte[] raw = new byte[CredentialEncoding.CredentialBytes];
            for (int i = 0; i < raw.Length; i++) raw[i] = (byte)((i * 7) + 3);

            string valid = CredentialEncoding.ToBase64Url(raw);
            string padded = Convert.ToBase64String(raw);

            yield return new TestCaseData(string.Empty).SetName("{m}(empty)");
            yield return new TestCaseData("abc").SetName("{m}(three characters)");
            yield return new TestCaseData(valid[..42]).SetName("{m}(forty two characters)");
            yield return new TestCaseData(valid + "A").SetName("{m}(forty four characters)");
            yield return new TestCaseData(padded).SetName("{m}(padded standard base64)");
            yield return new TestCaseData("+" + valid[1..]).SetName("{m}(standard base64 plus)");
            yield return new TestCaseData("/" + valid[1..]).SetName("{m}(standard base64 slash)");
            yield return new TestCaseData(valid[..42] + "!").SetName("{m}(outside the alphabet)");
            yield return new TestCaseData(string.Empty.PadRight(43)).SetName("{m}(only spaces)");

            // The same 32 bytes spelled a second way. The last character of a 43-character encoding carries
            // two bits that decode to nothing, so a lenient decoder would treat four different strings as
            // one credential, and the value the client holds would stop being the only one that works.
            int last = Alphabet.IndexOf(valid[42]);
            yield return new TestCaseData(valid[..42] + Alphabet[last ^ 1])
                .SetName("{m}(non canonical trailing bits)");
        }
    }

    /// <summary>
    /// The same issue-then-validate against the real database. The in-memory double cannot show that the
    /// hash survives a bytea round trip, that timestamptz gives back the expiry the service computed, or
    /// that the composite foreign key from a credential to a seat is satisfied by what the service writes,
    /// and those are exactly the ways this could work in tests and fail in production.
    /// </summary>
    [TestFixture]
    public sealed class MatchCredentialServicePostgresTests
    {
        static readonly DateTimeOffset Origin = new(2026, 9, 4, 12, 0, 0, TimeSpan.Zero);

        static CancellationToken Ct => CancellationToken.None;

        [Test]
        public async Task IssueThenValidate_AgainstPostgres_ReturnsTheStoredSeat()
        {
            PostgresTestDatabase database = await PostgresTestDatabase.GetAsync();
            await database.ResetAsync();
            await database.ApplyMigrationsAsync();

            var store = new PostgresMatchStore(database.DataSource, NullLogger<PostgresMatchStore>.Instance);
            var clock = new FakeTimeProvider(Origin);
            var service = new MatchCredentialService(
                store,
                Options.Create(new MatchHostingOptions { JoinTokenTtlSeconds = 120 }),
                clock,
                NullLogger<MatchCredentialService>.Instance);

            const string Seat0 = "76561190000000001";
            const string Seat1 = "76561190000000002";
            CreateMatchResult created = await store.CreateMatchForLobbyAsync(
                new CreateMatchRequest("109775200000000001", "annihilation 9 7 0 7 3 1 1 1 3 0",
                    "hexwars-engine/1", 2, "test-build", new[] { (Seat0, 0), (Seat1, 1) }, Origin), Ct);
            Guid matchId = created.Match.MatchId;

            IssuedCredential issued = await service.IssueAsync(matchId, Seat1, Ct);
            Assert.That(issued.ExpiresAt, Is.EqualTo(Origin.AddSeconds(120)));

            CredentialValidation? seat = await service.ValidateAsync(matchId, issued.Credential, Ct);
            Assert.That(seat, Is.Not.Null);
            Assert.That(seat!.MatchId, Is.EqualTo(matchId));
            Assert.That(seat.SteamId, Is.EqualTo(Seat1));
            Assert.That(seat.Seat, Is.EqualTo(1));

            IssuedCredential replacement = await service.IssueAsync(matchId, Seat1, Ct);
            Assert.That(await service.ValidateAsync(matchId, issued.Credential, Ct), Is.Null,
                "the superseded credential is revoked in the database, not merely forgotten");
            Assert.That(await service.ValidateAsync(matchId, replacement.Credential, Ct), Is.Not.Null);

            clock.Advance(TimeSpan.FromSeconds(120));
            Assert.That(await service.ValidateAsync(matchId, replacement.Credential, Ct), Is.Null,
                "the expiry that came back out of timestamptz is the one the service wrote");
        }
    }

    /// <summary>
    /// The composition root. A service that is registered but cannot be built is a startup crash in
    /// Development, where the container is validated when it is built, so both halves are worth asserting:
    /// the credential service resolves for a deployment that has a database, and a legacy deployment with
    /// no database still starts rather than failing on a service it was never going to use.
    /// </summary>
    [TestFixture]
    public sealed class MatchCredentialServiceCompositionTests
    {
        [Test]
        public void WithADatabaseConfigured_TheCredentialServiceResolves()
        {
            using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("DATABASE_URL", "postgres://u:p@db.invalid:5432/hexwars");
                // That host does not resolve. This test is about the container rather than about storage,
                // so the startup migration that would rightly refuse to boot goes.
                builder.ConfigureServices(services =>
                    services.Remove(services.Single(d => d.ImplementationType == typeof(MigrationHostedService))));
            });

            var service = factory.Services.GetRequiredService<IMatchCredentialService>();

            Assert.That(service, Is.TypeOf<MatchCredentialService>());
            Assert.That(factory.Services.GetRequiredService<IMatchCredentialService>(), Is.SameAs(service),
                "one instance: it holds no per-request state and the store behind it is already shared");
            Assert.That(factory.Services.GetRequiredService<TimeProvider>(), Is.SameAs(TimeProvider.System),
                "the real clock in production, and a seam a test can replace");
        }

        [Test]
        public void WithNoDatabaseConfigured_TheHostStillStarts()
        {
            using var factory = new WebApplicationFactory<Program>().WithWebHostBuilder(
                builder => builder.UseEnvironment("Development"));

            Assert.That(factory.Services.GetService<IMatchCredentialService>(), Is.Null,
                "a legacy deployment has no store, so there is nothing to issue credentials against");
            Assert.That(factory.Services.GetRequiredService<TimeProvider>(), Is.Not.Null);
        }
    }
}
