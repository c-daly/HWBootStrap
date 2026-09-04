using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Steam;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Options;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// One test per lobby rule, plus the edge cases that decide whether a hostile lobby can talk this
    /// server into a match it should refuse. Every failing case also asserts that the operator-facing
    /// Detail carries no Steam id, because those strings go to logs verbatim.
    /// </summary>
    [TestFixture]
    public class SteamLobbyValidationTests
    {
        const string OwnerId = "76561198000000001";
        const string GuestId = "76561198000000002";
        const string StrangerId = "76561198000000003";
        const string LobbyId = "109775241010407638";
        const uint AppId = 480000u;
        const string AppIdText = "480000";
        const int ProtocolVersion = 2;
        const string BuildId = "test-build";
        const int Seed = 4242;

        // ---- harness -------------------------------------------------------

        /// <summary>
        /// Produces a lobby that passes every rule, so each test mutates exactly the one thing it is
        /// about. Anything a test does not touch stays valid, which is what makes a failure informative.
        /// </summary>
        sealed class LobbyBuilder
        {
            readonly Dictionary<string, string> _metadata;
            readonly List<KeyValuePair<string, Dictionary<string, string>>> _members;
            string _lobbyId = LobbyId;
            string _owner = OwnerId;

            public LobbyBuilder()
            {
                _metadata = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    [SteamLobbyKeys.App] = AppIdText,
                    [SteamLobbyKeys.Protocol] = "2",
                    [SteamLobbyKeys.Build] = BuildId,
                    [SteamLobbyKeys.Ruleset] = SteamLobbyRules.QuickRuleset,
                    [SteamLobbyKeys.Setup] = SteamLobbyRules.QuickMatchSetup(Seed).ToWire(),
                };

                _members = new List<KeyValuePair<string, Dictionary<string, string>>>
                {
                    ReadyMember(OwnerId),
                    ReadyMember(GuestId),
                };
            }

            static KeyValuePair<string, Dictionary<string, string>> ReadyMember(string id) =>
                new(id, new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    [SteamLobbyKeys.MemberReady] = SteamLobbyKeys.ReadyTrue,
                });

            public LobbyBuilder WithOwner(string id)
            {
                _owner = id;
                return this;
            }

            /// <summary>Sets a lobby data value, or removes the key when the value is null.</summary>
            public LobbyBuilder WithMeta(string key, string? value)
            {
                if (value is null)
                {
                    _metadata.Remove(key);
                }
                else
                {
                    _metadata[key] = value;
                }

                return this;
            }

            /// <summary>Replaces the roster with the given ids, all of them ready.</summary>
            public LobbyBuilder WithMembers(params string[] ids)
            {
                _members.Clear();
                foreach (var id in ids)
                {
                    _members.Add(ReadyMember(id));
                }

                return this;
            }

            public LobbyBuilder WithMemberMeta(string id, string key, string? value)
            {
                foreach (var member in _members)
                {
                    if (!string.Equals(member.Key, id, StringComparison.Ordinal))
                    {
                        continue;
                    }

                    if (value is null)
                    {
                        member.Value.Remove(key);
                    }
                    else
                    {
                        member.Value[key] = value;
                    }
                }

                return this;
            }

            public SteamLobbySnapshot Snapshot() => new(
                _lobbyId,
                _owner,
                _members.Select(m => new SteamLobbyMember(m.Key, m.Value)).ToArray(),
                new Dictionary<string, string>(_metadata, StringComparer.Ordinal));
        }

        static LobbyBuilder Lobby() => new();

        static SteamLobbyValidator Validator(
            uint appId = AppId, int protocolVersion = ProtocolVersion, string[]? compatibleBuilds = null) =>
            new(
                Options.Create(new SteamOptions { AppId = appId, PublisherWebApiKey = "test-key" }),
                Options.Create(new MatchHostingOptions
                {
                    ProtocolVersion = protocolVersion,
                    CompatibleClientBuilds = compatibleBuilds ?? new[] { BuildId },
                }));

        static SteamApiException AssertRejected(
            SteamFailure expected,
            string detail,
            LobbyBuilder lobby,
            string requester = OwnerId,
            SteamLobbyValidator? validator = null)
        {
            var subject = validator ?? Validator();
            var snapshot = lobby.Snapshot();
            var error = Assert.Throws<SteamApiException>(
                () => subject.ValidateForMatchCreation(snapshot, requester))!;

            Assert.That(error.Failure, Is.EqualTo(expected));
            Assert.That(error.Detail, Does.Contain(detail));
            Assert.That(error.PlayerSafeMessage, Is.EqualTo(SteamFailureMessages.For(expected)));
            AssertNoSteamIds(error.Detail);
            return error;
        }

        static void AssertNoSteamIds(string detail)
        {
            foreach (var id in new[] { OwnerId, GuestId, StrangerId, LobbyId })
            {
                Assert.That(detail, Does.Not.Contain(id), "operator detail must not leak Steam ids");
            }
        }

        // ---- rule 1: id normalisation --------------------------------------

        [Test]
        public void UnparseableMemberId_IsLobbyChanged()
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "unparseable member id",
                Lobby().WithMembers(OwnerId, "not-a-steam-id"));
        }

        [Test]
        public void UnparseableOwnerId_IsLobbyChanged()
        {
            // 12345 is a well-formed number but below the individual-account base, so it is not an account.
            AssertRejected(SteamFailure.LobbyChanged, "unparseable member id", Lobby().WithOwner("12345"));
        }

        // ---- rule 2: membership --------------------------------------------

        [Test]
        public void RequesterOutsideTheLobby_IsNotLobbyMember()
        {
            AssertRejected(SteamFailure.NotLobbyMember, "lobby member", Lobby(), StrangerId);
        }

        [Test]
        public void UnparseableRequesterId_IsNotLobbyMember()
        {
            AssertRejected(SteamFailure.NotLobbyMember, "lobby member", Lobby(), "nonsense");
        }

        // ---- rule 3: ownership ---------------------------------------------

        [Test]
        public void GuestRequestingTheMatch_IsNotLobbyOwner()
        {
            AssertRejected(SteamFailure.NotLobbyOwner, "lobby owner", Lobby(), GuestId);
        }

        // ---- rule 4: app id -------------------------------------------------

        [TestCase(null)]
        [TestCase("")]
        [TestCase("480001")]
        [TestCase("not-a-number")]
        public void WrongOrMissingAppId_IsIncompatibleVersion(string? value)
        {
            AssertRejected(
                SteamFailure.IncompatibleVersion, "app id mismatch",
                Lobby().WithMeta(SteamLobbyKeys.App, value));
        }

        [Test]
        public void AppIdWithLeadingZeros_IsAccepted()
        {
            // hw_app is compared numerically after uint.TryParse, not as text, so a client that writes a
            // zero-padded App ID is still talking about the same application.
            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.App, "0480000").Snapshot(), OwnerId);

            Assert.That(result.OwnerSteamId, Is.EqualTo(OwnerId));
        }

        // ---- rule 5: protocol -----------------------------------------------

        [TestCase(null)]
        [TestCase("")]
        [TestCase("3")]
        [TestCase("two")]
        public void WrongOrMissingProtocol_IsIncompatibleVersion(string? value)
        {
            AssertRejected(
                SteamFailure.IncompatibleVersion, "protocol",
                Lobby().WithMeta(SteamLobbyKeys.Protocol, value));
        }

        // ---- rule 6: build allow-list ---------------------------------------

        [TestCase(null)]
        [TestCase("other-build")]
        public void BuildOutsideTheAllowList_IsIncompatibleVersion(string? value)
        {
            AssertRejected(
                SteamFailure.IncompatibleVersion, "build",
                Lobby().WithMeta(SteamLobbyKeys.Build, value));
        }

        [Test]
        public void EmptyBuildAllowList_AcceptsAnyBuild()
        {
            var validator = Validator(compatibleBuilds: Array.Empty<string>());

            var result = validator.ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.Build, "some-unreviewed-build").Snapshot(), OwnerId);

            Assert.That(result.ClientBuild, Is.EqualTo("some-unreviewed-build"));
        }

        [Test]
        public void EmptyBuildAllowList_AcceptsALobbyWithNoBuildAtAll()
        {
            var validator = Validator(compatibleBuilds: Array.Empty<string>());

            var result = validator.ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.Build, null).Snapshot(), OwnerId);

            Assert.That(result.ClientBuild, Is.Null);
        }

        // ---- rule 7: member count -------------------------------------------

        [Test]
        public void SoloLobby_IsLobbyChanged()
        {
            AssertRejected(SteamFailure.LobbyChanged, "member count", Lobby().WithMembers(OwnerId));
        }

        [Test]
        public void ThreeMemberLobby_IsLobbyChanged()
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "member count",
                Lobby().WithMembers(OwnerId, GuestId, StrangerId));
        }

        // ---- rule 8: readiness ----------------------------------------------

        [TestCase(null)]
        [TestCase("0")]
        [TestCase("true")]
        public void GuestNotReady_IsLobbyChanged(string? value)
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "not ready",
                Lobby().WithMemberMeta(GuestId, SteamLobbyKeys.MemberReady, value));
        }

        [Test]
        public void OwnerNotReady_IsLobbyChanged()
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "not ready",
                Lobby().WithMemberMeta(OwnerId, SteamLobbyKeys.MemberReady, "0"));
        }

        // ---- rule 9: ruleset -------------------------------------------------

        [TestCase(null)]
        [TestCase("")]
        [TestCase("quick-v2")]
        [TestCase("Custom")]
        public void UnknownRuleset_IsLobbyChanged(string? value)
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "ruleset",
                Lobby().WithMeta(SteamLobbyKeys.Ruleset, value));
        }

        // ---- rule 10: setup present -------------------------------------------

        [TestCase(null)]
        [TestCase("")]
        [TestCase("   ")]
        public void MissingSetup_IsLobbyChanged(string? value)
        {
            AssertRejected(
                SteamFailure.LobbyChanged, "setup missing",
                Lobby().WithMeta(SteamLobbyKeys.Setup, value));
        }

        // ---- rule 11: quick-v1 setup equality ---------------------------------

        [Test]
        public void QuickRulesetWithADifferentBoard_IsLobbyChanged()
        {
            var wider = new GameSetup(GameMode.Annihilation, 11, 7, 0, Seed, 3, 1, 1, 1, 3, false);

            AssertRejected(
                SteamFailure.LobbyChanged, "setup does not match quick ruleset",
                Lobby().WithMeta(SteamLobbyKeys.Setup, wider.ToWire()));
        }

        [Test]
        public void QuickRulesetWithADifferentPace_IsLobbyChanged()
        {
            var wholeArmy = new GameSetup(GameMode.Annihilation, 9, 7, 0, Seed, 3, 1, 1, 1, 0, false);

            AssertRejected(
                SteamFailure.LobbyChanged, "setup does not match quick ruleset",
                Lobby().WithMeta(SteamLobbyKeys.Setup, wholeArmy.ToWire()));
        }

        [TestCase(0)]
        [TestCase(-1)]
        [TestCase(10000)]
        public void QuickRulesetSeedOutsideTheClientRange_IsLobbyChanged(int seed)
        {
            // Seed 0 is the case that matters: the engine clamp would quietly turn it into 1 and make an
            // out-of-band setup look like a legitimate quick match, so the range is checked before that.
            var offRange = new GameSetup(GameMode.Annihilation, 9, 7, 0, seed, 3, 1, 1, 1, 3, false);

            AssertRejected(
                SteamFailure.LobbyChanged, "setup does not match quick ruleset",
                Lobby().WithMeta(SteamLobbyKeys.Setup, offRange.ToWire()));
        }

        [TestCase(SteamLobbyRules.MinSeed)]
        [TestCase(SteamLobbyRules.MaxSeed)]
        public void QuickRulesetSeedAtTheRangeEdges_IsAccepted(int seed)
        {
            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.Setup, SteamLobbyRules.QuickMatchSetup(seed).ToWire()).Snapshot(),
                OwnerId);

            Assert.That(result.Setup.Seed, Is.EqualTo(seed));
        }

        // ---- rule 12 and the happy path ---------------------------------------

        [Test]
        public void ValidQuickLobby_SeatsTheOwnerFirstAndReturnsTheQuickSetup()
        {
            var result = Validator().ValidateForMatchCreation(Lobby().Snapshot(), OwnerId);

            Assert.That(result.LobbyId, Is.EqualTo(LobbyId));
            Assert.That(result.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(result.Ruleset, Is.EqualTo(SteamLobbyRules.QuickRuleset));
            Assert.That(result.ClientBuild, Is.EqualTo(BuildId));
            Assert.That(result.Setup.ToWire(), Is.EqualTo(SteamLobbyRules.QuickMatchSetup(Seed).ToWire()));
            Assert.That(result.Players.Count, Is.EqualTo(2));
            Assert.That(result.Players[0].SteamId, Is.EqualTo(OwnerId));
            Assert.That(result.Players[0].Seat, Is.EqualTo(0));
            Assert.That(result.Players[1].SteamId, Is.EqualTo(GuestId));
            Assert.That(result.Players[1].Seat, Is.EqualTo(1));
        }

        [Test]
        public void OwnerListedSecond_StillTakesSeatZero()
        {
            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMembers(GuestId, OwnerId).Snapshot(), OwnerId);

            Assert.That(result.Players[0].SteamId, Is.EqualTo(OwnerId));
            Assert.That(result.Players[0].Seat, Is.EqualTo(0));
            Assert.That(result.Players[1].SteamId, Is.EqualTo(GuestId));
            Assert.That(result.Players[1].Seat, Is.EqualTo(1));
        }

        [TestCase("  76561198000000001  ")]
        [TestCase("076561198000000001")]
        public void NonCanonicalRequesterId_StillMatchesTheOwner(string requester)
        {
            var result = Validator().ValidateForMatchCreation(Lobby().Snapshot(), requester);

            Assert.That(result.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(result.Players[0].SteamId, Is.EqualTo(OwnerId));
        }

        [Test]
        public void NonCanonicalMemberIds_AreReturnedCanonical()
        {
            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMembers(" 76561198000000001", "076561198000000002")
                       .WithOwner("76561198000000001 ")
                       .Snapshot(),
                OwnerId);

            Assert.That(result.OwnerSteamId, Is.EqualTo(OwnerId));
            Assert.That(result.Players[1].SteamId, Is.EqualTo(GuestId));
        }

        // ---- custom ruleset ----------------------------------------------------

        [Test]
        public void CustomRulesetWithAnOversizedBoard_IsClamped()
        {
            var huge = new GameSetup(GameMode.Annihilation, 500, 7, 0, 77, 3, 1, 1, 1, 3, false);

            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.Ruleset, SteamLobbyRules.CustomRuleset)
                       .WithMeta(SteamLobbyKeys.Setup, huge.ToWire())
                       .Snapshot(),
                OwnerId);

            Assert.That(result.Ruleset, Is.EqualTo(SteamLobbyRules.CustomRuleset));
            Assert.That(result.Setup.Width, Is.EqualTo(64));
            Assert.That(result.Setup.Height, Is.EqualTo(7));
        }

        [Test]
        public void CustomRulesetIsNotHeldToTheQuickSeedRange()
        {
            var custom = new GameSetup(GameMode.Territory, 12, 12, 20, 55555, 5, 2, 2, 1, 0, true);

            var result = Validator().ValidateForMatchCreation(
                Lobby().WithMeta(SteamLobbyKeys.Ruleset, SteamLobbyRules.CustomRuleset)
                       .WithMeta(SteamLobbyKeys.Setup, custom.ToWire())
                       .Snapshot(),
                OwnerId);

            Assert.That(result.Setup.ToWire(), Is.EqualTo(custom.Sanitized().ToWire()));
        }

        // ---- EnsureMember --------------------------------------------------------

        [TestCase(GuestId)]
        [TestCase(OwnerId)]
        [TestCase("  76561198000000002 ")]
        public void EnsureMember_AcceptsAnAccountInTheLobby(string steamId)
        {
            Assert.DoesNotThrow(() => Validator().EnsureMember(Lobby().Snapshot(), steamId));
        }

        [TestCase(StrangerId)]
        [TestCase("nonsense")]
        [TestCase("")]
        public void EnsureMember_RejectsAnAccountOutsideTheLobby(string steamId)
        {
            var snapshot = Lobby().Snapshot();

            var error = Assert.Throws<SteamApiException>(
                () => Validator().EnsureMember(snapshot, steamId))!;

            Assert.That(error.Failure, Is.EqualTo(SteamFailure.NotLobbyMember));
            Assert.That(error.PlayerSafeMessage, Is.EqualTo(SteamFailureMessages.NotLobbyMember));
            AssertNoSteamIds(error.Detail);
        }

        // ---- rules helper --------------------------------------------------------

        [Test]
        public void SetupEquals_IgnoresDifferencesTheEngineWouldClampAway()
        {
            var wild = new GameSetup(GameMode.Annihilation, 9, 7, 0, Seed, 3, 1, 1, 1, 3, false);
            var alsoWild = new GameSetup(GameMode.Annihilation, 9, 7, -5, Seed, 3, 1, 1, 1, 3, false);

            Assert.That(SteamLobbyRules.SetupEquals(wild, alsoWild), Is.True);
        }

        // ---- composition -----------------------------------------------------------

        [Test]
        public void Composition_RegistersTheValidatorAsASingleton()
        {
            using var factory = new WebApplicationFactory<Program>();

            var first = factory.Services.GetRequiredService<SteamLobbyValidator>();
            var second = factory.Services.GetRequiredService<SteamLobbyValidator>();

            Assert.That(second, Is.SameAs(first));
        }
    }
}
