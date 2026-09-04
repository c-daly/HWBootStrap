#nullable enable
using System.Collections.Generic;
using NUnit.Framework;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// Pins the behaviour every consumer of <see cref="ISteamLobbyClient"/> is allowed to rely on:
    /// results and events arrive only from inside Pump, one queued item per call.
    /// </summary>
    public class FakeSteamLobbyClientTests
    {
        static FakeSteamLobbyClient NewClient()
        {
            return new FakeSteamLobbyClient
            {
                LocalSteamId = "76561197960287930",
                LocalDisplayName = "Local",
            };
        }

        [Test]
        public void CreateLobby_DeliversLobbyId_OnlyOnPump()
        {
            using var client = NewClient();
            string? lobbyId = "not-set";
            var calls = 0;

            client.CreateLobby(SteamLobbyVisibility.FriendsOnly, 2, id => { lobbyId = id; calls++; });

            Assert.That(calls, Is.EqualTo(0), "the callback must not run synchronously");
            Assert.That(client.CreateLobbyCalls, Is.EqualTo(1));

            client.Pump();

            Assert.That(calls, Is.EqualTo(1));
            Assert.That(lobbyId, Is.Not.Null.And.Not.Empty);
            Assert.That(client.GetLobby(lobbyId!)!.OwnerSteamId, Is.EqualTo(client.LocalSteamId));
        }

        [Test]
        public void CreateLobby_WhenSteamUnavailable_DeliversNullOnPump()
        {
            using var client = NewClient();
            client.IsAvailable = false;
            string? lobbyId = "not-set";

            client.CreateLobby(SteamLobbyVisibility.Public, 2, id => lobbyId = id);
            client.PumpAll();

            Assert.That(lobbyId, Is.Null);
        }

        [Test]
        public void JoinLobby_AddsLocalMemberToSnapshot()
        {
            using var client = NewClient();
            client.AvailableLobbies.Add(new SteamLobbySearchResult(
                "109775240000000900",
                new Dictionary<string, string> { { "hw_ruleset", "quick-v1" } },
                1));
            var joined = false;

            client.JoinLobby("109775240000000900", ok => joined = ok);
            Assert.That(joined, Is.False, "the join result must not arrive synchronously");

            client.PumpAll();

            Assert.That(joined, Is.True);
            var snapshot = client.GetLobby("109775240000000900");
            Assert.That(snapshot, Is.Not.Null);
            Assert.That(snapshot!.Members.Count, Is.EqualTo(2));
            Assert.That(snapshot.OwnerSteamId, Is.EqualTo(client.RemoteOwnerSteamId));
            CollectionAssert.Contains(MemberIds(snapshot), client.LocalSteamId);
            Assert.That(snapshot.Metadata["hw_ruleset"], Is.EqualTo("quick-v1"));
        }

        [Test]
        public void JoinLobby_UnknownLobby_ReportsFailure()
        {
            using var client = NewClient();
            var joined = true;

            client.JoinLobby("109775240000000999", ok => joined = ok);
            client.PumpAll();

            Assert.That(joined, Is.False);
            Assert.That(client.GetLobby("109775240000000999"), Is.Null);
        }

        [Test]
        public void SetLobbyData_ByNonOwner_ReturnsFalse()
        {
            using var client = NewClient();
            client.AvailableLobbies.Add(new SteamLobbySearchResult("109775240000000901", null, 1));
            client.JoinLobby("109775240000000901", _ => { });
            client.PumpAll();

            var accepted = client.SetLobbyData("109775240000000901", "hw_match", "abc");

            Assert.That(accepted, Is.False);
            Assert.That(client.GetLobby("109775240000000901")!.Metadata.ContainsKey("hw_match"), Is.False);
        }

        [Test]
        public void SetLobbyData_ByOwner_WritesMetadataAndRaisesChangeOnPump()
        {
            using var client = NewClient();
            var lobbyId = CreateOwnedLobby(client);
            var changed = new List<string>();
            client.LobbyDataChanged += id => changed.Add(id);

            var accepted = client.SetLobbyData(lobbyId, "hw_protocol", "2");

            Assert.That(accepted, Is.True);
            Assert.That(changed, Is.Empty, "the event must wait for Pump");

            client.PumpAll();

            Assert.That(changed, Is.EqualTo(new[] { lobbyId }));
            Assert.That(client.GetLobby(lobbyId)!.Metadata["hw_protocol"], Is.EqualTo("2"));
        }

        [Test]
        public void RemoteMemberEvents_AreDeliveredOneItemPerPump()
        {
            using var client = NewClient();
            var lobbyId = CreateOwnedLobby(client);
            var log = new List<string>();
            client.MemberJoined += (lobby, steamId) => log.Add("joined:" + steamId);
            client.MemberLeft += (lobby, steamId) => log.Add("left:" + steamId);

            client.AddRemoteMember(lobbyId, "76561197960287931", "Remote");
            client.RemoveRemoteMember(lobbyId, "76561197960287931");

            Assert.That(log, Is.Empty);

            client.Pump();
            Assert.That(log, Is.EqualTo(new[] { "joined:76561197960287931" }));

            client.Pump();
            Assert.That(log, Is.EqualTo(new[] { "joined:76561197960287931", "left:76561197960287931" }));
        }

        [Test]
        public void SetMemberData_SurfacesReadyFlagInSnapshot()
        {
            using var client = NewClient();
            var lobbyId = CreateOwnedLobby(client);
            client.AddRemoteMember(lobbyId, "76561197960287931", "Remote");
            client.SetMemberData(lobbyId, "hw_ready", "1");
            client.SetRemoteMemberData(lobbyId, "76561197960287931", "hw_ready", "0");
            client.PumpAll();

            var snapshot = client.GetLobby(lobbyId);

            Assert.That(snapshot, Is.Not.Null);
            Assert.That(MemberData(snapshot!, client.LocalSteamId)["hw_ready"], Is.EqualTo("1"));
            Assert.That(MemberData(snapshot!, "76561197960287931")["hw_ready"], Is.EqualTo("0"));
        }

        [Test]
        public void RequestAuthTicket_DeliversScriptedTicketOnPump()
        {
            using var client = NewClient();
            client.NextTicket = "DEADBEEF";
            string? ticket = "not-set";

            client.RequestAuthTicket(t => ticket = t);
            Assert.That(ticket, Is.EqualTo("not-set"));

            client.PumpAll();

            Assert.That(ticket, Is.EqualTo("DEADBEEF"));
            Assert.That(client.RequestAuthTicketCalls, Is.EqualTo(1));

            client.CancelAuthTicket();
            Assert.That(client.CancelAuthTicketCalls, Is.EqualTo(1));
        }

        [Test]
        public void RequestAuthTicket_IgnoresAResponseForAnAbandonedHandle()
        {
            using var client = NewClient();
            client.AutoDeliverAuthTickets = false;
            string? first = "not-set";
            string? second = "not-set";

            client.RequestAuthTicket(t => first = t);
            var stale = client.AuthTicketHandles[0];
            client.RequestAuthTicket(t => second = t);
            var current = client.AuthTicketHandles[1];

            Assert.That(client.CancelAuthTicketCalls, Is.EqualTo(1), "a new request releases the old handle");
            Assert.That(client.CurrentAuthTicketHandle, Is.EqualTo(current));

            client.DeliverAuthTicketResponse(stale, "STALE");

            Assert.That(client.StaleAuthTicketResponses, Is.EqualTo(1));
            Assert.That(first, Is.EqualTo("not-set"));
            Assert.That(second, Is.EqualTo("not-set"));

            client.DeliverAuthTicketResponse(current, "FRESH");

            Assert.That(second, Is.EqualTo("FRESH"));
        }

        [Test]
        public void SetLobbyData_CanBeScriptedToFail()
        {
            using var client = NewClient();
            string? lobbyId = null;
            client.CreateLobby(SteamLobbyVisibility.Public, 2, id => lobbyId = id);
            client.PumpAll();

            client.FailNextSetLobbyData = true;
            Assert.That(client.SetLobbyData(lobbyId!, "hw_app", "480"), Is.False);
            Assert.That(client.SetLobbyData(lobbyId!, "hw_app", "480"), Is.True, "the toggle is one shot");

            client.FailSetLobbyDataForKey = "hw_match";
            Assert.That(client.SetLobbyData(lobbyId!, "hw_match", "m-1"), Is.False);
            Assert.That(client.SetLobbyData(lobbyId!, "hw_match", "m-1"), Is.False, "that key stays refused");
            Assert.That(client.SetLobbyData(lobbyId!, "hw_name", "Local"), Is.True);
        }

        [Test]
        public void RequestAuthTicket_WhenScriptedNull_DeliversNull()
        {
            using var client = NewClient();
            client.NextTicket = null;
            string? ticket = "not-set";

            client.RequestAuthTicket(t => ticket = t);
            client.PumpAll();

            Assert.That(ticket, Is.Null);
        }

        [Test]
        public void RequestLobbyList_ReturnsOnlyLobbiesMatchingRequiredMetadata()
        {
            using var client = NewClient();
            client.AvailableLobbies.Add(new SteamLobbySearchResult(
                "109775240000000910",
                new Dictionary<string, string> { { "hw_app", "480" }, { "hw_ruleset", "quick-v1" } },
                1));
            client.AvailableLobbies.Add(new SteamLobbySearchResult(
                "109775240000000911",
                new Dictionary<string, string> { { "hw_app", "480" }, { "hw_ruleset", "custom" } },
                2));
            IReadOnlyList<SteamLobbySearchResult>? results = null;

            client.RequestLobbyList(
                new Dictionary<string, string> { { "hw_ruleset", "quick-v1" } },
                found => results = found);

            Assert.That(results, Is.Null, "search results must wait for Pump");

            client.PumpAll();

            Assert.That(results, Is.Not.Null);
            Assert.That(results!.Count, Is.EqualTo(1));
            Assert.That(results[0].LobbyId, Is.EqualTo("109775240000000910"));
            Assert.That(results[0].MemberCount, Is.EqualTo(1));
        }

        [Test]
        public void InviteAccepted_IsDeliveredOnPump()
        {
            using var client = NewClient();
            var accepted = new List<string>();
            client.InviteAccepted += id => accepted.Add(id);

            client.RaiseInviteAccepted("109775240000000920");
            Assert.That(accepted, Is.Empty);

            client.PumpAll();

            Assert.That(accepted, Is.EqualTo(new[] { "109775240000000920" }));
        }

        [Test]
        public void Dispose_ClearsEventHandlersAndPendingCallbacks()
        {
            var client = NewClient();
            var lobbyId = CreateOwnedLobby(client);
            client.LobbyDataChanged += _ => Assert.Fail("a disposed client must not raise events");
            client.MemberJoined += (_, _) => Assert.Fail("a disposed client must not raise events");

            client.AddRemoteMember(lobbyId, "76561197960287931", "Remote");
            Assert.That(client.PendingCallbackCount, Is.EqualTo(1));
            Assert.That(client.HasEventSubscribers, Is.True);

            client.Dispose();

            Assert.That(client.IsDisposed, Is.True);
            Assert.That(client.DisposeCalls, Is.EqualTo(1));
            Assert.That(client.PendingCallbackCount, Is.EqualTo(0));
            Assert.That(client.HasEventSubscribers, Is.False);

            client.PumpAll();
        }

        static string CreateOwnedLobby(FakeSteamLobbyClient client)
        {
            string? lobbyId = null;
            client.CreateLobby(SteamLobbyVisibility.FriendsOnly, 2, id => lobbyId = id);
            client.PumpAll();
            Assert.That(lobbyId, Is.Not.Null, "the fake failed to create a lobby");
            return lobbyId!;
        }

        static List<string> MemberIds(SteamLobbySnapshot snapshot)
        {
            var ids = new List<string>();
            foreach (var member in snapshot.Members) ids.Add(member.SteamId);
            return ids;
        }

        static IReadOnlyDictionary<string, string> MemberData(SteamLobbySnapshot snapshot, string steamId)
        {
            foreach (var member in snapshot.Members)
            {
                if (member.SteamId == steamId) return member.Data;
            }
            Assert.Fail("lobby snapshot has no member " + steamId);
            return new Dictionary<string, string>();
        }
    }
}
