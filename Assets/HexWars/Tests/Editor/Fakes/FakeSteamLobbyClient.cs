#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// Deterministic in-memory <see cref="ISteamLobbyClient"/> for tests. Nothing is delivered
    /// synchronously: every callback and every event is queued, and <see cref="Pump"/> releases
    /// exactly one queued item per call, so a test controls the interleaving precisely.
    /// <see cref="PumpAll"/> drains the queue.
    /// </summary>
    public sealed class FakeSteamLobbyClient : ISteamLobbyClient
    {
        /// <summary>The only member-data key Steam lets HexWars read back by name.</summary>
        public const string ReadyKey = "hw_ready";

        sealed class FakeLobby
        {
            public string LobbyId = string.Empty;
            public string OwnerSteamId = string.Empty;
            public SteamLobbyVisibility Visibility = SteamLobbyVisibility.Public;
            public int MaxMembers = 2;
            public readonly List<string> Members = new List<string>();
            public readonly Dictionary<string, string> Names = new Dictionary<string, string>(StringComparer.Ordinal);
            public readonly Dictionary<string, string> Data = new Dictionary<string, string>(StringComparer.Ordinal);
            public readonly Dictionary<string, Dictionary<string, string>> MemberData =
                new Dictionary<string, Dictionary<string, string>>(StringComparer.Ordinal);
        }

        readonly Dictionary<string, FakeLobby> _lobbies = new Dictionary<string, FakeLobby>(StringComparer.Ordinal);
        readonly Queue<Action> _pending = new Queue<Action>();
        ulong _nextLobbyId = 109775240000000001UL;
        bool _disposed;

        uint _nextAuthTicketHandle = 1;
        uint _currentAuthTicketHandle;
        Action<string?>? _authTicketCallback;

        public bool IsAvailable { get; set; } = true;
        public string LocalSteamId { get; set; } = "76561197960287930";
        public string LocalDisplayName { get; set; } = "LocalPlayer";
        public uint AppId { get; set; } = 480;

        /// <summary>Owner given to lobbies materialised out of <see cref="AvailableLobbies"/>.</summary>
        public string RemoteOwnerSteamId { get; set; } = "76561197960287931";

        public string RemoteOwnerDisplayName { get; set; } = "RemoteOwner";

        /// <summary>Ticket handed to the next RequestAuthTicket caller. Null scripts a failure.</summary>
        public string? NextTicket { get; set; } = "0A1B2C3D";

        /// <summary>
        /// When false, a test delivers ticket responses itself through
        /// <see cref="DeliverAuthTicketResponse"/>, which is how a stale one is staged.
        /// </summary>
        public bool AutoDeliverAuthTickets { get; set; } = true;

        /// <summary>Every handle RequestAuthTicket issued, in order.</summary>
        public List<uint> AuthTicketHandles { get; } = new List<uint>();

        /// <summary>Responses dropped because their handle was no longer the current one.</summary>
        public int StaleAuthTicketResponses { get; private set; }

        /// <summary>The next write of this key fails, whatever it is. Cleared once it has fired.</summary>
        public bool FailNextSetLobbyData { get; set; }

        /// <summary>
        /// The next CreateLobby fails outright, the way a call Steam could not even issue does:
        /// the caller is answered with a failure on the next Pump. Cleared once it has fired.
        /// </summary>
        public bool FailNextCreateLobby { get; set; }

        /// <summary>The same for JoinLobby. Cleared once it has fired.</summary>
        public bool FailNextJoinLobby { get; set; }

        /// <summary>Every write of this key fails. Null means no key is refused.</summary>
        public string? FailSetLobbyDataForKey { get; set; }

        /// <summary>Lobbies RequestLobbyList searches, and that JoinLobby can materialise.</summary>
        public List<SteamLobbySearchResult> AvailableLobbies { get; } = new List<SteamLobbySearchResult>();

        public string? LastInviteOverlayLobbyId { get; private set; }

        /// <summary>Visibility passed to the most recent CreateLobby call, or null when there was none.</summary>
        public SteamLobbyVisibility? LastCreateVisibility { get; private set; }

        /// <summary>Member cap passed to the most recent CreateLobby call.</summary>
        public int LastCreateMaxMembers { get; private set; }

        public int CreateLobbyCalls { get; private set; }
        public int RequestLobbyListCalls { get; private set; }
        public int JoinLobbyCalls { get; private set; }
        public int LeaveLobbyCalls { get; private set; }
        public int SetLobbyDataCalls { get; private set; }
        public int SetMemberDataCalls { get; private set; }
        public int GetLobbyCalls { get; private set; }
        public int OpenInviteOverlayCalls { get; private set; }
        public int RequestAuthTicketCalls { get; private set; }
        public int CancelAuthTicketCalls { get; private set; }
        public int PumpCalls { get; private set; }
        public int DisposeCalls { get; private set; }

        public int PendingCallbackCount { get { return _pending.Count; } }

        public bool IsDisposed { get { return _disposed; } }

        public bool HasEventSubscribers
        {
            get { return LobbyDataChanged != null || MemberJoined != null || MemberLeft != null || InviteAccepted != null; }
        }

        public event Action<string>? LobbyDataChanged;
        public event Action<string, string>? MemberJoined;
        public event Action<string, string>? MemberLeft;
        public event Action<string>? InviteAccepted;

        // ----- ISteamLobbyClient -------------------------------------------------------------

        public void CreateLobby(SteamLobbyVisibility visibility, int maxMembers, Action<string?> onDone)
        {
            CreateLobbyCalls++;
            LastCreateVisibility = visibility;
            LastCreateMaxMembers = maxMembers;
            if (FailNextCreateLobby)
            {
                FailNextCreateLobby = false;
                Enqueue(() => onDone(null));
                return;
            }
            if (!IsAvailable || _disposed)
            {
                Enqueue(() => onDone(null));
                return;
            }

            var lobby = new FakeLobby
            {
                LobbyId = NextLobbyId(),
                OwnerSteamId = LocalSteamId,
                Visibility = visibility,
                MaxMembers = Math.Max(1, maxMembers),
            };
            lobby.Members.Add(LocalSteamId);
            lobby.Names[LocalSteamId] = LocalDisplayName;
            _lobbies[lobby.LobbyId] = lobby;
            Enqueue(() => onDone(lobby.LobbyId));
        }

        public void RequestLobbyList(IReadOnlyDictionary<string, string> requiredMetadata, Action<IReadOnlyList<SteamLobbySearchResult>> onDone)
        {
            RequestLobbyListCalls++;
            if (!IsAvailable || _disposed)
            {
                Enqueue(() => onDone(Array.Empty<SteamLobbySearchResult>()));
                return;
            }

            var matches = new List<SteamLobbySearchResult>();
            foreach (var candidate in AvailableLobbies)
            {
                if (MatchesAll(candidate, requiredMetadata)) matches.Add(candidate);
            }
            Enqueue(() => onDone(matches));
        }

        public void JoinLobby(string lobbyId, Action<bool> onDone)
        {
            JoinLobbyCalls++;
            if (FailNextJoinLobby)
            {
                FailNextJoinLobby = false;
                Enqueue(() => onDone(false));
                return;
            }
            var lobby = string.IsNullOrEmpty(lobbyId) ? null : Resolve(lobbyId);
            if (!IsAvailable || _disposed || lobby == null)
            {
                Enqueue(() => onDone(false));
                return;
            }

            if (!lobby.Members.Contains(LocalSteamId))
            {
                if (lobby.Members.Count >= lobby.MaxMembers)
                {
                    Enqueue(() => onDone(false));
                    return;
                }
                lobby.Members.Add(LocalSteamId);
                lobby.Names[LocalSteamId] = LocalDisplayName;
            }
            Enqueue(() => onDone(true));
        }

        public void LeaveLobby(string lobbyId)
        {
            LeaveLobbyCalls++;
            if (string.IsNullOrEmpty(lobbyId) || !_lobbies.TryGetValue(lobbyId, out var lobby)) return;
            lobby.Members.Remove(LocalSteamId);
            lobby.MemberData.Remove(LocalSteamId);
        }

        public bool SetLobbyData(string lobbyId, string key, string value)
        {
            SetLobbyDataCalls++;
            if (FailNextSetLobbyData) { FailNextSetLobbyData = false; return false; }
            if (FailSetLobbyDataForKey != null
                && string.Equals(key, FailSetLobbyDataForKey, StringComparison.Ordinal)) return false;
            if (!IsAvailable || _disposed) return false;
            if (string.IsNullOrEmpty(lobbyId) || !_lobbies.TryGetValue(lobbyId, out var lobby)) return false;
            if (!string.Equals(lobby.OwnerSteamId, LocalSteamId, StringComparison.Ordinal)) return false;

            lobby.Data[key] = value ?? string.Empty;
            RaiseLobbyDataChanged(lobby.LobbyId);
            return true;
        }

        public void SetMemberData(string lobbyId, string key, string value)
        {
            SetMemberDataCalls++;
            if (!IsAvailable || _disposed) return;
            if (string.IsNullOrEmpty(lobbyId) || !_lobbies.TryGetValue(lobbyId, out var lobby)) return;

            MemberDataFor(lobby, LocalSteamId)[key] = value ?? string.Empty;
            RaiseLobbyDataChanged(lobby.LobbyId);
        }

        public SteamLobbySnapshot? GetLobby(string lobbyId)
        {
            GetLobbyCalls++;
            if (string.IsNullOrEmpty(lobbyId) || !_lobbies.TryGetValue(lobbyId, out var lobby)) return null;

            var members = new List<SteamLobbyMemberSnapshot>(lobby.Members.Count);
            foreach (var steamId in lobby.Members)
            {
                var data = new Dictionary<string, string>(StringComparer.Ordinal);
                if (lobby.MemberData.TryGetValue(steamId, out var stored) && stored.TryGetValue(ReadyKey, out var ready))
                {
                    data[ReadyKey] = ready;
                }
                lobby.Names.TryGetValue(steamId, out var name);
                members.Add(new SteamLobbyMemberSnapshot(steamId, name, data));
            }

            var metadata = new Dictionary<string, string>(lobby.Data, StringComparer.Ordinal);
            return new SteamLobbySnapshot(lobby.LobbyId, lobby.OwnerSteamId, members, metadata);
        }

        public void OpenInviteOverlay(string lobbyId)
        {
            OpenInviteOverlayCalls++;
            LastInviteOverlayLobbyId = lobbyId;
        }

        public uint CurrentAuthTicketHandle { get { return _currentAuthTicketHandle; } }

        public void RequestAuthTicket(Action<string?> onDone)
        {
            RequestAuthTicketCalls++;
            if (!IsAvailable || _disposed)
            {
                Enqueue(() => onDone(null));
                return;
            }

            // A second request cancels the first handle, exactly as the live client does, so only one
            // ticket is ever outstanding.
            if (_currentAuthTicketHandle != 0) CancelAuthTicket();

            var handle = _nextAuthTicketHandle++;
            _currentAuthTicketHandle = handle;
            _authTicketCallback = onDone;
            AuthTicketHandles.Add(handle);

            var ticket = NextTicket;
            if (AutoDeliverAuthTickets) Enqueue(() => DeliverAuthTicketResponse(handle, ticket));
        }

        /// <summary>
        /// Steam answering for one handle. A response for anything but the current handle is dropped,
        /// which is the whole point of correlating them: a late answer to an abandoned request must
        /// never be handed to a caller that has moved on.
        /// </summary>
        public void DeliverAuthTicketResponse(uint handle, string? ticket)
        {
            if (handle != _currentAuthTicketHandle) { StaleAuthTicketResponses++; return; }
            var done = _authTicketCallback;
            _authTicketCallback = null;
            if (done != null) done(ticket);
        }

        public void CancelAuthTicket()
        {
            CancelAuthTicketCalls++;
            _currentAuthTicketHandle = 0;
            _authTicketCallback = null;
        }

        /// <summary>Releases at most one queued callback, mirroring a single frame of SteamAPI.RunCallbacks.</summary>
        public void Pump()
        {
            PumpCalls++;
            if (_pending.Count == 0) return;
            _pending.Dequeue()();
        }

        public void Dispose()
        {
            DisposeCalls++;
            _disposed = true;
            _currentAuthTicketHandle = 0;
            _authTicketCallback = null;
            _pending.Clear();
            LobbyDataChanged = null;
            MemberJoined = null;
            MemberLeft = null;
            InviteAccepted = null;
        }

        // ----- test driver -------------------------------------------------------------------

        /// <summary>Drains the queue. Bounded so a callback that enqueues more work cannot hang a test.</summary>
        /// <summary>
        /// Releases one queued callback out of turn. Steam answers call results in whatever order
        /// they complete, so a request issued later can land before an earlier one that is still
        /// outstanding; FIFO delivery alone cannot stage that.
        /// </summary>
        public void PumpAt(int index)
        {
            PumpCalls++;
            if (index < 0 || index >= _pending.Count) return;

            var kept = new List<Action>(_pending.Count);
            Action? chosen = null;
            var i = 0;
            foreach (var pending in _pending)
            {
                if (i == index) chosen = pending; else kept.Add(pending);
                i++;
            }
            _pending.Clear();
            foreach (var keep in kept) _pending.Enqueue(keep);
            if (chosen != null) chosen();
        }

        public void PumpAll()
        {
            for (var i = 0; i < 1024 && _pending.Count > 0; i++) Pump();
        }

        public void AddRemoteMember(string lobbyId, string steamId, string displayName)
        {
            var lobby = Require(lobbyId);
            if (!lobby.Members.Contains(steamId)) lobby.Members.Add(steamId);
            lobby.Names[steamId] = displayName ?? string.Empty;
            Enqueue(() => MemberJoined?.Invoke(lobby.LobbyId, steamId));
        }

        public void RemoveRemoteMember(string lobbyId, string steamId)
        {
            var lobby = Require(lobbyId);
            lobby.Members.Remove(steamId);
            lobby.MemberData.Remove(steamId);
            Enqueue(() => MemberLeft?.Invoke(lobby.LobbyId, steamId));
        }

        public void SetRemoteMemberData(string lobbyId, string steamId, string key, string value)
        {
            var lobby = Require(lobbyId);
            MemberDataFor(lobby, steamId)[key] = value ?? string.Empty;
            RaiseLobbyDataChanged(lobby.LobbyId);
        }

        public void SetRemoteLobbyData(string lobbyId, string key, string value)
        {
            var lobby = Require(lobbyId);
            lobby.Data[key] = value ?? string.Empty;
            RaiseLobbyDataChanged(lobby.LobbyId);
        }

        public void SetLobbyOwner(string lobbyId, string steamId)
        {
            var lobby = Require(lobbyId);
            lobby.OwnerSteamId = steamId;
            if (!lobby.Members.Contains(steamId)) lobby.Members.Insert(0, steamId);
        }

        public void RaiseInviteAccepted(string lobbyId)
        {
            Enqueue(() => InviteAccepted?.Invoke(lobbyId));
        }

        // ----- internals ---------------------------------------------------------------------

        void Enqueue(Action action)
        {
            if (_disposed) return;
            _pending.Enqueue(action);
        }

        void RaiseLobbyDataChanged(string lobbyId)
        {
            Enqueue(() => LobbyDataChanged?.Invoke(lobbyId));
        }

        string NextLobbyId()
        {
            return (_nextLobbyId++).ToString(CultureInfo.InvariantCulture);
        }

        static Dictionary<string, string> MemberDataFor(FakeLobby lobby, string steamId)
        {
            if (!lobby.MemberData.TryGetValue(steamId, out var data))
            {
                data = new Dictionary<string, string>(StringComparer.Ordinal);
                lobby.MemberData[steamId] = data;
            }
            return data;
        }

        static bool MatchesAll(SteamLobbySearchResult lobby, IReadOnlyDictionary<string, string>? required)
        {
            if (required == null) return true;
            foreach (var pair in required)
            {
                if (!lobby.Metadata.TryGetValue(pair.Key, out var value)) return false;
                if (!string.Equals(value, pair.Value, StringComparison.Ordinal)) return false;
            }
            return true;
        }

        /// <summary>Returns a known lobby, materialising one from <see cref="AvailableLobbies"/> if needed.</summary>
        FakeLobby? Resolve(string lobbyId)
        {
            if (_lobbies.TryGetValue(lobbyId, out var known)) return known;

            foreach (var candidate in AvailableLobbies)
            {
                if (!string.Equals(candidate.LobbyId, lobbyId, StringComparison.Ordinal)) continue;

                var lobby = new FakeLobby
                {
                    LobbyId = lobbyId,
                    OwnerSteamId = RemoteOwnerSteamId,
                    MaxMembers = Math.Max(2, candidate.MemberCount + 1),
                };
                lobby.Members.Add(RemoteOwnerSteamId);
                lobby.Names[RemoteOwnerSteamId] = RemoteOwnerDisplayName;
                foreach (var pair in candidate.Metadata) lobby.Data[pair.Key] = pair.Value;
                _lobbies[lobbyId] = lobby;
                return lobby;
            }
            return null;
        }

        FakeLobby Require(string lobbyId)
        {
            var lobby = string.IsNullOrEmpty(lobbyId) ? null : Resolve(lobbyId);
            if (lobby == null) throw new InvalidOperationException("FakeSteamLobbyClient has no lobby " + lobbyId);
            return lobby;
        }
    }
}
