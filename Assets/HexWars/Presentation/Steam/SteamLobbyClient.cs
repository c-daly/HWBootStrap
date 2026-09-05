#nullable enable
using System;
using System.Collections.Generic;
using System.Globalization;
#if HEXWARS_STEAM && !UNITY_WEBGL && !DISABLESTEAMWORKS
using Steamworks;
#endif

namespace HexWars.Presentation
{
#if HEXWARS_STEAM && !UNITY_WEBGL && !DISABLESTEAMWORKS
    /// <summary>
    /// Steamworks.NET implementation of <see cref="ISteamLobbyClient"/>. Deliberately free of
    /// UnityEngine: diagnostics go to the <c>log</c> delegate supplied by the caller, so this file
    /// stays testable by inspection and compiles on any Steam-capable platform.
    /// </summary>
    public sealed class SteamLobbyClient : ISteamLobbyClient
    {
        const string ReadyKey = "hw_ready";
        const string WebApiIdentity = "hexwars-match";
        const int MaxSearchResults = 20;

        readonly Action<string> _log;
        readonly Queue<Action> _deferred = new Queue<Action>();

        bool _initialised;
        bool _ownsSteamApi;
        bool _disposed;

        // One registry per call type, keyed by the Steam call handle. A CallResult can only track a
        // single call at a time, so the old one-field-per-type arrangement silently dropped the first
        // request whenever a second went out before it answered: the player cancels a join and starts
        // another, and the first lobby is entered with nobody left holding a reference to leave it.
        readonly Dictionary<SteamAPICall_t, CallResult<LobbyCreated_t>> _createCalls =
            new Dictionary<SteamAPICall_t, CallResult<LobbyCreated_t>>();
        readonly Dictionary<SteamAPICall_t, CallResult<LobbyMatchList_t>> _listCalls =
            new Dictionary<SteamAPICall_t, CallResult<LobbyMatchList_t>>();
        readonly Dictionary<SteamAPICall_t, CallResult<LobbyEnter_t>> _enterCalls =
            new Dictionary<SteamAPICall_t, CallResult<LobbyEnter_t>>();

        Callback<LobbyDataUpdate_t>? _lobbyDataUpdate;
        Callback<LobbyChatUpdate_t>? _lobbyChatUpdate;
        Callback<GameLobbyJoinRequested_t>? _gameLobbyJoinRequested;
        Callback<GetTicketForWebApiResponse_t>? _webApiTicketResponse;

        Action<string?>? _onAuthTicket;
        HAuthTicket _authTicket = HAuthTicket.Invalid;

        /// <param name="log">Diagnostic sink. Defaults to discarding messages.</param>
        /// <param name="initializeSteamApi">
        /// False when the host already ran <c>SteamAPI.Init()</c> (for example Steamworks.NET own
        /// SteamManager). Only an instance that performed the init calls <c>SteamAPI.Shutdown()</c>.
        /// </param>
        public SteamLobbyClient(Action<string>? log = null, bool initializeSteamApi = true)
        {
            _log = log ?? (_ => { });
            try
            {
                if (initializeSteamApi)
                {
                    _initialised = SteamAPI.Init();
                    _ownsSteamApi = _initialised;
                    if (!_initialised)
                    {
                        _log("SteamAPI.Init() returned false; Steam lobby features are unavailable.");
                        return;
                    }
                }
                else
                {
                    _initialised = true;
                }
                RegisterCallbacks();
            }
            catch (Exception ex)
            {
                _initialised = false;
                _ownsSteamApi = false;
                _log("Steam initialisation failed: " + ex.Message);
            }
        }

        public bool IsAvailable
        {
            get
            {
                if (!_initialised || _disposed) return false;
                try { return SteamUser.BLoggedOn(); }
                catch (Exception ex) { _log("SteamUser.BLoggedOn() failed: " + ex.Message); return false; }
            }
        }

        public string LocalSteamId
        {
            get
            {
                if (!_initialised || _disposed) return string.Empty;
                try { return SteamUser.GetSteamID().m_SteamID.ToString(CultureInfo.InvariantCulture); }
                catch (Exception ex) { _log("SteamUser.GetSteamID() failed: " + ex.Message); return string.Empty; }
            }
        }

        public string LocalDisplayName
        {
            get
            {
                if (!_initialised || _disposed) return string.Empty;
                try { return SteamFriends.GetPersonaName() ?? string.Empty; }
                catch (Exception ex) { _log("SteamFriends.GetPersonaName() failed: " + ex.Message); return string.Empty; }
            }
        }

        public uint AppId
        {
            get
            {
                if (!_initialised || _disposed) return 0u;
                try { return SteamUtils.GetAppID().m_AppId; }
                catch (Exception ex) { _log("SteamUtils.GetAppID() failed: " + ex.Message); return 0u; }
            }
        }

        public void CreateLobby(SteamLobbyVisibility visibility, int maxMembers, Action<string?> onDone)
        {
            if (!IsAvailable) { Defer(() => onDone(null)); return; }
            try
            {
                var call = SteamMatchmaking.CreateLobby(ToLobbyType(visibility), Math.Max(1, maxMembers));
                Register(_createCalls, call, (result, ioFailure) => OnLobbyCreated(result, ioFailure, onDone),
                         () => onDone(null));
            }
            catch (Exception ex)
            {
                _log("SteamMatchmaking.CreateLobby failed: " + ex.Message);
                Defer(() => onDone(null));
            }
        }

        public void RequestLobbyList(IReadOnlyDictionary<string, string> requiredMetadata, Action<IReadOnlyList<SteamLobbySearchResult>> onDone)
        {
            if (!IsAvailable) { Defer(() => onDone(Array.Empty<SteamLobbySearchResult>())); return; }
            try
            {
                if (requiredMetadata != null)
                {
                    foreach (var pair in requiredMetadata)
                    {
                        SteamMatchmaking.AddRequestLobbyListStringFilter(pair.Key, pair.Value, ELobbyComparison.k_ELobbyComparisonEqual);
                    }
                }
                SteamMatchmaking.AddRequestLobbyListResultCountFilter(MaxSearchResults);
                var call = SteamMatchmaking.RequestLobbyList();
                Register(_listCalls, call, (result, ioFailure) => OnLobbyMatchList(result, ioFailure, onDone),
                         () => onDone(Array.Empty<SteamLobbySearchResult>()));
            }
            catch (Exception ex)
            {
                _log("SteamMatchmaking.RequestLobbyList failed: " + ex.Message);
                Defer(() => onDone(Array.Empty<SteamLobbySearchResult>()));
            }
        }

        public void JoinLobby(string lobbyId, Action<bool> onDone)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) { Defer(() => onDone(false)); return; }
            try
            {
                var call = SteamMatchmaking.JoinLobby(lobby);
                Register(_enterCalls, call, (result, ioFailure) => OnLobbyEnter(result, ioFailure, onDone),
                         () => onDone(false));
            }
            catch (Exception ex)
            {
                _log("SteamMatchmaking.JoinLobby failed: " + ex.Message);
                Defer(() => onDone(false));
            }
        }

        public void LeaveLobby(string lobbyId)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) return;
            try { SteamMatchmaking.LeaveLobby(lobby); }
            catch (Exception ex) { _log("SteamMatchmaking.LeaveLobby failed: " + ex.Message); }
        }

        public bool SetLobbyData(string lobbyId, string key, string value)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) return false;
            try
            {
                if (SteamMatchmaking.GetLobbyOwner(lobby).m_SteamID != SteamUser.GetSteamID().m_SteamID) return false;
                return SteamMatchmaking.SetLobbyData(lobby, key, value ?? string.Empty);
            }
            catch (Exception ex)
            {
                _log("SteamMatchmaking.SetLobbyData failed: " + ex.Message);
                return false;
            }
        }

        public void SetMemberData(string lobbyId, string key, string value)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) return;
            try { SteamMatchmaking.SetLobbyMemberData(lobby, key, value ?? string.Empty); }
            catch (Exception ex) { _log("SteamMatchmaking.SetLobbyMemberData failed: " + ex.Message); }
        }

        public SteamLobbySnapshot? GetLobby(string lobbyId)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) return null;
            try
            {
                var owner = SteamMatchmaking.GetLobbyOwner(lobby);
                var count = SteamMatchmaking.GetNumLobbyMembers(lobby);
                var members = new List<SteamLobbyMemberSnapshot>(Math.Max(0, count));
                for (var i = 0; i < count; i++)
                {
                    var member = SteamMatchmaking.GetLobbyMemberByIndex(lobby, i);
                    var data = new Dictionary<string, string>(StringComparer.Ordinal);
                    var ready = SteamMatchmaking.GetLobbyMemberData(lobby, member, ReadyKey);
                    if (!string.IsNullOrEmpty(ready)) data[ReadyKey] = ready;
                    members.Add(new SteamLobbyMemberSnapshot(
                        member.m_SteamID.ToString(CultureInfo.InvariantCulture),
                        SteamFriends.GetFriendPersonaName(member),
                        data));
                }
                return new SteamLobbySnapshot(
                    lobby.m_SteamID.ToString(CultureInfo.InvariantCulture),
                    owner.m_SteamID.ToString(CultureInfo.InvariantCulture),
                    members,
                    ReadLobbyMetadata(lobby));
            }
            catch (Exception ex)
            {
                _log("Reading lobby " + lobbyId + " failed: " + ex.Message);
                return null;
            }
        }

        public void OpenInviteOverlay(string lobbyId)
        {
            if (!IsAvailable || !TryParseLobby(lobbyId, out var lobby)) return;
            try { SteamFriends.ActivateGameOverlayInviteDialog(lobby); }
            catch (Exception ex) { _log("SteamFriends.ActivateGameOverlayInviteDialog failed: " + ex.Message); }
        }

        public uint CurrentAuthTicketHandle { get { return _authTicket.m_HAuthTicket; } }

        public void RequestAuthTicket(Action<string?> onDone)
        {
            if (!IsAvailable) { Defer(() => onDone(null)); return; }

            // Only one ticket may be outstanding: the previous handle is released first, so a ticket
            // this client will never read cannot linger on the account.
            CancelAuthTicket();
            try
            {
                var handle = SteamUser.GetAuthTicketForWebApi(WebApiIdentity);
                if (handle == HAuthTicket.Invalid)
                {
                    _log("SteamUser.GetAuthTicketForWebApi returned an invalid handle.");
                    Defer(() => onDone(null));
                    return;
                }
                _authTicket = handle;
                _onAuthTicket = onDone;
            }
            catch (Exception ex)
            {
                _log("SteamUser.GetAuthTicketForWebApi failed: " + ex.Message);
                _authTicket = HAuthTicket.Invalid;
                _onAuthTicket = null;
                Defer(() => onDone(null));
            }
        }

        public void CancelAuthTicket()
        {
            _onAuthTicket = null;
            if (!_initialised || _authTicket == HAuthTicket.Invalid) return;
            try { SteamUser.CancelAuthTicket(_authTicket); }
            catch (Exception ex) { _log("SteamUser.CancelAuthTicket failed: " + ex.Message); }
            finally { _authTicket = HAuthTicket.Invalid; }
        }

        public event Action<string>? LobbyDataChanged;
        public event Action<string, string>? MemberJoined;
        public event Action<string, string>? MemberLeft;
        public event Action<string>? InviteAccepted;

        public void Pump()
        {
            if (_disposed) return;
            for (var i = 0; i < 64 && _deferred.Count > 0; i++)
            {
                var next = _deferred.Dequeue();
                try { next(); }
                catch (Exception ex) { _log("A deferred Steam callback threw: " + ex.Message); }
            }
            if (!_initialised) return;
            try { SteamAPI.RunCallbacks(); }
            catch (Exception ex) { _log("SteamAPI.RunCallbacks() threw: " + ex.Message); }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;

            CancelAuthTicket();

            DisposeAll(_createCalls);
            DisposeAll(_listCalls);
            DisposeAll(_enterCalls);
            Dispose(ref _lobbyDataUpdate);
            Dispose(ref _lobbyChatUpdate);
            Dispose(ref _gameLobbyJoinRequested);
            Dispose(ref _webApiTicketResponse);

            _deferred.Clear();
            _onAuthTicket = null;

            LobbyDataChanged = null;
            MemberJoined = null;
            MemberLeft = null;
            InviteAccepted = null;

            if (_ownsSteamApi)
            {
                try { SteamAPI.Shutdown(); }
                catch (Exception ex) { _log("SteamAPI.Shutdown() threw: " + ex.Message); }
            }
            _initialised = false;
            _ownsSteamApi = false;
        }

        // ----- callbacks ---------------------------------------------------------------------

        void RegisterCallbacks()
        {
            _lobbyDataUpdate = Callback<LobbyDataUpdate_t>.Create(OnLobbyDataUpdate);
            _lobbyChatUpdate = Callback<LobbyChatUpdate_t>.Create(OnLobbyChatUpdate);
            _gameLobbyJoinRequested = Callback<GameLobbyJoinRequested_t>.Create(OnGameLobbyJoinRequested);
            _webApiTicketResponse = Callback<GetTicketForWebApiResponse_t>.Create(OnWebApiTicket);
        }

        void OnLobbyCreated(LobbyCreated_t callback, bool ioFailure, Action<string?> done)
        {
            if (ioFailure || callback.m_eResult != EResult.k_EResultOK)
            {
                _log("CreateLobby failed (ioFailure=" + ioFailure + ", result=" + callback.m_eResult + ").");
                done(null);
                return;
            }
            done(callback.m_ulSteamIDLobby.ToString(CultureInfo.InvariantCulture));
        }

        void OnLobbyMatchList(LobbyMatchList_t callback, bool ioFailure,
                              Action<IReadOnlyList<SteamLobbySearchResult>> done)
        {
            if (ioFailure)
            {
                _log("RequestLobbyList failed with an IO failure.");
                done(Array.Empty<SteamLobbySearchResult>());
                return;
            }

            var results = new List<SteamLobbySearchResult>();
            try
            {
                var found = (int)Math.Min(callback.m_nLobbiesMatching, (uint)MaxSearchResults);
                for (var i = 0; i < found; i++)
                {
                    var lobby = SteamMatchmaking.GetLobbyByIndex(i);
                    results.Add(new SteamLobbySearchResult(
                        lobby.m_SteamID.ToString(CultureInfo.InvariantCulture),
                        ReadLobbyMetadata(lobby),
                        SteamMatchmaking.GetNumLobbyMembers(lobby)));
                }
            }
            catch (Exception ex)
            {
                _log("Reading lobby search results failed: " + ex.Message);
            }
            done(results);
        }

        void OnLobbyEnter(LobbyEnter_t callback, bool ioFailure, Action<bool> done)
        {
            var entered = !ioFailure
                && callback.m_EChatRoomEnterResponse == (uint)EChatRoomEnterResponse.k_EChatRoomEnterResponseSuccess;
            if (!entered)
            {
                _log("JoinLobby failed (ioFailure=" + ioFailure + ", response=" + callback.m_EChatRoomEnterResponse + ").");
            }
            done(entered);
        }

        void OnLobbyDataUpdate(LobbyDataUpdate_t callback)
        {
            LobbyDataChanged?.Invoke(callback.m_ulSteamIDLobby.ToString(CultureInfo.InvariantCulture));
        }

        void OnLobbyChatUpdate(LobbyChatUpdate_t callback)
        {
            var lobbyId = callback.m_ulSteamIDLobby.ToString(CultureInfo.InvariantCulture);
            var memberId = callback.m_ulSteamIDUserChanged.ToString(CultureInfo.InvariantCulture);
            var change = callback.m_rgfChatMemberStateChange;

            if ((change & (uint)EChatMemberStateChange.k_EChatMemberStateChangeEntered) != 0)
            {
                MemberJoined?.Invoke(lobbyId, memberId);
            }

            const uint gone = (uint)EChatMemberStateChange.k_EChatMemberStateChangeLeft
                | (uint)EChatMemberStateChange.k_EChatMemberStateChangeDisconnected
                | (uint)EChatMemberStateChange.k_EChatMemberStateChangeKicked
                | (uint)EChatMemberStateChange.k_EChatMemberStateChangeBanned;
            if ((change & gone) != 0)
            {
                MemberLeft?.Invoke(lobbyId, memberId);
            }
        }

        void OnGameLobbyJoinRequested(GameLobbyJoinRequested_t callback)
        {
            InviteAccepted?.Invoke(callback.m_steamIDLobby.m_SteamID.ToString(CultureInfo.InvariantCulture));
        }

        void OnWebApiTicket(GetTicketForWebApiResponse_t callback)
        {
            // Steam delivers one of these per issued handle. Anything but the handle this client is
            // waiting on belongs to a request that was cancelled or superseded, and must be dropped
            // instead of being handed to whoever asked most recently.
            if (callback.m_hAuthTicket != _authTicket)
            {
                _log("Ignoring an auth ticket response for a handle we are no longer waiting on.");
                return;
            }

            var done = _onAuthTicket;
            _onAuthTicket = null;
            if (done == null) return;

            if (callback.m_eResult != EResult.k_EResultOK || callback.m_cubTicket <= 0 || callback.m_rgubTicket == null)
            {
                _log("GetAuthTicketForWebApi failed (result=" + callback.m_eResult + ").");
                done(null);
                return;
            }
            done(ToHex(callback.m_rgubTicket, callback.m_cubTicket));
        }

        // ----- helpers -----------------------------------------------------------------------

        void Defer(Action action)
        {
            if (_disposed) return;
            _deferred.Enqueue(action);
        }

        /// <summary>
        /// Tracks one outstanding Steam call. Each call gets its OWN CallResult, kept alive by the
        /// registry until its result arrives, so a second request never unregisters the first: both
        /// answer, in the order Steam delivers them.
        /// </summary>
        void Register<T>(Dictionary<SteamAPICall_t, CallResult<T>> registry, SteamAPICall_t call,
                         Action<T, bool> handler, Action onInvalid)
        {
            if (call == SteamAPICall_t.Invalid)
            {
                // Steam refused to issue the call. A CallResult registered for handle 0 waits for a
                // result that can never arrive, and the next invalid call overwrites that entry
                // without disposing it, so the caller is answered with a failure instead.
                _log("A Steam call could not be issued; answering the caller with a failure.");
                Defer(onInvalid);
                return;
            }

            var result = CallResult<T>.Create();
            registry[call] = result;
            result.Set(call, (payload, ioFailure) =>
            {
                // The CallResult is retired on the next Pump rather than from inside its own dispatch,
                // and it stays in the registry until then so Dispose still reaches it either way.
                Defer(() =>
                {
                    CallResult<T>? tracked;
                    if (!registry.TryGetValue(call, out tracked) || !ReferenceEquals(tracked, result)) return;
                    registry.Remove(call);
                    result.Dispose();
                });
                handler(payload, ioFailure);
            });
        }

        static void DisposeAll<T>(Dictionary<SteamAPICall_t, CallResult<T>> registry)
        {
            foreach (var pending in registry.Values) pending.Dispose();
            registry.Clear();
        }

        static void Dispose<T>(ref Callback<T>? callback)
        {
            if (callback == null) return;
            callback.Dispose();
            callback = null;
        }

        static ELobbyType ToLobbyType(SteamLobbyVisibility visibility)
        {
            switch (visibility)
            {
                case SteamLobbyVisibility.Private: return ELobbyType.k_ELobbyTypePrivate;
                case SteamLobbyVisibility.FriendsOnly: return ELobbyType.k_ELobbyTypeFriendsOnly;
                default: return ELobbyType.k_ELobbyTypePublic;
            }
        }

        static bool TryParseLobby(string lobbyId, out CSteamID lobby)
        {
            lobby = default;
            if (string.IsNullOrEmpty(lobbyId)) return false;
            if (!ulong.TryParse(lobbyId, NumberStyles.None, CultureInfo.InvariantCulture, out var raw)) return false;
            lobby = new CSteamID(raw);
            return true;
        }

        static Dictionary<string, string> ReadLobbyMetadata(CSteamID lobby)
        {
            var metadata = new Dictionary<string, string>(StringComparer.Ordinal);
            var count = SteamMatchmaking.GetLobbyDataCount(lobby);
            for (var i = 0; i < count; i++)
            {
                if (!SteamMatchmaking.GetLobbyDataByIndex(
                        lobby, i,
                        out var key, Constants.k_nMaxLobbyKeyLength + 1,
                        out var value, Constants.k_cubChatMetadataMax))
                {
                    continue;
                }
                if (!string.IsNullOrEmpty(key)) metadata[key] = value ?? string.Empty;
            }
            return metadata;
        }

        static string ToHex(byte[] bytes, int length)
        {
            const string digits = "0123456789ABCDEF";
            var take = Math.Min(length, bytes.Length);
            var chars = new char[take * 2];
            for (var i = 0; i < take; i++)
            {
                chars[i * 2] = digits[bytes[i] >> 4];
                chars[(i * 2) + 1] = digits[bytes[i] & 0x0F];
            }
            return new string(chars);
        }
    }
#else
    /// <summary>
    /// Inert stand-in used when the Steamworks.NET package is absent, on WebGL, or when
    /// DISABLESTEAMWORKS is defined. It is never available, and every request completes with a
    /// failure value on the next <see cref="Pump"/>, so callers take exactly the same code path
    /// they would on a real Steam failure.
    /// </summary>
    public sealed class SteamLobbyClient : ISteamLobbyClient
    {
        readonly Action<string> _log;
        readonly Queue<Action> _deferred = new Queue<Action>();
        bool _disposed;

        public SteamLobbyClient(Action<string>? log = null, bool initializeSteamApi = true)
        {
            _log = log ?? (_ => { });
            _log("Steamworks is not compiled into this build; the lobby client is inert.");
        }

        public bool IsAvailable { get { return false; } }
        public string LocalSteamId { get { return string.Empty; } }
        public string LocalDisplayName { get { return string.Empty; } }
        public uint AppId { get { return 0u; } }

        public void CreateLobby(SteamLobbyVisibility visibility, int maxMembers, Action<string?> onDone) { Defer(() => onDone(null)); }

        public void RequestLobbyList(IReadOnlyDictionary<string, string> requiredMetadata, Action<IReadOnlyList<SteamLobbySearchResult>> onDone)
        {
            Defer(() => onDone(Array.Empty<SteamLobbySearchResult>()));
        }

        public void JoinLobby(string lobbyId, Action<bool> onDone) { Defer(() => onDone(false)); }
        public void LeaveLobby(string lobbyId) { }
        public bool SetLobbyData(string lobbyId, string key, string value) { return false; }
        public void SetMemberData(string lobbyId, string key, string value) { }
        public SteamLobbySnapshot? GetLobby(string lobbyId) { return null; }
        public void OpenInviteOverlay(string lobbyId) { }
        public void RequestAuthTicket(Action<string?> onDone) { Defer(() => onDone(null)); }
        public void CancelAuthTicket() { }
        public uint CurrentAuthTicketHandle { get { return 0u; } }

#pragma warning disable 0067 // the stub never raises these; the interface still has to expose them
        public event Action<string>? LobbyDataChanged;
        public event Action<string, string>? MemberJoined;
        public event Action<string, string>? MemberLeft;
        public event Action<string>? InviteAccepted;
#pragma warning restore 0067

        public void Pump()
        {
            if (_disposed) return;
            for (var i = 0; i < 64 && _deferred.Count > 0; i++)
            {
                var next = _deferred.Dequeue();
                try { next(); }
                catch (Exception ex) { _log("A deferred callback threw: " + ex.Message); }
            }
        }

        public void Dispose()
        {
            _disposed = true;
            _deferred.Clear();
            LobbyDataChanged = null;
            MemberJoined = null;
            MemberLeft = null;
            InviteAccepted = null;
        }

        void Defer(Action action)
        {
            if (_disposed) return;
            _deferred.Enqueue(action);
        }
    }
#endif
}
