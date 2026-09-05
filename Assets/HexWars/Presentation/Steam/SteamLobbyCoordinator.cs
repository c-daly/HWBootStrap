#nullable enable
using System;
using System.Collections.Generic;
using HexWars.Engine;

namespace HexWars.Presentation
{
    /// <summary>Tunables for <see cref="SteamLobbyCoordinator"/>.</summary>
    public sealed class SteamLobbyConfig
    {
        /// <summary>The Steam App ID advertised in <see cref="SteamLobbyKeys.App"/>.</summary>
        public uint AppId { get; set; }

        /// <summary>The match protocol version advertised in <see cref="SteamLobbyKeys.Protocol"/>.</summary>
        public int ProtocolVersion { get; set; } = 2;

        /// <summary>This client build, usually <c>Application.version</c>.</summary>
        public string ClientBuild { get; set; } = string.Empty;

        /// <summary>Rolls the board seed for a Quick Match lobby. Defaults to a fixed in-range seed.</summary>
        public Func<int>? RollSeed { get; set; }

        /// <summary>How long a lobby search runs before this client hosts instead.</summary>
        public double SearchTimeoutSeconds { get; set; } = 8;

        /// <summary>How long a match-service allocation or join may take before it is abandoned.</summary>
        public double AllocationTimeoutSeconds { get; set; } = 15;

        /// <summary>A rolled seed, clamped into the advertised range.</summary>
        public int NextSeed()
        {
            var roll = RollSeed;
            return SteamLobbyRules.ClampSeed(roll == null ? SteamLobbyRules.MinSeed : roll());
        }
    }

    /// <summary>
    /// Drives the whole Steam matchmaking flow: find or host a lobby, agree readiness, have the owner
    /// allocate a server match, and hand both players a <see cref="SteamMatchTicket"/>.
    /// <para>
    /// Everything here is pure C#: it talks to Steam through <see cref="ISteamLobbyClient"/> and to the
    /// match service through <see cref="ISteamMatchApi"/>, so the entire state machine is unit testable.
    /// Call <see cref="Tick"/> once per frame, after pumping the Steam client, so timeouts fire.
    /// </para>
    /// <para>
    /// Every asynchronous callback captures the generation counter that was current when it was issued.
    /// Cancelling, retrying or starting a new operation bumps that counter, so a result belonging to an
    /// abandoned attempt is dropped instead of moving the flow somewhere the player did not ask for.
    /// </para>
    /// </summary>
    public sealed class SteamLobbyCoordinator : IDisposable
    {
        enum Operation
        {
            None,
            QuickMatch,
            Host,
            Invite,
            JoinInvited,
            Reconnect,
        }

        readonly ISteamLobbyClient _steam;
        readonly ISteamMatchApi _api;
        readonly SteamLobbyConfig _config;
        readonly Action<SteamLobbyStatus>? _onStatus;
        readonly Action<SteamMatchTicket>? _onMatchReady;

        int _generation;
        double _now;
        // Deadlines are relative: the duration is stored when the operation starts and anchored on the
        // first Tick that observes it. An absolute deadline built from an unprimed clock (which reads
        // zero until the first Tick) fires the instant the real game clock arrives.
        double _deadlineSeconds;
        double? _deadlineAnchor;
        bool _hasDeadline;
        bool _disposed;
        bool _detached;
        bool _idleAfterCancel;

        // The hw_match write is what tells the guest a match exists. A refused write is retried on the
        // next two Ticks before the allocation is abandoned.
        string? _pendingMatchKey;
        SteamMatchApiResult? _pendingMatchResult;
        int _matchKeyWritesLeft;

        Operation _operation = Operation.None;
        GameSetup _lastHostSetup = GameSetup.Default;
        SteamLobbyVisibility _lastHostVisibility = SteamLobbyVisibility.Public;
        string? _lastInvitedLobbyId;
        string? _lastReconnectMatchId;

        string _pendingRuleset = SteamLobbyRules.QuickRuleset;
        string _pendingSetupWire = string.Empty;

        SteamLobbyPhase _phase = SteamLobbyPhase.Idle;
        string _message = SteamLobbyMessages.Idle;
        string? _lobbyId;
        string? _matchId;
        bool _isOwner;
        bool _localReady;
        bool _remoteReady;
        string? _opponentName;

        string? _joinRequestedMatchId;
        bool _allocationStarted;
        bool _joinInFlight;
        string? _joiningLobbyId;

        SteamLobbyStatus _status;

        public SteamLobbyCoordinator(
            ISteamLobbyClient steam,
            ISteamMatchApi api,
            SteamLobbyConfig config,
            Action<SteamLobbyStatus>? onStatus,
            Action<SteamMatchTicket>? onMatchReady)
        {
            if (steam == null) throw new ArgumentNullException(nameof(steam));
            if (api == null) throw new ArgumentNullException(nameof(api));
            if (config == null) throw new ArgumentNullException(nameof(config));

            _steam = steam;
            _api = api;
            _config = config;
            _onStatus = onStatus;
            _onMatchReady = onMatchReady;
            _status = BuildStatus();

            _steam.LobbyDataChanged += OnLobbyDataChanged;
            _steam.MemberJoined += OnMemberJoined;
            _steam.MemberLeft += OnMemberLeft;
            _steam.InviteAccepted += OnInviteAccepted;
        }

        /// <summary>The current immutable snapshot. A new instance is published on every change.</summary>
        public SteamLobbyStatus Status { get { return _status; } }

        // ----- operations ----------------------------------------------------------------------

        /// <summary>Finds an open <c>quick-v1</c> lobby, or hosts one when there is none.</summary>
        public void QuickMatch()
        {
            if (!EnsureSteam()) return;

            BeginOperation(Operation.QuickMatch);
            _pendingRuleset = SteamLobbyRules.QuickRuleset;
            _pendingSetupWire = SteamLobbyRules.QuickMatchSetup(_config.NextSeed()).ToWire();

            var generation = _generation;
            SetPhase(SteamLobbyPhase.Searching);
            SetDeadline(_config.SearchTimeoutSeconds);
            Publish();

            var required = SteamLobbyRules.RequiredSearchMetadata(
                _config.AppId, _config.ProtocolVersion, SteamLobbyRules.QuickRuleset);
            _steam.RequestLobbyList(required, results => OnLobbyList(generation, results));
        }

        /// <summary>Hosts a lobby with a host-configured setup.</summary>
        public void HostGame(GameSetup setup, SteamLobbyVisibility visibility)
        {
            if (!EnsureSteam()) return;

            BeginOperation(Operation.Host);
            _lastHostSetup = setup;
            _lastHostVisibility = visibility;
            _pendingRuleset = SteamLobbyRules.CustomRuleset;
            _pendingSetupWire = setup.Sanitized().ToWire();
            CreateLobbyForCurrentOperation();
        }

        /// <summary>Hosts a friends-only Quick Match lobby and opens the Steam invite overlay.</summary>
        public void InviteFriend()
        {
            if (!EnsureSteam()) return;

            BeginOperation(Operation.Invite);
            _pendingRuleset = SteamLobbyRules.QuickRuleset;
            _pendingSetupWire = SteamLobbyRules.QuickMatchSetup(_config.NextSeed()).ToWire();
            CreateLobbyForCurrentOperation();
        }

        /// <summary>Joins a lobby the player was invited to.</summary>
        public void JoinInvited(string lobbyId)
        {
            if (!EnsureSteam()) return;
            if (string.IsNullOrEmpty(lobbyId)) return;

            BeginOperation(Operation.JoinInvited);
            _lastInvitedLobbyId = lobbyId;

            var generation = _generation;
            SetPhase(SteamLobbyPhase.Searching);
            SetDeadline(_config.SearchTimeoutSeconds);
            _joinInFlight = true;
            _joiningLobbyId = lobbyId;
            Publish();
            _steam.JoinLobby(lobbyId, ok => OnLobbyJoined(generation, lobbyId, ok));
        }

        /// <summary>Publishes the local readiness flag into member data.</summary>
        public void SetReady(bool ready)
        {
            if (!EnsureSteam()) return;
            if (string.IsNullOrEmpty(_lobbyId)) return;

            _localReady = ready;
            Publish();
            _steam.SetMemberData(_lobbyId!, SteamLobbyKeys.MemberReady,
                ready ? SteamLobbyKeys.ReadyTrue : SteamLobbyKeys.ReadyFalse);
        }

        /// <summary>
        /// Tries the failed step again. While the lobby is still held this re-runs the allocation or
        /// join without leaving it; otherwise it repeats the operation that started the flow.
        /// </summary>
        public void Retry()
        {
            if (!EnsureSteam()) return;

            if (!string.IsNullOrEmpty(_lobbyId) && IsRetryableInLobby(_phase))
            {
                // A retry starts a fresh exchange over an abandoned one, so the old request, its
                // ticket and any half-published match go first. The lobby itself is kept.
                _api.Cancel();
                ReleaseTicket();
                ClearPendingMatchKey();
                _generation++;
                _allocationStarted = false;
                _joinRequestedMatchId = null;
                ClearDeadline();
                // Always reset the line: a stale server message must not survive the retry.
                SetPhase(SteamLobbyPhase.WaitingForReady, SteamLobbyMessages.WaitingForReady);
                RefreshFromLobby();
                return;
            }

            switch (_operation)
            {
                case Operation.QuickMatch:
                    QuickMatch();
                    break;
                case Operation.Host:
                    HostGame(_lastHostSetup, _lastHostVisibility);
                    break;
                case Operation.Invite:
                    InviteFriend();
                    break;
                case Operation.JoinInvited:
                    if (!string.IsNullOrEmpty(_lastInvitedLobbyId)) JoinInvited(_lastInvitedLobbyId!);
                    break;
                case Operation.Reconnect:
                    if (!string.IsNullOrEmpty(_lastReconnectMatchId)) Reconnect(_lastReconnectMatchId!);
                    break;
            }
        }

        /// <summary>
        /// Abandons whatever is in flight from any non-idle phase and leaves the lobby. Every callback
        /// belonging to the abandoned attempt is dropped. The next <see cref="Tick"/> returns to idle.
        /// </summary>
        public void Cancel()
        {
            if (_disposed) return;
            if (_phase == SteamLobbyPhase.Idle) return;

            _generation++;
            _api.Cancel();
            ReleaseTicket();
            if (!string.IsNullOrEmpty(_lobbyId)) _steam.LeaveLobby(_lobbyId!);

            ClearSession();
            _idleAfterCancel = true;
            SetPhase(SteamLobbyPhase.Cancelled);
            Publish();
        }

        /// <summary>Rejoins an already allocated match, obtaining a fresh join credential.</summary>
        public void Reconnect(string matchId)
        {
            if (!EnsureSteam()) return;
            if (string.IsNullOrEmpty(matchId)) return;

            BeginOperation(Operation.Reconnect);
            _lastReconnectMatchId = matchId;
            _matchId = matchId;

            var generation = _generation;
            SetPhase(SteamLobbyPhase.RequestingTicket);
            SetDeadline(_config.AllocationTimeoutSeconds);
            Publish();
            _steam.RequestAuthTicket(ticket => OnReconnectTicket(generation, matchId, ticket));
        }

        /// <summary>Advances time. Call once per frame, after <c>steam.Pump()</c>.</summary>
        public void Tick(double nowSeconds)
        {
            _now = nowSeconds;
            if (_disposed) return;

            if (_idleAfterCancel && _phase == SteamLobbyPhase.Cancelled)
            {
                _idleAfterCancel = false;
                SetPhase(SteamLobbyPhase.Idle);
                Publish();
                return;
            }

            // The pending hw_match write belongs to one allocation. Anything that ended that
            // allocation - the opponent leaving above all - leaves the phase somewhere else, and a
            // write that lands then would hand the owner a ticket for a match with nobody in it.
            if (_matchKeyWritesLeft > 0)
            {
                if (_phase == SteamLobbyPhase.AllocatingMatch) { RetryMatchKeyWrite(); return; }
                ClearPendingMatchKey();
            }

            if (!DeadlineExpired()) return;

            if (_phase == SteamLobbyPhase.Searching)
            {
                ClearDeadline();
                _generation++;
                if (_joinInFlight)
                {
                    _joinInFlight = false;
                    _joiningLobbyId = null;
                    SetPhase(SteamLobbyPhase.BackendUnavailable);
                    Publish();
                    return;
                }
                if (_operation == Operation.QuickMatch)
                {
                    CreateLobbyForCurrentOperation();
                }
                else
                {
                    SetPhase(SteamLobbyPhase.Failed);
                    Publish();
                }
                return;
            }

            if (_phase == SteamLobbyPhase.CreatingLobby)
            {
                ClearDeadline();
                _generation++;
                SetPhase(SteamLobbyPhase.BackendUnavailable);
                Publish();
                return;
            }

            if (IsMatchServicePhase(_phase))
            {
                ClearDeadline();
                _generation++;
                _allocationStarted = false;
                _joinRequestedMatchId = null;
                _api.Cancel();
                ReleaseTicket();
                SetPhase(SteamLobbyPhase.BackendUnavailable);
                Publish();
            }
        }

        /// <summary>
        /// Full release: abandons the in-flight request, cancels the auth ticket, leaves the lobby this
        /// client still holds, and unsubscribes. Anything less strands an empty lobby in Steam and a
        /// live auth ticket on the account.
        /// </summary>
        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _generation++;
            _api.Cancel();
            ReleaseTicket();
            if (!string.IsNullOrEmpty(_lobbyId)) _steam.LeaveLobby(_lobbyId!);
            ClearSession();
            Detach();
        }

        /// <summary>
        /// Unsubscribes from Steam without releasing the session. The screen calls this the moment the
        /// match ticket is handed over, so a lobby event (an accepted invite above all) can no longer
        /// steer a coordinator whose work is done. The lobby is left later, by <see cref="Dispose"/>.
        /// </summary>
        public void Detach()
        {
            if (_detached) return;
            _detached = true;
            _steam.LobbyDataChanged -= OnLobbyDataChanged;
            _steam.MemberJoined -= OnMemberJoined;
            _steam.MemberLeft -= OnMemberLeft;
            _steam.InviteAccepted -= OnInviteAccepted;
        }

        // ----- lobby discovery -----------------------------------------------------------------

        void OnLobbyList(int generation, IReadOnlyList<SteamLobbySearchResult> results)
        {
            if (!IsCurrent(generation)) return;
            if (_phase != SteamLobbyPhase.Searching) return;

            var chosen = ChooseLobby(results);
            if (chosen == null)
            {
                ClearDeadline();
                CreateLobbyForCurrentOperation();
                return;
            }

            // The search is over the moment a join goes out, but the join needs a deadline of its own:
            // Steam can accept it and never answer, and the search deadline used to be cleared here,
            // which left the player on "Searching..." with nothing to end it. The join deadline is the
            // allocation one and it reports a backend outage, so it can never host a second lobby on
            // top of the one this client may still be entering.
            _joinInFlight = true;
            _joiningLobbyId = chosen;
            SetDeadline(_config.AllocationTimeoutSeconds);
            _steam.JoinLobby(chosen, ok => OnLobbyJoined(generation, chosen, ok));
        }

        string? ChooseLobby(IReadOnlyList<SteamLobbySearchResult>? results)
        {
            if (results == null) return null;
            foreach (var result in results)
            {
                if (result == null || string.IsNullOrEmpty(result.LobbyId)) continue;
                if (!SteamLobbyRules.IsCompatible(result.Metadata, _config.AppId, _config.ProtocolVersion)) continue;
                if (result.MemberCount != 1) continue;
                if (SteamLobbyRules.HasMatch(result.Metadata)) continue;
                return result.LobbyId;
            }
            return null;
        }

        void CreateLobbyForCurrentOperation()
        {
            var visibility = _operation == Operation.Host ? _lastHostVisibility
                : _operation == Operation.Invite ? SteamLobbyVisibility.FriendsOnly
                : SteamLobbyVisibility.Public;

            var generation = _generation;
            SetPhase(SteamLobbyPhase.CreatingLobby);
            // Steam can accept a create and never answer it. Without a deadline the player sat on
            // "Creating lobby..." for ever, with no Retry offered and nothing to time it out.
            SetDeadline(_config.AllocationTimeoutSeconds);
            Publish();
            _steam.CreateLobby(visibility, 2, lobbyId => OnLobbyCreated(generation, lobbyId));
        }

        void OnLobbyCreated(int generation, string? lobbyId)
        {
            if (!IsCurrent(generation))
            {
                // The player moved on while Steam was still creating: do not strand an empty lobby.
                if (!string.IsNullOrEmpty(lobbyId)) _steam.LeaveLobby(lobbyId!);
                return;
            }

            ClearDeadline();
            if (string.IsNullOrEmpty(lobbyId))
            {
                // Steam could not create it, or could not even issue the call. Either way it is the
                // backend that failed and trying again is worth offering.
                FailWith(SteamLobbyPhase.BackendUnavailable, null);
                return;
            }

            _lobbyId = lobbyId;
            _isOwner = true;
            // Steam refuses these writes when the local user is not the owner, or when the lobby has
            // already gone. An unpublished lobby is invisible to every search, so it must not be shown
            // to the player as one waiting for an opponent.
            var published =
                _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.App, SteamLobbyRules.Decimal(_config.AppId))
                & _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.Protocol, SteamLobbyRules.Decimal(_config.ProtocolVersion))
                & _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.Build, _config.ClientBuild ?? string.Empty)
                & _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.Ruleset, _pendingRuleset)
                & _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.Setup, _pendingSetupWire)
                & _steam.SetLobbyData(lobbyId!, SteamLobbyKeys.Name, _steam.LocalDisplayName ?? string.Empty);

            if (!published)
            {
                _steam.LeaveLobby(lobbyId!);
                ClearSession();
                FailWith(SteamLobbyPhase.Failed, SteamLobbyMessages.PublishFailed);
                return;
            }

            if (_operation == Operation.Invite) _steam.OpenInviteOverlay(lobbyId!);

            SetPhase(SteamLobbyPhase.WaitingForPlayer);
            RefreshFromLobby();
        }

        void OnLobbyJoined(int generation, string lobbyId, bool ok)
        {
            if (!IsCurrent(generation))
            {
                // Cancelled or timed out while Steam was still joining: the join still succeeded, so
                // this client is sitting in a lobby nobody is watching. Leave it.
                if (!ok || string.IsNullOrEmpty(lobbyId)) return;

                // Unless the retry landed on the SAME lobby. Steam answers call results in whatever
                // order they complete, so the abandoned join can succeed after the one that replaced
                // it. Leaving then ejects the player from the lobby this coordinator believes it
                // holds, and readiness and metadata never move again. One membership, one leave.
                if (string.Equals(lobbyId, _lobbyId, StringComparison.Ordinal)) return;
                if (string.Equals(lobbyId, _joiningLobbyId, StringComparison.Ordinal)) return;

                _steam.LeaveLobby(lobbyId);
                return;
            }

            ClearDeadline();
            _joinInFlight = false;
            _joiningLobbyId = null;
            if (!ok)
            {
                // A Quick Match race (somebody else took the slot) simply becomes hosting.
                if (_operation == Operation.QuickMatch) CreateLobbyForCurrentOperation();
                else FailWith(SteamLobbyPhase.BackendUnavailable, null);
                return;
            }

            var snapshot = _steam.GetLobby(lobbyId);
            if (snapshot == null)
            {
                // Joined something Steam cannot describe. Do not hold it: leave, and let the player
                // try again rather than sit in a lobby with no readable state.
                _steam.LeaveLobby(lobbyId);
                FailWith(SteamLobbyPhase.BackendUnavailable, null);
                return;
            }

            if (!SteamLobbyRules.IsCompatible(snapshot.Metadata, _config.AppId, _config.ProtocolVersion))
            {
                _steam.LeaveLobby(lobbyId);
                FailWith(SteamLobbyPhase.VersionMismatch, null);
                return;
            }

            _lobbyId = lobbyId;
            RefreshFromLobby();
        }

        // ----- lobby events --------------------------------------------------------------------

        void OnLobbyDataChanged(string lobbyId)
        {
            if (!IsCurrentLobby(lobbyId)) return;
            RefreshFromLobby();
        }

        void OnMemberJoined(string lobbyId, string steamId)
        {
            if (!IsCurrentLobby(lobbyId)) return;
            RefreshFromLobby();
        }

        void OnMemberLeft(string lobbyId, string steamId)
        {
            if (!IsCurrentLobby(lobbyId)) return;
            if (string.Equals(steamId, _steam.LocalSteamId, StringComparison.Ordinal)) return;
            HandleOpponentLeft();
        }

        /// <summary>
        /// An accepted invite starts a join only when nothing else is going on. Honouring it from any
        /// other phase would tear down a lobby the player is already in, or replace a live match.
        /// </summary>
        void OnInviteAccepted(string lobbyId)
        {
            if (_disposed) return;
            if (_phase != SteamLobbyPhase.Idle && _phase != SteamLobbyPhase.Cancelled) return;
            JoinInvited(lobbyId);
        }

        void HandleOpponentLeft()
        {
            if (_disposed) return;
            if (_phase == SteamLobbyPhase.MatchReady || _phase == SteamLobbyPhase.Idle
                || _phase == SteamLobbyPhase.Cancelled) return;

            if (IsMatchServicePhase(_phase))
            {
                _api.Cancel();
                ReleaseTicket();
                _generation++;
            }

            ClearDeadline();
            ClearPendingMatchKey();   // the allocation this write belonged to is over
            _allocationStarted = false;
            _joinRequestedMatchId = null;
            _matchId = null;
            _localReady = false;
            _remoteReady = false;
            _opponentName = null;

            var snapshot = string.IsNullOrEmpty(_lobbyId) ? null : _steam.GetLobby(_lobbyId!);
            if (snapshot != null)
            {
                _isOwner = string.Equals(snapshot.OwnerSteamId, _steam.LocalSteamId, StringComparison.Ordinal);
            }

            SetPhase(SteamLobbyPhase.WaitingForPlayer);
            Publish();

            // Clear our own ready flag in Steam so the next player does not walk into a stale "1".
            if (_isOwner && !string.IsNullOrEmpty(_lobbyId))
            {
                _steam.SetMemberData(_lobbyId!, SteamLobbyKeys.MemberReady, SteamLobbyKeys.ReadyFalse);
            }
        }

        /// <summary>Re-reads the lobby and moves the flow on when the handshake is complete.</summary>
        void RefreshFromLobby()
        {
            if (_disposed || string.IsNullOrEmpty(_lobbyId)) return;

            var snapshot = _steam.GetLobby(_lobbyId!);
            if (snapshot == null) return;

            _isOwner = string.Equals(snapshot.OwnerSteamId, _steam.LocalSteamId, StringComparison.Ordinal);

            var other = OtherMember(snapshot);
            _localReady = IsReady(MemberOf(snapshot, _steam.LocalSteamId));
            _remoteReady = IsReady(other);
            _opponentName = other == null
                ? null
                : (string.IsNullOrEmpty(other.DisplayName) ? other.SteamId : other.DisplayName);

            if (!IsLobbyPhase(_phase))
            {
                Publish();
                return;
            }

            string? matchId;
            snapshot.Metadata.TryGetValue(SteamLobbyKeys.Match, out matchId);
            var hasMatch = !string.IsNullOrEmpty(matchId);

            if (hasMatch && !_isOwner)
            {
                BeginGuestJoin(matchId!);
                return;
            }

            SetPhase(snapshot.Members.Count >= 2
                ? SteamLobbyPhase.WaitingForReady
                : SteamLobbyPhase.WaitingForPlayer);

            if (_isOwner && _localReady && _remoteReady && !hasMatch && !_allocationStarted
                && snapshot.Members.Count >= 2)
            {
                BeginAllocation(snapshot);
                return;
            }

            Publish();
        }

        // ----- match allocation ----------------------------------------------------------------

        void BeginAllocation(SteamLobbySnapshot snapshot)
        {
            _allocationStarted = true;
            string? setupWire;
            snapshot.Metadata.TryGetValue(SteamLobbyKeys.Setup, out setupWire);
            _pendingSetupWire = setupWire ?? _pendingSetupWire;

            var generation = _generation;
            SetPhase(SteamLobbyPhase.RequestingTicket);
            SetDeadline(_config.AllocationTimeoutSeconds);
            Publish();
            _steam.RequestAuthTicket(ticket => OnOwnerTicket(generation, ticket));
        }

        void OnOwnerTicket(int generation, string? ticket)
        {
            if (!IsCurrent(generation)) return;
            if (string.IsNullOrEmpty(ticket))
            {
                _allocationStarted = false;
                ClearDeadline();
                ReleaseTicket();
                FailWith(SteamLobbyPhase.Failed, null);
                return;
            }

            SetPhase(SteamLobbyPhase.AllocatingMatch);
            Publish();
            _api.CreateMatch(_lobbyId ?? string.Empty, ticket!, _pendingSetupWire,
                result => OnCreateMatchDone(generation, result));
        }

        void OnCreateMatchDone(int generation, SteamMatchApiResult result)
        {
            if (!IsCurrent(generation)) return;

            ClearDeadline();
            ReleaseTicket();   // one ticket per exchange: it is spent now, whatever happened
            if (result != null && result.Ok)
            {
                if (!string.IsNullOrEmpty(_lobbyId) && !string.IsNullOrEmpty(result.MatchId)
                    && !_steam.SetLobbyData(_lobbyId!, SteamLobbyKeys.Match, result.MatchId!))
                {
                    // Without hw_match the guest never learns the match exists, so the owner must not
                    // walk into it alone. Try again on the next two Ticks before abandoning it.
                    _pendingMatchKey = result.MatchId;
                    _pendingMatchResult = result;
                    _matchKeyWritesLeft = 2;
                    Publish();
                    return;
                }
                CompleteMatch(result);
                return;
            }

            MapApiFailure(result);
        }

        /// <summary>One more attempt at the hw_match write, from <see cref="Tick"/>.</summary>
        void RetryMatchKeyWrite()
        {
            var key = _pendingMatchKey;
            if (!string.IsNullOrEmpty(_lobbyId) && !string.IsNullOrEmpty(key)
                && _steam.SetLobbyData(_lobbyId!, SteamLobbyKeys.Match, key!))
            {
                var result = _pendingMatchResult!;
                ClearPendingMatchKey();
                CompleteMatch(result);
                return;
            }

            _matchKeyWritesLeft--;
            if (_matchKeyWritesLeft > 0) return;

            ClearPendingMatchKey();
            _generation++;
            _api.Cancel();
            // The ticket is already gone: this branch is only reachable through OnCreateMatchDone,
            // which releases it the moment the exchange completes, write outcome or not.
            if (!string.IsNullOrEmpty(_lobbyId)) _steam.LeaveLobby(_lobbyId!);
            ClearSession();
            // The allocated match is left for the server retention sweep to reclaim.
            FailWith(SteamLobbyPhase.Failed, SteamLobbyMessages.PublishFailed);
        }

        void ClearPendingMatchKey()
        {
            _matchKeyWritesLeft = 0;
            _pendingMatchKey = null;
            _pendingMatchResult = null;
        }

        void BeginGuestJoin(string matchId)
        {
            if (string.Equals(_joinRequestedMatchId, matchId, StringComparison.Ordinal))
            {
                Publish();
                return;
            }

            _joinRequestedMatchId = matchId;
            _matchId = matchId;

            var generation = _generation;
            SetPhase(SteamLobbyPhase.RequestingTicket);
            SetDeadline(_config.AllocationTimeoutSeconds);
            Publish();
            _steam.RequestAuthTicket(ticket => OnGuestTicket(generation, matchId, ticket));
        }

        void OnGuestTicket(int generation, string matchId, string? ticket)
        {
            if (!IsCurrent(generation)) return;
            if (string.IsNullOrEmpty(ticket))
            {
                _joinRequestedMatchId = null;
                ClearDeadline();
                ReleaseTicket();
                FailWith(SteamLobbyPhase.Failed, null);
                return;
            }

            SetPhase(SteamLobbyPhase.JoiningMatch);
            Publish();
            _api.JoinMatch(matchId, ticket!, result => OnJoinMatchDone(generation, result));
        }

        void OnReconnectTicket(int generation, string matchId, string? ticket)
        {
            if (!IsCurrent(generation)) return;
            if (string.IsNullOrEmpty(ticket))
            {
                ClearDeadline();
                ReleaseTicket();
                FailWith(SteamLobbyPhase.Failed, null);
                return;
            }

            SetPhase(SteamLobbyPhase.Reconnecting);
            Publish();
            _api.JoinMatch(matchId, ticket!, result => OnJoinMatchDone(generation, result));
        }

        void OnJoinMatchDone(int generation, SteamMatchApiResult result)
        {
            if (!IsCurrent(generation)) return;

            ClearDeadline();
            ReleaseTicket();   // one ticket per exchange: it is spent now, whatever happened
            if (result != null && result.Ok)
            {
                CompleteMatch(result);
                return;
            }

            MapApiFailure(result);
        }

        void CompleteMatch(SteamMatchApiResult result)
        {
            _matchId = result.MatchId;
            SetPhase(SteamLobbyPhase.MatchReady);
            Publish();
            var handler = _onMatchReady;
            if (handler != null)
            {
                handler(new SteamMatchTicket(result.MatchId, result.WebsocketUrl, result.JoinCredential, result.Seat));
            }
        }

        /// <summary>Turns a match-service error body into the phase the player sees.</summary>
        void MapApiFailure(SteamMatchApiResult? result)
        {
            _allocationStarted = false;
            _joinRequestedMatchId = null;

            var code = result == null ? string.Empty : (result.ErrorCode ?? string.Empty);
            var serverMessage = result == null || string.IsNullOrEmpty(result.Message) ? null : result.Message;

            if (string.Equals(code, SteamMatchErrorCodes.IncompatibleVersion, StringComparison.Ordinal))
            {
                FailWith(SteamLobbyPhase.VersionMismatch, null);
                return;
            }

            if (string.Equals(code, SteamMatchErrorCodes.ServiceUnavailable, StringComparison.Ordinal)
                || string.Equals(code, SteamMatchErrorCodes.RateLimited, StringComparison.Ordinal)
                || (result != null && result.HttpStatus == 0))
            {
                FailWith(SteamLobbyPhase.BackendUnavailable, null);
                return;
            }

            if (string.Equals(code, SteamMatchErrorCodes.LobbyChanged, StringComparison.Ordinal))
            {
                // The server saw a different lobby than we did. Drop our own ready so the retry is a
                // deliberate act by the player instead of an immediate re-allocation loop.
                _matchId = null;
                _localReady = false;
                SetPhase(SteamLobbyPhase.WaitingForReady, serverMessage ?? SteamLobbyMessages.WaitingForReady);
                Publish();
                if (!string.IsNullOrEmpty(_lobbyId))
                {
                    _steam.SetMemberData(_lobbyId!, SteamLobbyKeys.MemberReady, SteamLobbyKeys.ReadyFalse);
                }
                return;
            }

            FailWith(SteamLobbyPhase.Failed, serverMessage);
        }

        // ----- state plumbing ------------------------------------------------------------------

        /// <summary>
        /// Releases the Steam Web API auth ticket this exchange holds.
        /// <para>
        /// One ticket belongs to one exchange, and every path that abandons an exchange comes through
        /// here: a timeout, an opponent walking out mid-allocation, a ticket that never arrived, an
        /// API error, a retry, a hw_match write that could not be published. A ticket that is not
        /// cancelled stays live on the account for its full lifetime, and the paths that skipped it
        /// were exactly the paths nobody exercises by hand.
        /// </para>
        /// </summary>
        void ReleaseTicket()
        {
            _steam.CancelAuthTicket();
        }

        bool EnsureSteam()
        {
            if (_disposed) return false;
            if (_steam.IsAvailable) return true;

            SetPhase(SteamLobbyPhase.SteamUnavailable);
            Publish();
            return false;
        }

        void BeginOperation(Operation operation)
        {
            // Starting an operation over one that is still in flight abandons it. Bumping the
            // generation only makes this client ignore the answer: the match service request keeps
            // running and the Web API ticket stays live on the account unless both are released here.
            if (_phase != SteamLobbyPhase.Idle && _phase != SteamLobbyPhase.Cancelled)
            {
                _api.Cancel();
                ReleaseTicket();
            }

            _generation++;
            _operation = operation;
            _idleAfterCancel = false;
            _allocationStarted = false;
            _joinRequestedMatchId = null;
            ClearDeadline();

            if (!string.IsNullOrEmpty(_lobbyId)) _steam.LeaveLobby(_lobbyId!);
            ClearSession();
        }

        void ClearSession()
        {
            _lobbyId = null;
            _matchId = null;
            _isOwner = false;
            _localReady = false;
            _remoteReady = false;
            _opponentName = null;
            _allocationStarted = false;
            _joinRequestedMatchId = null;
            _joinInFlight = false;
            _joiningLobbyId = null;
            ClearPendingMatchKey();
            ClearDeadline();
        }

        /// <summary>Moves to a terminal phase, preferring the player-safe server message when there is one.</summary>
        void FailWith(SteamLobbyPhase phase, string? serverMessage)
        {
            SetPhase(phase, serverMessage ?? SteamLobbyMessages.For(phase));
            Publish();
        }

        void SetPhase(SteamLobbyPhase phase)
        {
            if (_phase == phase) return;
            _phase = phase;
            _message = SteamLobbyMessages.For(phase);
        }

        void SetPhase(SteamLobbyPhase phase, string message)
        {
            _phase = phase;
            _message = message ?? string.Empty;
        }

        /// <summary>Arms a relative deadline. It starts running on the next <see cref="Tick"/>.</summary>
        void SetDeadline(double seconds)
        {
            _hasDeadline = true;
            _deadlineSeconds = seconds;
            _deadlineAnchor = null;
        }

        void ClearDeadline()
        {
            _hasDeadline = false;
            _deadlineSeconds = 0;
            _deadlineAnchor = null;
        }

        /// <summary>True when an armed deadline has been observed for its full duration.</summary>
        bool DeadlineExpired()
        {
            if (!_hasDeadline) return false;
            if (_deadlineAnchor == null) { _deadlineAnchor = _now; return false; }
            return _now >= _deadlineAnchor.Value + _deadlineSeconds;
        }

        bool IsCurrent(int generation)
        {
            return !_disposed && generation == _generation;
        }

        bool IsCurrentLobby(string lobbyId)
        {
            return !_disposed
                && !string.IsNullOrEmpty(_lobbyId)
                && string.Equals(lobbyId, _lobbyId, StringComparison.Ordinal);
        }

        void Publish()
        {
            var next = BuildStatus();
            if (next.Matches(_status)) return;
            _status = next;
            var handler = _onStatus;
            if (handler != null) handler(next);
        }

        SteamLobbyStatus BuildStatus()
        {
            return new SteamLobbyStatus(
                _phase,
                _message,
                _lobbyId,
                _matchId,
                _isOwner,
                _localReady,
                _remoteReady,
                _opponentName,
                CanCancelIn(_phase),
                CanRetryIn(_phase),
                _phase == SteamLobbyPhase.WaitingForReady);
        }

        static bool IsLobbyPhase(SteamLobbyPhase phase)
        {
            return phase == SteamLobbyPhase.Searching
                || phase == SteamLobbyPhase.CreatingLobby
                || phase == SteamLobbyPhase.WaitingForPlayer
                || phase == SteamLobbyPhase.WaitingForReady;
        }

        static bool IsMatchServicePhase(SteamLobbyPhase phase)
        {
            return phase == SteamLobbyPhase.RequestingTicket
                || phase == SteamLobbyPhase.AllocatingMatch
                || phase == SteamLobbyPhase.JoiningMatch
                || phase == SteamLobbyPhase.Reconnecting;
        }

        static bool IsRetryableInLobby(SteamLobbyPhase phase)
        {
            return phase == SteamLobbyPhase.BackendUnavailable
                || phase == SteamLobbyPhase.Failed
                || phase == SteamLobbyPhase.WaitingForReady;
        }

        static bool CanCancelIn(SteamLobbyPhase phase)
        {
            return phase != SteamLobbyPhase.Idle
                && phase != SteamLobbyPhase.Cancelled
                && phase != SteamLobbyPhase.SteamUnavailable
                && phase != SteamLobbyPhase.MatchReady;
        }

        static bool CanRetryIn(SteamLobbyPhase phase)
        {
            return phase == SteamLobbyPhase.BackendUnavailable || phase == SteamLobbyPhase.Failed;
        }

        static bool IsReady(SteamLobbyMemberSnapshot? member)
        {
            if (member == null) return false;
            string? value;
            return member.Data.TryGetValue(SteamLobbyKeys.MemberReady, out value)
                && string.Equals(value, SteamLobbyKeys.ReadyTrue, StringComparison.Ordinal);
        }

        static SteamLobbyMemberSnapshot? MemberOf(SteamLobbySnapshot snapshot, string steamId)
        {
            foreach (var member in snapshot.Members)
            {
                if (string.Equals(member.SteamId, steamId, StringComparison.Ordinal)) return member;
            }
            return null;
        }

        SteamLobbyMemberSnapshot? OtherMember(SteamLobbySnapshot snapshot)
        {
            foreach (var member in snapshot.Members)
            {
                if (!string.Equals(member.SteamId, _steam.LocalSteamId, StringComparison.Ordinal)) return member;
            }
            return null;
        }
    }
}
