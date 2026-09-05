#nullable enable
using System;
using System.Collections.Generic;

namespace HexWars.Presentation.Tests
{
    /// <summary>
    /// Deterministic <see cref="ISteamMatchApi"/> for tests. Every call is recorded and answered from a
    /// scripted queue (falling back to a canned success). Results are delivered inline unless
    /// <see cref="Deferred"/> is set, in which case they wait for <see cref="CompletePending"/>. That is
    /// how a test holds a request open long enough for a timeout or a cancel to land.
    /// </summary>
    public sealed class FakeSteamMatchApi : ISteamMatchApi
    {
        /// <summary>One recorded call. <see cref="Kind"/> is <c>create</c> or <c>join</c>.</summary>
        public sealed class ApiCall
        {
            public ApiCall(string kind, string lobbyId, string matchId, string ticketHex, string setupWire)
            {
                Kind = kind;
                LobbyId = lobbyId ?? string.Empty;
                MatchId = matchId ?? string.Empty;
                TicketHex = ticketHex ?? string.Empty;
                SetupWire = setupWire ?? string.Empty;
            }

            public string Kind { get; }
            public string LobbyId { get; }
            public string MatchId { get; }
            public string TicketHex { get; }
            public string SetupWire { get; }
        }

        public const string CreateKind = "create";
        public const string JoinKind = "join";

        readonly Queue<Action> _pending = new Queue<Action>();

        /// <summary>Every CreateMatch/JoinMatch call in order.</summary>
        public List<ApiCall> Calls { get; } = new List<ApiCall>();

        /// <summary>Scripted CreateMatch results, consumed in order.</summary>
        public Queue<SteamMatchApiResult> CreateResults { get; } = new Queue<SteamMatchApiResult>();

        /// <summary>Scripted JoinMatch results, consumed in order.</summary>
        public Queue<SteamMatchApiResult> JoinResults { get; } = new Queue<SteamMatchApiResult>();

        /// <summary>Used when <see cref="CreateResults"/> runs dry.</summary>
        public SteamMatchApiResult DefaultCreateResult { get; set; } =
            SteamMatchApiResult.Success("match-0001", "wss://match.invalid/ws/v2", "owner-credential", 0);

        /// <summary>Used when <see cref="JoinResults"/> runs dry.</summary>
        public SteamMatchApiResult DefaultJoinResult { get; set; } =
            SteamMatchApiResult.Success("match-0001", "wss://match.invalid/ws/v2", "guest-credential", 1);

        /// <summary>When true, results wait for <see cref="CompletePending"/> instead of firing inline.</summary>
        public bool Deferred { get; set; }

        public int CancelCalls { get; private set; }

        public int PendingCount { get { return _pending.Count; } }

        public int CreateMatchCalls { get { return CountOf(CreateKind); } }

        public int JoinMatchCalls { get { return CountOf(JoinKind); } }

        public ApiCall? LastCall { get { return Calls.Count == 0 ? null : Calls[Calls.Count - 1]; } }

        public void CreateMatch(string lobbyId, string ticketHex, string requestedSetupWire, Action<SteamMatchApiResult> onDone)
        {
            Calls.Add(new ApiCall(CreateKind, lobbyId, string.Empty, ticketHex, requestedSetupWire));
            var result = CreateResults.Count > 0 ? CreateResults.Dequeue() : DefaultCreateResult;
            Deliver(() => onDone(result));
        }

        public void JoinMatch(string matchId, string ticketHex, Action<SteamMatchApiResult> onDone)
        {
            Calls.Add(new ApiCall(JoinKind, string.Empty, matchId, ticketHex, string.Empty));
            var result = JoinResults.Count > 0 ? JoinResults.Dequeue() : DefaultJoinResult;
            Deliver(() => onDone(result));
        }

        /// <summary>Abandons in-flight work: a deferred result queued before this never fires.</summary>
        public void Cancel()
        {
            CancelCalls++;
            _pending.Clear();
        }

        /// <summary>Releases every deferred result queued so far.</summary>
        public void CompletePending()
        {
            for (var i = 0; i < 1024 && _pending.Count > 0; i++) _pending.Dequeue()();
        }

        void Deliver(Action action)
        {
            if (Deferred) _pending.Enqueue(action);
            else action();
        }

        int CountOf(string kind)
        {
            var n = 0;
            foreach (var call in Calls)
            {
                if (string.Equals(call.Kind, kind, StringComparison.Ordinal)) n++;
            }
            return n;
        }
    }
}
