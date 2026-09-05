#nullable enable
// Duplicated from Assets/HexWars/Tests/Editor/Fakes/ - the EditMode and PlayMode suites are separate
// assemblies and neither may reference the other, so the fakes exist once per suite. Keep the two
// copies in step: the EditMode copy is the one linked into engine/HexWars.Engine.Tests.
using System;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace HexWars.Presentation.PlayModeTests
{
    /// <summary>
    /// Deterministic in-memory <see cref="ISteamSocketDriver"/>. Nothing happens on its own: a test
    /// raises Opened / Message / Closed for whichever attempt id it likes, which is exactly how an
    /// event belonging to an abandoned attempt is staged. A close can be left hanging, so the bounded
    /// wait a caller does before reopening can be exercised in both directions.
    /// </summary>
    public sealed class FakeSteamSocketDriver : ISteamSocketDriver
    {
        readonly List<TaskCompletionSource<bool>> _pendingCloses = new List<TaskCompletionSource<bool>>();

        public event Action<int>? Opened;
        public event Action<int, string>? Message;
        public event Action<int, string>? Closed;
        public event Action<int, string>? Error;

        /// <summary>Every (attemptId, url) Open was called with, in order.</summary>
        public List<KeyValuePair<int, string>> Opens { get; } = new List<KeyValuePair<int, string>>();

        /// <summary>Every (attemptId, text) Send was called with, in order.</summary>
        public List<KeyValuePair<int, string>> Sends { get; } = new List<KeyValuePair<int, string>>();

        /// <summary>Every attempt id CloseAsync was called with, in order.</summary>
        public List<int> CloseCalls { get; } = new List<int>();

        /// <summary>False leaves every CloseAsync task pending until <see cref="CompleteCloses"/>.</summary>
        public bool CompleteClosesImmediately { get; set; } = true;

        /// <summary>How many CloseAsync tasks have been handed out and not completed.</summary>
        public int OutstandingCloses { get { return _pendingCloses.Count; } }

        // ----- ISteamSocketDriver -------------------------------------------------------------

        public void Open(string url, int attemptId)
        {
            Opens.Add(new KeyValuePair<int, string>(attemptId, url ?? string.Empty));
        }

        public void Send(int attemptId, string text)
        {
            Sends.Add(new KeyValuePair<int, string>(attemptId, text ?? string.Empty));
        }

        public Task CloseAsync(int attemptId)
        {
            CloseCalls.Add(attemptId);
            if (CompleteClosesImmediately) return Task.CompletedTask;

            var pending = new TaskCompletionSource<bool>();
            _pendingCloses.Add(pending);
            return pending.Task;
        }

        // ----- test drivers -------------------------------------------------------------------

        public void RaiseOpened(int attemptId)
        {
            var handler = Opened;
            if (handler != null) handler(attemptId);
        }

        public void RaiseMessage(int attemptId, string text)
        {
            var handler = Message;
            if (handler != null) handler(attemptId, text);
        }

        public void RaiseClosed(int attemptId, string reason = "closed")
        {
            var handler = Closed;
            if (handler != null) handler(attemptId, reason);
        }

        public void RaiseError(int attemptId, string message)
        {
            var handler = Error;
            if (handler != null) handler(attemptId, message);
        }

        /// <summary>Completes every CloseAsync task handed out so far.</summary>
        public void CompleteCloses()
        {
            foreach (var pending in _pendingCloses) pending.TrySetResult(true);
            _pendingCloses.Clear();
        }

        /// <summary>The text of every Send made for one attempt, in order.</summary>
        public List<string> SendsFor(int attemptId)
        {
            var texts = new List<string>();
            foreach (var send in Sends)
            {
                if (send.Key == attemptId) texts.Add(send.Value);
            }
            return texts;
        }
    }
}
