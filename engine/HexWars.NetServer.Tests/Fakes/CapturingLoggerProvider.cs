using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// Keeps every formatted log message the host writes, so a test can assert that a secret never reached
    /// a log line. Thread-safe on purpose: the host logs from several threads while it starts.
    ///
    /// Scopes are captured too, and that is not a convenience. A scope is attached to every line written
    /// underneath it, so a value put in one is in the log as surely as a value in a message - and a leak
    /// audit that only read message text would be blind to exactly the mechanism this server uses to carry
    /// identifiers around. Each captured line therefore carries the scopes that were open when it was
    /// written, in the same string the assertion searches.
    /// </summary>
    public sealed class CapturingLoggerProvider : ILoggerProvider
    {
        readonly ConcurrentQueue<string> _messages = new();

        /// <summary>Scopes are per-thread and nested, which is how the logging contract defines them.</summary>
        static readonly AsyncLocal<Scope?> Current = new();

        /// <summary>A snapshot of everything captured so far, oldest first.</summary>
        public IReadOnlyList<string> Messages => _messages.ToArray();

        public bool Any(string fragment) =>
            _messages.ToArray().Any(m => m.Contains(fragment, StringComparison.Ordinal));

        /// <summary>The captured lines that contain <paramref name="fragment"/>. What a failing leak
        /// assertion prints, so the message names the line rather than only the secret.</summary>
        public IReadOnlyList<string> Containing(string fragment) =>
            _messages.ToArray().Where(m => m.Contains(fragment, StringComparison.Ordinal)).ToArray();

        public ILogger CreateLogger(string categoryName) => new CapturingLogger(categoryName, _messages);

        public void Dispose() { }

        sealed class CapturingLogger : ILogger
        {
            readonly string _category;
            readonly ConcurrentQueue<string> _messages;

            public CapturingLogger(string category, ConcurrentQueue<string> messages)
            {
                _category = category;
                _messages = messages;
            }

            public IDisposable BeginScope<TState>(TState state) where TState : notnull
            {
                var scope = new Scope(Describe(state), Current.Value);
                Current.Value = scope;
                return scope;
            }

            /// <summary>The scope as text an assertion can search. A scope is usually a key/value bag, and
            /// ToString on one of those names the dictionary type rather than anything it holds - which is
            /// how a leak audit ends up passing over a log full of the value it was looking for.</summary>
            static string Describe<TState>(TState state)
            {
                if (state is IEnumerable<KeyValuePair<string, object?>> pairs)
                    return string.Join(" ", pairs.Select(p => p.Key + "=" + p.Value));

                return state?.ToString() ?? string.Empty;
            }

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                _messages.Enqueue(_category + ": " + ScopeText() + formatter(state, exception));
                if (exception is not null) _messages.Enqueue(_category + ": " + exception);
            }

            static string ScopeText()
            {
                Scope? scope = Current.Value;
                if (scope is null) return string.Empty;

                var parts = new List<string>();
                for (Scope? walk = scope; walk is not null; walk = walk.Parent) parts.Add(walk.Text);
                parts.Reverse();
                return "[" + string.Join(" ", parts) + "] ";
            }
        }

        sealed class Scope : IDisposable
        {
            public Scope(string text, Scope? parent)
            {
                Text = text;
                Parent = parent;
            }

            public string Text { get; }

            public Scope? Parent { get; }

            /// <summary>Restores the scope that was open before this one. A test host disposes scopes on the
            /// thread that opened them, so unwinding to the parent is enough.</summary>
            public void Dispose() => Current.Value = Parent;
        }
    }
}
