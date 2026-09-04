using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// Keeps every formatted log message the host writes, so a test can assert that a secret never reached
    /// a log line. Thread-safe on purpose: the host logs from several threads while it starts.
    /// </summary>
    public sealed class CapturingLoggerProvider : ILoggerProvider
    {
        readonly ConcurrentQueue<string> _messages = new();

        /// <summary>A snapshot of everything captured so far, oldest first.</summary>
        public IReadOnlyList<string> Messages => _messages.ToArray();

        public bool Any(string fragment) =>
            _messages.ToArray().Any(m => m.Contains(fragment, StringComparison.Ordinal));

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

            public IDisposable BeginScope<TState>(TState state) where TState : notnull => NullScope.Instance;

            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                _messages.Enqueue(_category + ": " + formatter(state, exception));
                if (exception is not null) _messages.Enqueue(_category + ": " + exception);
            }
        }

        sealed class NullScope : IDisposable
        {
            public static readonly NullScope Instance = new();

            public void Dispose() { }
        }
    }
}
