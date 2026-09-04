using System.Collections.Concurrent;
using Microsoft.Extensions.Logging;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// Captures every formatted line the host logs, from every category, so a test can assert on what a
    /// real DI-composed server would have written to a real sink rather than on what one hand-built
    /// object does in isolation. It is registered as an ordinary provider, which is the point: anything
    /// the framework logs on the way to the wire lands here too.
    /// </summary>
    public sealed class CapturingLoggerProvider : ILoggerProvider
    {
        readonly ConcurrentQueue<string> _messages = new();

        /// <summary>Every captured line, "Category: message", oldest first.</summary>
        public IReadOnlyList<string> Messages => _messages.ToArray();

        public ILogger CreateLogger(string categoryName) => new CapturingLogger(categoryName, _messages);

        public void Dispose()
        {
        }

        sealed class CapturingLogger(string category, ConcurrentQueue<string> messages) : ILogger
        {
            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

            // Never a filter of its own: the whole question a test asks of this class is what the
            // configured filters let through, so it must not add one.
            public bool IsEnabled(LogLevel logLevel) => true;

            public void Log<TState>(
                LogLevel logLevel,
                EventId eventId,
                TState state,
                Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                // The exception is captured as well as the message: a failed request URL reaches a log
                // through the exception a sink renders, not only through the format string.
                var line = category + ": " + formatter(state, exception);
                if (exception is not null) line += " " + exception;
                messages.Enqueue(line);
            }
        }
    }
}
