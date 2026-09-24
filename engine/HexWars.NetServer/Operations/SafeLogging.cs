using HexWars.NetServer.Steam;

namespace HexWars.NetServer.Operations
{
    /// <summary>
    /// The one way this server logs an exception.
    ///
    /// Handing an exception object to a logger hands the sink everything that exception is quoting, and the
    /// exceptions that reach a catch block here are raised by code that was holding a secret when it
    /// failed: an HTTP transport error quotes the URL it could not reach, which for the Steam client is the
    /// publisher key and a player auth ticket, and a database failure quotes its target, which is
    /// DATABASE_URL. Neither is redacted by anything downstream, because the formatter simply calls
    /// ToString.
    ///
    /// So nothing here passes the exception through. What reaches the log is its TYPE and its text with the
    /// values taken out - stack frames included, since those name methods rather than values and are the
    /// part actually worth reading. It is an extension method so that every catch site is one call, and
    /// there is one place to fix when a new shape of secret turns up.
    /// </summary>
    public static class SafeLogging
    {
        /// <summary>The property the redacted failure is logged under, so a structured sink can index it.</summary>
        public const string FailureProperty = "{Failure}";

        /// <summary>
        /// Logs <paramref name="failure"/> as redacted text appended to <paramref name="message"/>.
        /// </summary>
        /// <param name="message">A message template, WITHOUT a placeholder for the failure: one is appended.</param>
        public static void LogRedacted(
            this ILogger logger,
            LogLevel level,
            Exception? failure,
            string message,
            params object?[] args)
        {
            ArgumentNullException.ThrowIfNull(logger);
            ArgumentNullException.ThrowIfNull(args);

            var all = new object?[args.Length + 1];
            Array.Copy(args, all, args.Length);
            all[args.Length] = SteamLogRedaction.Describe(failure);

            logger.Log(level, message + ": " + FailureProperty, all);
        }

        public static void LogRedactedWarning(
            this ILogger logger, Exception? failure, string message, params object?[] args) =>
            logger.LogRedacted(LogLevel.Warning, failure, message, args);

        public static void LogRedactedError(
            this ILogger logger, Exception? failure, string message, params object?[] args) =>
            logger.LogRedacted(LogLevel.Error, failure, message, args);
    }
}
