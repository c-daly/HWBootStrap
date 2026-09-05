using HexWars.NetServer.Contracts;
using Microsoft.AspNetCore.Server.Kestrel.Core;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// What this server will spend on a request before it has decided to serve it.
    ///
    /// Every number here is a ceiling on work an unauthenticated caller can make this process do. The body
    /// cap is the important one: both Steam endpoints read a small JSON object, and the only reason to send
    /// a large body to either is to make the server allocate. The connection caps bound how many sockets one
    /// host will hold at once, and the two timeouts close the two shapes of a connection that costs a slot
    /// while saying nothing - headers that never finish arriving, and a keep-alive nobody uses.
    /// </summary>
    public static class RequestLimits
    {
        /// <summary>
        /// The largest request body this server will accept, on any route.
        ///
        /// Comfortably above a Steam auth ticket wrapped in JSON and far below anything worth buffering. The
        /// endpoints apply their own, tighter cap when they read the body; this is the outer bound that
        /// applies before a handler is reached at all.
        /// </summary>
        public const long MaxRequestBodyBytes = 16 * 1024;

        public const int MaxConcurrentConnections = 2000;

        public const int MaxConcurrentUpgradedConnections = 1000;

        public static readonly TimeSpan RequestHeadersTimeout = TimeSpan.FromSeconds(10);

        public static readonly TimeSpan KeepAliveTimeout = TimeSpan.FromSeconds(120);

        /// <summary>Says the size was the problem and nothing else. A caller that sent a large body either
        /// has a bug or is probing, and neither is helped by a more specific sentence.</summary>
        public const string TooLargeMessage = "That request was too large.";

        /// <summary>The Kestrel side: the limits the server itself enforces, before any middleware runs.</summary>
        public static void Configure(KestrelServerOptions kestrel)
        {
            ArgumentNullException.ThrowIfNull(kestrel);

            kestrel.Limits.MaxRequestBodySize = MaxRequestBodyBytes;
            kestrel.Limits.MaxConcurrentConnections = MaxConcurrentConnections;
            kestrel.Limits.MaxConcurrentUpgradedConnections = MaxConcurrentUpgradedConnections;
            kestrel.Limits.RequestHeadersTimeout = RequestHeadersTimeout;
            kestrel.Limits.KeepAliveTimeout = KeepAliveTimeout;

            // The server header names the software and its version to everyone who asks, including the
            // people writing the scanner. It buys nothing.
            kestrel.AddServerHeader = false;
        }

        /// <summary>
        /// Refuses an over-large body up front, with the error shape every other refusal uses.
        ///
        /// Kestrel already enforces the same bound, but it does so by throwing when the body is READ, which
        /// reaches the client as a bare 413 with no body and reaches the log as an exception. Answering from
        /// the declared Content-Length instead means the caller gets the same JSON they get for every other
        /// refusal, and the body is never pulled off the wire at all. A request that declares no length is
        /// left to Kestrel, which is the only thing that can measure one as it arrives.
        /// </summary>
        public static IApplicationBuilder UseHexWarsRequestLimits(this IApplicationBuilder app)
        {
            ArgumentNullException.ThrowIfNull(app);

            return app.Use(async (context, next) =>
            {
                if (context.Request.ContentLength is long declared && declared > MaxRequestBodyBytes)
                {
                    context.Response.StatusCode = StatusCodes.Status413PayloadTooLarge;
                    await context.Response
                        .WriteAsJsonAsync(new ApiError(ApiErrors.InvalidRequest, TooLargeMessage))
                        .ConfigureAwait(false);
                    return;
                }

                await next().ConfigureAwait(false);
            });
        }
    }
}
