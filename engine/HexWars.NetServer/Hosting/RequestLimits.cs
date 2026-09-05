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
        /// Kestrel enforces the same bound, but only when the application READS that far, and this one never
        /// does: the endpoints stop at their own, much tighter JSON cap. So a body that simply omits
        /// Content-Length would sail past the transport limit and be answered 400 by the JSON reader, which
        /// is the wrong answer and, worse, means the documented 16 KB ceiling was never actually a ceiling.
        ///
        /// A declared length is answered from the header, without a byte being pulled off the socket. An
        /// undeclared one is measured: the body is buffered and read one byte past the cap, which is bounded
        /// work whatever the client intends to send, and rewound so the endpoint still sees it. Only requests
        /// that can carry a body are touched, so a websocket upgrade and a static file GET pay nothing.
        /// </summary>
        public static IApplicationBuilder UseHexWarsRequestLimits(this IApplicationBuilder app)
        {
            ArgumentNullException.ThrowIfNull(app);

            return app.Use(async (context, next) =>
            {
                if (context.Request.ContentLength is long declared)
                {
                    if (declared > MaxRequestBodyBytes)
                    {
                        await RefuseAsync(context).ConfigureAwait(false);
                        return;
                    }
                }
                else if (CarriesAnUndeclaredBody(context.Request))
                {
                    if (await IsOverTheCapAsync(context.Request).ConfigureAwait(false))
                    {
                        await RefuseAsync(context).ConfigureAwait(false);
                        return;
                    }
                }

                await next().ConfigureAwait(false);
            });
        }

        /// <summary>A request with no Content-Length that still intends to send something. GET, HEAD,
        /// DELETE, OPTIONS and TRACE do not, and a websocket upgrade is a GET.</summary>
        static bool CarriesAnUndeclaredBody(HttpRequest request) =>
            !HttpMethods.IsGet(request.Method)
            && !HttpMethods.IsHead(request.Method)
            && !HttpMethods.IsOptions(request.Method)
            && !HttpMethods.IsDelete(request.Method)
            && !HttpMethods.IsTrace(request.Method);

        /// <summary>
        /// Reads one byte past the cap and says whether it got there.
        ///
        /// Buffering first is what lets the endpoint read the same body afterwards. The buffer is bounded by
        /// the cap plus one, so a client streaming megabytes is measured in kilobytes and then refused.
        /// </summary>
        static async Task<bool> IsOverTheCapAsync(HttpRequest request)
        {
            request.EnableBuffering();

            var probe = new byte[MaxRequestBodyBytes + 1];
            var filled = 0;

            try
            {
                while (filled < probe.Length)
                {
                    int read = await request.Body
                        .ReadAsync(
                            probe.AsMemory(filled, probe.Length - filled),
                            request.HttpContext.RequestAborted)
                        .ConfigureAwait(false);

                    if (read == 0) break;

                    filled += read;
                }
            }
            catch (Microsoft.AspNetCore.Http.BadHttpRequestException)
            {
                // Kestrel enforces the same ceiling and reaches it first, one byte earlier than this probe
                // does. Its own answer is a 413 with an exception page rather than the error body every
                // other refusal on this server uses, so the exception is caught and answered here instead.
                return true;
            }

            request.Body.Position = 0;
            return filled > MaxRequestBodyBytes;
        }

        static Task RefuseAsync(HttpContext context)
        {
            context.Response.StatusCode = StatusCodes.Status413PayloadTooLarge;
            return context.Response.WriteAsJsonAsync(new ApiError(ApiErrors.InvalidRequest, TooLargeMessage));
        }
    }
}
