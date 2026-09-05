using System.Net.Http;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// Registers the Steam client as a typed HttpClient. It is registered unconditionally because it is
    /// inert until something resolves it: a Legacy-only deployment never touches Steam options, so a
    /// server with no Steam credentials still starts.
    /// </summary>
    public static class SteamWebApiRegistration
    {
        /// <summary>
        /// The name of the underlying HttpClient. Named explicitly rather than left to the type-name
        /// default so that a test, a log filter or an operator can address exactly this client.
        /// </summary>
        public const string HttpClientName = "SteamWebApi";

        /// <param name="configure">
        /// An optional hook on the builder, for a test that needs to replace the primary handler without
        /// reaching into HttpClientFactoryOptions by name.
        /// </param>
        public static IServiceCollection AddSteamWebApi(
            this IServiceCollection services, Action<IHttpClientBuilder>? configure = null)
        {
            services.TryAddSingleton(TimeProvider.System);

            var builder = services.AddHttpClient<ISteamWebApiClient, SteamWebApiClient>(
                HttpClientName,
                (provider, client) =>
                {
                    var options = provider.GetRequiredService<IOptions<SteamOptions>>().Value;

                    // A base address without a trailing slash silently drops its last path segment when a
                    // relative URI is resolved against it.
                    var baseUrl = options.WebApiBaseUrl.AbsoluteUri.EndsWith("/", StringComparison.Ordinal)
                        ? options.WebApiBaseUrl
                        : new Uri(options.WebApiBaseUrl.AbsoluteUri + "/");

                    client.BaseAddress = baseUrl;
                    client.Timeout = TimeSpan.FromSeconds(Math.Max(1, options.RequestTimeoutSeconds));

                    // Defence in depth behind the bounded read the client does itself: nothing about a
                    // Steam answer justifies buffering megabytes on this process.
                    client.MaxResponseContentBufferSize = SteamWebApiClient.MaxResponseBytes;
                })
                .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
                {
                    ConnectTimeout = TimeSpan.FromSeconds(3),
                    PooledConnectionLifetime = TimeSpan.FromMinutes(5),

                    // Every request to Steam carries the publisher key and an auth ticket in its query
                    // string. Following a redirect would resend both to whichever host the 3xx names, and
                    // would do it underneath the retry budget where nothing counts it.
                    AllowAutoRedirect = false,
                })
                // The factory pipeline installs its own loggers, and they write "Sending HTTP request GET
                // {Uri}" at Information - where that Uri is the full query string, key and ticket
                // included. There is no way to redact it in place, so the loggers go entirely; the client
                // logs the path, the status and the elapsed time itself.
                .RemoveAllLoggers();

            configure?.Invoke(builder);
            return services;
        }
    }
}
