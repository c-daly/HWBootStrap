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
        public static IServiceCollection AddSteamWebApi(this IServiceCollection services)
        {
            services.TryAddSingleton(TimeProvider.System);

            services.AddHttpClient<ISteamWebApiClient, SteamWebApiClient>((provider, client) =>
            {
                var options = provider.GetRequiredService<IOptions<SteamOptions>>().Value;

                // A base address without a trailing slash silently drops its last path segment when a
                // relative URI is resolved against it.
                var baseUrl = options.WebApiBaseUrl.AbsoluteUri.EndsWith("/", StringComparison.Ordinal)
                    ? options.WebApiBaseUrl
                    : new Uri(options.WebApiBaseUrl.AbsoluteUri + "/");

                client.BaseAddress = baseUrl;
                client.Timeout = TimeSpan.FromSeconds(Math.Max(1, options.RequestTimeoutSeconds));
            })
            .ConfigurePrimaryHttpMessageHandler(() => new SocketsHttpHandler
            {
                ConnectTimeout = TimeSpan.FromSeconds(3),
                PooledConnectionLifetime = TimeSpan.FromMinutes(5),
            });

            return services;
        }
    }
}
