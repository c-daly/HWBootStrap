using System.Text;
using System.Threading.RateLimiting;
using HexWars.Engine;
using HexWars.NetServer.Auth;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Endpoints;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using Microsoft.AspNetCore.HttpOverrides;
using Microsoft.AspNetCore.RateLimiting;
using Microsoft.AspNetCore.StaticFiles;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Hosting
{
    /// <summary>
    /// The whole server in two calls: AddHexWarsServer registers configuration and services on the
    /// builder, UseHexWarsServer wires middleware and endpoints on the built app. Keeping them apart is
    /// what lets an integration test replace a service between the two.
    /// </summary>
    public static class ServerComposition
    {
        /// <summary>How long a rate-limit budget lasts before it refills.</summary>
        public static readonly TimeSpan RateLimitWindow = TimeSpan.FromMinutes(1);

        /// <summary>Match creations one address may attempt per window.</summary>
        public const int CreatePermitsPerWindow = 5;

        /// <summary>Join attempts one address may make per window.</summary>
        public const int JoinPermitsPerWindow = 20;

        public static WebApplicationBuilder AddHexWarsServer(this WebApplicationBuilder builder)
        {
            // Belt and braces with RemoveAllLoggers() on the Steam client: these categories log the full
            // request URI at Information, and for the Steam client that URI is the publisher key and the
            // auth ticket. Scoped to that one client on purpose - only its requests carry secrets, and a
            // filter on the whole prefix would silently blind every other client this server ever adds.
            // The trailing dot matters: the category is matched by prefix, so without it this would also
            // silence any client whose name merely starts with this one, such as SteamWebApiMetrics.
            builder.Logging.AddFilter(
                "System.Net.Http.HttpClient." + SteamWebApiRegistration.HttpClientName + ".", LogLevel.None);

            builder.Services.AddHexWarsOptions(builder.Configuration, builder.Environment);

            // One clock for everything that needs one, so a test can wind it forward instead of waiting for
            // an expiry. TryAdd rather than Add: a host that has already supplied its own keeps it.
            builder.Services.TryAddSingleton(TimeProvider.System);

            // Read straight from configuration rather than the bound options: this runs before validation,
            // and a legacy deployment with no DATABASE_URL must not pull Npgsql into the container at all.
            if (!string.IsNullOrWhiteSpace(builder.Configuration["DATABASE_URL"]))
            {
                builder.Services.AddHexWarsPostgres();

                // Registered next to the store rather than unconditionally, because it cannot be built
                // without one. In Development the container is validated at build time, so registering it
                // for a legacy deployment with no database would turn a service nobody resolves into a
                // startup failure. Singleton for the same reason the store is: it holds no state itself.
                builder.Services.AddSingleton<IMatchCredentialService, MatchCredentialService>();
            }

            // Registered unconditionally: the typed client resolves its options lazily, so a
            // Legacy-only deployment with no Steam credentials is unaffected by it being here.
            builder.Services.AddSteamWebApi();
            // Stateless and options-driven: one instance serves every request, and resolving it here
            // rather than constructing it in an endpoint keeps the lobby rules out of the HTTP layer.
            builder.Services.AddSingleton<SteamLobbyValidator>();

            // Process-wide by necessity: a per-request throttle would count to one and never refuse.
            builder.Services.AddSingleton<AuthFailureThrottle>();

            builder.Services.AddRateLimiter(limiter =>
            {
                limiter.RejectionStatusCode = StatusCodes.Status429TooManyRequests;

                // The default rejection writes an empty body, which a client cannot tell apart from any
                // other 429 a proxy might have produced. Answering in the same shape as every other
                // refusal means the Unity client has one error path rather than two.
                limiter.OnRejected = async (context, token) =>
                {
                    context.HttpContext.Response.StatusCode = StatusCodes.Status429TooManyRequests;
                    await context.HttpContext.Response.WriteAsJsonAsync(
                        new ApiError(ApiErrors.RateLimited, SteamFailureMessages.RateLimited), token);
                };

                // Creating a match costs two Steam round trips and a write, so it is the expensive door
                // and gets the tight budget. Joining is cheap and is the call a reconnecting client
                // repeats, so it gets room to retry without a player being locked out of their own game.
                limiter.AddPolicy(SteamMatchEndpoints.CreateRateLimitPolicy, context =>
                    RateLimitPartition.GetFixedWindowLimiter(
                        SteamMatchEndpoints.CallerKey(context),
                        _ => new FixedWindowRateLimiterOptions
                        {
                            PermitLimit = CreatePermitsPerWindow,
                            Window = RateLimitWindow,

                            // Nothing is queued: a caller over budget is told so immediately rather than
                            // holding a connection open waiting for a permit it may never get.
                            QueueLimit = 0,
                        }));

                limiter.AddPolicy(SteamMatchEndpoints.JoinRateLimitPolicy, context =>
                    RateLimitPartition.GetFixedWindowLimiter(
                        SteamMatchEndpoints.CallerKey(context),
                        _ => new FixedWindowRateLimiterOptions
                        {
                            PermitLimit = JoinPermitsPerWindow,
                            Window = RateLimitWindow,
                            QueueLimit = 0,
                        }));
            });

            return builder;
        }

        public static WebApplication UseHexWarsServer(this WebApplication app)
        {
            // Resolving the options here is the fail-fast point: a half-configured Steam or Production
            // deployment throws OptionsValidationException naming the offending KEYS, never their values.
            var steam = app.Services.GetRequiredService<IOptions<SteamOptions>>().Value;
            var match = app.Services.GetRequiredService<IOptions<MatchHostingOptions>>().Value;

            // Before the first line is written: every Steam id that reaches a log goes through this key,
            // and a handle written under the startup default would not match the ones written after.
            if (!string.IsNullOrEmpty(match.LogPseudonymKey))
            {
                SteamLogRedaction.ConfigureKey(Encoding.UTF8.GetBytes(match.LogPseudonymKey));
            }

            app.Logger.LogInformation(
                "Environment report {Report}", EnvironmentReport.Describe(steam, match, app.Environment).ToJson());

            // First, before anything reads the connection: the rate limiter and the auth-failure
            // throttle both partition on the remote address, and behind a proxy every request carries the
            // proxy address until this runs. A single hop is trusted and the known-network lists are
            // cleared, which together mean exactly one X-Forwarded-For entry is honoured - the one the
            // proxy in front of this process appended. Off by default: a server reachable directly would
            // otherwise let any caller choose their own rate-limit partition with a header.
            if (match.TrustForwardedHeaders)
            {
                var forwarded = new ForwardedHeadersOptions
                {
                    ForwardedHeaders = ForwardedHeaders.XForwardedFor | ForwardedHeaders.XForwardedProto,
                    ForwardLimit = 1,
                };
                forwarded.KnownNetworks.Clear();
                forwarded.KnownProxies.Clear();
                app.UseForwardedHeaders(forwarded);
            }

            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client (index.html) from wwwroot/ when a deploy copies it in
            // Unity WebGL ships .unityweb/.data/.wasm; without these mappings Kestrel 404s them.
            var types = new FileExtensionContentTypeProvider();
            types.Mappings[".unityweb"] = "application/octet-stream"; // gzip payload; the loader decompresses (decompressionFallback)
            types.Mappings[".data"] = "application/octet-stream";
            types.Mappings[".wasm"] = "application/wasm";
            app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = types });

            // After routing (WebApplication inserts that first) and before the endpoints, so the limiter
            // can read the policy off the endpoint the request matched. Static files are already served
            // above: a rate limit on the WebGL bundle would throttle a page load, not an abuser.
            app.UseRateLimiter();

            app.MapGet("/healthz", () => "ok");

            if (match.LobbyProvider.HasFlag(LobbyProviders.Legacy))
            {
                // The lobby browser: open public games as JSON. Same origin as the WebGL client, no CORS.
                app.MapGet("/games", () =>
                {
                    IReadOnlyList<OpenGame> open = LegacyWebSocketServer.OpenGamesSnapshot();
                    return Results.Json(new
                    {
                        games = open.Select(g => new
                        {
                            code = g.Code,
                            mode = g.Setup.Mode.ToString(),
                            width = g.Setup.Width,
                            height = g.Setup.Height,
                            fog = g.Setup.Fog,
                            pace = g.Setup.TurnActions,
                            army = g.Setup.ArmySize,
                            ageSeconds = g.AgeSeconds,
                        }).ToArray(),
                    });
                });
                app.Map("/ws", LegacyWebSocketServer.Handle);
            }

            // Mapped only when the Steam provider is on. A Legacy-only deployment has no Steam
            // credentials and no database, so these routes could not serve a request; a 404 is a truer
            // answer than a 503 from a handler that was never going to work.
            if (match.LobbyProvider.HasFlag(LobbyProviders.Steam))
            {
                app.MapSteamMatchEndpoints();
            }

            return app;
        }
    }
}
