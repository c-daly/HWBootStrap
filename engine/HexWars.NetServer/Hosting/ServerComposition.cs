using System.Text;
using HexWars.Engine;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Steam;
using Microsoft.AspNetCore.StaticFiles;
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
        public static WebApplicationBuilder AddHexWarsServer(this WebApplicationBuilder builder)
        {
            // Belt and braces with RemoveAllLoggers() on the Steam client: these categories log the full
            // request URI at Information, and for the Steam client that URI is the publisher key and the
            // auth ticket. Nothing this server needs from them is worth that risk on any deployment.
            builder.Logging.AddFilter("System.Net.Http.HttpClient", LogLevel.None);

            builder.Services.AddHexWarsOptions(builder.Configuration, builder.Environment);
            // Registered unconditionally: the typed client resolves its options lazily, so a
            // Legacy-only deployment with no Steam credentials is unaffected by it being here.
            builder.Services.AddSteamWebApi();
            // Stateless and options-driven: one instance serves every request, and resolving it here
            // rather than constructing it in an endpoint keeps the lobby rules out of the HTTP layer.
            builder.Services.AddSingleton<SteamLobbyValidator>();
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

            app.UseWebSockets();
            app.UseDefaultFiles();   // serve the WebGL client (index.html) from wwwroot/ when a deploy copies it in
            // Unity WebGL ships .unityweb/.data/.wasm; without these mappings Kestrel 404s them.
            var types = new FileExtensionContentTypeProvider();
            types.Mappings[".unityweb"] = "application/octet-stream"; // gzip payload; the loader decompresses (decompressionFallback)
            types.Mappings[".data"] = "application/octet-stream";
            types.Mappings[".wasm"] = "application/wasm";
            app.UseStaticFiles(new StaticFileOptions { ContentTypeProvider = types });
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

            return app;
        }
    }
}
