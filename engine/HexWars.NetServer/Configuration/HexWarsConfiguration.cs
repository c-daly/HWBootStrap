using System.Globalization;
using HexWars.NetServer.Persistence;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Hosting;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Configuration
{
    /// <summary>The parsed configuration plus every reason it is unusable. An error is always
    /// "KEY: reason" and NEVER contains the configured value, so it is safe to log and safe to print.</summary>
    public sealed record ConfigurationResult(SteamOptions Steam, MatchHostingOptions Match, IReadOnlyList<string> Errors)
    {
        public bool IsValid => Errors.Count == 0;

        public IReadOnlyList<string> ErrorsFor(IReadOnlyCollection<string> keys) =>
            Errors.Where(e => keys.Contains(KeyOf(e), StringComparer.Ordinal)).ToArray();

        public IReadOnlyList<string> ErrorsExcept(IReadOnlyCollection<string> keys) =>
            Errors.Where(e => !keys.Contains(KeyOf(e), StringComparer.Ordinal)).ToArray();

        static string KeyOf(string error)
        {
            int colon = error.IndexOf(":", StringComparison.Ordinal);
            return colon < 0 ? error : error.Substring(0, colon);
        }
    }

    /// <summary>
    /// Binds the flat environment-variable surface (STEAM_*, DATABASE_URL, MATCH_*, LOBBY_PROVIDER,
    /// ALLOWED_WEB_ORIGINS) into <see cref="SteamOptions"/> and <see cref="MatchHostingOptions"/> and
    /// refuses to start on a half-configured Steam or Production deployment. Flat names, not sections,
    /// because every target platform (Render, Fly, docker run, a shell) sets plain environment variables.
    /// </summary>
    public static class HexWarsConfiguration
    {
        public const string SteamAppIdKey = "STEAM_APP_ID";
        public const string SteamPublisherWebApiKeyKey = "STEAM_PUBLISHER_WEB_API_KEY";
        public const string SteamWebApiBaseUrlKey = "STEAM_WEB_API_BASE_URL";
        public const string DatabaseUrlKey = "DATABASE_URL";
        public const string MatchPublicBaseUrlKey = "MATCH_PUBLIC_BASE_URL";
        public const string MatchJoinTokenTtlSecondsKey = "MATCH_JOIN_TOKEN_TTL_SECONDS";
        public const string MatchBuildIdKey = "MATCH_BUILD_ID";
        public const string MatchProtocolVersionKey = "MATCH_PROTOCOL_VERSION";
        public const string AllowedWebOriginsKey = "ALLOWED_WEB_ORIGINS";
        public const string LobbyProviderKey = "LOBBY_PROVIDER";
        public const string MatchCompatibleClientBuildsKey = "MATCH_COMPATIBLE_CLIENT_BUILDS";
        public const string MatchTrustForwardedHeadersKey = "MATCH_TRUST_FORWARDED_HEADERS";
        public const string MatchBlockedSteamIdsKey = "MATCH_BLOCKED_STEAM_IDS";
        public const string MatchMetricsTokenKey = "MATCH_METRICS_TOKEN";

        /// <summary>Keys whose failures belong to <see cref="SteamOptions"/> rather than the match host.</summary>
        public static readonly string[] SteamKeys =
            { SteamAppIdKey, SteamPublisherWebApiKeyKey, SteamWebApiBaseUrlKey };

        /// <summary>Substrings that mean somebody copied the sample file and never filled it in. Cheaper to
        /// fail at boot than to discover it when the first real player tries to sign in.</summary>
        static readonly string[] PlaceholderTokens = { "changeme", "placeholder", "your-", "xxx", "todo", "example" };

        /// <summary>Registers both options objects with validation that runs before the host serves traffic.</summary>
        public static IServiceCollection AddHexWarsOptions(
            this IServiceCollection services, IConfiguration config, IHostEnvironment env)
        {
            var result = Read(config, env);
            services.AddSingleton(result);

            services.AddOptions<SteamOptions>().Configure(o => CopyInto(result.Steam, o)).ValidateOnStart();
            services.AddSingleton<IValidateOptions<SteamOptions>>(
                new PrecomputedValidator<SteamOptions>(result.ErrorsFor(SteamKeys)));

            services.AddOptions<MatchHostingOptions>().Configure(o => CopyInto(result.Match, o)).ValidateOnStart();
            services.AddSingleton<IValidateOptions<MatchHostingOptions>>(
                new PrecomputedValidator<MatchHostingOptions>(result.ErrorsExcept(SteamKeys)));

            return services;
        }

        /// <summary>Parse and validate in one pass. Pure: no DI, no host, no I/O, so a test can assert on
        /// exactly the error list an operator would see.</summary>
        public static ConfigurationResult Read(IConfiguration config, IHostEnvironment env)
        {
            var errors = new List<string>();
            var steam = new SteamOptions();
            var match = new MatchHostingOptions();

            string? appIdRaw = Value(config, SteamAppIdKey);
            string? publisherKey = Value(config, SteamPublisherWebApiKeyKey);
            string? webApiBaseRaw = Value(config, SteamWebApiBaseUrlKey);
            string? databaseUrlRaw = Value(config, DatabaseUrlKey);
            string? publicBaseRaw = Value(config, MatchPublicBaseUrlKey);
            string? buildIdRaw = Value(config, MatchBuildIdKey);

            // A missing, unparseable or zero App ID all mean the same thing to an operator: not configured.
            if (appIdRaw is not null
                && uint.TryParse(appIdRaw, NumberStyles.None, CultureInfo.InvariantCulture, out uint appId))
                steam.AppId = appId;
            steam.PublisherWebApiKey = publisherKey ?? string.Empty;

            if (webApiBaseRaw is null)
                steam.WebApiBaseUrl = new Uri(SteamOptions.DefaultWebApiBaseUrl);
            else if (Uri.TryCreate(webApiBaseRaw, UriKind.Absolute, out Uri? webApiBase))
                steam.WebApiBaseUrl = webApiBase;
            else
                errors.Add(SteamWebApiBaseUrlKey + ": must be an absolute URL");

            match.DatabaseUrl = databaseUrlRaw ?? string.Empty;
            match.BuildId = buildIdRaw ?? string.Empty;
            match.MetricsToken = Value(config, MatchMetricsTokenKey);
            match.AllowedWebOrigins = ReadList(config, AllowedWebOriginsKey);
            match.CompatibleClientBuilds = ReadList(config, MatchCompatibleClientBuildsKey);
            match.BlockedSteamIds = ReadList(config, MatchBlockedSteamIdsKey);

            string? protocolRaw = Value(config, MatchProtocolVersionKey);
            if (protocolRaw is not null)
            {
                if (int.TryParse(protocolRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int protocol)
                    && protocol > 0)
                    match.ProtocolVersion = protocol;
                else
                    errors.Add(MatchProtocolVersionKey + ": must be a positive integer");
            }

            string? ttlRaw = Value(config, MatchJoinTokenTtlSecondsKey);
            if (ttlRaw is not null)
            {
                if (int.TryParse(ttlRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int ttl)
                    && ttl >= MatchHostingOptions.MinJoinTokenTtlSeconds
                    && ttl <= MatchHostingOptions.MaxJoinTokenTtlSeconds)
                    match.JoinTokenTtlSeconds = ttl;
                else
                    errors.Add(MatchJoinTokenTtlSecondsKey + ": must be an integer between "
                        + MatchHostingOptions.MinJoinTokenTtlSeconds + " and "
                        + MatchHostingOptions.MaxJoinTokenTtlSeconds);
            }

            string? trustRaw = Value(config, MatchTrustForwardedHeadersKey);
            if (trustRaw is not null)
            {
                if (TryParseBool(trustRaw, out bool trust)) match.TrustForwardedHeaders = trust;
                else errors.Add(MatchTrustForwardedHeadersKey + ": must be true, false, 1 or 0");
            }

            string? providerRaw = Value(config, LobbyProviderKey);
            if (providerRaw is not null)
            {
                var providers = LobbyProviders.None;
                foreach (string name in SplitList(providerRaw))
                {
                    LobbyProviders? named = NamedProvider(name);
                    if (named is null) { providers = LobbyProviders.None; break; }
                    providers |= named.Value;
                }
                if (providers == LobbyProviders.None) errors.Add(LobbyProviderKey + ": unknown provider name");
                else match.LobbyProvider = providers;
            }

            // Uri.TryCreate treats a rooted Unix path as an absolute file:// URI, so it is the scheme check
            // that actually rejects a value like /matches.
            if (publicBaseRaw is not null
                && Uri.TryCreate(publicBaseRaw, UriKind.Absolute, out Uri? publicBase)
                && (publicBase.Scheme == Uri.UriSchemeHttp || publicBase.Scheme == Uri.UriSchemeHttps))
                match.PublicBaseUrl = publicBase;

            // The Steam stack needs real credentials; so does anything calling itself Production, even while
            // it still only serves the legacy WebGL lobby.
            bool requiresProductionStack = match.LobbyProvider.HasFlag(LobbyProviders.Steam) || env.IsProduction();
            if (requiresProductionStack)
            {
                if (IsPlaceholder(appIdRaw)) errors.Add(SteamAppIdKey + ": placeholder value");
                else if (steam.AppId == 0) errors.Add(SteamAppIdKey + ": missing");

                if (publisherKey is null) errors.Add(SteamPublisherWebApiKeyKey + ": missing");
                else if (IsPlaceholder(publisherKey)) errors.Add(SteamPublisherWebApiKeyKey + ": placeholder value");

                if (databaseUrlRaw is null) errors.Add(DatabaseUrlKey + ": missing");
                else if (IsPlaceholder(databaseUrlRaw)) errors.Add(DatabaseUrlKey + ": placeholder value");
                else if (!IsUsableDatabaseUrl(databaseUrlRaw))
                    errors.Add(DatabaseUrlKey + ": not a valid PostgreSQL URL or connection string");

                if (publicBaseRaw is null) errors.Add(MatchPublicBaseUrlKey + ": missing");
                else if (IsPlaceholder(publicBaseRaw)) errors.Add(MatchPublicBaseUrlKey + ": placeholder value");
                else if (match.PublicBaseUrl is null) errors.Add(MatchPublicBaseUrlKey + ": must be an absolute http or https URL");
                else if (env.IsProduction()
                         && !string.Equals(match.PublicBaseUrl.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
                    errors.Add(MatchPublicBaseUrlKey + ": must use https in Production");

                if (buildIdRaw is null) errors.Add(MatchBuildIdKey + ": missing");
                else if (IsPlaceholder(buildIdRaw)) errors.Add(MatchBuildIdKey + ": placeholder value");
            }
            else if (publicBaseRaw is not null && match.PublicBaseUrl is null)
            {
                errors.Add(MatchPublicBaseUrlKey + ": must be an absolute http or https URL");
            }

            return new ConfigurationResult(steam, match, errors);
        }

        static LobbyProviders? NamedProvider(string name) => name.ToLowerInvariant() switch
        {
            "legacy" => LobbyProviders.Legacy,
            "steam" => LobbyProviders.Steam,
            _ => null,
        };

        static bool IsPlaceholder(string? value) =>
            value is not null && PlaceholderTokens.Any(t => value.Contains(t, StringComparison.OrdinalIgnoreCase));

        static bool IsUsableDatabaseUrl(string value)
        {
            try { DatabaseUrl.ToNpgsqlConnectionString(value); return true; }
            catch (FormatException) { return false; }
        }

        static bool TryParseBool(string raw, out bool value)
        {
            switch (raw.Trim().ToLowerInvariant())
            {
                case "true": case "1": value = true; return true;
                case "false": case "0": value = false; return true;
                default: value = false; return false;
            }
        }

        /// <summary>Absent, blank or whitespace-only all read as "not set".</summary>
        static string? Value(IConfiguration config, string key)
        {
            string? raw = config[key];
            if (raw is null) return null;
            raw = raw.Trim();
            return raw.Length == 0 ? null : raw;
        }

        static string[] ReadList(IConfiguration config, string key) => SplitList(Value(config, key) ?? string.Empty);

        static string[] SplitList(string raw) => raw
            .Split(",", StringSplitOptions.RemoveEmptyEntries)
            .Select(part => part.Trim())
            .Where(part => part.Length > 0)
            .ToArray();

        static void CopyInto(SteamOptions source, SteamOptions target)
        {
            target.AppId = source.AppId;
            target.PublisherWebApiKey = source.PublisherWebApiKey;
            target.WebApiBaseUrl = source.WebApiBaseUrl;
            target.RequestTimeoutSeconds = source.RequestTimeoutSeconds;
        }

        static void CopyInto(MatchHostingOptions source, MatchHostingOptions target)
        {
            target.DatabaseUrl = source.DatabaseUrl;
            target.PublicBaseUrl = source.PublicBaseUrl;
            target.JoinTokenTtlSeconds = source.JoinTokenTtlSeconds;
            target.BuildId = source.BuildId;
            target.ProtocolVersion = source.ProtocolVersion;
            target.AllowedWebOrigins = source.AllowedWebOrigins;
            target.LobbyProvider = source.LobbyProvider;
            target.CompatibleClientBuilds = source.CompatibleClientBuilds;
            target.TrustForwardedHeaders = source.TrustForwardedHeaders;
            target.BlockedSteamIds = source.BlockedSteamIds;
            target.MetricsToken = source.MetricsToken;
        }

        /// <summary>Environment variables cannot change under a running process, so the verdict is computed
        /// once at registration and simply replayed here — which keeps the exact same message on the
        /// startup failure and on the describe-environment output.</summary>
        sealed class PrecomputedValidator<TOptions> : IValidateOptions<TOptions> where TOptions : class
        {
            readonly IReadOnlyList<string> _errors;

            public PrecomputedValidator(IReadOnlyList<string> errors) => _errors = errors;

            public ValidateOptionsResult Validate(string? name, TOptions options) =>
                _errors.Count == 0 ? ValidateOptionsResult.Success : ValidateOptionsResult.Fail(_errors);
        }
    }
}
