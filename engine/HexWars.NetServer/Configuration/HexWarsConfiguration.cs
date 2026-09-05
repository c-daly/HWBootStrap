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
        public const string MatchTrustedProxyCidrsKey = "MATCH_TRUSTED_PROXY_CIDRS";
        public const string MatchTrustAllProxiesKey = "MATCH_TRUST_ALL_PROXIES";
        public const string MatchBlockedSteamIdsKey = "MATCH_BLOCKED_STEAM_IDS";
        public const string MatchMetricsTokenKey = "MATCH_METRICS_TOKEN";
        public const string MatchLogPseudonymKeyKey = "MATCH_LOG_PSEUDONYM_KEY";
        public const string MatchHeartbeatSecondsKey = "MATCH_HEARTBEAT_SECONDS";
        public const string MatchStaleConnectionSecondsKey = "MATCH_STALE_CONNECTION_SECONDS";
        public const string MatchOutboundQueueCapacityKey = "MATCH_OUTBOUND_QUEUE_CAPACITY";
        public const string MatchAuthTimeoutSecondsKey = "MATCH_AUTH_TIMEOUT_SECONDS";
        public const string MatchTerminalReconnectSecondsKey = "MATCH_TERMINAL_RECONNECT_SECONDS";
        public const string MatchMaxSocketsPerIpKey = "MATCH_MAX_SOCKETS_PER_IP";
        public const string MatchOutboundQueueBytesKey = "MATCH_OUTBOUND_QUEUE_BYTES";
        public const string MatchCredentialRecheckSecondsKey = "MATCH_CREDENTIAL_RECHECK_SECONDS";
        public const string MatchMaxRechecksPerCadenceKey = "MATCH_MAX_RECHECKS_PER_CADENCE";

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

            // Optional: without one the process generates a random key, which is a working default and
            // only costs cross-restart correlation. A short one, though, is worse than none, because it
            // reads as protection while staying cheap to brute force.
            string? pseudonymKeyRaw = Value(config, MatchLogPseudonymKeyKey);
            if (pseudonymKeyRaw is not null)
            {
                if (pseudonymKeyRaw.Length >= MatchHostingOptions.MinLogPseudonymKeyLength)
                    match.LogPseudonymKey = pseudonymKeyRaw;
                else
                    errors.Add(MatchLogPseudonymKeyKey + ": must be at least "
                        + MatchHostingOptions.MinLogPseudonymKeyLength + " characters");
            }

            string? protocolRaw = Value(config, MatchProtocolVersionKey);
            if (protocolRaw is not null)
            {
                // A supported value rather than any positive integer. The number is written into every
                // match row and compared against the number a later host carries, so a typo here does not
                // fail now - it fails months from now, as every match this host wrote becoming
                // unrecoverable. There is no code path for a protocol this build does not speak.
                if (int.TryParse(protocolRaw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int protocol)
                    && ProtocolContract.SupportedVersions.Contains(protocol))
                    match.ProtocolVersion = protocol;
                else
                    errors.Add(MatchProtocolVersionKey + ": must be one of " + ProtocolContract.SupportedList
                        + ", the wire protocol(s) this build speaks");
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

            match.HeartbeatIntervalSeconds = BoundedInt(
                config, MatchHeartbeatSecondsKey,
                MatchHostingOptions.MinHeartbeatIntervalSeconds,
                MatchHostingOptions.MaxHeartbeatIntervalSeconds,
                MatchHostingOptions.DefaultHeartbeatIntervalSeconds, errors);

            match.StaleConnectionSeconds = BoundedInt(
                config, MatchStaleConnectionSecondsKey,
                MatchHostingOptions.MinStaleConnectionSeconds,
                MatchHostingOptions.MaxStaleConnectionSeconds,
                MatchHostingOptions.DefaultStaleConnectionSeconds, errors);

            match.OutboundQueueCapacity = BoundedInt(
                config, MatchOutboundQueueCapacityKey,
                MatchHostingOptions.MinOutboundQueueCapacity,
                MatchHostingOptions.MaxOutboundQueueCapacity,
                MatchHostingOptions.DefaultOutboundQueueCapacity, errors);

            match.AuthFrameTimeoutSeconds = BoundedInt(
                config, MatchAuthTimeoutSecondsKey,
                MatchHostingOptions.MinAuthFrameTimeoutSeconds,
                MatchHostingOptions.MaxAuthFrameTimeoutSeconds,
                MatchHostingOptions.DefaultAuthFrameTimeoutSeconds, errors);

            match.TerminalReconnectSeconds = BoundedInt(
                config, MatchTerminalReconnectSecondsKey,
                MatchHostingOptions.MinTerminalReconnectSeconds,
                MatchHostingOptions.MaxTerminalReconnectSeconds,
                MatchHostingOptions.DefaultTerminalReconnectSeconds, errors);

            match.MaxSocketsPerIp = BoundedInt(
                config, MatchMaxSocketsPerIpKey,
                MatchHostingOptions.MinMaxSocketsPerIp,
                MatchHostingOptions.MaxMaxSocketsPerIp,
                MatchHostingOptions.DefaultMaxSocketsPerIp, errors);

            match.OutboundQueueBytes = BoundedInt(
                config, MatchOutboundQueueBytesKey,
                MatchHostingOptions.MinOutboundQueueBytes,
                MatchHostingOptions.MaxOutboundQueueBytes,
                MatchHostingOptions.DefaultOutboundQueueBytes, errors);

            match.CredentialRecheckSeconds = BoundedInt(
                config, MatchCredentialRecheckSecondsKey,
                MatchHostingOptions.MinCredentialRecheckSeconds,
                MatchHostingOptions.MaxCredentialRecheckSeconds,
                MatchHostingOptions.DefaultCredentialRecheckSeconds, errors);

            match.MaxRechecksPerCadence = BoundedInt(
                config, MatchMaxRechecksPerCadenceKey,
                MatchHostingOptions.MinMaxRechecksPerCadence,
                MatchHostingOptions.MaxMaxRechecksPerCadence,
                MatchHostingOptions.DefaultMaxRechecksPerCadence, errors);

            // Checked as a pair rather than as two ranges, because either value alone can be perfectly
            // reasonable and the combination still closes healthy sockets: a window that is not longer than
            // the ping cadence judges silence over an interval the client was never given a chance to
            // answer in, and the symptom is players being disconnected mid-game for no visible reason.
            if (match.StaleConnectionSeconds <= match.HeartbeatIntervalSeconds)
                errors.Add(MatchStaleConnectionSecondsKey + ": must be greater than " + MatchHeartbeatSecondsKey);

            string? trustRaw = Value(config, MatchTrustForwardedHeadersKey);
            if (trustRaw is not null)
            {
                if (TryParseBool(trustRaw, out bool trust)) match.TrustForwardedHeaders = trust;
                else errors.Add(MatchTrustForwardedHeadersKey + ": must be true, false, 1 or 0");
            }

            string? trustAllRaw = Value(config, MatchTrustAllProxiesKey);
            if (trustAllRaw is not null)
            {
                if (TryParseBool(trustAllRaw, out bool trustAll)) match.TrustAllProxies = trustAll;
                else errors.Add(MatchTrustAllProxiesKey + ": must be true, false, 1 or 0");
            }

            match.TrustedProxyCidrs = ReadList(config, MatchTrustedProxyCidrsKey);

            // Trusting every peer is a real deployment on a platform that does not publish its proxy
            // addresses. It is also exactly what a half-finished configuration looks like, and the
            // consequence of guessing wrong is that any caller can pick their own rate-limit partition by
            // writing a header. So it has to be said rather than arrived at.
            if (match.TrustForwardedHeaders && match.TrustedProxyCidrs.Length == 0 && !match.TrustAllProxies)
            {
                errors.Add(MatchTrustForwardedHeadersKey + ": set " + MatchTrustedProxyCidrsKey
                    + " to the proxy addresses, or " + MatchTrustAllProxiesKey
                    + "=true to confirm this service is reachable only through its platform proxy");
            }

            foreach (string entry in match.TrustedProxyCidrs)
            {
                // Named rather than skipped. A typo here silently trusts nobody, which turns every
                // forwarded address into the proxy address and quietly rate-limits the whole deployment
                // as one caller - a failure that looks like load rather than like a configuration error.
                if (!Hosting.TrustedProxies.IsValid(entry))
                    errors.Add(MatchTrustedProxyCidrsKey + ": each entry must be an IP address or a CIDR range");
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
            // Credentials, a query string or a fragment on this value would be published: it is serialised
            // into the environment report and logged once at startup. Refuse them rather than redact them.
            string? publicBaseError = null;
            if (publicBaseRaw is not null)
            {
                if (Uri.TryCreate(publicBaseRaw, UriKind.Absolute, out Uri? publicBase)
                    && (publicBase.Scheme == Uri.UriSchemeHttp || publicBase.Scheme == Uri.UriSchemeHttps))
                {
                    if (publicBase.UserInfo.Length > 0
                        || publicBase.Query.Length > 0
                        || publicBase.Fragment.Length > 0)
                        publicBaseError = MatchPublicBaseUrlKey
                            + ": must not contain credentials, a query string, or a fragment";
                    else
                        match.PublicBaseUrl = publicBase;
                }
                else
                {
                    publicBaseError = MatchPublicBaseUrlKey + ": must be an absolute http or https URL";
                }
            }

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
                else if (publicBaseError is not null) errors.Add(publicBaseError);
                else if (env.IsProduction()
                         && !string.Equals(match.PublicBaseUrl!.Scheme, Uri.UriSchemeHttps, StringComparison.OrdinalIgnoreCase))
                    errors.Add(MatchPublicBaseUrlKey + ": must use https in Production");

                if (buildIdRaw is null) errors.Add(MatchBuildIdKey + ": missing");
                else if (IsPlaceholder(buildIdRaw)) errors.Add(MatchBuildIdKey + ": placeholder value");
            }
            else if (publicBaseError is not null)
            {
                errors.Add(publicBaseError);
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

        /// <summary>An optional integer key with a floor and a ceiling. An absent key keeps the default; a
        /// value that is not an integer and one that is out of range are the same mistake to an operator,
        /// so they get the same message naming the range rather than a parser's vocabulary.</summary>
        static int BoundedInt(
            IConfiguration config, string key, int min, int max, int fallback, List<string> errors)
        {
            string? raw = Value(config, key);
            if (raw is null) return fallback;

            if (int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed)
                && parsed >= min && parsed <= max)
                return parsed;

            errors.Add(key + ": must be an integer between " + min + " and " + max);
            return fallback;
        }

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
            target.TrustedProxyCidrs = source.TrustedProxyCidrs;
            target.TrustAllProxies = source.TrustAllProxies;
            target.BlockedSteamIds = source.BlockedSteamIds;
            target.MetricsToken = source.MetricsToken;
            target.LogPseudonymKey = source.LogPseudonymKey;
            target.HeartbeatIntervalSeconds = source.HeartbeatIntervalSeconds;
            target.StaleConnectionSeconds = source.StaleConnectionSeconds;
            target.OutboundQueueCapacity = source.OutboundQueueCapacity;
            target.AuthFrameTimeoutSeconds = source.AuthFrameTimeoutSeconds;
            target.TerminalReconnectSeconds = source.TerminalReconnectSeconds;
            target.MaxSocketsPerIp = source.MaxSocketsPerIp;
            target.OutboundQueueBytes = source.OutboundQueueBytes;
            target.CredentialRecheckSeconds = source.CredentialRecheckSeconds;
            target.MaxRechecksPerCadence = source.MaxRechecksPerCadence;
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
