using System.Globalization;
using System.Net;
using System.Net.Http;
using System.Text.Json;
using System.Text.RegularExpressions;
using HexWars.NetServer.Configuration;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Options;

namespace HexWars.NetServer.Steam
{
    /// <summary>
    /// The Steam partner Web API, reduced to the three calls the match host makes. Three rules shape it:
    /// nothing that can be rejected locally costs a Valve call, nothing that reaches a log or an exception
    /// carries the publisher key or the auth ticket, and only the idempotent reads are retried.
    /// </summary>
    public sealed class SteamWebApiClient : ISteamWebApiClient
    {
        /// <summary>Must match the identity the client passed to GetAuthTicketForWebApi, or Valve refuses.</summary>
        public const string TicketIdentity = "hexwars-match";

        internal const string AuthenticatePath = "ISteamUserAuth/AuthenticateUserTicket/v1/";
        internal const string OwnershipPath = "ISteamUser/CheckAppOwnership/v2/";
        internal const string LobbyPath = "ILobbyMatchmakingService/GetLobbyData/v1/";

        const int MaxRetries = 2;
        const int MaxTicketLength = 4096;
        static readonly TimeSpan BaseBackoff = TimeSpan.FromMilliseconds(200);
        static readonly TimeSpan MaxHonouredRetryAfter = TimeSpan.FromSeconds(2);

        static readonly Regex HexOnly = new(
            @"^[0-9A-Fa-f]+$", RegexOptions.CultureInvariant | RegexOptions.Compiled);

        readonly HttpClient _http;
        readonly SteamOptions _options;
        readonly ILogger<SteamWebApiClient> _logger;
        readonly TimeProvider _time;

        public SteamWebApiClient(
            HttpClient http,
            IOptions<SteamOptions> options,
            ILogger<SteamWebApiClient> logger,
            TimeProvider? time = null)
        {
            _http = http ?? throw new ArgumentNullException(nameof(http));
            _options = (options ?? throw new ArgumentNullException(nameof(options))).Value;
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
            _time = time ?? TimeProvider.System;
            DelayAsync = (delay, ct) => Task.Delay(delay, _time, ct);
        }

        /// <summary>The retry sleep, injectable so the retry tests cost no wall-clock time.</summary>
        internal Func<TimeSpan, CancellationToken, Task> DelayAsync { get; set; }

        // ---- public surface -------------------------------------------------

        public async Task<SteamIdentity> AuthenticateUserTicketAsync(string ticketHex, CancellationToken ct)
        {
            // A ticket that cannot possibly be valid never becomes a URL and never costs a Valve call.
            if (!IsWellFormedTicket(ticketHex))
            {
                throw new SteamApiException(
                    SteamFailure.AuthenticationFailed, "ticket was not a well formed hex string");
            }

            var uri = BuildUri(
                AuthenticatePath,
                ("key", _options.PublisherWebApiKey),
                ("appid", AppId),
                ("ticket", ticketHex),
                ("identity", TicketIdentity));

            // Ticket auth is never retried: a ticket is single use, so a retry can only turn a transient
            // blip into a hard rejection of a ticket that Valve has already consumed.
            using var document = await SendAsync(uri, allowRetry: false, notFound: null, ct).ConfigureAwait(false);

            var response = RequireObject(document.RootElement, "response");

            if (TryGetObject(response, "error", out var error))
            {
                var code = ReadScalar(error, "errorcode") ?? "unknown";
                throw new SteamApiException(
                    SteamFailure.AuthenticationFailed, "steam rejected the ticket (errorcode " + code + ")");
            }

            var parameters = RequireObject(response, "params");

            if (!TryReadSteamId(parameters, "steamid", out var steamId))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, "response.params.steamid was missing or not a SteamID64");
            }

            // Without Family Sharing Valve omits ownersteamid; the player then owns the game themselves.
            var owner = TryReadSteamId(parameters, "ownersteamid", out var ownerId) ? ownerId : steamId;

            return new SteamIdentity(
                steamId, owner, ReadBool(parameters, "vacbanned"), ReadBool(parameters, "publisherbanned"));
        }

        public async Task<bool> CheckAppOwnershipAsync(string steamId, CancellationToken ct)
        {
            if (!SteamId64.TryNormalize(steamId, out var canonical))
            {
                throw new SteamApiException(SteamFailure.AuthenticationFailed, "steamid was not a SteamID64");
            }

            var uri = BuildUri(
                OwnershipPath,
                ("key", _options.PublisherWebApiKey),
                ("steamid", canonical),
                ("appid", AppId));

            using var document = await SendAsync(uri, allowRetry: true, notFound: null, ct).ConfigureAwait(false);

            var ownership = RequireObject(document.RootElement, "appownership");
            if (!TryReadBool(ownership, "ownsapp", out var ownsApp))
            {
                throw new SteamApiException(SteamFailure.MalformedResponse, "appownership.ownsapp was missing");
            }

            return ownsApp;
        }

        public async Task<SteamLobbySnapshot> GetLobbyDataAsync(string lobbyId, CancellationToken ct)
        {
            // A lobby ID is a SteamID64 of the chat type, so it does not satisfy the individual-account
            // floor SteamId64 enforces; it still has to be a plain non-zero decimal.
            if (!TryNormaliseLobbyId(lobbyId, out var canonicalLobby))
            {
                throw new SteamApiException(SteamFailure.LobbyChanged, "lobby not found");
            }

            var uri = BuildUri(
                LobbyPath,
                ("key", _options.PublisherWebApiKey),
                ("appid", AppId),
                ("steamid_lobby", canonicalLobby));

            using var document = await SendAsync(
                uri, allowRetry: true, notFound: SteamFailure.LobbyChanged, ct).ConfigureAwait(false);

            if (!TryGetObject(document.RootElement, "response", out var response))
            {
                throw new SteamApiException(SteamFailure.MalformedResponse, "the response object was missing");
            }

            // A lobby that has been disbanded comes back as an empty object rather than a 404.
            if (!response.EnumerateObject().Any() || TryGetObject(response, "error", out _))
            {
                throw new SteamApiException(SteamFailure.LobbyChanged, "lobby not found");
            }

            // Valve does not echo steamid_lobby on this endpoint, so the requested id is the fallback.
            var lobby = TryReadLobbyId(response, "steamid_lobby", out var echoed) ? echoed : canonicalLobby;

            if (!TryReadSteamId(response, "steamid_owner", out var owner))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, "response.steamid_owner was missing or not a SteamID64");
            }

            var members = new List<SteamLobbyMember>();
            if (TryGetArray(response, "members", out var memberArray))
            {
                foreach (var member in memberArray.EnumerateArray())
                {
                    if (member.ValueKind != JsonValueKind.Object) continue;

                    // Dropping an unreadable member would silently deny a player their seat, so fail closed.
                    if (!TryReadSteamId(member, "steamid", out var memberId))
                    {
                        throw new SteamApiException(
                            SteamFailure.MalformedResponse, "a lobby member had no usable steamid");
                    }

                    members.Add(new SteamLobbyMember(
                        memberId, ReadKeyValues(member, "member_metadata", "member_data")));
                }
            }

            return new SteamLobbySnapshot(
                lobby, owner, members, ReadKeyValues(response, "lobby_metadata", "lobby_data"));
        }

        // ---- transport ------------------------------------------------------

        string AppId => _options.AppId.ToString(CultureInfo.InvariantCulture);

        static bool IsWellFormedTicket(string ticketHex) =>
            !string.IsNullOrEmpty(ticketHex)
            && ticketHex.Length >= 2
            && ticketHex.Length <= MaxTicketLength
            && ticketHex.Length % 2 == 0
            && HexOnly.IsMatch(ticketHex);

        internal static bool TryNormaliseLobbyId(string? raw, out string canonical)
        {
            canonical = string.Empty;
            if (string.IsNullOrWhiteSpace(raw)) return false;

            var trimmed = raw.Trim();
            if (trimmed.Length == 0 || trimmed.Length > 20) return false;

            foreach (var c in trimmed)
            {
                if (!char.IsAsciiDigit(c)) return false;
            }

            if (!ulong.TryParse(trimmed, NumberStyles.None, CultureInfo.InvariantCulture, out var value)) return false;
            if (value == 0) return false;

            canonical = value.ToString(CultureInfo.InvariantCulture);
            return true;
        }

        Uri BuildUri(string relativePath, params (string Name, string Value)[] query)
        {
            var basePath = _options.WebApiBaseUrl.AbsolutePath;
            if (!basePath.EndsWith("/", StringComparison.Ordinal)) basePath += "/";

            var builder = new UriBuilder(_options.WebApiBaseUrl)
            {
                Path = basePath + relativePath,
                Query = string.Join(
                    "&",
                    query.Select(q => Uri.EscapeDataString(q.Name) + "=" + Uri.EscapeDataString(q.Value))),
            };
            return builder.Uri;
        }

        /// <summary>
        /// One GET plus its retry budget. Returns the parsed body on success and throws a mapped
        /// SteamApiException on every failure; the caller never sees a status code.
        /// </summary>
        async Task<JsonDocument> SendAsync(Uri uri, bool allowRetry, SteamFailure? notFound, CancellationToken ct)
        {
            for (var attempt = 0; ; attempt++)
            {
                HttpStatusCode status = 0;
                var body = string.Empty;
                Exception? transport = null;
                TimeSpan? retryAfter = null;
                var started = _time.GetTimestamp();

                try
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, uri);
                    using var response = await _http.SendAsync(request, ct).ConfigureAwait(false);
                    status = response.StatusCode;
                    retryAfter = ReadRetryAfter(response);
                    body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                }
                catch (OperationCanceledException) when (ct.IsCancellationRequested)
                {
                    throw;   // the caller gave up; that is not a Steam failure
                }
                catch (OperationCanceledException ex)
                {
                    transport = ex;   // the HttpClient timeout surfaces as a cancellation
                }
                catch (HttpRequestException ex)
                {
                    transport = ex;
                }

                // Never the full URL: the query string carries the publisher key and the auth ticket.
                _logger.LogInformation(
                    "{Method} {Path} -> {Status} in {Ms}ms",
                    "GET",
                    uri.AbsolutePath,
                    transport is null ? ((int)status).ToString(CultureInfo.InvariantCulture) : "transport-failure",
                    (long)_time.GetElapsedTime(started).TotalMilliseconds);

                if (transport is null)
                {
                    if (notFound.HasValue && status == HttpStatusCode.NotFound)
                    {
                        throw new SteamApiException(notFound.Value, "lobby not found");
                    }

                    if (status == HttpStatusCode.Unauthorized || status == HttpStatusCode.Forbidden)
                    {
                        // An operator problem, not a player problem: it will not fix itself on a retry.
                        _logger.LogError(
                            "Steam publisher key rejected on {Path} with status {Status}",
                            uri.AbsolutePath, (int)status);
                        throw new SteamApiException(
                            SteamFailure.ServiceUnavailable,
                            "steam rejected the publisher key (HTTP " + (int)status + ")");
                    }

                    if ((int)status < 400)
                    {
                        return ParseJson(body);
                    }
                }

                var transient = transport is not null
                    || status == HttpStatusCode.TooManyRequests
                    || (int)status >= 500;

                if (transient && allowRetry && attempt < MaxRetries)
                {
                    await DelayAsync(NextDelay(attempt, retryAfter), ct).ConfigureAwait(false);
                    continue;
                }

                throw Terminal(status, transport, attempt + 1);
            }
        }

        static SteamApiException Terminal(HttpStatusCode status, Exception? transport, int attempts)
        {
            if (transport is not null)
            {
                // Transport messages quote the request they failed on, so they are redacted before use.
                return new SteamApiException(
                    SteamFailure.ServiceUnavailable,
                    "transport failure after " + attempts + " attempt(s): " + SteamLogRedaction.Redact(transport.Message),
                    transport);
            }

            var failure = status == HttpStatusCode.TooManyRequests
                ? SteamFailure.RateLimited
                : SteamFailure.ServiceUnavailable;

            return new SteamApiException(failure, "HTTP " + (int)status + " after " + attempts + " attempt(s)");
        }

        TimeSpan? ReadRetryAfter(HttpResponseMessage response)
        {
            var header = response.Headers.RetryAfter;
            if (header is null) return null;
            if (header.Delta.HasValue) return header.Delta.Value;
            if (header.Date.HasValue) return header.Date.Value - _time.GetUtcNow();
            return null;
        }

        static TimeSpan NextDelay(int attempt, TimeSpan? retryAfter)
        {
            if (retryAfter.HasValue)
            {
                // Honour the server, but a hostile or careless Retry-After must not park a request thread.
                var honoured = retryAfter.Value;
                if (honoured < TimeSpan.Zero) honoured = TimeSpan.Zero;
                return honoured > MaxHonouredRetryAfter ? MaxHonouredRetryAfter : honoured;
            }

            var backoff = TimeSpan.FromMilliseconds(BaseBackoff.TotalMilliseconds * Math.Pow(2, attempt));
            return backoff + TimeSpan.FromMilliseconds(Random.Shared.Next(0, 101));
        }

        static JsonDocument ParseJson(string body)
        {
            try
            {
                return JsonDocument.Parse(body);
            }
            catch (JsonException ex)
            {
                throw new SteamApiException(SteamFailure.MalformedResponse, "the response body was not JSON", ex);
            }
        }

        // ---- lenient JSON reading -------------------------------------------

        static bool TryGetProperty(JsonElement parent, string name, out JsonElement value)
        {
            value = default;
            return parent.ValueKind == JsonValueKind.Object && parent.TryGetProperty(name, out value);
        }

        static bool TryGetObject(JsonElement parent, string name, out JsonElement value) =>
            TryGetProperty(parent, name, out value) && value.ValueKind == JsonValueKind.Object;

        static bool TryGetArray(JsonElement parent, string name, out JsonElement value) =>
            TryGetProperty(parent, name, out value) && value.ValueKind == JsonValueKind.Array;

        static JsonElement RequireObject(JsonElement parent, string name)
        {
            if (!TryGetObject(parent, name, out var value))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, name + " was missing from the response");
            }
            return value;
        }

        /// <summary>Steam sends ids as JSON strings on some endpoints and JSON numbers on others.</summary>
        static string? ScalarToString(JsonElement value) => value.ValueKind switch
        {
            JsonValueKind.String => value.GetString(),
            JsonValueKind.Number => value.GetRawText(),
            JsonValueKind.True => "true",
            JsonValueKind.False => "false",
            _ => null,
        };

        static string? ReadScalar(JsonElement parent, string name) =>
            TryGetProperty(parent, name, out var value) ? ScalarToString(value) : null;

        static bool TryReadSteamId(JsonElement parent, string name, out string canonical)
        {
            canonical = string.Empty;
            return TryGetProperty(parent, name, out var value)
                && SteamId64.TryNormalize(ScalarToString(value), out canonical);
        }

        static bool TryReadLobbyId(JsonElement parent, string name, out string canonical)
        {
            canonical = string.Empty;
            return TryGetProperty(parent, name, out var value)
                && TryNormaliseLobbyId(ScalarToString(value), out canonical);
        }

        static bool TryReadBool(JsonElement parent, string name, out bool result)
        {
            result = false;
            if (!TryGetProperty(parent, name, out var value)) return false;

            switch (value.ValueKind)
            {
                case JsonValueKind.True:
                    result = true;
                    return true;
                case JsonValueKind.False:
                    return true;
                case JsonValueKind.Number:
                    result = value.TryGetInt64(out var number) && number != 0;
                    return true;
                case JsonValueKind.String:
                    var text = value.GetString();
                    if (text is null) return false;
                    if (bool.TryParse(text, out var parsed))
                    {
                        result = parsed;
                        return true;
                    }
                    if (long.TryParse(text, NumberStyles.Integer, CultureInfo.InvariantCulture, out var numeric))
                    {
                        result = numeric != 0;
                        return true;
                    }
                    return false;
                default:
                    return false;
            }
        }

        static bool ReadBool(JsonElement parent, string name) => TryReadBool(parent, name, out var value) && value;

        /// <summary>
        /// Steam expresses key/value bags three ways depending on the endpoint and the era of the docs:
        /// an array of key_name/key_value pairs (what ILobbyMatchmakingService documents), an array of
        /// key/value pairs, or a plain JSON object. All three land in the same dictionary.
        /// </summary>
        static IReadOnlyDictionary<string, string> ReadKeyValues(JsonElement parent, params string[] names)
        {
            var result = new Dictionary<string, string>(StringComparer.Ordinal);

            foreach (var name in names)
            {
                if (!TryGetProperty(parent, name, out var value)) continue;

                if (value.ValueKind == JsonValueKind.Object)
                {
                    foreach (var property in value.EnumerateObject())
                    {
                        var scalar = ScalarToString(property.Value);
                        if (scalar is not null) result[property.Name] = scalar;
                    }
                }
                else if (value.ValueKind == JsonValueKind.Array)
                {
                    foreach (var entry in value.EnumerateArray())
                    {
                        if (entry.ValueKind != JsonValueKind.Object) continue;

                        var key = ReadScalar(entry, "key_name") ?? ReadScalar(entry, "key") ?? ReadScalar(entry, "name");
                        var entryValue = ReadScalar(entry, "key_value") ?? ReadScalar(entry, "value");

                        if (!string.IsNullOrEmpty(key) && entryValue is not null) result[key] = entryValue;
                    }
                }
            }

            return result;
        }
    }
}
