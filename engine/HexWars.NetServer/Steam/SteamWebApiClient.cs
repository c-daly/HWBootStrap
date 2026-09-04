using System.Globalization;
using System.Net;
using System.Net.Http;
using System.Text;
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

        /// <summary>The ceiling on a response body. Steam answers are kilobytes; this is a wide margin.</summary>
        internal const int MaxResponseBytes = 256 * 1024;

        static readonly TimeSpan BaseBackoff = TimeSpan.FromMilliseconds(200);
        static readonly TimeSpan MaxHonouredRetryAfter = TimeSpan.FromSeconds(2);

        /// <summary>The shortest wait between attempts, so Retry-After: 0 cannot become a tight loop.</summary>
        static readonly TimeSpan MinBackoff = TimeSpan.FromMilliseconds(50);

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
                throw new SteamApiException(
                    SteamFailure.AuthenticationFailed, "steam rejected the ticket (" + DescribeErrorCode(error) + ")");
            }

            var parameters = RequireObject(response, "params");

            // The verdict is the result field, not the presence of a steamid. Reading it the other way
            // round is what lets a rejection that still echoes an account id pass as a sign-in.
            if (!IsResultOk(parameters))
            {
                throw new SteamApiException(
                    SteamFailure.AuthenticationFailed, "response.params.result was not OK");
            }

            if (!TryReadSteamId(parameters, "steamid", out var steamId))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, "response.params.steamid was missing or not a SteamID64");
            }

            // Without Family Sharing Valve omits ownersteamid and the player owns the game themselves. A
            // present one we cannot read is a different thing entirely, and quietly substituting steamid
            // for it would run the ownership check against the wrong account.
            var owner = steamId;
            if (TryGetProperty(parameters, "ownersteamid", out _))
            {
                if (!TryReadSteamId(parameters, "ownersteamid", out var ownerId))
                {
                    throw new SteamApiException(
                        SteamFailure.MalformedResponse, "response.params.ownersteamid was not a SteamID64");
                }

                owner = ownerId;
            }

            return new SteamIdentity(
                steamId,
                owner,
                RequireBool(parameters, "response.params", "vacbanned"),
                RequireBool(parameters, "response.params", "publisherbanned"));
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

            // result is the verdict on the whole answer, so it has to be there and it has to say OK. A
            // FAILED body that still carries an ownsapp field is not a licence check we may act on, and
            // neither is a body that never states a verdict at all.
            if (!IsResultOk(ownership))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, "appownership.result was missing or not OK");
            }

            // Strictly a JSON boolean: a 2, or the string "true", is not Valve answering this question.
            if (!TryReadStrictBool(ownership, "ownsapp", out var ownsApp))
            {
                throw new SteamApiException(
                    SteamFailure.MalformedResponse, "appownership.ownsapp was missing or not a boolean");
            }

            if (TryGetProperty(ownership, "ownersteamid", out _))
            {
                if (!TryReadSteamId(ownership, "ownersteamid", out var licenceOwner))
                {
                    throw new SteamApiException(
                        SteamFailure.MalformedResponse, "appownership.ownersteamid was not a SteamID64");
                }

                if (!string.Equals(licenceOwner, canonical, StringComparison.Ordinal))
                {
                    // Family Sharing is legitimate and stays allowed, but it is worth a line. Both ids
                    // are hashed so the log does not become a record of who lends games to whom.
                    _logger.LogInformation(
                        "App licence for {Player} is family shared from {Owner}",
                        SteamLogRedaction.HashSteamId(canonical),
                        SteamLogRedaction.HashSteamId(licenceOwner));
                }
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

            // Valve does not echo steamid_lobby on this endpoint, so the requested id is the answer. When
            // a response does echo one it has to be the lobby we asked about: an answer describing some
            // other lobby would seat that lobby roster into this match.
            if (TryGetProperty(response, "steamid_lobby", out var echoedValue))
            {
                if (!TryNormaliseLobbyId(ScalarToString(echoedValue), out var echoed) ||
                    !string.Equals(echoed, canonicalLobby, StringComparison.Ordinal))
                {
                    throw new SteamApiException(
                        SteamFailure.MalformedResponse, "the response echoed a different lobby id");
                }
            }

            var lobby = canonicalLobby;

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
                    // Dropping an unreadable member would silently shrink the roster, and a two-player
                    // lobby that arrives looking like a one-player lobby is how a seat goes missing.
                    if (member.ValueKind != JsonValueKind.Object)
                    {
                        throw new SteamApiException(
                            SteamFailure.MalformedResponse, "a lobby member was not an object");
                    }

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
                SteamApiException? bodyFailure = null;
                TimeSpan? retryAfter = null;
                var started = _time.GetTimestamp();

                try
                {
                    using var request = new HttpRequestMessage(HttpMethod.Get, uri);
                    // ResponseHeadersRead so the status is known before a single byte of body is buffered.
                    using var response = await _http
                        .SendAsync(request, HttpCompletionOption.ResponseHeadersRead, ct)
                        .ConfigureAwait(false);
                    status = response.StatusCode;
                    retryAfter = ReadRetryAfter(response);

                    // Only a body we are going to parse is worth reading. An error body is never used, so
                    // a failing Steam cannot make this process buffer anything at all.
                    if (IsSuccess(status))
                    {
                        body = await ReadBoundedBodyAsync(response, ct).ConfigureAwait(false);
                    }
                }
                catch (SteamApiException ex)
                {
                    bodyFailure = ex;   // an oversized body: a verdict, not a transient blip
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
                catch (IOException ex)
                {
                    // The body read can fail long after the headers arrived, and .NET surfaces that as an
                    // IOException (HttpIOException on a real connection) rather than an HttpRequestException.
                    // Unmapped it would leave a request handler facing a raw transport exception.
                    transport = ex;
                }

                // Never the full URL: the query string carries the publisher key and the auth ticket.
                _logger.LogInformation(
                    "{Method} {Path} -> {Status} in {Ms}ms",
                    "GET",
                    uri.AbsolutePath,
                    transport is null ? ((int)status).ToString(CultureInfo.InvariantCulture) : "transport-failure",
                    (long)_time.GetElapsedTime(started).TotalMilliseconds);

                if (bodyFailure is not null) throw bodyFailure;

                if (transport is null)
                {
                    // Automatic redirects are off, so a 3xx here is Steam pointing our key and our ticket
                    // at some other host. Following it by hand would resend both; retrying it would spend
                    // the budget on a request that can never succeed.
                    if (IsRedirect(status))
                    {
                        _logger.LogError(
                            "unexpected redirect from Steam on {Path} with status {Status}",
                            uri.AbsolutePath, (int)status);
                        throw new SteamApiException(
                            SteamFailure.ServiceUnavailable,
                            "unexpected redirect from Steam (HTTP " + (int)status + ")");
                    }

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

                    if (IsSuccess(status))
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
                // A transport message is free-form text written by whatever failed, and it has been near
                // a URL carrying the publisher key and the auth ticket. Redaction only catches the shapes
                // it knows - a secret named in prose walks straight through it - so none of the message
                // is copied. What survives is the type name and the structured fields .NET exposes, and
                // the cause is not attached either, because ToString() would render its message verbatim.
                return new SteamApiException(
                    SteamFailure.ServiceUnavailable,
                    "transport failure after " + attempts + " attempt(s): " + DescribeTransport(transport));
            }

            var failure = status == HttpStatusCode.TooManyRequests
                ? SteamFailure.RateLimited
                : SteamFailure.ServiceUnavailable;

            return new SteamApiException(failure, "HTTP " + (int)status + " after " + attempts + " attempt(s)");
        }

        /// <summary>
        /// Everything an operator can act on that is not free-form text: the exception type, and for an
        /// HttpRequestException the two fields .NET fills in itself - an enum and a status code, neither
        /// of which can carry a secret.
        /// </summary>
        static string DescribeTransport(Exception transport)
        {
            var described = transport.GetType().Name;

            if (transport is HttpRequestException http)
            {
                described += " (" + http.HttpRequestError;
                if (http.StatusCode.HasValue)
                {
                    described += ", HTTP " + ((int)http.StatusCode.Value).ToString(CultureInfo.InvariantCulture);
                }

                described += ")";
            }

            return described;
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
                // Honour the server, but a hostile or careless Retry-After must not park a request
                // thread, and a Retry-After of zero must not turn the retry budget into a tight loop
                // against a Steam that is already struggling. Jittered like the backoff path, so a fleet
                // told to wait the same second does not come back as one wave.
                var honoured = retryAfter.Value;
                if (honoured < MinBackoff) honoured = MinBackoff;

                // The cap is on the total wait. Applied before the jitter it is not a cap at all: the
                // thread still parks for the ceiling plus whatever the jitter adds on top.
                var total = honoured + Jitter();
                return total > MaxHonouredRetryAfter ? MaxHonouredRetryAfter : total;
            }

            var backoff = TimeSpan.FromMilliseconds(BaseBackoff.TotalMilliseconds * Math.Pow(2, attempt));
            return backoff + Jitter();
        }

        static TimeSpan Jitter() => TimeSpan.FromMilliseconds(Random.Shared.Next(0, 101));

        static bool IsSuccess(HttpStatusCode status) => (int)status >= 200 && (int)status < 300;

        static bool IsRedirect(HttpStatusCode status) => (int)status >= 300 && (int)status < 400;

        /// <summary>
        /// Reads a response body with a hard ceiling. A Steam answer is a few kilobytes; a multi-megabyte
        /// one is a proxy error page or an attempt to make this process allocate, and in both cases the
        /// right move is to stop reading rather than to buffer it all and then decide.
        /// </summary>
        static async Task<string> ReadBoundedBodyAsync(HttpResponseMessage response, CancellationToken ct)
        {
            var declared = response.Content.Headers.ContentLength;
            if (declared.HasValue && declared.Value > MaxResponseBytes) throw OversizedBody();

            using var stream = await response.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
            using var buffered = new MemoryStream();
            var chunk = new byte[8192];

            while (true)
            {
                var read = await stream.ReadAsync(chunk.AsMemory(), ct).ConfigureAwait(false);
                if (read == 0) break;
                if (buffered.Length + read > MaxResponseBytes) throw OversizedBody();
                buffered.Write(chunk, 0, read);
            }

            return Encoding.UTF8.GetString(buffered.GetBuffer(), 0, (int)buffered.Length);
        }

        static SteamApiException OversizedBody() => new(
            SteamFailure.MalformedResponse,
            "the response body exceeded " + MaxResponseBytes.ToString(CultureInfo.InvariantCulture) + " bytes");

        static JsonDocument ParseJson(string body)
        {
            try
            {
                return JsonDocument.Parse(body);
            }
            catch (JsonException)
            {
                // The parser message quotes the body it choked on, so only the verdict is kept.
                throw new SteamApiException(SteamFailure.MalformedResponse, "the response body was not JSON");
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

        /// <summary>
        /// A flag is a JSON boolean and nothing else. The lenient reading this replaces - 0/1, "true",
        /// "1" - meant a field Valve never sends as a string could be forged into a false negative on a
        /// ban check by anything that could shape the body, and a 2 read as true.
        /// </summary>
        static bool TryReadStrictBool(JsonElement parent, string name, out bool result)
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
                default:
                    return false;
            }
        }

        /// <summary>
        /// A flag that has to be there. Treating an absent one as false is how a truncated or shaped body
        /// signs in an account whose ban state Valve never actually stated, so absence fails closed too.
        /// </summary>
        static bool RequireBool(JsonElement parent, string path, string name)
        {
            if (TryReadStrictBool(parent, name, out var value)) return value;

            throw new SteamApiException(
                SteamFailure.MalformedResponse, path + "." + name + " was missing or not a boolean");
        }

        /// <summary>True only for a literal string result of OK. Valve spells success exactly one way.</summary>
        static bool IsResultOk(JsonElement parent) =>
            TryGetProperty(parent, "result", out var value)
            && value.ValueKind == JsonValueKind.String
            && string.Equals(value.GetString(), "OK", StringComparison.Ordinal);

        /// <summary>
        /// The one part of a Valve error object that may be copied into an operator detail: a number.
        /// Anything else is echoed content, and echoed content is where a ticket ends up in a log.
        /// </summary>
        static string DescribeErrorCode(JsonElement error)
        {
            if (TryGetProperty(error, "errorcode", out var value))
            {
                if (value.ValueKind == JsonValueKind.Number && value.TryGetInt64(out var numeric))
                {
                    return "errorcode " + numeric.ToString(CultureInfo.InvariantCulture);
                }

                if (value.ValueKind == JsonValueKind.String &&
                    long.TryParse(value.GetString(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed))
                {
                    return "errorcode " + parsed.ToString(CultureInfo.InvariantCulture);
                }
            }

            return "error object";
        }

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
