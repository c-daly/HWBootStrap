#nullable enable
using System;
using Newtonsoft.Json;

namespace HexWars.Presentation
{
    /// <summary>POST body for match allocation (lobby owner). Field names are the wire contract.</summary>
    public sealed class SteamCreateMatchRequest
    {
        [JsonProperty("steamLobbyId")] public string? SteamLobbyId { get; set; }

        [JsonProperty("ticket")] public string? Ticket { get; set; }

        [JsonProperty("requestedSetup")] public string? RequestedSetup { get; set; }
    }

    /// <summary>POST body for joining an allocated match (the other lobby member).</summary>
    public sealed class SteamJoinMatchRequest
    {
        [JsonProperty("ticket")] public string? Ticket { get; set; }
    }

    /// <summary>The 200 body both match-service calls return.</summary>
    public sealed class SteamMatchResponse
    {
        [JsonProperty("matchId")] public string? MatchId { get; set; }

        [JsonProperty("protocolVersion")] public int ProtocolVersion { get; set; }

        [JsonProperty("websocketUrl")] public string? WebsocketUrl { get; set; }

        [JsonProperty("seat")] public int Seat { get; set; }

        [JsonProperty("joinCredential")] public string? JoinCredential { get; set; }

        [JsonProperty("credentialExpiresAt")] public string? CredentialExpiresAt { get; set; }
    }

    /// <summary>The failure body: a machine code plus a message that is safe to show a player.</summary>
    public sealed class SteamMatchErrorResponse
    {
        [JsonProperty("error")] public string? Error { get; set; }

        [JsonProperty("message")] public string? Message { get; set; }
    }

    /// <summary>The shipped <c>Resources/HexWarsSteamConfig</c> asset.</summary>
    public sealed class SteamMatchConfigBody
    {
        [JsonProperty("matchBaseUrl")] public string? MatchBaseUrl { get; set; }

        [JsonProperty("protocolVersion")] public int ProtocolVersion { get; set; }
    }

    /// <summary>
    /// Everything about the match-service HTTP API that needs no Unity: the URLs, the request bodies,
    /// and the translation from (HTTP status, body) to a <see cref="SteamMatchApiResult"/>. Kept pure so
    /// the wire contract is covered by dotnet tests rather than only by a Unity play session.
    /// </summary>
    public static class SteamMatchApiContracts
    {
        public const string MatchesPath = "/api/v1/steam/matches";

        /// <summary>The service answered, but not in a shape this build can read.</summary>
        public const string MalformedErrorCode = "malformed";

        /// <summary>The request never reached the service (DNS, TLS, timeout, offline).</summary>
        public const string NetworkErrorCode = "network";

        /// <summary>This build has no match-service URL, so Steam play is switched off.</summary>
        public const string NotConfiguredErrorCode = "not_configured";

        /// <summary>The service handed out a socket URL that is not encrypted.</summary>
        public const string InsecureTransportErrorCode = "insecure_transport";

        public const string InsecureTransportMessage =
            "The match service offered an unencrypted connection, so it was refused.";

        public const string IncompatibleVersionMessage =
            "The match service speaks a different protocol version. Update HexWars in Steam.";

        public const string MalformedMessage =
            "The match service sent a reply this build could not read. Try again shortly.";

        public const string NetworkMessage =
            "Could not reach the match service. Check your connection and try again.";

        public const string NotConfiguredMessage =
            "Online play over Steam is not configured for this build.";

        public const string UnknownFailureMessage =
            "The match service refused that request. Try again shortly.";

        static readonly char[] TrailingSlash = "/".ToCharArray();

        /// <summary>Owner side: POST here to allocate a match for a lobby.</summary>
        public static string CreateMatchUrl(string? baseUrl)
        {
            return NormalizeBase(baseUrl) + MatchesPath;
        }

        /// <summary>Guest side: POST here to take the second seat of an allocated match.</summary>
        public static string JoinMatchUrl(string? baseUrl, string? matchId)
        {
            return CreateMatchUrl(baseUrl) + "/" + Uri.EscapeDataString(matchId ?? string.Empty) + "/join";
        }

        public static string CreateMatchBody(string? lobbyId, string? ticketHex, string? requestedSetupWire)
        {
            return JsonConvert.SerializeObject(new SteamCreateMatchRequest
            {
                SteamLobbyId = lobbyId ?? string.Empty,
                Ticket = ticketHex ?? string.Empty,
                RequestedSetup = requestedSetupWire ?? string.Empty,
            });
        }

        public static string JoinMatchBody(string? ticketHex)
        {
            return JsonConvert.SerializeObject(new SteamJoinMatchRequest { Ticket = ticketHex ?? string.Empty });
        }

        /// <summary>
        /// One HTTP exchange as a result. Status 0 means the request never landed; a 2xx must carry a
        /// match id, a socket URL and a credential to count as success; anything unreadable in either
        /// direction is reported as <see cref="MalformedErrorCode"/> rather than guessed at.
        /// </summary>
        public static SteamMatchApiResult Parse(long httpStatus, string? body, int expectedProtocolVersion)
        {
            if (httpStatus == 0) return SteamMatchApiResult.Failure(0, NetworkErrorCode, NetworkMessage);

            if (httpStatus >= 200 && httpStatus < 300)
            {
                var ok = TryRead<SteamMatchResponse>(body);
                if (ok == null
                    || string.IsNullOrEmpty(ok.MatchId)
                    || string.IsNullOrEmpty(ok.WebsocketUrl)
                    || string.IsNullOrEmpty(ok.JoinCredential))
                {
                    return SteamMatchApiResult.Failure(httpStatus, MalformedErrorCode, MalformedMessage);
                }

                // A service on another protocol version would deal frames this build cannot read. A
                // missing field arrives as 0, which is a mismatch too, not a free pass.
                var expected = expectedProtocolVersion > 0 ? expectedProtocolVersion : SteamMatchConfig.DefaultProtocolVersion;
                if (ok.ProtocolVersion != expected)
                {
                    return SteamMatchApiResult.Failure(
                        httpStatus, SteamMatchErrorCodes.IncompatibleVersion, IncompatibleVersionMessage);
                }

                // The AUTH frame carries a single-use credential, so the socket has to be encrypted
                // everywhere except a loopback development server.
                if (!IsSecureSocketUrl(ok.WebsocketUrl))
                {
                    return SteamMatchApiResult.Failure(httpStatus, InsecureTransportErrorCode, InsecureTransportMessage);
                }

                return SteamMatchApiResult.Success(ok.MatchId!, ok.WebsocketUrl!, ok.JoinCredential!, ok.Seat);
            }

            var failure = TryRead<SteamMatchErrorResponse>(body);
            if (failure == null || string.IsNullOrEmpty(failure.Error))
                return SteamMatchApiResult.Failure(httpStatus, MalformedErrorCode, MalformedMessage);

            var message = string.IsNullOrEmpty(failure.Message) ? UnknownFailureMessage : failure.Message!;
            return SteamMatchApiResult.Failure(httpStatus, failure.Error!, message);
        }

        static T? TryRead<T>(string? body) where T : class
        {
            if (string.IsNullOrWhiteSpace(body)) return null;
            try { return JsonConvert.DeserializeObject<T>(body!); }
            catch (JsonException) { return null; }
        }

        internal static string NormalizeBase(string? baseUrl)
        {
            return (baseUrl ?? string.Empty).Trim().TrimEnd(TrailingSlash);
        }

        /// <summary>True for wss://, and for ws:// only when it points at this machine.</summary>
        internal static bool IsSecureSocketUrl(string? url)
        {
            if (string.IsNullOrEmpty(url)) return false;

            Uri? uri;
            if (!Uri.TryCreate(url!.Trim(), UriKind.Absolute, out uri)) return false;
            if (string.Equals(uri.Scheme, "wss", StringComparison.Ordinal)) return true;
            return string.Equals(uri.Scheme, "ws", StringComparison.Ordinal) && IsLoopbackHost(uri.Host);
        }

        /// <summary>True for the three ways a URL names this machine. Nothing else is loopback.</summary>
        internal static bool IsLoopbackHost(string? host)
        {
            if (string.IsNullOrEmpty(host)) return false;
            var bare = host!.Replace("[", string.Empty).Replace("]", string.Empty);
            return string.Equals(bare, "localhost", StringComparison.OrdinalIgnoreCase)
                || string.Equals(bare, "127.0.0.1", StringComparison.Ordinal)
                || string.Equals(bare, "::1", StringComparison.Ordinal);
        }
    }

    /// <summary>Where the match service lives for this build, and which protocol version it speaks.</summary>
    public sealed class SteamMatchSettings
    {
        public SteamMatchSettings(string? baseUrl, int protocolVersion)
        {
            BaseUrl = baseUrl ?? string.Empty;
            ProtocolVersion = protocolVersion;
        }

        /// <summary>Absolute http/https base URL with no trailing slash, or empty when unconfigured.</summary>
        public string BaseUrl { get; }

        public int ProtocolVersion { get; }

        /// <summary>False means Steam play stays switched off and the title says so.</summary>
        public bool IsConfigured { get { return BaseUrl.Length > 0; } }

        public static SteamMatchSettings NotConfigured
        {
            get { return new SteamMatchSettings(string.Empty, SteamMatchConfig.DefaultProtocolVersion); }
        }
    }

    /// <summary>
    /// Resolves the match-service base URL. This half is pure so the placeholder and validation rules
    /// are dotnet-tested; the Unity half (command line, environment, Resources asset) is in
    /// <c>SteamMatchConfig.cs</c>.
    /// </summary>
    public static partial class SteamMatchConfig
    {
        public const int DefaultProtocolVersion = 2;

        /// <summary>The value shipped in the config asset. It means: the owner has not filled this in.</summary>
        public const string PlaceholderBaseUrl = "OWNER-INPUT";

        public const string ResourceName = "HexWarsSteamConfig";

        public const string CommandLineFlag = "-hexwars-match-url";

        public const string EnvironmentVariable = "HEXWARS_MATCH_URL";

        /// <summary>Validates one candidate base URL. Anything not absolute http/https, and the shipped
        /// placeholder, count as not configured.</summary>
        public static SteamMatchSettings FromBaseUrl(string? rawBaseUrl)
        {
            return FromBaseUrl(rawBaseUrl, DefaultProtocolVersion);
        }

        public static SteamMatchSettings FromBaseUrl(string? rawBaseUrl, int protocolVersion)
        {
            var trimmed = SteamMatchApiContracts.NormalizeBase(rawBaseUrl);
            if (trimmed.Length == 0) return SteamMatchSettings.NotConfigured;
            if (string.Equals(trimmed, PlaceholderBaseUrl, StringComparison.OrdinalIgnoreCase))
                return SteamMatchSettings.NotConfigured;

            Uri? uri;
            if (!Uri.TryCreate(trimmed, UriKind.Absolute, out uri)) return SteamMatchSettings.NotConfigured;
            var isHttps = string.Equals(uri.Scheme, Uri.UriSchemeHttps, StringComparison.Ordinal);
            var isHttp = string.Equals(uri.Scheme, Uri.UriSchemeHttp, StringComparison.Ordinal);
            if (!isHttp && !isHttps) return SteamMatchSettings.NotConfigured;

            // Auth tickets and join credentials travel in these request bodies. Plain http is only
            // ever acceptable against a development server on this machine.
            if (isHttp && !SteamMatchApiContracts.IsLoopbackHost(uri.Host)) return SteamMatchSettings.NotConfigured;

            return new SteamMatchSettings(trimmed, protocolVersion > 0 ? protocolVersion : DefaultProtocolVersion);
        }

        /// <summary>Reads the shipped config asset. Unreadable JSON is treated as not configured.</summary>
        public static SteamMatchSettings ParseJson(string? json)
        {
            if (string.IsNullOrWhiteSpace(json)) return SteamMatchSettings.NotConfigured;

            SteamMatchConfigBody? body;
            try { body = JsonConvert.DeserializeObject<SteamMatchConfigBody>(json!); }
            catch (JsonException) { return SteamMatchSettings.NotConfigured; }

            if (body == null) return SteamMatchSettings.NotConfigured;
            return FromBaseUrl(body.MatchBaseUrl, body.ProtocolVersion);
        }
    }

    /// <summary>
    /// The protocol-v2 frames that are not part of the v1 message set. v1 messages (SEAT, CATALOG?,
    /// START, APPLY, REJECT, CMD, CATALOG) still go through <c>NetProtocol</c>; these are the extras
    /// the durable match server adds: the AUTH handshake, its failure reply, the keepalive pair, the
    /// graceful-restart notice, and the reject that means try the same command again.
    /// </summary>
    public static class SteamMatchProtocol
    {
        public const string AuthFailPrefix = "AUTH FAIL";

        /// <summary>The code reported when the server sent something outside the known set.</summary>
        public const string AuthFailUnknown = "unknown";

        /// <summary>The close code used when the far end broke the protocol.</summary>
        public const string ProtocolCloseCode = "protocol";

        /// <summary>Every AUTH FAIL code this client recognises. Anything else becomes "unknown".</summary>
        static readonly string[] AuthFailCodes = { "invalid", "expired", "unavailable", ProtocolCloseCode };

        public const string Ping = "PING";

        public const string Pong = "PONG";

        public const string ServerRestart = "SERVER RESTART";

        /// <summary>REJECT payload meaning the durable commit failed, not that the move was illegal.</summary>
        public const string TemporaryFailure = "TemporaryFailure";

        /// <summary>The first frame every v2 socket sends. The credential never goes in the URL.</summary>
        public static string AuthFrame(string? matchId, string? credential)
        {
            return "AUTH " + (matchId ?? string.Empty) + " " + (credential ?? string.Empty);
        }

        /// <summary>
        /// Recognises <c>AUTH FAIL &lt;code&gt;</c> and yields one of <c>invalid</c>, <c>expired</c>,
        /// <c>unavailable</c>, <c>protocol</c> - or <c>unknown</c> for anything else. A frame that
        /// merely starts with the same letters is not a match.
        /// <para>
        /// The payload is attacker-controlled: it is whatever the far end of the socket chose to send,
        /// before this client has decided it trusts that end at all. It reaches a Unity log and the
        /// bootstrap failure path, so it is mapped onto a code we chose and the raw text is dropped
        /// here, at the parse, rather than being carried around and hopefully sanitised later.
        /// </para>
        /// </summary>
        public static bool TryParseAuthFail(string? raw, out string code)
        {
            code = string.Empty;
            if (string.IsNullOrEmpty(raw)) return false;
            if (!raw!.StartsWith(AuthFailPrefix, StringComparison.Ordinal)) return false;

            var rest = raw.Substring(AuthFailPrefix.Length);
            if (rest.Length == 0) { code = AuthFailUnknown; return true; }
            if (!rest.StartsWith(" ", StringComparison.Ordinal)) return false;

            var payload = rest.Substring(1).Trim();
            foreach (var known in AuthFailCodes)
            {
                if (string.Equals(payload, known, StringComparison.Ordinal)) { code = known; return true; }
            }
            code = AuthFailUnknown;
            return true;
        }

        /// <summary>True when a REJECT payload is the retryable one.</summary>
        public static bool IsTemporaryFailure(string? rejectPayload)
        {
            return string.Equals(rejectPayload, TemporaryFailure, StringComparison.Ordinal);
        }
    }
}
