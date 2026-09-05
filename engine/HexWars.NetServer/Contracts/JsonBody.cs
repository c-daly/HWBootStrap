using System.Text.Json;
using Microsoft.Net.Http.Headers;

namespace HexWars.NetServer.Contracts
{
    /// <summary>
    /// Reads a request body, by hand, under the error contract these endpoints promise.
    ///
    /// The framework binder is deliberately not used. It answers a wrong content type with 415 and a
    /// broken body with a ProblemDetails 400, both of which are shapes the Unity client does not parse -
    /// so a client sending a malformed request would get an error it cannot read and report as an unknown
    /// failure. Every refusal here comes back as the same { error, message } object as every other one.
    ///
    /// It also bounds the work. The binder will happily buffer and parse whatever arrives before deciding
    /// it is nonsense; these endpoints are unauthenticated at the point the body is read, so the parse has
    /// to be capped before anything expensive happens rather than after.
    /// </summary>
    public static class JsonBody
    {
        /// <summary>Comfortably larger than the biggest legitimate request (a Steam auth ticket) and small
        /// enough that refusing one costs nothing.</summary>
        public const int DefaultMaxBytes = 8192;

        const string JsonMediaType = "application/json";

        static readonly JsonSerializerOptions Options = new(JsonSerializerDefaults.Web);

        /// <summary>
        /// The body as <typeparamref name="T"/>, or null when it is not one: wrong content type, absent,
        /// larger than <paramref name="maxBytes"/>, or not parseable JSON. The caller answers null with a
        /// single 400; which of those it was is a log line, not something to tell an unauthenticated
        /// caller.
        /// </summary>
        public static async Task<T?> ReadAsync<T>(
            HttpRequest request, int maxBytes = DefaultMaxBytes, CancellationToken ct = default)
            where T : class
        {
            if (!IsJson(request.ContentType)) return null;

            // A declared length over the cap is refused before a single byte is read. The header is a hint
            // rather than a promise, so the read below is capped independently.
            if (request.ContentLength > maxBytes) return null;

            // One byte past the cap, so a body of exactly maxBytes still fits and anything longer is
            // detected without ever holding more than that.
            var buffer = new byte[maxBytes + 1];
            int filled = 0;

            while (filled < buffer.Length)
            {
                int read = await request.Body
                    .ReadAsync(buffer.AsMemory(filled), ct).ConfigureAwait(false);

                if (read == 0) break;
                filled += read;
            }

            if (filled == 0 || filled > maxBytes) return null;

            try
            {
                return JsonSerializer.Deserialize<T>(buffer.AsSpan(0, filled), Options);
            }
            catch (JsonException)
            {
                return null;
            }
        }

        /// <summary>True for application/json and its parameterised and +json forms, ignoring any charset.
        /// A missing content type is not json: an unlabelled body is a client bug worth naming.</summary>
        internal static bool IsJson(string? contentType)
        {
            if (string.IsNullOrWhiteSpace(contentType)) return false;
            if (!MediaTypeHeaderValue.TryParse(contentType, out MediaTypeHeaderValue? media)) return false;

            return media.MatchesMediaType(JsonMediaType) ||
                   media.Suffix.Equals("json", StringComparison.OrdinalIgnoreCase);
        }
    }
}
