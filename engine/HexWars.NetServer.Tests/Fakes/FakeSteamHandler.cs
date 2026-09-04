using System.Net;
using System.Net.Http.Headers;
using System.Text;

namespace HexWars.NetServer.Tests.Fakes
{
    /// <summary>
    /// The only HttpMessageHandler a Steam test is allowed to use. It scripts responses per request path
    /// and records every request URI and header, so a test can prove what went on the wire without ever
    /// reaching Valve. When a path has several scripted responses they are handed out in order and the
    /// final one then repeats, which is what lets a single Respond(503) drive a whole retry budget.
    /// </summary>
    public sealed class FakeSteamHandler : HttpMessageHandler
    {
        public const string AuthPath = "/ISteamUserAuth/AuthenticateUserTicket/v1/";
        public const string OwnershipPath = "/ISteamUser/CheckAppOwnership/v2/";
        public const string LobbyPath = "/ILobbyMatchmakingService/GetLobbyData/v1/";

        readonly Dictionary<string, Queue<Func<HttpResponseMessage>>> _scripted =
            new(StringComparer.OrdinalIgnoreCase);

        /// <summary>Every request URI in order, query string included, for assertions.</summary>
        public List<Uri> Requests { get; } = new();

        /// <summary>Request headers, one dictionary per entry in <see cref="Requests"/>.</summary>
        public List<IReadOnlyDictionary<string, string>> Headers { get; } = new();

        /// <summary>Artificial latency on every response, used to provoke the client timeout.</summary>
        public TimeSpan Delay { get; set; } = TimeSpan.Zero;

        static string Normalise(string path)
        {
            var trimmed = path;
            while (trimmed.StartsWith("/", StringComparison.Ordinal)) trimmed = trimmed.Substring(1);
            while (trimmed.EndsWith("/", StringComparison.Ordinal)) trimmed = trimmed.Substring(0, trimmed.Length - 1);
            return "/" + trimmed + "/";
        }

        public FakeSteamHandler Respond(
            string path, HttpStatusCode status, string body, Action<HttpResponseMessage>? configure = null)
        {
            var key = Normalise(path);
            if (!_scripted.TryGetValue(key, out var queue))
            {
                queue = new Queue<Func<HttpResponseMessage>>();
                _scripted[key] = queue;
            }

            queue.Enqueue(() =>
            {
                var response = new HttpResponseMessage(status)
                {
                    Content = new StringContent(body, Encoding.UTF8, "application/json"),
                };
                configure?.Invoke(response);
                return response;
            });
            return this;
        }

        public FakeSteamHandler RespondJson(string path, string json) => Respond(path, HttpStatusCode.OK, json);

        public FakeSteamHandler RespondStatus(string path, HttpStatusCode status) => Respond(path, status, "{}");

        public FakeSteamHandler RespondRetryAfter(string path, HttpStatusCode status, int seconds) =>
            Respond(path, status, "{}", r => r.Headers.RetryAfter = new RetryConditionHeaderValue(TimeSpan.FromSeconds(seconds)));

        protected override async Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Requests.Add(request.RequestUri!);
            Headers.Add(request.Headers.ToDictionary(
                h => h.Key, h => string.Join(",", h.Value), StringComparer.OrdinalIgnoreCase));

            if (Delay > TimeSpan.Zero)
            {
                await Task.Delay(Delay, cancellationToken).ConfigureAwait(false);
            }

            var key = Normalise(request.RequestUri!.AbsolutePath);
            if (!_scripted.TryGetValue(key, out var queue) || queue.Count == 0)
            {
                throw new InvalidOperationException("FakeSteamHandler has no scripted response for " + key);
            }

            // Dequeue while alternatives remain; the last scripted response is sticky so a test can say
            // "always fail" in one line and still observe the full retry count.
            var factory = queue.Count > 1 ? queue.Dequeue() : queue.Peek();
            return factory();
        }
    }
}
