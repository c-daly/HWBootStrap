#nullable enable
using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace HexWars.Presentation
{
    /// <summary>
    /// The live <see cref="ISteamMatchApi"/>: one UnityWebRequest POST per call, at most one in flight.
    /// Starting a call abandons whatever came before it, and <see cref="Cancel"/> abandons the current
    /// one, so a coordinator that gave up never gets a late answer. Everything about the wire format
    /// lives in <see cref="SteamMatchApiContracts"/>; this class only moves bytes and reports the
    /// status. Ticket hex and join credentials never reach the log: only the URL path and the status.
    /// </summary>
    public sealed class SteamMatchApiClient : MonoBehaviour, ISteamMatchApi
    {
        /// <summary>Per-request timeout. A slower answer is reported as a network failure.</summary>
        public const int TimeoutSeconds = 10;

        string _baseUrl = string.Empty;
        Coroutine? _running;
        UnityWebRequest? _request;
        int _generation;

        /// <summary>Absolute match-service base URL, from <see cref="SteamMatchConfig"/>.</summary>
        public string BaseUrl
        {
            get { return _baseUrl; }
            set { _baseUrl = value ?? string.Empty; }
        }

        public bool IsConfigured { get { return _baseUrl.Length > 0; } }

        /// <summary>Adds a configured client to a scene object.</summary>
        public static SteamMatchApiClient Attach(GameObject host, string baseUrl)
        {
            var client = host.AddComponent<SteamMatchApiClient>();
            client.BaseUrl = baseUrl;
            return client;
        }

        public void CreateMatch(string lobbyId, string ticketHex, string requestedSetupWire, Action<SteamMatchApiResult> onDone)
        {
            Post(SteamMatchApiContracts.CreateMatchUrl(_baseUrl),
                 SteamMatchApiContracts.CreateMatchBody(lobbyId, ticketHex, requestedSetupWire),
                 onDone);
        }

        public void JoinMatch(string matchId, string ticketHex, Action<SteamMatchApiResult> onDone)
        {
            Post(SteamMatchApiContracts.JoinMatchUrl(_baseUrl, matchId),
                 SteamMatchApiContracts.JoinMatchBody(ticketHex),
                 onDone);
        }

        /// <summary>Abandons the in-flight request. Its callback will not fire.</summary>
        public void Cancel()
        {
            _generation++;
            AbortInFlight();
        }

        void Post(string url, string json, Action<SteamMatchApiResult> onDone)
        {
            _generation++;          // whatever was in flight has lost its listener
            AbortInFlight();

            if (onDone == null) return;
            if (!IsConfigured)
            {
                onDone(SteamMatchApiResult.Failure(
                    0, SteamMatchApiContracts.NotConfiguredErrorCode, SteamMatchApiContracts.NotConfiguredMessage));
                return;
            }

            _running = StartCoroutine(Run(_generation, url, json, onDone));
        }

        IEnumerator Run(int generation, string url, string json, Action<SteamMatchApiResult> onDone)
        {
            var request = new UnityWebRequest(url, "POST");
            _request = request;
            request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(json));
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            request.timeout = TimeoutSeconds;

            yield return request.SendWebRequest();

            // A transport failure (offline, DNS, TLS, timeout) is status 0; a 4xx/5xx is a ProtocolError
            // that still carries the JSON error body, so it keeps its real status.
            var transportFailed = request.result != UnityWebRequest.Result.Success
                               && request.result != UnityWebRequest.Result.ProtocolError;
            var status = transportFailed ? 0L : request.responseCode;
            var body = request.downloadHandler != null ? request.downloadHandler.text : string.Empty;
            var transportError = transportFailed ? (request.error ?? string.Empty) : string.Empty;

            if (ReferenceEquals(_request, request)) _request = null;
            _running = null;
            request.Dispose();

            if (generation != _generation) yield break;   // cancelled or superseded while in flight

            Debug.Log("[SteamMatch] POST " + PathOf(url) + " -> " + status
                      + (transportError.Length == 0 ? string.Empty : " (" + transportError + ")"));
            onDone(SteamMatchApiContracts.Parse(status, body));
        }

        /// <summary>Stops the coroutine first, so the aborted request is never touched after Dispose.</summary>
        void AbortInFlight()
        {
            if (_running != null) { StopCoroutine(_running); _running = null; }
            if (_request == null) return;

            var request = _request;
            _request = null;
            try { request.Abort(); }
            catch (Exception e) { Debug.LogWarning("[SteamMatch] abort failed: " + e.Message); }
            request.Dispose();
        }

        void OnDestroy()
        {
            _generation++;
            AbortInFlight();
        }

        /// <summary>Log-safe form of a request URL: path only, never a query string.</summary>
        internal static string PathOf(string? url)
        {
            if (string.IsNullOrEmpty(url)) return string.Empty;
            try { return new Uri(url!).AbsolutePath; }
            catch (Exception) { return "(unparsed)"; }
        }
    }
}
