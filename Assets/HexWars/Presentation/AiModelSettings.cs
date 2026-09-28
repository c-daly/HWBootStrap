using System;
using System.IO;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using HexWars.Engine.AI;
using Newtonsoft.Json;
using UnityEngine;
using UnityEngine.Networking;

namespace HexWars.Presentation
{
    /// <summary>A match owns this copy; later preference/catalog changes cannot replace its policy.</summary>
    public sealed class AiModelSelection
    {
        public AiModelSelection(AiModelCatalog catalog, string difficultyId, string serverUrl)
            : this(catalog, difficultyId, serverUrl, false) { }

        AiModelSelection(AiModelCatalog catalog, string id, string serverUrl, bool explicitModel)
        {
            catalog.Validate();
            var difficulty = explicitModel ? null : catalog.ResolveDifficulty(id);
            var source = catalog.Resolve(explicitModel ? id : difficulty.model_id);
            DifficultyId = difficulty?.id;
            DifficultyLabel = difficulty?.label ?? source.label;
            Model = new AiModelDefinition
            {
                id = source.id, label = source.label, kind = source.kind, package = source.package,
                checkpoint_sha256 = source.checkpoint_sha256, contract_version = source.contract_version,
                encoding_hash = source.encoding_hash, capacity_hash = source.capacity_hash,
            };
            CatalogRevision = catalog.catalog_revision;
            ModelsRoot = catalog.models_root;
            InferencePath = catalog.inference_path;
            ServerUrl = serverUrl;
        }

        internal static AiModelSelection ForExplicitModel(AiModelCatalog catalog, string modelId, string serverUrl) =>
            new AiModelSelection(catalog, modelId, serverUrl, true);
        public string DifficultyId { get; }
        public string DifficultyLabel { get; }
        public AiModelDefinition Model { get; }
        public string CatalogRevision { get; }
        public string ModelsRoot { get; }
        public string InferencePath { get; }
        public string ServerUrl { get; }
        public bool UsesHttp => !string.IsNullOrEmpty(ServerUrl);
        public string DisplayName => DifficultyLabel;
    }

    /// <summary>One project/deployment catalog for every unpinned AI entrypoint.</summary>
    public static class AiModelSettings
    {
        static AiModelCatalog _catalog;
        static Task _loading;
        static string _serverUrl;
        public static bool IsLoaded => _catalog != null;
        public static string Error { get; private set; }
        public static AiModelCatalog Catalog => _catalog ?? throw new InvalidOperationException(
            Error ?? "AI model settings are still loading.");
        public static string SelectionLabel => IsLoaded ? Capture().DisplayName : Error ?? "Loading AI models…";

        public static Task LoadAsync(bool refresh = false)
        {
            // Share only an active request. A failed load can recover on the next explicit attempt.
            if (_loading != null && !_loading.IsCompleted) return _loading;
            if (_catalog != null && !refresh) return Task.CompletedTask;
            return _loading = LoadCore();
        }

        static async Task LoadCore()
        {
            try
            {
                _serverUrl = ResolveServerUrl(Application.absoluteURL,
                    Environment.GetEnvironmentVariable("HEXWARS_AI_SERVER_URL"), IsBrowser);
                string json;
                if (!string.IsNullOrEmpty(_serverUrl))
                    json = await AiHttp.RequestAsync(_serverUrl + "/api/ai/models", null, 30000, CancellationToken.None);
                else
                    json = File.ReadAllText(CatalogPath(Application.streamingAssetsPath,
                        Environment.GetEnvironmentVariable("HEXWARS_AI_CONFIG_PATH")));
                var catalog = JsonConvert.DeserializeObject<AiModelCatalog>(json, JsonSettings);
                if (catalog == null) throw new InvalidDataException("AI catalog is empty.");
                if (string.IsNullOrEmpty(_serverUrl)) catalog.catalog_revision = Sha256(Encoding.UTF8.GetBytes(json));
                catalog.Validate();
                RequireHash(catalog.catalog_revision, "AI catalog revision");
                _catalog = catalog;
                Error = null;
            }
            catch (Exception error)
            {
                _catalog = null;
                Error = "AI models unavailable: " + error.Message;
                throw;
            }
        }

        public static AiModelSelection Capture(string difficultyId = null) =>
            new AiModelSelection(Catalog, difficultyId, _serverUrl);

        public static string DifficultyLabel(string difficultyId)
        {
            if (!IsLoaded) return Error ?? "Loading difficulties…";
            try { return Catalog.ResolveDifficulty(difficultyId).label; }
            catch (ArgumentException) { return "Difficulty unavailable"; }
        }

        public static AiModelSelection ForLevel(AiLevel level)
        {
            if (level == AiLevel.Easy) return AiModelSelection.ForExplicitModel(Catalog, "random", _serverUrl);
            if (level == AiLevel.Hard) return AiModelSelection.ForExplicitModel(Catalog, "greedy", _serverUrl);
            return Capture();
        }

        public static bool UsesTrainedModel(AiLevel level) =>
            (level == AiLevel.TrainedModel || level == AiLevel.Configured) && (!IsLoaded || Capture().Model.IsTrained);

        internal static string CatalogPath(string streamingAssets, string configured) =>
            string.IsNullOrWhiteSpace(configured) ? Path.Combine(streamingAssets, "ai-models.json") : Path.GetFullPath(configured);

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetForPlay() { _catalog = null; _loading = null; _serverUrl = null; Error = null; }

        internal static string ResolveServerUrl(string page, string configured, bool browser)
        {
            string value = configured;
            if (browser)
            {
                if (!Uri.TryCreate(page, UriKind.Absolute, out var location) ||
                    (location.Scheme != "http" && location.Scheme != "https"))
                    throw new InvalidOperationException("Open the game from its HTTP server to use hosted AI.");
                value = location.GetLeftPart(UriPartial.Authority);
            }
            if (string.IsNullOrWhiteSpace(value)) return null;
            if (!Uri.TryCreate(value, UriKind.Absolute, out var uri) ||
                (uri.Scheme != "http" && uri.Scheme != "https") ||
                !string.IsNullOrEmpty(uri.UserInfo) || !string.IsNullOrEmpty(uri.Query) || !string.IsNullOrEmpty(uri.Fragment))
                throw new InvalidOperationException("HEXWARS_AI_SERVER_URL must be an HTTP server URL.");
            return value.TrimEnd('/');
        }

        internal static readonly JsonSerializerSettings JsonSettings = new JsonSerializerSettings
        {
            MissingMemberHandling = MissingMemberHandling.Error, TypeNameHandling = TypeNameHandling.None,
            MetadataPropertyHandling = MetadataPropertyHandling.Ignore, CheckAdditionalContent = true,
        };

        internal static string Sha256(byte[] bytes)
        {
            using (var hash = SHA256.Create())
                return BitConverter.ToString(hash.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
        }

        internal static void RequireHash(string value, string label)
        {
            if (value == null || value.Length != 64 || Array.Exists(value.ToCharArray(),
                ch => !(ch >= '0' && ch <= '9') && !(ch >= 'a' && ch <= 'f')))
                throw new InvalidDataException(label + " must be lowercase SHA-256.");
        }

        static bool IsBrowser
        {
            get
            {
#if UNITY_WEBGL && !UNITY_EDITOR
                return true;
#else
                return false;
#endif
            }
        }
    }

    internal static class AiHttp
    {
        public static async Task<string> RequestAsync(string url, string body, int timeoutMs, CancellationToken cancellation)
        {
            using (var request = new UnityWebRequest(url, body == null ? "GET" : "POST"))
            {
                request.downloadHandler = new DownloadHandlerBuffer();
                if (body != null)
                {
                    request.uploadHandler = new UploadHandlerRaw(Encoding.UTF8.GetBytes(body));
                    request.SetRequestHeader("Content-Type", "application/json");
                }
                request.timeout = Math.Max(1, (timeoutMs + 999) / 1000);
                var operation = request.SendWebRequest();
                var started = System.Diagnostics.Stopwatch.StartNew();
                try
                {
                    while (!operation.isDone)
                    {
                        cancellation.ThrowIfCancellationRequested();
                        if (started.ElapsedMilliseconds >= timeoutMs) throw new TimeoutException("AI server request timed out.");
                        await Task.Yield();
                    }
                    cancellation.ThrowIfCancellationRequested();
                    if (request.result != UnityWebRequest.Result.Success)
                        throw new InvalidOperationException("AI server unavailable (HTTP " + request.responseCode + "): " + request.error);
                    return request.downloadHandler.text;
                }
                catch
                {
                    request.Abort();
                    throw;
                }
            }
        }
    }
}
