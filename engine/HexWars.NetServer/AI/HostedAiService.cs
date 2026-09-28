using System.Text.Json;
using HexWars.Engine.AI;

namespace HexWars.NetServer.AI;

public sealed class HostedAiService : IAsyncDisposable
{
    readonly AiRuntimeOptions _options;
    readonly IPolicyProcessFactory _factory;
    readonly ILogger<HostedAiService> _logger;
    readonly SemaphoreSlim _registry = new(1, 1);
    readonly Dictionary<string, ModelWorker> _workers = new(StringComparer.Ordinal);
    string? _revision;
    bool _stopped;

    internal HostedAiService(AiRuntimeOptions options, IPolicyProcessFactory factory, ILogger<HostedAiService> logger)
    { _options = options; _factory = factory; _logger = logger; }

    public async Task<AiModelCatalog> CatalogAsync(CancellationToken ct = default)
    {
        await _registry.WaitAsync(ct);
        try { return await LoadCatalogAsync(ct); }
        finally { _registry.Release(); }
    }

    async Task<AiModelCatalog> LoadCatalogAsync(CancellationToken ct)
    {
        try
        {
            if (_stopped) throw AiServiceException.Unavailable();
            if (new FileInfo(_options.ConfigPath).Length > 1024 * 1024) throw new IOException("AI catalog is too large.");
            string json = await File.ReadAllTextAsync(_options.ConfigPath, ct);
            using JsonDocument document = JsonDocument.Parse(json);
            AiJson.RequireUniqueProperties(document.RootElement);
            var catalog = JsonSerializer.Deserialize<AiModelCatalog>(json, AiJson.CatalogOptions)
                ?? throw new JsonException("Missing AI catalog.");
            catalog.Validate();
            if (catalog.inference_path != "/api/ai/decision") throw new JsonException("Unsupported hosted inference route.");
            catalog.catalog_revision = AiModelCatalog.RevisionOf(json);
            if (_revision != catalog.catalog_revision)
            {
                foreach (ModelWorker worker in _workers.Values) await worker.DisposeAsync();
                _workers.Clear();
                _revision = catalog.catalog_revision;
            }
            return catalog;
        }
        catch (Exception error) when (error is not OperationCanceledException and not AiServiceException)
        {
            _logger.LogWarning("Hosted AI catalog could not be loaded ({FailureType})", error.GetType().Name);
            throw AiServiceException.Unavailable();
        }
    }

    public async Task<AiDecisionResponse> DecideAsync(AiDecisionRequest request, CancellationToken ct = default)
    {
        if (string.IsNullOrWhiteSpace(request.DifficultyId)) throw AiServiceException.Invalid();
        ModelWorker worker;
        await _registry.WaitAsync(ct);
        try
        {
            AiModelCatalog catalog = await LoadCatalogAsync(ct);
            if (request.CatalogRevision != catalog.catalog_revision) throw AiServiceException.Stale();
            try
            {
                if (catalog.ResolveDifficulty(request.DifficultyId).model_id != request.ModelId)
                    throw AiServiceException.Stale();
            }
            catch (ArgumentException) { throw AiServiceException.Stale(); }
            AiModelDefinition? model = catalog.models.FirstOrDefault(x => x.id == request.ModelId);
            if (model is null || request.CheckpointSha256 != model.checkpoint_sha256) throw AiServiceException.Stale();
            if (!model.IsTrained) throw AiServiceException.Invalid();
            if (!_workers.TryGetValue(model.id, out worker!))
            {
                try
                {
                    AiPackage package = await AiPackage.ReadAsync(_options, catalog, model, ct);
                    worker = new ModelWorker(_options, _factory, model, package, _logger);
                    _workers.Add(model.id, worker);
                }
                catch (Exception error) when (error is not OperationCanceledException)
                {
                    _logger.LogWarning("Hosted AI package could not be loaded ({FailureType})", error.GetType().Name);
                    throw AiServiceException.Unavailable();
                }
            }
        }
        finally { _registry.Release(); }

        (long decision, int candidate) = await worker.DecideAsync(request, ct);
        // A config replacement while the process was thinking must not return an answer under a pin
        // that is no longer offered by the server. The client starts a new match explicitly.
        if ((await CatalogAsync(ct)).catalog_revision != request.CatalogRevision) throw AiServiceException.Stale();
        return new(request.DifficultyId, request.ModelId, request.CatalogRevision, request.CheckpointSha256, decision, candidate);
    }

    public async ValueTask DisposeAsync()
    {
        await _registry.WaitAsync();
        try
        {
            _stopped = true;
            foreach (ModelWorker worker in _workers.Values) await worker.DisposeAsync();
            _workers.Clear();
        }
        finally { _registry.Release(); }
    }

    internal sealed class ModelWorker(AiRuntimeOptions options, IPolicyProcessFactory factory,
        AiModelDefinition model, AiPackage package, ILogger logger) : IAsyncDisposable
    {
        readonly SemaphoreSlim _serial = new(1, 1);
        readonly CancellationTokenSource _lifetime = new();
        IPolicyProcess? _process;
        int _pending;

        internal async Task<(long, int)> DecideAsync(AiDecisionRequest request, CancellationToken requestCancellation)
        {
            var legal = package.ValidateDecision(request);
            if (Interlocked.Increment(ref _pending) > options.MaxPendingPerModel)
            { Interlocked.Decrement(ref _pending); throw AiServiceException.Busy(); }
            bool acquired = false;
            using var linked = CancellationTokenSource.CreateLinkedTokenSource(requestCancellation, _lifetime.Token);
            try
            {
                acquired = await _serial.WaitAsync(options.QueueTimeout, linked.Token);
                if (!acquired) throw AiServiceException.Busy();
                linked.Token.ThrowIfCancellationRequested();
                await package.VerifyUnchangedAsync(linked.Token);
                if (_process is null)
                {
                    using var startup = CancellationTokenSource.CreateLinkedTokenSource(linked.Token);
                    startup.CancelAfter(options.StartupTimeout);
                    _process = factory.Start(options, model, package);
                    using JsonDocument ready = JsonDocument.Parse(await _process.ReadLineAsync(128 * 1024, startup.Token));
                    package.ValidateReady(ready.RootElement, model);
                    await package.VerifyUnchangedAsync(startup.Token);
                }
                using var deadline = CancellationTokenSource.CreateLinkedTokenSource(linked.Token);
                deadline.CancelAfter(options.DecisionTimeout);
                string input = JsonSerializer.Serialize(new { seat = request.Seat, decision = request.Decision });
                await _process.WriteLineAsync(input, deadline.Token);
                using JsonDocument output = JsonDocument.Parse(await _process.ReadLineAsync(8192, deadline.Token));
                JsonElement result = output.RootElement;
                AiJson.RequireUniqueProperties(result);
                if (result.TryGetProperty("error", out JsonElement failure))
                {
                    string message = failure.GetString() ?? "";
                    if (message.StartsWith("ValueError:", StringComparison.Ordinal) ||
                        message.StartsWith("TypeError:", StringComparison.Ordinal)) throw AiServiceException.Invalid();
                    throw AiServiceException.Unavailable();
                }
                AiJson.Require(result.EnumerateObject().Count() == 2 &&
                    result.GetProperty("decision_id").GetInt64() == legal.DecisionId);
                int selected = result.GetProperty("candidate_id").GetInt32();
                AiJson.Require(legal.Candidates.Contains(selected));
                await package.VerifyUnchangedAsync(deadline.Token);
                return (legal.DecisionId, selected);
            }
            catch (OperationCanceledException) when (!requestCancellation.IsCancellationRequested)
            {
                if (acquired) await ResetAsync();
                if (_lifetime.IsCancellationRequested) throw AiServiceException.Stale();
                throw AiServiceException.Timeout();
            }
            catch (Exception error)
            {
                // An unread/late response can never become the next request's answer. Cancellation or
                // any protocol failure discards this owned process before the queue advances.
                if (acquired) await ResetAsync();
                if (error is OperationCanceledException or AiServiceException) throw;
                logger.LogWarning("Hosted AI inference failed for {ModelId} ({FailureType})", model.id, error.GetType().Name);
                throw AiServiceException.Unavailable();
            }
            finally
            {
                if (acquired) _serial.Release();
                Interlocked.Decrement(ref _pending);
            }
        }

        async Task ResetAsync()
        {
            IPolicyProcess? process = _process;
            _process = null;
            if (process is not null) await process.DisposeAsync();
        }

        public async ValueTask DisposeAsync()
        {
            _lifetime.Cancel();
            await _serial.WaitAsync();
            try { await ResetAsync(); }
            finally { _serial.Release(); }
        }
    }
}
