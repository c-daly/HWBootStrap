using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using HexWars.Engine.AI;
using HexWars.NetServer.AI;
using Microsoft.Extensions.Logging.Abstractions;
using NUnit.Framework;

namespace HexWars.NetServer.Tests;

[TestFixture]
public sealed class HostedAiServiceTests
{
    AiFixture _fixture = null!;
    HostedAiService _service = null!;

    [SetUp]
    public void SetUp()
    {
        _fixture = new AiFixture();
        _service = _fixture.CreateService();
    }

    [TearDown]
    public async Task TearDown()
    {
        await _service.DisposeAsync();
        _fixture.Dispose();
    }

    [Test]
    public async Task ReusesOneProcessAndPreservesBothNativeSeats()
    {
        foreach (int seat in new[] { 1, 0, 1 })
        {
            AiDecisionResponse response = await _service.DecideAsync(_fixture.Request(seat: seat));
            Assert.That(response.CandidateId, Is.EqualTo(17));
            Assert.That(response.DifficultyId, Is.EqualTo("normal"));
            Assert.That(response.DecisionId, Is.EqualTo(39));
            Assert.That(response.CheckpointSha256, Is.EqualTo(_fixture.Model.checkpoint_sha256));
            Assert.That(response.CatalogRevision, Is.EqualTo(_fixture.Revision));
        }
        Assert.That(_fixture.Factory.Processes, Has.Count.EqualTo(1));
        Assert.That(_fixture.Factory.Seats, Is.EqualTo(new[] { 1, 0, 1 }));
    }

    [TestCase("checkpoint_sha256")]
    [TestCase("encoding_hash")]
    [TestCase("capacity_hash")]
    [TestCase("package_sha256")]
    [TestCase("contract_version")]
    [TestCase("contract_hash")]
    public void RejectsWrongReadyMetadataBeforeAnyDecision(string field)
    {
        _fixture.Factory.ReadyMutation = seats => seats[1][field] = "wrong";
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 503);
        Assert.That(_fixture.Factory.Seats, Is.Empty);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
    }

    [TestCase("{\"decision_id\":39,\"candidate_id\":123}")]
    [TestCase("{\"decision_id\":38,\"candidate_id\":17}")]
    [TestCase("{\"decision_id\":39,\"candidate_id\":17,\"candidate_id\":18}")]
    [TestCase("not-json")]
    public async Task InvalidWorkerResponseIsNotAFallbackAndRestartsNextRequest(string output)
    {
        _fixture.Factory.Decision = (_, _) => Task.FromResult(output);
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 503);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
        _fixture.Factory.Decision = AiFixture.CorrectDecision;
        await _service.DecideAsync(_fixture.Request());
        Assert.That(_fixture.Factory.Processes, Has.Count.EqualTo(2));
    }

    [TestCase("seat")]
    [TestCase("duplicate")]
    [TestCase("capacity")]
    [TestCase("terminal")]
    [TestCase("empty")]
    public void RejectsInvalidInputBeforeSpawning(string invalid)
    {
        AiDecisionRequest request = _fixture.Request(invalid: invalid);
        AssertFailure(async () => await _service.DecideAsync(request), 400);
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [Test]
    public void RejectsStalePinsBeforeSpawning()
    {
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request(revision: new string('c', 64))), 409);
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request(checkpoint: new string('c', 64))), 409);
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [Test]
    public async Task ListedDifficultiesCanShareTheSamePersistentModelWorker()
    {
        File.WriteAllText(_fixture.Config, JsonSerializer.Serialize(new AiModelCatalog
        {
            default_difficulty_id = "normal", models = new[] { _fixture.Model },
            difficulties = new[]
            {
                new AiDifficultyDefinition { id = "normal", label = "Normal", model_id = _fixture.Model.id },
                new AiDifficultyDefinition { id = "hard", label = "Hard", model_id = _fixture.Model.id },
            },
        }, AiJson.CatalogOptions));
        AiDecisionResponse normal = await _service.DecideAsync(_fixture.Request());
        AiDecisionResponse hard = await _service.DecideAsync(_fixture.Request(difficulty: "hard"));
        Assert.That(normal.DifficultyId, Is.EqualTo("normal"));
        Assert.That(hard.DifficultyId, Is.EqualTo("hard"));
        Assert.That(_fixture.Factory.Processes, Has.Count.EqualTo(1));
    }

    [Test]
    public void GameplayCannotSubstituteAnotherModelForItsSelectedDifficulty()
    {
        File.WriteAllText(_fixture.Config, JsonSerializer.Serialize(new AiModelCatalog
        {
            default_difficulty_id = "normal", models = new[] { _fixture.Model,
                new AiModelDefinition { id = "greedy", label = "Greedy", kind = "greedy" } },
            difficulties = new[] { new AiDifficultyDefinition { id = "normal", label = "Normal", model_id = "greedy" } },
        }, AiJson.CatalogOptions));
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 409);
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [TestCase("missing", 409)]
    [TestCase("", 400)]
    public void MissingOrUnknownDifficultyIsRejected(string difficulty, int status)
    {
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request(difficulty: difficulty)), status);
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [TestCase("ValueError: malformed decision", 400)]
    [TestCase("RuntimeError: inference failed", 503)]
    public void PythonErrorsRemainExplicitAndNeverChooseAnotherPolicy(string error, int status)
    {
        _fixture.Factory.Decision = (_, _) => Task.FromResult(JsonSerializer.Serialize(new { error }));
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), status);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
    }

    [Test]
    public async Task CatalogReplacementDuringInferenceNeverReturnsAnOldPinnedAnswer()
    {
        _fixture.Factory.Decision = (input, ct) =>
        {
            _fixture.WriteCatalog("New label");
            return AiFixture.CorrectDecision(input, ct);
        };
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 409);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
        _fixture.Factory.Decision = AiFixture.CorrectDecision;
        await _service.DecideAsync(_fixture.Request());
        Assert.That(_fixture.Factory.Processes, Has.Count.EqualTo(2));
    }

    [Test]
    public async Task PendingBoundAndQueuedCancellationDoNotInterruptTheActiveWorker()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var release = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _fixture.Factory.Decision = async (input, ct) =>
        { entered.TrySetResult(); await release.Task.WaitAsync(ct); return await AiFixture.CorrectDecision(input, ct); };
        Task<AiDecisionResponse> active = _service.DecideAsync(_fixture.Request());
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        using var cancel = new CancellationTokenSource();
        Task<AiDecisionResponse> queued = _service.DecideAsync(_fixture.Request(), cancel.Token);
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 429);
        cancel.Cancel();
        Assert.That(async () => await queued, Throws.InstanceOf<OperationCanceledException>());
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.False);
        release.SetResult();
        await active;
    }

    [Test]
    public async Task InFlightCancellationDiscardsItsUnreadResponseBeforeNextRequest()
    {
        var entered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        _fixture.Factory.Decision = async (_, ct) =>
        { entered.TrySetResult(); await Task.Delay(Timeout.Infinite, ct); return "unreachable"; };
        using var cancel = new CancellationTokenSource();
        Task<AiDecisionResponse> active = _service.DecideAsync(_fixture.Request(), cancel.Token);
        await entered.Task.WaitAsync(TimeSpan.FromSeconds(5));
        cancel.Cancel();
        Assert.That(async () => await active, Throws.InstanceOf<OperationCanceledException>());
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
        _fixture.Factory.Decision = AiFixture.CorrectDecision;
        await _service.DecideAsync(_fixture.Request());
        Assert.That(_fixture.Factory.Processes, Has.Count.EqualTo(2));
    }

    [Test]
    public async Task DeadlineDiscardsWorkerAndReturnsExplicitTimeout()
    {
        await _service.DisposeAsync();
        _service = _fixture.CreateService(TimeSpan.FromMilliseconds(30));
        _fixture.Factory.Decision = async (_, ct) =>
        { await Task.Delay(Timeout.Infinite, ct); return "unreachable"; };
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 504);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
    }

    [TestCase(false)]
    [TestCase(true)]
    public async Task ChangedCheckpointCannotBeServedEvenAfterWorkerWasLoaded(bool afterInference)
    {
        await _service.DecideAsync(_fixture.Request());
        if (afterInference)
            _fixture.Factory.Decision = (input, ct) =>
            { File.WriteAllText(_fixture.Checkpoint, "changed"); return AiFixture.CorrectDecision(input, ct); };
        else File.WriteAllText(_fixture.Checkpoint, "changed");
        AssertFailure(async () => await _service.DecideAsync(_fixture.Request()), 503);
        Assert.That(_fixture.Factory.Processes.Single().Disposed, Is.True);
    }

    internal static void AssertFailure(AsyncTestDelegate action, int status)
    {
        AiServiceException? error = Assert.ThrowsAsync<AiServiceException>(action);
        Assert.That(error!.Status, Is.EqualTo(status));
    }
}

internal sealed class AiFixture : IDisposable
{
    internal string Root { get; } = Path.Combine(Path.GetTempPath(), "hexwars-ai-test-" + Guid.NewGuid().ToString("N"));
    internal string Checkpoint => Path.Combine(Root, "models", "ai", "test", "checkpoints", "best.pt");
    internal string Config => Path.Combine(Root, "ai-models.json");
    internal string Revision => AiModelCatalog.RevisionOf(File.ReadAllText(Config));
    internal AiModelDefinition Model { get; }
    internal FakePolicyFactory Factory { get; } = new();

    internal AiFixture()
    {
        Directory.CreateDirectory(Path.GetDirectoryName(Checkpoint)!);
        File.WriteAllText(Checkpoint, "fixture-weights");
        Model = new AiModelDefinition
        {
            id = "test", label = "Test model", kind = "trained", package = "test", contract_version = "tactical-v3",
            checkpoint_sha256 = Hash(File.ReadAllBytes(Checkpoint)), encoding_hash = new string('a', 64), capacity_hash = new string('b', 64),
        };
        string package = Path.GetDirectoryName(Path.GetDirectoryName(Checkpoint))!;
        var capacity = new Dictionary<string, int>();
        foreach (string table in Tables) capacity.Add("max_" + (table == "memory" ? "memory_records" : table), 4);
        string identity = JsonSerializer.Serialize(new
        { contract_version = Model.contract_version, contract_hash = new string('f', 64), encoding_hash = Model.encoding_hash, capacity_hash = Model.capacity_hash, capacity });
        File.WriteAllText(Path.Combine(package, "policy-identity.json"), identity);
        File.WriteAllText(Path.Combine(package, "run.json"), JsonSerializer.Serialize(new
        {
            schema_version = 2, kind = "tactical-v3-playable-export", state = "completed", latest_checkpoint = "checkpoints/best.pt",
            policy_identity = "policy-identity.json", checkpoint_sha256 = Model.checkpoint_sha256,
            package_id = "test-package", package_sha256 = new string('d', 64),
            files = new Dictionary<string, string> { ["policy-identity.json"] = Hash(Encoding.UTF8.GetBytes(identity)) },
        }));
        WriteCatalog("Test model");
    }

    static readonly string[] Tables = { "cells", "units", "templates", "capability_definitions", "capability_allocations", "rules", "memory", "relations", "candidates" };
    internal AiRuntimeOptions Options(TimeSpan? timeout = null) => new()
    { RuntimeRoot = Root, ConfigPath = Config, Python = "unused", MaxPendingPerModel = 2, DecisionTimeout = timeout ?? TimeSpan.FromSeconds(5) };
    internal HostedAiService CreateService(TimeSpan? timeout = null) => new(Options(timeout), Factory, NullLogger<HostedAiService>.Instance);
    internal void WriteCatalog(string label)
    {
        Model.label = label;
        File.WriteAllText(Config, JsonSerializer.Serialize(new AiModelCatalog
        {
            default_difficulty_id = "normal", models = new[] { Model },
            difficulties = new[] { new AiDifficultyDefinition { id = "normal", label = "Normal", model_id = Model.id } },
        }, AiJson.CatalogOptions));
    }
    internal AiDecisionRequest Request(int seat = 0, string? invalid = null, string? revision = null, string? checkpoint = null,
        string difficulty = "normal")
    {
        var observation = Tables.Where(x => x != "candidates").ToDictionary(x => x, _ => Array.Empty<object>());
        if (invalid == "capacity") observation["units"] = new object[5];
        object candidate = new { candidate_id = 17, decision_id = 39 };
        return new()
        {
            DifficultyId = difficulty, ModelId = Model.id, CatalogRevision = revision ?? Revision, CheckpointSha256 = checkpoint ?? Model.checkpoint_sha256, Seat = seat,
            Decision = JsonSerializer.SerializeToElement(new
            {
                seat = invalid == "seat" ? 1 - seat : seat, decision_id = 39, observation,
                candidates = invalid == "duplicate" ? new[] { candidate, candidate } : invalid == "empty" ? Array.Empty<object>() : new[] { candidate },
                terminated = invalid == "terminal", truncated = false,
            }),
        };
    }
    internal static Task<string> CorrectDecision(JsonElement input, CancellationToken _) => Task.FromResult(JsonSerializer.Serialize(new
    { decision_id = input.GetProperty("decision").GetProperty("decision_id").GetInt64(), candidate_id = 17 }));
    static string Hash(byte[] bytes) => Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();
    public void Dispose() => Directory.Delete(Root, true);
}

internal sealed class FakePolicyFactory : IPolicyProcessFactory
{
    internal List<FakePolicyProcess> Processes { get; } = new();
    internal List<int> Seats { get; } = new();
    internal Action<Dictionary<string, object>[]>? ReadyMutation { get; set; }
    internal Func<JsonElement, CancellationToken, Task<string>> Decision { get; set; } = AiFixture.CorrectDecision;
    public IPolicyProcess Start(AiRuntimeOptions _, AiModelDefinition model, AiPackage package)
    {
        var seats = Enumerable.Range(0, 2).Select(seat => new Dictionary<string, object>
        {
            ["seat"] = seat, ["kind"] = "run", ["inference_mode"] = "deterministic", ["environment"] = "tactical-v3",
            ["contract_version"] = model.contract_version, ["contract_hash"] = package.ContractHash, ["encoding_hash"] = model.encoding_hash, ["capacity_hash"] = model.capacity_hash,
            ["checkpoint_sha256"] = model.checkpoint_sha256, ["package_sha256"] = package.PackageSha256, ["package_id"] = package.PackageId,
        }).ToArray();
        ReadyMutation?.Invoke(seats);
        var process = new FakePolicyProcess(this, JsonSerializer.Serialize(new { ready = true, model_seats = new[] { 0, 1 }, seat_models = seats }));
        Processes.Add(process);
        return process;
    }
}

internal sealed class FakePolicyProcess(FakePolicyFactory factory, string ready) : IPolicyProcess
{
    bool _ready;
    JsonElement _input;
    internal bool Disposed { get; private set; }
    public Task<string> ReadLineAsync(int _, CancellationToken ct)
    {
        if (!_ready) { _ready = true; return Task.FromResult(ready); }
        return factory.Decision(_input, ct);
    }
    public Task WriteLineAsync(string line, CancellationToken ct)
    {
        _input = JsonSerializer.Deserialize<JsonElement>(line);
        factory.Seats.Add(_input.GetProperty("seat").GetInt32());
        return Task.CompletedTask;
    }
    public ValueTask DisposeAsync() { Disposed = true; return ValueTask.CompletedTask; }
}
