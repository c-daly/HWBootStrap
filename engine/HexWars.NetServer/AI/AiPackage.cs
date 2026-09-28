using System.Security.Cryptography;
using System.Text.Json;
using HexWars.Engine.AI;

namespace HexWars.NetServer.AI;

internal sealed record AiPackage(string Root, string PackageSha256, string PackageId, string ContractHash,
    IReadOnlyDictionary<string, string> Files, IReadOnlyDictionary<string, int> Capacity)
{
    internal static async Task<AiPackage> ReadAsync(AiRuntimeOptions options, AiModelCatalog catalog,
        AiModelDefinition model, CancellationToken ct)
    {
        string root = PlainPath(options.RuntimeRoot, catalog.models_root + "/" + model.package);
        string manifestPath = PlainPath(root, "run.json");
        string identityPath = PlainPath(root, "policy-identity.json");
        string checkpointPath = PlainPath(root, "checkpoints/best.pt");
        byte[] manifestBytes = await File.ReadAllBytesAsync(manifestPath, ct);
        byte[] identityBytes = await File.ReadAllBytesAsync(identityPath, ct);
        using JsonDocument manifest = JsonDocument.Parse(manifestBytes);
        using JsonDocument identity = JsonDocument.Parse(identityBytes);
        JsonElement m = manifest.RootElement, i = identity.RootElement;
        AiJson.RequireUniqueProperties(m);
        AiJson.RequireUniqueProperties(i);
        AiJson.Require(m.GetProperty("schema_version").GetInt32() == 2 &&
            AiJson.String(m, "kind") == "tactical-v3-playable-export" &&
            AiJson.String(m, "state") == "completed" &&
            AiJson.String(m, "latest_checkpoint") == "checkpoints/best.pt" &&
            AiJson.String(m, "policy_identity") == "policy-identity.json" &&
            AiJson.String(m, "checkpoint_sha256") == model.checkpoint_sha256 &&
            AiJson.String(i, "contract_version") == model.contract_version &&
            AiJson.String(i, "encoding_hash") == model.encoding_hash &&
            AiJson.String(i, "capacity_hash") == model.capacity_hash);
        string packageHash = AiJson.String(m, "package_sha256");
        AiModelCatalog.RequireSha256(packageHash, "package_sha256");
        string contractHash = AiJson.String(i, "contract_hash");
        AiModelCatalog.RequireSha256(contractHash, "contract_hash");
        var files = new Dictionary<string, string>(StringComparer.Ordinal)
        {
            [manifestPath] = Convert.ToHexString(SHA256.HashData(manifestBytes)).ToLowerInvariant(),
            [identityPath] = Convert.ToHexString(SHA256.HashData(identityBytes)).ToLowerInvariant(),
            [checkpointPath] = await DigestAsync(checkpointPath, ct),
        };
        AiJson.Require(files[checkpointPath] == model.checkpoint_sha256 &&
            files[identityPath] == AiJson.String(m.GetProperty("files"), "policy-identity.json"));
        var capacity = new Dictionary<string, int>(StringComparer.Ordinal);
        foreach (string table in new[] { "cells", "units", "templates", "capability_definitions",
                     "capability_allocations", "rules", "memory_records", "relations", "candidates" })
        {
            int max = i.GetProperty("capacity").GetProperty("max_" + table).GetInt32();
            AiJson.Require(max > 0);
            capacity.Add(table == "memory_records" ? "memory" : table, max);
        }
        return new(root, packageHash, AiJson.String(m, "package_id"), contractHash, files, capacity);
    }

    internal async Task VerifyUnchangedAsync(CancellationToken ct)
    {
        foreach ((string path, string expected) in Files)
        {
            _ = PlainPath(Root, Path.GetRelativePath(Root, path).Replace('\\', '/'));
            AiJson.Require(await DigestAsync(path, ct) == expected);
        }
    }

    internal static async Task<string> DigestAsync(string path, CancellationToken ct)
    {
        await using FileStream input = File.OpenRead(path);
        return Convert.ToHexString(await SHA256.HashDataAsync(input, ct)).ToLowerInvariant();
    }

    internal static string PlainPath(string root, string relative)
    {
        AiModelCatalog.RequireRelativePath(relative, "package path");
        string result = Path.GetFullPath(root);
        // No package component may redirect Python to data outside the configured runtime bundle.
        foreach (string component in relative.Split('/'))
        {
            result = Path.Combine(result, component);
            if ((File.GetAttributes(result) & FileAttributes.ReparsePoint) != 0)
                throw new IOException("AI package paths cannot contain symbolic links or reparse points.");
        }
        return result;
    }

    internal (long DecisionId, HashSet<int> Candidates) ValidateDecision(AiDecisionRequest request)
    {
        try
        {
            JsonElement view = request.Decision;
            AiJson.RequireUniqueProperties(view);
            AiJson.Require(request.Seat is 0 or 1 && view.GetProperty("seat").GetInt32() == request.Seat &&
                !view.GetProperty("terminated").GetBoolean() && !view.GetProperty("truncated").GetBoolean());
            long decisionId = view.GetProperty("decision_id").GetInt64();
            JsonElement observation = view.GetProperty("observation");
            foreach ((string table, int limit) in Capacity)
            {
                JsonElement rows = table == "candidates" ? view.GetProperty(table) : observation.GetProperty(table);
                AiJson.Require(rows.GetArrayLength() <= limit);
            }
            var candidates = new HashSet<int>();
            foreach (JsonElement candidate in view.GetProperty("candidates").EnumerateArray())
                AiJson.Require(candidate.GetProperty("decision_id").GetInt64() == decisionId &&
                    candidates.Add(candidate.GetProperty("candidate_id").GetInt32()));
            AiJson.Require(candidates.Count > 0);
            return (decisionId, candidates);
        }
        catch (Exception failure) when (failure is JsonException or InvalidOperationException or
                                       KeyNotFoundException or FormatException or OverflowException)
        { throw AiServiceException.Invalid(); }
    }

    internal void ValidateReady(JsonElement ready, AiModelDefinition model)
    {
        AiJson.RequireUniqueProperties(ready);
        AiJson.Require(ready.GetProperty("ready").GetBoolean());
        AiJson.Require(ready.GetProperty("model_seats").EnumerateArray().Select(x => x.GetInt32()).SequenceEqual(new[] { 0, 1 }));
        JsonElement seats = ready.GetProperty("seat_models");
        AiJson.Require(seats.GetArrayLength() == 2);
        for (int index = 0; index < 2; index++)
        {
            JsonElement seat = seats[index];
            AiJson.Require(seat.GetProperty("seat").GetInt32() == index &&
                AiJson.String(seat, "kind") == "run" &&
                AiJson.String(seat, "inference_mode") == "deterministic" &&
                AiJson.String(seat, "environment") == "tactical-v3" &&
                AiJson.String(seat, "contract_version") == model.contract_version &&
                AiJson.String(seat, "contract_hash") == ContractHash &&
                AiJson.String(seat, "encoding_hash") == model.encoding_hash &&
                AiJson.String(seat, "capacity_hash") == model.capacity_hash &&
                AiJson.String(seat, "checkpoint_sha256") == model.checkpoint_sha256 &&
                AiJson.String(seat, "package_sha256") == PackageSha256 &&
                AiJson.String(seat, "package_id") == PackageId);
        }
    }
}
