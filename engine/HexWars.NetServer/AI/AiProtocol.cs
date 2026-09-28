using System.Text.Json;
using System.Text.Json.Serialization;

namespace HexWars.NetServer.AI;

public sealed class AiDecisionRequest
{
    [JsonPropertyName("difficulty_id")] public string DifficultyId { get; init; } = "";
    [JsonPropertyName("model_id")] public string ModelId { get; init; } = "";
    [JsonPropertyName("catalog_revision")] public string CatalogRevision { get; init; } = "";
    [JsonPropertyName("checkpoint_sha256")] public string CheckpointSha256 { get; init; } = "";
    [JsonPropertyName("seat")] public int Seat { get; init; } = -1;
    [JsonPropertyName("decision")] public JsonElement Decision { get; init; }
}

public sealed record AiDecisionResponse(
    [property: JsonPropertyName("difficulty_id")] string DifficultyId,
    [property: JsonPropertyName("model_id")] string ModelId,
    [property: JsonPropertyName("catalog_revision")] string CatalogRevision,
    [property: JsonPropertyName("checkpoint_sha256")] string CheckpointSha256,
    [property: JsonPropertyName("decision_id")] long DecisionId,
    [property: JsonPropertyName("candidate_id")] int CandidateId);

public sealed class AiServiceException(int status, string code, string message) : Exception(message)
{
    public int Status { get; } = status;
    public string Code { get; } = code;
    public static AiServiceException Invalid() => new(400, "invalid_ai_decision", "The AI decision request is not valid.");
    public static AiServiceException Stale() => new(409, "ai_model_changed", "The selected AI model changed. Start a new match with the current model catalog.");
    public static AiServiceException Unavailable() => new(503, "ai_unavailable", "The selected AI model is unavailable.");
    public static AiServiceException Busy() => new(429, "ai_busy", "The AI server is busy. Please retry shortly.");
    public static AiServiceException Timeout() => new(504, "ai_timeout", "The selected AI model did not respond in time.");
}

internal static class AiJson
{
    internal static readonly JsonSerializerOptions CatalogOptions = new()
    {
        IncludeFields = true,
        IgnoreReadOnlyProperties = true,
    };

    internal static string String(JsonElement value, string name) => value.GetProperty(name).GetString()
        ?? throw new JsonException("Missing string.");

    internal static void Require(bool condition)
    {
        if (!condition) throw new JsonException("Invalid AI protocol data.");
    }

    // JsonElement otherwise accepts duplicate fields and silently chooses the last one. The two
    // processes must interpret precisely the same request, including nested candidate references.
    internal static void RequireUniqueProperties(JsonElement value)
    {
        if (value.ValueKind == JsonValueKind.Object)
        {
            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (JsonProperty property in value.EnumerateObject())
            {
                Require(names.Add(property.Name));
                RequireUniqueProperties(property.Value);
            }
        }
        else if (value.ValueKind == JsonValueKind.Array)
            foreach (JsonElement item in value.EnumerateArray()) RequireUniqueProperties(item);
    }
}
