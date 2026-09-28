using System.Text.Json;
using HexWars.NetServer.AI;
using HexWars.NetServer.Contracts;
using HexWars.NetServer.Hosting;

namespace HexWars.NetServer.Endpoints;

public static class AiEndpoints
{
    public const string RateLimitPolicy = "ai-inference";
    public const string CorsPolicy = "ai-webgl";
    // The published capacity includes 65,536 relation rows and 32,768 full projected candidates.
    // This ceiling accommodates those tables; the package capacity is checked again before inference.
    public const int MaxDecisionBytes = 32 * 1024 * 1024;

    public static void MapAiEndpoints(this WebApplication app)
    {
        app.MapGet("/api/ai/models", async (HttpContext context, HostedAiService service) =>
        {
            context.Response.Headers.CacheControl = "no-store";
            try { return Results.Json(await service.CatalogAsync(context.RequestAborted), AiJson.CatalogOptions); }
            catch (AiServiceException error) { return Failure(error); }
        }).RequireCors(CorsPolicy);

        app.MapPost("/api/ai/decision", async (HttpContext context, HostedAiService service) =>
        {
            context.Response.Headers.CacheControl = "no-store";
            try
            {
                using var upload = CancellationTokenSource.CreateLinkedTokenSource(context.RequestAborted);
                upload.CancelAfter(TimeSpan.FromSeconds(60));
                using JsonDocument? json = await ReadDecisionAsync(context.Request, upload.Token);
                if (json is null) throw AiServiceException.Invalid();
                AiJson.RequireUniqueProperties(json.RootElement);
                AiDecisionRequest? request = json.RootElement.Deserialize<AiDecisionRequest>();
                if (request is null) throw AiServiceException.Invalid();
                return Results.Json(await service.DecideAsync(request, context.RequestAborted));
            }
            catch (AiServiceException error) { return Failure(error); }
            catch (JsonException) { return Failure(AiServiceException.Invalid()); }
            catch (OperationCanceledException) when (!context.RequestAborted.IsCancellationRequested)
            { return ApiErrors.Failure(408, "ai_upload_timeout", "The AI request upload timed out."); }
        }).WithHexWarsRequestBody(MaxDecisionBytes)
            .RequireCors(CorsPolicy).RequireRateLimiting(RateLimitPolicy);
    }

    static IResult Failure(AiServiceException error) => ApiErrors.Failure(error.Status, error.Code, error.Message);

    static async Task<JsonDocument?> ReadDecisionAsync(HttpRequest request, CancellationToken ct)
    {
        if (!JsonBody.IsJson(request.ContentType) || request.ContentLength > MaxDecisionBytes) return null;
        // Grow with the actual view, rather than reserving the maximum 32 MiB for every small match.
        using var body = new MemoryStream();
        var buffer = new byte[8192];
        while (body.Length <= MaxDecisionBytes)
        {
            int count = await request.Body.ReadAsync(buffer.AsMemory(0,
                (int)Math.Min(buffer.Length, MaxDecisionBytes + 1 - body.Length)), ct);
            if (count == 0) break;
            body.Write(buffer, 0, count);
        }
        if (body.Length == 0 || body.Length > MaxDecisionBytes) return null;
        return JsonDocument.Parse(body.GetBuffer().AsMemory(0, (int)body.Length));
    }
}
