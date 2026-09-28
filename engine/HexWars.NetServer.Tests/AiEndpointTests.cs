using System.Net;
using System.Text;
using System.Text.Json;
using HexWars.NetServer.AI;
using HexWars.NetServer.Endpoints;
using HexWars.NetServer.Hosting;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.TestHost;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using NUnit.Framework;

namespace HexWars.NetServer.Tests;

[TestFixture]
public sealed class AiEndpointTests
{
    AiFixture _fixture = null!;
    WebApplication _app = null!;
    HttpClient _client = null!;

    [SetUp]
    public async Task Start()
    {
        _fixture = new AiFixture();
        var builder = WebApplication.CreateBuilder(new WebApplicationOptions { EnvironmentName = "Development" });
        builder.WebHost.UseTestServer();
        builder.Logging.ClearProviders();
        builder.Configuration.AddInMemoryCollection(new Dictionary<string, string?>
        { ["LOBBY_PROVIDER"] = "Legacy", ["MATCH_BUILD_ID"] = "ai-test", ["ALLOWED_WEB_ORIGINS"] = "https://game.example" });
        builder.AddHexWarsServer();
        builder.Services.AddSingleton(_fixture.Options());
        builder.Services.AddSingleton<IPolicyProcessFactory>(_fixture.Factory);
        _app = builder.Build();
        _app.UseHexWarsServer();
        _app.MapPost("/ordinary-body", () => "ordinary").WithHexWarsRequestBody();
        await _app.StartAsync();
        _client = _app.GetTestClient();
    }

    [TearDown]
    public async Task Stop()
    {
        _client.Dispose();
        await _app.StopAsync();
        await _app.DisposeAsync();
        _fixture.Dispose();
    }

    [Test]
    public async Task CatalogUsesSharedFieldsContentRevisionAndNoStore()
    {
        using HttpResponseMessage response = await _client.GetAsync("/api/ai/models");
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
        Assert.That(response.Headers.CacheControl!.NoStore, Is.True);
        using JsonDocument json = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.That(json.RootElement.GetProperty("catalog_revision").GetString(), Is.EqualTo(_fixture.Revision));
        Assert.That(json.RootElement.GetProperty("default_difficulty_id").GetString(), Is.EqualTo("normal"));
        Assert.That(json.RootElement.GetProperty("difficulties")[0].GetProperty("model_id").GetString(), Is.EqualTo("test"));
        Assert.That(json.RootElement.GetProperty("models")[0].TryGetProperty("IsTrained", out _), Is.False);
        Assert.That(_fixture.Factory.Processes, Is.Empty, "Reading the menu must not load a model.");
    }

    [TestCase(false)]
    [TestCase(true)]
    public async Task LargeTacticalRequestIsScopedToAiEndpoint(bool chunked)
    {
        string body = JsonSerializer.Serialize(_fixture.Request(seat: 1)) + new string(' ', 30_000);
        using var request = new HttpRequestMessage(HttpMethod.Post, "/api/ai/decision")
        { Content = new StringContent(body, Encoding.UTF8, "application/json") };
        if (chunked) { request.Headers.TransferEncodingChunked = true; request.Content.Headers.ContentLength = null; }
        using HttpResponseMessage response = await _client.SendAsync(request);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK), await response.Content.ReadAsStringAsync());
        using JsonDocument result = JsonDocument.Parse(await response.Content.ReadAsStringAsync());
        Assert.That(result.RootElement.GetProperty("candidate_id").GetInt32(), Is.EqualTo(17));
        Assert.That(result.RootElement.GetProperty("difficulty_id").GetString(), Is.EqualTo("normal"));
        Assert.That(_fixture.Factory.Seats, Is.EqualTo(new[] { 1 }));
        using HttpResponseMessage ordinary = await _client.PostAsync("/ordinary-body", new StringContent(body));
        Assert.That(ordinary.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge));
    }

    [Test]
    public async Task OverLimitDeclaredBodyIsRejectedBeforeModelLoad()
    {
        using var content = new StringContent("{}");
        content.Headers.ContentLength = AiEndpoints.MaxDecisionBytes + 1;
        using HttpResponseMessage response = await _client.PostAsync("/api/ai/decision", content);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.RequestEntityTooLarge));
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [Test]
    public async Task StalePinIs409AndMalformedJsonIs400WithoutLoading()
    {
        using HttpResponseMessage stale = await _client.PostAsync("/api/ai/decision", new StringContent(
            JsonSerializer.Serialize(_fixture.Request(revision: new string('e', 64))), Encoding.UTF8, "application/json"));
        Assert.That(stale.StatusCode, Is.EqualTo(HttpStatusCode.Conflict));
        using HttpResponseMessage malformed = await _client.PostAsync("/api/ai/decision", new StringContent("{", Encoding.UTF8, "application/json"));
        Assert.That(malformed.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        string duplicate = JsonSerializer.Serialize(_fixture.Request()).Insert(1, "\"seat\":0,");
        using HttpResponseMessage repeated = await _client.PostAsync("/api/ai/decision", new StringContent(duplicate, Encoding.UTF8, "application/json"));
        Assert.That(repeated.StatusCode, Is.EqualTo(HttpStatusCode.BadRequest));
        Assert.That(_fixture.Factory.Processes, Is.Empty);
    }

    [Test]
    public async Task BrowserPreflightPermitsOnlyConfiguredOriginAndAiPost()
    {
        using var request = new HttpRequestMessage(HttpMethod.Options, "/api/ai/decision");
        request.Headers.Add("Origin", "https://game.example");
        request.Headers.Add("Access-Control-Request-Method", "POST");
        request.Headers.Add("Access-Control-Request-Headers", "content-type");
        using HttpResponseMessage response = await _client.SendAsync(request);
        Assert.That(response.Headers.GetValues("Access-Control-Allow-Origin"), Is.EqualTo(new[] { "https://game.example" }));
        Assert.That(response.Headers.GetValues("Access-Control-Allow-Methods").Single(), Does.Contain("POST"));
        Assert.That(response.Headers.Contains("Access-Control-Allow-Credentials"), Is.False);
    }
}
