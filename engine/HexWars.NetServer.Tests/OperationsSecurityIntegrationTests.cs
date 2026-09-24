using System.Net;
using HexWars.NetServer.Operations;
using HexWars.NetServer.Tests.Fakes;
using HexWars.NetServer.Tests.Fixtures;
using Microsoft.AspNetCore.Hosting;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using NUnit.Framework;

namespace HexWars.NetServer.Tests;

[TestFixture]
public class OperationsSecurityIntegrationTests
{
    const string Secret = "synthetic-healthcheck-password";

    [TestCase(false)]
    [TestCase(true)]
    public async Task ReadinessFailure_DoesNotExposeDatabaseCredentialsInFrameworkLogs(bool recoveryFails)
    {
        var logs = new CapturingLoggerProvider();
        using var fixture = new SteamServerFactory { Logging = logs };
        Exception Failure() => new InvalidOperationException("Connection failed: postgres://test:"
            + Secret + "@localhost/test");

        if (recoveryFails) fixture.Counting.BeforeListOpenMatches = () => throw Failure();

        using var host = fixture.WithWebHostBuilder(builder => builder.ConfigureServices(services =>
        {
            if (recoveryFails) return;
            services.RemoveAll<DatabaseHealthCheck>();
            services.AddSingleton(new DatabaseHealthCheck(() => throw Failure()));
        }));
        using var client = host.CreateClient();
        using var response = await client.GetAsync(HealthEndpoints.ReadyRoute);
        Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.ServiceUnavailable));
        Assert.That(await response.Content.ReadAsStringAsync(), Does.Not.Contain(Secret));
        Assert.That(logs.Containing(Secret), Is.Empty,
            "HealthCheckResult.Exception is also rendered by ASP.NET's health-check logger");
    }
}
