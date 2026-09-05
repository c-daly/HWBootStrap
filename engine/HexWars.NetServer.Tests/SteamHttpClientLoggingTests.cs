using System.Net.Http;
using HexWars.NetServer.Configuration;
using HexWars.NetServer.Persistence;
using HexWars.NetServer.Steam;
using HexWars.NetServer.Tests.Fakes;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Http;
using Microsoft.Extensions.Logging;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// What the Steam client looks like once the real host has composed it, which is the only place the
    /// question can be answered: the secrets are in the request URI, and the code that would log that URI
    /// is not ours, it is the HttpClientFactory pipeline the container assembles around us.
    /// </summary>
    [TestFixture]
    public class SteamHttpClientLoggingTests
    {
        const string PublisherKey = "publisher-key-that-must-never-be-logged";
        const string Ticket = "0a1b2c3d4e5f";
        const string BaseUrl = "https://partner.steam-api.invalid";

        const string AuthOk =
            """{"response":{"params":{"result":"OK","steamid":"76561197960287930","vacbanned":false,"publisherbanned":false}}}""";

        /// <summary>
        /// A second named client, to prove the log filter is scoped to the Steam one. Named so that it
        /// starts with the Steam client name: the filter category is a prefix match, so a name the Steam
        /// one is a prefix of is exactly the client a filter without a terminating dot would silence.
        /// </summary>
        const string ProbeClientName = SteamWebApiRegistration.HttpClientName + "Metrics";

        static WebApplicationFactory<Program> Host(
            FakeSteamHandler handler,
            CapturingLoggerProvider captured,
            Action<IServiceCollection>? extraServices = null) =>
            new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                builder.UseSetting("LOBBY_PROVIDER", "Steam");
                builder.UseSetting("STEAM_APP_ID", "480000");
                builder.UseSetting("STEAM_PUBLISHER_WEB_API_KEY", PublisherKey);
                builder.UseSetting("STEAM_WEB_API_BASE_URL", BaseUrl);
                builder.UseSetting("DATABASE_URL", "postgres://hexwars:s3cr3t@db.invalid:5432/hexwars");
                builder.UseSetting("MATCH_PUBLIC_BASE_URL", "https://match.hexwars.invalid");
                builder.UseSetting("MATCH_BUILD_ID", "build-42");

                builder.ConfigureLogging(logging =>
                {
                    // Trace on purpose: a leak that only appears at a verbose level is still a leak, and
                    // an operator turning up the logs is exactly when it would have surfaced.
                    logging.SetMinimumLevel(LogLevel.Trace);
                    logging.AddProvider(captured);
                });

                builder.ConfigureServices(services =>
                {
                    // The DATABASE_URL above names a host that does not resolve; these tests are about the
                    // Steam client's logging, not storage, so drop the startup migration that would
                    // (correctly) refuse to boot. PostgresStartupTests covers that behaviour.
                    services.Remove(services.Single(d => d.ImplementationType == typeof(MigrationHostedService)));

                    // Replaces the primary handler the registration configured, leaving the rest of the
                    // factory pipeline - including its loggers - exactly as the server built it.
                    services.Configure<HttpClientFactoryOptions>(
                        SteamWebApiRegistration.HttpClientName,
                        options => options.HttpMessageHandlerBuilderActions.Add(
                            handlerBuilder => handlerBuilder.PrimaryHandler = handler));

                    extraServices?.Invoke(services);
                });
            });

        [Test]
        public async Task ResolvedSteamClient_LogsTheRequestPathButNeitherTheKeyNorTheTicket()
        {
            var handler = new FakeSteamHandler();
            handler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);
            var captured = new CapturingLoggerProvider();

            using var factory = Host(handler, captured);
            var client = factory.Services.GetRequiredService<ISteamWebApiClient>();

            var identity = await client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None);

            Assert.That(identity.SteamId, Is.EqualTo("76561197960287930"));
            Assert.That(handler.Requests, Has.Count.EqualTo(1));
            Assert.That(handler.Requests[0].Query, Does.Contain("ticket=" + Ticket),
                "the secrets must really be on the wire, or this test proves nothing");

            Assert.That(captured.Messages, Is.Not.Empty, "the host must be logging something");
            foreach (var line in captured.Messages)
            {
                Assert.That(line, Does.Not.Contain(PublisherKey));
                Assert.That(line, Does.Not.Contain(Ticket));
            }

            // The client keeps its own line: the path and the status are what an operator needs, and
            // removing the framework loggers must not leave the call invisible.
            Assert.That(
                captured.Messages.Any(m => m.Contains(FakeSteamHandler.AuthPath, StringComparison.Ordinal)),
                Is.True,
                "the client must still log the request path it called");
        }

        [Test]
        public async Task ResolvedSteamClient_LogsNoUriEvenWhenTheTransportFails()
        {
            var handler = new FakeSteamHandler();
            handler.Throw(
                FakeSteamHandler.AuthPath,
                () => new HttpRequestException(
                    "GET " + BaseUrl + "/auth?key=" + PublisherKey + "&ticket=" + Ticket + " failed"));
            var captured = new CapturingLoggerProvider();

            using var factory = Host(handler, captured);
            var client = factory.Services.GetRequiredService<ISteamWebApiClient>();

            var ex = Assert.ThrowsAsync<SteamApiException>(
                () => client.AuthenticateUserTicketAsync(Ticket, CancellationToken.None));

            Assert.That(ex!.Failure, Is.EqualTo(SteamFailure.ServiceUnavailable));
            foreach (var line in captured.Messages)
            {
                Assert.That(line, Does.Not.Contain(PublisherKey));
                Assert.That(line, Does.Not.Contain(Ticket));
            }

            await Task.CompletedTask;
        }

        [Test]
        public async Task FrameworkRequestLogging_IsSuppressedForTheSteamClientOnly()
        {
            var steamHandler = new FakeSteamHandler();
            steamHandler.RespondJson(FakeSteamHandler.AuthPath, AuthOk);

            var probeHandler = new FakeSteamHandler();
            probeHandler.RespondJson("/probe/", "{}");

            var captured = new CapturingLoggerProvider();

            using var factory = Host(steamHandler, captured, services =>
                services.AddHttpClient(ProbeClientName)
                    .ConfigurePrimaryHttpMessageHandler(() => probeHandler));

            var steam = factory.Services.GetRequiredService<ISteamWebApiClient>();
            await steam.AuthenticateUserTicketAsync(Ticket, CancellationToken.None);

            var probe = factory.Services.GetRequiredService<IHttpClientFactory>().CreateClient(ProbeClientName);
            using var response = await probe.GetAsync("https://probe.invalid/probe/");

            const string FrameworkCategory = "System.Net.Http.HttpClient.";

            // Only Steam puts secrets in a request URI. Another client losing its request diagnostics is a
            // silent hole in whatever that client is later used for.
            Assert.That(
                captured.Messages.Any(m => m.StartsWith(FrameworkCategory + ProbeClientName, StringComparison.Ordinal)),
                Is.True,
                "a client that is not the Steam one must keep its framework request logs");

            Assert.That(
                captured.Messages.Any(m => m.StartsWith(
                    FrameworkCategory + SteamWebApiRegistration.HttpClientName + ".", StringComparison.Ordinal)),
                Is.False,
                "the Steam request URI carries the publisher key and the auth ticket");
        }

        [Test]
        public void SteamHttpClient_RefusesRedirectsAndBoundsTheResponseBuffer()
        {
            HttpMessageHandler? primary = null;

            var services = new ServiceCollection();
            services.AddOptions<SteamOptions>().Configure(o =>
            {
                o.AppId = 480000u;
                o.PublisherWebApiKey = "test-key";
                o.WebApiBaseUrl = new Uri(BaseUrl);
                o.RequestTimeoutSeconds = 3;
            });
            services.AddSteamWebApi();
            services.Configure<HttpClientFactoryOptions>(
                SteamWebApiRegistration.HttpClientName,
                options => options.HttpMessageHandlerBuilderActions.Add(
                    handlerBuilder => primary = handlerBuilder.PrimaryHandler));

            using var provider = services.BuildServiceProvider();
            var http = provider.GetRequiredService<IHttpClientFactory>()
                .CreateClient(SteamWebApiRegistration.HttpClientName);

            Assert.That(primary, Is.InstanceOf<SocketsHttpHandler>());
            // A 3xx would resend the publisher key and the auth ticket to whatever host it names.
            Assert.That(((SocketsHttpHandler)primary!).AllowAutoRedirect, Is.False);
            Assert.That(http.MaxResponseContentBufferSize, Is.EqualTo(SteamWebApiClient.MaxResponseBytes));
        }
    }
}
