using System.Net;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc.Testing;
using NUnit.Framework;

namespace HexWars.NetServer.Tests
{
    /// <summary>
    /// The server doubles as the origin for the WebGL client, so the static-file pipeline is part of the
    /// contract: the client index is served at the root, a missing payload is a plain 404 rather than a
    /// fault, and the Unity payload extensions carry the content types the loader needs.
    ///
    /// The empty-web-root case is deliberately absent. The server project commits wwwroot/index.html, and
    /// the build publishes it through the static web assets manifest, which resolves from the project
    /// directory whatever content root or web root a test configures. Serving a deploy without a wwwroot
    /// therefore cannot be reproduced in process; the Docker image smoke test is where that is observed.
    /// </summary>
    [TestFixture]
    public class StaticFileTests
    {
        readonly List<string> _tempRoots = new();

        [TearDown]
        public void RemoveTempRoots()
        {
            foreach (string root in _tempRoots)
                try { Directory.Delete(root, recursive: true); } catch (IOException) { }
            _tempRoots.Clear();
        }

        /// <summary>The wwwroot of a fresh, empty content root.</summary>
        string NewWebRoot()
        {
            string content = Path.Combine(Path.GetTempPath(), "hexwars-webroot-" + Guid.NewGuid().ToString("N"));
            string webRoot = Path.Combine(content, "wwwroot");
            Directory.CreateDirectory(webRoot);
            _tempRoots.Add(content);
            return webRoot;
        }

        static WebApplicationFactory<Program> Factory(string? webRoot = null) =>
            new WebApplicationFactory<Program>().WithWebHostBuilder(builder =>
            {
                builder.UseEnvironment("Development");
                if (webRoot is not null) builder.UseContentRoot(Path.GetDirectoryName(webRoot)!);
            });

        [Test]
        public async Task Root_ServesTheWebGlClientAsHtml()
        {
            using var factory = Factory();
            using var client = factory.CreateClient();

            var response = await client.GetAsync("/");

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(response.Content.Headers.ContentType!.MediaType, Is.EqualTo("text/html"));
            Assert.That(await response.Content.ReadAsStringAsync(), Does.Contain("unity-canvas"));
        }

        [Test]
        public async Task MissingUnityPayload_Is404AndNotAnException()
        {
            using var factory = Factory(NewWebRoot());
            using var client = factory.CreateClient();

            var response = await client.GetAsync("/nonexistent.wasm");

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.NotFound));
        }

        [TestCase("build.wasm", "application/wasm")]
        [TestCase("build.data", "application/octet-stream")]
        [TestCase("build.unityweb", "application/octet-stream")]
        public async Task UnityPayloadExtensions_CarryTheContentTypeTheLoaderNeeds(string name, string expected)
        {
            string webRoot = NewWebRoot();
            await File.WriteAllBytesAsync(Path.Combine(webRoot, name), new byte[] { 1, 2, 3, 4 });
            using var factory = Factory(webRoot);
            using var client = factory.CreateClient();

            var response = await client.GetAsync("/" + name);

            Assert.That(response.StatusCode, Is.EqualTo(HttpStatusCode.OK));
            Assert.That(response.Content.Headers.ContentType!.MediaType, Is.EqualTo(expected));
        }
    }
}
