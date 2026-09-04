using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;

namespace HexWars.NetServer
{
    /// <summary>
    /// Process entry point. Composition lives in <see cref="ServerComposition"/> and the legacy v1 lobby in
    /// <see cref="LegacyWebSocketServer"/>; this class is deliberately NON-static so an integration test can
    /// boot the real server through WebApplicationFactory&lt;Program&gt;. Cloud-ready: binds 0.0.0.0 on $PORT
    /// when a host injects PORT, and serves the WebGL client from wwwroot when present (single origin).
    /// Subcommands: selftest drives two in-process clients through a move and asserts;
    /// describe-environment prints the resolved, validated configuration as JSON and exits.
    /// </summary>
    public class Program
    {
        public static async Task<int> Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "selftest") return await SelfTest.Run();

            if (args.Length > 0 && args[0] == "describe-environment")
            {
                // Built WITHOUT args on purpose: the command-line configuration provider rejects a bare
                // positional argument, and this subcommand needs nothing beyond the environment.
                var probe = WebApplication.CreateBuilder();
                return DescribeEnvironment(probe.Configuration, probe.Environment, Console.Out, Console.Error);
            }

            var builder = WebApplication.CreateBuilder(args);
            var port = Environment.GetEnvironmentVariable("PORT");
            if (!string.IsNullOrWhiteSpace(port)) builder.WebHost.UseUrls($"http://0.0.0.0:{port}");

            builder.AddHexWarsServer();
            var app = builder.Build();
            app.UseHexWarsServer();
            await app.RunAsync();
            return 0;
        }

        /// <summary>Validate the environment and print the report. Returns 0 when the process could serve
        /// traffic, 2 when it could not: the offending KEY names and reasons go to stderr, and the
        /// configured VALUES never do.</summary>
        internal static int DescribeEnvironment(
            IConfiguration config, IHostEnvironment env, TextWriter stdout, TextWriter stderr)
        {
            var result = HexWarsConfiguration.Read(config, env);
            if (!result.IsValid)
            {
                stderr.WriteLine("CONFIGURATION INVALID");
                foreach (string error in result.Errors) stderr.WriteLine("  " + error);
                return 2;
            }

            stdout.WriteLine(EnvironmentReport.Describe(result.Steam, result.Match, env).ToJson());
            return 0;
        }
    }
}
