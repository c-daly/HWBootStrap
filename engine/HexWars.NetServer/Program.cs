using HexWars.NetServer.Configuration;
using HexWars.NetServer.Hosting;
using HexWars.NetServer.Operations;

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

            // The durable proof: a match played on one process, continued by another over the same
            // database. It needs a throwaway Postgres and says so - exit 3, never a quiet 0 - because a
            // self-test that passed for want of anything to test is worse than one that did not run.
            if (args.Length > 0 && args[0] == "selftest-durable") return await SelfTest.RunDurable();
            if (args.Length > 0 && args[0] == "selftest-durable-crash") return await SelfTest.RunDurable(crash: true);
            if (args.Length > 0 && args[0] == "selftest-durable-host") return await SelfTest.RunDurableHost();

            // The read-only counterpart, and the only one of the two that may be pointed at real
            // data. selftest-durable proves a match survives a restart by building one; this proves
            // the matches already in a database can still be hosted, and writes nothing to do it.
            if (args.Length > 0 && args[0] == "verify-journals")
                return await JournalVerification.RunAsync(args, Console.Out);

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

            ConfigureLogging(builder);

            builder.AddHexWarsServer();
            var app = builder.Build();
            app.UseHexWarsServer();
            await app.RunAsync();
            return 0;
        }

        /// <summary>
        /// The console the platform actually reads, with scopes turned on.
        ///
        /// This is not cosmetic. Match and lobby ids are carried as logging SCOPES rather than repeated in
        /// every message, and the default console formatter does not render scopes at all - so without this
        /// the identifiers an operator searches for during an incident are computed, attached, and then
        /// dropped on the floor. JSON in Production because Render indexes structured lines and a search for
        /// one MatchId is the whole point; the readable formatter everywhere else, because a person is
        /// looking at it. The default provider is cleared first so a line is not written twice.
        /// </summary>
        internal static void ConfigureLogging(WebApplicationBuilder builder)
        {
            builder.Logging.ClearProviders();

            if (builder.Environment.IsProduction())
            {
                builder.Logging.AddJsonConsole(console =>
                {
                    console.IncludeScopes = true;

                    // One timezone in the log, and the one every other timestamp in this system is in.
                    console.UseUtcTimestamp = true;
                });

                return;
            }

            builder.Logging.AddSimpleConsole(console => console.IncludeScopes = true);
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
