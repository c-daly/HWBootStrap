using System.Diagnostics;
using HexWars.NetServer.Steam;

namespace HexWars.NetServer;

/// <summary>A disposable child server. Only its PostgreSQL rows survive a restart.</summary>
internal sealed class SelfTestProcess : IAsyncDisposable
{
    readonly Process process;
    readonly Task<string> errors;

    SelfTestProcess(Process process)
    {
        this.process = process;
        errors = process.StandardError.ReadToEndAsync();
    }

    internal int Id => process.Id;

    internal static async Task<SelfTestProcess> StartAsync(string databaseUrl, int expectedRecovered)
    {
        string executable = Environment.ProcessPath
            ?? throw new InvalidOperationException("Cannot locate the current .NET executable");
        var start = new ProcessStartInfo(executable)
        {
            UseShellExecute = false,
            RedirectStandardInput = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
        };
        if (Path.GetFileNameWithoutExtension(executable).Equals("dotnet", StringComparison.OrdinalIgnoreCase))
            start.ArgumentList.Add(typeof(Program).Assembly.Location);
        start.ArgumentList.Add("selftest-durable-host");
        start.Environment["HEXWARS_TEST_DATABASE_URL"] = databaseUrl;
        start.Environment["DOTNET_HOSTBUILDER__RELOADCONFIGONCHANGE"] = "false";

        var child = new SelfTestProcess(Process.Start(start)
            ?? throw new InvalidOperationException("The self-test server did not start"));
        try
        {
            string? ready = await child.process.StandardOutput.ReadLineAsync()
                .WaitAsync(TimeSpan.FromSeconds(30));
            string expected = $"SELFTEST-HOST READY {child.Id} {expectedRecovered}";
            if (ready != expected)
                throw new InvalidOperationException("The child did not report the expected recovery: " + ready);
            Console.WriteLine(expected);
            return child;
        }
        catch
        {
            await child.KillAsync();
            Console.Error.WriteLine(SteamLogRedaction.Redact(await child.errors));
            child.process.Dispose();
            throw;
        }
    }

    internal async Task CrashAsync()
    {
        await KillAsync();
        Console.WriteLine($"SELFTEST-HOST KILLED {Id}");
    }

    async Task KillAsync()
    {
        if (!process.HasExited) process.Kill(entireProcessTree: true);
        await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(10));
    }

    public async ValueTask DisposeAsync()
    {
        try
        {
            if (process.HasExited) return;
            await process.StandardInput.WriteLineAsync("stop");
            await process.StandardInput.FlushAsync();
            process.StandardInput.Close();
            await process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(35));
            if (process.ExitCode != 0)
                throw new InvalidOperationException("Child server exited " + process.ExitCode + ": "
                    + SteamLogRedaction.Redact(await errors));
            Console.WriteLine($"SELFTEST-HOST STOPPED {Id}");
        }
        finally
        {
            try { await KillAsync(); }
            finally { process.Dispose(); }
        }
    }
}
