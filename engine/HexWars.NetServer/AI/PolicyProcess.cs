using System.Diagnostics;
using System.Text;
using HexWars.Engine.AI;

namespace HexWars.NetServer.AI;

// A narrow transport seam: tests exercise the real queue, pinning and protocol around a controlled
// child. Production owns exactly the Process object it created, never a PID/name discovered elsewhere.
internal interface IPolicyProcess : IAsyncDisposable
{
    Task<string> ReadLineAsync(int maxCharacters, CancellationToken ct);
    Task WriteLineAsync(string line, CancellationToken ct);
}

internal interface IPolicyProcessFactory
{
    IPolicyProcess Start(AiRuntimeOptions options, AiModelDefinition model, AiPackage package);
}

internal sealed class PolicyProcessFactory(ILogger<PolicyProcessFactory> logger) : IPolicyProcessFactory
{
    public IPolicyProcess Start(AiRuntimeOptions options, AiModelDefinition model, AiPackage package)
    {
        var start = new ProcessStartInfo(options.Python)
        {
            UseShellExecute = false, CreateNoWindow = true,
            RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true,
            StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8,
            WorkingDirectory = Path.Combine(options.RuntimeRoot, "python"),
        };
        foreach (string argument in new[] { "-u", Path.Combine(options.RuntimeRoot, "python", "policy_server.py"),
                     "--p0", "run:" + package.Root, "--p1", "run:" + package.Root,
                     "--expected-environment", "tactical-v3", "--expected-contract-version", model.contract_version,
                     "--expected-encoding-hash", model.encoding_hash, "--expected-capacity-hash", model.capacity_hash })
            start.ArgumentList.Add(argument);
        start.Environment["CUDA_VISIBLE_DEVICES"] = "";
        start.Environment["OMP_NUM_THREADS"] = "1";
        start.Environment["MKL_NUM_THREADS"] = "1";
        start.Environment["OPENBLAS_NUM_THREADS"] = "1";
        start.Environment["PYTHONIOENCODING"] = "utf-8";
        var process = new Process { StartInfo = start };
        try
        {
            if (!process.Start()) throw new IOException("The policy process did not start.");
            logger.LogInformation("Started hosted AI process {ProcessId} for model {ModelId}", process.Id, model.id);
            return new PolicyProcess(process);
        }
        catch { process.Dispose(); throw; }
    }
}

internal sealed class PolicyProcess : IPolicyProcess
{
    readonly Process _process;
    readonly Task _stderr;
    readonly CancellationTokenSource _stop = new();

    internal PolicyProcess(Process process)
    {
        _process = process;
        // Drain in bounded chunks. A noisy child must never block inference or accumulate an unbounded
        // string, and arbitrary stderr (which can contain local paths) is never returned over HTTP.
        _stderr = DrainAsync();
    }

    async Task DrainAsync()
    {
        var buffer = new char[2048];
        try
        {
            while (await _process.StandardError.ReadAsync(buffer.AsMemory(), _stop.Token) != 0) { }
        }
        catch (Exception error) when (error is OperationCanceledException or IOException or ObjectDisposedException) { }
    }

    public async Task<string> ReadLineAsync(int maxCharacters, CancellationToken ct)
    {
        var line = new StringBuilder();
        var character = new char[1];
        while (line.Length <= maxCharacters)
        {
            int count = await _process.StandardOutput.ReadAsync(character.AsMemory(), ct);
            if (count == 0) throw new IOException("Policy process closed its output.");
            if (character[0] == '\n') return line.ToString().TrimEnd('\r');
            line.Append(character[0]);
        }
        throw new IOException("Policy process exceeded its response size limit.");
    }

    public async Task WriteLineAsync(string line, CancellationToken ct)
    {
        await _process.StandardInput.WriteLineAsync(line.AsMemory(), ct);
        await _process.StandardInput.FlushAsync(ct);
    }

    public async ValueTask DisposeAsync()
    {
        _stop.Cancel();
        try
        {
            if (!_process.HasExited) _process.Kill(entireProcessTree: true);
            await _process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds(5));
        }
        catch (InvalidOperationException) { }
        finally
        {
            _process.Dispose();
            await _stderr;
            _stop.Dispose();
        }
    }
}
