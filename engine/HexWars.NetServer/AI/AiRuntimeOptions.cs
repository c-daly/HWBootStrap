namespace HexWars.NetServer.AI;

public sealed class AiRuntimeOptions
{
    public required string RuntimeRoot { get; init; }
    public required string ConfigPath { get; init; }
    public required string Python { get; init; }
    public int MaxPendingPerModel { get; init; } = 4;
    public TimeSpan QueueTimeout { get; init; } = TimeSpan.FromSeconds(30);
    public TimeSpan StartupTimeout { get; init; } = TimeSpan.FromSeconds(90);
    public TimeSpan DecisionTimeout { get; init; } = TimeSpan.FromSeconds(30);

    public static AiRuntimeOptions Discover(IConfiguration config, IHostEnvironment environment)
    {
        string root = config["HEXWARS_AI_RUNTIME_ROOT"] ?? FindRoot(environment.ContentRootPath);
        root = Path.GetFullPath(root);
        string catalog = config["HEXWARS_AI_CONFIG_PATH"] ??
            (File.Exists(Path.Combine(root, "ai-models.json")) ? Path.Combine(root, "ai-models.json") :
                Path.Combine(root, "Assets", "StreamingAssets", "ai-models.json"));
        string localPython = OperatingSystem.IsWindows()
            ? Path.Combine(root, "python", "winenv", "Scripts", "python.exe")
            : Path.Combine(root, ".venv", "bin", "python");
        return new AiRuntimeOptions
        {
            RuntimeRoot = root,
            ConfigPath = Path.GetFullPath(catalog),
            Python = config["HEXWARS_AI_PYTHON"] ?? (File.Exists(localPython) ? localPython : "python3"),
        };
    }

    static string FindRoot(string contentRoot)
    {
        for (DirectoryInfo? directory = new(contentRoot); directory is not null; directory = directory.Parent)
        {
            if (File.Exists(Path.Combine(directory.FullName, "ai-models.json")) ||
                (File.Exists(Path.Combine(directory.FullName, "Assets", "StreamingAssets", "ai-models.json")) &&
                 File.Exists(Path.Combine(directory.FullName, "python", "policy_server.py"))))
                return directory.FullName;
        }
        return contentRoot;
    }
}
